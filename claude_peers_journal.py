#!/usr/bin/env python3
"""The messages of a claude_peers team (agent.py, multiagent_protocols/claude_peers.py; docs/setups.md), as one
journal per trial: what message-board.jsonl is for a Codex team on the message board.

Claude Code keeps no record of a team's messages beyond each agent's own logs, so the journal is read from them after
the run (agent.py writes AGENT_DIR/claude-peers.jsonl). Each agent's sessions are read from its transcripts
(claude-home/projects/**/*.jsonl) and from the runner's stream-json attempt logs (claude-attempt-*.jsonl); an item in
both counts once, so either one alone gives the whole journal:
- sends: each SendMessage call (tool_use) and its result (`success`, `message`, `msg_id`);
- receipts: each peer message Claude Code delivered, with its body: into the receiver's running turn, at its next tool
  call (in the transcript a queued_command attachment whose origin is a peer), or, for one that came while the
  receiver gave its final answer, as the prompt of one more turn before `claude -p` exits (a user message whose
  turnOrigin is "peer"); the stream-json logs have both as user messages with that origin;
- ListAgents calls and their results.
A receipt has its send's msg_id. One JSON object per line, in time order:
  {"event": "send", "ts", "from", "to" (as the sender wrote it), "to_agent" (the agent id it names), "message",
   "summary", "ok", "result" (the tool's answer), "msg_id", "received_ts", "session_id", "tool_use_id"}
  {"event": "receive", "ts" (it reached the receiver's queue), "seen_ts" (it entered the receiver's turn: at a tool
   call, or as the new turn's prompt), "agent" (the receiver), "from", "from_address", "message", "msg_id",
   "sent_ts", "delivery" ("in_turn" or "new_turn"), "session_id"}
  (With only the stream-json logs, "ts" is when the receiver's CLI logged the message, and "seen_ts" and "delivery"
  are null.)
  {"event": "list", "ts", "agent", "result", "session_id", "tool_use_id"}

    python claude_peers_journal.py AGENT_DIR [--out FILE] [--print]
        (re)write the journal of a trial's agent directory (jobs/JOB/TRIAL/agent) and print each agent's counts;
        --print also prints the messages
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

JOURNAL = "claude-peers.jsonl"
SEND_TOOL = "SendMessage"
LIST_TOOL = "ListAgents"
# agent.py's team layout: sessions-<index>-<id> (a relay's later legs, which no team has: -leg<n> after the id).
SESSION_DIR = re.compile(r"^sessions-(\d+)-(.+?)(?:-leg\d+)?$")


def events(path: Path, wanted=lambda line: True) -> Iterator[dict]:
    """The JSON objects of a log, one per line, of the lines WANTED keeps (logs of long runs are large: the journal
    parses only lines that can hold a call, its result or a peer message)."""
    try:
        stream = path.open(errors="replace")
    except OSError:
        return
    with stream:
        for line in stream:
            if not wanted(line):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                yield event


def agents(agent_dir: Path) -> list[tuple[str, Path]]:
    """Each agent's id and session directories, in agent order."""
    found = []
    for path in agent_dir.glob("sessions-*"):
        match = SESSION_DIR.match(path.name)
        if match and path.is_dir():
            found.append((int(match.group(1)), path.name, match.group(2), path))
    return [(agent_id, path) for _, _, agent_id, path in sorted(found)]


def agent_of(address: str) -> str:
    """The agent a SendMessage `to` names: a name, a name with its ref (`a2 [3fa9c1]`), or a socket address
    (`uds:/tmp/.claude-peers/a2.sock`: the `from` of a message, which Claude Code says to reply to)."""
    address = address.strip()
    if address.startswith("uds:"):
        return Path(address[len("uds:"):]).name.removesuffix(".sock")
    return re.sub(r"\s*\[[0-9a-f]+\]$", "", address)


def parts(event: dict) -> list[dict]:
    message = event.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    return [part for part in content if isinstance(part, dict)] if isinstance(content, list) else []


def result_text(part: dict) -> str:
    content = part.get("content")
    if isinstance(content, list):
        return "".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
    return content if isinstance(content, str) else ""


def peer_origin(event: dict) -> tuple[dict, str | None] | None:
    """The origin of a peer message Claude Code delivered, and how: in the receiver's running turn (a transcript's
    queued_command attachment) or as the prompt of a turn of its own (a transcript's user message whose turnOrigin is
    "peer"); None for a stream-json user message, which has the origin either way."""
    attachment = event.get("attachment")
    if event.get("type") == "attachment" and isinstance(attachment, dict) and attachment.get("type") == "queued_command":
        origin, delivery = attachment.get("origin"), "in_turn"
    elif event.get("type") == "user":
        origin = event.get("origin")
        delivery = "new_turn" if event.get("turnOrigin") == "peer" else None
    else:
        return None
    return (origin, delivery) if isinstance(origin, dict) and origin.get("kind") == "peer" else None


class Agent:
    """One agent's sends, receipts and ListAgents calls, gathered from its logs."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.calls: dict[str, dict] = {}      # tool_use id -> send or list record
        self.receipts: dict[str, dict] = {}   # msg_id -> receive record
        self.pending: set[str] = set()        # the calls whose result has not been read yet
        self.queued: dict[str, dict] = {}     # delivery_id -> receive record not yet in the receiver's turn
        self.enqueued: list[list] = []        # [ts, message] of each peer message the queue took, until matched

    def wanted(self, line: str) -> bool:
        """Whether a log line can matter: a call of either tool, a peer's message, a pending call's result, or the
        queue taking a peer's message into the turn."""
        return (SEND_TOOL in line or LIST_TOOL in line or '"peer"' in line or "<cross-session-message" in line
                or any(call in line for call in self.pending) or any(key in line for key in self.queued))

    def read(self, event: dict) -> None:
        ts = event.get("timestamp")
        session = event.get("sessionId") or event.get("session_id")
        if event.get("type") == "queue-operation":
            content = event.get("content")
            if event.get("operation") == "enqueue" and isinstance(content, str) \
                    and content.startswith("<cross-session-message"):
                self.enqueued.append([ts, content])
            elif event.get("deliveryId") in self.queued and event.get("operation") == "remove" \
                    and str(event.get("reason", "")).startswith("absorbed"):
                # The transcript's queue hands a peer's message to the running turn at its next tool call.
                self.queued.pop(event["deliveryId"])["seen_ts"] = ts
            return
        found = peer_origin(event)
        if found is not None:
            origin, delivery = found
            key = str(origin.get("msg_id") or event.get("uuid"))
            if key not in self.receipts:
                queued = self.queued_at(event) if delivery is not None else None
                self.receipts[key] = {
                    "event": "receive", "ts": queued or ts, "seen_ts": ts if delivery == "new_turn" else None,
                    "agent": self.agent_id, "from": origin.get("name") or agent_of(str(origin.get("from") or "")),
                    "from_address": origin.get("from"), "message": origin.get("body"), "msg_id": origin.get("msg_id"),
                    "sent_ts": None, "delivery": delivery, "session_id": session}
                attachment = event.get("attachment")
                if delivery == "in_turn" and isinstance(attachment, dict) and attachment.get("delivery_id"):
                    self.queued[str(attachment["delivery_id"])] = self.receipts[key]
            return
        for part in parts(event):
            if part.get("type") == "tool_use" and part.get("name") in (SEND_TOOL, LIST_TOOL):
                if str(part.get("id")) not in self.calls:
                    self.calls[str(part.get("id"))] = self.call(part, ts, session)
                    self.pending.add(str(part.get("id")))
            elif part.get("type") == "tool_result" and str(part.get("tool_use_id")) in self.pending:
                record = self.calls[str(part["tool_use_id"])]
                structured = event.get("toolUseResult", event.get("tool_use_result"))
                record.update(self.outcome(record, structured, result_text(part)))
                self.pending.discard(str(part["tool_use_id"]))

    def queued_at(self, event: dict) -> str | None:
        """When the queue took the peer message a transcript entry delivers: the first enqueue not yet matched whose
        text is the entry's (an attachment's prompt is that text; a new turn's prompt holds it)."""
        attachment = event.get("attachment")
        text = attachment.get("prompt") if isinstance(attachment, dict) else (event.get("message") or {}).get("content")
        if not isinstance(text, str):
            return None
        for index, (ts, content) in enumerate(self.enqueued):
            if content in text:
                del self.enqueued[index]
                return ts
        return None

    def call(self, part: dict, ts, session) -> dict:
        data = part.get("input") if isinstance(part.get("input"), dict) else {}
        if part.get("name") == LIST_TOOL:
            return {"event": "list", "ts": ts, "agent": self.agent_id, "result": None, "session_id": session,
                    "tool_use_id": part.get("id")}
        # (Claude 5.5 models also send SendMessage's old fields, recipient and content, which Claude Code drops.)
        to = str(data.get("to") or data.get("recipient") or "")
        message = data.get("message", data.get("content"))
        return {"event": "send", "ts": ts, "from": self.agent_id, "to": to, "to_agent": agent_of(to),
                "message": message if isinstance(message, str) or message is None else json.dumps(message),
                "summary": data.get("summary"), "ok": None, "result": None, "msg_id": None, "received_ts": None,
                "session_id": session, "tool_use_id": part.get("id")}

    @staticmethod
    def outcome(record: dict, structured: Any, text: str) -> dict:
        """A ListAgents call's text; a SendMessage call's ok, result and msg_id, from its structured result or the
        tool result's text (Claude Code's JSON; anything else, such as an error, is not ok)."""
        if record["event"] == "list":
            return {"result": text}
        if not isinstance(structured, dict):
            try:
                structured = json.loads(text)
            except ValueError:
                structured = None
        if not isinstance(structured, dict):
            return {"ok": False, "result": text, "msg_id": None}
        return {"ok": bool(structured.get("success")), "result": structured.get("message", text),
                "msg_id": structured.get("msg_id")}


def journal(agent_dir: Path) -> list[dict]:
    """The team's messages (see the module's docstring), in time order (records without a time last)."""
    records: list[dict] = []
    for agent_id, session_dir in agents(agent_dir):
        agent = Agent(agent_id)
        for path in [*sorted((session_dir / "claude-home" / "projects").glob("**/*.jsonl")),
                     *sorted(session_dir.glob("claude-attempt-*.jsonl"))]:
            for event in events(path, agent.wanted):
                agent.read(event)
        records += [*agent.calls.values(), *agent.receipts.values()]
    sent = {record["msg_id"]: record for record in records if record["event"] == "send" and record["msg_id"]}
    for receipt in (record for record in records if record["event"] == "receive"):
        send = sent.get(receipt["msg_id"])
        if send is not None:
            receipt["sent_ts"] = send["ts"]
            send["received_ts"] = send["received_ts"] or receipt["ts"]
    return sorted(records, key=lambda record: (record["ts"] is None, record["ts"] or "", record["event"]))


def summary(records: list[dict]) -> dict:
    """Per agent: messages sent, sends that failed (the receiver had no live session), sends that were accepted but
    never reached the receiver's turn (it ended first: its budget, a pause, the run's end), messages received,
    ListAgents calls."""
    found: dict[str, dict] = {}
    for record in records:
        agent = record["from"] if record["event"] == "send" else record["agent"]
        row = found.setdefault(agent, {"sent": 0, "send_failed": 0, "undelivered": 0, "received": 0, "lists": 0})
        if record["event"] == "send":
            row["sent"] += 1
            row["send_failed"] += 0 if record["ok"] else 1
            row["undelivered"] += 1 if record["ok"] and record["received_ts"] is None else 0
        else:
            row["received" if record["event"] == "receive" else "lists"] += 1
    return dict(sorted(found.items()))


def write(agent_dir: Path, out: Path | None = None) -> list[dict]:
    """Write the journal of AGENT_DIR (to OUT, default AGENT_DIR/JOURNAL) and return its records."""
    records = journal(agent_dir)
    (out or agent_dir / JOURNAL).write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the message journal of a claude_peers trial.")
    parser.add_argument("agent_dir", type=Path, help="a trial's agent directory (jobs/JOB/TRIAL/agent)")
    parser.add_argument("--out", type=Path, help=f"the journal (default AGENT_DIR/{JOURNAL})")
    parser.add_argument("--print", action="store_true", help="also print every message")
    args = parser.parse_args(argv)
    if not agents(args.agent_dir):
        raise SystemExit(f"no agent sessions (sessions-*) in {args.agent_dir}")
    records = write(args.agent_dir, args.out)
    if args.print:
        for record in records:
            if record["event"] == "send":
                print(f"{record['ts']} {record['from']} -> {record['to']} [{'ok' if record['ok'] else 'NOT SENT'}]: "
                      f"{record['message']}")
            elif record["event"] == "receive":
                print(f"{record['ts']} {record['agent']} <- {record['from']}: {record['message']}")
    print(json.dumps(summary(records), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
