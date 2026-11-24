#!/usr/bin/env python3
"""Token usage of Codex and Claude Code agents, and a per-agent token budget.

Both CLIs log every model response to disk as they go, so these totals also
cover a turn that was stopped part way, unlike their end-of-turn events:

- Codex appends its session's running totals to the session rollout
  (CODEX_HOME/sessions/**/rollout-*.jsonl) after every response, and
  `codex exec resume` keeps appending to the same file. Sub-agents (multi-agent
  v2's spawn_agent; in Ultra mode the agent spawns them on its own) are
  sessions of their own, with rollouts in the same home, and count with the
  agent's. A sub-agent forked with its parent's history opens its rollout with
  its own session_meta, then a copy of the parent's session_meta and history up
  to a thread_settings_applied event; a file's session is its first
  session_meta, and the totals the copied history carries are subtracted.
- Claude Code appends each response, with that response's usage, to the session
  transcript (CLAUDE_CONFIG_DIR/projects/*/<session>.jsonl). A response stored as
  several entries repeats its message id, so each id counts once.

Usage is normalized to input_tokens (all input, cached included),
cached_input_tokens (read from the cache), output_tokens (reasoning included),
and reasoning_output_tokens. Claude Code's cache writes count as uncached input.

A budget counts either output tokens or CAISI's weighted tokens (appendix A2 of
CAISI's DeepSeek evaluation): output tokens at 100%, unique (uncached) input
tokens at 23%, and total input tokens at 2%, so an uncached input token weighs
25% and a cached one 2%.

Run as `token_budget.py CLI HOME BUDGET METRIC MARKER USAGE_FILE`, it watches one
agent's CLI home. Once the agent's tokens reach the budget, it writes the marker
file and stops every process whose environment points at that home: the CLI and
every command it started, including ones moved to the background. It keeps doing
so until the runner stops it, so no work continues on an exhausted budget.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
METRICS = ("weighted", "output")
HOME_ENV = {"codex": "CODEX_HOME", "claude": "CLAUDE_CONFIG_DIR"}


def weighted_tokens(usage: dict[str, int]) -> float:
    unique_input = max(usage["input_tokens"] - usage["cached_input_tokens"], 0)
    return usage["output_tokens"] + 0.23 * unique_input + 0.02 * usage["input_tokens"]


def spent(usage: dict[str, int], metric: str) -> float:
    return weighted_tokens(usage) if metric == "weighted" else usage["output_tokens"]


def summed(totals: dict[str, dict[str, int]]) -> dict[str, int]:
    return {key: sum(usage[key] for usage in totals.values()) for key in USAGE_KEYS}


def parse_ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


class LogReader:
    """Reads one agent's session logs incrementally.

    totals maps each session id to its usage so far. With keep_timeline, every
    change is also recorded as (time, session, usage). Plan usage is the share of
    the subscription's weekly limit used, as first and last reported.
    """

    def __init__(self, home: str | Path, keep_timeline: bool = False) -> None:
        self.home = Path(home)
        self.keep_timeline = keep_timeline
        self.offsets: dict[Path, int] = {}
        self.totals: dict[str, dict[str, int]] = {}
        self.timeline: list[tuple[float, str, dict[str, int]]] = []
        self.plan_first_percent: float | None = None
        self.plan_used_percent: float | None = None

    def sources(self) -> list[Path]:
        raise NotImplementedError

    def read_event(self, path: Path, event: dict) -> None:
        raise NotImplementedError

    def poll(self) -> dict[str, dict[str, int]]:
        for path in self.sources():
            offset = self.offsets.get(path, 0)
            try:
                with path.open("rb") as stream:
                    stream.seek(offset)
                    data = stream.read()
            except OSError:
                continue
            # A line still being written is read on the next poll.
            end = data.rfind(b"\n") + 1
            self.offsets[path] = offset + end
            for line in data[:end].splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    self.read_event(path, event)
        return self.totals

    def set_total(self, when: str | None, session: str, usage: dict[str, int]) -> None:
        self.totals[session] = usage
        if self.keep_timeline and when:
            self.timeline.append((parse_ts(when), session, dict(usage)))

    def set_plan(self, percent: float) -> None:
        if self.plan_first_percent is None:
            self.plan_first_percent = percent
        self.plan_used_percent = percent


class CodexReader(LogReader):
    def __init__(self, home: str | Path, keep_timeline: bool = False) -> None:
        super().__init__(home, keep_timeline)
        self.session_of: dict[Path, str] = {}
        # A forked sub-agent's rollout: True while it reads the history copied from its parent,
        # and the totals that history carried (the parent's, not the sub-agent's).
        self.inherited: dict[Path, bool] = {}
        self.baseline: dict[Path, dict[str, int]] = {}

    def sources(self) -> list[Path]:
        return sorted(self.home.glob("sessions/**/rollout-*.jsonl"))

    def read_event(self, path: Path, event: dict) -> None:
        payload = event.get("payload")
        if not isinstance(payload, dict):
            return
        if event.get("type") == "session_meta":
            # The file's own session is its first session_meta; a forked sub-agent's copies its
            # parent's next.
            if path not in self.session_of:
                self.session_of[path] = str(payload.get("id") or path)
                self.inherited[path] = bool(payload.get("forked_from_id"))
            return
        if self.inherited.get(path):
            if payload.get("type") == "thread_settings_applied":
                self.inherited[path] = False
            elif payload.get("type") == "token_count":
                info = payload.get("info")
                total = info.get("total_token_usage") if isinstance(info, dict) else None
                if isinstance(total, dict):
                    self.baseline[path] = {key: int(total.get(key) or 0) for key in USAGE_KEYS}
            return
        if payload.get("type") != "token_count":
            return
        info = payload.get("info")
        total = info.get("total_token_usage") if isinstance(info, dict) else None
        if isinstance(total, dict):
            usage = {key: int(total.get(key) or 0) for key in USAGE_KEYS}
            if path in self.baseline:
                # The sub-agent's running total starts from its parent's copied one.
                usage = {key: max(usage[key] - self.baseline[path][key], 0) for key in USAGE_KEYS}
            session = self.session_of.get(path, str(path))
            previous = self.totals.get(session)
            # A session resumed into a new file repeats its running total.
            if previous is None or usage["output_tokens"] >= previous["output_tokens"]:
                self.set_total(event.get("timestamp"), session, usage)
        limits = payload.get("rate_limits")
        primary = limits.get("primary") if isinstance(limits, dict) else None
        if isinstance(primary, dict) and isinstance(primary.get("used_percent"), (int, float)):
            self.set_plan(float(primary["used_percent"]))


class ClaudeReader(LogReader):
    """Reads transcripts for usage, and the runner's attempt logs for plan usage."""

    def __init__(self, home: str | Path, keep_timeline: bool = False) -> None:
        super().__init__(home, keep_timeline)
        self.messages: dict[str, tuple[str, dict[str, int]]] = {}

    def sources(self) -> list[Path]:
        return (sorted(self.home.glob("projects/**/*.jsonl"))
                + sorted(self.home.parent.glob("claude-attempt-*.jsonl")))

    def read_event(self, path: Path, event: dict) -> None:
        if path.name.startswith("claude-attempt-"):
            info = event.get("rate_limit_info") if event.get("type") == "rate_limit_event" else None
            week = ((info or {}).get("unifiedWindows") or {}).get("seven_day") or {}
            if isinstance(week.get("utilization"), (int, float)):
                self.set_plan(100 * float(week["utilization"]))
            return
        message = event.get("message")
        if event.get("type") != "assistant" or not isinstance(message, dict):
            return
        raw = message.get("usage")
        if not isinstance(raw, dict):
            return
        cached = int(raw.get("cache_read_input_tokens") or 0)
        details = raw.get("output_tokens_details") or {}
        usage = {
            "input_tokens": (int(raw.get("input_tokens") or 0) + cached
                             + int(raw.get("cache_creation_input_tokens") or 0)),
            "cached_input_tokens": cached,
            "output_tokens": int(raw.get("output_tokens") or 0),
            "reasoning_output_tokens": int(details.get("thinking_tokens") or 0)
            if isinstance(details, dict) else 0,
        }
        key = str(message.get("id") or event.get("uuid") or len(self.messages))
        session = str(event.get("sessionId") or path.stem)
        when = event.get("timestamp")
        previous = self.messages.get(key)
        if previous is not None:
            if usage["output_tokens"] <= previous[1]["output_tokens"]:
                return
            self._add(when, *previous, sign=-1)
        self.messages[key] = (session, usage)
        self._add(when, session, usage, sign=1)

    def _add(self, when: str | None, session: str, usage: dict[str, int], sign: int) -> None:
        total = dict(self.totals.get(session) or dict.fromkeys(USAGE_KEYS, 0))
        for name in USAGE_KEYS:
            total[name] += sign * usage[name]
        self.set_total(when, session, total)


READERS = {"codex": CodexReader, "claude": ClaudeReader}


def session_totals(cli: str, home: str | Path) -> dict[str, dict[str, int]]:
    return READERS[cli](home).poll()


def agent_processes(env_name: str, home: str) -> list[int]:
    """PIDs of processes whose environment sets env_name to home."""
    entry = f"{env_name}={home}".encode()
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            environ = (proc / "environ").read_bytes()
        except OSError:
            continue
        if entry in environ.split(b"\0"):
            found.append(int(proc.name))
    return found


def stop_agent(env_name: str, home: str) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = agent_processes(env_name, home)
        for pid in pids:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        if pids and sig == signal.SIGTERM:
            time.sleep(5)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: Path, record: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)


def watch(cli: str, home: str, budget: int, metric: str, marker: Path, usage_file: Path,
          interval: float = 5.0) -> None:
    reader = READERS[cli](home)
    last = None
    while True:
        usage = summed(reader.poll())
        record = {"ts": now(), "budget": budget, "budget_metric": metric, "usage": usage,
                  "weighted_tokens": round(weighted_tokens(usage)),
                  "plan_used_percent": reader.plan_used_percent}
        if (usage, reader.plan_used_percent) != last:
            write_json(usage_file, record)
            last = (usage, reader.plan_used_percent)
        if spent(usage, metric) >= budget:
            if not marker.exists():
                write_json(marker, record)
            stop_agent(HOME_ENV[cli], home)
            interval = 2.0
        time.sleep(interval)


def main(argv: list[str]) -> None:
    cli, home, budget, metric, marker, usage_file = argv[:6]
    if cli not in READERS or metric not in METRICS:
        sys.exit(f"usage: token_budget.py {{{','.join(READERS)}}} HOME BUDGET "
                 f"{{{','.join(METRICS)}}} MARKER USAGE_FILE")
    watch(cli, home, int(budget), metric, Path(marker), Path(usage_file))


if __name__ == "__main__":
    main(sys.argv[1:])
