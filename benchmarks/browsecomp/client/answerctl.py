#!/usr/bin/env python3
"""The agents' commands: talk to the answer log.

    submit-answer [FILE]        submit a response (FILE, or standard input); the last submission counts
    answers [--full]            every submission so far, newest last
    answerctl.py episode-over   (the harness) exit 0 once this agent has submitted and its CLI has stopped

The commands are wrappers that set ANSWERCTL_COMMAND. AGENT_ID comes from the agent's runner.
"""
# Mounted into the agents' container: it must not name the benchmark (agents can read it).

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

CONFIG = Path(os.environ.get("ANSWER_LOG_CONFIG", "/opt/answer-log/task.json"))
# The CLIs whose running process means the agent's turn is still going (their process names).
CLI_NAMES = ("codex", "claude")


def config() -> dict:
    return json.loads(CONFIG.read_text())


def request(method: str, path: str, payload: dict | None = None, timeout: float = 30) -> tuple[int, dict]:
    """The first answer-log URL that answers: (status, JSON body)."""
    last_error = None
    for base in config()["answer_urls"]:
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.loads(error.read() or b"{}")
            except ValueError:
                return error.code, {"error": str(error)}
        except (urllib.error.URLError, OSError, ValueError) as error:
            last_error = error
    raise ConnectionError(f"the answer log is unreachable: {last_error}")


def agent_id() -> str:
    return os.environ.get("AGENT_ID") or "unknown"


def submit(argv: list[str]) -> int:
    if argv in (["-h"], ["--help"]) or len(argv) > 1:
        print("usage: submit-answer [FILE]   (the response from FILE, or from standard input)\n"
              "Submits your complete response in the required format (Explanation, Exact Answer,\n"
              "Confidence). The last submission is the answer that is graded.", file=sys.stderr)
        return 0 if argv in (["-h"], ["--help"]) else 2
    if argv and argv[0] != "-":
        try:
            response = Path(argv[0]).read_text()
        except (OSError, UnicodeDecodeError) as error:
            print(f"submit-answer: cannot read {argv[0]}: {error}", file=sys.stderr)
            return 2
    else:
        if sys.stdin.isatty():
            print("submit-answer: give the response on standard input (a heredoc) or as a file", file=sys.stderr)
            return 2
        response = sys.stdin.read()
    if not response.strip():
        print("submit-answer: the response is empty", file=sys.stderr)
        return 2
    try:
        status, body = request("POST", "/submit", {"agent_id": agent_id(), "response": response})
    except ConnectionError as error:
        print(f"submit-answer: {error}", file=sys.stderr)
        return 1
    if status != 200:
        print(f"submit-answer: refused: {body.get('error') or status}", file=sys.stderr)
        return 1
    print(f"Submitted as #{body['sid']} at {body['ts']}; it is now the answer that counts.")
    print(f"Exact Answer: {body.get('exact_answer')}")
    print(f"Confidence: {body.get('confidence')}")
    if not body.get("format_ok"):
        print("Warning: no 'Exact Answer:' line was found. Submit again in the required format "
              "(Explanation, Exact Answer, Confidence).")
    return 0


def answers(argv: list[str]) -> int:
    if any(arg in ("-h", "--help") for arg in argv):
        print("usage: answers [--full]   every submission so far, newest last (--full: with the responses).\n"
              "The last submission is the answer that counts.")
        return 0
    full = "--full" in argv
    try:
        status, body = request("GET", "/answers")
    except ConnectionError as error:
        print(f"answers: {error}", file=sys.stderr)
        return 1
    if status != 200:
        print(f"answers: {body.get('error') or status}", file=sys.stderr)
        return 1
    records = body.get("submissions") or []
    if not records:
        print("No submissions yet.")
        return 0
    for record in records:
        current = "  <- current answer" if record.get("sid") == body.get("current") else ""
        print(f"#{record['sid']}  {record['ts']}  {record['agent_id']}  "
              f"Exact Answer: {record.get('exact_answer')}  Confidence: {record.get('confidence')}{current}")
        if full:
            print(record.get("response", "").rstrip("\n"))
            print()
    print("The last submission is the answer that counts.")
    return 0


def cli_running(agent: str, pids=None) -> bool:
    """Whether a CLI process of this agent (its AGENT_ID in its environment) is running: among
    every process of the container, or the given pids."""
    entry = f"AGENT_ID={agent}".encode()
    procs = Path("/proc").iterdir() if pids is None else (Path("/proc") / str(pid) for pid in pids)
    for proc in procs:
        if not proc.name.isdigit():
            continue
        try:
            if (proc / "comm").read_text().strip() not in CLI_NAMES:
                continue
            if entry in (proc / "environ").read_bytes().split(b"\0"):
                return True
        except OSError:
            continue
    return False


def episode_over(argv: list[str]) -> int:
    """The harness's episode-over command (RELEASE_EPISODE_OVER_COMMAND): over once this agent has
    submitted and its turn has ended (its CLI is not running)."""
    agent = agent_id()
    try:
        status, body = request("GET", "/answers", timeout=10)
    except ConnectionError:
        return 1
    if status != 200 or not any(record.get("agent_id") == agent for record in body.get("submissions") or ()):
        return 1
    return 1 if cli_running(agent) else 0


COMMANDS = {"submit-answer": submit, "answers": answers, "episode-over": episode_over}


def main() -> int:
    command = os.environ.get("ANSWERCTL_COMMAND") or (sys.argv[1] if len(sys.argv) > 1 else "")
    argv = sys.argv[1:] if os.environ.get("ANSWERCTL_COMMAND") else sys.argv[2:]
    if command not in COMMANDS:
        print(__doc__, file=sys.stderr)
        return 2
    return COMMANDS[command](argv)


if __name__ == "__main__":
    sys.exit(main())
