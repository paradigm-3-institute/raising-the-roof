"""Team limit hold (run.py --team-limit-hold; on by default for a mixed team, docs/harness.md): while
any agent of a team waits for a usage limit, the others pause too. A mixed team's agents use different
logins, so one agent's limit would otherwise leave the others working without it, for hours if it is a
weekly limit, while the agents of a team on one login all wait together.

harness_guard.py's keeper runs this as one more watcher of each agent (GUARD_DIR/mod-hold.py), next to its
token-budget and working-time watchers, as root where the keeper is. Every POLL seconds each watcher:

- holds for its own agent while the agent waits for a usage limit: the runner's own wait, the one it logs as
  usage_limit_wait (the runner writes REQUEST in the agent's session directory for it), or a wait inside
  Claude Code that it announced as at least CLAUDE_WAIT_SECONDS long (ClaudeWaits). Its hold is a file in
  HOLDS_DIR, rewritten every poll (a heartbeat), and removed when the wait is over.
- pauses its own agent while another agent holds: it writes HELD in the agent's session directory, then
  stops the agent's CLI and everything the CLI started, as the working-time cap does. The runner (agent.py)
  waits while HELD is there, in steps it records as waiting (limit_resume.py plan-wait: not working time,
  and they count toward --limit-wait-max), then resumes the interrupted turn.

A hold counts while it is fresh (heartbeat at most STALE_SECONDS old), at most MARGIN_SECONDS past its planned
end, and while its agent has not ended, so a holder that crashed or finished cannot pause the team for long.
An agent that has ended or is ending (ENDED) is never paused, nor is one that holds itself. Each hold's start
and end, and each pause's, go to RECORD (team-holds.jsonl in the trial's agent directory).

    watch INDEX AGENT_ID CLI SESSION_DIR HOME_ENV HOME HOLDS_DIR RECORD POLL

Python 3.7+, standard library only.
"""

from __future__ import annotations

import glob
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

REQUEST = "team-hold-request.json"
HELD = "team-hold.json"
RECORD = "team-holds.jsonl"
# Claude Code's persistent-retry mode announces a usage-limit wait with the time to the reset (tens of
# minutes to hours; the attempt logs show it, then every 30 s the time left). Its other retries wait seconds:
# backoffs of at most about 30 s, short 429s under a minute. A wait announced as 5 minutes or more is a limit.
CLAUDE_WAIT_SECONDS = 300.0
STALE_SECONDS = 60.0
MARGIN_SECONDS = 600.0
# What an agent's runner and watchers leave once it has ended or is ending (agent.py, harness_guard.py).
ENDED = ("token-budget-exhausted.json", "time-cap-reached.json", "limit-wait-exhausted.json", "episode-over.json",
         "harness-stop.json", "runner-exit-status")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: str, data: dict) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as stream:
        json.dump(data, stream, sort_keys=True)
    os.replace(tmp, path)


def read_json(path: str):
    try:
        with open(path) as stream:
            data = json.load(stream)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def remove(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def append_line(path: str, record: dict) -> None:
    try:
        with open(path, "a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        pass


def agent_processes(env_name: str, home: str) -> list:
    """The processes whose environment points at the agent's CLI home: its CLI and all it started
    (harness_guard.py's keeper replaces this with its own, which also works as root)."""
    entry = f"{env_name}={home}".encode()
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == os.getpid():
            continue
        try:
            with open(f"/proc/{name}/environ", "rb") as stream:
                if entry in stream.read().split(b"\0"):
                    found.append(int(name))
        except OSError:
            continue
    return found


def stop_cli(env_name: str, home: str) -> bool:
    """Stop the agent's CLI and what it started; whether there was any."""
    stopped = False
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = agent_processes(env_name, home)
        for pid in pids:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        if not pids:
            return stopped
        stopped = True
        if sig == signal.SIGTERM:
            time.sleep(5)
    return stopped


class ClaudeWaits:
    """The wait Claude Code is in, from the api_retry events in the runner's attempt logs: each is written
    when its log changed (the CLI writes nothing else while it sleeps) and announces retry_delay_ms; any
    later event ends the wait before that. Events written while a wait lasts continue it, as the 30 s
    reminders of a usage-limit wait do. A wait is [start, end, longest announced delay]."""

    def __init__(self, session_dir: str):
        self.pattern = os.path.join(session_dir, "claude-attempt-*.jsonl")
        self.offsets: dict = {}
        self.wait: list | None = None

    def poll(self, now: float):
        for path in sorted(glob.glob(self.pattern), key=lambda p: (os.path.getmtime(p), p)):
            try:
                with open(path, "rb") as stream:
                    stream.seek(self.offsets.get(path, 0))
                    data = stream.read()
                    written = min(os.fstat(stream.fileno()).st_mtime, now)
            except OSError:
                continue
            end = data.rfind(b"\n")
            if end < 0:
                continue
            self.offsets[path] = self.offsets.get(path, 0) + end + 1
            for raw in data[:end].split(b"\n"):
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                delay = event.get("retry_delay_ms") if event.get("subtype") == "api_retry" else None
                if not isinstance(delay, (int, float)) or delay <= 0:
                    if self.wait is not None:
                        self.wait[1] = min(self.wait[1], max(written, self.wait[0]))
                    continue
                if self.wait is not None and written <= self.wait[1]:
                    self.wait = [self.wait[0], written + delay / 1000.0, max(self.wait[2], delay / 1000.0)]
                else:
                    self.wait = [written, written + delay / 1000.0, delay / 1000.0]
        return self.wait


def own_wait(session_dir: str, claude, now: float):
    """(source, planned end) of the usage-limit wait the agent is in, or None."""
    request = read_json(os.path.join(session_dir, REQUEST))
    if request:
        try:
            if float(request["start"]) <= now <= float(request["until"]):
                return "usage_limit_wait", float(request["until"])
        except (KeyError, TypeError, ValueError):
            pass
    wait = claude.poll(now) if claude is not None else None
    if wait is not None and wait[0] <= now < wait[1] and wait[2] >= CLAUDE_WAIT_SECONDS:
        return "claude_retry_wait", wait[1]
    return None


def ended(session_dir: str) -> bool:
    return any(os.path.exists(os.path.join(session_dir, name)) for name in ENDED)


def live_holds(holds_dir: str, index: int, now: float) -> list:
    """The other agents' holds that count now."""
    found = []
    for path in sorted(glob.glob(os.path.join(holds_dir, "*.json"))):
        hold = read_json(path)
        try:
            if (hold is None or int(hold["index"]) == index or now - float(hold["heartbeat"]) > STALE_SECONDS
                    or now > float(hold["until"]) + MARGIN_SECONDS or ended(str(hold["session_dir"]))):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        found.append(hold)
    return found


class Watch:
    """One agent's watcher: its own hold, and its pauses for the others'."""

    def __init__(self, index, agent, cli, session_dir, env_name, home, holds_dir, record):
        self.index, self.agent, self.cli = int(index), agent, cli
        self.session_dir, self.env_name, self.home = session_dir, env_name, home
        self.holds_dir, self.record = holds_dir, record
        os.makedirs(holds_dir, exist_ok=True)
        self.mine = os.path.join(holds_dir, f"{self.index}.json")
        self.held_path = os.path.join(session_dir, HELD)
        self.claude = ClaudeWaits(session_dir) if cli == "claude" else None
        # A watcher started again goes on with the hold or pause it left.
        self.holding = read_json(self.mine)
        self.held = read_json(self.held_path)

    def note(self, event: str, **data) -> None:
        append_line(self.record, {"ts": now_iso(), "epoch": round(time.time(), 3), "event": event,
                                  "agent": self.agent, "index": self.index, **data})

    def step(self, now: float) -> None:
        done = ended(self.session_dir)
        wait = own_wait(self.session_dir, self.claude, now)
        if wait is not None and not done:
            if self.holding is None:
                self.holding = {"index": self.index, "agent": self.agent, "cli": self.cli,
                                "session_dir": self.session_dir, "since": now}
                self.note("hold_start", source=wait[0], until=round(wait[1], 3))
            self.holding.update(source=wait[0], until=wait[1], heartbeat=now)
            write_json(self.mine, self.holding)
        elif self.holding is not None:
            self.end_hold(now)
        others = [] if done or wait is not None else live_holds(self.holds_dir, self.index, now)
        if others:
            holders = sorted(str(hold["agent"]) for hold in others)
            if self.held is None or self.held.get("holders") != holders:
                if self.held is None:
                    self.held = {"since": now, "stopped_cli": False}
                    self.note("pause_start", holders=holders)
                self.held.update(holders=holders, until=max(float(hold["until"]) for hold in others))
                # Written before the CLI is stopped: the runner then finds it when its attempt ends.
                write_json(self.held_path, self.held)
            if stop_cli(self.env_name, self.home) and not self.held["stopped_cli"]:
                self.held["stopped_cli"] = True
                write_json(self.held_path, self.held)
        elif self.held is not None:
            remove(self.held_path)
            self.note("pause_end", seconds=round(now - float(self.held["since"]), 1),
                      stopped_cli=bool(self.held.get("stopped_cli")))
            self.held = None

    def end_hold(self, now: float) -> None:
        remove(self.mine)
        self.note("hold_end", seconds=round(now - float(self.holding["since"]), 1))
        self.holding = None


class Stopped(BaseException):
    """The agent ended: the keeper stopped the watcher (SIGTERM)."""


def stopped(signum, frame) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise Stopped


def log_error(error: Exception) -> None:
    print(f"{now_iso()} team hold: {error!r}", file=sys.stderr, flush=True)


def watch(index, agent, cli, session_dir, env_name, home, holds_dir, record, poll) -> None:
    watcher = Watch(index, agent, cli, session_dir, env_name, home, holds_dir, record)
    signal.signal(signal.SIGTERM, stopped)
    try:
        while True:
            try:
                watcher.step(time.time())
            except Exception as error:  # noqa: BLE001 - keep watching
                log_error(error)
            time.sleep(float(poll))
    except Stopped:
        # The agent has ended: its hold ends with it, and it is paused no more.
        if watcher.holding is not None:
            watcher.end_hold(time.time())
        if watcher.held is not None:
            remove(watcher.held_path)
            watcher.note("pause_end", seconds=round(time.time() - float(watcher.held["since"]), 1),
                         stopped_cli=bool(watcher.held.get("stopped_cli")), agent_ended=True)


def main(argv) -> int:
    command, args = argv[0], argv[1:]
    if command == "watch":
        watch(*args)
        return 0
    print(f"unknown command {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
