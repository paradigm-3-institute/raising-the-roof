"""Usage-limit resumption for Claude Code and Codex agents (run.py's default; --no-limit-resume
turns it off).

agent.py embeds this source in each agent's runner, which runs it as
`python3 -c "$limit_resume_source" COMMAND ...`. Python 3.7+, standard library only.

How the runner resumes after a usage limit (docs/harness.md):
- Claude Code waits by itself: the runner sets CLAUDE_CODE_RETRY_WATCHDOG=1, with which the
  CLI sleeps until the reset its 429 reports and sends the same request again, so the turn just
  pauses. If the CLI still ends on an error, the runner resumes the interrupted turn with
  CLAUDE_CODE_RESUME_INTERRUPTED_TURN=1 and no prompt, so nothing is sent twice.
- Codex ends its turn on a usage limit, after recording that turn's prompt. The runner waits
  until the reset Codex recorded (`delay`), then resumes the session with RESUME_NOTE instead
  of the prompt (`input` tells whether the prompt is already in the session).

Waiting does not count against the agent's working time (--timeout), and each agent may wait
at most WAIT_MAX seconds in total (run.py --limit-wait-max, 48 h): `watch` adds up the waits,
the CLI's own (Claude Code's api_retry events) and the runner's (limit-waits.jsonl), and stops
the agent's CLI at either limit, leaving a marker for the runner.

Commands:
  watch CLI SESSION_DIR HOME_ENV HOME STARTED CAP WAIT_MAX POLL [STATE]
  plan-wait SESSION_DIR SECONDS WAIT_MAX   prints the seconds to sleep (at most the wait left)
                                           and records the wait
  exhaust SESSION_DIR WAIT_MAX             writes the wait-exhausted marker
  delay CLI ATTEMPT_LOG HOME SESSION_ID    if the attempt ended on a usage limit: prints the
                                           seconds to wait and the reset (epoch), else nothing
  input CLI HOME SESSION_ID PROMPT_FILE SINCE
                                           prints "resume" if the prompt is already in the
                                           session (recorded since SINCE), else "prompt"
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import re
import signal
import sys
import time
from datetime import datetime, timedelta, timezone

RESUME_NOTE = "Continue from where you left off."
WAITS = "limit-waits.jsonl"
STATE = "limit-wait.json"
EXHAUSTED = "limit-wait-exhausted.json"
TIME_CAP = "time-cap-reached.json"
# After a reported reset, wait a little longer, as Claude Code's own auto-resume does (30-90 s),
# so the agents of a team do not all resume at the same second.
JITTER = (30, 90)
# A usage limit whose reset Codex did not report: wait this long before resuming.
UNKNOWN_RESET_WAIT = 1800
USAGE_LIMIT = re.compile(r"usage.?limit", re.IGNORECASE)


def write_json(path: str, data: dict) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as stream:
        json.dump(data, stream)
    os.replace(tmp, path)


def read_json(path: str):
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def union_length(intervals, now: float) -> float:
    total, end = 0.0, -math.inf
    for start, stop in sorted((s, min(e, now)) for s, e in intervals if s < now):
        if stop <= end:
            continue
        total += stop - max(start, end)
        end = stop
    return total


def runner_waits(session_dir: str) -> list:
    waits = []
    try:
        with open(os.path.join(session_dir, WAITS)) as stream:
            for line in stream:
                try:
                    entry = json.loads(line)
                    waits.append((float(entry["start"]), float(entry["end"])))
                except (ValueError, KeyError, TypeError):
                    continue
    except OSError:
        pass
    return waits


def agent_processes(env_name: str, home: str) -> list:
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


def stop_cli(env_name: str, home: str) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        pids = agent_processes(env_name, home)
        for pid in pids:
            try:
                os.kill(pid, sig)
            except OSError:
                pass
        if not pids:
            return
        if sig == signal.SIGTERM:
            time.sleep(5)


class CliWaits:
    """Claude Code's own waits, from the api_retry events in the runner's attempt logs: each
    says how long the CLI sleeps before its next request. A wait lasts from when its event
    was written (the log's modification time, as the CLI writes nothing while it sleeps) until
    its delay is over, or until a later event shows the CLI working again."""

    def __init__(self, cli: str, session_dir: str):
        self.pattern = os.path.join(session_dir, f"{cli}-attempt-*.jsonl") if cli == "claude" else None
        self.offsets: dict = {}
        self.intervals: list = []

    def poll(self, now: float) -> list:
        if self.pattern is None:
            return self.intervals
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
                self.intervals = [(s, min(e, max(written, s))) for s, e in self.intervals]
                if event.get("type") == "system" and event.get("subtype") == "api_retry":
                    delay = event.get("retry_delay_ms")
                    if isinstance(delay, (int, float)) and delay > 0:
                        self.intervals.append((written, written + delay / 1000.0))
        return self.intervals

    def state(self) -> dict:
        """What a watcher started again needs to go on counting (the waits so far, merged, and
        how far each attempt log was read): read anew, a finished wait would have no length."""
        merged: list = []
        for start, end in sorted(self.intervals):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return {"offsets": self.offsets, "intervals": merged}

    def restore(self, state) -> None:
        if not isinstance(state, dict):
            return
        try:
            self.offsets = {str(path): int(offset) for path, offset in (state.get("offsets") or {}).items()}
            self.intervals = [(float(start), float(end)) for start, end in state.get("intervals") or []]
        except (TypeError, ValueError):
            self.offsets, self.intervals = {}, []


class Stopped(BaseException):
    """The runner ended and stopped the watcher (SIGTERM)."""


def stopped(signum, frame) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise Stopped


def watch(cli, session_dir, env_name, home, started, cap, wait_max, poll, state=None) -> None:
    """Enforce the working time and the most waiting allowed; it is the only thing that stops
    an agent at --timeout (Harbor's own timeout is far beyond), so it outlives any error.
    When the runner ends it stops the watcher, which then records the final times: between
    polls limit-wait.json is up to one poll old. With STATE (a file; harness_guard.py's keeper
    names one), a watcher that is killed and started again goes on from the waits counted so far."""
    started, wait_max, poll = float(started), float(wait_max), float(poll)
    cap = float(cap) if cap not in ("", None) else None
    cli_waits = CliWaits(cli, session_dir)
    saved = None
    if state:
        cli_waits.restore(read_json(state))
    try:
        signal.signal(signal.SIGTERM, stopped)
        while True:
            try:
                check(cli_waits, session_dir, env_name, home, started, cap, wait_max)
                if state and cli_waits.state() != saved:
                    saved = json.loads(json.dumps(cli_waits.state()))
                    write_json(state, saved)
            except Exception as error:  # noqa: BLE001 - keep enforcing the limits
                log_error(error)
            time.sleep(poll)
    except Stopped:
        try:
            record(cli_waits, session_dir, started, cap, wait_max)
        except Exception as error:  # noqa: BLE001
            log_error(error)


def log_error(error: Exception) -> None:
    print(f"{datetime.now(timezone.utc).isoformat()} limit watch: {error!r}", file=sys.stderr, flush=True)


def record(cli_waits, session_dir, started, cap, wait_max):
    """Write the agent's waiting and working time so far to STATE."""
    now = time.time()
    intervals = cli_waits.poll(now) + runner_waits(session_dir)
    waited = union_length(intervals, now)
    active = max(now - started - waited, 0.0)
    waiting = any(start <= now < end for start, end in intervals)
    write_json(os.path.join(session_dir, STATE), {
        "waited_seconds": round(waited, 1), "active_seconds": round(active, 1), "waiting": waiting,
        "timeout_seconds": cap, "max_wait_seconds": wait_max, "updated_at": now})
    return waited, active, waiting


def check(cli_waits, session_dir, env_name, home, started, cap, wait_max) -> None:
    """Record the agent's waiting and working time; past a limit, mark it and stop the CLI."""
    waited, active, waiting = record(cli_waits, session_dir, started, cap, wait_max)
    marker = None
    if waited >= wait_max:
        marker = EXHAUSTED
    elif cap is not None and active >= cap and not waiting:
        marker = TIME_CAP
    if marker is None:
        return
    path = os.path.join(session_dir, marker)
    if not os.path.exists(path):
        write_json(path, {"ts": datetime.now(timezone.utc).isoformat(), "waited_seconds": round(waited, 1),
                          "active_seconds": round(active, 1), "timeout_seconds": cap, "max_wait_seconds": wait_max})
    stop_cli(env_name, home)


def plan_wait(session_dir, seconds, wait_max) -> None:
    now = time.time()
    state = read_json(os.path.join(session_dir, STATE)) or {}
    waited = max(float(state.get("waited_seconds") or 0.0),
                 union_length(runner_waits(session_dir), now))
    grant = max(0, min(int(math.ceil(float(seconds))), int(math.floor(float(wait_max) - waited))))
    with open(os.path.join(session_dir, WAITS), "a") as stream:
        stream.write(json.dumps({"start": now, "end": now + grant, "planned": float(seconds)}) + "\n")
    print(grant)


def exhaust(session_dir, wait_max) -> None:
    path = os.path.join(session_dir, EXHAUSTED)
    if not os.path.exists(path):
        now = time.time()
        write_json(path, {"ts": datetime.now(timezone.utc).isoformat(), "max_wait_seconds": float(wait_max),
                          "waited_seconds": round(union_length(runner_waits(session_dir), now), 1)})


def events(path: str):
    try:
        with open(path, errors="replace") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    yield event
    except OSError:
        return


def session_file(cli: str, home: str, session_id: str):
    if not session_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", session_id):
        return None
    pattern = (os.path.join(home, "projects", "*", f"{session_id}.jsonl") if cli == "claude"
               else os.path.join(home, "sessions", "**", f"rollout-*{session_id}.jsonl"))
    found = sorted(glob.glob(pattern, recursive=True))
    return found[-1] if found else None


def codex_reset(home: str, session_id: str):
    """The latest reset Codex recorded for an exhausted rate-limit window of the session."""
    path = session_file("codex", home, session_id)
    latest = None
    for event in events(path) if path else ():
        payload = event.get("payload")
        if isinstance(payload, dict) and payload.get("type") == "token_count" and payload.get("rate_limits"):
            latest = payload["rate_limits"]
    if not isinstance(latest, dict):
        return None
    resets = []
    for window in latest.values():
        if isinstance(window, dict) and isinstance(window.get("resets_at"), (int, float)):
            used = window.get("used_percent")
            if isinstance(used, (int, float)) and used >= 100:
                resets.append(float(window["resets_at"]))
    return max(resets) if resets else None


def clock_reset(message: str, now: float):
    """"... try again at 3:05 PM." (Codex's message, in the container's local time)."""
    match = re.search(r"try again at (\d{1,2}):(\d{2})\s*([AP]M)", message, re.IGNORECASE)
    if not match:
        return None
    hour, minute = int(match.group(1)) % 12, int(match.group(2))
    hour += 12 if match.group(3).upper() == "PM" else 0
    local = datetime.fromtimestamp(now)
    target = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target.timestamp() <= now:
        target += timedelta(days=1)
    return target.timestamp()


def delay(cli, attempt_log, home, session_id) -> None:
    """Claude Code: its last rate-limit event rejected the account and names a future reset
    (the watchdog normally waits for it in the CLI). Codex: the turn failed on a usage limit;
    the reset is the one Codex recorded, else the time in its message, else unknown."""
    now = time.time()
    if cli == "claude":
        info = None
        for event in events(attempt_log):
            if event.get("type") == "rate_limit_event" and isinstance(event.get("rate_limit_info"), dict):
                info = event["rate_limit_info"]
        reset = info.get("resetsAt") if info and info.get("status") == "rejected" else None
        if not isinstance(reset, (int, float)) or reset <= now:
            return
    else:
        message = ""
        for event in events(attempt_log):
            if event.get("type") == "error":
                message = str(event.get("message") or "")
            elif event.get("type") == "turn.failed":
                error = event.get("error")
                message = str(error.get("message") if isinstance(error, dict) else error)
        if not USAGE_LIMIT.search(message):
            return
        reset = next((value for value in (codex_reset(home, session_id), clock_reset(message, now))
                      if value is not None and value > now), None)
        if reset is None:
            print(UNKNOWN_RESET_WAIT, int(now + UNKNOWN_RESET_WAIT))
            return
    print(int(math.ceil(reset - now)) + random.randint(*JITTER), int(reset))


def iso_time(value) -> float:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return -math.inf


def user_texts(cli: str, path: str, since: float):
    """The session's user prompts (not tool results or meta messages) recorded since SINCE."""
    for event in events(path):
        if iso_time(event.get("timestamp")) < since:
            continue
        if cli == "claude":
            if event.get("type") != "user" or event.get("isMeta"):
                continue
            content = (event.get("message") or {}).get("content")
            if isinstance(content, str):
                yield content
            elif isinstance(content, list):
                texts = [block.get("text", "") for block in content
                         if isinstance(block, dict) and block.get("type") == "text"]
                if texts and len(texts) == len(content):
                    yield "\n".join(texts)
        else:
            payload = event.get("payload")
            if (event.get("type") == "response_item" and isinstance(payload, dict)
                    and payload.get("type") == "message" and payload.get("role") == "user"):
                yield "\n".join(block.get("text", "") for block in payload.get("content") or []
                                if isinstance(block, dict))


def resume_input(cli, home, session_id, prompt_file, since) -> None:
    path = session_file(cli, home, session_id)
    try:
        with open(prompt_file, errors="replace") as stream:
            prompt = stream.read().strip()
    except OSError:
        prompt = None
    recorded = bool(path and prompt and any(text.strip() == prompt
                                            for text in user_texts(cli, path, float(since))))
    print("resume" if recorded else "prompt")


def main(argv) -> int:
    command, args = argv[0], argv[1:]
    if command == "watch":
        watch(*args)
    elif command == "plan-wait":
        plan_wait(*args)
    elif command == "exhaust":
        exhaust(*args)
    elif command == "delay":
        delay(*args)
    elif command == "input":
        resume_input(*args)
    else:
        print(f"unknown command {command}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
