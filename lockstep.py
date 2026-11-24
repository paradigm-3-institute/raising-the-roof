"""Lockstep token pacing (run.py --lockstep-gap; docs/setups.md): the agents of a team spend their token
budgets together. Agents spend at different speeds (gpt-6.1-sol about 1.5 times as many tokens per hour as
Claude Sonnet 5.5), so with a budget each the fast agent ends hours early and the slow one works on alone.
Only tokens are matched, not time, so making the fast agent wait costs nothing in fairness.

The rule (assess), every POLL seconds, for each agent: its lead is its weighted tokens minus the team's
reference, by default the slowest agent still working (or their median or mean). An agent more than GAP
ahead pauses; a paused agent goes on once it is no longer ahead (lead <= 0). Agents that have ended (budget
spent, time or waits used up, finished, died) drop out of the reference; with one agent left nothing pauses.
The reference agent is never paused, so the team cannot deadlock: a lead counts as ahead only beyond
LEAD_TOLERANCE (the reference agent's own lead is 0 but float rounding can make it +2e-10), and a pause ends
whenever every working agent is paused (released "all_paused"), so two agents can never both stay paused. Each agent's share of its own budget is
compared, with GAP for an agent of the team's mean budget: with equal budgets, plain tokens and GAP. A stall
guard: when no unpaused agent's tokens have grown for STALL seconds (a long build, a long thought; time
under a team limit hold does not count), every lockstep pause ends, and pausing starts again once one of
those agents' tokens grow.

harness_guard.py's keeper runs this file (GUARD_DIR/mod-step.py) as two watchers of each agent, as root
where the keeper is:

- watch CONFIG, in the team-hold watcher's place, which it also runs when the team limit hold is on
  (team_hold.py, GUARD_DIR/mod-hold.py). A pause is PAUSE in the agent's session directory: the runner
  (agent.py) does not start the CLI while it is there, waiting in steps that count as neither working time
  nor waiting, then resumes the interrupted turn as after a team hold. The CLI is stopped as the team hold
  stops it (team_hold.stop_cli), but only at a moment when no model response streams (Stream): once its
  session log shows a finished response whose tool calls are running, else after SAFE_WAIT_SECONDS. A
  response cut off part way can be billed but is never logged. Each pause's start and end go to the
  trial's team-holds.jsonl, as team_hold.py's do, with reason "lockstep", every agent's tokens and the
  reference.
- limit ARGS: limit_resume.py's working-time watcher (GUARD_DIR/mod-limit.py), which then leaves the
  agent's lockstep pauses (PAUSES, and PAUSE while it lasts) out of its working time and its waiting, and
  records their length in limit-wait.json (lockstep_paused_seconds).

    watch CONFIG    CONFIG: JSON, agent.py's (_lockstep_watchers)
    limit ARGS      limit_resume.py's arguments (watch CLI SESSION_DIR ...)

Python 3.7+, standard library only.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import signal
import sys
import time

PAUSE = "lockstep-pause.json"
PAUSES = "lockstep-pauses.jsonl"
STALL = "stall.json"
REFERENCES = ("slowest", "median", "mean")
# Seconds between looks at the CLI's session log while a stop waits for a moment between responses; the
# most it waits for one, after which it stops the CLI anyway (a turn of one long answer without tools).
TICK_SECONDS = 0.1
SAFE_WAIT_SECONDS = 600.0
# A pause whose CLI was stopped lasts at least this long, so the runner, whose attempt the stop ended, finds
# it and resumes the turn as a paused one.
MIN_PAUSE_SECONDS = 15.0
# A paused agent goes on once its lead is no more than this many of its own tokens: the reference agent's lead
# is tokens - (tokens / budget) * budget, 0 in exact arithmetic but +2.3e-10 for about 1 token count in 85
# (2065245.34 tokens of a 4M budget is one), which would keep it paused for good.
# Float error at these sizes is below 1e-6; a real lead of one token still counts as ahead.
LEAD_TOLERANCE = 1e-3
# A pause whose heartbeat is older than this was left by a watcher that is gone (a revived trial).
STALE_SECONDS = 60.0
# Claude Code's and Codex's tool calls and their results in their session logs.
CODEX_CALLS = ("function_call", "custom_tool_call", "local_shell_call")
CODEX_OUTPUTS = ("function_call_output", "custom_tool_call_output", "local_shell_call_output")


def _sibling(name: str, copy: str):
    """team_hold.py or limit_resume.py: in the container the keeper's copy next to this file, else the
    repo's module."""
    here = os.path.abspath(__file__)
    if os.path.basename(here) == "mod-step.py":
        spec = importlib.util.spec_from_file_location(f"hsv_{name}", os.path.join(os.path.dirname(here), copy))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(name)


team_hold = _sibling("team_hold", "mod-hold.py")
limit_resume = _sibling("limit_resume", "mod-limit.py")
# The keeper replaces this with its own, which also works as root (harness_guard.py's run_watcher); main()
# hands it on to the modules above.
agent_processes = team_hold.agent_processes
read_json, write_json, remove, append_line, now_iso = (team_hold.read_json, team_hold.write_json, team_hold.remove,
                                                       team_hold.append_line, team_hold.now_iso)


# ---- the rule ---------------------------------------------------------------------------------------------

def reference(shares: list, kind: str) -> float:
    ordered = sorted(shares)
    if kind == "slowest":
        return ordered[0]
    if kind == "median":
        middle = len(ordered) // 2
        return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    return sum(ordered) / len(ordered)


def assess(team: list, me: int, kind: str, gap: float, paused: bool) -> dict:
    """The rule for agent ME of TEAM (each agent {"index", "tokens", "budget", "ended"}): whether it is to be
    paused (it is PAUSED now), its lead over the reference and its gap, in its own tokens. Agents that have
    ended do not count; with fewer than two left, nothing pauses."""
    working = [agent for agent in team if not agent["ended"]]
    mine = next(agent for agent in team if agent["index"] == me)
    mean_budget = sum(float(agent["budget"]) for agent in team) / len(team)
    budget = float(mine["budget"])
    found = {"working": len(working), "gap": gap * budget / mean_budget, "lead": 0.0, "reference": None,
             "reference_share": None, "pause": False}
    if mine["ended"] or len(working) < 2:
        return found
    share = reference([float(agent["tokens"]) / float(agent["budget"]) for agent in working], kind)
    lead = float(mine["tokens"]) - share * budget
    found.update(lead=lead, reference=share * budget, reference_share=share,
                 pause=lead > LEAD_TOLERANCE if paused else lead > found["gap"])
    return found


# ---- when the CLI is between model responses ------------------------------------------------------------------

class Stream:
    """Whether the agent's CLI is running the tool calls of a finished model response, from its newest
    session log: no response streams then. Claude Code's transcript writes a response's entries once it has
    finished (each with its stop_reason), then a tool's result when the tool has run, after which the next
    request starts; Codex's rollout writes a response's items as they arrive, then a token_usage_record once
    it has finished, then each tool's output, after which (all outputs in) the next request starts. (Codex's
    token_count, and with it the token count the budget watcher sees, comes after the tools' outputs, as
    the next request starts.) A response without tool calls ends the turn: the CLI ends by itself, and is
    not stopped then, which would add a resumption the turn did not need."""

    def __init__(self, cli: str, home: str):
        self.cli, self.home = cli, home
        self.pattern = (os.path.join(home, "projects", "*", "*.jsonl") if cli == "claude"
                        else os.path.join(home, "sessions", "**", "rollout-*.jsonl"))
        self.path, self.offset, self.listed = None, 0, -1e9
        self.safe, self.pending = False, set()

    def newest(self, now: float):
        if now - self.listed >= 2.0 or self.path is None:
            self.listed = now
            found = []
            for path in glob.glob(self.pattern, recursive=True):
                try:
                    found.append((os.path.getmtime(path), path))
                except OSError:
                    continue
            if found:
                newest = max(found)[1]
                if newest != self.path:
                    # A new session log: what it holds so far, from its last 256 KiB.
                    self.path, self.safe, self.pending = newest, False, set()
                    try:
                        self.offset = max(os.path.getsize(newest) - 262144, 0)
                    except OSError:
                        self.offset = 0
        return self.path

    def idle(self, now: float) -> bool:
        path = self.newest(now)
        if path is None:
            return False
        try:
            with open(path, "rb") as stream:
                stream.seek(self.offset)
                data = stream.read()
        except OSError:
            return False
        end = data.rfind(b"\n")
        if end >= 0:
            lines = data[:end].split(b"\n")
            if self.offset and not data.startswith(b"{"):
                lines = lines[1:]   # the end of a line read before (the tail of a new log)
            self.offset += end + 1
            for raw in lines:
                try:
                    event = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    (self.claude if self.cli == "claude" else self.codex)(event)
        return self.safe

    def claude(self, event: dict) -> None:
        kind, message = event.get("type"), event.get("message")
        if kind == "assistant" and isinstance(message, dict):
            for block in message.get("content") or ():
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    self.pending.add(block.get("id"))
            self.safe = message.get("stop_reason") == "tool_use"
        elif kind == "user" and isinstance(message, dict):
            content = message.get("content")
            results = [block.get("tool_use_id") for block in content or () if isinstance(block, dict)
                       and block.get("type") == "tool_result"] if isinstance(content, list) else []
            self.pending.difference_update(results)
            if not results or not self.pending:
                self.pending.clear()
                self.safe = False

    def codex(self, event: dict) -> None:
        kind, payload = event.get("type"), event.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        if kind == "token_usage_record":
            self.safe = bool(self.pending)
        elif kind == "response_item":
            if payload.get("type") in CODEX_OUTPUTS:
                self.pending.discard(payload.get("call_id"))
                if self.pending:
                    return
            elif payload.get("type") in CODEX_CALLS:
                self.pending.add(payload.get("call_id"))
            else:
                self.pending.clear()
            self.safe = False
        elif kind in ("compacted", "turn_context"):
            self.pending.clear()
            self.safe = False


def cli_processes(env_name: str, home: str, binary: str) -> list:
    """The agent's CLI itself (harness_guard.py's cli_processes): of its processes, those running the
    CLI's binary."""
    real, name = os.path.realpath(binary), os.path.basename(binary)[:15]
    found = []
    for pid in agent_processes(env_name, home):
        try:
            if os.path.realpath(os.readlink(f"/proc/{pid}/exe")) == real:
                found.append(pid)
        except OSError:
            try:
                with open(f"/proc/{pid}/comm") as stream:
                    if stream.read().strip() == name:
                        found.append(pid)
            except OSError:
                continue
    return found


# ---- the watcher ------------------------------------------------------------------------------------------

def spent(record, metric: str) -> float:
    """Tokens as the budget counts them, from the token-budget watcher's token-usage.json."""
    usage = (record or {}).get("usage")
    if not isinstance(usage, dict):
        return 0.0
    try:
        output, total, cached = (float(usage.get(key) or 0) for key in
                                 ("output_tokens", "input_tokens", "cached_input_tokens"))
    except (TypeError, ValueError):
        return 0.0
    return output if metric == "output" else output + 0.23 * max(total - cached, 0.0) + 0.02 * total


class Lockstep:
    """One agent's lockstep: its pauses (PAUSE), and the team's stall guard (STALL in the team directory)."""

    def __init__(self, config: dict):
        self.config = config
        self.index, self.agent = int(config["index"]), str(config["agent"])
        self.session_dir, self.env_name, self.home = config["session_dir"], config["home_env"], config["home"]
        self.binary, self.record = config["binary"], config["record"]
        self.kind, self.gap = config["ref"], float(config["gap"])
        self.stall_seconds, self.metric = float(config["stall_seconds"]), config.get("metric", "weighted")
        self.team = config["team"]
        self.team_dir = config["team_dir"]
        os.makedirs(self.team_dir, exist_ok=True)
        self.path = os.path.join(self.session_dir, PAUSE)
        self.stream = Stream(config["cli"], self.home)
        self.growth: dict = {}       # index -> [tokens, when they last grew]
        self.due = None              # since when the CLI is to be stopped, while it is
        self.last = None             # the latest assessment
        # A watcher started again goes on with the pause it left.
        self.pause = read_json(self.path)

    def note(self, event: str, **data) -> None:
        append_line(self.record, {"ts": now_iso(), "epoch": round(time.time(), 3), "event": event,
                                  "agent": self.agent, "index": self.index, "reason": "lockstep", **data})

    def read_team(self) -> list:
        keeper = read_json(self.config["keeper_state"]) if self.config.get("keeper_state") else None
        units = (keeper or {}).get("units") or {}
        found = []
        for agent in self.team:
            unit = units.get(str(agent["index"])) or {}
            found.append({"index": int(agent["index"]), "agent": agent["agent"], "budget": float(agent["budget"]),
                          "tokens": spent(read_json(os.path.join(agent["session_dir"], "token-usage.json")),
                                          self.metric),
                          # Its own markers, or the keeper gave up its runner (killed too often).
                          "ended": team_hold.ended(agent["session_dir"]) or unit.get("done") is not None})
        return found

    @staticmethod
    def tokens(team: list) -> dict:
        return {agent["agent"]: round(agent["tokens"]) for agent in team}

    def paused(self, agent: dict) -> bool:
        if agent["index"] == self.index:
            return self.pause is not None
        return os.path.exists(os.path.join(next(a["session_dir"] for a in self.team
                                                if int(a["index"]) == agent["index"]), PAUSE))

    def step(self, now: float) -> None:
        team = self.read_team()
        for agent in team:
            before = self.growth.get(agent["index"])
            if before is None or agent["tokens"] > before[0]:
                self.growth[agent["index"]] = [agent["tokens"], now]
        if self.config.get("holds_dir") and team_hold.live_holds(self.config["holds_dir"], -1, now):
            # A teammate waits for a usage limit and the team holds: no one is stalled.
            for agent in team:
                self.growth[agent["index"]] = [agent["tokens"], now]
        if self.pause is not None and now - float(self.pause.get("heartbeat") or 0) > STALE_SECONDS:
            self.end(float(self.pause.get("heartbeat") or now), team, "stale")
        self.last = found = assess(team, self.index, self.kind, self.gap, self.pause is not None)
        stalled = self.stalled(team, now)
        mine = next(agent for agent in team if agent["index"] == self.index)
        if self.pause is None:
            if found["pause"] and not stalled:
                self.begin(now, team, found)
            return
        # Every working agent paused (this one included) is a deadlock whatever the leads say: no one's tokens
        # can move again, and the stall guard only looks at unpaused agents. Each ends its pause; the one still
        # GAP ahead pauses again at its next step, the slowest goes on.
        everyone = all(self.paused(agent) for agent in team if not agent["ended"])
        release = ("agent_ended" if mine["ended"] else "alone" if found["working"] < 2 else "stall" if stalled
                   else "all_paused" if everyone else None if found["pause"] else "caught_up")
        stopped = self.pause.get("cli_stopped_at")
        if release is not None and (release == "agent_ended" or not stopped
                                    or now - float(stopped) >= MIN_PAUSE_SECONDS):
            self.end(now, team, release)
            return
        self.pause.update(heartbeat=now, lead=round(found["lead"]),
                          lead_max=max(round(found["lead"]), int(self.pause.get("lead_max") or 0)))
        # The CLI may still run (it was mid-response), or the runner started it just as the pause began.
        if cli_processes(self.env_name, self.home, self.binary):
            self.due = self.due if self.due is not None else now
        else:
            self.due = None
            if self.pause.get("stopped") is None:
                self.pause.update(stopped=now, stop="not running")
        write_json(self.path, self.pause)

    def begin(self, now: float, team: list, found: dict) -> None:
        self.pause = {"since": now, "heartbeat": now, "stopped": None, "stopped_cli": False,
                      "lead": round(found["lead"]), "lead_max": round(found["lead"]), "gap": round(found["gap"]),
                      "reference": round(found["reference"]), "ref": self.kind}
        write_json(self.path, self.pause)
        self.note("pause_start", tokens=self.tokens(team), ref=self.kind, reference=round(found["reference"]),
                  reference_share=round(found["reference_share"], 6), lead=round(found["lead"]),
                  gap=round(found["gap"]), working=found["working"])
        self.due = now if cli_processes(self.env_name, self.home, self.binary) else None
        if self.due is None:
            self.pause.update(stopped=now, stop="not running")
            write_json(self.path, self.pause)

    def tick(self, now: float) -> None:
        """While the CLI is to be stopped: stop it once no response streams, or after SAFE_WAIT_SECONDS."""
        if self.pause is None or self.due is None:
            return
        idle = self.stream.idle(now)
        if not idle and now - self.due < SAFE_WAIT_SECONDS:
            return
        waited = now - self.due
        self.due = None
        # Recorded first: the agent can end while its CLI is stopped (its last response spent its budget), and
        # the keeper then ends this watcher in the middle of the stop.
        before = self.pause.get("stopped_cli"), self.pause.get("cli_stopped_at")
        self.pause.update(stopped=self.pause.get("stopped") or now, stop="between responses" if idle else "timed out",
                          stop_wait=round(waited, 1), stopped_cli=True, cli_stopped_at=now,
                          lead_at_stop=round(self.last["lead"]) if self.last else None)
        write_json(self.path, self.pause)
        if not team_hold.stop_cli(self.env_name, self.home):
            # It had ended by itself just before.
            self.pause.update(stopped_cli=bool(before[0]), cli_stopped_at=before[1])
            write_json(self.path, self.pause)

    def end(self, now: float, team: list, release: str) -> None:
        pause, self.pause, self.due = self.pause, None, None
        if pause.get("stopped") is not None and now > float(pause["stopped"]):
            # The time the agent did not work, which counts as neither working time nor waiting.
            append_line(os.path.join(self.session_dir, PAUSES), {"start": float(pause["stopped"]), "end": now})
        remove(self.path)
        found = self.last or {}
        self.note("pause_end", seconds=round(now - float(pause["since"]), 1), released=release,
                  stopped_cli=bool(pause.get("stopped_cli")), stop=pause.get("stop"), stop_wait=pause.get("stop_wait"),
                  stopped_at=None if pause.get("stopped") is None else round(float(pause["stopped"]), 3),
                  lead_at_stop=pause.get("lead_at_stop"), lead_max=pause.get("lead_max"), tokens=self.tokens(team),
                  reference=None if found.get("reference") is None else round(found["reference"]),
                  lead=round(found.get("lead") or 0), **({"agent_ended": True} if release == "agent_ended" else {}))

    def stalled(self, team: list, now: float) -> bool:
        """The stall guard: whether lockstep pauses are off because the agents it waits for are stuck.
        One that sees them stuck while it is paused says so for the team (STALL); once one of those agents'
        tokens grow (or they have all ended), pausing starts again."""
        path = os.path.join(self.team_dir, STALL)
        stall = read_json(path)
        tokens = {agent["index"]: agent["tokens"] for agent in team}
        working = {agent["index"] for agent in team if not agent["ended"]}
        if stall is not None:
            frozen = {int(index): float(value) for index, value in (stall.get("tokens") or {}).items()}
            if any(tokens.get(index, 0.0) > value for index, value in frozen.items()) or not set(frozen) & working:
                try:
                    os.unlink(path)
                except OSError:
                    return False
                self.note("lockstep_rearm", stalled_seconds=round(now - float(stall.get("since") or now), 1),
                          tokens=self.tokens(team))
                for agent in team:
                    self.growth[agent["index"]] = [agent["tokens"], now]
                return False
            return True
        if self.pause is None or self.stall_seconds <= 0:
            return False
        unpaused = [agent for agent in team if agent["index"] in working and not self.paused(agent)]
        if not unpaused or any(now - self.growth[agent["index"]][1] < self.stall_seconds for agent in unpaused):
            return False
        write_json(path, {"since": now, "by": self.agent, "tokens": {str(a["index"]): a["tokens"] for a in unpaused}})
        self.note("lockstep_stall", stalled=[agent["agent"] for agent in unpaused], tokens=self.tokens(team),
                  seconds=round(min(now - self.growth[agent["index"]][1] for agent in unpaused), 1))
        return True


def paused_intervals(session_dir: str, now: float) -> list:
    """The agent's lockstep pauses: those that ended (PAUSES) and the one that lasts (PAUSE), from when its
    CLI stopped."""
    found = []
    try:
        with open(os.path.join(session_dir, PAUSES)) as stream:
            for line in stream:
                try:
                    entry = json.loads(line)
                    found.append((float(entry["start"]), float(entry["end"])))
                except (ValueError, KeyError, TypeError):
                    continue
    except OSError:
        pass
    pause = read_json(os.path.join(session_dir, PAUSE))
    if pause and pause.get("stopped") is not None:
        # Under way (its end is past NOW, which every length is cut at).
        found.append((float(pause["stopped"]), now + STALE_SECONDS))
    return found


# ---- the commands ---------------------------------------------------------------------------------------------

class Stopped(BaseException):
    """The agent ended: the keeper stopped the watcher (SIGTERM)."""


def stopped(signum, frame) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    raise Stopped


def log_error(error: Exception) -> None:
    print(f"{now_iso()} lockstep: {error!r}", file=sys.stderr, flush=True)


def watch(config: dict) -> None:
    lockstep = Lockstep(config)
    hold = team_hold.Watch(config["index"], config["agent"], config["cli"], config["session_dir"],
                           config["home_env"], config["home"], config["holds_dir"], config["record"]) \
        if config.get("hold") else None
    poll = float(config["poll"])
    signal.signal(signal.SIGTERM, stopped)
    try:
        due = 0.0
        while True:
            now = time.time()
            if now >= due:
                due = now + poll
                for part in (hold, lockstep):
                    try:
                        part and part.step(now)
                    except Exception as error:  # noqa: BLE001 - keep watching
                        log_error(error)
            try:
                lockstep.tick(time.time())
            except Exception as error:  # noqa: BLE001
                log_error(error)
            time.sleep(TICK_SECONDS if lockstep.due is not None else max(min(due - time.time(), poll), 0.05))
    except Stopped:
        # The agent has ended: its hold and its pauses end with it.
        now = time.time()
        if hold is not None and hold.holding is not None:
            hold.end_hold(now)
        if hold is not None and hold.held is not None:
            remove(hold.held_path)
            hold.note("pause_end", seconds=round(now - float(hold.held["since"]), 1),
                      stopped_cli=bool(hold.held.get("stopped_cli")), agent_ended=True)
        if lockstep.pause is not None:
            try:
                team = lockstep.read_team()
            except Exception:  # noqa: BLE001
                team = []
            lockstep.end(now, team, "agent_ended")


def limit_record(cli_waits, session_dir, started, cap, wait_max):
    """limit_resume.py's record, with the agent's lockstep pauses left out of its working time (which the time
    cap counts) and out of its waiting (which --limit-wait-max counts): a pause that overlaps a wait counts
    as the wait."""
    now = time.time()
    intervals = cli_waits.poll(now) + limit_resume.runner_waits(session_dir)
    waited = limit_resume.union_length(intervals, now)
    both = intervals + paused_intervals(session_dir, now)
    away = limit_resume.union_length(both, now)
    active = max(now - started - away, 0.0)
    waiting = any(start <= now < end for start, end in both)
    limit_resume.write_json(os.path.join(session_dir, limit_resume.STATE), {
        "waited_seconds": round(waited, 1), "active_seconds": round(active, 1), "waiting": waiting,
        "lockstep_paused_seconds": round(away - waited, 1), "timeout_seconds": cap,
        "max_wait_seconds": wait_max, "updated_at": now})
    return waited, active, waiting


def limit(argv: list) -> int:
    """limit_resume.py's working-time watcher, with limit_record."""
    limit_resume.record = limit_record
    return limit_resume.main(argv)


def main(argv: list) -> int:
    # The keeper's way of finding the agent's processes, for the modules that stop them.
    team_hold.agent_processes = limit_resume.agent_processes = agent_processes
    command, args = argv[0], argv[1:]
    if command == "watch":
        watch(json.loads(args[0]))
        return 0
    if command == "limit":
        return limit(args)
    print(f"unknown command {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
