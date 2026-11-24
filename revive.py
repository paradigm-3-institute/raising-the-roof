#!/usr/bin/env python3
"""Revive a trial of this harness whose host process died (a hang or restart of the host, a reboot)
while its containers and logs survived (docs/harness.md, reviving a trial).

`harbor jobs resume` restarts such a trial from scratch. This instead starts the trial's own
stopped containers again, lets each agent continue its own CLI session where it stopped, with one
note as the only new user message, within the run's budgets counting what it had already used,
and then finishes the trial the way Harbor does: artifacts, verifier, result.json, teardown.

Two more cases (for instance when the host runs out of process slots):
- Harbor ended the trial on an infrastructure failure (an exception from the host or Docker, such
  as "can't start new thread", not an agent or verifier outcome) and it has no reward: its
  result.json is kept as result.before-revive-N.json and the trial is revived. Containers that
  still run are stopped first, so the trial restarts as after a host crash.
- The trial's containers are gone (Harbor's teardown removed them) but its logs, sessions and
  snapshot store survive (FrontierSWE's machinery: git-to-zig): fresh containers are started from the same images and compose files, the agent
  is set up as Harbor sets it up, and the graded workspace and each agent's working copy are
  restored from the newest snapshot, which must hold the last promotion; the restore is checked
  against the snapshot's manifest. The note says what was restored and from when.
Every container of a revived trial gets the run's --pids-limit (pids_limit.py), and a task's
sandbox timer, if its image has one, counts working time only.

    revive.py plan  TRIAL_DIR [--raise-timeout SECONDS --raise-reason TEXT] -- RUN_ARGS...
        what a revival would do; changes nothing
    revive.py start TRIAL_DIR [--note TEXT] [--raise-timeout SECONDS --raise-reason TEXT] -- RUN_ARGS...
        start the supervisor detached (it survives this shell; log TRIAL_DIR/revive-N.log)
    revive.py run   TRIAL_DIR [--note TEXT] [--raise-timeout ...] -- RUN_ARGS...   the supervisor, in the foreground

RUN_ARGS are the run's own run.py arguments (its --name is the job directory's name). The revival
renders each agent's runner from them with this code, and refuses to start unless every file the
run wrote from its task instruction and prompts is byte-identical to what this code writes.

--raise-timeout SECONDS raises the working-time cap (run.py --timeout) for the revived part, on
purpose and recorded (revive.json's working_time_cap_raised, with --raise-reason). RUN_ARGS keep
the run's own --timeout (or give SECONDS as --timeout). Only raising is allowed, and a later
revival must raise it again to at least the same cap. The new cap reaches the runner's time cap
(the limit watcher), Harbor's agent timeout (what is left of the new cap, plus the waiting the
run allows) and the agents' `resources` command, which is written again. Not for tasks whose
task directory carries --timeout (TIMEOUT_IN_TASK: their task text or sandbox timer keeps it).

What the supervisor does, in order:
1. Starts the trial's containers in their dependency order (the egress sidecar first), each
   once the services it depends on are healthy, with `docker start`: they keep their file
   systems. Compose files are only regenerated (Harbor's own, in /tmp, did not survive) so that
   Harbor's exec and teardown work. Bind-mount sources a reboot removed from /tmp are recreated.
2. Gives the agents' logs to the host user as Harbor does at the end of a run, and reads them:
   each agent's session, the turn it was in, its tokens and when it last worked. An agent that
   had ended (budget spent, gave up, ...) is not resumed.
3. Renders the revived runner against a recording stand-in for the container, which runs the
   byte-identity checks; nothing reaches the container if they fail (the containers are stopped
   again and the trial is left as it was).
4. Writes TRIAL_DIR/revive.json, which marks the trial as revived (also result.json's
   agent_result.metadata["revived"]).
5. Runs Harbor's trial lifecycle from the agent phase: the agent phase's network policy (checked:
   a host outside the allowlist must be unreachable), the revived runner (Harbor's agent timeout:
   what was left of it), artifacts, the verifier, result.json, teardown; then run.py's steps
   after Harbor and the job's result.json.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import difflib
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import tomllib
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import harness_guard  # noqa: E402
import limit_resume  # noqa: E402
import pids_limit  # noqa: E402
import token_budget  # noqa: E402

DEFAULT_NOTE = ("The machine restarted. Your files are intact, but any background processes you started are no "
                "longer running. Continue from where you left off.")
# The note for a trial Harbor ended on an infrastructure failure whose containers survived.
FAULT_NOTE = ("Your session was interrupted by a machine fault. Your files are intact, but any background processes "
              "you started are no longer running. Continue from where you left off.")
# The note for a trial rebuilt from its snapshot store ({app}: the graded workspace).
REBUILT_NOTE = ("Your session was interrupted by a machine fault and your container was rebuilt. {app} and your "
                "working copy were restored from the snapshot of {app} taken after the last submission (#{seq}, "
                "{time} UTC); changes after it, build outputs and any other files outside /logs/agent are lost, and "
                "background processes are no longer running. Continue from where you left off.")
REBUILT_NOTE_NO_SUBMISSION = (
    "Your session was interrupted by a machine fault and your container was rebuilt. {app} and your working copy "
    "were restored from the snapshot of {app} taken at {time} UTC, before any submission; changes after it, build "
    "outputs and any other files outside /logs/agent are lost, and background processes are no longer running. "
    "Continue from where you left off.")
# What the runner logs (agent.py's trial_revived event) per kind of revival.
EVENT_MESSAGES = {
    "finished": "Harbor ended this trial on an infrastructure failure; its containers were restarted and {cli} "
                "continues the same session with one note.",
    "rebuilt": "This trial lost its containers to an infrastructure failure; they were rebuilt, its files restored "
               "from the last snapshot, and {cli} continues the same session with one note.",
}
# Harbor's exceptions that are infrastructure failures, not agent or verifier outcomes: the
# exception type and a pattern its message must match. The host ran out of processes or threads
# ("can't start new thread", EAGAIN from fork), or Docker could not run a command for Harbor
# (the CLI could not start its compose plugin: "unknown flag: --project-name").
INFRA_FAILURES = (
    ("RuntimeError", r"^can't start new thread$", "the host could not start a thread"),
    ("RuntimeError", r"^Docker compose command failed\b.*(resource temporarily unavailable|cannot fork|"
                     r"unknown flag: --project-name|Cannot connect to the Docker daemon)",
     "Docker could not run Harbor's compose command"),
    ("RuntimeError", r"^Failed to create transfer archive\b.*resource temporarily unavailable",
     "Docker could not run Harbor's artifact transfer"),
    ("BlockingIOError", r"Resource temporarily unavailable", "the host could not start a process"),
)
TASK_CLOCK = "/sandbox-timer"  # FrontierSWE images: sandbox-timer's start anchor (start, started_at)
OWNERSHIP_DUMP = r"""
import json, os, sys
found = []
for root in sys.argv[2:]:
    if not os.path.lexists(root):
        continue
    names = [root]
    for current, dirs, files in os.walk(root):
        names += [os.path.join(current, name) for name in dirs + files]
    for name in names:
        try:
            info = os.lstat(name)
        except OSError:
            continue
        found.append([name, info.st_uid, info.st_gid])
with open(sys.argv[1], "w") as stream:
    json.dump(found, stream)
print(len(found))
"""
OWNERSHIP_RESTORE = r"""
import json, os, sys
changed = 0
for name, uid, gid in json.load(open(sys.argv[1])):
    try:
        info = os.lstat(name)
        if (info.st_uid, info.st_gid) != (uid, gid):
            os.lchown(name, uid, gid)
            changed += 1
    except OSError:
        pass
os.unlink(sys.argv[1])
print(changed)
"""
# Run as the task's agent user in a rebuilt container: the snapshot (SOURCE, mirroring the graded
# root) into the graded root, then each agent's working copy from it, as `workcopy` makes one,
# all under the promotion lock (benchmarks/frontier-swe/fswe.py).
RESTORE = r"""
import json, shutil, sys
tools, source, agents = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, tools)
import fswe
state = fswe.State(fswe.load_config())
done = {}
with fswe.Lock(state.lock):
    done["app"] = fswe.sync(source, state.app, state.graded)
    done["work"] = {}
    for agent in agents:
        work = state.work(agent)
        if work.exists():
            raise SystemExit(f"{work} exists already")
        staging = work.with_name(work.name + ".partial")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        done["work"][agent] = fswe.sync(state.app, staging, state.graded)
        staging.rename(work)
print(json.dumps(done, sort_keys=True))
"""
RECORD = "revive.json"
# A claude_peers team's session registry in the agent directory (agent.py CLAUDE_PEERS_REGISTRY), which Claude Code
# keeps private to the agents' user.
CLAUDE_PEERS_REGISTRY = "claude-peers-sessions"
HOME_DIRS = {"claude": "claude-home", "codex": "codex-home"}
ATTEMPT = re.compile(r"^(?:claude|codex)-attempt-(\d+)(?:-.*)?\.jsonl$")
SESSION_DIR = re.compile(r"^sessions-(\d+)-(.+)$")
# A relay's later legs (run.py --relay; agent.py relay_label): sessions-0-a1-leg2, agent-0-a1-leg2-output.log, ...
RELAY_LEG = re.compile(r"^(.+)-leg(\d+)$")
# A relay's leg that ended on one of these handed over to the next leg (a marker or event of finished()); any
# other end, or one of RELAY_FINAL_MARKERS (agent.py's), ended the relay.
RELAY_NEXT = ("token-budget-exhausted.json", "token_budget_exhausted", "token_budget_spent", "agent_gave_up",
              "optimization_continuations_completed", "content_filter_stop")
# The runner logs one of these when an agent ends on its own (the harness_ ones: harness protection
# gave it up, docs/harness.md); the watchers and the keeper leave these markers.
FINAL_EVENTS = ("token_budget_exhausted", "agent_gave_up", "optimization_continuations_completed",
                "time_cap_reached", "limit_wait_exhausted", "episode_over",
                "claude_api_retries_exhausted", "codex_api_retries_exhausted", "content_filter_stop",
                "harness_protection_failed", "harness_runner_restart_limit", "harness_cli_kill_limit")
FINAL_MARKERS = ("token-budget-exhausted.json", limit_resume.TIME_CAP, limit_resume.EXHAUSTED, "episode-over.json",
                 harness_guard.STOP_MARKER)
RELAY_FINAL_MARKERS = (limit_resume.TIME_CAP, limit_resume.EXHAUSTED, "episode-over.json", harness_guard.STOP_MARKER)
# Files a task's services rewrite when they start again; the run's versions are kept beside them.
PRESERVE = ("ale-bench/hardware.json",)
# Hosts no agent phase allows: if one answers, the egress allowlist is not in force.
CANARY_HOSTS = ("example.com", "www.wikipedia.org")
NETWORK_PROBE = r"""
import json, sys, urllib.error, urllib.request
found = {}
for host in sys.argv[1:]:
    try:
        with urllib.request.urlopen(f"https://{host}/", timeout=20) as reply:
            found[host] = f"answered {reply.status}"
    except urllib.error.HTTPError as error:
        found[host] = f"answered {error.code}"
    except Exception as error:  # noqa: BLE001
        found[host] = f"unreachable: {type(error).__name__}"
print(json.dumps(found))
"""
CODE_FILES = ("revive.py", "agent.py", "run.py", "token_budget.py", "limit_resume.py", "pids_limit.py",
              "frontier_swe.py", "harness_guard.py")
# Tasks whose task directory run.py builds from --timeout (the task's own time budget, its text or
# sandbox timer): a revival cannot raise their working-time cap, the task would keep the old one.
TIMEOUT_IN_TASK = ("frontier-swe",)


def iso(epoch: float | None) -> str | None:
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def events(path: Path):
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


def write_json(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_records(trial: Path) -> list[dict]:
    path = trial / RECORD
    return json.loads(path.read_text())["revivals"] if path.is_file() else []


def save_records(trial: Path, records: list[dict]) -> None:
    write_json(trial / RECORD, {
        "about": "This trial's host process died and the trial was revived (revive.py; docs/harness.md): one "
                 "entry per revival.",
        "revivals": records})


def infrastructure_failure(result: dict) -> str | None:
    """Why Harbor ended a trial, if that was an infrastructure failure the trial can be revived
    from: an exception of INFRA_FAILURES and no reward. None otherwise (a finished trial, a
    reward, an agent or verifier outcome such as a timeout, an unknown exception)."""
    info = result.get("exception_info") or {}
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    if not info or any(value is not None for value in rewards.values()):
        return None
    kind, message = info.get("exception_type") or "", info.get("exception_message") or ""
    for expected, pattern, label in INFRA_FAILURES:
        if kind == expected and re.search(pattern, message, re.IGNORECASE | re.DOTALL):
            first = message.splitlines()[0] if message else ""
            return f"{label} ({kind}: {first[:300]})"
    return None


def apply_pids_limit(config, limit: int | None) -> bool:
    """Give a trial's environment config the run's --pids-limit (pids_limit.py) if it has none:
    the revived trial's environments (a rebuilt one, the separate verifier) are capped as a new
    run's would be. Returns whether it changed the config (in memory; config.json stays the run's)."""
    environment = config.environment
    if limit is None or pids_limit.configured(environment) is not None:
        return False
    kind = getattr(environment.type, "value", environment.type)
    if environment.import_path is not None or kind not in (None, "docker"):
        raise ValueError(f"the trial's environment ({environment.import_path or kind}) is not Harbor's docker "
                         "environment; give the run's run.py arguments --pids-limit off to revive it uncapped")
    config.environment = environment.model_copy(update={
        "type": None, "import_path": pids_limit.IMPORT_PATH,
        "kwargs": {**(environment.kwargs or {}), pids_limit.KWARG: int(limit)}})
    return True


def newest_snapshot(trial: Path, layout) -> dict:
    """The snapshot a trial whose containers are gone is rebuilt from: the newest one with a stored
    manifest. It must hold the last promotion, or work the agents had submitted would be lost."""
    import frontier_swe

    state = trial / layout.state
    stored = [r for r in frontier_swe.snapshot_records(trial, layout)
              if r.get("digest") and (state / "manifests" / f"{r['digest']}.json").is_file()]
    if not stored:
        raise RuntimeError(f"the trial's containers are gone and {state} holds no snapshot to rebuild it from")
    snapshot = max(stored, key=lambda r: (r.get("ts") or "", int(r.get("seq") or 0)))
    promoted = [r for r in frontier_swe.read_jsonl(trial / "agent" / layout.log_name / "submissions.jsonl")
                if r.get("status") == "promoted"]
    last = max((int(r.get("seq") or 0) for r in promoted), default=None)
    if last is not None and (snapshot.get("submission_seq") is None or int(snapshot["submission_seq"]) < last):
        raise RuntimeError(f"the newest snapshot (seq {snapshot.get('seq')}, {snapshot.get('ts')}) holds submission "
                           f"#{snapshot.get('submission_seq')}, but #{last} was promoted later: rebuilding would lose it")
    return snapshot


# ---- the run's settings ----------------------------------------------------------------------------

def harness(run_args: list[str], job_name: str,
            timeout: int | None = None) -> tuple[argparse.Namespace, list[str], dict[str, str]]:
    """The run's run.py arguments, Harbor command and harness environment, as run.py builds them now
    (a dry run under another name: run.py writes nothing then, and the job already exists). With
    TIMEOUT (a raised working-time cap, --raise-timeout) the command and environment are built with
    it as --timeout; the arguments returned keep their own."""
    import run

    args = run.parser().parse_args(run_args)
    if args.name != job_name:
        raise ValueError(f"the run.py arguments name job {args.name!r}, not this trial's job {job_name!r}")
    if any(cli not in HOME_DIRS for cli in run.agent_clis(args)):
        raise ValueError("revive.py resumes Claude Code and Codex agents only")
    probe = copy.copy(args)
    probe.name = f"{job_name}-revive-probe"
    probe.dry_run = True
    if timeout is not None:
        probe.timeout = timeout
    cmd, env, _ = run.prepare(probe)
    return args, cmd, env


def run_timeout_for(args: argparse.Namespace, harbor_timeout: float) -> int | None:
    """The --timeout that, with the other arguments, gives Harbor's agent timeout HARBOR_TIMEOUT
    (run.harbor_agent_timeout is linear in --timeout), or None if no whole number of seconds does."""
    import run

    probe = copy.copy(args)
    probe.timeout = 0
    base = run.harbor_agent_timeout(probe)
    probe.timeout = 1
    slope = run.harbor_agent_timeout(probe) - base
    probe.timeout = round((harbor_timeout - base) / slope) if slope > 0 else 0
    if probe.timeout < 1 or abs(run.harbor_agent_timeout(probe) - harbor_timeout) >= 1:
        return None
    return probe.timeout


def check_settings(config, args: argparse.Namespace, cmd: list[str], task_timeout: float,
                   raise_to: int | None = None) -> int:
    """The arguments must be the trial's: its agent, model and --timeout. Returns the run's working-time
    cap (its --timeout). With RAISE_TO (--raise-timeout) the arguments' --timeout may be RAISE_TO
    instead: the run's cap is then the --timeout that gives the trial's agent timeout with the other
    arguments (raised_cap decides whether RAISE_TO is allowed)."""
    import run

    agent = cmd[cmd.index("--agent-import-path") + 1]
    model = cmd[cmd.index("-m") + 1]
    if config.agent.import_path != agent:
        raise ValueError(f"the trial ran {config.agent.import_path}, the arguments give {agent}")
    if config.agent.model_name != model:
        raise ValueError(f"the trial ran model {config.agent.model_name}, the arguments give {model}")
    harbor_timeout = (config.agent_timeout_multiplier or 1.0) * task_timeout
    if any(abs(harbor_timeout - value) < 1 for value in (args.timeout, run.harbor_agent_timeout(args))):
        return args.timeout
    if raise_to is not None and args.timeout == raise_to:
        found = run_timeout_for(args, harbor_timeout)
        if found is not None:
            return found
    raise ValueError(f"the trial's agent timeout was {harbor_timeout:.0f} s, which these arguments' "
                     f"--timeout {args.timeout} does not give"
                     + ("" if raise_to is not None else " (to raise the working-time cap on purpose, keep the "
                        "run's --timeout and give --raise-timeout SECONDS with --raise-reason TEXT)"))


def raised_cap(records: list[dict], run_cap: int, requested: int | None, task: str | None = None) -> int | None:
    """The working-time cap of the revived part when it is not the run's (RUN_CAP): REQUESTED
    (--raise-timeout), which must be above RUN_CAP and at least any cap an earlier revival raised it
    to (an earlier revival that failed before its agents ran does not count). None: the run's cap.
    Only raising is allowed, and a raised cap is never dropped silently: a later revival without
    --raise-timeout is refused."""
    earlier = [(r.get("revival"), int(r["working_time_cap_raised"]["to_seconds"])) for r in records
               if r.get("status") != "failed" and (r.get("working_time_cap_raised") or {}).get("to_seconds")]
    floor = max(earlier, key=lambda found: found[1], default=None)
    if requested is None:
        if floor is not None:
            raise ValueError(f"revival {floor[0]} raised this trial's working-time cap from {run_cap} s to {floor[1]} s; "
                             f"give --raise-timeout {floor[1]} (and --raise-reason) to keep it, a revival never "
                             "lowers it")
        return None
    if task in TIMEOUT_IN_TASK:
        raise ValueError(f"{task} builds its task from --timeout (its own time budget), so a revival cannot raise "
                         "its working-time cap")
    if requested <= run_cap:
        raise ValueError(f"--raise-timeout {requested} does not raise the run's working-time cap (--timeout "
                         f"{run_cap}); only raising is allowed")
    if floor is not None and requested < floor[1]:
        raise ValueError(f"--raise-timeout {requested} is below the {floor[1]} s revival {floor[0]} raised the "
                         "working-time cap to; only raising is allowed")
    return requested


def apply_environment(env: dict[str, str]) -> None:
    """This process's environment becomes the one run.py gives Harbor (agent.py reads it)."""
    for key in list(os.environ):
        if key not in env:
            del os.environ[key]
    os.environ.update(env)


# ---- reading the trial ------------------------------------------------------------------------------

def agent_cli(cli: str | list[str], index: int) -> str:
    """An agent's CLI: the run's, or a mixed team's agent's own (run.py --team: a list in agent order)."""
    return cli[index] if isinstance(cli, list) else cli


def sessions(trial: Path, cli: str | list[str]) -> tuple[str, list[dict]]:
    """The layout (team: run(); single: run_single()) and each agent's session directory and log."""
    agent_dir = trial / "agent"
    team = []
    for path in agent_dir.glob("sessions-*"):
        match = SESSION_DIR.match(path.name)
        if match and path.is_dir():
            # A relay's later leg is the same agent (its id) in a session of its own.
            leg = RELAY_LEG.match(match.group(2))
            team.append({"index": int(match.group(1)), "agent_id": leg.group(1) if leg else match.group(2),
                         "leg": int(leg.group(2)) if leg else 1, "dir": path,
                         "log": agent_dir / f"agent-{match.group(1)}-{match.group(2)}-output.log"})
    single = agent_dir / f"{agent_cli(cli, 0)}-single-session"
    if team and single.is_dir():
        raise ValueError(f"{agent_dir} has both team and single-agent sessions")
    if team:
        return "team", sorted(team, key=lambda s: (s["leg"], s["index"]))
    if single.is_dir():
        return "single", [{"index": 0, "agent_id": "single", "dir": single, "log": agent_dir / "agent-single-output.log"}]
    raise ValueError(f"no {cli} agent sessions in {agent_dir}")


def attempt_logs(session_dir: Path) -> list[Path]:
    """The runner's attempt logs, oldest first."""
    found = [path for path in session_dir.iterdir() if ATTEMPT.match(path.name)]
    return sorted(found, key=lambda path: (path.stat().st_mtime, path.name))


def turn_of(logs: list[Path]) -> int | None:
    """The turn the agent was in: the highest attempt number (0 is the task prompt's turn)."""
    turns = [int(ATTEMPT.match(path.name).group(1)) for path in logs]
    return max(turns) if turns else None


def session_id_of(cli: str, logs: list[Path]) -> str | None:
    """The session the runner would have resumed: the last one its attempt logs name."""
    found = None
    for path in logs:
        for event in events(path):
            if cli == "claude":
                value = event.get("session_id")
            else:
                value = event.get("thread_id") if event.get("type") == "thread.started" else None
            if isinstance(value, str) and value:
                found = value
    return found


def transcripts(cli: str, home: Path) -> list[Path]:
    return sorted(home.glob("projects/**/*.jsonl") if cli == "claude" else home.glob("sessions/**/rollout-*.jsonl"))


def home_session_id(cli: str, home: Path) -> str | None:
    """The newest session in the CLI's own session logs. The attempt logs may not name it: the
    runner's sed buffers the CLI's events, and a host that dies loses what was in the buffer."""
    newest = None
    for path in transcripts(cli, home):
        if cli == "claude":
            session = path.stem
        else:
            session = next((str(e["payload"].get("id")) for e in events(path)
                            if e.get("type") == "session_meta" and isinstance(e.get("payload"), dict)), None)
        if session and (newest is None or path.stat().st_mtime > newest[0]):
            newest = (path.stat().st_mtime, session)
    return newest[1] if newest else None


def last_activity(cli: str, home: Path) -> float | None:
    """When the agent last worked: the newest timestamp in its CLI's session logs."""
    latest = None
    for path in transcripts(cli, home):
        for event in events(path):
            stamp = limit_resume.iso_time(event.get("timestamp"))
            if stamp != float("-inf") and (latest is None or stamp > latest):
                latest = stamp
    return latest


def finished(session_dir: Path, log: Path) -> str | None:
    """Why the agent had already ended on its own, or None."""
    for marker in FINAL_MARKERS:
        if (session_dir / marker).exists():
            return marker
    for event in events(log):
        data = event.get("data") if event.get("type") == "environment.feedback" else None
        if isinstance(data, dict) and data.get("event") in FINAL_EVENTS:
            return data["event"]
    return None


def unlogged_attempt(log: Path, logs: list[Path]) -> Path | None:
    """The runner adds each attempt's log to the agent's log when the attempt ends; the attempt the
    host's death interrupted was not added. Returns it, or None."""
    if not logs:
        return None
    lines = [line for line in logs[-1].read_bytes().splitlines() if line.strip()]
    if not lines:
        return None
    return None if log.is_file() and lines[-1] in log.read_bytes() else logs[-1]


def agent_started(trial: Path, layout: str, cli: str | list[str]) -> float:
    """When Harbor started the agents: the first file the agent's run() wrote."""
    cli = agent_cli(cli, 0)
    first = (trial / "agent/multiagent-collab/protocol.json" if layout == "team"
             else trial / f"agent/{cli}-single-session/{cli}-prompt.txt")
    return first.stat().st_mtime


def inspect_agents(trial: Path, run_cli: str | list[str], budget: int | None, metric: str,
                   relay: int | None = None) -> tuple[str, list[dict]]:
    """Each agent's state (readable once the agents' logs belong to the host user). RELAY: the run's legs (run.py
    --relay), whose last may not have started its agent yet."""
    layout, found = sessions(trial, run_cli)
    if relay is None and any(session.get("leg", 1) > 1 for session in found):
        raise ValueError(f"{trial} is a relay (sessions of later legs, run.py --relay): give the run's --relay")
    last_leg = max((session.get("leg", 1) for session in found), default=1)
    agents = []
    for session in found:
        cli = agent_cli(run_cli, session["index"])
        logs = attempt_logs(session["dir"])
        home = session["dir"] / HOME_DIRS[cli]
        usage = token_budget.summed(token_budget.session_totals(cli, home)) if home.is_dir() else dict.fromkeys(
            token_budget.USAGE_KEYS, 0)
        spent = token_budget.spent(usage, metric)
        session_id = session_id_of(cli, logs) or (home_session_id(cli, home) if home.is_dir() else None)
        reason = finished(session["dir"], session["log"])
        if reason is None and budget and spent >= budget:
            reason = "token_budget_spent"
        agent = {"index": session["index"], "agent_id": session["agent_id"], "session_dir": str(session["dir"]),
                 "session_id": session_id, "turn": turn_of(logs), "finished_before": reason,
                 "usage_before": usage, "weighted_tokens_before": round(token_budget.weighted_tokens(usage)),
                 "last_activity": last_activity(cli, home),
                 "transcript": limit_resume.session_file(cli, str(home), session_id) if session_id else None,
                 "unlogged_attempt": unlogged_attempt(session["log"], logs), "log": session["log"]}
        if relay is not None:
            wait = read_json_or_none(session["dir"] / limit_resume.STATE) or {}
            agent.update(leg=session["leg"], waited_seconds=float(wait.get("waited_seconds") or 0.0),
                         relay_over=any((session["dir"] / name).exists() for name in RELAY_FINAL_MARKERS))
            # The last leg's agent may not have started: a leg that has spent nothing starts afresh.
            agent["fresh"] = (reason is None and session["leg"] == last_leg and spent == 0
                              and (session_id is None or agent["transcript"] is None or agent["turn"] is None))
            if agent["fresh"]:
                agents.append(agent)
                continue
        if reason is None and (session_id is None or agent["transcript"] is None or agent["turn"] is None):
            raise ValueError(f"agent {session['agent_id']} has no session to resume in {session['dir']}")
        agents.append(agent)
    return layout, agents


def relay_next(agents: list[dict], legs: int) -> tuple[list[dict], dict]:
    """What a relay's revival does (inspect_agents with relay): resume the agent of the last leg if it had not
    ended, start that leg afresh if its agent had not started, or start the next leg if the last ended on its own
    (RELAY_NEXT) and legs are left; else nothing. Returns the agents to resume and the plan's relay settings: the
    leg the revived runner starts with (relay_leg) and the earlier legs' waiting for usage limits (relay_waited)."""
    leg = max(agent["leg"] for agent in agents)
    current = [agent for agent in agents if agent["leg"] == leg]
    waited = sum(agent["waited_seconds"] for agent in agents if agent["leg"] < leg)
    if all(agent["finished_before"] is None for agent in current):
        if any(agent["fresh"] for agent in current):
            return [], {"relay_leg": leg, "relay_waited": round(waited, 3)}
        return current, {"relay_leg": leg, "relay_waited": round(waited, 3)}
    if leg < legs and all(agent["finished_before"] in RELAY_NEXT and not agent["relay_over"] for agent in current):
        waited += sum(agent["waited_seconds"] for agent in current)
        return [], {"relay_leg": leg + 1, "relay_waited": round(waited, 3)}
    return [], {}


def worked_before(records: list[dict], started: float, last: float) -> float:
    """Working time used before this revival: from the agents' start to their last activity, plus,
    for each earlier revival, from its start to the last activity after it. The time the host was
    down does not count."""
    if not records:
        return max(last - started, 0.0)
    previous = records[-1]
    return previous["worked_seconds_before"] + max(last - previous["revived_at_epoch"], 0.0)


# ---- Docker -------------------------------------------------------------------------------------------

def docker(*args: str, timeout: float = 120) -> str:
    done = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if done.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args)} failed: {done.stderr.strip()[-2000:]}")
    return done.stdout


def project_containers(project: str) -> list[dict]:
    ids = docker("ps", "-aq", "--filter", f"label=com.docker.compose.project={project}").split()
    return json.loads(docker("inspect", *ids)) if ids else []


def depends_on(container: dict) -> dict[str, str]:
    """Compose's depends_on label: "service:condition:restart,..." -> {service: condition}."""
    label = (container["Config"].get("Labels") or {}).get("com.docker.compose.depends_on", "")
    found = {}
    for part in filter(None, label.split(",")):
        service, _, rest = part.partition(":")
        found[service] = rest.split(":")[0] or "service_started"
    return found


def service_of(container: dict) -> str:
    return (container["Config"].get("Labels") or {})["com.docker.compose.service"]


def start_order(containers: list[dict]) -> list[dict]:
    """The containers in an order that starts each after the services it depends on."""
    by_service = {service_of(c): c for c in containers}
    ordered, done = [], set()
    while len(ordered) < len(by_service):
        ready = [s for s, c in sorted(by_service.items())
                 if s not in done and all(d in done or d not in by_service for d in depends_on(c))]
        if not ready:
            raise ValueError(f"circular depends_on among {sorted(set(by_service) - done)}")
        for service in ready:
            ordered.append(by_service[service])
            done.add(service)
    return ordered


def missing_bind_sources(containers: list[dict]) -> list[str]:
    """Bind-mount sources that no longer exist. Only /tmp is expected to lose them (a reboot)."""
    missing = set()
    for container in containers:
        for mount in container.get("Mounts") or []:
            if mount.get("Type") == "bind" and not Path(mount["Source"]).exists():
                missing.add(mount["Source"])
    return sorted(missing)


async def wait_ready(name: str, timeout: float = 900) -> str:
    """Wait until a started container runs, and is healthy if it has a healthcheck."""
    deadline = time.monotonic() + timeout
    while True:
        state = json.loads(docker("inspect", "--format", "{{json .State}}", name))
        health = (state.get("Health") or {}).get("Status")
        if state.get("Status") not in ("running", "restarting"):
            raise RuntimeError(f"{name} stopped while starting: {state.get('Status')} (exit {state.get('ExitCode')})")
        if state.get("Status") == "running" and health in (None, "healthy"):
            return health or "running"
        if health == "unhealthy" or time.monotonic() > deadline:
            raise RuntimeError(f"{name} did not become healthy ({health})")
        await asyncio.sleep(2)


def stale_judge_containers(task_dir: Path) -> list[str]:
    """ALE-Bench: case containers the trial's dead judge had created through the Docker socket
    (labelled with the task), never started or left stopped; the restarted judge makes its own."""
    config = task_dir / "scorer" / "config.json"
    if not (task_dir / "ale-bench.json").is_file() or not config.is_file():
        return []
    task_id = json.loads(config.read_text()).get("task_id")
    if not task_id:
        return []
    rows = docker("ps", "-a", "--filter", f"label=ttc.ale-bench.task={task_id}", "--format", "{{.ID}} {{.State}}")
    return [line.split()[0] for line in rows.splitlines() if line.split()[1:] and line.split()[1] in (
        "created", "exited", "dead")]


# ---- the dry run ---------------------------------------------------------------------------------------

class RecordingEnvironment:
    """Stands in for the container while the revived runner is rendered: records each command."""

    default_user = None

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        self.calls.append(command)
        return SimpleNamespace(return_code=0, stdout="", stderr="")


def rendered_runner(layout: str, calls: list[str]) -> str:
    """The revived runner a dry run rendered: the team runner's file, or the single runner's command."""
    if layout == "single":
        return calls[0]
    words = shlex.split(calls[0])
    return words[words.index("printf") + 2]


# ---- the revived trial ---------------------------------------------------------------------------------

def revived_trial_class():
    from harbor.agents.installed.base import NonZeroAgentExitCodeError
    from harbor.models.agent.context import AgentContext
    from harbor.models.job.lock import TrialLock
    from harbor.models.trial.result import TimingInfo
    from harbor.trial.errors import AgentTimeoutError
    from harbor.trial.single_step import SingleStepTrial

    class RevivedTrial(SingleStepTrial):
        """Harbor's single-step trial from its agent phase on, in containers that already exist."""

        def __init__(self, config, *, revival: "Revival", **kwargs):
            self.revival = revival
            super().__init__(config, **kwargs)

        @classmethod
        async def load(cls, config, revival: "Revival") -> "RevivedTrial":
            task, download = await cls._load_task(config)
            if task.has_steps:
                raise ValueError("multi-step tasks cannot be revived")
            return cls(config, revival=revival, _task=task, _task_download_result=download)

        def _write_trial_lock(self):
            # The run's own lock stays as it is.
            if self.paths.lock_path.is_file():
                return TrialLock.model_validate_json(self.paths.lock_path.read_text())
            return super()._write_trial_lock()

        def _init_result(self) -> None:
            config = self.paths.config_path.read_bytes() if self.paths.config_path.is_file() else None
            super()._init_result()
            if config is not None:
                self.paths.config_path.write_bytes(config)  # the run's own record stays
            self.result.started_at = datetime.fromtimestamp(self.revival.trial_started, timezone.utc)

        async def _prepare(self) -> None:
            # The containers run already (Revival.prepare); the agent was set up by the run.
            self.result.agent_info = self.agent.to_agent_info()

        async def _run_agent(self) -> None:
            try:
                if (self.revival.plan["agents"] or self.revival.plan.get("relay_leg")) and \
                        self.revival.timeout_left > 0:
                    await self.revival.check_network(self)
                    self.revival.plan["started"] = time.time() - self.revival.worked
                    self.agent.revive_plan = self.revival.plan
                    await self._run_agent_phase(target=self.result, instruction=self.task.instruction,
                                                timeout_sec=self.revival.timeout_left,
                                                user=self.task.config.agent.user, resume=True)
            except (AgentTimeoutError, NonZeroAgentExitCodeError) as exc:
                self._record_exception(exc)
            finally:
                if self.result.agent_result is None:
                    self.result.agent_result = AgentContext()
                await self._sync_agent_output(self.result)
                execution = self.result.agent_execution or TimingInfo(finished_at=self._now())
                execution.started_at = datetime.fromtimestamp(self.revival.agents_started, timezone.utc)
                self.result.agent_execution = execution
                self.result.agent_result.metadata = {**(self.result.agent_result.metadata or {}),
                                                     "revived": self.revival.summary()}

    return RevivedTrial


class Revival:
    """One revival of one trial."""

    def __init__(self, trial: Path, run_args: list[str], note: str | None = None,
                 raise_timeout: int | None = None, raise_reason: str | None = None) -> None:
        from harbor.models.trial.config import TrialConfig

        self.trial = Path(trial).resolve()
        self.job = self.trial.parent
        # A trial Harbor finished is revived only if it ended on an infrastructure failure
        # without a reward (infrastructure_failure); its result.json is kept beside.
        self.harbor_ended = None
        if (self.trial / "result.json").exists():
            self.harbor_ended = infrastructure_failure(json.loads((self.trial / "result.json").read_text()))
            if self.harbor_ended is None:
                raise ValueError(f"{self.trial} has a result.json: Harbor finished it (a trial is revived after "
                                 "Harbor ended it only for an infrastructure failure and without a reward)")
        self.config = TrialConfig.model_validate_json((self.trial / "config.json").read_text())
        self.run_args = list(run_args)
        self.note_given = note
        self.note = note or DEFAULT_NOTE
        if raise_timeout is not None and not (raise_reason or "").strip():
            raise ValueError("--raise-timeout needs --raise-reason: why the working-time cap is raised (revive.json "
                             "records it)")
        self.args, self.cmd, self.env = harness(self.run_args, self.job.name, raise_timeout)
        task_toml = tomllib.loads((Path(self.config.task.path) / "task.toml").read_text())
        self.harbor_timeout_run = (self.config.agent_timeout_multiplier or 1.0) * float(task_toml["agent"]["timeout_sec"])
        # The run's working-time cap (--timeout), and the revived part's if it is raised on purpose
        # (--raise-timeout, raised_cap): then self.args.timeout is the new cap, and the runner, Harbor's
        # agent timeout and the `resources` command use it (harness built the environment with it).
        self.run_cap = check_settings(self.config, self.args, self.cmd, float(task_toml["agent"]["timeout_sec"]),
                                      raise_timeout)
        self.records = load_records(self.trial)
        self.cap_raised = raised_cap(self.records, self.run_cap, raise_timeout, self.args.task)
        self.raise_reason = raise_reason.strip() if self.cap_raised is not None else None
        if self.cap_raised is not None:
            self.args.timeout = self.cap_raised
        # A mixed team's: each agent's, in agent order.
        team = getattr(self.args, "team", None)
        self.cli = [member["cli"] for member in team] if team else self.args.cli
        # Every container of the revived trial gets the run's process cap (a new run's default
        # when the run predates --pids-limit).
        self.pids_limit = getattr(self.args, "pids_limit", None)
        self.pids_added = apply_pids_limit(self.config, self.pids_limit)
        self.number = len(self.records) + 1
        self.trial_started = next((r["trial_started_at_epoch"] for r in self.records if "trial_started_at_epoch" in r),
                                  None) or self.trial.joinpath("config.json").stat().st_mtime
        self.record: dict = {}
        self.plan: dict = {}
        self.worked = 0.0
        self.timeout_left = 0.0
        self.agents_started = 0.0
        self.mode = "restarted"  # or "rebuilt" (prepare)
        self.snapshot: dict | None = None
        self.layout = None

    def kind(self) -> str:
        """classic (the host died), finished (Harbor ended it on an infrastructure failure) or
        rebuilt (its containers were gone)."""
        return "rebuilt" if self.mode == "rebuilt" else ("finished" if self.harbor_ended else "classic")

    # -- before Harbor -------------------------------------------------------------------------------

    async def restart(self, environment, trial=None) -> list[dict]:
        """Start the trial's containers again (see the module's step 1), or rebuild them when they
        are gone (rebuild)."""
        from harbor.environments.definition import should_use_prebuilt_docker_image
        from harbor.environments.docker.docker import _sanitize_docker_compose_project_name

        project = _sanitize_docker_compose_project_name(environment.session_id)
        containers = project_containers(project)
        if not containers and trial is not None:
            return await self.rebuild(environment, trial)
        environment._mounts_compose_path = environment._write_mounts_compose_file()
        environment._resources_compose_path = environment._write_resources_compose_file()
        environment._env_compose_path = environment._write_env_compose_file()
        environment._write_egress_control_services_compose_file()
        environment._use_prebuilt = should_use_prebuilt_docker_image(
            environment.environment_dir, docker_image=environment.task_env_config.docker_image, force_build=False)
        if environment._enable_egress_control:
            await environment._ensure_egress_control_sidecar_image_built()
        if "main" not in {service_of(c) for c in containers}:
            raise RuntimeError(f"the trial's main container is gone ({len(containers)} containers left)")
        running = [c for c in reversed(start_order(containers)) if c["State"].get("Running")]
        if self.harbor_ended and running:
            # Harbor gave the trial up while its containers ran on: stop them (main first), so it
            # restarts as after a host crash, without the processes the agents had left.
            for container in running:
                docker("stop", "--time", "30", container["Name"].lstrip("/"), timeout=180)
            self.record["stopped_running_containers"] = [c["Name"].lstrip("/") for c in running]
            containers = project_containers(project)
        for name in PRESERVE:
            path = self.trial / name
            if path.is_file():
                kept = path.with_name(f"{path.stem}.before-revive-{self.number}{path.suffix}")
                shutil.copy2(path, kept)
                self.record.setdefault("preserved", []).append(str(kept.relative_to(self.trial)))
        stale = stale_judge_containers(Path(self.config.task.path))
        if stale:
            docker("rm", *stale)
        self.record["removed_stale_containers"] = len(stale)
        created = []
        for source in missing_bind_sources(containers):
            if not source.startswith("/tmp/"):
                raise RuntimeError(f"bind-mount source {source} is missing (only /tmp is expected to lose them)")
            Path(source).mkdir(parents=True, exist_ok=True)
            Path(source).chmod(0o777)
            created.append(source)
        self.record["created_bind_sources"] = created
        started = []
        for container in start_order(containers):
            name = container["Name"].lstrip("/")
            if not container["State"].get("Running"):
                docker("start", name)
            started.append({"service": service_of(container), "name": name, "state": await wait_ready(name)})
        self.record["containers"] = started
        self.cap_processes(project)
        return started

    def cap_processes(self, project: str) -> None:
        """Each container of the trial gets the run's pids limit if it has none or a higher one
        (`docker update`, no restart; a lower limit of the task's stays)."""
        if self.pids_limit is None:
            return
        found = {}
        for container in project_containers(project):
            name = container["Name"].lstrip("/")
            limit = (container.get("HostConfig") or {}).get("PidsLimit")
            if not limit or limit < 0 or limit > self.pids_limit:
                docker("update", "--pids-limit", str(self.pids_limit), name)
                found[name] = {"before": limit, "after": self.pids_limit}
            else:
                found[name] = {"before": limit, "after": limit}
        self.record["pids_limit"] = {"limit": self.pids_limit, "added_to_config": self.pids_added,
                                     "containers": found}

    async def rebuild(self, environment, trial) -> list[dict]:
        """The trial's containers are gone: start fresh ones from its images and compose files, as
        Harbor starts a trial's environment (with the process cap), if the trial has a snapshot
        store to restore its files from (restore_snapshot)."""
        import frontier_swe
        from harbor.environments.docker.docker import _sanitize_docker_compose_project_name

        self.layout = frontier_swe.trial_layout(self.trial)
        if self.layout is None:
            raise RuntimeError("the trial's containers are gone, and it has no snapshot store to restore its files "
                               "from (only frontier-swe trials have one)")
        self.snapshot = newest_snapshot(self.trial, self.layout)
        for artifact in frontier_swe.artifact_configs(Path(self.config.task.path)):
            if getattr(artifact, "destination", None):
                raise RuntimeError(f"artifact {artifact.source} has a destination; the restore maps sources only")
        self.mode = "rebuilt"
        shared = self.trial / "agent" / self.layout.log_name
        kept = []
        for name in ("ready", "finished"):  # rewritten by the agent's setup / would stop the snapshot grader
            path = shared / name
            if path.is_file():
                path.replace(path.with_name(f"{name}.before-revive-{self.number}"))
                kept.append(str(path.relative_to(self.trial)))
        self.record["rebuild"] = {"snapshot_seq": self.snapshot.get("seq"), "snapshot_at": self.snapshot.get("ts"),
                                  "snapshot_digest": self.snapshot["digest"],
                                  "snapshot_files": self.snapshot.get("files"),
                                  "submission_seq": self.snapshot.get("submission_seq"), "kept_aside": kept}
        save_records(self.trial, [*self.records, self.record])
        await asyncio.wait_for(environment.start(force_build=False), timeout=trial._environment_build_timeout_sec)
        project = _sanitize_docker_compose_project_name(environment.session_id)
        started = [{"service": service_of(c), "name": c["Name"].lstrip("/"), "state": c["State"].get("Status"),
                    "image": c["Config"].get("Image")} for c in start_order(project_containers(project))]
        self.record["containers"] = started
        self.cap_processes(project)
        return started

    async def setup_agent(self, trial) -> None:
        """A rebuilt container: Harbor's agent setup, as at the start of the run (the harness tools,
        the CLI's links)."""
        env, agent = trial.agent_environment, trial.agent
        with env.with_default_user(trial.task.config.agent.user):
            with env.scoped_exec_env(agent.extra_env):
                await asyncio.wait_for(agent.setup(environment=env), timeout=trial._agent_setup_timeout_sec)

    async def restore_snapshot(self, trial, agent_ids: list[str]) -> dict:
        """A rebuilt container: the graded workspace from the snapshot, and each agent's working
        copy from it; then /app is captured the way the snapshot grader captures it and must equal
        the snapshot."""
        import tempfile

        import frontier_swe
        from harbor.models.task.artifacts import source_host_path

        env = trial.agent_environment
        task_dir = Path(self.config.task.path)
        state = self.trial / self.layout.state
        done = await env.exec(f"cat {shlex.quote(self.layout.tools_mount)}/config.json", user="root")
        tools = json.loads(done.stdout or "{}")
        app_root, agent_user = tools["app_root"], tools["agent_user"]
        manifest = json.loads((state / "manifests" / f"{self.snapshot['digest']}.json").read_text())
        container = frontier_swe.container_name(self.trial.name)
        target = f"/tmp/revive-{self.number}-restore"
        with tempfile.TemporaryDirectory(prefix=f"revive-{self.number}-restore-", dir=self.trial) as scratch:
            root = Path(scratch) / "artifacts"
            frontier_swe.materialize(manifest, state / "blobs", root)
            mirror = source_host_path(root, app_root)
            mirror.mkdir(parents=True, exist_ok=True)
            pack = subprocess.Popen(["tar", "-C", str(mirror), "-cf", "-", "."], stdout=subprocess.PIPE)
            unpack = subprocess.run(["docker", "exec", "-i", "-u", agent_user, container, "sh", "-c",
                                     f"rm -rf {target} && mkdir -p {target} && tar -xpf - -C {target}"],
                                    stdin=pack.stdout, capture_output=True, text=True, timeout=600)
            pack.stdout.close()
            if pack.wait() != 0 or unpack.returncode != 0:
                raise RuntimeError(f"could not copy the snapshot into the container: {unpack.stderr.strip()[-1000:]}")
        command = (f"python3 -c {shlex.quote(RESTORE)} {shlex.quote(self.layout.tools_mount)} {target} "
                   + " ".join(shlex.quote(agent) for agent in agent_ids) + f"; code=$?; rm -rf {target}; exit $code")
        done = await env.exec(command, user=agent_user, timeout_sec=1800)
        if done.return_code != 0:
            raise RuntimeError(f"restoring the snapshot failed: {(done.stdout or '')[-1000:]} {(done.stderr or '')[-1000:]}")
        restored = json.loads((done.stdout or "").strip().splitlines()[-1])
        # The check: /app as the snapshot grader captures it, against the snapshot's manifest.
        described = json.loads((task_dir / self.layout.description).read_text())
        watch = frontier_swe.TrialWatch(self.trial, task_dir, described["graded"], {}, None, False, lambda _: None,
                                        self.layout)
        prefix = source_host_path(Path("."), app_root).as_posix()
        with tempfile.TemporaryDirectory(prefix=f"revive-{self.number}-check-", dir=self.trial) as scratch:
            captured = Path(scratch) / "artifacts"
            captured.mkdir()
            await watch.capture(captured)
            now = frontier_swe.tree_manifest(captured)

        def graded_part(found: dict) -> dict:
            return {key: value for key, value in found.items() if key == prefix or key.startswith(prefix + "/")}

        after, expected = graded_part(now), graded_part(manifest)
        if after != expected:
            differing = sorted(key for key in set(after) | set(expected) if after.get(key) != expected.get(key))
            raise RuntimeError(f"{app_root} after the restore differs from snapshot {self.snapshot.get('seq')} in "
                               f"{len(differing)} entries: {differing[:10]}")
        self.record["rebuild"].update(app_root=app_root, restored=restored, checked_against_snapshot=True,
                                      graded_entries=len(expected))
        return restored

    async def align_task_clock(self, env, started: float) -> None:
        """A task image's sandbox timer (FrontierSWE's sandbox-timer: elapsed and remaining of its time
        budget) counts from its start anchor; moving the anchor to `started` makes it count working
        time only, as the runner's clock does (the downtime and a rebuilt container's new boot do not
        count)."""
        anchor = int(round(started))
        script = (f'd={TASK_CLOCK}; [ -f "$d/start" ] || exit 3; old=$(cat "$d/start"); '
                  f'printf %s {anchor} > "$d/start"; '
                  f'date -u -d @{anchor} +%Y-%m-%dT%H:%M:%SZ > "$d/started_at"; printf %s "$old"')
        done = await env.exec(script, user="root")
        if done.return_code == 3:
            return
        if done.return_code != 0:
            self.record["task_clock"] = {"error": (done.stderr or done.stdout or "")[-300:]}
            return
        before = (done.stdout or "").strip()
        self.record["task_clock"] = {"path": f"{TASK_CLOCK}/start", "before": before, "before_at": iso(
            float(before)) if before.isdigit() else None, "after": anchor, "after_at": iso(anchor)}

    async def dump_ownership(self, env) -> str | None:
        """Who owns each file of the trial's log mounts, before Harbor's prepare_logs_for_host gives
        them to the host user (so the host can read the agents' logs); restore_ownership gives them
        back, so an agent that runs as a user of its own can write its session again."""
        targets = [str(target) for target in env._mount_targets(writable_only=True)]
        path = f"/tmp/revive-{self.number}-ownership.json"
        done = await env.exec(f"python3 -c {shlex.quote(OWNERSHIP_DUMP)} {path} "
                              + " ".join(shlex.quote(t) for t in targets), user="root", timeout_sec=600)
        if done.return_code != 0:
            self.record["ownership"] = {"error": (done.stderr or done.stdout or "")[-300:]}
            return None
        self.record["ownership"] = {"mounts": targets, "entries": int((done.stdout or "0").strip() or 0)}
        return path

    async def restore_ownership(self, env, path: str | None) -> None:
        if path is None:
            return
        done = await env.exec(f"python3 -c {shlex.quote(OWNERSHIP_RESTORE)} {path}", user="root", timeout_sec=600)
        self.record.setdefault("ownership", {})["restored"] = (
            int((done.stdout or "0").strip() or 0) if done.return_code == 0 else (done.stderr or "")[-300:])

    async def give_back(self, trial, env) -> None:
        """The files the agents' runner writes (each agent's session directory and output log, the
        team's shared directory, a claude_peers team's session registry, the task's harness log
        directory) belong to the task's agent user, as when the run wrote them, even if they belonged
        to the host user before the revival (Harbor gives every log to the host user when it tears a
        trial down)."""
        user = trial.task.config.agent.user
        if user in (None, "", "root", 0, "0"):
            return
        _, found = sessions(self.trial, self.cli)
        local = [*(s["dir"] for s in found), *(s["log"] for s in found), self.trial / "agent" / "multiagent-collab",
                 self.trial / "agent" / CLAUDE_PEERS_REGISTRY]
        import frontier_swe
        layout = frontier_swe.trial_layout(self.trial)
        if layout is not None:
            local.append(self.trial / "agent" / layout.log_name)
        targets = [(PurePosixPath("/logs/agent") / path.relative_to(self.trial / "agent")).as_posix()
                   for path in local if path.exists()]
        if not targets:
            return
        done = await env.exec(f"chown -R {shlex.quote(str(user))}: " + " ".join(shlex.quote(t) for t in targets),
                              user="root", timeout_sec=600)
        self.record.setdefault("ownership", {})["agent_user"] = {
            "user": str(user), "paths": targets,
            "result": "ok" if done.return_code == 0 else (done.stderr or done.stdout or "")[-300:]}

    def keep_previous_result(self, trial) -> None:
        """A trial Harbor had ended: its result.json and exception.txt are kept beside as
        *.before-revive-N.*, and what its teardown had begun to collect into artifacts/ (Harbor
        collects again at the end) goes to artifacts.before-revive-N/. The job's result.json is
        copied the same way (finish_job rewrites it)."""
        n = self.number
        kept = {}
        for name, new in (("result.json", f"result.before-revive-{n}.json"),
                          ("exception.txt", f"exception.before-revive-{n}.txt")):
            if (self.trial / name).exists():
                (self.trial / name).replace(self.trial / new)
                kept[name] = new
        artifacts = trial.paths.artifacts_dir
        mount = trial._main_artifacts_mount_dir.relative_to(artifacts).parts[0]
        stale = [child for child in sorted(artifacts.iterdir()) if child.name != mount] if artifacts.is_dir() else []
        if stale:
            aside = self.trial / f"artifacts.before-revive-{n}"
            aside.mkdir(exist_ok=True)
            for child in stale:
                child.replace(aside / child.name)
            kept["artifacts"] = [f"{aside.name}/{child.name}" for child in stale]
        job_result = self.job / "result.json"
        if job_result.is_file() and not (self.job / f"result.before-revive-{n}.json").exists():
            shutil.copy2(job_result, self.job / f"result.before-revive-{n}.json")
            kept["job_result"] = f"../result.before-revive-{n}.json"
        self.record["harbor_ended"] = {"reason": self.harbor_ended, "kept": kept}

    async def prepare(self, trial) -> None:
        """Restart, read the agents, dry-run the revived runner, write the record."""
        env = trial.agent_environment
        self.record.update(revival=self.number, status="starting", supervisor_pid=os.getpid(),
                           log=f"revive-{self.number}.log", note=self.note, run_args=self.run_args,
                           trial_started_at_epoch=self.trial_started, trial_started_at=iso(self.trial_started),
                           code={**{name: sha256(ROOT / name) for name in CODE_FILES}, "repo": str(ROOT),
                                 "harbor": _harbor_version()})
        if self.cap_raised is not None:
            self.record["working_time_cap_raised"] = self.cap_raise_record()
        if self.harbor_ended:
            self.record["harbor_ended"] = {"reason": self.harbor_ended}
        await self.restart(env, trial)
        ownership = await self.dump_ownership(env)
        try:
            await env.prepare_logs_for_host()  # Harbor's step at the end of a run: the host can read the logs
            await self._prepare_agents(trial, env)
        finally:
            # The logs go back to their owners (an agent running as its own user writes them).
            await self.restore_ownership(env, ownership)
            await self.give_back(trial, env)
        if self.harbor_ended:
            self.keep_previous_result(trial)
        self.record["status"] = "running"
        save_records(self.trial, [*self.records, self.record])

    async def _prepare_agents(self, trial, env) -> None:
        await env.run_healthcheck()
        agent = trial.agent
        if self.mode == "rebuilt":
            await self.setup_agent(trial)
        command = agent.get_version_command()
        if command and getattr(agent, "_version", None) is None:
            done = await env.exec(command)
            if done.return_code == 0 and done.stdout:
                agent._version = agent.parse_version(done.stdout)
        if self.mode == "rebuilt":
            _, found = sessions(self.trial, self.cli)
            # Each agent's working copy is /work/<its AGENT_ID> (the single-agent runner's: single).
            await self.restore_snapshot(trial, list(dict.fromkeys(s["agent_id"] for s in found)))

        budget, metric = self.args.token_budget, self.args.budget_metric
        relay = getattr(self.args, "relay", None)
        layout, agents = inspect_agents(self.trial, self.cli, budget, metric, relay)
        self.agents_started = next((r["agent_started_at_epoch"] for r in self.records if "agent_started_at_epoch" in r),
                                   None) or agent_started(self.trial, layout, self.cli)
        lasts = [a["last_activity"] for a in agents if a["last_activity"] is not None]
        last = max(lasts) if lasts else self.agents_started
        now = time.time()
        # A revival that failed before its agents ran (status failed, or never got that far) adds
        # no working time.
        self.worked = worked_before([r for r in self.records if r.get("status") != "failed"
                                     and "worked_seconds_before" in r and "revived_at_epoch" in r],
                                    self.agents_started, last)
        import run
        harbor_timeout = run.harbor_agent_timeout(self.args)
        self.timeout_left = self.agent_timeout_left(self.worked)
        resumed = [a for a in agents if a["finished_before"] is None]
        relay_plan: dict = {}
        if relay is not None:
            resumed, relay_plan = relay_next(agents, relay)
        appended = []
        for a in resumed:
            if a["unlogged_attempt"] is not None:
                data = a["unlogged_attempt"].read_bytes()
                with a["log"].open("ab") as stream:
                    stream.write(data if data.endswith(b"\n") else data + b"\n")
                appended.append(str(a["unlogged_attempt"].relative_to(self.trial)))
        self.note = self.note_for()
        self.plan = {"revival": self.number, "note": self.note, "layout": layout, "started": now - self.worked,
                     "agents": [{"index": a["index"], "session_id": a["session_id"], "turn": a["turn"]}
                                for a in resumed], **relay_plan}
        if self.kind() in EVENT_MESSAGES:
            self.plan["event_message"] = EVENT_MESSAGES[self.kind()].format(cli=agent.CLI_LABEL)
        await self.align_task_clock(env, now - self.worked)
        self.record.update(
            layout=layout, agent_started_at_epoch=self.agents_started, agent_started_at=iso(self.agents_started),
            last_activity_at_epoch=last, last_activity_at=iso(last), revived_at_epoch=now, revived_at=iso(now),
            worked_seconds_before=round(self.worked, 1), downtime_seconds=round(now - last, 1),
            working_time_cap_seconds=self.args.timeout, harbor_agent_timeout_seconds=harbor_timeout,
            agent_timeout_left_seconds=round(self.timeout_left, 1), token_budget=budget, budget_metric=metric,
            appended_interrupted_attempt_logs=appended,
            agents=[{"index": a["index"], "agent_id": a["agent_id"], "session_id": a["session_id"],
                     "session_dir": str(Path(a["session_dir"]).relative_to(self.trial)), "turn": a["turn"],
                     "resumed": any(a is r for r in resumed), "finished_before": a["finished_before"],
                     "usage_before": a["usage_before"], "weighted_tokens_before": a["weighted_tokens_before"],
                     "last_activity_at": iso(a["last_activity"]), **({"leg": a["leg"]} if relay is not None else {})}
                    for a in agents])
        if relay is not None:
            # A relay (run.py --relay): the leg the revived runner starts with, resumed or afresh, and the legs after.
            self.record["relay"] = {"legs": relay, **relay_plan, "resumes_leg": bool(resumed)}
        self.record.update(kind=self.kind(), note=self.note)
        await self.dry_run(trial, layout)
        if self.rewrites_resources() and self.plan["agents"] and relay is None:
            await self.reinstall_resources(trial, env, agents, now - self.worked)
        self.record["differences"] = self.differences(layout)

    def agent_timeout_left(self, worked: float) -> float:
        """Harbor's agent timeout for the revived part: the run's (or, raised, the new cap's) less the
        working time before."""
        import run

        return run.harbor_agent_timeout(self.args) - worked

    def rewrites_resources(self) -> bool:
        """Whether the `resources` command is written again (with --resources-tool): for a revival that
        was not a host restart (a rebuilt container lacks it), and whenever the cap is raised (the run's
        command has the run's cap baked in)."""
        return self.kind() != "classic" or self.cap_raised is not None

    def cap_raise_record(self) -> dict:
        """revive.json's working_time_cap_raised: the run's cap, the new one, and why."""
        import run

        return {"from_seconds": self.run_cap, "to_seconds": self.cap_raised, "on_purpose": True,
                "reason": self.raise_reason, "harbor_agent_timeout_seconds_run": round(self.harbor_timeout_run, 1),
                "harbor_agent_timeout_seconds": run.harbor_agent_timeout(self.args),
                "applies_to": "the revived part: the runner's time cap (limit watcher), Harbor's agent timeout (what "
                              "is left of it after the working time before) and, with --resources-tool, the agents' "
                              "`resources` command (written again)"}

    async def reinstall_resources(self, trial, env, agents: list[dict], started: float) -> None:
        """--resources-tool: the `resources` command lives in the container (a rebuilt one lacks it, and the
        run's has the run's cap baked in); it is written again with the revival's settings (a raised cap),
        counting time left from working time only."""
        agent = trial.agent
        homes = [(PurePosixPath("/logs/agent") / Path(a["session_dir"]).relative_to(self.trial / "agent")
                  / HOME_DIRS[agent_cli(self.cli, a["index"])]).as_posix() for a in agents]
        command = agent._resources_install(homes, started)
        if not command:
            return
        done = await env.exec(command, user="root")
        if done.return_code != 0:
            raise RuntimeError(f"could not install the resources command: {(done.stderr or done.stdout or '')[-500:]}")
        timeout = os.environ.get("RELEASE_AGENT_TIMEOUT", "").strip()
        self.record["resources_tool"] = {"reinstalled": True, "started_at": iso(started),
                                         "timeout_seconds": float(timeout) if timeout else None}

    def note_for(self) -> str:
        """The note each resumed agent gets: the one given, else what happened (DEFAULT_NOTE,
        FAULT_NOTE, REBUILT_NOTE with the snapshot)."""
        if self.note_given:
            return self.note_given
        if self.mode == "rebuilt":
            app = (self.record.get("rebuild") or {}).get("app_root") or "/app"
            moment = datetime.fromisoformat(self.snapshot["ts"].replace("Z", "+00:00")).astimezone(timezone.utc)
            seq = self.snapshot.get("submission_seq")
            template = REBUILT_NOTE if seq is not None else REBUILT_NOTE_NO_SUBMISSION
            return template.format(app=app, seq=seq, time=moment.strftime("%H:%M"))
        return FAULT_NOTE if self.harbor_ended else DEFAULT_NOTE

    async def dry_run(self, trial, layout: str) -> None:
        """Render the revived runner without touching the container; the agent's checks run here."""
        if not self.plan["agents"] and not self.plan.get("relay_leg"):
            self.record["checks"] = []
            return
        agent = trial.agent
        plan = copy.deepcopy(self.plan)
        recorder = RecordingEnvironment()
        quiet = logging.getLogger("revive.dry-run")
        quiet.propagate = False
        saved, agent.logger = agent.logger, quiet
        agent.revive_plan = plan
        try:
            await agent.resume(trial.task.instruction, recorder, SimpleNamespace())
        finally:
            agent.logger = saved
            agent.revive_plan = None
        self.record["checks"] = plan.get("checked", [])
        # The agent phase relies on these checks: by then the run's files belong to their owners
        # again (restore_ownership) and the host user may not read them.
        self.plan["checks_passed"] = True
        # With harness protection the runner is a file the keeper's launcher reads (agent.py keeps it).
        runner = (agent.last_runner_script if os.environ.get("RELEASE_HARNESS_GUARD") == "1"
                  else rendered_runner(layout, recorder.calls))
        (self.trial / f"revive-{self.number}-runner.sh").write_text(runner)
        original = self.trial / "agent/multiagent-collab/run-agent.sh"
        if layout == "team" and original.is_file():
            diff = difflib.unified_diff(original.read_text().splitlines(keepends=True),
                                        runner.splitlines(keepends=True), "run-agent.sh (the run's)",
                                        f"run-agent-revive-{self.number}.sh (this revival's)")
            (self.trial / f"revive-{self.number}-runner.diff").write_text("".join(diff))
            self.record["original_runner_had_limit_resume"] = "start_limit_watch" in original.read_text()

    def differences(self, layout: str) -> list[str]:
        runner = f"revive-{self.number}-runner.sh"
        if layout == "team" and isinstance(self.cli, str):  # (a mixed team's runners are one per agent)
            runner += f"; its differences from the run's in revive-{self.number}-runner.diff"
        rebuilt = self.mode == "rebuilt"
        found = [
            "Each resumed agent got one new user message, the note, in the turn it was in; the runner then went "
            "on with the continuation turns from the next one.",
            *([] if rebuilt else ["Processes the agents had started in the background died with the host; the "
                                  "containers' files survived."]),
            f"The revived runner was rendered by this code with the run's run.py arguments ({runner}); the files "
            "the run wrote from its task instruction and prompts were checked byte-identical to this code's.",
            "Working time counts from the agents' start to their last activity before the crash; the time the host "
            "was down (downtime_seconds) does not count. result.json's agent_execution.started_at is the original "
            "start, so wall-clock hours computed from it include the downtime.",
            *([] if rebuilt else ["Every service of the trial restarted: a task judge or scorer lost gradings that "
                                  "were queued or running at the crash and restored the finished ones from its "
                                  "log."]),
        ]
        if os.environ.get("RELEASE_LIMIT_RESUME") == "1" and not self.record.get("original_runner_had_limit_resume", True):
            found.append("Usage-limit resumption (docs/harness.md) is on for the revived part only.")
        if self.record.get("appended_interrupted_attempt_logs"):
            found.append("The interrupted turn's attempt log was added to the agent's output log by the revival "
                         "(the runner adds each attempt's log when the attempt ends).")
        if self.harbor_ended:
            found.append(f"Harbor had ended the trial on an infrastructure failure, {self.harbor_ended}, without a "
                         f"reward; its result.json and exception.txt are kept as *.before-revive-{self.number}.* "
                         "(harbor_ended)." + (" Its containers, still running, were stopped and started again "
                                              "(stopped_running_containers), so processes the agents had left "
                                              "behind stopped." if self.record.get("stopped_running_containers")
                                              else ""))
        if self.mode == "rebuilt":
            rebuild = self.record.get("rebuild") or {}
            found.append(
                f"The trial's containers were gone: fresh ones were started from the same images and compose files "
                f"(plus the process cap), the agent was set up as Harbor sets it up, and {rebuild.get('app_root')} and "
                f"each agent's working copy were restored from snapshot {rebuild.get('snapshot_seq')} "
                f"({rebuild.get('snapshot_at')}, after submission #{rebuild.get('submission_seq')}), checked against "
                "the snapshot's manifest. Changes after the snapshot, build outputs and other files outside the log "
                "mounts were lost; the note says so.")
        if (self.record.get("pids_limit") or {}).get("limit"):
            found.append(f"Every container of the trial has a pids limit of at most {self.pids_limit} "
                         "(pids_limit; added by the revival where the run had none).")
        if (self.record.get("task_clock") or {}).get("after"):
            found.append("The task's sandbox timer was moved to count working time only (task_clock), like the "
                         "runner's clock.")
        if self.note != DEFAULT_NOTE:
            found.append("The note is the one for this kind of revival (note), not the host-restart note.")
        if self.cap_raised is not None:
            resources = ((self.record.get("resources_tool") or {}).get("reinstalled")
                         and ", and the agents' `resources` command was written again and shows the new limit" or "")
            found.append(f"The working-time cap (run.py --timeout) was raised on purpose from {self.run_cap} s to "
                         f"{self.cap_raised} s for the revived part (working_time_cap_raised; reason: "
                         f"{self.raise_reason}). The runner stops an agent at the new cap, Harbor's agent timeout is "
                         f"what is left of the new one{resources}; an uninterrupted run would have stopped at the "
                         "run's cap.")
        return found

    async def check_network(self, trial) -> None:
        """Under the agent phase's network policy, a host no policy allows must be unreachable."""
        env = trial.agent_environment
        plan = trial._network_plan(None)
        allowed = [host for host in plan.agent_phase.allowed_hosts
                   if not host.startswith("*") and not re.fullmatch(r"[0-9a-fA-F:./]+", host)][:2]
        async with trial._phase_network_policy(env, baseline_policy=plan.agent_env_baseline,
                                               phase_policy=plan.agent_phase):
            done = await env.exec(f"python3 -c {shlex.quote(NETWORK_PROBE)} "
                                  + " ".join(shlex.quote(h) for h in (*CANARY_HOSTS, *allowed)), timeout_sec=180)
        try:
            found = json.loads((done.stdout or "").strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise RuntimeError(f"the network check failed: {done.stdout} {done.stderr}") from None
        self.record["network"] = {"agent_phase": plan.agent_phase.network_mode.value,
                                  "allowed_hosts": list(plan.agent_phase.allowed_hosts), "probe": found}
        save_records(self.trial, [*self.records, self.record])
        open_canaries = [h for h in CANARY_HOSTS if found.get(h, "").startswith("answered")]
        if plan.agent_phase.allowed_hosts and open_canaries:
            raise RuntimeError(f"the egress allowlist is not in force: {open_canaries} answered")

    def summary(self) -> dict:
        found = {"revivals": self.number, "kind": self.kind(), "record": RECORD,
                 "worked_seconds_before": round(self.worked, 1),
                 "downtime_seconds": self.record.get("downtime_seconds"),
                 "weighted_tokens_before": sum(a["weighted_tokens_before"] for a in self.record.get("agents", []))}
        if self.cap_raised is not None:
            found["working_time_cap_raised"] = {"from_seconds": self.run_cap, "to_seconds": self.cap_raised,
                                                "reason": self.raise_reason}
        return found

    # -- after Harbor ----------------------------------------------------------------------------------

    def finish(self, result) -> None:
        after = {"exception": result.exception_info.exception_type if result.exception_info else None,
                 "rewards": result.verifier_result.rewards if result.verifier_result else None}
        try:
            _, agents = inspect_agents(self.trial, self.cli, None, self.args.budget_metric,
                                       getattr(self.args, "relay", None))
            usage = {key: sum(a["usage_before"][key] for a in agents) for key in token_budget.USAGE_KEYS}
            after.update(usage=usage, weighted_tokens=round(token_budget.weighted_tokens(usage)),
                         ended={a["agent_id"] + (f"-leg{a['leg']}" if a.get("leg", 1) > 1 else ""):
                                a["finished_before"] for a in agents})
        except Exception as error:  # noqa: BLE001 - the trial's own records are written already
            after["inspect_error"] = f"{type(error).__name__}: {error}"
        self.record.update(status="finished", finished_at=iso(time.time()), after=after)
        save_records(self.trial, [*self.records, self.record])

    def fail(self, error: BaseException) -> None:
        self.record.update(status="failed", error=f"{type(error).__name__}: {error}", failed_at=iso(time.time()))
        if (self.trial / "config.json").is_file():
            save_records(self.trial, [*self.records, self.record])


def _harbor_version() -> str | None:
    try:
        from importlib.metadata import version
        return version("harbor")
    except Exception:  # noqa: BLE001
        return None


async def finish_job(job: Path) -> None:
    """The job's result.json as Harbor's Job writes it when its trials are done."""
    from harbor.job import Job
    from harbor.metrics.mean import Mean
    from harbor.models.job.config import JobConfig
    from harbor.models.job.result import JobResult, JobStats
    from harbor.models.trial.result import TrialResult
    from harbor.utils.pass_at_k import compute_pass_at_k_by_evals

    path = job / "result.json"
    job_result = JobResult.model_validate_json(path.read_text())
    results = [TrialResult.model_validate_json(p.read_text()) for p in sorted(job.glob("*/result.json"))]
    stats = JobStats.from_trial_results(results, n_total_trials=job_result.n_total_trials,
                                        n_retries=job_result.stats.n_retries)
    try:
        metrics = await Job._resolve_metrics(JobConfig.model_validate_json((job / "config.json").read_text()),
                                             [r.config.task for r in results])
    except Exception:  # noqa: BLE001 - Harbor's default
        metrics = defaultdict(lambda: [Mean()])
    rewards = defaultdict(list)
    for result in results:
        model = result.agent_info.model_info.name if result.agent_info.model_info else None
        key = JobStats.format_agent_evals_key(result.agent_info.name, model, result.source or "adhoc")
        rewards[key].append(result.verifier_result.rewards if result.verifier_result else None)
    for key, values in rewards.items():
        for metric in metrics[key.split("__")[-1]] or [Mean()]:
            stats.evals[key].metrics.append(metric.compute(values))
    for key, pass_at_k in compute_pass_at_k_by_evals(results).items():
        stats.evals[key].pass_at_k = pass_at_k
    job_result.stats = stats
    now = datetime.now()
    job_result.updated_at = now
    if len(results) >= job_result.n_total_trials:
        job_result.finished_at = now
    job_result.stats.update_progress_counts(n_total_trials=job_result.n_total_trials, n_running_trials=0,
                                            n_retries=job_result.stats.n_retries)
    path.write_text(job_result.model_dump_json(indent=4, exclude={"trial_results"}))


async def revive(trial_dir: Path, run_args: list[str], note: str | None = None,
                 raise_timeout: int | None = None, raise_reason: str | None = None) -> int:
    """The supervisor: revive the trial and finish it (the module's steps 1-5)."""
    import run

    revival = Revival(trial_dir, run_args, note, raise_timeout, raise_reason)
    apply_environment(revival.env)
    trial = await revived_trial_class().load(revival.config, revival)
    try:
        await revival.prepare(trial)
    except BaseException as error:
        # Leave the trial as it was: containers stopped (or, rebuilt ones, removed again), its
        # result.json, if Harbor had written one, where it was (it is moved last).
        revival.fail(error)
        if revival.mode == "rebuilt":
            try:
                await trial.agent_environment._run_docker_compose_command(["down", "--remove-orphans"])
            except Exception as down:  # noqa: BLE001
                print(f"could not remove the rebuilt containers: {down}", flush=True)
        else:
            for container in reversed(project_containers_of(trial)):
                subprocess.run(["docker", "stop", container], capture_output=True, timeout=180)
        trial._close_logger_handler()
        raise
    print(f"revival {revival.number} of {revival.trial}: resuming {len(revival.plan['agents'])} agent(s), "
          + (f"relay leg {revival.plan['relay_leg']} ({'resumed' if revival.plan['agents'] else 'afresh'}), "
             if revival.plan.get("relay_leg") else "") +
          f"{revival.record['worked_seconds_before']} s worked before, {revival.timeout_left:.0f} s of Harbor's "
          f"agent timeout left, working-time cap {revival.args.timeout} s"
          + (f" (raised from {revival.run_cap} s)" if revival.cap_raised is not None else ""), flush=True)
    worker, log, grace = run.start_host_worker(revival.args, host_worker_command(revival.cmd, revival.config.task.path),
                                               revival.env, revival.job, revive=True)
    finished = False
    try:
        result = await trial.run()
        finished = True
    finally:
        run.stop_host_worker(worker, log, grace, finished)
    revival.finish(result)
    run.after_harbor(revival.args, revival.job)
    await finish_job(revival.job)
    print(f"revival {revival.number} finished: {result.verifier_result.rewards if result.verifier_result else None}",
          flush=True)
    return 0


def host_worker_command(cmd: list[str], task_path) -> list[str]:
    """The Harbor command run.py's host worker reads (the snapshot grader takes the task directory
    from its -p): harness() built it under a probe name, whose task directory a dry run never
    writes, so -p becomes the trial's own task directory."""
    found = list(cmd)
    found[found.index("-p") + 1] = str(task_path)
    return found


def project_containers_of(trial) -> list[str]:
    from harbor.environments.docker.docker import _sanitize_docker_compose_project_name
    try:
        containers = project_containers(_sanitize_docker_compose_project_name(trial.agent_environment.session_id))
        return [c["Name"].lstrip("/") for c in start_order(containers)]
    except Exception:  # noqa: BLE001
        return []


def read_json_or_none(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def plan_only(trial_dir: Path, run_args: list[str], raise_timeout: int | None = None,
              raise_reason: str | None = None) -> dict:
    """What a revival would do, from what the host can read now (the agents' logs may belong to the
    container's user until a revival hands them over)."""
    import run

    trial = Path(trial_dir).resolve()
    revival = Revival(trial, run_args, raise_timeout=raise_timeout, raise_reason=raise_reason)
    layout, found = sessions(trial, revival.cli)
    agents, lasts = [], []
    for session in found:
        logs = attempt_logs(session["dir"])
        usage_file = session["dir"] / "token-usage.json"
        cli = agent_cli(revival.cli, session["index"])
        last = last_activity(cli, session["dir"] / HOME_DIRS[cli])
        if last is not None:
            lasts.append(last)
        wait = read_json_or_none(session["dir"] / limit_resume.STATE) or {}
        agents.append({"agent_id": session["agent_id"],
                       **({"leg": session["leg"]} if getattr(revival.args, "relay", None) else {}),
                       "session_id": session_id_of(cli, logs),
                       "turn": turn_of(logs), "finished_before": finished(session["dir"], session["log"]),
                       "last_attempt_log_written": iso(logs[-1].stat().st_mtime) if logs else None,
                       "last_activity": iso(last),
                       "token_usage": json.loads(usage_file.read_text()) if usage_file.is_file() else None,
                       # The runner's own count when it stopped (limit_resume.py's limit-wait.json).
                       "runner_working_time": {key: wait.get(key) for key in (
                           "active_seconds", "waited_seconds", "timeout_seconds")} if wait else None,
                       "interrupted_attempt_not_in_log": str(unlogged_attempt(session["log"], logs) or "")})
    # The working time before and what is left, as the revival would count them now (None while the
    # CLIs' session logs are unreadable here).
    harbor_timeout = run.harbor_agent_timeout(revival.args)
    clock = {"working_time_cap_seconds": revival.args.timeout, "run_working_time_cap_seconds": revival.run_cap,
            "working_time_cap_raised": revival.cap_raise_record() if revival.cap_raised is not None else None,
            "harbor_agent_timeout_seconds": harbor_timeout,
            "harbor_agent_timeout_seconds_run": round(revival.harbor_timeout_run, 1)}
    if lasts:
        started = next((r["agent_started_at_epoch"] for r in revival.records if "agent_started_at_epoch" in r),
                       None) or agent_started(trial, layout, revival.cli)
        worked = worked_before([r for r in revival.records if r.get("status") != "failed"
                                and "worked_seconds_before" in r and "revived_at_epoch" in r], started, max(lasts))
        clock.update(worked_seconds_before=round(worked, 1),
                    working_time_left_seconds=round(revival.args.timeout - worked, 1),
                    agent_timeout_left_seconds=round(harbor_timeout - worked, 1))
    else:
        clock.update(worked_seconds_before=None, working_time_left_seconds=None, agent_timeout_left_seconds=None)
    from harbor.environments.docker.docker import _sanitize_docker_compose_project_name

    containers = project_containers(_sanitize_docker_compose_project_name(f"{trial.name}__env"))
    what = {"containers": {c["Name"].lstrip("/"): c["State"].get("Status") for c in containers},
            "harbor_ended": revival.harbor_ended, "pids_limit": revival.pids_limit,
            "pids_limit_added_to_config": revival.pids_added}
    if not containers:
        import frontier_swe
        layout_found = frontier_swe.trial_layout(trial)
        try:
            snapshot = newest_snapshot(trial, layout_found) if layout_found else None
            what["action"] = ("rebuild, restoring snapshot " + json.dumps(
                {k: snapshot.get(k) for k in ("seq", "ts", "digest", "submission_seq", "files")}) if snapshot
                else "refuse: the containers are gone and the trial has no snapshot store")
        except RuntimeError as error:
            what["action"] = f"refuse: {error}"
    else:
        running = [c["Name"].lstrip("/") for c in containers if c["State"].get("Running")]
        what["action"] = ("stop the running containers, then start them again" if revival.harbor_ended and running
                          else "start the stopped containers")
    return {"trial": str(trial), "revival": revival.number, "layout": layout, "agents": agents, **what,
            "agent_timeout_multiplier": revival.config.agent_timeout_multiplier, "time": clock,
            "harness": {key: revival.env.get(key) for key in sorted(revival.env) if key.startswith(
                ("RELEASE_", "MULTIAGENT_")) and key != "RELEASE_CONTINUATION"}}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    run_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("plan", "start", "run"))
    parser.add_argument("trial", type=Path)
    parser.add_argument("--note", help="the one message each resumed agent gets (default: the note for what "
                                       "happened: DEFAULT_NOTE, FAULT_NOTE or REBUILT_NOTE)")
    parser.add_argument("--raise-timeout", type=int, metavar="SECONDS",
                        help="raise the working-time cap (run.py --timeout) to SECONDS for the revived part, on "
                             "purpose (only raising; needs --raise-reason; RUN_ARGS keep the run's --timeout or give "
                             "SECONDS)")
    parser.add_argument("--raise-reason", metavar="TEXT", help="why the cap is raised (revive.json records it)")
    args = parser.parse_args(argv)
    if not run_args:
        parser.error("give the run's run.py arguments after --")
    if args.raise_reason and args.raise_timeout is None:
        parser.error("--raise-reason goes with --raise-timeout")
    trial = args.trial.resolve()
    raising = {"raise_timeout": args.raise_timeout, "raise_reason": args.raise_reason}
    if args.command == "plan":
        print(json.dumps(plan_only(trial, run_args, **raising), indent=2, sort_keys=True))
        return 0
    if args.command == "start":
        Revival(trial, run_args, args.note, **raising)  # fail here, not in the background, on a wrong argument
        number = len(load_records(trial)) + 1
        log, pid_file = trial / f"revive-{number}.log", trial / f"revive-{number}.pid"
        pid_file.unlink(missing_ok=True)
        options = [*(["--note", args.note] if args.note else []),
                   *(["--raise-timeout", str(args.raise_timeout), "--raise-reason", args.raise_reason]
                     if args.raise_timeout is not None else [])]
        # `setsid -f`: a process left in the caller's session can die with it when that session closes.
        with log.open("a") as stream:
            subprocess.run(["setsid", "-f", "nohup", "nice", "-n", "10", sys.executable, str(Path(__file__).resolve()),
                            "run", str(trial), *options, "--", *run_args],
                           cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, check=True)
        for _ in range(100):
            if pid_file.is_file():
                break
            time.sleep(0.1)
        print(json.dumps({"pid": int(pid_file.read_text()) if pid_file.is_file() else None, "log": str(log)}))
        return 0
    (trial / f"revive-{len(load_records(trial)) + 1}.pid").write_text(str(os.getpid()))
    return asyncio.run(revive(trial, run_args, args.note, **raising))


if __name__ == "__main__":
    sys.exit(main())
