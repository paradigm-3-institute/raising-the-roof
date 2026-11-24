#!/usr/bin/env python3
"""Harness protection: an agent's own commands must not stop the harness (docs/harness.md).

An agent's shell runs in the same container as the harness's runner (the bash loop that starts the
agent's CLI turn after turn) and its watchers (token budget, working time and usage-limit waits,
episode). `pkill -x python3`, `pkill -f <pattern>`, `killall`, `kill -9 -1` and the like, meant for
the agent's own processes, used to hit them too: the budget and the time cap stopped being enforced,
or the trial ended early.

agent.py installs this file in the container as GUARD_DIR/hsvd, next to links to the container's
python3, bash and sed under names no pattern is likely to match (hsvi, hsvx, hsvs), and starts
`hsvi hsvd k`: the keeper. Where the task's agent is not root, the keeper runs as root, so the
agent cannot signal it or the watchers at all. The keeper

- runs every agent's watchers and starts one again within a second if it dies (logged in the
  agent's log); a watcher that keeps dying ends its agent cleanly (harness-stop.json), never
  silently;
- starts each agent's runner as the agent's user with the environment `docker exec` gives it, and
  starts it again if a signal ends it; the new runner waits for the agent's CLI if that still
  runs, then goes on where the old one was (agent.py's recover_runner);
- ignores every signal that can be ignored, as the runner does (its own trap).

agent.py starts the keeper again if it dies (a root agent's `kill -9 -1` takes everything): the new
keeper adopts runners that still run and starts the rest in recover mode. Python 3.7+, standard
library only.

Commands (run as `hsvi hsvd COMMAND`; the settings are GUARD_DIR/conf.json):
  keep (k)                      the keeper; HSV_RECOVER set: after a keeper that died
  watch (w) KEY                 one watcher of one agent (the keeper starts these): KEY is the first
                                letter of token, limit, episode or hold (team_hold.py) and the agent's index
  cli-wait HOME                 returns once no CLI process of the agent with that CLI home runs
  recover-state CLI SESSION_DIR HOME LOG
                                where a runner that was killed had got to, as TURN|STATE|SESSION|SINCE|LOG:
                                the turn, whether its last attempt is complete or interrupted, the session
                                id, when the turn's prompt was written, and the last attempt's log; also
                                adds an attempt log the dead runner had not yet added to the agent's log
  home-session CLI HOME         the newest session id in the CLI's own session logs
  pending-command CLI LOG       the shell command the CLI was running when its attempt log ends, if any
  status [--kick SECONDS]       prints "hsv-status {json}"; continues a stopped keeper; with --kick,
                                kills a keeper whose heartbeat is older than SECONDS
  stop REASON                   ends every agent cleanly: stops their CLIs, runners and watchers and
                                marks each session (harness-stop.json, an event in the agent's log)
  probe                         prints "hsv-mode privileged" if this user can run the keeper for another
"""

from __future__ import annotations

import importlib.util
import json
import mmap
import os
import re
import runpy
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

STOP_MARKER = "harness-stop.json"
EXIT_FILE = "runner-exit-status"
LIMIT_STATE = "limit-watch-state.json"
# What a run (or a revival) must not find from an earlier runner: the watchers' markers and state (with
# the team limit hold, team_hold.py's too).
FRESH = ("token-budget-exhausted.json", "limit-wait-exhausted.json", "time-cap-reached.json", "episode-over.json",
         STOP_MARKER, EXIT_FILE, LIMIT_STATE, "team-hold.json", "team-hold-request.json")
IGNORED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT, signal.SIGUSR1, signal.SIGUSR2,
           signal.SIGALRM)
# A runner or watcher that dies within QUICK_SECONDS of its start, MAX_QUICK_DEATHS times in a row, is not
# started again.
QUICK_SECONDS = 60.0
MAX_QUICK_DEATHS = 5
POLL_SECONDS = 0.5
HEARTBEAT_SECONDS = 5.0
ATTEMPT = re.compile(r"^(?:claude|codex)-attempt-(\d+)(?:-.*)?\.jsonl$")
WATCHER_LABELS = {"token": "token-budget watcher", "limit": "working-time watcher", "episode": "episode watcher",
                  "hold": "team-hold watcher"}
# The watchers that record something when they are stopped get SIGTERM, the others SIGKILL.
TERM_WATCHERS = ("limit", "hold")


def here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json(path: str, data) -> None:
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as stream:
        json.dump(data, stream, sort_keys=True)
    os.replace(tmp, path)


def read_json(path: str):
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def load_conf(directory: str) -> dict:
    global IDENTITY
    conf = read_json(os.path.join(directory, "conf.json"))
    if not isinstance(conf, dict):
        raise SystemExit(f"harness guard: no settings in {directory}/conf.json")
    IDENTITY = conf.get("identity")
    return conf


def cli_setting(conf: dict, spec: dict, key: str):
    """A setting of the agent's CLI (home_env, binary, label): a mixed team's agents have their own in
    their unit (agent.py's MixedTeamAgent), the others the trial's."""
    return spec.get(key, conf.get(key))


# ---- processes ---------------------------------------------------------------------------------------

def proc_stat(pid: int):
    """(state, start time in clock ticks) of a process, or None. The start time tells a process
    from a later one with the same pid."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as stream:
            fields = stream.read().rsplit(b")", 1)[1].split()
        return fields[0].decode(), int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def alive(pid, start=None) -> bool:
    found = proc_stat(int(pid)) if pid else None
    return bool(found) and found[0] not in ("Z", "X") and (start is None or found[1] == start)


def environ_of(pid: int):
    try:
        with open(f"/proc/{pid}/environ", "rb") as stream:
            return stream.read().split(b"\0")
    except OSError:
        return None


def pids() -> list:
    return [int(name) for name in os.listdir("/proc") if name.isdigit() and int(name) != os.getpid()]


# The agent's identity (conf.json's), for a keeper or watcher that runs as root for an agent that is not.
IDENTITY = None


def readable(entry: str) -> list:
    """(start, pid) of the processes this user may read that started with ENTRY in their environment."""
    wanted = entry.encode()
    found = []
    for pid in pids():
        environ = environ_of(pid)
        stat = proc_stat(pid) if environ and wanted in environ else None
        if stat and stat[0] not in ("Z", "X"):
            found.append([stat[1], pid])
    return found


def as_agent(function, *args):
    """FUNCTION's result when run as the agent's user in a child process, or None. Root in a container
    cannot read another user's /proc/<pid>/environ (that takes CAP_SYS_PTRACE, which Docker does not
    give), so it finds an agent's processes as that agent."""
    identity = IDENTITY
    if not identity or os.geteuid() != 0 or int(identity["uid"]) == 0:
        return None
    read, write = os.pipe()
    child = os.fork()
    if child == 0:
        try:
            os.close(read)
            os.setgroups([int(group) for group in identity["groups"]])
            os.setgid(int(identity["gid"]))
            os.setuid(int(identity["uid"]))
            os.write(write, json.dumps(function(*args)).encode())
        finally:
            os._exit(0)
    os.close(write)
    data = b""
    with os.fdopen(read, "rb") as stream:
        data = stream.read()
    os.waitpid(child, 0)
    try:
        return json.loads(data)
    except ValueError:
        return None


def tagged(entry: str) -> list:
    """The processes started with ENTRY in their environment, oldest first."""
    found = {pid: start for start, pid in readable(entry)}
    for start, pid in as_agent(readable, entry) or ():
        found[pid] = start
    found.pop(os.getpid(), None)
    return [pid for _, pid in sorted((start, pid) for pid, start in found.items())]


def agent_processes(env_name: str, home: str) -> list:
    """Every process whose environment points at the agent's CLI home: its CLI and all it started."""
    return tagged(f"{env_name}={home}")


def cli_processes(env_name: str, home: str, binary: str) -> list:
    """The agent's CLI itself: of the processes above, those running the CLI's binary."""
    real = os.path.realpath(binary)
    name = os.path.basename(binary)[:15]
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


def send(pid: int, sig) -> None:
    try:
        os.kill(pid, sig)
    except OSError:
        pass


def stop_agent(env_name: str, home: str) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        found = agent_processes(env_name, home)
        for pid in found:
            send(pid, sig)
        if not found:
            return
        if sig == signal.SIGTERM:
            time.sleep(5)


# ---- the logs ----------------------------------------------------------------------------------------

def append_line(path: str, record: dict) -> None:
    try:
        with open(path, "ab") as stream:
            stream.write((json.dumps(record) + "\n").encode())
    except OSError:
        pass


def guard_log(conf: dict, event: str, **data) -> None:
    append_line(conf["log"], {"ts": now_iso(), "event": event, **data})


def agent_event(unit: dict, event: str, message: str, **data) -> None:
    """An event in the agent's log, in the runner's own format."""
    append_line(unit["log"], {"type": "environment.feedback", "data": {"event": event, "message": message, **data}})


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


# ---- where a killed runner had got to ----------------------------------------------------------------

def attempt_logs(session_dir: str) -> list:
    """The runner's attempt logs, oldest first."""
    found = []
    try:
        names = os.listdir(session_dir)
    except OSError:
        return []
    for name in names:
        if ATTEMPT.match(name):
            path = os.path.join(session_dir, name)
            try:
                found.append((os.stat(path).st_mtime, name, path))
            except OSError:
                continue
    return [path for _, _, path in sorted(found)]


def session_id_of(cli: str, logs: list):
    found = None
    for path in logs:
        for event in events(path):
            value = event.get("session_id") if cli == "claude" else (
                event.get("thread_id") if event.get("type") == "thread.started" else None)
            if isinstance(value, str) and value:
                found = value
    return found


def home_session(cli: str, home: str):
    """The newest session in the CLI's own session logs (an attempt log may not name it: sed
    buffers the CLI's events, and a kill loses what was in the buffer)."""
    newest = None
    top = os.path.join(home, "projects" if cli == "claude" else "sessions")
    for root, _, names in os.walk(top):
        for name in names:
            if not name.endswith(".jsonl") or (cli == "codex" and not name.startswith("rollout-")):
                continue
            path = os.path.join(root, name)
            if cli == "claude":
                session = name[:-len(".jsonl")]
            else:
                session = next((str(e["payload"].get("id")) for e in events(path)
                                if e.get("type") == "session_meta" and isinstance(e.get("payload"), dict)), None)
            try:
                stamp = os.stat(path).st_mtime
            except OSError:
                continue
            if session and (newest is None or stamp > newest[0]):
                newest = (stamp, session)
    return newest[1] if newest else None


def attempt_complete(cli: str, path: str) -> bool:
    """Whether the CLI ended the attempt's turn itself and without an error (as the runner's
    finish_<cli>_attempt reads it); otherwise the turn was interrupted or failed."""
    complete = False
    for event in events(path):
        kind = event.get("type")
        if cli == "claude":
            if kind == "result":
                complete = not event.get("is_error")
        elif kind in ("turn.failed", "error"):
            complete = False
        elif kind == "turn.completed":
            complete = True
    return complete


def pending_command(cli: str, path: str) -> str:
    """The shell command the CLI had started and not finished when the attempt's log ends, or "".
    A CLI that is killed has not saved that command in its session yet, so the agent, resumed, does
    not know it ran it."""
    pending: dict = {}
    for event in events(path):
        kind = event.get("type")
        if cli == "claude":
            blocks = (event.get("message") or {}).get("content")
            for block in blocks if isinstance(blocks, list) else ():
                if not isinstance(block, dict):
                    continue
                if kind == "assistant" and block.get("type") == "tool_use":
                    command = (block.get("input") or {}).get("command")
                    if isinstance(command, str):
                        pending[block.get("id")] = command
                elif kind == "user" and block.get("type") == "tool_result":
                    pending.pop(block.get("tool_use_id"), None)
            if kind == "result":
                pending.clear()
        else:
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "command_execution":
                if kind == "item.started" and isinstance(item.get("command"), str):
                    pending[item.get("id")] = item["command"]
                elif kind == "item.completed":
                    pending.pop(item.get("id"), None)
            elif kind in ("turn.completed", "turn.failed"):
                pending.clear()
    commands = [command.strip() for command in pending.values() if command.strip()]
    return commands[-1][:2000] if commands else ""


def append_unlogged(log: str, attempt: str) -> bool:
    """The runner adds each attempt's log to the agent's log when the attempt ends; a runner that
    was killed had not added its last one."""
    try:
        with open(attempt, "rb") as stream:
            data = stream.read()
    except OSError:
        return False
    lines = [line for line in data.splitlines() if line.strip()]
    if not lines:
        return False
    try:
        with open(log, "rb") as stream:
            if os.fstat(stream.fileno()).st_size:
                with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as view:
                    if view.find(lines[-1]) >= 0:
                        return False
    except (OSError, ValueError):
        pass
    try:
        with open(log, "ab") as stream:
            stream.write(data if data.endswith(b"\n") else data + b"\n")
    except OSError:
        return False
    return True


def recover_state(cli: str, session_dir: str, home: str, log: str) -> list:
    """[turn, complete or interrupted, session id, when the turn's prompt was written, last attempt
    log]; the turn is empty if the runner had not started the CLI yet."""
    logs = attempt_logs(session_dir)
    if not logs:
        return ["", "none", "", "0", ""]
    turn = max(int(ATTEMPT.match(os.path.basename(path)).group(1)) for path in logs)
    last = logs[-1]
    append_unlogged(log, last)
    session = session_id_of(cli, logs) or home_session(cli, home) or ""
    since = 0.0
    if turn > 0:
        try:
            since = os.stat(os.path.join(session_dir, f"{cli}-no-action-feedback-{turn}.txt")).st_mtime
        except OSError:
            since = 0.0
    return [str(turn), "complete" if attempt_complete(cli, last) else "interrupted", session, f"{since:.3f}", last]


def cli_wait(conf: dict, home: str, poll: float = 2.0) -> None:
    spec = next((unit for unit in conf.get("units") or () if unit.get("home") == home), {})
    while cli_processes(cli_setting(conf, spec, "home_env"), home, cli_setting(conf, spec, "binary")):
        time.sleep(poll)


# ---- the keeper --------------------------------------------------------------------------------------

def ignore_signals() -> None:
    for sig in IGNORED:
        signal.signal(sig, signal.SIG_IGN)


def child_setup(identity, ignored=()):
    """Run in a child before its command: the signals as `docker exec` leaves them (the keeper
    ignores them, and an ignored signal stays ignored in every process the child starts), then
    IGNORED ones, and the agent's user if the keeper runs as root for an agent that is not."""
    def setup() -> None:
        for sig in IGNORED:
            signal.signal(sig, signal.SIG_DFL)
        for name in ignored:
            signal.signal(getattr(signal, "SIG" + name), signal.SIG_IGN)
        if identity and os.geteuid() == 0 and int(identity["uid"]) != 0:
            os.setgroups([int(group) for group in identity["groups"]])
            os.setgid(int(identity["gid"]))
            os.setuid(int(identity["uid"]))
    return setup


def initial_environ() -> dict:
    """The environment the keeper was started with. Not os.environ: Python adds LC_CTYPE to that
    when the locale is C (its locale coercion), and no runner had that variable before."""
    try:
        with open("/proc/self/environ", "rb") as stream:
            entries = stream.read().split(b"\0")
    except OSError:
        return dict(os.environ)
    found = {}
    for entry in entries:
        name, separator, value = entry.partition(b"=")
        if separator and name:
            found[os.fsdecode(name)] = os.fsdecode(value)
    return found


def runner_env(conf: dict, spec: dict) -> dict:
    """The environment an agent's runner has always had: what `docker exec` gives the agent's user,
    and, for a team's runner (a child of the shell `docker exec` started, which the single-agent
    runner was itself), that shell's PATH and each agent's own variables in place of the team's.
    Without SHLVL the runner's shell is at level 1, as both runners were (bash lowers the level for
    a command it starts with `&`, as the team's runners were started)."""
    env = {name: value for name, value in initial_environ().items() if not name.startswith("HSV_")}
    identity = conf.get("identity")
    if identity and identity.get("home"):
        env["HOME"] = identity["home"]
    env.pop("SHLVL", None)
    if conf.get("layout") == "team":
        env["PATH"] = f"{env.get('HOME', '')}/.local/bin:{env.get('PATH', '')}"
    for name, source in (spec.get("env_from") or {}).items():
        env[name] = env.get(source, "")
    for name in spec.get("env_unset") or ():
        env.pop(name, None)
    return env


class Unit:
    """One agent: its runner and its watchers."""

    def __init__(self, conf: dict, spec: dict, directory: str):
        self.conf = conf
        self.spec = spec
        self.directory = directory
        self.index = int(spec["index"])
        self.runner = None          # a Popen, or (pid, start) of a runner adopted from a dead keeper
        self.watchers: dict = {}    # name -> {"process", "started", "quick", "restarts", "due"}
        self.done = None            # the runner's exit code once the agent has ended
        self.restarts = 0
        self.quick = 0
        self.started = 0.0
        self.due = None             # when to start the runner again
        self.failed = None          # why protection failed, once it has
        self.last_stop = 0.0

    @property
    def exit_file(self) -> str:
        return os.path.join(self.spec["session_dir"], EXIT_FILE)

    # -- the runner

    def start_runner(self, recover: int = 0) -> None:
        try:
            os.unlink(self.exit_file)
        except OSError:
            pass
        env = runner_env(self.conf, self.spec)
        env["HSV_UNIT"] = unit_tag(self.directory, self.index)
        if recover:
            env["HSV_RECOVER"] = str(recover)
        self.runner = subprocess.Popen(
            [os.path.join(self.directory, "hsvx"), os.path.join(self.directory, "hsvr")], env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
            preexec_fn=child_setup(self.conf.get("identity"), self.spec.get("ignore_signals", ())))
        self.started = time.monotonic()
        self.due = None
        guard_log(self.conf, "runner_started", agent=self.spec["id"], pid=self.runner.pid, recover=recover)

    def runner_pid(self):
        if self.runner is None:
            return None
        return self.runner.pid if isinstance(self.runner, subprocess.Popen) else self.runner[0]

    def runner_ended(self):
        """None while the runner runs, else (its exit code, whether a signal ended it)."""
        if self.runner is None:
            return None
        if isinstance(self.runner, subprocess.Popen):
            code = self.runner.poll()
            return None if code is None else (code, code < 0)
        if alive(*self.runner):
            return None
        # Not this keeper's child: the runner leaves its exit code as it ends; killed, it leaves none.
        code = read_exit(self.exit_file)
        return (-9, True) if code is None else (code, False)

    # -- the watchers

    def start_watcher(self, name: str) -> None:
        spec = self.spec["watchers"][name]
        state = self.watchers.setdefault(name, {"quick": 0, "restarts": 0})
        env = {"PATH": os.environ.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
               "HOME": "/"}
        identity = None
        if spec.get("as_agent"):
            # With the agent's identity and environment, as the runner started it before.
            env, identity = runner_env(self.conf, self.spec), self.conf.get("identity")
        env.update(spec.get("env") or {})
        env["HSV_WATCH"] = watch_tag(self.directory, name, self.index)
        with open(spec["log"], "ab") as errors:
            state["process"] = subprocess.Popen(
                [os.path.join(self.directory, "hsvi"), os.path.join(self.directory, "hsvd"), "w",
                 f"{name[0]}{self.index}"], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=errors, start_new_session=True, preexec_fn=child_setup(identity))
        state["started"] = time.monotonic()
        state["due"] = None

    def stop_watchers(self) -> None:
        """The limit watcher records the final times when it is stopped; give it up to 6 s."""
        for name, state in self.watchers.items():
            process = state.get("process")
            if process is not None and process.poll() is None:
                send(process.pid, signal.SIGTERM if name in TERM_WATCHERS else signal.SIGKILL)
        deadline = time.monotonic() + 6
        for state in self.watchers.values():
            process = state.get("process")
            if process is None:
                continue
            try:
                process.wait(timeout=max(deadline - time.monotonic(), 0.1))
            except subprocess.TimeoutExpired:
                send(process.pid, signal.SIGKILL)
                process.wait()
            state["process"] = None

    def check_watchers(self, now: float) -> None:
        for name in self.spec["watchers"]:
            state = self.watchers.get(name)
            if state is None:
                continue
            process = state.get("process")
            if process is not None:
                code = process.poll()
                if code is None:
                    continue
                state["process"] = None
                lived = now - state["started"]
                state["quick"] = state["quick"] + 1 if lived < QUICK_SECONDS else 0
                state["restarts"] += 1
                guard_log(self.conf, "watcher_died", agent=self.spec["id"], watcher=name, exit=code,
                          lived_seconds=round(lived, 1))
                if state["quick"] >= MAX_QUICK_DEATHS:
                    self.fail(f"its {WATCHER_LABELS.get(name, name)} ended {state['quick']} times in a row "
                              f"within {QUICK_SECONDS:.0f} s of starting")
                    return
                state["due"] = now + min(2 ** state["quick"], 30) - 1
                state["exit"] = code
            if state.get("due") is not None and now >= state["due"] and self.failed is None:
                self.start_watcher(name)
                agent_event(self.spec, "harness_watcher_restarted",
                            f"The harness's {WATCHER_LABELS.get(name, name)} for this agent had ended (a signal, "
                            "probably from an agent's own command); it was started again.",
                            watcher=name, exitCode=state.get("exit"), restart=state["restarts"])
                guard_log(self.conf, "watcher_restarted", agent=self.spec["id"], watcher=name,
                          restart=state["restarts"], pid=state["process"].pid)

    # -- ending

    def fail(self, reason: str) -> None:
        """Protection failed for this agent: it must not go on unwatched. Mark its session and stop
        its CLI; its runner then ends cleanly (agent.py's harness_stop_reached)."""
        if self.failed is not None:
            return
        self.failed = reason
        mark_stopped(self.conf, self.spec, reason)

    def stop_if_failed(self, now: float) -> None:
        if self.failed is not None and now - self.last_stop >= 2:
            self.last_stop = now
            for pid in agent_processes(cli_setting(self.conf, self.spec, "home_env"), self.spec["home"]):
                send(pid, signal.SIGKILL)

    def finish(self, code: int) -> None:
        self.stop_watchers()
        self.done = code
        self.runner = None
        guard_log(self.conf, "agent_ended", agent=self.spec["id"], exit=code)


def unit_tag(directory: str, index: int) -> str:
    """What a runner's environment starts with (HSV_UNIT; its launcher unsets it, so no process
    the runner starts has it): the guard directory keeps one container's, or one test's, apart."""
    return f"{directory}#{index}"


def watch_tag(directory: str, name: str, index: int) -> str:
    return f"{directory}#{name}:{index}"


def old_watchers(directory: str, index: int) -> list:
    return [pid for name in WATCHER_LABELS for pid in tagged(f"HSV_WATCH={watch_tag(directory, name, index)}")]


def read_exit(path: str):
    try:
        with open(path) as stream:
            return int(stream.read().strip())
    except (OSError, ValueError):
        return None


def mark_stopped(conf: dict, spec: dict, reason: str) -> None:
    marker = os.path.join(spec["session_dir"], STOP_MARKER)
    if not os.path.exists(marker):
        try:
            write_json(marker, {"ts": now_iso(), "reason": reason})
        except OSError:
            pass
        agent_event(spec, "harness_protection_failed",
                    f"The harness could not keep this agent's protection running ({reason}), so "
                    f"{spec.get('label', conf.get('label', 'the CLI'))} was stopped; the run is graded as usual.",
                    reason=reason)
    guard_log(conf, "protection_failed", agent=spec["id"], reason=reason)


def state_path(directory: str) -> str:
    return os.path.join(directory, "state", "keeper.json")


def summary(conf: dict, exits: dict) -> tuple:
    """What the runner's shell printed and returned: each agent's exit code, and success if any is 0."""
    ids = {str(spec["index"]): spec["id"] for spec in conf["units"]}
    text = "Agent exits: " + ", ".join(f"{ids.get(index, index)}={code}" for index, code in sorted(
        exits.items(), key=lambda item: int(item[0])))
    return text, 0 if any(code == 0 for code in exits.values()) else 1


def finish_output(conf: dict, exits: dict) -> int:
    text, code = summary(conf, exits)
    print(text, flush=True)
    if code:
        # The single-agent runner's own output (bash errors, a missing credential) went to this file.
        for spec in conf["units"]:
            path = spec.get("output")
            if path and exits.get(str(spec["index"])) not in (None, 0):
                try:
                    with open(path, "rb") as stream:
                        sys.stdout.write(stream.read()[-4000:].decode(errors="replace"))
                except OSError:
                    pass
        sys.stdout.flush()
    return code


def keep(directory: str, recover: int) -> int:
    ignore_signals()
    conf = load_conf(directory)
    path = state_path(directory)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    previous = read_json(path) or {}
    me = (os.getpid(), (proc_stat(os.getpid()) or ("", 0))[1])

    # One keeper at a time. Its `docker exec` can end while it runs on: then follow it.
    while previous.get("state") == "running" and alive(previous.get("pid"), previous.get("start")) \
            and previous.get("pid") != me[0]:
        time.sleep(1)
        previous = read_json(path) or previous
    if previous.get("state") == "finished" and recover:
        return finish_output(conf, previous.get("exits") or {})

    units = [Unit(conf, spec, directory) for spec in conf["units"] if spec["index"] in conf["running"]]
    earlier = previous.get("units") or {}
    for unit in units:
        if recover:
            before = earlier.get(str(unit.index)) or {}
            unit.restarts = int(before.get("restarts") or 0)
            for pid in old_watchers(directory, unit.index):
                send(pid, signal.SIGKILL)
            old = tagged(f"HSV_UNIT={unit_tag(directory, unit.index)}")
            if before.get("done") is not None:
                unit.done = int(before["done"])
            elif old:
                unit.runner = (old[0], proc_stat(old[0])[1])
                unit.started = time.monotonic()
            elif read_exit(unit.exit_file) is not None:
                unit.done = read_exit(unit.exit_file)   # it ended by itself while no keeper ran
        elif conf.get("fresh"):
            for name in FRESH:
                try:
                    os.unlink(os.path.join(unit.spec["session_dir"], name))
                except OSError:
                    pass
    guard_log(conf, "keeper_started", pid=me[0], recover=recover, uid=os.geteuid(),
              agents=[unit.spec["id"] for unit in units])

    def save(state: str = "running", exits=None) -> None:
        write_json(path, {
            "pid": me[0], "start": me[1], "state": state, "heartbeat": time.time(), "recover": recover,
            "exits": exits, "units": {str(unit.index): {
                "runner": unit.runner_pid(), "restarts": unit.restarts, "done": unit.done, "failed": unit.failed,
                "watchers": {name: (state_.get("process").pid if state_.get("process") else None)
                             for name, state_ in unit.watchers.items()}} for unit in units}})

    # "running" before anything starts: a keeper that dies at once is not taken for an earlier one
    # that finished.
    save()
    for unit in units:
        if unit.done is not None:
            continue
        for name in unit.spec["watchers"]:
            unit.start_watcher(name)
        if recover:
            agent_event(unit.spec, "harness_keeper_restarted",
                        "The harness's supervisor in this container had ended (a signal, probably from an agent's "
                        "own command); it was started again and its watchers with it.", restart=recover)
        if unit.runner is None:
            unit.restarts += 1 if recover else 0
            unit.start_runner(unit.restarts if recover else 0)
    save()

    beat = time.monotonic()
    while any(unit.done is None for unit in units):
        now = time.monotonic()
        for unit in units:
            if unit.done is not None:
                continue
            ended = unit.runner_ended()
            if ended is not None and unit.due is None:
                code, killed = ended
                lived = now - unit.started
                if killed and unit.failed is None:
                    unit.quick = unit.quick + 1 if lived < QUICK_SECONDS else 0
                    guard_log(conf, "runner_killed", agent=unit.spec["id"], exit=code, lived_seconds=round(lived, 1),
                              quick=unit.quick)
                    if unit.quick >= MAX_QUICK_DEATHS:
                        agent_event(unit.spec, "harness_runner_restart_limit",
                                    f"The harness's runner for this agent was ended by a signal {unit.quick} times in "
                                    f"a row within {QUICK_SECONDS:.0f} s of starting, so it was not started again.",
                                    restarts=unit.restarts, exitCode=code)
                        unit.finish(137)
                    else:
                        unit.restarts += 1
                        unit.runner = None
                        unit.due = now + min(2 ** unit.quick, 30) - 1
                else:
                    unit.finish(code if code >= 0 else 128 - code)
            if unit.done is None and unit.due is not None and now >= unit.due:
                unit.start_runner(unit.restarts)
            if unit.done is None:
                unit.check_watchers(now)
                unit.stop_if_failed(now)
        if now - beat >= HEARTBEAT_SECONDS:
            beat = now
            for unit in units:
                # A stopped process (SIGSTOP cannot be ignored) is continued.
                for pid in [unit.runner_pid()] + [state["process"].pid for state in unit.watchers.values()
                                                   if state.get("process") is not None]:
                    found = proc_stat(pid) if pid else None
                    if found and found[0] in ("T", "t"):
                        send(pid, signal.SIGCONT)
                        guard_log(conf, "continued_stopped_process", agent=unit.spec["id"], pid=pid)
            save()
        time.sleep(POLL_SECONDS)

    exits = {str(unit.index): unit.done for unit in units}
    save("finished", exits)
    guard_log(conf, "keeper_finished", exits=exits)
    return finish_output(conf, exits)


# ---- the other commands ------------------------------------------------------------------------------

def run_watcher(directory: str, key: str) -> None:
    conf = load_conf(directory)
    name, index = next(name for name in WATCHER_LABELS if name[0] == key[0]), key[1:]
    spec = next(unit for unit in conf["units"] if unit["index"] == int(index))["watchers"][name]
    # Only SIGKILL ends a watcher, except the limit watcher, which records its final times on SIGTERM.
    ignore_signals()
    path = os.path.join(directory, spec["module"])
    argv = [str(arg) for arg in spec["argv"]]
    if name == "episode":
        # The episode watcher is a script; it runs as the agent's user and finds its processes itself.
        sys.argv = [path, *argv]
        runpy.run_path(path, run_name="__main__")
        return
    # token_budget.py and limit_resume.py, finding the agent's processes as above (as_agent).
    loaded = importlib.util.spec_from_file_location(f"hsv_{name}", path)
    module = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(module)
    module.agent_processes = agent_processes
    sys.argv = [path, *argv]
    sys.exit(module.main(argv))


def status(directory: str, kick=None) -> dict:
    state = read_json(state_path(directory))
    if not isinstance(state, dict):
        return {"state": "none", "alive": False}
    found = proc_stat(int(state.get("pid") or 0))
    running = bool(found) and found[1] == state.get("start") and found[0] not in ("Z", "X")
    age = time.time() - float(state.get("heartbeat") or 0)
    report = {"state": state.get("state"), "alive": running, "heartbeat_age": round(age, 1),
              "exits": state.get("exits"), "units": state.get("units"), "stopped": state.get("stopped")}
    if running and state.get("state") == "running":
        if found[0] in ("T", "t"):
            send(int(state["pid"]), signal.SIGCONT)
            report["continued"] = True
        elif kick is not None and age > kick:
            send(int(state["pid"]), signal.SIGKILL)
            report["kicked"] = True
    elif state.get("state") == "running":
        report["state"] = "dead"
    return report


def stop(directory: str, reason: str) -> None:
    """End every agent cleanly: the keeper could not be kept running."""
    conf = load_conf(directory)
    state = read_json(state_path(directory)) or {}
    exits, active = {}, []
    for spec in conf["units"]:
        if spec["index"] not in conf["running"]:
            continue
        done = ((state.get("units") or {}).get(str(spec["index"])) or {}).get("done")
        exits[str(spec["index"])] = done if done is not None else 1
        if done is None:
            active.append(spec)
            mark_stopped(conf, spec, reason)
    # The keeper first, and at once the record that says why: agent.py asks for it when the keeper's
    # command returns, and must not start the keeper again.
    if alive(state.get("pid"), state.get("start")):
        send(int(state["pid"]), signal.SIGKILL)
    try:
        write_json(state_path(directory), {**state, "state": "finished", "exits": exits, "stopped": reason,
                                           "heartbeat": time.time()})
    except OSError:
        pass
    for spec in active:
        for pid in old_watchers(directory, spec["index"]):
            send(pid, signal.SIGKILL)
        for pid in tagged(f"HSV_UNIT={unit_tag(directory, spec['index'])}"):
            send(pid, signal.SIGKILL)
    for spec in active:
        stop_agent(cli_setting(conf, spec, "home_env"), spec["home"])
    guard_log(conf, "stopped", reason=reason, exits=exits)


def probe(directory: str) -> str:
    """Whether the keeper can run as this user for an agent that is another: it must be able to
    take the agent's identity, signal its processes and read its session logs."""
    conf = load_conf(directory)
    identity = conf.get("identity")
    if os.geteuid() != 0 or not identity or int(identity["uid"]) == 0:
        return "unprivileged"
    needed = {"CAP_KILL": 5, "CAP_SETGID": 6, "CAP_SETUID": 7, "CAP_DAC_OVERRIDE": 1}
    effective = 0
    try:
        with open("/proc/self/status") as stream:
            for line in stream:
                if line.startswith("CapEff:"):
                    effective = int(line.split()[1], 16)
    except (OSError, ValueError):
        return "unprivileged"
    if any(not effective >> bit & 1 for bit in needed.values()):
        return "unprivileged"
    try:
        done = subprocess.run([sys.executable, "-c", "pass"], preexec_fn=child_setup(identity))
    except (OSError, subprocess.SubprocessError):
        return "unprivileged"
    return "privileged" if done.returncode == 0 else "unprivileged"


def main(argv: list) -> int:
    directory = here()
    command, args = argv[0], argv[1:]
    # The two that run for the whole trial go by one letter, the watchers by a letter and the agent's
    # index (`hsvd w t0`): a command line with no word in it.
    command = {"k": "keep", "w": "watch"}.get(command, command)
    if command == "keep":
        recover = os.environ.get("HSV_RECOVER", "")
        return keep(directory, int(recover) if recover.isdigit() else 0)
    if command == "watch":
        run_watcher(directory, args[0])
    elif command == "cli-wait":
        ignore_signals()
        cli_wait(load_conf(directory), args[0])
    elif command == "recover-state":
        print("|".join(recover_state(*args)))
    elif command == "home-session":
        print(home_session(*args) or "")
    elif command == "pending-command":
        print(pending_command(*args[:2]) if len(args) >= 2 and args[1] else "")
    elif command == "status":
        kick = float(args[1]) if args[:1] == ["--kick"] else None
        print("hsv-status " + json.dumps(status(directory, kick), sort_keys=True))
    elif command == "stop":
        ignore_signals()
        stop(directory, args[0] if args else "the harness's supervisor could not be kept running")
    elif command == "probe":
        mode = probe(directory)
        identity = load_conf(directory).get("identity")
        if mode == "unprivileged" and os.geteuid() == 0 and identity:
            # The keeper will run as the agent's user: its state directory is that user's.
            os.chown(os.path.join(directory, "state"), int(identity["uid"]), int(identity["gid"]))
        print(f"hsv-mode {mode}")
    else:
        print(f"unknown command {command}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
