"""Harness protection (harness_guard.py, agent.py, run.py --harness-guard).

The end-to-end tests run an agent as Harbor does: agent.py's run() against an environment that
executes each command on this machine instead of in a container (/logs/agent and the guard directory
become temporary directories), the pinned Claude Code and Codex releases pointed at
tests/limit_mock_api.py. They then do to the harness what an agent's `pkill`, `killall` or
`kill -9 -1` did, by pid: never by name or pattern, since other trials run on this machine. They are
skipped without the pinned releases.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from cli_render import BASE_ENV, CASES, CLASSES, INSTRUCTION, ROOT, render
from test_limit_resume import (CLAUDE_BINARY, CODEX_BINARY, MARK, NOTE, Mock, codex_hook, continuation_text,
                               needs_claude, needs_codex, polyomino_continuation, user_texts)

sys.path.insert(0, str(ROOT))
import harness_guard  # noqa: E402
import limit_resume  # noqa: E402
import token_budget  # noqa: E402

GUARD = {"RELEASE_HARNESS_GUARD": "1"}
LIMIT = {"RELEASE_LIMIT_RESUME": "1", "RELEASE_LIMIT_WAIT_MAX": "172800", "RELEASE_AGENT_TIMEOUT": "3600"}
HOME_ENV = {"claude": "CLAUDE_CONFIG_DIR", "codex": "CODEX_HOME"}
HOME_DIR = {"claude": "claude-home", "codex": "codex-home"}
# Each mock response is 100 uncached input tokens and 10 output tokens: 10 + 23 + 2 weighted.
WEIGHTED_PER_RESPONSE = 35


def wait_for(condition, timeout: float = 60, what: str = "the condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        found = condition()
        if found:
            return found
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def kill(pids, sig=signal.SIGKILL) -> None:
    for pid in pids if isinstance(pids, (list, tuple, set)) else [pids]:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def children(pid: int) -> list[int]:
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            fields = Path(f"/proc/{name}/stat").read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        if int(fields[1]) == pid and fields[0] not in ("Z", "X"):
            found.append(int(name))
    return found


def comm(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return ""


# ---- the machine ---------------------------------------------------------------------------------------

class Local:
    """Stands in for the container: runs each command agent.py sends on this machine."""

    default_user = None

    def __init__(self, tmp: Path, base: dict, after=()):
        self.agent_dir = tmp / "agent"
        self.guard_dir = tmp / "hsv"
        self.base = base
        self.calls: list[dict] = []
        self.after = list(after)   # called with each command once it has run
        self.dead = False          # the host died: nothing more is run

    def patch(self, command: str) -> str:
        return (command.replace("/logs/agent", str(self.agent_dir)).replace("/usr/lib/hsv", str(self.guard_dir))
                .replace("rand()*30", "rand()*0").replace("JITTER = (30, 90)", "JITTER = (0, 0)"))

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        if self.dead:
            raise RuntimeError("the host is dead")
        self.calls.append({"command": command, "user": user, "env": dict(env or {})})
        process = await asyncio.create_subprocess_exec(
            "bash", "-c", self.patch(command), env={**self.base, **(env or {})}, stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        output, _ = await process.communicate()
        for hook in self.after:
            hook(command)
        return SimpleNamespace(return_code=process.returncode, stdout=output.decode(errors="replace"), stderr=None)


class Run:
    """One trial's agent phase, running in the background."""

    def __init__(self, tmp_path: Path, cli: str, plan: list, case: str = "ale-solo", env: dict | None = None,
                 guard: bool = True, revive_plan: dict | None = None, mock_dir: str = "mock", after=()):
        self.tmp, self.cli, self.case = tmp_path, cli, case
        (tmp_path / mock_dir).mkdir(parents=True)
        self.mock = Mock(tmp_path / mock_dir, plan)
        home = tmp_path / "home"
        home.mkdir(exist_ok=True)
        base = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(home),
                "HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9", "NO_PROXY": "127.0.0.1,localhost",
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{self.mock.port}",
                "RELEASE_TASK_CLI_HOOK": str(codex_hook(tmp_path / mock_dir, self.mock.port)) if cli == "codex" else "",
                "META_DUO_CLAUDE_API_RETRY_BASE_SLEEP": "2", "META_DUO_CODEX_API_RETRY_BASE_SLEEP": "2"}
        self.local = Local(tmp_path, base, after)
        spec = CASES[case]
        self.env = {
            **BASE_ENV, "RELEASE_TASK": spec["task"], "MULTIAGENT_N_AGENT": str(spec["agents"]),
            "MULTIAGENT_PROTOCOL": f"multiagent_protocols.{spec['protocol']}:MultiAgentProtocol",
            "RELEASE_CONTINUATION": (ROOT / "benchmarks" / spec["task"]
                                     / "continuation.txt").read_text().rstrip("\n"),
            **LIMIT, "RELEASE_CONTINUATIONS": "1", "MULTIAGENT_AGENT_EFFORT": "",
            "META_DUO_CONTAINER_CLAUDE_BINARY": str(CLAUDE_BINARY), "META_DUO_CONTAINER_CODEX_BINARY": str(CODEX_BINARY),
            "CODEX_API_KEY": "sk-mock", **(GUARD if guard else {}), **(env or {})}
        self.error: BaseException | None = None
        self.agent = None
        self.revive_plan = revive_plan
        self.thread = threading.Thread(target=self._main, daemon=True)
        self.started = time.time()
        self.thread.start()

    def _main(self) -> None:
        try:
            with mock.patch.dict(os.environ, self.env, clear=False):
                sys.modules.pop("agent", None)
                module = importlib.import_module("agent")
                self.module = module
                cls = getattr(module, CLASSES[self.cli])
                self.agent = cls(logs_dir=self.local.agent_dir,
                                 model_name="claude-sonnet-5-5" if self.cli == "claude" else "gpt-mock")

                async def go() -> None:
                    if self.revive_plan is None:
                        await self.agent.run(INSTRUCTION, self.local, SimpleNamespace())
                    else:
                        self.agent.revive_plan = self.revive_plan
                        await self.agent.resume(INSTRUCTION, self.local, SimpleNamespace())

                asyncio.run(go())
        except BaseException as error:  # noqa: BLE001 - the test reads it
            self.error = error

    # -- what runs

    @property
    def session(self) -> Path:
        found = sorted(self.local.agent_dir.glob("sessions-*"))
        return found[0] if found else self.local.agent_dir / f"{self.cli}-single-session"

    @property
    def home(self) -> Path:
        return self.session / HOME_DIR[self.cli]

    def keeper(self) -> int | None:
        state = harness_guard.read_json(str(self.local.guard_dir / "state" / "keeper.json")) or {}
        pid = state.get("pid")
        return pid if pid and harness_guard.alive(pid, state.get("start")) else None

    def runners(self, index: int = 0) -> list[int]:
        return harness_guard.tagged(f"HSV_UNIT={harness_guard.unit_tag(str(self.local.guard_dir), index)}")

    def watchers(self, name: str, index: int = 0) -> list[int]:
        return harness_guard.tagged(f"HSV_WATCH={harness_guard.watch_tag(str(self.local.guard_dir), name, index)}")

    def clis(self) -> list[int]:
        binary = CLAUDE_BINARY if self.cli == "claude" else CODEX_BINARY
        return harness_guard.cli_processes(HOME_ENV[self.cli], str(self.home), str(binary))

    def answered(self, count: int, timeout: float = 90) -> None:
        wait_for(lambda: len(self.mock.main()) >= count, timeout, f"{count} model requests")

    def finish(self, timeout: float = 240) -> "Run":
        self.thread.join(timeout)
        alive = self.thread.is_alive()
        self.close()
        assert not alive, "the run did not end"
        self.elapsed = time.time() - self.started
        return self

    def close(self) -> None:
        """Stop the mock and whatever of the trial still runs (by pid)."""
        self.mock.stop() if self.mock.process.poll() is None else None
        kill([pid for pid in [self.keeper()] if pid])
        for index in range(CASES[self.case]["agents"]):
            kill(self.runners(index))
            for name in harness_guard.WATCHER_LABELS:
                kill(self.watchers(name, index))
        for session in [*self.local.agent_dir.glob("sessions-*"), self.local.agent_dir / f"{self.cli}-single-session"]:
            kill(harness_guard.agent_processes(HOME_ENV[self.cli], str(session / HOME_DIR[self.cli])))

    # -- what happened

    @property
    def log(self) -> str:
        logs = sorted(self.local.agent_dir.glob("agent-*-output.log"))
        return logs[0].read_text(errors="replace") if logs else ""

    def events(self, event: str) -> list[dict]:
        found = []
        for line in self.log.splitlines():
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict) and entry.get("type") == "environment.feedback" \
                    and entry["data"].get("event") == event:
                found.append(entry["data"])
        return found

    def guard_events(self, event: str) -> list[dict]:
        path = self.local.agent_dir / "harness-guard.jsonl"
        lines = path.read_text().splitlines() if path.exists() else []
        return [entry for entry in map(json.loads, lines) if entry["event"] == event]

    def texts(self, request: int = -1) -> list[str]:
        return user_texts(self.mock.main()[request])

    def keep_calls(self) -> list[dict]:
        return [call for call in self.local.calls if call["command"].endswith("hsvd k")]

    def abandon(self) -> None:
        """The host dies: everything of the trial is gone at once, and nothing is started again."""
        self.local.dead = True
        self.close()
        self.thread.join(60)


@pytest.fixture
def started(tmp_path):
    """Start runs with started(cli, plan, ...); whatever they left running is stopped afterwards."""
    runs: list[Run] = []

    def start(cli: str, plan: list, **kwargs) -> Run:
        run = Run(tmp_path / f"run{len(runs)}", cli, plan, **kwargs)
        runs.append(run)
        return run

    yield start
    for run in runs:
        run.close()


def kill_notes(run: Run, command: str) -> list[str]:
    """The messages that told the agent a signal ended its session, each naming the command that was
    running (the CLI had not saved it, so the resumed session does not hold it)."""
    module = run.module
    found = [text for text in run.texts() if text.startswith(module.KILL_RESUME_NOTE)]
    for text in found:
        assert text.rstrip().endswith(module.KILL_RESUME_END) and module.KILL_RESUME_COMMAND in text, text
        assert command in text, text
    return found


def assert_each_message_once(run: Run, continuation: str, turns: int = 1) -> list[str]:
    """The last request carries the task prompt once and the continuation once per turn. Returns the
    user-written texts, with the run's own directory left out of them."""
    texts = run.texts()
    assert sum(MARK in text for text in texts) == 1, texts
    assert sum(text.strip() == continuation for text in texts) == turns, texts
    return [text.replace(str(run.tmp), "TMP") for text in texts]


# ---- what the harness sends ------------------------------------------------------------------------------

def test_off_sends_nothing_new():
    for cli in ("claude", "codex"):
        text = "\n".join(call["command"] for call in render("ale-team3", cli=cli, env_overrides=LIMIT))
        for word in ("hsv", "harness_stop", "harness_python", "runner_recover", "cli_ended_by_signal", "trap ':'"):
            assert word not in text, (cli, word)


def written(calls: list[dict], name: str) -> str:
    """The text a command writes to a file of that name (printf %s TEXT > PATH)."""
    for call in calls:
        words = shlex.split(call["command"])
        for index, word in enumerate(words):
            if word == ">" and words[index + 1].endswith(name) and words[index - 2] == "%s":
                return words[index - 1]
    raise AssertionError(f"no command writes {name}")


@pytest.mark.parametrize("cli", ["claude", "codex"])
@pytest.mark.parametrize("case", ["ale-solo", "ale-single", "ale-team3", "ale-team4"])
def test_on_commands(cli, case, tmp_path):
    env = {**LIMIT, **GUARD, "RELEASE_TOKEN_BUDGET": "2000000"}
    calls = render(case, cli=cli, env_overrides=env)
    for index, call in enumerate(calls):
        (tmp_path / f"{index}.sh").write_text(call["command"])
        assert subprocess.run(["bash", "-n", str(tmp_path / f"{index}.sh")], capture_output=True).returncode == 0
    # One long-lived command, with nothing in it for an agent's `pkill -f <pattern>` to match: no task text,
    # no model, no path under /logs, no "python", "bash", "agent", "claude", "codex" or "run".
    keep = [call for call in calls if call["command"].endswith("hsvd k")]
    assert [call["command"] for call in keep] == ["exec /usr/lib/hsv/hsvi /usr/lib/hsv/hsvd k"]
    single = CASES[case]["agents"] == 1 and case != "ale-solo"
    conf = json.loads(written(calls, "conf.json"))
    assert conf["layout"] == ("single" if single else "team") and conf["running"] == list(range(CASES[case]["agents"]))
    for unit in conf["units"]:
        assert set(unit["watchers"]) == {"token", "limit"}
        assert unit["watchers"]["token"]["argv"][:4] == [cli, unit["home"], "2000000", "weighted"]
        assert unit["watchers"]["limit"]["argv"][:5] == ["watch", cli, unit["session_dir"], HOME_ENV[cli], unit["home"]]
        assert unit["watchers"]["limit"]["argv"][-1].endswith("limit-watch-state.json")
    # The runner starts no watcher itself and no longer carries their sources; it traps the signals it can.
    runner = written(calls, "/runner")
    assert "token_budget_source" not in runner and "watch @cli@" not in runner and '" watch ' not in runner
    assert "trap ':' TERM HUP INT QUIT" in runner and "recover_runner" in runner and "cli_ended_by_signal" in runner
    assert subprocess.run(["bash", "-n"], input=runner, text=True, capture_output=True).returncode == 0
    launcher = written(calls, "/hsvr")
    assert INSTRUCTION.splitlines()[0] not in launcher and "unset HSV_UNIT HSV_RECOVER" in launcher
    assert subprocess.run(["bash", "-n"], input=launcher, text=True, capture_output=True).returncode == 0
    compile(written(calls, "/hsvd"), "hsvd", "exec")
    # The keeper's files are written as root; the stop marker and the time cap are checked afterwards.
    assert all(call["user"] == "root" for call in calls if "/usr/lib/hsv/hsvd " in call["command"]
               and "printf %s" in call["command"])
    assert "harness-stop.json" in calls[-2]["command"] and "time-cap-reached.json" in calls[-1]["command"]
    # Every command fits in one argument of execve (131072 bytes).
    assert max(len(call["command"].encode()) for call in calls) < 120000


def run_py(*argv: str):
    sys.path.insert(0, str(ROOT))
    import run
    return run.prepare(run.parser().parse_args([*argv, "--dry-run"]))


@pytest.mark.parametrize("cli", ["claude", "codex"])
def test_run_py_turns_it_on_by_default(cli):
    _, env, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", cli, "--model", "m")
    assert env["RELEASE_HARNESS_GUARD"] == "1"
    _, env, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", cli, "--model", "m", "--no-harness-guard")
    assert "RELEASE_HARNESS_GUARD" not in env


# ---- where a killed runner had got to --------------------------------------------------------------------

def write_events(path: Path, events: list, mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join((event if isinstance(event, str) else json.dumps(event)) + "\n" for event in events))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_recover_state_claude(tmp_path):
    session, log = tmp_path / "sessions-0-a1", tmp_path / "agent-0-a1-output.log"
    home = session / "claude-home"
    assert harness_guard.recover_state("claude", str(session), str(home), str(log)) == ["", "none", "", "0", ""]
    first = write_events(session / "claude-attempt-0.jsonl", [
        {"type": "system", "session_id": "s-a"}, {"type": "result", "is_error": False, "session_id": "s-a"}], 100)
    log.write_bytes(first.read_bytes())
    # The first turn's attempt ended by itself: complete, and already in the agent's log.
    assert harness_guard.recover_state("claude", str(session), str(home), str(log)) == [
        "0", "complete", "s-a", "0.000", str(first)]
    assert log.read_bytes() == first.read_bytes()
    # The second turn was cut short: its log is added to the agent's log, once.
    feedback = session / "claude-no-action-feedback-1.txt"
    feedback.write_text("go on")
    os.utime(feedback, (150, 150))
    second = write_events(session / "claude-attempt-1.jsonl", [{"type": "assistant", "session_id": "s-a"}, '{"partial'], 200)
    found = harness_guard.recover_state("claude", str(session), str(home), str(log))
    assert found == ["1", "interrupted", "s-a", "150.000", str(second)]
    assert log.read_bytes() == first.read_bytes() + second.read_bytes()
    harness_guard.recover_state("claude", str(session), str(home), str(log))
    assert log.read_bytes() == first.read_bytes() + second.read_bytes()
    # A turn that ended on an error is not complete either.
    write_events(session / "claude-attempt-1-recover-1.jsonl", [{"type": "result", "is_error": True, "session_id": "s-a"}], 300)
    assert harness_guard.recover_state("claude", str(session), str(home), str(log))[:2] == ["1", "interrupted"]


def test_recover_state_codex_reads_the_session_from_the_home(tmp_path):
    session, log = tmp_path / "codex-single-session", tmp_path / "agent-single-output.log"
    home = session / "codex-home"
    write_events(session / "codex-attempt-0.jsonl", ['{"partial'], 100)   # sed held the first events
    write_events(home / "sessions/2026/10/12/rollout-2026-10-12T10-00-00-t9.jsonl",
                 [{"type": "session_meta", "payload": {"id": "t9"}}])
    assert harness_guard.recover_state("codex", str(session), str(home), str(log))[:3] == ["0", "interrupted", "t9"]
    assert harness_guard.home_session("codex", str(home)) == "t9"
    write_events(session / "codex-attempt-0-recover-1.jsonl", [
        {"type": "thread.started", "thread_id": "t9"}, {"type": "turn.failed", "error": {"message": "x"}},
        {"type": "turn.completed", "usage": {}}], 200)
    assert harness_guard.recover_state("codex", str(session), str(home), str(log))[:3] == ["0", "complete", "t9"]


def test_pending_command(tmp_path):
    claude = write_events(tmp_path / "claude.jsonl", [
        {"type": "assistant", "message": {"content": [{"type": "tool_use", "id": "t1", "name": "Bash",
                                                       "input": {"command": "echo one"}}]}},
        {"type": "user", "message": {"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "one"}]}},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "now"}, {
            "type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "kill -9 -1"}}]}}, '{"partial'])
    assert harness_guard.pending_command("claude", str(claude)) == "kill -9 -1"
    with claude.open("a") as stream:
        stream.write(json.dumps({"type": "result", "is_error": False}) + "\n")
    assert harness_guard.pending_command("claude", str(claude)) == ""
    codex = write_events(tmp_path / "codex.jsonl", [
        {"type": "item.started", "item": {"id": "i1", "type": "command_execution", "command": "ls"}},
        {"type": "item.completed", "item": {"id": "i1", "type": "command_execution", "command": "ls"}},
        {"type": "item.started", "item": {"id": "i2", "type": "command_execution", "command": "pkill -f codex"}}])
    assert harness_guard.pending_command("codex", str(codex)) == "pkill -f codex"
    assert harness_guard.pending_command("codex", str(tmp_path / "missing.jsonl")) == ""


def test_limit_watcher_started_again_keeps_the_waits(tmp_path):
    """Claude Code's own waits are read from its attempt logs as they are written; a watcher that starts
    again reads them anew, where a finished wait has no length. With the keeper it keeps its count."""
    log = tmp_path / "claude-attempt-0.jsonl"
    log.write_text(json.dumps({"type": "system", "subtype": "api_retry", "retry_delay_ms": 60000}) + "\n")
    waits = limit_resume.CliWaits("claude", str(tmp_path))
    waits.poll(100.0)
    with log.open("a") as stream:
        stream.write(json.dumps({"type": "assistant", "message": {}}) + "\n")
    waits.poll(140.0)
    assert limit_resume.union_length(waits.intervals, 200.0) == 40.0
    again = limit_resume.CliWaits("claude", str(tmp_path))
    assert limit_resume.union_length(again.poll(200.0), 200.0) == 0.0   # read anew: lost
    kept = limit_resume.CliWaits("claude", str(tmp_path))
    kept.restore(json.loads(json.dumps(waits.state())))
    assert limit_resume.union_length(kept.poll(200.0), 200.0) == 40.0

    # The watcher itself: 30 s waited before it was killed; started again, it still counts them.
    session = tmp_path / "s"
    session.mkdir()
    state = session / harness_guard.LIMIT_STATE
    now = time.time()
    state.write_text(json.dumps({"offsets": {}, "intervals": [[now - 40, now - 10]]}))
    watcher = subprocess.Popen([sys.executable, str(ROOT / "limit_resume.py"), "watch", "claude", str(session),
                                "CLAUDE_CONFIG_DIR", str(session / "claude-home"), str(now - 50), "", "1000", "0.2",
                                str(state)])
    try:
        wait_for(lambda: (session / limit_resume.STATE).exists(), 10, "limit-wait.json")
    finally:
        watcher.kill()
    assert 29 <= json.loads((session / limit_resume.STATE).read_text())["waited_seconds"] <= 31


def test_revive_knows_the_guards_events_and_files():
    import revive
    assert "harness-stop.json" in revive.FINAL_MARKERS and "harness_guard.py" in revive.CODE_FILES
    assert {"harness_protection_failed", "harness_runner_restart_limit", "harness_cli_kill_limit"} <= set(revive.FINAL_EVENTS)


# ---- the keeper, with a runner that is only a script ------------------------------------------------------

TOY_RUNNER = r"""
trap ':' TERM HUP INT QUIT
session=@session@
trap 'runner_exit=$?; printf "%s\n" "$runner_exit" > "$session/runner-exit-status"' EXIT
printf '%s %s\n' "$$" "${runner_recover:-0}" >> "$session/starts"
grep SigIgn /proc/self/status > "$session/signals"
env > "$session/environment"
while [ ! -e "$session/finish" ]; do
  [ -e "$session/harness-stop.json" ] && exit 0
  command sleep 0.2
done
exit "$(cat "$session/finish")"
"""


class Toy:
    """A guard directory with a keeper whose runner is TOY_RUNNER and whose watcher is the real
    token-budget watcher on an empty CLI home."""

    def __init__(self, tmp: Path, agents: int = 1, layout: str = "team", edit=None):
        sys.modules.pop("agent", None)
        with mock.patch.dict(os.environ, {"RELEASE_CONTINUATION": "c", "RELEASE_TASK": "ale-bench", **GUARD,
                                          "RELEASE_TOKEN_BUDGET": "1000000"}, clear=False):
            module = importlib.import_module("agent")
            agent = module.CodexAgent(logs_dir=tmp, model_name="m")
            self.directory = tmp / "hsv"
            units, arguments = [], {}
            for index in range(agents):
                session = tmp / f"sessions-{index}-a{index + 1}"
                (session / "codex-home").mkdir(parents=True)
                unit = agent._guard_unit(index, f"a{index + 1}", agents, str(session), str(tmp / f"agent-{index}.log"), 0.0)
                unit["ignore_signals"] = ["INT", "QUIT"] if layout == "team" else []
                units.append(unit)
                arguments[index] = shlex.quote(str(session))
            self.units = units
            self.conf = {"cli": "codex", "label": "Codex", "home_env": "CODEX_HOME", "binary": "/nonexistent/codex",
                         "layout": layout, "log": str(tmp / "harness-guard.jsonl"), "identity": None, "fresh": True,
                         "units": units, "running": list(range(agents))}
            runner = TOY_RUNNER.replace("@session@", '"$1"')
            tools, trial = agent._guard_install(self.conf, runner, agent._guard_launcher(units, arguments, False), {})
        for command in (tools, trial):
            subprocess.run(["bash", "-c", command.replace("/usr/lib/hsv", str(self.directory))], check=True,
                           capture_output=True)
        if edit:
            edit(self)
        self.process = None
        self.tmp = tmp

    def keep(self, recover: int = 0, env: dict | None = None) -> subprocess.Popen:
        full = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(self.tmp), **(env or {}),
                **({"HSV_RECOVER": str(recover)} if recover else {})}
        self.process = subprocess.Popen([str(self.directory / "hsvi"), str(self.directory / "hsvd"), "k"], env=full,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        start_new_session=True)
        return self.process

    def session(self, index: int = 0) -> Path:
        return Path(self.units[index]["session_dir"])

    def runners(self, index: int = 0) -> list[int]:
        return harness_guard.tagged(f"HSV_UNIT={harness_guard.unit_tag(str(self.directory), index)}")

    def watchers(self, index: int = 0) -> list[int]:
        return harness_guard.tagged(f"HSV_WATCH={harness_guard.watch_tag(str(self.directory), 'token', index)}")

    def starts(self, index: int = 0) -> list[list[str]]:
        path = self.session(index) / "starts"
        return [line.split() for line in path.read_text().splitlines()] if path.exists() else []

    def events(self, index: int = 0) -> list[dict]:
        path = Path(self.units[index]["log"])
        lines = path.read_text().splitlines() if path.exists() else []
        return [json.loads(line)["data"] for line in lines if line.startswith('{"type"')]

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
        for index in range(len(self.units)):
            kill(self.runners(index))
            kill(self.watchers(index))


@pytest.fixture
def toy(tmp_path):
    made: list[Toy] = []

    def make(**kwargs) -> Toy:
        made.append(Toy(tmp_path, **kwargs))
        return made[-1]

    yield make
    for one in made:
        one.close()


def test_keeper_starts_again_what_is_killed(toy):
    box = toy(agents=2)
    keeper = box.keep(env={"ARCCTL_SESSION_ID_0": "x", "HSV_SOMETHING": "leak"})
    wait_for(lambda: box.starts(0) and box.starts(1) and box.watchers(0) and box.watchers(1), 20, "the keeper's start")
    runner, watcher = box.runners(0)[0], box.watchers(0)[0]
    assert comm(runner) == "hsvx" and comm(watcher) == "hsvi" and comm(keeper.pid) == "hsvi"
    for pid in (runner, watcher, keeper.pid):   # no word an agent's pattern is likely to hold
        line = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode().replace(str(box.directory), "/usr/lib/hsv")
        assert line.strip() in ("/usr/lib/hsv/hsvi /usr/lib/hsv/hsvd k", "/usr/lib/hsv/hsvi /usr/lib/hsv/hsvd w t0",
                                "/usr/lib/hsv/hsvx /usr/lib/hsv/hsvr"), line
    # The runner's environment has neither the keeper's variables nor, as a team's runner, another's.
    environment = (box.session(0) / "environment").read_text()
    assert "HSV_" not in environment and "SHLVL=1" in environment and "LC_CTYPE" not in environment
    # As the asynchronous command of a shell without job control: SIGINT and SIGQUIT ignored, nothing else.
    assert int((box.session(0) / "signals").read_text().split()[1], 16) == (1 << 1) | (1 << 2)

    # Signals that can be ignored are: the keeper, the runner and the watcher all stay.
    for pid in (keeper.pid, runner, watcher):
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
            os.kill(pid, sig)
    time.sleep(1.5)
    assert keeper.poll() is None and box.runners(0) == [runner] and box.watchers(0) == [watcher]

    # SIGKILL: the watcher is back within a second or two, the runner in recover mode; the other agent's stay.
    other = box.runners(1)[0]
    kill([watcher, runner])
    wait_for(lambda: box.watchers(0) and box.watchers(0) != [watcher], 10, "a new watcher")
    wait_for(lambda: len(box.starts(0)) == 2, 10, "a new runner")
    assert box.starts(0)[1][1] == "1" and box.runners(1) == [other] and len(box.starts(1)) == 1
    assert [event["event"] for event in box.events(0)] == ["harness_watcher_restarted"]
    assert box.events(0)[0]["watcher"] == "token" and box.events(0)[0]["exitCode"] == -9

    # A stopped process is continued.
    stopped = box.runners(0)[0]
    os.kill(stopped, signal.SIGSTOP)
    wait_for(lambda: harness_guard.proc_stat(stopped)[0] in ("T", "t"), 5, "the stop")
    wait_for(lambda: harness_guard.proc_stat(stopped)[0] not in ("T", "t"), 15, "the keeper to continue it")

    # The agents end: one with 0, one with 3. The keeper prints what the runner's shell printed, and succeeds.
    (box.session(0) / "finish").write_text("0")
    (box.session(1) / "finish").write_text("3")
    assert keeper.wait(timeout=30) == 0
    assert "Agent exits: a1=0, a2=3" in keeper.stdout.read()
    assert not box.watchers(0) and not box.watchers(1) and not box.runners(0)
    state = json.loads((box.directory / "state/keeper.json").read_text())
    assert state["state"] == "finished" and state["exits"] == {"0": 0, "1": 3}


def test_keeper_fails_when_every_agent_failed(toy):
    box = toy(layout="single")
    keeper = box.keep()
    wait_for(lambda: box.starts(0), 20, "the runner")
    assert int((box.session(0) / "signals").read_text().split()[1], 16) == 0   # the single runner: nothing ignored
    assert "SHLVL=1" in (box.session(0) / "environment").read_text()
    (box.session(0) / "finish").write_text("7")
    assert keeper.wait(timeout=30) == 1 and "Agent exits: a1=7" in keeper.stdout.read()


def faster(box: Toy) -> None:
    path = box.directory / "hsvd"
    path.write_text(path.read_text().replace("MAX_QUICK_DEATHS = 5", "MAX_QUICK_DEATHS = 3"))


def test_a_watcher_that_keeps_dying_ends_its_agent_cleanly(toy):
    """Protection that cannot be kept never fails silently: the agent's session is marked, its log says
    why, and its runner ends as after a spent budget."""
    def broken(box: Toy) -> None:
        faster(box)
        (box.directory / "mod-token.py").write_text("import sys\nsys.exit(3)\n")

    box = toy(edit=broken)
    keeper = box.keep()
    assert keeper.wait(timeout=60) == 0   # the toy runner exits 0 on the marker, as agent.py's does
    marker = json.loads((box.session(0) / "harness-stop.json").read_text())
    assert "token-budget watcher ended 3 times in a row" in marker["reason"]
    events = [event["event"] for event in box.events(0)]
    assert events == ["harness_watcher_restarted", "harness_watcher_restarted", "harness_protection_failed"]


def test_a_runner_that_keeps_being_killed_is_given_up(toy):
    box = toy(edit=faster)
    keeper = box.keep()
    for count in (1, 2, 3):
        wait_for(lambda: len(box.starts(0)) == count and box.runners(0), 30, f"runner start {count}")
        kill(box.runners(0))
    assert keeper.wait(timeout=30) == 1 and "Agent exits: a1=137" in keeper.stdout.read()
    assert box.events(0)[-1]["event"] == "harness_runner_restart_limit" and len(box.starts(0)) == 3


def test_a_new_keeper_adopts_the_runner_of_a_dead_one(toy):
    """A root agent's `kill -9` can end the keeper too. agent.py starts it again: runners that still run
    go on undisturbed, the watchers are started again, and the runner's exit code is still found."""
    box = toy(agents=2)
    first = box.keep()
    wait_for(lambda: box.starts(0) and box.starts(1) and box.watchers(0), 20, "the keeper's start")
    runner, watcher = box.runners(0)[0], box.watchers(0)[0]
    kill([first.pid, box.runners(1)[0]])   # the keeper and the second agent's runner
    first.wait()
    assert harness_guard.status(str(box.directory))["state"] == "dead"
    second = box.keep(recover=1)
    wait_for(lambda: len(box.starts(1)) == 2, 20, "the second agent's new runner")
    assert box.starts(1)[1][1] == "1"                       # in recover mode
    assert box.runners(0) == [runner] and len(box.starts(0)) == 1   # adopted, not started again
    wait_for(lambda: box.watchers(0) and box.watchers(0) != [watcher], 10, "new watchers")
    assert not harness_guard.alive(watcher)
    assert box.events(0)[-1]["event"] == "harness_keeper_restarted"
    status = harness_guard.status(str(box.directory))
    assert status["state"] == "running" and status["alive"]
    (box.session(0) / "finish").write_text("0")
    (box.session(1) / "finish").write_text("0")
    assert second.wait(timeout=30) == 0 and "Agent exits: a1=0, a2=0" in second.stdout.read()
    # A keeper started once more after that only reports the end.
    third = box.keep(recover=2)
    assert third.wait(timeout=30) == 0 and "Agent exits: a1=0, a2=0" in third.stdout.read()


def test_stop_ends_every_agent_cleanly(toy):
    box = toy()
    keeper = box.keep()
    wait_for(lambda: box.starts(0) and box.watchers(0), 20, "the keeper's start")
    done = subprocess.run([str(box.directory / "hsvi"), str(box.directory / "hsvd"), "stop", "a test"], timeout=60)
    assert done.returncode == 0 and keeper.wait(timeout=10) != 0
    assert not box.runners(0) and not box.watchers(0)
    assert json.loads((box.session(0) / "harness-stop.json").read_text())["reason"] == "a test"
    assert box.events(0)[-1]["event"] == "harness_protection_failed"
    assert harness_guard.status(str(box.directory))["state"] == "finished"


# ---- end to end -----------------------------------------------------------------------------------------

PROBE = ("echo PROBE ign=$(grep SigIgn /proc/self/status | cut -f2) blk=$(grep SigBlk /proc/self/status | cut -f2) "
         "shlvl=$SHLVL hsv=$(env | grep -c ^HSV_) names=$(env | cut -d= -f1 | sort | tr '\\n' ,) "
         "umask=$(umask) home=$HOME path=$PATH")


def probe_line(run: Run) -> str:
    """What the agent's first command printed (it is in the next request), without the run's directory."""
    found = re.search(r"PROBE ign=[0-9a-f]{16} blk=[0-9a-f]{16} shlvl=\d+ hsv=\d+ names=[A-Za-z0-9_,]+ umask=\d+ "
                      r"home=[^\s\"\\]+ path=[^\s\"\\]+", json.dumps(run.mock.main()[1]["body"]))
    assert found, "the probe's output is not in the request"
    return found.group(0).replace(str(run.tmp), "TMP")


@pytest.mark.parametrize("cli, case", [pytest.param("claude", "ale-solo", marks=needs_claude),
                                       pytest.param("codex", "ale-single", marks=needs_codex)])
def test_the_agent_sees_what_it_saw_without_the_keeper(started, cli, case):
    """The same run with and without the keeper: the model gets the same messages, and a command the
    agent runs has the same environment variables and the same signals ignored and blocked."""
    plan = [{"type": "tool", "command": PROBE}, {"type": "text", "text": "done-0"}, {"type": "text", "text": "done-1"}]
    budget = {"RELEASE_TOKEN_BUDGET": "100000"}
    old = started(cli, plan, case=case, guard=False, env=budget).finish()
    new = started(cli, plan, case=case, env=budget).finish()
    assert old.error is None and new.error is None, (old.error, new.error)
    continuation = continuation_text() if case == "ale-solo" else polyomino_continuation()
    assert assert_each_message_once(new, continuation) == assert_each_message_once(old, continuation)
    print("without the keeper:", probe_line(old), "\nwith the keeper:   ", probe_line(new))
    assert probe_line(new) == probe_line(old) and " hsv=0 " in probe_line(new)
    # With the keeper: its files, both watchers' records, the final times, and nothing left running.
    assert [event["event"] for event in map(json.loads, (new.local.agent_dir / "harness-guard.jsonl").read_text()
                                           .splitlines())][0] == "keeper_started"
    assert new.guard_events("keeper_finished")[0]["exits"] == {"0": 0}
    # (The token watcher polls every 5 s and is stopped with the agent, so its file may be behind.)
    usage = json.loads((new.session / "token-usage.json").read_text())
    spent = token_budget.weighted_tokens(token_budget.summed(token_budget.session_totals(cli, new.home)))
    assert usage["budget"] == 100000 and usage["weighted_tokens"] <= round(spent)
    assert json.loads((new.session / "limit-wait.json").read_text())["waiting"] is False
    assert (new.session / "runner-exit-status").read_text().strip() == "0"
    assert not new.runners() and not new.watchers("token") and not new.watchers("limit") and new.keeper() is None
    assert len(new.keep_calls()) == 1 and not new.events("harness_runner_restarted")


@needs_claude
def test_killed_watchers_are_started_again_and_the_budget_still_stops_the_agent(started):
    """`pkill -x python3` would kill both watchers and let the agent run past its budget. Now they are
    back at once, and the budget stops the agent."""
    budget = 5 * WEIGHTED_PER_RESPONSE
    plan = [{"type": "tool", "command": "sleep 6"}, *[{"type": "tool", "command": "sleep 3"}] * 20]
    run = started("claude", plan, env={"RELEASE_TOKEN_BUDGET": str(budget)})
    run.answered(1)
    token, limit = wait_for(lambda: (run.watchers("token"), run.watchers("limit")) if run.watchers("token")
                            and run.watchers("limit") else None, 30, "the watchers")
    kill(token + limit)
    wait_for(lambda: run.watchers("token") not in ([], token) and run.watchers("limit") not in ([], limit), 15,
             "new watchers")
    run.finish()
    assert run.error is None, run.error
    assert sorted(event["watcher"] for event in run.events("harness_watcher_restarted")) == ["limit", "token"]
    assert (run.session / "token-budget-exhausted.json").is_file() and len(run.events("token_budget_exhausted")) == 1
    # Five responses spend it; the watcher (every 5 s) may let one or two more through.
    assert 5 <= len(run.mock.main()) <= 8, [request["response"] for request in run.mock.main()]
    assert json.loads((run.session / "limit-wait.json").read_text())["active_seconds"] > 10


@needs_claude
def test_a_killed_runner_does_not_end_the_agents_turn(started):
    """`pkill -f ab.sh` can match the runner's command line and end the trial. A signal the
    runner cannot ignore now only brings a new runner, which waits for the agent's CLI: the agent's turn
    goes on, and the model sees nothing of it."""
    plan = [{"type": "tool", "command": "sleep 12"}, {"type": "tool", "command": "echo after"},
            {"type": "text", "text": "done-0"}, {"type": "text", "text": "done-1"}]
    run = started("claude", plan)
    run.answered(1)
    runner, cli = wait_for(lambda: (run.runners(), run.clis()) if run.runners() and run.clis() else None, 30, "the CLI")
    os.kill(runner[0], signal.SIGTERM)   # what pkill sends: ignored
    time.sleep(1)
    assert run.runners() == runner
    kill(runner[0])
    wait_for(lambda: run.runners() and run.runners()[0] != runner[0], 15, "a new runner")
    assert run.clis() == cli             # the CLI was not touched
    run.finish()
    assert run.error is None, run.error
    texts = assert_each_message_once(run, continuation_text())
    assert NOTE not in texts and not kill_notes(run, "")
    assert [request["response"] for request in run.mock.main()] == ["tool", "tool", "text", "text"]
    [event] = run.events("harness_runner_restarted")
    assert event["turn"] == 0 and event["turnState"] == "complete" and event["restart"] == 1
    assert not (run.session / "claude-attempt-0-recover-1.jsonl").exists()
    assert (run.session / "claude-attempt-1.jsonl").is_file() and len(run.keep_calls()) == 1
    # The first turn's log reached the agent's log once, though its runner died before adding it.
    assert sum('"type":"result"' in line for line in run.log.splitlines()) == 2


@pytest.mark.parametrize("cli, case", [pytest.param("claude", "ale-solo", marks=needs_claude),
                                       pytest.param("codex", "ale-single", marks=needs_codex)])
def test_kill_9_minus_1_of_the_agents_user_resumes_the_turn(started, cli, case):
    """A non-root agent's `kill -9 -1` ends everything of its user: the runner, the CLI, its sed. The
    keeper (root in the container) starts the runner again, which resumes the interrupted turn in the
    same session without sending its prompt again."""
    plan = [{"type": "tool", "command": "sleep 60"}, {"type": "tool", "command": "echo after"},
            {"type": "text", "text": "done-0"}, {"type": "text", "text": "done-1"}]
    run = started(cli, plan, case=case)
    run.answered(1)
    runner, clis = wait_for(lambda: (run.runners(), run.clis()) if run.runners() and run.clis() else None, 30, "the CLI")
    time.sleep(2)
    victims = runner + clis + [pid for pid in children(runner[0])] + harness_guard.agent_processes(
        HOME_ENV[cli], str(run.home))
    kill(victims)
    run.finish()
    assert run.error is None, run.error
    continuation = continuation_text() if case == "ale-solo" else polyomino_continuation()
    texts = assert_each_message_once(run, continuation)
    # One message more: the resumption's, which says a signal ended the session and names the command.
    assert len(kill_notes(run, "sleep 60")) == 1 and NOTE not in texts
    [event] = run.events("harness_runner_restarted")
    assert event["turnState"] == "interrupted" and event["turn"] == 0
    assert (run.session / f"{cli}-attempt-0-recover-1.jsonl").is_file()
    # One session throughout.
    sessions = "projects/*/*.jsonl" if cli == "claude" else "sessions/**/rollout-*.jsonl"
    assert len(list(run.home.glob(sessions))) == 1


@needs_codex
def test_a_killed_cli_is_resumed_by_its_runner(started):
    """`pkill -f codex` ends the agent's own CLI (and its teammates'). The runner takes it as an
    interrupted turn and resumes it; before, the agent ended and with it, for a single agent, the trial."""
    plan = [{"type": "tool", "command": "sleep 60"}, {"type": "text", "text": "done-0"}, {"type": "text", "text": "done-1"}]
    run = started("codex", plan)   # the team runner
    run.answered(1)
    runner, clis = wait_for(lambda: (run.runners(), run.clis()) if run.runners() and run.clis() else None, 30, "the CLI")
    time.sleep(2)
    kill(clis + harness_guard.agent_processes("CODEX_HOME", str(run.home)))
    run.finish()
    assert run.error is None, run.error
    assert run.runners() == [] and len(run.guard_events("runner_started")) == 1   # the runner itself stayed
    texts = assert_each_message_once(run, continuation_text())
    assert len(kill_notes(run, "sleep 60")) == 1 and NOTE not in texts
    [event] = run.events("harness_cli_resumed")
    assert event["exitCode"] >= 128 and event["attemptLog"] == "codex-attempt-0.jsonl"
    assert (run.session / "codex-attempt-0-api-retry-1.jsonl").is_file()
    assert not run.events("harness_runner_restarted")


@needs_claude
def test_kill_9_minus_1_as_root_brings_everything_back(started):
    """A root agent's `kill -9 -1` ends the keeper and the watchers too. agent.py starts the keeper
    again; it starts the watchers and, in recover mode, the runner, which resumes the turn. The budget
    is still enforced afterwards."""
    budget = 6 * WEIGHTED_PER_RESPONSE
    plan = [{"type": "tool", "command": "sleep 60"}, *[{"type": "tool", "command": "sleep 3"}] * 20]
    run = started("claude", plan, env={"RELEASE_TOKEN_BUDGET": str(budget)})
    run.answered(1)
    wait_for(lambda: run.runners() and run.clis() and run.watchers("token") and run.watchers("limit"), 30, "the run")
    time.sleep(2)
    keeper = run.keeper()
    kill([keeper, *run.runners(), *children(run.runners()[0]), *run.clis(), *run.watchers("token"), *run.watchers("limit"),
          *harness_guard.agent_processes("CLAUDE_CONFIG_DIR", str(run.home))])
    wait_for(lambda: run.keeper() not in (None, keeper), 30, "a new keeper")
    run.finish()
    assert run.error is None, run.error
    assert len(run.keep_calls()) == 2 and run.keep_calls()[1]["env"]["HSV_RECOVER"] == "1"
    assert len(run.events("harness_keeper_restarted")) == 1
    [event] = run.events("harness_runner_restarted")
    assert event["turnState"] == "interrupted"
    assert len(kill_notes(run, "sleep 60")) == 1 and sum(MARK in text for text in run.texts()) == 1
    assert (run.session / "token-budget-exhausted.json").is_file() and len(run.events("token_budget_exhausted")) == 1
    assert 6 <= len(run.mock.main()) <= 9, [request["response"] for request in run.mock.main()]


def test_a_keeper_that_cannot_be_kept_ends_the_trial_cleanly(started, tmp_path):
    """If the keeper cannot run at all, the agents are stopped, each log says why, and agent.py raises an
    error Harbor records before it grades the trial: never a trial that runs on unprotected."""
    def lose_the_shell(command: str) -> None:
        if "hsvd probe" in command:
            (tmp_path / "run0/hsv/hsvx").unlink()

    # No CLI is needed: the keeper never gets to start one.
    run = started("codex", [{"type": "text", "text": "done-0"}], after=[lose_the_shell],
                  env={"META_DUO_CONTAINER_CODEX_BINARY": "/bin/true"})
    run.finish(timeout=120)
    assert type(run.error).__name__ == "HarnessProtectionError", run.error
    assert "ended 5 times in a row" in str(run.error)
    assert len(run.keep_calls()) == 5 and "hsvd stop" in run.local.calls[-1]["command"]
    marker = json.loads((run.session / "harness-stop.json").read_text())
    assert "ended 5 times in a row" in marker["reason"]
    assert run.events("harness_protection_failed")[0]["reason"] == marker["reason"]


@needs_codex
def test_a_trial_stopped_by_hand_ends_cleanly(started):
    """`hsvd stop REASON`, run in the container, is how to end a running trial so that it is graded:
    killing an agent's CLI no longer ends the agent."""
    run = started("codex", [{"type": "tool", "command": "sleep 600"}])
    run.answered(1)
    wait_for(lambda: run.runners() and run.clis() and run.watchers("limit"), 30, "the run")
    done = subprocess.run([str(run.local.guard_dir / "hsvi"), str(run.local.guard_dir / "hsvd"), "stop", "by hand"],
                          timeout=60)
    assert done.returncode == 0
    run.finish(timeout=60)
    assert type(run.error).__name__ == "HarnessProtectionError" and "by hand" in str(run.error)
    assert json.loads((run.session / "harness-stop.json").read_text())["reason"] == "by hand"
    assert run.events("harness_protection_failed")[0]["reason"] == "by hand"
    assert not run.runners() and not run.clis() and not run.watchers("limit") and len(run.keep_calls()) == 1


@needs_codex
def test_revived_with_the_keeper(started, tmp_path):
    """A trial run with the keeper, its host dead (everything gone at once), revived with the keeper:
    the revival finds the runner the run kept, and the agent continues its session with the note."""
    import revive
    plan = [{"type": "tool", "command": "echo before-crash"}, {"type": "tool", "command": "sleep 600"}]
    first = started("codex", plan, case="ale-single", env={"RELEASE_TOKEN_BUDGET": "100000"})
    first.answered(2)
    wait_for(lambda: first.runners() and first.clis(), 30, "the CLI")
    time.sleep(3)
    first.abandon()
    tmp_path = first.tmp
    assert (first.local.agent_dir / "codex-single-runner.sh").is_file()
    layout, agents = revive.inspect_agents(tmp_path, "codex", 100000, "weighted")
    assert layout == "single" and agents[0]["finished_before"] is None and agents[0]["turn"] == 0
    for agent in agents:
        if agent["unlogged_attempt"] is not None:
            with agent["log"].open("ab") as stream:
                stream.write(agent["unlogged_attempt"].read_bytes())
    note = revive.DEFAULT_NOTE
    plan2 = {"revival": 1, "note": note, "layout": layout, "started": time.time(),
             "agents": [{"index": 0, "session_id": agents[0]["session_id"], "turn": 0}]}
    second = Run(tmp_path, "codex", [{"type": "text", "text": "done-0"}, {"type": "text", "text": "done-1"}],
                 case="ale-single", env={"RELEASE_TOKEN_BUDGET": "100000"}, revive_plan=plan2, mock_dir="mock2")
    try:
        second.finish()
    finally:
        second.close()
    assert second.error is None, second.error
    assert {entry["path"].rsplit("/", 1)[-1] for entry in plan2["checked"]} == {"codex-prompt.txt", "codex-single-runner.sh"}
    texts = second.texts()
    assert sum(MARK in text for text in texts) == 1 and texts.count(note) == 1
    assert sum(text.strip() == polyomino_continuation() for text in texts) == 1
    assert (second.local.agent_dir / "codex-single-runner-revive-1.sh").is_file()
    assert "revive_session_ids=([0]=" in second.agent.last_runner_script
    assert len(second.events("trial_revived")) == 1


def test_guard_install_too_long_for_execve_goes_out_in_pieces(tmp_path):
    """A team of 96 (with a small runner): the keeper's install command is over one argument of execve (as for the
    team of 32), so it goes out as commands that
    each fit, and they leave the same guard directory as the one command; a command that fits is sent as it is."""
    agents = 96
    sys.modules.pop("agent", None)
    with mock.patch.dict(os.environ, {"RELEASE_CONTINUATION": "c", "RELEASE_TASK": "ale-bench", **GUARD, **LIMIT,
                                      "RELEASE_TOKEN_BUDGET": "1000000", "RELEASE_TOKEN_BUDGET_METRIC": "weighted"},
                         clear=False):
        module = importlib.import_module("agent")
        agent = module.CodexAgent(logs_dir=tmp_path, model_name="m")
        units, arguments = [], {}
        for index in range(agents):
            session = tmp_path / f"sessions-{index}-a{index + 1}"
            (session / "codex-home").mkdir(parents=True)
            units.append(agent._guard_unit(index, f"a{index + 1}", agents, str(session),
                                           str(tmp_path / f"agent-{index}.log"), 0.0))
            arguments[index] = shlex.quote(str(session))
        conf = {"cli": "codex", "label": "Codex", "home_env": "CODEX_HOME", "binary": "/nonexistent/codex",
                "layout": "team", "log": str(tmp_path / "harness-guard.jsonl"), "identity": None, "fresh": True,
                "units": units, "running": list(range(agents))}
        runner = TOY_RUNNER.replace("@session@", '"$1"')
        tools, trial = agent._guard_install(conf, runner, agent._guard_launcher(units, arguments, False), {})
        commands = agent._guard_trial_commands(conf, trial)
    assert len(trial.encode()) > module.EXECVE_ARG_MAX
    assert len(commands) >= 3 and all(len(command.encode()) <= module.EXECVE_ARG_MAX for command in commands)
    assert commands[-1].endswith("/usr/lib/hsv/hsvi /usr/lib/hsv/hsvd probe")
    assert agent._guard_trial_commands(conf, "set -euo pipefail; true") == ["set -euo pipefail; true"]
    directory = tmp_path / "hsv"

    def install(sequence):
        subprocess.run(["rm", "-rf", str(directory)], check=True)
        for command in [tools, *sequence]:
            # Through stdin: the one command is too long for an argument.
            subprocess.run(["bash"], input=command.replace("/usr/lib/hsv", str(directory)), text=True, check=True,
                           capture_output=True)
        return {path.relative_to(directory).as_posix(): os.readlink(path) if path.is_symlink() else path.read_bytes()
                for path in sorted(directory.rglob("*")) if path.is_file() or path.is_symlink()
                if "state" not in path.relative_to(directory).parts}

    one, pieces = install([trial]), install(commands)
    assert one == pieces
    assert json.loads(pieces["conf.json"]) == json.loads(
        json.dumps(conf, indent=1, sort_keys=True).replace("/usr/lib/hsv", str(directory)))
