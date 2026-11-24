"""Reviving a trial whose host died (revive.py, agent.py's resume()).

The end-to-end tests run an agent the way test_limit_resume.py does (agent.py's commands run on
this machine, the pinned CLIs pointed at tests/limit_mock_api.py), with /logs/agent at a temporary
directory so the files a run writes are where its commands say. They kill the whole run partway
(the host dying: no process gets to clean up), then revive it with this code and check what the
model saw. They are skipped without the pinned releases.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest import mock

import pytest

from cli_render import INSTRUCTION, ROOT
from test_limit_resume import CLAUDE_BINARY, CODEX_BINARY, MARK, Mock, codex_hook, feedback, user_texts

sys.path.insert(0, str(ROOT))
import revive  # noqa: E402
import token_budget  # noqa: E402
from harbor.models.trial.paths import EnvironmentPaths  # noqa: E402

needs_claude = pytest.mark.skipif(not CLAUDE_BINARY.is_file(), reason="pinned Claude Code release missing")
needs_codex = pytest.mark.skipif(not CODEX_BINARY.is_file(), reason="pinned Codex release missing")
NOTE = revive.DEFAULT_NOTE
CLASSES = {"claude": "ClaudeCodeCollaborationAgent", "codex": "CodexCollaborationAgent"}
ON = {"RELEASE_LIMIT_RESUME": "1", "RELEASE_LIMIT_WAIT_MAX": "172800", "RELEASE_AGENT_TIMEOUT": "3600"}
# Each mock response is 100 uncached input tokens and 10 output tokens: 10 + 23 + 2 weighted.
WEIGHTED_PER_RESPONSE = 35
# For tests that only write a run's files: the setup checks the CLI binary is executable.
ANY_BINARY = {"META_DUO_CONTAINER_CLAUDE_BINARY": "/bin/sh", "META_DUO_CONTAINER_CODEX_BINARY": "/bin/sh"}


def continuation() -> str:
    return (ROOT / "benchmarks/ale-bench/continuation.txt").read_text().rstrip("\n")


def harness_env(protocol: str, extra: dict | None = None) -> dict:
    """The environment run.py gives Harbor for a solo ALE-Bench run (as the pilot's: one agent)."""
    return {"RELEASE_TASK": "ale-bench", "MULTIAGENT_N_AGENT": "1",
            "MULTIAGENT_PROTOCOL": f"multiagent_protocols.{protocol}:MultiAgentProtocol",
            "RELEASE_CONTINUATION": continuation(), "RELEASE_CONTINUATIONS": "1", "MULTIAGENT_AGENT_EFFORT": "",
            "CLAUDE_CODE_OAUTH_TOKEN": "fake-claude-oauth-token",
            "CODEX_API_KEY": "sk-mock", "RELEASE_TOKEN_BUDGET": "100000",
            "META_DUO_CONTAINER_CLAUDE_BINARY": str(CLAUDE_BINARY),
            "META_DUO_CONTAINER_CODEX_BINARY": str(CODEX_BINARY), **(extra or {})}


class FakeEnvironment(revive.RecordingEnvironment):
    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        self.envs.append(dict(env or {}))
        return await super().exec(command, user, env, cwd, timeout_sec)

    def __init__(self) -> None:
        super().__init__()
        self.envs: list[dict] = []


class Calls(list):
    """The commands an agent sent, with each one's environment."""

    def __init__(self, environment: FakeEnvironment):
        super().__init__(environment.calls)
        self.envs = environment.envs


def render(agent_dir: Path, cli: str, env: dict, revive_plan: dict | None = None) -> Calls:
    """The commands agent.py sends (run(), or resume() with REVIVE_PLAN), with /logs/agent at
    AGENT_DIR, after install()."""
    with mock.patch.dict(os.environ, env, clear=False), \
            mock.patch.object(EnvironmentPaths, "agent_dir", PurePosixPath(agent_dir)):
        sys.modules.pop("agent", None)
        module = importlib.import_module("agent")
        agent = getattr(module, CLASSES[cli])(logs_dir=agent_dir,
                                              model_name="claude-sonnet-5-5" if cli == "claude" else "gpt-mock")
        environment = FakeEnvironment()
        if revive_plan is None:
            asyncio.run(agent.run(INSTRUCTION, environment, SimpleNamespace()))
        else:
            agent.revive_plan = revive_plan
            asyncio.run(agent.resume(INSTRUCTION, environment, SimpleNamespace()))
    return Calls(environment)


class Host:
    """The machine: runs a trial's commands here and can die."""

    def __init__(self, tmp_path: Path, cli: str):
        self.tmp = tmp_path
        self.cli = cli
        self.home = tmp_path / "home"
        self.home.mkdir(exist_ok=True)

    def base(self, mock_api: Mock) -> dict:
        return {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(self.home),
                "HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
                "NO_PROXY": "127.0.0.1,localhost", "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{mock_api.port}",
                "RELEASE_TASK_CLI_HOOK": str(codex_hook(self.tmp, mock_api.port)) if self.cli == "codex" else "",
                "META_DUO_CLAUDE_API_RETRY_BASE_SLEEP": "2", "META_DUO_CODEX_API_RETRY_BASE_SLEEP": "2"}

    @staticmethod
    def command(text: str) -> str:
        return ("set -o pipefail; " + text.replace("rand()*30", "rand()*0")
                .replace("JITTER = (30, 90)", "JITTER = (0, 0)"))

    def run(self, calls: Calls, mock_api: Mock, extra: dict | None = None, timeout: float = 240) -> list:
        env = {**self.base(mock_api), **(extra or {})}
        return [subprocess.run(["bash", "-c", self.command(call)], env={**env, **call_env}, capture_output=True,
                               text=True, timeout=timeout) for call, call_env in zip(calls, calls.envs)]

    def run_until_killed(self, calls: Calls, mock_api: Mock, requests: int, extra: dict | None = None) -> None:
        """Set up, then run the agent until the mock has answered REQUESTS main requests and the
        last answer has had 3 s to start working; then kill everything at once, as a dying host does."""
        env = {**self.base(mock_api), **(extra or {})}
        # Harbor's trial.log records each command it runs.
        (self.tmp / "trial.log").write_text("".join(f"Running command: {call}\n" for call in calls))
        for call, call_env in zip(calls[:-1], calls.envs[:-1]):
            subprocess.run(["bash", "-c", self.command(call)], env={**env, **call_env}, check=True, timeout=60)
        process = subprocess.Popen(["bash", "-c", self.command(calls[-1])], env={**env, **calls.envs[-1]},
                                   start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.time() + 120
        while len(mock_api.main()) < requests:
            assert time.time() < deadline and process.poll() is None, "the run ended or stalled before the crash"
            time.sleep(0.2)
        time.sleep(3)
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        home_env = "CLAUDE_CONFIG_DIR" if self.cli == "claude" else "CODEX_HOME"
        for session in (self.tmp / "agent").glob("*session*"):
            token_budget.stop_agent(home_env, str(session / revive.HOME_DIRS[self.cli]))


def revive_plan(agent_dir: Path, cli: str, budget: int = 100000) -> tuple[dict, list[dict]]:
    layout, agents = revive.inspect_agents(agent_dir.parent, cli, budget, "weighted")
    plan = {"revival": 1, "note": NOTE, "layout": layout, "started": time.time(),
            "agents": [{"index": a["index"], "session_id": a["session_id"], "turn": a["turn"]}
                       for a in agents if a["finished_before"] is None]}
    for a in agents:
        if a["unlogged_attempt"] is not None:
            data = a["unlogged_attempt"].read_bytes()
            with a["log"].open("ab") as stream:
                stream.write(data if data.endswith(b"\n") else data + b"\n")
    return plan, agents


def session_ids(cli: str, session: Path) -> set:
    return {revive.session_id_of(cli, [path]) for path in revive.attempt_logs(session)} - {None}


def weighted(session: Path, cli: str = "claude") -> int:
    """The agent's weighted tokens, read from its CLI's session logs as the budget watcher reads them."""
    usage = token_budget.summed(token_budget.session_totals(cli, session / revive.HOME_DIRS[cli]))
    return round(token_budget.weighted_tokens(usage))


# ---- end to end -----------------------------------------------------------------------------------

@needs_claude
def test_claude_team_runner_revived_after_dying_mid_tool(tmp_path):
    """The pilot's layout (run(), one agent, antiherding_matched) with the old runner: the host dies
    while the agent's second tool call runs. Revived with this code, the agent continues the same
    session: the model sees the task prompt once, the work before the crash, the note once, and
    then the continuation once; the agent's log keeps everything from before."""
    agent_dir = tmp_path / "agent"
    host = Host(tmp_path, "claude")
    original = render(agent_dir, "claude", harness_env("antiherding_matched"))
    (tmp_path / "m1").mkdir()
    first = Mock(tmp_path / "m1", [{"type": "tool", "command": "echo before-crash"},
                                   {"type": "tool", "command": "sleep 600"}])
    try:
        host.run_until_killed(original, first, requests=2)
    finally:
        first.stop()
    session = agent_dir / "sessions-0-a1"
    before_ids = session_ids("claude", session)
    plan, agents = revive_plan(agent_dir, "claude")
    assert plan["layout"] == "team" and plan["agents"] == [
        {"index": 0, "session_id": next(iter(before_ids)), "turn": 0}]
    usage_before = agents[0]["weighted_tokens_before"]
    assert usage_before == weighted(session) == 2 * WEIGHTED_PER_RESPONSE
    log_before = (agent_dir / "agent-0-a1-output.log").read_bytes()

    revived = render(agent_dir, "claude", harness_env("antiherding_matched", ON), plan)
    assert len(revived) == 3  # write the revived runner, run it, the time-cap check
    assert "run-agent-revive-1.sh" in revived[0] and "rm -rf" not in revived[0]
    assert ">>" + str(agent_dir / "agent-0-a1-output.log") in revived[1]
    (tmp_path / "m2").mkdir()
    second = Mock(tmp_path / "m2", [{"type": "tool", "command": "echo after-crash"}, {"type": "text", "text": "done-0"},
                                    {"type": "text", "text": "done-1"}])
    try:
        results = host.run(revived, second)
    finally:
        second.stop()
    assert results[1].returncode == 0, results[1].stdout[-3000:] + results[1].stderr[-3000:]
    requests = second.main()
    texts = user_texts(requests[0])
    assert sum(MARK in text for text in texts) == 1 and texts[-1] == NOTE and texts.count(NOTE) == 1
    assert "before-crash" in json.dumps(requests[0]["body"])  # the work before the crash is in context
    last = user_texts(requests[-1])
    assert sum(MARK in text for text in last) == 1 and last.count(NOTE) == 1
    assert sum(text.strip() == continuation() for text in last) == 1
    # One session throughout, the turn it was in resumed, then the next turn.
    assert session_ids("claude", session) == before_ids
    assert (session / "claude-attempt-0-revive-1.jsonl").is_file() and (session / "claude-attempt-1.jsonl").is_file()
    assert not (session / "claude-attempt-2.jsonl").exists()
    # The log keeps the run's events, then the revival's.
    log = (agent_dir / "agent-0-a1-output.log").read_bytes()
    assert log.startswith(log_before) and len(feedback(SimpleNamespace(log=log.decode()), "trial_revived")) == 1
    # Tokens continue from the run's: every response counted once, none twice.
    answered = sum(1 for r in requests if r["response"] in ("tool", "text"))
    assert weighted(session) == usage_before + answered * WEIGHTED_PER_RESPONSE


@needs_claude
def test_claude_revived_while_retrying_an_api_error(tmp_path):
    """As the pilot runs died: the CLI ended on an API error and the runner was waiting to retry
    when the host died (the attempt's log is in the agent's log). The revival resumes the session
    with the note and adds nothing else."""
    agent_dir = tmp_path / "agent"
    host = Host(tmp_path, "claude")
    original = render(agent_dir, "claude", harness_env("antiherding_matched"))
    (tmp_path / "m1").mkdir()
    first = Mock(tmp_path / "m1", [{"type": "tool", "command": "echo before-crash"},
                                   {"type": "limit", "seconds": 3600, "status": 500}])
    try:
        host.run_until_killed(original, first, requests=2, extra={
            "CLAUDE_CODE_MAX_RETRIES": "0", "META_DUO_CLAUDE_API_RETRY_BASE_SLEEP": "600"})
    finally:
        first.stop()
    assert feedback(SimpleNamespace(log=(agent_dir / "agent-0-a1-output.log").read_text()), "claude_api_retry")
    plan, agents = revive_plan(agent_dir, "claude")
    assert agents[0]["unlogged_attempt"] is None and plan["agents"][0]["turn"] == 0
    revived = render(agent_dir, "claude", harness_env("antiherding_matched", ON), plan)
    (tmp_path / "m2").mkdir()
    second = Mock(tmp_path / "m2", [{"type": "text", "text": "done-0"}, {"type": "text", "text": "done-1"}])
    try:
        results = host.run(revived, second)
    finally:
        second.stop()
    assert results[1].returncode == 0, results[1].stdout[-3000:]
    texts = user_texts(second.main()[0])
    assert sum(MARK in text for text in texts) == 1 and texts.count(NOTE) == 1 and texts[-1] == NOTE
    last = user_texts(second.main()[-1])
    assert last.count(NOTE) == 1 and sum(text.strip() == continuation() for text in last) == 1


@needs_claude
def test_revived_agent_stops_at_the_budget_counting_tokens_used_before(tmp_path):
    """The token budget counts the run's tokens from before the crash: with room for two more
    responses, the revived agent is stopped after about two, not after the budget's worth."""
    agent_dir = tmp_path / "agent"
    host = Host(tmp_path, "claude")
    budget = 7 * WEIGHTED_PER_RESPONSE
    original = render(agent_dir, "claude", harness_env("antiherding_matched", {"RELEASE_TOKEN_BUDGET": str(budget)}))
    (tmp_path / "m1").mkdir()
    first = Mock(tmp_path / "m1", [*[{"type": "tool", "command": "echo work"}] * 4,
                                   {"type": "tool", "command": "sleep 600"}])
    try:
        host.run_until_killed(original, first, requests=5)
    finally:
        first.stop()
    plan, agents = revive_plan(agent_dir, "claude", budget)
    assert agents[0]["weighted_tokens_before"] == 5 * WEIGHTED_PER_RESPONSE
    revived = render(agent_dir, "claude", harness_env("antiherding_matched", {**ON, "RELEASE_TOKEN_BUDGET": str(budget)}),
                     plan)
    (tmp_path / "m2").mkdir()
    second = Mock(tmp_path / "m2", [{"type": "tool", "command": "sleep 4"}] * 12)
    try:
        results = host.run(revived, second)
    finally:
        second.stop()
    assert results[1].returncode == 0, results[1].stdout[-3000:]
    session = agent_dir / "sessions-0-a1"
    assert (session / "token-budget-exhausted.json").is_file()
    assert feedback(SimpleNamespace(log=(agent_dir / "agent-0-a1-output.log").read_text()), "token_budget_exhausted")
    # Two more responses spend it; the watcher (every 5 s) may let one or two more through.
    assert 2 <= len(second.main()) <= 4, [r["response"] for r in second.main()]


@needs_codex
def test_codex_single_runner_revived(tmp_path):
    """Codex on the single-agent runner (run_single()): `exec resume` with the note."""
    agent_dir = tmp_path / "agent"
    host = Host(tmp_path, "codex")
    original = render(agent_dir, "codex", harness_env("antiherding"))
    (tmp_path / "m1").mkdir()
    first = Mock(tmp_path / "m1", [{"type": "tool", "command": "echo before-crash"},
                                   {"type": "tool", "command": "sleep 600"}])
    try:
        host.run_until_killed(original, first, requests=2)
    finally:
        first.stop()
    session = agent_dir / "codex-single-session"
    # The old runner's sed held Codex's first events when the host died: the session is found in
    # Codex's own rollout.
    thread = revive.home_session_id("codex", session / "codex-home")
    plan, agents = revive_plan(agent_dir, "codex")
    assert plan["layout"] == "single" and plan["agents"] == [{"index": 0, "session_id": thread, "turn": 0}]
    revived = render(agent_dir, "codex", harness_env("antiherding", ON), plan)
    assert len(revived) == 2 and "rm -rf" not in revived[0]  # the runner, then the time-cap check
    (tmp_path / "m2").mkdir()
    second = Mock(tmp_path / "m2", [{"type": "text", "text": "done-0"}, {"type": "text", "text": "done-1"}])
    try:
        results = host.run(revived, second)
    finally:
        second.stop()
    assert results[0].returncode == 0, results[0].stdout[-3000:]
    texts = user_texts(second.main()[0])
    assert sum(MARK in text for text in texts) == 1 and texts.count(NOTE) == 1
    last = user_texts(second.main()[-1])
    assert last.count(NOTE) == 1 and sum(text.strip() == continuation() for text in last) == 1
    assert session_ids("codex", session) == {thread}
    [rollout] = (session / "codex-home").glob("sessions/**/rollout-*.jsonl")
    import limit_resume
    recorded = list(limit_resume.user_texts("codex", str(rollout), 0))
    assert sum(MARK in text for text in recorded) == 1 and recorded.count(NOTE) == 1


# ---- the checks before anything reaches the container ---------------------------------------------

def written_run(tmp_path: Path, cli: str = "claude", protocol: str = "antiherding_matched",
                extra: dict | None = None) -> Path:
    """A run's files as its setup and first turn left them (without running a CLI)."""
    agent_dir = tmp_path / "agent"
    calls = render(agent_dir, cli, harness_env(protocol, {**ANY_BINARY, **(extra or {})}))
    subprocess.run(["bash", "-c", calls[0]], check=True)  # the setup
    session = agent_dir / ("sessions-0-a1" if protocol == "antiherding_matched" else f"{cli}-single-session")
    if protocol == "antiherding_matched":
        intro = "You are solving this task alone. Follow the solo work protocol exactly.\n\n"
        shared = (agent_dir / "multiagent-collab/shared-prompt.md").read_text()
        instruction = (session / "task-instruction.txt").read_text()
        (session / f"{cli}-prompt.txt").write_text(f"{intro}{shared}\n\nTask instruction:\n{instruction}\n")
    return agent_dir


def plan_for(layout: str) -> dict:
    return {"revival": 1, "note": NOTE, "layout": layout, "started": 0.0,
            "agents": [{"index": 0, "session_id": "s-1", "turn": 3}]}


def test_resume_renders_after_checking_the_runs_files(tmp_path):
    agent_dir = written_run(tmp_path)
    plan = plan_for("team")
    calls = render(agent_dir, "claude", harness_env("antiherding_matched", ON), plan)
    checked = {entry["path"].rsplit("/", 1)[-1] for entry in plan["checked"]}
    assert checked == {"protocol.json", "shared-prompt.md", "task-instruction.txt", "claude-prompt.txt", "run-agent.sh"}
    runner = revive.rendered_runner("team", calls)
    assert "revive_session_ids=([0]=s-1)" in runner and "revive_turns=([0]=3)" in runner
    assert 'no_action_attempt=$((revive_turn + 1))' in runner and ': > "$log_file"' not in runner
    assert subprocess.run(["bash", "-n"], input=runner, text=True).returncode == 0


def test_resume_refuses_another_task_instruction(tmp_path):
    agent_dir = written_run(tmp_path)
    (agent_dir / "sessions-0-a1/task-instruction.txt").write_text(INSTRUCTION + " (changed)")
    with pytest.raises(RuntimeError, match="task-instruction.txt differs"):
        render(agent_dir, "claude", harness_env("antiherding_matched", ON), plan_for("team"))


def test_resume_refuses_another_continuation(tmp_path):
    agent_dir = written_run(tmp_path)
    with pytest.raises(RuntimeError, match="continuation prompt differs"):
        render(agent_dir, "claude", harness_env("antiherding_matched", {**ON, "RELEASE_CONTINUATION": "Go on."}),
               plan_for("team"))


def test_single_runner_resume_refuses_another_prompt(tmp_path):
    agent_dir = tmp_path / "agent"
    calls = render(agent_dir, "codex", harness_env("antiherding", ANY_BINARY))
    subprocess.run(["bash", "-c", calls[0]], check=True)  # the setup writes the prompt file
    (tmp_path / "trial.log").write_text("Running command: " + calls[1])
    plan = plan_for("single")
    revived = render(agent_dir, "codex", harness_env("antiherding", ON), plan)
    assert 'revive_session_ids=([0]=s-1)' in revived[0] and "codex-attempt-$revive_turn-revive-1.jsonl" in revived[0]
    (agent_dir / "codex-single-session/codex-prompt.txt").write_text("another prompt")
    with pytest.raises(RuntimeError, match="codex-prompt.txt differs"):
        render(agent_dir, "codex", harness_env("antiherding", ON), plan_for("single"))


def test_run_renders_as_before_without_a_revive_plan(tmp_path):
    """run() and run_single() send what they always did (the full render comparison is in the
    integration record); a resume() without a plan is Harbor's own, unsupported."""
    for protocol, cli in (("antiherding_matched", "claude"), ("antiherding", "codex")):
        text = "\n".join(render(tmp_path / "agent", cli, harness_env(protocol)))
        assert "revive_" not in text and "trial_revived" not in text and ': > "$log_file"' in text
    with mock.patch.dict(os.environ, harness_env("antiherding"), clear=False):
        sys.modules.pop("agent", None)
        agent = importlib.import_module("agent").CodexCollaborationAgent(logs_dir=tmp_path, model_name="m")
        with pytest.raises(NotImplementedError):
            asyncio.run(agent.resume(INSTRUCTION, FakeEnvironment(), SimpleNamespace()))


# ---- reading a trial --------------------------------------------------------------------------------

def touch(path: Path, text: str = "", mtime: float | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_turn_session_and_unlogged_attempt(tmp_path):
    session = tmp_path / "agent/sessions-0-a1"
    log = touch(tmp_path / "agent/agent-0-a1-output.log", '{"session_id":"s-a"}\n{"x":1}\n')
    touch(session / "claude-attempt-0.jsonl", '{"session_id":"s-a"}\n{"x":1}\n', 100)
    touch(session / "claude-attempt-1-api-retry-1.jsonl", '{"session_id":"s-a"}\n', 200)
    touch(session / "claude-attempt-2.jsonl", '{"session_id":"s-b"}\n{"partial', 300)
    touch(session / "claude-no-action-feedback-2.txt", "go on")
    logs = revive.attempt_logs(session)
    assert [p.name for p in logs] == ["claude-attempt-0.jsonl", "claude-attempt-1-api-retry-1.jsonl",
                                      "claude-attempt-2.jsonl"]
    assert revive.turn_of(logs) == 2 and revive.session_id_of("claude", logs) == "s-b"
    assert revive.unlogged_attempt(log, logs) == logs[-1]
    assert revive.unlogged_attempt(log, logs[:1]) is None
    layout, found = revive.sessions(tmp_path, "claude")
    assert layout == "team" and found[0]["agent_id"] == "a1" and found[0]["log"] == log


def test_finished_agents_are_not_resumed(tmp_path):
    session = tmp_path / "s"
    log = touch(tmp_path / "log", '{"type":"environment.feedback","data":{"event":"claude_api_retry"}}\n')
    session.mkdir()
    assert revive.finished(session, log) is None
    touch(log, '{"type":"environment.feedback","data":{"event":"agent_gave_up"}}\n')
    assert revive.finished(session, log) == "agent_gave_up"
    touch(log, "")
    touch(session / "token-budget-exhausted.json", "{}")
    assert revive.finished(session, log) == "token-budget-exhausted.json"


def test_working_time_before_a_revival():
    assert revive.worked_before([], 1000.0, 4600.0) == 3600.0
    # A second revival: the first counted 3600 s, then the agent worked 600 s after it.
    first = {"worked_seconds_before": 3600.0, "revived_at_epoch": 10000.0}
    assert revive.worked_before([first], 1000.0, 10600.0) == 4200.0
    assert revive.worked_before([first], 1000.0, 9000.0) == 3600.0  # no work since


def container(service: str, depends: str = "", health: bool = False, mounts=()) -> dict:
    labels = {"com.docker.compose.service": service}
    if depends:
        labels["com.docker.compose.depends_on"] = depends
    return {"Name": f"/p-{service}-1", "Config": {"Labels": labels}, "State": {"Running": False},
            "Mounts": [{"Type": "bind", "Source": source} for source in mounts]}


def test_containers_start_after_what_they_depend_on(tmp_path):
    sidecar = "harbor-docker-egress-control-sidecar"
    containers = [container("main", f"{sidecar}:service_healthy:false,scorer:service_healthy:false"),
                  container("scorer", f"{sidecar}:service_healthy:false", mounts=[str(tmp_path / "gone"), str(tmp_path)]),
                  container(sidecar)]
    assert [revive.service_of(c) for c in revive.start_order(containers)] == [sidecar, "scorer", "main"]
    assert revive.depends_on(containers[0]) == {sidecar: "service_healthy", "scorer": "service_healthy"}
    assert revive.missing_bind_sources(containers) == [str(tmp_path / "gone")]
    with pytest.raises(ValueError, match="circular"):
        revive.start_order([container("a", "b:service_started:false"), container("b", "a:service_started:false")])


# ---- the run's settings and the job's record -----------------------------------------------------

PILOT = ["ale-bench", "--problem", "ahc014", "--ale-no-private-lane", "--ale-judge-cpus", "8,10,12,14", "--cli",
         "claude", "--claude-binary", ".cache/claude-2.1.289/claude", "--model", "claude-opus-5-5", "--effort", "high",
         "--protocol", "antiherding_matched", "--agents", "1", "--trials", "1", "--token-budget", "2000000",
         "--timeout", "86400", "--name", "pilot2m-ahc014-opus"]


def test_harness_settings_are_run_pys(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake")
    args, cmd, env = revive.harness(PILOT, "pilot2m-ahc014-opus")
    assert env["RELEASE_TOKEN_BUDGET"] == "2000000" and env["MULTIAGENT_AGENT_EFFORT"] == "high"
    assert env["RELEASE_TASK"] == "ale-bench" and env["RELEASE_LIMIT_RESUME"] == "1"
    assert env["RELEASE_CONTINUATION"] == (ROOT / "benchmarks/ale-bench/continuation.txt").read_text().rstrip("\n")
    assert env["MULTIAGENT_PROTOCOL"] == "multiagent_protocols.antiherding_matched:MultiAgentProtocol"
    assert not (ROOT / "jobs/pilot2m-ahc014-opus-revive-probe").exists()
    config = SimpleNamespace(agent=SimpleNamespace(import_path="agent:ClaudeCodeCollaborationAgent",
                                                   model_name="claude-opus-5-5"), agent_timeout_multiplier=8.0)
    revive.check_settings(config, args, cmd, 10800.0)  # the run's own Harbor timeout, before --limit-resume
    with pytest.raises(ValueError, match="--timeout"):
        revive.check_settings(config, revive.harness([*PILOT[:-4], "--timeout", "3600", *PILOT[-2:]],
                                                     "pilot2m-ahc014-opus")[0], cmd, 10800.0)
    with pytest.raises(ValueError, match="model"):
        revive.check_settings(SimpleNamespace(agent=SimpleNamespace(import_path=config.agent.import_path,
                                                                    model_name="claude-sonnet-5-5"),
                                              agent_timeout_multiplier=8.0), args, cmd, 10800.0)
    with pytest.raises(ValueError, match="not this trial's job"):
        revive.harness(PILOT, "pilot2m-ahc038-opus")


def test_finish_job_writes_harbors_job_result(tmp_path):
    from harbor.models.job.result import JobResult, JobStats
    from harbor.models.trial.config import TaskConfig, TrialConfig
    from harbor.models.trial.result import AgentInfo, ModelInfo, TrialResult
    from harbor.models.verifier.result import VerifierResult

    job = tmp_path / "job"
    trial = job / "t__1"
    trial.mkdir(parents=True)
    (job / "config.json").write_text(json.dumps({"job_name": "job", "jobs_dir": str(tmp_path)}))
    started = JobResult(id="00645a98-67b6-4388-ae09-c3578e827579", started_at="2026-10-10T06:21:44",
                        n_total_trials=1, stats=JobStats(n_running_trials=1))
    (job / "result.json").write_text(started.model_dump_json(indent=4))
    config = TrialConfig(task=TaskConfig(path=tmp_path / "task"), trial_name="t__1", trials_dir=job)
    result = TrialResult(task_name="t", trial_name="t__1", trial_uri=trial.as_uri(),
                         task_id=config.task.get_task_id(), task_checksum="x", config=config,
                         agent_info=AgentInfo(name="a", version="1", model_info=ModelInfo(name="m")),
                         verifier_result=VerifierResult(rewards={"reward": 1650.0}), source="src")
    (trial / "result.json").write_text(result.model_dump_json(indent=4))
    asyncio.run(revive.finish_job(job))
    done = JobResult.model_validate_json((job / "result.json").read_text())
    assert done.finished_at is not None and done.started_at == started.started_at
    assert done.stats.n_completed_trials == 1 and done.stats.n_running_trials == 0
    assert done.stats.evals["a__m__src"].metrics == [{"mean": 1650.0}]


# ---- a trial Harbor ended on an infrastructure failure; a trial whose containers are gone ---------

# Three git-to-zig trials whose host ran out of process slots, as their result.json
# recorded them (the compose commands shortened).
R1 = "can't start new thread"
R2 = ("Docker compose command failed for environment git-to-zig. Command: docker compose --project-name "
      "frontier-swe-git-to-zig__jetjbyx__env -f /tmp/x.json exec --no-TTY harbor-docker-egress-control-sidecar "
      "network-policy allow. Return code: 125. Stdout: unknown flag: --project-name\n. Stderr: None. ")
R3 = ("Docker compose command failed for environment git-to-zig. Command: docker compose --project-name "
      "frontier-swe-git-to-zig__he7mjj6__env -f /tmp/x.json exec --no-TTY harbor-docker-egress-control-sidecar "
      "network-policy allow. Return code: 1. Stdout: fork/exec /usr/libexec/docker/cli-plugins/docker-compose: "
      "resource temporarily unavailable\n. Stderr: None. ")


def harbor_result(kind: str | None, message: str = "", rewards: dict | None = None) -> dict:
    return {"exception_info": {"exception_type": kind, "exception_message": message} if kind else None,
            "verifier_result": None if rewards is None else {"rewards": rewards}}


def test_only_infrastructure_failures_without_a_reward_are_revived():
    assert revive.infrastructure_failure(harbor_result("RuntimeError", R1)).startswith("the host could not start")
    assert revive.infrastructure_failure(harbor_result("RuntimeError", R2)).startswith("Docker could not run")
    assert revive.infrastructure_failure(harbor_result("RuntimeError", R3)).startswith("Docker could not run")
    assert revive.infrastructure_failure(harbor_result("BlockingIOError", "[Errno 11] Resource temporarily unavailable"))
    assert revive.infrastructure_failure(harbor_result("RuntimeError", R1, {"reward": None}))
    assert revive.infrastructure_failure(harbor_result("RuntimeError", R1, {"reward": 0.12})) is None
    for kind, message in (("AgentTimeoutError", "Agent execution timed out after 347400.0 seconds"),
                          ("NonZeroAgentExitCodeError", "Command failed (exit 1): fork: resource temporarily unavailable"),
                          ("VerifierTimeoutError", "Verifier timed out"), ("RewardFileNotFoundError", "no reward file"),
                          ("RuntimeError", "Docker compose command failed for environment x. Return code: 1. "
                                           "Stdout: service \"main\" is not running"),
                          ("RuntimeError", "Agent install failed: exit 2"), (None, "")):
        assert revive.infrastructure_failure(harbor_result(kind, message)) is None, (kind, message)


def ended_trial(tmp_path: Path, result: dict, *extra: str) -> tuple[Path, list[str]]:
    """An ALE-Bench trial Harbor ended with RESULT, and its run.py arguments."""
    from harbor.models.trial.config import AgentConfig, TaskConfig, TrialConfig
    import ale_bench_task
    import run

    args = ["ale-bench", "--problem", "ahc014", "--cli", "claude", "--model", "m", "--timeout", "3600", "--name", "pytest-revive-ended", *extra]
    cmd, _, _ = run.prepare(run.parser().parse_args([*args, "--dry-run"]))
    job = tmp_path / "pytest-revive-ended"
    trial = job / "ahc014__x"
    trial.mkdir(parents=True)
    task = tmp_path / "task"  # the run's task directory: its task.toml, as the dry run builds it
    task.mkdir()
    built = ale_bench_task.build_job(tmp_path / "ale-job", ["ahc014"], agents=1, cli="claude", dry_run=True)
    (task / "task.toml").write_bytes(built.tasks[0].files["task.toml"])
    config = TrialConfig(task=TaskConfig(path=task), trial_name=trial.name, trials_dir=job,
                         agent=AgentConfig(import_path="agent:ClaudeCodeCollaborationAgent", model_name="m"),
                         agent_timeout_multiplier=float(cmd[cmd.index("--agent-timeout-multiplier") + 1]))
    (trial / "config.json").write_text(config.model_dump_json(indent=4))
    (trial / "result.json").write_text(json.dumps(result))
    return trial, args


def test_a_trial_harbor_ended_is_revived_with_the_process_cap(tmp_path, monkeypatch):
    import pids_limit

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake")
    trial, args = ended_trial(tmp_path, harbor_result("RuntimeError", R3))
    revival = revive.Revival(trial, args)
    assert revival.harbor_ended.startswith("Docker could not run") and revival.kind() == "finished"
    assert revival.pids_added and pids_limit.configured(revival.config.environment) == 16384
    assert json.loads((trial / "config.json").read_text())["environment"]["import_path"] is None  # the run's record
    assert revival.note_for() == revive.FAULT_NOTE
    off = revive.Revival(*ended_trial(tmp_path / "off", harbor_result("RuntimeError", R1), "--pids-limit", "off"))
    assert not off.pids_added and off.config.environment.import_path is None
    rewarded, args = ended_trial(tmp_path / "rewarded", harbor_result("RuntimeError", R1, {"reward": 0.5}))
    with pytest.raises(ValueError, match="Harbor finished it"):
        revive.Revival(rewarded, args)
    timed_out, args = ended_trial(tmp_path / "timeout", harbor_result("AgentTimeoutError", "timed out"))
    with pytest.raises(ValueError, match="infrastructure failure"):
        revive.Revival(timed_out, args)


def test_the_process_cap_only_for_docker_trials():
    from harbor.models.trial.config import TaskConfig, TrialConfig

    def config(**environment):
        return TrialConfig(task=TaskConfig(path=Path("/t")), trial_name="t__1", trials_dir=Path("/j"),
                           environment=environment)

    plain = config(type="docker")
    assert revive.apply_pids_limit(plain, 16384) and plain.environment.kwargs == {"pids_limit": 16384}
    assert plain.environment.type is None and plain.environment.import_path == "pids_limit:CappedDockerEnvironment"
    assert not revive.apply_pids_limit(plain, 4096)  # the trial's own cap stays
    untouched = config(type="docker")
    assert not revive.apply_pids_limit(untouched, None) and untouched.environment.import_path is None
    with pytest.raises(ValueError, match="--pids-limit off"):
        revive.apply_pids_limit(config(type="daytona"), 16384)


def keep_aside_revival(trial: Path, number: int = 1) -> revive.Revival:
    revival = object.__new__(revive.Revival)
    revival.trial, revival.job, revival.number = trial, trial.parent, number
    revival.harbor_ended, revival.record, revival.records = "the host could not start a thread", {}, []
    return revival


def test_the_ended_trials_records_are_kept_aside(tmp_path):
    trial = tmp_path / "job" / "t__1"
    for name in ("result.json", "exception.txt", "artifacts/manifest.json", "artifacts/app/partial.zig",
                 "artifacts/logs/artifacts/published.txt"):
        touch(trial / name, name)
    touch(tmp_path / "job" / "result.json", "job result")
    harbor = SimpleNamespace(paths=SimpleNamespace(artifacts_dir=trial / "artifacts"),
                             _main_artifacts_mount_dir=trial / "artifacts/logs/artifacts")
    revival = keep_aside_revival(trial)
    revival.keep_previous_result(harbor)
    assert not (trial / "result.json").exists() and (trial / "result.before-revive-1.json").read_text() == "result.json"
    assert (trial / "exception.before-revive-1.txt").is_file() and not (trial / "exception.txt").exists()
    assert sorted(p.name for p in (trial / "artifacts").iterdir()) == ["logs"]  # the /logs/artifacts mount stays
    assert (trial / "artifacts/logs/artifacts/published.txt").is_file()
    assert (trial / "artifacts.before-revive-1/app/partial.zig").is_file()
    assert (tmp_path / "job/result.before-revive-1.json").read_text() == "job result"
    assert (tmp_path / "job/result.json").is_file()  # finish_job rewrites it at the end
    assert revival.record["harbor_ended"]["kept"]["result.json"] == "result.before-revive-1.json"


def snapshot_store(trial: Path, records: list[dict], promoted: list[int]) -> Path:
    state = trial / "frontier-swe"
    (state / "manifests").mkdir(parents=True)
    for record in records:
        if record.get("digest"):
            (state / "manifests" / f"{record['digest']}.json").write_text("{}")
    (state / "snapshots.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    touch(trial / "agent/frontier-swe/submissions.jsonl",
          "".join(json.dumps({"seq": seq, "status": "promoted"}) + "\n" for seq in promoted)
          + json.dumps({"seq": 99, "status": "rejected"}) + "\n")
    return trial


SNAPSHOTS = [{"seq": 0, "ts": "2026-10-11T08:10:07.271Z", "status": "baseline", "digest": "d0"},
             {"seq": 4, "ts": "2026-10-11T09:44:34.632Z", "status": "graded", "digest": "d4", "submission_seq": 2},
             {"seq": 5, "ts": "2026-10-11T09:56:48.116Z", "status": "graded", "digest": "d5", "submission_seq": 3},
             {"seq": 6, "ts": "2026-10-11T10:05:00.000Z", "status": "error", "error": "capture: failed"}]


def test_a_rebuild_restores_the_newest_snapshot_with_the_last_promotion(tmp_path):
    import frontier_swe

    trial = snapshot_store(tmp_path / "a", SNAPSHOTS, [1, 2, 3])
    assert revive.newest_snapshot(trial, frontier_swe.FRONTIER_SWE)["seq"] == 5
    later = snapshot_store(tmp_path / "b", SNAPSHOTS, [1, 2, 3, 4])
    with pytest.raises(RuntimeError, match="#4 was promoted later"):
        revive.newest_snapshot(later, frontier_swe.FRONTIER_SWE)
    with pytest.raises(RuntimeError, match="no snapshot"):
        revive.newest_snapshot(snapshot_store(tmp_path / "c", SNAPSHOTS[3:], []), frontier_swe.FRONTIER_SWE)
    baseline = snapshot_store(tmp_path / "d", SNAPSHOTS[:1], [])
    assert revive.newest_snapshot(baseline, frontier_swe.FRONTIER_SWE)["seq"] == 0


def test_the_note_says_what_happened():
    revival = keep_aside_revival(Path("/j/t"))
    revival.note_given, revival.mode = None, "rebuilt"
    revival.snapshot = {"ts": "2026-10-11T09:56:48.116Z", "submission_seq": 3}
    revival.record = {"rebuild": {"app_root": "/app"}}
    note = revival.note_for()
    assert note == ("Your session was interrupted by a machine fault and your container was rebuilt. /app and your "
                    "working copy were restored from the snapshot of /app taken after the last submission (#3, "
                    "09:56 UTC); changes after it, build outputs and any other files outside /logs/agent are lost, "
                    "and background processes are no longer running. Continue from where you left off.")
    revival.snapshot = {"ts": "2026-10-11T08:10:07Z", "submission_seq": None}
    assert "taken at 08:10 UTC, before any submission" in revival.note_for()
    revival.mode = "restarted"
    assert revival.note_for() == revive.FAULT_NOTE
    revival.harbor_ended = None
    assert revival.note_for() == revive.DEFAULT_NOTE
    revival.note_given = "Carry on."
    assert revival.note_for() == "Carry on."


def test_the_revived_runner_logs_what_happened(tmp_path):
    agent_dir = written_run(tmp_path)
    plan = {**plan_for("team"), "event_message": revive.EVENT_MESSAGES["rebuilt"].format(cli="Claude Code")}
    runner = revive.rendered_runner("team", render(agent_dir, "claude", harness_env("antiherding_matched", ON), plan))
    assert "they were rebuilt, its files restored from the last snapshot, and Claude Code continues" in runner
    assert "The host of this trial died" not in runner
    classic = revive.rendered_runner("team", render(agent_dir, "claude", harness_env("antiherding_matched", ON),
                                                    plan_for("team")))
    assert "The host of this trial died; its containers were restarted and Claude Code continues" in classic
    with pytest.raises(ValueError, match="quotes"):
        render(agent_dir, "claude", harness_env("antiherding_matched", ON), {**plan_for("team"), "event_message": "it's"})


def test_failed_revivals_add_no_working_time():
    first = {"status": "failed", "trial_started_at_epoch": 5.0}
    second = {"status": "running", "worked_seconds_before": 3600.0, "revived_at_epoch": 10000.0}
    usable = [r for r in (first, second) if r.get("status") != "failed"
              and "worked_seconds_before" in r and "revived_at_epoch" in r]
    assert revive.worked_before(usable, 1000.0, 10600.0) == 4200.0
    assert revive.worked_before([], 1000.0, 4600.0) == 3600.0


def test_the_snapshot_grader_of_a_revived_trial_reads_the_trials_task(tmp_path, monkeypatch):
    import run

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake")
    argv = ["frontier-swe", "--task", "git-to-zig", "--cli", "claude", "--model", "m", "--name", "pytest-revive-fswe"]
    try:
        args, cmd, env = revive.harness(argv, "pytest-revive-fswe")
    except ValueError as error:
        pytest.skip(f"needs the frontier-swe-v2 clone: {error}")
    assert not Path(cmd[cmd.index("-p") + 1]).exists()  # the probe's task directory: never written
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text("[verifier]\ntimeout_sec = 60.0\n")
    fixed = revive.host_worker_command(cmd, task)
    assert fixed[fixed.index("-p") + 1] == str(task) and len(fixed) == len(cmd)
    started = []
    monkeypatch.setattr(run.subprocess, "Popen", lambda argv, **kwargs: started.append(argv))
    run.start_host_worker(args, fixed, env, tmp_path / "job", revive=True)
    assert started[0][started[0].index("--task-dir") + 1] == str(task)
    assert started[0][started[0].index("--pids-limit") + 1] == "16384"


def test_the_agent_phase_trusts_the_dry_runs_checks(tmp_path):
    """revive.py's dry run checks the run's files as the host user; by the agent phase they belong to
    the agent's user again (0700 run-agent.sh), so the agent phase must not read them again."""
    agent_dir = written_run(tmp_path)
    runner = agent_dir / "multiagent-collab/run-agent.sh"
    runner.chmod(0)
    try:
        if os.access(runner, os.R_OK):
            pytest.skip("running as a user who reads any file")
        with pytest.raises(RuntimeError, match="run-agent.sh is unreadable"):
            render(agent_dir, "claude", harness_env("antiherding_matched", ON), plan_for("team"))
        plan = {**plan_for("team"), "checks_passed": True}
        calls = render(agent_dir, "claude", harness_env("antiherding_matched", ON), plan)
        assert "revive_session_ids=([0]=s-1)" in revive.rendered_runner("team", calls) and "checked" not in plan
    finally:
        runner.chmod(0o700)


def test_the_agents_files_go_back_to_the_agent_user(tmp_path):
    """After a teardown every log belongs to the host user; the files the runner writes go back to
    the task's agent user (a root agent needs nothing)."""
    trial = tmp_path / "job" / "t__1"
    for name in ("agent/sessions-0-a1/claude-prompt.txt", "agent/agent-0-a1-output.log",
                 "agent/multiagent-collab/run-agent.sh", "agent/frontier-swe/submissions.jsonl",
                 "frontier-swe/snapshots.jsonl", "agent/setup/stdout.txt"):
        touch(trial / name)
    revival = keep_aside_revival(trial)
    revival.cli = "claude"
    environment = revive.RecordingEnvironment()

    def harbor(user):
        return SimpleNamespace(task=SimpleNamespace(config=SimpleNamespace(agent=SimpleNamespace(user=user))))

    asyncio.run(revival.give_back(harbor("agent"), environment))
    assert environment.calls == ["chown -R agent: /logs/agent/sessions-0-a1 /logs/agent/agent-0-a1-output.log "
                                 "/logs/agent/multiagent-collab /logs/agent/frontier-swe"]
    assert revival.record["ownership"]["agent_user"]["result"] == "ok"
    for user in (None, "root"):
        environment = revive.RecordingEnvironment()
        asyncio.run(revival.give_back(harbor(user), environment))
        assert environment.calls == []


# ---- raising the working-time cap (--raise-timeout) ---------------------------------------------------

REASON = "test: the agents used their working time long before their token budget"


def test_a_raised_cap_only_raises():
    assert revive.raised_cap([], 345600, None) is None
    assert revive.raised_cap([], 345600, 864000, "ale-bench") == 864000
    for lower in (345600, 100000):
        with pytest.raises(ValueError, match="only raising"):
            revive.raised_cap([], 345600, lower, "ale-bench")
    with pytest.raises(ValueError, match="builds its task from --timeout"):
        revive.raised_cap([], 3600, 7200, "frontier-swe")
    earlier = {"revival": 1, "status": "finished",
               "working_time_cap_raised": {"from_seconds": 345600, "to_seconds": 864000}}
    # A later revival keeps the raised cap only on purpose too; it never drops it silently.
    for status in ("finished", "running"):
        with pytest.raises(ValueError, match="give --raise-timeout 864000"):
            revive.raised_cap([{**earlier, "status": status}], 345600, None)
    assert revive.raised_cap([earlier], 345600, 864000, "ale-bench") == 864000
    assert revive.raised_cap([earlier], 345600, 900000, "ale-bench") == 900000
    with pytest.raises(ValueError, match="below the 864000 s revival 1"):
        revive.raised_cap([earlier], 345600, 500000, "ale-bench")
    # A revival that failed before its agents ran raised nothing.
    assert revive.raised_cap([{**earlier, "status": "failed"}], 345600, None) is None


def test_the_settings_check_with_a_raised_cap(monkeypatch):
    import run

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake")
    args, cmd, env = revive.harness(PILOT, "pilot2m-ahc014-opus")
    config = SimpleNamespace(agent=SimpleNamespace(import_path="agent:ClaudeCodeCollaborationAgent",
                                                   model_name="claude-opus-5-5"),
                             agent_timeout_multiplier=run.harbor_agent_timeout(args) / 10800.0)
    assert revive.check_settings(config, args, cmd, 10800.0) == 86400
    assert revive.check_settings(config, args, cmd, 10800.0, raise_to=864000) == 86400
    raised_argv = [*PILOT[:-4], "--timeout", "864000", *PILOT[-2:]]
    raised, raised_cmd, raised_env = revive.harness(raised_argv, "pilot2m-ahc014-opus")
    with pytest.raises(ValueError, match="give --raise-timeout SECONDS"):  # refused without the option
        revive.check_settings(config, raised, raised_cmd, 10800.0)
    assert revive.check_settings(config, raised, raised_cmd, 10800.0, raise_to=864000) == 86400
    with pytest.raises(ValueError, match="does not give"):
        revive.check_settings(config, raised, raised_cmd, 10800.0, raise_to=900000)
    # With a raised cap the command and the environment are built with it; the arguments keep the run's.
    kept, _, built = revive.harness(PILOT, "pilot2m-ahc014-opus", 864000)
    assert kept.timeout == 86400 and env["RELEASE_AGENT_TIMEOUT"] == "86400"
    assert built["RELEASE_AGENT_TIMEOUT"] == raised_env["RELEASE_AGENT_TIMEOUT"] == "864000"


def test_a_revival_raises_the_working_time_cap_on_purpose(tmp_path, monkeypatch):
    import run

    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake")
    trial, args = ended_trial(tmp_path, harbor_result("RuntimeError", R1), "--resources-tool")
    (trial / "result.json").unlink()  # a trial whose host died: a classic revival
    longer = ["7200" if word == "3600" else word for word in args]
    with pytest.raises(ValueError, match="--raise-timeout SECONDS"):
        revive.Revival(trial, longer)
    with pytest.raises(ValueError, match="only raising"):
        revive.Revival(trial, args, raise_timeout=1800, raise_reason=REASON)
    with pytest.raises(ValueError, match="--raise-reason"):
        revive.Revival(trial, args, raise_timeout=7200)
    revival = revive.Revival(trial, args, raise_timeout=7200, raise_reason=REASON)
    assert revival.kind() == "classic" and revival.rewrites_resources()
    assert (revival.run_cap, revival.cap_raised, revival.args.timeout) == (3600, 7200, 7200)
    assert revival.env["RELEASE_AGENT_TIMEOUT"] == "7200" and revival.env["RELEASE_RESOURCES_TOOL"] == "1"
    waiting = run.harbor_agent_timeout(revival.args) - 7200  # the usage-limit waiting allowed, and the grace
    assert waiting == 48 * 3600 + run.LIMIT_RESUME_GRACE_SECONDS
    assert revival.harbor_timeout_run == 3600 + waiting
    assert revival.agent_timeout_left(5000.0) == 7200 + waiting - 5000.0  # from the new cap
    record = revival.cap_raise_record()
    assert (record["from_seconds"], record["to_seconds"], record["reason"], record["on_purpose"]) == (
        3600, 7200, REASON, True)
    assert record["harbor_agent_timeout_seconds"] == 7200 + waiting
    assert any("raised on purpose from 3600 s to 7200 s" in line and REASON in line
               for line in revival.differences("team"))
    assert revival.summary()["working_time_cap_raised"] == {"from_seconds": 3600, "to_seconds": 7200,
                                                            "reason": REASON}
    # The new cap as the arguments' --timeout works too; the run's cap comes from the trial's agent timeout.
    same = revive.Revival(trial, longer, raise_timeout=7200, raise_reason=REASON)
    assert (same.run_cap, same.args.timeout, same.env["RELEASE_AGENT_TIMEOUT"]) == (3600, 7200, "7200")
    # Without the option, nothing changes.
    plain = revive.Revival(trial, args)
    assert (plain.run_cap, plain.cap_raised, plain.args.timeout, plain.env["RELEASE_AGENT_TIMEOUT"]) == (
        3600, None, 3600, "3600")
    assert not plain.rewrites_resources() and "working_time_cap_raised" not in plain.summary()
    # A later revival must raise it again, to at least the same cap.
    revive.save_records(trial, [{"revival": 1, "status": "finished", "working_time_cap_raised": record}])
    with pytest.raises(ValueError, match="raised this trial's working-time cap from 3600 s to 7200 s"):
        revive.Revival(trial, args)
    with pytest.raises(ValueError, match="below the 7200 s revival 1"):
        revive.Revival(trial, args, raise_timeout=5000, raise_reason=REASON)
    assert revive.Revival(trial, args, raise_timeout=7200, raise_reason=REASON).args.timeout == 7200


def written_text(calls: list[str], name: str) -> str:
    """The text a command writes to a file of that name (printf %s TEXT > PATH)."""
    for call in calls:
        words = shlex.split(call)
        for index, word in enumerate(words):
            if word == ">" and words[index + 1].endswith(name) and words[index - 2] == "%s":
                return words[index - 1]
    raise AssertionError(f"no command writes {name}")


def test_the_raised_cap_reaches_the_runner_the_keeper_and_resources(tmp_path):
    """A run with --resources-tool and a 3600 s cap, revived with 864000 s: the files the run wrote pass
    the byte-identity checks, and the runner's time cap, the keeper's limit watcher and the rewritten
    `resources` command have the new cap."""
    run_env = {**ON, "RELEASE_RESOURCES_TOOL": "1", "RELEASE_AGENT_TIMEOUT": "3600"}
    agent_dir = written_run(tmp_path, extra=run_env)
    assert "`resources`" in (agent_dir / "sessions-0-a1/task-instruction.txt").read_text()
    raised = {**run_env, "RELEASE_AGENT_TIMEOUT": "864000"}
    plan = plan_for("team")
    runner = revive.rendered_runner("team", render(agent_dir, "claude", harness_env("antiherding_matched", raised), plan))
    checked = {entry["path"].rsplit("/", 1)[-1] for entry in plan["checked"]}
    assert checked == {"protocol.json", "shared-prompt.md", "task-instruction.txt", "claude-prompt.txt", "run-agent.sh"}
    assert "limit_timeout=864000" in runner and "limit_timeout=3600" not in runner
    guarded = render(agent_dir, "claude", harness_env("antiherding_matched", {**raised, "RELEASE_HARNESS_GUARD": "1"}),
                     plan_for("team"))
    conf = json.loads(written_text(guarded, "conf.json"))
    assert conf["units"][0]["watchers"]["limit"]["argv"][6] == "864000"
    # The `resources` command, written again by the revival, shows the new limit.
    revival = keep_aside_revival(tmp_path)
    revival.cli, revival.harbor_ended, revival.mode, revival.cap_raised = "claude", None, "restarted", 864000
    assert revival.rewrites_resources()
    environment = revive.RecordingEnvironment()
    with mock.patch.dict(os.environ, harness_env("antiherding_matched", raised), clear=False):
        sys.modules.pop("agent", None)
        agent = importlib.import_module("agent").ClaudeCodeCollaborationAgent(logs_dir=agent_dir,
                                                                              model_name="claude-sonnet-5-5")
        asyncio.run(revival.reinstall_resources(SimpleNamespace(agent=agent), environment,
                                                [{"index": 0, "session_dir": str(agent_dir / "sessions-0-a1")}],
                                                time.time() - 3600))
    assert revival.record["resources_tool"]["timeout_seconds"] == 864000.0
    script = tmp_path / "resources"
    script.write_text(written_text(environment.calls, "/resources"))
    shown = subprocess.run([sys.executable, str(script), "--home", "/logs/agent/sessions-0-a1/claude-home"],
                           capture_output=True, text=True, check=True).stdout
    row = next(line for line in shown.splitlines() if line.startswith("| Time"))
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[0] == "Time" and cells[2] == "240h" and int(cells[3].split("h")[0]) in (238, 239), shown
