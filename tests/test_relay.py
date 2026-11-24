"""The restarted single agent, a relay (run.py --relay; agent.py RELAY_ENV; docs/setups.md).

One agent's run in legs: each leg a fresh CLI session with its own token budget, session directory, log and
watchers, the same agent id (so the same working copy), and every leg after the first the first leg's prompt with
RELAY_NOTE before it. The render tests read the commands agent.py sends; the end-to-end tests run them on this
machine with the pinned Claude Code and Codex releases against tests/limit_mock_api.py (test_harness_guard's Run).
"""

from __future__ import annotations

import asyncio
import importlib
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest import mock

import pytest
from harbor.models.trial.paths import EnvironmentPaths

from cli_render import BASE_ENV, CASES, CLASSES, INSTRUCTION, ROOT, FakeEnvironment
from test_cli_agents import run_py
from test_harness_guard import GUARD, HOME_DIR, LIMIT, Run, wait_for
from test_limit_resume import MARK, NOTE, needs_claude, needs_codex, user_texts

sys.path.insert(0, str(ROOT))
import harness_guard  # noqa: E402
import limit_resume  # noqa: E402
import revive  # noqa: E402
import token_budget  # noqa: E402

NOW = 1_700_000_000.0
BUDGET = {"RELEASE_TOKEN_BUDGET": "2000000", "RELEASE_TOKEN_BUDGET_METRIC": "weighted"}


def relay_env(legs: int = 3, protocol: str = "bare_solo", guard: bool = True, **extra: str) -> dict:
    return {**LIMIT, **(GUARD if guard else {}), **BUDGET, "RELEASE_RELAY": str(legs),
            "MULTIAGENT_PROTOCOL": f"multiagent_protocols.{protocol}:MultiAgentProtocol", **extra}


class Container(FakeEnvironment):
    """FakeEnvironment that answers the relay's questions: each ended leg's limit-wait.json (WAITED seconds) and
    the markers a leg left (MARKERS: leg number -> marker names)."""

    def __init__(self, waited: float = 0.0, markers: dict[int, list[str]] | None = None) -> None:
        super().__init__()
        self.waited = waited
        self.markers = markers or {}

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        result = await super().exec(command, user, env, cwd, timeout_sec)
        if "limit-wait.json" in command and "{ cat" in command:
            result.stdout = json.dumps({"waited_seconds": self.waited}) + "\n"
        elif command.startswith("for marker in") and "episode-over.json" in command:
            leg = int(m.group(1)) if (m := re.search(r"sessions-0-a1-leg(\d+)/", command)) else 1
            result.stdout = "".join(f"/logs/agent/x/{name}\n" for name in self.markers.get(leg, []))
        return result


def render(env: dict, case: str = "ale-single", cli: str = "codex", container: Container | None = None,
           revive_plan: dict | None = None, reload: bool = True) -> tuple[list[dict], object]:
    """agent.py's commands for CASE with ENV, at a fixed time; the calls and the agent."""
    spec = CASES[case]
    full = {**BASE_ENV, "RELEASE_TASK": spec["task"], "MULTIAGENT_N_AGENT": str(spec["agents"]),
            "MULTIAGENT_PROTOCOL": f"multiagent_protocols.{spec['protocol']}:MultiAgentProtocol",
            "RELEASE_CONTINUATION": (ROOT / "benchmarks/ale-bench/continuation.txt").read_text().rstrip("\n"),
            **env}
    counter = itertools.count()
    environment = container or Container()
    with mock.patch.dict(os.environ, full, clear=False), \
            mock.patch("secrets.token_urlsafe", lambda n: f"tok{next(counter)}"), mock.patch("time.time", lambda: NOW):
        if reload:
            sys.modules.pop("agent", None)
        module = importlib.import_module("agent")
        logs = ROOT / ".cache" / "render-logs"
        logs.mkdir(parents=True, exist_ok=True)
        agent = getattr(module, CLASSES[cli])(logs_dir=logs, model_name="m")

        async def go() -> None:
            if revive_plan is None:
                await agent.run(INSTRUCTION, environment, SimpleNamespace())
            else:
                agent.revive_plan = revive_plan
                await agent.resume(INSTRUCTION, environment, SimpleNamespace())

        asyncio.run(go())
    return environment.calls, module


def written(calls: list[dict], path: str) -> list[str]:
    """Every text a call wrote to PATH with `printf %s TEXT > PATH`."""
    found = []
    for call in calls:
        if path not in call["command"]:
            continue
        words = shlex.split(call["command"])
        for index, word in enumerate(words[:-3]):
            if word == "printf" and words[index + 1] == "%s" and words[index + 3] == ">" and words[index + 4] == path:
                found.append(words[index + 2])
    return found


def runner_of(calls: list[dict], leg: int) -> str:
    name = "run-agent.sh" if leg == 1 else f"run-agent-leg{leg}.sh"
    found = written(calls, f"/logs/agent/multiagent-collab/{name}")
    assert len(found) == 1, (name, len(found))
    return found[0]


def keeper_confs(calls: list[dict]) -> list[dict]:
    return [json.loads(text) for text in written(calls, "/usr/lib/hsv/conf.json")]


def bare(call: dict) -> str:
    """A call's command without the prefix Harbor's exec_as_root and exec_as_agent add."""
    return call["command"].removeprefix("set -o pipefail; ")


def resources_installs(calls: list[dict]) -> list[str]:
    """The root commands that write the `resources` command (agent.py _resources_install)."""
    return [bare(call) for call in calls
            if bare(call).startswith("set -euo pipefail; mkdir -p /usr/local/libexec /usr/local/bin")]


def stops(calls: list[dict]) -> list[dict]:
    """The commands that stop what a leg left running."""
    return [call for call in calls if bare(call).startswith("set +o pipefail; for entry in")]


def stop_entries(calls: list[dict]) -> list[str]:
    """What they look for: each a CLI home variable and its value."""
    return [re.search(r"for entry in (\S+?); do", bare(call)).group(1) for call in stops(calls)]


# ---- run.py ------------------------------------------------------------------------------------------------

def test_run_py_sets_the_legs():
    base = ("ale-bench", "--problem", "ahc014", "--cli", "codex", "--model", "m", "--protocol", "bare_solo", "--token-budget", "2000000")
    _, env, _ = run_py(*base, "--relay", "4")
    assert env["RELEASE_RELAY"] == "4" and env["RELEASE_TOKEN_BUDGET"] == "2000000"
    _, env, _ = run_py(*base)
    assert "RELEASE_RELAY" not in env
    _, env, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", "claude", "--model", "m", "--protocol", "antiherding_matched",
                       "--token-budget", "2000000", "--relay", "2")
    assert env["RELEASE_RELAY"] == "2"


@pytest.mark.parametrize("argv, message", [
    (("--relay", "1"), "2 or more legs"),
    (("--relay", "4", "--agents", "2"), "--agents 1"),
    (("--relay", "4", "--protocol", "antiherding"), "bare_solo or antiherding_matched"),
    (("--relay", "4", "--protocol", "board", "--agents", "1"), "needs a team"),
])
def test_run_py_refuses(argv, message):
    argv = list(argv)
    if "--protocol" not in argv:
        argv += ["--protocol", "bare_solo"]
    with pytest.raises(ValueError, match=re.escape(message)):
        run_py("ale-bench", "--problem", "ahc014", "--cli", "codex", "--model", "m", "--token-budget", "2000000", *argv)
    with pytest.raises(ValueError, match="needs --token-budget"):
        run_py("ale-bench", "--problem", "ahc014", "--cli", "codex", "--model", "m", "--protocol", "bare_solo", "--relay", "3")


# ---- what agent.py sends -----------------------------------------------------------------------------------

@pytest.mark.parametrize("guard", [True, False])
@pytest.mark.parametrize("protocol", ["bare_solo", "antiherding_matched"])
@pytest.mark.parametrize("cli", ["codex", "claude"])
def test_legs(cli, protocol, guard, tmp_path):
    env = relay_env(3, protocol, guard, RELEASE_RESOURCES_TOOL="1")
    calls, module = render(env, cli=cli)
    plain, _ = render({key: value for key, value in env.items() if key != "RELEASE_RELAY"}, cli=cli)
    home = HOME_DIR[cli]
    # The first leg is the plain solo run: the same runner.
    assert runner_of(calls, 1) == runner_of(plain, 1)
    first, second, third = (runner_of(calls, leg) for leg in (1, 2, 3))
    note = shlex.quote(module.RELAY_NOTE)
    assert note not in first and note in second and second == third
    # Only the prompt differs: the note's line before the first leg's prompt.
    assert second.replace(f"    printf %s {note}\n", "", 1).replace("  {\n    printf %s \"$instruction\"\n  } >",
                                                                      '  printf %s "$instruction" >') == first
    for runner in (first, second, third):
        assert subprocess.run(["bash", "-n"], input=runner, text=True, capture_output=True).returncode == 0
    # Each leg has its own session directory, log and instruction file; the same agent id and index.
    text = "\n".join(call["command"] for call in calls)
    for label in ("0-a1", "0-a1-leg2", "0-a1-leg3"):
        assert f"mkdir -p /logs/agent/sessions-{label}/{home}" in text
        assert written(calls, f"/logs/agent/sessions-{label}/task-instruction.txt") == written(
            calls, "/logs/agent/sessions-0-a1/task-instruction.txt")
    assert "sessions-0-a1-leg4" not in text
    record = json.loads(written(calls, "/logs/agent/multiagent-collab/protocol.json")[0])
    assert record["relay"] == {"legs": 3, "leg_token_budget": 2000000, "note": module.RELAY_NOTE}
    assert "relay" not in json.loads(written(plain, "/logs/agent/multiagent-collab/protocol.json")[0])
    # The `resources` command is written for every leg, counting that leg's tokens.
    resources = resources_installs(calls)
    assert len(resources) == 3 and len(resources_installs(plain)) == 1
    for resource, label in zip(resources, ("0-a1", "0-a1-leg2", "0-a1-leg3")):
        assert f'"homes": ["/logs/agent/sessions-{label}/{home}"]' in resource
    # Between legs, whatever the last leg left running is stopped.
    assert stop_entries(calls) == [
        f"{'CODEX_HOME' if cli == 'codex' else 'CLAUDE_CONFIG_DIR'}=/logs/agent/sessions-{label}/{home}"
        for label in ("0-a1", "0-a1-leg2")]
    assert all(call["user"] == "root" for call in stops(calls))
    assert stop_entries(plain) == []
    if guard:
        confs = keeper_confs(calls)
        assert [conf["units"][0]["session_dir"] for conf in confs] == [
            f"/logs/agent/sessions-{label}" for label in ("0-a1", "0-a1-leg2", "0-a1-leg3")]
        assert all(conf["units"][0]["id"] == "a1" and conf["units"][0]["index"] == 0 and conf["fresh"] for conf in confs)
        assert [call["command"] for call in calls].count("exec /usr/lib/hsv/hsvi /usr/lib/hsv/hsvd k") == 3
    else:
        runs = [call["command"] for call in calls if " bash /logs/agent/multiagent-collab/run-agent" in call["command"]]
        assert len(runs) == 3 and "0 a1 1 /logs/agent/sessions-0-a1-leg2 /logs/agent/agent-0-a1-leg2-output.log" in runs[1]
        assert all(" >/logs/agent/agent-0-a1" in run for run in runs)


def test_the_note_comes_first_in_the_prompt(tmp_path):
    """Leg 2's runner writes its prompt file as the first leg's, with the note before it."""
    calls, module = render(relay_env(2, "antiherding_matched", guard=False))
    runner = runner_of(calls, 2)
    session = tmp_path / "s"
    session.mkdir()
    shared = tmp_path / "shared.md"
    shared.write_text("SHARED")
    block = runner[runner.index("  {\n    printf %s"):]
    block = block[:block.index('> "$prompt_file"\n') + len('> "$prompt_file"\n')]
    script = (f'instruction="TASK"; shared_prompt_path={shlex.quote(str(shared))}; '
              f'prompt_file={shlex.quote(str(session / "p"))}\n{block}')
    subprocess.run(["bash", "-c", script], check=True)
    assert (session / "p").read_text() == (module.RELAY_NOTE + "You are solving this task alone. Follow the solo work "
                                           "protocol exactly.\n\nSHARED\n\nTask instruction:\nTASK\n")


def test_waiting_carries_over(tmp_path):
    """A leg counts working time from the run's start less the earlier legs' waiting, and may wait what they
    left of the most allowed."""
    calls, _ = render(relay_env(3), container=Container(waited=100.0))
    wait_max = int(LIMIT["RELEASE_LIMIT_WAIT_MAX"])
    for leg, waited in ((1, 0), (2, 100), (3, 200)):
        runner = runner_of(calls, leg)
        assert f"limit_started={NOW + waited:.3f}" in runner
        assert f"limit_wait_max={wait_max - waited}" in runner
    limits = [conf["units"][0]["watchers"]["limit"]["argv"] for conf in keeper_confs(calls)]
    assert [(argv[5], argv[7]) for argv in limits] == [
        (f"{NOW + waited:.3f}", str(wait_max - waited)) for waited in (0, 100, 200)]
    assert resources_installs(calls) == []  # (no --resources-tool in this run)
    # With it, each leg's `resources` counts working time as its watcher does.
    calls, _ = render(relay_env(3, RELEASE_RESOURCES_TOOL="1"), container=Container(waited=100.0))
    starts = [re.findall(r'"started": ([0-9.]+)', command) for command in resources_installs(calls)]
    assert [[float(value) for value in found] for found in starts] == [[NOW], [NOW + 100], [NOW + 200]]


@pytest.mark.parametrize("marker", ["limit-wait-exhausted.json", "episode-over.json"])
def test_a_leg_that_used_up_the_run_ends_the_relay(marker):
    calls, _ = render(relay_env(3), container=Container(markers={1: [marker]}))
    assert len(keeper_confs(calls)) == 1 and "sessions-0-a1-leg2" not in "\n".join(c["command"] for c in calls)


def test_the_time_cap_ends_the_relay():
    class TimeCap(Container):
        async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
            result = await super().exec(command, user, env, cwd, timeout_sec)
            if command.startswith("for marker in") and "time-cap-reached.json" in command and "episode" not in command:
                result.stdout = "/logs/agent/sessions-0-a1/time-cap-reached.json\n"
            return result

    with pytest.raises(asyncio.TimeoutError):
        render(relay_env(3), container=TimeCap())


def test_no_waiting_left_ends_the_relay():
    calls, _ = render(relay_env(3), container=Container(waited=float(LIMIT["RELEASE_LIMIT_WAIT_MAX"])))
    assert len(keeper_confs(calls)) == 1


def test_a_relay_is_one_agent_with_a_budget():
    with pytest.raises(ValueError, match="not a team"):
        render(relay_env(3), case="ale-team3")
    with pytest.raises(ValueError, match="needs --token-budget"):
        render({key: value for key, value in relay_env(3).items() if not key.startswith("RELEASE_TOKEN_BUDGET")})
    with pytest.raises(ValueError, match="team runner"):
        # A protocol without a solo prompt runs one agent through the single-agent runner.
        render(relay_env(3, protocol="antiherding"))
    with pytest.raises(ValueError, match="2 or more"):
        render(relay_env(1))


# ---- revival -----------------------------------------------------------------------------------------------

def leg_agent(leg: int, finished: str | None = None, fresh: bool = False, waited: float = 0.0,
              over: bool = False) -> dict:
    return {"index": 0, "agent_id": "a1", "leg": leg, "session_id": None if fresh else f"s{leg}", "turn": 3,
            "finished_before": finished, "fresh": fresh, "waited_seconds": waited, "relay_over": over}


@pytest.mark.parametrize("agents, resumed, plan", [
    # The last leg's agent had not ended: resumed, with the earlier legs' waiting.
    ([leg_agent(1, "token_budget_exhausted", waited=5), leg_agent(2, waited=7)], [2],
     {"relay_leg": 2, "relay_waited": 5}),
    # It had not started: that leg starts afresh.
    ([leg_agent(1, "agent_gave_up", waited=5), leg_agent(2, fresh=True)], [], {"relay_leg": 2, "relay_waited": 5}),
    # It had ended on its own: the next leg starts, after both legs' waiting.
    ([leg_agent(1, "token-budget-exhausted.json", waited=5), leg_agent(2, "optimization_continuations_completed",
                                                                      waited=7)],
     [], {"relay_leg": 3, "relay_waited": 12}),
    # It had used up the run's working time (or ended on an error), or it was the last leg: nothing to do.
    ([leg_agent(1, "token_budget_exhausted"), leg_agent(2, "time-cap-reached.json", over=True)], [], {}),
    ([leg_agent(1, "token_budget_exhausted"), leg_agent(2, "codex_api_retries_exhausted")], [], {}),
    ([leg_agent(1, "token_budget_exhausted"), leg_agent(2, "token_budget_exhausted"),
      leg_agent(3, "token_budget_exhausted")], [], {}),
])
def test_what_a_relays_revival_does(agents, resumed, plan):
    found, settings = revive.relay_next(agents, 3)
    assert [agent["leg"] for agent in found] == resumed and settings == plan


def touch(path: Path, text: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_revive_reads_a_relays_legs(tmp_path):
    agent_dir = tmp_path / "agent"
    for label in ("0-a1", "0-a1-leg2"):
        (agent_dir / f"sessions-{label}" / "codex-home").mkdir(parents=True)
        touch(agent_dir / f"agent-{label}-output.log")
    touch(agent_dir / "sessions-0-a1" / "token-budget-exhausted.json", "{}")
    touch(agent_dir / "sessions-0-a1" / limit_resume.STATE, json.dumps({"waited_seconds": 42.5}))
    layout, found = revive.sessions(tmp_path, "codex")
    assert layout == "team" and [(s["agent_id"], s["leg"], s["dir"].name, s["log"].name) for s in found] == [
        ("a1", 1, "sessions-0-a1", "agent-0-a1-output.log"),
        ("a1", 2, "sessions-0-a1-leg2", "agent-0-a1-leg2-output.log")]
    with pytest.raises(ValueError, match="give the run's --relay"):
        revive.inspect_agents(tmp_path, "codex", 2000000, "weighted")
    _, agents = revive.inspect_agents(tmp_path, "codex", 2000000, "weighted", relay=4)
    assert [(a["leg"], a["finished_before"], a["fresh"], a["waited_seconds"]) for a in agents] == [
        (1, "token-budget-exhausted.json", False, 42.5), (2, None, True, 0.0)]
    assert revive.relay_next(agents, 4) == ([], {"relay_leg": 2, "relay_waited": 42.5})
    # agent.py's markers that end a relay are revive's.
    sys.modules.pop("agent", None)
    with mock.patch.dict(os.environ, {"RELEASE_CONTINUATION": "c"}, clear=False):
        assert importlib.import_module("agent").RELAY_FINAL_MARKERS == revive.RELAY_FINAL_MARKERS


def revive_plan(agents: list[dict], legs: int, **extra) -> dict:
    resumed, settings = revive.relay_next(agents, legs)
    return {"revival": 1, "note": NOTE, "layout": "team", "started": NOW - 500, "checks_passed": True,
            "agents": [{"index": a["index"], "session_id": a["session_id"], "turn": a["turn"]} for a in resumed],
            **settings, **extra}


def test_revived_relay_resumes_its_leg_then_runs_the_rest():
    plan = revive_plan([leg_agent(1, "token_budget_exhausted", waited=30), leg_agent(2, waited=10)], 3)
    calls, module = render(relay_env(3), container=Container(waited=10.0), revive_plan=plan)
    text = "\n".join(call["command"] for call in calls)
    assert "rm -rf" not in text and "run-agent.sh" not in text.replace("multiagent-collab/run-agent.sh'", "")
    revived = written(calls, "/logs/agent/multiagent-collab/run-agent-leg2-revive-1.sh")
    assert len(revived) == 1 and "revive_session_ids=([0]=s2)" in revived[0]
    assert f"limit_started={NOW - 500 + 30:.3f}" in revived[0]
    confs = keeper_confs(calls)
    assert [conf["units"][0]["session_dir"] for conf in confs] == [
        "/logs/agent/sessions-0-a1-leg2", "/logs/agent/sessions-0-a1-leg3"]
    # The resumed leg appends to its log; the next starts afresh with the note.
    assert runner_of(calls, 3).count(shlex.quote(module.RELAY_NOTE)) == 1
    assert f"limit_started={NOW - 500 + 40:.3f}" in runner_of(calls, 3)


def test_revived_relay_starts_a_leg_afresh():
    plan = revive_plan([leg_agent(1, "agent_gave_up", waited=30)], 3)
    assert plan["agents"] == [] and plan["relay_leg"] == 2
    calls, module = render(relay_env(3), revive_plan=plan)
    assert [conf["units"][0]["session_dir"] for conf in keeper_confs(calls)] == [
        "/logs/agent/sessions-0-a1-leg2", "/logs/agent/sessions-0-a1-leg3"]
    leg2 = runner_of(calls, 2)
    assert "revive_session_ids" not in leg2 and leg2.count(shlex.quote(module.RELAY_NOTE)) == 1
    assert f"limit_started={NOW - 500 + 30:.3f}" in leg2
    assert "mkdir -p /logs/agent/sessions-0-a1-leg2/codex-home" in "\n".join(c["command"] for c in calls)


def test_a_dry_run_renders_the_revived_leg_only(tmp_path):
    """revive.py's dry run (no checks_passed) checks the run's files (the first leg's and the revived leg's) and
    renders the leg it revives, not the ones after."""
    plan = revive_plan([leg_agent(1, "token_budget_exhausted"), leg_agent(2)], 3)
    del plan["checks_passed"]
    checked = []

    def check_files(self, revive, expected):
        checked.extend(expected)

    with mock.patch.dict(os.environ, {"RELEASE_CONTINUATION": "c"}, clear=False):
        sys.modules.pop("agent", None)
        cls = importlib.import_module("agent").CliAgent
    with mock.patch.object(cls, "_check_revived_files", check_files), \
            mock.patch.object(cls, "_check_revived_continuation", lambda self, revive, runner, source: None):
        calls, _ = render(relay_env(3), revive_plan=plan, reload=False)
    assert checked == ["/logs/agent/multiagent-collab/protocol.json", "/logs/agent/multiagent-collab/shared-prompt.md",
                       "/logs/agent/sessions-0-a1/task-instruction.txt", "/logs/agent/sessions-0-a1/codex-prompt.txt",
                       "/logs/agent/sessions-0-a1-leg2/task-instruction.txt",
                       "/logs/agent/sessions-0-a1-leg2/codex-prompt.txt"]
    assert [conf["units"][0]["session_dir"] for conf in keeper_confs(calls)] == ["/logs/agent/sessions-0-a1-leg2"]


# ---- end to end ----------------------------------------------------------------------------------------------

def legs_of(run: Run) -> list[Path]:
    return sorted(run.local.agent_dir.glob("sessions-0-a1*"), key=lambda path: (len(path.name), path.name))


def leg_events(run: Run, label: str, event: str) -> list[dict]:
    log = run.local.agent_dir / f"agent-{label}-output.log"
    found = []
    for line in (log.read_text(errors="replace") if log.exists() else "").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and entry.get("type") == "environment.feedback" and entry["data"].get("event") == event:
            found.append(entry["data"])
    return found


def first_requests(run: Run) -> list[list[str]]:
    """The user texts of each session's first request: a fresh session has only its prompt (the requests after
    a tool call in that turn have the same texts)."""
    found, previous = [], None
    for texts in map(user_texts, run.mock.main()):
        if len(texts) == 1 and texts != previous:
            found.append(texts)
        previous = texts
    return found


def relay_run(tmp_path: Path, cli: str, plan: list, legs: int, budget: int = 1000000, **kwargs) -> Run:
    env = {"RELEASE_RELAY": str(legs), "RELEASE_TOKEN_BUDGET": str(budget), "RELEASE_TOKEN_BUDGET_METRIC": "weighted",
           "MULTIAGENT_PROTOCOL": "multiagent_protocols.bare_solo:MultiAgentProtocol", **kwargs.pop("env", {})}
    return Run(tmp_path, cli, plan, case="ale-single", env=env, **kwargs)


def needs(cli: str):
    return needs_claude if cli == "claude" else needs_codex


@pytest.mark.parametrize("guard", [True, False])
@pytest.mark.parametrize("cli", [pytest.param("codex", marks=needs_codex), pytest.param("claude", marks=needs_claude)])
def test_three_legs_one_after_another(cli, guard, tmp_path):
    run = relay_run(tmp_path, cli, [], legs=3, guard=guard)
    try:
        run.finish(timeout=420)
    finally:
        run.close()
    assert run.error is None, run.error
    module = run.module
    assert [path.name for path in legs_of(run)] == ["sessions-0-a1", "sessions-0-a1-leg2", "sessions-0-a1-leg3"]
    # Each leg: the task prompt, then one continuation; then the next leg, a fresh session.
    firsts = first_requests(run)
    assert len(firsts) == 3 and len(run.mock.main()) == 6
    assert not firsts[0][0].startswith(module.RELAY_NOTE) and MARK in firsts[0][0]
    for texts in firsts[1:]:
        assert texts[0].startswith(module.RELAY_NOTE) and texts[0][len(module.RELAY_NOTE):] == firsts[0][0]
    for label, session in zip(("0-a1", "0-a1-leg2", "0-a1-leg3"), legs_of(run)):
        assert len(leg_events(run, label, "optimization_continuations_completed")) == 1, label
        assert (session / f"{cli}-prompt.txt").read_text().startswith(module.RELAY_NOTE) == (label != "0-a1")
        # Each leg's own watcher (its file may be up to a poll behind: legs this short end within one).
        assert json.loads((session / "token-usage.json").read_text())["budget"] == 1000000
        assert token_budget.weighted_tokens(token_budget.summed(token_budget.session_totals(
            cli, session / HOME_DIR[cli]))) > 0
        assert json.loads((session / "limit-wait.json").read_text())["waiting"] is False
    if guard:
        assert len(run.guard_events("keeper_finished")) == 3


@pytest.mark.parametrize("cli", [pytest.param("codex", marks=needs_codex), pytest.param("claude", marks=needs_claude)])
def test_a_leg_ends_on_its_budget_and_the_next_begins(cli, tmp_path):
    """Leg 1 spends its budget in one response and works on (the watcher polls every 5 s; Codex's tool call
    returns after 10 s, still running); the watcher stops it. Leg 2 starts with its own budget and the note."""
    pidfile = tmp_path / "sleep.pid"
    plan = [{"type": "tool", "command": f"echo $$ > {pidfile}; exec sleep 120", "output_tokens": 60000},
            {"type": "tool", "command": "sleep 15"}, {"type": "tool", "command": "sleep 15"}]
    run = relay_run(tmp_path, cli, plan, legs=2, budget=50000)
    try:
        run.finish(timeout=420)
    finally:
        run.close()
    assert run.error is None, run.error
    first, second = legs_of(run)
    assert (first / "token-budget-exhausted.json").is_file() and not (second / "token-budget-exhausted.json").exists()
    assert len(leg_events(run, "0-a1", "token_budget_exhausted")) == 1
    assert len(leg_events(run, "0-a1-leg2", "optimization_continuations_completed")) == 1
    spent = [token_budget.weighted_tokens(token_budget.summed(token_budget.session_totals(cli, leg / HOME_DIR[cli])))
             for leg in (first, second)]
    assert spent[0] >= 50000 and 0 < spent[1] < 50000, spent
    firsts = first_requests(run)
    assert len(firsts) == 2 and firsts[1][0].startswith(run.module.RELAY_NOTE)
    assert not harness_guard.alive(int(pidfile.read_text()))


@pytest.mark.parametrize("cli", [pytest.param("codex", marks=needs_codex), pytest.param("claude", marks=needs_claude)])
def test_what_a_leg_left_running_is_stopped(cli, tmp_path):
    """Leg 1 leaves a command running in the background and ends after its continuation: it is stopped before
    leg 2 starts."""
    pidfile = tmp_path / "left.pid"
    plan = [{"type": "tool", "command": f"nohup setsid sleep 300 > /dev/null 2>&1 < /dev/null & echo $! > {pidfile}"},
            {"type": "text", "text": "done-0"},
            {"type": "tool", "command": f"kill -0 $(cat {pidfile}) 2>/dev/null; echo leg1-probe-$?"},
            {"type": "text", "text": "done-1"},
            {"type": "tool", "command": f"kill -0 $(cat {pidfile}) 2>/dev/null; echo leg2-probe-$?"},
            {"type": "text", "text": "done-2"}]
    run = relay_run(tmp_path, cli, plan, legs=2)
    try:
        run.finish(timeout=420)
    finally:
        run.close()
    assert run.error is None, run.error
    bodies = [json.dumps(request["body"]) for request in run.mock.main()]
    # (The tool's output: the command's own text has "-$?".)
    assert "leg1-probe-0" in bodies[3]  # in leg 1, after its first turn, it ran
    assert "leg2-probe-1" in bodies[5] and "leg2-probe-0" not in bodies[5]  # in leg 2 it was gone


@pytest.mark.parametrize("cli", [pytest.param("codex", marks=needs_codex), pytest.param("claude", marks=needs_claude)])
def test_a_relay_revived_in_its_second_leg(cli, tmp_path):
    """The host dies in leg 2's tool call. The revival resumes leg 2's session with the note, then runs leg 3
    afresh."""
    plan = [{"type": "text", "text": "leg1-0"}, {"type": "text", "text": "leg1-1"},
            {"type": "tool", "command": "echo leg2-before-crash"}, {"type": "tool", "command": "sleep 600"}]
    run = relay_run(tmp_path, cli, plan, legs=3)
    try:
        run.answered(4, timeout=240)
        wait_for(lambda: any(p.name == "sessions-0-a1-leg2" for p in legs_of(run)), 30, "leg 2")
        time.sleep(3)
        run.abandon()
    finally:
        run.close()
    _, agents = revive.inspect_agents(tmp_path, cli, 1000000, "weighted", relay=3)
    resumed, settings = revive.relay_next(agents, 3)
    assert [a["leg"] for a in resumed] == [2] and settings["relay_leg"] == 2
    plan2 = {"revival": 1, "note": NOTE, "layout": "team", "started": time.time() - 60,
             "agents": [{"index": 0, "session_id": a["session_id"], "turn": a["turn"]} for a in resumed], **settings}
    # revive.py's dry run: the run's files (the first leg's and leg 2's prompts, ...) must be what this code
    # writes, and it renders leg 2 only. (Local wrote the run's /logs/agent at the trial's directory.)
    checked = dict(plan2)
    agent_dir = tmp_path / "agent"
    with mock.patch.dict(os.environ, run.env, clear=False), \
            mock.patch.object(EnvironmentPaths, "agent_dir", PurePosixPath(agent_dir)):
        sys.modules.pop("agent", None)
        module = importlib.import_module("agent")
        dry = getattr(module, CLASSES[cli])(logs_dir=agent_dir,
                                               model_name="claude-sonnet-5-5" if cli == "claude" else "gpt-mock")
        dry.revive_plan = checked
        recorder = revive.RecordingEnvironment()
        asyncio.run(dry.resume(INSTRUCTION, recorder, SimpleNamespace()))
    assert [Path(entry["path"]).relative_to(agent_dir).as_posix() for entry in checked["checked"]] == [
        "multiagent-collab/protocol.json", "multiagent-collab/shared-prompt.md", "sessions-0-a1/task-instruction.txt",
        f"sessions-0-a1/{cli}-prompt.txt", "sessions-0-a1-leg2/task-instruction.txt",
        f"sessions-0-a1-leg2/{cli}-prompt.txt", "multiagent-collab/run-agent.sh"]
    assert sum(" hsvd k" in call for call in recorder.calls) == 1  # the revived leg only
    env = {"RELEASE_RELAY": "3", "RELEASE_TOKEN_BUDGET": "1000000", "RELEASE_TOKEN_BUDGET_METRIC": "weighted",
           "MULTIAGENT_PROTOCOL": "multiagent_protocols.bare_solo:MultiAgentProtocol"}
    revived = Run(tmp_path, cli, [{"type": "text", "text": "leg2-after"}], case="ale-single",
                  env=env, revive_plan={**plan2, "checks_passed": True}, mock_dir="mock2")
    try:
        revived.finish(timeout=420)
    finally:
        revived.close()
    assert revived.error is None, revived.error
    assert [path.name for path in legs_of(revived)] == ["sessions-0-a1", "sessions-0-a1-leg2", "sessions-0-a1-leg3"]
    requests = revived.mock.main()
    texts = user_texts(requests[0])
    assert texts[-1] == NOTE and texts[0].startswith(revived.module.RELAY_NOTE)  # leg 2, resumed
    assert "leg2-before-crash" in json.dumps(requests[0]["body"])
    firsts = first_requests(revived)
    assert len(firsts) == 1 and firsts[0][0].startswith(revived.module.RELAY_NOTE)  # leg 3, afresh
    assert len(leg_events(revived, "0-a1-leg2", "trial_revived")) == 1
    assert len(leg_events(revived, "0-a1-leg3", "optimization_continuations_completed")) == 1
