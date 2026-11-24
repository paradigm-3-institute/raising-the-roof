"""Checks for token budgets and the matched single-agent prompt (no Docker needed)."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cli_render import ROOT, render
from test_cli_agents import agent_module, bash, run_py, scripts, write_jsonl  # noqa: F401

sys.path.insert(0, str(ROOT))
from multiagent_protocols import antiherding, antiherding_matched  # noqa: E402
from multiagent_protocols.base import ProtocolPaths  # noqa: E402
from token_budget import HOME_ENV, session_totals, weighted_tokens  # noqa: E402

MATCHED = "multiagent_protocols.antiherding_matched:MultiAgentProtocol"
PATHS = ProtocolPaths.default("/logs/agent/multiagent-collab")
MODELS = {"codex": "gpt-6-luna", "claude": "claude-sonnet-4-6"}
AGENT_CLASSES = {"codex": "CodexAgent", "claude": "ClaudeCodeAgent"}
HOME_DIRS = {"codex": "codex-home", "claude": "claude-home"}


def prompt(protocol, agents: int, monkeypatch) -> str:
    monkeypatch.setenv("MULTIAGENT_N_AGENT", str(agents))
    return protocol.MultiAgentProtocol().build(instruction="", paths=PATHS, model="m").agents[0].prompt


def test_team_prompt_is_antiherding(monkeypatch):
    assert prompt(antiherding_matched, 4, monkeypatch) == prompt(antiherding, 4, monkeypatch)


def test_solo_prompt_only_drops_peer_lines(monkeypatch):
    team = prompt(antiherding, 1, monkeypatch).split("\n")
    solo = prompt(antiherding_matched, 1, monkeypatch).split("\n")
    rewritten = {new for _, new in antiherding_matched.EDITS if new is not None}
    # Every solo line is a team line or one of the listed rewrites, in team order.
    kept = [line for line in solo if line not in rewritten]
    team_lines = iter(team)
    assert all(line in team_lines for line in kept)
    assert len(solo) == len(kept) + len(rewritten)
    assert not any(word in "\n".join(solo) for word in ("peer", "ADOPT", "adoption", "team"))


def matched_calls(case: str, cli: str = "codex", budget: str = "") -> list[dict]:
    overrides = {"MULTIAGENT_PROTOCOL": MATCHED}
    if budget:
        overrides["RELEASE_TOKEN_BUDGET"] = budget
    return render(case, cli=cli, model=MODELS[cli], env_overrides=overrides)


@pytest.mark.parametrize("cli", ["codex", "claude"])
def test_matched_single_agent_runs_the_solo_prompt(cli):
    calls = matched_calls("ale-single", cli)
    setup = next(call["command"] for call in calls if "shared-prompt.md" in call["command"])
    assert "Paths:" in setup and "Shared paths:" not in setup
    assert "You are solving this task alone." in scripts(calls)[1]


@pytest.mark.parametrize("cli", ["codex", "claude"])
@pytest.mark.parametrize("case", ["ale-single", "ale-team3"])
def test_token_budget_in_runner(cli, case, tmp_path):
    runners = scripts(matched_calls(case, cli, budget="1000"))
    for index, script in enumerate(runners):
        path = tmp_path / f"{index}.sh"
        path.write_text(script)
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    runner = runners[1]
    assert "token_budget=1000" in runner and "token_budget_metric=weighted" in runner
    assert f'start_token_budget "${cli}_log_dir"' in runner
    assert runner.count("if token_budget_exhausted; then") == 2
    assert "token_budget=\n" in scripts(matched_calls(case, cli))[1]


@pytest.mark.parametrize("cli", ["codex", "claude"])
def test_attempt_stopped_by_budget_ends_cleanly(agent_module, tmp_path, cli):
    agent = getattr(agent_module, AGENT_CLASSES[cli])(logs_dir=tmp_path)
    marker = tmp_path / "token-budget-exhausted.json"
    script = "\n".join([
        agent._action_feedback_helpers(),
        agent._retry_function(""),
        f'token_budget_marker={marker}; log_file={tmp_path}/log; session_id=""',
        f'invoke_{cli}() {{ touch {marker}; echo "{{}}" > "$2"; return 143; }}',
        f"run_{cli}_with_api_retries /dev/null {tmp_path}/attempt.jsonl",
    ])
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def rollout(path: Path, session: str, outputs: list[int], tail: str = "") -> None:
    """A Codex rollout whose running output totals step through outputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    events = [{"type": "session_meta", "payload": {"id": session}}]
    for output in outputs:
        events.append({"type": "event_msg", "timestamp": "2026-10-05T10:00:00Z", "payload": {
            "type": "token_count",
            "info": {"total_token_usage": {"input_tokens": output * 10, "cached_input_tokens": output * 9,
                                           "output_tokens": output, "reasoning_output_tokens": output // 2}},
            "rate_limits": {"primary": {"used_percent": 19.0}}}})
    path.write_text("".join(json.dumps(event) + "\n" for event in events) + tail)


def transcript(path: Path, session: str, outputs: list[int], ids: list[str] | None = None) -> None:
    """A Claude Code transcript with one response per output count, each stored
    as two entries (two content blocks) that repeat the message id and usage.
    Like rollout(), input is ten times the output and cache reads nine times."""
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, output in enumerate(outputs):
        usage = {"input_tokens": 0, "cache_creation_input_tokens": output,
                 "cache_read_input_tokens": output * 9, "output_tokens": output,
                 "output_tokens_details": {"thinking_tokens": output // 2}}
        entry = {"type": "assistant", "sessionId": session, "timestamp": "2026-10-05T10:00:00Z",
                 "message": {"id": (ids or [f"{session}-m{i}" for i in range(len(outputs))])[index],
                             "usage": usage}}
        entries += [entry, entry]
    entries.append({"type": "user", "sessionId": session, "message": {"content": "tool output"}})
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    # The runner's attempt log, next to the home, carries plan usage.
    rate_limit = {"type": "rate_limit_event", "rate_limit_info": {
        "unifiedWindows": {"seven_day": {"utilization": 0.19}}}}
    (path.parents[3] / "claude-attempt-0.jsonl").write_text(json.dumps(rate_limit) + "\n")


def usage_logs(cli: str, home: Path, session: str = "s1") -> None:
    """Logs adding up to 1,200 output tokens and 12,000 input, 10,800 cached."""
    if cli == "codex":
        rollout(home / "sessions" / "rollout-a.jsonl", session, [400, 1200])
    else:
        transcript(home / "projects" / "-app" / f"{session}.jsonl", session, [400, 800])


def test_weighted_tokens():
    # CAISI: output 100%, unique input 23%, total input 2%.
    usage = {"input_tokens": 1000, "cached_input_tokens": 800, "output_tokens": 50,
             "reasoning_output_tokens": 30}
    assert weighted_tokens(usage) == pytest.approx(50 + 0.23 * 200 + 0.02 * 1000)


def test_codex_session_totals(tmp_path):
    sessions = tmp_path / "codex-home" / "sessions" / "2026" / "10" / "05"
    rollout(sessions / "rollout-a.jsonl", "s1", [10, 30], tail='{"type": "event_msg", "payl')
    rollout(sessions / "rollout-b.jsonl", "s1", [30, 50])  # s1 resumed into a new file
    rollout(sessions / "rollout-c.jsonl", "s2", [7])
    totals = session_totals("codex", tmp_path / "codex-home")
    assert {session: usage["output_tokens"] for session, usage in totals.items()} == {"s1": 50, "s2": 7}


def test_claude_session_totals(tmp_path):
    projects = tmp_path / "claude-home" / "projects" / "-app"
    transcript(projects / "s1.jsonl", "s1", [10, 20])
    # A resumed session that copied s1's history counts each response once.
    transcript(projects / "s2.jsonl", "s2", [10, 20, 5], ids=["s1-m0", "s1-m1", "s2-m2"])
    totals = session_totals("claude", tmp_path / "claude-home")
    assert {session: usage["output_tokens"] for session, usage in totals.items()} == {"s1": 30, "s2": 5}
    assert totals["s1"] == {"input_tokens": 300, "cached_input_tokens": 270,
                            "output_tokens": 30, "reasoning_output_tokens": 15}


@pytest.mark.parametrize("cli", ["codex", "claude"])
def test_accounting_counts_stopped_turns(agent_module, tmp_path, cli):
    class Context:
        n_input_tokens = n_cache_tokens = n_output_tokens = metadata = None

    # The last turn was stopped by the budget, so the CLI never reported its end-of-turn
    # usage; the session logs still count it.
    write_jsonl(tmp_path / "agent-0-a1-output.log", [
        {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 90, "output_tokens": 10}},
        {"type": "result", "usage": {"input_tokens": 100, "output_tokens": 10}}])
    usage_logs(cli, tmp_path / "sessions-0-a1" / HOME_DIRS[cli], "s1")
    usage_logs(cli, tmp_path / "sessions-1-a2" / HOME_DIRS[cli], "s2")
    context = Context()
    getattr(agent_module, AGENT_CLASSES[cli])(logs_dir=tmp_path).populate_context_post_run(context)
    assert (context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens) == (24000, 21600, 2400)
    assert context.metadata == {"weighted_tokens": 2 * 1716}


@pytest.mark.parametrize("cli", ["codex", "claude"])
def test_runner_starts_the_watcher(agent_module, tmp_path, cli):
    agent = getattr(agent_module, AGENT_CLASSES[cli])(logs_dir=tmp_path)
    home = tmp_path / "session" / HOME_DIRS[cli]
    usage_logs(cli, home)
    script = "\n".join([
        agent._action_feedback_helpers(),
        "token_budget=1500 token_budget_metric=weighted",
        f"token_budget_source={shlex.quote(agent_module.TOKEN_BUDGET_SOURCE)}",
        f"start_token_budget {home}",
        "for i in $(seq 100); do token_budget_exhausted && exit 0; sleep 0.2; done; exit 1",
    ])
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    record = json.loads((tmp_path / "session" / "token-budget-exhausted.json").read_text())
    assert (record["budget_metric"], record["weighted_tokens"]) == ("weighted", 1716)


@pytest.mark.skipif(not Path("/proc/self/environ").exists(), reason="needs /proc")
@pytest.mark.parametrize("cli", ["codex", "claude"])
@pytest.mark.parametrize("metric, stops", [("weighted", True), ("output", False)])
def test_watcher_stops_everything_under_the_home(tmp_path, cli, metric, stops):
    # 1,200 output tokens and 12,000 input, 10,800 of them cached: 1,716 weighted,
    # so a 1,500 budget is spent in weighted tokens but not in output tokens.
    home = tmp_path / "session" / HOME_DIRS[cli]
    usage_logs(cli, home)
    marker, usage = tmp_path / "exhausted.json", tmp_path / "usage.json"
    name = HOME_ENV[cli]
    under_budget = subprocess.Popen(["sleep", "60"], env={**os.environ, name: str(home)})
    bystander = subprocess.Popen(["sleep", "60"], env={**os.environ, name: f"{home}-other"})
    watcher = subprocess.Popen([sys.executable, str(ROOT / "token_budget.py"),
                                cli, str(home), "1500", metric, str(marker), str(usage)])
    try:
        deadline = time.time() + (20 if stops else 6)
        while under_budget.poll() is None and time.time() < deadline:
            time.sleep(0.2)
        record = json.loads(usage.read_text())
        assert record["usage"]["output_tokens"] == 1200 and record["weighted_tokens"] == 1716
        assert record["plan_used_percent"] == 19.0 and record["budget_metric"] == metric
        assert (under_budget.poll() is not None) == stops == marker.is_file()
        assert bystander.poll() is None
    finally:
        for process in (watcher, under_budget, bystander):
            process.kill()
            process.wait()


def test_run_py_token_budget():
    _, env, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", "codex", "--model", "m", "--token-budget", "5000",
                       "--protocol", "antiherding_matched")
    assert env["RELEASE_TOKEN_BUDGET"] == "5000"
    assert env["RELEASE_TOKEN_BUDGET_METRIC"] == "weighted"
    assert env["MULTIAGENT_PROTOCOL"] == MATCHED
    _, env, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", "claude", "--model", "m", "--token-budget", "5000")
    assert env["RELEASE_TOKEN_BUDGET_METRIC"] == "weighted"
