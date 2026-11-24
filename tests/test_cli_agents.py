"""Checks for the Claude Code and Codex agent backends (no Docker needed)."""

from __future__ import annotations

import importlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from cli_render import CASES, ROOT, render


def scripts(calls: list[dict]) -> list[str]:
    """The team runner written by the setup step, and the final run command."""
    found = [calls[-1]["command"]]
    for call in calls:
        tokens = shlex.split(call["command"])
        for i, token in enumerate(tokens):
            if token.endswith("run-agent.sh") and tokens[i - 1] == ">":
                found.append(tokens[i - 2])
    return found


@pytest.fixture
def agent_module(monkeypatch):
    monkeypatch.setenv("RELEASE_CONTINUATION", "continue")
    monkeypatch.setenv("RELEASE_TASK", "ale-bench")
    sys.modules.pop("agent", None)
    sys.path.insert(0, str(ROOT))
    return importlib.import_module("agent")


def bash(helpers: str, call: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", f"{helpers}\n{call}"], capture_output=True, text=True)


def write_jsonl(path: Path, events: list) -> Path:
    path.write_text("".join((e if isinstance(e, str) else json.dumps(e)) + "\n" for e in events))
    return path


@pytest.mark.parametrize("cli", ["claude", "codex"])
@pytest.mark.parametrize("case", CASES)
def test_generated_scripts_parse(cli, case, tmp_path):
    for index, script in enumerate(scripts(render(case, cli=cli))):
        path = tmp_path / f"{index}.sh"
        path.write_text(script)
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_claude_invocation_and_auth():
    calls = render("ale-team3", cli="claude", model="claude-sonnet-4.6",
                   env_overrides={"ANTHROPIC_API_KEY": "fake-api-key"})
    run_call = calls[-1]
    assert "claude-sonnet-4-6" in run_call["command"]
    assert run_call["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "fake-claude-oauth-token"
    assert "ANTHROPIC_API_KEY" not in run_call["env"]
    runner = scripts(calls)[1]
    assert "--tools Bash," in runner and "--disallowedTools Agent,Task," in runner
    assert "--strict-mcp-config" in runner and "--resume \"$session_id\"" in runner


def test_codex_invocation(tmp_path):
    calls = render("ale-single", cli="codex", model="gpt-5.6-sol")
    run_cmd = calls[-1]["command"]
    assert 'exec "$@" --json' in run_cmd and 'resume "$session_id"' in run_cmd
    assert 'multi_agent = false' in run_cmd and 'web_search = "disabled"' in run_cmd
    assert calls[-1]["env"]["MULTIAGENT_CODEX_EFFORT"] == "max"


@pytest.mark.parametrize("model, requested, expected", [
    ("gpt-5.5", "max", "xhigh"),
    ("gpt-5.6-sol", "max", "max"),
    ("gpt-5.5", "medium", "medium"),
    ("unknown-model", "max", "max"),
])
def test_codex_effort_clamp(agent_module, model, requested, expected):
    catalog = {"models": [
        {"slug": "gpt-5.5", "supported_reasoning_levels": [
            {"effort": level} for level in ("low", "medium", "high", "xhigh")]},
        {"slug": "gpt-5.6-sol", "supported_reasoning_levels": [
            {"effort": level} for level in ("low", "medium", "high", "xhigh", "max", "ultra")]},
    ]}
    result = subprocess.run(
        [sys.executable, "-c", agent_module.CODEX_EFFORT_CLAMP_SOURCE, model, requested],
        input=json.dumps(catalog), capture_output=True, text=True, check=True)
    assert result.stdout.strip() == expected


CLAUDE_TOOL_OUTPUT_429 = {"type": "user", "session_id": "s1", "message": {
    "content": [{"type": "tool_result", "content": "HTTP 429 from my own script"}]}}


@pytest.mark.parametrize("events, retryable", [
    ([{"type": "result", "is_error": True, "session_id": "s1",
       "result": "You've hit your limit · resets 5pm"}], True),
    ([{"type": "result", "is_error": True, "session_id": "s1",
       "result": "API Error: 529 overloaded_error"}], True),
    ([{"type": "result", "is_error": True, "session_id": "s1",
       "result": "API Error: Usage credits required for 1M context"}], False),
    ([CLAUDE_TOOL_OUTPUT_429, {"type": "result", "is_error": True, "session_id": "s1",
                               "result": "Prompt is too long"}], False),
])
def test_claude_retry_classifier(agent_module, tmp_path, events, retryable):
    helpers = agent_module.ClaudeCodeAgent(logs_dir=tmp_path)._action_feedback_helpers()
    log = write_jsonl(tmp_path / "attempt.jsonl", events)
    result = bash(helpers, f"is_retryable_claude_api_error {log}")
    assert (result.returncode == 0) == retryable, result.stderr


@pytest.mark.parametrize("events, retryable", [
    ([{"type": "turn.failed", "error": {"message": "You've hit your usage limit. Try again at 3 PM."}}], True),
    ([{"type": "error", "message": "stream disconnected before completion"}], True),
    ([{"type": "error", "message": "Selected model is at capacity. Please try a different model."},
      {"type": "turn.failed", "error": {"message": "Selected model is at capacity. Please try a different model."}}],
     True),
    ([{"type": "turn.failed", "error": {"message": "unsupported_value: 'max' is not supported"}}], False),
])
def test_codex_retry_classifier(agent_module, tmp_path, events, retryable):
    helpers = agent_module.CodexAgent(logs_dir=tmp_path)._action_feedback_helpers()
    log = write_jsonl(tmp_path / "attempt.jsonl", events)
    result = bash(helpers, f"is_retryable_codex_api_error {log}")
    assert (result.returncode == 0) == retryable, result.stderr


def test_session_ids_and_exit_status(agent_module, tmp_path):
    claude = agent_module.ClaudeCodeAgent(logs_dir=tmp_path)._action_feedback_helpers()
    log = write_jsonl(tmp_path / "claude.jsonl", [
        "not json", {"type": "system", "subtype": "init", "session_id": "s1"},
        {"type": "result", "is_error": True, "session_id": "s1", "result": "boom"}])
    assert bash(claude, f"extract_claude_session_id {log}").stdout.strip() == "s1"
    assert bash(claude, f"finish_claude_attempt {log} 0").returncode == 1

    codex = agent_module.CodexAgent(logs_dir=tmp_path)._action_feedback_helpers()
    log = write_jsonl(tmp_path / "codex.jsonl", [
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "error", "message": "Reconnecting... 1/5"},
        {"type": "turn.completed", "usage": {}}])
    assert bash(codex, f"extract_codex_session_id {log}").stdout.strip() == "t1"
    assert bash(codex, f"finish_codex_attempt {log} 0").returncode == 0


def test_token_accounting(agent_module, tmp_path):
    class Context:
        n_input_tokens = n_cache_tokens = n_output_tokens = None

    claude_dir = tmp_path / "claude"
    claude_dir.mkdir()
    usage = {"input_tokens": 5, "cache_read_input_tokens": 100,
             "cache_creation_input_tokens": 10, "output_tokens": 7}
    write_jsonl(claude_dir / "agent-single-output.log", [
        {"type": "result", "usage": usage}, {"type": "result", "usage": usage}])
    context = Context()
    agent_module.ClaudeCodeAgent(logs_dir=claude_dir).populate_context_post_run(context)
    assert (context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens) == (230, 200, 14)

    codex_dir = tmp_path / "codex"
    codex_dir.mkdir()
    write_jsonl(codex_dir / "agent-0-a1-output.log", [
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "turn.completed", "usage": {"input_tokens": 100, "cached_input_tokens": 50, "output_tokens": 5}},
        {"type": "thread.started", "thread_id": "t1"},
        {"type": "turn.completed", "usage": {"input_tokens": 300, "cached_input_tokens": 200, "output_tokens": 9}}])
    write_jsonl(codex_dir / "agent-1-a2-output.log", [
        {"type": "thread.started", "thread_id": "t2"},
        {"type": "turn.completed", "usage": {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 1}}])
    context = Context()
    agent_module.CodexAgent(logs_dir=codex_dir).populate_context_post_run(context)
    assert (context.n_input_tokens, context.n_cache_tokens, context.n_output_tokens) == (310, 200, 10)


def run_py(*argv: str):
    sys.path.insert(0, str(ROOT))
    import run
    return run.prepare(run.parser().parse_args([*argv, "--dry-run"]))


@pytest.mark.parametrize("cli, agent_class", [
    ("claude", "ClaudeCodeCollaborationAgent"),
    ("codex", "CodexCollaborationAgent"),
])
def test_run_py_selects_backend(cli, agent_class):
    cmd, env, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", cli, "--model", "m")
    assert cmd[cmd.index("--agent-import-path") + 1] == f"agent:{agent_class}"
    targets = [m["target"] for m in json.loads(cmd[cmd.index("--mounts-json") + 1])]
    assert any(target.startswith(f"/mnt/release-{cli}") for target in targets)


@pytest.mark.parametrize("cli", ["claude", "codex"])
@pytest.mark.parametrize("case", ["ale-team3", "ale-solo", "ale-team4"])
def test_instruction_stays_off_command_lines(cli, case):
    """The run step (a long-lived shell, and run-agent.sh's arguments) never carries the task
    instruction, so an agent's `pkill -f <word from its task>` cannot stop its own runner.
    The setup step writes it to each agent's task-instruction.txt instead."""
    from cli_render import INSTRUCTION
    calls = render(case, cli=cli)
    first_line = INSTRUCTION.splitlines()[0]
    assert first_line not in calls[-1]["command"]
    setup = next(c["command"] for c in calls if "task-instruction.txt" in c["command"])
    assert shlex.quote(INSTRUCTION) in setup or first_line in setup
    runner = scripts(calls)[1]
    assert 'instruction="$(cat "$5"; printf x)"; instruction="${instruction%x}"' in runner
    assert "/task-instruction.txt" in calls[-1]["command"]


def test_instruction_file_keeps_every_byte(agent_module, tmp_path):
    """run-agent.sh reads the instruction file back exactly, trailing newlines included,
    so the prompt is the same as when the instruction was an argument."""
    text = "Solve it.\nQuotes ' \" and $dollars and `ticks`.\n\n"
    path = tmp_path / "task-instruction.txt"
    path.write_text(text)
    read = 'set -- a b c d "$1"; instruction="$(cat "$5"; printf x)"; instruction="${instruction%x}"; printf %s "$instruction"'
    result = subprocess.run(["bash", "-c", read, "bash", str(path)], capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout == text


TOOL_LINES = {
    "claude": '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Bash"}]}}',
    "codex": '{"type":"item.completed","item":{"id":"i","type":"command_execution","command":"ls"}}',
}


def run_give_up_loop(agent_cls, tmp_path, turns):
    """Run the GIVE_UP_TURNS bash for a sequence of continuation turns, each (tool calls,
    seconds it lasted). Returns (exit code, stdout, the runner log)."""
    init, start, check = agent_cls._give_up_parts(agent_cls, "")
    cli = agent_cls.CLI
    lines = [f'log_file={shlex.quote(str(tmp_path / "log.jsonl"))}', "is_arc_task=0", ": > \"$log_file\"", init]
    for n, (tools, seconds) in enumerate(turns, 1):
        attempt = tmp_path / f"attempt-{n}.jsonl"
        attempt.write_text("".join(TOOL_LINES[cli] + "\n" for _ in range(tools)) + '{"type":"result"}\n')
        lines += [f"no_action_attempt={n}", f"attempt_log={shlex.quote(str(attempt))}", start,
                  f"attempt_started=$((attempt_started - {seconds}))", check]
    lines.append("echo survived")
    result = subprocess.run(["bash", "-c", "\n".join(lines)], capture_output=True, text=True)
    return result.returncode, result.stdout, (tmp_path / "log.jsonl").read_text()


@pytest.mark.parametrize("cls", ["CodexAgent", "ClaudeCodeAgent"])
def test_agent_that_gave_up_is_stopped(agent_module, tmp_path, cls):
    agent_cls = getattr(agent_module, cls)
    code, out, log = run_give_up_loop(agent_cls, tmp_path, [(1, 5)] * 12)
    assert code == 0 and "survived" not in out
    event = json.loads(log.strip().splitlines()[-1])
    assert event["data"]["event"] == "agent_gave_up"
    assert event["data"]["consecutiveIdleTurns"] == 10 and event["data"]["continuationTurn"] == 10


@pytest.mark.parametrize("cls", ["CodexAgent", "ClaudeCodeAgent"])
def test_a_working_turn_resets_the_count(agent_module, tmp_path, cls):
    agent_cls = getattr(agent_module, cls)
    # 9 idle turns, one with real work (5 tool calls), 9 idle, one long turn, 9 idle: never 10 in a row.
    turns = [(0, 5)] * 9 + [(5, 5)] + [(2, 5)] * 9 + [(0, 300)] + [(1, 5)] * 9
    code, out, log = run_give_up_loop(agent_cls, tmp_path, turns)
    assert code == 0 and "survived" in out and "agent_gave_up" not in log


