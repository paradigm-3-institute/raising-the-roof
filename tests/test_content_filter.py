"""Content-filter blocks (agent.py CONTENT_FILTER_*): a reply the API blocked ends the turn, the
agent continues in the same session with the usual continuation and nothing else, and 5 blocks in
a row with no completed output in between stop it cleanly.

The end-to-end tests run the pinned CLIs against tests/limit_mock_api.py, as test_limit_resume.py
does. Its "block" step returns the API's real block: for Claude Code the stream error "Output
blocked by content filtering policy" after a finished thinking block, as in
jobs/solo8m-fswe-gitzig-opus-r1-blocked (or, before any output, as an HTTP 400); for Codex a
response.incomplete for "content_filter". They are skipped without the pinned releases.
"""

from __future__ import annotations

import importlib
import json
import shlex
import subprocess
import sys

import pytest

from cli_render import ROOT, render
from test_limit_resume import (MARK, NOTE, ON, continuation_text, feedback, needs_claude, needs_codex,
                               polyomino_continuation, run_agent, user_texts)

sys.path.insert(0, str(ROOT))
import limit_resume  # noqa: E402
import revive  # noqa: E402
import token_budget  # noqa: E402

BLOCK_TEXT = "API Error: Output blocked by content filtering policy"
BLOCK_WORDS = ("content filtering", "content_filter", "API Error", "blocked")
TOOL = {"type": "tool", "command": "echo one"}
BLOCK = {"type": "block"}


def blocks(run) -> list[dict]:
    return feedback(run, "content_filter_block")


def assert_no_duplicates(run, continuation: str, turns: int) -> None:
    """No request carries the task prompt more than once or a note; the last carries the
    continuation once per continuation turn (each turn's continuation is the same text)."""
    for request in run.mock.main():
        texts = user_texts(request)
        assert sum(MARK in text for text in texts) == 1, texts
        assert NOTE not in texts, texts
    texts = user_texts(run.mock.main()[-1])
    assert sum(text.strip() == continuation for text in texts) == turns, texts
    assert all(MARK in text or text.strip() == continuation for text in texts), texts


def assert_told_nothing(run) -> None:
    """Nothing the harness sends mentions the block."""
    for request in run.mock.main():
        for text in user_texts(request):
            assert not any(word in text for word in BLOCK_WORDS), text


def events(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---- end to end: Claude Code ------------------------------------------------------------------

@needs_claude
def test_claude_block_mid_turn_then_success(tmp_path):
    """A tool call, then a reply the filter blocks part way: the turn ends there, and the next turn
    is the usual continuation in the same session, after which the agent works on as usual."""
    plan = [TOOL, BLOCK, {"type": "tool", "command": "echo two"}, {"type": "text", "text": "done-1"}]
    run = run_agent(tmp_path, "claude", plan)
    assert run.results[1].returncode == 0, run.results[1].stdout[-2000:] + run.results[1].stderr[-2000:]
    assert [r["response"] for r in run.mock.main()] == ["tool", "block", "tool", "text"]
    # Claude Code's events for the block are the blocked run's: a <synthetic> assistant message and
    # an error result with the API's message, after the blocked reply's finished thinking block.
    attempt = events(run.session / "claude-attempt-0.jsonl")
    synthetic = [e for e in attempt if e.get("type") == "assistant" and e["message"]["model"] == "<synthetic>"]
    assert [e["message"]["content"][0]["text"] for e in synthetic] == [BLOCK_TEXT]
    [result] = [e for e in attempt if e.get("type") == "result"]
    assert result["is_error"] and result["result"] == BLOCK_TEXT and result["terminal_reason"] == "api_error"
    assert [b for b in blocks(run)] == [{
        "event": "content_filter_block", "message": "The API blocked a reply of Claude Code (content filter), which "
        "ended the turn; the agent continues with the usual continuation.", "consecutiveBlocks": 1, "totalBlocks": 1,
        "maxConsecutiveBlocks": 5, "attemptLog": "claude-attempt-0.jsonl"}]
    # The next turn's message is the usual continuation, byte for byte.
    assert (run.session / "claude-no-action-feedback-1.txt").read_text() == continuation_text() + "\n"
    assert_no_duplicates(run, continuation_text(), 1)
    assert_told_nothing(run)
    # Claude Code leaves the blocked reply out of the conversation it sends.
    after = run.mock.main()[2]["body"]["messages"]
    assert not any(isinstance(block, dict) and block.get("type") == "thinking"
                   for message in after if message["role"] == "assistant" for block in message["content"])
    [transcript] = run.home.glob("projects/*/*.jsonl")
    assert sum(MARK in text for text in limit_resume.user_texts("claude", str(transcript), 0)) == 1
    # Tokens of the blocked reply count as usual (each mock reply reports 100 input tokens).
    usage = token_budget.summed(token_budget.session_totals("claude", run.home))
    assert usage["input_tokens"] == 100 * len(run.mock.main())
    assert not feedback(run, "claude_api_retry")


@needs_claude
@pytest.mark.parametrize("block", [BLOCK, {"type": "block", "http": 400}], ids=["stream", "http-400"])
def test_claude_block_on_the_first_turn(tmp_path, block):
    """The task prompt's first reply is blocked (single-agent runner): the task prompt is not sent
    again; the next message is the usual continuation."""
    run = run_agent(tmp_path, "claude", [block, {"type": "text", "text": "done-1"}], case="ale-single")
    assert run.results[1].returncode == 0, run.results[1].stdout[-2000:] + run.results[1].stderr[-2000:]
    assert [r["response"] for r in run.mock.main()] == ["block", "text"]
    assert [(b["consecutiveBlocks"], b["totalBlocks"]) for b in blocks(run)] == [(1, 1)]
    assert_no_duplicates(run, polyomino_continuation(), 1)
    assert_told_nothing(run)


@needs_claude
def test_claude_output_between_blocks_restarts_the_count(tmp_path):
    """Four blocks in a row, but a tool call ran before the first of them: no stop."""
    plan = [BLOCK, TOOL, BLOCK, BLOCK, BLOCK, BLOCK, {"type": "text", "text": "done-5"}]
    run = run_agent(tmp_path, "claude", plan, env={"RELEASE_CONTINUATIONS": "6"})
    assert run.results[1].returncode == 0, run.results[1].stdout[-2000:]
    assert [(b["consecutiveBlocks"], b["totalBlocks"]) for b in blocks(run)] == [(1, 1), (1, 2), (2, 3), (3, 4),
                                                                                  (4, 5)]
    assert not feedback(run, "content_filter_stop") and feedback(run, "optimization_continuations_completed")
    assert_no_duplicates(run, continuation_text(), 6)


# ---- end to end: both CLIs --------------------------------------------------------------------

@pytest.mark.parametrize("cli, case", [pytest.param("claude", "ale-solo", marks=needs_claude),
                                       pytest.param("codex", "ale-single", marks=needs_codex)])
def test_five_blocks_in_a_row_stop_the_agent(tmp_path, cli, case):
    """A tool call, then 5 blocked turns in a row: the runner stops the agent cleanly after the 5th
    and sends nothing more; the run ends as usual (exit 0) and is graded."""
    plan = [TOOL, BLOCK, BLOCK, BLOCK, BLOCK, BLOCK, {"type": "text", "text": "never"}]
    run = run_agent(tmp_path, cli, plan, case=case, env={"RELEASE_CONTINUATIONS": "10"})
    assert run.results[1].returncode == 0, run.results[1].stdout[-2000:] + run.results[1].stderr[-2000:]
    assert [r["response"] for r in run.mock.main()] == ["tool"] + ["block"] * 5
    assert [b["consecutiveBlocks"] for b in blocks(run)] == [1, 2, 3, 4, 5]
    [stop] = feedback(run, "content_filter_stop")
    assert stop["consecutiveBlocks"] == 5 and stop["totalBlocks"] == 5
    assert not feedback(run, "optimization_continuations_completed")
    assert_no_duplicates(run, continuation_text() if case == "ale-solo" else polyomino_continuation(), 4)
    assert_told_nothing(run)
    assert not run.results[2].stdout.strip()  # no time-cap marker: the trial is graded as usual


# ---- end to end: Codex ------------------------------------------------------------------------

@needs_codex
def test_codex_block_ends_the_turn(tmp_path):
    """Codex's turn fails on a "content_filter" response: before, the runner took it for a dropped
    stream, backed off and resumed with a note; now the turn is over and the usual continuation
    follows. Codex itself adds its <content_filter_guidance> developer message; the harness adds
    nothing."""
    plan = [TOOL, BLOCK, {"type": "text", "text": "done-1"}]
    run = run_agent(tmp_path, "codex", plan, case="ale-single")
    assert run.results[1].returncode == 0, run.results[1].stdout[-2000:]
    assert [r["response"] for r in run.mock.main()] == ["tool", "block", "text"]
    assert [b["attemptLog"] for b in blocks(run)] == ["codex-attempt-0.jsonl"]
    assert not feedback(run, "codex_api_retry") and not feedback(run, "interrupted_turn_resumed")
    assert (run.session / "codex-no-action-feedback-1.txt").read_text() == polyomino_continuation() + "\n"
    assert_no_duplicates(run, polyomino_continuation(), 1)
    assert_told_nothing(run)
    developer = [content.get("text", "") for item in run.mock.main()[-1]["body"]["input"]
                 if item.get("role") == "developer" for content in item.get("content", [])]
    assert any(text.startswith("<content_filter_guidance>") for text in developer)


@needs_codex
def test_codex_block_after_a_usage_limit(tmp_path):
    """A usage limit, then a block in the resumed turn: the runner waits for the reset and resumes
    with the note (usage-limit resumption), then the block ends the turn and the usual
    continuation follows. Each message is sent once."""
    plan = [TOOL, {"type": "limit", "seconds": 4}, BLOCK, {"type": "text", "text": "done-1"}]
    run = run_agent(tmp_path, "codex", plan, case="ale-single")
    assert run.results[1].returncode == 0, run.results[1].stdout[-2000:]
    assert [r["response"] for r in run.mock.main()] == ["tool", "429", "block", "text"]
    assert len(feedback(run, "usage_limit_wait")) == 1 and len(feedback(run, "interrupted_turn_resumed")) == 1
    assert len(blocks(run)) == 1
    texts = user_texts(run.mock.main()[-1])
    assert sum(MARK in text for text in texts) == 1 and texts.count(NOTE) == 1
    assert sum(text.strip() == polyomino_continuation() for text in texts) == 1


# ---- the outcome of an attempt (no CLI) -------------------------------------------------------

@pytest.fixture
def agent_module(monkeypatch):
    monkeypatch.setenv("RELEASE_CONTINUATION", "continue")
    monkeypatch.setenv("RELEASE_TASK", "ale-bench")
    sys.modules.pop("agent", None)
    return importlib.import_module("agent")


# The end of the blocked run's attempt log (jobs/solo8m-fswe-gitzig-opus-r1-blocked), shortened.
CLAUDE_PARTIAL = {"type": "assistant", "message": {"model": "claude-opus-5-5", "id": "msg_1", "role": "assistant",
                  "content": [{"type": "thinking", "thinking": "", "signature": "x"}], "stop_reason": None},
                  "session_id": "s1", "request_id": "req_1"}
CLAUDE_SYNTHETIC = {"type": "assistant", "message": {"model": "<synthetic>", "role": "assistant",
                    "content": [{"type": "text", "text": BLOCK_TEXT}]}, "session_id": "s1", "request_id": "req_1",
                    "error": "unknown", "is_api_error_message": True}
CLAUDE_BLOCK_RESULT = {"type": "result", "is_error": True, "subtype": "success", "terminal_reason": "api_error",
                       "api_error_status": None, "result": BLOCK_TEXT, "session_id": "s1"}
CLAUDE_TOOL_RESULT = {"type": "user", "message": {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}, "session_id": "s1"}
# An agent that prints the message itself is no block.
CLAUDE_TOOL_PRINTS_IT = {"type": "user", "message": {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "t2", "content": BLOCK_TEXT}]}, "session_id": "s1"}
CODEX_BLOCKED = "stream disconnected before completion: Incomplete response returned, reason: content_filter"
CODEX_COMMAND = {"type": "item.completed", "item": {"id": "i1", "type": "command_execution", "command": "ls",
                                                     "aggregated_output": CODEX_BLOCKED}}


@pytest.mark.parametrize("cli, lines, outcome", [
    ("claude", [CLAUDE_PARTIAL, CLAUDE_SYNTHETIC, CLAUDE_BLOCK_RESULT], "blocked"),
    ("claude", [CLAUDE_TOOL_RESULT, CLAUDE_PARTIAL, CLAUDE_SYNTHETIC, CLAUDE_BLOCK_RESULT], "blocked output"),
    ("claude", [CLAUDE_SYNTHETIC, {"type": "result", "is_error": True, "result": "API Error: 529 overloaded_error"}],
     "other"),
    ("claude", [CLAUDE_TOOL_PRINTS_IT, {"type": "result", "is_error": True, "result": "API Error: 500"}],
     "other output"),
    ("claude", [CLAUDE_TOOL_PRINTS_IT, {"type": "result", "is_error": False, "result": BLOCK_TEXT}], "other output"),
    ("codex", [{"type": "error", "message": f"Reconnecting... 1/5 ({CODEX_BLOCKED})"},
               {"type": "error", "message": CODEX_BLOCKED},
               {"type": "turn.failed", "error": {"message": CODEX_BLOCKED}}], "blocked"),
    ("codex", [{"type": "item.completed", "item": {"id": "i0", "type": "error", "message": "metadata"}},
               {"type": "item.completed", "item": {"id": "i1", "type": "reasoning", "text": ""}},
               {"type": "turn.failed", "error": {"message": CODEX_BLOCKED}}], "blocked"),
    ("codex", [CODEX_COMMAND, {"type": "turn.failed", "error": {"message": CODEX_BLOCKED}}], "blocked output"),
    # Codex's own retry got past the block: the turn completed.
    ("codex", [{"type": "error", "message": f"Reconnecting... 1/5 ({CODEX_BLOCKED})"},
               {"type": "turn.failed", "error": {"message": CODEX_BLOCKED}}, {"type": "turn.started"},
               {"type": "turn.completed", "usage": {}}], "other"),
    ("codex", [CODEX_COMMAND, {"type": "turn.failed", "error": {"message": "stream disconnected before completion"}}],
     "other output"),
])
def test_attempt_outcome(agent_module, tmp_path, cli, lines, outcome):
    cls = agent_module.ClaudeCodeAgent if cli == "claude" else agent_module.CodexAgent
    log = tmp_path / "attempt.jsonl"
    log.write_text("".join(json.dumps(line) + "\n" for line in lines) + "not json\n")
    helpers = cls(logs_dir=tmp_path)._action_feedback_helpers()
    result = subprocess.run(["bash", "-c", f"{helpers}\n{cli}_attempt_outcome {shlex.quote(str(log))}"],
                            capture_output=True, text=True)
    assert result.stdout.strip() == outcome, result.stderr


# ---- the give-up stop (GIVE_UP_TURNS) ---------------------------------------------------------

def run_turns(agent_cls, tmp_path, turns) -> tuple[str, str]:
    """The GIVE_UP_TURNS bash for continuation turns (tool calls, seconds, blocked); returns the
    runner's stdout and log."""
    init, start, check = agent_cls._give_up_parts(agent_cls, "")
    lines = [f'log_file={shlex.quote(str(tmp_path / "log.jsonl"))}', "is_arc_task=0", ': > "$log_file"', init]
    for n, (tools, seconds, blocked) in enumerate(turns, 1):
        attempt = tmp_path / f"attempt-{n}.jsonl"
        attempt.write_text('{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Bash"}]}}\n' * tools)
        lines += [f"no_action_attempt={n}", f"attempt_log={shlex.quote(str(attempt))}", start,
                  f"content_filter_turn_blocked={int(blocked)}", f"attempt_started=$((attempt_started - {seconds}))",
                  check]
    lines.append("echo survived")
    result = subprocess.run(["bash", "-c", "\n".join(lines)], capture_output=True, text=True)
    assert not result.stderr, result.stderr
    return result.stdout, (tmp_path / "log.jsonl").read_text()


def test_a_blocked_turn_never_counts_as_idle(agent_module, tmp_path):
    """9 idle turns, 3 short blocked ones, 1 idle: the 10th idle turn is the 13th turn."""
    out, log = run_turns(agent_module.ClaudeCodeAgent, tmp_path,
                         [(0, 5, False)] * 9 + [(0, 5, True)] * 3 + [(0, 5, False)] * 2)
    assert "survived" not in out
    event = json.loads(log.strip().splitlines()[-1])["data"]
    assert event["event"] == "agent_gave_up" and event["continuationTurn"] == 13


def test_a_blocked_turn_that_worked_resets_the_idle_count(agent_module, tmp_path):
    out, log = run_turns(agent_module.CodexAgent, tmp_path,
                         [(0, 5, False)] * 9 + [(0, 300, True)] + [(0, 5, False)] * 9)
    assert "survived" in out and "agent_gave_up" not in log


# ---- what the harness sends -------------------------------------------------------------------

@pytest.mark.parametrize("cli", ["claude", "codex"])
@pytest.mark.parametrize("limit", [False, True], ids=["limit-off", "limit-on"])
def test_runners_handle_blocks(cli, limit, tmp_path):
    calls = render("ale-team3", cli=cli, env_overrides=ON if limit else {"RELEASE_LIMIT_RESUME": "0"})
    text = "\n".join(call["command"] for call in calls)
    assert f"{cli}_attempt_outcome()" in text and 'if content_filter_block "$attempt_log_file"; then' in text
    assert "content_filter_stop" in text and '"$content_filter_in_a_row" -ge 5 ]' in text
    for index, call in enumerate(calls):
        path = tmp_path / f"{index}.sh"
        path.write_text(call["command"])
        assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_the_stop_is_final_for_revive(agent_module):
    assert agent_module.CONTENT_FILTER_MAX_BLOCKS == 5
    assert '"event":"content_filter_stop"' in agent_module.CONTENT_FILTER_HELPERS
    assert "content_filter_stop" in revive.FINAL_EVENTS
