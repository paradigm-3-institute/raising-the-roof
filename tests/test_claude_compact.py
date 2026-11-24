"""Claude Code compacts at about 244k tokens, as Codex does (run.py --claude-compact-window, the agent
kwarg claude_compact_window, CLAUDE_CODE_AUTO_COMPACT_WINDOW).

Claude Code 2.1.289 compacts when a conversation reaches its auto-compact window minus 33k tokens
(20k for the summary, a 13k buffer); the window is the model's (1M) unless
CLAUDE_CODE_AUTO_COMPACT_WINDOW sets it. The end-to-end tests run the pinned Claude Code against
tests/limit_mock_api.py, whose replies report the input tokens a plan step names.
"""

from __future__ import annotations

import importlib
import json
import sys

import pytest

from cli_render import CASES, ROOT, render
from test_limit_resume import continuation_text, needs_claude, run_agent

sys.path.insert(0, str(ROOT))
import run  # noqa: E402

WINDOW = run.CLAUDE_COMPACT_WINDOW
LINE = f"    CLAUDE_CODE_AUTO_COMPACT_WINDOW={WINDOW} \\"


def compactions(run_result) -> list[dict]:
    """Claude Code's compact_boundary events in the agent's attempt logs."""
    found = []
    for path in sorted(run_result.session.glob("claude-attempt-*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "system" and event.get("subtype") in ("compact_boundary", "status") and (
                    event.get("subtype") == "compact_boundary" or event.get("status") == "compacting"):
                found.append(event)
    return found


# ---- end to end -------------------------------------------------------------------------------

@needs_claude
@pytest.mark.parametrize("window, tokens, compacts", [
    (WINDOW, 245_000, True),
    (WINDOW, 243_000, False),
    (None, 250_000, False),  # Claude Code's own window: 1M, compacting at about 967k
    ("off", 250_000, False),
], ids=["window-above-244k", "window-below-244k", "no-window", "off"])
def test_compacts_above_244k_only(tmp_path, window, tokens, compacts):
    """Four tool calls, then a reply that reports TOKENS input tokens: with the window, Claude Code
    compacts before its next request if that is above 277000 - 33000 = 244000, and not below."""
    plan = [*[{"type": "tool", "command": f"echo step{i}"} for i in range(4)],
            {"type": "tool", "command": "echo big", "input_tokens": tokens},
            *[{"type": "text", "text": f"done-{i}"} for i in range(3)]]
    result = run_agent(tmp_path, "claude", plan, agent_kwargs={} if window is None else {"claude_compact_window": window})
    assert result.results[1].returncode == 0, result.results[1].stdout[-2000:] + result.results[1].stderr[-2000:]
    runner = (tmp_path / "agent/multiagent-collab/run-agent.sh").read_text()
    assert (LINE in runner) == (window == WINDOW)
    events = compactions(result)
    boundaries = [e for e in events if e["subtype"] == "compact_boundary"]
    if not compacts:
        assert events == []
        assert max(len(r["body"]["messages"]) for r in result.mock.main()) >= 17  # never shortened
        return
    [boundary] = boundaries
    assert boundary["compact_metadata"]["trigger"] == "auto"
    assert 244_000 <= boundary["compact_metadata"]["pre_tokens"] < 246_000
    sizes = [len(r["body"]["messages"]) for r in result.mock.main()]
    assert sizes[-2] < sizes[5]  # the turn after the compaction sends the short conversation
    # The run goes on as usual: the continuation is sent once, after the compaction.
    last = result.mock.main()[-1]["body"]["messages"]
    texts = [b.get("text", "") for m in last if m["role"] == "user"
             for b in (m["content"] if isinstance(m["content"], list) else [{"text": m["content"]}])]
    assert sum(text.strip() == continuation_text() for text in texts) == 1


# ---- what the harness sends -------------------------------------------------------------------

@pytest.mark.parametrize("case", CASES)
def test_claude_commands_change_only_by_the_env_var(case):
    """With the window, each Claude Code runner gains exactly one line, the variable, in invoke_claude;
    without it (or "off"), the commands are those of a run before this option."""
    plain = render(case, cli="claude")
    assert render(case, cli="claude", agent_kwargs={"claude_compact_window": "off"}) == plain
    windowed = render(case, cli="claude", agent_kwargs={"claude_compact_window": WINDOW})
    assert len(windowed) == len(plain)
    changed = 0
    for before, after in zip(plain, windowed):
        assert before["env"] == after["env"] and before["user"] == after["user"]
        if before["command"] == after["command"]:
            continue
        old, new = before["command"].split("\n"), after["command"].split("\n")
        added = [line for line in new if line not in old]
        assert [line.strip() for line in added] == [LINE.strip()]  # (the team runner indents it more)
        new.remove(added[0])
        assert new == old
        changed += 1
    assert changed == 1  # the call that holds the runner


@pytest.mark.parametrize("value", [99_999, "abc", True, -1, "277000.5"])
def test_bad_windows_are_refused(value, monkeypatch):
    monkeypatch.setenv("RELEASE_CONTINUATION", "continue")
    monkeypatch.setenv("RELEASE_TASK", "ale-bench")
    sys.modules.pop("agent", None)
    agent = importlib.import_module("agent")
    with pytest.raises(ValueError, match="claude_compact_window"):
        agent.ClaudeCodeAgent(logs_dir=ROOT / ".cache" / "render-logs", model_name="m", claude_compact_window=value)


def test_claude_bash_commands_may_run_an_hour(monkeypatch):
    """Every Claude Code runner raises the Bash tool's cap for a command with its own timeout from 10 minutes to
    1 hour; the 2-minute default (BASH_DEFAULT_TIMEOUT_MS) is left alone."""
    monkeypatch.setenv("RELEASE_CONTINUATION", "continue")
    monkeypatch.setenv("RELEASE_TASK", "ale-bench")
    sys.modules.pop("agent", None)
    agent = importlib.import_module("agent")
    made = agent.ClaudeCodeAgent(logs_dir=ROOT / ".cache" / "render-logs", model_name="m")
    lines = made._invoke_lines("", "code")
    assert "    BASH_MAX_TIMEOUT_MS=3600000 \\" in lines
    assert not any("BASH_DEFAULT_TIMEOUT_MS" in line for line in lines)


def test_harbor_kwargs_reach_the_agent(monkeypatch, tmp_path):
    """Harbor's own path for run.py's --ak: `harbor run` parses claude_compact_window=277000 to an int
    (=off to a string) into the agent config's kwargs, which config.json records, and its agent
    factory passes them to the agent's constructor."""
    from harbor.agents.factory import AgentFactory
    from harbor.cli.utils import parse_kwargs
    from harbor.models.trial.config import AgentConfig
    monkeypatch.setenv("RELEASE_CONTINUATION", "continue")
    monkeypatch.setenv("RELEASE_TASK", "ale-bench")
    sys.modules.pop("agent", None)
    for argument, expected in ((f"claude_compact_window={WINDOW}", WINDOW), ("claude_compact_window=off", None)):
        config = AgentConfig(import_path="agent:ClaudeCodeCollaborationAgent", model_name="claude-opus-5-5",
                             kwargs=parse_kwargs([argument]))
        assert json.loads(config.model_dump_json())["kwargs"] == {"claude_compact_window": expected or "off"}
        made = AgentFactory.create_agent_from_config(config, logs_dir=tmp_path)
        assert made.claude_compact_window == expected
        assert any("CLAUDE_CODE_AUTO_COMPACT_WINDOW" in line for line in made._invoke_lines("", "code")) == bool(expected)


# ---- run.py -----------------------------------------------------------------------------------

def run_py(*argv: str):
    return run.prepare(run.parser().parse_args([*argv, "--dry-run"]))


def agent_kwargs(cmd: list[str]) -> list[str]:
    return [cmd[i + 1] for i, word in enumerate(cmd) if word == "--ak"]


def test_run_py_default_and_option():
    assert WINDOW == 277_000 and WINDOW - 20_000 - 13_000 == 244_000
    cmd, _, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", "claude", "--model", "m")
    assert agent_kwargs(cmd) == ["claude_compact_window=277000"]
    cmd, _, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", "claude", "--model", "m", "--claude-compact-window", "off")
    assert agent_kwargs(cmd) == ["claude_compact_window=off"]
    cmd, _, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", "claude", "--model", "m", "--claude-compact-window", "400000")
    assert agent_kwargs(cmd) == ["claude_compact_window=400000"]
    with pytest.raises(SystemExit):
        run.parser().parse_args(["ale-bench", "--problem", "ahc014", "--cli", "claude", "--model", "m", "--claude-compact-window", "50000"])


@pytest.mark.parametrize("cli", ["codex"])
def test_run_py_other_clis_are_unchanged(cli):
    cmd, _, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", cli, "--model", "m")
    assert agent_kwargs(cmd) == []
    with pytest.raises(ValueError, match="only supported with --cli claude"):
        run_py("ale-bench", "--problem", "ahc014", "--cli", cli, "--model", "m", "--claude-compact-window", "300000")
