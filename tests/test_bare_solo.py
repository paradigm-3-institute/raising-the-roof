"""bare_solo (multiagent_protocols/bare_solo.py): one agent, the task instruction alone, through the team runner."""

import sys

import pytest

from cli_render import ROOT, render
from test_cli_agents import run_py, scripts

sys.path.insert(0, str(ROOT))
from multiagent_protocols import bare_solo  # noqa: E402
from multiagent_protocols.base import ProtocolPaths  # noqa: E402


def calls(protocol: str, cli: str = "codex") -> list[dict]:
    return render("ale-single", cli=cli, model="gpt-6.1-sol",
                  env_overrides={"MULTIAGENT_PROTOCOL": f"multiagent_protocols.{protocol}:MultiAgentProtocol"})


def test_protocol(monkeypatch):
    monkeypatch.setenv("MULTIAGENT_N_AGENT", "1")
    spec = bare_solo.MultiAgentProtocol().build(instruction="x", paths=ProtocolPaths.default(), model="m")
    assert [a.id for a in spec.agents] == ["a1"] and spec.metadata["task_only"] is True
    assert spec.metadata["name"] == "bare_solo" and bare_solo.MultiAgentProtocol.has_solo


@pytest.mark.parametrize("cli", ["codex", "claude"])
def test_one_agent_gets_the_instruction_alone_from_the_team_runner(cli):
    found = scripts(calls("bare_solo", cli))
    assert len(found) == 2, "the team runner (run-agent.sh), as for the protocol solos, not run_single"
    runner = found[1]
    assert 'printf %s "$instruction" > "$prompt_file"' in runner
    # The protocol solo it stands next to: the same runner, with the solo protocol text.
    matched = scripts(calls("antiherding_matched", cli))
    assert len(matched) == 2 and "solo work protocol" in matched[1]
    assert "solo work protocol" not in runner and "single-session" not in found[0]


def test_run_py():
    cmd, env, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", "codex", "--model", "m", "--protocol", "bare_solo")
    assert env["MULTIAGENT_PROTOCOL"] == "multiagent_protocols.bare_solo:MultiAgentProtocol"
    with pytest.raises(ValueError, match="bare_solo is one agent"):
        run_py("ale-bench", "--problem", "ahc014", "--cli", "codex", "--model", "m", "--protocol", "bare_solo", "--agents", "2")
