"""A team's setup goes out in commands that each fit in one argument of execve (agent.setup_commands): a team of 8
on a long task can otherwise fail with "Argument list too long"."""
from __future__ import annotations

import asyncio
import importlib
import itertools
import os
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from cli_render import BASE_ENV, ROOT, FakeEnvironment

sys.path.insert(0, str(ROOT))
import agent as agent_module  # noqa: E402

HEAD = "set -euo pipefail; "


def test_parts_that_fit_stay_one_command():
    parts = ["mkdir -p /a", "printf %s 'x' > /a/b", "chmod 700 /a/b"]
    assert agent_module.setup_commands(parts) == [HEAD + " && ".join(parts)]
    assert agent_module.setup_commands([]) == []


def test_parts_are_cut_in_order_under_the_limit():
    parts = [f"printf %s {'x' * 30} > /f{i}" for i in range(20)]
    commands = agent_module.setup_commands(parts, limit=200)
    assert len(commands) > 1
    assert all(len(c.encode()) <= 200 and c.startswith(HEAD) for c in commands)
    assert [p for c in commands for p in c[len(HEAD):].split(" && ")] == parts


def test_a_part_over_the_limit_is_refused():
    with pytest.raises(ValueError, match="cannot fit"):
        agent_module.setup_commands(["x" * 300], limit=200)


def render_team(agents: int, instruction: str) -> list[dict]:
    env = {**BASE_ENV, "RELEASE_TASK": "ale-bench", "MULTIAGENT_N_AGENT": str(agents),
           "MULTIAGENT_PROTOCOL": "multiagent_protocols.antiherding_matched:MultiAgentProtocol",
           "RELEASE_CONTINUATION": (ROOT / "benchmarks" / "ale-bench" / "continuation.txt").read_text().rstrip("\n"),
           "RELEASE_LIMIT_RESUME": "1", "RELEASE_HARNESS_GUARD": "1", "RELEASE_RESOURCES_TOOL": "1",
           "RELEASE_TOKEN_BUDGET": "1000000"}
    counter = itertools.count()
    with mock.patch.dict(os.environ, env, clear=False), \
         mock.patch("secrets.token_urlsafe", lambda n: f"tok{next(counter)}"):
        sys.modules.pop("agent", None)
        module = importlib.import_module("agent")
        logs = ROOT / ".cache" / "render-logs"
        logs.mkdir(parents=True, exist_ok=True)
        team = module.ClaudeCodeCollaborationAgent(logs_dir=logs, model_name="claude-opus-5-5",
                                                   claude_compact_window=277000)
        environment = FakeEnvironment()

        async def go() -> None:
            await team.install(environment)
            await team.run(instruction, environment, SimpleNamespace())

        asyncio.run(go())
    return environment.calls


def test_a_team_of_8_on_a_long_task_fits_in_execve():
    # About the size of AutoResearchExam coreset's instruction (6.5 KiB), with quotes that shell quoting expands.
    instruction = ("Pick 240 rows. The agent's selector can't see the slice's attribute; it's scored on the worst one.\n"
                   * 70)
    calls = render_team(8, instruction)
    assert max(len(call["command"].encode()) for call in calls) < 120000
    setup = "\n".join(call["command"] for call in calls)
    for index in range(8):
        assert f"/logs/agent/sessions-{index}-a{index + 1}/task-instruction.txt" in setup
