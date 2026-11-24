"""Render the commands a CLI agent sends to its container, without Docker.

Each case sets the harness environment that run.py would export, runs the
agent's install() and run() against a fake environment, and records every exec
call. Session tokens are fixed so the output is deterministic.
"""

from __future__ import annotations

import asyncio
import importlib
import itertools
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]

BASE_ENV = {
    "RELEASE_CONTINUATIONS": "1000",
    "MULTIAGENT_AGENT_EFFORT": "max",
    "CLAUDE_CODE_OAUTH_TOKEN": "fake-claude-oauth-token",
}

# Stand-in cases: ALE-Bench runs in a plain container (no harness tools of its own), so these exercise the generic
# runner paths. single: no solo prompt, so run_single; solo: a protocol with a solo prompt, through the team runner.
CASES = {
    "ale-single": {"task": "ale-bench", "agents": 1, "protocol": "antiherding"},
    "ale-solo": {"task": "ale-bench", "agents": 1, "protocol": "antiherding_matched"},
    "ale-team3": {"task": "ale-bench", "agents": 3, "protocol": "antiherding"},
    "ale-team4": {"task": "ale-bench", "agents": 4, "protocol": "antiherding"},
}

CLASSES = {
    "claude": "ClaudeCodeCollaborationAgent",
    "codex": "CodexCollaborationAgent",
}

INSTRUCTION = "Solve the task.\nIt's a test with 'quotes' and $dollars."


class FakeEnvironment:
    default_user = None

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        self.calls.append({"user": user, "env": dict(sorted((env or {}).items())),
                           "command": command})
        return SimpleNamespace(return_code=0, stdout="", stderr="")


def render(case: str, cli: str = "codex", model: str = "gpt-6.1-sol",
           env_overrides: dict[str, str] | None = None, agent_kwargs: dict | None = None) -> list[dict]:
    spec = CASES[case]
    env = {
        **BASE_ENV,
        "RELEASE_TASK": spec["task"],
        "MULTIAGENT_N_AGENT": str(spec["agents"]),
        "MULTIAGENT_PROTOCOL": f"multiagent_protocols.{spec['protocol']}:MultiAgentProtocol",
        "RELEASE_CONTINUATION": (ROOT / "benchmarks" / spec["task"] / "continuation.txt").read_text().rstrip("\n"),
        **spec.get("extra", {}),
        **(env_overrides or {}),
    }
    counter = itertools.count()
    with mock.patch.dict(os.environ, env, clear=False), \
         mock.patch("secrets.token_urlsafe", lambda n: f"tok{next(counter)}"):
        sys.modules.pop("agent", None)
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        module = importlib.import_module("agent")
        cls = getattr(module, CLASSES[cli])
        logs = ROOT / ".cache" / "render-logs"
        logs.mkdir(parents=True, exist_ok=True)
        agent = cls(logs_dir=logs, model_name=model, **(agent_kwargs or {}))
        environment = FakeEnvironment()

        async def go() -> None:
            await agent.install(environment)
            await agent.run(INSTRUCTION, environment, SimpleNamespace())

        asyncio.run(go())
    return environment.calls
