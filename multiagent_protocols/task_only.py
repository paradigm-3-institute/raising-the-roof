from __future__ import annotations

import os

from multiagent_protocols.base import AgentSpec, ProtocolPaths, ProtocolSpec

# No coordination protocol: the agent gets the task instruction alone (agent.py writes the prompt from the
# instruction only when the spec's metadata says "task_only"). BrowseComp's single agent runs it; bare_solo, the
# single agent of the other tasks, builds on it.
DESCRIPTION = ("No coordination protocol: each agent receives only the task instruction, and "
               "any communication is the task's own.")


class MultiAgentProtocol:
    name = "task_only"

    def build(
        self,
        instruction: str,
        paths: ProtocolPaths,
        model: str,
    ) -> ProtocolSpec:
        del instruction, paths, model
        raw = os.environ.get("MULTIAGENT_N_AGENT", "1")
        try:
            count = int(raw)
        except ValueError as exc:
            raise ValueError(f"Invalid agent count: {raw!r}") from exc
        if count < 1:
            raise ValueError("Agent count must be >= 1")
        return ProtocolSpec(
            agents=[AgentSpec(id=f"a{index + 1}", prompt=DESCRIPTION) for index in range(count)],
            shared_paths=[],
            metadata={"name": self.name, "agent_count": count, "task_only": True, "description": DESCRIPTION},
        )
