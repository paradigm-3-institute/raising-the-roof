from __future__ import annotations

import os

from multiagent_protocols.base import ProtocolPaths, ProtocolSpec


# Ultra mode (ultra.py; docs/setups.md): one Codex agent that delegates to its own helpers, at the run's own
# effort, with at most N agents, the root included. No coordination protocol: run.py runs one agent
# (MULTIAGENT_N_AGENT=1, so agent.py gives it the task's text in its solo form) and passes N in
# RELEASE_ULTRA_AGENTS; agent.py adds the cap line and Codex's proactive mode message. A team of processes never
# runs this protocol.
DESCRIPTION = ("One Codex agent that delegates to its own helpers (Codex's proactive multi-agent mode at the run's "
               "effort), at most N agents at once, the root included; no coordination protocol. Codex only.")


class MultiAgentProtocol:
    name = "ultra"
    # agent.py: one `codex exec` whose helpers are its own threads, N of them at most (RELEASE_ULTRA_AGENTS).
    ultra = True
    # One agent gets the task's text alone (agent.py run_single), as without a protocol.
    has_solo = False

    def build(
        self,
        instruction: str,
        paths: ProtocolPaths,
        model: str,
    ) -> ProtocolSpec:
        del instruction, paths, model
        raise ValueError("the ultra protocol runs one Codex agent with its own helpers (MULTIAGENT_N_AGENT=1, "
                         f"RELEASE_ULTRA_AGENTS=N), not a team of {os.environ.get('MULTIAGENT_N_AGENT')} agents: "
                         "use run.py --protocol ultra --agents N")
