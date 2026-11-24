from __future__ import annotations

from multiagent_protocols import task_only

# One agent with the task instruction alone: the plain single agent, the baseline for the self-organized teams,
# whose prompt adds only a short team intro to the same task text (the matched single agent,
# antiherding_matched --agents 1, follows a solo version of the protocol prompt). has_solo sends it through the
# team runner with one agent, as those solos run, so everything but the prompt is theirs: the session layout
# (sessions-0-a1, agent a1), token budget, limit resumption, resources tool. The prompt is the instruction alone
# (task_only's metadata), the task's single-agent text.


class MultiAgentProtocol(task_only.MultiAgentProtocol):
    name = "bare_solo"
    has_solo = True
