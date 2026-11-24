from __future__ import annotations

import os

from multiagent_protocols import board
from multiagent_protocols.base import AgentSpec, ProtocolPaths, ProtocolSpec


# Self-organized Claude teams: the board protocol's shared goal and no protocol, with Claude Code's own channel
# between agents in place of Codex's message board. Each agent is a `claude -p` session of its own, named by its id;
# the agents message one another with Claude Code's SendMessage and see who is reachable with ListAgents (its
# cross-session messaging). Claude Code delivers a message into the receiver's running turn, at its next tool call,
# and has no tool that waits for one: hence the last sentence.
# The text is board.py's BODY with its paragraph about the board rewritten for this channel and nothing else changed.
# Each agent's Claude Code tells it its own name in one appended system-prompt line (agent.py CLAUDE_PEERS_IDENTITY),
# as each board agent's Codex tells it its path in a developer message. Claude Code only (agent.py,
# claude_peers_journal.py; docs/setups.md).
PEERS = """\
You can message one another, like direct messages in a team Slack: each message goes
to one teammate.
{messaging}"""

BODY = board.BODY.replace(board.BOARD, PEERS)

DESCRIPTION = ("The board protocol with Claude Code's cross-session messaging (SendMessage, ListAgents) in place of "
               "Codex's message board: a shared goal, no protocol. Claude Code only.")


def agent_count() -> int:
    raw = os.environ.get("MULTIAGENT_N_AGENT", "2")
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid agent count: {raw!r}") from exc
    if count < 2:
        raise ValueError("claude_peers needs a team (--agents 2 or more)")
    return count


def messaging(ids: list[str]) -> str:
    """Who the teammates are (their names in Claude Code, the agent ids), the tools, and how to wait."""
    return (f"Your teammates are {', '.join(ids)}. Message a teammate with the SendMessage tool; ListAgents "
            "shows who is reachable. Messages from teammates reach you automatically while you work. If you "
            "want to wait for a teammate, run a short sleep.")


def peers_paragraph(ids: list[str]) -> str:
    return PEERS.format(messaging=messaging(ids))


def peers_metadata(ids: list[str]) -> dict:
    """The team's record in protocol.json: who is in it, by name (the agent ids)."""
    return {"names": list(ids)}


class MultiAgentProtocol:
    name = "claude_peers"
    # agent.py: a team whose agents message one another with Claude Code's cross-session messaging.
    claude_peers = True

    def build(
        self,
        instruction: str,
        paths: ProtocolPaths,
        model: str,
    ) -> ProtocolSpec:
        del instruction, paths, model
        if BODY == board.BODY:
            raise ValueError("board.BODY changed: its paragraph about the board is no longer where claude_peers "
                             "replaces it")
        count = agent_count()
        ids = board.agent_ids(count)
        prompt = BODY.format(n=count, messaging=messaging(ids))
        return ProtocolSpec(
            agents=[AgentSpec(id=agent_id, prompt=prompt) for agent_id in ids],
            shared_paths=[],
            metadata={
                "name": self.name,
                "agent_count": count,
                "description": DESCRIPTION,
                "claude_peers": peers_metadata(ids),
                # As board.py: the text opens with the team.
                "role_intro": "",
            },
        )
