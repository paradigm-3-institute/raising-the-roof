from __future__ import annotations

import os
from pathlib import Path

from multiagent_protocols import board
from multiagent_protocols.base import AgentSpec, ProtocolPaths, ProtocolSpec


# Self-organized mixed teams: the board protocol's shared goal and no protocol, with a team chat run
# by upstream SwarmChat's own code (vendor/swarmchat, github.com/Pranjal2041/swarmchat, unmodified; agent.py and
# swarmchat_glue.py run it in the container, as upstream's own team runner does) in place of Codex's message board or
# Claude Code's peer messaging. The chat is a shell command, `swarmchat`, the same for Claude Code and Codex agents,
# and each agent's new messages are pushed into its conversation after its tool calls (a PostToolUse hook in both
# CLIs). The text is board.py's BODY with its paragraph about the board rewritten for this channel and nothing else
# changed: the teammates' ids, then upstream's own extras/programbench/prompts/chat.md, word for word. Each agent's CLI
# tells it its own name in one system-level line (agent.py SWARMCHAT_IDENTITY), as board and claude_peers do. Claude
# Code, Codex and mixed teams (agent.py; docs/setups.md).
CHAT = """\
The team shares a chat, like a team Slack: channels, direct messages and threads.
{messaging}"""

BODY = board.BODY.replace(board.BOARD, CHAT)

# Upstream's prompts/chat.md, its lines joined (board.py wraps its own text; the words are upstream's).
UPSTREAM_CHAT = " ".join((Path(__file__).resolve().parents[1] / "vendor/swarmchat/extras/programbench/prompts/chat.md")
                         .read_text().split())

DESCRIPTION = ("The board protocol with upstream SwarmChat's team chat (the `swarmchat` command, new messages pushed "
               "after tool calls) in place of Codex's message board: a shared goal, no protocol. Claude Code, Codex and "
               "mixed teams.")


def agent_count() -> int:
    raw = os.environ.get("MULTIAGENT_N_AGENT", "2")
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid agent count: {raw!r}") from exc
    if count < 2:
        raise ValueError("swarmchat needs a team (--agents 2 or more)")
    return count


def messaging(ids: list[str]) -> str:
    """Who the teammates are, then upstream's own words about the chat (prompts/chat.md)."""
    return f"Your teammates are {', '.join(ids)}. {UPSTREAM_CHAT}"


def chat_metadata(ids: list[str]) -> dict:
    """What agent.py reads to run the chat: who is in it, and the channel everyone starts in."""
    return {"members": list(ids), "channels": {"general": list(ids)}}


class MultiAgentProtocol:
    name = "swarmchat"
    # agent.py: a team whose agents share upstream SwarmChat's chat (swarmchat_glue.py).
    swarmchat = True

    def build(
        self,
        instruction: str,
        paths: ProtocolPaths,
        model: str,
    ) -> ProtocolSpec:
        del instruction, paths, model
        if BODY == board.BODY:
            raise ValueError("board.BODY changed: its paragraph about the board is no longer where swarmchat "
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
                "swarmchat": chat_metadata(ids),
                # As board.py: the text opens with the team.
                "role_intro": "",
            },
        )
