from __future__ import annotations

import os

from multiagent_protocols.base import AgentSpec, ProtocolPaths, ProtocolSpec


# Self-organized teams on Codex's message board: give the agents a shared goal and primitive tools, and let them
# work out how to communicate (as Noam Brown described OpenAI's multi-agent setup on the Dwarkesh Podcast,
# 2026-09-17). Its {messaging} names the team's paths on the board and where the board's tools are. The board is
# Codex's own message board, its tools and their descriptions (agent.py, message_board.py; docs/setups.md), so it
# runs with Codex agents only. Each agent's Codex tells it which path is its own; the prompt is the same for all.
# No slots, no shared files, no rules: a difference from antiherding_matched measures the whole prompt and the
# channel together.
BOARD = """\
The team shares a message board, like a team Slack: channels, threads and posts that
everyone can read, search and subscribe to.
{messaging}"""

BODY = """\
You are one of {n} agents working on this task at the same time, in this container, all
with this same prompt. You are one team: the team is scored on its best result, so
whatever any one of you achieves counts for all of you.

""" + BOARD + """

How you split the work, what you share, and when you talk is up to you. Work it out
among yourselves.
"""

DESCRIPTION = ("Noam Brown's setup: a shared goal and Codex's message board, no "
               "protocol. Codex only.")


def agent_count() -> int:
    raw = os.environ.get("MULTIAGENT_N_AGENT", "2")
    try:
        count = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid agent count: {raw!r}") from exc
    if count < 2:
        raise ValueError("the message board needs a team (--agents 2 or more)")
    return count


def agent_ids(count: int) -> list[str]:
    return [f"a{index + 1}" for index in range(count)]


def board_path(agent_id: str) -> str:
    """An agent's path on the board (message_board.py)."""
    return f"/root/{agent_id}"


def messaging(ids: list[str]) -> str:
    """The board guide: the teammates' paths on the board and the board's tools."""
    return (f"Your teammates are {', '.join(board_path(agent_id) for agent_id in ids)}. "
            "The board's tools (post, read_thread, subscribe, ...) are collaboration tools.")


def board_paragraph(ids: list[str]) -> str:
    return BOARD.format(messaging=messaging(ids))


def board_metadata(ids: list[str]) -> dict:
    """What agent.py reads to run the board: who is on it. The board is the agents' only channel to each
    other: Codex's direct messages (send_message, followup_task) reach only an agent's own sub-agents, in its
    own process, as without the board."""
    return {"members": [board_path(agent_id) for agent_id in ids]}


class MultiAgentProtocol:
    name = "board"
    # agent.py: a team whose agents share Codex's message board.
    message_board = True

    def build(
        self,
        instruction: str,
        paths: ProtocolPaths,
        model: str,
    ) -> ProtocolSpec:
        del instruction, paths, model
        count = agent_count()
        ids = agent_ids(count)
        prompt = BODY.format(n=count, messaging=messaging(ids))
        return ProtocolSpec(
            agents=[AgentSpec(id=agent_id, prompt=prompt) for agent_id in ids],
            shared_paths=[],
            metadata={
                "name": self.name,
                "agent_count": count,
                "description": DESCRIPTION,
                "message_board": board_metadata(ids),
                # The text opens with the team; the harness's "Follow this coordination protocol exactly"
                # would prescribe what it leaves to the agents.
                "role_intro": "",
            },
        )
