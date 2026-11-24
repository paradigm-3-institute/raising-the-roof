from __future__ import annotations

from multiagent_protocols import antiherding
from multiagent_protocols.base import AgentSpec, ProtocolPaths, ProtocolSpec


# Anti-herding teams with a matched single agent. A team gets the antiherding
# prompt byte for byte. A single agent gets the same text minus what only makes
# sense with peers: it still claims the one slot and keeps the same approach,
# findings, disconfirmation, and board files, now as private notes. The edits
# below are the only differences, and each one fails loudly if antiherding.BODY
# changes under it.

# Sections about adopting peers' approaches, dropped from the first line up to
# (not including) the second.
DROPPED_SECTIONS = ("5. ADOPTION REQUIRES A BAR:", "7. BOARD:")

# (team line, solo line); None drops the line.
EDITS = (
    ("{n} agents share this container and solve the same task together. Identical prompt",
     None),
    ("for all agents. This protocol is designed to prevent premature convergence.",
     "This protocol is designed to prevent premature convergence."),
    ("Shared paths:", "Paths:"),
    ("  - adoption log:        {adoptions}", None),
    ("   If a lower-numbered slot has the same approach, change yours. Your job is not", None),
    ("   to agree early; your job is to cover a different part of the search space.", None),
    ("3. SHARE WITHOUT HERDING:", "3. RECORD FINDINGS:"),
    ("   Include evidence and cost. Label weak claims as weak. Do not write persuasive",
     "   Include evidence and cost. Label weak claims as weak."),
    ("   prose that asks everyone to copy you unless the evidence is strong.", None),
    ("   Spend part of your work trying to falsify the leading shared hypothesis or",
     "   Spend part of your work trying to falsify the leading hypothesis or"),
    ("7. BOARD:", "5. BOARD:"),
    ("- Read slot approaches first, then do independent work.", None),
    ("- Only converge when the adoption bar is met.", None),
    ("- Every turn must run a command: claim, build, test, disconfirm, publish, adopt, or update.",
     "- Every turn must run a command: claim, build, test, disconfirm, publish, or update."),
)


def solo_template(team: str = antiherding.BODY) -> str:
    lines = team.split("\n")
    start, stop = (lines.index(marker) for marker in DROPPED_SECTIONS)
    del lines[start:stop]
    for old, new in EDITS:
        if lines.count(old) != 1:
            raise ValueError(f"antiherding.BODY changed; this edit no longer applies: {old!r}")
        index = lines.index(old)
        if new is None:
            del lines[index]
        else:
            lines[index] = new
    return "\n".join(lines)


SOLO_BODY = solo_template()


def format_args(paths: ProtocolPaths, count: int) -> dict[str, object]:
    """The values antiherding.MultiAgentProtocol formats BODY with."""
    base = f"{paths.comm_dir}/anti-herding"
    return {
        "n": count,
        "nm1": count - 1,
        "base": base,
        "slots": f"{base}/slots",
        "findings": f"{base}/findings.log",
        "disconfirm": f"{base}/disconfirmations.log",
        "adoptions": f"{base}/adoptions.log",
        "board": f"{base}/leaderboard",
    }


class MultiAgentProtocol(antiherding.MultiAgentProtocol):
    name = "antiherding_matched"
    # A single agent runs this protocol's solo prompt instead of the bare task.
    has_solo = True

    def build(
        self,
        instruction: str,
        paths: ProtocolPaths,
        model: str,
    ) -> ProtocolSpec:
        spec = super().build(instruction=instruction, paths=paths, model=model)
        if len(spec.agents) > 1:
            return ProtocolSpec(
                agents=spec.agents,
                shared_paths=spec.shared_paths,
                metadata={**spec.metadata, "name": self.name},
            )
        return ProtocolSpec(
            agents=[AgentSpec(id="a1", prompt=SOLO_BODY.format(**format_args(paths, 1)))],
            shared_paths=[],
            metadata={
                "name": self.name,
                "agent_count": 1,
                "description": (
                    "The antiherding prompt minus the parts about peers; the agent keeps "
                    "the same notes files privately."
                ),
            },
        )
