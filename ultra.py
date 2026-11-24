"""Ultra mode (run.py --protocol ultra --agents N; docs/setups.md): one Codex agent that delegates to
its own helpers, at the run's own effort and with a controllable team size.

Codex's `--effort ultra` (gpt-5.6-sol and later) is two things in Codex 0.162.1: the root gets the
proactive multi-agent mode message instead of "only when asked" (core/src/session/multi_agents.rs,
effective_multi_agent_mode), and every request goes out at the model's multi_agent_reasoning_effort, else
max (protocol/src/openai_models/reasoning_effort.rs). This mode keeps the first and not the second:

- one `codex exec` process, as for one agent: its helpers are threads of that process (spawn_agent), each
  with its own rollout in the agent's codex-home/sessions, all under the agent's one token budget,
  `resources` command, usage-limit waits and keeper unit;
- the mode message: Codex's own proactive text (PROACTIVE_MODE_TEXT, verbatim from
  prompts/src/model_messages/multi_agent.rs), sent through features.multi_agent_v2.multi_agent_mode_hint_text,
  which Codex sends in place of the effort-chosen text (MultiAgentMode::Custom); the role messages stay
  Codex's own (`/root` and the sub-agent's), as in every Sol run of the study;
- the effort: the run's --effort for the root and every helper. A helper starts at its parent's effective
  effort (core/src/agent/child_config.rs, build_agent_spawn_config), forked or fresh;
  agents.default_subagent_reasoning_effort, set to the same effort, also covers a spawn that names another
  model without an effort (which would otherwise take that model's default);
- the cap: features.multi_agent_v2.max_concurrent_threads_per_session = N, which counts the root (the role
  message says "up to N agents can be active at once, including you"; effective_agent_max_threads is N - 1
  sub-agents). The limit is the whole tree's: every thread of the process shares one execution limiter and
  one residency count (core/src/agent/control/execution.rs, residency.rs), helpers' own sub-agents
  included; multi-agent v2 has no depth limit. An idle helper is unloaded to make room for a new one.
- no coordination protocol: the prompt is CAP_LINE and WAIT_LINE, then the task's own text in its solo form, with the
  per-agent-copy tweaks of the task's adapter (`workcopy NAME`: the helpers share one $AGENT_ID).

The adapters with per-agent copies use workcopy_script and submit_script for their `workcopy NAME` and
`submit` (from inside a copy) commands, so a submission records the copy's NAME as its agent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re

PROTOCOL = "ultra"
# run.py sets it to N (the thread cap, the root included) for agent.py; MULTIAGENT_N_AGENT is 1.
ENV = "RELEASE_ULTRA_AGENTS"
# The Codex this mode was read against (multi_agent_mode_hint_text, the v2 thread cap and residency).
MIN_CODEX = (0, 162, 1)
# Codex 0.162.1, prompts/src/model_messages/multi_agent.rs, PROACTIVE_MULTI_AGENT_MODE_TEXT, verbatim.
PROACTIVE_MODE_TEXT = (
    "Proactive multi-agent delegation is active. Any earlier developer instruction requiring an explicit user "
    "request before spawning sub-agents no longer applies. This mode remains active until a later multi-agent "
    "mode developer message changes it. User requests override this hint.\n\nIf at any point you can parallelize "
    "work by delegating tasks to another agent (no matter if you are root or subagent), you should do so using "
    "collaboration tools if it could save time or improve quality.")
# The prompt's first line.
CAP_LINE = "At most {n} agents, you included, can work on this task at the same time."
# The prompt's second line: `codex exec` ends the process, and with it every helper still
# working, when the root's turn ends; the harness's continuation brings back only the root. A fact about the
# setup, like the cap, not a coordination rule.
WAIT_LINE = "Your helpers stop when you end your turn, so wait for them to finish before you end it."
# The end of the `resources` paragraph (agent.py RESOURCES_PROMPT) in this mode.
RESOURCES_SUFFIX = "; it counts your helpers too"
# The ultra.json record agent.py writes in the agent directory: the run's mode, for reading the run back.
RECORD = "ultra.json"
# A working copy's NAME: what fswe.py's agent ids and the scored-artifact judge's agent_id both accept.
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def check_agents(value) -> int:
    """N: the thread cap, the root included; at least 2 (one helper)."""
    try:
        count = int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{ENV} must be a whole number of agents: {value!r}") from exc
    if count < 2:
        raise ValueError(f"ultra mode needs room for a helper: --agents 2 or more (the root included), not {count}")
    return count


def agents_from_env() -> int | None:
    """N from the harness environment, or None outside ultra mode."""
    raw = os.environ.get(ENV, "").strip()
    return check_agents(raw) if raw else None


def cap_line(agents: int) -> str:
    return CAP_LINE.format(n=check_agents(agents))


def prompt(instruction: str, agents: int) -> str:
    """The cap line and the wait line, then the task's text."""
    return cap_line(agents) + "\n" + WAIT_LINE + "\n\n" + instruction


def codex_config(base: str, agents: int) -> str:
    """The harness's config.toml (agent.py CODEX_CONFIG_TOML) for ultra mode, and nothing else changed:
    multi-agent v2 on by name (gpt-6 and gpt-5.6-sol run it already; `multi_agent_v2 = false` leaves the
    model's own version, and a table cannot sit next to that key), the thread cap, and the proactive mode
    message. The edit fails loudly if the base changes."""
    old = "multi_agent_v2 = false\n"
    if base.count(old) != 1:
        raise ValueError(f"CODEX_CONFIG_TOML changed; the ultra edit no longer applies: {old!r}")
    return base.replace(old, "") + (
        "\n[features.multi_agent_v2]\n"
        "enabled = true\n"
        f"max_concurrent_threads_per_session = {check_agents(agents)}\n"
        f"multi_agent_mode_hint_text = {json.dumps(PROACTIVE_MODE_TEXT)}\n")


def record(agents: int, effort: str) -> dict:
    """What the run's ultra.json says (the mode text by digest)."""
    return {"protocol": PROTOCOL, "agents": check_agents(agents), "effort": effort,
            "multi_agent_mode_hint_sha256": hashlib.sha256(PROACTIVE_MODE_TEXT.encode()).hexdigest()}


# ---- the per-agent-copy commands --------------------------------------------------------------------------

# A NAME the scripts refuse (sh patterns): empty, hidden, an option, a staging copy, or other characters.
_BAD_NAME = '""|.*|-*|_*|*.partial|*[!A-Za-z0-9_-]*'


def workcopy_script(work_root: str, run: str, about: str) -> str:
    """`workcopy NAME`: RUN (a shell line using "$name") creates the copy WORK_ROOT/NAME once and prints its
    path. Without a usable NAME it says what to do and lists the copies there are."""
    return f"""\
#!/bin/sh
# Ultra mode: {about}
# workcopy NAME creates (once) and prints the working copy {work_root}/NAME.
usage() {{
  echo 'usage: workcopy NAME   (every agent, you and each helper, works in its own copy: give each its own NAME of letters, digits, - and _)' >&2
  for d in {work_root}/*/; do
    [ -d "$d" ] || continue
    d="${{d%/}}"; d="${{d##*/}}"
    case "$d" in *.partial) continue ;; esac
    echo "  existing copy: $d" >&2
  done
  exit 2
}}
[ "$#" -eq 1 ] || usage
name="$1"
case "$name" in {_BAD_NAME}) usage ;; esac
[ "${{#name}}" -le 64 ] || usage
{run}
"""


def submit_script(work_root: str, run: str, about: str) -> str:
    """`submit ...` run from inside a copy WORK_ROOT/NAME: RUN (a shell line using "$name" and "$@") submits
    that copy as NAME's."""
    return f"""\
#!/bin/sh
# Ultra mode: {about}
# submit, run from inside a working copy {work_root}/NAME, submits that copy as NAME's.
usage() {{
  echo "[submit] run submit from inside your copy: cd \\"\\$(workcopy NAME)\\" first, then submit there ({work_root}/NAME)" >&2
  exit 2
}}
here="$(pwd -P)"
case "$here/" in
  {work_root}/*/) name="${{here#{work_root}/}}"; name="${{name%%/*}}" ;;
  *) name="" ;;
esac
case "$name" in {_BAD_NAME}) usage ;; esac
[ "${{#name}}" -le 64 ] || usage
{run}
"""
