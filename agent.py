# Claude Code and Codex run through the same runner; each CLI supplies only its
# install step, invocation, session-id parsing, and retry classification.
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import pathlib
import posixpath
import secrets
import shlex
import textwrap
import time
from typing import Any

from harbor.agents.installed.base import BaseInstalledAgent, NonZeroAgentExitCodeError, with_prompt_template
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trial.paths import EnvironmentPaths

from multiagent_protocols.base import ProtocolPaths, build_protocol, protocol_to_dict
from token_budget import METRICS as BUDGET_METRICS, session_totals, summed, weighted_tokens
import harness_guard
import lockstep
import message_board
import team_hold
import ultra

DEFAULT_PROTOCOL = "multiagent_protocols.antiherding:MultiAgentProtocol"
DEFAULT_CONTAINER_CLAUDE_BINARY = "/mnt/release-claude/claude"
CLAUDE_BINARY_ENV = "META_DUO_CONTAINER_CLAUDE_BINARY"
DEFAULT_CONTAINER_CODEX_BINARY = "/mnt/release-codex/bin/codex"
CODEX_BINARY_ENV = "META_DUO_CONTAINER_CODEX_BINARY"
CONTAINER_CODEX_AUTH = "/mnt/release-codex-auth/auth.json"
COMM_DIR = f"{EnvironmentPaths.agent_dir.as_posix()}/multiagent-collab"
# The Claude Code tools agents get: shell, background-shell control, file view/edit,
# and search. Names a CLI version lacks are ignored.
CLAUDE_AVAILABLE_TOOLS = (
    "Bash",
    "BashOutput",
    "TaskOutput",
    "Monitor",
    "KillShell",
    "TaskStop",
    "Read",
    "Glob",
    "Grep",
    "Edit",
    "Write",
)
# The longest a Claude Code Bash command may run when the agent sets its own timeout: 1 hour,
# like the ALE `submit` wait Codex gets, instead of Claude Code's 10-minute cap, which made agents waiting on long
# experiments or the judge wake up every 10 minutes (each wake-up a model turn). The 2-minute default stays: a command
# without a timeout still moves to the background after it and the agent keeps working, as Codex does.
CLAUDE_BASH_MAX_TIMEOUT_MS = 3_600_000
CLAUDE_EXCLUDED_TOOLS = (
    "Agent",
    "Task",
    "Workflow",
    "SendMessage",
    "ListAgents",
    "WebFetch",
    "WebSearch",
    "Skill",
)
# Codex keeps exec_command, write_stdin, apply_patch, and view_image. Everything
# below removes web search, sub-agents, connectors, plugins, skills, image
# generation, browser/computer use, and AGENTS.md loading. The shell keeps the
# full environment minus the API key.
CODEX_CONFIG_TOML = """\
web_search = "disabled"
project_doc_max_bytes = 0
check_for_update_on_startup = false

[analytics]
enabled = false

[feedback]
enabled = false

[skills]
include_instructions = false

[skills.bundled]
enabled = false

[shell_environment_policy]
inherit = "all"
ignore_default_excludes = true
exclude = ["CODEX_API_KEY", "OPENAI_API_KEY"]

[features]
multi_agent = false
multi_agent_v2 = false
apps = false
plugins = false
remote_plugin = false
image_generation = false
browser_use = false
browser_use_external = false
in_app_browser = false
computer_use = false
tool_suggest = false
memories = false
goals = false
skill_search = false
workspace_dependencies = false
sleep_tool = false
realtime_conversation = false
"""
TOKEN_BUDGET_SOURCE = pathlib.Path(__file__).with_name("token_budget.py").read_text()
RESOURCES_TOOL_SOURCE = pathlib.Path(__file__).with_name("resources_tool.py").read_text()
LIMIT_RESUME_SOURCE = pathlib.Path(__file__).with_name("limit_resume.py").read_text()
HARNESS_GUARD_SOURCE = pathlib.Path(__file__).with_name("harness_guard.py").read_text()
TEAM_HOLD_SOURCE = pathlib.Path(__file__).with_name("team_hold.py").read_text()
LOCKSTEP_SOURCE = pathlib.Path(__file__).with_name("lockstep.py").read_text()

# The most continuation turns an agent gets after its first (run.py --continuations).
CONTINUATIONS = int(os.environ.get("RELEASE_CONTINUATIONS", "1000"))

# run.py --resources-tool (Codex and Claude Code): "1" gives every agent a `resources`
# command (resources_tool.py: its tokens against --token-budget, the team's, and the time
# used and left of --timeout, which run.py passes as RELEASE_AGENT_TIMEOUT), a paragraph
# about it at the end of its task instruction, and the remaining budget at the end of every
# continuation. Unset, every command the harness sends is unchanged.
RESOURCES_TOOL_ENV = "RELEASE_RESOURCES_TOOL"
AGENT_TIMEOUT_ENV = "RELEASE_AGENT_TIMEOUT"
# The harness's copy, which the runner runs for continuations, and the agents' command, a
# link to it in /usr/local/bin: that stays on PATH in the CLIs' login shells (Codex runs
# `bash -lc`), where Debian's /etc/profile drops ~/.local/bin.
RESOURCES_SCRIPT = "/usr/local/libexec/resources"
RESOURCES_COMMAND = "/usr/local/bin/resources"
RESOURCES_PROMPT = ("You have been given enough resources (tokens and time) to keep working on this task for a "
                    "long time. Check your remaining resources at any time by running `resources` in your "
                    "shell{team}.")
RESOURCES_PROMPT_TEAM = "; it also shows your team's total"


def _resources_tool_on() -> bool:
    return os.environ.get(RESOURCES_TOOL_ENV, "") == "1"


# Usage-limit resumption (Codex and Claude Code; run.py sets it unless --no-limit-resume): "1"
# makes an agent that hits a usage limit wait for the reset and continue its turn without its
# prompt being sent again (limit_resume.py). Waiting does not count against --timeout
# (RELEASE_AGENT_TIMEOUT, which the runner enforces itself; run.py gives Harbor that plus the
# most waiting allowed, RELEASE_LIMIT_WAIT_MAX seconds per agent, after which the agent is
# stopped). Unset, every command the harness sends is unchanged.
LIMIT_RESUME_ENV = "RELEASE_LIMIT_RESUME"
LIMIT_WAIT_MAX_ENV = "RELEASE_LIMIT_WAIT_MAX"
DEFAULT_LIMIT_WAIT_MAX = 48 * 3600
LIMIT_POLL_SECONDS = 2
# Claude Code's own text for resuming an interrupted turn (its CLAUDE_CODE_RESUME_PROMPT
# default), which Codex sessions get too.
LIMIT_RESUME_NOTE = "Continue from where you left off."
LIMIT_TIME_CAP_MARKER = "time-cap-reached.json"


def _limit_resume_on() -> bool:
    return os.environ.get(LIMIT_RESUME_ENV, "") == "1"


# Reviving a trial whose host process died while its containers and logs survived (revive.py,
# docs/harness.md): resume() runs a new runner that continues each agent's session
# where it stopped, with one note as the only new user message, and then the continuation loop
# from the next turn. revive.py sets the agent's revive_plan.
REVIVE_EVENT = "trial_revived"

# The restarted single agent, a relay (run.py --relay N; docs/setups.md): one agent's run is N legs, one after
# another in the same container, each a fresh CLI session with its own --token-budget. Every leg is the same agent
# to the task (its id, so its working copy and submissions are the previous leg's) with its own session directory
# (sessions-0-a1, then sessions-0-a1-leg2, ...), log and watchers. The next leg starts once a leg ends on its
# budget, by giving up or after its continuations, and gets the first leg's prompt with RELAY_NOTE before it.
# The working time (--timeout) and the most waiting (--limit-wait-max) are the run's, shared by its legs. Unset,
# every command the harness sends is unchanged.
RELAY_ENV = "RELEASE_RELAY"
RELAY_NOTE = ("You are building on top of a previous agent's work: another agent worked on this task before you, "
              "in this same environment, and what it left is still here. Continue the task.\n\n")
# A leg that left one of these ends the relay: the run's working time or waiting is used up, the task's episode is
# over, or harness protection had to stop the agent.
RELAY_FINAL_MARKERS = (LIMIT_TIME_CAP_MARKER, "limit-wait-exhausted.json", "episode-over.json",
                       harness_guard.STOP_MARKER)
def relay_legs() -> int | None:
    """The number of legs of a relay (RELAY_ENV), or None."""
    raw = os.environ.get(RELAY_ENV, "").strip()
    if not raw:
        return None
    if not raw.isdigit() or int(raw) < 2:
        raise ValueError(f"{RELAY_ENV} must be a number of legs, 2 or more: {raw!r}")
    return int(raw)


def relay_label(label: str, leg: int) -> str:
    """A leg's name in its session directory and log: the agent's own (0-a1) for the first leg, then 0-a1-leg2, ..."""
    return label if leg == 1 else f"{label}-leg{leg}"

# Harness protection (run.py sets it unless --no-harness-guard; docs/harness.md): "1" keeps an agent's own commands (`pkill -x python3`, `pkill -f <pattern>`,
# `kill -9 -1`) from stopping the watchers or the runner. A keeper (harness_guard.py) that agent.py
# installs in GUARD_DIR runs the watchers and each agent's runner, as root where the task's agent is
# another user, and starts again whatever dies; the runner traps the signals it can and takes a CLI
# that a signal ended as an interrupted turn, which it resumes; agent.py starts the keeper again if
# it dies, and ends the trial cleanly (HarnessProtectionError, graded) if it cannot. Unset, every
# command the harness sends is unchanged.
HARNESS_GUARD_ENV = "RELEASE_HARNESS_GUARD"
# Short names under a short path: an agent's `pkill -f <pattern>` matches a whole command line.
GUARD_DIR = "/usr/lib/hsv"
GUARD_LOG = "harness-guard.jsonl"
# The keeper is started again unless it dies within GUARD_QUICK_SECONDS of its start
# GUARD_MAX_QUICK_RESTARTS times in a row; every GUARD_PROBE_SECONDS agent.py asks for its status,
# which continues a stopped keeper and ends one whose heartbeat is GUARD_STALE_SECONDS old.
GUARD_QUICK_SECONDS = 60
GUARD_MAX_QUICK_RESTARTS = 5
GUARD_PROBE_SECONDS = 120
GUARD_STALE_SECONDS = 300
# A CLI that a signal ends within CLI_QUICK_SECONDS of its start CLI_MAX_QUICK_KILLS times in a row
# is not started again.
CLI_QUICK_SECONDS = 30
CLI_MAX_QUICK_KILLS = 5
# What an agent is told when its turn is resumed after a signal ended its CLI (in place of
# LIMIT_RESUME_NOTE, which says nothing happened). Without it a model that finds its last command
# unanswered runs it again: in a smoke run Claude Sonnet 5.5, as root, ran `kill -9 -1` after every
# resumption until the keeper's restart limit ended the trial.
# The CLI had not yet saved the command it was running when it was killed, so the resumed session
# does not hold it: the note names it (KILL_RESUME_COMMAND and the command, from the runner's log of the
# CLI's output, between the two parts) or the model would simply issue it again.
KILL_RESUME_NOTE = ("A signal ended your session while your last command was running (a kill, pkill or killall in this "
                    "container can do that). Your files are intact, but processes you had started may no longer be "
                    "running.")
KILL_RESUME_COMMAND = "The command that was running, which is not in your history above:"
KILL_RESUME_END = ("If that command was such a kill, it has taken effect: do not run it again. Continue from where you "
                   "left off.")


def _harness_guard_on() -> bool:
    return os.environ.get(HARNESS_GUARD_ENV, "") == "1"


# Team limit hold (run.py --team-limit-hold, on by default for a mixed team; docs/harness.md): "1"
# pauses a team while any of its agents waits for a usage limit. Each agent's keeper-run watcher
# (team_hold.py) holds for its agent during such a wait (its runner's own, which the runner announces in
# team_hold.REQUEST, or one inside Claude Code) and pauses the agent while a teammate holds (team_hold.HELD);
# the runner then waits in TEAM_HOLD_STEP_SECONDS steps that count as waiting, and resumes the turn the pause
# stopped. Needs usage-limit resumption and harness protection. Unset, every command is unchanged.
TEAM_HOLD_ENV = "RELEASE_TEAM_LIMIT_HOLD"
TEAM_HOLD_POLL_SECONDS = 2
TEAM_HOLD_STEP_SECONDS = 10
# Test only, never in a run's settings: "AFTER:SECONDS" makes the first agent wait AFTER seconds before its
# first turn and then wait SECONDS once as after a usage limit, so the others are paused without a real limit.
TEAM_HOLD_TEST_ENV = "RELEASE_TEAM_HOLD_TEST"


def _team_hold_on() -> bool:
    return os.environ.get(TEAM_HOLD_ENV, "") == "1"


def _team_hold_test() -> tuple[int, int] | None:
    value = os.environ.get(TEAM_HOLD_TEST_ENV, "").strip()
    if not value:
        return None
    after, _, seconds = value.partition(":")
    if not (after.isdigit() and seconds.isdigit()):
        raise ValueError(f"{TEAM_HOLD_TEST_ENV} must be AFTER:SECONDS: {value!r}")
    return int(after), int(seconds)


# Lockstep (run.py --lockstep-gap, off by default; docs/setups.md): a team's agents spend their token budgets
# together. Each agent's keeper-run watcher (lockstep.py, in the team-hold watcher's place) pauses its agent while
# it is more than the gap (weighted tokens) ahead of the team's reference (LOCKSTEP_REF_ENV: slowest, median or
# mean of the agents still working), until it is no longer ahead; and its working-time watcher counts the pause as
# neither working time nor waiting. The runner waits in LOCKSTEP_STEP_SECONDS steps while paused, then resumes the
# turn the pause stopped. LOCKSTEP_STALL_ENV: seconds without any unpaused agent's tokens growing after which every
# pause ends. Needs usage-limit resumption, harness protection and a token budget. Unset, every command is unchanged.
LOCKSTEP_GAP_ENV = "RELEASE_LOCKSTEP_GAP"
LOCKSTEP_REF_ENV = "RELEASE_LOCKSTEP_REF"
LOCKSTEP_STALL_ENV = "RELEASE_LOCKSTEP_STALL"
DEFAULT_LOCKSTEP_STALL = 3600
LOCKSTEP_STEP_SECONDS = 2
# What ends an agent while it waits (team_hold.ENDED, but for the runner's own exit status).
LOCKSTEP_ENDED = tuple(name for name in team_hold.ENDED if name != "runner-exit-status")


def _lockstep_settings() -> dict[str, Any] | None:
    gap = os.environ.get(LOCKSTEP_GAP_ENV, "").strip()
    if not gap:
        return None
    ref = os.environ.get(LOCKSTEP_REF_ENV, "").strip() or "slowest"
    stall = os.environ.get(LOCKSTEP_STALL_ENV, "").strip() or str(DEFAULT_LOCKSTEP_STALL)
    if not gap.isdigit() or int(gap) <= 0 or ref not in lockstep.REFERENCES or not stall.isdigit():
        raise ValueError(f"{LOCKSTEP_GAP_ENV} must be a positive number of tokens, {LOCKSTEP_REF_ENV} one of "
                         f"{', '.join(lockstep.REFERENCES)} and {LOCKSTEP_STALL_ENV} seconds: "
                         f"{gap!r}, {ref!r}, {stall!r}")
    return {"gap": int(gap), "ref": ref, "stall_seconds": int(stall)}


# Message board (the board protocol, Codex only; docs/setups.md): the
# agents of a Codex team share one board through Codex's remote message board
# (features.multi_agent_v2.message_board_remote, Codex 0.162.0 and later), which message_board.py serves in
# the container on 127.0.0.1:BOARD_PORT. Each agent stays its own `codex exec` process with its own home,
# token budget, usage-limit waits, keeper unit and `resources` command: only its config gains the board
# (board_codex_config) and one developer message with its path on the board (BOARD_IDENTITY), and its runner
# gives Codex the agent's credential. agent.py runs the board as root next to the agents (and starts it again
# if it dies); the board's journal is in the agent directory (BOARD_JOURNAL) as it is written, and as SQLite
# (BOARD_SQLITE) after the run. A protocol without message_board leaves every command unchanged.
BOARD_PORT = 39613
BOARD_SERVICE = f"{GUARD_DIR}/hsbd"
# python3 under another name, as the keeper's hsvi: an agent's `pkill -x python3` does not match the board.
BOARD_PYTHON = f"{GUARD_DIR}/hsbi"
BOARD_TOKEN_ENV = "MULTIAGENT_BOARD_TOKEN"
BOARD_SECRET_ENV = message_board.SECRET_ENV
BOARD_JOURNAL = "message-board.jsonl"
BOARD_SQLITE = "message-board.sqlite"
BOARD_SERVICE_LOG = "message-board-service.log"
BOARD_READY_SECONDS = 60
BOARD_SOURCE = pathlib.Path(__file__).with_name("message_board.py").read_text()
# Each agent's Codex is /root in its own process; this developer message tells it its path on the board.
BOARD_IDENTITY = "On the team message board you are `{path}`."
BOARD_AGENT = "@board_agent@"

# Claude peers (the protocol claude_peers, Claude Code only; docs/setups.md): the agents of a Claude Code team
# message one another with Claude Code's own cross-session messaging, its SendMessage and ListAgents tools (Claude
# Code 2.1.289). Each agent stays its own `claude -p` process with its own home, token budget, usage-limit waits,
# keeper unit and `resources` command: only its command line gains the two tools, its name (its id), its inbox and
# the setting that takes peers' messages (claude_peers_flags), and its configure step links its home's session
# registry to the team's and makes the inboxes' directory (CLAUDE_PEERS_CONFIGURE). Every session of an agent (its
# first turn, its continuations, a resumed turn, a revival) has the same name and listens on the same socket, so its
# address never changes; while none of its sessions runs (between turns, while the runner waits or a pause has stopped
# the CLI) a message to it fails at the sender. After the run, agent.py writes the team's messages to
# CLAUDE_PEERS_JOURNAL in the agent directory (claude_peers_journal.py). A protocol without claude_peers leaves every
# command unchanged.
CLAUDE_PEERS_TOOLS = ("SendMessage", "ListAgents")
# Each agent's inbox is <id>.sock here: a fixed path (a socket's path has at most 107 bytes), in the container rather
# than under /logs, and in a dot directory that an agent's `rm -rf /tmp/*` does not match.
CLAUDE_PEERS_SOCKETS = "/tmp/.claude-peers"
# The team's session registry, in the agent directory: each agent's home links it as its sessions/ (by a relative
# link, which also resolves on the host), where Claude Code lists the live sessions, their sockets and inbox keys.
CLAUDE_PEERS_REGISTRY = "claude-peers-sessions"
# Claude Code holds the messages a session that bypasses permissions (every agent here) gets from its peers for its
# user's approval, unless this setting says to take them.
CLAUDE_PEERS_SETTINGS = '{"crossSessionInbound":"accept"}'
# Claude Code's appended system prompt tells each agent its name, as BOARD_IDENTITY tells a board agent its path.
CLAUDE_PEERS_IDENTITY = "In the team you are `{name}`."
CLAUDE_PEERS_JOURNAL = "claude-peers.jsonl"
# The lines configure_claude_command runs, once the home exists, for an agent of a claude_peers team.
CLAUDE_PEERS_CONFIGURE = (
    f'  mkdir -p "{EnvironmentPaths.agent_dir.as_posix()}/{CLAUDE_PEERS_REGISTRY}" || return 1\n'
    '  if [ ! -e "$claude_log_dir/sessions" ] && [ ! -L "$claude_log_dir/sessions" ]; then\n'
    f'    ln -s "../../{CLAUDE_PEERS_REGISTRY}" "$claude_log_dir/sessions" || return 1\n'
    f'  elif [ "$(readlink "$claude_log_dir/sessions")" != "../../{CLAUDE_PEERS_REGISTRY}" ]; then\n'
    '    echo "claude_peers: $claude_log_dir/sessions exists and is not the link to the team session registry" >&2\n'
    "    return 1\n"
    "  fi\n"
    f"  mkdir -p {CLAUDE_PEERS_SOCKETS} && chmod 700 {CLAUDE_PEERS_SOCKETS} || return 1\n")


def claude_peers_flags() -> str:
    """The flags of a claude_peers agent's `claude -p` line ($agent_id: its id, which is its name)."""
    prefix, _, suffix = CLAUDE_PEERS_IDENTITY.partition("{name}")
    return (f'--name "$agent_id" --messaging-socket-path "{CLAUDE_PEERS_SOCKETS}/$agent_id.sock" '
            f"--settings {shlex.quote(CLAUDE_PEERS_SETTINGS)} "
            f'--append-system-prompt {shlex.quote(prefix)}"$agent_id"{shlex.quote(suffix)}')


# run.py --codex-compact-limit: Codex compacts at min(model_auto_compact_token_limit, 90% of
# model_context_window), and the window otherwise comes from OpenAI's model list at start. Pinned: the window at 500k, raised to fit a larger limit, never past gpt-6.1's max_context_window.
CODEX_PINNED_WINDOW = 500000
CODEX_MAX_WINDOW = 872000


def codex_context_config(base: str, limit: int | None) -> str:
    """base config.toml with the compaction point and context window pinned (top-level keys, so first)."""
    if limit is None:
        return base
    window = min(CODEX_MAX_WINDOW, max(CODEX_PINNED_WINDOW, -(-int(limit) * 10 // 9)))
    return f"model_context_window = {window}\nmodel_auto_compact_token_limit = {int(limit)}\n" + base


def board_codex_config(base: str = CODEX_CONFIG_TOML) -> str:
    """CODEX_CONFIG_TOML for an agent on the message board, and nothing else changed. gpt-6 models already run
    Codex's multi-agent v2 under CODEX_CONFIG_TOML (`multi_agent_v2 = false` leaves the model's own version, and
    `[agents]` is on by default): the same role message, spawn_agent with the same 4 threads, send_message and
    followup_task for the agent's own sub-agents. Turning the feature on by name changes none of that; it lets
    the board's tools in. BOARD_AGENT is the agent's id, which the runner fills in (CODEX_BOARD_LINES). Each edit
    fails loudly if the base changes."""
    edits = (
        ("check_for_update_on_startup = false\n",
         # agent_message_board is "under development" in Codex.
         "check_for_update_on_startup = false\nsuppress_unstable_features_warning = true\n"
         f'developer_instructions = "{BOARD_IDENTITY.format(path="/root/" + BOARD_AGENT)}"\n'),
        ('exclude = ["CODEX_API_KEY", "OPENAI_API_KEY"]\n',
         f'exclude = ["CODEX_API_KEY", "OPENAI_API_KEY", "{BOARD_TOKEN_ENV}", "{BOARD_SECRET_ENV}"]\n'),
        ("multi_agent_v2 = false\n", "agent_message_board = true\n"),
    )
    for old, new in edits:
        if base.count(old) != 1:
            raise ValueError(f"CODEX_CONFIG_TOML changed; this board edit no longer applies: {old!r}")
        base = base.replace(old, new)
    return base + (
        "\n[features.multi_agent_v2]\n"
        "enabled = true\n"
        f'message_board_remote = {{ url = "http://127.0.0.1:{BOARD_PORT}", bearer_token_env_var = "{BOARD_TOKEN_ENV}" }}\n')


# The lines configure_codex_command runs after it writes config.toml, for an agent on the board: its path on
# the board and its credential (message_board.token), from the run's secret.
CODEX_BOARD_LINES = (
    f'  sed -i "s|{BOARD_AGENT}|$agent_id|" "$codex_log_dir/config.toml"\n'
    f'  export {BOARD_TOKEN_ENV}="{message_board.token("$agent_id", "${" + BOARD_SECRET_ENV + ":-}")}"\n')


def _board_protocol(import_path: str) -> bool:
    """Whether a protocol runs its team on the message board (its class's message_board)."""
    module, _, attribute = import_path.partition(":")
    protocol = getattr(importlib.import_module(module), attribute or "MultiAgentProtocol")
    return bool(getattr(protocol, "message_board", False))


def _claude_peers_protocol(import_path: str) -> bool:
    """Whether a protocol runs its team on Claude Code's cross-session messaging (its class's claude_peers)."""
    module, _, attribute = import_path.partition(":")
    protocol = getattr(importlib.import_module(module), attribute or "MultiAgentProtocol")
    return bool(getattr(protocol, "claude_peers", False))


# Swarmchat (the protocol swarmchat: Claude Code, Codex and mixed teams; docs/setups.md): the team shares a chat
# run by upstream SwarmChat's own code (vendor/swarmchat, github.com/Pranjal2041/swarmchat, unmodified) in the
# container, with swarmchat_glue.py as the only glue, as upstream's own team runner (extras/programbench) runs it:
# upstream's server as root on 127.0.0.1:SWARMCHAT_PORT (started again if it dies; its data in the agent directory),
# one user per agent made with upstream's admin calls, upstream's `swarmchat` command on PATH, and upstream's notifier
# and PostToolUse hook, which add each agent's new messages to its conversation after its tool calls: Claude Code's
# hook from `--settings`, Codex's from hooks.json in its home (with --dangerously-bypass-hook-trust, as Codex runs a
# user hook only once trusted). Each agent also gets one system-level line with its name (SWARMCHAT_IDENTITY).
# Nothing else changes; a protocol without swarmchat leaves every command unchanged.
SWARMCHAT_PORT = 39614
SWARMCHAT_GLUE = f"{GUARD_DIR}/hscd"
# Upstream's package, in its own layout (swarmchat/, extras/programbench/), where swarmchat_glue.py looks for it.
SWARMCHAT_LIB = f"{GUARD_DIR}/hsc"
# python3 under another name, as the board's: an agent's `pkill -x python3` does not match the server.
SWARMCHAT_PYTHON = f"{GUARD_DIR}/hsci"
# In the agent directory: upstream's data folder (root only: the admin token is in it), and per agent its token,
# notice folder, the notifier's saved positions and the hook's shown.jsonl.
SWARMCHAT_DATA = "swarmchat"
SWARMCHAT_AGENTS = "swarmchat-agents"
SWARMCHAT_SERVICE_LOG = "swarmchat-service.log"
SWARMCHAT_READY_SECONDS = 60
# Upstream's `swarmchat serve` flags: each write forced to disk, as the board's journal is (a revival after a host
# crash keeps every message).
SWARMCHAT_SERVE_FLAGS = ("--fsync",)
SWARMCHAT_VENDOR = pathlib.Path(__file__).with_name("vendor") / "swarmchat"
# What the container gets of the vendored copy: the package (not its viewer's page) and the two runner modules the
# glue imports, with the license.
SWARMCHAT_FILES = ("LICENSE", "extras/programbench/guest.py", "extras/programbench/run.py",
                   *sorted(f"swarmchat/{path.name}" for path in (SWARMCHAT_VENDOR / "swarmchat").glob("*.py")))
SWARMCHAT_GLUE_SOURCE = pathlib.Path(__file__).with_name("swarmchat_glue.py").read_text()
# Upstream's command, as its runner writes it (extras/programbench/run.py: /opt/swarmchat/bin/swarmchat).
SWARMCHAT_COMMAND = "/usr/local/bin/swarmchat"
SWARMCHAT_WRAPPER = f'#!/bin/sh\nPYTHONPATH={SWARMCHAT_LIB} exec python3 -m swarmchat "$@"\n'
SWARMCHAT_IDENTITY = "On the team chat you are `{name}`."
SWARMCHAT_AGENT = "@swarmchat_agent@"
SWARMCHAT_HOOK_SECONDS = 30  # upstream's (extras/programbench/run.py CLAUDE_HOOKS)


def swarmchat_bundle() -> str:
    """SWARMCHAT_FILES as a gzipped tar in base64, the same bytes every time."""
    import base64, gzip, io, tarfile  # noqa: E401 - only here
    data = io.BytesIO()
    with gzip.GzipFile(fileobj=data, mode="wb", mtime=0) as packed, tarfile.open(fileobj=packed, mode="w") as tar:
        for name in SWARMCHAT_FILES:
            body = (SWARMCHAT_VENDOR / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(body), 0o644
            tar.addfile(info, io.BytesIO(body))
    return base64.b64encode(data.getvalue()).decode()


def swarmchat_hooks(cli: str) -> str:
    """The PostToolUse hook, as each CLI reads it (Claude Code: settings JSON, matcher "*" as upstream's; Codex:
    hooks.json, no matcher = every tool, its context never spilled to a file)."""
    handler: dict[str, Any] = {"type": "command", "command": f"python3 {SWARMCHAT_GLUE} hook",
                               "timeout": SWARMCHAT_HOOK_SECONDS}
    group: dict[str, Any] = {"hooks": [handler]}
    if cli == "claude":
        group["matcher"] = "*"
    else:
        handler["additionalContextLimit"] = 0
    return json.dumps({"hooks": {"PostToolUse": [group]}}, separators=(",", ":"))


def swarmchat_configure() -> str:
    """The lines a configure step runs for an agent of a swarmchat team: upstream's SWARMCHAT_SERVER and its own
    SWARMCHAT_TOKEN (the glue wrote it when it made the agent's user), and its notice folder for the hook."""
    folder = f"{EnvironmentPaths.agent_dir.as_posix()}/{SWARMCHAT_AGENTS}/$agent_id"
    return (f'  export SWARMCHAT_SERVER="http://127.0.0.1:{SWARMCHAT_PORT}" TTC_SWARMCHAT_NOTICES="{folder}/notices"\n'
            f'  SWARMCHAT_TOKEN="$(cat "{folder}/token")" && [ -n "$SWARMCHAT_TOKEN" ] || '
            '{ echo "swarmchat: no token for $agent_id" >&2; return 1; }\n'
            "  export SWARMCHAT_TOKEN\n")


def swarmchat_claude_flags() -> str:
    """The flags of a swarmchat agent's `claude -p` line: the hook and its name ($agent_id)."""
    prefix, _, suffix = SWARMCHAT_IDENTITY.partition("{name}")
    return (f"--settings {shlex.quote(swarmchat_hooks('claude'))} "
            f'--append-system-prompt {shlex.quote(prefix)}"$agent_id"{shlex.quote(suffix)}')


def swarmchat_codex_config(base: str = CODEX_CONFIG_TOML) -> str:
    """CODEX_CONFIG_TOML for an agent of a swarmchat team: one developer message with its name (SWARMCHAT_AGENT,
    which the configure step fills in), as the board's, and nothing else changed."""
    old = "check_for_update_on_startup = false\n"
    if base.count(old) != 1:
        raise ValueError(f"CODEX_CONFIG_TOML changed; the swarmchat edit no longer applies: {old!r}")
    return base.replace(old, old + f'developer_instructions = "{SWARMCHAT_IDENTITY.format(name=SWARMCHAT_AGENT)}"\n')


# The lines configure_codex_command runs after it writes config.toml, for an agent of a swarmchat team.
SWARMCHAT_CODEX_LINES = (
    f'  sed -i "s|{SWARMCHAT_AGENT}|$agent_id|" "$codex_log_dir/config.toml"\n'
    f"  printf '%s\\n' {shlex.quote(swarmchat_hooks('codex'))} > \"$codex_log_dir/hooks.json\"\n"
    + swarmchat_configure())


def _swarmchat_protocol(import_path: str) -> bool:
    """Whether a protocol runs its team on the harness's team chat (its class's swarmchat)."""
    module, _, attribute = import_path.partition(":")
    protocol = getattr(importlib.import_module(module), attribute or "MultiAgentProtocol")
    return bool(getattr(protocol, "swarmchat", False))


# Ultra mode (the ultra protocol, Codex only; ultra.py, docs/setups.md): one agent, the run's single runner,
# whose Codex delegates to its own helpers, threads of the same `codex exec`, at most ultra.ENV of them at once,
# the root included. Its config gains the thread cap and Codex's proactive mode message (ultra.codex_config), its
# command line the run's effort for every helper, and its prompt the cap line; its token budget, `resources`,
# usage-limit waits and keeper unit are the single agent's, and so count the whole tree. Without ultra.ENV every
# command is unchanged.
def _ultra_protocol(import_path: str) -> bool:
    """Whether a protocol is ultra mode (its class's ultra)."""
    module, _, attribute = import_path.partition(":")
    protocol = getattr(importlib.import_module(module), attribute or "MultiAgentProtocol")
    return bool(getattr(protocol, "ultra", False))


class HarnessProtectionError(NonZeroAgentExitCodeError):
    """The harness could not keep its protection running, so it stopped the agents cleanly; Harbor
    grades the trial, as after an agent's non-zero exit."""


def resources_script(config: dict[str, Any], lockstep_pauses: bool = False) -> str:
    """The `resources` command with the run's settings: cli, homes (each agent's CLI home, in
    agent order), budget (None without one), metric, timeout (seconds), started (epoch). With
    LOCKSTEP_PAUSES, an agent's lockstep pauses do not count as its time used either."""
    return ("#!/usr/bin/env python3\n" + RESOURCES_TOOL_SOURCE.rstrip("\n") + "\n\n\n"
            + (RESOURCES_LOCKSTEP if lockstep_pauses else "") +
            'if __name__ == "__main__":\n'
            f"    sys.exit(main({json.dumps(config, sort_keys=True)!r}, {TOKEN_BUDGET_SOURCE!r}))\n")


# With lockstep: the time an agent was paused (limit-wait.json's lockstep_paused_seconds, which the
# working-time watcher writes) is not time used, as waiting is not.
RESOURCES_LOCKSTEP = '''\
waited_for_limits = waited


def waited(home: str) -> float:
    try:
        paused = json.loads((Path(home).parent / "limit-wait.json").read_text()).get("lockstep_paused_seconds")
        paused = float(paused or 0.0)
    except (OSError, ValueError, AttributeError, TypeError):
        paused = 0.0
    return waited_for_limits(home) + paused


'''

# A task whose episode can end while its agents still run (BrowseComp: an agent's work ends
# when its turn ends after it has submitted) names a shell command that succeeds once the
# agent's own episode is over; run.py sets it in this variable. Each agent's runner then polls it in a watcher, which stops the CLI and
# everything it started (as a spent token budget does), and runs no further continuation.
# Unset for every other task, whose runners are unchanged.
EPISODE_OVER_ENV = "RELEASE_EPISODE_OVER_COMMAND"
EPISODE_POLL_SECONDS = 5


def _episode_over_command() -> str:
    return os.environ.get(EPISODE_OVER_ENV, "").strip()


# The watcher: argv is the command, the CLI's home variable and home, the marker file,
# and the poll interval. It runs the command with the agent's identity until it succeeds,
# then writes the marker and stops every process whose environment points at the home.
EPISODE_WATCH_SOURCE = """\
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

command, env_name, home, marker, interval = sys.argv[1:6]


def over():
    try:
        return subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=30).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def agent_processes():
    entry = f"{env_name}={home}".encode()
    found = []
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == os.getpid():
            continue
        try:
            with open(f"/proc/{name}/environ", "rb") as stream:
                if entry in stream.read().split(b"\\0"):
                    found.append(int(name))
        except OSError:
            continue
    return found


while True:
    if over():
        if not os.path.exists(marker):
            with open(marker, "w") as stream:
                json.dump({"ts": datetime.now(timezone.utc).isoformat(), "event": "episode_over"}, stream)
        for sig in (signal.SIGTERM, signal.SIGKILL) if env_name else ():
            pids = agent_processes()
            for pid in pids:
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass
            if pids and sig == signal.SIGTERM:
                time.sleep(5)
    time.sleep(float(interval))
"""

EPISODE_HELPERS = r"""
start_episode_watch() {
  episode_over_marker="${1%/*}/episode-over.json"
  episode_identity=("$2" "$3" "$4")
  rm -f "$episode_over_marker"
  AGENT_ID="$2" AGENT_INDEX="$3" AGENT_COUNT="$4" \
    python3 -c "$episode_watch_source" "$episode_over_command" "$episode_home_env" "$1" \
    "$episode_over_marker" "$episode_poll_seconds" > /dev/null 2>> "${1%/*}/episode-watch.log" &
  episode_watcher=$!
  trap 'kill ${token_budget_watcher:-} ${episode_watcher:-} 2>/dev/null' EXIT
}

# Over if the watcher saw it, or (an agent that just ended its own game stops before the
# watcher's next poll) if the command says so now.
episode_over() {
  [ -n "${episode_over_marker:-}" ] || return 1
  [ -e "$episode_over_marker" ] && return 0
  AGENT_ID="${episode_identity[0]}" AGENT_INDEX="${episode_identity[1]}" \
    AGENT_COUNT="${episode_identity[2]}" bash -c "$episode_over_command" > /dev/null 2>&1
}

append_episode_over_event() {
  printf '{"type":"environment.feedback","data":{"event":"episode_over","message":"The episode ended for this agent, so @label@ was stopped."}}\n' >> "$1"
}
"""


def _no_action_feedback_body() -> str:
    return os.environ["RELEASE_CONTINUATION"]


def _continuation_enabled() -> bool:
    return CONTINUATIONS > 0


def _frontier_swe() -> bool:
    return os.environ.get("RELEASE_TASK") in HARNESS_TOOLS


# FrontierSWE's harness tools (benchmarks/frontier-swe/fswe.py), mounted by the task.
FRONTIER_SWE_TOOLS = "/opt/fswe/fswe.py"
HARNESS_TOOLS = {"frontier-swe": FRONTIER_SWE_TOOLS}


def _harness_tools() -> str:
    return HARNESS_TOOLS[os.environ["RELEASE_TASK"]]


# Runner helpers shared by every CLI. @cli@ is the CLI command name (codex),
# @CLI@ its upper-case form, and @label@ its display name (Codex).
CONTINUATION_HELPERS = r"""
write_no_action_feedback() {
  local feedback_file="$1"
  cat > "$feedback_file" <<'EOF'
__NO_ACTION_FEEDBACK_BODY__
EOF
}

append_optimization_continuation_event() {
  local log_file="$1"
  local attempt="$2"
  local max_attempts="$3"
  printf '{"type":"environment.feedback","data":{"event":"optimization_continuation","message":"Previous turn ended; continuing the same @label@ session for another optimization pass.","attempt":%s,"maxAttempts":%s}}\n' "$attempt" "$max_attempts" >> "$log_file"
}
"""

API_RETRY_HELPERS = r"""
@cli@_api_retry_sleep_seconds() {
  local retry="$1"
  local base="${META_DUO_@CLI@_API_RETRY_BASE_SLEEP:-30}"
  local cap="${META_DUO_@CLI@_API_RETRY_MAX_SLEEP:-600}"
  case "$retry" in ''|*[!0-9]*) retry=1 ;; esac
  case "$base" in ''|*[!0-9]*) base=30 ;; esac
  case "$cap" in ''|*[!0-9]*) cap=600 ;; esac
  if [ "$retry" -lt 1 ]; then
    retry=1
  fi
  if [ "$base" -lt 1 ]; then
    base=30
  fi
  if [ "$cap" -lt "$base" ]; then
    cap="$base"
  fi
  local delay="$base"
  local step=1
  while [ "$step" -lt "$retry" ] && [ "$delay" -lt "$cap" ]; do
    delay=$((delay * 2))
    if [ "$delay" -gt "$cap" ]; then
      delay="$cap"
    fi
    step=$((step + 1))
  done
  if [ "$delay" -gt "$cap" ]; then
    delay="$cap"
  fi
  if [ "$delay" -lt "$cap" ]; then
    delay=$((delay + (RANDOM % base)))
    if [ "$delay" -gt "$cap" ]; then
      delay="$cap"
    fi
  fi
  echo "$delay"
}

append_@cli@_api_retry_event() {
  local log_file="$1"
  local retry="$2"
  local max_retries="$3"
  local code="$4"
  local delay="$5"
  printf '{"type":"environment.feedback","data":{"event":"@cli@_api_retry","message":"@label@ CLI returned a retryable 401/429/API error; backing off and resuming the same session instead of failing the run.","attempt":%s,"maxAttempts":%s,"exitCode":%s,"sleepSeconds":%s}}\n' "$retry" "$max_retries" "$code" "$delay" >> "$log_file"
}

append_@cli@_api_retry_exhausted_event() {
  local log_file="$1"
  local max_retries="$2"
  local code="$3"
  printf '{"type":"environment.feedback","data":{"event":"@cli@_api_retries_exhausted","message":"@label@ CLI kept returning retryable 401/429/API errors after all retry attempts.","maxAttempts":%s,"exitCode":%s}}\n' "$max_retries" "$code" >> "$log_file"
}
"""

# Session ids, exit status, and retryable errors for Claude Code and Codex come
# from their own JSONL events, never from tool output an agent printed, so an
# agent that greps for "429" cannot trigger a retry by accident.
CLAUDE_HELPERS = r"""
configure_claude_command() {
  claude_command=("$claude_binary")
  mkdir -p "$claude_log_dir"
  start_token_budget "$claude_log_dir"
  run_task_cli_hook claude || return 1
}

extract_claude_session_id() {
  local source_log="$1"
  python3 - "$source_log" <<'PY'
import json
import sys

session_id = ""
try:
    lines = open(sys.argv[1], errors="replace")
except OSError:
    lines = ()
for line in lines:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(event, dict):
        value = event.get("session_id")
        if isinstance(value, str) and value:
            session_id = value
if session_id:
    print(session_id)
PY
}

# A result event with is_error is a failed attempt even if the CLI exited 0.
finish_claude_attempt() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

code = int(sys.argv[2]) if sys.argv[2].isdigit() else 1
failed = False
try:
    lines = open(sys.argv[1], errors="replace")
except OSError:
    lines = ()
for line in lines:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(event, dict) and event.get("type") == "result":
        failed = bool(event.get("is_error"))
sys.exit(1 if code == 0 and failed else code)
PY
}
"""

# With a token budget, a watcher reads the agent's session logs in its CLI home
# (the argument) and stops the CLI once the budget is spent; the runner then ends
# the agent cleanly. Its files sit in the session directory, the home's parent.
TOKEN_BUDGET_HELPERS = r"""
start_token_budget() {
  token_budget_marker=""
  [ -n "${token_budget:-}" ] || return 0
  local budget_dir="${1%/*}"
  token_budget_marker="$budget_dir/token-budget-exhausted.json"
  rm -f "$token_budget_marker"
  python3 -c "$token_budget_source" @cli@ "$1" "$token_budget" "$token_budget_metric" \
    "$token_budget_marker" "$budget_dir/token-usage.json" \
    > /dev/null 2>> "$budget_dir/token-budget-watcher.log" &
  token_budget_watcher=$!
  trap 'kill "$token_budget_watcher" 2>/dev/null' EXIT
}

token_budget_exhausted() {
  [ -n "${token_budget_marker:-}" ] && [ -e "$token_budget_marker" ]
}

append_token_budget_event() {
  printf '{"type":"environment.feedback","data":{"event":"token_budget_exhausted","message":"The agent used its token budget, so @label@ was stopped.","budget":%s,"budgetMetric":"%s"}}\n' "$token_budget" "$token_budget_metric" >> "$1"
}
"""

# Usage-limit resumption (LIMIT_RESUME_ENV). A watcher adds up the agent's waiting (the CLI's
# own and the runner's) and stops its CLI once the agent has used its working time (--timeout
# minus waiting) or waited the most allowed, leaving a marker in the session directory, the
# CLI home's parent. limit_wait sleeps at most the waiting left; limit_stop_reached tells
# whether a limit stopped the agent and logs it once.
LIMIT_RESUME_HELPERS = r"""
start_limit_watch() {
  limit_dir="${1%/*}"
  limit_stop_logged=0
  rm -f "$limit_dir/limit-wait-exhausted.json" "$limit_dir/time-cap-reached.json"
  printf '%s' "$limit_resume_note" > "$limit_dir/@cli@-resume-note.txt"
  python3 -c "$limit_resume_source" watch @cli@ "$limit_dir" @home_env@ "$1" "$limit_started" \
    "$limit_timeout" "$limit_wait_max" "$limit_poll_seconds" > /dev/null 2>> "$limit_dir/limit-watch.log" &
  limit_watcher=$!
  trap stop_watchers EXIT
}

# Stopped, the limit watcher records the final waiting and working time; give it up to 5 s.
stop_watchers() {
  kill ${token_budget_watcher:-} ${episode_watcher:-} ${limit_watcher:-} 2>/dev/null
  local tries=0
  while [ "$tries" -lt 25 ] && kill -0 "$limit_watcher" 2>/dev/null; do
    sleep 0.2
    tries=$((tries + 1))
  done
}

limit_stop_reached() {
  [ -n "${limit_dir:-}" ] || return 1
  if [ -e "$limit_dir/limit-wait-exhausted.json" ]; then
    [ "$limit_stop_logged" -eq 1 ] || printf '{"type":"environment.feedback","data":{"event":"limit_wait_exhausted","message":"The agent waited the most allowed in total for usage limits and API retries (run.py --limit-wait-max), so @label@ was stopped.","maxWaitSeconds":%s}}\n' "$limit_wait_max" >> "$log_file"
  elif [ -e "$limit_dir/time-cap-reached.json" ]; then
    [ "$limit_stop_logged" -eq 1 ] || printf '{"type":"environment.feedback","data":{"event":"time_cap_reached","message":"The agent used its working time (--timeout, not counting time spent waiting for usage limits and API retries), so @label@ was stopped.","timeoutSeconds":%s}}\n' "${limit_timeout:-null}" >> "$log_file"
  else
    return 1
  fi
  limit_stop_logged=1
  return 0
}

limit_wait() {
  local planned="$1"
  local granted
  granted="$(python3 -c "$limit_resume_source" plan-wait "$limit_dir" "$planned" "$limit_wait_max")"
  case "$granted" in ''|*[!0-9]*) granted="$planned" ;; esac
  sleep "$granted"
  if [ "$granted" -lt "$planned" ]; then
    python3 -c "$limit_resume_source" exhaust "$limit_dir" "$limit_wait_max"
    return 1
  fi
  return 0
}

append_usage_limit_wait_event() {
  printf '{"type":"environment.feedback","data":{"event":"usage_limit_wait","message":"@label@ hit a usage limit; waiting until it resets, then resuming the turn without sending its prompt again.","attempt":%s,"exitCode":%s,"sleepSeconds":%s,"resetsAt":%s}}\n' "$2" "$3" "$4" "$5" >> "$1"
}

append_turn_resumed_event() {
  printf '{"type":"environment.feedback","data":{"event":"interrupted_turn_resumed","message":"The session already has the prompt of this turn, so @label@ resumes the interrupted turn instead of sending it again.","attempt":%s}}\n' "$2" >> "$1"
}
"""

CLAUDE_LIMIT_RESUME_HELPERS = r"""
# Continue the session's interrupted turn without a new prompt: Claude Code re-runs a prompt that
# got no answer, or continues after the last tool results with its meta "Continue from where you
# left off." (CLAUDE_CODE_RESUME_PROMPT), reading no input from stdin.
resume_claude_turn() {
  CLAUDE_CODE_RESUME_INTERRUPTED_TURN=1 CLAUDE_CODE_RESUME_INTERRUPTED_TURN_MAX_AGE_MS=2592000000 \
    CLAUDE_CODE_RESUME_PROMPT="$limit_resume_note" \
    invoke_claude /dev/null "$1" --resume "$session_id" --input-format stream-json
}
"""

CODEX_LIMIT_RESUME_HELPERS = r"""
# Codex starts no turn without input: resume the session with the note instead of the prompt.
resume_codex_turn() {
  invoke_codex "$limit_dir/codex-resume-note.txt" "$1" resume "$session_id"
}
"""

# Team limit hold (TEAM_HOLD_ENV): the runner announces its own usage-limit wait to its watcher
# (team_hold.REQUEST), and waits while the watcher says a teammate waits for one (team_hold.HELD).
TEAM_HOLD_HELPERS = r"""
team_hold_request() {
  local start
  start="$(date +%s)"
  printf '{"start":%s,"until":%s}\n' "$start" "$((start + $1))" > "$limit_dir/@hold_request@.tmp"
  mv -f "$limit_dir/@hold_request@.tmp" "$limit_dir/@hold_request@"
}

team_hold_request_end() {
  rm -f "$limit_dir/@hold_request@"
}

team_held() {
  [ -e "$limit_dir/@held@" ]
}

# Wait while a teammate holds, in steps that count as waiting ($1: the attempt the hold stopped, if any).
# Returns 1 if the agent was stopped while it waited (it waited the most allowed).
team_hold_pause() {
  team_held || return 0
  local paused=$SECONDS
  printf '{"type":"environment.feedback","data":{"event":"team_limit_hold","message":"A teammate is waiting for a usage limit to reset, so the team pauses: @label@ waits until the teammate goes on%s. The pause counts as waiting, not as working time.","hold":%s,"attemptLog":"%s"}}\n' "${1:+, and its interrupted turn is resumed then}" "$(cat "$limit_dir/@held@" 2>/dev/null || printf null)" "${1##*/}" >> "$log_file"
  while team_held; do
    if ! limit_wait @hold_step@; then
      limit_stop_reached
      return 1
    fi
  done
  printf '{"type":"environment.feedback","data":{"event":"team_limit_hold_ended","message":"The teammate goes on, so @label@ does too.","pausedSeconds":%s}}\n' "$((SECONDS - paused))" >> "$log_file"
  return 0
}

# The hold stopped the attempt $1: wait, then resume its turn with the note for a turn a signal ended if a
# command was running then, else with the usual one.
team_hold_stopped() {
  [ -n "$session_id" ] || session_id="$(python3 "$harness_tool" home-session @cli@ "$@cli@_log_dir")"
  if [ -n "$(python3 "$harness_tool" pending-command @cli@ "$1")" ]; then
    harness_killed_note "$1"
  fi
  team_hold_pause "$1"
}
"""

# Test only (TEAM_HOLD_TEST_ENV): the first agent fakes one usage-limit wait before its first turn.
TEAM_HOLD_TEST_HELPERS = r"""
team_hold_test() {
  [ "$idx" = 0 ] && [ ! -e "$limit_dir/team-hold-test.done" ] || return 0
  : > "$limit_dir/team-hold-test.done"
  sleep @test_after@
  printf '{"type":"environment.feedback","data":{"event":"team_hold_test","message":"Test only (@test_env@): a usage-limit wait of @test_seconds@ s is faked."}}\n' >> "$log_file"
  append_usage_limit_wait_event "$log_file" 0 0 @test_seconds@ "$(($(date +%s) + @test_seconds@))"
  team_hold_request @test_seconds@
  limit_wait @test_seconds@
  team_hold_request_end
}
"""

# Lockstep (LOCKSTEP_GAP_ENV): the runner starts no attempt while its watcher has paused the agent
# (lockstep.PAUSE) or, with the team limit hold, a teammate holds (team_hold.HELD), and resumes the turn a
# pause stopped. A hold's steps count as waiting; a lockstep pause's count as neither waiting nor working
# time (the working-time watcher leaves the pause out of both).
LOCKSTEP_HELPERS = r"""
lockstep_paused() {
  [ -e "$limit_dir/@lockstep@" ]
}

lockstep_held() {
  @held_test@
}

team_paused() {
  lockstep_paused || lockstep_held
}

# Wait while paused or held ($1: the attempt the pause stopped, if any). Returns 1 if the agent ended
# meanwhile (its budget, working time or waiting used up, its episode over, or the harness stopped it).
team_pause() {
  team_paused || return 0
  local paused=$SECONDS lockstep_seen=0 held_seen=0 marker
  while team_paused; do
    if [ "$lockstep_seen" -eq 0 ] && lockstep_paused; then
      lockstep_seen=1
      printf '{"type":"environment.feedback","data":{"event":"lockstep_pause","message":"@label@ is ahead of its team in tokens, so it pauses until the team catches up%s. The pause counts as neither working time nor waiting.","pause":%s,"attemptLog":"%s"}}\n' "${1:+, and its interrupted turn is resumed then}" "$(cat "$limit_dir/@lockstep@" 2>/dev/null || printf null)" "${1##*/}" >> "$log_file"
    fi
    if [ "$held_seen" -eq 0 ] && lockstep_held; then
      held_seen=1
      printf '{"type":"environment.feedback","data":{"event":"team_limit_hold","message":"A teammate is waiting for a usage limit to reset, so the team pauses: @label@ waits until the teammate goes on%s. The pause counts as waiting, not as working time.","hold":%s,"attemptLog":"%s"}}\n' "${1:+, and its interrupted turn is resumed then}" "$(cat "$limit_dir/@held@" 2>/dev/null || printf null)" "${1##*/}" >> "$log_file"
    fi
    for marker in @ended@; do
      if [ -e "$limit_dir/$marker" ]; then
        limit_stop_reached
        return 1
      fi
    done
    if lockstep_held; then
      if ! limit_wait @hold_step@; then
        limit_stop_reached
        return 1
      fi
    else
      sleep @lockstep_step@
    fi
  done
  [ "$lockstep_seen" -eq 0 ] || printf '{"type":"environment.feedback","data":{"event":"lockstep_pause_ended","message":"The team has caught up, so @label@ goes on.","pausedSeconds":%s}}\n' "$((SECONDS - paused))" >> "$log_file"
  [ "$held_seen" -eq 0 ] || printf '{"type":"environment.feedback","data":{"event":"team_limit_hold_ended","message":"The teammate goes on, so @label@ does too.","pausedSeconds":%s}}\n' "$((SECONDS - paused))" >> "$log_file"
  return 0
}

# A pause stopped the attempt $1: wait, then resume its turn with the note for a turn a signal ended if a
# command was running then, else with the usual one.
team_pause_stopped() {
  [ -n "$session_id" ] || session_id="$(python3 "$harness_tool" home-session @cli@ "$@cli@_log_dir")"
  if [ -n "$(python3 "$harness_tool" pending-command @cli@ "$1")" ]; then
    harness_killed_note "$1"
  fi
  team_pause "$1"
}
"""

# Harness protection (HARNESS_GUARD_ENV): the runner's own part. Only SIGKILL ends the runner (a trapped
# signal is back at its default in every command the runner starts, so the CLI and what it starts get
# the signals they always got); its helper commands run under names an agent's `pkill python3` or
# `pkill sed` does not match; a sleep that is killed is slept again. start_runner_guard leaves the
# runner's exit code where the keeper finds it (a runner that was killed leaves none).
HARNESS_GUARD_HELPERS = r"""
harness_python=@guard_dir@/hsvi
harness_tool=@guard_dir@/hsvd
harness_sed=@guard_dir@/hsvs
trap ':' TERM HUP INT QUIT
python3() { "$harness_python" "$@"; }
sed() {
  if [ "$BASH_SUBSHELL" -gt 0 ]; then
    trap '' TERM HUP INT QUIT
    exec "$harness_sed" "$@"
  fi
  "$harness_sed" "$@"
}
sleep() {
  case "$1" in ''|*[!0-9]*) command sleep "$@"; return ;; esac
  local sleep_end=$((SECONDS + 10#$1)) sleep_tries=0
  while [ "$SECONDS" -lt "$sleep_end" ] && [ "$sleep_tries" -lt 100000 ]; do
    command sleep "$((sleep_end - SECONDS))"
    sleep_tries=$((sleep_tries + 1))
  done
}

start_runner_guard() {
  runner_session_dir="$1"
  harness_stop_marker="$1/@stop_marker@"
  runner_exit_file="$1/@exit_file@"
  trap 'runner_exit=$?; printf "%s\n" "$runner_exit" > "$runner_exit_file"' EXIT
}

# The keeper could not keep this agent's watchers running and stopped its CLI (it logged why).
harness_stop_reached() {
  [ -n "${harness_stop_marker:-}" ] && [ -e "$harness_stop_marker" ]
}

# The CLI ended on a signal the harness did not send (a command of an agent such as `pkill -f claude`
# or `kill -9 -1`, or the kernel): its turn was interrupted, not finished, so the runner resumes it
# ($1 the exit code, $2 when the attempt started, $3 its log). Returns 1, and the agent ends as it did
# before, once the CLI has ended so within @quick@ s of its start @max_kills@ times in a row.
cli_ended_by_signal() {
  [ "$1" -ge 128 ] || return 1
  if [ $((SECONDS - $2)) -lt @quick@ ]; then
    cli_quick_kills=$((${cli_quick_kills:-0} + 1))
  else
    cli_quick_kills=0
  fi
  if [ "$cli_quick_kills" -ge @max_kills@ ]; then
    printf '{"type":"environment.feedback","data":{"event":"harness_cli_kill_limit","message":"@label@ was ended by a signal @max_kills@ times in a row within @quick@ s of starting, so it was not started again.","exitCode":%s}}\n' "$1" >> "$log_file"
    return 1
  fi
  [ -n "$session_id" ] || session_id="$(python3 "$harness_tool" home-session @cli@ "$@cli@_log_dir")"
  harness_killed_note "$3"
  printf '{"type":"environment.feedback","data":{"event":"harness_cli_resumed","message":"@label@ was ended by a signal the harness did not send (a command of an agent, or the kernel) in the middle of its turn; the runner resumes the turn in the same session.","exitCode":%s,"attemptLog":"%s"}}\n' "$1" "${3##*/}" >> "$log_file"
  return 0
}

# The next resumption of an interrupted turn tells the agent that a signal ended its session, and which
# command was running ($1: the interrupted attempt's log), instead of only "Continue from where you left
# off."; the resumption after that is the usual one again.
harness_killed_note() {
  local pending
  pending="$(python3 "$harness_tool" pending-command @cli@ "${1:-}")"
  harness_resume_note="${harness_kill_note:-}"
  if [ -n "$pending" ]; then
    harness_resume_note="$harness_resume_note ${harness_kill_command:-}

$pending
"
  fi
  harness_resume_note="$harness_resume_note
${harness_kill_end:-}"
  harness_resume_note_file="${limit_dir:-}/@cli@-kill-note.txt"
  [ -z "${limit_dir:-}" ] || printf '%s' "$harness_resume_note" > "$harness_resume_note_file"
}

harness_resume_turn() {
  resume_@cli@_turn "$1"
  local resumed=$?
  harness_resume_note=""
  harness_resume_note_file=""
  return "$resumed"
}
"""

# With the keeper, the watchers are its own: the runner starts none and only reads their markers.
TOKEN_BUDGET_GUARD_START = r"""
start_token_budget() {
  token_budget_marker=""
  [ -n "${token_budget:-}" ] || return 0
  token_budget_marker="${1%/*}/token-budget-exhausted.json"
}

"""
TOKEN_BUDGET_GUARD_HELPERS = (TOKEN_BUDGET_GUARD_START
                              + TOKEN_BUDGET_HELPERS[TOKEN_BUDGET_HELPERS.index("token_budget_exhausted() {"):])
LIMIT_RESUME_GUARD_START = r"""
start_limit_watch() {
  limit_dir="${1%/*}"
  limit_stop_logged=0
  printf '%s' "$limit_resume_note" > "$limit_dir/@cli@-resume-note.txt"
}

"""
LIMIT_RESUME_GUARD_HELPERS = (LIMIT_RESUME_GUARD_START
                              + LIMIT_RESUME_HELPERS[LIMIT_RESUME_HELPERS.index("limit_stop_reached() {"):])
EPISODE_GUARD_START = r"""
start_episode_watch() {
  episode_over_marker="${1%/*}/episode-over.json"
  episode_identity=("$2" "$3" "$4")
}

"""
EPISODE_GUARD_HELPERS = (EPISODE_GUARD_START
                         + EPISODE_HELPERS[EPISODE_HELPERS.index("# Over if the watcher saw it"):])

# A reply the API's content filter blocked ends the CLI's turn with an error: Claude Code's "Output
# blocked by content filtering policy" (Anthropic's safeguard against reproducing existing
# material), Codex's response that came back incomplete for "content_filter" (after Codex's own
# stream retries). The runner takes it as the end of that turn: it logs a content_filter_block
# event, and the continuation loop goes on as after any turn, sending the usual continuation in the
# same session, so the model is told nothing. After CONTENT_FILTER_MAX_BLOCKS blocks in a row with
# no completed model output in between, it stops the agent cleanly (a content_filter_stop event),
# and the run is graded as usual.
CONTENT_FILTER_MAX_BLOCKS = 5

# Claude Code raises an auto-compact window (CLAUDE_CODE_AUTO_COMPACT_WINDOW) below this to it; see
# run.py's CLAUDE_COMPACT_WINDOW for the window every Claude Code agent gets by default.
CLAUDE_COMPACT_WINDOW_MIN = 100000

# Every command the harness sends must fit in one argument of execve (MAX_ARG_STRLEN, 128 KiB). A team's setup writes
# each agent's instruction beside the runner and protocol.json, which for 8 agents on a long task can exceed it ("Argument
# list too long"), so the setup goes out as consecutive commands of at most this many bytes.
SETUP_COMMAND_MAX = 100000
# The longest command `bash -c` takes: one execve argument (MAX_ARG_STRLEN, 128 KiB) less its NUL.
EXECVE_ARG_MAX = 131071
# conf.json in pieces of this many characters where the keeper's install command is too long (_guard_trial_commands).
GUARD_CONF_PIECE = 60000


def setup_commands(parts: list[str], limit: int = SETUP_COMMAND_MAX) -> list[str]:
    """PARTS as `set -euo pipefail; A && B && ...` commands of at most LIMIT bytes each, in order: one command when
    they all fit, as before. The parts share no shell state (each is a whole command on absolute paths), so where
    they are cut changes nothing but the number of execs."""
    head = "set -euo pipefail; "
    commands, current = [], []
    for part in parts:
        if len((head + part).encode()) > limit:
            raise ValueError(f"a setup command of {len(part.encode())} bytes cannot fit in one of {limit}")
        if current and len((head + " && ".join([*current, part])).encode()) > limit:
            commands.append(head + " && ".join(current))
            current = []
        current.append(part)
    if current:
        commands.append(head + " && ".join(current))
    return commands

# Whether the attempt that just failed ($1) ended on a block: if so, count and log it and return 0
# (the turn is over), else return 1. A completed model response in the attempt (Claude Code: a tool
# ran; Codex: an item completed) restarts the count of blocks in a row, as does a turn that ends
# without error (the retry function).
CONTENT_FILTER_HELPERS = r"""
content_filter_block() {
  local outcome
  outcome="$(@cli@_attempt_outcome "$1")"
  case "$outcome" in *output*) content_filter_in_a_row=0 ;; esac
  case "$outcome" in blocked*) ;; *) return 1 ;; esac
  content_filter_in_a_row=$((${content_filter_in_a_row:-0} + 1))
  content_filter_total=$((${content_filter_total:-0} + 1))
  content_filter_turn_blocked=1
  printf '{"type":"environment.feedback","data":{"event":"content_filter_block","message":"The API blocked a reply of @label@ (content filter), which ended the turn; the agent continues with the usual continuation.","consecutiveBlocks":%s,"totalBlocks":%s,"maxConsecutiveBlocks":@max_blocks@,"attemptLog":"%s"}}\n' "$content_filter_in_a_row" "$content_filter_total" "${1##*/}" >> "$log_file"
  if [ "$content_filter_in_a_row" -ge @max_blocks@ ]; then
    printf '{"type":"environment.feedback","data":{"event":"content_filter_stop","message":"The API blocked @max_blocks@ replies of @label@ in a row (content filter) with no completed output in between, so @label@ was stopped.","consecutiveBlocks":%s,"totalBlocks":%s}}\n' "$content_filter_in_a_row" "$content_filter_total" >> "$log_file"
    exit 0
  fi
  return 0
}
"""

# Prints "blocked" if the attempt log ($1) ends on a block (else "other"), and " output" if the model
# completed any output in it. Both read the CLI's own events, never tool output.
CLAUDE_ATTEMPT_OUTCOME = r"""
claude_attempt_outcome() {
  python3 - "$1" <<'PY'
import json
import sys

BLOCKED = "Output blocked by content filtering policy"
blocked = output = False
try:
    lines = open(sys.argv[1], errors="replace")
except OSError:
    lines = ()
for line in lines:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if not isinstance(event, dict):
        continue
    message = event.get("message")
    if event.get("type") == "user" and isinstance(message, dict) and isinstance(message.get("content"), list):
        output = output or any(isinstance(block, dict) and block.get("type") == "tool_result"
                               for block in message["content"])
    elif event.get("type") == "result":
        texts = [str(event.get("result") or "")] + [str(error) for error in event.get("errors") or ()]
        blocked = bool(event.get("is_error")) and any(BLOCKED in text for text in texts)
print(("blocked" if blocked else "other") + (" output" if output else ""))
PY
}
"""

CODEX_ATTEMPT_OUTCOME = r"""
codex_attempt_outcome() {
  python3 - "$1" <<'PY'
import json
import sys

BLOCKED = "Incomplete response returned, reason: content_filter"
blocked = output = False
try:
    lines = open(sys.argv[1], errors="replace")
except OSError:
    lines = ()
for line in lines:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if not isinstance(event, dict):
        continue
    item = event.get("item")
    if event.get("type") == "item.completed" and isinstance(item, dict):
        output = output or item.get("type") not in ("error", "reasoning")
    elif event.get("type") == "turn.failed":
        error = event.get("error")
        blocked = BLOCKED in str(error.get("message") if isinstance(error, dict) else error)
    elif event.get("type") == "turn.completed":
        blocked = False
print(("blocked" if blocked else "other") + (" output" if output else ""))
PY
}
"""

# A task can add flags to the Claude Code or Codex command line: its container sets
# RELEASE_TASK_CLI_HOOK to a bash file, which the configure step sources with the CLI's
# name ($agent_id and the CLI home are in scope) and which fills task_cli_args (BrowseComp
# uses it to turn on Codex's web search). Without the variable nothing changes.
TASK_CLI_HOOK_HELPERS = r"""
run_task_cli_hook() {
  task_cli_args=()
  [ -n "${RELEASE_TASK_CLI_HOOK:-}" ] || return 0
  if [ ! -r "$RELEASE_TASK_CLI_HOOK" ]; then
    echo "task CLI hook $RELEASE_TASK_CLI_HOOK is not readable" >&2
    return 1
  fi
  . "$RELEASE_TASK_CLI_HOOK" "$1"
}
"""

CLAUDE_RETRY_CLASSIFIER = r"""
is_retryable_claude_api_error() {
  python3 - "$1" <<'PY'
import json
import re
import sys

RETRYABLE = re.compile(
    r"\b(401|408|409|429|500|502|503|504|529)\b|rate.?limit|overloaded|usage limit|"
    r"hit your\b.*\blimit|limit reached|too many requests|temporarily limiting|"
    r"timed? ?out|ECONNRESET|ECONNREFUSED|ETIMEDOUT|EAI_AGAIN|ENOTFOUND|socket hang up|"
    r"fetch failed|network|connection error|unauthori[sz]ed|authentication_error|"
    r"oauth|token (?:has )?expired",
    re.IGNORECASE,
)
PERMANENT = re.compile(
    r"usage credits required|not yet available for this subscription|"
    r"invalid_request_error|not_found_error|invalid model|model .{0,80}not (?:found|available)",
    re.IGNORECASE,
)
texts = []
try:
    lines = open(sys.argv[1], errors="replace")
except OSError:
    sys.exit(1)
for line in lines:
    line = line.strip()
    if not line:
        continue
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        texts.append(line)
        continue
    if not isinstance(event, dict):
        continue
    if event.get("type") == "result" and event.get("is_error"):
        texts.append(str(event.get("result") or ""))
        texts.append(str(event.get("api_error_status") or ""))
        texts.extend(str(error) for error in event.get("errors") or ())
    elif event.get("type") == "rate_limit_event":
        info = event.get("rate_limit_info") or {}
        if isinstance(info, dict) and info.get("status") == "rejected":
            texts.append("rate limit rejected")
text = "\n".join(texts)
sys.exit(0 if RETRYABLE.search(text) and not PERMANENT.search(text) else 1)
PY
}
"""

CODEX_HELPERS = r"""
configure_codex_command() {
  codex_command=("$codex_binary")
  mkdir -p "$codex_log_dir"
  cat > "$codex_log_dir/config.toml" <<'TOML'
@codex_config_toml@TOML
  if [ -e @codex_auth@ ]; then
    ln -sfn @codex_auth@ "$codex_log_dir/auth.json"
  fi
  if [ -z "${CODEX_API_KEY:-}" ]; then
    if [ ! -r "$codex_log_dir/auth.json" ]; then
      echo "Codex has no credentials: mount auth.json at @codex_auth@ (readable by uid $(id -u)) or set CODEX_API_KEY" >&2
      return 1
    fi
    if [ ! -w "$codex_log_dir/auth.json" ]; then
      echo "warning: @codex_auth@ is read-only, so Codex cannot save refreshed tokens" >&2
    fi
  fi
  codex_effort="${MULTIAGENT_CODEX_EFFORT:-}"
  if [ -n "$codex_effort" ]; then
    codex_effort="$(CODEX_HOME="$codex_log_dir" timeout 120 "$codex_binary" debug models 2>/dev/null \
      | python3 -c "$codex_effort_clamp_source" "$model" "$codex_effort")"
    printf '{"model":"%s","requested":"%s","effective":"%s"}\n' \
      "$model" "$MULTIAGENT_CODEX_EFFORT" "$codex_effort" > "$codex_log_dir/reasoning-effort.json"
  fi
  start_token_budget "$codex_log_dir"
  run_task_cli_hook codex || return 1
}

extract_codex_session_id() {
  local source_log="$1"
  python3 - "$source_log" <<'PY'
import json
import sys

session_id = ""
try:
    lines = open(sys.argv[1], errors="replace")
except OSError:
    lines = ()
for line in lines:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if isinstance(event, dict) and event.get("type") == "thread.started":
        value = event.get("thread_id")
        if isinstance(value, str) and value:
            session_id = value
if session_id:
    print(session_id)
PY
}

# A turn that failed after its last completion is a failed attempt even if the
# CLI exited 0.
finish_codex_attempt() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

code = int(sys.argv[2]) if sys.argv[2].isdigit() else 1
failed = False
try:
    lines = open(sys.argv[1], errors="replace")
except OSError:
    lines = ()
for line in lines:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if not isinstance(event, dict):
        continue
    if event.get("type") in ("turn.failed", "error"):
        failed = True
    elif event.get("type") == "turn.completed":
        failed = False
sys.exit(1 if code == 0 and failed else code)
PY
}
"""

CODEX_RETRY_CLASSIFIER = r"""
is_retryable_codex_api_error() {
  python3 - "$1" <<'PY'
import json
import re
import sys

RETRYABLE = re.compile(
    r"\b(401|408|409|429|500|502|503|504)\b|rate.?limit|usage limit|too many requests|"
    r"overloaded|capacity|server_error|internal server error|service unavailable|"
    r"stream disconnected|stream error|timed? ?out|connection|network|"
    r"unauthori[sz]ed|refresh token|token (?:is )?expired|try again",
    re.IGNORECASE,
)
PERMANENT = re.compile(
    r"unsupported_value|invalid_request_error|model_not_found|does not exist|"
    r"not supported (?:with|by|for)",
    re.IGNORECASE,
)
texts = []
try:
    lines = open(sys.argv[1], errors="replace")
except OSError:
    sys.exit(1)
for line in lines:
    line = line.strip()
    if not line:
        continue
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        texts.append(line)
        continue
    if not isinstance(event, dict):
        continue
    if event.get("type") == "error":
        texts.append(str(event.get("message") or ""))
    elif event.get("type") == "turn.failed":
        error = event.get("error")
        texts.append(str(error.get("message") if isinstance(error, dict) else error))
text = "\n".join(texts)
sys.exit(0 if RETRYABLE.search(text) and not PERMANENT.search(text) else 1)
PY
}
"""

# Picks the strongest reasoning effort the model supports that does not exceed
# the requested one (Codex rejects unsupported values with a 400). Reads the
# `codex debug models` catalog on stdin; argv is model and requested effort.
CODEX_EFFORT_CLAMP_SOURCE = """\
import json
import sys

model, requested = sys.argv[1], sys.argv[2]
order = ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]
try:
    catalog = json.load(sys.stdin)
except ValueError:
    catalog = {}
entries = catalog.get("models", []) if isinstance(catalog, dict) else catalog
levels = []
for entry in entries if isinstance(entries, list) else []:
    if isinstance(entry, dict) and entry.get("slug") == model:
        for level in entry.get("supported_reasoning_levels") or []:
            name = level.get("effort") if isinstance(level, dict) else level
            if name in order:
                levels.append(name)
if not levels or requested in levels or requested not in order:
    print(requested)
else:
    lower = [level for level in levels if order.index(level) <= order.index(requested)]
    print(max(lower, key=order.index) if lower else min(levels, key=order.index))
"""


class CliAgent(BaseInstalledAgent):
    """Runs one or more headless CLI sessions through the shared runner.

    Subclasses describe a CLI: how to install it, the bash that invokes it, how
    to resume a session, and how to recognize retryable API errors.
    """

    CLI = ""
    CLI_LABEL = ""
    BINARY_ENV = ""
    DEFAULT_BINARY = ""
    LOG_DIR_NAME = ""
    RESUME_ARGS = ""
    VERSION_COMMAND = ""
    GH_STUB_MESSAGE = "gh is disabled in this benchmark container."
    CLI_HELPERS = ""
    RETRY_CLASSIFIER = ""
    # Whether the CLI supports --token-budget; its configure step then calls
    # start_token_budget with its home directory.
    TOKEN_BUDGET = False
    # The variable that points the CLI at its home, which every process it starts
    # inherits; the episode watcher stops the processes that carry it.
    HOME_ENV_NAME = ""
    # Whether its configure step calls run_task_cli_hook (TASK_CLI_HOOK_HELPERS).
    TASK_CLI_HOOK = False
    # Whether each agent's task instruction reaches run-agent.sh as a file instead of an
    # argument. As an argument it sits in the command line of run-agent.sh and of the shell
    # that starts it, so an agent's own `pkill -f solution` (a pattern the instruction
    # matches) stops its runner.
    INSTRUCTION_FILE = False
    # Stop an agent that has given up: after this many continuation turns in a row that each
    # last under GIVE_UP_SECONDS and run at most GIVE_UP_TOOLS tool calls (an agent that only
    # repeats "I'm done"), the runner ends it and logs an agent_gave_up event. Each such turn
    # still costs a full re-read of the conversation. 0 = never.
    GIVE_UP_TURNS = 0
    GIVE_UP_SECONDS = 60
    GIVE_UP_TOOLS = 2
    # Bash that prints the number of tool calls in one attempt log ($1).
    TOOL_CALL_COUNT = ""
    # Whether the CLI supports --resources-tool (RESOURCES_TOOL_ENV).
    RESOURCES_TOOL = False
    # Whether the CLI supports usage-limit resumption (LIMIT_RESUME_ENV), and its bash that
    # resumes an interrupted turn without the turn's prompt (resume_<cli>_turn).
    LIMIT_RESUME = False
    LIMIT_RESUME_CLI_HELPERS = ""
    # Its bash that tells whether an attempt ended on a content-filter block (<cli>_attempt_outcome);
    # with it the runner takes a block as the end of the turn (CONTENT_FILTER_HELPERS). "" = no
    # handling.
    CONTENT_FILTER_OUTCOME = ""
    # Whether the CLI supports harness protection (HARNESS_GUARD_ENV).
    HARNESS_GUARD = False
    # Whether the CLI has a message board for the protocols that use one (message_board). Only Codex does;
    # every other agent, a mixed team included, refuses those protocols.
    MESSAGE_BOARD = False
    # Whether the CLI messages its teammates natively for claude_peers (claude_peers). Only Claude Code does; every other
    # agent, a mixed team included, refuses it.
    CLAUDE_PEERS = False
    # Whether the CLI can join the harness's team chat for swarmchat (swarmchat): a shell command and a PostToolUse
    # hook. Claude Code, Codex and a mixed team of them do.
    SWARMCHAT = False
    # Whether the CLI runs ultra mode (its own sub-agents at the run's effort, ultra.py). Only Codex does.
    ULTRA = False
    # The variables that hold the CLI's login. A mixed team's other agents do not get them (unset_env).
    LOGIN_ENV: tuple[str, ...] = ()
    # Set by MixedTeamAgent on each of its agents (run.py --team): the agent's effort (None: run.py's
    # --effort, MULTIAGENT_AGENT_EFFORT) and the variables its runner unsets (the other CLIs' logins).
    effort: str | None = None
    unset_env: tuple[str, ...] = ()
    # The runner the last run() or resume() rendered with the keeper (revive.py records it).
    last_runner_script: str | None = None
    # Set by revive.py before resume(): {"revival": n, "note": text, "layout": "team" or
    # "single", "started": the epoch the runner counts working time from, "agents": [{"index",
    # "session_id", "turn"}, ...] (the agents to resume), "checked": [] (filled by the checks),
    # optionally "event_message" (the trial_revived event's text) and "checks_passed" (revive.py's
    # dry run made the checks; the agent phase does not read the run's files again)}.
    revive_plan: dict[str, Any] | None = None

    async def install(self, environment: BaseEnvironment) -> None:
        if _frontier_swe():
            # FrontierSWE images are complete and offline (the agent may reach only its
            # model API), and installing packages would change them: git-to-zig keeps
            # real git root-only. Install the harness tools instead.
            await self.exec_as_root(environment, command=f"python3 {_harness_tools()} setup")
        else:
            await self.exec_as_root(environment, command=(
                "apt-get update && apt-get install -y --no-install-recommends "
                "ca-certificates curl git ripgrep"), env={"DEBIAN_FRONTEND": "noninteractive"})
        await self._install_cli(environment)

    async def _install_cli(self, environment: BaseEnvironment) -> None:
        """The agent's own part of install(): the gh stub and the link to the mounted CLI."""
        await self.exec_as_agent(environment, command=(
            "set -e; mkdir -p $HOME/.local/bin; "
            "printf '%s\n' '#!/usr/bin/env bash' "
            f"'echo \"{self.GH_STUB_MESSAGE}\" >&2' "
            "'exit 127' > $HOME/.local/bin/gh; chmod 700 $HOME/.local/bin/gh; "
            f"test -x {self.DEFAULT_BINARY}; "
            f"ln -sf {self.DEFAULT_BINARY} $HOME/.local/bin/{self.CLI}"))

    def get_version_command(self) -> str | None:
        return self.VERSION_COMMAND

    def parse_version(self, stdout: str) -> str:
        return stdout.strip()

    def _cli_model(self) -> str:
        return (self.model_name or "claude-sonnet-4.6").split("/")[-1]

    def _build_env(self, protocol_import: str) -> dict[str, str]:
        raise NotImplementedError

    def _runner_locals(self) -> list[str]:
        """Extra shell assignments placed before the CLI is configured. Values
        may span lines, so callers indent only the start of each assignment."""
        if not self.TOKEN_BUDGET:
            return []
        budget = os.environ.get("RELEASE_TOKEN_BUDGET", "").strip()
        if budget and not budget.isdigit():
            raise ValueError(f"RELEASE_TOKEN_BUDGET must be a whole number: {budget!r}")
        metric = os.environ.get("RELEASE_TOKEN_BUDGET_METRIC", "weighted").strip()
        if metric not in BUDGET_METRICS:
            raise ValueError(f"RELEASE_TOKEN_BUDGET_METRIC must be one of {BUDGET_METRICS}: {metric!r}")
        return [
            f"token_budget={budget}",
            f"token_budget_metric={metric}",
            # With the keeper, the watchers are files in GUARD_DIR, not sources the runner carries.
            *([] if self._guard() else [f"token_budget_source={shlex.quote(TOKEN_BUDGET_SOURCE)}"]),
        ]

    def _episode_locals(self) -> list[str]:
        """The episode watcher's settings, when the task has an episode-over command."""
        command = _episode_over_command()
        if not command:
            return []
        return [
            f"episode_over_command={shlex.quote(command)}",
            f"episode_home_env={shlex.quote(self.HOME_ENV_NAME)}",
            f"episode_poll_seconds={EPISODE_POLL_SECONDS}",
            *([] if self._guard() else [f"episode_watch_source={shlex.quote(EPISODE_WATCH_SOURCE)}"]),
        ]

    def _guard(self) -> bool:
        if not _harness_guard_on():
            return False
        if not self.HARNESS_GUARD:
            raise ValueError(f"{HARNESS_GUARD_ENV}=1 is only supported with Codex and Claude Code agents")
        return True

    def _start_episode_watch(self, identity: str, indent: str) -> str:
        """The line that starts the episode watcher, after the CLI is configured."""
        if not _episode_over_command():
            return ""
        return f'{indent}start_episode_watch "${self.CLI}_log_dir" {identity}\n'

    def _agent_effort(self) -> str:
        """The agent's effort: its own in a mixed team, else run.py's --effort."""
        return self.effort if self.effort is not None else os.environ.get("MULTIAGENT_AGENT_EFFORT", "").strip()

    def _member_locals(self) -> list[str]:
        """A mixed team's agent: its runner unsets the other CLIs' logins. Nothing otherwise."""
        return [f"unset {' '.join(self.unset_env)}"] if self.unset_env else []

    def _effort_flag(self) -> str:
        agent_effort = self._agent_effort()
        # Apply --effort only to models that support it: Haiku rejects reasoning-effort
        # configuration, so skip it for haiku models.
        if agent_effort:
            _eff = shlex.quote("--effort " + agent_effort)
            return f'$(case "$model" in auto|*haiku*) ;; *) printf %s {_eff} ;; esac) '
        return ""

    def _context_flag(self) -> str:
        return ""

    def _invoke_lines(self, identity_env: str, code_assignment: str) -> list[str]:
        """The invoke_<cli> function, starting at column zero."""
        raise NotImplementedError

    def _invoke_function(self, identity_env: str, code_assignment: str, indent: str) -> str:
        return textwrap.indent("\n".join(self._invoke_lines(identity_env, code_assignment)), indent)

    def _after_attempt(self) -> str:
        """Lines run after each attempt, before its exit code is examined."""
        lines = ""
        if self.TOKEN_BUDGET:
            lines += (
                "    if token_budget_exhausted; then\n"
                "      return 0\n"
                "    fi\n"
            )
        if _episode_over_command():
            lines += (
                "    if episode_over; then\n"
                "      return 0\n"
                "    fi\n"
            )
        if self._limit_resume():
            lines += (
                "    if limit_stop_reached; then\n"
                "      return 0\n"
                "    fi\n"
            )
        if self._guard():
            lines += (
                "    if harness_stop_reached; then\n"
                "      return 0\n"
                "    fi\n"
            )
        return lines

    def _before_continuation(self, indent: str) -> str:
        """Lines run before each continuation attempt."""
        lines = ""
        if self.TOKEN_BUDGET:
            lines += (
                "if token_budget_exhausted; then\n"
                '  append_token_budget_event "$log_file"\n'
                "  exit 0\n"
                "fi\n"
            )
        if _episode_over_command():
            lines += (
                "if episode_over; then\n"
                '  append_episode_over_event "$log_file"\n'
                "  exit 0\n"
                "fi\n"
            )
        if self._limit_resume():
            lines += (
                "if limit_stop_reached; then\n"
                "  exit 0\n"
                "fi\n"
            )
        if self._guard():
            lines += (
                "if harness_stop_reached; then\n"
                "  exit 0\n"
                "fi\n"
            )
        return textwrap.indent(lines, indent) if lines else ""

    def _board(self) -> bool:
        """Whether the protocol runs the team on the message board; raises for an agent without one."""
        import_path = self._protocol_import()
        if not _board_protocol(import_path):
            return False
        if not self.MESSAGE_BOARD:
            raise ValueError(f"{import_path} runs the team on Codex's message board, and {self.CLI_LABEL} has none "
                             "(the board protocol is for --cli codex teams, not Claude Code or --team)")
        return True

    def _peers(self) -> bool:
        """Whether the protocol runs the team on Claude Code's cross-session messaging; raises for an agent without it."""
        import_path = self._protocol_import()
        if not _claude_peers_protocol(import_path):
            return False
        if not self.CLAUDE_PEERS:
            raise ValueError(f"{import_path} runs the team on Claude Code's cross-session messaging, and "
                             f"{self.CLI_LABEL} has none (claude_peers is for --cli claude teams, not Codex "
                             "or --team)")
        return True

    def _swarmchat(self) -> bool:
        """Whether the protocol runs the team on the harness's team chat; raises for an agent that cannot join it."""
        import_path = self._protocol_import()
        if not _swarmchat_protocol(import_path):
            return False
        if not self.SWARMCHAT:
            raise ValueError(f"{import_path} runs the team on the harness's team chat, which {self.CLI_LABEL} cannot "
                             "join (swarmchat is for --cli claude, --cli codex and --team)")
        return True

    def _ultra(self) -> int | None:
        """N in ultra mode (ultra.ENV: one Codex agent and its own helpers, at most N threads at once, the root
        included), else None; raises for half a setting or an agent without the mode."""
        agents = ultra.agents_from_env()
        protocol = _ultra_protocol(self._protocol_import())
        if agents is None and not protocol:
            return None
        if agents is None or not protocol:
            raise ValueError(f"ultra mode needs both the ultra protocol and {ultra.ENV}: run.py --protocol ultra "
                             "--agents N sets them")
        if not self.ULTRA:
            raise ValueError(f"ultra mode runs Codex's own sub-agents, and {self.CLI_LABEL} has none here "
                             "(--protocol ultra is for --cli codex)")
        return agents

    def _ultra_prompt(self, text: str) -> str:
        """An agent's task instruction: in ultra mode the cap line first (ultra.prompt), else as it is."""
        agents = self._ultra()
        return text if agents is None else ultra.prompt(text, agents)

    def _cli_helpers(self) -> str:
        """The CLI's bash helpers (its configure step among them)."""
        return self.CLI_HELPERS

    def _action_feedback_helpers(self) -> str:
        guard = self._guard()
        parts = (HARNESS_GUARD_HELPERS,) if guard else ()
        parts += (self._cli_helpers(), CONTINUATION_HELPERS, self.RETRY_CLASSIFIER, API_RETRY_HELPERS)
        if self.TOKEN_BUDGET:
            parts += (TOKEN_BUDGET_GUARD_HELPERS if guard else TOKEN_BUDGET_HELPERS,)
        if _episode_over_command():
            parts += (EPISODE_GUARD_HELPERS if guard else EPISODE_HELPERS,)
        if self.TASK_CLI_HOOK:
            parts += (TASK_CLI_HOOK_HELPERS,)
        if self._limit_resume():
            parts += (LIMIT_RESUME_GUARD_HELPERS if guard else LIMIT_RESUME_HELPERS, self.LIMIT_RESUME_CLI_HELPERS)
        if self._team_hold():
            hold = (TEAM_HOLD_HELPERS.replace("@hold_request@", team_hold.REQUEST).replace("@held@", team_hold.HELD)
                    .replace("@hold_step@", str(TEAM_HOLD_STEP_SECONDS)))
            test = _team_hold_test()
            if test is not None:
                hold += (TEAM_HOLD_TEST_HELPERS.replace("@test_after@", str(test[0]))
                         .replace("@test_seconds@", str(test[1])).replace("@test_env@", TEAM_HOLD_TEST_ENV))
            parts += (hold,)
        if self._lockstep():
            held = f'[ -e "$limit_dir/{team_hold.HELD}" ]' if self._team_hold() else "return 1"
            parts += (LOCKSTEP_HELPERS.replace("@lockstep@", lockstep.PAUSE).replace("@held_test@", held)
                      .replace("@held@", team_hold.HELD).replace("@ended@", " ".join(LOCKSTEP_ENDED))
                      .replace("@hold_step@", str(TEAM_HOLD_STEP_SECONDS))
                      .replace("@lockstep_step@", str(LOCKSTEP_STEP_SECONDS)),)
        if self.CONTENT_FILTER_OUTCOME:
            parts += (self.CONTENT_FILTER_OUTCOME, CONTENT_FILTER_HELPERS)
        helpers = "\n\n".join(part.strip() for part in parts)
        if guard:
            # A turn resumed after a kill gets KILL_RESUME_NOTE (harness_killed_note) as the CLI's resume text.
            helpers = (helpers
                       .replace('CLAUDE_CODE_RESUME_PROMPT="$limit_resume_note"',
                                'CLAUDE_CODE_RESUME_PROMPT="${harness_resume_note:-$limit_resume_note}"')
                       .replace('invoke_codex "$limit_dir/codex-resume-note.txt"',
                                'invoke_codex "${harness_resume_note_file:-$limit_dir/codex-resume-note.txt}"'))
        return (
            helpers.replace("@cli@", self.CLI)
            .replace("@CLI@", self.CLI.upper())
            .replace("@label@", self.CLI_LABEL)
            .replace("@home_env@", self.HOME_ENV_NAME)
            .replace("@max_blocks@", str(getattr(self, "content_filter_max_blocks", None) or CONTENT_FILTER_MAX_BLOCKS))
            .replace("@guard_dir@", GUARD_DIR)
            .replace("@stop_marker@", harness_guard.STOP_MARKER)
            .replace("@exit_file@", harness_guard.EXIT_FILE)
            .replace("@quick@", str(CLI_QUICK_SECONDS))
            .replace("@max_kills@", str(CLI_MAX_QUICK_KILLS))
            .replace("__NO_ACTION_FEEDBACK_BODY__", _no_action_feedback_body())
        )

    def _content_filter_parts(self) -> tuple[str, str, str]:
        """Bash for content-filter blocks, spliced into run_<cli>_with_api_retries: (at its start,
        when an attempt succeeds, before the retryable-error check). Three empty strings for a CLI
        without CONTENT_FILTER_OUTCOME, so its runner is unchanged."""
        if not self.CONTENT_FILTER_OUTCOME:
            return "", "", ""
        return ("  content_filter_turn_blocked=0\n",
                "      content_filter_in_a_row=0\n",
                '    if content_filter_block "$attempt_log_file"; then\n'
                "      return 0\n"
                "    fi\n")

    def _limit_resume(self) -> bool:
        if not _limit_resume_on():
            return False
        if not self.LIMIT_RESUME:
            raise ValueError(f"{LIMIT_RESUME_ENV}=1 is only supported with Codex and Claude Code agents")
        return True

    def _team_hold(self) -> bool:
        if not _team_hold_on():
            return False
        if not (self._limit_resume() and self._guard()):
            raise ValueError(f"{TEAM_HOLD_ENV}=1 needs {LIMIT_RESUME_ENV}=1 and {HARNESS_GUARD_ENV}=1")
        return True

    def _lockstep(self) -> bool:
        if _lockstep_settings() is None:
            return False
        if not (self._limit_resume() and self._guard()) or not os.environ.get("RELEASE_TOKEN_BUDGET", "").strip():
            raise ValueError(f"{LOCKSTEP_GAP_ENV} needs {LIMIT_RESUME_ENV}=1, {HARNESS_GUARD_ENV}=1 and a token "
                             "budget (RELEASE_TOKEN_BUDGET)")
        return True

    def _team_hold_parts(self) -> tuple[str, str, str, str]:
        """Bash for the team limit hold and lockstep, spliced into run_<cli>_with_api_retries with usage-limit
        resumption: (before each attempt, after it, after the runner logs a usage-limit wait, after any
        wait). Four empty strings when both are off, so the runner is unchanged."""
        cli = self.CLI
        if self._lockstep():
            hold = self._team_hold()
            return ("    team_pause || return 0\n",
                    '    if [ "$code" -ne 0 ] && team_paused; then\n'
                    '      team_pause_stopped "$attempt_log_file" || return 0\n'
                    "      retry=$((retry + 1))\n"
                    f'      resume_input="$(python3 -c "$limit_resume_source" input {cli} "${cli}_log_dir" '
                    '"$session_id" "$attempt_prompt_file" "$chain_started")"\n'
                    '      if [ -n "$session_id" ] && [ "$resume_input" = resume ]; then\n'
                    '        append_turn_resumed_event "$log_file" "$retry"\n'
                    "      fi\n"
                    "      continue\n"
                    "    fi\n",
                    '      team_hold_request "$delay"\n' if hold else "",
                    "    team_hold_request_end\n" if hold else "")
        if not self._team_hold():
            return "", "", "", ""
        return ("    team_hold_pause || return 0\n",
                '    if [ "$code" -ne 0 ] && team_held; then\n'
                '      team_hold_stopped "$attempt_log_file" || return 0\n'
                "      retry=$((retry + 1))\n"
                f'      resume_input="$(python3 -c "$limit_resume_source" input {cli} "${cli}_log_dir" "$session_id" '
                '"$attempt_prompt_file" "$chain_started")"\n'
                '      if [ -n "$session_id" ] && [ "$resume_input" = resume ]; then\n'
                '        append_turn_resumed_event "$log_file" "$retry"\n'
                "      fi\n"
                "      continue\n"
                "    fi\n",
                '      team_hold_request "$delay"\n',
                "    team_hold_request_end\n")

    def _limit_locals(self, started: float, wait_max_left: int | None = None) -> list[str]:
        """The runner's settings for usage-limit resumption: the source of limit_resume.py, the
        note Codex resumes with, when Harbor started the agents, the working time (--timeout,
        empty if run.py gave none) and the most waiting allowed (a relay's later legs: WAIT_MAX_LEFT,
        what the earlier legs left of it)."""
        if not self._limit_resume():
            return []
        timeout = os.environ.get(AGENT_TIMEOUT_ENV, "").strip()
        wait_max = (str(wait_max_left) if wait_max_left is not None
                    else os.environ.get(LIMIT_WAIT_MAX_ENV, "").strip() or str(DEFAULT_LIMIT_WAIT_MAX))
        for name, value in ((AGENT_TIMEOUT_ENV, timeout), (LIMIT_WAIT_MAX_ENV, wait_max)):
            if value and not value.replace(".", "", 1).isdigit():
                raise ValueError(f"{name} must be a number of seconds: {value!r}")
        return [
            f"limit_resume_source={shlex.quote(LIMIT_RESUME_SOURCE)}",
            f"limit_resume_note={shlex.quote(LIMIT_RESUME_NOTE)}",
            *([f"harness_kill_note={shlex.quote(KILL_RESUME_NOTE)}",
               f"harness_kill_command={shlex.quote(KILL_RESUME_COMMAND)}",
               f"harness_kill_end={shlex.quote(KILL_RESUME_END)}"] if self._guard() else []),
            f"limit_started={started:.3f}",
            f"limit_timeout={shlex.quote(timeout)}",
            f"limit_wait_max={int(float(wait_max))}",
            f"limit_poll_seconds={LIMIT_POLL_SECONDS}",
        ]

    def _start_limit_watch(self, indent: str) -> str:
        """The line that starts the limit watcher, after the CLI is configured."""
        if not self._limit_resume():
            return ""
        return f'{indent}start_limit_watch "${self.CLI}_log_dir"\n'

    async def _raise_if_time_cap_reached(self, environment: BaseEnvironment, session_dirs: list[str]) -> None:
        """An agent that used its working time was stopped by its runner, which then ended
        cleanly; report it as Harbor reports an agent that ran out of time."""
        if not self._limit_resume():
            return
        markers = " ".join(shlex.quote(f"{session_dir}/{LIMIT_TIME_CAP_MARKER}") for session_dir in session_dirs)
        result = await environment.exec(command=f"for marker in {markers}; do [ -e \"$marker\" ] && echo \"$marker\"; done; true")
        stopped = (getattr(result, "stdout", "") or "").split()
        if stopped:
            timeout = os.environ.get(AGENT_TIMEOUT_ENV, "").strip()
            raise asyncio.TimeoutError(
                f"{len(stopped)} agent(s) used their working time ({timeout} s, not counting usage-limit waits)")

    def _revive_locals(self, revive: dict[str, Any] | None) -> list[str]:
        """A revived runner's settings: the note, and each resumed agent's session id and the
        turn it was in, indexed by agent index (the single-agent runner is index 0)."""
        if revive is None:
            return []
        agents = sorted(revive["agents"], key=lambda agent: int(agent["index"]))
        sessions = " ".join(f"[{int(a['index'])}]={shlex.quote(str(a['session_id']))}" for a in agents)
        turns = " ".join(f"[{int(a['index'])}]={int(a['turn'])}" for a in agents)
        return [f"revive_note={shlex.quote(revive['note'])}", f"revive_session_ids=({sessions})",
                f"revive_turns=({turns})"]

    def _start_block(self, revive: dict[str, Any] | None, session_dir: str, index: str, indent: str,
                     submissions: str = "") -> tuple[str, str]:
        """The runner's first attempt and the number of its first continuation turn. A run sends
        the task prompt as turn 0 and continues from turn 1. A revived runner keeps the agent's
        log, resumes its session with the note (the only new user message), logs a trial_revived
        event, and continues from the turn after the one the agent was in."""
        cli = self.CLI
        if revive is None:
            return (f'{indent}: > "$log_file"\n'
                    f'{indent}session_id=""\n'
                    f"{submissions}"
                    f'{indent}attempt_log="{session_dir}/{cli}-attempt-0.jsonl"\n'
                    f'{indent}run_{cli}_with_api_retries "$prompt_file" "$attempt_log"\n'), "1"
        revival = int(revive["revival"])
        # revive.py's message for a revival that was not a host restart (its plan's event_message).
        message = revive.get("event_message") or (f"The host of this trial died; its containers were restarted and "
                                                  f"{self.CLI_LABEL} continues the same session with one note.")
        if any(char in message for char in "\\\"'%"):
            raise ValueError(f"a revival's event message cannot hold quotes, backslashes or %: {message!r}")
        event = ('{"type":"environment.feedback","data":{"event":"' + REVIVE_EVENT + '","message":"'
                 f'{message}","revival":{revival},"turn":%s,"sessionId":"%s"}}}}\\n')
        return (f'{indent}session_id="${{revive_session_ids[{index}]}}"\n'
                f"{submissions}"
                f'{indent}revive_turn="${{revive_turns[{index}]}}"\n'
                f'{indent}revive_note_file="{session_dir}/{cli}-revive-{revival}-note.txt"\n'
                f'{indent}printf %s "$revive_note" > "$revive_note_file"\n'
                f"{indent}printf '{event}' \"$revive_turn\" \"$session_id\" >> \"$log_file\"\n"
                f'{indent}attempt_log="{session_dir}/{cli}-attempt-$revive_turn-revive-{revival}.jsonl"\n'
                f'{indent}run_{cli}_with_api_retries "$revive_note_file" "$attempt_log"\n'), "$((revive_turn + 1))"

    def _check_revived_files(self, revive: dict[str, Any], expected: dict[str, str]) -> None:
        """Before a revived runner starts: the files the run wrote from its task instruction and
        prompts (container paths under the agent directory) must be byte-identical to what this
        code writes from the same settings, or the revived agent would not be continuing the
        same task. Raises before anything is sent to the container."""
        agent_dir = EnvironmentPaths.agent_dir.as_posix()
        for path, text in expected.items():
            local = pathlib.Path(self.logs_dir) / posixpath.relpath(path, agent_dir)
            try:
                recorded = local.read_bytes()
            except OSError as exc:
                raise RuntimeError(f"cannot revive: the run's {path} is unreadable ({exc})") from exc
            if recorded != text.encode():
                raise RuntimeError(f"cannot revive: the run's {path} differs from what this code writes from the "
                                   "same settings (different task, prompts or run.py arguments)")
            revive.setdefault("checked", []).append(
                {"path": path, "bytes": len(recorded), "sha256": hashlib.sha256(recorded).hexdigest()})

    def _check_revived_continuation(self, revive: dict[str, Any], runner: pathlib.Path, source: str) -> None:
        """The continuation prompt this code would send must be the one the run's own runner sent
        (RUNNER: the run's runner, or Harbor's trial.log with the command it ran)."""
        try:
            runner_text = runner.read_text(errors="replace")
        except OSError as exc:
            raise RuntimeError(f"cannot revive: the run's {source} is unreadable ({exc})") from exc
        body = f"<<'EOF'\n{_no_action_feedback_body()}\nEOF\n"
        if body not in runner_text:
            raise RuntimeError(f"cannot revive: the continuation prompt differs from the one in the run's {source}")
        revive.setdefault("checked", []).append(
            {"path": source, "continuation_sha256": hashlib.sha256(_no_action_feedback_body().encode()).hexdigest()})

    def _give_up_parts(self, indent: str) -> tuple[str, str, str]:
        """Bash for GIVE_UP_TURNS, spliced into the continuation loop: (before the loop, before
        each continuation turn, after it). Three empty strings when it is off, so the runner
        is byte-identical to upstream's."""
        if not self.GIVE_UP_TURNS:
            return "", "", ""
        i1, i2 = indent, indent + "  "
        # A turn a content-filter block ended never counts as idle (the agent did not give up);
        # if it worked long enough, it resets the count like any other turn.
        idle = ('[ "${content_filter_turn_blocked:-0}" -eq 1 ] || idle_turns=$((idle_turns + 1))'
                if self.CONTENT_FILTER_OUTCOME else "idle_turns=$((idle_turns + 1))")
        init = f"{i1}idle_turns=0\n{i1}count_tool_calls() {{ {self.TOOL_CALL_COUNT}; }}\n"
        start = f"{i2}attempt_started=$(date +%s)\n"
        message = (f"The agent ended %s continuation turns in a row, each under {self.GIVE_UP_SECONDS} s "
                   f"with at most {self.GIVE_UP_TOOLS} tool calls, so it was stopped as having given up.")
        event = ('{"type":"environment.feedback","data":{"event":"agent_gave_up","message":"' + message
                 + '","consecutiveIdleTurns":%s,"continuationTurn":%s}}\\n')
        check = (
            f'{i2}attempt_tools=$(count_tool_calls "$attempt_log" | tr -dc 0-9)\n'
            f'{i2}if [ $(( $(date +%s) - attempt_started )) -lt {self.GIVE_UP_SECONDS} ] '
            f'&& [ "${{attempt_tools:-0}}" -le {self.GIVE_UP_TOOLS} ]; then\n'
            f'{i2}  {idle}\n'
            f'{i2}else\n'
            f'{i2}  idle_turns=0\n'
            f'{i2}fi\n'
            f'{i2}if [ "$idle_turns" -ge {self.GIVE_UP_TURNS} ]; then\n'
            f"{i2}  printf '{event}' \"$idle_turns\" \"$idle_turns\" \"$no_action_attempt\" >> \"$log_file\"\n"
            f'{i2}  exit 0\n'
            f'{i2}fi\n'
        )
        return init, start, check

    def _resources_tool(self) -> bool:
        if not _resources_tool_on():
            return False
        if not self.RESOURCES_TOOL:
            raise ValueError(f"{RESOURCES_TOOL_ENV}=1 is only supported with Codex and Claude Code agents")
        return True

    def _resources_install(self, homes: list[str], started: float) -> str:
        """The root command that installs the `resources` command (RESOURCES_SCRIPT, which agents
        run as RESOURCES_COMMAND), or "" when the option is off."""
        if not self._resources_tool():
            return ""
        budget = os.environ.get("RELEASE_TOKEN_BUDGET", "").strip()
        timeout = os.environ.get(AGENT_TIMEOUT_ENV, "").strip()
        config = {"cli": self.CLI, "homes": homes, "budget": int(budget) if budget else None,
                  "metric": os.environ.get("RELEASE_TOKEN_BUDGET_METRIC", "weighted").strip(),
                  "timeout": float(timeout) if timeout else None, "started": started}
        clis = self._home_clis(homes)
        if clis is not None:
            config["clis"] = clis
        script, command = shlex.quote(RESOURCES_SCRIPT), shlex.quote(RESOURCES_COMMAND)
        parts = [f"mkdir -p {shlex.quote(posixpath.dirname(RESOURCES_SCRIPT))} "
                 f"{shlex.quote(posixpath.dirname(RESOURCES_COMMAND))}",
                 f"printf %s {shlex.quote(resources_script(config, self._lockstep()))} > {script}",
                 f"chmod 755 {script}"]
        # Never replace a command of the image's own.
        parts.append(f"{{ [ ! -e {command} ] || [ -L {command} ] || "
                     f"{{ echo {shlex.quote(f'{RESOURCES_COMMAND} already exists in this image')} >&2; "
                     "exit 1; }; }")
        parts.append(f"ln -sfn {script} {command}")
        return "set -euo pipefail; " + " && ".join(parts)

    def _home_clis(self, homes: list[str]) -> list[str] | None:
        """Each home's CLI for the `resources` command when the agents run different CLIs (MixedTeamAgent);
        None when they all run this one."""
        return None

    def _resources_instruction(self, text: str, agents: int) -> str:
        """An agent's task instruction, ending with the paragraph about `resources` when the
        option is on."""
        if not self._resources_tool():
            return text
        if self._ultra() is not None:
            # One agent whose `resources` counts its helpers' tokens with its own (the whole process tree's).
            team = ultra.RESOURCES_SUFFIX
        else:
            team = RESOURCES_PROMPT_TEAM if agents > 1 else ""
        return text.rstrip("\n") + "\n\n" + RESOURCES_PROMPT.format(team=team)

    def _resources_continuation(self, indent: str) -> str:
        """The line that appends the remaining budget to each continuation, after the task's
        text; "" when the option is off."""
        if not self._resources_tool():
            return ""
        return (f'{indent}python3 {shlex.quote(RESOURCES_SCRIPT)} --continuation --home "${self.CLI}_log_dir" '
                f'>> "$feedback_file" 2>> "${{{self.CLI}_log_dir%/*}}/resources-errors.log"\n')

    def _guard_retry_parts(self, limit: bool) -> tuple[str, str]:
        """Bash for a CLI that a signal ended (HARNESS_GUARD_HELPERS' cli_ended_by_signal), spliced
        into run_<cli>_with_api_retries: (before each attempt, when an attempt ended on an error
        that is not retryable). The turn is resumed as after a usage limit: without its prompt if the
        session has it (LIMIT: usage-limit resumption is on), else by sending the prompt again in the
        same session. Two empty strings without the keeper, so the runner is unchanged."""
        if not self._guard():
            return "", ""
        cli = self.CLI
        resume = (
            f'        resume_input="$(python3 -c "$limit_resume_source" input {cli} "${cli}_log_dir" "$session_id" '
            '"$attempt_prompt_file" "$chain_started")"\n'
            '        if [ -n "$session_id" ] && [ "$resume_input" = resume ]; then\n'
            '          append_turn_resumed_event "$log_file" "$retry"\n'
            "        fi\n" if limit else "")
        return ("    attempt_started_at=$SECONDS\n",
                '      if [ "$retry" -lt "$max_api_retries" ] && '
                'cli_ended_by_signal "$code" "$attempt_started_at" "$attempt_log_file"; then\n'
                "        retry=$((retry + 1))\n"
                f"{resume}"
                "        continue\n"
                "      fi\n")

    def _retry_function(self, indent: str) -> str:
        if self._limit_resume():
            return self._limit_retry_function(indent)
        cli = self.CLI
        filter_init, filter_reset, filter_check = self._content_filter_parts()
        attempt_start, killed = self._guard_retry_parts(False)
        text = f"""run_{cli}_with_api_retries() {{
  local attempt_prompt_file="$1"
  local attempt_log_base="$2"
  local max_api_retries="${{META_DUO_{cli.upper()}_API_RETRIES:-288}}"
  case "$max_api_retries" in ''|*[!0-9]*) max_api_retries=288 ;; esac
  local retry=0
  local code=0
{filter_init}  while true; do
    local attempt_log_file="$attempt_log_base"
    if [ "$retry" -gt 0 ]; then
      attempt_log_file="${{attempt_log_base%.jsonl}}-api-retry-$retry.jsonl"
    fi
{attempt_start}    if [ -n "$session_id" ]; then
      invoke_{cli} "$attempt_prompt_file" "$attempt_log_file" {self.RESUME_ARGS}
    else
      invoke_{cli} "$attempt_prompt_file" "$attempt_log_file"
    fi
    code=$?
    cat "$attempt_log_file" >> "$log_file"
    new_session_id="$(extract_{cli}_session_id "$attempt_log_file")"
    if [ -n "$new_session_id" ]; then
      session_id="$new_session_id"
    fi
{self._after_attempt()}    if [ "$code" -eq 0 ]; then
{filter_reset}      return 0
    fi
{filter_check}    if ! is_retryable_{cli}_api_error "$attempt_log_file"; then
{killed}      return "$code"
    fi
    if [ "$retry" -ge "$max_api_retries" ]; then
      append_{cli}_api_retry_exhausted_event "$log_file" "$max_api_retries" "$code"
      return "$code"
    fi
    retry=$((retry + 1))
    delay="$({cli}_api_retry_sleep_seconds "$retry")"
    append_{cli}_api_retry_event "$log_file" "$retry" "$max_api_retries" "$code" "$delay"
    sleep "$delay"
  done
}}"""
        return textwrap.indent(text, indent)

    def _limit_retry_function(self, indent: str) -> str:
        """run_<cli>_with_api_retries with usage-limit resumption: after a usage limit it waits
        until the reset the CLI reported, otherwise it backs off as before; each wait goes
        through limit_wait. A retry sends the attempt's prompt again only if the session does
        not have it yet; otherwise it resumes the interrupted turn without it."""
        cli = self.CLI
        filter_init, filter_reset, filter_check = self._content_filter_parts()
        attempt_start, killed = self._guard_retry_parts(True)
        hold_before, hold_after, hold_request, hold_end = self._team_hold_parts()
        hold_stop_end = hold_end and "  " + hold_end
        # A runner started again after a kill (recover_runner) enters with the interrupted turn's own
        # input and start.
        resume_init, chain_init = "  local resume_input=prompt\n", '  chain_started="$(date +%s.%N)"\n'
        resume_turn = "harness_resume_turn" if self._guard() else f"resume_{cli}_turn"
        if self._guard():
            resume_init = '  local resume_input="${recover_resume_input:-prompt}"\n'
            chain_init = ('  chain_started="${recover_chain_started:-$(date +%s.%N)}"\n'
                          '  recover_resume_input=""\n'
                          '  recover_chain_started=""\n')
        text = f"""run_{cli}_with_api_retries() {{
  local attempt_prompt_file="$1"
  local attempt_log_base="$2"
  local max_api_retries="${{META_DUO_{cli.upper()}_API_RETRIES:-288}}"
  case "$max_api_retries" in ''|*[!0-9]*) max_api_retries=288 ;; esac
  local retry=0
  local code=0
{resume_init}  local chain_started
  local wait_reset
{chain_init}{filter_init}  while true; do
    local attempt_log_file="$attempt_log_base"
    if [ "$retry" -gt 0 ]; then
      attempt_log_file="${{attempt_log_base%.jsonl}}-api-retry-$retry.jsonl"
    fi
{hold_before}{attempt_start}    if [ -n "$session_id" ] && [ "$resume_input" = resume ]; then
      {resume_turn} "$attempt_log_file"
    elif [ -n "$session_id" ]; then
      invoke_{cli} "$attempt_prompt_file" "$attempt_log_file" {self.RESUME_ARGS}
    else
      invoke_{cli} "$attempt_prompt_file" "$attempt_log_file"
    fi
    code=$?
    cat "$attempt_log_file" >> "$log_file"
    new_session_id="$(extract_{cli}_session_id "$attempt_log_file")"
    if [ -n "$new_session_id" ]; then
      session_id="$new_session_id"
    fi
{self._after_attempt()}{hold_after}    if [ "$code" -eq 0 ]; then
{filter_reset}      return 0
    fi
{filter_check}    if ! is_retryable_{cli}_api_error "$attempt_log_file"; then
{killed}      return "$code"
    fi
    if [ "$retry" -ge "$max_api_retries" ]; then
      append_{cli}_api_retry_exhausted_event "$log_file" "$max_api_retries" "$code"
      return "$code"
    fi
    retry=$((retry + 1))
    wait_reset="$(python3 -c "$limit_resume_source" delay {cli} "$attempt_log_file" "${cli}_log_dir" "$session_id")"
    if [ -n "$wait_reset" ]; then
      delay="${{wait_reset%% *}}"
      append_usage_limit_wait_event "$log_file" "$retry" "$code" "$delay" "${{wait_reset##* }}"
{hold_request}    else
      delay="$({cli}_api_retry_sleep_seconds "$retry")"
      append_{cli}_api_retry_event "$log_file" "$retry" "$max_api_retries" "$code" "$delay"
    fi
    if ! limit_wait "$delay"; then
{hold_stop_end}      limit_stop_reached
      return 0
    fi
{hold_end}    resume_input="$(python3 -c "$limit_resume_source" input {cli} "${cli}_log_dir" "$session_id" "$attempt_prompt_file" "$chain_started")"
    if [ -n "$session_id" ] && [ "$resume_input" = resume ]; then
      append_turn_resumed_event "$log_file" "$retry"
    fi
  done
}}"""
        return textwrap.indent(text, indent)

    # ---- harness protection (HARNESS_GUARD_ENV) --------------------------------------------------

    def _recover_function(self, indent: str) -> str:
        """recover_runner, for a runner the keeper started again after a signal ended the one before
        ($runner_recover: which restart). It waits for the agent's CLI if that still runs (the agent's
        turn goes on undisturbed), reads where the dead runner had got to, and takes the turn as
        finished or resumes it, leaving the attempt's exit code in recover_code and the next turn in
        recover_next_turn. Returns 1 if the dead runner had not started the CLI yet: the caller then
        starts as a new runner does. "" without the keeper."""
        if not self._guard():
            return ""
        cli = self.CLI
        resume = (
            '  recover_chain_started="$since"\n'
            f'  recover_resume_input="$(python3 -c "$limit_resume_source" input {cli} "${cli}_log_dir" "$session_id" '
            '"$recover_prompt" "$since")"\n'
            '  harness_killed_note "$last_log"\n' if self._limit_resume() else "")
        event = ('{"type":"environment.feedback","data":{"event":"harness_runner_restarted","message":"The runner of '
                 "this agent was ended by a signal (probably a command of an agent); the harness started it again "
                 f'and it continues the same {self.CLI_LABEL} session.","restart":%s,"turn":%s,"turnState":"%s"}}}}\\n')
        text = f"""recover_runner() {{
  local found turn state found_session since last_log recover_prompt number
  # The agent's CLI may still be working on its turn: wait for it instead of interrupting it.
  until python3 "$harness_tool" cli-wait "${cli}_log_dir"; do
    command sleep 1
  done
  found="$(python3 "$harness_tool" recover-state {cli} "$runner_session_dir" "${cli}_log_dir" "$log_file")"
  IFS='|' read -r turn state found_session since last_log <<< "$found"
  case "$turn" in ''|*[!0-9]*) return 1 ;; esac
  session_id="$found_session"
  printf '{event}' "$runner_recover" "$turn" "$state" >> "$log_file"
  recover_next_turn=$((turn + 1))
  recover_code=0
{self._before_continuation("  ")}  if [ "$state" = complete ]; then
    return 0
  fi
  if [ "$turn" -eq 0 ]; then
    recover_prompt="$prompt_file"
  else
    recover_prompt="$runner_session_dir/{cli}-no-action-feedback-$turn.txt"
    [ -e "$recover_prompt" ] || write_no_action_feedback "$recover_prompt"
  fi
  number="$runner_recover"
  while [ -e "$runner_session_dir/{cli}-attempt-$turn-recover-$number.jsonl" ]; do
    number=$((number + 1))
  done
{resume}  run_{cli}_with_api_retries "$recover_prompt" "$runner_session_dir/{cli}-attempt-$turn-recover-$number.jsonl"
  recover_code=$?
  return 0
}}"""
        return "\n\n" + textwrap.indent(text, indent)

    def _guard_start(self, start_block: str, first_turn: str, indent: str) -> tuple[str, str]:
        """The runner's first attempt and first continuation turn with the keeper: a runner started
        again after a kill recovers instead (recover_runner). Unchanged without the keeper."""
        if not self._guard():
            return start_block, first_turn
        return (f'{indent}if [ -n "${{runner_recover:-}}" ] && recover_runner; then\n'
                f"{indent}  start_code=$recover_code\n"
                f"{indent}else\n"
                f"{start_block}"
                f"{indent}  start_code=$?\n"
                f"{indent}fi\n"
                f'{indent}(exit "$start_code")\n'), f'"${{recover_next_turn:-{first_turn}}}"'

    def _guard_unit(self, index: int, agent_id: str, count: int, session_dir: str, log: str,
                    started: float, wait_max_left: int | None = None) -> dict[str, Any]:
        """One agent in the keeper's settings: its session, its log, and its watchers with the
        arguments the runner gave them when it started them itself (WAIT_MAX_LEFT: as _limit_locals')."""
        cli, home = self.CLI, f"{session_dir}/{self.LOG_DIR_NAME}"
        watchers: dict[str, Any] = {}
        budget = os.environ.get("RELEASE_TOKEN_BUDGET", "").strip()
        if self.TOKEN_BUDGET and budget:
            metric = os.environ.get("RELEASE_TOKEN_BUDGET_METRIC", "weighted").strip()
            watchers["token"] = {
                "module": "mod-token.py", "log": f"{session_dir}/token-budget-watcher.log",
                "argv": [cli, home, budget, metric, f"{session_dir}/token-budget-exhausted.json",
                         f"{session_dir}/token-usage.json"]}
        if self._limit_resume():
            timeout = os.environ.get(AGENT_TIMEOUT_ENV, "").strip()
            wait_max = (str(wait_max_left) if wait_max_left is not None
                        else os.environ.get(LIMIT_WAIT_MAX_ENV, "").strip() or str(DEFAULT_LIMIT_WAIT_MAX))
            watchers["limit"] = {
                "module": "mod-limit.py", "log": f"{session_dir}/limit-watch.log",
                "argv": ["watch", cli, session_dir, self.HOME_ENV_NAME, home, f"{started:.3f}", timeout,
                         str(int(float(wait_max))), str(LIMIT_POLL_SECONDS),
                         f"{session_dir}/{harness_guard.LIMIT_STATE}"]}
        if _episode_over_command():
            # As the agent's user and with its identity, like the command the runner polls itself.
            watchers["episode"] = {
                "module": "mod-episode.py", "log": f"{session_dir}/episode-watch.log", "as_agent": True,
                "env": {"AGENT_ID": agent_id, "AGENT_INDEX": str(index), "AGENT_COUNT": str(count)},
                "argv": [_episode_over_command(), self.HOME_ENV_NAME, home, f"{session_dir}/episode-over.json",
                         str(EPISODE_POLL_SECONDS)]}
        if self._team_hold():
            watchers["hold"] = {
                "module": "mod-hold.py", "log": f"{session_dir}/team-hold-watch.log",
                "argv": ["watch", str(index), agent_id, cli, session_dir, self.HOME_ENV_NAME, home,
                         f"{GUARD_DIR}/state/holds", f"{EnvironmentPaths.agent_dir.as_posix()}/{team_hold.RECORD}",
                         str(TEAM_HOLD_POLL_SECONDS)]}
        return {"index": index, "id": agent_id, "session_dir": session_dir, "home": home, "log": log,
                "watchers": watchers}

    def _lockstep_watchers(self, units: list[dict[str, Any]], members: list[CliAgent]) -> None:
        """With lockstep: each agent's hold watcher runs lockstep.py (which runs the team limit hold too
        when that is on), with the team's sessions and budgets, and its working-time watcher leaves the
        agent's lockstep pauses out of both its working time and its waiting."""
        settings = _lockstep_settings()
        budget = int(os.environ["RELEASE_TOKEN_BUDGET"].strip())
        team = [{"index": unit["index"], "agent": unit["id"], "session_dir": unit["session_dir"], "budget": budget}
                for unit in units]
        for unit, member in zip(units, members):
            config = {**settings, "index": unit["index"], "agent": unit["id"], "cli": member.CLI,
                      "session_dir": unit["session_dir"], "home_env": member.HOME_ENV_NAME, "home": unit["home"],
                      "binary": member._guard_cli()["binary"], "hold": self._team_hold(),
                      "record": f"{EnvironmentPaths.agent_dir.as_posix()}/{team_hold.RECORD}",
                      "poll": TEAM_HOLD_POLL_SECONDS, "holds_dir": f"{GUARD_DIR}/state/holds",
                      "team_dir": f"{GUARD_DIR}/state/lockstep", "keeper_state": f"{GUARD_DIR}/state/keeper.json",
                      "metric": os.environ.get("RELEASE_TOKEN_BUDGET_METRIC", "weighted").strip(), "team": team}
            unit["watchers"]["hold"] = {"module": "mod-step.py", "log": f"{unit['session_dir']}/lockstep-watch.log",
                                        "argv": ["watch", json.dumps(config, sort_keys=True)]}
            limit = unit["watchers"]["limit"]
            limit.update(module="mod-step.py", argv=["limit", *limit["argv"]])

    def _guard_cli(self) -> dict[str, str]:
        """The agent's CLI in the keeper's settings: the trial's, or a mixed team agent's own (its unit's)."""
        return {"cli": self.CLI, "label": self.CLI_LABEL, "home_env": self.HOME_ENV_NAME,
                "binary": os.environ.get(self.BINARY_ENV, self.DEFAULT_BINARY)}

    def _guard_launcher(self, units: list[dict[str, Any]], arguments: dict[int, str], append: bool,
                        per_agent: bool = False) -> str:
        """GUARD_DIR/hsvr, which the keeper runs for each agent (HSV_UNIT: which; HSV_RECOVER: after a
        kill): it sends the runner's output where it went before and reads the runner, with the agent's
        arguments, into the same shell. Neither variable reaches the runner's environment. PER_AGENT
        (a mixed team): each agent's own runner, GUARD_DIR/runner-<index>."""
        runner_suffix = '-"$hsv_unit"' if per_agent else ""
        cases = ""
        for unit in units:
            log = shlex.quote(unit.get("output") or unit["log"])
            first = ">>" if append or unit.get("output") else ">"
            cases += (f"{unit['index']})\n"
                      f'  if [ -n "$runner_recover" ]; then exec >>{log} 2>&1; else exec {first}{log} 2>&1; fi\n'
                      f"  set -- {arguments[unit['index']]}\n"
                      "  ;;\n")
        return ('hsv_unit="${HSV_UNIT##*#}"\n'
                'runner_recover="${HSV_RECOVER:-}"\n'
                "unset HSV_UNIT HSV_RECOVER\n"
                f'case "$hsv_unit" in\n{cases}*)\n  exit 64\n  ;;\nesac\n'
                f". {shlex.quote(GUARD_DIR + '/runner')}{runner_suffix}\n")

    async def _guard_identity(self, environment: BaseEnvironment) -> dict[str, str]:
        """Who `docker exec` makes the task's agent (the keeper gives the runners the same user, groups
        and HOME), and the container's python3, bash and sed as that user finds them. Without an
        answer the keeper runs as the agent's user itself and finds the programs on its own."""
        command = (
            'export PATH="$HOME/.local/bin:$PATH"; '
            "printf 'hsv-id %s\\n' \"uid=$(id -u)\" \"gid=$(id -g)\" \"groups=$(id -G | tr ' ' ,)\" \"home=$HOME\" "
            "\"python=$(python3 -c 'import os, sys; print(os.path.realpath(sys.executable))')\" "
            '"bash=$(readlink -f "$BASH")" "sed=$(readlink -f "$(command -v sed)")"')
        result = await environment.exec(command=command)
        found: dict[str, str] = {}
        for line in (getattr(result, "stdout", "") or "").splitlines():
            if line.startswith("hsv-id "):
                key, _, value = line[len("hsv-id "):].partition("=")
                found[key] = value.strip()
        return found

    def _guard_install(self, conf: dict[str, Any], runner: str | list[str], launcher: str,
                       found: dict[str, str]) -> tuple[str, str]:
        """The two root commands that install the keeper: its programs (this code's, the same for
        every trial), then the trial's runner, launcher and settings. A mixed team's runners (a list:
        each agent's, runner-<index>) are written in between (_guard_runner_writes). Each link is to the
        real program, or, where that does not run under another name (a multi-call binary), a wrapper."""
        d = shlex.quote(GUARD_DIR)
        runners = {"runner": runner} if isinstance(runner, str) else {
            f"runner-{index}": text for index, text in enumerate(runner)}

        def write(name: str, text: str) -> str:
            return f"printf %s {shlex.quote(text)} > {d}/{name}"

        def link(name: str, key: str, find: str, plain: str, test: str) -> str:
            return f"link_tool {name} {shlex.quote(found.get(key, ''))} {shlex.quote(find)} {plain} {shlex.quote(test)}"

        link_tool = (
            'link_tool() { local target="$2"; [ -n "$target" ] && [ -x "$target" ] || target="$(eval "$3")"; '
            f'ln -sfn "$target" {d}/"$1"; if ! eval "$5" > /dev/null 2>&1; then rm -f {d}/"$1"; '
            f'printf \'#!/bin/sh\\nexec %s "$@"\\n\' "$4" > {d}/"$1"; chmod 755 {d}/"$1"; fi; }}')
        # With the team limit hold, its watcher too (lockstep's watcher runs it, or uses its functions).
        hold = ["mod-hold.py"] if self._team_hold() or self._lockstep() else []
        tools = " && ".join([
            f"mkdir -p {d}/state", f"chmod 755 {d}", link_tool,
            write("hsvd", HARNESS_GUARD_SOURCE), write("mod-token.py", TOKEN_BUDGET_SOURCE),
            write("mod-limit.py", LIMIT_RESUME_SOURCE), write("mod-episode.py", EPISODE_WATCH_SOURCE),
            *(write(name, TEAM_HOLD_SOURCE) for name in hold),
            link("hsvi", "python", "python3 -c 'import os, sys; print(os.path.realpath(sys.executable))'", "python3",
                 f'{GUARD_DIR}/hsvi -c "import json, mmap, runpy, signal, subprocess"'),
            link("hsvx", "bash", 'readlink -f "$BASH"', "bash", f'[ "$({GUARD_DIR}/hsvx -c "echo ok")" = ok ]'),
            link("hsvs", "sed", 'readlink -f "$(command -v sed)"', "sed",
                 f'[ "$(printf "a\\n" | {GUARD_DIR}/hsvs -u -E "s/a/b/")" = b ]'),
            f"chmod 644 {d}/hsvd {d}/mod-token.py {d}/mod-limit.py {d}/mod-episode.py"
            + "".join(f" {d}/{name}" for name in hold),
        ])
        one = isinstance(runner, str)
        trial = " && ".join([
            *(write(name, text) for name, text in runners.items() if one), write("hsvr", launcher),
            write("conf.json", json.dumps(conf, indent=1, sort_keys=True)),
            "chmod 644 " + " ".join(f"{d}/{name}" for name in [*runners, "hsvr", "conf.json"]),
            f"rm -f {d}/state/keeper.json",
            f"{d}/hsvi {d}/hsvd probe",
        ])
        return "set -euo pipefail; " + tools, "set -euo pipefail; " + trial

    @staticmethod
    def _guard_trial_commands(conf: dict[str, Any], trial: str) -> list[str]:
        """TRIAL (_guard_install's second command) as the commands to send, in order: [TRIAL] whenever it fits in
        one argument of execve, as before. conf.json grows by about 3.5 KiB per agent, and for a team of 32 the one
        command was 157 KB ("Argument list too long"); then the same
        steps go out as consecutive commands: the writes before conf.json, conf.json in pieces, the rest."""
        if len(trial.encode()) <= EXECVE_ARG_MAX:
            return [trial]
        d = shlex.quote(GUARD_DIR)
        head = "set -euo pipefail; "
        settings = json.dumps(conf, indent=1, sort_keys=True)
        conf_write = f" && printf %s {shlex.quote(settings)} > {d}/conf.json && "
        if not trial.startswith(head) or trial.count(conf_write) != 1:
            raise ValueError("the keeper's install command does not hold its conf.json write once")
        before, after = trial.split(conf_write)
        pieces = [settings[i:i + GUARD_CONF_PIECE] for i in range(0, len(settings), GUARD_CONF_PIECE)]
        commands = [before, *(f"{head}printf %s {shlex.quote(piece)} {'>' if i == 0 else '>>'} {d}/conf.json"
                              for i, piece in enumerate(pieces)), head + after]
        for command in commands:
            if len(command.encode()) > EXECVE_ARG_MAX:
                raise ValueError(f"a keeper install command of {len(command.encode())} bytes cannot fit in one "
                                 f"argument of execve ({EXECVE_ARG_MAX})")
        return commands

    def _guard_runner_writes(self, runner: list[str]) -> list[str]:
        """The root commands that write a mixed team's runners for the keeper, one each: every command
        must fit in one argument of execve (128 KiB), and one runner is about half of that."""
        return [f"set -euo pipefail; printf %s {shlex.quote(text)} > {shlex.quote(GUARD_DIR)}/runner-{index}"
                for index, text in enumerate(runner)]

    async def _guard_status(self, environment: BaseEnvironment, user: str | None) -> dict[str, Any] | None:
        """The keeper's own account of itself (harness_guard.py status), or None if the container
        does not answer."""
        try:
            result = await environment.exec(command=f"{GUARD_DIR}/hsvi {GUARD_DIR}/hsvd status", user=user,
                                            timeout_sec=120)
        except Exception:  # noqa: BLE001
            return None
        for line in reversed((getattr(result, "stdout", "") or "").splitlines()):
            if line.startswith("hsv-status "):
                try:
                    status = json.loads(line[len("hsv-status "):])
                except ValueError:
                    return None
                return status if isinstance(status, dict) else None
        return None

    async def _guard_probe(self, environment: BaseEnvironment, user: str | None) -> None:
        """While the keeper runs, ask for its status now and then: that continues a keeper that was
        stopped (SIGSTOP) and ends one whose heartbeat is old, which is then started again."""
        while True:
            await asyncio.sleep(GUARD_PROBE_SECONDS)
            try:
                await environment.exec(command=f"{GUARD_DIR}/hsvi {GUARD_DIR}/hsvd status --kick {GUARD_STALE_SECONDS}",
                                       user=user, timeout_sec=120)
            except Exception:  # noqa: BLE001
                pass

    async def _guard_supervise(self, environment: BaseEnvironment, env: dict[str, str], user: str | None) -> None:
        """Run the keeper until every agent has ended. A keeper that dies (a root agent's `kill -9 -1`
        ends everything in the container) is started again: it adopts the runners that still run and
        starts the others in recover mode. One that cannot be kept running ends the trial cleanly."""
        command = f"exec {GUARD_DIR}/hsvi {GUARD_DIR}/hsvd k"
        restarts = quick = 0
        while True:
            began = time.monotonic()
            probe = asyncio.ensure_future(self._guard_probe(environment, user))
            try:
                result = await environment.exec(
                    command=command, user=user, env={**env, **({"HSV_RECOVER": str(restarts)} if restarts else {})})
            finally:
                probe.cancel()
            if result.return_code == 0:
                return
            status = await self._guard_status(environment, user)
            if status is not None and status.get("stopped"):
                # `hsvd stop REASON`, run by hand in the container, ended the agents cleanly.
                raise HarnessProtectionError(f"The agents were stopped: {status['stopped']}.")
            if status is None or status.get("state") == "finished":
                # Every agent ended on an error, or the container is gone: the trial's own failure, as before.
                raise self._classify_exec_error(command, result)
            quick = quick + 1 if time.monotonic() - began < GUARD_QUICK_SECONDS else 0
            if quick >= GUARD_MAX_QUICK_RESTARTS:
                reason = (f"the harness supervisor ended {quick} times in a row within {GUARD_QUICK_SECONDS} s of "
                          f"starting (last exit code {result.return_code})")
                try:
                    await environment.exec(command=f"{GUARD_DIR}/hsvi {GUARD_DIR}/hsvd stop {shlex.quote(reason)}",
                                           user=user, timeout_sec=300)
                except Exception as exc:  # noqa: BLE001
                    self.logger.warning(f"Harness protection: could not stop the agents cleanly: {exc}")
                raise HarnessProtectionError(f"Harness protection failed: {reason}; the agents were stopped.")
            restarts += 1
            self.logger.warning(f"Harness protection: the supervisor ended (exit {result.return_code}, state "
                                f"{status.get('state')}); starting it again (restart {restarts}).")
            await asyncio.sleep(min(2 ** quick, 30) - 1)

    async def _raise_if_harness_stopped(self, environment: BaseEnvironment, session_dirs: list[str]) -> None:
        """An agent whose watchers could not be kept running was stopped by the keeper and its runner
        ended cleanly; report it, so that the trial's result says so. Harbor grades the trial."""
        if not self._guard():
            return
        markers = " ".join(shlex.quote(f"{session_dir}/{harness_guard.STOP_MARKER}") for session_dir in session_dirs)
        result = await environment.exec(command=f"for marker in {markers}; do [ -e \"$marker\" ] && echo \"$marker\"; done; true")
        stopped = (getattr(result, "stdout", "") or "").split()
        if stopped:
            raise HarnessProtectionError(
                f"Harness protection failed for {len(stopped)} agent(s), which were stopped "
                "(harness_protection_failed in their logs).")

    async def _run_guarded(self, environment: BaseEnvironment, *, layout: str, runner: str | list[str],
                           units: list[dict[str, Any]], running: list[int], arguments: dict[int, str],
                           env: dict[str, str], append: bool) -> None:
        """Run the agents' runners under the keeper instead of under the shell `docker exec` starts
        (RUNNER: a list for a mixed team, each agent's own)."""
        found = await self._guard_identity(environment)
        identity = None
        if found.get("uid", "").isdigit() and found.get("gid", "").isdigit():
            identity = {"uid": int(found["uid"]), "gid": int(found["gid"]), "home": found.get("home", ""),
                        "groups": [int(group) for group in found.get("groups", "").split(",") if group.isdigit()]}
        conf = {**self._guard_cli(), "layout": layout,
                "log": f"{EnvironmentPaths.agent_dir.as_posix()}/{GUARD_LOG}", "identity": identity,
                "fresh": True, "units": units, "running": running}
        launcher = self._guard_launcher(units, arguments, append, per_agent=not isinstance(runner, str))
        tools, trial = self._guard_install(conf, runner, launcher, found)
        await self.exec_as_root(environment, command=tools)
        if self._lockstep():
            # Lockstep's watchers, by a command of their own: every command must fit in one argument of execve.
            await self.exec_as_root(environment, command=(
                f"set -euo pipefail; printf %s {shlex.quote(LOCKSTEP_SOURCE)} > {shlex.quote(GUARD_DIR)}/mod-step.py"
                f" && chmod 644 {shlex.quote(GUARD_DIR)}/mod-step.py"))
        for command in [] if isinstance(runner, str) else self._guard_runner_writes(runner):
            await self.exec_as_root(environment, command=command)
        *writes, trial = self._guard_trial_commands(conf, trial)
        for command in writes:
            await self.exec_as_root(environment, command=command)
        installed = await self.exec_as_root(environment, command=trial)
        self.last_runner_script = runner if isinstance(runner, str) else "".join(
            f"# ---- {GUARD_DIR}/runner-{index} ----\n{text}" for index, text in enumerate(runner))
        # As root where that keeps the agent from signalling the keeper and the watchers at all; else
        # (the agent is root itself, or root here cannot take its identity) as the agent's user.
        privileged = "hsv-mode privileged" in (getattr(installed, "stdout", "") or "")
        await self._guard_supervise(environment, env, "root" if privileged else None)

    @with_prompt_template
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del context
        started = time.time()  # Harbor's agent timeout runs from here
        await self._run_team(instruction, environment, started)

    async def resume(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Revive (revive.py): continue the agents of a trial whose host died, with the runner of
        the run's layout (the team runner run() or the single-agent runner run_single())."""
        if self.revive_plan is None:
            await super().resume(instruction, environment, context)
            return
        instruction = self.render_instruction(instruction)
        if self.revive_plan["layout"] == "single":
            await self._run_single(instruction, environment, float(self.revive_plan["started"]), self.revive_plan)
        else:
            await self._run_team(instruction, environment, float(self.revive_plan["started"]), self.revive_plan)

    def _team_runner(self, revive: dict[str, Any] | None, started: float, prompt_lines: str,
                     wait_max_left: int | None = None) -> str:
        """The team runner (run-agent.sh) with this agent's CLI, which _run_team starts for each agent
        with its index, id, session directory, log, model, the shared prompt and its instruction.
        PROMPT_LINES write the agent's first prompt ("" when revived); WAIT_MAX_LEFT: as _limit_locals'."""
        cli = self.CLI
        agent_dir = EnvironmentPaths.agent_dir.as_posix()
        cli_binary = os.environ.get(self.BINARY_ENV, self.DEFAULT_BINARY)
        identity_env = (
            'AGENT_ID="$agent_id" AGENT_INDEX="$idx" AGENT_COUNT="$agent_count" '
        )
        argument_init = (
            '  idx="$1"\n'
            '  agent_id="$2"\n'
            '  agent_count="$3"\n'
            "  shift 3\n"
        )
        continuation_enabled = "1" if _continuation_enabled() else "0"
        runner_locals = "".join(f"  {assignment}\n"
                                for assignment in [*self._runner_locals(), *self._episode_locals(),
                                                   *self._limit_locals(started, wait_max_left),
                                                   *self._revive_locals(revive), *self._member_locals()])
        # With INSTRUCTION_FILE, $5 is the path of the agent's instruction file; reading it
        # keeps every byte, trailing newlines included (hence the sentinel x).
        instruction_line = (
            '  instruction="$(cat "$5"; printf x)"; instruction="${instruction%x}"'
            if self.INSTRUCTION_FILE else '  instruction="$5"'
        )
        give_up_init, give_up_start, give_up_check = self._give_up_parts("  ")
        start_block, first_turn = self._start_block(
            revive, "$session_dir", "$idx", "  ", f'  submissions_file="{agent_dir}/submissions.jsonl"\n')
        # With the keeper (HARNESS_GUARD_ENV): the runner leaves its exit code for it, and one started
        # again after a kill writes no prompt, waits no jitter and recovers instead of starting.
        guard = self._guard()
        guard_start = '  start_runner_guard "$session_dir"\n' if guard else ""
        jitter = "[ -n \"${runner_recover:-}\" ] || " if guard else ""
        if guard and prompt_lines:
            prompt_lines = f'  if [ -z "${{runner_recover:-}}" ]; then\n{prompt_lines}  fi\n'
        if revive is None and self._team_hold() and _team_hold_test():
            # Test only (TEAM_HOLD_TEST_ENV): the first agent fakes one usage-limit wait, once its log is new.
            start_block = start_block.replace(': > "$log_file"\n', ': > "$log_file"\n  team_hold_test\n', 1)
        start_block, first_turn = self._guard_start(start_block, first_turn, "  ")
        return f"""#!/usr/bin/env bash
set +e
{cli}_binary={shlex.quote(cli_binary)}
{self._action_feedback_helpers()}

run_agent() {{
{argument_init}  session_dir="$1"
  log_file="$2"
  model="$3"
  shared_prompt_path="$4"
{instruction_line}
  prompt_file="$session_dir/{cli}-prompt.txt"
  {cli}_log_dir="$session_dir/{self.LOG_DIR_NAME}"
{guard_start}{runner_locals}  mkdir -p "$session_dir" "${cli}_log_dir"
  configure_{cli}_command
  config_code=$?
  if [ "$config_code" -ne 0 ]; then
    exit "$config_code"
  fi
{self._start_episode_watch('"$agent_id" "$idx" "$agent_count"', "  ")}{self._start_limit_watch("  ")}  continuation_enabled={continuation_enabled}
{prompt_lines}  # Startup jitter to spread {self.CLI_LABEL} model-load API hits across concurrent workers
  {jitter}sleep "$(awk 'BEGIN{{srand(); print int(rand()*30)}}')"

{self._invoke_function(identity_env, "local code", "  ")}

{self._retry_function("  ")}{self._recover_function("  ")}

{start_block}  code=$?
  if [ "$code" -ne 0 ]; then
    exit "$code"
  fi
  if [ "$continuation_enabled" -ne 1 ]; then
    exit 0
  fi

  max_no_action_reprompts={CONTINUATIONS}
  # Retain the original submission-log probe and its shell output behavior.
  prev_submissions=$(wc -l < "$submissions_file" 2>/dev/null || echo 0)
  case "$prev_submissions" in ''|*[!0-9]*) prev_submissions=0 ;; esac
{give_up_init}  no_action_attempt={first_turn}
  while [ "$no_action_attempt" -le "$max_no_action_reprompts" ]; do
{self._before_continuation("    ")}    feedback_file="$session_dir/{cli}-no-action-feedback-$no_action_attempt.txt"
    attempt_log="$session_dir/{cli}-attempt-$no_action_attempt.jsonl"
    write_no_action_feedback "$feedback_file"
{self._resources_continuation("    ")}    append_optimization_continuation_event "$log_file" "$no_action_attempt" "$max_no_action_reprompts"
{give_up_start}    run_{cli}_with_api_retries "$feedback_file" "$attempt_log"
    code=$?
    if [ "$code" -ne 0 ]; then
      exit "$code"
    fi
{give_up_check}    no_action_attempt=$((no_action_attempt + 1))
  done

  printf '{{"type":"environment.feedback","data":{{"event":"optimization_continuations_completed","message":"Completed all configured optimization continuation passes.","maxAttempts":%s}}}}\\n' "$max_no_action_reprompts" >> "$log_file"
  exit 0
}}

run_agent "$@"
"""

    def _binary_check(self) -> str:
        """The setup's check that the mounted CLI can run."""
        cli_binary = os.environ.get(self.BINARY_ENV, self.DEFAULT_BINARY)
        return (f"[ -x {shlex.quote(cli_binary)} ] || "
                f"(echo {shlex.quote(f'Mounted {self.CLI_LABEL} binary is not executable at {cli_binary}')} >&2; exit 1)")

    def _members(self, count: int) -> list[CliAgent]:
        """The agent whose CLI, model and runner each of COUNT team members has: this one for all of
        them, or a mixed team's own (MixedTeamAgent)."""
        return [self] * count

    def _team_record(self) -> list[dict[str, str]] | None:
        """Each agent's CLI, model and effort for protocol.json in a mixed team; None otherwise."""
        return None

    async def _run_team(self, instruction: str, environment: BaseEnvironment, started: float,
                        revive: dict[str, Any] | None = None) -> None:
        """The team runner (run()). With REVIVE, the runner that continues the agents of a run
        whose host died (resume()): it rewrites nothing the run wrote, after checking that this
        code would have written the same, and appends to the agents' logs."""
        cli = self.CLI
        protocol_import = self._protocol_import()
        board = self._board()
        # claude_peers: refused here for an agent that is not Claude Code's (Codex, a mixed team); each Claude
        # Code agent's runner does the rest.
        self._peers()
        # swarmchat: refused here for an agent that cannot join the chat; each agent's runner does its part.
        chat = self._swarmchat()
        if self._ultra() is not None:
            raise ValueError("ultra mode runs one Codex agent with its own helpers (MULTIAGENT_N_AGENT=1), not a "
                             "team of processes: use run.py --protocol ultra --agents N")
        model = self._cli_model()
        paths = ProtocolPaths.default(COMM_DIR)
        spec = build_protocol(protocol_import, instruction, paths, model)
        env = self._build_env(protocol_import)
        if board:
            # The agents' credentials on the board come from it (message_board.token).
            env[BOARD_SECRET_ENV] = secrets.token_urlsafe(24)
        members = self._members(len(spec.agents))
        relay = self._relay(len(spec.agents), board)
        # The leg this call starts with: the first, or, reviving a relay, the one revive.py found (relay_leg). Its
        # agent's session is resumed (RESUME, the revival's plan), or, with none to resume, the leg starts afresh.
        leg = int(revive["relay_leg"]) if relay is not None and revive is not None else 1
        if not 1 <= leg <= (relay or 1):
            raise ValueError(f"a relay of {relay} legs has no leg {leg}")
        resume = None if relay is not None and revive is not None and not revive["agents"] else revive

        agent_dir = EnvironmentPaths.agent_dir.as_posix()

        def leg_paths(leg: int) -> tuple[list[str], list[str]]:
            """Each agent's session directory and log in a leg (a run that is no relay has only the first)."""
            session_dirs: list[str] = []
            agent_logs: list[str] = []
            for idx, agent in enumerate(spec.agents):
                runtime_label = relay_label(f"{idx}-{agent.id}", leg)
                session_dirs.append(f"{agent_dir}/sessions-{runtime_label}")
                # Persist logs directly so timeouts do not lose them.
                agent_logs.append(f"{agent_dir}/agent-{runtime_label}-output.log")
            return session_dirs, agent_logs

        session_dirs, agent_logs = leg_paths(leg)

        protocol_spec_record: dict[str, Any] = protocol_to_dict(spec)
        protocol_record: dict[str, Any] = {
            "protocol_import": protocol_import,
            "protocol_source_sha256": self._protocol_source_sha(protocol_import),
            "model": model,
            "shared_prompt_path": paths.shared_prompt,
            "spec": protocol_spec_record,
            "cli_backend": cli,
        }
        team = self._team_record()
        if team is not None:
            protocol_record["team"] = team
        if relay is not None:
            protocol_record["relay"] = {"legs": relay, "leg_token_budget": int(os.environ["RELEASE_TOKEN_BUDGET"]),
                                        "note": RELAY_NOTE}
        protocol_json = json.dumps(protocol_record, indent=2, sort_keys=True)
        shared_prompt = spec.agents[0].prompt
        agent_role_intro = (
            # A protocol whose text opens with the team itself (multiagent_protocols/board.py).
            spec.metadata["role_intro"] if "role_intro" in spec.metadata
            else "You are solving this task alone. Follow the solo work protocol exactly.\n\n"
            if len(spec.agents) == 1
            else (
                "You are one agent of many concurrent agents solving a task together. "
                "Follow this coordination protocol exactly.\n\n"
            )
        )
        if spec.metadata.get("task_only"):
            # A protocol without coordination text (multiagent_protocols/task_only.py):
            # each agent gets the task instruction alone, as a single agent does.
            first_prompt_lines = '  printf %s "$instruction" > "$prompt_file"\n'
        else:
            first_prompt_lines = (
                "  {\n"
                f"    printf %s {shlex.quote(agent_role_intro)}\n"
                '    cat "$shared_prompt_path"\n'
                "    printf '\\n\\nTask instruction:\\n%s\\n' \"$instruction\"\n"
                '  } > "$prompt_file"\n'
            )

        def leg_prompt_lines(leg: int) -> str:
            """The lines that write each agent's first prompt in a leg that starts afresh: a relay's later legs
            get the first leg's prompt with RELAY_NOTE before it."""
            if leg == 1:
                return first_prompt_lines
            body = ('    printf %s "$instruction"\n' if spec.metadata.get("task_only") else
                    f"    printf %s {shlex.quote(agent_role_intro)}\n"
                    '    cat "$shared_prompt_path"\n'
                    "    printf '\\n\\nTask instruction:\\n%s\\n' \"$instruction\"\n")
            return f'  {{\n    printf %s {shlex.quote(RELAY_NOTE)}\n{body}  }} > "$prompt_file"\n'

        # The run wrote each agent's prompt file before its first turn; a revival leaves it as it is.
        prompt_lines = "" if resume is not None else leg_prompt_lines(leg)
        # The runner: run-agent.sh for every agent, or in a mixed team each agent's own,
        # run-agent-<index>.sh; a relay's later legs run-agent-leg<n>.sh. A revival writes its runners next to the
        # run's.
        per_agent = any(member is not self for member in members)

        def leg_runners(leg: int) -> list[str]:
            return [f"{paths.comm_dir}/run-agent{f'-leg{leg}' if leg > 1 else ''}{f'-{idx}' if per_agent else ''}.sh"
                    for idx in range(len(members))]

        run_runners = leg_runners(leg)
        runner_paths = run_runners if resume is None else [
            f"{path.removesuffix('.sh')}-revive-{int(revive['revival'])}.sh" for path in run_runners]
        # Working time counts from STARTED less the waiting for usage limits; a relay's leg counts its own waiting,
        # so it starts its count later by the earlier legs' (WAITED: a revival's plan has theirs, relay_waited), and
        # may wait what they left of the most allowed.
        waited = float(revive.get("relay_waited") or 0.0) if relay is not None and revive is not None else 0.0
        leg_started, wait_left = started + waited, self._relay_wait_left(waited) if leg > 1 else None
        runners: dict[str, str] = {}
        for path, member in zip(runner_paths, members):
            if path not in runners:
                runners[path] = member._team_runner(resume, leg_started, prompt_lines, wait_left)

        def runner_parts(runners: dict[str, str]) -> list[str]:
            return [part for path, script in runners.items()
                    for part in (f"printf %s {shlex.quote(script)} > {shlex.quote(path)}",
                                 f"chmod 700 {shlex.quote(path)}")]

        runner_setup = runner_parts(runners)
        # A mixed team's runners are written each by a command of its own, after the setup: every command
        # must fit in one argument of execve (128 KiB), and one runner is about half of that.
        runner_writes = [f"set -euo pipefail; {runner_setup[i]} && {runner_setup[i + 1]}"
                         for i in range(0, len(runner_setup), 2)] if per_agent else []
        guard = self._guard()

        setup_parts = [
            f"rm -rf {shlex.quote(COMM_DIR)}",
            f"mkdir -p {shlex.quote(COMM_DIR)}",
            f"printf %s {shlex.quote(protocol_json)} > {shlex.quote(paths.comm_dir + '/protocol.json')}",
            f"printf %s {shlex.quote(shared_prompt)} > {shlex.quote(paths.shared_prompt)}",
            *([] if per_agent else runner_setup),
            *dict.fromkeys(member._binary_check() for member in members),
        ]
        for path in spec.shared_paths:
            if path.endswith("/"):
                setup_parts.append(f"mkdir -p {shlex.quote(path)}")
            else:
                setup_parts.append(
                    f"mkdir -p {shlex.quote(str(pathlib.PurePosixPath(path).parent))}"
                )
                setup_parts.append(f": > {shlex.quote(path)}")
        agent_instructions = [self._resources_instruction(instruction, len(spec.agents))
                              for idx in range(len(spec.agents))]

        def leg_setup(session_dirs: list[str]) -> tuple[list[str], list[str]]:
            """The setup of each agent's session in a leg (its directories, its instruction file), and its
            instruction files."""
            parts = [f"mkdir -p {session_dir}/{member.LOG_DIR_NAME}" for session_dir, member in zip(session_dirs, members)]
            files = [f"{session_dir}/task-instruction.txt" for session_dir in session_dirs]
            if self.INSTRUCTION_FILE:
                parts += [f"printf %s {shlex.quote(text)} > {shlex.quote(path)}"
                          for text, path in zip(agent_instructions, files)]
            return parts, files

        session_setup, instruction_files = leg_setup(session_dirs)
        setup_parts += session_setup

        def expected_prompt(idx: int, leg: int) -> str:
            """The first prompt the run wrote for an agent in a leg."""
            return (RELAY_NOTE if leg > 1 else "") + (
                agent_instructions[idx] if spec.metadata.get("task_only") else
                f"{agent_role_intro}{shared_prompt}\n\nTask instruction:\n{agent_instructions[idx]}\n")

        running = range(len(spec.agents))
        log_redirect = ">"
        if revive is not None:
            # Everything the setup above wrote, and each agent's first prompt, must be what the
            # run wrote; the setup then only adds the revived runner next to the run's own. (A relay: the first
            # leg's files, and those of the leg revived; a leg that starts afresh writes its own.)
            expected = {paths.comm_dir + "/protocol.json": protocol_json, paths.shared_prompt: shared_prompt}
            for checked in dict.fromkeys([1, leg if resume is not None else 1]):
                checked_dirs = leg_paths(checked)[0]
                for idx in range(len(spec.agents)):
                    if self.INSTRUCTION_FILE:
                        expected[f"{checked_dirs[idx]}/task-instruction.txt"] = agent_instructions[idx]
                    expected[f"{checked_dirs[idx]}/{members[idx].CLI}-prompt.txt"] = expected_prompt(idx, checked)
            if not revive.get("checks_passed"):
                # revive.py's dry run makes these checks as the host user just before the agent
                # phase (checks_passed); then the run's files may belong to the agent's user again.
                self._check_revived_files(revive, expected)
                for runner_path in dict.fromkeys(leg_runners(1)):
                    self._check_revived_continuation(
                        revive, pathlib.Path(self.logs_dir) / posixpath.relpath(runner_path, agent_dir), runner_path)
            setup_parts = [] if per_agent else runner_setup
            if resume is not None:
                running = sorted(int(agent["index"]) for agent in revive["agents"])
                log_redirect = ">>"
            else:
                setup_parts += session_setup

        for command in setup_commands(setup_parts):
            await self.exec_as_agent(environment, command=command, env=env)
        for command in runner_writes:
            await self.exec_as_agent(environment, command=command)

        async def install_resources(session_dirs: list[str], leg_started: float) -> None:
            resources_install = self._resources_install(
                [f"{session_dir}/{member.LOG_DIR_NAME}" for session_dir, member in zip(session_dirs, members)],
                leg_started)
            # A revival keeps the run's; a relay's every leg counts its own tokens and working time.
            if resources_install and (revive is None or relay is not None):
                await self.exec_as_root(environment, command=resources_install)

        await install_resources(session_dirs, leg_started)

        async def run_leg(session_dirs: list[str], agent_logs: list[str], runner_paths: list[str],
                          runners: dict[str, str], instruction_files: list[str], leg_started: float,
                          wait_left: int | None, running, log_redirect: str, append: bool) -> None:
            """Run the agents of one leg (every run has one) until they end."""
            if guard:
                # The keeper starts each runner as the shell below would: ignoring SIGINT and SIGQUIT, as the
                # asynchronous commands of a shell without job control do.
                units, arguments = [], {}
                for idx, agent in enumerate(spec.agents):
                    unit = members[idx]._guard_unit(idx, agent.id, len(spec.agents), session_dirs[idx],
                                                    agent_logs[idx], leg_started, wait_left)
                    unit.update(ignore_signals=["INT", "QUIT"])
                    if per_agent:
                        unit.update(members[idx]._guard_cli())
                    units.append(unit)
                    arguments[idx] = (
                        f"{idx} {shlex.quote(agent.id)} {len(spec.agents)} "
                        f"{shlex.quote(session_dirs[idx])} {shlex.quote(agent_logs[idx])} "
                        f"{shlex.quote(members[idx]._cli_model())} {shlex.quote(paths.shared_prompt)} "
                        + shlex.quote(instruction_files[idx] if self.INSTRUCTION_FILE else agent_instructions[idx]))
                if self._lockstep():
                    self._lockstep_watchers(units, members)
                runner = list(runners.values()) if per_agent else runners[runner_paths[0]]
                await self._run_guarded(environment, layout="team", runner=runner, units=units,
                                        running=list(running), arguments=arguments, env=env,
                                        append=append)
                await self._raise_if_harness_stopped(environment, session_dirs)
            else:
                run_parts = ['export PATH="$HOME/.local/bin:$PATH"', "set +e"]
                for idx, agent in ((idx, spec.agents[idx]) for idx in running):
                    identity_args = (
                        f"{idx} {shlex.quote(agent.id)} {len(spec.agents)} "
                    )
                    escaped_instruction = shlex.quote(
                        instruction_files[idx] if self.INSTRUCTION_FILE else agent_instructions[idx])
                    run_parts.append(
                        f"bash {shlex.quote(runner_paths[idx])} "
                        f"{identity_args}"
                        f"{shlex.quote(session_dirs[idx])} {shlex.quote(agent_logs[idx])} "
                        f"{shlex.quote(members[idx]._cli_model())} {shlex.quote(paths.shared_prompt)} "
                        f"{escaped_instruction} "
                        f"{log_redirect}{agent_logs[idx]} 2>&1 & PID_{idx}=$!"
                    )
                for idx in running:
                    run_parts.append(f"wait $PID_{idx}; EXIT_{idx}=$?")
                exit_summary = ", ".join(
                    f"{spec.agents[idx].id}=$EXIT_{idx}" for idx in running
                )
                run_parts.append(f'echo "Agent exits: {exit_summary}"')
                success_check = " || ".join(f"[ $EXIT_{idx} -eq 0 ]" for idx in running)
                run_parts.append(success_check)
                run_command = "; ".join(run_parts)
                await self.exec_as_agent(environment, command=run_command, env=env)
            await self._raise_if_time_cap_reached(environment, session_dirs)

        board_service = await self._board_start(environment, spec.metadata["message_board"], env) if board else None
        chat_service = await self._swarmchat_start(environment, spec.metadata["swarmchat"]) if chat else None
        try:
            await run_leg(session_dirs, agent_logs, runner_paths, runners, instruction_files, leg_started, wait_left,
                          running, log_redirect, append=resume is not None)
            # A relay's next legs, each once the last has ended on its own: its token budget, giving up, its
            # continuations. (revive.py's dry run renders the leg it revives, not the ones after.)
            while relay is not None and leg < relay and (revive is None or revive.get("checks_passed")):
                if await self._relay_over(environment, session_dirs):
                    break
                await self._relay_stop(environment, members, session_dirs)
                waited += await self._relay_waited(environment, session_dirs)
                wait_left = self._relay_wait_left(waited)
                if self._limit_resume() and wait_left <= 0:
                    break
                leg += 1
                leg_started = started + waited
                session_dirs, agent_logs = leg_paths(leg)
                runner_paths = leg_runners(leg)
                runners = {}
                for path, member in zip(runner_paths, members):
                    if path not in runners:
                        runners[path] = member._team_runner(None, leg_started, leg_prompt_lines(leg), wait_left)
                session_setup, instruction_files = leg_setup(session_dirs)
                for command in setup_commands([*runner_parts(runners), *session_setup]):
                    await self.exec_as_agent(environment, command=command, env=env)
                await install_resources(session_dirs, leg_started)
                await run_leg(session_dirs, agent_logs, runner_paths, runners, instruction_files, leg_started,
                              wait_left, range(len(spec.agents)), ">", append=False)
        finally:
            if board_service is not None:
                await self._board_stop(environment, board_service, env)
            if chat_service is not None:
                await self._swarmchat_stop(environment, chat_service)
            await self._frontier_swe_finish(environment)

    # ---- a relay's legs (run.py --relay; RELAY_ENV) ---------------------------------------------------------

    def _relay(self, agents: int, board: bool) -> int | None:
        """The number of legs of a relay, or None; raises for a run that cannot be one."""
        legs = relay_legs()
        if legs is None:
            return None
        if agents != 1 or board:
            raise ValueError(f"a relay is one agent's run in {legs} legs (run.py --relay with --agents 1), not a team")
        if not self.TOKEN_BUDGET or not os.environ.get("RELEASE_TOKEN_BUDGET", "").strip():
            raise ValueError("a relay's legs end on their token budget: it needs --token-budget, which Codex and "
                             "Claude Code agents have")
        return legs

    def _relay_wait_left(self, waited: float) -> int:
        """What the earlier legs, which waited WAITED seconds for usage limits, left of the most waiting allowed."""
        wait_max = os.environ.get(LIMIT_WAIT_MAX_ENV, "").strip() or str(DEFAULT_LIMIT_WAIT_MAX)
        return max(int(float(wait_max) - waited), 0)

    async def _relay_waited(self, environment: BaseEnvironment, session_dirs: list[str]) -> float:
        """How long the agents of the leg that just ended waited for usage limits, as their working-time watchers
        recorded it last (limit-wait.json, which a stopped watcher writes)."""
        if not self._limit_resume():
            return 0.0
        files = [f"{session_dir}/limit-wait.json" for session_dir in session_dirs]
        result = await self.exec_as_root(environment, command=" ".join(
            f"{{ cat {shlex.quote(path)} 2>/dev/null; echo; }};" for path in files) + " true")
        waited = 0.0
        for line in (getattr(result, "stdout", "") or "").splitlines():
            try:
                waited += float(json.loads(line)["waited_seconds"])
            except (ValueError, KeyError, TypeError):
                continue
        return waited

    async def _relay_over(self, environment: BaseEnvironment, session_dirs: list[str]) -> bool:
        """Whether the leg that just ended ends the relay (RELAY_FINAL_MARKERS)."""
        markers = " ".join(shlex.quote(f"{session_dir}/{name}")
                           for session_dir in session_dirs for name in RELAY_FINAL_MARKERS)
        result = await environment.exec(command=f"for marker in {markers}; do [ -e \"$marker\" ] && echo \"$marker\"; done; true")
        return bool((getattr(result, "stdout", "") or "").split())

    async def _relay_stop(self, environment: BaseEnvironment, members: list[CliAgent], session_dirs: list[str]) -> None:
        """Before the next leg: stop whatever the last leg's agent left running (commands it started in the
        background, which a spent token budget would have stopped), so the next leg starts with none of it."""
        entries = " ".join(shlex.quote(f"{member.HOME_ENV_NAME}={session_dir}/{member.LOG_DIR_NAME}")
                           for session_dir, member in zip(session_dirs, members))
        # (Without pipefail, which Harbor's exec sets: grep -q may end before tr has written everything.)
        await self.exec_as_root(environment, command=(
            f"set +o pipefail; for entry in {entries}; do for sig in TERM KILL; do found=0; for proc in /proc/[0-9]*; do "
            'pid="${proc#/proc/}"; [ "$pid" = "$$" ] && continue; '
            "if tr '\\0' '\\n' < \"$proc/environ\" 2>/dev/null | grep -qxF -- \"$entry\"; then "
            'kill -s "$sig" "$pid" 2>/dev/null; found=1; fi; done; [ "$found" -eq 1 ] || break; '
            '[ "$sig" = KILL ] || sleep 5; done; done; true'))

    # ---- the message board (protocols with message_board) ---------------------------------------------------

    def _board_command(self, action: str) -> str:
        return f"{BOARD_PYTHON} {BOARD_SERVICE} {action} --port {BOARD_PORT}"

    async def _board_start(self, environment: BaseEnvironment, board: dict[str, Any],
                           env: dict[str, str]) -> asyncio.Task:
        """Install message_board.py, start it as root for the team BOARD (the protocol's message_board) and
        wait until it answers. A revival's board replays the run's journal and goes on from there."""
        agent_dir = EnvironmentPaths.agent_dir.as_posix()
        ids = [posixpath.basename(member) for member in board["members"]]
        if len(ids) < 2 or any(member != f"/root/{agent_id}" for member, agent_id in zip(board["members"], ids)):
            raise ValueError(f"the message board needs a team of /root/<id> members: {board['members']}")
        await self.exec_as_root(environment, command="set -euo pipefail; " + " && ".join([
            f"mkdir -p {GUARD_DIR}", f"chmod 755 {GUARD_DIR}",
            f"printf %s {shlex.quote(BOARD_SOURCE)} > {BOARD_SERVICE}", f"chmod 644 {BOARD_SERVICE}",
            "ln -sfn \"$(python3 -c 'import os, sys; print(os.path.realpath(sys.executable))')\" " + BOARD_PYTHON,
        ]))
        serve = (f"exec {self._board_command('serve')} --journal {agent_dir}/{BOARD_JOURNAL} --agents {','.join(ids)} "
                 f"2>> {agent_dir}/{BOARD_SERVICE_LOG}")
        self._board_stopping = False
        task = asyncio.ensure_future(self._board_supervise(environment, serve, {BOARD_SECRET_ENV: env[BOARD_SECRET_ENV]}))
        ready = await environment.exec(command=f"{self._board_command('wait')} --timeout {BOARD_READY_SECONDS}",
                                       user="root", timeout_sec=BOARD_READY_SECONDS + 60)
        if ready.return_code != 0:
            self._board_stopping = True
            task.cancel()
            raise RuntimeError(f"The message board did not start ({(ready.stdout or '').strip()}); its errors are in "
                               f"{agent_dir}/{BOARD_SERVICE_LOG}.")
        return task

    async def _board_supervise(self, environment: BaseEnvironment, command: str, env: dict[str, str]) -> None:
        """Keep the board running until _board_stop: one that dies (a root agent's `kill -9 -1`) is started
        again from its journal; meanwhile the agents' board calls fail and Codex reopens their notifications."""
        quick = 0
        while not self._board_stopping:
            began = time.monotonic()
            try:
                result = await environment.exec(command=command, user="root", env=env)
                ended = f"exit {result.return_code}"
            except Exception as exc:  # noqa: BLE001 - the container may be restarting
                ended = repr(exc)
            if self._board_stopping:
                return
            quick = quick + 1 if time.monotonic() - began < GUARD_QUICK_SECONDS else 0
            self.logger.warning(f"Message board: the board ended ({ended}); starting it again.")
            await asyncio.sleep(min(2 ** quick, 30) - 1)

    async def _board_stop(self, environment: BaseEnvironment, task: asyncio.Task, env: dict[str, str]) -> None:
        """End the board once the agents have, and keep it as SQLite next to its journal."""
        self._board_stopping = True
        try:
            await environment.exec(command=self._board_command("stop"), user="root",
                                   env={BOARD_SECRET_ENV: env[BOARD_SECRET_ENV]}, timeout_sec=60)
            await asyncio.wait_for(task, timeout=60)
        except Exception as exc:  # noqa: BLE001 - the journal is already complete
            task.cancel()
            self.logger.warning(f"Message board: could not stop the board cleanly: {exc!r}")
        journal = pathlib.Path(self.logs_dir) / BOARD_JOURNAL
        if journal.is_file():
            try:
                message_board.to_sqlite(str(journal), str(pathlib.Path(self.logs_dir) / BOARD_SQLITE))
            except Exception as exc:  # noqa: BLE001 - best effort; message_board.py sqlite makes it later
                self.logger.warning(f"Message board: could not write {BOARD_SQLITE}: {exc!r}")

    # ---- the team chat (protocols with swarmchat) ---------------------------------------------------------------

    def _swarmchat_command(self, action: str) -> str:
        return (f"{SWARMCHAT_PYTHON} {SWARMCHAT_GLUE} {action} "
                f"--agents-dir {EnvironmentPaths.agent_dir.as_posix()}/{SWARMCHAT_AGENTS}")

    async def _swarmchat_start(self, environment: BaseEnvironment, chat: dict[str, Any]) -> asyncio.Task:
        """Install upstream SwarmChat (vendor/swarmchat), swarmchat_glue.py and upstream's `swarmchat` command, start
        the glue's `run` as root for the team CHAT (the protocol's swarmchat) and wait until it has made the agents'
        users. A revival's server reads its data folder and goes on from there, with the same tokens."""
        agent_dir = EnvironmentPaths.agent_dir.as_posix()
        ids = list(chat["members"])
        if len(ids) < 2:
            raise ValueError(f"the team chat needs a team: {ids}")
        bundle, packed = swarmchat_bundle(), f"{GUARD_DIR}/hsc.tgz.b64"
        command = shlex.quote(SWARMCHAT_COMMAND)
        await self.exec_as_root(environment, command="set -euo pipefail; " + " && ".join([
            "python3 -c 'import sys; sys.exit(sys.version_info < (3, 11))' || "
            "{ echo 'swarmchat needs python3 3.11 or later in the image' >&2; exit 1; }",
            f"mkdir -p {GUARD_DIR}", f"chmod 755 {GUARD_DIR}", f": > {packed}",
            # A revival's container may still have the last run's: `ready` must wait for this one.
            f"rm -f {agent_dir}/{SWARMCHAT_AGENTS}/ready {agent_dir}/{SWARMCHAT_AGENTS}/run.pid"]))
        # In pieces: one argument of a command holds at most 128 KiB.
        for offset in range(0, len(bundle), 65536):
            await self.exec_as_root(environment, command=f"printf %s {bundle[offset:offset + 65536]} >> {packed}")
        await self.exec_as_root(environment, command="set -euo pipefail; " + " && ".join([
            f"rm -rf {SWARMCHAT_LIB}", f"mkdir -p {SWARMCHAT_LIB}",
            f"base64 -d {packed} | tar -xz -C {SWARMCHAT_LIB} --no-same-owner", f"rm -f {packed}",
            f"chmod -R a+rX {SWARMCHAT_LIB}",
            f"printf %s {shlex.quote(SWARMCHAT_GLUE_SOURCE)} > {SWARMCHAT_GLUE}", f"chmod 644 {SWARMCHAT_GLUE}",
            "ln -sfn \"$(python3 -c 'import os, sys; print(os.path.realpath(sys.executable))')\" " + SWARMCHAT_PYTHON,
            f"mkdir -p {shlex.quote(posixpath.dirname(SWARMCHAT_COMMAND))}",
            # Never replace a command of the image's own.
            f"{{ [ ! -e {command} ] || [ \"$(cat {command})\" = {shlex.quote(SWARMCHAT_WRAPPER.rstrip(chr(10)))} ] || "
            f"{{ echo {shlex.quote(f'{SWARMCHAT_COMMAND} already exists in this image')} >&2; exit 1; }}; }}",
            f"printf %s {shlex.quote(SWARMCHAT_WRAPPER)} > {command}", f"chmod 755 {command}",
        ]))
        channels = json.dumps(chat["channels"], separators=(",", ":"))
        serve = (f"exec {self._swarmchat_command('run')} --data {agent_dir}/{SWARMCHAT_DATA} --port {SWARMCHAT_PORT} "
                 f"--agents {','.join(ids)} --channels {shlex.quote(channels)} -- {' '.join(SWARMCHAT_SERVE_FLAGS)} "
                 f">> {agent_dir}/{SWARMCHAT_SERVICE_LOG} 2>&1")
        self._swarmchat_stopping = False
        task = asyncio.ensure_future(self._swarmchat_supervise(environment, serve))
        ready = await environment.exec(command=f"{self._swarmchat_command('ready')} --timeout {SWARMCHAT_READY_SECONDS}",
                                       user="root", timeout_sec=SWARMCHAT_READY_SECONDS + 60)
        if ready.return_code != 0:
            self._swarmchat_stopping = True
            task.cancel()
            raise RuntimeError(f"The team chat did not start ({(ready.stdout or '').strip()}); its errors are in "
                               f"{agent_dir}/{SWARMCHAT_SERVICE_LOG}.")
        return task

    async def _swarmchat_supervise(self, environment: BaseEnvironment, command: str) -> None:
        """Keep the chat running until _swarmchat_stop: one that dies (a root agent's `kill -9 -1`) is started again
        from its data folder; meanwhile the agents' commands say it is unreachable and the hook has nothing new."""
        quick = 0
        while not self._swarmchat_stopping:
            began = time.monotonic()
            try:
                result = await environment.exec(command=command, user="root")
                ended = f"exit {result.return_code}"
            except Exception as exc:  # noqa: BLE001 - the container may be restarting
                ended = repr(exc)
            if self._swarmchat_stopping:
                return
            quick = quick + 1 if time.monotonic() - began < GUARD_QUICK_SECONDS else 0
            self.logger.warning(f"Team chat: the server ended ({ended}); starting it again.")
            await asyncio.sleep(min(2 ** quick, 30) - 1)

    async def _swarmchat_stop(self, environment: BaseEnvironment, task: asyncio.Task) -> None:
        """End the chat once the agents have (upstream's server saves its read positions on SIGTERM)."""
        self._swarmchat_stopping = True
        try:
            await environment.exec(command=self._swarmchat_command("stop"), user="root", timeout_sec=60)
            await asyncio.wait_for(task, timeout=60)
        except Exception as exc:  # noqa: BLE001 - every message is on disk already
            task.cancel()
            self.logger.warning(f"Team chat: could not stop the server cleanly: {exc!r}")

    async def _frontier_swe_finish(self, environment: BaseEnvironment) -> None:
        """FrontierSWE: let a promotion still in flight reach /app before Harbor
        collects it, and mark the run finished for the snapshot grader."""
        if not _frontier_swe():
            return
        try:
            await environment.exec(command=f"python3 {_harness_tools()} finish", timeout_sec=960)
        except Exception as exc:  # best effort; Harbor still collects /app
            self.logger.warning(f"FrontierSWE finish step failed: {exc}")

    @with_prompt_template
    async def run_single(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        del context
        started = time.time()  # Harbor's agent timeout runs from here
        await self._run_single(instruction, environment, started)

    async def _run_single(self, instruction: str, environment: BaseEnvironment, started: float,
                          revive: dict[str, Any] | None = None) -> None:
        """The single-agent runner (run_single()). With REVIVE, as _run_team's."""
        if relay_legs() is not None:
            raise ValueError("a relay runs through the team runner, with a protocol that has a solo prompt "
                             "(run.py --protocol bare_solo or antiherding_matched)")
        if _board_protocol(self._protocol_import()):
            raise ValueError(f"{self._protocol_import()} runs a team on the message board; one agent has no board "
                             "(--agents 2 or more)")
        if _claude_peers_protocol(self._protocol_import()):
            raise ValueError(f"{self._protocol_import()} runs a team on Claude Code's cross-session messaging; one "
                             "agent has no peers (--agents 2 or more)")
        if _swarmchat_protocol(self._protocol_import()):
            raise ValueError(f"{self._protocol_import()} runs a team on the harness's team chat; one agent has no "
                             "team (--agents 2 or more)")
        cli = self.CLI
        model = self._cli_model()
        env = self._build_env(DEFAULT_PROTOCOL)
        agent_dir = EnvironmentPaths.agent_dir.as_posix()
        session_dir = f"{agent_dir}/{cli}-single-session"
        log_file = f"{agent_dir}/agent-single-output.log"
        prompt_file = f"{session_dir}/{cli}-prompt.txt"
        cli_log_dir = f"{session_dir}/{self.LOG_DIR_NAME}"
        cli_binary = os.environ.get(self.BINARY_ENV, self.DEFAULT_BINARY)
        prompt = self._resources_instruction(self._ultra_prompt(instruction), 1)
        ultra_agents = self._ultra()
        # Ultra mode: the run's record of the mode (ultra.json: the protocol, the agent cap, the effort).
        ultra_record = [] if ultra_agents is None else [
            f"printf '%s\\n' {shlex.quote(json.dumps(ultra.record(ultra_agents, self._agent_effort()), sort_keys=True))} "
            f"> {shlex.quote(f'{agent_dir}/{ultra.RECORD}')}"]

        setup_cmd = " && ".join(
            [
                f"rm -rf {shlex.quote(session_dir)}",
                f"mkdir -p {shlex.quote(cli_log_dir)}",
                f"printf %s {shlex.quote(prompt)} "
                f"> {shlex.quote(prompt_file)}",
                *ultra_record,
                f"[ -x {shlex.quote(cli_binary)} ] || "
                f"(echo {shlex.quote(f'Mounted {self.CLI_LABEL} binary is not executable at {cli_binary}')} >&2; exit 1)",
            ]
        )
        if revive is None:
            await self.exec_as_agent(environment, command=f"set -euo pipefail; {setup_cmd}", env=env)
            resources_install = self._resources_install([cli_log_dir], started)
            if resources_install:
                await self.exec_as_root(environment, command=resources_install)
        else:
            # The run's session directory stays as it is; its prompt must be this code's (checked by
            # revive.py's dry run when checks_passed, as for the team runner).
            if not revive.get("checks_passed"):
                self._check_revived_files(revive, {prompt_file: prompt})
                # A run with the keeper kept its runner in the agent directory; one without sent it as
                # a command, which Harbor's trial.log records.
                record = pathlib.Path(self.logs_dir) / f"{cli}-single-runner.sh"
                if record.is_file():
                    self._check_revived_continuation(revive, record, record.name)
                else:
                    self._check_revived_continuation(revive, pathlib.Path(self.logs_dir).parent / "trial.log",
                                                     "trial.log")

        continuation_enabled = "1" if _continuation_enabled() else "0"
        runner_locals = "".join(f"{assignment}\n"
                                for assignment in [*self._runner_locals(), *self._episode_locals(),
                                                   *self._limit_locals(started), *self._revive_locals(revive)])
        give_up_init, give_up_start, give_up_check = self._give_up_parts("")
        start_block, first_turn = self._start_block(revive, shlex.quote(session_dir), "0", "")
        # With the keeper (HARNESS_GUARD_ENV), as in the team runner.
        guard = self._guard()
        guard_start = f"start_runner_guard {shlex.quote(session_dir)}\n" if guard else ""
        jitter = "[ -n \"${runner_recover:-}\" ] || " if guard else ""
        start_block, first_turn = self._guard_start(start_block, first_turn, "")
        run_cmd = f"""
export PATH="$HOME/.local/bin:$PATH"
set +e
{cli}_binary={shlex.quote(cli_binary)}
model={shlex.quote(model)}
log_file={shlex.quote(log_file)}
prompt_file={shlex.quote(prompt_file)}
{cli}_log_dir={shlex.quote(cli_log_dir)}
agent_dir={shlex.quote(agent_dir)}
agent_id=single
{runner_locals}continuation_enabled={continuation_enabled}
export AGENT_ID="$agent_id" AGENT_INDEX=0 AGENT_COUNT=1
{self._action_feedback_helpers()}
{guard_start}configure_{cli}_command
config_code=$?
if [ "$config_code" -ne 0 ]; then
  exit "$config_code"
fi
{self._start_episode_watch('"$AGENT_ID" "$AGENT_INDEX" "$AGENT_COUNT"', "")}{self._start_limit_watch("")}
{self._invoke_function("", "code", "")}

{self._retry_function("")}{self._recover_function("")}

# Startup jitter to spread {self.CLI_LABEL} model-load API hits across concurrent workers
{jitter}sleep "$(awk 'BEGIN{{srand(); print int(rand()*30)}}')"

{start_block}code=$?
if [ "$code" -ne 0 ]; then
  exit "$code"
fi
if [ "$continuation_enabled" -ne 1 ]; then
  exit 0
fi

max_no_action_reprompts={CONTINUATIONS}
{give_up_init}no_action_attempt={first_turn}
while [ "$no_action_attempt" -le "$max_no_action_reprompts" ]; do
{self._before_continuation("  ")}  feedback_file="{shlex.quote(session_dir)}/{cli}-no-action-feedback-$no_action_attempt.txt"
  attempt_log="{shlex.quote(session_dir)}/{cli}-attempt-$no_action_attempt.jsonl"
  write_no_action_feedback "$feedback_file"
{self._resources_continuation("  ")}  append_optimization_continuation_event "$log_file" "$no_action_attempt" "$max_no_action_reprompts"
{give_up_start}  run_{cli}_with_api_retries "$feedback_file" "$attempt_log"
  code=$?
  if [ "$code" -ne 0 ]; then
    exit "$code"
  fi
{give_up_check}  no_action_attempt=$((no_action_attempt + 1))
done

printf '{{"type":"environment.feedback","data":{{"event":"optimization_continuations_completed","message":"Completed all configured optimization continuation passes.","maxAttempts":%s}}}}\\n' "$max_no_action_reprompts" >> "$log_file"
exit 0
"""
        try:
            if guard:
                # The runner was the command itself, which ran with pipefail (BaseInstalledAgent._exec);
                # now it is a file the keeper's launcher reads. A copy stays in the agent directory.
                runner = "set -o pipefail\n" + run_cmd
                record = (f"{agent_dir}/{cli}-single-runner.sh" if revive is None
                          else f"{agent_dir}/{cli}-single-runner-revive-{int(revive['revival'])}.sh")
                await self.exec_as_agent(environment, command=f"printf %s {shlex.quote(runner)} > {shlex.quote(record)}")
                unit = self._guard_unit(0, "single", 1, session_dir, log_file, started)
                unit["output"] = f"{session_dir}/runner-output.log"
                await self._run_guarded(environment, layout="single", runner=runner, units=[unit], running=[0],
                                        arguments={0: ""}, env=env, append=revive is not None)
                await self._raise_if_harness_stopped(environment, [session_dir])
            else:
                await self.exec_as_agent(environment, command=run_cmd, env=env)
            await self._raise_if_time_cap_reached(environment, [session_dir])
        finally:
            await self._frontier_swe_finish(environment)

    def _populate_from_sessions(self, context: AgentContext) -> bool:
        """Totals from the CLI's session logs, which also count a turn the token
        budget stopped part way. False if there are none."""
        sessions: dict[str, dict[str, int]] = {}
        for cli, home_dir in self._session_homes():
            for home in self.logs_dir.glob(f"**/{home_dir}"):
                sessions.update(session_totals(cli, home))
        if not sessions:
            return False
        usage = summed(sessions)
        context.n_input_tokens = usage["input_tokens"]
        context.n_cache_tokens = usage["cached_input_tokens"]
        context.n_output_tokens = usage["output_tokens"]
        context.metadata = {**(context.metadata or {}),
                            "weighted_tokens": round(weighted_tokens(usage))}
        return True

    def _session_homes(self) -> list[tuple[str, str]]:
        """The CLIs whose session logs hold the agents' tokens, and their home directories' names."""
        return [(self.CLI, self.LOG_DIR_NAME)]

    def _agent_log_events(self):
        for usage_file in self.logs_dir.glob("agent-*-output.log"):
            try:
                # errors="replace": agent logs can contain truncated/invalid UTF-8
                # (e.g. an agent dumping binary-ish output); a strict decode here
                # raises UnicodeDecodeError mid trial-recovery and crashes harbor.
                text = usage_file.read_text(errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict):
                    yield usage_file, event

    def _protocol_import(self) -> str:
        return os.environ.get("MULTIAGENT_PROTOCOL", DEFAULT_PROTOCOL)

    def _protocol_source_sha(self, import_path: str) -> str | None:
        spec = importlib.util.find_spec(import_path.split(":", 1)[0])
        if not spec or not spec.origin:
            return None
        try:
            return hashlib.sha256(pathlib.Path(spec.origin).read_bytes()).hexdigest()
        except OSError:
            return None


class ClaudeCodeAgent(CliAgent):
    CLI = "claude"
    CLI_LABEL = "Claude Code"
    BINARY_ENV = CLAUDE_BINARY_ENV
    DEFAULT_BINARY = DEFAULT_CONTAINER_CLAUDE_BINARY
    LOG_DIR_NAME = "claude-home"
    RESUME_ARGS = '--resume "$session_id"'
    VERSION_COMMAND = 'export PATH="$HOME/.local/bin:$PATH"; claude --version'
    CLI_HELPERS = CLAUDE_HELPERS
    RETRY_CLASSIFIER = CLAUDE_RETRY_CLASSIFIER
    TOKEN_BUDGET = True
    HOME_ENV_NAME = "CLAUDE_CONFIG_DIR"
    TASK_CLI_HOOK = True
    INSTRUCTION_FILE = True
    GIVE_UP_TURNS = 10
    RESOURCES_TOOL = True
    LIMIT_RESUME = True
    LIMIT_RESUME_CLI_HELPERS = CLAUDE_LIMIT_RESUME_HELPERS
    CONTENT_FILTER_OUTCOME = CLAUDE_ATTEMPT_OUTCOME
    HARNESS_GUARD = True
    CLAUDE_PEERS = True
    SWARMCHAT = True
    LOGIN_ENV = ("CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY")
    # tool_use blocks in a Claude Code stream-json log.
    TOOL_CALL_COUNT = "grep -o '\"type\":\"tool_use\"' \"$1\" 2>/dev/null | wc -l"

    def __init__(self, *args: Any, claude_compact_window: int | str | None = None,
                 content_filter_max_blocks: int | str | None = None, **kwargs: Any) -> None:
        """claude_compact_window: Claude Code's auto-compact window (CLAUDE_CODE_AUTO_COMPACT_WINDOW,
        at least CLAUDE_COMPACT_WINDOW_MIN), or None or "off" for Claude Code's own. run.py passes it
        as a Harbor agent kwarg (--claude-compact-window), so the job's config records it.
        content_filter_max_blocks: content-filter blocks in a row before the agent is stopped (default
        CONTENT_FILTER_MAX_BLOCKS); run.py passes it only with --content-filter-max-blocks."""
        super().__init__(*args, **kwargs)
        if content_filter_max_blocks is not None:
            if isinstance(content_filter_max_blocks, bool) or not str(content_filter_max_blocks).isdigit() \
                    or int(content_filter_max_blocks) < 1:
                raise ValueError(f"content_filter_max_blocks must be a whole number >= 1: {content_filter_max_blocks!r}")
            self.content_filter_max_blocks = int(content_filter_max_blocks)
        window = None if claude_compact_window in (None, "off") else claude_compact_window
        if window is not None and (isinstance(window, bool) or not str(window).isdigit()
                                   or int(window) < CLAUDE_COMPACT_WINDOW_MIN):
            raise ValueError(f"claude_compact_window must be off or a whole number of at least "
                             f"{CLAUDE_COMPACT_WINDOW_MIN}: {claude_compact_window!r}")
        self.claude_compact_window = None if window is None else int(window)

    @staticmethod
    def name() -> str:
        return "release-collaboration-claude-code"

    def _cli_model(self) -> str:
        # Accept dotted ids (claude-sonnet-4.6) as well as Claude ids.
        model = super()._cli_model()
        return model.replace(".", "-") if model.startswith("claude-") else model

    def _build_env(self, protocol_import: str) -> dict[str, str]:
        env = {
            "IS_SANDBOX": "1",
            "MULTIAGENT_PROTOCOL": protocol_import,
        }
        # Prefer the subscription token; the CLI would otherwise bill an API key.
        token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = token
        elif api_key:
            env["ANTHROPIC_API_KEY"] = api_key
        else:
            raise RuntimeError(
                "Neither CLAUDE_CODE_OAUTH_TOKEN nor ANTHROPIC_API_KEY is set in the "
                "harness environment; run `claude setup-token` for a subscription token."
            )
        return env

    def _cli_helpers(self) -> str:
        home_made = '  mkdir -p "$claude_log_dir"\n'
        if self._swarmchat():
            # A swarmchat agent: its configure step makes its token on the team chat.
            if CLAUDE_HELPERS.count(home_made) != 1:
                raise ValueError("CLAUDE_HELPERS changed; the swarmchat lines no longer have their place")
            return CLAUDE_HELPERS.replace(home_made, home_made + swarmchat_configure())
        if not self._peers():
            return self.CLI_HELPERS
        # A claude_peers agent: its configure step links the team's session registry and makes the inboxes' directory.
        if CLAUDE_HELPERS.count(home_made) != 1:
            raise ValueError("CLAUDE_HELPERS changed; the claude_peers lines no longer have their place")
        return CLAUDE_HELPERS.replace(home_made, home_made + CLAUDE_PEERS_CONFIGURE)

    def _invoke_lines(self, identity_env: str, code_assignment: str) -> list[str]:
        # claude_peers: SendMessage and ListAgents move from the excluded tools to the agent's, and its name, inbox and
        # the setting that takes peers' messages join its command line.
        peers = self._peers()
        tools = shlex.quote(",".join(CLAUDE_AVAILABLE_TOOLS + (CLAUDE_PEERS_TOOLS if peers else ())))
        disallowed = shlex.quote(",".join(tool for tool in CLAUDE_EXCLUDED_TOOLS
                                          if not (peers and tool in CLAUDE_PEERS_TOOLS)))
        # With usage-limit resumption, Claude Code waits out 429s (and 529s) itself and sends the
        # same request again (CLAUDE_CODE_RETRY_WATCHDOG); CLAUDE_CODE_MAX_RETRIES keeps other
        # errors at its usual 10 retries instead of the watchdog's 300. Each event goes straight to
        # the attempt's log as the CLI writes it, so the limit watcher sees the waits (api_retry
        # events) at once, and a runner started again after a kill can read the attempt's log.
        limit = self._limit_resume()
        watchdog = "    CLAUDE_CODE_RETRY_WATCHDOG=1 CLAUDE_CODE_MAX_RETRIES=10 \\" if limit else None
        # The auto-compact window (claude_compact_window): Claude Code compacts at about it minus 33k.
        window = self.claude_compact_window
        compact = f"    CLAUDE_CODE_AUTO_COMPACT_WINDOW={window} \\" if window is not None else None
        return [line for line in [
            "invoke_claude() {",
            '  local attempt_prompt_file="$1"',
            '  local attempt_log_file="$2"',
            "  shift 2",
            f'  {identity_env}CLAUDE_CONFIG_DIR="$claude_log_dir" DISABLE_AUTOUPDATER=1 \\',
            "    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 CLAUDE_CODE_DISABLE_CLAUDE_MDS=1 \\",
            "    CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 \\",
            f"    BASH_MAX_TIMEOUT_MS={CLAUDE_BASH_MAX_TIMEOUT_MS} \\",
            compact,
            watchdog,
            f'    "${{claude_command[@]}}" -p --model "$model" {self._effort_flag()}--output-format stream-json --verbose \\',
            "    --dangerously-skip-permissions --strict-mcp-config --setting-sources user \\",
            f'    --tools {tools} --disallowedTools {disallowed} "${{task_cli_args[@]}}" \\',
            f"    {claude_peers_flags()} \\" if peers else None,
            # swarmchat: the hook that shows new messages after each tool call, and the agent's name.
            f"    {swarmchat_claude_flags()} \\" if self._swarmchat() else None,
            '    "$@" < "$attempt_prompt_file" > "$attempt_log_file" 2>&1',
            f"  {code_assignment}=$?",
            '  finish_claude_attempt "$attempt_log_file" "$code"',
            "}",
        ] if line is not None]

    def populate_context_post_run(self, context: AgentContext) -> None:
        self._write_peers_journal()
        if self._populate_from_sessions(context):
            return
        # Each `claude -p` call ends with one result event whose usage covers
        # that call only, so the per-call totals add up.
        total_input = total_cache = total_output = 0
        for _, event in self._agent_log_events():
            usage = event.get("usage") if event.get("type") == "result" else None
            if not isinstance(usage, dict):
                continue
            cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
            total_input += (int(usage.get("input_tokens", 0) or 0) + cache_read
                            + int(usage.get("cache_creation_input_tokens", 0) or 0))
            total_cache += cache_read
            total_output += int(usage.get("output_tokens", 0) or 0)
        context.n_input_tokens = total_input
        context.n_cache_tokens = total_cache
        context.n_output_tokens = total_output

    def _write_peers_journal(self) -> None:
        """claude_peers: the team's messages (claude_peers_journal.py) in CLAUDE_PEERS_JOURNAL, next to the agents'
        logs. Here, after the run, because Harbor has given the logs to the host user by then (Claude Code keeps its
        transcripts private to the agent's user). Best effort: the run's results and token counts do not depend on it.
        Nothing for any other protocol."""
        try:
            if not self._peers():
                return
            import claude_peers_journal
            claude_peers_journal.write(pathlib.Path(self.logs_dir), pathlib.Path(self.logs_dir) / CLAUDE_PEERS_JOURNAL)
        except Exception as exc:  # noqa: BLE001 - `python claude_peers_journal.py AGENT_DIR` writes it later
            self.logger.warning(f"Claude peers: could not write {CLAUDE_PEERS_JOURNAL}: {exc!r}")


class CodexAgent(CliAgent):
    CLI = "codex"
    CLI_LABEL = "Codex"
    BINARY_ENV = CODEX_BINARY_ENV
    DEFAULT_BINARY = DEFAULT_CONTAINER_CODEX_BINARY
    LOG_DIR_NAME = "codex-home"
    RESUME_ARGS = 'resume "$session_id"'
    VERSION_COMMAND = 'export PATH="$HOME/.local/bin:$PATH"; codex --version'
    CLI_HELPERS = (
        CODEX_HELPERS.replace("@codex_config_toml@", CODEX_CONFIG_TOML)
        .replace("@codex_auth@", CONTAINER_CODEX_AUTH)
    )
    RETRY_CLASSIFIER = CODEX_RETRY_CLASSIFIER
    TOKEN_BUDGET = True
    HOME_ENV_NAME = "CODEX_HOME"
    TASK_CLI_HOOK = True
    INSTRUCTION_FILE = True
    GIVE_UP_TURNS = 10
    RESOURCES_TOOL = True
    LIMIT_RESUME = True
    LIMIT_RESUME_CLI_HELPERS = CODEX_LIMIT_RESUME_HELPERS
    CONTENT_FILTER_OUTCOME = CODEX_ATTEMPT_OUTCOME
    HARNESS_GUARD = True
    MESSAGE_BOARD = True
    SWARMCHAT = True
    ULTRA = True
    # Its ChatGPT login is the mounted auth.json, which the runner links into the agent's home.
    LOGIN_ENV = ("CODEX_API_KEY",)
    # Completed commands, file edits and MCP tool calls in a `codex exec --json` log.
    TOOL_CALL_COUNT = ("grep -c -E '\"type\":\"item.completed\".*\"type\":\"(command_execution|file_change|mcp_tool_call)\"' "
                       '"$1" 2>/dev/null')

    def __init__(self, *args: Any, codex_compact_limit: int | str | None = None, **kwargs: Any) -> None:
        """codex_compact_limit: the tokens at which Codex compacts each agent's context, with its window pinned
        (codex_context_config), or None for Codex's own. run.py passes it as a Harbor agent kwarg
        (--codex-compact-limit), so the job's config records it."""
        super().__init__(*args, **kwargs)
        if codex_compact_limit is not None and (isinstance(codex_compact_limit, bool)
                                                or not str(codex_compact_limit).isdigit()
                                                or not 20000 <= int(codex_compact_limit) <= 784800):
            raise ValueError(f"codex_compact_limit must be a whole number from 20000 to 784800: "
                             f"{codex_compact_limit!r}")
        self.codex_compact_limit = None if codex_compact_limit is None else int(codex_compact_limit)

    @staticmethod
    def name() -> str:
        return "release-collaboration-codex"

    def _build_env(self, protocol_import: str) -> dict[str, str]:
        env = {
            "IS_SANDBOX": "1",
            "MULTIAGENT_PROTOCOL": protocol_import,
            "MULTIAGENT_CODEX_EFFORT": self._agent_effort(),
        }
        # Without an API key, the runner uses the mounted ChatGPT auth.json.
        if os.environ.get("CODEX_API_KEY"):
            env["CODEX_API_KEY"] = os.environ["CODEX_API_KEY"]
        return env

    def _runner_locals(self) -> list[str]:
        return [f"codex_effort_clamp_source={shlex.quote(CODEX_EFFORT_CLAMP_SOURCE)}",
                *super()._runner_locals()]

    def _cli_helpers(self) -> str:
        agents = self._ultra()
        limit = getattr(self, "codex_compact_limit", None)
        if agents is not None:
            # Ultra mode: the harness's config plus the thread cap and Codex's proactive mode message.
            return (CODEX_HELPERS.replace("@codex_config_toml@",
                                          codex_context_config(ultra.codex_config(CODEX_CONFIG_TOML, agents), limit))
                    .replace("@codex_auth@", CONTAINER_CODEX_AUTH))
        heredoc_end = "@codex_config_toml@TOML\n"
        if self._swarmchat():
            # On the team chat: the config with the agent's name, then its name, hook and token.
            if CODEX_HELPERS.count(heredoc_end) != 1:
                raise ValueError("CODEX_HELPERS changed; the swarmchat lines no longer have their place")
            return (CODEX_HELPERS.replace(heredoc_end, heredoc_end + SWARMCHAT_CODEX_LINES)
                    .replace("@codex_config_toml@", codex_context_config(swarmchat_codex_config(), limit))
                    .replace("@codex_auth@", CONTAINER_CODEX_AUTH))
        if not self._board():
            if limit is None:
                return self.CLI_HELPERS
            return (CODEX_HELPERS.replace("@codex_config_toml@", codex_context_config(CODEX_CONFIG_TOML, limit))
                    .replace("@codex_auth@", CONTAINER_CODEX_AUTH))
        # On the message board: the board's config, then the agent's path on the board and its credential.
        heredoc_end = "@codex_config_toml@TOML\n"
        if CODEX_HELPERS.count(heredoc_end) != 1:
            raise ValueError("CODEX_HELPERS changed; the board's lines no longer have their place")
        return (CODEX_HELPERS.replace(heredoc_end, heredoc_end + CODEX_BOARD_LINES)
                .replace("@codex_config_toml@", codex_context_config(board_codex_config(), limit))
                .replace("@codex_auth@", CONTAINER_CODEX_AUTH))

    def _member_locals(self) -> list[str]:
        # In a mixed team the runner's environment is the team's: the agent's own effort goes in the runner.
        effort = [] if self.effort is None else [f"MULTIAGENT_CODEX_EFFORT={shlex.quote(self.effort)}"]
        return [*super()._member_locals(), *effort]

    def _effort_flag(self) -> str:
        if self._ultra() is not None:
            # Ultra mode: a helper starts at its parent's effort; this also covers one spawned with another
            # model, which would otherwise take that model's default effort.
            return ('${codex_effort:+-c "model_reasoning_effort=$codex_effort" '
                    '-c "agents.default_subagent_reasoning_effort=$codex_effort"} ')
        return '${codex_effort:+-c "model_reasoning_effort=$codex_effort"} '

    def _invoke_lines(self, identity_env: str, code_assignment: str) -> list[str]:
        return [
            "invoke_codex() {",
            '  local attempt_prompt_file="$1"',
            '  local attempt_log_file="$2"',
            "  shift 2",
            f'  {identity_env}CODEX_HOME="$codex_log_dir" \\',
            '    "${codex_command[@]}" exec "$@" --json --skip-git-repo-check \\',
            # swarmchat: Codex runs the hook in hooks.json only once trusted; this trusts it for the run.
            f'    --dangerously-bypass-approvals-and-sandbox {"--dangerously-bypass-hook-trust " if self._swarmchat() else ""}'
            f'--model "$model" {self._effort_flag()}"${{task_cli_args[@]}}" \\',
            # Each event goes straight to the attempt's log: a runner started again after a kill reads it (the
            # session, the command that was running).
            '    - < "$attempt_prompt_file" > "$attempt_log_file" 2>&1',
            f"  {code_assignment}=$?",
            '  finish_codex_attempt "$attempt_log_file" "$code"',
            "}",
        ]

    def populate_context_post_run(self, context: AgentContext) -> None:
        if self._populate_from_sessions(context):
            return
        # turn.completed carries the session's running total, which survives
        # `codex exec resume`, so keep the last total of each session.
        totals: dict[tuple[str, str], dict[str, Any]] = {}
        threads: dict[str, str] = {}
        for usage_file, event in self._agent_log_events():
            key = str(usage_file)
            if event.get("type") == "thread.started":
                threads[key] = str(event.get("thread_id") or "")
            elif event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                totals[(key, threads.get(key, ""))] = event["usage"]
        context.n_input_tokens = sum(int(u.get("input_tokens", 0) or 0) for u in totals.values())
        context.n_cache_tokens = sum(
            int(u.get("cached_input_tokens", 0) or 0) for u in totals.values())
        context.n_output_tokens = sum(int(u.get("output_tokens", 0) or 0) for u in totals.values())


def _has_solo_prompt(import_path: str) -> bool:
    module, _, attribute = import_path.partition(":")
    protocol = getattr(importlib.import_module(module), attribute or "MultiAgentProtocol")
    return bool(getattr(protocol, "has_solo", False))


async def _run_collaboration(agent: CliAgent, instruction: str, environment: BaseEnvironment,
                             context: AgentContext) -> None:
    # A single agent gets the bare task unless the protocol has a solo prompt.
    if os.environ["MULTIAGENT_N_AGENT"] == "1" and not _has_solo_prompt(agent._protocol_import()):
        await agent.run_single(instruction, environment, context)
    else:
        await CliAgent.run(agent, instruction, environment, context)


class ClaudeCodeCollaborationAgent(ClaudeCodeAgent):
    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        await _run_collaboration(self, instruction, environment, context)


class CodexCollaborationAgent(CodexAgent):
    async def run(self, instruction: str, environment: BaseEnvironment,
                  context: AgentContext) -> None:
        await _run_collaboration(self, instruction, environment, context)


class MixedTeamAgent(CliAgent):
    """A team whose agents run different CLIs or models (run.py --team). Each agent runs through its
    own CLI's class above, with its own model, effort and login: its runner, token budget, usage-limit
    waits, give-up rule, continuations, revival and token counts are that CLI's. The protocol, the
    team's files, the `resources` command and the keeper are shared, as in any team."""

    CLI = "mixed"
    CLI_LABEL = "each agent's CLI"
    INSTRUCTION_FILE = True
    RESOURCES_TOOL = True
    LIMIT_RESUME = True
    HARNESS_GUARD = True
    # Every member can join the team chat (each renders its own part).
    SWARMCHAT = True
    MEMBER_CLASSES: dict[str, type[CliAgent]] = {"claude": ClaudeCodeAgent, "codex": CodexAgent}

    def __init__(self, *args: Any, team: list[dict[str, str]] | None = None,
                 claude_compact_window: int | str | None = None, **kwargs: Any) -> None:
        """team: each agent's {"cli", "model", "effort"}, in agent order; claude_compact_window: as
        ClaudeCodeAgent's, for the Claude Code agents. run.py passes both as Harbor agent kwargs, so
        the job's config.json records them."""
        super().__init__(*args, **kwargs)
        if not isinstance(team, list) or len(team) < 2:
            raise ValueError(f"a mixed team needs a list of at least two agents: {team!r}")
        self.members: list[CliAgent] = []
        for member in team:
            cls = self.MEMBER_CLASSES.get(member.get("cli")) if isinstance(member, dict) else None
            if cls is None or not member.get("model"):
                raise ValueError(f"each agent of a mixed team is {{cli: claude or codex, model, effort}}: {member!r}")
            extra = {"claude_compact_window": claude_compact_window} if cls is ClaudeCodeAgent else {}
            agent = cls(logs_dir=self.logs_dir, model_name=member["model"], **extra)
            agent.effort = member.get("effort") or None
            self.members.append(agent)
        for agent in self.members:
            agent.unset_env = tuple(dict.fromkeys(name for other in self.members if other.CLI != agent.CLI
                                                  for name in other.LOGIN_ENV))

    @staticmethod
    def name() -> str:
        return "release-collaboration-mixed-team"

    def _per_cli(self) -> list[CliAgent]:
        """One agent for each CLI the team runs, in agent order."""
        found: dict[str, CliAgent] = {}
        for member in self.members:
            found.setdefault(member.CLI, member)
        return list(found.values())

    async def install(self, environment: BaseEnvironment) -> None:
        # The harness tools (or packages) once, and every CLI's link.
        first, *others = self._per_cli()
        await first.install(environment)
        for member in others:
            await member._install_cli(environment)

    def get_version_command(self) -> str | None:
        return "; ".join(member.VERSION_COMMAND for member in self._per_cli())

    def parse_version(self, stdout: str) -> str:
        return " + ".join(line.strip() for line in stdout.splitlines() if line.strip())

    def _build_env(self, protocol_import: str) -> dict[str, str]:
        # Every CLI's login; each agent's runner unsets the ones of the others (unset_env).
        env: dict[str, str] = {}
        for member in self.members:
            env.update(member._build_env(protocol_import))
        return env

    def _members(self, count: int) -> list[CliAgent]:
        if count != len(self.members):
            raise ValueError(f"the protocol has {count} agents and the team {len(self.members)} "
                             "(run.py --team sets --agents)")
        return self.members

    def _team_record(self) -> list[dict[str, str]]:
        return [{"cli": member.CLI, "model": member._cli_model(), "effort": member._agent_effort()}
                for member in self.members]

    def _home_clis(self, homes: list[str]) -> list[str]:
        by_home = {member.LOG_DIR_NAME: member.CLI for member in self.members}
        return [by_home[posixpath.basename(home)] for home in homes]

    def _session_homes(self) -> list[tuple[str, str]]:
        return [(member.CLI, member.LOG_DIR_NAME) for member in self._per_cli()]

    def populate_context_post_run(self, context: AgentContext) -> None:
        # Every CLI's session logs; the per-CLI fallbacks from the agents' output logs do not apply.
        self._populate_from_sessions(context)
