from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from scored_artifact import BENCHMARKS as SCORED_ARTIFACT_TASKS  # noqa: E402
import network_allowlist  # noqa: E402
import pids_limit  # noqa: E402
import ultra  # noqa: E402

TASKS = ("frontier-swe", "ale-bench", *SCORED_ARTIFACT_TASKS, "browsecomp")
# FrontierSWE v2 tasks this harness runs.
FRONTIER_SWE_TASKS = ("git-to-zig",)
PROTOCOLS = ("antiherding", "antiherding_matched", "bare_solo", "task_only", "board", "claude_peers", "swarmchat",
             "ultra")
# Protocols whose team shares Codex's message board (agent.py, message_board.py; docs/setups.md): Codex teams only,
# on a Codex with the remote board (features.multi_agent_v2.message_board_remote).
BOARD_PROTOCOLS = ("board",)
BOARD_MIN_CODEX = (0, 162, 0)
# Protocols whose team messages one another with Claude Code's own cross-session messaging, SendMessage and ListAgents
# (agent.py, multiagent_protocols/claude_peers.py): Claude Code teams only.
CLAUDE_PEERS_PROTOCOLS = ("claude_peers",)
# The Claude Code release the messaging was built and tested on; its flags, the session registry and the logs the
# journal reads are those of this release.
CLAUDE_PEERS_CLAUDE = (2, 1, 289)
# Protocols whose team shares a chat run by upstream SwarmChat (vendor/swarmchat; agent.py, swarmchat_glue.py,
# multiagent_protocols/swarmchat.py): Claude Code, Codex and mixed teams. Its push is a PostToolUse hook in both CLIs,
# tested on Claude Code 2.1.289 (exactly, as claude_peers) and Codex 0.162.1 or later (its hooks.json and
# --dangerously-bypass-hook-trust). Codex agents should run the patched build (docs/codex-patch.md): stock Codex adds
# an estimate of all earlier reasoning to its context count after any mid-session message, so every pushed message
# could make it compact early.
SWARMCHAT_PROTOCOLS = ("swarmchat",)
SWARMCHAT_MIN_CODEX = (0, 162, 1)
# Ultra mode (--protocol ultra --agents N; ultra.py, docs/setups.md): one Codex agent that delegates to its own
# helpers, at most N agents at once, the root included, at the run's --effort. Each task's per-agent working copies
# take a NAME (`workcopy NAME`), since the helpers share one $AGENT_ID; BrowseComp has no copies to share. Each run gets
# the team's resources (--agents N times the task's per-agent CPUs and memory, or --cpus/--memory-mb, as for a team).
ULTRA_TASKS = ("frontier-swe", "ale-bench", "perf-takehome", "browsecomp")
# Harbor import paths for the team/solo dispatcher.
AGENT_CLASSES = {
    "claude": "ClaudeCodeCollaborationAgent",
    "codex": "CodexCollaborationAgent",
}
# --team: a team whose agents run different CLIs or models (agent.py's MixedTeamAgent).
TEAM_AGENT_CLASS = "MixedTeamAgent"
TEAM_CLIS = ("claude", "codex")
EFFORTS = ("low", "medium", "high", "xhigh", "max")


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not number >= 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not number > 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


# Claude Code compacts its context at about its auto-compact window minus 33k tokens (20k for the
# summary, a 13k buffer): with a 1M-context model, at about 967k. Codex (a 258,400-token window)
# compacts at about 244k. So every Claude Code agent gets the window 277000 (277000 - 33000 =
# 244000) unless --claude-compact-window says otherwise. Harbor passes it to the agent as the agent
# kwarg claude_compact_window, which the job's config.json records ("off" included).
CLAUDE_COMPACT_WINDOW = 277000
# Claude Code raises a smaller window to this.
CLAUDE_COMPACT_WINDOW_MIN = 100000


# Codex compacts at min(model_auto_compact_token_limit, 90% of model_context_window). Its window comes from
# OpenAI's model list at start. --codex-compact-limit pins both in each agent's config.toml (agent.py
# codex_context_config): the window at 500k (raised to fit a larger limit, up to gpt-6.1's 872k) and the
# compaction point at the limit. Harbor passes it as the agent kwarg codex_compact_limit (config.json records it).
CODEX_COMPACT_LIMIT_MIN = 20000
CODEX_COMPACT_LIMIT_MAX = 784800  # 90% of gpt-6.1's max_context_window (872000)


def codex_compact_limit(value: str) -> int:
    number = int(value)
    if not CODEX_COMPACT_LIMIT_MIN <= number <= CODEX_COMPACT_LIMIT_MAX:
        raise argparse.ArgumentTypeError(f"must be between {CODEX_COMPACT_LIMIT_MIN} and {CODEX_COMPACT_LIMIT_MAX}")
    return number


def compact_window(value: str) -> int | str:
    if value == "off":
        return value
    number = int(value)
    if number < CLAUDE_COMPACT_WINDOW_MIN:
        raise argparse.ArgumentTypeError(f"must be off or at least {CLAUDE_COMPACT_WINDOW_MIN} "
                                         "(Claude Code raises a smaller window to that)")
    return number


# With --limit-resume each agent's runner enforces --timeout itself, not counting time spent
# waiting for usage limits, and stops an agent that has waited --limit-wait-max in total.
# Harbor's agent timeout becomes a backstop above both, with room for the steps after the
# agents (FrontierSWE's finish step takes up to 16 minutes).
LIMIT_RESUME_GRACE_SECONDS = 1800
# Lockstep's reference agent and stall: the slowest agent still working, and every pause ends after an hour in which
# no unpaused agent's tokens grew (a long build, a long thought).
LOCKSTEP_REF = "slowest"
LOCKSTEP_STALL_MINUTES = 60.0


def team_spec(value: str) -> list[dict[str, str | None]]:
    """--team: CLI:MODEL[:EFFORT] for each agent, in agent order, separated by commas."""
    team = []
    for part in value.split(","):
        fields = part.strip().split(":")
        if (len(fields) not in (2, 3) or fields[0] not in TEAM_CLIS or not fields[1]
                or (len(fields) == 3 and fields[2] not in EFFORTS)):
            raise argparse.ArgumentTypeError(f"each agent is CLI:MODEL or CLI:MODEL:EFFORT, with CLI one of "
                                             f"{', '.join(TEAM_CLIS)} and EFFORT one of {', '.join(EFFORTS)}: {part!r}")
        team.append({"cli": fields[0], "model": fields[1], "effort": fields[2] if len(fields) == 3 else None})
    if len(team) < 2:
        raise argparse.ArgumentTypeError("a team has at least two agents")
    return team


def agent_clis(args: argparse.Namespace) -> list[str]:
    """The CLIs a run's agents use, each once: --team's, else --cli."""
    team = getattr(args, "team", None)
    return list(dict.fromkeys(member["cli"] for member in team)) if team else [args.cli]


def limit_resume(args: argparse.Namespace) -> bool:
    """Whether usage-limit resumption is on (the default)."""
    chosen = getattr(args, "limit_resume", None)
    return True if chosen is None else chosen


def harness_guard(args: argparse.Namespace) -> bool:
    """Whether harness protection is on (the default)."""
    chosen = getattr(args, "harness_guard", None)
    return True if chosen is None else chosen


def team_limit_hold(args: argparse.Namespace) -> bool:
    """Whether a team pauses while one of its agents waits for a usage limit: by default for --team."""
    chosen = getattr(args, "team_limit_hold", None)
    return getattr(args, "team", None) is not None if chosen is None else chosen


def lockstep(args: argparse.Namespace) -> bool:
    """Whether the team's agents spend their budgets in lockstep (--lockstep-gap)."""
    return getattr(args, "lockstep_gap", None) is not None


def harbor_agent_timeout(args: argparse.Namespace) -> float:
    """Harbor's agent timeout: --timeout, plus the waiting allowed with --limit-resume. With --lockstep-gap,
    once more: an agent's lockstep pauses are neither working time nor waiting, and last while a teammate
    behind it works or waits, up to a teammate's whole working time and waiting."""
    if not limit_resume(args):
        return args.timeout
    timeout = args.timeout + round(args.limit_wait_max * 3600) + LIMIT_RESUME_GRACE_SECONDS
    if lockstep(args):
        timeout += args.timeout + round(args.limit_wait_max * 3600)
    return timeout


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run single agents and teams of Claude Code and Codex agents on "
                                                 "FrontierSWE v2's git-to-zig (frontier-swe), ALE-Bench's AtCoder "
                                                 "Heuristic Contest problems (ale-bench), Anthropic's performance "
                                                 "take-home (perf-takehome) or BrowseComp questions (browsecomp).")
    result.add_argument("task", choices=TASKS)
    result.add_argument("--cli", choices=tuple(AGENT_CLASSES),
                        help="agent CLI: claude (Claude Code) or codex; required unless --team is given")
    result.add_argument("--model", help="required unless --team is given")
    result.add_argument("--team", type=team_spec, metavar="CLI:MODEL[:EFFORT],...",
                        help="a team whose agents run different CLIs or models, one CLI:MODEL[:EFFORT] per agent "
                             "in agent order (CLI claude or codex; EFFORT default --effort), in place of --cli and "
                             "--model; sets --agents. Each agent keeps its CLI's own login, budget, usage-limit "
                             "waits and revival; the agents reach every CLI's model API")
    result.add_argument("--team-limit-hold", action=argparse.BooleanOptionalAction,
                        help="teams (default: on with --team, off otherwise): while any agent waits for a usage "
                             "limit, the others pause too (their turn stopped and resumed afterwards), so the team "
                             "never works short-handed; the pause counts as waiting. Needs --limit-resume and "
                             "--harness-guard (team_hold.py)")
    result.add_argument("--lockstep-gap", type=positive, metavar="TOKENS",
                        help="teams (default: off): pause an agent that is more than TOKENS weighted tokens ahead of "
                             "the slowest agent still working until it is no longer ahead, so the agents spend their "
                             "budgets together; its turn is stopped between model responses and resumed "
                             "afterwards, and the pause counts as neither working time nor waiting. Needs "
                             "--token-budget, --limit-resume and --harness-guard (lockstep.py)")
    result.add_argument("--agents", type=positive, default=1)
    result.add_argument("--protocol", choices=PROTOCOLS)
    result.add_argument("--trials", type=positive, default=1)
    result.add_argument("--concurrent", type=positive, default=1)
    result.add_argument("--timeout", type=positive, default=3600, help="agent seconds per trial")
    result.add_argument("--continuations", type=nonnegative, default=1000,
                        help="maximum additional sessions (default: 1000)")
    result.add_argument("--token-budget", type=positive,
                        help="stop each agent once its weighted tokens reach this (--protocol ultra: the agent and "
                             "all its helpers); --timeout stays a safety cap")
    result.add_argument("--relay", type=positive, metavar="LEGS",
                        help="the restarted single agent: one agent's run as LEGS fresh sessions in a row in the same "
                             "container, each with --token-budget; each after the first gets the same prompt with "
                             "a note that it builds on a previous agent's work (agent.py RELAY_NOTE). Needs "
                             "--agents 1, --token-budget and --protocol bare_solo or antiherding_matched; "
                             "--timeout and --limit-wait-max cover the whole run")
    result.add_argument("--resources-tool", action="store_true",
                        help="give agents a `resources` command (their tokens against --token-budget, the team's, "
                             "time left of --timeout), point their prompt at it, and state the remaining budget in "
                             "every continuation (resources_tool.py)")
    result.add_argument("--limit-resume", action=argparse.BooleanOptionalAction,
                        help="default: on. An agent that hits a usage limit waits for the reset and continues its "
                             "turn without its prompt being sent again; waiting does not count against --timeout "
                             "(limit_resume.py)")
    result.add_argument("--limit-wait-max", type=positive_float, default=48.0, metavar="HOURS",
                        help="with --limit-resume: the most an agent may wait in total for usage limits and API "
                             "retries before it is stopped (default: 48)")
    result.add_argument("--harness-guard", action=argparse.BooleanOptionalAction,
                        help="default: on. Keep an agent's own commands (pkill, killall, kill -9 -1) from stopping "
                             "the harness's watchers and runner; a supervisor in the container starts again whatever "
                             "dies, and a trial whose protection cannot be kept ends cleanly and is graded "
                             "(harness_guard.py)")
    result.add_argument("--codex-compact-limit", type=codex_compact_limit, metavar="TOKENS",
                        help="Codex: compact each agent's context at this many tokens (model_auto_compact_token_"
                             "limit), with its context window pinned at 500k (more for a limit over 450k). "
                             "Default: Codex's own, from OpenAI's model list at start")
    result.add_argument("--content-filter-max-blocks", type=int, default=None, metavar="N",
                        help="Claude Code: stop an agent after N content-filter blocks in a row (default "
                             "agent.CONTENT_FILTER_MAX_BLOCKS = 5)")
    result.add_argument("--claude-compact-window", type=compact_window, metavar="TOKENS|off",
                        help=f"Claude Code: the window its auto-compaction counts from "
                             f"(CLAUDE_CODE_AUTO_COMPACT_WINDOW); it compacts at about this minus 33k tokens. "
                             f"Default {CLAUDE_COMPACT_WINDOW}: compacts at about 244k, as Codex does. off: Claude "
                             "Code's own (about 967k with a 1M context)")
    result.add_argument("--effort", choices=EFFORTS, default="max",
                        help="reasoning effort (default: max). --protocol ultra --agents N: the root and every "
                             "helper at this effort, at most N agents")
    result.add_argument("--cpus", type=positive)
    result.add_argument("--memory-mb", type=positive)
    result.add_argument("--pids-limit", type=pids_limit.parse, default=pids_limit.DEFAULT, metavar="N|off",
                        help=f"processes each container of a trial may run: the agents' container, its sidecars "
                             f"(scorer, judge, relay), a separate verifier and snapshot gradings (default: "
                             f"{pids_limit.DEFAULT}; a task's own lower limit stays; off: no limit). Recorded "
                             "in the job's config.json (pids_limit.py)")
    result.add_argument("--name")
    result.add_argument("--claude-binary", default=os.environ.get("CLAUDE_BINARY"))
    result.add_argument("--codex-binary", default=os.environ.get("CODEX_BINARY"))
    result.add_argument("--codex-auth", type=Path,
                        default=Path(os.environ.get("CODEX_AUTH_JSON", Path.home() / ".codex/auth.json")),
                        help="Codex ChatGPT login shared read-write with every agent "
                             "(default: ~/.codex/auth.json)")
    result.add_argument("--problem", help="ale-bench: an AtCoder Heuristic Contest problem (ahc014), lite, all, or "
                                          "ids with commas; browsecomp: question ids (1-1266 or browsecomp-0042) "
                                          "with commas, or all (1,266)")
    result.add_argument("--examples", type=positive, metavar="N",
                        help="browsecomp: upstream's --examples N, the seed-0 sample of N questions "
                             "(random.Random(0).sample, as simple-evals draws it)")
    result.add_argument("--task", dest="frontier_swe_task", metavar="NAME", choices=FRONTIER_SWE_TASKS,
                        help="frontier-swe: the FrontierSWE v2 task, tasks/NAME at the pinned commit (git-to-zig)")
    result.add_argument("--frontier-swe", type=Path, default=ROOT / ".cache/frontier-swe-v2",
                        help="frontier-swe: frontier-swe-v2 clone, made on first use and kept at "
                             "the pinned commit")
    result.add_argument("--snapshot-interval", type=nonnegative_float, metavar="MINUTES",
                        help="frontier-swe: besides after every promotion, grade /app this often when it changed "
                             "(default: 30; 0: only after promotions)")
    result.add_argument("--no-snapshots", action="store_true",
                        help="frontier-swe: grade only the final /app")
    result.add_argument("--expose-scores", action="store_true",
                        help="frontier-swe: show agents the verifier's snapshot rewards")
    result.add_argument("--fastgrade", action="store_true",
                        help="frontier-swe: grade snapshots with benchmarks/frontier-swe/fastgrade (the task's own "
                             "verifier in a plain container, build on tmpfs, warm zig cache: same scores, ~0 disk "
                             "writes). Off by default; runs started without it keep Harbor's grader")
    result.add_argument("--allow-host", action="append", default=[], metavar="HOST",
                        help="frontier-swe: also let agents reach HOST while they run (repeatable)")
    ale = result.add_argument_group("ale-bench", "AtCoder Heuristic Contest problems with ALE-Bench's own judge "
                                                 "(see ale_bench_task.py)")
    ale.add_argument("--ale-cases", choices=("full", "lite"),
                     help="ale-bench: the full version's cases (default) or the lite version's (5 public cases, the "
                          "first 10%% of a long contest's private ones; default with --problem lite)")
    ale.add_argument("--ale-language", choices=("cpp20", "cpp17", "cpp23", "python", "rust"),
                     help="ale-bench: the language of the programs (default: cpp20, the leaderboard's)")
    ale.add_argument("--ale-variant", choices=("official", "outputs"),
                     help="ale-bench: official (programs, private cases, AtCoder performance; default) or the "
                          "non-official outputs variant (output files for 50 fixed inputs; batch problems only)")
    ale.add_argument("--ale-judge-cpus", metavar="CPUS",
                     help="ale-bench: the CPUs the judge pins its cases to, one case per CPU at a time across "
                          "every ale-bench job on the host (default: the last two physical cores)")
    ale.add_argument("--ale-no-private-lane", action="store_true",
                     help="ale-bench: do not grade the run's final candidate on the private cases during the run "
                          "(the curves then have only the final grade)")
    result.add_argument("--dry-run", action="store_true", help="print the command without starting trials")
    return result


def linux_binary(cli: str, explicit: str | None, dry_run: bool) -> Path:
    """Resolve the standalone Linux executable mounted into every task container."""
    label = {"claude": "Claude Code", "codex": "Codex"}[cli]
    cached = ROOT / ".cache" / cli / ("bin/codex" if cli == "codex" else cli)
    binary = explicit or (str(cached) if cached.is_file() else shutil.which(cli))
    if not binary and dry_run:
        binary = str(Path.home() / f".local/bin/{cli}")
    if not binary:
        raise ValueError(f"install the standalone Linux {label} CLI (docs/harness.md) or pass --{cli}-binary")
    binary = Path(binary).expanduser().resolve()
    if dry_run:
        return binary
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise ValueError(f"{label} binary must be an executable file: {binary}")
    with binary.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError(f"--{cli}-binary must point to the standalone Linux {cli} "
                             f"executable, not a wrapper script: {binary}")
    return binary


def codex_version(package: Path, protocol_label: str, minimum: tuple[int, int, int], why: str,
                  dry_run: bool) -> None:
    """Check the Codex release package's version (its codex-package.json) against MINIMUM; a dry run without the
    package skips the check."""
    manifest = package / "codex-package.json"
    if not manifest.is_file():
        if dry_run:
            return
        raise ValueError(f"{protocol_label} needs --codex-binary from a codex-package release (its "
                         "codex-package.json gives the version)")
    version = str(json.loads(manifest.read_text()).get("version") or "")
    found = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if not found or tuple(int(part) for part in found.groups()) < minimum:
        raise ValueError(f"{protocol_label} needs Codex {'.'.join(map(str, minimum))} or later ({why}); "
                         f"--codex-binary is {version or 'of an unknown version'}: {package}")


def claude_peers_claude(binary: Path, dry_run: bool, protocol: str = "claude_peers") -> None:
    """claude_peers and swarmchat need exactly Claude Code 2.1.289, the release they were tested on, read from the
    binary's --version, which it prints without touching any home (run in an empty one); a dry run skips the check."""
    if dry_run:
        return
    with tempfile.TemporaryDirectory() as home:
        done = subprocess.run([str(binary), "--version"], capture_output=True, text=True, timeout=60,
                              env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home,
                                   "CLAUDE_CONFIG_DIR": home, "DISABLE_AUTOUPDATER": "1"})
    found = re.match(r"(\d+)\.(\d+)\.(\d+)", done.stdout.strip())
    if not found or tuple(int(part) for part in found.groups()) != CLAUDE_PEERS_CLAUDE:
        what = "its cross-session messaging" if protocol == "claude_peers" else "its PostToolUse hook"
        raise ValueError(f"--protocol {protocol} needs exactly Claude Code {'.'.join(map(str, CLAUDE_PEERS_CLAUDE))} "
                         f"({what} as tested); --claude-binary is "
                         f"{done.stdout.strip() or 'of an unknown version'}: {binary}")


def claude_auth(env: dict[str, str], dry_run: bool) -> None:
    token_file = Path.home() / ".claude/oauth-token"
    if not env.get("CLAUDE_CODE_OAUTH_TOKEN") and token_file.is_file():
        env["CLAUDE_CODE_OAUTH_TOKEN"] = token_file.read_text().strip()
    if env.get("CLAUDE_CODE_OAUTH_TOKEN"):
        # Claude Code prefers an API key when both are set; keep runs on the subscription.
        env.pop("ANTHROPIC_API_KEY", None)
    elif not env.get("ANTHROPIC_API_KEY") and not dry_run:
        raise ValueError("run `claude setup-token` and set CLAUDE_CODE_OAUTH_TOKEN "
                         "(or save it in ~/.claude/oauth-token), or set ANTHROPIC_API_KEY")


def codex_auth(env: dict[str, str], auth_json: Path, dry_run: bool) -> list[dict]:
    auth_json = auth_json.expanduser()
    if auth_json.is_file():
        env.pop("CODEX_API_KEY", None)
        # Shared read-write rather than copied: Codex rewrites auth.json in place when
        # it refreshes the ChatGPT token and rereads it after a 401, so every agent,
        # concurrent trial, and the host stay on one refresh-token chain. Private
        # copies would each spend the same single-use refresh token. Harbor makes a
        # mount writable by leaving out read_only.
        return [{"type": "bind", "source": str(auth_json.resolve()),
                 "target": "/mnt/release-codex-auth/auth.json",
                 "bind": {"create_host_path": False}}]
    if env.get("CODEX_API_KEY") or dry_run:
        return []
    raise ValueError(f"no Codex login at {auth_json}: run `codex login`, pass --codex-auth, "
                     "or set CODEX_API_KEY")


def prepare(args: argparse.Namespace) -> tuple[list[str], dict[str, str], Path]:
    team = getattr(args, "team", None)
    if team is not None:
        if args.model is not None or args.cli is not None:
            raise ValueError("--team gives each agent's CLI and model; leave out --cli and --model")
        if args.agents not in (1, len(team)):
            raise ValueError(f"--team has {len(team)} agents, --agents {args.agents}")
        args.agents = len(team)
    elif args.cli is None or args.model is None:
        raise ValueError("--cli and --model are required (or --team, for a team whose agents run different CLIs or "
                         "models)")
    clis = agent_clis(args)
    # What the task adapters get: the agents' CLI (a team's CLIs joined by "+") and whether a Codex
    # agent signs in with the mounted ChatGPT login.
    task_cli = "+".join(clis)
    codex_login = "codex" in clis and args.codex_auth.expanduser().is_file()
    default_name = args.task
    if args.task == "ale-bench":
        import ale_bench_task
        if args.problem is None:
            raise ValueError("ale-bench needs --problem: an AtCoder Heuristic Contest problem such as ahc014, lite "
                             "(the 10-problem lite set), all (the 40 problems), or ids separated by commas")
        ale_problems, ale_lite_set = ale_bench_task.check_problems(args.problem)
        ale_lite = (args.ale_cases or ("lite" if ale_lite_set else "full")) == "lite"
        default_name = ale_bench_task.job_dir_name(ale_problems, ale_lite, args.problem)
    elif args.task == "browsecomp":
        import browsecomp_task
        bc_questions = browsecomp_task.select(args.problem, getattr(args, "examples", None))
        default_name = browsecomp_task.job_dir_name(args.problem, getattr(args, "examples", None), bc_questions)
    elif args.problem is not None:
        raise ValueError("--problem requires the ale-bench or browsecomp task")
    if args.task != "browsecomp" and getattr(args, "examples", None) is not None:
        raise ValueError("--examples requires the browsecomp task")
    check_fastgrade(args)
    if args.task == "frontier-swe":
        import frontier_swe
        if args.frontier_swe_task is None:
            raise ValueError("frontier-swe needs --task NAME, a FrontierSWE v2 task: "
                             + ", ".join(FRONTIER_SWE_TASKS))
        fswe_name = frontier_swe.check_name(args.frontier_swe_task)
        default_name = frontier_swe.task_dir_name(fswe_name)
    elif (args.frontier_swe_task is not None or args.snapshot_interval is not None or args.no_snapshots
          or args.expose_scores or args.allow_host):
        raise ValueError("--task, --snapshot-interval, --no-snapshots, --expose-scores, and --allow-host "
                         "require the frontier-swe task")
    if args.task != "ale-bench" and (args.ale_cases is not None or args.ale_language is not None
                                     or args.ale_variant is not None or args.ale_judge_cpus is not None
                                     or args.ale_no_private_lane):
        raise ValueError("--ale-* options require the ale-bench task")
    if args.task == "browsecomp" and clis != ["codex"] and not args.dry_run:
        raise ValueError("browsecomp searches the web with Codex's built-in web search, which runs on OpenAI's "
                         "side: use --cli codex. Claude Code's WebSearch and WebFetch would need the container to "
                         "reach the web; --cli claude is supported with --dry-run only")
    if getattr(args, "content_filter_max_blocks", None) is not None and (clis != ["claude"] or args.team
                                                                         or args.content_filter_max_blocks < 1):
        raise ValueError("--content-filter-max-blocks needs --cli claude (not a mixed --team) and N >= 1")
    if args.claude_compact_window is not None and "claude" not in clis:
        raise ValueError("--claude-compact-window is only supported with --cli claude")
    if args.codex_compact_limit is not None and (clis != ["codex"] or args.team):
        raise ValueError("--codex-compact-limit is only supported with --cli codex (not a mixed --team)")
    if team_limit_hold(args) and (args.agents < 2 or not limit_resume(args) or not harness_guard(args)):
        raise ValueError("--team-limit-hold needs a team (--agents 2 or more), --limit-resume and --harness-guard")
    if lockstep(args) and (args.agents < 2 or args.token_budget is None or not limit_resume(args)
                           or not harness_guard(args)):
        raise ValueError("--lockstep-gap needs a team (--agents 2 or more), --token-budget, --limit-resume and "
                         "--harness-guard")
    # A mixed swarmchat team always spends in lockstep, else its fast agents finish hours early and the team becomes
    # a relay.
    if args.protocol in SWARMCHAT_PROTOCOLS and args.team and not lockstep(args):
        raise ValueError("--protocol swarmchat with a mixed --team needs --lockstep-gap (with --token-budget, "
                         "--limit-resume and --harness-guard), so a fast agent waits instead of finishing hours early")
    name = args.name or f"{default_name}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("--name must contain only letters, digits, underscores, and hyphens")
    protocol = args.protocol or "antiherding"
    if protocol == "task_only" and args.task != "browsecomp":
        raise ValueError("task_only gives every agent the task instruction alone; use bare_solo for a single agent "
                         "with the task text (task_only is BrowseComp's single agent)")
    if protocol == "task_only" and args.agents != 1:
        raise ValueError("browsecomp's task_only is one agent with the question alone (--agents 1); a team needs "
                         "a protocol (antiherding_matched, board, ...)")
    if protocol == "bare_solo" and args.agents != 1:
        raise ValueError("bare_solo is one agent with the task instruction alone (--agents 1)")
    if protocol in BOARD_PROTOCOLS:
        if team is not None or clis != ["codex"]:
            raise ValueError(f"--protocol {protocol} runs the team on Codex's message board, which only Codex has: "
                             "use --cli codex (not --team or --cli claude)")
        if args.agents < 2:
            raise ValueError(f"--protocol {protocol} needs a team: --agents 2 or more")
    if protocol in CLAUDE_PEERS_PROTOCOLS:
        if team is not None or clis != ["claude"]:
            raise ValueError(f"--protocol {protocol} runs the team on Claude Code's cross-session messaging (SendMessage, "
                             "ListAgents), which only Claude Code has: use --cli claude (not --team or --cli codex)")
        if args.agents < 2:
            raise ValueError(f"--protocol {protocol} needs a team: --agents 2 or more")
    if protocol in SWARMCHAT_PROTOCOLS and args.agents < 2:
        raise ValueError(f"--protocol {protocol} needs a team: --agents 2 or more")
    ultra_mode = protocol == ultra.PROTOCOL
    if ultra_mode:
        if team is not None or clis != ["codex"]:
            raise ValueError("--protocol ultra is one Codex agent that runs its own sub-agents: use --cli codex (not "
                             "--team or --cli claude)")
        if args.agents < 2:
            raise ValueError("--protocol ultra needs room for a helper: --agents N is the most agents at once, the "
                             "root included (2 or more)")
        if args.task not in ULTRA_TASKS:
            raise ValueError(f"--protocol ultra needs a task whose working copies take a name (its helpers share one "
                             f"$AGENT_ID): {', '.join(ULTRA_TASKS)}")
        if lockstep(args) or getattr(args, "team_limit_hold", None):
            raise ValueError("--lockstep-gap and --team-limit-hold pace a team of agents; --protocol ultra is one "
                             "agent, its helpers under its own budget and usage-limit waits")
    relay = getattr(args, "relay", None)
    if relay is not None:
        # The restarted single agent (agent.py RELAY_ENV): one agent's run in legs, each a fresh session that ends on
        # its own token budget, through the team runner as the solos of bare_solo and antiherding_matched run.
        if relay < 2:
            raise ValueError("--relay LEGS needs 2 or more legs (one leg is a plain solo run)")
        if team is not None or args.agents != 1:
            raise ValueError("--relay is one agent's run in legs: --agents 1 (not --team)")
        if protocol not in ("bare_solo", "antiherding_matched"):
            raise ValueError("--relay runs a solo with its own prompt: --protocol bare_solo or antiherding_matched")
        if args.token_budget is None:
            raise ValueError("--relay needs --token-budget: each leg ends on it")
    fswe_task = None
    sa_task = None
    ale_job = None
    bc_job = None
    if args.task == "ale-bench":
        if (ROOT / "jobs" / name).exists():
            raise ValueError(f"job already exists: {ROOT / 'jobs' / name}; choose a new --name")
        # One task per problem, run as Harbor's local dataset: ALE-Bench's judge next to the
        # agents, resources in each task's [environment] (k agents get k times an agent's;
        # --cpus/--memory-mb set the team's total), agents reach only their model API.
        # The dataset, images and tools are prepared on first use (not on --dry-run).
        ale_job = ale_bench_task.build_job(
            ale_bench_task.TASKS / name, ale_problems, agents=args.agents, cli=task_cli,
            codex_login=codex_login,
            cpus=args.cpus, memory_mb=args.memory_mb, lite=ale_lite,
            language=args.ale_language or ale_bench_task.DEFAULT_LANGUAGE, variant=args.ale_variant or "official",
            judge_cpus=ale_bench_task.parse_cpus(args.ale_judge_cpus) if args.ale_judge_cpus else None,
            private_lane=not args.ale_no_private_lane, dry_run=args.dry_run, ultra=ultra_mode)
        task = ale_job.path
        settings = tomllib.loads(ale_job.task_toml)
    elif args.task in SCORED_ARTIFACT_TASKS:
        import scored_artifact
        if (ROOT / "jobs" / name).exists():
            raise ValueError(f"job already exists: {ROOT / 'jobs' / name}; choose a new --name")
        # Resources go into the task's [environment]: k agents get k times the
        # benchmark's CPUs and memory, and --cpus/--memory-mb set the team's total.
        # Written at the end, once every other check has passed (not on --dry-run).
        bench = scored_artifact.benchmark(args.task)
        sa_task = scored_artifact.build_task(
            bench, scored_artifact.tasks_root(bench.name) / name, agents=args.agents, cli=task_cli,
            codex_login=codex_login,
            cpus=args.cpus, memory_mb=args.memory_mb, **({"ultra": True} if ultra_mode else {}))
        task = sa_task.path
        settings = tomllib.loads(sa_task.task_toml)
    elif args.task == "frontier-swe":
        if (ROOT / "jobs" / name).exists():
            raise ValueError(f"job already exists: {ROOT / 'jobs' / name}; choose a new --name")
        # Resources go into the task's [environment] (k agents get k times the task's
        # CPUs and memory) rather than --override-cpus, which would also resize the
        # verifier. Written at the end, once every other check has passed.
        fswe_task = frontier_swe.build_task(
            fswe_name, frontier_swe.TASKS / name, clone=args.frontier_swe, agents=args.agents, cli=task_cli,
            cpus=args.cpus, memory_mb=args.memory_mb, timeout=args.timeout,
            expose_scores=args.expose_scores,
            snapshot_interval=None if args.no_snapshots else (
                30.0 if args.snapshot_interval is None else args.snapshot_interval), ultra=ultra_mode)
        task = fswe_task.path
        settings = tomllib.loads(fswe_task.task_toml)
        # While agents run they reach their model API and nothing else (network_allowlist.py).
        fswe_hosts = network_allowlist.allowed_hosts(
            task_cli, codex_login=codex_login,
            extra=args.allow_host)
    else:
        if (ROOT / "jobs" / name).exists():
            raise ValueError(f"job already exists: {ROOT / 'jobs' / name}; choose a new --name")
        # One task per question, run as Harbor's local dataset (browsecomp_task.py): the answer log
        # next to the agents, resources in each task's [environment] (k agents get k times an agent's;
        # --cpus/--memory-mb set the team's total), agents reach only their model API (Codex's web
        # search runs on OpenAI's side). The dataset is fetched on first use (not on --dry-run).
        bc_job = browsecomp_task.build_job(
            browsecomp_task.TASKS / name, bc_questions, agents=args.agents, protocol=protocol, cli=task_cli,
            codex_login=codex_login, cpus=args.cpus, memory_mb=args.memory_mb, dry_run=args.dry_run,
            selection=(f"examples:{args.examples}" if getattr(args, "examples", None) is not None
                       else str(args.problem)), ultra=ultra_mode)
        task = bc_job.path
        settings = tomllib.loads(bc_job.task_toml)
    env = dict(os.environ)
    mounts = []
    # Each CLI's binary and login (a team's CLIs each once).
    for cli in clis:
        if cli == "claude":
            binary = linux_binary("claude", args.claude_binary, args.dry_run)
            if protocol in CLAUDE_PEERS_PROTOCOLS or protocol in SWARMCHAT_PROTOCOLS:
                claude_peers_claude(binary, args.dry_run, protocol)
            mounts.append({"type": "bind", "source": str(binary),
                           "target": "/mnt/release-claude/claude", "read_only": True,
                           "bind": {"create_host_path": False}})
            claude_auth(env, args.dry_run)
        else:
            binary = linux_binary("codex", args.codex_binary, args.dry_run)
            # Models that run tools in code mode (gpt-5.6 and later) need the
            # codex-code-mode-host next to codex, so mount the release package.
            package = binary.parent.parent
            if (package / "codex-package.json").is_file():
                mounts.append({"type": "bind", "source": str(package),
                               "target": "/mnt/release-codex", "read_only": True,
                               "bind": {"create_host_path": False}})
            elif args.dry_run or (binary.parent / "codex-code-mode-host").is_file():
                mounts.append({"type": "bind", "source": str(binary.parent),
                               "target": "/mnt/release-codex/bin", "read_only": True,
                               "bind": {"create_host_path": False}})
            else:
                raise ValueError("--codex-binary must come from the codex-package release "
                                 "(docs/harness.md) so codex-code-mode-host sits next to codex")
            if protocol in BOARD_PROTOCOLS:
                codex_version(package, f"--protocol {protocol}", BOARD_MIN_CODEX, "its remote message board",
                              args.dry_run)
            if protocol in SWARMCHAT_PROTOCOLS:
                codex_version(package, "--protocol swarmchat", SWARMCHAT_MIN_CODEX, "its PostToolUse hook",
                              args.dry_run)
            if ultra_mode:
                codex_version(package, "--protocol ultra", ultra.MIN_CODEX,
                              "its multi_agent_mode_hint_text and thread cap", args.dry_run)
            mounts += codex_auth(env, args.codex_auth, args.dry_run)
            if fswe_task is not None:
                # The image's agent uid may not own the login; the harness setup links it,
                # or relays a copy the agent can open, at the path the runner expects.
                for mount in mounts:
                    if mount["target"] == "/mnt/release-codex-auth/auth.json":
                        mount["target"] = frontier_swe.CODEX_AUTH_HOST
    if args.token_budget is not None:
        env["RELEASE_TOKEN_BUDGET"] = str(args.token_budget)
        env["RELEASE_TOKEN_BUDGET_METRIC"] = "weighted"
    else:
        env.pop("RELEASE_TOKEN_BUDGET", None)
        env.pop("RELEASE_TOKEN_BUDGET_METRIC", None)
    if relay is not None:
        # agent.py runs the agent's legs one after another (RELAY_ENV).
        env["RELEASE_RELAY"] = str(relay)
    else:
        env.pop("RELEASE_RELAY", None)
    if args.resources_tool:
        # agent.py's `resources` command reports time left of the agent's limit, --timeout.
        env["RELEASE_RESOURCES_TOOL"] = "1"
    else:
        env.pop("RELEASE_RESOURCES_TOOL", None)
    if limit_resume(args):
        # agent.py's runners resume after usage limits, enforce --timeout themselves without
        # the waiting, and stop an agent after --limit-wait-max of waiting.
        env["RELEASE_LIMIT_RESUME"] = "1"
        env["RELEASE_LIMIT_WAIT_MAX"] = str(round(args.limit_wait_max * 3600))
    else:
        env.pop("RELEASE_LIMIT_RESUME", None)
        env.pop("RELEASE_LIMIT_WAIT_MAX", None)
    if args.resources_tool or limit_resume(args):
        env["RELEASE_AGENT_TIMEOUT"] = str(args.timeout)
    else:
        env.pop("RELEASE_AGENT_TIMEOUT", None)
    if harness_guard(args):
        # agent.py runs the watchers and the runners under a keeper that an agent's own commands
        # cannot stop for good (harness_guard.py).
        env["RELEASE_HARNESS_GUARD"] = "1"
    else:
        env.pop("RELEASE_HARNESS_GUARD", None)
    if team_limit_hold(args):
        # A usage-limit wait of any agent pauses the whole team (agent.py, team_hold.py).
        env["RELEASE_TEAM_LIMIT_HOLD"] = "1"
    else:
        env.pop("RELEASE_TEAM_LIMIT_HOLD", None)
    if lockstep(args):
        # The team's agents spend their budgets together (agent.py, lockstep.py).
        env["RELEASE_LOCKSTEP_GAP"] = str(args.lockstep_gap)
        env["RELEASE_LOCKSTEP_REF"] = LOCKSTEP_REF
        env["RELEASE_LOCKSTEP_STALL"] = str(round(LOCKSTEP_STALL_MINUTES * 60))
    else:
        for setting in ("RELEASE_LOCKSTEP_GAP", "RELEASE_LOCKSTEP_REF", "RELEASE_LOCKSTEP_STALL"):
            env.pop(setting, None)
    if sa_task is not None:
        continuation = sa_task.continuation.rstrip("\n")
    elif ale_job is not None:
        continuation = ale_job.continuation.rstrip("\n")
    elif fswe_task is not None:
        continuation = fswe_task.continuation.rstrip("\n")
    else:
        continuation = bc_job.continuation.rstrip("\n")
    # Ultra mode is one agent (agent.py's single runner) whose Codex may run args.agents threads, itself included.
    env.update(MULTIAGENT_N_AGENT=str(1 if ultra_mode else args.agents),
               MULTIAGENT_PROTOCOL=f"multiagent_protocols.{protocol}:MultiAgentProtocol",
               RELEASE_CONTINUATION=continuation, RELEASE_CONTINUATIONS=str(args.continuations),
               MULTIAGENT_AGENT_EFFORT=args.effort or "", RELEASE_TASK=args.task)
    if ultra_mode:
        env[ultra.ENV] = str(args.agents)
    else:
        env.pop(ultra.ENV, None)
    if bc_job is not None:
        # An agent's work ends when its turn ends after it has submitted (agent.py's episode watcher).
        env["RELEASE_EPISODE_OVER_COMMAND"] = browsecomp_task.EPISODE_OVER_COMMAND
    else:
        env.pop("RELEASE_EPISODE_OVER_COMMAND", None)
    env.pop("GH_TOKEN", None)
    env.pop("GITHUB_TOKEN", None)
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # The environment: Docker, with every container's processes capped unless --pids-limit off
    # (pids_limit.py; the cap is recorded in the job's config.json).
    cmd = ["harbor", "run", "-p", str(task), "--yes",
           *pids_limit.harbor_args(getattr(args, "pids_limit", pids_limit.DEFAULT)),
           "--no-force-build",
           "--mounts-json", json.dumps(mounts), "--jobs-dir", str(ROOT / "jobs"),
           "--job-name", name, "-k", str(args.trials), "-n", str(args.concurrent),
           # A team's model is its agents' models joined by "+"; the team kwarg below has each one.
           "--agent-import-path", f"agent:{TEAM_AGENT_CLASS if team else AGENT_CLASSES[args.cli]}",
           "-m", "+".join(member["model"] for member in team) if team else args.model,
           "--agent-timeout-multiplier", str(harbor_agent_timeout(args) / settings["agent"]["timeout_sec"])]
    if "claude" in clis:
        window = CLAUDE_COMPACT_WINDOW if args.claude_compact_window is None else args.claude_compact_window
        cmd += ["--ak", f"claude_compact_window={window}"]
        if getattr(args, "content_filter_max_blocks", None) is not None:
            cmd += ["--ak", f"content_filter_max_blocks={args.content_filter_max_blocks}"]
    if args.codex_compact_limit is not None:
        cmd += ["--ak", f"codex_compact_limit={args.codex_compact_limit}"]
    if team:
        # Each agent's CLI, model and effort, which the job's config.json records.
        members = [{**member, "effort": member["effort"] or args.effort} for member in team]
        cmd += ["--ak", "team=" + json.dumps(members, separators=(",", ":"))]
    if sa_task is not None:
        # The agents' allowlist (their model API) and the job's record of what it ran.
        cmd += sa_task.harbor_args
        if not args.dry_run:
            sa_task.write()
    if fswe_task is not None:
        cmd += network_allowlist.harbor_args(fswe_hosts)
        cmd += fswe_task.harbor_args
        # Recorded in the job's config.json: the task and the pinned commit.
        for key, value in fswe_task.verifier_env.items():
            cmd += ["--ve", f"{key}={value}"]
        if not args.dry_run:
            fswe_task.write()
    if ale_job is not None:
        # The agents' allowlist (their model API) and the job's record of what it ran.
        cmd += ale_job.harbor_args
        if not args.dry_run:
            ale_job.write()
    if bc_job is not None:
        # The agents' allowlist (their model API) and the job's record of what it ran.
        cmd += bc_job.harbor_args
        if not args.dry_run:
            bc_job.write()
    return cmd, env, ROOT / "jobs" / name


# git-to-zig's verifier writes its test suite's trash directories to /tmp: in RAM, each snapshot grading writes ~9x
# less to the disk and runs ~2x faster, with the same rewards. Needs the pids cap (the capped environment writes the
# mount).
GITZIG_SNAPSHOT_TMPFS = "/tmp:rw,exec,nosuid,size=6g,mode=1777"


def check_fastgrade(args: argparse.Namespace) -> None:
    if getattr(args, "fastgrade", False) and not (
            args.task == "frontier-swe" and getattr(args, "frontier_swe_task", None) == "git-to-zig"
            and getattr(args, "pids_limit", None) is not None and not args.no_snapshots):
        raise ValueError("--fastgrade grades frontier-swe git-to-zig snapshots only (with the pids cap and snapshots on)")


def snapshot_tmpfs(args: argparse.Namespace) -> list[str]:
    if (args.task == "frontier-swe" and getattr(args, "frontier_swe_task", None) == "git-to-zig"
            and getattr(args, "pids_limit", None) is not None):
        return ["--tmpfs", GITZIG_SNAPSHOT_TMPFS]
    return []


def snapshot_worker(args: argparse.Namespace, cmd: list[str], env: dict[str, str],
                    job: Path, revive: bool = False) -> tuple[subprocess.Popen, object, float]:
    """frontier-swe: the host process that grades snapshots of /app during the run, and how long
    to let it finish a grading in flight once Harbor is done."""
    check_fastgrade(args)
    task = Path(cmd[cmd.index("-p") + 1])
    settings = tomllib.loads((task / "task.toml").read_text())
    interval = 30.0 if args.snapshot_interval is None else args.snapshot_interval
    job.mkdir(parents=True, exist_ok=revive)
    log = (job / f"{args.task}-snapshots.log").open("a" if revive else "w")
    worker = subprocess.Popen([
        sys.executable, str(ROOT / "frontier_swe.py"), "watch", "--job", str(job), "--task-dir", str(task),
        "--interval", str(interval), *(["--expose-scores"] if args.expose_scores else []),
        *([] if getattr(args, "pids_limit", None) is None else ["--pids-limit", str(args.pids_limit)]),
        *snapshot_tmpfs(args), *(["--fastgrade"] if getattr(args, "fastgrade", False) else []),
        "--watch-pid", str(os.getpid())], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    return worker, log, float(settings["verifier"]["timeout_sec"]) + 600


def start_host_worker(args: argparse.Namespace, cmd: list[str], env: dict[str, str], job: Path,
                      revive: bool = False) -> tuple[subprocess.Popen | None, object, float | None]:
    """The host process some tasks run next to Harbor (frontier-swe's snapshot grader), its log, and
    the grace for a grading in flight. revive.py starts it again for a revived trial (revive: appending
    to its log)."""
    if args.task == "frontier-swe" and not args.no_snapshots:
        return snapshot_worker(args, cmd, env, job, revive)
    return None, None, None


def stop_host_worker(worker: subprocess.Popen | None, log, grace: float | None, finished: bool) -> None:
    if worker is not None and grace is not None:
        # SIGTERM lets a snapshot grading in flight finish (its snapshot was taken
        # during the run); after an interruption, SIGINT cancels it.
        worker.send_signal(signal.SIGTERM if finished else signal.SIGINT)
        try:
            worker.wait(timeout=grace if finished else 300)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
    elif worker is not None:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
    if log is not None:
        log.close()


def after_harbor(args: argparse.Namespace, job: Path) -> None:
    """Records some tasks add once Harbor is done (also after a revived trial)."""
    if args.task == "ale-bench":
        # The judge's log with the agents' cumulative tokens (ale_bench_task.py).
        try:
            import ale_bench_task
            ale_bench_task.annotate_job(job)
        except Exception as exc:  # analysis only; the run's results are already written
            print(f"warning: could not add tokens to the ale-bench log: {exc}", file=sys.stderr)
    if args.task == "browsecomp":
        # Every submission graded by the judge on the host, then the audit (browsecomp_task.py);
        # `python browsecomp_task.py grade jobs/JOB` does it again.
        try:
            import browsecomp_task
            judge = browsecomp_task.Judge(auth=getattr(args, "codex_auth", None) or browsecomp_task.Judge().auth)
            browsecomp_task.annotate_job(job, judge)
        except Exception as exc:  # the run's results are already written
            print(f"warning: could not grade the browsecomp job ({exc}); run python browsecomp_task.py grade {job}",
                  file=sys.stderr)


def main() -> int:
    args = parser().parse_args()
    try:
        cmd, env, job = prepare(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.dry_run:
        print(shlex.join(cmd))
        return 0
    if job.exists():
        raise SystemExit(f"job already exists: {job}; choose a new --name")
    worker = None
    log = None
    grace = None
    finished = False
    try:
        worker, log, grace = start_host_worker(args, cmd, env, job)
        code = subprocess.call(cmd, cwd=ROOT, env=env)
        finished = True
        after_harbor(args, job)
        return code
    finally:
        stop_host_worker(worker, log, grace, finished)


if __name__ == "__main__":
    raise SystemExit(main())
