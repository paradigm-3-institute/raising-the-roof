#!/usr/bin/env python3
"""FrontierSWE v2 tasks under this repository's agents, protocols, and Harbor.

`run.py frontier-swe --task NAME` writes one task directory per job under
.cache/frontier-swe-tasks/<job>/ from tasks/NAME of Proximal's frontier-swe-v2 at the
pinned commit (a blobless, sparse clone in .cache/frontier-swe-v2 that fetches only the
tasks you run). The task keeps its pinned agent and verifier images, its [agent] user,
its separate verifier, its graded [[artifacts]], and its network allowlist, so Harbor
0.23.0 runs and grades it as px-eval does. The adapter adds:

- resources: the task's CPUs and memory per agent, so a team of k gets k times both
  (a task that asks for GPUs stops before it starts);
- network: the task's allowlist (no hosts), plus the agent CLI's model API while the
  agents run, which run.py adds (network_allowlist.py, shared with other tasks);
- the harness section of instruction.md and the tools in benchmarks/frontier-swe:
  every agent works in its own copy of the graded files (/work/<agent>) and promotes
  improvements into /app with `submit`, which keeps /app at the best version by the
  agents' local self-check scores;
- snapshot grading: a host worker grades /app with the task's own verifier, in a fresh
  container from the verifier image (Harbor's regrade path, as for the final reward),
  after promotions and at an interval, and records the reward with the time, in
  <trial>/frontier-swe/snapshots.jsonl, out of the agents' reach.

    python frontier_swe.py fetch git-to-zig            # check out one task
    python frontier_swe.py task git-to-zig --agents 2  # write a task to read
    python frontier_swe.py check git-to-zig            # reference solution, end to end
    python frontier_swe.py regrade jobs/<job> --final  # grade a trial's final /app again
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import ultra

ROOT = Path(__file__).resolve().parent
BENCH = ROOT / "benchmarks" / "frontier-swe"
CLONE = ROOT / ".cache" / "frontier-swe-v2"
TASKS = ROOT / ".cache" / "frontier-swe-tasks"
REPO_URL = "https://github.com/Proximal-Labs/frontier-swe-v2.git"
COMMIT = "da83f84f8fbcec3cbf0f2b17c98ebc811355c2df"
SITE = "https://www.frontierswe.com/tasks/"
LICENSE_NOTE = ("FrontierSWE v2 by Proximal Labs (github.com/Proximal-Labs/frontier-swe-v2): "
                "task content and test harnesses are provided for evaluation purposes; see each "
                "task directory for attribution of upstream projects and datasets.")

@dataclass(frozen=True)
class Layout:
    """The names this module's machinery (working copies, promotions, snapshot grading) uses:
    run.py's task name, the harness tools and logs inside the agents' container, and the
    snapshot state in each trial on the host."""

    benchmark: str  # run.py's task, RELEASE_TASK
    tools_mount: str  # the harness tools, mounted read-only
    log_name: str  # /logs/agent/<log_name> inside, <trial>/agent/<log_name> on the host
    state: str  # <trial>/<state>/: snapshots; not mounted into any container
    description: str  # the task directory's record of what the job runs
    env_prefix: str  # <env_prefix>_TASK and _COMMIT in the job's verifier env
    snapshot_prefix: str  # trial names of snapshot gradings
    lock_dir: str  # the promotion lock's directory, next to the working copies

    @property
    def log_dir(self) -> str:
        return f"/logs/agent/{self.log_name}"

    @property
    def lock(self) -> str:
        return f"{self.lock_dir}/promote.lock"


FRONTIER_SWE = Layout("frontier-swe", "/opt/fswe", "frontier-swe", "frontier-swe", "frontier-swe.json",
                      "FRONTIER_SWE", "fswe-snap-", "/work/.fswe")
LAYOUTS = (FRONTIER_SWE,)

# Inside the task container (see benchmarks/frontier-swe/fswe.py).
TOOLS_MOUNT = FRONTIER_SWE.tools_mount
APP = "/app"
WORK_ROOT = "/work"
LOG_DIR = FRONTIER_SWE.log_dir
LOCK = FRONTIER_SWE.lock
CODEX_AUTH_TARGET = "/mnt/release-codex-auth/auth.json"  # what agent.py's runner links
CODEX_AUTH_HOST = "/mnt/release-codex-auth-host/auth.json"  # the shared login, mounted
# Per trial, next to agent/ and verifier/; not mounted into any container.
STATE = FRONTIER_SWE.state
SNAPSHOT_PREFIX = FRONTIER_SWE.snapshot_prefix

# Build outputs that submit never copies into /app. The task's verifier deletes them
# before it builds (verilog-simulator-in-swift's verify.py removes /app/.build), so
# leaving them out cannot change a reward; it keeps /app and the snapshots small.
PROMOTE_EXCLUDE = {
    "verilog-simulator-in-swift": (".build",),
}
# frontierswe.com task pages use their own slugs.
SITE_SLUGS = {
    "astronomy-toolkit": "autonomous-astrometry-toolkit", "crash-proof-flash-filesystem": "flash-fs",
    "dart-style-in-haskell": "dart-style-haskell", "ffmpeg-libswscale-optimization": "ffmpeg-swscale-rewrite",
    "fitness-recap-video-in-remotion": "remotion-video-generation",
    "flight-sim-renderer-in-opengl": "opengl-scene-engine", "frogsgame-post-training": "frogsgame-rl",
    "kolmogorov-audio-compression": "audio-compression-csound",
    "lean-4-kernel-type-checker-in-pascal": "lean-checker-pascal", "libexpat-optimization": "libexpat-to-x86asm",
    "machine-learned-interatomic-potential": "materials-ml-potential", "optimizer-design": "optimizer-design-v2",
    "postgresql-18-on-sqlite": "postgres-sqlite-wire-adapter", "quantum-espresso-pwx-in-rust": "qe-pwx-rust",
    "reconnaissance-blind-chess-recovery": "rbc-chess-recovery",
    "sglang-inference-system-optimization": "inference-system-optimization-v2",
    "spice-circuit-simulator-in-rust": "spice-sim-rust", "stepper-music-sequencer-gba": "gba-stepper",
    "verilog-simulator-in-swift": "verilog-sim-swift", "vision-only-torcs-racing-bot": "torcs-ai-bot",
    "wan-2.1-on-max-mojo": "modular-stack-wan21",
}
# Keys of a task's job.yaml runtime profile that Harbor's docker environment honours.
PROFILE_POLICIES = {"cpu_enforcement_policy": "--cpus", "memory_enforcement_policy": "--memory"}

# The harness section of instruction.md for a team. A single agent gets the same text
# minus what only makes sense with peers (SOLO_EDITS), as antiherding_matched does for
# the protocol; each edit fails loudly if HARNESS changes under it.
HARNESS = """\
## Working copy and submissions

/app is the graded workspace: when the run ends, the task's verifier grades {graded}.
{n} agents share this container and /app, so nobody edits /app directly.
Each agent works in its own copy of the graded files:

    workcopy    # copies /app's graded files to /work/$AGENT_ID once; prints the path

Edit, build and test only in your copy, and keep build outputs inside it. Task tools
that name /app paths act on /app: to run one on your copy, copy the tool elsewhere and
change those paths.

When your copy beats the version in /app, promote it:

    submit --score S --note "what changed"

S is your copy's result on the task's own self-check, where higher is better (give the
negative of a lower-is-better number). submit replaces /app's graded files with your
copy's only if S is at least the score of the version /app holds; --force skips that
check. Every call is logged in /logs/agent/frontier-swe/submissions.jsonl, which all
agents read. To build on another agent's version, copy its files from /app or from its
/work directory into your copy.

Only what /app holds when the run ends is graded, and the run can end without warning:
submit as soon as your copy beats the version in /app, and again after every improvement.
{verifier_line}
"""
SOLO_EDITS = (
    ("{n} agents share this container and /app, so nobody edits /app directly.",
     "Do not edit /app directly."),
    ("Each agent works in its own copy of the graded files:",
     "Work in your own copy of the graded files:"),
    ("check. Every call is logged in /logs/agent/frontier-swe/submissions.jsonl, which all",
     "check. Every call is logged in /logs/agent/frontier-swe/submissions.jsonl."),
    ("agents read. To build on another agent's version, copy its files from /app or from its",
     None),
    ("/work directory into your copy.", None),
)
# Ultra mode (run.py --protocol ultra; ultra.py): one agent and its own helpers, who share one $AGENT_ID, so each
# names its copy. The team's text with these edits; the cap line comes before the task's text
# (agent.py) and the resources line says it counts the helpers.
ULTRA_EDITS = (
    ("{n} agents share this container and /app, so nobody edits /app directly.",
     "Do not edit /app directly. Every agent, you and each helper, works in its own copy of the graded files:"),
    ("Each agent works in its own copy of the graded files:", None),
    ("    workcopy    # copies /app's graded files to /work/$AGENT_ID once; prints the path",
     "    workcopy NAME    # copies /app's graded files to /work/NAME once; prints the path"),
    ("Edit, build and test only in your copy, and keep build outputs inside it. Task tools",
     "Give each agent its own NAME. Edit, build and test only in your copy, and keep build outputs inside it. "
     "Task tools"),
    ("When your copy beats the version in /app, promote it:",
     "When your copy beats the version in /app, promote it from inside your copy:"),
)
# The continuation's two mentions of the commands, in ultra mode (each must appear once).
ULTRA_CONTINUATION_EDITS = (
    ("your working copy (run `workcopy` to print its path)", "your working copy (`workcopy NAME` prints its path)"),
    ('then run `submit --score S --note "..."` to promote', 'then run `submit --score S --note "..."` from inside it '
                                                           'to promote'),
)
ULTRA_ABOUT = "every agent, the root and each of its helpers, works in its own copy (they share one $AGENT_ID)."
VERIFIER_HIDDEN = "The verifier's scores are not shown during the run."
VERIFIER_EXPOSED = ("After promotions the verifier also grades /app; its scores appear in\n"
                    "/logs/agent/frontier-swe/verifier-scores.jsonl minutes later.")

ORACLE_FLAG = "HARBOR_ORACLE_FLAG"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


# ----------------------------------------------------------------------------
# The pinned upstream checkout


def check_name(name) -> str:
    name = str(name).strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", name):
        raise ValueError(f"--task must be a FrontierSWE v2 task name such as git-to-zig, not {name!r}")
    return name


def _git(clone: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(clone), *args], capture_output=True, text=True)


def ensure_clone(clone: Path | None = None, *, repo_url: str | None = None,
                 commit: str | None = None) -> Path:
    """A blobless, sparse clone of frontier-swe-v2 checked out at the pinned commit."""
    clone, repo_url, commit = clone or CLONE, repo_url or REPO_URL, commit or COMMIT
    clone = Path(clone).expanduser().resolve()
    fresh = not (clone / ".git").exists()
    if fresh:
        clone.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", "--sparse",
                                 "--no-checkout", repo_url, str(clone)], capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"could not clone {repo_url} into {clone}: {result.stderr.strip()}")
    if _git(clone, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        fetched = _git(clone, "fetch", "--quiet", "--filter=blob:none", "origin", commit)
        if fetched.returncode != 0:
            raise ValueError(f"could not fetch commit {commit} into {clone}: {fetched.stderr.strip()}")
    head = _git(clone, "rev-parse", "HEAD")
    if fresh or head.returncode != 0 or head.stdout.strip() != commit:
        checkout = _git(clone, "-c", "advice.detachedHead=false", "checkout", "--quiet", commit)
        if checkout.returncode != 0:
            raise ValueError(f"could not check out {commit} in {clone}: {checkout.stderr.strip()}")
    return clone


def available_tasks(clone: Path | None = None, commit: str | None = None) -> list[str]:
    clone, commit = clone or CLONE, commit or COMMIT
    listed = _git(Path(clone), "ls-tree", "--name-only", f"{commit}:tasks")
    return sorted(listed.stdout.split()) if listed.returncode == 0 else []


def fetch_task(name, clone: Path | None = None, *, repo_url: str | None = None,
               commit: str | None = None) -> Path:
    """Check out tasks/NAME at the pinned commit; only its files are downloaded."""
    commit = commit or COMMIT
    name = check_name(name)
    clone = ensure_clone(clone, repo_url=repo_url, commit=commit)
    relative = f"tasks/{name}"
    kind = _git(clone, "cat-file", "-t", f"{commit}:{relative}")
    if kind.returncode != 0 or kind.stdout.strip() != "tree":
        known = available_tasks(clone, commit)
        raise ValueError(f"FrontierSWE v2 has no task {name!r} at commit {commit[:12]}; tasks: "
                         + ", ".join(known))
    path = clone / relative
    if not (path / "task.toml").is_file():
        added = _git(clone, "sparse-checkout", "add", relative)
        if added.returncode != 0 or not (path / "task.toml").is_file():
            raise ValueError(f"could not check out {relative} in {clone}: "
                             f"{added.stderr.strip() or 'task.toml still missing'}")
    return path


@dataclass
class Upstream:
    """A task as frontier-swe-v2 ships it."""

    name: str
    path: Path
    commit: str
    toml_text: str
    config: dict

    @property
    def environment(self) -> dict:
        return self.config.get("environment") or {}

    @property
    def verifier(self) -> dict:
        return self.config.get("verifier") or {}

    @property
    def verifier_environment(self) -> dict:
        return self.verifier.get("environment") or {}

    @property
    def agent_user(self) -> str | None:
        return (self.config.get("agent") or {}).get("user")

    @property
    def agent_timeout(self) -> float:
        return float((self.config.get("agent") or {}).get("timeout_sec") or 3600)

    @property
    def graded(self) -> list[dict]:
        """The [[artifacts]] entries as {source, exclude}."""
        entries = []
        for artifact in self.config.get("artifacts") or []:
            if isinstance(artifact, str):
                entries.append({"source": artifact, "exclude": []})
            else:
                entries.append({"source": artifact["source"], "exclude": list(artifact.get("exclude") or [])})
        return entries

    @property
    def profile(self) -> dict:
        """The environment block of the task's job.yaml runtime profile, if any."""
        path = self.path / "job.yaml"
        if not path.is_file():
            return {}
        import yaml
        return (yaml.safe_load(path.read_text()) or {}).get("environment") or {}


def load_task(name, clone: Path | None = None, *, fetch: bool = True, repo_url: str | None = None,
              commit: str | None = None) -> Upstream:
    name = check_name(name)
    clone, commit = clone or CLONE, commit or COMMIT
    path = fetch_task(name, clone, repo_url=repo_url, commit=commit) if fetch \
        else Path(clone).expanduser().resolve() / "tasks" / name
    if not (path / "task.toml").is_file():
        raise ValueError(f"FrontierSWE task {name} is not checked out at {path}")
    text = (path / "task.toml").read_text()
    return Upstream(name, path, commit, text, tomllib.loads(text))


# ----------------------------------------------------------------------------
# task.toml edits


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f"cannot write {value!r} to TOML")


def _header_pattern(table: str) -> re.Pattern:
    return re.compile(rf"^\s*\[\s*{re.escape(table)}\s*\]\s*(#.*)?$")


def set_toml(text: str, edits: dict[tuple[str, str], object]) -> str:
    """Set scalar keys of standard tables ("environment", "cpus"): 4, editing lines in
    place so comments and layout survive. The result is parsed and must equal the
    original with exactly these keys changed."""
    lines = text.splitlines()
    expected = tomllib.loads(text)
    for (table, key), value in edits.items():
        node = expected
        for part in table.split("."):
            node = node.setdefault(part, {})
        node[key] = value
        header = next((i for i, line in enumerate(lines) if _header_pattern(table).match(line)), None)
        if header is None:
            lines += ["", f"[{table}]", f"{key} = {_toml_value(value)}"]
            continue
        end = next((i for i in range(header + 1, len(lines)) if re.match(r"^\s*\[", lines[i])), len(lines))
        assignment = re.compile(rf"^\s*{re.escape(key)}\s*=")
        found = next((i for i in range(header + 1, end) if assignment.match(lines[i])), None)
        if found is not None:
            comment = re.search(r"\s+#[^\"']*$", lines[found])
            lines[found] = f"{key} = {_toml_value(value)}" + (comment.group(0) if comment else "")
        else:
            last = max((i for i in range(header + 1, end) if lines[i].strip()
                        and not lines[i].lstrip().startswith("#")), default=header)
            lines.insert(last + 1, f"{key} = {_toml_value(value)}")
    result = "\n".join(lines) + "\n"
    try:
        produced = tomllib.loads(result)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"task.toml edit did not produce valid TOML ({exc}); edits: {edits}") from exc
    if produced != expected:
        raise ValueError(f"task.toml edit did not produce the intended configuration; edits: {edits}")
    return result


# ----------------------------------------------------------------------------
# The task directory run.py launches


def harness_text(agents: int, graded: list[dict], expose_scores: bool, ultra_mode: bool = False) -> str:
    sources = []
    for entry in graded:
        source = entry["source"].rstrip("/") or "/"
        if source not in sources:
            sources.append(source)
    described = " and ".join(sources) + " (the exact paths are in /opt/fswe/config.json)"
    text = HARNESS.replace("{graded}", described).replace(
        "{verifier_line}", VERIFIER_EXPOSED if expose_scores else VERIFIER_HIDDEN)
    if ultra_mode:
        return solo_text(text, ULTRA_EDITS)
    if agents > 1:
        return text.replace("{n}", str(agents))
    return solo_text(text, SOLO_EDITS)


def solo_text(text: str, edits) -> str:
    """A team's harness text for one agent: each (line, replacement) edit rewrites one
    whole line, or deletes it when the replacement is None; an edit that no longer
    matches exactly one line fails loudly."""
    lines = text.split("\n")
    for old, new in edits:
        if lines.count(old) != 1:
            raise ValueError(f"HARNESS changed; this solo edit no longer applies: {old!r}")
        index = lines.index(old)
        if new is None:
            del lines[index]
        else:
            lines[index] = new
    return "\n".join(lines)


def instruction(upstream_text: str, agents: int, graded: list[dict], expose_scores: bool,
                ultra_mode: bool = False) -> str:
    return upstream_text.rstrip("\n") + "\n\n" + harness_text(agents, graded, expose_scores, ultra_mode)


def ultra_tools(tools_mount: str = TOOLS_MOUNT, work_root: str = WORK_ROOT) -> dict[str, str]:
    """Ultra mode's `workcopy NAME` and `submit` (from inside a copy), on fswe.py's own --agent: the copy's NAME is
    its agent id, which every submission record carries (agent_id, source)."""
    return {
        "workcopy": ultra.workcopy_script(work_root, f'exec python3 {tools_mount}/fswe.py workcopy --agent "$name"',
                                          ULTRA_ABOUT),
        "submit": ultra.submit_script(work_root, f'exec python3 {tools_mount}/fswe.py submit --agent "$name" "$@"',
                                      ULTRA_ABOUT),
    }


def edited(text: str, edits) -> str:
    """TEXT with each (old, new) replacement made, each old appearing exactly once."""
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError(f"the text changed; this edit no longer applies: {old!r}")
        text = text.replace(old, new)
    return text


def compose() -> str:
    """The main service mounts the harness tools."""
    lines = ["services:", "  main:", "    volumes:", f"      - ./fswe:{TOOLS_MOUNT}:ro"]
    return "\n".join(lines) + "\n"


def task_dir_name(name: str) -> str:
    return f"frontier-swe-{name}"


@dataclass
class Task:
    """A FrontierSWE task as run.py launches it. `files` maps paths relative to the task
    directory to their text, `copies` to files copied as they are, and `trees` to
    directories copied whole."""

    path: Path
    upstream: Upstream
    agents: int
    files: dict[str, str]
    copies: dict[str, Path]
    trees: dict[str, Path]
    description: dict
    harbor_args: list[str] = field(default_factory=list)
    layout: Layout = FRONTIER_SWE
    tools_dir: str = "environment/fswe"  # mounted at layout.tools_mount
    continuation_path: Path = BENCH / "continuation.txt"
    # (old, new) replacements in the continuation (ultra mode's ULTRA_CONTINUATION_EDITS); none by default.
    continuation_edits: tuple = ()

    @property
    def verifier_env(self) -> dict[str, str]:
        """Recorded in Harbor's job config, so the job shows what it ran."""
        prefix = self.layout.env_prefix
        return {f"{prefix}_TASK": self.upstream.name, f"{prefix}_COMMIT": self.upstream.commit}

    @property
    def continuation(self) -> str:
        text = self.continuation_path.read_text()
        return edited(text, self.continuation_edits) if self.continuation_edits else text

    @property
    def task_toml(self) -> str:
        return self.files["task.toml"]

    def write(self) -> Path:
        if self.path.exists():
            raise ValueError(f"task directory already exists: {self.path}; choose a new --name")
        staging = self.path.with_name(self.path.name + ".partial")
        if staging.exists():
            shutil.rmtree(staging)
        for relative, source in self.trees.items():
            shutil.copytree(source, staging / relative, symlinks=True)
        for relative, source in self.copies.items():
            (staging / relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staging / relative)
        for relative, text in self.files.items():
            (staging / relative).parent.mkdir(parents=True, exist_ok=True)
            (staging / relative).write_text(text, encoding="utf-8")
        for path in (staging / self.tools_dir).rglob("*"):
            if path.is_file():
                path.chmod(0o755 if path.parent.name == "bin" or path.suffix == ".py" else 0o644)
        staging.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(self.path)
        return self.path


def build_task(name, dest_parent: Path, *, clone: Path | None = None, agents: int = 1, cli: str | None = None,
               cpus: int | None = None, memory_mb: int | None = None,
               timeout: float | None = None, expose_scores: bool = False,
               snapshot_interval: float | None = None, oracle_flag: str | None = None,
               fetch: bool = True,
               repo_url: str | None = None, commit: str | None = None, ultra: bool = False) -> Task:
    """Everything needed to write the task directory dest_parent/frontier-swe-NAME. CLI: the agents'
    CLI, or a mixed team's CLIs joined by "+" (run.py --team). ULTRA (run.py --protocol ultra): one agent
    and its own helpers, AGENTS of them at most, the root included: the team's resources, the solo text
    with named copies (ULTRA_EDITS), `workcopy NAME` and `submit` from inside a copy (ultra_tools)."""
    ultra_mode = ultra
    if ultra_mode and agents < 2:
        raise ValueError("ultra mode needs --agents 2 or more (the most agents at once, the root included)")
    upstream = load_task(name, clone, fetch=fetch, repo_url=repo_url, commit=commit)
    config = upstream.config
    if upstream.verifier.get("environment_mode") != "separate" or not upstream.verifier_environment:
        raise ValueError(f"{upstream.name} has no separate verifier environment; FrontierSWE v2 tasks "
                         "grade in their own verifier image")
    if not upstream.agent_user:
        raise ValueError(f"{upstream.name} names no [agent] user")
    for entry in upstream.graded:
        source = entry["source"].rstrip("/")
        if source != APP and not source.startswith(APP + "/"):
            raise ValueError(f"{upstream.name} grades {entry['source']}, outside {APP}; the adapter's "
                             "working copies need every graded path under /app")
    env, verifier_env = upstream.environment, upstream.verifier_environment
    if env.get("gpus") or verifier_env.get("gpus"):
        raise ValueError(f"{upstream.name} needs GPUs; this adapter runs CPU-only tasks")

    per_cpus, per_memory = int(env.get("cpus") or 1), int(env.get("memory_mb") or 2048)
    total_cpus = cpus if cpus is not None else per_cpus * agents
    total_memory = memory_mb if memory_mb is not None else per_memory * agents
    budget = int(timeout if timeout is not None else upstream.agent_timeout)
    edits: dict[tuple[str, str], object] = {
        ("environment", "cpus"): total_cpus,
        ("environment", "memory_mb"): total_memory,
        # The sandbox timer (sandbox-timer --help) reports this as the run's time budget.
        ("environment.env", "TASK_BUDGET_SECS"): str(budget),
    }
    if oracle_flag:
        edits[("environment.env", ORACLE_FLAG)] = oracle_flag
    toml_text = set_toml(upstream.toml_text, edits)

    harbor_args = []
    profile = upstream.profile
    for key, flag in PROFILE_POLICIES.items():
        if profile.get(key):
            harbor_args += [flag, str(profile[key])]
    ignored_profile = {key: value for key, value in profile.items() if key not in PROFILE_POLICIES}

    promote_exclude = list(PROMOTE_EXCLUDE.get(upstream.name, ()))
    tools_config = {
        "task": upstream.name, "app_root": APP, "agent_user": upstream.agent_user, "agents": agents,
        "graded": upstream.graded, "promote_exclude": promote_exclude, "work_root": WORK_ROOT,
        "log_dir": LOG_DIR, "lock": LOCK, "expose_scores": expose_scores,
        "codex_auth": ({"host": CODEX_AUTH_HOST, "target": CODEX_AUTH_TARGET}
                       if "codex" in (cli or "").split("+") else None),
    }
    if ultra_mode:
        tools_config["ultra"] = True
    description = {
        "task": upstream.name, "commit": upstream.commit, "upstream_dir": str(upstream.path),
        "repository": REPO_URL.removesuffix(".git"), "site": SITE + SITE_SLUGS.get(upstream.name, upstream.name),
        "agents": agents, "cli": cli,
        **({"mode": "ultra", "ultra": {"agents": agents, "processes": 1,
                                        "copies": "workcopy NAME, one per agent (the helpers share one $AGENT_ID)"}}
           if ultra_mode else {}),
        "resources": {"cpus": total_cpus, "memory_mb": total_memory, "per_agent_cpus": per_cpus,
                      "per_agent_memory_mb": per_memory, "gpus": env.get("gpus") or 0,
                      "gpu_types": env.get("gpu_types"), "verifier_cpus": verifier_env.get("cpus"),
                      "verifier_memory_mb": verifier_env.get("memory_mb"),
                      "verifier_gpus": verifier_env.get("gpus") or 0},
        "images": {"agent": env.get("docker_image"), "verifier": verifier_env.get("docker_image")},
        # run.py adds the CLI's model API hosts to the agent allowlist (network_allowlist.py).
        "network": {"agent": env.get("network_mode"), "task_allowed_hosts": env.get("allowed_hosts") or [],
                    "verifier": verifier_env.get("network_mode") or "public"},
        "agent_user": upstream.agent_user, "graded": upstream.graded, "promote_exclude": promote_exclude,
        "task_budget_secs": budget, "verifier_timeout_sec": upstream.verifier.get("timeout_sec"),
        "snapshots": {"interval_minutes": snapshot_interval, "expose_scores": expose_scores},
        "harbor_args": harbor_args, "ignored_runtime_profile": ignored_profile,
        "license": LICENSE_NOTE,
    }
    files = {
        "instruction.md": instruction((upstream.path / "instruction.md").read_text(), agents,
                                      upstream.graded, expose_scores, ultra_mode),
        "task.toml": toml_text,
        "frontier-swe.json": json.dumps(description, indent=2) + "\n",
        "environment/docker-compose.yaml": compose(),
        "environment/fswe/config.json": json.dumps(tools_config, indent=2) + "\n",
    }
    copies = {
        "environment/fswe/fswe.py": BENCH / "fswe.py",
        "environment/fswe/bin/submit": BENCH / "bin" / "submit",
        "environment/fswe/bin/workcopy": BENCH / "bin" / "workcopy",
    }
    if ultra_mode:
        for tool, text in ultra_tools().items():
            del copies[f"environment/fswe/bin/{tool}"]
            files[f"environment/fswe/bin/{tool}"] = text
    trees = {}
    for directory in ("solution", "tests", "preflight"):
        if (upstream.path / directory).is_dir():
            trees[directory] = upstream.path / directory
    path = Path(dest_parent).resolve() / task_dir_name(upstream.name)
    return Task(path, upstream, agents, files, copies, trees, description, harbor_args,
                **({"continuation_edits": ULTRA_CONTINUATION_EDITS} if ultra_mode else {}))


# ----------------------------------------------------------------------------
# Reading jobs (run.py, score-over-budget curves, re-grades)


@dataclass
class JobTask:
    name: str
    commit: str | None
    task_dir: Path | None
    layout: Layout = FRONTIER_SWE


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def job_task(path: Path) -> JobTask | None:
    """The FrontierSWE task a job or trial directory ran,
    or None for other benchmarks."""
    config = _read_json(Path(path) / "config.json")
    tasks = config.get("tasks") or ([config["task"]] if isinstance(config.get("task"), dict) else [])
    task_dir = Path(tasks[0]["path"]) if tasks and tasks[0].get("path") else None
    env = (config.get("verifier") or {}).get("env") or {}
    for layout in LAYOUTS:
        described = _read_json(task_dir / layout.description) if task_dir else {}
        if described.get("task"):
            return JobTask(described["task"], described.get("commit"), task_dir, layout)
        if env.get(f"{layout.env_prefix}_TASK"):
            return JobTask(env[f"{layout.env_prefix}_TASK"], env.get(f"{layout.env_prefix}_COMMIT"), task_dir,
                           layout)
    return None


def task_layout(task_dir: Path) -> Layout:
    """The layout of a task directory written by this module."""
    for layout in LAYOUTS:
        if (Path(task_dir) / layout.description).is_file():
            return layout
    raise ValueError(f"{task_dir} has none of " + ", ".join(layout.description for layout in LAYOUTS))


def trial_layout(trial: Path) -> Layout | None:
    """The layout whose snapshot state a trial directory holds, if any."""
    return next((layout for layout in LAYOUTS if (Path(trial) / layout.state).is_dir()), None)


def trials(job: Path) -> list[Path]:
    job = Path(job)
    if (job / "agent").is_dir():
        return [job]
    return sorted(path for path in job.iterdir() if (path / "agent").is_dir())


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    records = []
    for line in Path(path).read_text(errors="replace").splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _append_jsonl(path: Path, record: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def snapshot_records(trial: Path, layout: Layout | None = None) -> list[dict]:
    layout = layout or trial_layout(trial) or FRONTIER_SWE
    return read_jsonl(Path(trial) / layout.state / "snapshots.jsonl")


def scored_snapshots(trial: Path, layout: Layout | None = None) -> list[dict]:
    """Snapshots with a verifier reward, oldest first."""
    found = [r for r in snapshot_records(trial, layout)
             if r.get("status") in ("graded", "unchanged") and isinstance(r.get("reward"), (int, float))]
    return sorted(found, key=lambda r: r["ts"])


# ----------------------------------------------------------------------------
# Snapshot grading


def container_name(trial_name: str) -> str:
    """The agent's main container: Harbor's compose project <trial>__env, service main."""
    project = trial_name.lower()
    if not re.match(r"^[a-z0-9]", project):
        project = "0" + project
    return re.sub(r"[^a-z0-9_-]", "-", project + "__env") + "-main-1"


def pack_script(graded: list[dict]) -> str:
    """Archive each graded path into the directory $1 exactly as Harbor's artifact
    download does: `tar czf ARCHIVE --exclude=PATTERN ... -C SOURCE .` for a directory,
    the file itself otherwise."""
    steps = []
    for index, entry in enumerate(graded):
        source = shlex.quote(entry["source"])
        flags = " ".join(f"--exclude={shlex.quote(pattern)}" for pattern in entry["exclude"])
        steps.append(f'if [ -d {source} ]; then tar czf "$1/{index}.tgz" {flags} -C {source} .; '
                     f'elif [ -e {source} ]; then cp -p {source} "$1/{index}.file"; fi')
    return "set -e; " + "; ".join(steps)


def pack_command(graded: list[dict], tools: str = TOOLS_MOUNT) -> str:
    """Shell run as root in the agent container: pack_script under the promotion lock,
    then the archives as one tar on stdout."""
    return ("set -e; umask 077; d=$(mktemp -d /tmp/fswe-snapshot.XXXXXX); "
            f"python3 {tools}/fswe.py locked -- sh -c {shlex.quote(pack_script(graded))} sh \"$d\"; "
            'tar cf - -C "$d" .; rm -rf "$d"')


def artifact_configs(task_dir: Path):
    from harbor.models.task.config import TaskConfig as HarborTaskConfig
    config = HarborTaskConfig.model_validate_toml((Path(task_dir) / "task.toml").read_text())
    return config.artifacts


def _keep_links_filter(member: tarfile.TarInfo, dest_path: str) -> tarfile.TarInfo | None:
    """tarfile's "data" filter, except that a symlink whose target is absolute or outside the archive
    is kept as that symlink instead of failing the whole extraction. Agents leave such links in /app
    (zig-out/bin/git-receive-pack -> /app/..., test leftovers); the "data" filter refused them, so
    every later snapshot of the run would fail to capture. Nothing on the host follows these links (tree_manifest records links as links,
    copies keep them as links), and members inside a linked directory still fail the "data" checks."""
    try:
        return tarfile.data_filter(member, dest_path)
    except (tarfile.AbsoluteLinkError, tarfile.LinkOutsideDestinationError):
        if member.issym():
            return member.replace(mode=member.mode & 0o777, uid=None, gid=None, uname=None, gname=None,
                                  deep=False)
        raise


def unpack(stream_path: Path, graded: list[dict], task_dir: Path, artifacts_dir: Path) -> None:
    """Lay out the archives from pack_command where Harbor puts collected artifacts."""
    from harbor.trial.artifact_handler import artifact_host_path
    configs = artifact_configs(task_dir)
    with tempfile.TemporaryDirectory(prefix="fswe-unpack-") as scratch, tarfile.open(stream_path) as outer:
        outer.extractall(scratch, filter="data")
        for index, artifact in enumerate(configs):
            target = artifact_host_path(artifacts_dir, artifact)
            archive, single = Path(scratch) / f"{index}.tgz", Path(scratch) / f"{index}.file"
            if archive.is_file():
                target.mkdir(parents=True, exist_ok=True)
                with tarfile.open(archive, "r:gz") as inner:
                    inner.extractall(path=target, filter=_keep_links_filter)
            elif single.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(single, target)


def tree_manifest(root: Path) -> dict[str, dict]:
    """Every path under root with its kind, mode and content digest."""
    entries = {}
    root = Path(root)
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        mode = stat.S_IMODE(path.lstat().st_mode)
        if path.is_symlink():
            entries[relative] = {"kind": "link", "target": os.readlink(path)}
        elif path.is_dir():
            entries[relative] = {"kind": "dir", "mode": mode}
        elif path.is_file():
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            entries[relative] = {"kind": "file", "mode": mode, "sha256": digest.hexdigest(),
                                 "size": path.stat().st_size}
    return entries


def manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()


def store_blobs(root: Path, manifest: dict, blobs: Path) -> None:
    for relative, entry in manifest.items():
        if entry["kind"] != "file":
            continue
        blob = blobs / entry["sha256"][:2] / entry["sha256"]
        if not blob.exists():
            blob.parent.mkdir(parents=True, exist_ok=True)
            # Its own name: a run's grader and a `keep` process may store the same blob at once.
            temporary = blob.with_name(f"{blob.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
            shutil.copyfile(root / relative, temporary)
            temporary.replace(blob)


def materialize(manifest: dict, blobs: Path, root: Path) -> None:
    for relative, entry in sorted(manifest.items()):
        path = Path(root) / relative
        if entry["kind"] == "dir":
            path.mkdir(parents=True, exist_ok=True)
        elif entry["kind"] == "link":
            path.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(entry["target"], path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(blobs / entry["sha256"][:2] / entry["sha256"], path)
    for relative, entry in sorted(manifest.items(), reverse=True):
        if entry["kind"] in ("dir", "file"):
            os.chmod(Path(root) / relative, entry["mode"])


async def grade_artifacts(task_dir: Path, artifacts: Path, trials_dir: Path, trial_name: str,
                          verifier_env: dict[str, str] | None = None, pids_limit: int | None = None,
                          tmpfs: list[str] | None = None):
    """Grade a collected artifacts directory the way Harbor grades a finished trial:
    a fresh container from the task's verifier image, the artifacts uploaded to their
    sources, the task's tests/test.sh, its timeout and reward parsing (Harbor's
    regrade path). pids_limit caps the grading's containers as the run's (pids_limit.py).
    Returns Harbor's TrialResult."""
    from harbor.models.trial.config import SourceTrialConfig, TaskConfig, TrialConfig, VerifierConfig
    from harbor.trial.regrade import RegradeTrial
    from harbor.trial.trial import Trial

    import pids_limit as pids

    class SnapshotGradeTrial(RegradeTrial):
        async def _prepare(self) -> None:
            await asyncio.to_thread(shutil.copytree, artifacts, self.paths.artifacts_dir, symlinks=True,
                                    dirs_exist_ok=True)

    config = TrialConfig(
        task=TaskConfig(path=Path(task_dir)), trial_name=trial_name, trials_dir=Path(trials_dir),
        environment=pids.environment_config(pids_limit, tmpfs=tmpfs, delete=True),
        verifier=VerifierConfig(env=dict(verifier_env or {})),
        source_trial=SourceTrialConfig(action="regrade", type="local", path=Path(artifacts).parent))
    task, download = await Trial._load_task(config)
    trial = SnapshotGradeTrial(config, _task=task, _task_download_result=download,
                               _source_trial_dir=Path(artifacts).parent)
    return await trial.run()


async def grade_with_probe_retry(task_dir: Path, artifacts: Path, trials_dir: Path, name: str,
                                 verifier_env: dict[str, str] | None, pids_limit: int | None, log=print,
                                 tries: int = 3, wait: float = 60.0, tmpfs: list[str] | None = None,
                                 fast: bool = False):
    """grade_artifacts, tried again when Harbor refuses the task's network allowlist.

    Harbor caches its egress-control probe for the process (functools.cache on
    DockerEnvironment._egress_control_kernel_support): one probe that times out while the host's disk
    stalls makes the snapshot grader refuse every later grading of its run ("network_mode='allowlist' is
    not supported"). So on that refusal the cached answer is cleared and the grading tried again, up to
    `tries` times, `wait` seconds apart."""
    if fast:
        # git-to-zig only: the task's own verifier in a plain container, suite pre-staged, build on tmpfs with a warm
        # zig cache (benchmarks/frontier-swe/fastgrade; same rewards as this path)
        sys.path.insert(0, str(BENCH / "fastgrade"))
        import fastgrade_inrun
        return await fastgrade_inrun.grade_artifacts(task_dir, artifacts, trials_dir, name, verifier_env, pids_limit,
                                                     tmpfs)
    for attempt in range(1, tries + 1):
        try:
            return await grade_artifacts(task_dir, artifacts, trials_dir, name if attempt == 1 else
                                         f"{name}-try{attempt}", verifier_env, pids_limit, tmpfs=tmpfs)
        except ValueError as exc:
            if "allowlist" not in str(exc) or attempt == tries:
                raise
            from harbor.environments.docker.docker import DockerEnvironment
            clear = getattr(DockerEnvironment._egress_control_kernel_support, "cache_clear", None)
            if clear:
                clear()
            log(f"{name}: network allowlist refused (attempt {attempt}/{tries}); probe cache cleared, "
                f"trying again in {wait:.0f} s")
            await asyncio.sleep(wait)


def reward_of(result) -> float | None:
    rewards = (result.verifier_result.rewards if result and result.verifier_result else None) or {}
    value = rewards.get("reward")
    return float(value) if isinstance(value, (int, float)) else None


class TrialWatch:
    """Snapshots of one running trial."""

    def __init__(self, trial: Path, task_dir: Path, graded: list[dict], verifier_env: dict[str, str],
                 interval: float | None, expose: bool, log, layout: Layout = FRONTIER_SWE,
                 pids_limit: int | None = None, tmpfs: list[str] | None = None, fast: bool = False):
        self.trial, self.task_dir, self.graded = Path(trial), Path(task_dir), graded
        self.fast = fast  # watch --fastgrade (git-to-zig)
        self.verifier_env, self.interval, self.expose, self.log = verifier_env, interval, expose, log
        self.layout = layout
        self.pids_limit = pids_limit
        self.tmpfs = list(tmpfs) if tmpfs else None  # RAM mounts for the gradings' containers (watch --tmpfs)
        self.state = self.trial / layout.state
        self.shared = self.trial / "agent" / layout.log_name  # the container's log dir
        self.records_path = self.state / "snapshots.jsonl"
        self.submissions = self.shared / "submissions.jsonl"
        self.container = container_name(self.trial.name)
        previous = read_jsonl(self.records_path)
        self.seq = max((int(r.get("seq") or 0) for r in previous), default=-1) + 1
        graded_before = [r for r in previous if r.get("digest")]
        self.last_digest = graded_before[-1]["digest"] if graded_before else None
        rewarded = [r for r in previous if isinstance(r.get("reward"), (int, float))]
        self.last_reward = rewarded[-1]["reward"] if rewarded else None
        self.last_time = time.time() if previous else 0.0
        self.seen = len(read_jsonl(self.submissions))
        self.pending: dict | None = None
        self.busy = False
        self.done = False
        self.readers: dict[Path, object] = {}
        # Every promotion's code, kept as it is promoted (keep_latest), so each submission can be graded later,
        # also those the grader skips while it is busy (regrade --kept): a large team can promote faster than one
        # grading at a time, and a skipped promotion's code is otherwise gone, overwritten in /app.
        self.kept_path = self.state / "kept.jsonl"
        self.kept_upto = max((int(r.get("submission_seq") or 0) for r in read_jsonl(self.kept_path)), default=0)
        self.keeping = False

    def finished(self) -> bool:
        return (self.trial / "result.json").exists() or (self.shared / "finished").exists()

    def ready(self) -> bool:
        return (self.shared / "ready").exists()

    def trigger(self) -> dict | None:
        """What a snapshot now would be for: the start, a promotion, or the interval."""
        records = read_jsonl(self.submissions)
        promoted = [r for r in records[self.seen:] if r.get("status") == "promoted"]
        self.seen = len(records)
        if promoted:
            self.pending = {"trigger": "submit", "submission": promoted[-1]}
        if self.pending:
            return self.pending
        if self.last_digest is None and self.seq == 0:
            return {"trigger": "start"}
        if self.interval and time.time() - self.last_time >= self.interval * 60:
            return {"trigger": "interval"}
        return None

    def tokens(self) -> dict:
        from token_budget import READERS, summed, weighted_tokens
        for cli, home_dir in (("codex", "codex-home"), ("claude", "claude-home")):
            for home in (self.trial / "agent").glob(f"**/{home_dir}"):
                if home not in self.readers:
                    self.readers[home] = READERS[cli](home)
        totals = {}
        for home, reader in self.readers.items():
            for session, usage in reader.poll().items():
                totals[f"{home}:{session}"] = usage
        usage = summed(totals) if totals else {}
        if not usage:
            return {}
        return {"weighted_tokens": round(weighted_tokens(usage)), **usage}

    def record(self, record: dict) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        with self.records_path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.log(f"{self.trial.name} snapshot {record['seq']} {record['trigger']}: {record['status']}"
                 + (f" reward {record['reward']}" if record.get("reward") is not None else ""))
        if self.expose and record.get("status") in ("graded", "unchanged"):
            exposed = self.shared / "verifier-scores.jsonl"
            with exposed.open("a") as stream:
                stream.write(json.dumps({k: record.get(k) for k in (
                    "seq", "ts", "reward", "submission_seq", "submission_agent")}, sort_keys=True) + "\n")

    async def capture(self, destination: Path) -> None:
        archive = destination.with_suffix(".tar")
        with archive.open("wb") as stream:
            process = await asyncio.create_subprocess_exec(
                "docker", "exec", "-u", "root", self.container, "sh", "-c",
                pack_command(self.graded, self.layout.tools_mount),
                stdout=stream, stderr=asyncio.subprocess.PIPE)
            _, error = await process.communicate()
        try:
            if process.returncode != 0:
                raise RuntimeError(f"docker exec failed ({process.returncode}): "
                                   f"{error.decode(errors='replace').strip()[-500:]}")
            unpack(archive, self.graded, self.task_dir, destination)
        finally:
            archive.unlink(missing_ok=True)
        # Harbor also uploads the agent's publish dir, /logs/artifacts (a host mount).
        from harbor.models.task.artifacts import source_host_path
        published = source_host_path(self.trial / "artifacts", "/logs/artifacts")
        if published.is_dir() and any(published.iterdir()):
            shutil.copytree(published, source_host_path(destination, "/logs/artifacts"), symlinks=True,
                            dirs_exist_ok=True)

    def promotions(self) -> list[dict]:
        return [r for r in read_jsonl(self.submissions) if r.get("status") == "promoted"]

    def keep_due(self) -> bool:
        """A promotion whose code is not kept yet."""
        promoted = self.promotions()
        return bool(promoted) and int(promoted[-1].get("seq") or 0) > self.kept_upto

    def kept(self, record: dict) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        with self.kept_path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.log(f"{self.trial.name} submission {record['submission_seq']} {record['status']}"
                 + (f" {record['digest'][:12]}" if record.get("digest") else ""))

    async def keep_latest(self, tries: int = 5) -> None:
        """Keep the code of the newest promotion: /app copied under the promotion lock, as the grader copies it,
        into the same manifest and blob store. Promotions write /app and their submissions.jsonl record under
        that lock, so the copy holds the newest promoted record's code whenever that record is the same before
        and after the copy; if a promotion landed in between, copy again. A promotion that /app no longer held
        by then (two within one poll) is recorded as lost."""
        self.keeping = True
        try:
            for _ in range(tries):
                promoted = self.promotions()
                if not promoted or int(promoted[-1].get("seq") or 0) <= self.kept_upto:
                    return
                newest = promoted[-1]
                staging = self.state / "staging-keep"
                if staging.exists():
                    shutil.rmtree(staging)
                artifacts = staging / "artifacts"
                artifacts.mkdir(parents=True)
                try:
                    try:
                        await self.capture(artifacts)
                    except Exception:
                        if self.finished():
                            return
                        raise
                    after = self.promotions()
                    if not after or after[-1].get("seq") != newest.get("seq"):
                        continue
                    manifest = tree_manifest(artifacts)
                    digest = manifest_digest(manifest)
                    manifests = self.state / "manifests"
                    manifests.mkdir(parents=True, exist_ok=True)
                    store_blobs(artifacts, manifest, self.state / "blobs")
                    if not (manifests / f"{digest}.json").exists():  # after its blobs, and whole or not at all
                        temporary = manifests / f"{digest}.json.{os.getpid()}.{secrets.token_hex(4)}.tmp"
                        temporary.write_text(json.dumps(manifest, sort_keys=True))
                        temporary.replace(manifests / f"{digest}.json")
                    for lost in promoted:
                        if self.kept_upto < int(lost.get("seq") or 0) < int(newest["seq"]):
                            self.kept({"submission_seq": lost["seq"], "submission_agent": lost.get("agent_id"),
                                       "submission_local_score": lost.get("local_score"), "status": "lost",
                                       "overwritten_by": newest["seq"], "ts": now_iso()})
                    self.kept({"submission_seq": newest["seq"], "submission_agent": newest.get("agent_id"),
                               "submission_local_score": newest.get("local_score"), "submission_ts": newest.get("ts"),
                               "status": "kept", "digest": digest,
                               "files": sum(1 for e in manifest.values() if e["kind"] == "file"), "ts": now_iso(),
                               **self.tokens()})
                    self.kept_upto = int(newest["seq"])
                    return
                finally:
                    shutil.rmtree(staging, ignore_errors=True)
            self.log(f"{self.trial.name} keep: a promotion landed during each of {tries} copies; next poll retries")
        except Exception as exc:  # a failed copy never stops the worker; the next poll tries again
            self.log(f"{self.trial.name} keep failed: {exc!r}")
        finally:
            self.keeping = False

    async def snapshot(self, trigger: dict, slots: asyncio.Semaphore) -> None:
        self.busy = True
        try:
            if trigger["trigger"] == "start":
                await self._snapshot(trigger, grade=False)
            else:
                async with slots:
                    if self.finished():
                        return
                    if trigger is self.pending:
                        self.pending = None
                    await self._snapshot(trigger, grade=True)
        except Exception as exc:  # a failed snapshot never stops the worker
            self.log(f"{self.trial.name} snapshot failed: {exc!r}")
        finally:
            self.busy = False
            self.last_time = time.time()

    async def _snapshot(self, trigger: dict, grade: bool) -> None:
        seq = self.seq
        self.seq += 1
        submission = trigger.get("submission") or {}
        if not submission:
            promoted = [r for r in read_jsonl(self.submissions) if r.get("status") == "promoted"]
            submission = promoted[-1] if promoted else {}
        record = {"seq": seq, "trigger": trigger["trigger"], "ts": now_iso(),
                  "submission_seq": submission.get("seq"), "submission_agent": submission.get("agent_id"),
                  "submission_local_score": submission.get("local_score")}
        staging = self.state / f"staging-{seq}"
        if staging.exists():
            shutil.rmtree(staging)
        artifacts = staging / "artifacts"
        artifacts.mkdir(parents=True)
        try:
            try:
                await self.capture(artifacts)
            except Exception as exc:
                if self.finished():
                    return
                record.update(status="error", error=f"capture: {exc}")
                self.record(record)
                return
            record.update(self.tokens())
            manifest = tree_manifest(artifacts)
            digest = manifest_digest(manifest)
            record.update(digest=digest, files=sum(1 for e in manifest.values() if e["kind"] == "file"))
            (self.state / "manifests").mkdir(parents=True, exist_ok=True)
            (self.state / "manifests" / f"{digest}.json").write_text(json.dumps(manifest, sort_keys=True))
            store_blobs(artifacts, manifest, self.state / "blobs")
            if not grade:
                self.last_digest = digest
                record["status"] = "baseline"
                self.record(record)
                return
            if digest == self.last_digest:
                # Nothing to grade: /app is as last graded (or still as at the start).
                record.update(status="unchanged", reward=self.last_reward)
                self.record(record)
                return
            started = time.time()
            name = f"{self.layout.snapshot_prefix}{self.trial.name}-{seq}"
            try:
                result = await grade_with_probe_retry(self.task_dir, artifacts, self.state / "grades", name,
                                                      self.verifier_env, self.pids_limit, self.log,
                                                      tmpfs=self.tmpfs, fast=self.fast)
            except Exception as exc:
                # Kept in snapshots.jsonl (digest, tokens, time), so the snapshot can be graded later.
                record.update(status="error", error=f"grade: {exc!r}"[:500], graded_at=now_iso())
                self.record(record)
                raise
            reward = reward_of(result)
            if self.tmpfs:
                record["tmpfs"] = True
            if self.fast:
                record["grader"] = "fastgrade"
            record.update(graded_at=now_iso(), verifier_seconds=round(time.time() - started, 1),
                          reward=reward, rewards=(result.verifier_result.rewards if result.verifier_result
                                                  else None),
                          grade_dir=str(self.state / "grades" / name))
            # The verifier's raw numbers, where it writes them.
            metrics = _read_json(self.state / "grades" / name / "verifier" / "metrics.json")
            if metrics:
                record["metrics"] = metrics
            if reward is None:
                info = result.exception_info
                record.update(status="error", error=f"{info.exception_type}: {info.exception_message}"
                              if info else "no reward")
            else:
                record["status"] = "graded"
                self.last_digest, self.last_reward = digest, reward
            self.record(record)
            shutil.rmtree(Path(record["grade_dir"]) / "artifacts", ignore_errors=True)
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def _alive(pid: int | None) -> bool:
    if not pid:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


async def watch(job: Path, task_dir: Path, *, interval: float | None, expose: bool, concurrency: int = 1,
                watch_pid: int | None = None, poll: float | None = None, pids_limit: int | None = None,
                tmpfs: list[str] | None = None, fast: bool = False, keep: bool = True, grade: bool = True) -> None:
    """Snapshot every trial of a running job until run.py exits or sends SIGTERM (then
    finish the gradings in flight) or SIGINT (cancel them; their containers stop).
    pids_limit: run.py's --pids-limit, for the gradings' containers. keep: also keep every promotion's code
    (TrialWatch.keep_latest), polling every 2 s instead of 10 so two promotions rarely land within one poll;
    grade=False only keeps (the `keep` command, for a run whose watcher predates keeping)."""
    poll = poll if poll is not None else (2.0 if keep else 10.0)
    sys.path.insert(0, str(ROOT))
    layout = task_layout(task_dir)
    described = _read_json(Path(task_dir) / layout.description)
    graded = described["graded"]
    job_config = _read_json(Path(job) / "config.json")
    verifier_env = ((job_config.get("verifier") or {}).get("env")) or {}

    def log(message: str) -> None:
        print(f"{now_iso()} {message}", flush=True)

    slots = asyncio.Semaphore(concurrency)
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stopping.set)
    main_task = asyncio.current_task()
    loop.add_signal_handler(signal.SIGINT, main_task.cancel)
    watches: dict[Path, TrialWatch] = {}
    running: set[asyncio.Task] = set()
    log(f"watching {job} (interval {interval} min, expose {expose}, concurrency {concurrency}"
        + ("" if pids_limit is None else f", pids limit {pids_limit}")
        + (f", tmpfs {', '.join(tmpfs)}" if tmpfs else "")
        + (", keeping every promotion" if keep else "") + ("" if grade else ", no grading") + ")")
    try:
        while not stopping.is_set() and _alive(watch_pid):
            try:
                if Path(job).is_dir():
                    for trial in trials(job):
                        if trial not in watches:
                            if not verifier_env:
                                job_config = _read_json(Path(job) / "config.json")
                                verifier_env = ((job_config.get("verifier") or {}).get("env")) or {}
                            watches[trial] = TrialWatch(trial, task_dir, graded, verifier_env, interval,
                                                        expose, log, layout, pids_limit, tmpfs, fast)
                for trial_watch in watches.values():
                    if keep and not trial_watch.done and not trial_watch.keeping and trial_watch.ready() \
                            and not trial_watch.finished() and trial_watch.keep_due():
                        task = asyncio.create_task(trial_watch.keep_latest())
                        running.add(task)
                        task.add_done_callback(running.discard)
                    if not grade or trial_watch.done or trial_watch.busy:
                        continue
                    if trial_watch.finished():
                        trial_watch.done = True
                        continue
                    if not trial_watch.ready():
                        continue
                    trigger = trial_watch.trigger()
                    if trigger:
                        task = asyncio.create_task(trial_watch.snapshot(trigger, slots))
                        running.add(task)
                        task.add_done_callback(running.discard)
            except Exception as exc:  # keep watching; the next poll retries
                log(f"poll failed: {exc!r}")
            try:
                await asyncio.wait_for(stopping.wait(), timeout=poll)
            except asyncio.TimeoutError:
                pass
        if running:
            log(f"finishing {len(running)} snapshot(s) in flight")
            await asyncio.gather(*running, return_exceptions=True)
    except asyncio.CancelledError:
        log("cancelled; stopping snapshot gradings")
        for task in running:
            task.cancel()
        await asyncio.gather(*running, return_exceptions=True)
    log("done")


# ----------------------------------------------------------------------------
# Regrading


def _snapshot_label(trial: Path, record: dict) -> str:
    return f"{trial.parent.name}/{trial.name}#{record['seq']}"


async def regrade(targets: list[Path], *, seqs: list[int] | None = None, best: int | None = None,
                  final: bool = False, repeats: int = 1, fast: bool = False, kept: bool = False) -> list[dict]:
    """Grade stored snapshots (or, with final, a trial's final artifacts) again with the
    task's verifier, `repeats` times each, one at a time. kept: grade instead every kept submission
    (kept.jsonl) not graded yet, into kept-grades.jsonl; one whose code a graded snapshot holds gets that
    snapshot's grade there, without grading again."""
    sys.path.insert(0, str(ROOT))
    import pids_limit as pids

    targets = [Path(target).resolve() for target in targets]  # the fast grader bind-mounts paths under them
    plans = []  # (label, original, task_dir, artifacts source, verifier_env, trial, layout, kept record)
    caps = {}  # trial -> the pids limit its run had (pids_limit.py), for the regrades' containers
    for target in targets:
        job_info = job_task(target if (Path(target) / "config.json").exists() else Path(target).parent)
        for trial in trials(target):
            info = job_task(trial) or job_info
            if info is None or info.task_dir is None:
                raise ValueError(f"{trial} is not a frontier-swe trial with its task directory")
            trial_config = _read_json(trial / "config.json")
            verifier_env = ((trial_config.get("verifier") or {}).get("env")) or {}
            caps[trial] = pids.configured(trial_config.get("environment"))
            if kept:
                state = trial / info.layout.state
                graded = {r["digest"]: r for r in scored_snapshots(trial, info.layout)
                          if r.get("status") == "graded" and r.get("digest")}
                done = {r.get("submission_seq") for r in read_jsonl(state / "kept-grades.jsonl")}
                for record in read_jsonl(state / "kept.jsonl"):
                    if record.get("status") != "kept" or record.get("submission_seq") in done:
                        continue
                    done.add(record.get("submission_seq"))
                    snapshot = graded.get(record["digest"])
                    if snapshot is not None:
                        _append_jsonl(state / "kept-grades.jsonl", {
                            "submission_seq": record["submission_seq"], "digest": record["digest"],
                            "reward": snapshot.get("reward"), "source": f"snapshot {snapshot.get('seq')}",
                            "grader": snapshot.get("grader", "official"), "graded_at": now_iso()})
                        continue
                    plans.append((f"{trial.parent.name}/{trial.name}#submission{record['submission_seq']}", None,
                                  info.task_dir, state / "manifests" / f"{record['digest']}.json", verifier_env,
                                  trial, info.layout, record))
                continue
            if final:
                result = _read_json(trial / "result.json")
                rewards = ((result.get("verifier_result") or {}).get("rewards")) or {}
                plans.append((f"{trial.parent.name}/{trial.name}#final", rewards.get("reward"), info.task_dir,
                              trial / "artifacts", verifier_env, trial, info.layout, None))
                continue
            records = [r for r in scored_snapshots(trial, info.layout) if r.get("status") == "graded"]
            if seqs:
                records = [r for r in records if r["seq"] in seqs]
            if best:
                distinct, seen = [], set()
                for record in sorted(records, key=lambda r: -r["reward"]):
                    if record["digest"] not in seen:
                        seen.add(record["digest"])
                        distinct.append(record)
                records = distinct[:best]
            for record in records:
                plans.append((_snapshot_label(trial, record), record["reward"], info.task_dir,
                              trial / info.layout.state / "manifests" / f"{record['digest']}.json", verifier_env,
                              trial, info.layout, None))
    results = []
    for label, original, task_dir, source, verifier_env, trial, layout, kept_record in plans:
        scores = []
        state = trial / layout.state
        for repeat in range(repeats):
            with tempfile.TemporaryDirectory(prefix="fswe-regrade-", dir=trial) as scratch:
                artifacts = Path(scratch) / "artifacts"
                if source.suffix == ".json":
                    materialize(json.loads(source.read_text()), state / "blobs", artifacts)
                else:
                    shutil.copytree(source, artifacts, symlinks=True)
                    (artifacts / "manifest.json").unlink(missing_ok=True)
                name = f"{layout.snapshot_prefix}regrade-{secrets.token_hex(3)}"
                result = await grade_with_probe_retry(task_dir, artifacts, state / "regrades", name, verifier_env,
                                                      caps.get(trial), fast=fast) if fast else \
                    await grade_artifacts(task_dir, artifacts, state / "regrades", name, verifier_env, caps.get(trial))
                scores.append(reward_of(result))
                shutil.rmtree(state / "regrades" / name / "artifacts", ignore_errors=True)
                if kept_record is not None and scores[-1] is not None:  # a failed grading is tried again next time
                    _append_jsonl(state / "kept-grades.jsonl", {
                        "submission_seq": kept_record["submission_seq"], "digest": kept_record["digest"],
                        "reward": scores[-1], "source": f"regrade {name}",
                        "grader": "fastgrade" if fast else "official", "graded_at": now_iso()})
            print(f"repeat {repeat + 1}: {label} {scores[-1]}", flush=True)
        valid = [s for s in scores if s is not None]
        mean = sum(valid) / len(valid) if valid else None
        stdev = (sum((s - mean) ** 2 for s in valid) / (len(valid) - 1)) ** 0.5 if len(valid) > 1 else 0.0
        results.append({"program": label, "original": original, "mean": mean, "stdev": stdev,
                        "min": min(valid, default=None), "max": max(valid, default=None), "scores": scores})
    return results


# ----------------------------------------------------------------------------
# Command line


def harbor_check_command(built: Task, name: str, flag: str) -> list[str]:
    cmd = ["harbor", "run", "-p", str(built.path), "--yes", "--env", "docker", "--no-force-build",
           "--jobs-dir", str(ROOT / "jobs"), "--job-name", name, "-k", "1", "-n", "1",
           "--agent", "oracle", "--ae", f"{ORACLE_FLAG}={flag}", "--ve", f"{ORACLE_FLAG}={flag}",
           *built.harbor_args]
    for key, value in built.verifier_env.items():
        cmd += ["--ve", f"{key}={value}"]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="check out one task in the frontier-swe-v2 clone")
    listing = commands.add_parser("tasks", help="list the tasks at the pinned commit")
    task = commands.add_parser("task", help="write a task directory to read")
    check = commands.add_parser("check", help="run the task's reference solution through Harbor's oracle "
                                              "agent and the task's verifier")
    for sub in (fetch, listing, task, check):
        sub.add_argument("--frontier-swe", type=Path, default=CLONE, help="frontier-swe-v2 clone")
    for sub in (fetch, task, check):
        sub.add_argument("task")
    task.add_argument("--agents", type=int, default=1)
    task.add_argument("--out", type=Path, default=TASKS / "manual", help="parent directory")
    check.add_argument("--name", help="job name (default: fswe-check-<task>-<time>)")
    watcher = commands.add_parser("watch", help="(run.py) grade snapshots of a running job")
    watcher.add_argument("--job", type=Path, required=True)
    watcher.add_argument("--task-dir", type=Path, required=True)
    watcher.add_argument("--interval", type=float, default=30.0, help="minutes; 0: only on submit")
    watcher.add_argument("--expose-scores", action="store_true")
    watcher.add_argument("--concurrency", type=int, default=1)
    watcher.add_argument("--pids-limit", type=int, help="processes each grading's containers may run (run.py)")
    watcher.add_argument("--watch-pid", type=int)
    watcher.add_argument("--tmpfs", action="append", metavar="PATH:OPTIONS",
                         help="a RAM mount for each grading's container (compose tmpfs syntax, repeatable), e.g. "
                              "/tmp:rw,exec,nosuid,size=6g,mode=1777 for git-to-zig, whose test suite writes there")
    watcher.add_argument("--fastgrade", action="store_true",
                         help="git-to-zig: grade with benchmarks/frontier-swe/fastgrade (same verifier, no Harbor)")
    watcher.add_argument("--no-keep", action="store_true",
                         help="do not keep every promotion's code (kept.jsonl; grade them later with regrade --kept)")
    keeper = commands.add_parser("keep", help="only keep every promotion's code of a running job (for a run whose "
                                              "watcher predates keeping); grade them later with regrade --kept")
    keeper.add_argument("--job", type=Path, required=True)
    keeper.add_argument("--task-dir", type=Path, required=True)
    keeper.add_argument("--watch-pid", type=int, help="stop when this process (the run's run.py) exits")
    again = commands.add_parser("regrade", help="grade stored snapshots or final artifacts again")
    again.add_argument("targets", nargs="+", type=Path, help="job or trial directories")
    again.add_argument("--seq", type=int, nargs="*", help="snapshot numbers (default: all graded)")
    again.add_argument("--best", type=int, help="only the N best distinct snapshots per trial")
    again.add_argument("--final", action="store_true", help="the artifacts Harbor collected at the end")
    again.add_argument("--repeats", type=int, default=1)
    again.add_argument("--fast", action="store_true", help="git-to-zig: grade with benchmarks/frontier-swe/fastgrade")
    again.add_argument("--kept", action="store_true",
                       help="grade every kept submission not graded yet (kept.jsonl), into kept-grades.jsonl")
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            print(fetch_task(args.task, args.frontier_swe))
            return 0
        if args.command == "tasks":
            clone = ensure_clone(args.frontier_swe)
            print("\n".join(available_tasks(clone)))
            return 0
        if args.command == "task":
            print(build_task(args.task, args.out, clone=args.frontier_swe, agents=args.agents).write())
            return 0
        if args.command == "watch":
            asyncio.run(watch(args.job, args.task_dir, interval=args.interval or None,
                              expose=args.expose_scores, concurrency=args.concurrency,
                              watch_pid=args.watch_pid, pids_limit=args.pids_limit, tmpfs=args.tmpfs,
                              fast=args.fastgrade, keep=not args.no_keep))
            return 0
        if args.command == "keep":
            asyncio.run(watch(args.job, args.task_dir, interval=None, expose=False, watch_pid=args.watch_pid,
                              keep=True, grade=False))
            return 0
        if args.command == "regrade":
            for line in asyncio.run(regrade(args.targets, seqs=args.seq, best=args.best, final=args.final,
                                            repeats=args.repeats, fast=args.fast, kept=args.kept)):
                print(json.dumps(line))
            return 0
        name = check_name(args.task)
        job = args.name or f"fswe-check-{name}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        if (ROOT / "jobs" / job).exists():
            raise ValueError(f"job already exists: {ROOT / 'jobs' / job}; choose a new --name")
        flag = secrets.token_hex(16)
        built = build_task(name, TASKS / job, clone=args.frontier_swe, oracle_flag=flag)
        built.write()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    code = subprocess.call(harbor_check_command(built, job, flag), cwd=ROOT)
    for trial in trials(ROOT / "jobs" / job) if (ROOT / "jobs" / job).is_dir() else []:
        result = _read_json(trial / "result.json")
        rewards = ((result.get("verifier_result") or {}).get("rewards")) or {}
        timing = result.get("verifier") or {}
        print(json.dumps({"trial": str(trial), "reward": rewards.get("reward"), "rewards": rewards,
                          "verifier_started": timing.get("started_at"),
                          "verifier_finished": timing.get("finished_at"),
                          "exception": (result.get("exception_info") or {}).get("exception_message")}))
    return code


if __name__ == "__main__":
    sys.exit(main())
