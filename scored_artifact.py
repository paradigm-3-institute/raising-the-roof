#!/usr/bin/env python3
"""Scored-artifact tasks: the agent produces an artifact, a trusted scorer returns a number.

A benchmark plugs in with a subclass of Benchmark (see perf_takehome_task.py) and a
directory benchmarks/<name>/ holding its scoring module (scorer/score.py), the
agents' image (environment/Dockerfile), and its continuation prompt. Registering it
in BENCHMARKS makes `python run.py <name> ...` work with every CLI, protocol, team
size, token budget and job option of the other tasks. This layer supplies the rest:

- the scorer (benchmarks/scored-artifact/scorer/server.py), its own container next
  to the agents' one: it grades from a pristine read-only copy of the benchmark,
  runs agent code only in a sandbox, queues submissions round robin across agents,
  and writes the submission log to <trial>/scorer/ on the host, out of the agents'
  reach (submissions.jsonl, every submission's files, final.json);
- agent commands (benchmarks/scored-artifact/client): `workcopy` (each agent's own
  working copy, /work/$AGENT_ID), `submit`, `scores`, `fetch-submission`; in ultra mode
  (run.py --protocol ultra, for a benchmark with ultra_instruction) `workcopy NAME` and
  `submit` from inside a copy, since the helpers share one $AGENT_ID (ultra.py);
- the verifier (benchmarks/scored-artifact/tests): grades each agent's final working
  copy, then the scorer re-grades the best valid submission; that is the reward;
- the network: Harbor's egress allowlist. While agents run they reach only their
  model API; the setup step before them reaches only the Ubuntu archive (the CLI
  install runs apt-get), and nothing else is reachable at any time;
- resources: each agent gets the benchmark's CPUs and memory, a team of k gets k
  times both (--cpus/--memory-mb set the team's total); the scorer has its own;
- reading jobs back (regrade).

    python scored_artifact.py task perf-takehome --agents 2         # write a task to read
    python scored_artifact.py check perf-takehome --solution F      # grade F end to end (oracle agent)
    python scored_artifact.py grade perf-takehome F [F ...]         # grade files with the scorer only
    python scored_artifact.py regrade jobs/JOB [--top 3 --repeats 1] # re-grade a job's best submissions
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import network_allowlist
import ultra

ROOT = Path(__file__).resolve().parent
LAYER = ROOT / "benchmarks" / "scored-artifact"

# run.py task name -> "module:attribute" of its Benchmark. A new benchmark adds a line.
BENCHMARKS = {
    "perf-takehome": "perf_takehome_task:BENCHMARK",
}

SCORER_PORT = 18090
# Inside the agents' container.
CLIENT_MOUNT = "/opt/scored-artifact"
WORK_ROOT = "/work"
CLIENT_COMMANDS = ("submit", "workcopy", "scores", "fetch-submission")
# Harbor's egress allowlist puts every service of the task's compose file in one
# network namespace (the egress sidecar's), so the agents reach the scorer on
# localhost; "scorer" is its address when a task runs without the allowlist.
SCORER_URLS = (f"http://127.0.0.1:{SCORER_PORT}", f"http://scorer:{SCORER_PORT}")
STATE = "scorer"  # <trial>/scorer/: the scorer's log, files and final grade
SANDBOX_UID = 20000  # first of the scorer's sandbox users, one per worker

# The model API each agent CLI needs, and nothing else: the repository's one list
# (network_allowlist.py), which the FrontierSWE v2 adapter uses too.
MODEL_API_HOSTS = network_allowlist.MODEL_API_HOSTS
# Reachable before the agents start, when Harbor installs the CLI (apt-get), and
# while the verifier runs; never while agents run.
SETUP_HOSTS = ("archive.ubuntu.com", "security.ubuntu.com")


def model_api_hosts(cli: str | None, codex_login: bool) -> list[str]:
    if cli is None:
        return []
    if "+" in cli:
        # A mixed team (run.py --team): each of its CLIs' hosts.
        return list(dict.fromkeys(host for part in cli.split("+") for host in model_api_hosts(part, codex_login)))
    if cli == "codex":
        return list(MODEL_API_HOSTS["codex" if codex_login else "codex-api-key"])
    return list(MODEL_API_HOSTS[cli])


@dataclass
class Prepared:
    """What a benchmark contributes to one task: the agents' starter files, the
    scorer's pristine data (both relative path -> bytes), the scorer's benchmark
    configuration (may hold secrets; agents never see it), and a description."""

    starter: dict[str, bytes]
    data: dict[str, bytes]
    scorer_config: dict
    description: dict


class Benchmark:
    """A scored-artifact benchmark. Subclasses set the class attributes and implement
    prepare() and instruction()."""

    name = ""  # run.py task name, also the task directory's name
    title = ""
    source = ""  # upstream URL, for task.toml
    authors: tuple[dict, ...] = ()
    keywords: tuple[str, ...] = ()
    artifact_files: tuple[str, ...] = ()  # what one submission is, relative to a working copy
    max_artifact_bytes = 1_000_000
    metric_name = "metric"
    metric_better = "lower"  # or "higher"
    score_definition = ""  # how the higher-is-better score follows from the metric
    per_agent_cpus = 2
    per_agent_memory_mb = 6144
    agent_timeout_sec = 10800.0
    scorer_cpus = 2
    scorer_memory_mb = 8192
    scorer_workers = 2
    grade_timeout_sec = 900
    max_pending_per_agent = 4
    final_candidates = 5
    # Two-stage grading (see the scorer's server.py): score.py also defines final_score,
    # the grade the run is judged on, which agents never see; `score` is their feedback.
    final_stage = False
    final_grade_timeout_sec: float | None = None  # final_score's time limit (default grade_timeout_sec)
    max_unfinished_per_agent: int | None = None  # submissions an agent may have in either stage
    final_prefer_latest = False  # among equal best scores, the final grade takes the latest
    scorer_cpuset: str | None = None  # pin the scorer's container to these host CPUs (Docker cpuset)

    @property
    def directory(self) -> Path:
        return ROOT / "benchmarks" / self.name

    def prepare(self, fetch: bool = True) -> Prepared:
        raise NotImplementedError

    def instruction(self, agents: int, prepared: Prepared) -> str:
        raise NotImplementedError

    def continuation(self) -> str:
        return (self.directory / "continuation.txt").read_text()

    def agent_dockerfile(self) -> Path:
        return self.directory / "environment" / "Dockerfile"

    def scorer_dockerfile(self) -> Path:
        own = self.directory / "environment" / "Dockerfile.scorer"
        return own if own.is_file() else self.agent_dockerfile()

    def scorer_files(self) -> Path:
        return self.directory / "scorer"

    def verifier_env(self, prepared: Prepared) -> dict[str, str]:
        """Recorded in Harbor's job config.json (--ve), so the job shows what it ran."""
        return {"SCORED_ARTIFACT_BENCHMARK": self.name}

    def verifier_timeout(self, agents: int) -> float:
        """The verifier's time limit: queued gradings of the whole team, the final working
        copies, and the re-grades."""
        return self.grade_timeout_sec * (2 + self.final_candidates) + 600

    # Hooks for a benchmark whose task differs from this layer's default one; by default each returns what
    # build_task writes without them.

    def task_toml(self, prepared: Prepared, cpus: int, memory_mb: int, token: str, verifier_timeout: float) -> str:
        return task_toml(self, prepared, cpus, memory_mb, token, verifier_timeout)

    def compose(self, task_dir: Path) -> str:
        return compose(task_dir, self)

    def client_settings(self, prepared: Prepared, agents: int) -> dict:
        """Keys added to the agents' client configuration (client/task.json)."""
        return {}

    def scorer_settings(self, prepared: Prepared, agents: int) -> dict:
        """Keys added to the scorer's configuration (scorer/config.json, outside "benchmark")."""
        return {}

    def extra_files(self, prepared: Prepared, task_dir: Path, agents: int) -> dict[str, tuple[bytes, int]]:
        """More files of the task directory: relative path -> (content, permissions)."""
        return {}

    # Ultra mode (run.py --protocol ultra; ultra.py): one agent and its own helpers, AGENTS of them at most, the
    # root included, who share one $AGENT_ID, so each names its copy (`workcopy NAME`). A benchmark that has it
    # overrides ultra_instruction (its team text with named copies) and lists its continuation's edits.
    ultra_continuation_edits: tuple[tuple[str, str], ...] = ()

    def ultra_instruction(self, agents: int, prepared: Prepared) -> str:
        raise ValueError(f"{self.name} has no ultra mode: its text has no named working copies (`workcopy NAME`)")


def benchmark(name: str) -> Benchmark:
    if name not in BENCHMARKS:
        raise ValueError(f"unknown scored-artifact benchmark {name!r}; known: {', '.join(BENCHMARKS)}")
    module, _, attribute = BENCHMARKS[name].partition(":")
    return getattr(importlib.import_module(module), attribute)


def tasks_root(name: str) -> Path:
    return ROOT / ".cache" / f"{name}-tasks"


def solo_text(team: str, edits: tuple[tuple[str, str | None], ...]) -> str:
    """The team text minus what only makes sense with peers, as antiherding_matched
    does for the protocol. (team line, solo line or None); each must match once."""
    lines = team.split("\n")
    for old, new in edits:
        if lines.count(old) != 1:
            raise ValueError(f"the team text changed; this solo edit no longer applies: {old!r}")
        index = lines.index(old)
        if new is None:
            del lines[index]
        else:
            lines[index] = new
    return "\n".join(lines)


def edited(text: str, edits: tuple[tuple[str, str], ...]) -> str:
    """TEXT with each (old, new) replacement made; each old must appear exactly once."""
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError(f"the text changed; this edit no longer applies: {old!r}")
        text = text.replace(old, new)
    return text


# ----------------------------------------------------------------------------
# The task directory run.py launches


def _quoted(text: str) -> str:
    """A YAML double-quoted string with $ doubled, so Compose does not interpolate it."""
    return json.dumps(text.replace("$", "$$"))


def compose(task_dir: Path, bench: Benchmark) -> str:
    client = task_dir / "client"
    command_mounts = "\n".join(
        f"      - {_quoted(f'{client}/bin/{command}:/usr/local/bin/{command}:ro')}"
        for command in CLIENT_COMMANDS)
    cpuset = f"    cpuset: {json.dumps(bench.scorer_cpuset)}\n" if bench.scorer_cpuset else ""
    return f"""\
services:
  main:
    # Harbor builds explicitly before starting services.
    pull_policy: never
    depends_on:
      scorer:
        condition: service_healthy
    environment:
      SCORED_ARTIFACT_CONFIG: "{CLIENT_MOUNT}/task.json"
    volumes:
      - {_quoted(f'{client}:{CLIENT_MOUNT}:ro')}
{command_mounts}

  scorer:
    build:
      context: ${{CONTEXT_DIR}}
      dockerfile: Dockerfile.scorer
    entrypoint: ["python3", "-I", "/scorer/app/server.py", "--config", "/scorer/app/config.json"]
    cpus: {bench.scorer_cpus}
{cpuset}    mem_limit: "{bench.scorer_memory_mb}m"
    pids_limit: 4096
    # Every 2 s only while the scorer starts, then once a minute (as ale_bench_task.py).
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:{SCORER_PORT}/health', timeout=3)"]
      start_period: 60s
      start_interval: 2s
      interval: 60s
      timeout: 5s
      retries: 3
    volumes:
      - {_quoted(f'{task_dir / "scorer"}:/scorer/app:ro')}
      - {_quoted(f'{task_dir / "data"}:/scorer/data:ro')}
      - ${{HOST_VERIFIER_LOGS_PATH:?run this task through Harbor}}/../{STATE}:/scorer/state
"""


def task_toml(bench: Benchmark, prepared: Prepared, cpus: int, memory_mb: int, token: str,
              verifier_timeout: float) -> str:
    metadata = {f"scored_artifact_{key}": str(value) for key, value in {
        "benchmark": bench.name, "metric": bench.metric_name, "metric_better": bench.metric_better,
        "score": bench.score_definition, "artifact": ",".join(bench.artifact_files),
        **{k: v for k, v in prepared.description.items() if isinstance(v, (str, int, float))},
    }.items()}
    lines = "\n".join(f"{key} = {json.dumps(value)}" for key, value in metadata.items())
    authors = ", ".join("{ " + ", ".join(f"{k} = {json.dumps(v)}" for k, v in a.items()) + " }"
                        for a in bench.authors)
    keywords = ", ".join(json.dumps(k) for k in bench.keywords)
    hosts = ", ".join(json.dumps(h) for h in SETUP_HOSTS)
    return f"""\
schema_version = "1.1"
source = {json.dumps(bench.source)}
artifacts = []

[task]
name = {json.dumps(f"scored-artifact/{bench.name}")}
authors = [{authors}]
keywords = [{keywords}]

[metadata]
{lines}

[verifier]
timeout_sec = {float(verifier_timeout)}

[verifier.env]
SCORER_TOKEN = {json.dumps(token)}

# While agents run they reach only the hosts run.py adds with --allow-agent-host
# (their model API). The environment's own allowlist applies before (the CLI
# install) and after (the verifier).
[agent]
timeout_sec = {float(bench.agent_timeout_sec)}
network_mode = "allowlist"
allowed_hosts = []

[environment]
build_timeout_sec = 600.0
cpus = {cpus}
memory_mb = {memory_mb}
storage_mb = 4096
gpus = 0
network_mode = "allowlist"
allowed_hosts = [{hosts}]
"""


def wrapper(command: str) -> str:
    return f'#!/bin/sh\nSCORECTL_COMMAND={command} exec python3 {CLIENT_MOUNT}/scorectl.py "$@"\n'


ULTRA_ABOUT = "every agent, the root and each of its helpers, works in its own copy (they share one $AGENT_ID)."


def ultra_wrapper(command: str, work_root: str = WORK_ROOT, client_mount: str = CLIENT_MOUNT) -> str:
    """Ultra mode's `workcopy NAME` and `submit` (from inside a copy): scorectl.py as the copy's NAME, its agent
    id (the scorer's log records it), the other commands as they are."""
    if command == "workcopy":
        return ultra.workcopy_script(work_root, f'AGENT_ID="$name" SCORECTL_COMMAND=workcopy exec python3 '
                                                f'{client_mount}/scorectl.py', ULTRA_ABOUT)
    if command == "submit":
        return ultra.submit_script(work_root, f'AGENT_ID="$name" SCORECTL_COMMAND=submit exec python3 '
                                              f'{client_mount}/scorectl.py "$@"', ULTRA_ABOUT)
    return wrapper(command)


@dataclass
class Task:
    """A scored-artifact task as run.py launches it. `files` maps paths relative to the
    task directory to their content; `modes` to their permissions."""

    path: Path
    bench: Benchmark
    agents: int
    files: dict[str, bytes]
    modes: dict[str, int]
    description: dict
    agent_hosts: list[str] = field(default_factory=list)
    ultra: bool = False  # ultra mode's task (build_task's ULTRA)

    @property
    def verifier_env(self) -> dict[str, str]:
        return self.description["verifier_env"]

    @property
    def continuation(self) -> str:
        text = self.bench.continuation()
        return edited(text, self.bench.ultra_continuation_edits) if self.ultra else text

    @property
    def task_toml(self) -> str:
        return self.files["task.toml"].decode()

    @property
    def harbor_args(self) -> list[str]:
        args = []
        for host in self.agent_hosts:
            args += ["--allow-agent-host", host]
        for key, value in self.verifier_env.items():
            args += ["--ve", f"{key}={value}"]
        return args

    def write(self) -> Path:
        if self.path.exists():
            raise ValueError(f"task directory already exists: {self.path}; choose a new --name")
        staging = self.path.with_name(self.path.name + ".partial")
        if staging.exists():
            shutil.rmtree(staging)
        for relative, data in self.files.items():
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(self.modes.get(relative, 0o644))
        staging.rename(self.path)
        return self.path


def build_task(bench: Benchmark, dest_parent: Path, *, agents: int = 1, cli: str | None = None,
               codex_login: bool = False, cpus: int | None = None, memory_mb: int | None = None,
               solution: Path | None = None, fetch: bool = True, ultra: bool = False) -> Task:
    """Everything needed to write the task directory dest_parent/<benchmark>. ULTRA (run.py --protocol ultra):
    one agent and its own helpers, AGENTS of them at most, the root included: the team's resources, the
    benchmark's ultra_instruction, `workcopy NAME` and `submit` from inside a copy (ultra_wrapper)."""
    ultra_mode = ultra
    if ultra_mode and agents < 2:
        raise ValueError("ultra mode needs --agents 2 or more (the most agents at once, the root included)")
    hosts = model_api_hosts(cli, codex_login)
    prepared = bench.prepare(fetch=fetch)
    path = Path(dest_parent).resolve() / bench.name
    total_cpus = cpus if cpus is not None else bench.per_agent_cpus * agents
    total_memory = memory_mb if memory_mb is not None else bench.per_agent_memory_mb * agents
    token = secrets.token_hex(16)
    # Queued gradings of the whole team, the final working copies, and the re-grades.
    verifier_timeout = bench.verifier_timeout(agents)
    client_config = {
        "benchmark": bench.name, "starter": f"{CLIENT_MOUNT}/starter", "work_root": WORK_ROOT,
        "artifact_files": list(bench.artifact_files), "scorer_urls": list(SCORER_URLS),
        "metric_name": bench.metric_name, "score_definition": bench.score_definition,
        "submit_wait_sec": bench.grade_timeout_sec * 3, "finalize_wait_sec": verifier_timeout,
        **bench.client_settings(prepared, agents),
    }
    scorer_config = {
        "benchmark_name": bench.name, "module": "/scorer/app/benchmark/score.py",
        "port": SCORER_PORT, "state_dir": "/scorer/state", "token": token,
        "host_uid": os.getuid(), "host_gid": os.getgid(), "sandbox_uid": SANDBOX_UID,
        "workers": bench.scorer_workers, "grade_timeout_sec": bench.grade_timeout_sec,
        "max_pending_per_agent": bench.max_pending_per_agent, "final_candidates": bench.final_candidates,
        "metric_name": bench.metric_name,
        "artifact": {"files": list(bench.artifact_files), "max_bytes": bench.max_artifact_bytes},
        "public": {"benchmark": bench.name, "metric": bench.metric_name, "metric_better": bench.metric_better,
                   "score": bench.score_definition, "artifact": list(bench.artifact_files)},
        "benchmark": prepared.scorer_config,
        **bench.scorer_settings(prepared, agents),
    }
    if bench.final_stage:
        scorer_config.update(final_stage=True, final_prefer_latest=bench.final_prefer_latest,
                             final_grade_timeout_sec=bench.final_grade_timeout_sec or bench.grade_timeout_sec,
                             max_unfinished_per_agent=bench.max_unfinished_per_agent)
    elif bench.final_prefer_latest:
        scorer_config["final_prefer_latest"] = True
    description = {
        "benchmark": bench.name, "title": bench.title, "source": bench.source, "agents": agents, "cli": cli,
        **({"mode": "ultra", "ultra": {"agents": agents, "processes": 1,
                                        "copies": "workcopy NAME, one per agent (the helpers share one $AGENT_ID)"}}
           if ultra_mode else {}),
        "artifact_files": list(bench.artifact_files), "metric": bench.metric_name,
        "metric_better": bench.metric_better, "score": bench.score_definition,
        "resources": {"cpus": total_cpus, "memory_mb": total_memory, "per_agent_cpus": bench.per_agent_cpus,
                      "per_agent_memory_mb": bench.per_agent_memory_mb, "scorer_cpus": bench.scorer_cpus,
                      "scorer_memory_mb": bench.scorer_memory_mb, "scorer_workers": bench.scorer_workers},
        "network": {"agent_phase": hosts, "setup_and_verifier": list(SETUP_HOSTS)},
        **({"final_stage": True} if bench.final_stage else {}),
        "verifier_env": bench.verifier_env(prepared),
        **prepared.description,
    }
    files: dict[str, bytes] = {
        "instruction.md": (bench.ultra_instruction(agents, prepared) if ultra_mode
                           else bench.instruction(agents, prepared)).encode(),
        "task.toml": bench.task_toml(prepared, total_cpus, total_memory, token, verifier_timeout).encode(),
        "scored-artifact.json": (json.dumps(description, indent=2) + "\n").encode(),
        "environment/Dockerfile": bench.agent_dockerfile().read_bytes(),
        "environment/Dockerfile.scorer": bench.scorer_dockerfile().read_bytes(),
        "environment/docker-compose.yaml": bench.compose(path).encode(),
        "client/task.json": (json.dumps(client_config, indent=2) + "\n").encode(),
        "client/scorectl.py": (LAYER / "client" / "scorectl.py").read_bytes(),
        "scorer/server.py": (LAYER / "scorer" / "server.py").read_bytes(),
        "scorer/sandbox.py": (LAYER / "scorer" / "sandbox.py").read_bytes(),
        # Secrets (the finalize token, hidden inputs): readable by the task's owner only,
        # so not by the scorer's sandbox users. The agents' container never mounts it.
        "scorer/config.json": (json.dumps(scorer_config, indent=2) + "\n").encode(),
        "tests/test.sh": (LAYER / "tests" / "test.sh").read_bytes(),
        "tests/finalize.py": (LAYER / "tests" / "finalize.py").read_bytes(),
    }
    modes = {"scorer/config.json": 0o600, "tests/test.sh": 0o755}
    for command in CLIENT_COMMANDS:
        files[f"client/bin/{command}"] = (ultra_wrapper if ultra_mode else wrapper)(command).encode()
        modes[f"client/bin/{command}"] = 0o755
    for source in sorted(bench.scorer_files().rglob("*")):
        if source.is_file() and "__pycache__" not in source.parts:
            files[f"scorer/benchmark/{source.relative_to(bench.scorer_files())}"] = source.read_bytes()
    for relative, data in prepared.starter.items():
        files[f"client/starter/{relative}"] = data
    for relative, data in prepared.data.items():
        files[f"data/{relative}"] = data
    for relative, (data, mode) in bench.extra_files(prepared, path, agents).items():
        files[relative] = data
        modes[relative] = mode
    if solution is not None:
        if Path(solution).is_dir():
            # A directory holding (some of) the artifact's files, for artifacts of several
            # files; the oracle copies them over its working copy.
            names = [name for name in bench.artifact_files if (Path(solution) / name).is_file()]
            if not names:
                raise ValueError(f"{solution} holds none of the artifact's files")
            for name in names:
                files[f"solution/artifact/{name}"] = (Path(solution) / name).read_bytes()
            copy = 'cp -r /solution/artifact/. "$dest/"\n'
        else:
            if len(bench.artifact_files) != 1:
                raise ValueError("check grades single-file artifacts given as a file; pass a directory "
                                 "holding this benchmark's artifact files")
            files[f"solution/{bench.artifact_files[0]}"] = Path(solution).read_bytes()
            copy = f"cp /solution/{bench.artifact_files[0]} \"$dest/{bench.artifact_files[0]}\"\n"
        files["solution/solve.sh"] = (
            "#!/bin/bash\nset -euo pipefail\nexport AGENT_ID=oracle\n"
            + (f'dest=$(workcopy oracle)\n{copy}cd "$dest"\nsubmit\n' if ultra_mode else
               f"dest=$(workcopy)\n{copy}submit\n")).encode()
        modes["solution/solve.sh"] = 0o755
    return Task(path, bench, agents, files, modes, description, hosts, ultra=ultra_mode)


# ----------------------------------------------------------------------------
# Reading finished jobs (score-over-budget curves, re-grades)


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def job_description(job: Path) -> dict | None:
    """The scored-artifact.json of the task a job (or trial) ran, or None if the job
    is not a scored-artifact job."""
    job = Path(job)
    for config_path in (job / "config.json", job.parent / "config.json"):
        config = _read_json(config_path)
        tasks = config.get("tasks") or ([config["task"]] if isinstance(config.get("task"), dict) else [])
        for task in tasks:
            if task.get("path") and (Path(task["path"]) / "scored-artifact.json").is_file():
                return {**_read_json(Path(task["path"]) / "scored-artifact.json"), "task_dir": task["path"]}
        env = (config.get("verifier") or {}).get("env") or {}
        if env.get("SCORED_ARTIFACT_BENCHMARK"):
            return {"benchmark": env["SCORED_ARTIFACT_BENCHMARK"], "task_dir": None}
    return None


def is_scored_trial(trial: Path) -> bool:
    return (Path(trial) / STATE / "submissions.jsonl").is_file() or (Path(trial) / STATE / "final.json").is_file()


def trial_records(trial: Path) -> list[dict]:
    """The scorer's log of a trial: one record per graded submission, oldest first."""
    log = Path(trial) / STATE / "submissions.jsonl"
    records = []
    if log.is_file():
        for line in log.read_text(errors="replace").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return sorted(records, key=lambda r: r.get("ts") or "")


def trial_final(trial: Path) -> dict | None:
    final = _read_json(Path(trial) / STATE / "final.json")
    return final or None


def trials(job: Path) -> list[Path]:
    job = Path(job)
    if (job / "agent").is_dir():
        return [job]
    return sorted(path for path in job.iterdir() if (path / "agent").is_dir())


# ----------------------------------------------------------------------------
# Grading outside Harbor: the same scorer, in a throwaway container


def scorer_image(task_dir: Path) -> str:
    """Build (or reuse) the scorer image of a written task."""
    dockerfile = Path(task_dir) / "environment" / "Dockerfile.scorer"
    import hashlib
    tag = "ttc-scored-artifact-scorer:" + hashlib.sha256(dockerfile.read_bytes()).hexdigest()[:12]
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True).returncode != 0:
        subprocess.run(["docker", "build", "-q", "-t", tag, "-f", str(dockerfile),
                        str(dockerfile.parent)], check=True, capture_output=True)
    return tag


def grade_offline(task_dir: Path, artifacts: list[Path], *, cpus: int | None = None,
                  name_prefix: str | None = None, cpuset: str | None = None, memory_mb: int | None = None,
                  final_only: bool = False, docker_args: list[str] | tuple[str, ...] = (),
                  options: dict | None = None) -> list[dict]:
    """Grade artifact directories with a task's scorer (server.py grade) in a container
    with no network, as the run's scorer grades them. cpuset pins the container to host
    CPUs; final_only (two-stage benchmarks) skips the agents' feedback; docker_args are
    more `docker run` options (a scorer that grades in containers of its own: the Docker
    socket, its lock directory); options are the submissions' (config submission_options)."""
    name_prefix = name_prefix or os.environ.get("SCORED_ARTIFACT_CONTAINER_PREFIX", "ttc-scored-artifact-grade")
    task_dir = Path(task_dir).resolve()
    image = scorer_image(task_dir)
    described = _read_json(task_dir / "scored-artifact.json")
    cpus = cpus or (described.get("resources") or {}).get("scorer_cpus") or 2
    with tempfile.TemporaryDirectory(prefix="scored-artifact-grade-") as scratch:
        mounts, targets = [], []
        for index, artifact in enumerate(artifacts):
            target = f"/grade/{index}"
            mounts += ["-v", f"{Path(artifact).resolve()}:{target}:ro"]
            targets.append(target)
        name = f"{name_prefix}-{secrets.token_hex(4)}"
        command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--cpus", str(cpus),
                   *(["--cpuset-cpus", cpuset] if cpuset else []),
                   "--memory", f"{memory_mb}m" if memory_mb else "8g", "--pids-limit", "4096",
                   "-v", f"{task_dir / 'scorer'}:/scorer/app:ro", "-v", f"{task_dir / 'data'}:/scorer/data:ro",
                   "-v", f"{scratch}:/scorer/state", *mounts, *docker_args, "--entrypoint", "python3", image,
                   "-I", "/scorer/app/server.py", "--config", "/scorer/app/config.json", "grade", *targets,
                   *(["--final-only"] if final_only else []),
                   *(["--options", json.dumps(options)] if options else [])]
        output = subprocess.run(command, capture_output=True, text=True)
        if output.returncode != 0:
            raise RuntimeError(f"grading failed: {output.stderr[-2000:]}")
    results = [json.loads(line) for line in output.stdout.splitlines() if line.startswith("{")]
    for result, artifact in zip(results, artifacts):
        result["directory"] = str(artifact)
    return results


def best_submissions(job: Path, top: int) -> list[tuple[str, dict, Path]]:
    """(label, log record, files directory) of a job's top distinct valid submissions (never a
    feedback-only grade: "eligible" false)."""
    candidates = []
    for trial in trials(job):
        for record in trial_records(trial):
            if record.get("status") == "done" and record.get("valid") and record.get("eligible") is not False:
                candidates.append((float(record.get("score") or 0), trial, record))
    found, seen = [], set()
    for score, trial, record in sorted(candidates, key=lambda c: -c[0]):
        if record.get("sha256") in seen:
            continue
        seen.add(record.get("sha256"))
        files = trial / STATE / "submissions" / str(record["sid"]) / "files"
        if files.is_dir():
            found.append((f"{trial.parent.name}/{trial.name}#{record['sid']}", record, files))
        if len(found) == top:
            break
    return found


def regrade(jobs: list[Path], top: int = 3, repeats: int = 1, **grade_options) -> list[dict]:
    """Grade each job's top distinct valid submissions again (grade_options: grade_offline's
    cpus, cpuset, memory_mb). Two-stage jobs get their final grade only."""
    rows = []
    for job in jobs:
        described = job_description(job)
        if not described or not described.get("task_dir"):
            raise ValueError(f"{job} is not a scored-artifact job with its task directory")
        programs = best_submissions(job, top)
        scores: dict[str, list[dict]] = {label: [] for label, _, _ in programs}
        options = {**grade_options, **({"final_only": True} if described.get("final_stage") else {})}
        for _ in range(repeats):
            for (label, _, _), result in zip(programs, grade_offline(Path(described["task_dir"]),
                                                                     [p for _, _, p in programs], **options)):
                scores[label].append(result)
        for label, record, _ in programs:
            results = scores[label]
            rows.append({"program": label, "benchmark": described.get("benchmark"),
                         "original_metric": record.get("metric"), "original_score": record.get("score"),
                         "metrics": [r.get("metric") for r in results], "scores": [r.get("score") for r in results],
                         "valid": [r.get("valid") for r in results]})
    return rows


# ----------------------------------------------------------------------------
# Command line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    task = commands.add_parser("task", help="write a task directory to read")
    check = commands.add_parser("check", help="grade one artifact through the scorer and the final "
                                              "verifier, with Harbor's oracle agent")
    grade = commands.add_parser("grade", help="grade artifact files or directories with the scorer only")
    again = commands.add_parser("regrade", help="re-grade the best submissions of jobs")
    for sub in (task, check, grade):
        sub.add_argument("benchmark", choices=tuple(BENCHMARKS))
    task.add_argument("--agents", type=int, default=1)
    task.add_argument("--out", type=Path, help="parent directory (default: .cache/<benchmark>-tasks/manual)")
    check.add_argument("--solution", type=Path, required=True,
                       help="the artifact file to grade, or a directory holding the artifact's files")
    check.add_argument("--name", help="job name (default: <benchmark>-check-<time>)")
    grade.add_argument("artifacts", nargs="+", type=Path, help="artifact files (single-file artifacts) "
                                                               "or directories")
    again.add_argument("jobs", nargs="+", type=Path)
    again.add_argument("--top", type=int, default=3)
    again.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    try:
        if args.command == "regrade":
            for row in regrade(args.jobs, args.top, args.repeats):
                print(json.dumps(row))
            return 0
        bench = benchmark(args.benchmark)
        if args.command == "task":
            out = args.out or tasks_root(bench.name) / "manual"
            print(build_task(bench, out / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
                             agents=args.agents).write())
            return 0
        if args.command == "grade":
            with tempfile.TemporaryDirectory(prefix="scored-artifact-task-") as scratch:
                built = build_task(bench, Path(scratch)).write()
                with tempfile.TemporaryDirectory(prefix="scored-artifact-files-") as staging:
                    directories = []
                    for index, artifact in enumerate(args.artifacts):
                        if artifact.is_dir():
                            directories.append(artifact)
                            continue
                        target = Path(staging) / str(index)
                        target.mkdir()
                        shutil.copyfile(artifact, target / bench.artifact_files[0])
                        directories.append(target)
                    for artifact, result in zip(args.artifacts, grade_offline(built, directories)):
                        print(json.dumps({**result, "directory": str(artifact)}))
            return 0
        name = args.name or f"{bench.name}-check-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        if (ROOT / "jobs" / name).exists():
            raise ValueError(f"job already exists: {ROOT / 'jobs' / name}; choose a new --name")
        built = build_task(bench, tasks_root(bench.name) / name, solution=args.solution)
        built.write()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    cmd = ["harbor", "run", "-p", str(built.path), "--yes", "--env", "docker", "--no-force-build",
           "--jobs-dir", str(ROOT / "jobs"), "--job-name", name, "-k", "1", "-n", "1",
           "--agent", "oracle", *built.harbor_args]
    code = subprocess.call(cmd, cwd=ROOT)
    for trial in trials(ROOT / "jobs" / name) if (ROOT / "jobs" / name).is_dir() else []:
        result = _read_json(trial / "result.json")
        print(json.dumps({"trial": str(trial), "final": trial_final(trial),
                          "reward": ((result.get("verifier_result") or {}).get("rewards") or {}),
                          "submissions": len(trial_records(trial))}))
    return code


if __name__ == "__main__":
    sys.exit(main())
