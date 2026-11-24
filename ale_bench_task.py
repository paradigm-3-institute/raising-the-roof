#!/usr/bin/env python3
"""ALE-Bench (Sakana AI): AtCoder Heuristic Contest problems with ALE-Bench's own judge.

`run.py ale-bench --problem ID|lite|all ...` runs problems of ALE-Bench (Imajuku et al.,
NeurIPS 2025 Datasets and Benchmarks; github.com/SakanaAI/ALE-Bench, Apache-2.0; data:
huggingface.co/datasets/SakanaAI/ALE-Bench, CC BY-ND 4.0, provided by AtCoder) with every
CLI, protocol, team size, token budget and job option of the other tasks. Each job writes
a dataset of one task per problem to .cache/ale-bench-tasks/<job>/<problem>/:

- the judge (benchmarks/ale-bench/scorer/judge.py), its own container next to the agents'
  with the host's Docker socket: ALE-Bench's `ale_bench` package at the pinned commit
  grades every submission with a Session's public_eval (the agents' feedback), the run's
  final candidate with private_eval in the background (for the curves, never shown), and
  at the end the best public-ACCEPTED submission with private_eval: raw score, rank and
  AtCoder performance, the reward. Every container ALE-Bench starts is pinned to one of
  the judge's CPUs, one at a time machine-wide; the queue, the log and the HTTP API are
  the scored-artifact layer's;
- the agents' image (ALE-Bench's execution image for the language) and their commands:
  `workcopy`, `submit`, `scores`, `fetch-submission`, and `ale-local` (the problem's
  generator, tester and visualizer, and a local case evaluation like the judge's);
- the verifier: the scored-artifact layer's (each agent's final copy is submitted, then
  the judge's final grade).

The dataset is fetched at a pinned revision and never vendored; the agents get the
problem's public material only (statements, figures, examples, the contest's tools and
the inputs of seeds 0-99), never the private seeds, standings or per-case results.

    python ale_bench_task.py fetch [--problems lite]       # clone the pinned commit, fetch the dataset
    python ale_bench_task.py images [--language cpp20]     # pull/build the judge, tools and scorer images
    python ale_bench_task.py tools ahc039                  # build a problem's tools (upstream's cargo build)
    python ale_bench_task.py problems                      # the 40 problems and their limits
    python ale_bench_task.py task ahc039 --agents 2        # write a task to read
    python ale_bench_task.py grade ahc039 Main.cpp --private   # the judge only, no agent
    python ale_bench_task.py direct ahc039 Main.cpp        # ale_bench.start() + private_eval, unmodified
    python ale_bench_task.py check ahc039 --solution Main.cpp  # through the whole task (oracle agent)
    python ale_bench_task.py regrade jobs/JOB              # the final candidate again, private cases
    python ale_bench_task.py summary jobs/A jobs/B         # performances, average and rating
    python ale_bench_task.py submissions jobs/JOB          # every submission with the team's tokens
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import network_allowlist  # noqa: E402
from scored_artifact import edited, solo_text  # noqa: E402
import ultra  # noqa: E402

BENCH = ROOT / "benchmarks" / "ale-bench"
LAYER = ROOT / "benchmarks" / "scored-artifact"
CACHE = ROOT / ".cache" / "ale-bench"
TASKS = ROOT / ".cache" / "ale-bench-tasks"
CLONE = CACHE / "ALE-Bench"
REPO_URL = "https://github.com/SakanaAI/ALE-Bench"
# 2026-09-11, "Upgrade dependencies and add models (#42)".
COMMIT = "3da9b12fb5d112dabb3af693d1a42031c95142bc"
DATASET = "SakanaAI/ALE-Bench"
# 2025-06-17: the dataset's revision.
REVISION = "0f426173b4e4e73b09b2b3631ae0490f66b75f99"
DATA = CACHE / "data" / REVISION
TOOLS = CACHE / "tools" / REVISION
REFERENCE = BENCH / "dataset.json"  # sha256 of every dataset file and each problem's facts
JUDGE_VERSION = "202301"  # ALE-Bench's default judge (AtCoder's 2023 language update)
# ALE-Bench's execution images for the 202301 judge (Docker Hub yimjk/ale-bench, linux/amd64).
BASE_IMAGES = {
    "cpp17": "sha256:40a2d4be9701fd80f2f2bc05c5841167062ada9fb883f9148f6d6057b0d247f0",
    "cpp20": "sha256:946af1b209a84160594e43262b5157aec933938c99e2585b53042cac9bc3f43c",
    "cpp23": "sha256:bfd5debe10fc3a992a4a1b1283d67eeba47ccd2f9a2159bb88e445a949979dfb",
    "python": "sha256:95e0416e23056539775f5a68afbf578168f89167accd1856ed5f7975c9d5432c",
    "rust": "sha256:c99fb41f4f370b82a5efd1afc5e732b1cabb4b87611d7ad018787d41d25e1d19",
}
DEFAULT_LANGUAGE = "cpp20"  # the leaderboard's language
# The file the agents write, per language (the judge's own path is the profile's).
SUBMISSION_NAMES = {"cpp17": "Main.cpp", "cpp20": "Main.cpp", "cpp23": "Main.cpp", "python": "Main.py",
                    "rust": "main.rs"}
# ale_bench.constants: the image ALE-Bench builds and runs the problems' Rust tools in.
TOOL_IMAGE = "rust:1.79.0-buster"
TOOL_IMAGE_DIGEST = "sha256:3ccb54f58e7853d290648010f0b6cc3432590d304f419485ce022e2b18cd1b46"
SCORER_DOCKERFILE = BENCH / "environment" / "Dockerfile.scorer"
SCORER_REQUIREMENTS = BENCH / "environment" / "scorer-requirements.txt"

SCORER_PORT = 18090
SCORER_URLS = (f"http://127.0.0.1:{SCORER_PORT}", f"http://scorer:{SCORER_PORT}")
CLIENT_MOUNT = "/opt/ale-bench/client"
WORK_ROOT = "/work"
CLIENT_COMMANDS = ("submit", "workcopy", "scores", "fetch-submission")
STATE = "ale-bench"  # <trial>/ale-bench/: the judge's log, private lane, final grade
LOCK_DIR = Path(os.environ.get("ALE_BENCH_LOCK_DIR", "/tmp/ttc-ale-bench-locks"))
# Hosts reachable outside the agent phase: none (the agents' image has what the CLI's
# setup asks apt for; the judge and the verifier need no network).
SETUP_HOSTS: tuple[str, ...] = ()

# Each agent: 2 CPUs and 6 GiB (a team of k: k times both, in one container).
PER_AGENT_CPUS = 2
PER_AGENT_MEMORY_MB = 6144
JUDGE_CPUS_DEFAULT = 2  # one ALE-Bench container per judge CPU at a time
SCORER_CPUS = 1  # the judge's own process (its cases run in ALE-Bench's containers)
SCORER_MEMORY_MB = 4096
MAX_PENDING_PER_AGENT = 3
FINAL_CANDIDATES = 3
AGENT_TIMEOUT_SEC = 10800.0
CONTAINER_OVERHEAD_SEC = 2.0  # starting and removing a case's containers, roughly
# The verifier's time limit, at least: grading_seconds assumes an idle judge, but with 8 team runs sharing 6 judge
# CPUs on a fully loaded host a final re-grade can take over 4 h, and a verifier timeout throws the run's grade away.
VERIFIER_MIN_SEC = 86400.0
VARIANT_SEEDS = tuple(range(50))  # the outputs variant's fixed inputs: seeds 0-49
SOLUTION_ALLOW = ("official", "outputs")


# ----------------------------------------------------------------------------
# The pinned upstream


def _git(clone: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(clone), *args], capture_output=True, text=text)


def fetch_clone(clone: Path = CLONE) -> Path:
    """Clone ALE-Bench once and make sure the pinned commit is there (files are read from
    git at COMMIT, whatever the working tree holds)."""
    clone = Path(clone)
    if not (clone / ".git").exists():
        clone.parent.mkdir(parents=True, exist_ok=True)
        done = subprocess.run(["git", "clone", "-q", "--no-checkout", REPO_URL, str(clone)], capture_output=True,
                              text=True)
        if done.returncode != 0:
            raise ValueError(f"could not clone {REPO_URL}: {done.stderr.strip()}")
    if _git(clone, "cat-file", "-e", f"{COMMIT}^{{commit}}").returncode != 0:
        _git(clone, "fetch", "-q", "origin")
        if _git(clone, "cat-file", "-e", f"{COMMIT}^{{commit}}").returncode != 0:
            raise ValueError(f"{clone} does not have the pinned commit {COMMIT}")
    return clone


def upstream_file(path: str, clone: Path = CLONE) -> bytes:
    shown = _git(clone, "show", f"{COMMIT}:{path}", text=False)
    if shown.returncode != 0:
        raise ValueError(f"cannot read {path} at {COMMIT} in {clone}; run `python ale_bench_task.py fetch`")
    return shown.stdout


def judge_profiles(clone: Path = CLONE) -> dict[str, dict[str, str]]:
    """ALE-Bench's compile and run commands for JUDGE_VERSION (judge_command_profiles.py,
    a plain data module, executed in a namespace of its own)."""
    import types

    module = types.ModuleType("ale_bench_judge_command_profiles")
    sys.modules[module.__name__] = module  # dataclasses look their module up
    try:
        exec(compile(upstream_file("src/ale_bench/judge_command_profiles.py", clone), "judge_command_profiles.py",
                     "exec"), module.__dict__)
    finally:
        del sys.modules[module.__name__]
    profiles = module.JUDGE_LANGUAGE_PROFILES[JUDGE_VERSION]
    return {language: {"compile_command": profile.compile_command, "run_command": profile.run_command,
                       "submission_file": profile.submission_file_path, "object_file": profile.object_file_path}
            for language, profile in profiles.items()}


def language_texts(clone: Path = CLONE) -> tuple[dict[str, str], dict[str, str]]:
    """The language names and library lists ale_bench_eval's prompts give models
    (prompts/texts.py), for JUDGE_VERSION."""
    tree = ast.parse(upstream_file("src/ale_bench_eval/prompts/texts.py", clone))
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id in ("CODE_LANGUAGE_STRING", "CODE_LANGUAGE_LIBRARIES"):
                    found[target.id] = node.value
    names = ast.literal_eval(found["CODE_LANGUAGE_STRING"])[JUDGE_VERSION]
    libraries_node = found["CODE_LANGUAGE_LIBRARIES"]
    libraries = {}
    for key, value in zip(libraries_node.keys, libraries_node.values):
        if ast.literal_eval(key) == JUDGE_VERSION:
            libraries = ast.literal_eval(value)
    return names, libraries


# ----------------------------------------------------------------------------
# The dataset (fetched at REVISION, never vendored)


def reference() -> dict:
    return json.loads(REFERENCE.read_text())


def problem_ids(which: str = "all") -> list[str]:
    ref = reference()
    return list(ref["lite"] if which == "lite" else ref["all"])


def check_problems(spec: str) -> tuple[list[str], bool]:
    """The problems of a --problem value (an id, lite, all, or ids with commas) and
    whether the spec names the lite set."""
    spec = (spec or "").strip()
    if spec in ("lite", "all"):
        return problem_ids(spec), spec == "lite"
    known = problem_ids("all")
    chosen = [part.strip() for part in spec.split(",") if part.strip()]
    unknown = [pid for pid in chosen if pid not in known]
    if not chosen or unknown:
        raise ValueError(f"unknown ALE-Bench problem {', '.join(unknown) or spec!r}; "
                         "use an id such as ahc039, lite, all, or ids separated by commas "
                         "(python ale_bench_task.py problems lists them)")
    return list(dict.fromkeys(chosen)), False


def hf_url(name: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}"


def fetch_dataset(names=None, data: Path = DATA) -> Path:
    """Download the dataset's files at REVISION (only those missing), each checked
    against its pinned sha256."""
    files = reference()["files"]
    names = list(files) if names is None else list(names)
    data.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = data / name
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == files[name]:
            continue
        partial = path.with_name(path.name + ".partial")
        done = subprocess.run(["curl", "-sfL", "--retry", "3", "-o", str(partial), hf_url(name)],
                              capture_output=True, text=True)
        if done.returncode != 0:
            raise ValueError(f"could not download {hf_url(name)}: {done.stderr.strip()}")
        if hashlib.sha256(partial.read_bytes()).hexdigest() != files[name]:
            partial.unlink()
            raise ValueError(f"{name} at revision {REVISION} does not match its pinned sha256")
        partial.rename(path)
    return data


def _zip_member(archive: zipfile.ZipFile, pid: str, name: str) -> bytes:
    for candidate in (f"{pid}/{name}", f"{pid}//{name}"):  # some archives double the slash
        try:
            return archive.read(candidate)
        except KeyError:
            continue
    raise KeyError(name)


def _zip_tree(archive: zipfile.ZipFile, pid: str) -> dict[str, bytes]:
    """Every file of a problem's zip, by its path below the problem's directory."""
    files = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        relative = info.filename.split("/", 1)[1].lstrip("/") if "/" in info.filename else info.filename
        files[relative] = archive.read(info)
    return files


def problem_facts(pid: str, data: Path = DATA) -> dict:
    """A problem's facts from its zip (data.json; counts only, never the seeds)."""
    with zipfile.ZipFile(data / f"{pid}.zip") as archive:
        meta = json.loads(_zip_member(archive, pid, "data.json"))
        standings = _zip_member(archive, pid, "standings_scores.csv").decode().strip().splitlines()
    metadata, constraints, seeds = meta["metadata"], meta["constraints"], meta["seeds"]
    start = datetime.fromisoformat(metadata["start_at"])
    end = datetime.fromisoformat(metadata["end_at"])
    return {"title": metadata["title"], "contest_url": metadata["contest_url"],
            "problem_type": metadata["problem_type"], "score_type": metadata["score_type"],
            "relative_score_type": metadata.get("relative_score_type"),
            "time_limit": constraints["time_limit"], "memory_limit": constraints["memory_limit"],
            "n_public": len(seeds["public"]), "n_public_lite": len(seeds["public_lite"]),
            "public_seeds": seeds["public"], "public_seeds_lite": seeds["public_lite"],
            "n_private": len(seeds["private"]), "n_private_lite": len(seeds["private_lite"]),
            "contest_hours": round((end - start).total_seconds() / 3600, 2), "start_at": metadata["start_at"],
            "end_at": metadata["end_at"], "standings_entries": len(standings) - 1}


def make_reference(data: Path = DATA) -> dict:
    """benchmarks/ale-bench/dataset.json from a full download: hashes and facts only."""
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(data.iterdir()) if path.is_file() and not path.name.endswith(".partial")}
    all_ids = (data / "problem_ids.txt").read_text().split()
    lite = (data / "problem_ids_lite.txt").read_text().split()
    return {"dataset": DATASET, "revision": REVISION, "license": "CC BY-ND 4.0", "files": files,
            "all": all_ids, "lite": lite, "problems": {pid: problem_facts(pid, data) for pid in all_ids}}


# ----------------------------------------------------------------------------
# Images


def _docker(*args: str, check: bool = False, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=text, check=check)


def image_digests(image: str) -> list[str]:
    done = _docker("image", "inspect", image, "--format", "{{json .RepoDigests}}")
    if done.returncode != 0:
        return []
    return [entry.split("@", 1)[1] for entry in json.loads(done.stdout or "[]") or [] if "@" in entry]


def pull_pinned(image: str, digest: str, pull: bool = True) -> None:
    """The tag ALE-Bench uses, holding the pinned digest."""
    if digest in image_digests(image):
        return
    if not pull:
        raise ValueError(f"{image} is missing or not {digest}; run `python ale_bench_task.py images`")
    done = _docker("pull", "-q", image)
    if done.returncode != 0 or digest not in image_digests(image):
        raise ValueError(f"could not pull {image} at {digest}: {done.stderr.strip() or image_digests(image)}")


def base_image(language: str) -> str:
    return f"yimjk/ale-bench:{language}-{JUDGE_VERSION}"


def judge_image(language: str) -> str:
    """ALE-Bench's local image name (code_language.get_docker_image_name)."""
    return f"ale-bench:{language}-{JUDGE_VERSION}"


def scorer_tag(clone: Path = CLONE) -> str:
    digest = hashlib.sha256(SCORER_DOCKERFILE.read_bytes() + SCORER_REQUIREMENTS.read_bytes()
                            + COMMIT.encode()).hexdigest()[:12]
    return f"ttc-ale-bench-scorer:{digest}"


def image_exists(image: str) -> bool:
    return _docker("image", "inspect", image).returncode == 0


def ensure_images(language: str = DEFAULT_LANGUAGE, build: bool = True, clone: Path = CLONE) -> dict:
    """The language's execution image (pulled at its pinned digest) and ALE-Bench's local
    image built from it with upstream's Dockerfile for this host's user (as its
    scripts/docker_build_202301.sh does), the tools image, and the judge's image."""
    if language not in BASE_IMAGES:
        raise ValueError(f"unknown language {language!r}; choose from {', '.join(BASE_IMAGES)}")
    pull_pinned(base_image(language), BASE_IMAGES[language], pull=build)
    pull_pinned(TOOL_IMAGE, TOOL_IMAGE_DIGEST, pull=build)
    uid, gid = os.getuid(), os.getgid()
    label = f"{BASE_IMAGES[language]}:{uid}:{gid}"
    judge = judge_image(language)
    have = _docker("image", "inspect", judge, "--format", '{{index .Config.Labels "ttc.ale-bench.base"}}')
    if have.returncode != 0 or have.stdout.strip() != label:
        if not build:
            raise ValueError(f"{judge} is missing; run `python ale_bench_task.py images --language {language}`")
        fetch_clone(clone)
        with tempfile.TemporaryDirectory(prefix="ale-bench-image-") as context:
            dockerfile = Path(context) / "Dockerfile"
            dockerfile.write_bytes(upstream_file(f"dockerfiles/Dockerfile_{language}_{JUDGE_VERSION}", clone))
            done = _docker("build", "-q", "-t", judge, "-f", str(dockerfile), "--build-arg", f"UID={uid}",
                           "--build-arg", f"GID={gid}", "--label", f"ttc.ale-bench.base={label}", context)
            if done.returncode != 0:
                raise ValueError(f"could not build {judge}: {done.stderr[-2000:]}")
    scorer = scorer_tag(clone)
    if not image_exists(scorer):
        if not build:
            raise ValueError(f"{scorer} is missing; run `python ale_bench_task.py images`")
        fetch_clone(clone)
        with tempfile.TemporaryDirectory(prefix="ale-bench-scorer-") as context:
            shutil.copyfile(SCORER_DOCKERFILE, Path(context) / "Dockerfile")
            shutil.copyfile(SCORER_REQUIREMENTS, Path(context) / "scorer-requirements.txt")
            archive = _git(clone, "archive", "--format=tar", COMMIT, "src", "LICENSE", text=False)
            if archive.returncode != 0:
                raise ValueError(f"git archive failed: {archive.stderr}")
            (Path(context) / "ALE-Bench").mkdir()
            subprocess.run(["tar", "-x", "-C", str(Path(context) / "ALE-Bench")], input=archive.stdout, check=True)
            done = _docker("build", "-q", "-t", scorer, context)
            if done.returncode != 0:
                raise ValueError(f"could not build {scorer}: {done.stderr[-2000:]}")
    return {"base": f"{base_image(language)}@{BASE_IMAGES[language]}", "judge": judge,
            "tools": f"{TOOL_IMAGE}@{TOOL_IMAGE_DIGEST}", "scorer": scorer}


# ----------------------------------------------------------------------------
# A problem's tools and the agents' starter files


def sources_digest(tools_dir: Path) -> str:
    """sha256 over tools/src (paths relative to tools/), as judge.install_tools checks."""
    digest = hashlib.sha256()
    for path in sorted(p for p in (tools_dir / "src").rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(tools_dir)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def ensure_tools(pid: str, build: bool = True, data: Path = DATA) -> Path:
    """TOOLS/<pid>: gen, tester and vis built from the problem's tools/ as upstream's
    data.build_rust_tools builds them (`cargo build --release` in rust:1.79.0-buster,
    RUSTFLAGS=-Awarnings, as this host's user; cargo downloads the tools' crates), their
    manifest, and the inputs of seeds 0-99 from gen (the contest's in/), all made in
    containers."""
    out = TOOLS / pid
    if (out / "manifest.json").is_file():
        return out
    if not build:
        raise ValueError(f"the tools of {pid} are not built; run `python ale_bench_task.py tools {pid}`")
    fetch_dataset([f"{pid}.zip"], data)
    pull_pinned(TOOL_IMAGE, TOOL_IMAGE_DIGEST)
    staging = out.with_name(out.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "src-tree").mkdir(parents=True)
    with zipfile.ZipFile(data / f"{pid}.zip") as archive:
        for relative, content in _zip_tree(archive, pid).items():
            if relative.startswith("tools/"):
                target = staging / "src-tree" / relative[len("tools/"):]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    tree = staging / "src-tree"
    user = f"{os.getuid()}:{os.getgid()}"
    done = _docker("run", "--rm", "--name", f"adapter-ale-bench-tools-{secrets.token_hex(3)}", "--user", user,
                   "--group-add", str(os.getgid()), "-e", "RUSTFLAGS=-Awarnings", "--cpus", "2",
                   "-v", f"{tree}:/workdir", "-w", "/workdir", TOOL_IMAGE, "cargo", "build", "--release")
    if done.returncode != 0:
        raise ValueError(f"building {pid}'s tools failed: {done.stderr[-3000:]}")
    binaries = []
    for tool in ("gen", "tester", "vis"):
        if (tree / "src" / "bin" / f"{tool}.rs").is_file():
            built = tree / "target" / "release" / tool
            if not built.is_file() or built.stat().st_size == 0:
                raise ValueError(f"{pid}'s {tool} was not built")
            shutil.copyfile(built, staging / tool)
            (staging / tool).chmod(0o755)
            binaries.append(tool)
    shutil.rmtree(tree / "target")
    # The contest's in/ (seeds 0-99), from gen in a container without network.
    (staging / "in").mkdir()
    (staging / "seeds.txt").write_text("".join(f"{seed}\n" for seed in range(100)))
    done = _docker("run", "--rm", "--name", f"adapter-ale-bench-gen-{secrets.token_hex(3)}", "--network", "none",
                   "--user", user, "--cpus", "1", "-v", f"{staging}:/work", "-w", "/work", TOOL_IMAGE,
                   "/work/gen", "/work/seeds.txt")
    if done.returncode != 0:
        raise ValueError(f"generating {pid}'s inputs failed: {done.stderr[-2000:]}")
    manifest = {"problem_id": pid, "revision": REVISION, "image": f"{TOOL_IMAGE}@{TOOL_IMAGE_DIGEST}",
                "sources_sha256": sources_digest(tree), "binaries": binaries,
                "sha256": {tool: hashlib.sha256((staging / tool).read_bytes()).hexdigest() for tool in binaries},
                "inputs": len(list((staging / "in").glob("*.txt")))}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.rmtree(out, ignore_errors=True)
    staging.rename(out)
    return out


def default_judge_cpus(count: int = JUDGE_CPUS_DEFAULT) -> list[int]:
    """The first logical CPU of each of the host's last `count` physical cores (no two
    judge CPUs share a core); ALE_BENCH_JUDGE_CPUS (e.g. "12,14") overrides."""
    if os.environ.get("ALE_BENCH_JUDGE_CPUS"):
        return parse_cpus(os.environ["ALE_BENCH_JUDGE_CPUS"])
    cores: dict[tuple[str, str], int] = {}
    for cpu_dir in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        try:
            key = ((cpu_dir / "topology/physical_package_id").read_text().strip(),
                   (cpu_dir / "topology/core_id").read_text().strip())
        except OSError:
            continue
        cores.setdefault(key, int(cpu_dir.name[3:]))
    firsts = sorted(cores.values())
    if not firsts:
        firsts = list(range(os.cpu_count() or 1))
    return firsts[-count:]


def parse_cpus(text: str) -> list[int]:
    cpus = []
    for part in str(text).split(","):
        first, _, last = part.strip().partition("-")
        if first:
            cpus += list(range(int(first), int(last or first) + 1))
    if not cpus:
        raise ValueError("no judge CPUs given")
    return sorted(dict.fromkeys(cpus))


# ----------------------------------------------------------------------------
# Task text

TEAM_INSTRUCTION = """\
You are a world-class algorithm engineer, and you are very good at programming. Now, you
are participating in a programming contest. You are asked to solve a heuristic problem,
known as an NP-hard problem.

The problem is {title} ({problem_id}, AtCoder: {contest_url}), as ALE-Bench (Sakana AI)
poses it. Run `workcopy` once: it creates your own working copy at /work/$AGENT_ID and
prints its path. Work only in your copy. {n} agents share this container, and each has
its own copy; do not edit another agent's copy.

In your copy:
- problem/statement.md: the problem statement. Read it first, in full. (statement_ja.md
  is the Japanese original; problem/images/ holds its figures.)
- Execution time limit: {time_limit} sec / Memory limit: {memory_limit} MiB, per test case.
- tools/: the contest's local tools (README.md, their Rust sources in tools/src) built as
  tools/bin/gen (input generator), tools/bin/tester ({tester_role}) and tools/bin/vis
  (visualizer); in/0000.txt ... in/0099.txt are the inputs for seeds 0-99.
- problem.json: these facts and the judge's commands.

{task_section}
- `ale-local test` compiles and runs a program on cases of your copy the way the judge
  does (`ale-local --help`; `ale-local gen` makes more inputs). Its times are measured on
  your own CPU share in this container, not on the judge's CPUs.
- `scores` lists every graded submission of the team, best public score first.
  `fetch-submission SID DIR` (or `fetch-submission best DIR`) writes a submission's files
  into DIR, so you can adopt or build on a teammate's accepted submission.

The container is offline (you cannot install packages). The {n} of you share its {cpus}
CPUs: keep to {share} CPU-heavy processes at a time.
"""

OFFICIAL_SECTION = """\
Implement your solution in {language_name}, as {submission_name} in your copy. You can use
external libraries as follows:
{libraries}
The judge compiles it with `{compile_command}`
and runs it as `{run_command}`{reactive_note}.

Scoring:
- `submit` sends your copy's {submission_name} to the judge, which runs ALE-Bench's public
  evaluation: your program on the {n_public} public cases (seeds {public_seeds}, the inputs
  in/{public_files}), each on one of the judge's CPUs (shared with other work on this
  machine, so times vary: leave a margin) under the time and memory limits (execution
  time is the larger of CPU and wall time), scored by the problem's
  tester. It prints the overall judge result (ACCEPTED only if every case is), the total
  absolute score and each case's score, or the first failing case. `submit FILE` grades
  another file. {better_sentence}
- The result of the run is decided once, at the end: the submission (of any agent) with
  the best public total among those ACCEPTED on every public case (the earliest of equal
  ones) is run on the contest's {n_private} hidden system-test cases, and its total there{relative_note}
  is ranked among the contest's human participants and converted into an AtCoder
  performance. Your copy's {submission_name} is also submitted when the run ends.
  Submit early, and after every improvement."""

OUTPUTS_SECTION = """\
This is a non-official variant of the task, graded on output files instead of a program.
Your submission is out/0000.txt ... out/{last_output} in your copy: an output for each of
the {n_variant} fixed inputs in/0000.txt ... in/{last_input} (seeds 0-{last_seed}), made by any
means and with no time limit.

Scoring:
- `submit` sends your copy's out/ to the judge, which scores every output against its
  input with the problem's tester (tools/bin/tester). A submission is valid only if every
  output is accepted; its total absolute score is the sum over the {n_variant} cases.
  {better_sentence} Its score is {variant_score}.
- The result of the run is the best valid submission of any agent; your copy's out/ is
  also submitted when the run ends. Submit early, and after every improvement."""

SOLO_EDITS = (
    ("prints its path. Work only in your copy. {n} agents share this container, and each has",
     "prints its path. Work only in your copy."),
    ("its own copy; do not edit another agent's copy.", None),
    ("- `scores` lists every graded submission of the team, best public score first.",
     "- `scores` lists your graded submissions, best public score first."),
    ("  into DIR, so you can adopt or build on a teammate's accepted submission.", "  into DIR."),
    ("The container is offline (you cannot install packages). The {n} of you share its {cpus}",
     "The container is offline (you cannot install packages). You have its {cpus} CPUs."),
    ("CPUs: keep to {share} CPU-heavy processes at a time.", None),
)
OFFICIAL_SOLO_EDITS = (
    ("- The result of the run is decided once, at the end: the submission (of any agent) with",
     "- The result of the run is decided once, at the end: your submission with"),
)
OUTPUTS_SOLO_EDITS = (
    ("- The result of the run is the best valid submission of any agent; your copy's out/ is",
     "- The result of the run is your best valid submission; your copy's out/ is"),
)
# Ultra mode (run.py --protocol ultra; ultra.py): one agent and its own helpers, who share one $AGENT_ID, so each
# names its copy (`workcopy NAME`) and submits from inside it; every copy is submitted at the end, and the best
# counts, as for a team; the CPU sentence gives the team's CPUs. The team's text with these edits; the cap line
# comes before it (agent.py) and the resources line says it counts the helpers.
ULTRA_EDITS = (
    ("poses it. Run `workcopy` once: it creates your own working copy at /work/$AGENT_ID and",
     "poses it. Every agent, you and each helper, works in its own copy: `workcopy NAME` creates"),
    ("prints its path. Work only in your copy. {n} agents share this container, and each has",
     "the working copy /work/NAME once and prints its path. Give each agent its own NAME. Work"),
    ("its own copy; do not edit another agent's copy.", "only in your copy; do not edit another agent's copy."),
    ("- `scores` lists every graded submission of the team, best public score first.",
     "- `scores` lists every graded submission, yours and your helpers', best public score first."),
    ("  into DIR, so you can adopt or build on a teammate's accepted submission.",
     "  into DIR, so you can adopt or build on another agent's accepted submission."),
    ("The container is offline (you cannot install packages). The {n} of you share its {cpus}",
     "The container is offline (you cannot install packages). You and your helpers share its {cpus}"),
    ("CPUs: keep to {share} CPU-heavy processes at a time.", "CPUs: keep to {share} CPU-heavy processes per agent at a time."),
)
OFFICIAL_ULTRA_EDITS = (
    ("- `submit` sends your copy's {submission_name} to the judge, which runs ALE-Bench's public",
     "- `submit`, run from inside your copy, sends its {submission_name} to the judge, which runs ALE-Bench's public"),
    ("  performance. Your copy's {submission_name} is also submitted when the run ends.",
     "  performance. Every copy's {submission_name} is also submitted when the run ends."),
)
OUTPUTS_ULTRA_EDITS = (
    ("- `submit` sends your copy's out/ to the judge, which scores every output against its",
     "- `submit`, run from inside your copy, sends its out/ to the judge, which scores every output against its"),
    ("- The result of the run is the best valid submission of any agent; your copy's out/ is",
     "- The result of the run is the best valid submission of any agent; every copy's out/ is"),
)
ULTRA_ABOUT = "every agent, the root and each of its helpers, works in its own copy (they share one $AGENT_ID)."
# The continuation's mention of `submit`, in ultra mode (it must appear once), as FrontierSWE's.
ULTRA_CONTINUATION_EDITS = (
    ("and run `submit` to get a fresh public score.", "and run `submit` from inside your copy to get a fresh public "
                                                       "score."),
)


def seeds_text(seeds: list[int]) -> str:
    seeds = sorted(seeds)
    if seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}-{seeds[-1]}"
    return ",".join(map(str, seeds))


def instruction(facts: dict, language: str, lite: bool, variant: str, agents: int, cpus: int,
                profiles: dict, names: dict, libraries: dict, ultra_mode: bool = False) -> str:
    profile = profiles[language]
    public = facts["public_seeds_lite"] if lite else facts["public_seeds"]
    maximize = facts["score_type"] == "maximize"
    better = ("Higher total scores are better." if maximize else "Lower total scores are better.")
    outputs_section, official_section = OUTPUTS_SECTION, OFFICIAL_SECTION
    if ultra_mode:
        outputs_section = solo_text(OUTPUTS_SECTION, OUTPUTS_ULTRA_EDITS)
        official_section = solo_text(OFFICIAL_SECTION, OFFICIAL_ULTRA_EDITS)
    if variant == "outputs":
        n_variant = len(VARIANT_SEEDS)
        section = outputs_section.format(
            last_output=f"{n_variant - 1:04d}.txt", n_variant=n_variant, last_input=f"{n_variant - 1:04d}.txt",
            last_seed=n_variant - 1, better_sentence=better,
            variant_score=("the total" if maximize else f"{n_variant} x 10^9 / total (higher is better)"))
    else:
        section = official_section.format(
            language_name=names[language], submission_name=SUBMISSION_NAMES[language],
            libraries=libraries.get(language, "- No additional libraries are available."),
            compile_command=profile["compile_command"], run_command=profile["run_command"],
            reactive_note=(" under the tester, which plays the judge's side of the interaction"
                           if facts["problem_type"] == "reactive" else ""),
            n_public=len(public), public_seeds=seeds_text(public),
            public_files=(f"{min(public):04d}.txt ... in/{max(public):04d}.txt" if len(public) > 1
                          else f"{public[0]:04d}.txt"),
            better_sentence=better,
            relative_note=(" (each case's score made\n  relative to the contestants' scores on it, as in the contest's ranking)"
                           if facts.get("relative_score_type") else ""),
            n_private=facts["n_private_lite"] if lite else facts["n_private"])
    share = max(1, cpus // max(1, agents))
    team = TEAM_INSTRUCTION.replace("{task_section}", section)
    if ultra_mode:
        team = solo_text(team, ULTRA_EDITS)
    elif agents == 1:
        team = solo_text(team, SOLO_EDITS)
        edits = OUTPUTS_SOLO_EDITS if variant == "outputs" else OFFICIAL_SOLO_EDITS
        team = solo_text(team, edits)
    tester_role = ("scores an output; for this interactive problem it runs your program and plays the judge"
                   if facts["problem_type"] == "reactive" else "scores an output against its input")
    text = (team.replace("{n}", str(agents)).replace("{cpus}", str(cpus)).replace("{share}", str(share))
            .replace("{title}", facts["title"]).replace("{problem_id}", facts["problem_id"])
            .replace("{contest_url}", facts["contest_url"]).replace("{time_limit}", f"{facts['time_limit']:g}")
            .replace("{memory_limit}", str(facts["memory_limit"] // 1024 // 1024))
            .replace("{tester_role}", tester_role))
    return text


# ----------------------------------------------------------------------------
# Tasks and jobs


def _quoted(text: str) -> str:
    """A YAML double-quoted string with $ doubled, so Compose does not interpolate it."""
    return json.dumps(text.replace("$", "$$"))


def compose(task_dir: Path, pid: str, scorer: str, tools_dir: Path, data: Path, lock_dir: Path) -> str:
    client = task_dir / "client"
    scratch = task_dir / "scratch"
    commands = [*CLIENT_COMMANDS, "ale-local"]
    command_mounts = "\n".join(f"      - {_quoted(f'{client}/bin/{c}:/usr/local/bin/{c}:ro')}" for c in commands)
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
    # ALE-Bench's judge for {pid}: it starts ALE-Bench's own containers through the
    # host's Docker socket (as `ale_bench` sessions do), with bind mounts from the scratch
    # directory, which is mounted at its host path.
    image: {scorer}
    pull_policy: never
    user: "0:0"
    entrypoint: ["python3", "-I", "/scorer/app/judge.py", "--config", "/scorer/app/config.json"]
    cpus: {SCORER_CPUS}
    mem_limit: "{SCORER_MEMORY_MB}m"
    pids_limit: 4096
    environment:
      TMPDIR: {_quoted(str(scratch))}
      ALE_BENCH_DATA: "/data"
      ALE_BENCH_CACHE: {_quoted(str(scratch / "cache"))}
      HF_HUB_OFFLINE: "1"
    # Every 2 s only while the scorer starts (up to 15 min), then once a minute: the check only gates the
    # agent's start, and a Python process every 2 s for a whole run costs the host ~1 thread.
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:{SCORER_PORT}/health', timeout=3)"]
      start_period: 900s
      start_interval: 2s
      interval: 60s
      timeout: 5s
      retries: 3
    volumes:
      - {_quoted(f'{task_dir / "scorer"}:/scorer/app:ro')}
      - {_quoted(f'{data}:/data:ro')}
      - {_quoted(f'{tools_dir}:/tools:ro')}
      - {_quoted(f'{scratch}:{scratch}')}
      - {_quoted(f'{lock_dir}:/locks')}
      - "/var/run/docker.sock:/var/run/docker.sock"
      - ${{HOST_VERIFIER_LOGS_PATH:?run this task through Harbor}}/../{STATE}:/scorer/state
"""


def grading_seconds(facts: dict, lite: bool, workers: int) -> dict[str, int]:
    """Rough upper bounds on an idle judge: a public evaluation, a private one."""
    per_case = facts["time_limit"] + CONTAINER_OVERHEAD_SEC
    public = facts["n_public_lite"] if lite else facts["n_public"]
    private = facts["n_private_lite"] if lite else facts["n_private"]
    return {"public": math.ceil(60 + public * per_case / workers),
            "private": math.ceil(120 + private * per_case / workers)}


def task_toml(pid: str, facts: dict, agents: int, cpus: int, memory_mb: int, token: str, verifier_timeout: float,
              metadata: dict) -> str:
    lines = "\n".join(f"{key} = {json.dumps(str(value))}" for key, value in metadata.items())
    return f"""\
schema_version = "1.1"
source = "{REPO_URL}"
artifacts = []

[task]
name = "ale-bench/{pid}"
authors = [{{ name = "Yuki Imajuku" }}, {{ name = "Kohki Horie" }}, {{ name = "Yoichi Iwata" }}, {{ name = "Kensho Aoki" }}, {{ name = "Naohiro Takahashi" }}, {{ name = "Takuya Akiba" }}]
keywords = ["ale-bench", "atcoder", "heuristic", "optimization", "algorithm-engineering"]

[metadata]
{lines}

[verifier]
timeout_sec = {float(verifier_timeout)}

[verifier.env]
SCORER_TOKEN = {json.dumps(token)}
SCORED_ARTIFACT_CONFIG = "{CLIENT_MOUNT}/task.json"

# While agents run they reach only the hosts run.py adds with --allow-agent-host
# (their model API); before and after, nothing.
[agent]
timeout_sec = {AGENT_TIMEOUT_SEC}
network_mode = "allowlist"
allowed_hosts = []

[environment]
build_timeout_sec = 1800.0
cpus = {cpus}
memory_mb = {memory_mb}
storage_mb = 8192
gpus = 0
network_mode = "allowlist"
allowed_hosts = []
"""


def wrapper(command: str) -> str:
    if command == "ale-local":
        return f'#!/bin/sh\nexec python3 {CLIENT_MOUNT}/alelocal.py "$@"\n'
    return f'#!/bin/sh\nALECTL_COMMAND={command} exec python3 {CLIENT_MOUNT}/alectl.py "$@"\n'


def ultra_wrapper(command: str, work_root: str = WORK_ROOT, client_mount: str = CLIENT_MOUNT) -> str:
    """Ultra mode's `workcopy NAME` and `submit` (from inside a copy): alectl.py as the copy's NAME, its agent id
    (the judge's log records it), the other commands as they are."""
    if command == "workcopy":
        return ultra.workcopy_script(work_root, f'AGENT_ID="$name" ALECTL_COMMAND=workcopy exec python3 '
                                                f'{client_mount}/alectl.py', ULTRA_ABOUT)
    if command == "submit":
        return ultra.submit_script(work_root, f'AGENT_ID="$name" ALECTL_COMMAND=submit exec python3 '
                                              f'{client_mount}/alectl.py "$@"', ULTRA_ABOUT)
    return wrapper(command)


def starter_files(pid: str, facts: dict, language: str, lite: bool, variant: str, profiles: dict, names: dict,
                  tools_dir: Path, data: Path = DATA) -> dict[str, bytes]:
    """The agents' copy: the problem's public material and tools, never its seeds,
    standings or results."""
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(data / f"{pid}.zip") as archive:
        tree = _zip_tree(archive, pid)
    for relative, content in tree.items():
        if relative == "statement_en.md":
            files["problem/statement.md"] = content
        elif relative == "statement_ja.md":
            files["problem/statement_ja.md"] = content
        elif relative.startswith("images/"):
            files[f"problem/{relative}"] = content
        elif relative in ("example_input.txt", "example_output.txt"):
            files[f"problem/{relative}"] = content
        elif relative.startswith("tools/") and "/target/" not in relative:
            files[relative] = content
    for tool in json.loads((tools_dir / "manifest.json").read_text())["binaries"]:
        files[f"tools/bin/{tool}"] = (tools_dir / tool).read_bytes()
    for path in sorted((tools_dir / "in").glob("*.txt")):
        files[f"in/{path.name}"] = path.read_bytes()
    profile = profiles[language]
    public = facts["public_seeds_lite"] if lite else facts["public_seeds"]
    problem = {
        "problem_id": pid, "title": facts["title"], "contest_url": facts["contest_url"],
        "problem_type": facts["problem_type"], "score_type": facts["score_type"],
        "time_limit": facts["time_limit"], "memory_limit": facts["memory_limit"], "lite_version": lite,
        "variant": variant, "language": language, "language_name": names.get(language, language),
        "judge_version": JUDGE_VERSION, "submission_name": SUBMISSION_NAMES[language], **profile,
        "public_seeds_text": seeds_text(public), "n_public_cases": len(public),
        "n_private_cases": facts["n_private_lite"] if lite else facts["n_private"],
        "ale_bench_commit": COMMIT, "dataset_revision": REVISION,
    }
    files["problem.json"] = (json.dumps(problem, indent=2) + "\n").encode()
    if variant == "outputs":
        files["out/README.txt"] = (f"Write out/0000.txt ... out/{len(VARIANT_SEEDS) - 1:04d}.txt here: one output "
                                   "per input in/0000.txt ... (the tester checks each).\n").encode()
    return files


@dataclass
class Task:
    """One problem's task directory. `files` maps paths relative to it to their content."""

    path: Path
    pid: str
    files: dict[str, bytes]
    modes: dict[str, int]
    description: dict

    @property
    def task_toml(self) -> str:
        return self.files["task.toml"].decode()

    def write(self) -> Path:
        for relative, data in self.files.items():
            target = self.path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(self.modes.get(relative, 0o644))
        (self.path / "scratch").mkdir(exist_ok=True)
        return self.path


@dataclass
class Job:
    """One Harbor job: a dataset directory with a task per problem."""

    path: Path
    tasks: list[Task]
    description: dict
    agent_hosts: list[str] = field(default_factory=list)
    # (old, new) replacements in the continuation (ultra mode's ULTRA_CONTINUATION_EDITS); none by default.
    continuation_edits: tuple = ()

    @property
    def continuation(self) -> str:
        text = (BENCH / "continuation.txt").read_text()
        return edited(text, self.continuation_edits) if self.continuation_edits else text

    @property
    def task_toml(self) -> str:
        return self.tasks[0].task_toml

    @property
    def verifier_env(self) -> dict[str, str]:
        """Recorded in Harbor's job config.json (--ve), so the job shows what it ran."""
        d = self.description
        return {"ALE_BENCH_PROBLEMS": ",".join(d["problems"]), "ALE_BENCH_LITE": str(int(d["lite_version"])),
                "ALE_BENCH_LANGUAGE": d["language"], "ALE_BENCH_VARIANT": d["variant"],
                "ALE_BENCH_COMMIT": COMMIT, "ALE_BENCH_REVISION": REVISION}

    @property
    def harbor_args(self) -> list[str]:
        args = network_allowlist.harbor_args(self.agent_hosts)
        for key, value in self.verifier_env.items():
            args += ["--ve", f"{key}={value}"]
        return args

    def write(self) -> Path:
        if self.path.exists():
            raise ValueError(f"task directory already exists: {self.path}; choose a new --name")
        staging = self.path.with_name(self.path.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        for task in self.tasks:
            final = task.path
            task.path = staging / final.name
            task.write()
            task.path = final
        (staging / "ale-bench-job.json").write_text(json.dumps(self.description, indent=2) + "\n")
        staging.rename(self.path)
        LOCK_DIR.mkdir(parents=True, exist_ok=True)
        try:
            LOCK_DIR.chmod(0o777)
        except PermissionError:
            pass
        return self.path


def build_task(pid: str, job_dir: Path, *, agents: int, cpus: int, memory_mb: int, lite: bool, language: str,
               variant: str, judge_cpus: list[int], private_lane: bool, scorer: str, profiles: dict, names: dict,
               libraries: dict, solution: Path | None = None, data: Path = DATA,
               container_prefix: str | None = None, ultra_mode: bool = False) -> Task:
    facts = {**reference()["problems"][pid], "problem_id": pid}
    if variant == "outputs" and facts["problem_type"] != "batch":
        raise ValueError(f"{pid} is interactive; the outputs variant applies to batch problems only")
    tools_dir = TOOLS / pid
    path = Path(job_dir).resolve() / pid
    token = secrets.token_hex(16)
    workers = len(judge_cpus)
    seconds = grading_seconds(facts, lite, workers)
    verifier_timeout = max(VERIFIER_MIN_SEC, (agents + MAX_PENDING_PER_AGENT) * seconds["public"]
                           + 2 * FINAL_CANDIDATES * seconds["private"] + 900)
    artifact =([f"out/{seed:04d}.txt" for seed in VARIANT_SEEDS] if variant == "outputs"
                else [SUBMISSION_NAMES[language]])
    max_bytes = 256 * 2**20 if variant == "outputs" else 524288  # ALE-Bench's code limit: 512 KiB
    client_config = {
        "benchmark": "ale-bench", "problem_id": pid, "agents": agents, "starter": f"{CLIENT_MOUNT}/starter",
        "work_root": WORK_ROOT, "artifact_files": artifact, "scorer_urls": list(SCORER_URLS),
        "score_type": facts["score_type"], "variant": variant, "metric_name": "public_score",
        "submit_wait_sec": seconds["public"] * (agents * MAX_PENDING_PER_AGENT + 2) + seconds["private"],
        "finalize_wait_sec": verifier_timeout,
    }
    public_info = {"benchmark": "ale-bench", "problem_id": pid, "language": language, "variant": variant,
                   "lite_version": lite, "score_type": facts["score_type"], "artifact": artifact}
    scorer_config = {
        "benchmark_name": f"ale-bench/{pid}", "problem_id": pid, "lite_version": lite, "language": language,
        "judge_version": JUDGE_VERSION, "variant": variant, "score_type": facts["score_type"],
        "variant_seeds": list(VARIANT_SEEDS), "port": SCORER_PORT, "state_dir": "/scorer/state", "token": token,
        "host_uid": os.getuid(), "host_gid": os.getgid(), "docker_gid": docker_gid(), "workers": 1,
        "judge_cpus": judge_cpus, "judge_workers": workers, "lock_dir": "/locks", "tools_dir": "/tools",
        "private_lane": private_lane and variant == "official", "max_pending_per_agent": MAX_PENDING_PER_AGENT,
        "final_candidates": FINAL_CANDIDATES, "metric_name": "public_score",
        "artifact": {"files": artifact, "max_bytes": max_bytes}, "public": public_info,
        "ale_bench_commit": COMMIT, "dataset_revision": REVISION, "task_id": f"{path.parent.name}/{pid}",
        "container_prefix": container_prefix or f"ale-bench-judge-{pid}",
        "images": [judge_image(language), TOOL_IMAGE],
        # ALE-Bench's containers on an optional Docker daemon whose state lives in RAM, when its socket is in the
        # lock directory at the judge's start, else on the main daemon (judge.py).
        "docker_host": "unix:///locks/docker/ale-docker.sock",
    }
    description = {
        "benchmark": "ale-bench", "problem_id": pid, "title": facts["title"], "agents": agents,
        **({"mode": "ultra"} if ultra_mode else {}),
        "lite_version": lite, "language": language, "judge_version": JUDGE_VERSION, "variant": variant,
        "official": variant == "official", "score_type": facts["score_type"], "problem_type": facts["problem_type"],
        "time_limit": facts["time_limit"], "memory_limit": facts["memory_limit"],
        "n_public_cases": facts["n_public_lite"] if lite else facts["n_public"],
        "n_private_cases": facts["n_private_lite"] if lite else facts["n_private"],
        "artifact_files": artifact if len(artifact) < 5 else [artifact[0], "...", artifact[-1]],
        "metric": "private_score" if variant == "official" else "total_absolute_score",
        "score": ("AtCoder performance of the final submission's private (system test) score"
                  if variant == "official" else
                  ("total absolute score" if facts["score_type"] == "maximize" else "n x 1e9 / total absolute score")),
        "judge_cpus": judge_cpus, "private_lane": scorer_config["private_lane"],
        "resources": {"cpus": cpus, "memory_mb": memory_mb, "per_agent_cpus": PER_AGENT_CPUS,
                      "per_agent_memory_mb": PER_AGENT_MEMORY_MB, "scorer_cpus": SCORER_CPUS,
                      "scorer_memory_mb": SCORER_MEMORY_MB, "judge_cpus": judge_cpus},
        "grading_seconds_idle": seconds, "commit": COMMIT, "revision": REVISION, "scorer_image": scorer,
        "judge_image": judge_image(language), "base_image": f"{base_image(language)}@{BASE_IMAGES[language]}",
        "tools": json.loads((tools_dir / "manifest.json").read_text()) if (tools_dir / "manifest.json").is_file()
        else None,
    }
    metadata = {"ale_bench_problem": pid, "ale_bench_lite": int(lite), "ale_bench_language": language,
                "ale_bench_variant": variant, "ale_bench_commit": COMMIT, "ale_bench_revision": REVISION,
                "ale_bench_time_limit": facts["time_limit"], "ale_bench_problem_type": facts["problem_type"]}
    dockerfile = (BENCH / "environment" / "Dockerfile").read_text().replace(
        "{base_image}", f"{base_image(language)}@{BASE_IMAGES[language]}")
    files: dict[str, bytes] = {
        "instruction.md": instruction(facts, language, lite, variant, agents, cpus, profiles, names,
                                      libraries, ultra_mode).encode(),
        "task.toml": task_toml(pid, facts, agents, cpus, memory_mb, token, verifier_timeout, metadata).encode(),
        "ale-bench.json": (json.dumps(description, indent=2) + "\n").encode(),
        "environment/Dockerfile": dockerfile.encode(),
        "environment/docker-compose.yaml": compose(path, pid, scorer, tools_dir, data, LOCK_DIR).encode(),
        "client/task.json": (json.dumps(client_config, indent=2) + "\n").encode(),
        "client/alectl.py": (BENCH / "client" / "alectl.py").read_bytes(),
        "client/alelocal.py": (BENCH / "client" / "alelocal.py").read_bytes(),
        "scorer/judge.py": (BENCH / "scorer" / "judge.py").read_bytes(),
        "scorer/server.py": (LAYER / "scorer" / "server.py").read_bytes(),
        "scorer/sandbox.py": (LAYER / "scorer" / "sandbox.py").read_bytes(),
        # The finalize token: readable by the task's owner (the judge drops to that user).
        "scorer/config.json": (json.dumps(scorer_config, indent=2) + "\n").encode(),
        "tests/test.sh": (LAYER / "tests" / "test.sh").read_bytes(),
        "tests/finalize.py": (LAYER / "tests" / "finalize.py").read_bytes(),
    }
    modes = {"scorer/config.json": 0o600, "tests/test.sh": 0o755}
    for command in (*CLIENT_COMMANDS, "ale-local"):
        files[f"client/bin/{command}"] = (ultra_wrapper if ultra_mode else wrapper)(command).encode()
        modes[f"client/bin/{command}"] = 0o755
    if (tools_dir / "manifest.json").is_file():
        for relative, content in starter_files(pid, facts, language, lite, variant, profiles, names, tools_dir,
                                               data).items():
            files[f"client/starter/{relative}"] = content
            if relative.startswith("tools/bin/"):
                modes[f"client/starter/{relative}"] = 0o755
    if solution is not None:
        solution = Path(solution)
        if variant == "outputs":
            if not solution.is_dir():
                raise ValueError("the outputs variant's solution is a directory holding out/0000.txt ...")
            for name in artifact:
                files[f"solution/artifact/{name}"] = (solution / name).read_bytes()
            copy = 'cp -r /solution/artifact/. "$dest/"\n'
        else:
            files[f"solution/{artifact[0]}"] = solution.read_bytes()
            copy = f'cp /solution/{artifact[0]} "$dest/{artifact[0]}"\n'
        files["solution/solve.sh"] = ("#!/bin/bash\nset -euo pipefail\nexport AGENT_ID=oracle\n"
                                      + (f'dest=$(workcopy oracle)\n{copy}cd "$dest"\nsubmit\n' if ultra_mode else
                                         f"dest=$(workcopy)\n{copy}submit\n")).encode()
        modes["solution/solve.sh"] = 0o755
    return Task(path, pid, files, modes, description)


def docker_gid() -> int:
    try:
        return os.stat("/var/run/docker.sock").st_gid
    except OSError:
        return 0


def build_job(dest: Path, problems: list[str], *, agents: int = 1, cli: str | None = None, codex_login: bool = False,
              cpus: int | None = None, memory_mb: int | None = None, lite: bool = False,
              language: str = DEFAULT_LANGUAGE, variant: str = "official", judge_cpus: list[int] | None = None,
              private_lane: bool = True, dry_run: bool = False, solution: Path | None = None,
              clone: Path = CLONE, container_prefix: str | None = None, ultra: bool = False) -> Job:
    """A job's tasks (dry_run: without fetching, building or writing anything). ULTRA (run.py --protocol
    ultra): one agent and its own helpers, AGENTS of them at most, the root included: the team's resources, the
    team's text with named copies (ULTRA_EDITS), `workcopy NAME` and `submit` from inside a copy."""
    ultra_mode = ultra
    if ultra_mode and agents < 2:
        raise ValueError("ultra mode needs --agents 2 or more (the most agents at once, the root included)")
    if language not in BASE_IMAGES:
        raise ValueError(f"--ale-language must be one of {', '.join(BASE_IMAGES)}")
    if variant not in SOLUTION_ALLOW:
        raise ValueError(f"--ale-variant must be one of {', '.join(SOLUTION_ALLOW)}")
    ref = reference()
    if lite:
        outside = [pid for pid in problems if pid not in ref["lite"]]
        if outside:
            raise ValueError(f"lite cases exist only for the lite set ({', '.join(ref['lite'])}); "
                             f"not for {', '.join(outside)}")
    if variant == "outputs":
        interactive = [pid for pid in problems if ref["problems"][pid]["problem_type"] != "batch"]
        if interactive:
            raise ValueError("the outputs variant applies to batch problems only; interactive: "
                             + ", ".join(interactive))
    hosts = network_allowlist.allowed_hosts(cli, codex_login=codex_login) if cli else []
    total_cpus = cpus if cpus is not None else PER_AGENT_CPUS * agents
    total_memory = memory_mb if memory_mb is not None else PER_AGENT_MEMORY_MB * agents
    judge_cpus = judge_cpus or default_judge_cpus()
    description = {
        "benchmark": "ale-bench", "problems": problems, "agents": agents,
        "mode": "ultra" if ultra_mode else "team" if agents > 1 else "solo",
        "cli": cli, "lite_version": lite, "language": language, "judge_version": JUDGE_VERSION, "variant": variant,
        "judge_cpus": judge_cpus, "private_lane": private_lane and variant == "official",
        "resources": {"cpus": total_cpus, "memory_mb": total_memory, "per_agent_cpus": PER_AGENT_CPUS,
                      "per_agent_memory_mb": PER_AGENT_MEMORY_MB},
        "network": {"agent_phase": hosts, "setup_and_verifier": list(SETUP_HOSTS)},
        "commit": COMMIT, "revision": REVISION, "lock_dir": str(LOCK_DIR),
    }
    job = Job(Path(dest).resolve(), [], description, hosts,
              **({"continuation_edits": ULTRA_CONTINUATION_EDITS} if ultra_mode else {}))
    profiles = names = libraries = None
    if dry_run:
        # Only what the run's command line needs: the first problem's task.toml.
        pid = problems[0]
        facts = ref["problems"][pid]
        seconds = grading_seconds(facts, lite, len(judge_cpus))
        toml = task_toml(pid, facts, agents, total_cpus, total_memory, "", seconds["private"], {})
        job.tasks.append(Task(job.path / pid, pid, {"task.toml": toml.encode()}, {}, {}))
        return job
    fetch_clone(clone)
    fetch_dataset([*(f"{pid}.zip" for pid in problems), "problem_ids.txt", "problem_ids_lite.txt"])
    images = ensure_images(language, clone=clone)
    profiles = judge_profiles(clone)
    names, libraries = language_texts(clone)
    for pid in problems:
        ensure_tools(pid)
    description.update(images=images)
    for pid in problems:
        job.tasks.append(build_task(pid, job.path, agents=agents, cpus=total_cpus, memory_mb=total_memory, lite=lite,
                                    language=language, variant=variant, judge_cpus=judge_cpus,
                                    private_lane=private_lane, scorer=images["scorer"], profiles=profiles,
                                    names=names, libraries=libraries, solution=solution,
                                    container_prefix=container_prefix, ultra_mode=ultra_mode))
    return job


def job_dir_name(problems: list[str], lite: bool, spec: str) -> str:
    label = spec if spec in ("lite", "all") else (problems[0] if len(problems) == 1 else f"{len(problems)}problems")
    return f"ale-bench-{label}" + ("-lite" if lite and spec != "lite" else "")


# ----------------------------------------------------------------------------
# Reading finished jobs (regrade, summary)


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if Path(path).is_file():
        for line in Path(path).read_text(errors="replace").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows


def job_description(job: Path) -> dict | None:
    """The ale-bench-job.json of the tasks a job (or trial) ran, or None."""
    job = Path(job)
    for config_path in (job / "config.json", job.parent / "config.json"):
        config = _read_json(config_path)
        for dataset in config.get("datasets") or []:
            path = Path(dataset.get("path") or "")
            if dataset.get("path") and (path / "ale-bench-job.json").is_file():
                return {**_read_json(path / "ale-bench-job.json"), "job_dir": str(path)}
        env = (config.get("verifier") or {}).get("env") or {}
        if env.get("ALE_BENCH_PROBLEMS"):
            return {"benchmark": "ale-bench", "problems": env["ALE_BENCH_PROBLEMS"].split(","),
                    "lite_version": env.get("ALE_BENCH_LITE") == "1", "language": env.get("ALE_BENCH_LANGUAGE"),
                    "variant": env.get("ALE_BENCH_VARIANT"), "job_dir": None}
    return None


def is_trial(trial: Path) -> bool:
    return (Path(trial) / STATE / "submissions.jsonl").is_file() or (Path(trial) / STATE / "final.json").is_file()


def trials(job: Path) -> list[Path]:
    job = Path(job)
    if (job / "agent").is_dir():
        return [job]
    return sorted(path for path in job.iterdir() if (path / "agent").is_dir())


def trial_task(trial: Path) -> dict:
    """The ale-bench.json of the task a trial ran (its problem, cases, language)."""
    config = _read_json(Path(trial) / "config.json")
    task = config.get("task") or {}
    path = task.get("path")
    return _read_json(Path(path) / "ale-bench.json") if path else {}


def trial_records(trial: Path) -> list[dict]:
    """One record per graded submission, oldest first, for score-over-budget curves: the judge's
    log with each submission's private grade from the private lane where it has one
    (`score`: its AtCoder performance, `metric`: its private score); submissions the lane
    did not grade (it grades only the run's final candidate) have no score. Outputs-variant
    records carry their variant score."""
    state = Path(trial) / STATE
    private = {}
    for entry in _read_jsonl(state / "private.jsonl"):
        if entry.get("status") == "done":
            private[str(entry["sid"])] = entry
    records = []
    for record in _read_jsonl(state / "submissions.jsonl"):
        row = {key: record.get(key) for key in ("sid", "agent_id", "submitted_ts", "ts", "status", "valid",
                                                "public_score", "public_judge", "final", "note", "sha256")}
        lane = private.get(str(record.get("sid")))
        if "variant_score" in record:  # the outputs variant: its own score, no private lane
            valid = bool(record.get("valid"))
            row.update(score=record.get("variant_score") if valid else None,
                       metric=record.get("public_score") if valid else None, metric_name="total_absolute_score")
        elif lane is not None:
            row.update(score=lane.get("performance"), metric=lane.get("private_score"), metric_name="private_score",
                       rank=lane.get("rank"), private_judge=lane.get("private_judge"), private_ts=lane.get("ts"))
        else:
            row.update(score=None, metric=None, metric_name="private_score")
        records.append(row)
    return sorted(records, key=lambda r: r.get("ts") or "")


def trial_final(trial: Path) -> dict | None:
    final = _read_json(Path(trial) / STATE / "final.json")
    return final or None


# The CLIs' home directories under a trial's agent directory, by CLI (agent.py LOG_DIR_NAME).
HOME_DIRS = {"codex": "codex-home", "claude": "claude-home"}


def token_steps(agent_dir: Path) -> tuple[list[float], list[dict[str, int]]]:
    """Times at which the trial's summed token usage changed, and the totals then."""
    from token_budget import READERS, summed

    events = []  # (time, session, usage total)
    for cli, home_dir in HOME_DIRS.items():
        for home in sorted(agent_dir.glob(f"**/{home_dir}")):
            reader = READERS[cli](home, keep_timeline=True)
            reader.poll()
            events += reader.timeline
    latest: dict[str, dict[str, int]] = {}
    times, totals = [], []
    for when, session, usage in sorted(events, key=lambda event: event[0]):
        latest[session] = usage
        times.append(when)
        totals.append(summed(latest))
    return times, totals


def submissions_with_tokens(job: Path) -> list[dict]:
    """Every submission of a job's trials with the team's cumulative tokens at the
    moment it was graded (CAISI weighted and output tokens)."""
    import bisect

    from token_budget import parse_ts, weighted_tokens

    rows = []
    for trial in trials(job):
        times, totals = token_steps(trial / "agent")
        for record in trial_records(trial):
            when = parse_ts(record["ts"]) if record.get("ts") else None
            index = bisect.bisect_right(times, when) - 1 if when is not None else -1
            usage = totals[index] if index >= 0 else None
            rows.append({"trial": trial.name, **record,
                         "weighted_tokens": round(weighted_tokens(usage)) if usage else 0,
                         "output_tokens": usage["output_tokens"] if usage else 0})
    return rows


def annotate_job(job: Path) -> None:
    """After a run: <trial>/ale-bench/submissions-tokens.jsonl, every submission with its
    public result, private grade (where the lane has one) and the team's tokens."""
    rows = submissions_with_tokens(job)
    for trial in trials(job):
        if (trial / STATE).is_dir():
            mine = [row for row in rows if row["trial"] == trial.name]
            (trial / STATE / "submissions-tokens.jsonl").write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in mine))


# ----------------------------------------------------------------------------
# Grading outside Harbor: the same judge in a one-shot container


def scorer_run(task_dir: Path, args: list[str], name: str, extra_mounts: list[str] | None = None,
               env: dict | None = None) -> subprocess.CompletedProcess:
    """judge.py in the task's scorer image with the task's mounts (as the run's judge)."""
    task_dir = Path(task_dir).resolve()
    config = _read_json(task_dir / "ale-bench.json")
    scratch = task_dir / "scratch"
    scratch.mkdir(exist_ok=True)
    state = Path(tempfile.mkdtemp(prefix="state-", dir=scratch))
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    command = ["docker", "run", "--rm", "--name", name, "--network", "none", "--cpus", str(SCORER_CPUS),
               "--memory", f"{SCORER_MEMORY_MB}m", "--user", "0:0",
               "-e", f"TMPDIR={scratch}", "-e", "ALE_BENCH_DATA=/data", "-e", f"ALE_BENCH_CACHE={scratch / 'cache'}",
               "-e", "HF_HUB_OFFLINE=1",
               "-v", f"{task_dir / 'scorer'}:/scorer/app:ro", "-v", f"{DATA}:/data:ro",
               "-v", f"{TOOLS / config['problem_id']}:/tools:ro", "-v", f"{scratch}:{scratch}",
               "-v", f"{LOCK_DIR}:/locks", "-v", "/var/run/docker.sock:/var/run/docker.sock",
               "-v", f"{state}:/scorer/state"]
    for key, value in (env or {}).items():
        command += ["-e", f"{key}={value}"]
    command += extra_mounts or []
    command += ["--entrypoint", "python3", config["scorer_image"], "-I", "/scorer/app/judge.py",
                "--config", "/scorer/app/config.json", *args]
    try:
        return subprocess.run(command, capture_output=True, text=True)
    finally:
        shutil.rmtree(state, ignore_errors=True)


def one_shot_task(pid: str, lite: bool, language: str, variant: str = "official",
                  judge_cpus: list[int] | None = None, private_lane: bool = False) -> Path:
    """A task directory for grading without an agent (under .cache/ale-bench-tasks/manual)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{secrets.token_hex(2)}"
    job = build_job(TASKS / "manual" / f"{pid}-{stamp}", [pid], lite=lite, language=language, variant=variant,
                    judge_cpus=judge_cpus, private_lane=private_lane,
                    container_prefix=f"adapter-ale-bench-judge-{pid}")
    job.write()
    return job.path / pid


def grade_files(pid: str, files: list[Path], *, lite: bool = False, language: str = DEFAULT_LANGUAGE,
                public: bool = True, private: bool = False, repeats: int = 1, variant: str = "official",
                judge_cpus: list[int] | None = None, direct: bool = False) -> list[dict]:
    """Grade programs (or outputs directories) with the judge alone; with direct, with
    ale_bench.start() and private_eval as a user of the package would."""
    task_dir = one_shot_task(pid, lite, language, variant, judge_cpus)
    mounts, targets = [], []
    for index, path in enumerate(files):
        target = f"/grade/{index}" + ("" if Path(path).is_dir() else Path(path).suffix)
        mounts += ["-v", f"{Path(path).resolve()}:{target}:ro"]
        targets.append(target)
    if direct:
        args = ["direct", *targets]
    else:
        args = ["grade", *(["--public"] if public else []), *(["--private"] if private else []),
                "--repeats", str(repeats), *targets]
    try:
        done = scorer_run(task_dir, args, f"adapter-ale-bench-grade-{secrets.token_hex(3)}", mounts)
    finally:
        shutil.rmtree(task_dir.parent, ignore_errors=True)  # the one-shot job directory
    if done.returncode != 0:
        raise RuntimeError(f"grading failed: {done.stderr[-3000:]}")
    rows = [json.loads(line) for line in done.stdout.splitlines() if line.startswith("{")]
    for row in rows:
        index = int(Path(row["file"]).name.split(".")[0])
        row["file"] = str(files[index])
    return rows


def regrade(jobs: list[Path], top: int = 1, repeats: int = 1, judge_cpus: list[int] | None = None) -> list[dict]:
    """Each trial's final candidate (and the next best public submissions, up to `top`)
    graded again on the private cases by the same judge, with the trial's problem,
    cases and language."""
    sys.path.insert(0, str(BENCH / "scorer"))
    sys.path.insert(0, str(LAYER / "scorer"))
    from judge import final_candidates  # noqa: E402  (pure; needs no ale_bench)

    rows = []
    for job in jobs:
        for trial in trials(job):
            described = trial_task(trial)
            if not described:
                raise ValueError(f"{trial} is not an ale-bench trial with its task directory")
            records = _read_jsonl(trial / STATE / "submissions.jsonl")
            chosen, seen = [], set()
            for record in final_candidates(records, described["score_type"]):
                if record.get("sha256") in seen:
                    continue
                seen.add(record.get("sha256"))
                chosen.append(record)
                if len(chosen) == top:
                    break
            if not chosen:
                continue
            paths = [trial / STATE / "submissions" / str(r["sid"]) / "files" / described["artifact_files"][0]
                     for r in chosen]
            graded = grade_files(described["problem_id"], paths, lite=described["lite_version"],
                                 language=described["language"], public=False, private=True, repeats=repeats,
                                 judge_cpus=judge_cpus)
            final = trial_final(trial) or {}
            for record, path in zip(chosen, paths):
                results = [row["private"] for row in graded if row["file"] == str(path)]
                with (trial / STATE / "regrades.jsonl").open("a") as stream:  # kept with the trial
                    for result in results:
                        stream.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), "sid": record["sid"],
                                                 **{k: v for k, v in result.items() if not k.startswith("case_")}},
                                                sort_keys=True) + "\n")
                rows.append({"trial": f"{trial.parent.name}/{trial.name}", "sid": record["sid"],
                             "agent_id": record["agent_id"], "public_score": record.get("public_score"),
                             "final": record["sid"] == final.get("sid"),
                             "original_performance": final.get("performance") if record["sid"] == final.get("sid")
                             else None,
                             "original_private_score": final.get("private_score")
                             if record["sid"] == final.get("sid") else None,
                             "private_scores": [r.get("private_score") for r in results],
                             "performances": [r.get("performance") for r in results],
                             "ranks": [r.get("rank") for r in results],
                             "judges": [r.get("private_judge") for r in results]})
    return rows


def summary(jobs: list[Path]) -> dict:
    """Every trial's final grade, and per job the average performance and the AHC rating
    of the problems it ran (upstream's RatingCalculator, in the judge's image)."""
    rows, by_job = [], {}
    for job in jobs:
        described = job_description(job) or {}
        for trial in trials(job):
            final = trial_final(trial) or {}
            task = trial_task(trial)
            row = {"job": Path(job).name, "trial": trial.name, "problem": task.get("problem_id") or final.get(
                "problem_id"), "performance": final.get("performance"), "rank": final.get("rank"),
                "private_score": final.get("private_score"), "valid": final.get("valid"),
                "lite_version": task.get("lite_version", described.get("lite_version"))}
            rows.append(row)
            by_job.setdefault(Path(job).name, []).append(row)
    out = {"trials": rows, "jobs": {}}
    for name, entries in by_job.items():
        graded = [e for e in entries if isinstance(e["performance"], (int, float))]
        per_problem: dict[str, list[int]] = {}
        for entry in graded:
            per_problem.setdefault(entry["problem"], []).append(entry["performance"])
        means = {pid: sum(v) / len(v) for pid, v in per_problem.items()}
        out["jobs"][name] = {"problems": len(means), "trials": len(entries), "graded": len(graded),
                             "average_performance": (sum(means.values()) / len(means)) if means else None,
                             "per_problem_mean_performance": means}
    return out


def rating(performances: dict[str, int]) -> int | None:
    """ALE-Bench's RatingCalculator (final contest ahc046, as its docs require), run in
    the judge's image."""
    images = [line for line in _docker("images", "--format", "{{.Repository}}:{{.Tag}}").stdout.split()
              if line.startswith("ttc-ale-bench-scorer:")]
    if not images or not performances:
        return None
    code = ("import json, sys; from ale_bench.data import RatingCalculator; "
            "print(RatingCalculator().calculate_rating(json.loads(sys.argv[1]), 'ahc046'))")
    done = _docker("run", "--rm", "--network", "none", "-e", "ALE_BENCH_DATA=/data", "-v", f"{DATA}:/data:ro",
                   "--name", f"adapter-ale-bench-rating-{secrets.token_hex(3)}", images[0], "python3", "-I", "-c",
                   code, json.dumps(performances))
    return int(done.stdout.strip()) if done.returncode == 0 and done.stdout.strip().isdigit() else None


# ----------------------------------------------------------------------------
# Command line


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="clone the pinned commit and fetch the dataset")
    fetch.add_argument("--problems", default="all", help="lite, all, or ids with commas (default: all)")
    images = commands.add_parser("images", help="pull and build the images a language needs")
    images.add_argument("--language", default=DEFAULT_LANGUAGE, choices=tuple(BASE_IMAGES))
    tools = commands.add_parser("tools", help="build problems' tools and inputs (upstream's cargo build)")
    tools.add_argument("problems")
    commands.add_parser("problems", help="list the problems and their facts")
    commands.add_parser("reference", help="recompute benchmarks/ale-bench/dataset.json from a full download")
    task = commands.add_parser("task", help="write a task to read")
    check = commands.add_parser("check", help="grade a program through the whole task (oracle agent)")
    grade = commands.add_parser("grade", help="grade programs with the judge only")
    direct = commands.add_parser("direct", help="ale_bench.start() and private_eval, unmodified (in the judge image)")
    for sub in (task, check, grade, direct):
        sub.add_argument("problem")
        sub.add_argument("--lite", action="store_true", help="the lite version's cases (lite-set problems)")
        sub.add_argument("--language", default=DEFAULT_LANGUAGE, choices=tuple(BASE_IMAGES))
    for sub in (task, check, grade):
        sub.add_argument("--variant", default="official", choices=SOLUTION_ALLOW)
        sub.add_argument("--judge-cpus", help="e.g. 12,14 (default: the last physical cores)")
    task.add_argument("--agents", type=int, default=1)
    task.add_argument("--out", type=Path, default=TASKS / "manual")
    for sub in (task, check):
        sub.add_argument("--ultra", action="store_true",
                         help="ultra mode's task (run.py --protocol ultra): one agent and its helpers, at most --agents "
                              "(task) or 2 (check), with named copies")
    check.add_argument("--solution", type=Path, required=True)
    check.add_argument("--name")
    grade.add_argument("files", nargs="+", type=Path)
    grade.add_argument("--public", action="store_true", help="public evaluation (default unless --private)")
    grade.add_argument("--private", action="store_true", help="private evaluation: score, rank, performance")
    grade.add_argument("--repeats", type=int, default=1)
    direct.add_argument("files", nargs="+", type=Path)
    again = commands.add_parser("regrade", help="grade jobs' final candidates again on the private cases")
    again.add_argument("jobs", nargs="+", type=Path)
    again.add_argument("--top", type=int, default=1)
    again.add_argument("--repeats", type=int, default=1)
    again.add_argument("--judge-cpus")
    summ = commands.add_parser("summary", help="final performances, averages and ratings of jobs")
    summ.add_argument("jobs", nargs="+", type=Path)
    log = commands.add_parser("submissions", help="every submission of jobs with the team's tokens")
    log.add_argument("jobs", nargs="+", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "fetch":
            fetch_clone()
            problems = problem_ids("all") if args.problems == "all" else check_problems(args.problems)[0]
            print(fetch_dataset([*(f"{p}.zip" for p in problems), "problem_ids.txt", "problem_ids_lite.txt",
                                 "schedule.csv", "ranking.csv"]))
            return 0
        if args.command == "images":
            print(json.dumps(ensure_images(args.language), indent=2))
            return 0
        if args.command == "tools":
            for pid in check_problems(args.problems)[0]:
                print(ensure_tools(pid))
            return 0
        if args.command == "problems":
            ref = reference()
            for pid in ref["all"]:
                facts = ref["problems"][pid]
                print(f"{pid:26} {'lite' if pid in ref['lite'] else '    '} {facts['problem_type']:8} "
                      f"{facts['score_type']:8} TL {facts['time_limit']:g}s  public {facts['n_public']}/"
                      f"{facts['n_public_lite']}  private {facts['n_private']}/{facts['n_private_lite']}  "
                      f"{facts['title']}")
            return 0
        if args.command == "reference":
            fetch_dataset(list(reference()["files"]) if REFERENCE.is_file() else None)
            REFERENCE.write_text(json.dumps(make_reference(), indent=2) + "\n")
            print(REFERENCE)
            return 0
        if args.command in ("task", "check", "grade", "direct"):
            pid = check_problems(args.problem)[0]
            if len(pid) != 1:
                raise ValueError("give one problem id")
            pid = pid[0]
            cpus = parse_cpus(args.judge_cpus) if getattr(args, "judge_cpus", None) else None
            if args.command == "task":
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
                job = build_job(args.out / f"{pid}-{stamp}", [pid], agents=args.agents, lite=args.lite,
                                language=args.language, variant=args.variant, judge_cpus=cpus, ultra=args.ultra)
                print(job.write())
                return 0
            if args.command in ("grade", "direct"):
                rows = grade_files(pid, args.files, lite=args.lite, language=args.language,
                                   public=getattr(args, "public", False) or not getattr(args, "private", False),
                                   private=getattr(args, "private", False), repeats=getattr(args, "repeats", 1),
                                   variant=getattr(args, "variant", "official"), judge_cpus=cpus,
                                   direct=args.command == "direct")
                for row in rows:
                    print(json.dumps(row))
                return 0
            name = args.name or f"ale-bench-check-{pid}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
            if (ROOT / "jobs" / name).exists():
                raise ValueError(f"job already exists: {ROOT / 'jobs' / name}; choose a new --name")
            job = build_job(TASKS / name, [pid], lite=args.lite, language=args.language, variant=args.variant,
                            judge_cpus=cpus, solution=args.solution,
                            **({"agents": 2, "ultra": True} if args.ultra else {}))
            job.write()
            code = subprocess.call(["harbor", "run", "-p", str(job.path), "--yes", "--env", "docker",
                                    "--no-force-build", "--jobs-dir", str(ROOT / "jobs"), "--job-name", name,
                                    "-k", "1", "-n", "1", "--agent", "oracle", *job.harbor_args], cwd=ROOT)
            for trial in trials(ROOT / "jobs" / name) if (ROOT / "jobs" / name).is_dir() else []:
                result = _read_json(trial / "result.json")
                print(json.dumps({"trial": str(trial), "final": trial_final(trial),
                                  "reward": (result.get("verifier_result") or {}).get("rewards"),
                                  "submissions": len(trial_records(trial))}))
            return code
        if args.command == "regrade":
            cpus = parse_cpus(args.judge_cpus) if args.judge_cpus else None
            for row in regrade(args.jobs, args.top, args.repeats, cpus):
                print(json.dumps(row))
            return 0
        if args.command == "summary":
            result = summary(args.jobs)
            for name, entry in result["jobs"].items():
                entry["rating"] = rating({pid: round(v) for pid, v in entry["per_problem_mean_performance"].items()})
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "submissions":
            for job in args.jobs:
                for row in submissions_with_tokens(job):
                    print(json.dumps(row))
            return 0
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return 1


if __name__ == "__main__":
    sys.exit(main())
