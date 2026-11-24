#!/usr/bin/env python3
"""Anthropic's original performance take-home as a scored-artifact task.

`run.py perf-takehome` runs https://github.com/anthropics/original_performance_takehome
at a pinned commit: agents optimize KernelBuilder.build_kernel in perf_takehome.py, a
kernel (tree traversal plus a six-stage hash) for a Python simulator of a one-core
VLIW SIMD machine, starting from the full version's scalar starter (147,734 cycles).

The repository has no license file, so this repository holds none of its code: the
first run clones it into .cache/perf-takehome and every task reads the files at the
pinned commit from git, checked against the hashes below. Each agent gets the whole
repository as its working copy; a submission is its perf_takehome.py. The scorer
(benchmarks/perf-takehome/scorer/score.py) builds the kernel in a sandbox, then runs
the instructions on its own pristine tests/frozen_problem.py: valid if correct on
the job's 9 fixed inputs and 8 fresh random ones, cycles = the worst of the 9 fixed
inputs, score = 147734 / cycles.

    python perf_takehome_task.py fetch                       # clone and check the pinned commit
    python scored_artifact.py grade perf-takehome a.py b.py   # grade kernels with the scorer
    python scored_artifact.py check perf-takehome --solution a.py
"""

from __future__ import annotations

import hashlib
import secrets
import subprocess
import sys
from pathlib import Path

from scored_artifact import Benchmark, Prepared, solo_text

ROOT = Path(__file__).resolve().parent
REPO_URL = "https://github.com/anthropics/original_performance_takehome"
COMMIT = "5452f74bd977807ac2e74f3d29432b9df6f25197"  # 2026-01-21, "Add notes about proper verification"
CLONE = ROOT / ".cache" / "perf-takehome"
# sha256 of every file at COMMIT; tests/frozen_problem.py is a copy of problem.py.
FILES = {
    ".gitignore": "5227ca275d1bdbc64db8277658fa373fb2b63c559e3fe2c67afd89e6df64d7de",
    "Readme.md": "398566c82976fb1cd72b4d63d718936e34ca98d122f6ffff6093915331dba0d6",
    "perf_takehome.py": "af14cbb2e8666aaba375aa6875e731b176875cb89b6f870473ae05cda979c15d",
    "problem.py": "fadb0f0858e2259f5759077a5544b9906dad3ceee80d37b4f0aa77da730c93c9",
    "tests/frozen_problem.py": "fadb0f0858e2259f5759077a5544b9906dad3ceee80d37b4f0aa77da730c93c9",
    "tests/submission_tests.py": "11c57cc999da93acb41201191073cd657ddffa87635359b3157c6e177c18ea0a",
    "watch_trace.html": "eff04f921d3758329041b29f77d703fbec35f888c1e487575a54b56766d15bfe",
    "watch_trace.py": "05fbe7690434048897961f2c9f4bc572302f8f78f499b35a4229b8bf8776da03",
}
BASELINE_CYCLES = 147734  # BASELINE in tests/submission_tests.py: the starter's cycles
# do_kernel_test(10, 16, 256) in tests/submission_tests.py
FOREST_HEIGHT, ROUNDS, BATCH_SIZE = 10, 16, 256
FIXED_INPUTS = 9  # worst of 9 seeded runs, as on the community leaderboard
FRESH_INPUTS = 8  # CorrectnessTests runs 8 random inputs
BUILD_TIMEOUT_SEC = 300
SIMULATION_TIMEOUT_SEC = 120  # one input; the starter takes about 0.5 s

TEAM_INSTRUCTION = """\
You are optimizing a kernel in Anthropic's original performance take-home
(github.com/anthropics/original_performance_takehome, the full version, whose
starter kernel takes 147734 cycles).

Run `workcopy` once: it creates your own copy of the take-home repository at
/work/$AGENT_ID (with a git commit of the starter, so `git diff` shows your
changes) and prints its path. Work only in your copy. {n} agents share this
container, and each has its own copy; do not edit another agent's copy.

The task, as perf_takehome.py states it: optimize the kernel (in
KernelBuilder.build_kernel) as much as possible in the available time, as measured
by test_kernel_cycles on a frozen separate copy of the simulator. Validate your
results using `python3 tests/submission_tests.py` without modifying anything in the
tests/ folder. Read Readme.md and the top of perf_takehome.py first, then problem.py.

Scoring:
- `submit` sends your copy's perf_takehome.py to the scorer, waits for the grade and
  prints it; `submit FILE` grades another file as perf_takehome.py. The same scorer
  grades the final result. It runs in a separate container with its own pristine
  copy of the repository and takes only perf_takehome.py from you: changes to
  problem.py or tests/ have no effect on the score.
- It builds your kernel with KernelBuilder().build_kernel(10, 2047, 256, 16) (at most
  300 s) and runs the instructions on the frozen simulator, as tests/submission_tests.py
  does. The kernel counts only if its output values are correct on every input:
  9 fixed inputs, kept secret and the same for every submission of this run, and 8
  fresh random inputs.
- Cycles are the worst of the 9 fixed inputs (cycle counts can depend on the data
  when a kernel branches). Score = 147734 / cycles, higher is better; a kernel that
  fails scores 0.
- The result of the run is the best valid submission of any agent. Submit early,
  and after every improvement.
- `scores` lists every graded submission of the team, best first.
  `fetch-submission SID DIR` (or `fetch-submission best DIR`) writes a submission's
  perf_takehome.py into DIR, so you can adopt or merge a teammate's verified
  improvement into your own copy.

The container is offline (you cannot install packages); python3, g++ and git are
available.
"""
SOLO_EDITS = (
    ("changes) and prints its path. Work only in your copy. {n} agents share this",
     "changes) and prints its path. Work only in your copy."),
    ("container, and each has its own copy; do not edit another agent's copy.", None),
    ("- The result of the run is the best valid submission of any agent. Submit early,",
     "- The result of the run is your best valid submission. Submit early,"),
    ("- `scores` lists every graded submission of the team, best first.",
     "- `scores` lists your graded submissions, best first."),
    ("  perf_takehome.py into DIR, so you can adopt or merge a teammate's verified",
     "  perf_takehome.py into DIR."),
    ("  improvement into your own copy.", None),
)
# Ultra mode (run.py --protocol ultra; ultra.py): one agent and its own helpers, who share one $AGENT_ID, so each
# names its copy (`workcopy NAME`) and submits from inside it; every copy is submitted at the end, and the best
# counts, as for a team. The team's text with these edits, as ale_bench_task.py's; the cap line comes before it
# (agent.py) and the resources line says it counts the helpers.
ULTRA_EDITS = (
    ("Run `workcopy` once: it creates your own copy of the take-home repository at",
     "Every agent, you and each helper, works in its own copy: `workcopy NAME` creates"),
    ("/work/$AGENT_ID (with a git commit of the starter, so `git diff` shows your",
     "a copy of the take-home repository at /work/NAME once (with a git commit of the starter, so `git diff` "
     "shows your"),
    ("changes) and prints its path. Work only in your copy. {n} agents share this",
     "changes) and prints its path. Give each agent its own NAME. Work only in your copy;"),
    ("container, and each has its own copy; do not edit another agent's copy.", "do not edit another agent's copy."),
    ("- `submit` sends your copy's perf_takehome.py to the scorer, waits for the grade and",
     "- `submit`, run from inside your copy, sends its perf_takehome.py to the scorer, waits for the grade and"),
    ("- `scores` lists every graded submission of the team, best first.",
     "- `scores` lists every graded submission, yours and your helpers', best first."),
    ("  perf_takehome.py into DIR, so you can adopt or merge a teammate's verified",
     "  perf_takehome.py into DIR, so you can adopt or merge another agent's verified"),
)
# The continuation's mention of `submit`, in ultra mode (it must appear once), as FrontierSWE's.
ULTRA_CONTINUATION_EDITS = (
    ("and run `submit` to get a fresh graded score.", "and run `submit` from inside your copy to get a fresh graded "
                                                      "score."),
)


def _git(clone: Path, *args: str, text: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(clone), *args], capture_output=True, text=text)


def fetch(clone: Path = CLONE) -> Path:
    """Clone the take-home once (it is small) and make sure it has the pinned commit."""
    clone = Path(clone)
    if not (clone / ".git").exists():
        clone.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(["git", "clone", "-q", "--no-checkout", REPO_URL, str(clone)],
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"could not clone {REPO_URL}: {result.stderr.strip()}")
    if _git(clone, "cat-file", "-e", f"{COMMIT}^{{commit}}").returncode != 0:
        _git(clone, "fetch", "-q", "origin")
        if _git(clone, "cat-file", "-e", f"{COMMIT}^{{commit}}").returncode != 0:
            raise ValueError(f"{clone} does not have the pinned commit {COMMIT}")
    return clone


def upstream_files(clone: Path = CLONE, fetch_clone: bool = True) -> dict[str, bytes]:
    """Every file of the take-home at the pinned commit, read from git (whatever the
    clone's working tree holds) and checked against FILES."""
    clone = fetch(clone) if fetch_clone else Path(clone)
    listed = _git(clone, "ls-tree", "-r", "--name-only", COMMIT)
    if listed.returncode != 0:
        raise ValueError(f"cannot read commit {COMMIT} in {clone}: {listed.stderr.strip()}")
    names = listed.stdout.split()
    if sorted(names) != sorted(FILES):
        raise ValueError(f"the take-home at {COMMIT} has files {names}, expected {sorted(FILES)}")
    files = {}
    for name in names:
        shown = _git(clone, "show", f"{COMMIT}:{name}", text=False)
        if shown.returncode != 0 or hashlib.sha256(shown.stdout).hexdigest() != FILES[name]:
            raise ValueError(f"{name} at {COMMIT} in {clone} does not match its pinned hash")
        files[name] = shown.stdout
    return files


class PerfTakehome(Benchmark):
    name = "perf-takehome"
    title = "Anthropic's original performance take-home"
    source = REPO_URL
    authors = ({"name": "Tristan Hume"}, {"name": "Anthropic"})
    keywords = ("performance-engineering", "vliw", "simd", "kernel-optimization")
    artifact_files = ("perf_takehome.py",)
    max_artifact_bytes = 1_000_000  # the leaderboard's 1,000 KB
    metric_name = "cycles"
    metric_better = "lower"
    score_definition = f"{BASELINE_CYCLES} / cycles (the speedup tests/submission_tests.py prints)"
    per_agent_cpus = 2
    per_agent_memory_mb = 6144
    scorer_cpus = 2
    scorer_workers = 2
    grade_timeout_sec = 900

    def __init__(self, clone: Path = CLONE) -> None:
        self.clone = clone

    def prepare(self, fetch: bool = True) -> Prepared:
        files = upstream_files(self.clone, fetch_clone=fetch)
        seeds = [secrets.randbits(63) for _ in range(FIXED_INPUTS)]
        return Prepared(
            starter=files,
            data={f"upstream/{name}": data for name, data in files.items()},
            scorer_config={
                "upstream_dir": "/scorer/data/upstream", "baseline_cycles": BASELINE_CYCLES,
                "forest_height": FOREST_HEIGHT, "rounds": ROUNDS, "batch_size": BATCH_SIZE,
                "seeds": seeds, "fresh_inputs": FRESH_INPUTS, "build_timeout_sec": BUILD_TIMEOUT_SEC,
                "simulation_timeout_sec": SIMULATION_TIMEOUT_SEC, "simulation_workers": 2,
                "simulation_memory_mb": 3072,
            },
            description={"commit": COMMIT, "repository": REPO_URL, "baseline_cycles": BASELINE_CYCLES,
                         "fixed_inputs": FIXED_INPUTS, "fresh_inputs": FRESH_INPUTS,
                         "build_timeout_sec": BUILD_TIMEOUT_SEC,
                         "simulation_timeout_sec": SIMULATION_TIMEOUT_SEC,
                         "license": "no license file upstream; fetched at run time, not redistributed"})

    def instruction(self, agents: int, prepared: Prepared) -> str:
        if agents > 1:
            return TEAM_INSTRUCTION.replace("{n}", str(agents))
        return solo_text(TEAM_INSTRUCTION, SOLO_EDITS)

    ultra_continuation_edits = ULTRA_CONTINUATION_EDITS

    def ultra_instruction(self, agents: int, prepared: Prepared) -> str:
        """The team's text with named copies (ULTRA_EDITS), the same for every N (the cap line says N)."""
        del agents, prepared
        return solo_text(TEAM_INSTRUCTION, ULTRA_EDITS)

    def verifier_env(self, prepared: Prepared) -> dict[str, str]:
        return {"SCORED_ARTIFACT_BENCHMARK": self.name, "PERF_TAKEHOME_COMMIT": COMMIT}


BENCHMARK = PerfTakehome()


if __name__ == "__main__":
    if sys.argv[1:] != ["fetch"]:
        sys.exit(__doc__)
    print(fetch())
    print(f"{len(upstream_files())} files at {COMMIT} match their pinned hashes")
