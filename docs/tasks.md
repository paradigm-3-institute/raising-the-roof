# Tasks

Each task gives every agent its own working copy, a way to submit, and feedback on its submissions, and grades
outside the agents' container. A team's score is its best submission, graded as a single agent's would be.

## git-to-zig (FrontierSWE v2)

`run.py frontier-swe --task git-to-zig` (`frontier_swe.py`, `benchmarks/frontier-swe/`).

Reimplement git in Zig so that it behaves like the real git. The task comes from Proximal Labs'
[frontier-swe-v2](https://github.com/Proximal-Labs/frontier-swe-v2) at commit `da83f84f8fbcec3cbf0f2b17c98ebc811355c2df`
(a sparse clone in `.cache/frontier-swe-v2`, made on first use). The task keeps its own agent and verifier
images, its separate verifier (git's own test scripts; the reward is roughly the share of 743 of them that pass),
and its network policy. The adapter adds:

- working copies: `workcopy` copies the graded files of `/app` to `/work/<agent>`; `submit --score S` promotes a
  copy into `/app` when its self-check score `S` is at least that of the version `/app` holds. Every call is logged
  in `/logs/agent/frontier-swe/submissions.jsonl`, which all agents read. Only `/app` is graded.
- snapshot grading: a host process (`frontier_swe.py watch`, started by `run.py`) grades `/app` with the task's
  own verifier after promotions and every 30 minutes, and records the reward with the time in
  `<trial>/frontier-swe/snapshots.jsonl`. The scores are not shown to the agents. `--fastgrade` runs the same
  verifier in a plain container with the build on tmpfs (same rewards, far fewer disk writes).
- resources: a single agent gets the task's 4 CPUs and 16 GB; the post's teams were given 8 CPUs and 32 GB
  (`--cpus 8 --memory-mb 32768`) whatever their size.

## AHC014 and AHC038 (ALE-Bench)

`run.py ale-bench --problem ahc014|ahc038` (`ale_bench_task.py`, `benchmarks/ale-bench/`).

AtCoder Heuristic Contest problems from Sakana AI's [ALE-Bench](https://github.com/SakanaAI/ALE-Bench) at commit
`3da9b12fb5d112dabb3af693d1a42031c95142bc`, with the dataset
([SakanaAI/ALE-Bench](https://huggingface.co/datasets/SakanaAI/ALE-Bench), revision
`0f426173b4e4e73b09b2b3631ae0490f66b75f99`) fetched on first use, never vendored. Agents write a C++20 program.

- the judge (`benchmarks/ale-bench/scorer/judge.py`) runs ALE-Bench's own evaluation in its own container: each
  submission on the 50 public cases (the agents' feedback, via `submit` and `scores`), and at the end the best
  public submission on the contest's private cases. The reward is the AtCoder performance that result would have
  earned against the contest's human contestants. Every case is pinned to one of the judge's CPUs
  (`--ale-judge-cpus`), one at a time machine-wide, so timing does not depend on load.
- agents also get `ale-local`: the problem's generator, tester and visualizer, and a local evaluation like the
  judge's.
- resources: 2 CPUs and 6 GB per agent; a team of N gets N times both.

## Performance take-home (Anthropic)

`run.py perf-takehome` (`perf_takehome_task.py` on `scored_artifact.py`; `benchmarks/perf-takehome/`,
`benchmarks/scored-artifact/`).

Anthropic's [original performance take-home](https://github.com/anthropics/original_performance_takehome) at
commit `5452f74bd977807ac2e74f3d29432b9df6f25197`: make `KernelBuilder.build_kernel` in `perf_takehome.py` run in
as few cycles as possible on a simulated VLIW SIMD machine, starting from the scalar starter (147,734 cycles). The
repository has no license file, so this repository holds none of its code: the first run clones it into
`.cache/perf-takehome` and checks every file against pinned hashes.

- the scorer (`benchmarks/scored-artifact/scorer/server.py`) builds each submitted kernel in a sandbox and runs it
  on its own pristine copy of the problem: valid if correct on 9 fixed and 8 fresh random inputs; the cycle count
  is the worst of the 9 fixed ones. The reward is 147,734 divided by the cycles of the best valid submission,
  graded again at the end; the post reports the cycles.
- resources: 2 CPUs and 6 GB per agent; a team of N gets N times both.

## BrowseComp

See [browsecomp.md](browsecomp.md).
