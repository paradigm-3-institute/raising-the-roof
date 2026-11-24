#!/usr/bin/env python3
"""The trusted ALE-Bench judge of an ale-bench task (see ale_bench_task.py).

It runs in its own container (the task's `scorer` service) next to the agents' one and
grades with ALE-Bench's own code, the `ale_bench` package at the pinned commit: a
Session per problem (`ale_bench.data.load_problem` from the pinned dataset revision, the
problem's Rust tools prebuilt from the same sources, public and private inputs from the
problem's generator), whose `public_eval` and `private_eval` compile and run every case in
ALE-Bench's language images exactly as upstream does: one container per step, 1 CPU,
2 GiB, no network, the problem's time and memory limits, the tester's score. The judge
has the host's Docker socket for that; the agents' container never sees it, the dataset,
the private seeds or this directory, and reaches the judge only through its HTTP API.

The queue, the submission log and the HTTP API are the scored-artifact layer's
(benchmarks/scored-artifact/scorer/server.py, copied next to this file unchanged): agents
`submit` their program, submissions are queued round robin across agents with a few
waiting per agent, and every graded submission goes to /scorer/state/submissions.jsonl
(<trial>/ale-bench/ on the host). What agents see of a submission is its public result
only: the overall judge result, the public score and upstream's feedback text.

Timing: every container ALE-Bench starts is pinned to one of the judge's CPUs
(`judge_cpus`) and holds that CPU's machine-wide lock (a file in `lock_dir`, shared by
every ale-bench judge on the host) while it runs, so no two judge steps of any ale-bench
task ever share a CPU. Each evaluation records the host's load and how busy its CPUs were.
Nothing a case does waits on the host's disk (on a host whose disk other runs saturate, a case
can otherwise hold its CPU for 30-60 s instead of ~4 s):
- ALE-Bench ends each run command with `; sync`, a flush of every filesystem on the host,
  after the timed program; the tester and the judge read the outputs through the host's
  page cache (bind mounts, one kernel), so the judge runs `; true` there instead (the
  command's exit status stays 0, as with sync);
- with `docker_host` in the config (an optional Docker daemon whose state lives in RAM),
  ALE-Bench's containers run there; same images,
  limits, pinning and cgroup placement, only Docker's own bookkeeping is off the disk.
  hardware.json records which daemon the judge used.

Three lanes:
- public: each submission's `public_eval` (the agents' feedback), in submission order;
- private (for the analysis, never shown to agents): `private_eval` of the run's current
  final candidate, the best public-ACCEPTED submission so far, with one worker, whenever
  that candidate changes (a newer candidate replaces one not yet started); results go to
  private.jsonl, and give the score-over-tokens curves the AtCoder performance the run
  would get if it ended then;
- final (the verifier's /finalize): the best public-ACCEPTED submission (ties: the
  earliest), as ALE-Bench's evaluation selects it, graded by `private_eval`: raw score,
  rank, performance. That is the reward (final.json).

The non-official `outputs` variant grades output files for fixed inputs with the
problem's tester (ALE-Bench's batch judge step) instead of running a program.

    judge.py --config config.json                                   # serve
    judge.py --config config.json grade [--public] [--private] FILE...  # grade programs once
    judge.py --config config.json direct FILE...                    # ale_bench.start() + private_eval, unmodified
"""

from __future__ import annotations

import argparse
import base64
import collections
import concurrent.futures
import datetime as dt
import fcntl
import json
import os
import platform
import re
import secrets
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from server import Scorer, handler, now_iso  # noqa: E402  (the scored-artifact layer)

PUBLIC_FIELDS = ("sid", "agent_id", "status", "valid", "public_score", "public_judge", "detail", "submitted_ts",
                 "ts", "sha256", "final", "note", "position", "language", "variant_score")
STDERR_TAIL = 4000  # characters of a failing case's standard error shown to the agent
ACCEPTED = "ACCEPTED"
# ALE-Bench turns an exception in its own judging code into an INTERNAL_ERROR case. Such a
# case is first run again (trace_internal_errors); an evaluation that still has one is run
# again as a whole, and a public one that fails again is not counted against the agent.
INTERNAL_ERROR_RETRIES = 2
# The private lane's thread; once the run ends, its evaluation in progress is cancelled
# (at its next container) so the final grading has the judge's CPUs to itself.
LANE_THREAD = "ale-bench-private-lane"
LANE_CANCEL = threading.Event()


class LaneCancelled(Exception):
    """The run ended while the private lane was grading."""


# ----------------------------------------------------------------------------
# Pure helpers (tested without ale_bench)


def better(a: float, b: float | None, score_type: str) -> bool:
    """Whether public score a beats b for a maximize or minimize problem."""
    if b is None:
        return True
    return a > b if score_type == "maximize" else a < b


def final_candidates(records, score_type: str) -> list[dict]:
    """Graded, public-ACCEPTED submissions, best public score first, ties by submission
    order: ALE-Bench's selection (the code that is ACCEPTED on the public cases with the
    best total score; ale_bench_eval's argmax/argmin keeps the first of equal scores)."""
    valid = [r for r in records if r.get("status") == "done" and r.get("valid")
             and r.get("public_score") is not None]
    sign = -1 if score_type == "maximize" else 1
    return sorted(valid, key=lambda r: (sign * float(r["public_score"]), int(r["sid"])))


def verdict_counts(case_results) -> dict[str, int]:
    counts = collections.Counter(_verdict(case) for case in case_results)
    return dict(sorted(counts.items()))


def _value(judge) -> str:
    """A JudgeResult's name (or the string itself)."""
    return getattr(judge, "value", judge)


def _verdict(case) -> str:
    return _value(case.judge_result)


def feedback(overall_judge, overall_absolute_score, case_results, time_limit: float) -> str:
    """The public result as ale_bench_eval shows it to models (prompts/builder.py,
    result_feedback), plus the longest execution time; a failing case's standard error
    is cut to its last STDERR_TAIL characters."""
    overall = getattr(overall_judge, "value", overall_judge)
    text = f"[Public test result]\nOverall judge result: {overall}\n"
    cases = list(case_results)
    if overall == ACCEPTED:
        text += f"Overall absolute score: {overall_absolute_score}\n"
        text += "\n".join(f"- Case {i}: {case.absolute_score}" for i, case in enumerate(cases, 1))
    else:
        index = next((i for i, case in enumerate(cases) if _verdict(case) == overall), 0)
        case = cases[index]
        error = case.error_str
        if error is not None and len(error) > STDERR_TAIL:
            error = "..." + error[-STDERR_TAIL:]
        text += (f"- Case {index + 1}:\n"
                 f"    Absolute score: {case.absolute_score}\n"
                 f"    Execution time: {case.execution_time:.3f} sec\n"
                 f"    Memory usage: {case.memory_usage // 1024 // 1024} MB\n"
                 f"    Standard error: \"{error}\"\n"
                 f"    Message: \"{case.message}\"")
    if cases:
        longest = max(range(len(cases)), key=lambda i: cases[i].execution_time)
        text += (f"\nLongest execution time: {cases[longest].execution_time:.3f} sec "
                 f"(case {longest + 1}; time limit {time_limit:g} sec)")
    return text


def loadavg() -> float | None:
    try:
        return float(Path("/proc/loadavg").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def cpu_times(cpus) -> dict[int, tuple[int, int]]:
    """(busy, total) jiffies of each CPU from /proc/stat (host-wide in a container)."""
    found = {}
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            match = re.match(r"cpu(\d+)\s+(.*)", line)
            if match and int(match.group(1)) in cpus:
                values = [int(v) for v in match.group(2).split()]
                idle = values[3] + (values[4] if len(values) > 4 else 0)
                found[int(match.group(1))] = (sum(values[:8]) - idle, sum(values[:8]))
    except OSError:
        pass
    return found


class LoadProbe:
    """The host's load and the judge CPUs' busy fraction over one evaluation (our own
    judge steps included)."""

    def __init__(self, cpus) -> None:
        self.cpus = list(cpus)
        self.load = loadavg()
        self.times = cpu_times(self.cpus)
        self.logs = collections.Counter(LOG_STATS)
        self.start = time.monotonic()

    def done(self) -> dict:
        after = cpu_times(self.cpus)
        busy = {}
        for cpu, (b0, t0) in self.times.items():
            b1, t1 = after.get(cpu, (b0, t0))
            busy[str(cpu)] = round((b1 - b0) / (t1 - t0), 3) if t1 > t0 else None
        return {"cpus": self.cpus, "loadavg_start": self.load, "loadavg_end": loadavg(), "cpu_busy": busy,
                "seconds": round(time.monotonic() - self.start, 3),
                "docker_logs": dict(collections.Counter(LOG_STATS) - self.logs)}


# ----------------------------------------------------------------------------
# The judge's CPUs: one ALE-Bench container per CPU, machine-wide


class CorePool:
    """Hands out the judge's CPUs, one container at a time each, across every ale-bench
    judge on the host (flock on lock_dir/ale-bench-cpu<N>.lock; the lock is released when
    ALE-Bench removes the container)."""

    def __init__(self, cpus, lock_dir: Path) -> None:
        self.cpus = [int(cpu) for cpu in cpus]
        self.lock_dir = Path(lock_dir)
        self.guard = threading.Lock()
        self.held: set[int] = set()

    def acquire(self) -> tuple[int, int]:
        while True:
            with self.guard:
                for cpu in self.cpus:
                    if cpu in self.held:
                        continue
                    fd = os.open(self.lock_dir / f"ale-bench-cpu{cpu}.lock", os.O_RDWR | os.O_CREAT, 0o666)
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        os.close(fd)
                        continue
                    self.held.add(cpu)
                    return cpu, fd
            time.sleep(0.05)

    def release(self, cpu: int, fd: int) -> None:
        with self.guard:
            self.held.discard(cpu)
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)


# Containers that run the problem's tester (ALE-Bench's TESTER_BIN): their standard error
# always holds the tester's verdict ("Score = N" or an error). On a busy host Docker has
# returned such a finished container's log empty, which ALE-Bench reports as a
# RUNTIME_ERROR (interactive problems) or an exception (batch problems) for a correct case.
# An empty log of these containers is read again for up to LOG_RETRY_SECONDS; if it stays
# empty, the case is run again (trace_internal_errors).
TESTER_IN_CONTAINER = "/judge/target/release/tester"
LOG_RETRY_SECONDS = 2.0
LOG_STATS = collections.Counter()


class TesterLogLost(Exception):
    """Docker returned an empty log for a container that ran the tester."""


UPSTREAM_SYNC = "; sync"  # how ALE-Bench's run commands end (case_runner.build_*_command)


def without_sync(command):
    """ALE-Bench's `bash -c` command with its trailing `; sync` (a flush of every
    filesystem on the host, after the timed program) replaced by `; true`: the outputs
    reach the tester through the host's page cache anyway, and the exit status stays 0."""
    if (isinstance(command, (list, tuple)) and command and isinstance(command[-1], str)
            and command[-1].endswith(UPSTREAM_SYNC)):
        return [*command[:-1], command[-1][: -len(UPSTREAM_SYNC)] + "; true"]
    return command


def pin_containers(pool: CorePool, labels: dict[str, str], name_prefix: str) -> None:
    """Every container ALE-Bench starts (docker-py's ContainerCollection.run) runs on one
    CPU of the pool, which it holds until ALE-Bench removes it (each of its steps does,
    in a finally block); upstream's own limits (1 CPU quota, 2 GiB, no network) stay."""
    import docker.models.containers as containers

    original_run = containers.ContainerCollection.run
    counter = iter(range(1, 1 << 62))
    process = secrets.token_hex(4)

    def run(self, image, command=None, stdout=True, stderr=False, remove=False, **kwargs):
        if LANE_CANCEL.is_set() and threading.current_thread().name == LANE_THREAD:
            raise LaneCancelled("the run ended")
        command = without_sync(command)
        cpu, fd = pool.acquire()
        kwargs["cpuset_cpus"] = str(cpu)
        kwargs["labels"] = {**(kwargs.get("labels") or {}), **labels}
        kwargs.setdefault("name", f"{name_prefix}-{process}-{next(counter)}")
        try:
            container = original_run(self, image, command, stdout, stderr, remove, **kwargs)
        except BaseException:
            pool.release(cpu, fd)
            raise
        if not hasattr(container, "remove"):
            pool.release(cpu, fd)
            return container
        original_remove = container.remove
        released = threading.Event()

        def remove_and_release(*args, **kw):
            try:
                return original_remove(*args, **kw)
            finally:
                if not released.is_set():
                    released.set()
                    pool.release(cpu, fd)

        container.remove = remove_and_release
        joined = " ".join(command) if isinstance(command, (list, tuple)) else str(command or "")
        if TESTER_IN_CONTAINER in joined and hasattr(container, "logs"):
            original_logs = container.logs

            def logs_again_if_empty(*args, **kw):
                output = original_logs(*args, **kw)
                if output.strip():
                    return output
                LOG_STATS["empty"] += 1
                deadline = time.monotonic() + LOG_RETRY_SECONDS
                delay = 0.05
                while time.monotonic() < deadline:
                    time.sleep(delay)
                    delay = min(delay * 2, 0.5)
                    output = original_logs(*args, **kw)
                    if output.strip():
                        LOG_STATS["recovered"] += 1
                        return output
                LOG_STATS["lost"] += 1
                raise TesterLogLost("Docker returned an empty log for a tester container")

            container.logs = logs_again_if_empty
        return container

    containers.ContainerCollection.run = run


INTERNAL_TRACEBACKS: collections.deque = collections.deque(maxlen=20)
CASE_RETRIES = 2  # a case whose judging raised is run again, from scratch, this often


def trace_internal_errors() -> None:
    """Run a case again when ALE-Bench's judging of it raised (the program's behaviour is
    a verdict, never an exception: an exception is the judge's own failure, such as a
    tester log Docker lost), and keep the traceback, which run_cases reduces to an
    INTERNAL_ERROR message (or, with one worker, lets end the whole evaluation)."""
    import ale_bench.tool_wrappers.case_runner as case_runner

    original = case_runner.case_iter_func

    def traced(*args, **kwargs):
        for attempt in range(1 + CASE_RETRIES):
            try:
                return original(*args, **kwargs)
            except LaneCancelled:
                raise
            except Exception:
                INTERNAL_TRACEBACKS.append({"ts": now_iso(), "attempt": attempt + 1,
                                            "traceback": traceback.format_exc()[-3000:]})
                if attempt == CASE_RETRIES:
                    raise
                LOG_STATS["case_reruns"] += 1

    case_runner.case_iter_func = traced


# ----------------------------------------------------------------------------
# ALE-Bench sessions


def open_session(config: dict, num_workers: int, lite_version: bool | None = None):
    """A Session as ale_bench.start() makes one, for the pinned dataset (ALE_BENCH_DATA)
    and with the problem's tools prebuilt by upstream's build step from the same sources,
    without start()'s limit of one private evaluation and its session duration."""
    from ale_bench.data import load_problem
    from ale_bench.result import ResourceUsage
    from ale_bench.session import Session

    lite = config["lite_version"] if lite_version is None else lite_version
    problem, seeds, standings, rank_performance_map, data_root = load_problem(config["problem_id"], lite_version=lite)
    install_tools(config, data_root)
    huge = 10**18
    session = Session(
        problem=problem, lite_version=lite, public_seeds=seeds.public, private_seeds=seeds.private,
        standings=standings, rank_performance_map=rank_performance_map, tool_dir=data_root,
        use_same_time_scale=False,
        maximum_resource_usage=ResourceUsage(num_case_gen=huge, num_case_eval=huge, num_call_public_eval=huge,
                                             num_call_private_eval=1, execution_time_case_eval=1e18),
        session_duration=dt.timedelta(days=3650), num_workers=num_workers, visualization_server_port=None)
    return session, seeds


def install_tools(config: dict, data_root: Path) -> None:
    """The prebuilt gen/tester/vis (built from this problem's tools/ by upstream's
    build_rust_tools, `cargo build --release` in rust:1.79.0-buster) where a session
    expects them, after checking they were built from these sources."""
    import hashlib

    tools = Path(config["tools_dir"])
    manifest = json.loads((tools / "manifest.json").read_text())
    digest = hashlib.sha256()
    for path in sorted(p for p in (data_root / "tools" / "src").rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(data_root / "tools")).encode() + b"\0" + path.read_bytes())
    if digest.hexdigest() != manifest["sources_sha256"]:
        raise RuntimeError(f"the prebuilt tools were not built from {config['problem_id']}'s tools/src")
    release = data_root / "tools" / "target" / "release"
    release.mkdir(parents=True, exist_ok=True)
    for name in manifest["binaries"]:
        shutil.copyfile(tools / name, release / name)
        (release / name).chmod(0o755)


def reset_private_count(session) -> None:
    """Allow another private_eval on this session, as ale_bench_eval/evaluate.py does
    between the solutions it evaluates."""
    from ale_bench.result import ResourceUsage

    usage = session.current_resource_usage
    session._current_resource_usage = ResourceUsage(**{**usage.model_dump(), "num_call_private_eval": 0})  # noqa: SLF001


class Evaluator:
    """ALE-Bench sessions of one problem: one for public and final evaluations (the
    judge's workers), one with a single worker for the private lane."""

    def __init__(self, config: dict, lane: bool) -> None:
        self.config = config
        self.language = config["language"]
        self.judge_version = config["judge_version"]
        self.cpus = config["judge_cpus"]
        self.workers = int(config.get("judge_workers") or len(self.cpus))
        self.variant = config.get("variant", "official")
        self.session, self.seeds = open_session(config, self.workers)
        self.problem = self.session.problem
        self.score_type = self.problem.metadata.score_type.value
        self.time_limit = self.problem.constraints.time_limit
        self.lane_session = open_session(config, 1)[0] if lane and self.variant == "official" else None
        self.variant_inputs = None
        if self.variant == "outputs":
            from ale_bench.tool_wrappers import generate_inputs
            self.variant_inputs = generate_inputs(config["variant_seeds"], {}, self.session.tool_dir)

    def close(self) -> None:
        for session in (self.session, self.lane_session):
            if session is not None:
                session.close()

    def public(self, code: str) -> tuple[dict, dict]:
        """(log fields, details for result.json) of the official public evaluation."""
        from ale_bench.error import AleBenchError

        probe = LoadProbe(self.cpus)
        for attempt in range(1 + INTERNAL_ERROR_RETRIES):
            try:
                result = self.session.public_eval(code, code_language=self.language,
                                                  judge_version=self.judge_version)
            except AleBenchError as exc:  # empty or oversized code, for example
                return ({"status": "done", "valid": False, "public_score": None, "public_judge": "REJECTED",
                         "detail": f"Submission rejected: {exc}", "judge": probe.done()}, {})
            if _value(result.overall_judge_result) != "INTERNAL_ERROR":
                break
        cases = list(result.case_results)
        if _value(result.overall_judge_result) == "INTERNAL_ERROR":
            # The judge's own failure (Docker under load), not the program's: not counted.
            return ({"status": "error", "valid": False, "public_score": None, "public_judge": "INTERNAL_ERROR",
                     "detail": "the judge failed while grading (an internal error, not counted against you); "
                               "submit again", "judge": probe.done(), "attempts": attempt + 1,
                     "info": {"tracebacks": list(INTERNAL_TRACEBACKS)[-3:],
                              "messages": [case.message[-300:] for case in cases
                                           if _verdict(case) == "INTERNAL_ERROR"][:3]}}, {})
        fields = {
            "status": "done", "valid": _value(result.overall_judge_result) == ACCEPTED,
            "public_score": result.overall_absolute_score, "public_judge": _value(result.overall_judge_result),
            "public_cases": verdict_counts(cases), "n_public_cases": len(cases),
            "max_time": max((case.execution_time for case in cases), default=None),
            "detail": feedback(result.overall_judge_result, result.overall_absolute_score, cases,
                               self.time_limit),
            "judge": probe.done(),
        }
        details = {"case_scores": [case.absolute_score for case in cases],
                   "case_verdicts": [_verdict(case) for case in cases],
                   "case_times": [round(case.execution_time, 4) for case in cases],
                   "case_memory": [case.memory_usage for case in cases]}
        return fields, details

    def private(self, code: str, lane: bool = False, attempts: int = 1 + INTERNAL_ERROR_RETRIES) -> dict:
        """The official private evaluation: raw score, rank and performance (again, up to
        `attempts` times, while a case ends in the judge's own INTERNAL_ERROR)."""
        session = self.lane_session if lane else self.session
        probe = LoadProbe(self.cpus)
        for attempt in range(attempts):
            reset_private_count(session)
            result, rank, performance = session.private_eval(code, code_language=self.language,
                                                             judge_version=self.judge_version)
            if not any(_verdict(case) == "INTERNAL_ERROR" for case in result.case_results):
                break
        cases = list(result.case_results)
        counts = verdict_counts(cases)
        return {
            "status": "error" if counts.get("INTERNAL_ERROR") else "done",
            "private_judge": _value(result.overall_judge_result), "private_cases": counts,
            "n_private_cases": len(cases), "private_absolute_score": result.overall_absolute_score,
            "private_relative_score": result.overall_relative_score,
            # The standings compare relative scores where the problem has them (it then
            # sums relative scores; otherwise these equal the absolute scores).
            "private_score": (result.overall_relative_score if result.overall_relative_score is not None
                              else result.overall_absolute_score),
            "rank": rank, "performance": performance, "attempts": attempt + 1,
            "max_time": max((case.execution_time for case in cases), default=None),
            # How fragile the grade is on a busy host: cases within 10% of the time limit.
            "time_limit": self.time_limit,
            "near_limit_cases": sum(case.execution_time >= 0.9 * self.time_limit for case in cases),
            # The first failing cases' messages, for the log (never shown to agents).
            "failures": [{"case": i + 1, "verdict": _verdict(case), "message": case.message[-500:]}
                         for i, case in enumerate(cases) if _verdict(case) != ACCEPTED][:5],
            "internal_tracebacks": (list(INTERNAL_TRACEBACKS) if counts.get("INTERNAL_ERROR") else []),
            "case_absolute_scores": [case.absolute_score for case in cases],
            "case_verdicts": [_verdict(case) for case in cases],
            "case_times": [round(case.execution_time, 4) for case in cases],
            "judge": probe.done(),
        }

    def outputs(self, files: dict[str, bytes]) -> tuple[dict, dict]:
        """The `outputs` variant: each output file scored by the problem's tester against
        its fixed input, with ALE-Bench's batch judge step."""
        from ale_bench.result import CaseResult
        from ale_bench.tool_wrappers.case_runner import (HostPathsBatchJudge, build_batch_judge_command,
                                                         get_batch_judge_volumes, run_batch_judge_container)

        probe = LoadProbe(self.cpus)
        names = self.config["artifact"]["files"]
        scores: list[int | None] = [None] * len(names)
        messages: list[str] = [""] * len(names)
        with tempfile.TemporaryDirectory(prefix="outputs-") as scratch:
            def judge(index: int) -> None:
                case = Path(scratch) / f"{index:04d}"
                case.mkdir()
                (case / "input.txt").write_text(self.variant_inputs[index])
                (case / "output.txt").write_bytes(files[names[index]])
                (case / "profiles.json").touch()
                paths = HostPathsBatchJudge(input_file=case / "input.txt", output_file=case / "output.txt",
                                            profiles_file=case / "profiles.json")
                outcome = run_batch_judge_container(get_batch_judge_volumes(paths, self.session.tool_dir),
                                                    build_batch_judge_command(), 0.0, None, None, None)
                if isinstance(outcome, CaseResult):
                    messages[index] = outcome.message[-500:]
                else:
                    scores[index] = int(outcome)

            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(judge, range(len(names))))
        failed = [i for i, score in enumerate(scores) if score is None]
        total = sum(score for score in scores if score is not None)
        valid = not failed
        variant_score = None
        if valid:
            variant_score = float(total) if self.score_type == "maximize" else (
                len(names) * 1e9 / total if total > 0 else None)
            valid = variant_score is not None
        detail = (f"{len(names) - len(failed)} of {len(names)} outputs scored; total absolute score {total}"
                  + (f"; first failure {names[failed[0]]}: {messages[failed[0]]}" if failed else ""))
        fields = {"status": "done", "valid": valid, "public_score": total if valid else None,
                  "public_judge": ACCEPTED if not failed else "WRONG_ANSWER", "variant_score": variant_score,
                  "detail": detail, "judge": probe.done()}
        return fields, {"case_scores": scores, "messages": messages}


# ----------------------------------------------------------------------------
# The service


class AleJudge(Scorer):
    """The scored-artifact scorer (queue, log, HTTP API) with ALE-Bench grading."""

    def __init__(self, config: dict, evaluator: Evaluator | None) -> None:
        self.evaluator = evaluator
        self.score_type = config["score_type"]
        self.variant = config.get("variant", "official")
        self.lane_best: dict | None = None
        self.lane_done: set[str] = set()
        self.lane_running: str | None = None
        super().__init__(config)
        self._restore_lane()

    # -- what agents see ---------------------------------------------------

    def public(self, record: dict) -> dict:
        shown = {key: record[key] for key in PUBLIC_FIELDS if key in record}
        if record.get("status") == "queued":
            shown["position"] = self.position(record["sid"])
        return shown

    # -- the public lane ----------------------------------------------------

    def grade_submission(self, sid: str) -> tuple[dict, dict]:
        files_dir = self.state / "submissions" / sid / "files"
        if self.variant == "outputs":
            files = {name: (files_dir / name).read_bytes() for name in self.artifact["files"]}
            return self.evaluator.outputs(files)
        code = (files_dir / self.artifact["files"][0]).read_text(errors="replace")
        return self.evaluator.public(code)

    def worker(self, index: int) -> None:
        while True:
            with self.lock:
                sid = self._next()
                while sid is None:
                    self.lock.wait()
                    sid = self._next()
                record = self.records[sid]
                record["status"] = "running"
                record["started_ts"] = now_iso()
                self.running += 1
            try:
                fields, details = self.grade_submission(sid)
            except Exception:  # noqa: BLE001 - a judge failure, not the agent's
                fields = {"status": "error", "valid": False, "public_score": None,
                          "detail": "the judge failed while grading (not counted against you)",
                          "info": {"traceback": traceback.format_exc()[-4000:]}}
                details = {}
            with self.lock:
                record.update(fields)
                record["ts"] = now_iso()
                record["language"] = self.config["language"]
                self.running -= 1
                self._write(self.state / "submissions" / sid / "result.json",
                            json.dumps({**record, **details}, indent=2, sort_keys=True) + "\n")
                self._append_log(record)
                self._consider_for_lane(record)
                self.lock.notify_all()

    # -- the private lane (analysis only) -----------------------------------

    def _consider_for_lane(self, record: dict) -> None:
        if (self.variant != "official" or record.get("final") or record.get("status") != "done"
                or not record.get("valid") or record.get("public_score") is None):
            return
        best = self.lane_best
        if best is None or better(float(record["public_score"]), float(best["public_score"]), self.score_type):
            self.lane_best = record

    def _restore_lane(self) -> None:
        for entry in self._read_jsonl(self.state / "private.jsonl"):
            self.lane_done.add(str(entry.get("sid")))
        for record in sorted(self.records.values(), key=lambda r: int(r["sid"])):
            self._consider_for_lane(record)

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        rows = []
        if path.is_file():
            for line in path.read_text(errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    def lane(self) -> None:
        while True:
            with self.lock:
                while not self.closed and (self.lane_best is None or self.lane_best["sid"] in self.lane_done):
                    self.lock.wait()
                if self.closed:
                    return
                target = self.lane_best
                self.lane_running = target["sid"]
            started = now_iso()
            code = (self.state / "submissions" / target["sid"] / "files" / self.artifact["files"][0]).read_text(
                errors="replace")
            try:
                result = self.evaluator.private(code, lane=True)
            except LaneCancelled:
                result = {"status": "cancelled"}
            except Exception:  # noqa: BLE001
                result = {"status": "error", "traceback": traceback.format_exc()[-4000:]}
            entry = {"sid": target["sid"], "agent_id": target["agent_id"], "submitted_ts": target.get("submitted_ts"),
                     "public_ts": target.get("ts"), "public_score": target.get("public_score"), "started_ts": started,
                     "ts": now_iso(), "lane": "private", **result}
            with self.lock:
                with (self.state / "private.jsonl").open("a") as stream:
                    stream.write(json.dumps(entry, sort_keys=True) + "\n")
                self._own(self.state / "private.jsonl")
                self.lane_done.add(target["sid"])  # after its log line, which readers may wait for
                self.lane_running = None
                self.lock.notify_all()

    # -- the end of the run -------------------------------------------------

    def finalize(self) -> dict:
        with self.lock:
            if self.final is not None:
                return self.final
            self.closed = True
            LANE_CANCEL.set()
            self.lock.notify_all()
            while any(self.queues.values()) or self.running:
                self.lock.wait()
            ranked = final_candidates(list(self.records.values()), self.score_type)
        if self.variant == "outputs":
            result = self._finalize_outputs(ranked)
        else:
            result = self._finalize_official(ranked)
        result.update(ts=now_iso(), submissions=len(self.records), valid_submissions=len(ranked),
                      problem_id=self.config["problem_id"], lite_version=self.config["lite_version"],
                      language=self.config["language"], variant=self.variant)
        with self.lock:
            self.final = result
            self._write(self.state / "final.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result

    def _log_final(self, entry: dict) -> None:
        with (self.state / "final-evaluations.jsonl").open("a") as stream:
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
        self._own(self.state / "final-evaluations.jsonl")

    def _finalize_official(self, ranked: list[dict]) -> dict:
        tried = []
        for record in ranked[:int(self.config.get("final_candidates", 3))]:
            code = (self.state / "submissions" / record["sid"] / "files" / self.artifact["files"][0]).read_text(
                errors="replace")
            for attempt in range(2):  # an exception in the judge is retried once
                try:
                    result = self.evaluator.private(code)  # INTERNAL_ERROR cases: graded again
                except Exception:  # noqa: BLE001
                    result = {"status": "error", "traceback": traceback.format_exc()[-4000:]}
                entry = {"sid": record["sid"], "agent_id": record["agent_id"], "call": attempt + 1,
                         "public_score": record.get("public_score"), "ts": now_iso(), **result}
                tried.append(entry)
                self._log_final(entry)
                if result.get("status") == "done":
                    return {"valid": True, "score": result["performance"], "metric_name": "private_score",
                            "metric": result["private_score"], "sid": record["sid"],
                            "agent_id": record["agent_id"], "public_score": record.get("public_score"),
                            "candidates_tried": len(tried),
                            **{key: result.get(key) for key in (
                                "performance", "rank", "private_score", "private_absolute_score",
                                "private_relative_score", "private_judge", "private_cases", "n_private_cases",
                                "max_time", "time_limit", "near_limit_cases", "judge")}}
        # No public-ACCEPTED submission: the grade of a program that does not compile
        # (upstream's evaluation submits ale_bench_eval's compilation-error code then).
        try:
            result = self.evaluator.private(ce_code(self.config["language"]))
            self._log_final({"sid": None, "ts": now_iso(), "note": "compilation-error code", **result})
            performance, rank = result.get("performance"), result.get("rank")
        except Exception:  # noqa: BLE001
            performance, rank = None, None
        return {"valid": False, "score": performance, "performance": performance, "rank": rank,
                "metric_name": "private_score", "metric": 0, "sid": None, "agent_id": None,
                "candidates_tried": len(tried),
                "detail": "no public-ACCEPTED submission" if not ranked else "every candidate's grading failed"}

    def _finalize_outputs(self, ranked: list[dict]) -> dict:
        ordered = sorted(ranked, key=lambda r: (-float(r.get("variant_score") or 0), int(r["sid"])))
        for record in ordered[:int(self.config.get("final_candidates", 3))]:
            files_dir = self.state / "submissions" / record["sid"] / "files"
            files = {name: (files_dir / name).read_bytes() for name in self.artifact["files"]}
            fields, _ = self.evaluator.outputs(files)
            self._log_final({"sid": record["sid"], "ts": now_iso(), **fields})
            if fields.get("valid"):
                return {"valid": True, "score": fields["variant_score"], "metric_name": "total_absolute_score",
                        "metric": fields["public_score"], "sid": record["sid"], "agent_id": record["agent_id"],
                        "original_score": record.get("variant_score")}
        return {"valid": False, "score": 0.0, "metric_name": "total_absolute_score", "metric": None, "sid": None,
                "agent_id": None, "detail": "no valid submission"}


def ce_code(language: str) -> str:
    """ale_bench_eval's compilation-error program for the language."""
    import ale_bench

    codes = Path(ale_bench.__file__).resolve().parent.parent / "ale_bench_eval" / "codes"
    found = sorted(codes.glob(f"ce_{language}.*"))
    return found[0].read_text() if found else "this code intentionally fails"


# ----------------------------------------------------------------------------
# Setup and command line


def drop_privileges(config: dict) -> None:
    """Started as root to own the state directory; then the host user's uid, with the
    Docker socket's group, so ALE-Bench's containers run as that user (as upstream runs
    them, with images built for it) and every file left on the host is theirs."""
    if os.geteuid() != 0:
        return
    uid, gid = int(config["host_uid"]), int(config["host_gid"])
    state = Path(config.get("state_dir", "/scorer/state"))
    state.mkdir(parents=True, exist_ok=True)
    for path in [state, *state.rglob("*")]:
        os.lchown(path, uid, gid)
    os.setgroups([int(config["docker_gid"])])
    os.setgid(gid)
    os.setuid(uid)
    os.environ["HOME"] = os.environ.get("TMPDIR", "/tmp")


def use_docker_host(config: dict) -> None:
    """ALE-Bench's containers on the config's `docker_host` (unix:///PATH) when that socket
    exists, else on the host's main daemon (the socket at /var/run/docker.sock)."""
    host = str(config.get("docker_host") or "")
    if host.startswith("unix://") and Path(host[len("unix://"):]).exists():
        os.environ["DOCKER_HOST"] = host
    elif host:
        print(f"docker_host {host} not found; using the main Docker daemon", file=sys.stderr, flush=True)


def hardware(config: dict, evaluator: Evaluator | None) -> dict:
    info = {"ts": now_iso(), "judge_cpus": config["judge_cpus"], "machine": platform.machine(),
            "kernel": platform.release(), "cpu_model": None, "logical_cpus": os.cpu_count(),
            "ale_bench_commit": config.get("ale_bench_commit"), "dataset_revision": config.get("dataset_revision"),
            "docker_host": os.environ.get("DOCKER_HOST", "unix:///var/run/docker.sock"),
            "run_command_ends": "; true (ALE-Bench's ; sync dropped)", "images": {}}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        import docker

        client = docker.from_env()
        info["docker_server"] = client.version().get("Version")
        for image in config.get("images", []):
            try:
                info["images"][image] = client.images.get(image).id
            except Exception:  # noqa: BLE001
                info["images"][image] = None
        client.close()
    except Exception:  # noqa: BLE001
        pass
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("/scorer/app/config.json"))
    commands = parser.add_subparsers(dest="command")
    once = commands.add_parser("grade", help="grade programs once and print JSON lines")
    once.add_argument("--public", action="store_true")
    once.add_argument("--private", action="store_true")
    once.add_argument("--repeats", type=int, default=1)
    once.add_argument("files", nargs="+", type=Path)
    direct = commands.add_parser("direct", help="ale_bench.start() and private_eval, unmodified")
    direct.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    drop_privileges(config)
    use_docker_host(config)
    if args.command == "direct":
        # The package used directly: upstream's start() (its own tool build and input
        # generation) and one private_eval per session, with no pinning or locks.
        import ale_bench

        trace_internal_errors()
        for path in args.files:
            session = ale_bench.start(config["problem_id"], lite_version=config["lite_version"],
                                      num_workers=len(config["judge_cpus"]))
            result, rank, performance = session.private_eval(path.read_text(), code_language=config["language"],
                                                             judge_version=config["judge_version"])
            print(json.dumps({"file": str(path), "private_absolute_score": result.overall_absolute_score,
                              "private_relative_score": result.overall_relative_score,
                              "private_judge": _value(result.overall_judge_result), "rank": rank,
                              "performance": performance,
                              "private_cases": verdict_counts(result.case_results),
                              "failures": [{"case": i + 1, "verdict": _verdict(case), "message": case.message[-500:]}
                                           for i, case in enumerate(result.case_results)
                                           if _verdict(case) != ACCEPTED][:5],
                              "internal_tracebacks": list(INTERNAL_TRACEBACKS)}), flush=True)
            session.close()
        return 0
    trace_internal_errors()
    pin_containers(CorePool(config["judge_cpus"], Path(config["lock_dir"])),
                   {"ttc.ale-bench": "judge", "ttc.ale-bench.task": str(config.get("task_id", ""))},
                   str(config.get("container_prefix") or "ale-bench-judge"))
    if args.command == "grade":
        evaluator = Evaluator(config, lane=False)
        try:
            for path in args.files:
                for repeat in range(args.repeats):
                    row = {"file": str(path), "repeat": repeat + 1}
                    if config.get("variant") == "outputs":
                        files = {name: (path / name).read_bytes() for name in config["artifact"]["files"]}
                        row.update(evaluator.outputs(files)[0])
                    else:
                        code = path.read_text()
                        if args.public or not args.private:
                            fields, details = evaluator.public(code)
                            row["public"] = {**fields, **details}
                        if args.private:
                            row["private"] = evaluator.private(code)
                    print(json.dumps(row), flush=True)
        finally:
            evaluator.close()
        return 0
    evaluator = Evaluator(config, lane=bool(config.get("private_lane", True)))
    # Docker stops the container with SIGTERM: exit through atexit, where the sessions
    # remove their copies of the problem's data from the scratch directory.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    judge = AleJudge(config, evaluator)
    previous = judge.state / "hardware.json"
    if previous.is_file():  # a restarted judge keeps the earlier start's record
        previous.rename(judge.state / f"hardware.{int(previous.stat().st_mtime)}.json")
    judge._write(judge.state / "hardware.json", json.dumps(hardware(config, evaluator), indent=2) + "\n")
    for index in range(int(config.get("workers", 1))):
        threading.Thread(target=judge.worker, args=(index,), daemon=True).start()
    if evaluator.lane_session is not None:
        threading.Thread(target=judge.lane, name=LANE_THREAD, daemon=True).start()
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("0.0.0.0", int(config.get("port", 18090))),
                                 handler(judge, str(config.get("token") or "")))
    server.daemon_threads = True
    print(f"ALE-Bench judge for {config['problem_id']} listening on port {server.server_port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
