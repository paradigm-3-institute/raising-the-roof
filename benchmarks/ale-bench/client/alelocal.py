#!/usr/bin/env python3
"""ale-local: the problem's tools in your working copy, run the way ALE-Bench's judge runs them.

    ale-local gen SEED [SEED ...] [--dir DIR]       inputs from the problem's generator (tools/bin/gen)
    ale-local test [SOURCE] [--seeds 0-49] [--input FILE ...] [-j N] [--tl SEC] [--keep DIR]
                                                    compile SOURCE (default: your copy's program) with the
                                                    judge's command, run each case under the time limit,
                                                    score it with the problem's tester, print the total
    ale-local vis INPUT OUTPUT                      the problem's visualizer (vis.html or out.svg) and score

`test` is ALE-Bench's case_eval in your own container: the judge's compile and run
commands, GNU time, the time limit (max of CPU and wall time) and the memory limit, the
tester's "Score = N" (for interactive problems the tester runs your program). Times here
are measured in your container on your CPU share, not on the judge's dedicated CPUs, and
the judge's public cases are seeds 0-49 (lite: 0-4). Standard library only.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ale_bench/constants.py TIME_OUTPUT_FORMAT: the fields GNU time writes as JSON.
TIME_OUTPUT_FORMAT = (
    "{"
    '\\"command\\": \\"%C\\", '
    '\\"exit_status\\": \\"%x\\", '
    '\\"elapsed_time\\": \\"%E\\", '
    '\\"elapsed_time_seconds\\": \\"%e\\", '
    '\\"system_cpu_seconds\\": \\"%S\\", '
    '\\"user_cpu_seconds\\": \\"%U\\", '
    '\\"cpu_percent\\": \\"%P\\", '
    '\\"average_resident_set_size_kbytes\\": \\"%t\\", '
    '\\"max_resident_set_size_kbytes\\": \\"%M\\", '
    '\\"average_total_memory_kbytes\\": \\"%K\\", '
    '\\"signals_delivered\\": \\"%k\\", '
    '\\"page_size_bytes\\": \\"%Z\\", '
    '\\"minor_page_faults\\": \\"%R\\", '
    '\\"major_page_faults\\": \\"%F\\", '
    '\\"swaps\\": \\"%W\\", '
    '\\"file_system_inputs\\": \\"%I\\", '
    '\\"file_system_outputs\\": \\"%O\\", '
    '\\"average_shared_text_kbytes\\": \\"%X\\", '
    '\\"average_unshared_data_kbytes\\": \\"%D\\", '
    '\\"average_unshared_stack_kbytes\\": \\"%p\\", '
    '\\"voluntary_context_switches\\": \\"%w\\", '
    '\\"involuntary_context_switches\\": \\"%c\\", '
    '\\"socket_messages_sent\\": \\"%s\\", '
    '\\"socket_messages_received\\": \\"%r\\"'
    "}"
)
VERDICT_ORDER = ("INTERNAL_ERROR", "WRONG_ANSWER", "RUNTIME_ERROR", "TIME_LIMIT_EXCEEDED",
                 "MEMORY_LIMIT_EXCEEDED", "OUTPUT_LIMIT_EXCEEDED", "COMPILATION_ERROR")
COMPILE_TIMEOUT = 60


def workcopy() -> Path:
    """The working copy holding problem.json: the current directory or a parent, else
    /work/$AGENT_ID."""
    for directory in [Path.cwd(), *Path.cwd().parents]:
        if (directory / "problem.json").is_file():
            return directory
    fallback = Path("/work") / (os.environ.get("AGENT_ID") or "single")
    if (fallback / "problem.json").is_file():
        return fallback
    raise SystemExit("ale-local: no problem.json here or in /work/$AGENT_ID; run `workcopy` first")


def seeds_from(text: str) -> list[int]:
    found = []
    for part in text.split(","):
        first, _, last = part.strip().partition("-")
        found += list(range(int(first), int(last or first) + 1))
    return found


def generate(root: Path, seeds: list[int], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ale-gen-") as scratch:
        (Path(scratch) / "seeds.txt").write_text("".join(f"{seed}\n" for seed in seeds))
        subprocess.run([str(root / "tools" / "bin" / "gen"), "seeds.txt"], cwd=scratch, check=True,
                       stdout=subprocess.DEVNULL)
        for index, seed in enumerate(seeds):
            shutil.move(str(Path(scratch) / "in" / f"{index:04d}.txt"), directory / f"{seed:04d}.txt")


def compile_program(root: Path, problem: dict, source: Path, build: Path) -> str | None:
    """None, or the compiler's standard error."""
    template = Path(problem.get("workdir_template") or "/workdir")
    if template.is_dir():
        shutil.copytree(template, build, dirs_exist_ok=True, symlinks=True)
    target = build / problem["submission_file"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    try:
        done = subprocess.run(["/bin/bash", "--noprofile", "--norc", "-c", problem["compile_command"]], cwd=build,
                              capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
    except subprocess.TimeoutExpired:
        return f"compilation timed out ({COMPILE_TIMEOUT}s)"
    stderr = done.stderr.strip()
    product = build / problem["object_file"]
    interpreted = problem["language"] in ("python", "pypy")
    if (done.returncode != 0 or (not interpreted and (not product.is_file() or product.stat().st_size == 0))
            or (interpreted and "SyntaxError" in stderr)):
        return stderr or f"exit status {done.returncode}"
    return None


def run_command(problem: dict, time_limit: float, tester: Path | None, input_file: Path, output_file: Path,
                profiles: Path) -> str:
    """case_runner.build_batch_run_command / build_reactive_judge_command."""
    command = f"{problem['run_command']} < {shlex.quote(str(input_file))} > {shlex.quote(str(output_file))}"
    command = f'/usr/bin/time -f "{TIME_OUTPUT_FORMAT}" -o {shlex.quote(str(profiles))} {command}'
    if tester is not None:
        command = f"{shlex.quote(str(tester))} {command}"
    ceiling = math.ceil(time_limit + 0.1)
    return f"timeout {ceiling + 0.2} prlimit --cpu={ceiling + 0.1} {command}"


def parse_profiles(text: str, wall: float, time_limit: float, memory_limit: int) -> tuple[str, float, int]:
    """(verdict, execution time, memory) as case_runner.parse_profiles decides them;
    verdict "" when the run was fine."""
    tle = signal137 = False
    if text == "":
        return ("TIME_LIMIT_EXCEEDED" if wall > time_limit else "RUNTIME_ERROR"), wall, 0
    if text.startswith("Command terminated by signal 9"):
        text = text.partition("\n")[2]
        tle = True
    elif text.startswith("Command exited with non-zero status"):
        status, _, text = text.partition("\n")
        code = status.rsplit(" ", 1)[-1].rstrip(".")
        signal137 = code.isdigit() and int(code) == 137
    try:
        profile = json.loads(text.strip())
        elapsed = float(profile["elapsed_time_seconds"])
        cpu = float(profile["user_cpu_seconds"]) + float(profile["system_cpu_seconds"])
        exit_status = int(profile["exit_status"])
        memory = int(profile["max_resident_set_size_kbytes"]) * 1024
    except (ValueError, KeyError, TypeError):
        return "WRONG_ANSWER", min(wall, time_limit + 0.1), 0
    used = max(elapsed, cpu)
    wrapper_timeout = signal137 and used > time_limit
    if exit_status != 0 and not wrapper_timeout:
        return "RUNTIME_ERROR", min(used, time_limit + 0.1), memory
    if used > time_limit or tle or wrapper_timeout:
        return "TIME_LIMIT_EXCEEDED", min(used, time_limit + 0.1), memory
    if memory > memory_limit:
        return "MEMORY_LIMIT_EXCEEDED", used, memory
    return "", used, memory


def tester_score(stderr: str) -> int | None:
    lines = stderr.strip().splitlines()
    if not lines or "wrong answer: " in stderr:
        return None
    match = re.match(r"Score = (\d+)", lines[-1])
    return int(match.group(1)) if match else None


def run_case(root: Path, problem: dict, build: Path, input_file: Path, time_limit: float, keep: Path | None) -> dict:
    import time

    tools = root / "tools" / "bin"
    reactive = problem["problem_type"] == "reactive"
    with tempfile.TemporaryDirectory(prefix="ale-case-") as scratch:
        output, profiles = Path(scratch) / "output.txt", Path(scratch) / "profiles.json"
        output.touch()
        profiles.touch()
        command = run_command(problem, time_limit, tools / "tester" if reactive else None, input_file, output,
                              profiles)
        start = time.perf_counter()
        # No `; sync` after the case, unlike the judge's command: sync flushes every file system on the host, the
        # output is read back through the same page cache anyway, and with ~20 runs testing at once these flushes
        # pile up into minute-long stalls of the whole machine (and they would be timed as the case's).
        done = subprocess.run(["/bin/bash", "--noprofile", "--norc", "-c", command], cwd=build,
                              capture_output=True, text=True)
        wall = time.perf_counter() - start
        stderr = done.stderr
        if keep is not None:
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(output, keep / input_file.name)
        if done.returncode != 0 or (reactive and stderr.strip() == ""):
            verdict = "TIME_LIMIT_EXCEEDED" if wall > time_limit else "RUNTIME_ERROR"
            checked, used, memory = parse_profiles(profiles.read_text(), wall, time_limit, problem["memory_limit"])
            return {"case": input_file.name, "verdict": checked or verdict, "score": 0, "time": used,
                    "memory": memory, "stderr": stderr[-2000:]}
        verdict, used, memory = parse_profiles(profiles.read_text(), wall, time_limit, problem["memory_limit"])
        if verdict:
            return {"case": input_file.name, "verdict": verdict, "score": 0, "time": used, "memory": memory,
                    "stderr": stderr[-2000:]}
        if reactive:
            score = tester_score(stderr)
        else:
            judged = subprocess.run([str(tools / "tester"), str(input_file), str(output)], capture_output=True,
                                    text=True)
            score = tester_score(judged.stderr) if judged.returncode == 0 else None
            stderr = judged.stderr if score is None else stderr
        if score is None:
            return {"case": input_file.name, "verdict": "WRONG_ANSWER", "score": 0, "time": used, "memory": memory,
                    "stderr": stderr[-2000:]}
        return {"case": input_file.name, "verdict": "ACCEPTED", "score": score, "time": used, "memory": memory}


def cmd_gen(args) -> int:
    root = workcopy()
    directory = Path(args.dir) if args.dir else root / "in"
    generate(root, [int(seed) for seed in args.seeds], directory)
    print(f"wrote {len(args.seeds)} inputs to {directory}")
    return 0


def cmd_test(args) -> int:
    root = workcopy()
    problem = json.loads((root / "problem.json").read_text())
    source = Path(args.source) if args.source else root / problem["submission_name"]
    if not source.is_file():
        raise SystemExit(f"ale-local: no program at {source}")
    if args.input:
        inputs = [Path(path).resolve() for path in args.input]
    else:
        seeds = seeds_from(args.seeds or problem["public_seeds_text"])
        missing = [seed for seed in seeds if not (root / "in" / f"{seed:04d}.txt").is_file()]
        if missing:
            generate(root, missing, root / "in")
        inputs = [root / "in" / f"{seed:04d}.txt" for seed in seeds]
    time_limit = args.tl or problem["time_limit"]
    with tempfile.TemporaryDirectory(prefix="ale-build-") as build:
        error = compile_program(root, problem, source, Path(build))
        if error is not None:
            print(f"COMPILATION_ERROR\n{error[-4000:]}")
            return 1
        keep = Path(args.keep) if args.keep else None
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            results = list(pool.map(lambda path: run_case(root, problem, Path(build), path, time_limit, keep),
                                    inputs))
    for result in results:
        line = f"{result['case']:>10}  {result['verdict']:<22} score={result['score']:<14} time={result['time']:.3f}s"
        print(line + (f"  stderr: {result['stderr'].strip()[-300:]!r}" if result.get("stderr") and
                      result["verdict"] != "ACCEPTED" else ""))
    counts = collections.Counter(result["verdict"] for result in results)
    overall = next((verdict for verdict in VERDICT_ORDER if counts.get(verdict)), "ACCEPTED")
    total = sum(result["score"] for result in results if result["verdict"] == "ACCEPTED")
    longest = max((result["time"] for result in results), default=0.0)
    print(f"overall {overall}: {dict(counts)}; total absolute score {total} over {len(results)} cases "
          f"(mean {total / max(1, len(results)):.1f}); longest {longest:.3f}s of {time_limit:g}s")
    return 0 if overall == "ACCEPTED" else 1


def cmd_vis(args) -> int:
    root = workcopy()
    done = subprocess.run([str(root / "tools" / "bin" / "vis"), args.input, args.output])
    return done.returncode


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="ale-local", description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    gen = commands.add_parser("gen", help="generate inputs for seeds")
    gen.add_argument("seeds", nargs="+")
    gen.add_argument("--dir", help="output directory (default: in/ of your copy)")
    test = commands.add_parser("test", help="compile and run cases like the judge")
    test.add_argument("source", nargs="?")
    test.add_argument("--seeds", help="seeds, e.g. 0-49 or 0,3,7 (default: the judge's public seeds)")
    test.add_argument("--input", nargs="+", help="input files instead of seeds")
    test.add_argument("-j", "--jobs", type=int, default=1, help="cases at a time (keep within your CPU share)")
    test.add_argument("--tl", type=float, help="time limit in seconds (default: the problem's)")
    test.add_argument("--keep", help="copy each case's output into this directory")
    vis = commands.add_parser("vis", help="run the visualizer")
    vis.add_argument("input")
    vis.add_argument("output")
    args = parser.parse_args(argv)
    return {"gen": cmd_gen, "test": cmd_test, "vis": cmd_vis}[args.command](args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
