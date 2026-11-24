"""Trusted scorer of Anthropic's original performance take-home (scored-artifact layer).

A submission is one file, perf_takehome.py. Grading follows tests/submission_tests.py
at the pinned commit, in two stages as on the community leaderboard
(paradigm.xyz/puzzles/anthropic-challenge):

1. Untrusted, in the sandbox: build_kernel.py imports the submitted perf_takehome.py
   next to the pristine problem.py, calls KernelBuilder().build_kernel(10, 2047, 256,
   16) and saves kb.instrs as plain data.
2. Trusted, here: the instructions are checked (engines limited to the simulator's)
   and run by the pristine tests/frozen_problem.py on the take-home's inputs, exactly
   as do_kernel_test does (Tree.generate, Input.generate, Machine with pauses and
   debug off, output values compared with reference_kernel2).

The kernel is valid if its output values are right on every input: the job's 9
fixed inputs and 8 fresh random ones, as the test's CorrectnessTests runs 8 random
inputs. The metric is the worst (largest) cycle count over the 9 fixed inputs: cycle
counts can depend on the data when a kernel branches, so the leaderboard scores the
worst of 9 seeded runs. The 9 seeds are drawn when the job's task is written, fixed
for the whole job, and never shown to agents, so a kernel cannot be tuned to them.
The score, higher is better, is the speedup the test prints: 147734 / cycles.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import random
import resource
import shutil
import stat
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENGINES = ("alu", "valu", "load", "store", "flow", "debug")


def to_tuples(value):
    """JSON lists back to the tuples the kernel builder made (the simulator unpacks
    slots, so lists and tuples behave the same; tuples keep it exact)."""
    if isinstance(value, list):
        return tuple(to_tuples(item) for item in value)
    return value


def load_program(path: Path, owner: int | None) -> list[dict]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or (owner is not None and info.st_uid != owner):
            raise ValueError("program.json is not a regular file written by the kernel build")
        data = json.loads(stream.read())
    if not isinstance(data, list):
        raise ValueError("KernelBuilder.instrs must be a list of instruction bundles")
    program = []
    for index, bundle in enumerate(data):
        if not isinstance(bundle, dict):
            raise ValueError(f"instruction {index} is a {type(bundle).__name__}, not a dict of engine -> slots")
        for engine, slots in bundle.items():
            if engine not in ENGINES:
                raise ValueError(f"instruction {index} uses engine {engine!r}; the simulator has "
                                 + ", ".join(ENGINES))
            if not isinstance(slots, list) or not all(isinstance(slot, list) for slot in slots):
                raise ValueError(f"instruction {index}: the {engine!r} slots must be a list of tuples")
        program.append({engine: [to_tuples(slot) for slot in slots] for engine, slots in bundle.items()})
    return program


def _simulate(upstream: str, program, shape, seed: int, connection, memory_mb: int) -> None:
    """Child process: do_kernel_test on one input with the frozen simulator."""
    try:
        limit = memory_mb * 2**20
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        sys.path.insert(0, os.path.join(upstream, "tests"))
        from frozen_problem import (DebugInfo, Input, Machine, N_CORES, Tree, build_mem_image,
                                    reference_kernel2)
        forest_height, rounds, batch_size = shape
        random.seed(seed)
        forest = Tree.generate(forest_height)
        inp = Input.generate(forest, batch_size, rounds)
        mem = build_mem_image(forest, inp)
        machine = Machine(mem, program, DebugInfo(scratch_map={}), n_cores=N_CORES)
        machine.enable_pause = False
        machine.enable_debug = False
        try:
            machine.run()
        except Exception as exc:  # noqa: BLE001 - the kernel broke the simulator
            core = machine.cores[0]
            message = str(exc)[:300]
            if isinstance(exc, AssertionError) and not message:
                # The only assertion with debug off: Machine.step's slot limit check.
                message = "an instruction uses more slots of an engine than SLOT_LIMITS allows"
            connection.send({"error": f"{type(exc).__name__}: {message}", "pc": core.pc - 1,
                             "cycle": machine.cycle})
            return
        for ref_mem in reference_kernel2(mem):
            pass
        inp_values_p = ref_mem[6]
        correct = (machine.mem[inp_values_p: inp_values_p + len(inp.values)]
                   == ref_mem[inp_values_p: inp_values_p + len(inp.values)])
        connection.send({"correct": bool(correct), "cycles": machine.cycle})
    except BaseException as exc:  # noqa: BLE001
        connection.send({"error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    finally:
        connection.close()


def simulate(upstream: Path, program, shape, seeds: list[int], workers: int, timeout: float,
             memory_mb: int) -> list[dict]:
    """Run do_kernel_test's simulation on each seed, `workers` at a time."""
    context = multiprocessing.get_context("fork")
    results: list[dict | None] = [None] * len(seeds)
    pending = list(range(len(seeds)))
    active: dict[int, tuple] = {}
    while pending or active:
        while pending and len(active) < workers:
            index = pending.pop(0)
            receive, send = context.Pipe(duplex=False)
            child = context.Process(target=_simulate, args=(str(upstream), program, shape, seeds[index],
                                                            send, memory_mb))
            child.start()
            send.close()
            active[index] = (child, receive, time.monotonic())
        time.sleep(0.02)
        for index, (child, receive, started) in list(active.items()):
            if receive.poll():
                try:
                    results[index] = receive.recv()
                except EOFError:
                    results[index] = {"error": "the simulator process died (out of memory?)"}
            elif not child.is_alive():
                results[index] = {"error": "the simulator process died (out of memory?)"}
            elif time.monotonic() - started > timeout:
                child.kill()
                results[index] = {"error": f"the simulation ran longer than {timeout:.0f} s", "timeout": True}
            else:
                continue
            child.join(5)
            if child.is_alive():
                child.kill()
                child.join()
            del active[index]
    return results


def score(submission: Path, sandbox, config: dict) -> dict:
    upstream = Path(config["upstream_dir"])
    shape = (int(config["forest_height"]), int(config["rounds"]), int(config["batch_size"]))
    n_nodes = 2 ** (shape[0] + 1) - 1
    baseline = float(config["baseline_cycles"])
    fixed = [int(seed) for seed in config["seeds"]]
    fresh = [int.from_bytes(os.urandom(8), "big") for _ in range(int(config["fresh_inputs"]))]

    # 1. Build, untrusted.
    work = sandbox.directory("build")
    shutil.copyfile(submission / "perf_takehome.py", work / "perf_takehome.py")
    shutil.copyfile(upstream / "problem.py", work / "problem.py")
    if sandbox.enabled:
        for name in ("perf_takehome.py", "problem.py"):
            os.chown(work / name, sandbox.uid, sandbox.uid)
    built = sandbox.run([sys.executable, "-I", str(HERE / "build_kernel.py"), str(work),
                         str(shape[0]), str(n_nodes), str(shape[2]), str(shape[1])],
                        work, timeout=float(config["build_timeout_sec"]))
    info = {"build_seconds": round(built.seconds, 3)}
    if built.timed_out:
        return {"valid": False, "detail": f"building the kernel took longer than {config['build_timeout_sec']} s",
                "info": info}
    if built.returncode != 0:
        output = (built.stderr or built.stdout).strip()
        return {"valid": False, "detail": "building the kernel failed:\n" + output[-2000:], "info": info}
    try:
        program = load_program(work / "program.json", sandbox.uid if sandbox.enabled else None)
    except (OSError, ValueError) as exc:
        return {"valid": False, "detail": f"the kernel is not a valid instruction list: {exc}", "info": info}
    info["instructions"] = len(program)

    # 2. Simulate, trusted.
    seeds = fixed + fresh
    results = simulate(upstream, program, shape, seeds, int(config.get("simulation_workers", 2)),
                       float(config["simulation_timeout_sec"]), int(config.get("simulation_memory_mb", 3072)))
    labels = [f"fixed input {i + 1}" for i in range(len(fixed))] + \
             [f"fresh random input {i + 1}" for i in range(len(fresh))]
    info["fixed_cycles"] = [r.get("cycles") for r in results[:len(fixed)]]
    info["fresh_cycles"] = [r.get("cycles") for r in results[len(fixed):]]
    failures = []
    for label, result in zip(labels, results):
        if "error" in result:
            where = f" at pc {result['pc']}, cycle {result['cycle']}" if "pc" in result else ""
            failures.append(f"{label}: {result['error']}{where}")
        elif not result["correct"]:
            failures.append(f"{label}: incorrect output values")
    if failures:
        info["failures"] = failures
        shown = "; ".join(failures[:3]) + (f"; and {len(failures) - 3} more" if len(failures) > 3 else "")
        return {"valid": False, "info": info,
                "detail": f"wrong on {len(failures)} of {len(seeds)} inputs: {shown}"}
    cycles = max(info["fixed_cycles"])
    return {"valid": True, "metric": cycles, "score": baseline / cycles, "info": info,
            "detail": (f"correct on {len(fixed)} fixed and {len(fresh)} fresh random inputs; "
                       f"{cycles} cycles (the worst of the {len(fixed)} fixed inputs); "
                       f"speedup {baseline / cycles:.4f}x over the {int(baseline)}-cycle starter")}
