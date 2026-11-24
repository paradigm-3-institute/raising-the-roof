#!/usr/bin/env python3
"""The git-to-zig verifier pipeline (verify.py + runner.py from /root/tests), with switchable speedups.

The steps a flag replaces are adapted from git-to-zig's tests/verify.py and tests/runner.py in Proximal Labs'
FrontierSWE v2 (https://github.com/Proximal-Labs/frontier-swe-v2, the commit pinned in frontier_swe.py).

Runs inside a container of the official verifier image, as root, exactly where verify.py runs. It imports the
official modules (verify, runner, compute_reward, reset_zig) and reuses every step unchanged except the ones a
flag replaces:

  --prestaged         the suite is already staged at /tmp/git-build (a bind mount prepared once with
                      runner.stage_suite's exact operations), so the 526 MB copy is skipped
  --zig-cache DIR     a warm zig global cache (DIR/zig), copied to /home/agent/.cache/zig, the path the official
                      build uses (a different path changes the binary's bytes); the master copy is never written
  --jobs N            scripts at a time (official: 4)
  --order FILE        run scripts in this order (one name per line; official: sorted names)
  --no-ext-chainlint  GIT_TEST_EXT_CHAIN_LINT=0: skip test-lib's per-script chainlint.pl run (the scripts are
                      fixed and all pass it: on each, the chainlint.pl run prints nothing and exits 0)
  --hang-detect S     end a script early when it is provably stuck: every process of its session asleep in a
                      wait that nothing in the session can end (hangwatch.py), unchanged for S seconds; it is
                      then scored exactly as the official kill at its time limit (exit 137, its TAP so far)
  --binary PATH       skip reset+build: use this prebuilt binary (a build-cache hit; evidence copied from cache)

Writes the same /logs/verifier files as verify.py (verifier.log, build.log, evidence.json, details.json,
reward.json) plus fastgrade.json (phase timings).
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/root/tests")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.dont_write_bytecode = True
import verify  # noqa: E402  (official pipeline functions; main() is not run)
import runner  # noqa: E402

PHASES: dict = {}
# OOM protection for the verifier itself: every test script (and all it starts, by inheritance) gets oom_score_adj
# +OOM_SHIFT. The same offset for every test process keeps the kernel's choice among them as in the official grader;
# the verifier's python3 (0) is only picked when no test process is left. Raising one's own score needs no privilege.
OOM_SHIFT = int(os.environ.get("FASTGRADE_OOM_SHIFT", "500"))
PROFILE = False


def setup_profile(candidate_bin: str) -> None:
    """/tmp/prof-bin/git: cpuwrap built with the candidate's path, logging every top-level candidate call's CPU."""
    os.makedirs("/tmp/prof-bin", exist_ok=True)
    os.makedirs("/tmp/prof", exist_ok=True)
    os.chmod("/tmp/prof", 0o777)
    shutil.copy2("/fastgrade/prof/cpuwrap", "/tmp/prof-bin/cpuwrap")
    subprocess.run(["zig", "cc", "-O2", "-static", f"-DCANDIDATE=\"{candidate_bin}\"", "-o", "/tmp/prof-bin/git",
                    "/fastgrade/prof/gitwrap.c"], check=True, env={**os.environ, "HOME": "/tmp/prof-home"})
    subprocess.run(["chmod", "-R", "a+rx", "/tmp/prof-bin"], check=False)


def mark(name: str, started: float) -> None:
    PHASES[name] = round(time.monotonic() - started, 3)


def install_cached_build(cached: str) -> dict:
    """verify.build_candidate's result for a build-cache hit: the cached build log and exit code, the cached binary at
    the path the build put it, and the same inspection (file, readelf) of that binary."""
    meta = json.loads(Path(cached + ".build.json").read_text())
    for cache in ("zig-out", ".zig-cache", "zig-cache"):
        shutil.rmtree(verify.AGENT_DIR / cache, ignore_errors=True)
    if Path(cached + ".build.log").exists():
        shutil.copy2(cached + ".build.log", verify.VDIR / "build.log")
    else:
        (verify.VDIR / "build.log").write_bytes(b"")
    build = {"exit_code": meta["exit_code"], "binary_path": "", "binary_type": "", "binary_size": 0, "links_libgit2": False}
    if build["exit_code"] == 0 and meta.get("binary_rel"):
        cand = verify.AGENT_DIR / meta["binary_rel"]
        cand.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, cand)
        subprocess.run(["chown", "-R", "agent:agent", str(verify.AGENT_DIR)], check=False)
        build["binary_path"] = str(cand)
        build["binary_size"] = cand.stat().st_size
        build["binary_type"] = subprocess.run(["file", str(cand)], capture_output=True, text=True).stdout.strip()
        readelf = subprocess.run(["readelf", "-d", str(cand)], capture_output=True, text=True).stdout
        build["links_libgit2"] = any("NEEDED" in ln and "libgit2" in ln.lower() for ln in readelf.splitlines())
    return build


def stage_or_check(args, candidate_bin: str) -> dict:
    """runner.stage_suite, or (prestaged) the same result for the bind-mounted staged copy."""
    stage_dir = "/tmp/git-build"
    if not args.prestaged:
        return runner.stage_suite(str(verify.SUITE), stage_dir, False, candidate_bin)
    staged = os.access(os.path.join(stage_dir, "t", "helper", "test-tool"), os.X_OK)
    return {"staged": staged, "bin_dir": os.path.dirname(candidate_bin), "candidate_bin": candidate_bin}


def run_one(name, suite_t, bin_dir, results_dir, agent_scratch, caps, extra_env, hang):
    """runner._run_one, with optional extra env for the script and the hang watch."""
    odir = os.path.join(agent_scratch, name)
    os.makedirs(os.path.join(odir, "home"), exist_ok=True)
    subprocess.run(["chown", "-R", "agent:agent", odir], check=False)
    argv = runner.test_argv(name, bin_dir, os.path.join(odir, "home"), out_dir=odir, caps=caps)
    if PROFILE:  # profiling only: per-script CPU via cpuwrap around the script, candidate via a wrapping git
        argv = [a if not a.startswith("GIT_TEST_INSTALLED=") else "GIT_TEST_INSTALLED=/tmp/prof-bin" for a in argv]
        i = argv.index(f"./{name}.sh")
        argv[i:i] = ["/tmp/prof-bin/cpuwrap", "/tmp/prof/scripts.log", name]
    if extra_env:
        argv = [argv[0], *extra_env, *argv[1:]]  # env VAR=... (after "env")
    argv = ["runuser", "-u", "agent", "--", *argv]
    cap = runner.cap_for(name, caps)
    started = time.monotonic()
    early = None
    with open(f"{results_dir}/{name}.tap", "wb") as tap, open(f"{results_dir}/{name}.err", "wb") as err:
        if OOM_SHIFT:  # a shell that raises its own oom_score_adj, then exec()s runuser: same pid, same tree
            argv = ["sh", "-c", f'echo {OOM_SHIFT} > /proc/self/oom_score_adj; exec "$@"', "sh", *argv]
        proc = subprocess.Popen(argv, cwd=suite_t, stdout=tap, stderr=err, start_new_session=True)
        if hang:
            early = hang.watch(proc, cap, started, name)
        try:
            rc = proc.wait(timeout=cap + 60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            rc = 124
        if hang:
            hang.unwatch(proc)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    elapsed = time.monotonic() - started
    if early is not None:
        early = {k: early.get(k) for k in ("killed", "at", "closure", "why", "stuck_max", "stuck_at") if early.get(k)}
    with open(f"{results_dir}/{name}.exit", "w") as f:
        f.write(str(rc))
    with open(f"{results_dir}/{name}.secs", "w") as f:
        f.write(f"{elapsed:.3f}")
    shutil.rmtree(odir, ignore_errors=True)
    return {"name": name, "rc": rc, "secs": round(elapsed, 3), "early": early}


def run_suite(args, info, results, scratch, caps):
    suite_t = os.path.join("/tmp/git-build", "t")
    names = runner.read_manifest(str(verify.SCORED))
    if args.order:
        wanted = [n.strip() for n in Path(args.order).read_text().split() if n.strip()]
        rank = {n: i for i, n in enumerate(wanted)}
        names = sorted(names, key=lambda n: (rank.get(n, len(rank)), n))
    extra_env = ["GIT_TEST_EXT_CHAIN_LINT=0"] if args.no_ext_chainlint else []
    hang = None
    if args.hang_detect:
        import hangwatch
        # --hang-detect S > 0 kills after S quiet seconds; S < 0 only records (probe)
        hang = hangwatch.HangWatch(max(args.hang_detect, 0.0), log_path=str(verify.VDIR / "hangwatch.jsonl"))
    deadline = time.monotonic() + runner.MAX_SUITE_SECONDS
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, n, suite_t, info["bin_dir"], str(results), scratch, caps, extra_env, hang)
                   for n in names]
        for fut in as_completed(futures):
            rows.append(fut.result())
            if time.monotonic() > deadline:
                for f in futures:
                    f.cancel()
                print("runner: soft-deadline hit — unfinished scripts score 0")
                break
    captured = len([f for f in os.listdir(results) if f.endswith(".tap")])
    print(f"runner: scripts={len(names)} captured_tap={captured}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prestaged", action="store_true")
    p.add_argument("--zig-cache")
    p.add_argument("--jobs", type=int, default=runner.SUITE_JOBS)
    p.add_argument("--order")
    p.add_argument("--no-ext-chainlint", action="store_true")
    p.add_argument("--hang-detect", type=float, default=0.0)
    p.add_argument("--binary")
    p.add_argument("--keep-binary", help="copy the built binary here (for the build cache)")
    p.add_argument("--build-only", action="store_true", help="stop after the build (timing experiments)")
    p.add_argument("--profile", action="store_true", help="profiling only (changes the run): CPU per script and per candidate call")
    args = p.parse_args()

    t0 = time.monotonic()
    VDIR = verify.VDIR
    VDIR.mkdir(parents=True, exist_ok=True)
    os.chmod(VDIR, 0o700)
    if not verify.assets_present():
        return
    log = open(VDIR / "verifier.log", "w")
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    print(f"=== git-to-zig verifier — {time.ctime()} ===")
    is_oracle = verify.detect_oracle()
    assert not is_oracle
    results, scratch = verify.make_channels()

    anti_cheat = verify.reset_to_pristine(is_oracle)
    subprocess.run(["chown", "-R", "agent:agent", str(verify.APP)], check=False)
    mark("reset", t0)
    verify.remove_stray_git()
    if args.binary and Path(args.binary + ".build.json").exists():
        build = install_cached_build(args.binary)
        print(f"build exit={build['exit_code']} (build cache hit: same zig sources, byte-identical build)")
    else:
        if args.zig_cache:
            # Warm global cache at the official path (/home/agent/.cache/zig: a tmpfs here). Zig's cache is
            # content-addressed, so a hit returns what a cold build would make (binary sha256 checked equal).
            home_cache = "/home/agent/.cache"
            os.makedirs(home_cache, exist_ok=True)
            subprocess.run(["cp", "-a", args.zig_cache.rstrip("/") + "/zig", home_cache + "/"], check=False)
            subprocess.run(["chown", "-R", "agent:agent", home_cache], check=False)
        build = verify.build_candidate()
        if args.keep_binary:
            if build.get("binary_path"):
                shutil.copy2(build["binary_path"], args.keep_binary + ".tmp")
                os.replace(args.keep_binary + ".tmp", args.keep_binary)
            if (VDIR / "build.log").exists():
                shutil.copy2(VDIR / "build.log", args.keep_binary + ".build.log")
            Path(args.keep_binary + ".build.json").write_text(json.dumps(
                {"exit_code": build["exit_code"], "binary_rel": os.path.relpath(build["binary_path"], verify.AGENT_DIR)
                 if build.get("binary_path") else ""}))
    mark("build", t0)
    print(f"candidate binary: {build['binary_path'] or 'none'}")
    if build.get("binary_path") and Path(build["binary_path"]).is_file():
        import hashlib
        PHASES["binary_sha256"] = hashlib.sha256(Path(build["binary_path"]).read_bytes()).hexdigest()

    if args.build_only:
        mark("total", t0)
        (VDIR / "fastgrade.json").write_text(json.dumps({"phases": PHASES, "build": build}, indent=1))
        return
    tests_ran = False
    rows = []
    if build["binary_path"] and Path(build["binary_path"]).is_file():
        caps = runner.load_caps(str(verify.REFERENCE))
        if caps:
            print(f"caps: n={len(caps)} min={min(caps.values())}s max={max(caps.values())}s "
                  f"sum={sum(caps.values())}s")
        if args.profile:
            global PROFILE
            PROFILE = True
            setup_profile(build["binary_path"])
        info = stage_or_check(args, build["binary_path"])
        mark("stage", t0)
        if info["staged"]:
            rows = run_suite(args, info, results, scratch, caps or None)
            tests_ran = True
        else:
            print("runner: suite staging failed (no prebuilt test-tool) — skipping")
        subprocess.run(["pkill", "-u", "agent"], check=False)
        time.sleep(1)
        subprocess.run(["pkill", "-9", "-u", "agent"], check=False)
        mark("suite", t0)
    verify.score(is_oracle, anti_cheat, build, tests_ran, results)
    mark("score", t0)
    print(f"=== done {time.ctime()} ===")
    if PROFILE:
        for f in ("scripts.log", "git.log"):
            if os.path.exists(f"/tmp/prof/{f}"):
                shutil.copy2(f"/tmp/prof/{f}", VDIR / f"prof-{f}")
    (VDIR / "fastgrade.json").write_text(json.dumps({"phases": PHASES, "scripts": rows}, indent=1))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        try:
            if not (verify.VDIR / "reward.json").exists():
                verify.write_invalid()
        except Exception:
            pass
        sys.exit(0)
