"""Fast grader for git-to-zig snapshot gradings: frontier_swe.py uses it for `watch --fastgrade` and `regrade --fast`.

    result = await fastgrade_inrun.grade_artifacts(task_dir, artifacts, trials_dir, trial_name, verifier_env,
                                                   pids_limit, tmpfs)

Same signature and the same outputs as frontier_swe.grade_artifacts for the git-to-zig task: it writes
<trials_dir>/<trial_name>/verifier/ (details.json, reward.json, evidence.json, verifier.log, build.log, test-stdout.txt)
and returns an object with .verifier_result.rewards (reward.json) and .exception_info, which is all TrialWatch reads.

What it does instead of Harbor's regrade path (each step measured, and checked to give the same rewards):
  - a plain `docker run` of the task's verifier image (no compose project, no network setup), same limits: the task's
    CPUs and memory (memory-swap 2x as Harbor sets it), the pids cap, /tmp tmpfs 6 GB, the image's entrypoint and the
    keepalive command; /app emptied (chmod 777) and the artifacts streamed in with modes and mtimes as `docker cp` keeps
    them; the verifier run as root via `bash -c '(...) > /logs/verifier/test-stdout.txt 2>&1'`, the job's verifier env,
    stdin a pipe with no writer (as `docker compose exec` with stdin /dev/null leaves it);
  - /app and /home/agent/.cache on tmpfs (no build writes to the disk);
  - the suite staged once per host (runner.stage_suite's own output, root-owned, bind-mounted at /tmp/git-build) instead
    of a 526 MB copy per grading;
  - a warm zig global cache copied to /home/agent/.cache/zig (the official path; the binary is byte-identical);
  - the container removed in the background (its overlay unmount syncs the whole disk; it is off the critical path).
The verifier logic is the task's own: fastverify.py imports verify.py, runner.py, reset_zig.py and compute_reward.py
from /root/tests and only replaces the staging copy (and, optionally, the suite's job count and order).

prepare_host() makes the staged suite and the warm cache (one container each, ~2 min, ~500 MB in /dev/shm).
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE = ("us-west1-docker.pkg.dev/proximal-core-0/proximal-evals-public/git-to-zig@sha256:"
         "f467a57643d695f5c8c998cb931ed42ac3ee8b0ec4d5b72863a161c33d135ecd")
STATE = Path(os.environ.get("FASTGRADE_STATE", "/dev/shm/fastgrade"))
STAGED = STATE / "staged" / "git-build"
ZIGCACHE = STATE / "zigcache-home"  # ZIGCACHE/zig = a /home/agent/.cache/zig after one build
KEEPALIVE = ["sh", "-c", "while :; do sleep infinity; done"]
EMPTY_APP = ("if [ -L /app ] || { [ -e /app ] && [ ! -d /app ]; }; then rm -rf /app; fi && mkdir -p /app && "
             "find /app -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + && chmod 777 /app")
# Suite scheduling: 4 at a time in sorted order = the official runner. Keep 4: 5 at a time raised memory (OOM risk) and
# triggered the t0008 SIGPIPE race under load.
JOBS = int(os.environ.get("FASTGRADE_JOBS", "4"))


@dataclass
class _ExceptionInfo:
    exception_type: str
    exception_message: str


@dataclass
class _VerifierResult:
    rewards: dict | None


@dataclass
class FastResult:
    verifier_result: _VerifierResult | None
    exception_info: _ExceptionInfo | None = None
    timing: dict = field(default_factory=dict)


def _run(*argv, check=True, **kw):
    return subprocess.run(list(argv), check=check, **kw)


def prepare_host(image: str = IMAGE) -> None:
    """The staged suite and the warm zig cache, once per host (idempotent)."""
    STATE.mkdir(parents=True, exist_ok=True)
    if not (STAGED / "t" / "helper" / "test-tool").exists():
        code = ("import sys; sys.path.insert(0,'/root/tests'); import runner; "
                "runner.stage_suite('/root/tests/git-test-suite', '/out/git-build', False, '/app/zig-git/zig-out/bin/git')")
        (STATE / "staged").mkdir(parents=True, exist_ok=True)
        _run("docker", "run", "--rm", "--cpus", "1", "-v", f"{STATE / 'staged'}:/out", "--entrypoint", "python3",
             image, "-c", code)
    if not (ZIGCACHE / "zig").is_dir():
        ZIGCACHE.mkdir(parents=True, exist_ok=True)
        # build the image's own scaffold once, as the verifier builds (agent, HOME=/home/agent), and keep the cache
        _run("docker", "run", "--rm", "--cpus", "4", "-v", f"{ZIGCACHE}:/out", "--entrypoint", "bash", image, "-c",
             "cd /root/tests && python3 -c 'import reset_zig; reset_zig.reset(\"/root/tests/pristine/zig-scaffold\", "
             "\"/app/zig-git\")' && chown -R agent:agent /app && cd /app/zig-git && "
             "runuser -u agent -- zig build && cp -a /home/agent/.cache/zig /out/")


def _container_name(trial_name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9_.-]", "-", f"fastgrade-{trial_name.lower()}")[:120] + f"-{secrets.token_hex(2)}"


def _grade_sync(task_dir: Path, artifacts: Path, trials_dir: Path, trial_name: str, verifier_env: dict | None,
                pids_limit: int | None, tmpfs: list[str] | None, image: str, cpus: int, memory_mb: int) -> FastResult:
    prepare_host(image)
    vdir = Path(trials_dir) / trial_name / "verifier"
    vdir.mkdir(parents=True, exist_ok=True)
    name = _container_name(trial_name)
    t0 = time.time()
    run = ["docker", "run", "-d", "--name", name, "--cpus", str(cpus), "--memory", f"{memory_mb}m",
           "--memory-swap", f"{2 * memory_mb}m"]
    if pids_limit:
        run += ["--pids-limit", str(pids_limit)]
    for entry in tmpfs or ["/tmp:rw,exec,nosuid,size=6g,mode=1777"]:
        run += ["--tmpfs", entry]
    run += ["--tmpfs", "/app:rw,exec,size=4g,mode=777",
            "--tmpfs", "/home/agent/.cache:rw,exec,uid=1001,gid=1001,mode=755,size=2g",
            "-v", f"{vdir}:/logs/verifier", "-v", f"{HERE}:/fastgrade:ro", "-v", f"{STAGED}:/tmp/git-build",
            "-v", f"{ZIGCACHE}:/fastgrade-zigcache:ro",
            "--entrypoint", "/usr/local/bin/entrypoint.sh", image, *KEEPALIVE]
    try:
        _run(*run, stdout=subprocess.DEVNULL)
        _run("docker", "exec", "-u", "root", name, "bash", "-c", EMPTY_APP)
        app = Path(artifacts) / "app"
        tar = subprocess.Popen(["tar", "-C", str(app), "-cf", "-", "."], stdout=subprocess.PIPE)
        _run("docker", "exec", "-i", "-u", "root", name, "tar", "-C", "/app", "-xf", "-", "--no-same-owner",
             "--no-overwrite-dir", stdin=tar.stdout)
        tar.wait()
        flags = ["--prestaged", "--zig-cache", "/fastgrade-zigcache", "--jobs", str(JOBS)]
        envs = []
        for key, value in (verifier_env or {}).items():
            envs += ["-e", f"{key}={value}"]
        command = f"(python3 /fastgrade/fastverify.py {' '.join(flags)}) > /logs/verifier/test-stdout.txt 2>&1"
        _run("docker", "exec", "-i", *envs, name, "bash", "-c", command, stdin=subprocess.DEVNULL, check=False)
        _run("docker", "exec", "-u", "root", name, "chown", "-R", f"{os.getuid()}:{os.getgid()}", "/logs/verifier",
             check=False)
    except Exception as exc:  # noqa: BLE001
        return FastResult(None, _ExceptionInfo(type(exc).__name__, str(exc)), {"seconds": time.time() - t0})
    finally:
        subprocess.Popen(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    reward_path = vdir / "reward.json"
    rewards = json.loads(reward_path.read_text()) if reward_path.exists() else None
    result = FastResult(_VerifierResult(rewards), None if rewards else _ExceptionInfo("RewardFileNotFoundError",
                                                                                       "no reward.json"),
                        {"seconds": round(time.time() - t0, 1)})
    (vdir.parent / "result.json").write_text(json.dumps({"trial_name": trial_name, "grader": "fastgrade",
                                                         "verifier_result": {"rewards": rewards},
                                                         "seconds": result.timing["seconds"]}, indent=2))
    return result


async def grade_artifacts(task_dir, artifacts, trials_dir, trial_name, verifier_env=None, pids_limit=None,
                          tmpfs=None, image: str = IMAGE, cpus: int = 4, memory_mb: int = 16384) -> FastResult:
    return await asyncio.to_thread(_grade_sync, task_dir and Path(task_dir), Path(artifacts), Path(trials_dir), trial_name,
                                   verifier_env, pids_limit, tmpfs, image, cpus, memory_mb)
