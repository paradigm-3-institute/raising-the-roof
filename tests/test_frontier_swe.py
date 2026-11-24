"""Checks for running FrontierSWE v2 tasks (frontier_swe.py, benchmarks/frontier-swe,
run.py frontier-swe, analysis). No Docker needed.

A throwaway git repository stands in for frontier-swe-v2, cloned blobless and sparse
like .cache/frontier-swe-v2, so fetching runs the real git commands. The in-container
tools (fswe.py) run on temporary directories that play /app, /work, and /logs.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from cli_render import ROOT, render
from test_cli_agents import run_py, scripts

sys.path.insert(0, str(ROOT))
import frontier_swe  # noqa: E402

BENCH = ROOT / "benchmarks" / "frontier-swe"
spec = importlib.util.spec_from_file_location("fswe", BENCH / "fswe.py")
fswe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fswe)

CPU_TOML = """\
schema_version = "1.3"

[[artifacts]]
source = "/app/"
exclude = [
    "**/target/**",
    "tests/**",
]

[task]
name = "proximal-evals/demo-cpu"

[agent]
timeout_sec = 72000.0
user = "agent"

[verifier]
# a comment that must survive
timeout_sec = 5400.0
environment_mode = "separate"

[verifier.env]

[verifier.environment]
docker_image = "example.com/demo@sha256:verifier"
cpus = 2
memory_mb = 4096

[environment]
docker_image = "example.com/demo@sha256:agent"
network_mode = "allowlist"
build_timeout_sec = 1500   # a trailing comment
cpus = 2
memory_mb = 4096
gpus = 0

[environment.env]
"""
GPU_TOML = CPU_TOML.replace("gpus = 0\n", 'gpus = 1\ngpu_types = ["L4"]\n').replace(
    'docker_image = "example.com/demo@sha256:verifier"\n',
    'docker_image = "example.com/demo@sha256:verifier"\ngpus = 1\ngpu_types = ["L4"]\n')
SHARED_TOML = CPU_TOML.replace('environment_mode = "separate"', 'environment_mode = "shared"').split(
    "[verifier.environment]")[0] + "[environment]\ncpus = 2\n"
OUTSIDE_TOML = CPU_TOML.replace('source = "/app/"', 'source = "/srv/out/"')
PROFILE = "environment:\n  type: modal\n  cpu_enforcement_policy: guarantee\n  kwargs:\n    modal_vm_runtime: true\n"
INSTRUCTION = "Implement the demo.\n\nThe machine is offline.\n"


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def write_task(root: Path, name: str, toml: str, profile: str | None = None) -> None:
    task = root / "tasks" / name
    (task / "solution").mkdir(parents=True)
    (task / "environment").mkdir()
    (task / "task.toml").write_text(toml)
    (task / "instruction.md").write_text(INSTRUCTION)
    (task / "solution" / "solve.sh").write_text("#!/bin/bash\necho oracle\n")
    (task / "environment" / "Dockerfile").write_text("FROM scratch\n")
    if profile:
        (task / "job.yaml").write_text(profile)


@pytest.fixture(scope="module")
def upstream(tmp_path_factory):
    """A fake frontier-swe-v2 origin and the commit to pin."""
    origin = tmp_path_factory.mktemp("fswe") / "origin"
    origin.mkdir()
    git("init", "-q", "-b", "main", cwd=origin)
    git("config", "uploadpack.allowFilter", "true", cwd=origin)
    git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=origin)
    (origin / "README.md").write_text("fake FrontierSWE v2\n")
    write_task(origin, "demo-cpu", CPU_TOML)
    write_task(origin, "demo-gpu", GPU_TOML)
    write_task(origin, "demo-shared", SHARED_TOML)
    write_task(origin, "demo-outside", OUTSIDE_TOML)
    write_task(origin, "demo-profile", CPU_TOML, PROFILE)
    git("add", ".", cwd=origin)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "tasks", cwd=origin)
    return f"file://{origin}", git("rev-parse", "HEAD", cwd=origin).strip()


@pytest.fixture
def pinned(upstream, monkeypatch, tmp_path):
    url, commit = upstream
    monkeypatch.setattr(frontier_swe, "REPO_URL", url)
    monkeypatch.setattr(frontier_swe, "COMMIT", commit)
    return tmp_path / "clone"


# ---------------------------------------------------------------------------
# The pinned checkout


def test_fetch_checks_out_only_the_task_at_the_pinned_commit(pinned, upstream):
    path = frontier_swe.fetch_task("demo-cpu", pinned)
    assert path == pinned.resolve() / "tasks/demo-cpu"
    assert (path / "task.toml").read_text() == CPU_TOML
    assert not (pinned / "tasks/demo-gpu").exists()
    assert git("rev-parse", "HEAD", cwd=pinned).strip() == upstream[1]
    assert frontier_swe.available_tasks(pinned) == [
        "demo-cpu", "demo-gpu", "demo-outside", "demo-profile", "demo-shared"]
    # A second fetch reuses the clone and adds only the new task.
    frontier_swe.fetch_task("demo-gpu", pinned)
    assert (pinned / "tasks/demo-gpu/task.toml").is_file()


@pytest.mark.parametrize("name, message", [
    ("missing-task", "has no task 'missing-task'.*demo-cpu"),
    ("Bad Name", "must be a FrontierSWE v2 task name"),
])
def test_fetch_rejects_unknown_tasks(pinned, name, message):
    with pytest.raises(ValueError, match=message):
        frontier_swe.fetch_task(name, pinned)


# ---------------------------------------------------------------------------
# Graded paths: GNU tar's exclusion rules, as Harbor's artifact download applies them

TREE = [
    "Package.swift", "Sources/vsim/main.swift", ".build/release/vsim", ".build/x.o",
    "target/a", "sub/target/b", "sub/targets/c", "tests/t/t0001.sh", "tests/keep", "a/tests/x",
    "zig-git/zig-out/bin/git", "zig-git/.zig-cache/h/o", "zig-git/src/main.zig",
    ".git/HEAD", "deep/.git/config", "pkg/__pycache__/m.pyc", "node_modules/x/y.js",
    "data/big.bin", "data/small.txt", "notes.log", "x/notes.log", "checkpoint/adapter/model.bin",
]
PATTERNS = [
    ["**/target/**", "**/.git/**", "**/__pycache__/**", "**/node_modules/**", "**/.venv/**"],
    ["tests/**", "**/zig-out/**", "**/.zig-cache/**", "**/zig-cache/**", "**/.git/**"],
    [".build", "data/*.bin", "*.log", "checkpoint/**"],
    ["./tests", "sub/*"],
]


def make_tree(root: Path) -> None:
    for relative in TREE:
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(relative)
    (root / "empty-dir").mkdir()
    (root / "link").symlink_to("Package.swift")


@pytest.mark.parametrize("patterns", PATTERNS)
def test_graded_paths_match_gnu_tar(tmp_path, patterns):
    make_tree(tmp_path / "app")
    flags = [f"--exclude={pattern}" for pattern in patterns]
    archive = subprocess.run(["tar", "cf", "-", *flags, "-C", str(tmp_path / "app"), "."],
                             capture_output=True, check=True).stdout
    listed = subprocess.run(["tar", "tf", "-"], input=archive, capture_output=True, check=True).stdout
    names = {line.decode().removeprefix("./").rstrip("/") for line in listed.splitlines()} - {"", "."}
    graded = fswe.Graded([{"source": "/app/", "exclude": patterns}], "/app")
    dirs, files = graded.tree(tmp_path / "app")
    assert dirs | set(files) == names


def test_file_entries_and_subdirectories(tmp_path):
    make_tree(tmp_path / "app")
    graded = fswe.Graded([{"source": "/app/zig-git/", "exclude": ["**/zig-out/**"]},
                          {"source": "/app/checkpoint/adapter/model.bin", "exclude": []}], "/app")
    dirs, files = graded.tree(tmp_path / "app")
    assert set(files) == {"zig-git/.zig-cache/h/o", "zig-git/src/main.zig", "checkpoint/adapter/model.bin"}
    assert "zig-git" in dirs and "zig-git/zig-out" in dirs and "zig-git/zig-out/bin" not in dirs
    with pytest.raises(ValueError, match="not under /app"):
        fswe.Graded([{"source": "/srv/x/", "exclude": []}], "/app")


# ---------------------------------------------------------------------------
# workcopy and submit, on directories standing in for /app, /work, and /logs


@pytest.fixture
def container(tmp_path):
    """fswe.py with a config whose /app, /work, and log dir live under tmp_path."""
    tools = tmp_path / "opt"
    tools.mkdir()
    shutil.copy2(BENCH / "fswe.py", tools / "fswe.py")
    app = tmp_path / "app"
    make_tree(app)
    config = {"task": "demo", "app_root": str(app), "agent_user": "agent", "agents": 2,
              "graded": [{"source": f"{app}/", "exclude": ["**/target/**", "tests/**", "**/.git/**"]}],
              "promote_exclude": [".build"], "work_root": str(tmp_path / "work"),
              "log_dir": str(tmp_path / "logs"), "lock": str(tmp_path / "work/.fswe/promote.lock"),
              "expose_scores": False, "codex_auth": None}
    (tools / "config.json").write_text(json.dumps(config))
    (tmp_path / "work/.fswe").mkdir(parents=True)

    def tool(*argv: str, agent: str = "a1", check: bool = True) -> subprocess.CompletedProcess:
        env = {**os.environ, "AGENT_ID": agent, "CODEX_HOME": str(tmp_path / "codex-home")}
        result = subprocess.run([sys.executable, str(tools / "fswe.py"), *argv], capture_output=True,
                                text=True, env=env)
        if check and result.returncode != 0:
            raise AssertionError(result.stdout + result.stderr)
        return result

    return tool, app, tmp_path


def log(tmp_path: Path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "logs/submissions.jsonl").read_text().splitlines()]


def test_workcopy_copies_only_graded_files(container):
    tool, app, tmp_path = container
    work = Path(tool("workcopy").stdout.strip())
    assert work == tmp_path / "work/a1"
    assert (work / "Sources/vsim/main.swift").read_text() == "Sources/vsim/main.swift"
    assert not (work / ".build").exists() and not (work / "target/a").exists()
    assert (work / "tests").is_dir() and not any((work / "tests").iterdir())
    assert (work / "link").is_symlink() and os.readlink(work / "link") == "Package.swift"
    (work / "Package.swift").write_text("edited")
    assert Path(tool("workcopy").stdout.strip()) == work  # once only
    assert (work / "Package.swift").read_text() == "edited"


def test_submit_promotes_by_local_score(container):
    tool, app, tmp_path = container
    a1, a2 = (Path(tool("workcopy", agent=agent).stdout.strip()) for agent in ("a1", "a2"))
    (a1 / "Sources/vsim/main.swift").write_text("a1 version")
    (a1 / ".build").mkdir(exist_ok=True)
    (a1 / ".build/cache").write_text("build output")
    out = tool("submit", "--score", "0.5", "--note", "first")
    assert "#1 promoted" in out.stdout
    assert (app / "Sources/vsim/main.swift").read_text() == "a1 version"
    assert not (app / ".build/cache").exists()  # promote_exclude: build outputs stay out of /app
    assert (app / ".build/x.o").exists() and (app / "tests/t/t0001.sh").exists()  # ungraded: untouched

    (a2 / "Sources/vsim/main.swift").write_text("a2 version")
    (a2 / "Sources/vsim/extra.swift").write_text("new file")
    lower = tool("submit", "--score", "0.4", agent="a2", check=False)
    assert lower.returncode == 1 and "rejected" in lower.stdout
    assert (app / "Sources/vsim/main.swift").read_text() == "a1 version"
    unscored = tool("submit", agent="a2", check=False)
    assert unscored.returncode == 1 and "give --score" in unscored.stdout
    tool("submit", "--score", "0.4", "--force", agent="a2")
    assert (app / "Sources/vsim/extra.swift").read_text() == "new file"

    # A promotion makes /app's graded files equal to the copy's: removals carry over.
    tool("submit", "--score", "0.6")
    assert not (app / "Sources/vsim/extra.swift").exists()
    assert (app / "Sources/vsim/main.swift").read_text() == "a1 version"
    records = log(tmp_path)
    assert [(r["seq"], r["agent_id"], r["status"]) for r in records] == [
        (1, "a1", "promoted"), (2, "a2", "rejected"), (3, "a2", "rejected"), (4, "a2", "promoted"),
        (5, "a1", "promoted")]
    assert records[3]["forced"] and records[3]["previous_score"] == 0.5
    assert records[4]["files_deleted"] == 1


def test_submit_needs_a_working_copy(container):
    tool, _, _ = container
    result = tool("submit", "--score", "1", check=False)
    assert result.returncode == 2 and "run `workcopy` first" in result.stderr


def test_finish_completes_an_interrupted_promotion(container):
    tool, app, tmp_path = container
    work = Path(tool("workcopy").stdout.strip())
    (work / "Package.swift").write_text("half promoted")
    (tmp_path / "work/.fswe/pending-promotion.json").write_text(json.dumps({"seq": 9, "source": str(work)}))
    out = tool("finish")
    assert "completed interrupted promotion #9" in out.stdout
    assert (app / "Package.swift").read_text() == "half promoted"
    assert (tmp_path / "logs/finished").is_file()
    assert not (tmp_path / "work/.fswe/pending-promotion.json").exists()


def test_locked_runs_a_command_under_the_lock(container):
    tool, _, tmp_path = container
    out = tool("locked", "--", "sh", "-c", "echo held")
    assert out.stdout.strip() == "held"


# ---------------------------------------------------------------------------
# The harness text, task.toml, and the task directory


def test_solo_harness_only_drops_peer_lines():
    graded = [{"source": "/app/", "exclude": []}]
    team = frontier_swe.harness_text(3, graded, False).split("\n")
    solo = frontier_swe.harness_text(1, graded, False).split("\n")
    rewritten = {new for _, new in frontier_swe.SOLO_EDITS if new is not None}
    kept = [line for line in solo if line not in rewritten]
    team_lines = iter(team)
    assert all(line in team_lines for line in kept)
    assert len(solo) == len(kept) + len(rewritten)
    assert "3 agents share this container" in "\n".join(team)
    assert not any(word in "\n".join(solo) for word in ("agents", "another agent", "share"))
    exposed = frontier_swe.harness_text(1, graded, True)
    assert frontier_swe.VERIFIER_EXPOSED in exposed and frontier_swe.VERIFIER_HIDDEN not in exposed


def test_set_toml_keeps_comments_and_changes_only_the_keys():
    edited = frontier_swe.set_toml(CPU_TOML, {("environment", "cpus"): 6,
                                              ("environment.env", "TASK_BUDGET_SECS"): "3600",
                                              ("verifier.environment", "gpus"): 0})
    assert "# a comment that must survive" in edited and "build_timeout_sec = 1500   # a trailing comment" in edited
    config, original = tomllib.loads(edited), tomllib.loads(CPU_TOML)
    assert config["environment"]["cpus"] == 6 and config["environment"]["env"] == {"TASK_BUDGET_SECS": "3600"}
    assert config["verifier"]["environment"] == {**original["verifier"]["environment"], "gpus": 0}
    added = frontier_swe.set_toml(CPU_TOML.replace("[environment.env]\n", ""),
                                  {("environment.env", "X"): "1"})
    assert tomllib.loads(added)["environment"]["env"] == {"X": "1"}
    with pytest.raises(ValueError, match="did not produce"):
        frontier_swe.set_toml(CPU_TOML.replace("[environment.env]\n", "").replace(
            "gpus = 0\n", "gpus = 0\nenv = { A = \"1\" }\n"), {("environment.env", "X"): "1"})


def test_task_for_a_team(pinned, tmp_path):
    built = frontier_swe.build_task("demo-cpu", tmp_path / "tasks", clone=pinned, agents=3, cli="codex",
                                    timeout=7200)
    task = built.write()
    assert task.name == "frontier-swe-demo-cpu"
    config, original = tomllib.loads((task / "task.toml").read_text()), tomllib.loads(CPU_TOML)
    # k agents get k times the task's CPUs and memory; nothing else about the task changes.
    assert (config["environment"]["cpus"], config["environment"]["memory_mb"]) == (6, 12288)
    assert config["environment"]["env"] == {"TASK_BUDGET_SECS": "7200"}
    for key in ("artifacts", "agent", "verifier", "task"):
        assert config[key] == original[key], key
    assert {k: v for k, v in config["environment"].items() if k not in ("cpus", "memory_mb", "env")} == \
        {k: v for k, v in original["environment"].items() if k not in ("cpus", "memory_mb", "env")}
    from harbor.models.task.config import TaskConfig
    harbor = TaskConfig.model_validate_toml((task / "task.toml").read_text())
    assert harbor.environment.network_mode.value == "allowlist" and harbor.agent.user == "agent"
    assert harbor.verifier.environment.cpus == 2
    assert (task / "instruction.md").read_text() == INSTRUCTION.rstrip("\n") + "\n\n" + \
        frontier_swe.harness_text(3, built.upstream.graded, False)
    assert (task / "solution/solve.sh").read_text().startswith("#!/bin/bash")
    assert not (task / "environment/Dockerfile").exists()  # the pinned image only
    tools = json.loads((task / "environment/fswe/config.json").read_text())
    assert tools["graded"] == [{"source": "/app/", "exclude": ["**/target/**", "tests/**"]}]
    assert tools["agent_user"] == "agent" and tools["codex_auth"]["host"] == frontier_swe.CODEX_AUTH_HOST
    for tool in ("fswe.py", "bin/submit", "bin/workcopy"):
        assert os.access(task / "environment/fswe" / tool, os.X_OK), tool
    compose = (task / "environment/docker-compose.yaml").read_text()
    assert "./fswe:/opt/fswe:ro" in compose and "nvidia" not in compose
    described = json.loads((task / "frontier-swe.json").read_text())
    assert described["commit"] == frontier_swe.COMMIT and described["resources"]["per_agent_cpus"] == 2
    with pytest.raises(ValueError, match="already exists"):
        built.write()


def test_resources_and_profiles(pinned, tmp_path):
    def build(name="demo-cpu", **kwargs):
        return frontier_swe.build_task(name, tmp_path, clone=pinned, **kwargs)
    solo = tomllib.loads(build().task_toml)
    assert (solo["environment"]["cpus"], solo["environment"]["memory_mb"]) == (2, 4096)
    assert solo["environment"]["env"]["TASK_BUDGET_SECS"] == "72000"
    custom = tomllib.loads(build(agents=4, cpus=5, memory_mb=9000).task_toml)
    assert (custom["environment"]["cpus"], custom["environment"]["memory_mb"]) == (5, 9000)
    profiled = build("demo-profile")
    assert profiled.harbor_args == ["--cpus", "guarantee"]
    assert profiled.description["ignored_runtime_profile"] == {"type": "modal", "kwargs": {"modal_vm_runtime": True}}
    with pytest.raises(ValueError, match="no separate verifier"):
        build("demo-shared")
    with pytest.raises(ValueError, match="outside /app"):
        build("demo-outside")


def test_gpu_tasks_are_refused(pinned, tmp_path):
    with pytest.raises(ValueError, match="demo-gpu needs GPUs; this adapter runs CPU-only tasks"):
        frontier_swe.build_task("demo-gpu", tmp_path, clone=pinned)


# ---------------------------------------------------------------------------
# run.py frontier-swe and the agent runner


def test_run_py_frontier_swe(pinned, tmp_path, monkeypatch):
    import run
    monkeypatch.setattr(run, "FRONTIER_SWE_TASKS", ("git-to-zig", "demo-cpu"))  # the fake upstream's task
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    name = "pytest-frontier-swe-dry-run"
    cmd, env, job = run_py("frontier-swe", "--task", "demo-cpu", "--frontier-swe", str(pinned),
                           "--cli", "codex", "--codex-auth", str(auth), "--model", "m",
                           "--protocol", "antiherding_matched", "--agents", "2", "--token-budget", "300000",
                           "--timeout", "7200", "--name", name)
    task = frontier_swe.TASKS / name / "frontier-swe-demo-cpu"
    assert cmd[cmd.index("-p") + 1] == str(task)
    assert not task.exists()  # --dry-run writes nothing
    assert "--override-cpus" not in cmd and "--override-memory-mb" not in cmd
    # The task's allowlist is empty; while agents run they reach their model API.
    hosts = [cmd[i + 1] for i, token in enumerate(cmd) if token == "--allow-agent-host"]
    assert hosts == ["chatgpt.com", "auth.openai.com"]
    # Harbor's timeout: --timeout, 48 h of usage-limit waiting and 30 min (run.py --limit-resume).
    assert cmd[cmd.index("--agent-timeout-multiplier") + 1] == str((7200 + 48 * 3600 + 1800) / 72000)
    verifier_env = [cmd[i + 1] for i, token in enumerate(cmd) if token == "--ve"]
    assert verifier_env == ["FRONTIER_SWE_TASK=demo-cpu", f"FRONTIER_SWE_COMMIT={frontier_swe.COMMIT}"]
    mounts = json.loads(cmd[cmd.index("--mounts-json") + 1])
    assert {"type": "bind", "source": str(auth.resolve()), "target": frontier_swe.CODEX_AUTH_HOST,
            "bind": {"create_host_path": False}} in mounts
    assert env["RELEASE_TASK"] == "frontier-swe" and env["RELEASE_TOKEN_BUDGET"] == "300000"
    assert env["RELEASE_CONTINUATION"] == (BENCH / "continuation.txt").read_text().rstrip("\n")
    assert job == ROOT / "jobs" / name


@pytest.mark.parametrize("argv, message", [
    (["frontier-swe", "--cli", "codex", "--model", "m"], "frontier-swe needs --task"),
    (["ale-bench", "--problem", "ahc014", "--cli", "codex", "--model", "m", "--expose-scores"],
     "require the frontier-swe task"),
    (["ale-bench", "--problem", "ahc014", "--cli", "codex", "--model", "m", "--allow-host", "example.org"],
     "require the frontier-swe task"),
])
def test_run_py_frontier_swe_errors(pinned, argv, message):
    with pytest.raises(ValueError, match=message):
        run_py(*argv, "--frontier-swe", str(pinned))


@pytest.mark.parametrize("cli", ["codex", "claude"])
@pytest.mark.parametrize("case", ["ale-single", "ale-team3"])
def test_runner_installs_tools_and_finishes(cli, case):
    calls = render(case, cli=cli, model="m", env_overrides={"RELEASE_TASK": "frontier-swe"})
    commands = [call["command"] for call in calls]
    assert not any("apt-get" in command for command in commands)
    assert calls[0] == {"user": "root", "env": {}, "command": "set -o pipefail; python3 /opt/fswe/fswe.py setup"}
    assert commands[-1] == "python3 /opt/fswe/fswe.py finish"
    others = render(case, cli=cli, model="m")
    assert any("apt-get" in call["command"] for call in others)
    assert not any("fswe" in call["command"] for call in others)
    assert scripts(calls[:-1]) == scripts(others)  # the runner itself is unchanged


# ---------------------------------------------------------------------------
# Snapshot bookkeeping and analysis


def test_container_name_and_pack_command():
    assert frontier_swe.container_name("frontier-swe-git-to-zig__AbC1234") == \
        "frontier-swe-git-to-zig__abc1234__env-main-1"
    graded = [{"source": "/app/", "exclude": ["tests/**", "**/.git/**"]},
              {"source": "/app/model.bin", "exclude": []}]
    script = frontier_swe.pack_script(graded)
    # The same tar invocation as Harbor's artifact download, under the promotion lock.
    assert "tar czf \"$1/0.tgz\" --exclude='tests/**' --exclude='**/.git/**' -C /app/ ." in script
    assert "cp -p /app/model.bin \"$1/1.file\"" in script
    command = frontier_swe.pack_command(graded)
    assert f"python3 /opt/fswe/fswe.py locked -- sh -c {shlex.quote(script)} sh \"$d\"" in command


def test_snapshot_triggers(tmp_path, monkeypatch):
    trial = tmp_path / "job" / "frontier-swe-demo__abc1234"
    shared = trial / "agent" / "frontier-swe"
    shared.mkdir(parents=True)
    watch = frontier_swe.TrialWatch(trial, tmp_path, [], {}, 10.0, False, lambda message: None)
    assert watch.container == "frontier-swe-demo__abc1234__env-main-1"
    assert not watch.ready() and not watch.finished()
    (shared / "ready").write_text("{}")
    assert watch.ready() and watch.trigger() == {"trigger": "start"}
    watch.seq, watch.last_digest, watch.last_time = 1, "d0", frontier_swe.time.time()
    assert watch.trigger() is None
    with (shared / "submissions.jsonl").open("a") as stream:
        for seq, status in ((1, "promoted"), (2, "rejected"), (3, "promoted")):
            stream.write(json.dumps({"seq": seq, "status": status, "agent_id": "a1"}) + "\n")
    # Promotions made since the last look coalesce into one snapshot of the newest.
    assert watch.trigger() == {"trigger": "submit", "submission": {"seq": 3, "status": "promoted", "agent_id": "a1"}}
    watch.pending = None
    assert watch.trigger() is None
    monkeypatch.setattr(frontier_swe.time, "time", lambda: watch.last_time + 601)
    assert watch.trigger() == {"trigger": "interval"}
    watch.interval = None
    assert watch.trigger() is None
    (shared / "finished").write_text("")
    assert watch.finished()


def test_snapshot_store_roundtrip(tmp_path):
    root = tmp_path / "artifacts"
    make_tree(root / "app")
    os.chmod(root / "app/Package.swift", 0o755)
    manifest = frontier_swe.tree_manifest(root)
    frontier_swe.store_blobs(root, manifest, tmp_path / "blobs")
    frontier_swe.materialize(manifest, tmp_path / "blobs", tmp_path / "again")
    assert frontier_swe.tree_manifest(tmp_path / "again") == manifest
    assert frontier_swe.manifest_digest(manifest) == frontier_swe.manifest_digest(
        frontier_swe.tree_manifest(tmp_path / "again"))
    (tmp_path / "again/app/Package.swift").write_text("changed")
    assert frontier_swe.manifest_digest(frontier_swe.tree_manifest(tmp_path / "again")) != \
        frontier_swe.manifest_digest(manifest)


def make_job(root: Path, task_dir: Path, snapshots: list[dict], final: float | None) -> Path:
    job = root / "job"
    trial = job / "frontier-swe-demo-cpu__abc1234"
    home = trial / "agent/sessions-0-a1/codex-home/sessions"
    home.mkdir(parents=True)
    events = [{"type": "session_meta", "payload": {"id": "s1"}}]
    for clock, output in (("10:01", 100), ("10:31", 1000), ("11:01", 2000)):
        events.append({"type": "event_msg", "timestamp": f"2026-10-07T{clock}:00Z", "payload": {
            "type": "token_count", "info": {"total_token_usage": {
                "input_tokens": output * 10, "cached_input_tokens": output * 9, "output_tokens": output,
                "reasoning_output_tokens": 0}}}})
    (home / "rollout-a.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (trial / frontier_swe.STATE).mkdir()
    (trial / frontier_swe.STATE / "snapshots.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in snapshots))
    config = {"tasks": [{"path": str(task_dir)}], "verifier": {"env": {"FRONTIER_SWE_TASK": "demo-cpu"}}}
    (job / "config.json").write_text(json.dumps(config))
    result = {"agent_execution": {"started_at": "2026-10-07T10:00:00Z"}}
    if final is not None:
        result["verifier_result"] = {"rewards": {"reward": final}}
    (trial / "result.json").write_text(json.dumps(result))
    return job


def test_keep_every_promotion(tmp_path, monkeypatch):
    """Each promotion's code is kept as it lands (the grader skips promotions made while it is busy, and /app
    overwrites their code): two within one poll leave the first lost, a promotion during the copy
    makes it copy again, and a restarted watcher goes on after the last kept one."""
    trial = tmp_path / "job" / "frontier-swe-demo__abc1234"
    shared = trial / "agent" / "frontier-swe"
    shared.mkdir(parents=True)
    (shared / "ready").write_text("{}")
    logs = []
    watch = frontier_swe.TrialWatch(trial, tmp_path, [], {}, 10.0, False, logs.append)
    monkeypatch.setattr(watch, "tokens", lambda: {"weighted_tokens": 7})
    app = {"code": "v0"}
    during = []   # promotions to land during the next copy

    def promote(seq, code, status="promoted"):
        app["code"] = code if status == "promoted" else app["code"]
        with (shared / "submissions.jsonl").open("a") as stream:
            stream.write(json.dumps({"seq": seq, "status": status, "agent_id": f"a{seq}", "local_score": seq}) + "\n")

    async def capture(destination):
        (destination / "app").mkdir(parents=True)
        (destination / "app" / "main.zig").write_text(app["code"])
        while during:
            promote(*during.pop(0))

    monkeypatch.setattr(watch, "capture", capture)

    def kept():
        return [(r["submission_seq"], r["status"]) for r in frontier_swe.read_jsonl(watch.kept_path)]

    def code(seq):
        record = [r for r in frontier_swe.read_jsonl(watch.kept_path) if r["submission_seq"] == seq][-1]
        manifest = json.loads((watch.state / "manifests" / f"{record['digest']}.json").read_text())
        frontier_swe.materialize(manifest, watch.state / "blobs", tmp_path / f"out-{seq}")
        return (tmp_path / f"out-{seq}" / "app" / "main.zig").read_text()

    assert not watch.keep_due()
    promote(1, "v1")
    assert watch.keep_due()
    asyncio.run(watch.keep_latest())
    assert kept() == [(1, "kept")] and code(1) == "v1" and not watch.keep_due()
    promote(2, "v2", "rejected")
    assert not watch.keep_due()        # a rejected submission changes nothing in /app
    promote(3, "v3")
    promote(4, "v4")                   # two promotions within one poll: /app holds only the second
    asyncio.run(watch.keep_latest())
    assert kept()[1:] == [(3, "lost"), (4, "kept")] and code(4) == "v4"
    assert [r for r in frontier_swe.read_jsonl(watch.kept_path) if r["status"] == "lost"][0]["overwritten_by"] == 4
    promote(5, "v5")
    during.append((6, "v6"))           # lands while /app is being copied: that copy may hold either, so copy again
    asyncio.run(watch.keep_latest())
    assert kept()[3:] == [(5, "lost"), (6, "kept")] and code(6) == "v6"
    record = frontier_swe.read_jsonl(watch.kept_path)[-1]
    assert record["weighted_tokens"] == 7 and record["submission_agent"] == "a6" and record["files"] == 1
    assert not list(watch.state.glob("staging-keep*")) and not list(watch.state.rglob("*.tmp"))
    again = frontier_swe.TrialWatch(trial, tmp_path, [], {}, 10.0, False, logs.append)
    assert again.kept_upto == 6 and not again.keep_due()


def test_regrade_kept_grades_each_kept_submission_once(pinned, tmp_path, monkeypatch):
    task = frontier_swe.build_task("demo-cpu", tmp_path / "tasks", clone=pinned).write()
    trees = {}
    for name in ("graded", "skipped"):
        root = tmp_path / name / "artifacts"
        (root / "app").mkdir(parents=True)
        (root / "app" / "main.zig").write_text(name)
        trees[name] = frontier_swe.tree_manifest(root)
    digests = {name: frontier_swe.manifest_digest(m) for name, m in trees.items()}
    snapshots = [{"seq": 1, "status": "graded", "digest": digests["graded"], "reward": 0.25, "grader": "fastgrade",
                  "ts": "2026-10-07T10:30:00Z"}]
    job = make_job(tmp_path, task, snapshots, final=None)
    trial = frontier_swe.trials(job)[0]
    state = trial / frontier_swe.STATE
    for name, manifest in trees.items():
        (state / "manifests").mkdir(exist_ok=True)
        (state / "manifests" / f"{digests[name]}.json").write_text(json.dumps(manifest))
        frontier_swe.store_blobs(tmp_path / name / "artifacts", manifest, state / "blobs")
    (state / "kept.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (
        {"submission_seq": 1, "status": "kept", "digest": digests["graded"]},
        {"submission_seq": 2, "status": "lost", "overwritten_by": 3},
        {"submission_seq": 3, "status": "kept", "digest": digests["skipped"]})))
    graded = []

    async def grade(task_dir, artifacts, grades, name, env, pids_limit, *args, **kwargs):
        graded.append((artifacts / "app" / "main.zig").read_text())
        return SimpleNamespace(verifier_result=SimpleNamespace(rewards={"reward": 0.5}))

    monkeypatch.setattr(frontier_swe, "grade_artifacts", grade)
    monkeypatch.setattr(frontier_swe, "grade_with_probe_retry", grade)
    results = asyncio.run(frontier_swe.regrade([job], kept=True, fast=True))
    assert graded == ["skipped"] and [r["mean"] for r in results] == [0.5]
    rows = frontier_swe.read_jsonl(state / "kept-grades.jsonl")
    assert [(r["submission_seq"], r["reward"], r["grader"], r["source"].split()[0]) for r in rows] == [
        (1, 0.25, "fastgrade", "snapshot"), (3, 0.5, "fastgrade", "regrade")]
    assert asyncio.run(frontier_swe.regrade([job], kept=True, fast=True)) == [] and graded == ["skipped"]
    # A grading that fails leaves no row, so the next regrade tries it again.
    (state / "kept-grades.jsonl").write_text((state / "kept-grades.jsonl").read_text().splitlines(True)[0])

    async def failing(*args, **kwargs):
        return SimpleNamespace(verifier_result=None)

    monkeypatch.setattr(frontier_swe, "grade_with_probe_retry", failing)
    assert [r["mean"] for r in asyncio.run(frontier_swe.regrade([job], kept=True, fast=True))] == [None]
    assert len(frontier_swe.read_jsonl(state / "kept-grades.jsonl")) == 1


def test_watch_keeps_promotions_without_grading(pinned, tmp_path, monkeypatch):
    """The `keep` command's loop: every promotion kept within a poll, no grading, stops when run.py exits."""
    task = frontier_swe.build_task("demo-cpu", tmp_path / "tasks", clone=pinned).write()
    job = make_job(tmp_path, task, [], final=None)
    trial = frontier_swe.trials(job)[0]
    (trial / "result.json").unlink()   # still running
    shared = trial / "agent" / "frontier-swe"
    shared.mkdir(parents=True)
    (shared / "ready").write_text("{}")

    async def capture(self, destination):
        (destination / "app").mkdir(parents=True)
        (destination / "app" / "main.zig").write_text(str(len(self.promotions())))

    async def no_grade(*args, **kwargs):
        raise AssertionError("keep must not grade")

    monkeypatch.setattr(frontier_swe.TrialWatch, "capture", capture)
    monkeypatch.setattr(frontier_swe.TrialWatch, "snapshot", no_grade)
    runner = subprocess.Popen(["sleep", "4"])

    async def go():
        watching = asyncio.create_task(frontier_swe.watch(job, task, interval=None, expose=False, watch_pid=runner.pid,
                                                          poll=0.2, keep=True, grade=False))
        for seq in (1, 2):
            await asyncio.sleep(0.8)
            with (shared / "submissions.jsonl").open("a") as stream:
                stream.write(json.dumps({"seq": seq, "status": "promoted", "agent_id": "a1"}) + "\n")
        await asyncio.sleep(0.8)
        runner.wait()
        await asyncio.wait_for(watching, 10)

    asyncio.run(go())
    kept = frontier_swe.read_jsonl(trial / frontier_swe.STATE / "kept.jsonl")
    assert [(r["submission_seq"], r["status"]) for r in kept] == [(1, "kept"), (2, "kept")]
    assert not (trial / frontier_swe.STATE / "snapshots.jsonl").read_text().strip()
