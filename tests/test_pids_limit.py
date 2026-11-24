"""The process cap for every container of a trial (pids_limit.py, run.py --pids-limit)."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pids_limit  # noqa: E402
import run  # noqa: E402

SIDECAR = "harbor-docker-egress-control-sidecar"


def run_py(*argv: str):
    return run.prepare(run.parser().parse_args([*argv, "--dry-run"]))


def without_cap(cmd: list[str]) -> list[str]:
    """The command with run.py's environment arguments taken out."""
    found = list(cmd)
    start = found.index("--env")
    if found[start + 1] == pids_limit.IMPORT_PATH:
        assert found[start + 2] == "--ek" and found[start + 3].startswith(f"{pids_limit.KWARG}=")
        del found[start:start + 4]
    else:
        assert found[start + 1] == "docker"
        del found[start:start + 2]
    return found


def test_option_values():
    assert pids_limit.parse("off") is None and pids_limit.parse("OFF") is None
    assert pids_limit.parse("16384") == 16384 and pids_limit.parse(512) == 512
    for bad in ("0", "-1", "lots", "1.5"):
        with pytest.raises(argparse.ArgumentTypeError):
            pids_limit.parse(bad)
    assert run.parser().parse_args(["ale-bench", "--problem", "ahc014", "--model", "m"]).pids_limit == pids_limit.DEFAULT == 16384
    assert run.parser().parse_args(["ale-bench", "--problem", "ahc014", "--model", "m", "--pids-limit", "off"]).pids_limit is None
    with pytest.raises(SystemExit):
        run.parser().parse_args(["ale-bench", "--problem", "ahc014", "--model", "m", "--pids-limit", "0"])


@pytest.mark.parametrize("argv", [
    ("ale-bench", "--problem", "ahc014", "--cli", "codex", "--model", "m", "--protocol", "antiherding_matched",
     "--name", "pytest-pids-ale"),
    ("perf-takehome", "--cli", "claude", "--model", "m", "--agents", "2", "--name", "pytest-pids-perf"),
])
def test_run_py_command(argv):
    default, _, _ = run_py(*argv)
    at = default.index("--env")
    assert default[at:at + 4] == ["--env", "pids_limit:CappedDockerEnvironment", "--ek", "pids_limit=16384"]
    off, _, _ = run_py(*argv, "--pids-limit", "off")
    at = off.index("--env")
    assert off[at:at + 2] == ["--env", "docker"] and "--ek" not in off
    # Otherwise the command is the same; off is today's command.
    assert without_cap(default) == without_cap(off)
    lower, _, _ = run_py(*argv, "--pids-limit", "4096")
    assert lower[lower.index("--ek") + 1] == "pids_limit=4096" and without_cap(lower) == without_cap(off)


def test_recorded_cap_round_trip():
    config = pids_limit.environment_config(4096, delete=True)
    assert config.import_path == pids_limit.IMPORT_PATH and config.type is None
    assert config.kwargs == {"pids_limit": 4096} and config.delete is True
    assert pids_limit.configured(config) == 4096
    assert pids_limit.configured(json.loads(config.model_dump_json())) == 4096
    plain = pids_limit.environment_config(None, delete=True)
    assert plain.type.value == "docker" and plain.import_path is None and pids_limit.configured(plain) is None
    assert pids_limit.configured({"type": "docker", "kwargs": {}}) is None and pids_limit.configured(None) is None


def compose_files(tmp_path: Path) -> list[Path]:
    task = tmp_path / "docker-compose.yaml"
    task.write_text("services:\n  main:\n    volumes: ['./x:/x:ro']\n"
                    "  scorer:\n    image: busybox\n    pids_limit: 4096\n"
                    "  judge:\n    image: busybox\n    deploy:\n      resources:\n        limits:\n          pids: 2048\n"
                    "  open:\n    image: busybox\n    pids_limit: -1\n")
    egress = tmp_path / "egress.yaml"
    egress.write_text(f"services:\n  {SIDECAR}:\n    image: ${{EGRESS}}\n")
    resources = tmp_path / "resources.json"
    resources.write_text(json.dumps({"services": {"main": {"cpus": 4.0}}}))
    return [resources, task, egress]


def test_override_caps_every_service_and_keeps_lower_limits(tmp_path):
    paths = compose_files(tmp_path)
    assert pids_limit.override(paths, 16384) == {"services": {
        SIDECAR: {"pids_limit": 16384}, "judge": {"pids_limit": 2048, "deploy": {"resources": {"limits": {"pids": 2048}}}},
        "main": {"pids_limit": 16384}, "open": {"pids_limit": 16384}, "scorer": {"pids_limit": 4096}}}
    low = pids_limit.override(paths, 1000)["services"]
    assert {name: entry["pids_limit"] for name, entry in low.items()} == dict.fromkeys(
        (SIDECAR, "judge", "main", "open", "scorer"), 1000)
    assert low["judge"]["deploy"] == {"resources": {"limits": {"pids": 1000}}}


def environment(tmp_path: Path, monkeypatch, limit):
    from harbor.models.task.config import EnvironmentConfig as TaskEnvironment
    from harbor.models.task.config import NetworkMode, NetworkPolicy
    from harbor.models.trial.paths import TrialPaths

    task = tmp_path / "environment"
    task.mkdir(exist_ok=True)
    (task / "docker-compose.yaml").write_text(
        "services:\n  main:\n    volumes: ['./x:/x:ro']\n  scorer:\n    image: busybox:latest\n    pids_limit: 4096\n")
    cls = pids_limit.CappedDockerEnvironment
    monkeypatch.setattr(cls, "_egress_control_kernel_support", staticmethod(lambda: True))
    paths = TrialPaths(trial_dir=tmp_path / "trial")
    paths.mkdir()
    env = cls(environment_dir=task, environment_name="pids", session_id=f"pytest-pids-{limit}__env",
              trial_paths=paths, task_env_config=TaskEnvironment(docker_image="busybox:latest"), pids_limit=limit,
              network_policy=NetworkPolicy(network_mode=NetworkMode.ALLOWLIST, allowed_hosts=["api.anthropic.com"]))
    env._mounts_compose_path = env._write_mounts_compose_file()
    env._resources_compose_path = env._write_resources_compose_file()
    env._env_compose_path = env._write_env_compose_file()
    env._write_egress_control_services_compose_file()
    env._use_prebuilt = True
    env._env_vars.egress_control_sidecar_image_name = "busybox:latest"
    return env


def test_capped_environment_adds_the_override_last(tmp_path, monkeypatch):
    from harbor.environments.docker.docker import DockerEnvironment

    env = environment(tmp_path, monkeypatch, 16384)
    paths = env._docker_compose_paths
    assert paths[-1].name == "pytest-pids-16384__env-docker-compose-pids.json"
    assert paths[-1].parent == env._resources_compose_path.parent  # removed with Harbor's resources override
    # The agents' container also gets a first process that outlives a root agent's `kill -9 -1`
    # (Harbor's compose file gives it `sh -c "sleep infinity"`; docs/harness.md).
    assert json.loads(paths[-1].read_text()) == {"services": {
        SIDECAR: {"pids_limit": 16384}, "main": {"pids_limit": 16384, "command": pids_limit.KEPT_MAIN_COMMAND},
        "scorer": {"pids_limit": 4096}}}
    assert paths[:-1] == DockerEnvironment._docker_compose_paths.fget(env)
    off = environment(tmp_path, monkeypatch, None)
    assert not any(path.name.endswith(pids_limit.COMPOSE_NAME) for path in off._docker_compose_paths)
    env._cleanup_resources_compose_file()
    assert not any(path.name.endswith(pids_limit.COMPOSE_NAME) for path in env._docker_compose_paths)


@pytest.mark.skipif(shutil.which("docker") is None, reason="needs docker compose")
def test_docker_compose_merges_the_cap(tmp_path, monkeypatch):
    """`docker compose config` (starts nothing): the merged project has the cap on every service."""
    env = environment(tmp_path, monkeypatch, 8192)
    done = asyncio.run(env._run_docker_compose_command(["config", "--format", "json"], check=False))
    assert done.return_code == 0, done.stdout
    merged = json.loads(done.stdout[done.stdout.index("{"):])
    assert {name: service.get("pids_limit") for name, service in merged["services"].items()} == {
        "main": 8192, "scorer": 4096, SIDECAR: 8192}
    assert merged["services"]["main"]["command"] == pids_limit.KEPT_MAIN_COMMAND


def test_main_keeps_its_own_command_or_entrypoint(tmp_path):
    """Only Harbor's own first process is replaced: a task that gives the agents' container a command
    or an entrypoint of its own keeps it."""
    harbor = tmp_path / "harbor.yaml"
    harbor.write_text('services:\n  main:\n    command: [ "sh", "-c", "sleep infinity" ]\n')
    assert pids_limit.main_command([harbor]) == pids_limit.HARBOR_MAIN_COMMAND
    assert pids_limit.override([harbor], 100)["services"]["main"] == {
        "pids_limit": 100, "command": ["sh", "-c", "while :; do sleep infinity; done"]}
    own = tmp_path / "task.yaml"
    own.write_text('services:\n  main:\n    command: ["/opt/server", "--port", "1"]\n')
    assert pids_limit.override([harbor, own], 100)["services"]["main"] == {"pids_limit": 100}
    own.write_text('services:\n  main:\n    entrypoint: ["/init"]\n')
    assert pids_limit.main_command([harbor, own]) is None
    assert pids_limit.override([harbor, own], 100)["services"]["main"] == {"pids_limit": 100}


def test_snapshot_grader_gets_the_cap(tmp_path, monkeypatch):
    started = []

    class Popen:
        def __init__(self, argv, **kwargs):
            started.append(argv)

    monkeypatch.setattr(run.subprocess, "Popen", Popen)
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text("[verifier]\ntimeout_sec = 60.0\n")
    cmd = ["harbor", "run", "-p", str(task)]
    for value, expected in ((16384, ["--pids-limit", "16384"]), (None, [])):
        args = run.parser().parse_args(["frontier-swe", "--task", "git-to-zig", "--model", "m"])
        args.pids_limit = value
        run.snapshot_worker(args, cmd, {}, tmp_path / f"job-{value}")
        argv = started[-1]
        assert ("--pids-limit" in argv) == bool(expected)
        if expected:
            assert argv[argv.index("--pids-limit"):argv.index("--pids-limit") + 2] == expected


