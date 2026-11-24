#!/usr/bin/env python3
"""A process cap (pids limit) for every container Harbor starts for this harness's trials.

Without a pids limit, one agent container can take every process slot of the host (a test suite
run inside it can), and then dockerd, other trials' judges and Harbor itself can no longer fork.
run.py's `--pids-limit N|off` (default DEFAULT) sets the cap:

- run.py passes Harbor `--env pids_limit:CappedDockerEnvironment --ek pids_limit=N` instead of
  `--env docker`, so the job's and every trial's config.json record the cap
  (environment.import_path and environment.kwargs.pids_limit). `--pids-limit off` passes
  `--env docker` as before: the same command and the same compose files.
- CappedDockerEnvironment is Harbor's DockerEnvironment with one more compose override, written
  next to Harbor's resources override (<session>-docker-compose-pids.json beside
  <session>-docker-compose-resources.json) and applied after every other compose file: each
  service of the project gets `pids_limit`, the lower of N and the limit the task's own compose
  files set. That covers the main container, the task's scorer/judge sidecars and Harbor's
  egress sidecar, in the agent environment and in a separate verifier environment (Harbor
  creates it from the same environment config).
- The FrontierSWE snapshot grader (frontier_swe.py) grades in Harbor environments too; run.py
  passes it the cap, and environment_config() gives its environments the same class and kwargs.

Nothing else changes: the override only adds pids_limit (and lowers a task's own
deploy.resources.limits.pids to the cap, which compose requires to equal pids_limit) and, for the
agents' container, a first process that outlives a root agent's `kill -9 -1` (KEPT_MAIN_COMMAND).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT = 16384
KWARG = "pids_limit"
CLASS_NAME = "CappedDockerEnvironment"
IMPORT_PATH = f"{Path(__file__).stem}:{CLASS_NAME}"
COMPOSE_NAME = "docker-compose-pids.json"
# The first process of the agents' container. Harbor's is a shell waiting for one `sleep infinity`:
# a root agent's `kill -9 -1` cannot signal PID 1 itself but ends that sleep, and with it the shell,
# the container and the trial, ungraded. The override starts the sleep again instead (docs/harness.md,
# harness protection); PID 1 stays a shell, which reaps orphaned processes as before. Only where the compose
# files give the container Harbor's own command and no entrypoint.
MAIN_SERVICE = "main"
HARBOR_MAIN_COMMAND = ["sh", "-c", "sleep infinity"]
KEPT_MAIN_COMMAND = ["sh", "-c", "while :; do sleep infinity; done"]


def parse(value: str | int) -> int | None:
    """--pids-limit: a positive number of processes, or off (None)."""
    if isinstance(value, str) and value.strip().lower() == "off":
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"must be a positive number or off, not {value!r}") from None
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive number or off")
    return number


def harbor_args(limit: int | None) -> list[str]:
    """Harbor's environment arguments for run.py: the capped environment, or plain docker."""
    if limit is None:
        return ["--env", "docker"]
    return ["--env", IMPORT_PATH, "--ek", f"{KWARG}={int(limit)}"]


def configured(environment: Any) -> int | None:
    """The cap a job's or trial's environment config (config.json's "environment", a dict or
    Harbor's model) gives, or None."""
    if environment is None:
        return None
    get = environment.get if isinstance(environment, dict) else (lambda key: getattr(environment, key, None))
    if get("import_path") != IMPORT_PATH:
        return None
    value = (get("kwargs") or {}).get(KWARG)
    return None if value is None else parse(value)


def environment_config(limit: int | None, tmpfs: list[str] | None = None, **fields: Any):
    """Harbor's TrialEnvironmentConfig for a docker environment, capped when limit is set
    (the snapshot grader's environments). tmpfs (needs a limit): compose tmpfs entries for
    the main container, e.g. ["/tmp:rw,exec,nosuid,size=6g,mode=1777"], so a grading's scratch stays in
    RAM (charged to the container's memory limit) instead of the disk."""
    from harbor.models.trial.config import EnvironmentConfig

    if limit is None:
        if tmpfs:
            raise ValueError("tmpfs needs a pids limit (the capped environment writes it)")
        return EnvironmentConfig(type="docker", **fields)
    kwargs: dict[str, Any] = {KWARG: int(limit)}
    if tmpfs:
        kwargs["tmpfs"] = list(tmpfs)
    return EnvironmentConfig(type=None, import_path=IMPORT_PATH, kwargs=kwargs, **fields)


def _number(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None  # unset, or a ${VARIABLE} compose resolves later: not a limit we can read
    return number if number > 0 else None  # 0 and -1 mean no limit to Docker


def own_limits(paths: list[Path]) -> dict[str, dict[str, int | None]]:
    """Each service the compose files define, with the pids limits they set themselves
    (pids_limit and deploy.resources.limits.pids; a later file overrides an earlier one)."""
    import yaml

    services: dict[str, dict[str, int | None]] = {}
    for path in paths:
        try:
            document = yaml.safe_load(Path(path).read_text())
        except (OSError, yaml.YAMLError):
            continue
        found = document.get("services") if isinstance(document, dict) else None
        if not isinstance(found, dict):
            continue
        for name, service in found.items():
            entry = services.setdefault(str(name), {"pids_limit": None, "deploy": None})
            if not isinstance(service, dict):
                continue
            if "pids_limit" in service:
                entry["pids_limit"] = _number(service["pids_limit"])
            limits = ((service.get("deploy") or {}).get("resources") or {}).get("limits") or {}
            if isinstance(limits, dict) and "pids" in limits:
                entry["deploy"] = _number(limits["pids"])
    return services


def main_command(paths: list[Path]):
    """The command the compose files give the agents' container (a later file overrides an earlier
    one), or None if one of them sets its entrypoint."""
    import yaml

    command = None
    for path in paths:
        try:
            document = yaml.safe_load(Path(path).read_text())
        except (OSError, yaml.YAMLError):
            continue
        service = ((document.get("services") if isinstance(document, dict) else None) or {}).get(MAIN_SERVICE)
        if not isinstance(service, dict):
            continue
        if "entrypoint" in service:
            return None
        if "command" in service:
            command = service["command"]
    return command


def override(paths: list[Path], limit: int, tmpfs: list[str] | None = None) -> dict:
    """The compose override that caps every service of these compose files at limit, and keeps the
    agents' container alive through a root agent's `kill -9 -1` (KEPT_MAIN_COMMAND); with tmpfs,
    those RAM mounts on the main container too (environment_config)."""
    services = {}
    for name, own in sorted(own_limits(paths).items()):
        values = [value for value in own.values() if value is not None]
        value = min([int(limit), *values])
        entry: dict[str, Any] = {"pids_limit": value}
        if "deploy" in own and own["deploy"] is not None:
            entry["deploy"] = {"resources": {"limits": {"pids": value}}}
        services[name] = entry
    if main_command(paths) == HARBOR_MAIN_COMMAND:
        services.setdefault(MAIN_SERVICE, {})["command"] = KEPT_MAIN_COMMAND
    if tmpfs:
        services.setdefault(MAIN_SERVICE, {})["tmpfs"] = list(tmpfs)
    return {"services": services}


def _capped_class():
    from harbor.environments.docker.docker import DockerEnvironment

    class CappedDockerEnvironment(DockerEnvironment):
        """Harbor's DockerEnvironment whose compose project caps every service's processes."""

        def __init__(self, *args: Any, pids_limit: int | str | None = None, tmpfs: list[str] | None = None,
                     **kwargs: Any) -> None:
            self._pids_limit = None if pids_limit is None else parse(pids_limit)
            self._tmpfs = list(tmpfs) if tmpfs else None
            self._pids_compose_path: Path | None = None
            super().__init__(*args, **kwargs)

        def _write_resources_compose_file(self) -> Path | None:
            # Harbor writes the resources override before every `compose up` (and revive.py calls
            # this to regenerate the overrides of a trial whose /tmp files are gone); the cap is
            # written next to it, in the same temporary directory, so it goes with it.
            path = super()._write_resources_compose_file()
            self._pids_compose_path = None
            if path is not None and self._pids_limit is not None:
                others = [p for p in super()._docker_compose_paths if p != path]
                capped = Path(path).with_name(f"{self.session_id}-{COMPOSE_NAME}")
                capped.write_text(json.dumps(override([path, *others], self._pids_limit, self._tmpfs), indent=2))
                self._pids_compose_path = capped
            return path

        @property
        def _docker_compose_paths(self) -> list[Path]:
            paths = super()._docker_compose_paths
            if self._pids_compose_path is not None and self._pids_compose_path.is_file():
                paths.append(self._pids_compose_path)  # last: the cap wins over every other file
            return paths

    CappedDockerEnvironment.__module__ = __name__
    CappedDockerEnvironment.__qualname__ = CLASS_NAME
    return CappedDockerEnvironment


_CLASS = None


def __getattr__(name: str):
    # Harbor imports the class by its import path; run.py imports this module for the option
    # alone, without loading Harbor.
    global _CLASS
    if name == CLASS_NAME:
        if _CLASS is None:
            _CLASS = _capped_class()
        return _CLASS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
