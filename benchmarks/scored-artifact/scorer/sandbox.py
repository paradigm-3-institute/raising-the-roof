"""Run untrusted programs inside the scorer container.

A benchmark's scoring module calls Sandbox.run for every step that executes code
the agents wrote (for example building a kernel from their Python). The program
runs as an unprivileged user that cannot read the scorer's configuration, write to
its state, or touch the pristine data, with limits on memory, CPU time, file size
and processes, in a fresh directory that holds only the files it is given. When it
ends, every process of that user is killed, so nothing it left behind can interfere
with the trusted steps that follow. Each scorer worker has its own user, so
concurrent gradings never kill each other's programs.
"""

from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

OUTPUT_TAIL = 4000  # characters of stdout/stderr kept for feedback


@dataclass
class Outcome:
    returncode: int | None  # None: killed at the time limit
    timed_out: bool
    seconds: float
    stdout: str
    stderr: str
    directory: Path


def _tail(stream) -> str:
    """The end of an output file, read through the descriptor the run wrote to: the
    sandboxed user owns its directory and could replace the file's name with a link to
    something only the scorer may read."""
    try:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(size - OUTPUT_TAIL, 0))
        return stream.read().decode("utf-8", "replace")
    except OSError:
        return ""


def processes_of(uid: int) -> list[int]:
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid == uid:
                found.append(int(proc.name))
        except OSError:
            pass
    return found


def kill_user(uid: int) -> None:
    """SIGKILL every process owned by uid until none is left."""
    for _ in range(50):
        pids = processes_of(uid)
        if not pids:
            return
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(0.1)


class Sandbox:
    def __init__(self, uid: int, root: Path, memory_mb: int = 4096, max_file_mb: int = 512,
                 max_processes: int = 64) -> None:
        self.uid = uid
        self.root = Path(root)
        self.memory_mb = memory_mb
        self.max_file_mb = max_file_mb
        self.max_processes = max_processes
        self.enabled = os.geteuid() == 0  # switching users needs root (tests run without)

    def directory(self, name: str) -> Path:
        """A fresh, empty directory the sandboxed user owns."""
        path = self.root / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)
        if self.enabled:
            os.chown(path, self.uid, self.uid)
            path.chmod(0o700)
        return path

    def _limits(self, cpu_seconds: int):
        memory = self.memory_mb * 2**20
        file_size = self.max_file_mb * 2**20
        processes = self.max_processes

        def apply() -> None:
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 5))
            resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))
            resource.setrlimit(resource.RLIMIT_NPROC, (processes, processes))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            os.umask(0o077)

        return apply

    def run(self, argv: list[str], directory: Path, timeout: float) -> Outcome:
        """Run argv in directory (from self.directory) as the sandboxed user."""
        stdout, stderr = directory / ".stdout", directory / ".stderr"
        env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(directory), "LANG": "C.UTF-8",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
        kwargs = {}
        if self.enabled:
            kwargs = {"user": self.uid, "group": self.uid, "extra_groups": []}
        start = time.monotonic()
        with stdout.open("w+b") as out, stderr.open("w+b") as err:
            if self.enabled:
                os.chown(stdout, self.uid, self.uid)
                os.chown(stderr, self.uid, self.uid)
            process = subprocess.Popen(argv, cwd=directory, env=env, stdin=subprocess.DEVNULL,
                                       stdout=out, stderr=err, start_new_session=True,
                                       preexec_fn=self._limits(int(timeout) + 1), **kwargs)
            timed_out = False
            try:
                returncode = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                returncode = None
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    pass
                process.wait()
            finally:
                if self.enabled:
                    kill_user(self.uid)
            tails = _tail(out), _tail(err)
        return Outcome(returncode, timed_out, time.monotonic() - start, *tails, directory)
