#!/usr/bin/env python3
"""FrontierSWE harness tools inside the task container (Python 3.8+, standard library only).

The adapter mounts this directory read-only at /opt/fswe, next to config.json, which
names the task's graded paths (its [[artifacts]] entries with their tar excludes).

Agents work in a private copy of the graded files and promote it into /app, the
graded workspace:

    workcopy                                  create /work/$AGENT_ID from /app and print it
    submit --score S [--note TEXT] [--force]  promote that copy into /app

submit copies the working copy's graded files into /app only if S, the agent's local
self-check result, is at least the score of the version /app holds (--force skips the
check), and logs every call in /logs/agent/frontier-swe/submissions.jsonl. Promotions
hold /work/.fswe/promote.lock, which the snapshot grader also takes before it copies
/app, so nobody ever reads a half-promoted /app.

The agent installer runs `fswe.py setup` as root, and the runner calls `fswe.py finish`
when the agents stop, so a promotion still in flight completes before Harbor collects
/app.

Which paths are graded follows Harbor's artifact download exactly: a directory entry is
`tar czf ... --exclude=PATTERN ... -C SOURCE .`, so GNU tar's exclusion rules apply
(see tar_excluded).
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import fnmatch
import json
import math
import os
import pwd
import shutil
import signal
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config.json"
TOOLS = ("submit", "workcopy")
# Environment variables whose holders the token-budget watcher stops; a promotion
# drops them so a budget stop cannot interrupt it half way.
WATCHED_ENV = ("CODEX_HOME", "CLAUDE_CONFIG_DIR")
FINISH_WAIT_SEC = 900


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def load_config() -> dict:
    return json.loads(CONFIG.read_text())


# ---------------------------------------------------------------------------
# Graded paths


def tar_excluded(relative: str, patterns) -> bool:
    """Whether GNU tar, archiving `-C root .` with these --exclude patterns, leaves out
    the member at `relative` (a POSIX path below root, without "./").

    tar names members "./a/b", matches exclusion patterns with wildcards that also
    match "/", unanchored (against the whole name or any part after a "/"), and does
    not descend into an excluded directory, so an excluded ancestor excludes too."""
    if not patterns or relative in ("", "."):
        return False
    parts = relative.split("/")
    for depth in range(1, len(parts) + 1):
        name = "./" + "/".join(parts[:depth])
        candidates = [name] + [name[index + 1:] for index, char in enumerate(name) if char == "/"]
        for pattern in patterns:
            if any(fnmatch.fnmatchcase(candidate, pattern) for candidate in candidates):
                return True
    return False


class Graded:
    """The graded paths of one task, relative to app_root (normally /app)."""

    def __init__(self, entries, app_root: str = "/app", extra_exclude=()):
        self.app_root = app_root.rstrip("/") or "/"
        self.extra_exclude = list(extra_exclude)
        self.entries = []
        for entry in entries:
            source = entry["source"]
            relative = os.path.relpath(source.rstrip("/") or "/", self.app_root)
            if relative.startswith(".."):
                raise ValueError(f"graded path {source} is not under {self.app_root}")
            relative = "" if relative == "." else relative
            self.entries.append((relative, list(entry.get("exclude") or []), source.endswith("/")))

    def _is_dir_entry(self, root: Path, relative: str, slash: bool) -> bool:
        path = root / relative if relative else root
        if path.is_dir() and not path.is_symlink():
            return True
        if path.exists() or path.is_symlink():
            return False
        # Harbor's fallback when the source is missing: no suffix means a directory.
        return slash or not Path(relative).suffix

    def tree(self, root) -> tuple[set, dict]:
        """(directories, files) that count as graded under root, which mirrors
        app_root. files maps each path to "file" or "link"."""
        root = Path(root)
        dirs, files = set(), {}
        for relative, exclude, slash in self.entries:
            patterns = exclude + self.extra_exclude
            base = root / relative if relative else root
            if not self._is_dir_entry(root, relative, slash):
                if base.is_symlink():
                    files[relative] = "link"
                elif base.is_file():
                    files[relative] = "file"
                continue
            if not base.is_dir():
                continue
            if relative:
                dirs.add(relative)
            for current, dirnames, filenames in os.walk(base):
                inner_dir = os.path.relpath(current, base)
                inner_dir = "" if inner_dir == "." else inner_dir
                kept = []
                for name in sorted(dirnames):
                    inner = f"{inner_dir}/{name}" if inner_dir else name
                    full = os.path.join(current, name)
                    if tar_excluded(inner, patterns):
                        continue
                    target = f"{relative}/{inner}" if relative else inner
                    if os.path.islink(full):
                        files[target] = "link"
                    else:
                        dirs.add(target)
                        kept.append(name)
                dirnames[:] = kept
                for name in filenames:
                    inner = f"{inner_dir}/{name}" if inner_dir else name
                    if tar_excluded(inner, patterns):
                        continue
                    full = os.path.join(current, name)
                    target = f"{relative}/{inner}" if relative else inner
                    if os.path.islink(full):
                        files[target] = "link"
                    elif os.path.isfile(full):
                        files[target] = "file"
        return dirs, files


def _same(source: Path, target: Path, kind: str) -> bool:
    if kind == "link":
        return target.is_symlink() and os.readlink(source) == os.readlink(target)
    if target.is_symlink() or not target.is_file():
        return False
    first, second = source.stat(), target.stat()
    if first.st_size != second.st_size or stat.S_IMODE(first.st_mode) != stat.S_IMODE(second.st_mode):
        return False
    with source.open("rb") as one, target.open("rb") as two:
        while True:
            a, b = one.read(1 << 20), two.read(1 << 20)
            if a != b:
                return False
            if not a:
                return True


def _replace(source: Path, target: Path, kind: str) -> None:
    """Copy one file or link so that readers see either the old or the new version."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    temporary = target.parent / f".fswe-tmp-{os.getpid()}-{target.name}"
    if temporary.is_symlink() or temporary.exists():
        temporary.unlink()
    if kind == "link":
        os.symlink(os.readlink(source), temporary)
    else:
        shutil.copy2(source, temporary)
    os.replace(temporary, target)


def sync(source_root, target_root, graded: Graded) -> dict:
    """Make target_root's graded paths equal to source_root's; leave the rest alone."""
    source_root, target_root = Path(source_root), Path(target_root)
    source_dirs, source_files = graded.tree(source_root)
    target_dirs, target_files = graded.tree(target_root)
    changed = deleted = 0
    for relative in sorted(source_dirs):
        path = target_root / relative
        if path.is_symlink() or (path.exists() and not path.is_dir()):
            path.unlink()
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(path, stat.S_IMODE((source_root / relative).stat().st_mode))
        except OSError:
            pass
    for relative, kind in sorted(source_files.items()):
        if not _same(source_root / relative, target_root / relative, kind):
            _replace(source_root / relative, target_root / relative, kind)
            changed += 1
    for relative in sorted(set(target_files) - set(source_files)):
        path = target_root / relative
        if path.is_symlink() or path.is_file():
            path.unlink()
            deleted += 1
    for relative in sorted(target_dirs - source_dirs, reverse=True):
        try:
            (target_root / relative).rmdir()  # only if nothing ungraded is left inside
        except OSError:
            pass
    return {"files_changed": changed, "files_deleted": deleted, "graded_files": len(source_files)}


# ---------------------------------------------------------------------------
# Shared state: lock, journal, submission log


class Lock:
    def __init__(self, path, timeout: float | None = None):
        self.path, self.timeout = Path(path), timeout

    def __enter__(self):
        self.handle = open(self.path, "a")
        if self.timeout is None:
            fcntl.flock(self.handle, fcntl.LOCK_EX)
            return self
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return self
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EACCES) or time.monotonic() > deadline:
                    self.handle.close()
                    raise
                time.sleep(0.5)

    def __exit__(self, *exc_info):
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


class State:
    def __init__(self, config: dict):
        self.config = config
        self.app = Path(config["app_root"])
        self.work_root = Path(config["work_root"])
        self.log_dir = Path(config["log_dir"])
        self.lock = Path(config["lock"])
        self.journal = self.lock.parent / "pending-promotion.json"
        self.log = self.log_dir / "submissions.jsonl"
        self.graded = Graded(config["graded"], config["app_root"], config.get("promote_exclude") or ())

    def work(self, agent: str) -> Path:
        return self.work_root / agent

    def records(self) -> list[dict]:
        if not self.log.is_file():
            return []
        found = []
        for line in self.log.read_text(errors="replace").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                found.append(record)
        return found

    def append(self, record: dict) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with self.log.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def finish_pending(self) -> dict | None:
        """Redo a promotion that was interrupted part way (caller holds the lock)."""
        if not self.journal.is_file():
            return None
        pending = json.loads(self.journal.read_text())
        result = sync(pending["source"], self.app, self.graded)
        self.journal.unlink()
        return {**pending, **result}


def agent_id(explicit: str | None) -> str:
    agent = explicit or os.environ.get("AGENT_ID") or "single"
    if not agent.replace("-", "").replace("_", "").isalnum():
        raise SystemExit(f"unusable agent id {agent!r}")
    return agent


# ---------------------------------------------------------------------------
# Agent commands


def cmd_workcopy(args) -> int:
    state = State(load_config())
    work = state.work(agent_id(args.agent))
    if not work.is_dir():
        staging = work.with_name(work.name + ".partial")
        with Lock(state.lock):
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            sync(state.app, staging, state.graded)
            staging.rename(work)
    print(work)
    return 0


def parse_score(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        value = float(text)
    except ValueError:
        raise SystemExit(f"--score must be a number, not {text!r}")
    if not math.isfinite(value):
        raise SystemExit("--score must be finite")
    return value


def cmd_submit(args) -> int:
    config = load_config()
    state = State(config)
    agent = agent_id(args.agent)
    work = state.work(agent)
    if not work.is_dir():
        print(f"no working copy at {work}: run `workcopy` first, then edit and test there", file=sys.stderr)
        return 2
    request = {"agent_id": agent, "score": parse_score(args.score), "note": args.note or "",
               "force": bool(args.force), "source": str(work)}
    # The promotion runs in its own session without the CLI's home in its environment,
    # so a token-budget stop or a killed shell cannot interrupt it half way.
    environment = {key: value for key, value in os.environ.items() if key not in WATCHED_ENV}
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_promote", json.dumps(request)],
                               env=environment, start_new_session=True, stdout=subprocess.PIPE)
    output, _ = process.communicate()
    sys.stdout.write(output.decode(errors="replace"))
    return process.returncode


def cmd_promote(args) -> int:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    request = json.loads(args.request)
    config = load_config()
    state = State(config)
    with Lock(state.lock):
        state.finish_pending()
        records = state.records()
        promoted = [r for r in records if r.get("status") == "promoted"]
        current = promoted[-1] if promoted else None
        held = current.get("local_score") if current else None
        record = {"seq": 1 + max((int(r.get("seq") or 0) for r in records), default=0), "ts": now(),
                  "agent_id": request["agent_id"], "local_score": request["score"], "note": request["note"],
                  "forced": request["force"], "source": request["source"],
                  "previous_seq": current.get("seq") if current else None, "previous_score": held}
        reason = None
        if not request["force"] and held is not None:
            if request["score"] is None:
                reason = (f"/app holds submission #{current['seq']} with local score {held}; "
                          "give --score (or --force)")
            elif request["score"] < held:
                reason = (f"local score {request['score']} is below {held}, the score of submission "
                          f"#{current['seq']} by {current.get('agent_id')} that /app holds")
        if reason:
            record.update(status="rejected", reason=reason)
            state.append(record)
            print(f"[submit] #{record['seq']} rejected: {reason}. Use --force to promote anyway.")
            return 1
        state.journal.write_text(json.dumps({"seq": record["seq"], "source": request["source"]}))
        record.update(sync(request["source"], state.app, state.graded))
        state.journal.unlink()
        record["status"] = "promoted"
        state.append(record)
    print(f"[submit] #{record['seq']} promoted into {state.app}: {record['files_changed']} file(s) "
          f"changed, {record['files_deleted']} removed (local score {request['score']}).")
    if config.get("expose_scores"):
        print(f"[submit] Verifier scores of /app appear in {state.log_dir / 'verifier-scores.jsonl'} "
              "once graded (minutes).")
    return 0


def cmd_locked(args) -> int:
    """Run a command holding the promotion lock (the snapshot grader copies /app so)."""
    command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not command:
        raise SystemExit("locked needs a command")
    state = State(load_config())
    with Lock(state.lock, timeout=args.timeout):
        return subprocess.call(command)


def cmd_finish(args) -> int:
    """End of the agent run: let any promotion finish, then mark the run finished."""
    state = State(load_config())
    try:
        with Lock(state.lock, timeout=args.timeout):
            done = state.finish_pending()
    except OSError as exc:
        print(f"[finish] could not take {state.lock}: {exc}", file=sys.stderr)
        return 1
    if done:
        print(f"[finish] completed interrupted promotion #{done.get('seq')}")
    state.log_dir.mkdir(parents=True, exist_ok=True)
    (state.log_dir / "finished").write_text(now() + "\n")
    return 0


# ---------------------------------------------------------------------------
# Root setup and the Codex login relay


def _can_read_write(path: Path, uid: int, gids: set) -> bool:
    info = path.stat()
    mode = info.st_mode
    if info.st_uid == uid:
        return bool(mode & stat.S_IRUSR and mode & stat.S_IWUSR)
    if info.st_gid in gids:
        return bool(mode & stat.S_IRGRP and mode & stat.S_IWGRP)
    return bool(mode & stat.S_IROTH and mode & stat.S_IWOTH)


def _write_in_place(path: Path, data: bytes) -> None:
    # A bind-mounted file cannot be replaced by a rename, and the CLI rereads it.
    with open(path, "r+b") as stream:
        stream.seek(0)
        stream.write(data)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())


def _valid_login(data: bytes) -> bool:
    try:
        return isinstance(json.loads(data), dict)
    except ValueError:
        return False


def relay(host: Path, target: Path, interval: float = 2.0) -> None:
    """Keep target (the agent's copy) and host (the shared login) equal; whichever
    changed since the last sync wins, so one refresh-token chain survives."""
    last = host.read_bytes()
    while True:
        time.sleep(interval)
        try:
            ours, theirs = target.read_bytes(), host.read_bytes()
        except OSError:
            continue
        if ours != last and _valid_login(ours):
            _write_in_place(host, ours)
            last = ours
        elif theirs != last and _valid_login(theirs):
            _write_in_place(target, theirs)
            last = theirs


def _daemonize(function, *args) -> None:
    if os.fork():
        return
    os.setsid()
    if os.fork():
        os._exit(0)
    null = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(null, fd)
    try:
        function(*args)
    finally:
        os._exit(0)


def setup_codex_login(auth: dict, user: pwd.struct_passwd) -> str:
    host, target = Path(auth["host"]), Path(auth["target"])
    if not host.is_file():
        return "absent"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    gids = {user.pw_gid, *(os.getgrouplist(user.pw_name, user.pw_gid) if hasattr(os, "getgrouplist") else ())}
    if _can_read_write(host, user.pw_uid, gids):
        target.symlink_to(host)
        return "shared"
    # The agent user cannot open the shared login (the image's agent uid differs from
    # the file owner's), so it gets its own copy that a root relay keeps in sync.
    shutil.copy2(host, target)
    os.chown(target, user.pw_uid, user.pw_gid)
    os.chmod(target, 0o600)
    _daemonize(relay, host, target)
    return "relayed"


def install_files(items) -> None:
    """Copy harness files over task files of the image (config "install": source, a path
    under this directory; target, absolute; mode, octal)."""
    for item in items:
        source, target = HERE / item["source"], Path(item["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or (target.exists() and not target.is_file()):
            raise SystemExit(f"{target} is not a regular file in this image; cannot replace it")
        shutil.copyfile(source, target)
        os.chmod(target, int(str(item.get("mode", "644")), 8))


def cmd_setup(args) -> int:
    config = load_config()
    state = State(config)
    user = pwd.getpwnam(config["agent_user"])
    for directory in (state.work_root, state.lock.parent, state.log_dir):
        directory.mkdir(parents=True, exist_ok=True)
        os.chown(directory, user.pw_uid, user.pw_gid)
    # The log dir is a host mount; the snapshot grader (the host user) may write there.
    os.chmod(state.log_dir, 0o777)
    state.lock.touch()
    os.chown(state.lock, user.pw_uid, user.pw_gid)
    os.chmod(state.lock, 0o666)
    for tool in TOOLS:
        link, wanted = Path("/usr/local/bin") / tool, HERE / "bin" / tool
        if link.is_symlink() and os.readlink(link) == str(wanted):
            continue
        if link.exists() or link.is_symlink():
            raise SystemExit(f"{link} already exists in this image; cannot install the harness tool")
        link.symlink_to(wanted)
    install_files(config.get("install") or [])
    login = setup_codex_login(config["codex_auth"], user) if config.get("codex_auth") else "none"
    ready = {"ts": now(), "agent_user": user.pw_name, "uid": user.pw_uid, "codex_login": login}
    (state.log_dir / "ready").write_text(json.dumps(ready, sort_keys=True) + "\n")
    os.chown(state.log_dir / "ready", user.pw_uid, user.pw_gid)
    print(json.dumps(ready, sort_keys=True))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    workcopy = commands.add_parser("workcopy", help="create (once) and print your working copy")
    workcopy.add_argument("--agent", help="agent id (default: $AGENT_ID)")
    submit = commands.add_parser("submit", help="promote your working copy into /app")
    submit.add_argument("--score", help="your copy's result on the task's own self-check (higher is better)")
    submit.add_argument("--note", help="what changed")
    submit.add_argument("--force", action="store_true", help="promote even if --score is lower")
    submit.add_argument("--agent", help="agent id (default: $AGENT_ID)")
    promote = commands.add_parser("_promote")
    promote.add_argument("request")
    finish = commands.add_parser("finish")
    finish.add_argument("--timeout", type=float, default=FINISH_WAIT_SEC)
    commands.add_parser("setup")
    locked = commands.add_parser("locked", help="run a command holding the promotion lock")
    locked.add_argument("--timeout", type=float, default=FINISH_WAIT_SEC)
    locked.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    handler = {"workcopy": cmd_workcopy, "submit": cmd_submit, "_promote": cmd_promote,
               "finish": cmd_finish, "setup": cmd_setup, "locked": cmd_locked}[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
