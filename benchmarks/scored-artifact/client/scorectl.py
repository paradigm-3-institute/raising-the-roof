#!/usr/bin/env python3
"""Agent-side commands of a scored-artifact task (Python standard library only).

    workcopy                        create your working copy (/work/$AGENT_ID) once; print its path
    submit [PATH] [--note TEXT]     grade your working copy, or PATH (a directory, or the
                                    artifact file when it is a single file), and print the score
    scores [--all]                  every graded submission of the run, best first
    fetch-submission SID|best DEST  copy a graded submission's files into DEST

The scorer runs in another container and keeps the log the run is graded on; these
commands only talk to it. Exit codes of submit: 0 graded (valid or not), 2 missing
files, 3 the scorer refused or could not be reached, 4 no result in time.

Tree artifacts (task.json "tree"): the agents work in the task's
own workspace (a single agent in the workspace itself, such as /app; each agent of a team
in a copy of it, /work/$AGENT_ID), and a submission is everything under the task's
artifact paths in that workspace, sent as one archive laid out at the declared paths.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

CONFIG = Path(os.environ.get("SCORED_ARTIFACT_CONFIG", "/opt/scored-artifact/task.json"))


def config() -> dict:
    return json.loads(CONFIG.read_text())


def agent_id() -> str:
    return os.environ.get("AGENT_ID") or "single"


def urls(task: dict) -> list[str]:
    return [url.rstrip("/") for url in task["scorer_urls"]]


def request(task: dict, method: str, path: str, body: dict | None = None, timeout: float = 60):
    """(status, JSON reply) from the first scorer address that answers."""
    data = json.dumps(body).encode() if body is not None else None
    last = None
    for base in urls(task):
        req = urllib.request.Request(base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as reply:
                return reply.status, json.loads(reply.read() or b"{}")
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read() or b"{}")
            except ValueError:
                return exc.code, {"error": str(exc)}
        except (urllib.error.URLError, OSError, ValueError) as exc:
            last = exc
    raise ConnectionError(f"cannot reach the scorer ({last})")


def workcopy_path(task: dict) -> Path:
    tree = task.get("tree")
    if tree and not tree.get("team"):
        return Path(tree["workspace"])
    if tree and agent_id() in ("", ".", ".."):
        raise ValueError(f"bad AGENT_ID {agent_id()!r}")
    return Path(task["work_root"]) / agent_id()


def tree_workcopy(task: dict, path: Path) -> None:
    """A team agent's copy of the task's workspace, as it is in the image (git repository
    included): symbolic links stay links, modes and times are kept."""
    staging = path.with_name(path.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(task["tree"]["workspace"], staging, symlinks=True)
    staging.rename(path)


def tree_archive(task: dict, workspace: Path) -> str:
    """The submission of a tree artifact: everything under the artifact paths of the workspace,
    as a gzipped tar laid out at the declared paths (app/output/... for /app/output), with their
    permissions, base64.
    Raises FileNotFoundError for anything a snapshot cannot hold (as upstream refuses symbolic
    links in a snapshot) or a submission over the size limit."""
    buffer = io.BytesIO()
    total, limit = 0, int(task["tree"].get("max_bytes") or 0)
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for artifact in task["tree"]["artifacts"]:
            source = workspace / artifact["relative"]
            name = artifact["declared"].lstrip("/")
            if not os.path.lexists(source):
                continue  # as upstream, a missing artifact is left out of the snapshot
            paths = [source]
            if source.is_dir() and not source.is_symlink():
                paths += sorted(source.rglob("*"))
            seen = {}
            for path in paths:
                mode = path.lstat()
                arcname = name if path == source else f"{name}/{path.relative_to(source).as_posix()}"
                if stat.S_ISLNK(mode.st_mode):
                    raise FileNotFoundError(f"{path} is a symbolic link; snapshots hold only regular files "
                                            "and directories (copy the file instead)")
                if stat.S_ISDIR(mode.st_mode):
                    info = tarfile.TarInfo(arcname)
                    info.type, info.mode, info.mtime = tarfile.DIRTYPE, mode.st_mode & 0o777, mode.st_mtime
                    archive.addfile(info)
                    continue
                if not stat.S_ISREG(mode.st_mode):
                    raise FileNotFoundError(f"{path} is not a regular file or a directory")
                if mode.st_nlink > 1 or (mode.st_dev, mode.st_ino) in seen:
                    raise FileNotFoundError(f"{path} is a hard link (st_nlink {mode.st_nlink}); the graders "
                                            "reject hard links: make it a copy of its own")
                seen[(mode.st_dev, mode.st_ino)] = path
                total += mode.st_size
                if limit and total > limit:
                    raise FileNotFoundError(f"the artifact paths hold more than {limit} bytes")
                info = tarfile.TarInfo(arcname)
                info.size, info.mode, info.mtime = mode.st_size, mode.st_mode & 0o777, mode.st_mtime
                with open(path, "rb") as stream:
                    archive.addfile(info, stream)
    return base64.b64encode(buffer.getvalue()).decode()


def show_resources(task: dict) -> None:
    """The agent's remaining budget after its feedback, when the run has the `resources` command."""
    command = task.get("resources_command")
    if not command or not os.access(command, os.X_OK):
        return
    try:
        done = subprocess.run([command], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return
    if done.returncode == 0 and done.stdout.strip():
        print("[submit] remaining budget (`resources`):")
        print(done.stdout.rstrip())


def cmd_workcopy(task: dict, args) -> int:
    path = workcopy_path(task)
    if task.get("tree"):
        if not path.exists():
            tree_workcopy(task, path)
        print(path)
        return 0
    if not path.exists():
        staging = path.with_name(path.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(task["starter"], staging)
        for item in [staging, *staging.rglob("*")]:
            item.chmod(0o755 if item.is_dir() else 0o644)
        if shutil.which("git"):
            # A git history of the starter, so `git diff` shows your changes.
            git = ["git", "-C", str(staging), "-c", "user.name=starter", "-c", "user.email=starter@localhost"]
            subprocess.run([*git[:3], "init", "-q"], check=False)
            subprocess.run([*git, "add", "-A"], check=False)
            subprocess.run([*git, "commit", "-q", "-m", "starter"], check=False)
        staging.rename(path)
    print(path)
    return 0


def read_artifact(task: dict, target: Path) -> dict[str, str]:
    names = task["artifact_files"]
    if target.is_file():
        if len(names) != 1:
            raise FileNotFoundError(f"a submission is {len(names)} files ({', '.join(names)}); "
                                    "pass the directory that holds them")
        return {names[0]: base64.b64encode(target.read_bytes()).decode()}
    files = {}
    for name in names:
        path = target / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")
        files[name] = base64.b64encode(path.read_bytes()).decode()
    return files


def describe(record: dict, task: dict) -> str:
    if record.get("status") in ("queued", "running"):
        return f"sid={record['sid']} {record['status']}"
    if record.get("status") == "error":
        return f"sid={record['sid']} scorer error: {record.get('detail')}"
    if record.get("valid"):
        return (f"sid={record['sid']} valid {task['metric_name']}={record.get('metric')} "
                f"score={float(record.get('score') or 0):.4f}")
    return f"sid={record['sid']} INVALID score=0"


def candidates(records: list[dict]) -> list[dict]:
    """The valid graded submissions that can be the run's grade (not feedback-only ones: "eligible" false)."""
    return [r for r in records if r.get("status") == "done" and r.get("valid") and r.get("eligible") is not False]


def rank(task: dict | None):
    """Sort key, best first: by score, or (task "rank_by" "metric") by the raw metric; earliest first among equals."""
    if (task or {}).get("rank_by") == "metric":
        sign = 1 if task.get("metric_better", "lower") == "lower" else -1
        return lambda r: (r.get("metric") is None, sign * float(r.get("metric") or 0), r.get("ts") or "",
                          int(r["sid"]))
    return lambda r: (-float(r.get("score") or 0), int(r["sid"]))


def best(records: list[dict], task: dict | None = None) -> dict | None:
    return min(candidates(records), key=rank(task), default=None)


def cmd_submit(task: dict, args) -> int:
    target = Path(args.path) if args.path else workcopy_path(task)
    try:
        if task.get("tree"):
            if not target.is_dir():
                raise FileNotFoundError(f"{target} is not a directory")
            payload = {"archive": tree_archive(task, target)}
        else:
            payload = {"files": read_artifact(task, target)}
    except FileNotFoundError as exc:
        hint = "" if args.path or target.exists() else " (run `workcopy` first)"
        print(f"[submit] ERROR: {exc}{hint}", file=sys.stderr)
        return 2
    try:
        status, reply = request(task, "POST", "/submit", {"agent_id": agent_id(), **payload,
                                                          "note": args.note or ""})
    except ConnectionError as exc:
        print(f"[submit] ERROR: {exc}", file=sys.stderr)
        return 3
    if status != 200:
        print(f"[submit] ERROR: {reply.get('error') or status}", file=sys.stderr)
        return 3
    sid = reply["sid"]
    if reply.get("duplicate_of"):
        print(f"[submit] identical to submission {reply['duplicate_of']}")
    print(f"[submit] sid={sid} agent={agent_id()} queued"
          + (f" ({reply['position']} gradings ahead)" if reply.get("position") else ""), flush=True)
    deadline = time.time() + float(task.get("submit_wait_sec", 1800))
    record = reply
    while record.get("status") in ("queued", "running") and time.time() < deadline:
        time.sleep(2)
        try:
            status, record = request(task, "GET", f"/result/{sid}")
        except ConnectionError:
            continue
    if record.get("status") in ("queued", "running"):
        print(f"[submit] no result yet for sid={sid}; check later with `scores`", file=sys.stderr)
        return 4
    print(f"[submit] {describe(record, task)}")
    if record.get("detail"):
        print(f"[submit] detail: {record['detail']}")
    try:
        _, listing = request(task, "GET", "/submissions")
        top = best(listing.get("submissions", []), task)
        if top is not None:
            print(f"[submit] best so far: {task['metric_name']}={top.get('metric')} "
                  f"score={float(top['score']):.4f} by {top['agent_id']} (sid {top['sid']})")
    except ConnectionError:
        pass
    show_resources(task)
    return 0


def cmd_scores(task: dict, args) -> int:
    _, listing = request(task, "GET", "/submissions")
    records = listing.get("submissions", [])
    if not args.all:
        records = candidates(records)
    records.sort(key=rank(task))
    print(f"{'sid':>5} {'agent':<10} {'status':<8} {task['metric_name']:>10} {'score':>10}  submitted")
    for r in records:
        status = r.get("status") if r.get("status") != "done" else ("valid" if r.get("valid") else "invalid")
        print(f"{r['sid']:>5} {r['agent_id']:<10} {status:<8} {str(r.get('metric') or '-'):>10} "
              f"{float(r.get('score') or 0):>10.4f}  {r.get('submitted_ts', '')}"
              + (f"  [hidden grade: {r['final_status']}]" if r.get("final_status") else "")
              + (f"  [{' '.join(str(v) for v in r['options'].values())}]" if r.get("options") else "")
              + ("  (feedback only)" if r.get("eligible") is False else "")
              + (f"  {r['note']}" if r.get("note") else ""))
    if not records:
        print("(no graded submissions yet)" if not args.all else "(no submissions yet)")
    return 0


def cmd_fetch(task: dict, args) -> int:
    sid = args.sid
    if sid == "best":
        _, listing = request(task, "GET", "/submissions")
        top = best(listing.get("submissions", []), task)
        if top is None:
            print("no valid submission yet", file=sys.stderr)
            return 1
        sid = top["sid"]
    status, reply = request(task, "GET", f"/artifact/{sid}")
    if status != 200:
        print(f"no submission {sid}", file=sys.stderr)
        return 1
    dest = Path(args.dest)
    for name, data in reply["files"].items():
        path = dest / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(data))
        print(path)
    return 0


def main(argv: list[str]) -> int:
    command = Path(argv[0]).name
    parser = argparse.ArgumentParser(prog=command, description=__doc__.splitlines()[0])
    if command == "submit":
        parser.add_argument("path", nargs="?", help="directory (default: your working copy) or file")
        parser.add_argument("--note", help="a short note stored with the submission")
        handler = cmd_submit
    elif command == "workcopy":
        handler = cmd_workcopy
    elif command == "scores":
        parser.add_argument("--all", action="store_true", help="also queued, invalid and failed ones")
        handler = cmd_scores
    elif command == "fetch-submission":
        parser.add_argument("sid", help="a submission id from `scores`, or best")
        parser.add_argument("dest", help="directory to write the submission's files into")
        handler = cmd_fetch
    else:
        print(__doc__)
        return 2
    args = parser.parse_args(argv[1:])
    try:
        return handler(config(), args)
    except ConnectionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main([os.environ.get("SCORECTL_COMMAND", sys.argv[0]), *sys.argv[1:]]))
