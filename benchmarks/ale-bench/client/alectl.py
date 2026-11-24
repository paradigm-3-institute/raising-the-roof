#!/usr/bin/env python3
"""Agent-side commands of an ale-bench task (Python standard library only).

    workcopy                        create your working copy (/work/$AGENT_ID) once; print its path
    submit [PATH] [--note TEXT]     public evaluation of your program (default: your copy's), printed
    scores [--all]                  every graded submission of the run, best public score first
    fetch-submission SID|best DEST  copy a graded submission's files into DEST

The judge runs in another container and keeps the log the run is graded on; these
commands only talk to it (the scored-artifact layer's HTTP API). Exit codes of submit:
0 graded (accepted or not), 2 missing files, 3 the judge refused or could not be
reached, 4 no result in time.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CONFIG = Path(os.environ.get("SCORED_ARTIFACT_CONFIG", "/opt/ale-bench/client/task.json"))


def config() -> dict:
    return json.loads(CONFIG.read_text())


def agent_id() -> str:
    return os.environ.get("AGENT_ID") or "single"


def request(task: dict, method: str, path: str, body: dict | None = None, timeout: float = 60):
    """(status, JSON reply) from the first judge address that answers."""
    data = json.dumps(body).encode() if body is not None else None
    last = None
    for base in task["scorer_urls"]:
        req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method,
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
    raise ConnectionError(f"cannot reach the judge ({last})")


def workcopy_path(task: dict) -> Path:
    return Path(task["work_root"]) / agent_id()


def cmd_workcopy(task: dict, args) -> int:
    path = workcopy_path(task)
    if not path.exists():
        staging = path.with_name(path.name + ".partial")
        shutil.rmtree(staging, ignore_errors=True)
        shutil.copytree(task["starter"], staging, symlinks=True)
        for item in [staging, *staging.rglob("*")]:
            if item.is_symlink():
                continue
            executable = item.is_dir() or item.parent.name == "bin"
            item.chmod(0o755 if executable else 0o644)
        if shutil.which("git"):
            git = ["git", "-C", str(staging), "-c", "user.name=starter", "-c", "user.email=starter@localhost"]
            (staging / ".gitignore").write_text("in/\nout/\n*.out\n/build/\n")
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
            raise FileNotFoundError(f"a submission is {len(names)} files ({names[0]} ...); pass the directory "
                                    "that holds them")
        return {names[0]: base64.b64encode(target.read_bytes()).decode()}
    files = {}
    for name in names:
        path = target / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")
        files[name] = base64.b64encode(path.read_bytes()).decode()
    return files


def better(a: float, b: float, task: dict) -> bool:
    return a > b if task["score_type"] == "maximize" else a < b


def ranking_key(record: dict, task: dict):
    if task.get("variant") == "outputs":
        return (-float(record.get("variant_score") or 0), int(record["sid"]))
    sign = -1 if task["score_type"] == "maximize" else 1
    return (sign * float(record["public_score"]), int(record["sid"]))


def graded(records: list[dict]) -> list[dict]:
    return [r for r in records if r.get("status") == "done" and r.get("valid") and r.get("public_score") is not None]


def best(records: list[dict], task: dict) -> dict | None:
    valid = sorted(graded(records), key=lambda r: ranking_key(r, task))
    return valid[0] if valid else None


def describe(record: dict, task: dict) -> str:
    if record.get("status") in ("queued", "running"):
        return f"sid={record['sid']} {record['status']}"
    if record.get("status") == "error":
        return f"sid={record['sid']} judge error: {record.get('detail')}"
    verdict = record.get("public_judge") or ("ACCEPTED" if record.get("valid") else "not accepted")
    text = f"sid={record['sid']} {verdict}"
    if record.get("valid"):
        text += f" public_score={record.get('public_score')}"
        if task.get("variant") == "outputs":
            text += f" score={float(record.get('variant_score') or 0):.6g}"
    return text


def cmd_submit(task: dict, args) -> int:
    target = Path(args.path) if args.path else workcopy_path(task)
    try:
        files = read_artifact(task, target)
    except FileNotFoundError as exc:
        hint = "" if args.path or target.exists() else " (run `workcopy` first)"
        print(f"[submit] ERROR: {exc}{hint}", file=sys.stderr)
        return 2
    try:
        status, reply = request(task, "POST", "/submit", {"agent_id": agent_id(), "files": files,
                                                          "note": args.note or ""})
    except ConnectionError as exc:
        print(f"[submit] ERROR: {exc}", file=sys.stderr)
        return 3
    if status != 200:
        print(f"[submit] ERROR: {reply.get('error') or status}", file=sys.stderr)
        return 3
    sid = reply["sid"]
    print(f"[submit] sid={sid} agent={agent_id()} queued"
          + (f" ({reply['position']} gradings ahead)" if reply.get("position") else ""), flush=True)
    deadline = time.time() + float(task.get("submit_wait_sec", 3600))
    record = reply
    while record.get("status") in ("queued", "running") and time.time() < deadline:
        time.sleep(2)
        try:
            _, record = request(task, "GET", f"/result/{sid}")
        except ConnectionError:
            continue
    if record.get("status") in ("queued", "running"):
        print(f"[submit] no result yet for sid={sid}; check later with `scores`", file=sys.stderr)
        return 4
    print(f"[submit] {describe(record, task)}")
    if record.get("detail"):
        print(record["detail"])
    try:
        _, listing = request(task, "GET", "/submissions")
        top = best(listing.get("submissions", []), task)
        if top is not None:
            who = "the team" if task.get("agents", 1) > 1 else "you"
            print(f"[submit] best of {who} so far: public_score={top.get('public_score')} by {top['agent_id']} "
                  f"(sid {top['sid']})")
    except ConnectionError:
        pass
    return 0


def cmd_scores(task: dict, args) -> int:
    _, listing = request(task, "GET", "/submissions")
    records = listing.get("submissions", [])
    shown = sorted(graded(records), key=lambda r: ranking_key(r, task))
    if args.all:
        rest = [r for r in records if r not in shown]
        shown += sorted(rest, key=lambda r: int(r["sid"]))
    print(f"{'sid':>5} {'agent':<10} {'result':<22} {'public_score':>14}  submitted")
    for r in shown:
        result = r.get("public_judge") or r.get("status")
        if r.get("status") in ("queued", "running", "error"):
            result = r["status"]
        print(f"{r['sid']:>5} {r['agent_id']:<10} {str(result):<22} {str(r.get('public_score') or '-'):>14}  "
              f"{r.get('submitted_ts', '')}" + (f"  {r['note']}" if r.get("note") else ""))
    if not shown:
        print("(no accepted submissions yet)" if not args.all else "(no submissions yet)")
    return 0


def cmd_fetch(task: dict, args) -> int:
    sid = args.sid
    if sid == "best":
        _, listing = request(task, "GET", "/submissions")
        top = best(listing.get("submissions", []), task)
        if top is None:
            print("no accepted submission yet", file=sys.stderr)
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
        parser.add_argument("path", nargs="?", help="your program (default: your working copy's) or a directory")
        parser.add_argument("--note", help="a short note stored with the submission")
        handler = cmd_submit
    elif command == "workcopy":
        handler = cmd_workcopy
    elif command == "scores":
        parser.add_argument("--all", action="store_true", help="also queued, rejected and failed ones")
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
    sys.exit(main([os.environ.get("ALECTL_COMMAND", sys.argv[0]), *sys.argv[1:]]))
