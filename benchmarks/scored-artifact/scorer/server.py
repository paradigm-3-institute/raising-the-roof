#!/usr/bin/env python3
"""The trusted scorer of a scored-artifact task (see scored_artifact.py).

It runs in its own container next to the agents. Agents send artifacts with the
`submit` command; the scorer queues them fairly across agents (round robin, a few
waiting per agent at most), grades each with the benchmark's scoring module from a
pristine read-only copy of the benchmark, and appends every result to
/scorer/state/submissions.jsonl, a host directory (<trial>/scorer/) that no agent can
reach. The log has one record per graded submission: when it was submitted and
graded, the agent, the raw metric, a higher-is-better score, and whether it was
valid. Each submission's files and full result stay in submissions/<sid>/.

When the run ends, the verifier submits each agent's final working copy and calls
/finalize: the scorer stops taking submissions, waits for the queue, re-grades the
best valid submission (or, if that re-grade fails, the next best), and writes the
final grade to final.json.

    server.py --config /scorer/config.json                # serve
    server.py --config /scorer/config.json grade DIR...   # grade artifact directories once

The benchmark's module (config "module") defines

    score(submission: Path, sandbox: Sandbox, config: dict) -> dict

returning {"valid": bool, "metric": number or None, "score": number or None,
"detail": str, "info": {...}}. It runs in a child process; every step that runs
agent code must go through the sandbox. "detail" is shown to the agent; "info" is
kept for analysis only.

Two-stage grading (config "final_stage"; for benchmarks graded on data the agents must
not see, such as held-out instances). The module also defines

    final_score(submission: Path, sandbox: Sandbox, config: dict) -> dict

`score` is the agent's feedback: `submit` prints it, `scores` and `fetch-submission
best` use it. Every submission graded so then waits for `final_score`, the grade the
run is judged on, in a separate lane (one worker, agents in turn), which agents never
see. Its record enters the log once both are done: valid, metric, score and detail
are the final grade, feedback_valid, feedback_metric, feedback_score and
feedback_detail what the agent saw, final_ts when the final grade was done. The final
grade picks and re-grades the best submission by its final grade. With
"max_unfinished_per_agent", an agent cannot submit while that many of its submissions
wait for either stage, so the hidden lane cannot fall behind without bound.

Tree artifacts (config artifact.tree): a submission is every
file and directory under the artifact's roots (paths relative to the submission, such as
app/output), sent as one gzipped tar ("archive" in place of "files"); only regular files
and directories, with their permissions. Hidden grades in one stage (config final_rule
"hidden"): `score` returns the agent's feedback and also "hidden", the grade the run is
judged on, which goes into the record's "hidden" field and is never shown to agents; the
final grade is the recorded hidden grade of the best submission by feedback score (ties:
the earliest graded), without a re-grade.

Options of their own, for benchmarks that take them (each off without its config
key): "submission_options" names the options a submission may carry ({name: {"type": "int" or
"str", "min", "max", "choices", "max_chars"}}); a submission's options are in its record and reach
the module as its config's "options". A module result with "eligible": false is a grade that never
becomes the final grade (feedback only, such as an estimate on part of the data); `scores` and
`fetch-submission best` skip it too. "rank_by": "metric" picks the final grade's candidates by the
raw metric ("metric_better": "lower" or "higher"), the earliest graded among equal ones, instead of
by score. "floor_score" is the score of an invalid grade and of a run without a valid submission
(default 0). With "max_unfinished_per_agent" and one stage, an agent cannot submit while that many
of its submissions are waiting or being graded (one at a time: a second one is refused, not queued).
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import hmac
import importlib.util
import io
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbox import Sandbox, kill_user  # noqa: E402

AGENT_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
FILE_NAME = re.compile(r"^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$")
PUBLIC_FIELDS = ("sid", "agent_id", "status", "valid", "metric", "metric_name", "score", "detail",
                 "submitted_ts", "ts", "sha256", "final", "note", "position", "final_status", "options", "eligible")
# Two-stage grading: what an agent sees of a submission whose final grade is hidden.
FEEDBACK_FIELDS = ("valid", "metric", "score", "detail")
STAGES = ("score", "final_score")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("benchmark_scorer", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def artifact_digest(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode() + b"\0" + hashlib.sha256(files[name]).digest())
    return digest.hexdigest()


def grade_child(directory: str, worker: int, result_path: str, stage: str = "score") -> int:
    """`server.py grade-child`: grade one artifact directory (configuration on stdin)
    with the module's `score` (or, stage "final_score", its final grade) and write
    ["ok", result] or ["error", traceback] to result_path. A submission's options
    (stdin's "submission") reach the module as its config's "options"."""
    try:
        if stage not in STAGES:
            raise ValueError(f"unknown grading stage {stage!r}")
        config = json.loads(sys.stdin.read())
        module = load_module(Path(config["module"]))
        sandbox = Sandbox(int(config["sandbox_uid"]) + worker,
                          Path(config.get("sandbox_root", "/tmp/sandbox")) / f"w{worker}",
                          memory_mb=int(config.get("sandbox_memory_mb", 4096)))
        benchmark = config.get("benchmark", {})
        options = (config.get("submission") or {}).get("options")
        if options:
            benchmark = {**benchmark, "options": options}
        message = ["ok", getattr(module, stage)(Path(directory), sandbox, benchmark)]
    except BaseException:  # noqa: BLE001 - reported as a scorer error
        message = ["error", traceback.format_exc()[-4000:]]
    Path(result_path).write_text(json.dumps(message))
    return 0


def grade(config: dict, directory: Path, worker: int = 0, stage: str = "score", options: dict | None = None) -> dict:
    """Grade one artifact directory in a fresh Python process (never a fork of this
    threaded server) with the grading time limit. OPTIONS: the submission's (config
    "submission_options")."""
    limit = float(config.get("grade_timeout_sec", 900))
    if stage == "final_score":
        limit = float(config.get("final_grade_timeout_sec", limit))
    floor = float(config.get("floor_score", 0.0))
    start = time.monotonic()
    message = None
    with tempfile.TemporaryDirectory(prefix="grade-") as scratch:
        result_path = Path(scratch) / "result.json"
        child = subprocess.Popen([sys.executable, "-I", str(Path(__file__).resolve()), "grade-child",
                                  str(directory), str(worker), str(result_path),
                                  *([stage] if stage != "score" else [])],
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                 start_new_session=True)
        payload = {**config, "submission": {"options": options}} if options else config
        try:
            _, stderr = child.communicate(json.dumps(payload).encode(), timeout=limit)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except OSError:
                pass
            _, stderr = child.communicate()
        if os.geteuid() == 0:
            kill_user(int(config["sandbox_uid"]) + worker)
        try:
            message = json.loads(result_path.read_text())
        except (OSError, ValueError):
            message = None
    seconds = round(time.monotonic() - start, 3)
    if message is None:
        if time.monotonic() - start >= limit:
            return {"status": "done", "valid": False, "metric": None, "score": floor, "seconds": seconds,
                    "detail": f"grading took longer than the {limit:.0f} s limit"}
        return {"status": "error", "valid": False, "metric": None, "score": None, "seconds": seconds,
                "detail": "the scorer crashed while grading (not counted against you)",
                "info": {"stderr": stderr.decode("utf-8", "replace")[-2000:]}}
    kind, payload = message
    if kind == "error":
        return {"status": "error", "valid": False, "metric": None, "score": None, "seconds": seconds,
                "detail": "the scorer failed while grading (not counted against you)",
                "info": {"traceback": payload}}
    result = dict(payload)
    valid = bool(result.get("valid"))
    return {"status": "done", "valid": valid,
            "metric": result.get("metric") if valid else None,
            "score": float(result["score"]) if valid and result.get("score") is not None else floor,
            "detail": str(result.get("detail") or ""), "info": result.get("info") or {},
            "seconds": seconds,
            # A grade that never becomes the final grade (feedback only).
            **({"eligible": False} if result.get("eligible") is False else {}),
            # Hidden grades in one stage (final_rule "hidden"): never in an agent's view (PUBLIC_FIELDS).
            **({"hidden": result["hidden"]} if isinstance(result.get("hidden"), dict) else {})}


# Tree artifacts: names of the files and directories under the artifact's roots.
TREE_NAME = re.compile(r"^[^\x00-\x1f/\\][^\x00-\x1f\\]*$")


def tree_digest(entries: list[tuple[str, str, int, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, kind, mode, data in sorted(entries):
        digest.update(f"{kind}\0{name}\0{mode & 0o777:o}\0".encode() + hashlib.sha256(data).digest())
    return digest.hexdigest()


def read_tree(archive: bytes, roots: list[str], max_bytes: int, max_files: int) -> list[tuple[str, str, int, bytes]]:
    """(name, "file" or "dir", mode, content) of a tree artifact's gzipped tar; ValueError for
    anything that is not a regular file or a directory under one of the roots."""
    entries, total = [], 0
    try:
        tar = tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz")
    except (tarfile.TarError, OSError) as exc:
        raise ValueError(f"the submission is not a gzipped tar archive ({exc})") from None
    with tar:
        for member in tar:
            name = member.name.rstrip("/")
            parts = name.split("/")
            if (not TREE_NAME.match(name) or any(part in ("", ".", "..") for part in parts)
                    or not any(name == root or name.startswith(root + "/") for root in roots)):
                raise ValueError(f"bad path in the submission: {member.name!r}")
            if member.isdir():
                entries.append((name, "dir", member.mode, b""))
            elif member.isreg():
                total += member.size
                if total > max_bytes:
                    raise ValueError(f"the submission has more than {max_bytes} bytes")
                entries.append((name, "file", member.mode, tar.extractfile(member).read()))
            else:
                kind = "symbolic link" if member.issym() else "hard link" if member.islnk() else "special file"
                raise ValueError(f"{name} is a {kind}; a submission holds only regular files and directories")
            if len(entries) > max_files:
                raise ValueError(f"the submission has more than {max_files} files and directories")
    names = [name for name, _, _, _ in entries]
    if len(names) != len(set(names)):
        raise ValueError("the submission names a path twice")
    return entries


class Scorer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.state = Path(config.get("state_dir", "/scorer/state"))
        self.host_uid = config.get("host_uid")
        self.host_gid = config.get("host_gid")
        self.artifact = config["artifact"]
        self.metric_name = config.get("metric_name", "metric")
        self.lock = threading.Condition()
        self.records: dict[str, dict] = {}
        self.queues: dict[str, collections.deque] = {}
        self.rotation: list[str] = []
        self.turn = 0
        self.running = 0
        self.next_sid = 1
        self.closed = False
        self.final: dict | None = None
        # Two-stage grading: submissions whose feedback is done wait here for their final
        # grade, agents in turn.
        self.two_stage = bool(config.get("final_stage"))
        self.final_queues: dict[str, collections.deque] = {}
        self.final_rotation: list[str] = []
        self.final_turn = 0
        self.final_running = 0
        self.state.mkdir(parents=True, exist_ok=True)
        (self.state / "submissions").mkdir(exist_ok=True)
        self._own(self.state)
        self._own(self.state / "submissions")
        self.state.chmod(0o700)
        self._restore()

    # -- files -------------------------------------------------------------

    def _own(self, path: Path) -> None:
        """Host files belong to the user who launched the run, so they can remove them."""
        if self.host_uid is not None and os.geteuid() == 0:
            os.chown(path, int(self.host_uid), int(self.host_gid if self.host_gid is not None else self.host_uid))

    def _write(self, path: Path, text: str) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(text)
        self._own(temporary)
        temporary.replace(path)

    def _append_log(self, record: dict) -> None:
        log = self.state / "submissions.jsonl"
        with log.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        self._own(log)

    def _restore(self) -> None:
        """Pick up the log after a scorer restart, so no submission is lost: the graded ones from the log, and
        the ones still waiting or being graded at the restart (otherwise lost, and the next submission would fail
        on the existing directory of the first lost sid) queued again in sid order, from
        the submission.json each submission's directory gets when it is taken."""
        log = self.state / "submissions.jsonl"
        if log.is_file():
            for line in log.read_text(errors="replace").splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict) and record.get("sid"):
                    self.records[str(record["sid"])] = record
                    self.next_sid = max(self.next_sid, int(record["sid"]) + 1)
        final = self.state / "final.json"
        if final.is_file():
            self.final = json.loads(final.read_text())
            self.closed = True
        waiting = []
        for directory in (self.state / "submissions").iterdir():
            if not directory.name.isdigit():
                continue
            self.next_sid = max(self.next_sid, int(directory.name) + 1)
            taken = directory / "submission.json"
            if directory.name in self.records or self.closed or not taken.is_file():
                continue
            try:
                record = json.loads(taken.read_text())
            except ValueError:
                continue
            if isinstance(record, dict) and str(record.get("sid")) == directory.name and record.get("agent_id"):
                waiting.append(record)
        for record in sorted(waiting, key=lambda r: int(r["sid"])):
            record = {key: value for key, value in record.items() if key != "position"}
            record["status"] = "queued"
            self.records[record["sid"]] = record
            if record["agent_id"] not in self.queues:
                self.queues[record["agent_id"]] = collections.deque()
                self.rotation.append(record["agent_id"])
            self.queues[record["agent_id"]].append(record["sid"])

    # -- submissions -------------------------------------------------------

    def check_files(self, files: dict) -> dict[str, bytes]:
        if not isinstance(files, dict) or not files:
            raise ValueError("no files in the submission")
        wanted = set(self.artifact["files"])
        names = set(files)
        if names != wanted:
            missing, extra = sorted(wanted - names), sorted(names - wanted)
            raise ValueError("a submission is exactly " + ", ".join(sorted(wanted))
                             + (f"; missing {', '.join(missing)}" if missing else "")
                             + (f"; not part of it: {', '.join(extra)}" if extra else ""))
        decoded, total = {}, 0
        for name, data in files.items():
            if not FILE_NAME.match(name) or ".." in name.split("/"):
                raise ValueError(f"bad file name {name!r}")
            decoded[name] = base64.b64decode(data, validate=True)
            total += len(decoded[name])
        if total > int(self.artifact.get("max_bytes", 2**20)):
            raise ValueError(f"the submission has {total} bytes; the limit is {self.artifact['max_bytes']}")
        return decoded

    def check_archive(self, archive) -> list[tuple[str, str, int, bytes]]:
        tree = self.artifact.get("tree")
        if not isinstance(tree, dict):
            raise ValueError("this benchmark's submissions are files, not an archive")
        if not isinstance(archive, str):
            raise ValueError("no archive in the submission")
        return read_tree(base64.b64decode(archive, validate=True), list(tree["roots"]),
                         int(self.artifact.get("max_bytes", 2**20)), int(tree.get("max_files", 100000)))

    def check_options(self, options) -> dict:
        """A submission's options, checked against config "submission_options" (none without it)."""
        if options is None or options == {}:
            return {}
        known = self.config.get("submission_options") or {}
        if not known:
            raise ValueError("this benchmark's submissions take no options")
        if not isinstance(options, dict):
            raise ValueError("options must be a JSON object")
        checked = {}
        for name, value in options.items():
            spec = known.get(name)
            if not isinstance(spec, dict):
                raise ValueError(f"unknown option {name!r}; known: {', '.join(sorted(known))}")
            if spec.get("type") == "int":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"option {name} must be a whole number")
                if ("min" in spec and value < spec["min"]) or ("max" in spec and value > spec["max"]):
                    raise ValueError(f"option {name} must be between {spec.get('min')} and {spec.get('max')}")
            else:
                if not isinstance(value, str) or len(value) > int(spec.get("max_chars", 200)):
                    raise ValueError(f"option {name} must be text of at most {spec.get('max_chars', 200)} characters")
                if "choices" in spec and value not in spec["choices"]:
                    raise ValueError(f"option {name} must be one of {', '.join(spec['choices'])}")
            checked[name] = value
        return checked

    def submit(self, agent_id: str, files: dict, note: str = "", final: bool = False, archive=None,
               options=None) -> dict:
        if not AGENT_ID.match(agent_id or ""):
            raise ValueError("agent_id must be 1-64 letters, digits, or _.:-")
        options = self.check_options(options)
        if isinstance(self.artifact.get("tree"), dict):
            entries = self.check_archive(archive)
            decoded = {name: data for name, kind, _, data in entries if kind == "file"}
            sha = tree_digest(entries)
        else:
            entries = None
            decoded = self.check_files(files)
            sha = artifact_digest(decoded)
        with self.lock:
            if self.closed:
                raise PermissionError("the run has finished; the scorer takes no more submissions")
            if final:
                # A final working copy identical to an earlier submission is already graded.
                for record in self.records.values():
                    if record.get("sha256") == sha:
                        return {**self.public(record), "duplicate_of": record["sid"]}
            unfinished_limit = self.config.get("max_unfinished_per_agent")
            if not self.two_stage and unfinished_limit and not final:
                unfinished = sum(1 for r in self.records.values()
                                 if r.get("agent_id") == agent_id and r.get("status") in ("queued", "running"))
                if unfinished >= int(unfinished_limit):
                    raise OverflowError(f"agent {agent_id} already has {unfinished} submission(s) being graded; "
                                        "submissions do not queue: wait for its result, then submit again")
            pending = self.queues.get(agent_id, ())
            limit = int(self.config.get("max_pending_per_agent", 4))
            if len(pending) >= limit and not final:
                raise OverflowError(f"agent {agent_id} already has {len(pending)} submissions waiting; "
                                    "wait for their results before submitting more")
            if self.two_stage and unfinished_limit and not final:
                unfinished = sum(1 for r in self.records.values() if r.get("agent_id") == agent_id and (
                    r.get("status") in ("queued", "running") or r.get("final_status") in ("queued", "running")))
                if unfinished >= int(unfinished_limit):
                    raise OverflowError(
                        f"agent {agent_id} already has {unfinished} submissions still being graded (their "
                        "hidden grade takes longer than the feedback); wait for one to finish "
                        "(`scores --all` shows each one's hidden grade as queued, running or done) "
                        "before submitting more")
            sid = str(self.next_sid)
            self.next_sid += 1
            directory = self.state / "submissions" / sid
            directory.mkdir(parents=True)
            self._own(directory)
            if entries is None:
                for name, data in decoded.items():
                    path = directory / "files" / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
            else:
                # The tree as sent: its directories (empty ones too) and files with their permissions,
                # the directories' set once their files are written.
                (directory / "files").mkdir()
                for name, kind, mode, data in sorted(entries):
                    path = directory / "files" / name
                    if kind == "dir":
                        path.mkdir(parents=True, exist_ok=True)
                        continue
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                    path.chmod(mode & 0o777)
                for name, kind, mode, _ in sorted(entries, reverse=True):
                    if kind == "dir":
                        (directory / "files" / name).chmod(mode & 0o777)
            for path in [directory / "files", *(directory / "files").rglob("*")]:
                self._own(path)
            record = {"sid": sid, "agent_id": agent_id, "status": "queued", "submitted_ts": now_iso(),
                      "sha256": sha, "bytes": sum(map(len, decoded.values())), "final": bool(final),
                      "note": str(note)[:int(self.config.get("max_note_chars", 500))],
                      "metric_name": self.metric_name, **({"options": options} if options else {})}
            self._write(directory / "submission.json", json.dumps(record, indent=2, sort_keys=True) + "\n")
            self.records[sid] = record
            if agent_id not in self.queues:
                self.queues[agent_id] = collections.deque()
                self.rotation.append(agent_id)
            self.queues[agent_id].append(sid)
            self.lock.notify_all()
            return self.public(record)

    def position(self, sid: str) -> int | None:
        """Gradings ahead of a queued submission, counting round-robin turns."""
        record = self.records.get(sid)
        if not record or record.get("status") != "queued":
            return None
        own = list(self.queues[record["agent_id"]]).index(sid)
        ahead = 0
        for agent, queue in self.queues.items():
            ahead += min(len(queue), own + (1 if agent != record["agent_id"] else 0))
        return ahead + self.running

    def public(self, record: dict) -> dict:
        shown = {key: record[key] for key in PUBLIC_FIELDS if key in record}
        if "feedback_score" in record:
            # Two-stage grading: the agent sees its feedback, never the final grade.
            for key in FEEDBACK_FIELDS:
                shown[key] = record.get(f"feedback_{key}")
        if record.get("status") == "queued":
            shown["position"] = self.position(record["sid"])
        return shown

    @staticmethod
    def _take(queues: dict[str, collections.deque], rotation: list[str], turn: int) -> tuple[str | None, int]:
        for offset in range(len(rotation)):
            agent = rotation[(turn + offset) % len(rotation)]
            if queues[agent]:
                return queues[agent].popleft(), (turn + offset + 1) % len(rotation)
        return None, turn

    def _next(self) -> str | None:
        sid, self.turn = self._take(self.queues, self.rotation, self.turn)
        return sid

    def _next_final(self) -> str | None:
        sid, self.final_turn = self._take(self.final_queues, self.final_rotation, self.final_turn)
        return sid

    def _finish(self, record: dict) -> None:
        self._write(self.state / "submissions" / record["sid"] / "result.json",
                    json.dumps(record, indent=2, sort_keys=True) + "\n")
        self._append_log(record)

    def worker(self, index: int) -> None:
        while True:
            with self.lock:
                sid = self._next()
                while sid is None:
                    self.lock.wait()
                    sid = self._next()
                record = self.records[sid]
                record["status"] = "running"
                record["started_ts"] = now_iso()
                self.running += 1
            result = grade(self.config, self.state / "submissions" / sid / "files", index,
                           options=record.get("options"))
            with self.lock:
                record.update(result)
                record["ts"] = now_iso()
                record["metric_name"] = self.metric_name
                self.running -= 1
                if self.two_stage and result["status"] == "done":
                    # The agent has its feedback; the log gets the record with its final grade.
                    record["final_status"] = "queued"
                    agent = record["agent_id"]
                    if agent not in self.final_queues:
                        self.final_queues[agent] = collections.deque()
                        self.final_rotation.append(agent)
                    self.final_queues[agent].append(sid)
                else:
                    self._finish(record)
                self.lock.notify_all()

    def final_worker(self, index: int) -> None:
        """Two-stage grading: the final grade of each submission whose feedback is done."""
        while True:
            with self.lock:
                sid = self._next_final()
                while sid is None:
                    self.lock.wait()
                    sid = self._next_final()
                record = self.records[sid]
                record["final_status"] = "running"
                record["final_started_ts"] = now_iso()
                self.final_running += 1
            result = grade(self.config, self.state / "submissions" / sid / "files", index, "final_score",
                           options=record.get("options"))
            with self.lock:
                for key in (*FEEDBACK_FIELDS, "info", "seconds"):
                    record[f"feedback_{key}"] = record.get(key)
                record.update(result)
                record["final_status"] = "done"
                record["final_ts"] = now_iso()
                self.final_running -= 1
                self._finish(record)
                self.lock.notify_all()

    def _busy(self) -> bool:
        return bool(any(self.queues.values()) or self.running
                    or any(self.final_queues.values()) or self.final_running)

    # -- the end of the run ------------------------------------------------

    def finalize(self) -> dict:
        with self.lock:
            if self.final is not None:
                return self.final
            self.closed = True
            while self._busy():
                self.lock.wait()
            graded = [r for r in self.records.values() if r.get("status") == "done" and r.get("valid")
                      and (not self.two_stage or r.get("final_status") == "done") and r.get("eligible") is not False]
        if self.config.get("final_rule") == "hidden":
            return self._finalize_hidden(graded)
        # Best first; among equal scores the earliest graded, or with "final_prefer_latest"
        # the latest submitted (a benchmark whose own rule keeps the last of its best).
        if self.config.get("final_prefer_latest"):
            graded.sort(key=lambda r: (r.get("submitted_ts") or "", int(r.get("sid") or 0)), reverse=True)
        else:
            graded.sort(key=lambda r: r.get("ts") or "")
        if self.config.get("rank_by") == "metric":
            # By the raw metric (a benchmark whose rule compares it, not a rounded score); none last.
            sign = 1 if self.config.get("metric_better", "lower") == "lower" else -1
            graded.sort(key=lambda r: (r.get("metric") is None, sign * float(r.get("metric") or 0)))
        else:
            graded.sort(key=lambda r: -float(r.get("score") or 0))
        stage = "final_score" if self.two_stage else "score"
        tried, seen, final = [], set(), None
        for record in graded:
            if record["sha256"] in seen:
                continue
            seen.add(record["sha256"])
            if len(tried) >= int(self.config.get("final_candidates", 5)):
                break
            regrade = grade(self.config, self.state / "submissions" / record["sid"] / "files", 0, stage,
                            options=record.get("options"))
            entry = {"sid": record["sid"], "agent_id": record["agent_id"], "original_metric": record.get("metric"),
                     "original_score": record.get("score"), "regrade": regrade, "ts": now_iso()}
            tried.append(entry)
            with (self.state / "regrades.jsonl").open("a") as stream:
                stream.write(json.dumps(entry, sort_keys=True) + "\n")
            self._own(self.state / "regrades.jsonl")
            if regrade["status"] == "done" and regrade["valid"]:
                final = entry
                break
        result = {"ts": now_iso(), "metric_name": self.metric_name,
                  "submissions": len(self.records), "valid_submissions": len(graded),
                  "candidates_tried": len(tried)}
        if final is None:
            result.update(valid=False, score=float(self.config.get("floor_score", 0.0)), metric=None, sid=None,
                          agent_id=None,
                          detail="no valid submission" if not graded else "no best submission passed its re-grade")
        else:
            result.update(valid=True, score=final["regrade"]["score"], metric=final["regrade"]["metric"],
                          sid=final["sid"], agent_id=final["agent_id"],
                          original_score=final["original_score"], original_metric=final["original_metric"],
                          detail=final["regrade"].get("detail", ""))
        with self.lock:
            self.final = result
            self._write(self.state / "final.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result

    @staticmethod
    def select_hidden(graded: list[dict]) -> dict | None:
        """final_rule "hidden": the submission with the strictly highest feedback score, the
        earliest graded among equal ones (in the order their gradings finished)."""
        best = None
        for record in sorted(graded, key=lambda r: (r.get("ts") or "", int(r.get("sid") or 0))):
            if best is None or float(record.get("score") or 0) > float(best.get("score") or 0):
                best = record
        return best

    def _finalize_hidden(self, graded: list[dict]) -> dict:
        """The final grade is the selected submission's hidden grade as recorded, not a re-grade."""
        best = self.select_hidden(graded)
        result = {"ts": now_iso(), "metric_name": self.metric_name, "rule": "hidden",
                  "submissions": len(self.records), "valid_submissions": len(graded), "candidates_tried": 0}
        hidden = (best or {}).get("hidden") or {}
        if best is None:
            result.update(valid=False, score=0.0, metric=None, sid=None, agent_id=None,
                          detail="no submission with a feedback score")
        else:
            valid = bool(hidden.get("valid"))
            result.update(valid=valid, score=float(hidden["score"]) if valid else 0.0,
                          metric=hidden.get("metric"), sid=best["sid"], agent_id=best["agent_id"],
                          original_score=best.get("score"), original_metric=best.get("metric"),
                          hidden={key: value for key, value in hidden.items() if key not in ("stdout",)},
                          detail="the hidden grade of the submission with the best feedback score"
                          + ("" if valid else " (it has none: its hidden grading failed)"))
        with self.lock:
            self.final = result
            self._write(self.state / "final.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
        return result


def handler(scorer: Scorer, token: str):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # quiet
            pass

        def reply(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            limit = int(scorer.artifact.get("max_bytes", 2**20)) * 2 + 65536
            if length > limit:
                raise ValueError("request too large")
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError("expected a JSON object")
            return data

        def authorized(self, data: dict) -> bool:
            return bool(token) and hmac.compare_digest(str(data.get("token") or ""), token)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?")[0].rstrip("/")
            with scorer.lock:
                if path == "/health":
                    return self.reply(200, {"ok": True, "open": not scorer.closed})
                if path == "/info":
                    return self.reply(200, scorer.config.get("public", {}))
                if path == "/submissions":
                    return self.reply(200, {"submissions": [scorer.public(r) for r in scorer.records.values()]})
                match = re.fullmatch(r"/result/(\d+)", path)
                if match and match.group(1) in scorer.records:
                    return self.reply(200, scorer.public(scorer.records[match.group(1)]))
                match = re.fullmatch(r"/artifact/(\d+)", path)
                if match and match.group(1) in scorer.records:
                    files = scorer.state / "submissions" / match.group(1) / "files"
                    paths = [p for p in sorted(files.rglob("*")) if p.is_file()]
                    return self.reply(200, {"sid": match.group(1), "files": {
                        str(p.relative_to(files)): base64.b64encode(p.read_bytes()).decode() for p in paths},
                        "modes": {str(p.relative_to(files)): p.stat().st_mode & 0o777 for p in paths}})
            self.reply(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.rstrip("/")
            try:
                data = self.body()
                if path == "/submit":
                    final = bool(data.get("final"))
                    if final and not self.authorized(data):
                        return self.reply(403, {"error": "only the verifier submits final copies"})
                    return self.reply(200, scorer.submit(str(data.get("agent_id") or ""), data.get("files"),
                                                         str(data.get("note") or ""), final,
                                                         archive=data.get("archive"), options=data.get("options")))
                if path == "/finalize":
                    if not self.authorized(data):
                        return self.reply(403, {"error": "only the verifier can end the run"})
                    return self.reply(200, scorer.finalize())
                return self.reply(404, {"error": "not found"})
            except PermissionError as exc:
                return self.reply(409, {"error": str(exc)})
            except OverflowError as exc:
                return self.reply(429, {"error": str(exc)})
            except (ValueError, TypeError) as exc:
                return self.reply(400, {"error": str(exc)})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("/scorer/config.json"))
    commands = parser.add_subparsers(dest="command")
    once = commands.add_parser("grade", help="grade artifact directories once and print JSON lines")
    once.add_argument("directories", nargs="+", type=Path)
    once.add_argument("--final-only", action="store_true",
                      help="two-stage grading: the final grade only, without the agents' feedback")
    once.add_argument("--options", type=json.loads, default=None, metavar="JSON",
                      help="the submissions' options (config submission_options), as a JSON object")
    child = commands.add_parser("grade-child", help=argparse.SUPPRESS)
    child.add_argument("directory")
    child.add_argument("worker", type=int)
    child.add_argument("result")
    child.add_argument("stage", nargs="?", default="score")
    args = parser.parse_args()
    if args.command == "grade-child":
        return grade_child(args.directory, args.worker, args.result, args.stage)
    config = json.loads(args.config.read_text())
    if args.command == "grade":
        for directory in args.directories:
            if not config.get("final_stage"):
                result = grade(config, directory, options=args.options)
            else:
                # As a run grades: the feedback, then the final grade (the record's grade).
                feedback = {} if args.final_only else grade(config, directory)
                result = grade(config, directory, 0, "final_score")
                result.update({f"feedback_{key}": feedback.get(key) for key in (*FEEDBACK_FIELDS, "info")
                               if feedback})
            print(json.dumps({"directory": str(directory), **result}), flush=True)
        return 0
    scorer = Scorer(config)
    workers = int(config.get("workers", 1))
    for index in range(workers):
        threading.Thread(target=scorer.worker, args=(index,), daemon=True).start()
    if scorer.two_stage:
        # Its own sandbox user (workers + index), so the lanes never kill each other's programs.
        for index in range(int(config.get("final_workers", 1))):
            threading.Thread(target=scorer.final_worker, args=(workers + index,), daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", int(config.get("port", 18090))),
                                 handler(scorer, str(config.get("token") or "")))
    server.daemon_threads = True
    print(f"scorer for {config.get('benchmark_name')} listening on port {server.server_port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
