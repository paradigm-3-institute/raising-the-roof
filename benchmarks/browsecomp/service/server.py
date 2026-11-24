#!/usr/bin/env python3
"""The answer log of a BrowseComp trial (browsecomp_task.py), in its own container next to the agents'.

Agents submit responses here (`submit-answer`) and list them (`answers`); the service appends every
submission to STATE/submissions.jsonl on the host, out of the agents' reach. It holds no answer key
and grades nothing: responses are graded on the host after the run. The team's answer is the last
submission. The verifier closes the log with the finalize token: STATE/answer.json then records the
team's answer, and later submissions are refused.

    server.py --config config.json            # serve
Endpoints: GET /health, GET /answers, POST /submit {"agent_id", "response"},
POST /finalize (header X-Finalize-Token).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# The fields of upstream's response format (simple-evals QUERY_TEMPLATE), read for display only.
EXACT_ANSWER = re.compile(r"^\W*exact\s+answer\W*?:[\s*_]*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
CONFIDENCE = re.compile(r"^\W*confidence\W*?:[\s*_]*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
AGENT_ID = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def fields(response: str) -> dict:
    """The response's Exact Answer and Confidence lines (the last of each), or None."""
    answers = EXACT_ANSWER.findall(response)
    confidences = CONFIDENCE.findall(response)
    return {"exact_answer": answers[-1][:500] if answers else None,
            "confidence": confidences[-1][:100] if confidences else None,
            "format_ok": bool(answers)}


class AnswerLog:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.state = Path(config["state_dir"])
        self.state.mkdir(parents=True, exist_ok=True)
        self.log = self.state / "submissions.jsonl"
        self.closed_file = self.state / "answer.json"
        self.lock = threading.Lock()
        self.records: list[dict] = []
        if self.log.is_file():
            # A service started again (a revived trial) goes on from its log.
            for line in self.log.read_text(errors="replace").splitlines():
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    self.records.append(record)
        self.closed = self.closed_file.is_file()
        self._own(self.state)

    def _own(self, path: Path) -> None:
        """Files on the host belong to the host user who started the run."""
        try:
            os.chown(path, int(self.config["host_uid"]), int(self.config["host_gid"]))
        except (OSError, KeyError, ValueError):
            pass

    def _append(self, record: dict) -> None:
        with self.log.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._own(self.log)

    def submit(self, agent_id: str, response: str) -> tuple[int, dict]:
        if not isinstance(agent_id, str) or not AGENT_ID.fullmatch(agent_id):
            return 400, {"error": "agent_id must be 1-64 letters, digits, '.', '_' or '-'"}
        if not isinstance(response, str) or not response.strip():
            return 400, {"error": "the response is empty"}
        data = response.encode("utf-8", errors="strict")
        if len(data) > int(self.config["max_response_bytes"]):
            return 413, {"error": f"the response is {len(data)} bytes; the limit is "
                                  f"{self.config['max_response_bytes']}"}
        with self.lock:
            if self.closed:
                return 409, {"error": "the run is over; no more submissions are accepted"}
            mine = sum(1 for record in self.records if record.get("agent_id") == agent_id)
            if mine >= int(self.config["max_submissions_per_agent"]):
                return 429, {"error": f"you have made {mine} submissions, the most allowed"}
            record = {"sid": len(self.records) + 1, "ts": now_iso(), "agent_id": agent_id,
                      "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), **fields(response),
                      "response": response}
            self._append(record)
            self.records.append(record)
        return 200, {"sid": record["sid"], "ts": record["ts"], "exact_answer": record["exact_answer"],
                     "confidence": record["confidence"], "format_ok": record["format_ok"],
                     "submissions": len(self.records)}

    def answers(self) -> dict:
        with self.lock:
            records = list(self.records)
        return {"rule": "the team's answer is the last submission", "closed": self.closed,
                "current": records[-1]["sid"] if records else None, "submissions": records}

    def finalize(self) -> dict:
        with self.lock:
            if not self.closed:
                last = self.records[-1] if self.records else None
                answer = {"ts": now_iso(), "rule": "last submission", "submissions": len(self.records),
                          "agents": sorted({r.get("agent_id") for r in self.records}),
                          "answer": None if last is None else {key: last.get(key) for key in (
                              "sid", "ts", "agent_id", "sha256", "bytes", "exact_answer", "confidence",
                              "format_ok")}}
                temporary = self.closed_file.with_name(self.closed_file.name + ".tmp")
                temporary.write_text(json.dumps(answer, indent=2, sort_keys=True) + "\n")
                temporary.replace(self.closed_file)
                self._own(self.closed_file)
                self.closed = True
            return json.loads(self.closed_file.read_text())


def handler(service: AnswerLog, token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "answer-log"

        def log_message(self, format, *args):  # noqa: A002 - quiet
            pass

        def _send(self, status: int, payload: dict) -> None:
            body = (json.dumps(payload) + "\n").encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict | None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return None
            if length > int(service.config["max_response_bytes"]) * 2 + 4096:
                return None
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, UnicodeDecodeError):
                return None
            return data if isinstance(data, dict) else None

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"ok": True})
            elif self.path == "/answers":
                self._send(200, service.answers())
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path == "/submit":
                body = self._body()
                if body is None:
                    self._send(400, {"error": "send a JSON object {agent_id, response}"})
                    return
                try:
                    status, payload = service.submit(body.get("agent_id"), body.get("response"))
                except UnicodeEncodeError:
                    status, payload = 400, {"error": "the response is not valid UTF-8"}
                self._send(status, payload)
            elif self.path == "/finalize":
                if self.headers.get("X-Finalize-Token") != token:
                    self._send(403, {"error": "forbidden"})
                    return
                self._send(200, service.finalize())
            else:
                self._send(404, {"error": "not found"})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    service = AnswerLog(config)
    server = ThreadingHTTPServer(("0.0.0.0", int(config["port"])), handler(service, config["finalize_token"]))
    server.daemon_threads = True
    print(f"answer log on port {config['port']}, {len(service.records)} submissions", file=sys.stderr, flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
