#!/usr/bin/env python3
"""A local mock of the OpenAI Responses API for a team of Codex agents on the message board
(tests/test_message_board.py): one process serves every agent, each from its own plan.

    board_mock_api.py --plans plans.json --log requests.jsonl --ready FILE

plans.json maps an agent id to its steps. A request belongs to the agent whose board path
(`/root/<id>`, from its multi-agent role message) it carries, and takes that agent's next step:
  {"type": "text", "text": "..."}                       final answer, ending the turn
  {"type": "shell", "command": "echo hi"}               one shell tool call
  {"type": "call", "namespace": "collaboration", "name": "post", "arguments": {...}}
                                                        one call of a namespaced tool
  {"type": "js", "code": "text(await tools.collaboration__post({...}))"}
                                                        one call of the code-mode exec tool
  "after": {"agent": "a2", "requests": 2}               on any step: answer only once that agent has
                                                        made this many requests
After its last step an agent's requests get the text "done". Every request is logged with its body.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Condition()
PLANS: dict = {}
STEPS: dict = {}
REQUESTS: dict = {}
LOG = None
ROLE = re.compile(r"you are `/root/([a-z0-9_]+)`")


def agent_of(body: dict) -> str:
    found = ROLE.search(json.dumps(body))
    return found.group(1) if found else "?"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    def send(self, status: int, data: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if "/models" in self.path:
            self.send(200, json.dumps({"object": "list", "data": [], "models": []}).encode(), "application/json")
        else:
            self.send(404, b'{"error":"not found"}', "application/json")

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.split("?")[0].endswith("/responses"):
            self.send(404, b'{"error":"not found"}', "application/json")
            return
        agent = agent_of(body)
        with LOCK:
            REQUESTS[agent] = REQUESTS.get(agent, 0) + 1
            LOCK.notify_all()
            index = STEPS.get(agent, 0)
            plan = PLANS.get(agent, [])
            step = plan[index] if index < len(plan) else {"type": "text", "text": "done"}
            STEPS[agent] = index + 1
            gate = step.get("after")
            if gate:
                LOCK.wait_for(lambda: REQUESTS.get(gate["agent"], 0) >= gate["requests"], timeout=120)
        with LOCK:
            LOG.write(json.dumps({"ts": time.time(), "agent": agent, "step": step, "body": body}) + "\n")
            LOG.flush()
        if step["type"] == "shell":
            item = {"type": "function_call", "id": "fc_" + uuid.uuid4().hex[:20], "call_id": "call_" + uuid.uuid4().hex[:20],
                    "name": "exec_command", "arguments": json.dumps({"cmd": step["command"]})}
        elif step["type"] == "call":
            item = {"type": "function_call", "id": "fc_" + uuid.uuid4().hex[:20], "call_id": "call_" + uuid.uuid4().hex[:20],
                    "namespace": step["namespace"], "name": step["name"], "arguments": json.dumps(step["arguments"])}
        elif step["type"] == "js":
            item = {"type": "custom_tool_call", "id": "ctc_" + uuid.uuid4().hex[:20],
                    "call_id": "call_" + uuid.uuid4().hex[:20], "name": "exec", "input": step["code"]}
        else:
            item = {"type": "message", "role": "assistant", "id": "msg_" + uuid.uuid4().hex[:20], "status": "completed",
                    "content": [{"type": "output_text", "text": step.get("text", "ok"), "annotations": []}]}
        usage = {"input_tokens": 100, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": 10,
                 "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": 110}
        resp = "resp_" + uuid.uuid4().hex[:20]
        events = [
            ("response.created", {"type": "response.created", "sequence_number": 0,
                                  "response": {"id": resp, "object": "response", "status": "in_progress", "output": []}}),
            ("response.output_item.added", {"type": "response.output_item.added", "sequence_number": 1,
                                            "output_index": 0, "item": item}),
            ("response.output_item.done", {"type": "response.output_item.done", "sequence_number": 2,
                                           "output_index": 0, "item": item}),
            ("response.completed", {"type": "response.completed", "sequence_number": 3,
                                    "response": {"id": resp, "object": "response", "status": "completed",
                                                 "output": [item], "usage": usage}}),
        ]
        data = "".join(f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events).encode()
        self.send(200, data, "text/event-stream")


def main() -> None:
    global LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--ready", required=True)
    args = parser.parse_args()
    with open(args.plans) as stream:
        PLANS.update(json.load(stream))
    LOG = open(args.log, "a")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    with open(args.ready + ".tmp", "w") as stream:
        stream.write(str(server.server_address[1]))
    os.replace(args.ready + ".tmp", args.ready)
    server.serve_forever()


if __name__ == "__main__":
    main()
