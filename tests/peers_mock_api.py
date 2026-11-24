#!/usr/bin/env python3
"""A local mock of the Anthropic Messages API for a claude_peers team (tests/test_claude_peers.py): one process serves
every agent, each from its own plan.

    peers_mock_api.py --plans plans.json --log requests.jsonl --ready FILE

plans.json maps an agent id to its steps. A request belongs to the agent whose name its system prompt gives (agent.py's
appended line "In the team you are `<id>`."), and a main request (one that offers tools) takes that agent's next step:
  {"type": "text", "text": "..."}                         final answer, ending the turn
  {"type": "tool", "name": "SendMessage", "input": {...}}  one call of any tool
  "after": {"agent": "a2", "requests": 2}                 on any step: answer only once that agent has made this many
                                                          main requests (at most 60 s)
  "delay": 5                                              on any step: answer this many seconds later, as a model
                                                          still writing its answer
After its last step an agent's requests get the text "done". Other requests get a short text. Every request is
logged with its agent and full body.
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
NAME = re.compile(r"In the team you are `([A-Za-z0-9_-]+)`\.")


def agent_of(body: dict) -> str:
    found = NAME.search(json.dumps(body.get("system", "")))
    return found.group(1) if found else "?"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, payload: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(payload)))
        self.send_header("request-id", "req_" + uuid.uuid4().hex[:20])
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self._send(json.dumps({"data": [], "models": []}).encode(), "application/json")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("content-length", "0")
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        if not self.path.split("?")[0].endswith("/v1/messages"):
            self._send(json.dumps({"input_tokens": 100}).encode(), "application/json")
            return
        agent, main = agent_of(body), bool(body.get("tools"))
        step = {"type": "text", "text": "ok"}
        with LOCK:
            if main:
                REQUESTS[agent] = REQUESTS.get(agent, 0) + 1
                plan, index = PLANS.get(agent, []), STEPS.get(agent, 0)
                step = plan[index] if index < len(plan) else {"type": "text", "text": "done"}
                STEPS[agent] = index + 1
                LOCK.notify_all()
            after = step.get("after")
            if after:
                LOCK.wait_for(lambda: REQUESTS.get(after["agent"], 0) >= after["requests"], timeout=60)
            LOG.write(json.dumps({"ts": time.time(), "agent": agent, "main": main, "step": step, "body": body}) + "\n")
            LOG.flush()
        if step.get("delay"):
            time.sleep(float(step["delay"]))
        if step["type"] == "tool":
            block = {"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:20], "name": step["name"], "input": {}}
            delta = {"type": "input_json_delta", "partial_json": json.dumps(step["input"])}
            stop = "tool_use"
        else:
            block, delta, stop = {"type": "text", "text": ""}, {"type": "text_delta", "text": step["text"]}, "end_turn"
        usage = {"input_tokens": 100, "output_tokens": 10, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        message = {"id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "role": "assistant",
                   "model": body.get("model", "claude-mock"), "content": [], "stop_reason": None,
                   "stop_sequence": None, "usage": usage}
        if not body.get("stream"):
            content = {**block, "input": step["input"]} if step["type"] == "tool" else {"type": "text", "text": step["text"]}
            self._send(json.dumps({**message, "content": [content], "stop_reason": stop}).encode(), "application/json")
            return
        events = [("message_start", {"type": "message_start", "message": message}),
                  ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": block}),
                  ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
                  ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                  ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                     "usage": {"output_tokens": 10}}),
                  ("message_stop", {"type": "message_stop"})]
        self._send("".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode(),
                   "text/event-stream")


def main() -> None:
    global LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--plans", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--ready", required=True)
    args = parser.parse_args()
    PLANS.update(json.load(open(args.plans)))
    LOG = open(args.log, "a")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    with open(args.ready + ".tmp", "w") as stream:
        stream.write(str(server.server_address[1]))
    os.replace(args.ready + ".tmp", args.ready)
    server.serve_forever()


if __name__ == "__main__":
    main()
