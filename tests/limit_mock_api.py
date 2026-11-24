#!/usr/bin/env python3
"""A local mock of the Anthropic Messages API and the OpenAI Responses API that answers from a
plan and can reject requests with each provider's usage-limit error for a while.

    limit_mock_api.py --port P --plan plan.json --log requests.jsonl [--ready FILE]

The plan is a JSON list of steps. Each "main" request (Claude Code: a /v1/messages request that
offers tools; Codex: a /responses request) takes the next step:
  {"type": "text", "text": "..."}         final answer, ending the turn
  {"type": "tool", "command": "echo hi"}  one shell tool call (Claude Code: Bash; Codex: its
                                          exec tool, named as in the request)
  (Claude Code and Codex: "input_tokens": N on a text or tool step reports N input tokens instead of 100)
  ("output_tokens": N on a text or tool step reports N output tokens instead of 10; "delay": S
   answers it S seconds after the request came, as a model that takes that long to respond: each
   such request also logs {"answered": ts, "seq": the request's} once its answer is sent)
  {"type": "limit", "seconds": 60}        a usage limit starts now and lasts this long: this and
                                          every main request until it ends get the provider's
                                          real usage-limit 429; the next request after it ends
                                          takes the following step
  {"type": "limit", "seconds": 60, "status": 529}   the same with an overload (529) or server
                                          error (500) instead of the usage limit
  {"type": "block"}                       the API's output content filter blocks the reply: Claude
                                          Code: a thinking block, the start of a text block, then
                                          the stream's error event "Output blocked by content
                                          filtering policy" (with "http": 400, that error as an HTTP
                                          400 before any output); Codex: response.incomplete with
                                          incomplete_details.reason "content_filter"
After the last step every main request gets the text "done". Requests that are not main
(Claude Code's side queries without tools, token counting, model lists) get a short answer and
do not take a step. Every request is logged with its full body.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()
STATE = {"step": 0, "limit_until": 0.0, "main": 0, "status": 429, "seq": 0}
# The Anthropic API's error for a reply its output content filter blocked (Claude Code 2.1.289
# recognizes it by this message, with no HTTP status or 400).
BLOCKED = "Output blocked by content filtering policy"
PLAN: list = []
LOG = None


def log(entry: dict) -> None:
    with LOCK:
        LOG.write(json.dumps(entry) + "\n")
        LOG.flush()


def next_step() -> tuple[dict, float]:
    """The step for a main request, and the limit's end time if it is rejected (else 0)."""
    with LOCK:
        STATE["main"] += 1
        now = time.time()
        if now < STATE["limit_until"]:
            return {"type": "limit"}, STATE["limit_until"]
        if STATE["step"] < len(PLAN):
            step = PLAN[STATE["step"]]
            STATE["step"] += 1
            if step["type"] == "limit":
                STATE["limit_until"] = now + float(step["seconds"])
                STATE["status"] = int(step.get("status", 429))
                return step, STATE["limit_until"]
            return step, 0.0
        return {"type": "text", "text": "done"}, 0.0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _body(self) -> dict:
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            return json.loads(raw or b"{}")
        except ValueError:
            return {"_raw": raw.decode("utf-8", "replace")}

    def _send(self, status: int, payload, headers: dict | None = None, content_type="application/json"):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _sse(self, events: list[tuple[str, dict]], headers: dict | None = None):
        data = "".join(f"event: {name}\ndata: {json.dumps(body)}\n\n" for name, body in events).encode()
        self._send(200, data, headers, content_type="text/event-stream")

    @staticmethod
    def _delayed(entry: dict, step: dict) -> int | None:
        """A step with "delay": wait that long before answering; the request's number, to log the answer."""
        if not step.get("delay"):
            return None
        with LOCK:
            STATE["seq"] += 1
            entry["seq"] = STATE["seq"]
        return entry["seq"]

    def _answered(self, seq: int | None, step: dict, received: float) -> None:
        if seq is not None:
            log({"answered": time.time(), "seq": seq, "received": received, "delay": step["delay"]})

    def do_GET(self):
        log({"ts": time.time(), "method": "GET", "path": self.path})
        if "/models" in self.path:
            self._send(200, {"object": "list", "data": [], "models": []})
        else:
            self._send(404, {"error": "not found"})

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("content-length", "0")
        self.end_headers()

    def do_POST(self):
        body = self._body()
        path = self.path.split("?")[0]
        if path.endswith("/v1/messages"):
            self.anthropic(body)
        elif path.endswith("/responses"):
            self.openai(body)
        elif path.endswith("/count_tokens"):
            log({"ts": time.time(), "method": "POST", "path": self.path, "kind": "count_tokens"})
            self._send(200, {"input_tokens": 100})
        else:
            log({"ts": time.time(), "method": "POST", "path": self.path, "kind": "unknown", "body": body})
            self._send(404, {"error": "not found"})

    # ---- Anthropic Messages -------------------------------------------------------------
    def anthropic(self, body: dict):
        main = bool(body.get("tools"))
        step, limit_until = next_step() if main else ({"type": "text", "text": "ok"}, 0.0)
        entry = {"ts": time.time(), "method": "POST", "path": self.path, "provider": "anthropic",
                 "main": main, "step": step, "body": body}
        if limit_until and STATE["status"] != 429:
            entry["response"] = str(STATE["status"])
            log(entry)
            kind = "overloaded_error" if STATE["status"] == 529 else "api_error"
            self._send(STATE["status"], {"type": "error", "error": {"type": kind, "message": "Overloaded"
                                                                     if STATE["status"] == 529 else "Internal server error"}},
                       {"request-id": "req_" + uuid.uuid4().hex[:20]})
            return
        if limit_until:
            reset = int(limit_until) + 1
            entry["response"] = "429"
            log(entry)
            headers = {
                "anthropic-ratelimit-unified-status": "rejected",
                "anthropic-ratelimit-unified-reset": str(reset),
                "anthropic-ratelimit-unified-representative-claim": "five_hour",
                "anthropic-ratelimit-unified-5h-status": "rejected",
                "anthropic-ratelimit-unified-5h-utilization": "1.0",
                "anthropic-ratelimit-unified-5h-reset": str(reset),
                "anthropic-ratelimit-unified-7d-status": "allowed",
                "anthropic-ratelimit-unified-7d-utilization": "0.4",
                "anthropic-ratelimit-unified-7d-reset": str(reset + 86400),
                "anthropic-ratelimit-unified-overage-status": "rejected",
                "anthropic-ratelimit-unified-overage-disabled-reason": "org_level_disabled",
                "anthropic-ratelimit-unified-fallback-percentage": "0.5",
                "retry-after": str(max(1, reset - int(time.time()))),
                "request-id": "req_" + uuid.uuid4().hex[:20],
                "x-should-retry": "true",
            }
            self._send(429, {"type": "error", "error": {"type": "rate_limit_error", "message": "Error"},
                             "request_id": headers["request-id"]}, headers)
            return
        entry["response"] = step["type"]
        seq = self._delayed(entry, step)
        log(entry)
        if step["type"] == "block":
            self.anthropic_block(body, step)
            return
        if seq is not None:
            time.sleep(float(step["delay"]))
        allowed = {
            "anthropic-ratelimit-unified-status": "allowed",
            "anthropic-ratelimit-unified-reset": str(int(time.time()) + 18000),
            "anthropic-ratelimit-unified-representative-claim": "five_hour",
            "anthropic-ratelimit-unified-5h-utilization": "0.1",
            "anthropic-ratelimit-unified-5h-reset": str(int(time.time()) + 18000),
            "anthropic-ratelimit-unified-7d-utilization": "0.3",
            "anthropic-ratelimit-unified-7d-reset": str(int(time.time()) + 400000),
            "anthropic-ratelimit-unified-overage-status": "rejected",
            "anthropic-ratelimit-unified-overage-disabled-reason": "org_level_disabled",
            "request-id": "req_" + uuid.uuid4().hex[:20],
        }
        msg_id = "msg_" + uuid.uuid4().hex[:24]
        model = body.get("model", "claude-mock")
        usage = {"input_tokens": int(step.get("input_tokens", 100)), "output_tokens": 1,
                 "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        if step["type"] == "tool":
            block = {"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:20], "name": "Bash", "input": {}}
            delta = {"type": "input_json_delta",
                     "partial_json": json.dumps({"command": step["command"], "description": "run it"})}
            stop = "tool_use"
        else:
            block = {"type": "text", "text": ""}
            delta = {"type": "text_delta", "text": step.get("text", "ok")}
            stop = "end_turn"
        if not body.get("stream"):
            content = ({**block, "input": json.loads(delta["partial_json"])} if step["type"] == "tool"
                       else {"type": "text", "text": delta["text"]})
            self._send(200, {"id": msg_id, "type": "message", "role": "assistant", "model": model,
                             "content": [content], "stop_reason": stop, "stop_sequence": None,
                             "usage": {**usage, "output_tokens": int(step.get("output_tokens", 10))}}, allowed)
            self._answered(seq, step, entry["ts"])
            return
        self._sse([
            ("message_start", {"type": "message_start", "message": {
                "id": msg_id, "type": "message", "role": "assistant", "model": model, "content": [],
                "stop_reason": None, "stop_sequence": None, "usage": usage}}),
            ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": block}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": delta}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                               "usage": {"output_tokens": int(step.get("output_tokens", 10))}}),
            ("message_stop", {"type": "message_stop"}),
        ], allowed)
        self._answered(seq, step, entry["ts"])

    def anthropic_block(self, body: dict, step: dict):
        """The content filter stops the reply: mid-stream, after a finished thinking block and part
        of a text block, as in the blocked run's log; or, with "http", as an HTTP 400."""
        request_id = "req_" + uuid.uuid4().hex[:20]
        error = {"type": "error", "error": {"type": "invalid_request_error", "message": BLOCKED}}
        if step.get("http"):
            self._send(int(step["http"]), {**error, "request_id": request_id}, {"request-id": request_id})
            return
        usage = {"input_tokens": 100, "output_tokens": 3, "cache_creation_input_tokens": 0,
                 "cache_read_input_tokens": 0}
        self._sse([
            ("message_start", {"type": "message_start", "message": {
                "id": "msg_" + uuid.uuid4().hex[:24], "type": "message", "role": "assistant",
                "model": body.get("model", "claude-mock"), "content": [], "stop_reason": None,
                "stop_sequence": None, "usage": usage}}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "thinking_delta", "thinking": "Writing the file."}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                     "delta": {"type": "signature_delta", "signature": "c2lnbmF0dXJl"}}),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("content_block_start", {"type": "content_block_start", "index": 1,
                                     "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                     "delta": {"type": "text_delta", "text": "Here is the"}}),
            ("error", error),
        ], {"request-id": request_id})

    # ---- OpenAI Responses ---------------------------------------------------------------
    def openai(self, body: dict):
        step, limit_until = next_step()
        entry = {"ts": time.time(), "method": "POST", "path": self.path, "provider": "openai",
                 "main": True, "step": step, "body": body}
        if limit_until and STATE["status"] != 429:
            entry["response"] = str(STATE["status"])
            log(entry)
            self._send(STATE["status"], {"error": {"type": "server_error", "message": "Internal server error"}},
                       {"x-request-id": "req_" + uuid.uuid4().hex[:20]})
            return
        if limit_until:
            reset = int(limit_until) + 1
            entry["response"] = "429"
            log(entry)
            seconds = max(1, reset - int(time.time()))
            headers = {
                "x-codex-primary-used-percent": "100.0",
                "x-codex-primary-window-minutes": "300",
                "x-codex-primary-reset-at": str(reset),
                "x-codex-primary-reset-after-seconds": str(seconds),
                "x-codex-secondary-used-percent": "40.0",
                "x-codex-secondary-window-minutes": "10080",
                "x-codex-secondary-reset-at": str(reset + 86400),
                "x-codex-secondary-reset-after-seconds": str(seconds + 86400),
                "x-request-id": "req_" + uuid.uuid4().hex[:20],
            }
            self._send(429, {"error": {"type": "usage_limit_reached",
                                       "message": "The usage limit has been reached",
                                       "plan_type": "pro", "resets_at": reset,
                                       "resets_in_seconds": seconds}}, headers)
            return
        entry["response"] = step["type"]
        seq = self._delayed(entry, step)
        log(entry)
        if seq is not None:
            time.sleep(float(step["delay"]))
        resp_id = "resp_" + uuid.uuid4().hex[:20]
        if step["type"] == "tool":
            name, args = self.codex_tool(body, step["command"])
            item = {"type": "function_call", "id": "fc_" + uuid.uuid4().hex[:20],
                    "call_id": "call_" + uuid.uuid4().hex[:20], "name": name, "arguments": json.dumps(args)}
        else:
            item = {"type": "message", "role": "assistant", "id": "msg_" + uuid.uuid4().hex[:20],
                    "status": "completed",
                    "content": [{"type": "output_text", "text": step.get("text", "ok"), "annotations": []}]}
        output, inputs = int(step.get("output_tokens", 10)), int(step.get("input_tokens", 100))
        usage = {"input_tokens": inputs, "input_tokens_details": {"cached_tokens": 0}, "output_tokens": output,
                 "output_tokens_details": {"reasoning_tokens": 0}, "total_tokens": inputs + output}
        headers = {"x-codex-primary-used-percent": "10.0", "x-codex-primary-window-minutes": "300",
                   "x-codex-primary-reset-at": str(int(time.time()) + 18000),
                   "x-codex-primary-reset-after-seconds": "18000",
                   "x-request-id": "req_" + uuid.uuid4().hex[:20]}
        if step["type"] == "failed":  # experiment: a response.failed with an error code
            self._sse([
                ("response.created", {"type": "response.created", "sequence_number": 0,
                                      "response": {"id": resp_id, "object": "response", "status": "in_progress",
                                                   "output": []}}),
                ("response.failed", {"type": "response.failed", "sequence_number": 1, "response": {
                    "id": resp_id, "object": "response", "status": "failed", "output": [],
                    "error": {"code": step["code"], "message": step["message"]}}}),
            ], headers)
            return
        if step["type"] == "block":
            self._sse([
                ("response.created", {"type": "response.created", "sequence_number": 0,
                                      "response": {"id": resp_id, "object": "response", "status": "in_progress",
                                                   "output": []}}),
                ("response.incomplete", {"type": "response.incomplete", "sequence_number": 1, "response": {
                    "id": resp_id, "object": "response", "status": "incomplete",
                    "incomplete_details": {"reason": "content_filter"}, "output": [], "usage": usage}}),
            ], headers)
            return
        self._sse([
            ("response.created", {"type": "response.created", "sequence_number": 0,
                                  "response": {"id": resp_id, "object": "response", "status": "in_progress",
                                               "output": []}}),
            ("response.output_item.added", {"type": "response.output_item.added", "sequence_number": 1,
                                            "output_index": 0, "item": item}),
            ("response.output_item.done", {"type": "response.output_item.done", "sequence_number": 2,
                                           "output_index": 0, "item": item}),
            ("response.completed", {"type": "response.completed", "sequence_number": 3,
                                    "response": {"id": resp_id, "object": "response", "status": "completed",
                                                 "output": [item], "usage": usage}}),
        ], headers)
        self._answered(seq, step, entry["ts"])

    @staticmethod
    def codex_tool(body: dict, command: str) -> tuple[str, dict]:
        """The request's shell tool and arguments for the command."""
        for tool in body.get("tools") or []:
            name = tool.get("name") or (tool.get("function") or {}).get("name")
            params = tool.get("parameters") or (tool.get("function") or {}).get("parameters") or {}
            props = params.get("properties") or {}
            if name in ("exec_command", "shell_command", "shell", "local_shell"):
                if "cmd" in props:
                    return name, {"cmd": command}
                if "command" in props:
                    kind = (props["command"] or {}).get("type")
                    return name, {"command": ["bash", "-lc", command] if kind == "array" else command}
        return "exec_command", {"cmd": command}


def main() -> None:
    global PLAN, LOG
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--ready", help="write the port here once listening")
    args = parser.parse_args()
    with open(args.plan) as stream:
        PLAN = json.load(stream)
    LOG = open(args.log, "a")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    if args.ready:
        with open(args.ready + ".tmp", "w") as stream:
            stream.write(str(server.server_address[1]))
        import os
        os.replace(args.ready + ".tmp", args.ready)
    server.serve_forever()


if __name__ == "__main__":
    main()
