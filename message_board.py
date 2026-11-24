#!/usr/bin/env python3
"""The message board of a team of Codex agents that run as separate processes (agent.py; the board
protocol; docs/setups.md).

Codex's own board (feature agent_message_board) belongs to one Codex process tree: its local SQLite file
keys every row by the tree's session id, and a post notifies only threads of its own process. Since
0.162.0 Codex can instead use a board that "the research host" provides over HTTP
(features.multi_agent_v2.message_board_remote: a URL and a bearer token), with the same nine tools, tool
descriptions and notifications. This program is that host for one team, inside the team's container: one
board, every agent's `codex exec` process a member, each told apart by its token. It speaks the protocol of
codex-rs/agent-message-board-client (rust-v0.162.1) and follows the semantics of Codex's local board
(codex-rs/ext/agent-message-board/src/local.rs, local/queries.rs, local/paging.rs): the same limits, sort
orders, offset cursors, preview sizes, subscriptions and error texts. Only the standard library, so it runs
with any task image's python3.

Agents are /root/<id> on the board (/root/a1, /root/a2, ...). Each agent's Codex is /root in its own process,
so a reference to /root (or one relative to it, which Codex resolves against /root) means the caller itself
or one of its teammates.

Every change is appended to a journal (JSON lines), which is the board's store: serve replays it when it
starts again, so posts survive a restart of the board. The journal also records every read, every
notification and whether it reached a running turn. `dump` prints the posts; `sqlite` writes the board in
the schema of Codex's local board (agent_message_board_1.sqlite) plus the reads and notifications.

    message_board.py serve --port P --journal FILE --agents a1,a2,...   (MULTIAGENT_BOARD_SECRET set)
    message_board.py wait --port P [--timeout S]      exits 0 once the board answers
    message_board.py stop --port P                    ends serve (MULTIAGENT_BOARD_SECRET set)
    message_board.py dump JOURNAL [--json]           every post, by channel and thread
    message_board.py sqlite JOURNAL OUT              the board as SQLite
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import queue
import re
import signal
import struct
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SECRET_ENV = "MULTIAGENT_BOARD_SECRET"
MAX_POST_BYTES = 64 * 1024
MAX_CHANNEL_BYTES = 128
MAX_READ_CHARS = 20_000
MAX_PAGE = 50
NOTICE_CHARS = 150
# The client drops a remote notice whose content, author and recipient exceed 1024 bytes
# (codex-rs/core/src/agent_message_board.rs); a longer preview is shortened to fit.
NOTICE_BYTES = 1024
MAX_REQUEST_BYTES = 1024 * 1024
HEARTBEAT_SECONDS = 10.0
ROOT = "/root"
NAME = re.compile(r"^[a-z0-9_]+$")
TIME = re.compile(r"^(\d{4})-(\d{2})-(\d{2})[Tt ](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?(Z|z|[+-]\d{2}:\d{2})$")
# The journal's events that change the board, which serve replays.
CHANGES = ("channel", "post", "subscribe", "unsubscribe", "join")


class Invalid(Exception):
    """A request the board refuses (HTTP 400): the agent sees "invalid_request: <message>"."""


def token(agent: str, secret: str) -> str:
    """An agent's credential: Codex accepts 32-4096 printable ASCII characters."""
    return f"ttc-message-board-{agent}-{secret}"


def admin_token(secret: str) -> str:
    return f"ttc-message-board-admin-{secret}"


def now_us() -> int:
    return time.time_ns() // 1000


def iso(us: int) -> str:
    seconds, micros = divmod(us, 1_000_000)
    return datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + f".{micros:06d}Z"


def parse_time(text) -> int:
    """RFC 3339 (chrono's serialization of DateTime<Utc>) to microseconds since the epoch."""
    found = TIME.match(text) if isinstance(text, str) else None
    if not found:
        raise Invalid("invalid timestamp")
    year, month, day, hour, minute, second, fraction, zone = found.groups()
    stamp = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), tzinfo=timezone.utc)
    us = int(stamp.timestamp()) * 1_000_000 + int((fraction or "0")[:6].ljust(6, "0"))
    if zone not in ("Z", "z"):
        sign = 1 if zone[0] == "+" else -1
        us -= sign * (int(zone[1:3]) * 3600 + int(zone[4:6]) * 60) * 1_000_000
    return us


def uuid7() -> str:
    """A version 7 UUID, as Codex's Uuid::now_v7, for post ids."""
    ms = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    value = (ms & ((1 << 48) - 1)) << 80 | 0x7 << 76 | (rand >> 68 & 0xFFF) << 64 | 0b10 << 62 | rand & ((1 << 62) - 1)
    return str(uuid.UUID(int=value))


def fold(text: str) -> str:
    """Case folding for case-insensitive search (Codex: caseless::default_case_fold_str)."""
    return text.casefold()


def cursor_offset(cursor) -> int:
    if cursor is None:
        return 0
    if not isinstance(cursor, str) or "=" in cursor:
        raise Invalid("invalid cursor")
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
    except (ValueError, TypeError):
        raise Invalid("invalid cursor")
    if len(raw) != 4:
        raise Invalid("invalid cursor")
    return struct.unpack(">I", raw)[0]


def page_of(results: list, page: dict) -> dict:
    """Offset pagination as Codex's local board (local/paging.rs Window): RESULTS are every match in order."""
    offset = cursor_offset(page.get("cursor"))
    limit = min(positive(page.get("limit"), "limit"), MAX_PAGE)
    window = results[offset:offset + limit + 1]
    has_more = len(window) > limit
    window = window[:limit]
    next_cursor = None
    if has_more:
        if offset + len(window) > 0xFFFFFFFF:
            raise Invalid("cursor offset exceeds the board limit")
        next_cursor = base64.urlsafe_b64encode(struct.pack(">I", offset + len(window))).rstrip(b"=").decode()
    return {"results": window, "n_returned": len(window), "has_more": next_cursor is not None,
            "next_cursor": next_cursor}


def positive(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Invalid(f"{name} must be a positive integer")
    return value


def validate_channel(name) -> str:
    if (not isinstance(name, str) or not name or len(name.encode()) > MAX_CHANNEL_BYTES or name.strip() != name
            or any(unicodedata.category(char) == "Cc" for char in name)):
        raise Invalid("channel names must contain 1–128 bytes without edge whitespace or control characters")
    return name


def target_key(target: dict) -> str:
    """A subscription target as Codex stores it: the JSON of {"Channel": name} or {"Thread": id}."""
    return json.dumps(target, separators=(",", ":"))


def one_variant(value, variants: tuple) -> tuple:
    if not isinstance(value, dict) or len(value) != 1 or next(iter(value)) not in variants:
        raise Invalid(f"expected one of {', '.join(variants)}")
    return next(iter(value.items()))


class Watch:
    """One running turn's notification stream."""

    def __init__(self, caller: str, turn_id: str):
        self.caller = caller
        self.turn_id = turn_id
        self.queue: queue.Queue = queue.Queue()

    def close(self) -> None:
        self.queue.put(None)


class Board:
    """One team's board. Every operation runs under one lock; notifications go out after it is released."""

    def __init__(self, agents: list, journal: str | None):
        for agent in agents:
            if not NAME.match(agent) or agent == "root":
                raise ValueError(f"agent ids must be lowercase letters, digits and underscores: {agent!r}")
        self.members = [f"{ROOT}/{agent}" for agent in agents]
        self.lock = threading.RLock()
        self.channels: dict = {}
        self.posts: list = []
        self.by_id: dict = {}
        self.by_request: dict = {}
        self.subscriptions: set = set()
        self.opt_outs: set = set()
        self.watches: dict = {member: [] for member in self.members}
        self.sessions: dict = {}
        self.seq = 0
        self.journal_path = journal
        self.journal = None
        if journal:
            self.replay(journal)
            self.journal = open(journal, "a", encoding="utf-8")

    # ---- the journal

    def replay(self, path: str) -> None:
        try:
            with open(path, encoding="utf-8", errors="replace") as stream:
                lines = stream.read().splitlines()
        except FileNotFoundError:
            return
        for line in lines:
            try:
                event = json.loads(line)
            except ValueError:
                continue  # a line cut short when serve was killed
            if isinstance(event, dict) and event.get("e") in CHANGES:
                self.apply(event)

    def apply(self, event: dict) -> None:
        kind = event["e"]
        if kind == "channel":
            self.channels[event["name"]] = {"name": event["name"], "search": fold(event["name"]),
                                            "us": event["us"], "author": event["author"]}
        elif kind == "post":
            post = {name: event[name] for name in ("seq", "id", "channel", "root", "author", "us", "text",
                                                   "request_id", "request")}
            self.posts.append(post)
            self.by_id[post["id"]] = post
            self.by_request[post["request_id"]] = post
            self.seq = max(self.seq, post["seq"])
        elif kind == "subscribe":
            key = (event["target"], event["agent"])
            if event.get("explicit"):
                self.opt_outs.discard(key)
                self.subscriptions.add(key)
            elif key not in self.opt_outs:
                self.subscriptions.add(key)
        elif kind == "unsubscribe":
            key = (event["target"], event["agent"])
            self.subscriptions.discard(key)
            self.opt_outs.add(key)
        elif kind == "join":
            self.sessions.setdefault(event["agent"], []).append(event["session"])

    def record(self, event: dict, change: bool = False) -> None:
        """Append to the journal, one whole line at a time; a change is applied first and synced to disk."""
        with self.lock:
            if change:
                self.apply(event)
            if self.journal is None:
                return
            event = {"t": iso(now_us()), **event}
            self.journal.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
            self.journal.flush()
            if change:
                os.fsync(self.journal.fileno())

    # ---- members

    def resolve(self, path, caller: str) -> str:
        if not isinstance(path, str):
            raise Invalid("agent path must be a string")
        if path == ROOT:
            return caller
        if path in self.members:
            return path
        raise Invalid(f"agent {path} is not on this board; the team is {', '.join(self.members)}")

    def joined(self, agent: str, board: str) -> None:
        with self.lock:
            if board not in self.sessions.get(agent, []):
                self.record({"e": "join", "agent": agent, "session": board}, change=True)

    # ---- records as Codex serializes them

    @staticmethod
    def metadata(post: dict) -> dict:
        return {"message_id": post["id"], "channel_name": post["channel"], "author": post["author"],
                "thread_id": post["root"], "created_at": iso(post["us"])}

    def preview(self, post: dict, chars: int) -> dict:
        n_chars = len(post["text"])
        return {**self.metadata(post), "text_preview": post["text"][:chars], "n_chars": n_chars,
                "truncated": n_chars > chars}

    def channel_posts(self, name: str) -> list:
        return [post for post in self.posts if post["channel"] == name]

    @staticmethod
    def newest(posts: list):
        return max(posts, key=lambda post: (post["us"], post["seq"])) if posts else None

    def channel_summary(self, name: str) -> dict:
        channel = self.channels.get(name)
        if channel is None:
            raise Invalid("channel not found in this board")
        posts = self.channel_posts(name)
        last = self.newest(posts)
        return {"channel_name": name, "created_at": iso(channel["us"]), "created_by": channel["author"],
                "message_count": len(posts), "last_message_id": last["id"] if last else None}

    def load_post(self, post_id) -> dict:
        post = self.by_id.get(post_id) if isinstance(post_id, str) else None
        if post is None:
            raise Invalid("post not found in this board")
        return post

    def thread_root(self, post_id) -> dict:
        post = self.load_post(post_id)
        if post["root"] != post["id"]:
            raise Invalid("thread_id must identify a top-level post")
        return post

    def subscribe_implicitly(self, target: dict, agent: str) -> None:
        """Participation subscribes by default, without overriding an explicit opt-out."""
        key = target_key(target)
        if (key, agent) not in self.opt_outs and (key, agent) not in self.subscriptions:
            self.record({"e": "subscribe", "target": key, "agent": agent}, change=True)

    def insert_channel(self, name: str, author: str, us: int) -> None:
        if name in self.channels:
            raise Invalid("channel already exists")
        self.record({"e": "channel", "name": name, "us": us, "author": author}, change=True)

    # ---- operations (codex-rs/ext/agent-message-board/src/api.rs)

    def call(self, caller: str, session: str, body: dict) -> tuple:
        """One /call: (result, notices to deliver). Raises Invalid."""
        method, params = body.get("method"), body.get("params")
        if not isinstance(params, dict):
            raise Invalid("params must be an object")
        stamp = body.get("timestamp")
        handler = getattr(self, f"op_{method}", None) if isinstance(method, str) else None
        if handler is None:
            raise Invalid(f"unknown method {method!r}")
        notices: list = []
        with self.lock:
            result = handler(caller, session, params, None if stamp is None else parse_time(stamp), notices)
        return result, notices

    def op_create_channel(self, caller, session, params, stamp, notices):
        name = validate_channel(params.get("channel_name"))
        change = params.get("subscription")
        if change not in ("Subscribe", "Unsubscribe"):
            raise Invalid("subscription must be Subscribe or Unsubscribe")
        if stamp is None:
            raise Invalid("creating a channel needs the caller's clock")
        self.insert_channel(name, caller, stamp)
        if change == "Subscribe":
            self.subscribe_implicitly({"Channel": name}, caller)
        return self.channel_summary(name)

    def op_list_channels(self, caller, session, params, stamp, notices):
        query = fold(params.get("query") or "")
        names = [name for name, channel in self.channels.items() if query in channel["search"]]

        def activity(name):
            last = self.newest(self.channel_posts(name))
            return (last["us"] if last else self.channels[name]["us"], name)

        names.sort(key=activity, reverse=direction(params) == "DESC")
        page = page_of(names, params.get("page") or {})
        page["results"] = [self.channel_summary(name) for name in page["results"]]
        return page

    def op_post(self, caller, session, params, stamp, notices):
        text, request_id = params.get("text"), params.get("request_id")
        recipients_in = params.get("agents_to_notify") or []
        if (not isinstance(text, str) or not isinstance(request_id, str) or not isinstance(recipients_in, list)
                or len(text.encode()) > MAX_POST_BYTES or not text or not request_id or len(request_id) > 512
                or len(recipients_in) > 256):
            raise Invalid("post text, request ID or recipient count exceeds the board limits")
        key = f"{session}:{request_id}"
        request = json.dumps(params, sort_keys=True, separators=(",", ":"))
        existing = self.by_request.get(key)
        if existing is not None:
            if existing["request"] != request:
                raise Invalid("request ID was already used for a different post")
            return self.metadata(existing)
        recipients = {self.resolve(path, caller) for path in recipients_in}
        if stamp is None:
            raise Invalid("posting needs the caller's clock; no post was created")
        kind, value = one_variant(params.get("destination"), ("Channel", "NewChannel", "Thread"))
        post_id = uuid7()
        if kind == "Channel":
            if value not in self.channels:
                raise Invalid("channel not found in this board")
            channel, root, target = value, post_id, {"Channel": value}
        elif kind == "NewChannel":
            validate_channel(value)
            self.insert_channel(value, caller, stamp)
            self.subscribe_implicitly({"Channel": value}, caller)
            channel, root, target = value, post_id, {"Channel": value}
        else:
            parent = self.thread_root(value)
            channel, root, target = parent["channel"], parent["id"], {"Thread": parent["id"]}
        subscribed = target_key(target)
        recipients |= {agent for found, agent in self.subscriptions if found == subscribed}
        recipients.discard(caller)
        self.seq += 1
        self.record({"e": "post", "seq": self.seq, "id": post_id, "channel": channel, "root": root, "author": caller,
                     "us": stamp, "text": text, "request_id": key, "request": request}, change=True)
        self.subscribe_implicitly({"Thread": root}, caller)
        post = self.by_id[post_id]
        for recipient in sorted(recipients):
            notices.append((recipient, post))
        return self.metadata(post)

    def op_list_threads(self, caller, session, params, stamp, notices):
        name = params.get("channel_name")
        if name not in self.channels:
            raise Invalid("channel not found in this board")
        sort = params.get("sort")
        if sort not in ("created", "activity"):
            raise Invalid("sort must be created or activity")
        posts = self.channel_posts(name)
        roots = [post for post in posts if post["id"] == post["root"]]
        latest = {}
        for post in posts:
            found = latest.get(post["root"])
            if found is None or (post["us"], post["seq"]) > (found["us"], found["seq"]):
                latest[post["root"]] = post

        def order(root):
            return (latest[root["id"]]["us"] if sort == "activity" else root["us"], root["seq"])

        roots.sort(key=order, reverse=direction(params) == "DESC")
        page = page_of(roots, params.get("page") or {})
        limit = min(positive((params.get("page") or {}).get("limit"), "limit"), MAX_PAGE)
        fetched = len(roots[cursor_offset((params.get("page") or {}).get("cursor")):][:limit + 1])
        chars = min(positive(params.get("max_chars_per_post"), "max_chars_per_post"),
                    MAX_READ_CHARS // (2 * max(1, min(fetched, limit))))
        threads = []
        for root in page["results"]:
            replies = [post for post in posts if post["root"] == root["id"] and post["id"] != root["id"]]
            last = self.newest(replies)
            threads.append({"thread_id": root["id"], "root_post": self.preview(root, chars),
                            "reply_count": len(replies),
                            "last_activity_at": iso(max(last["us"], root["us"]) if last else root["us"]),
                            "latest_reply": self.preview(last, chars) if last else None})
        page["results"] = threads
        return page

    def op_search_posts(self, caller, session, params, stamp, notices):
        found = list(self.posts)
        if params.get("channel_name") is not None:
            found = [post for post in found if post["channel"] == params["channel_name"]]
        if params.get("author") is not None:
            # As in Codex's local board, an author who never posted matches nothing (no error).
            author = caller if params["author"] == ROOT else params["author"]
            found = [post for post in found if post["author"] == author]
        if params.get("query") is not None:
            query = fold(params["query"])
            found = [post for post in found if query in fold(post["text"])]
        if params.get("after_message_id") is not None:
            after = self.load_post(params["after_message_id"])
            found = [post for post in found if (post["us"], post["seq"]) > (after["us"], after["seq"])]
        found.sort(key=lambda post: (post["us"], post["seq"]), reverse=True)
        page_request = params.get("page") or {}
        page = page_of(found, page_request)
        limit = min(positive(page_request.get("limit"), "limit"), MAX_PAGE)
        fetched = len(found[cursor_offset(page_request.get("cursor")):][:limit + 1])
        chars = min(positive(params.get("max_chars_per_post"), "max_chars_per_post"),
                    MAX_READ_CHARS // max(1, min(fetched, limit)))
        page["results"] = [self.preview(post, chars) for post in page["results"]]
        return page

    def op_read_thread(self, caller, session, params, stamp, notices):
        root = self.thread_root(params.get("thread_id"))
        replies = [post for post in self.posts if post["root"] == root["id"] and post["id"] != root["id"]]
        replies.sort(key=lambda post: (post["us"], post["seq"]), reverse=True)
        page_request = params.get("page") or {}
        page = page_of(replies, page_request)
        limit = min(positive(page_request.get("limit"), "limit"), MAX_PAGE)
        fetched = len(replies[cursor_offset(page_request.get("cursor")):][:limit + 1])
        chars = min(positive(params.get("max_chars_per_post"), "max_chars_per_post"),
                    MAX_READ_CHARS // (min(fetched, limit) + 1))
        page["results"] = [self.preview(post, chars) for post in page["results"]]
        return {"root_post": self.preview(root, chars), **page}

    def op_read_post(self, caller, session, params, stamp, notices):
        post = self.load_post(params.get("message_id"))
        offset = params.get("offset_chars", 0)
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise Invalid("offset_chars must be a nonnegative integer")
        n_chars = len(post["text"])
        offset = min(offset, n_chars)
        text = post["text"][offset:offset + min(positive(params.get("limit_chars"), "limit_chars"), MAX_READ_CHARS)]
        return {**self.metadata(post), "text": text, "n_chars": n_chars, "next_offset_chars": offset + len(text)}

    def op_set_subscription(self, caller, session, params, stamp, notices):
        agent = self.resolve(params["target_agent"], caller) if params.get("target_agent") is not None else caller
        change = params.get("change")
        if change not in ("Subscribe", "Unsubscribe"):
            raise Invalid("change must be Subscribe or Unsubscribe")
        kind, value = one_variant(params.get("target"), ("Channel", "Thread"))
        if kind == "Channel":
            channel, thread, last = value, None, self.channel_summary(value)["last_message_id"]
        else:
            root = self.thread_root(value)
            newest = self.newest([post for post in self.posts if post["root"] == root["id"]])
            channel, thread, last = root["channel"], root["id"], newest["id"]
        key = target_key({kind: value})
        if change == "Subscribe":
            self.record({"e": "subscribe", "target": key, "agent": agent, "by": caller, "explicit": True}, change=True)
        else:
            self.record({"e": "unsubscribe", "target": key, "agent": agent, "by": caller}, change=True)
        return {"channel_name": channel, "thread_id": thread, "target_agent": agent,
                "enabled": change == "Subscribe", "last_message_id": last}

    # ---- notifications

    def watch(self, agent: str, watch: Watch) -> None:
        with self.lock:
            live = self.watches[agent]
            # A new turn of the same Codex session replaces its earlier turns' streams.
            for old in [old for old in live if old.caller == watch.caller]:
                old.close()
                live.remove(old)
            live.append(watch)
            self.record({"e": "watch", "agent": agent, "caller": watch.caller, "turn": watch.turn_id})

    def unwatch(self, agent: str, watch: Watch) -> None:
        with self.lock:
            if watch in self.watches[agent]:
                self.watches[agent].remove(watch)
                self.record({"e": "unwatch", "agent": agent, "caller": watch.caller, "turn": watch.turn_id})

    def notice_text(self, post: dict, chars: int) -> dict:
        """The preview a notification carries: NOTICE_CHARS characters, fewer if the rendered notice
        would exceed what the client accepts."""
        while True:
            preview = self.preview(post, chars)
            rendered = (f"Message Type: CHANNEL_POST\nSender: {post['author']}\nChannel: {post['channel']}\n"
                        f"Message ID: {post['id']}\nThread ID: {post['root']}\nPayload:\n{preview['text_preview']}"
                        + ("\n[Use read_post for the rest.]" if preview["truncated"] else ""))
            if len(rendered.encode()) + len(post["author"]) + len(ROOT) <= NOTICE_BYTES - 64 or chars == 0:
                return preview
            chars //= 2

    def deliver(self, notices: list) -> None:
        """Best effort, once: only turns that are running get the notice; nothing is kept for later."""
        for recipient, post in notices:
            with self.lock:
                live = list(self.watches.get(recipient, ()))
                preview = self.notice_text(post, NOTICE_CHARS)
                for watch in live:
                    watch.queue.put({"recipient": watch.caller, "turn_id": watch.turn_id, "post": preview})
                self.record({"e": "notify", "post": post["id"], "recipient": recipient,
                             "turns": [watch.turn_id for watch in live]})


def direction(params: dict) -> str:
    value = params.get("direction")
    if value not in ("NewestFirst", "OldestFirst"):
        raise Invalid("direction must be NewestFirst or OldestFirst")
    return "DESC" if value == "NewestFirst" else "ASC"


# ---- the HTTP service ----------------------------------------------------------------------------------

BOARD_PATH = re.compile(r"^/v1/boards/([^/]+)/(call|notifications)$")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ttc-message-board"
    board: Board = None  # set by serve
    secret = ""
    tokens: dict = {}

    def log_message(self, fmt, *args):  # quiet: the journal is the record
        pass

    def send_json(self, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def fail(self, status: int, code: str, message: str) -> None:
        self.send_json(status, {"code": code, "message": message})

    def body(self):
        length = int(self.headers.get("content-length") or 0)
        if length > MAX_REQUEST_BYTES:
            raise Invalid("board request exceeds the service limit")
        raw = self.rfile.read(length) if length else b""
        try:
            value = json.loads(raw or b"{}")
        except ValueError:
            raise Invalid("request body is not JSON")
        if not isinstance(value, dict):
            raise Invalid("request body must be an object")
        return value

    def bearer(self) -> str:
        found = self.headers.get("authorization") or ""
        return found[len("Bearer "):] if found.startswith("Bearer ") else ""

    def do_GET(self):
        if self.path == "/healthz":
            self.send_json(200, {"ok": True, "members": self.board.members})
        else:
            self.fail(404, "not_found", "no such endpoint")

    def do_PUT(self):
        self.fail(405, "unsupported", "this board is provisioned by the harness")

    def do_DELETE(self):
        self.fail(405, "unsupported", "this board is provisioned by the harness")

    def do_POST(self):
        if self.path == "/v1/admin/stop":
            if self.bearer() != admin_token(self.secret):
                self.fail(401, "unauthorized", "unknown board credential")
                return
            self.send_json(200, {"stopping": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        found = BOARD_PATH.match(self.path)
        if not found:
            self.fail(404, "not_found", "no such endpoint")
            return
        agent = self.tokens.get(self.bearer())
        if agent is None:
            self.fail(401, "unauthorized", "unknown board credential")
            return
        session, endpoint = found.groups()
        try:
            body = self.body()
            caller = body.get("caller")
            if not isinstance(caller, str) or not caller:
                raise Invalid("caller is missing")
            self.board.joined(agent, session)
            if endpoint == "notifications":
                self.stream(agent, caller, body)
                return
            started = time.monotonic()
            try:
                result, notices = self.board.call(agent, caller, body)
            except Invalid as error:
                self.board.record({"e": "call", "agent": agent, "method": body.get("method"),
                                   "params": body.get("params"), "error": str(error)})
                raise
            if body.get("method") != "post":
                self.board.record({"e": "call", "agent": agent, "method": body.get("method"),
                                   "params": body.get("params"),
                                   "ms": round((time.monotonic() - started) * 1000, 1)})
            self.send_json(200, result)
            self.board.deliver(notices)
        except Invalid as error:
            self.fail(400, "invalid_request", str(error))
        except Exception as error:  # noqa: BLE001 - the client retries a server error
            self.board.record({"e": "error", "agent": agent, "error": f"{type(error).__name__}: {error}"})
            self.fail(500, "internal", f"{type(error).__name__}: {error}")

    def chunk(self, text: str) -> None:
        data = text.encode()
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def stream(self, agent: str, caller: str, body: dict) -> None:
        turn = body.get("turn_id")
        if not isinstance(turn, str) or not turn:
            raise Invalid("turn_id is missing")
        watch = Watch(caller, turn)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        self.close_connection = True
        self.board.watch(agent, watch)
        try:
            self.chunk("event: ready\ndata: {}\n\n")
            while True:
                try:
                    item = watch.queue.get(timeout=HEARTBEAT_SECONDS)
                except queue.Empty:
                    self.chunk(": heartbeat\n\n")
                    continue
                if item is None:
                    break
                self.chunk("event: notification\ndata: " + json.dumps(item, ensure_ascii=False) + "\n\n")
            self.wfile.write(b"0\r\n\r\n")
        except (OSError, ValueError):
            pass  # the turn ended and Codex closed the stream
        finally:
            self.board.unwatch(agent, watch)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(port: int, journal: str, agents: list) -> int:
    secret = os.environ.get(SECRET_ENV, "")
    if len(secret) < 8:
        print(f"{SECRET_ENV} must hold the team's board secret", file=sys.stderr)
        return 2
    board = Board(agents, journal)
    Handler.board = board
    Handler.secret = secret
    Handler.tokens = {token(agent, secret): f"{ROOT}/{agent}" for agent in agents}
    try:
        server = Server(("127.0.0.1", port), Handler)
    except OSError as error:
        print(f"the message board cannot listen on 127.0.0.1:{port}: {error}", file=sys.stderr)
        return 2
    board.record({"e": "start", "pid": os.getpid(), "port": port, "members": board.members,
                  "posts": len(board.posts)})
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        with board.lock:
            for live in board.watches.values():
                for watch in live:
                    watch.close()
        board.record({"e": "stop", "pid": os.getpid()})
        server.server_close()
    return 0


def wait(port: int, timeout: float) -> int:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as response:
                if response.status == 200:
                    return 0
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() >= deadline:
            print(f"the message board did not answer on 127.0.0.1:{port} within {timeout:.0f} s", file=sys.stderr)
            return 1
        time.sleep(0.2)


def stop(port: int) -> int:
    request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/admin/stop", data=b"{}", method="POST",
                                     headers={"authorization": f"Bearer {admin_token(os.environ.get(SECRET_ENV, ''))}",
                                              "content-type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10):
            return 0
    except (OSError, urllib.error.URLError) as error:
        print(f"could not stop the message board: {error}", file=sys.stderr)
        return 1


# ---- reading a journal ----------------------------------------------------------------------------------

def load(path: str) -> list:
    events = []
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                events.append(event)
    return events


def posts_of(events: list) -> list:
    return [event for event in events if event.get("e") == "post"]


def dump(path: str, as_json: bool) -> int:
    events = load(path)
    posts = posts_of(events)
    if as_json:
        for post in posts:
            print(json.dumps({"seq": post["seq"], "id": post["id"], "channel": post["channel"], "thread": post["root"],
                              "author": post["author"], "created_at": iso(post["us"]), "text": post["text"]},
                             ensure_ascii=False))
        return 0
    channels = [event for event in events if event.get("e") == "channel"]
    reads = sum(1 for event in events if event.get("e") == "call" and "error" not in event)
    print(f"{len(posts)} posts in {len(channels)} channels; {reads} reads")
    for channel in channels:
        print(f"\n#{channel['name']}  (created by {channel['author']}, {iso(channel['us'])})")
        roots = [post for post in posts if post["channel"] == channel["name"] and post["id"] == post["root"]]
        for root in roots:
            for post in [root] + [p for p in posts if p["root"] == root["id"] and p["id"] != root["id"]]:
                indent = "  " if post is root else "      "
                print(f"{indent}[{iso(post['us'])}] {post['author']}:")
                for line in post["text"].splitlines() or [""]:
                    print(f"{indent}  {line}")
    return 0


# Codex's local schema (codex-rs/ext/agent-message-board/src/local.rs SCHEMA), one board, plus what only
# this board records.
SQLITE_SCHEMA = """
CREATE TABLE deleted_boards (board TEXT PRIMARY KEY NOT NULL);
CREATE TABLE channels (board TEXT NOT NULL, name TEXT NOT NULL, name_search TEXT NOT NULL, created_at TEXT NOT NULL,
 timestamp INTEGER NOT NULL, author TEXT NOT NULL, PRIMARY KEY(board,name));
CREATE TABLE posts (seq INTEGER PRIMARY KEY AUTOINCREMENT, board TEXT NOT NULL, id TEXT NOT NULL, channel TEXT NOT NULL,
 root TEXT NOT NULL, author TEXT NOT NULL, timestamp INTEGER NOT NULL, body_search TEXT NOT NULL, payload TEXT NOT NULL,
 request_id TEXT NOT NULL, request TEXT NOT NULL, UNIQUE(board,id), UNIQUE(board,request_id));
CREATE TABLE subscriptions (board TEXT NOT NULL, target TEXT NOT NULL, agent TEXT NOT NULL, PRIMARY KEY(board,target,agent));
CREATE TABLE subscription_opt_outs (board TEXT NOT NULL, target TEXT NOT NULL, agent TEXT NOT NULL,
 PRIMARY KEY(board,target,agent));
CREATE TABLE reads (time TEXT NOT NULL, agent TEXT NOT NULL, method TEXT, params TEXT, error TEXT);
CREATE TABLE notifications (time TEXT NOT NULL, post TEXT NOT NULL, recipient TEXT NOT NULL, turns TEXT NOT NULL);
CREATE TABLE sessions (agent TEXT NOT NULL, session TEXT NOT NULL);
"""
SQLITE_BOARD = "team"


def to_sqlite(path: str, out: str) -> int:
    import sqlite3

    events = load(path)
    if os.path.exists(out):
        os.unlink(out)
    board = Board([], None)
    for event in events:
        if event.get("e") in CHANGES and event.get("e") != "join":
            board.apply(event)
    db = sqlite3.connect(out)
    with db:
        db.executescript(SQLITE_SCHEMA)
        for name, channel in board.channels.items():
            db.execute("INSERT INTO channels VALUES(?,?,?,?,?,?)",
                       (SQLITE_BOARD, name, channel["search"], iso(channel["us"]), channel["us"], channel["author"]))
        for post in board.posts:
            payload = {"metadata": Board.metadata(post), "text": post["text"]}
            db.execute("INSERT INTO posts VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (post["seq"], SQLITE_BOARD, post["id"], post["channel"], post["root"], post["author"], post["us"],
                        fold(post["text"]), json.dumps(payload, ensure_ascii=False), post["request_id"],
                        post["request"]))
        db.executemany("INSERT INTO subscriptions VALUES(?,?,?)",
                       [(SQLITE_BOARD, target, agent) for target, agent in sorted(board.subscriptions)])
        db.executemany("INSERT INTO subscription_opt_outs VALUES(?,?,?)",
                       [(SQLITE_BOARD, target, agent) for target, agent in sorted(board.opt_outs)])
        for event in events:
            if event.get("e") == "call":
                db.execute("INSERT INTO reads VALUES(?,?,?,?,?)",
                           (event.get("t", ""), event.get("agent", ""), event.get("method"),
                            json.dumps(event.get("params"), ensure_ascii=False), event.get("error")))
            elif event.get("e") == "notify":
                db.execute("INSERT INTO notifications VALUES(?,?,?,?)",
                           (event.get("t", ""), event["post"], event["recipient"], json.dumps(event.get("turns"))))
            elif event.get("e") == "join":
                db.execute("INSERT INTO sessions VALUES(?,?)", (event["agent"], event["session"]))
    db.close()
    return 0


def main(argv: list) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("serve")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--journal", required=True)
    p.add_argument("--agents", required=True)
    p = sub.add_parser("wait")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--timeout", type=float, default=60.0)
    p = sub.add_parser("stop")
    p.add_argument("--port", type=int, required=True)
    p = sub.add_parser("dump")
    p.add_argument("journal")
    p.add_argument("--json", action="store_true")
    p = sub.add_parser("sqlite")
    p.add_argument("journal")
    p.add_argument("out")
    args = parser.parse_args(argv)
    if args.command == "serve":
        return serve(args.port, args.journal, [agent for agent in args.agents.split(",") if agent])
    if args.command == "wait":
        return wait(args.port, args.timeout)
    if args.command == "stop":
        return stop(args.port)
    if args.command == "dump":
        return dump(args.journal, args.json)
    return to_sqlite(args.journal, args.out)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
