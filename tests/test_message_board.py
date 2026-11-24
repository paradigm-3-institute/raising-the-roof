"""The message board for Codex teams (message_board.py, the board protocol, agent.py, run.py; docs/setups.md).

The service tests talk to message_board.py over HTTP as Codex's remote board client does
(codex-rs/agent-message-board-client, rust-v0.162.1). The end-to-end test runs a team of two real
`codex exec` processes (the pinned 0.162.1 release, skipped without it) on this machine against
tests/board_mock_api.py, with agent.py's runners, keeper and board: no network, no model.
"""

from __future__ import annotations

import asyncio
import base64
import http.client
import importlib
import json
import os
import re
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from cli_render import BASE_ENV, INSTRUCTION, ROOT, FakeEnvironment

sys.path.insert(0, str(ROOT))
import message_board  # noqa: E402
from multiagent_protocols.base import ProtocolPaths, build_protocol  # noqa: E402

SERVICE = ROOT / "message_board.py"
SECRET = "test-board-secret-0123456789"
BOARD_CODEX = ROOT / ".cache/codex-0.162.1/bin/codex"
OLD_CODEX = ROOT / ".cache/codex-0.160.0/bin/codex"
needs_board_codex = pytest.mark.skipif(not BOARD_CODEX.is_file(), reason="pinned Codex 0.162.1 release missing")
# The board protocol's prompt (BODY in multiagent_protocols/board.py), written out in full.
BOARD_MD = """\
You are one of {n} agents working on this task at the same time, in this container, all
with this same prompt. You are one team: the team is scored on its best result, so
whatever any one of you achieves counts for all of you.

The team shares a message board, like a team Slack: channels, threads and posts that
everyone can read, search and subscribe to.
{messaging}

How you split the work, what you share, and when you talk is up to you. Work it out
among yourselves.
"""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# ---- the service ------------------------------------------------------------------------------------

class Service:
    def __init__(self, tmp: Path, agents=("a1", "a2", "a3")):
        self.tmp, self.agents = tmp, agents
        self.port = free_port()
        self.journal = tmp / "board.jsonl"
        self.process = None
        self.start()

    def start(self) -> None:
        self.process = subprocess.Popen(
            [sys.executable, str(SERVICE), "serve", "--port", str(self.port), "--journal", str(self.journal),
             "--agents", ",".join(self.agents)], env={**os.environ, message_board.SECRET_ENV: SECRET},
            stderr=subprocess.PIPE, text=True)
        done = subprocess.run([sys.executable, str(SERVICE), "wait", "--port", str(self.port), "--timeout", "20"],
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr

    def stop(self) -> None:
        done = subprocess.run([sys.executable, str(SERVICE), "stop", "--port", str(self.port)],
                              env={**os.environ, message_board.SECRET_ENV: SECRET}, capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
        assert self.process.wait(timeout=20) == 0

    def request(self, path: str, body, token: str | None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=20)
        headers = {"content-type": "application/json"}
        if token:
            headers["authorization"] = f"Bearer {token}"
        connection.request("POST", path, json.dumps(body), headers)
        response = connection.getresponse()
        data = json.loads(response.read() or b"null")
        connection.close()
        return response.status, data

    def call(self, agent: str, method: str, params: dict, stamp: str | None = "2026-10-15T10:00:00.123456789Z",
             session: str | None = None):
        session = session or f"thread-{agent}"
        return self.request(f"/v1/boards/tree-{agent}/call",
                            {"caller": session, "timestamp": stamp, "method": method, "params": params},
                            message_board.token(agent, SECRET))

    def ok(self, agent: str, method: str, params: dict, **kwargs):
        status, data = self.call(agent, method, params, **kwargs)
        assert status == 200, data
        return data


@pytest.fixture
def service(tmp_path):
    found = Service(tmp_path)
    yield found
    if found.process.poll() is None:
        found.process.kill()
        found.process.wait()


def page(limit=20, cursor=None) -> dict:
    return {"cursor": cursor, "limit": limit}


def post(service, agent, text, destination, notify=(), request="call-1", **kwargs):
    return service.ok(agent, "post", {"request_id": f"turn:{request}", "destination": destination, "text": text,
                                      "agents_to_notify": list(notify)}, **kwargs)


def stamp(second: int) -> str:
    return f"2026-10-15T10:00:{second:02d}Z"


def test_posts_threads_and_reads_follow_codex_local_board(service):
    root = post(service, "a1", "Plan: I take the parser.", {"NewChannel": "plan"}, stamp=stamp(1))
    assert root["author"] == "/root/a1" and root["channel_name"] == "plan" and root["thread_id"] == root["message_id"]
    assert root["created_at"] == "2026-10-15T10:00:01.000000Z"
    reply = post(service, "a2", "I take the code generator.", {"Thread": root["message_id"]}, request="c2",
                 stamp=stamp(2))
    assert reply["thread_id"] == root["message_id"] and reply["author"] == "/root/a2"
    second = post(service, "a3", "Tests next.", {"Channel": "plan"}, request="c3", stamp=stamp(3))
    channels = service.ok("a2", "list_channels", {"query": "PL", "direction": "NewestFirst", "page": page()})
    assert channels["results"][0] == {"channel_name": "plan", "created_at": "2026-10-15T10:00:01.000000Z",
                                      "created_by": "/root/a1", "message_count": 3,
                                      "last_message_id": second["message_id"]}
    assert channels["has_more"] is False and channels["next_cursor"] is None and channels["n_returned"] == 1
    threads = service.ok("a1", "list_threads", {"channel_name": "plan", "sort": "created", "direction": "NewestFirst",
                                                "page": page(), "max_chars_per_post": 1000})
    assert [t["thread_id"] for t in threads["results"]] == [second["message_id"], root["message_id"]]
    first = threads["results"][1]
    assert first["reply_count"] == 1 and first["latest_reply"]["message_id"] == reply["message_id"]
    assert first["last_activity_at"] == "2026-10-15T10:00:02.000000Z"
    by_activity = service.ok("a1", "list_threads", {"channel_name": "plan", "sort": "activity",
                                                    "direction": "OldestFirst", "page": page(),
                                                    "max_chars_per_post": 5})
    assert [t["thread_id"] for t in by_activity["results"]] == [root["message_id"], second["message_id"]]
    assert by_activity["results"][0]["root_post"]["text_preview"] == "Plan:"
    assert by_activity["results"][0]["root_post"]["truncated"] is True
    found = service.ok("a3", "search_posts", {"channel_name": None, "query": "CODE", "after_message_id": None,
                                              "author": None, "page": page(), "max_chars_per_post": 1000})
    assert [p["message_id"] for p in found["results"]] == [reply["message_id"]]
    mine = service.ok("a2", "search_posts", {"channel_name": None, "query": None, "after_message_id": None,
                                             "author": "/root", "page": page(), "max_chars_per_post": 1000})
    assert [p["author"] for p in mine["results"]] == ["/root/a2"]
    after = service.ok("a2", "search_posts", {"channel_name": "plan", "query": None,
                                              "after_message_id": root["message_id"], "author": None, "page": page(),
                                              "max_chars_per_post": 1000})
    assert [p["message_id"] for p in after["results"]] == [second["message_id"], reply["message_id"]]
    thread = service.ok("a3", "read_thread", {"thread_id": root["message_id"], "page": page(),
                                              "max_chars_per_post": 1000})
    assert thread["root_post"]["message_id"] == root["message_id"]
    assert [p["message_id"] for p in thread["results"]] == [reply["message_id"]]
    content = service.ok("a3", "read_post", {"message_id": root["message_id"], "offset_chars": 6, "limit_chars": 5})
    assert content["text"] == "I tak" and content["n_chars"] == 24 and content["next_offset_chars"] == 11


def test_paging_uses_codex_offset_cursors(service):
    post(service, "a1", "zero", {"NewChannel": "log"}, request="p0", stamp=stamp(0))
    for second in range(1, 5):
        post(service, "a1", f"post {second}", {"Channel": "log"}, request=f"p{second}", stamp=stamp(second))
    first = service.ok("a2", "list_threads", {"channel_name": "log", "sort": "created", "direction": "OldestFirst",
                                              "page": page(2), "max_chars_per_post": 100})
    assert first["has_more"] is True and first["next_cursor"] == base64.urlsafe_b64encode(
        (2).to_bytes(4, "big")).rstrip(b"=").decode()
    second = service.ok("a2", "list_threads", {"channel_name": "log", "sort": "created", "direction": "OldestFirst",
                                               "page": page(2, first["next_cursor"]), "max_chars_per_post": 100})
    third = service.ok("a2", "list_threads", {"channel_name": "log", "sort": "created", "direction": "OldestFirst",
                                              "page": page(2, second["next_cursor"]), "max_chars_per_post": 100})
    texts = [t["root_post"]["text_preview"] for result in (first, second, third) for t in result["results"]]
    assert texts == ["zero", "post 1", "post 2", "post 3", "post 4"] and third["next_cursor"] is None
    status, error = service.call("a2", "list_threads", {"channel_name": "log", "sort": "created",
                                                        "direction": "OldestFirst", "page": page(2, "!!"),
                                                        "max_chars_per_post": 100})
    assert status == 400 and error == {"code": "invalid_request", "message": "invalid cursor"}


def test_retries_errors_and_credentials(service):
    params = {"request_id": "turn-1:call-1", "destination": {"NewChannel": "plan"}, "text": "hi",
              "agents_to_notify": []}
    first = service.ok("a1", "post", params)
    assert service.ok("a1", "post", params) == first  # a retry of the same call
    status, error = service.call("a1", "post", {**params, "text": "changed"})
    assert status == 400 and error["message"] == "request ID was already used for a different post"
    status, error = service.call("a1", "post", {**params, "request_id": "turn-1:call-2"})
    assert status == 400 and error["message"] == "channel already exists"
    status, error = service.call("a1", "post", {**params, "request_id": "x", "destination": {"Channel": "nope"}})
    assert status == 400 and error["message"] == "channel not found in this board"
    status, error = service.call("a1", "post", {**params, "request_id": "y", "agents_to_notify": ["/root/zz"]})
    assert status == 400 and "not on this board" in error["message"]
    status, error = service.call("a1", "post", {**params, "request_id": "z", "text": "x" * (64 * 1024 + 1)})
    assert status == 400 and error["message"] == "post text, request ID or recipient count exceeds the board limits"
    status, error = service.call("a1", "post", {**params, "request_id": "w"}, stamp=None)
    assert status == 400 and "clock" in error["message"]
    status, error = service.call("a1", "create_channel", {"channel_name": " padded", "subscription": "Subscribe"})
    assert status == 400 and error["message"].startswith("channel names must contain")
    status, error = service.request("/v1/boards/tree/call", {"caller": "t", "method": "list_channels", "params": {}},
                                    "ttc-message-board-a1-wrong-secret-000000")
    assert status == 401 and error["code"] == "unauthorized"
    status, error = service.request("/v1/boards/tree/call", {"caller": "t", "method": "list_channels", "params": {}},
                                    None)
    assert status == 401


class Stream:
    """One turn's notification stream, as the Codex client opens it."""

    def __init__(self, service, agent: str, turn: str, session: str | None = None):
        self.connection = http.client.HTTPConnection("127.0.0.1", service.port, timeout=30)
        self.connection.request("POST", f"/v1/boards/tree-{agent}/notifications",
                                json.dumps({"caller": session or f"thread-{agent}", "turn_id": turn}),
                                {"authorization": f"Bearer {message_board.token(agent, SECRET)}",
                                 "content-type": "application/json"})
        self.response = self.connection.getresponse()
        assert self.response.status == 200 and self.response.getheader("content-type") == "text/event-stream"
        assert self.event() == ("ready", "{}")

    def event(self) -> tuple[str, str]:
        name, data = "", ""
        while True:
            line = self.response.readline().decode()
            if line == "":
                return "", ""
            line = line.rstrip("\r\n")
            if line.startswith(":"):
                continue
            if not line:
                if name:
                    return name, data
                continue
            key, _, value = line.partition(": ")
            name, data = (value, data) if key == "event" else (name, value)

    def close(self) -> None:
        self.connection.close()


def test_notifications_reach_subscribers_and_named_agents_with_a_running_turn(service):
    a2 = Stream(service, "a2", "turn-7")
    a3 = Stream(service, "a3", "turn-1")
    root = post(service, "a1", "Who takes the lexer?", {"NewChannel": "plan"}, notify=["/root/a2"], stamp=stamp(1))
    name, data = a2.event()
    notice = json.loads(data)
    assert name == "notification" and notice["recipient"] == "thread-a2" and notice["turn_id"] == "turn-7"
    assert notice["post"]["text_preview"] == "Who takes the lexer?" and notice["post"]["author"] == "/root/a1"
    # a3 was neither named nor subscribed: its turn gets nothing; subscribing (as a1, for a3) changes that.
    state = service.ok("a1", "set_subscription", {"target": {"Channel": "plan"}, "target_agent": "/root/a3",
                                                  "change": "Subscribe"})
    assert state == {"channel_name": "plan", "thread_id": None, "target_agent": "/root/a3", "enabled": True,
                     "last_message_id": root["message_id"]}
    long = "x" * 400
    post(service, "a1", long, {"Channel": "plan"}, request="c2", stamp=stamp(2))
    name, data = a3.event()
    notice = json.loads(data)["post"]
    assert notice["text_preview"] == "x" * 150 and notice["truncated"] is True and notice["n_chars"] == 400
    # Replying subscribes a2 to the thread; a1, who started it, hears of the reply.
    a1 = Stream(service, "a1", "turn-2")
    post(service, "a2", "I do.", {"Thread": root["message_id"]}, request="c3", stamp=stamp(3))
    name, data = a1.event()
    assert json.loads(data)["post"]["text_preview"] == "I do."
    for stream in (a1, a2, a3):
        stream.close()
    events = message_board.load(str(service.journal))
    delivered = [e for e in events if e["e"] == "notify"]
    assert [(e["recipient"], e["turns"]) for e in delivered] == [
        ("/root/a2", ["turn-7"]), ("/root/a3", ["turn-1"]), ("/root/a1", ["turn-2"])]


def test_an_idle_agent_gets_no_notice_and_a_new_turn_replaces_the_old(service):
    post(service, "a1", "Nobody is listening.", {"NewChannel": "plan"}, notify=["/root/a2"], stamp=stamp(1))
    old = Stream(service, "a2", "turn-1")
    new = Stream(service, "a2", "turn-2")
    assert old.event() == ("", "")  # ended by the new turn's stream
    post(service, "a1", "Now someone is.", {"Channel": "plan"}, notify=["/root/a2"], request="c2", stamp=stamp(2))
    assert json.loads(new.event()[1])["turn_id"] == "turn-2"
    new.close()
    events = [e for e in message_board.load(str(service.journal)) if e["e"] == "notify"]
    assert [e["turns"] for e in events] == [[], ["turn-2"]]


def test_unsubscribe_is_kept_when_the_agent_posts_again(service):
    root = post(service, "a1", "topic", {"NewChannel": "plan"}, stamp=stamp(1))
    service.ok("a1", "set_subscription", {"target": {"Thread": root["message_id"]}, "target_agent": None,
                                          "change": "Unsubscribe"})
    post(service, "a1", "more", {"Thread": root["message_id"]}, request="c2", stamp=stamp(2))
    stream = Stream(service, "a1", "turn-1")
    post(service, "a2", "reply", {"Thread": root["message_id"]}, request="c3", stamp=stamp(3))
    post(service, "a2", "again", {"Thread": root["message_id"]}, notify=["/root/a1"], request="c4", stamp=stamp(4))
    # Only the post that names a1 reaches it.
    assert json.loads(stream.event()[1])["post"]["text_preview"] == "again"
    stream.close()


def test_the_journal_survives_a_restart_and_reads_as_sqlite(service, tmp_path):
    root = post(service, "a1", "Straße 🦀 plan", {"NewChannel": "plan"}, stamp=stamp(1))
    post(service, "a2", "ok", {"Thread": root["message_id"]}, request="c2", stamp=stamp(2))
    service.ok("a3", "read_post", {"message_id": root["message_id"], "offset_chars": 0, "limit_chars": 100})
    service.stop()
    service.start()
    thread = service.ok("a3", "read_thread", {"thread_id": root["message_id"], "page": page(),
                                              "max_chars_per_post": 1000})
    assert thread["root_post"]["text_preview"] == "Straße 🦀 plan" and len(thread["results"]) == 1
    # Idempotency survives too.
    assert post(service, "a1", "Straße 🦀 plan", {"NewChannel": "plan"}, stamp=stamp(1)) == root
    later = post(service, "a3", "after the restart", {"Channel": "plan"}, request="c9", stamp=stamp(9))
    service.stop()
    out = tmp_path / "board.sqlite"
    assert message_board.main(["sqlite", str(service.journal), str(out)]) == 0
    db = sqlite3.connect(out)
    rows = db.execute("SELECT seq, author, channel, root, json_extract(payload, '$.text') FROM posts ORDER BY seq").fetchall()
    assert rows == [(1, "/root/a1", "plan", root["message_id"], "Straße 🦀 plan"),
                    (2, "/root/a2", "plan", root["message_id"], "ok"),
                    (3, "/root/a3", "plan", later["message_id"], "after the restart")]
    assert db.execute("SELECT agent FROM reads WHERE method='read_post'").fetchall() == [("/root/a3",)]
    assert ("/root/a2", "tree-a2") in db.execute("SELECT agent, session FROM sessions").fetchall()
    db.close()
    dump = subprocess.run([sys.executable, str(SERVICE), "dump", str(service.journal)], capture_output=True, text=True)
    assert dump.returncode == 0 and "3 posts in 1 channels" in dump.stdout and "/root/a2:" in dump.stdout


def test_a_long_notice_is_shortened_to_what_codex_accepts():
    found = message_board.Board(["a1", "a2"], None)
    post_ = {"id": "0" * 36, "root": "0" * 36, "channel": "c" * 128, "author": "/root/a1", "text": "🦀" * 150,
             "us": 0, "seq": 1}
    # With the harness's agent ids a full preview always fits (about 930 of the 1024 bytes).
    assert found.notice_text(post_, 150)["text_preview"] == "🦀" * 150
    preview = found.notice_text({**post_, "author": "/root/" + "a" * 120}, 150)
    assert 0 < len(preview["text_preview"]) < 150 and preview["truncated"] is True


# ---- the protocols ----------------------------------------------------------------------------------

def build(protocol: str, agents: int):
    with mock.patch.dict(os.environ, {"MULTIAGENT_N_AGENT": str(agents)}):
        return build_protocol(f"multiagent_protocols.{protocol}:MultiAgentProtocol", INSTRUCTION,
                              ProtocolPaths.default("/logs/agent/multiagent-collab"), "gpt-6.1-sol")


def test_board_prompt_is_ttc_study_board_md():
    spec = build("board", 3)
    messaging = ("Your teammates are /root/a1, /root/a2, /root/a3. The board's tools (post, read_thread, "
                 "subscribe, ...) are collaboration tools.")
    assert [agent.prompt for agent in spec.agents] == [BOARD_MD.format(n=3, messaging=messaging)] * 3
    assert spec.metadata["message_board"] == {"members": ["/root/a1", "/root/a2", "/root/a3"]}
    assert spec.metadata["role_intro"] == "" and spec.shared_paths == []
    with pytest.raises(ValueError, match="needs a team"):
        build("board", 1)


def load_agent(env: dict):
    with mock.patch.dict(os.environ, env):
        sys.modules.pop("agent", None)
        return importlib.import_module("agent")


def team_env(protocol: str = "board", agents: int = 2, extra: dict | None = None) -> dict:
    return {**BASE_ENV, "RELEASE_TASK": "ale-bench", "MULTIAGENT_N_AGENT": str(agents),
            "MULTIAGENT_PROTOCOL": f"multiagent_protocols.{protocol}:MultiAgentProtocol", "MULTIAGENT_AGENT_EFFORT": "high",
            "RELEASE_CONTINUATION": (ROOT / "benchmarks/ale-bench/continuation.txt").read_text().rstrip("\n"),
            **(extra or {})}


def test_board_config_is_the_harness_config_plus_the_board():
    module = load_agent(team_env())
    base = tomllib.loads(module.CODEX_CONFIG_TOML)
    found = tomllib.loads(module.board_codex_config())
    assert found.pop("suppress_unstable_features_warning") is True
    assert found.pop("developer_instructions") == "On the team message board you are `/root/@board_agent@`."
    # The board, and no other multi-agent setting: the model's own multi-agent v2, as without the board.
    v2 = found["features"].pop("multi_agent_v2")
    assert v2 == {"enabled": True, "message_board_remote": {"url": "http://127.0.0.1:39613",
                                                            "bearer_token_env_var": "MULTIAGENT_BOARD_TOKEN"}}
    assert found["features"].pop("agent_message_board") is True
    assert base["features"].pop("multi_agent_v2") is False
    assert found["shell_environment_policy"].pop("exclude") == [
        "CODEX_API_KEY", "OPENAI_API_KEY", "MULTIAGENT_BOARD_TOKEN", "MULTIAGENT_BOARD_SECRET"]
    base["shell_environment_policy"].pop("exclude")
    assert found == base


class BoardFake(FakeEnvironment):
    """FakeEnvironment whose board runs until it is stopped."""

    def __init__(self) -> None:
        super().__init__()
        self.stopped = asyncio.Event()

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        await asyncio.sleep(0.01)  # a real exec waits for the container, so other tasks run meanwhile
        result = await super().exec(command, user=user, env=env, cwd=cwd, timeout_sec=timeout_sec)
        if command.startswith("exec /usr/lib/hsv/hsbi /usr/lib/hsv/hsbd serve "):
            await self.stopped.wait()
        elif command == "/usr/lib/hsv/hsbi /usr/lib/hsv/hsbd stop --port 39613":
            self.stopped.set()
        return result


def render_board(tmp_path: Path, cls: str = "CodexCollaborationAgent", protocol: str = "board", agents: int = 2,
                 extra: dict | None = None, **kwargs) -> tuple[object, list[dict]]:
    env = team_env(protocol, agents, extra)
    module = load_agent(env)
    with mock.patch.dict(os.environ, env), mock.patch("secrets.token_urlsafe", lambda n: "s" * 32):
        agent = getattr(module, cls)(logs_dir=tmp_path, model_name="gpt-6.1-sol", **kwargs)
        environment = BoardFake()

        async def go() -> None:
            await agent.install(environment)
            await agent.run(INSTRUCTION, environment, SimpleNamespace())

        asyncio.run(go())
    return agent, environment.calls


def test_a_codex_team_on_the_board(tmp_path):
    agent, calls = render_board(tmp_path, extra={"RELEASE_HARNESS_GUARD": "1", "RELEASE_LIMIT_RESUME": "1",
                                                 "RELEASE_TOKEN_BUDGET": "4000000", "RELEASE_RESOURCES_TOOL": "1",
                                                 "RELEASE_AGENT_TIMEOUT": "172800"})
    commands = [call["command"] for call in calls]
    install = next(i for i, c in enumerate(commands) if "/usr/lib/hsv/hsbd" in c and "printf %s" in c)
    serve = next(i for i, c in enumerate(commands) if c.startswith("exec /usr/lib/hsv/hsbi /usr/lib/hsv/hsbd serve "))
    keeper = next(i for i, c in enumerate(commands) if c.startswith("exec /usr/lib/hsv/hsvi /usr/lib/hsv/hsvd k"))
    stop = next(i for i, c in enumerate(commands) if c.endswith(" stop --port 39613"))
    assert install < serve < keeper < stop
    assert calls[install]["user"] == "root" and calls[serve]["user"] == "root"
    assert commands[serve] == ("exec /usr/lib/hsv/hsbi /usr/lib/hsv/hsbd serve --port 39613 --journal "
                               "/logs/agent/message-board.jsonl --agents a1,a2 2>> /logs/agent/message-board-service.log")
    assert calls[serve]["env"] == {"MULTIAGENT_BOARD_SECRET": "s" * 32}
    assert calls[keeper]["env"]["MULTIAGENT_BOARD_SECRET"] == "s" * 32
    runner = agent.last_runner_script
    config = runner.split("<<'TOML'\n", 1)[1].split("TOML\n", 1)[0]
    assert config == sys.modules["agent"].board_codex_config()
    assert ('TOML\n  sed -i "s|@board_agent@|$agent_id|" "$codex_log_dir/config.toml"\n'
            '  export MULTIAGENT_BOARD_TOKEN="ttc-message-board-$agent_id-${MULTIAGENT_BOARD_SECRET:-}"\n') in runner
    prompt = next(c for c in commands if "shared-prompt.md" in c and "printf %s" in c)
    assert "Follow this coordination protocol exactly" not in "".join(commands)
    assert "Your teammates are /root/a1, /root/a2." in prompt
    # Everything else is the team's usual machinery: each agent's own budget watcher, limit watcher, resources.
    trial = next(c for c in commands if "> /usr/lib/hsv/conf.json" in c)
    conf = json.loads(re.search(r"printf %s '(\{\n.*?\n\})' > /usr/lib/hsv/conf\.json", trial, re.S).group(1))
    assert [sorted(unit["watchers"]) for unit in conf["units"]] == [["limit", "token"], ["limit", "token"]]


def test_the_board_runs_a_revived_team_from_its_journal(tmp_path):
    env = team_env(extra={"RELEASE_HARNESS_GUARD": "1"})
    module = load_agent(env)
    with mock.patch.dict(os.environ, env), mock.patch("secrets.token_urlsafe", lambda n: "s" * 32):
        agent = module.CodexCollaborationAgent(logs_dir=tmp_path, model_name="gpt-6.1-sol")
        agent.revive_plan = {"revival": 1, "note": "Continue.", "layout": "team", "started": 1.0,
                             "checks_passed": True, "agents": [{"index": 0, "session_id": "s0", "turn": 2},
                                                               {"index": 1, "session_id": "s1", "turn": 3}]}
        environment = BoardFake()
        asyncio.run(agent.resume(INSTRUCTION, environment, SimpleNamespace()))
    assert any(" serve --port 39613 --journal /logs/agent/message-board.jsonl" in c["command"] for c in environment.calls)


@pytest.mark.parametrize("cls, kwargs", [
    ("ClaudeCodeCollaborationAgent", {}),
    ("MixedTeamAgent", {"team": [{"cli": "claude", "model": "claude-sonnet-5-5"}, {"cli": "codex", "model": "gpt-6.1-sol"}]}),
])
@pytest.mark.parametrize("protocol", ["board"])
def test_teams_without_a_board_refuse_it(tmp_path, cls, kwargs, protocol):
    with pytest.raises(ValueError, match="message board, and .* has none"):
        render_board(tmp_path, cls=cls, protocol=protocol, **kwargs)


def test_one_codex_agent_refuses_the_board(tmp_path):
    with pytest.raises(ValueError, match="one agent has no board"):
        render_board(tmp_path, agents=1)


# ---- run.py ---------------------------------------------------------------------------------------------

def run_py(*argv: str):
    import run
    args = run.parser().parse_args(list(argv))
    args.dry_run = True
    return run.prepare(args)


SOL_BOARD = ("frontier-swe", "--task", "git-to-zig", "--cli", "codex", "--codex-binary",
             ".cache/codex-0.162.1/bin/codex", "--model", "gpt-6.1-sol", "--effort", "high", "--agents", "2",
             "--token-budget", "4000000", "--timeout", "172800", "--resources-tool", "--cpus", "8",
             "--memory-mb", "32768", "--name", "board-dry")


@pytest.mark.parametrize("protocol", ["board"])
def test_run_py_runs_a_codex_team_on_the_board(protocol):
    cmd, env, _ = run_py(*SOL_BOARD, "--protocol", protocol)
    assert env["MULTIAGENT_PROTOCOL"] == f"multiagent_protocols.{protocol}:MultiAgentProtocol"
    assert cmd[cmd.index("--agent-import-path") + 1] == "agent:CodexCollaborationAgent"
    assert env["RELEASE_HARNESS_GUARD"] == "1" and env["RELEASE_LIMIT_RESUME"] == "1"


@pytest.mark.parametrize("argv, message", [
    (("--cli", "claude", "--model", "claude-sonnet-5-5", "--agents", "2"), "which only Codex has"),
    (("--team", "claude:claude-sonnet-5-5,codex:gpt-6.1-sol"), "which only Codex has"),
    (("--team", "codex:gpt-6.1-sol,codex:gpt-6-luna", "--codex-binary", ".cache/codex-0.162.1/bin/codex"),
     "which only Codex has"),
    (("--cli", "codex", "--model", "gpt-6.1-sol", "--codex-binary", ".cache/codex-0.162.1/bin/codex", "--agents", "1"),
     "needs a team"),
    (("--cli", "codex", "--model", "gpt-6.1-sol", "--codex-binary", ".cache/codex-0.160.0/bin/codex", "--agents", "2"),
     r"needs Codex 0\.162\.0 or later"),
])
def test_run_py_refuses_the_board(argv, message):
    if "0.160.0" in " ".join(argv) and not OLD_CODEX.is_file():
        pytest.skip("pinned Codex 0.160.0 release missing")
    with pytest.raises(ValueError, match=message):
        run_py("ale-bench", "--problem", "ahc014", "--protocol", "board", *argv)


# ---- end to end: two `codex exec` processes on one board ------------------------------------------------

class BoardMock:
    def __init__(self, tmp: Path, plans: dict):
        (tmp / "plans.json").write_text(json.dumps(plans))
        self.log = tmp / "requests.jsonl"
        ready = tmp / "port"
        self.process = subprocess.Popen([sys.executable, str(ROOT / "tests/board_mock_api.py"), "--plans",
                                         str(tmp / "plans.json"), "--log", str(self.log), "--ready", str(ready)])
        for _ in range(200):
            if ready.exists():
                break
            time.sleep(0.05)
        self.port = int(ready.read_text())

    def requests(self, agent: str) -> list:
        return [entry for entry in map(json.loads, self.log.read_text().splitlines()) if entry["agent"] == agent]

    def stop(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=10)


class Team:
    """A 2-agent Codex team of PROTOCOL run by agent.py (runners, keeper, board) on this machine, its agents
    answered by board_mock_api.py from PLANS."""

    def __init__(self, tmp: Path, protocol: str, model: str, plans: dict):
        from test_harness_guard import GUARD, LIMIT, Local

        tmp.mkdir(parents=True, exist_ok=True)
        self.mock = BoardMock(tmp, plans)
        port = free_port()
        provider = {"name": "mock", "base_url": f"http://127.0.0.1:{self.mock.port}/v1", "wire_api": "responses",
                    "env_key": "CODEX_API_KEY", "supports_websockets": "false", "request_max_retries": "0",
                    "stream_max_retries": "0"}
        args = ["-c", 'model_provider="mock"']
        for key, value in provider.items():
            args += ["-c", f"model_providers.mock.{key}={value if value in ('false', '0') else json.dumps(value)}"]
        hook = tmp / "hook.sh"
        hook.write_text("task_cli_args=(" + " ".join(f"'{arg}'" for arg in args) + ")\n")
        home = tmp / "home"
        home.mkdir()
        base = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(home), "HTTPS_PROXY": "http://127.0.0.1:9",
                "HTTP_PROXY": "http://127.0.0.1:9", "NO_PROXY": "127.0.0.1,localhost", "RELEASE_TASK_CLI_HOOK": str(hook),
                "META_DUO_CODEX_API_RETRY_BASE_SLEEP": "2"}

        class BoardLocal(Local):
            def patch(self, command: str) -> str:
                return super().patch(command).replace("39613", str(port))

        self.local = BoardLocal(tmp, base)
        env = {**team_env(protocol, extra={**LIMIT, **GUARD, "RELEASE_TOKEN_BUDGET": "4000000"}),
               "RELEASE_CONTINUATIONS": "0", "MULTIAGENT_AGENT_EFFORT": "",
               "META_DUO_CONTAINER_CODEX_BINARY": str(BOARD_CODEX), "CODEX_API_KEY": "sk-mock"}
        self.error: list = []

        def main() -> None:
            try:
                module = load_agent(env)
                with mock.patch.dict(os.environ, env):
                    agent = module.CodexCollaborationAgent(logs_dir=self.local.agent_dir, model_name=model)
                    asyncio.run(agent.run(INSTRUCTION, self.local, SimpleNamespace()))
            except BaseException as exc:  # noqa: BLE001
                self.error.append(exc)

        thread = threading.Thread(target=main, daemon=True)
        thread.start()
        thread.join(300)
        self.ended = not thread.is_alive()

    def requests(self, agent: str) -> list:
        return self.mock.requests(agent)

    def journal(self) -> list:
        return message_board.load(str(self.local.agent_dir / "message-board.jsonl"))

    def close(self) -> None:
        from test_harness_guard import kill
        import harness_guard

        self.mock.stop()
        state = harness_guard.read_json(str(self.local.guard_dir / "state" / "keeper.json")) or {}
        if state.get("pid") and harness_guard.alive(state["pid"], state.get("start")):
            kill(state["pid"])
        for session in self.local.agent_dir.glob("sessions-*"):
            kill(harness_guard.agent_processes("CODEX_HOME", str(session / "codex-home")))


@pytest.fixture
def team(tmp_path):
    teams: list[Team] = []

    def start(*args, **kwargs) -> Team:
        found = Team(tmp_path / f"team{len(teams)}", *args, **kwargs)
        teams.append(found)
        assert found.ended, "the run did not end"
        assert not found.error, repr(found.error)
        return found

    yield start
    for found in teams:
        found.close()


def outputs(request: dict) -> list[str]:
    return [item.get("output") if isinstance(item.get("output"), str) else json.dumps(item.get("output"))
            for item in request["body"].get("input", [])
            if item.get("type") in ("function_call_output", "custom_tool_call_output")]


def notices(request: dict) -> list[dict]:
    return [item for item in request["body"]["input"] if item.get("type") == "agent_message"]


def tools_of(request: dict) -> dict:
    """Every tool a request offers, by namespace.name: in `tools`, or in the additional_tools item that code-mode
    models get."""
    found = {}
    groups = list(request["body"].get("tools") or [])
    for item in request["body"].get("input", []):
        if item.get("type") == "additional_tools":
            groups += item.get("tools", [])
    for group in groups:
        for tool in group.get("tools", [group]):
            found[f"{group.get('name')}.{tool.get('name')}"] = tool
    return found


def developer_texts(request: dict) -> list[str]:
    """Each text part of the developer messages (Codex puts several in one message)."""
    return [part.get("text", "") for item in request["body"]["input"]
            if item.get("type") == "message" and item.get("role") == "developer" for part in item.get("content", [])]


def assert_board_record(found: Team) -> None:
    """The journal holds a1's post, the notice that reached a2's running turn, and a2's read; the run's
    SQLite has them too; each agent's tokens are its own session's, as in any team."""
    import token_budget

    events = found.journal()
    posts = [e for e in events if e["e"] == "post"]
    assert [(p["author"], p["channel"], p["text"]) for p in posts] == [("/root/a1", "plan", "a1 takes the parser.")]
    notified = [e for e in events if e["e"] == "notify"]
    assert notified[0]["recipient"] == "/root/a2" and len(notified[0]["turns"]) == 1
    assert any(e["e"] == "call" and e["agent"] == "/root/a2" and e["method"] == "search_posts" for e in events)
    joined = {e["agent"]: e["session"] for e in events if e["e"] == "join"}
    assert set(joined) == {"/root/a1", "/root/a2"} and joined["/root/a1"] != joined["/root/a2"]
    db = sqlite3.connect(found.local.agent_dir / "message-board.sqlite")
    assert db.execute("SELECT author, json_extract(payload, '$.text') FROM posts").fetchall() == [
        ("/root/a1", "a1 takes the parser.")]
    db.close()
    for index in (0, 1):
        home_dir = next(found.local.agent_dir.glob(f"sessions-{index}-*")) / "codex-home"
        totals = token_budget.summed(token_budget.session_totals("codex", home_dir))
        assert totals["output_tokens"] == 10 * len(found.requests(f"a{index + 1}"))


POST = {"new_channel_name": "plan", "text": "a1 takes the parser.", "agents_to_notify": ["a2"]}


@needs_board_codex
def test_two_codex_processes_share_one_board(team):
    """Direct tool calls (a model without code mode): a1 posts while a2's turn runs; a2 is notified in that
    turn and reads the post from its own process."""
    found = team("board", "gpt-mock", {
        "a1": [{"type": "call", "namespace": "collaboration", "name": "post", "arguments": POST,
                "after": {"agent": "a2", "requests": 1}},
               {"type": "text", "text": "done"}],
        "a2": [{"type": "shell", "command": "sleep 6"},
               {"type": "call", "namespace": "collaboration", "name": "search_posts", "arguments": {}},
               {"type": "text", "text": "done"}]})
    a1, a2 = found.requests("a1"), found.requests("a2")
    names = set(tools_of(a1[0]))
    assert {f"collaboration.{name}" for name in ("post", "read_thread", "subscribe", "search_posts", "get_channels",
                                                 "list_threads", "read_post", "unsubscribe", "create_channel")} <= names
    assert "On the team message board you are `/root/a2`." in developer_texts(a2[0])
    assert not any("/root/a1`." in text for text in developer_texts(a2[0]))
    assert '"author":"/root/a1"' in outputs(a1[1])[0]
    received = notices(a2[1])
    assert len(received) == 1 and received[0]["author"] == "/root/a1"
    assert "Sender: /root/a1" in json.dumps(received) and "a1 takes the parser." in json.dumps(received)
    assert "a1 takes the parser." in " ".join(outputs(a2[2]))
    assert_board_record(found)


@needs_board_codex
def test_code_mode_agents_use_the_board_from_exec(team):
    """gpt-6.1-sol, as in the runs: the board's tools sit in the code-mode exec tool (tools.collaboration__post);
    a2 waits for mail with Codex's own wait_agent, and the notice ends the wait."""
    found = team("board", "gpt-6.1-sol", {
        "a1": [{"type": "js", "code": f"text(JSON.stringify(await tools.collaboration__post({json.dumps(POST)})));",
                "after": {"agent": "a2", "requests": 1}},
               {"type": "text", "text": "done"}],
        "a2": [{"type": "call", "namespace": "collaboration", "name": "wait_agent", "arguments": {"timeout_ms": 60000}},
               {"type": "js", "code": "text(JSON.stringify(await tools.collaboration__search_posts({})));"},
               {"type": "text", "text": "done"}]})
    a1, a2 = found.requests("a1"), found.requests("a2")
    assert "declare const tools: { collaboration__post(args:" in tools_of(a1[0])["functions.exec"]["description"]
    posted = json.dumps(outputs(a1[1]))
    assert "Script completed" in posted and "/root/a1" in posted and "message_id" in posted
    received = notices(a2[1])
    assert len(received) == 1 and "a1 takes the parser." in json.dumps(received)
    assert "a1 takes the parser." in " ".join(outputs(a2[2]))
    assert_board_record(found)


@needs_board_codex
def test_the_board_adds_only_the_board_to_what_the_model_sees(team):
    """gpt-6.1-sol on 0.162.1: against antiherding_matched, a board protocol's request has the same instructions,
    the same tools (the board's are nested in exec's description) and the same developer messages but the
    agent's path on the board; the prompt is the protocol's."""
    base = team("antiherding_matched", "gpt-6.1-sol", {"?": []}).requests("?")[0]
    for protocol in ("board",):
        other = team(protocol, "gpt-6.1-sol", {"a1": [], "a2": []}).requests("a1")[0]
        assert other["body"].get("instructions") == base["body"].get("instructions")
        old, new = tools_of(base), tools_of(other)
        assert set(old) == set(new) and {name for name in old if old[name] != new[name]} == {"functions.exec"}
        assert {"collaboration.spawn_agent", "collaboration.send_message", "collaboration.followup_task"} <= set(new)
        added = [text for text in developer_texts(other) if text not in developer_texts(base)]
        assert added == ["On the team message board you are `/root/a1`."]
        assert [text for text in developer_texts(base) if text not in developer_texts(other)] == []
        description = new["functions.exec"]["description"]
        assert description.startswith(old["functions.exec"]["description"].split("### `collaboration__")[0][:200])
        for name in ("post", "get_channels", "list_threads", "search_posts", "read_thread", "read_post", "subscribe",
                     "unsubscribe", "create_channel"):
            assert f"collaboration__{name}" in description and f"collaboration__{name}" not in old["functions.exec"]["description"]
