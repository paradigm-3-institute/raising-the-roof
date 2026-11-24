"""The swarmchat protocol: upstream SwarmChat (vendor/swarmchat, unmodified) run by swarmchat_glue.py, agent.py,
multiagent_protocols/swarmchat.py (docs/setups.md).

- The vendored copy is upstream's commit, byte for byte (vendor/SWARMCHAT-VENDORED.md).
- The glue as real processes on this machine, with upstream's own `swarmchat` command and hook.
- A fake mixed team (no CLI, no model): a Claude-style and a Codex-style participant that call the hook as their
  CLIs would after each tool call, one of them paused in the middle as lockstep pauses it, then the chat killed and
  started again (a revived trial). Nothing lost, nothing shown twice.
- What agent.py renders: each CLI's flags, config and configure lines.
- run.py's refusals and version gates.
- In a real task image (skipped without Docker or the image): agent.py's own install and start commands, a non-root
  agent's command and hook, the chat killed and started again by agent.py's supervisor, and the stop.
- End to end, skipped without the pinned releases: a mixed team of the real Claude Code 2.1.289 and Codex 0.162.1,
  each on a local mock of its model API (tests/limit_mock_api.py: no network, no model), with the keeper.
"""
from __future__ import annotations

import asyncio
import hashlib
import importlib
import importlib.util
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from cli_render import BASE_ENV, INSTRUCTION, ROOT, FakeEnvironment

sys.path.insert(0, str(ROOT))
from multiagent_protocols import board, swarmchat  # noqa: E402

VENDOR = ROOT / "vendor/swarmchat"
GLUE = ROOT / "swarmchat_glue.py"
FULL = {"RELEASE_LIMIT_RESUME": "1", "RELEASE_LIMIT_WAIT_MAX": "172800", "RELEASE_AGENT_TIMEOUT": "3600",
        "RELEASE_HARNESS_GUARD": "1", "RELEASE_TOKEN_BUDGET": "100000", "RELEASE_RESOURCES_TOOL": "1"}
LOCKSTEP = {"RELEASE_LOCKSTEP_GAP": "250000", "RELEASE_LOCKSTEP_REF": "slowest", "RELEASE_LOCKSTEP_STALL": "3600",
            "RELEASE_TEAM_LIMIT_HOLD": "1"}
MIXED = [{"cli": "claude", "model": "claude-sonnet-5-5", "effort": "high"},
         {"cli": "codex", "model": "gpt-6.1-sol", "effort": "high"}]
CLAUDE_BINARY = ROOT / ".cache/claude-2.1.289/claude"
# The patched build the mixed teams' Codex agents ran (docs/codex-patch.md).
CODEX_BINARY = ROOT / ".cache/codex-0.162.1-fix948/bin/codex"
# Upstream's code writes no bytecode into vendor/ (its hashes are checked).
NO_BYTECODE = {"PYTHONDONTWRITEBYTECODE": "1"}


def wait_for(test, seconds: float = 20, what: str = "condition"):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        found = test()
        if found:
            return found
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


# ---- the vendored copy ---------------------------------------------------------------------------------------

def test_the_vendored_copy_is_upstream_byte_for_byte():
    listed = {}
    text = (ROOT / "vendor/SWARMCHAT-VENDORED.md").read_text()
    assert "5ee0c0e307e7b58eb6aa881ee9a7b60d1ace5149" in text
    for line in text.split("```\n")[1].splitlines():
        digest, path = line.split("  ", 1)
        listed[path] = digest
    found = {str(path.relative_to(VENDOR)): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in VENDOR.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    assert found == listed
    assert "MIT License" in (VENDOR / "LICENSE").read_text()


# ---- the prompt ----------------------------------------------------------------------------------------------

def test_the_prompt_is_the_board_prompt_with_upstreams_chat_paragraph():
    with mock.patch.dict(os.environ, {"MULTIAGENT_N_AGENT": "3"}):
        from multiagent_protocols.base import ProtocolPaths, build_protocol
        spec = build_protocol("multiagent_protocols.swarmchat:MultiAgentProtocol", "task", ProtocolPaths.default(),
                              "m")
        peers = build_protocol("multiagent_protocols.claude_peers:MultiAgentProtocol", "task",
                               ProtocolPaths.default(), "m")
    prompt = spec.agents[0].prompt
    assert [agent.id for agent in spec.agents] == ["a1", "a2", "a3"]
    assert all(agent.prompt == prompt for agent in spec.agents)
    ids = ["a1", "a2", "a3"]
    assert prompt == board.BODY.format(n=3, messaging=board.messaging(ids)).replace(
        board.board_paragraph(ids), swarmchat.CHAT.format(messaging=swarmchat.messaging(ids)))
    head, tail = board.BODY.split(board.BOARD)
    assert prompt.startswith(head.format(n=3)) and prompt.endswith(tail)
    assert peers.agents[0].prompt.startswith(head.format(n=3))
    # Upstream's prompts/chat.md, word for word.
    upstream = " ".join((VENDOR / "extras/programbench/prompts/chat.md").read_text().split())
    assert upstream.startswith("Talk to the others with the `swarmchat` command. All of you are in the #general")
    assert f"Your teammates are a1, a2, a3. {upstream}\n" in prompt
    assert spec.metadata["swarmchat"] == {"members": ids, "channels": {"general": ids}}
    assert spec.metadata["role_intro"] == ""
    with mock.patch.dict(os.environ, {"MULTIAGENT_N_AGENT": "1"}), pytest.raises(ValueError, match="needs a team"):
        swarmchat.MultiAgentProtocol().build("t", ProtocolPaths.default(), "m")


# ---- the glue, as real processes ------------------------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as found:
        found.bind(("127.0.0.1", 0))
        return found.getsockname()[1]


class Chat:
    """swarmchat_glue.py run, as agent.py starts it in the container, with the vendored package."""

    def __init__(self, tmp: Path, agents=("a1", "a2")):
        self.tmp, self.agents, self.port = tmp, list(agents), free_port()
        self.agents_dir, self.data = tmp / "swarmchat-agents", tmp / "swarmchat"
        self.home = tmp / "home"
        self.home.mkdir(parents=True, exist_ok=True)
        self.base = {"PATH": os.environ["PATH"], "HOME": str(self.home), "TTC_SWARMCHAT_LIB": str(VENDOR),
                     **NO_BYTECODE}
        self.process = None
        self.start()

    def glue(self, *argv: str, **kwargs):
        return subprocess.run([sys.executable, str(GLUE), *argv], env=self.base, capture_output=True, text=True,
                              timeout=60, **kwargs)

    def start(self) -> None:
        for stale in ("ready", "run.pid"):  # as agent.py's install step
            (self.agents_dir / stale).unlink(missing_ok=True)
        channels = json.dumps({"general": self.agents})
        self.process = subprocess.Popen(
            [sys.executable, str(GLUE), "run", "--agents-dir", str(self.agents_dir), "--data", str(self.data),
             "--port", str(self.port), "--agents", ",".join(self.agents), "--channels", channels, "--", "--fsync"],
            env=self.base, stdout=open(self.tmp / "service.log", "a"), stderr=subprocess.STDOUT)
        assert self.glue("ready", "--agents-dir", str(self.agents_dir), "--timeout", "30").returncode == 0

    def token(self, agent: str) -> str:
        return (self.agents_dir / agent / "token").read_text().strip()

    def env(self, agent: str, token: str | None = None) -> dict:
        """An agent's environment, as its configure step makes it."""
        return {**self.base, "PYTHONPATH": str(VENDOR), "SWARMCHAT_SERVER": f"http://127.0.0.1:{self.port}",
                "SWARMCHAT_TOKEN": token or self.token(agent),
                "TTC_SWARMCHAT_NOTICES": str(self.agents_dir / agent / "notices")}

    def cli(self, agent: str, *argv: str, env: dict | None = None):
        """Upstream's `swarmchat` command, as the wrapper agent.py installs runs it."""
        return subprocess.run([sys.executable, "-m", "swarmchat", *argv], capture_output=True, text=True,
                              env=env or self.env(agent), timeout=60)

    def notices(self, agent: str) -> list[str]:
        return sorted(p.name for p in (self.agents_dir / agent / "notices").glob("*.json"))

    def hook(self, agent: str, event: dict, env: dict | None = None) -> dict | None:
        done = subprocess.run([sys.executable, str(GLUE), "hook"], input=json.dumps(event), capture_output=True,
                              text=True, env=env or self.env(agent), timeout=60)
        assert done.returncode == 0 and not done.stderr, done.stderr
        return json.loads(done.stdout) if done.stdout.strip() else None

    def kill(self) -> None:
        self.process.kill()
        self.process.wait(timeout=10)

    def stop(self) -> None:
        assert self.glue("stop", "--agents-dir", str(self.agents_dir)).returncode == 0
        self.process.wait(timeout=10)

    def posts(self) -> list[tuple[str, str, str]]:
        """Every message in upstream's chat logs: (id, from, text)."""
        found = []
        for log in sorted(self.data.glob("chats/*/log/*.jsonl")):
            chat = log.parent.parent.name
            for line in log.read_text().splitlines():
                entry = json.loads(line)
                if entry["kind"] == "message":
                    found.append((f"{chat}:{entry['seq']}", entry["from"], entry["text"]))
        return found


@pytest.fixture
def chat(tmp_path):
    found = Chat(tmp_path)
    yield found
    if found.process.poll() is None:
        found.kill()


def test_the_glue_makes_the_team_on_upstreams_server(chat):
    # One user per agent, each token in its own file; #general with everyone; the admin token for root only.
    assert chat.cli("a1", "whoami", "--json").returncode == 0
    assert oct((chat.agents_dir / "a1" / "token").stat().st_mode & 0o777) == "0o644"
    assert oct(chat.data.stat().st_mode & 0o777) == "0o700"
    assert oct((chat.data / "admin.token").stat().st_mode & 0o777) == "0o600"
    assert oct((chat.agents_dir / "a2" / "notices").stat().st_mode & 0o777) == "0o777"
    sent = chat.cli("a1", "send", "#general", "hello @a2")
    assert sent.returncode == 0, sent.stderr
    assert "#general" in chat.cli("a2", "chats").stdout
    assert "a1: hello @a2" in chat.cli("a2", "read", "--peek").stdout
    # Upstream's notifier wrote it as a notice for a2 (peek: still unread), none for its sender.
    wait_for(lambda: chat.notices("a2"), what="a2's notice")
    assert chat.notices("a1") == []
    assert "a1: hello @a2" in chat.cli("a2", "read").stdout
    assert chat.cli("a2", "read").stdout.strip() == "No new messages."
    # Upstream's exit codes: not allowed 3, unreachable 5.
    assert chat.cli("a1", "read", env=chat.env("a1", token="forged")).returncode == 3
    chat.stop()
    assert chat.cli("a1", "read").returncode == 5


def test_the_hook_is_upstreams_and_never_fails_a_tool(chat):
    event = {"session_id": "thread-1", "turn_id": "t1", "hook_event_name": "PostToolUse", "tool_name": "Bash"}
    assert chat.hook("a2", event) is None
    chat.cli("a1", "send", "#general", "found a faster parser")
    message_id = chat.posts()[0][0]
    wait_for(lambda: chat.notices("a2"), what="a2's notice")
    output = chat.hook("a2", event)["hookSpecificOutput"]
    assert output == {"hookEventName": "PostToolUse",
                      "additionalContext": f"[swarmchat] #general, from a1 ({message_id}):\nfound a faster parser"}
    assert chat.hook("a2", event) is None and chat.notices("a2") == []
    shown = [json.loads(line) for line in (chat.agents_dir / "a2" / "shown.jsonl").read_text().splitlines()]
    assert [(s["ids"], s["session"], s["turn"], s["tool"]) for s in shown] == [([message_id], "thread-1", "t1", "Bash")]
    # Upstream's cut at 2,000 characters.
    chat.cli("a1", "send", "#general", "x" * 2500)
    wait_for(lambda: chat.notices("a2"), what="the long notice")
    text = chat.hook("a2", event)["hookSpecificOutput"]["additionalContext"]
    assert text.endswith("x" * 2000 + " [cut here: `swarmchat read` shows all of it]")
    # Bad input, no notice folder, no server: nothing printed, exit 0.
    assert chat.hook("a2", "not json") is None
    done = subprocess.run([sys.executable, str(GLUE), "hook"], input="{}", capture_output=True, text=True,
                          env={**chat.base, "TTC_SWARMCHAT_NOTICES": str(chat.tmp / "missing")})
    assert done.returncode == 0 and done.stdout == ""
    done = subprocess.run([sys.executable, str(GLUE), "hook"], input="{}", capture_output=True, text=True,
                          env=chat.base)
    assert done.returncode == 0 and done.stdout == ""


class Participant:
    """An agent as its CLI drives the chat: each tool call runs the hook with that CLI's PostToolUse input, and what
    the hook prints joins its conversation. PAUSED: a lockstep pause (lockstep.PAUSE in its session directory): its
    CLI is stopped, so no tool runs and no hook either."""

    EVENTS = {"claude": lambda n: {"session_id": "claude-session", "hook_event_name": "PostToolUse",
                                   "tool_name": "Bash", "tool_use_id": f"toolu_{n}"},
              "codex": lambda n: {"session_id": "codex-thread", "turn_id": f"turn-{n}",
                                  "hook_event_name": "PostToolUse", "tool_name": "Bash"}}

    def __init__(self, chat: Chat, agent: str, cli: str, session_dir: Path):
        self.chat, self.agent, self.cli, self.session_dir = chat, agent, cli, session_dir
        session_dir.mkdir(parents=True, exist_ok=True)
        self.conversation: list[str] = []
        self.calls = 0

    def paused(self) -> bool:
        import lockstep
        return (self.session_dir / lockstep.PAUSE).exists()

    def tool_call(self, command: str | None = None) -> None:
        assert not self.paused(), "a paused agent runs no tool"
        self.calls += 1
        if command:
            done = self.chat.cli(self.agent, *shlex.split(command))
            assert done.returncode == 0, done.stderr
        found = self.chat.hook(self.agent, self.EVENTS[self.cli](self.calls))
        if found:
            self.conversation.append(found["hookSpecificOutput"]["additionalContext"])

    def shown(self) -> list[str]:
        return [line for text in self.conversation for line in text.split("\n") if line and not line.startswith("[")]


def test_a_fake_mixed_team_through_a_lockstep_pause_and_a_revival(tmp_path):
    import lockstep
    chat = Chat(tmp_path)
    settle = lambda agent, n: wait_for(lambda: len(chat.notices(agent)) >= n, what=f"{n} notices for {agent}")  # noqa
    try:
        claude = Participant(chat, "a1", "claude", tmp_path / "sessions-0-a1")
        codex = Participant(chat, "a2", "codex", tmp_path / "sessions-1-a2")
        claude.tool_call("send '#general' 'plan: I take the parser'")
        settle("a2", 1)
        codex.tool_call()
        assert codex.shown() == ["plan: I take the parser"]
        codex.tool_call("send '#general' 'ok, I take the emitter @a1'")
        settle("a1", 1)
        claude.tool_call()
        assert claude.shown()[-1] == "ok, I take the emitter @a1"
        # a2 (Sol) gets ahead: lockstep pauses it. Its CLI is stopped between responses; a1 keeps posting.
        (codex.session_dir / lockstep.PAUSE).write_text(json.dumps({"since": time.time(), "heartbeat": time.time()}))
        for index in range(10):
            claude.tool_call(f"send '#general' 'progress {index}'")
        with pytest.raises(AssertionError, match="no tool"):
            codex.tool_call()
        settle("a2", 10)
        unread = json.loads(chat.cli("a2", "chats", "--json").stdout.splitlines()[0])["unread"]
        assert unread == 11
        # The pause ends; the runner resumes the turn and its first tool call shows what waited, oldest first, 8 at
        # a time (upstream's hook).
        (codex.session_dir / lockstep.PAUSE).unlink()
        codex.tool_call()
        assert codex.shown()[1:] == [f"progress {index}" for index in range(8)]
        codex.tool_call()
        assert codex.shown()[-2:] == ["progress 8", "progress 9"]
        codex.tool_call()
        assert len(codex.shown()) == 11  # nothing twice
        # The host dies: the chat is killed mid-run, one notice not yet taken. The revived trial starts the glue
        # again on the same folders: the same tokens, the notifier from its saved positions.
        claude.tool_call("send '#general' 'last before the crash'")
        settle("a2", 1)
        tokens = {agent: chat.token(agent) for agent in ("a1", "a2")}
        chat.kill()
        assert chat.cli("a1", "send", "#general", "while down").returncode == 5  # upstream: unreachable
        chat.start()
        assert {agent: chat.token(agent) for agent in ("a1", "a2")} == tokens
        claude.tool_call("send '#general' 'after the revival'")
        settle("a2", 2)
        time.sleep(1)  # a notifier that started from the wrong place would write more by now
        assert len(chat.notices("a2")) == 2
        codex.tool_call()
        assert codex.shown()[-2:] == ["last before the crash", "after the revival"] and len(codex.shown()) == 13
        codex.tool_call()
        assert len(codex.shown()) == 13
        # Read positions survived: a2 has read nothing, so everything is still unread.
        unread = chat.cli("a2", "read", "--json").stdout.splitlines()
        assert len(unread) == 13 and json.loads(unread[0])["text"] == "plan: I take the parser"
        assert chat.cli("a2", "read").stdout.strip() == "No new messages."
        shown = [json.loads(line) for line in (chat.agents_dir / "a2" / "shown.jsonl").read_text().splitlines()]
        ids = [message for entry in shown for message in entry["ids"]]
        assert len(ids) == len(set(ids)) == 13
        # A lost token file: the glue rotates that agent's token with upstream's admin call; the old one stops.
        old = chat.token("a1")
        chat.kill()
        (chat.agents_dir / "a1" / "token").unlink()
        chat.start()
        assert chat.token("a1") != old and chat.token("a2") == tokens["a2"]
        assert chat.cli("a1", "read", env=chat.env("a1", token=old)).returncode == 3
        assert chat.cli("a1", "whoami").returncode == 0
        chat.stop()
    finally:
        if chat.process.poll() is None:
            chat.kill()


# ---- agent.py --------------------------------------------------------------------------------------------------

def load_module(path: Path, name: str):
    from importlib.machinery import SourceFileLoader
    spec = importlib.util.spec_from_file_location(name, path, loader=SourceFileLoader(name, str(path)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def team_env(protocol: str = "swarmchat", agents: int = 2, extra: dict | None = None) -> dict:
    return {**BASE_ENV, "RELEASE_TASK": "ale-bench", "MULTIAGENT_N_AGENT": str(agents), 
            "MULTIAGENT_PROTOCOL": f"multiagent_protocols.{protocol}:MultiAgentProtocol", "MULTIAGENT_AGENT_EFFORT": "high",
            "RELEASE_CONTINUATION": (ROOT / "benchmarks/ale-bench/continuation.txt").read_text().rstrip("\n"),
            **(extra or {})}


RUN = "exec /usr/lib/hsv/hsci /usr/lib/hsv/hscd run --agents-dir /logs/agent/swarmchat-agents "
STOP = "/usr/lib/hsv/hsci /usr/lib/hsv/hscd stop --agents-dir /logs/agent/swarmchat-agents"


class ChatFake(FakeEnvironment):
    """FakeEnvironment whose chat runs until it is stopped (test_message_board.py's BoardFake)."""

    def __init__(self) -> None:
        super().__init__()
        self.stopped = asyncio.Event()

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        await asyncio.sleep(0.01)  # a real exec waits for the container, so other tasks run meanwhile
        result = await super().exec(command, user=user, env=env, cwd=cwd, timeout_sec=timeout_sec)
        if command.startswith(RUN):
            await self.stopped.wait()
        elif command == STOP:
            self.stopped.set()
        return result


def render(tmp_path: Path, cls: str = "MixedTeamAgent", protocol: str = "swarmchat", agents: int = 2,
           extra: dict | None = None, revive: dict | None = None, module_path: Path | None = None, **kwargs):
    """The commands agent.py (or MODULE_PATH, another copy of it) sends to a fake container for this team."""
    env = team_env(protocol, agents, extra)
    if cls == "MixedTeamAgent":
        kwargs.setdefault("team", MIXED)
        kwargs.setdefault("model_name", "claude-sonnet-5-5+gpt-6.1-sol")
    kwargs.setdefault("model_name", "claude-sonnet-5-5" if "Claude" in cls else "gpt-6.1-sol")
    with mock.patch.dict(os.environ, env):
        sys.modules.pop("agent", None)
        module = (importlib.import_module("agent") if module_path is None
                  else load_module(module_path, "agent_before_swarmchat"))
    with mock.patch.dict(os.environ, env), mock.patch("secrets.token_urlsafe", lambda n: "s" * 32), \
            mock.patch("time.time", lambda: 1700000000.0):
        agent = getattr(module, cls)(logs_dir=tmp_path, **kwargs)
        environment = ChatFake() if protocol == "swarmchat" else FakeEnvironment()

        async def go() -> None:
            await agent.install(environment)
            if revive is None:
                await agent.run(INSTRUCTION, environment, SimpleNamespace())
            else:
                agent.revive_plan = json.loads(json.dumps(revive))
                await agent.resume(INSTRUCTION, environment, SimpleNamespace())

        asyncio.run(go())
    return module, agent, environment.calls


def runners(agent) -> dict[int, str]:
    parts = agent.last_runner_script.split("# ---- /usr/lib/hsv/runner-")
    if len(parts) == 1:
        return {0: agent.last_runner_script}
    return {int(part.split(" ", 1)[0]): part.split("----\n", 1)[1] for part in parts[1:]}


def invoke(runner: str, cli: str) -> str:
    start = runner.index(f"invoke_{cli}() {{")
    return runner[start:runner.index("\n  }\n", start)]


def configure(runner: str, cli: str) -> str:
    start = runner.index(f"configure_{cli}_command() {{")
    return runner[start:runner.index("\n}\n", start)]


@pytest.mark.parametrize("extra", [FULL, {**FULL, **LOCKSTEP}], ids=["full", "lockstep"])
def test_a_mixed_team_on_the_chat(tmp_path, extra):
    module, agent, calls = render(tmp_path, extra=extra, claude_compact_window=277000)
    found = runners(agent)
    claude_runner, codex_runner = found[0], found[1]
    # Claude Code: the hook and its name; nothing else in its command line changes.
    claude_invoke = invoke(claude_runner, "claude")
    assert claude_invoke.count(module.swarmchat_claude_flags()) == 1
    _, plain, _ = render(tmp_path / "plain", protocol="antiherding_matched", extra=extra, claude_compact_window=277000)
    plain_runners = runners(plain)
    assert claude_invoke.replace(f"      {module.swarmchat_claude_flags()} \\\n", "") == invoke(plain_runners[0], "claude")
    assert configure(claude_runner, "claude").count(module.swarmchat_configure()) == 1
    assert configure(claude_runner, "claude").replace(module.swarmchat_configure(), "") == configure(
        plain_runners[0], "claude")
    # Codex: the flag that trusts the hook, hooks.json and its name in config.toml; nothing else.
    codex_invoke = invoke(codex_runner, "codex")
    assert "--dangerously-bypass-hook-trust " in codex_invoke
    assert codex_invoke.replace("--dangerously-bypass-hook-trust ", "") == invoke(plain_runners[1], "codex")
    codex_configure = configure(codex_runner, "codex")
    assert module.SWARMCHAT_CODEX_LINES in codex_configure
    assert 'developer_instructions = "On the team chat you are `@swarmchat_agent@`."\n' in codex_configure
    assert codex_configure.replace(module.SWARMCHAT_CODEX_LINES, "").replace(
        'developer_instructions = "On the team chat you are `@swarmchat_agent@`."\n', "") == configure(
        plain_runners[1], "codex")
    # Everything outside the two functions is the same runner (lockstep's pause, the hold, the keeper's parts), but
    # for the first prompt's opening line, which board-style protocols leave empty (role_intro).
    intro = "printf %s " + shlex.quote("You are one agent of many concurrent agents solving a task together. "
                                       "Follow this coordination protocol exactly.\n\n")
    for index in (0, 1):
        cli = ("claude", "codex")[index]
        strip = lambda text: text.replace(invoke(text, cli), "").replace(configure(text, cli), "")  # noqa: E731
        assert strip(found[index]) == strip(plain_runners[index]).replace(intro, "printf %s ''")
    # The chat: installed and started as root, with the team, before the agents; stopped after. No secret in any
    # command's environment (each agent's token is in its file).
    commands = [call["command"] for call in calls]
    install = next(index for index, c in enumerate(commands) if "/usr/lib/hsv/hsc.tgz.b64" in c)
    wrapper = next(index for index, c in enumerate(commands) if "> /usr/local/bin/swarmchat" in c)
    serve = next(index for index, c in enumerate(commands) if c.startswith(RUN))
    keeper = next(index for index, c in enumerate(commands) if "/usr/lib/hsv/runner-" in c)
    stop = next(index for index, c in enumerate(commands) if c == STOP)
    assert install < wrapper < serve < keeper < stop
    assert calls[serve]["user"] == "root" and calls[serve]["env"] == {}
    assert ("--data /logs/agent/swarmchat --port 39614 --agents a1,a2 --channels "
            "'{\"general\":[\"a1\",\"a2\"]}' -- --fsync >> /logs/agent/swarmchat-service.log 2>&1") in commands[serve]
    assert shlex.quote(module.SWARMCHAT_WRAPPER) in commands[wrapper] and calls[wrapper]["user"] == "root"
    assert all("SECRET" not in json.dumps(call["env"]) for call in calls)
    assert "Talk to the others with the `swarmchat` command" in "\n".join(commands)


def test_the_install_unpacks_upstreams_files_exactly(tmp_path):
    """agent.py's install commands, run here with the container's paths moved under TMP: upstream's files land
    byte for byte, each command fits in one argument, and the wrapper runs upstream's CLI."""
    module, _, calls = render(tmp_path / "render", extra=FULL)
    root = tmp_path / "c"
    swaps = (("/usr/lib/hsv", str(root / "hsv")), ("/usr/local/bin", str(root / "bin")),
             ("/logs/agent", str(root / "agent")))
    patch = lambda text: [text := text.replace(old, new) for old, new in swaps][-1]  # noqa: E731
    commands = [c["command"] for c in calls]
    first = next(index for index, c in enumerate(commands) if "/usr/lib/hsv/hsc.tgz.b64" in c)
    last = next(index for index, c in enumerate(commands) if "> /usr/local/bin/swarmchat" in c)
    install = commands[first:last + 1]
    assert len(install) == 4 and all(len(command) < 128 * 1024 for command in install)
    for command in install:
        done = subprocess.run(["bash", "-c", patch(command)], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
    for name in module.SWARMCHAT_FILES:
        assert (root / "hsv/hsc" / name).read_bytes() == (VENDOR / name).read_bytes()
    assert sorted(str(p.relative_to(root / "hsv/hsc")) for p in (root / "hsv/hsc").rglob("*") if p.is_file()) == \
        sorted(module.SWARMCHAT_FILES)
    assert (root / "hsv/hscd").read_text() == GLUE.read_text()
    wrapper = root / "bin/swarmchat"
    assert wrapper.read_text() == patch(module.SWARMCHAT_WRAPPER)
    done = subprocess.run([str(wrapper), "--version"], capture_output=True, text=True, env={**os.environ, **NO_BYTECODE})
    assert done.stdout.strip() == "swarmchat 0.1.0"
    # Again (a revival in the same container): the wrapper it wrote is taken as its own; an image's own is not.
    for command in install:
        assert subprocess.run(["bash", "-c", patch(command)], capture_output=True).returncode == 0
    wrapper.write_text("#!/bin/sh\necho the image's own\n")
    for command in install[:-1]:
        assert subprocess.run(["bash", "-c", patch(command)], capture_output=True).returncode == 0
    last = subprocess.run(["bash", "-c", patch(install[-1])], capture_output=True, text=True)
    assert last.returncode == 1 and "already exists in this image" in last.stderr


def test_one_cli_teams_on_the_chat(tmp_path):
    for cls, cli in (("ClaudeCodeCollaborationAgent", "claude"), ("CodexCollaborationAgent", "codex")):
        module, agent, calls = render(tmp_path / cli, cls=cls, extra=FULL)
        runner = runners(agent)[0]
        assert "/usr/lib/hsv/hscd hook" in runner and module.swarmchat_configure() in runner
        assert any(call["command"].startswith(RUN) for call in calls)


def test_the_configure_lines_read_the_agents_own_token(tmp_path):
    module = importlib.import_module("agent")
    agents = tmp_path / "swarmchat-agents"
    for agent_id in ("a1", "a2"):
        (agents / agent_id).mkdir(parents=True)
        (agents / agent_id / "token").write_text(f"token-of-{agent_id}\n")
    lines = module.swarmchat_configure().replace("/logs/agent", str(tmp_path))
    script = (f"configure() {{\n{lines}}}\nagent_id=a2\nconfigure || exit 9\n"
              'printf "%s|%s|%s" "$SWARMCHAT_TOKEN" "$SWARMCHAT_SERVER" "$TTC_SWARMCHAT_NOTICES"\necho\n'
              'bash -c \'printf "%s\\n" "$SWARMCHAT_TOKEN"\'')
    done = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    token, server, notices = done.stdout.splitlines()[0].split("|")
    assert token == "token-of-a2" and server == "http://127.0.0.1:39614"
    assert notices == f"{agents}/a2/notices" and done.stdout.splitlines()[1] == "token-of-a2"
    missing = subprocess.run(["bash", "-c", f"configure() {{\n{lines}}}\nagent_id=a3\nconfigure"],
                             capture_output=True, text=True)
    assert missing.returncode == 1 and "swarmchat: no token for a3" in missing.stderr


def test_the_flags_reach_the_clis_as_separate_arguments():
    module = importlib.import_module("agent")
    found = subprocess.run(["bash", "-c", f'agent_id=a2; set -- {module.swarmchat_claude_flags()}; printf "%s\\n" "$@"'],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    assert found[0] == "--settings" and found[2:] == ["--append-system-prompt", "On the team chat you are `a2`."]
    assert json.loads(found[1]) == {"hooks": {"PostToolUse": [{"matcher": "*", "hooks": [
        {"type": "command", "command": "python3 /usr/lib/hsv/hscd hook", "timeout": 30}]}]}}
    assert json.loads(module.swarmchat_hooks("codex")) == {"hooks": {"PostToolUse": [{"hooks": [
        {"type": "command", "command": "python3 /usr/lib/hsv/hscd hook", "timeout": 30,
         "additionalContextLimit": 0}]}]}}


def test_a_revived_team_runs_the_same_command_lines_and_the_same_chat(tmp_path):
    plan = {"revival": 1, "note": "Continue.", "layout": "team", "started": 1.0, "checks_passed": True,
            "agents": [{"index": 0, "session_id": "s0", "turn": 2}, {"index": 1, "session_id": "s1", "turn": 3}]}
    module, agent, calls = render(tmp_path, extra=FULL)
    _, revived, revived_calls = render(tmp_path, extra=FULL, revive=plan)
    for index, cli in ((0, "claude"), (1, "codex")):
        assert invoke(runners(revived)[index], cli) == invoke(runners(agent)[index], cli)
        assert configure(runners(revived)[index], cli) == configure(runners(agent)[index], cli)
    chat_commands = lambda found: [c["command"] for c in found if "hsc.tgz" in c["command"] or c["command"] == STOP  # noqa
                                   or c["command"].startswith(RUN) or "> /usr/local/bin/swarmchat" in c["command"]]
    assert chat_commands(revived_calls) == chat_commands(calls)
    # revive.py's dry run (an environment whose every command returns 0 at once) goes through as well.
    import revive
    with mock.patch.dict(os.environ, team_env(extra=FULL)):
        sys.modules.pop("agent", None)
        dry = importlib.import_module("agent").MixedTeamAgent(logs_dir=tmp_path / "dry", team=MIXED,
                                                              model_name="claude-sonnet-5-5+gpt-6.1-sol")
        dry.revive_plan = json.loads(json.dumps(plan))
        recorder = revive.RecordingEnvironment()
        asyncio.run(asyncio.wait_for(dry.resume(INSTRUCTION, recorder, SimpleNamespace()), 120))
    assert STOP in recorder.calls and any("/hscd ready " in command for command in recorder.calls)


def test_one_agent_refuses_the_chat(tmp_path):
    with pytest.raises(ValueError, match="one agent has no team"):
        render(tmp_path, cls="CodexCollaborationAgent", agents=1)
    with pytest.raises(ValueError, match="one agent has no team"):
        render(tmp_path, cls="ClaudeCodeCollaborationAgent", agents=1)


# ---- run.py ----------------------------------------------------------------------------------------------------

def run_py(*argv: str):
    import run
    args = run.parser().parse_args(list(argv))
    args.dry_run = True
    with mock.patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "fake"}):
        return run.prepare(args)


@pytest.mark.parametrize("argv, agent_class", [
    (("--team", "claude:claude-opus-5-5:high,codex:gpt-6.1-sol:high"), "agent:MixedTeamAgent"),
    (("--cli", "claude", "--model", "claude-sonnet-5-5", "--agents", "3"), "agent:ClaudeCodeCollaborationAgent"),
    (("--cli", "codex", "--model", "gpt-6.1-sol", "--agents", "2"), "agent:CodexCollaborationAgent"),
])
def test_run_py_runs_a_team_on_the_chat(argv, agent_class):
    cmd, env, _ = run_py("perf-takehome", *argv, "--protocol", "swarmchat", "--token-budget", "100000",
                         "--lockstep-gap", "250000", "--name", "swarmchat-dry")
    assert env["MULTIAGENT_PROTOCOL"] == "multiagent_protocols.swarmchat:MultiAgentProtocol"
    assert cmd[cmd.index("--agent-import-path") + 1] == agent_class
    assert env["RELEASE_LOCKSTEP_GAP"] == "250000"


@pytest.mark.parametrize("argv, message", [
    (("--cli", "claude", "--model", "claude-sonnet-5-5", "--agents", "1"), "needs a team"),
])
def test_run_py_refuses_it(argv, message):
    with pytest.raises(ValueError, match=message):
        run_py("ale-bench", "--problem", "ahc014", "--protocol", "swarmchat", *argv, "--name", "swarmchat-dry")


def test_run_py_needs_lockstep_for_a_mixed_team():
    # A mixed swarmchat team always spends in lockstep; one-CLI teams may run without it.
    with pytest.raises(ValueError, match="swarmchat with a mixed --team needs --lockstep-gap"):
        run_py("perf-takehome", "--team", "claude:claude-opus-5-5:high,codex:gpt-6.1-sol:high", "--protocol",
               "swarmchat", "--token-budget", "100000", "--name", "swarmchat-dry")
    cmd, env, _ = run_py("perf-takehome", "--cli", "claude", "--model", "claude-sonnet-5-5", "--agents", "2",
                         "--protocol", "swarmchat", "--token-budget", "100000", "--name", "swarmchat-dry")
    assert "RELEASE_LOCKSTEP_GAP" not in env


def test_run_py_checks_the_cli_versions(tmp_path):
    import run
    package = tmp_path / "codex-old"
    package.mkdir()
    (package / "codex-package.json").write_text('{"version": "0.162.0"}')
    with pytest.raises(ValueError, match=r"needs Codex 0\.162\.1 or later"):
        run.codex_version(package, "--protocol swarmchat", run.SWARMCHAT_MIN_CODEX, "its PostToolUse hook", False)
    (package / "codex-package.json").write_text('{"version": "0.163.0"}')
    run.codex_version(package, "--protocol swarmchat", run.SWARMCHAT_MIN_CODEX, "its PostToolUse hook", False)


# ---- in a real task image ---------------------------------------------------------------------------------------

IMAGES = ["ahc014__aagcidy__env-main:latest", "perf-takehome__ftbacpn__env-main:latest"]


def image_present(image: str) -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode == 0


class Docker:
    """A throwaway container of IMAGE (no network) that stands in for the trial's: agent.py's commands run in it
    with `docker exec`, as Harbor runs them."""

    def __init__(self, image: str):
        self.name = f"ttc-swarmchat-test-{uuid.uuid4().hex[:10]}"
        subprocess.run(["docker", "run", "-d", "--rm", "--network", "none", "--name", self.name, "--entrypoint",
                        "sleep", image, "infinity"], check=True, capture_output=True)
        self.calls: list[str] = []

    async def exec(self, command, user=None, env=None, cwd=None, timeout_sec=None):
        self.calls.append(command)
        argv = ["docker", "exec", "-u", str(user or "root")]
        for key, value in (env or {}).items():
            argv += ["-e", f"{key}={value}"]
        process = await asyncio.create_subprocess_exec(*argv, self.name, "bash", "-c", command,
                                                       stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        out, err = await asyncio.wait_for(process.communicate(), timeout_sec or 3600)
        return SimpleNamespace(return_code=process.returncode, stdout=out.decode(), stderr=err.decode())

    def run(self, command: str, user: str = "root", env: dict | None = None, stdin: str | None = None):
        argv = ["docker", "exec", "-i", "-u", user]
        for key, value in (env or {}).items():
            argv += ["-e", f"{key}={value}"]
        return subprocess.run([*argv, self.name, "bash", "-c", command], input=stdin, capture_output=True, text=True,
                              timeout=120)

    def close(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


@pytest.mark.parametrize("image", IMAGES)
def test_in_a_real_task_image(tmp_path, image):
    if not image_present(image):
        pytest.skip(f"{image} is not on this machine")
    with mock.patch.dict(os.environ, team_env(extra=FULL)):
        sys.modules.pop("agent", None)
        module = importlib.import_module("agent")
        agent = module.MixedTeamAgent(logs_dir=tmp_path, team=MIXED, model_name="claude-sonnet-5-5+gpt-6.1-sol")
    box = Docker(image)
    try:
        box.run("mkdir -p /logs/agent && chmod 777 /logs/agent")
        chat_spec = {"members": ["a1", "a2"], "channels": {"general": ["a1", "a2"]}}

        async def scenario():
            task = await agent._swarmchat_start(box, chat_spec)
            # Each agent as a non-root user (nobody), with its configure lines, upstream's command and the hook.
            def as_agent(agent_id: str, command: str, stdin: str | None = None):
                script = (f"configure() {{\n{module.swarmchat_configure()}}}\nagent_id={agent_id}\n"
                          f"configure || exit 9\n{command}")
                return box.run(script, user="nobody", env={"HOME": "/tmp"}, stdin=stdin)
            hook = shlex.quote(json.loads(module.swarmchat_hooks("claude"))["hooks"]["PostToolUse"][0]["hooks"][0]
                               ["command"])
            sent = as_agent("a1", "swarmchat send '#general' 'hello from a1'")
            assert sent.returncode == 0, sent.stderr + sent.stdout
            event = json.dumps({"session_id": "s", "tool_name": "Bash"})
            shown = await asyncio.to_thread(wait_for, lambda: as_agent("a2", f"bash -c {hook}", event).stdout.strip(),
                                            20, "a2's hook output")
            context = json.loads(shown)["hookSpecificOutput"]["additionalContext"]
            assert context.startswith("[swarmchat] #general, from a1 (") and context.endswith("hello from a1")
            assert as_agent("a2", f"bash -c {hook}", event).stdout.strip() == ""
            assert as_agent("a2", "cat /logs/agent/swarmchat/admin.token").returncode != 0
            # A root agent's `kill -9` of the chat: agent.py's supervisor starts it again, with the same tokens.
            box.run("kill -9 $(cat /logs/agent/swarmchat-agents/run.pid)")
            await asyncio.to_thread(
                wait_for, lambda: as_agent("a1", "swarmchat send '#general' 'after the restart'").returncode == 0,
                60, "the restarted chat")
            shown = await asyncio.to_thread(wait_for, lambda: as_agent("a2", f"bash -c {hook}", event).stdout.strip(),
                                            20, "a2's hook output after the restart")
            assert json.loads(shown)["hookSpecificOutput"]["additionalContext"].endswith("after the restart")
            assert "hello from a1" in as_agent("a2", "swarmchat read").stdout
            await agent._swarmchat_stop(box, task)
            assert as_agent("a1", "swarmchat read").returncode == 5

        asyncio.run(scenario())
        python = box.run("python3 --version").stdout.strip()
        assert python.startswith("Python 3.1")
    finally:
        box.close()


# ---- end to end: the real CLIs, each on a mock of its model API -------------------------------------------------

needs_clis = pytest.mark.skipif(not (CLAUDE_BINARY.is_file() and CODEX_BINARY.is_file()),
                                reason="pinned Claude Code 2.1.289 or Codex 0.162.1 missing")


class ChatRun:
    """A Claude Code agent (a1) and a Codex agent (a2) on swarmchat, with the keeper, each CLI on its own mock. The
    container's fixed paths (the port, /usr/local/bin, /usr/lib/hsv, /logs/agent) are moved under TMP."""

    def __init__(self, tmp: Path, claude_plan: list, codex_plan: list, continuations: int = 0):
        from test_harness_guard import GUARD, LIMIT, Local
        from test_limit_resume import Mock

        self.tmp = tmp
        self.mocks = {}
        for cli, plan in (("claude", claude_plan), ("codex", codex_plan)):
            (tmp / f"mock-{cli}").mkdir(parents=True)
            self.mocks[cli] = Mock(tmp / f"mock-{cli}", plan)
        provider = {"name": "mock", "base_url": f"http://127.0.0.1:{self.mocks['codex'].port}/v1",
                    "wire_api": "responses", "env_key": "CODEX_API_KEY", "supports_websockets": "false",
                    "request_max_retries": "0", "stream_max_retries": "0"}
        args = ["-c", 'model_provider="mock"']
        for key, value in provider.items():
            args += ["-c", f"model_providers.mock.{key}={value if value in ('false', '0') else json.dumps(value)}"]
        hook = tmp / "hook.sh"
        hook.write_text('task_cli_args=()\nif [ "$1" = codex ]; then task_cli_args=('
                        + " ".join(f"'{arg}'" for arg in args) + "); fi\n")
        home = tmp / "home"
        home.mkdir()
        self.bin = tmp / "usr-local"
        port = free_port()
        base = {"PATH": f"{self.bin}/bin:/usr/local/bin:/usr/bin:/bin", "HOME": str(home),
                "HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
                "NO_PROXY": "127.0.0.1,localhost", "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{self.mocks['claude'].port}",
                "RELEASE_TASK_CLI_HOOK": str(hook), "META_DUO_CLAUDE_API_RETRY_BASE_SLEEP": "2",
                "META_DUO_CODEX_API_RETRY_BASE_SLEEP": "2"}
        local_bin = str(self.bin)

        class ChatLocal(Local):
            def patch(self, command: str) -> str:
                return (super().patch(command).replace("/usr/local/bin", f"{local_bin}/bin")
                        .replace("39614", str(port)))

        self.local = ChatLocal(tmp, base)
        self.env = {**team_env(extra={**LIMIT, **GUARD, "RELEASE_TOKEN_BUDGET": "4000000"}),
                    "RELEASE_CONTINUATIONS": str(continuations), "MULTIAGENT_AGENT_EFFORT": "",
                    "META_DUO_CONTAINER_CLAUDE_BINARY": str(CLAUDE_BINARY),
                    "META_DUO_CONTAINER_CODEX_BINARY": str(CODEX_BINARY), "CODEX_API_KEY": "sk-mock"}
        self.error = None

        def main() -> None:
            try:
                with mock.patch.dict(os.environ, self.env, clear=False):
                    sys.modules.pop("agent", None)
                    module = importlib.import_module("agent")
                    agent = module.MixedTeamAgent(
                        logs_dir=self.local.agent_dir, model_name="claude-sonnet-5-5+gpt-mock",
                        team=[{"cli": "claude", "model": "claude-sonnet-5-5"}, {"cli": "codex", "model": "gpt-mock"}])
                    asyncio.run(agent.run(INSTRUCTION, self.local, SimpleNamespace()))
            except BaseException as error:  # noqa: BLE001 - the test reads it
                self.error = error

        thread = threading.Thread(target=main, daemon=True)
        thread.start()
        thread.join(300)
        self.ended = not thread.is_alive()

    def requests(self, cli: str) -> list[str]:
        return [json.dumps(request["body"]) for request in self.mocks[cli].main()]

    def posts(self) -> list[tuple[str, str, str]]:
        found = []
        for log in sorted((self.local.agent_dir / "swarmchat").glob("chats/*/log/*.jsonl")):
            for line in log.read_text().splitlines():
                entry = json.loads(line)
                if entry["kind"] == "message":
                    found.append((f"{log.parent.parent.name}:{entry['seq']}", entry["from"], entry["text"]))
        return sorted(found, key=lambda post: post[2])

    def shown(self, agent: str) -> list[dict]:
        path = self.local.agent_dir / "swarmchat-agents" / agent / "shown.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []

    def close(self) -> None:
        import harness_guard
        from test_harness_guard import kill

        for found in self.mocks.values():
            found.stop() if found.process.poll() is None else None
        state = harness_guard.read_json(str(self.local.guard_dir / "state" / "keeper.json")) or {}
        if state.get("pid") and harness_guard.alive(state["pid"], state.get("start")):
            kill(state["pid"])
        for index, (env_name, home) in enumerate((("CLAUDE_CONFIG_DIR", "claude-home"), ("CODEX_HOME", "codex-home"))):
            for session in self.local.agent_dir.glob(f"sessions-{index}-*"):
                kill(harness_guard.agent_processes(env_name, str(session / home)))
        pid = self.local.agent_dir / "swarmchat-agents" / "run.pid"
        if pid.is_file():
            try:
                os.kill(int(pid.read_text()), signal.SIGKILL)
            except (OSError, ValueError):
                pass


@needs_clis
def test_a_real_claude_and_codex_team_talk_through_the_chat(tmp_path):
    """a1 (Claude Code) posts in #general; a2 (Codex), in the middle of a command then, is shown the message after
    that command (its PostToolUse hook) and answers a1 directly; a1 is shown the answer after its next command.
    Each agent's own name reaches its model; upstream's logs have every message, shown.jsonl every push."""
    tool = lambda command, delay=0: {"type": "tool", "command": command, **({"delay": delay} if delay else {})}  # noqa
    run = ChatRun(tmp_path,
                  claude_plan=[tool("swarmchat send '#general' 'a1 here: I take the parser'"), tool("sleep 14"),
                               tool("echo after"), {"type": "text", "text": "done-a1"}],
                  codex_plan=[tool("sleep 8"), tool("swarmchat send a1 'a2 here: emitter is mine'"),
                              {"type": "text", "text": "done-a2"}])
    try:
        assert run.ended and run.error is None, repr(run.error)
        codex, claude = run.requests("codex"), run.requests("claude")
        # Codex: its name in the developer message; the post after `sleep 8`, not before.
        assert "On the team chat you are `a2`." in codex[0]
        assert "a1 here: I take the parser" not in codex[0]
        assert "[swarmchat] #general, from a1" in codex[1] and "a1 here: I take the parser" in codex[1]
        # Claude Code: its name in the system prompt; a2's direct message after its `sleep 14`.
        assert "On the team chat you are `a1`." in claude[0]
        assert any("[swarmchat] a2, from a2" in body and "a2 here: emitter is mine" in body for body in claude[2:])
        posts = run.posts()
        assert [(sender, text) for _, sender, text in posts] == [("a1", "a1 here: I take the parser"),
                                                                 ("a2", "a2 here: emitter is mine")]
        assert [entry["ids"] for entry in run.shown("a2")] == [[posts[0][0]]]
        assert [entry["ids"] for entry in run.shown("a1")] == [[posts[1][0]]]
        # The chat was stopped after the agents.
        assert not (tmp_path / "agent/swarmchat-agents/run.pid").is_file() or \
            subprocess.run(["kill", "-0", (tmp_path / "agent/swarmchat-agents/run.pid").read_text()]).returncode != 0
    finally:
        run.close()


@needs_clis
def test_a_resumed_session_keeps_its_hook(tmp_path):
    """A continuation turn resumes each CLI's session (`claude -p --resume`, `codex exec resume`), as the runner
    resumes a turn after a lockstep pause or a team hold: the hook still runs there. a1 posts while a2's resumed
    session runs a command; a2 is shown the post after that command."""
    tool = lambda command: {"type": "tool", "command": command}  # noqa: E731
    run = ChatRun(tmp_path,
                  claude_plan=[tool("sleep 3"), tool("swarmchat send '#general' 'posted during a2s resumed turn'"),
                               {"type": "text", "text": "done-0"}, tool("sleep 1"), {"type": "text", "text": "done-1"}],
                  codex_plan=[tool("echo first"), {"type": "text", "text": "done-0"}, tool("sleep 8"),
                              {"type": "text", "text": "done-1"}],
                  continuations=1)
    try:
        assert run.ended and run.error is None, repr(run.error)
        codex = run.requests("codex")
        posted = run.posts()[0]
        shown = run.shown("a2")
        assert [entry["ids"] for entry in shown] == [[posted[0]]]
        later = [index for index, body in enumerate(codex) if "posted during a2s resumed turn" in body]
        assert later and later[0] >= 3  # in the continuation turn, after its first tool call there
        attempts = sorted(path.name for path in run.local.agent_dir.glob("sessions-1-a2/codex-attempt-*.jsonl"))
        assert attempts == ["codex-attempt-0.jsonl", "codex-attempt-1.jsonl"]
    finally:
        run.close()
