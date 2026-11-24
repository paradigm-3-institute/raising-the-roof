"""claude_peers: a Claude Code team on Claude Code's own cross-session messaging (multiagent_protocols/claude_peers.py,
agent.py, run.py, claude_peers_journal.py; docs/setups.md).

The prompt is board.py's with its paragraph about the channel rewritten; each agent's `claude -p` line gains
SendMessage and ListAgents, its name, a fixed inbox socket, the setting that takes peers' messages and one appended
system-prompt line with its name; its configure step links the team's session registry and makes the socket
directory. The journal tests read real Claude Code 2.1.289 logs (tests/fixtures/claude_peers: two pairs of agents,
x1/x2 and y1/y2, from a probe run, trimmed of their system-prompt snapshots).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from cli_render import BASE_ENV, INSTRUCTION, ROOT, FakeEnvironment

sys.path.insert(0, str(ROOT))
import claude_peers_journal  # noqa: E402
from multiagent_protocols import board, claude_peers  # noqa: E402
from multiagent_protocols.base import ProtocolPaths, build_protocol  # noqa: E402

FIXTURE = ROOT / "tests/fixtures/claude_peers/agent"
FULL = {"RELEASE_LIMIT_RESUME": "1", "RELEASE_LIMIT_WAIT_MAX": "172800", "RELEASE_AGENT_TIMEOUT": "3600",
        "RELEASE_HARNESS_GUARD": "1", "RELEASE_TOKEN_BUDGET": "100000", "RELEASE_RESOURCES_TOOL": "1"}
PEERS_LINE = ('--name "$agent_id" --messaging-socket-path "/tmp/.claude-peers/$agent_id.sock" '
              "--settings '{\"crossSessionInbound\":\"accept\"}' "
              "--append-system-prompt 'In the team you are `'\"$agent_id\"'`.'")


def build(protocol: str, agents: int):
    with mock.patch.dict(os.environ, {"MULTIAGENT_N_AGENT": str(agents)}):
        return build_protocol(f"multiagent_protocols.{protocol}:MultiAgentProtocol", INSTRUCTION,
                              ProtocolPaths.default("/logs/agent/multiagent-collab"), "claude-sonnet-5-5")


# ---- the prompt --------------------------------------------------------------------------------------------

def test_the_prompt_is_the_board_prompt_with_its_channel_paragraph_rewritten():
    spec = build("claude_peers", 3)
    prompt = spec.agents[0].prompt
    assert [agent.prompt for agent in spec.agents] == [prompt] * 3
    assert [agent.id for agent in spec.agents] == ["a1", "a2", "a3"]
    board_prompt = build("board", 3).agents[0].prompt
    head, tail = board_prompt.split(board.board_paragraph(["a1", "a2", "a3"]))
    assert prompt == head + claude_peers.peers_paragraph(["a1", "a2", "a3"]) + tail
    assert prompt == (
        "You are one of 3 agents working on this task at the same time, in this container, all\n"
        "with this same prompt. You are one team: the team is scored on its best result, so\n"
        "whatever any one of you achieves counts for all of you.\n\n"
        "You can message one another, like direct messages in a team Slack: each message goes\n"
        "to one teammate.\n"
        "Your teammates are a1, a2, a3. Message a teammate with the SendMessage tool; ListAgents shows who is "
        "reachable. Messages from teammates reach you automatically while you work. If you want to wait for a "
        "teammate, run a short sleep.\n\n"
        "How you split the work, what you share, and when you talk is up to you. Work it out\n"
        "among yourselves.\n")
    assert spec.metadata["claude_peers"] == {"names": ["a1", "a2", "a3"]}
    assert spec.metadata["role_intro"] == "" and spec.shared_paths == []
    assert "message_board" not in spec.metadata
    assert not getattr(claude_peers.MultiAgentProtocol, "has_solo", False)
    with pytest.raises(ValueError, match="claude_peers needs a team"):
        build("claude_peers", 1)


# ---- agent.py -------------------------------------------------------------------------------------------

def load_agent(env: dict):
    with mock.patch.dict(os.environ, env):
        sys.modules.pop("agent", None)
        return importlib.import_module("agent")


def team_env(protocol: str = "claude_peers", agents: int = 2, extra: dict | None = None) -> dict:
    return {**BASE_ENV, "RELEASE_TASK": "ale-bench", "MULTIAGENT_N_AGENT": str(agents), 
            "MULTIAGENT_PROTOCOL": f"multiagent_protocols.{protocol}:MultiAgentProtocol", "MULTIAGENT_AGENT_EFFORT": "high",
            "RELEASE_CONTINUATION": (ROOT / "benchmarks/ale-bench/continuation.txt").read_text().rstrip("\n"),
            **(extra or {})}


def render(tmp_path: Path, cls: str = "ClaudeCodeCollaborationAgent", protocol: str = "claude_peers",
           agents: int = 2, extra: dict | None = None, revive: dict | None = None, **kwargs):
    env = team_env(protocol, agents, extra)
    module = load_agent(env)
    with mock.patch.dict(os.environ, env), mock.patch("secrets.token_urlsafe", lambda n: "s" * 32), \
            mock.patch("time.time", lambda: 1700000000.0):
        agent = getattr(module, cls)(logs_dir=tmp_path, model_name="claude-sonnet-5-5", **kwargs)
        environment = FakeEnvironment()

        async def go() -> None:
            await agent.install(environment)
            if revive is None:
                await agent.run(INSTRUCTION, environment, SimpleNamespace())
            else:
                agent.revive_plan = json.loads(json.dumps(revive))
                await agent.resume(INSTRUCTION, environment, SimpleNamespace())

        asyncio.run(go())
    return module, agent, environment.calls


def runner_of(agent, calls: list[dict]) -> str:
    """The team runner: the keeper's copy, or the one the setup writes."""
    if agent.last_runner_script:
        return agent.last_runner_script
    setup = next(call["command"] for call in calls if "/run-agent" in call["command"] and "printf %s" in call["command"])
    words = shlex.split(setup)
    for index, word in enumerate(words):
        if word.endswith(".sh") and words[index - 1] == ">" and "/run-agent" in word:
            return words[index - 2]
    raise AssertionError("no runner in the setup")


def invoke_function(runner: str) -> str:
    start = runner.index("invoke_claude() {")
    return runner[start:runner.index("\n  }\n", start)]


@pytest.mark.parametrize("extra", [{}, FULL], ids=["plain", "full"])
def test_each_agent_gets_the_tools_its_name_its_inbox_and_the_setting(tmp_path, extra):
    module, agent, calls = render(tmp_path, extra=extra, claude_compact_window=277000)
    runner = runner_of(agent, calls)
    invoke = invoke_function(runner)
    assert invoke.count(PEERS_LINE) == 1
    assert ("--tools Bash,BashOutput,TaskOutput,Monitor,KillShell,TaskStop,Read,Glob,Grep,Edit,Write,SendMessage,"
            "ListAgents --disallowedTools Agent,Task,Workflow,WebFetch,WebSearch,Skill ") in invoke
    # Only those lines differ from the same team's runner without claude_peers.
    _, plain_agent, plain_calls = render(tmp_path / "plain", protocol="antiherding_matched",
                                         extra=extra, claude_compact_window=277000)
    plain = invoke_function(runner_of(plain_agent, plain_calls))
    assert invoke.replace(f"      {PEERS_LINE} \\\n", "").replace(
        "Write,SendMessage,ListAgents --disallowedTools Agent,Task,Workflow,WebFetch,WebSearch,Skill",
        "Write --disallowedTools Agent,Task,Workflow,SendMessage,ListAgents,WebFetch,WebSearch,Skill") == plain
    assert runner.count(module.CLAUDE_PEERS_CONFIGURE) == 1
    assert '  mkdir -p "$claude_log_dir"\n' + module.CLAUDE_PEERS_CONFIGURE in runner
    # Every session of an agent goes through invoke_claude: its first turn, continuations, resumed turns.
    assert runner.count('"${claude_command[@]}"') == 1
    commands = "\n".join(call["command"] for call in calls)
    assert "Your teammates are a1, a2. Message a teammate with the SendMessage tool" in commands
    assert "Follow this coordination protocol exactly" not in commands


def test_the_flags_reach_claude_as_separate_arguments():
    module = load_agent(team_env())
    found = subprocess.run(["bash", "-c", f'agent_id=a2; set -- {module.claude_peers_flags()}; printf "%s\\n" "$@"'],
                           capture_output=True, text=True, check=True).stdout.splitlines()
    assert found == ["--name", "a2", "--messaging-socket-path", "/tmp/.claude-peers/a2.sock",
                     "--settings", '{"crossSessionInbound":"accept"}', "--append-system-prompt",
                     "In the team you are `a2`."]
    assert module.CLAUDE_PEERS_IDENTITY.format(name="a2") == "In the team you are `a2`."
    assert json.loads(module.CLAUDE_PEERS_SETTINGS) == {"crossSessionInbound": "accept"}
    # A socket's path has at most 107 bytes; agent ids are short (a1 ... a99).
    assert len(f"{module.CLAUDE_PEERS_SOCKETS}/a99.sock".encode()) < 100


def configure(tmp_path: Path, module, home: Path) -> subprocess.CompletedProcess:
    """CLAUDE_PEERS_CONFIGURE in bash, its container paths moved under TMP_PATH."""
    lines = (module.CLAUDE_PEERS_CONFIGURE.replace("/logs/agent", str(tmp_path / "agent"))
             .replace(module.CLAUDE_PEERS_SOCKETS, str(tmp_path / "sockets")))
    script = f"configure() {{\n{lines}}}\nclaude_log_dir={shlex.quote(str(home))}\nconfigure"
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True)


def test_the_configure_step_links_the_registry_and_makes_the_socket_directory(tmp_path):
    module = load_agent(team_env())
    homes = [tmp_path / "agent" / f"sessions-{index}-a{index + 1}" / "claude-home" for index in range(2)]
    for home in homes:
        home.mkdir(parents=True)
        for _ in range(2):  # again, as a continuation's runner restarted by the keeper or a revival does
            assert configure(tmp_path, module, home).returncode == 0
        link = home / "sessions"
        assert link.is_symlink() and os.readlink(link) == "../../claude-peers-sessions"
        assert link.resolve() == (tmp_path / "agent" / "claude-peers-sessions").resolve()
    (tmp_path / "agent/claude-peers-sessions/123.json").write_text("{}")
    assert sorted(path.name for path in homes[1].joinpath("sessions").iterdir()) == ["123.json"]
    assert oct((tmp_path / "sockets").stat().st_mode & 0o777) == "0o700"
    # A home whose sessions/ is a directory of its own, or a link elsewhere, cannot share the team's registry: the
    # agent does not start.
    other = tmp_path / "agent/sessions-2-a3/claude-home"
    (other / "sessions").mkdir(parents=True)
    elsewhere = tmp_path / "agent/sessions-3-a4/claude-home"
    elsewhere.mkdir(parents=True)
    (elsewhere / "sessions").symlink_to(tmp_path / "agent/claude-peers-sessions")
    for home in (other, elsewhere):
        failed = configure(tmp_path, module, home)
        assert failed.returncode == 1 and "is not the link to the team session registry" in failed.stderr


def test_a_revived_team_runs_the_same_command_line(tmp_path):
    plan = {"revival": 1, "note": "Continue.", "layout": "team", "started": 1.0, "checks_passed": True,
            "agents": [{"index": 0, "session_id": "s0", "turn": 2}, {"index": 1, "session_id": "s1", "turn": 3}]}
    module, agent, calls = render(tmp_path, extra=FULL)
    _, revived, revived_calls = render(tmp_path, extra=FULL, revive=plan)
    run_runner, revived_runner = runner_of(agent, calls), runner_of(revived, revived_calls)
    assert invoke_function(revived_runner) == invoke_function(run_runner)
    assert revived_runner.count(module.CLAUDE_PEERS_CONFIGURE) == 1
    # The revival writes nothing of the team's but its runner: the registry and each agent's link are the run's,
    # and each runner's configure step makes the socket directory again (a rebuilt container has none).
    assert not any("claude-peers" in call["command"] and "run-agent" not in call["command"]
                   and "/usr/lib/hsv" not in call["command"] for call in revived_calls)


@pytest.mark.parametrize("cls, kwargs", [
    ("CodexCollaborationAgent", {}),
    ("MixedTeamAgent", {"team": [{"cli": "claude", "model": "claude-sonnet-5-5"},
                                 {"cli": "claude", "model": "claude-opus-5-5"}]}),
])
def test_agents_without_claude_code_messaging_refuse_it(tmp_path, cls, kwargs):
    with pytest.raises(ValueError, match="cross-session messaging, and .* has none"):
        render(tmp_path, cls=cls, **kwargs)


def test_one_agent_refuses_it(tmp_path):
    with pytest.raises(ValueError, match="one agent has no peers"):
        render(tmp_path, agents=1)


def test_other_protocols_leave_the_claude_command_line_as_it_was(tmp_path):
    module, agent, calls = render(tmp_path, protocol="antiherding", extra=FULL)
    runner = runner_of(agent, calls)
    assert "--messaging-socket-path" not in runner and "claude-peers" not in runner
    assert "--disallowedTools Agent,Task,Workflow,SendMessage,ListAgents,WebFetch,WebSearch,Skill " in runner


# ---- run.py and revive.py -----------------------------------------------------------------------------------

def run_py(*argv: str):
    import run
    args = run.parser().parse_args(list(argv))
    args.dry_run = True
    return run.prepare(args)


PEERS_RUN = ("perf-takehome", "--cli", "claude", "--model", "claude-sonnet-5-5", "--protocol", "claude_peers",
             "--agents", "2", "--token-budget", "100000", "--resources-tool", "--timeout", "3600")


def test_run_py_runs_a_claude_team_on_its_messaging():
    with mock.patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "fake"}):
        cmd, env, _ = run_py(*PEERS_RUN, "--name", "peers-dry")
    assert env["MULTIAGENT_PROTOCOL"] == "multiagent_protocols.claude_peers:MultiAgentProtocol"
    assert env["MULTIAGENT_N_AGENT"] == "2"
    assert cmd[cmd.index("--agent-import-path") + 1] == "agent:ClaudeCodeCollaborationAgent"
    assert env["RELEASE_HARNESS_GUARD"] == "1" and env["RELEASE_LIMIT_RESUME"] == "1"


@pytest.mark.parametrize("argv, message", [
    (("--cli", "codex", "--model", "gpt-6.1-sol", "--agents", "2"), "which only Claude Code has"),
    (("--team", "claude:claude-sonnet-5-5,claude:claude-opus-5-5"), "which only Claude Code has"),
    (("--team", "claude:claude-sonnet-5-5,codex:gpt-6.1-sol"), "which only Claude Code has"),
    (("--cli", "claude", "--model", "claude-sonnet-5-5", "--agents", "1"), "needs a team"),
    (("--cli", "claude", "--model", "claude-sonnet-5-5", "--relay", "2", "--token-budget", "1000"), "needs a team"),
])
def test_run_py_refuses_it(argv, message):
    with mock.patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "fake"}), pytest.raises(ValueError, match=message):
        run_py("ale-bench", "--problem", "ahc014", "--protocol", "claude_peers", *argv, "--name", "peers-dry")


def test_run_py_checks_the_claude_version(tmp_path):
    import run
    binary = ROOT / ".cache/claude-2.1.289/claude"
    if binary.is_file():
        run.claude_peers_claude(binary, dry_run=False)
    old = tmp_path / "claude"
    old.write_text("#!/bin/sh\necho '2.1.200 (Claude Code)'\n")
    old.chmod(0o755)
    with pytest.raises(ValueError, match=r"needs exactly Claude Code 2\.1\.289 .*2\.1\.200 \(Claude Code\)"):
        run.claude_peers_claude(old, dry_run=False)
    new = tmp_path / "claude-new"
    new.write_text("#!/bin/sh\necho '2.1.290 (Claude Code)'\n")
    new.chmod(0o755)
    with pytest.raises(ValueError, match=r"needs exactly Claude Code 2\.1\.289 .*2\.1\.290 \(Claude Code\)"):
        run.claude_peers_claude(new, dry_run=False)
    run.claude_peers_claude(old, dry_run=True)


def test_a_revival_gives_the_registry_back_to_the_agent_user(tmp_path):
    import revive
    from test_revive import keep_aside_revival, touch
    trial = tmp_path / "job" / "t__1"
    for name in ("agent/sessions-0-a1/claude-prompt.txt", "agent/sessions-1-a2/claude-prompt.txt",
                 "agent/agent-0-a1-output.log", "agent/agent-1-a2-output.log", "agent/multiagent-collab/run-agent.sh",
                 "agent/claude-peers-sessions/12.json"):
        touch(trial / name)
    revival = keep_aside_revival(trial)
    revival.cli = "claude"
    environment = revive.RecordingEnvironment()
    harbor = SimpleNamespace(task=SimpleNamespace(config=SimpleNamespace(agent=SimpleNamespace(user="agent"))))
    asyncio.run(revival.give_back(harbor, environment))
    assert environment.calls == ["chown -R agent: /logs/agent/sessions-0-a1 /logs/agent/sessions-1-a2 "
                                 "/logs/agent/agent-0-a1-output.log /logs/agent/agent-1-a2-output.log "
                                 "/logs/agent/multiagent-collab /logs/agent/claude-peers-sessions"]
    assert revive.CLAUDE_PEERS_REGISTRY == load_agent(team_env()).CLAUDE_PEERS_REGISTRY


def test_revive_builds_the_runs_settings():
    import revive
    with mock.patch.dict(os.environ, {"CLAUDE_CODE_OAUTH_TOKEN": "fake"}):
        args, cmd, env = revive.harness([*PEERS_RUN, "--name", "peers-r1"], "peers-r1")
    assert args.protocol == "claude_peers"
    assert env["MULTIAGENT_PROTOCOL"] == "multiagent_protocols.claude_peers:MultiAgentProtocol"


# ---- the journal ---------------------------------------------------------------------------------------------

def test_agent_py_names_the_journal_as_the_journal_module_does():
    assert load_agent(team_env()).CLAUDE_PEERS_JOURNAL == claude_peers_journal.JOURNAL


def test_the_journal_has_every_send_receipt_and_listing():
    records = claude_peers_journal.journal(FIXTURE)
    sends = [r for r in records if r["event"] == "send"]
    receipts = [r for r in records if r["event"] == "receive"]
    assert [(r["from"], r["to"], r["message"], r["ok"]) for r in sends] == [
        ("x2", "x1", "ping from x2", True), ("x2", "x1", "second ping from x2", True),
        ("x1", "x2", "pong from x1", True), ("y2", "y1", "ping from y2", True), ("y1", "y2", "pong from y1", True)]
    assert [(r["agent"], r["from"], r["message"], r["delivery"]) for r in receipts] == [
        ("x1", "x2", "ping from x2", "in_turn"), ("x1", "x2", "second ping from x2", "in_turn"),
        ("x2", "x1", "pong from x1", "in_turn"), ("y1", "y2", "ping from y2", "in_turn"),
        # y1's pong came while y2 gave its final answer: `claude -p` took it as one more turn before it exited.
        ("y2", "y1", "pong from y1", "new_turn")]
    for receipt in receipts:
        # Queued, then in the turn at the receiver's next tool call (after its sleep), or as the prompt of the turn
        # `claude -p` runs for it after the final answer (y2: queued at 14:46:13.283, its turn at 14:46:16.581).
        assert receipt["seen_ts"] > receipt["ts"]
    assert receipts[-1]["ts"] == "2026-11-07T14:46:13.283Z" and receipts[-1]["seen_ts"] == "2026-11-07T14:46:16.581Z"
    for send in sends:
        receipt = next(r for r in receipts if r["msg_id"] == send["msg_id"])
        assert send["received_ts"] == receipt["ts"] and receipt["sent_ts"] == send["ts"] < receipt["ts"]
        assert send["to_agent"] == receipt["agent"] and receipt["from_address"] == f"uds:/tmp/.claude-peers/{send['from']}.sock"
    assert [r["ts"] for r in records] == sorted(r["ts"] for r in records)
    lists = [r for r in records if r["event"] == "list"]
    assert [r["agent"] for r in lists] == ["x2", "x2"]
    assert lists[0]["result"].startswith("This session is x2 [")
    assert claude_peers_journal.summary(records) == {
        "x1": {"sent": 1, "send_failed": 0, "undelivered": 0, "received": 2, "lists": 0},
        "x2": {"sent": 2, "send_failed": 0, "undelivered": 0, "received": 1, "lists": 2},
        "y1": {"sent": 1, "send_failed": 0, "undelivered": 0, "received": 1, "lists": 0},
        "y2": {"sent": 1, "send_failed": 0, "undelivered": 0, "received": 1, "lists": 0}}


def copy_without(tmp_path: Path, pattern: str) -> Path:
    """The fixture without the files matching PATTERN (relative to each session directory)."""
    import shutil
    target = tmp_path / "agent"
    shutil.copytree(FIXTURE, target)
    for session in target.glob("sessions-*"):
        for path in session.glob(pattern):
            path.unlink()
    return target


@pytest.mark.parametrize("left_out", ["claude-attempt-*.jsonl", "claude-home/projects/*/*.jsonl"],
                         ids=["transcripts-only", "stream-json-only"])
def test_either_log_alone_gives_the_same_messages(tmp_path, left_out):
    def key(record):   # (the stream-json logs do not tell how or when a message reached the queue and the turn)
        left_out = ("received_ts",) if record["event"] == "send" else ("ts", "seen_ts", "delivery")
        return {k: v for k, v in record.items() if k not in left_out}
    assert [key(r) for r in claude_peers_journal.journal(copy_without(tmp_path, left_out))] == \
        [key(r) for r in claude_peers_journal.journal(FIXTURE)]


def test_a_send_to_a_peer_between_sessions_fails_at_the_sender(tmp_path):
    session = tmp_path / "agent/sessions-0-a2"
    session.mkdir(parents=True)
    tool_use = {"type": "tool_use", "id": "toolu_1", "name": "SendMessage",
                "input": {"to": "uds:/tmp/.claude-peers/a1.sock", "message": "ping", "summary": "s"}}
    failure = {"success": False, "message": "No agent named 'a1' is reachable.\nUse ListAgents to see everyone you "
                                            "can message."}
    (session / "claude-attempt-3.jsonl").write_text("\n".join(json.dumps(event) for event in [
        {"type": "assistant", "timestamp": "2026-11-07T14:40:07.569Z", "session_id": "s",
         "message": {"content": [tool_use]}},
        {"type": "user", "timestamp": "2026-11-07T14:40:07.578Z", "session_id": "s", "tool_use_result": failure,
         "message": {"content": [{"type": "tool_result", "tool_use_id": "toolu_1",
                                  "content": [{"type": "text", "text": json.dumps(failure)}]}]}},
    ]) + "\n")
    [send] = claude_peers_journal.journal(tmp_path / "agent")
    assert send["ok"] is False and send["to_agent"] == "a1" and send["msg_id"] is None
    assert send["result"].startswith("No agent named 'a1' is reachable.")
    assert claude_peers_journal.summary([send]) == {
        "a2": {"sent": 1, "send_failed": 1, "undelivered": 0, "received": 0, "lists": 0}}
    accepted = {**send, "ok": True, "msg_id": "m1"}   # accepted, but the receiver ended before its next tool call
    assert claude_peers_journal.summary([accepted])["a2"]["undelivered"] == 1


def test_the_journal_is_written_after_the_run(tmp_path):
    import shutil
    logs = tmp_path / "agent"
    shutil.copytree(FIXTURE, logs)
    for protocol, written in (("antiherding", False), ("claude_peers", True)):
        env = team_env(protocol)
        module = load_agent(env)
        with mock.patch.dict(os.environ, env):
            module.ClaudeCodeCollaborationAgent(logs_dir=logs, model_name="claude-sonnet-5-5") \
                .populate_context_post_run(SimpleNamespace(metadata=None))
        assert (logs / "claude-peers.jsonl").exists() is written
    lines = (logs / "claude-peers.jsonl").read_text().splitlines()
    assert [json.loads(line) for line in lines] == claude_peers_journal.journal(FIXTURE)
    found = subprocess.run([sys.executable, str(ROOT / "claude_peers_journal.py"), str(logs), "--out",
                            str(tmp_path / "again.jsonl"), "--print"], capture_output=True, text=True, check=True)
    assert (tmp_path / "again.jsonl").read_text() == (logs / "claude-peers.jsonl").read_text()
    assert "x2 -> x1 [ok]: ping from x2" in found.stdout and '"send_failed": 0' in found.stdout


# ---- end to end: two `claude -p` processes message each other ---------------------------------------------------

class PeersTeam:
    """A 2-agent claude_peers team run by agent.py (its runners, keeper and watchers) on this machine, with the pinned
    Claude Code, each agent answered from its own plan by tests/peers_mock_api.py: no network, no model."""

    def __init__(self, tmp: Path, plans: dict, during=None):
        import threading
        import time
        from test_harness_guard import GUARD, LIMIT, Local
        from test_limit_resume import CLAUDE_BINARY

        tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "plans.json").write_text(json.dumps(plans))
        self.log, ready = tmp / "requests.jsonl", tmp / "port"
        self.mock = subprocess.Popen([sys.executable, str(ROOT / "tests/peers_mock_api.py"), "--plans",
                                      str(tmp / "plans.json"), "--log", str(self.log), "--ready", str(ready)])
        for _ in range(200):
            if ready.exists():
                break
            time.sleep(0.05)
        home = tmp / "home"
        home.mkdir()
        self.sockets = tmp / "s"   # in place of /tmp/.claude-peers: a socket's path has at most 107 bytes
        base = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(home), "HTTPS_PROXY": "http://127.0.0.1:9",
                "HTTP_PROXY": "http://127.0.0.1:9", "NO_PROXY": "127.0.0.1,localhost",
                "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{ready.read_text()}"}
        sockets = self.sockets

        class PeersLocal(Local):
            def patch(self, command: str) -> str:
                return super().patch(command).replace("/tmp/.claude-peers", str(sockets))

        self.local = PeersLocal(tmp, base)
        env = {**team_env(extra={**LIMIT, **GUARD, "RELEASE_TOKEN_BUDGET": "4000000"}), "RELEASE_CONTINUATIONS": "1",
               "MULTIAGENT_AGENT_EFFORT": "", "META_DUO_CONTAINER_CLAUDE_BINARY": str(CLAUDE_BINARY)}
        self.error: list = []

        def main() -> None:
            try:
                module = load_agent(env)
                with mock.patch.dict(os.environ, env):
                    agent = module.ClaudeCodeCollaborationAgent(logs_dir=self.local.agent_dir,
                                                                model_name="claude-sonnet-5-5")
                    asyncio.run(agent.run(INSTRUCTION, self.local, SimpleNamespace()))
                    agent.populate_context_post_run(SimpleNamespace(metadata=None))  # as Harbor does after the run
            except BaseException as exc:  # noqa: BLE001
                self.error.append(exc)

        thread = threading.Thread(target=main, daemon=True)
        thread.start()
        if during is not None:
            during(self)
        thread.join(300)
        self.ended = not thread.is_alive()

    def requests(self, agent: str) -> list[dict]:
        return [entry for entry in map(json.loads, self.log.read_text().splitlines())
                if entry["agent"] == agent and entry["main"]]

    def close(self) -> None:
        import harness_guard
        self.mock.terminate()
        self.mock.wait(timeout=10)
        for session in self.local.agent_dir.glob("sessions-*"):
            for pid in harness_guard.agent_processes("CLAUDE_CONFIG_DIR", str(session / "claude-home")):
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass


def tool_results(request: dict) -> list[str]:
    found = []
    for message in request["body"]["messages"]:
        for part in message["content"] if isinstance(message["content"], list) else []:
            if part.get("type") == "tool_result":
                content = part.get("content")
                found.append(content if isinstance(content, str) else
                             "".join(item.get("text", "") for item in content or () if isinstance(item, dict)))
    return found


@pytest.mark.skipif(not (ROOT / ".cache/claude-2.1.289/claude").is_file(), reason="pinned Claude Code release missing")
def test_two_claude_sessions_message_each_other_across_a_continuation(tmp_path):
    send = lambda to, text: {"type": "tool", "name": "SendMessage", "input": {"to": to, "message": text}}  # noqa: E731
    bash = lambda command: {"type": "tool", "name": "Bash", "input": {"command": command}}  # noqa: E731
    plans = {
        # a1's first turn: list, message a2 while a2 sleeps, end; its continuation (a new `claude -p --resume`) sleeps
        # while a2 messages it there.
        "a1": [{"type": "tool", "name": "ListAgents", "input": {}, "after": {"agent": "a2", "requests": 1}},
               send("a2", "hello a2"), bash("sleep 1"), {"type": "text", "text": "turn 0 over"},
               bash("sleep 8"), {"type": "text", "text": "turn 1 over"}],
        # a2 sleeps through a1's message; its next request waits until a1's continuation has started (a1's fifth).
        "a2": [bash("sleep 3"), {**send("a1", "hello a1 again"), "after": {"agent": "a1", "requests": 5}},
               {"type": "text", "text": "a2 over"}],
    }
    team = PeersTeam(tmp_path / "run", plans)
    try:
        assert team.ended and not team.error, team.error
        a1, a2 = team.requests("a1"), team.requests("a2")
        # a1 saw a2 by name; a2 got a1's message in its running turn, after its sleep.
        listed = tool_results(a1[1])[-1]
        assert "This session is a1 [" in listed and "a2 [" in listed
        assert '"success":true' in tool_results(a1[2])[-1]
        delivered = json.dumps(a2[1]["body"]["messages"])
        assert 'from-name=\\"a1\\"' in delivered and "hello a2" in delivered
        # a1's continuation, a new process, kept its name and socket: a2's message reached it there.
        assert '"success":true' in tool_results(a2[2])[-1]
        assert a1[4]["step"] == bash("sleep 8")
        later = json.dumps(a1[5]["body"]["messages"])
        assert 'from-name=\\"a2\\"' in later and "hello a1 again" in later
        assert "hello a1 again" not in json.dumps(a1[4]["body"]["messages"])
        # The journal, written after the run.
        records = [json.loads(line) for line in (team.local.agent_dir / "claude-peers.jsonl").read_text().splitlines()]
        sends = [(r["from"], r["to"], r["message"], r["ok"]) for r in records if r["event"] == "send"]
        receipts = [(r["agent"], r["from"], r["message"], r["from_address"]) for r in records if r["event"] == "receive"]
        assert sends == [("a1", "a2", "hello a2", True), ("a2", "a1", "hello a1 again", True)]
        assert receipts == [("a2", "a1", "hello a2", f"uds:{team.sockets}/a1.sock"),
                            ("a1", "a2", "hello a1 again", f"uds:{team.sockets}/a2.sock")]
        for index, agent in enumerate(("a1", "a2")):
            link = team.local.agent_dir / f"sessions-{index}-{agent}" / "claude-home" / "sessions"
            assert os.readlink(link) == "../../claude-peers-sessions"
    finally:
        team.close()


@pytest.mark.skipif(not (ROOT / ".cache/claude-2.1.289/claude").is_file(), reason="pinned Claude Code release missing")
def test_a_resumed_turn_takes_a_message_that_comes_during_its_final_answer(tmp_path):
    """A turn the runner resumes (after a kill, a team hold or a lockstep pause: `--resume --input-format
    stream-json`, stdin at its end) keeps the agent's address, and a message that comes while it writes its final
    answer gets one more turn of its own; the CLI does not hang."""
    import time
    import harness_guard
    from test_limit_resume import CLAUDE_BINARY
    bash = lambda command: {"type": "tool", "name": "Bash", "input": {"command": command}}  # noqa: E731
    plans = {
        # a1's first command is killed; the resumed turn answers slowly, and a2's message comes meanwhile.
        "a1": [bash("sleep 30"), {"type": "text", "text": "resumed turn over", "delay": 6},
               {"type": "text", "text": "read the late message"}],
        "a2": [bash("sleep 1"), {"type": "tool", "name": "SendMessage", "input": {"to": "a1", "message": "late hello"},
                                 "after": {"agent": "a1", "requests": 2}},
               {"type": "text", "text": "a2 over"}],
    }

    def kill_a1_mid_command(team: PeersTeam) -> None:
        home = team.local.agent_dir / "sessions-0-a1" / "claude-home"
        deadline = time.time() + 60
        while time.time() < deadline and not (team.log.exists() and team.requests("a1")):
            time.sleep(0.2)
        time.sleep(1)
        clis = harness_guard.cli_processes("CLAUDE_CONFIG_DIR", str(home), str(CLAUDE_BINARY))
        assert clis, "a1's CLI is not running"
        for pid in clis:
            os.kill(pid, 9)

    team = PeersTeam(tmp_path / "run", plans, during=kill_a1_mid_command)
    try:
        assert team.ended and not team.error, team.error
        a1 = team.requests("a1")
        log = (team.local.agent_dir / "agent-0-a1-output.log").read_text()
        assert '"event":"harness_cli_resumed"' in log
        assert [request["step"].get("text") for request in a1[1:3]] == ["resumed turn over", "read the late message"]
        assert "late hello" not in json.dumps(a1[1]["body"]["messages"])
        assert "Another Claude session sent a message" in json.dumps(a1[2]["body"]["messages"][-1])
        assert "late hello" in json.dumps(a1[2]["body"]["messages"][-1])
        records = [json.loads(line) for line in (team.local.agent_dir / "claude-peers.jsonl").read_text().splitlines()]
        [receipt] = [r for r in records if r["event"] == "receive"]
        assert (receipt["agent"], receipt["message"], receipt["delivery"]) == ("a1", "late hello", "new_turn")
        assert receipt["seen_ts"] > receipt["ts"]
    finally:
        team.close()
