"""Usage-limit resumption (limit_resume.py, agent.py, run.py --limit-resume).

The end-to-end tests render an agent's commands with agent.py (cli_render), run them on this
machine instead of in a container (/logs/agent becomes a temporary directory), and point the
pinned Claude Code and Codex releases at tests/limit_mock_api.py on 127.0.0.1, which answers
from a plan and rejects requests with each provider's real usage-limit 429 for a while. No
network: anything else goes to a dead proxy. They are skipped without the pinned releases.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from cli_render import ROOT, render

sys.path.insert(0, str(ROOT))
import limit_resume  # noqa: E402

CLAUDE_BINARY = ROOT / ".cache/claude-2.1.289/claude"
CODEX_BINARY = ROOT / ".cache/codex-0.160.0/bin/codex"
MOCK = Path(__file__).with_name("limit_mock_api.py")
NOTE = limit_resume.RESUME_NOTE
ON = {"RELEASE_LIMIT_RESUME": "1", "RELEASE_LIMIT_WAIT_MAX": "172800", "RELEASE_AGENT_TIMEOUT": "3600"}
# A line of the instruction that appears nowhere else in a prompt.
MARK = "It's a test with 'quotes' and $dollars."

needs_claude = pytest.mark.skipif(not CLAUDE_BINARY.is_file(), reason="pinned Claude Code release missing")
needs_codex = pytest.mark.skipif(not CODEX_BINARY.is_file(), reason="pinned Codex release missing")


# ---- the mock and the local runner ------------------------------------------------------------

class Mock:
    def __init__(self, directory: Path, plan: list):
        self.directory = directory
        (directory / "plan.json").write_text(json.dumps(plan))
        self.log = directory / "requests.jsonl"
        ready = directory / "port"
        self.process = subprocess.Popen([sys.executable, str(MOCK), "--plan", str(directory / "plan.json"),
                                         "--log", str(self.log), "--ready", str(ready)])
        for _ in range(100):
            if ready.exists():
                break
            time.sleep(0.05)
        self.port = int(ready.read_text())

    def requests(self) -> list:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines() if line.strip()]

    def main(self) -> list:
        return [r for r in self.requests() if r.get("main")]

    def stop(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=10)


def codex_hook(directory: Path, port: int) -> Path:
    """A task CLI hook (agent.py's RELEASE_TASK_CLI_HOOK) that points Codex at the mock."""
    provider = {"name": "mock", "base_url": f"http://127.0.0.1:{port}/v1", "wire_api": "responses",
                "env_key": "CODEX_API_KEY", "supports_websockets": "false", "request_max_retries": "0",
                "stream_max_retries": "0"}
    args = ["-c", 'model_provider="mock"']
    for key, value in provider.items():
        args += ["-c", f"model_providers.mock.{key}={value if value in ('false', '0') else json.dumps(value)}"]
    hook = directory / "codex-hook.sh"
    hook.write_text("task_cli_args=(" + " ".join(f"'{arg}'" for arg in args) + ")\n")
    return hook


def run_agent(tmp_path: Path, cli: str, plan: list, case: str = "ale-solo", env: dict | None = None,
              patches: dict | None = None, timeout: float = 240, agent_kwargs: dict | None = None) -> SimpleNamespace:
    """Render CASE's commands for CLI with usage-limit resumption on (and AGENT_KWARGS, the Harbor
    agent kwargs), and run its setup and run commands here against a fresh mock answering from PLAN."""
    mock = Mock(tmp_path, plan)
    agent_dir = tmp_path / "agent"
    overrides = {**ON, "RELEASE_CONTINUATIONS": "1", "MULTIAGENT_AGENT_EFFORT": "",
                 "META_DUO_CONTAINER_CLAUDE_BINARY": str(CLAUDE_BINARY),
                 "META_DUO_CONTAINER_CODEX_BINARY": str(CODEX_BINARY), "CODEX_API_KEY": "sk-mock",
                 **(env or {})}
    model = "claude-sonnet-5-5" if cli == "claude" else "gpt-mock"
    calls = render(case, cli=cli, model=model, env_overrides=overrides, agent_kwargs=agent_kwargs)
    commands = []
    for call in calls[2:]:  # after install()
        command = call["command"].replace("/logs/agent", str(agent_dir)).replace("rand()*30", "rand()*0")
        command = command.replace("JITTER = (30, 90)", "JITTER = (0, 0)")
        for old, new in (patches or {}).items():
            command = command.replace(old, new)
        commands.append((command, call["env"]))
    home = tmp_path / "home"
    home.mkdir()
    base = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(home),
            "HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9", "NO_PROXY": "127.0.0.1,localhost",
            "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{mock.port}",
            "RELEASE_TASK_CLI_HOOK": str(codex_hook(tmp_path, mock.port)) if cli == "codex" else "",
            "META_DUO_CLAUDE_API_RETRY_BASE_SLEEP": "2", "META_DUO_CODEX_API_RETRY_BASE_SLEEP": "2",
            **{k: v for k, v in (env or {}).items() if k.startswith("META_DUO_")}}
    results = []
    started = time.time()
    try:
        for command, call_env in commands:
            results.append(subprocess.run(["bash", "-c", f"set -o pipefail; {command}"], env={**base, **call_env},
                                          capture_output=True, text=True, timeout=timeout))
    finally:
        mock.stop()
    elapsed = time.time() - started
    session = next(iter(sorted(agent_dir.glob("sessions-*"))), None) or agent_dir / f"{cli}-single-session"
    logs = sorted(agent_dir.glob("agent-*-output.log"))
    log_text = logs[0].read_text(errors="replace") if logs else ""
    return SimpleNamespace(mock=mock, results=results, session=session, log=log_text, elapsed=elapsed,
                           home=session / ("claude-home" if cli == "claude" else "codex-home"), cli=cli)


def feedback(run: SimpleNamespace, event: str) -> list:
    found = []
    for line in run.log.splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if entry.get("type") == "environment.feedback" and entry["data"].get("event") == event:
            found.append(entry["data"])
    return found


def user_texts(request: dict) -> list[str]:
    """The user-written texts of a request's conversation (not tool results, not the CLIs' own
    context blocks)."""
    body = request["body"]
    texts = []
    if request["provider"] == "anthropic":
        for message in body.get("messages", []):
            if message.get("role") != "user":
                continue
            content = message["content"]
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
            texts += [b["text"] for b in blocks if b.get("type") == "text"
                      and not b["text"].startswith("<system-reminder>")]
    else:
        for item in body.get("input", []):
            if item.get("type") == "message" and item.get("role") == "user":
                texts += [c["text"] for c in item.get("content", []) if c.get("type") == "input_text"
                          and not c["text"].startswith("<environment_context>")]
    return texts


def assert_each_message_once(run: SimpleNamespace, continuation: str, turns: int) -> list[str]:
    """The last request carries the task prompt once and each continuation turn's prompt once
    (all continuations have the same text, so it appears once per turn)."""
    texts = user_texts(run.mock.main()[-1])
    assert sum(MARK in text for text in texts) == 1, texts
    assert sum(text.strip() == continuation for text in texts) == turns, texts
    return texts


def assert_waited_for_reset(run: SimpleNamespace, limits: int) -> None:
    """Each usage limit rejected one request, and the next request came only after the limit
    had ended: nothing was sent, so no tokens spent, while waiting."""
    main = run.mock.main()
    rejected = [i for i, request in enumerate(main) if request["response"] == "429"]
    assert len(rejected) == limits, [r["response"] for r in main]
    for index in rejected:
        limit_end = main[index]["ts"] + main[index]["step"]["seconds"]
        assert main[index + 1]["ts"] >= limit_end


def continuation_text() -> str:
    return (ROOT / "benchmarks/ale-bench/continuation.txt").read_text().rstrip("\n")


def polyomino_continuation() -> str:
    return (ROOT / "benchmarks/ale-bench/continuation.txt").read_text().rstrip("\n")


# ---- end to end: Claude Code ------------------------------------------------------------------

@needs_claude
def test_claude_waits_out_limits_in_the_turn(tmp_path):
    """A limit on the task prompt's first request, one after a tool call, and one on the
    continuation's first request: Claude Code waits each out itself and sends the same request
    again, so the conversation gains no message at all."""
    plan = [{"type": "limit", "seconds": 8}, {"type": "tool", "command": "echo one"},
            {"type": "limit", "seconds": 8}, {"type": "text", "text": "done-0"},
            {"type": "limit", "seconds": 8}, {"type": "tool", "command": "echo two"}, {"type": "text", "text": "done-1"}]
    run = run_agent(tmp_path, "claude", plan)
    assert run.results[1].returncode == 0, run.results[1].stdout[-2000:] + run.results[1].stderr[-2000:]
    texts = assert_each_message_once(run, continuation_text(), 1)
    assert NOTE not in texts
    assert_waited_for_reset(run, 3)
    assert not feedback(run, "interrupted_turn_resumed") and not feedback(run, "usage_limit_wait")
    state = json.loads((run.session / "limit-wait.json").read_text())
    assert 20 <= state["waited_seconds"] <= 45, state


@needs_claude
def test_claude_resumes_an_interrupted_turn_without_its_prompt(tmp_path):
    """If Claude Code still ends on an error (here a 500 with no retries left), the runner
    resumes the interrupted turn: the prompt is not sent again; Claude Code adds its meta
    "Continue from where you left off."."""
    plan = [{"type": "tool", "command": "echo one"}, {"type": "limit", "seconds": 1, "status": 500},
            {"type": "tool", "command": "echo two"}, {"type": "text", "text": "done-0"}, {"type": "text", "text": "done-1"}]
    run = run_agent(tmp_path, "claude", plan, patches={"CLAUDE_CODE_MAX_RETRIES=10": "CLAUDE_CODE_MAX_RETRIES=0"})
    assert run.results[1].returncode == 0, run.results[1].stdout[-2000:]
    texts = assert_each_message_once(run, continuation_text(), 1)
    assert texts.count(NOTE) == 1
    assert len(feedback(run, "interrupted_turn_resumed")) == 1
    [transcript] = run.home.glob("projects/*/*.jsonl")
    assert sum(MARK in text for text in limit_resume.user_texts("claude", str(transcript), 0)) == 1


# ---- end to end: Codex ------------------------------------------------------------------------

@needs_codex
def test_codex_waits_for_the_reset_and_never_resends_a_prompt(tmp_path):
    """The same three limits with Codex, on the single-agent runner: each time the runner waits
    until the reset Codex recorded, then resumes with the note instead of the recorded prompt."""
    plan = [{"type": "limit", "seconds": 6}, {"type": "tool", "command": "echo one"},
            {"type": "limit", "seconds": 6}, {"type": "text", "text": "done-0"},
            {"type": "limit", "seconds": 6}, {"type": "tool", "command": "echo two"}, {"type": "text", "text": "done-1"}]
    run = run_agent(tmp_path, "codex", plan, case="ale-single")
    assert run.results[1].returncode == 0, run.results[1].stdout[-3000:]
    texts = assert_each_message_once(run, polyomino_continuation(), 1)
    assert texts.count(NOTE) == 3
    assert_waited_for_reset(run, 3)
    waits = feedback(run, "usage_limit_wait")
    assert len(waits) == 3 and len(feedback(run, "interrupted_turn_resumed")) == 3
    [rollout] = run.home.glob("sessions/**/rollout-*.jsonl")
    recorded = list(limit_resume.user_texts("codex", str(rollout), 0))
    assert sum(MARK in text for text in recorded) == 1 and recorded.count(NOTE) == 3


@needs_codex
def test_codex_team_runner(tmp_path):
    """The team runner (run()) with Codex: a limit on the task prompt's first request."""
    plan = [{"type": "limit", "seconds": 5}, {"type": "text", "text": "done-0"}, {"type": "text", "text": "done-1"}]
    run = run_agent(tmp_path, "codex", plan)
    assert run.results[1].returncode == 0, run.results[1].stdout[-3000:]
    texts = assert_each_message_once(run, continuation_text(), 1)
    assert texts.count(NOTE) == 1
    assert_waited_for_reset(run, 1)


# ---- the 48-hour cap and the working time -----------------------------------------------------

@pytest.mark.parametrize("cli", [pytest.param("claude", marks=needs_claude), pytest.param("codex", marks=needs_codex)])
def test_a_limit_that_never_clears_ends_the_agent_at_the_wait_cap(tmp_path, cli):
    plan = [{"type": "tool", "command": "echo one"}, {"type": "limit", "seconds": 10 ** 7}]
    run = run_agent(tmp_path, cli, plan, env={"RELEASE_LIMIT_WAIT_MAX": "12"})
    assert run.results[1].returncode == 0, run.results[1].stdout[-3000:]
    assert run.elapsed < 150  # not the limit's 10^7 s
    # Codex: the runner waits exactly the 12 s left. Claude Code: the watcher stops the CLI at
    # its first poll past 12 s. Either way the final record counts until the runner ended.
    assert 12 <= json.loads((run.session / "limit-wait.json").read_text())["waited_seconds"] <= 20
    [event] = feedback(run, "limit_wait_exhausted")
    assert event["maxWaitSeconds"] == 12
    assert (run.session / limit_resume.EXHAUSTED).exists()
    assert [r["response"] for r in run.mock.main()] == ["tool", "429"]  # nothing sent while waiting
    assert not (run.session / limit_resume.TIME_CAP).exists()


@pytest.mark.parametrize("cli", [pytest.param("claude", marks=needs_claude), pytest.param("codex", marks=needs_codex)])
def test_working_time_excludes_waiting(tmp_path, cli):
    """--timeout 40 s with a 15 s limit after the first tool call: the agent keeps working past
    40 s of wall-clock time and is stopped once it has worked 40 s (start-up counts as work,
    so a loaded machine leaves less for the tool calls)."""
    plan = [{"type": "tool", "command": "sleep 3"}, {"type": "limit", "seconds": 15},
            *[{"type": "tool", "command": "sleep 6"}] * 12, {"type": "text", "text": "done"}]
    run = run_agent(tmp_path, cli, plan, env={"RELEASE_AGENT_TIMEOUT": "40"})
    assert run.results[1].returncode == 0, run.results[1].stdout[-3000:]
    assert len(feedback(run, "time_cap_reached")) == 1
    assert (run.session / limit_resume.TIME_CAP).exists()
    responses = [r["response"] for r in run.mock.main()]
    assert "429" in responses and responses.index("429") < len(responses) - 2  # worked after the wait
    assert responses[-1] == "tool"  # stopped in the middle of the work, not at its end
    state = json.loads((run.session / "limit-wait.json").read_text())
    assert state["waited_seconds"] >= 14 and run.elapsed >= 40 + 14
    # The run's last command reports it, and agent.py raises as Harbor's own timeout would.
    assert run.results[2].stdout.strip()


# ---- the watcher and the helper commands (no CLI) ---------------------------------------------

def test_union_length():
    assert limit_resume.union_length([(0, 10), (5, 15), (20, 30)], 100) == 25
    assert limit_resume.union_length([(0, 10), (5, 15), (20, 30)], 25) == 20
    assert limit_resume.union_length([], 5) == 0


def test_cli_waits_end_when_the_cli_works_again(tmp_path):
    log = tmp_path / "claude-attempt-0.jsonl"
    waits = limit_resume.CliWaits("claude", str(tmp_path))
    log.write_text(json.dumps({"type": "system", "subtype": "api_retry", "retry_delay_ms": 60000}) + "\n")
    assert waits.poll(100.0) == [(100.0, 160.0)]
    with log.open("a") as stream:  # the watchdog's next 30 s chunk, then the answer
        stream.write(json.dumps({"type": "system", "subtype": "api_retry", "retry_delay_ms": 30000}) + "\n")
    waits.poll(130.0)
    with log.open("a") as stream:
        stream.write(json.dumps({"type": "assistant", "message": {}}) + "\n" + '{"partial')
    intervals = waits.poll(158.0)
    assert limit_resume.union_length(intervals, 200.0) == 58.0
    assert limit_resume.CliWaits("codex", str(tmp_path)).poll(200.0) == []


def test_watch_stops_at_the_wait_cap_and_at_the_working_time(tmp_path):
    session = tmp_path / "s"
    session.mkdir()
    home = session / "claude-home"
    home.mkdir()
    now = time.time()
    # 30 s of runner waiting already, a 20 s cap on waiting: exhausted.
    (session / limit_resume.WAITS).write_text(json.dumps({"start": now - 40, "end": now - 10}) + "\n")
    watcher = subprocess.Popen([sys.executable, str(ROOT / "limit_resume.py"), "watch", "claude", str(session),
                                "CLAUDE_CONFIG_DIR", str(home), str(now - 50), "", "20", "0.2"])
    try:
        for _ in range(50):
            if (session / limit_resume.EXHAUSTED).exists():
                break
            time.sleep(0.1)
    finally:
        watcher.kill()
    assert (session / limit_resume.EXHAUSTED).exists()
    state = json.loads((session / limit_resume.STATE).read_text())
    assert 29 <= state["waited_seconds"] <= 31 and 19 <= state["active_seconds"] <= 22

    other = tmp_path / "t"
    other.mkdir()
    (other / limit_resume.WAITS).write_text(json.dumps({"start": now - 40, "end": now - 10}) + "\n")
    # Started 50 s ago, 30 s of it waiting: 20 s of work against a 15 s working time.
    watcher = subprocess.Popen([sys.executable, str(ROOT / "limit_resume.py"), "watch", "codex", str(other),
                                "CODEX_HOME", str(other / "codex-home"), str(now - 50), "15", "1000", "0.2"])
    try:
        for _ in range(50):
            if (other / limit_resume.TIME_CAP).exists():
                break
            time.sleep(0.1)
    finally:
        watcher.kill()
    assert (other / limit_resume.TIME_CAP).exists() and not (other / limit_resume.EXHAUSTED).exists()


def test_watch_records_the_final_times_when_stopped(tmp_path):
    """The runner stops the watcher as it ends; the watcher then records the final times, so
    limit-wait.json is not a poll behind (a 12 s wait once ended recorded as 11.6 s)."""
    now = time.time()
    (tmp_path / limit_resume.WAITS).write_text(json.dumps({"start": now, "end": now + 2}) + "\n")
    watcher = subprocess.Popen([sys.executable, str(ROOT / "limit_resume.py"), "watch", "codex", str(tmp_path),
                                "CODEX_HOME", str(tmp_path / "codex-home"), str(now), "", "1000", "1000"])
    try:
        for _ in range(100):
            if (tmp_path / limit_resume.STATE).exists():
                break
            time.sleep(0.1)
        first = json.loads((tmp_path / limit_resume.STATE).read_text())
        time.sleep(max(now + 2.5 - time.time(), 0))
        watcher.terminate()
        assert watcher.wait(timeout=10) == 0
    finally:
        watcher.kill()
    final = json.loads((tmp_path / limit_resume.STATE).read_text())
    assert final["updated_at"] > first["updated_at"]
    assert final["waited_seconds"] == 2.0 and not final["waiting"]
    assert not (tmp_path / limit_resume.EXHAUSTED).exists() and not (tmp_path / limit_resume.TIME_CAP).exists()


def test_plan_wait_grants_at_most_the_wait_left(tmp_path, capsys):
    now = time.time()
    (tmp_path / limit_resume.STATE).write_text(json.dumps({"waited_seconds": 90}))
    limit_resume.plan_wait(str(tmp_path), "30", "100")
    assert capsys.readouterr().out.strip() == "10"
    limit_resume.plan_wait(str(tmp_path), "5", "1000")
    assert capsys.readouterr().out.strip() == "5"
    waits = [json.loads(line) for line in (tmp_path / limit_resume.WAITS).read_text().splitlines()]
    assert [round(w["end"] - w["start"]) for w in waits] == [10, 5] and waits[0]["start"] >= now - 1


def write_events(path: Path, events: list) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    return str(path)


def test_delay_claude(tmp_path, capsys):
    now = time.time()
    rejected = {"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "resetsAt": now + 600}}
    allowed = {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "resetsAt": now + 9000}}
    log = write_events(tmp_path / "a.jsonl", [allowed, rejected, {"type": "result", "is_error": True}])
    limit_resume.delay("claude", log, str(tmp_path), "s1")
    seconds, reset = capsys.readouterr().out.split()
    assert 600 + 30 <= int(seconds) <= 601 + 90 and int(reset) == int(now + 600)
    # A rejection the CLI already waited out (the last event allows) is no usage limit.
    limit_resume.delay("claude", write_events(tmp_path / "b.jsonl", [rejected, allowed]), str(tmp_path), "s1")
    assert capsys.readouterr().out == ""


def test_delay_codex(tmp_path, capsys):
    now = time.time()
    home = tmp_path / "codex-home"
    failed = [{"type": "error", "message": "You've hit your usage limit. Try again at 3:05 PM."},
              {"type": "turn.failed", "error": {"message": "You've hit your usage limit. Try again at 3:05 PM."}}]
    log = write_events(tmp_path / "a.jsonl", failed)
    token_count = {"type": "event_msg", "payload": {"type": "token_count", "rate_limits": {
        "primary": {"used_percent": 100.0, "window_minutes": 300, "resets_at": now + 300},
        "secondary": {"used_percent": 100.0, "window_minutes": 10080, "resets_at": now + 7200},
        "credits": None}}}
    write_events(home / "sessions/2026/10/10/rollout-2026-10-10T10-00-00-t1.jsonl", [token_count])
    limit_resume.delay("codex", log, str(home), "t1")
    seconds, reset = capsys.readouterr().out.split()
    assert int(reset) == int(now + 7200)  # both windows exhausted: the later reset
    # No recorded reset: the clock time in the message; neither: wait UNKNOWN_RESET_WAIT.
    limit_resume.delay("codex", log, str(home), "other")
    seconds, reset = capsys.readouterr().out.split()
    assert int(seconds) <= 24 * 3600 + 90 and datetime_local(int(reset)) == (15, 5)
    limit_resume.delay("codex", write_events(tmp_path / "b.jsonl", [{"type": "turn.failed", "error": {
        "message": "You've hit your usage limit."}}]), str(home), "other")
    assert capsys.readouterr().out.split()[0] == str(limit_resume.UNKNOWN_RESET_WAIT)
    limit_resume.delay("codex", write_events(tmp_path / "c.jsonl", [{"type": "turn.failed", "error": {
        "message": "stream disconnected"}}]), str(home), "t1")
    assert capsys.readouterr().out == ""


def datetime_local(epoch: int) -> tuple[int, int]:
    from datetime import datetime
    moment = datetime.fromtimestamp(epoch)
    return moment.hour, moment.minute


def test_input_tells_whether_the_prompt_is_recorded(tmp_path, capsys):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Keep going.\n")
    home = tmp_path / "claude-home"
    before, after = "2026-10-10T08:00:00.000Z", "2026-10-10T09:00:00.000Z"
    since = limit_resume.iso_time("2026-10-10T08:30:00Z")
    rows = [{"type": "user", "timestamp": before, "message": {"role": "user", "content": "Keep going."}},
            {"type": "assistant", "timestamp": before, "message": {"content": []}}]
    path = home / "projects/-work/s1.jsonl"
    write_events(path, rows)
    # The same text from an earlier turn does not count.
    limit_resume.resume_input("claude", str(home), "s1", str(prompt), str(since))
    assert capsys.readouterr().out.strip() == "prompt"
    write_events(path, rows + [{"type": "user", "isMeta": True, "timestamp": after,
                                "message": {"content": "Keep going."}}])
    limit_resume.resume_input("claude", str(home), "s1", str(prompt), str(since))
    assert capsys.readouterr().out.strip() == "prompt"
    write_events(path, rows + [{"type": "user", "timestamp": after, "message": {"content": [
        {"type": "text", "text": "Keep going."}]}}])
    limit_resume.resume_input("claude", str(home), "s1", str(prompt), str(since))
    assert capsys.readouterr().out.strip() == "resume"
    limit_resume.resume_input("claude", str(home), "missing", str(prompt), str(since))
    assert capsys.readouterr().out.strip() == "prompt"

    codex_home = tmp_path / "codex-home"
    item = {"timestamp": after, "type": "response_item", "payload": {
        "type": "message", "role": "user", "content": [{"type": "input_text", "text": "Keep going."}]}}
    write_events(codex_home / "sessions/2026/10/10/rollout-2026-10-10T09-00-00-t1.jsonl", [item])
    limit_resume.resume_input("codex", str(codex_home), "t1", str(prompt), str(since))
    assert capsys.readouterr().out.strip() == "resume"
    limit_resume.resume_input("codex", str(codex_home), "t1", str(prompt), str(since + 7200))
    assert capsys.readouterr().out.strip() == "prompt"


# ---- what the harness sends -------------------------------------------------------------------

def test_off_sends_nothing_new():
    for cli in ("claude", "codex"):
        text = "\n".join(call["command"] for call in render("ale-team3", cli=cli))
        for word in ("limit_resume_source", "RETRY_WATCHDOG", "start_limit_watch", "sed -u", "resume_note"):
            assert word not in text, (cli, word)


@pytest.mark.parametrize("cli", ["claude", "codex"])
@pytest.mark.parametrize("case", ["ale-solo", "ale-single", "ale-team3"])
def test_on_runners(cli, case, tmp_path):
    calls = render(case, cli=cli, env_overrides=ON)
    text = "\n".join(call["command"] for call in calls)
    assert "start_limit_watch" in text and "limit_wait" in text and "resume_" + cli + "_turn" in text
    assert ("CLAUDE_CODE_RETRY_WATCHDOG=1 CLAUDE_CODE_MAX_RETRIES=10" in text) == (cli == "claude")
    assert calls[-1]["command"].startswith("for marker in ") and "time-cap-reached.json" in calls[-1]["command"]
    for index, call in enumerate(calls):
        path = tmp_path / f"{index}.sh"
        path.write_text(call["command"])
        assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_time_cap_marker_raises_like_harbors_timeout(monkeypatch):
    monkeypatch.setenv("RELEASE_CONTINUATION", "continue")
    monkeypatch.setenv("RELEASE_TASK", "ale-bench")
    monkeypatch.setenv("RELEASE_LIMIT_RESUME", "1")
    monkeypatch.setenv("RELEASE_AGENT_TIMEOUT", "3600")
    sys.modules.pop("agent", None)
    agent_module = importlib.import_module("agent")
    agent = agent_module.CodexAgent(logs_dir=ROOT / ".cache" / "render-logs", model_name="m")

    class Environment:
        def __init__(self, stdout):
            self.stdout = stdout

        async def exec(self, command, **kwargs):
            return SimpleNamespace(return_code=0, stdout=self.stdout, stderr="")

    asyncio.run(agent._raise_if_time_cap_reached(Environment(""), ["/logs/agent/s"]))
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(agent._raise_if_time_cap_reached(Environment("/logs/agent/s/time-cap-reached.json\n"),
                                                     ["/logs/agent/s"]))


def run_py(*argv: str):
    sys.path.insert(0, str(ROOT))
    import run
    return run.prepare(run.parser().parse_args([*argv, "--dry-run"]))


def multiplier(cmd: list) -> float:
    return float(cmd[cmd.index("--agent-timeout-multiplier") + 1])


@pytest.mark.parametrize("cli", ["claude", "codex"])
def test_run_py_turns_it_on_by_default(cli):
    import run
    cmd, env, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", cli, "--model", "m", "--timeout", "7200")
    assert env["RELEASE_LIMIT_RESUME"] == "1" and env["RELEASE_LIMIT_WAIT_MAX"] == "172800"
    assert env["RELEASE_AGENT_TIMEOUT"] == "7200"
    off, off_env, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", cli, "--model", "m", "--timeout", "7200", "--no-limit-resume")
    assert not {"RELEASE_LIMIT_RESUME", "RELEASE_LIMIT_WAIT_MAX", "RELEASE_AGENT_TIMEOUT"} & set(off_env)
    # Harbor's agent timeout: --timeout, plus 48 h of waiting and the grace period.
    assert multiplier(cmd) / multiplier(off) == pytest.approx((7200 + 172800 + run.LIMIT_RESUME_GRACE_SECONDS) / 7200)
    cmd, env, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", cli, "--model", "m", "--timeout", "7200", "--limit-wait-max", "1.5")
    assert env["RELEASE_LIMIT_WAIT_MAX"] == "5400"
    assert multiplier(cmd) / multiplier(off) == pytest.approx((7200 + 5400 + run.LIMIT_RESUME_GRACE_SECONDS) / 7200)
    # With --resources-tool and no limit resumption, the `resources` command still gets --timeout.
    _, env, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", cli, "--model", "m", "--timeout", "7200", "--no-limit-resume",
                       "--resources-tool")
    assert env["RELEASE_AGENT_TIMEOUT"] == "7200" and "RELEASE_LIMIT_RESUME" not in env


def test_resources_time_excludes_waiting(tmp_path):
    sys.modules.pop("resources_tool", None)
    resources = importlib.import_module("resources_tool")
    home = tmp_path / "s" / "codex-home"
    home.mkdir(parents=True)
    token_budget = resources.token_budget_module((ROOT / "token_budget.py").read_text())
    config = {"cli": "codex", "homes": [str(home)], "budget": None, "metric": "weighted", "timeout": 3600,
              "started": 1000.0}
    assert resources.measure(config, str(home), token_budget, 2000.0)["elapsed"] == 1000.0
    (home.parent / "limit-wait.json").write_text(json.dumps({"waited_seconds": 400}))
    numbers = resources.measure(config, str(home), token_budget, 2000.0)
    assert numbers["elapsed"] == 600.0 and numbers["time_left"] == 3000.0
