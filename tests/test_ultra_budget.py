"""Ultra mode at the BrowseComp runs' settings: the pinned
Codex 0.162.1 against the local mock of the model API (tests/ultra_mock_api.py), as tests/test_ultra.py's
end-to-end test, but at N = 4 (root + 3 helpers) and with the run stopped by the token budget and by the time cap
while helpers are still working.

Every mock response is 100 uncached input + 10 output tokens: 35 weighted (test_harness_guard.WEIGHTED_PER_RESPONSE).
No real model is called (the mock is the only provider; HTTPS_PROXY points at a closed port).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


from cli_render import INSTRUCTION
from test_ultra import CODEX, UltraMock, developer_texts, load_agent, needs_codex, outputs, ultra_env
import ultra

PER_RESPONSE = 35


def spawn(name: str, message: str | None = None) -> dict:
    return {"type": "call", "namespace": "collaboration", "name": "spawn_agent",
            "arguments": {"task_name": name, "message": message or f"HELPER {name}: reply ready",
                          "fork_turns": "none"}}


WAIT = {"type": "call", "namespace": "collaboration", "name": "wait_agent", "arguments": {"timeout_ms": 120000}}


def run_agent(tmp_path: Path, plans: dict, agents: int, extra: dict, join: float = 300):
    """One ultra agent run against the mock; returns (mock, local, errors, seconds)."""
    from test_harness_guard import GUARD, LIMIT, Local

    mock_api = UltraMock(tmp_path, plans)
    provider = {"name": "mock", "base_url": f"http://127.0.0.1:{mock_api.port}/v1", "wire_api": "responses",
                "env_key": "CODEX_API_KEY", "supports_websockets": "false", "request_max_retries": "0",
                "stream_max_retries": "0"}
    args = ["-c", 'model_provider="mock"']
    for key, value in provider.items():
        args += ["-c", f"model_providers.mock.{key}={value if value in ('false', '0') else json.dumps(value)}"]
    hook = tmp_path / "hook.sh"
    hook.write_text("task_cli_args=(" + " ".join(f"'{arg}'" for arg in args) + ")\n")
    home = tmp_path / "home"
    home.mkdir()
    local = Local(tmp_path, {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(home),
                             "HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
                             "NO_PROXY": "127.0.0.1,localhost", "RELEASE_TASK_CLI_HOOK": str(hook),
                             "META_DUO_CODEX_API_RETRY_BASE_SLEEP": "2"})
    env = {**ultra_env(agents, {**LIMIT, **GUARD, **extra}), "RELEASE_CONTINUATIONS": "0",
           "META_DUO_CONTAINER_CODEX_BINARY": str(CODEX), "CODEX_API_KEY": "sk-mock"}
    errors: list = []

    def main() -> None:
        try:
            module = load_agent(env)
            with mock.patch.dict(os.environ, env):
                agent = module.CodexCollaborationAgent(logs_dir=local.agent_dir, model_name="gpt-6.1-sol")
                asyncio.run(agent.run(INSTRUCTION, local, SimpleNamespace()))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    started = time.time()
    thread = threading.Thread(target=main, daemon=True)
    thread.start()
    thread.join(join)
    return mock_api, local, errors, time.time() - started, thread


def cleanup(mock_api, local) -> None:
    import harness_guard
    from test_harness_guard import kill
    mock_api.stop()
    state = harness_guard.read_json(str(local.guard_dir / "state" / "keeper.json")) or {}
    if state.get("pid") and harness_guard.alive(state["pid"], state.get("start")):
        kill(state["pid"])
    kill(harness_guard.agent_processes("CODEX_HOME", str(local.agent_dir / "codex-single-session/codex-home")))


def thread_requests(mock_api) -> dict[str, list]:
    found: dict[str, list] = {}
    for entry in map(json.loads, mock_api.log.read_text().splitlines()):
        found.setdefault(entry["agent"], []).append(entry)
    return found


def report(name: str, record: dict) -> None:
    out = Path(os.environ.get("ULTRA_MAC_REPORT_DIR", "/tmp"))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


@needs_codex
def test_n4_is_root_plus_three_helpers_at_effort_high_with_every_token_counted(tmp_path):
    """N = 4: the root spawns w1, w2, w3 (all accepted), then w4 while they run, which Codex refuses (4 threads at
    once, the root included); then waits. Mode message verbatim, slots 4, cap and wait lines, effort high on every
    request of every thread, tokens of all four rollouts counted once."""
    import token_budget
    plans = {"root": [spawn("w1"), spawn("w2"), spawn("w3"), spawn("w4"), WAIT, {"type": "text", "text": "done"}],
             **{f"w{i}": [{"type": "text", "text": f"w{i} ready", "after": {"agent": "root", "requests": 5}}]
                for i in (1, 2, 3)}}
    mock_api, local, errors, seconds, thread = run_agent(tmp_path, plans, 4, {"RELEASE_TOKEN_BUDGET": "4000000"})
    try:
        assert not thread.is_alive(), "the run did not end"
        assert not errors, repr(errors)
        by = thread_requests(mock_api)
        root = [entry for entry in by["root"]]
        texts = developer_texts(root[0])
        assert any(ultra.PROACTIVE_MODE_TEXT in text for text in texts)
        assert not any("Do not spawn sub-agents unless" in text for text in texts)
        role = next(text for text in texts if "You are `/root`, the primary agent" in text)
        slots = re.search(r"There are (\d+) available concurrency slots", role)
        assert slots and slots.group(1) == "4", role[:400]
        user = json.dumps([item for item in root[0]["body"]["input"] if item.get("role") == "user"])
        assert "At most 4 agents, you included, can work on this task at the same time." in user
        assert ultra.WAIT_LINE in user
        spawn_outputs = [" ".join(outputs(root[i])) for i in (1, 2, 3, 4)]
        assert not any("agent thread limit reached" in text for text in spawn_outputs[:3]), spawn_outputs[:3]
        assert "agent thread limit reached" in spawn_outputs[3], spawn_outputs[3]
        assert "w4" not in by
        efforts = {name: sorted({str((entry["body"].get("reasoning") or {}).get("effort")) for entry in entries})
                   for name, entries in by.items()}
        for name, values in efforts.items():
            assert set(values) <= {"high", "None"}, (name, values)
        session = local.agent_dir / "codex-single-session"
        totals = token_budget.summed(token_budget.session_totals("codex", session / "codex-home"))
        requests = sum(len(entries) for entries in by.values())
        assert totals["output_tokens"] == 10 * requests
        assert round(token_budget.weighted_tokens(totals)) == PER_RESPONSE * requests
        config = (session / "codex-home" / "config.toml").read_text()
        assert "max_concurrent_threads_per_session = 4" in config
        report("n4", {"requests_by_thread": {k: len(v) for k, v in by.items()}, "efforts": efforts,
                      "slots_in_role_message": slots.group(1), "fourth_spawn_refused": True,
                      "counted_output_tokens": totals["output_tokens"], "counted_weighted": token_budget.weighted_tokens(totals),
                      "seconds": round(seconds, 1)})
    finally:
        cleanup(mock_api, local)


@needs_codex
def test_the_token_budget_stops_root_and_helpers_counting_the_helpers(tmp_path):
    """Budget 1,500 weighted. The root makes only 5 requests (175 weighted) and then waits; the three helpers each
    plan 40 one-second shell steps (35 weighted a request). Only the helpers' tokens can reach the budget, so the run
    stops on budget long before the helpers finish (which would be about 4,400 weighted)."""
    import token_budget
    budget = 1500
    plans = {"root": [spawn("w1"), spawn("w2"), spawn("w3"), WAIT, WAIT, WAIT, WAIT, {"type": "text", "text": "done"}],
             **{f"w{i}": [{"type": "shell", "command": "sleep 1"}] * 40 + [{"type": "text", "text": "done"}]
                for i in (1, 2, 3)}}
    mock_api, local, errors, seconds, thread = run_agent(tmp_path, plans, 4, {"RELEASE_TOKEN_BUDGET": str(budget)})
    try:
        assert not thread.is_alive(), "the run did not end"
        assert not errors, repr(errors)
        session = local.agent_dir / "codex-single-session"
        marker = session / "token-budget-exhausted.json"
        assert marker.is_file(), "the budget watcher did not stop the run"
        stop = json.loads(marker.read_text())
        by = thread_requests(mock_api)
        helper_requests = sum(len(v) for k, v in by.items() if k != "root")
        assert helper_requests < 3 * 41, helper_requests          # the helpers were cut off
        assert PER_RESPONSE * len(by["root"]) < budget             # the root alone never reached it
        assert stop["weighted_tokens"] >= budget
        per_session = token_budget.session_totals("codex", session / "codex-home")
        totals = token_budget.summed(per_session)
        assert len(per_session) == 4                               # root + 3 helper rollouts
        # Every answered request is counted, except those still in flight when the watcher killed the process
        # (at most one per thread: answered by the mock, never written to a rollout).
        served = sum(len(v) for v in by.values())
        counted = round(token_budget.weighted_tokens(totals))
        assert PER_RESPONSE * (served - 4) <= counted <= PER_RESPONSE * served, (counted, served)
        overshoot = token_budget.weighted_tokens(totals) - budget
        report("budget", {"budget": budget, "stopped_at_weighted": stop["weighted_tokens"],
                          "final_weighted": token_budget.weighted_tokens(totals), "overshoot": overshoot,
                          "requests_served_by_mock": served, "requests_counted": counted // PER_RESPONSE,
                          "rollouts_counted": len(per_session),
                          "requests_by_thread": {k: len(v) for k, v in by.items()}, "seconds": round(seconds, 1)})
    finally:
        cleanup(mock_api, local)


@needs_codex
def test_the_time_cap_stops_root_and_helpers(tmp_path):
    """--timeout 25 s of working time (RELEASE_AGENT_TIMEOUT, enforced by the runner's limit watcher): the root waits
    on three helpers whose 100 two-second shell steps would take over 3 minutes; the run ends at the cap and no
    Codex process of the agent is left."""
    import harness_guard
    cap = 25
    plans = {"root": [spawn("w1"), spawn("w2"), spawn("w3")] + [WAIT] * 10 + [{"type": "text", "text": "done"}],
             **{f"w{i}": [{"type": "shell", "command": "sleep 2"}] * 100 + [{"type": "text", "text": "done"}]
                for i in (1, 2, 3)}}
    mock_api, local, errors, seconds, thread = run_agent(
        tmp_path, plans, 4, {"RELEASE_TOKEN_BUDGET": "4000000", "RELEASE_AGENT_TIMEOUT": str(cap)}, join=240)
    try:
        assert not thread.is_alive(), "the run did not end"
        # The runner ends a run at its working-time cap with TimeoutError, which Harbor records as the agent's
        # timeout (AgentTimeoutError) and then runs the verifier, as at its own timeout.
        assert len(errors) == 1 and isinstance(errors[0], TimeoutError), repr(errors)
        assert "used their working time (25 s" in str(errors[0])
        session = local.agent_dir / "codex-single-session"
        assert (session / "time-cap-reached.json").is_file(), sorted(p.name for p in session.iterdir())
        by = thread_requests(mock_api)
        helper_requests = sum(len(v) for k, v in by.items() if k != "root")
        assert helper_requests < 3 * 50, helper_requests
        left = harness_guard.agent_processes("CODEX_HOME", str(session / "codex-home"))
        assert not left, f"{len(left)} Codex processes still running"
        assert seconds < cap + 90, seconds
        report("timecap", {"cap_seconds": cap, "run_seconds": round(seconds, 1), "error": str(errors[0]),
                           "requests_by_thread": {k: len(v) for k, v in by.items()}, "processes_left": len(left)})
    finally:
        cleanup(mock_api, local)
