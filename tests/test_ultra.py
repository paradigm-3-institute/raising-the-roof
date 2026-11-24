"""Ultra mode (ultra.py, multiagent_protocols/ultra.py, agent.py, run.py, frontier_swe.py, ale_bench_task.py,
scored_artifact.py, perf_takehome_task.py; docs/setups.md): everything testable
without Docker or a model, and one run of the pinned Codex 0.162.1 against a local mock of the model API (the
cap, the mode message, the effort and the token count, as Codex itself applies them).
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import re
import shlex
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
from test_cli_agents import run_py
from test_frontier_swe import INSTRUCTION as FSWE_INSTRUCTION, container, log, pinned, upstream  # noqa: F401
from test_ale_bench import LIBRARIES, NAMES, PROFILES, facts, synthetic  # noqa: F401 (fixture)
from test_scored_artifact import fake_takehome, load, scorer_config, server  # noqa: F401 (fixtures)

sys.path.insert(0, str(ROOT))
import ale_bench_task  # noqa: E402
import frontier_swe  # noqa: E402
import perf_takehome_task  # noqa: E402
import scored_artifact  # noqa: E402
import ultra  # noqa: E402

CODEX_SRC = Path(os.environ.get("CODEX_SRC", ROOT / ".cache/codex-src/codex-rs"))  # a checkout of rust-v0.162.1
CODEX = ROOT / ".cache/codex-0.162.1/bin/codex"
OLD_CODEX = ROOT / ".cache/codex-0.160.0/bin/codex"
MAIN_CLONE = ROOT / ".cache/frontier-swe-v2"
needs_codex = pytest.mark.skipif(not CODEX.is_file(), reason="pinned Codex 0.162.1 release missing")
PROACTIVE_SHA256 = "d8fc778b064ebe519f4127812a897a61605000754d56140eb93c099ad26ec68a"

# The git-to-zig prompt for N = 2, our part of it:
# Codex adds its role message and the proactive mode message.
# GITZIG_TASK stands for git-to-zig's own instruction.md, which the test reads from the FrontierSWE clone.
GITZIG_TASK = "<git-to-zig's instruction.md>"
GITZIG_N2 = """\
At most 2 agents, you included, can work on this task at the same time.
Your helpers stop when you end your turn, so wait for them to finish before you end it.

<git-to-zig's instruction.md>

## Working copy and submissions

/app is the graded workspace: when the run ends, the task's verifier grades /app (the exact paths are in /opt/fswe/config.json).
Do not edit /app directly. Every agent, you and each helper, works in its own copy of the graded files:

    workcopy NAME    # copies /app's graded files to /work/NAME once; prints the path

Give each agent its own NAME. Edit, build and test only in your copy, and keep build outputs inside it. Task tools
that name /app paths act on /app: to run one on your copy, copy the tool elsewhere and
change those paths.

When your copy beats the version in /app, promote it from inside your copy:

    submit --score S --note "what changed"

S is your copy's result on the task's own self-check, where higher is better (give the
negative of a lower-is-better number). submit replaces /app's graded files with your
copy's only if S is at least the score of the version /app holds; --force skips that
check. Every call is logged in /logs/agent/frontier-swe/submissions.jsonl, which all
agents read. To build on another agent's version, copy its files from /app or from its
/work directory into your copy.

Only what /app holds when the run ends is graded, and the run can end without warning:
submit as soon as your copy beats the version in /app, and again after every improvement.
The verifier's scores are not shown during the run.

You have been given enough resources (tokens and time) to keep working on this task for a long time. Check your remaining resources at any time by running `resources` in your shell; it counts your helpers too."""


def load_agent(env: dict):
    with mock.patch.dict(os.environ, env):
        sys.modules.pop("agent", None)
        return importlib.import_module("agent")


def ultra_env(agents: int = 2, extra: dict | None = None, task: str = "ale-bench") -> dict:
    return {**BASE_ENV, "RELEASE_TASK": task, "MULTIAGENT_N_AGENT": "1",
            "MULTIAGENT_PROTOCOL": "multiagent_protocols.ultra:MultiAgentProtocol", ultra.ENV: str(agents),
            "MULTIAGENT_AGENT_EFFORT": "high",
            "RELEASE_CONTINUATION": (ROOT / "benchmarks/ale-bench/continuation.txt").read_text().rstrip("\n"),
            **(extra or {})}


def render(tmp_path: Path, env: dict, cls: str = "CodexCollaborationAgent", instruction: str = INSTRUCTION,
           **kwargs) -> tuple[object, list[dict]]:
    module = load_agent(env)
    with mock.patch.dict(os.environ, env), mock.patch("secrets.token_urlsafe", lambda n: "s" * 32):
        agent = getattr(module, cls)(logs_dir=tmp_path, model_name="gpt-6.1-sol", **kwargs)
        environment = FakeEnvironment()

        async def go() -> None:
            await agent.install(environment)
            await agent.run(instruction, environment, SimpleNamespace())

        asyncio.run(go())
    return agent, environment.calls


def first_prompt(calls: list[dict]) -> str:
    setup = next(call["command"] for call in calls if "codex-prompt.txt" in call["command"])
    return shlex.split(re.search(r"printf %s ('(?:[^']|'\"'\"')*') > ", setup).group(1))[0]


# ---- ultra.py ---------------------------------------------------------------------------------------------

def rust_str(path: Path, name: str) -> str:
    found = re.search(rf'const {name}: &str = "((?:[^"\\]|\\.)*)";', path.read_text())
    return found.group(1).encode().decode("unicode_escape")


def test_the_mode_message_is_codexs_proactive_text_verbatim():
    assert hashlib.sha256(ultra.PROACTIVE_MODE_TEXT.encode()).hexdigest() == PROACTIVE_SHA256
    source = CODEX_SRC / "prompts/src/model_messages/multi_agent.rs"
    if not source.is_file():
        pytest.skip("Codex 0.162.1 source missing")
    assert ultra.PROACTIVE_MODE_TEXT == rust_str(source, "PROACTIVE_MULTI_AGENT_MODE_TEXT")


def test_the_config_is_the_harness_config_plus_the_cap_and_the_mode_message():
    module = load_agent(ultra_env(4))
    base = tomllib.loads(module.CODEX_CONFIG_TOML)
    found = tomllib.loads(ultra.codex_config(module.CODEX_CONFIG_TOML, 4))
    assert found["features"].pop("multi_agent_v2") == {
        "enabled": True, "max_concurrent_threads_per_session": 4, "multi_agent_mode_hint_text": ultra.PROACTIVE_MODE_TEXT}
    assert base["features"].pop("multi_agent_v2") is False
    assert found == base
    with pytest.raises(ValueError, match="no longer applies"):
        ultra.codex_config("x = 1\n", 2)
    for bad in (1, 0, "x", ""):
        with pytest.raises(ValueError):
            ultra.check_agents(bad)
    assert ultra.prompt("Task.", 8) == ("At most 8 agents, you included, can work on this task at the same time.\n"
                                        "Your helpers stop when you end your turn, so wait for them to finish before "
                                        "you end it.\n\nTask.")


def sh(script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["sh", str(script), *args], cwd=cwd, capture_output=True, text=True)


def test_workcopy_needs_a_name_and_submit_runs_from_inside_a_copy(tmp_path):
    work = tmp_path.resolve() / "work"
    for directory in ("alpha/src", ".fswe", "beta.partial"):
        (work / directory).mkdir(parents=True)
    workcopy, submit = tmp_path / "workcopy", tmp_path / "submit"
    workcopy.write_text(ultra.workcopy_script(str(work), 'echo "copy $name"', "test"))
    submit.write_text(ultra.submit_script(str(work), 'echo "submit $name $*"', "test"))
    assert sh(workcopy, "alpha", cwd=tmp_path).stdout == "copy alpha\n"
    assert sh(workcopy, "w-2_b", cwd=tmp_path).stdout == "copy w-2_b\n"
    missing = sh(workcopy, cwd=tmp_path)
    assert missing.returncode == 2 and "usage: workcopy NAME" in missing.stderr
    assert "existing copy: alpha" in missing.stderr and ".fswe" not in missing.stderr and "beta" not in missing.stderr
    for bad in ("", ".x", "-x", "_x", "a/b", "a b", "x.partial", "a" * 65, "a;b", "$(id)"):
        assert sh(workcopy, bad, cwd=tmp_path).returncode == 2, bad
    assert sh(workcopy, "a", "b", cwd=tmp_path).returncode == 2
    inside = sh(submit, "--score", "3", "--note", "x y", cwd=work / "alpha" / "src")
    assert inside.returncode == 0 and inside.stdout == "submit alpha --score 3 --note x y\n"
    for outside in (tmp_path, work, work / ".fswe", work / "beta.partial"):
        result = sh(submit, "--score", "3", cwd=outside)
        assert result.returncode == 2 and "from inside your copy" in result.stderr, outside
        assert 'cd "$(workcopy NAME)"' in result.stderr


# ---- FrontierSWE --------------------------------------------------------------------------------------------

def test_the_harness_text_is_the_teams_with_named_copies():
    graded = [{"source": "/app/", "exclude": []}]
    found = frontier_swe.harness_text(2, graded, False, ultra_mode=True)
    expected = GITZIG_N2.split("## Working copy and submissions", 1)[1].rsplit("\n\nYou have been given", 1)[0]
    assert found == "## Working copy and submissions" + expected + "\n"
    # The same text for every N: the cap line says N.
    assert frontier_swe.harness_text(8, graded, False, ultra_mode=True) == found
    assert "$AGENT_ID" not in found and "{n}" not in found
    exposed = frontier_swe.harness_text(4, graded, True, ultra_mode=True)
    assert exposed.endswith(frontier_swe.VERIFIER_EXPOSED + "\n")
    # Team and solo texts as before.
    assert frontier_swe.harness_text(2, graded, False) == frontier_swe.harness_text(2, graded, False, ultra_mode=False)
    assert "2 agents share this container" in frontier_swe.harness_text(2, graded, False)


def test_an_ultra_task_has_the_teams_resources_and_named_copy_tools(pinned, tmp_path):
    common = dict(clone=pinned, cli="codex", timeout=7200)
    built = frontier_swe.build_task("demo-cpu", tmp_path / "u", agents=4, ultra=True, **common)
    team = frontier_swe.build_task("demo-cpu", tmp_path / "t", agents=4, **common)
    solo = frontier_swe.build_task("demo-cpu", tmp_path / "s", agents=1, **common)
    config = tomllib.loads(built.files["task.toml"])
    assert (config["environment"]["cpus"], config["environment"]["memory_mb"]) == (8, 16384)  # 4 x the task's
    assert built.files["task.toml"] == team.files["task.toml"]
    assert built.files["instruction.md"] == FSWE_INSTRUCTION.rstrip("\n") + "\n\n" + \
        frontier_swe.harness_text(4, built.upstream.graded, False, ultra_mode=True)
    assert "environment/fswe/bin/workcopy" not in built.copies and "environment/fswe/bin/submit" not in built.copies
    assert built.files["environment/fswe/bin/workcopy"] == frontier_swe.ultra_tools()["workcopy"]
    assert 'exec python3 /opt/fswe/fswe.py submit --agent "$name" "$@"' in built.files["environment/fswe/bin/submit"]
    assert json.loads(built.files["environment/fswe/config.json"])["ultra"] is True
    described = json.loads(built.files["frontier-swe.json"])
    assert described["mode"] == "ultra" and described["ultra"]["agents"] == 4
    assert "`workcopy NAME` prints its path" in built.continuation and "from inside it to promote" in built.continuation
    # Without ultra mode nothing changes: the team's and the solo task's files are the ones they always were.
    for other in (team, solo):
        assert "ultra" not in other.files["environment/fswe/config.json"]
        assert "mode" not in json.loads(other.files["frontier-swe.json"])
        assert other.copies["environment/fswe/bin/workcopy"] == frontier_swe.BENCH / "bin" / "workcopy"
        assert other.continuation == (frontier_swe.BENCH / "continuation.txt").read_text()
    task = built.write()
    for tool in ("workcopy", "submit"):
        assert os.access(task / "environment/fswe/bin" / tool, os.X_OK)
    with pytest.raises(ValueError, match="--agents 2 or more"):
        frontier_swe.build_task("demo-cpu", tmp_path / "x", agents=1, ultra=True, **common)


def test_the_named_copies_drive_fswe_py(container):
    """`workcopy NAME` and `submit` from inside the copy, on the real fswe.py: every helper's submission is
    logged under its copy's NAME (the helpers all run with the agent's one AGENT_ID)."""
    tool, app, tmp_path = container
    tools = frontier_swe.ultra_tools(tools_mount=str(tmp_path / "opt"), work_root=str(tmp_path.resolve() / "work"))
    for name, text in tools.items():
        (tmp_path / name).write_text(text)
    env = {**os.environ, "AGENT_ID": "single", "CODEX_HOME": str(tmp_path / "codex-home")}

    def run(*argv: str, cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run(["sh", *argv], cwd=cwd, env=env, capture_output=True, text=True)

    w1 = Path(run(str(tmp_path / "workcopy"), "w1", cwd=tmp_path).stdout.strip())
    w2 = Path(run(str(tmp_path / "workcopy"), "w2", cwd=tmp_path).stdout.strip())
    assert (w1.name, w2.name) == ("w1", "w2") and w1.parent.resolve() == (tmp_path / "work").resolve()
    (w2 / "Sources/vsim/main.swift").write_text("w2 version")
    done = run(str(tmp_path / "submit"), "--score", "0.5", "--note", "w2's", cwd=w2 / "Sources")
    assert done.returncode == 0, done.stdout + done.stderr
    assert (app / "Sources/vsim/main.swift").read_text() == "w2 version"
    lower = run(str(tmp_path / "submit"), "--score", "0.4", cwd=w1)
    assert lower.returncode == 1 and "by w2" in lower.stdout
    outside = run(str(tmp_path / "submit"), "--score", "1", cwd=tmp_path)
    assert outside.returncode == 2
    records = log(tmp_path)
    assert [(r["agent_id"], Path(r["source"]).name, r["status"]) for r in records] == [
        ("w2", "w2", "promoted"), ("w1", "w1", "rejected")]


@pytest.mark.skipif(not (MAIN_CLONE / "tasks/git-to-zig/instruction.md").is_file(),
                    reason="the FrontierSWE clone with git-to-zig is missing")
def test_the_git_to_zig_prompt_is_the_agreed_one(tmp_path):
    built = frontier_swe.build_task("git-to-zig", tmp_path, clone=MAIN_CLONE, agents=2, cli="codex",
                                    ultra=True, fetch=False)
    env = ultra_env(2, {"RELEASE_RESOURCES_TOOL": "1", "RELEASE_AGENT_TIMEOUT": "172800"}, task="frontier-swe")
    _, calls = render(tmp_path / "logs", env, instruction=built.files["instruction.md"])
    task = (MAIN_CLONE / "tasks/git-to-zig/instruction.md").read_text().rstrip("\n")
    expected = GITZIG_N2.replace(GITZIG_TASK, task)
    assert first_prompt(calls) == expected
    for agents in (4, 8):
        env = ultra_env(agents, {"RELEASE_RESOURCES_TOOL": "1", "RELEASE_AGENT_TIMEOUT": "172800"}, task="frontier-swe")
        _, calls = render(tmp_path / f"logs{agents}", env, instruction=built.files["instruction.md"])
        assert first_prompt(calls) == expected.replace("At most 2 agents", f"At most {agents} agents")


# ---- ALE-Bench ------------------------------------------------------------------------------------------------

def test_the_ale_bench_text_is_the_teams_with_named_copies_and_the_teams_cpus():
    team = ale_bench_task.instruction(facts(), "cpp20", False, "official", 4, 8, PROFILES, NAMES, LIBRARIES)
    found = ale_bench_task.instruction(facts(), "cpp20", False, "official", 4, 8, PROFILES, NAMES, LIBRARIES,
                                       ultra_mode=True)
    assert "Every agent, you and each helper, works in its own copy: `workcopy NAME` creates" in found
    assert "the working copy /work/NAME once and prints its path. Give each agent its own NAME. Work" in found
    assert "You and your helpers share its 8\nCPUs: keep to 2 CPU-heavy processes per agent at a time." in found
    assert "- `submit`, run from inside your copy, sends its Main.cpp to the judge" in found
    assert "the submission (of any agent) with" in found and "Every copy's Main.cpp is also submitted" in found
    assert "yours and your helpers'" in found and "another agent's accepted submission" in found
    for gone in ("$AGENT_ID", "4 agents share", "teammate", "The 4 of you", "{"):
        assert gone not in found, gone
    # Every other line is the team's.
    changed = {line for line in found.splitlines()} - {line for line in team.splitlines()}
    assert len(changed) == 9, sorted(changed)
    outputs = ale_bench_task.instruction(facts("ahc027"), "cpp20", True, "outputs", 2, 4, PROFILES, NAMES,
                                         LIBRARIES, ultra_mode=True)
    assert "- `submit`, run from inside your copy, sends its out/ to the judge" in outputs
    assert "of any agent; every copy's out/ is" in outputs and "{" not in outputs
    # Team and solo texts as before.
    assert ale_bench_task.instruction(facts(), "cpp20", False, "official", 1, 2, PROFILES, NAMES, LIBRARIES) == \
        ale_bench_task.instruction(facts(), "cpp20", False, "official", 1, 2, PROFILES, NAMES, LIBRARIES,
                                   ultra_mode=False)


def test_an_ultra_ale_bench_task_wraps_workcopy_and_submit(synthetic):
    program = synthetic.tmp / "Main.cpp"
    program.write_text("int main() {}")
    common = dict(cpus=8, memory_mb=4 * 6144, lite=False, language="cpp20", variant="official", judge_cpus=[12, 14],
                  private_lane=True, scorer="s", profiles=PROFILES, names=NAMES, libraries=LIBRARIES,
                  data=synthetic.data, solution=program)
    built = ale_bench_task.build_task("ahc999", synthetic.tmp / "u", agents=4, ultra_mode=True, **common)
    team = ale_bench_task.build_task("ahc999", synthetic.tmp / "t", agents=4, **common)
    workcopy, submit = (built.files[f"client/bin/{name}"].decode() for name in ("workcopy", "submit"))
    assert 'AGENT_ID="$name" ALECTL_COMMAND=workcopy exec python3 /opt/ale-bench/client/alectl.py' in workcopy
    assert 'AGENT_ID="$name" ALECTL_COMMAND=submit exec python3 /opt/ale-bench/client/alectl.py "$@"' in submit
    for name in ("scores", "fetch-submission", "ale-local"):
        assert built.files[f"client/bin/{name}"] == team.files[f"client/bin/{name}"]
    assert team.files["client/bin/workcopy"] == ale_bench_task.wrapper("workcopy").encode()
    assert built.files["task.toml"].split(b"SCORER_TOKEN")[0] == team.files["task.toml"].split(b"SCORER_TOKEN")[0]
    assert built.description["mode"] == "ultra" and "mode" not in team.description
    assert b'dest=$(workcopy oracle)\ncp /solution/Main.cpp "$dest/Main.cpp"\ncd "$dest"\nsubmit\n' in \
        built.files["solution/solve.sh"]
    assert team.files["solution/solve.sh"].endswith(b'dest=$(workcopy)\ncp /solution/Main.cpp "$dest/Main.cpp"\nsubmit\n')
    # The verifier submits every copy under /work (finalize.py), so each helper's final copy counts.
    finalize = (ROOT / "benchmarks/scored-artifact/tests/finalize.py").read_text()
    assert "for directory in sorted(p for p in work.iterdir() if p.is_dir()" in finalize


def test_alectl_works_in_the_copy_its_name_gives(tmp_path):
    """alectl.py as ultra mode's `workcopy NAME` runs it: AGENT_ID is the copy's NAME."""
    starter = tmp_path / "starter"
    (starter / "problem").mkdir(parents=True)
    (starter / "problem/statement.md").write_text("S")
    config = tmp_path / "task.json"
    config.write_text(json.dumps({"starter": str(starter), "work_root": str(tmp_path / "work"),
                                  "artifact_files": ["Main.cpp"], "scorer_urls": ["http://127.0.0.1:9"]}))
    env = {**os.environ, "SCORED_ARTIFACT_CONFIG": str(config), "AGENT_ID": "w3", "ALECTL_COMMAND": "workcopy"}
    made = subprocess.run([sys.executable, str(ROOT / "benchmarks/ale-bench/client/alectl.py")], env=env,
                          capture_output=True, text=True)
    assert made.returncode == 0 and Path(made.stdout.strip()) == tmp_path / "work" / "w3"
    assert (tmp_path / "work/w3/problem/statement.md").read_text() == "S"


def test_the_ale_bench_continuation_says_submit_from_inside_the_copy():
    original = (ale_bench_task.BENCH / "continuation.txt").read_text()
    built = ale_bench_task.build_job(Path("/tmp/u"), ["ahc014"], agents=4, dry_run=True, ultra=True)
    team = ale_bench_task.build_job(Path("/tmp/t"), ["ahc014"], agents=4, dry_run=True)
    assert team.continuation == original
    assert built.continuation == original.replace("and run `submit` to get a fresh public score.",
                                                  "and run `submit` from inside your copy to get a fresh public score.")
    assert built.continuation != original


class LiveScorer:
    """The scored-artifact layer's scorer (the HTTP API alectl.py and scorectl.py talk to, ALE-Bench's judge
    included) on a stand-in benchmark, and a task.json for the clients: the artifact is answer.txt, a number."""

    def __init__(self, config: dict, tmp: Path):
        self.scorer = server.Scorer(config)
        threading.Thread(target=self.scorer.worker, args=(0,), daemon=True).start()
        self.httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.handler(self.scorer, "tok"))
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        starter = tmp / "starter"
        starter.mkdir()
        (starter / "answer.txt").write_text("80")
        self.work = tmp.resolve() / "work"
        self.config = tmp / "task.json"
        self.config.write_text(json.dumps({
            "benchmark": "fake", "starter": str(starter), "work_root": str(self.work), "agents": 2,
            "artifact_files": ["answer.txt"], "scorer_urls": [f"http://127.0.0.1:{self.httpd.server_port}"],
            "metric_name": "cycles", "score_type": "maximize", "variant": "official", "submit_wait_sec": 60}))

    def records(self) -> list[dict]:
        with self.scorer.lock:
            return sorted((dict(r) for r in self.scorer.records.values()), key=lambda r: int(r["sid"]))

    def stop(self) -> None:
        self.httpd.shutdown()


def drive_named_copies(tmp_path: Path, live: LiveScorer, tools: dict[str, str], scores: list[str]) -> None:
    """The root's AGENT_ID (single) for every call, as in ultra mode: two named copies, a submission from inside
    each (one of another file), one from outside a copy (refused), `workcopy` without a NAME (refused)."""
    for name, text in tools.items():
        (tmp_path / name).write_text(text)
    env = {**os.environ, "AGENT_ID": "single", "SCORED_ARTIFACT_CONFIG": str(live.config)}

    def run(*argv: str, cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run(["sh", *argv], cwd=cwd, env=env, capture_output=True, text=True)

    w1 = Path(run(str(tmp_path / "workcopy"), "w1", cwd=tmp_path).stdout.strip())
    w2 = Path(run(str(tmp_path / "workcopy"), "w2", cwd=tmp_path).stdout.strip())
    assert (w1, w2) == (live.work / "w1", live.work / "w2")
    assert (w1 / "answer.txt").read_text() == "80" and (w1 / ".git").is_dir()
    assert Path(run(str(tmp_path / "workcopy"), "w1", cwd=tmp_path).stdout.strip()) == w1  # once
    (w2 / "answer.txt").write_text("20")
    (w2 / "sub").mkdir()
    done = run(str(tmp_path / "submit"), "--note", "w2's", cwd=w2 / "sub")
    assert done.returncode == 0 and "agent=w2" in done.stdout, done.stdout + done.stderr
    (tmp_path / "other.txt").write_text("40")
    other = run(str(tmp_path / "submit"), str(tmp_path / "other.txt"), cwd=w1)
    assert other.returncode == 0 and "agent=w1" in other.stdout, other.stdout + other.stderr
    outside = run(str(tmp_path / "submit"), cwd=tmp_path)
    assert outside.returncode == 2 and "from inside your copy" in outside.stderr
    missing = run(str(tmp_path / "workcopy"), cwd=tmp_path)
    assert missing.returncode == 2 and "existing copy: w1" in missing.stderr and "existing copy: w2" in missing.stderr
    assert not (live.work / "single").exists()
    assert [(r["agent_id"], r["metric"], r["note"]) for r in live.records()] == [("w2", 20, "w2's"), ("w1", 40, "")]
    listed = subprocess.run(scores, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert listed.returncode == 0 and " w2 " in listed.stdout and " w1 " in listed.stdout
    # The verifier submits every copy under /work as its NAME (finalize.py), so each helper's final copy counts.
    finalize = load("scored_artifact_finalize", ROOT / "benchmarks/scored-artifact/tests/finalize.py")
    (live.work / "w3.partial").mkdir()
    task = json.loads(live.config.read_text())
    assert [agent for agent, _, _ in finalize.final_copies(task, live.work)] == ["w1", "w2"]


def test_the_named_copies_drive_alectl(scorer_config, tmp_path):
    """Ultra mode's ALE-Bench `workcopy NAME` and `submit` on the real alectl.py: every submission is logged under
    its copy's NAME."""
    live = LiveScorer(scorer_config, tmp_path)
    client = ROOT / "benchmarks/ale-bench/client"
    try:
        tools = {name: ale_bench_task.ultra_wrapper(name, work_root=str(live.work), client_mount=str(client))
                 for name in ("workcopy", "submit")}
        drive_named_copies(tmp_path, live, tools, ["env", "ALECTL_COMMAND=scores", sys.executable,
                                                   str(client / "alectl.py"), "--all"])
        gitignore = (live.work / "w1" / ".gitignore").read_text()
        assert "out/" in gitignore  # alectl.py's own copy
    finally:
        live.stop()


# ---- The performance take-home (the scored-artifact layer) -------------------------------------------------------

def test_the_take_home_text_is_the_teams_with_named_copies():
    bench = perf_takehome_task.BENCHMARK
    team = bench.instruction(4, None)
    found = bench.ultra_instruction(4, None)
    assert found == bench.ultra_instruction(8, None) == bench.ultra_instruction(2, None)  # the cap line says N
    assert "Every agent, you and each helper, works in its own copy: `workcopy NAME` creates" in found
    assert "a copy of the take-home repository at /work/NAME once (with a git commit" in found
    assert "changes) and prints its path. Give each agent its own NAME. Work only in your copy;" in found
    assert "- `submit`, run from inside your copy, sends its perf_takehome.py to the scorer" in found
    assert "- The result of the run is the best valid submission of any agent." in found
    assert "yours and your helpers'" in found and "another agent's verified" in found
    for gone in ("$AGENT_ID", "agents share", "teammate", "{"):
        assert gone not in found, gone
    # Every other line is the team's.
    changed = set(found.splitlines()) - set(team.splitlines())
    assert len(changed) == 7 and len(found.splitlines()) == len(team.splitlines()), sorted(changed)
    with pytest.raises(ValueError, match="no longer applies"):
        scored_artifact.solo_text("changed text", perf_takehome_task.ULTRA_EDITS)


def test_an_ultra_take_home_task_has_the_teams_resources_and_wraps_workcopy_and_submit(fake_takehome, tmp_path):
    kernel = tmp_path / "kernel.py"
    kernel.write_text("kernel")
    common = dict(agents=4, cli="codex", codex_login=True, solution=kernel)
    built = scored_artifact.build_task(perf_takehome_task.BENCHMARK, tmp_path / "u", ultra=True, **common)
    team = scored_artifact.build_task(perf_takehome_task.BENCHMARK, tmp_path / "t", **common)
    for task in (built, team):  # 4 x the take-home's 2 CPUs and 6 GiB
        environment = tomllib.loads(task.task_toml)["environment"]
        assert (environment["cpus"], environment["memory_mb"]) == (8, 4 * 6144)
    assert built.task_toml.split("SCORER_TOKEN")[0] == team.task_toml.split("SCORER_TOKEN")[0]
    assert built.files["instruction.md"].decode() == perf_takehome_task.BENCHMARK.ultra_instruction(4, None)
    workcopy, submit = (built.files[f"client/bin/{name}"].decode() for name in ("workcopy", "submit"))
    assert workcopy == scored_artifact.ultra_wrapper("workcopy")
    assert 'AGENT_ID="$name" SCORECTL_COMMAND=workcopy exec python3 /opt/scored-artifact/scorectl.py' in workcopy
    assert 'AGENT_ID="$name" SCORECTL_COMMAND=submit exec python3 /opt/scored-artifact/scorectl.py "$@"' in submit
    for name in ("scores", "fetch-submission"):
        assert built.files[f"client/bin/{name}"] == team.files[f"client/bin/{name}"]
    assert team.files["client/bin/workcopy"] == scored_artifact.wrapper("workcopy").encode()
    for name in ("client/scorectl.py", "scorer/server.py", "tests/finalize.py", "environment/Dockerfile"):
        assert built.files[name] == team.files[name]
    assert (built.files["environment/docker-compose.yaml"].decode().replace(str(built.path), "T")
            == team.files["environment/docker-compose.yaml"].decode().replace(str(team.path), "T"))
    client = json.loads(built.files["client/task.json"])
    assert client == json.loads(team.files["client/task.json"])
    described = json.loads(built.files["scored-artifact.json"])
    assert described["mode"] == "ultra" and described["ultra"]["agents"] == 4
    assert "mode" not in json.loads(team.files["scored-artifact.json"])
    assert built.files["solution/solve.sh"].endswith(
        b'dest=$(workcopy oracle)\ncp /solution/perf_takehome.py "$dest/perf_takehome.py"\ncd "$dest"\nsubmit\n')
    assert team.files["solution/solve.sh"].endswith(
        b'dest=$(workcopy)\ncp /solution/perf_takehome.py "$dest/perf_takehome.py"\nsubmit\n')
    original = perf_takehome_task.BENCHMARK.continuation()
    assert team.continuation == original
    assert built.continuation == original.replace("and run `submit` to get a fresh graded score.",
                                                  "and run `submit` from inside your copy to get a fresh graded score.")
    task = built.write()
    for tool in ("workcopy", "submit"):
        assert os.access(task / "client/bin" / tool, os.X_OK)
    with pytest.raises(ValueError, match="--agents 2 or more"):
        scored_artifact.build_task(perf_takehome_task.BENCHMARK, tmp_path / "x", agents=1, ultra=True)


def test_the_named_copies_drive_scorectl(scorer_config, tmp_path):
    """Ultra mode's scored-artifact `workcopy NAME` and `submit` on the real scorectl.py: every submission is
    logged under its copy's NAME."""
    live = LiveScorer(scorer_config, tmp_path)
    client = scored_artifact.LAYER / "client"
    try:
        tools = {name: scored_artifact.ultra_wrapper(name, work_root=str(live.work), client_mount=str(client))
                 for name in ("workcopy", "submit")}
        drive_named_copies(tmp_path, live, tools, ["env", "SCORECTL_COMMAND=scores", sys.executable,
                                                   str(client / "scorectl.py"), "--all"])
    finally:
        live.stop()


def test_scored_artifact_benchmarks_without_named_copies_refuse_ultra():
    class Plain(scored_artifact.Benchmark):
        name = "plain"

    with pytest.raises(ValueError, match="plain has no ultra mode"):
        Plain().ultra_instruction(2, None)
    for name in scored_artifact.BENCHMARKS:
        bench = scored_artifact.benchmark(name)
        has = type(bench).ultra_instruction is not scored_artifact.Benchmark.ultra_instruction
        assert has == (name == "perf-takehome"), name


# ---- agent.py --------------------------------------------------------------------------------------------------

def test_one_codex_agent_with_the_cap_line_the_config_and_the_effort(tmp_path):
    env = ultra_env(4, {"RELEASE_HARNESS_GUARD": "1", "RELEASE_LIMIT_RESUME": "1", "RELEASE_TOKEN_BUDGET": "8000000",
                        "RELEASE_RESOURCES_TOOL": "1", "RELEASE_AGENT_TIMEOUT": "172800"})
    agent, calls = render(tmp_path, env)
    commands = [call["command"] for call in calls]
    # One runner, the single agent's: no team directory, no protocol text.
    assert any("codex-single-session" in c for c in commands) and not any("sessions-0-" in c for c in commands)
    assert "Follow this coordination protocol" not in "".join(commands) and "solo work protocol" not in "".join(commands)
    prompt = first_prompt(calls)
    assert prompt.startswith("At most 4 agents, you included, can work on this task at the same time.\n"
                             + ultra.WAIT_LINE + "\n\n" + INSTRUCTION)
    assert prompt.endswith("by running `resources` in your shell; it counts your helpers too.")
    runner = agent.last_runner_script
    config = runner.split("<<'TOML'\n", 1)[1].split("TOML\n", 1)[0]
    assert config == ultra.codex_config(sys.modules["agent"].CODEX_CONFIG_TOML, 4)
    assert ('${codex_effort:+-c "model_reasoning_effort=$codex_effort" '
            '-c "agents.default_subagent_reasoning_effort=$codex_effort"} ') in runner
    record = next(c for c in commands if "ultra.json" in c)
    assert '{"agents": 4, "effort": "high", "multi_agent_mode_hint_sha256": "' + PROACTIVE_SHA256 in record
    # One budget, one keeper unit, one `resources` home: the whole tree's.
    trial = next(c for c in commands if "> /usr/lib/hsv/conf.json" in c)
    conf = json.loads(re.search(r"printf %s '(\{\n.*?\n\})' > /usr/lib/hsv/conf\.json", trial, re.S).group(1))
    assert len(conf["units"]) == 1 and sorted(conf["units"][0]["watchers"]) == ["limit", "token"]
    resources = next(c for c in commands if "/usr/local/libexec/resources" in c and "printf %s" in c)
    assert resources.count("codex-single-session/codex-home") == 1


def test_without_ultra_nothing_changes(tmp_path):
    """The same agent without ultra.ENV and the protocol: no cap line, the harness's config, one effort flag."""
    env = {**ultra_env(2), "MULTIAGENT_PROTOCOL": "multiagent_protocols.antiherding:MultiAgentProtocol"}
    env.pop(ultra.ENV)
    agent, calls = render(tmp_path, env)
    assert first_prompt(calls) == INSTRUCTION
    runner = agent.last_runner_script or "".join(call["command"] for call in calls)
    assert "default_subagent_reasoning_effort" not in runner and "multi_agent_mode_hint_text" not in runner
    assert not any("ultra.json" in call["command"] for call in calls)


@pytest.mark.parametrize("change, message", [
    ({"MULTIAGENT_PROTOCOL": "multiagent_protocols.antiherding:MultiAgentProtocol"}, "needs both"),
    ({ultra.ENV: ""}, "needs both"),
    ({ultra.ENV: "1"}, "room for a helper"),
    ({"MULTIAGENT_N_AGENT": "2"}, "not a team"),
])
def test_half_a_setting_or_a_team_is_refused(tmp_path, change, message):
    env = {**ultra_env(2), **change}
    with pytest.raises(ValueError, match=message):
        render(tmp_path, env)


@pytest.mark.parametrize("cls, kwargs", [
    ("ClaudeCodeCollaborationAgent", {}),
])
def test_other_clis_refuse_ultra(tmp_path, cls, kwargs):
    with pytest.raises(ValueError, match="Codex's own sub-agents"):
        render(tmp_path, ultra_env(2), cls=cls, **kwargs)


# ---- run.py -------------------------------------------------------------------------------------------------------

SOL = ("--cli", "codex", "--codex-binary", ".cache/codex-0.162.1/bin/codex", "--model", "gpt-6.1-sol", "--effort", "high")


def test_run_py_runs_one_agent_with_the_cap_in_its_environment(pinned, monkeypatch):
    import run
    monkeypatch.setattr(run, "FRONTIER_SWE_TASKS", ("git-to-zig", "demo-cpu"))  # the fake upstream's task
    cmd, env, _ = run_py("frontier-swe", "--task", "demo-cpu", "--frontier-swe", str(pinned), *SOL, "--protocol",
                         "ultra", "--agents", "4", "--token-budget", "8000000", "--timeout", "7200", "--name", "u-dry")
    assert env["MULTIAGENT_N_AGENT"] == "1" and env[ultra.ENV] == "4"
    assert env["MULTIAGENT_PROTOCOL"] == "multiagent_protocols.ultra:MultiAgentProtocol"
    assert env["MULTIAGENT_AGENT_EFFORT"] == "high" and env["RELEASE_TOKEN_BUDGET"] == "8000000"
    assert cmd[cmd.index("--agent-import-path") + 1] == "agent:CodexCollaborationAgent"
    assert "`workcopy NAME` prints its path" in env["RELEASE_CONTINUATION"]
    _, plain, _ = run_py("frontier-swe", "--task", "demo-cpu", "--frontier-swe", str(pinned), *SOL, "--protocol",
                         "antiherding_matched", "--agents", "4", "--name", "t-dry")
    assert ultra.ENV not in plain and plain["MULTIAGENT_N_AGENT"] == "4"


def test_run_py_ale_bench_ultra_gets_the_teams_cpus():
    cmd, env, _ = run_py("ale-bench", "--problem", "ahc014", *SOL, "--protocol", "ultra", "--agents", "4",
                         "--name", "u-ale-dry")
    assert env["MULTIAGENT_N_AGENT"] == "1" and env[ultra.ENV] == "4"
    task = Path(cmd[cmd.index("-p") + 1])
    assert task.name == "u-ale-dry"
    job = ale_bench_task.build_job(Path("/tmp/u-ale-dry"), ["ahc014"], agents=4, dry_run=True, ultra=True)
    assert tomllib.loads(job.task_toml)["environment"]["cpus"] == 8 and job.description["mode"] == "ultra"
    assert job.description["resources"]["memory_mb"] == 4 * ale_bench_task.PER_AGENT_MEMORY_MB
    assert "run `submit` from inside your copy" in env["RELEASE_CONTINUATION"]
    assert "--override-cpus" not in cmd and "--override-memory-mb" not in cmd
    _, plain, _ = run_py("ale-bench", "--problem", "ahc014", *SOL, "--protocol", "antiherding_matched", "--agents",
                         "4", "--name", "t-ale-dry")
    assert ultra.ENV not in plain and plain["RELEASE_CONTINUATION"] == ale_bench_task.Job(
        Path("/x"), [], {}).continuation.rstrip("\n")


def test_run_py_perf_takehome_ultra_runs_one_agent_with_named_copies(fake_takehome, tmp_path):
    auth = tmp_path / "auth.json"
    auth.write_text("{}")
    common = (*SOL, "--trials", "1", "--token-budget", "8000000", "--resources-tool", "--codex-auth", str(auth))
    cmd, env, _ = run_py("perf-takehome", *common, "--protocol", "ultra", "--agents", "4", "--name", "u-perf-dry")
    assert env["MULTIAGENT_N_AGENT"] == "1" and env[ultra.ENV] == "4"
    assert env["MULTIAGENT_PROTOCOL"] == "multiagent_protocols.ultra:MultiAgentProtocol"
    assert "run `submit` from inside your copy" in env["RELEASE_CONTINUATION"]
    assert "--override-cpus" not in cmd and "--override-memory-mb" not in cmd
    assert not Path(cmd[cmd.index("-p") + 1]).exists()  # --dry-run writes nothing
    _, plain, _ = run_py("perf-takehome", *common, "--protocol", "antiherding_matched", "--agents", "4",
                         "--name", "t-perf-dry")
    assert ultra.ENV not in plain and plain["RELEASE_CONTINUATION"] == (
        ROOT / "benchmarks/perf-takehome/continuation.txt").read_text().rstrip("\n")


@pytest.mark.parametrize("argv, message", [
    (("--cli", "claude", "--model", "claude-sonnet-5-5", "--agents", "2"), "one Codex agent"),
    (("--team", "codex:gpt-6.1-sol,codex:gpt-6-luna", "--codex-binary", ".cache/codex-0.162.1/bin/codex"),
     "one Codex agent"),
    ((*SOL, "--agents", "1"), "room for a helper"),
    ((*SOL, "--agents", "2", "--team-limit-hold"), "pace a team"),
    ((*SOL, "--agents", "2", "--token-budget", "100", "--lockstep-gap", "10"), "pace a team"),
])
def test_run_py_refuses(argv, message):
    with pytest.raises(ValueError, match=message):
        run_py("ale-bench", "--problem", "ahc014", "--protocol", "ultra", *argv)


@pytest.mark.skipif(not OLD_CODEX.is_file(), reason="pinned Codex 0.160.0 release missing")
def test_run_py_refuses_an_older_codex():
    with pytest.raises(ValueError, match=r"needs Codex 0\.162\.1 or later"):
        run_py("ale-bench", "--problem", "ahc014", "--protocol", "ultra", "--cli", "codex", "--codex-binary",
               str(OLD_CODEX), "--model", "gpt-6.1-sol", "--effort", "high", "--agents", "2")


# ---- end to end: Codex 0.162.1 and its own helpers, against a mock of the model API ----------------------------------

class UltraMock:
    def __init__(self, tmp: Path, plans: dict):
        (tmp / "plans.json").write_text(json.dumps(plans))
        self.log = tmp / "requests.jsonl"
        ready = tmp / "port"
        self.process = subprocess.Popen([sys.executable, str(ROOT / "tests/ultra_mock_api.py"), "--plans",
                                         str(tmp / "plans.json"), "--log", str(self.log), "--ready", str(ready)])
        for _ in range(200):
            if ready.exists():
                break
            time.sleep(0.05)
        self.port = int(ready.read_text())

    def requests(self, thread: str) -> list:
        return [entry for entry in map(json.loads, self.log.read_text().splitlines()) if entry["agent"] == thread]

    def stop(self) -> None:
        self.process.terminate()
        self.process.wait(timeout=10)


def developer_texts(request: dict) -> list[str]:
    return [part.get("text", "") for item in request["body"]["input"]
            if item.get("type") == "message" and item.get("role") == "developer" for part in item.get("content", [])]


def outputs(request: dict) -> list[str]:
    return [item.get("output") if isinstance(item.get("output"), str) else json.dumps(item.get("output"))
            for item in request["body"].get("input", [])
            if item.get("type") in ("function_call_output", "custom_tool_call_output")]


@needs_codex
def test_codex_runs_its_helpers_under_the_cap_at_the_runs_effort(tmp_path):
    """N = 2: the root spawns w1 (fresh), then a second helper while w1 still runs, which Codex refuses (the cap
    counts the root), then waits for w1. The mode message, the role message's slots, every thread's effort and the
    one token budget over both rollouts, as Codex applies them."""
    from test_harness_guard import GUARD, LIMIT, Local, kill
    import harness_guard
    import token_budget

    mock_api = UltraMock(tmp_path, {
        "root": [{"type": "call", "namespace": "collaboration", "name": "spawn_agent",
                  "arguments": {"task_name": "w1", "message": "HELPER w1: reply ready", "fork_turns": "none"}},
                 {"type": "call", "namespace": "collaboration", "name": "spawn_agent",
                  "arguments": {"task_name": "w2", "message": "HELPER w2: reply ready", "fork_turns": "none"}},
                 {"type": "call", "namespace": "collaboration", "name": "wait_agent", "arguments": {"timeout_ms": 60000}},
                 {"type": "text", "text": "done"}],
        "w1": [{"type": "text", "text": "w1 ready", "after": {"agent": "root", "requests": 3}}]})
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
    env = {**ultra_env(2, {**LIMIT, **GUARD, "RELEASE_TOKEN_BUDGET": "4000000"}), "RELEASE_CONTINUATIONS": "0",
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

    thread = threading.Thread(target=main, daemon=True)
    thread.start()
    thread.join(300)
    try:
        assert not thread.is_alive(), "the run did not end"
        assert not errors, repr(errors)
        root, w1 = mock_api.requests("root"), mock_api.requests("w1")
        assert len(root) >= 4 and len(w1) >= 1
        texts = developer_texts(root[0])
        assert any(ultra.PROACTIVE_MODE_TEXT in text for text in texts)
        assert not any("Do not spawn sub-agents unless" in text for text in texts)  # not the explicit-only text
        assert any("You are `/root`, the primary agent" in text and "There are 2 available concurrency slots" in text
                   for text in texts)
        user = json.dumps([item for item in root[0]["body"]["input"] if item.get("role") == "user"])
        assert "At most 2 agents, you included, can work on this task at the same time." in user
        refused = " ".join(outputs(root[2]))
        assert "agent thread limit reached" in refused
        assert "w1 ready" in json.dumps(root[-1]["body"]) or "w1 ready" in json.dumps(root[-2]["body"])
        # Every request at the run's effort, the helper's too.
        for request in (*root, *w1):
            effort = (request["body"].get("reasoning") or {}).get("effort")
            assert effort in (None, "high"), effort
        session = local.agent_dir / "codex-single-session"
        totals = token_budget.summed(token_budget.session_totals("codex", session / "codex-home"))
        assert totals["output_tokens"] == 10 * (len(root) + len(w1))
    finally:
        mock_api.stop()
        state = harness_guard.read_json(str(local.guard_dir / "state" / "keeper.json")) or {}
        if state.get("pid") and harness_guard.alive(state["pid"], state.get("start")):
            kill(state["pid"])
        kill(harness_guard.agent_processes("CODEX_HOME", str(local.agent_dir / "codex-single-session/codex-home")))
