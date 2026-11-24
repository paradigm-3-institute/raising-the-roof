"""Checks for --resources-tool: the `resources` command, the prompt paragraph and the
continuation sentence (no Docker needed)."""

from __future__ import annotations

import ast
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from cli_render import CASES, INSTRUCTION, ROOT, render
from test_cli_agents import agent_module, run_py, scripts  # noqa: F401
from test_matched_budget import AGENT_CLASSES, HOME_DIRS, usage_logs

sys.path.insert(0, str(ROOT))
import resources_tool  # noqa: E402
from token_budget import HOME_ENV  # noqa: E402

ON = {"RELEASE_RESOURCES_TOOL": "1", "RELEASE_AGENT_TIMEOUT": "7200"}
PARAGRAPH = ("You have been given enough resources (tokens and time) to keep working on this task for a long "
             "time. Check your remaining resources at any time by running `resources` in your shell")
INSTALL = "mkdir -p /usr/local/libexec /usr/local/bin"
WEIGHTS = ("Tokens are weighted: an output token counts 1, an uncached input token 0.25 and a cached input "
           "token 0.02.")
# A token-budget watcher's record (token-usage.json) that differs from the logs of usage_logs():
# 2,000 output and 20,000 input, 18,000 cached = 2,860 weighted (the logs: 1,200 and 1,716).
RECORD_USAGE = {"input_tokens": 20000, "cached_input_tokens": 18000, "output_tokens": 2000,
                "reasoning_output_tokens": 0}


def texts(calls: list[dict]) -> list[str]:
    """Every shell word the harness sends."""
    return [token for call in calls for token in shlex.split(call["command"])]


def instruction_of(calls: list[dict]) -> list[str]:
    return [token for token in texts(calls) if token.startswith(INSTRUCTION.splitlines()[0])]


def install_of(calls: list[dict]) -> list[str]:
    """The install command, which runs as root (/usr/local stays on the CLIs' login PATH)."""
    found = [call for call in calls if INSTALL in call["command"][:200]]
    assert all(call["user"] == "root" for call in found)
    return [call["command"] for call in found]


@pytest.mark.parametrize("cli", ["claude", "codex"])
@pytest.mark.parametrize("case", CASES)
def test_off_changes_nothing(cli, case):
    """Unset (or anything but "1"), every command is what it was without the option; the
    proof against main's own code is in resources-tool-report.md."""
    calls = render(case, cli=cli)
    assert calls == render(case, cli=cli, env_overrides={"RELEASE_RESOURCES_TOOL": "0",
                                                          "RELEASE_AGENT_TIMEOUT": "3600"})
    joined = "\n".join(call["command"] for call in calls)
    assert "libexec/resources" not in joined and "Check your remaining resources" not in joined


@pytest.mark.parametrize("cli", ["claude", "codex"])
@pytest.mark.parametrize("case, team", [("ale-single", False), ("ale-solo", False), ("ale-team3", True)])
def test_on_adds_command_paragraph_and_continuation(cli, case, team, tmp_path):
    calls = render(case, cli=cli, env_overrides={**ON, "RELEASE_TOKEN_BUDGET": "300000"})
    instructions = instruction_of(calls)
    assert instructions and all(
        text == INSTRUCTION + "\n\n" + PARAGRAPH + ("; it also shows your team's total." if team else ".")
        for text in instructions)
    [install] = install_of(calls)
    assert "ln -sfn /usr/local/libexec/resources /usr/local/bin/resources" in install
    config = installed_config(install)
    agents = 3 if team else 1
    assert len(config["homes"]) == agents and all(home.endswith(HOME_DIRS[cli]) for home in config["homes"])
    assert (config["cli"], config["budget"], config["metric"], config["timeout"]) == (cli, 300000, "weighted", 7200.0)
    line = (f'python3 /usr/local/libexec/resources --continuation --home "${cli}_log_dir" '
            f'>> "$feedback_file" 2>> "${{{cli}_log_dir%/*}}/resources-errors.log"')
    runners = scripts(calls)
    assert sum(script.count(line) for script in runners) == 1
    for index, script in enumerate(runners):
        path = tmp_path / f"{index}.sh"
        path.write_text(script)
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def installed_config(install: str) -> dict:
    """The settings an install command bakes into the command."""
    script = next(token for token in shlex.split(install) if token.startswith("#!/usr/bin/env python3"))
    call = ast.parse("main(" + script.rsplit("sys.exit(main(", 1)[1].rstrip()[:-1]).body[0].value
    return json.loads(call.args[0].value)


def rows(out: str) -> dict[str, list[str]]:
    """A report's table: label -> [used, limit, left]."""
    return {line.split("|")[1].strip(): [cell.strip() for cell in line.split("|")[2:5]]
            for line in out.splitlines() if line.startswith("| ")}


def test_run_py_option(monkeypatch):
    _, env, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", "codex", "--model", "m", "--resources-tool", "--timeout", "5400")
    assert (env["RELEASE_RESOURCES_TOOL"], env["RELEASE_AGENT_TIMEOUT"]) == ("1", "5400")
    monkeypatch.setenv("RELEASE_RESOURCES_TOOL", "1")  # a stale value from the caller's shell is dropped
    _, env, _ = run_py("ale-bench", "--problem", "ahc014", "--cli", "claude", "--model", "m", "--no-limit-resume")
    assert "RELEASE_RESOURCES_TOOL" not in env and "RELEASE_AGENT_TIMEOUT" not in env


def run_command(tmp_path: Path, config: dict, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """The installed command, as a file, run like an agent runs it."""
    import agent
    path = tmp_path / "resources"
    path.write_text(agent.resources_script(config))
    path.chmod(0o755)
    clean = {key: value for key, value in os.environ.items()
             if key not in ("CODEX_HOME", "CLAUDE_CONFIG_DIR", "AGENT_INDEX")}
    return subprocess.run([sys.executable, str(path), *args], env={**clean, **(env or {})},
                          capture_output=True, text=True)


def homes(tmp_path: Path, cli: str, agents: int) -> list[str]:
    return [str(tmp_path / f"sessions-{index}-a{index + 1}" / HOME_DIRS[cli]) for index in range(agents)]


def config(cli: str, agent_homes: list[str], budget: int | None, metric: str = "weighted",
           timeout: float | None = 7200.0, started: float = 1000.0) -> dict:
    return {"cli": cli, "homes": agent_homes, "budget": budget, "metric": metric, "timeout": timeout,
            "started": started}


def write_record(home: str, usage: dict) -> None:
    (Path(home).parent / "token-usage.json").write_text(json.dumps(
        {"budget": 300000, "budget_metric": "weighted", "usage": usage, "weighted_tokens": 0,
         "plan_used_percent": 19.0}))


def report(capsys, settings: dict, home: str | None = None, now: float = 1000.0 + 3725, *args: str) -> str:
    environ = {} if home is None else {HOME_ENV[settings["cli"]]: home}
    assert resources_tool.main(json.dumps(settings), (ROOT / "token_budget.py").read_text(), list(args),
                               environ, now) == 0
    return capsys.readouterr().out


@pytest.mark.parametrize("cli", ["codex", "claude"])
def test_solo_report_uses_the_watchers_numbers(cli, tmp_path, capsys):
    [home] = homes(tmp_path, cli, 1)
    usage_logs(cli, Path(home))
    write_record(home, RECORD_USAGE)
    assert report(capsys, config(cli, [home], 300000), home) == "\n".join([
        "Reaching your token budget or the time limit ends your work on the task.",
        "",
        "| Resource    | Used     | Limit   | Left    |",
        "|-------------|----------|---------|---------|",
        "| Your tokens | 2,860    | 300,000 | 297,140 |",
        "| Time        | 1h 2m 5s | 2h      | 57m 55s |",
        "",
        WEIGHTS,
        ""])
    assert report(capsys, config(cli, [home], 300000), home, 1000.0 + 3725, "--continuation") == (
        "\nRemaining resources: 297,140 of your 300,000 weighted tokens and 57m 55s of your 2h. "
        "Keep working on the task until your token budget is used up.\n")


@pytest.mark.parametrize("cli", ["codex", "claude"])
def test_output_metric(cli, tmp_path, capsys):
    [home] = homes(tmp_path, cli, 1)
    usage_logs(cli, Path(home))
    write_record(home, RECORD_USAGE)
    out = report(capsys, config(cli, [home], 5000, metric="output"), home)
    assert rows(out)["Your output tokens"] == ["2,000", "5,000", "3,000"]
    assert WEIGHTS not in out
    assert report(capsys, config(cli, [home], 5000, metric="output"), home, 1000.0, "--continuation") == (
        "\nRemaining resources: 3,000 of your 5,000 output tokens and 2h of your 2h. "
        "Keep working on the task until your token budget is used up.\n")


@pytest.mark.parametrize("cli", ["codex", "claude"])
def test_without_a_budget_the_logs_count(cli, tmp_path, capsys):
    """No --token-budget: no watcher, no budget; the command reads the CLI's session logs (a
    stale token-usage.json is ignored) and the continuation names no budget."""
    [home] = homes(tmp_path, cli, 1)
    usage_logs(cli, Path(home))
    write_record(home, RECORD_USAGE)
    out = report(capsys, config(cli, [home], None), home)
    assert out.splitlines()[0] == "Reaching the time limit ends your work on the task."
    assert rows(out)["Your tokens"] == ["1,716", "none", "-"]
    continuation = report(capsys, config(cli, [home], None), home, 1000.0 + 60, "--continuation")
    assert continuation == ("\nRemaining resources: 1h 59m of your 2h; you have used 1,716 weighted tokens. "
                            "Keep working on the task until your time is used up.\n")
    assert "budget" not in continuation


@pytest.mark.parametrize("cli", ["codex", "claude"])
@pytest.mark.parametrize("budget", [None, 300000])
def test_team_totals(cli, budget, tmp_path, capsys):
    team = homes(tmp_path, cli, 3)
    for index, home in enumerate(team[:2]):
        usage_logs(cli, Path(home), f"s{index}")
        if budget:
            write_record(home, RECORD_USAGE)
    own, total = (2860, 5720) if budget else (1716, 3432)
    table = rows(report(capsys, config(cli, team, budget), team[1]))
    assert table["Your tokens"][0] == f"{own:,}" and table["Team tokens (3 agents)"][0] == f"{total:,}"
    if budget:
        assert table["Team tokens (3 agents)"][1:] == ["900,000", f"{900000 - total:,}"]
    continuation = report(capsys, config(cli, team, budget), team[1], 1000.0, "--continuation")
    assert f"Your team has used {total:,} weighted tokens in total." in continuation


@pytest.mark.parametrize("cli", ["codex", "claude"])
def test_the_command_finds_its_agent(cli, tmp_path):
    """Run as a file: the CLI's home variable (which every command the CLI runs inherits)
    picks the agent, AGENT_INDEX is the fallback, and an unknown home is an error."""
    team = homes(tmp_path, cli, 2)
    usage_logs(cli, Path(team[1]))
    settings = config(cli, team, None)
    by_home = run_command(tmp_path, settings, env={HOME_ENV[cli]: team[1]})
    by_index = run_command(tmp_path, settings, env={"AGENT_INDEX": "1"})
    assert by_home.returncode == by_index.returncode == 0, by_home.stderr + by_index.stderr
    assert rows(by_home.stdout)["Your tokens"][0] == "1,716"
    assert by_home.stdout.split("| Time")[0] == by_index.stdout.split("| Time")[0]
    lost = run_command(tmp_path, settings, env={HOME_ENV[cli]: str(tmp_path / "elsewhere")})
    assert lost.returncode == 1 and "cannot tell which agent is asking" in lost.stderr
    assert run_command(tmp_path, settings, "--bogus").returncode == 2


def installed_here(agent_module, monkeypatch, tmp_path: Path) -> tuple[Path, Path]:  # noqa: F811
    """Point the install at tmp_path instead of /usr/local; returns (script, command)."""
    script, command = tmp_path / "usr/local/libexec/resources", tmp_path / "usr/local/bin/resources"
    monkeypatch.setattr(agent_module, "RESOURCES_SCRIPT", str(script))
    monkeypatch.setattr(agent_module, "RESOURCES_COMMAND", str(command))
    for key, value in ON.items():
        monkeypatch.setenv(key, value)
    return script, command


@pytest.mark.parametrize("cli", ["codex", "claude"])
def test_continuation_file(agent_module, cli, tmp_path, monkeypatch):  # noqa: F811
    """The install command, then the runner's lines: the task's continuation text, then the sentence."""
    agent = getattr(agent_module, AGENT_CLASSES[cli])(logs_dir=tmp_path)
    home = tmp_path / "session" / HOME_DIRS[cli]
    usage_logs(cli, home)
    _, command = installed_here(agent_module, monkeypatch, tmp_path)
    install = agent._resources_install([str(home)], started=0.0)
    line = agent._resources_continuation("")
    feedback = tmp_path / "feedback.txt"
    script = "\n".join([agent._action_feedback_helpers(), install, f"{cli}_log_dir={shlex.quote(str(home))}",
                        f"feedback_file={shlex.quote(str(feedback))}", 'write_no_action_feedback "$feedback_file"',
                        line])
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    text = feedback.read_text()
    # The fixture's continuation text is "continue"; the run started at epoch 0, so no time is left.
    assert text == ("continue\n\nRemaining resources: 0s of your 2h; you have used 1,716 weighted tokens. "
                    "Keep working on the task until your time is used up.\n")
    assert command.is_symlink() and os.access(command, os.X_OK)
    ran = subprocess.run([str(command)], env={**os.environ, HOME_ENV[cli]: str(home)}, capture_output=True, text=True)
    assert ran.returncode == 0 and rows(ran.stdout)["Your tokens"][0] == "1,716", ran.stderr
    # Installing again (a second trial in a reused container) is fine.
    assert subprocess.run(["bash", "-c", install], capture_output=True).returncode == 0


def test_an_image_command_is_never_replaced(agent_module, tmp_path, monkeypatch):  # noqa: F811
    _, command = installed_here(agent_module, monkeypatch, tmp_path)
    command.parent.mkdir(parents=True)
    command.write_text("#!/bin/sh\necho the image's own\n")
    install = agent_module.CodexAgent(logs_dir=tmp_path)._resources_install(["/x/codex-home"], started=0.0)
    result = subprocess.run(["bash", "-c", install], capture_output=True, text=True)
    assert result.returncode == 1 and "already exists in this image" in result.stderr
    assert command.read_text() == "#!/bin/sh\necho the image's own\n"


def test_duration_format():
    # inspect_boltons' cases.
    assert resources_tool.duration(0) == "0s"
    assert resources_tool.duration(129_600) == "36h"
    assert resources_tool.duration(45) == "45s"
    assert resources_tool.duration(3 * 3600 + 25 * 60) == "3h 25m"
