"""The `resources` command of `run.py --resources-tool`, for Codex and Claude Code agents.

agent.py writes this file, followed by a call to main() with the run's settings, into the
agents' container (resources_script()). An agent runs it from its CLI's shell, which carries
the CLI's home variable (CODEX_HOME, CLAUDE_CONFIG_DIR), and gets what Epoch's
LeanOpenProblems agents get from their `resources` tool (inspect_boltons.tools.resources):
each limit, how much of it is used, and here also how much is left:

- its tokens against its --token-budget, counted as the budget counts them (CAISI weighted
  tokens or output tokens): the token-budget watcher's latest record (token-usage.json in
  its session directory, which the watcher rewrites every few seconds), or, without a budget
  or before the watcher's first record, the same totals read from its CLI's session logs
  with token_budget.py's readers;
- for a team, the team's total;
- wall-clock time used and left of --timeout, counted from when the harness started the
  agents, minus the time spent waiting for usage limits (limit_resume.py).

With --continuation it prints instead the sentence the runner adds to the task's
continuation text before each continuation turn, and with --json the numbers as JSON (for a task's own
tools). Python 3.7+, standard library only.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HOME_ENV = {"codex": "CODEX_HOME", "claude": "CLAUDE_CONFIG_DIR"}
WEIGHTS = ("Tokens are weighted: an output token counts 1, an uncached input token 0.25 and a cached "
           "input token 0.02.")
USAGE = "usage: resources [--continuation | --json] [--home CLI_HOME]"


def token_budget_module(source: str) -> dict:
    """token_budget.py's functions, from its source (the container has no copy of the file)."""
    namespace = {"__name__": "token_budget"}
    exec(compile(source, "token_budget.py", "exec"), namespace)
    return namespace


def count(value: int) -> str:
    return f"{value:,}"


def duration(seconds: float) -> str:
    """inspect_boltons' format: 129600 -> "36h", 75 -> "1m 15s", 0 -> "0s"."""
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def run_clis(config: dict) -> list:
    """The run's CLIs, each once: a mixed team's (run.py --team; clis gives each home's), else cli."""
    return list(dict.fromkeys(config.get("clis") or [config["cli"]]))


def home_cli(config: dict, home: str) -> str | None:
    """The CLI of the agent with this home (None: not one of a mixed team's homes)."""
    if not config.get("clis"):
        return config["cli"]
    for other, cli in zip(config["homes"], config["clis"]):
        if os.path.realpath(other) == os.path.realpath(home):
            return cli
    return None


def agent_home(config: dict, environ) -> str | None:
    """The calling agent's CLI home: from the CLI's home variable, which every command the
    CLI runs inherits, else from AGENT_INDEX, else the only agent's."""
    homes = config["homes"]
    for cli in run_clis(config):
        value = environ.get(HOME_ENV[cli])
        if value:
            for home in homes:
                if os.path.realpath(home) == os.path.realpath(value):
                    return home
    index = environ.get("AGENT_INDEX", "")
    if index.isdigit() and int(index) < len(homes):
        return homes[int(index)]
    return homes[0] if len(homes) == 1 else None


def tokens_used(config: dict, home: str, token_budget: dict) -> int:
    """One agent's tokens as its budget counts them."""
    usage = None
    if config["budget"] is not None:
        try:
            record = json.loads((Path(home).parent / "token-usage.json").read_text())
            usage = {key: int(record["usage"][key]) for key in token_budget["USAGE_KEYS"]}
        except (OSError, ValueError, KeyError, TypeError):
            usage = None
    if usage is None:
        usage = token_budget["summed"](token_budget["READERS"][home_cli(config, home)](home).poll())
    return round(token_budget["spent"](usage, config["metric"]))


def waited(home: str) -> float:
    """Seconds the agent has spent waiting for usage limits and API retries, which do not count
    as working time (the limit watcher's limit-wait.json in its session directory; none without
    usage-limit resumption)."""
    try:
        return float(json.loads((Path(home).parent / "limit-wait.json").read_text())["waited_seconds"])
    except (OSError, ValueError, KeyError, TypeError):
        return 0.0


def measure(config: dict, home: str, token_budget: dict, now: float) -> dict:
    homes = config["homes"]
    used = tokens_used(config, home, token_budget)
    team = None
    if len(homes) > 1:
        team = sum(used if other == home else tokens_used(config, other, token_budget) for other in homes)
    elapsed = max(now - config["started"] - waited(home), 0.0)
    timeout = config.get("timeout")
    return {"used": used, "team": team, "agents": len(homes), "budget": config["budget"],
            "elapsed": elapsed, "timeout": timeout,
            "time_left": None if timeout is None else max(timeout - elapsed, 0.0)}


def table(rows: list[list[str]]) -> str:
    """A GitHub-style table, as inspect_boltons' tool prints (tabulate, tablefmt="github")."""
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    lines = ["| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |" for row in rows]
    lines.insert(1, "|" + "|".join("-" * (width + 2) for width in widths) + "|")
    return "\n".join(lines)


def report(config: dict, numbers: dict) -> str:
    budget, weighted = numbers["budget"], config["metric"] == "weighted"
    unit = "tokens" if weighted else "output tokens"
    limits = [name for name, limit in (("your token budget", budget), ("the time limit", numbers["timeout"]))
              if limit is not None]
    header = (f"Reaching {' or '.join(limits)} ends your work on the task." if limits
              else "No limit is set for your work on the task.")
    rows = [["Resource", "Used", "Limit", "Left"]]
    if budget is None:
        rows.append([f"Your {unit}", count(numbers["used"]), "none", "-"])
    else:
        rows.append([f"Your {unit}", count(numbers["used"]), count(budget),
                     count(max(budget - numbers["used"], 0))])
    if numbers["team"] is not None:
        label = f"Team {unit} ({numbers['agents']} agents)"
        if budget is None:
            rows.append([label, count(numbers["team"]), "none", "-"])
        else:
            team_budget = budget * numbers["agents"]
            rows.append([label, count(numbers["team"]), count(team_budget),
                         count(max(team_budget - numbers["team"], 0))])
    if numbers["timeout"] is None:
        rows.append(["Time", duration(numbers["elapsed"]), "none", "-"])
    else:
        rows.append(["Time", duration(numbers["elapsed"]), duration(numbers["timeout"]),
                     duration(numbers["time_left"])])
    return "\n".join([header, "", table(rows)] + (["", WEIGHTS] if weighted else []))


def continuation(config: dict, numbers: dict) -> str:
    """The sentence added to the continuation text: what is left, and to keep working."""
    unit = "weighted tokens" if config["metric"] == "weighted" else "output tokens"
    budget, timeout = numbers["budget"], numbers["timeout"]
    left = []
    if budget is not None:
        left.append(f"{count(max(budget - numbers['used'], 0))} of your {count(budget)} {unit}")
    if timeout is not None:
        left.append(f"{duration(numbers['time_left'])} of your {duration(timeout)}")
    text = "Remaining resources: " + (" and ".join(left) if left else "no limit is set")
    if budget is None:
        text += f"; you have used {count(numbers['used'])} {unit}"
    text += "."
    if numbers["team"] is not None:
        text += f" Your team has used {count(numbers['team'])} {unit} in total."
    if budget is not None:
        text += " Keep working on the task until your token budget is used up."
    elif timeout is not None:
        text += " Keep working on the task until your time is used up."
    return text


def main(config_json: str, token_budget_source: str, argv=None, environ=None, now=None) -> int:
    config = json.loads(config_json)
    argv = list(sys.argv[1:] if argv is None else argv)
    environ = os.environ if environ is None else environ
    mode_continuation, mode_json, home = False, False, None
    while argv:
        option = argv.pop(0)
        if option == "--continuation":
            mode_continuation = True
        elif option == "--json":
            mode_json = True
        elif option == "--home" and argv:
            home = argv.pop(0)
        elif option in ("-h", "--help"):
            print(USAGE)
            return 0
        else:
            print(USAGE, file=sys.stderr)
            return 2
    home = home or agent_home(config, environ)
    if home is None or home_cli(config, home) is None:
        names = " or ".join(HOME_ENV[cli] for cli in run_clis(config))
        print(f"resources: cannot tell which agent is asking ({names} is not one of this "
              f"run's agents); run it from your own shell", file=sys.stderr)
        return 1
    numbers = measure(config, home, token_budget_module(token_budget_source), time.time() if now is None else now)
    if mode_continuation:
        print("\n" + continuation(config, numbers))
    elif mode_json:
        print(json.dumps({**numbers, "metric": config["metric"]}, sort_keys=True))
    else:
        print(report(config, numbers))
    return 0
