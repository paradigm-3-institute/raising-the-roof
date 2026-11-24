"""The glue between this harness and upstream SwarmChat (vendor/swarmchat, unmodified; docs/setups.md).

agent.py copies the vendored package and this file into the container and runs them there. Everything that talks to
the chat is upstream's code; this file only starts it, keeps its state in the trial's agent directory, and reads it
back after a restart:

  run    as root, one process: upstream's server (`swarmchat serve`, in the main thread), then the team's users and
         channels made with upstream's admin calls (as extras/programbench/run.py start_chat), then one thread per
         agent running upstream's notifier (extras/programbench/run.py notify), which turns each new message into a
         notice file. Its read positions are saved, so a restarted notifier never writes a notice twice.
  ready  waits until `run` has made the users and written each agent's token.
  stop   ends `run` (SIGTERM: upstream's server saves its read positions on the way out).
  hook   the agents' PostToolUse hook: upstream's extras/programbench/guest.py `warn` hook (up to 8 notices as
         hookSpecificOutput.additionalContext) on the agent's own notice folder, and a line in its shown.jsonl.

Paths come from the command line and the environment, never from constants here, so a test can move them.
Standard library only; the notifier needs Python 3.11 (upstream's run.py imports tomllib), the rest 3.9.
"""
import argparse
import importlib.util
import json
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
# Where agent.py unpacks the vendored package (upstream's own layout), unless the caller says otherwise.
LIB = os.environ.get("TTC_SWARMCHAT_LIB") or os.path.join(HERE, "hsc")
NOTICES_ENV = "TTC_SWARMCHAT_NOTICES"  # an agent's notice folder, for the hook


def upstream(name, relative):
    """One of upstream's extras/programbench modules, loaded from LIB."""
    if LIB not in sys.path:
        sys.path.insert(0, LIB)
    spec = importlib.util.spec_from_file_location(name, os.path.join(LIB, relative))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, text, mode=0o644):
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.chmod(path + ".tmp", mode)
    os.replace(path + ".tmp", path)


# ---- run --------------------------------------------------------------------------------------------------------

def setup(a, stop):
    """The team's users, tokens and channels, then the notifiers. A restart keeps every token that still works."""
    from swarmchat.client import Client, ClientError
    programbench = upstream("swarmchat_programbench_run", "extras/programbench/run.py")
    url = f"http://127.0.0.1:{a.port}"
    while True:  # until this process's server answers
        try:
            with open(os.path.join(a.data, "admin.token")) as f:
                admin = Client(url, f.read().strip())
            admin.policy_show()
            break
        except (OSError, ClientError):
            time.sleep(0.1)
    ids, tokens = a.agents.split(","), {}
    for agent in ids:
        folder = os.path.join(a.agents_dir, agent)
        os.makedirs(os.path.join(folder, "notices"), exist_ok=True)
        os.chmod(os.path.join(folder, "notices"), 0o777)  # the agent's hook takes its notices (upstream: writable())
        shown = os.path.join(folder, "shown.jsonl")
        if not os.path.exists(shown):
            open(shown, "a").close()
        os.chmod(shown, 0o666)  # the agent's hook appends to it
        token_file = os.path.join(folder, "token")
        try:
            with open(token_file) as f:
                tokens[agent] = f.read().strip()
            Client(url, tokens[agent]).whoami()
            continue
        except (OSError, ClientError):
            pass
        try:
            tokens[agent] = admin.user_add(id=agent, name=agent)["token"]
        except ClientError:  # it exists, but its token file is gone or no longer works
            tokens[agent] = admin.token_rotate(id=agent)["token"]
        write(token_file, tokens[agent] + "\n")
    for name, members in json.loads(a.channels).items():
        first = Client(url, tokens[members[0]])
        if not any(chat.get("name") == name for chat in first.chats(limit=1000)["chats"]):
            first.channel_create(name=name, members=",".join(members))
    for agent in ids:
        client = Positions(url, tokens[agent], os.path.join(a.agents_dir, agent, "positions.json"), Client)
        threading.Thread(target=programbench.notify, daemon=True,
                         args=(client, Path(os.path.join(a.agents_dir, agent, "notices")), stop)).start()
    write(os.path.join(a.agents_dir, "ready"), str(os.getpid()))


def Positions(url, token, path, Client):
    """Upstream's Client, but each `wait` the notifier makes starts from the positions it saved last (and saves the
    ones it was given): what the previous wait returned is in notice files by the time notify asks again."""

    class Saved(Client):
        def call(self, name, params=None, http_timeout=60):
            if name == "wait":
                after = dict((params or {}).get("after") or {})
                if after:
                    write(path, json.dumps(after), 0o600)
                try:
                    with open(path) as f:
                        after = {**json.load(f), **after}
                except (OSError, ValueError):
                    pass
                params = {**(params or {}), "after": after}
            return super().call(name, params, http_timeout)

    return Saved(url, token)


def run(a):
    sys.path.insert(0, LIB)
    for stale in ("ready", "run.pid"):
        if os.path.exists(os.path.join(a.agents_dir, stale)):
            os.remove(os.path.join(a.agents_dir, stale))
    os.makedirs(a.data, mode=0o700, exist_ok=True)
    os.chmod(a.data, 0o700)  # the admin token is in it
    os.makedirs(a.agents_dir, exist_ok=True)
    os.chmod(a.agents_dir, 0o755)
    write(os.path.join(a.agents_dir, "run.pid"), str(os.getpid()))
    stop = threading.Event()

    def guarded():
        try:
            setup(a, stop)
        except BaseException as exc:  # noqa: BLE001 - without its users the chat is useless: end, and be restarted
            print(f"swarmchat setup failed: {exc!r}", file=sys.stderr, flush=True)
            os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=guarded, daemon=True).start()
    from swarmchat.cli import main
    code = main(["serve", "--data", a.data, "--port", str(a.port), *a.serve])
    stop.set()
    return code


def ready(a):
    deadline = time.monotonic() + a.timeout
    while time.monotonic() < deadline:
        try:
            with open(os.path.join(a.agents_dir, "ready")) as f:
                pid = int(f.read())
            os.kill(pid, 0)
            return 0
        except (OSError, ValueError):
            time.sleep(0.2)
    print("the team chat did not start", flush=True)
    return 1


def stop(a):
    try:
        with open(os.path.join(a.agents_dir, "run.pid")) as f:
            pid = int(f.read())
        os.kill(pid, signal.SIGTERM)
    except (OSError, ValueError):
        return 0
    for _ in range(300):
        try:
            os.kill(pid, 0)
        except OSError:
            return 0
        time.sleep(0.1)
    os.kill(pid, signal.SIGKILL)
    return 0


# ---- hook -------------------------------------------------------------------------------------------------------

NOTICE = re.compile(r"^\[swarmchat\] .*, from \S+ \(([^()\s]+)\):$", re.M)


def hook(a):
    """Upstream's PostToolUse hook (guest.py hook_warn) on this agent's notices: its time warning is off (our runs
    have their own limits), and what it hands over is logged. Never fails the tool."""
    try:
        notices = os.environ[NOTICES_ENV]
        guest = upstream("swarmchat_programbench_guest", "extras/programbench/guest.py")
        guest.NOTICES = notices
        guest.load = lambda: {"deadline": float("inf")}
        take = guest.take_notices
        raw = sys.stdin.read()
        sys.stdin = _Replay(raw)

        def logged():
            text = take()
            if text:
                try:  # the notices are taken: they reach the agent even if this line cannot be written
                    event = json.loads(raw or "{}")
                    with open(os.path.join(os.path.dirname(notices), "shown.jsonl"), "a") as f:
                        f.write(json.dumps({"t": time.time(), "ids": NOTICE.findall(text),
                                            "session": event.get("session_id"), "turn": event.get("turn_id"),
                                            "tool": event.get("tool_name")}) + "\n")
                except Exception as exc:  # noqa: BLE001
                    print(f"swarmchat hook: shown.jsonl: {exc!r}", file=sys.stderr)
            return text

        guest.take_notices = logged
        guest.hook_warn()
    except Exception as exc:  # noqa: BLE001 - a hook must never break the agent's tool call
        print(f"swarmchat hook: {exc!r}", file=sys.stderr)
    return 0


class _Replay:
    def __init__(self, text):
        self.text = text

    def read(self):
        return self.text


def main(argv=None):
    p = argparse.ArgumentParser(prog="swarmchat-glue")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "ready", "stop"):
        c = sub.add_parser(name)
        c.add_argument("--agents-dir", required=True)
        if name == "run":
            c.add_argument("--data", required=True)
            c.add_argument("--port", type=int, required=True)
            c.add_argument("--agents", required=True)
            c.add_argument("--channels", required=True, help='JSON: {"general": ["a1", "a2"]}')
            c.add_argument("serve", nargs="*", help="more flags for `swarmchat serve`, after --")
        if name == "ready":
            c.add_argument("--timeout", type=float, default=60)
    sub.add_parser("hook")
    a = p.parse_args(argv)
    return {"run": run, "ready": ready, "stop": stop, "hook": hook}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
