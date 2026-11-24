"""Early, faithful handling of hung test scripts.

A script the official runner kills at its time limit scores 0 (exit 137, no trusted plan) after burning its whole
limit (180-900 s). This module ends such a script as soon as it is *provably* stuck and kills it exactly as `timeout -s
KILL` does at the limit (SIGKILL to timeout's process group), so the outcome is the official one: exit 137, the TAP
printed so far.

Provably stuck = every process in the script's wait closure sleeps in a kernel wait that only another process of the
closure could end, and nothing in the closure changes (CPU ticks, states, process set) for QUIET seconds:
  - pipe read (anon_pipe_read/pipe_read/pipe_wait...): waits on the writers of every pipe/fifo it holds for reading
    (an over-approximation: we cannot see which fd it blocks on); a pipe nobody else holds for writing would give EOF,
    so a reader with no writer is not stuck;
  - pipe write (anon_pipe_write/pipe_write): waits on every reader of the pipes it holds for writing;
  - wait for a child (do_wait/kernel_wait4/...; `timeout` and `runuser` too): waits on its children;
Anything else (running, disk sleep, nanosleep/timers, poll/select/epoll, futex, sockets, inotify, unknown wchans)
counts as able to make progress, so the script runs on to its normal limit.

Only the agent user can read the agent's /proc entries (wchan, fds) without CAP_SYS_PTRACE, so a helper started
with `runuser -u agent` samples /proc and the root runner decides. A candidate that tampers with the helper can only
stop early kills (the normal limit still applies) or end its own scripts early (scores it 0, which a time-out gives
anyway); it cannot raise a score.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time

AGENT_UID = 1001
PIPE_READ = ("anon_pipe_read", "pipe_read", "pipe_wait", "pipe_wait_readable", "fifo_open")
PIPE_WRITE = ("anon_pipe_write", "pipe_write", "pipe_wait_writable")
CHILD_WAIT = ("do_wait", "kernel_wait4", "do_wait.part.0", "__do_sys_wait4", "wait_consider_task")


# --------------------------------------------------------------------------------------------- helper (as agent)

def sample_procs() -> dict:
    procs = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        base = f"/proc/{pid}"
        try:
            if os.stat(base).st_uid != AGENT_UID:
                continue
            stat = open(f"{base}/stat").read()
            comm = stat[stat.index("(") + 1:stat.rindex(")")]
            rest = stat[stat.rindex(")") + 2:].split()
            state, ppid, pgid, sid = rest[0], int(rest[1]), int(rest[2]), int(rest[3])
            ticks = int(rest[11]) + int(rest[12])
            threads = int(rest[17])
            wchan = open(f"{base}/wchan").read().strip()
            reads, writes, other = [], [], 0
            for fd in os.listdir(f"{base}/fd"):
                try:
                    target = os.readlink(f"{base}/fd/{fd}")
                except OSError:
                    continue
                key = target if target.startswith("pipe:") else (_fifo_key(f"{base}/fd/{fd}") if _is_path(target) else None)
                if key:
                    flags = 0
                    for line in open(f"{base}/fdinfo/{fd}"):
                        if line.startswith("flags:"):
                            flags = int(line.split()[1], 8)
                            break
                    acc = flags & 3
                    if acc in (0, 2):
                        reads.append(key)
                    if acc in (1, 2):
                        writes.append(key)
                elif target.startswith(("socket:", "anon_inode:")):
                    other += 1
            procs[pid] = {"comm": comm, "state": state, "ppid": ppid, "pgid": pgid, "sid": sid, "ticks": ticks,
                          "threads": threads, "wchan": wchan, "r": sorted(set(reads)), "w": sorted(set(writes)),
                          "sock": other}
        except (OSError, ValueError, IndexError):
            continue
    return procs


def _is_path(target: str) -> bool:
    return target.startswith("/")


def _fifo_key(fd_link: str) -> str | None:
    """A named pipe's identity (device:inode), or None for anything that is not a FIFO."""
    import stat as st_mod
    try:
        st = os.stat(fd_link)
    except OSError:
        return None
    return f"fifo:{st.st_dev}:{st.st_ino}" if st_mod.S_ISFIFO(st.st_mode) else None


def helper(period: float) -> None:
    out = sys.stdout
    while True:
        try:
            out.write(json.dumps({"t": time.monotonic(), "procs": sample_procs()}) + "\n")
            out.flush()
        except BrokenPipeError:
            return
        time.sleep(period)


# --------------------------------------------------------------------------------------------- analysis (root)

def stuck_closure(procs: dict, roots: list[int]) -> tuple[bool, list[int], str]:
    """(stuck, closure pids, reason if not stuck)."""
    procs = {int(k): v for k, v in procs.items()}
    children: dict[int, list[int]] = {}
    for pid, p in procs.items():
        children.setdefault(p["ppid"], []).append(pid)
    readers: dict[str, set[int]] = {}
    writers: dict[str, set[int]] = {}
    for pid, p in procs.items():
        for k in p["r"]:
            readers.setdefault(k, set()).add(pid)
        for k in p["w"]:
            writers.setdefault(k, set()).add(pid)
    seen, todo = set(), [r for r in roots if r in procs]
    if not todo:
        return False, [], "no processes"
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        p = procs[pid]
        if p["state"] not in ("S",) or p["threads"] != 1:
            return False, sorted(seen), f"{pid} {p['comm']} state={p['state']} threads={p['threads']}"
        w = p["wchan"]
        deps: set[int] = set()
        if w in CHILD_WAIT or p["comm"] in ("timeout", "runuser") and w in ("sigsuspend", "do_sigtimedwait", "sys_pause"):
            kids = children.get(pid, [])
            if not kids:
                return False, sorted(seen), f"{pid} {p['comm']} waits with no child"
            deps.update(kids)
        elif w in PIPE_READ and w != "fifo_open":
            # A pipe with no writer at all gives EOF at once, so the read that blocks is on a pipe with a writer.
            live = [k for k in p["r"] if writers.get(k)]
            if not live:
                return False, sorted(seen), f"{pid} {p['comm']} blocked in a pipe read but holds no pipe with a writer"
            for k in live:
                deps.update(writers[k] - {pid})
        elif w in PIPE_WRITE:
            if not p["w"]:
                return False, sorted(seen), f"{pid} {p['comm']} pipe write without a pipe"
            for k in p["w"]:
                deps.update(readers.get(k, set()) - {pid})
        else:
            return False, sorted(seen), f"{pid} {p['comm']} wchan={w or '-'}"
        todo.extend(d for d in deps if d not in seen)
    return True, sorted(seen), ""


class HangWatch:
    def __init__(self, quiet: float, period: float = 1.0, min_age: float = 5.0, log_path: str | None = None):
        self.quiet, self.min_age = quiet, min_age
        self.lock = threading.Lock()
        self.scripts: dict[int, dict] = {}  # runuser pid -> state
        self.log = open(log_path, "a") if log_path else None
        self.helper = subprocess.Popen(["runuser", "-u", "agent", "--", sys.executable, os.path.abspath(__file__),
                                        "helper", str(period)], stdout=subprocess.PIPE, stdin=subprocess.DEVNULL,
                                       text=True, start_new_session=True)
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        for line in self.helper.stdout:
            try:
                snap = json.loads(line)
            except ValueError:
                continue
            procs = {int(k): v for k, v in snap["procs"].items()}
            now = time.monotonic()
            with self.lock:
                items = list(self.scripts.items())
            for runuser_pid, st in items:
                if st.get("done") or now - st["started"] < self.min_age:
                    continue
                # the script's roots: runuser and every process of its session
                roots = [runuser_pid] + [pid for pid, p in procs.items() if p["sid"] == runuser_pid]
                ok, closure, why = stuck_closure(procs, roots)
                if not ok:
                    st["since"], st["sig"], st["why"] = None, None, why
                    continue
                st["why"] = "stuck"
                sig = tuple((pid, procs[pid]["ticks"], procs[pid]["wchan"]) for pid in closure)
                if st.get("sig") != sig:
                    st["since"], st["sig"] = now, sig
                    continue
                st["stuck_max"] = max(st.get("stuck_max", 0.0), now - st["since"])
                st["stuck_at"] = st.get("stuck_at") or round(st["since"] - st["started"], 1)
                if self.quiet > 0 and now - st["since"] >= self.quiet:
                    self._kill(runuser_pid, st, procs, closure, now)

    def _kill(self, runuser_pid, st, procs, closure, now):
        """What `timeout -s KILL` does at the limit: SIGKILL to its own process group."""
        timeout_pids = [pid for pid, p in procs.items() if p["comm"] == "timeout" and p["ppid"] == runuser_pid]
        if not timeout_pids:
            return
        st["done"] = True
        st["killed"] = True
        st["at"] = round(now - st["started"], 1)
        st["closure"] = [(pid, procs[pid]["comm"], procs[pid]["wchan"]) for pid in closure]
        try:
            os.killpg(procs[timeout_pids[0]]["pgid"], signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        if self.log:
            self.log.write(json.dumps({"name": st["name"], "at": st["at"], "cap": st["cap"], "closure": st["closure"]}) + "\n")
            self.log.flush()

    def watch(self, proc, cap, started, name=""):
        st = {"started": started, "cap": cap, "name": name, "since": None, "sig": None}
        with self.lock:
            self.scripts[proc.pid] = st
        return st

    def unwatch(self, proc):
        with self.lock:
            st = self.scripts.pop(proc.pid, None)
        if st is not None:
            st["done"] = True
        return st


if __name__ == "__main__":
    if sys.argv[1] == "helper":
        helper(float(sys.argv[2]))
