#!/usr/bin/env python3
"""NERO V3.77 STALE PROCESS CHECK - leftover processes of earlier simulation / real-robot runs.

V3.77 STALE PROCESS CHECK

Why (user, V3.76): before a simulation starts, leftover processes of an earlier run must be found and stopped.  The
launcher's V3.07 check (start_nero_sim.sh) only knew ros2_control_node / move_group / spawner / rviz2 /
robot_state_publisher, the backend and the GPU server; it missed `ros2 launch agx_arm_ctrl` (real driver), any other
NERO_*.py script (e.g. a backend started from another release folder) and python-launched ros2 launches.

  * stale_procs(procs=None, me=())   -> [(pid, kind, cmd)]   argv-based (an editor / `tail -f` that merely mentions a
                                       NERO file is never matched; the console itself is never matched)
  * stop_procs(pids, step_s, log)    -> {"stopped": [...], "left": [...]}  SIGINT -> SIGTERM -> SIGKILL, each step
                                       bounded by step_s, process groups of ros2 launch / rviz included
  * main: `python3 NERO_V377_Stale_Procs.py --list` prints "<pid> <cmd>" lines (used by start_nero_sim.sh)
"""
import os
import signal
import sys
import time

VERSION = "V3.77"
MARKER = "V3.77 STALE PROCESS CHECK"
BINS = {"ros2_control_node", "move_group", "spawner", "rviz2", "robot_state_publisher", "agx_arm_ctrl_single_node",
        "component_container", "component_container_mt"}
LAUNCH_PKGS = {"agx_arm_ctrl", "agx_arm_moveit"}
KEEP_SCRIPTS = {"NERO_V312_Console.py", "NERO_V377_Stale_Procs.py"}      # the console and this check itself
EXTRA_SCRIPTS = {"NERO_GPU_Planner_Server.py", "spawner"}


def proc_list(proc_root="/proc"):
    out = []
    try:
        names = os.listdir(proc_root)
    except Exception:
        return out
    for d in names:
        if not d.isdigit():
            continue
        try:
            raw = open(os.path.join(proc_root, d, "cmdline"), "rb").read()
        except Exception:
            continue
        argv = [a.decode(errors="replace") for a in raw.split(b"\0") if a]
        if argv:
            out.append((int(d), argv))
    return out


def classify(argv):
    """-> kind string or None"""
    b0 = os.path.basename(argv[0])
    if b0 in BINS:
        return b0
    if b0 == "ros2" and len(argv) >= 3 and argv[1] == "launch" and argv[2] in LAUNCH_PKGS:
        return f"ros2 launch {argv[2]}"
    if b0.startswith("python"):
        args = [a for a in argv[1:4] if not a.startswith("-")]
        for a in args[:2]:
            base = os.path.basename(a)
            if base in KEEP_SCRIPTS:
                return None
            if (base.startswith("NERO_") and base.endswith(".py")) or base in EXTRA_SCRIPTS:
                return base
        if len(args) >= 3 and os.path.basename(args[0]) == "ros2" and args[1] == "launch" and args[2] in LAUNCH_PKGS:
            return f"ros2 launch {args[2]}"
    return None


def stale_procs(procs=None, me=()):
    """leftover NERO / ROS processes -> [(pid, kind, cmd)], without the pids in `me` and this process' ancestors"""
    procs = proc_list() if procs is None else procs
    skip = set(int(p) for p in me) | {os.getpid(), os.getppid()}
    out = []
    for pid, argv in procs:
        if pid in skip:
            continue
        kind = classify(argv)
        if kind:
            out.append((pid, kind, " ".join(argv)[:200]))
    return out


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                          # a zombie is not alive for our purpose
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except Exception:
        return True


def stop_procs(pids, step_s=2.5, log=None, sleep=time.sleep):
    """SIGINT -> SIGTERM -> SIGKILL, each step at most step_s (whole process group too when the pid leads one)"""
    log = log or (lambda s: None)
    pids = [int(p) for p in pids]
    left = list(pids)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        for p in left:
            try:
                if os.getpgid(p) == p:
                    os.killpg(p, sig)
                else:
                    os.kill(p, sig)
            except Exception:
                pass
        t_end = time.time() + float(step_s)
        while time.time() < t_end and any(_alive(p) for p in left):
            sleep(0.1)
        left = [p for p in left if _alive(p)]
        if not left:
            break
    stopped = [p for p in pids if p not in left]
    log(f"[{'OK' if not left else 'WARN'}] [V3.77 STALE] stopped {len(stopped)} leftover process(es)"
        + (f"; still running: {left} (stop them by hand: kill -9 {' '.join(map(str, left))})" if left else ""))
    return {"stopped": stopped, "left": left}


if __name__ == "__main__":
    for opt in ("--list", "--stop"):
        if opt in sys.argv:
            me = [int(a) for a in sys.argv[sys.argv.index(opt) + 1:] if a.isdigit()]
            hits = stale_procs(me=me)
            if opt == "--list":
                for pid, kind, cmd in hits:
                    print(pid, cmd)
            elif hits:
                stop_procs([h[0] for h in hits], log=print)
