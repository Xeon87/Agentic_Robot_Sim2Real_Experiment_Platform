#!/usr/bin/env python3
"""NERO V3.13 real-robot bring-up: logic for the console's 实机 NERO tab + helper subprocesses.

V3.13 REAL ROBOT PIPELINE (six steps, each must be green before the next):
  1 环境       ROS2 + agx_arm_ctrl + pyAgxArm + can-utils + USB-CAN present; no leftover ROS/NERO processes; sim not running
  2 CAN 口     can0 UP / ERROR-ACTIVE / 1 Mbit/s  (configured with ONE pkexec password prompt when needed)
  3 CAN push   pyAgxArm (firmware 1.11): connect, set_normal_mode() until enable() succeeds  (= the user's nero_start.py)
  4 CAN 流量   raw SocketCAN listener: frames keep flowing (251-257, 261-267, 2A1-2A9); 517/518 = gripper ID not changed
  5 ROS2 驱动  ros2 launch agx_arm_ctrl start_single_agx_arm_moveit.launch.py (arguments built here, arm_type=nero fixed);
               /feedback/joint_states rate ~200 Hz, no "Failed to get firmware version"
  6 MoveIt/RViz  /move_group + MoveGroup action server present, rviz2 running

This file is imported by NERO_V312_Console.py (no Tk here) and is also the helper program:
  python3 NERO_V313_Real_Robot.py push    --channel can0 --fw v111 --timeout 15   (pyAgxArm, JSON lines on stdout)
  python3 NERO_V313_Real_Robot.py monitor                                        (rclpy, needs a sourced ROS env;
                                                                                   JSON lines out, JSON commands in)
Only the standard library is needed for the logic; pyAgxArm / rclpy are imported inside the helper commands only.
"""
import json
import math
import os
import queue
import re
import shlex
import signal
import socket
import struct
import subprocess
import sys
import threading
import time

VERSION = "V3.47.1"
HERE = os.path.dirname(os.path.realpath(__file__))

ROS_DISTRO = os.environ.get("ROS_DISTRO") or "humble"
ROS_SETUP = f"/opt/ros/{ROS_DISTRO}/setup.bash"
AGX_ARM_WS = os.environ.get("AGX_ARM_WS", os.path.expanduser("~/agx_arm_ws"))
AGX_SETUP = os.path.join(AGX_ARM_WS, "install", "setup.bash")
DEFAULT_DOMAIN_ID = os.environ.get("NERO_V313_ROS_DOMAIN_ID", "42")

GREY, YELLOW, GREEN, RED, AMBER = "grey", "yellow", "green", "red", "amber"
STEP_KEYS = ["env", "can", "push", "flow", "driver", "moveit"]
STEP_LABELS = ["1 环境", "2 CAN 口", "3 CAN push / 使能", "4 CAN 流量", "5 ROS2 驱动", "6 MoveIt / RViz"]

# CAN id groups seen on a pushing NERO (V3.13 handoff notes / candump of the user's arm)
CAN_GROUPS = {
    "joint_fb_251_257": range(0x251, 0x258),
    "drive_261_267": range(0x261, 0x268),
    "arm_2A1_2A9": range(0x2A1, 0x2AA),
}
GRIPPER_OLD_IDS = (0x517, 0x518, 0x527, 0x528)
GRIPPER_NERO_IDS = (0x51A, 0x52A)
FLOW_OK_FPS = float(os.environ.get("NERO_V313_FLOW_OK_FPS", "500"))
FLOW_SPARSE_FPS = 1.0

STALE_BINS = {"ros2_control_node", "move_group", "rviz2", "robot_state_publisher", "agx_arm_ctrl_single",
              "agx_arm_ctrl_single_node", "spawner"}
STALE_SCRIPTS = {"NERO_V2_7_1_Unified_Backend.py", "NERO_GPU_Planner_Server.py", "nero_start.py",
                 "NERO_V313_Real_Robot.py"}


# =========================================================================== small utilities
def which(name):
    for d in os.environ.get("PATH", "").split(os.pathsep) + ["/usr/sbin", "/sbin", "/usr/bin", "/bin"]:
        p = os.path.join(d, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


QT_ENV_KEYS = ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH", "QT_QPA_FONTDIR")


def clean_child_env(env=None):
    """V3.46 QT ENV GUARD: pip opencv-python points QT_QPA_PLATFORM_PLUGIN_PATH (and friends) at its bundled
    cv2/qt/plugins the first time cv2 is imported; a child started afterwards (ros2 launch -> rviz2) then loads that
    incompatible Qt plugin and aborts ('Could not load the Qt platform plugin "xcb" in .../cv2/qt/plugins').
    Every child of the console gets an environment without the cv2 entries."""
    e = dict(os.environ if env is None else env)
    for k in QT_ENV_KEYS:
        v = e.get(k)
        if v and "cv2" in v:
            keep = [x for x in v.split(os.pathsep) if x and "cv2" not in x]
            if keep:
                e[k] = os.pathsep.join(keep)
            else:
                e.pop(k, None)
    return e


def run(cmd, timeout=10.0, env=None):
    """Run a command, never raise: returns (rc, stdout, stderr)."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=clean_child_env(env),
                           shell=isinstance(cmd, str), executable="/bin/bash" if isinstance(cmd, str) else None)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except Exception as exc:
        return 127, "", str(exc)


def ros_env_prefix(domain_id):
    """Shell prefix that sources ROS2 + the AgileX workspace and sets the domain."""
    return (f"source {shlex.quote(ROS_SETUP)} >/dev/null 2>&1; source {shlex.quote(AGX_SETUP)} >/dev/null 2>&1; "
            f"export ROS_DOMAIN_ID={int(domain_id)}; ")


def proc_list(proc_root="/proc"):
    """[(pid, argv)] of every process (Linux)."""
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


def find_stale(procs, me=None):
    """Leftover ROS / NERO processes (argv[0] or the python/ros2 script name), excluding our own pids."""
    me = set(me or ())
    hits = []
    for pid, argv in procs:
        if pid in me:
            continue
        b0 = os.path.basename(argv[0])
        hit = b0 in STALE_BINS
        if not hit and b0.startswith("python"):
            hit = any(os.path.basename(a) in STALE_SCRIPTS for a in argv[1:3] if not a.startswith("-"))
            hit = hit or (len(argv) >= 4 and os.path.basename(argv[1]) == "ros2" and argv[2] == "launch"
                          and argv[3] in ("agx_arm_ctrl", "agx_arm_moveit"))
        if not hit and b0 == "ros2" and len(argv) >= 3 and argv[1] == "launch" and argv[2] in ("agx_arm_ctrl", "agx_arm_moveit"):
            hit = True
        if hit:
            hits.append((pid, " ".join(argv)[:160]))
    return hits


def sim_running(procs):
    for _pid, argv in procs:
        if any(os.path.basename(a) == "NERO_V2_7_1_Unified_Backend.py" for a in argv[1:3]):
            return True
    return False


# =========================================================================== step 1: environment
def check_environment(procs=None, sim_runner_active=False, me=None):
    """Returns (state, items[(name, ok, detail)], stale[(pid, cmd)])."""
    procs = proc_list() if procs is None else procs
    items = []
    items.append(("ROS2 " + ROS_DISTRO, os.path.isfile(ROS_SETUP), ROS_SETUP))
    items.append(("agx_arm_ws", os.path.isfile(AGX_SETUP), AGX_SETUP))
    pkg_ok = False
    pkg_detail = "agx_arm_ctrl not found"
    if os.path.isfile(ROS_SETUP) and os.path.isfile(AGX_SETUP):
        rc, out, err = run(ros_env_prefix(0) + "ros2 pkg prefix agx_arm_ctrl", timeout=20)
        pkg_ok = rc == 0 and bool(out.strip())
        pkg_detail = out.strip() or err.strip()[:120]
    items.append(("agx_arm_ctrl", pkg_ok, pkg_detail))
    rc, out, err = run([sys.executable, "-c", "import pyAgxArm,os;print(os.path.dirname(pyAgxArm.__file__))"], timeout=20)
    items.append(("pyAgxArm", rc == 0, (out or err).strip()[:120]))
    items.append(("can-utils", bool(which("candump") and which("cansend")), which("candump") or "sudo apt install can-utils"))
    rc, out, err = run(["ip", "-details", "link", "show", CAN_PORT_DEFAULT], timeout=5)
    items.append(("USB-CAN " + CAN_PORT_DEFAULT, rc == 0, (out.split("\n")[0] if rc == 0 else (err or "no such device")).strip()[:120]))
    items.append(("pkexec (CAN setup)", bool(which("pkexec")), which("pkexec") or "policykit-1 missing"))
    stale = find_stale(procs, me=me)
    items.append(("no leftover ROS/NERO processes", not stale, "; ".join(c for _p, c in stale)[:200] or "none"))
    sim = sim_runner_active or sim_running(procs)
    items.append(("simulation not running", not sim, "sim backend running" if sim else "ok"))
    required = [ok for name, ok, _d in items]
    return (GREEN if all(required) else RED), items, stale


CAN_PORT_DEFAULT = os.environ.get("NERO_V313_CAN_PORT", "can0")


# =========================================================================== step 2: CAN port
def parse_ip_details(json_text):
    """`ip -details -statistics -json link show can0` -> compact dict."""
    try:
        d = json.loads(json_text)[0]
    except Exception:
        return None
    info = (d.get("linkinfo") or {}).get("info_data") or {}
    bt = info.get("bittiming") or {}
    stats = d.get("stats64") or d.get("stats") or {}
    rx, tx = stats.get("rx") or {}, stats.get("tx") or {}
    berr = info.get("berr_counter") or {}
    up = "UP" in (d.get("flags") or []) and d.get("operstate") in ("UP", "UNKNOWN")
    return {"up": bool(up), "operstate": d.get("operstate"), "flags": d.get("flags") or [],
            "state": info.get("state"), "bitrate": bt.get("bitrate"), "restart_ms": info.get("restart_ms"),
            "rx_packets": rx.get("packets"), "rx_errors": rx.get("errors"), "tx_packets": tx.get("packets"),
            "tx_errors": tx.get("errors"), "tx_err_counter": berr.get("tx"), "rx_err_counter": berr.get("rx")}


def can_port_status(port):
    rc, out, err = run(["ip", "-details", "-statistics", "-json", "link", "show", port], timeout=5)
    if rc != 0:
        return None, (err or out).strip()
    return parse_ip_details(out), ""


def evaluate_can_port(st):
    if not st:
        return RED, "CAN port missing"
    if not st["up"]:
        return RED, f"{st.get('operstate')} (not UP)"
    if st.get("bitrate") != 1000000:
        return RED, f"bitrate {st.get('bitrate')} != 1000000"
    if st.get("state") != "ERROR-ACTIVE":
        return RED, f"CAN state {st.get('state')}"
    return GREEN, "UP / ERROR-ACTIVE / 1 Mbit/s"


def can_setup_command(port):
    """ONE pkexec prompt for down / bitrate / up (V3.12 choice B: password dialog every time)."""
    ip = which("ip") or "/usr/sbin/ip"
    p = shlex.quote(port)
    # V3.13.1: gs_usb adapters reject restart-ms ("Device doesn't support restart from Bus Off") -> retry without it
    script = (f"{ip} link set {p} down 2>/dev/null; "
              f"{{ {ip} link set {p} type can bitrate 1000000 restart-ms 100 2>/dev/null "
              f"|| {ip} link set {p} type can bitrate 1000000; }} && {ip} link set {p} up")
    return ["pkexec", "/bin/sh", "-c", script]


# =========================================================================== step 4: CAN flow monitor
CAN_FRAME_FMT = "<IB3x8s"
CAN_FRAME_SIZE = struct.calcsize(CAN_FRAME_FMT)
CAN_EFF_FLAG, CAN_ERR_FLAG = 0x80000000, 0x20000000


def parse_can_frame(raw):
    can_id, dlc, data = struct.unpack(CAN_FRAME_FMT, raw[:CAN_FRAME_SIZE])
    return can_id, dlc, data[:dlc]


def open_can_socket(port):
    s = socket.socket(socket.PF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
    s.bind((port,))
    s.settimeout(0.5)
    return s


class CanMonitor:
    """Read-only listener (a second SocketCAN reader never disturbs the bus). Keeps per-ID counts and a 1 s rate."""

    def __init__(self, port, opener=open_can_socket, window_s=1.0):
        self.port, self.opener, self.window_s = port, opener, window_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self.error = None
        self.reset()

    def reset(self):
        with getattr(self, "_lock", threading.Lock()):
            self.counts = {}
            self.total = 0
            self.error_frames = 0
            self._times = []
            self.last_data = {}

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.error = None
        self._thread = threading.Thread(target=self._loop, daemon=True, name="v313_can_monitor")
        self._thread.start()

    def stop(self):
        self._stop.set()

    def running(self):
        return bool(self._thread and self._thread.is_alive())

    def _loop(self):
        try:
            sock = self.opener(self.port)
        except Exception as exc:
            self.error = f"cannot open {self.port}: {exc}"
            return
        try:
            while not self._stop.is_set():
                try:
                    raw = sock.recv(CAN_FRAME_SIZE)
                except socket.timeout:
                    continue
                except Exception as exc:
                    self.error = str(exc)
                    break
                if not raw:
                    continue
                self.feed(raw)
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def feed(self, raw, now=None):
        can_id, _dlc, data = parse_can_frame(raw)
        now = time.time() if now is None else now
        with self._lock:
            if can_id & CAN_ERR_FLAG:
                self.error_frames += 1
                return
            cid = can_id & (0x1FFFFFFF if can_id & CAN_EFF_FLAG else 0x7FF)
            self.counts[cid] = self.counts.get(cid, 0) + 1
            self.last_data[cid] = data
            self.total += 1
            self._times.append(now)
            cut = now - self.window_s
            if len(self._times) > 50000 or (self._times and self._times[0] < cut - 5):
                self._times = [t for t in self._times if t >= cut]

    def fps(self, now=None):
        now = time.time() if now is None else now
        with self._lock:
            cut = now - self.window_s
            self._times = [t for t in self._times if t >= cut]
            return len(self._times) / self.window_s

    def snapshot(self, now=None):
        f = self.fps(now)
        with self._lock:
            ids = set(self.counts)
            groups = {g: any(i in ids for i in r) for g, r in CAN_GROUPS.items()}
            return {"fps": round(f, 1), "total": self.total, "error_frames": self.error_frames, "groups": groups,
                    "gripper_old_id": sorted(hex(i) for i in ids if i in GRIPPER_OLD_IDS),
                    "gripper_nero_id": sorted(hex(i) for i in ids if i in GRIPPER_NERO_IDS),
                    "n_ids": len(ids), "error": self.error}


def evaluate_flow(snap):
    """Returns (state, message). Yellow = only sporadic frames: the arm is not pushing yet (do step 3)."""
    if snap.get("error"):
        return RED, snap["error"]
    fps = snap.get("fps", 0.0)
    if fps < FLOW_SPARSE_FPS:
        return RED, "no CAN frames"
    groups = snap.get("groups", {})
    if fps < FLOW_OK_FPS or not all(groups.values()):
        missing = [g for g, ok in groups.items() if not ok]
        return YELLOW, f"{fps:.0f} fps, missing {missing or '-'}: arm not pushing (step 3)"
    msg = f"{fps:.0f} fps, all arm groups refreshing"
    if snap.get("gripper_old_id"):
        return AMBER, msg + f"; gripper still on old ID {snap['gripper_old_id']} -> keep effector=none"
    return GREEN, msg


def push_release_decision(fps_with_push, fps_after_release):
    """After step 3, the push helper is stopped once: does the arm keep pushing on its own?"""
    if fps_with_push <= 0:
        return "unknown"
    return "released" if fps_after_release >= 0.5 * fps_with_push else "keep_until_driver"


# =========================================================================== step 5/6: driver launch + ROS checks
def launch_args(cfg):
    """Argument list for start_single_agx_arm_moveit.launch.py. arm_type is fixed to nero."""
    eff = cfg.get("effector_type", "none")
    if eff not in ("none", "agx_gripper"):
        eff = "none"
    speed = int(cfg.get("speed_percent", 10))
    if not cfg.get("speed_unlocked"):
        speed = max(1, min(30, speed))
    speed = max(1, min(100, speed))
    args = [f"can_port:={cfg.get('can_port', 'can0')}", "arm_type:=nero", f"effector_type:={eff}",
            f"follow:={'true' if cfg.get('follow', True) else 'false'}",
            f"auto_control_gate:={'true' if cfg.get('auto_control_gate', False) else 'false'}",
            "fast_mode:=false", f"speed_percent:={speed}",
            f"use_rviz:={'true' if cfg.get('use_rviz', True) else 'false'}"]
    fw = str(cfg.get("fw_version", "v111") or "").strip()
    if fw:
        args.append(f"fw_version:={fw}")
    return args


def launch_command(cfg):
    return "ros2 launch agx_arm_ctrl start_single_agx_arm_moveit.launch.py " + " ".join(launch_args(cfg))


def launch_shell(cfg):
    return ros_env_prefix(cfg.get("domain_id", DEFAULT_DOMAIN_ID)) + "exec " + launch_command(cfg)


DRIVER_FAIL_PATTERNS = ("Failed to get firmware version", "process has died", "Traceback (most recent call last)")


def is_rviz_death(line):
    """V3.46: 'process has died' of rviz2 only (visualisation) - not a driver failure."""
    return "process has died" in line and ("rviz2" in line or "[rviz2-" in line)


def driver_log_failure(line):
    if is_rviz_death(line):
        return None
    return next((p for p in DRIVER_FAIL_PATTERNS if p in line), None)


def evaluate_driver(mon, launch_alive, log_failure):
    if log_failure:
        return RED, f"driver log: {log_failure}"
    if not launch_alive:
        return RED, "launch process not running"
    if not mon:
        return YELLOW, "waiting for ROS monitor"
    hz = float(mon.get("js_hz") or 0.0)
    if hz < 50.0:
        return YELLOW, f"/feedback/joint_states {hz:.0f} Hz (waiting)"
    return GREEN, f"/feedback/joint_states {hz:.0f} Hz, {len((mon.get('joints') or {}).get('names') or [])} joints"


def evaluate_moveit(mon, rviz_alive, use_rviz=True, rviz_died=None):
    if not mon:
        return YELLOW, "waiting for ROS monitor"
    if not mon.get("has_move_group"):
        return YELLOW, "waiting for /move_group"
    if not mon.get("has_move_action"):
        return YELLOW, "waiting for MoveGroup action server"
    if use_rviz and not rviz_alive and rviz_died:
        return AMBER, f"/move_group + MoveGroup action OK; RViz exited ({rviz_died}) - only the 3D view is missing"
    if use_rviz and not rviz_alive:
        return YELLOW, "waiting for rviz2"
    return GREEN, "/move_group + MoveGroup action" + (" + RViz" if use_rviz else "")


def rviz_alive(procs=None):
    procs = proc_list() if procs is None else procs
    return any(os.path.basename(argv[0]) == "rviz2" for _p, argv in procs)


# =========================================================================== generic process runner
class ProcRunner:
    """A child in its own session; stdout lines (or JSON events) go to a queue; exit reported even if a grandchild
    keeps the pipe open.  stdin stays open (PIPE) so the monitor helper can receive commands."""

    def __init__(self, tag, log_path=None):
        self.tag, self.log_path = tag, log_path
        self.proc = None
        self.lines = queue.Queue()
        self.exit_code = None
        self.stop_requested = False
        self._wlock = threading.Lock()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, argv, env=None, cwd=None, stdin_pipe=False):
        if self.running():
            raise RuntimeError(f"{self.tag} already running")
        self.exit_code, self.stop_requested = None, False
        self.proc = subprocess.Popen(argv, cwd=cwd or HERE, env=clean_child_env(env), start_new_session=True, text=True, bufsize=1,
                                     errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.PIPE if stdin_pipe else subprocess.DEVNULL)
        rd = threading.Thread(target=self._reader, args=(self.proc,), daemon=True)
        rd.start()
        threading.Thread(target=self._watch, args=(self.proc, rd), daemon=True).start()
        return self.proc.pid

    def _reader(self, proc):
        logf = None
        try:
            logf = open(self.log_path, "a", encoding="utf-8") if self.log_path else None
        except Exception:
            logf = None
        try:
            for line in proc.stdout:
                self.lines.put(line.rstrip("\n"))
                if logf:
                    logf.write(f"[{self.tag}] {line}")
                    logf.flush()
        except Exception:
            pass
        if logf:
            logf.close()

    def _watch(self, proc, rd):
        rc = proc.wait()
        rd.join(timeout=2.0)
        self.exit_code = rc
        self.lines.put(None)

    def send(self, obj):
        if not self.running() or self.proc.stdin is None:
            return False
        with self._wlock:
            try:
                self.proc.stdin.write(json.dumps(obj) + "\n")
                self.proc.stdin.flush()
                return True
            except Exception:
                return False

    def stop(self, sig=signal.SIGINT, grace_s=15.0):
        if not self.running():
            return
        self.stop_requested = True
        try:
            os.killpg(self.proc.pid, sig)        # ros2 launch forwards SIGINT to its children
        except Exception:
            pass
        threading.Thread(target=self._escalate, args=(self.proc, grace_s), daemon=True).start()

    def _escalate(self, proc, grace_s):
        t_end = time.time() + grace_s
        while time.time() < t_end:
            if proc.poll() is not None:
                return
            time.sleep(0.2)
        self.lines.put(f"[V3.13] {self.tag} still alive after {grace_s:.0f}s -> SIGKILL")
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def helper_argv(cmd, *args):
    return [sys.executable, "-u", os.path.join(HERE, "NERO_V313_Real_Robot.py"), cmd] + list(args)


def monitor_shell(domain_id):
    return ros_env_prefix(domain_id) + "exec " + " ".join(shlex.quote(a) for a in helper_argv("monitor"))


# =========================================================================== helper: push (pyAgxArm)
def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    for attr in ("_asdict", "to_dict", "__dict__"):
        try:
            d = getattr(v, attr)
            d = d() if callable(d) else d
            if isinstance(d, dict):
                return {str(k): _jsonable(x) for k, x in d.items() if not str(k).startswith("__")}
        except Exception:
            pass
    return str(v)


def _emit(**kw):
    kw.setdefault("t", round(time.time(), 3))
    print(json.dumps(kw, ensure_ascii=False), flush=True)


def cmd_push(channel="can0", fw="v111", timeout_s=15.0, period_s=0.5, max_reports=0):
    """The user's nero_start.py: connect, set_normal_mode() until enable() succeeds, then report joints/status
    every period_s until SIGINT/SIGTERM (max_reports > 0 only for tests)."""
    stop = threading.Event()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, lambda *_a: stop.set())
        except Exception:
            pass
    try:
        from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW
    except Exception as exc:
        _emit(event="error", stage="import", error=f"pyAgxArm import failed: {exc}")
        return 1
    fw_attr = str(fw or "v111").upper()
    if not hasattr(NeroFW, fw_attr):
        _emit(event="error", stage="config", error=f"NeroFW has no {fw_attr}")
        return 1
    try:
        cfg = create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=getattr(NeroFW, fw_attr),
                                    interface="socketcan", channel=channel)
        robot = AgxArmFactory.create_arm(cfg)
        _emit(event="connecting", channel=channel, fw=fw_attr)
        robot.connect()
        _emit(event="connected")
    except Exception as exc:
        _emit(event="error", stage="connect", error=str(exc)[:300])
        return 1
    t_end = time.time() + float(timeout_s)
    attempts = 0
    enabled = False
    while not stop.is_set() and time.time() < t_end:
        attempts += 1
        try:
            if robot.enable():
                enabled = True
                break
            robot.set_normal_mode()
        except Exception as exc:
            _emit(event="warn", error=str(exc)[:200])
        time.sleep(0.01)
    if not enabled:
        _emit(event="timeout" if not stop.is_set() else "stopped", attempts=attempts)
        return 2 if not stop.is_set() else 0
    _emit(event="enabled", attempts=attempts)
    n = 0
    while not stop.is_set():
        try:
            _emit(event="state", joints=_jsonable(robot.get_joint_angles()), status=_jsonable(robot.get_arm_status()))
        except Exception as exc:
            _emit(event="warn", error=str(exc)[:200])
        n += 1
        if max_reports and n >= max_reports:
            break
        stop.wait(period_s)
    _emit(event="stopped")
    return 0


TEACH_CTRL_MODES = (0x02, 0x06)       # TEACHING_MODE, LINKAGE_TEACHING_INPUT_MODE (pyAgxArm nero_api.md)


def _find_key(d, key):
    if isinstance(d, dict):
        if key in d:
            return d[key]
        for v in d.values():
            r = _find_key(v, key)
            if r is not None:
                return r
    elif isinstance(d, (list, tuple)):
        for v in d:
            r = _find_key(v, key)
            if r is not None:
                return r
    return None


def ctrl_mode_of(status):
    """int ctrl_mode from a pyAgxArm / ROS arm status (dict, object, enum or 'CtrlMode.X' string); None if unknown."""
    v = _find_key(_jsonable(status), "ctrl_mode")
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    sv = str(v).upper()
    m = re.search(r"(0X[0-9A-F]+|\b\d+\b)", sv)
    if m:
        try:
            return int(m.group(1), 0)
        except ValueError:
            pass
    if "LINKAGE" in sv:
        return 0x06
    if "TEACH" in sv:
        return 0x02
    if "STANDBY" in sv:
        return 0x00
    return None


def is_teach_ctrl_mode(cm):
    return cm in TEACH_CTRL_MODES


def _leader_stamp(robot):
    """(timestamp or counter, angles) of pyAgxArm get_leader_joint_angles(); (None, None) if nothing."""
    f = getattr(robot, "get_leader_joint_angles", None)
    if not callable(f):
        return None, None
    try:
        v = f()
    except Exception:
        return None, None
    a = _angles7(v)
    if not a:
        return None, None
    ts = None
    for attr in ("timestamp", "_timestamp"):
        try:
            ts = v[attr] if isinstance(v, dict) else getattr(v, attr)
        except Exception:
            ts = None
        if ts is not None:
            break
    return ts, a


def _call_first(robot, names, *args):
    """V3.22: call the first available pyAgxArm method of names -> name used or None."""
    for n in names:
        f = getattr(robot, n, None)
        if callable(f):
            try:
                f(*args)
                return n
            except TypeError:
                continue
            except Exception:
                return None
    return None


SPEED_APIS = ("set_speed_percent", "set_motion_speed", "set_speed", "set_joint_speed_percent")
MOVEJ_APIS = ("move_j", "move_joint", "joint_ctrl", "set_joint_angles")


def cmd_mode(channel="can0", fw="v111", mode="teach", enable_timeout_s=6.0, verify_s=3.0, hold=None, speed_pct=10):
    """V3.19 TEACH MODE (V3.20: correct API + verification). The ROS driver offers no way to enter hand guiding.
    teach  -> set_leader_mode() (pyAgxArm nero_api: leader = linkage teaching input, moved by hand); older
              pyAgxArm: set_master_mode(). Verified: ctrl_mode must become 0x06 / 0x02.
    normal -> set_normal_mode() (fw <= 1.11) or set_follower_mode() (fw >= 1.12), then enable().
    AgileX: wait ~1 s before and after a mode switch. Prints a 'diag' line (available mode APIs, ctrl_mode)."""
    try:
        from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW
    except Exception as exc:
        _emit(event="mode_result", ok=False, mode=mode, error=f"pyAgxArm import failed: {exc}")
        return 1
    fw_attr = str(fw or "v111").upper()
    try:
        cfg = create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=getattr(NeroFW, fw_attr),
                                    interface="socketcan", channel=channel)
        robot = AgxArmFactory.create_arm(cfg)
        robot.connect()
        time.sleep(1.0)

        def status():
            try:
                return robot.get_arm_status()
            except Exception:
                return None
        apis = sorted(m for m in dir(robot) if not m.startswith("_") and any(
            k in m for k in ("mode", "leader", "master", "follower", "teach", "drag", "enable", "disable")))
        cm0 = ctrl_mode_of(status())
        _emit(event="diag", apis=apis, ctrl_mode_before=cm0, fw=fw_attr)
        if mode == "teach":
            fn_name = next((n for n in ("set_leader_mode", "set_master_mode") if callable(getattr(robot, n, None))), None)
            if fn_name is None:
                _emit(event="mode_result", ok=False, mode=mode, error="pyAgxArm 没有 set_leader_mode / set_master_mode",
                      apis=apis, ctrl_mode=cm0)
                return 2
            ts0, a0 = _leader_stamp(robot)
            getattr(robot, fn_name)()
            time.sleep(1.0)
            # V3.22: on NERO V111 ctrl_mode may keep reporting 0x01 although the arm is limp and streams leader
            # angles (CAN 0x501-0x507) -> leader angles that start / keep updating also count as "entered".
            cm, evidence, t_end = None, None, time.time() + float(verify_s)
            while time.time() < t_end:
                cm = ctrl_mode_of(status())
                if is_teach_ctrl_mode(cm):
                    evidence = f"ctrl_mode=0x{cm:02X}"
                    break
                ts, a = _leader_stamp(robot)
                if a is not None and (a0 is None or (ts is not None and ts != ts0)):
                    evidence = "主导臂角度在更新（get_leader_joint_angles）"
                    break
                time.sleep(0.1)
            ok = evidence is not None
            _emit(event="mode_result", ok=ok, mode=mode, api=fn_name, ctrl_mode=cm, ctrl_mode_before=cm0,
                  evidence=evidence, status=_jsonable(status()),
                  error=None if ok else f"{fn_name}() 已发送，但 ctrl_mode = {cm} 且没有主导臂角度")
            return 0 if ok else 2
        fn_name = next((n for n in (("set_follower_mode", "set_normal_mode") if fw_attr >= "V112"
                                    else ("set_normal_mode", "set_follower_mode"))
                        if callable(getattr(robot, n, None))), None)
        if fn_name:
            getattr(robot, fn_name)()
            time.sleep(1.0)
        # V3.22 SLOW EXIT: low speed + hold the current pose BEFORE the motors are enabled again, so the firmware
        # does not jump back to its last target (the zero pose) at full speed.
        sp_api = _call_first(robot, SPEED_APIS, int(speed_pct))
        mj_api = _call_first(robot, MOVEJ_APIS, list(hold)) if hold else None
        enabled, t_end = False, time.time() + float(enable_timeout_s)
        while time.time() < t_end:
            if robot.enable():
                enabled = True
                break
            if fn_name:
                getattr(robot, fn_name)()
            time.sleep(0.05)
        if enabled:
            _call_first(robot, SPEED_APIS, int(speed_pct))
            if hold:
                for _k in range(5):
                    mj_api = _call_first(robot, MOVEJ_APIS, list(hold)) or mj_api
                    time.sleep(0.1)
        _emit(event="diag", hold=hold, speed_api=sp_api, movej_api=mj_api, speed_pct=speed_pct)
        cm = ctrl_mode_of(status())
        ok = bool(enabled) and not is_teach_ctrl_mode(cm)
        _emit(event="mode_result", ok=ok, mode=mode, api=fn_name, enabled=enabled, ctrl_mode=cm, ctrl_mode_before=cm0,
              status=_jsonable(status()), speed_api=sp_api, movej_api=mj_api,
              error=None if ok else (f"ctrl_mode 仍是 {cm}（示教）" if enabled else "enable() 没有成功"))
        return 0 if ok else 2
    except Exception as exc:
        _emit(event="mode_result", ok=False, mode=mode, error=str(exc)[:300])
        return 1


def _angles7(v):
    """7 joint angles (rad) from a pyAgxArm reading (list, message object with .msg / ._msg, dict)."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)) and len(v) >= 7 and all(isinstance(x, (int, float)) for x in v[:7]):
        return [float(x) for x in v[:7]]
    for attr in ("msg", "_msg", "joints", "position", "angles"):
        try:
            inner = v[attr] if isinstance(v, dict) else getattr(v, attr)
        except Exception:
            continue
        r = _angles7(inner)
        if r:
            return r
    if isinstance(v, dict):
        for inner in v.values():
            r = _angles7(inner) if isinstance(inner, (list, tuple, dict)) else None
            if r:
                return r
    return None


def cmd_leader(channel="can0", fw="v111", period_s=0.05, max_reports=0):
    """V3.21 LEADER READER: while NERO is in leader (hand-guiding) mode, stream the hand-guided joint angles with
    pyAgxArm get_leader_joint_angles() (V111: CAN 0x501-0x507) — the ROS driver may not publish them for NERO.
    Also reports get_joint_angles() so the GUI can see which source moves."""
    stop = threading.Event()
    for sgn in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sgn, lambda *_a: stop.set())
        except Exception:
            pass
    try:
        from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW
        cfg = create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=getattr(NeroFW, str(fw or "v111").upper()),
                                    interface="socketcan", channel=channel)
        robot = AgxArmFactory.create_arm(cfg)
        robot.connect()
    except Exception as exc:
        _emit(event="leader_error", error=f"pyAgxArm: {exc}"[:300])
        return 1
    has_leader = callable(getattr(robot, "get_leader_joint_angles", None))
    _emit(event="leader_ready", has_leader_api=has_leader)
    n = 0
    while not stop.is_set():
        out = {}
        for key, fn in (("leader", "get_leader_joint_angles"), ("joints", "get_joint_angles")):
            f = getattr(robot, fn, None)
            if callable(f):
                try:
                    a = _angles7(f())
                    if a:
                        out[key] = [round(x, 5) for x in a]
                except Exception:
                    pass
        _emit(event="leader", **out)
        n += 1
        if max_reports and n >= max_reports:
            break
        stop.wait(period_s)
    return 0


# =========================================================================== helper: monitor (rclpy)
MONITORED_SERVICES = ("emergency_stop", "enable_agx_arm", "control_enable", "move_home", "exit_teach_mode")


def cmd_monitor(period_s=1.0):
    """Persistent rclpy node. Every period_s prints one JSON status line; reads JSON commands from stdin:
       {"call": "/emergency_stop", "value": null|true|false, "id": n}  -> {"event": "call_result", ...}
       V3.14: {"move": {...}} (MoveGroup action joint goal), {"cancel": true}, {"get_model": true};
       fast {"event": "js"} lines (10 Hz), {"event": "model"} once URDF/SRDF are known, TCP from TF in status."""
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from rosidl_runtime_py.utilities import get_message, get_service
        from rosidl_runtime_py.convert import message_to_ordereddict
    except Exception as exc:
        _emit(event="error", stage="import", error=f"rclpy unavailable (ROS env not sourced?): {exc}")
        return 1
    rclpy.init()
    node = Node("nero_v313_console_monitor")
    qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
    st = {"js_stamps": [], "joints": None, "arm_status": None, "gripper_status": None, "subs": {}}
    cmds = queue.Queue()

    def stdin_reader():
        for line in sys.stdin:
            try:
                cmds.put(json.loads(line))
            except Exception:
                pass
    threading.Thread(target=stdin_reader, daemon=True).start()

    def on_ljs(msg):                 # V3.20: leader / teaching-input joint angles (hand guiding)
        st["ljs"] = {"names": list(msg.name), "position": [round(float(v), 5) for v in msg.position]}
        st["ljs_new"] = True

    def on_js(msg):
        st["js_stamps"].append(time.time())
        st["joints"] = {"names": list(msg.name), "position": [round(float(v), 5) for v in msg.position]}
        st["js_new"] = True

    # ---- V3.14 JOINT CONTROL: robot model (URDF/SRDF), MoveGroup action, TF ----------------------------------
    try:
        import NERO_V314_Joint_Control as JC
    except Exception as exc:          # pragma: no cover
        JC = None
        _emit(event="warn", error=f"joint control module missing: {exc}")
    st.update({"urdf": None, "srdf": None, "model_sent": None, "desc_subs": False, "param_t": 0.0,
               "action": None, "action_name": None, "goal": None, "move": None, "tf": None, "tf_err": None})

    def _desc_cb(key):
        def cb(msg):
            st[key] = getattr(msg, "data", None) or st[key]
        return cb

    def ensure_desc():
        if st["desc_subs"]:
            return
        try:
            from rclpy.qos import DurabilityPolicy, ReliabilityPolicy as _RP
            from std_msgs.msg import String
            q = QoSProfile(depth=1, reliability=_RP.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            node.create_subscription(String, "/robot_description", _desc_cb("urdf"), q)
            node.create_subscription(String, "/robot_description_semantic", _desc_cb("srdf"), q)
        except Exception as exc:
            st["desc_err"] = str(exc)
        st["desc_subs"] = True

    def try_params():
        """Fallback: robot_description(_semantic) as parameters of /move_group or /robot_state_publisher."""
        if (st["urdf"] and st["srdf"]) or time.time() - st["param_t"] < 5.0:
            return
        st["param_t"] = time.time()
        try:
            from rcl_interfaces.srv import GetParameters
        except Exception:
            return
        for nd, keys in (("/move_group", ("robot_description", "robot_description_semantic")),
                         ("/robot_state_publisher", ("robot_description",))):
            want = [k for k in keys if not st["urdf" if k == "robot_description" else "srdf"]]
            if not want:
                continue
            try:
                cli = node.create_client(GetParameters, nd + "/get_parameters")
                if not cli.wait_for_service(timeout_sec=0.5):
                    node.destroy_client(cli)
                    continue
                req = GetParameters.Request()
                req.names = list(want)
                fut = cli.call_async(req)
                rclpy.spin_until_future_complete(node, fut, timeout_sec=2.0)
                if fut.done() and fut.result() is not None:
                    for k, v in zip(want, fut.result().values):
                        if getattr(v, "string_value", ""):
                            st["urdf" if k == "robot_description" else "srdf"] = v.string_value
                node.destroy_client(cli)
            except Exception as exc:
                st["param_err"] = str(exc)[:200]

    def maybe_emit_model(force=False):
        if JC is None:
            return
        key = (bool(st["urdf"]), bool(st["srdf"]), len(st["urdf"] or ""), len(st["srdf"] or ""))
        if not force and key == st["model_sent"]:
            return
        if not st["urdf"] and not force:
            return
        fb = (st["joints"] or {}).get("names")
        m = JC.build_model(st["urdf"], st["srdf"], fb, source="ROS /robot_description" if st["urdf"] else "fallback")
        st["model"] = m
        st["model_sent"] = key
        _emit(event="model", model=m)
        if st["urdf"]:
            _emit(event="urdf", xml=st["urdf"])              # V3.15: the GUI compares it with the sim URDF
            try:
                st["urdf_parsed"] = JC.parse_urdf(st["urdf"])
            except Exception:
                st["urdf_parsed"] = None

    def ensure_tf():
        if st["tf"] is not None or st["tf_err"]:
            return
        try:
            import tf2_ros
            buf = tf2_ros.Buffer()
            st["tf"] = (buf, tf2_ros.TransformListener(buf, node))
        except Exception as exc:
            st["tf_err"] = f"tf2_ros unavailable: {exc}"[:200]

    def tf_links():
        """V3.15: TF poses of the watched links (consistency test), relative to the model base."""
        m = st.get("model") or {}
        base = m.get("base_link") or "base_link"
        out = {}
        for ln in st.get("tf_watch") or ["link7"]:
            out[ln] = tf_lookup(base, ln)
        return out

    def tf_lookup(base, tip):
        if st["tf"] is None:
            return {"error": st["tf_err"] or "TF not ready"}
        try:
            import rclpy.time
            t = st["tf"][0].lookup_transform(base, tip, rclpy.time.Time()).transform
            q = t.rotation
            return {"xyz": [round(t.translation.x, 5), round(t.translation.y, 5), round(t.translation.z, 5)],
                    "rpy_deg": JC.quat_to_rpy_deg(q.x, q.y, q.z, q.w), "base": base, "tip": tip}
        except Exception as exc:
            return {"error": f"TF {base}->{tip}: {str(exc)[:120]}", "base": base, "tip": tip}

    def tcp_pose():
        m = st.get("model") or {}
        if st["tf"] is None:
            return {"error": st["tf_err"] or "TF not ready"}
        base, tip = m.get("base_link") or "base_link", m.get("tip_link") or "link7"
        try:
            import rclpy.time
            t = st["tf"][0].lookup_transform(base, tip, rclpy.time.Time()).transform
            q = t.rotation
            return {"xyz": [round(t.translation.x, 4), round(t.translation.y, 4), round(t.translation.z, 4)],
                    "rpy_deg": JC.quat_to_rpy_deg(q.x, q.y, q.z, q.w), "base": base, "tip": tip}
        except Exception as exc:
            return {"error": f"TF {base}->{tip}: {str(exc)[:120]}", "base": base, "tip": tip}

    def ensure_action():
        if st["action"] is not None:
            return True
        try:
            from rclpy.action import ActionClient, get_action_names_and_types
            from moveit_msgs.action import MoveGroup
        except Exception as exc:
            st["action_err"] = f"moveit_msgs / rclpy.action unavailable: {exc}"
            return False
        name = "/move_action"
        try:
            for n, types in get_action_names_and_types(node):
                if "moveit_msgs/action/MoveGroup" in types:
                    name = n
                    break
        except Exception:
            pass
        st["action"] = ActionClient(node, MoveGroup, name)
        st["action_name"] = name
        return True

    def do_move(mv):
        mid = mv.get("id")
        if st["move"] and st["move"].get("phase") not in ("done",):
            _emit(event="move_result", id=mid, ok=False, error="another motion is in progress")
            return
        if not ensure_action():
            _emit(event="move_result", id=mid, ok=False, error=st.get("action_err"))
            return
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import Constraints, JointConstraint
        cli = st["action"]
        if not cli.wait_for_server(timeout_sec=2.0):
            _emit(event="move_result", id=mid, ok=False, error=f"MoveGroup action {st['action_name']} not available")
            return
        goal = MoveGroup.Goal()
        r = goal.request
        r.group_name = str(mv["group"])
        r.num_planning_attempts = 3
        r.allowed_planning_time = 5.0
        r.max_velocity_scaling_factor = float(mv.get("vel", 0.1))
        r.max_acceleration_scaling_factor = float(mv.get("acc", 0.1))
        r.start_state.is_diff = True
        c = Constraints()
        tol = float(mv.get("tol", 0.0017))
        for n, p in zip(mv["joints"], mv["positions"]):
            jc = JointConstraint()
            jc.joint_name, jc.position = str(n), float(p)
            jc.tolerance_above = jc.tolerance_below = tol
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        r.goal_constraints = [c]
        # V3.15: plan first (plan_only) when the user wants only a plan or the path fence is on; the planned
        # trajectory is checked against the fence and only then sent to /execute_trajectory.
        fence = mv.get("fence")
        two_phase = bool(mv.get("plan_only") or fence)
        goal.planning_options.plan_only = two_phase
        goal.planning_options.replan = False
        st["move"] = {"id": mid, "phase": "sending", "t0": time.time()}

        def on_fb(fb):
            ph = getattr(getattr(fb, "feedback", fb), "state", "")
            if st["move"] and st["move"]["id"] == mid and ph and ph != st["move"].get("phase"):
                st["move"]["phase"] = ph
                _emit(event="move_feedback", id=mid, phase=ph)

        def finish(**kw):
            _emit(event="move_result", id=mid, elapsed_s=round(time.time() - st["move"]["t0"], 2), **kw)
            st["move"]["phase"] = "done"
            st["goal"] = None

        def on_exec_result(fut, plan_info):
            try:
                wr = fut.result()
                code = int(wr.result.error_code.val)
                finish(ok=(code == 1), error_code=code, error=None if code == 1 else JC.moveit_error_text(code),
                       status=int(getattr(wr, "status", 0)), **plan_info)
            except Exception as exc:
                finish(ok=False, error=f"execute result error: {exc}"[:300], **plan_info)

        def execute(traj, plan_info):
            try:
                from rclpy.action import ActionClient
                from moveit_msgs.action import ExecuteTrajectory
            except Exception as exc:
                finish(ok=False, error=f"ExecuteTrajectory unavailable: {exc}"[:300], **plan_info)
                return
            if st.get("exec") is None:
                st["exec"] = ActionClient(node, ExecuteTrajectory, "/execute_trajectory")
            if not st["exec"].wait_for_server(timeout_sec=2.0):
                finish(ok=False, error="/execute_trajectory action not available (move_group capability missing?)",
                       **plan_info)
                return
            eg = ExecuteTrajectory.Goal()
            eg.trajectory = traj
            st["move"]["phase"] = "执行轨迹"
            _emit(event="move_feedback", id=mid, phase="执行轨迹")

            def on_exec_goal(f2):
                try:
                    gh2 = f2.result()
                except Exception as exc:
                    finish(ok=False, error=f"execute send failed: {exc}"[:300], **plan_info)
                    return
                if not gh2.accepted:
                    finish(ok=False, error="trajectory rejected by move_group", **plan_info)
                    return
                st["goal"] = gh2
                gh2.get_result_async().add_done_callback(lambda f3: on_exec_result(f3, plan_info))
            st["exec"].send_goal_async(eg).add_done_callback(on_exec_goal)

        def on_result(fut):
            try:
                wr = fut.result()
                res = wr.result
                code = int(res.error_code.val)
                dur, names, pts = 0.0, [], []
                jt = getattr(getattr(res, "planned_trajectory", None), "joint_trajectory", None)
                try:
                    if jt is not None and jt.points:
                        tf_ = jt.points[-1].time_from_start
                        dur = float(tf_.sec) + float(tf_.nanosec) * 1e-9
                except Exception:
                    pass
                try:
                    names = list(jt.joint_names)
                    pts = [list(p.positions) for p in jt.points]
                except Exception:
                    names, pts = [], []
                info = {"planning_time": round(float(getattr(res, "planning_time", 0.0)), 3),
                        "trajectory_s": round(dur, 2), "n_points": len(pts)}
                if code != 1 or not two_phase:
                    finish(ok=(code == 1), error_code=code, error=None if code == 1 else JC.moveit_error_text(code),
                           status=int(getattr(wr, "status", 0)), **info)
                    return
                if fence:
                    try:
                        import NERO_V315_Consistency_Safety as CS
                        urdf = st.get("urdf_parsed")
                        if urdf is None:
                            finish(ok=False, error="FENCE: robot URDF not available -> path not checked, not executed", **info)
                            return
                        viol = CS.path_fence_check(urdf, mv.get("arm_joints") or mv["joints"], names, pts,
                                                   mv.get("base") or "base_link", fence, mv.get("tool_T"))
                    except Exception as exc:
                        finish(ok=False, error=f"FENCE: check failed ({exc}) -> not executed"[:300], **info)
                        return
                    info["fence_checked"] = True
                    if viol:
                        finish(ok=False, error_code=0, fence_violation=viol,
                               error=f"FENCE: 规划路径第 {viol['waypoint'] + 1}/{viol['of']} 点越界 -> 未执行: "
                                     + CS.violation_text(viol["violations"]), **info)
                        return
                if mv.get("plan_only"):
                    finish(ok=True, error_code=1, plan_only=True, **info)
                    return
                execute(res.planned_trajectory, info)
            except Exception as exc:
                finish(ok=False, error=f"result error: {exc}"[:300])

        def on_goal(fut):
            try:
                gh = fut.result()
            except Exception as exc:
                _emit(event="move_result", id=mid, ok=False, error=f"send_goal failed: {exc}"[:300])
                st["move"]["phase"] = "done"
                return
            if not gh.accepted:
                _emit(event="move_result", id=mid, ok=False, error="goal rejected by move_group")
                st["move"]["phase"] = "done"
                return
            st["goal"] = gh
            st["move"]["phase"] = "accepted"
            _emit(event="move_accepted", id=mid, action=st["action_name"])
            gh.get_result_async().add_done_callback(on_result)
        cli.send_goal_async(goal, feedback_callback=on_fb).add_done_callback(on_goal)

    def _svc(stype_mod, stype_name, name, fill, timeout=5.0):
        mod = __import__(stype_mod, fromlist=[stype_name])
        T = getattr(mod, stype_name)
        cli = node.create_client(T, name)
        try:
            if not cli.wait_for_service(timeout_sec=2.0):
                return None, f"service {name} not available"
            req = T.Request()
            fill(req)
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
            if not fut.done():
                return None, f"{name} timeout"
            return fut.result(), None
        finally:
            node.destroy_client(cli)

    def existing_scene_ids(prefix):
        try:
            def fill(req):
                req.components.components = 2 | 4          # WORLD_OBJECT_NAMES | WORLD_OBJECT_GEOMETRY
            res, err = _svc("moveit_msgs.srv", "GetPlanningScene", "/get_planning_scene", fill)
            if res is None:
                return None
            return [o.id for o in res.scene.world.collision_objects if str(o.id).startswith(prefix)]
        except Exception:
            return None

    def do_scene(sc):
        """V3.15: replace all collision boxes with id prefix sc['prefix'] by sc['objects'] (ApplyPlanningScene)."""
        sid = sc.get("id")
        try:
            from moveit_msgs.msg import PlanningScene, CollisionObject
            from shape_msgs.msg import SolidPrimitive
            from geometry_msgs.msg import Pose
        except Exception as exc:
            _emit(event="scene_result", id=sid, ok=False, error=f"moveit_msgs/shape_msgs unavailable: {exc}"[:300])
            return
        prefix = sc.get("prefix", "v315_")
        old_ids = existing_scene_ids(prefix)
        listed = old_ids is not None
        old_ids = set(old_ids or []) | set(st.get("scene_ids") or [])
        ps = PlanningScene()
        ps.is_diff = True
        frame = sc.get("frame") or "base_link"
        new_ids = [o["id"] for o in sc.get("objects", [])]
        for oid in sorted(old_ids - set(new_ids)):
            co = CollisionObject()
            co.id = oid
            co.header.frame_id = frame
            co.operation = CollisionObject.REMOVE
            ps.world.collision_objects.append(co)
        for o in sc.get("objects", []):
            co = CollisionObject()
            co.id = o["id"]
            co.header.frame_id = frame
            prim = SolidPrimitive()
            prim.type = SolidPrimitive.BOX
            prim.dimensions = [float(v) for v in o["size"]]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (float(v) for v in o["pos"])
            yaw = float(o.get("yaw") or 0.0)            # V3.23: walls along the zone edges are rotated about z
            pose.orientation.z, pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
            co.primitives = [prim]
            co.primitive_poses = [pose]
            co.operation = CollisionObject.ADD
            ps.world.collision_objects.append(co)
        try:
            def fill(req):
                req.scene = ps
            res, err = _svc("moveit_msgs.srv", "ApplyPlanningScene", "/apply_planning_scene", fill)
        except Exception as exc:
            res, err = None, str(exc)[:300]
        ok = res is not None and bool(getattr(res, "success", False))
        if ok:
            st["scene_ids"] = new_ids
        _emit(event="scene_result", id=sid, ok=ok, error=err or (None if ok else "ApplyPlanningScene success=false"),
              added=new_ids, removed=sorted(old_ids - set(new_ids)), listed_existing=listed)

    def do_cancel():
        gh = st.get("goal")
        if gh is None:
            _emit(event="cancel_result", ok=False, error="no active goal")
            return
        try:
            gh.cancel_goal_async()
            _emit(event="cancel_result", ok=True)
        except Exception as exc:
            _emit(event="cancel_result", ok=False, error=str(exc)[:200])

    def make_cb(key):
        def cb(msg):
            try:
                st[key] = json.loads(json.dumps(message_to_ordereddict(msg), default=str))
            except Exception as exc:
                st[key] = {"error": str(exc)}
        return cb

    def ensure_subs():
        topics = dict(node.get_topic_names_and_types())
        for topic, key in (("/feedback/joint_states", "js"), ("/feedback/arm_status", "arm_status"),
                           ("/feedback/gripper_status", "gripper_status"), ("/feedback/leader_joint_states", "ljs")):
            if key in st["subs"] or topic not in topics:
                continue
            try:
                mt = get_message(topics[topic][0])
                cb = on_js if key == "js" else (on_ljs if key == "ljs" else make_cb(key))
                st["subs"][key] = node.create_subscription(mt, topic, cb, qos)
            except Exception as exc:
                st[key + "_error"] = str(exc)

    pending_calls = []          # V3.19 ASYNC SERVICE CALLS: the spin loop keeps running (joint feedback keeps flowing)

    def do_call(c):
        """Start a service call; the result is emitted later by poll_calls(). -> None, or an immediate error dict."""
        name = str(c.get("call") or "")
        services = dict(node.get_service_names_and_types())
        if name not in services:
            return {"ok": False, "error": f"service {name} not available"}
        stype = services[name][0]
        cli = None
        try:
            cli = node.create_client(get_service(stype), name)
            req = get_service(stype).Request()
            if "value" in c and c["value"] is not None and hasattr(req, "data"):
                req.data = bool(c["value"])
            if not cli.wait_for_service(timeout_sec=2.0):
                node.destroy_client(cli)
                return {"ok": False, "error": "service not ready", "type": stype}
            fut = cli.call_async(req)
            pending_calls.append({"c": c, "fut": fut, "cli": cli, "type": stype, "t0": time.time(),
                                  "timeout": float(c.get("timeout") or 10.0)})
            return None
        except Exception as exc:
            if cli is not None:
                try:
                    node.destroy_client(cli)
                except Exception:
                    pass
            return {"ok": False, "error": str(exc)[:300], "type": stype}

    def poll_calls():
        for pc in list(pending_calls):
            fut, c = pc["fut"], pc["c"]
            done = fut.done()
            if not done and time.time() - pc["t0"] < pc["timeout"]:
                continue
            pending_calls.remove(pc)
            if done:
                try:
                    res = fut.result()
                    out = json.loads(json.dumps(message_to_ordereddict(res), default=str)) if res is not None else None
                    ok = bool(out.get("success", True)) if isinstance(out, dict) else True
                    r = {"ok": ok, "type": pc["type"], "response": out}
                except Exception as exc:
                    r = {"ok": False, "type": pc["type"], "error": str(exc)[:300]}
            else:
                r = {"ok": False, "type": pc["type"], "error": f"timeout {pc['timeout']:.0f} s (no response)"}
            r["elapsed_s"] = round(time.time() - pc["t0"], 2)
            try:
                node.destroy_client(pc["cli"])
            except Exception:
                pass
            _emit(event="call_result", id=c.get("id"), call=c.get("call"), value=c.get("value"), **r)

    def do_hold(h):
        """V3.22 FAKE-CONTROLLER SYNC: while the driver's control gate is closed, move the ROS-side joint controller
        (ros2_control arm_controller, fake hardware) to the real arm's pose, so re-opening the gate does not send the
        arm back to the stale controller position (= the zero pose) at full speed. Published for ~1 s."""
        topics = dict(node.get_topic_names_and_types())
        cands = [t for t, ty in topics.items() if t.endswith("/joint_trajectory") and
                 any("JointTrajectory" in x for x in ty)]
        want = str(h.get("controller") or "arm_controller")
        cands.sort(key=lambda t: (want not in t, t))
        topic = cands[0] if cands else f"/{want}/joint_trajectory"
        try:
            from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
            from builtin_interfaces.msg import Duration
            pub = st["subs"].get("hold_pub:" + topic) or node.create_publisher(JointTrajectory, topic, 10)
            st["subs"]["hold_pub:" + topic] = pub
            msg = JointTrajectory()
            msg.joint_names = [str(n) for n in h["joints"]]
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in h["positions"]]
            pt.time_from_start = Duration(sec=0, nanosec=int(float(h.get("t", 0.5)) * 1e9))
            msg.points = [pt]
            st["hold"] = {"pub": pub, "msg": msg, "until": time.time() + float(h.get("duration", 1.0)), "next": 0.0}
            _emit(event="hold_result", id=h.get("id"), ok=True, topic=topic, found=bool(cands))
        except Exception as exc:
            _emit(event="hold_result", id=h.get("id"), ok=False, topic=topic, error=str(exc)[:300])

    _emit(event="monitor_ready", domain=os.environ.get("ROS_DOMAIN_ID"))
    next_report = time.time() + period_s
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            while not cmds.empty():
                c = cmds.get()
                if c.get("call"):
                    r = do_call(c)
                    if r is not None:
                        _emit(event="call_result", id=c.get("id"), call=c.get("call"), value=c.get("value"), **r)
                elif c.get("move"):
                    try:
                        do_move(c["move"])
                    except Exception as exc:
                        if st.get("move"):
                            st["move"]["phase"] = "done"
                        _emit(event="move_result", id=c["move"].get("id"), ok=False, error=f"move error: {exc}"[:300])
                elif c.get("cancel"):
                    do_cancel()
                elif c.get("scene") is not None:
                    try:
                        do_scene(c["scene"])
                    except Exception as exc:
                        _emit(event="scene_result", id=c["scene"].get("id"), ok=False, error=f"scene error: {exc}"[:300])
                elif c.get("tf_watch") is not None:
                    st["tf_watch"] = [str(x) for x in c["tf_watch"]][:12]
                elif c.get("get_model"):
                    maybe_emit_model(force=True)
                elif c.get("hold"):
                    do_hold(c["hold"])
                elif c.get("quit"):
                    raise KeyboardInterrupt
            poll_calls()
            now = time.time()
            hd = st.get("hold")
            if hd and now >= hd["next"]:
                if now > hd["until"]:
                    st["hold"] = None
                else:
                    hd["next"] = now + 0.1
                    try:
                        hd["pub"].publish(hd["msg"])
                    except Exception:
                        st["hold"] = None
            if st.get("ljs_new") and now - st.get("ljs_emit_t", 0.0) >= 0.1:
                st["ljs_new"], st["ljs_emit_t"] = False, now
                _emit(event="ljs", joints=st["ljs"])
            if st.get("js_new") and now - st.get("js_emit_t", 0.0) >= 0.1:
                st["js_new"], st["js_emit_t"] = False, now
                _emit(event="js", joints=st["joints"])
            if now < next_report:
                continue
            next_report = now + period_s
            ensure_subs()
            try:
                ensure_desc()
                try_params()
                maybe_emit_model()
                ensure_tf()
            except Exception as exc:
                st["v314_err"] = str(exc)[:200]
            st["js_stamps"] = [t for t in st["js_stamps"] if t >= now - 2.0]
            hz = len(st["js_stamps"]) / 2.0
            nodes = ["/".join(p for p in (ns.strip("/"), n) if p) for n, ns in node.get_node_names_and_namespaces()]
            services = dict(node.get_service_names_and_types())
            has_action = "/move_action/_action/send_goal" in services
            if not has_action:
                try:
                    from rclpy.action import get_action_names_and_types
                    has_action = any(n == "/move_action" for n, _t in get_action_names_and_types(node))
                except Exception:
                    pass
            _emit(event="status", js_hz=round(hz, 1), joints=st["joints"], arm_status=st["arm_status"],
                  gripper_status=st["gripper_status"], nodes=sorted(set("/" + n for n in nodes)),
                  has_move_group=("/move_group" in {"/" + n for n in nodes}),
                  has_move_action=bool(has_action),
                  services={s: services[s][0] for s in services if s.rsplit("/", 1)[-1] in MONITORED_SERVICES},
                  tcp=(tcp_pose() if JC is not None and st.get("model") else None),
                  tf_links=(tf_links() if JC is not None and st.get("model") else None),
                  model_known=bool(st.get("model")), has_urdf=bool(st["urdf"]), has_srdf=bool(st["srdf"]),
                  move=(st["move"] or {}).get("phase"))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
    _emit(event="monitor_stopped")
    return 0


# =========================================================================== pipeline controller (no Tk)
def compact_status(d, limit=14):
    """A message dict -> short 'key=value' list for the status panel (header/stamps dropped)."""
    if not isinstance(d, dict):
        return ""
    flat = {}

    def walk(x, pre):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("header", "stamp"):
                    continue
                walk(v, f"{pre}{k}.")
        elif isinstance(x, list):
            if len(x) <= 8 and all(not isinstance(v, (dict, list)) for v in x):
                flat[pre[:-1]] = x
        else:
            flat[pre[:-1]] = x
    walk(d, "")
    items = list(flat.items())[:limit]
    return "  ".join(f"{k}={v}" for k, v in items)


class RealPipeline:
    """Six-step bring-up.  Blocking step methods are meant to run in a worker thread; a pump thread drains the
    helper processes.  The GUI reads .state / snapshots and .events (log lines)."""

    def __init__(self, cfg_fn, sim_active_fn=lambda: False, confirm_fn=lambda title, text: True,
                 log_path=None, can_opener=open_can_socket, env_checker=check_environment,
                 port_status_fn=can_port_status, setup_runner=run):
        self.cfg_fn, self.sim_active_fn, self.confirm_fn = cfg_fn, sim_active_fn, confirm_fn
        self.env_checker, self.port_status_fn, self.setup_runner = env_checker, port_status_fn, setup_runner
        self.events = queue.Queue()
        self.lock = threading.Lock()
        self.state = {k: (GREY, "") for k in STEP_KEYS}
        self.busy = None
        self.push = ProcRunner("push", log_path)
        self.driver = ProcRunner("driver", log_path)
        self.monitor = ProcRunner("ros_monitor", log_path)
        self.leader = ProcRunner("leader", None)        # V3.21: pyAgxArm leader angles while hand guiding
        self.can = CanMonitor(cfg_fn().get("can_port", "can0"), opener=can_opener)
        self.push_info = {"enabled": False, "last": None, "error": None, "joints": None}
        self.mon_status = None
        self.call_results = {}
        self.driver_fail = None
        self.rviz_died = None                    # V3.46: RViz exit text (driver / MoveIt keep running)
        self.push_mode = None
        self.stale = []
        self._call_id = 0
        # V3.14 joint control
        self.model = None
        self.last_js, self.last_js_t = None, None
        self.move_state = {}
        self._move_id = 0
        self._move_ev = threading.Event()
        # V3.15 consistency + safety
        import collections
        self.real_urdf_xml, self.real_urdf = None, None
        self.safety_fn = None                  # set by the safety tab: -> {"fence", "path_check", "tool_T"} or None
        self.motion_log = None
        self.teach_mode = False
        self.last_ljs, self.last_ljs_t = None, None
        self.last_pleader, self.last_pleader_t = None, None
        self.last_pjoints, self.last_pjoints_t = None, None
        self.js_hist = collections.deque(maxlen=1200)
        self.scene_results = {}
        self._scene_id = 0
        self._scene_ev = threading.Event()
        self._pump_stop = threading.Event()
        threading.Thread(target=self._pump_loop, daemon=True, name="v313_pump").start()

    # ---------------------------------------------------------------- plumbing
    def log(self, line):
        self.events.put(line)

    def set(self, key, lamp, msg):
        with self.lock:
            self.state[key] = (lamp, msg)
        tag = {GREEN: "[OK]", RED: "[FAIL]", YELLOW: "[WAIT]", AMBER: "[WARN]", GREY: "[INFO]"}[lamp]
        self.log(f"{tag} [V3.13 {STEP_LABELS[STEP_KEYS.index(key)]}] {msg}")

    def lamp(self, key):
        with self.lock:
            return self.state[key][0]

    def active(self):
        return self.push.running() or self.driver.running() or self.monitor.running() or self.leader.running()

    def own_pids(self):
        return {r.proc.pid for r in (self.push, self.driver, self.monitor, self.leader) if r.proc is not None}

    def prev_ok(self, key):
        i = STEP_KEYS.index(key)
        return all(self.lamp(k) in (GREEN, AMBER) for k in STEP_KEYS[:i])

    def _driver_alive(self):
        """V3.13.1: a launch that just died still counts as alive until its last output lines are pumped, so the
        real failure line (e.g. 'Failed to get firmware version') wins over 'launch process not running'."""
        d = self.driver
        return d.running() or (d.proc is not None and (d.exit_code is None or not d.lines.empty()))

    def _pump_loop(self):
        while not self._pump_stop.is_set():
            self.pump()
            time.sleep(0.05)

    def pump(self):
        for runner, handler in ((self.push, self._on_push), (self.driver, self._on_driver), (self.monitor, self._on_monitor),
                                (self.leader, self._on_leader)):
            while True:
                try:
                    line = runner.lines.get_nowait()
                except queue.Empty:
                    break
                handler(line)

    def _json(self, line):
        if line and line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                return None
        return None

    def _on_push(self, line):
        if line is None:
            self.push_info["enabled"] = False
            self.log(f"[INFO] [V3.13 push] helper exited rc={self.push.exit_code}")
            return
        ev = self._json(line)
        if not ev:
            self.log(f"[push] {line}")
            return
        e = ev.get("event")
        self.push_info["last"] = e
        if e == "enabled":
            self.push_info["enabled"] = True
            self.log(f"[OK] [V3.13 push] NERO ENABLED after {ev.get('attempts')} attempt(s)")
        elif e == "state":
            self.push_info["joints"] = ev.get("joints")
            self.push_info["status"] = ev.get("status")
        elif e in ("error", "timeout"):
            self.push_info["error"] = ev.get("error") or e
            self.log(f"[FAIL] [V3.13 push] {e}: {ev.get('error', '')}")
        elif e not in ("state",):
            self.log(f"[push] {line}")

    def _on_leader(self, line):
        if line is None:
            if self.teach_mode:
                self.log(f"[WARN] [V3.21 teach] pyAgxArm 主导臂读数进程退出 rc={self.leader.exit_code}")
            return
        ev = self._json(line)
        if not ev:
            return
        e = ev.get("event")
        if e == "leader":
            now = time.time()
            if ev.get("leader"):
                self.last_pleader, self.last_pleader_t = ev["leader"], now
            if ev.get("joints"):
                self.last_pjoints, self.last_pjoints_t = ev["joints"], now
        elif e == "leader_ready":
            self.log(f"[INFO] [V3.21 teach] pyAgxArm 主导臂读数已启动（get_leader_joint_angles "
                     f"{'可用' if ev.get('has_leader_api') else '不存在 → 只用 get_joint_angles'}）")
        elif e == "leader_error":
            self.log(f"[FAIL] [V3.21 teach] pyAgxArm 主导臂读数: {ev.get('error')}")

    def _on_driver(self, line):
        if line is None:
            self.log(f"[INFO] [V3.13 driver] ros2 launch exited rc={self.driver.exit_code}")
            return
        f = driver_log_failure(line)
        if f and not self.driver.stop_requested and not self.driver_fail:
            self.driver_fail = f                      # keep the first (root-cause) failure line
        if is_rviz_death(line) and not self.driver.stop_requested:
            self.rviz_died = "exit " + (line.split("exit code", 1)[1].split(",")[0].strip() if "exit code" in line else "?")
            self.log(f"[driver] {line}")
            self.log("[WARN] [V3.46 driver] RViz 退出了（只影响 3D 显示；驱动 / MoveIt 不受影响）。常见原因：pip 版 "
                     "opencv-python 的 Qt 插件，见 README V3.46")
            return
        self.log(f"[driver] {line}")

    def _on_monitor(self, line):
        if line is None:
            self.mon_status = None
            return
        ev = self._json(line)
        if not ev:
            self.log(f"[ros_monitor] {line}")
            return
        e = ev.get("event")
        if e == "status":
            self.mon_status = ev
            js = ev.get("joints")
            if js and js.get("names") and (self.last_js_t is None or time.time() - self.last_js_t > 0.5) \
                    and float(ev.get("js_hz") or 0) > 0:
                self.last_js, self.last_js_t = js, time.time()
        elif e == "js":
            if ev.get("joints"):
                self.last_js, self.last_js_t = ev["joints"], time.time()
                self.js_hist.append((self.last_js_t, ev["joints"]))
        elif e == "ljs":
            if ev.get("joints"):
                self.last_ljs, self.last_ljs_t = ev["joints"], time.time()
        elif e == "urdf":
            self.real_urdf_xml = ev.get("xml")
            try:
                import NERO_V314_Joint_Control as JC
                self.real_urdf = JC.parse_urdf(self.real_urdf_xml)
            except Exception as exc:
                self.real_urdf = None
                self.log(f"[WARN] [V3.15] driver URDF not parsable: {exc}")
        elif e == "scene_result":
            self.scene_results[ev.get("id")] = ev
            self._scene_ev.set()
        elif e == "model":
            self.model = ev.get("model")
        elif e in ("move_accepted", "move_feedback"):
            if ev.get("id") == self.move_state.get("id"):
                self.move_state["phase"] = "执行中" if e == "move_accepted" else str(ev.get("phase"))
        elif e == "move_result":
            if ev.get("id") == self.move_state.get("id"):
                self.move_state.update(result=ev, ok=bool(ev.get("ok")), phase="完成" if ev.get("ok") else "失败")
                self._move_ev.set()
        elif e == "cancel_result":
            self.log(f"{'[OK]' if ev.get('ok') else '[WARN]'} [V3.14 joints] 停止运动: {ev.get('error') or 'cancel sent'}")
        elif e == "hold_result":
            self.call_results[("hold", ev.get("id"))] = ev
        elif e == "call_result":
            self.call_results[ev.get("id")] = ev
            self.log(f"{'[OK]' if ev.get('ok') else '[FAIL]'} [V3.13 service] {ev.get('call')} "
                     f"value={ev.get('value')} -> {ev.get('response') or ev.get('error')}")
        else:
            self.log(f"[ros_monitor] {line}")

    def wait_until(self, fn, timeout_s, period=0.2):
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            r = fn()
            if r:
                return r
            time.sleep(period)
        return fn()

    # ---------------------------------------------------------------- steps
    def step_env(self, kill_stale=False):
        if kill_stale and self.stale:
            self.log(f"[INFO] [V3.13 1 环境] stopping leftovers: {[p for p, _c in self.stale]}")
            for pid, _c in self.stale:
                try:
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass
            time.sleep(2.0)
        lamp, items, stale = self.env_checker(sim_runner_active=self.sim_active_fn(), me=self.own_pids())
        self.stale = stale
        for name, ok, detail in items:
            self.log(f"{'[OK]' if ok else '[FAIL]'} [V3.13 1 环境] {name}: {detail}")
        bad = [n for n, ok, _d in items if not ok]
        self.set("env", lamp, "all checks passed" if lamp == GREEN else f"failed: {bad}")
        return lamp

    def step_can(self, execute=True):
        port = self.cfg_fn().get("can_port", "can0")
        st, err = self.port_status_fn(port)
        lamp, msg = evaluate_can_port(st)
        if lamp != GREEN and execute:
            self.log(f"[INFO] [V3.13 2 CAN 口] {msg} -> configuring {port} (system password dialog)")
            rc, out, err2 = self.setup_runner(can_setup_command(port), timeout=120)
            if rc != 0:
                self.set("can", RED, f"pkexec/ip failed rc={rc}: {(err2 or out).strip()[:160]}")
                return RED
            time.sleep(0.5)
            st, err = self.port_status_fn(port)
            lamp, msg = evaluate_can_port(st)
        self.set("can", lamp, msg if st else (err or msg))
        if lamp == GREEN:
            self.can.port = port
            self.can.start()
        return lamp

    def step_push(self):
        if self.push.running() and self.push_info.get("enabled"):
            self.set("push", GREEN, "push helper already running, arm enabled")
            return GREEN
        if not self.confirm_fn("电机即将使能", "第 3 步会通过 pyAgxArm 连接 NERO 并使能电机（手臂会锁住当前姿态）。\n"
                                             "确认手臂周围无人、无障碍物？"):
            self.set("push", GREY, "cancelled by user")
            return GREY
        cfg = self.cfg_fn()
        self.push_info = {"enabled": False, "last": None, "error": None, "joints": None}
        self.push.start(helper_argv("push", "--channel", cfg.get("can_port", "can0"),
                                    "--fw", cfg.get("push_fw", "v111") or "v111", "--timeout", "15"))
        self.set("push", YELLOW, "connecting + set_normal_mode() until enable() ...")
        ok = self.wait_until(lambda: self.push_info.get("enabled") or self.push_info.get("error")
                             or not self.push.running(), 25.0)
        if self.push_info.get("enabled"):
            self.wait_until(lambda: self.push_info.get("joints") is not None, 3.0)
            self.set("push", GREEN, f"enabled; joints={self.push_info.get('joints')}")
            return GREEN
        self.set("push", RED, f"not enabled: {self.push_info.get('error') or 'helper exited'}")
        if self.push.running():
            self.push.stop()
        return RED

    def step_flow(self, settle_s=2.0):
        if not self.can.running():
            self.can.start()
        time.sleep(settle_s)
        snap = self.can.snapshot()
        lamp, msg = evaluate_flow(snap)
        self.set("flow", lamp, msg)
        return lamp

    def _measure_fps(self, seconds):
        time.sleep(seconds)
        return self.can.fps()

    def step_driver(self, release_test_s=3.0, timeout_s=45.0):
        cfg = dict(self.cfg_fn())
        snap = self.can.snapshot()
        if snap.get("gripper_old_id") and cfg.get("effector_type") != "none":
            cfg["effector_type"] = "none"
            self.log("[WARN] [V3.13 5 ROS2 驱动] gripper still on old CAN ID -> effector_type forced to none")
        self.driver_fail = None
        self.rviz_died = None
        keep_push = False
        if self.push.running():
            fps_with = self.can.fps()
            self.log(f"[INFO] [V3.13 5 ROS2 驱动] push-release test: stop push helper, measure CAN {release_test_s:.0f} s")
            self.push.stop(signal.SIGINT, 5.0)
            self.wait_until(lambda: not self.push.running(), 6.0)
            fps_after = self._measure_fps(release_test_s)
            self.push_mode = push_release_decision(fps_with, fps_after)
            self.log(f"[INFO] [V3.13 5 ROS2 驱动] CAN {fps_with:.0f} fps with push -> {fps_after:.0f} fps after release "
                     f"=> {self.push_mode}")
            if self.push_mode == "keep_until_driver":
                keep_push = True
                self.push_info = {"enabled": False, "last": None, "error": None, "joints": None}
                self.push.start(helper_argv("push", "--channel", cfg.get("can_port", "can0"),
                                            "--fw", cfg.get("push_fw", "v111") or "v111", "--timeout", "15"))
                if not self.wait_until(lambda: self.push_info.get("enabled") or not self.push.running(), 20.0) \
                        or not self.push_info.get("enabled"):
                    self.set("driver", RED, "push helper could not re-enable before the driver start")
                    return RED
        env = dict(os.environ)
        env["ROS_DOMAIN_ID"] = str(int(cfg.get("domain_id", DEFAULT_DOMAIN_ID)))
        self.log(f"[INFO] [V3.13 5 ROS2 驱动] ROS_DOMAIN_ID={env['ROS_DOMAIN_ID']} | {launch_command(cfg)}")
        self.driver.start(["/bin/bash", "-c", launch_shell(cfg)], env=env)
        if not self.monitor.running():
            self.monitor.start(["/bin/bash", "-c", monitor_shell(cfg.get("domain_id", DEFAULT_DOMAIN_ID))],
                               env=env, stdin_pipe=True)
        self.set("driver", YELLOW, "ros2 launch started; waiting for /feedback/joint_states")

        def good():
            lamp, _m = evaluate_driver(self.mon_status, self._driver_alive(), self.driver_fail)
            return lamp in (GREEN, RED) and lamp
        res = self.wait_until(good, timeout_s)
        lamp, msg = evaluate_driver(self.mon_status, self._driver_alive(), self.driver_fail)
        if lamp == GREEN and keep_push:
            self.log("[INFO] [V3.13 5 ROS2 驱动] driver has feedback -> stopping push helper (single bus master)")
            self.push.stop(signal.SIGINT, 5.0)
            self.wait_until(lambda: not self.push.running(), 6.0)
            time.sleep(2.0)
            lamp, msg = evaluate_driver(self.mon_status, self._driver_alive(), self.driver_fail)
            if lamp != GREEN:
                msg = "feedback lost after the push helper stopped: " + msg
                lamp = RED
        if lamp == YELLOW:
            lamp, msg = RED, f"timeout {timeout_s:.0f}s: {msg}"
        self.set("driver", lamp, msg)
        return lamp

    def step_moveit(self, timeout_s=90.0):
        use_rviz = bool(self.cfg_fn().get("use_rviz", True))

        def good():
            lamp, _m = evaluate_moveit(self.mon_status, rviz_alive(), use_rviz, getattr(self, "rviz_died", None))
            return (lamp in (GREEN, AMBER) or not self.driver.running()) and True
        self.wait_until(good, timeout_s)
        lamp, msg = evaluate_moveit(self.mon_status, rviz_alive(), use_rviz, getattr(self, "rviz_died", None))
        if not self.driver.running():
            lamp, msg = RED, "driver/MoveIt launch exited"
        elif lamp == YELLOW:
            lamp, msg = RED, f"timeout {timeout_s:.0f}s: {msg}"
        self.set("moveit", lamp, msg)
        return lamp

    STEP_FUNCS = ("step_env", "step_can", "step_push", "step_flow", "step_driver", "step_moveit")

    def run_step(self, key):
        if self.sim_active_fn():
            self.set(key, RED, "simulation is running: real-robot steps are locked")
            return RED
        if not self.prev_ok(key):
            self.log(f"[WARN] [V3.13] {key}: previous step is not green")
            return None
        return getattr(self, self.STEP_FUNCS[STEP_KEYS.index(key)])()

    def run_all(self):
        for key in STEP_KEYS:
            if self.lamp(key) == GREEN and key in ("env", "can", "flow"):
                # cheap checks are always re-run; push/driver/moveit are kept when already green
                pass
            elif self.lamp(key) == GREEN:
                continue
            r = self.run_step(key)
            if r not in (GREEN, AMBER):
                self.log(f"[FAIL] [V3.13 一键启动] stopped at {STEP_LABELS[STEP_KEYS.index(key)]}")
                return False
        self.log("[READY] [V3.13] NERO real robot: driver + MoveIt + RViz ready")
        return True

    # ---------------------------------------------------------------- services / stop
    def service_name(self, suffix):
        svcs = (self.mon_status or {}).get("services") or {}
        for s in svcs:
            if s.rsplit("/", 1)[-1] == suffix:
                return s
        return None

    def call_service(self, suffix, value=None, timeout_s=10.0):
        name = self.service_name(suffix)
        if not name or not self.monitor.running():
            self.log(f"[FAIL] [V3.13 service] {suffix}: driver/monitor not running or service missing")
            return None
        self._call_id += 1
        self.monitor.send({"call": name, "value": value, "id": self._call_id, "timeout": float(timeout_s)})
        self.log(f"[INFO] [V3.13 service] -> {name} value={value} ({(self.mon_status or {}).get('services', {}).get(name)})")
        return self._call_id

    # ---------------------------------------------------------------- V3.19 teach mode
    def wait_call(self, cid, timeout_s):
        t_end = time.time() + timeout_s
        while time.time() < t_end and cid not in self.call_results:
            time.sleep(0.1)
        return self.call_results.get(cid)

    def _gate(self, value):
        """driver control gate (control_enable): closed while teaching so the driver stops commanding the arm."""
        if not self.service_name("control_enable"):
            return None
        cid = self.call_service("control_enable", value, timeout_s=5.0)
        r = self.wait_call(cid, 6.0) if cid else None
        self.log(f"{'[OK]' if r and r.get('ok') else '[WARN]'} [V3.20 teach] 驱动控制门 {'开' if value else '关'}: "
                 f"{(r or {}).get('response') or (r or {}).get('error') or 'no result'}")
        return r

    EXIT_SPEED = 0.10            # V3.22: leaving teach mode -> back to the zero pose at 10 % (MoveIt)

    def leader_active(self):
        """V3.22: the arm streams hand-guided (leader) angles while its normal joint feedback is silent — NERO is in
        hand-guiding mode whatever ctrl_mode says (V111 keeps reporting 0x01)."""
        now = time.time()
        ljs = self.last_ljs_t is not None and now - self.last_ljs_t <= self.FRESH_S
        pl = self.last_pleader_t is not None and now - self.last_pleader_t <= self.FRESH_S and self.leader.running()
        js_silent = self.last_js_t is None or now - self.last_js_t > self.FRESH_S
        return (ljs or pl) and js_silent

    def _hand_pose_rad(self):
        """current hand-guided pose (rad, joint1..7) from the leader sources, else joint_states."""
        _n, js, t = self.feedback_source()
        pos = (js or {}).get("position") or []
        if t is None or time.time() - t > 2.0 or len(pos) < 7:
            return None
        names = list((js or {}).get("names") or [])
        want = [f"joint{i}" for i in range(1, 8)]
        if all(n in names for n in want):
            return [float(pos[names.index(n)]) for n in want]
        return [float(v) for v in pos[:7]]

    def _run_mode_helper(self, on, timeout_s, hold=None):
        cfg = self.cfg_fn()
        argv = helper_argv("mode", "--channel", cfg.get("can_port", "can0"), "--fw", cfg.get("push_fw", "v111") or "v111",
                           "--mode", "teach" if on else "normal", "--speed", str(int(round(self.EXIT_SPEED * 100))))
        if hold:
            argv += ["--hold", json.dumps([round(v, 5) for v in hold])]
        diags, ev, p = [], None, None
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
            for l in p.stdout.splitlines():
                if l.startswith("{"):
                    d = json.loads(l)
                    if d.get("event") == "diag":
                        diags.append(d)
                    elif d.get("event") == "mode_result":
                        ev = d
        except Exception as exc:
            ev = {"ok": False, "error": f"mode helper: {exc}"}
        ev = ev or {"ok": False, "error": f"mode helper produced no result (rc={getattr(p, 'returncode', '?')}) "
                                          f"{(getattr(p, 'stderr', '') or '')[-300:]}"}
        for d in diags:
            if d.get("apis") is not None:
                self.log(f"[INFO] [V3.20 teach] pyAgxArm 模式接口: {', '.join(d.get('apis') or [])} | "
                         f"切换前 ctrl_mode={d.get('ctrl_mode_before')}")
            if "hold" in d:
                self.log(f"[INFO] [V3.22 teach] 重新使能前：速度接口={d.get('speed_api') or '无'}（{d.get('speed_pct')}%）"
                         f" · 保持当前位姿接口={d.get('movej_api') or '无'}")
        return ev

    def _sync_fake_controller(self, q):
        """V3.22: ROS-side arm_controller -> real pose while the gate is closed (see monitor do_hold)."""
        if not q or not self.monitor.running():
            return None
        self._call_id += 1
        cid = self._call_id
        self.monitor.send({"hold": {"id": cid, "joints": [f"joint{i}" for i in range(1, 8)], "positions": list(q),
                                    "t": 0.5, "duration": 1.2}})
        r = self.wait_call(("hold", cid), 3.0)
        time.sleep(1.3)
        self.log(f"{'[OK]' if r and r.get('ok') else '[WARN]'} [V3.22 teach] ROS 控制器同步到当前位姿："
                 f"{(r or {}).get('topic') or ''} {(r or {}).get('error') or ('' if r else 'no result')}")
        return r

    def set_teach_mode(self, on, timeout_s=30.0, home_model=None, home_fn=None):
        """V3.20: on = close the driver's control gate, pyAgxArm set_leader_mode (hand guiding), verify;
        off = exit_teach_mode service (if offered), set_normal_mode / set_follower_mode + enable, reopen the gate.
        V3.22: entry is also confirmed by leader angles that start streaming (ctrl_mode may stay 0x01 on V111);
        exit holds the current pose (driver gate closed, ROS controller synced, firmware target = current pose,
        speed 10 %) and then — with home_model — returns to the zero pose via MoveIt at 10 %.
        Blocking (worker thread). -> result dict"""
        if on:
            self._gate(False)
            t0 = time.time()
            ev = self._run_mode_helper(True, timeout_s)
            if not ev.get("ok"):
                # V3.22: the ROS leader topic / silent joint_states are evidence too
                ok2 = self.wait_until(lambda: self.last_ljs_t is not None and self.last_ljs_t > t0 and self.leader_active(),
                                      2.5, 0.1)
                if ok2:
                    ev = dict(ev, ok=True, evidence="/feedback/leader_joint_states 在更新且 /feedback/joint_states 停发",
                              error=None)
            if ev.get("ok"):
                self.teach_mode = True
                cfg = self.cfg_fn()
                if not self.leader.running():
                    self.leader.start(helper_argv("leader", "--channel", cfg.get("can_port", "can0"),
                                                  "--fw", cfg.get("push_fw", "v111") or "v111"))
            elif not self.leader_active():
                self._gate(True)                   # entering really failed: give control back to the driver
            cm = ev.get("ctrl_mode")
            self.log(f"{'[OK]' if ev.get('ok') else '[FAIL]'} [V3.22 teach] 进入示教模式（手动拖动）: api={ev.get('api')} "
                     f"ctrl_mode={cm if cm is None else hex(cm)} 依据={ev.get('evidence') or '-'} {ev.get('error') or ''}"[:500])
            return ev
        # ---------------- exit
        q = self._hand_pose_rad()
        if q and home_model:
            lim = home_model.get("limits") or []
            for j, (lo, hi) in enumerate(lim[:7]):
                c = min(max(q[j], lo + 0.01), hi - 0.01)
                if abs(c - q[j]) > 1e-6 and abs(c - q[j]) <= 0.2:
                    self.log(f"[INFO] [V3.22 teach] J{j + 1} {math.degrees(q[j]):+.1f}° 超出关节限位，"
                             f"保持点取 {math.degrees(c):+.1f}°（10% 速度微调）")
                    q[j] = c
        self.log(f"[INFO] [V3.22 teach] 退出示教：保持当前位姿 "
                 f"{'[' + ', '.join(f'{math.degrees(v):+.1f}' for v in q) + ']°' if q else '（没有当前角度）'}")
        self._gate(False)                          # driver must not replay its stale (zero) command meanwhile
        self._sync_fake_controller(q)
        if self.service_name("exit_teach_mode"):
            cid = self.call_service("exit_teach_mode", timeout_s=10.0)
            r = self.wait_call(cid, 12.0) if cid else None
            self.log(f"{'[OK]' if r and r.get('ok') else '[WARN]'} [V3.19 teach] exit_teach_mode 服务: "
                     f"{(r or {}).get('response') or (r or {}).get('error') or 'no result'}")
        if self.leader.running():
            self.leader.stop(signal.SIGINT, 3.0)
        ev = self._run_mode_helper(False, timeout_s, hold=q)
        if ev.get("ok"):
            self.teach_mode = False
            self.wait_until(lambda: self.last_js_t is not None and time.time() - self.last_js_t < 0.5, 5.0, 0.1)
            self._sync_fake_controller(self._hand_pose_rad() or q)
            self._gate(True)
        cm = ev.get("ctrl_mode")
        self.log(f"{'[OK]' if ev.get('ok') else '[FAIL]'} [V3.22 teach] 退出示教模式: "
                 f"api={ev.get('api')} ctrl_mode={cm if cm is None else hex(cm)} {ev.get('error') or ''}"[:500])
        if ev.get("ok") and home_model:
            time.sleep(1.0)
            a = self.actual_arm_deg(home_model)
            if a is not None and max(abs(x) for x in a) < 0.2:
                self.log("[INFO] [V3.22 teach] 已在零位")
            else:
                self.log(f"[INFO] [V3.22 teach] 以 {self.EXIT_SPEED * 100:.0f}% 速度经 MoveIt 回零位")
                res = home_fn(home_model, self.EXIT_SPEED) if home_fn else \
                    self.move_and_settle(home_model, [0.0] * 7, self.EXIT_SPEED, label="退出示教回零")
                ok = bool(res and res.get("ok"))
                self.log(f"{'[OK]' if ok else '[FAIL]'} [V3.22 teach] 回零位: "
                         f"{'完成' if ok else (res or {}).get('error') or '失败'}"
                         f"{'' if ok else '（机械臂保持在当前位置；可用关节控制或「回 Home」再试）'}")
                ev = dict(ev, home=res)
        return ev

    # ---------------------------------------------------------------- V3.14 joint control
    def move_busy(self):
        return bool(self.move_state) and self.move_state.get("result") is None and not self.move_state.get("abandoned")

    FRESH_S = 1.0

    def _fresh(self, t):
        return t is not None and time.time() - t <= self.FRESH_S

    def feedback_source(self):
        """V3.21: which joint-angle source is used. While teaching: ROS leader topic > pyAgxArm leader angles >
        pyAgxArm joint angles > /feedback/joint_states. -> (name, js dict, timestamp)"""
        if self.teach_mode:
            if self.last_ljs and self._fresh(self.last_ljs_t):
                return "ROS /feedback/leader_joint_states", self.last_ljs, self.last_ljs_t
            if self.last_pleader and self._fresh(self.last_pleader_t):
                return "pyAgxArm get_leader_joint_angles", {"names": None, "position": self.last_pleader}, self.last_pleader_t
            if self.last_pjoints and self._fresh(self.last_pjoints_t) and not (self.last_js_t and self._fresh(self.last_js_t)):
                return "pyAgxArm get_joint_angles", {"names": None, "position": self.last_pjoints}, self.last_pjoints_t
        elif self.last_ljs and self.leader_active():
            # V3.22: hand guiding detected without the console's teach flag (e.g. ctrl_mode never changed)
            return "ROS /feedback/leader_joint_states（检测到手动拖动）", self.last_ljs, self.last_ljs_t
        return "ROS /feedback/joint_states", self.last_js, self.last_js_t

    def feedback_js(self):
        """(joint_states dict, timestamp) — see feedback_source()."""
        _n, js, t = self.feedback_source()
        return js, t

    def feedback_diag(self):
        now = time.time()
        age = lambda t: "无" if t is None else f"{now - t:.1f} s 前"
        return (f"/feedback/joint_states {age(self.last_js_t)} · /feedback/leader_joint_states {age(self.last_ljs_t)} · "
                f"pyAgxArm 主导臂 {age(self.last_pleader_t)} · pyAgxArm 关节 {age(self.last_pjoints_t)}"
                f"{'' if self.leader.running() else ' · pyAgxArm 读数进程未运行'}")

    def actual_arm_deg(self, model, max_age_s=2.0):
        _src, js, t = self.feedback_source()
        js = js or {}
        names, pos = list(js.get("names") or []), js.get("position") or []
        if len(pos) >= 7 and not all(n in names for n in model["joints"]) and (self.teach_mode or self.leader_active()) and \
                (t is not None and time.time() - t <= max_age_s):
            return [math.degrees(v) for v in pos[:7]]                  # leader sources: joint order 1..7
        if not t or time.time() - t > max_age_s:
            return None
        try:
            return [math.degrees(pos[names.index(n)]) for n in model["joints"]]
        except (ValueError, IndexError, KeyError):
            return None

    def _send_move(self, label, group, joints, positions, speed, timeout_s, extra=None):
        if not self.monitor.running():
            return {"ok": False, "error": "ROS monitor not running (step 5/6)"}
        if self.move_busy():
            return {"ok": False, "error": "another motion is in progress"}
        self._move_id += 1
        mid = self._move_id
        self._move_ev.clear()
        self.move_state = {"id": mid, "label": label, "phase": "规划中", "result": None, "t0": time.time()}
        sp = max(0.01, min(1.0, float(speed)))
        mv = {"id": mid, "group": group, "joints": list(joints), "positions": [float(p) for p in positions],
              "vel": sp, "acc": sp}
        mv.update(extra or {})
        self.monitor.send({"move": mv})
        if not self._move_ev.wait(timeout_s):
            self.monitor.send({"cancel": True})
            self.move_state["abandoned"] = True
            return {"ok": False, "error": f"no result within {timeout_s:.0f} s (cancel sent)"}
        return dict(self.move_state.get("result") or {"ok": False, "error": "no result"})

    def safety(self):
        try:
            return self.safety_fn() if self.safety_fn else None
        except Exception as exc:
            self.log(f"[WARN] [V3.15 safety] safety settings unavailable: {exc}")
            return None

    def fence_goal_check(self, model, target_rad):
        """V3.15: target pose vs the software fence (FK with the driver URDF, else the sim URDF).
        -> (ok, text, violations)"""
        sf = self.safety()
        if not sf or not sf.get("fence"):
            return True, "", []
        import NERO_V315_Consistency_Safety as CS
        urdf = self.real_urdf or sf.get("sim_urdf")
        if urdf is None:
            return False, "FENCE: no URDF for the fence check -> refused", []
        ok, v, _pts = CS.fence_check(urdf, model["joints"], target_rad, model.get("base_link") or "base_link",
                                     sf["fence"], sf.get("tool_T"))
        return ok, ("" if ok else "FENCE: 目标姿态越界 -> 未发送: " + CS.violation_text(v, sf["fence"], sf.get("corners"))), v

    def move_joints(self, model, target_rad, speed, label="move", timeout_s=None, plan_only=False):
        import NERO_V314_Joint_Control as JC
        if self.sim_active_fn():
            return {"ok": False, "error": "simulation is running"}
        if self.lamp("moveit") != GREEN:
            return {"ok": False, "error": "step 6 MoveIt / RViz is not green"}
        if getattr(self, "teach_mode", False):
            return {"ok": False, "error": "NERO 在示教模式（零力拖动）：先「退出示教模式」再运动"}
        lims = model["limits"]
        for i, (q, (lo, hi)) in enumerate(zip(target_rad, lims)):
            if not (lo - 1e-6 <= q <= hi + 1e-6):
                return {"ok": False, "error": f"J{i + 1} target {math.degrees(q):.1f} deg outside limit"}
        ok, txt, v = self.fence_goal_check(model, target_rad)
        if not ok:
            return {"ok": False, "error": txt, "fence_violation": {"waypoint": "goal", "violations": v}}
        extra = {"plan_only": bool(plan_only)}
        sf = self.safety()
        if sf and sf.get("max_speed"):                 # V3.16: teach / inspect modes cap the speed
            speed = min(float(speed), float(sf["max_speed"]))
        if sf and sf.get("fence") and sf.get("path_check"):
            extra.update(fence=sf["fence"], tool_T=sf.get("tool_T"), base=model.get("base_link") or "base_link",
                         arm_joints=list(model["joints"]))
        return self._send_move(label, model["arm_group"], model["joints"], target_rad, speed,
                               timeout_s or JC.MOVE_TIMEOUT_S, extra)

    def move_and_settle(self, model, target_rad, speed, label="move", plan_only=False, settle_s=0.8, timeout_s=None):
        """move_joints + settle + final error + one row in logs/real_motion_log.csv."""
        res = self.move_joints(model, target_rad, speed, label=label, timeout_s=timeout_s, plan_only=plan_only)
        res = dict(res or {"ok": False, "error": "no result"})
        tgt = [math.degrees(v) for v in target_rad]
        act = None
        if res.get("ok") and not plan_only:
            time.sleep(settle_s)
            act = self.actual_arm_deg(model)
            if act is not None:
                e = [abs(a - b) for a, b in zip(tgt, act)]
                res["final_error_deg"] = max(e)
                res["final_error_joint"] = e.index(max(e))
        res["actual_deg"] = act
        try:
            if self.motion_log is None:
                import NERO_V315_Consistency_Safety as CS
                self.motion_log = CS.MotionLog()
            self.motion_log.add(label, "plan" if plan_only else "move", speed, res, tgt, act)
        except Exception as exc:
            self.log(f"[WARN] [V3.15] motion log: {exc}")
        return res

    def apply_scene(self, objects, frame="base_link", prefix="v315_", timeout_s=15.0):
        if not self.monitor.running():
            return {"ok": False, "error": "ROS monitor not running (step 5/6)"}
        self._scene_id += 1
        sid = self._scene_id
        self._scene_ev.clear()
        self.monitor.send({"scene": {"id": sid, "objects": objects, "frame": frame, "prefix": prefix}})
        t_end = time.time() + timeout_s
        while time.time() < t_end and sid not in self.scene_results:
            self._scene_ev.wait(0.2)
        return self.scene_results.get(sid) or {"ok": False, "error": "no scene result"}

    def set_tf_watch(self, links):
        if self.monitor.running():
            self.monitor.send({"tf_watch": list(links)})

    def js_samples(self, model, t_from):
        out = []
        for t, js in list(self.js_hist):
            if t < t_from:
                continue
            try:
                out.append((t, [math.degrees(js["position"][js["names"].index(n)]) for n in model["joints"]]))
            except (ValueError, KeyError, IndexError):
                pass
        return out

    def move_gripper(self, model, mm, speed, timeout_s=30.0):
        import NERO_V314_Joint_Control as JC
        g = (model or {}).get("gripper")
        if not g:
            return {"ok": False, "error": "no gripper planning group"}
        q = JC.gripper_mm_to_joint(mm, g)
        return self._send_move(f"夹爪 {mm:.0f} mm", g["group"], g["joints"], [q], speed, timeout_s)

    def cancel_move(self):
        if not self.monitor.running():
            self.log("[FAIL] [V3.14 joints] 停止运动：ROS monitor 未运行 -> 用急停")
            return False
        self.monitor.send({"cancel": True})
        self.log("[INFO] [V3.14 joints] 停止运动：已向 move_group 发送 cancel")
        return True

    def stop_all(self):
        self.log("[INFO] [V3.13 全部停止] stopping RViz/MoveIt/driver, ROS monitor, push helper (CAN port stays up)")
        self.monitor.send({"quit": True})
        for r, g in ((self.driver, 20.0), (self.monitor, 5.0), (self.push, 5.0), (self.leader, 3.0)):
            if r.running():
                r.stop(signal.SIGINT, g)
        self.wait_until(lambda: not self.active(), 25.0)
        for k in ("push", "driver", "moveit"):
            self.set(k, GREY, "stopped")
        self.mon_status = None
        self.model, self.last_js, self.last_js_t = None, None, None
        self.real_urdf_xml, self.real_urdf = None, None
        if self.move_busy():
            self.move_state["abandoned"] = True
            self._move_ev.set()
        return not self.active()

    def shutdown(self):
        self._pump_stop.set()
        self.can.stop()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="NERO V3.13 real-robot helper")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("--channel", default="can0")
    p.add_argument("--fw", default="v111")
    p.add_argument("--timeout", type=float, default=15.0)
    p.add_argument("--period", type=float, default=0.5)
    p.add_argument("--max-reports", type=int, default=0)
    md = sub.add_parser("mode")
    md.add_argument("--channel", default="can0")
    md.add_argument("--fw", default="v111")
    md.add_argument("--mode", choices=["teach", "normal"], required=True)
    md.add_argument("--hold", default="", help="V3.22: JSON list of 7 joint angles (rad) to hold when re-enabling")
    md.add_argument("--speed", type=int, default=10)
    ld = sub.add_parser("leader")
    ld.add_argument("--channel", default="can0")
    ld.add_argument("--fw", default="v111")
    ld.add_argument("--period", type=float, default=0.05)
    ld.add_argument("--max-reports", type=int, default=0)
    m = sub.add_parser("monitor")
    m.add_argument("--period", type=float, default=1.0)
    a = ap.parse_args(argv)
    if a.cmd == "push":
        return cmd_push(a.channel, a.fw, a.timeout, a.period, a.max_reports)
    if a.cmd == "mode":
        return cmd_mode(a.channel, a.fw, a.mode, hold=json.loads(a.hold) if a.hold else None, speed_pct=a.speed)
    if a.cmd == "leader":
        return cmd_leader(a.channel, a.fw, a.period, a.max_reports)
    return cmd_monitor(a.period)


if __name__ == "__main__":
    sys.exit(main())
