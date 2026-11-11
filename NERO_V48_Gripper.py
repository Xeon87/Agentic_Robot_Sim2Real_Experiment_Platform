#!/usr/bin/env python3
"""NERO V4.8 GRIPPER: the AgileX AgxGripper (NERO original two-finger parallel gripper) next to the SO-101, one
global choice for all pages.

Control path (decided with the user, 2026-11-10): the gripper is driven ONLY through pyAgxArm (official SDK) in a
helper process (`python3 NERO_V48_Gripper.py serve ...`, JSON lines on stdin / stdout); the ROS2 driver is always
launched with effector_type:=none, so exactly one program sends gripper frames (CAN 0x159).  pyAgxArm's connect()
only opens the bus and reads; the helper calls gripper functions only (+ the J7 controller-limit test on request).

pyAgxArm AgxGripper API used (pyAgxArm/protocols/can_protocol/drivers/effector/agx_gripper/default/driver.py):
  move_gripper_m(value m, force N)  - width 0 .. 0.07 / 0.1 m, force 0.5 .. 3.0 (the SDK's "N"; NOT fingertip newtons:
                                      the spec sheet says 40 N rated / 50 N max -> 「力控标定」 maps it with a gauge)
  get_gripper_status()              - value (m), force, foc_status: voltage_too_low, motor_overheating,
                                      driver_overcurrent, driver_overheating, sensor_status, driver_error_status,
                                      driver_enable_status, homing_status
  disable_gripper(), reset_gripper(), calibrate_gripper(timeout)  (zero = the current position, ack 0x75)
  get_gripper_teaching_pendant_param() -> max_range_config 0.07 / 0.1 m (stroke version)
NERO arm (J7 test): get_joint_angle_vel_limits(joint_index), set_joint_angle_vel_limits(joint_index, min, max).

Tests / no hardware: NERO_V48_FAKE_GRIPPER=1 (or backend="fake") -> FakeAgxBackend, same request protocol.
"""
from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

VERSION = "V4.8"
MARKER = "V4.8 NERO AGX GRIPPER + GRIPPER SWITCH"
HERE = os.path.dirname(os.path.realpath(__file__))
CFG_PATH = os.path.join(HERE, "config", "gripper.json")

TYPES = ("so101", "agx_gripper")
LABELS = {"so101": "SO-101（舵机 · 串口）", "agx_gripper": "NERO 原装 AgxGripper（CAN）"}
SHORT = {"so101": "SO-101", "agx_gripper": "NERO 原装"}

# AgxGripper facts: spec sheet (user) + official URDF agx_arm_urdf/nero/urdf/nero_with_gripper_description.xacro
AGX = {
    "stroke_mm": 100.0,            # 0-100 mm (0-70 mm version: read from max_range_config)
    "force_min": 0.5, "force_max": 3.0, "force_default": 1.0,   # SDK / ROS driver range ("N" of the SDK)
    "rated_force_n": 40.0, "max_force_n": 50.0,                 # spec sheet, at the fingertips
    "mass_kg": 0.50,               # URDF: base 0.45 + 2 x 0.025 (= spec 500 g)
    "flange_l7_x_mm": 31.0,        # flange face on link7 x (tool axis at link7 z = -23.5 mm)
    "base_offset_mm": 6.0,         # gripper_base above the flange
    "finger_origin_mm": 138.0,     # finger slides above gripper_base = the fingertips
    "tcp_l7_x_mm": 155.3,          # the URDF tool frame the planners use (pad centre, 20 mm inside the tips)
    "held_min_gap_mm": 1.5,        # stopped with at least this gap = something between the fingers
    "speed_mm_s": 80.0,            # for timeouts only (the SDK has no speed parameter)
    "can_ids": (0x51A, 0x52A),     # ID 10 (NERO); 0x517 / 0x518 = old ID 7: not usable
}
SPEED_DEFAULT_PCT, SPEED_MIN_PCT = 40.0, 10.0
SPEED_CHOICES = (30, 40, 50, 70, 100)
RAMP_HZ = 25.0
BRACKET = {"mass_kg": 0.06, "aruco_plate_mm": (40.0, 40.0)}     # the user's printed wrist-camera bracket (V7.1)
FAULTS = (("voltage_too_low", "电压过低"), ("motor_overheating", "电机过热"), ("driver_overcurrent", "驱动过流"),
          ("driver_overheating", "驱动过热"), ("sensor_status", "传感器异常"), ("driver_error_status", "驱动错误"))
PRESET_KEYS = (("egg", "鸡蛋"), ("cup", "纸杯 / 塑料杯"), ("block", "方块"))
DEFAULTS = {
    "type": "so101",
    "agx": {"channel": "can0", "fw": "v111", "backend": "pyagxarm", "default_force": 1.0,
            "tip_from_flange_mm": 144.0,        # V4.8.1: measured flange face -> fingertips (official URDF: 6 + 138)
            "speed_pct": 40.0,                  # V4.8.2: finger speed, % of full_speed_mm_s (100 = no limit)
            "full_speed_mm_s": 150.0},          # V4.8.2: what "100 %" means (the gripper's own top speed, estimate)
    # per-object open width (mm) + force level; starting values, tune them on the real objects (力控标定 ③)
    "presets": {"egg": {"open_mm": 60.0, "force": 0.5}, "cup": {"open_mm": 65.0, "force": 1.0},
                "block": {"open_mm": 60.0, "force": 1.5}},
    "force_table": [],             # [[setting, measured_N], ...] from 力控标定 ② (gauge); empty = linear guess
    "baseline": None,              # 力控标定 ①: {"max_force", "threshold", "saved"}
    "zero": None,                  # 校正归零: {"saved", "stroke_mm", "closed_mm", "open_mm"}
    "j7_test": None,               # J7 controller-limit test result
}


# ============================================================================ global choice
_listeners: List[Callable[[str], None]] = []
_blockers: List[Callable[[], Optional[str]]] = []


def _merge(d, base):
    out = json.loads(json.dumps(base))
    for k, v in (d or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def load_cfg(path=None) -> Dict[str, Any]:
    try:
        with open(path or CFG_PATH, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = {}
    c = _merge(d, DEFAULTS)
    env = os.environ.get("NERO_GRIPPER")
    if env in TYPES:
        c["type"] = env
    if c.get("type") not in TYPES:
        c["type"] = "so101"
    return c


def save_cfg(c, path=None) -> None:
    p = path or CFG_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(c, f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)


def update_cfg(**kw) -> Dict[str, Any]:
    c = load_cfg()
    c.update(kw)
    save_cfg(c)
    return c


def active_type() -> str:
    return load_cfg()["type"]


def is_agx() -> bool:
    return active_type() == "agx_gripper"


def add_listener(fn: Callable[[str], None]) -> None:
    if fn not in _listeners:
        _listeners.append(fn)


def add_blocker(fn: Callable[[], Optional[str]]) -> None:
    """fn() -> reason the gripper may not be switched now (recording, task running...) or None"""
    if fn not in _blockers:
        _blockers.append(fn)


def switch_blocked() -> Optional[str]:
    for b in list(_blockers):
        try:
            why = b()
        except Exception:
            why = None
        if why:
            return why
    return None


def set_active(t: str) -> Dict[str, Any]:
    """-> {"ok", "changed", "error", "todo": [steps after the switch]}"""
    if t not in TYPES:
        return {"ok": False, "error": f"未知夹爪类型 {t}"}
    if t == active_type():
        return {"ok": True, "changed": False, "todo": []}
    why = switch_blocked()
    if why:
        return {"ok": False, "error": f"现在不能切换夹爪：{why}"}
    c = load_cfg()
    c["type"] = t
    save_cfg(c)
    for fn in list(_listeners):
        try:
            fn(t)
        except Exception:
            pass
    return {"ok": True, "changed": True, "todo": todo_after_switch(t)}


def todo_after_switch(t: str) -> List[str]:
    if t == "agx_gripper":
        return ["实机NERO调试：重新执行第 5/6 步（驱动固定 effector_type:=none，夹爪由 pyAgxArm 控制）",
                "实机NERO调试 → 夹爪：连接，第一次用先做「校正归零」",
                "腕部相机支架换了：腕部手眼标定重做（内参不变）",
                "主相机 ArUco 换到支架侧板（40 mm 板）：主相机手眼标定重做",
                "示教模式负载：约 0.56 kg（夹爪 0.50 + 支架 / 相机 0.06）"]
    return ["实机NERO调试：重新执行第 5/6 步", "Agentic实机任务 → SO-101 夹爪：连接",
            "腕部 / 主相机手眼标定换回 SO-101 的那一套（各自单独保存）"]


def set_speed_pct(pct) -> float:
    """V4.8.2: save the AgxGripper finger speed (%) -> the value saved"""
    v = min(100.0, max(SPEED_MIN_PCT, float(pct)))
    c = load_cfg()
    c["agx"]["speed_pct"] = v
    save_cfg(c)
    return v


def speed_text(pct=None) -> str:
    v = float(load_cfg()["agx"].get("speed_pct", SPEED_DEFAULT_PCT)) if pct is None else float(pct)
    return f"{v:.0f}%"


def parse_speed(text) -> Optional[float]:
    try:
        return float(str(text).strip().rstrip("%").strip())
    except ValueError:
        return None


def force_level(force_n: float, table=None) -> float:
    """a task's requested fingertip force (N) -> the SDK force setting (0.5 .. 3.0).  With a gauge table
    (力控标定 ②) by interpolation, else linear: rated 40 N ~ 3.0 (a guess until calibrated)."""
    tab = table if table is not None else (load_cfg().get("force_table") or [])
    lo, hi = AGX["force_min"], AGX["force_max"]
    pts = sorted((float(s), float(n)) for s, n in tab if n is not None)
    if len(pts) >= 2:
        for (s0, n0), (s1, n1) in zip(pts, pts[1:]):
            if n0 <= force_n <= n1 and n1 > n0:
                return round(min(hi, max(lo, s0 + (force_n - n0) * (s1 - s0) / (n1 - n0))), 2)
        return lo if force_n < pts[0][1] else hi
    return round(min(hi, max(lo, float(force_n) * hi / AGX["rated_force_n"])), 2)


def pct_to_force(pct: float) -> float:
    """0 .. 100 % (the SO-101 torque-limit knob) -> 0.5 .. 3.0"""
    p = min(100.0, max(0.0, float(pct)))
    return round(AGX["force_min"] + (AGX["force_max"] - AGX["force_min"]) * p / 100.0, 2)


# ============================================================================ backends (request protocol)
class FakeAgxBackend:
    """simulated AgxGripper + NERO J7 limits (tests, no hardware).  The fingers move at speed_mm_s toward the
    target and stop on an object of width `object_mm` (the force reading then = the setting)."""

    def __init__(self, stroke_mm=100.0):
        self.lock = threading.Lock()
        self.stroke = float(stroke_mm)
        self.width, self.target, self.force_set = 50.0, 50.0, 1.0
        self.t = time.time()
        self.enabled, self.homed = False, True
        self.zero_offset = 0.0               # what the encoder thinks "0" is, relative to the true closed position
        self.object_mm = None
        self.faults = {k: False for k, _ in FAULTS}
        self.alive = True
        self.teach_mode, self.teach_accepts = False, True
        self.j_limits = {7: (-math.pi / 2, math.pi / 2)}
        self.j7_now = 0.0
        self.log: List[dict] = []
        self.speed = AGX["speed_mm_s"] * 4      # tests run fast

    def _step(self):
        now = time.time()
        dt, self.t = now - self.t, now
        if not self.enabled:
            return
        stop_at = self.object_mm if (self.object_mm is not None and self.target < self.object_mm <= self.width + 1e-9) else None
        goal = max(self.target, stop_at) if stop_at is not None else self.target
        d = goal - self.width
        step = self.speed * dt
        self.width = goal if abs(d) <= step else self.width + math.copysign(step, d)

    def request(self, cmd: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            self._step()
            c = cmd.get("cmd")
            self.log.append(dict(cmd))
            if not self.alive:
                return {"ok": False, "error": "no gripper frames"}
            if c == "status":
                blocked = self.object_mm is not None and abs(self.width - self.object_mm) < 0.05 and self.target < self.object_mm
                return {"ok": True, "width_m": round((self.width - self.zero_offset) / 1000.0, 6),
                        "force": self.force_set if blocked else 0.05, "is_ok": True, "fps": 200.0,
                        "foc": dict(self.faults, driver_enable_status=self.enabled, homing_status=self.homed)}
            if c == "move":
                if self.teach_mode and not self.teach_accepts:
                    return {"ok": True}                     # sent, but the firmware ignores it
                m = float(cmd.get("width_m", 0.0)) * 1000.0 + self.zero_offset
                self.target = min(self.stroke, max(0.0, m))
                self.force_set = float(cmd.get("force", 1.0))
                self.enabled = True
                return {"ok": True}
            if c == "disable":
                self.enabled = False
                return {"ok": True, "disabled": True}
            if c == "reset":
                self.faults = {k: False for k in self.faults}
                return {"ok": True}
            if c == "calibrate":
                if self.enabled:
                    return {"ok": True, "acked": False}
                self.zero_offset = self.width
                self.homed = True
                return {"ok": True, "acked": True}
            if c == "teach_param":
                return {"ok": True, "max_range_m": self.stroke / 1000.0}
            if c == "get_jlim":
                lo, hi = self.j_limits.get(int(cmd.get("joint", 7)), (None, None))
                return {"ok": True, "min_rad": lo, "max_rad": hi}
            if c == "set_jlim":
                self.j_limits[int(cmd.get("joint", 7))] = (float(cmd["min_rad"]), float(cmd["max_rad"]))
                return {"ok": True, "acked": True}
            if c == "joint7":
                return {"ok": True, "rad": self.j7_now}
            if c == "ping":
                return {"ok": True, "pong": True}
            return {"ok": False, "error": f"unknown cmd {c}"}

    def close(self):
        pass


class HelperBackend:
    """the pyAgxArm helper process (this file, `serve`), JSON lines"""

    def __init__(self, channel="can0", fw="v111", python=None, log=None):
        self.log = log or (lambda s: None)
        self.q: "queue.Queue" = queue.Queue()
        self.lock = threading.Lock()
        self.n = 0
        self.proc = subprocess.Popen([python or sys.executable, os.path.realpath(__file__), "serve", "--channel", channel,
                                      "--fw", fw], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1)
        threading.Thread(target=self._reader, daemon=True, name="v48_gripper_reader").start()
        hello = self._get(None, 15.0)
        if not hello or not hello.get("ok"):
            err = (hello or {}).get("error") or self._stderr_tail() or "pyAgxArm 助手没有响应"
            self.close()
            raise RuntimeError(err)
        self.hello = hello

    def _stderr_tail(self):
        try:
            if self.proc.poll() is not None:
                return (self.proc.stderr.read() or "")[-300:]
        except Exception:
            pass
        return ""

    def _reader(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line.startswith("{"):
                try:
                    self.q.put(json.loads(line))
                except ValueError:
                    pass
        self.q.put({"_eof": True})

    def _get(self, rid, timeout):
        t_end = time.time() + timeout
        while time.time() < t_end:
            try:
                d = self.q.get(timeout=max(0.01, t_end - time.time()))
            except queue.Empty:
                break
            if d.get("_eof"):
                return {"ok": False, "error": "pyAgxArm 助手退出了：" + (self._stderr_tail() or "?")}
            if rid is None or d.get("id") == rid:
                return d
        return None

    def request(self, cmd):
        with self.lock:
            if self.proc.poll() is not None:
                return {"ok": False, "error": "pyAgxArm 助手没有在运行"}
            self.n += 1
            c = dict(cmd, id=self.n)
            try:
                self.proc.stdin.write(json.dumps(c) + "\n")
                self.proc.stdin.flush()
            except Exception as exc:
                return {"ok": False, "error": f"写不进助手：{exc}"}
            r = self._get(self.n, float(cmd.get("timeout", 2.0)) + 1.5)
            return r if r is not None else {"ok": False, "error": "助手超时"}

    def close(self):
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                self.proc.stdin.flush()
                self.proc.wait(3.0)
        except Exception:
            pass
        try:
            if self.proc.poll() is None:
                self.proc.kill()
        except Exception:
            pass


def make_backend(kind=None, channel="can0", fw="v111", log=None):
    kind = kind or ("fake" if os.environ.get("NERO_V48_FAKE_GRIPPER") == "1" else "pyagxarm")
    if kind == "fake":
        return FakeAgxBackend()
    return HelperBackend(channel, fw, log=log)


# ============================================================================ the device (SO101RealGripper-like)
class AgxGripper:
    """same interface as NERO_V365_STS3215_Gripper.SO101RealGripper (connect / status / open_gap / clamp / relax /
    held / last / cfg), widths in mm (the real opening between the pads), force = the SDK setting 0.5 .. 3.0"""

    kind = "agx_gripper"

    def __init__(self, cfg=None, log=None, backend=None):
        self.log = log or (lambda s: None)
        c = load_cfg()
        self.cfg = dict(c["agx"], **(cfg or {}))
        self.cfg.setdefault("stall_torque_nm", 0.0)      # read by old SO-101 code paths; unused here
        self.backend = backend
        self.last: Optional[Dict[str, Any]] = None
        self.held = False
        self.last_clamp = None
        self.stroke_mm = AGX["stroke_mm"]
        self._stop_ramp = False

    # -- connection
    @property
    def connected(self) -> bool:
        return self.backend is not None

    def connect(self, port=None, baud=None, servo_id=None, channel=None, backend=None):
        """port / baud / servo_id accepted for the SO-101 call signature and ignored; channel = CAN port"""
        if self.backend is not None:
            self.disconnect()
        ch = channel or self.cfg.get("channel") or "can0"
        try:
            self.backend = backend or make_backend(self.cfg.get("backend") if os.environ.get("NERO_V48_FAKE_GRIPPER") != "1"
                                                   else "fake", ch, self.cfg.get("fw", "v111"), self.log)
        except Exception as exc:
            self.backend = None
            return False, f"pyAgxArm 连接失败：{str(exc)[:200]}"
        st = None
        for _ in range(10):
            st = self.status()
            if st is not None and st.get("is_ok"):
                break
            time.sleep(0.1)
        if st is None or not st.get("is_ok"):
            self.disconnect()
            return False, f"CAN {ch} 上收不到夹爪反馈（夹爪 ID 要是 10；ID 7 的旧夹爪不能用；供电 24 V？）"
        tp = self.backend.request({"cmd": "teach_param", "timeout": 1.5})
        if tp.get("ok") and tp.get("max_range_m") in (0.07, 0.1):
            self.stroke_mm = float(tp["max_range_m"]) * 1000.0
        return True, (f"NERO 原装夹爪 · CAN {ch} · 行程 0–{self.stroke_mm:.0f} mm · 当前 {st['gap_mm']:.1f} mm · "
                      f"{'已使能' if st['enabled'] else '未使能'} · {'已归零' if st['homed'] else '未归零'}")

    def disconnect(self):
        if self.backend is not None:
            try:
                self.backend.close()
            except Exception:
                pass
        self.backend = None

    # -- telemetry
    def status(self) -> Optional[Dict[str, Any]]:
        if self.backend is None:
            return None
        r = self.backend.request({"cmd": "status"})
        if not r.get("ok"):
            return None
        foc = r.get("foc") or {}
        w = float(r.get("width_m") or 0.0) * 1000.0
        s = {"gap_mm": round(w, 2), "width_mm": round(w, 2), "open_deg": None, "force": round(float(r.get("force") or 0.0), 3),
             "enabled": bool(foc.get("driver_enable_status")), "homed": bool(foc.get("homing_status")),
             "faults": [cn for k, cn in FAULTS if foc.get(k)], "is_ok": bool(r.get("is_ok")), "fps": r.get("fps"),
             "stroke_mm": self.stroke_mm, "load_pct": None, "current_ma": None}
        s["errors"] = list(s["faults"])
        s["_t"] = time.time()
        self.last = s
        return s

    # -- motion
    def speed_pct(self):
        """V4.8.2: the global finger speed (%), read fresh (the panel / Agentic tab change it)"""
        try:
            v = float(load_cfg()["agx"].get("speed_pct", SPEED_DEFAULT_PCT))
        except Exception:
            v = SPEED_DEFAULT_PCT
        return min(100.0, max(SPEED_MIN_PCT, v))

    def move(self, width_mm, force=None, speed_pct=None):
        """V4.8.2 SPEED LIMIT: the SDK has no finger speed (CAN 0x159 = width, force, mode only) -> below 100 % the
        target is walked there in RAMP_HZ steps of speed x 1/RAMP_HZ (the gripper is position-controlled and follows);
        blocks until the last step is sent (callers run in worker threads)"""
        if self.backend is None:
            return {"ok": False, "error": "夹爪没有连接"}
        f = float(self.cfg.get("default_force", AGX["force_default"]) if force is None else force)
        f = min(AGX["force_max"], max(AGX["force_min"], f))
        w = min(self.stroke_mm, max(0.0, float(width_mm)))
        pct = self.speed_pct() if speed_pct is None else min(100.0, max(SPEED_MIN_PCT, float(speed_pct)))
        st = self.last if (self.last and time.time() - float(self.last.get("_t", 0.0)) < 0.5) else self.status()
        w0 = None if st is None else st.get("gap_mm")
        self._stop_ramp = False
        if pct < 100.0 and w0 is not None and abs(w - w0) > 1.0:
            try:
                full = float(load_cfg()["agx"].get("full_speed_mm_s", 150.0))
            except Exception:
                full = 150.0
            step = max(0.2, full * pct / 100.0 / RAMP_HZ)
            n = int(math.ceil(abs(w - w0) / step))
            for k in range(1, n):
                if self._stop_ramp:
                    return {"ok": False, "error": "停止"}
                wk = w0 + (w - w0) * k / n
                r = self.backend.request({"cmd": "move", "width_m": wk / 1000.0, "force": f})
                if not r.get("ok"):
                    return r
                time.sleep(1.0 / RAMP_HZ)
        return self.backend.request({"cmd": "move", "width_m": w / 1000.0, "force": f})

    def _wait(self, goal_mm, timeout_s, tol_mm=0.8, stall_s=0.3):
        t_end = time.time() + timeout_s
        last_w, t_still, st = None, None, None
        while time.time() < t_end:
            st = self.status()
            if st is None:
                time.sleep(0.05)
                continue
            w = st["gap_mm"]
            if abs(w - goal_mm) <= tol_mm:
                return st, "reached"
            if last_w is not None and abs(w - last_w) < 0.1:
                t_still = t_still or time.time()
                if time.time() - t_still >= stall_s:
                    return st, "stalled"
            else:
                t_still = None
            last_w = w
            time.sleep(0.03)
        return st, "timeout"

    def open_gap(self, gap_mm, timeout_s=None, force=None):
        if self.backend is None:
            return {"ok": False, "error": "夹爪没有连接"}
        g = min(self.stroke_mm, max(0.0, float(gap_mm)))
        r = self.move(g, force)
        if not r.get("ok"):
            return {"ok": False, "error": r.get("error")}
        st, how = self._wait(g, timeout_s or (1.0 + self.stroke_mm / AGX["speed_mm_s"]))
        self.held = False
        if st is None:
            return {"ok": False, "error": "no telemetry"}
        return {"ok": how == "reached", "how": how, "gap_mm": st["gap_mm"], "open_deg": None, "force": st["force"],
                "errors": st["errors"], "load_pct": None}

    def open_deg(self, a, timeout_s=3.0):
        """SO-101 call; here a = mm"""
        return self.open_gap(a, timeout_s)

    def clamp(self, torque_pct=None, squeeze_deg=None, timeout_s=None, force=None):
        """close with a force setting (force, or the SO-101 torque % mapped onto 0.5 .. 3.0); held = stopped on
        something with a gap >= held_min_gap_mm"""
        if self.backend is None:
            return {"ok": False, "held": False, "error": "夹爪没有连接"}
        f = float(force) if force is not None else pct_to_force(torque_pct if torque_pct is not None else 20.0)
        r = self.move(0.0, f)
        if not r.get("ok"):
            return {"ok": False, "held": False, "error": r.get("error")}
        st, how = self._wait(0.0, timeout_s or (1.0 + self.stroke_mm / AGX["speed_mm_s"]), tol_mm=0.4)
        if st is None:
            return {"ok": False, "held": False, "error": "no telemetry"}
        time.sleep(0.1)
        st = self.status() or st
        held = bool(how != "reached" and st["gap_mm"] >= AGX["held_min_gap_mm"])
        self.held = held
        self.last_clamp = {"ok": True, "held": held, "how": how, "gap_mm": st["gap_mm"], "open_deg": None,
                           "force": st["force"], "force_setting": f, "torque_limit_pct": torque_pct,
                           "load_pct": None, "current_ma": None, "errors": st["errors"]}
        return dict(self.last_clamp)

    def relax(self):
        """disable the driver (fingers free) -> (ok, msg) like the SO-101 bus write"""
        self.held = False
        self._stop_ramp = True
        if self.backend is None:
            return False, "夹爪没有连接"
        r = self.backend.request({"cmd": "disable"})
        return bool(r.get("ok")), r.get("error") or "已失能"

    def calibrate_touch(self, *a, **k):
        return {"ok": False, "error": "原装夹爪不用空夹标定：在「实机NERO调试 → 夹爪」用「校正归零」"}

    # -- service
    def reset_faults(self):
        return self.backend.request({"cmd": "reset"}) if self.backend else {"ok": False, "error": "夹爪没有连接"}

    def calibrate_zero(self):
        """the CURRENT position becomes zero (do it disabled, fingers closed by hand)"""
        if self.backend is None:
            return {"ok": False, "error": "夹爪没有连接"}
        r = self.backend.request({"cmd": "calibrate", "timeout": 1.0})
        if not r.get("ok") or not r.get("acked"):
            return {"ok": False, "error": r.get("error") or "夹爪没有确认归零（先失能、手动合拢）"}
        self.last = None                          # V4.8.2: the old reading is in the old reference (speed ramp start)
        return {"ok": True}

    def stroke_check(self):
        """after zeroing: close -> full open -> close; readings vs the stroke"""
        a = self.open_gap(0.0)
        b = self.open_gap(self.stroke_mm)
        c = self.open_gap(0.0)
        if any(x.get("gap_mm") is None for x in (a, b, c)):
            return {"ok": False, "error": "没有读数"}
        ok = abs(a["gap_mm"]) <= 1.0 and abs(b["gap_mm"] - self.stroke_mm) <= 1.5 and abs(c["gap_mm"]) <= 1.0
        return {"ok": ok, "closed_mm": a["gap_mm"], "open_mm": b["gap_mm"], "stroke_mm": self.stroke_mm,
                "error": None if ok else f"读数 {a['gap_mm']:.1f} / {b['gap_mm']:.1f} mm，应为 0 / {self.stroke_mm:.0f} mm"}

    def force_baseline(self, cycles=2):
        """力控标定 ①: empty open / close at the default force, the force readings while moving -> contact threshold"""
        vals = []
        for _ in range(int(cycles)):
            for goal in (self.stroke_mm, 0.0):
                self.move(goal)
                t_end = time.time() + 1.0 + self.stroke_mm / AGX["speed_mm_s"]
                while time.time() < t_end:
                    st = self.status()
                    if st is None:
                        break
                    vals.append(st["force"])
                    if abs(st["gap_mm"] - goal) <= 0.8:
                        break
                    time.sleep(0.03)
        if not vals:
            return {"ok": False, "error": "没有读数"}
        mx = max(vals)
        return {"ok": True, "max_force": round(mx, 3), "mean_force": round(sum(vals) / len(vals), 3),
                "threshold": round(mx * 1.5 + 0.1, 3), "n": len(vals)}

    def teach_mode_test(self, delta_mm=10.0):
        """does the gripper follow a command now (call it while the arm is in teach mode)?"""
        st0 = self.status()
        if st0 is None:
            return {"ok": False, "error": "夹爪没有连接"}
        goal = st0["gap_mm"] + (delta_mm if st0["gap_mm"] + delta_mm <= self.stroke_mm else -delta_mm)
        r = self.open_gap(goal, timeout_s=2.0)
        moved = abs((r.get("gap_mm") or st0["gap_mm"]) - st0["gap_mm"])
        self.open_gap(st0["gap_mm"], timeout_s=2.0)
        return {"ok": True, "follows": moved >= 0.6 * abs(delta_mm), "moved_mm": round(moved, 1)}

    # -- J7 controller limit test
    def get_joint_limits(self, joint=7):
        r = self.backend.request({"cmd": "get_jlim", "joint": joint, "timeout": 1.5}) if self.backend else {"ok": False}
        if not r.get("ok") or r.get("min_rad") is None:
            return {"ok": False, "error": r.get("error") or "控制器没有返回限位"}
        return {"ok": True, "min_deg": math.degrees(r["min_rad"]), "max_deg": math.degrees(r["max_rad"])}

    def set_joint_limits(self, min_deg, max_deg, joint=7):
        if not (-90.0 <= min_deg < max_deg <= 90.0) and joint == 7:
            return {"ok": False, "error": "J7 限位只能设在 ±90° 以内"}
        r = self.backend.request({"cmd": "set_jlim", "joint": joint, "min_rad": math.radians(min_deg),
                                  "max_rad": math.radians(max_deg), "timeout": 1.5}) if self.backend else {"ok": False}
        return {"ok": bool(r.get("ok") and r.get("acked")), "error": r.get("error")}

    def joint7_deg(self):
        r = self.backend.request({"cmd": "joint7"}) if self.backend else {"ok": False}
        return math.degrees(r["rad"]) if r.get("ok") and r.get("rad") is not None else None


class GripperProxy:
    """what the pages hold as `grip`: forwards to the SO-101 or the AgxGripper, whichever is selected"""

    def __init__(self, so101, agx=None, log=None):
        self.devices = {"so101": so101, "agx_gripper": agx or AgxGripper(log=log)}

    @property
    def kind(self):
        return active_type()

    @property
    def dev(self):
        return self.devices[active_type()]

    def __getattr__(self, name):
        if name in ("devices",):
            raise AttributeError(name)
        return getattr(self.devices[active_type()], name)

    def __setattr__(self, name, value):
        if name == "devices":
            object.__setattr__(self, name, value)
        else:
            setattr(self.devices[active_type()], name, value)

    @property
    def connected(self):
        return bool(self.dev is not None and self.dev.connected)


# ============================================================================ helper process (real pyAgxArm)
def _emit(**kw):
    sys.stdout.write(json.dumps(kw, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


class PyAgxServer:
    """the pyAgxArm side of the request protocol (the helper process runs one; tests run it in-process on
    pyAgxArm's virtual CAN with AgileX's own gripper slave)"""

    def __init__(self, channel="can0", fw="v111", interface=None):
        from pyAgxArm import create_agx_arm_config, AgxArmFactory, ArmModel, NeroFW
        import NERO_V313_Real_Robot as RR
        _fw_attr, fw_const = RR._resolve_nero_fw(NeroFW, fw)
        cfg = create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=fw_const,
                                    interface=interface or os.environ.get("NERO_V48_CAN_INTERFACE", "socketcan"),
                                    channel=channel)
        self.robot = AgxArmFactory.create_arm(cfg)
        self.eff = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.robot.connect()                 # opens the bus and reads; sends nothing

    def handle(self, c: Dict[str, Any]) -> Dict[str, Any]:
        eff, robot, cmd = self.eff, self.robot, c.get("cmd")
        try:
            if cmd == "status":
                gs = eff.get_gripper_status()
                if gs is None:
                    return {"ok": True, "is_ok": bool(eff.is_ok()), "width_m": None, "force": None, "foc": {},
                            "fps": eff.get_fps()}
                m = gs.msg
                foc = {k: bool(getattr(m.foc_status, k, False)) for k, _ in FAULTS}
                foc.update(driver_enable_status=bool(getattr(m.foc_status, "driver_enable_status", False)),
                           homing_status=bool(getattr(m.foc_status, "homing_status", False)))
                return {"ok": True, "is_ok": bool(eff.is_ok()), "width_m": float(m.value), "force": float(m.force),
                        "foc": foc, "fps": eff.get_fps()}
            if cmd == "move":
                eff.move_gripper_m(value=float(c["width_m"]), force=float(c["force"]))
                return {"ok": True}
            if cmd == "disable":
                return {"ok": True, "disabled": bool(eff.disable_gripper())}
            if cmd == "reset":
                return {"ok": bool(eff.reset_gripper())}
            if cmd == "calibrate":
                return {"ok": True, "acked": bool(eff.calibrate_gripper(timeout=float(c.get("timeout", 1.0))))}
            if cmd == "teach_param":
                p = eff.get_gripper_teaching_pendant_param(timeout=float(c.get("timeout", 1.0)))
                return {"ok": p is not None, "max_range_m": float(p.msg.max_range_config) if p is not None else None}
            if cmd == "get_jlim":
                r = robot.get_joint_angle_vel_limits(joint_index=int(c.get("joint", 7)), timeout=float(c.get("timeout", 1.0)))
                return {"ok": r is not None, "min_rad": float(r.msg.min_angle_limit) if r is not None else None,
                        "max_rad": float(r.msg.max_angle_limit) if r is not None else None}
            if cmd == "set_jlim":
                ok = robot.set_joint_angle_vel_limits(joint_index=int(c.get("joint", 7)),
                                                      min_angle_limit=float(c["min_rad"]),
                                                      max_angle_limit=float(c["max_rad"]),
                                                      timeout=float(c.get("timeout", 1.0)))
                return {"ok": True, "acked": bool(ok)}
            if cmd == "joint7":
                ja = robot.get_joint_angles()
                vals = list(getattr(ja, "msg", ja) or []) if ja is not None else []
                v = float(vals[6]) if len(vals) >= 7 else None
                return {"ok": v is not None, "rad": v}
            if cmd == "ping":
                return {"ok": True, "pong": True}
            return {"ok": False, "error": f"unknown cmd {cmd}"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:250]}

    def request(self, c):
        return self.handle(c)

    def close(self):
        try:
            self.robot.disconnect()
        except Exception:
            pass


def serve(channel="can0", fw="v111"):
    try:
        srv = PyAgxServer(channel, fw)
    except Exception as exc:
        _emit(ok=False, error=f"pyAgxArm: {str(exc)[:250]}")
        return 1
    _emit(ok=True, hello=True, channel=channel)
    for line in sys.stdin:
        try:
            c = json.loads(line)
        except ValueError:
            continue
        if c.get("cmd") == "quit":
            break
        _emit(id=c.get("id"), **srv.handle(c))
    srv.close()
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=f"NERO {VERSION} AgxGripper helper (pyAgxArm)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--channel", default="can0")
    s.add_argument("--fw", default="v111")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        sys.path.insert(0, HERE)
        return serve(a.channel, a.fw)
    return 2


if __name__ == "__main__":
    sys.exit(main())
