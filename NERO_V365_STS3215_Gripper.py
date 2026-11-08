#!/usr/bin/env python3
"""NERO V3.65 REAL SO-101 GRIPPER - Feetech STS3215 serial bus servo driving the SO-101 moving jaw.

Protocol: Feetech SCS/STS half-duplex serial (0xFF 0xFF ID LEN INSTR PARAMS CHK, little endian, sign bit 15), taken
from the user's sts3215_gui_tool.py (registers, packet transfer, telemetry block read, WritePosEx).

Angle convention (user, 2026-11-02): the servo was re-zeroed so that 180.0 deg (= 2048 steps) is the CLOSED gripper
(tip distance 0); a positive angle offset opens it (e.g. +60 deg = wide open).  Here
    a = opening angle in degrees from closed,  steps = CLOSED_STEPS + OPEN_SIGN * a * 4096 / 360.
OPEN_SIGN can be flipped (NERO_V365_GRIPPER_OPEN_SIGN=-1) if the jaw moves the wrong way.

Force: the STS3215 has NO closed current / torque loop (modes: 0 position, 1 constant speed, 2 open-loop PWM,
3 step).  It does have a torque limit (register 48, 0..1000 = 0..100 % of the PWM output) and reports load (60) and
current (69).  The clamp therefore is POSITION CONTROL WITH A TORQUE LIMIT: the goal is SQUEEZE_DEG past "closed",
the jaw stops on the cup wall and presses with at most the limited torque (stall), which is held while the cup is
carried.  "Held" = the jaw stopped with a tip gap >= MIN_HELD_GAP_MM (something between the tips); stopping at ~0 mm
= empty.  The torque limit is derived from the skill's clamp force: F * lever / stall torque.

Tip gap vs angle uses the SO-101 jaw geometry of NERO_V364_SO101_Gripper (the moving pad rotates about the jaw
hinge); a = 0 is where the tips touch.  The simulation's virtual opening (backend units, 2000 * q1) maps to the
same jaw angle, so the cup skills' opening values (e.g. pregrasp = 32 mm tip gap) mean the same on the real gripper.
"""
import math
import os
import threading
import time

VERSION = "V3.67"
MARKER = "V3.65 REAL SO-101 GRIPPER"

# ------------------------------------------------------------------------------------------------ registers (STS3215)
REG = {"TORQUE_ENABLE": 40, "ACC": 41, "GOAL_POSITION": 42, "GOAL_TIME": 44, "GOAL_SPEED": 46, "TORQUE_LIMIT": 48,
       "LOCK": 55, "PRESENT_POSITION": 56, "PRESENT_SPEED": 58, "PRESENT_LOAD": 60, "PRESENT_VOLTAGE": 62,
       "PRESENT_TEMPERATURE": 63, "SERVO_STATUS": 65, "MOVING": 66, "PRESENT_CURRENT": 69, "MODE": 33,
       "MAX_TORQUE_LIMIT": 16}
INSTR = {"PING": 0x01, "READ": 0x02, "WRITE": 0x03}
ERROR_BITS = {0: "电压异常", 1: "磁编码器故障", 2: "过温", 3: "过流", 5: "过载保护"}
STEPS_PER_REV = 4096
SIM_PORT = "SIMULATOR"


def _envf(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


DEFAULTS = {
    "port": os.environ.get("NERO_V365_GRIPPER_PORT", "/dev/ttyACM0"),
    "baud": int(_envf("NERO_V365_GRIPPER_BAUD", 1000000)),
    "servo_id": int(_envf("NERO_V365_GRIPPER_ID", 1)),
    "closed_steps": int(_envf("NERO_V365_GRIPPER_CLOSED_STEPS", 2048)),      # 180.0 deg = closed (tips touching)
    "open_sign": 1 if _envf("NERO_V365_GRIPPER_OPEN_SIGN", 1) >= 0 else -1,
    "max_open_deg": _envf("NERO_V365_GRIPPER_MAX_OPEN_DEG", 70.0),
    "move_torque_pct": 50.0,          # torque limit while opening / free moves
    "speed_steps": 600,               # goal speed (steps/s; 600 = ~53 deg/s)
    "acc": 30,
    "squeeze_deg": 8.0,               # clamp goal: this far past "closed" (the wall stops the jaw before)
    "min_held_gap_mm": 0.8,           # stopped with a tip gap >= this = something is held
    # V3.67 EMPTY-CLAMP CALIBRATION: servo angle (deg from closed_steps, opening positive) where the real tips / pads
    # actually touch (an empty clamp stalls there; first real test: +1.85 deg = 2.4 mm "gap" read as held).  All jaw angles
    # a are counted from this touch point once it is calibrated.
    "touch_deg": 0.0,
    "touch_calibrated": False,
    "min_held_gap_cal_mm": 0.3,       # held threshold once the touch point is calibrated (party-cup wall ~0.5 mm)
    "stall_torque_nm": 1.91,          # STS3215 19 kg.cm @ 7.4 V
    "min_clamp_pct": 15.0, "max_clamp_pct": 60.0,
}


def encode_s15(v):
    v = int(round(v))
    return (abs(v) & 0x7FFF) | 0x8000 if v < 0 else v & 0x7FFF


def decode_s15(raw):
    return -(raw & 0x7FFF) if raw & 0x8000 else raw & 0x7FFF


def checksum(body):
    return (~sum(body)) & 0xFF


def build_packet(servo_id, instr, params=()):
    body = [servo_id & 0xFF, (len(params) + 2) & 0xFF, instr & 0xFF] + [p & 0xFF for p in params]
    return bytes([0xFF, 0xFF] + body + [checksum(body)])


# ------------------------------------------------------------------------------------------------ jaw geometry
def _geom():
    import NERO_V364_SO101_Gripper as G
    return G


_THETA0 = None


def theta_touch_deg():
    """simulation jaw angle (NERO_V364 convention) where the two tip pads touch (tip gap 0) = real a = 0"""
    global _THETA0
    if _THETA0 is None:
        G = _geom()
        lo, hi = -40.0, 40.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if G.tip_gap_mm(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        _THETA0 = 0.5 * (lo + hi)
    return _THETA0


def gap_mm_for_open_deg(a):
    return max(0.0, float(_geom().tip_gap_mm(theta_touch_deg() + float(a))))


def open_deg_for_gap_mm(gap):
    lo, hi = 0.0, 120.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if gap_mm_for_open_deg(mid) < gap:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def open_deg_for_virtual_mm(mm):
    """cup-skill opening (backend units, 2000 * q1) -> real opening angle a (deg from closed)"""
    return _geom().jaw_angle_deg(mm) - theta_touch_deg()


def virtual_mm_for_open_deg(a):
    G = _geom()
    th = theta_touch_deg() + float(a)
    th = min(max(th, math.degrees(G.JAW["theta_min"])), math.degrees(G.JAW["theta_max"]))
    return G.opening_mm_for_angle(th)


def lever_m():
    """jaw hinge -> moving pad centre (m), for the clamp force -> torque conversion"""
    G = _geom()
    h = G.HINGE_POS
    p = (-G.PAD_FIXED_CENTRE[0], 0.0, G.PAD_FIXED_CENTRE[2])
    return math.hypot(p[0] - h[0], p[2] - h[2])


def clamp_torque_pct(force_n, cfg=None):
    c = dict(DEFAULTS, **(cfg or {}))
    pct = 100.0 * float(force_n) * lever_m() / float(c["stall_torque_nm"])
    return round(min(max(pct, float(c["min_clamp_pct"])), float(c["max_clamp_pct"])), 1)


# ------------------------------------------------------------------------------------------------ bus
class SimServo:
    """simulated STS3215 behind the same register interface (tests / dry runs without hardware).  An optional
    obstacle stops the closing jaw at a tip gap (the cup wall); the load then rises to the torque limit."""

    def __init__(self, servo_id=1, closed_steps=2048, open_sign=1, start_open_deg=10.0):
        self.id = servo_id
        self.closed, self.sign = closed_steps, open_sign
        self.pos = float(closed_steps + open_sign * start_open_deg * STEPS_PER_REV / 360.0)
        self.goal = self.pos
        self.speed = 600.0
        self.torque_on = 0
        self.torque_limit = 1000
        self.obstacle_gap_mm = None
        self.t = time.time()
        self.load = 0
        self.mode = 0

    def _a(self, steps):
        return self.sign * (steps - self.closed) * 360.0 / STEPS_PER_REV

    def step(self):
        now = time.time()
        dt = min(0.2, max(0.0, now - self.t))
        self.t = now
        self.load = 30 if self.torque_on else 0
        if not self.torque_on:
            return
        diff = self.goal - self.pos
        mv = math.copysign(min(abs(diff), max(50.0, self.speed) * dt), diff)
        new = self.pos + mv
        if self.obstacle_gap_mm is not None:
            a_obs = open_deg_for_gap_mm(self.obstacle_gap_mm)
            if self._a(new) < a_obs <= self._a(self.pos) + 1e-6:          # closing onto the wall
                new = self.closed + self.sign * a_obs * STEPS_PER_REV / 360.0
            if abs(self._a(new) - a_obs) < 1e-3 and self._a(self.goal) < a_obs:
                self.load = int(self.torque_limit)                        # pressing on the wall (stall)
        self.pos = new

    def regs(self):
        self.step()
        m = bytearray(128)

        def u16(a, v):
            m[a], m[a + 1] = v & 0xFF, (v >> 8) & 0xFF
        m[REG["TORQUE_ENABLE"]] = self.torque_on
        u16(REG["GOAL_POSITION"], encode_s15(self.goal))
        u16(REG["TORQUE_LIMIT"], int(self.torque_limit))
        u16(REG["PRESENT_POSITION"], encode_s15(self.pos))
        moving = abs(self.goal - self.pos) > 1.0 and self.load < self.torque_limit
        u16(REG["PRESENT_SPEED"], encode_s15(self.speed if moving else 0))
        u16(REG["PRESENT_LOAD"], int(min(1000, self.load)))
        m[REG["PRESENT_VOLTAGE"]], m[REG["PRESENT_TEMPERATURE"]] = 74, 31
        m[REG["MOVING"]] = 1 if moving else 0
        u16(REG["PRESENT_CURRENT"], int(self.load * 0.4))
        m[REG["MODE"]] = self.mode
        return m

    def write(self, addr, data):
        d = list(data)
        for i, b in enumerate(d):
            a = addr + i
            if a == REG["TORQUE_ENABLE"]:
                self.torque_on = 1 if b == 1 else 0
            elif a == REG["MODE"]:
                self.mode = b
        if addr <= REG["GOAL_POSITION"] < addr + len(d):
            k = REG["GOAL_POSITION"] - addr
            self.goal = float(decode_s15(d[k] | (d[k + 1] << 8)))
            self.torque_on = 1
        if addr <= REG["GOAL_SPEED"] < addr + len(d):
            k = REG["GOAL_SPEED"] - addr
            self.speed = float(decode_s15(d[k] | (d[k + 1] << 8))) or 600.0
        if addr <= REG["TORQUE_LIMIT"] < addr + len(d):
            k = REG["TORQUE_LIMIT"] - addr
            self.torque_limit = d[k] | (d[k + 1] << 8)


def fix_port_permission(port, log=None):
    """V3.65/V4.3 SERIAL PERMISSION AUTO-REPAIR: when opening /dev/ttyACM* or /dev/ttyUSB* is refused (Errno 13
    Permission denied):
    1. Try non-interactive `sudo -n chmod 666 <port>` and `chmod 666 <port>` first (fast, non-blocking);
    2. If a password is required and a Tkinter GUI root is active, prompt via a Tkinter password dialog (`sudo -S`)
       and also install a permanent `/etc/udev/rules.d/99-nero-serial.rules` rule so `/dev/ttyACM*` is permanently 0666;
    3. Otherwise fall back to `pkexec chmod 666 <port>` if desktop PolicyKit is available."""
    import shutil
    import subprocess
    p = str(port or "").strip()
    if not p or p == SIM_PORT:
        return False, "invalid port"
    cmds = (["sudo", "-n", "chmod", "666", p], ["chmod", "666", p])
    last_err = ""
    for cmd in cmds:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1.5, stdin=subprocess.DEVNULL)
            if r.returncode == 0 or (os.path.exists(p) and os.access(p, os.R_OK | os.W_OK)):
                if log:
                    log(f"[OK] [V3.65 GRIPPER] 串口权限被拒绝，已自动执行 {' '.join(cmd)} 修复权限")
                return True, " ".join(cmd)
            last_err = (r.stderr or r.stdout or "").strip()
        except Exception as exc:
            last_err = str(exc)

    # Interactive GUI password dialog (sudo -S) + permanent udev rule installation
    udev_rule = 'KERNEL=="ttyACM[0-9]*", MODE="0666"\nKERNEL=="ttyUSB[0-9]*", MODE="0666"\n'
    perm_shell = (
        f"chmod 666 {p} /dev/ttyACM* /dev/ttyUSB* 2>/dev/null; "
        f"printf '{udev_rule}' > /etc/udev/rules.d/99-nero-serial.rules 2>/dev/null; "
        "udevadm control --reload-rules 2>/dev/null; udevadm trigger 2>/dev/null; "
        f"usermod -aG dialout {os.environ.get('USER', 'qiuren')} 2>/dev/null; true"
    )
    try:
        import tkinter as _tk
        from tkinter import simpledialog as _sd
        if getattr(_tk, "_default_root", None) is not None:
            pw = _sd.askstring(
                "串口权限自动修复",
                f"打开夹爪串口 {p} 需要管理员权限。\n请输入 Ubuntu 登录密码（将同时写入永久 udev 规则，以后不再提示）：",
                show="*",
                parent=_tk._default_root,
            )
            if pw:
                r_pw = subprocess.run(
                    ["sudo", "-S", "-p", "", "sh", "-c", perm_shell],
                    input=pw + "\n",
                    capture_output=True,
                    text=True,
                    timeout=8.0,
                )
                if r_pw.returncode == 0 or (os.path.exists(p) and os.access(p, os.R_OK | os.W_OK)):
                    if log:
                        log(f"[OK] [V4.3 GRIPPER] 已通过密码授权修复 {p} 权限，并写入永久 udev 规则 (/etc/udev/rules.d/99-nero-serial.rules)")
                    return True, f"sudo -S chmod 666 {p}"
                last_err = (r_pw.stderr or r_pw.stdout or "").strip()
    except Exception:
        pass

    # Fallback to pkexec graphical dialog if available on desktop
    if os.environ.get("DISPLAY") and shutil.which("pkexec"):
        try:
            r_pk = subprocess.run(
                ["pkexec", "sh", "-c", perm_shell],
                capture_output=True,
                text=True,
                timeout=20.0,
                stdin=subprocess.DEVNULL,
            )
            if r_pk.returncode == 0 or (os.path.exists(p) and os.access(p, os.R_OK | os.W_OK)):
                if log:
                    log(f"[OK] [V4.3 GRIPPER] 已通过 pkexec 图形授权修复 {p} 权限并写入永久 udev 规则")
                return True, f"pkexec chmod 666 {p}"
            last_err = (r_pk.stderr or r_pk.stdout or "").strip()
        except Exception as exc:
            last_err = str(exc)

    if log:
        log(f"[WARN] [V3.65 GRIPPER] 自动修复 {p} 权限未成功（{last_err[:120]}）：请在终端运行一次 sudo chmod 666 {p}")
    return False, last_err


class STSBus:
    """Feetech STS serial bus (same packet handling as the user's sts3215_gui_tool.py), or the simulator."""

    def __init__(self, log=None):
        self.ser = None
        self.sim = None
        self.lock = threading.Lock()
        self.log = log or (lambda s: None)

    @property
    def connected(self):
        return self.sim is not None or (self.ser is not None and self.ser.is_open)

    def connect(self, port, baud=1000000, timeout=0.06, sim_servo=None):
        self.disconnect()
        if port == SIM_PORT:
            self.sim = sim_servo or SimServo()
            return True, "simulated STS3215"
        try:
            import serial
        except Exception:
            return False, "pyserial 未安装：pip3 install pyserial"
        try:
            self.ser = serial.Serial(port=port, baudrate=int(baud), bytesize=8, parity="N", stopbits=1,
                                     timeout=timeout, write_timeout=timeout)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            return True, f"{port} @ {baud}"
        except Exception as exc:
            self.ser = None
            msg_e = str(exc)
            is_perm = (isinstance(exc, PermissionError) or "permission denied" in msg_e.lower()
                       or "errno 13" in msg_e.lower()
                       or (os.path.exists(str(port)) and not os.access(str(port), os.R_OK | os.W_OK)))
            if is_perm and fix_port_permission(port, log=self.log)[0]:
                try:
                    self.ser = serial.Serial(port=port, baudrate=int(baud), bytesize=8, parity="N", stopbits=1,
                                             timeout=timeout, write_timeout=timeout)
                    self.ser.reset_input_buffer()
                    self.ser.reset_output_buffer()
                    return True, f"{port} @ {baud}（已自动修复串口权限）"
                except Exception as exc2:
                    self.ser = None
                    return False, f"{port}: {exc2}"
            return False, f"{port}: {exc}"

    def disconnect(self):
        with self.lock:
            self.sim = None
            if self.ser is not None:
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None

    @staticmethod
    def list_ports():
        try:
            import serial.tools.list_ports as lp
            return [p.device for p in lp.comports()]
        except Exception:
            return []

    def _transfer(self, servo_id, instr, params=()):
        pkt = build_packet(servo_id, instr, params)
        with self.lock:
            if self.sim is not None:
                if servo_id != self.sim.id:
                    return False, "no reply (timeout)", b""
                if instr == INSTR["READ"]:
                    a, n = params[0], params[1]
                    return True, "OK", bytes(self.sim.regs()[a:a + n])
                if instr == INSTR["WRITE"]:
                    self.sim.write(params[0], params[1:])
                return True, "OK", b""
            if self.ser is None or not self.ser.is_open:
                return False, "not connected", b""
            try:
                self.ser.reset_input_buffer()
                self.ser.write(pkt)
                self.ser.flush()
                raw = self.ser.read(5)
                if len(raw) < 5:
                    return False, "no reply (timeout)", b""
                if raw[0] != 0xFF or raw[1] != 0xFF:
                    buf = raw + self.ser.read(32)
                    i = buf.find(b"\xff\xff")
                    if i < 0 or len(buf) < i + 5:
                        return False, "frame sync failed", b""
                    rid, rlen, rerr = buf[i + 2], buf[i + 3], buf[i + 4]
                    rest = buf[i + 5:i + 5 + rlen - 1]
                    if len(rest) < rlen - 1:
                        rest += self.ser.read(rlen - 1 - len(rest))
                else:
                    rid, rlen, rerr = raw[2], raw[3], raw[4]
                    rest = self.ser.read(max(0, rlen - 1))
                if len(rest) < rlen - 1:
                    return False, "short reply", b""
                data, chk = rest[:-1], rest[-1]
                if chk != checksum([rid, rlen, rerr] + list(data)):
                    return False, "checksum error", b""
                return True, ("OK" if not rerr else f"status 0x{rerr:02X}"), bytes(data)
            except Exception as exc:
                return False, f"serial error: {exc}"[:200], b""

    def ping(self, sid):
        return self._transfer(sid, INSTR["PING"])[:2]

    def read(self, sid, addr, n):
        return self._transfer(sid, INSTR["READ"], [addr, n])

    def write(self, sid, addr, data):
        return self._transfer(sid, INSTR["WRITE"], [addr] + [int(b) & 0xFF for b in data])[:2]

    def write_u16(self, sid, addr, v):
        v = int(v) & 0xFFFF
        return self.write(sid, addr, [v & 0xFF, v >> 8])


# ------------------------------------------------------------------------------------------------ gripper
class SO101RealGripper:
    """SO-101 moving jaw on one STS3215.  All angles: a = opening degrees from closed (tips touching)."""

    def __init__(self, cfg=None, log=None, bus=None):
        self.cfg = dict(DEFAULTS, **(cfg or {}))
        self.log = log or (lambda s: None)
        self.bus = bus or STSBus(self.log)
        self.last = None
        self.held = False
        self.last_clamp = None

    # -- conversions
    def steps_for_deg(self, a):
        a_servo = float(a) + float(self.cfg.get("touch_deg") or 0.0)          # V3.67: a counted from the touch point
        return int(round(self.cfg["closed_steps"] + self.cfg["open_sign"] * a_servo * STEPS_PER_REV / 360.0))

    def deg_for_steps(self, steps):
        return (self.cfg["open_sign"] * (float(steps) - self.cfg["closed_steps"]) * 360.0 / STEPS_PER_REV
                - float(self.cfg.get("touch_deg") or 0.0))

    def held_threshold_mm(self):
        return float(self.cfg["min_held_gap_cal_mm"] if self.cfg.get("touch_calibrated") else self.cfg["min_held_gap_mm"])

    # -- connection
    def connect(self, port=None, baud=None, servo_id=None, sim_servo=None):
        if port is not None:
            self.cfg["port"] = port
        if baud is not None:
            self.cfg["baud"] = int(baud)
        if servo_id is not None:
            self.cfg["servo_id"] = int(servo_id)
        ok, msg = self.bus.connect(self.cfg["port"], self.cfg["baud"], sim_servo=sim_servo)
        if not ok:
            return False, msg
        st = self.status()
        if st is None:
            self.bus.disconnect()
            return False, f"舵机 ID {self.cfg['servo_id']} 无应答（端口 / 波特率 / ID？）"
        mode = self.bus.read(self.cfg["servo_id"], REG["MODE"], 1)
        if mode[0] and mode[2] and mode[2][0] != 0:
            self.bus.disconnect()
            return False, f"舵机在模式 {mode[2][0]}（需要模式 0 位置伺服）：用调试工具切回位置模式"
        return True, f"{msg} · ID {self.cfg['servo_id']} · 当前张开 {st['open_deg']:+.1f}° (缝 {st['gap_mm']:.1f} mm)"

    def disconnect(self):
        self.bus.disconnect()

    @property
    def connected(self):
        return self.bus.connected

    def status(self):
        ok, msg, d = self.bus.read(self.cfg["servo_id"], 40, 31)
        if not ok or len(d) < 31:
            return None

        def u16(reg):
            i = reg - 40
            return d[i] | (d[i + 1] << 8)
        pos = decode_s15(u16(REG["PRESENT_POSITION"]))
        load_raw = u16(REG["PRESENT_LOAD"])
        a = self.deg_for_steps(pos)
        stat = d[REG["SERVO_STATUS"] - 40]
        s = {"position_steps": pos, "servo_deg": pos * 360.0 / STEPS_PER_REV, "open_deg": round(a, 2),
             "gap_mm": round(gap_mm_for_open_deg(max(0.0, a)), 2),
             "goal_steps": decode_s15(u16(REG["GOAL_POSITION"])), "speed": decode_s15(u16(REG["PRESENT_SPEED"])),
             "load_pct": round((load_raw & 0x3FF) / 10.0, 1), "current_ma": round(decode_s15(u16(REG["PRESENT_CURRENT"])) * 6.5, 1),
             "voltage_v": d[REG["PRESENT_VOLTAGE"] - 40] / 10.0, "temp_c": d[REG["PRESENT_TEMPERATURE"] - 40],
             "moving": bool(d[REG["MOVING"] - 40]), "torque_on": bool(d[0]),
             "torque_limit_pct": u16(REG["TORQUE_LIMIT"]) / 10.0,
             "errors": [t for b, t in ERROR_BITS.items() if stat & (1 << b)]}
        self.last = s
        return s

    def set_torque_limit(self, pct):
        return self.bus.write_u16(self.cfg["servo_id"], REG["TORQUE_LIMIT"], int(round(min(max(pct, 0.0), 100.0) * 10)))

    def _goto(self, a, speed=None, acc=None):
        steps = self.steps_for_deg(a)
        raw = encode_s15(steps)
        sp = int(min(3400, max(50, speed or self.cfg["speed_steps"])))
        ac = int(min(254, max(0, acc if acc is not None else self.cfg["acc"])))
        return self.bus.write(self.cfg["servo_id"], REG["ACC"], [ac, raw & 0xFF, raw >> 8, 0, 0, sp & 0xFF, sp >> 8])

    def _wait(self, a_goal, timeout_s=3.0, tol_deg=1.5, stall_s=0.30):
        """until at the goal (within tol) or stalled (no motion for stall_s) -> last status"""
        t_end = time.time() + timeout_s
        last_a, t_still, st = None, None, None
        while time.time() < t_end:
            st = self.status()
            if st is None:
                time.sleep(0.05)
                continue
            a = st["open_deg"]
            if abs(a - a_goal) <= tol_deg:
                return st, "reached"
            if last_a is not None and abs(a - last_a) < 0.15:
                t_still = t_still or time.time()
                if time.time() - t_still >= stall_s:
                    return st, "stalled"
            else:
                t_still = None
            last_a = a
            time.sleep(0.04)
        return st, "timeout"

    def open_deg(self, a, timeout_s=3.0):
        a = min(max(float(a), 0.0), float(self.cfg["max_open_deg"]))
        self.set_torque_limit(self.cfg["move_torque_pct"])
        ok, msg = self._goto(a)
        if not ok:
            return {"ok": False, "error": f"写目标失败：{msg}"}
        st, how = self._wait(a, timeout_s, tol_deg=0.5)
        self.held = False
        if st is None:
            return {"ok": False, "error": "no telemetry"}
        return {"ok": how == "reached", "how": how, "open_deg": st["open_deg"], "gap_mm": st["gap_mm"],
                "load_pct": st["load_pct"], "errors": st["errors"]}

    def open_gap(self, gap_mm, timeout_s=3.0):
        return self.open_deg(open_deg_for_gap_mm(gap_mm), timeout_s)

    def clamp(self, torque_pct, squeeze_deg=None, timeout_s=4.0):
        """close past "closed" with a torque limit; held = stopped on something with a gap >= min_held_gap_mm"""
        sq = float(self.cfg["squeeze_deg"] if squeeze_deg is None else squeeze_deg)
        self.set_torque_limit(torque_pct)
        ok, msg = self._goto(-sq, speed=min(self.cfg["speed_steps"], 300))
        if not ok:
            return {"ok": False, "held": False, "error": f"写目标失败：{msg}"}
        st, how = self._wait(-sq, timeout_s, tol_deg=0.8, stall_s=0.35)
        if st is None:
            return {"ok": False, "held": False, "error": "no telemetry"}
        time.sleep(0.15)
        st = self.status() or st
        held = bool(how != "reached" and st["gap_mm"] >= self.held_threshold_mm())
        self.held = held
        self.last_clamp = {"ok": True, "held": held, "how": how, "open_deg": st["open_deg"], "gap_mm": st["gap_mm"],
                           "load_pct": st["load_pct"], "current_ma": st["current_ma"], "torque_limit_pct": torque_pct,
                           "errors": st["errors"]}
        return dict(self.last_clamp)

    def calibrate_touch(self, torque_pct=20.0, timeout_s=4.0):
        """V3.67 EMPTY-CLAMP CALIBRATION: with NOTHING between the jaws, clamp and record where the tips stall
        (= the real "closed" angle); the jaw then reopens to 10 deg.  -> dict(ok, touch_deg, ...)"""
        old = float(self.cfg.get("touch_deg") or 0.0)
        r0 = self.open_deg(10.0)
        if not r0.get("ok"):
            return {"ok": False, "error": f"先张开 10° 失败：{r0.get('error') or r0.get('how')}"}
        self.set_torque_limit(torque_pct)
        sq = float(self.cfg["squeeze_deg"])
        ok, msg = self._goto(-sq, speed=min(self.cfg["speed_steps"], 300))
        if not ok:
            return {"ok": False, "error": f"写目标失败：{msg}"}
        st, how = self._wait(-sq, timeout_s, tol_deg=0.3, stall_s=0.35)
        time.sleep(0.15)
        st = self.status() or st
        if st is None:
            return {"ok": False, "error": "no telemetry"}
        new = old + float(st["open_deg"])
        res = {"how": how, "touch_deg": round(new, 2), "previous_touch_deg": round(old, 2),
               "load_pct": st["load_pct"], "current_ma": st["current_ma"]}
        if how == "reached":
            res.update(ok=False, error=f"夹爪一直合到了 {-sq:.0f}°（没有停住）：指尖没碰到？检查指垫，或加大扭矩")
        elif not (-6.5 <= new <= 6.0):
            res.update(ok=False, error=f"停在 {new:+.2f}°，超出合理范围 −6.5…+6°：夹爪里有东西？先拿走再标定")
        else:
            self.cfg["touch_deg"], self.cfg["touch_calibrated"] = round(new, 2), True
            res["ok"] = True
        self.held = False
        self.open_deg(10.0)
        return res

    def relax(self):
        """torque off (manual handling); the jaw may open under the cup's elasticity"""
        self.held = False
        return self.bus.write(self.cfg["servo_id"], REG["TORQUE_ENABLE"], [0])
