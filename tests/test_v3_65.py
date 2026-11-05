#!/usr/bin/env python3
"""V3.65 offline tests (stack_cups_so101 skills + parameters on the real NERO; SO-101 jaw on an STS3215 servo):
W1  Versions V3.65 + markers; NERO_V308 / V326 / Agent / Camera tab / backend / fixed-camera process / cup perception /
    cup skills / scenes / SO-101 model unchanged (the real task reuses them, nothing in the simulation changed).
W2  STS3215 protocol: packets byte-identical to the user's sts3215_gui_tool.py framing (ping, WritePosEx, read), sign-
    bit coding; angle convention 180 deg (2048) = closed, +a opens (OPEN_SIGN flips it).
W3  Jaw mapping + clamp: a = 0 where the tips touch; the skills' virtual openings map to the same jaw angle as in the
    simulation (pregrasp = 32 mm tip gap); clamp torque from the skill force (5 N -> ~18 %, bounded); simulated servo:
    clamp on a 2 mm wall -> held (gap ~2 mm, load = torque limit), empty clamp -> not held; open stops at the goal.
W4  Joint-path timing: starts at 0, strictly increasing, no joint faster than the limit, no TCP faster than the line
    limit, softer start / stop.
W5  NERO_V313 RealPipeline.move_joint_path (fake ROS monitor subprocess): gates (MoveIt not green, teach mode, joint
    limit, malformed times, monitor without joint-path capability) and the sent command (path + times, fence fields);
    PipelineArm executes through it and falls back to short MoveIt joint goals on an old monitor.
W6  The monitor's do_path (source extracted from cmd_monitor, run against fake ROS messages): ExecuteTrajectory goal
    with the exact points / times / zero end velocities, malformed path refused, fence violation refused (nothing
    sent), dispatch placed before do_move, status advertises caps=["joint_path"].
W7  RealIO end to end on the MuJoCo SO-101 twin as the real body (twin camera = D435i frames + T_base_cam, joint
    paths executed exactly, STS3215-style gripper adapter): mode 3 cup_1 -> cup_2 nested and back at the start
    posture, every path starts at the current joints inside the limits; mode 2 hover: no clamp, never below the
    pregrasp height near the cups, back at the start; mode 1 plan: no motion at all.
W8  Console: the 「NERO实机任务」 tab is added right after 「技能参数旋钮」 (mocked Tk, sim + real); the tab builds,
    shows the CS skill + SO-101 profile, two camera monitors, run modes 1-3; settings go to NERO_CALIB_DIR only;
    the release folder is not written.
"""
import ast
import hashlib
import importlib
import json
import math
import os
import sys
import tempfile
import textwrap
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v365_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
for k in ("NERO_SCENE", "NERO_V343_ACTIVE_SCENE", "NERO_V362_CUP_SOLREF", "NERO_V308_DEPTH_NOISE_K",
          "NERO_V365_GRIPPER_OPEN_SIGN", "NERO_V365_GRIPPER_CLOSED_STEPS"):
    os.environ.pop(k, None)
H = lambda b: hashlib.sha256(b).hexdigest()


def snapshot():
    out = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rel = line.split(None, 1)[1].strip()
            out[rel] = H((ROOT / rel).read_bytes()) if (ROOT / rel).is_file() else None
    out["__config__"] = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "config").rglob("*") if p.is_file())
    return out


SNAP0 = snapshot()
import numpy as np                          # noqa: E402
import NERO_V312_Console as CON             # noqa: E402
import NERO_V313_Real_Robot as RR           # noqa: E402
import NERO_V347_Yolo_Perception as YP      # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49   # noqa: E402
import NERO_V362_Cup_Skills as CK           # noqa: E402
import NERO_V365_STS3215_Gripper as GR      # noqa: E402
import NERO_V365_Real_Cup_Task as RT        # noqa: E402

try:
    import mujoco  # noqa: F401
    HAVE_MJ = True
except Exception:
    HAVE_MJ = False

# ------------------------------------------------------------------ W1
for m in (CON, RR, YP, W49, GR, RT):
    assert m.VERSION in ("V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81"), (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
for fn, h in (("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a"),
              ("NERO_V2_7_1_Agent.py", "3e52d05ea84d94c8359d5875fa0fee8c2cc42d612b52e59109e1166912439696"),
              ("NERO_V325_Camera.py", "013fc8c308092f863c7ccb1a37598c80536deed2cfbcef2b0daccc417da2806d"),
              ("NERO_V2_7_1_Unified_Backend.py", "5be2136f84ec4ccdfe6f8bc8d8998410ff9b3b827af4d571ea44373dc427527e"),
              ("NERO_V331_Fixed_Camera.py", "8314f82cce52016f3dd08bad22ce08dbd9bdb1b263fa6aef23453467a9b3a341"),
              ("NERO_V363_Cup_Perception.py", "b9633693657127fccc101bd902ad38e6bbc014fc907c41bd54227c1711a732c5")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
for fn in ("NERO_V362_Cup_Skills.py", "NERO_V343_Scenes.py", "NERO_V364_SO101_Gripper.py", "NERO_V359_Strategy.py"):
    assert "V3.65" not in (ROOT / fn).read_text(encoding="utf-8"), f"{fn} must stay the V3.64 simulation code"
for fn, mk in (("NERO_V365_STS3215_Gripper.py", GR.MARKER), ("NERO_V365_Real_Cup_Task.py", RT.MARKER),
               ("NERO_V313_Real_Robot.py", "V3.65 JOINT PATH EXECUTION"), ("NERO_V312_Console.py", "V3.65 NERO REAL CUP TASK TAB")):
    assert mk in (ROOT / fn).read_text(encoding="utf-8"), (fn, mk)
print("[W1] V3.65; simulation modules (backend, skills, scenes, SO-101 model, perception) unchanged")

# ------------------------------------------------------------------ W2
assert GR.build_packet(1, GR.INSTR["PING"]) == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])
raw = GR.encode_s15(2730)
pkt = GR.build_packet(1, GR.INSTR["WRITE"], [GR.REG["ACC"], 30, raw & 0xFF, raw >> 8, 0, 0, 600 & 0xFF, 600 >> 8])
body = [1, 10, 3, 41, 30, 2730 & 0xFF, 2730 >> 8, 0, 0, 0x58, 0x02]
assert pkt == bytes([0xFF, 0xFF] + body + [(~sum(body)) & 0xFF])
assert GR.build_packet(1, GR.INSTR["READ"], [40, 31]) == bytes([0xFF, 0xFF, 1, 4, 2, 40, 31, (~(1 + 4 + 2 + 40 + 31)) & 0xFF])
for v in (0, 1, 2048, -5, -2047, 30000):
    assert GR.decode_s15(GR.encode_s15(v)) == v
assert GR.encode_s15(-5) == 0x8005
g = GR.SO101RealGripper()
assert g.steps_for_deg(0) == 2048 and g.steps_for_deg(60) == 2048 + 683 and abs(g.deg_for_steps(2731) - 60.03) < 0.01
g2 = GR.SO101RealGripper({"open_sign": -1})
assert g2.steps_for_deg(60) == 2048 - 683 and abs(g2.deg_for_steps(2048 - 683) - 60.03) < 0.01
print("[W2] STS3215 packets = user tool framing; 2048 = closed, +a opens (sign switchable)")

# ------------------------------------------------------------------ W3
import NERO_V364_SO101_Gripper as G  # noqa: E402
assert abs(GR.gap_mm_for_open_deg(0.0)) < 1e-6 and GR.gap_mm_for_open_deg(10.0) > 10.0
a_pre = GR.open_deg_for_virtual_mm(G.opening_mm_for_tip_gap(32.0))
assert abs(GR.gap_mm_for_open_deg(a_pre) - 32.0) < 0.05 and 15.0 < a_pre < 40.0
assert abs(GR.open_deg_for_virtual_mm(GR.virtual_mm_for_open_deg(17.0)) - 17.0) < 1e-6
assert abs(GR.open_deg_for_gap_mm(GR.gap_mm_for_open_deg(12.0)) - 12.0) < 1e-3
assert 60.0 < GR.lever_m() * 1000 < 80.0
assert 15.0 <= GR.clamp_torque_pct(5.0) <= 25.0 and GR.clamp_torque_pct(0.1) == 15.0 and GR.clamp_torque_pct(100) == 60.0
sv = GR.SimServo()
gr = GR.SO101RealGripper(log=lambda s: None)
ok, msg = gr.connect(GR.SIM_PORT, sim_servo=sv)
assert ok, msg
r = gr.open_gap(32.0)
assert r["ok"] and abs(r["gap_mm"] - 32.0) < 0.8, r
sv.obstacle_gap_mm = 2.0
c = gr.clamp(GR.clamp_torque_pct(5.0))
assert c["held"] is True and 1.6 < c["gap_mm"] < 2.5 and c["load_pct"] >= 15.0 and gr.held, c
sv.obstacle_gap_mm = None
gr.open_gap(20.0)
c = gr.clamp(20.0)
assert c["held"] is False and c["gap_mm"] < 0.5 and not gr.held, c
assert gr.open_deg(500.0)["open_deg"] <= gr.cfg["max_open_deg"] + 0.6
bad = GR.SO101RealGripper()
assert not bad.connect(GR.SIM_PORT, servo_id=7, sim_servo=GR.SimServo(servo_id=1))[0]       # wrong ID: no reply
print(f"[W3] tips touch at a=0; pregrasp 32 mm = {a_pre:.1f} deg; 5 N -> {GR.clamp_torque_pct(5.0)} % torque; "
      "clamp on 2 mm wall held, empty clamp detected")

# ------------------------------------------------------------------ W4
path = [[0.0] * 7] + [[0.01 * i] + [0.0] * 6 for i in range(1, 41)]
tcp = [[0.0, 0.0, 0.003 * i] for i in range(41)]
ts = RT.time_joint_path(path, math.radians(10.0), tcp, 0.02)
assert ts[0] == 0.0 and all(b > a for a, b in zip(ts, ts[1:]))
for i in range(1, 41):
    dt = ts[i] - ts[i - 1]
    assert 0.01 / dt <= math.radians(10.0) + 1e-6 and 0.003 / dt <= 0.02 + 1e-9
mid = ts[21] - ts[20]
assert ts[1] - ts[0] > 1.5 * mid and ts[-1] - ts[-2] > 1.5 * mid
print(f"[W4] joint path timing OK ({ts[-1]:.1f} s, soft start / stop)")

# ------------------------------------------------------------------ W5
FAKE_MON = TMP / "fake_mon.py"
FAKE_MON.write_text(textwrap.dedent("""
    import json, os, sys, threading, time
    stop = threading.Event(); pos = [0.0] * 7; caps = os.environ.get("FAKE_CAPS", "1") == "1"
    log = open(os.environ["FAKE_LOG"], "a")
    def rd():
        for line in sys.stdin:
            c = json.loads(line); log.write(line); log.flush()
            if c.get("quit"): stop.set(); return
            mv = c.get("move")
            if mv:
                print(json.dumps({"event": "move_accepted", "id": mv["id"]}), flush=True)
                time.sleep(0.1)
                if mv.get("path"):
                    ok = max(abs(a - b) for a, b in zip(mv["path"][0], pos)) < 1e-6 and mv["times"][0] == 0
                    if ok:
                        pos[:] = mv["path"][-1]
                    print(json.dumps({"event": "move_result", "id": mv["id"], "ok": ok, "joint_path": True,
                                      "n_points": len(mv["path"]), "error": None if ok else "start mismatch"}), flush=True)
                else:
                    pos[:] = mv["positions"]
                    print(json.dumps({"event": "move_result", "id": mv["id"], "ok": True, "error_code": 1}), flush=True)
    threading.Thread(target=rd, daemon=True).start()
    while not stop.is_set():
        print(json.dumps({"event": "js", "joints": {"names": ["joint%d" % i for i in range(1, 8)], "position": pos}}), flush=True)
        st = {"event": "status", "js_hz": 10, "joints": None}
        if caps:
            st["caps"] = ["joint_path"]
        print(json.dumps(st), flush=True)
        stop.wait(0.1)
"""))


def make_pipe(caps=True):
    p = RR.RealPipeline(lambda: {"can_port": "can0", "speed_percent": 10, "domain_id": 42},
                        can_opener=lambda _p: (_ for _ in ()).throw(OSError("no can")))
    logf = TMP / f"mon_{int(caps)}.jsonl"
    p.monitor.start([sys.executable, "-u", str(FAKE_MON)], env=dict(os.environ, FAKE_CAPS="1" if caps else "0",
                                                                    FAKE_LOG=str(logf)), stdin_pipe=True)
    return p, logf


model = {"joints": [f"joint{i}" for i in range(1, 8)], "limits": [tuple(v) for v in CK.JOINT_LIMITS], "arm_group": "arm"}
pipe, logf = make_pipe(True)
time.sleep(0.6)
qa = [0.0] * 7
qb = [0.05, -0.02, 0.0, 0.1, 0.0, 0.05, 0.0]
pth = [list(np.asarray(qa) + (np.asarray(qb) - qa) * u) for u in np.linspace(0, 1, 6)]
tms = RT.time_joint_path(pth, math.radians(12.0))
assert pipe.move_joint_path(model, pth, tms)["error"].startswith("step 6")                   # MoveIt not green
for k in RR.STEP_KEYS:
    pipe.state[k] = (RR.GREEN, "test")
assert pipe.supports_joint_path()
pipe.teach_mode = True
assert "示教" in pipe.move_joint_path(model, pth, tms)["error"]
pipe.teach_mode = False
assert "outside limit" in pipe.move_joint_path(model, pth[:-1] + [[0, 0, 0, 0, 0, 1.2, 0]], tms)["error"]
assert "malformed" in pipe.move_joint_path(model, pth, [0.0] + tms[2:] + [tms[-1]])["error"]
assert "malformed" in pipe.move_joint_path(model, pth, [0.1] + tms[1:])["error"]
r = pipe.move_joint_path(model, pth, tms, label="w5")
assert r["ok"] and r.get("joint_path") and r.get("n_points") == 6, r
sent = [json.loads(l) for l in logf.read_text().splitlines() if '"path"' in l]
assert len(sent) == 1 and sent[0]["move"]["times"] == tms and len(sent[0]["move"]["path"]) == 6
arm = RT.PipelineArm(pipe, lambda: model, log=lambda s: None)
time.sleep(0.3)
assert max(abs(a - b) for a, b in zip(arm.q(), qb)) < 1e-6
back = [list(np.asarray(qb) + (np.asarray(qa) - qb) * u) for u in np.linspace(0, 1, 6)]
assert arm.exec_path(back, RT.time_joint_path(back, math.radians(12.0)), "w5 back")["kind"] == "path"
pipe.monitor.send({"quit": True})
pipe.monitor.stop()
pipe2, logf2 = make_pipe(False)                            # an old monitor without the capability
time.sleep(0.6)
for k in RR.STEP_KEYS:
    pipe2.state[k] = (RR.GREEN, "test")
assert pipe2.move_joint_path(model, pth, tms).get("unsupported")
logs = []
arm2 = RT.PipelineArm(pipe2, lambda: model, log=logs.append)
r2 = arm2.exec_path(pth, tms, "w5 fallback")
assert r2["ok"] and r2["kind"] == "steps" and any("short MoveIt joint goals" in l for l in logs), r2
assert not any('"path"' in l for l in logf2.read_text().splitlines())
pipe2.monitor.send({"quit": True})
pipe2.monitor.stop()
print("[W5] RealPipeline.move_joint_path gates + command; PipelineArm path / fallback to short joint goals")

# ------------------------------------------------------------------ W6
SRC = (ROOT / "NERO_V313_Real_Robot.py").read_text(encoding="utf-8")
tree = ast.parse(SRC)
mon = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cmd_monitor")
dp = next(n for n in ast.walk(mon) if isinstance(n, ast.FunctionDef) and n.name == "do_path")
dp_src = textwrap.dedent(ast.get_source_segment(SRC, dp))
i_path = SRC.index('elif c.get("move") and c["move"].get("path"):')
i_move = SRC.index('elif c.get("move"):\n                    try:\n                        do_move(c["move"])')
assert i_path < i_move and 'caps=["joint_path"]' in SRC


class _Fut:
    def __init__(self, v):
        self.v = v

    def result(self):
        return self.v

    def add_done_callback(self, cb):
        cb(self)


class _Msg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


sent_goals = []


class _GH:
    accepted = True

    def get_result_async(self):
        return _Fut(_Msg(result=_Msg(error_code=_Msg(val=1)), status=4))


class _AC:
    def __init__(self, node, typ, name):
        self.name = name

    def wait_for_server(self, timeout_sec=0):
        return True

    def send_goal_async(self, goal):
        sent_goals.append(goal)
        return _Fut(_GH())


def _mod(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


class _JT:
    def __init__(self):
        self.joint_names, self.points = [], []


class _JTP:
    def __init__(self):
        self.positions, self.velocities, self.time_from_start = [], [], None


class _ET:
    class Goal:
        trajectory = None


_mod("rclpy")
_mod("rclpy.action", ActionClient=_AC)
_mod("moveit_msgs")
_mod("moveit_msgs.action", ExecuteTrajectory=_ET)
_mod("moveit_msgs.msg", RobotTrajectory=type("RobotTrajectory", (), {"joint_trajectory": None}))
_mod("trajectory_msgs")
_mod("trajectory_msgs.msg", JointTrajectory=_JT, JointTrajectoryPoint=_JTP)
_mod("builtin_interfaces")
_mod("builtin_interfaces.msg", Duration=lambda sec=0, nanosec=0: _Msg(sec=sec, nanosec=nanosec))
events = []
import NERO_V314_Joint_Control as JC  # noqa: E402
ns = {"st": {"move": None, "urdf_parsed": None}, "node": object(), "_emit": lambda **kw: events.append(kw), "JC": JC,
      "time": time}
exec(dp_src, ns)
do_path = ns["do_path"]
names = model["joints"]
mv = {"id": 1, "joints": names, "path": pth, "times": tms}
do_path(mv)
assert events[-1]["event"] == "move_result" and events[-1]["ok"] and events[-1]["joint_path"], events[-1]
tj = sent_goals[-1].trajectory.joint_trajectory
assert tj.joint_names == names and len(tj.points) == 6
for p, q, t in zip(tj.points, pth, tms):
    assert p.positions == [float(v) for v in q]
    assert abs(p.time_from_start.sec + p.time_from_start.nanosec * 1e-9 - t) < 1e-6
assert tj.points[0].velocities == [0.0] * 7 and tj.points[-1].velocities == [0.0] * 7
v2 = tj.points[2].velocities
assert all(abs(v2[j] - (pth[3][j] - pth[1][j]) / (tms[3] - tms[1])) < 1e-9 for j in range(7))
n_goals = len(sent_goals)
do_path(dict(mv, id=2, times=tms[:-1]))
assert not events[-1]["ok"] and "malformed" in events[-1]["error"] and len(sent_goals) == n_goals
import NERO_V315_Consistency_Safety as CS  # noqa: E402
_orig_pfc = CS.path_fence_check
CS.path_fence_check = lambda *a, **k: {"waypoint": 3, "of": 6, "violations": [{"text": "test violation", "kind": "z", "value": 0.01}]}
ns["st"]["urdf_parsed"] = {"x": 1}
try:
    do_path(dict(mv, id=3, fence={"enabled": True}))
finally:
    CS.path_fence_check = _orig_pfc
assert not events[-1]["ok"] and "FENCE" in events[-1]["error"] and len(sent_goals) == n_goals
ns["st"]["urdf_parsed"] = None
do_path(dict(mv, id=4, fence={"enabled": True}))
assert not events[-1]["ok"] and "not checked" in events[-1]["error"] and len(sent_goals) == n_goals
for k in [k for k in sys.modules if k.split(".")[0] in ("rclpy", "moveit_msgs", "trajectory_msgs", "builtin_interfaces")]:
    sys.modules.pop(k)
print("[W6] monitor do_path: exact points / times / velocities sent; malformed and fence-violating paths not sent")

# ------------------------------------------------------------------ W7
if HAVE_MJ:
    import v365_real_twin as RTW  # noqa: E402
    gl_ok = True
    try:
        CK.TwinIO(gripper="so101").render_fixed_camera()
    except Exception as exc:
        gl_ok, gl_reason = False, str(exc)[:80]
    if gl_ok:
        log7 = []
        io, tw, arm7, grip7 = RTW.make(log=log7.append)
        out = RT.run_plan_preview(io, "cup_1", "cup_2")
        assert out["ok"] and not arm7.calls and not grip7.clamps, out["results"][-1]
        io, tw, arm7, grip7 = RTW.make(log=log7.append)
        q_start = list(tw.q)
        out = RT.run_hover(io, "cup_1", "cup_2")
        assert out["ok"], out.get("steps")
        assert not grip7.clamps and max(abs(a - b) for a, b in zip(tw.q, q_start)) < 1e-6
        z_pre = float(out["skills"].plan["_wps"]["pregrasp"][0][2])
        assert min(c_["min_z"] for c_ in arm7.calls) >= z_pre - 1e-3, [(c_["label"], c_["min_z"]) for c_ in arm7.calls]
        # V3.67: the real profile arrives above the cup and rotates in place first (align-rotate-descend)
        assert [c_["label"] for c_ in arm7.calls][0].startswith(("V3.65 hover cup_1", "V3.63 align above cup_1"))
        io, tw, arm7, grip7 = RTW.make(log=log7.append)
        out = RT.run_execute(io, "cup_1", "cup_2")
        a_, b_ = tw.cup_pose("cup_1"), tw.cup_pose("cup_2")
        st7 = CK.SC.cup_task_state(a_[0], a_[1], b_[0], b_[1], origin="rim")
        assert out["ok"] and st7["nested_full"], [r_.get("reason") for r_ in out["results"] if not r_.get("ok")]
        assert max(abs(a - b) for a, b in zip(tw.q, [0.0] * 7)) < 1e-6
        assert grip7.clamps and all(15.0 <= p_ <= 25.0 for p_, _f in grip7.clamps)
        assert max(c_["max_step_deg"] for c_ in arm7.calls) <= 2.0 + 1e-6
        # V3.66: the real perception is the Camera tab's YOLOE pipeline (twin camera: YOLOE stand-in)
        assert any("V3.65 REAL CUP PERCEPTION" in l or "V3.66 REAL CUP YOLOE" in l for l in log7)
        print(f"[W7] twin as the real body: plan = no motion; hover OK (no clamp, back at start); execute nested, "
              f"{len(arm7.calls)} joint paths, back at the start posture")
    else:
        print(f"[W7] skipped: no OpenGL for the twin camera ({gl_reason})")
else:
    print("[W7] skipped: MuJoCo not installed")

# ------------------------------------------------------------------ W8
import unittest.mock as mock  # noqa: E402


class FakeVar:
    def __init__(self, master=None, value=None, **_kw):
        self._v = value

    def get(self):
        return self._v

    def set(self, v):
        self._v = v

    def trace_add(self, *_a, **_k):
        return "t"


mtk, mttk = mock.MagicMock(), mock.MagicMock()
for nm in ("StringVar", "BooleanVar", "IntVar", "DoubleVar"):
    setattr(mtk, nm, FakeVar)
mtk.ttk = mttk
mtk.Text.return_value.yview.return_value = (0.0, 1.0)
mtk.Text.return_value.index.return_value = "1.0"
sys.modules["tkinter"] = mtk
sys.modules["tkinter.ttk"] = mttk
for sub in ("font", "messagebox", "simpledialog", "filedialog"):
    sys.modules[f"tkinter.{sub}"] = mock.MagicMock()
root = mock.MagicMock()
root.winfo_screenwidth.return_value = 1920
root.winfo_screenheight.return_value = 1080
mtk.Tk.return_value = root
root.mainloop.side_effect = lambda: None
for tab in ("sim", "real"):
    mttk.Notebook.return_value.add.reset_mock()
    CON.run_console(tab)
    texts = [c_.kwargs.get("text") for c_ in mttk.Notebook.return_value.add.call_args_list]
    assert "NERO实机任务" in texts and texts.index("NERO实机任务") == texts.index("技能参数旋钮") + 1, texts
logs8 = []


class _Pipe:
    teach_mode = False

    def lamp(self, k):
        return "grey"

    def supports_joint_path(self):
        return False

    def actual_arm_deg(self, m):
        return None

    def move_busy(self):
        return False


class _Cam:
    src = wrist_src = result = calib = worker = None


tab = RT.RealCupTaskTabUI(mtk, mttk, mock.MagicMock(), _Pipe(), types.SimpleNamespace(model=None), _Cam(),
                          log=logs8.append)
assert "[CS]" in tab.v_skill.get() and "SO-101" in tab.v_skill.get() and "固定指在杯外" in tab.v_skill.get()
assert tab.v_mode.get() == "plan" and [k for k, _ in RT.MODES] == ["plan", "hover", "execute"]
assert tab.DW / tab.DH == 16 / 9
tab._run_mode("execute")
tab.poll()
assert any("不能开始" in l and "主相机" in l and "夹爪" in l for l in logs8), logs8
tab._save()
saved = json.loads((Path(os.environ["NERO_CALIB_DIR"]) / "real_cup_task.json").read_text(encoding="utf-8"))
assert saved["gripper"]["servo_id"] == 1 and "motion" in saved
SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("[W8] console: NERO实机任务 right after 技能参数旋钮; tab shows CS + SO-101; preflight refuses without "
      "camera / gripper; settings in NERO_CALIB_DIR; release folder unchanged")
print("ALL V3.65 TESTS PASSED")
