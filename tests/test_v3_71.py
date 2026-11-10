#!/usr/bin/env python3
"""V3.71 offline tests (real cup task: arrival, speed cap, recovery, held-cup recheck, stack verify, live monitor):
C1  Versions V3.71 + markers; frozen modules unchanged; the simulation gripper profiles keep every V3.71 option off;
    the real profile has them on (recheck held, stack by cluster, perception retries, recover) + measured nested height.
C2  Arrive and settle: a slow real arm (speed_percent cap) - the motion waits until the arm is really at the goal and
    still (V3.70: the next step started 80 deg away); the slow arrival teaches the speed cap (WARN) and the next path is
    timed with it; an arm that never arrives -> ARM_NOT_AT_GOAL; wait_still waits for a moving arm.
C3  Fence detour: a joint path refused by the fence (69 mm < 70 mm) -> straight up to the safe height, then the joint
    path again from there.
C4  Recover after a failed motion: RealIO.recover_after_failure waits still + goes straight up; CupSkills calls it after
    MOTION_FAILED / TARGET_UNREACHABLE only when the profile says so.
C5  Perception retry: a cup YOLOE missed once is looked for again (2 retries) instead of CUP_NOT_PERCEIVED.
C6  Plan early stop: a candidate with >= 8 deg margin is taken at once (no further candidates).
C7  Stack verify by cluster: one cluster on the target's place, 4..40 mm (or measured nested + 8 mm) taller than the
    target alone = stacked even when the height class says "cup"; too low / a single cup left -> not stacked.
C8  Held cup geometry (NERO_V366 held_geometry): cone axis of the lower wall of a lifted point cloud -> base x-y.
C9  Twin (needs MuJoCo + OpenGL): the grasp offset is made 10 mm wrong after the pick; the recheck above the target
    measures it, corrects it, the cup still nests; verified.
C10 Live monitor: every task perception reaches the monitor overlay (time-stamped, "识别于 x s 前", old = dashed); GPU
    hint names CUDA_VISIBLE_DEVICES; start_nero.sh clears it; 「实测套叠增量」 field saved and passed to the profile.
"""
import hashlib
import json
import math
import os
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v371_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
os.environ["NERO_V368_TASK_DIR"] = str(TMP / "task")
for k in ("NERO_SCENE", "NERO_V343_ACTIVE_SCENE", "NERO_V362_CUP_SOLREF", "NERO_V308_DEPTH_NOISE_K"):
    os.environ.pop(k, None)
H = lambda b: hashlib.sha256(b).hexdigest()


def snapshot():
    out = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rel = line.split(None, 1)[1].strip()
            out[rel] = H((ROOT / rel).read_bytes()) if (ROOT / rel).is_file() else None
    return out


SNAP0 = snapshot()
import numpy as np                          # noqa: E402
import NERO_V312_Console as CON             # noqa: E402
import NERO_V313_Real_Robot as RR           # noqa: E402
import NERO_V347_Yolo_Perception as YP      # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49   # noqa: E402
import NERO_V362_Cup_Skills as CK           # noqa: E402
import NERO_V365_Real_Cup_Task as RT        # noqa: E402
import NERO_V366_Real_Cup_Yolo as YC        # noqa: E402

try:
    import mujoco  # noqa: F401
    HAVE_MJ = True
except Exception:
    HAVE_MJ = False

# ------------------------------------------------------------------ C1
for m in (CON, RR, YP, W49, RT, YC):
    assert m.VERSION in ("V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2"), (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
for fn, h in (("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a"),
              ("NERO_V2_7_1_Agent.py", "3e52d05ea84d94c8359d5875fa0fee8c2cc42d612b52e59109e1166912439696"),
              ("NERO_V325_Camera.py", "82f2416d73671e7018d09d0cd72c17186154934a315b47c0d19fdb0427afd931"),
              ("NERO_V2_7_1_Unified_Backend.py", "5be2136f84ec4ccdfe6f8bc8d8998410ff9b3b827af4d571ea44373dc427527e"),
              ("NERO_V363_Cup_Perception.py", "b9633693657127fccc101bd902ad38e6bbc014fc907c41bd54227c1711a732c5")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
for fn, mk in (("NERO_V365_Real_Cup_Task.py", "V3.71 ARRIVE AND SETTLE"), ("NERO_V365_Real_Cup_Task.py", "V3.71 SPEED CAP"),
               ("NERO_V365_Real_Cup_Task.py", "V3.71 FENCE DETOUR"), ("NERO_V365_Real_Cup_Task.py", "V3.71 RECOVER"),
               ("NERO_V365_Real_Cup_Task.py", "V3.71 LIVE OVERLAY"), ("NERO_V365_Real_Cup_Task.py", "V3.71 GPU REASON"),
               ("NERO_V362_Cup_Skills.py", "V3.71 HELD CUP RECHECK"), ("NERO_V362_Cup_Skills.py", "V3.71 STACK VERIFY BY CLUSTER"),
               ("NERO_V362_Cup_Skills.py", "V3.71 PERCEPTION RETRY"), ("NERO_V362_Cup_Skills.py", "V3.71 PLAN EARLY STOP"),
               ("NERO_V366_Real_Cup_Yolo.py", "V3.71 HELD CUP RECHECK"), ("start_nero.sh", "V3.71 GPU ENV")):
    assert mk in (ROOT / fn).read_text(encoding="utf-8"), (fn, mk)
for prof in (CK.OFFICIAL_PROFILE, CK.so101_profile()):
    assert not (prof.recheck_held_before_place or prof.stack_by_cluster or prof.perception_retries
                or prof.recover_after_motion_failure) and prof.nested_dz_real_mm is None
rp = RT.real_profile(CK)
assert rp.recheck_held_before_place and rp.stack_by_cluster and rp.perception_retries == 2
assert rp.recover_after_motion_failure and rp.nested_dz_real_mm is None
assert RT.real_profile(CK, cfg={"nested_dz_mm": 12.5}).nested_dz_real_mm == 12.5
print("[C1] V3.71; frozen modules unchanged; simulation profiles: V3.71 options off; real profile: on")

# ------------------------------------------------------------------ C2 arrive and settle + speed cap
Q_LOW = [-0.0426, 0.4456, 0.0533, 1.5554, 0.2194, 0.4861, 1.0801]     # TCP ~0.15 m above the table, margins >= 8 deg


class LagArm:
    """real arm double: exec_path returns at once (like /execute_trajectory), the joints follow at v_real deg/s
    (speed_percent cap); stuck_frac < 1 = the arm stops on the way"""

    def __init__(self, q0, v_real=18.0, stuck_frac=1.0):
        self.q0 = [float(v) for v in q0]
        self.v_real, self.stuck = float(v_real), float(stuck_frac)
        self.path, self.t0, self.calls = None, None, []
        self.model = {"joints": [f"joint{i}" for i in range(1, 8)]}
        self.lock = threading.RLock()

    def q(self):
        with self.lock:
            if self.path is None:
                return list(self.q0)
            a, b = np.asarray(self.path[0]), np.asarray(self.path[-1])
            dist = math.degrees(float(np.max(np.abs(b - a)))) or 1e-9
            u = min(self.stuck, (time.time() - self.t0) * self.v_real / dist)
            return list(a + (b - a) * u)

    def exec_path(self, path, times, label):
        q_now = self.q()
        with self.lock:
            self.calls.append({"label": label, "T": times[-1], "n": len(path)})
            self.q0 = q_now
            self.path, self.t0 = [list(p) for p in path], time.time()
        return {"ok": True}


logs2 = []
cfg = dict(RT.DEFAULT_CFG, joint_vmax_deg_s=48.0, settle_s=0.05)
arm = LagArm(Q_LOW, v_real=18.0)
io = RT.RealIO(arm, None, lambda: None, log=logs2.append, cfg=cfg)
q_goal = list(np.asarray(Q_LOW) + np.radians([40.0, 0, 0, 0, 0, 0, 0]))
t0 = time.time()
r = io.move_joints("test 40 deg", q_goal)
dt = time.time() - t0
assert r["ok"], r
assert r["arrived_s"] > r["planned_s"] * 1.3 and dt >= 40.0 / 18.0 - 0.1, (r, dt)
assert io.joint_cap_deg_s is not None and io.joint_cap_deg_s < 30.0, io.joint_cap_deg_s
assert any("[V3.71 REAL] 机械臂实际比设定慢" in l and "speed_percent" in l for l in logs2), logs2
T_first = arm.calls[-1]["T"]
r2 = io.move_joints("test back", Q_LOW)
assert r2["ok"] and arm.calls[-1]["T"] > T_first * 1.4, (arm.calls[-1]["T"], T_first)
print(f"[C2] slow arm: waited {dt:.1f} s (planned {r['planned_s']} s) -> OK; learned cap {io.joint_cap_deg_s} deg/s; "
      f"next path timed {arm.calls[-1]['T']:.1f} s")

arm_s = LagArm(Q_LOW, v_real=60.0, stuck_frac=0.5)
io_s = RT.RealIO(arm_s, None, lambda: None, log=logs2.append,
                 cfg=dict(RT.DEFAULT_CFG, joint_vmax_deg_s=60.0, settle_s=0.05, arrive_timeout_min_s=1.5))
r3 = io_s.move_joints("stuck", list(np.asarray(Q_LOW) + np.radians([6.0, 0, 0, 0, 0, 0, 0])))
assert not r3["ok"] and "ARM_NOT_AT_GOAL" in r3["error"], r3
print(f"[C2] arm stops half way -> {r3['error'][:60]}...")

arm_m = LagArm(Q_LOW, v_real=10.0)
io_m = RT.RealIO(arm_m, None, lambda: None, log=logs2.append, cfg=cfg)
arm_m.exec_path([Q_LOW, list(np.asarray(Q_LOW) + np.radians([8.0, 0, 0, 0, 0, 0, 0]))], [0.0, 0.5], "external")
ok_m, _q, why_m = io_m.wait_still(0.3)
assert not ok_m and "moving" in why_m
ok_m, _q, _w = io_m.wait_still(3.0)
assert ok_m
print("[C2] wait_still: moving arm -> not still; after it stops -> still")

# ------------------------------------------------------------------ C3 fence detour


class FenceArm(LagArm):
    def exec_path(self, path, times, label):
        fa = CK.FastArm()
        z0 = float(fa.fk(np.asarray(path[0], float))[2, 3])
        if "detour" not in label and z0 < 0.20 and "test fence" in label:
            self.calls.append({"label": label, "refused": True})
            return {"ok": False, "error": "FENCE: 关节路径第 45/78 点越界 -> 未执行: tool 离墙 69 mm < 70 mm（M4–M1 墙）"}
        return LagArm.exec_path(self, path, times, label)


logs3 = []
arm_f = FenceArm(Q_LOW, v_real=500.0)
io_f = RT.RealIO(arm_f, None, lambda: None, log=logs3.append, cfg=dict(RT.DEFAULT_CFG, settle_s=0.02))
goal = list(np.asarray(Q_LOW) + np.radians([-25.0, 0, 0, 0, 0, 0, 0]))
rf = io_f.move_joints("test fence", goal)
labels = [c["label"] for c in arm_f.calls]
assert rf["ok"] and rf.get("detour") == "up first", rf
assert labels[0] == "test fence" and "detour up" in labels[1] and labels[2] == "test fence", labels
assert any("先竖直升到安全高度" in l for l in logs3), logs3
assert max(abs(a_ - b_) for a_, b_ in zip(io_f.current_q(), goal)) < 1e-3    # ended at the requested joints
print(f"[C3] fence refusal -> straight up to {RT.DEFAULT_CFG['safe_z_m']} m, then the joint path: {labels}")

# ------------------------------------------------------------------ C4 recover
logs4 = []
arm_r = LagArm(Q_LOW, v_real=500.0)
io_r = RT.RealIO(arm_r, None, lambda: None, log=logs4.append, cfg=dict(RT.DEFAULT_CFG, settle_s=0.02))
rr = io_r.recover_after_failure("pick_cup: align above cup_1")
assert rr["ok"] and abs(float(io_r.tcp_pose()[0][2]) - RT.DEFAULT_CFG["safe_z_m"]) < 0.004, io_r.tcp_pose()
assert any("运动失败后收尾" in l for l in logs4)
rec_calls = []


def stub_io(prof, perceive=None):
    return types.SimpleNamespace(pose_mode=lambda: "camera" if perceive else "truth", emit=lambda s: None,
                                 gripper_profile=lambda: prof, current_q=lambda: [0.0] * 7,
                                 recover_after_failure=lambda lab: rec_calls.append(lab) or {"ok": True},
                                 clear_latch=lambda l: (True, ""), gripper_state=lambda: {"hold_active": False},
                                 tcp_pose=lambda: ([-0.3, 0.0, 0.3], [1.0, 0, 0, 0]), perceive=perceive)


for prof, want in ((rp, 1), (CK.so101_profile(), 0)):
    rec_calls.clear()
    sk4 = CK.CupSkills(stub_io(prof), *CK.cup_knobs())
    out4 = sk4._abort_result("pick_cup", sk4._Abort("MOTION_FAILED", {"during": "align above cup_1"}), {})
    assert len(rec_calls) == want and (out4["physical_verification"].get("recovered") is True) == bool(want), out4
print("[C4] recover: arm still -> straight up to the safe height; skills call it only with the real profile")

# ------------------------------------------------------------------ C5 perception retry


def cup_e(x, y, h=110.0, kind="cup"):
    return {"kind": kind, "base_xyz_m": [x, y, 0.0], "height_mm": h, "tilt_deg": 0.0, "dets": []}


seq5 = [{"cups": [cup_e(-0.36, -0.05)]}, {"cups": [cup_e(-0.36, -0.05)]},
        {"cups": [cup_e(-0.36, -0.05), cup_e(-0.37, 0.14, 116.0)]}]
n5 = {"i": 0}


def perceive5(tag):
    r_ = seq5[min(n5["i"], len(seq5) - 1)]
    n5["i"] += 1
    return dict(r_, dets=r_["cups"])


sk5 = CK.CupSkills(stub_io(rp, perceive5), *CK.cup_knobs())
sc5 = sk5.scene()
assert set(sc5["objects"]) == {"cup_1", "cup_2"} and n5["i"] == 3, (n5, sc5["objects"].keys())
sk5b = CK.CupSkills(stub_io(CK.so101_profile(), perceive5), *CK.cup_knobs())
n5["i"] = 0
try:
    sk5b.scene()
    raise AssertionError("simulation profile must not retry")
except CK.CupSkills._Abort as e:
    assert e.reason == "CUP_NOT_PERCEIVED"
print("[C5] cup missed in one frame -> looked for again (found on the 2nd retry); simulation profile unchanged")

# ------------------------------------------------------------------ C6 plan early stop
calls6 = []


def fake_plan(knobs, *a, profile=None, allow_wide=True, deadline=None, _wide=False, **k):
    calls6.append("wide" if _wide else knobs["cup_grasp_rim_angle_deg"])
    m = 9.0 if knobs["cup_grasp_rim_angle_deg"] == 25.0 else 12.0
    return {"ok": True, "min_joint_margin_deg": m, "grasp_rim_angle_rel_deg": 0.0, "place_rim_angle_rel_deg": 0.0,
            "margin_ok": True, "grasp_margin_deg": m, "place_margin_deg": m, "grasp_binding": "J7",
            "place_binding": "J7", "_wps": {}, "_q": {}, "_qpath": {}}


logs6 = []
io6 = types.SimpleNamespace(pose_mode=lambda: "truth", emit=logs6.append, gripper_profile=lambda: rp,
                            current_q=lambda: [0.0] * 7)
sk6 = CK.CupSkills(io6, *CK.cup_knobs())
_orig = CK.plan_cup_stack
CK.plan_cup_stack = fake_plan
try:
    s_ = {"rim": np.array([-0.35, -0.13, 0.12]), "quat": [1, 0, 0, 0]}
    t_ = {"rim": np.array([-0.35, 0.13, 0.12]), "quat": [1, 0, 0, 0]}
    p6 = sk6._plan_best_of(s_, t_, [0.0] * 7, (25.0, 50.0))
    assert calls6 == [25.0] and p6["min_joint_margin_deg"] == 9.0, calls6
finally:
    CK.plan_cup_stack = _orig
print("[C6] last-good candidate 9 deg >= 8 deg -> taken at once (1 candidate instead of 3)")

# ------------------------------------------------------------------ C7 stack verify by cluster
logs7 = []


def sk_after_release(heights, cups_after, nd=None):
    prof = RT.real_profile(CK, cfg={"nested_dz_mm": nd})
    io7 = stub_io(prof, lambda tag: {"cups": cups_after, "dets": cups_after})
    io7.emit = logs7.append
    sk = CK.CupSkills(io7, *CK.cup_knobs())
    sk.single_h = dict(heights)
    sk.placed, sk.placed_xy = ("cup_1", "cup_2"), [-0.37, 0.14]
    sk.cam = {"cup_2": sk._entry([-0.37, 0.14, 0.0], 0.0)}
    return sk


h0 = {"cup_1": 107.0, "cup_2": 116.0}
st = sk_after_release(h0, [cup_e(-0.368, 0.141, 128.0)])._state("cup_1", "cup_2")
assert st["stacked"] and st["base_dz_mm"] == 12.0 and st["nested_full"] is None, st
st = sk_after_release(h0, [cup_e(-0.368, 0.141, 128.0)], nd=12.0)._state("cup_1", "cup_2")
assert st["stacked"] and st["nested_full"] is True, st
st = sk_after_release(h0, [cup_e(-0.368, 0.141, 117.0)])._state("cup_1", "cup_2")
assert not st["stacked"], st                                    # only 1 mm taller: the cup is not in it
st = sk_after_release(h0, [cup_e(-0.368, 0.141, 128.0), cup_e(-0.36, -0.05, 107.0)])._state("cup_1", "cup_2")
assert not st["stacked"], st                                    # the source cup still stands on the table
st = sk_after_release(h0, [cup_e(-0.368, 0.141, 190.0)])._state("cup_1", "cup_2")
assert not st["stacked"], st                                    # sitting on the rim, not nested
assert any("已叠好" in l for l in logs7) and any("不算叠好" in l for l in logs7)
print("[C7] one cluster +12 mm on the target -> stacked; +1 mm / source still on the table / +74 mm -> not stacked")

# ------------------------------------------------------------------ C8 held cup geometry
g = CK.SC.cup_geometry()
rng = np.random.default_rng(3)
zb, cx, cy = 0.15, -0.36, 0.12
pts = []
cam = np.array([0.30, 0.0])
for _ in range(4000):
    z = rng.uniform(0.0, g["h"] - 0.03)
    r_ = g["r_bot"] + (g["r_top"] - g["r_bot"]) / g["h"] * z
    a = math.atan2(cam[1] - cy, cam[0] - cx) + rng.uniform(-1.2, 1.2)        # camera-side wall only
    pts.append([cx + r_ * math.cos(a), cy + r_ * math.sin(a), zb + z])
T_cam = np.eye(4)
T_cam[:2, 3] = cam
hg = YC.held_geometry(np.asarray(pts), T_cam, g)
assert math.hypot(hg["held_base_xyz_m"][0] - cx, hg["held_base_xyz_m"][1] - cy) < 0.002, hg
assert abs(hg["held_base_xyz_m"][2] - zb) < 0.003 and hg["held_axis_rms_mm"] < 1.0, hg
print(f"[C8] held cup geometry: axis {hg['held_base_xyz_m'][:2]} (true {[cx, cy]}), rms {hg['held_axis_rms_mm']} mm")

# ------------------------------------------------------------------ C9 twin
gl_ok, gl_reason = False, "MuJoCo not installed"
if HAVE_MJ:
    try:
        CK.TwinIO(gripper="so101").render_fixed_camera()
        gl_ok = True
    except Exception as exc:
        gl_reason = str(exc)[:80]
if gl_ok:
    import v365_real_twin as RTW  # noqa: E402
    CK._LAST_ANGLES.clear()
    log9 = []
    io9, tw9, arm9, grip9 = RTW.make(log=log9.append)
    seen = []
    io9.on_perception = lambda res: seen.append(len(res.get("dets") or []))
    sk9 = RT.make_skills(io9)
    for name, args in (("observe_scene", {}), ("plan_cup_stack", {"source_cup": "cup_1", "target_cup": "cup_2"}),
                       ("pick_cup", {"source_cup": "cup_1"})):
        rr9 = sk9.run_tool(name, args)
        assert rr9["ok"], (name, rr9)
    tcp, qt = io9.tcp_pose()
    sk9.grip_offset = sk9.grip_offset + CK.quat_xyzw_to_R(qt).T @ np.array([0.010, 0.0, 0.0])   # 10 mm wrong
    rp9 = sk9.run_tool("place_cup_into", {"target_cup": "cup_2"})
    chk = rp9["physical_verification"].get("held_recheck") or {}
    first = (chk.get("checks") or [{}])[0]
    assert rp9["ok"] and first.get("applied") and 6.0 <= first["offset_mm"] <= 14.0, rp9["physical_verification"]
    rv = sk9.run_tool("verify_stack", {"upper_cup": "cup_1", "lower_cup": "cup_2"})
    assert rv["ok"], rv
    assert any("放置前复查" in l for l in log9) and len(seen) >= 4
    print(f"[C9] twin: offset made 10 mm wrong -> recheck measured {first['offset_mm']} mm, corrected, nested, verified; "
          f"{len(seen)} perceptions sent to the monitor")
else:
    print(f"[C9] skipped: no twin camera ({gl_reason})")

# ------------------------------------------------------------------ C10 live monitor, GPU, nested field
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
mtk.Text.return_value.yview.return_value = (0.0, 1.0)


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
    src = wrist_src = worker = result = calib = None


tab = RT.RealCupTaskTabUI(mtk, mttk, mock.MagicMock(), _Pipe(), types.SimpleNamespace(model=None), _Cam(),
                          log=lambda s: None, visible_fn=lambda: False)
res10 = {"cups": [], "dets": [{"det_id": "#1", "kind": "cup", "contour": [[10, 10], [40, 10], [40, 50]], "conf": 0.8,
                               "base_xyz_m": [-0.36, 0.0, 0.0], "height_mm": 110.0}]}
tab._apply_overlay(res10, None)
assert tab.overlay.get("dets") and RT.overlay_age_s(tab.overlay) < 1.0
assert RT.overlay_age_s({"t": time.time() - 10.0}) > RT.OVERLAY_OLD_S and RT.overlay_age_s({}) is None
src = (ROOT / "NERO_V365_Real_Cup_Task.py").read_text(encoding="utf-8")
assert "io.on_perception = lambda res, io_=io: self._ui(lambda: self._apply_overlay(res" in src
assert "识别于 {age:.1f} s 前" in src
tab.v_nested.set("12.5")
assert tab.geometry_cfg()["nested_dz_mm"] == 12.5
tab._save()
saved = json.loads((Path(os.environ["NERO_CALIB_DIR"]) / "real_cup_task.json").read_text(encoding="utf-8"))
assert saved["geometry"]["nested_dz_mm"] == 12.5
tab.v_nested.set("90")
try:
    tab.geometry_cfg()
    raise AssertionError("90 mm must be refused")
except ValueError as exc:
    assert "套叠增量" in str(exc)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    hint = RT.gpu_hint(None)
finally:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
assert "CUDA_VISIBLE_DEVICES=''" in hint and "nvidia-smi" in hint
sh = (ROOT / "start_nero.sh").read_text(encoding="utf-8")
assert "unset CUDA_VISIBLE_DEVICES" in sh and sh.index("unset CUDA_VISIBLE_DEVICES") < sh.index('exec "$PY" "$CONSOLE"')
print("[C10] live overlay with age stamp; 实测套叠增量 field saved (3-40 mm); GPU hint; start_nero.sh clears "
      "CUDA_VISIBLE_DEVICES")

SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("ALL V3.71 TESTS PASSED")
