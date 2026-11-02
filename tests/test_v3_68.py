#!/usr/bin/env python3
"""V3.68 offline tests (real cup task: lenient pick, fast plan, faster perception, fingertip overlay, roll sweep,
task records + summary):
Z1  Versions V3.68 + markers; frozen modules unchanged; the simulation's gripper profiles keep their behaviour
    (lenient_pick off, wide step 15 deg, no plan time budget, no remembered angles); task/ is git-ignored.
Z2  Lenient real pick on the twin: a cup measured 8 deg tilted (6-10 deg made every V3.67 pick abort after the
    descent on a stale value) is clamped, lifted and stacked; a clamp that reports no hold although the cup left the
    table -> NO_TWO_FINGER_CONTACT with a detail (both checks must agree); only a fresh view of a cup pushed > 30 mm
    stops the pick before the clamp.
Z3  Fast plan: the second plan starts from the remembered rim angles (log line, faster); a plan that needs the wide
    search finishes within the time budget (30 deg step) and is executable.
Z4  Perception: timings per stage in the result and the log line; RealIO.perceive no longer writes snapshots (only a
    failed run does); fits use at most FIT_MAX_PTS wall points.
Z5  Task record: one folder per task under NERO_V368_TASK_DIR with before / after images of every tool call,
    steps.jsonl, task.json, summary.txt; step lines "[V3.68 STEP n]" with the key numbers; the end-of-task block
    "[REAL TASK SUMMARY GREEN]" (STATUS / TASK / SETUP / CUPS / TOOLS / LAST FAILURE / TIME / RECORD).
Z6  Fingertip model: fixed / moving fingertip distance = the jaw gap for the opening; the roll rotates the tips about
    link7 x; roll sweep over three rolls returns one row each (the model roll plans).
Z7  Tab (mocked Tk): 「主相机画面画出模型指尖」 + 「偏转角扫描」, show_tips saved; the overlay draws the F / M fingertips and
    the jaw line for the arm pose; roll sweep refused without camera; the task's own YOLOE detector instance is used
    only for a real YoloEDetector.
"""
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v368_"))
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
import NERO_V365_STS3215_Gripper as GR      # noqa: E402
import NERO_V366_Real_Cup_Yolo as YC        # noqa: E402
import NERO_V368_Real_Task_Record as TR     # noqa: E402

try:
    import mujoco  # noqa: F401
    HAVE_MJ = True
except Exception:
    HAVE_MJ = False

# ------------------------------------------------------------------ Z1
for m in (CON, RR, YP, W49, RT, YC, TR):
    assert m.VERSION == "V3.68", (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() == "V3.68"
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
for fn, h in (("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a"),
              ("NERO_V2_7_1_Agent.py", "3e52d05ea84d94c8359d5875fa0fee8c2cc42d612b52e59109e1166912439696"),
              ("NERO_V325_Camera.py", "5f41c189ea742fc4c3b4d4b33b7190c5a456ba6e6eb6adf39bde92bf1ed82688"),
              ("NERO_V2_7_1_Unified_Backend.py", "bb9b7e5e927ed3fda9112b2edff0d99a482ade9ac3dc6a0b66901b70d257a04a"),
              ("NERO_V331_Fixed_Camera.py", "8314f82cce52016f3dd08bad22ce08dbd9bdb1b263fa6aef23453467a9b3a341"),
              ("NERO_V363_Cup_Perception.py", "b9633693657127fccc101bd902ad38e6bbc014fc907c41bd54227c1711a732c5")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
for fn, mk in (("NERO_V362_Cup_Skills.py", "V3.68 LENIENT REAL PICK"), ("NERO_V362_Cup_Skills.py", "V3.68 FAST PLAN"),
               ("NERO_V366_Real_Cup_Yolo.py", "V3.68 FAST PERCEPTION"), ("NERO_V365_Real_Cup_Task.py", "V3.68 FINGERTIP OVERLAY"),
               ("NERO_V365_Real_Cup_Task.py", "V3.68 REAL TASK RECORD"), ("NERO_V368_Real_Task_Record.py", TR.MARKER)):
    assert mk in (ROOT / fn).read_text(encoding="utf-8"), (fn, mk)
for prof in (CK.OFFICIAL_PROFILE, CK.so101_profile()):
    assert not prof.lenient_pick and prof.wide_step_deg == 15 and prof.plan_time_budget_s is None
    assert not prof.remember_angles
rp = RT.real_profile(CK)
assert rp.lenient_pick and rp.wide_step_deg == 30 and rp.plan_time_budget_s == 10.0 and rp.remember_angles
assert "task/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
print("[Z1] V3.68; frozen modules unchanged; simulation profiles unchanged; real profile lenient / fast; task/ ignored")

gl_ok = False
if HAVE_MJ:
    try:
        CK.TwinIO(gripper="so101").render_fixed_camera()
        gl_ok = True
    except Exception as exc:
        gl_reason = str(exc)[:80]
if gl_ok:
    import v365_real_twin as RTW  # noqa: E402

    def tilt_patch(io, deg):
        """perception reports the cups `deg` tilted (reliable) - like a 6-10 deg real measurement"""
        orig = io.perceive

        def p(tag):
            res = orig(tag)
            for c in res.get("cups") or []:
                if c.get("kind") in ("cup", "stack"):
                    c["tilt_deg"], c["tilt_reliable"] = float(deg), True
            return res
        io.perceive = p

    # ---------------------------------------------------------------- Z2
    CK._LAST_ANGLES.clear()
    log2 = []
    io, tw2, arm2, grip2 = RTW.make(log=log2.append)
    tilt_patch(io, 8.0)
    out = RT.run_execute(io, "cup_1", "cup_2")
    a_, b_ = tw2.cup_pose("cup_1"), tw2.cup_pose("cup_2")
    st2 = CK.SC.cup_task_state(a_[0], a_[1], b_[0], b_[1], origin="rim")
    assert st2["nested_full"], [(r_.get("tool"), r_.get("reason")) for r_ in out["results"] if not r_.get("ok")]
    pick = next(r_ for r_ in out["results"] if r_.get("tool") == "pick_cup")
    assert pick["ok"], pick
    # clamp reports no hold although the cup was lifted -> both checks must agree
    io, tw3, arm3, grip3 = RTW.make()
    _c = grip3.clamp
    grip3.clamp = lambda pct, squeeze_deg=None: dict(_c(pct), held=False)
    sk3 = RT.make_skills(io)
    assert sk3.run_tool("observe_scene", {})["ok"] and sk3.run_tool("plan_cup_stack", {"source_cup": "cup_1",
                                                                                         "target_cup": "cup_2"})["ok"]
    r3 = sk3.run_tool("pick_cup", {"source_cup": "cup_1"})
    assert not r3["ok"] and r3["reason"] == "NO_TWO_FINGER_CONTACT" and "did not stop" in r3["physical_verification"]["detail"], r3
    # a fresh view of a cup pushed far away stops the pick before the clamp
    io, tw4, arm4, grip4 = RTW.make()
    sk4 = RT.make_skills(io)
    sk4.run_tool("observe_scene", {})
    sk4.run_tool("plan_cup_stack", {"source_cup": "cup_1", "target_cup": "cup_2"})
    orig4 = io.perceive

    def far(tag):
        res = orig4(tag)
        if "after the descent" in tag:
            for c in res["cups"]:
                if c["base_xyz_m"][1] < 0:
                    c["kind"], c["base_xyz_m"] = "cup", [c["base_xyz_m"][0] - 0.04, c["base_xyz_m"][1], c["base_xyz_m"][2]]
        return res
    io.perceive = far
    r4 = sk4.run_tool("pick_cup", {"source_cup": "cup_1"})
    assert not r4["ok"] and r4["reason"] == "CUP_PUSHED_DURING_DESCENT" and "moved" in r4["physical_verification"]["detail"]
    assert not grip4.clamps
    print("[Z2] lenient pick: 8 deg tilt reading -> clamped, lifted, nested; clamp without hold -> NO_TWO_FINGER_CONTACT; "
          "cup pushed 40 mm (fresh view) -> stop before the clamp")

    # ---------------------------------------------------------------- Z3
    CK._LAST_ANGLES.clear()
    log3 = []
    io, tw5, _a, _g = RTW.make(log=log3.append)
    sk5 = RT.make_skills(io)
    sk5.run_tool("observe_scene", {})
    t = time.time()
    p1 = sk5.run_tool("plan_cup_stack", {"source_cup": "cup_1", "target_cup": "cup_2"})
    t1 = time.time() - t
    t = time.time()
    p2 = sk5.run_tool("plan_cup_stack", {"source_cup": "cup_1", "target_cup": "cup_2"})
    t2 = time.time() - t
    assert p1["ok"] and p2["ok"] and any("started from the last good rim angles" in l for l in log3)
    k_far = dict(sk5.k, cup_grasp_rim_angle_deg=180.0, cup_place_rim_angle_deg=180.0)
    s_, d_ = sk5._cup("cup_1"), sk5._cup("cup_2")
    t = time.time()
    pw = CK.plan_cup_stack(k_far, s_["rim"], s_["quat"], d_["rim"], d_["quat"], q_start=io.current_q(),
                           profile=io.prof)
    tw_ = time.time() - t
    assert pw["ok"] and pw["min_joint_margin_deg"] >= CK.MIN_MARGIN_RELAXED_DEG and tw_ < 25.0, (pw.get("reason"), tw_)
    print(f"[Z3] plan {t1:.1f} s -> remembered angles {t2:.1f} s; wide search (30 deg, 10 s budget) {tw_:.1f} s, "
          f"margin {pw['min_joint_margin_deg']} deg")

    # ---------------------------------------------------------------- Z4
    log4 = []
    io, tw6, _a, _g = RTW.make(log=log4.append)
    snaps0 = set(os.listdir(TMP / "cameracalib" / "real_cup_logs")) if (TMP / "cameracalib" / "real_cup_logs").is_dir() else set()
    res = io.perceive("t")
    assert set(res["timing"]) >= {"yolo_s", "fuse_s", "locate_s", "cup_s"} and "YOLOE" in YC.timing_text(res)
    assert any("[V3.66 REAL CUP YOLOE] t |" in l and "（帧" in l or "（YOLOE" in l for l in log4), log4[-1]
    snaps1 = set(os.listdir(TMP / "cameracalib" / "real_cup_logs")) if (TMP / "cameracalib" / "real_cup_logs").is_dir() else set()
    assert snaps1 == snaps0
    assert YC.FIT_MAX_PTS == 1500
    print(f"[Z4] perception timings {res['timing']}; no snapshot on a normal perception")

    # ---------------------------------------------------------------- Z5
    CK._LAST_ANGLES.clear()
    log5 = []
    io, tw7, _a, _g = RTW.make(log=log5.append)
    snap_fn = lambda: (tw7.render_fixed_camera()[0], None)
    rec = TR.TaskRecorder("Stack cup_1 into cup_2", "execute", snap_fn=snap_fn, log=log5.append, meta={"setup": "x"})
    t0 = time.time()
    out = RT.run_execute(io, "cup_1", "cup_2", recorder=rec)
    text, ok = TR.task_summary("Stack cup_1 into cup_2", "execute", rec.steps, out["ok"], time.time() - t0,
                               setup="skill [CS]", cups="#1 cup · #2 cup", folder=rec.folder,
                               stats={"n_perc": len(io.perc_times), "perc_avg_s": 1.0, "n_motion": len(io.motion_log),
                                      "motion_s": 10.0})
    rec.finish(text, ok)
    assert out["ok"] and ok and text.startswith("[REAL TASK SUMMARY GREEN]"), text[:200]
    for k_ in ("STATUS: SUCCESS", "TASK:", "SETUP:", "CUPS:", "TOOLS:", "LAST FAILURE: none", "TIME:", "RECORD:"):
        assert k_ in text, k_
    files = sorted(os.listdir(rec.folder))
    n = len(rec.steps)
    assert n >= 5 and sum(f.endswith("_before_" + "") or "_before_" in f for f in files) == n
    assert sum("_after_" in f for f in files) == n and "summary.txt" in files and "task.json" in files
    assert len((Path(rec.folder) / "steps.jsonl").read_text(encoding="utf-8").strip().splitlines()) == n
    assert str(TMP / "task") in rec.folder
    lines = [l for l in log5 if "[V3.68 STEP " in l]
    assert len(lines) == n and any("pick_cup" in l and "夹紧停在缝" in l for l in lines), lines
    tj = json.loads((Path(rec.folder) / "task.json").read_text(encoding="utf-8"))
    assert tj["ok"] is True and tj["duration_s"] > 0
    print(f"[Z5] task record: {n} tool calls, {2 * n} images, steps.jsonl, summary.txt; step lines + GREEN summary")

    # ---------------------------------------------------------------- Z6
    rows = TR.roll_sweep(sk5.k, (s_["rim"], s_["quat"], d_["rim"], d_["quat"]), io.current_q(), {},
                         rolls=[-45, 45, 135], budget_s=3.0)
    assert [r_["roll"] for r_ in rows] == [-45.0, 45.0, 135.0] and rows[0]["ok"], rows
    print(f"[Z6b] roll sweep rows: {[(r_['roll'], r_['ok'], r_.get('margin')) for r_ in rows]}")
else:
    print(f"[Z2-Z5] skipped: no twin camera ({gl_reason if HAVE_MJ else 'MuJoCo not installed'})")

# ------------------------------------------------------------------ Z6 fingertips (no twin needed)
T_tcp = CK.FastArm().fk(np.zeros(7))
for a in (0.0, 10.0, 23.8):
    fx, mv = TR.fingertip_points(T_tcp, {}, a)
    gap = float(np.linalg.norm(fx - mv)) * 1000.0
    exp = GR.gap_mm_for_open_deg(a) + 2 * 0.0                      # pad faces vs pad centres: same x offset model
    assert abs(gap - (exp + 2 * (abs(-0.007) - 0.005) * 1000.0)) < 1.5, (a, gap, exp)
fx0, _ = TR.fingertip_points(T_tcp, {"roll_deg": -45.0}, 0.0)
fx1, _ = TR.fingertip_points(T_tcp, {"roll_deg": 45.0}, 0.0)
assert float(np.linalg.norm(fx0 - fx1)) > 0.005          # 7 mm off the jaw axis, rotated 90 deg
print("[Z6] fingertip model: tip distance = jaw gap for the opening; roll moves the tips")

# ------------------------------------------------------------------ Z7
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
        return "green"

    def supports_joint_path(self):
        return True

    def actual_arm_deg(self, m):
        return [0.0, 20.0, 0.0, 60.0, 0.0, 30.0, 0.0]

    def move_busy(self):
        return False


class _Cam:
    src = wrist_src = worker = result = calib = None


logs7 = []
tab = RT.RealCupTaskTabUI(mtk, mttk, mock.MagicMock(), _Pipe(), types.SimpleNamespace(model={"joints": [1]}), _Cam(),
                          log=logs7.append, visible_fn=lambda: False)
assert tab.v_tips.get() is True
pts = []
cam_T = np.array([[1.0, 0, 0, -0.30], [0, -1.0, 0, 0.0], [0, 0, -1.0, 1.0], [0, 0, 0, 1.0]])
Tcb = np.linalg.inv(cam_T)


def proj(p):
    pc = Tcb[:3, :3] @ np.asarray(p) + Tcb[:3, 3]
    pts.append(p)
    return (900 * pc[0] / pc[2] + 640) * 0.4, (900 * pc[1] / pc[2] + 360) * 0.4
tab.cv = mock.MagicMock()
tab._draw_tips(proj)
assert len(pts) == 2 and tab.cv.create_line.called
texts = [c_.kwargs.get("text") for c_ in tab.cv.create_text.call_args_list]
assert "F 固定指" in texts and "M 动指" in texts, texts
tab.v_tips.set(False)
tab._save()
saved = json.loads((Path(os.environ["NERO_CALIB_DIR"]) / "real_cup_task.json").read_text(encoding="utf-8"))
assert saved["show_tips"] is False
tab.on_roll_sweep()
tab.poll()
assert any("偏转角扫描" in l or "ROLL SWEEP" in l for l in logs7) and any("不能开始" in l for l in logs7), logs7[-3:]


class _RealDet(YP.YoloEDetector):
    pass


md = TMP / "models"
md.mkdir(exist_ok=True)
(md / "yoloe-26s-seg.pt").write_bytes(b"x")
cam2 = types.SimpleNamespace(det_yolo=_RealDet(), v_ydir=types.SimpleNamespace(get=lambda: str(md)),
                             v_ymodel=types.SimpleNamespace(get=lambda: "yoloe-26s-seg.pt"),
                             v_yconf=types.SimpleNamespace(get=lambda: "0.25"), src=None, wrist_src=None)
tab.cam_tab = cam2
c1 = tab.yolo_cfg()
assert c1["detector"] is not cam2.det_yolo and isinstance(c1["detector"], YP.YoloEDetector)
assert tab.yolo_cfg()["detector"] is c1["detector"]
print("[Z7] tab: fingertip overlay F / M + jaw line, show_tips saved, roll sweep refused without camera, own YOLOE "
      "detector instance (same model / confidence)")

SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
assert not (ROOT / "task").exists() or not any((ROOT / "task").iterdir())
print("ALL V3.68 TESTS PASSED")
