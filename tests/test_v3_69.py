#!/usr/bin/env python3
"""V3.69 offline tests (real cup task after the first real stack):
B1  Versions V3.69 + markers; frozen modules unchanged; the simulation's gripper profiles keep clear_view_before_verify
    and camera_decides_grip off; the real profile has both on and a 15 s planning budget.
B2  Clear view before verify (twin): verify_stack first sends the arm back to the start posture, then looks; when the
    stack is still hidden by the gripper it retreats again (higher) and looks once more.
B3  Camera decides the grip (twin): the servo reports no hold but the camera sees the cup leave the table -> pick OK
    with a warning (the first real stack: clamped 2.62 mm vs empty 2.39 mm).
B4  Best-of plan: last-good / narrow / wide candidates within ONE budget from the start; the largest margin wins
    (V3.68 used a 4.9 deg wide plan although the narrow one had 6.1 deg); the wide search is skipped after the budget.
B5  YOLOE device: CUDA available -> "cuda:0" forced with an [OK] log line naming the GPU; not available -> CPU with
    the reason.
B6  "Lifted" needs the cup next to the gripper: a cup without depth near its bottom far from the gripper is not lifted.
B7  Speed multiplier: 1X = the fields; joint speeds scale up to 72 deg/s, straight segments near the cups up to
    60 mm/s, settle / dwell shrink; the tab has the selector left of 「▶ 运行」 (saved).
B8  Task video: TaskVideo writes task_video.mp4 (time stamp / step header) into the task record folder at the set
    frame rate; the tab has 「任务录像」 (saved).
"""
import hashlib
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v369_"))
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
import NERO_V368_Real_Task_Record as TR     # noqa: E402

try:
    import mujoco  # noqa: F401
    HAVE_MJ = True
except Exception:
    HAVE_MJ = False

# ------------------------------------------------------------------ B1
for m in (CON, RR, YP, W49, RT, YC, TR):
    assert m.VERSION in ("V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8"), (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
for fn, h in (("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a"),
              ("NERO_V2_7_1_Agent.py", "3e52d05ea84d94c8359d5875fa0fee8c2cc42d612b52e59109e1166912439696"),
              ("NERO_V325_Camera.py", "82f2416d73671e7018d09d0cd72c17186154934a315b47c0d19fdb0427afd931"),
              ("NERO_V2_7_1_Unified_Backend.py", "5be2136f84ec4ccdfe6f8bc8d8998410ff9b3b827af4d571ea44373dc427527e"),
              ("NERO_V363_Cup_Perception.py", "b9633693657127fccc101bd902ad38e6bbc014fc907c41bd54227c1711a732c5")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
for fn, mk in (("NERO_V362_Cup_Skills.py", "V3.69 CLEAR VIEW VERIFY"), ("NERO_V362_Cup_Skills.py", "V3.69 CAMERA DECIDES GRIP"),
               ("NERO_V362_Cup_Skills.py", "V3.69 BEST-OF PLAN"), ("NERO_V365_Real_Cup_Task.py", "V3.69 SPEED MULTIPLIER"),
               ("NERO_V365_Real_Cup_Task.py", "V3.69 YOLOE DEVICE"), ("NERO_V366_Real_Cup_Yolo.py", "V3.69 LIFTED NEAR GRIPPER"),
               ("NERO_V368_Real_Task_Record.py", TR.MARKER_V369)):
    assert mk in (ROOT / fn).read_text(encoding="utf-8"), (fn, mk)
for prof in (CK.OFFICIAL_PROFILE, CK.so101_profile()):
    assert not prof.clear_view_before_verify and not prof.camera_decides_grip and prof.plan_time_budget_s is None
rp = RT.real_profile(CK)
assert rp.clear_view_before_verify and rp.camera_decides_grip and rp.plan_time_budget_s == 15.0
print("[B1] V3.69; frozen modules unchanged; simulation profiles unchanged; real profile clear view / camera grip / 15 s")

# ------------------------------------------------------------------ B4 (no twin needed)
calls = []


def fake_plan(knobs, *a, profile=None, allow_wide=True, deadline=None, _wide=False, **k):
    calls.append({"wide": _wide, "deadline": deadline, "grasp": knobs["cup_grasp_rim_angle_deg"]})
    m = 4.9 if _wide else (6.1 if knobs["cup_grasp_rim_angle_deg"] == 40.0 else 5.5)
    return {"ok": True, "min_joint_margin_deg": m, "grasp_rim_angle_rel_deg": 0.0, "place_rim_angle_rel_deg": 0.0,
            "margin_ok": False, "grasp_margin_deg": m, "place_margin_deg": m, "grasp_binding": "J7",
            "place_binding": "J7", "_wps": {}, "_q": {}, "_qpath": {}, "tag": "wide" if _wide else "narrow"}


io_ = types.SimpleNamespace(pose_mode=lambda: "truth", emit=lambda s: logs4.append(s), gripper_profile=lambda: rp,
                            current_q=lambda: [0.0] * 7)
logs4 = []
sk = CK.CupSkills(io_, *CK.cup_knobs())
_orig_plan = CK.plan_cup_stack
CK.plan_cup_stack = fake_plan
try:
    s_ = {"rim": np.array([-0.35, -0.13, 0.12]), "quat": [1, 0, 0, 0]}
    t_ = {"rim": np.array([-0.35, 0.13, 0.12]), "quat": [1, 0, 0, 0]}
    p = sk._plan_best_of(s_, t_, [0.0] * 7, (25.0, 50.0))
    assert p["tag"] == "narrow" and p["min_joint_margin_deg"] == 6.1, p
    assert [c["wide"] for c in calls] == [False, False, True] and len({c["deadline"] for c in calls}) == 1
    assert any("[V3.69 CUP PLAN]" in l and "-> narrow (6.1°" in l for l in logs4), logs4
    calls.clear()
    rp.plan_time_budget_s = 0.0
    p2 = sk._plan_best_of(s_, t_, [0.0] * 7, None)
    assert [c["wide"] for c in calls] == [False], calls            # budget over -> no wide search
    rp.plan_time_budget_s = 15.0
finally:
    CK.plan_cup_stack = _orig_plan
print("[B4] best-of plan: last 5.5 / narrow 6.1 / wide 4.9 -> narrow; one deadline; no wide search after the budget")

# ------------------------------------------------------------------ B5
logs5 = []
fake_torch = types.SimpleNamespace(__version__="2.14.1", cuda=types.SimpleNamespace(is_available=lambda: True,
                                                                                   get_device_name=lambda i: "RTX 4060"))
sys.modules["torch"], _old_torch = fake_torch, sys.modules.get("torch")
try:
    assert RT.yolo_device(logs5.append) == "cuda:0" and "RTX 4060" in logs5[-1] and "[OK]" in logs5[-1]
    fake_torch.cuda.is_available = lambda: False
    assert RT.yolo_device(logs5.append) is None and "CPU" in logs5[-1] and "is_available() = False" in logs5[-1]
finally:
    if _old_torch is not None:
        sys.modules["torch"] = _old_torch
    else:
        sys.modules.pop("torch", None)
print("[B5] YOLOE device: CUDA -> cuda:0 (GPU named in the log); no CUDA -> CPU with the reason")

# ------------------------------------------------------------------ B7
base = dict(RT.DEFAULT_CFG)
for mult, jv, lv in ((1, 12.0, 20.0), (2, 24.0, 40.0), (3, 36.0, 60.0), (5, 60.0, 60.0), (10, 72.0, 60.0)):
    c, d = RT.apply_speed(base, mult)
    assert (c["joint_vmax_deg_s"], c["line_vmax_mm_s"]) == (jv, lv), (mult, c)
    assert c["settle_s"] <= base["settle_s"] and c["hover_dwell_s"] >= 1.0 and f"{mult}X" in d
assert "上限" in RT.apply_speed(base, 10)[1] and "上限" not in RT.apply_speed(base, 2)[1]
assert RT.SPEED_CHOICES == ("1X", "2X", "3X", "4X", "5X", "10X")
print("[B7] speed: 1X 12/20, 2X 24/40, 3X 36/60, 5X 60/60, 10X 72/60 (deg/s / mm/s, caps), settle / dwell shrink")

# ------------------------------------------------------------------ B8
import NERO_V325_Camera as CAM  # noqa: E402
cv2 = CAM._import_cv2()
rec = TR.TaskRecorder("video test", "plan", snap_fn=None, log=lambda s: None)
frame = np.zeros((360, 640, 3), np.uint8) + 60
vid = TR.TaskVideo(rec, lambda: (frame, frame), fps=10.0, width=640)
vid.start()
rec.status = {"step": 1, "tool": "observe_scene()", "phase": "done", "result": "OK", "stage": "OBSERVED", "held": None}
time.sleep(1.2)
vid.stop()
cap = cv2.VideoCapture(vid.path)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ok, img = cap.read()
cap.release()
assert os.path.isfile(vid.path) and vid.frames >= 6 and n >= 6 and ok and img.shape[1] == 640, (vid.frames, n)
assert str(TMP / "task") in vid.path
print(f"[B8] task video: {vid.frames} frames at 10 fps in {os.path.basename(rec.folder)}/task_video.mp4")

# ------------------------------------------------------------------ B2 / B3 / B6 need the twin camera
gl_ok = False
if HAVE_MJ:
    try:
        CK.TwinIO(gripper="so101").render_fixed_camera()
        gl_ok = True
    except Exception as exc:
        gl_reason = str(exc)[:80]
if gl_ok:
    import mujoco as mj  # noqa: E402
    import v365_real_twin as RTW  # noqa: E402
    CK._LAST_ANGLES.clear()
    log2 = []
    io, tw2, arm2, grip2 = RTW.make(log=log2.append)
    hidden = {"n": 0}
    orig = io.perceive

    def p_(tag):
        res = orig(tag)
        if tag.startswith("check cup_1 in cup_2") and cv_calls and hidden["n"] < 1:   # first verify look: hidden
            hidden["n"] += 1
            for c in res["cups"]:
                c["kind"] = "occluded"
        return res
    io.perceive = p_
    cv_calls = []
    _cv = io.clear_view
    io.clear_view = lambda a, l: (cv_calls.append(a), _cv(a, l))[1]
    out = RT.run_execute(io, "cup_1", "cup_2")
    ver = [r_ for r_ in out["results"] if r_.get("tool") == "verify_stack"][-1]
    assert ver["ok"] and ver["physical_verification"].get("view") == "second clear-view retreat", ver
    assert cv_calls == [0, 1], cv_calls
    labels = [c_["label"] for c_ in arm2.calls]
    assert any("clear view" in l for l in labels), labels[-6:]
    print("[B2] verify: back to the start posture, still hidden -> higher, look again -> verified")

    io, tw3, arm3, grip3 = RTW.make(log=log2.append)
    _c = grip3.clamp
    grip3.clamp = lambda pct, squeeze_deg=None: dict(_c(pct), held=False)
    sk3 = RT.make_skills(io)
    sk3.run_tool("observe_scene", {})
    sk3.run_tool("plan_cup_stack", {"source_cup": "cup_1", "target_cup": "cup_2"})
    r3 = sk3.run_tool("pick_cup", {"source_cup": "cup_1"})
    assert r3["ok"] and "camera confirms" in r3["physical_verification"]["clamp_note"], r3
    assert any("以相机为准" in l for l in log2)
    print("[B3] servo: no hold, camera: cup left the table -> pick OK + warning")

    tw = CK.TwinIO(gripper="so101", pose_mode="camera")
    camfn = RTW.twin_camera(tw)
    bid = mj.mj_name2id(tw.m, mj.mjtObj.mjOBJ_BODY, "obj_cup_1")
    adr = tw.m.jnt_qposadr[tw.m.body_jntadr[bid]]
    r1 = tw.cup_pose("cup_1")[0]
    tw.d.qpos[adr:adr + 3] = [r1[0], r1[1], r1[2] + 0.06]
    mj.mj_forward(tw.m, tw.d)
    # V3.74 LIFTED NEEDS THE GRIPPER AT THE CUP: the stand-in gripper sits at the raised rim (it was 6 cm above it)
    near = [(np.array([r1[0], r1[1], r1[2] + 0.07]), np.eye(3), np.array([0.02, 0.02, 0.02]))]
    far = [(np.array([-0.10, 0.30, 0.40]), np.eye(3), np.array([0.02, 0.02, 0.02]))]
    k_near = sorted(e["kind"] for e in YC.perceive_cups_yolo(camfn(), robot_boxes=near)["dets"])
    res_far = YC.perceive_cups_yolo(camfn(), robot_boxes=far)
    k_far = sorted(e["kind"] for e in res_far["dets"])
    assert "lifted" in k_near and "lifted" not in k_far, (k_near, k_far)
    assert any("不在夹爪旁" in w for e in res_far["dets"] for w in e.get("warnings") or [])
    print(f"[B6] lifted only next to the gripper: near {k_near}, far {k_far}")
else:
    print(f"[B2/B3/B6] skipped: no twin camera ({gl_reason if HAVE_MJ else 'MuJoCo not installed'})")

# ------------------------------------------------------------------ B7/B8 tab (mocked Tk)
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
assert tab.v_speed.get() == "1X" and tab.v_video.get() is False
tab.v_speed.set("3X")
tab.v_video.set(True)
tab._save()
saved = json.loads((Path(os.environ["NERO_CALIB_DIR"]) / "real_cup_task.json").read_text(encoding="utf-8"))
assert saved["speed_mult"] == "3X" and saved["record_video"] is True
vals = [c_.kwargs.get("values") for c_ in mttk.Combobox.call_args_list]
assert RT.SPEED_CHOICES in vals
texts = [c_.kwargs.get("text") for c_ in mttk.Checkbutton.call_args_list]
assert "任务录像" in texts
print("[B7/B8 tab] speed selector + 任务录像 in the tab, both saved")

SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("ALL V3.69 TESTS PASSED")
