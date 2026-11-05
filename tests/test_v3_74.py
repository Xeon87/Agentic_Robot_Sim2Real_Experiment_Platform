#!/usr/bin/env python3
"""V3.74 offline tests (real cup perception: geometry measures only + LLM vision review):
G1 Versions V3.74 + markers; frozen modules unchanged.
G2 The user's failure: two identical cups measured 98 mm and 116 mm (V3.72 real run) - with the V3.66 height classes the
   98 mm one was "unknown" and observe_scene failed (CUP_NOT_PERCEIVED); now every standing YOLOE cup is a "cup" and
   both cups are found.  classify_by_height=True still gives the V3.66 classes (reference).
G3 "Lifted" needs the gripper down at the cup (gripper_reaches_down_to): an arm parked next to a cup above it does not
   make the cup "lifted".
G4 Vision review: annotated JPEG with the outlines, prompt with the measured numbers, JSON answer parsed; the answer
   sets the kinds (nested -> stack, not a cup -> dropped, in the gripper -> lifted, edge stays edge); disagreements
   listed; the same layout is not asked twice (cache); an API error -> one WARN, geometry used.
G5 Request format: OpenAI Responses (input_text + input_image data URL) and chat providers (text + image_url).
G6 RealIO: the review runs only at the key perceptions (observe / 识别杯子 / locate / check / verify, not during the
   pick motions); a "[V3.74 VISION]" log line; table offset vs the fence z_table -> one WARN above 5 mm, value kept.
G7 Tab: 「LLM 看图复核」 checkbox (default on, saved); the ②/③ confirmation names a table offset > 10 mm.
G8 Twin (needs MuJoCo + OpenGL): two standing cups -> two "cup"; the nested pair -> "cup" by default and "stack" with
   classify_by_height=True; release folder unchanged.
"""
import hashlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v374_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
os.environ["NERO_V368_TASK_DIR"] = str(TMP / "task")
for k in ("NERO_SCENE", "NERO_V343_ACTIVE_SCENE"):
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
import NERO_V374_Vision_Review as VR        # noqa: E402

try:
    import mujoco  # noqa: F401
    HAVE_MJ = True
except Exception:
    HAVE_MJ = False

# ------------------------------------------------------------------ G1
for m in (CON, RR, YP, W49, RT, YC, VR):
    assert m.VERSION in ("V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81"), (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
for fn, h in (("NERO_V2_7_1_Unified_Backend.py", "5be2136f84ec4ccdfe6f8bc8d8998410ff9b3b827af4d571ea44373dc427527e"),
              ("NERO_V2_7_1_Agent.py", "3e52d05ea84d94c8359d5875fa0fee8c2cc42d612b52e59109e1166912439696"),
              ("NERO_V325_Camera.py", "013fc8c308092f863c7ccb1a37598c80536deed2cfbcef2b0daccc417da2806d"),
              ("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
for fn, mk in (("NERO_V366_Real_Cup_Yolo.py", "V3.74 GEOMETRY MEASURES ONLY"),
               ("NERO_V366_Real_Cup_Yolo.py", "V3.74 LIFTED NEEDS THE GRIPPER AT THE CUP"),
               ("NERO_V374_Vision_Review.py", VR.MARKER), ("NERO_V365_Real_Cup_Task.py", "V3.74 TABLE OFFSET CHECK"),
               ("NERO_V365_Real_Cup_Task.py", "V3.74 LLM VISION REVIEW")):
    assert mk in (ROOT / fn).read_text(encoding="utf-8"), (fn, mk)
print("[G1] V3.74; frozen modules unchanged; markers")

# ------------------------------------------------------------------ G2 (skill level, the user's numbers)


def cup_e(x, y, h, kind="cup"):
    return {"kind": kind, "base_xyz_m": [x, y, 0.0], "height_mm": h, "tilt_deg": 0.0}


g = CK.SC.cup_geometry()
assert YC.classify(98.0, g, g["nested_dz"]) == "unknown"              # what V3.72 did with the 98 mm cup
logs2 = []
res_user = {"cups": [cup_e(-0.381, -0.087, 98.0), cup_e(-0.409, 0.139, 117.0)]}
io2 = types.SimpleNamespace(pose_mode=lambda: "camera", emit=logs2.append, gripper_profile=lambda: RT.real_profile(CK),
                            current_q=lambda: [0.0] * 7, perceive=lambda tag: dict(res_user, dets=res_user["cups"]),
                            gripper_state=lambda: {"hold_active": False}, tcp_pose=lambda: ([-0.3, 0, 0.3], [1, 0, 0, 0]))
sk2 = CK.CupSkills(io2, *CK.cup_knobs())
sc2 = sk2.scene()
assert set(sc2["objects"]) == {"cup_1", "cup_2"}, sc2["objects"].keys()
print("[G2] cups measured 98 mm and 117 mm (the user's run) -> both found as cups (V3.66 classes: 98 mm = unknown)")

# ------------------------------------------------------------------ G3
box_above = [(np.array([-0.38, -0.09, 0.30]), np.eye(3), np.array([0.04, 0.03, 0.06]))]      # parked 18 cm higher
box_at = [(np.array([-0.38, -0.09, 0.13]), np.eye(3), np.array([0.04, 0.03, 0.06]))]          # jaws down at the cup
assert not YC.gripper_reaches_down_to(box_above, 0.11) and YC.gripper_reaches_down_to(box_at, 0.11)
assert YC.gripper_reaches_down_to(None, 0.11)
print("[G3] lifted needs the gripper down at the cup: parked above -> no, at the cup -> yes")

# ------------------------------------------------------------------ G4
rgb = np.zeros((360, 640, 3), np.uint8) + 90
dets = [{"det_id": "#1", "kind": "cup", "conf": 0.86, "base_xyz_m": [-0.38, -0.09, 0.0], "height_mm": 98.0,
         "contour": [[100, 100], [160, 100], [160, 200], [100, 200]]},
        {"det_id": "#2", "kind": "cup", "conf": 0.61, "base_xyz_m": [-0.41, 0.14, 0.0], "height_mm": 140.0,
         "contour": [[300, 90], [370, 90], [370, 210], [300, 210]]},
        {"det_id": "#3", "kind": "cup", "conf": 0.40, "base_xyz_m": [-0.30, 0.02, 0.0], "height_mm": 60.0,
         "contour": [[450, 150], [500, 150], [500, 200], [450, 200]]},
        {"det_id": "#4", "kind": "edge", "conf": 0.5, "base_xyz_m": [-0.50, 0.30, 0.0], "height_mm": None,
         "contour": [[600, 0], [639, 0], [639, 40], [600, 40]]},
        {"det_id": "#5", "kind": "cup", "conf": 0.5, "base_xyz_m": [-0.33, -0.20, 0.0], "height_mm": 110.0,
         "contour": [[200, 250], [250, 250], [250, 330], [200, 330]]}]
sent = []
answer = {"objects": [{"id": "#1", "what": "cup", "upright": True, "in_gripper": False, "note": "red cup"},
                      {"id": "#2", "what": "nested_cups", "upright": True, "in_gripper": False, "note": "two cups"},
                      {"id": "#3", "what": "not_a_cup", "upright": True, "in_gripper": False, "note": "a mug lid"},
                      {"id": "#4", "what": "cup", "upright": True, "in_gripper": False},
                      {"id": "#5", "what": "cup", "upright": True, "in_gripper": True}], "scene": "two cups, one lid"}


def fake_post(provider, model, key, prompt, jpeg):
    sent.append({"provider": provider, "model": model, "prompt": prompt, "jpeg": jpeg})
    return "Sure:\n```json\n" + json.dumps(answer) + "\n```"


logs4 = []
rv = VR.VisionReviewer(lambda: {"provider": "openai", "model": "gpt-vision-test", "api_key": "k"}, log=logs4.append,
                       post=fake_post)
r1 = rv.review(rgb, dets, "observe_scene")
assert r1["ok"] and len(sent) == 1 and sent[0]["jpeg"][:2] == b"\xff\xd8"
assert "#1: detector conf 0.86" in sent[0]["prompt"] and "height 98 mm" in sent[0]["prompt"]
assert "nested_cups" in sent[0]["prompt"] and "JSON only" in sent[0]["prompt"]
res4 = {"dets": [dict(d) for d in dets]}
notes = VR.apply_review(res4, r1)
kinds = {d["det_id"]: d["kind"] for d in res4["dets"]}
assert kinds == {"#1": "cup", "#2": "stack", "#3": "not_cup", "#4": "edge", "#5": "lifted"}, kinds
assert [c["det_id"] for c in res4["cups"]] == ["#1", "#2", "#4", "#5"]
assert any("#2" in n and "nested_cups" in n for n in notes) and any("#3" in n for n in notes)
r2 = rv.review(rgb, dets, "check cup_1 in cup_2")
assert r2.get("cached") and len(sent) == 1                                     # same layout: no second request


def bad_post(*a, **k):
    raise RuntimeError("OpenAI HTTP 400: model does not support image input")


rv_bad = VR.VisionReviewer(lambda: {"provider": "openai", "model": "text-only", "api_key": "k"}, log=logs4.append,
                           post=bad_post)
assert not rv_bad.review(rgb, dets, "x")["ok"] and not rv_bad.review(rgb, dets[:2], "y")["ok"]
assert sum("[WARN] [V3.74 VISION]" in l for l in logs4) == 1 and "下一步" in logs4[-1]
assert VR.parse_reply("no json here") is None
print(f"[G4] review: kinds {kinds}; {len(notes)} disagreements logged; cached repeat; API error -> 1 WARN")

# ------------------------------------------------------------------ G5
import urllib.request  # noqa: E402
cap = []


class _Resp:
    def __init__(self, obj):
        self.obj = obj

    def read(self):
        return json.dumps(self.obj).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_urlopen(req, timeout=None):
    body = json.loads(req.data.decode())
    cap.append((req.full_url, body))
    if req.full_url.endswith("/responses"):
        return _Resp({"output": [{"type": "message", "content": [{"type": "output_text", "text": '{"objects": []}'}]}]})
    return _Resp({"choices": [{"message": {"content": '{"objects": []}'}}]})


_orig = urllib.request.urlopen
urllib.request.urlopen = fake_urlopen
try:
    t1 = VR.post_vision("openai", "gpt-x", "k1", "hello", b"\xff\xd8abc")
    t2 = VR.post_vision("kimi", "kimi-k3", "k2", "hello", b"\xff\xd8abc")
finally:
    urllib.request.urlopen = _orig
(u1, b1), (u2, b2) = cap
c1 = b1["input"][0]["content"]
assert u1.endswith("/responses") and c1[0]["type"] == "input_text" and c1[1]["type"] == "input_image"
assert c1[1]["image_url"].startswith("data:image/jpeg;base64,")
c2 = b2["messages"][0]["content"]
assert u2.endswith("/chat/completions") and c2[1]["type"] == "image_url" and c2[1]["image_url"]["url"].startswith("data:")
assert t1 == '{"objects": []}' and t2 == '{"objects": []}'
print("[G5] request format: OpenAI input_image / chat image_url, replies parsed")

# ------------------------------------------------------------------ G6
logs6 = []


class _Arm:
    model = {"joints": [f"joint{i}" for i in range(1, 8)]}

    def q(self):
        return [0.0] * 7

    def exec_path(self, *a, **k):
        return {"ok": True}


base_res = {"cups": [dict(d) for d in dets[:2]], "dets": [dict(d) for d in dets[:2]], "table_z_m": -0.012, "n_2d": 2,
            "timing": {}, "device": "GPU"}
_orig_p = YC.perceive_cups_yolo
YC.perceive_cups_yolo = lambda cam, **k: json.loads(json.dumps(base_res))
try:
    io6 = RT.RealIO(_Arm(), None, lambda: {"rgb": rgb, "z_table": 0.0}, log=logs6.append, motion_allowed=False)
    sent.clear()
    io6.vision = VR.VisionReviewer(lambda: {"provider": "openai", "model": "gpt-vision-test", "api_key": "k"},
                                   log=logs6.append, post=fake_post)
    r_obs = io6.perceive("observe_scene")
    assert len(sent) == 1 and {d["det_id"]: d["kind"] for d in r_obs["dets"]} == {"#1": "cup", "#2": "stack"}
    assert any(l.startswith("[V3.74 VISION] observe_scene | gpt-vision-test") for l in logs6), logs6
    io6.vision._cache = None
    io6.perceive("pick cup_1")
    assert len(sent) == 1                                                       # no review during the pick motions
    assert io6.table_offset_mm == -12.0
    warns = [l for l in logs6 if "[V3.74 REAL] 识别到的桌面" in l]
    assert len(warns) == 1 and "低 12 mm" in warns[0] and "先别跑 ③" in warns[0], warns
finally:
    YC.perceive_cups_yolo = _orig_p
print("[G6] RealIO: review at observe_scene only (not at 'pick'); table 12 mm low -> one WARN, offset kept")

# ------------------------------------------------------------------ G7
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
assert tab.v_vision.get() is True
texts = [c_.kwargs.get("text") for c_ in mttk.Checkbutton.call_args_list]
assert "LLM 看图复核" in texts
tab.v_vision.set(False)
tab._save()
saved = json.loads((Path(os.environ["NERO_CALIB_DIR"]) / "real_cup_task.json").read_text(encoding="utf-8"))
assert saved["vision_review"] is False
src = (ROOT / "NERO_V365_Real_Cup_Task.py").read_text(encoding="utf-8")
assert "上次识别的桌面和安全区设定差" in src and "last_table_offset_mm" in src
print("[G7] tab: 「LLM 看图复核」 (default on, saved); ②/③ confirmation names a table offset > 10 mm")

# ------------------------------------------------------------------ G8
gl_ok, gl_reason = False, "MuJoCo not installed"
if HAVE_MJ:
    try:
        CK.TwinIO(gripper="so101").render_fixed_camera()
        gl_ok = True
    except Exception as exc:
        gl_reason = str(exc)[:80]
if gl_ok:
    import mujoco as mj  # noqa: E402
    import v365_real_twin as RTW  # noqa: E402
    tw = CK.TwinIO(gripper="so101", pose_mode="camera")
    camfn = RTW.twin_camera(tw)
    k2 = sorted(e["kind"] for e in YC.perceive_cups_yolo(camfn())["cups"])
    assert k2 == ["cup", "cup"], k2
    b1 = mj.mj_name2id(tw.m, mj.mjtObj.mjOBJ_BODY, "obj_cup_1")
    adr = tw.m.jnt_qposadr[tw.m.body_jntadr[b1]]
    r2_ = tw.cup_pose("cup_2")[0]
    tw.d.qpos[adr:adr + 3] = [r2_[0], r2_[1], r2_[2] + 0.0253]
    mj.mj_forward(tw.m, tw.d)
    for _ in range(300):
        mj.mj_step(tw.m, tw.d)
    nested_default = [e["kind"] for e in YC.perceive_cups_yolo(camfn())["cups"]]
    nested_height = [e["kind"] for e in YC.perceive_cups_yolo(camfn(), classify_by_height=True)["cups"]]
    assert nested_default == ["cup"] and nested_height == ["stack"], (nested_default, nested_height)
    print("[G8] twin: two standing cups -> cup, cup; nested pair -> cup (default) / stack (classify_by_height)")
else:
    print(f"[G8] skipped: no twin camera ({gl_reason})")

SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("ALL V3.74 TESTS PASSED")
