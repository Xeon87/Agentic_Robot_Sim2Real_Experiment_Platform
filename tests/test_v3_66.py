#!/usr/bin/env python3
"""V3.66 offline tests (the real cup task sees cups with the Camera tab's YOLOE pipeline):
X1  Versions V3.66 + markers; the Camera tab (NERO_V325), the YOLOE module's functions, NERO_V308, the V3.63 colour
    perception, the cup skills, scenes, SO-101 model, backend and Agent are unchanged (the real task reuses them).
X2  Same settings as the Camera tab: yolo_cfg_from_tab takes the Camera tab's detector object, model folder / model /
    confidence, and resolves text-prompt vs *-pf exactly like CameraTabUI._resolve_object_query (4 cases); missing
    detector / model / bad confidence -> "error" (YOLOE unavailable).
X3  Same pipeline as the Camera tab: perceive_cups_yolo calls detector.detect(rgb, mode, path, prompts, conf),
    YP.fuse_depth(depths, dets) and YP.locate_objects(dets, fused depth, K, T, table_z, fence_poly, calib_note)
    with the Camera tab's arguments; the V3.63 colour method is never called; camera_from_tab grabs
    YP.FUSE_FRAMES frames through the Camera tab's _grab_frames and passes the fence / calibration note through.
X4  Cup geometry on the MuJoCo twin camera (YOLOE stand-in = segmentation masks): single cups -> kind cup, axis error
    <= 3 mm, silhouette height within 3 mm of the true rim top; D435i rim depth dropout (no depth on the top 2.5 cm of
    the cup) -> the silhouette height does not change (the depth top does); nested pair -> one "stack"
    ~25 mm taller; a cup in the air -> "lifted"; outside the fence -> "outside"; two masks of one cup -> merged.
X5  YOLOE unavailable (cfg error / detector raising YoloUnavailable) -> RealIO.perceive logs "[FAIL] [V3.66 REAL CUP
    YOLOE] YOLOE 不可用" and raises; mode ① ends with PERCEPTION_UNAVAILABLE, no colour fallback, no motion.
X6  RealIO end to end on the twin with the YOLOE pipeline: mode ③ cup_2 -> cup_1 nested and back at the start
    posture (cup_1 -> cup_2 is test_v3_65 W7); the final verification sees the pair as one "stack".
X7  Tab: prompt field (default "cup", saved in NERO_CALIB_DIR), YOLOE description from the Camera tab settings,
    preflight refuses a run when YOLOE is unavailable, overlay_dets gives contours + labels for the main monitor.
"""
import hashlib
import json
import math
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v366_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
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
import NERO_V325_Camera as CAM              # noqa: E402
import NERO_V347_Yolo_Perception as YP      # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49   # noqa: E402
import NERO_V362_Cup_Skills as CK           # noqa: E402
import NERO_V363_Cup_Perception as CP       # noqa: E402
import NERO_V365_Real_Cup_Task as RT        # noqa: E402
import NERO_V366_Real_Cup_Yolo as YC        # noqa: E402

try:
    import mujoco  # noqa: F401
    HAVE_MJ = True
except Exception:
    HAVE_MJ = False

# ------------------------------------------------------------------ X1
for m in (CON, RR, YP, W49, RT, YC):
    assert m.VERSION in ("V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73"), (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
for fn, h in (("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a"),
              ("NERO_V2_7_1_Agent.py", "3e52d05ea84d94c8359d5875fa0fee8c2cc42d612b52e59109e1166912439696"),
              ("NERO_V325_Camera.py", "a58e69b363a5a861b3ad0806ba276c49601fc214ffdfc07ffba9bdc0c2c65797"),
              ("NERO_V2_7_1_Unified_Backend.py", "5be2136f84ec4ccdfe6f8bc8d8998410ff9b3b827af4d571ea44373dc427527e"),
              ("NERO_V331_Fixed_Camera.py", "8314f82cce52016f3dd08bad22ce08dbd9bdb1b263fa6aef23453467a9b3a341"),
              ("NERO_V363_Cup_Perception.py", "b9633693657127fccc101bd902ad38e6bbc014fc907c41bd54227c1711a732c5")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
yp_src = (ROOT / "NERO_V347_Yolo_Perception.py").read_text(encoding="utf-8")
assert "V3.66" not in yp_src.replace('VERSION = "V3.66"', ""), "the YOLOE module itself must not change"
for fn in ("NERO_V362_Cup_Skills.py", "NERO_V343_Scenes.py", "NERO_V364_SO101_Gripper.py", "NERO_V359_Strategy.py",
           "NERO_V365_STS3215_Gripper.py"):
    assert "V3.66" not in (ROOT / fn).read_text(encoding="utf-8"), fn
assert YC.MARKER in (ROOT / "NERO_V366_Real_Cup_Yolo.py").read_text(encoding="utf-8")
assert YC.MARKER in (ROOT / "NERO_V365_Real_Cup_Task.py").read_text(encoding="utf-8")
assert "NERO_V366_Real_Cup_Yolo.py" in (ROOT / "start_nero.sh").read_text(encoding="utf-8")
print("[X1] V3.66; Camera tab, YOLOE functions, colour perception, cup skills, simulation modules unchanged")


# ------------------------------------------------------------------ X2
class _V:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v


class _Det:
    def __init__(self, dets=None, exc=None):
        self.dets, self.exc, self.calls = dets or [], exc, []
        self.info = {"device": "CPU"}

    def detect(self, rgb, mode, path, ref=None, prompts=None, conf=0.25):
        self.calls.append({"mode": mode, "path": path, "prompts": list(prompts), "conf": conf, "shape": rgb.shape})
        if self.exc:
            raise self.exc
        return [d for d in self.dets if d["conf"] >= conf]


def model_dir(name, files):
    d = TMP / name
    d.mkdir(parents=True, exist_ok=True)
    for f in files:
        (d / f).write_bytes(b"x")
    return str(d)


enc = YP.text_encoder_name(str(TMP / YP.DEFAULT_MODEL))
cases = [("both", [YP.DEFAULT_MODEL, enc, YP.DEFAULT_PF_MODEL], YP.DEFAULT_MODEL),        # text with encoder
         ("noenc", [YP.DEFAULT_MODEL, YP.DEFAULT_PF_MODEL], YP.DEFAULT_MODEL),            # no encoder -> pf
         ("pfsel", [YP.DEFAULT_MODEL, enc, YP.DEFAULT_PF_MODEL], YP.DEFAULT_PF_MODEL),    # pf selected
         ("textonly", [YP.DEFAULT_MODEL], YP.DEFAULT_MODEL)]                              # no encoder, no pf
for name, files, sel in cases:
    d = model_dir(name, files)
    stub = types.SimpleNamespace(YP=YP, v_obj_names=_V("cup"), v_yprompt=_V(YP.DEFAULT_TEXT_PROMPTS), v_ydir=_V(d))
    cam_mode, cam_path, cam_prompts = CAM.CameraTabUI._resolve_object_query(stub, YP.find_model(d, sel))
    tab = types.SimpleNamespace(det_yolo=_Det(), v_ydir=_V(d), v_ymodel=_V(sel), v_yconf=_V("0.30"))
    cfg = YC.yolo_cfg_from_tab(tab, "cup")
    assert (cfg["mode"], cfg["path"], cfg["prompts"]) == (cam_mode, cam_path, cam_prompts), (name, cfg, cam_mode)
    assert cfg["detector"] is tab.det_yolo and cfg["conf"] == 0.30 and not cfg.get("error")
assert "error" in YC.yolo_cfg_from_tab(types.SimpleNamespace(), "cup")
tab = types.SimpleNamespace(det_yolo=_Det(), v_ydir=_V(str(TMP / "empty")), v_ymodel=_V(YP.DEFAULT_MODEL),
                            v_yconf=_V("0.30"))
assert "找不到 YOLOE 模型" in YC.yolo_cfg_from_tab(tab, "cup")["error"]
tab.v_ydir, tab.v_yconf = _V(model_dir("both", [])), _V("2")
assert "置信度" in YC.yolo_cfg_from_tab(tab, "cup")["error"]
print("[X2] YOLOE settings = the Camera tab's (detector object, model, confidence, text / pf resolution); "
      "unusable settings -> error")

# ------------------------------------------------------------------ X3 / X4 / X5 / X6 need the twin camera
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

    def set_cup(tw, name, xyz):
        bid = mj.mj_name2id(tw.m, mj.mjtObj.mjOBJ_BODY, f"obj_{name}")
        adr = tw.m.jnt_qposadr[tw.m.body_jntadr[bid]]
        tw.d.qpos[adr:adr + 3] = xyz
        tw.d.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
        tw.d.qvel[:] = 0.0
        mj.mj_forward(tw.m, tw.d)

    tw = CK.TwinIO(gripper="so101", pose_mode="camera")
    camfn = RTW.twin_camera(tw)
    cam = camfn()
    det = _Det(cam["yolo"]["detector"].next)
    calls = {}
    _fuse, _loc, _col = YP.fuse_depth, YP.locate_objects, CP.perceive_cups

    def spy_fuse(depths, dets=None, **kw):
        calls["fuse"] = (len(depths), dets)
        return _fuse(depths, dets, **kw)

    def spy_loc(dets, depth, K, T, table_z=None, fence_poly=None, calib_note=None):
        calls["loc"] = {"dets": dets, "table_z": table_z, "fence": fence_poly, "note": calib_note, "depth": depth}
        return _loc(dets, depth, K, T, table_z=table_z, fence_poly=fence_poly, calib_note=calib_note)

    def no_colour(*a, **k):
        raise AssertionError("the V3.63 colour perception must not be used on the real arm")
    fence = [[-0.8, -0.4], [-0.1, -0.4], [-0.1, 0.4], [-0.8, 0.4]]
    YP.fuse_depth, YP.locate_objects, CP.perceive_cups = spy_fuse, spy_loc, no_colour
    try:
        c2 = dict(cam, depths=[cam["depths"][0], cam["depths"][0].copy()], fence_poly=fence, calib_note="标定还没保存",
                  yolo=dict(cam["yolo"], detector=det, conf=0.4, prompts=["cup"]))
        res = YC.perceive_cups_yolo(c2)
    finally:
        YP.fuse_depth, YP.locate_objects = _fuse, _loc
    assert det.calls[-1]["mode"] == "text" and det.calls[-1]["prompts"] == ["cup"] and det.calls[-1]["conf"] == 0.4
    assert calls["fuse"][0] == 2 and calls["fuse"][1] is calls["loc"]["dets"]
    assert calls["loc"]["fence"] == fence and calls["loc"]["note"] == "标定还没保存" and calls["loc"]["table_z"] == 0.0
    assert len(res["cups"]) == 2 and res["n_2d"] == 2
    # camera_from_tab: Camera-tab frames + fence + note
    grabbed = {}

    class _Src:
        K = dict(cam["K"])

        def running(self):
            return True

    class _Tab:
        src, result = _Src(), {"T_base_cam": np.asarray(cam["T_base_cam"]).tolist()}

        def _grab_frames(self, n, timeout=2.0):
            grabbed["n"] = n
            return cam["rgb"], [cam["depths"][0]] * n

        def _fence(self):
            return {"z_table": 0.004, "polygon": fence}

        def _calib_note(self):
            return None
    cf = RT.camera_from_tab(_Tab(), yolo={"x": 1})
    assert grabbed["n"] == YP.FUSE_FRAMES and len(cf["depths"]) == YP.FUSE_FRAMES and cf["z_table"] == 0.004
    assert cf["fence_poly"] == fence and cf["yolo"] == {"x": 1} and cf["rgb"] is cam["rgb"]
    print("[X3] same pipeline as the Camera tab: detect -> fuse_depth -> locate_objects (fence, z_table, note); "
          "camera_from_tab grabs FUSE_FRAMES via the Camera tab; colour method never called")

    # ---------------------------------------------------------------- X4
    g = CK.SC.cup_geometry()
    top_true = lambda name: (tw.cup_pose(name)[0][2] + g["rim_r"]) * 1000.0
    by_y = sorted(res["cups"], key=lambda c_: c_["base_xyz_m"][1])
    for name, c in zip(("cup_1", "cup_2"), by_y):
        rim = tw.cup_pose(name)[0]
        assert c["kind"] == "cup", c
        assert math.hypot(c["base_xyz_m"][0] - rim[0], c["base_xyz_m"][1] - rim[1]) <= 0.003, (c["base_xyz_m"], rim)
        assert abs(c["height_mm"] - top_true(name)) <= 3.0, (c["height_mm"], top_true(name))
    # rim depth dropout (the first real test: depth top 99-108 mm for a 121 mm cup)
    dep = cam["depths"][0].copy()
    Tb = np.asarray(cam["T_base_cam"], float)
    for d in det.dets:                         # no depth on the top 2.5 cm of the cup (rim region), like the real D435i
        vv, uu = np.nonzero(d["mask"] & (dep > 0))
        P = CP.backproject_mask(dep, d["mask"], cam["K"], Tb)
        hi = P[:, 2] > 0.100
        dep[vv[hi], uu[hi]] = 0.0
    res_d = YC.perceive_cups_yolo(dict(cam, depths=[dep], yolo=dict(cam["yolo"], detector=det)))
    for a, b in zip(sorted(res["cups"], key=lambda c_: c_["base_xyz_m"][1]),
                    sorted(res_d["cups"], key=lambda c_: c_["base_xyz_m"][1])):
        assert b["kind"] == "cup" and abs(a["height_mm"] - b["height_mm"]) <= 1.0, (a["height_mm"], b["height_mm"])
        assert b["height_depth_mm"] < a["height_depth_mm"] - 15.0, (a["height_depth_mm"], b["height_depth_mm"])
    # two masks of one cup -> merged
    det.dets = cam["yolo"]["detector"].next + [dict(cam["yolo"]["detector"].next[0], conf=0.5)]
    res_m = YC.perceive_cups_yolo(dict(cam, yolo=dict(cam["yolo"], detector=det)))
    assert len(res_m["cups"]) == 2 and any("合并" in w for c_ in res_m["cups"] for w in c_.get("warnings") or [])
    # outside the fence
    res_o = YC.perceive_cups_yolo(dict(cam, fence_poly=[[-0.8, 0.0], [-0.1, 0.0], [-0.1, 0.4], [-0.8, 0.4]],
                                       yolo=dict(cam["yolo"], detector=det)))
    kinds = sorted(c_["kind"] for c_ in res_o["cups"])
    assert kinds == ["cup", "outside"], kinds
    # nested pair
    r2 = tw.cup_pose("cup_2")[0]
    set_cup(tw, "cup_1", [r2[0], r2[1], r2[2] + g["nested_dz"]])
    res_s = YC.perceive_cups_yolo(camfn())
    assert len(res_s["cups"]) == 1 and res_s["cups"][0]["kind"] == "stack", YC.summary(res_s)
    assert abs(res_s["cups"][0]["height_mm"] - (top_true("cup_2") + g["nested_dz"] * 1000.0)) <= 4.0
    # cup in the air (held)
    set_cup(tw, "cup_1", [-0.40, -0.13, r2[2] + 0.08])
    res_l = YC.perceive_cups_yolo(camfn())
    assert sorted(c_["kind"] for c_ in res_l["dets"]) == ["cup", "lifted"], YC.summary(res_l)
    print(f"[X4] twin: singles cup (axis <= 3 mm, height +-3 mm), rim depth dropout -> silhouette height unchanged "
          f"(depth top {a['height_depth_mm']:.0f} -> {b['height_depth_mm']:.0f} mm); nested pair = stack "
          f"{res_s['cups'][0]['height_mm']:.0f} mm; lifted / outside / merged")

    # ---------------------------------------------------------------- X5
    log5 = []
    io, tw5, arm5, grip5 = RTW.make(log=log5.append)
    fn0 = io.camera_fn
    io.camera_fn = lambda: dict(fn0(), yolo={"error": "找不到 YOLOE 模型 yoloe-26s-seg.pt"})
    out = RT.run_plan_preview(io, "cup_1", "cup_2")
    unavailable = lambda rs: any("PERCEPTION_UNAVAILABLE" in (str(r_.get("reason")) + str(r_.get("error")))
                                 for r_ in rs)
    assert not out["ok"] and unavailable(out["results"]), out["results"]
    assert any("[FAIL] [V3.66 REAL CUP YOLOE] YOLOE 不可用" in l for l in log5) and not arm5.calls
    io.camera_fn = lambda: dict(fn0(), yolo=dict(fn0()["yolo"], detector=_Det(exc=YP.YoloUnavailable("ultralytics 不可用"))))
    log5.clear()
    out = RT.run_plan_preview(io, "cup_1", "cup_2")
    assert not out["ok"] and unavailable(out["results"]) and any("ultralytics 不可用" in l for l in log5)
    assert not arm5.calls
    CP.perceive_cups = _col
    print("[X5] YOLOE unavailable -> [FAIL] YOLOE 不可用, PERCEPTION_UNAVAILABLE, no colour fallback, no motion")

    # ---------------------------------------------------------------- X6
    log6 = []
    io, tw6, arm6, grip6 = RTW.make(log=log6.append)
    out = RT.run_execute(io, "cup_2", "cup_1")
    a_, b_ = tw6.cup_pose("cup_2"), tw6.cup_pose("cup_1")
    st6 = CK.SC.cup_task_state(a_[0], a_[1], b_[0], b_[1], origin="rim")
    assert out["ok"] and st6["nested_full"], [r_.get("reason") for r_ in out["results"] if not r_.get("ok")]
    assert max(abs(v) for v in tw6.q) < 1e-6
    last = [l for l in log6 if "[V3.66 REAL CUP YOLOE]" in l][-1]
    assert " stack@" in last, last
    print(f"[X6] twin + YOLOE pipeline: cup_2 -> cup_1 nested, back at the start; last view: {last.split('|')[-1][:90]}")
else:
    CP_reason = gl_reason if HAVE_MJ else "MuJoCo not installed"
    print(f"[X3-X6] skipped: no twin camera ({CP_reason})")

# ------------------------------------------------------------------ X7
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
        return [0.0] * 7

    def move_busy(self):
        return False


class _Src2:
    K = {"width": 1280, "height": 720, "fx": 900.0, "fy": 900.0, "ppx": 640.0, "ppy": 360.0}

    def running(self):
        return True


class _Cam2:
    src, wrist_src, worker = _Src2(), None, None
    result = calib = {"T_base_cam": np.eye(4).tolist()}


logs7 = []
tab = RT.RealCupTaskTabUI(mtk, mttk, mock.MagicMock(), _Pipe(), types.SimpleNamespace(model={"joints": []}), _Cam2(),
                          log=logs7.append, visible_fn=lambda: False)
assert tab.v_cup_prompt.get() == "cup"
cfg = tab.yolo_cfg()
assert cfg.get("error") and "YOLOE 不可用" in tab.v_yolo.get()
tab._run_mode("plan")
tab.poll()
assert any("不能开始" in l and "YOLOE 不可用" in l for l in logs7), logs7[-3:]
tab.v_cup_prompt.set("red cup")
tab._save()
saved = json.loads((Path(os.environ["NERO_CALIB_DIR"]) / "real_cup_task.json").read_text(encoding="utf-8"))
assert saved["cup_prompt"] == "red cup"
tab2 = RT.RealCupTaskTabUI(mtk, mttk, mock.MagicMock(), _Pipe(), types.SimpleNamespace(model={"joints": []}), _Cam2(),
                           log=logs7.append, visible_fn=lambda: False)
assert tab2.v_cup_prompt.get() == "red cup"
ov = RT.overlay_dets({"dets": [{"det_id": "#1", "kind": "cup", "height_mm": 124.6, "conf": 0.91,
                                "contour": [[10, 50], [40, 20], [70, 50]]},
                               {"det_id": "#2", "kind": "stack", "height_mm": 150.0, "conf": 0.8,
                                "contour": [[100, 90], [130, 60], [160, 90]]},
                               {"det_id": "#3", "kind": "lifted", "conf": 0.7, "contour": []}]})
assert len(ov) == 2 and ov[0]["label"] == "#1 单杯 125mm 0.91" and ov[0]["anchor"] == [40, 20]
assert ov[1]["label"].startswith("#2 已叠好") and ov[0]["color"] != ov[1]["color"]
print("[X7] tab: prompt field (saved in NERO_CALIB_DIR), YOLOE settings shown, run refused when YOLOE is "
      "unavailable, overlay contours + labels")

SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("ALL V3.66 TESTS PASSED")
