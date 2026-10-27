#!/usr/bin/env python3
"""V3.47: YOLOE-26 egg perception on the real D435i (Camera调试 → 视觉定位), with a fake ultralytics + a ray-cast scene.

Y1 visual prompt: the detector gets BGR (stream is RGB), refer_image = saved reference, visual_prompts boxes / cls,
   YOLOEVPSegPredictor; masks from the polygons
Y2 3D: two lying eggs 55 x 42 mm -> trusted, centre error < 3 mm (YOLO mask 2 px too big); AirPods-like box ->
   not trusted (silhouette); egg cut by the image border -> not trusted; local table plane (z_table 12 mm off in the
   safety settings is reported, results unaffected)   [V3.47.1 semantics]
Y3 reference with a "非鸡蛋" box -> that object comes back as class 1 -> rejected; reference save / load / checks
Y4 egg 58 x 44 mm while 55 x 42 is set -> still found (size search), centre < 3 mm; a standing egg is fine too
Y5 text prompt: encoder missing -> clear message, no download attempt; present -> set_classes in the model folder,
   working directory restored; ultralytics missing / model missing -> clear message
Y6 sample saving (jpg + json); NERO_V308 unchanged; simulation / wrist / fixed camera do not use the new module
"""
import hashlib
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import v347_fake as F                                  # noqa: E402
import NERO_V347_Yolo_Perception as YP                 # noqa: E402

assert YP.VERSION in ("V3.47", "V3.47.1") and (ROOT / "VERSION").read_text().strip() in ("V3.47", "V3.47.1")
TD = tempfile.mkdtemp(prefix="v347_")
YP.REF_JSON, YP.REF_IMG, YP.SAMPLES_DIR = os.path.join(TD, "yolo_ref.json"), os.path.join(TD, "yolo_ref.jpg"), \
    os.path.join(TD, "samples")
MD = os.path.join(TD, "objectmodels")
os.makedirs(MD)
MP = os.path.join(MD, "yoloe-26s-seg.pt")
Path(MP).write_bytes(b"fake-weights")
FENCE = [[-0.8, -0.4], [-0.1, -0.4], [-0.1, 0.4], [-0.8, 0.4]]
logs = []
F.install()

# ------------------------------------------------------------------ Y1
obs = F.default_objects()
rgb, depth, ids = F.render(obs)
F.STATE.scene = (obs, ids)
ref = YP.ref_save(rgb, [[540, 380, 600, 440]], [0])
assert ref and ref["cls"] == [0] and os.path.isfile(YP.REF_IMG)
det = YP.YoloEDetector(log=logs.append)
assert YP.find_model(MD, "yoloe-26s-seg.pt") == MP and YP.list_models(MD) == ["yoloe-26s-seg.pt"]
d2 = det.detect(rgb, "vp", MP, ref=ref)
c = F.STATE.calls[-1]
assert np.array_equal(c["source"], rgb[..., ::-1]), "detector must get BGR"
assert c["refer_image"] == YP.REF_IMG and c["predictor"] is F.YOLOEVPSegPredictor
assert c["visual_prompts"]["bboxes"].shape == (1, 4) and list(c["visual_prompts"]["cls"]) == [0]
assert len(d2) == 4 and all(d["mask"].shape == depth.shape and d["mask"].sum() > 500 for d in d2)
assert any("sha256" in m and "vp" in m for m in logs)
print(f"[Y1] visual prompt: BGR input, refer_image = saved reference, boxes / cls, VP seg predictor; {len(d2)} masks")

# ------------------------------------------------------------------ Y2
res, diag = YP.locate(d2, depth, F.K, F.T_BC, table_z=-0.012, fence_poly=FENCE)
by = {}
for r in res:
    best = min(obs, key=lambda o: abs(o["conf"] - r["yolo"]["conf"]))
    by[best["name"]] = r
for n in ("E1", "E2"):
    r = by[n]
    t = next(o for o in obs if o["name"] == n)["c"]
    err = np.linalg.norm(np.subtract(r["xyz_m"], t)) * 1000
    assert r["status"] == "egg" and r["trusted"] and err < 3.0 and r["reasons"] == [], (n, r, err)
    assert abs(r["height_mm"] - 42) < 3 and "location" not in r
ap = by["AP"]                      # V3.47.1: an AirPods-like box does not match an egg silhouette -> not trusted
assert not ap["trusted"] and ap["status"] in ("untrusted", "rejected") and ap["reasons"], ap
e3 = by["E3"]
assert e3["status"] == "untrusted" and not e3["trusted"] and any("边缘" in w for w in e3["reasons"]), e3
assert abs(diag["ground_minus_z_table_mm"] - 12.0) < 1.5, diag
assert [r["det_id"] for r in res][:2] == ["egg_1", "egg_2"]
print(f"[Y2] eggs trusted (centre err < 3 mm, height {by['E1']['height_mm']:.0f} mm); AirPods-like box not trusted "
      f"({ap['reasons'][0]}); egg at the border not trusted; local table {diag['ground_minus_z_table_mm']:+.0f} mm vs z_table")

# ------------------------------------------------------------------ Y3
F.STATE.not_egg_names = {"AP"}
ref2 = YP.ref_save(rgb, [[540, 380, 600, 440], [300, 500, 360, 560]], [0, 1])
d2b = det.detect(rgb, "vp", MP, ref=ref2)
res_b, _ = YP.locate(d2b, depth, F.K, F.T_BC, table_z=0.0, fence_poly=FENCE)
apb = [r for r in res_b if r["yolo"]["cls"] == 1]
assert len(apb) == 1 and apb[0]["status"] == "rejected" and apb[0]["reasons"] == ["参考里的非鸡蛋"]
assert "1 个非鸡蛋" in YP.ref_text(ref2) and YP.ref_load()["cls"] == [0, 1]
for bad in (([[1, 1, 30, 30]], [1]), ([], [])):
    try:
        YP.ref_save(rgb, *bad)
        raise AssertionError("reference without an egg box accepted")
    except ValueError:
        pass
try:
    det.detect(rgb, "vp", MP, ref=None)
    raise AssertionError("vp without reference")
except YP.YoloUnavailable as exc:
    assert "参考" in str(exc)
F.STATE.not_egg_names = set()
print("[Y3] reference with a 非鸡蛋 box -> that object rejected; reference needs >= 1 egg box; vp without reference -> message")

# ------------------------------------------------------------------ Y4
obs4 = [{"name": "B1", "kind": "egg", "c": [-0.47, -0.02, 0.022], "yaw": math.radians(-20), "conf": 0.9,
         "size": [0.058, 0.044]},
        {"name": "S1", "kind": "egg", "c": [-0.36, 0.10, F.A_EGG], "yaw": 0.0, "pitch": math.radians(90),
         "conf": 0.9}]                                             # standing egg (as in a holder)
rgb4, depth4, ids4 = F.render(obs4, seed=4)
F.STATE.scene = (obs4, ids4)
res4, _ = YP.locate(det.detect(rgb4, "vp", MP, ref=ref), depth4, F.K, F.T_BC, table_z=0.0, fence_poly=FENCE)
for r, o in zip(sorted(res4, key=lambda r: r["xyz_m"][0]), sorted(obs4, key=lambda o: o["c"][0])):
    err = np.linalg.norm(np.subtract(r["xyz_m"], o["c"])) * 1000
    assert r["status"] in ("egg", "untrusted") and err < 3.5, (o["name"], err, r)
    assert all("不像鸡蛋" not in w for w in r["reasons"])
big = min(res4, key=lambda r: abs(r["xyz_m"][0] + 0.47))
assert big["mask_bleed_px"] > 0.5 and big["status"] == "egg" and big["pose"] == "平躺", big     # bigger egg: mask > model
st = min(res4, key=lambda r: abs(r["xyz_m"][0] + 0.36))
assert st["axis_world"][2] > 0.95 and st["pose"] == "竖立" and abs(st["height_mm"] - 55) < 4, st
print(f"[Y4] 58 x 44 mm egg with 55 x 42 set -> found (mask {big['mask_bleed_px']:+.1f} px), standing egg found; centres < 3.5 mm")

# ------------------------------------------------------------------ Y5
F.STATE.scene = (obs, ids)
tdet = YP.YoloEDetector(log=logs.append)
try:
    tdet.detect(rgb, "text", MP, prompts=["egg"])
    raise AssertionError("text prompt without encoder")
except YP.YoloUnavailable as exc:
    assert "mobileclip2_b.ts" in str(exc) and not F.STATE.set_classes_calls
Path(MD, "mobileclip2_b.ts").write_bytes(b"enc")
cwd0 = os.getcwd()
dt = tdet.detect(rgb, "text", MP, prompts=YP.parse_prompts("white round object，white egg, egg"))
assert F.STATE.set_classes_calls[-1] == (["white round object", "white egg", "egg"], os.path.realpath(MD)) or \
    F.STATE.set_classes_calls[-1][1] == MD, F.STATE.set_classes_calls
assert os.getcwd() == cwd0 and len(dt) == 4 and dt[0]["name"] == "white round object"
n_sc = len(F.STATE.set_classes_calls)
tdet.detect(rgb, "text", MP, prompts=["white round object", "white egg", "egg"])
assert len(F.STATE.set_classes_calls) == n_sc, "same prompts -> model reused"
try:
    YP.YoloEDetector().detect(rgb, "vp", os.path.join(MD, "missing.pt"), ref=ref)
    raise AssertionError("missing model")
except YP.YoloUnavailable as exc:
    assert "找不到模型" in str(exc)
F.uninstall()
sys.modules["ultralytics"] = None
try:
    YP.YoloEDetector().detect(rgb, "vp", MP, ref=ref)
    raise AssertionError("no ultralytics")
except YP.YoloUnavailable as exc:
    assert "ultralytics 不可用" in str(exc)
del sys.modules["ultralytics"]
F.install()
print("[Y5] text prompt: missing mobileclip2_b.ts -> message (no download); set_classes in the model folder, cwd "
      "restored, model reused; missing model / ultralytics -> message")

# ------------------------------------------------------------------ Y6
sp = YP.save_sample(rgb, d2, res, {"mode": "vp"})
assert os.path.isfile(sp) and os.path.isfile(sp[:-4] + ".json")
assert hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest().startswith("020cc6680f181872")
for f in ("NERO_V2_7_1_Unified_Backend.py", "NERO_V326_Wrist_Camera.py", "NERO_V331_Fixed_Camera.py", "NERO_V2_7_1_Agent.py"):
    assert "NERO_V347" not in (ROOT / f).read_text(), f
CAMS = (ROOT / "NERO_V325_Camera.py").read_text()
assert 'LOCATE_METHODS = ("YOLOE 视觉提示", "YOLOE 文字提示", "几何法 V3.08")' in CAMS
assert "P8.perceive_eggs(rgb, depth, dict(self.src.K), T, stride=2)" in CAMS       # geometric method still there
print("[Y6] samples saved (jpg + json); NERO_V308 unchanged; sim / wrist / fixed camera do not use the new module; "
      "geometric method kept")
print("ALL V3.47 TESTS PASSED")
