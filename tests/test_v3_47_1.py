#!/usr/bin/env python3
"""V3.47.1: egg pose without depth on the egg (YOLO silhouette + local table plane + egg size).

P1 D435-like egg depth (holes / flattened to 15 mm / ideal) -> both eggs trusted, centre error < 3 mm, same result
P2 cables next to an egg + an object near it: table plane from the 3-6 cm band stays right (rms < 2 mm, offset < 2 mm)
P3 YOLO mask 5 px too big -> still within 3 mm; a flat thing (paper, 3 mm) -> rejected as flat
P4 unverified hand-eye calibration (calib_note) -> results listed but none trusted, note in the reasons
P5 no table depth around the object -> z_table fallback, not trusted; nothing at all -> clear reason
P6 calib_valid(); depth settings / file constants; NERO_V308 unchanged
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
import NERO_V325_Camera as CAM                         # noqa: E402

assert YP.VERSION == "V3.47.1" and (ROOT / "VERSION").read_text().strip() == "V3.47.1"
TD = tempfile.mkdtemp(prefix="v3471_")
YP.REF_JSON, YP.REF_IMG = os.path.join(TD, "yolo_ref.json"), os.path.join(TD, "yolo_ref.jpg")
MP = os.path.join(TD, "yoloe-26s-seg.pt")
Path(MP).write_bytes(b"w")
FENCE = [[-0.8, -0.4], [-0.1, -0.4], [-0.1, 0.4], [-0.8, 0.4]]
F.install()
det = YP.YoloEDetector(log=lambda *a: None)


def run(obs, depth_mod=None, bleed=2, table_z=0.0, calib_note=None, seed=0):
    rgb, depth, ids = F.render(obs, seed=seed)
    F.STATE.scene, F.STATE.bleed_px = (obs, ids), bleed
    ref = YP.ref_load() or YP.ref_save(rgb, [[500, 400, 560, 460]], [0])
    d2 = det.detect(rgb, "vp", MP, ref=ref)
    if depth_mod:
        depth = depth_mod(depth, ids, obs)
    res, diag = YP.locate(d2, depth, F.K, F.T_BC, table_z=table_z, fence_poly=FENCE, calib_note=calib_note)
    named = {}
    for r in res:
        o = min((o for o in obs if o["conf"] > 0), key=lambda o: abs(o["conf"] - r["yolo"]["conf"]))
        named[o["name"]] = (r, o)
    return named, diag


def err_mm(r, o):
    return float(np.linalg.norm(np.subtract(r["xyz_m"], o["c"])) * 1000.0)


def holes(depth, ids, obs):
    d = depth.copy()
    for k, o in enumerate(obs):
        if o["kind"] == "egg":
            d[ids == k] = 0
    return d


def flatten(depth, ids, obs, h=0.015):
    d = depth.copy()
    O = F.T_BC[:3, 3]
    for k, o in enumerate(obs):
        if o["kind"] == "egg":
            vv, uu = np.nonzero(ids == k)
            dc = np.stack([(uu - F.K["ppx"]) / F.K["fx"], (vv - F.K["ppy"]) / F.K["fy"], np.ones(len(uu))], 1)
            D = dc @ F.T_BC[:3, :3].T
            d[vv, uu] = (h - O[2]) / D[:, 2]
    return d


EGGS = [o for o in F.default_objects() if o["name"] in ("E1", "E2")]
# ------------------------------------------------------------------ P1
out = {}
for tag, mod in (("ideal", None), ("holes", holes), ("flat15", flatten)):
    named, diag = run(EGGS, mod)
    for n in ("E1", "E2"):
        r, o = named[n]
        e = err_mm(r, o)
        assert r["status"] == "egg" and r["trusted"] and e < 3.0 and r["pose"] == "平躺", (tag, n, e, r)
        out[(tag, n)] = r["xyz_m"]
    if tag == "holes":
        assert named["E1"][0]["height_mm"] is None and "深度不足" in named["E1"][0]["notes"][0]
    if tag == "flat15":
        assert abs(named["E1"][0]["height_mm"] - 15) < 2
for n in ("E1", "E2"):
    assert np.linalg.norm(np.subtract(out[("ideal", n)], out[("holes", n)])) < 1e-6, "pose must not use egg depth"
print(f"[P1] egg depth ideal / missing / flattened to 15 mm -> same pose, centre < 3 mm (E1 "
      f"{err_mm(*run(EGGS, holes)[0]['E1']):.1f} mm)")

# ------------------------------------------------------------------ P2
cab = [{"name": "C1", "kind": "box", "c": [-0.45, 0.06, 0.004], "yaw": math.radians(30), "half": [0.12, 0.004, 0.004],
        "conf": 0.0},
       {"name": "C2", "kind": "box", "c": [-0.40, -0.02, 0.005], "yaw": math.radians(-70), "half": [0.10, 0.005, 0.005],
        "conf": 0.0}]
named, diag = run(EGGS + cab, holes)
for n in ("E1", "E2"):
    r, o = named[n]
    assert r["trusted"] and err_mm(r, o) < 3.0 and abs(r["ground_z_m"]) < 0.002 and r["plane_rms_mm"] < 2.0, (n, r)
print(f"[P2] cables 1-4 cm from the eggs: plane rms {diag['plane_rms_mm']:.2f} mm, table z {diag['ground_z_m'] * 1000:+.1f} mm,"
      " eggs still < 3 mm")

# ------------------------------------------------------------------ P3
named, _ = run(EGGS, holes, bleed=5)
for n in ("E1", "E2"):
    r, o = named[n]
    assert err_mm(r, o) < 3.0 and r["status"] in ("egg", "untrusted"), (n, err_mm(r, o), r)
paper = [{"name": "PA", "kind": "box", "c": [-0.50, 0.08, 0.0015], "yaw": 0.3, "half": [0.025, 0.02, 0.0015],
          "conf": 0.9}]
named, _ = run(EGGS + paper)
pa = named["PA"][0]
assert pa["status"] == "rejected" and "扁平" in pa["reasons"][0], pa
print(f"[P3] YOLO mask 5 px too big -> centre < 3 mm; flat paper -> rejected ({pa['reasons'][0]})")

# ------------------------------------------------------------------ P4
named, _ = run(EGGS, calib_note="标定读取后还没校验")
for n in ("E1", "E2"):
    r = named[n][0]
    assert r["status"] == "untrusted" and not r["trusted"] and "标定读取后还没校验" in r["reasons"]
print("[P4] unverified calibration -> eggs listed, none trusted, reason shown")

# ------------------------------------------------------------------ P5
def no_table(depth, ids, obs):
    d = depth.copy()
    d[ids < 0] = 0
    return d


named, _ = run(EGGS, no_table, table_z=0.0)
r = named["E1"][0]
assert r["status"] == "untrusted" and any("z_table" in w for w in r["reasons"]) and err_mm(r, named["E1"][1]) < 3.0, r
named, _ = run(EGGS, no_table, table_z=None)
assert named["E1"][0]["status"] == "rejected" and "测不到桌面平面" in named["E1"][0]["reasons"][0]
print("[P5] no table depth -> z_table fallback (not trusted); without z_table -> rejected with reason")

# ------------------------------------------------------------------ P6
good = {"T_base_cam": np.eye(4).tolist(), "marker_in_flange": [0, 0, 0.1]}
assert CAM.calib_valid(good) and not CAM.calib_valid({"T_base_cam": [[1]]}) and not CAM.calib_valid(None)
assert not CAM.calib_valid(dict(good, load_error="x"))
assert CAM.DEPTH_RES == ("848×480", "1280×720") and CAM.DEPTH_PRESETS[0] == "High Accuracy"
assert CAM.CALIB_DIR.endswith(os.path.join("Downloads", "cameracalib")) or os.environ.get("NERO_CALIB_DIR")
src = CAM.RealSenseSource(log=lambda *a: None)
assert (src.depth_res, src.depth_preset) == ("848×480", "High Accuracy")       # default unchanged
S = (ROOT / "NERO_V325_Camera.py").read_text()
assert "dw, dh = (1280, 720) if self.depth_res == DEPTH_RES[1] else (848, 480)" in S
assert hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest().startswith("020cc6680f181872")
print("[P6] calib_valid; depth 848×480 High Accuracy by default (1280×720 / presets selectable); NERO_V308 unchanged")
print("ALL V3.47.1 TESTS PASSED")
