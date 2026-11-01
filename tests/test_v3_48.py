#!/usr/bin/env python3
"""V3.48: TCP calibration + hover verification, general YOLOE-26 object recognition (text & -pf built-in vocabulary),
and clear hand-eye calibration.

R1 TCP calibration (solve_3x3, solve_tcp_pivot, calibrate_tcp): recovers true TCP in link7 within < 1.5 mm from noisy
   single-point touch poses, rejects an outlier pose, computes LOO & diff vs URDF default; rejects narrow rotation / < 4
R2 Hover IK (plan_hover_target): from zero home pose and work pose, places TCP 100 mm above targets (< 1 mm error,
   link7 above TCP, >= 3 deg soft-limit margin, safety fence checked)
R3 YOLOE-26 prompt-free (*-pf.pt) built-in vocabulary: runs without mobileclip2_b.ts and without set_classes(); wildcard
   returns all built-in classes, object name input filters matching classes and records pf_seen
R4 YOLOE-26 text-prompt general objects + locate_objects(): 3D centres in base_link within < 8 mm, height above local
   3-6 cm table plane, footprint size; missing surface depth falls back to centroid ray + plane
R5 short_egg_label for objects ('obj_1' + 'cup' -> 'o1:cup') + non-overlapping label placement
R6 NERO_V308_Camera_Perception.py unchanged; LOCATE_METHODS unchanged; sim / wrist / fixed camera untouched
"""
import hashlib
import math
import os
import random
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
import NERO_V315_Consistency_Safety as CS              # noqa: E402

assert YP.VERSION in ("V3.48", "V3.49", "V3.50", "V3.51", "V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58") and (ROOT / "VERSION").read_text().strip() in ("V3.48", "V3.49", "V3.50", "V3.51", "V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58")
TD = tempfile.mkdtemp(prefix="v348_")
YP.REF_JSON, YP.REF_IMG = os.path.join(TD, "yolo_ref.json"), os.path.join(TD, "yolo_ref.jpg")
MP = os.path.join(TD, "yoloe-26s-seg.pt")
MP_PF = os.path.join(TD, "yoloe-26s-seg-pf.pt")
Path(MP).write_bytes(b"w")
Path(MP_PF).write_bytes(b"pf")
FENCE = [[-0.8, -0.4], [-0.1, -0.4], [-0.1, 0.4], [-0.8, 0.4]]
F.install()

# ------------------------------------------------------------------ R1
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
LIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708)]
true_tcp = [0.1525, -0.0025, -0.0195]
touch_pt = [-0.42, 0.04, 0.015]
T0 = CS.fk(U, dict(zip(J, [0.0, 0.5, 0.0, 1.3, 0.0, 0.6, 0.0])), "base_link", "link7")
R0 = [list(map(float, r[:3])) for r in T0[:3]]
rnd = random.Random(12)
samples = []
for degs in ((0, 0, 0), (22, 15, -25), (-20, -18, 28), (18, -24, 22), (-16, 22, -30), (25, 12, 35)):
    R = CAM.mat_mul(CAM.rodrigues([math.radians(d) for d in degs]), R0)
    rt = CAM.mat_vec(R, true_tcp)
    tf = [touch_pt[i] - rt[i] + rnd.gauss(0, 0.0003) for i in range(3)]
    samples.append({"q_deg": [0.0] * 7, "T_base_flange": CAM.T_from(R, tf)})
samples_bad = list(samples) + [{"q_deg": [0.0] * 7,
                                "T_base_flange": CAM.T_from(R0, [touch_pt[0], touch_pt[1] + 0.025, touch_pt[2]])}]
rtcp = CAM.calibrate_tcp(samples_bad)
assert rtcp["ok"] and rtcp["good"] and rtcp["rejected"] == [6], rtcp
err_tcp = CAM.vnorm(CAM.vsub(rtcp["tcp_in_flange"], true_tcp)) * 1000.0
err_pt = CAM.vnorm(CAM.vsub(rtcp["point_in_base"], touch_pt)) * 1000.0
assert err_tcp < 1.5 and err_pt < 1.5 and rtcp["loo_max_mm"] < 3.0 and rtcp["diff_urdf_mm"] > 2.0
assert CAM.tcp_valid(rtcp) and not CAM.tcp_valid({"tcp_in_flange": [1, 2]}) and not CAM.tcp_valid(None)
assert not CAM.calibrate_tcp(samples[:3])["ok"]
same_R = [{"q_deg": [0] * 7, "T_base_flange": CAM.T_from(R0, [0, 0, i * 0.01])} for i in range(5)]
assert not CAM.calibrate_tcp(same_R)["ok"] and "朝向变化" in CAM.calibrate_tcp(same_R)["error"]
print(f"[R1] calibrate_tcp: TCP err {err_tcp:.2f} mm, touch point err {err_pt:.2f} mm, outlier rejected, LOO max "
      f"{rtcp['loo_max_mm']:.2f} mm, diff vs URDF {rtcp['diff_urdf_mm']:.1f} mm")

# ------------------------------------------------------------------ R2
tool_T = CS.tool_transform_from_sim(U)
for i in range(3):
    tool_T[i][3] = rtcp["tcp_in_flange"][i]
for q_start in ([0.0] * 7, [0.0, 0.5, 0.0, 1.3, 0.0, 0.6, 0.0]):
    for tgt in ([-0.45, 0.02, 0.121], [-0.38, -0.10, 0.121], [-0.50, 0.08, 0.130]):
        q_h, info = CAM.plan_hover_target(U, J, LIM, q_start, tool_T, tgt, fence_fn=lambda q: (True, ""))
        assert q_h is not None and info["err_mm"] < 1.0 and info["flange_z_m"] >= tgt[2] - 0.005, (q_start, tgt, info)
        for v, (lo, hi) in zip(q_h, LIM):
            assert lo + math.radians(3.0) - 1e-4 <= v <= hi - math.radians(3.0) + 1e-4
q_blk, why_blk = CAM.plan_hover_target(U, J, LIM, [0.0] * 7, tool_T, [-0.45, 0.02, 0.121],
                                       fence_fn=lambda q: (False, "超出桌面边界"))
assert q_blk is None and "安全区拦截" in why_blk
print("[R2] plan_hover_target: TCP error < 1 mm from home & work poses, flange above TCP, soft limits + fence enforced")

# ------------------------------------------------------------------ R3
obs_gen = [
    {"name": "CUP", "label": "cup", "kind": "box", "c": [-0.44, 0.05, 0.040], "yaw": 0.1,
     "half": [0.035, 0.035, 0.040], "conf": 0.91},
    {"name": "BOT", "label": "bottle", "kind": "box", "c": [-0.37, -0.09, 0.055], "yaw": -0.3,
     "half": [0.030, 0.030, 0.055], "conf": 0.88},
    {"name": "BOX", "label": "box", "kind": "box", "c": [-0.52, -0.06, 0.025], "yaw": 0.2,
     "half": [0.040, 0.030, 0.025], "conf": 0.84},
]
rgb_g, depth_g, ids_g = F.render(obs_g := obs_gen)
F.STATE.scene = (obs_g, ids_g)
F.STATE.set_classes_calls.clear()
logs = []
det = YP.YoloEDetector(log=logs.append)
assert YP.is_pf_model(MP_PF) and not YP.is_pf_model(MP)
# Notice: no mobileclip2_b.ts in TD yet, and -pf.pt must work without it and without calling set_classes()!
d_all = det.detect(rgb_g, "pf", MP_PF, prompts=["*"])
assert len(d_all) == 3 and not F.STATE.set_classes_calls
assert {d["name"] for d in d_all} == {"cup", "bottle", "box"}
d_filt = det.detect(rgb_g, "pf", MP_PF, prompts=["cup", "bottle"])
assert [d["name"] for d in d_filt] == ["cup", "bottle"] and set(det.info.get("pf_seen") or []) == {"cup", "bottle", "box"}
print("[R3] YOLOE-26 *-pf.pt built-in vocabulary: runs without mobileclip2_b.ts / set_classes; wildcard & name filter ok")

# ------------------------------------------------------------------ R4
Path(TD, "mobileclip2_b.ts").write_bytes(b"enc")
d_txt = det.detect(rgb_g, "text", MP, prompts=["cup", "bottle", "box"])
assert len(d_txt) == 3 and F.STATE.set_classes_calls[-1][0] == ["cup", "bottle", "box"]
res_o, diag_o = YP.locate_objects(d_txt, depth_g, F.K, F.T_BC, table_z=0.0, fence_poly=FENCE)
assert diag_o["n_obj"] == 3 and [r["det_id"] for r in res_o] == ["obj_1", "obj_2", "obj_3"]
for r in res_o:
    ob = next(o for o in obs_g if o["label"] == r["obj_name"])
    xy_err = math.hypot(r["xyz_m"][0] - ob["c"][0], r["xyz_m"][1] - ob["c"][1]) * 1000.0
    assert r["trusted"] and r["status"] == "object" and xy_err < 8.0 and abs(r["height_mm"] - ob["half"][2] * 2000) < 6.0, (r, xy_err)
# Missing depth on the object itself -> centroid ray + local plane fallback still locates it
depth_holes = depth_g.copy()
depth_holes[ids_g == 0] = 0
res_h, _ = YP.locate_objects(d_txt[:1], depth_holes, F.K, F.T_BC, table_z=0.0, fence_poly=FENCE)
assert res_h[0]["trusted"] and math.hypot(res_h[0]["xyz_m"][0] - obs_g[0]["c"][0], res_h[0]["xyz_m"][1] - obs_g[0]["c"][1]) * 1000.0 < 8.0
assert any("轮廓中心" in n for n in res_h[0]["notes"])
print(f"[R4] locate_objects: cup / bottle / box located within < 8 mm, heights {[round(r['height_mm']) for r in res_o]} mm; "
      "missing-depth fallback ok")

# ------------------------------------------------------------------ R5
assert CAM.short_egg_label("obj_1", "cup") == "o1:cup" and CAM.short_egg_label("egg_2") == "e2"
pl = CAM.place_yolo_labels([{"box": r["yolo"]["box"], "status": r["status"],
                             "label": CAM.short_egg_label(r["det_id"], r["obj_name"])} for r in res_o])
assert [p["label"] for p in pl] == ["o1:cup", "o2:bottle", "o3:box"]
print("[R5] short_egg_label ('obj_1'+'cup' -> 'o1:cup') & place_yolo_labels ok")

# ------------------------------------------------------------------ R6
assert hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest().startswith("020cc6680f181872")
assert CAM.LOCATE_METHODS == ("YOLOE 视觉提示", "YOLOE 文字提示", "几何法 V3.08")
for f in ("NERO_V2_7_1_Unified_Backend.py", "NERO_V326_Wrist_Camera.py", "NERO_V331_Fixed_Camera.py", "NERO_V2_7_1_Agent.py"):
    assert "NERO_V347" not in (ROOT / f).read_text(), f
print("[R6] NERO_V308_Camera_Perception.py unchanged; LOCATE_METHODS unchanged; sim / wrist / fixed camera untouched")
print("ALL V3.48 TESTS PASSED")
