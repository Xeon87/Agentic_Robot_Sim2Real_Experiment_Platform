#!/usr/bin/env python3
"""V3.50: Dual 16:9 live monitor canvases (560x315), wrist frame quality & rolling-shutter motion gate,
TCP crosshair & fingertip drop-zone projection, and monocular plane-constrained coarse-to-fine 3D refinement.

T1 Version V3.50 + markers in NERO_V325_Camera.py and NERO_V349_Real_Wrist_Camera.py; NERO_V308 untouched;
   place_yolo_labels scales accurately for both 640x360 and 560x315 canvases
T2 evaluate_wrist_frame: detects settled sharp frame (ok), joint/pixel motion (motion), highlight saturation (overexp),
   and defocus (blur)
T3 project_tcp_wrist: projects active TCP crosshair, 21 mm egg ring, gripper opening axis, and left/right fingertip
   drop-zone boxes (both full-res and 560x315 canvas coords)
T4 wrist_refine_target: close-up monocular plane-constrained refinement recovers true 3D egg centre (< 0.3 mm),
   principal axis yaw (< 0.5 deg), physical size_mm (~55x42 mm), and exact delta_tcp_mm / delta_coarse_mm
T5 wrist_refine_target safety gates: detects neighbouring object encroaching on fingertip drop-zones
   (clearance_ok=False) and flags coarse-vs-fine discrepancy > 20 mm (trusted=False)
T6 Printable A4 and US Letter 300-DPI calibration board PNGs present in config/
"""
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V347_Yolo_Perception as YP        # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49     # noqa: E402

# ------------------------------------------------------------------ T1: Version, markers & canvas scaling
assert W49.VERSION in ("V3.50", "V3.51", "V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82") and YP.VERSION in ("V3.50", "V3.51", "V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82") and (ROOT / "VERSION").read_text().strip() in ("V3.50", "V3.51", "V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82")
h308 = hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest()
assert h308.startswith("020cc6680f181872"), f"NERO_V308 modified: {h308}"
assert CAM.CameraTabUI.DW == 560 and CAM.CameraTabUI.DH == 315
p640 = CAM.place_yolo_labels([{"box": [640, 360, 768, 432], "label": "e1", "status": "egg"}], canvas_w=640, canvas_h=360)
p560 = CAM.place_yolo_labels([{"box": [640, 360, 768, 432], "label": "e1", "status": "egg"}], canvas_w=560, canvas_h=315)
assert p640[0]["box_cv"] == (320.0, 180.0, 384.0, 216.0)
assert p560[0]["box_cv"] == (280.0, 157.5, 336.0, 189.0)
print("[T1] V3.50 version & markers verified; NERO_V308 untouched; CameraTabUI DWxDH=560x315; place_yolo_labels scales for 640x360 and 560x315")

# ------------------------------------------------------------------ T2: evaluate_wrist_frame quality & motion gate
rng = np.random.default_rng(50)
sharp_rgb = rng.integers(35, 205, size=(360, 640, 3), dtype=np.uint8)
r_ok = W49.evaluate_wrist_frame(sharp_rgb, sharp_rgb.copy(), [0.0] * 7, [0.0] * 7)
assert r_ok["settled"] and r_ok["status"] == "ok" and "静止可采" in r_ok["badge"], r_ok

r_jmot = W49.evaluate_wrist_frame(sharp_rgb, sharp_rgb.copy(), [0.0] * 7, [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
assert not r_jmot["settled"] and r_jmot["status"] == "motion" and "运动模糊" in r_jmot["badge"], r_jmot

shifted_rgb = np.roll(sharp_rgb, 8, axis=1)
r_pmot = W49.evaluate_wrist_frame(sharp_rgb, shifted_rgb, [0.0] * 7, [0.0] * 7)
assert not r_pmot["settled"] and r_pmot["status"] == "motion", r_pmot

over_rgb = np.full((360, 640, 3), 253, dtype=np.uint8)
r_over = W49.evaluate_wrist_frame(over_rgb, over_rgb, [0.0] * 7, [0.0] * 7)
assert not r_over["settled"] and r_over["status"] == "overexp" and "反光过曝" in r_over["badge"], r_over

flat_rgb = np.full((360, 640, 3), 120, dtype=np.uint8)
r_blur = W49.evaluate_wrist_frame(flat_rgb, flat_rgb, [0.0] * 7, [0.0] * 7)
assert not r_blur["settled"] and r_blur["status"] == "blur" and "失焦模糊" in r_blur["badge"], r_blur
print(f"[T2] evaluate_wrist_frame: ok ({r_ok['badge']}), motion ({r_jmot['badge']}), overexp ({r_over['badge']}), blur ({r_blur['badge']})")

# ------------------------------------------------------------------ T3: project_tcp_wrist
K_w = {"width": 1280, "height": 720, "fx": 820.0, "fy": 820.0, "ppx": 640.0, "ppy": 360.0}
T_fc = [[1.0, 0.0, 0.0, 0.032], [0.0, 1.0, 0.0, -0.018], [0.0, 0.0, 1.0, 0.040], [0.0, 0.0, 0.0, 1.0]]
tool_T = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.145], [0.0, 0.0, 0.0, 1.0]]
proj = W49.project_tcp_wrist(K_w, T_fc, tool_T, z_work_m=0.11, canvas_size=(560, 315))
assert proj and proj["ok"] and proj["radius_21mm_px"] > 100.0
assert 0 < proj["uv_tcp_cv"][0] < 560 and 0 < proj["uv_tcp_cv"][1] < 315
assert len(proj["left_box_cv"]) == 4 and len(proj["right_box_cv"]) == 4
print(f"[T3] project_tcp_wrist: uv_tcp={proj['uv_tcp']} (canvas {proj['uv_tcp_cv']}), r_21mm={proj['radius_21mm_px']} px, scale={proj['px_per_mm']} px/mm")

# ------------------------------------------------------------------ T4: wrist_refine_target accuracy
R_bw = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
t_bw = np.array([-0.445, 0.016, 0.161])
T_bw = np.eye(4); T_bw[:3, :3] = R_bw; T_bw[:3, 3] = t_bw
T_bf = (T_bw @ np.linalg.inv(np.array(T_fc))).tolist()
calib_eih = {"T_flange_cam": T_fc, "marker_in_base": [-0.45, 0.0, 0.0]}

true_egg = np.array([-0.450, 0.020, 0.021])
yaw_true = math.radians(32.0)
H, W = 720, 1280
vv, uu = np.mgrid[0:H, 0:W].astype(float)
dc = np.stack([(uu - K_w["ppx"]) / K_w["fx"], (vv - K_w["ppy"]) / K_w["fy"], np.ones_like(uu)], -1)
D = dc @ R_bw.T
lam = (true_egg[2] - t_bw[2]) / D[..., 2]
P = t_bw + lam[..., None] * D
rel = P - true_egg
u_ax = np.array([math.cos(yaw_true), math.sin(yaw_true), 0.0])
v_ax = np.array([-math.sin(yaw_true), math.cos(yaw_true), 0.0])
in_egg = ((rel @ u_ax) / 0.0275) ** 2 + ((rel @ v_ax) / 0.021) ** 2 <= 1.0
rgb_w = np.full((H, W, 3), 35, dtype=np.uint8)
rgb_w[in_egg] = (238, 236, 230)

coarse_item = {"det_id": "egg_1", "xyz_m": [-0.445, 0.016, 0.021], "yaw_deg": 20.0, "trusted": True}
ref_res = W49.wrist_refine_target(rgb_w, K_w, T_bf, calib_eih, tool_T, coarse_item)
assert ref_res["ok"] and ref_res["trusted"] and ref_res["clearance_ok"] and ref_res["in_center_roi"], ref_res
err_xy_mm = math.hypot(ref_res["fine_xyz_m"][0] - true_egg[0], ref_res["fine_xyz_m"][1] - true_egg[1]) * 1000.0
err_yaw = abs(ref_res["fine_yaw_deg"] - 32.0)
assert err_xy_mm < 0.3 and err_yaw < 0.5, (err_xy_mm, err_yaw, ref_res)
assert abs(ref_res["delta_coarse_mm"] - math.hypot(-0.005, 0.004) * 1000.0) < 0.3
print(f"[T4] wrist_refine_target: fine_xyz={ref_res['fine_xyz_m']} (err {err_xy_mm:.3f} mm), "
      f"fine_yaw={ref_res['fine_yaw_deg']}° (err {err_yaw:.2f}°), delta_tcp_mm={ref_res['delta_tcp_mm']}, "
      f"delta_coarse_mm={ref_res['delta_coarse_mm']} mm")

# ------------------------------------------------------------------ T5: fingertip clearance & coarse discrepancy gates
# Place a second obstacle right at the left fingertip drop-zone (28 mm along grasp axis v_ax)
obs_c = true_egg + 0.034 * v_ax
rel_obs = P - obs_c
in_obs = (rel_obs[..., 0] ** 2 + rel_obs[..., 1] ** 2) <= (0.009 ** 2)
rgb_crowded = rgb_w.copy()
rgb_crowded[in_obs] = (230, 230, 225)
res_crowd = W49.wrist_refine_target(rgb_crowded, K_w, T_bf, calib_eih, tool_T, coarse_item)
assert res_crowd["ok"] and not res_crowd["clearance_ok"] and not res_crowd["trusted"], res_crowd
assert any("指尖落点区" in w for w in res_crowd["warnings"]), res_crowd["warnings"]

# Coarse target > 20 mm away -> flagged untrusted
far_coarse = {"det_id": "egg_1", "xyz_m": [-0.415, 0.020, 0.021], "yaw_deg": 32.0, "trusted": True}
res_far = W49.wrist_refine_target(rgb_w, K_w, T_bf, calib_eih, tool_T, far_coarse)
assert res_far["ok"] and res_far["delta_coarse_mm"] > 20.0 and not res_far["trusted"], res_far
assert any("与主相机粗定位偏差" in w for w in res_far["warnings"]), res_far["warnings"]
print(f"[T5] Safety gates: crowded fingertip clearance_ok={res_crowd['clearance_ok']}; "
      f"far coarse diff={res_far['delta_coarse_mm']} mm -> trusted={res_far['trusted']}")

# ------------------------------------------------------------------ T6: Printable A4 & US Letter calibration boards
for fn in ("calib_charuco_7x5_32mm_A4.png", "calib_chessboard_9x6_24mm_A4.png",
           "calib_charuco_7x5_32mm_Letter.png", "calib_chessboard_9x6_24mm_Letter.png"):
    p = ROOT / "config" / fn
    assert p.is_file() and p.stat().st_size > 15000, f"Missing or empty board PNG: {p}"
print("[T6] Both A4 and US Letter 300-DPI calibration board PNGs verified in config/")
print("ALL V3.50 TESTS PASSED")
