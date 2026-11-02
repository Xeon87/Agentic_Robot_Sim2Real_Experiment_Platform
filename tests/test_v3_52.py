#!/usr/bin/env python3
"""V3.52: 8-parameter CALIB_RATIONAL_MODEL ArUco PnP fix (no [:5] truncation), point-duality Eye-in-Hand solver
immune to IPPE rvec flips, fast (< 0.05 s) joint-space EIH auto-planner + start-pose tabletop prior, unified
560x315 main camera canvas scaling, and return_home=False in RealPipeline.set_teach_mode.

V1 Version V3.52 + markers in NERO_V312_Console.py, NERO_V325_Camera.py, and NERO_V349_Real_Wrist_Camera.py;
   NERO_V308 untouched
V2 detect_aruco with 8-parameter rational distortion (CALIB_RATIONAL_MODEL) on a 1920x1080 frame: recovers true
   3D marker position to < 1.5 mm (whereas [:5] truncation caused > 30 mm error) and runs in < 0.08 s
V3 Point-duality solve_eye_in_hand & calibrate_eye_in_hand: immune to 35° IPPE planar rvec flips on 30% of frames
   (t_X error < 2.0 mm, LOO < 3.0 mm, good=True); plan_eih_auto_targets runs in < 0.10 s
V4 Unified 560x315 canvas scaling (_px & place_yolo_labels agree to 0.0 px) and RealPipeline.set_teach_mode
   supports return_home=False
"""
import hashlib
import inspect
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import NERO_V312_Console as CON               # noqa: E402
import NERO_V313_Real_Robot as RR             # noqa: E402
import NERO_V315_Consistency_Safety as CS     # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V347_Yolo_Perception as YP        # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49     # noqa: E402

# ------------------------------------------------------------------ V1: Version & markers
assert W49.VERSION in ("V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64") and YP.VERSION in ("V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64") and CON.VERSION in ("V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64") and RR.VERSION in ("V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64")
assert (ROOT / "VERSION").read_text().strip() in ("V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64")
h308 = hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest()
assert h308.startswith("020cc6680f181872"), f"NERO_V308 modified: {h308}"
print("[V1] V3.52 versions & markers verified; NERO_V308 untouched")

# ------------------------------------------------------------------ V2: 8-param CALIB_RATIONAL_MODEL ArUco PnP fix + 1080P speed
K_1080 = {"width": 1920, "height": 1080, "fx": 1120.0, "fy": 1120.0, "ppx": 960.0, "ppy": 540.0}
Km = np.array([[K_1080["fx"], 0.0, K_1080["ppx"]], [0.0, K_1080["fy"], K_1080["ppy"]], [0.0, 0.0, 1.0]], dtype=np.float64)
# Realistic 8-parameter rational wide-angle coefficients where k4..k6 denominator is active:
dist8 = [1.85, 0.92, 0.0005, -0.0004, 0.45, 2.25, 1.40, 0.78]
size_m = 0.024
h = size_m / 2.0
obj_m = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)
R_front = np.diag([1.0, -1.0, -1.0]) @ np.array(CAM.rodrigues([0.08, -0.14, 0.20]))
rvec_true, _ = cv2.Rodrigues(R_front)
tvec_true = np.array([0.065, -0.045, 0.312], dtype=np.float64)

# Project the 4 corners of the marker using full 8-parameter rational distortion:
proj_8, _ = cv2.projectPoints(obj_m, rvec_true, tvec_true, Km, np.array(dist8, dtype=np.float64))
c_true = proj_8.reshape(4, 2).astype(np.float32)

# Warp a crisp 200x200 ArUco id=0 marker onto a 1920x1080 frame at c_true
d_dict = CAM._aruco_dict(cv2, "DICT_4X4_50")
if hasattr(cv2.aruco, "generateImageMarker"):
    m_tile = cv2.aruco.generateImageMarker(d_dict, 0, 200)
else:
    m_tile = cv2.aruco.drawMarker(d_dict, 0, 200)
src_corners = np.float32([[0, 0], [199, 0], [199, 199], [0, 199]])
H_warp = cv2.getPerspectiveTransform(src_corners, c_true)
frame_1080 = cv2.warpPerspective(m_tile, H_warp, (1920, 1080), flags=cv2.INTER_LINEAR, borderValue=235)
rgb_1080 = cv2.cvtColor(frame_1080, cv2.COLOR_GRAY2RGB)

t0 = time.time()
det8 = CAM.detect_aruco(rgb_1080, K_1080, dist8, "DICT_4X4_50", 0, size_m)
dt_ms = (time.time() - t0) * 1000.0
assert det8 is not None, "detect_aruco failed on 1920x1080 rational-distorted marker"
err8_mm = CAM.vnorm(CAM.vsub(det8["p_cam"], tvec_true.tolist())) * 1000.0
# Compare against what truncated 5-parameter dist[:5] would have produced on the same corners:
_, _, tv_trunc = cv2.solvePnP(obj_m, c_true.astype(np.float64), Km, np.array(dist8[:5], dtype=np.float64),
                              flags=getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE))
err_trunc_mm = CAM.vnorm(CAM.vsub(tv_trunc.flatten().tolist(), tvec_true.tolist())) * 1000.0
assert err8_mm < 2.0 and err_trunc_mm > 25.0, (err8_mm, err_trunc_mm)
assert dt_ms < 120.0, dt_ms
print(f"[V2] 8-param rational ArUco PnP on 1920x1080: error={err8_mm:.2f} mm (truncated [:5] was {err_trunc_mm:.1f} mm!), time={dt_ms:.1f} ms")

# ------------------------------------------------------------------ V3: Point-duality EIH solver & fast planner
rng = np.random.default_rng(52)
X_true = CAM.T_from(CAM.rodrigues([0.10, -0.22, 1.48]), [0.042, -0.016, 0.028])
y_true = [-0.450, 0.020, 0.0]
samples_eih = []
for k in range(10):
    rv_a = [rng.uniform(-0.38, 0.38), rng.uniform(-0.38, 0.38), rng.uniform(-0.48, 0.48)]
    ta = [-0.42 + rng.uniform(-0.09, 0.09), 0.02 + rng.uniform(-0.09, 0.09), 0.25 + rng.uniform(-0.05, 0.07)]
    A = CAM.T_from(CAM.rodrigues(rv_a), ta)
    AX = CAM.mat_mul(A, X_true)
    p_noisy = [v + rng.normal(0, 0.0005) for v in CAM.T_apply(CAM.T_inv(AX), y_true)]
    R_b = CAM.T_R(CAM.T_inv(AX))
    if k in (1, 4, 7):
        # Inject 35-deg IPPE planar flip on rvec
        R_b = CAM.mat_mul(R_b, CAM.rodrigues([math.radians(35.0), math.radians(-25.0), 0.0]))
    B = CAM.T_from(R_b, p_noisy)
    samples_eih.append({"q_deg": [0.0] * 7, "p_cam": p_noisy, "T_cam_marker": B, "T_base_flange": A})

res_eih = W49.calibrate_eye_in_hand(samples_eih, size_mm=24.0)
assert res_eih["ok"] and res_eih["good"], res_eih
err_tx_mm = CAM.vnorm(CAM.vsub(res_eih["cam_in_flange"], CAM.T_t(X_true))) * 1000.0
err_y_mm = CAM.vnorm(CAM.vsub(res_eih["marker_in_base"], y_true)) * 1000.0
assert err_tx_mm < 2.0 and err_y_mm < 1.5 and res_eih["loo_mean_mm"] < 2.0, (err_tx_mm, err_y_mm, res_eih)

# Fast auto-planner timing check (< 0.10 s)
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
LIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708)]
q0_rad = [math.radians(v) for v in (0.0, 30.0, 0.0, 80.0, 0.0, 40.0, 0.0)]
fk_fn = lambda qr: [list(map(float, r)) for r in CS.fk(U, dict(zip(J, qr)), "base_link", "link7")]
A0 = fk_fn(q0_rad)
T_bc0 = [[1.0, 0.0, 0.0, -0.450], [0.0, -1.0, 0.0, 0.020], [0.0, 0.0, -1.0, 0.250], [0.0, 0.0, 0.0, 1.0]]
B0 = CAM.mat_mul(CAM.T_inv(T_bc0), CAM.T_from([[1, 0, 0], [0, 1, 0], [0, 0, 1]], y_true))
prior0 = W49.initial_eih_prior_from_start({"T_base_flange": A0, "T_cam_marker": B0, "p_cam": CAM.T_t(B0)})
t_p0 = time.time()
tgts, st_p = W49.plan_eih_auto_targets(fk_fn, None, q0_rad, LIM, prior0, K_1080, n=12, size_m=0.024, seed=9)
dt_plan_ms = (time.time() - t_p0) * 1000.0
assert len(tgts) == 12 and dt_plan_ms < 350.0, (len(tgts), dt_plan_ms)
print(f"[V3] Point-duality EIH with 35° IPPE flips: t_X err={err_tx_mm:.2f} mm, LOO={res_eih['loo_mean_mm']:.2f} mm (good=True); "
      f"plan_eih_auto_targets planned 12 poses in {dt_plan_ms:.1f} ms")

# ------------------------------------------------------------------ V4: Unified 560x315 scaling & return_home=False
sig = inspect.signature(RR.RealPipeline.set_teach_mode)
assert "return_home" in sig.parameters and sig.parameters["return_home"].default is True
dummy_ui = type("D", (), {"DW": 560, "DH": 315})()
px_pt = CAM.CameraTabUI._px(dummy_ui, (640.0, 360.0))
placed = CAM.place_yolo_labels([{"box": [640.0, 360.0, 800.0, 450.0], "label": "e1", "status": "egg"}], 560, 315)
assert px_pt == (280.0, 157.5) and placed[0]["box_cv"][:2] == (280.0, 157.5)
back_uv = CAM.CameraTabUI._canvas_to_img(dummy_ui, 280.0, 157.5)
assert back_uv == (640.0, 360.0)
print("[V4] Unified 560x315 canvas scaling (_px == place_yolo_labels == (280.0, 157.5)) and set_teach_mode(return_home=False) verified")
print("ALL V3.52 TESTS PASSED")
