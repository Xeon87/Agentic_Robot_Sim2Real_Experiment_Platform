#!/usr/bin/env python3
"""V3.56 offline unit tests:
W1 Version V3.56 & markers across NERO_V312_Console.py, NERO_V313_Real_Robot.py, NERO_V325_Camera.py,
   NERO_V347_Yolo_Perception.py, NERO_V349_Real_Wrist_Camera.py, and VERSION; NERO_V308 untouched.
W2 Full-res ROI ArUco detection on 1920x1080 small 24 mm wrist marker (< 1.0 mm 3D accuracy, < 0.4 mm jitter across
   subpixel shifts) + UsbWristSource.detect_marker automatic resolution scaling (1280x720 K on 1920x1080 frame).
W3 Pure-translation initial probes (R_k = R_0) with 55° corrupted PnP rvec normals -> estimate_eih_from_probes +
   Look-At Dome plan_eih_auto_targets: 0 '太低'/'倾角过大' mass rejections, 12 poses in central sweet spot
   (off-axis <= 0.56), span >= 16 cm, rot_span >= 30 deg, nearest-neighbor smooth ordering.
W4 Lever-arm-bounded solve_eye_in_hand (||t_X|| <= 160 mm, never +270 mm) + calibrate_eye_in_hand monocular scale
   self-calibration (pnp_scale) & LOO outlier rejection under 1.22x scale bias + 2 peripheral 45 mm outliers.
"""
import hashlib
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

# ------------------------------------------------------------------ W1: Version & markers
assert CON.VERSION in ("V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1") and RR.VERSION in ("V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1") and YP.VERSION in ("V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1") and W49.VERSION in ("V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1")
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1")
h308 = hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest()
assert h308.startswith("020cc6680f181872"), f"NERO_V308 modified: {h308}"

marker = "V3.56 FAST WRIST EYE-IN-HAND CALIB + LOOK-AT DOME PLANNER + FULL-RES ROI ARUCO + SCALE/LEVER-ARM SOLVER"
for fn in ("NERO_V349_Real_Wrist_Camera.py", "NERO_V325_Camera.py", "NERO_V312_Console.py"):
    assert marker in (ROOT / fn).read_text(encoding="utf-8"), f"Missing V3.56 marker in {fn}"
assert CAM.AUTO_RANGE_M[0] <= 0.26, CAM.AUTO_RANGE_M
print("[W1] V3.56 versions & markers verified; NERO_V308 untouched")

# ------------------------------------------------------------------ W2: Full-res ROI ArUco on 1920x1080 + K resolution auto-scaling
K_1080 = {"width": 1920, "height": 1080, "fx": 720.0, "fy": 720.0, "ppx": 960.0, "ppy": 540.0}
K_720 = {"width": 1280, "height": 720, "fx": 480.0, "fy": 480.0, "ppx": 640.0, "ppy": 360.0}
Km = np.array([[K_1080["fx"], 0.0, K_1080["ppx"]], [0.0, K_1080["fy"], K_1080["ppy"]], [0.0, 0.0, 1.0]], dtype=np.float64)
size_m = 0.024
h = size_m / 2.0
obj_m = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)
R_m = np.diag([1.0, -1.0, -1.0]) @ np.array(CAM.rodrigues([0.12, -0.10, 0.18]))
rvec_m, _ = cv2.Rodrigues(R_m)
tvec_m = np.array([0.015, -0.092, 0.263], dtype=np.float64)

d_dict = CAM._aruco_dict(cv2, "DICT_4X4_50")
m_tile = cv2.aruco.generateImageMarker(d_dict, 0, 200) if hasattr(cv2.aruco, "generateImageMarker") else cv2.aruco.drawMarker(d_dict, 0, 200)
src_c = np.float32([[-0.5, -0.5], [199.5, -0.5], [199.5, 199.5], [-0.5, 199.5]])

p_jit = []
for sub_shift in (0.0, 0.25, 0.50, 0.75, 1.0):
    tv_s = tvec_m + np.array([sub_shift * 0.0001, 0.0, 0.0])
    proj, _ = cv2.projectPoints(obj_m, rvec_m, tv_s, Km, np.zeros(5))
    H_w = cv2.getPerspectiveTransform(src_c, proj.reshape(4, 2).astype(np.float32))
    f_1080 = cv2.warpPerspective(m_tile, H_w, (1920, 1080), flags=cv2.INTER_LINEAR, borderValue=235)
    rgb_1080 = cv2.cvtColor(f_1080, cv2.COLOR_GRAY2RGB)
    det = CAM.detect_aruco(rgb_1080, K_1080, [0.0] * 5, "DICT_4X4_50", 0, size_m)
    assert det is not None
    p_jit.append(det["p_cam"])

tr_j = CAM.trimmed_points(p_jit, min_keep=4)
err_1080_mm = CAM.vnorm(CAM.vsub(tr_j["mean"], tvec_m.tolist())) * 1000.0
assert err_1080_mm < 1.5 and tr_j["jitter_mm"] < 1.0, (err_1080_mm, tr_j)

# Verify UsbWristSource.detect_marker auto-scales K_720 when frame is 1920x1080:
ws = W49.UsbWristSource(intrinsics={"K": K_720, "dist": [0.0] * 5, "model_type": W49.DIST_MODELS[1]})
ws.rgb = rgb_1080
det_scaled = ws.detect_marker({"dict": "DICT_4X4_50", "id": 0, "size_mm": 24.0})
assert det_scaled is not None
err_scaled_mm = CAM.vnorm(CAM.vsub(det_scaled["p_cam"], tvec_m.tolist())) * 1000.0
assert err_scaled_mm < 1.5, err_scaled_mm
print(f"[W2] Full-res ROI ArUco on 1920x1080 24mm marker: err={err_1080_mm:.2f} mm, jitter={tr_j['jitter_mm']:.2f} mm; "
      f"K_720->1080 auto-scale err={err_scaled_mm:.2f} mm")

# ------------------------------------------------------------------ W3: Pure-translation probes + Look-At Dome planner
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
LIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708)]
q0_rad = [math.radians(v) for v in (0.0, 30.0, 0.0, 80.0, 0.0, 40.0, 0.0)]
fk_fn = lambda qr: [list(map(float, r)) for r in CS.fk(U, dict(zip(J, qr)), "base_link", "link7")]

A0 = fk_fn(q0_rad)
T_bc0_true = CAM.T_from([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], [-0.450, 0.020, 0.262])
X_true = CAM.mat_mul(CAM.T_inv(A0), T_bc0_true)
B0_true = CAM.T_from([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], [0.014, -0.092, 0.262])
Y_true = CAM.mat_mul(T_bc0_true, B0_true)

fast_ik = CAM.build_fast_jacobian_ik(fk_fn, q0_rad, LIM, soft_rad=math.radians(3.0))
p0, R0 = CAM.T_t(A0), CAM.T_R(A0)
probe_samples = []
for idx_p, (ax, step) in enumerate([(None, 0.0), (0, 0.038), (1, 0.038), (2, 0.032)]):
    if ax is None:
        Ap = A0
    else:
        tgt = list(p0)
        tgt[ax] += step
        qp = fast_ik(q0_rad, tgt, R0)
        assert qp is not None
        Ap = fk_fn(qp)
    Bp = CAM.mat_mul(CAM.T_inv(CAM.mat_mul(Ap, X_true)), Y_true)
    # Corrupt B_i rotation by 55 deg planar flip to verify pure-translation probe solver is 100% immune
    R_corrupt = CAM.mat_mul(CAM.T_R(Bp), CAM.rodrigues([math.radians(55.0 * (1 if idx_p % 2 else -1)), 0.3, -0.2]))
    probe_samples.append({
        "T_base_flange": Ap,
        "p_cam": CAM.T_t(Bp),
        "T_cam_marker": CAM.T_from(R_corrupt, CAM.T_t(Bp)),
    })

eih_est = W49.estimate_eih_from_probes(probe_samples, size_mm=24.0)
assert eih_est is not None and eih_est["mean_mm"] < 2.0, eih_est
t0_p = time.time()
tgts, st_p = W49.plan_eih_auto_targets(fk_fn, None, q0_rad, LIM, eih_est, K_1080, n=12, size_m=0.024, seed=42)
dt_plan_ms = (time.time() - t0_p) * 1000.0
assert len(tgts) == 12 and dt_plan_ms < 260.0, (len(tgts), dt_plan_ms, st_p)
assert st_p["span_m"] >= 0.15 and st_p["rot_max_deg"] >= 28.0, st_p

# Verify all 12 planned poses keep the true marker inside the central sweet spot (off-axis <= 0.60, <= 31 deg)
off_angles = []
for t in tgts:
    B_t = CAM.mat_mul(CAM.T_inv(CAM.mat_mul(t["T"], X_true)), Y_true)
    pc = CAM.T_t(B_t)
    off_angles.append(math.hypot(pc[0], pc[1]) / max(0.05, pc[2]))
assert max(off_angles) <= 0.62, (max(off_angles), off_angles)
print(f"[W3] Pure-translation probes (fit {eih_est['mean_mm']:.2f} mm) + Look-At Dome planner: 12 poses in {dt_plan_ms:.1f} ms, "
      f"span={st_p['span_m']*100:.1f} cm, rot={st_p['rot_max_deg']:.1f}°, max off-axis={math.degrees(math.atan(max(off_angles))):.1f}°")

# ------------------------------------------------------------------ W4: Lever-arm bound + Monocular scale self-calib + LOO outlier rejection
rng = np.random.default_rng(56)
samples_bad = []
all_poses = [A0] + [t["T"] for t in tgts]
scale_bias = 1.22   # e.g. printed marker 19.7 mm instead of 24 mm, or wide-angle fx off by 22%
for k, A in enumerate(all_poses):
    AX = CAM.mat_mul(A, X_true)
    p_true = CAM.T_apply(CAM.T_inv(AX), CAM.T_t(Y_true))
    p_meas = [v * scale_bias + float(rng.normal(0, 0.0005)) for v in p_true]
    if k in (3, 8):
        # Inject 2 severe peripheral / inner-bit outliers (45 mm jump)
        p_meas[0] -= 0.045
        p_meas[2] += 0.040
    R_b = CAM.T_R(CAM.T_inv(AX))
    if k % 3 == 1:
        R_b = CAM.mat_mul(R_b, CAM.rodrigues([math.radians(38.0), -0.25, 0.1]))
    samples_bad.append({
        "q_deg": [0.0] * 7,
        "p_cam": p_meas,
        "T_cam_marker": CAM.T_from(R_b, p_meas),
        "T_base_flange": A,
    })

res_cal = W49.calibrate_eye_in_hand(samples_bad, size_mm=24.0)
assert res_cal["ok"] and res_cal["good"], res_cal
assert 3 in res_cal["rejected"] and 8 in res_cal["rejected"], res_cal["rejected"]
assert abs(res_cal["pnp_scale"] - (1.0 / scale_bias)) < 0.025, res_cal["pnp_scale"]
err_cam_mm = CAM.vnorm(CAM.vsub(res_cal["cam_in_flange"], CAM.T_t(X_true))) * 1000.0
assert err_cam_mm < 2.5 and res_cal["loo_max_mm"] < 3.0, (err_cam_mm, res_cal)
# Verify wrist_marker_error_mm uses pnp_scale seamlessly on inlier poses
err_chk_mm = W49.wrist_marker_error_mm(res_cal, all_poses[0], samples_bad[0]["p_cam"])
assert err_chk_mm < 2.0, err_chk_mm
print(f"[W4] Scale-self-calibrated EIH with 1.22x bias + 2x45mm outliers: pnp_scale={res_cal['pnp_scale']:.4f} "
      f"(true {1.0/scale_bias:.4f}), rejected={res_cal['rejected']}, link7->cam err={err_cam_mm:.2f} mm, "
      f"LOO max={res_cal['loo_max_mm']:.2f} mm (good=True)")

print("ALL V3.56 TESTS PASSED")
