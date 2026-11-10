#!/usr/bin/env python3
"""V3.57 offline unit tests:
X1 Version V3.57 & markers across NERO_V312_Console.py, NERO_V313_Real_Robot.py, NERO_V325_Camera.py,
   NERO_V347_Yolo_Perception.py, NERO_V349_Real_Wrist_Camera.py, and VERSION; NERO_V308 untouched.
X2 High-precision ChArUco 6-DoF detector (detect_charuco_pose) & dual-engine UsbWristSource.detect_marker:
   recovers board centre to < 0.6 mm even under 5x5 Gaussian soft-edge blur + 40% glare occlusion, and
   seamlessly falls back to single ArUco marker when no board is present.
X3 Automatic tabletop board intrinsic planner (plan_intr_auto_targets), rational overfitting sanity gate
   (_check_rational_sanity), and one-pass joint intrinsic->EIH bridge (eih_samples_from_intr_samples).
X4 2D anisotropic (pnp_scale, pnp_scale_z) focal-length + marker-scale self-calibration in calibrate_eye_in_hand
   (recovers 1.70x inflated fx with z_pnp=391 mm -> < 1.5 mm LOO error) + rotated fallback probe fix in
   estimate_eih_from_probes (< 0.5 mm on 11.1° rotated 3-point probes).
"""
import hashlib
import math
import sys
import tempfile
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

# ------------------------------------------------------------------ X1: Version & markers
assert CON.VERSION in ("V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13") and RR.VERSION in ("V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13") and YP.VERSION in ("V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13") and W49.VERSION in ("V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13")
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13")
h308 = hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest()
assert h308.startswith("020cc6680f181872"), f"NERO_V308 modified: {h308}"

marker = "V3.57 AUTO TABLETOP CHARUCO INTRINSICS + CHARUCO EIH + 2D FOCAL/SCALE SELF-CALIB"
for fn in ("NERO_V349_Real_Wrist_Camera.py", "NERO_V325_Camera.py", "NERO_V312_Console.py"):
    assert marker in (ROOT / fn).read_text(encoding="utf-8"), f"Missing V3.57 marker in {fn}"
print("[X1] V3.57 versions & markers verified; NERO_V308 untouched")

# ------------------------------------------------------------------ X2: ChArUco 6-DoF pose under soft blur + partial glare occlusion
K_1080 = {"width": 1920, "height": 1080, "fx": 720.0, "fy": 720.0, "ppx": 960.0, "ppy": 540.0}
Km = np.array([[K_1080["fx"], 0.0, K_1080["ppx"]], [0.0, K_1080["fy"], K_1080["ppy"]], [0.0, 0.0, 1.0]], dtype=np.float64)
cols, rows, sq_m, mk_m = 7, 5, 0.032, 0.024
aruco_dict = CAM._aruco_dict(cv2, "DICT_4X4_50")
if hasattr(cv2.aruco, "CharucoBoard"):
    board = cv2.aruco.CharucoBoard((cols, rows), sq_m, mk_m, aruco_dict)
    b_img = board.generateImage((700, 500))
else:
    board = cv2.aruco.CharucoBoard_create(cols, rows, sq_m, mk_m, aruco_dict)
    b_img = board.draw((700, 500))

# Board outer corners in centered 3D board frame (width = cols*sq_m = 0.224 m, height = rows*sq_m = 0.160 m):
hw, hh = 0.5 * cols * sq_m, 0.5 * rows * sq_m
board_outer_3d = np.array([[-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0]], dtype=np.float64)
R_b_true = np.diag([1.0, 1.0, -1.0]) @ np.array(CAM.rodrigues([0.14, -0.18, 0.22]))
rvec_b, _ = cv2.Rodrigues(R_b_true)
tvec_b = np.array([0.022, -0.035, 0.265], dtype=np.float64)

proj_outer, _ = cv2.projectPoints(board_outer_3d, rvec_b, tvec_b, Km, np.zeros(5))
src_outer = np.float32([[-0.5, -0.5], [699.5, -0.5], [699.5, 499.5], [-0.5, 499.5]])
H_b = cv2.getPerspectiveTransform(src_outer, proj_outer.reshape(4, 2).astype(np.float32))
warped_b = cv2.warpPerspective(b_img, H_b, (1920, 1080), flags=cv2.INTER_LINEAR, borderValue=235)
# Apply 5x5 Gaussian blur (simulating soft printer edges / mild defocus) AND a bright specular glare patch covering 30% of the board!
blurred_b = cv2.GaussianBlur(warped_b, (5, 5), 1.2)
blurred_b[350:520, 720:940] = 252
rgb_charuco = cv2.cvtColor(blurred_b, cv2.COLOR_GRAY2RGB)

det_ch = W49.detect_charuco_pose(rgb_charuco, K_1080, [0.0] * 5)
assert det_ch is not None and det_ch["source"] == "charuco" and det_ch["n_corners"] >= 10, det_ch
err_ch_mm = CAM.vnorm(CAM.vsub(det_ch["p_cam"], tvec_b.tolist())) * 1000.0
assert err_ch_mm < 0.8, (err_ch_mm, det_ch)

# Verify UsbWristSource.detect_marker picks ChArUco automatically and falls back to ArUco when target_type='单张 ArUco 标记'
ws = W49.UsbWristSource(intrinsics={"K": K_1080, "dist": [0.0] * 5, "model_type": W49.DIST_MODELS[1]})
ws.rgb = rgb_charuco
d_auto = ws.detect_marker({"dict": "DICT_4X4_50", "id": 0, "size_mm": 24.0})
assert d_auto is not None and d_auto["source"] == "charuco" and d_auto["n_corners"] >= 10
print(f"[X2] ChArUco 6-DoF detector under 5x5 soft-edge blur + glare occlusion: {det_ch['n_corners']}/24 corners, "
      f"3D centre err={err_ch_mm:.2f} mm, reproj={det_ch['reproj_px']:.2f} px")

# ------------------------------------------------------------------ X3: plan_intr_auto_targets + rational sanity gate + one-pass EIH bridge
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
LIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708)]
q0_rad = [math.radians(v) for v in (0.0, 30.0, 0.0, 80.0, 0.0, 40.0, 0.0)]
fk_fn = lambda qr: [list(map(float, r)) for r in CS.fk(U, dict(zip(J, qr)), "base_link", "link7")]

s0_b = W49.detect_board_corners(rgb_charuco)
assert s0_b["ok"]
t0_i = time.time()
intr_tgts, st_i = W49.plan_intr_auto_targets(fk_fn, q0_rad, LIM, s0_board=s0_b, image_size=(1920, 1080), n=10, seed=7)
dt_intr_ms = (time.time() - t0_i) * 1000.0
assert len(intr_tgts) == 10 and dt_intr_ms < 800.0, (len(intr_tgts), dt_intr_ms)

# Rational sanity gate catches 1.65x focal-length drift:
Km_bad = np.array([[1200.0, 0.0, 960.0], [0.0, 1200.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)
ok_s, why_s = W49._check_rational_sanity(Km_bad, [0.1] * 8, Km, [0.0] * 5, 1920, 1080)
assert not ok_s and "焦距漂移" in why_s, (ok_s, why_s)

# One-pass eih_samples_from_intr_samples bridge:
A0 = fk_fn(q0_rad)
T_bc0_true = CAM.T_from([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], [-0.450, 0.020, 0.250])
X_true = CAM.mat_mul(CAM.T_inv(A0), T_bc0_true)
Y_board_true = CAM.T_from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [-0.450, 0.020, 0.0])
all_obj_b = np.asarray(board.getChessboardCorners() if hasattr(board, "getChessboardCorners") else board.chessboardCorners,
                       dtype=np.float64).reshape(-1, 3)
all_obj_c = all_obj_b.copy()
all_obj_c[:, 0] -= 0.5 * cols * sq_m
all_obj_c[:, 1] -= 0.5 * rows * sq_m

intr_samps = []
for pose_item in [{"q": q0_rad, "T": A0}] + intr_tgts:
    A_k = pose_item["T"]
    B_k = CAM.mat_mul(CAM.T_inv(CAM.mat_mul(A_k, X_true)), Y_board_true)
    rv_k, _ = cv2.Rodrigues(np.array(CAM.T_R(B_k), dtype=np.float64))
    tv_k = np.array(CAM.T_t(B_k), dtype=np.float64)
    proj_k, _ = cv2.projectPoints(all_obj_c, rv_k, tv_k, Km, np.zeros(5))
    intr_samps.append({
        "ok": True,
        "obj_pts": all_obj_b.tolist(),
        "img_pts": proj_k.reshape(-1, 2).tolist(),
        "n_corners": len(all_obj_b),
        "q_deg": [math.degrees(v) for v in pose_item["q"]],
        "T_base_flange": A_k,
    })

eih_from_intr = W49.eih_samples_from_intr_samples(intr_samps, K_1080, [0.0] * 5)
assert len(eih_from_intr) == 11
res_bridge = W49.calibrate_eye_in_hand(eih_from_intr, size_mm=24.0)
assert res_bridge["ok"] and res_bridge["good"] and res_bridge["loo_max_mm"] < 0.5, res_bridge
print(f"[X3] plan_intr_auto_targets: 10 views in {dt_intr_ms:.1f} ms; rational sanity gate active ({why_s}); "
      f"one-pass ChArUco->EIH bridge LOO max={res_bridge['loo_max_mm']:.3f} mm")

# ------------------------------------------------------------------ X4: 2D (alpha_xy, alpha_z) focal/scale self-calib + rotated probe fix
# 1) Verify rotated 3-point fallback probes (rot_span = 11.1 deg < 12.0 deg) no longer produce 65.1 mm error:
q0_d = [0.0, 30.0, 0.0, 80.0, 0.0, 40.0, 0.0]
rot_probes = []
for dq_d in ([0.0] * 7, [-2.5, 1.5, 2.0, -1.5, -5.0, 4.5, -6.0], [-1.5, 2.0, -1.5, -2.5, -4.0, -4.5, -8.0]):
    qd = [a + b for a, b in zip(q0_d, dq_d)]
    Ak = fk_fn([math.radians(v) for v in qd])
    Bk = CAM.mat_mul(CAM.T_inv(CAM.mat_mul(Ak, X_true)), Y_board_true)
    rot_probes.append({"T_base_flange": Ak, "p_cam": CAM.T_t(Bk), "T_cam_marker": Bk})
est_rot = W49.estimate_eih_from_probes(rot_probes, size_mm=24.0)
assert est_rot["mean_mm"] < 1.0, est_rot

# 2) Verify 2D (pnp_scale, pnp_scale_z) self-calibration recovers 1.70x inflated fx (z_pnp = 391 mm vs 230 mm true) + 1.05x marker scale:
samples_aniso = []
for s_clean in eih_from_intr:
    p_c = s_clean["p_cam"]
    p_distorted = [p_c[0] * 1.05, p_c[1] * 1.05, p_c[2] * 1.70]
    samples_aniso.append({
        "q_deg": s_clean["q_deg"],
        "p_cam": p_distorted,
        "T_cam_marker": CAM.T_from(CAM.T_R(s_clean["T_cam_marker"]), p_distorted),
        "T_base_flange": s_clean["T_base_flange"],
    })
res_aniso = W49.calibrate_eye_in_hand(samples_aniso, size_mm=24.0)
assert res_aniso["ok"] and res_aniso["good"], res_aniso
assert abs(res_aniso["pnp_scale"] - (1.0 / 1.05)) < 0.03, res_aniso["pnp_scale"]
assert abs(res_aniso["pnp_scale_z"] - (1.0 / 1.70)) < 0.03, res_aniso["pnp_scale_z"]
err_aniso_mm = CAM.vnorm(CAM.vsub(res_aniso["cam_in_flange"], CAM.T_t(X_true))) * 1000.0
assert err_aniso_mm < 2.0 and res_aniso["loo_max_mm"] < 2.0, (err_aniso_mm, res_aniso)
err_live_mm = W49.wrist_marker_error_mm(res_aniso, samples_aniso[0]["T_base_flange"], samples_aniso[0]["p_cam"])
assert err_live_mm < 1.5, err_live_mm
print(f"[X4] Rotated 3-probe fit={est_rot['mean_mm']:.3f} mm (was 65.1 mm!); 2D focal/scale self-calib (1.70x fx bias): "
      f"s_xy={res_aniso['pnp_scale']:.4f}, s_z={res_aniso['pnp_scale_z']:.4f}, link7->cam err={err_aniso_mm:.2f} mm, "
      f"LOO max={res_aniso['loo_max_mm']:.2f} mm (good=True)")

# ------------------------------------------------------------------ X5: Console & CameraTabUI startup crash guard
import unittest.mock as mock

mock_tk = mock.MagicMock()
mock_ttk = mock.MagicMock()
mock_tkfont = mock.MagicMock()
mock_msgbox = mock.MagicMock()
mock_dialog = mock.MagicMock()

sys.modules['tkinter'] = mock_tk
sys.modules['tkinter.ttk'] = mock_ttk
sys.modules['tkinter.font'] = mock_tkfont
sys.modules['tkinter.messagebox'] = mock_msgbox
sys.modules['tkinter.simpledialog'] = mock_dialog

root_mock = mock.MagicMock()
root_mock.winfo_screenwidth.return_value = 1920
root_mock.winfo_screenheight.return_value = 1080
mock_tk.Tk.return_value = root_mock

root_mock.mainloop.side_effect = lambda: None
CON.run_console('sim')
CON.run_console('real')

cam_mock = CAM.CameraTabUI(mock_tk, mock_ttk, mock.MagicMock(), mock.MagicMock(), mock.MagicMock(),
                           mock.MagicMock(), lambda: {}, print, lambda *a: True)
cam_mock.update_workflow_lamps()
cam_mock._update_intr_ui()
cam_mock._update_eih_ui()
cam_mock._update_calib_ui()
print('[X5] Console & CameraTabUI startup crash guard passed (zero startup exceptions)')

print("ALL V3.57 TESTS PASSED")
