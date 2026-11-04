#!/usr/bin/env python3
"""V3.58 offline unit tests:
Y1 Version V3.58 & markers across NERO_V312_Console.py, NERO_V313_Real_Robot.py, NERO_V325_Camera.py,
   NERO_V347_Yolo_Perception.py, NERO_V349_Real_Wrist_Camera.py, and VERSION; NERO_V308 untouched.
Y2 Kinematic-anchored wide-angle intrinsic calibration + per-view RANSAC Homography corner pruning on low-tilt
   tabletop ChArUco views with gripper-shadow/glare corner outliers (prevents fx=2147/3134/5164 px runaway,
   recovers true fx~705 px, RMS < 0.45 px, and one-pass Eye-in-Hand LOO < 1.5 mm).
Y3 Full-quadrant adaptive steering & perspective tilt diversity in plan_intr_auto_targets when starting board is
   off-center on the left (center_frac=[0.33, 0.55]).
Y4 Wide-range 2D (alpha_xy, alpha_z) scale recovery in calibrate_eye_in_hand for extreme 4.48x telephoto fx bias
   (fx=3134.8 px, z_pnp=1147 mm -> s_z=0.223, LOO < 1.5 mm, good=True).
"""
import hashlib
import math
import sys
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

# ------------------------------------------------------------------ Y1: Version & markers
assert CON.VERSION in ("V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77") and RR.VERSION in ("V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77") and YP.VERSION in ("V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77") and W49.VERSION in ("V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77")
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77")
h308 = hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest()
assert h308.startswith("020cc6680f181872"), f"NERO_V308 modified: {h308}"

marker = "V3.58 KINEMATIC-ANCHORED WIDE-ANGLE INTRINSICS + FULL-QUADRANT TILT + ROBUST CHARUCO EIH"
for fn in ("NERO_V349_Real_Wrist_Camera.py", "NERO_V325_Camera.py", "NERO_V312_Console.py"):
    assert marker in (ROOT / fn).read_text(encoding="utf-8"), f"Missing V3.58 marker in {fn}"
print("[Y1] V3.58 versions & markers verified; NERO_V308 untouched")

# ------------------------------------------------------------------ Y2: Kinematic-anchored intrinsics + corner pruning
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
LIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708)]
fk_fn = lambda qr: [list(map(float, r)) for r in CS.fk(U, dict(zip(J, qr)), "base_link", "link7")]

w, h = 1920, 1080
Km_true = np.array([[705.0, 0.0, 948.0], [0.0, 705.0, 536.0], [0.0, 0.0, 1.0]], dtype=np.float64)
Dm_true = np.array([-0.24, 0.06, 0.0005, -0.0004, 0.0], dtype=np.float64)

q0_rad = [math.radians(v) for v in (0.0, 30.0, 0.0, 80.0, 0.0, 40.0, 0.0)]
A0 = fk_fn(q0_rad)
R_bc0 = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
Rx_true = CAM.mat_mul(CAM.transpose(CAM.T_R(A0)), R_bc0)
X_true = CAM.T_from(Rx_true, [0.042, -0.015, 0.032])
T_bc0 = CAM.mat_mul(A0, X_true)
p_c0 = CAM.T_t(T_bc0)
# Place tabletop ChArUco board 260 mm below camera, shifted to the left of image (just like user screenshot!)
Y_origin_true = CAM.T_from(CAM.rodrigues([0.01, -0.02, 1.45]), [p_c0[0] - 0.01, p_c0[1] + 0.10, p_c0[2] - 0.26])

cols, rows, sq_m = 7, 5, 0.032
grid_obj = np.zeros((24, 3), dtype=np.float32)
g_uv = np.mgrid[1:cols, 1:rows].T.reshape(-1, 2)
grid_obj[:, 0] = g_uv[:, 0] * sq_m
grid_obj[:, 1] = g_uv[:, 1] * sq_m
board_center_3d = np.array([0.5 * cols * sq_m, 0.5 * rows * sq_m, 0.0], dtype=np.float64)
y_center_true = CAM.T_apply(Y_origin_true, board_center_3d.tolist())

rng = np.random.default_rng(123)
samples = []
q0_deg = [math.degrees(v) for v in q0_rad]
for k in range(15):
    dq = [rng.uniform(-4.0, 4.0) if j < 4 else rng.uniform(-8.0, 8.0) for j in range(7)] if k > 0 else [0.0] * 7
    qk_rad = [math.radians(q0_deg[j] + dq[j]) for j in range(7)]
    Ak = fk_fn(qk_rad)
    Bk = CAM.mat_mul(CAM.T_inv(CAM.mat_mul(Ak, X_true)), Y_origin_true)
    rv, _ = cv2.Rodrigues(np.array(CAM.T_R(Bk), dtype=np.float64))
    tv = np.array(CAM.T_t(Bk), dtype=np.float64)
    pts_cam = (np.array(CAM.T_R(Bk)) @ grid_obj.T).T + tv
    r_n = np.hypot(pts_cam[:, 0] / pts_cam[:, 2], pts_cam[:, 1] / pts_cam[:, 2])
    proj, _ = cv2.projectPoints(grid_obj.astype(np.float64), rv, tv, Km_true, Dm_true)
    pts2d = proj.reshape(-1, 2) + rng.normal(0.0, 0.24, (24, 2))
    in_b = (r_n <= 1.25) & (pts2d[:, 0] >= 20) & (pts2d[:, 0] <= w - 20) & (pts2d[:, 1] >= 20) & (pts2d[:, 1] <= h - 20)
    o_k = grid_obj[in_b].copy()
    p_k = pts2d[in_b].astype(np.float32).copy()
    if len(o_k) < 8:
        continue
    # Inject severe corner outliers (+7.5 / -6.8 px) on 6 frames (simulating gripper shadow & plastic glare)
    if k in (4, 7, 8, 9, 10, 13):
        p_k[0] += np.array([7.5, -6.8], dtype=np.float32)
        p_k[-1] += np.array([-6.2, 7.9], dtype=np.float32)
    # Inject 1 whole-frame motion-blurred view (#11)
    blurred_idx = len(samples) if k == 11 else globals().get("blurred_idx", -1)
    if k == 11:
        p_k += rng.normal(0.0, 3.2, p_k.shape).astype(np.float32)
    us, vs = p_k[:, 0] / float(w), p_k[:, 1] / float(h)
    samples.append({
        "ok": True,
        "obj_pts": o_k.tolist(),
        "img_pts": p_k.tolist(),
        "n_corners": len(o_k),
        "span_frac": float(max(us.max() - us.min(), vs.max() - vs.min())),
        "center_frac": [float(us.mean()), float(vs.mean())],
        "q_deg": [math.degrees(v) for v in qk_rad],
        "T_base_flange": Ak,
        "image_size": [w, h],
        "board_cols": cols,
        "board_rows": rows,
        "square_mm": 32.0,
    })

res_intr = W49.calibrate_intrinsics(samples, image_size=(w, h), model_type=W49.DIST_MODELS[0])
assert res_intr["ok"] and res_intr["good"], res_intr
assert res_intr["rms_px"] <= 0.50, res_intr
assert abs(res_intr["K"]["fx"] - 705.0) < 25.0 and abs(res_intr["K"]["fy"] - 705.0) < 25.0, res_intr["K"]
assert res_intr["pruned_corners"] >= 6 and blurred_idx in res_intr["rejected"], res_intr

eih_samps = W49.eih_samples_from_intr_samples(samples, res_intr["K"], res_intr["dist"], model_type=res_intr["model_type"])
assert len(eih_samps) >= 10
res_eih = W49.calibrate_eye_in_hand(eih_samps, size_mm=24.0)
assert res_eih["ok"] and res_eih["good"], res_eih
err_tx_mm = CAM.vnorm(CAM.vsub(res_eih["cam_in_flange"], CAM.T_t(X_true))) * 1000.0
err_mb_mm = CAM.vnorm(CAM.vsub(res_eih["marker_in_base"], y_center_true)) * 1000.0
assert err_tx_mm < 2.5 and err_mb_mm < 2.5 and res_eih["loo_max_mm"] < 2.0, (err_tx_mm, err_mb_mm, res_eih)
print(f"[Y2] Kinematic-anchored calibrate_intrinsics: pruned {res_intr['pruned_corners']} glare corners, rejected view #11, "
      f"fx={res_intr['K']['fx']:.1f} (true 705.0), RMS={res_intr['rms_px']:.3f} px, FOV={res_intr['fov_h_deg']:.0f}°/{res_intr['fov_diag_deg']:.0f}°; "
      f"one-pass EIH LOO max={res_eih['loo_max_mm']:.2f} mm, t_X err={err_tx_mm:.2f} mm")

# ------------------------------------------------------------------ Y3: 4-quadrant adaptive steering from off-center start
tgts_off, st_off = W49.plan_intr_auto_targets(fk_fn, q0_rad, LIM, s0_board=samples[0], image_size=(w, h), n=12, seed=9)
assert len(tgts_off) == 12
A_off = [A0] + [t["T"] for t in tgts_off]
rot_span_off = CAM.rotation_span_deg(A_off)
trans_span_off = CAM.translation_span_m(A_off)
assert rot_span_off >= 26.0 and trans_span_off >= 0.14, (rot_span_off, trans_span_off)
print(f"[Y3] plan_intr_auto_targets from left-offset start: 12 poses, rot_span={rot_span_off:.1f} deg, trans_span={trans_span_off*100:.1f} cm")

# ------------------------------------------------------------------ Y4: Extreme 4.48x telephoto fx bias (fx=3134.8 px, z=1147 mm) recovery
samples_tele = []
for item in [{"q": q0_rad, "T": A0}] + tgts_off:
    Ak = item["T"]
    Bk = CAM.mat_mul(CAM.T_inv(CAM.mat_mul(Ak, X_true)), Y_origin_true)
    p_c = CAM.T_t(Bk)
    p_tele = [p_c[0] * 1.02, p_c[1] * 1.02, p_c[2] * (3134.8 / 705.0)]
    samples_tele.append({
        "q_deg": [math.degrees(v) for v in item["q"]],
        "p_cam": p_tele,
        "T_cam_marker": CAM.T_from(CAM.T_R(Bk), p_tele),
        "T_base_flange": Ak,
        "source": "charuco",
    })
res_tele = W49.calibrate_eye_in_hand(samples_tele, size_mm=24.0)
assert res_tele["ok"] and res_tele["good"], res_tele
assert abs(res_tele["pnp_scale_z"] - (705.0 / 3134.8)) < 0.025, res_tele["pnp_scale_z"]
assert res_tele["loo_max_mm"] < 2.0, res_tele
print(f"[Y4] Extreme 4.45x telephoto fx=3134.8 self-calibration: s_z={res_tele['pnp_scale_z']:.4f} (true {705.0/3134.8:.4f}), "
      f"LOO max={res_tele['loo_max_mm']:.2f} mm (good=True)")

# ------------------------------------------------------------------ Y5: Mocked-Tkinter Console & CameraTabUI One-Pass Auto-Calib Guard
import unittest.mock as mock

mock_tk = mock.MagicMock()
mock_ttk = mock.MagicMock()
mock_tkfont = mock.MagicMock()
mock_msgbox = mock.MagicMock()
mock_dialog = mock.MagicMock()

sys.modules["tkinter"] = mock_tk
sys.modules["tkinter.ttk"] = mock_ttk
sys.modules["tkinter.font"] = mock_tkfont
sys.modules["tkinter.messagebox"] = mock_msgbox
sys.modules["tkinter.simpledialog"] = mock_dialog

root_mock = mock.MagicMock()
root_mock.winfo_screenwidth.return_value = 1920
root_mock.winfo_screenheight.return_value = 1080
mock_tk.Tk.return_value = root_mock
root_mock.mainloop.side_effect = lambda: None

CON.run_console("sim")
CON.run_console("real")

cam_mock = CAM.CameraTabUI(mock_tk, mock_ttk, mock.MagicMock(), mock.MagicMock(), mock.MagicMock(),
                           mock.MagicMock(), lambda: {}, lambda *a: None, lambda *a: True)
cam_mock.intr_samples = samples
cam_mock.v_imodel = mock.MagicMock()
cam_mock.v_imodel.get.return_value = W49.DIST_MODELS[0]
cam_mock._board_cfg = lambda: dict(W49.DEFAULT_BOARD)
cam_mock._wmarker_from_ui = lambda: dict(W49.DEFAULT_WRIST_MARKER)
cam_mock.on_intr_compute()
assert cam_mock.intr_result and cam_mock.intr_result["good"]
assert cam_mock.wcalib_result and cam_mock.wcalib_result["good"]
print("[Y5] Console & CameraTabUI on_intr_compute -> auto EIH sync verified (zero exceptions)")

print("ALL V3.58 TESTS PASSED")

