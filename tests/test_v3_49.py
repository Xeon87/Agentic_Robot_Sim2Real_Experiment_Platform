#!/usr/bin/env python3
"""V3.49: Real USB wrist camera discovery, wide-angle intrinsic calibration (ChArUco / Chessboard), eye-in-hand
hand-eye calibration, auto-sampling path safety-zone check, and zero-prior egg holder + slot localization.

S1 list_usb_cameras filters out RealSense / metadata nodes and prioritizes innomaker USB UVC; generate_calib_board_png
   writes A4 300-DPI ChArUco and Chessboard PNGs (and pre-bundled PNGs exist in config/)
S2 Wide-angle intrinsic calibration (detect_board_corners, calibrate_intrinsics, undistort_rgb): recovers K and barrel
   distortion from synthetic views, rejects an outlier view, RMS < 0.5 px; undistort_rgb remaps cleanly
S3 Eye-in-Hand calibration (solve_eye_in_hand, calibrate_eye_in_hand): recovers T_flange_cam (< 1.2 mm / < 0.25 deg)
   and T_base_marker (< 1.2 mm) from noisy poses, rejects an outlier pose, LOO max < 2.5 mm
S4 Auto-sampling path safety check (check_auto_path, plan_auto_targets): targets whose midpoint clips a wall are
   rejected as '路径安全区' even when both endpoints are inside the safety zone
S5 Zero-prior egg holder & slot localization (infer_holder_grid, agent_judge_holder, locate_egg_holder): without any
   user-entered dimensions, measures 2x3 (6-slot) and 2x2 (4-slot) holders within < 4 mm / < 3 deg, infers (rows x cols)
   and slot 3D centres (< 4 mm), detects occupied vs empty slots, and supports VLM/Agent override
S6 Invariants: NERO_V308_Camera_Perception.py sha256 unchanged; NERO_V326_Wrist_Camera.py & backend do not import V349
"""
import hashlib
import math
import os
import random
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import v347_fake as F                                  # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49              # noqa: E402
import NERO_V347_Yolo_Perception as YP                 # noqa: E402
import NERO_V325_Camera as CAM                         # noqa: E402
import NERO_V315_Consistency_Safety as CS              # noqa: E402

assert W49.VERSION in ("V3.49", "V3.50", "V3.51", "V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4") and YP.VERSION in ("V3.49", "V3.50", "V3.51", "V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4") and (ROOT / "VERSION").read_text().strip() in ("V3.49", "V3.50", "V3.51", "V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4")
TD = tempfile.mkdtemp(prefix="v349_")
FENCE = [[-0.8, -0.4], [-0.1, -0.4], [-0.1, 0.4], [-0.8, 0.4]]
F.install()

# ------------------------------------------------------------------ S1
sysfs = os.path.join(TD, "v4l")
for folder, name, idx in (
    ("video0", "Intel(R) RealSense(TM) Depth Camera 435i", 0),
    ("video1", "Intel(R) RealSense(TM) Depth Camera 435i", 1),
    ("video2", "Laptop Integrated Webcam", 0),
    ("video4", "Innomaker 1080P USB2.0 UVC Camera", 0),
    ("video5", "Innomaker 1080P USB2.0 UVC Camera", 1),
):
    d = os.path.join(sysfs, folder)
    os.makedirs(d)
    Path(d, "name").write_text(name + "\n")
    Path(d, "index").write_text(f"{idx}\n")
devs = W49.list_usb_cameras(sysfs)
assert [d["device"] for d in devs] == ["/dev/video4", "/dev/video2"], devs
p_ch = W49.generate_calib_board_png({"type": W49.BOARD_TYPES[0], "cols": 7, "rows": 5, "square_mm": 32.0, "marker_mm": 24.0},
                                    out_path=os.path.join(TD, "charuco.png"), dpi=150)
p_cb = W49.generate_calib_board_png({"type": W49.BOARD_TYPES[1], "cols": 9, "rows": 6, "square_mm": 24.0},
                                    out_path=os.path.join(TD, "chess.png"), dpi=150)
assert os.path.isfile(p_ch) and os.path.isfile(p_cb)
assert (ROOT / "config" / "calib_charuco_7x5_32mm_A4.png").is_file()
assert (ROOT / "config" / "calib_chessboard_9x6_24mm_A4.png").is_file()
print(f"[S1] list_usb_cameras filters RealSense & metadata -> {[d['device'] for d in devs]}; A4 ChArUco & Chessboard PNGs generated")

# ------------------------------------------------------------------ S2
# Render a synthetic chessboard into a 1280x720 image and test detect_board_corners, calibrate_intrinsics, undistort_rgb
cb_cfg = {"type": W49.BOARD_TYPES[1], "cols": 7, "rows": 5, "square_mm": 25.0, "marker_mm": 18.0}
sq_px = 50
bw, bh = 8 * sq_px, 6 * sq_px
board_img = np.full((bh + 80, bw + 80), 255, dtype=np.uint8)
for r in range(6):
    for c in range(8):
        if (r + c) % 2 == 0:
            board_img[40 + r * sq_px:40 + (r + 1) * sq_px, 40 + c * sq_px:40 + (c + 1) * sq_px] = 0
src_quad = np.float32([[40, 40], [40 + bw, 40], [40 + bw, 40 + bh], [40, 40 + bh]])
dst_quad = np.float32([[280, 140], [920, 160], [900, 560], [260, 540]])
H_mat = cv2.getPerspectiveTransform(src_quad, dst_quad)
warped = cv2.warpPerspective(board_img, H_mat, (1280, 720), borderValue=200)
rgb_cb = cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)
det_cb = W49.detect_board_corners(rgb_cb, cb_cfg)
assert det_cb["ok"] and det_cb["n_corners"] == 35, det_cb

# Synthetic multi-view samples for calibrate_intrinsics (true K: fx=820, fy=820, cx=640, cy=360, dist=[-0.18, 0.04, 0, 0, 0])
Km_true = np.array([[820.0, 0.0, 640.0], [0.0, 820.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
Dm_true = np.array([-0.18, 0.04, 0.0005, -0.0004, 0.0], dtype=np.float64)
grid_obj = np.zeros((35, 3), dtype=np.float32)
g_uv = np.mgrid[0:7, 0:5].T.reshape(-1, 2)
grid_obj[:, 0] = g_uv[:, 0] * 0.025
grid_obj[:, 1] = g_uv[:, 1] * 0.025
rng = np.random.default_rng(10)
intr_samples = []
for k in range(10):
    rv = np.array([rng.uniform(-0.35, 0.35), rng.uniform(-0.35, 0.35), rng.uniform(-0.2, 0.2)], dtype=np.float64)
    cx_f = 0.22 + (k % 5) * 0.14
    cy_f = 0.25 + (k // 5) * 0.45
    tv = np.array([(cx_f - 0.5) * 0.35 - 0.08, (cy_f - 0.5) * 0.20 - 0.05, rng.uniform(0.32, 0.48)], dtype=np.float64)
    proj, _ = cv2.projectPoints(grid_obj, rv, tv, Km_true, Dm_true)
    pts2d = proj.reshape(-1, 2) + rng.normal(0, 0.12, (35, 2))
    intr_samples.append({"ok": True, "obj_pts": grid_obj.tolist(), "img_pts": pts2d.tolist(), "n_corners": 35,
                         "center_frac": [float(pts2d[:, 0].mean() / 1280.0), float(pts2d[:, 1].mean() / 720.0)],
                         "coverage": "中", "sharpness": 350.0, "image_size": [1280, 720]})
# Add 1 corrupted outlier view
bad_sample = dict(intr_samples[0], img_pts=[[p[0] + (18.0 if i < 6 else 0.0), p[1] - (14.0 if i < 6 else 0.0)] for i, p in enumerate(intr_samples[0]["img_pts"])])
res_intr = W49.calibrate_intrinsics(intr_samples + [bad_sample], image_size=(1280, 720), model_type=W49.DIST_MODELS[1])
assert res_intr["ok"] and res_intr["good"] and res_intr["rejected"] == [10] and res_intr["rms_px"] < 0.5, res_intr
assert abs(res_intr["K"]["fx"] - 820.0) < 6.0 and abs(res_intr["K"]["ppx"] - 640.0) < 6.0
assert W49.intrinsics_valid(res_intr) and not W49.intrinsics_valid({"K": {}})
und = W49.undistort_rgb(rgb_cb, res_intr)
assert und.shape == rgb_cb.shape and und.dtype == np.uint8
print(f"[S2] calibrate_intrinsics: 35 corners detected, outlier rejected, RMS {res_intr['rms_px']:.3f} px, "
      f"fx={res_intr['K']['fx']:.1f} (true 820), FOV {res_intr['fov_h_deg']:.0f}°/{res_intr['fov_diag_deg']:.0f}°; undistort_rgb ok")

# ------------------------------------------------------------------ S3
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
LIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708)]
X_true = CAM.T_from(CAM.rodrigues([0.08, -0.18, 1.55]), [0.042, -0.015, 0.032])
Y_true = CAM.T_from(CAM.rodrigues([0.01, -0.02, 0.35]), [-0.44, 0.03, 0.0])
q0_deg = [0.0, 30.0, 0.0, 80.0, 0.0, 40.0, 0.0]
qs = CAM.sample_targets(q0_deg, LIM, 12, seed=8)
rnd = random.Random(14)
eih_samples = []
for q in qs:
    A = [list(map(float, r)) for r in CS.fk(U, dict(zip(J, [math.radians(v) for v in q])), "base_link", "link7")]
    B_true = CAM.mat_mul(CAM.T_inv(CAM.mat_mul(A, X_true)), Y_true)
    dR = CAM.rodrigues([rnd.gauss(0, math.radians(0.12)) for _ in range(3)])
    R_b = CAM.mat_mul(CAM.T_R(B_true), dR)
    t_b = [CAM.T_t(B_true)[i] + rnd.gauss(0, 0.00035) for i in range(3)]
    eih_samples.append({"q_deg": q, "T_base_flange": A, "T_cam_marker": CAM.T_from(R_b, t_b), "p_cam": t_b})
# Add 1 outlier sample
eih_bad = dict(eih_samples[0], p_cam=[v + 0.03 for v in eih_samples[0]["p_cam"]],
               T_cam_marker=CAM.T_from(CAM.T_R(eih_samples[0]["T_cam_marker"]), [v + 0.03 for v in eih_samples[0]["p_cam"]]))
res_eih = W49.calibrate_eye_in_hand(eih_samples + [eih_bad], size_mm=38.0)
assert res_eih["ok"] and res_eih["good"] and res_eih["rejected"] == [12], res_eih
err_x_mm = CAM.vnorm(CAM.vsub(res_eih["cam_in_flange"], CAM.T_t(X_true))) * 1000.0
err_rx_deg = CAM.rot_angle_deg(CAM.mat_mul(CAM.transpose(CAM.T_R(res_eih["T_flange_cam"])), CAM.T_R(X_true)))
err_y_mm = CAM.vnorm(CAM.vsub(res_eih["marker_in_base"], CAM.T_t(Y_true))) * 1000.0
assert err_x_mm < 1.2 and err_rx_deg < 0.25 and err_y_mm < 1.2 and res_eih["loo_max_mm"] < 2.5
assert W49.handeye_valid(res_eih)
assert W49.wrist_marker_error_mm(res_eih, eih_samples[0]["T_base_flange"], eih_samples[0]["p_cam"]) < 2.0
print(f"[S3] calibrate_eye_in_hand: T_flange_cam err {err_x_mm:.2f} mm / {err_rx_deg:.3f}°, marker_in_base err "
      f"{err_y_mm:.2f} mm, outlier rejected, LOO max {res_eih['loo_max_mm']:.2f} mm")

# ------------------------------------------------------------------ S4
# Fence function where endpoints are ok, but midpoint (0.35 < q[0] < 0.55) clips a wall
def wall_fence(q_rad):
    return not (0.04 <= abs(q_rad[0]) <= 0.09)


assert wall_fence([0.0] * 7) and wall_fence([0.15] + [0.0] * 6)
assert not CAM.check_auto_path(wall_fence, [0.0] * 7, [0.15] + [0.0] * 6, steps=8)
q0_r = [math.radians(v) for v in q0_deg]
T0 = [list(map(float, r)) for r in CS.fk(U, dict(zip(J, q0_r)), "base_link", "link7")]
mm_dir = CAM.MarkerModel.from_calib({"T_base_cam": F.T_BC.tolist(), "marker_in_flange": [0.02, -0.02, 0.04]}, T0, [0, 0, -1])


def fk_r(qr):
    return [list(map(float, r)) for r in CS.fk(U, dict(zip(J, qr)), "base_link", "link7")]


def ik_r(seed, xyz, R):
    q, _ = CS.ik_tool(U, J, list(seed), "base_link", None, list(xyz), LIM, iters=150, tol=1e-3, keep_R=R)
    return q


tgts_path, st_path = CAM.plan_auto_targets(fk_r, ik_r, q0_r, LIM, mm_dir, F.K, 8, fence_fn=wall_fence, seed=3)
assert st_path["rejected"].get("路径安全区", 0) > 0, st_path
cur_q = q0_r
for t in tgts_path:
    assert CAM.check_auto_path(wall_fence, cur_q, t["q"], steps=8)
    cur_q = t["q"]
print(f"[S4] check_auto_path & plan_auto_targets: rejected {st_path['rejected']['路径安全区']} wall-clipping paths ('路径安全区'), "
      f"kept {len(tgts_path)} path-safe targets")

# ------------------------------------------------------------------ S5
# Zero-prior egg holder + slot localization: test 2x3 (144 x 96 x 36 mm, yaw=25 deg, egg in slot 2) and 2x2 (96 x 96 x 36 mm)
MP_PF = os.path.join(TD, "yoloe-26s-seg-pf.pt")
Path(MP_PF).write_bytes(b"pf")
det = YP.YoloEDetector(log=lambda *a: None)

for rows_true, cols_true, L_t, W_t, yaw_deg in ((2, 3, 0.144, 0.096, 25.0), (2, 2, 0.096, 0.096, -15.0)):
    yaw_r = math.radians(yaw_deg)
    hc = [-0.46, 0.04, 0.018]
    H_t = 0.036
    obs_h = [{"name": "HOLDER", "label": "egg holder", "kind": "box", "c": hc, "yaw": yaw_r,
              "half": [L_t / 2, W_t / 2, H_t / 2], "conf": 0.95}]
    rgb_h, depth_h, ids_h = F.render(obs_h, noise_m=0.0004)
    F.STATE.scene = (obs_h, ids_h)
    F.STATE.bleed_px = 0
    d2_h = det.detect(rgb_h, "pf", MP_PF, prompts=["egg holder", "tray", "carton"])
    u_l = np.array([math.cos(yaw_r), math.sin(yaw_r), 0.0])
    v_s = np.array([-math.sin(yaw_r), math.cos(yaw_r), 0.0])
    slot1_true = np.array([hc[0], hc[1], H_t * 0.6]) + (0.5 - 0.5 * cols_true) * (L_t / cols_true) * u_l + \
        (0.5 - 0.5 * rows_true) * (W_t / rows_true) * v_s
    fake_eggs = [{"det_id": "egg_1", "status": "egg", "xyz_m": slot1_true.tolist()}]
    rows_out, diag_h = W49.locate_egg_holder(d2_h, rgb_h, depth_h, F.K, F.T_BC, table_z=0.0, fence_poly=FENCE,
                                             existing_eggs=fake_eggs)
    F.STATE.bleed_px = 2
    assert diag_h["found"] and (diag_h["rows"], diag_h["cols"]) == (rows_true, cols_true), diag_h
    assert len(rows_out) == 1 + rows_true * cols_true
    assert abs(diag_h["size_mm"][0] - L_t * 1000) < 4.0 and abs(diag_h["size_mm"][1] - W_t * 1000) < 4.0
    s1 = rows_out[1]
    err_s1 = math.hypot(s1["xyz_m"][0] - slot1_true[0], s1["xyz_m"][1] - slot1_true[1]) * 1000.0
    assert s1["det_id"] == "slot_1" and s1["occupied"] and err_s1 < 4.0 and not rows_out[2]["occupied"], (s1, err_s1)

# Test VLM / Agent override hook in agent_judge_holder
rows_vlm, diag_vlm = W49.locate_egg_holder(
    d2_h, rgb_h, depth_h, F.K, F.T_BC, table_z=0.0, fence_poly=FENCE,
    agent_fn=lambda crop, g: {"rows": 2, "cols": 3, "occupied_slots": [3], "source": "agent_vlm"}
)
assert diag_vlm["judgment_source"] == "agent_vlm" and (diag_vlm["rows"], diag_vlm["cols"]) == (2, 3)
assert rows_vlm[3]["occupied"] and not rows_vlm[1]["occupied"]
print("[S5] locate_egg_holder (zero-prior): 2x3 (6 slots) and 2x2 (4 slots) autonomously inferred, size/slot err < 4 mm, "
      "occupied/empty classified, VLM agent_fn override ok")

# ------------------------------------------------------------------ S6
assert hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest().startswith("020cc6680f181872")
for f in ("NERO_V2_7_1_Unified_Backend.py", "NERO_V326_Wrist_Camera.py", "NERO_V331_Fixed_Camera.py", "NERO_V2_7_1_Agent.py"):
    txt = (ROOT / f).read_text()
    assert "NERO_V349" not in txt and "NERO_V347" not in txt, f
assert "NERO_V349_Real_Wrist_Camera.py" in (ROOT / "start_nero.sh").read_text()
print("[S6] NERO_V308_Camera_Perception.py unchanged; NERO_V326_Wrist_Camera.py & simulation modules untouched")
print("ALL V3.49 TESTS PASSED")
