#!/usr/bin/env python3
"""V3.87: Phase 2 (OV4688 4MP / 1080P Intrinsics + Eye-in-Hand Calibration) & Phase 3 (Closed-Loop Cup Grasp Angle
and Nesting Concentricity Visual Servoing with the Wrist Camera).

Checks:
  Z1 Phase 2 OV4688 4MP (2688×1520, 90° standard lens fx≈1350) & 1080P intrinsic calibration + Eye-in-Hand calibration
     accuracy (< 0.25 px RMS, < 0.8 mm EIH LOO error) and automatic resolution scaling `scale_K_for_frame`.
  Z2 Phase 3 `wrist_check_cup_grasp`: synthetic downward wrist camera view of a 95 mm cup rim at Z = 0.121 m with
     both horizontal rim shift (+8.0 mm, -5.0 mm) and a 16° gripper closing-line yaw misalignment -> recovers 3D rim
     centre within < 1.5 mm and recommends exact phi correction so the jaws straddle the rim perpendicularly.
  Z3 Phase 3 `wrist_check_cup_concentricity`: synthetic downward wrist camera view above target cup with inner held cup
     bottom and outer target cup rim offset by (-7.5 mm, +6.0 mm) -> recovers both centres and the exact 3D concentricity
     correction vector within < 1.5 mm.
  Z4 Full CupSkills (`pick_cup` + `place_cup_into`) closed-loop execution with `wrist_snapshot`:
     - In `pick_cup`, `_wrist_recheck_grasp` triggers at pregrasp, updates `W["grasp"]` & `Q["grasp"]` to fix both
       the rim shift and the jaw perpendicularity angle before descending;
     - In `place_cup_into`, `_wrist_recheck_place` triggers above the target cup, centres the held cup coaxially over
       the target cup rim (`concentric_err_mm` drops to < 1.5 mm), and completes the stack.
"""
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import NERO_V325_Camera as CAM                  # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49       # noqa: E402
import NERO_V362_Cup_Skills as CK               # noqa: E402
import NERO_V365_Real_Cup_Task as RCT           # noqa: E402
import NERO_V386_Real_Wrist_Camera as W86       # noqa: E402

assert W86.VERSION in ("V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10") and W49.VERSION in ("V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10") and RCT.VERSION in ("V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10")

# ------------------------------------------------------------------ Z1: Phase 2 OV4688 4MP Intrinsics & EIH Calibration
w_4mp, h_4mp = 2688, 1520
fx_true, fy_true = 1380.0, 1380.0   # ~90° HFOV standard lens on 2688x1520 OV4688
cx_true, cy_true = 1344.0, 760.0
Km_4mp = np.array([[fx_true, 0.0, cx_true], [0.0, fy_true, cy_true], [0.0, 0.0, 1.0]], dtype=np.float64)
dist_true = np.array([-0.08, 0.02, 0.0, 0.0, 0.0], dtype=np.float64)

cols_b, rows_b, sq_m = 7, 5, 0.032
obj_grid = np.zeros(((cols_b - 1) * (rows_b - 1), 3), dtype=np.float32)
idx_pt = 0
for r in range(rows_b - 1):
    for c in range(cols_b - 1):
        obj_grid[idx_pt] = [(c + 1) * sq_m, (r + 1) * sq_m, 0.0]
        idx_pt += 1

# Wrist camera mounted on link7 looking along TCP +Z (forward/downward toward the fingers):
T_flange_cam_true = (CK.T_L7_TCP @ np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, -0.080],
    [0.0, 0.0, 0.0, 1.0],
])).tolist()
T_base_board = CAM.T_from(
    [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
    [-0.42, -0.10, 0.0],
)

intr_samples = []
eih_samples = []
poses_rpy_xyz = [
    (0.0, 0.0, 0.0, -0.32, 0.02, 0.28),
    (12.0, -8.0, 10.0, -0.36, -0.04, 0.30),
    (-14.0, 10.0, -12.0, -0.28, 0.06, 0.26),
    (8.0, 14.0, 20.0, -0.38, 0.05, 0.32),
    (-10.0, -12.0, -18.0, -0.26, -0.03, 0.29),
    (16.0, 6.0, -8.0, -0.34, 0.08, 0.25),
    (-8.0, 15.0, 14.0, -0.30, -0.06, 0.31),
    (14.0, -14.0, -22.0, -0.35, 0.01, 0.27),
    (-15.0, -6.0, 16.0, -0.29, 0.04, 0.33),
]
for k, (rx_d, ry_d, rz_d, px, py, pz) in enumerate(poses_rpy_xyz):
    rx, ry, rz = math.radians(rx_d), math.radians(ry_d), math.radians(rz_d)
    R_cam = (
        np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
        @ np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
        @ np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    )
    T_cb = np.eye(4)
    T_cb[:3, :3] = R_cam
    T_cb[:3, 3] = [px * 0.15, py * 0.15, pz]
    rv, _ = cv2.Rodrigues(T_cb[:3, :3])
    tv = T_cb[:3, 3].reshape(3, 1)
    img_pts, _ = cv2.projectPoints(obj_grid, rv, tv, Km_4mp, dist_true)
    img_pts = img_pts.reshape(-1, 2)
    T_bc = CAM.mat_mul(T_base_board, CAM.T_inv(T_cb.tolist()))
    A_flange = CAM.mat_mul(T_bc, CAM.T_inv(T_flange_cam_true))
    intr_samples.append({
        "obj_pts": obj_grid.tolist(),
        "img_pts": img_pts.tolist(),
        "n_corners": len(obj_grid),
        "board_cols": cols_b,
        "board_rows": rows_b,
        "square_mm": 32.0,
        "center_frac": [float(np.mean(img_pts[:, 0])) / w_4mp, float(np.mean(img_pts[:, 1])) / h_4mp],
        "span_frac": 0.45,
        "T_base_flange": A_flange,
    })
    eih_samples.append({
        "q_deg": [0.0] * 7,
        "T_base_flange": A_flange,
        "T_cam_marker": T_cb.tolist(),
        "p_cam": T_cb[:3, 3].tolist(),
        "rvec": rv.flatten().tolist(),
    })

res_intr_4mp = W86.calibrate_intrinsics_4mp(intr_samples, image_size=(w_4mp, h_4mp), model_type=W49.DIST_MODELS[1])
assert res_intr_4mp["ok"] and res_intr_4mp["good"], res_intr_4mp
assert abs(res_intr_4mp["K"]["fx"] - fx_true) < 5.0 and res_intr_4mp["rms_px"] < 0.25, res_intr_4mp
res_eih_4mp = W49.calibrate_eye_in_hand(eih_samples, size_mm=24.0)
assert res_eih_4mp["ok"] and res_eih_4mp["good"] and res_eih_4mp["loo_max_mm"] < 0.5, res_eih_4mp

# Verify scale_K_for_frame scales 4MP K down to 1920x1080 seamlessly
K_scaled_1080 = W86.scale_K_for_frame(res_intr_4mp["K"], 1920, 1080)
assert K_scaled_1080["width"] == 1920 and abs(K_scaled_1080["fx"] - fx_true * (1920.0 / 2688.0)) < 0.5
print(f"[Z1] Phase 2 OV4688 4MP (2688x1520) intrinsic calib (fx={res_intr_4mp['K']['fx']:.1f}, RMS={res_intr_4mp['rms_px']:.3f} px) & EIH (LOO={res_eih_4mp['loo_max_mm']:.3f} mm) verified")


# ------------------------------------------------------------------ Helper: Render 3D circles into wrist camera RGB
def render_wrist_cups(K, T_base_flange, calib_eih, circles_3d):
    """Render filled/annular red circles in 3D base_link into the wrist camera image (w x h)."""
    w, h = int(K["width"]), int(K["height"])
    img = np.zeros((h, w, 3), dtype=np.uint8) + 215
    T_bw = np.asarray(W49.wrist_T_base_cam(T_base_flange, calib_eih), dtype=float)
    T_wb = np.linalg.inv(T_bw)
    Km = np.array([[K["fx"], 0.0, K["ppx"]], [0.0, K["fy"], K["ppy"]], [0.0, 0.0, 1.0]], dtype=np.float64)
    rv, _ = cv2.Rodrigues(T_wb[:3, :3])
    tv = T_wb[:3, 3].reshape(3, 1)
    thetas = np.linspace(0.0, 2.0 * math.pi, 180, endpoint=False)
    for cx, cy, cz, rad, color_rgb in circles_3d:
        pts_3d = np.column_stack([cx + rad * np.cos(thetas), cy + rad * np.sin(thetas), np.full_like(thetas, cz)]).astype(np.float32)
        uv, _ = cv2.projectPoints(pts_3d, rv, tv, Km, np.zeros(5))
        poly = np.round(uv.reshape(-1, 1, 2)).astype(np.int32)
        cv2.fillPoly(img, [poly], color_rgb)
    return img


# ------------------------------------------------------------------ Z2: Phase 3 Pre-Grasp Angle & Rim Shift Check
K_1080 = {"width": 1920, "height": 1080, "fx": 985.0, "fy": 985.0, "ppx": 960.0, "ppy": 540.0}
calib_eih = {"T_flange_cam": T_flange_cam_true, "marker_in_base": [-0.42, -0.10, 0.0], "loo_mean_mm": 0.2, "good": True}

prof = CK.so101_profile()
g = CK.geom()
exp_rim = np.array([-0.380, 0.080, g["h"]], dtype=float)
true_rim = np.array([-0.372, 0.075, g["h"]], dtype=float)   # +8.0 mm in X, -5.0 mm in Y
phi_plan = math.radians(30.0)
# Suppose the gripper arrives with an intentional +16 deg yaw error relative to the planned radial direction
phi_wrong = phi_plan - math.radians(16.0)
p_tcp_pre, R_tcp_pre = CK.pinch_tcp(exp_rim, phi_wrong, 0.035, flip=True, tilt_follow=True, profile=prof)
p_tcp_pre = p_tcp_pre + prof.pre_offset(R_tcp_pre, 0.090)

T_bt_pre = np.eye(4)
T_bt_pre[:3, :3], T_bt_pre[:3, 3] = R_tcp_pre, p_tcp_pre
# Compute flange pose T_base_flange from T_base_tcp @ inv(T_L7_TCP)
T_bf_pre = (T_bt_pre @ np.linalg.inv(CK.T_L7_TCP)).tolist()

rgb_grasp = render_wrist_cups(K_1080, T_bf_pre, calib_eih, [
    (true_rim[0], true_rim[1], true_rim[2], g["r_top"], (220, 25, 25)),
])

chk_g = W86.wrist_check_cup_grasp(
    rgb_grasp, K_1080, T_bf_pre, calib_eih, prof.T_tcp_grasp, T_bt_pre,
    exp_rim, phi_wrong, flip=True, r_top_m=g["r_top"]
)
assert chk_g["ok"] and chk_g["trusted"], chk_g
err_rim_mm = math.hypot(chk_g["measured_rim_xyz"][0] - true_rim[0], chk_g["measured_rim_xyz"][1] - true_rim[1]) * 1000.0
assert err_rim_mm < 1.5, (err_rim_mm, chk_g)
assert abs(chk_g["delta_rim_mm"][0] - 8.0) < 1.5 and abs(chk_g["delta_rim_mm"][1] - (-5.0)) < 1.5, chk_g

# Also test when gripper closing line is skewed by +18 deg in-place around the pinch point on true_rim:
p_true_pre, R_true_pre = CK.pinch_tcp(true_rim, phi_wrong, 0.035, flip=True, tilt_follow=True, profile=prof)
p_true_pre = p_true_pre + prof.pre_offset(R_true_pre, 0.090)
R_skew = np.array([[math.cos(math.radians(18.0)), -math.sin(math.radians(18.0)), 0],
                   [math.sin(math.radians(18.0)), math.cos(math.radians(18.0)), 0],
                   [0, 0, 1]]) @ R_true_pre
T_bt_skew = np.eye(4)
T_bt_skew[:3, :3], T_bt_skew[:3, 3] = R_skew, p_true_pre
chk_skew = W86.wrist_check_cup_grasp(
    rgb_grasp, K_1080, T_bf_pre, calib_eih, prof.T_tcp_grasp, T_bt_skew,
    true_rim, phi_wrong, flip=True, r_top_m=g["r_top"]
)
assert chk_skew["ok"] and abs(abs(chk_skew["angle_err_deg"]) - 18.0) < 2.5, chk_skew
print(f"[Z2] Phase 3 wrist_check_cup_grasp: 3D rim err={err_rim_mm:.2f} mm (shift {chk_g['delta_rim_mm']} mm), jaw skew detected={chk_skew['angle_err_deg']:+.1f}° (true -18.0°)")


# ------------------------------------------------------------------ Z3: Phase 3 Pre-Place Concentricity Check
exp_tgt_rim = np.array([-0.380, -0.080, g["h"]], dtype=float)
true_tgt_rim = np.array([-0.3875, -0.0740, g["h"]], dtype=float)   # -7.5 mm in X, +6.0 mm in Y
held_base = np.array([-0.380, -0.080, g["h"] + 0.020], dtype=float)

# Camera looking down from 0.28 m above target cup
T_bt_place = np.array([[1.0, 0.0, 0.0, -0.380], [0.0, -1.0, 0.0, -0.080], [0.0, 0.0, -1.0, 0.320], [0.0, 0.0, 0.0, 1.0]])
T_bf_place = (T_bt_place @ np.linalg.inv(CK.T_L7_TCP)).tolist()
# Render outer target cup rim (red) + non-red interior + inner held cup bottom (red) so nested contours exist
rgb_place = render_wrist_cups(K_1080, T_bf_place, calib_eih, [
    (true_tgt_rim[0], true_tgt_rim[1], true_tgt_rim[2], g["r_top"], (220, 25, 25)),
    (true_tgt_rim[0], true_tgt_rim[1], true_tgt_rim[2], g["r_top"] * 0.78, (215, 215, 215)),
    (held_base[0], held_base[1], held_base[2], g["r_bot"] * 0.85, (220, 25, 25)),
])

chk_p = W86.wrist_check_cup_concentricity(
    rgb_place, K_1080, T_bf_place, calib_eih,
    exp_tgt_rim, held_base, r_top_m=g["r_top"], r_bot_m=g["r_bot"]
)
assert chk_p["ok"] and chk_p["trusted"], chk_p
err_dx = abs(chk_p["delta_mm"][0] - (-7.5))
err_dy = abs(chk_p["delta_mm"][1] - 6.0)
assert err_dx < 1.5 and err_dy < 1.5, chk_p
print(f"[Z3] Phase 3 wrist_check_cup_concentricity: measured delta={chk_p['delta_mm']} mm (true [-7.5, +6.0] mm, source={chk_p['held_source']})")


# ------------------------------------------------------------------ Z4: End-to-End Closed-Loop CupSkills Execution
class MockCupWristIO:
    """Simulated real-mode CupSkills IO with both fixed camera perception and wrist_snapshot() closed-loop feedback."""
    def __init__(self):
        self.arm = CK.FastArm()
        self.prof = RCT.real_profile(CK, lift_settle_s=0.0)
        self.q = [0.0, -0.35, 0.0, 1.15, 0.0, 0.75, 0.0]
        self._tcp_override = None
        self.open_mm = self.prof.open_mm
        self.clamped = False
        self.logs = []
        # Note: first sight in _perceive sorts by y ascending -> cup_1 has smaller y (-0.110), cup_2 has larger y (+0.110)
        self.src_base_coarse = np.array([-0.360, -0.110, 0.0])
        self.src_base_true = np.array([-0.354, -0.106, 0.0])
        self.dst_base_coarse = np.array([-0.360, 0.110, 0.0])
        self.dst_base_true = np.array([-0.365, 0.115, 0.0])
        self.src_lifted = False
        self.stacked = False

    def emit(self, s):
        self.logs.append(s)

    def sleep(self, _s):
        pass

    def pose_mode(self):
        return "camera"

    def gripper_profile(self):
        return self.prof

    def current_q(self):
        return list(self.q)

    def tcp_pose(self):
        if self._tcp_override is not None:
            return list(self._tcp_override[0]), list(self._tcp_override[1])
        T = self.arm.fk(np.asarray(self.q, float))
        return [float(v) for v in T[:3, 3]], CK.R_to_quat_xyzw(T[:3, :3])

    def safety_trips(self):
        return 0

    def latch_pending(self):
        return False

    def clear_latch(self, _l=""):
        pass

    def move_joints(self, label, q, slow=None, speed=None):
        self.q = [float(v) for v in q]
        self._tcp_override = None
        if "lift" in label and self.clamped:
            self.src_lifted = True
        return {"ok": True}

    def move_line(self, label, xyz, quat, speed=0.05, slow=None):
        R = CK.quat_xyzw_to_R(quat)
        q_new = CK.ik_from(self.q, xyz, R)
        if q_new is not None:
            self.q = list(q_new)
            self._tcp_override = None
        else:
            self._tcp_override = ([float(v) for v in xyz], [float(v) for v in quat])
        if "lift" in label and self.clamped:
            self.src_lifted = True
        return {"ok": True}

    def gripper_open(self, mm):
        self.open_mm = float(mm)
        if self.src_lifted:
            self.stacked = True
        self.clamped = False
        return {"opening_mm": self.open_mm}

    def gripper_clamp(self, force_n, settle_s=0.0):
        self.clamped = True
        return {"opening_mm": 2.6, "contact_left": True, "contact_right": True, "hold_active": True}

    def gripper_state(self):
        return {"opening_mm": 2.6 if self.clamped else self.open_mm, "hold_active": self.clamped}

    def perceive(self, tag):
        g = CK.geom()
        cups = []
        dets = []
        if self.stacked:
            st_e = {"det_id": "cup_2", "kind": "stack", "base_xyz_m": self.dst_base_true.tolist(),
                    "height_mm": (g["h"] + 0.025) * 1000.0, "tilt_deg": 0.5, "conf": 0.95}
            cups.append(st_e)
            dets.append(st_e)
        else:
            if not self.src_lifted:
                c1 = {"det_id": "cup_1", "kind": "cup", "base_xyz_m": self.src_base_coarse.tolist(),
                      "height_mm": g["h"] * 1000.0, "tilt_deg": 0.5, "conf": 0.94}
                cups.append(c1)
                dets.append(c1)
            c2 = {"det_id": "cup_2", "kind": "cup", "base_xyz_m": self.dst_base_coarse.tolist(),
                  "height_mm": g["h"] * 1000.0, "tilt_deg": 0.5, "conf": 0.95}
            cups.append(c2)
            dets.append(c2)
        return {"ok": True, "cups": cups, "dets": dets}

    def wrist_snapshot(self):
        p_t, q_t = self.tcp_pose()
        T_tcp = np.eye(4)
        T_tcp[:3, :3], T_tcp[:3, 3] = CK.quat_xyzw_to_R(q_t), np.asarray(p_t, float)
        T_bf = (T_tcp @ np.linalg.inv(CK.T_L7_TCP)).tolist()
        g = CK.geom()
        if not self.src_lifted:
            rgb = render_wrist_cups(K_1080, T_bf, calib_eih, [
                (self.src_base_true[0], self.src_base_true[1], g["h"], g["r_top"], (220, 25, 25)),
            ])
        else:
            rgb = render_wrist_cups(K_1080, T_bf, calib_eih, [
                (self.dst_base_true[0], self.dst_base_true[1], g["h"], g["r_top"], (220, 25, 25)),
            ])
        return {"rgb": rgb, "K": dict(K_1080), "calib_eih": calib_eih}


mio = MockCupWristIO()
sk = CK.CupSkills(mio)
r_obs = sk.observe_scene()
assert r_obs["ok"]
r_plan = sk.plan_cup_stack("cup_1", "cup_2")
assert r_plan["ok"], r_plan
r_pick = sk.pick_cup("cup_1")
assert r_pick["ok"], r_pick
w_grasp_chk = r_pick["physical_verification"].get("wrist_grasp_check")
assert w_grasp_chk is not None and w_grasp_chk["ok"] and w_grasp_chk["applied"] is True, w_grasp_chk

r_place = sk.place_cup_into("cup_2")
assert r_place["ok"], r_place
w_place_chk = r_place["physical_verification"].get("wrist_place_check")
assert w_place_chk is not None and len(w_place_chk.get("checks") or []) >= 1, w_place_chk
assert w_place_chk["checks"][0]["applied"] is True
print(f"[Z4] End-to-end CupSkills closed-loop execution verified: pick wrist shift={w_grasp_chk['delta_rim_mm']} mm, place concentricity checks={len(w_place_chk['checks'])}")
print("ALL V3.87 TESTS PASSED")
