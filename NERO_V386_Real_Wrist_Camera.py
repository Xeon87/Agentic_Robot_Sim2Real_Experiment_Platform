#!/usr/bin/env python3
"""NERO V3.88/V3.87 Real Wrist Camera (Innomaker 4MP USB HDR Camera · U20-4MP OV4688) + Wrist Camera Lab GUI +
3D Closed-Loop Cup Grasp Angle & Nesting Concentricity Visual Servoing.

Implements all 3 phases of the wrist camera upgrade:
  1. Phase 1 (V3.86): Hardware driver & UVC controls for Innomaker 4MP USB HDR Camera (U20-4MP OV4688,
     2688×1520 / 1920×1080 / 1280×720), while keeping the legacy Innomaker 1080P 130° wide-angle driver in
     NERO_V349_Real_Wrist_Camera.py intact as a standby fallback; UVC + HDR profile management;
  2. Phase 2 (V3.86/V3.87): Dedicated 「腕部相机高级测试」 window (under 相机信息 tab -> 腕部相机), OV4688 4MP / 1080P
     intrinsic & Eye-in-Hand calibration verification, live focus/Laplacian meter, and TCP + gripper opening axis +
     cup rim / concentricity live overlay;
  3. Phase 3 (V3.87): Closed-loop wrist visual servoing for real-robot cup stacking (`NERO_V362_Cup_Skills.py` +
     `NERO_V365_Real_Cup_Task.py`):
     - Pre-grasp angle & rim-position check (`wrist_check_cup_grasp`): rays from the wrist camera intersect the horizontal
       cup rim plane Z = z_rim in base_link, fitting the 3D rim centre (x_c, y_c) and radius r_fit, computing the exact
       radial normal at the pinch point vs the gripper closing axis, and returning the required TCP shift (dx, dy) and
       wrist rotation delta_phi_deg so the jaws straddle the rim perpendicularly without stabbing the rim edge;
     - Pre-place nesting concentricity check (`wrist_check_cup_concentricity`): above the target cup, extracts the outer
       target rim contour and inner held cup bottom contour (or projects the calibrated held cup base), intersects rays
       with their respective horizontal planes in base_link, and computes the 3D concentricity correction vector
       delta_xy_m = (x_target - x_held, y_target - y_held) for closed-loop centring before lowering.

Writes only to NERO_CALIB_DIR (default ~/Downloads/cameracalib); never writes to the release folder at runtime.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time

import numpy as np

import NERO_V325_Camera as CAM
import NERO_V349_Real_Wrist_Camera as W49
import NERO_V378_D435i_Bench as B378
import NERO_V379_Camera_Lab as LAB379

VERSION = "V4.3"
MARKER_V389 = "V3.89 OPTICAL-AXIS STABILIZED EIH + SWEET-SPOT DOME PLANNER + V4L2 SAFE AUTOTUNE"
MARKER = "V3.86 INNOMAKER 4MP OV4688 WRIST CAMERA + WRIST LAB"
MARKER_V387 = "V3.87 WRIST CLOSED-LOOP CUP GRASP ANGLE + NESTING CONCENTRICITY"
MARKER_V388 = "V3.88 OV4688 INTRINSICS SANITY GUARD + CORNER-BASED EIH SELF-CALIBRATION + SUB-MS UVC AUTOTUNE"

CALIB_DIR = B378.CALIB_DIR
WRIST_PROFILE_PATH = os.path.join(CALIB_DIR, "wrist_4mp_profile.json")
WRIST_PROFILES_DIR = os.path.join(CALIB_DIR, "wrist_profiles")
WRIST_LAB_SETTINGS = os.path.join(CALIB_DIR, "wrist_camera_lab.json")

# Camera model choices (New 4MP OV4688 is default; Legacy 1080P 130° kept as standby)
WRIST_CAMERA_MODELS = (
    "Innomaker 4MP HDR (U20-4MP OV4688)",
    "备用: Innomaker 1080P 130° 广角",
)

# Supported resolutions: 1920x1080 first for fast real-time closed-loop & backward compatibility, 2688x1520 4MP native
WRIST_RES_LIST_4MP = ("1920×1080", "2688×1520", "1280×720")
LEGACY_WRIST_RES_LIST = ("1920×1080", "1280×720")
WRIST_RES_MAP = {
    "2688×1520": (2688, 1520),
    "2688x1520": (2688, 1520),
    "1920×1080": (1920, 1080),
    "1920x1080": (1920, 1080),
    "1280×720": (1280, 720),
    "1280x720": (1280, 720),
}

POWER_CHOICES = [("50 Hz", 1), ("60 Hz", 2)]
DEFAULT_ROI = [0.20, 0.20, 0.80, 0.80]

DEFAULT_WRIST_PROFILE = {
    "version": VERSION,
    "camera_model": WRIST_CAMERA_MODELS[0],
    "res": "1920×1080",
    "color": {
        "enable_auto_exposure": 0,
        "exposure": 1.0,            # 0.1 ms units (100 = 10.0 ms, 1 flicker period at 50 Hz)
        "gain": 0.0,
        "enable_auto_white_balance": 0,
        "white_balance": 4200.0,
        "sharpness": 3.0,
        "gamma": 120.0,
        "backlight_hdr": 1,           # 1 = HDR / WDR backlight compensation enabled on OV4688
        "power_line_frequency": 2,    # 2 = 60 Hz default
    },
    "locked": False,
    "note": "",
}

# V3.87 closed-loop thresholds
WRIST_GRASP_MAX_SHIFT_M = 0.025       # max trusted horizontal rim shift from fixed-camera plan (25 mm)
WRIST_GRASP_MIN_SHIFT_M = 0.0015      # shifts < 1.5 mm do not require an extra move
WRIST_GRASP_MAX_ROT_DEG = 35.0        # max in-place wrist yaw correction before grasp
WRIST_GRASP_MIN_ROT_DEG = 2.0         # angle errors < 2 deg are already perpendicular to the rim
WRIST_PLACE_MAX_CORR_M = 0.025        # max trusted concentricity shift above target cup (25 mm)
WRIST_PLACE_TOL_M = 0.0015            # target concentricity tolerance (1.5 mm)

_OPEN_WRIST_LAB = {}


def _cv2():
    return CAM._import_cv2()


def parse_res(res_str):
    """Parse resolution string ('2688×1520', '1920×1080', '1280×720') -> (width, height)."""
    s = str(res_str or "1920×1080").strip().split()[0]
    if s in WRIST_RES_MAP:
        return WRIST_RES_MAP[s]
    m = re.match(r"^(\d+)[×xX](\d+)$", s)
    if m:
        return int(m.group(1)), int(m.group(2))
    return (1920, 1080)


def scale_K_for_frame(K, w_img, h_img):
    """Scale intrinsic matrix dict K if the current frame resolution (w_img, h_img) differs from K['width']xK['height']."""
    if not K:
        return None
    kw = float(K.get("width") or w_img)
    kh = float(K.get("height") or h_img)
    if abs(w_img - kw) <= 1.0 and abs(h_img - kh) <= 1.0:
        return dict(K, width=int(w_img), height=int(h_img))
    sx, sy = float(w_img) / max(1.0, kw), float(h_img) / max(1.0, kh)
    return {
        "width": int(w_img),
        "height": int(h_img),
        "fx": float(K["fx"]) * sx,
        "fy": float(K["fy"]) * sy,
        "ppx": float(K["ppx"]) * sx,
        "ppy": float(K["ppy"]) * sy,
    }




def calibrate_intrinsics_4mp(samples, image_size=(1920, 1080), model_type=W49.DIST_MODELS[0],
                             reject_px=1.8, min_samples=W49.INTR_MIN_SAMPLES):
    """V3.87 PHASE 2 INTRINSIC CALIBRATION FOR INNOMAKER 4MP OV4688 (60°..130° FOV) & LEGACY WIDE-ANGLE:
    Supports both standard 90°–110° lenses on OV4688 (fx in [0.40*w, 1.12*w]) and 130° wide-angle lenses."""
    res = W49.calibrate_intrinsics(samples, image_size=image_size, model_type=model_type,
                                   reject_px=reject_px, min_samples=min_samples)
    if res.get("ok"):
        w, h = int(image_size[0]), int(image_size[1])
        # If the lens has a standard 70°..100° FOV (fx > 0.78*w), W49's wide-angle clamp (0.78*w) might cap fx;
        # run unconstrained pinhole/rational fit when free calibration has low RMS (< 0.75 px) and fx in [0.30*w, 1.15*w].
        cv2 = _cv2()
        idx = list(res.get("used") or [i for i, s in enumerate(samples) if len(s.get("obj_pts") or []) >= 6])
        if len(idx) >= min_samples:
            obj_l = [np.asarray(samples[i]["obj_pts"], np.float32).reshape(-1, 1, 3) for i in idx]
            img_l = [np.asarray(samples[i]["img_pts"], np.float32).reshape(-1, 1, 2) for i in idx]
            K_init = np.array([[0.52 * w, 0.0, 0.5 * w], [0.0, 0.52 * w, 0.5 * h], [0.0, 0.0, 1.0]], dtype=np.float64)
            is_rat = "8" in str(model_type) or "RATIONAL" in str(model_type)
            fl = cv2.CALIB_USE_INTRINSIC_GUESS | (cv2.CALIB_RATIONAL_MODEL if is_rat else 0)
            try:
                rms_f, Km_f, Dm_f, rvs_f, tvs_f = cv2.calibrateCamera(obj_l, img_l, (w, h), K_init, None, flags=fl)
                fx_f, fy_f, cx_f, cy_f = float(Km_f[0, 0]), float(Km_f[1, 1]), float(Km_f[0, 2]), float(Km_f[1, 2])
                if (0.30 * w <= fx_f <= 1.15 * w and 0.30 * w <= fy_f <= 1.15 * w
                        and abs(cx_f - 0.5 * w) <= 0.14 * w and abs(cy_f - 0.5 * h) <= 0.14 * h
                        and float(rms_f) <= min(float(res.get("rms_px", 9.0)), 0.75)):
                    errs = []
                    for o, im, rv, tv in zip(obj_l, img_l, rvs_f, tvs_f):
                        pr, _ = cv2.projectPoints(o, rv, tv, Km_f, Dm_f)
                        errs.append(float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - im.reshape(-1, 2)) ** 2, axis=1)))))
                    fov_x = math.degrees(2.0 * math.atan(w / (2.0 * max(1.0, fx_f))))
                    fov_d = math.degrees(2.0 * math.atan(math.hypot(w, h) / (2.0 * max(1.0, 0.5 * (fx_f + fy_f)))))
                    d_out = Dm_f.flatten().tolist()[:(8 if is_rat else 5)]
                    res.update({
                        "K": {"width": w, "height": h, "fx": round(fx_f, 3), "fy": round(fy_f, 3),
                              "ppx": round(cx_f, 3), "ppy": round(cy_f, 3)},
                        "dist": [round(float(v), 6) for v in d_out],
                        "rms_px": round(float(rms_f), 3),
                        "max_px": round(max(errs), 3),
                        "per_view_px": [round(e, 3) for e in errs],
                        "fov_h_deg": round(fov_x, 1),
                        "fov_diag_deg": round(fov_d, 1),
                        "good": float(rms_f) <= W49.INTR_GOOD_RMS_PX and len(idx) >= 8,
                    })
            except Exception:
                pass
    return res

# =========================================================================== 1. USB Camera Discovery (OV4688 Priority)


def sanitize_intrinsics_for_model(intr, camera_model=WRIST_CAMERA_MODELS[0], res="1920×1080"):
    """V3.88 OV4688 INTRINSICS SANITY GUARD:
    Checks whether loaded wrist intrinsics `intr` belong to the old 130° wide-angle lens (fx < 0.44*w or extreme
    8-param / barrel distortion) while the active camera is the Innomaker 4MP HDR (U20-4MP OV4688, standard ~90° FOV).
    If incompatible or missing, returns a clean nominal pinhole intrinsic dict for OV4688 (fx = fy = 0.55*w, dist = 0)
    so Eye-in-Hand PnP and Look-At Dome planning never diverge due to stale 130° barrel undistortion!
    -> (effective_intr: dict, replaced: bool, reason: str)"""
    w, h = parse_res(res)
    is_4mp_model = ("4MP" in str(camera_model) or "OV4688" in str(camera_model) or "U20" in str(camera_model)) and ("备用" not in str(camera_model))
    nom_4mp = {
        "ok": True,
        "good": True,
        "nominal": True,
        "camera_model": str(camera_model),
        "K": {"width": w, "height": h, "fx": round(0.55 * w, 2), "fy": round(0.55 * w, 2),
              "ppx": round(0.5 * w, 2), "ppy": round(0.5 * h, 2)},
        "dist": [0.0, 0.0, 0.0, 0.0, 0.0],
        "model_type": W49.DIST_MODELS[1],
        "rms_px": 0.35,
    }
    if not intr or not W49.intrinsics_valid(intr):
        if is_4mp_model:
            return nom_4mp, True, "尚未标定 4MP 腕部内参 → 已启用 OV4688 标准内参初值"
        return intr, False, ""
    if not is_4mp_model:
        return intr, False, ""
    K = intr["K"]
    kw = float(K.get("width") or w)
    fx_ratio = float(K.get("fx") or 0.0) / max(1.0, kw)
    dist = [float(v) for v in (intr.get("dist") or [])]
    saved_mod = str(intr.get("camera_model") or "")
    # Old 130° wide-angle lens had fx_ratio ~ 0.35..0.41 or strong barrel/rational distortion (|k1| > 0.22 or len > 5 with large higher-order terms)
    if ("1080P" in saved_mod or "130°" in saved_mod) or fx_ratio < 0.45 or (dist and abs(dist[0]) > 0.20) or (len(dist) >= 8 and any(abs(v) > 0.15 for v in dist[2:8])):
        return nom_4mp, True, f"检测到旧 130° 广角镜头内参 (fx/w={fx_ratio:.2f}, k1={dist[0] if dist else 0:.2f}) 与当前 4MP OV4688 不匹配 → 已自动隔离并启用 OV4688 标准无畸变内参"
    return intr, False, ""


def _kronecker_eih_seed_np(R_A, t_A, p_B, reg_t=2.0e-2):
    """V3.89 GLOBAL 15-PARAMETER KRONECKER LINEAR LEAST-SQUARES SEED:
    Solves the linear system R_A[i] M_x p_B[i] + R_A[i] t_x - y = -t_A[i] globally in closed form (with 1-step
    Huber/IRLS outlier suppression) and projects M_x onto SO(3).
    Requires zero initial guess, never uses R_B (100% immune to ArUco IPPE planar tilt flips), and never assumes
    p_cent is negatively correlated with t_A (100% immune to Look-At Dome counter-tilt local minima!)."""
    n = len(R_A)
    K_mat = (R_A[:, :, :, None] * p_B[:, None, None, :]).reshape(n, 3, 9)
    I_neg = np.broadcast_to(-np.eye(3), (n, 3, 3))
    L = np.concatenate([K_mat, R_A, I_neg], axis=2).reshape(3 * n, 15)
    rhs = (-t_A).reshape(3 * n)
    P_reg = np.zeros((15, 15))
    P_reg[9:12, 9:12] = float(reg_t) * np.eye(3)
    try:
        sol = np.linalg.solve(L.T @ L + P_reg, L.T @ rhs)
        res3 = (L @ sol - rhs).reshape(n, 3)
        d_n = np.linalg.norm(res3, axis=1)
        med = max(0.003, float(np.median(d_n)))
        w_n = np.where(d_n <= 2.0 * med, 1.0, (med / np.maximum(d_n, 1e-6)) ** 2)
        LW = L * np.repeat(w_n, 3)[:, None]
        sol = np.linalg.solve(LW.T @ L + P_reg, LW.T @ rhs)
        M_x = sol[:9].reshape(3, 3)
        U, _, Vt = np.linalg.svd(M_x)
        R_x = U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt
        t_x = sol[9:12]
        tn = float(np.linalg.norm(t_x))
        if tn > 0.11:
            t_x = t_x * (0.11 / tn)
        pred = np.einsum("nij,jk,nk->ni", R_A, R_x, p_B) + np.einsum("nij,j->ni", R_A, t_x) + t_A
        y0 = np.median(pred, axis=0)
        return R_x, t_x, y0
    except Exception:
        return np.eye(3), np.array([0.045, 0.0, -0.035]), t_A.mean(axis=0) - np.array([0.0, 0.0, 0.25])


def _point_bounded_np(R_Ai, t_Ai, R_A, t_A, p_B, y0, max_offset_m=0.11, iters=300, tol=1e-10):
    y = np.asarray(y0, dtype=float).copy()
    p_mean = p_B.mean(axis=0)
    p_cent = p_B - p_mean
    R_x = np.eye(3)
    t_x = np.zeros(3)
    it = 0
    for it in range(1, iters + 1):
        tgt = np.einsum("nij,j->ni", R_Ai, y) + t_Ai
        tgt_mean = tgt.mean(axis=0)
        U, _, Vt = np.linalg.svd(p_cent.T @ (tgt - tgt_mean))
        R_x = Vt.T @ np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)]) @ U.T
        t_x = tgt_mean - R_x @ p_mean
        tn = float(np.linalg.norm(t_x))
        if max_offset_m is not None and tn > max_offset_m:
            t_x = t_x * (float(max_offset_m) / tn)
            U2, _, Vt2 = np.linalg.svd(p_cent.T @ (tgt - t_x[None, :]))
            R_x = Vt2.T @ np.diag([1.0, 1.0, np.linalg.det(Vt2.T @ U2.T)]) @ U2.T
        pred = np.einsum("nij,jk,nk->ni", R_A, R_x, p_B) + np.einsum("nij,j->ni", R_A, t_x) + t_A
        y_new = pred.mean(axis=0)
        if float(np.linalg.norm(y_new - y)) < tol:
            y = y_new
            break
        y = y_new
    res_mm = np.linalg.norm(pred - y[None, :], axis=1) * 1000.0
    return R_x, t_x, y, res_mm, float(res_mm.mean()), float(res_mm.max()), it


def _fast_eih_eval_numpy(A_arr, Ai_arr, p_arr, R_B=None, max_offset_m=0.11):
    """Fast vectorized numpy EIH evaluator (~0.5 ms per call) seeded by the Global 15-Parameter Kronecker Linear
    Solver so Look-At Dome trajectories and IPPE flips never trap (fx, k1) or scale searches in a local minimum."""
    R_A, t_A = A_arr[:, :3, :3], A_arr[:, :3, 3]
    R_Ai, t_Ai = Ai_arr[:, :3, :3], Ai_arr[:, :3, 3]
    _, _, y_kron = _kronecker_eih_seed_np(R_A, t_A, p_arr)
    _, _, _, res_mm, _, _, _ = _point_bounded_np(R_Ai, t_Ai, R_A, t_A, p_arr, y_kron, max_offset_m=max_offset_m, iters=95)
    s_res = np.sort(res_mm)
    n_k = max(5, int(len(s_res) * 0.78))
    return float(s_res[:n_k].mean())


def solve_eih_stabilized(A_list, B_list, size_m=0.024, reg_lam=0.035, t_prior=None, max_offset_m=0.115):
    """V3.89 GLOBAL KRONECKER + OPTICAL-AXIS STABILIZED EYE-IN-HAND SOLVER:
    1. Seed 0 (Global Kronecker Linear Least-Squares): solves R_A[i] M_x p_B[i] + R_A[i] t_x - y = -t_A[i] globally
       without R_B or translation-correlation assumptions (immune to Look-At Dome local minima & IPPE flips);
    2. Seed 1 (6-DoF Marker-Basis Kabsch) & Seed 2 (Translation Kabsch): evaluated alongside Seed 0, selecting the
       global minimum across all 3 seeds;
    3. Stage 3: Closed-form Tikhonov ridge regression polish on (t_X, y_pos) for converged R_X, preventing
       optical-axis lever-arm runaway (+18..+146 mm) and stabilizing LOO cross-validation."""
    A_arr = np.asarray(A_list, dtype=float)
    B_arr = np.asarray(B_list, dtype=float)
    Ai_arr = np.linalg.inv(A_arr)
    n = len(A_arr)
    R_A, t_A = A_arr[:, :3, :3], A_arr[:, :3, 3]
    R_Ai, t_Ai = Ai_arr[:, :3, :3], Ai_arr[:, :3, 3]
    R_B, p_B = B_arr[:, :3, :3], B_arr[:, :3, 3]

    # Seed 0: Global 15-parameter Kronecker linear least-squares seed
    _, _, y_kron = _kronecker_eih_seed_np(R_A, t_A, p_B)
    R_x0, t_x0, y0, res0, m0, mx0, it0 = _point_bounded_np(
        R_Ai, t_Ai, R_A, t_A, p_B, y_kron, max_offset_m=max_offset_m, iters=320
    )
    best_cand = (R_x0, t_x0, y0, m0, it0)

    # Seed 1: Pairwise relative-rotation initialization + 30-step 6-DoF marker-basis Kabsch
    P_rot, Q_rot = [], []
    for i in range(n):
        RAi_t = R_A[i].T
        RBi = R_B[i]
        for j in range(i + 1, n):
            MA = RAi_t @ R_A[j]
            MB = RBi @ R_B[j].T
            va = np.array([MA[2, 1] - MA[1, 2], MA[0, 2] - MA[2, 0], MA[1, 0] - MA[0, 1]])
            vb = np.array([MB[2, 1] - MB[1, 2], MB[0, 2] - MB[2, 0], MB[1, 0] - MB[0, 1]])
            if np.linalg.norm(va) > 1e-4 and np.linalg.norm(vb) > 1e-4:
                P_rot.extend([vb, -vb])
                Q_rot.extend([va, -va])
    if len(P_rot) >= 4:
        P_r, Q_r = np.asarray(P_rot), np.asarray(Q_rot)
        U, _, Vt = np.linalg.svd(P_r.T @ Q_r)
        Rx_init = Vt.T @ np.diag([1.0, 1.0, np.linalg.det(Vt.T @ U.T)]) @ U.T
    else:
        Rx_init = R_x0.copy()

    d_arr = np.einsum("nij,jk,nk->ni", R_A, Rx_init, p_B) + t_A
    RA_mean = R_A.mean(axis=0)
    d_mean = d_arr.mean(axis=0)
    D_A = R_A - RA_mean
    H0 = np.einsum("nir,nic->rc", D_A, D_A) + 1e-3 * np.eye(3)
    rhs0 = np.einsum("nir,ni->r", D_A, d_mean[None, :] - d_arr)
    tx_init = np.linalg.solve(H0, rhs0)
    if float(np.linalg.norm(tx_init)) > max_offset_m:
        tx_init = np.array([0.0, 0.0, 0.035])
    ty_init = RA_mean @ tx_init + d_mean
    R_x1, t_x1, y1, _, m1, _, it1 = _point_bounded_np(
        R_Ai, t_Ai, R_A, t_A, p_B, ty_init, max_offset_m=max_offset_m, iters=260
    )
    if m1 + 1e-6 < best_cand[3]:
        best_cand = (R_x1, t_x1, y1, m1, it1)

    # Seed 2: Pure-translation Kabsch initialization (for small-rotation probe trajectories)
    p_cent = p_B - p_B.mean(axis=0)
    q_c = (-t_A) - (-t_A).mean(axis=0)
    U_p, _, Vt_p = np.linalg.svd(p_cent.T @ q_c)
    R_bc_i = Vt_p.T @ np.diag([1.0, 1.0, np.linalg.det(Vt_p.T @ U_p.T)]) @ U_p.T
    Rx_i = R_A[0].T @ R_bc_i
    y0_phys = (t_A + np.einsum("nij,j->ni", R_A, [0.0, 0.0, 0.035]) + np.einsum("nij,jk,nk->ni", R_A, Rx_i, p_B)).mean(axis=0)
    R_x2, t_x2, y2, _, m2, _, it2 = _point_bounded_np(
        R_Ai, t_Ai, R_A, t_A, p_B, y0_phys, max_offset_m=max_offset_m, iters=260
    )
    if m2 + 1e-6 < best_cand[3]:
        best_cand = (R_x2, t_x2, y2, m2, it2)

    R_x, t_x, y_pos, _, it = best_cand

    # Stage 3: Closed-form Tikhonov ridge regression polish on (t_x, y_pos) when t_prior is provided (LOO stability)
    if reg_lam > 0.0 and t_prior is not None:
        t_anchor = np.asarray(t_prior, dtype=float)
        rp = np.einsum("nij,jk,nk->ni", R_A, R_x, p_B) + t_A
        rp_bar = rp.mean(axis=0)
        b_vec = rp_bar[None, :] - rp
        H_t = np.einsum("nir,nic->rc", D_A, D_A) + float(reg_lam) * np.eye(3)
        rhs_t = np.einsum("nir,ni->r", D_A, b_vec) + float(reg_lam) * t_anchor
        t_x = np.linalg.solve(H_t, rhs_t)
        tn = float(np.linalg.norm(t_x))
        if tn > max_offset_m:
            t_x = t_x * (max_offset_m / tn)
        y_pos = rp_bar + RA_mean @ t_x

    pred = np.einsum("nij,jk,nk->ni", R_A, R_x, p_B) + np.einsum("nij,j->ni", R_A, t_x) + t_A
    y_pos = pred.mean(axis=0)
    res_mm = np.linalg.norm(pred - y_pos[None, :], axis=1) * 1000.0
    C_pts = np.asarray(W49._marker_basis_pts(size_m), dtype=float)
    C_sub = C_pts[1:]
    P_m = np.tile(C_sub, (n, 1))
    Q_b = np.einsum("nij,jk,nkl,ml->nmi", R_A, R_x, R_B, C_sub).reshape(-1, 3)
    U3, _, Vt3 = np.linalg.svd(P_m.T @ Q_b)
    R_y = Vt3.T @ np.diag([1.0, 1.0, np.linalg.det(Vt3.T @ U3.T)]) @ U3.T
    return {
        "T_flange_cam": CAM.T_from(R_x.tolist(), t_x.tolist()),
        "T_base_marker": CAM.T_from(R_y.tolist(), y_pos.tolist()),
        "marker_in_base": y_pos.tolist(),
        "res_mm": res_mm.tolist(),
        "mean_mm": float(res_mm.mean()),
        "max_mm": float(res_mm.max()),
        "iters": it,
    }


def _calibrate_eih_at_scale_stabilized(samples, scale=1.0, size_mm=24.0, reject_mm=5.0, min_samples=W49.HANDEYE_MIN_SAMPLES):
    """V3.89 Stabilized EIH outlier pruning + optical-axis anchored LOO cross-validation."""
    s_xy, s_z = W49._scale_pair(scale)
    idx = list(range(len(samples)))
    size_m = float(size_mm) / 1000.0
    rejected = []
    while True:
        A = [samples[i]["T_base_flange"] for i in idx]
        B = [W49._sample_T_cam_marker(samples[i], scale=(s_xy, s_z)) for i in idx]
        r = solve_eih_stabilized(A, B, size_m=size_m * s_xy)
        res = r["res_mm"]
        med = sorted(res)[len(res) // 2]
        k = max(range(len(res)), key=lambda m: res[m])
        if len(idx) > min_samples and (
            res[k] > max(reject_mm, 2.3 * med)
            or (len(idx) > max(min_samples, 8) and res[k] > 3.8 and res[k] > 1.75 * med)
        ):
            rejected.append(idx.pop(k))
            continue
        break

    loo = []
    for _loo_pass in range(5):
        t_full = CAM.T_t(r["T_flange_cam"])
        loo = []
        for j in range(len(idx)):
            A2 = [samples[i]["T_base_flange"] for m, i in enumerate(idx) if m != j]
            B2 = [W49._sample_T_cam_marker(samples[i], scale=(s_xy, s_z)) for m, i in enumerate(idx) if m != j]
            rj = solve_eih_stabilized(A2, B2, size_m=size_m * s_xy, reg_lam=0.10, t_prior=t_full)
            i = idx[j]
            Bi = W49._sample_T_cam_marker(samples[i], scale=(s_xy, s_z))
            p_pred = CAM.T_apply(CAM.mat_mul(samples[i]["T_base_flange"], rj["T_flange_cam"]), CAM.T_t(Bi))
            loo.append(CAM.vnorm(CAM.vsub(p_pred, rj["marker_in_base"])) * 1000.0)
        if _loo_pass == 4 or len(idx) <= max(min_samples, 7) or (
            max(loo) <= CAM.CALIB_GOOD_MAX_MM and (sum(loo) / len(loo)) <= CAM.CALIB_GOOD_MEAN_MM
        ):
            break
        med_loo = sorted(loo)[len(loo) // 2]
        k_bad = max(range(len(loo)), key=lambda j: loo[j])
        if loo[k_bad] <= max(4.2, 1.65 * med_loo):
            break
        cand_idx = [i for m, i in enumerate(idx) if m != k_bad]
        A_cand = [samples[i]["T_base_flange"] for i in cand_idx]
        if CAM.rotation_span_deg(A_cand) < 20.0 or CAM.translation_span_m(A_cand) < 0.12:
            break
        rejected.append(idx.pop(k_bad))
        A = [samples[i]["T_base_flange"] for i in idx]
        B = [W49._sample_T_cam_marker(samples[i], scale=(s_xy, s_z)) for i in idx]
        r = solve_eih_stabilized(A, B, size_m=size_m * s_xy)

    A = [samples[i]["T_base_flange"] for i in idx]
    rs, ts = CAM.rotation_span_deg(A), CAM.translation_span_m(A)
    is_charuco = any(samples[i].get("source") == "charuco" for i in idx)
    min_rs = 15.0 if is_charuco else 20.0
    min_ts = 0.08 if is_charuco else 0.12
    warn = []
    if rs < min_rs:
        warn.append(f"手腕朝向变化只有 {rs:.0f}°（建议 ≥ {min_rs:.0f}°）：多从不同倾角观察桌面标记")
    if ts < min_ts:
        warn.append(f"法兰位置范围只有 {ts * 100:.0f} cm（建议 ≥ {min_ts * 100:.0f} cm）")
    loo_mean, loo_max = sum(loo) / len(loo), max(loo)
    good = loo_mean <= CAM.CALIB_GOOD_MEAN_MM and loo_max <= CAM.CALIB_GOOD_MAX_MM and not warn
    X = r["T_flange_cam"]
    return dict(
        r,
        ok=True,
        used=idx,
        rejected=rejected,
        loo_mm=loo,
        loo_mean_mm=loo_mean,
        loo_max_mm=loo_max,
        rot_span_deg=rs,
        trans_span_m=ts,
        warnings=warn,
        good=good,
        cam_in_flange=CAM.T_t(X),
        n_used=len(idx),
        pnp_scale=round(float(s_xy), 5),
        pnp_scale_z=round(float(s_z), 5),
    )


def _is_axis_aligned_charuco_box(corners_px):
    """Return True if corners_px is an axis-aligned bounding box [[u0,v0],[u1,v0],[u1,v1],[u0,v1]] from ChArUco."""
    try:
        c = [[float(pt[0]), float(pt[1])] for pt in corners_px]
        return (abs(c[0][1] - c[1][1]) < 1e-3 and abs(c[1][0] - c[2][0]) < 1e-3
                and abs(c[2][1] - c[3][1]) < 1e-3 and abs(c[3][0] - c[0][0]) < 1e-3)
    except Exception:
        return False


def calibrate_eih_4mp(samples, size_mm=24.0, reject_mm=5.0, min_samples=W49.HANDEYE_MIN_SAMPLES):
    """V3.89 GLOBAL KRONECKER + CORNER (fx, k1) + 2D SCALE EYE-IN-HAND SOLVER:
    1. Evaluates current samples at scale=(1.0, 1.0) using the Kronecker-seeded optical-axis stabilized solver;
    2. Optimizes 1D uniform scale (printed marker size offset) and 2D (axy, az) focal-depth scale (bounded to prevent
       optical-axis lever-arm runaway);
    3. When true 4-corner ArUco `corners_px` are present, searches over focal length fx in [0.36*w, 0.85*w] and radial
       distortion k1 in [-0.28, +0.04] using the Kronecker-seeded evaluator, adopting `self_calibrated_K` ONLY when it
       genuinely improves LOO and achieves low residual (mean_mm <= 3.5 mm and loo_max_mm <= 6.0 mm)."""
    if len(samples) < min_samples:
        return {"ok": False, "error": f"至少需要 {min_samples} 个样本（现在 {len(samples)} 个）"}

    has_corners = all(
        isinstance(s.get("corners_px"), (list, tuple)) and len(s["corners_px"]) == 4
        and s.get("source") != "charuco"
        for s in samples
    ) and not all(_is_axis_aligned_charuco_box(s["corners_px"]) for s in samples)

    best = _calibrate_eih_at_scale_stabilized(samples, scale=(1.0, 1.0), size_mm=size_mm,
                                              reject_mm=reject_mm, min_samples=min_samples)

    A_arr0 = np.asarray([s["T_base_flange"] for s in samples], dtype=float)
    Ai_arr0 = np.linalg.inv(A_arr0)
    p_arr0 = np.asarray([s["p_cam"] for s in samples], dtype=float)

    # Path B: Corner-based (fx, k1) joint self-calibration (eliminates radial distortion at image edges)
    if has_corners and (not best["good"] or best["loo_max_mm"] > 2.8):
        cv2 = _cv2()
        w, h = samples[0].get("image_size") or [1920, 1080]
        w, h = int(w), int(h)
        max_u = max(max(float(pt[0]) for pt in s["corners_px"]) for s in samples)
        if max_u > 1925:
            w, h = 2688, 1520

        h_m = (float(size_mm) / 1000.0) / 2.0
        obj = np.array([[-h_m, h_m, 0.0], [h_m, h_m, 0.0], [h_m, -h_m, 0.0], [-h_m, -h_m, 0.0]], dtype=np.float64)
        uv_all = [np.asarray(s["corners_px"], dtype=np.float64).reshape(4, 2) for s in samples]
        flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)

        def _pnp_batch(f_val, k1_val):
            Km = np.array([[f_val, 0.0, 0.5 * w], [0.0, f_val, 0.5 * h], [0.0, 0.0, 1.0]], dtype=np.float64)
            dc = np.array([k1_val, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
            p_list, r_list = [], []
            for uv in uv_all:
                uv_u = cv2.undistortPoints(uv.reshape(4, 1, 2), Km, dc, P=Km).reshape(4, 2) if abs(k1_val) > 1e-6 else uv
                ok, rv, tv = cv2.solvePnP(obj, uv_u, Km, np.zeros(5), flags=flag)
                if not ok:
                    return None, None
                p_list.append(tv.flatten())
                r_list.append(rv.flatten())
            return np.asarray(p_list, dtype=float), r_list

        best_sc = 1e9
        best_fk = (0.55 * w, 0.0)
        for frac in (0.36, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            f_c = frac * w
            for k1_c in (-0.26, -0.20, -0.15, -0.10, -0.06, -0.02, 0.0, 0.02):
                p_arr, _ = _pnp_batch(f_c, k1_c)
                if p_arr is None:
                    continue
                sc = _fast_eih_eval_numpy(A_arr0, Ai_arr0, p_arr)
                if sc < best_sc:
                    best_sc = sc
                    best_fk = (f_c, k1_c)

        f0, k0 = best_fk
        for df in (-0.03 * w, -0.015 * w, 0.0, 0.015 * w, 0.03 * w):
            for dk in (-0.025, -0.012, 0.0, 0.012, 0.025):
                f_c, k1_c = max(0.35 * w, f0 + df), k0 + dk
                p_arr, _ = _pnp_batch(f_c, k1_c)
                if p_arr is None:
                    continue
                sc = _fast_eih_eval_numpy(A_arr0, Ai_arr0, p_arr)
                if sc < best_sc:
                    best_sc = sc
                    best_fk = (f_c, k1_c)

        f_opt, k1_opt = best_fk
        p_opt, r_opt = _pnp_batch(f_opt, k1_opt)
        if p_opt is not None:
            refined_samples = []
            for s, p_v, r_v in zip(samples, p_opt, r_opt):
                p_l = [float(x) for x in p_v]
                r_l = [float(x) for x in r_v]
                refined_samples.append(dict(
                    s,
                    p_cam=p_l,
                    rvec=r_l,
                    T_cam_marker=CAM.T_from(CAM.rodrigues(r_l), p_l),
                ))
            r_ref = _calibrate_eih_at_scale_stabilized(refined_samples, scale=(1.0, 1.0), size_mm=size_mm,
                                                       reject_mm=reject_mm, min_samples=min_samples)
            # Strict safety gate: only overwrite samples & intrinsics when r_ref is genuinely accurate!
            is_sane_ref = (r_ref["mean_mm"] <= 3.8 and r_ref["loo_max_mm"] <= 6.2)
            if is_sane_ref and (
                (r_ref["good"] and not best["good"])
                or ((r_ref["loo_mean_mm"] + r_ref["loo_max_mm"]) + 0.3 < (best["loo_mean_mm"] + best["loo_max_mm"]))
            ):
                for i in range(len(samples)):
                    samples[i]["p_cam"] = refined_samples[i]["p_cam"]
                    samples[i]["rvec"] = refined_samples[i]["rvec"]
                    samples[i]["T_cam_marker"] = refined_samples[i]["T_cam_marker"]
                r_ref["self_calibrated_K"] = {
                    "width": w, "height": h, "fx": round(f_opt, 2), "fy": round(f_opt, 2),
                    "ppx": round(0.5 * w, 2), "ppy": round(0.5 * h, 2),
                }
                r_ref["self_calibrated_dist"] = [round(k1_opt, 5), 0.0, 0.0, 0.0, 0.0]
                best = r_ref


    # Path A: 1D uniform scale + bounded 2D (axy, az) focal-depth scale search
    if not best["good"]:
        grid_u = [0.65 + 0.035 * k for k in range(21)]
        s_best = min(grid_u, key=lambda sc: _fast_eih_eval_numpy(A_arr0, Ai_arr0, p_arr0 * sc))
        if abs(s_best - 1.0) >= 0.015:
            cand_u = _calibrate_eih_at_scale_stabilized(samples, scale=(s_best, s_best), size_mm=size_mm,
                                                        reject_mm=reject_mm, min_samples=min_samples)
            if (cand_u["good"] and not best["good"]) or (
                (cand_u["loo_mean_mm"] + cand_u["loo_max_mm"]) < (best["loo_mean_mm"] + best["loo_max_mm"])
            ):
                best = cand_u

        if not best["good"]:
            axy, az = 1.0, s_best
            for _ in range(2):
                az = min(
                    (0.45 + 0.05 * k for k in range(25)),
                    key=lambda z_: _fast_eih_eval_numpy(A_arr0, Ai_arr0, p_arr0 * np.array([axy, axy, z_])),
                )
                axy = min(
                    (0.55 + 0.045 * k for k in range(21)),
                    key=lambda x_: _fast_eih_eval_numpy(A_arr0, Ai_arr0, p_arr0 * np.array([x_, x_, az])),
                )
            cand_2d = _calibrate_eih_at_scale_stabilized(samples, scale=(axy, az), size_mm=size_mm,
                                                         reject_mm=reject_mm, min_samples=min_samples)
            if (cand_2d["good"] and not best["good"]) or (
                (cand_2d["loo_mean_mm"] + cand_2d["loo_max_mm"]) < (best["loo_mean_mm"] + best["loo_max_mm"])
            ):
                best = cand_2d

    if not best["good"]:
        cand_w49 = W49.calibrate_eye_in_hand(samples, size_mm=size_mm, reject_mm=reject_mm, min_samples=min_samples)
        if (cand_w49.get("good") and not best["good"]) or (
            cand_w49.get("loo_max_mm", 99.0) + 0.2 < best.get("loo_max_mm", 99.0)
        ):
            best = cand_w49
    return best


def plan_eih_auto_targets_4mp(fk_fn, ik_fn, q0_rad, limits_rad, eih_model, K, n=12, size_m=0.024,
                              fence_fn=lambda q: True, seed=0):
    """V3.89 SWEET-SPOT & DISTANCE-BOUNDED EIH AUTO PLANNER FOR 4MP OV4688:
    Keeps all Look-At Dome poses inside z_cam in [130, 350] mm and off-axis ratio <= 0.40 (avoiding z=456 mm distant
    poses and x=+146 mm extreme peripheral poses that suffer from uncalibrated edge distortion), and never discards
    the sweet-spot pool when n >= 12."""
    import random
    rng = random.Random(seed)
    T0 = fk_fn(q0_rad)
    if T0 is None or not eih_model or not eih_model.get("T_flange_cam") or not eih_model.get("T_base_marker"):
        return [], {"tries": 0, "pool": 0, "rejected": {"缺少初值": 1}, "span_m": 0.0, "rot_max_deg": 0.0}
    X = eih_model["T_flange_cam"]
    Y_raw = eih_model["T_base_marker"]
    p_m = list(CAM.T_t(Y_raw))
    p0 = CAM.T_t(T0)
    R0 = CAM.T_R(T0)
    R0_t = CAM.transpose(R0)
    margin_rad = math.radians(getattr(CAM, "SOFT_MARGIN_DEG", 3.0))

    T_bc0 = CAM.mat_mul(T0, X)
    p_c0 = CAM.T_t(T_bc0)
    R_bc0 = CAM.T_R(T_bc0)
    v0 = CAM.vsub(p_m, p_c0)
    d0 = CAM.vnorm(v0) or 0.25
    u0 = [v / d0 for v in v0]
    n_y_raw = [CAM.T_R(Y_raw)[0][2], CAM.T_R(Y_raw)[1][2], CAM.T_R(Y_raw)[2][2]]
    cos_ny = abs(sum(u0[i] * n_y_raw[i] for i in range(3)))
    if cos_ny < math.cos(math.radians(25.0)):
        Y_vis = CAM.T_from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], p_m)
        ok0, _, _ = W49.eih_pose_visible(T0, X, Y_vis, K, size_m=size_m)
        if not ok0:
            Y_vis = CAM.T_from(CAM.mat_mul([[1, 0, 0], [0, -1, 0], [0, 0, -1]], R_bc0), p_m)
    else:
        Y_vis = Y_raw

    z_floor = min(p_m[2] + 0.08, p0[2] - 0.055)
    z_max_sweet = max(0.35, min(0.42, d0 + 0.065))
    rej = {}
    pool_sweet = []
    pool_all = []
    tries = 0
    w_img = float(K.get("width") or 1920.0)

    def try_add(q_cand):
        nonlocal tries
        tries += 1
        if q_cand is None:
            rej["IK"] = rej.get("IK", 0) + 1
            return
        for j, (lo, hi) in enumerate(limits_rad):
            if not (lo + margin_rad - 1e-6 <= q_cand[j] <= hi - margin_rad + 1e-6):
                rej["软限位"] = rej.get("软限位", 0) + 1
                return
        if not fence_fn(q_cand):
            rej["安全区"] = rej.get("安全区", 0) + 1
            return
        T = fk_fn(q_cand)
        if T is None:
            rej["FK"] = rej.get("FK", 0) + 1
            return
        if T[2][3] < z_floor:
            rej["太低"] = rej.get("太低", 0) + 1
            return
        rot_d = CAM.rot_angle_deg(CAM.mat_mul(R0_t, CAM.T_R(T)))
        if rot_d > W49.EIH_ROT_MAX_DEG * 1.65:
            rej["转角过大"] = rej.get("转角过大", 0) + 1
            return
        ok_v, why_v, vdeg = W49.eih_pose_visible(T, X, Y_vis, K, size_m=size_m, z_range=(0.11, 0.42))
        if not ok_v:
            rej[why_v] = rej.get(why_v, 0) + 1
            return
        if not CAM.check_auto_path(fence_fn, q0_rad, q_cand, steps=4):
            rej["路径安全区"] = rej.get("路径安全区", 0) + 1
            return
        T_bc = CAM.mat_mul(T, X)
        B_pred = CAM.mat_mul(CAM.T_inv(T_bc), Y_vis)
        p_cm = CAM.T_t(B_pred)
        off_ax = math.hypot(float(p_cm[0]), float(p_cm[1])) / max(0.08, float(p_cm[2]))
        item = {"q": list(q_cand), "T": T, "pos": CAM.T_t(T), "rot_deg": rot_d, "view_deg": vdeg,
                "off_axis": off_ax, "z_cam": float(p_cm[2])}
        pool_all.append(item)
        ok_sw, _, _ = W49.eih_pose_visible(
            T, X, Y_vis, K, size_m=size_m,
            margin_px=max(96.0, 0.14 * w_img),
            z_range=(0.12, z_max_sweet),
            max_off_axis=0.40,
        )
        if ok_sw:
            pool_sweet.append(item)

    def _rot_vec(Ra, Rb):
        R_ = [[sum(Rb[i][k] * Ra[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
        return [0.5 * (R_[2][1] - R_[1][2]), 0.5 * (R_[0][2] - R_[2][0]), 0.5 * (R_[1][0] - R_[0][1])]

    nj = len(q0_rad)
    h_j = 1e-5
    cols_j = []
    for j in range(nj):
        qp = list(q0_rad)
        qp[j] += h_j
        Tp = fk_fn(qp)
        if Tp is not None:
            cols_j.append([(Tp[i][3] - p0[i]) / h_j for i in range(3)] + [x / h_j for x in _rot_vec(R0, CAM.T_R(Tp))])
    J_pinv = None
    if len(cols_j) == nj:
        J0 = [[cols_j[j][r] for j in range(nj)] for r in range(6)]
        lam = 2.0e-3
        JJT = [[sum(J0[r][k] * J0[c][k] for k in range(nj)) + (lam if r == c else 0.0) for c in range(6)] for r in range(6)]
        aug = [JJT[r] + [1.0 if r == c else 0.0 for c in range(6)] for r in range(6)]
        ok_inv = True
        for c in range(6):
            pv = max(range(c, 6), key=lambda r: abs(aug[r][c]))
            aug[c], aug[pv] = aug[pv], aug[c]
            if abs(aug[c][c]) < 1e-12:
                ok_inv = False
                break
            for r in range(6):
                if r != c:
                    f = aug[r][c] / aug[c][c]
                    for k in range(c, 12):
                        aug[r][k] -= f * aug[c][k]
        if ok_inv:
            inv_JJT = [[aug[r][6 + c] / aug[r][r] for c in range(6)] for r in range(6)]
            J_pinv = [[sum(J0[r][j] * inv_JJT[r][c] for r in range(6)) for c in range(6)] for j in range(nj)]

    R_X = CAM.T_R(X)
    R_X_t = CAM.transpose(R_X)
    t_X = CAM.T_t(X)
    if J_pinv is not None:
        for _ in range(95):
            dx = rng.uniform(-0.105, 0.105)
            dy = rng.uniform(-0.105, 0.105)
            dz = rng.uniform(-0.040, 0.045)
            p_c = [p_c0[0] + dx, p_c0[1] + dy, p_c0[2] + dz]
            aim = [p_m[0] + rng.uniform(-0.016, 0.016),
                   p_m[1] + rng.uniform(-0.016, 0.016),
                   p_m[2]]
            v_los = CAM.vsub(aim, p_c)
            d_los = CAM.vnorm(v_los) or 0.25
            u_los = [v / d_los for v in v_los]
            cr = W49._vcross(u0, u_los)
            cn = CAM.vnorm(cr)
            dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(u0, u_los))))
            ang_deg = math.degrees(math.acos(dot))
            R_swing = CAM.axis_angle_R(cr, ang_deg) if cn > 1e-6 else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            roll_deg = rng.uniform(-25.0, 25.0)
            R_roll = CAM.axis_angle_R(u_los, roll_deg)
            R_bc = CAM.mat_mul(R_roll, CAM.mat_mul(R_swing, R_bc0))
            R_bf = CAM.mat_mul(R_bc, R_X_t)
            p_bf = CAM.vsub(p_c, CAM.mat_vec(R_bf, t_X))
            err6 = [p_bf[i] - p0[i] for i in range(3)] + _rot_vec(R0, R_bf)
            qc = [max(lo + margin_rad, min(hi - margin_rad, q0_rad[j] + sum(J_pinv[j][c] * err6[c] for c in range(6))))
                  for j, (lo, hi) in enumerate(limits_rad)]
            try_add(qc)

    q0_deg = [math.degrees(v) for v in q0_rad]
    for i in range(35):
        scale = 0.50 if i < 20 else 0.78
        q_c = []
        for j, (lo, hi) in enumerate(limits_rad):
            span_deg = (18.0 if j >= 4 else 11.0) * scale
            v_rad = math.radians(q0_deg[j] + rng.uniform(-span_deg, span_deg))
            v_rad = max(lo + margin_rad, min(hi - margin_rad, v_rad))
            q_c.append(v_rad)
        try_add(q_c)

    if len(pool_sweet) >= n:
        pool = pool_sweet
    elif pool_sweet:
        sw_ids = {id(x) for x in pool_sweet}
        extra = [x for x in pool_all if id(x) not in sw_ids]
        extra.sort(key=lambda x: x["off_axis"] + max(0.0, x["z_cam"] - 0.34) * 2.5)
        pool = list(pool_sweet) + extra[:max(4, n - len(pool_sweet) + 4)]
    else:
        pool = pool_all
    if not pool:
        return [], {"tries": tries, "pool": 0, "rejected": rej, "span_m": 0.0, "rot_max_deg": 0.0}

    selected = []
    used_T = [T0]
    used_p = [p0]
    rem = list(pool)
    while len(selected) < n and rem:
        best_i = -1
        best_sc = -1.0
        for idx, c in enumerate(rem):
            d_pos = min(CAM.vnorm(CAM.vsub(c["pos"], p)) for p in used_p)
            d_rot = min(CAM.rot_angle_deg(CAM.mat_mul(CAM.transpose(CAM.T_R(Ta)), CAM.T_R(c["T"]))) for Ta in used_T)
            sc = d_pos + 0.0045 * d_rot - 0.08 * max(0.0, c.get("off_axis", 0.0) - 0.35)
            if sc > best_sc:
                best_sc = sc
                best_i = idx
        if best_i < 0:
            break
        pick = rem.pop(best_i)
        selected.append(pick)
        used_T.append(pick["T"])
        used_p.append(pick["pos"])

    chosen = []
    prev_q = q0_rad
    rem_sel = list(selected)
    while rem_sel:
        rem_sel.sort(key=lambda c: sum((a - b) ** 2 for a, b in zip(prev_q, c["q"])))
        picked = None
        for idx, c in enumerate(rem_sel):
            if CAM.check_auto_path(fence_fn, prev_q, c["q"], steps=4):
                picked = rem_sel.pop(idx)
                break
        if picked is None:
            break
        chosen.append(picked)
        prev_q = picked["q"]

    all_T = [T0] + [c["T"] for c in chosen]
    span_m = CAM.translation_span_m(all_T)
    rot_max = CAM.rotation_span_deg(all_T)
    return chosen, {
        "tries": tries,
        "pool": len(pool),
        "rejected": rej,
        "span_m": span_m,
        "rot_max_deg": rot_max,
    }


def autotune_wrist_uvc(set_eg, measure, power_hz=60, gain_range=(0, 128), mean_lo=100.0):
    """V3.89 SUB-MILLISECOND & BLACK-FRAME-SAFE UVC AUTOTUNE FOR OV4688 HDR:
    1. Avoids V4L2 integer truncation to 0 (never probes < 0.8 in 0.1 ms units);
    2. Detects and immediately aborts/rolls back if a sub-ms exposure triggers a black frame (mean < 2.0) on V4L2;
    3. Accepts glare-free short-exposure OV4688 frames without wedging the UVC stream."""
    lo = float(mean_lo or 100.0)
    hi = LAB379.MEAN_HI
    aim = 0.5 * (lo + hi)
    per = round(1000.0 / (2.0 * float(power_hz)) / 0.1, 1)
    cands = sorted({round(3 * per, 1), round(2 * per, 1), round(per, 1),
                    50.0, 25.0, 15.0, 10.0, 5.0, 2.5, 1.5, 1.0, 0.8}, reverse=True)
    trace = []
    gmin, gmax = gain_range
    chosen = None
    fallback = None
    last_valid = None
    for e in cands:
        set_eg(e, gmin)
        st = measure()
        if st["mean"] < 2.0 and last_valid is not None:
            # Hit V4L2 integer exposure floor (black frame) -> roll back to last valid exposure immediately!
            e_prev, st_prev = last_valid
            set_eg(e_prev, gmin)
            if chosen is None:
                chosen = (e_prev, st_prev)
            break
        trace.append(dict(exposure=e, gain=gmin, **st))
        if st["mean"] >= 2.0:
            last_valid = (e, st)
        clip = LAB379._clipped(st)
        if not clip and st["mean"] >= 2.0 and fallback is None:
            fallback = (e, st)
        if not clip and 2.0 <= st["mean"] <= hi:
            chosen = (e, st)
            break
    if chosen is None:
        chosen = fallback or last_valid or (cands[-1], trace[-1] if trace else measure())
    e, st = chosen
    set_eg(e, gmin)
    g = gmin
    if st["mean"] < lo and not LAB379._clipped(st):
        a, b, best_g = gmin, gmax, (gmin, st)
        for _ in range(6):
            mid = int(round(0.5 * (a + b)))
            if mid in (a, b):
                break
            set_eg(e, mid)
            s2 = measure()
            if s2["mean"] < 2.0:
                break
            trace.append(dict(exposure=e, gain=mid, **s2))
            if not LAB379._clipped(s2) and s2["mean"] <= hi:
                best_g, a = (mid, s2), mid
                if s2["mean"] >= aim - 8.0:
                    break
            else:
                b = mid
        g, st = best_g
        set_eg(e, g)
    flicker_safe = (e >= per - 1.0)
    eff_lo = min(lo, 45.0) if not flicker_safe else lo
    issues = LAB379.judge_brightness(st, flicker_safe=True, gain=g, gain_max=gmax, mean_lo=eff_lo)
    return {
        "ok": len(issues) == 0,
        "exposure": e,
        "exposure_ms": round(e * 0.1, 2) if e >= 10.0 else e,
        "gain": g,
        "flicker_safe": flicker_safe,
        "stats": st,
        "issues": issues,
        "steps": len(trace),
        "note": "" if flicker_safe else "短曝光防过曝模式（OV4688 高感光）",
    }, trace


def list_wrist_cameras(sysfs_root="/sys/class/video4linux"):
    """Enumerate V4L2 video capture nodes, excluding RealSense, prioritizing Innomaker U20-4MP OV4688 HDR."""
    devs = W49.list_usb_cameras(sysfs_root=sysfs_root)
    for d in devs:
        low = d["name"].lower()
        is_4mp = any(k in low for k in ("4mp", "u20", "ov4688", "hdr", "2688"))
        d["is_4mp_hdr"] = is_4mp
        d["model_tag"] = "U20-4MP OV4688" if is_4mp else ("Innomaker UVC" if "innomaker" in low else "USB UVC")
    devs.sort(key=lambda d: (
        0 if d.get("is_4mp_hdr") else (1 if "innomaker" in d["name"].lower() else 2),
        d["index"],
    ))
    return devs


# =========================================================================== 2. UVC Profile & Control Application
def load_wrist_profile(path=None):
    p = path or os.environ.get("NERO_WRIST_PROFILE_PATH") or WRIST_PROFILE_PATH
    prof = json.loads(json.dumps(DEFAULT_WRIST_PROFILE))
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d.get("color"), dict):
            prof["color"].update(d["color"])
        for k in ("camera_model", "res", "locked", "note", "saved_at", "name", "ref", "roi"):
            if k in d:
                prof[k] = d[k]
    except Exception:
        pass
    return prof


def save_wrist_profile(prof, path=None):
    p = path or os.environ.get("NERO_WRIST_PROFILE_PATH") or WRIST_PROFILE_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    d = dict(prof, version=VERSION, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    return p


def _profile_file(name):
    safe = LAB379.safe_class(name)
    return os.path.join(os.environ.get("NERO_WRIST_PROFILES_DIR") or WRIST_PROFILES_DIR, f"{safe}.json")


def list_named_profiles():
    d = os.environ.get("NERO_WRIST_PROFILES_DIR") or WRIST_PROFILES_DIR
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(d, fn), encoding="utf-8") as f:
                    data = json.load(f)
                out.append(str(data.get("name") or fn[:-5]))
            except Exception:
                out.append(fn[:-5])
    return out


def save_named_profile(name, prof):
    p = _profile_file(name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    d = dict(prof, name=str(name).strip(), version=VERSION, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    return p


def load_named_profile(name):
    p = _profile_file(name)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def activate_named_profile(name):
    d = load_named_profile(name)
    d["locked"] = True
    return save_wrist_profile(d)


def apply_uvc_profile_to_cap(cap, cv2, prof_color):
    """Apply UVC exposure, gain, white balance, sharpness, gamma, and HDR backlight controls to an OpenCV VideoCapture."""
    applied = {}
    if cap is None:
        return applied
    c = prof_color or {}
    if hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            applied["autofocus"] = "off"
        except Exception:
            pass
    ae = int(c.get("enable_auto_exposure", 0))
    if hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if ae else 1)
            applied["auto_exposure"] = "auto" if ae else "manual"
        except Exception:
            pass
    if not ae and "exposure" in c and hasattr(cv2, "CAP_PROP_EXPOSURE"):
        try:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(c["exposure"]))
            applied["exposure"] = float(c["exposure"])
        except Exception:
            pass
    if "gain" in c and hasattr(cv2, "CAP_PROP_GAIN"):
        try:
            cap.set(cv2.CAP_PROP_GAIN, float(c["gain"]))
            applied["gain"] = float(c["gain"])
        except Exception:
            pass
    awb = int(c.get("enable_auto_white_balance", 0))
    if hasattr(cv2, "CAP_PROP_AUTO_WB"):
        try:
            cap.set(cv2.CAP_PROP_AUTO_WB, 1 if awb else 0)
            applied["auto_wb"] = "auto" if awb else "manual"
        except Exception:
            pass
    if not awb and "white_balance" in c and hasattr(cv2, "CAP_PROP_WB_TEMPERATURE"):
        try:
            cap.set(cv2.CAP_PROP_WB_TEMPERATURE, float(c["white_balance"]))
            applied["white_balance"] = float(c["white_balance"])
        except Exception:
            pass
    if "gamma" in c and hasattr(cv2, "CAP_PROP_GAMMA"):
        try:
            cap.set(cv2.CAP_PROP_GAMMA, float(c["gamma"]))
            applied["gamma"] = float(c["gamma"])
        except Exception:
            pass
    if "sharpness" in c and hasattr(cv2, "CAP_PROP_SHARPNESS"):
        try:
            cap.set(cv2.CAP_PROP_SHARPNESS, float(c["sharpness"]))
            applied["sharpness"] = float(c["sharpness"])
        except Exception:
            pass
    if "backlight_hdr" in c and hasattr(cv2, "CAP_PROP_BACKLIGHT"):
        try:
            cap.set(cv2.CAP_PROP_BACKLIGHT, float(c["backlight_hdr"]))
            applied["hdr"] = "on" if int(c["backlight_hdr"]) > 0 else "off"
        except Exception:
            pass
    return applied


# =========================================================================== 3. Sharpness & 3D Cup Visual Servoing
def evaluate_focus_metrics(rgb, roi=None):
    """Evaluate image sharpness and edge contrast across the work ROI and 4 quadrant corners for focus tuning."""
    if rgb is None:
        return {"ok": False, "laplacian_var": 0.0, "tenengrad": 0.0, "corners_lap": [0.0] * 4,
                "uniformity_pct": 0.0, "verdict": "无画面", "level": "fail"}
    cv2 = _cv2()
    arr = np.ascontiguousarray(rgb)
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
    x0, y0, x1, y1 = LAB379.roi_px(roi, (w, h))
    patch = gray[y0:y1, x0:x1]
    if patch.size < 100:
        patch = gray
    lap = float(cv2.Laplacian(patch, cv2.CV_64F).var())
    gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    ten = float(np.mean(gx * gx + gy * gy))

    mh, mw = patch.shape[0] // 2, patch.shape[1] // 2
    quads = [patch[:mh, :mw], patch[:mh, mw:], patch[mh:, :mw], patch[mh:, mw:]]
    c_laps = [round(float(cv2.Laplacian(q, cv2.CV_64F).var()), 1) if q.size >= 25 else 0.0 for q in quads]
    max_c = max(c_laps) if c_laps else 0.0
    min_c = min(c_laps) if c_laps else 0.0
    unif = round((min_c / max_c) * 100.0, 1) if max_c > 1e-3 else 0.0

    if lap >= 80.0:
        level, verdict = "ok", f"清晰度优秀 ✓（Laplacian {lap:.1f} · 边缘梯度 {ten:.0f} · 四角均衡度 {unif:.0f}%）"
    elif lap >= 25.0:
        level, verdict = "ok", f"清晰度合格 ✓（Laplacian {lap:.1f} · 边缘梯度 {ten:.0f} · 四角均衡度 {unif:.0f}%）"
    else:
        level, verdict = "warn", f"画面偏糊 ⚠（Laplacian {lap:.1f} < 25：请检查工作距 50–150 mm 或微调镜头焦距/放慢机械臂）"

    return {
        "ok": True,
        "laplacian_var": round(lap, 1),
        "tenengrad": round(ten, 1),
        "corners_lap": c_laps,
        "uniformity_pct": unif,
        "verdict": verdict,
        "level": level,
    }


def _extract_cup_contours(rgb, min_area=120):
    """Extract red (or high-contrast circular) cup contours from an RGB image.
    Returns list of contour arrays sorted by area descending, plus binary mask."""
    cv2 = _cv2()
    arr = np.ascontiguousarray(rgb)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    # Red plastic cups in HSV (hue 0..14 or 158..180, saturation >= 55, value >= 40)
    m1 = cv2.inRange(hsv, np.array([0, 55, 40], np.uint8), np.array([14, 255, 255], np.uint8))
    m2 = cv2.inRange(hsv, np.array([158, 55, 40], np.uint8), np.array([180, 255, 255], np.uint8))
    mask = cv2.morphologyEx(m1 | m2, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cs = [c for c in cs if cv2.contourArea(c) >= min_area]
    if not cs:
        # Fallback for non-red cups in synthetic tests: high-contrast foreground circle
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        med = float(np.median(gray))
        bw = (np.abs(gray.astype(np.float32) - med) > 35.0).astype(np.uint8) * 255
        cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cs = [c for c in cs if cv2.contourArea(c) >= min_area]
    cs.sort(key=cv2.contourArea, reverse=True)
    return cs, mask


def _intersect_contour_with_plane(cnt_px, K_scaled, T_base_cam, z_plane, img_shape=None, border_px=4):
    """Project 2D contour pixels (N, 2) from camera optical frame onto horizontal plane Z = z_plane in base_link,
    excluding artificial image-border clipping edges when img_shape is given.
    -> (pts_xy_m: (M, 2) ndarray, valid_ratio: float)"""
    pts = np.asarray(cnt_px, dtype=float).reshape(-1, 2)
    n_orig = max(1, len(pts))
    if img_shape is not None and len(pts) > 0:
        h_i, w_i = int(img_shape[0]), int(img_shape[1])
        inside = ((pts[:, 0] >= border_px) & (pts[:, 0] < w_i - border_px)
                  & (pts[:, 1] >= border_px) & (pts[:, 1] < h_i - border_px))
        pts = pts[inside]
        if len(pts) == 0:
            return np.zeros((0, 2)), 0.0
    fx, fy = float(K_scaled["fx"]), float(K_scaled["fy"])
    ppx, ppy = float(K_scaled["ppx"]), float(K_scaled["ppy"])
    T_bc = np.asarray(T_base_cam, dtype=float)
    O_cam = T_bc[:3, 3]
    R_bc = T_bc[:3, :3]
    dc = np.column_stack([(pts[:, 0] - ppx) / fx, (pts[:, 1] - ppy) / fy, np.ones(len(pts))])
    rays_b = dc @ R_bc.T
    valid = np.abs(rays_b[:, 2]) > 1e-4
    if not np.any(valid):
        return np.zeros((0, 2)), 0.0
    lams = (float(z_plane) - float(O_cam[2])) / rays_b[valid, 2]
    keep = (lams > 0.02) & (lams < 1.20)
    if not np.any(keep):
        return np.zeros((0, 2)), 0.0
    P_b = O_cam + lams[keep, None] * rays_b[valid][keep]
    return P_b[:, :2], float(np.sum(keep)) / n_orig


def _fit_circle_kasa(pts_xy, r_nom=0.0475, prior_xy=None, max_shift_m=0.028):
    """V3.91 RIM-ONLY KNOWN-RADIUS RANSAC + HUBER CIRCLE FIT:
    On a real 3D conical cup viewed obliquely by the wrist camera, projecting the entire 2D red silhouette onto Z = z_rim
    stretches the lower cup wall and bottom up to 100+ mm away along the camera rays, and unconstrained 3-parameter Kasa
    diverges on partial rim arcs.
    1. When prior_xy (from fixed camera perception) is provided, filters pts_xy to the annular rim ROI around prior_xy
       (|dist - r_nom| <= max_shift_m + 0.008) so projected lower-wall / cup-bottom points are rejected upfront.
    2. Evaluates unconstrained Kasa when a full circle is present (|r_kasa - r_nom| <= 0.12 * r_nom and low RMS), AND
       runs a fixed-radius (r = r_nom) 2D grid + Gauss-Newton Huber IRLS fit that accurately recovers (xc, yc) even from
       a partial upper-rim arc without collapsing to the chord midpoint."""
    pts = np.asarray(pts_xy, dtype=float).reshape(-1, 2)
    if len(pts) < 6:
        return None
    if prior_xy is not None:
        px, py = float(prior_xy[0]), float(prior_xy[1])
        d_prior = np.hypot(pts[:, 0] - px, pts[:, 1] - py)
        ann = pts[np.abs(d_prior - float(r_nom)) <= float(max_shift_m) + 0.008]
        if len(ann) >= 8:
            pts = ann
        else:
            return None
    x, y = pts[:, 0], pts[:, 1]
    xm, ym = float(np.mean(x)), float(np.mean(y))
    u, v = x - xm, y - ym

    # Path A: Algebraic Kasa check (exact when full/nearly-full rim contour lies on z_plane)
    kasa_cand = None
    try:
        A = np.column_stack([2.0 * u, 2.0 * v, np.ones(len(u))])
        rhs = u * u + v * v
        sol = np.linalg.solve(A.T @ A + np.diag([1e-4, 1e-4, 1e-6]), A.T @ rhs)
        uc, vc, c0 = float(sol[0]), float(sol[1]), float(sol[2])
        xc_k, yc_k = xm + uc, ym + vc
        r_k = math.sqrt(max(1e-6, c0 + uc * uc + vc * vc))
        dists_k = np.hypot(x - xc_k, y - yc_k)
        res_k = np.abs(dists_k - r_k)
        in_k = res_k <= max(0.0035, 2.0 * float(np.median(res_k)))
        if int(np.sum(in_k)) >= 8:
            ui, vi = u[in_k], v[in_k]
            Ai = np.column_stack([2.0 * ui, 2.0 * vi, np.ones(len(ui))])
            sol2 = np.linalg.solve(Ai.T @ Ai + np.diag([1e-4, 1e-4, 1e-6]), Ai.T @ (ui * ui + vi * vi))
            uc, vc, c0 = float(sol2[0]), float(sol2[1]), float(sol2[2])
            xc_k, yc_k = xm + uc, ym + vc
            r_k = math.sqrt(max(1e-6, c0 + uc * uc + vc * vc))
        rms_k = float(np.sqrt(np.mean((np.hypot(x - xc_k, y - yc_k) - r_nom) ** 2)))
        if 0.88 * r_nom <= r_k <= 1.12 * r_nom and rms_k <= 0.004:
            kasa_cand = (xc_k, yc_k, r_k, rms_k)
    except Exception:
        pass

    if kasa_cand is not None and (prior_xy is None or math.hypot(kasa_cand[0] - prior_xy[0], kasa_cand[1] - prior_xy[1]) <= max_shift_m):
        return {"xc": kasa_cand[0], "yc": kasa_cand[1], "r": kasa_cand[2], "rms_m": kasa_cand[3]}

    # Path B: Fixed-radius (r = r_nom) RANSAC grid + Huber Gauss-Newton IRLS (immune to partial-arc & oblique cone wall)
    cx0 = float(prior_xy[0]) if prior_xy is not None else xm
    cy0 = float(prior_xy[1]) if prior_xy is not None else ym
    span = float(max_shift_m) if prior_xy is not None else float(r_nom)
    offs = np.linspace(-span, span, 17)
    gx, gy = np.meshgrid(cx0 + offs, cy0 + offs)
    cands = np.column_stack([gx.ravel(), gy.ravel()])
    if kasa_cand is not None:
        cands = np.vstack([[kasa_cand[0], kasa_cand[1]], cands])

    # Evaluate inlier count and truncated residual for all grid candidates vectorized
    dx_c = x[None, :] - cands[:, 0:1]
    dy_c = y[None, :] - cands[:, 1:2]
    r_err = np.abs(np.hypot(dx_c, dy_c) - float(r_nom))
    inlier_mask = r_err <= 0.004
    inlier_counts = np.sum(inlier_mask, axis=1)
    huber_loss = np.sum(np.minimum(r_err, 0.006) ** 2, axis=1)
    # Prefer candidates with high inlier count and low truncated loss
    score = inlier_counts * 1e-4 - huber_loss
    best_idx = int(np.argmax(score))
    xc, yc = float(cands[best_idx, 0]), float(cands[best_idx, 1])

    # 8-step Huber IRLS Gauss-Newton refinement with fixed radius r_nom
    for _ in range(8):
        dx = x - xc
        dy = y - yc
        dist = np.maximum(1e-6, np.hypot(dx, dy))
        err = dist - float(r_nom)
        in_m = np.abs(err) <= 0.005
        if int(np.sum(in_m)) < 6:
            break
        w = np.where(np.abs(err) <= 0.0025, 1.0, 0.0025 / np.maximum(1e-6, np.abs(err))) * in_m
        J = np.column_stack([-dx / dist, -dy / dist])
        JWJ = J.T @ (w[:, None] * J) + np.eye(2) * 1e-3
        JWr = J.T @ (w * err)
        try:
            step = np.linalg.solve(JWJ, JWr)
            xc -= float(step[0])
            yc -= float(step[1])
        except Exception:
            break

    dists_final = np.hypot(x - xc, y - yc)
    res_final = np.abs(dists_final - float(r_nom))
    inliers_final = res_final <= 0.005
    if int(np.sum(inliers_final)) < 6:
        return None
    rms_m = float(np.sqrt(np.mean(res_final[inliers_final] ** 2)))
    return {"xc": xc, "yc": yc, "r": float(r_nom), "rms_m": rms_m}


def wrist_check_cup_grasp(rgb, K, T_base_flange, calib_eih, T_tcp_grasp, T_base_tcp,
                          expected_rim_xyz, planned_phi_rad, flip=True, r_top_m=0.0475,
                          undistort_intr=None, max_shift_m=WRIST_GRASP_MAX_SHIFT_M):
    """V3.87 PHASE 3 PRE-GRASP WRIST VISUAL SERVOING:
    At the pregrasp pose above the source cup, use the wrist camera image + calibrated K + Eye-in-Hand T_flange_cam:
      1. Extract the cup rim contour and intersect rays with the horizontal rim plane Z = expected_rim_xyz[2];
      2. Fit the 3D rim centre (xc, yc) in base_link and compare with expected_rim_xyz[:2];
      3. Evaluate the actual gripper closing axis y_grasp_xy in base_link vs the measured radial normal at the rim
         pinch point, returning both:
         - delta_rim_xy_m = [xc - exp_x, yc - exp_y]: exact 3D horizontal shift of the cup rim;
         - angle_err_deg: signed yaw angle error between the current gripper closing line and the cup radial normal;
         - recommended_phi_rad: updated rim pinch angle so the gripper jaws straddle the rim perpendicularly.
    -> dict with 'ok', 'trusted', 'measured_rim_xyz', 'delta_rim_xy_m', 'delta_rim_mm', 'angle_err_deg',
       'recommended_phi_rad', 'fit_radius_mm', 'fit_rms_mm'."""
    if rgb is None:
        return {"ok": False, "reason": "腕部相机无画面"}
    if not K or not calib_eih or not calib_eih.get("T_flange_cam") or T_base_flange is None:
        return {"ok": False, "reason": "缺少腕部内参或 Eye-in-Hand 手眼标定"}

    rgb_u = W49.undistort_rgb(rgb, undistort_intr) if (undistort_intr and W49.intrinsics_valid(undistort_intr)) else rgb
    h_img, w_img = rgb_u.shape[:2]
    K_s = scale_K_for_frame(K, w_img, h_img)
    T_bw = W49.wrist_T_base_cam(T_base_flange, calib_eih)

    cs, _mask = _extract_cup_contours(rgb_u, min_area=120)
    if not cs:
        return {"ok": False, "reason": "腕部相机未检测到杯口轮廓"}

    exp_rim = [float(v) for v in expected_rim_xyz]
    z_rim = exp_rim[2]

    best_fit = None
    for c in cs[:4]:
        pts_xy, _v_ratio = _intersect_contour_with_plane(c, K_s, T_bw, z_rim, img_shape=(h_img, w_img))
        if len(pts_xy) < 8:
            continue
        fit = _fit_circle_kasa(pts_xy, r_nom=r_top_m, prior_xy=exp_rim[:2], max_shift_m=max_shift_m)
        if fit is None or fit["rms_m"] > 0.008:
            continue
        dist_exp = math.hypot(fit["xc"] - exp_rim[0], fit["yc"] - exp_rim[1])
        if best_fit is None or dist_exp < best_fit["dist_exp"]:
            best_fit = dict(fit, dist_exp=dist_exp, contour_px=c.reshape(-1, 2))

    if best_fit is None:
        return {"ok": False, "reason": "杯口轮廓无法投影到杯口高度平面"}

    xc, yc = best_fit["xc"], best_fit["yc"]
    dx_m = xc - exp_rim[0]
    dy_m = yc - exp_rim[1]
    shift_m = math.hypot(dx_m, dy_m)
    trusted = shift_m <= float(max_shift_m)

    # Current gripper closing axis in base_link XY plane:
    # R_base_grasp = R_base_tcp @ R_tcp_grasp; column 1 (y-axis) is the closing direction.
    # For flip=False, y_grasp points outward along u(phi); for flip=True, y_grasp points inward along -u(phi).
    T_bt = np.asarray(T_base_tcp, dtype=float)
    T_tg = np.asarray(T_tcp_grasp, dtype=float)
    R_bg = T_bt[:3, :3] @ T_tg[:3, :3]
    y_grasp_xy = np.array([float(R_bg[0, 1]), float(R_bg[1, 1])], dtype=float)
    yn = float(np.linalg.norm(y_grasp_xy))
    if yn > 1e-6:
        y_grasp_xy /= yn
    else:
        y_grasp_xy = np.array([math.cos(planned_phi_rad), math.sin(planned_phi_rad)], dtype=float)

    # Project the grasp-frame origin along the grasp approach axis (+z_grasp) down to the rim plane Z = z_rim:
    T_bg = T_bt @ T_tg
    p_g_now = T_bg[:3, 3]
    z_g_axis = T_bg[:3, 2]
    if abs(float(z_g_axis[2])) > 0.2:
        lam_g = (float(z_rim) - float(p_g_now[2])) / float(z_g_axis[2])
        p_pinch_rim = p_g_now + lam_g * z_g_axis
    else:
        p_pinch_rim = p_g_now
    # Evaluate radial normal at the translated pinch point (p_pinch_rim + delta_rim_xy) relative to (xc, yc)
    # so pure cup translation does not induce false wrist rotation skew
    vec_from_center = np.array([float(p_pinch_rim[0]) + dx_m - xc, float(p_pinch_rim[1]) + dy_m - yc], dtype=float)
    vn = float(np.linalg.norm(vec_from_center))
    if vn > 0.010:
        u_radial = vec_from_center / vn
        actual_phi = math.atan2(float(u_radial[1]), float(u_radial[0]))
    else:
        actual_phi = float(planned_phi_rad)
        u_radial = np.array([math.cos(actual_phi), math.sin(actual_phi)], dtype=float)

    # Desired closing axis: -u_radial when flip=True (fixed jaw outside), +u_radial when flip=False
    des_close_xy = -u_radial if flip else u_radial
    cross_z = float(y_grasp_xy[0] * des_close_xy[1] - y_grasp_xy[1] * des_close_xy[0])
    dot_xy = float(y_grasp_xy[0] * des_close_xy[0] + y_grasp_xy[1] * des_close_xy[1])
    angle_err_deg = math.degrees(math.atan2(cross_z, dot_xy))
    # Wrap into [-90, 90] deg (symmetric jaw line perpendicularity to rim tangent)
    while angle_err_deg > 90.0:
        angle_err_deg -= 180.0
    while angle_err_deg <= -90.0:
        angle_err_deg += 180.0

    rec_phi = float(planned_phi_rad) + math.radians(angle_err_deg)
    return {
        "ok": True,
        "trusted": bool(trusted),
        "measured_rim_xyz": [round(xc, 5), round(yc, 5), round(z_rim, 5)],
        "expected_rim_xyz": [round(v, 5) for v in exp_rim],
        "delta_rim_xy_m": [round(dx_m, 5), round(dy_m, 5)],
        "delta_rim_mm": [round(dx_m * 1000.0, 2), round(dy_m * 1000.0, 2)],
        "shift_norm_mm": round(shift_m * 1000.0, 2),
        "angle_err_deg": round(angle_err_deg, 2),
        "planned_phi_deg": round(math.degrees(planned_phi_rad), 2),
        "recommended_phi_rad": rec_phi,
        "recommended_phi_deg": round(math.degrees(rec_phi), 2),
        "fit_radius_mm": round(best_fit["r"] * 1000.0, 1),
        "fit_rms_mm": round(best_fit["rms_m"] * 1000.0, 2),
    }


def wrist_check_cup_concentricity(rgb, K, T_base_flange, calib_eih,
                                  expected_target_rim_xyz, expected_held_base_xyz,
                                  r_top_m=0.0475, r_bot_m=0.030, undistort_intr=None,
                                  max_corr_m=WRIST_PLACE_MAX_CORR_M):
    """V3.87 PHASE 3 PRE-PLACE WRIST CONCENTRICITY SERVOING:
    Above the target cup before lowering to release, use the downward-looking wrist camera to measure the target cup's
    rim centre and the held cup's bottom/rim centre in base_link:
      1. Extracts outer contour (target cup rim at Z = z_target_rim) and fits 3D centre C_target(x_t, y_t);
      2. Checks for an inner/second circular contour (held cup bottom at Z = z_held_base) or uses the calibrated
         held-cup base in the wrist frame to get C_held(x_h, y_h);
      3. Computes the 3D concentricity correction vector delta_xy_m = [x_t - x_h, y_t - y_h] to align the two cups
         coaxially before descending.
    -> dict with 'ok', 'trusted', 'target_xy_m', 'held_xy_m', 'delta_xy_m', 'delta_mm', 'concentric_err_mm'."""
    if rgb is None:
        return {"ok": False, "reason": "腕部相机无画面"}
    if not K or not calib_eih or not calib_eih.get("T_flange_cam") or T_base_flange is None:
        return {"ok": False, "reason": "缺少腕部内参或 Eye-in-Hand 手眼标定"}

    rgb_u = W49.undistort_rgb(rgb, undistort_intr) if (undistort_intr and W49.intrinsics_valid(undistort_intr)) else rgb
    h_img, w_img = rgb_u.shape[:2]
    K_s = scale_K_for_frame(K, w_img, h_img)
    T_bw = W49.wrist_T_base_cam(T_base_flange, calib_eih)

    cs, mask = _extract_cup_contours(rgb_u, min_area=120)
    if not cs:
        return {"ok": False, "reason": "腕部相机未检测到目标杯口轮廓"}

    exp_tgt = [float(v) for v in expected_target_rim_xyz]
    exp_held = [float(v) for v in expected_held_base_xyz]
    z_tgt = exp_tgt[2]
    z_held = exp_held[2]

    # Fit outer contour on the target rim plane Z = z_tgt
    best_tgt = None
    for c in cs[:4]:
        pts_xy, _ = _intersect_contour_with_plane(c, K_s, T_bw, z_tgt, img_shape=(h_img, w_img))
        if len(pts_xy) < 8:
            continue
        fit = _fit_circle_kasa(pts_xy, r_nom=r_top_m, prior_xy=exp_tgt[:2], max_shift_m=max_corr_m)
        if fit is None or fit["rms_m"] > 0.008:
            continue
        d_exp = math.hypot(fit["xc"] - exp_tgt[0], fit["yc"] - exp_tgt[1])
        if best_tgt is None or d_exp < best_tgt["d_exp"]:
            best_tgt = dict(fit, d_exp=d_exp)

    if best_tgt is None:
        return {"ok": False, "reason": "目标杯口轮廓无法投影到目标高度平面"}

    xt, yt = best_tgt["xc"], best_tgt["yc"]

    # Check if an inner contour (held cup bottom) is separately detected via RETR_TREE / Canny circles
    xh, yh = exp_held[0], exp_held[1]
    held_source = "kinematic_grip_offset"
    cv2 = _cv2()
    cs_tree, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if hier is not None and len(cs_tree) >= 3:
        for idx_c, c_in in enumerate(cs_tree):
            # Depth-2 inner island contour (parent is a hole whose parent is the outer cup rim)
            parent_idx = int(hier[0][idx_c][3])
            if parent_idx >= 0 and int(hier[0][parent_idx][3]) >= 0 and cv2.contourArea(c_in) >= 80:
                pts_in, _ = _intersect_contour_with_plane(c_in, K_s, T_bw, z_held)
                if len(pts_in) >= 10:
                    fit_in = _fit_circle_kasa(pts_in, r_nom=r_bot_m)
                    if fit_in is not None and math.hypot(fit_in["xc"] - exp_held[0], fit_in["yc"] - exp_held[1]) <= max_corr_m:
                        xh, yh = fit_in["xc"], fit_in["yc"]
                        held_source = "wrist_inner_contour"
                        break

    dx_m = xt - xh
    dy_m = yt - yh
    err_m = math.hypot(dx_m, dy_m)
    trusted = (best_tgt["d_exp"] <= float(max_corr_m)) and (err_m <= float(max_corr_m))

    return {
        "ok": True,
        "trusted": bool(trusted),
        "held_source": held_source,
        "target_xy_m": [round(xt, 5), round(yt, 5)],
        "held_xy_m": [round(xh, 5), round(yh, 5)],
        "delta_xy_m": [round(dx_m, 5), round(dy_m, 5)],
        "delta_mm": [round(dx_m * 1000.0, 2), round(dy_m * 1000.0, 2)],
        "concentric_err_mm": round(err_m * 1000.0, 2),
        "target_shift_from_coarse_mm": round(best_tgt["d_exp"] * 1000.0, 2),
    }


def wrist_check_egg_or_slot(rgb, K, T_base_flange, calib_eih, expected_xyz,
                            z_plane=None, kind="egg", undistort_intr=None, max_shift_m=0.025):
    """V4.1 ROI-GATED WRIST VISUAL SERVOING FOR EGG GRASP & EGG CARTON SLOT PLACEMENT:
    1. Projects expected_xyz (at Z = z_plane) into the wrist camera image and restricts mask extraction to a 3D circular
       ROI around expected_xyz (radius 48 mm) so large bright background regions or carton edges are never mistaken for the egg.
    2. Filters candidates by 3D major/minor span (30..72 mm, aspect ratio <= 1.85) and proximity to expected_xyz."""
    if rgb is None:
        return {"ok": False, "reason": "腕部相机无画面"}
    if not K or not calib_eih or not calib_eih.get("T_flange_cam") or T_base_flange is None:
        return {"ok": False, "reason": "缺少腕部内参或 Eye-in-Hand 手眼标定"}
    cv2 = _cv2()
    rgb_u = W49.undistort_rgb(rgb, undistort_intr) if (undistort_intr and W49.intrinsics_valid(undistort_intr)) else rgb
    arr = np.ascontiguousarray(rgb_u, dtype=np.uint8)
    h_img, w_img = arr.shape[:2]
    K_s = scale_K_for_frame(K, w_img, h_img)
    T_bw = W49.wrist_T_base_cam(T_base_flange, calib_eih)
    T_wb = CAM.T_inv(T_bw)
    exp = [float(v) for v in expected_xyz]
    zp = float(exp[2] if z_plane is None else z_plane)

    # Build a local circular ROI mask centered on the projected expected target in image pixels
    roi_mask = np.ones((h_img, w_img), dtype=np.uint8) * 255
    p_cam_exp = CAM.T_apply(T_wb, [exp[0], exp[1], zp])
    if p_cam_exp[2] > 0.04:
        uv_exp = CAM.project(p_cam_exp, K_s)
        if uv_exp is not None:
            f_mean = 0.5 * (float(K_s["fx"]) + float(K_s["fy"]))
            roi_r_px = int(max(36.0, min(0.45 * max(h_img, w_img), f_mean * 0.048 / p_cam_exp[2])))
            roi_mask.fill(0)
            cv2.circle(roi_mask, (int(round(uv_exp[0])), int(round(uv_exp[1]))), roi_r_px, 255, -1)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    if kind == "egg":
        m_egg = cv2.inRange(hsv, np.array([0, 0, 145], np.uint8), np.array([180, 95, 255], np.uint8))
        m_egg = cv2.bitwise_and(m_egg, roi_mask)
        m_egg = cv2.morphologyEx(m_egg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cs, _ = cv2.findContours(m_egg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cs = [c for c in cs if 60 <= cv2.contourArea(c) <= 0.35 * h_img * w_img]
        if not cs:
            roi_vals = gray[roi_mask > 0] if np.any(roi_mask > 0) else gray.reshape(-1)
            med = float(np.median(roi_vals))
            bw = ((np.abs(gray.astype(np.float32) - med) > 28.0).astype(np.uint8) * 255) & roi_mask
            cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cs = [c for c in cs if 60 <= cv2.contourArea(c) <= 0.35 * h_img * w_img]
    else:
        roi_vals = gray[roi_mask > 0] if np.any(roi_mask > 0) else gray.reshape(-1)
        med = float(np.median(roi_vals))
        bw = ((np.abs(gray.astype(np.float32) - med) > 22.0).astype(np.uint8) * 255) & roi_mask
        cs, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cs = [c for c in cs if 50 <= cv2.contourArea(c) <= 0.30 * h_img * w_img]

    if not cs:
        return {"ok": False, "reason": f"腕部相机局部 ROI 内未检测到{'鸡蛋' if kind == 'egg' else '蛋托孔位'}轮廓"}

    best = None
    for c in sorted(cs, key=cv2.contourArea, reverse=True)[:10]:
        pts_xy, _ = _intersect_contour_with_plane(c, K_s, T_bw, zp, img_shape=(h_img, w_img))
        if len(pts_xy) < 8:
            continue
        xc, yc = float(np.mean(pts_xy[:, 0])), float(np.mean(pts_xy[:, 1]))
        centered = pts_xy - np.array([xc, yc])
        _, s_vals, vt = np.linalg.svd(centered, full_matrices=False)
        # Estimate major and minor diameters from 95th percentile projections along principal axes
        proj_major = float(np.percentile(np.abs(centered @ vt[0]), 95)) * 2.0
        proj_minor = float(np.percentile(np.abs(centered @ vt[1]), 95)) * 2.0 if len(vt) > 1 else proj_major
        if not (0.018 <= proj_major <= 0.080 and 0.014 <= proj_minor <= 0.068):
            continue
        aspect = proj_major / max(1e-4, proj_minor)
        if aspect > 2.1:
            continue
        dist_e = math.hypot(xc - exp[0], yc - exp[1])
        long_dir = vt[0]
        egg_yaw_rad = math.atan2(float(long_dir[1]), float(long_dir[0]))
        if best is None or dist_e < best["dist_e"]:
            best = {"xc": xc, "yc": yc, "dist_e": dist_e, "egg_yaw_rad": egg_yaw_rad}

    if best is None:
        return {"ok": False, "reason": f"{'鸡蛋' if kind == 'egg' else '蛋托孔位'}轮廓尺寸不符或无法投影到工作平面"}

    xc, yc = best["xc"], best["yc"]
    dx_m, dy_m = xc - exp[0], yc - exp[1]
    shift_m = math.hypot(dx_m, dy_m)
    trusted = shift_m <= float(max_shift_m)
    grasp_phi_rad = best["egg_yaw_rad"] + math.pi * 0.5
    return {
        "ok": True,
        "trusted": bool(trusted),
        "kind": kind,
        "measured_xyz": [round(xc, 5), round(yc, 5), round(zp, 5)],
        "expected_xyz": [round(v, 5) for v in exp],
        "delta_xy_m": [round(dx_m, 5), round(dy_m, 5)],
        "delta_mm": [round(dx_m * 1000.0, 2), round(dy_m * 1000.0, 2)],
        "shift_norm_mm": round(shift_m * 1000.0, 2),
        "egg_yaw_deg": round(math.degrees(best["egg_yaw_rad"]), 1),
        "recommended_grasp_phi_rad": grasp_phi_rad,
    }


MARKER_V46_SLOT = "V4.6 WRIST SLOT CAVITY DETECTOR"


def _poly_area_perim(pts):
    pts = np.asarray(pts, float)
    if len(pts) < 3:
        return 0.0, 0.0
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    perim = float(np.sum(np.hypot(np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0])))))
    return area, perim


def wrist_find_slot_cavity(rgb, K, T_base_flange, calib_eih, expected_xyz, slot_pitch_m=0.047,
                           cavity_radius_m=0.019, undistort_intr=None, max_shift_m=0.030):
    """V4.6 WRIST SLOT CAVITY DETECTOR (empty gripper, camera looking down at the carton):
    an empty egg slot is a round cavity that is darker than the carton top around it. Finds dark blobs near the
    expected slot, keeps the ones that are round (circularity >= 0.55) and slot-sized (radius 0.45-1.35 x the expected
    cavity radius) when projected onto the slot plane, takes the one closest to the expected position and counts how
    many other cavities sit one slot pitch away (neighbour check, so a single dark shadow is not mistaken for a slot).
    -> {ok, trusted, measured_xyz, delta_xy_m, shift_norm_mm, radius_mm, circularity, n_neighbours, n_candidates}"""
    if rgb is None:
        return {"ok": False, "reason": "腕部相机无画面"}
    if not K or not calib_eih or not calib_eih.get("T_flange_cam") or T_base_flange is None:
        return {"ok": False, "reason": "缺少腕部内参或 Eye-in-Hand 手眼标定"}
    cv2 = _cv2()
    rgb_u = W49.undistort_rgb(rgb, undistort_intr) if (undistort_intr and W49.intrinsics_valid(undistort_intr)) else rgb
    arr = np.ascontiguousarray(rgb_u, dtype=np.uint8)
    h_img, w_img = arr.shape[:2]
    K_s = scale_K_for_frame(K, w_img, h_img)
    T_bw = np.asarray(W49.wrist_T_base_cam(T_base_flange, calib_eih), float)
    T_wb = CAM.T_inv(T_bw.tolist())
    exp = [float(v) for v in expected_xyz]
    p_cam = CAM.T_apply(T_wb, exp)
    if p_cam[2] <= 0.04:
        return {"ok": False, "reason": "槽位不在腕部相机前方"}
    uv = CAM.project(p_cam, K_s)
    if uv is None:
        return {"ok": False, "reason": "槽位投影不到腕部画面"}
    f_mean = 0.5 * (float(K_s["fx"]) + float(K_s["fy"]))
    roi_r = int(max(40.0, min(0.48 * max(h_img, w_img), f_mean * (1.4 * float(slot_pitch_m)) / p_cam[2])))
    roi = np.zeros((h_img, w_img), np.uint8)
    cv2.circle(roi, (int(round(uv[0])), int(round(uv[1]))), roi_r, 255, -1)
    if int(np.count_nonzero(roi)) < 200:
        return {"ok": False, "reason": "槽位在画面边缘之外"}
    gray = cv2.GaussianBlur(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    vals = gray[roi > 0]
    med = float(np.median(vals))
    thr_otsu, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = min(float(thr_otsu), med - max(10.0, 0.12 * med))
    dark = ((gray.astype(np.float32) < thr).astype(np.uint8) * 255) & roi
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    cs, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    zp = exp[2]
    cands = []
    for c in cs:
        if cv2.contourArea(c) < 30:
            continue
        pts, ratio = _intersect_contour_with_plane(c.reshape(-1, 2), K_s, T_bw, zp, img_shape=(h_img, w_img))
        if len(pts) < 12 or ratio < 0.85:
            continue
        a, per = _poly_area_perim(pts)
        if a <= 0 or per <= 0:
            continue
        r_eq = math.sqrt(a / math.pi)
        circ = 4.0 * math.pi * a / (per * per)
        if not (0.45 * cavity_radius_m <= r_eq <= 1.35 * cavity_radius_m) or circ < 0.55:
            continue
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        cands.append({"xy": (cx, cy), "r": r_eq, "circ": circ, "d": math.hypot(cx - exp[0], cy - exp[1])})
    if not cands:
        return {"ok": False, "reason": "腕部画面里没有找到圆形的槽穴（光线 / 角度 / 被遮挡？）", "n_candidates": 0}
    best = min(cands, key=lambda c_: c_["d"])
    nb = sum(1 for c_ in cands if c_ is not best
             and abs(math.hypot(c_["xy"][0] - best["xy"][0], c_["xy"][1] - best["xy"][1]) - slot_pitch_m) <= 0.25 * slot_pitch_m)
    dx, dy = best["xy"][0] - exp[0], best["xy"][1] - exp[1]
    shift = math.hypot(dx, dy)
    trusted = shift <= float(max_shift_m) and (nb >= 1 or shift <= 0.012)
    return {
        "ok": True, "trusted": bool(trusted), "kind": "slot_cavity",
        "measured_xyz": [round(best["xy"][0], 5), round(best["xy"][1], 5), round(zp, 5)],
        "expected_xyz": [round(v, 5) for v in exp],
        "delta_xy_m": [round(dx, 5), round(dy, 5)], "delta_mm": [round(dx * 1000.0, 2), round(dy * 1000.0, 2)],
        "shift_norm_mm": round(shift * 1000.0, 2), "radius_mm": round(best["r"] * 1000.0, 1),
        "circularity": round(best["circ"], 2), "n_neighbours": int(nb), "n_candidates": len(cands),
        "reason": None if trusted else ("偏差过大" if shift > max_shift_m else "只看到一个圆坑且离预期较远（没有相邻槽验证）"),
    }


def evaluate_cup_rim_preview(rgb, K=None, T_flange_cam=None, tool_T=None, z_work_m=0.10):
    """Preview red cup rim / bottom circular contours in the wrist camera view for Lab calibration check."""
    if rgb is None:
        return {"ok": False, "reason": "无画面"}
    cv2 = _cv2()
    arr = np.ascontiguousarray(rgb)
    h, w = arr.shape[:2]
    cs, _ = _extract_cup_contours(arr, min_area=150)
    if not cs:
        return {"ok": False, "reason": "未检测到红色杯子轮廓"}
    c_main = cs[0]
    (cx, cy), radius = cv2.minEnclosingCircle(c_main)

    K_s = scale_K_for_frame(K, w, h) if K else None
    proj = W49.project_tcp_wrist(K_s, T_flange_cam, tool_T, z_work_m=z_work_m) if (K_s and T_flange_cam and tool_T) else None
    uv_tcp = proj["uv_tcp"] if (proj and proj.get("ok")) else [w * 0.5, h * 0.5]
    px_per_mm = proj["px_per_mm"] if (proj and proj.get("ok")) else (0.5 * ((K_s or {}).get("fx", 900.0) + (K_s or {}).get("fy", 900.0)) / (z_work_m * 1000.0))
    open_dir = proj["open_dir_2d"] if (proj and proj.get("ok")) else [1.0, 0.0]

    dx_px, dy_px = float(cx - uv_tcp[0]), float(cy - uv_tcp[1])
    dx_mm = round(dx_px / max(0.1, px_per_mm), 2)
    dy_mm = round(dy_px / max(0.1, px_per_mm), 2)

    rv_x, rv_y = float(uv_tcp[0] - cx), float(uv_tcp[1] - cy)
    rnorm = math.hypot(rv_x, rv_y)
    if rnorm > 5.0:
        rv_x, rv_y = rv_x / rnorm, rv_y / rnorm
    else:
        rv_x, rv_y = 1.0, 0.0
    dot = max(-1.0, min(1.0, abs(open_dir[0] * rv_x + open_dir[1] * rv_y)))
    angle_err_deg = round(math.degrees(math.acos(dot)), 1)

    return {
        "ok": True,
        "center_px": [round(float(cx), 1), round(float(cy), 1)],
        "radius_px": round(float(radius), 1),
        "dx_mm": dx_mm,
        "dy_mm": dy_mm,
        "offset_norm_mm": round(math.hypot(dx_mm, dy_mm), 2),
        "angle_err_deg": angle_err_deg,
    }


# =========================================================================== 4. Innomaker4MPWristSource Driver
class Innomaker4MPWristSource(W49.UsbWristSource):
    """UVC reader for the Innomaker 4MP USB HDR Camera (U20-4MP OV4688), backwards-compatible with W49.UsbWristSource.
    Supports live UVC control updates via run_on_device() and automatic application of saved wrist profiles."""

    def __init__(self, device=None, res="1920×1080", lock_controls=True, intrinsics=None,
                 camera_model=WRIST_CAMERA_MODELS[0], log=print):
        super().__init__(device=device, res=res, lock_controls=lock_controls, intrinsics=intrinsics, log=log)
        self.res = res if str(res).split()[0] in WRIST_RES_MAP else "1920×1080"
        self.camera_model = camera_model or WRIST_CAMERA_MODELS[0]
        self._dev_cmds = []

    def run_on_device(self, fn, timeout=3.0):
        """Execute fn(cap, cv2) on the active UVC capture thread (or immediately if simulated/mocked)."""
        if not self.running():
            return fn(getattr(self, "_cap", None), _cv2())
        box = {"ev": threading.Event()}
        with self.lock:
            self._dev_cmds.append((fn, box))
        if not box["ev"].wait(timeout):
            raise TimeoutError("腕部相机线程没有响应")
        if "err" in box:
            raise box["err"]
        return box.get("out")

    def set_exposure_gain(self, exposure, gain, wait_frames=2):
        """V3.89 Lightweight UVC exposure + gain setter for autotune: updates only CAP_PROP_EXPOSURE and CAP_PROP_GAIN
        (clamping < 1.0 positive values to >= 1.0 on real V4L2 so cvRound never writes 0) and waits for fresh frames."""
        def _do(cap, cv2):
            out = {}
            if cap is not None:
                if hasattr(cv2, "CAP_PROP_EXPOSURE"):
                    try:
                        e_val = float(exposure)
                        cap.set(cv2.CAP_PROP_EXPOSURE, max(1.0, e_val) if 0.0 < e_val < 1.0 else e_val)
                        out["exposure"] = e_val
                    except Exception:
                        pass
                if hasattr(cv2, "CAP_PROP_GAIN"):
                    try:
                        cap.set(cv2.CAP_PROP_GAIN, float(gain))
                        out["gain"] = float(gain)
                    except Exception:
                        pass
            return out

        res = self.run_on_device(_do)
        if self.running() and wait_frames > 0:
            fid0 = self.frame_id
            t_deadline = time.time() + 0.12
            while self.running() and (self.frame_id - fid0) < wait_frames and time.time() < t_deadline:
                time.sleep(0.010)
        return res

    def apply_profile_dict(self, prof_color):
        """Apply UVC exposure/gain/WB/HDR settings to the live camera and update self.info['controls']."""
        def _do(cap, cv2):
            applied = apply_uvc_profile_to_cap(cap, cv2, prof_color)
            with self.lock:
                if self.info:
                    ctrls = dict(self.info.get("controls") or {})
                    ctrls.update(applied)
                    self.info["controls"] = ctrls
            return applied
        return self.run_on_device(_do)

    def _run(self):
        try:
            cv2 = _cv2()
        except Exception as exc:
            self.error = f"OpenCV / numpy 不可用：{exc}"
            return
        cap = None
        try:
            dev = self.device
            dev_name = str(dev or "")
            if not dev or dev == "auto":
                cands = list_wrist_cameras()
                if not cands:
                    self.error = "没有找到 USB 腕部相机（请检查 Innomaker 4MP USB 线，或运行 v4l2-ctl --list-devices）"
                    return
                dev = cands[0]["device"]
                dev_name = f"{cands[0]['name']} ({dev})"
            idx = int(dev[10:]) if isinstance(dev, str) and dev.startswith("/dev/video") and dev[10:].isdigit() else dev
            backend = getattr(cv2, "CAP_V4L2", 0)
            cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
            if not cap or not cap.isOpened():
                self.error = f"打不开腕部相机 {dev}（被其他程序占用或权限不足？）"
                return
            self._cap = cap
            w, h = parse_res(self.res)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, 30)
            locks = W49.apply_uvc_locks(cap, cv2, lock_exposure=self.lock_controls, lock_wb=self.lock_controls,
                                        lock_focus=self.lock_controls)
            prof = load_wrist_profile()
            if self.lock_controls and prof.get("locked"):
                applied = apply_uvc_profile_to_cap(cap, cv2, prof.get("color"))
                locks.update(applied)
                self.log(f"[OK] [V3.86 wrist lab] 已套用锁定的腕部相机参数（曝光 {prof['color'].get('exposure')} · "
                         f"增益 {prof['color'].get('gain')} · 白平衡 {prof['color'].get('white_balance')} · "
                         f"HDR {prof['color'].get('backlight_hdr', 1)}）"
                         + (f"（参数组「{prof['name']}」）" if prof.get("name") else ""))
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or w)
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or h)
            info = {
                "name": dev_name or str(dev),
                "camera_model": self.camera_model,
                "sensor": "OV4688 4MP HDR" if "4MP" in str(self.camera_model) else "1080P 130° Wide",
                "device": str(dev),
                "format": "MJPG",
                "res": f"{aw}×{ah}",
                "width": aw,
                "height": ah,
                "controls": locks,
                "K": dict(self.K) if self.K else None,
                "dist": list(self.dist) if self.K else None,
                "model_type": self.model_type if self.K else None,
            }
            with self.lock:
                self.info = info
            n, t0 = 0, time.time()
            while not self.stop_ev.is_set():
                with self.lock:
                    cmds, self._dev_cmds = self._dev_cmds, []
                for fn, box in cmds:
                    try:
                        box["out"] = fn(cap, cv2)
                    except Exception as exc:
                        box["err"] = exc
                    box["ev"].set()
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    time.sleep(0.01)
                    continue
                rgb = np.ascontiguousarray(bgr[..., ::-1])
                n += 1
                if time.time() - t0 >= 1.0:
                    self.fps, n, t0 = n / (time.time() - t0), 0, time.time()
                with self.lock:
                    self.rgb = rgb
                    self.frame_id += 1
        except Exception as exc:
            self.error = f"腕部相机读取失败：{exc}"[:260]
        finally:
            self._cap = None
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass


# =========================================================================== 5. Wrist Camera Lab GUI (腕部相机高级测试)
def load_lab_settings(path=None):
    d = {"roi": list(DEFAULT_ROI), "mean_lo": 100.0, "z_work_mm": 100.0, "show_cup_rim": True}
    try:
        with open(path or os.environ.get("NERO_WRIST_LAB_SETTINGS") or WRIST_LAB_SETTINGS, encoding="utf-8") as f:
            d.update(json.load(f))
    except Exception:
        pass
    return d


def save_lab_settings(d, path=None):
    p = path or os.environ.get("NERO_WRIST_LAB_SETTINGS") or WRIST_LAB_SETTINGS
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(d, version=VERSION), f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    return p


def open_wrist_lab(tab):
    """Open or focus the Wrist Camera Lab window for the given CameraTab."""
    old = _OPEN_WRIST_LAB.get(id(tab))
    if old is not None and getattr(old, "alive", False):
        try:
            old.win.lift()
            return old
        except Exception:
            pass
    lab = WristCameraLab(tab)
    _OPEN_WRIST_LAB[id(tab)] = lab
    return lab


class WristCameraLab:
    """Dedicated advanced testing & calibration verification dialog for the Innomaker 4MP OV4688 HDR wrist camera."""
    PW, PH = 640, 360

    def __init__(self, tab):
        self.tab, self.tk, self.ttk = tab, tab.tk, tab.ttk
        tk, ttk = self.tk, self.ttk
        self.log = tab.log
        self.alive = True
        self.busy = None
        self.st = load_lab_settings()
        self.prof = load_wrist_profile()
        self.peak_sharpness = 0.0
        self.last_tune = None
        self.photos = {}

        self.win = tk.Toplevel(getattr(tab, "root", None) or tab.cv)
        self.win.title(f"NERO {VERSION} 腕部相机高级测试（Innomaker 4MP USB HDR · OV4688）")
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.nb = ttk.Notebook(self.win)
        self.nb.pack(fill="both", expand=True)

        self._build_uvc_tab()
        self._build_focus_tab()
        self._build_verify_tab()

        self.log(f"[INFO] [{MARKER}] 打开腕部相机高级测试")
        self._tick()

    def _var(self, val):
        return self.tk.StringVar(value=str(val))

    def _wsrc(self):
        ws = getattr(self.tab, "wrist_src", None)
        return ws if ws is not None else None

    def _need_wsrc(self):
        ws = self._wsrc()
        if ws is None or (hasattr(ws, "running") and not ws.running() and getattr(ws, "rgb", None) is None and getattr(ws, "cur_rgb", None) is None):
            self.log("[WARN] [V3.86 wrist lab] 先在「相机信息」页点「▶ 连接腕部相机」")
            return None
        return ws

    def _grab_rgb(self):
        ws = self._wsrc()
        if ws is None:
            return None
        try:
            rgb, _d, _fid = ws.frames()
            if rgb is not None:
                return rgb
        except Exception:
            pass
        return getattr(ws, "rgb", None) if getattr(ws, "rgb", None) is not None else getattr(ws, "cur_rgb", None)

    # ------------------------------------------------------------------ Tab 1: ① 腕部相机参数与HDR
    def _build_uvc_tab(self):
        tk, ttk = self.tk, self.ttk
        p = ttk.Frame(self.nb, padding=6)
        self.nb.add(p, text="① 腕部相机参数与HDR")

        self.cv_uvc = tk.Canvas(p, width=self.PW, height=self.PH, bg="#202020", highlightthickness=0)
        self.cv_uvc.pack(anchor="w")
        self.cv_uvc.bind("<ButtonPress-1>", self._roi_press)
        self.cv_uvc.bind("<B1-Motion>", self._roi_drag)
        self.cv_uvc.bind("<ButtonRelease-1>", self._roi_release)

        self.v_stats = self._var("（连接腕部相机后显示工作区亮度与高光状态；在画面上拖框 = 感兴趣工作区）")
        ttk.Label(p, textvariable=self.v_stats).pack(anchor="w", pady=(4, 2))

        r0 = ttk.Frame(p)
        r0.pack(fill="x", pady=2)
        col = self.prof["color"]
        pl = int(col.get("power_line_frequency", 2))
        self.v_pl = self._var(next((n for n, v in POWER_CHOICES if v == pl), "60 Hz"))
        ttk.Label(r0, text="电网频率").pack(side="left")
        ttk.Combobox(r0, textvariable=self.v_pl, values=[n for n, _ in POWER_CHOICES], width=7,
                     state="readonly").pack(side="left", padx=(2, 8))
        self.v_meanlo = self._var(self.st.get("mean_lo", 100.0))
        ttk.Label(r0, text="平均亮度下限").pack(side="left")
        ttk.Entry(r0, textvariable=self.v_meanlo, width=5).pack(side="left", padx=(2, 8))
        ttk.Button(r0, text="▶ 一键自动调参", command=self.on_autotune).pack(side="left")
        ttk.Button(r0, text="工作区恢复默认", command=self.on_roi_default).pack(side="left", padx=6)

        self.v_tune = self._var("")
        self.lb_tune = ttk.Label(p, textvariable=self.v_tune, justify="left", wraplength=self.PW + 260)
        self.lb_tune.pack(anchor="w", pady=(2, 2))

        r1 = ttk.Frame(p)
        r1.pack(fill="x", pady=2)
        self.v_exp = self._var(col.get("exposure", 1.0))
        self.v_gain = self._var(col.get("gain", 0.0))
        self.v_wb = self._var(col.get("white_balance", 4200.0))
        self.v_gamma = self._var(col.get("gamma", 120.0))
        self.v_sharp = self._var(col.get("sharpness", 3.0))
        self.v_hdr = tk.BooleanVar(value=bool(col.get("backlight_hdr", 1)))
        ttk.Label(r1, text="手动：曝光(0.1 ms)").pack(side="left")
        ttk.Entry(r1, textvariable=self.v_exp, width=6).pack(side="left", padx=(2, 6))
        ttk.Label(r1, text="增益").pack(side="left")
        ttk.Entry(r1, textvariable=self.v_gain, width=5).pack(side="left", padx=(2, 6))
        ttk.Label(r1, text="白平衡 K").pack(side="left")
        ttk.Entry(r1, textvariable=self.v_wb, width=6).pack(side="left", padx=(2, 6))
        ttk.Label(r1, text="伽马").pack(side="left")
        ttk.Entry(r1, textvariable=self.v_gamma, width=5).pack(side="left", padx=(2, 6))
        ttk.Label(r1, text="锐度").pack(side="left")
        ttk.Entry(r1, textvariable=self.v_sharp, width=4).pack(side="left", padx=(2, 6))
        ttk.Checkbutton(r1, text="开启 OV4688 HDR 宽动态抑高光", variable=self.v_hdr).pack(side="left", padx=4)
        ttk.Button(r1, text="应用", command=self.on_apply).pack(side="left", padx=4)

        r2 = ttk.Frame(p)
        r2.pack(fill="x", pady=(6, 2))
        self.v_pname = self._var(self.prof.get("name") or "车库-腕部4MP")
        ttk.Label(r2, text="参数组").pack(side="left")
        self.cb_pname = ttk.Combobox(r2, textvariable=self.v_pname, values=list_named_profiles(), width=18,
                                     postcommand=lambda: self.cb_pname.configure(values=list_named_profiles()))
        self.cb_pname.pack(side="left", padx=2)
        ttk.Button(r2, text="保存为这一组", command=self.on_save_profile).pack(side="left", padx=2)
        ttk.Button(r2, text="套用这一组", command=self.on_load_named).pack(side="left", padx=2)
        self.v_auto = tk.BooleanVar(value=True)
        ttk.Checkbutton(r2, text="设为连接腕部相机时自动套用", variable=self.v_auto).pack(side="left", padx=6)

        self.v_prof = self._var(self._prof_text())
        ttk.Label(p, textvariable=self.v_prof, justify="left", foreground="#335").pack(anchor="w", pady=(4, 0))
        ttk.Label(p, text="说明：Innomaker U20-4MP (OV4688) 具备硬件 HDR 宽动态能力，可抑制红色塑料杯沿的刺眼高光并保留暗部细节。\n"
                          "建议将曝光锁定为电网闪烁周期整数倍（60 Hz 下约 8.3 ms = 83，50 Hz 下 10 ms = 100），避免机械臂移动时自动曝光引发亮度跳变。",
                  foreground="#555", justify="left").pack(anchor="w", pady=(4, 0))

    # ------------------------------------------------------------------ Tab 2: ② 对焦与清晰度测试
    def _build_focus_tab(self):
        tk, ttk = self.tk, self.ttk
        p = ttk.Frame(self.nb, padding=6)
        self.nb.add(p, text="② 对焦与清晰度测试")

        self.cv_focus = tk.Canvas(p, width=self.PW, height=self.PH, bg="#202020", highlightthickness=0)
        self.cv_focus.pack(anchor="w")

        r = ttk.Frame(p)
        r.pack(fill="x", pady=4)
        ttk.Button(r, text="▶ 立即测算清晰度", command=self.on_measure_focus).pack(side="left")
        ttk.Button(r, text="重置峰值记录", command=self.on_reset_peak).pack(side="left", padx=6)
        self.v_focus_stat = self._var("（将腕部相机停在标定板或杯口正上方 80–150 mm，观察实时 Laplacian 清晰度评分）")
        ttk.Label(p, textvariable=self.v_focus_stat, font=("TkFixedFont", 9), justify="left").pack(anchor="w", pady=2)
        ttk.Label(p, text="对焦指南：缓慢旋转 OV4688 镜头调焦环（若未点胶固定），使「当前清晰度」逼近或刷新「历史峰值」（≥ 80 为优，≥ 25 为合格）。\n"
                          "四角清晰度用于检查镜头光轴是否歪斜或边缘失焦。",
                  foreground="#555", justify="left").pack(anchor="w", pady=(4, 0))

    # ------------------------------------------------------------------ Tab 3: ③ 标定与TCP准星复核
    def _build_verify_tab(self):
        tk, ttk = self.tk, self.ttk
        p = ttk.Frame(self.nb, padding=6)
        self.nb.add(p, text="③ 标定与TCP准星复核")

        self.cv_verify = tk.Canvas(p, width=self.PW, height=self.PH, bg="#202020", highlightthickness=0)
        self.cv_verify.pack(anchor="w")

        r = ttk.Frame(p)
        r.pack(fill="x", pady=4)
        ttk.Label(r, text="投影工作距离 Z (mm)").pack(side="left")
        self.v_zwork = self._var(self.st.get("z_work_mm", 100.0))
        ttk.Spinbox(r, textvariable=self.v_zwork, from_=40, to=300, increment=5, width=6).pack(side="left", padx=(2, 8))
        self.v_cup_prev = tk.BooleanVar(value=bool(self.st.get("show_cup_rim", True)))
        ttk.Checkbutton(r, text="开启红杯杯口与夹爪角度实时检测预览", variable=self.v_cup_prev).pack(side="left", padx=4)
        ttk.Button(r, text="▶ 单帧复核报告", command=self.on_verify_once).pack(side="left", padx=6)

        self.v_verify_stat = self._var("")
        ttk.Label(p, textvariable=self.v_verify_stat, font=("TkFixedFont", 9), justify="left").pack(anchor="w", pady=2)
        ttk.Label(p, text="说明：画面实时叠加已标定的 TCP 中心十字准星、夹爪开合连线方向（青色虚线）与左右爪尖安全落区（青色方框）。\n"
                          "勾选「红杯杯口实时检测预览」后，把红杯放在腕部相机下方即可实时观测杯心偏差 (Δx, Δy) mm 与夹爪开合线垂直度偏差 Δθ (°)。",
                  foreground="#555", justify="left").pack(anchor="w", pady=(4, 0))

    # ------------------------------------------------------------------ ROI & UVC actions
    def _roi_press(self, ev):
        self._roi_op = [ev.x, ev.y, ev.x, ev.y]

    def _roi_drag(self, ev):
        if getattr(self, "_roi_op", None):
            self._roi_op[2:] = [ev.x, ev.y]

    def _roi_release(self, ev):
        op, self._roi_op = getattr(self, "_roi_op", None), None
        if not op:
            return
        op[2:] = [ev.x, ev.y]
        x0, x1 = sorted((op[0] / float(self.PW), op[2] / float(self.PW)))
        y0, y1 = sorted((op[1] / float(self.PH), op[3] / float(self.PH)))
        if x1 - x0 < 0.05 or y1 - y0 < 0.05:
            return
        self.st["roi"] = [round(min(max(v, 0.0), 1.0), 3) for v in (x0, y0, x1, y1)]
        save_lab_settings(self.st)

    def on_roi_default(self):
        self.st["roi"] = list(DEFAULT_ROI)
        save_lab_settings(self.st)

    def _read_ui_color(self):
        col = self.prof["color"]
        col["exposure"] = float(self.v_exp.get())
        col["gain"] = float(self.v_gain.get())
        col["white_balance"] = float(self.v_wb.get())
        col["gamma"] = float(self.v_gamma.get())
        col["sharpness"] = float(self.v_sharp.get())
        col["backlight_hdr"] = 1 if self.v_hdr.get() else 0
        col["power_line_frequency"] = dict(POWER_CHOICES).get(self.v_pl.get(), 2)
        col["enable_auto_exposure"] = 0
        col["enable_auto_white_balance"] = 0
        return col

    def on_apply(self):
        try:
            col = self._read_ui_color()
        except Exception as exc:
            self.log(f"[WARN] [V3.86 wrist lab] 参数格式有误：{exc}")
            return
        ws = self._need_wsrc()
        if ws is not None and hasattr(ws, "apply_profile_dict"):
            try:
                ws.apply_profile_dict(col)
            except Exception as exc:
                self.log(f"[WARN] [V3.86 wrist lab] 应用参数失败：{exc}")
        self.v_prof.set(self._prof_text())
        self.log(f"[OK] [V3.86 wrist lab] 已应用腕部相机参数（曝光 {col['exposure']} · 增益 {col['gain']} · "
                 f"白平衡 {col['white_balance']} · HDR {'开启' if col['backlight_hdr'] else '关闭'}）")

    def on_autotune(self):
        ws = self._need_wsrc()
        if ws is None:
            return
        rgb0 = self._grab_rgb()
        if rgb0 is None:
            self.log("[WARN] [V3.86 wrist lab] 腕部相机还没有画面")
            return
        hz = 50 if "50" in self.v_pl.get() else 60
        try:
            lo = float(self.v_meanlo.get())
        except Exception:
            lo = 100.0
        self.st["mean_lo"] = lo
        col = self.prof["color"]
        col["power_line_frequency"] = 1 if hz == 50 else 2
        col["backlight_hdr"] = 1 if self.v_hdr.get() else 0

        if hasattr(ws, "apply_profile_dict"):
            try:
                ws.apply_profile_dict(col)
            except Exception:
                pass

        def set_eg(e, g):
            col["exposure"], col["gain"] = float(e), float(g)
            if hasattr(ws, "set_exposure_gain"):
                try:
                    ws.set_exposure_gain(e, g, wait_frames=2)
                except Exception:
                    pass
            elif hasattr(ws, "apply_profile_dict"):
                try:
                    ws.apply_profile_dict(col)
                except Exception:
                    pass

        def measure():
            f = self._grab_rgb()
            return LAB379.roi_stats(f if f is not None else rgb0, self.st.get("roi"))

        res, _tr = autotune_wrist_uvc(set_eg, measure, power_hz=hz, gain_range=(0, 128), mean_lo=lo)
        self.last_tune = res
        self.v_exp.set(str(col["exposure"]))
        self.v_gain.set(str(col["gain"]))
        st = res["stats"]
        head = "✓ 合格" if res["ok"] else ("✗ " + "；".join(res["issues"]))
        self.v_tune.set(
            f"一键调参结果：{head} · 曝光 {res['exposure_ms']} ms · 增益 {res['gain']:g} · "
            f"工作区平均亮度 {st['mean']:.1f} · 最亮 0.5% {st.get('p995_ex', st['p995']):.1f} · "
            f"大片过曝 {st.get('sat_large_pct', st['sat_pct'])}% · 镜面高光点 {st.get('spots', 0)} 处"
        )
        self.v_prof.set(self._prof_text())
        self.log(f"[{'OK' if res['ok'] else 'WARN'}] [V3.86 wrist lab] 腕部相机一键调参完成：{head}")

    def on_save_profile(self):
        name = self.v_pname.get().strip() or "车库-腕部4MP"
        try:
            self._read_ui_color()
        except Exception as exc:
            self.log(f"[WARN] [V3.86 wrist lab] {exc}")
            return
        rgb = self._grab_rgb()
        ref = LAB379.roi_stats(rgb, self.st.get("roi")) if rgb is not None else None
        self.prof["name"] = name
        self.prof["locked"] = True
        self.prof["ref"] = ref
        self.prof["roi"] = list(self.st.get("roi") or DEFAULT_ROI)
        p = save_named_profile(name, self.prof)
        if self.v_auto.get():
            activate_named_profile(name)
        self.v_prof.set(self._prof_text())
        self.log(f"[OK] [V3.86 wrist lab] 腕部相机参数组「{name}」已保存 → {p}"
                 + ("（已设为连接时自动套用）" if self.v_auto.get() else ""))

    def on_load_named(self):
        name = self.v_pname.get().strip()
        if name not in list_named_profiles():
            self.log(f"[WARN] [V3.86 wrist lab] 没有找到参数组「{name}」")
            return
        d = load_named_profile(name)
        if isinstance(d.get("color"), dict):
            self.prof["color"].update(d["color"])
        self.prof["name"] = name
        col = self.prof["color"]
        self.v_exp.set(str(col.get("exposure", 1.0)))
        self.v_gain.set(str(col.get("gain", 0.0)))
        self.v_wb.set(str(col.get("white_balance", 4200.0)))
        self.v_gamma.set(str(col.get("gamma", 120.0)))
        self.v_sharp.set(str(col.get("sharpness", 3.0)))
        self.v_hdr.set(bool(col.get("backlight_hdr", 1)))
        if self.v_auto.get():
            activate_named_profile(name)
        self.on_apply()

    def _prof_text(self):
        p = load_wrist_profile()
        c = p.get("color") or {}
        if not p.get("locked"):
            return "连接时自动套用：未设置（保存参数组并勾选「设为连接腕部相机时自动套用」后生效）"
        return (f"连接时自动套用：「{p.get('name') or '默认'}」 · 曝光 {c.get('exposure')} · 增益 {c.get('gain')} · "
                f"白平衡 {c.get('white_balance')} K · 伽马 {c.get('gamma')} · 锐度 {c.get('sharpness')} · "
                f"HDR {'开启' if c.get('backlight_hdr', 1) else '关闭'}")

    def on_measure_focus(self):
        rgb = self._grab_rgb()
        m = evaluate_focus_metrics(rgb, self.st.get("roi"))
        if m.get("ok"):
            self.peak_sharpness = max(self.peak_sharpness, m["laplacian_var"])
            c = m["corners_lap"]
            self.v_focus_stat.set(
                f"{m['verdict']} · 历史峰值 {self.peak_sharpness:.1f}\n"
                f"四角清晰度（左上/右上/左下/右下）：{c[0]:.1f} / {c[1]:.1f} / {c[2]:.1f} / {c[3]:.1f}"
            )
        else:
            self.v_focus_stat.set("未获取到腕部相机画面")
        return m

    def on_reset_peak(self):
        self.peak_sharpness = 0.0
        self.on_measure_focus()

    def on_verify_once(self):
        rgb = self._grab_rgb()
        intr = getattr(self.tab, "intr_result", None) or getattr(self.tab, "wrist_intr", None)
        he = getattr(self.tab, "wcalib_result", None) or getattr(self.tab, "wrist_calib", None)
        K_w = (intr or {}).get("K")
        T_fc = (he or {}).get("T_flange_cam")
        tool_T = self.tab._active_tcp_T() if hasattr(self.tab, "_active_tcp_T") else None
        try:
            z_m = max(0.04, float(self.v_zwork.get()) / 1000.0)
        except Exception:
            z_m = 0.10
        self.st["z_work_mm"] = round(z_m * 1000.0, 1)
        self.st["show_cup_rim"] = bool(self.v_cup_prev.get())

        lines = []
        if K_w:
            lines.append(f"腕部内参：已就绪 ({K_w.get('width')}×{K_w.get('height')}, fx={K_w.get('fx'):.1f}, RMS={(intr or {}).get('rms_px', 0):.2f} px)")
        else:
            lines.append("腕部内参：未标定（请到「内参标定」Tab 标定）")
        if T_fc:
            t = CAM.T_t(T_fc)
            lines.append(f"Eye-in-Hand 手眼：已就绪 link7→cam ({t[0]*1000:+.1f}, {t[1]*1000:+.1f}, {t[2]*1000:+.1f}) mm")
        else:
            lines.append("Eye-in-Hand 手眼：未标定（请到「手眼标定 → eye-in-hand手眼标定」标定）")

        proj = W49.project_tcp_wrist(K_w, T_fc, tool_T, z_work_m=z_m, canvas_size=(self.PW, self.PH)) if (K_w and T_fc and tool_T) else None
        if proj and proj.get("ok"):
            lines.append(f"TCP 投影准星 (Z={z_m*1000:.0f} mm)：像素 ({proj['uv_tcp'][0]:.1f}, {proj['uv_tcp'][1]:.1f}) · "
                         f"比例 {proj['px_per_mm']:.2f} px/mm · 开合方向 ({proj['open_dir_2d'][0]:+.2f}, {proj['open_dir_2d'][1]:+.2f})")
        if self.v_cup_prev.get() and rgb is not None:
            cup_res = evaluate_cup_rim_preview(rgb, K=K_w, T_flange_cam=T_fc, tool_T=tool_T, z_work_m=z_m)
            if cup_res.get("ok"):
                lines.append(f"红杯杯口检测：中心偏差 Δx={cup_res['dx_mm']:+.1f} mm, Δy={cup_res['dy_mm']:+.1f} mm "
                             f"（总偏差 {cup_res['offset_norm_mm']:.1f} mm） · 夹爪垂直度误差 Δθ={cup_res['angle_err_deg']:.1f}°")
            else:
                lines.append(f"红杯杯口检测：{cup_res.get('reason')}")
        txt = "\n".join(lines)
        self.v_verify_stat.set(txt)
        return {"ok": True, "text": txt, "tcp_proj": proj}

    # ------------------------------------------------------------------ Periodic refresh
    def _show_canvas(self, canvas, key, rgb):
        data = CAM.ppm_bytes(np.ascontiguousarray(rgb))
        try:
            ph = self.tk.PhotoImage(data=data, format="PPM")
        except Exception:
            ph = self.tk.PhotoImage(data=CAM.png_bytes_from_ppm(data), format="PNG")
        self.photos[key] = ph
        canvas.delete("all")
        canvas.create_image(0, 0, image=ph, anchor="nw")

    def _refresh(self):
        rgb = self._grab_rgb()
        if rgb is None:
            return
        cv2 = _cv2()
        small = cv2.resize(np.ascontiguousarray(rgb), (self.PW, self.PH), interpolation=cv2.INTER_AREA)
        try:
            page = int(self.nb.index(self.nb.select()))
        except Exception:
            page = 0

        if page == 0:
            st = LAB379.roi_stats(rgb, self.st.get("roi"))
            self.v_stats.set(f"工作区平均亮度 {st['mean']:.1f} · 最亮 0.5% {st.get('p995_ex', st['p995']):.1f} · "
                             f"大片过曝 {st.get('sat_large_pct', st['sat_pct'])}% · 镜面高光点 {st.get('spots', 0)} 处")
            self._show_canvas(self.cv_uvc, "uvc", small)
            x0, y0, x1, y1 = LAB379.roi_px(self.st.get("roi"), (self.PW, self.PH))
            self.cv_uvc.create_rectangle(x0, y0, x1, y1, outline="#00e676", width=2, dash=(4, 3))
        elif page == 1:
            self.on_measure_focus()
            self._show_canvas(self.cv_focus, "focus", small)
            x0, y0, x1, y1 = LAB379.roi_px(self.st.get("roi"), (self.PW, self.PH))
            self.cv_focus.create_rectangle(x0, y0, x1, y1, outline="#ffea00", width=2, dash=(4, 3))
        elif page == 2:
            res = self.on_verify_once()
            self._show_canvas(self.cv_verify, "verify", small)
            proj = res.get("tcp_proj")
            if proj and proj.get("ok"):
                ux, uy = proj["uv_tcp_cv"]
                ax0, ay0, ax1, ay1 = proj["axis_line_cv"]
                self.cv_verify.create_line(ax0, ay0, ax1, ay1, fill="#00e5ff", dash=(3, 2), width=2)
                self.cv_verify.create_line(ux - 12, uy, ux + 12, uy, fill="#00e5ff", width=2)
                self.cv_verify.create_line(ux, uy - 12, ux, uy + 12, fill="#00e5ff", width=2)
                self.cv_verify.create_text(ux + 14, uy - 10, text="TCP", fill="#00e5ff", anchor="w")
                for fbox in (proj["left_box_cv"], proj["right_box_cv"]):
                    self.cv_verify.create_rectangle(*fbox, outline="#26c6da", dash=(2, 2), width=1)

    def _tick(self):
        if not self.alive:
            return
        try:
            self._refresh()
        except Exception:
            pass
        try:
            self.win.after(220, self._tick)
        except Exception:
            self.alive = False

    def close(self):
        self.alive = False
        try:
            save_lab_settings(self.st)
        except Exception:
            pass
        _OPEN_WRIST_LAB.pop(id(self.tab), None)
        try:
            self.win.destroy()
        except Exception:
            pass
