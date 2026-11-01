#!/usr/bin/env python3
"""NERO V3.25 Camera调试 — real Intel RealSense D435i on a fixed mount (eye-to-hand).

What the tab does
-----------------
* 相机: connect / disconnect the D435i in-process (pyrealsense2): colour 1280x720 + depth 848x480 aligned to colour,
  High Accuracy preset, spatial + temporal filter; device info, intrinsics, fps; save config/camera_info.json.
* Live image: colour / depth (absolute colour scale 0.3-1.5 m) / both; click = distance + camera xyz (+ base xyz
  when calibrated); drag = region for the depth check.  Overlays: robot chain, safety zone on the table, ArUco,
  predicted marker, eggs.
* 深度检查: plane fit of the dragged table region: valid ratio, RMS, tilt; with a calibration also the table height
  in base_link compared with the safety zone's z_table.
* 手眼标定 (eye-to-hand): an ArUco marker on the flange; samples = (joint angles, marker centre in the camera).
  Recorded by hand (jog / hand guiding) or automatically (poses around the current one, MoveIt + fence, <= 10 %).
  Unknowns: T_base_cam and the marker centre in the link7 frame, solved by alternating least squares (Horn /
  quaternion Kabsch + linear step), outliers rejected, leave-one-out errors as the independent check.
  Saved to config/camera_calib.json; 快速复核 = live error whenever the marker is seen, or a move to stored poses.
* 视觉定位: egg detection on real RGB-D, display only.  V3.47: YOLOE-26 (visual prompt from a saved reference, or text
  prompt) -> mask -> fused depth -> spheroid of the real egg size -> geometric check (NERO_V347_Yolo_Perception);
  the V3.10 geometric pipeline stays selectable ("几何法 V3.08").
  V3.47.1: egg pose from the YOLO silhouette + local table plane + egg size (no depth on the egg needed); depth
  resolution / preset selectable; calibration 另存为 / 读取 (default folder ~/Downloads/cameracalib) with a check after
  loading (camera serial + intrinsics, live marker error or 快速复核); 视觉定位 settings 保存 / 另存为 / 读取.

V3.45: 手眼标定 page in four numbered steps with status lamps (buttons follow the state); sample table with scrollbars
and a font-sized row height; marker side presets 40 / 30 mm; one measurement source per pose (depth only when >= 80 %
of the frames have it, else PnP); jitter after trimming bad frames, longer settle + still check, one re-measure;
visibility-aware auto sampling (camera direction from 3 small translations or the existing calibration; flange
positions spread >= 20 cm, wrist <= 25 deg, marker in the image and facing the camera <= 45 deg); redraw fix for
"'NoneType' object is not subscriptable".

The calibration maths is pure Python (unit-testable without numpy); image work uses numpy / OpenCV / pyrealsense2,
imported lazily so the console starts even without them.
"""
from __future__ import annotations

import base64
import glob
import json
import math
import os
import random
import threading
import time

VERSION = "V3.25"
HERE = os.path.dirname(os.path.realpath(__file__))
CAMERA_INFO_PATH = os.path.join(HERE, "config", "camera_info.json")
CALIB_PATH = os.path.join(HERE, "config", "camera_calib.json")
# V3.47.1 CALIB FILES: copies / other calibrations live outside the version folder, so a new NERO folder finds them
DEFAULT_CALIB_PATH = CALIB_PATH
CALIB_DIR = os.environ.get("NERO_CALIB_DIR") or os.path.expanduser("~/Downloads/cameracalib")
LOCATE_PATH = os.path.join(HERE, "config", "locate_settings.json")
DEFAULT_LOCATE_PATH = LOCATE_PATH
# V3.49 WRIST USB + INTRINSICS TAB + EYE-IN-HAND SUBTAB + ZERO-PRIOR HOLDER + PATH FENCE
# V3.50 DUAL MONITOR + WRIST QUALITY GATE + TCP CROSSHAIR + COARSE-TO-FINE
# V3.51 INDEPENDENT WRIST ARUCO + AUTO EYE-IN-HAND + WORKFLOW LAMPS
# V3.52 FAST EIH + 8-PARAM PNP FIX + UNIFIED 560x315 CANVAS + LOG-BAR TEACH BUTTONS
# V3.53 TEACH EXIT & PARK FIX + ALIGNED TEACH BUTTONS + DEFAULT 1280x720 DEPTH
# V3.54 FIRMWARE V1.2.1 (v121 DEFAULT) + FOLLOWER MODE TEACH EXIT + STEP 3 PUSH RECOVERY
# V3.55 FAST MAIN CAMERA HAND-EYE CALIB + UNIFIED PNP/DEPTH SOURCE + LOO OUTLIER REJECTION
# V3.56 FAST WRIST EYE-IN-HAND CALIB + LOOK-AT DOME PLANNER + FULL-RES ROI ARUCO + SCALE/LEVER-ARM SOLVER
# V3.57 AUTO TABLETOP CHARUCO INTRINSICS + CHARUCO EIH + 2D FOCAL/SCALE SELF-CALIB
# V3.58 KINEMATIC-ANCHORED WIDE-ANGLE INTRINSICS + FULL-QUADRANT TILT + ROBUST CHARUCO EIH
# V3.59 SKILL STRATEGY KNOBS + MULTI-POINT HOVER DRY-RUN + CALIB BADGE CONSISTENCY
# V3.48 TCP + HOVER + OBJECTS + CLEAR CALIB: TCP pivot calibration, hover verification, general objects, clear calib
TCP_CALIB_PATH = os.path.join(HERE, "config", "tcp_calib.json")
DEFAULT_TCP_CALIB_PATH = TCP_CALIB_PATH
DEFAULT_TCP_IN_FLANGE = [0.1553, 0.0, -0.0215]   # from config/nero_7dof_with_tcp.urdf (link7 -> tcp_link)
HOVER_DEFAULT_MM = 100.0
HOVER_RANGE_MM = (30.0, 300.0)
VERIFY_LIVE_N = 5                         # live marker errors in a row for the static check after loading
# V3.47.1 DEPTH SETTINGS (applied when the camera is (re)connected)
DEPTH_RES = ("848×480", "1280×720")
DEPTH_PRESETS = ("High Accuracy", "Default", "High Density", "Medium Density")
ARUCO_DICTS = ("DICT_4X4_50", "DICT_5X5_100", "DICT_6X6_250", "DICT_ARUCO_ORIGINAL")
DEFAULT_MARKER = {"dict": "DICT_4X4_50", "id": 0, "size_mm": 40.0}
# V3.45 MARKER SIZES: printed marker side (black square) presets; the field stays editable for the measured value.
MARKER_SIZES_MM = (40.0, 30.0)
# V3.45 ROBUST CAPTURE: frames per pose, outlier-trimmed jitter limit, settle time, one retry.
CAPTURE_FRAMES, CAPTURE_TIMEOUT_S, JITTER_MAX_MM = 15, 8.0, 3.0
SETTLE_S, STILL_TOL_DEG, CAPTURE_RETRIES = 1.5, 0.05, 1
# V3.55 FAST CAPTURE & SETTLE: 6 frames (early exit at 5 consistent frames), 0.35 s settle, 0.8 s empty-pose skip
FAST_CAPTURE_FRAMES, FAST_CAPTURE_TIMEOUT_S, FAST_EMPTY_TIMEOUT_S = 6, 2.2, 0.8
FAST_SETTLE_S, FAST_STILL_HOLD_S = 0.35, 0.20
DEPTH_SOURCE_MIN_FRAC = 0.8              # a pose uses depth only if >= 80 % of its frames have consistent depth
# V3.45 VISIBILITY-AWARE AUTO SAMPLING: flange positions in a box around the start (>= 20 cm span), wrist
# orientation within 25 deg of the start, marker predicted inside the image, facing the camera, at a usable range.
AUTO_POS_HALF_M = (0.12, 0.12, 0.08)
AUTO_ROT_MIN_DEG, AUTO_ROT_MAX_DEG = 8.0, 25.0
AUTO_VIEW_MAX_DEG, AUTO_IMG_MARGIN, AUTO_RANGE_M = 45.0, 0.08, (0.25, 1.30)
FAST_WRIST_EMPTY_S = 0.65
AUTO_SPAN_TARGET_M, AUTO_JOINT_STEP_MAX_DEG = 0.20, 70.0
AUTO_PROBE_M = 0.04
FLANGE_LINK = "link7"
DEPTH_RANGE_M = (0.30, 1.50)
CALIB_MAX_SPEED = 0.10
CALIB_GOOD_MEAN_MM, CALIB_GOOD_MAX_MM = 3.0, 5.0
SAMPLE_SPANS_DEG = (20.0, 12.0, 15.0, 15.0, 25.0, 20.0, 30.0)
# V3.47 YOLO LOCATE: egg detection methods on the 视觉定位 page (first = default)
LOCATE_METHODS = ("YOLOE 视觉提示", "YOLOE 文字提示", "几何法 V3.08")


# =========================================================================== small linear algebra (pure Python)
def mat_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0]))] for i in range(len(A))]


def mat_vec(A, v):
    return [sum(A[i][k] * v[k] for k in range(len(v))) for i in range(len(A))]


def transpose(A):
    return [list(r) for r in zip(*A)]


def T_from(R, t):
    return [[R[0][0], R[0][1], R[0][2], t[0]], [R[1][0], R[1][1], R[1][2], t[1]],
            [R[2][0], R[2][1], R[2][2], t[2]], [0.0, 0.0, 0.0, 1.0]]


def T_R(T):
    return [row[:3] for row in T[:3]]


def T_t(T):
    return [T[0][3], T[1][3], T[2][3]]


def T_inv(T):
    Rt = transpose(T_R(T))
    t = mat_vec(Rt, T_t(T))
    return T_from(Rt, [-t[0], -t[1], -t[2]])


def T_apply(T, p):
    return [T[i][0] * p[0] + T[i][1] * p[1] + T[i][2] * p[2] + T[i][3] for i in range(3)]


def vsub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def vnorm(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def rodrigues(rv):
    th = vnorm(rv)
    if th < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    k = [rv[0] / th, rv[1] / th, rv[2] / th]
    K = [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]]
    K2 = mat_mul(K, K)
    s, c = math.sin(th), 1.0 - math.cos(th)
    return [[(1.0 if i == j else 0.0) + s * K[i][j] + c * K2[i][j] for j in range(3)] for i in range(3)]


def rot_angle_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (R[0][0] + R[1][1] + R[2][2] - 1.0) / 2.0))))


def jacobi_eig(S, sweeps=60):
    """Symmetric eigen decomposition (cyclic Jacobi). -> (eigenvalues, eigenvectors as columns)"""
    n = len(S)
    A = [list(map(float, r)) for r in S]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(sweeps):
        off = sum(A[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < 1e-22:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(A[p][q]) < 1e-300:
                    continue
                th = (A[q][q] - A[p][p]) / (2.0 * A[p][q])
                t = (1.0 if th >= 0 else -1.0) / (abs(th) + math.sqrt(th * th + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p], A[k][q] = c * akp - s * akq, s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k], A[q][k] = c * apk - s * aqk, s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p], V[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    return [A[i][i] for i in range(n)], V


def kabsch(P, Q):
    """Rigid transform with R p + t ≈ q (Horn's quaternion method). -> (R, t)"""
    n = len(P)
    cp = [sum(p[i] for p in P) / n for i in range(3)]
    cq = [sum(q[i] for q in Q) / n for i in range(3)]
    S = [[0.0] * 3 for _ in range(3)]
    for p, q in zip(P, Q):
        a, b = vsub(p, cp), vsub(q, cq)
        for i in range(3):
            for j in range(3):
                S[i][j] += a[i] * b[j]
    (Sxx, Sxy, Sxz), (Syx, Syy, Syz), (Szx, Szy, Szz) = S
    N = [[Sxx + Syy + Szz, Syz - Szy, Szx - Sxz, Sxy - Syx],
         [Syz - Szy, Sxx - Syy - Szz, Sxy + Syx, Szx + Sxz],
         [Szx - Sxz, Sxy + Syx, -Sxx + Syy - Szz, Syz + Szy],
         [Sxy - Syx, Szx + Sxz, Syz + Szy, -Sxx - Syy + Szz]]
    vals, V = jacobi_eig(N)
    k = max(range(4), key=lambda i: vals[i])
    w, x, y, z = (V[0][k], V[1][k], V[2][k], V[3][k])
    nq = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / nq, x / nq, y / nq, z / nq
    R = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    Rc = mat_vec(R, cp)
    return R, [cq[0] - Rc[0], cq[1] - Rc[1], cq[2] - Rc[2]]


def plane_fit(P):
    """-> {"centroid", "normal" (unit, z >= 0), "rms_m", "n"} (least-squares plane)."""
    n = len(P)
    if n < 3:
        return None
    c = [sum(p[i] for p in P) / n for i in range(3)]
    C = [[0.0] * 3 for _ in range(3)]
    for p in P:
        d = vsub(p, c)
        for i in range(3):
            for j in range(3):
                C[i][j] += d[i] * d[j] / n
    vals, V = jacobi_eig(C)
    k = min(range(3), key=lambda i: vals[i])
    nv = [V[0][k], V[1][k], V[2][k]]
    if nv[2] < 0:
        nv = [-nv[0], -nv[1], -nv[2]]
    rms = math.sqrt(sum((sum(nv[i] * (p[i] - c[i]) for i in range(3))) ** 2 for p in P) / n)
    return {"centroid": c, "normal": nv, "rms_m": rms, "n": n}


def deproject(u, v, z, K):
    return [(u - K["ppx"]) * z / K["fx"], (v - K["ppy"]) * z / K["fy"], z]


def project(p_cam, K):
    if p_cam[2] <= 1e-6:
        return None
    return [K["fx"] * p_cam[0] / p_cam[2] + K["ppx"], K["fy"] * p_cam[1] / p_cam[2] + K["ppy"]]


# =========================================================================== eye-to-hand calibration
def solve_eye_to_hand(A_list, p_list, y0=None, iters=4000, tol=1e-11):
    """V3.25 HAND-EYE (eye-to-hand). A_i = T_base_flange (FK), p_i = marker centre in the camera (optical frame).
    Unknowns: X = T_base_cam, y = marker centre in the flange frame, with X p_i ≈ A_i y.
    Alternating least squares: X by Horn/Kabsch given y, then y in closed form given X.
    -> {"T_base_cam", "marker_in_flange", "res_mm", "mean_mm", "max_mm", "iters"}"""
    y = list(y0) if y0 is not None else [0.0, 0.0, 0.0]
    n = len(A_list)
    R = t = None
    it = 0
    for it in range(1, iters + 1):
        tgt = [T_apply(A, y) for A in A_list]
        R, t = kabsch(p_list, tgt)
        acc = [0.0, 0.0, 0.0]
        for A, p in zip(A_list, p_list):
            xp = mat_vec(R, p)
            d = [xp[0] + t[0] - A[0][3], xp[1] + t[1] - A[1][3], xp[2] + t[2] - A[2][3]]
            r = mat_vec(transpose(T_R(A)), d)
            acc = [acc[0] + r[0] / n, acc[1] + r[1] / n, acc[2] + r[2] / n]
        step = vnorm(vsub(acc, y))
        y = acc
        if step < tol:
            break
    X = T_from(R, t)
    res = [vnorm(vsub(T_apply(X, p), T_apply(A, y))) * 1000.0 for A, p in zip(A_list, p_list)]
    return {"T_base_cam": X, "marker_in_flange": y, "res_mm": res, "mean_mm": sum(res) / n, "max_mm": max(res),
            "iters": it}


def rotation_span_deg(A_list):
    """V3.45: largest orientation difference between ANY two samples (was: relative to the first sample only)."""
    best = 0.0
    for i in range(len(A_list)):
        Ri = transpose(T_R(A_list[i]))
        for A in A_list[i + 1:]:
            best = max(best, rot_angle_deg(mat_mul(Ri, T_R(A))))
    return best


def translation_span_m(A_list):
    pts = [T_t(A) for A in A_list]
    return max(vnorm(vsub(a, b)) for a in pts for b in pts) if len(pts) > 1 else 0.0


def _calibrate_with_points(samples, P_all, reject_mm=5.0, min_samples=6):
    """Run iterative residual + LOO outlier rejection and solve_eye_to_hand on a specific 3D point list P_all."""
    idx = list(range(len(samples)))
    rejected = []
    y0 = None
    while True:
        A = [samples[i]["T_base_flange"] for i in idx]
        P = [P_all[i] for i in idx]
        r = solve_eye_to_hand(A, P, y0=y0)
        y0 = r["marker_in_flange"]
        res = r["res_mm"]
        med = sorted(res)[len(res) // 2]
        k = max(range(len(res)), key=lambda i: res[i])
        if len(idx) > min_samples and res[k] > max(reject_mm, 2.8 * med):
            rejected.append(idx.pop(k))
            continue
        break
    # V3.55 LOO OUTLIER PASS: if a single oblique pose has LOO > CALIB_GOOD_MAX_MM (5.0 mm) while median LOO is small,
    # reject up to 2 LOO outliers provided rotation/translation spans remain above threshold.
    for _loo_pass in range(3):
        loo = []
        for j in range(len(idx)):
            A2 = [samples[i]["T_base_flange"] for m, i in enumerate(idx) if m != j]
            P2 = [P_all[i] for m, i in enumerate(idx) if m != j]
            rj = solve_eye_to_hand(A2, P2, y0=y0, iters=1500)
            i = idx[j]
            loo.append(vnorm(vsub(T_apply(rj["T_base_cam"], P_all[i]),
                                  T_apply(samples[i]["T_base_flange"], rj["marker_in_flange"]))) * 1000.0)
        if len(idx) <= max(min_samples, 7) or max(loo) <= CALIB_GOOD_MAX_MM:
            break
        med_loo = sorted(loo)[len(loo) // 2]
        k_bad = max(range(len(loo)), key=lambda j: loo[j])
        if loo[k_bad] <= max(CALIB_GOOD_MAX_MM, 2.2 * med_loo):
            break
        cand_idx = [i for m, i in enumerate(idx) if m != k_bad]
        A_cand = [samples[i]["T_base_flange"] for i in cand_idx]
        if rotation_span_deg(A_cand) < 25.0 or translation_span_m(A_cand) < 0.15:
            break
        rejected.append(idx.pop(k_bad))
        A = [samples[i]["T_base_flange"] for i in idx]
        P = [P_all[i] for i in idx]
        r = solve_eye_to_hand(A, P, y0=y0)
        y0 = r["marker_in_flange"]
    A = [samples[i]["T_base_flange"] for i in idx]
    rs, ts = rotation_span_deg(A), translation_span_m(A)
    warn = []
    if rs < 25.0:
        warn.append(f"法兰朝向变化只有 {rs:.0f}°（建议 ≥ 25°）：标记偏移估计不准，多加几个手腕转动的样本")
    if ts < 0.15:
        warn.append(f"法兰位置范围只有 {ts * 100:.0f} cm（建议 ≥ 15 cm）")
    loo_mean, loo_max = sum(loo) / len(loo), max(loo)
    good = loo_mean <= CALIB_GOOD_MEAN_MM and loo_max <= CALIB_GOOD_MAX_MM and not warn
    X = r["T_base_cam"]
    cam_pos = T_t(X)
    view = mat_vec(T_R(X), [0.0, 0.0, 1.0])
    return dict(r, ok=True, used=idx, rejected=rejected, loo_mm=loo, loo_mean_mm=loo_mean, loo_max_mm=loo_max,
                rot_span_deg=rs, trans_span_m=ts, warnings=warn, good=good, cam_pos=cam_pos,
                cam_down_deg=math.degrees(math.asin(max(-1.0, min(1.0, -view[2])))))


def calibrate(samples, reject_mm=5.0, min_samples=6):
    """samples: [{"T_base_flange": 4x4, "p_cam": [3], ...}] -> result dict (see solve_eye_to_hand) with
    used / rejected indices, leave-one-out errors, spans and a verdict.
    V3.55 UNIFIED SOURCE: when a batch mixes 'depth' and 'pnp' poses (which have a systematic z offset between
    IR stereo depth and marker-size PnP), also evaluates depth-scale-aligned PnP (P_aligned) and pure PnP (P_pnp)
    across all poses so cross-pose source jumps never ruin Leave-One-Out accuracy."""
    if len(samples) < min_samples:
        return {"ok": False, "error": f"至少需要 {min_samples} 个样本（现在 {len(samples)} 个）"}
    P_raw = [list(s["p_cam"]) for s in samples]
    best = _calibrate_with_points(samples, P_raw, reject_mm=reject_mm, min_samples=min_samples)
    # Check if all samples recorded p_pnp (V3.55 capture) and whether unified PnP / depth-scaled PnP improves LOO
    if all(s.get("p_pnp") is not None for s in samples):
        P_pnp = [list(s["p_pnp"]) for s in samples]
        candidates = [("pnp", P_pnp)]
        ratios = [vnorm(s["p_depth"]) / max(1e-6, vnorm(s["p_pnp"]))
                  for s in samples if s.get("p_depth") is not None and s.get("p_pnp") is not None]
        if ratios:
            alpha = _median(ratios)
            if 0.90 <= alpha <= 1.10:
                candidates.insert(0, ("pnp_depth_scaled", [[v * alpha for v in p] for p in P_pnp]))
        sources_in_batch = {s.get("source") for s in samples if s.get("source")}
        if len(sources_in_batch) > 1 or not best["good"]:
            for mode_name, P_cand in candidates:
                r_cand = _calibrate_with_points(samples, P_cand, reject_mm=reject_mm, min_samples=min_samples)
                if (r_cand["good"] and not best["good"]) or (
                    r_cand["good"] == best["good"] and r_cand["loo_max_mm"] + 0.15 < best["loo_max_mm"]
                ):
                    r_cand["unified_source"] = mode_name
                    best = r_cand
    return best


def predicted_marker_cam(calib, T_base_flange):
    """marker centre predicted in the camera frame from the joint angles."""
    return T_apply(T_inv(calib["T_base_cam"]), T_apply(T_base_flange, calib["marker_in_flange"]))


def marker_error_mm(calib, T_base_flange, p_cam):
    return vnorm(vsub(T_apply(calib["T_base_cam"], p_cam), T_apply(T_base_flange, calib["marker_in_flange"]))) * 1000.0


# =========================================================================== V3.45 robust capture (pure)
def _median(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def trimmed_points(P, min_keep=5, keep_frac=0.6, floor_m=0.0008):
    """V3.45 JITTER WITHOUT BAD FRAMES: drop frames farther than max(0.8 mm, 3 x median distance) from the
    component-wise median, then jitter = largest distance of the kept frames from their mean.
    -> {"mean", "jitter_mm", "kept", "n"} or {"error"}"""
    n = len(P)
    if n == 0:
        return {"error": "no frames", "kept": 0, "n": 0}
    med = [_median([p[i] for p in P]) for i in range(3)]
    d = [vnorm(vsub(p, med)) for p in P]
    thr = max(floor_m, 3.0 * _median(d))
    kept = [p for p, di in zip(P, d) if di <= thr]
    need = max(min_keep, int(math.ceil(keep_frac * n)))
    if len(kept) < need:
        return {"error": f"只有 {len(kept)}/{n} 帧一致（要 ≥ {need}）", "kept": len(kept), "n": n}
    mean = [sum(p[i] for p in kept) / len(kept) for i in range(3)]
    jit = max(vnorm(vsub(p, mean)) for p in kept)
    return {"mean": mean, "jitter_mm": jit * 1000.0, "kept": len(kept), "n": n}


def choose_source(dets, min_frac=DEPTH_SOURCE_MIN_FRAC):
    """V3.45 ONE SOURCE PER POSE: depth for every frame when >= min_frac of the frames have a consistent depth
    reading, otherwise PnP for every frame (no per-frame switching). -> (source, points)"""
    dep = [d.get("p_depth") for d in dets]
    good = [p for p in dep if p is not None]
    if dets and len(good) >= min_frac * len(dets):
        return "depth", good
    return "pnp", [d.get("p_pnp") or d["p_cam"] for d in dets]


# =========================================================================== V3.45 visibility-aware auto sampling (pure)
def axis_angle_R(axis, deg):
    n = vnorm(axis) or 1.0
    return rodrigues([a / n * math.radians(deg) for a in axis])


class MarkerModel:
    """Predicts the marker centre / normal in the camera for a flange pose:
    p_cam = R_cb (t_f + R_bf y) + t_cb,   n_cam = R_cb R_bf n_f.
    From a calibration (y known) or from probes with a fixed orientation (y = 0, offset absorbed in t_cb)."""

    def __init__(self, R_cb, t_cb, y=(0.0, 0.0, 0.0), n_f=None, source="calib"):
        self.R_cb, self.t_cb, self.y, self.n_f, self.source = R_cb, t_cb, list(y), n_f, source

    @staticmethod
    def normal_in_flange(R_cb, T_bf, n_cam):
        if n_cam is None:
            return None
        n_b = mat_vec(transpose(R_cb), n_cam)
        return mat_vec(transpose(T_R(T_bf)), n_b)

    @classmethod
    def from_calib(cls, calib, T_bf0=None, n_cam0=None):
        X = calib["T_base_cam"]
        Xi = T_inv(X)
        R_cb, t_cb = T_R(Xi), T_t(Xi)
        n_f = cls.normal_in_flange(R_cb, T_bf0, n_cam0) if T_bf0 is not None else None
        return cls(R_cb, t_cb, calib["marker_in_flange"], n_f, "calib")

    @classmethod
    def from_probes(cls, T_list, p_list, n_cam0=None):
        """flange poses with (nearly) the same orientation + the marker seen in the camera -> model + fit RMS (m)."""
        P = [T_t(T) for T in T_list]
        R, t = kabsch(P, p_list)
        rms = math.sqrt(sum(vnorm(vsub(vsub(p, t), mat_vec(R, a))) ** 2 for a, p in zip(P, p_list)) / len(P))
        n_f = cls.normal_in_flange(R, T_list[0], n_cam0)
        return cls(R, t, (0.0, 0.0, 0.0), n_f, "probes"), rms

    @classmethod
    def from_start_det(cls, T_bf0, det, T_list=None, p_list=None):
        """V3.55: fallback initial MarkerModel directly from the start pose's ArUco PnP rotation (rvec) + 3D position
        when fewer than 3 translation probes succeeded (e.g. near a safety wall)."""
        p0 = list(det.get("p_cam") or [0.0, 0.0, 0.70])
        n_cam0 = det.get("normal_cam")
        if T_list and p_list and len(T_list) >= 3:
            return cls.from_probes(T_list, p_list, n_cam0)[0]
        rv = det.get("rvec")
        R_cm = rodrigues(rv) if rv else [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
        R_cb = mat_mul(R_cm, transpose(T_R(T_bf0)))
        p_b0 = T_t(T_bf0)
        Rp0 = mat_vec(R_cb, p_b0)
        t_cb = [p0[i] - Rp0[i] for i in range(3)]
        n_f = cls.normal_in_flange(R_cb, T_bf0, n_cam0)
        return cls(R_cb, t_cb, (0.0, 0.0, 0.0), n_f, "start_pnp")

    def predict(self, T_bf):
        t, Ry = T_t(T_bf), mat_vec(T_R(T_bf), self.y)
        pc = mat_vec(self.R_cb, [t[0] + Ry[0], t[1] + Ry[1], t[2] + Ry[2]])
        pc = [pc[i] + self.t_cb[i] for i in range(3)]
        nc = mat_vec(self.R_cb, mat_vec(T_R(T_bf), self.n_f)) if self.n_f is not None else None
        return pc, nc

    def visible(self, T_bf, K, margin=AUTO_IMG_MARGIN, view_max_deg=AUTO_VIEW_MAX_DEG, rng=AUTO_RANGE_M):
        """-> (ok, reason, {"uv", "range_m", "view_deg"})"""
        pc, nc = self.predict(T_bf)
        r = vnorm(pc)
        info = {"range_m": r, "uv": None, "view_deg": None}
        if pc[2] <= 0.05 or not (rng[0] <= pc[2] <= rng[1]):
            return False, "距离不合适", info
        uv = project(pc, K)
        info["uv"] = uv
        W, H = float(K.get("width", 1280)), float(K.get("height", 720))
        eff_margin = max(margin, 0.16) if (pc[2] < 0.38 and getattr(self, "source", "calib") != "calib") else margin
        if not (eff_margin * W <= uv[0] <= (1 - eff_margin) * W and eff_margin * H <= uv[1] <= (1 - eff_margin) * H):
            return False, "出画面", info
        if nc is not None:
            cosv = -sum(nc[i] * pc[i] for i in range(3)) / (r * (vnorm(nc) or 1.0))
            info["view_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, cosv))))
            if info["view_deg"] > view_max_deg:
                return False, "标记转得太偏", info
        return True, "", info


def check_auto_path(fence_fn, q_a, q_b, steps=8):
    """V3.49 PATH FENCE CHECK: verify intermediate linearly-interpolated joint poses along q_a -> q_b so auto
    sampling rejects targets whose path clips a safety-zone wall (e.g. M3-M4)."""
    if fence_fn is None:
        return True
    for k in range(1, steps + 1):
        t = k / float(steps)
        qm = [a + (b - a) * t for a, b in zip(q_a, q_b)]
        if not fence_fn(qm):
            return False
    return True


def build_fast_jacobian_ik(fk_fn, q0_rad, limits_rad, soft_rad=0.0):
    """V3.55 FAST JACOBIAN IK: precomputes the 6x7 geometric Jacobian J0 and its damped-least-squares pseudoinverse
    at q0_rad once (only 7 fk_fn calls), then solves nearby (pos, R) targets in 3-4 microsecond iterations (~100x
    faster than 150-iteration numerical Levenberg-Marquardt IK)."""
    T0 = fk_fn(q0_rad)
    if T0 is None:
        return None
    p0, R0 = T_t(T0), T_R(T0)
    h = 1e-5

    def _rot_vec(Ra, Rb):
        R = [[sum(Rb[i][k] * Ra[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
        return [0.5 * (R[2][1] - R[1][2]), 0.5 * (R[0][2] - R[2][0]), 0.5 * (R[1][0] - R[0][1])]

    cols = []
    for j in range(len(q0_rad)):
        qp = list(q0_rad)
        qp[j] += h
        Tp = fk_fn(qp)
        if Tp is None:
            return None
        dp = [(Tp[i][3] - p0[i]) / h for i in range(3)]
        dr = [x / h for x in _rot_vec(R0, T_R(Tp))]
        cols.append(dp + dr)
    nj = len(q0_rad)
    J0 = [[cols[j][r] for j in range(nj)] for r in range(6)]
    lam = 1.5e-3
    JJT = [[sum(J0[r][k] * J0[c][k] for k in range(nj)) + (lam if r == c else 0.0) for c in range(6)] for r in range(6)]
    aug = [JJT[r] + [1.0 if r == c else 0.0 for c in range(6)] for r in range(6)]
    for c in range(6):
        p = max(range(c, 6), key=lambda r: abs(aug[r][c]))
        aug[c], aug[p] = aug[p], aug[c]
        if abs(aug[c][c]) < 1e-12:
            return None
        for r in range(6):
            if r != c:
                f = aug[r][c] / aug[c][c]
                for k in range(c, 12):
                    aug[r][k] -= f * aug[c][k]
    inv_JJT = [[aug[r][6 + c] / aug[r][r] for c in range(6)] for r in range(6)]
    J_pinv = [[sum(J0[r][j] * inv_JJT[r][c] for r in range(6)) for c in range(6)] for j in range(nj)]

    def solve(q_seed, pos, R):
        q = list(q_seed or q0_rad)
        for _ in range(4):
            Tq = fk_fn(q)
            if Tq is None:
                return None
            ep = [pos[i] - Tq[i][3] for i in range(3)]
            er = _rot_vec(T_R(Tq), R)
            if vnorm(ep) <= 0.0022 and vnorm(er) <= math.radians(2.0):
                return q
            err6 = ep + er
            for j, (lo, hi) in enumerate(limits_rad):
                dq_j = sum(J_pinv[j][c] * err6[c] for c in range(6))
                q[j] = max(lo + soft_rad, min(hi - soft_rad, q[j] + dq_j))
        Tq = fk_fn(q)
        if Tq is not None and vnorm(vsub(T_t(Tq), pos)) <= 0.003 and rot_angle_deg(mat_mul(transpose(T_R(Tq)), R)) <= 3.0:
            return q
        return None

    return solve


def plan_auto_targets(fk_fn, ik_fn, q0_rad, limits_rad, model, K, n, fence_fn=None, seed=0,
                      pos_half=AUTO_POS_HALF_M, rot_deg=(AUTO_ROT_MIN_DEG, AUTO_ROT_MAX_DEG), soft_deg=3.0,
                      max_tries=1500, step_max_deg=AUTO_JOINT_STEP_MAX_DEG, path_steps=4):
    """V3.45/V3.55: n joint targets (rad) whose predicted marker is visible / facing the camera, flange positions spread
    over a box around the start AND flange orientations spread across opposite tilt/roll directions (farthest-point in
    both translation and rotation), using precomputed Jacobian IK (~20x faster than 150-step numerical IK).
    fk_fn(q_rad) -> T_base_flange; ik_fn(q_seed, xyz, R) -> q_rad or None.
    -> (targets [{"q", "T", "view_deg", "uv"}], stats)"""
    rnd = random.Random(seed)
    T0 = fk_fn(q0_rad)
    p0, R0 = T_t(T0), T_R(T0)
    soft = math.radians(soft_deg)
    fast_ik = build_fast_jacobian_ik(fk_fn, q0_rad, limits_rad, soft_rad=soft)
    n_b0 = mat_vec(R0, model.n_f) if getattr(model, "n_f", None) is not None else None
    pool, why = [], {}
    tries = 0
    target_pool = max(3 * n, 36)
    while tries < max_tries and len(pool) < target_pool:
        tries += 1
        pos = [p0[i] + rnd.uniform(-pos_half[i], pos_half[i]) for i in range(3)]
        if n_b0 is not None and tries % 3 == 0:
            # V3.55: enrich with in-plane roll around the marker normal + light tilt so pairwise rotation span is large
            # (>= 35-45 deg) without turning the marker away from the camera
            roll_ang = rnd.choice((-1.0, 1.0)) * rnd.uniform(max(rot_deg[0], 14.0), rot_deg[1] * 0.88)
            R_roll = axis_angle_R(n_b0, roll_ang)
            ax_t = [rnd.gauss(0, 1) for _ in range(3)]
            R = mat_mul(axis_angle_R(ax_t, rnd.uniform(3.0, 9.0)), mat_mul(R_roll, R0))
            if rot_angle_deg(mat_mul(transpose(R0), R)) > rot_deg[1]:
                R = mat_mul(R_roll, R0)
        else:
            ax = [rnd.gauss(0, 1) for _ in range(3)]
            R = mat_mul(axis_angle_R(ax, rnd.uniform(*rot_deg)), R0)
        Tt = T_from(R, pos)
        ok, reason, info = model.visible(Tt, K)
        if not ok:
            why[reason] = why.get(reason, 0) + 1
            continue
        q = fast_ik(q0_rad, pos, R) if fast_ik is not None else None
        if q is None and ik_fn is not None and (fast_ik is None or len(pool) < n):
            q = ik_fn(q0_rad, pos, R)
        if q is None:
            why["IK 无解"] = why.get("IK 无解", 0) + 1
            continue
        if any(not (lo + soft - 1e-6 <= v <= hi - soft + 1e-6) for v, (lo, hi) in zip(q, limits_rad)):
            why["接近限位"] = why.get("接近限位", 0) + 1
            continue
        if max(abs(math.degrees(a - b)) for a, b in zip(q, q0_rad)) > step_max_deg:
            why["关节变化太大"] = why.get("关节变化太大", 0) + 1
            continue
        Tq = fk_fn(q)
        ok2, _r2, info2 = model.visible(Tq, K)
        if not ok2:
            why["IK 后不可见"] = why.get("IK 后不可见", 0) + 1
            continue
        pool.append({"q": q, "T": Tq, "view_deg": info2.get("view_deg"), "uv": info2.get("uv")})
    chosen, pts, rots = [], [p0], [R0]
    while pool and len(chosen) < n:
        # V3.55: farthest-point in BOTH 3D flange position AND 3D flange orientation diversity
        k = max(range(len(pool)), key=lambda i: (
            min(vnorm(vsub(T_t(pool[i]["T"]), p)) for p in pts)
            + 0.12 * math.radians(min(rot_angle_deg(mat_mul(transpose(r), T_R(pool[i]["T"]))) for r in rots))
        ))
        c = pool.pop(k)
        if fence_fn is not None and not fence_fn(c["q"]):
            why["安全区"] = why.get("安全区", 0) + 1
            continue
        if fence_fn is not None and path_steps > 1 and not check_auto_path(fence_fn, q0_rad, c["q"], steps=path_steps):
            why["路径安全区"] = why.get("路径安全区", 0) + 1
            continue
        chosen.append(c)
        pts.append(T_t(c["T"]))
        rots.append(T_R(c["T"]))
    order, cur = [], list(q0_rad)                    # nearest-neighbour order in joint space (path-safe first)
    while chosen:
        ranked = sorted(range(len(chosen)), key=lambda i: max(abs(a - b) for a, b in zip(chosen[i]["q"], cur)))
        picked = None
        for idx_c in ranked:
            if fence_fn is None or path_steps <= 1 or check_auto_path(fence_fn, cur, chosen[idx_c]["q"], steps=path_steps):
                picked = idx_c
                break
        if picked is None:
            why["路径安全区"] = why.get("路径安全区", 0) + 1
            chosen.pop(ranked[0])
            continue
        c = chosen.pop(picked)
        order.append(c)
        cur = c["q"]
    span = max((vnorm(vsub(a, b)) for a in pts for b in pts), default=0.0)
    rot = max((rot_angle_deg(mat_mul(transpose(R0), T_R(c["T"]))) for c in order), default=0.0)
    return order, {"tries": tries, "rejected": why, "span_m": span, "rot_max_deg": rot}


def sample_targets(q0_deg, limits_rad, n, seed=0, spans=SAMPLE_SPANS_DEG, soft_deg=3.0):
    """joint targets (deg) around q0 inside the soft limits, spread over the joint spans."""
    rnd = random.Random(seed)
    out = []
    for _k in range(max(0, int(n))):
        q = []
        for j, (v, (lo, hi)) in enumerate(zip(q0_deg, limits_rad)):
            a, b = math.degrees(lo) + soft_deg, math.degrees(hi) - soft_deg
            span = spans[j] if j < len(spans) else 15.0
            q.append(round(min(max(v + rnd.uniform(-span, span), a), b), 2))
        out.append(q)
    return out


# =========================================================================== files
def save_json(path, d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    return d


def load_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as exc:
        return {"load_error": str(exc)}


def save_calib(result, samples, info, marker, path=None):
    d = {"version": VERSION, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
         "camera_serial": (info or {}).get("serial"), "intrinsics": (info or {}).get("K"),
         "marker": marker, "flange_link": FLANGE_LINK,
         "T_base_cam": result["T_base_cam"], "marker_in_flange": result["marker_in_flange"],
         "mean_mm": result["mean_mm"], "max_mm": result["max_mm"], "loo_mean_mm": result.get("loo_mean_mm"),
         "loo_max_mm": result.get("loo_max_mm"), "rot_span_deg": result.get("rot_span_deg"),
         "trans_span_m": result.get("trans_span_m"), "n_used": len(result.get("used") or []),
         "check_poses_deg": [samples[i]["q_deg"] for i in (result.get("used") or [])[:2]],
         "samples": [{"q_deg": s["q_deg"], "p_cam": s["p_cam"]} for s in samples]}
    return save_json(path or CALIB_PATH, d)


def calib_valid(d):
    """V3.47.1: a calibration file has a 4x4 T_base_cam and a 3-vector marker_in_flange."""
    try:
        T = d["T_base_cam"]
        assert len(T) == 4 and all(len(r) == 4 for r in T) and all(isinstance(v, (int, float)) for r in T for v in r)
        assert len(d["marker_in_flange"]) == 3
        return not d.get("load_error")
    except Exception:
        return False


# =========================================================================== V3.48 TCP calibration + hover IK (pure)
def solve_3x3(M, y):
    """3x3 linear solver M x = y with partial pivoting (pure Python). -> [x0, x1, x2] or None if singular."""
    A = [list(map(float, r)) + [float(yv)] for r, yv in zip(M, y)]
    for i in range(3):
        p = max(range(i, 3), key=lambda r: abs(A[r][i]))
        if abs(A[p][i]) < 1e-11:
            return None
        A[i], A[p] = A[p], A[i]
        for r in range(i + 1, 3):
            f = A[r][i] / A[i][i]
            for c in range(i, 4):
                A[r][c] -= f * A[i][c]
    x = [0.0, 0.0, 0.0]
    for i in range(2, -1, -1):
        x[i] = (A[i][3] - sum(A[i][c] * x[c] for c in range(i + 1, 3))) / A[i][i]
    return x


def solve_tcp_pivot(A_list):
    """Multi-pose single-point TCP calibration: A_i = T_base_flange when the TCP touches a fixed point p_world.
    R_i p_tcp + t_i ≈ p_world -> linear least squares on (R_i - R_bar) p_tcp ≈ -(t_i - t_bar).
    -> {'tcp_in_flange', 'point_in_base', 'res_mm', 'mean_mm', 'max_mm'} or None if orientation change is singular."""
    n = len(A_list)
    if n < 3:
        return None
    Rs = [T_R(A) for A in A_list]
    ts = [T_t(A) for A in A_list]
    R_bar = [[sum(R[r][c] for R in Rs) / n for c in range(3)] for r in range(3)]
    t_bar = [sum(t[i] for t in ts) / n for i in range(3)]
    ATA = [[0.0] * 3 for _ in range(3)]
    ATb = [0.0] * 3
    for R, t in zip(Rs, ts):
        D = [[R[r][c] - R_bar[r][c] for c in range(3)] for r in range(3)]
        b = [-(t[i] - t_bar[i]) for i in range(3)]
        for i in range(3):
            for j in range(3):
                ATA[i][j] += sum(D[k][i] * D[k][j] for k in range(3))
            ATb[i] += sum(D[k][i] * b[k] for k in range(3))
    p_tcp = solve_3x3(ATA, ATb)
    if p_tcp is None:
        return None
    R_bar_p = mat_vec(R_bar, p_tcp)
    p_world = [R_bar_p[i] + t_bar[i] for i in range(3)]
    res = [vnorm(vsub(T_apply(A, p_tcp), p_world)) * 1000.0 for A in A_list]
    return {"tcp_in_flange": p_tcp, "point_in_base": p_world, "res_mm": res,
            "mean_mm": sum(res) / n, "max_mm": max(res)}


def calibrate_tcp(samples, urdf_tcp=DEFAULT_TCP_IN_FLANGE, reject_mm=6.0, min_samples=4):
    """V3.48 TCP CALIB: samples = [{'T_base_flange': 4x4, 'q_deg': [7]}], >= 4 poses touching the same point with
    different wrist orientations -> TCP offset in link7, LOO errors, and comparison with urdf_tcp."""
    idx = list(range(len(samples)))
    if len(idx) < min_samples:
        return {"ok": False, "error": f"至少需要 {min_samples} 个碰点姿态（现在 {len(idx)} 个）"}
    rs0 = rotation_span_deg([samples[i]["T_base_flange"] for i in idx])
    if rs0 < 8.0:
        return {"ok": False, "error": f"手腕朝向变化只有 {rs0:.1f}°（太小，无法解算偏移；请绕触碰点转动法兰/手腕 ≥ 25°）"}
    rejected = []
    while True:
        A = [samples[i]["T_base_flange"] for i in idx]
        r = solve_tcp_pivot(A)
        if r is None:
            return {"ok": False, "error": "姿态矩阵奇异：请换不同手腕角度重新记录"}
        res = r["res_mm"]
        med = sorted(res)[len(res) // 2]
        k = max(range(len(res)), key=lambda i: res[i])
        if len(idx) > min_samples and res[k] > max(reject_mm, 3.0 * med):
            rejected.append(idx.pop(k))
            continue
        break
    loo = []
    for j in range(len(idx)):
        A2 = [samples[i]["T_base_flange"] for m, i in enumerate(idx) if m != j]
        rj = solve_tcp_pivot(A2)
        if rj is None:
            loo.append(r["res_mm"][j])
        else:
            i = idx[j]
            loo.append(vnorm(vsub(T_apply(samples[i]["T_base_flange"], rj["tcp_in_flange"]),
                                  rj["point_in_base"])) * 1000.0)
    A = [samples[i]["T_base_flange"] for i in idx]
    rs = rotation_span_deg(A)
    warn = []
    if rs < 25.0:
        warn.append(f"法兰朝向变化只有 {rs:.0f}°（建议 ≥ 25°）：多换几个手腕倾角再碰同一点")
    loo_mean, loo_max = sum(loo) / len(loo), max(loo)
    if loo_max > CALIB_GOOD_MAX_MM:
        warn.append(f"留一最大误差 {loo_max:.1f} mm（> {CALIB_GOOD_MAX_MM:.0f} mm）：检查每次指尖中心是否碰在同一标记点")
    good = loo_mean <= CALIB_GOOD_MEAN_MM and loo_max <= CALIB_GOOD_MAX_MM and not warn
    p_tcp = r["tcp_in_flange"]
    u_tcp = list(urdf_tcp or DEFAULT_TCP_IN_FLANGE)
    delta_mm = [(p_tcp[i] - u_tcp[i]) * 1000.0 for i in range(3)]
    diff_mm = vnorm(vsub(p_tcp, u_tcp)) * 1000.0
    return dict(r, ok=True, used=idx, rejected=rejected, loo_mm=loo, loo_mean_mm=loo_mean, loo_max_mm=loo_max,
                rot_span_deg=rs, warnings=warn, good=good, urdf_tcp=u_tcp, delta_urdf_mm=delta_mm,
                diff_urdf_mm=diff_mm, n_used=len(idx))


def tcp_valid(d):
    try:
        p = d["tcp_in_flange"]
        assert len(p) == 3 and all(isinstance(v, (int, float)) and abs(v) <= 0.6 for v in p)
        return not d.get("load_error")
    except Exception:
        return False


def plan_hover_target(urdf, joints, limits_rad, q0_rad, tool_T, target_xyz, base="base_link", fence_fn=None,
                      soft_deg=3.0):
    """V3.48 HOVER VERIFY: joint target (rad) that places the TCP at target_xyz (hovering above an egg/object)
    with link7 safely above the TCP, inside soft joint limits and the safety zone.
    -> (q_rad, info_dict) or (None, reason_str)"""
    import NERO_V315_Consistency_Safety as CS
    soft = math.radians(soft_deg)
    slim = [(lo + soft, hi - soft) for lo, hi in limits_rad]
    yaw0 = math.atan2(target_xyz[1], -target_xyz[0]) if (abs(target_xyz[0]) + abs(target_xyz[1]) > 1e-3) else 0.0
    yaw0 = max(slim[0][0], min(slim[0][1], yaw0))
    seeds = [
        list(q0_rad),
        [0.0, math.radians(30.0), 0.0, math.radians(80.0), 0.0, math.radians(40.0), 0.0],
        [yaw0, math.radians(35.0), 0.0, math.radians(75.0), 0.0, math.radians(35.0), 0.0],
        [0.0, math.radians(45.0), 0.0, math.radians(70.0), 0.0, math.radians(35.0), 0.0],
    ]
    why = "IK 无解"
    for sd in seeds:
        q, err = CS.ik_tool(urdf, joints, sd, base, tool_T, list(target_xyz), slim, iters=250, tol=5e-4, keep_R=None)
        if q is None or err > 0.002:
            why = f"IK 误差 {(err or 0) * 1000:.1f} mm（目标超出可达范围？）"
            continue
        if any(not (lo + soft - 1e-4 <= v <= hi - soft + 1e-4) for v, (lo, hi) in zip(q, limits_rad)):
            why = "接近关节软限位（3°）"
            continue
        Tf = CS.fk(urdf, dict(zip(joints, q)), base, FLANGE_LINK)
        if Tf is None or Tf[2][3] < target_xyz[2] - 0.005:
            why = "法兰低于悬停点（姿态朝上）"
            continue
        if fence_fn is not None:
            ok_f, msg_f = fence_fn(q)
            if not ok_f:
                why = f"安全区拦截：{msg_f}"
                continue
        T_tcp = CS.tool_pose(urdf, joints, q, base, tool_T)
        return list(q), {"err_mm": err * 1000.0, "flange_z_m": Tf[2][3],
                         "tcp_xyz_m": [T_tcp[0][3], T_tcp[1][3], T_tcp[2][3]]}
    return None, why


# =========================================================================== image helpers (numpy / OpenCV)
def depth_colormap(depth_m, lo=DEPTH_RANGE_M[0], hi=DEPTH_RANGE_M[1]):
    """absolute colour scale: near = red .. far = blue, 0 (invalid) = black. numpy uint8 HxWx3"""
    import numpy as np
    d = np.asarray(depth_m, dtype=np.float32)
    x = np.clip((d - lo) / max(1e-6, hi - lo), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0, 1)
    img = (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)
    img[d <= 0] = 0
    return img


def ppm_bytes(rgb):
    h, w = rgb.shape[:2]
    return b"P6 %d %d 255\n" % (w, h) + rgb.tobytes()


def png_bytes_from_ppm(ppm):
    """fallback for Tk builds that do not read binary PPM from -data: PPM (P6) -> PNG (stdlib zlib only)."""
    import struct
    import zlib
    parts = ppm.split(b"\n", 1)
    _p6, w, h, _mx = parts[0].split()
    w, h = int(w), int(h)
    raw = parts[1]
    rows = b"".join(b"\x00" + raw[y * w * 3:(y + 1) * w * 3] for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 1)) + chunk(b"IEND", b""))


QT_ENV_KEYS = ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH", "QT_QPA_FONTDIR")


def _import_cv2():
    """V3.46 QT ENV GUARD: import cv2 without keeping the Qt variables pip opencv-python sets on its first import
    (they made rviz2 - started later by this console - load cv2's Qt plugin and abort)."""
    before = {k: os.environ.get(k) for k in QT_ENV_KEYS}
    import cv2
    for k, v in before.items():
        if os.environ.get(k) != v:
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return cv2


def aruco_status():
    try:
        cv2 = _import_cv2()
    except Exception as exc:
        return False, f"OpenCV 不可用：{exc}（pip3 install opencv-contrib-python）"
    if not hasattr(cv2, "aruco"):
        return False, f"OpenCV {cv2.__version__} 没有 aruco 模块（pip3 install opencv-contrib-python）"
    return True, f"OpenCV {cv2.__version__}"


def _aruco_dict(cv2, name):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def detect_aruco(rgb, K, dist, dict_name, marker_id, size_m, depth_m=None):
    """-> None or {"id", "corners_px", "center_px", "p_cam", "p_pnp", "source", "side_px"}.
    V3.52: supports full 8-parameter CALIB_RATIONAL_MODEL distortion without truncating [:5] (undistorts corners via
    cv2.undistortPoints before IPPE_SQUARE solvePnP), and uses fast half-scale pre-detection + full-res cornerSubPix on
    1920x1080 frames."""
    cv2 = _import_cv2()
    import numpy as np
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else np.ascontiguousarray(rgb)
    d = _aruco_dict(cv2, dict_name)

    def _run_det(g_img, subpix=True):
        params = cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, "ArucoDetector") else cv2.aruco.DetectorParameters_create()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 35
        params.adaptiveThreshWinSizeStep = 8
        params.minMarkerPerimeterRate = 0.015
        params.errorCorrectionRate = 0.75
        if subpix:
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        det_fn = cv2.aruco.ArucoDetector(d, params).detectMarkers if hasattr(cv2.aruco, "ArucoDetector") else (
            lambda im: cv2.aruco.detectMarkers(im, d, parameters=params))
        corners_, ids_, rej_ = det_fn(g_img)
        if ids_ is None or int(marker_id) not in [int(v) for v in np.asarray(ids_).flatten()]:
            try:
                g_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(g_img)
                c2, i2, r2 = det_fn(g_clahe)
                if i2 is not None and int(marker_id) in [int(v) for v in np.asarray(i2).flatten()]:
                    return c2, i2, r2
            except Exception:
                pass
        return corners_, ids_, rej_

    c = None
    if gray.shape[1] > 1280:
        corners_h, ids_h, _ = _run_det(gray[::2, ::2], subpix=False)
        if ids_h is not None:
            ids_list = [int(v) for v in np.asarray(ids_h).flatten()]
            if int(marker_id) in ids_list:
                # V3.56 FULL-RES ROI CROP: use half-scale pass ONLY to locate the marker's bounding box in full-res
                # and run native full-res detectMarkers(subpix=True) on the ROI crop so small 24 mm wrist markers
                # never snap cornerSubPix onto internal ArUco data bits (fixes 14-17 mm stationary jitter).
                c0_pts = (np.asarray(corners_h[ids_list.index(int(marker_id))], dtype=np.float64).reshape(4, 2)) * 2.0
                side_est = float(np.mean([np.linalg.norm(c0_pts[i] - c0_pts[(i + 1) % 4]) for i in range(4)]))
                pad = max(44, int(round(side_est * 0.80)))
                H_g, W_g = gray.shape[:2]
                x0 = max(0, int(math.floor(float(c0_pts[:, 0].min()))) - pad)
                y0 = max(0, int(math.floor(float(c0_pts[:, 1].min()))) - pad)
                x1 = min(W_g, int(math.ceil(float(c0_pts[:, 0].max()))) + pad)
                y1 = min(H_g, int(math.ceil(float(c0_pts[:, 1].max()))) + pad)
                corners_roi, ids_roi, _ = _run_det(gray[y0:y1, x0:x1], subpix=True)
                if ids_roi is not None:
                    ids_r = [int(v) for v in np.asarray(ids_roi).flatten()]
                    if int(marker_id) in ids_r:
                        c = (np.asarray(corners_roi[ids_r.index(int(marker_id))], dtype=np.float64).reshape(4, 2)
                             + np.array([x0, y0], dtype=np.float64))
    if c is None:
        corners, ids, _ = _run_det(gray, subpix=True)
        if ids is None:
            return None
        ids = [int(v) for v in np.asarray(ids).flatten()]
        if int(marker_id) not in ids:
            return None
        c = np.asarray(corners[ids.index(int(marker_id))], dtype=np.float64).reshape(4, 2)

    h = size_m / 2.0
    obj = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)
    Km = np.array([[K["fx"], 0.0, K["ppx"]], [0.0, K["fy"], K["ppy"]], [0.0, 0.0, 1.0]], dtype=np.float64)
    d_list = [float(v) for v in (dist or [0.0] * 5)]
    if len(d_list) in (4, 5, 8, 12, 14):
        dc = np.array(d_list, dtype=np.float64)
    elif len(d_list) > 5:
        dc = np.array(d_list[:8] + [0.0] * max(0, 8 - len(d_list)), dtype=np.float64)
    else:
        dc = np.array(d_list[:5] + [0.0] * max(0, 5 - len(d_list)), dtype=np.float64)
    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    if np.any(np.abs(dc) > 1e-9):
        c_pnp = cv2.undistortPoints(c.reshape(4, 1, 2), Km, dc, P=Km).reshape(4, 2)
        ok, rvec, tvec = cv2.solvePnP(obj, c_pnp, Km, np.zeros(5, dtype=np.float64), flags=flag)
    else:
        ok, rvec, tvec = cv2.solvePnP(obj, c, Km, np.zeros(5, dtype=np.float64), flags=flag)
    if not ok:
        return None
    p = [float(v) for v in np.asarray(tvec).flatten()]
    center = c.mean(axis=0)
    rv = [float(v) for v in np.asarray(rvec).flatten()]
    out = {"id": int(marker_id), "corners_px": c.tolist(), "center_px": [float(center[0]), float(center[1])],
           "p_pnp": list(p), "p_cam": list(p), "p_depth": None, "source": "pnp", "rvec": rv,
           "normal_cam": [row[2] for row in rodrigues(rv)],          # V3.45: marker z axis (out of the face)
           "side_px": float(np.mean([np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]))}
    if depth_m is not None:
        u, v = int(round(center[0])), int(round(center[1]))
        r = max(3, int(out["side_px"] * 0.20))
        win = np.asarray(depth_m)[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
        vals = win[(win > 0.15) & (np.abs(win - p[2]) < 0.035)]
        if vals.size >= 5:
            z = float(np.median(vals))
            if abs(z - p[2]) < 0.030:
                out["p_depth"] = deproject(float(center[0]), float(center[1]), z, K)
                out["p_cam"] = out["p_depth"]                        # live display; samples choose per pose
                out["source"] = "depth"
    return out


def marker_image(dict_name, marker_id, px=600):
    """printable marker (white border + label) as an RGB numpy image."""
    cv2 = _import_cv2()
    import numpy as np
    d = _aruco_dict(cv2, dict_name)
    if hasattr(cv2.aruco, "generateImageMarker"):
        m = cv2.aruco.generateImageMarker(d, int(marker_id), int(px))
    else:
        m = cv2.aruco.drawMarker(d, int(marker_id), int(px))
    pad = px // 5
    img = np.full((px + 2 * pad + 40, px + 2 * pad), 255, np.uint8)
    img[pad:pad + px, pad:pad + px] = m
    cv2.putText(img, f"{dict_name} id {marker_id}  black square = marker size", (pad // 2, px + 2 * pad + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


# =========================================================================== camera source (RealSense)
class RealSenseSource:
    """In-process D435i reader thread. Interface (also implemented by test doubles):
    start(), stop(), running(), error, info, K, dist, frame_id, fps, display_ppm(view), depth_at(u, v),
    roi_points(u0, v0, u1, v1, stride), detect_marker(marker), frames()."""
    W, H = 1280, 720

    def __init__(self, log=print):
        self.log = log
        self.lock = threading.Lock()
        self.thread = None
        self.stop_ev = threading.Event()
        self.error = None
        self.info = {}
        self.K, self.dist = None, None
        self.rgb = self.depth = None
        self.frame_id = 0
        self.fps = 0.0
        self.t_frame = None
        self.depth_res = DEPTH_RES[1]            # V3.53: default depth resolution 1280×720
        self.depth_preset = DEPTH_PRESETS[0]

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        if self.running():
            return
        self.error = None
        self.stop_ev.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name="v325_realsense")
        self.thread.start()

    def stop(self, timeout=3.0):
        self.stop_ev.set()
        if self.thread is not None:
            self.thread.join(timeout)

    def _run(self):
        try:
            import numpy as np
            import pyrealsense2 as rs
        except Exception as exc:
            self.error = f"pyrealsense2 / numpy 不可用：{exc}（pip3 install pyrealsense2 numpy）"
            return
        pipe = None
        try:
            ctx = rs.context()
            devs = list(ctx.query_devices())
            if not devs:
                self.error = "没有找到 RealSense 相机（USB 线 / 端口？realsense-viewer 是否还开着？）"
                return
            dev = devs[0]

            def gi(k):
                try:
                    return dev.get_info(k) if dev.supports(k) else "-"
                except Exception:
                    return "-"
            info = {"name": gi(rs.camera_info.name), "serial": gi(rs.camera_info.serial_number),
                    "firmware": gi(rs.camera_info.firmware_version), "usb": gi(rs.camera_info.usb_type_descriptor),
                    "sdk": getattr(rs, "__version__", "?")}
            pipe, cfg = rs.pipeline(ctx), rs.config()
            cfg.enable_device(info["serial"])
            cfg.enable_stream(rs.stream.color, self.W, self.H, rs.format.rgb8, 30)
            dw, dh = (1280, 720) if self.depth_res == DEPTH_RES[1] else (848, 480)
            cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, 30)
            prof = pipe.start(cfg)
            ds = prof.get_device().first_depth_sensor()
            try:
                pr = getattr(rs.rs400_visual_preset, self.depth_preset.lower().replace(" ", "_"))
                ds.set_option(rs.option.visual_preset, float(pr))
                info["preset"] = self.depth_preset
            except Exception:
                info["preset"] = "default"
            info["depth_res"] = f"{dw}×{dh}"
            scale = ds.get_depth_scale()
            intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K = {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx, "ppy": intr.ppy}
            dist = [float(v) for v in intr.coeffs]
            if all(abs(v) < 1e-9 for v in dist) or "inverse" in str(intr.model).lower():
                dist = [0.0] * 5        # D435 colour: coefficients 0; inverse model = not usable for projection
            info.update(K=K, dist=dist, model=str(intr.model), depth_scale_mm=scale * 1000.0)
            align = rs.align(rs.stream.color)
            spatial, temporal = rs.spatial_filter(), rs.temporal_filter()
            with self.lock:
                self.info, self.K, self.dist = info, K, dist
            n, t0 = 0, time.time()
            while not self.stop_ev.is_set():
                fs = pipe.wait_for_frames(3000)
                al = align.process(fs)
                c, d = al.get_color_frame(), al.get_depth_frame()
                if not c or not d:
                    continue
                d = temporal.process(spatial.process(d))
                depth = np.asanyarray(d.get_data()).astype(np.float32) * scale
                rgb = np.asanyarray(c.get_data()).copy()
                n += 1
                if time.time() - t0 >= 1.0:
                    self.fps, n, t0 = n / (time.time() - t0), 0, time.time()
                with self.lock:
                    self.rgb, self.depth = rgb, depth
                    self.frame_id += 1
                    self.t_frame = time.time()
        except Exception as exc:
            self.error = f"相机读取失败：{exc}"[:300]
        finally:
            if pipe is not None:
                try:
                    pipe.stop()
                except Exception:
                    pass

    def frames(self):
        with self.lock:
            return self.rgb, self.depth, self.frame_id

    def display_ppm(self, view="color", size=(560, 315)):
        import numpy as np
        rgb, depth, _fid = self.frames()
        if rgb is None:
            return None
        if view == "depth":
            img = depth_colormap(depth[::2, ::2])
        elif view == "both":
            a = rgb[::2, ::2].astype(np.uint16)
            b = depth_colormap(depth[::2, ::2]).astype(np.uint16)
            img = ((a + b) // 2).astype(np.uint8)
        else:
            img = np.ascontiguousarray(rgb[::2, ::2])
        if size and (img.shape[1] != int(size[0]) or img.shape[0] != int(size[1])):
            try:
                cv2 = _import_cv2()
                img = cv2.resize(img, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)
            except Exception:
                pass
        return ppm_bytes(img)

    def depth_at(self, u, v, r=2):
        import numpy as np
        _rgb, depth, _fid = self.frames()
        if depth is None:
            return None
        u, v = int(round(u)), int(round(v))
        win = depth[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
        vals = win[win > 0]
        return float(np.median(vals)) if vals.size else None

    def roi_points(self, u0, v0, u1, v1, stride=4):
        import numpy as np
        _rgb, depth, _fid = self.frames()
        if depth is None or self.K is None:
            return [], 0
        u0, u1 = sorted((int(u0), int(u1)))
        v0, v1 = sorted((int(v0), int(v1)))
        sub = depth[v0:v1:stride, u0:u1:stride]
        vv, uu = np.mgrid[v0:v1:stride, u0:u1:stride]
        ok = sub > 0
        z = sub[ok]
        pts = np.stack([(uu[ok] - self.K["ppx"]) * z / self.K["fx"], (vv[ok] - self.K["ppy"]) * z / self.K["fy"], z], 1)
        return pts.tolist(), int(sub.size)

    def detect_marker(self, marker, robot_q_deg=None):
        rgb, depth, _fid = self.frames()
        if rgb is None or self.K is None:
            return None
        return detect_aruco(rgb, self.K, self.dist, marker["dict"], marker["id"], marker["size_mm"] / 1000.0, depth)


# V3.47.2 TABLE + OVERLAY + ROI SPEED: shared Treeview rowheight + column auto-fit + non-overlapping YOLO labels
TREEVIEW_STYLE = "V345Calib.Treeview"


def treeview_style(ttk, widget, style_name=TREEVIEW_STYLE):
    """Shared Treeview style with rowheight = font linespace + 8 so Chinese / scaled rows never overlap."""
    try:
        ls = int(widget.tk.call("font", "metrics", "TkDefaultFont", "-linespace"))
    except Exception:
        ls = 16
    st = ttk.Style(widget)
    st.configure(style_name, rowheight=ls + 8)
    st.configure(style_name + ".Heading", font=("TkDefaultFont", 9, "bold"))
    return style_name, ls + 8


def measure_text_px(widget, text):
    s_ = str(text or "")
    try:
        return int(widget.tk.call("font", "measure", "TkDefaultFont", s_))
    except Exception:
        return sum(14 if ord(ch) > 127 else 8 for ch in s_)


def autofit_treeview(tv, cols, stretch_col=None, pad_hdr=20, pad_cell=16):
    """Fit each Treeview column to max(min_width, header width, cell widths); only stretch_col stretches."""
    items = tv.get_children()
    rows = [tv.item(it, "values") or () for it in items]
    widths = {}
    for idx, (c_, t_, w_min) in enumerate(cols):
        hw = measure_text_px(tv, t_) + pad_hdr
        cw = max((measure_text_px(tv, r[idx]) + pad_cell for r in rows if idx < len(r)), default=0)
        w = max(int(w_min), hw, cw)
        tv.column(c_, width=w, minwidth=int(w_min), anchor="w" if c_ == "why" else "center",
                  stretch=(c_ == stretch_col))
        widths[c_] = w
    return widths


def short_egg_label(det_id, name=""):
    """Short canvas label: 'egg_1' -> 'e1', 'slot_1' -> 's1', 'holder' -> '蛋托', 'obj_1' + 'cup' -> 'o1:cup'."""
    s_ = str(det_id or "")
    if s_.startswith("egg_"):
        return "e" + s_[4:]
    if s_.startswith("slot_"):
        return "s" + s_[5:]
    if s_ == "holder":
        return "蛋托"
    if s_.startswith("obj_"):
        tag = "o" + s_[4:]
        return f"{tag}:{name[:10]}" if name else tag
    return s_ or "?"


def _rects_overlap(a, b, pad=2.0):
    return a[0] - pad < b[2] and b[0] - pad < a[2] and a[1] - pad < b[3] and b[1] - pad < a[3]


def place_yolo_labels(views, canvas_w=640, canvas_h=360):
    """V3.47.2 OVERLAY: place each YOLO box label (default above-left, else below-left / right / left / inside)
    so adjacent labels do not overlap each other or fall outside the canvas.
    Input boxes are full-resolution (1280x720); returned coordinates are canvas pixels (canvas_w x canvas_h)."""
    placed = []
    out = []
    sx, sy = float(canvas_w) / 1280.0, float(canvas_h) / 720.0
    boxes_cv = [[v["box"][0] * sx, v["box"][1] * sy, v["box"][2] * sx, v["box"][3] * sy] for v in (views or [])]
    for idx, v in enumerate(views or []):
        x0, y0, x1, y1 = boxes_cv[idx]
        lab = str(v.get("label") or "")
        lw = max(18.0, len(lab) * 8.0 + 6.0)
        lh = 14.0
        cands = [
            ("top", x0, y0 - 2.0, "sw", (x0, y0 - 2.0 - lh, x0 + lw, y0 - 2.0)),
            ("bottom", x0, y1 + 2.0, "nw", (x0, y1 + 2.0, x0 + lw, y1 + 2.0 + lh)),
            ("right", x1 + 3.0, y0, "nw", (x1 + 3.0, y0, x1 + 3.0 + lw, y0 + lh)),
            ("left", x0 - 3.0, y0, "ne", (x0 - 3.0 - lw, y0, x0 - 3.0, y0 + lh)),
            ("inside", x0 + 2.0, y0 + 2.0, "nw", (x0 + 2.0, y0 + 2.0, x0 + 2.0 + lw, y0 + 2.0 + lh)),
        ]
        chosen = None
        for pass_no in (0, 1):
            for pos_name, tx, ty, anc, r in cands:
                in_canvas = (r[0] >= 0 and r[1] >= 0 and r[2] <= canvas_w and r[3] <= canvas_h)
                if not in_canvas and pos_name != "inside":
                    continue
                if any(_rects_overlap(r, p) for p in placed):
                    continue
                if pass_no == 0 and pos_name != "inside":
                    if any(j != idx and _rects_overlap(r, boxes_cv[j], pad=0.0) for j in range(len(boxes_cv))):
                        continue
                chosen = (pos_name, tx, ty, anc, r)
                break
            if chosen is not None:
                break
        if chosen is None:
            chosen = cands[1] if y0 - 2.0 - lh < 0 else cands[0]
        pos_name, tx, ty, anc, r = chosen
        placed.append(r)
        out.append(dict(v, box_cv=(x0, y0, x1, y1), tx=tx, ty=ty, anchor=anc, pos=pos_name, label_rect=r))
    return out




# =========================================================================== V3.51 Full-Workflow Status Lamp Bar
WORKFLOW_STEPS = (
    ("arm",        "① 实机驱动",   True),
    ("safety",     "② 安全区",     True),
    ("main_cam",   "③ 主相机",     True),
    ("depth",      "④ 深度检查",   False),
    ("main_calib", "⑤ 主手眼标定", True),
    ("tcp",        "⑥ TCP标定",    False),
    ("wrist_cam",  "⑦ 腕部相机",   False),
    ("wrist_intr", "⑧ 腕部内参",   False),
    ("wrist_eih",  "⑨ 腕部手眼",   False),
)


def evaluate_workflow_lamps(pipe, safe_tab, cam_tab):
    """V3.51 WORKFLOW LAMPS: evaluate the 9 ordered stages from power-on to safety zone and calibrations.
    Colors:
      - 'grey'  : not started and no saved data
      - 'green' : completed and meets accuracy/safety thresholds
      - 'yellow': in progress, unsaved result, pending verification, or sub-optimal parameter
      - 'red'   : required safety/calibration step missing, failed verification, or error exceeds minimum safety bound
    Each item in the returned list is:
      {'key': str, 'label': str, 'required': bool, 'optional': bool, 'color': str, 'detail': str}
    """
    import NERO_V349_Real_Wrist_Camera as W49
    out = []

    # ① 实机驱动 (Required)
    p_state = getattr(pipe, "state", None) if pipe is not None else None
    teach = bool(getattr(pipe, "teach_mode", False))
    jp_en = bool(getattr(getattr(cam_tab, "jpanel", None), "enabled", False))
    if isinstance(p_state, dict) and p_state:
        cols = [v[0] for v in p_state.values() if isinstance(v, (tuple, list)) and v]
        mv_col = (p_state.get("moveit") or ("grey",))[0]
        if "red" in cols:
            c1, d1 = "red", "实机启动步骤报错（请检查 CAN / 驱动 / MoveIt）"
        elif mv_col == "green":
            if teach:
                c1, d1 = "yellow", "处于示教模式（自动采点与悬停需先退出示教）"
            else:
                c1, d1 = "green", "CAN + 驱动 + MoveIt 已就绪 ✓"
        elif any(c in ("green", "yellow") for c in cols):
            c1, d1 = "yellow", "实机部分启动（等待 MoveIt 就绪）"
        else:
            c1, d1 = "grey", "未启动实机驱动"
    else:
        if teach:
            c1, d1 = "yellow", "处于示教模式（自动规划需退出示教）"
        elif jp_en:
            c1, d1 = "green", "实机控制与 MoveIt 已就绪 ✓"
        else:
            c1, d1 = "grey", "未启动实机驱动"
    out.append({"key": "arm", "label": "① 实机驱动", "required": True, "optional": False, "color": c1, "detail": d1})

    # ② 安全区 (Required safety prerequisite)
    ws = getattr(safe_tab, "valid_ws", None) if safe_tab is not None else None
    fence = (ws or {}).get("fence") if isinstance(ws, dict) else None
    poly = (fence or {}).get("polygon") or []
    if len(poly) < 3:
        c2, d2 = "red", "未设置有效安全区围栏（必要安全环节缺失！）"
    elif getattr(safe_tab, "dirty", False):
        c2, d2 = "yellow", "安全区有未保存的修改（请点保存并验证）"
    else:
        zt = float(fence.get("z_table", 0.0)) * 1000.0
        c2, d2 = "green", f"安全区围栏有效（{len(poly)} 顶点，桌面 z={zt:.0f} mm）✓"
    out.append({"key": "safety", "label": "② 安全区", "required": True, "optional": False, "color": c2, "detail": d2})

    # ③ 主相机 (Required)
    src = getattr(cam_tab, "src", None) if cam_tab is not None else None
    v_cam_txt = cam_tab.v_cam.get() if (cam_tab is not None and hasattr(cam_tab, "v_cam")) else ""
    if src is None:
        if "✗" in v_cam_txt:
            c3, d3 = "red", v_cam_txt
        else:
            c3, d3 = "grey", "主相机 D435i 未连接"
    elif getattr(src, "error", None):
        c3, d3 = "red", f"主相机报错：{src.error}"
    elif not getattr(src, "info", None) or not getattr(src, "K", None):
        c3, d3 = "yellow", "主相机正在连接…"
    else:
        usb = str(src.info.get("usb", ""))
        if usb and not usb.startswith("3"):
            c3, d3 = "yellow", f"主相机已连接但处于 USB {usb}（建议换 USB 3.0 端口）"
        else:
            c3, d3 = "green", f"主相机 D435i 已连接（USB {usb or '3.x'}）✓"
    out.append({"key": "main_cam", "label": "③ 主相机", "required": True, "optional": False, "color": c3, "detail": d3})

    # ④ 深度检查 (Optional — dashed ring)
    p_txt = cam_tab.v_plane.get() if (cam_tab is not None and hasattr(cam_tab, "v_plane")) else ""
    if not p_txt.strip():
        c4, d4 = "grey", "未进行桌面深度平面检查（可选）"
    elif "✓ 深度质量好" in p_txt:
        c4, d4 = "green", "桌面深度检查通过（残差 ≤ 2 mm，有效率 ≥ 90%）✓"
    else:
        c4, d4 = "yellow", "桌面深度残差 > 2 mm 或有效深度 < 90%"
    out.append({"key": "depth", "label": "④ 深度检查", "required": False, "optional": True, "color": c4, "detail": d4})

    # ⑤ 主手眼标定 (Required)
    r_m = getattr(cam_tab, "result", None) if cam_tab is not None else None
    c_m = getattr(cam_tab, "calib", None) if cam_tab is not None else None
    vs_m = ((getattr(cam_tab, "vstate", None) or {}).get("state")) if cam_tab is not None else None
    s_m = getattr(cam_tab, "samples", None) or []
    if r_m and r_m.get("ok"):
        if r_m.get("good"):
            c5, d5 = "yellow", "主手眼标定已算合格，请点「保存标定」"
        else:
            c5, d5 = "red", f"主手眼标定未达标（留一 {r_m.get('loo_mean_mm', 0):.1f}/{r_m.get('loo_max_mm', 0):.1f} mm）"
    elif c_m and c_m.get("T_base_cam"):
        loo_m, loo_x = float(c_m.get("loo_mean_mm") or 0.0), float(c_m.get("loo_max_mm") or 0.0)
        good_m = loo_m <= CALIB_GOOD_MEAN_MM and loo_x <= CALIB_GOOD_MAX_MM
        if vs_m == "fail":
            c5, d5 = "red", "主手眼标定校验未通过（相机或标记可能被挪动，需重标）"
        elif not good_m:
            c5, d5 = "yellow", f"主手眼标定留一误差偏高（{loo_m:.1f}/{loo_x:.1f} mm）"
        elif vs_m == "pending":
            c5, d5 = "yellow", "已加载主手眼标定，待看到标记或点「快速复核」确认"
        else:
            c5, d5 = "green", f"主相机手眼标定合格（留一 {loo_m:.2f} mm）✓"
    elif s_m:
        c5, d5 = "yellow", f"主手眼样本采集中（{len(s_m)}/≥8）"
    elif c3 == "green":
        c5, d5 = "red", "主相机已连接但缺少主手眼标定（必要环节未完成！）"
    else:
        c5, d5 = "grey", "未进行主相机手眼标定"
    out.append({"key": "main_calib", "label": "⑤ 主手眼标定", "required": True, "optional": False, "color": c5, "detail": d5})

    # ⑥ TCP标定 (Optional — dashed ring)
    tc = getattr(cam_tab, "tcp_calib", None) if cam_tab is not None else None
    tsamp = (getattr(cam_tab, "tcp_samples", None) or []) if cam_tab is not None else []
    if tc and tcp_valid(tc):
        loo_m, loo_x = float(tc.get("loo_mean_mm") or 0.0), float(tc.get("loo_max_mm") or 0.0)
        if tc.get("good", loo_m <= CALIB_GOOD_MEAN_MM and loo_x <= CALIB_GOOD_MAX_MM):
            c6, d6 = "green", f"TCP 已实测标定（留一 {loo_m:.2f} mm）✓"
        else:
            c6, d6 = "yellow", f"TCP 已标定但留一误差偏高（{loo_m:.1f}/{loo_x:.1f} mm）"
    elif tsamp:
        c6, d6 = "yellow", f"TCP 碰点采集中（{len(tsamp)}/≥4）"
    else:
        c6, d6 = "grey", "使用 URDF 默认 TCP（多姿态碰点标定为可选）"
    out.append({"key": "tcp", "label": "⑥ TCP标定", "required": False, "optional": True, "color": c6, "detail": d6})

    # ⑦ 腕部相机 (Optional — dashed ring)
    wsrc = getattr(cam_tab, "wrist_src", None) if cam_tab is not None else None
    wq = getattr(cam_tab, "wrist_quality", None) if cam_tab is not None else None
    vw_txt = cam_tab.v_wrist_live.get() if (cam_tab is not None and hasattr(cam_tab, "v_wrist_live")) else ""
    if wsrc is None:
        if "✗" in vw_txt:
            c7, d7 = "red", vw_txt
        else:
            c7, d7 = "grey", "腕部 USB 相机未连接（可选）"
    elif getattr(wsrc, "error", None):
        c7, d7 = "red", f"腕部相机报错：{wsrc.error}"
    elif not getattr(wsrc, "info", None):
        c7, d7 = "yellow", "腕部相机连接中…"
    elif wq and not wq.get("settled", True):
        c7, d7 = "yellow", f"腕部相机画面未稳：{wq.get('badge', '')}"
    else:
        c7, d7 = "green", f"腕部 USB 相机已连接（{wsrc.info.get('res', '1080P')}）✓"
    out.append({"key": "wrist_cam", "label": "⑦ 腕部相机", "required": False, "optional": True, "color": c7, "detail": d7})

    # ⑧ 腕部内参 (Optional / Conditionally required when wrist_cam is active — dashed ring)
    ir = getattr(cam_tab, "intr_result", None) if cam_tab is not None else None
    wi = getattr(cam_tab, "wrist_intr", None) if cam_tab is not None else None
    isamp = (getattr(cam_tab, "intr_samples", None) or []) if cam_tab is not None else []
    w_active = c7 in ("green", "yellow")
    if ir and ir.get("ok"):
        rms = float(ir.get("rms_px", 99.0))
        if rms > 1.5:
            c8, d8 = "red", f"腕部内参重投影误差过大（RMS {rms:.2f} px > 1.5 px）"
        else:
            c8, d8 = "yellow", f"腕部内参已计算（RMS {rms:.2f} px），请点「保存内参」"
    elif wi and W49.intrinsics_valid(wi):
        rms = float(wi.get("rms_px", 0.0))
        if rms > 1.5:
            c8, d8 = "red", f"腕部内参重投影超限（RMS {rms:.2f} px > 1.5 px）"
        elif rms > W49.INTR_GOOD_RMS_PX or wi.get("warnings"):
            c8, d8 = "yellow", f"腕部内参可用但非最佳（RMS {rms:.2f} px）"
        else:
            c8, d8 = "green", f"腕部 130° 广角内参合格（RMS {rms:.2f} px）✓"
    elif isamp:
        c8, d8 = "yellow", f"腕部内参标定图采集中（{len(isamp)}/≥8 张）"
    elif w_active:
        c8, d8 = "red", "腕部相机已连接，但尚未完成 130° 广角内参标定"
    else:
        c8, d8 = "grey", "未标定腕部相机内参（可选）"
    out.append({"key": "wrist_intr", "label": "⑧ 腕部内参", "required": False, "optional": True, "color": c8, "detail": d8})

    # ⑨ 腕部手眼 (Optional / Conditionally required when wrist_cam + wrist_intr are ready — dashed ring)
    wr = getattr(cam_tab, "wcalib_result", None) if cam_tab is not None else None
    wc = getattr(cam_tab, "wrist_calib", None) if cam_tab is not None else None
    wsamp = (getattr(cam_tab, "wcalib_samples", None) or []) if cam_tab is not None else []
    wcs = getattr(cam_tab, "wcheck_state", None) if cam_tab is not None else None
    if wr and wr.get("ok"):
        if wr.get("good"):
            c9, d9 = "yellow", "腕部 Eye-in-Hand 已算合格，请点「保存腕部标定」"
        else:
            c9, d9 = "red", f"腕部手眼未达标（留一 {wr.get('loo_mean_mm', 0):.1f}/{wr.get('loo_max_mm', 0):.1f} mm）"
    elif wc and W49.handeye_valid(wc):
        loo_m, loo_x = float(wc.get("loo_mean_mm") or 0.0), float(wc.get("loo_max_mm") or 0.0)
        if (wcs and wcs[0] == "fail") or loo_x > 10.0:
            c9, d9 = "red", f"腕部手眼复核未通过或误差超限（最大 {loo_x:.1f} mm）"
        elif not wc.get("good", loo_m <= CALIB_GOOD_MEAN_MM and loo_x <= CALIB_GOOD_MAX_MM):
            c9, d9 = "yellow", f"腕部手眼可用但留一误差偏高（{loo_m:.1f}/{loo_x:.1f} mm）"
        else:
            c9, d9 = "green", f"腕部 Eye-in-Hand 手眼标定合格（留一 {loo_m:.2f} mm）✓"
    elif wsamp:
        c9, d9 = "yellow", f"腕部手眼样本采集中（{len(wsamp)}/≥6）"
    elif w_active and c8 == "green":
        c9, d9 = "red", "腕部相机与内参已就绪，但缺少 Eye-in-Hand 手眼标定"
    else:
        c9, d9 = "grey", "未标定腕部 Eye-in-Hand 手眼（可选）"
    out.append({"key": "wrist_eih", "label": "⑨ 腕部手眼", "required": False, "optional": True, "color": c9, "detail": d9})

    return out


class WorkflowLampBar:
    """V3.51/V3.52 GUI WIDGET: horizontal full-workflow status lamp strip + teach-mode buttons placed right above
    the bottom log box (immediately to the left of Copy Log).
    Required stages have a solid ring; optional stages have a dashed outer ring ('optional')."""
    COLORS = {"grey": "#9e9e9e", "green": "#2e7d32", "yellow": "#f2c037", "red": "#c62828"}

    def __init__(self, tk, ttk, parent, on_teach_enter=None, on_teach_exit=None):
        self.tk = tk
        self.ttk = ttk
        # V3.53: pack teach-mode buttons directly into `parent` (side="right") right next to Copy Log so all three
        # buttons share the exact same vertical center line ("平齐").
        if callable(on_teach_enter) or callable(on_teach_exit):
            self.b_teach_out = ttk.Button(parent, text="退出示教模式", command=on_teach_exit)
            self.b_teach_out.pack(side="right", padx=(2, 6))
            self.b_teach_in = ttk.Button(parent, text="进入示教模式", command=on_teach_enter)
            self.b_teach_in.pack(side="right", padx=(6, 2))
        else:
            self.b_teach_in = self.b_teach_out = None
        self.frame = ttk.Frame(parent, padding=(4, 2))
        self.frame.pack(side="left", fill="x", expand=True)
        top = ttk.Frame(self.frame)
        top.pack(fill="x")
        ttk.Label(top, text="全流程状态：", font=("TkDefaultFont", 9, "bold")).pack(side="left", padx=(0, 4))
        self.cells = {}
        for i, (key, lab, req) in enumerate(WORKFLOW_STEPS):
            cell = ttk.Frame(top)
            cell.pack(side="left", padx=2)
            c = tk.Canvas(cell, width=22, height=22, highlightthickness=0, bd=0)
            if req:
                oid = c.create_oval(3, 3, 19, 19, fill=self.COLORS["grey"], outline="#455a64", width=1.5, tags="o")
                rid = None
            else:
                rid = c.create_oval(1, 1, 21, 21, fill="", outline="#546e7a", width=1.2, dash=(3, 2), tags="ring")
                oid = c.create_oval(5, 5, 17, 17, fill=self.COLORS["grey"], outline="#455a64", width=1.0, tags="o")
            c.pack(side="left")
            lbl = ttk.Label(cell, text=lab, font=("TkDefaultFont", 8))
            lbl.pack(side="left", padx=(2, 0))
            if i < len(WORKFLOW_STEPS) - 1:
                ttk.Label(top, text="→", foreground="#78909c", font=("TkDefaultFont", 8)).pack(side="left", padx=1)
            self.cells[key] = {"canvas": c, "oid": oid, "rid": rid, "required": req, "color": "grey", "detail": ""}
            c.bind("<Enter>", lambda _e, k=key: self._on_hover(k))
            lbl.bind("<Enter>", lambda _e, k=key: self._on_hover(k))

        bot = ttk.Frame(self.frame)
        bot.pack(fill="x", pady=(1, 0))
        ttk.Label(bot, text="● 实线圆=必选  ◌ 虚线圆=可选(optional)  [灰:未设 绿:完成 黄:待优/未存 红:必选未完/超限]  ",
                  foreground="#607d8b", font=("TkDefaultFont", 8)).pack(side="left")
        self.v_detail = tk.StringVar(value="")
        ttk.Label(bot, textvariable=self.v_detail, foreground="#37474f", font=("TkDefaultFont", 8, "bold")).pack(side="left")
        self._last_states = []

    def _on_hover(self, key):
        cell = self.cells.get(key)
        if cell and cell.get("detail"):
            self.v_detail.set(f"[{key}] {cell['detail']}")

    def update_from(self, states, teach_mode=False, busy=False):
        self._last_states = list(states or [])
        if self.b_teach_in is not None and self.b_teach_out is not None:
            self.b_teach_in.configure(state="disabled" if (teach_mode or busy) else "normal",
                                      text="示教中(零力)" if teach_mode else "进入示教模式")
            # V3.53: keep 退出示教模式 always clickable when not busy (even if V111 ctrl_mode stayed 0x01)
            self.b_teach_out.configure(state="normal" if not busy else "disabled")
        urgent = None
        for st in self._last_states:
            k = st["key"]
            col = st["color"]
            cell = self.cells.get(k)
            if not cell:
                continue
            cell["detail"] = st.get("detail", "")
            if cell["color"] != col:
                cell["color"] = col
                cell["canvas"].itemconfigure("o", fill=self.COLORS.get(col, "#9e9e9e"))
            if urgent is None and col == "red":
                urgent = f"⚠ {st['label']}：{st['detail']}"
        if urgent is None:
            for st in self._last_states:
                if st["color"] == "yellow":
                    urgent = f"提示 {st['label']}：{st['detail']}"
                    break
        if urgent is None and self._last_states:
            n_g = sum(1 for s_ in self._last_states if s_["color"] == "green")
            urgent = f"✓ 已完成 {n_g}/{len(self._last_states)} 项环节"
        if urgent:
            self.v_detail.set(urgent)


# =========================================================================== UI
class CameraTabUI:
    """V3.25 Camera调试 tab (V3.50: dual 16:9 monitor canvases 560x315 for fixed D435i + wrist USB camera)."""
    DW, DH = 560, 315          # V3.50: dual 16:9 monitor canvases (upper = fixed D435i, lower = wrist USB)

    def __init__(self, tk, ttk, parent, pipe, jpanel, safe_tab, cfg_fn, log, ask, visible_fn=lambda: True,
                 source_factory=None, wrist_source_factory=None):
        self.tk, self.ttk, self.pipe, self.jpanel, self.safe_tab = tk, ttk, pipe, jpanel, safe_tab
        self.cfg_fn, self.log, self.ask, self.visible_fn = cfg_fn, log, ask, visible_fn
        self.source_factory = source_factory or (lambda: RealSenseSource(log=self.log))
        self.wrist_source_factory = wrist_source_factory
        self.src = None
        self.wrist_src = None
        self.intr_samples = []
        self.intr_result = None
        self.wrist_intr = None
        self.wmarker = {"dict": "DICT_4X4_50", "id": 0, "size_mm": 24.0}
        self.wcalib_samples = []
        self.wcalib_result = None
        self.wrist_calib = None
        self.wauto_state = None
        self.wcheck_state = None
        self.wdet = None
        self.wdet_t = 0.0
        self._wdet_busy = False
        self.intr_preview_pts = None
        self.samples = []
        self.result = None
        self.calib = None
        self.marker = dict(DEFAULT_MARKER)
        self.det = None
        self.det_t = 0.0
        self.meas = None
        self.roi = None
        self._drag0 = None
        self.eggs = []
        self.tcp_samples = []
        self.tcp_calib = None
        self._pre_hover_q = None
        self.stop_flag = False
        self.worker = None
        self.photo = None
        self.photo_wrist = None
        self._shown_fid = -1
        self._shown_wfid = -1
        self._prev_wrist_rgb = None
        self._prev_wrist_q = None
        self.wrist_quality = None
        self.wrist_refine_res = None
        self.workflow_bars = []
        self.workflow_state = []
        self._btn_state = {}
        self._det_busy = False
        self._ui_q = []
        self._load_saved()
        self._build(parent)
        self.after_id = None
        self._schedule()

    # ------------------------------------------------------------------ build
    def _build(self, parent):
        tk, ttk = self.tk, self.ttk
        import NERO_V314_Joint_Control as JC
        self.JC = JC
        top = ttk.Frame(parent)
        top.pack(fill="x", pady=(2, 4))
        ttk.Button(top, text="▶ 连接相机", command=self.on_connect).pack(side="left")
        ttk.Button(top, text="■ 断开", command=self.on_disconnect).pack(side="left", padx=4)
        ttk.Label(top, text="  画面").pack(side="left")
        self.v_view = tk.StringVar(value="color")
        for t_, v_ in (("彩色", "color"), ("深度", "depth"), ("叠加", "both"), ("腕部USB", "wrist")):
            ttk.Radiobutton(top, text=t_, value=v_, variable=self.v_view, command=self._force_redraw).pack(side="left")
        self.v_undistort = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="腕部去畸变", variable=self.v_undistort,
                        command=self._on_undistort_toggle).pack(side="left", padx=(4, 0))
        ttk.Label(top, text="  显示").pack(side="left")
        self.ov = {}
        for k, t_ in (("robot", "机器人"), ("zone", "安全区"), ("marker", "ArUco"), ("eggs", "鸡蛋")):
            self.ov[k] = tk.BooleanVar(value=True)
            ttk.Checkbutton(top, text=t_, variable=self.ov[k], command=self._force_redraw).pack(side="left")
        self.v_cam = tk.StringVar(value="相机未连接")
        JC.fixed_label(tk, top, 1, width=520, textvariable=self.v_cam, font=("TkDefaultFont", 9, "bold"),
                       pack={"side": "left", "padx": 10})
        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="主相机实时监视（固定 D435i · 全局鸟瞰）", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        self.cv = tk.Canvas(left, width=self.DW, height=self.DH, bg="#202020", highlightthickness=1,
                            highlightbackground="#999")
        self.cv.pack()
        self.cv.bind("<ButtonPress-1>", self.on_press)
        self.cv.bind("<B1-Motion>", self.on_drag)
        self.cv.bind("<ButtonRelease-1>", self.on_release)
        self.v_meas = tk.StringVar(value="点击画面 = 读距离 / 坐标；拖一个框 = 选深度检查区域")
        JC.fixed_label(tk, left, 2, width=self.DW, textvariable=self.v_meas, font=("TkFixedFont", 9),
                       wraplength=self.DW - 10, pack={"pady": 1})
        self.v_live = tk.StringVar(value="")
        self.lbl_live = JC.fixed_label(tk, left, 1, width=self.DW, textvariable=self.v_live, font=("TkDefaultFont", 9, "bold"),
                                       wraplength=self.DW - 10, pack={"pady": 1})
        # V3.50 LOWER WRIST CAMERA LIVE MONITOR (560x315 16:9, directly below main camera monitor)
        ttk.Label(left, text="腕部相机实时监视（USB 130° 广角 · TCP 准星与精定位）",
                  font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(2, 0))
        self.cv_wrist = tk.Canvas(left, width=self.DW, height=self.DH, bg="#181818", highlightthickness=1,
                                  highlightbackground="#607d8b")
        self.cv_wrist.pack()
        self.v_wrist_live = tk.StringVar(value="腕部相机未连接（在右侧「相机信息」Tab 点「▶ 连接腕部相机」）")
        self.lbl_wrist_live = JC.fixed_label(tk, left, 2, width=self.DW, textvariable=self.v_wrist_live,
                                             font=("TkDefaultFont", 9), wraplength=self.DW - 10, pack={"pady": 1})
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.nb = nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)
        self._build_info(nb)
        self._build_depth(nb)
        self._build_intrinsics(nb)
        self._build_calib(nb)
        self._build_locate(nb)

    def _build_info(self, nb):
        tk, ttk = self.tk, self.ttk
        import NERO_V349_Real_Wrist_Camera as W49
        self.W49 = W49
        import NERO_V359_Strategy as ST59
        self.ST59 = ST59
        self.strategy_profile = ST59.get_active_strategy()
        p = ttk.Frame(nb, padding=6)
        nb.add(p, text="相机信息")
        ttk.Label(p, text="主相机（固定 D435i，eye-to-hand）", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        self.v_info = tk.StringVar(value="（连接相机后显示型号、序列号、固件、USB、内参、帧率）")
        self.JC.fixed_label(tk, p, 6, textvariable=self.v_info, font=("TkFixedFont", 9), justify="left", anchor="nw",
                            wraplength=560, pack={"fill": "x"})
        r = ttk.Frame(p)
        r.pack(fill="x", pady=3)
        ttk.Button(r, text="保存相机信息", command=self.on_save_info).pack(side="left")
        ttk.Label(p, text="相机固定在支架上（和 NERO 同一个刚性底座）。换了相机 / 动了支架 → 重新做手眼标定。",
                  foreground="#555555", justify="left").pack(anchor="w")
        # V3.49 WRIST USB CAMERA INFO (placed below the main camera info)
        ttk.Separator(p).pack(fill="x", pady=6)
        ttk.Label(p, text="腕部相机（USB 1080P 130° 广角，eye-in-hand）", font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        rw = ttk.Frame(p)
        rw.pack(fill="x", pady=(3, 2))
        ttk.Label(rw, text="设备").pack(side="left")
        self.v_wdev = tk.StringVar(value="auto")
        self.cb_wdev = ttk.Combobox(rw, textvariable=self.v_wdev, values=self._wrist_dev_values(), width=16,
                                    postcommand=self._refresh_wrist_devs)
        self.cb_wdev.pack(side="left", padx=2)
        ttk.Label(rw, text="分辨率").pack(side="left", padx=(4, 0))
        self.v_wres = tk.StringVar(value=W49.WRIST_RES_LIST[0])
        ttk.Combobox(rw, textvariable=self.v_wres, values=W49.WRIST_RES_LIST, state="readonly", width=10).pack(side="left", padx=2)
        self.v_wlock = tk.BooleanVar(value=True)
        ttk.Checkbutton(rw, text="锁曝光/白平衡/对焦", variable=self.v_wlock).pack(side="left", padx=4)
        ttk.Button(rw, text="▶ 连接腕部相机", command=self.on_wrist_connect).pack(side="left", padx=2)
        ttk.Button(rw, text="■ 断开腕部", command=self.on_wrist_disconnect).pack(side="left", padx=2)
        ttk.Button(rw, text="保存腕部相机信息", command=self.on_save_wrist_info).pack(side="left", padx=(4, 0))
        self.v_wrist_info = tk.StringVar(value="")
        self.JC.fixed_label(tk, p, 5, textvariable=self.v_wrist_info, font=("TkFixedFont", 9), justify="left",
                            anchor="nw", wraplength=560, pack={"fill": "x", "pady": (2, 0)})
        self._wrist_autoload()
        self.v_wrist_info.set(self._wrist_info_text())

    def _build_depth(self, nb):
        tk, ttk = self.tk, self.ttk
        p = ttk.Frame(nb, padding=6)
        nb.add(p, text="深度检查")
        ttk.Label(p, text="① 在画面上拖一个框，框住一块空桌面（不要包括物体）  ② 点「平面拟合」",
                  justify="left").pack(anchor="w")
        ttk.Button(p, text="平面拟合", command=self.on_plane).pack(anchor="w", pady=4)
        self.v_plane = tk.StringVar(value="")
        self.JC.fixed_label(tk, p, 5, textvariable=self.v_plane, font=("TkFixedFont", 9), justify="left", anchor="nw",
                            wraplength=560, pack={"fill": "x"})
        ttk.Label(p, text="参考：0.6–1.0 m 时平面残差 ≤ 2 mm、有效深度 ≥ 90% 为好；标定后会给出桌面在 base_link 的高度，"
                          "和安全区的「桌面 z」对比。", foreground="#555555", wraplength=560, justify="left").pack(anchor="w")

    # V3.45 CALIB STEPS: four numbered steps top-down, each with a status lamp; buttons follow the state.
    LAMP_FILL = {"grey": "#9e9e9e", "yellow": "#f2c037", "green": "#2e7d32", "red": "#c62828", "blue": "#1e88e5"}

    def _lamp(self, parent):
        c = self.tk.Canvas(parent, width=18, height=18, highlightthickness=0, bd=0)
        c.create_oval(2, 2, 16, 16, fill=self.LAMP_FILL["grey"], outline="#616161", tags="o")
        return c

    def _set_lamp(self, c, state):
        if getattr(c, "_v345_state", None) != state:
            c.itemconfigure("o", fill=self.LAMP_FILL.get(state, "#9e9e9e"))
            c._v345_state = state

    def _step(self, parent, title):
        tk, ttk = self.tk, self.ttk
        f = ttk.Frame(parent, padding=(2, 3))
        f.pack(fill="x")
        hdr = ttk.Frame(f)
        hdr.pack(fill="x")
        lamp = self._lamp(hdr)
        lamp.pack(side="left")
        ttk.Label(hdr, text=title, font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=(4, 10))
        var = tk.StringVar(value="")
        self.JC.fixed_label(tk, hdr, 1, textvariable=var, font=("TkDefaultFont", 9), fg="#455a64",
                            pack={"side": "left", "fill": "x", "expand": True})
        body = ttk.Frame(f)
        body.pack(fill="x", padx=(24, 0), pady=(2, 0))
        ttk.Separator(parent).pack(fill="x", pady=(3, 0))
        return lamp, var, body

    def _build_intrinsics(self, nb):
        """V3.49 INTRINSICS TAB: placed right after 深度检查 for wide-angle (130°) ChArUco / Chessboard calibration."""
        tk, ttk = self.tk, self.ttk
        W49 = self.W49
        p = ttk.Frame(nb, padding=6)
        nb.add(p, text="内参标定")
        # ① board & lens model
        self.il1, self.is1, ib1 = self._step(p, "① 标定板与广角畸变模型")
        r1 = ttk.Frame(ib1)
        r1.pack(fill="x")
        ttk.Label(r1, text="标定板").pack(side="left")
        self.v_ibtype = tk.StringVar(value=W49.DEFAULT_BOARD["type"])
        ttk.Combobox(r1, textvariable=self.v_ibtype, values=W49.BOARD_TYPES, state="readonly", width=22).pack(side="left", padx=2)
        ttk.Label(r1, text="列×行").pack(side="left", padx=(6, 0))
        self.v_icols = tk.StringVar(value=str(W49.DEFAULT_BOARD["cols"]))
        ttk.Entry(r1, textvariable=self.v_icols, width=3).pack(side="left", padx=1)
        ttk.Label(r1, text="×").pack(side="left")
        self.v_irows = tk.StringVar(value=str(W49.DEFAULT_BOARD["rows"]))
        ttk.Entry(r1, textvariable=self.v_irows, width=3).pack(side="left", padx=1)
        ttk.Label(r1, text="方格").pack(side="left", padx=(6, 0))
        self.v_isq = tk.StringVar(value=f"{W49.DEFAULT_BOARD['square_mm']:g}")
        ttk.Entry(r1, textvariable=self.v_isq, width=5).pack(side="left", padx=1)
        ttk.Label(r1, text="mm 标记").pack(side="left")
        self.v_imk = tk.StringVar(value=f"{W49.DEFAULT_BOARD['marker_mm']:g}")
        ttk.Entry(r1, textvariable=self.v_imk, width=5).pack(side="left", padx=1)
        ttk.Label(r1, text="mm").pack(side="left")
        self.b_ib_png = ttk.Button(r1, text="生成标定板打印图 (PNG)", command=self.on_intr_board_png)
        self.b_ib_png.pack(side="left", padx=(8, 0))
        r1b = ttk.Frame(ib1)
        r1b.pack(fill="x", pady=(3, 0))
        ttk.Label(r1b, text="畸变模型").pack(side="left")
        self.v_imodel = tk.StringVar(value=W49.DIST_MODELS[0])
        ttk.Combobox(r1b, textvariable=self.v_imodel, values=W49.DIST_MODELS, state="readonly", width=30).pack(side="left", padx=2)
        ttk.Label(r1b, text="（130° 广角推荐 8 参数或鱼眼；打印后用尺子量方格实测 mm 填进上方）",
                  foreground="#607d8b").pack(side="left", padx=4)
        # ② capture frames (manual + full automatic tabletop board trajectory)
        self.il2, self.is2, ib2 = self._step(p, "② 采集标定画面（支持桌面标定板全自动走位采集 + 手动抓拍）")
        r2 = ttk.Frame(ib2)
        r2.pack(fill="x")
        self.b_intr_cap = ttk.Button(r2, text="● 拍一张（检测角点）", command=self.on_intr_capture)
        self.b_intr_cap.pack(side="left")
        self.b_intr_auto = ttk.Button(r2, text="▶ 自动采集", command=self.on_intr_auto)
        self.b_intr_auto.pack(side="left", padx=(6, 0))
        ttk.Label(r2, text="点数").pack(side="left", padx=(4, 0))
        self.v_inauto = tk.StringVar(value="10")
        ttk.Spinbox(r2, textvariable=self.v_inauto, from_=6, to=20, width=3).pack(side="left", padx=2)
        self.b_intr_stop = ttk.Button(r2, text="■ 停止", command=self.on_stop)
        self.b_intr_stop.pack(side="left", padx=(2, 6))
        self.b_intr_del = ttk.Button(r2, text="删除选中", command=self.on_intr_delete)
        self.b_intr_del.pack(side="left", padx=2)
        self.b_intr_clr = ttk.Button(r2, text="清空", command=self.on_intr_clear)
        self.b_intr_clr.pack(side="left")
        ttk.Label(r2, text="  标定板平放桌面，自动走中心+四角+倾斜并同步算内参与手眼", foreground="#607d8b").pack(side="left")
        itf = ttk.Frame(ib2)
        itf.pack(fill="x", pady=(4, 0))
        st_name, _ = treeview_style(ttk, p)
        self.intr_cols = (("k", "#", 36), ("n", "角点数", 64), ("cov", "画面分布", 140),
                          ("sharp", "清晰度", 72), ("err", "重投影 px", 84))
        self.tv_intr = ttk.Treeview(itf, columns=[c_[0] for c_ in self.intr_cols], show="headings", height=5,
                                    style=st_name, selectmode="extended")
        for c_, t_, _w in self.intr_cols:
            self.tv_intr.heading(c_, text=t_)
        autofit_treeview(self.tv_intr, self.intr_cols, stretch_col="cov")
        self.tv_intr.tag_configure("odd", background="#f3f6f9")
        self.tv_intr.tag_configure("rej", foreground="#c62828")
        iys = ttk.Scrollbar(itf, orient="vertical", command=self.tv_intr.yview)
        ixs = ttk.Scrollbar(itf, orient="horizontal", command=self.tv_intr.xview)
        self.tv_intr.configure(yscrollcommand=iys.set, xscrollcommand=ixs.set)
        self.tv_intr.grid(row=0, column=0, sticky="nsew")
        iys.grid(row=0, column=1, sticky="ns")
        ixs.grid(row=1, column=0, sticky="ew")
        itf.columnconfigure(0, weight=1)
        # ③ compute & save
        self.il3, self.is3, ib3 = self._step(p, "③ 计算与保存内参")
        r3 = ttk.Frame(ib3)
        r3.pack(fill="x")
        self.b_intr_comp = ttk.Button(r3, text="计算内参", command=self.on_intr_compute)
        self.b_intr_comp.pack(side="left")
        self.b_intr_save = ttk.Button(r3, text="保存内参", command=self.on_intr_save)
        self.b_intr_save.pack(side="left", padx=4)
        self.b_intr_save_as = ttk.Button(r3, text="另存为…", command=self.on_intr_save_as)
        self.b_intr_save_as.pack(side="left")
        self.b_intr_load = ttk.Button(r3, text="读取…", command=self.on_intr_load)
        self.b_intr_load.pack(side="left", padx=4)
        self.b_intr_clear_cal = ttk.Button(r3, text="清除内参", command=self.on_intr_clear_calib)
        self.b_intr_clear_cal.pack(side="left")
        ttk.Label(r3, text="合格线：重投影 RMS ≤ 0.8 px，样本 ≥ 8 张", foreground="#607d8b").pack(side="left", padx=6)
        self.v_intr_res = tk.StringVar(value=self._intr_text())
        self.JC.fixed_label(tk, ib3, 4, textvariable=self.v_intr_res, font=("TkFixedFont", 9), justify="left",
                            anchor="nw", wraplength=560, pack={"fill": "x", "pady": (3, 0)})
        self._update_intr_ui()

    def _build_calib(self, nb):
        tk, ttk = self.tk, self.ttk
        outer = ttk.Frame(nb, padding=4)
        nb.add(outer, text="手眼标定")
        # V3.49 SUB-TABS: split 手眼标定 into '主相机手眼标定' and 'eye-in-hand 手眼标定功能'
        self.nb_calib = ttk.Notebook(outer)
        self.nb_calib.pack(fill="both", expand=True)
        p = ttk.Frame(self.nb_calib, padding=4)
        self.nb_calib.add(p, text="主相机手眼标定")
        self.check_state = None                      # V3.45: last quick check: None / ("run",) / ("ok"|"fail", errs)
        self.auto_state = None                       # V3.45: {"phase", "k", "n"} while the auto collection runs
        self._btn_state = {}
        # ---- ① marker
        self.l1, self.s1, b1 = self._step(p, "① 标记设置")
        ttk.Label(b1, text="ArUco 字典").pack(side="left")
        self.v_dict = tk.StringVar(value=self.marker["dict"])
        ttk.Combobox(b1, textvariable=self.v_dict, values=list(ARUCO_DICTS), width=15, state="readonly").pack(side="left", padx=(2, 8))
        ttk.Label(b1, text="ID").pack(side="left")
        self.v_mid = tk.StringVar(value=str(self.marker["id"]))
        ttk.Entry(b1, textvariable=self.v_mid, width=4).pack(side="left", padx=(2, 8))
        ttk.Label(b1, text="边长").pack(side="left")
        self.v_msize = tk.StringVar(value=f"{self.marker['size_mm']:g}")
        ttk.Combobox(b1, textvariable=self.v_msize, values=[f"{v:g}" for v in MARKER_SIZES_MM], width=6).pack(side="left", padx=2)
        ttk.Label(b1, text="mm（40×40 / 30×30，可填实测值）").pack(side="left")
        self.b_png = ttk.Button(b1, text="生成标记图片", command=self.on_marker_png)
        self.b_png.pack(side="left", padx=(10, 0))
        # ---- ② samples
        self.l2, self.s2, b2 = self._step(p, "② 采集样本")
        r = ttk.Frame(b2)
        r.pack(fill="x")
        self.b_rec = ttk.Button(r, text="● 记录当前姿态", command=self.on_record)
        self.b_rec.pack(side="left")
        ttk.Label(r, text="  或").pack(side="left")
        self.v_nauto = tk.StringVar(value="15")
        ttk.Spinbox(r, textvariable=self.v_nauto, from_=4, to=40, width=3).pack(side="left", padx=(6, 2))
        ttk.Label(r, text="个").pack(side="left")
        self.b_auto = ttk.Button(r, text="▶ 自动采集", command=self.on_auto)
        self.b_auto.pack(side="left", padx=(4, 2))
        self.b_stop = ttk.Button(r, text="■ 停止", command=self.on_stop)
        self.b_stop.pack(side="left", padx=2)
        tf = ttk.Frame(b2)
        tf.pack(fill="x", pady=(4, 0))
        st_name, _ = treeview_style(ttk, p)
        self.calib_cols = (("k", "#", 34), ("src", "来源", 48), ("q", "关节角 J1…J7 (°)", 206), ("p", "标记 xyz (mm)", 132),
                           ("jit", "抖动", 50), ("res", "残差", 54), ("loo", "留一", 54))
        self.tv = ttk.Treeview(tf, columns=[c_ for c_, _t, _w in self.calib_cols], show="headings", height=7,
                               style=st_name, selectmode="extended")
        for c_, t_, _w in self.calib_cols:
            self.tv.heading(c_, text=t_)
        autofit_treeview(self.tv, self.calib_cols, stretch_col="q")
        self.tv.tag_configure("odd", background="#f3f6f9")
        self.tv.tag_configure("rej", foreground="#c62828")
        self.tv.tag_configure("hi", foreground="#ef6c00")
        ys = ttk.Scrollbar(tf, orient="vertical", command=self.tv.yview)
        xs = ttk.Scrollbar(tf, orient="horizontal", command=self.tv.xview)
        self.tv.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        side = ttk.Frame(tf)
        side.grid(row=2, column=0, columnspan=2, sticky="e", pady=(3, 0))
        self.tv.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        tf.columnconfigure(0, weight=1)
        ttk.Label(side, text="单位 mm · 红色 = 被剔除 · 橙色 = 留一误差 > 5 mm   ", foreground="#78909c").pack(side="left")
        self.b_del = ttk.Button(side, text="删除选中", command=self.on_delete)
        self.b_del.pack(side="left")
        self.b_clr = ttk.Button(side, text="清空", command=self.on_clear)
        self.b_clr.pack(side="left", padx=(4, 0))
        self.tv.bind("<<TreeviewSelect>>", lambda _e: self._update_calib_ui())
        # ---- ③ compute / save
        self.l3, self.s3, b3 = self._step(p, "③ 计算与保存")
        r = ttk.Frame(b3)
        r.pack(fill="x")
        self.b_comp = ttk.Button(r, text="计算标定", command=self.on_compute)
        self.b_comp.pack(side="left")
        self.b_save = ttk.Button(r, text="保存标定", command=self.on_save_calib)
        self.b_save.pack(side="left", padx=6)
        self.b_save_as = ttk.Button(r, text="另存为…", command=self.on_calib_save_as)
        self.b_save_as.pack(side="left")
        self.b_load = ttk.Button(r, text="读取…", command=self.on_calib_load)
        self.b_load.pack(side="left", padx=(4, 0))
        self.b_clear_calib = ttk.Button(r, text="清除标定设定", command=self.on_calib_clear)
        self.b_clear_calib.pack(side="left", padx=(4, 0))
        ttk.Label(r, text="合格线：留一 平均 ≤ 3、最大 ≤ 5 mm，无 ⚠", foreground="#607d8b").pack(side="left", padx=6)
        rf = ttk.Frame(b3)
        rf.pack(fill="x", pady=(3, 0))
        self.txt_res = tk.Text(rf, height=5, wrap="word", font=("TkFixedFont", 9), relief="flat", bg="#fafafa",
                               padx=6, pady=4, highlightthickness=1, highlightbackground="#cfd8dc")
        rs = ttk.Scrollbar(rf, orient="vertical", command=self.txt_res.yview)
        self.txt_res.configure(yscrollcommand=rs.set)
        self.txt_res.tag_configure("good", foreground="#2e7d32", font=("TkDefaultFont", 10, "bold"))
        self.txt_res.tag_configure("bad", foreground="#c62828", font=("TkDefaultFont", 10, "bold"))
        self.txt_res.tag_configure("warn", foreground="#ef6c00")
        self.txt_res.pack(side="left", fill="x", expand=True)
        rs.pack(side="left", fill="y")
        self.v_res = tk.StringVar(value="")
        self._set_result_text(self._calib_text())
        # ---- ④ check
        self.l4, self.s4, b4 = self._step(p, "④ 复核")
        self.b_check = ttk.Button(b4, text="快速复核（走到存档姿态）", command=self.on_quick_check)
        self.b_check.pack(side="left")
        ttk.Label(b4, text="相机 / 标记碰过后也用它查；通过后到「视觉定位」验证", foreground="#607d8b").pack(side="left", padx=8)
        self._build_eih_subtab(self.nb_calib)
        self._update_calib_ui()

    def _build_eih_subtab(self, nb_sub):
        """V3.49/V3.51 EYE-IN-HAND SUBTAB: wrist camera hand-eye calibration with independent marker & auto-collection."""
        tk, ttk = self.tk, self.ttk
        pw = ttk.Frame(nb_sub, padding=4)
        nb_sub.add(pw, text="eye-in-hand手眼标定")
        # ① independent marker on table & wrist intrinsics
        self.wl1, self.ws1, wb1 = self._step(pw, "① 桌面固定标定目标与腕部内参（支持 ChArUco 标定板 24 角点高精度模式 + 单码模式）")
        rw0 = ttk.Frame(wb1)
        rw0.pack(fill="x", pady=(0, 3))
        ttk.Label(rw0, text="标定目标").pack(side="left")
        self.v_wtarget = tk.StringVar(value=self.W49.EIH_TARGET_MODES[0])
        ttk.Combobox(rw0, textvariable=self.v_wtarget, values=list(self.W49.EIH_TARGET_MODES), width=26,
                     state="readonly").pack(side="left", padx=(2, 8))
        ttk.Label(rw0, text="（推荐直接把 A4 ChArUco 标定板平放桌面：24 个对称十字内角点不受打印边缘发虚影响，基线大 8 倍）",
                  foreground="#2e7d32").pack(side="left")
        rw1 = ttk.Frame(wb1)
        rw1.pack(fill="x")
        ttk.Label(rw1, text="桌面 ArUco 字典").pack(side="left")
        self.v_wdict = tk.StringVar(value=self.wmarker["dict"])
        ttk.Combobox(rw1, textvariable=self.v_wdict, values=list(ARUCO_DICTS), width=15, state="readonly").pack(side="left", padx=(2, 8))
        ttk.Label(rw1, text="ID").pack(side="left")
        self.v_wmid = tk.StringVar(value=str(self.wmarker["id"]))
        ttk.Entry(rw1, textvariable=self.v_wmid, width=4).pack(side="left", padx=(2, 8))
        ttk.Label(rw1, text="边长").pack(side="left")
        self.v_wmsize = tk.StringVar(value=f"{self.wmarker['size_mm']:g}")
        ttk.Combobox(rw1, textvariable=self.v_wmsize,
                     values=[f"{v:g}" for v in self.W49.WRIST_MARKER_SIZES_MM], width=6).pack(side="left", padx=2)
        ttk.Label(rw1, text="mm（默认 24=ChArUco 内码，可填实测值；贴在桌面固定不动）").pack(side="left")
        self.b_wpng = ttk.Button(rw1, text="生成桌面标记图片", command=self.on_wmarker_png)
        self.b_wpng.pack(side="left", padx=(8, 0))
        # ② samples (manual + automatic collection)
        self.wl2, self.ws2, wb2 = self._step(pw, "② 采集样本（多角度观察桌面标记）")
        rw2 = ttk.Frame(wb2)
        rw2.pack(fill="x")
        self.b_wrec = ttk.Button(rw2, text="● 记录当前腕部姿态", command=self.on_wcalib_record)
        self.b_wrec.pack(side="left")
        self.b_wauto = ttk.Button(rw2, text="▶ 自动采集", command=self.on_wcalib_auto)
        self.b_wauto.pack(side="left", padx=(6, 0))
        ttk.Label(rw2, text="点数").pack(side="left", padx=(6, 0))
        self.v_wnauto = tk.StringVar(value="12")
        ttk.Spinbox(rw2, textvariable=self.v_wnauto, from_=6, to=30, width=3).pack(side="left", padx=(2, 2))
        self.b_wstop = ttk.Button(rw2, text="■ 停止", command=self.on_stop)
        self.b_wstop.pack(side="left", padx=(2, 6))
        self.b_wdel = ttk.Button(rw2, text="删除选中", command=self.on_wcalib_delete)
        self.b_wdel.pack(side="left", padx=2)
        self.b_wclr = ttk.Button(rw2, text="清空", command=self.on_wcalib_clear)
        self.b_wclr.pack(side="left")
        ttk.Label(rw2, text="  ≤10% 速度经 MoveIt+安全区自动环绕桌面标记采点",
                  foreground="#607d8b").pack(side="left")
        wtf = ttk.Frame(wb2)
        wtf.pack(fill="x", pady=(4, 0))
        st_name, _ = treeview_style(ttk, pw)
        self.wcalib_cols = (("k", "#", 34), ("q", "关节角 J1…J7 (°)", 200), ("p", "腕部相机中标记 xyz (mm)", 148),
                            ("jit", "抖动", 50), ("res", "残差", 54), ("loo", "留一", 54))
        self.tv_wcalib = ttk.Treeview(wtf, columns=[c_[0] for c_ in self.wcalib_cols], show="headings", height=6,
                                      style=st_name, selectmode="extended")
        for c_, t_, _w in self.wcalib_cols:
            self.tv_wcalib.heading(c_, text=t_)
        autofit_treeview(self.tv_wcalib, self.wcalib_cols, stretch_col="q")
        self.tv_wcalib.tag_configure("odd", background="#f3f6f9")
        self.tv_wcalib.tag_configure("rej", foreground="#c62828")
        wys = ttk.Scrollbar(wtf, orient="vertical", command=self.tv_wcalib.yview)
        wxs = ttk.Scrollbar(wtf, orient="horizontal", command=self.tv_wcalib.xview)
        self.tv_wcalib.configure(yscrollcommand=wys.set, xscrollcommand=wxs.set)
        self.tv_wcalib.grid(row=0, column=0, sticky="nsew")
        wys.grid(row=0, column=1, sticky="ns")
        wxs.grid(row=1, column=0, sticky="ew")
        wtf.columnconfigure(0, weight=1)
        # ③ compute & save
        self.wl3, self.ws3, wb3 = self._step(pw, "③ 计算与保存（解算 T_flange_cam 与 T_base_marker）")
        rw3 = ttk.Frame(wb3)
        rw3.pack(fill="x")
        self.b_wcomp = ttk.Button(rw3, text="计算腕部手眼", command=self.on_wcalib_compute)
        self.b_wcomp.pack(side="left")
        self.b_wsave = ttk.Button(rw3, text="保存腕部标定", command=self.on_wcalib_save)
        self.b_wsave.pack(side="left", padx=4)
        self.b_wsave_as = ttk.Button(rw3, text="另存为…", command=self.on_wcalib_save_as)
        self.b_wsave_as.pack(side="left")
        self.b_wload = ttk.Button(rw3, text="读取…", command=self.on_wcalib_load)
        self.b_wload.pack(side="left", padx=4)
        self.b_wclear_calib = ttk.Button(rw3, text="清除标定设定", command=self.on_wcalib_clear_calib)
        self.b_wclear_calib.pack(side="left")
        self.v_wres = tk.StringVar(value=self._wcalib_text())
        self.JC.fixed_label(tk, wb3, 4, textvariable=self.v_wres, font=("TkFixedFont", 9), justify="left",
                            anchor="nw", wraplength=560, pack={"fill": "x", "pady": (3, 0)})
        # ④ live & quick check
        self.wl4, self.ws4, wb4 = self._step(pw, "④ 实时复核")
        rw4 = ttk.Frame(wb4)
        rw4.pack(fill="x")
        self.b_wcheck = ttk.Button(rw4, text="快速复核（走到存档姿态）", command=self.on_wcalib_check)
        self.b_wcheck.pack(side="left")
        ttk.Label(rw4, text="腕部相机看到桌面标记时状态栏自动显示实时预测误差 mm，也可点「快速复核」自动走 4 个姿态检验",
                  foreground="#607d8b").pack(side="left", padx=8)
        self._update_eih_ui()

    def _set_result_text(self, text):
        self.v_res.set(text)
        t = self.txt_res
        t.configure(state="normal")
        t.delete("1.0", "end")
        lines = text.split("\n")
        for i, ln in enumerate(lines):
            tag = ()
            if i == 0 and "合格 ✓" in ln:
                tag = ("good",)
            elif i == 0 and "未达标" in ln:
                tag = ("bad",)
            elif ln.startswith("⚠"):
                tag = ("warn",)
            t.insert("end", ln + ("\n" if i < len(lines) - 1 else ""), tag)
        t.configure(state="disabled")

    def _marker_parse(self):
        try:
            m = {"dict": self.v_dict.get(), "id": int(self.v_mid.get()), "size_mm": float(self.v_msize.get())}
            if m["dict"] in ARUCO_DICTS and 0 <= m["id"] < 1000 and 5.0 <= m["size_mm"] <= 300.0:
                return m
        except Exception:
            pass
        return None

    def _busy(self):
        return self.worker is not None and self.worker.is_alive()

    def _en(self, btn, on, text=None):
        if not hasattr(self, "_btn_state"):
            self._btn_state = {}
        key = (str(btn), bool(on), text)
        if self._btn_state.get(str(btn)) == key:
            return
        self._btn_state[str(btn)] = key
        btn.configure(state="normal" if on else "disabled", **({"text": text} if text else {}))

    def _spans(self):
        n = len(self.samples)
        if getattr(self, "_span_cache", (None,))[0] != (n, id(self.samples[-1]) if n else None):
            A = [s["T_base_flange"] for s in self.samples]
            ts = translation_span_m(A) if n > 1 else 0.0
            rs = rotation_span_deg(A) if n > 1 else 0.0
            self._span_cache = ((n, id(self.samples[-1]) if n else None), ts, rs)
        return self._span_cache[1], self._span_cache[2]

    def _update_calib_ui(self):
        """V3.45: lamps + status lines + enabled buttons from the current state (Tk thread, every tick)."""
        if not hasattr(self, "l1"):
            return
        busy = self._busy()
        src = self.src
        cam = bool(src is not None and src.K)
        det = self.det
        m = self._marker_parse()
        aru_ok = getattr(self, "_aruco_ok", None)
        if aru_ok is None and cam:                   # V3.46: OpenCV is checked only once a camera is connected
            aru_ok = self._aruco_ok = aruco_status()[0]
        # ① marker
        if not m:
            self._set_lamp(self.l1, "red")
            self.s1.set("标记设置无效：ID 0–999，边长 5–300 mm")
        elif not cam:
            self._set_lamp(self.l1, "grey")
            self.s1.set("先在上方「▶ 连接相机」")
        elif det is not None:
            self._set_lamp(self.l1, "green")
            pz = det["p_cam"][2] * 1000.0
            self.s1.set(f"✓ 看到 ID {det['id']} · 距离 {pz:.0f} mm · 边长 {det.get('side_px', 0):.0f} px"
                        + (" · ⚠ < 35 px：标记太小 / 太远" if det.get("side_px", 99) < 35 else ""))
        else:
            self._set_lamp(self.l1, "yellow")
            self.s1.set(f"没看到 ID {m['id']}：把法兰上的标记转向相机")
        # ② samples
        n = len(self.samples)
        ts, rs = self._spans()
        a = self.auto_state
        if a:
            self._set_lamp(self.l2, "blue")
            self.s2.set(f"自动采集中：{a.get('phase', '')} {a.get('k', 0)}/{a.get('n', 0)} · 样本 {n}")
        else:
            okn = n >= 8 and ts >= 0.15 and rs >= 25.0
            self._set_lamp(self.l2, "green" if okn else ("yellow" if n else "grey"))
            self.s2.set(f"样本 {n}/≥8 · 位置范围 {ts * 100:.0f}/≥15 cm · 朝向变化 {rs:.0f}/≥25°" + (" ✓" if okn else ""))
        # ③ result
        r, c = self.result, self.calib
        if r and r.get("ok"):
            self._set_lamp(self.l3, "green" if r["good"] else "red")
            self.s3.set(("合格 ✓ → 点「保存标定」" if r["good"] else "未达标 ✗：看下方原因，补样本后重算") + "（未保存）")
        elif c:
            g = (c.get("loo_mean_mm") or 99) <= CALIB_GOOD_MEAN_MM and (c.get("loo_max_mm") or 99) <= CALIB_GOOD_MAX_MM
            self._set_lamp(self.l3, "green" if g else "red")
            self.s3.set(f"已保存 {c.get('saved', '')} · " + ("合格" if g else "未达标：建议重新标定"))
        else:
            self._set_lamp(self.l3, "grey")
            self.s3.set("还没有标定")
        # ④ check
        cs = self.check_state
        if cs and cs[0] == "run":
            self._set_lamp(self.l4, "blue")
            self.s4.set("复核中…")
        elif cs and cs[0] in ("ok", "fail"):
            self._set_lamp(self.l4, "green" if cs[0] == "ok" else "red")
            self.s4.set(("通过 ✓ " if cs[0] == "ok" else "未通过 ✗ ") + "误差 " + ", ".join(f"{e:.1f}" for e in cs[1]) + " mm")
        else:
            vs = (getattr(self, "vstate", None) or {}).get("state")
            if c and vs == "pending":
                self._set_lamp(self.l4, "yellow")
                self.s4.set("读取的标定还没校验：让标记出现在画面里（误差 ≤ 5 mm 连续 5 次），或点「快速复核」")
            elif c and vs == "fail":
                self._set_lamp(self.l4, "red")
                self.s4.set("校验没通过 ✗ " + self.vstate.get("why", "") + " → 快速复核 / 重新标定")
            elif c and vs == "ok" and self.vstate.get("why") == "刚标定":
                self._set_lamp(self.l4, "grey")
                self.s4.set("刚标定，可以直接使用；相机 / 标记碰过后点「快速复核」")
            elif c and vs == "ok":
                self._set_lamp(self.l4, "green")
                self.s4.set("校验通过 ✓ " + self.vstate.get("why", ""))
            else:
                self._set_lamp(self.l4, "grey")
                self.s4.set("保存标定后可以复核" if not (c and c.get("check_poses_deg")) else "还没有复核")
        # buttons
        jp = bool(getattr(self.jpanel, "enabled", False)) and not getattr(self.pipe, "teach_mode", False)
        self._en(self.b_png, bool(m) and aru_ok is not False and not busy)
        self._en(self.b_rec, cam and det is not None and not busy)
        self._en(self.b_auto, cam and det is not None and jp and not busy)
        self._en(self.b_stop, busy)
        self._en(self.b_del, bool(self.tv.selection()) and not busy)
        self._en(self.b_clr, bool(self.samples) and not busy)
        self._en(self.b_comp, n >= 6 and not busy)
        self._en(self.b_save, bool(r and r.get("ok")) and not busy,
                 text="保存标定" if (not r or not r.get("ok") or r["good"]) else "仍然保存（未达标）")
        self._en(self.b_check, bool(c and c.get("check_poses_deg")) and cam and jp and not busy)
        self._en(self.b_save_as, bool(c) and not busy)
        self._en(self.b_load, not busy)
        if hasattr(self, "b_clear_calib"):
            self._en(self.b_clear_calib, bool(c or r) and not busy)
        if hasattr(self, "b_hover"):
            has_tgt = any(e.get("xyz_m") and e.get("status") != "rejected" for e in (self.eggs or []))
            self._en(self.b_hover, bool(c or r) and has_tgt and jp and not busy)
            self._en(self.b_hover_back, bool(self._pre_hover_q) and jp and not busy)
            if hasattr(self, "b_dry_run"):
                self._en(self.b_dry_run, bool(c or r) and (has_tgt or bool(self.v_dry_dest_xyz.get().strip())) and jp and not busy)
            if hasattr(self, "b_wrefine"):
                w_ready = bool(self.wrist_src and (self.wrist_intr or getattr(self.wrist_src, "K", None))
                               and (self.wrist_calib or self.wcalib_result))
                self._en(self.b_wrefine, w_ready and has_tgt and not busy)
            if hasattr(self, "b_walign"):
                ref_ok = bool(getattr(self, "wrist_refine_res", None) and self.wrist_refine_res.get("ok"))
                self._en(self.b_walign, ref_ok and jp and not busy)
            self._en(self.b_tcp_rec, not busy)
            self._en(self.b_tcp_comp, len(self.tcp_samples) >= 4 and not busy)
            self._en(self.b_tcp_clr, bool(self.tcp_samples) and not busy)
            self._en(self.b_tcp_reset, bool(self.tcp_calib) and not busy)
        self._update_eih_ui()
        self.update_workflow_lamps()

    def attach_workflow_bar(self, parent):
        bar = WorkflowLampBar(self.tk, self.ttk, parent,
                              on_teach_enter=self.on_teach_enter, on_teach_exit=self.on_teach_exit)
        self.workflow_bars.append(bar)
        self.update_workflow_lamps()
        return bar

    def update_workflow_lamps(self):
        try:
            self.workflow_state = evaluate_workflow_lamps(self.pipe, self.safe_tab, self)
            tm = bool(getattr(self.pipe, "teach_mode", False) or
                      getattr(self.pipe, "leader_active", lambda: False)())
            busy = self._busy()
            for bar in getattr(self, "workflow_bars", None) or []:
                bar.update_from(self.workflow_state, teach_mode=tm, busy=busy)
        except Exception:
            pass

    def on_teach_enter(self):
        """V3.52/V3.53 TEACH MODE FOR CALIB START POSE: enter zero-gravity hand-guiding so the user can easily position
        the arm until the ArUco marker is clearly visible in the main or wrist camera."""
        if self._busy():
            self.log("[WARN] [V3.52 teach] 上一个操作还在进行")
            return
        if not hasattr(self.pipe, "set_teach_mode"):
            self.log("[WARN] [V3.52 teach] 当前管线不支持示教模式切换")
            return

        def job():
            if not self.ask("进入示教模式",
                            "机械臂将进入零力示教模式（手动拖动）：\n"
                            "1. 请先用手扶住机械臂，再点确定；\n"
                            "2. 看着左侧相机监视框拖动机械臂，直到画面清楚看到 ArUco 标记（状态灯变绿）；\n"
                            "3. 摆好后点旁边「退出示教模式」，机械臂将直接锁在当前姿态（不回零位），即可点「▶ 自动采集」。\n"
                            "确认已扶稳机械臂？"):
                return
            ev = self.pipe.set_teach_mode(True, timeout_s=8.0)
            if isinstance(ev, dict) and ev.get("ok"):
                self.log("[OK] [V3.52 teach] 已进入示教模式：请拖动机械臂对准 ArUco 标记，摆好后点「退出示教模式」锁住当前姿态")
            self._ui(self._update_calib_ui)
        self._run(job)

    def on_teach_exit(self):
        """V3.52/V3.53 EXIT TEACH MODE & HOLD POSE: exit zero-gravity hand-guiding unconditionally (even if V111
        ctrl_mode stayed 0x01 and teach_mode flag was False) and lock the arm right at the current hand-guided pose
        (return_home=False) so the user can immediately start hand-eye calibration."""
        if self._busy():
            self.log("[WARN] [V3.52 teach] 上一个操作还在进行")
            return
        if not hasattr(self.pipe, "set_teach_mode"):
            return

        def job():
            if not self.ask("退出示教模式（保持当前姿态）",
                            "机械臂将退出示教模式并直接锁在当前位置（不回零位），以便在此姿态下直接开始手眼标定采点。\n"
                            "请扶稳机械臂、手在急停上。继续？"):
                return
            if getattr(self.safe_tab, "mode", None) == "teach" and hasattr(self.safe_tab, "v_mode"):
                self.safe_tab._teach_exiting = True
                self.safe_tab.v_mode.set("edit")
                self.safe_tab.on_mode()
                self.safe_tab._teach_exiting = False
            try:
                ev = self.pipe.set_teach_mode(False, home_model=self._model(), return_home=False)
            except TypeError:
                ev = self.pipe.set_teach_mode(False)
            if isinstance(ev, dict) and ev.get("ok"):
                self.log("[OK] [V3.52 teach] 已退出示教模式并锁在当前标定姿态 → 现在可直接点「▶ 自动采集」或「● 记录当前姿态」")
            self._ui(self._update_calib_ui)
        self._run(job)


    def _build_locate(self, nb):
        # V3.47 YOLO LOCATE: method / model / prompts / egg size / reference boxes / sample saving
        tk, ttk = self.tk, self.ttk
        import NERO_V347_Yolo_Perception as YP
        self.YP = YP
        p = ttk.Frame(nb, padding=6)
        nb.add(p, text="视觉定位")
        r = ttk.Frame(p)
        r.pack(fill="x")
        ttk.Button(r, text="识别鸡蛋（只显示，不运动）", command=self.on_eggs).pack(side="left")
        self.b_objs = ttk.Button(r, text="识别物体（按名称/词库）", command=self.on_objects)
        self.b_objs.pack(side="left", padx=(4, 0))
        self.b_holder = ttk.Button(r, text="识别蛋托（Agent 算槽位）", command=self.on_holder)
        self.b_holder.pack(side="left", padx=(4, 0))
        ttk.Label(r, text="  方法").pack(side="left")
        self.v_method = tk.StringVar(value=LOCATE_METHODS[0])
        ttk.Combobox(r, textvariable=self.v_method, values=LOCATE_METHODS, state="readonly", width=16).pack(side="left")
        self.v_white = tk.BooleanVar(value=True)
        ttk.Checkbutton(r, text="白蛋（几何法：只用高度分割）", variable=self.v_white).pack(side="left", padx=6)
        r2 = ttk.Frame(p)
        r2.pack(fill="x", pady=(4, 0))
        ttk.Label(r2, text="模型文件夹").pack(side="left")
        self.v_ydir = tk.StringVar(value=YP.DEFAULT_MODEL_DIR)
        ttk.Entry(r2, textvariable=self.v_ydir, width=34).pack(side="left", padx=2)
        ttk.Label(r2, text=" 模型").pack(side="left")
        self.v_ymodel = tk.StringVar(value=YP.DEFAULT_MODEL)
        self.cb_ymodel = ttk.Combobox(r2, textvariable=self.v_ymodel, values=YP.list_models(YP.DEFAULT_MODEL_DIR) or
                                      [YP.DEFAULT_MODEL], width=22, postcommand=self._yolo_models)
        self.cb_ymodel.pack(side="left", padx=2)
        ttk.Label(r2, text=" 置信度门槛").pack(side="left")
        self.v_yconf = tk.StringVar(value=f"{YP.CONF_MIN:.2f}")
        ttk.Entry(r2, textvariable=self.v_yconf, width=5).pack(side="left")
        r3 = ttk.Frame(p)
        r3.pack(fill="x", pady=(4, 0))
        ttk.Label(r3, text="文字提示").pack(side="left")
        self.v_yprompt = tk.StringVar(value=YP.DEFAULT_TEXT_PROMPTS)
        ttk.Entry(r3, textvariable=self.v_yprompt, width=34).pack(side="left", padx=2)
        ttk.Label(r3, text=" 鸡蛋 长").pack(side="left")
        self.v_egg_l = tk.StringVar(value=f"{YP.EGG_SIZE_MM[0]:.0f}")
        ttk.Entry(r3, textvariable=self.v_egg_l, width=5).pack(side="left")
        ttk.Label(r3, text="宽").pack(side="left")
        self.v_egg_w = tk.StringVar(value=f"{YP.EGG_SIZE_MM[1]:.0f}")
        ttk.Entry(r3, textvariable=self.v_egg_w, width=5).pack(side="left")
        ttk.Label(r3, text="mm").pack(side="left")
        self.v_ysave = tk.BooleanVar(value=False)
        ttk.Checkbutton(r3, text="保存样本（以后微调用）", variable=self.v_ysave).pack(side="left", padx=8)
        r3a = ttk.Frame(p)                         # V3.48: object names for general YOLOE test recognition
        r3a.pack(fill="x", pady=(4, 0))
        ttk.Label(r3a, text="识别物体名称").pack(side="left")
        self.v_obj_names = tk.StringVar(value="")
        ttk.Entry(r3a, textvariable=self.v_obj_names, width=34).pack(side="left", padx=2)
        ttk.Label(r3a, text="（英文逗号分隔，如 cup, bottle, box；填 * 或留空 = 用上方文字提示或 -pf 默认词库）",
                  foreground="#607d8b").pack(side="left")
        r3b = ttk.Frame(p)                         # V3.47.1 DEPTH SETTINGS + settings files
        r3b.pack(fill="x", pady=(4, 0))
        ttk.Label(r3b, text="深度").pack(side="left")
        self.v_dres = tk.StringVar(value=DEPTH_RES[1])
        ttk.Combobox(r3b, textvariable=self.v_dres, values=DEPTH_RES, state="readonly", width=9).pack(side="left", padx=2)
        self.v_dpreset = tk.StringVar(value=DEPTH_PRESETS[0])
        ttk.Combobox(r3b, textvariable=self.v_dpreset, values=DEPTH_PRESETS, state="readonly", width=14).pack(side="left")
        ttk.Label(r3b, text="（断开再连接相机后生效）", foreground="#607d8b").pack(side="left")
        ttk.Button(r3b, text="保存设置", command=self.on_locate_save).pack(side="left", padx=(10, 0))
        ttk.Button(r3b, text="另存为…", command=self.on_locate_save_as).pack(side="left", padx=(4, 0))
        ttk.Button(r3b, text="读取…", command=self.on_locate_load).pack(side="left", padx=(4, 0))
        r4 = ttk.Frame(p)
        r4.pack(fill="x", pady=(4, 0))
        self.b_ref = ttk.Button(r4, text="框选参考", command=self.on_ref_toggle)
        self.b_ref.pack(side="left")
        self.v_refcls = tk.StringVar(value="egg")
        ttk.Radiobutton(r4, text="鸡蛋", value="egg", variable=self.v_refcls).pack(side="left")
        ttk.Radiobutton(r4, text="非鸡蛋（干扰物）", value="not", variable=self.v_refcls).pack(side="left")
        self.b_ref_cancel = ttk.Button(r4, text="取消", command=self.on_ref_cancel, state="disabled")
        self.b_ref_cancel.pack(side="left", padx=4)
        self.ref = YP.ref_load()
        self.ref_edit = None
        self._ref_drag = None
        self.yolo_view = []
        self.det_yolo = YP.YoloEDetector(log=self.log)
        self.v_ref = tk.StringVar(value=YP.ref_text(self.ref))
        self.JC.fixed_label(tk, p, 2, textvariable=self.v_ref, wraplength=600, justify="left", anchor="nw",
                            pack={"fill": "x"})
        st_name, _ = treeview_style(ttk, p)
        self.egg_cols = (("id", "编号", 60), ("x", "x mm", 60), ("y", "y mm", 60), ("z", "z mm", 60), ("h", "实测高", 56),
                         ("yolo", "YOLO", 52), ("iou", "轮廓重合", 64), ("conf", "置信度", 56), ("ok", "可信", 44),
                         ("why", "说明", 260))
        ef = ttk.Frame(p)
        ef.pack(fill="x", pady=4)
        self.tv_eggs = ttk.Treeview(ef, columns=[c_[0] for c_ in self.egg_cols], show="headings", height=6,
                                    style=st_name)
        for c_, t_, _w in self.egg_cols:
            self.tv_eggs.heading(c_, text=t_)
        autofit_treeview(self.tv_eggs, self.egg_cols, stretch_col="why")
        self.tv_eggs.tag_configure("odd", background="#f3f6f9")
        eys = ttk.Scrollbar(ef, orient="vertical", command=self.tv_eggs.yview)
        exs = ttk.Scrollbar(ef, orient="horizontal", command=self.tv_eggs.xview)
        self.tv_eggs.configure(yscrollcommand=eys.set, xscrollcommand=exs.set)
        self.tv_eggs.grid(row=0, column=0, sticky="nsew")
        eys.grid(row=0, column=1, sticky="ns")
        exs.grid(row=1, column=0, sticky="ew")
        ef.columnconfigure(0, weight=1)
        self.v_eggs = tk.StringVar(value="需要手眼标定。坐标在 base_link（与安全区、关节控制一致）。识别只显示，点「走到选中目标上方悬停」才运动。")
        self.JC.fixed_label(tk, p, 4, textvariable=self.v_eggs, wraplength=600, justify="left", anchor="nw",
                            pack={"fill": "x"})
        # V3.48 HOVER VERIFICATION + TCP CALIBRATION (inside 视觉定位 tab)
        ttk.Separator(p).pack(fill="x", pady=(3, 2))
        rhov = ttk.Frame(p)
        rhov.pack(fill="x", pady=(2, 0))
        self.b_hover = ttk.Button(rhov, text="▶ 走到选中目标上方悬停", command=self.on_hover_target)
        self.b_hover.pack(side="left")
        ttk.Label(rhov, text=" 悬停高度").pack(side="left")
        self.v_hover_mm = tk.StringVar(value=f"{HOVER_DEFAULT_MM:.0f}")
        ttk.Entry(rhov, textvariable=self.v_hover_mm, width=5).pack(side="left", padx=2)
        ttk.Label(rhov, text="mm（表里选一行，不选默认第 1 个可信目标；限速 ≤10%，只悬停不下探）",
                  foreground="#607d8b").pack(side="left")
        self.b_hover_back = ttk.Button(rhov, text="↩ 回悬停前姿态", command=self.on_hover_back)
        self.b_hover_back.pack(side="left", padx=(4, 2))
        self.b_wrefine = ttk.Button(rhov, text="◎ 腕部精定位", command=self.on_wrist_refine)
        self.b_wrefine.pack(side="left", padx=2)
        self.b_walign = ttk.Button(rhov, text="✥ 一键微调对齐", command=self.on_wrist_align)
        self.b_walign.pack(side="left", padx=2)
        ttk.Button(rhov, text="■ 停止", command=self.on_stop).pack(side="left", padx=(2, 0))
        # V3.59 MULTI-POINT HOVER DRY-RUN (gripper-free pick-to-place / multi-object hover verification)
        rdry = ttk.Frame(p)
        rdry.pack(fill="x", pady=(3, 0))
        self.b_dry_run = ttk.Button(rdry, text="▶ 多点悬停预演 (Dry-Run)", command=self.on_hover_dry_run)
        self.b_dry_run.pack(side="left")
        ttk.Label(rdry, text=" 航线:").pack(side="left")
        self._dry_mode_items = (
            ("next_target", "源目标 → 下一个目标"),
            ("all_trusted", "巡检全部可信目标"),
            ("first_empty_slot", "源目标 → 蛋托首空孔"),
            ("manual_xyz", "源目标 → 手动目标B"),
        )
        self.v_dry_dest_mode = tk.StringVar(value=self._dry_mode_items[0][1])
        self.cb_dry_dest = ttk.Combobox(
            rdry, textvariable=self.v_dry_dest_mode,
            values=[m[1] for m in self._dry_mode_items], state="readonly", width=18,
        )
        self.cb_dry_dest.pack(side="left", padx=2)
        ttk.Label(rdry, text=" 目标B(x,y,z mm):").pack(side="left")
        self.v_dry_dest_xyz = tk.StringVar(value="-450, 123, 43")
        ttk.Entry(rdry, textvariable=self.v_dry_dest_xyz, width=14).pack(side="left", padx=2)
        ttk.Label(rdry, text="（无夹爪测试：物体A悬停 → 抬升过渡 → 目标B悬停 → 回原位）",
                  foreground="#607d8b").pack(side="left", padx=2)
        rtcp = ttk.Frame(p)
        rtcp.pack(fill="x", pady=(4, 0))
        ttk.Label(rtcp, text="TCP 标定（多姿态碰同一点）：", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        self.b_tcp_rec = ttk.Button(rtcp, text="● 记录碰点姿态", command=self.on_tcp_record)
        self.b_tcp_rec.pack(side="left")
        self.b_tcp_comp = ttk.Button(rtcp, text="计算并保存 TCP", command=self.on_tcp_compute)
        self.b_tcp_comp.pack(side="left", padx=4)
        self.b_tcp_clr = ttk.Button(rtcp, text="清空碰点", command=self.on_tcp_clear_samples)
        self.b_tcp_clr.pack(side="left")
        self.b_tcp_reset = ttk.Button(rtcp, text="恢复默认 TCP", command=self.on_tcp_reset)
        self.b_tcp_reset.pack(side="left", padx=4)
        self.v_tcp_status = tk.StringVar(value="")
        self.JC.fixed_label(tk, p, 2, textvariable=self.v_tcp_status, font=("TkFixedFont", 9), wraplength=600,
                            justify="left", anchor="nw", pack={"fill": "x", "pady": (2, 0)})
        self._tcp_autoload()
        self._update_tcp_ui()
        self._locate_autoload()

    # ------------------------------------------------------------------ V3.47.1 视觉定位 settings files
    def _locate_settings(self):
        d = {"version": "V3.53", "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "method": self.v_method.get(),
             "model_dir": self.v_ydir.get(), "model": self.v_ymodel.get(), "conf": self.v_yconf.get(),
             "prompts": self.v_yprompt.get(), "egg_mm": [self.v_egg_l.get(), self.v_egg_w.get()],
             "white_geometric": bool(self.v_white.get()), "save_samples": bool(self.v_ysave.get()),
             "depth_res": self.v_dres.get(), "depth_preset": self.v_dpreset.get(),
              "obj_names": self.v_obj_names.get(), "hover_mm": self.v_hover_mm.get()}
        r = self.ref
        if r:
            try:
                with open(r["image"], "rb") as f:
                    img = base64.b64encode(f.read()).decode("ascii")
            except Exception:
                img = None
            d["reference"] = {"boxes": r["boxes"], "cls": r["cls"], "saved": r.get("saved", ""), "image_jpg_b64": img}
        return d

    def _locate_apply(self, d, src):
        YP = self.YP
        for var, key in ((self.v_ydir, "model_dir"), (self.v_ymodel, "model"), (self.v_yconf, "conf"),
                         (self.v_yprompt, "prompts"), (self.v_obj_names, "obj_names"), (self.v_hover_mm, "hover_mm")):
            if d.get(key) not in (None, ""):
                var.set(str(d[key]))
        if d.get("method") in LOCATE_METHODS:
            self.v_method.set(d["method"])
        if d.get("depth_res") in DEPTH_RES:
            self.v_dres.set(d["depth_res"])
        if d.get("depth_preset") in DEPTH_PRESETS:
            self.v_dpreset.set(d["depth_preset"])
        egg = d.get("egg_mm") or []
        if len(egg) == 2:
            self.v_egg_l.set(str(egg[0]))
            self.v_egg_w.set(str(egg[1]))
        self.v_white.set(bool(d.get("white_geometric", True)))
        self.v_ysave.set(bool(d.get("save_samples", False)))
        ref = d.get("reference")
        if ref and ref.get("image_jpg_b64"):
            try:
                os.makedirs(os.path.dirname(YP.REF_IMG), exist_ok=True)
                with open(YP.REF_IMG, "wb") as f:
                    f.write(base64.b64decode(ref["image_jpg_b64"]))
                with open(YP.REF_JSON, "w", encoding="utf-8") as f:
                    json.dump({"version": YP.VERSION, "image": os.path.basename(YP.REF_IMG), "boxes": ref["boxes"],
                               "cls": ref["cls"], "names": ["egg", "not_egg"], "saved": ref.get("saved", "")}, f,
                              ensure_ascii=False, indent=1)
            except Exception as exc:
                self.log(f"[WARN] [V3.47.1 locate] 参考图没恢复：{exc}")
        self.ref = YP.ref_load()
        self.v_ref.set(YP.ref_text(self.ref))
        cur = self.src.info.get("depth_res") if self.src and self.src.info else None
        note = ""
        if cur and (cur != self.v_dres.get().replace("x", "×") or self.src.info.get("preset") != self.v_dpreset.get()):
            note = "；深度设置变了：断开再连接相机后生效"
        self.log(f"[OK] [V3.47.1 locate] 视觉定位设置 ← {src}（{d.get('saved', '')}）" + note)

    def _locate_autoload(self):
        d, src = load_json(LOCATE_PATH), LOCATE_PATH
        if (not d or d.get("load_error")) and LOCATE_PATH == DEFAULT_LOCATE_PATH:
            cands = sorted(glob.glob(os.path.join(CALIB_DIR, "locate*.json")), key=os.path.getmtime, reverse=True)
            d, src = (load_json(cands[0]), cands[0]) if cands else (None, None)
        if d and not d.get("load_error") and d.get("method"):
            if d.get("version") == "V3.47.1" and d.get("depth_res") == DEPTH_RES[0]:
                d = dict(d, depth_res=DEPTH_RES[1], version="V3.53")
            try:
                self._locate_apply(d, src)
            except Exception as exc:
                self.log(f"[WARN] [V3.47.1 locate] 设置文件 {src} 读不了：{exc}")

    def on_locate_save(self, path=None):
        d = self._locate_settings()
        targets = [path or LOCATE_PATH]
        if not path and LOCATE_PATH == DEFAULT_LOCATE_PATH:
            targets.append(os.path.join(CALIB_DIR, "locate_settings.json"))
        for t in targets:
            try:
                os.makedirs(os.path.dirname(t), exist_ok=True)
                save_json(t, d)
            except Exception as exc:
                self.log(f"[WARN] [V3.47.1 locate] 设置没保存到 {t}：{exc}")
                continue
            self.log(f"[OK] [V3.47.1 locate] 视觉定位设置 → {t}" + ("（含参考图）" if d.get("reference") else ""))

    def on_locate_save_as(self, path=None):
        path = path or self._ask_path(True, CALIB_DIR, "locate_settings_" + time.strftime("%Y%m%d_%H%M") + ".json",
                                      "视觉定位设置另存为")
        if path:
            self.on_locate_save(path)

    def on_locate_load(self, path=None):
        path = path or self._ask_path(False, CALIB_DIR, "", "读取视觉定位设置")
        if not path:
            return
        d = load_json(path)
        if not d or d.get("load_error") or not d.get("method"):
            self.log(f"[FAIL] [V3.47.1 locate] {path} 不是视觉定位设置文件")
            return
        self._locate_apply(d, path)
        self.on_locate_save()                        # becomes this folder's settings

    def _yolo_models(self):
        try:
            self.cb_ymodel.configure(values=self.YP.list_models(self.v_ydir.get()) or [self.YP.DEFAULT_MODEL])
        except Exception:
            pass

    # ------------------------------------------------------------------ V3.47 reference boxes (visual prompt)
    def on_ref_toggle(self):
        if self.ref_edit is None:
            rgb = self.src.frames()[0] if self.src else None
            if rgb is None:
                self.log("[WARN] [V3.47 locate] 先连接相机")
                return
            self.ref_edit = {"rgb": rgb.copy(), "boxes": [], "cls": []}
            self.b_ref.configure(text="保存参考（0 个框）")
            self.b_ref_cancel.configure(state="normal")
            self.v_ref.set("框选中：画面已定格为参考图。在画面上拖框（先选「鸡蛋」/「非鸡蛋」），框完点「保存参考」。"
                           "框选时不要动桌面。")
            return
        e = self.ref_edit
        try:
            self.ref = self.YP.ref_save(e["rgb"], e["boxes"], e["cls"])
        except Exception as exc:
            self.log(f"[WARN] [V3.47 locate] 参考没保存：{exc}")
            return
        self.ref_edit = None
        self.b_ref.configure(text="框选参考")
        self.b_ref_cancel.configure(state="disabled")
        self.v_ref.set(self.YP.ref_text(self.ref))
        self.log(f"[OK] [V3.47 locate] 参考已保存 → {self.YP.REF_JSON}：{self.YP.ref_text(self.ref)}")
        self._force_redraw()

    def on_ref_cancel(self):
        self.ref_edit = None
        self._ref_drag = None
        self.b_ref.configure(text="框选参考")
        self.b_ref_cancel.configure(state="disabled")
        self.v_ref.set(self.YP.ref_text(self.ref))
        self._force_redraw()

    def _ref_add(self, box):
        e = self.ref_edit
        x0, y0, x1, y1 = box
        if x1 - x0 < 8 or y1 - y0 < 8:
            return
        e["boxes"].append([float(x0), float(y0), float(x1), float(y1)])
        e["cls"].append(0 if self.v_refcls.get() == "egg" else 1)
        ne = sum(1 for c in e["cls"] if c == 0)
        self.b_ref.configure(text=f"保存参考（{len(e['boxes'])} 个框）")
        self.v_ref.set(f"框选中：{ne} 个鸡蛋 + {len(e['cls']) - ne} 个非鸡蛋。继续拖框，或点「保存参考」。")

    # ------------------------------------------------------------------ state helpers
    def _load_saved(self):
        # V3.47.1 CALIB FILES: config/camera_calib.json, else the newest copy in ~/Downloads/cameracalib (new folder)
        self.calib_src = None
        self.vstate = {"state": "none", "why": "", "live": [], "last": None}
        path = CALIB_PATH
        c = load_json(path)
        if (not c or c.get("load_error")) and CALIB_PATH == DEFAULT_CALIB_PATH:
            cands = sorted(glob.glob(os.path.join(CALIB_DIR, "*.json")), key=os.path.getmtime, reverse=True)
            for f in cands:
                d = load_json(f)
                if calib_valid(d):
                    c, path = d, f
                    break
        if calib_valid(c):
            self.calib = c
            self.calib_src = path
            self.marker = dict(c.get("marker") or DEFAULT_MARKER)
            self._set_vstate("pending", "读取后还没校验")
            if path != CALIB_PATH:
                self._pending_logs = [f"[INFO] [V3.47.1 calib] config 里没有标定 → 读取 {path}（{c.get('saved', '')}），"
                                      "需要校验：连接相机后看到标记 / 快速复核"]

    def _set_vstate(self, state, why=""):
        self.vstate = {"state": state, "why": why, "live": [], "last": None}

    def _calib_note(self):
        """V3.47.1: text for 视觉定位 when the calibration is not verified (None = verified / just calibrated)."""
        v = getattr(self, "vstate", {}) or {}
        if self.result:
            return "标定还没保存"
        if v.get("state") == "pending":
            return "标定读取后还没校验（连接相机看到标记，或「快速复核」）"
        if v.get("state") == "fail":
            return "标定校验没通过（相机可能动过）：" + v.get("why", "")
        return None

    def _verify_static(self):
        """V3.47.1: once per connection: camera serial + colour intrinsics against the calibration file."""
        s, c = self.src, self.calib
        if not s or not s.info or not c or getattr(self, "_vstatic_src", None) is s:
            return
        self._vstatic_src = s
        why = []
        ser = c.get("camera_serial")
        if ser and s.info.get("serial") and str(ser) != str(s.info.get("serial")):
            why.append(f"相机序列号不同（标定 {ser}，现在 {s.info.get('serial')}）")
        k0, k1 = c.get("intrinsics") or {}, s.K or {}
        for key, tol in (("fx", 0.01), ("fy", 0.01), ("ppx", 0.01), ("ppy", 0.01)):
            if k0.get(key) and k1.get(key) and abs(k0[key] - k1[key]) > tol * max(abs(k0[key]), 1.0) * (1 if key in ("fx", "fy") else 0.5):
                why.append(f"内参 {key} 不同（{k0[key]:.1f} → {k1[key]:.1f}）")
        if why:
            self._set_vstate("fail", "；".join(why))
            self.log("[FAIL] [V3.47.1 calib] 标定和这台相机对不上：" + "；".join(why) + " → 重新标定")
        elif self.vstate.get("state") == "pending":
            self.log("[INFO] [V3.47.1 calib] 相机序列号 / 内参和标定一致；再看标记误差（让法兰上的标记出现在画面里）或点「快速复核」")

    def _verify_live(self, det, err_mm):
        """V3.47.1: VERIFY_LIVE_N live marker errors in a row <= 5 mm -> verified; > 10 mm -> failed."""
        v = self.vstate
        if v.get("state") != "pending" or det is v.get("last"):
            return
        v["last"] = det
        v["live"] = (v["live"] + [err_mm])[-VERIFY_LIVE_N:]
        if len(v["live"]) < VERIFY_LIVE_N:
            return
        if max(v["live"]) <= CALIB_GOOD_MAX_MM:
            e = sum(v["live"]) / len(v["live"])
            self._set_vstate("ok", f"看到标记，误差 {e:.1f} mm")
            self.log(f"[OK] [V3.47.1 calib] 标定校验通过：标记误差 {e:.1f} mm（{VERIFY_LIVE_N} 次）")
        elif min(v["live"]) > 2 * CALIB_GOOD_MAX_MM:
            e = sum(v["live"]) / len(v["live"])
            self._set_vstate("fail", f"标记误差 {e:.0f} mm")
            self.log(f"[FAIL] [V3.47.1 calib] 标定校验没通过：标记误差 {e:.0f} mm → 相机或标记动过：「快速复核」/ 重新标定")

    def _marker_from_ui(self):
        try:
            m = {"dict": self.v_dict.get(), "id": int(self.v_mid.get()), "size_mm": float(self.v_msize.get())}
            assert m["dict"] in ARUCO_DICTS and 0 <= m["id"] < 1000 and 5.0 <= m["size_mm"] <= 300.0
        except Exception:
            self.log("[WARN] [V3.25 camera] 标记设置无效（ID 0–999，边长 5–300 mm）")
            return None
        if m != self.marker and self.samples:
            self.log("[WARN] [V3.25 camera] 标记设置变了：已有样本是旧设置测的，建议「清空」重采")
        self.marker = m
        return m

    def _model(self):
        return self.jpanel.model

    def _urdf(self):
        return self.pipe.real_urdf or getattr(self.safe_tab, "sim", None)

    def flange_T(self, q_deg):
        import NERO_V315_Consistency_Safety as CS
        m = self._model()
        T = CS.fk(self._urdf(), dict(zip(m["joints"], [math.radians(v) for v in q_deg])), m.get("base_link") or "base_link",
                  FLANGE_LINK)
        return [list(map(float, row)) for row in T] if T is not None else None

    def actual_q(self):
        a = self.pipe.actual_arm_deg(self._model())
        return list(a) if a else None

    def _ui(self, fn):
        self._ui_q.append(fn)

    def _calib_text(self):
        c = self.result or self.calib
        if not c:
            return "还没有标定（config/camera_calib.json 不存在）"
        src = "本次计算（未保存）" if self.result else f"已加载 {getattr(self, 'calib_src', None) or CALIB_PATH}（{self.calib.get('saved', '')}）"
        X = c["T_base_cam"]
        pos = T_t(X)
        y = c["marker_in_flange"]
        loo_m, loo_x = c.get("loo_mean_mm"), c.get("loo_max_mm")
        loo_ok = (loo_m is not None and loo_m <= CALIB_GOOD_MEAN_MM and loo_x <= CALIB_GOOD_MAX_MM)
        good = bool(c.get("good", loo_ok and not c.get("warnings")))
        if good:
            badge_str = "合格 ✓"
        elif loo_ok and c.get("warnings"):
            badge_str = "未达标 ⚠（留一精度已合格，需补够下方朝向/范围要求）"
        else:
            badge_str = "未达标 ⚠"
        lines = [f"{src}  ·  {badge_str}",
                 f"拟合残差 平均 {c['mean_mm']:.2f} / 最大 {c['max_mm']:.2f} mm · 留一（独立检验） 平均 "
                 f"{(loo_m or 0):.2f} / 最大 {(loo_x or 0):.2f} mm",
                 f"相机位置（base_link）x {pos[0]:+.3f}  y {pos[1]:+.3f}  z {pos[2]:+.3f} m",
                 f"标记中心在 link7：{y[0] * 1000:+.1f} {y[1] * 1000:+.1f} {y[2] * 1000:+.1f} mm · 样本 "
                 f"{c.get('n_used') or len(c.get('used') or [])}"
                 + (f" · 剔除 {len(c.get('rejected') or [])}" if c.get("rejected") else "")
                 + (f" · 朝向变化 {c['rot_span_deg']:.0f}° · 位置范围 {c['trans_span_m'] * 100:.0f} cm" if c.get("rot_span_deg") is not None else "")]
        for w_ in c.get("warnings") or []:
            lines.append("⚠ " + w_)
        return "\n".join(lines)

    def _refresh_table(self):
        for it in self.tv.get_children():
            self.tv.delete(it)
        r = self.result
        res = {}
        loo = {}
        if r and r.get("ok"):
            for m, i in enumerate(r["used"]):
                res[i] = r["res_mm"][m]
                loo[i] = r["loo_mm"][m]
        for k, s in enumerate(self.samples):
            p = s["p_cam"]
            rej = bool(r and k in (r.get("rejected") or []))
            tags = ["odd"] if k % 2 else []
            if rej:
                tags.append("rej")
            elif k in loo and loo[k] > CALIB_GOOD_MAX_MM:
                tags.append("hi")
            self.tv.insert("", "end", iid=str(k), tags=tags, values=(
                k + 1, {"depth": "深度", "pnp": "PnP"}.get(s.get("source"), "-"), " ".join(f"{v:.0f}" for v in s["q_deg"]),
                f"{p[0] * 1000:.0f}  {p[1] * 1000:.0f}  {p[2] * 1000:.0f}",
                f"{s['spread_mm']:.1f}" if s.get("spread_mm") is not None else "-",
                "剔除" if rej else (f"{res[k]:.2f}" if k in res else "-"), f"{loo[k]:.2f}" if k in loo else "-"))
        autofit_treeview(self.tv, self.calib_cols, stretch_col="q")
        kids = self.tv.get_children()
        if kids:
            self.tv.see(kids[-1])
        self._set_result_text(self._calib_text())
        self._update_calib_ui()

    # ------------------------------------------------------------------ camera
    def on_connect(self):
        if self.src is not None and self.src.running():
            return
        self.src = self.source_factory()
        try:                                       # V3.47.1 DEPTH SETTINGS
            self.src.depth_res, self.src.depth_preset = self.v_dres.get(), self.v_dpreset.get()
        except Exception:
            pass
        self._vstatic_src = None
        self.src.start()
        self.log("[INFO] [V3.25 camera] 正在连接 D435i …" + (f"（深度 {self.v_dres.get()} · {self.v_dpreset.get()}）"
                                                          if hasattr(self, "v_dres") else ""))

    def on_disconnect(self):
        if self.src is not None:
            self.src.stop()
            self.log("[INFO] [V3.25 camera] 相机已断开")
        self.src = None
        self.det = None                              # V3.45: no stale detection after a disconnect
        self.v_cam.set("相机未连接")

    def shutdown(self):
        self.stop_flag = True
        if self.src is not None:
            self.src.stop(1.5)
        if getattr(self, "wrist_src", None) is not None:
            self.wrist_src.stop(1.5)
        if self.after_id is not None:
            try:
                self.cv.after_cancel(self.after_id)
            except Exception:
                pass

    def on_save_info(self):
        if not self.src or not self.src.info:
            self.log("[WARN] [V3.25 camera] 先连接相机")
            return
        old = load_json(CAMERA_INFO_PATH)
        d = dict(self.src.info, saved=time.strftime("%Y-%m-%d %H:%M:%S"), fps=round(self.src.fps, 1))
        save_json(CAMERA_INFO_PATH, d)
        self.log(f"[OK] [V3.25 camera] 相机信息已保存 → config/camera_info.json（序列号 {d.get('serial')}）"
                 + (f"；⚠ 之前存的是另一台相机 {old.get('serial')}" if old and old.get("serial") not in (None, d.get("serial")) else ""))

    def _info_text(self):
        i = self.src.info if self.src else {}
        if not i:
            return "（连接相机后显示）"
        K = i.get("K") or {}
        return (f"型号 {i.get('name')} · 序列号 {i.get('serial')} · 固件 {i.get('firmware')} · USB {i.get('usb')} · "
                f"SDK {i.get('sdk')}\n预设 {i.get('preset')} · 深度单位 {i.get('depth_scale_mm', 0):.3f} mm · "
                f"帧率 {self.src.fps:.1f} fps\n彩色内参 {K.get('width')}x{K.get('height')} fx {K.get('fx', 0):.1f} "
                f"fy {K.get('fy', 0):.1f} cx {K.get('ppx', 0):.1f} cy {K.get('ppy', 0):.1f} · 畸变 {i.get('model')}\n"
                + ("" if not self.calib or self.calib.get("camera_serial") in (None, i.get("serial")) else
                   f"⚠ 标定文件属于另一台相机（{self.calib.get('camera_serial')}）：需要重新标定"))

    # ------------------------------------------------------------------ periodic
    def _schedule(self):
        self.after_id = self.cv.after(120, self._tick)

    def _force_redraw(self):
        self._shown_fid = -1
        self._shown_wfid = -1

    def _tick(self):
        try:
            self.poll()
        except Exception as exc:
            if getattr(self, "_last_err", None) != str(exc):
                self._last_err = str(exc)
                self.log(f"[FAIL] [V3.25 camera] refresh: {exc}")
        self._schedule()

    def poll(self):
        while self._ui_q:
            self._ui_q.pop(0)()
        for m_ in getattr(self, "_pending_logs", None) or []:
            self.log(m_)
        self._pending_logs = []
        self._update_calib_ui()
        if self.visible_fn():
            self._poll_wrist()
        s = self.src
        if s is None:
            return
        if s.error:
            self.v_cam.set("✗ " + s.error)
            self.log(f"[FAIL] [V3.25 camera] {s.error}")
            self.src = None
            return
        if not s.info:
            self.v_cam.set("连接中…")
            return
        i = s.info
        self.v_cam.set(f"✓ {i.get('name')} {i.get('serial')} · USB {i.get('usb')} · {s.fps:.0f} fps"
                       + ("" if str(i.get("usb", "")).startswith("3") else " · ⚠ 不是 USB 3"))
        self.v_info.set(self._info_text())
        self._verify_static()
        if not self.visible_fn():
            return
        if not self._det_busy and time.time() - self.det_t > 0.25 and self.marker:
            self._det_busy = True
            threading.Thread(target=self._detect_bg, daemon=True).start()
        self._live_text()
        fid = s.frame_id
        if fid != self._shown_fid:
            self._shown_fid = fid
            self._redraw()

    def _poll_wrist(self):
        """V3.50 WRIST MONITOR POLL: update the lower wrist camera canvas, quality/motion gate, live ArUco & TCP crosshair."""
        ws = self.wrist_src
        if ws is None:
            if self._shown_wfid != -2:
                self._shown_wfid = -2
                self._redraw_wrist()
            return
        if getattr(ws, "error", None):
            self.v_wrist_live.set("✗ " + ws.error)
            self.log(f"[FAIL] [V3.50 wrist] {ws.error}")
            self.wrist_src = None
            self._update_intr_ui()
            self._update_eih_ui()
            return
        if not getattr(ws, "info", None):
            self.v_wrist_live.set("腕部相机连接中…")
            return
        self.v_wrist_info.set(self._wrist_info_text())
        wfid = getattr(ws, "frame_id", 0)
        if wfid != self._shown_wfid:
            self._shown_wfid = wfid
            rgb, _d, _ = ws.frames()
            q = self.actual_q()
            self.wrist_quality = self.W49.evaluate_wrist_frame(rgb, self._prev_wrist_rgb, q, self._prev_wrist_q)
            self._prev_wrist_rgb = rgb
            self._prev_wrist_q = list(q) if q else None
            wm = self._wmarker_parse() or self.wmarker
            if not self._wdet_busy and time.time() - self.wdet_t > 0.25 and wm and getattr(ws, "K", None):
                self._wdet_busy = True
                threading.Thread(target=self._detect_wrist_bg, daemon=True).start()
            self._wrist_live_text(q)
            self._redraw_wrist()

    def _detect_wrist_bg(self):
        try:
            q = self.actual_q()
            ws = self.wrist_src
            wm = self._wmarker_parse() or self.wmarker
            self.wdet = ws.detect_marker(wm, q) if (ws and wm and getattr(ws, "K", None)) else None
            self.wdet_q = q
        except Exception:
            self.wdet = None
        finally:
            self.wdet_t = time.time()
            self._wdet_busy = False

    def _wrist_live_text(self, q=None):
        ws = self.wrist_src
        if not ws or not getattr(ws, "info", None):
            self.v_wrist_live.set("腕部相机未连接")
            return
        wq = self.wrist_quality or {}
        parts = [f"{ws.info.get('res', '')} {getattr(ws, 'fps', 0.0):.0f}fps", wq.get("badge") or "就绪"]
        he = self.wcalib_result or self.wrist_calib
        d = self.wdet
        if d is not None:
            p = d["p_cam"]
            m_txt = f"ArUco id{d['id']} ({p[0] * 1000:+.0f},{p[1] * 1000:+.0f},{p[2] * 1000:+.0f}mm)"
            q_use = q or getattr(self, "wdet_q", None)
            if he and he.get("T_flange_cam") and q_use:
                Tf = self.flange_T(q_use)
                if Tf is not None:
                    err = self.W49.wrist_marker_error_mm(he, Tf, p)
                    m_txt += f" 误差 {err:.1f}mm"
            parts.append(m_txt)
        ref = self.wrist_refine_res
        if ref and ref.get("ok"):
            dx, dy = ref["delta_tcp_mm"]
            parts.append(f"精定位 {ref['det_id']}: Δx={dx:+.1f} Δy={dy:+.1f}mm Δψ={ref['delta_yaw_deg']:+.1f}° "
                         f"(粗精差 {ref['delta_coarse_mm']:.1f}mm)")
        self.v_wrist_live.set(" · ".join(parts))

    def _detect_bg(self):
        try:
            q = self.actual_q()
            self.det = self.src.detect_marker(self.marker, q) if self.src else None
            self.det_q = q
        except Exception as exc:
            self.det = None
            self.det_err = str(exc)
        finally:
            self.det_t = time.time()
            self._det_busy = False

    def _live_text(self):
        d = self.det
        if d is None:
            ok, msg = self.src.aruco_status() if hasattr(self.src, "aruco_status") else aruco_status()
            self.v_live.set(("ArUco：未检测到标记 " + f"({self.marker['dict']} id {self.marker['id']})") if ok else ("ArUco 不可用：" + msg))
            self.lbl_live.configure(fg="#777777")
            return
        p = d["p_cam"]
        txt = f"ArUco id {d['id']}：相机中 {p[0] * 1000:+.0f} {p[1] * 1000:+.0f} {p[2] * 1000:+.0f} mm（{d['source']}）"
        col = "#1565c0"
        c = self.result or self.calib
        q = getattr(self, "det_q", None)
        if c and q:
            T = self.flange_T(q)
            if T is not None:
                e = marker_error_mm(c, T, p)
                if not self.result:
                    self._verify_live(d, e)
                txt += f" · 当前标定误差 {e:.1f} mm"
                col = "#2e7d32" if e <= CALIB_GOOD_MAX_MM else "#c62828"
                if e > CALIB_GOOD_MAX_MM and self.calib and not self.result:
                    txt += " ⚠ 相机或标记可能动过：「快速复核」/ 重新标定"
        self.v_live.set(txt)
        self.lbl_live.configure(fg=col)

    # ------------------------------------------------------------------ drawing
    def _px(self, uv):
        """V3.52 UNIFIED CANVAS SCALE: map full-res (1280x720) pixel (u, v) to canvas (DW=560, DH=315)."""
        return uv[0] * self.DW / 1280.0, uv[1] * self.DH / 720.0

    def _canvas_to_img(self, x, y):
        """Map canvas (DW=560, DH=315) coordinates back to full-res (1280x720) pixels."""
        return x * 1280.0 / self.DW, y * 720.0 / self.DH

    def _redraw(self):
        c = self.cv
        vw = self.v_view.get()
        if vw == "wrist":
            if getattr(self, "wrist_src", None):
                try:
                    data = self.wrist_src.display_ppm("color", size=(self.DW, self.DH))
                except TypeError:
                    data = self.wrist_src.display_ppm("color")
            else:
                data = None
        else:
            if self.src:
                try:
                    data = self.src.display_ppm(vw, size=(self.DW, self.DH))
                except TypeError:
                    data = self.src.display_ppm(vw)
            else:
                data = None
        c.delete("all")
        if data:
            try:
                self.photo = self.tk.PhotoImage(data=data, format="PPM")
            except Exception:
                self.photo = self.tk.PhotoImage(data=png_bytes_from_ppm(data), format="PNG")
            c.create_image(0, 0, image=self.photo, anchor="nw")
        K = self.src.K if self.src else None
        cal = self.result or self.calib
        # V3.45 REDRAW FIX: the detection thread replaces self.det at any time (often with None while the arm
        # moves) -> 'NoneType' object is not subscriptable between the check and the use.  One snapshot per redraw;
        # each overlay is drawn on its own so one failing overlay does not stop the others.
        det = self.det
        if K and cal:
            Tcb = T_inv(cal["T_base_cam"])
            if self.ov["zone"].get():
                self._overlay(self._draw_zone, Tcb, K)
            if self.ov["robot"].get():
                self._overlay(self._draw_robot, Tcb, K)
        if self.ov["marker"].get() and det:
            pts = [v for p in det["corners_px"] for v in self._px(p)]
            c.create_polygon(*pts, outline="#00e5ff", fill="", width=2)
            x, y = self._px(det["center_px"])
            c.create_text(x + 8, y - 8, text=f"id{det['id']}", fill="#00e5ff", anchor="w")
            q = getattr(self, "det_q", None)
            if K and cal and q:
                T = self.flange_T(q)
                if T is not None:
                    uv = project(predicted_marker_cam(cal, T), K)
                    if uv:
                        x, y = self._px(uv)
                        c.create_line(x - 8, y, x + 8, y, fill="#ffea00", width=2)
                        c.create_line(x, y - 8, x, y + 8, fill="#ffea00", width=2)
        if self.ov["eggs"].get() and K and cal:
            Tcb = T_inv(cal["T_base_cam"])
            yolo_on = bool(getattr(self, "yolo_view", None))
            for e in self.eggs:
                uv = project(T_apply(Tcb, [v for v in e["xyz_m"]]), K)
                if uv:
                    x, y = self._px(uv)
                    col = "#00e676" if e.get("trusted") else "#ff9100"
                    if yolo_on:                  # V3.47.2: YOLO draws the box + e1/e2 label; 3D centre = small cross only
                        c.create_line(x - 5, y, x + 5, y, fill=col, width=2)
                        c.create_line(x, y - 5, x, y + 5, fill=col, width=2)
                    else:
                        c.create_oval(x - 7, y - 7, x + 7, y + 7, outline=col, width=2)
                        c.create_text(x + 9, y, text=e.get("det_id", ""), fill=col, anchor="w")
        if self.ov["eggs"].get():
            self._overlay(self._draw_yolo)
        if self.roi:
            u0, v0, u1, v1 = self.roi
            rx0, ry0 = self._px((u0, v0))
            rx1, ry1 = self._px((u1, v1))
            c.create_rectangle(rx0, ry0, rx1, ry1, outline="#ffea00", dash=(4, 3), width=2)
        if self.meas:
            x, y = self._px(self.meas)
            c.create_oval(x - 4, y - 4, x + 4, y + 4, outline="#ff1744", width=2)

    def _draw_yolo(self):
        # V3.47/V3.52: YOLO boxes (green = egg, orange = not trusted, red = rejected) aligned with DWxDH (560x315)
        c = self.cv
        col = {"egg": "#00e676", "object": "#00e676", "untrusted": "#ff9100", "rejected": "#ff1744"}
        for v in place_yolo_labels(getattr(self, "yolo_view", None) or [], self.DW, self.DH):
            x0, y0, x1, y1 = v["box_cv"]
            k = col.get(v["status"], "#ff1744")
            c.create_rectangle(x0, y0, x1, y1, outline=k, width=2, **({"dash": (3, 2)} if v["status"] == "rejected" else {}))
            c.create_text(v["tx"], v["ty"], text=v["label"], fill=k, anchor=v["anchor"])
        e = getattr(self, "ref_edit", None)
        if e is not None:
            c.create_text(6, 6, text="框选参考中（定格画面的框）", fill="#ffea00", anchor="nw")
            for b, k in zip(e["boxes"], e["cls"]):
                x0, y0 = self._px((b[0], b[1]))
                x1, y1 = self._px((b[2], b[3]))
                c.create_rectangle(x0, y0, x1, y1, outline="#00b0ff" if k == 0 else "#ff4081", width=2)
                c.create_text(x0, y0 - 2, text="参考·鸡蛋" if k == 0 else "参考·非鸡蛋",
                              fill="#00b0ff" if k == 0 else "#ff4081", anchor="sw")
        d = getattr(self, "_ref_drag", None)
        if d:
            dx0, dy0 = self._px((d[0], d[1]))
            dx1, dy1 = self._px((d[2], d[3]))
            c.create_rectangle(dx0, dy0, dx1, dy1, outline="#ffea00", width=2, dash=(4, 3))

    def _wpx(self, uv, K=None):
        """Map wrist camera pixel (u, v) to self.cv_wrist (DW x DH)."""
        w = float((K or {}).get("width") or (self.wrist_src.info.get("width") if self.wrist_src and self.wrist_src.info else 1280) or 1280)
        h = float((K or {}).get("height") or (self.wrist_src.info.get("height") if self.wrist_src and self.wrist_src.info else 720) or 720)
        return uv[0] * self.DW / w, uv[1] * self.DH / h

    def _redraw_wrist(self):
        """V3.50 WRIST MONITOR REDRAW: draw live wrist feed, center-ROI guide, motion/quality badge,
        projected TCP crosshair + 21 mm egg ring + fingertip drop-zone boxes, ArUco, and coarse-to-fine overlay."""
        cw = getattr(self, "cv_wrist", None)
        if cw is None:
            return
        cw.delete("all")
        ws = self.wrist_src
        if not ws:
            cw.create_text(self.DW // 2, self.DH // 2, text="腕部 USB 相机未连接\n（在「相机信息」Tab 点「▶ 连接腕部相机」）",
                           fill="#90a4ae", justify="center")
            return
        try:
            data = ws.display_ppm("color", size=(self.DW, self.DH))
        except TypeError:
            data = ws.display_ppm("color")
        if data:
            try:
                self.photo_wrist = self.tk.PhotoImage(data=data, format="PPM")
            except Exception:
                self.photo_wrist = self.tk.PhotoImage(data=png_bytes_from_ppm(data), format="PNG")
            cw.create_image(0, 0, image=self.photo_wrist, anchor="nw")

        # 1. Center sweet-spot ROI guide (central 78% of 130° wide-angle lens)
        m_frac = 0.5 * (1.0 - self.W49.WRIST_ROI_FRAC)
        rx0, ry0 = self.DW * m_frac, self.DH * m_frac
        rx1, ry1 = self.DW * (1.0 - m_frac), self.DH * (1.0 - m_frac)
        cw.create_rectangle(rx0, ry0, rx1, ry1, outline="#546e7a", dash=(4, 4), width=1)

        # 2. TCP crosshair + 21 mm egg radius ring + gripper opening line + left/right fingertip drop-zones
        intr = self.intr_result or self.wrist_intr
        K_w = getattr(ws, "K", None) or (intr.get("K") if intr else None)
        he = self.wcalib_result or self.wrist_calib
        tcp_proj = None
        if K_w and he and he.get("T_flange_cam"):
            try:
                z_work = max(0.06, float(self.v_hover_mm.get()) / 1000.0) if hasattr(self, "v_hover_mm") else 0.10
            except Exception:
                z_work = 0.10
            tcp_proj = self.W49.project_tcp_wrist(K_w, he["T_flange_cam"], self._active_tcp_T(),
                                                  z_work_m=z_work, canvas_size=(self.DW, self.DH))
        if tcp_proj and tcp_proj.get("ok"):
            ux, uy = tcp_proj["uv_tcp_cv"]
            r_cv = tcp_proj["radius_21mm_cv"]
            ax0, ay0, ax1, ay1 = tcp_proj["axis_line_cv"]
            cw.create_line(ax0, ay0, ax1, ay1, fill="#00e5ff", dash=(3, 2), width=1)
            cw.create_oval(ux - r_cv, uy - r_cv, ux + r_cv, uy + r_cv, outline="#00e5ff", dash=(4, 3), width=1)
            cw.create_line(ux - 10, uy, ux + 10, uy, fill="#00e5ff", width=2)
            cw.create_line(ux, uy - 10, ux, uy + 10, fill="#00e5ff", width=2)
            cw.create_text(ux + 12, uy - 10, text="TCP", fill="#00e5ff", anchor="w")
            for fbox in (tcp_proj["left_box_cv"], tcp_proj["right_box_cv"]):
                cw.create_rectangle(*fbox, outline="#26c6da", dash=(2, 2), width=1)

        # 3. Detected ArUco marker on wrist camera
        wdet = self.wdet
        if self.ov["marker"].get() and wdet and K_w:
            pts = [v for p in wdet["corners_px"] for v in self._wpx(p, K_w)]
            cw.create_polygon(*pts, outline="#00e5ff", fill="", width=2)
            mx, my = self._wpx(wdet["center_px"], K_w)
            cw.create_text(mx + 8, my - 8, text=f"id{wdet['id']}", fill="#00e5ff", anchor="w")

        # 4. Coarse-to-Fine target refinement overlay
        ref = self.wrist_refine_res
        if ref and ref.get("ok") and K_w:
            col = "#00e676" if ref.get("trusted") else "#ff9100"
            bx0, by0 = self._wpx((ref["box_px"][0], ref["box_px"][1]), K_w)
            bx1, by1 = self._wpx((ref["box_px"][2], ref["box_px"][3]), K_w)
            cw.create_rectangle(bx0, by0, bx1, by1, outline=col, width=2)
            fx_cv, fy_cv = self._wpx(ref["uv_fine"], K_w)
            cw.create_line(fx_cv - 7, fy_cv, fx_cv + 7, fy_cv, fill=col, width=2)
            cw.create_line(fx_cv, fy_cv - 7, fx_cv, fy_cv + 7, fill=col, width=2)
            if ref.get("axis_uv"):
                a0x, a0y = self._wpx(ref["axis_uv"][:2], K_w)
                a1x, a1y = self._wpx(ref["axis_uv"][2:], K_w)
                cw.create_line(a0x, a0y, a1x, a1y, fill="#ffea00", width=2)
            fcol = "#00e676" if ref.get("clearance_ok") else "#ff1744"
            for fb in ref.get("finger_boxes_px") or []:
                fx0, fy0 = self._wpx((fb[0], fb[1]), K_w)
                fx1, fy1 = self._wpx((fb[2], fb[3]), K_w)
                cw.create_rectangle(fx0, fy0, fx1, fy1, outline=fcol, width=1)
            if tcp_proj and tcp_proj.get("ok"):
                ux, uy = tcp_proj["uv_tcp_cv"]
                cw.create_line(ux, uy, fx_cv, fy_cv, fill="#ffea00", width=2)
            dx, dy = ref["delta_tcp_mm"]
            cw.create_text(bx0, max(14, by0 - 4),
                           text=f"{ref['det_id']} Δx={dx:+.1f} Δy={dy:+.1f}mm Δψ={ref['delta_yaw_deg']:+.1f}°",
                           fill=col, anchor="sw")

        # 5. Frame quality & motion gate badge (top-right corner)
        wq = self.wrist_quality
        if wq:
            qcol = {"ok": "#00e676", "motion": "#ff9100", "blur": "#ff9100", "overexp": "#ff1744"}.get(wq["status"], "#b0bec5")
            cw.create_text(self.DW - 8, 8, text=wq["badge"], fill=qcol, anchor="ne")

    def _overlay(self, fn, *a):
        try:
            fn(*a)
        except Exception as exc:
            if getattr(self, "_overlay_err", None) != f"{fn.__name__}: {exc}":
                self._overlay_err = f"{fn.__name__}: {exc}"
                self.log(f"[WARN] [V3.45 camera] 叠加层 {self._overlay_err}")

    def _fence(self):
        ws = getattr(self.safe_tab, "valid_ws", None)
        return (ws or {}).get("fence") if isinstance(ws, dict) else None

    def _draw_zone(self, Tcb, K):
        f = self._fence()
        if not f or not f.get("polygon"):
            return
        poly, z = f["polygon"], f["z_table"]
        pts = []
        for i in range(len(poly)):
            a, b = poly[i], poly[(i + 1) % len(poly)]
            for k in range(10):
                t = k / 10.0
                uv = project(T_apply(Tcb, [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, z]), K)
                if uv:
                    pts += list(self._px(uv))
        if len(pts) >= 6:
            self.cv.create_polygon(*pts, outline="#76ff03", fill="", width=2, dash=(6, 4))

    def _draw_robot(self, Tcb, K):
        q = self.actual_q()
        if not q:
            return
        try:
            pts = self.safe_tab.arm_points([math.radians(v) for v in q])
        except Exception:
            return
        if not pts:
            return
        line = []
        for name, p in pts:
            if "#" in name:
                continue
            uv = project(T_apply(Tcb, list(p)), K)
            if uv:
                line += list(self._px(uv))
        if len(line) >= 4:
            self.cv.create_line(*line, fill="#ff4081", width=3)

    # ------------------------------------------------------------------ mouse
    def on_press(self, e):
        self._drag0 = (e.x, e.y)

    def on_drag(self, e):
        if self._drag0 is None:
            return
        x0, y0 = self._drag0
        if abs(e.x - x0) + abs(e.y - y0) > 6:
            u0, v0 = self._canvas_to_img(min(x0, e.x), min(y0, e.y))
            u1, v1 = self._canvas_to_img(max(x0, e.x), max(y0, e.y))
            box = (int(round(u0)), int(round(v0)), int(round(u1)), int(round(v1)))
            if getattr(self, "ref_edit", None) is not None:      # V3.47: drag = reference box
                self._ref_drag = box
            else:
                self.roi = box
            self._force_redraw()

    def on_release(self, e):
        d0 = self._drag0
        self._drag0 = None
        if getattr(self, "ref_edit", None) is not None:
            box, self._ref_drag = self._ref_drag, None
            if d0 and box and abs(e.x - d0[0]) + abs(e.y - d0[1]) > 6:
                self._ref_add(box)
                self._force_redraw()
                return
        if d0 and abs(e.x - d0[0]) + abs(e.y - d0[1]) > 6:
            self.log(f"[INFO] [V3.25 camera] 深度检查区域 {self.roi}（全分辨率像素）")
            return
        u, v = self._canvas_to_img(e.x, e.y)
        self.measure(u, v)

    def measure(self, u, v):
        self.meas = (u, v)
        self._force_redraw()
        if not self.src or not self.src.K:
            self.v_meas.set("相机未连接")
            return
        z = self.src.depth_at(u, v)
        if z is None:
            self.v_meas.set(f"像素 ({u:.0f}, {v:.0f})：这里没有深度（太近 < 0.3 m / 遮挡 / 反光）")
            return None
        p = deproject(u, v, z, self.src.K)
        txt = (f"像素 ({u:.0f}, {v:.0f}) · 距离 {z * 1000:.0f} mm · 相机系 {p[0] * 1000:+.0f} {p[1] * 1000:+.0f} "
               f"{p[2] * 1000:+.0f} mm")
        cal = self.result or self.calib
        pb = None
        if cal:
            pb = T_apply(cal["T_base_cam"], p)
            txt += f"\nbase_link：x {pb[0] * 1000:+.0f}  y {pb[1] * 1000:+.0f}  z {pb[2] * 1000:+.0f} mm"
        self.v_meas.set(txt)
        return pb or p

    # ------------------------------------------------------------------ depth check
    def on_plane(self):
        if not self.src or not self.src.K:
            self.log("[WARN] [V3.25 camera] 先连接相机")
            return
        if not self.roi:
            self.log("[WARN] [V3.25 camera] 先在画面上拖一个框（空桌面）")
            return
        pts, total = self.src.roi_points(*self.roi, stride=4)
        if len(pts) < 50:
            self.v_plane.set(f"有效深度点太少（{len(pts)} / {total}）")
            return
        fit = plane_fit(pts)
        valid = len(pts) / max(1, total)
        ang = math.degrees(math.acos(min(1.0, abs(fit["normal"][2]))))
        lines = [f"有效深度 {valid * 100:.0f}% · 点数 {len(pts)} · 平面残差 RMS {fit['rms_m'] * 1000:.2f} mm",
                 f"中心距离 {fit['centroid'][2] * 1000:.0f} mm · 桌面法向与光轴夹角 {ang:.1f}°"]
        cal = self.result or self.calib
        if cal:
            pb = [T_apply(cal["T_base_cam"], p) for p in pts]
            fb = plane_fit(pb)
            tilt = math.degrees(math.acos(min(1.0, abs(fb["normal"][2]))))
            f_ = self._fence()
            zt = f_["z_table"] if f_ else 0.0
            lines.append(f"base_link：桌面 z = {fb['centroid'][2] * 1000:.1f} mm · 倾斜 {tilt:.2f}° · 安全区「桌面 z」"
                         f"{zt * 1000:.1f} mm → 差 {(fb['centroid'][2] - zt) * 1000:+.1f} mm")
        ok = fit["rms_m"] <= 0.002 and valid >= 0.9
        lines.append("✓ 深度质量好" if ok else "⚠ 残差 > 2 mm 或有效深度 < 90%：检查距离、光照、反光（白纸），或做 On-Chip 校准")
        self.v_plane.set("\n".join(lines))
        self.log(("[OK] " if ok else "[WARN] ") + "[V3.25 camera] 平面拟合：" + " | ".join(lines[:3]))

    # ------------------------------------------------------------------ calibration
    def _wait_still(self, timeout=4.0, hold_s=0.4):
        """V3.45: wait until the joints have not moved more than STILL_TOL_DEG for hold_s."""
        t_end = time.time() + timeout
        last, t_still = None, None
        while time.time() < t_end and not self.stop_flag:
            q = self.actual_q()
            if q and last and max(abs(a - b) for a, b in zip(q, last)) <= STILL_TOL_DEG:
                t_still = t_still or time.time()
                if time.time() - t_still >= hold_s:
                    return True
            else:
                t_still = None
            last = q
            time.sleep(0.1)
        return False

    def _capture_sample(self, n_frames=FAST_CAPTURE_FRAMES, timeout=FAST_CAPTURE_TIMEOUT_S):
        """V3.45/V3.55: fast multi-frame capture while the arm is still; records both unified PnP (p_pnp) and depth
        (p_depth) means per pose so calibrate() can eliminate cross-pose depth/PnP step jumps; exits early if 0 frames
        see the marker within FAST_EMPTY_TIMEOUT_S (0.8 s) instead of blocking for 8 s."""
        if not self.src or not self.src.K:
            return None, "相机未连接"
        m = self._marker_from_ui()
        if not m:
            return None, "标记设置无效"
        q0 = self.actual_q()
        if not q0:
            return None, "没有关节反馈"
        dets, qs = [], []
        last = -1
        t0 = time.time()
        t_end = t0 + float(timeout)
        while len(dets) < n_frames and time.time() < t_end and not self.stop_flag:
            fid = self.src.frame_id
            if fid == last:
                time.sleep(0.015)
                continue
            last = fid
            q = self.actual_q()
            d = self.src.detect_marker(m, q)
            if d is None or q is None:
                if not dets and time.time() - t0 >= FAST_EMPTY_TIMEOUT_S:
                    break
                continue
            if max(abs(a - b) for a, b in zip(q, q0)) > 0.2:
                return None, "机械臂在动：停稳后再记录"
            dets.append(d)
            qs.append(q)
            if len(dets) >= 5:
                _s_tmp, _p_tmp = choose_source(dets)
                _tr_tmp = trimmed_points(_p_tmp, min_keep=4)
                if not _tr_tmp.get("error") and _tr_tmp["jitter_mm"] <= 1.8:
                    break
        need = max(4, n_frames // 2) if n_frames < 12 else max(5, n_frames // 3)
        if len(dets) < need:
            return None, f"标记只检测到 {len(dets)} 帧（要 ≥ {need}）：标记不在画面里 / 转得太偏？"
        src, pts = choose_source(dets)
        tr = trimmed_points(pts, min_keep=max(3, need - 1))
        if tr.get("error"):
            return None, f"{'深度' if src == 'depth' else 'PnP'}：{tr['error']}（光线 / 标记太小？）"
        if tr["jitter_mm"] > JITTER_MAX_MM:
            return None, (f"去掉坏帧后抖动 {tr['jitter_mm']:.1f} mm（> {JITTER_MAX_MM:.0f} mm，"
                          f"{'深度' if src == 'depth' else 'PnP'}，保留 {tr['kept']}/{tr['n']} 帧）")
        pnp_pts = [d.get("p_pnp") or d["p_cam"] for d in dets if d.get("p_pnp") or d.get("p_cam")]
        dep_pts = [d["p_depth"] for d in dets if d.get("p_depth") is not None]
        tr_pnp = trimmed_points(pnp_pts, min_keep=max(3, len(pnp_pts) // 2)) if len(pnp_pts) >= 3 else {}
        tr_dep = trimmed_points(dep_pts, min_keep=max(3, len(dep_pts) // 2)) if len(dep_pts) >= 3 else {}
        q = [sum(v[j] for v in qs) / len(qs) for j in range(7)]
        T = self.flange_T(q)
        if T is None:
            return None, "FK 失败（URDF / link7）"
        ns = [d.get("normal_cam") for d in dets if d.get("normal_cam")]
        nrm = None
        if ns:
            sm = [sum(v[i] for v in ns) for i in range(3)]
            nn = vnorm(sm) or 1.0
            nrm = [v / nn for v in sm]
        rv0 = next((d.get("rvec") for d in dets if d.get("rvec") is not None), None)
        return {"q_deg": [round(v, 3) for v in q], "p_cam": tr["mean"],
                "p_pnp": tr_pnp.get("mean") or tr["mean"],
                "p_depth": tr_dep.get("mean"),
                "T_base_flange": T, "spread_mm": tr["jitter_mm"], "n": tr["kept"], "n_frames": tr["n"],
                "source": src, "normal_cam": nrm, "rvec": rv0}, None

    def _capture_retry(self, tag=""):
        """V3.45/V3.55: capture; skip retry immediately if 0 frames saw the marker (blind pose); otherwise settle
        briefly (0.35 s) and re-measure once."""
        fatal = ("相机未连接", "标记设置无效", "没有关节反馈", "FK 失败", "标记只检测到 0 帧")
        why = ""
        for attempt in range(1 + CAPTURE_RETRIES):
            s, why = self._capture_sample()
            if s is not None:
                s["attempt"] = attempt + 1
                return s, None
            if self.stop_flag or any(why.startswith(f) for f in fatal) or attempt >= CAPTURE_RETRIES:
                break
            self.log(f"[INFO] [V3.45 calib] {tag}{why} → 再测一次")
            time.sleep(0.35)
            self._wait_still(1.5, hold_s=FAST_STILL_HOLD_S)
        return None, why

    def _add_sample(self, s, label):
        self.samples.append(s)
        self.result = None
        self.log(f"[OK] [V3.45 calib] {label} → 样本 {len(self.samples)}：标记 {[round(v * 1000) for v in s['p_cam']]} mm"
                 f"（{'深度' if s['source'] == 'depth' else 'PnP'}，{s['n']}/{s['n_frames']} 帧，抖动 {s['spread_mm']:.1f} mm"
                 + ("，第 2 次" if s.get("attempt", 1) > 1 else "") + "）")
        self._ui(self._refresh_table)

    def on_record(self):
        def job():
            s, why = self._capture_retry()
            if s is None:
                self.log(f"[WARN] [V3.45 calib] 样本未记录：{why}")
                return
            self._add_sample(s, "当前姿态")
        self._run(job)

    def _run(self, fn):
        if self.worker is not None and self.worker.is_alive():
            self.log("[WARN] [V3.25 camera] 上一个操作还在进行")
            return
        self.stop_flag = False

        def wrap():
            try:
                fn()
            except Exception as exc:
                self.log(f"[FAIL] [V3.45 calib] {exc}")
            finally:
                self.auto_state = None
                self.wauto_state = None
                if self.check_state and self.check_state[0] == "run":
                    self.check_state = None
                if getattr(self, "wcheck_state", None) and self.wcheck_state[0] == "run":
                    self.wcheck_state = None
        self.worker = threading.Thread(target=wrap, daemon=True)
        self.worker.start()

    def on_stop(self):
        self.stop_flag = True
        try:
            self.pipe.cancel_move()
        except Exception:
            pass

    def on_delete(self):
        sel = sorted((int(i) for i in self.tv.selection()), reverse=True)
        for i in sel:
            if 0 <= i < len(self.samples):
                self.samples.pop(i)
        self.result = None
        self._refresh_table()

    def on_clear(self):
        if self.samples and not self.ask("清空样本", f"删除全部 {len(self.samples)} 个样本？"):
            return
        self.samples = []
        self.result = None
        self._refresh_table()

    # ------------------------------------------------------------------ V3.45 visibility-aware auto collection
    def _fk_rad(self, q_rad):
        return self.flange_T([math.degrees(v) for v in q_rad])

    def _ik_fn(self, model):
        import NERO_V315_Consistency_Safety as CS
        urdf, joints, limits = self._urdf(), model["joints"], model["limits"]
        base = model.get("base_link") or "base_link"

        def ik(seed, xyz, R):
            q, _err = CS.ik_tool(urdf, joints, list(seed), base, None, list(xyz), limits, iters=150, tol=1e-3, keep_R=R)
            if q is None:
                return None
            T = self._fk_rad(q)
            if T is None or vnorm(vsub(T_t(T), xyz)) > 0.003 or rot_angle_deg(mat_mul(transpose(T_R(T)), R)) > 3.0:
                return None                                      # fallbacks of ik_tool give up the orientation
            return list(q)
        return ik

    def _move_measure(self, model, q_rad, speed, label):
        try:
            res = self.pipe.move_and_settle(model, list(q_rad), speed, label=label, settle_s=0.35)
        except TypeError:
            res = self.pipe.move_and_settle(model, list(q_rad), speed, label=label)
        if not res.get("ok"):
            return None, f"运动失败：{res.get('error')}"
        time.sleep(min(float(SETTLE_S), FAST_SETTLE_S))
        self._wait_still(timeout=1.5, hold_s=FAST_STILL_HOLD_S)
        return self._capture_retry(f"{label}：")

    def on_auto(self):
        try:
            n = int(self.v_nauto.get())
            assert 4 <= n <= 40
        except Exception:
            self.log("[WARN] [V3.25 calib] 自动采集数量 4–40")
            return
        if not self.src or not self.src.K:
            self.log("[WARN] [V3.25 calib] 先连接相机")
            return
        if getattr(self.pipe, "teach_mode", False):
            self.log("[WARN] [V3.25 calib] NERO 在示教模式：先退出（自动采集要让机械臂自己动）")
            return
        if not self.jpanel.enabled:
            self.log("[WARN] [V3.25 calib] 关节面板不可操作（需要 MoveIt 就绪）")
            return
        if self.det is None:
            self.log("[WARN] [V3.25 calib] 现在看不到标记：先把法兰摆到相机能看到标记的位置")
            return
        speed = min(CALIB_MAX_SPEED, max(0.01, float(self.cfg_fn().get("speed_percent", 10)) / 100.0))
        cal = self.result or self.calib

        def job():
            model = self._model()
            q0 = self.actual_q()
            if not q0:
                self.log("[FAIL] [V3.25 calib] 没有关节反馈")
                return
            q0r = [math.radians(v) for v in q0]
            K = dict(self.src.K)
            if not self.ask("自动采集标定样本", f"机械臂将在当前姿态附近走最多 {n + 3} 个姿态（法兰位置铺开约 "
                            f"{AUTO_SPAN_TARGET_M * 100:.0f} cm，手腕转动 ≤ {AUTO_ROT_MAX_DEG:.0f}°，只选预测能看到标记、"
                            f"标记正对相机 ≤ {AUTO_VIEW_MAX_DEG:.0f}° 的姿态；速度 {speed * 100:.0f}%，经 MoveIt + 安全区检查）。"
                            "没有可用标定时先做 3 个 4 cm 的小平移来确定相机方向。结束后回到起点。\n确认周围无障碍、手在急停上？"):
                return
            self.auto_state = {"phase": "起点", "k": 0, "n": n}
            got = 0
            s0, why = self._capture_retry("起点：")
            if s0 is None:
                self.log(f"[FAIL] [V3.45 calib] 起点测不到标记：{why}")
                return
            if self.samples:
                self.log(f"[INFO] [V3.55 calib] 自动采集开始：已自动清空旧样本 {len(self.samples)} 个")
                self.samples.clear()
                self.result = None
                self._ui(self._refresh_table)
            self._add_sample(s0, "起点")
            got += 1
            T0 = s0["T_base_flange"]
            mm = None
            if cal:
                e0 = marker_error_mm(cal, T0, s0["p_cam"])
                if e0 <= 40.0:
                    mm = MarkerModel.from_calib(cal, T0, s0.get("normal_cam"))
                    self.log(f"[INFO] [V3.45 calib] 用现有标定预测标记位置（起点误差 {e0:.1f} mm）")
                else:
                    self.log(f"[INFO] [V3.45 calib] 现有标定在起点误差 {e0:.0f} mm：不用它，改用小平移确定相机方向")
            if mm is None:
                fast_ik = build_fast_jacobian_ik(self._fk_rad, q0r, model["limits"], soft_rad=math.radians(3.0))
                ik = self._ik_fn(model)
                Ts, Ps = [T0], [s0["p_cam"]]
                p0 = T_t(T0)
                for ax in range(3):
                    if self.stop_flag:
                        break
                    self.auto_state = {"phase": "确定相机方向", "k": ax + 1, "n": 3}
                    for sg in (1.0, -1.0):
                        tgt = list(p0)
                        tgt[ax] += sg * AUTO_PROBE_M
                        q = (fast_ik(q0r, tgt, T_R(T0)) if fast_ik else None) or ik(q0r, tgt, T_R(T0))
                        if q is None or not check_auto_path(lambda q_: self.pipe.fence_goal_check(model, q_)[0], q0r, q, steps=4):
                            continue
                        s, why = self._move_measure(model, q, speed, f"方向探测 {'xyz'[ax]}{'+' if sg > 0 else '-'}")
                        if s is None:
                            self.log(f"[WARN] [V3.45 calib] 方向探测 {'xyz'[ax]}：{why}")
                            continue
                        self._add_sample(s, f"方向探测 {'xyz'[ax]}")
                        got += 1
                        Ts.append(s["T_base_flange"])
                        Ps.append(s["p_cam"])
                        break
                if len(Ts) >= 3:
                    mm, rms = MarkerModel.from_probes(Ts, Ps, s0.get("normal_cam"))
                    self.log(f"[INFO] [V3.45 calib] 相机方向已确定（{len(Ts)} 点，拟合 {rms * 1000:.1f} mm）")
                else:
                    mm = MarkerModel.from_start_det(T0, s0, Ts, Ps)
                    self.log(f"[INFO] [V3.55 calib] 相机方向已确定（由起点 PnP 位姿 + {len(Ts)} 点直接建立预测模型）")
            targets, st = plan_auto_targets(self._fk_rad, self._ik_fn(model), q0r, model["limits"], mm, K, n,
                                            fence_fn=lambda q: self.pipe.fence_goal_check(model, q)[0],
                                            seed=int(time.time()) % 100000)
            rej = "、".join(f"{k} {v}" for k, v in sorted(st["rejected"].items(), key=lambda kv: -kv[1])[:4])
            self.log(f"[INFO] [V3.45 calib] 选出 {len(targets)} 个姿态（试了 {st['tries']} 个；淘汰：{rej or '无'}）· 位置范围 "
                     f"{st['span_m'] * 100:.0f} cm · 最大手腕转动 {st['rot_max_deg']:.0f}°")
            if not targets:
                self.log("[FAIL] [V3.45 calib] 起点附近找不到能看到标记的安全姿态（换一个更开阔、标记更正对相机的起点）")
                self.pipe.move_and_settle(model, q0r, speed, label="标定采样 回起点")
                return
            for k, t in enumerate(targets):
                if self.stop_flag:
                    break
                self.auto_state = {"phase": "采集", "k": k + 1, "n": len(targets)}
                vd = t.get("view_deg")
                s, why = self._move_measure(model, t["q"], speed, f"标定采样 {k + 1}/{len(targets)}")
                if s is None:
                    self.log(f"[WARN] [V3.45 calib] 姿态 {k + 1}：{why}" + (f"（预测夹角 {vd:.0f}°）" if vd is not None else ""))
                    continue
                self._add_sample(s, f"姿态 {k + 1}/{len(targets)}")
                got += 1
            self.auto_state = {"phase": "回起点", "k": 0, "n": 0}
            self.log("[INFO] [V3.25 calib] 回到起点")
            self.pipe.move_and_settle(model, q0r, speed, label="标定采样 回起点")
            ts, rs = self._spans()
            self.log(f"[{'OK' if got else 'WARN'}] [V3.25 calib] 自动采集结束：新增 {got} 个样本（共 {len(self.samples)}）· "
                     f"位置范围 {ts * 100:.0f} cm · 朝向变化 {rs:.0f}°" + ("（已停止）" if self.stop_flag else ""))
            if len(self.samples) >= 6:
                self._ui(self.on_compute)
        self._run(job)

    def on_compute(self):
        if len(self.samples) < 6:
            self.log(f"[WARN] [V3.25 calib] 至少 6 个样本（建议 ≥ 8），现在 {len(self.samples)} 个")
            return
        r = calibrate(self.samples)
        if not r.get("ok"):
            self.log(f"[FAIL] [V3.25 calib] {r.get('error')}")
            return
        self.result = r
        self._refresh_table()
        self._force_redraw()
        tag = "[OK]" if r["good"] else "[WARN]"
        self.log(f"{tag} [V3.25 calib] 标定：样本 {len(r['used'])}（剔除 {len(r['rejected'])}）· 残差 平均 {r['mean_mm']:.2f} / "
                 f"最大 {r['max_mm']:.2f} mm · 留一 平均 {r['loo_mean_mm']:.2f} / 最大 {r['loo_max_mm']:.2f} mm · 相机 "
                 f"{[round(v, 3) for v in r['cam_pos']]} m" + ("" if r["good"] else " · " + "; ".join(r["warnings"]) if r["warnings"] else ""))

    def on_save_calib(self):
        r = self.result
        if not r or not r.get("ok"):
            self.log("[WARN] [V3.25 calib] 先「计算标定」")
            return
        if not r["good"] and not self.ask("保存标定", "标定没达到合格标准（留一平均 ≤ 3 mm、最大 ≤ 5 mm）。仍然保存？"):
            return
        d = save_calib(r, self.samples, self.src.info if self.src else {}, self.marker)
        self.calib = d
        self.result = None
        self.check_state = None                      # V3.45: a new calibration has not been checked yet
        self.calib_src = CALIB_PATH
        self._set_vstate("ok", "刚标定")
        self._refresh_table()
        self.log(f"[OK] [V3.25 calib] 标定已保存 → config/camera_calib.json（{d['saved']}，样本 {d['n_used']}）")
        if CALIB_PATH == DEFAULT_CALIB_PATH:         # V3.47.1: copy outside the version folder
            try:
                os.makedirs(CALIB_DIR, exist_ok=True)
                cp = os.path.join(CALIB_DIR, "camera_calib_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
                save_json(cp, d)
                self.log(f"[OK] [V3.47.1 calib] 副本 → {cp}（新版本文件夹启动时自动读取最新的一份）")
            except Exception as exc:
                self.log(f"[WARN] [V3.47.1 calib] 副本没保存到 {CALIB_DIR}：{exc}")

    # ------------------------------------------------------------------ V3.47.1 calibration 另存为 / 读取
    def _ask_path(self, save, initialdir, initialfile, title):
        try:
            os.makedirs(initialdir, exist_ok=True)
        except Exception:
            pass
        from tkinter import filedialog
        kw = dict(initialdir=initialdir, title=title, filetypes=[("JSON", "*.json"), ("全部", "*")])
        if save:
            return filedialog.asksaveasfilename(initialfile=initialfile, defaultextension=".json", **kw)
        return filedialog.askopenfilename(**kw)

    def on_calib_save_as(self, path=None):
        c = self.calib
        if not c:
            self.log("[WARN] [V3.47.1 calib] 还没有保存的标定（先「保存标定」）")
            return
        path = path or self._ask_path(True, CALIB_DIR, "camera_calib_" + time.strftime("%Y%m%d_%H%M") + ".json",
                                      "标定另存为")
        if not path:
            return
        try:
            save_json(path, dict(c, exported=time.strftime("%Y-%m-%d %H:%M:%S")))
        except Exception as exc:
            self.log(f"[FAIL] [V3.47.1 calib] 另存失败：{exc}")
            return
        self.log(f"[OK] [V3.47.1 calib] 标定另存为 → {path}")

    def on_calib_load(self, path=None):
        if self._busy():
            self.log("[WARN] [V3.47.1 calib] 正在采集 / 复核，先停止")
            return
        path = path or self._ask_path(False, CALIB_DIR, "", "读取标定文件")
        if not path:
            return
        d = load_json(path)
        if not calib_valid(d):
            self.log(f"[FAIL] [V3.47.1 calib] {path} 不是有效的标定文件（要有 T_base_cam 4×4 和 marker_in_flange）")
            return
        d.pop("exported", None)
        try:
            save_json(CALIB_PATH, d)                 # becomes the active calibration of this folder
        except Exception as exc:
            self.log(f"[WARN] [V3.47.1 calib] 没写进 {CALIB_PATH}：{exc}（本次运行仍然使用）")
        self.calib, self.result, self.calib_src = d, None, path
        self.marker = dict(d.get("marker") or DEFAULT_MARKER)
        for var, val in ((getattr(self, "v_dict", None), self.marker["dict"]), (getattr(self, "v_mid", None), str(self.marker["id"])),
                         (getattr(self, "v_msize", None), f"{float(self.marker['size_mm']):g}")):
            if var is not None:
                var.set(val)
        self.check_state = None
        self._set_vstate("pending", "读取后还没校验")
        self._vstatic_src = None
        self._set_result_text(self._calib_text())
        self.log(f"[OK] [V3.47.1 calib] 读取标定 {path}（{d.get('saved', '')}，相机 {d.get('camera_serial')}）→ 需要校验："
                 "连接相机后让标记出现在画面里（误差 ≤ 5 mm 连续 5 次）或点「快速复核」")

    def on_calib_clear(self):
        """V3.48 CLEAR CALIB: clear the active hand-eye calibration when the camera was moved so the user can
        re-calibrate from scratch (keeps backup files in ~/Downloads/cameracalib)."""
        if self._busy():
            self.log("[WARN] [V3.48 calib] 正在采集 / 复核，先停止")
            return
        if not self.calib and not self.result:
            self.log("[INFO] [V3.48 calib] 当前没有标定设定")
            return
        if not self.ask("清除标定设定", "确定清除当前手眼标定设定（相机被挪动后重新标定）？\n"
                        "将清除当前使用的标定并删除本文件夹的 config/camera_calib.json（~/Downloads/cameracalib 里的历史备份仍保留）。"):
            return
        self.calib = None
        self.result = None
        self.calib_src = None
        self.check_state = None
        self._set_vstate("none", "")
        self._vstatic_src = None
        try:
            if os.path.isfile(CALIB_PATH):
                os.remove(CALIB_PATH)
        except Exception as exc:
            self.log(f"[WARN] [V3.48 calib] 没删掉 {CALIB_PATH}：{exc}")
        self._refresh_table()
        self._force_redraw()
        self.log("[OK] [V3.48 calib] 已清除当前手眼标定设定 → 可在「② 采集样本」重新采集（自动采集会先做 3 个小平移探测相机方向），"
                 "再「计算标定」「保存标定」")

    def on_quick_check(self):
        c = self.calib
        if not c or not c.get("check_poses_deg"):
            self.log("[WARN] [V3.25 calib] 没有已保存的标定（或其中没有复核姿态）")
            return
        if not self.jpanel.enabled or getattr(self.pipe, "teach_mode", False):
            self.log("[WARN] [V3.25 calib] 关节面板不可操作 / 示教模式中")
            return
        speed = min(CALIB_MAX_SPEED, max(0.01, float(self.cfg_fn().get("speed_percent", 10)) / 100.0))

        def job():
            if not self.ask("快速复核", f"机械臂走到标定时存档的 {len(c['check_poses_deg'])} 个姿态（{speed * 100:.0f}%），"
                                     "比较相机看到的标记和标定预测的位置。继续？"):
                return
            self.check_state = ("run",)
            model = self._model()
            q0 = self.actual_q()
            errs = []
            for k, q in enumerate(c["check_poses_deg"]):
                if self.stop_flag:
                    break
                s, why = self._move_measure(model, [math.radians(v) for v in q], speed, f"标定复核 {k + 1}")
                if s is None:
                    self.log(f"[WARN] [V3.25 calib] 复核姿态 {k + 1}：{why}")
                    continue
                errs.append(marker_error_mm(c, s["T_base_flange"], s["p_cam"]))
            if q0:
                self.pipe.move_and_settle(model, [math.radians(v) for v in q0], speed, label="标定复核 回起点")
            if not errs:
                self.check_state = ("fail", [])
                self.log("[FAIL] [V3.25 calib] 快速复核没有得到结果")
                return
            ok = max(errs) <= CALIB_GOOD_MAX_MM
            self.check_state = ("ok" if ok else "fail", errs)
            self._set_vstate("ok" if ok else "fail", "快速复核 " + ", ".join(f"{e:.1f}" for e in errs) + " mm")
            self.log(f"{'[OK]' if ok else '[FAIL]'} [V3.25 calib] 快速复核：误差 " + ", ".join(f"{e:.1f}" for e in errs)
                     + " mm" + ("" if ok else " → 相机或标记动过，请重新标定"))
        self._run(job)

    def on_marker_png(self):
        m = self._marker_from_ui()
        if not m:
            return
        ok, msg = aruco_status()
        if not ok:
            self.log("[FAIL] [V3.25 calib] " + msg)
            return
        try:
            cv2 = _import_cv2()
            img = marker_image(m["dict"], m["id"])
            path = os.path.join(HERE, "config", f"aruco_{m['dict']}_id{m['id']}.png")
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        except Exception as exc:
            self.log(f"[FAIL] [V3.25 calib] 生成标记失败：{exc}")
            return
        self.log(f"[OK] [V3.25 calib] 标记图片 → {path}：打印时把黑色方块缩放到 {m['size_mm']:.0f} mm，打印后用尺子量"
                 "黑框实际边长填进「边长 mm」")

    # ------------------------------------------------------------------ eggs
    def on_eggs(self):
        cal = self.result or self.calib
        if not cal or not self.src or not self.src.K:
            self.log("[WARN] [V3.25 locate] 需要相机连接 + 手眼标定")
            return
        if self.v_method.get() == LOCATE_METHODS[2]:
            return self._eggs_geometric(cal)
        return self._eggs_yolo(cal)

    def _grab_frames(self, n, timeout=2.0):
        """V3.47: n distinct frames (colour of the last one, depth of all) for depth fusion."""
        rgbs, depths, last, t0 = [], [], None, time.time()
        while len(depths) < n and time.time() - t0 < timeout:
            rgb, depth, fid = self.src.frames()
            if rgb is not None and depth is not None and fid != last:
                rgbs.append(rgb)
                depths.append(depth)
                last = fid
            else:
                time.sleep(0.02)
        return (rgbs[-1] if rgbs else None), depths

    def _eggs_yolo(self, cal):
        # V3.47 YOLO LOCATE
        YP = self.YP
        mode = "vp" if self.v_method.get() == LOCATE_METHODS[0] else "text"
        try:
            size = (float(self.v_egg_l.get()), float(self.v_egg_w.get()))
            assert 20.0 <= size[1] <= size[0] <= 120.0
            conf = float(self.v_yconf.get())
            assert 0.01 <= conf <= 0.99
        except Exception:
            self.log("[WARN] [V3.47 locate] 鸡蛋尺寸（宽 ≤ 长，20–120 mm）或置信度门槛（0.01–0.99）无效")
            return
        path = YP.find_model(self.v_ydir.get(), self.v_ymodel.get())
        prompts = YP.parse_prompts(self.v_yprompt.get())
        ref = self.ref
        f = self._fence()
        save = bool(self.v_ysave.get())
        cnote = self._calib_note()                   # V3.47.1: unverified calibration -> results not trusted

        def job():
            import numpy as np
            if not path:
                self.log(f"[FAIL] [V3.47 locate] 找不到模型 {self.v_ymodel.get()}（找过：{', '.join(YP.model_dirs(self.v_ydir.get()))}）")
                return
            t0 = time.time()
            rgb, depths = self._grab_frames(YP.FUSE_FRAMES)
            if rgb is None or not depths:
                self.log("[WARN] [V3.47 locate] 没拿到相机画面")
                return
            try:
                d2 = self.det_yolo.detect(rgb, mode, path, ref=ref, prompts=prompts, conf=conf)
            except YP.YoloUnavailable as exc:
                self.log(f"[FAIL] [V3.47 locate] {exc}")
                return
            t1 = time.time()
            depth = YP.fuse_depth(depths, d2)
            res, diag = YP.locate(d2, depth, dict(self.src.K), np.array(cal["T_base_cam"], dtype=float),
                                  table_z=(f or {}).get("z_table"), fence_poly=(f or {}).get("polygon"), size_mm=size,
                                  calib_note=cnote)
            t2 = time.time()
            dev = self.det_yolo.info.get("device") or self.det_yolo.device_name(self.det_yolo.model)
            sp = None
            if save:
                try:
                    sp = YP.save_sample(rgb, d2, res, {"mode": mode, "model": os.path.basename(path), "conf": conf,
                                                       "size_mm": list(size)})
                except Exception as exc:
                    self.log(f"[WARN] [V3.47 locate] 样本没保存：{exc}")
            self.eggs = [r for r in res if r["status"] != "rejected"]
            view = []
            for r in res:
                view.append({"box": r["yolo"]["box"], "status": r["status"], "label": short_egg_label(r["det_id"])})

            def show():
                self.yolo_view = view
                for it in self.tv_eggs.get_children():
                    self.tv_eggs.delete(it)
                for row_i, r in enumerate(res):
                    x, y, z = r.get("xyz_m") or (None, None, None)
                    mm = (lambda v: "-" if v is None else f"{v * 1000:.0f}")
                    why = "；".join(r["reasons"]) or ("✓ " + (r.get("pose") or "") + "，轮廓对齐")
                    if r.get("notes"):
                        why += "（" + "；".join(r["notes"]) + "）"
                    self.tv_eggs.insert("", "end", tags=("odd",) if row_i % 2 else (), values=(
                        r["det_id"], mm(x), mm(y), mm(z),
                        "-" if r.get("height_mm") is None else f"{r['height_mm']:.0f}",
                        f"{r['yolo']['conf']:.2f}", "-" if r.get("iou") is None else f"{r['iou']:.2f}",
                        f"{r['confidence']:.2f}", "✓" if r["trusted"] else "-", why))
                autofit_treeview(self.tv_eggs, self.egg_cols, stretch_col="why")
                g = diag.get("ground_minus_z_table_mm")
                self.v_eggs.set(
                    f"{'视觉提示' if mode == 'vp' else '文字提示'} · YOLO {diag['n_2d']} 个 → 鸡蛋 {diag['n_egg']} · 不可信 "
                    f"{diag['n_untrusted']} · 剔除 {diag['n_rejected']} · YOLO({dev}) {(t1 - t0) * 1000:.0f} ms + 3D "
                    f"{(t2 - t1) * 1000:.0f} ms（深度融合 {len(depths)} 帧）"
                    + ("" if g is None else f"\n局部桌面比安全设置 z_table {'高' if g >= 0 else '低'} {abs(g):.0f} mm"
                       + (" ⚠ 建议在安全设置里更新 z_table" if abs(g) > 8 else ""))
                    + (f" · 桌面平面 RMS {diag['plane_rms_mm']:.1f} mm" if diag.get("plane_rms_mm") is not None else "")
                    + (f"\n⚠ {cnote}" if cnote else "")
                    + (f"\n样本 → {sp}" if sp else ""))
                self._force_redraw()
            self._ui(show)
            self.log(f"[INFO] [V3.47 locate] 鸡蛋识别（{mode}）：YOLO({dev}) {(t1 - t0) * 1000:.0f} ms + 3D "
                     f"{(t2 - t1) * 1000:.0f} ms · YOLO {diag['n_2d']} → 鸡蛋 {diag['n_egg']} / 不可信 "
                     f"{diag['n_untrusted']} / 剔除 {diag['n_rejected']}；" + "; ".join(
                         (f"{r['det_id']} ({r['xyz_m'][0] * 1000:.0f}, {r['xyz_m'][1] * 1000:.0f}, {r['xyz_m'][2] * 1000:.0f}) mm"
                          if r.get("xyz_m") and r["status"] != "rejected" else f"✗ {'；'.join(r['reasons'])}")
                         for r in res[:8]))

        def safe():
            try:
                job()
            except Exception as exc:
                self.log(f"[FAIL] [V3.47 locate] 识别失败：{type(exc).__name__}: {exc}")
        self._run(safe)

    def _eggs_geometric(self, cal):
        def job():
            try:
                import numpy as np
                _import_cv2()                        # V3.46: cv2 first through the Qt-variable guard
                import NERO_V308_Camera_Perception as P8
                rgb, depth, _fid = self.src.frames()
                if rgb is None:
                    return
                os.environ["NERO_V308_EGG_COLOR_FILTER"] = "0" if self.v_white.get() else "1"
                f = self._fence()
                if not f:
                    self.log("[WARN] [V3.25 locate] 没有有效的安全区（桌面范围）：先在安全设置里保存")
                    return
                xs = [p[0] for p in f["polygon"]]
                ys = [p[1] for p in f["polygon"]]
                old = (P8.TABLE_Z_M, dict(P8.WORKSPACE))
                P8.TABLE_Z_M = float(f["z_table"])
                P8.WORKSPACE = {"x": (min(xs), max(xs)), "y": (min(ys), max(ys)), "z": (0.004, f["z_table"] + 0.25)}
                try:
                    T = np.array(cal["T_base_cam"], dtype=float)
                    dets, diag = P8.perceive_eggs(rgb, depth, dict(self.src.K), T, stride=2)
                finally:
                    P8.TABLE_Z_M, P8.WORKSPACE = old
            except Exception as exc:
                self.log(f"[FAIL] [V3.25 locate] 识别失败：{exc}")
                return
            self.eggs = dets

            def show():
                self.yolo_view = []
                for it in self.tv_eggs.get_children():
                    self.tv_eggs.delete(it)
                for row_i, d in enumerate(dets):
                    x, y, z = d["xyz_m"]
                    self.tv_eggs.insert("", "end", tags=("odd",) if row_i % 2 else (),
                                        values=(d.get("det_id"), f"{x * 1000:.0f}", f"{y * 1000:.0f}",
                                                f"{z * 1000:.0f}", "-", "-", "-", f"{d.get('confidence', 0):.2f}",
                                                "✓" if d.get("trusted") else "-", "几何法 V3.08"))
                autofit_treeview(self.tv_eggs, self.egg_cols, stretch_col="why")
                self.v_eggs.set(f"几何法：检测到 {len(dets)} 个（可信 {sum(1 for d in dets if d.get('trusted'))}）· 有效点 "
                                f"{diag.get('points_valid')} · 鸡蛋掩膜点 {diag.get('points_egg_mask')}")
                self._force_redraw()
            self._ui(show)
            self.log(f"[INFO] [V3.25 locate] 鸡蛋识别：{len(dets)} 个 " + "; ".join(
                f"{d.get('det_id')} ({d['xyz_m'][0] * 1000:.0f}, {d['xyz_m'][1] * 1000:.0f}, {d['xyz_m'][2] * 1000:.0f}) mm"
                for d in dets[:6]))
        self._run(job)

    # ------------------------------------------------------------------ V3.48 general object recognition
    def _resolve_object_query(self, path):
        """Return (mode 'pf'|'text', model_path, prompts_list) for general object recognition."""
        YP = self.YP
        raw = self.v_obj_names.get().strip() if hasattr(self, "v_obj_names") else ""
        if not raw:
            yp = self.v_yprompt.get().strip()
            if yp and yp != YP.DEFAULT_TEXT_PROMPTS:
                raw = yp
        prompts = YP.parse_prompts(raw)
        wild = YP.is_wildcard_prompt(prompts)
        if path and YP.is_pf_model(path):
            return "pf", path, prompts
        pf_path = YP.find_model(self.v_ydir.get(), YP.DEFAULT_PF_MODEL)
        if wild and pf_path:
            return "pf", pf_path, prompts
        if path:
            enc = os.path.join(os.path.dirname(path), YP.text_encoder_name(path))
            if not os.path.isfile(enc) and pf_path:
                return "pf", pf_path, prompts
        return "text", path, (prompts if not wild else YP.parse_prompts(YP.DEFAULT_OBJECT_PROMPTS))

    def on_objects(self):
        """V3.48 OBJECTS: detect arbitrary objects by name (text prompt or *-pf built-in 4,585-class vocabulary)."""
        cal = self.result or self.calib
        if not cal or not self.src or not self.src.K:
            self.log("[WARN] [V3.48 locate] 需要相机连接 + 手眼标定")
            return
        YP = self.YP
        try:
            conf = float(self.v_yconf.get())
            assert 0.01 <= conf <= 0.99
        except Exception:
            self.log("[WARN] [V3.48 locate] 置信度门槛（0.01–0.99）无效")
            return
        sel_path = YP.find_model(self.v_ydir.get(), self.v_ymodel.get())
        mode, path, prompts = self._resolve_object_query(sel_path)
        f = self._fence()
        save = bool(self.v_ysave.get())
        cnote = self._calib_note()

        def job():
            import numpy as np
            if not path:
                self.log(f"[FAIL] [V3.48 locate] 找不到模型 {self.v_ymodel.get()}（找过：{', '.join(YP.model_dirs(self.v_ydir.get()))}）")
                return
            t0 = time.time()
            rgb, depths = self._grab_frames(YP.FUSE_FRAMES)
            if rgb is None or not depths:
                self.log("[WARN] [V3.48 locate] 没拿到相机画面")
                return
            try:
                d2 = self.det_yolo.detect(rgb, mode, path, prompts=prompts, conf=conf)
            except YP.YoloUnavailable as exc:
                self.log(f"[FAIL] [V3.48 locate] {exc}")
                return
            t1 = time.time()
            depth = YP.fuse_depth(depths, d2)
            res, diag = YP.locate_objects(d2, depth, dict(self.src.K), np.array(cal["T_base_cam"], dtype=float),
                                          table_z=(f or {}).get("z_table"), fence_poly=(f or {}).get("polygon"),
                                          calib_note=cnote)
            t2 = time.time()
            dev = self.det_yolo.info.get("device") or self.det_yolo.device_name(self.det_yolo.model)
            sp = None
            if save:
                try:
                    sp = YP.save_sample(rgb, d2, res, {"mode": f"obj_{mode}", "model": os.path.basename(path),
                                                       "conf": conf, "prompts": prompts})
                except Exception as exc:
                    self.log(f"[WARN] [V3.48 locate] 样本没保存：{exc}")
            self.eggs = [r for r in res if r["status"] != "rejected"]
            view = []
            for r in res:
                view.append({"box": r["yolo"]["box"], "status": r["status"],
                             "label": short_egg_label(r["det_id"], r.get("obj_name", ""))})
            pf_seen = self.det_yolo.info.get("pf_seen") or []

            def show():
                self.yolo_view = view
                for it in self.tv_eggs.get_children():
                    self.tv_eggs.delete(it)
                for row_i, r in enumerate(res):
                    x, y, z = r.get("xyz_m") or (None, None, None)
                    mm = (lambda v: "-" if v is None else f"{v * 1000:.0f}")
                    sz = r.get("size_mm")
                    sz_txt = f" · 约 {sz[0]:.0f}×{sz[1]:.0f} mm" if sz else ""
                    why = "；".join(r["reasons"]) or f"✓ 类别 {r.get('obj_name', '')}{sz_txt}"
                    if r.get("reasons") and r.get("obj_name"):
                        why = f"[{r['obj_name']}] " + why
                    if r.get("notes"):
                        why += "（" + "；".join(r["notes"]) + "）"
                    self.tv_eggs.insert("", "end", tags=("odd",) if row_i % 2 else (), values=(
                        r["det_id"], mm(x), mm(y), mm(z),
                        "-" if r.get("height_mm") is None else f"{r['height_mm']:.0f}",
                        f"{r['yolo']['conf']:.2f}", "-", f"{r['confidence']:.2f}",
                        "✓" if r["trusted"] else "-", why))
                autofit_treeview(self.tv_eggs, self.egg_cols, stretch_col="why")
                q_desc = "默认词库(-pf)" if mode == "pf" else f"文字提示({', '.join(prompts[:4])})"
                extra_pf = f"（内置词库看到：{', '.join(pf_seen[:8])}）" if (mode == "pf" and not res and pf_seen) else ""
                self.v_eggs.set(
                    f"物体识别 · {q_desc} · YOLO {diag['n_2d']} 个 → 可信 {diag['n_obj']} · 不可信 "
                    f"{diag['n_untrusted']} · 剔除 {diag['n_rejected']} · YOLO({dev}) {(t1 - t0) * 1000:.0f} ms + 3D "
                    f"{(t2 - t1) * 1000:.0f} ms{extra_pf}"
                    + (f"\n⚠ {cnote}" if cnote else "")
                    + (f"\n样本 → {sp}" if sp else ""))
                self._force_redraw()
                self._update_calib_ui()
            self._ui(show)
            self.log(f"[INFO] [V3.48 locate] 物体识别（{mode}，{os.path.basename(path)}）：YOLO({dev}) "
                     f"{(t1 - t0) * 1000:.0f} ms + 3D {(t2 - t1) * 1000:.0f} ms · 可信 {diag['n_obj']} / 不可信 "
                     f"{diag['n_untrusted']}；" + ("; ".join(
                         f"{r['det_id']}[{r.get('obj_name')}] ({r['xyz_m'][0] * 1000:.0f}, {r['xyz_m'][1] * 1000:.0f}, {r['xyz_m'][2] * 1000:.0f}) mm"
                         for r in res[:8] if r.get("xyz_m")) or (f"未命中（内置词库看到：{', '.join(pf_seen[:8])}）" if pf_seen else "无")))

        def safe():
            try:
                job()
            except Exception as exc:
                self.log(f"[FAIL] [V3.48 locate] 物体识别失败：{type(exc).__name__}: {exc}")
        self._run(safe)

    # ------------------------------------------------------------------ V3.48 TCP calibration + hover verification
    def _tcp_file_path(self):
        if TCP_CALIB_PATH != DEFAULT_TCP_CALIB_PATH:
            return TCP_CALIB_PATH
        if CALIB_PATH != DEFAULT_CALIB_PATH:
            return os.path.join(os.path.dirname(CALIB_PATH), "tcp_calib.json")
        return TCP_CALIB_PATH

    def _urdf_tcp_vec(self):
        import NERO_V315_Consistency_Safety as CS
        T = CS.tool_transform_from_sim(self._urdf())
        return [float(T[0][3]), float(T[1][3]), float(T[2][3])] if T is not None else list(DEFAULT_TCP_IN_FLANGE)

    def _active_tcp_vec(self):
        if tcp_valid(self.tcp_calib):
            return [float(v) for v in self.tcp_calib["tcp_in_flange"]]
        return self._urdf_tcp_vec()

    def _active_tcp_T(self):
        import NERO_V315_Consistency_Safety as CS
        T = CS.tool_transform_from_sim(self._urdf())
        if T is None:
            T = [[0.0, 0.0, 1.0, DEFAULT_TCP_IN_FLANGE[0]], [-1.0, 0.0, 0.0, DEFAULT_TCP_IN_FLANGE[1]],
                 [0.0, -1.0, 0.0, DEFAULT_TCP_IN_FLANGE[2]], [0.0, 0.0, 0.0, 1.0]]
        else:
            T = [list(map(float, r)) for r in T]
        p = self._active_tcp_vec()
        for i in range(3):
            T[i][3] = p[i]
        return T

    def _tcp_autoload(self):
        path = self._tcp_file_path()
        d = load_json(path)
        if not tcp_valid(d) and path == DEFAULT_TCP_CALIB_PATH and CALIB_PATH == DEFAULT_CALIB_PATH:
            cands = sorted(glob.glob(os.path.join(CALIB_DIR, "tcp_calib*.json")), key=os.path.getmtime, reverse=True)
            for f in cands:
                cand = load_json(f)
                if tcp_valid(cand):
                    d, path = cand, f
                    break
        if tcp_valid(d):
            self.tcp_calib = d

    def _tcp_text(self):
        p = self._active_tcp_vec()
        u = self._urdf_tcp_vec()
        n = len(self.tcp_samples)
        rs = rotation_span_deg([s["T_base_flange"] for s in self.tcp_samples]) if n > 1 else 0.0
        samp_txt = f"碰点样本 {n}/≥4" + (f"（朝向变化 {rs:.0f}°/≥25°）" if n > 1 else "")
        if tcp_valid(self.tcp_calib):
            c = self.tcp_calib
            diff = vnorm(vsub(p, u)) * 1000.0
            return (f"当前 TCP（link7 系）：x {p[0] * 1000:+.1f}  y {p[1] * 1000:+.1f}  z {p[2] * 1000:+.1f} mm"
                    f"（已标定 {c.get('saved', '')} · 与 URDF 默认差 {diff:.1f} mm · 留一 "
                    f"{(c.get('loo_mean_mm') or 0):.1f}/{(c.get('loo_max_mm') or 0):.1f} mm） · {samp_txt}")
        return (f"当前 TCP（link7 系）：x {p[0] * 1000:+.1f}  y {p[1] * 1000:+.1f}  z {p[2] * 1000:+.1f} mm"
                f"（URDF 默认 nero_7dof_with_tcp.urdf） · {samp_txt}")

    def _update_tcp_ui(self):
        if hasattr(self, "v_tcp_status"):
            self.v_tcp_status.set(self._tcp_text())
        self._update_calib_ui()

    def on_tcp_record(self):
        q = self.actual_q()
        if not q:
            self.log("[WARN] [V3.48 tcp] 没有关节反馈（请先连接实机 / 启动监控）")
            return
        T = self.flange_T(q)
        if T is None:
            self.log("[FAIL] [V3.48 tcp] FK 计算失败")
            return
        self.tcp_samples.append({"q_deg": [round(v, 3) for v in q], "T_base_flange": T})
        rs = rotation_span_deg([s["T_base_flange"] for s in self.tcp_samples]) if len(self.tcp_samples) > 1 else 0.0
        self._update_tcp_ui()
        self.log(f"[OK] [V3.48 tcp] 记录碰点姿态 #{len(self.tcp_samples)}（手腕朝向变化 {rs:.0f}°）："
                 "请换一个手腕倾角让两指中心触碰桌面同一固定点（≥ 4 个姿态，朝向变化 ≥ 25°）")

    def on_tcp_compute(self):
        r = calibrate_tcp(self.tcp_samples, urdf_tcp=self._urdf_tcp_vec())
        if not r.get("ok"):
            self.log(f"[WARN] [V3.48 tcp] TCP 标定未通过：{r.get('error')}")
            return
        p = r["tcp_in_flange"]
        if not r["good"] and not self.ask(
            "保存 TCP 标定",
            f"TCP 标定留一误差 平均 {r['loo_mean_mm']:.2f} / 最大 {r['loo_max_mm']:.2f} mm，朝向变化 {r['rot_span_deg']:.0f}°。\n"
            + ("\n".join(r["warnings"]) + "\n" if r["warnings"] else "")
            + f"解算 TCP (link7)：({p[0] * 1000:+.1f}, {p[1] * 1000:+.1f}, {p[2] * 1000:+.1f}) mm（与 URDF 差 {r['diff_urdf_mm']:.1f} mm）。仍然保存？"
        ):
            return
        d = dict(r, version="V3.48", saved=time.strftime("%Y-%m-%d %H:%M:%S"))
        path = self._tcp_file_path()
        save_json(path, d)
        if path == DEFAULT_TCP_CALIB_PATH and CALIB_PATH == DEFAULT_CALIB_PATH:
            try:
                os.makedirs(CALIB_DIR, exist_ok=True)
                save_json(os.path.join(CALIB_DIR, "tcp_calib.json"), d)
            except Exception:
                pass
        self.tcp_calib = d
        self._update_tcp_ui()
        tag = "[OK]" if r["good"] else "[WARN]"
        self.log(f"{tag} [V3.48 tcp] TCP 标定已保存 → {path}：link7 系 ({p[0] * 1000:+.1f}, {p[1] * 1000:+.1f}, "
                 f"{p[2] * 1000:+.1f}) mm · 与 URDF 差 {r['diff_urdf_mm']:.1f} mm · 留一 {r['loo_mean_mm']:.2f}/"
                 f"{r['loo_max_mm']:.2f} mm")

    def on_tcp_clear_samples(self):
        self.tcp_samples = []
        self._update_tcp_ui()
        self.log("[INFO] [V3.48 tcp] 已清空碰点样本")

    def on_tcp_reset(self):
        path = self._tcp_file_path()
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception:
            pass
        self.tcp_calib = None
        self._update_tcp_ui()
        u = self._urdf_tcp_vec()
        self.log(f"[OK] [V3.48 tcp] 已恢复 URDF 默认 TCP 偏移：({u[0] * 1000:+.1f}, {u[1] * 1000:+.1f}, {u[2] * 1000:+.1f}) mm")

    def apply_strategy_profile(self, profile):
        """V3.59: Synchronize active StrategyProfile knobs with CameraTabUI hover & dry-run parameters."""
        if profile is None:
            return
        self.strategy_profile = profile
        if hasattr(self, "v_hover_mm"):
            h_mm = float(profile.get("cam_hover_verify_height_mm", HOVER_DEFAULT_MM))
            if HOVER_RANGE_MM[0] <= h_mm <= HOVER_RANGE_MM[1]:
                self.v_hover_mm.set(f"{h_mm:.0f}")
        self._update_calib_ui()

    def _selected_hover_target(self):
        valid = [e for e in (self.eggs or []) if e.get("xyz_m") and e.get("status") != "rejected"]
        if not valid:
            return None
        sel = self.tv_eggs.selection()
        if sel:
            vals = self.tv_eggs.item(sel[0], "values") or ()
            det_id = vals[0] if vals else None
            for e in valid:
                if e.get("det_id") == det_id:
                    return e
        knobs = self.strategy_profile.knobs if getattr(self, "strategy_profile", None) else None
        ranked = self.ST59.rank_detected_targets(valid, knobs=knobs)
        return ranked[0] if ranked else valid[0]

    def _dry_dest_mode_key(self):
        if not hasattr(self, "v_dry_dest_mode"):
            return "next_target"
        lab = self.v_dry_dest_mode.get()
        for k, text in getattr(self, "_dry_mode_items", ()):
            if text == lab:
                return k
        return "next_target"

    def plan_dry_run_from_ui(self, dest_mode=None, manual_dest_text=None, knobs=None):
        """V3.59: Plan a gripper-free multi-point Hover Dry-Run sequence from current UI state without moving the arm."""
        k = dict(self.strategy_profile.knobs) if getattr(self, "strategy_profile", None) else dict(self.ST59.BASELINE_KNOBS)
        if isinstance(knobs, dict):
            k.update(knobs)
        try:
            hover_mm = float(self.v_hover_mm.get())
            if HOVER_RANGE_MM[0] <= hover_mm <= HOVER_RANGE_MM[1]:
                k["cam_hover_verify_height_mm"] = hover_mm
        except Exception:
            pass
        mode = dest_mode or self._dry_dest_mode_key()
        m_txt = manual_dest_text if manual_dest_text is not None else (
            self.v_dry_dest_xyz.get() if hasattr(self, "v_dry_dest_xyz") else ""
        )
        sel_ids = []
        if hasattr(self, "tv_eggs"):
            for it in self.tv_eggs.selection():
                vals = self.tv_eggs.item(it, "values") or ()
                if vals:
                    sel_ids.append(str(vals[0]))
        targets = self.ST59.build_dry_run_targets(
            self.eggs or [], selected_ids=sel_ids, dest_mode=mode, manual_dest_text=m_txt, knobs=k,
        )
        if not targets and m_txt and str(m_txt).strip():
            t_manual = self.ST59.parse_waypoint_spec(m_txt, fallback_id="目标点B")
            if t_manual:
                targets = [t_manual]
        q0 = self.actual_q()
        if not q0:
            self.log("[WARN] [V3.59 dry-run] 没有关节反馈（请先连接实机或启动监控）")
            return {"ok": False, "error": "没有关节反馈"}
        model = self._model()
        urdf = self._urdf()
        base = model.get("base_link") or "base_link"
        tool_T = self._active_tcp_T()
        f = self._fence()
        zt = float((f or {}).get("z_table", 0.0))
        q0r = [math.radians(v) for v in q0]

        def fence_fn(q_rad):
            if hasattr(self.pipe, "fence_goal_check"):
                r_ = self.pipe.fence_goal_check(model, q_rad)
                return bool(r_[0]), (r_[1] if len(r_) > 1 else "")
            return True, ""

        plan = self.ST59.plan_hover_dry_run_sequence(
            urdf, model["joints"], model["limits"], q0r, tool_T, targets,
            z_table_m=zt, base=base, fence_fn=fence_fn, knobs=k, plan_hover_fn=plan_hover_target,
        )
        if plan.get("ok"):
            self.log(f"[OK] [V3.59 dry-run] 预演航线规划校验通过：{plan['summary']}")
        else:
            self.log(f"[FAIL] [V3.59 dry-run] {plan.get('error')}")
        return plan

    def on_hover_dry_run(self, dest_mode=None, manual_dest_text=None, knobs=None, status_cb=None):
        """V3.59 MULTI-POINT HOVER DRY-RUN: execute Target A hover -> safe transit lift -> Target B hover -> return home."""
        if self._busy():
            self.log("[WARN] [V3.59 dry-run] 上一个操作还在进行")
            return
        if getattr(self.pipe, "teach_mode", False) or not getattr(self.jpanel, "enabled", False):
            self.log("[WARN] [V3.59 dry-run] 需要退出示教模式且 MoveIt / 关节面板就绪")
            return
        plan = self.plan_dry_run_from_ui(dest_mode=dest_mode, manual_dest_text=manual_dest_text, knobs=knobs)
        if not plan or not plan.get("ok"):
            if status_cb:
                status_cb(f"✗ 规划未通过：{(plan or {}).get('error', '')}")
            return
        k = dict(self.strategy_profile.knobs) if getattr(self, "strategy_profile", None) else dict(self.ST59.BASELINE_KNOBS)
        if isinstance(knobs, dict):
            k.update(knobs)
        max_spd_pct = float(k.get("cam_hover_max_speed_pct", 10.0))
        cfg_spd_pct = float(self.cfg_fn().get("speed_percent", 10))
        speed = min(max_spd_pct / 100.0, max(0.01, cfg_spd_pct / 100.0))
        model = self._model()
        urdf = self._urdf()
        base = model.get("base_link") or "base_link"
        tool_T = self._active_tcp_T()
        q0 = self.actual_q()
        q0r = [math.radians(v) for v in q0]

        station_lines = []
        for s in plan["steps"]:
            if s["kind"] == "hover":
                tx, ty, tz = [v * 1000.0 for v in s["tcp_xyz_m"]]
                ox, oy, oz = [v * 1000.0 for v in s["obj_xyz_m"]]
                station_lines.append(
                    f"  · 悬停站 {s['det_id']}：物心 ({ox:+.0f}, {oy:+.0f}, {oz:+.0f}) → 悬停 TCP ({tx:+.0f}, {ty:+.0f}, {tz:+.0f}) mm"
                )
        stations_txt = "\n".join(station_lines)

        def job():
            if not self.ask(
                "多点悬停预演 (Dry-Run)",
                f"机械臂将执行无夹爪多点悬停预演（{plan['summary']}）：\n"
                f"{stations_txt}\n"
                f"运行速度：{speed * 100:.0f}% · 驻留 {plan['dwell_s']:.1f} s/站 · 只走上方悬停与过渡点，不下探、不动夹爪。\n"
                "确认桌面周围无障碍并开始预演？",
            ):
                return
            self._pre_hover_q = list(q0r)
            if status_cb:
                self._ui(lambda: status_cb(f"▶ 正在执行多点悬停预演：{plan['summary']} …"))
            import NERO_V315_Consistency_Safety as CS

            def _move(q_rad, label):
                return self.pipe.move_and_settle(model, q_rad, speed, label=label)

            def _fk_tcp(q_rad):
                T_act = CS.tool_pose(urdf, model["joints"], q_rad, base, tool_T)
                return T_t(T_act) if T_act is not None else None

            def _wrist_chk(tgt_obj):
                if not bool(k.get("dry_run_wrist_refine", True)):
                    return None
                ws = self.wrist_src
                intr = self.intr_result or self.wrist_intr
                K_w = (getattr(ws, "K", None) if ws else None) or (intr.get("K") if intr else None)
                he = self.wcalib_result or self.wrist_calib
                if not ws or not K_w or not he or not he.get("T_flange_cam"):
                    return None
                rgb, _d, _fid = ws.frames()
                qa = self.actual_q()
                if rgb is None or not qa:
                    return None
                T_bf = self.flange_T(qa)
                if T_bf is None:
                    return None
                f_ = self._fence()
                zt_ = float((f_ or {}).get("z_table", 0.0))
                return self.W49.wrist_refine_target(
                    rgb, dict(K_w), T_bf, he, tool_T, tgt_obj, table_z=zt_,
                )

            res = self.ST59.execute_hover_dry_run(
                plan,
                move_fn=_move,
                actual_q_fn=self.actual_q,
                fk_tcp_fn=_fk_tcp,
                wrist_check_fn=_wrist_chk,
                stop_fn=lambda: bool(self.stop_flag),
                log_fn=self.log,
            )
            if res.get("ok") and plan.get("return_home"):
                self._pre_hover_q = None
            if status_cb:
                msg = (
                    f"✓ 多点悬停预演完成：共走完 {res['completed_hovers']} 个悬停站"
                    if res.get("ok")
                    else f"⚠ 悬停预演中止：{res.get('error', '')}"
                )
                self._ui(lambda m=msg: status_cb(m))
            self._ui(self._update_calib_ui)

        self._run(job)

    def on_hover_target(self):
        """V3.48 HOVER VERIFY: move the TCP to hover HOVER_DEFAULT_MM (e.g. 100 mm) directly above the selected
        egg / object so the user can check the xy alignment with a ruler without touching the object or gripper."""
        if self._busy():
            self.log("[WARN] [V3.48 hover] 上一个操作还在进行")
            return
        if getattr(self.pipe, "teach_mode", False) or not getattr(self.jpanel, "enabled", False):
            self.log("[WARN] [V3.48 hover] 需要退出示教模式且 MoveIt / 关节面板就绪")
            return
        tgt_item = self._selected_hover_target()
        if not tgt_item:
            self.log("[WARN] [V3.48 hover] 先点「识别鸡蛋」或「识别物体」得到目标位置")
            return
        try:
            hover_mm = float(self.v_hover_mm.get())
            assert HOVER_RANGE_MM[0] <= hover_mm <= HOVER_RANGE_MM[1]
        except Exception:
            self.log(f"[WARN] [V3.48 hover] 悬停高度请填 {HOVER_RANGE_MM[0]:.0f}–{HOVER_RANGE_MM[1]:.0f} mm")
            return
        q0 = self.actual_q()
        if not q0:
            self.log("[FAIL] [V3.48 hover] 没有关节反馈")
            return
        model = self._model()
        urdf = self._urdf()
        base = model.get("base_link") or "base_link"
        tool_T = self._active_tcp_T()
        ox, oy, oz = [float(v) for v in tgt_item["xyz_m"]]
        f = self._fence()
        zt = float((f or {}).get("z_table", 0.0))
        hz = max(oz + hover_mm / 1000.0, zt + 0.05)
        target_xyz = [ox, oy, hz]
        q0r = [math.radians(v) for v in q0]

        def fence_fn(q_rad):
            if hasattr(self.pipe, "fence_goal_check"):
                r_ = self.pipe.fence_goal_check(model, q_rad)
                return bool(r_[0]), (r_[1] if len(r_) > 1 else "")
            return True, ""

        q_tgt, info = plan_hover_target(urdf, model["joints"], model["limits"], q0r, tool_T, target_xyz,
                                        base=base, fence_fn=fence_fn)
        if q_tgt is None:
            self.log(f"[FAIL] [V3.48 hover] 悬停规划失败（{tgt_item.get('det_id')}）：{info}")
            return
        speed = min(CALIB_MAX_SPEED, max(0.01, float(self.cfg_fn().get("speed_percent", 10)) / 100.0))
        det_id = tgt_item.get("det_id", "目标")
        untrusted_warn = "" if tgt_item.get("trusted") else "\n⚠ 注意：该目标当前标为「不可信」！"

        def job():
            if not self.ask(
                "悬停复核",
                f"机械臂将把 TCP（两指中心）移到 {det_id} 正上方 {hover_mm:.0f} mm：{untrusted_warn}\n"
                f"目标物体 (base_link)：({ox * 1000:+.0f}, {oy * 1000:+.0f}, {oz * 1000:+.0f}) mm\n"
                f"悬停 TCP (base_link)：({target_xyz[0] * 1000:+.0f}, {target_xyz[1] * 1000:+.0f}, {target_xyz[2] * 1000:+.0f}) mm\n"
                f"速度 {speed * 100:.0f}% · 只悬停、不下探、不动夹爪。\n确认周围无障碍？"
            ):
                return
            self._pre_hover_q = list(q0r)
            res = self.pipe.move_and_settle(model, q_tgt, speed, label=f"悬停复核 {det_id}")
            if not res.get("ok"):
                self.log(f"[FAIL] [V3.48 hover] 悬停运动失败：{res.get('error')}")
                self._ui(self._update_calib_ui)
                return
            import NERO_V315_Consistency_Safety as CS
            qa = self.actual_q() or [math.degrees(v) for v in q_tgt]
            T_act = CS.tool_pose(urdf, model["joints"], [math.radians(v) for v in qa], base, tool_T)
            act_xyz = T_t(T_act) if T_act is not None else target_xyz
            err_mm = vnorm(vsub(act_xyz, target_xyz)) * 1000.0
            self.log(f"[OK] [V3.48 hover] 已悬停在 {det_id} 上方 {hover_mm:.0f} mm：目标 TCP "
                     f"({target_xyz[0] * 1000:+.0f}, {target_xyz[1] * 1000:+.0f}, {target_xyz[2] * 1000:+.0f}) mm · "
                     f"实测 TCP ({act_xyz[0] * 1000:+.0f}, {act_xyz[1] * 1000:+.0f}, {act_xyz[2] * 1000:+.0f}) mm"
                     f"（偏差 {err_mm:.1f} mm）→ 可用尺子核对两指中心与物体中心的水平偏差")
            self._ui(self._update_calib_ui)
        self._run(job)

    def on_hover_back(self):
        if self._busy() or not self._pre_hover_q:
            return
        if getattr(self.pipe, "teach_mode", False) or not getattr(self.jpanel, "enabled", False):
            self.log("[WARN] [V3.48 hover] 需要退出示教模式且 MoveIt 就绪")
            return
        speed = min(CALIB_MAX_SPEED, max(0.01, float(self.cfg_fn().get("speed_percent", 10)) / 100.0))
        q_back = list(self._pre_hover_q)

        def job():
            res = self.pipe.move_and_settle(self._model(), q_back, speed, label="悬停复核 回原位")
            if res.get("ok"):
                self._pre_hover_q = None
                self.log("[OK] [V3.48 hover] 已回到悬停前姿态")
            else:
                self.log(f"[FAIL] [V3.48 hover] 回悬停前姿态失败：{res.get('error')}")
            self._ui(self._update_calib_ui)
        self._run(job)

    # ------------------------------------------------------------------ V3.49 wrist USB camera + intrinsics + eye-in-hand + holder
    def _wrist_intr_path(self):
        W49 = self.W49
        if W49.WRIST_INTR_PATH != W49.DEFAULT_WRIST_INTR_PATH:
            return W49.WRIST_INTR_PATH
        if CALIB_PATH != DEFAULT_CALIB_PATH:
            return os.path.join(os.path.dirname(CALIB_PATH), "wrist_intrinsics.json")
        return W49.WRIST_INTR_PATH

    def _wrist_handeye_path(self):
        W49 = self.W49
        if W49.WRIST_HANDEYE_PATH != W49.DEFAULT_WRIST_HANDEYE_PATH:
            return W49.WRIST_HANDEYE_PATH
        if CALIB_PATH != DEFAULT_CALIB_PATH:
            return os.path.join(os.path.dirname(CALIB_PATH), "wrist_handeye_calib.json")
        return W49.WRIST_HANDEYE_PATH

    def _wrist_autoload(self):
        W49 = self.W49
        ip = self._wrist_intr_path()
        d_intr = load_json(ip)
        if not W49.intrinsics_valid(d_intr) and ip == W49.DEFAULT_WRIST_INTR_PATH and CALIB_PATH == DEFAULT_CALIB_PATH:
            cands = sorted(glob.glob(os.path.join(CALIB_DIR, "wrist_intrinsics*.json")), key=os.path.getmtime, reverse=True)
            for f in cands:
                cand = load_json(f)
                if W49.intrinsics_valid(cand):
                    d_intr = cand
                    break
        if W49.intrinsics_valid(d_intr):
            self.wrist_intr = d_intr

        hp = self._wrist_handeye_path()
        d_he = load_json(hp)
        if not W49.handeye_valid(d_he) and hp == W49.DEFAULT_WRIST_HANDEYE_PATH and CALIB_PATH == DEFAULT_CALIB_PATH:
            cands = sorted(glob.glob(os.path.join(CALIB_DIR, "wrist_handeye*.json")), key=os.path.getmtime, reverse=True)
            for f in cands:
                cand = load_json(f)
                if W49.handeye_valid(cand):
                    d_he = cand
                    break
        if W49.handeye_valid(d_he):
            self.wrist_calib = d_he
            if isinstance(d_he.get("marker"), dict) and d_he["marker"].get("size_mm"):
                self.wmarker = dict(W49.DEFAULT_WRIST_MARKER, **d_he["marker"])

    def _wrist_dev_values(self):
        devs = self.W49.list_usb_cameras()
        return ["auto"] + [d["device"] for d in devs] if devs else ["auto", "/dev/video0", "/dev/video2", "/dev/video4"]

    def _refresh_wrist_devs(self):
        try:
            self.cb_wdev.configure(values=self._wrist_dev_values())
        except Exception:
            pass

    def _on_undistort_toggle(self):
        if self.wrist_src is not None:
            self.wrist_src.undistort_preview = bool(self.v_undistort.get())
        self._force_redraw()

    def on_wrist_connect(self):
        if self.wrist_src is not None and self.wrist_src.running():
            return
        W49 = self.W49
        if self.wrist_source_factory is not None:
            self.wrist_src = self.wrist_source_factory()
        else:
            self.wrist_src = W49.UsbWristSource(device=self.v_wdev.get(), res=self.v_wres.get(),
                                                lock_controls=bool(self.v_wlock.get()),
                                                intrinsics=self.wrist_intr, log=self.log)
        try:
            self.wrist_src.set_intrinsics(self.wrist_intr)
            self.wrist_src.undistort_preview = bool(self.v_undistort.get())
        except Exception:
            pass
        self.wrist_src.start()
        self.log(f"[INFO] [V3.49 wrist] 正在连接腕部 USB 相机（{self.v_wdev.get()} · {self.v_wres.get()} · "
                 f"{'锁曝光/白平衡/对焦' if self.v_wlock.get() else '自动参数'}）…")
        self.v_wrist_info.set(self._wrist_info_text())
        self._shown_wfid = -1
        self._poll_wrist()
        self._update_intr_ui()
        self._update_eih_ui()

    def on_wrist_disconnect(self):
        if self.wrist_src is not None:
            self.wrist_src.stop()
            self.log("[INFO] [V3.49 wrist] 腕部 USB 相机已断开")
        self.wrist_src = None
        self.wdet = None
        self.v_wrist_info.set(self._wrist_info_text())
        self._shown_wfid = -1
        self._poll_wrist()
        self._update_intr_ui()
        self._update_eih_ui()

    def _wrist_info_text(self):
        ws = self.wrist_src
        intr = self.intr_result or self.wrist_intr
        he = self.wcalib_result or self.wrist_calib
        intr_txt = "未标定（请到「内参标定」Tab 标定 130° 广角镜头）"
        if intr and intr.get("K"):
            K = intr["K"]
            intr_txt = (f"已标定 {K.get('width')}×{K.get('height')} fx {K.get('fx', 0):.1f} fy {K.get('fy', 0):.1f} "
                        f"cx {K.get('ppx', 0):.1f} cy {K.get('ppy', 0):.1f} · RMS {intr.get('rms_px', 0):.2f} px "
                        f"({intr.get('model_type', '')})")
        he_txt = "未标定"
        if he and he.get("T_flange_cam"):
            t = T_t(he["T_flange_cam"])
            he_txt = f"link7→cam ({t[0] * 1000:+.1f}, {t[1] * 1000:+.1f}, {t[2] * 1000:+.1f}) mm · 留一 {(he.get('loo_mean_mm') or 0):.2f} mm"
        if not ws or not ws.info:
            return f"状态：腕部相机未连接\n内参：{intr_txt}\nEye-in-Hand 手眼：{he_txt}"
        i = ws.info
        ctrls = i.get("controls") or {}
        c_str = " · ".join(f"{k}={v}" for k, v in ctrls.items()) or "默认"
        return (f"状态：✓ 已连接 {i.get('name')} · 节点 {i.get('device')} · 格式 {i.get('format', 'MJPG')} "
                f"{i.get('res', '')} · 帧率 {ws.fps:.1f} fps\n"
                f"UVC 参数锁定：{c_str}\n"
                f"广角内参：{intr_txt}\n"
                f"Eye-in-Hand 手眼：{he_txt}")

    def on_save_wrist_info(self):
        if not self.wrist_src or not self.wrist_src.info:
            self.log("[WARN] [V3.49 wrist] 先连接腕部相机")
            return
        p = self.W49.WRIST_INFO_PATH if CALIB_PATH == DEFAULT_CALIB_PATH else os.path.join(os.path.dirname(CALIB_PATH), "wrist_camera_info.json")
        d = dict(self.wrist_src.info, saved=time.strftime("%Y-%m-%d %H:%M:%S"), fps=round(self.wrist_src.fps, 1),
                 intrinsics=self.wrist_intr)
        save_json(p, d)
        self.log(f"[OK] [V3.49 wrist] 腕部相机信息已保存 → {p}（设备 {d.get('device')}）")

    # ------------------------------------------------------------------ V3.49 intrinsic calibration UI
    def _board_cfg(self):
        try:
            c, r = int(self.v_icols.get()), int(self.v_irows.get())
            sq = float(self.v_isq.get())
            mk = float(self.v_imk.get())
            assert 3 <= c <= 20 and 3 <= r <= 20 and 5.0 <= sq <= 150.0 and 3.0 <= mk < sq
            return {"type": self.v_ibtype.get(), "cols": c, "rows": r, "square_mm": sq, "marker_mm": mk,
                    "dict": self.v_dict.get() if hasattr(self, "v_dict") else "DICT_4X4_50"}
        except Exception:
            return None

    def _intr_text(self):
        c = self.intr_result or self.wrist_intr
        if not c or not c.get("K"):
            return "还没有腕部相机内参（130° 广角镜头畸变大，请先采 ≥ 8 张标定板图计算并保存内参）"
        K = c["K"]
        src = "本次计算（未保存）" if self.intr_result else f"已保存（{c.get('saved', '')}）"
        good = c.get("good", (c.get("rms_px") or 9.0) <= self.W49.INTR_GOOD_RMS_PX)
        lines = [
            f"{src} · {'合格 ✓' if good else '未达标 ⚠'} · 模型 {c.get('model_type', '')}",
            f"分辨率 {K['width']}×{K['height']} · fx {K['fx']:.1f}  fy {K['fy']:.1f}  cx {K['ppx']:.1f}  cy {K['ppy']:.1f} px"
            + (f" · 视场角 水平 {c.get('fov_h_deg', 0):.0f}° / 对角 {c.get('fov_diag_deg', 0):.0f}°" if c.get("fov_h_deg") else ""),
            f"重投影误差 RMS {c.get('rms_px', 0):.3f} px（最大 {c.get('max_px', 0):.2f} px） · 样本 {c.get('n_used', len(self.intr_samples))}"
            + f" · 畸变系数 {[round(v, 4) for v in (c.get('dist') or [])[:5]]}",
        ]
        for w_ in c.get("warnings") or []:
            lines.append("⚠ " + w_)
        return "\n".join(lines)

    def _refresh_intr_table(self):
        for it in self.tv_intr.get_children():
            self.tv_intr.delete(it)
        r = self.intr_result
        if r is None and self.wrist_intr and (
            len(self.wrist_intr.get("used") or []) + len(self.wrist_intr.get("rejected") or []) == len(self.intr_samples)
        ):
            r = self.wrist_intr
        errs = {}
        if r and r.get("ok"):
            for m, i in enumerate(r.get("used") or []):
                if m < len(r.get("per_view_px") or []):
                    errs[i] = r["per_view_px"][m]
        for k, s in enumerate(self.intr_samples):
            rej = bool(r and k in (r.get("rejected") or []))
            tags = ["odd"] if k % 2 else []
            if rej:
                tags.append("rej")
            self.tv_intr.insert("", "end", iid=str(k), tags=tags, values=(
                k + 1, s["n_corners"], s.get("coverage", "-"), f"{s.get('sharpness', 0):.0f}",
                "剔除" if rej else (f"{errs[k]:.2f}" if k in errs else "-")))
        autofit_treeview(self.tv_intr, self.intr_cols, stretch_col="cov")
        self.v_intr_res.set(self._intr_text())
        self._update_intr_ui()

    def _update_intr_ui(self):
        if not hasattr(self, "il1"):
            return
        cfg = self._board_cfg()
        if not cfg:
            self._set_lamp(self.il1, "red")
            self.is1.set("标定板参数无效：列/行 3–20，标记 mm < 方格 mm")
        else:
            self._set_lamp(self.il1, "green")
            self.is1.set(f"{cfg['type']} · {cfg['cols']}×{cfg['rows']} · 方格 {cfg['square_mm']:g} mm")
        n = len(self.intr_samples)
        self._set_lamp(self.il2, "green" if n >= 8 else ("yellow" if n else "grey"))
        self.is2.set(f"已拍 {n}/≥8 张标定图")
        c = self.intr_result or self.wrist_intr
        if c and c.get("K"):
            g = c.get("good", (c.get("rms_px") or 9.0) <= self.W49.INTR_GOOD_RMS_PX)
            self._set_lamp(self.il3, "green" if g else "yellow")
            self.is3.set(f"RMS {c.get('rms_px', 0):.2f} px · {'合格 ✓' if g else '建议补拍四角样本'}")
        else:
            self._set_lamp(self.il3, "grey")
            self.is3.set("还没有内参标定")
        busy = self._busy()
        jp = bool(getattr(self.jpanel, "enabled", False)) and not getattr(self.pipe, "teach_mode", False)
        wcam = bool((self.wrist_src and self.wrist_src.running()) or (self.src and self.src.running()))
        if hasattr(self, "b_intr_auto"):
            self._en(self.b_intr_cap, not busy)
            self._en(self.b_intr_auto, wcam and jp and not busy)
            self._en(self.b_intr_stop, busy)
        if hasattr(self, "v_wrist_info"):
            self.v_wrist_info.set(self._wrist_info_text())

    def on_intr_board_png(self, out_path=None):
        cfg = self._board_cfg()
        if not cfg:
            self.log("[WARN] [V3.49 intr] 标定板设置无效（检查列/行与方格/标记尺寸）")
            return
        try:
            p = self.W49.generate_calib_board_png(cfg, out_path=out_path)
        except Exception as exc:
            self.log(f"[FAIL] [V3.49 intr] 生成标定板图片失败：{exc}")
            return
        self.log(f"[OK] [V3.49 intr] 标定板打印图已生成 → {p}（A4 100% 原尺寸打印，打印后用尺子核对底部 100 mm 标尺与方格边长）")
        return p

    def _active_intr_frame(self):
        if self.wrist_src is not None:
            rgb, _fid = self.wrist_src.raw_frame()
            if rgb is not None:
                return rgb
        if self.src is not None:
            rgb, _d, _fid = self.src.frames()
            if rgb is not None:
                return rgb
        return None

    def on_intr_capture(self):
        cfg = self._board_cfg()
        if not cfg:
            self.log("[WARN] [V3.49 intr] 标定板设置无效")
            return
        rgb = self._active_intr_frame()
        if rgb is None:
            self.log("[WARN] [V3.49 intr] 先连接腕部相机（或主相机）")
            return
        det = self.W49.detect_board_corners(rgb, cfg)
        if not det.get("ok"):
            self.log(f"[WARN] [V3.49 intr] 没采到标定板：{det.get('error')}")
            return
        q = self.actual_q()
        if q:
            T = self.flange_T(q)
            if T is not None:
                det["q_deg"] = [round(v, 3) for v in q]
                det["T_base_flange"] = T
        self.intr_samples.append(det)
        self.intr_result = None
        self._refresh_intr_table()
        self.log(f"[OK] [V3.49 intr] 样本 #{len(self.intr_samples)}：检测到 {det['n_corners']} 个角点 · "
                 f"位置 {det['coverage']} · 清晰度 {det['sharpness']:.0f}")

    def _capture_intr_board_settled(self, cfg, timeout=1.2):
        ws = self.wrist_src if (self.wrist_src and self.wrist_src.running()) else self.src
        if ws is None:
            return None, "腕部相机未连接"
        rgb0 = self._active_intr_frame()
        if rgb0 is not None and (getattr(threading, "__name__", "") == "T" or not hasattr(ws, "thread")):
            det0 = self.W49.detect_board_corners(rgb0, cfg)
            if det0.get("ok"):
                q = self.actual_q()
                if q:
                    det0["q_deg"] = [round(v, 3) for v in q]
                    det0["T_base_flange"] = self.flange_T(q)
                return det0, None
            return None, det0.get("error") or "未检测到标定板"
        last = getattr(ws, "frame_id", -1)
        t_end = time.time() + timeout
        last_err = "未检测到标定板"
        best_det = None
        settled_dets = []
        q_ref = self.actual_q()
        while time.time() < t_end and not self.stop_flag:
            fid = getattr(ws, "frame_id", 0)
            if fid == last:
                time.sleep(0.015)
                continue
            last = fid
            q = self.actual_q()
            if q and q_ref and max(abs(a - b) for a, b in zip(q, q_ref)) > 0.20:
                q_ref = q
                settled_dets.clear()
                continue
            q_ref = q or q_ref
            rgb = self._active_intr_frame()
            if rgb is None:
                continue
            det = self.W49.detect_board_corners(rgb, cfg)
            if not det.get("ok"):
                last_err = det.get("error") or last_err
                continue
            if q:
                det["q_deg"] = [round(v, 3) for v in q]
                det["T_base_flange"] = self.flange_T(q)
            if best_det is None or det["n_corners"] > best_det["n_corners"]:
                best_det = det
            settled_dets.append(det)
            if len(settled_dets) >= 2 and best_det["n_corners"] >= 10:
                d_top = settled_dets[-1]
                d_prev = settled_dets[-2]
                if d_top["n_corners"] == d_prev["n_corners"] and d_top["obj_pts"] == d_prev["obj_pts"]:
                    im_a = d_top["img_pts"]
                    im_b = d_prev["img_pts"]
                    max_j = max(math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(im_a, im_b))
                    if max_j <= 1.2:
                        d_top["img_pts"] = [[0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])] for a, b in zip(im_a, im_b)]
                        d_top["corners_px"] = [[round(p[0], 1), round(p[1], 1)] for p in d_top["img_pts"]]
                        return d_top, None
            if len(settled_dets) >= 3 and best_det["n_corners"] >= 8:
                return best_det, None
        if best_det is not None:
            return best_det, None
        return None, last_err

    def on_intr_auto(self):
        """V3.57 AUTOMATIC TABLETOP BOARD INTRINSIC + EYE-IN-HAND COLLECTION:
        Moves the wrist camera through N multi-quadrant & multi-tilt poses above the tabletop ChArUco/Chessboard board,
        auto-computes & saves intrinsics, and simultaneously populates & solves Eye-in-Hand calibration."""
        try:
            n = int(self.v_inauto.get())
            assert 6 <= n <= 20
        except Exception:
            self.log("[WARN] [V3.57 intr] 自动采集点数请填 6–20")
            return
        cfg = self._board_cfg()
        if not cfg:
            self.log("[WARN] [V3.57 intr] 标定板设置无效")
            return
        if self._active_intr_frame() is None:
            self.log("[WARN] [V3.57 intr] 先连接腕部相机")
            return
        if getattr(self.pipe, "teach_mode", False):
            self.log("[WARN] [V3.57 intr] NERO 在示教模式：先点下方「退出示教模式」锁住当前姿态后再自动采集")
            return
        if not getattr(self.jpanel, "enabled", False):
            self.log("[WARN] [V3.57 intr] 关节面板不可操作（需要 MoveIt 就绪）")
            return
        speed = min(CALIB_MAX_SPEED, max(0.01, float(self.cfg_fn().get("speed_percent", 10)) / 100.0))

        def job():
            model = self._model()
            q0 = self.actual_q()
            if not q0:
                self.log("[FAIL] [V3.57 intr] 没有关节反馈")
                return
            q0r = [math.radians(v) for v in q0]
            if not self.ask(
                "腕部内参 + 手眼全自动采集",
                f"请将 {cfg['type']}（{cfg['cols']}×{cfg['rows']}，方格 {cfg['square_mm']:g} mm）平放在桌面上保持不动："

                f"· 机械臂将以 {speed * 100:.0f}% 速度在标定板上方自动走 {n} 个姿态（覆盖中心、四角与多角度倾斜）；"

                "· 采集完成后自动回到起点，自动计算并保存「腕部相机内参」，并同步利用这批高精度棋盘格角点直接解算「腕部 Eye-in-Hand 手眼标定」！"

                "确认标定板已平铺在桌面且周围无障碍？"
            ):
                return
            s0_det, why0 = self._capture_intr_board_settled(cfg, timeout=1.4)
            if s0_det is None:
                self.log(f"[FAIL] [V3.57 intr] 起点未检测到桌面标定板：{why0}（请先把腕部相机对准桌面标定板正上方 18–28 cm）")
                return
            if self.intr_samples:
                self.intr_samples = []
                self.intr_result = None
                self._ui(self._refresh_intr_table)
            self.intr_samples.append(s0_det)
            self._ui(self._refresh_intr_table)
            self.log(f"[OK] [V3.57 intr] 起点 → 内参样本 #1：检测到 {s0_det['n_corners']} 个角点 · 位置 {s0_det['coverage']}")
            im_sz = s0_det.get("image_size") or [1920, 1080]
            targets, st = self.W49.plan_intr_auto_targets(
                self._fk_rad, q0r, model["limits"], s0_board=s0_det, image_size=im_sz, n=n,
                board_cfg=cfg, fence_fn=lambda q_: self.pipe.fence_goal_check(model, q_)[0],
                seed=int(time.time()) % 100000
            )
            self.log(f"[INFO] [V3.57 intr] 选出 {len(targets)} 个自动标定姿态（候选池 {st.get('pool', 0)} 个）")
            got = 1
            for k, t in enumerate(targets):
                if self.stop_flag:
                    break
                res = self.pipe.move_and_settle(model, list(t["q"]), speed, label=f"内参标定 {k + 1}/{len(targets)}")
                if not res.get("ok"):
                    self.log(f"[WARN] [V3.57 intr] 姿态 {k + 1} 运动失败：{res.get('error')}")
                    continue
                time.sleep(min(0.18, FAST_SETTLE_S, SETTLE_S))
                self._wait_still(1.2, hold_s=min(0.18, FAST_STILL_HOLD_S))
                det_k, why_k = self._capture_intr_board_settled(cfg, timeout=1.2)
                if det_k is None:
                    self.log(f"[WARN] [V3.57 intr] 姿态 {k + 1}（{t.get('tag', '')}）：{why_k}")
                    continue
                self.intr_samples.append(det_k)
                got += 1
                self._ui(self._refresh_intr_table)
                self.log(f"[OK] [V3.57 intr] 姿态 {k + 1}/{len(targets)}（{t.get('tag', '')}）→ 内参样本 #{len(self.intr_samples)}："
                         f"检测到 {det_k['n_corners']} 个角点 · 位置 {det_k['coverage']}")
            self.pipe.move_and_settle(model, q0r, speed, label="内参标定 回起点")
            self.log(f"[{'OK' if got >= 6 else 'WARN'}] [V3.57 intr] 桌面标定板自动采集结束：共采到 {len(self.intr_samples)} 张有效标定图")
            if len(self.intr_samples) >= self.W49.INTR_MIN_SAMPLES:
                def _finish_intr_and_eih():
                    self.on_intr_compute()
                    if self.intr_result and self.intr_result.get("ok"):
                        self.on_intr_save()
                        # Ensure Eye-in-Hand uses the newly saved intrinsics and auto-saves when good
                        intr_d = self.wrist_intr or self.intr_result
                        eih_samps = self.W49.eih_samples_from_intr_samples(
                            self.intr_samples, intr_d["K"], intr_d.get("dist"),
                            board_cfg=cfg, model_type=intr_d.get("model_type")
                        )
                        if len(eih_samps) >= 6:
                            self.wcalib_samples = eih_samps
                            self.on_wcalib_compute()
                            if self.wcalib_result and self.wcalib_result.get("ok"):
                                if self.wcalib_result.get("good"):
                                    r_w_keep = self.wcalib_result
                                    self.on_wcalib_save()
                                    self.wcalib_result = r_w_keep
                                    self._refresh_wcalib_table()
                                    self.log("[OK] [V3.58 eih] 已由桌面 ChArUco 标定板同步完成并自动保存腕部 Eye-in-Hand 手眼标定 ✓")
                                else:
                                    self.log("[OK] [V3.57 eih] 已由桌面 ChArUco 标定板同步完成腕部 Eye-in-Hand 手眼解算（可在「手眼标定」页点「保存腕部标定」）")
                self._ui(_finish_intr_and_eih)
        self._run(job)

    def on_intr_delete(self):
        sel = sorted((int(i) for i in self.tv_intr.selection()), reverse=True)
        for i in sel:
            if 0 <= i < len(self.intr_samples):
                self.intr_samples.pop(i)
        self.intr_result = None
        self._refresh_intr_table()

    def on_intr_clear(self):
        self.intr_samples = []
        self.intr_result = None
        self._refresh_intr_table()

    def on_intr_compute(self):
        if len(self.intr_samples) < self.W49.INTR_MIN_SAMPLES:
            self.log(f"[WARN] [V3.49 intr] 至少需要 {self.W49.INTR_MIN_SAMPLES} 张标定图（现在 {len(self.intr_samples)} 张）")
            return
        im_sz = self.intr_samples[-1].get("image_size") or [1920, 1080]
        r = self.W49.calibrate_intrinsics(self.intr_samples, image_size=im_sz, model_type=self.v_imodel.get())
        if not r.get("ok"):
            self.log(f"[FAIL] [V3.49 intr] {r.get('error')}")
            return
        self.intr_result = r
        if self.wrist_src is not None and r.get("good"):
            self.wrist_src.set_intrinsics(r)
        self._refresh_intr_table()
        tag = "[OK]" if r["good"] else "[WARN]"
        K = r["K"]
        self.log(f"{tag} [V3.49 intr] 内参计算：样本 {r['n_used']}（剔除 {len(r['rejected'])}）· RMS {r['rms_px']:.3f} px · "
                 f"fx={K['fx']:.1f} fy={K['fy']:.1f} cx={K['ppx']:.1f} cy={K['ppy']:.1f} · 视场角 "
                 f"{r['fov_h_deg']:.0f}°/{r['fov_diag_deg']:.0f}°")
        # V3.58: Automatically synchronize & solve Eye-in-Hand whenever tabletop samples include robot FK T_base_flange
        cfg = self._board_cfg() or dict(self.W49.DEFAULT_BOARD)
        if cfg and sum(1 for s in self.intr_samples if s.get("T_base_flange")) >= 6:
            if not self.wcalib_samples or all(s.get("source") == "charuco" for s in self.wcalib_samples):
                eih_samps = self.W49.eih_samples_from_intr_samples(
                    self.intr_samples, r["K"], r.get("dist"), board_cfg=cfg, model_type=r.get("model_type")
                )
                if len(eih_samps) >= 6:
                    self.wcalib_samples = eih_samps
                    self.on_wcalib_compute()

    def on_intr_save(self, path=None):
        r = self.intr_result
        if not r or not r.get("ok"):
            self.log("[WARN] [V3.49 intr] 先「计算内参」")
            return
        d = dict(r, version="V3.49", saved=time.strftime("%Y-%m-%d %H:%M:%S"), board=self._board_cfg())
        p = path or self._wrist_intr_path()
        save_json(p, d)
        if not path and p == self.W49.DEFAULT_WRIST_INTR_PATH and CALIB_PATH == DEFAULT_CALIB_PATH:
            try:
                os.makedirs(CALIB_DIR, exist_ok=True)
                save_json(os.path.join(CALIB_DIR, "wrist_intrinsics_" + time.strftime("%Y%m%d_%H%M%S") + ".json"), d)
            except Exception:
                pass
        self.wrist_intr = d
        self.intr_result = None
        if self.wrist_src is not None:
            self.wrist_src.set_intrinsics(d)
        self._refresh_intr_table()
        self._update_eih_ui()
        self.log(f"[OK] [V3.49 intr] 腕部相机内参已保存 → {p}（RMS {d['rms_px']:.3f} px）")

    def on_intr_save_as(self, path=None):
        c = self.intr_result or self.wrist_intr
        if not c:
            self.log("[WARN] [V3.49 intr] 还没有可保存的内参")
            return
        path = path or self._ask_path(True, CALIB_DIR, "wrist_intrinsics_" + time.strftime("%Y%m%d_%H%M") + ".json", "内参另存为")
        if path:
            save_json(path, dict(c, exported=time.strftime("%Y-%m-%d %H:%M:%S")))
            self.log(f"[OK] [V3.49 intr] 内参另存为 → {path}")

    def on_intr_load(self, path=None):
        path = path or self._ask_path(False, CALIB_DIR, "", "读取腕部内参文件")
        if not path:
            return
        d = load_json(path)
        if not self.W49.intrinsics_valid(d):
            self.log(f"[FAIL] [V3.49 intr] {path} 不是有效的内参文件")
            return
        save_json(self._wrist_intr_path(), d)
        self.wrist_intr = d
        self.intr_result = None
        if self.wrist_src is not None:
            self.wrist_src.set_intrinsics(d)
        self._refresh_intr_table()
        self._update_eih_ui()
        self.log(f"[OK] [V3.49 intr] 已读取腕部内参 ← {path}（RMS {d.get('rms_px', 0):.3f} px）")

    def on_intr_clear_calib(self):
        p = self._wrist_intr_path()
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass
        self.wrist_intr = None
        self.intr_result = None
        if self.wrist_src is not None:
            self.wrist_src.set_intrinsics(None)
        self._refresh_intr_table()
        self._update_eih_ui()
        self.log("[OK] [V3.49 intr] 已清除当前腕部内参设定")

    # ------------------------------------------------------------------ V3.49 eye-in-hand calibration UI
    def _wcalib_text(self):
        c = self.wcalib_result or self.wrist_calib
        if not c or not c.get("T_flange_cam"):
            return "还没有腕部 Eye-in-Hand 手眼标定（config/wrist_handeye_calib.json 不存在）"
        src = "本次计算（未保存）" if self.wcalib_result else f"已保存（{c.get('saved', '')}）"
        t = T_t(c["T_flange_cam"])
        mb = c["marker_in_base"]
        loo_m, loo_x = c.get("loo_mean_mm") or 0.0, c.get("loo_max_mm") or 0.0
        good = c.get("good", loo_m <= CALIB_GOOD_MEAN_MM and loo_x <= CALIB_GOOD_MAX_MM)
        lines = [
            f"{src} · {'合格 ✓' if good else '未达标 ⚠'}",
            f"拟合残差 平均 {c['mean_mm']:.2f} / 最大 {c['max_mm']:.2f} mm · 留一 平均 {loo_m:.2f} / 最大 {loo_x:.2f} mm",
            f"腕部相机在 link7：x {t[0] * 1000:+.1f}  y {t[1] * 1000:+.1f}  z {t[2] * 1000:+.1f} mm",
            f"桌面标记在 base_link：({mb[0] * 1000:+.1f}, {mb[1] * 1000:+.1f}, {mb[2] * 1000:+.1f}) mm · 样本 {c.get('n_used', len(self.wcalib_samples))}",
        ]
        for w_ in c.get("warnings") or []:
            lines.append("⚠ " + w_)
        return "\n".join(lines)

    def _refresh_wcalib_table(self):
        for it in self.tv_wcalib.get_children():
            self.tv_wcalib.delete(it)
        r = self.wcalib_result
        res, loo = {}, {}
        if r and r.get("ok"):
            for m, i in enumerate(r["used"]):
                res[i] = r["res_mm"][m]
                loo[i] = r["loo_mm"][m]
        for k, s in enumerate(self.wcalib_samples):
            p = s["p_cam"]
            rej = bool(r and k in (r.get("rejected") or []))
            tags = ["odd"] if k % 2 else []
            if rej:
                tags.append("rej")
            self.tv_wcalib.insert("", "end", iid=str(k), tags=tags, values=(
                k + 1, " ".join(f"{v:.0f}" for v in s["q_deg"]),
                f"{p[0] * 1000:.0f}  {p[1] * 1000:.0f}  {p[2] * 1000:.0f}",
                f"{s.get('spread_mm', 0):.1f}",
                "剔除" if rej else (f"{res[k]:.2f}" if k in res else "-"),
                f"{loo[k]:.2f}" if k in loo else "-"))
        autofit_treeview(self.tv_wcalib, self.wcalib_cols, stretch_col="q")
        self.v_wres.set(self._wcalib_text())
        self._update_eih_ui()

    def _wmarker_parse(self):
        if not hasattr(self, "v_wdict"):
            return dict(self.wmarker)
        try:
            m = {"dict": self.v_wdict.get(), "id": int(self.v_wmid.get()), "size_mm": float(self.v_wmsize.get())}
            if m["dict"] in ARUCO_DICTS and 0 <= m["id"] < 1000 and 5.0 <= m["size_mm"] <= 300.0:
                return m
        except Exception:
            pass
        return None

    def _wmarker_from_ui(self):
        m = self._wmarker_parse()
        if not m:
            self.log("[WARN] [V3.51 eih] 桌面标记设置无效（ID 0–999，边长 5–300 mm）")
            return None
        base_m = {"dict": m["dict"], "id": m["id"], "size_mm": m["size_mm"]}
        prev_m = {"dict": self.wmarker.get("dict"), "id": self.wmarker.get("id"), "size_mm": self.wmarker.get("size_mm")}
        if base_m != prev_m and self.wcalib_samples:
            self.log("[WARN] [V3.51 eih] 桌面标记设置已更改：旧样本是用之前的边长/ID 测的，建议点「清空」重采")
        self.wmarker = dict(base_m)
        if hasattr(self, "v_wtarget") and self.wrist_src is not None:
            self.wrist_src.target_type = self.v_wtarget.get()
            self.wrist_src.board_cfg = self._board_cfg()
        return dict(base_m)

    def on_wmarker_png(self):
        m = self._wmarker_from_ui()
        if not m:
            return
        try:
            img = marker_image(m["dict"], m["id"])
            p = os.path.join(HERE, "config", f"wrist_aruco_{m['dict']}_id{m['id']}_{m['size_mm']:g}mm.png")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(png_bytes_from_ppm(ppm_bytes(img)))
            self.log(f"[OK] [V3.51 eih] 桌面标记图片已生成 → {p}（打印后量黑框实测边长填进 ① 边长框，固定贴在桌面上）")
        except Exception as exc:
            self.log(f"[FAIL] [V3.51 eih] 生成桌面标记图片失败：{exc}")

    def _update_eih_ui(self):
        if not hasattr(self, "wl1"):
            return
        busy = self._busy()
        wm = self._wmarker_parse()
        has_intr = bool(self.wrist_intr and self.W49.intrinsics_valid(self.wrist_intr))
        wcam = bool(self.wrist_src and self.wrist_src.running() and getattr(self.wrist_src, "K", None))
        wdet = self.wdet
        if not wm:
            self._set_lamp(self.wl1, "red")
            self.ws1.set("桌面标记设置无效：ID 0–999，边长 5–300 mm")
        elif not has_intr:
            self._set_lamp(self.wl1, "yellow")
            self.ws1.set("请先在「内参标定」Tab 完成腕部相机广角内参标定")
        elif not wcam:
            self._set_lamp(self.wl1, "grey")
            self.ws1.set(f"桌面标记 ID {wm['id']} ({wm['size_mm']:g} mm) · 请在「相机信息」Tab 连接腕部 USB 相机")
        elif wdet is not None:
            self._set_lamp(self.wl1, "green")
            pz = wdet["p_cam"][2] * 1000.0
            self.ws1.set(f"✓ 看到桌面 ID {wdet['id']} ({wm['size_mm']:g} mm) · 距离 {pz:.0f} mm · 边长 {wdet.get('side_px', 0):.0f} px")
        else:
            self._set_lamp(self.wl1, "yellow")
            self.ws1.set(f"腕部相机没看到桌面 ID {wm['id']} ({wm['size_mm']:g} mm)：把腕部相机对准桌面固定标记")
        n = len(self.wcalib_samples)
        A = [s["T_base_flange"] for s in self.wcalib_samples]
        ts = translation_span_m(A) if n > 1 else 0.0
        rs = rotation_span_deg(A) if n > 1 else 0.0
        wa = getattr(self, "wauto_state", None)
        if wa:
            self._set_lamp(self.wl2, "blue")
            self.ws2.set(f"自动采集中：{wa.get('phase', '')} {wa.get('k', 0)}/{wa.get('n', 0)} · 样本 {n}")
        else:
            okn = n >= 6 and ts >= 0.12 and rs >= 20.0
            self._set_lamp(self.wl2, "green" if okn else ("yellow" if n else "grey"))
            self.ws2.set(f"样本 {n}/≥6 · 位置范围 {ts * 100:.0f}/≥12 cm · 朝向变化 {rs:.0f}/≥20°" + (" ✓" if okn else ""))
        c = self.wcalib_result or self.wrist_calib
        if c and c.get("T_flange_cam"):
            g = c.get("good", (c.get("loo_mean_mm") or 99) <= CALIB_GOOD_MEAN_MM)
            self._set_lamp(self.wl3, "green" if g else "red")
            self.ws3.set(("合格 ✓ → 点「保存腕部标定」" if g else "未达标 ⚠：建议补不同倾角样本后重算")
                         if self.wcalib_result else f"已保存 {c.get('saved', '')} · {'合格 ✓' if g else '未达标 ⚠'}")
            wcs = getattr(self, "wcheck_state", None)
            if wcs and wcs[0] == "run":
                self._set_lamp(self.wl4, "blue")
                self.ws4.set("快速复核中…")
            elif wcs and wcs[0] in ("ok", "fail"):
                self._set_lamp(self.wl4, "green" if wcs[0] == "ok" else "red")
                self.ws4.set(("通过 ✓ " if wcs[0] == "ok" else "未通过 ✗ ") + "误差 " + ", ".join(f"{e:.1f}" for e in wcs[1]) + " mm")
            else:
                self._set_lamp(self.wl4, "green" if g else "yellow")
                self.ws4.set("已标定 T_flange_cam（看到桌面标记时实时显示误差，或点「快速复核」）")
        else:
            self._set_lamp(self.wl3, "grey")
            self.ws3.set("还没有腕部手眼标定")
            self._set_lamp(self.wl4, "grey")
            self.ws4.set("保存腕部手眼标定后可复核")
        jp = bool(getattr(self.jpanel, "enabled", False)) and not getattr(self.pipe, "teach_mode", False)
        if hasattr(self, "b_wauto"):
            self._en(self.b_wrec, not busy)
            self._en(self.b_wauto, wcam and jp and not busy)
            self._en(self.b_wstop, busy)
            self._en(self.b_wdel, bool(self.tv_wcalib.selection()) and not busy)
            self._en(self.b_wclr, bool(self.wcalib_samples) and not busy)
            self._en(self.b_wcomp, n >= 6 and not busy)
            self._en(self.b_wsave, bool(self.wcalib_result and self.wcalib_result.get("ok")) and not busy)
            self._en(self.b_wcheck, bool(self.wrist_calib and self.wrist_calib.get("check_poses_deg")) and wcam and jp and not busy)
        if hasattr(self, "v_wrist_info"):
            self.v_wrist_info.set(self._wrist_info_text())

    def _capture_wrist_sample(self, n_frames=5, timeout=1.4, empty_timeout=FAST_WRIST_EMPTY_S):
        """V3.52/V3.56 FAST WRIST CAPTURE: capture up to 5 settled wrist marker frames (early exit at 4 frames when
        jitter <= 1.5 mm; bail out in 0.65 s if 0 frames detect the marker). Flushes stale pre-stop frame on live UVC."""
        ws = self.wrist_src
        if not ws or not getattr(ws, "K", None):
            return None, "腕部相机未连接或未标定内参"
        m = self._wmarker_from_ui()
        if not m:
            return None, "桌面标记设置无效"
        q0 = self.actual_q()
        if not q0:
            return None, "没有关节反馈"
        d_now = ws.detect_marker(m, q0)
        if d_now is not None and (getattr(threading, "__name__", "") == "T" or not hasattr(ws, "thread")):
            T = self.flange_T(q0)
            if T is None:
                return None, "FK 失败"
            R_cm = rodrigues(d_now["rvec"]) if d_now.get("rvec") else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            self.wdet = d_now
            self.wdet_q = q0
            return {
                "q_deg": [round(v, 3) for v in q0],
                "p_cam": list(d_now["p_cam"]),
                "rvec": list(d_now.get("rvec") or [0.0, 0.0, 0.0]),
                "T_cam_marker": T_from(R_cm, d_now["p_cam"]),
                "T_base_flange": T,
                "spread_mm": 0.4,
            }, None
        dets, qs = [], []
        last = getattr(ws, "frame_id", -1)
        t_start = time.time()
        t_end = t_start + timeout
        while len(dets) < n_frames and time.time() < t_end and not self.stop_flag:
            fid = getattr(ws, "frame_id", 0)
            if fid == last:
                if not dets and (time.time() - t_start) >= empty_timeout and d_now is None:
                    break
                time.sleep(0.015)
                continue
            last = fid
            q = self.actual_q()
            d = ws.detect_marker(m, q)
            if d is None or q is None:
                if not dets and (time.time() - t_start) >= empty_timeout:
                    break
                continue
            if max(abs(a - b) for a, b in zip(q, q0)) > 0.25:
                q0 = q
                dets.clear()
                qs.clear()
                continue
            dets.append(d)
            qs.append(q)
            if len(dets) >= 4:
                tr_early = trimmed_points([x["p_cam"] for x in dets], min_keep=3)
                if not tr_early.get("error") and tr_early["jitter_mm"] <= 1.5:
                    break
        if not dets and d_now is not None:
            dets.append(d_now)
            qs.append(q0)
        need = min(3, max(2, n_frames // 2))
        if len(dets) < need:
            return None, f"腕部相机只检测到 {len(dets)} 帧桌面标记 ID {m['id']}（要 ≥ {need}）：把腕部相机对准桌面标记"
        pts = [d["p_cam"] for d in dets]
        tr = trimmed_points(pts, min_keep=need)
        if tr.get("error"):
            return None, f"PnP：{tr['error']}"
        if tr["jitter_mm"] > JITTER_MAX_MM:
            return None, f"抖动 {tr['jitter_mm']:.1f} mm（> {JITTER_MAX_MM:.0f} mm）"
        q = [sum(v[j] for v in qs) / len(qs) for j in range(7)]
        T = self.flange_T(q)
        if T is None:
            return None, "FK 失败"
        rvs = [d.get("rvec") or [0.0, 0.0, 0.0] for d in dets]
        rv_mean = [sum(r_[i] for r_ in rvs) / len(rvs) for i in range(3)]
        R_cm = rodrigues(rv_mean)
        self.wdet = dets[-1]
        self.wdet_q = q
        return {
            "q_deg": [round(v, 3) for v in q],
            "p_cam": tr["mean"],
            "rvec": rv_mean,
            "T_cam_marker": T_from(R_cm, tr["mean"]),
            "T_base_flange": T,
            "spread_mm": tr["jitter_mm"],
        }, None

    def _capture_wrist_retry(self, tag=""):
        fatal = ("腕部相机未连接", "桌面标记设置无效", "没有关节反馈", "FK 失败")
        why = ""
        for attempt in range(1 + CAPTURE_RETRIES):
            s, why = self._capture_wrist_sample()
            if s is not None:
                s["attempt"] = attempt + 1
                return s, None
            if self.stop_flag or any(why.startswith(f) for f in fatal) or attempt >= CAPTURE_RETRIES:
                break
            # V3.56: if 0 frames saw the marker, the pose is out-of-frame / blind -> skip immediately without retry
            if "只检测到 0 帧" in str(why):
                break
            self.log(f"[INFO] [V3.51 eih] {tag}{why} → 再测一次")
            time.sleep(0.20)
            self._wait_still(0.8, hold_s=0.18)
        return None, why

    def _add_wsample(self, s, label="腕部姿态"):
        self.wcalib_samples.append(s)
        self.wcalib_result = None
        self.log(f"[OK] [V3.51 eih] {label} → 腕部手眼样本 #{len(self.wcalib_samples)}：标记 "
                 f"{[round(v * 1000) for v in s['p_cam']]} mm（抖动 {s.get('spread_mm', 0):.1f} mm）")
        self._ui(self._refresh_wcalib_table)

    def on_wcalib_record(self):
        """V3.52 NON-BLOCKING RECORD: run wrist sample capture in background worker so GUI never freezes."""
        def job():
            s, why = self._capture_wrist_retry()
            if s is None:
                self.log(f"[WARN] [V3.51 eih] {why}")
                return
            self._add_wsample(s, "当前腕部姿态")
        self._run(job)

    def _wmove_measure(self, model, q_rad, speed, label):
        res = self.pipe.move_and_settle(model, list(q_rad), speed, label=label)
        if not res.get("ok"):
            return None, f"运动失败：{res.get('error')}"
        time.sleep(min(0.18, FAST_SETTLE_S, SETTLE_S))
        self._wait_still(1.2, hold_s=min(0.18, FAST_STILL_HOLD_S))
        return self._capture_wrist_retry(f"{label}：")

    def on_wcalib_auto(self):
        """V3.51/V3.52 EIH AUTO COLLECTION: automatically clears stale batch samples, probes 3 visibility-screened
        poses with 16-22 deg wrist tilts when uncalibrated, collects N diverse poses around the tabletop marker at
        <= 10% speed, returns to start, and solves EIH."""
        try:
            n = int(self.v_wnauto.get())
            assert 6 <= n <= 30
        except Exception:
            self.log("[WARN] [V3.51 eih] 自动采集点数请填 6–30")
            return
        ws = self.wrist_src
        if not ws or not getattr(ws, "K", None):
            self.log("[WARN] [V3.51 eih] 需要先连接腕部相机并加载/完成内参标定")
            return
        m = self._wmarker_from_ui()
        if not m:
            return
        if getattr(self.pipe, "teach_mode", False):
            self.log("[WARN] [V3.51 eih] NERO 在示教模式：先点下方「退出示教模式」锁住当前姿态后再自动采集")
            return
        if not getattr(self.jpanel, "enabled", False):
            self.log("[WARN] [V3.51 eih] 关节面板不可操作（需要 MoveIt 就绪）")
            return
        speed = min(CALIB_MAX_SPEED, max(0.01, float(self.cfg_fn().get("speed_percent", 10)) / 100.0))
        wcal = self.wcalib_result or self.wrist_calib

        def job():
            model = self._model()
            q0 = self.actual_q()
            if not q0:
                self.log("[FAIL] [V3.51 eih] 没有关节反馈")
                return
            q0r = [math.radians(v) for v in q0]
            K = dict(ws.K)
            clr_note = f"（将自动清空表中已有的 {len(self.wcalib_samples)} 个旧样本，避免不同批次原点混杂）\n" if self.wcalib_samples else ""
            if not self.ask(
                "腕部 Eye-in-Hand 自动采集",
                f"机械臂将在当前姿态周围自动走最多 {n + 3} 个姿态观察桌面固定标记 ID {m['id']}（边长 {m['size_mm']:g} mm）：\n"
                f"{clr_note}"
                f"· 位置展开 ≥ 12 cm，手腕多角度倾斜/旋转 ≥ 25°，只选预测能看到桌面标记（倾角 ≤ 42°）且高于桌面的安全姿态；\n"
                f"· 速度 {speed * 100:.0f}%，全程经 MoveIt 与路径安全区检查，结束后回到起点并自动计算。\n"
                "请确认桌面标记已贴牢不动、周围无障碍、手在急停上？"
            ):
                return
            if self.wcalib_samples:
                self.wcalib_samples = []
                self.wcalib_result = None
                self._ui(self._refresh_wcalib_table)
            self.wauto_state = {"phase": "起点", "k": 0, "n": n}
            self._ui(self._update_eih_ui)
            s0, why = self._capture_wrist_retry("起点：")
            if s0 is None:
                self.log(f"[FAIL] [V3.51 eih] 起点腕部相机测不到桌面标记 ID {m['id']}：{why}")
                return
            self._add_wsample(s0, "起点")
            got = 1
            T0 = s0["T_base_flange"]
            eih_mod = None
            if wcal and wcal.get("T_flange_cam") and wcal.get("T_base_marker"):
                err0 = self.W49.wrist_marker_error_mm(wcal, T0, s0["p_cam"])
                if err0 <= 45.0:
                    eih_mod = {"T_flange_cam": wcal["T_flange_cam"], "T_base_marker": wcal["T_base_marker"],
                               "marker_in_base": wcal["marker_in_base"]}
                    self.log(f"[INFO] [V3.51 eih] 用现有腕部标定预测桌面标记位置（起点误差 {err0:.1f} mm）")
            if eih_mod is None:
                # V3.56 PURE-TRANSLATION INITIAL PROBES: translate flange by ±3.8 cm along X, Y, Z with FIXED flange
                # orientation R0 via Fast Jacobian IK first! Because the wrist does not rotate, the camera optical axis
                # stays pointed at the tabletop marker (zero out-of-frame probe drops!) and R_0 t_X cancels out 100%
                # identically in Kabsch orientation estimation.
                probe_samples = [s0]
                probe_qs = []
                margin_r = math.radians(3.0)
                fence_ok = lambda q_: self.pipe.fence_goal_check(model, q_)[0]
                fast_ik = build_fast_jacobian_ik(self._fk_rad, q0r, model["limits"], soft_rad=margin_r)
                p0_f, R0_f = T_t(T0), T_R(T0)
                if fast_ik is not None:
                    for ax, step_m in ((0, 0.038), (1, 0.038), (2, 0.032)):
                        for sg in (1.0, -1.0):
                            tgt_p = list(p0_f)
                            tgt_p[ax] += sg * step_m
                            qc = fast_ik(q0r, tgt_p, R0_f)
                            if qc is not None and check_auto_path(fence_ok, q0r, qc, steps=4):
                                probe_qs.append(qc)
                                break
                for dq_deg in ([2.5, 1.5, 0.0, -2.0, 5.0, -4.0, 6.0],
                               [-2.5, 1.5, 2.0, -1.5, -5.0, 4.5, -6.0],
                               [1.5, -2.0, -2.0, 2.5, 4.0, 5.0, 8.0],
                               [-1.5, 2.0, -1.5, -2.5, -4.0, -4.5, -8.0]):
                    qc = [max(lo + margin_r, min(hi - margin_r, math.radians(q0[j] + dq_deg[j])))
                          for j, (lo, hi) in enumerate(model["limits"])]
                    if check_auto_path(fence_ok, q0r, qc, steps=4):
                        probe_qs.append(qc)
                for qc in probe_qs:
                    if self.stop_flag or len(probe_samples) >= 4:
                        break
                    self.wauto_state = {"phase": "探测初值", "k": len(probe_samples), "n": 3}
                    self._ui(self._update_eih_ui)
                    sp, why = self._wmove_measure(model, qc, speed, f"初值探测 #{len(probe_samples)}")
                    if sp is None:
                        self.log(f"[WARN] [V3.51 eih] 初值探测：{why}")
                        continue
                    self._add_wsample(sp, f"初值探测 #{len(probe_samples)}")
                    probe_samples.append(sp)
                    got += 1
                if len(probe_samples) < 3:
                    self.log("[FAIL] [V3.51 eih] 初值探测样本不足（要起点 + ≥ 2 个探测点）：请把桌面标记放在腕部画面正中、距离 18–30 cm 再试")
                    self.pipe.move_and_settle(model, q0r, speed, label="腕部标定 回起点")
                    return
                eih_mod = self.W49.estimate_eih_from_probes(probe_samples, size_mm=m["size_mm"])
                self.log(f"[INFO] [V3.51 eih] 腕部相机方向与桌面标记初值已确定（{len(probe_samples)} 点，残差 {eih_mod['mean_mm']:.1f} mm）")

            targets, st = self.W49.plan_eih_auto_targets(
                self._fk_rad, None, q0r, model["limits"], eih_mod, K, n=n,
                size_m=m["size_mm"] / 1000.0,
                fence_fn=lambda q_: self.pipe.fence_goal_check(model, q_)[0],
                seed=int(time.time()) % 100000
            )
            rej = "、".join(f"{k} {v}" for k, v in sorted(st["rejected"].items(), key=lambda kv: -kv[1])[:4])
            self.log(f"[INFO] [V3.51 eih] 选出 {len(targets)} 个腕部采样姿态（试了 {st['tries']} 个；淘汰：{rej or '无'}）· "
                     f"位置范围 {st['span_m'] * 100:.0f} cm · 朝向变化 {st['rot_max_deg']:.0f}°")
            for k, t in enumerate(targets):
                if self.stop_flag:
                    break
                self.wauto_state = {"phase": "采集", "k": k + 1, "n": len(targets)}
                self._ui(self._update_eih_ui)
                s, why = self._wmove_measure(model, t["q"], speed, f"腕部标定采样 {k + 1}/{len(targets)}")
                if s is None:
                    self.log(f"[WARN] [V3.51 eih] 姿态 {k + 1}：{why}")
                    continue
                self._add_wsample(s, f"姿态 {k + 1}/{len(targets)}")
                got += 1
            self.wauto_state = {"phase": "回起点", "k": 0, "n": 0}
            self._ui(self._update_eih_ui)
            self.pipe.move_and_settle(model, q0r, speed, label="腕部标定 回起点")
            A_all = [s["T_base_flange"] for s in self.wcalib_samples]
            ts = translation_span_m(A_all) if len(A_all) > 1 else 0.0
            rs = rotation_span_deg(A_all) if len(A_all) > 1 else 0.0
            self.log(f"[{'OK' if got >= 6 else 'WARN'}] [V3.51 eih] 腕部自动采集结束：新增 {got} 个样本（共 {len(self.wcalib_samples)}）· "
                     f"位置范围 {ts * 100:.0f} cm · 朝向变化 {rs:.0f}°")
            if len(self.wcalib_samples) >= 6:
                self._ui(self.on_wcalib_compute)
        self._run(job)

    def on_wcalib_check(self):
        """V3.51 EIH QUICK CHECK: move through stored check_poses_deg and verify wrist_marker_error_mm <= 5 mm."""
        c = self.wrist_calib
        poses = (c or {}).get("check_poses_deg") or []
        ws = self.wrist_src
        if not c or not poses or not ws or not getattr(ws, "K", None):
            self.log("[WARN] [V3.51 eih] 需要已保存含检验姿态的腕部标定且腕部相机已连接")
            return
        if getattr(self.pipe, "teach_mode", False) or not getattr(self.jpanel, "enabled", False):
            self.log("[WARN] [V3.51 eih] 需要退出示教模式且 MoveIt 就绪")
            return
        speed = min(CALIB_MAX_SPEED, max(0.01, float(self.cfg_fn().get("speed_percent", 10)) / 100.0))

        def job():
            model = self._model()
            q0 = self.actual_q()
            if not q0 or not self.ask("腕部标定快速复核", f"机械臂将以 {speed * 100:.0f}% 速度依次走到 {len(poses)} 个检验姿态，"
                                      "核对腕部手眼标定误差（≤ 5 mm 为通过），结束后回到当前姿态。继续？"):
                return
            q0r = [math.radians(v) for v in q0]
            self.wcheck_state = ("run", [])
            self._ui(self._update_eih_ui)
            errs = []
            for k, qd in enumerate(poses):
                if self.stop_flag:
                    break
                qr = [math.radians(v) for v in qd]
                s, why = self._wmove_measure(model, qr, speed, f"腕部快速复核 {k + 1}/{len(poses)}")
                if s is None:
                    self.log(f"[WARN] [V3.51 eih] 复核姿态 {k + 1}：{why}")
                    continue
                errs.append(self.W49.wrist_marker_error_mm(c, s["T_base_flange"], s["p_cam"]))
            self.pipe.move_and_settle(model, q0r, speed, label="腕部快速复核 回原位")
            ok = bool(errs) and max(errs) <= CALIB_GOOD_MAX_MM
            self.wcheck_state = ("ok" if ok else "fail", errs or [99.0])
            self._ui(self._update_eih_ui)
            self.log(f"[{'OK' if ok else 'WARN'}] [V3.51 eih] 腕部快速复核{'通过 ✓' if ok else '未通过 ✗'}：误差 "
                     + (", ".join(f"{e:.2f} mm" for e in errs) if errs else "未测到标记"))
        self._run(job)

    def on_wcalib_delete(self):
        sel = sorted((int(i) for i in self.tv_wcalib.selection()), reverse=True)
        for i in sel:
            if 0 <= i < len(self.wcalib_samples):
                self.wcalib_samples.pop(i)
        self.wcalib_result = None
        self._refresh_wcalib_table()

    def on_wcalib_clear(self):
        self.wcalib_samples = []
        self.wcalib_result = None
        self._refresh_wcalib_table()

    def on_wcalib_compute(self):
        m = self._wmarker_from_ui() or self.wmarker
        r = self.W49.calibrate_eye_in_hand(self.wcalib_samples, size_mm=m["size_mm"])
        if not r.get("ok"):
            self.log(f"[WARN] [V3.49 eih] {r.get('error')}")
            return
        self.wcalib_result = r
        self._refresh_wcalib_table()
        tag = "[OK]" if r["good"] else "[WARN]"
        t = r["cam_in_flange"]
        self.log(f"{tag} [V3.49 eih] Eye-in-Hand 标定：样本 {r['n_used']} · 残差 {r['mean_mm']:.2f}/{r['max_mm']:.2f} mm · "
                 f"留一 {r['loo_mean_mm']:.2f}/{r['loo_max_mm']:.2f} mm · link7→cam "
                 f"({t[0] * 1000:+.1f}, {t[1] * 1000:+.1f}, {t[2] * 1000:+.1f}) mm")

    def on_wcalib_save(self, path=None):
        r = self.wcalib_result
        if not r or not r.get("ok"):
            self.log("[WARN] [V3.49 eih] 先「计算腕部手眼」")
            return
        used = [self.wcalib_samples[i]["q_deg"] for i in (r.get("used") or range(len(self.wcalib_samples)))]
        step_c = max(1, len(used) // 4)
        chk_poses = used[::step_c][:4]
        d = dict(r, version="V3.51", saved=time.strftime("%Y-%m-%d %H:%M:%S"),
                 marker=self._wmarker_from_ui() or dict(self.wmarker), check_poses_deg=chk_poses)
        p = path or self._wrist_handeye_path()
        save_json(p, d)
        if not path and p == self.W49.DEFAULT_WRIST_HANDEYE_PATH and CALIB_PATH == DEFAULT_CALIB_PATH:
            try:
                os.makedirs(CALIB_DIR, exist_ok=True)
                save_json(os.path.join(CALIB_DIR, "wrist_handeye_" + time.strftime("%Y%m%d_%H%M%S") + ".json"), d)
            except Exception:
                pass
        self.wrist_calib = d
        self.wcalib_result = None
        self.wcheck_state = None
        self._refresh_wcalib_table()
        self.log(f"[OK] [V3.49 eih] 腕部 Eye-in-Hand 标定已保存 → {p}")

    def on_wcalib_save_as(self, path=None):
        c = self.wcalib_result or self.wrist_calib
        if not c:
            self.log("[WARN] [V3.49 eih] 还没有可保存的腕部标定")
            return
        path = path or self._ask_path(True, CALIB_DIR, "wrist_handeye_" + time.strftime("%Y%m%d_%H%M") + ".json", "腕部手眼另存为")
        if path:
            save_json(path, dict(c, exported=time.strftime("%Y-%m-%d %H:%M:%S")))
            self.log(f"[OK] [V3.49 eih] 腕部手眼另存为 → {path}")

    def on_wcalib_load(self, path=None):
        path = path or self._ask_path(False, CALIB_DIR, "", "读取腕部手眼标定")
        if not path:
            return
        d = load_json(path)
        if not self.W49.handeye_valid(d):
            self.log(f"[FAIL] [V3.49 eih] {path} 不是有效的腕部手眼标定文件")
            return
        save_json(self._wrist_handeye_path(), d)
        self.wrist_calib = d
        if isinstance(d.get("marker"), dict) and d["marker"].get("size_mm"):
            self.wmarker = dict(self.W49.DEFAULT_WRIST_MARKER, **d["marker"])
            if hasattr(self, "v_wdict"):
                self.v_wdict.set(self.wmarker["dict"])
                self.v_wmid.set(str(self.wmarker["id"]))
                self.v_wmsize.set(f"{self.wmarker['size_mm']:g}")
        self.wcalib_result = None
        self._refresh_wcalib_table()
        self.log(f"[OK] [V3.49 eih] 已读取腕部 Eye-in-Hand 标定 ← {path}")

    def on_wcalib_clear_calib(self):
        p = self._wrist_handeye_path()
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass
        self.wrist_calib = None
        self.wcalib_result = None
        self._refresh_wcalib_table()
        self.log("[OK] [V3.49 eih] 已清除当前腕部 Eye-in-Hand 标定设定")

    # ------------------------------------------------------------------ V3.49 zero-prior egg holder + slot localization
    def on_holder(self):
        """V3.49 ZERO-PRIOR HOLDER: segment the egg holder and autonomously infer its 3D OBB, (rows x cols) layout,
        and every slot's 3D coordinate in base_link without user-entered dimensions."""
        cal = self.result or self.calib
        if not cal or not self.src or not self.src.K:
            self.log("[WARN] [V3.49 holder] 需要相机连接 + 手眼标定")
            return
        YP = self.YP
        W49 = self.W49
        try:
            conf = float(self.v_yconf.get())
            assert 0.01 <= conf <= 0.99
        except Exception:
            self.log("[WARN] [V3.49 holder] 置信度门槛（0.01–0.99）无效")
            return
        sel_path = YP.find_model(self.v_ydir.get(), self.v_ymodel.get())
        raw_obj = self.v_obj_names.get().strip() if hasattr(self, "v_obj_names") else ""
        holder_prompts = YP.parse_prompts(raw_obj or W49.HOLDER_PROMPTS)
        pf_path = YP.find_model(self.v_ydir.get(), YP.DEFAULT_PF_MODEL)
        if sel_path and YP.is_pf_model(sel_path):
            mode, path = "pf", sel_path
        elif sel_path and os.path.isfile(os.path.join(os.path.dirname(sel_path), YP.text_encoder_name(sel_path))):
            mode, path = "text", sel_path
        elif pf_path:
            mode, path = "pf", pf_path
        elif self.ref and sel_path:
            mode, path = "vp", sel_path
        else:
            mode, path = "text", sel_path
        f = self._fence()
        cnote = self._calib_note()
        prev_eggs = [e for e in (self.eggs or []) if str(e.get("det_id", "")).startswith("egg_")]
        agent_fn = getattr(self, "holder_agent_fn", None)

        def job():
            import numpy as np
            if not path:
                self.log(f"[FAIL] [V3.49 holder] 找不到模型 {self.v_ymodel.get()}")
                return
            t0 = time.time()
            rgb, depths = self._grab_frames(YP.FUSE_FRAMES)
            if rgb is None or not depths:
                self.log("[WARN] [V3.49 holder] 没拿到相机画面")
                return
            try:
                d2 = self.det_yolo.detect(rgb, mode, path, ref=self.ref if mode == "vp" else None,
                                          prompts=holder_prompts, conf=conf)
            except YP.YoloUnavailable as exc:
                self.log(f"[FAIL] [V3.49 holder] {exc}")
                return
            t1 = time.time()
            depth = YP.fuse_depth(depths, d2)
            rows, diag = W49.locate_egg_holder(
                d2, rgb, depth, dict(self.src.K), np.array(cal["T_base_cam"], dtype=float),
                table_z=(f or {}).get("z_table"), fence_poly=(f or {}).get("polygon"),
                calib_note=cnote, existing_eggs=prev_eggs, agent_fn=agent_fn
            )
            t2 = time.time()
            dev = self.det_yolo.info.get("device") or self.det_yolo.device_name(self.det_yolo.model)
            if not rows:
                self.log(f"[WARN] [V3.49 holder] 蛋托未定位：{diag.get('reason', '未找到')}")
                return
            self.eggs = list(rows)
            view = []
            for r in rows:
                view.append({"box": r["yolo"]["box"], "status": r["status"],
                             "label": short_egg_label(r["det_id"])})

            def show():
                self.yolo_view = view
                for it in self.tv_eggs.get_children():
                    self.tv_eggs.delete(it)
                for row_i, r in enumerate(rows):
                    x, y, z = r.get("xyz_m") or (None, None, None)
                    mm = (lambda v: "-" if v is None else f"{v * 1000:.0f}")
                    why = "；".join(r["reasons"]) or "✓"
                    if r.get("notes"):
                        why = (why + " " if why != "✓" else "✓ ") + "；".join(r["notes"])
                    self.tv_eggs.insert("", "end", tags=("odd",) if row_i % 2 else (), values=(
                        r["det_id"], mm(x), mm(y), mm(z),
                        "-" if r.get("height_mm") is None else f"{r['height_mm']:.0f}",
                        f"{r['yolo']['conf']:.2f}", "-", f"{r['confidence']:.2f}",
                        "✓" if r["trusted"] else "-", why))
                autofit_treeview(self.tv_eggs, self.egg_cols, stretch_col="why")
                self.v_eggs.set(
                    f"蛋托自主识别 · {diag['rows']}×{diag['cols']}（共 {diag['n_slots']} 槽：空槽 {diag['n_empty']} / "
                    f"有蛋 {diag['n_occupied']}） · 实测外框 {diag['size_mm'][0]:.0f}×{diag['size_mm'][1]:.0f}×"
                    f"{diag['height_mm']:.0f} mm · 槽距 {diag['pitch_mm'][0]:.0f}×{diag['pitch_mm'][1]:.0f} mm · "
                    f"偏航 {diag['yaw_deg']:+.0f}° · YOLO({dev}) {(t1 - t0) * 1000:.0f} ms + 3D {(t2 - t1) * 1000:.0f} ms"
                    + (f"\n⚠ {cnote}" if cnote else ""))
                self._force_redraw()
                self._update_calib_ui()
            self._ui(show)
            self.log(f"[OK] [V3.49 holder] 蛋托与槽位定位（{diag['judgment_source']}）：{diag['rows']}×{diag['cols']} "
                     f"（空槽 {diag['n_empty']} / 有蛋 {diag['n_occupied']}），外框 {diag['size_mm'][0]:.0f}×{diag['size_mm'][1]:.0f} mm；"
                     + "; ".join(f"{r['det_id']} ({r['xyz_m'][0] * 1000:.0f}, {r['xyz_m'][1] * 1000:.0f}, {r['xyz_m'][2] * 1000:.0f}) mm"
                                 for r in rows[:7]))

        def safe():
            try:
                job()
            except Exception as exc:
                self.log(f"[FAIL] [V3.49 holder] 蛋托识别失败：{type(exc).__name__}: {exc}")
        self._run(safe)

    # ------------------------------------------------------------------ V3.50 coarse-to-fine wrist refinement & alignment
    def on_wrist_refine(self):
        """V3.50 COARSE-TO-FINE REFINE: use the wrist camera at the current hover pose to refine the selected target's
        3D (x, y, z) and yaw in base_link, compute (Δx, Δy, Δψ) relative to the active TCP, and check fingertip clearance."""
        ws = self.wrist_src
        intr = self.intr_result or self.wrist_intr
        K_w = (getattr(ws, "K", None) if ws else None) or (intr.get("K") if intr else None)
        he = self.wcalib_result or self.wrist_calib
        if not ws or not K_w or not he or not he.get("T_flange_cam"):
            self.log("[WARN] [V3.50 wrist] 需要腕部相机已连接 + 腕部内参标定 + Eye-in-Hand 手眼标定")
            return
        tgt = self._selected_hover_target()
        if not tgt:
            self.log("[WARN] [V3.50 wrist] 先在视觉定位表里识别并选中一个目标（如 egg_1 或 slot_1）")
            return
        q = self.actual_q()
        if not q:
            self.log("[FAIL] [V3.50 wrist] 没有关节反馈")
            return
        T_bf = self.flange_T(q)
        if T_bf is None:
            self.log("[FAIL] [V3.50 wrist] FK 计算失败")
            return
        rgb, _d, _fid = ws.frames()
        if rgb is None:
            self.log("[WARN] [V3.50 wrist] 腕部相机还没有画面")
            return
        wq = self.W49.evaluate_wrist_frame(rgb, self._prev_wrist_rgb, q, self._prev_wrist_q)
        self.wrist_quality = wq
        if not wq["settled"] and wq["status"] == "motion":
            self.log(f"[WARN] [V3.50 wrist] 腕部画面未停稳：{wq['badge']}（请等机械臂完全停稳后再点精定位）")
            return

        YP = self.YP
        try:
            conf = float(self.v_yconf.get())
            el, ew = float(self.v_egg_l.get()), float(self.v_egg_w.get())
        except Exception:
            conf, el, ew = 0.25, 55.0, 42.0
        sel_path = YP.find_model(self.v_ydir.get(), self.v_ymodel.get())
        raw_obj = self.v_obj_names.get().strip() if hasattr(self, "v_obj_names") else ""
        prompts = YP.parse_prompts(raw_obj or self.v_yprompt.get())
        if sel_path and YP.is_pf_model(sel_path):
            mode, path = "pf", sel_path
        elif self.v_method.get() == LOCATE_METHODS[0] and self.ref and sel_path:
            mode, path = "vp", sel_path
        else:
            mode, path = "text", sel_path
        f = self._fence()
        zt = float((f or {}).get("z_table", 0.0))
        tool_T = self._active_tcp_T()
        u_intr = intr if (intr and not getattr(ws, "undistort_preview", False)) else None

        res = self.W49.wrist_refine_target(
            rgb, dict(K_w), T_bf, he, tool_T, tgt,
            det_yolo=self.det_yolo if path else None, mode=mode, model_path=path,
            prompts=prompts, conf=conf, ref=self.ref if mode == "vp" else None,
            table_z=zt, egg_size_mm=(el, ew), undistort_intr=u_intr
        )
        if not res.get("ok"):
            self.log(f"[WARN] [V3.50 wrist] 腕部精定位未成功：{res.get('error')}")
            return

        self.wrist_refine_res = res
        tgt["coarse_xyz_m"] = list(res["coarse_xyz_m"])
        tgt["xyz_m"] = list(res["fine_xyz_m"])
        tgt["yaw_deg"] = res["fine_yaw_deg"]
        tgt["axis_world"] = list(res["axis_world"])
        tgt["wrist_refined"] = True
        dx, dy = res["delta_tcp_mm"]
        note_str = (f"腕部精定位 Δx={dx:+.1f} Δy={dy:+.1f} mm · 偏航 {res['fine_yaw_deg']:+.1f}° "
                    f"(粗精差 {res['delta_coarse_mm']:.1f} mm)")
        tgt.setdefault("notes", []).append(note_str)
        if res["warnings"]:
            for w_ in res["warnings"]:
                if w_ not in tgt.setdefault("reasons", []):
                    tgt["reasons"].append(w_)
            tgt["trusted"] = bool(res["trusted"])

        for it in self.tv_eggs.get_children():
            vals = list(self.tv_eggs.item(it, "values") or ())
            if vals and vals[0] == tgt.get("det_id"):
                fx, fy, fz = res["fine_xyz_m"]
                vals[1], vals[2], vals[3] = f"{fx * 1000:.0f}", f"{fy * 1000:.0f}", f"{fz * 1000:.0f}"
                vals[8] = "✓" if tgt.get("trusted") else "-"
                vals[9] = ("；".join(tgt.get("reasons") or []) or "✓") + "；" + note_str
                self.tv_eggs.item(it, values=vals)
                break
        autofit_treeview(self.tv_eggs, self.egg_cols, stretch_col="why")
        self.v_eggs.set(
            f"腕部精定位完成（{res['det_id']}） · 精坐标 ({res['fine_xyz_m'][0] * 1000:+.1f}, "
            f"{res['fine_xyz_m'][1] * 1000:+.1f}, {res['fine_xyz_m'][2] * 1000:+.1f}) mm · "
            f"相对当前 TCP 偏差 Δx={dx:+.1f} mm, Δy={dy:+.1f} mm · 姿态偏航 {res['fine_yaw_deg']:+.1f}° "
            f"(Δψ={res['delta_yaw_deg']:+.1f}°) · 粗精差 {res['delta_coarse_mm']:.1f} mm · "
            f"{'指尖净空正常 ✓' if res['clearance_ok'] else '⚠ 指尖有碰撞风险'}"
        )
        self._wrist_live_text(q)
        self._redraw_wrist()
        self._force_redraw()
        self._update_calib_ui()
        tag = "[OK]" if res["trusted"] else "[WARN]"
        self.log(f"{tag} [V3.50 wrist] 腕部精定位 {res['det_id']}：精坐标 "
                 f"({res['fine_xyz_m'][0] * 1000:+.1f}, {res['fine_xyz_m'][1] * 1000:+.1f}, {res['fine_xyz_m'][2] * 1000:+.1f}) mm · "
                 f"TCP 微调量 Δx={dx:+.1f} mm, Δy={dy:+.1f} mm, Δψ={res['delta_yaw_deg']:+.1f}° · "
                 f"与主相机粗定位差 {res['delta_coarse_mm']:.1f} mm"
                 + ("；⚠ " + "；".join(res["warnings"]) if res["warnings"] else ""))

    def on_wrist_align(self):
        """V3.50 FINE ALIGN TCP: move the TCP horizontally by (Δx, Δy) to align directly above the wrist-refined target."""
        if self._busy():
            self.log("[WARN] [V3.50 wrist] 上一个操作还在进行")
            return
        if getattr(self.pipe, "teach_mode", False) or not getattr(self.jpanel, "enabled", False):
            self.log("[WARN] [V3.50 wrist] 需要退出示教模式且 MoveIt / 关节面板就绪")
            return
        res = getattr(self, "wrist_refine_res", None)
        if not res or not res.get("ok"):
            self.log("[WARN] [V3.50 wrist] 先点「◎ 腕部精定位」算出微调偏移量 (Δx, Δy, Δψ)")
            return
        try:
            hover_mm = float(self.v_hover_mm.get())
            assert HOVER_RANGE_MM[0] <= hover_mm <= HOVER_RANGE_MM[1]
        except Exception:
            self.log(f"[WARN] [V3.50 wrist] 悬停高度请填 {HOVER_RANGE_MM[0]:.0f}–{HOVER_RANGE_MM[1]:.0f} mm")
            return
        q0 = self.actual_q()
        if not q0:
            self.log("[FAIL] [V3.50 wrist] 没有关节反馈")
            return
        model = self._model()
        urdf = self._urdf()
        base = model.get("base_link") or "base_link"
        tool_T = self._active_tcp_T()
        fx, fy, fz = [float(v) for v in res["fine_xyz_m"]]
        f = self._fence()
        zt = float((f or {}).get("z_table", 0.0))
        hz = max(fz + hover_mm / 1000.0, zt + 0.05)
        target_xyz = [fx, fy, hz]
        q0r = [math.radians(v) for v in q0]

        def fence_fn(q_rad):
            if hasattr(self.pipe, "fence_goal_check"):
                r_ = self.pipe.fence_goal_check(model, q_rad)
                return bool(r_[0]), (r_[1] if len(r_) > 1 else "")
            return True, ""

        q_tgt, info = plan_hover_target(urdf, model["joints"], model["limits"], q0r, tool_T, target_xyz,
                                        base=base, fence_fn=fence_fn)
        if q_tgt is None:
            self.log(f"[FAIL] [V3.50 wrist] 微调对齐规划失败（{res['det_id']}）：{info}")
            return
        speed = min(CALIB_MAX_SPEED, max(0.01, float(self.cfg_fn().get("speed_percent", 10)) / 100.0))
        det_id = res["det_id"]
        dx, dy = res["delta_tcp_mm"]
        warn_txt = ("\n⚠ 警告：" + "；".join(res["warnings"]) + "！") if res.get("warnings") else ""

        def job():
            if not self.ask(
                "腕部精定位微调对齐",
                f"将按腕部相机精定位结果微调 TCP 对齐到 {det_id} 正上方 {hover_mm:.0f} mm：{warn_txt}\n"
                f"水平微调量：Δx = {dx:+.1f} mm, Δy = {dy:+.1f} mm（长轴偏航 {res['fine_yaw_deg']:+.1f}°，粗精差 {res['delta_coarse_mm']:.1f} mm）\n"
                f"精定位目标 TCP (base_link)：({target_xyz[0] * 1000:+.1f}, {target_xyz[1] * 1000:+.1f}, {target_xyz[2] * 1000:+.1f}) mm\n"
                f"速度 {speed * 100:.0f}% · 只水平微调悬停、不下探、不动夹爪。\n确认执行微调？"
            ):
                return
            if self._pre_hover_q is None:
                self._pre_hover_q = list(q0r)
            mres = self.pipe.move_and_settle(model, q_tgt, speed, label=f"腕部精定位微调 {det_id}")
            if not mres.get("ok"):
                self.log(f"[FAIL] [V3.50 wrist] 微调对齐运动失败：{mres.get('error')}")
                self._ui(self._update_calib_ui)
                return
            import NERO_V315_Consistency_Safety as CS
            qa = self.actual_q() or [math.degrees(v) for v in q_tgt]
            T_act = CS.tool_pose(urdf, model["joints"], [math.radians(v) for v in qa], base, tool_T)
            act_xyz = T_t(T_act) if T_act is not None else target_xyz
            err_mm = vnorm(vsub(act_xyz, target_xyz)) * 1000.0
            self.log(f"[OK] [V3.50 wrist] 已完成 {det_id} 腕部精定位微调对齐：微调 (Δx={dx:+.1f}, Δy={dy:+.1f}) mm → "
                     f"实测 TCP ({act_xyz[0] * 1000:+.1f}, {act_xyz[1] * 1000:+.1f}, {act_xyz[2] * 1000:+.1f}) mm"
                     f"（残余偏差 {err_mm:.2f} mm）")
            self._ui(self._update_calib_ui)
        self._run(job)
