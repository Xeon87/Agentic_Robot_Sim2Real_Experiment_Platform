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
DEPTH_SOURCE_MIN_FRAC = 0.8              # a pose uses depth only if >= 80 % of its frames have consistent depth
# V3.45 VISIBILITY-AWARE AUTO SAMPLING: flange positions in a box around the start (>= 20 cm span), wrist
# orientation within 25 deg of the start, marker predicted inside the image, facing the camera, at a usable range.
AUTO_POS_HALF_M = (0.12, 0.12, 0.08)
AUTO_ROT_MIN_DEG, AUTO_ROT_MAX_DEG = 8.0, 25.0
AUTO_VIEW_MAX_DEG, AUTO_IMG_MARGIN, AUTO_RANGE_M = 45.0, 0.08, (0.35, 1.30)
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


def calibrate(samples, reject_mm=8.0, min_samples=6):
    """samples: [{"T_base_flange": 4x4, "p_cam": [3], ...}] -> result dict (see solve_eye_to_hand) with
    used / rejected indices, leave-one-out errors, spans and a verdict."""
    idx = list(range(len(samples)))
    if len(idx) < min_samples:
        return {"ok": False, "error": f"至少需要 {min_samples} 个样本（现在 {len(idx)} 个）"}
    rejected = []
    y0 = None
    while True:
        A = [samples[i]["T_base_flange"] for i in idx]
        P = [samples[i]["p_cam"] for i in idx]
        r = solve_eye_to_hand(A, P, y0=y0)
        y0 = r["marker_in_flange"]
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
        P2 = [samples[i]["p_cam"] for m, i in enumerate(idx) if m != j]
        rj = solve_eye_to_hand(A2, P2, y0=y0, iters=1500)
        i = idx[j]
        loo.append(vnorm(vsub(T_apply(rj["T_base_cam"], samples[i]["p_cam"]),
                              T_apply(samples[i]["T_base_flange"], rj["marker_in_flange"]))) * 1000.0)
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
        if not (margin * W <= uv[0] <= (1 - margin) * W and margin * H <= uv[1] <= (1 - margin) * H):
            return False, "出画面", info
        if nc is not None:
            cosv = -sum(nc[i] * pc[i] for i in range(3)) / (r * (vnorm(nc) or 1.0))
            info["view_deg"] = math.degrees(math.acos(max(-1.0, min(1.0, cosv))))
            if info["view_deg"] > view_max_deg:
                return False, "标记转得太偏", info
        return True, "", info


def plan_auto_targets(fk_fn, ik_fn, q0_rad, limits_rad, model, K, n, fence_fn=None, seed=0,
                      pos_half=AUTO_POS_HALF_M, rot_deg=(AUTO_ROT_MIN_DEG, AUTO_ROT_MAX_DEG), soft_deg=3.0,
                      max_tries=1500, step_max_deg=AUTO_JOINT_STEP_MAX_DEG):
    """V3.45: n joint targets (rad) whose predicted marker is visible / facing the camera, flange positions spread
    over a box around the start (farthest-point choice), orientation within rot_deg of the start.
    fk_fn(q_rad) -> T_base_flange; ik_fn(q_seed, xyz, R) -> q_rad or None.
    -> (targets [{"q", "T", "view_deg", "uv"}], stats)"""
    rnd = random.Random(seed)
    T0 = fk_fn(q0_rad)
    p0, R0 = T_t(T0), T_R(T0)
    soft = math.radians(soft_deg)
    pool, why = [], {}
    tries = 0
    while tries < max_tries and len(pool) < 4 * n:
        tries += 1
        pos = [p0[i] + rnd.uniform(-pos_half[i], pos_half[i]) for i in range(3)]
        ax = [rnd.gauss(0, 1) for _ in range(3)]
        R = mat_mul(axis_angle_R(ax, rnd.uniform(*rot_deg)), R0)
        Tt = T_from(R, pos)
        ok, reason, info = model.visible(Tt, K)
        if not ok:
            why[reason] = why.get(reason, 0) + 1
            continue
        q = ik_fn(q0_rad, pos, R)
        if q is None:
            why["IK 无解"] = why.get("IK 无解", 0) + 1
            continue
        if any(not (lo + soft <= v <= hi - soft) for v, (lo, hi) in zip(q, limits_rad)):
            why["接近限位"] = why.get("接近限位", 0) + 1
            continue
        if max(abs(math.degrees(a - b)) for a, b in zip(q, q0_rad)) > step_max_deg:
            why["关节变化太大"] = why.get("关节变化太大", 0) + 1
            continue
        if fence_fn is not None and not fence_fn(q):
            why["安全区"] = why.get("安全区", 0) + 1
            continue
        Tq = fk_fn(q)
        ok2, _r2, info2 = model.visible(Tq, K)
        if not ok2:
            why["IK 后不可见"] = why.get("IK 后不可见", 0) + 1
            continue
        pool.append({"q": q, "T": Tq, "view_deg": info2.get("view_deg"), "uv": info2.get("uv")})
    chosen, pts = [], [p0]
    while pool and len(chosen) < n:                  # farthest-point: spread the flange positions
        k = max(range(len(pool)), key=lambda i: min(vnorm(vsub(T_t(pool[i]["T"]), p)) for p in pts))
        c = pool.pop(k)
        chosen.append(c)
        pts.append(T_t(c["T"]))
    order, cur = [], list(q0_rad)                    # nearest-neighbour order in joint space
    while chosen:
        k = min(range(len(chosen)), key=lambda i: max(abs(a - b) for a, b in zip(chosen[i]["q"], cur)))
        c = chosen.pop(k)
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
    Pose by solvePnP (IPPE_SQUARE); the range along the ray is replaced by the measured depth at the centre when
    both agree within 25 mm (depth is more accurate than PnP range for a small marker)."""
    cv2 = _import_cv2()
    import numpy as np
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)
    d = _aruco_dict(cv2, dict_name)
    if hasattr(cv2.aruco, "ArucoDetector"):
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _rej = cv2.aruco.ArucoDetector(d, params).detectMarkers(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _rej = cv2.aruco.detectMarkers(gray, d, parameters=params)
    if ids is None:
        return None
    ids = [int(v) for v in np.asarray(ids).flatten()]
    if int(marker_id) not in ids:
        return None
    c = np.asarray(corners[ids.index(int(marker_id))], dtype=np.float64).reshape(4, 2)
    h = size_m / 2.0
    obj = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)
    Km = np.array([[K["fx"], 0.0, K["ppx"]], [0.0, K["fy"], K["ppy"]], [0.0, 0.0, 1.0]])
    dc = np.array(list(dist or [0.0] * 5)[:5] + [0.0] * max(0, 5 - len(dist or [])), dtype=np.float64)
    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(obj, c, Km, dc, flags=flag)
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
        r = max(2, int(out["side_px"] * 0.12))
        win = np.asarray(depth_m)[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
        vals = win[win > 0]
        if vals.size >= 5:
            z = float(np.median(vals))
            if abs(z - p[2]) < 0.025:
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
        self.depth_res = DEPTH_RES[0]            # V3.47.1: set by the page before start()
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

    def display_ppm(self, view="color"):
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


# =========================================================================== UI
class CameraTabUI:
    """V3.25 Camera调试 tab."""
    DW, DH = 640, 360          # displayed image (half resolution)

    def __init__(self, tk, ttk, parent, pipe, jpanel, safe_tab, cfg_fn, log, ask, visible_fn=lambda: True,
                 source_factory=None):
        self.tk, self.ttk, self.pipe, self.jpanel, self.safe_tab = tk, ttk, pipe, jpanel, safe_tab
        self.cfg_fn, self.log, self.ask, self.visible_fn = cfg_fn, log, ask, visible_fn
        self.source_factory = source_factory or (lambda: RealSenseSource(log=self.log))
        self.src = None
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
        self.stop_flag = False
        self.worker = None
        self.photo = None
        self._shown_fid = -1
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
        for t_, v_ in (("彩色", "color"), ("深度", "depth"), ("叠加", "both")):
            ttk.Radiobutton(top, text=t_, value=v_, variable=self.v_view, command=self._force_redraw).pack(side="left")
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
        self.cv = tk.Canvas(left, width=self.DW, height=self.DH, bg="#202020", highlightthickness=1,
                            highlightbackground="#999")
        self.cv.pack()
        self.cv.bind("<ButtonPress-1>", self.on_press)
        self.cv.bind("<B1-Motion>", self.on_drag)
        self.cv.bind("<ButtonRelease-1>", self.on_release)
        self.v_meas = tk.StringVar(value="点击画面 = 读距离 / 坐标；拖一个框 = 选深度检查区域")
        JC.fixed_label(tk, left, 2, width=self.DW, textvariable=self.v_meas, font=("TkFixedFont", 9),
                       wraplength=self.DW - 10, pack={"pady": 2})
        self.v_live = tk.StringVar(value="")
        self.lbl_live = JC.fixed_label(tk, left, 2, width=self.DW, textvariable=self.v_live, font=("TkDefaultFont", 9, "bold"),
                                       wraplength=self.DW - 10, pack={"pady": 2})
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)
        self._build_info(nb)
        self._build_depth(nb)
        self._build_calib(nb)
        self._build_locate(nb)

    def _build_info(self, nb):
        tk, ttk = self.tk, self.ttk
        p = ttk.Frame(nb, padding=6)
        nb.add(p, text="相机信息")
        self.v_info = tk.StringVar(value="（连接相机后显示型号、序列号、固件、USB、内参、帧率）")
        self.JC.fixed_label(tk, p, 7, textvariable=self.v_info, font=("TkFixedFont", 9), justify="left", anchor="nw",
                            wraplength=560, pack={"fill": "x"})
        r = ttk.Frame(p)
        r.pack(fill="x", pady=4)
        ttk.Button(r, text="保存相机信息", command=self.on_save_info).pack(side="left")
        ttk.Label(p, text="相机固定在支架上（和 NERO 同一个刚性底座）。换了相机 / 动了支架 → 重新做手眼标定。\n"
                          "realsense-viewer 和本页不能同时占用相机。", foreground="#555555", justify="left").pack(anchor="w")

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

    def _build_calib(self, nb):
        tk, ttk = self.tk, self.ttk
        p = ttk.Frame(nb, padding=6)
        nb.add(p, text="手眼标定")
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
        try:
            ls = int(p.tk.call("font", "metrics", "TkDefaultFont", "-linespace"))
        except Exception:
            ls = 16
        st = ttk.Style(p)
        st.configure("V345Calib.Treeview", rowheight=ls + 8)
        st.configure("V345Calib.Treeview.Heading", font=("TkDefaultFont", 9, "bold"))
        cols = (("k", "#", 34), ("src", "来源", 48), ("q", "关节角 J1…J7 (°)", 206), ("p", "标记 xyz (mm)", 132),
                ("jit", "抖动", 50), ("res", "残差", 54), ("loo", "留一", 54))
        self.tv = ttk.Treeview(tf, columns=[c_ for c_, _t, _w in cols], show="headings", height=7,
                               style="V345Calib.Treeview", selectmode="extended")
        for c_, t_, w_ in cols:
            self.tv.heading(c_, text=t_)
            self.tv.column(c_, width=w_, minwidth=w_, anchor="center", stretch=(c_ == "q"))
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
        self._update_calib_ui()

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
        r3b = ttk.Frame(p)                         # V3.47.1 DEPTH SETTINGS + settings files
        r3b.pack(fill="x", pady=(4, 0))
        ttk.Label(r3b, text="深度").pack(side="left")
        self.v_dres = tk.StringVar(value=DEPTH_RES[0])
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
        cols = (("id", "编号", 60), ("x", "x mm", 60), ("y", "y mm", 60), ("z", "z mm", 60), ("h", "实测高", 56),
                ("yolo", "YOLO", 52), ("iou", "轮廓重合", 64), ("conf", "置信度", 56), ("ok", "可信", 44),
                ("why", "说明", 260))
        self.tv_eggs = ttk.Treeview(p, columns=[c_[0] for c_ in cols], show="headings", height=6)
        for c_, t_, w_ in cols:
            self.tv_eggs.heading(c_, text=t_)
            self.tv_eggs.column(c_, width=w_, anchor="w" if c_ == "why" else "center", stretch=(c_ == "why"))
        self.tv_eggs.pack(fill="x", pady=4)
        self.v_eggs = tk.StringVar(value="需要手眼标定。坐标在 base_link（与安全区、关节控制一致）。只显示识别结果，不运动。")
        self.JC.fixed_label(tk, p, 5, textvariable=self.v_eggs, wraplength=600, justify="left", anchor="nw",
                            pack={"fill": "x"})
        self._locate_autoload()

    # ------------------------------------------------------------------ V3.47.1 视觉定位 settings files
    def _locate_settings(self):
        d = {"version": "V3.47.1", "saved": time.strftime("%Y-%m-%d %H:%M:%S"), "method": self.v_method.get(),
             "model_dir": self.v_ydir.get(), "model": self.v_ymodel.get(), "conf": self.v_yconf.get(),
             "prompts": self.v_yprompt.get(), "egg_mm": [self.v_egg_l.get(), self.v_egg_w.get()],
             "white_geometric": bool(self.v_white.get()), "save_samples": bool(self.v_ysave.get()),
             "depth_res": self.v_dres.get(), "depth_preset": self.v_dpreset.get()}
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
                         (self.v_yprompt, "prompts")):
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
        good = (loo_m is not None and loo_m <= CALIB_GOOD_MEAN_MM and loo_x <= CALIB_GOOD_MAX_MM)
        lines = [f"{src}  ·  {'合格 ✓' if good else '未达标 ⚠'}",
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
        return uv[0] / 2.0, uv[1] / 2.0

    def _redraw(self):
        c = self.cv
        data = self.src.display_ppm(self.v_view.get()) if self.src else None
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
            for e in self.eggs:
                uv = project(T_apply(Tcb, [v for v in e["xyz_m"]]), K)
                if uv:
                    x, y = self._px(uv)
                    col = "#00e676" if e.get("trusted") else "#ff9100"
                    c.create_oval(x - 7, y - 7, x + 7, y + 7, outline=col, width=2)
                    c.create_text(x + 9, y, text=e.get("det_id", ""), fill=col, anchor="w")
        if self.ov["eggs"].get():
            self._overlay(self._draw_yolo)
        if self.roi:
            u0, v0, u1, v1 = self.roi
            c.create_rectangle(u0 / 2, v0 / 2, u1 / 2, v1 / 2, outline="#ffea00", dash=(4, 3), width=2)
        if self.meas:
            x, y = self._px(self.meas)
            c.create_oval(x - 4, y - 4, x + 4, y + 4, outline="#ff1744", width=2)

    def _draw_yolo(self):
        # V3.47: YOLO boxes (green = egg, orange = not trusted, red = rejected) and the reference being edited
        c = self.cv
        col = {"egg": "#00e676", "untrusted": "#ff9100", "rejected": "#ff1744"}
        for v in getattr(self, "yolo_view", None) or []:
            x0, y0, x1, y1 = [t / 2.0 for t in v["box"]]
            k = col.get(v["status"], "#ff1744")
            c.create_rectangle(x0, y0, x1, y1, outline=k, width=2, **({"dash": (3, 2)} if v["status"] == "rejected" else {}))
            c.create_text(x0, y0 - 2, text=v["label"], fill=k, anchor="sw")
        e = getattr(self, "ref_edit", None)
        if e is not None:
            c.create_text(6, 6, text="框选参考中（定格画面的框）", fill="#ffea00", anchor="nw")
            for b, k in zip(e["boxes"], e["cls"]):
                x0, y0, x1, y1 = [t / 2.0 for t in b]
                c.create_rectangle(x0, y0, x1, y1, outline="#00b0ff" if k == 0 else "#ff4081", width=2)
                c.create_text(x0, y0 - 2, text="参考·鸡蛋" if k == 0 else "参考·非鸡蛋",
                              fill="#00b0ff" if k == 0 else "#ff4081", anchor="sw")
        d = getattr(self, "_ref_drag", None)
        if d:
            c.create_rectangle(d[0] / 2, d[1] / 2, d[2] / 2, d[3] / 2, outline="#ffea00", width=2, dash=(4, 3))

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
            box = (min(x0, e.x) * 2, min(y0, e.y) * 2, max(x0, e.x) * 2, max(y0, e.y) * 2)
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
        self.measure(e.x * 2, e.y * 2)

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

    def _capture_sample(self, n_frames=CAPTURE_FRAMES, timeout=CAPTURE_TIMEOUT_S):
        """V3.45: n new frames while the arm is still; ONE source for the whole pose (depth only if >= 80 % of the
        frames have consistent depth, else PnP); outlier frames trimmed; jitter of the kept frames <= 3 mm.
        -> (sample, None) or (None, reason)"""
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
        t_end = time.time() + timeout
        while len(dets) < n_frames and time.time() < t_end and not self.stop_flag:
            fid = self.src.frame_id
            if fid == last:
                time.sleep(0.02)
                continue
            last = fid
            q = self.actual_q()
            d = self.src.detect_marker(m, q)
            if d is None or q is None:
                continue
            if max(abs(a - b) for a, b in zip(q, q0)) > 0.2:
                return None, "机械臂在动：停稳后再记录"
            dets.append(d)
            qs.append(q)
        need = max(5, n_frames // 3)
        if len(dets) < need:
            return None, f"标记只检测到 {len(dets)} 帧（要 ≥ {need}）：标记不在画面里 / 转得太偏？"
        src, pts = choose_source(dets)
        tr = trimmed_points(pts)
        if tr.get("error"):
            return None, f"{'深度' if src == 'depth' else 'PnP'}：{tr['error']}（光线 / 标记太小？）"
        if tr["jitter_mm"] > JITTER_MAX_MM:
            return None, (f"去掉坏帧后抖动 {tr['jitter_mm']:.1f} mm（> {JITTER_MAX_MM:.0f} mm，"
                          f"{'深度' if src == 'depth' else 'PnP'}，保留 {tr['kept']}/{tr['n']} 帧）")
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
        return {"q_deg": [round(v, 3) for v in q], "p_cam": tr["mean"], "T_base_flange": T,
                "spread_mm": tr["jitter_mm"], "n": tr["kept"], "n_frames": tr["n"], "source": src,
                "normal_cam": nrm}, None

    def _capture_retry(self, tag=""):
        """V3.45: capture; on a failed measurement settle once more and measure again (CAPTURE_RETRIES)."""
        fatal = ("相机未连接", "标记设置无效", "没有关节反馈", "FK 失败")
        why = ""
        for attempt in range(1 + CAPTURE_RETRIES):
            s, why = self._capture_sample()
            if s is not None:
                s["attempt"] = attempt + 1
                return s, None
            if self.stop_flag or any(why.startswith(f) for f in fatal) or attempt >= CAPTURE_RETRIES:
                break
            self.log(f"[INFO] [V3.45 calib] {tag}{why} → 再测一次")
            time.sleep(1.0)
            self._wait_still(3.0)
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
                if self.check_state and self.check_state[0] == "run":
                    self.check_state = None
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
        res = self.pipe.move_and_settle(model, list(q_rad), speed, label=label)
        if not res.get("ok"):
            return None, f"运动失败：{res.get('error')}"
        time.sleep(SETTLE_S)
        self._wait_still()
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
                        q = ik(q0r, tgt, T_R(T0))
                        if q is None or not self.pipe.fence_goal_check(model, q)[0]:
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
                if len(Ts) < 3:
                    self.log("[FAIL] [V3.45 calib] 方向探测不够（要起点 + ≥ 2 个方向）：把起点摆到画面中间、离相机 0.6–0.9 m 再试")
                    self.pipe.move_and_settle(model, q0r, speed, label="标定采样 回起点")
                    return
                mm, rms = MarkerModel.from_probes(Ts, Ps, s0.get("normal_cam"))
                self.log(f"[INFO] [V3.45 calib] 相机方向已确定（{len(Ts)} 点，拟合 {rms * 1000:.1f} mm）")
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
            depth = YP.fuse_depth(depths)
            res, diag = YP.locate(d2, depth, dict(self.src.K), np.array(cal["T_base_cam"], dtype=float),
                                  table_z=(f or {}).get("z_table"), fence_poly=(f or {}).get("polygon"), size_mm=size,
                                  calib_note=cnote)
            t2 = time.time()
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
                lab = f"{r['det_id']} {r['yolo']['conf']:.2f}" if r["status"] != "rejected" else \
                    f"✗ {(r['reasons'] or ['?'])[0][:14]}"
                view.append({"box": r["yolo"]["box"], "status": r["status"], "label": lab})

            def show():
                self.yolo_view = view
                for it in self.tv_eggs.get_children():
                    self.tv_eggs.delete(it)
                for r in res:
                    x, y, z = r.get("xyz_m") or (None, None, None)
                    mm = (lambda v: "-" if v is None else f"{v * 1000:.0f}")
                    why = "；".join(r["reasons"]) or ("✓ " + (r.get("pose") or "") + "，轮廓对齐")
                    if r.get("notes"):
                        why += "（" + "；".join(r["notes"]) + "）"
                    self.tv_eggs.insert("", "end", values=(
                        r["det_id"], mm(x), mm(y), mm(z),
                        "-" if r.get("height_mm") is None else f"{r['height_mm']:.0f}",
                        f"{r['yolo']['conf']:.2f}", "-" if r.get("iou") is None else f"{r['iou']:.2f}",
                        f"{r['confidence']:.2f}", "✓" if r["trusted"] else "-", why))
                g = diag.get("ground_minus_z_table_mm")
                self.v_eggs.set(
                    f"{'视觉提示' if mode == 'vp' else '文字提示'} · YOLO {diag['n_2d']} 个 → 鸡蛋 {diag['n_egg']} · 不可信 "
                    f"{diag['n_untrusted']} · 剔除 {diag['n_rejected']} · YOLO {(t1 - t0) * 1000:.0f} ms + 3D "
                    f"{(t2 - t1) * 1000:.0f} ms（深度融合 {len(depths)} 帧）"
                    + ("" if g is None else f"\n局部桌面比安全设置 z_table {'高' if g >= 0 else '低'} {abs(g):.0f} mm"
                       + (" ⚠ 建议在安全设置里更新 z_table" if abs(g) > 8 else ""))
                    + (f" · 桌面平面 RMS {diag['plane_rms_mm']:.1f} mm" if diag.get("plane_rms_mm") is not None else "")
                    + (f"\n⚠ {cnote}" if cnote else "")
                    + (f"\n样本 → {sp}" if sp else ""))
                self._force_redraw()
            self._ui(show)
            self.log(f"[INFO] [V3.47 locate] 鸡蛋识别（{mode}）：YOLO {diag['n_2d']} → 鸡蛋 {diag['n_egg']} / 不可信 "
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
                for d in dets:
                    x, y, z = d["xyz_m"]
                    self.tv_eggs.insert("", "end", values=(d.get("det_id"), f"{x * 1000:.0f}", f"{y * 1000:.0f}",
                                                           f"{z * 1000:.0f}", "-", "-", "-", f"{d.get('confidence', 0):.2f}",
                                                           "✓" if d.get("trusted") else "-", "几何法 V3.08"))
                self.v_eggs.set(f"几何法：检测到 {len(dets)} 个（可信 {sum(1 for d in dets if d.get('trusted'))}）· 有效点 "
                                f"{diag.get('points_valid')} · 鸡蛋掩膜点 {diag.get('points_egg_mask')}")
                self._force_redraw()
            self._ui(show)
            self.log(f"[INFO] [V3.25 locate] 鸡蛋识别：{len(dets)} 个 " + "; ".join(
                f"{d.get('det_id')} ({d['xyz_m'][0] * 1000:.0f}, {d['xyz_m'][1] * 1000:.0f}, {d['xyz_m'][2] * 1000:.0f}) mm"
                for d in dets[:6]))
        self._run(job)
