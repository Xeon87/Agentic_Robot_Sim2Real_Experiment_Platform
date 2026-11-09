#!/usr/bin/env python3
"""NERO V3.89 regression & verification suite:
  A1. Global Kronecker-seeded + optical-axis stabilized Eye-in-Hand solver + corner-based (fx, k1) self-calibration on
      both standard dome and counter-tilted dome with ArUco IPPE planar flips (verifies good=True, LOO <= 5.0 mm, and
      link7->cam stability across 18, 15, 12, 11, and 8 sample subsets without local-minimum traps or optical-axis runaway).
  A2. ChArUco bounding-box `corners_px` immunity (`_is_axis_aligned_charuco_box` + safety gate on `self_calibrated_K`).
  A3. Sweet-spot & distance-bounded Look-At Dome planner (`plan_eih_auto_targets_4mp`) for n=15 poses.
  A4. V4L2-safe sub-millisecond `autotune_wrist_uvc` + `set_exposure_gain`.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="nero_v389_"))
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")

import NERO_V325_Camera as CAM                  # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49       # noqa: E402
import NERO_V386_Real_Wrist_Camera as W86       # noqa: E402

assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11")
assert W86.VERSION in ("V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11")
assert hasattr(W86, "MARKER_V389")

# ------------------------------------------------------------------ A1: Garage 18-pose + Counter-Tilt Dome + IPPE Flips
rng = np.random.default_rng(19)
X_true = CAM.T_from(
    [[0.0, -0.37, 0.93], [-1.0, 0.0, 0.0], [0.0, -0.93, -0.37]],
    [0.062, -0.005, -0.040],
)
Y_true = CAM.T_from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [0.30, 0.00, 0.02])
fx_true, fx_nom = 1280.0, 1056.0
w, h = 1920, 1080
K_true = np.array([[fx_true, 0.0, w / 2.0], [0.0, fx_true, h / 2.0], [0.0, 0.0, 1.0]], dtype=float)
dist_true = np.array([-0.15, 0.02, 0.0, 0.0, 0.0], dtype=float)
Km_nom = np.array([[fx_nom, 0.0, w / 2.0], [0.0, fx_nom, h / 2.0], [0.0, 0.0, 1.0]], dtype=float)
s_m = 0.024
obj = np.array([[-s_m / 2, s_m / 2, 0.0], [s_m / 2, s_m / 2, 0.0],
                [s_m / 2, -s_m / 2, 0.0], [-s_m / 2, -s_m / 2, 0.0]], dtype=float)

samples = []
for i in range(18):
    dx = rng.uniform(-0.12, 0.12)
    dy = rng.uniform(-0.12, 0.12)
    dz = rng.uniform(0.22, 0.28)
    p_c = np.array([0.30 + dx, 0.0 + dy, 0.02 + dz])
    # Counter-tilted Look-At Dome where p_cent has negative correlation with -t_A
    aim = np.array([0.30 - 0.35 * dx + rng.uniform(-0.015, 0.015),
                    -0.35 * dy + rng.uniform(-0.015, 0.015), 0.02])
    z_ax = (aim - p_c) / np.linalg.norm(aim - p_c)
    x_ax = np.cross(np.array([0.0, 1.0, 0.0]), z_ax)
    x_ax /= np.linalg.norm(x_ax)
    R_bc0 = np.column_stack([x_ax, np.cross(z_ax, x_ax), z_ax])
    roll = rng.uniform(-0.40, 0.40)
    R_bc = R_bc0 @ np.array([[np.cos(roll), -np.sin(roll), 0.0],
                             [np.sin(roll), np.cos(roll), 0.0],
                             [0.0, 0.0, 1.0]])
    T_bc = np.eye(4)
    T_bc[:3, :3] = R_bc
    T_bc[:3, 3] = p_c
    T_bf = T_bc @ np.linalg.inv(np.array(X_true))
    T_cm_true = np.linalg.inv(T_bc) @ np.array(Y_true)
    rv_t, _ = cv2.Rodrigues(T_cm_true[:3, :3])
    tv_t = T_cm_true[:3, 3]
    uv, _ = cv2.projectPoints(obj, rv_t, tv_t, K_true, dist_true)
    uv = uv.reshape(4, 2) + rng.normal(0.0, 0.40, (4, 2))
    ok, rv_n, tv_n = cv2.solvePnP(obj, uv, Km_nom, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)
    assert ok
    p_n = tv_n.flatten().tolist()
    r_n = rv_n.flatten().tolist()
    samples.append({
        "q_deg": [float(i)] * 7,
        "p_cam": p_n,
        "rvec": r_n,
        "T_cam_marker": CAM.T_from(CAM.rodrigues(r_n), p_n),
        "T_base_flange": T_bf.tolist(),
        "corners_px": uv.tolist(),
        "image_size": [w, h],
    })

t0 = time.time()
res_full = W86.calibrate_eih_4mp([dict(s) for s in samples], size_mm=24.0)
dt = time.time() - t0
assert res_full["ok"] and res_full["good"], f"Expected good=True on counter-tilt dome, got: {res_full}"
assert res_full["loo_mean_mm"] <= 3.0 and res_full["loo_max_mm"] <= 5.0, res_full
assert "self_calibrated_K" in res_full, "Expected self_calibrated_K to be populated"
assert abs(res_full["self_calibrated_K"]["fx"] - fx_true) < 90.0

t_vectors = []
for n_sub in (18, 15, 12, 11, 8):
    sub_s = [dict(s) for s in samples[:n_sub]]
    r_sub = W86.calibrate_eih_4mp(sub_s, size_mm=24.0)
    assert r_sub["ok"] and r_sub["good"], f"Subset n={n_sub} failed: {r_sub}"
    t_mm = [v * 1000.0 for v in r_sub["cam_in_flange"]]
    t_vectors.append(t_mm)
t_arr = np.asarray(t_vectors, dtype=float)
max_drift_mm = float(np.max(np.linalg.norm(t_arr - t_arr[0], axis=1)))
assert max_drift_mm < 15.0, f"link7->cam drifted too much across subsets: {t_vectors} (drift={max_drift_mm:.1f} mm)"
print(f"[A1] Kronecker-seeded EIH + (fx, k1) self-calibration passed in {dt:.2f}s: "
      f"LOO={res_full['loo_mean_mm']:.2f}/{res_full['loo_max_mm']:.2f} mm, subset drift={max_drift_mm:.1f} mm ✓")

# ------------------------------------------------------------------ A2: ChArUco axis-aligned box_corners immunity
samps_box = [dict(s, corners_px=[[740.0, 390.0], [1180.0, 390.0], [1180.0, 690.0], [740.0, 690.0]]) for s in samples]
res_box = W86.calibrate_eih_4mp(samps_box, size_mm=24.0)
assert res_box["ok"] and res_box["good"], f"ChArUco box_corners fallback failed: {res_box}"
assert "self_calibrated_K" not in res_box, "ChArUco box_corners must not trigger 4-corner ArUco PnP"
print(f"[A2] ChArUco box_corners immunity verified: LOO={res_box['loo_mean_mm']:.2f}/{res_box['loo_max_mm']:.2f} mm ✓")

# ------------------------------------------------------------------ A3: Sweet-spot & distance-bounded Dome Planner
K_4mp = {"width": 1920, "height": 1080, "fx": 1056.0, "fy": 1056.0, "ppx": 960.0, "ppy": 540.0}
eih_mod = {"T_flange_cam": X_true, "T_base_marker": Y_true, "marker_in_base": [0.30, 0.00, 0.02]}
T_bc0 = CAM.T_from([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], [0.30, 0.00, 0.29])
T_bf0 = CAM.mat_mul(T_bc0, CAM.T_inv(X_true))
p_bf0 = CAM.T_t(T_bf0)
R_bf0 = CAM.T_R(T_bf0)


def mock_fk(q):
    dx = 0.25 * q[0] + 0.15 * q[2]
    dy = 0.25 * q[1] + 0.15 * q[3]
    dz = 0.20 * q[4]
    rx, ry, rz = 0.8 * q[3], 0.8 * q[4], 0.8 * q[5] + 0.5 * q[6]
    R_d = CAM.rodrigues([rx, ry, rz])
    return CAM.T_from(CAM.mat_mul(R_d, R_bf0), [p_bf0[0] + dx, p_bf0[1] + dy, p_bf0[2] + dz])


limits_rad = [(-2.5, 2.5)] * 7
targets, st = W86.plan_eih_auto_targets_4mp(mock_fk, None, [0.0] * 7, limits_rad, eih_mod, K_4mp, n=15, size_m=0.024)
assert len(targets) == 15, f"Expected 15 targets, got {len(targets)}"
for t in targets:
    assert 0.11 <= t["z_cam"] <= 0.42, f"Pose z_cam={t['z_cam']} out of safe bounds"
    assert t["off_axis"] <= 0.48, f"Pose off_axis={t['off_axis']} too close to edge"
print(f"[A3] Sweet-spot dome planner generated {len(targets)} poses (max z={max(t['z_cam'] for t in targets)*1000:.0f} mm, "
      f"max off_axis={max(t['off_axis'] for t in targets):.2f}) ✓")

# ------------------------------------------------------------------ A4: V4L2-safe autotune (no black-frame mean=0)
state = {"exp": 100.0, "gain": 0}


def set_eg_sim(e, g):
    state["exp"], state["gain"] = float(e), int(g)


def measure_sim():
    e = state["exp"]
    if e < 0.6:
        return {"mean": 0.0, "p995": 0.0, "p995_ex": 0.0, "sat_pct": 0.0, "sat_large_pct": 0.0, "spots": 0}
    mean_v = min(255.0, 72.0 * e + 0.4 * state["gain"])
    sat_v = max(0.0, (e - 1.2) * 18.0)
    return {
        "mean": round(mean_v, 1),
        "p995": 255.0 if sat_v > 0 else min(240.0, mean_v * 1.45),
        "p995_ex": 255.0 if sat_v > 0 else min(235.0, mean_v * 1.40),
        "sat_pct": round(sat_v, 2),
        "sat_large_pct": round(sat_v, 2),
        "spots": 0,
    }


res_tune, trace = W86.autotune_wrist_uvc(set_eg_sim, measure_sim, power_hz=60, gain_range=(0, 128), mean_lo=100.0)
assert all(t["exposure"] >= 0.8 for t in trace), f"Autotune probed unsafe <0.8 exposure: {trace}"
assert res_tune["stats"]["mean"] > 50.0, f"Autotune ended on dark frame: {res_tune}"
assert res_tune["ok"], f"Expected autotune ok=True, got: {res_tune}"
print(f"[A4] V4L2-safe autotune passed: chosen exp={res_tune['exposure']}, gain={res_tune['gain']}, "
      f"mean={res_tune['stats']['mean']} ✓")
print("ALL V3.89 TESTS PASSED")
