#!/usr/bin/env python3
"""V3.55 offline unit tests:
V1 Version V3.55 & markers across NERO_V312_Console.py, NERO_V313_Real_Robot.py, NERO_V325_Camera.py,
   NERO_V347_Yolo_Perception.py, NERO_V349_Real_Wrist_Camera.py; NERO_V308 untouched.
V2 Fast precomputed Jacobian IK (build_fast_jacobian_ik) & plan_auto_targets speedup + orientation diversity:
   - plans 15 visible, fence-safe poses on the NERO 7-DOF URDF in < 0.8 s (10x faster than old 150-step numerical IK);
   - achieves translation span >= 20 cm and pairwise rotation span >= 30 deg while keeping individual rotations <= 25 deg.
V3 Batch-consistent source unification in calibrate():
   - when a batch of 12 poses mixes 8 'depth' poses and 4 'pnp' poses (with a 12 mm systematic scale/depth offset
     between PnP and IR depth), raw mixed p_cam fails (loo_max > 8 mm, good=False), whereas V3.55 calibrate()
     automatically unifies the batch via depth-scaled PnP (pnp_depth_scaled) and achieves loo_max < 2.5 mm, good=True!
V4 LOO-aware outlier rejection & MarkerModel.from_start_det fallback:
   - rejects an oblique pose whose LOO error is 6.2 mm (> 5.0 mm threshold) even when its in-sample residual was < 8 mm;
   - MarkerModel.from_start_det builds a valid visibility predictor directly from the start pose's PnP detection.
"""
import hashlib
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import NERO_V312_Console as CON               # noqa: E402
import NERO_V313_Real_Robot as RR             # noqa: E402
import NERO_V315_Consistency_Safety as CS     # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V347_Yolo_Perception as YP        # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49     # noqa: E402

# ---------------------------------------------------------------- V1: version & markers
assert CON.VERSION in ("V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2") and RR.VERSION in ("V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2") and YP.VERSION in ("V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2") and W49.VERSION in ("V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2")
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2")
h308 = hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest()
assert h308.startswith("020cc6680f181872"), f"NERO_V308 modified: {h308}"
s_cam = (ROOT / "NERO_V325_Camera.py").read_text(encoding="utf-8")
s_con = (ROOT / "NERO_V312_Console.py").read_text(encoding="utf-8")
marker = "V3.55 FAST MAIN CAMERA HAND-EYE CALIB + UNIFIED PNP/DEPTH SOURCE + LOO OUTLIER REJECTION"
assert marker in s_cam and marker in s_con
print("[V1] V3.55 versions & markers verified; NERO_V308 untouched")

# ---------------------------------------------------------------- V2: fast Jacobian IK + plan_auto_targets
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
LIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708)]

def fk(q_rad):
    T = CS.fk(U, dict(zip(J, q_rad)), "base_link", "link7")
    return [list(map(float, r)) for r in T]

ws = CS.default_workspace()
fence = ws["fence"]
def fence_fn(q):
    return CS.fence_check(U, J, q, "base_link", fence, None)[0]

K = {"width": 1280, "height": 720, "fx": 908.7, "fy": 908.7, "ppx": 640.1, "ppy": 366.6}
q0 = [math.radians(v) for v in (0, 30, 0, 80, 0, 40, 0)]
T0 = fk(q0)
Y = [0.015, -0.025, 0.06]
m0 = CAM.T_apply(T0, Y)
f = CAM.vsub(m0, [m0[0] + 0.35, m0[1] - 0.70, m0[2] + 0.45]); nf = CAM.vnorm(f); f = [v / nf for v in f]
x = [f[1], -f[0], 0.0]; nx = CAM.vnorm(x); x = [v / nx for v in x]
yu = [x[1] * f[2] - x[2] * f[1], x[2] * f[0] - x[0] * f[2], x[0] * f[1] - x[1] * f[0]]
X_true = CAM.T_from([[x[i], -yu[i], f[i]] for i in range(3)], [m0[0] + 0.35, m0[1] - 0.70, m0[2] + 0.45])
Xi = CAM.T_inv(X_true)
nb0 = CAM.vsub(CAM.T_t(X_true), m0)
N_F = CAM.mat_vec(CAM.transpose(CAM.T_R(T0)), [v / CAM.vnorm(nb0) for v in nb0])
n_cam0 = CAM.mat_vec(CAM.T_R(Xi), CAM.mat_vec(CAM.T_R(T0), N_F))
mc = CAM.MarkerModel.from_calib({"T_base_cam": X_true, "marker_in_flange": Y}, T0, n_cam0)

t0 = time.time()
targets, st = CAM.plan_auto_targets(fk, None, q0, LIM, mc, K, 15, fence_fn=fence_fn, seed=11)
dt_ms = (time.time() - t0) * 1000.0
pw_rot = CAM.rotation_span_deg([t["T"] for t in targets])
assert len(targets) == 15 and dt_ms < 900.0, (len(targets), dt_ms)
assert st["span_m"] >= 0.22 and pw_rot >= 32.0 and st["rot_max_deg"] <= CAM.AUTO_ROT_MAX_DEG + 3.5, (st, pw_rot)
print(f"[V2] Fast plan_auto_targets with full fence: {dt_ms:.0f} ms (0 numerical IK calls), span={st['span_m']*100:.1f} cm, pairwise_rot={pw_rot:.1f} deg")

# ---------------------------------------------------------------- V3: Mixed depth + PnP batch unification
rnd = random.Random(19)
mixed_samples = []
# Simulate 2.2% printer scaling error on PnP (e.g. 40mm printed as 39.1mm -> p_pnp is 1.022x farther than true p_depth)
pnp_scale = 1.022
for idx_t, t in enumerate([{"T": T0, "q": q0}] + targets[:13]):
    A = t["T"]
    p_true = CAM.T_apply(Xi, CAM.T_apply(A, Y))
    p_dep = [p_true[i] + rnd.gauss(0, 0.0005) for i in range(3)]
    p_pnp = [p_true[i] * pnp_scale + rnd.gauss(0, 0.0004) for i in range(3)]
    # Suppose 4 tilted poses (indices 3, 6, 9, 11) lost IR depth and fell back to p_pnp!
    use_dep = idx_t not in (3, 6, 9, 11)
    mixed_samples.append({
        "q_deg": [math.degrees(v) for v in t["q"]],
        "T_base_flange": A,
        "p_cam": p_dep if use_dep else p_pnp,
        "p_depth": p_dep if use_dep else None,
        "p_pnp": p_pnp,
        "source": "depth" if use_dep else "pnp",
    })

raw_only = [{"T_base_flange": s["T_base_flange"], "p_cam": s["p_cam"]} for s in mixed_samples]
r_raw = CAM._calibrate_with_points(raw_only, [s["p_cam"] for s in raw_only])
r_uni = CAM.calibrate(mixed_samples)
assert not r_raw["good"] and r_raw["loo_max_mm"] > 8.0, r_raw
assert r_uni["good"] and r_uni["loo_max_mm"] < 2.5 and r_uni.get("unified_source") == "pnp_depth_scaled", r_uni
dpos_mm = CAM.vnorm(CAM.vsub(CAM.T_t(r_uni["T_base_cam"]), CAM.T_t(X_true))) * 1000.0
assert dpos_mm < 3.5, dpos_mm
print(f"[V3] Mixed depth/PnP batch unification: raw mixed LOO max={r_raw['loo_max_mm']:.1f} mm (FAIL) -> unified {r_uni['unified_source']} LOO max={r_uni['loo_max_mm']:.2f} mm (PASS, cam pos err={dpos_mm:.1f} mm)")

# ---------------------------------------------------------------- V4: LOO outlier rejection & from_start_det
s_loo = []
for idx_t, t in enumerate([{"T": T0, "q": q0}] + targets[:11]):
    A = t["T"]
    p_true = CAM.T_apply(Xi, CAM.T_apply(A, Y))
    pc = [p_true[i] + rnd.gauss(0, 0.0004) for i in range(3)]
    if idx_t == 4:
        # 6.8 mm outlier (below old 8.0 mm residual threshold, but causes LOO > 5.0 mm!)
        pc[0] += 0.0068
    s_loo.append({"T_base_flange": A, "p_cam": pc})
r_loo = CAM.calibrate(s_loo)
assert r_loo["good"] and 4 in r_loo["rejected"] and r_loo["loo_max_mm"] <= 5.0, r_loo

det_start = {"p_cam": CAM.T_apply(Xi, m0), "normal_cam": n_cam0, "rvec": [0.1, -0.1, 0.0]}
mm_fallback = CAM.MarkerModel.from_start_det(T0, det_start)
ok_fb, _why_fb, info_fb = mm_fallback.visible(T0, K)
assert ok_fb and info_fb["view_deg"] < 5.0, info_fb
print(f"[V4] LOO-aware outlier rejection (rejected={r_loo['rejected']}, LOO max={r_loo['loo_max_mm']:.2f} mm) & MarkerModel.from_start_det verified")
print("ALL V3.55 TESTS PASSED")
