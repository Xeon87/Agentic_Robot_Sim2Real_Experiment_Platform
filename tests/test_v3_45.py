#!/usr/bin/env python3
"""V3.45 Camera调试 / 手眼标定 (offline, no camera, no ROS).

D1 marker sizes: 40 x 40 mm (default) and 30 x 30 mm presets
D2 one measurement source per pose: depth only if >= 80 % of the frames have it, else PnP for every frame
D3 jitter without bad frames: outlier frames trimmed, jitter of the kept frames; too few consistent frames -> error
D4 camera direction from 3 small translations (Kabsch) predicts the marker within a few mm; visibility check
   (inside the image, facing the camera <= 45 deg, range)
D5 auto targets on the NERO URDF: all predicted visible, position span >= 20 cm, wrist rotation <= 25 deg, soft limits
D6 rotation span = largest pairwise difference
D7 detect_aruco returns both p_pnp and p_depth + the marker normal (needs numpy + OpenCV aruco)
D8 redraw / overlays: no direct det / valid_ws subscripting left (the 'NoneType' refresh bug)
"""
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V325_Camera as CAM                 # noqa: E402
import NERO_V315_Consistency_Safety as CS      # noqa: E402

SRC = (ROOT / "NERO_V325_Camera.py").read_text()

# ------------------------------------------------------------------ D1
assert CAM.MARKER_SIZES_MM == (40.0, 30.0) and CAM.DEFAULT_MARKER["size_mm"] == 40.0
assert 'values=[f"{v:g}" for v in MARKER_SIZES_MM]' in SRC and "40×40 / 30×30" in SRC
print("[D1] marker side presets 40 x 40 mm (default) / 30 x 30 mm, editable for the measured value")

# ------------------------------------------------------------------ D2
P = [0.10, -0.20, 0.80]
dets = [{"p_pnp": [P[0], P[1], P[2] + 0.01], "p_depth": P, "p_cam": P} for _ in range(9)] + \
       [{"p_pnp": P, "p_depth": None, "p_cam": P}]
src, pts = CAM.choose_source(dets)
assert src == "depth" and len(pts) == 9 and all(p == P for p in pts)
dets2 = dets[:7] + [{"p_pnp": P, "p_depth": None, "p_cam": P}] * 3
src2, pts2 = CAM.choose_source(dets2)
assert src2 == "pnp" and len(pts2) == 10                         # 70 % depth -> PnP for ALL frames, no mixing
print("[D2] one source per pose: 90 % depth -> depth frames only; 70 % -> PnP for every frame")

# ------------------------------------------------------------------ D3
rnd = random.Random(4)
good = [[P[0] + rnd.gauss(0, 0.0004), P[1] + rnd.gauss(0, 0.0004), P[2] + rnd.gauss(0, 0.0004)] for _ in range(13)]
bad = [[P[0] + 0.03, P[1], P[2]], [P[0], P[1] - 0.08, P[2] + 0.05]]
tr = CAM.trimmed_points(good + bad)
assert tr["kept"] == 13 and tr["jitter_mm"] < 2.0 and CAM.vnorm(CAM.vsub(tr["mean"], P)) < 0.0005, tr
old_style = max(CAM.vnorm(CAM.vsub(p, [sorted(q[i] for q in good + bad)[7] for i in range(3)])) for p in good + bad)
assert old_style > 0.05                                           # V3.25 max-deviation would have rejected the pose
noisy = [[P[0] + rnd.uniform(-0.02, 0.02), P[1] + rnd.uniform(-0.02, 0.02), P[2]] for _ in range(15)]
assert CAM.trimmed_points(noisy).get("error") or CAM.trimmed_points(noisy)["jitter_mm"] > 3.0
assert CAM.CAPTURE_FRAMES == 15 and CAM.JITTER_MAX_MM == 3.0 and CAM.SETTLE_S >= 1.5 and CAM.CAPTURE_RETRIES == 1
print(f"[D3] 2 bad frames of 15 trimmed -> jitter {tr['jitter_mm']:.1f} mm (V3.25 max deviation {old_style * 1000:.0f} mm "
      "would have failed); uniformly noisy pose still rejected; 15 frames, settle 1.5 s + still check, 1 retry")

# ------------------------------------------------------------------ fixture: NERO URDF + a fixed camera
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
LIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708)]


def fk(q_rad):
    T = CS.fk(U, dict(zip(J, q_rad)), "base_link", "link7")
    return [list(map(float, r)) for r in T]


def look_at(pos, tgt):
    f = CAM.vsub(tgt, pos); n = CAM.vnorm(f); f = [v / n for v in f]
    x = [f[1], -f[0], 0.0]; n = CAM.vnorm(x); x = [v / n for v in x]
    yu = [x[1] * f[2] - x[2] * f[1], x[2] * f[0] - x[0] * f[2], x[0] * f[1] - x[1] * f[0]]
    return CAM.T_from([[x[i], -yu[i], f[i]] for i in range(3)], pos)


K = {"width": 1280, "height": 720, "fx": 908.7, "fy": 908.7, "ppx": 640.1, "ppy": 366.6}
q0 = [math.radians(v) for v in (0, 30, 0, 80, 0, 40, 0)]
T0 = fk(q0)
Y = [0.015, -0.025, 0.06]
m0 = CAM.T_apply(T0, Y)
X = look_at([m0[0] + 0.35, m0[1] - 0.70, m0[2] + 0.45], m0)          # camera ~0.9 m away, looking down at the marker
Xi = CAM.T_inv(X)
n_b0 = CAM.vsub(CAM.T_t(X), m0)
nn = CAM.vnorm(n_b0)
N_F = CAM.mat_vec(CAM.transpose(CAM.T_R(T0)), [v / nn for v in n_b0])   # marker glued facing the camera at q0


def observe(T):
    p = CAM.T_apply(Xi, CAM.T_apply(T, Y))
    nc = CAM.mat_vec(CAM.T_R(Xi), CAM.mat_vec(CAM.T_R(T), N_F))
    return p, nc


# ------------------------------------------------------------------ D4
p_s, n_s = observe(T0)
Ts, Ps = [T0], [p_s]
for ax in range(3):
    tgt = CAM.T_t(T0)
    tgt[ax] += CAM.AUTO_PROBE_M
    q, _e = CS.ik_tool(U, J, q0, "base_link", None, tgt, LIM, iters=150, tol=1e-3, keep_R=CAM.T_R(T0))
    Tq = fk(q)
    Ts.append(Tq)
    Ps.append(observe(Tq)[0])
mm, rms = CAM.MarkerModel.from_probes(Ts, Ps, n_s)
assert rms < 0.003, rms                                           # orientation fixed -> exact rigid fit
errs = []
for k in range(40):
    ax = [rnd.gauss(0, 1) for _ in range(3)]
    R = CAM.mat_mul(CAM.axis_angle_R(ax, rnd.uniform(0, 25)), CAM.T_R(T0))
    pos = [CAM.T_t(T0)[i] + rnd.uniform(-0.1, 0.1) for i in range(3)]
    Tt = CAM.T_from(R, pos)
    pc_true, nc_true = observe(Tt)
    pc, nc = mm.predict(Tt)
    errs.append(CAM.vnorm(CAM.vsub(pc, pc_true)))
    assert CAM.rot_angle_deg([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) == 0.0
    assert math.degrees(math.acos(max(-1, min(1, sum(a * b for a, b in zip(nc, nc_true)))))) < 1.0
assert max(errs) < 0.035 and sum(errs) / len(errs) < 0.02, (max(errs), sum(errs) / len(errs))
mc = CAM.MarkerModel.from_calib({"T_base_cam": X, "marker_in_flange": Y}, T0, n_s)
assert CAM.vnorm(CAM.vsub(mc.predict(T0)[0], p_s)) < 1e-9
ok, why, info = mc.visible(T0, K)
assert ok and info["view_deg"] < 1.0, (why, info)
R_away = CAM.mat_mul(CAM.axis_angle_R([1, 0, 0], 80), CAM.T_R(T0))
assert not mc.visible(CAM.T_from(R_away, CAM.T_t(T0)), K)[0]
far = CAM.T_from(CAM.T_R(T0), [CAM.T_t(T0)[0] + 0.6, CAM.T_t(T0)[1], CAM.T_t(T0)[2]])
assert mc.visible(far, K)[1] in ("出画面", "距离不合适", "标记转得太偏")
print(f"[D4] probes (start + 3 x 4 cm) -> camera direction, fit {rms * 1000:.2f} mm; prediction error over +-10 cm / "
      f"<= 25 deg: mean {sum(errs) / len(errs) * 1000:.0f}, max {max(errs) * 1000:.0f} mm (enough for visibility); "
      "facing / turned-away / out-of-image classified")


# ------------------------------------------------------------------ D5
def ik(seed, xyz, R):
    q, _e = CS.ik_tool(U, J, list(seed), "base_link", None, list(xyz), LIM, iters=150, tol=1e-3, keep_R=R)
    if q is None:
        return None
    T = fk(q)
    if CAM.vnorm(CAM.vsub(CAM.T_t(T), xyz)) > 0.003 or CAM.rot_angle_deg(CAM.mat_mul(CAM.transpose(CAM.T_R(T)), R)) > 3:
        return None
    return q


targets, st = CAM.plan_auto_targets(fk, ik, q0, LIM, mm, K, 15, fence_fn=lambda q: True, seed=11)
assert len(targets) == 15, (len(targets), st)
assert st["span_m"] >= CAM.AUTO_SPAN_TARGET_M, st
for t in targets:
    ptrue, ntrue = observe(t["T"])
    uv = CAM.project(ptrue, K)
    cosv = -sum(a * b for a, b in zip(ntrue, ptrue)) / CAM.vnorm(ptrue)
    assert 0 < uv[0] < 1280 and 0 < uv[1] < 720 and math.degrees(math.acos(min(1, cosv))) < 50, (uv, cosv)
    assert CAM.rot_angle_deg(CAM.mat_mul(CAM.transpose(CAM.T_R(T0)), CAM.T_R(t["T"]))) <= CAM.AUTO_ROT_MAX_DEG + 3.5
    for v, (lo, hi) in zip(t["q"], LIM):
        assert lo + math.radians(3) - 1e-9 <= v <= hi - math.radians(3) + 1e-9
samples = []
for t in [{"T": T0}] + targets:
    pc, _n = observe(t["T"])
    samples.append({"T_base_flange": t["T"], "p_cam": [v + rnd.gauss(0, 0.0008) for v in pc]})
r = CAM.calibrate(samples)
assert r["ok"] and r["trans_span_m"] >= 0.2 and r["rot_span_deg"] >= 25 and r["good"], (r["trans_span_m"], r["rot_span_deg"], r["loo_mean_mm"])
print(f"[D5] 15 auto targets: all truly visible (in image, <= 50 deg), position span {st['span_m'] * 100:.0f} cm, wrist "
      f"<= {st['rot_max_deg']:.0f} deg, soft limits kept ({st['tries']} tries) -> calibration with 0.8 mm noise: "
      f"loo {r['loo_mean_mm']:.2f}/{r['loo_max_mm']:.2f} mm, 合格")

# ------------------------------------------------------------------ D6
A = [CAM.T_from(CAM.axis_angle_R([0, 0, 1], a), [0, 0, 0]) for a in (0, 15, -15)]
assert abs(CAM.rotation_span_deg(A) - 30.0) < 1e-6
print("[D6] rotation span = largest pairwise difference (+15 / -15 deg -> 30 deg)")

# ------------------------------------------------------------------ D7
try:
    import numpy as np
    import cv2
    have_cv = hasattr(cv2, "aruco")
except Exception:
    have_cv = False
if have_cv:
    S, pad = 400, 60
    dct = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    mk = cv2.aruco.generateImageMarker(dct, 0, S) if hasattr(cv2.aruco, "generateImageMarker") else cv2.aruco.drawMarker(dct, 0, S)
    mkp = np.full((S + 2 * pad, S + 2 * pad), 255, np.uint8)
    mkp[pad:pad + S, pad:pad + S] = mk
    size, h = 0.040, 0.020
    Rm = CAM.mat_mul(CAM.rodrigues([0.3, -0.2, 0.1]), [[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    tm = [0.05, -0.02, 0.75]
    sc = size / S
    srcp, dstp = [], []
    for (px, py) in ((0, 0), (S + 2 * pad, 0), (S + 2 * pad, S + 2 * pad), (0, S + 2 * pad)):
        mx, my = (px - pad) * sc - h, h - (py - pad) * sc
        srcp.append([px, py])
        dstp.append(CAM.project(CAM.T_apply(CAM.T_from(Rm, tm), [mx, my, 0.0]), K))
    Hm = cv2.getPerspectiveTransform(np.float32(srcp), np.float32(dstp))
    rgb = cv2.cvtColor(cv2.warpPerspective(mkp, Hm, (1280, 720), borderValue=180), cv2.COLOR_GRAY2RGB)
    nrm = [Rm[0][2], Rm[1][2], Rm[2][2]]
    vv, uu = np.mgrid[0:720, 0:1280]
    den = nrm[0] * (uu - K["ppx"]) / K["fx"] + nrm[1] * (vv - K["ppy"]) / K["fy"] + nrm[2]
    depth = ((nrm[0] * tm[0] + nrm[1] * tm[1] + nrm[2] * tm[2]) / den).astype(np.float32)
    d0 = CAM.detect_aruco(rgb, K, [0] * 5, "DICT_4X4_50", 0, size)
    d1 = CAM.detect_aruco(rgb, K, [0] * 5, "DICT_4X4_50", 0, size, depth_m=depth)
    assert d0["p_depth"] is None and d0["source"] == "pnp" and d1["p_depth"] is not None and d1["source"] == "depth"
    assert d1["p_pnp"] == d0["p_pnp"] and CAM.vnorm(CAM.vsub(d1["p_depth"], tm)) < 0.002
    ang = math.degrees(math.acos(max(-1, min(1, sum(a * b for a, b in zip(d1["normal_cam"], nrm))))))
    assert ang < 5.0, ang
    print(f"[D7] detect_aruco: p_pnp + p_depth (depth error {CAM.vnorm(CAM.vsub(d1['p_depth'], tm)) * 1000:.1f} mm) + "
          f"marker normal within {ang:.1f} deg")
else:
    print("[D7] numpy / OpenCV aruco missing -> skipped")

# ------------------------------------------------------------------ D8
red = SRC[SRC.index("    def _redraw(self):"):SRC.index("    def _overlay(self, fn, *a):")]
assert "self.det[" not in red and "det = self.det" in red and "self._overlay(self._draw_zone" in red
assert 'valid_ws["fence"]' not in SRC and "def _fence(self):" in SRC
assert "self.det = None                              # V3.45" in SRC
print("[D8] redraw uses one detection snapshot; overlays isolated; missing safety zone handled (refresh 'NoneType' bug)")
print("ALL V3.45 TESTS PASSED")
