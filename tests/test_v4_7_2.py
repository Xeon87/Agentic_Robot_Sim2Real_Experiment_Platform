#!/usr/bin/env python3
"""NERO V4.7.2 offline tests (the GUI life cycle is in tests/test_v4_7_2_gui.py).

H1 grade_eih with the user's real results: 1.40/2.00 mm, LOO 2.96/5.07 mm -> usable (was silently not saved);
   35.6 / 67 / 85 / 110 mm -> not usable; good stays good.
H2 fingerprint_match: same intrinsics (also at another resolution) match; 10 % fx -> mismatch; old file -> unknown.
H3 check_consistency: poses agree -> ok; target moved 30 mm -> ok + moved; poses disagree -> not ok.
H4 PLANAR POSE AMBIGUITY, ArUco 23.8 mm marker seen from above (the user's set-up): solvePnP(IPPE_SQUARE) as V3.25
   sometimes picks the clearly worse (mirror) pose (0.6 px corner noise: ~4 % of the views; a single small marker's
   tilt is only good to 2-12 deg anyway); with both solutions kept (planar_pose_candidates) the clearly better one is
   taken in every such view; the solve is no worse than before and within 3 mm of the true hand-eye.
H5 the same for the ChArUco board path (eih_samples_from_intr_samples): every view's rotation within 0.5 deg.
H6 P10: an 8-parameter rational model whose k4..k6 cancel k1..k3 (coefficients > 8) is NOT cut to 5 parameters.
H7 versions / markers.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ["NERO_CALIB_DIR"] = tempfile.mkdtemp(prefix="nero_v472_calib_")

import cv2                                       # noqa: E402
import NERO_V325_Camera as CAM                   # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49        # noqa: E402
import NERO_V386_Real_Wrist_Camera as W86        # noqa: E402
import NERO_V472_Wrist_Calib as WC               # noqa: E402

K4 = {"width": 1920, "height": 1080, "fx": 905.0, "fy": 905.0, "ppx": 961.0, "ppy": 538.0}
KM = np.array([[905.0, 0, 961.0], [0, 905.0, 538.0], [0, 0, 1]], float)


def rot(axis, deg):
    return cv2.Rodrigues(np.asarray(axis, float) / np.linalg.norm(axis) * math.radians(deg))[0]


def T(R, t):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


# link7 -> camera like the user's (log 2026-11-10: +59.3, +46.2, +0.1 mm), camera looking along link7 +x
X_TRUE = T(rot([0, 1, 0], 90) @ rot([0, 0, 1], 90), [0.0593, 0.0462, 0.0001])
Y_MARK = T(np.eye(3), [-0.42, 0.05, 0.0])                       # target flat on the table


def views(n=12, seed=3):
    """camera poses ~ 22 cm above the target looking down, tilts up to 18 deg, wrist turns up to 40 deg"""
    rng = np.random.default_rng(seed)
    out = []
    down = np.diag([1.0, -1.0, -1.0])                           # camera z down onto the table
    for _ in range(n):
        R_c = rot([0, 0, 1], rng.uniform(-40, 40)) @ rot([1, 0, 0], rng.uniform(-18, 18)) @ \
            rot([0, 1, 0], rng.uniform(-18, 18)) @ down
        c = Y_MARK[:3, 3] + np.array([rng.uniform(-0.04, 0.04), rng.uniform(-0.04, 0.04), rng.uniform(0.19, 0.26)])
        T_bc = T(R_c, c)
        out.append(T_bc @ np.linalg.inv(X_TRUE))                # base -> flange
    return out


def angle_deg(Ra, Rb):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2))))


class TestV472(unittest.TestCase):
    def test_h1_grade(self):
        g, why = WC.grade_eih({"ok": True, "good": False, "mean_mm": 1.40, "max_mm": 2.0, "loo_mean_mm": 2.96,
                               "loo_max_mm": 5.07, "warnings": []})
        self.assertEqual(g, "usable")
        self.assertIn("5.07", why[0])
        for m, lm, lx in ((35.6, 50.0, 80.0), (67.0, 90.0, 120.0), (85.0, 99.0, 140.0), (110.0, 130.0, 170.0)):
            self.assertEqual(WC.grade_eih({"ok": True, "good": False, "mean_mm": m, "loo_mean_mm": lm,
                                           "loo_max_mm": lx})[0], "bad")
        self.assertEqual(WC.grade_eih({"ok": True, "good": True, "mean_mm": 0.5, "loo_mean_mm": 0.8,
                                       "loo_max_mm": 1.2})[0], "good")
        self.assertEqual(WC.grade_eih(None)[0], "bad")

    def test_h2_fingerprint(self):
        d = {"K": dict(K4), "dist": [0.0] * 8, "rms_px": 0.3}
        fp = WC.intr_fingerprint(d)
        self.assertTrue(WC.fingerprint_match(fp, d)[0])
        half = {"K": {k: (v / 2.0 if k != "width" and k != "height" else v // 2) for k, v in K4.items()},
                "dist": [0.0] * 8}
        self.assertTrue(WC.fingerprint_match(fp, half)[0], WC.fingerprint_match(fp, half))
        other = {"K": dict(K4, fx=K4["fx"] * 1.10, fy=K4["fy"] * 1.10), "dist": [0.0] * 8}
        self.assertFalse(WC.fingerprint_match(fp, other)[0])
        self.assertIsNone(WC.fingerprint_match(None, d)[0])

    def test_h3_consistency(self):
        calib = {"marker_in_base": [-0.42, 0.05, 0.0]}
        pts = [[-0.42 + 0.0005 * k, 0.05, 0.0] for k in range(4)]
        r = WC.check_consistency(calib, pts)
        self.assertTrue(r["ok"] and not r["moved"])
        r = WC.check_consistency(calib, [[p[0] + 0.03, p[1], p[2]] for p in pts])
        self.assertTrue(r["ok"] and r["moved"])
        self.assertAlmostEqual(r["moved_mm"], 30.75, delta=1.0)
        r = WC.check_consistency(calib, [[-0.42, 0.05, 0.0], [-0.40, 0.05, 0.0]])
        self.assertFalse(r["ok"])

    def test_h4_aruco_flip(self):
        size_mm = 23.8
        h = size_mm / 2000.0
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)
        rng = np.random.default_rng(0)
        alts_s, Bs, mirror = [], [], 0
        for A in views(14):
            B = np.linalg.inv(A @ X_TRUE) @ Y_MARK                      # marker in camera
            uv, _ = cv2.projectPoints(obj, cv2.Rodrigues(B[:3, :3])[0], B[:3, 3], KM, np.zeros(5))
            uv = uv.reshape(4, 2) + rng.normal(0, 0.6, (4, 2))        # real corner noise
            ok, rv, tv = cv2.solvePnP(obj, uv, KM, np.zeros(5), flags=cv2.SOLVEPNP_IPPE_SQUARE)   # as V3.25
            R1 = cv2.Rodrigues(rv)[0]
            s = {"q_deg": [0.0] * 7, "p_cam": [float(v) for v in tv.flatten()], "rvec": [float(v) for v in rv.flatten()],
                 "T_cam_marker": CAM.T_from(R1.tolist(), [float(v) for v in tv.flatten()]),
                 "T_base_flange": A.tolist(), "spread_mm": 0.4, "source": "pnp", "image_size": [1920, 1080]}
            c = W49.planar_pose_candidates(obj, uv, KM)
            alts_s.append(dict(s, pose_alts=[[list(a[0]), list(a[1]), a[2]] for a in c] if len(c) >= 2 else []))
            Bs.append(B)
            if len(c) >= 2:
                e = [angle_deg(np.asarray(CAM.rodrigues(a[0])), B[:3, :3]) for a in c]
                mirror += e[0] > e[1] + 6.0                              # IPPE took the clearly worse one
        self.assertGreaterEqual(mirror, 2, "the fixture must reproduce the mirror poses")
        fixed, n_fix = W49.disambiguate_planar_samples(alts_s)
        self.assertGreaterEqual(n_fix, mirror)
        for s, B in zip(fixed, Bs):                                     # the clearly better candidate is taken
            if len(s.get("pose_alts") or []) >= 2:
                e = [angle_deg(np.asarray(CAM.rodrigues(a[0])), B[:3, :3]) for a in s["pose_alts"]]
                if abs(e[0] - e[1]) > 6.0:
                    self.assertEqual(s["pose_pick"], int(np.argmin(e)), e)
        r1 = W49.calibrate_eye_in_hand(alts_s, size_mm=size_mm)
        r2 = W86.calibrate_eih_4mp(alts_s, size_mm=size_mm)
        t_err = lambda r: np.linalg.norm(np.asarray(r["cam_in_flange"]) - X_TRUE[:3, 3]) * 1000.0
        plain = [dict(x, pose_alts=[]) for x in alts_s]                 # the old way: IPPE's pick only
        r0 = W86.calibrate_eih_4mp(plain, size_mm=size_mm)
        print(f"\n[H4] 23.8 mm marker, 14 views, {mirror} mirror picks: old LOO max {r0['loo_max_mm']:.2f} mm, "
              f"used {r0['n_used']} / err {t_err(r0):.2f} mm -> new {r2['loo_max_mm']:.2f} mm, used {r2['n_used']} / "
              f"err {t_err(r2):.2f} mm")
        for r in (r1, r2):
            self.assertTrue(r["ok"])
            self.assertGreaterEqual(r["planar_flips_fixed"], mirror)
            self.assertLess(r["loo_max_mm"], 5.0, r["loo_max_mm"])
            self.assertLess(t_err(r), 3.0, t_err(r))
        self.assertLessEqual(r2["loo_max_mm"], r0["loo_max_mm"] + 0.05)

    def test_h5_board_flip(self):
        cols, rows, sq = 7, 5, 0.032
        obj_b = np.array([[x * sq, y * sq, 0.0] for y in range(1, rows) for x in range(1, cols)], float)
        rng = np.random.default_rng(11)
        samples, truth = [], []
        cx, cy = 0.5 * cols * sq, 0.5 * rows * sq
        for A in views(10, seed=5):
            B = np.linalg.inv(A @ X_TRUE) @ Y_MARK                      # board centre in camera
            o_c = obj_b - [cx, cy, 0.0]
            uv, _ = cv2.projectPoints(o_c, cv2.Rodrigues(B[:3, :3])[0], B[:3, 3], KM, np.zeros(5))
            uv = uv.reshape(-1, 2) + rng.normal(0, 0.25, (len(o_c), 2))
            samples.append({"obj_pts": obj_b.tolist(), "img_pts": uv.tolist(), "T_base_flange": A.tolist(),
                            "q_deg": [0.0] * 7})
            truth.append(B)
        ss = W49.eih_samples_from_intr_samples(samples, K4, [0.0] * 8, model_type=W49.DIST_MODELS[0])
        self.assertEqual(len(ss), len(samples))
        ss2, _n = W49.disambiguate_planar_samples(ss)
        for s, B in zip(ss2, truth):
            self.assertLess(angle_deg(np.asarray(s["T_cam_marker"])[:3, :3], B[:3, :3]), 0.5)
        r = W86.calibrate_eih_4mp(ss, size_mm=24.0)
        self.assertLess(r["loo_max_mm"], 2.0, r["loo_max_mm"])

    def test_h6_rational_not_cut(self):
        cols, rows, sq = 7, 5, 0.032
        obj_b = np.array([[x * sq, y * sq, 0.0] for y in range(1, rows) for x in range(1, cols)], float)
        A = views(1, seed=2)[0]
        B = np.linalg.inv(A @ X_TRUE) @ Y_MARK
        o_c = obj_b - [0.5 * cols * sq, 0.5 * rows * sq, 0.0]
        uv, _ = cv2.projectPoints(o_c, cv2.Rodrigues(B[:3, :3])[0], B[:3, 3], KM, np.zeros(5))
        s = {"obj_pts": obj_b.tolist(), "img_pts": uv.reshape(-1, 2).tolist(), "T_base_flange": A.tolist()}
        # (1 + 12 r^2) / (1 + 12 r^2) == 1: no distortion at all, but k1 = k4 = 12 > 8
        d = [12.0, 0.0, 0.0, 0.0, 0.0, 12.0, 0.0, 0.0]
        ss = W49.eih_samples_from_intr_samples([s], K4, d, model_type=W49.DIST_MODELS[0])
        self.assertEqual(len(ss), 1)
        self.assertLess(np.linalg.norm(np.asarray(ss[0]["p_cam"]) - B[:3, 3]) * 1000.0, 1.0)

    def test_h7_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual(v, C.VERSION)
        self.assertEqual(v, RR.VERSION)
        self.assertIn(v, ("V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3", "V4.8.4"))
        self.assertEqual(WC.MARKER, "V4.7.2 WRIST CALIBRATION CONSISTENCY")
        self.assertIn("_capture_wrist_sample", WC.PATCHES)
        self.assertTrue(W86.SELF_CAL_ALLOWED)
        self.assertTrue(hasattr(W49, "PLANNING_K_FN"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
