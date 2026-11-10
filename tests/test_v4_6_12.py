#!/usr/bin/env python3
"""NERO V4.6.12 offline tests.

R1 REAL FRAME (the user's 2x3 pulp carton, 2026-11-09 10:03, D435i 848x480 Default, depth only - no colour image is
   stored): V4.6.11 put 4 of the 6 slot centres ON or OUTSIDE the carton outline (the user saw them on the ridges
   and the egg rolled into the next cup / fell out).  Now every slot centre lies inside the outline, the egg that
   stands in a cup is within 10 mm of a slot, and that slot is the occupied one.
R2 an empty cup whose depth scatters upward (rise +11 mm, middle far below the rim top) is NOT occupied; an egg that
   sticks out of the cup is.
R3 synthetic cartons (V4.6.6 / V4.6.8 scenes) still fit within 2 mm, also with the extra mask strip.
R4 table height option A: with Try harder 15-25 mm only warns (higher table used), > 25 mm and without Try harder
   still refuse.
R5 diagnostic picture: the right panel text is drawn after the flip (V4.6.11: mirrored labels).
R6 versions / markers.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTS = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["NERO_CALIB_DIR"] = tempfile.mkdtemp(prefix="nero_v4612_calib_")
os.environ.pop("NERO_EGG_CARTON_GRID", None)

import cv2                                   # noqa: E402
import NERO_V349_Real_Wrist_Camera as W      # noqa: E402
import NERO_V365_Real_Cup_Task as RT         # noqa: E402
import v466_synth_carton as SY               # noqa: E402

DATA = os.path.join(TESTS, "data", "v4612_real_carton.npz")


def real_frame():
    z = np.load(DATA)
    H, Wd = [int(v) for v in z["shape"]]
    y0, y1, x0, x1 = [int(v) for v in z["crop"]]
    d = np.zeros((H, Wd), np.float32)
    d[y0:y1, x0:x1] = z["depth_mm"].astype(np.float32) / 1000.0
    m = np.zeros((H, Wd), np.uint8)
    cv2.fillPoly(m, [z["contour"]], 1)
    return d, m.astype(bool), json.loads(str(z["K"])), z["T"], z["contour"], z["egg_in_cup_xyz"], z["old_slots_xy"]


def inside(contour, K, T, xy, z=0.0):
    Ti = np.linalg.inv(T)
    p = Ti @ np.array([xy[0], xy[1], z, 1.0])
    u, v = K["fx"] * p[0] / p[2] + K["ppx"], K["fy"] * p[1] / p[2] + K["ppy"]
    return cv2.pointPolygonTest(contour.reshape(-1, 1, 2).astype(np.float32), (float(u), float(v)), True)


class TestV4612(unittest.TestCase):
    def test_r1_real_frame(self):
        d, m, K, T, contour, egg, old = real_frame()
        n_old_out = sum(inside(contour, K, T, xy) < 3.0 for xy in old)
        self.assertGreaterEqual(n_old_out, 3)                             # the V4.6.11 result (on / outside the edge)
        dets = [{"cls": 0, "conf": 0.97, "box": [648, 472, 870, 667], "mask": m}]
        rows, diag = W.locate_egg_holder(dets, None, d, K, T, table_z=0.0, existing_eggs=[{"xyz_m": [float(v) for v in egg], "status": "egg"}])
        slots = [r for r in rows if str(r["det_id"]).startswith("slot_")]
        self.assertEqual(len(slots), 6)
        for s in slots:
            self.assertGreater(inside(contour, K, T, s["xyz_m"][:2]), 3.0, s["det_id"])
        dist = {s["det_id"]: math.hypot(s["xyz_m"][0] - egg[0], s["xyz_m"][1] - egg[1]) * 1000.0 for s in slots}
        k_egg = min(dist, key=dist.get)
        self.assertLess(dist[k_egg], 10.0, dist)
        occ = [s["det_id"] for s in slots if s["occupied"]]
        self.assertEqual(occ, [k_egg], (occ, k_egg))
        t = diag["template"]
        self.assertGreaterEqual(t["n_cavities"], 4)
        self.assertGreater(t["depth_cov"], 0.9)

    def test_r2_occupied_rule(self):
        self.assertFalse(W.rise_says_egg({"rise_mm": 11.0, "core_med_mm": -24.1, "rim_top_mm": -3.6}))   # real empty cup
        self.assertTrue(W.rise_says_egg({"rise_mm": 18.0, "core_med_mm": 40.0, "rim_top_mm": 35.0}))
        self.assertFalse(W.rise_says_egg({"rise_mm": 4.0, "core_med_mm": 40.0, "rim_top_mm": 35.0}))
        self.assertFalse(W.rise_says_egg(None))
        self.assertTrue(W.rise_says_egg({"rise_mm": 9.0}))                # old record without the new fields

    def test_r3_synthetic_still_fit(self):
        for yaw, extra in ((20.0, 0.0), (87.0, 30.0), (-9.0, 30.0)):
            sc = SY.scene(yaw_deg=yaw, extra_mm=extra)
            rows, _d = W.locate_egg_holder(sc["dets"], sc["rgb"], sc["depth"], sc["K"], sc["T"], table_z=0.0)
            sl = [r for r in rows if str(r["det_id"]).startswith("slot_")]
            e = [min(math.hypot(s["xyz_m"][0] - tx, s["xyz_m"][1] - ty) for s in sl) * 1000.0 for tx, ty in sc["truth"]]
            self.assertLess(max(e), 2.0, (yaw, extra, e))

    def test_r4_table_option_a(self):
        obs = {"table_z_m": -0.017, "fence_z_table_m": 0.0}
        z, src, err = RT.conservative_table_z(obs)
        self.assertIsNotNone(err)                                         # unchanged without Try harder
        z, src, err = RT.conservative_table_z(obs, try_harder=True)
        self.assertIsNone(err)
        self.assertEqual(z, 0.0)                                          # the higher table
        self.assertTrue(src.startswith("WARN"))
        z, src, err = RT.conservative_table_z({"table_z_m": -0.030, "fence_z_table_m": 0.0}, try_harder=True)
        self.assertIsNotNone(err)
        z, src, err = RT.conservative_table_z({"table_z_m": -0.004, "fence_z_table_m": 0.0}, try_harder=True)
        self.assertIsNone(err)
        self.assertFalse(src.startswith("WARN"))
        s = open(os.path.join(HERE, "NERO_V365_Real_Cup_Task.py"), encoding="utf-8").read()
        self.assertIn('sk.try_harder = bool(try_harder)', s)
        self.assertIn('conservative_table_z(self.last_obs, getattr(self, "try_harder", False))', s)

    def test_r5_diag_text(self):
        s = open(os.path.join(HERE, "NERO_V349_Real_Wrist_Camera.py"), encoding="utf-8").read()
        i_flip = s.index("right = cv2.flip(right, 0)                             # plane frame y up")
        i_txt = s.index("# V4.6.12: text AFTER the flip (was mirrored)")
        self.assertLess(i_flip, i_txt)

    def test_r6_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual((C.VERSION, RR.VERSION), (v, v))
        self.assertIn(v, ("V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2"))
        self.assertEqual(W.MARKER_V4612_TEMPLATE, "V4.6.12 CUPS INSIDE THE OUTLINE + OCCUPIED NEEDS RIM TOP")
        self.assertEqual(RT.MARKER_V4612, "V4.6.12 TABLE HEIGHT 15-25 MM WARN WITH TRY HARDER")


if __name__ == "__main__":
    unittest.main(verbosity=1)
