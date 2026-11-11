#!/usr/bin/env python3
"""NERO V4.6.13 offline tests: the user's printed adapter (SO-101 <-> NERO ring + wrist camera cradle + ArUco plate)
in the place planner's table-clearance check.

A1 the STL (assets/adapter/SO101_NERO_adapter_v5_PRINT.stl) splits into ring / ArUco plate / camera cradle with the
   measured extents; the turn about the tool axis follows the wrist hand-eye camera centre.
A2 clear_of_table: tool axis horizontal with the cradle hanging down -> refused, turned over -> fine.
A3 the user's place (slot_6 of the 2026-11-09 15:18 run): the +85 deg insertion the real arm used puts the camera
   cradle below the table (that is the reported contact); the planner now takes -85 deg (cradle up, lowest part the
   ring >= 8 mm above the table) and logs why the other insertion was refused.
A4 only +pitch allowed -> no candidate, the reason "相机 / 支架离桌面太近" is logged (the arm is never sent there).
A5 config: enabled=false switches the check off; clearance from NERO_TOOL_CLEAR_MM.
A6 versions / markers.
"""
from __future__ import annotations

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
os.environ["NERO_CALIB_DIR"] = tempfile.mkdtemp(prefix="nero_v4613_calib_")

import NERO_V4613_Tool_Collision as TC      # noqa: E402
import NERO_V365_Real_Cup_Task as RT        # noqa: E402
import test_v4_6_8 as T8                    # noqa: E402

USER_SLOTS = [(-0.315, -0.011), (-0.359, -0.000), (-0.404, 0.011), (-0.326, -0.057), (-0.371, -0.046), (-0.415, -0.035)]


def user_scene():
    T8.SLOTS = list(USER_SLOTS)
    sc = T8.Scene(eggs=[[-0.301, 0.183, 0.021], [-0.392, 0.132, 0.021]])
    sk, io = T8.make(sc)
    sk.observe_scene()
    return sk, io


class TestV4613(unittest.TestCase):
    def test_a1_stl_and_mount(self):
        parts = TC.stl_parts(TC.load_stl())
        self.assertEqual(set(parts), {"ring", "aruco_plate", "camera_cradle"})
        lo, hi = parts["camera_cradle"]
        self.assertAlmostEqual(float(hi[1]), 66.6, delta=0.5)
        self.assertAlmostEqual(float(hi[2]), 32.2, delta=0.5)
        lo, hi = parts["aruco_plate"]
        self.assertAlmostEqual(float(hi[0]), 73.0, delta=0.5)
        T = np.eye(4)
        T[:3, 3] = [0.1031, 0.0086, -0.0502]                      # the user's wrist hand-eye (log 2026-11-09)
        m = TC.ToolModel(T_flange_cam=T)
        self.assertTrue(m.ok)
        self.assertAlmostEqual(m.roll_deg, TC.auto_roll_deg(T[:3, 3], m.axis_z), places=6)
        _n, c, R, h = [b for b in m.boxes if b[0] == "camera_cradle"][0]
        d_c = np.array([c[1], c[2] - m.axis_z])
        d_cam = np.array([T[1, 3], T[2, 3] - m.axis_z])
        ang = math.degrees(math.acos(float(d_c @ d_cam / np.linalg.norm(d_c) / np.linalg.norm(d_cam))))
        self.assertLess(ang, 3.0)                                 # the cradle faces the calibrated camera
        self.assertIn("camera_head", [b[0] for b in m.boxes])
        self.assertIn("腕部手眼标定", m.note)

    def test_a2_clearance(self):
        m = TC.ToolModel(cfg={"roll_deg": 180.0})                 # cradle towards link7 -z
        T = np.eye(4)                                             # link7 x horizontal, link7 z up
        T[:3, 3] = [-0.4, 0.0, 0.06]
        ok, dz, part = m.clear_of_table(T, 0.0)
        self.assertFalse(ok)
        self.assertIn(part, ("camera_cradle", "aruco_plate"))
        T2 = T.copy()
        T2[:3, :3] = TC._rx(math.pi)                              # turned over about the tool axis
        T2[2, 3] = 0.06 + 2 * 0.0235
        ok2, dz2, part2 = m.clear_of_table(T2, 0.0)
        self.assertGreater(dz2, dz + 40.0)

    def test_a3_user_place(self):
        sk, io = user_scene()
        r = sk.plan_egg_place(source_egg="egg_1", target_slot="slot_6")
        self.assertTrue(r["ok"], io.logs[-3:])
        self.assertLess(sk.plan["place_pitch_deg"], 0.0)
        zt = float(sk.plan.get("table_z_m", 0.0))
        tm = sk._tool_model()
        Ti = np.linalg.inv(sk.CK.T_L7_TCP)
        ok, dz, part = tm.clear_of_table(sk._fast.fk(np.asarray(sk.plan["_q"]["release"], float)) @ Ti, zt)
        self.assertTrue(ok and dz >= 8.0, (dz, part))
        # what the real arm did: +85 deg -> the cradle under the table
        R_g = sk._slot_vertical_grasp_R(180.0, 85.0)
        p_a = RT.egg_place_grasp_origin(np.asarray(sk.plan["slot_tip_target"], float), R_g,
                                        np.asarray(sk.plan["egg_in_G_m"]), 0.0275)
        p_rel, R_rel = sk.prof.tcp_from_grasp(p_a, R_g)
        sol = None
        for sd in ([0.2, 0.55, -0.35, 1.25, -0.35, 0.30, 0.55], list(sk.plan["_q"]["release"]),
                   [math.pi, 0.4, 0.0, 1.2, 0.0, 0.3, 0.0]):
            sol = sk._fast.ik([sd], p_rel, R_rel, iters=80)
            if sol and sol[0] is not None:
                break
        self.assertTrue(sol and sol[0] is not None)
        ok2, dz2, part2 = tm.clear_of_table(sk._fast.fk(np.asarray(sol[0])) @ Ti, zt)
        self.assertFalse(ok2)
        self.assertEqual(part2, "camera_cradle")
        self.assertLess(dz2, 0.0)
        self.assertTrue(any("[V4.6.13 TOOL]" in s for s in io.logs))
        self.assertTrue(any("[V4.6.13 PLACE] slot_6：用俯仰 -" in s for s in io.logs), [s for s in io.logs if "V4.6.13" in s])

    def test_a4_only_plus(self):
        sk, io = user_scene()
        orig = RT.EGG_PITCH_ORDER
        RT.EGG_PITCH_ORDER = (85.0,)
        try:
            r = sk.plan_egg_place(source_egg="egg_1", target_slot="slot_6")
        finally:
            RT.EGG_PITCH_ORDER = orig
        bad = [s for s in io.logs if "[V4.6.13 PLACE]" in s and "两种插法都不行" in s]
        self.assertFalse(r["ok"])
        self.assertTrue(bad, [s for s in io.logs if "V4.6.13" in s])

    def test_a5_config(self):
        m = TC.ToolModel(cfg={"enabled": False})
        self.assertFalse(m.ok)
        os.environ["NERO_TOOL_CLEAR_MM"] = "15"
        try:
            self.assertEqual(TC.load_cfg()["clearance_mm"], 15.0)
        finally:
            os.environ.pop("NERO_TOOL_CLEAR_MM")
        self.assertEqual(TC.load_cfg()["clearance_mm"], 8.0)

    def test_a6_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual((C.VERSION, RR.VERSION), (v, v))
        self.assertIn(v, ("V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2"))
        self.assertEqual(TC.MARKER, "V4.6.13 ADAPTER + WRIST CAMERA COLLISION")
        self.assertEqual(RT.MARKER_V4613, "V4.6.13 ADAPTER + WRIST CAMERA TABLE CLEARANCE IN PLANNING")
        self.assertTrue(os.path.isfile(TC.STL_PATH))


if __name__ == "__main__":
    unittest.main(verbosity=1)
