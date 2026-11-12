#!/usr/bin/env python3
"""NERO V4.8.5 offline tests (user run 2026-11-12: egg gripped at its end -> slipped; "IK 无解" for an egg at yaw -139).

V1 the carton stands on the camera-measured table: with the fence z_table 10 mm above it (the user's office) the
   end-grasp offset is no longer inflated (V4.8.4: 23 mm, now the same as without the difference); the fingertip /
   body clearance still uses the higher table; a difference > 15 mm is not trusted.
V2 more arm postures: the egg at (-0.268, -0.242), yaw -139 deg plans (V4.8.4: GRASP_IK_NO_SOLUTION); the extra
   seeds are only tried after the usual 5; the base does not swing more than 120 deg.
V3 when no grasp direction works, the reason is counted per kind and a hint is given (V4.8.4: detail null).
V4 versions / markers.
"""
from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTS = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
import test_v4_8_4 as T484                     # noqa: E402  (temp config, the user's wrist hand-eye)

RT = T484.RT


def scene(fence_dz=0.010):
    T484.use("agx_gripper")
    sk, io = T484.T13.user_scene()
    z0 = float(sk.last_obs.get("table_z_m", 0.0))
    sk.last_obs["fence_z_table_m"] = z0 + fence_dz
    return sk, io, z0


class TestV485(unittest.TestCase):
    def tearDown(self):
        T484.TC.find_handeye = T484._find

    def test_v1_carton_on_measured_table(self):
        sk, io, z0 = scene(0.0)
        r0 = sk.plan_egg_place(source_egg="egg_1", target_slot="slot_1")
        self.assertTrue(r0["ok"], r0)
        off0 = sk.plan["egg_offset_mm"]
        sk, io, z0 = scene(0.010)
        self.assertAlmostEqual(sk._carton_base_z(z0 + 0.010), z0, places=6)
        r1 = sk.plan_egg_place(source_egg="egg_1", target_slot="slot_1")
        self.assertTrue(r1["ok"], r1)
        self.assertLessEqual(sk.plan["egg_offset_mm"], off0 + 0.5)        # not inflated (V4.8.4: 23 instead of 17)
        self.assertLessEqual(sk.plan["egg_offset_mm"], 18.0)
        self.assertIn("两者差 10.0 mm，取较高者", sk.plan["table_z_source"])        # tips: still the higher table
        self.assertAlmostEqual(float(sk.plan["table_z_m"]), z0 + 0.010, places=4)
        self.assertAlmostEqual(sk._carton_base_z(z0 + 0.020), z0 + 0.020, places=6)   # > 15 mm: not trusted

    def test_v2_more_postures(self):
        sk, io, z0 = scene(0.010)
        e = sk.cam["egg_1"]
        e["rim"] = np.array([-0.268, -0.242, z0 + 0.021])
        e["yaw_deg"] = -139.0
        r = sk.plan_egg_place(source_egg="egg_1", target_slot="slot_1")
        self.assertTrue(r["ok"], r)
        q0 = np.asarray(io.current_q(), float)
        self.assertLessEqual(abs(sk.plan["_q"]["pregrasp"][0] - q0[0]), math.radians(RT.IK_EXTRA_MAX_J1_DEG) + 1e-6)
        self.assertEqual(len(sk._extra_ik_seeds(0.3)), 16 + RT.IK_EXTRA_RANDOM_SEEDS)
        self.assertEqual(sk._extra_ik_seeds(0.3)[0][0], 0.3)

    def test_v3_reasons(self):
        sk, io, z0 = scene(0.010)
        e = sk.cam["egg_1"]
        e["rim"] = np.array([-0.75, 0.0, z0 + 0.021])                # far out of reach
        r = sk.plan_egg_place(source_egg="egg_1", target_slot="slot_1")
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "GRASP_IK_NO_SOLUTION")
        d = r["physical_verification"]["detail"]
        self.assertIn("个夹取方向", d)
        self.assertIn("种手臂起始姿态都没找到解", d)
        self.assertIn("转 90°", d)

    def test_v4_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual((v, C.VERSION, RR.VERSION), ("V4.8.5", "V4.8.5", "V4.8.5"))
        self.assertEqual(RT.CARTON_TABLE_MAX_DIFF_M, 0.015)


if __name__ == "__main__":
    unittest.main(verbosity=1)
