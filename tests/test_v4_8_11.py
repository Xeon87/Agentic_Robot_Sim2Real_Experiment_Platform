#!/usr/bin/env python3
"""V4.8.11 tests: Pure hardware geometry exposure & 3D rotated OBB->AABB perception."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestV4811PureGeometryAndAABB(unittest.TestCase):
    def test_01_version_and_no_empirical_hints_in_gripper_specs(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        import NERO_V489_General_Pick_Place as GP
        import tests.test_v4_8_10 as T10

        v = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        self.assertEqual((v, RR.VERSION), (C.VERSION, C.VERSION))
        self.assertIn(v, ("V4.8.11", "V4.9"))
        self.assertIn(GP.VERSION, ("V4.8.11", "V4.9"))

        io = T10._MockSimGeneralIO()
        sk = GP.GeneralPickPlaceSkills(io)
        obs = sk.perceive_objects(prompts=["egg", "cup"], use_finetuned=True)
        gs = obs["gripper_specs"]
        # V4.9: the gripper geometry comes from the collision model (NERO_V49_Geometry.FingerModel); no held-gap rule
        self.assertIn("max_opening_mm", gs)
        self.assertIn("fingertip_below_grasp_point_mm", gs)
        self.assertNotIn("min_held_gap_mm", gs)
        self.assertNotIn("note", gs)
        self.assertNotIn("35..45", GP.GENERAL_SYSTEM_INSTRUCTIONS)
        self.assertNotIn("-20 to -25", GP.GENERAL_SYSTEM_INSTRUCTIONS)

    def test_02_world_aabb_from_rotated_local_bounds(self):
        import NERO_V489_General_Pick_Place as GP

        # 1) Upright cup (rim at z=0.1215, identity quaternion [x,y,z,w]=[0,0,0,1])
        c_up, top_up, h_up, sz_up, _yaw, tilt_up, up_flag = GP._body_world_aabb(
            [-0.34, -0.16, 0.1215], [0.0, 0.0, 0.0, 1.0],
            (-0.0475, -0.0475, -0.121), (0.0475, 0.0475, 0.0),
            quat_wxyz=False,
        )
        self.assertTrue(up_flag)
        self.assertAlmostEqual(tilt_up, 0.0, places=1)
        self.assertAlmostEqual(top_up, 0.1215, places=3)
        self.assertAlmostEqual(h_up, 121.0, places=1)
        self.assertEqual(sz_up, [95.0, 95.0])
        self.assertAlmostEqual(c_up[2], 0.061, places=3)

        # 2) Cup lying 90 deg on its side (rotated +90 deg around Y: [x,y,z,w] = [0, sin(45), 0, cos(45)])
        s45 = math.sqrt(0.5)
        c_side, top_side, h_side, sz_side, _yaw2, tilt_side, up_flag2 = GP._body_world_aabb(
            [-0.34, -0.16, 0.0475], [0.0, s45, 0.0, s45],
            (-0.0475, -0.0475, -0.121), (0.0475, 0.0475, 0.0),
            quat_wxyz=False,
        )
        self.assertFalse(up_flag2)
        self.assertAlmostEqual(tilt_side, 90.0, places=1)
        self.assertAlmostEqual(top_side, 0.095, places=3)
        self.assertAlmostEqual(h_side, 95.0, places=1)
        # X span is now the 121 mm length, Y span remains 95 mm!
        self.assertAlmostEqual(sz_side[0], 121.0, places=1)
        self.assertAlmostEqual(sz_side[1], 95.0, places=1)
        self.assertAlmostEqual(c_side[2], 0.0475, places=3)


if __name__ == "__main__":
    unittest.main()
