#!/usr/bin/env python3
"""NERO V4.9.1 offline tests (user run 2026-11-16, gpt-6.1-sol / high: cup_1 stacked into cup_2 with no hints, but
33 tool calls: 11 grasp plans rejected for "no arm posture", 14 place plans rejected, and one plan that passed
was refused at execution: "V4.9 carry lift: J7 only 3.5 deg from its limit").

R1 dense line check: every straight line the IO executes (carry lift, lowering, descent, lift) is checked at plan
   time point by point (4 mm / 2 deg, IK continuity, every joint >= 5 deg); a carry lift that fails it is
   PLACE_PLAN_FAILED with the reason (V4.9 accepted it and the execution refused it).
R2 grasp reachability: a failed grasp plan carries the same grasp turned about the object's vertical axis in 30 deg
   steps (offset and closing direction together) with the status of each; probing changes no plan / stage / log.
R3 place reachability: a failed place plan carries the same request with other yaw_change_deg, carry_lift_mm and
   held_bottom_above_target_top_mm values.
R4 (mujoco) the user's scene: the 11-16 first grasp (z +8 above the rim, lift 160) fails and the map shows turns
   that plan with a lower pregrasp / lift.
R5 versions / markers.
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
os.environ.setdefault("NERO_CALIB_DIR", tempfile.mkdtemp(prefix="nero_v491_"))

try:
    import mujoco  # noqa: F401
    HAS_MJ = True
except Exception:
    HAS_MJ = False


def mock():
    import NERO_V489_General_Pick_Place as GP
    import test_v4_8_10 as T10
    io = T10._MockSimGeneralIO()
    sk = GP.GeneralPickPlaceSkills(io)
    sk.perceive_objects(prompts=["egg", "cup"])
    return GP, io, sk


def hold_egg(sk):
    r = sk.plan_grasp_chunk(object_id="egg_1", xy_offset_mm=[0, 0], grasp_z_offset_from_top_mm=-5.0, closing_yaw_deg=90.0,
                            pitch_deg=0.0, open_mm=60.0, yaw_tolerance_deg=40.0, pregrasp_height_mm=70.0,
                            lift_height_mm=120.0)
    assert r["ok"], r
    assert sk.execute_grasp_chunk(clamp_force_n=4.2)["ok"]


class TestV491(unittest.TestCase):
    def test_r1_dense_line(self):
        import NERO_V362_Cup_Skills as CK
        GP, io, sk = mock()
        q0 = [0.3, 0.4, 0.0, 1.2, 0.0, 0.4, 0.2]
        T = sk._fast.fk(np.asarray(q0, float))
        qs, chk = sk._dense_line(q0, T[:3, 3] + np.array([0, 0, 0.05]), T[:3, :3])
        self.assertIsNotNone(qs)
        self.assertGreaterEqual(len(qs), 13)                        # 50 mm / 4 mm
        self.assertAlmostEqual(chk["min_margin_deg"], CK.check_joint_path(qs, jump_deg=8.0)["min_margin_deg"], places=1)
        self.assertEqual(GP.DENSE_LINE_MARGIN_DEG, 5.0)
        hold_egg(sk)
        sk._dense_line = lambda *a, **k: (None, {"ok": False, "why": "J7 only 3.5 deg from its limit"})
        r = sk.plan_place_chunk(target_id_or_xyz="cup_1", xy_offset_mm=[0, 0], held_bottom_above_target_top_mm=30.0,
                                yaw_change_deg=0.0, approach_height_mm=60.0, release_open_mm=60.0, carry_lift_mm=30.0)
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "PLACE_PLAN_FAILED")
        self.assertIn("J7 only 3.5 deg", " ".join(r["physical_verification"]["problems"]))
        self.assertIsNone(sk.place_plan)

    def test_r2_grasp_map(self):
        GP, io, sk = mock()
        n_log = len(io.logs)
        r = sk.plan_grasp_chunk(object_id="egg_1", xy_offset_mm=[0, 3.0], grasp_z_offset_from_top_mm=-5.0,
                                closing_yaw_deg=90.0, pitch_deg=0.0, open_mm=60.0, yaw_tolerance_deg=0.0,
                                pregrasp_height_mm=70.0, lift_height_mm=900.0)          # unreachable lift
        self.assertFalse(r["ok"])
        rows = r["physical_verification"]["reachability"]["around_object"]
        self.assertEqual(len(rows), 12)
        for row in rows:
            if "xy_offset_mm" in row:
                self.assertAlmostEqual(math.hypot(*row["xy_offset_mm"]), 3.0, delta=0.2)
        self.assertEqual([row["turn_deg"] for row in rows], list(range(0, 360, 30)))
        self.assertTrue(any("plans with" in row["status"] for row in rows), rows)
        self.assertIsNone(sk.grasp_plan)
        self.assertEqual(sk.stage, "OBSERVED")
        self.assertEqual(len(io.logs) - n_log, 1)                    # only the plan_grasp_chunk result line

    def test_r3_place_map(self):
        GP, io, sk = mock()
        hold_egg(sk)
        r = sk.plan_place_chunk(target_id_or_xyz="cup_1", xy_offset_mm=[0, 0], held_bottom_above_target_top_mm=-60.0,
                                yaw_change_deg=0.0, approach_height_mm=60.0, release_open_mm=60.0, carry_lift_mm=0.0)
        self.assertFalse(r["ok"])
        m = r["physical_verification"]["reachability"]
        self.assertEqual(len(m["yaw_change_deg"]), 12)
        self.assertEqual(len(m["carry_lift_mm"]), 6)
        self.assertGreaterEqual(len(m["held_bottom_above_target_top_mm"]), 5)
        self.assertTrue(any(v.startswith("plans") for v in m["held_bottom_above_target_top_mm"].values()), m)
        self.assertIsNone(sk.place_plan)
        self.assertEqual(sk.stage, "HOLDING")

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_r4_user_scene_map(self):
        import NERO_V489_General_Pick_Place as GP
        io = GP.TwinGeneralIO()
        sk = GP.GeneralPickPlaceSkills(io)
        sk.run_tool("perceive_objects", {"prompts": ["cup"], "use_finetuned": True})
        r = sk.run_tool("plan_grasp_chunk", dict(object_id="cup_1", xy_offset_mm=[45.7, 0], grasp_z_offset_from_top_mm=8,
                                                 closing_yaw_deg=0, pitch_deg=0, open_mm=24, yaw_tolerance_deg=15,
                                                 pregrasp_height_mm=60, lift_height_mm=150))
        self.assertFalse(r["ok"])
        rows = r["physical_verification"]["reachability"]["around_object"]
        self.assertTrue(any("plans" in row["status"] for row in rows), rows)

    def test_r5_versions(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        import NERO_V489_General_Pick_Place as GP
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual((v, RR.VERSION, GP.VERSION), (C.VERSION, C.VERSION, C.VERSION))
        self.assertIn(v, ("V4.9.1", "V4.9.2"))
        self.assertEqual(GP.MARKER_V491, "V4.9.1 DENSE LINES + REACHABILITY")


if __name__ == "__main__":
    unittest.main(verbosity=1)
