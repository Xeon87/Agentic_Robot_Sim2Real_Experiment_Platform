#!/usr/bin/env python3
"""NERO V4.9 MuJoCo twin tests of the general pick & place tools (needs the mujoco package; skips without it).

T1 the user's general_pick_place scene: egg_1 into cup_1, then cup_1 stacked into cup_2 (the 11-15 failure),
   driven by the test stand-in reasoner (tests/v49_reasoner.py) - success from the simulator truth.
T2 random scenes (NERO_V49_Random_Scenes, seeds 1..8): success rate >= 50 %.
Takes a few minutes; the full evaluation is tests/v49_random_eval.py.
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTS = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
try:
    import mujoco  # noqa: F401
except Exception:
    print("[SKIP] test_v4_9_twin: no mujoco")
    sys.exit(0)
os.environ["NERO_CALIB_DIR"] = tempfile.mkdtemp(prefix="nero_v49_twin_")


class TestV49Twin(unittest.TestCase):
    def test_t1_user_scene(self):
        import numpy as np
        import NERO_V489_General_Pick_Place as GP
        from v49_reasoner import GeometricReasoner
        io = GP.TwinGeneralIO()
        sk = GP.GeneralPickPlaceSkills(io)
        out = GeometricReasoner(sk).run("egg_1", "cup_1", "into")
        self.assertTrue(out["ok"], out)
        io.step(1500)
        egg, cup = io.d.body("obj_egg_1").xpos, io.d.body("obj_cup_1").xpos
        self.assertLess(float(np.hypot(*(egg[:2] - cup[:2]))), 0.040)
        self.assertLess(float(egg[2]), float(cup[2]) - 0.05)
        io2 = GP.TwinGeneralIO()
        sk2 = GP.GeneralPickPlaceSkills(io2)
        out2 = GeometricReasoner(sk2).run("cup_1", "cup_2", "into")
        self.assertTrue(out2["ok"], out2)
        io2.step(1500)
        c1, c2 = io2.d.body("obj_cup_1").xpos, io2.d.body("obj_cup_2").xpos
        self.assertLess(float(np.hypot(*(c1[:2] - c2[:2]))), 0.020)
        self.assertLess(float(c1[2]) - float(c2[2]), 0.060)              # nested (one cup height = 121 mm)
        self.assertEqual(io2.safety_trips(), 0)

    def test_t2_random(self):
        import v49_random_eval as E
        res = [E.episode(s) for s in range(1, 9)]
        for r in res:
            print(("OK " if r["success"] else "BAD"), r["seed"], r["task"], r["detail"], flush=True)
        k = sum(r["success"] for r in res)
        self.assertGreaterEqual(k, 4, res)


if __name__ == "__main__":
    unittest.main(verbosity=2)
