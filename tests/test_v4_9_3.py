#!/usr/bin/env python3
"""NERO V4.9.3 offline tests (user GUI run V4.9.2: a runaway during the carry, then the next task could not start;
the egg rolled for ever).

S1 singularity guard: Jacobian sigma_min (position rows / 0.3 m) is small at J4 = 0 (elbow) and J6 = +-90 deg
   (wrist, J5 // J7), normal at J6 = 0; planned lines that pass sigma < SINGULAR_SIGMA_MIN are refused.
S2 safety stop = stop: a safety event during the carry ends execute_place_chunk with SAFETY_STOP, the facts say
   whether the object is still held, and no fallback motion follows (V4.9.2 fell back to move_joints).
S3 task-start recovery: STOP+RECOVER when a latch is pending or new safety stops happened, hold released, home,
   joint probe; one retry; a simulator that does not follow commands -> (False, why).
S4 egg contact condim 6 (rolling friction): a rolling egg stops within 1 s (condim 4: still rolling after 4 s).
S5 versions / markers.
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
os.environ.setdefault("NERO_CALIB_DIR", tempfile.mkdtemp(prefix="nero_v493_"))

try:
    import mujoco  # noqa: F401
    HAS_MJ = True
except Exception:
    HAS_MJ = False

D = math.radians


class FakeIO:
    """just what recover_for_task uses"""

    def __init__(self, fail_moves=0, pending=True, trips=3):
        self.log, self.cleared, self.moves, self.homes = [], 0, 0, 0
        self.fail_moves, self.pending, self.trips, self.motion_allowed = fail_moves, pending, trips, True
        self.hold = True
        self.opened = None

    def emit(self, s):
        self.log.append(s)

    def safety_trips(self):
        return self.trips

    def latch_pending(self):
        return self.pending

    def clear_latch(self, label):
        self.cleared += 1
        self.pending = False
        return True, "recovered"

    def gripper_state(self):
        return {"hold_active": self.hold, "force_mode": self.hold}

    def gripper_open(self, mm):
        self.hold, self.opened = False, mm
        return {"opening_mm": mm}

    def go_home(self, label):
        self.homes += 1
        return {"ok": True}

    def current_q(self):
        return [0.0] * 7

    def move_joints(self, label, q, slow=None):
        self.moves += 1
        if self.fail_moves:
            self.fail_moves -= 1
            return {"ok": False, "error": "TRACKING MISMATCH 90 deg"}
        return {"ok": True}


class V493(unittest.TestCase):
    def test_s3_recovery(self):
        import NERO_V489_General_Pick_Place as GP
        GP._V493_SEEN["trips"] = 0
        io = FakeIO()
        ok, why = GP.recover_for_task(io)
        self.assertTrue(ok, why)
        self.assertEqual(io.cleared, 1)
        self.assertEqual(io.opened, 60.0)
        self.assertEqual((io.homes, io.moves), (1, 2))
        ok, _ = GP.recover_for_task(io)                     # nothing new -> no STOP+RECOVER
        self.assertTrue(ok)
        self.assertEqual(io.cleared, 1)
        io2 = FakeIO(fail_moves=1, pending=False, trips=3)  # first probe fails -> STOP+RECOVER, retry passes
        ok, why = GP.recover_for_task(io2)
        self.assertTrue(ok, why)
        self.assertEqual(io2.cleared, 1)
        self.assertEqual(io2.homes, 2)
        io3 = FakeIO(fail_moves=99, pending=False, trips=3)
        ok, why = GP.recover_for_task(io3)
        self.assertFalse(ok)
        self.assertIn("probe", why)

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_s1_singularity(self):
        import NERO_V489_General_Pick_Place as GP
        sk = GP.GeneralPickPlaceSkills(GP.TwinGeneralIO())
        base = [0.0, D(40), 0.0, D(80), 0.0, D(-30), 0.0]
        self.assertGreater(sk._sing_sigma(base), GP.SINGULAR_SIGMA_MIN)
        j6z = list(base)
        j6z[5] = 0.0
        self.assertGreater(sk._sing_sigma(j6z), GP.SINGULAR_SIGMA_MIN)     # J6 = 0 is NOT singular on NERO
        elbow = list(base)
        elbow[3] = D(2)
        self.assertLess(sk._sing_sigma(elbow), GP.SINGULAR_SIGMA_MIN)
        wrist = list(base)
        wrist[5] = D(88)
        self.assertLess(sk._sing_sigma(wrist), GP.SINGULAR_SIGMA_MIN)
        ok, s, why = sk._sing_check([base, wrist])
        self.assertFalse(ok)
        self.assertIn("wrist J6", why)
        ok, s, why = sk._sing_check([base, j6z])
        self.assertTrue(ok)

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_s2_safety_stop(self):
        import NERO_V489_General_Pick_Place as GP
        io = GP.TwinGeneralIO()
        sk = GP.GeneralPickPlaceSkills(io)
        sk.run_tool("perceive_objects", {"prompts": ["egg", "cup"], "use_finetuned": True})
        r = sk.run_tool("plan_grasp_chunk", {"object_id": "egg_1", "xy_offset_mm": [0, 0],
                                             "grasp_z_offset_from_top_mm": -9, "closing_yaw_deg": 90, "pitch_deg": 0,
                                             "open_mm": 60, "yaw_tolerance_deg": 30, "pregrasp_height_mm": 80,
                                             "lift_height_mm": 150})
        self.assertTrue(r["ok"], r)
        r = sk.run_tool("execute_grasp_chunk", {"clamp_force_n": 8})
        self.assertTrue(r["ok"], r)
        r = sk.run_tool("plan_place_chunk", {"target_id_or_xyz": "cup_1", "xy_offset_mm": [0, 0],
                                             "held_bottom_above_target_top_mm": 15, "yaw_change_deg": 0,
                                             "approach_height_mm": 40, "release_open_mm": 60, "carry_lift_mm": 0})
        self.assertTrue(r["ok"], r)
        calls = []
        io.move_line = lambda *a, **k: (calls.append("line"), {"ok": False, "safety": True,
                                                               "error": "RUNAWAY joint5 6.3 rad/s"})[1]
        io.move_joints = lambda *a, **k: (calls.append("joints"), {"ok": True})[1]
        r = sk.run_tool("execute_place_chunk", {"retreat_up_mm": 60})
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "SAFETY_STOP")
        self.assertEqual(calls, ["line"])                               # no fallback motion
        txt = str(r)
        self.assertIn("object_still_held", txt)
        self.assertIn("no further motion", txt)

    def test_s4_egg_condim_xml(self):
        import NERO_V343_Scenes as SC
        import NERO_V49_Random_Scenes as RS
        self.assertIn('condim="6"', SC.general_egg_body_xml("obj_egg_1", (0.0, 0.0)))
        self.assertIn('condim="6"', RS._egg("obj_egg_9", (0.0, 0.0), {}, "1 1 1 1", 0.0))

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_s4_egg_stops_rolling(self):
        import NERO_V489_General_Pick_Place as GP
        io = GP.TwinGeneralIO()
        m, d = io.m, io.d
        b = d.body("obj_egg_1")
        va = int(m.jnt_dofadr[m.body_jntadr[b.id]])
        for _ in range(200):
            mujoco.mj_step(m, d)
        d.qvel[va:va + 6] = [0.3, 0, 0, 0, 0, 0]
        for _ in range(int(1.0 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        self.assertLess(float(np.linalg.norm(d.qvel[va:va + 3])), 0.01)

    def test_s5_versions(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        import NERO_V489_General_Pick_Place as GP
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual((v, RR.VERSION, GP.VERSION), (C.VERSION, C.VERSION, C.VERSION))
        self.assertIn(v, ("V4.9.3", "V4.10", "V4.10.1", "V4.10.2", "V4.10.3", "V4.11"))
        self.assertEqual(GP.MARKER_V493, "V4.9.3 TASK-START RECOVERY + SAFETY STOP + SINGULARITY + EGG ROLL")


if __name__ == "__main__":
    unittest.main(verbosity=1)
