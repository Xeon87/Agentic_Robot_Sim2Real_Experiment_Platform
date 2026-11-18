#!/usr/bin/env python3
"""NERO V4.10.3 offline tests (user 11-17: 「先做前三项（不动后端）」).

S1 soft weld: the general / household hold welds use solref 0.012 1, torquescale 1 (XML, the backend hook
   SC.wrap_activate_hold and the twin); a centred held object 0.12 m under link7 (weld only, no pads) moves < 1 mm /
   < 1 deg under gravity.  Measured in the twin (stand-in reasoner, kinematic arm - arm impulses are not modelled):
   0.02 / 0.3 broke the end-gripped remote -> mug insert (test_v4_10_1 P3), 0.02 / 1.0 raised the mug's finger shocks,
   0.012 / 1.0 kept every test, random seeds 1-20 the same 8/20 and finger-joint shock steps 759 -> 682.
S2 short box walls: no box of any household object (fixed H1-H5, random seeds 1-40) is thin and >= 45 mm long,
   so the backend's frozen V2-44 R2 basket-rim rule cannot cut it to 14 mm wherever the object stands.
S3 auto reset: > 5 safety stops in one tool call -> the objects are reset once at the next task start (arm at
   home); <= 5 -> no reset; a simulator without reset_objects only logs.
S4 versions / marker; backend untouched by this release (same hash as V4.10.2).
"""
from __future__ import annotations

import hashlib
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
os.environ.setdefault("NERO_CALIB_DIR", tempfile.mkdtemp(prefix="nero_v4103_"))

try:
    import mujoco  # noqa: F401
    HAS_MJ = True
except Exception:
    HAS_MJ = False

from test_v4_10_2 import _Env, _xml  # noqa: E402


class V4103(unittest.TestCase):
    def test_s1_weld_params_xml(self):
        import NERO_V343_Scenes as SC
        self.assertEqual(SC.weld_params(), (0.012, 1.0, 1.0))
        with _Env("household_pick_place", "fixed:H1"):
            with open(_xml(tempfile.mkdtemp())) as fh:
                txt = fh.read()
        self.assertIn('<weld name="hold_obj_apple_1" body1="link7" body2="obj_apple_1" anchor="0 0 0" active="false" '
                      'solref="0.012 1" torquescale="1"', txt)
        with _Env("sort_items"):
            self.assertFalse(SC.rigid_hold())

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_s1_soft_weld_holds(self):
        import mujoco
        import NERO_V343_Scenes as SC
        with _Env("household_pick_place", "fixed:H1"):
            m = mujoco.MjModel.from_xml_path(_xml(tempfile.mkdtemp()))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        l7, obj = m.body("link7").id, m.body("obj_remote_1").id
        eq = m.equality("hold_obj_remote_1").id
        j = m.jnt_qposadr[m.body_jntadr[obj]]
        d.qpos[j:j + 3] = d.xpos[l7] + np.array([0.0, 0.0, -0.12])       # centred under the grasp point
        mujoco.mj_forward(m, d)
        act = SC.wrap_activate_hold(lambda b: True, m, d, {obj: eq}, l7, lambda: None)
        self.assertTrue(act(obj))
        self.assertAlmostEqual(float(m.eq_data[eq, 10]), 1.0)
        self.assertAlmostEqual(float(m.eq_solref[eq, 0]), 0.012)
        rel0 = d.xpos[obj] - d.xpos[l7]
        R0 = d.xmat[obj].reshape(3, 3).copy()
        ja, dj = m.jnt_qposadr[m.joint("j").id], m.jnt_dofadr[m.joint("j").id]
        for _ in range(3000):                                           # link7 fixed in space, gravity on
            d.qpos[ja:ja + 4] = [1.0, 0.0, 0.0, 0.0]
            d.qvel[dj:dj + 3] = 0.0
            mujoco.mj_step(m, d)
        sag = float(np.linalg.norm((d.xpos[obj] - d.xpos[l7]) - rel0))
        ang = float(np.degrees(np.arccos(np.clip((np.trace(R0.T @ d.xmat[obj].reshape(3, 3)) - 1) / 2, -1, 1))))
        self.assertLess(sag, 0.001, sag)
        self.assertLess(ang, 1.0, ang)

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_s1_twin_soft_weld(self):
        import NERO_V410_Household_Scenes as HS
        io = HS.make_twin(HS.fixed_scene("H1"))
        self.assertEqual(io.hold_kind, "rigid")
        bid = io.m.body("obj_apple_1").id
        io._activate_hold(bid)
        eq = io.m.equality("hold_obj_apple_1").id
        self.assertEqual(int(io.d.eq_active[eq]), 1)
        self.assertAlmostEqual(float(io.m.eq_data[eq, 10]), 1.0)
        self.assertAlmostEqual(float(io.m.eq_solref[eq, 0]), 0.012)

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_s2_no_long_thin_boxes(self):
        import mujoco
        import NERO_V410_Household_Scenes as HS
        scenes = [HS.fixed_scene(s) for s in HS.scene_ids()] + [HS.random_scene(k) for k in range(1, 41)]
        n_box = 0
        for sc in scenes:
            m = mujoco.MjModel.from_xml_string(f'<mujoco><asset>{sc.assets(visual=False)}</asset><worldbody>'
                                               f'{sc.world_xml(visual=False)}</worldbody></mujoco>')
            for g in range(m.ngeom):
                if int(m.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                    continue
                name = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
                if any(k in name for k in ("floor", "bottom", "base_plate")):
                    continue
                sx, sy, sz = (float(v) for v in m.geom_size[g])
                n_box += 1
                thin_long = min(sx, sy) <= 0.020 and max(sx, sy) >= 0.045 and 2 * sz > 0.0145
                self.assertFalse(thin_long, f"{getattr(sc, 'sid', sc.seed)}: {name} {sx:.3f} {sy:.3f} {sz:.3f}")
        self.assertGreater(n_box, 50)

    def test_s3_auto_reset(self):
        import NERO_V489_General_Pick_Place as GP

        class FakeIO:
            motion_allowed = False

            def __init__(self, has_reset=True):
                self.trips, self.msgs, self.resets = 0, [], 0
                if has_reset:
                    self.reset_objects = self._reset

            def _reset(self):
                self.resets += 1
                return True, "[V4.10 RESET] 5 objects back to their start poses"

            def safety_trips(self):
                return self.trips

            def emit(self, s):
                self.msgs.append(s)

            def gripper_state(self):
                return {}

            def go_home(self, label):
                return {"ok": True}

        def tool_with_trips(io, n):
            sk = GP.GeneralPickPlaceSkills.__new__(GP.GeneralPickPlaceSkills)
            sk.io = io

            def fake(**k):
                io.trips += n
                return {"ok": False}
            sk.perceive_objects = fake
            for nm in ("preview_grasp", "plan_grasp_chunk", "execute_grasp_chunk", "plan_place_chunk",
                       "execute_place_chunk", "verify_and_park_chunk"):
                setattr(sk, nm, fake)
            return sk.run_tool("perceive_objects", {})

        GP._V4103_BLOWUP.update(pending=False, why="")
        io = FakeIO()
        tool_with_trips(io, 5)
        self.assertFalse(GP._V4103_BLOWUP["pending"])
        self.assertTrue(GP.recover_for_task(io, probe=False)[0])
        self.assertEqual(io.resets, 0)
        r = tool_with_trips(io, 6)
        self.assertEqual(r["physical_verification"]["safety_events"]["count"], 6)
        self.assertTrue(GP._V4103_BLOWUP["pending"])
        self.assertTrue(GP.recover_for_task(io, probe=False)[0])
        self.assertEqual(io.resets, 1)
        self.assertFalse(GP._V4103_BLOWUP["pending"])
        self.assertTrue(any("V4.10.3 AUTO RESET" in s for s in io.msgs))
        self.assertTrue(GP.recover_for_task(io, probe=False)[0])
        self.assertEqual(io.resets, 1)                                   # once only
        io2 = FakeIO(has_reset=False)
        tool_with_trips(io2, 9)
        self.assertTrue(GP.recover_for_task(io2, probe=False)[0])
        self.assertTrue(any("not available" in s for s in io2.msgs))
        self.assertTrue(hasattr(GP.SimGeneralIO, "reset_objects"))

    def test_s4_versions(self):
        import NERO_V489_General_Pick_Place as GP
        self.assertIn(GP.VERSION, ("V4.10.3", "V4.11"))
        self.assertIn("SOFT WELD", GP.MARKER_V4103)
        self.assertIn(open(os.path.join(HERE, "VERSION")).read().strip(), ("V4.10.3", "V4.11"))
        h = hashlib.sha256(open(os.path.join(HERE, "NERO_V2_7_1_Unified_Backend.py"), "rb").read()).hexdigest()
        self.assertEqual(h, "1a1da2c8c8a9836e335b42444f95b112b0839be9277af62871ba9f19cd24f2fa")


if __name__ == "__main__":
    unittest.main(verbosity=2)
