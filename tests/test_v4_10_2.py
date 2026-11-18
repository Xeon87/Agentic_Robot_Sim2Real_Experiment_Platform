#!/usr/bin/env python3
"""NERO V4.10.2 offline tests (user 11-17: 「球铰改成刚性」 - the GUI simulation holds rigidly in the general scenes).

R1 scene module: general_pick_place / household_pick_place write WELD hold equalities, every other scene keeps
   CONNECT; the general pick & place IO reports hold_kind "rigid" there (held_object_turns_with_gripper).
R2 (mujoco) the backend hook (SC.wrap_activate_hold, the same call the backend's new A2 block makes) on the
   household MJCF: the held object keeps its pose in link7 while link7 turns 90 deg (weld), with no jump at
   activation; a CONNECT equality still goes to the original function.
R3 backend: exactly one new A2 block (12), right after activate_connect_hold, outside the frozen loop (B section
   unchanged, test_v3_44); versions / marker.
"""
from __future__ import annotations

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
os.environ.setdefault("NERO_CALIB_DIR", tempfile.mkdtemp(prefix="nero_v4102_"))

try:
    import mujoco  # noqa: F401
    HAS_MJ = True
except Exception:
    HAS_MJ = False

KEYS = ("NERO_SCENE", "NERO_V343_ACTIVE_SCENE", "NERO_V410_HOUSEHOLD")


class _Env:
    def __init__(self, scene, spec=None):
        self.scene, self.spec = scene, spec

    def __enter__(self):
        self.old = {k: os.environ.get(k) for k in KEYS}
        os.environ["NERO_SCENE"] = os.environ["NERO_V343_ACTIVE_SCENE"] = self.scene
        if self.spec:
            os.environ["NERO_V410_HOUSEHOLD"] = self.spec

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _xml(td):
    import NERO_V343_Scenes as SC
    src = os.path.join(td, "nero_motor.xml")
    with open(src, "w") as f:
        f.write("<mujoco model='t'><compiler angle='radian'/><asset></asset><worldbody><body name='link7' "
                "pos='-0.40 0.10 0.30'><joint name='j' type='ball'/><geom type='sphere' size='0.01' contype='0' "
                "conaffinity='0' mass='1'/></body></worldbody></mujoco>")
    return SC.make_store_in_safe_xml(src)


class V4102(unittest.TestCase):
    def test_r1_scene_holds(self):
        import NERO_V343_Scenes as SC
        import NERO_V489_General_Pick_Place as GP
        for scene, rigid in (("general_pick_place", True), ("household_pick_place", True), ("sort_items", False),
                             ("stack_cups", False), ("tidy_desk", False), ("fill_egg_holder", False)):
            with _Env(scene, "fixed:H1"):
                self.assertEqual(SC.rigid_hold(), rigid, scene)
                if scene in ("general_pick_place", "household_pick_place"):
                    io = GP.SimGeneralIO.__new__(GP.SimGeneralIO)
                    self.assertEqual(io.hold_kind, "rigid")
        with _Env("household_pick_place", "fixed:H1"):
            txt = open(_xml(tempfile.mkdtemp())).read()
            self.assertIn('<weld name="hold_obj_apple_1"', txt)
            self.assertNotIn('<connect name="hold_', txt)

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_r2_backend_hook(self):
        import mujoco
        import NERO_V343_Scenes as SC
        with _Env("household_pick_place", "fixed:H1"):
            m = mujoco.MjModel.from_xml_path(_xml(tempfile.mkdtemp()))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        l7 = m.body("link7").id
        obj = m.body("obj_remote_1").id
        eq = {obj: m.equality("hold_obj_remote_1").id}
        calls = []
        act = SC.wrap_activate_hold(lambda b: calls.append(b) or True, m, d, eq, l7, lambda: None)
        # put the remote 0.12 m under link7 first (as if gripped), then hold
        j = m.jnt_qposadr[m.body_jntadr[obj]]
        d.qpos[j:j + 3] = d.xpos[l7] + np.array([0.0, 0.0, -0.12])
        mujoco.mj_forward(m, d)
        rel0 = d.xpos[obj] - d.xpos[l7]
        Rrel0 = d.xmat[l7].reshape(3, 3).T @ d.xmat[obj].reshape(3, 3)
        self.assertTrue(act(obj))
        self.assertEqual(calls, [])
        m.opt.gravity[:] = 0.0
        ja = m.jnt_qposadr[m.joint("j").id]
        dj = m.jnt_dofadr[m.joint("j").id]
        for k in range(5000):                          # turn link7 by 90 deg about x in 3 s, then hold 2 s
            a = np.radians(90.0) * min(1.0, k / 3000.0)
            d.qpos[ja:ja + 4] = [np.cos(a / 2), np.sin(a / 2), 0.0, 0.0]
            d.qvel[dj:dj + 3] = [np.radians(90.0) / 3.0 if k < 3000 else 0.0, 0.0, 0.0]
            mujoco.mj_step(m, d)
        R7 = d.xmat[l7].reshape(3, 3)
        rel1 = R7.T @ (d.xpos[obj] - d.xpos[l7])
        self.assertLess(float(np.linalg.norm(rel1 - rel0)), 0.01)              # still 0.12 m under link7 (local)
        Ro = d.xmat[obj].reshape(3, 3)
        self.assertLess(float(np.linalg.norm(R7.T @ Ro - Rrel0)), 0.03)         # turned with link7 (same pose in it)
        self.assertGreater(abs(float(Ro[2, 0])) + abs(float(Ro[2, 1])), 0.5)   # its long axis is no longer level
        m2 = mujoco.MjModel.from_xml_string("<mujoco><worldbody><body name='a'><freejoint/><geom size='.01'/></body>"
                                            "<body name='b'><freejoint/><geom size='.01'/></body></worldbody><equality>"
                                            "<connect name='c' body1='a' body2='b' anchor='0 0 0'/></equality></mujoco>")
        d2 = mujoco.MjData(m2)
        act2 = SC.wrap_activate_hold(lambda b: calls.append(b) or True, m2, d2, {2: 0}, 1, lambda: None)
        act2(2)
        self.assertEqual(calls, [2])                   # CONNECT -> the backend's own function

    def test_r3_backend_and_versions(self):
        import NERO_V312_Console as C
        import NERO_V489_General_Pick_Place as GP
        with open(os.path.join(HERE, "NERO_V2_7_1_Unified_Backend.py")) as f:
            B = f.read()
        self.assertEqual(B.count("# V3.43 A2 BEGIN"), 12)
        i = B.index("_V343.wrap_activate_hold(activate_connect_hold")
        self.assertLess(B.index("def activate_connect_hold"), i)
        self.assertLess(i, B.index("with mujoco.viewer.launch_passive(model, data) as viewer:"))
        with open(os.path.join(HERE, "VERSION")) as f:
            self.assertEqual(f.read().strip(), C.VERSION)
        self.assertIn(GP.VERSION, ("V4.10.2", "V4.10.3", "V4.11", "V4.11.1", "V4.11.2"))
        self.assertEqual(GP.MARKER_V4102, "V4.10.2 RIGID WELD HOLD IN THE GUI SIMULATION")


if __name__ == "__main__":
    unittest.main(verbosity=1)
