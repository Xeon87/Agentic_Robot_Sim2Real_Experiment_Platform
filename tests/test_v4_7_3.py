#!/usr/bin/env python3
"""NERO V4.7.3 offline tests (the GUI parts are in tests/test_v4_7_3_gui.py).

R1 ready_to_work: the built-in pose is the user's 2026-11-09 one; the shipped config/real_poses.json holds it; a
   pose library entry "ready_to_work" wins; a broken library file falls back to the built-in pose.
R2 versions / markers.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
TMP = tempfile.mkdtemp(prefix="nero_v473_")
os.environ["NERO_CALIB_DIR"] = os.path.join(TMP, "calib")

import NERO_V4_Console_Tab as VT                # noqa: E402
import NERO_V314_Joint_Control as JC            # noqa: E402

USER_POSE = [-0.2, 21.6, -0.1, 61.2, 0.1, 0.2, 44.8]


class TestV473(unittest.TestCase):
    def test_r1_ready_to_work(self):
        self.assertEqual(VT.READY_TO_WORK_DEFAULT_DEG, USER_POSE)
        self.assertEqual(VT.READY_TO_WORK_SPEED, 0.10)
        q, src = VT.ready_to_work_deg(os.path.join(HERE, "config", "real_poses.json"))
        self.assertEqual(q, USER_POSE)
        self.assertIn("姿态库", src)
        p = os.path.join(TMP, "poses.json")
        q, src = VT.ready_to_work_deg(p)
        self.assertEqual((q, "内置" in src), (USER_POSE, True))
        JC.PoseLibrary(p).save("ready_to_work", [0, 10, 0, 50, 0, 0, 30])
        self.assertEqual(VT.ready_to_work_deg(p)[0], [0.0, 10.0, 0.0, 50.0, 0.0, 0.0, 30.0])
        with open(p, "w") as f:
            f.write("{broken")
        self.assertEqual(VT.ready_to_work_deg(p)[0], USER_POSE)

    def test_r2_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        import NERO_V47_VLA_Tools as TL
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual(v, C.VERSION)
        self.assertEqual(v, RR.VERSION)
        self.assertIn(v, ("V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3", "V4.8.4", "V4.8.5"))
        self.assertEqual(VT.MARKER_V473, "V4.7.3 READY_TO_WORK BEFORE TEACH RECORDING")
        self.assertTrue(hasattr(TL.EpisodePlayer, "_work_yolo"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
