#!/usr/bin/env python3
"""NERO V4.7.1: the wrist intrinsics the user calibrated are kept across restarts.

I1 the user's 4MP calibration (fx = 1452.7 px on 1920 px = 0.76 w, 8-parameter rational model with large higher
   terms, saved with camera_model = the 4MP name) is KEPT (V4.7: replaced by the nominal fx = 0.55 w at every start,
   so the saved wrist hand-eye no longer matched).
I2 the same numbers without a camera name are kept too (fx / w decides).
I3 the old 130 deg lens (fx / w 0.36) or a file named after the 1080P / 130 deg camera is still isolated.
I4 the stand-by 1080P camera keeps its 130 deg intrinsics; markers / versions.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import NERO_V349_Real_Wrist_Camera as W49       # noqa: E402
import NERO_V386_Real_Wrist_Camera as W86       # noqa: E402

USER = {"K": {"width": 1920, "height": 1080, "fx": 1452.7, "fy": 1445.1, "ppx": 974.5, "ppy": 512.7},
        "dist": [0.05, -0.31, 0.001, -0.002, 0.42, 0.37, -0.28, 0.55],
        "model_type": W49.DIST_MODELS[0], "rms_px": 0.583, "good": True,
        "camera_model": W86.WRIST_CAMERA_MODELS[0]}


class TestV471(unittest.TestCase):
    def test_i1_user_calibration_kept(self):
        eff, rep, why = W86.sanitize_intrinsics_for_model(USER, camera_model=W86.WRIST_CAMERA_MODELS[0], res="1920×1080")
        self.assertFalse(rep, why)
        self.assertEqual(eff["K"]["fx"], 1452.7)
        self.assertEqual(eff["dist"], USER["dist"])

    def test_i2_no_name(self):
        d = dict(USER)
        d.pop("camera_model")
        eff, rep, why = W86.sanitize_intrinsics_for_model(d, camera_model=W86.WRIST_CAMERA_MODELS[0], res="1920×1080")
        self.assertFalse(rep, why)

    def test_i3_old_lens_still_isolated(self):
        old = {"K": {"width": 1920, "height": 1080, "fx": 685.0, "fy": 685.0, "ppx": 980.0, "ppy": 520.0},
               "dist": [-0.42, 0.24, 0.002, -0.003, -0.08, -0.15, 0.05, -0.02], "model_type": W49.DIST_MODELS[0],
               "rms_px": 0.42, "good": True}
        eff, rep, why = W86.sanitize_intrinsics_for_model(old, camera_model=W86.WRIST_CAMERA_MODELS[0], res="1920×1080")
        self.assertTrue(rep)
        self.assertIn("130°", why)
        self.assertIn("重新标定内参", why)
        named = dict(USER, camera_model="1080P 130° 广角")
        eff, rep, why = W86.sanitize_intrinsics_for_model(named, camera_model=W86.WRIST_CAMERA_MODELS[0], res="1920×1080")
        self.assertTrue(rep)

    def test_i4_standby_and_version(self):
        old = {"K": {"width": 1920, "height": 1080, "fx": 685.0, "fy": 685.0, "ppx": 980.0, "ppy": 520.0},
               "dist": [-0.42, 0.24, 0, 0, -0.08], "model_type": W49.DIST_MODELS[1], "rms_px": 0.4, "good": True}
        eff, rep, _ = W86.sanitize_intrinsics_for_model(old, camera_model=W86.WRIST_CAMERA_MODELS[1], res="1920×1080")
        self.assertFalse(rep)
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual(v, C.VERSION)
        self.assertEqual(v, RR.VERSION)
        self.assertIn(v, ("V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1"))
        s = open(os.path.join(HERE, "NERO_V386_Real_Wrist_Camera.py"), encoding="utf-8").read()
        self.assertIn("V4.7.1: ONLY the focal length", s)


if __name__ == "__main__":
    unittest.main(verbosity=1)
