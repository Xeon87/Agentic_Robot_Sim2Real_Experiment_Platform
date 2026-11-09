#!/usr/bin/env python3
"""NERO V4.6.4 offline tests (视觉定位 -> 视觉精度验证: closed fingertip above the target, 3 measuring modes).

V1 closed-fingertip model: the meeting point of the closed jaws = the fixed pad's inner face at the tip end (gripper
   frame [PAD_FACE_X, 0, PAD_FIXED_CENTRE_z - PAD_HALF_z] + the Agentic tab's fingertip correction), the same mount
   as the egg task (real_mount); tcp_for_tip / tip_from_tcp_T are exact inverses; the tool points straight down.
V2 target top: egg = centre + half its width; slot = rim height; hover = fingertip - top (default 10 mm).
V3 decision rules: ① main only; ② wrist only (no two agreeing reads -> no hover); ③ wrist correction accepted only
   when two reads agree and it is <= 15 mm, else main.
V4 a full run on the fake arm (egg task's test IO): gripper closed (open_deg(0)) first, refused without the empty-clamp
   calibration; the fingertip (forward kinematics of the final joints) ends exactly at target top + hover in all three
   modes, at the wrist position in ② and in ③ (<= 15 mm), at the main position in ③ for a 20 mm wrist reading;
   an unreachable target (IK / joint margin / fence with the fingertip straight down) is refused without motion.
V5 records: CSV + summary per mode under NERO_CALIB_DIR/vision_check (not the release folder); axis hint.
V6 wiring / versions: panel attached to Camera调试 -> 视觉定位 by the console; NERO_V325_Camera.py untouched.
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
os.environ["NERO_CALIB_DIR"] = tempfile.mkdtemp(prefix="nero_v464_calib_")

import NERO_V362_Cup_Skills as CK            # noqa: E402
import NERO_V364_SO101_Gripper as G          # noqa: E402
import NERO_V365_Real_Cup_Task as RT         # noqa: E402
import NERO_V386_Real_Wrist_Camera as RWC    # noqa: E402
import NERO_V464_Vision_Check as VC          # noqa: E402
import test_v4_5 as T45                      # noqa: E402
import test_v4_6 as T46                      # noqa: E402


def wrist_fake(seq):
    """fake wrist detectors: next delta (m) relative to the expected position (None = no lock)"""
    it = iter(seq)

    def f(rgb, K, T_fl, calib, exp_xyz, *a, **k):
        d = next(it, None)
        if d is None:
            return {"ok": False, "reason": "test: no lock"}
        m = np.asarray(exp_xyz, float) + np.array([d[0], d[1], 0.0])
        return {"ok": True, "trusted": True, "measured_xyz": m.tolist(), "delta_mm": [d[0] * 1000, d[1] * 1000]}
    return f


class Grip(T45.Grip):
    def __init__(self, calibrated=True):
        self.cfg = dict(T45.Grip.cfg, touch_calibrated=calibrated)
        self.opened = []

    def open_deg(self, deg):
        self.opened.append(deg)
        return {"ok": True, "open_deg": deg, "gap_mm": 0.0}


class TestV464(unittest.TestCase):
    def test_v1_closed_tip_model(self):
        corr = [1.0, -2.0, 3.0]
        cfg = {"tip_corr_mm": corr}
        prof = RT.real_profile(CK, cfg=cfg)
        T_l7g = RT.real_mount(cfg)[0]
        tip_g = np.array([G.PAD_FACE_X, 0.0, G.PAD_FIXED_CENTRE[2] - G.PAD_HALF[2]]) + np.asarray(corr) / 1000.0
        want = T_l7g[:3, :3] @ tip_g + T_l7g[:3, 3]                     # in link7, flange at identity
        T_tcp = np.eye(4) @ G.T_L7_TCP
        np.testing.assert_allclose(VC.tip_from_tcp_T(prof, T_tcp), want, atol=1e-9)
        # pad-centre (touch calibration point) and the closed-jaw meeting point differ by 2 mm along the closing axis
        p_touch, _ = RT.model_fixed_tip_l7(cfg)
        self.assertAlmostEqual(float(np.linalg.norm(want - p_touch)) * 1000.0, 2.0, places=6)
        for yaw in (0.0, 37.0, -120.0):
            R = VC.grasp_R(yaw)
            p_tip = np.array([-0.40, 0.02, 0.052])
            p, Rt = VC.tcp_for_tip(prof, p_tip, R)
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = Rt, p
            np.testing.assert_allclose(VC.tip_from_tcp_T(prof, T), p_tip, atol=1e-9)
            np.testing.assert_allclose(R[:, 2], [0, 0, -1], atol=1e-12)         # approach straight down
            T_G = T @ np.linalg.inv(prof.T_grasp_tcp)
            self.assertAlmostEqual(T_G[2, 3] - p_tip[2], CK.TIP_BEYOND_GRASP_M, places=9)   # tip 10 mm below G

    def test_v2_target_top(self):
        egg = {"rim": np.array([-0.4, 0.0, 0.021]), "width_m": 0.042}
        self.assertAlmostEqual(VC.target_top_z(egg, "egg"), 0.042)
        self.assertAlmostEqual(VC.target_top_z({"rim": np.array([-0.3, 0.1, 0.022])}, "slot"), 0.022)
        self.assertEqual(VC.HOVER_DEFAULT_MM, 10.0)

    def test_v3_decide(self):
        m = [-0.400, 0.000]
        self.assertEqual(VC.decide("main", m, [-0.39, 0.0])[1], "main")
        xy, src, why = VC.decide("wrist", m, None)
        self.assertIsNone(xy)
        self.assertIn("不悬停", why)
        self.assertEqual(VC.decide("wrist", m, [-0.39, 0.0])[1], "wrist")
        xy, src, why = VC.decide("assist", m, [-0.392, 0.006])
        self.assertEqual(src, "wrist")
        xy, src, why = VC.decide("assist", m, [-0.380, 0.0])
        self.assertEqual(src, "main")
        self.assertIn("上限", why)
        self.assertEqual(VC.decide("assist", m, None)[1], "main")
        self.assertIsNone(VC.agree_pair([np.array([0.0, 0.0])]))
        self.assertIsNone(VC.agree_pair([np.array([0.0, 0.0]), np.array([0.005, 0.0])]))
        np.testing.assert_allclose(VC.agree_pair([np.array([0.0, 0.0]), np.array([0.003, 0.0])]), [0.0015, 0.0])

    def _vc(self, grip=None):
        io = T46.egg_io(grip=grip or Grip(), wrist=True)
        sk = RT.RealEggSkills(io)
        vc = VC.VisionCheck(io, sk, log=io.logs.append, photo_fn=lambda: (np.zeros((4, 4, 3), np.uint8), None),
                            book=VC.RecordBook())
        vc.observe()
        return vc, io

    def _tip_end(self, vc):
        return vc._tip_now()

    def test_v4_runs(self):
        vc, io = self._vc()
        self.assertIn("egg_1", vc.targets)
        self.assertIn("slot_1", vc.targets)
        for tid in ("egg_1", "slot_12"):
            t = vc.targets[tid]
            res = vc.run("main", tid, 10.0)
            self.assertTrue(res["ok"], res)
            tip = self._tip_end(vc)
            np.testing.assert_allclose(tip, [t["xyz"][0], t["xyz"][1], t["top_z"] + 0.010], atol=0.0015)
            self.assertEqual(res["row"]["used_from"], "main")
            self.assertLess(res["row"]["arrive_err_mm"], 1.5)
            self.assertTrue(res["row"]["photo_main"].startswith(os.environ["NERO_CALIB_DIR"]))
        self.assertEqual(io.grip.opened[0], 0.0)                     # gripper closed before the first motion
        # ② wrist only: hover at the wrist position (two agreeing reads 6 mm off)
        d = (0.006, -0.002)
        for tid, fn in (("egg_1", "wrist_check_egg_or_slot"), ("slot_12", "wrist_find_slot_cavity")):
            vc, io = self._vc()
            t = vc.targets[tid]
            with T46.Patch(RWC, fn, wrist_fake([d, d])):
                res = vc.run("wrist", tid, 10.0)
            self.assertTrue(res["ok"], res)
            np.testing.assert_allclose(self._tip_end(vc)[:2], t["xyz"][:2] + np.array(d), atol=0.0015)
            self.assertEqual(res["row"]["used_from"], "wrist")
            self.assertEqual(res["row"]["wrist_reads"], 2)
        # ② no agreeing reads -> no hover at all
        vc, io = self._vc()
        with T46.Patch(RWC, "wrist_check_egg_or_slot", wrist_fake([(0.006, 0), (0.0, 0.0), None, None, None, None])):
            res = vc.run("wrist", "egg_1", 10.0)
        self.assertFalse(res["ok"])
        self.assertIn("不悬停", res["error"])
        # ③ assist: 6 mm -> wrist; 20 mm -> main
        for d, src in (((0.006, 0.0), "wrist"), ((0.020, 0.0), "main")):
            vc, io = self._vc()
            t = vc.targets["egg_1"]
            with T46.Patch(RWC, "wrist_check_egg_or_slot", wrist_fake([d, d])):
                res = vc.run("assist", "egg_1", 15.0)
            self.assertTrue(res["ok"], res)
            self.assertEqual(res["row"]["used_from"], src)
            want = t["xyz"][:2] + (np.array(d) if src == "wrist" else 0.0)
            np.testing.assert_allclose(self._tip_end(vc)[:2], want, atol=0.0015)
            self.assertAlmostEqual(self._tip_end(vc)[2], t["top_z"] + 0.015, delta=0.0015)
        # lift: straight up to 80 mm above the top
        vc.lift("egg_1")
        self.assertAlmostEqual(self._tip_end(vc)[2], vc.targets["egg_1"]["top_z"] + 0.08, delta=0.0015)
        # a slot the closed fingertip cannot reach pointing straight down (26 cm from the base): refused, no motion
        vc, io = self._vc()
        n0 = len(io.motion_log)
        res = vc.run("main", "slot_1", 10.0)
        self.assertFalse(res["ok"])
        self.assertIn("不可达", res["error"])
        self.assertEqual(len(io.motion_log), n0)
        # no empty-clamp calibration -> refused before any motion
        vc, io = self._vc(grip=Grip(calibrated=False))
        n0 = len(io.motion_log)
        with self.assertRaises(RuntimeError):
            vc.run("main", "egg_1", 10.0)
        self.assertEqual(len(io.motion_log), n0)

    def test_v5_records(self):
        book = VC.RecordBook(day="20991231")
        self.assertTrue(book.csv_path.startswith(os.path.join(os.environ["NERO_CALIB_DIR"], "vision_check")))
        for mode, dx, dy in (("main", 12.0, -3.0), ("main", 14.0, -5.0), ("wrist", 1.0, 0.0)):
            book.append({"mode": mode, "target": "egg_1", "user_dx_mm": dx, "user_dy_mm": dy})
        book.append({"mode": "assist", "target": "egg_1", "user_note": "没有悬停"})
        rows = book.rows()
        self.assertEqual(len(rows), 4)
        s = VC.summarize(rows)
        self.assertTrue(any("① 只用主相机：2 点 · 平均 dx +13.0 / dy -4.0 mm" in x for x in s), s)
        self.assertTrue(any("② 只用腕部相机：1 点" in x for x in s), s)
        self.assertFalse(any("③" in x for x in s), s)
        cam = {"K": {"fx": 900.0, "fy": 900.0, "ppx": 640.0, "ppy": 360.0},
               "T_base_cam": [[1.0, 0, 0, -0.40], [0, -1.0, 0, 0.0], [0, 0, -1.0, 0.75], [0, 0, 0, 1.0]]}
        self.assertEqual(VC.axis_hint(cam, [-0.40, 0.0, 0.04]), "主相机画面里：+X = 画面右，+Y = 画面上")

    def test_v6_wiring(self):
        con = open(os.path.join(HERE, "NERO_V312_Console.py"), encoding="utf-8").read()
        self.assertIn("VC.attach(tk, ttk, cam_tab, rct_tab, pipe, jpanel, log=cam_log_line)", con)
        self.assertLess(con.index("rct_tab = RCT.RealCupTaskTabUI("), con.index("VC.attach("))
        h = hashlib.sha256(open(os.path.join(HERE, "NERO_V325_Camera.py"), "rb").read()).hexdigest()
        self.assertEqual(h, "82f2416d73671e7018d09d0cd72c17186154934a315b47c0d19fdb0427afd931")
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        v = open(os.path.join(HERE, "VERSION")).read().strip()
        self.assertIn(v, ("V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10"))
        self.assertEqual((C.VERSION, RR.VERSION), (v, v))
        self.assertEqual(VC.MARKER_V464, "V4.6.4 VISION ACCURACY CHECK")
        src = open(os.path.join(HERE, "NERO_V464_Vision_Check.py"), encoding="utf-8").read()
        import re
        self.assertFalse(re.search("[\U0001F300-\U0001FAFF⚠]", src))       # no emoji / warning sign in the GUI


if __name__ == "__main__":
    unittest.main(verbosity=1)
