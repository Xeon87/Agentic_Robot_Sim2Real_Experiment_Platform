#!/usr/bin/env python3
"""NERO V4.8.2 offline tests: AgxGripper speed limit + the chain review for the new gripper / new room.

S1 speed: 100 % = one command; 40 % (default) walks 0 -> 70 mm in ~1.2 s, 30 % slower; a clamp at 40 % still
   detects the held object; the setting is global, clipped to 10..100 %; relax() stops a running ramp.
S2 Smol VLA observation: the AgxGripper opening is normalised by ITS stroke (60 of 70 mm -> 0.857, before: 1.0).
S3 recorded episodes name the gripper (AgxGripper / SO-101).
S4 a gripper that does not open stops the Agentic task (no descent with closed fingers).
S5 task-start warnings: changed room lighting, AgxGripper not zeroed in this folder, no wrist hand-eye (bracket
   not in the collision model), the speed; a gripper fault blocks execute.
S6 the camera-lab lighting check keeps its result for the tasks.
S7 versions / markers.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ["NERO_CALIB_DIR"] = tempfile.mkdtemp(prefix="nero_v482_calib_")
os.environ["NERO_V48_FAKE_GRIPPER"] = "1"
os.environ.pop("NERO_GRIPPER", None)

import NERO_V48_Gripper as GG                  # noqa: E402

GG.CFG_PATH = os.path.join(tempfile.mkdtemp(prefix="nero_v482_"), "gripper.json")


def agx(speed=None):
    c = GG.load_cfg()
    c["type"] = "agx_gripper"
    if speed is not None:
        c["agx"]["speed_pct"] = speed
    GG.save_cfg(c)
    g = GG.AgxGripper()
    g.connect()
    return g


class TestV482(unittest.TestCase):
    def test_s1_speed(self):
        self.assertEqual(GG.DEFAULTS["agx"]["speed_pct"], 40.0)
        times = {}
        for pct in (100, 40, 30):
            g = agx(pct)
            g.open_gap(0.0)
            n0 = len(g.backend.log)
            t0 = time.time()
            r = g.open_gap(70.0)
            times[pct] = time.time() - t0
            moves = [e for e in g.backend.log[n0:] if e.get("cmd") == "move"]
            self.assertTrue(r["ok"], r)
            if pct == 100:
                self.assertEqual(len(moves), 1)
            else:
                self.assertGreater(len(moves), 20)
                w = [e["width_m"] for e in moves]
                self.assertEqual(w, sorted(w))                     # monotonic, no overshoot
        self.assertLess(times[100], 0.6)
        self.assertAlmostEqual(times[40], 70.0 / (150.0 * 0.4), delta=0.35)
        self.assertGreater(times[30], times[40])
        g = agx(40)
        g.backend.object_mm = 38.0
        self.assertTrue(g.clamp(force=0.5)["held"])
        self.assertEqual(GG.set_speed_pct(3), GG.SPEED_MIN_PCT)
        self.assertEqual(GG.set_speed_pct(250), 100.0)
        self.assertEqual(GG.parse_speed(" 50 % "), 50.0)
        self.assertIsNone(GG.parse_speed("fast"))
        g = agx(20)
        g.open_gap(0.0)
        out = {}
        th = threading.Thread(target=lambda: out.setdefault("r", g.move(70.0)))
        th.start()
        time.sleep(0.25)
        g.relax()
        th.join(3.0)
        self.assertFalse(out["r"]["ok"])
        self.assertLess(g.backend.target, 50.0)

    def test_s2_bridge(self):
        import NERO_V4_Observation_Bridge as OB
        b = OB.ObservationBridge.__new__(OB.ObservationBridge)
        b.lock = threading.Lock()
        b.update_gripper_state(gap_mm=60.0, max_gap_mm=70.0)
        self.assertAlmostEqual(b._gripper_norm, 60.0 / 70.0, places=4)
        b.update_gripper_state(gap_mm=60.0)
        self.assertEqual(b._gripper_norm, 1.0)                     # SO-101 scale (45 mm) unchanged

    def test_s3_recorder(self):
        import NERO_V4_Dataset_Recorder as DR
        agx()
        self.assertEqual(DR._gripper_name(), "AgxGripper")
        c = GG.load_cfg()
        c["type"] = "so101"
        GG.save_cfg(c)
        self.assertEqual(DR._gripper_name(), "SO-101")

    def test_s4_open_failure_stops(self):
        import NERO_V365_Real_Cup_Task as RT
        g0 = agx(100)
        proxy = GG.GripperProxy(types.SimpleNamespace(connected=False, held=False))
        proxy.devices["agx_gripper"] = g0
        io = RT.RealIO(None, proxy, lambda: {}, log=lambda s: None, motion_allowed=True)
        self.assertTrue(io.gripper_open(60.0)["ok"])
        g0.backend.alive = False
        with self.assertRaises(RuntimeError) as cm:
            io.gripper_open(60.0)
        self.assertIn("不再下探", str(cm.exception))

    def test_s5_warnings(self):
        import NERO_V365_Real_Cup_Task as RT
        import NERO_V379_Camera_Lab as CL
        import NERO_V4613_Tool_Collision as TC
        g = agx(40)
        t = RT.RealCupTaskTabUI.__new__(RT.RealCupTaskTabUI)
        t.GG = GG
        t.grip = types.SimpleNamespace(last={"homed": False, "faults": []}, connected=True)
        CL.LAST_LIGHTING.clear()
        CL.LAST_LIGHTING.update({"bad": True, "ref_mean": 44.0, "mean": 78.0, "change_pct": 76, "sat_pct": 0.99})
        orig = TC.find_handeye
        TC.find_handeye = lambda calib_dir=None: (None, None)
        try:
            w = t.preflight_warnings("execute")
        finally:
            TC.find_handeye = orig
            CL.LAST_LIGHTING.clear()
        txt = "\n".join(w)
        for key in ("44 → 78", "校正归零", "未归零", "没有腕部手眼", "开合速度 40%"):
            self.assertIn(key, txt)
        self.assertEqual(t.preflight_warnings("plan"), [])        # plan: no gripper notes, lighting cleared
        with open(os.path.join(HERE, "NERO_V365_Real_Cup_Task.py"), encoding="utf-8") as f:
            self.assertIn("NERO 原装夹爪有故障", f.read())

    def test_s6_lighting_kept(self):
        import NERO_V379_Camera_Lab as CL

        class Src:
            def frames(self):
                return np.full((480, 640, 3), 200, np.uint8), None, 30
        CL.LAST_LIGHTING.clear()
        CL.lighting_check(Src(), {"ref": {"mean": 100.0, "roi": None}}, log=lambda s: None, wait_s=1.0)
        self.assertTrue(CL.LAST_LIGHTING.get("bad"))
        self.assertGreater(CL.LAST_LIGHTING["change_pct"], 50)

    def test_s7_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual((v, RR.VERSION), (C.VERSION, C.VERSION))
        self.assertIn(v, ("V4.8.2", "V4.8.3", "V4.8.4", "V4.8.5"))
        self.assertEqual(GG.SPEED_CHOICES, (30, 40, 50, 70, 100))


if __name__ == "__main__":
    unittest.main(verbosity=1)
