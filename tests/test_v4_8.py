#!/usr/bin/env python3
"""NERO V4.8 offline tests (the panel is in tests/test_v4_8_gui.py).

P1 global gripper choice: default SO-101, saved in config/gripper.json, NERO_GRIPPER overrides, blockers refuse a
   switch, listeners are told, the to-do list after a switch names effector_type:=none and the hand-eye redo.
P2 force mapping: pct 0..100 -> 0.5..3.0; requested N -> setting (linear guess 40 N ~ 3.0, gauge table interpolation,
   clipped to 0.5..3.0).
P3 AgxGripper on the simulated gripper (request protocol): connect reads the stroke version (70 mm), open_gap,
   clamp holds an object / empty clamp is not "held", relax disables, zero only acknowledged while disabled,
   stroke check, baseline threshold, teach-mode test, J7 limits read / write / range check.
P4 GripperProxy: forwards to the selected device (attributes, held), kind follows the global choice.
P5 the Agentic task I/O (RealIO) with the AgxGripper: gripper_open = real opening mm, gripper_clamp maps the skill's
   N to a force setting and reports held, gripper_state; the SO-101 path still uses its own conversions.
P6 the replay / teach helpers drive the AgxGripper with its stroke and default force; the console launches the ROS2
   driver with effector_type:=none when the AgxGripper is selected.
P7 helper protocol (pyAgxArm side) with a fake SDK: status / move / calibrate / joint-limit commands map onto the
   documented pyAgxArm calls (move_gripper_m(value, force), calibrate_gripper, set_joint_angle_vel_limits).
P8 versions / markers.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
TMP = tempfile.mkdtemp(prefix="nero_v48_")
os.environ["NERO_CALIB_DIR"] = os.path.join(TMP, "calib")
os.environ["NERO_V48_FAKE_GRIPPER"] = "1"
os.environ.pop("NERO_GRIPPER", None)

import NERO_V48_Gripper as GG                 # noqa: E402

GG.CFG_PATH = os.path.join(TMP, "gripper.json")


def fresh():
    if os.path.exists(GG.CFG_PATH):
        os.remove(GG.CFG_PATH)
    GG._blockers.clear()
    GG._listeners.clear()


class TestV48(unittest.TestCase):
    def setUp(self):
        fresh()

    def test_p1_switch(self):
        self.assertEqual(GG.active_type(), "so101")
        seen = []
        GG.add_listener(seen.append)
        flag = {"b": "任务运行中"}
        GG.add_blocker(lambda: flag["b"])
        r = GG.set_active("agx_gripper")
        self.assertFalse(r["ok"])
        self.assertIn("任务运行中", r["error"])
        flag["b"] = None
        r = GG.set_active("agx_gripper")
        self.assertTrue(r["ok"] and r["changed"])
        self.assertEqual(seen, ["agx_gripper"])
        self.assertTrue(GG.is_agx())
        self.assertTrue(any("effector_type:=none" in t for t in r["todo"]))
        self.assertTrue(any("腕部手眼" in t for t in r["todo"]))
        self.assertFalse(GG.set_active("agx_gripper")["changed"])
        self.assertFalse(GG.set_active("bogus")["ok"])
        os.environ["NERO_GRIPPER"] = "so101"
        try:
            self.assertEqual(GG.active_type(), "so101")
        finally:
            os.environ.pop("NERO_GRIPPER")

    def test_p2_force(self):
        self.assertEqual(GG.pct_to_force(0), 0.5)
        self.assertEqual(GG.pct_to_force(100), 3.0)
        self.assertEqual(GG.pct_to_force(150), 3.0)
        self.assertEqual(GG.force_level(20.0, []), 1.5)
        self.assertEqual(GG.force_level(1.0, []), 0.5)
        self.assertEqual(GG.force_level(80.0, []), 3.0)
        tab = [[0.5, 6.0], [1.0, 14.0], [3.0, 45.0]]
        self.assertAlmostEqual(GG.force_level(10.0, tab), 0.75, places=2)
        self.assertEqual(GG.force_level(2.0, tab), 0.5)
        self.assertEqual(GG.force_level(60.0, tab), 3.0)

    def test_p3_device(self):
        be = GG.FakeAgxBackend(stroke_mm=70.0)
        g = GG.AgxGripper()
        ok, msg = g.connect(backend=be)
        self.assertTrue(ok, msg)
        self.assertEqual(g.stroke_mm, 70.0)
        self.assertIn("0–70 mm", msg)
        r = g.open_gap(40.0)
        self.assertTrue(r["ok"])
        self.assertLessEqual(abs(r["gap_mm"] - 40.0), 1.0)
        self.assertEqual(g.open_gap(500)["gap_mm"] <= 70.0, True)
        be.object_mm = 30.0
        r = g.clamp(force=0.8)
        self.assertTrue(r["held"])
        self.assertAlmostEqual(r["gap_mm"], 30.0, delta=0.2)
        self.assertEqual(be.force_set, 0.8)
        be.object_mm = None
        self.assertFalse(g.clamp(force=0.8)["held"])
        self.assertEqual(g.clamp(torque_pct=100)["force_setting"], 3.0)
        g.move(10, force=9.0)
        self.assertEqual(be.force_set, 3.0)                      # clipped
        self.assertFalse(g.calibrate_zero()["ok"])                # enabled -> no ack
        g.open_gap(0.0)                                           # (the user closes the fingers fully before zeroing)
        self.assertTrue(g.relax()[0])
        self.assertFalse(be.enabled)
        self.assertTrue(g.calibrate_zero()["ok"])
        self.assertTrue(g.stroke_check()["ok"])
        b = g.force_baseline(1)
        self.assertTrue(b["ok"] and b["threshold"] > b["max_force"])
        self.assertTrue(g.teach_mode_test()["follows"])
        be.teach_mode, be.teach_accepts = True, False
        self.assertFalse(g.teach_mode_test()["follows"])
        be.teach_mode = False
        self.assertAlmostEqual(g.get_joint_limits()["max_deg"], 90.0, places=3)
        self.assertTrue(g.set_joint_limits(-87, 87)["ok"])
        self.assertAlmostEqual(g.get_joint_limits()["max_deg"], 87.0, places=3)
        self.assertFalse(g.set_joint_limits(-95, 95)["ok"])
        be.faults["motor_overheating"] = True
        self.assertEqual(g.status()["faults"], ["电机过热"])
        g.reset_faults()
        self.assertEqual(g.status()["faults"], [])
        be.alive = False
        self.assertIsNone(g.status())
        g.disconnect()
        self.assertFalse(g.connected)
        self.assertFalse(g.open_gap(10)["ok"])

    def test_p4_proxy(self):
        so = types.SimpleNamespace(connected=True, held=False, kind=None, open_gap=lambda mm: {"so101": mm})
        p = GG.GripperProxy(so)
        self.assertEqual(p.kind, "so101")
        self.assertEqual(p.open_gap(5), {"so101": 5})
        p.held = True
        self.assertTrue(so.held)
        GG.set_active("agx_gripper")
        self.assertEqual(p.kind, "agx_gripper")
        self.assertFalse(p.connected)
        p.devices["agx_gripper"].connect(backend=GG.FakeAgxBackend())
        self.assertTrue(p.connected)
        self.assertTrue(p.open_gap(20)["ok"])

    def test_p5_realio(self):
        import NERO_V365_Real_Cup_Task as RT
        GG.set_active("agx_gripper")
        g = GG.GripperProxy(types.SimpleNamespace(connected=False, held=False))
        be = GG.FakeAgxBackend()
        g.devices["agx_gripper"].connect(backend=be)
        io = RT.RealIO(None, g, lambda: {}, log=lambda s: None, motion_allowed=True)
        r = io.gripper_open(64.0)
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(r["opening_mm"], 64.0, delta=1.0)
        be.object_mm = 41.0
        c = io.gripper_clamp(20.0)
        self.assertTrue(c["hold_active"] and c["contact_left"])
        self.assertEqual(c["force_setting"], GG.force_level(20.0))
        self.assertAlmostEqual(c["gap_mm"], 41.0, delta=0.3)
        st = io.gripper_state()
        self.assertTrue(st["connected"] and st["hold_active"])
        self.assertAlmostEqual(st["opening_mm"], 41.0, delta=0.3)
        io2 = RT.RealIO(None, g, lambda: {}, log=lambda s: None, motion_allowed=False)
        self.assertTrue(io2.gripper_open(30.0)["skipped"])

    def test_p6_helpers(self):
        import NERO_V4_Dataset_Recorder as DR
        GG.set_active("agx_gripper")
        g = GG.GripperProxy(types.SimpleNamespace(connected=False, held=False))
        be = GG.FakeAgxBackend()
        g.devices["agx_gripper"].connect(backend=be)
        logs = []
        DR._grip(g, "open", logs.append)
        self.assertAlmostEqual(be.width, 100.0, delta=1.0)
        DR._grip(g, "close", logs.append)
        self.assertEqual(be.force_set, 1.0)
        with open(os.path.join(HERE, "NERO_V312_Console.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn('if GG.is_agx():\n                eff = "none"', src)
        with open(os.path.join(HERE, "NERO_V4_Console_Tab.py"), encoding="utf-8") as f:
            vt = f.read()
        self.assertIn("g.clamp(force=float(c[\"agx\"].get(\"default_force\", 1.0)))", vt)

    def test_p7_helper_protocol(self):
        calls = []

        class Msg:
            def __init__(self, **k):
                self.__dict__.update(k)

        class Eff:
            def get_gripper_status(self):
                foc = Msg(voltage_too_low=False, motor_overheating=False, driver_overcurrent=True, driver_overheating=False,
                          sensor_status=False, driver_error_status=False, driver_enable_status=True, homing_status=True)
                return Msg(msg=Msg(value=0.0425, force=0.7, foc_status=foc))

            def is_ok(self):
                return True

            def get_fps(self):
                return 199.0

            def move_gripper_m(self, value, force):
                calls.append(("move_gripper_m", value, force))

            def calibrate_gripper(self, timeout):
                calls.append(("calibrate_gripper", timeout))
                return True

            def disable_gripper(self):
                return True

        class Robot:
            def set_joint_angle_vel_limits(self, joint_index, min_angle_limit, max_angle_limit, timeout):
                calls.append(("set_joint_angle_vel_limits", joint_index, round(min_angle_limit, 4), round(max_angle_limit, 4)))
                return True

            def get_joint_angle_vel_limits(self, joint_index, timeout):
                return Msg(msg=Msg(min_angle_limit=-1.5708, max_angle_limit=1.5708))

        srv = GG.PyAgxServer.__new__(GG.PyAgxServer)
        srv.eff, srv.robot = Eff(), Robot()
        st = srv.handle({"cmd": "status"})
        self.assertEqual((st["width_m"], st["force"], st["foc"]["driver_overcurrent"]), (0.0425, 0.7, True))
        srv.handle({"cmd": "move", "width_m": 0.05, "force": 1.5})
        self.assertIn(("move_gripper_m", 0.05, 1.5), calls)
        self.assertTrue(srv.handle({"cmd": "calibrate", "timeout": 1.0})["acked"])
        srv.handle({"cmd": "set_jlim", "joint": 7, "min_rad": -1.5184, "max_rad": 1.5184})
        self.assertIn(("set_joint_angle_vel_limits", 7, -1.5184, 1.5184), calls)
        self.assertAlmostEqual(srv.handle({"cmd": "get_jlim", "joint": 7})["max_rad"], 1.5708)
        self.assertFalse(srv.handle({"cmd": "nope"})["ok"])
        g = GG.AgxGripper()
        g.backend = srv
        s = g.status()
        self.assertEqual((s["gap_mm"], s["faults"], s["enabled"]), (42.5, ["驱动过流"], True))

    def test_p8_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        import NERO_V48_Gripper_UI as GUI
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual(v, C.VERSION)
        self.assertEqual(v, RR.VERSION)
        self.assertIn(v, ("V4.8", "V4.8.1", "V4.8.2", "V4.8.3", "V4.8.4"))
        self.assertEqual(GG.MARKER, "V4.8 NERO AGX GRIPPER + GRIPPER SWITCH")
        self.assertEqual(GUI.MARKER, "V4.8 GRIPPER PANEL: ZERO + FORCE CALIBRATION + TEACH TEST + J7 LIMIT")
        self.assertEqual(GG.AGX["mass_kg"], 0.5)
        self.assertEqual(GG.AGX["can_ids"], (0x51A, 0x52A))


if __name__ == "__main__":
    unittest.main(verbosity=1)
