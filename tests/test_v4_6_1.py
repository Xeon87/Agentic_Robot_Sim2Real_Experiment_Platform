#!/usr/bin/env python3
"""NERO V4.6.1 offline tests (the user's 7-point list after three failed egg runs on V4.6).

N1 config/egg_carton.json: held_gap_mm 26.5 (measured), margin 6, regrasp off, pre-pick slot look off.
N2 the user's three good grasps (26.7 / 26.5 / 26.5 mm) are NOT off-centre; 18 mm is suspected but only logged by
   default (egg kept, pick continues); release / regrasp only with offcentre_regrasp = true.
N3 wrist egg reading: logged only - even the 30 mm reading that stopped V4.6 run 2 no longer moves or stops anything.
N4 pre-pick slot look is off by default: no extra motion, the cavity detector is never called.
N5 wrist view gate: the gripper region of the wrist image is excluded; an over-exposed (white) search area gives
   "unusable" and no number, in both wrist detectors; a blob cut by the excluded region is not measured.
N6 tab names: 「Agentic实机任务」 and 「Smol VLA实机任务」; the SmolVLA tab mirrors the Agentic tab's monitors
   (identical headings; drawing parity is checked with real Tk in tests/test_v4_6_1_gui.py).
N7 kept from V4.6: strict arrival / no release when not precise / release check / SmolVLA gate (tests/test_v4_6.py).
"""
from __future__ import annotations

import json
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTS = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import NERO_V349_Real_Wrist_Camera as W49  # noqa: E402
import NERO_V365_Real_Cup_Task as RT  # noqa: E402
import NERO_V386_Real_Wrist_Camera as RWC  # noqa: E402
import test_v4_6 as T46  # noqa: E402


class TestV461(unittest.TestCase):
    def test_n1_config(self):
        raw = json.load(open(os.path.join(HERE, "config", "egg_carton.json"), encoding="utf-8"))
        self.assertEqual(raw["held_gap_mm"], 26.5)
        cfg = W49.egg_task_config()
        self.assertEqual((cfg["held_gap_mm"], cfg["offcentre_margin_mm"]), (26.5, 6.0))
        self.assertFalse(cfg["offcentre_regrasp"])
        self.assertFalse(cfg["wrist_slot_look_before_pick"])
        self.assertEqual(W49.egg_carton_config()["slot_pitch_mm"], [46.0, 47.0])   # V4.6.6: measured 2x3 carton
        bad = os.path.join(self.tmp_dir(), "bad.json")
        json.dump({"held_gap_mm": "x", "offcentre_regrasp": 1}, open(bad, "w"))
        c2 = W49.egg_task_config(bad)
        self.assertEqual(c2["held_gap_mm"], 26.5)                        # bad value -> default
        self.assertTrue(c2["offcentre_regrasp"])

    def tmp_dir(self):
        import tempfile
        return tempfile.mkdtemp(prefix="nero_v461_")

    def _pick(self, gaps, **cfg):
        with T46.Patch(W49, "egg_task_config", T46.egg_cfg(**cfg)):
            io = T46.egg_io(grip=T46.SeqGrip(gaps))
            sk = T46.TestV46Egg._sk(self, io)
            r = sk.pick_egg("egg_1")
        return sk, io, r

    def test_n2_measured_baseline(self):
        for g in (26.7, 26.5, 26.5, 21.0):
            sk, io, r = self._pick([g])
            self.assertTrue(r["ok"], (g, r))
            self.assertFalse(r["physical_verification"]["camera_agreement"].get("offcentre_suspected", False), g)
            self.assertTrue(any(f"夹紧读数 {g:.1f} mm（基准 26.5 mm" in s for s in io.logs), io.logs[-5:])
        opened = []
        sk, io, r = self._pick([18.0])
        self.assertTrue(r["ok"], r)
        self.assertEqual(sk.held, "egg_1")
        self.assertTrue(r["physical_verification"]["camera_agreement"]["offcentre_suspected"])
        self.assertTrue(any("只记录，继续" in s for s in io.logs))
        sk, io, r = self._pick([18.0, 17.5], offcentre_regrasp=True)
        self.assertEqual(r["reason"], "EGG_GRASP_OFF_CENTRE")

    def test_n3_wrist_egg_log_only(self):
        for d in ((-0.0001, -0.0304), (0.012, 0.0), (0.004, 0.0)):
            io = T46.egg_io(wrist=True)
            sk = T46.TestV46Egg._sk(self, io)
            g0 = np.asarray(sk.plan["_wps"]["grasp"][0], float).copy()
            with T46.Patch(RWC, "wrist_check_egg_or_slot", T46.egg_reads([d, d, d])):
                r = sk.pick_egg("egg_1")
            self.assertTrue(r["ok"], (d, r))
            np.testing.assert_allclose(np.asarray(sk.plan["_wps"]["grasp"][0]), g0)
            self.assertFalse(any("second look" in m["label"] for m in io.motion_log))
            self.assertTrue(any("[V4.6.1 CAMERA AGREEMENT]" in s and "只记录" in s for s in io.logs))

    def test_n4_slot_look_off_by_default(self):
        io = T46.egg_io(wrist=True)
        sk = RT.RealEggSkills(io)
        sk.observe_scene()
        self.assertTrue(sk.plan_egg_place("egg_1", "slot_12")["ok"])

        def boom(*a, **k):
            raise AssertionError("the slot detector must not run when the switch is off")
        with T46.Patch(RWC, "wrist_find_slot_cavity", boom), \
                T46.Patch(RWC, "wrist_check_egg_or_slot", T46.egg_reads([None, None])):
            r = sk.pick_egg("egg_1")
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["physical_verification"]["camera_agreement"]["slot"]["skipped"], "switch_off")
        self.assertFalse(any("wrist look" in m["label"] for m in io.motion_log))
        self.assertTrue(any("夹蛋前看槽：关闭" in s for s in io.logs))

    def test_n5_view_gate(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("synth", os.path.join(TESTS, "v46_synth_slot.py"))
        syn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(syn)
        # the normal scene still works (slot in the upper, visible half)
        img, K, T, he, exp = syn.scene()
        r = RWC.wrist_find_slot_cavity(img, K, T, he, exp)
        self.assertTrue(r["ok"] and r["trusted"], r)
        self.assertAlmostEqual(r["delta_mm"][0], 8.0, delta=1.0)
        # white / over-exposed view -> unusable, no number (both detectors)
        white = np.full_like(img, 252)
        r1 = RWC.wrist_find_slot_cavity(white, K, T, he, exp)
        r2 = RWC.wrist_check_egg_or_slot(white, K, T, he, exp, kind="egg")
        for r_ in (r1, r2):
            self.assertFalse(r_["ok"])
            self.assertTrue(r_.get("unusable"))
            self.assertIn("过曝", r_["reason"])
            self.assertNotIn("delta_xy_m", r_)
        # slot projected into the gripper band (lower part of the image) -> excluded, no reading
        img3, K, T, he, exp = syn.scene(cam_dx=+0.06)
        r3 = RWC.wrist_find_slot_cavity(img3, K, T, he, exp)
        self.assertFalse(r3.get("ok") and r3.get("trusted"), r3)      # no trusted number for a hidden slot
        cfg = W49.egg_task_config()
        self.assertEqual(cfg["wrist_gripper_mask"][0], [0.0, 0.60])
        self.assertEqual(RWC.MARKER_V461_VIEW, "V4.6.1 WRIST VIEW GATE")

    def test_n6_tabs(self):
        import NERO_V312_Console as C
        self.assertEqual(C.TAB_AGENTIC, "Agentic实机任务")
        self.assertEqual(C.TAB_SMOLVLA, "Smol VLA实机任务")
        con = open(os.path.join(HERE, "NERO_V312_Console.py"), encoding="utf-8").read()
        self.assertIn("nb.add(rct_outer, text=TAB_AGENTIC)", con)
        self.assertIn("nb.add(vla_outer, text=TAB_SMOLVLA)", con)
        self.assertNotIn('text="NERO实机任务"', con)
        self.assertNotIn('text="VLA / LeRobot (V4)"', con)
        self.assertTrue(hasattr(RT, "CameraMonitorMirror"))
        tab = open(os.path.join(HERE, "NERO_V4_Console_Tab.py"), encoding="utf-8").read()
        self.assertIn("RCT.CameraMonitorMirror(tk, ttk, mons, self.rct_tab, yolo_parent=", tab)   # V4.6.2: log moved below
        self.assertIn("self.monitor.refresh()", tab)
        rct = open(os.path.join(HERE, "NERO_V365_Real_Cup_Task.py"), encoding="utf-8").read()
        self.assertEqual(rct.count("text=MON_MAIN_TITLE"), 2)        # same heading in both tabs
        self.assertEqual(rct.count("text=MON_WRIST_TITLE"), 2)

    def test_n8_arrival_wait_and_frozen_camera(self):
        """MoveIt 'done' != arrived: move_and_settle keeps reading while the arm moves; NERO_V325_Camera.py frozen."""
        import hashlib
        import NERO_V313_Real_Robot as RR

        class F:
            ARRIVE_TOL_DEG, ARRIVE_TIMEOUT_S, ARRIVE_STILL_S = 0.3, 8.0, 0.5

            def __init__(self, seq):
                self.seq, self.logs = list(seq), []

            def actual_arm_deg(self, m):
                return self.seq.pop(0) if len(self.seq) > 1 else self.seq[0]

            def log(self, s_):
                self.logs.append(s_)
        tgt = [0, 0, 0, 62.6, 0, 0, 0]
        # J4 still catching up (53.4 -> 58 -> 62.5): waits and returns the arrived pose
        f = F([[0, 0, 0, 53.4, 0, 0, 0], [0, 0, 0, 58.0, 0, 0, 0], [0, 0, 0, 62.5, 0, 0, 0]])
        act = RR.RealPipeline._v46_wait_arrival(f, None, tgt, "标定复核 1")
        self.assertAlmostEqual(act[3], 62.5)
        self.assertEqual(f.logs, [])
        # stopped short (does not move any more): no long wait, a warning with the joint and the error
        f = F([[0, 0, 0, 53.4, 0, 0, 0]])
        import time as _t
        t0 = _t.time()
        act = RR.RealPipeline._v46_wait_arrival(f, None, tgt, "标定复核 1")
        self.assertLess(_t.time() - t0, 2.0)
        self.assertTrue(any("J4 仍差 9.2°" in s_ for s_ in f.logs), f.logs)
        h = hashlib.sha256(open(os.path.join(HERE, "NERO_V325_Camera.py"), "rb").read()).hexdigest()
        self.assertEqual(h, "82f2416d73671e7018d09d0cd72c17186154934a315b47c0d19fdb0427afd931")
        self.assertEqual(RR.VERSION, open(os.path.join(HERE, "VERSION")).read().strip())

    def test_n7_markers(self):
        with open(os.path.join(HERE, "VERSION")) as f:
            self.assertIn(f.read().strip(), ("V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2"))
        self.assertEqual(RT.MARKER_V461, "V4.6.1 MEASURED HELD GAP + WRIST LOG ONLY")
        self.assertEqual((RT.STRICT_POS_MM, RT.STRICT_ROT_DEG, RT.STRICT_JOINT_DEG), (2.0, 1.0, 0.5))


if __name__ == "__main__":
    unittest.main(verbosity=1)
