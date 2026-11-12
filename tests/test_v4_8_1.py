#!/usr/bin/env python3
"""NERO V4.8.1 offline tests: the Agentic real tasks with the NERO AgxGripper.

Q1 AgxGripper geometry (official URDF): fingertips 19.7 mm past the TCP, the fingers travel along link7 z, housing
   37..102 mm on link7 x; the user's wrist hand-eye puts the camera at grasp-frame (+54, +2, -33) mm (above the
   housing, behind the tips); the table check covers housing / bracket / camera (fingertips have their 5 mm rule).
Q2 switching: real_mount / real_profile / fingertip depth / fence tool point follow the selected gripper; SO-101
   values unchanged.
Q3 egg with the AgxGripper (the user's measured slots): every slot plans; the egg is gripped by its end on the side
   AWAY from the camera (egg_in_G x = -24 mm), turned upright with pitch > 0 (egg down, camera up); the fingers stay
   >= 5 mm above the carton top; the gripper body >= 8 mm above the table.
Q4 eggs already standing next to the slot: no planned pose puts the fingers / housing / bracket / camera into them.
Q5 SO-101 egg plans unchanged (pitch < 0, fixed-jaw offset 20 mm).
Q6 cup stacking with the AgxGripper plans (official profile: 60 mm opening, both pinch directions).
Q7 egg numbers: held 30..48 mm, open 60 mm, long-axis warning > 50 mm; force 4.2 N -> setting 0.5.
Q8 presets / tip length: cup preset 65 mm (the 70 mm stroke); tip_from_flange_mm changes the fingertip depth;
   the sort task refuses hover / execute with the AgxGripper; 一键启动 connects the AgxGripper.
Q9 versions / markers.
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
os.environ["NERO_CALIB_DIR"] = tempfile.mkdtemp(prefix="nero_v481_calib_")
os.environ.pop("NERO_GRIPPER", None)

import NERO_V48_Gripper as GG                  # noqa: E402

GG.CFG_PATH = os.path.join(tempfile.mkdtemp(prefix="nero_v481_"), "gripper.json")
import NERO_V481_Agx_Geom as AG                # noqa: E402
import NERO_V4613_Tool_Collision as TC         # noqa: E402
import NERO_V362_Cup_Skills as CK              # noqa: E402
import NERO_V365_Real_Cup_Task as RT           # noqa: E402
import test_v4_6_13 as T13                     # noqa: E402

USER_CAM = np.array([[1, 0, 0, 0.1221], [0, 1, 0, -0.0537], [0, 0, 1, -0.0250], [0, 0, 0, 1.0]])   # 2026-11-10
_find = TC.find_handeye


def use(kind, cam=True):
    c = GG.load_cfg()
    c["type"] = kind
    c["agx"]["tip_from_flange_mm"] = 144.0
    GG.save_cfg(c)
    TC.find_handeye = (lambda calib_dir=None: (USER_CAM.copy(), "user")) if cam else (lambda calib_dir=None: (None, None))


def finger_low(sk, q):
    T = sk._fast.fk(np.asarray(q, float))
    return sk._agx_finger_low(T[:3, 3], T[:3, :3])


class TestV481(unittest.TestCase):
    def tearDown(self):
        TC.find_handeye = _find

    def test_q1_geometry(self):
        use("agx_gripper")
        self.assertAlmostEqual(AG.tip_drop_m(), 0.0197, places=4)
        m = AG.AgxToolModel(T_flange_cam=USER_CAM)
        cam = m.cam_G * 1000.0
        self.assertTrue(np.allclose(cam, [53.7, 1.5, -33.2], atol=0.2), cam)
        b = {n: c for n, c, R, h in m.boxes}
        self.assertAlmostEqual(b["finger_a"][2] + b["finger_b"][2], 2 * -0.0235, places=3)   # symmetric about link7 z
        self.assertAlmostEqual(b["finger_a"][0], b["finger_b"][0], places=6)
        self.assertAlmostEqual(b["housing"][0], 0.0695, places=3)
        R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])           # TCP pointing down, 30 mm above the table
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, [-0.4, 0.0, 0.030]
        Tl7 = T @ np.linalg.inv(AG.T_L7_TCP)
        ok, dz, part = m.clear_of_table(Tl7, 0.0)
        self.assertTrue(ok and part != "finger_a", (dz, part))
        z_f, n_f = m.lowest_of(Tl7, ("finger_a", "finger_b"))
        self.assertAlmostEqual(z_f * 1000.0, 30.0 - 19.7 - 3.0, delta=0.2)          # tips 19.7 mm below the TCP
        self.assertIsInstance(TC.make_tool_model(), AG.AgxToolModel)
        use("so101")
        self.assertIsInstance(TC.make_tool_model(), TC.ToolModel)

    def test_q2_switch(self):
        use("so101")
        T_s, Ttg_s, m_s = RT.real_mount()
        self.assertEqual(RT.real_profile(CK).name, "so101")
        d_s = RT.so101_tip_drop_m()
        use("agx_gripper")
        T_a, Ttg_a, m_a = RT.real_mount()
        self.assertEqual(m_a["gripper"], "agx_gripper")
        self.assertTrue(np.allclose(T_a, AG.T_l7_grasp()))
        prof = RT.real_profile(CK)
        self.assertEqual((prof.name, prof.open_mm, tuple(prof.flips)), ("agx_gripper", 60.0, (False, True)))
        self.assertTrue(prof.lenient_pick and prof.camera_decides_grip and prof.recheck_held_before_place)
        self.assertAlmostEqual(RT.so101_tip_drop_m(), 0.0197, places=4)
        self.assertNotAlmostEqual(d_s, RT.so101_tip_drop_m(), places=3)
        tip = RT.so101_tip_tool_T(T_a)
        self.assertTrue(np.allclose(tip, AG.T_l7_tip()))
        self.assertAlmostEqual(tip[0, 3], 0.175, places=3)                # tips at link7 x 175 mm

    def test_q3_egg_agx(self):
        use("agx_gripper")
        sk, io = T13.user_scene()
        self.assertTrue(sk.agx)
        for slot in ("slot_1", "slot_2", "slot_3", "slot_4", "slot_5", "slot_6"):
            r = sk.plan_egg_place(source_egg="egg_1", target_slot=slot)
            self.assertTrue(r["ok"], (slot, r, [s for s in io.logs if "PLACE" in s][-1:]))
            pl = sk.plan
            self.assertTrue(-0.0245 <= pl["egg_in_G_m"][0] <= -0.004, pl["egg_in_G_m"])   # V4.8.4: smallest offset
            self.assertGreater(pl["place_pitch_deg"], 0.0)
            zt = float(pl.get("table_z_m", 0.0))
            low = finger_low(sk, pl["_q"]["release"])
            self.assertGreaterEqual(low - zt - AG.CARTON_TOP_ABOVE_TABLE_M, AG.PAD_ABOVE_CARTON_M - 1e-6, slot)
            tm = sk._tool_model()
            ok, dz, part = tm.clear_of_table(sk._fast.fk(np.asarray(pl["_q"]["release"])) @
                                             np.linalg.inv(sk.CK.T_L7_TCP), zt)
            self.assertTrue(ok, (slot, dz, part))
            # the camera points up at the release (egg down)
            T_rel = sk._fast.fk(np.asarray(pl["_q"]["release"], float))
            cam_w = T_rel[:3, :3] @ USER_CAM[:3, 3] + T_rel[:3, 3]
            self.assertGreater(cam_w[2], T_rel[2, 3])

    def test_q4_neighbours(self):
        use("agx_gripper")
        sk, io = T13.user_scene()
        sk.plan_egg_place(source_egg="egg_1", target_slot="slot_2")
        occ = [np.asarray(sk.cam[s]["rim"], float)[:2] for s in ("slot_1", "slot_3", "slot_5")]
        sk._occupied_xy = lambda exclude=None: [list(o) for o in occ]
        r = sk.plan_egg_place(source_egg="egg_1", target_slot="slot_2")
        if r["ok"]:
            T = sk._fast.fk(np.asarray(sk.plan["_q"]["release"], float))
            self.assertTrue(sk._agx_neighbours_clear(T[:3, 3], T[:3, :3], occ, float(sk.plan.get("table_z_m", 0.0))))
        else:
            self.assertTrue(any("PLACE" in s for s in io.logs))           # refused with a reason, never a collision

    def test_q5_egg_so101(self):
        use("so101", cam=False)                       # the SO-101 run of test_v4_6_13 (no AgxGripper hand-eye)
        sk, io = T13.user_scene()
        self.assertFalse(sk.agx)
        r = sk.plan_egg_place(source_egg="egg_1", target_slot="slot_1")
        self.assertTrue(r["ok"])
        self.assertLess(sk.plan["place_pitch_deg"], 0.0)
        self.assertAlmostEqual(sk.plan["egg_in_G_m"][1], 0.0203, delta=0.002)

    def test_q6_cup_agx(self):
        use("agx_gripper")
        knobs = CK.cup_knobs()[0]
        r = CK.plan_cup_stack(knobs, [-0.42, 0.10, 0.1226], [1, 0, 0, 0], [-0.32, -0.12, 0.1226], [1, 0, 0, 0],
                              q_start=[0.0, 0.4, 0.0, 1.2, 0.0, 0.3, 0.0], profile=RT.real_profile(CK))
        self.assertTrue(r.get("ok"), r.get("error"))
        self.assertGreater(float(r.get("min_joint_margin_deg", 0.0)), 5.0)

    def test_q7_egg_numbers(self):
        use("agx_gripper")
        sk, io = T13.user_scene()
        self.assertEqual(RT.egg_open_mm(), 60.0)
        self.assertEqual(RT.egg_long_axis_gap_mm(), 50.0)
        self.assertEqual(sk._is_egg_held(None, {"gap_mm": 42.0})[0], True)
        self.assertEqual(sk._is_egg_held(None, {"gap_mm": 3.0})[0], False)
        self.assertEqual(sk._is_egg_held(None, {"gap_mm": 60.0})[0], False)
        self.assertEqual(sk._egg_cfg()["held_gap_mm"], 42.0)
        self.assertEqual(GG.force_level(RT.EGG_CLAMP_N), 0.5)
        use("so101")
        self.assertEqual(RT.egg_open_mm(), RT.EGG_OPEN_MM)

    def test_q8_presets_tip_sort_autoconnect(self):
        self.assertEqual(GG.DEFAULTS["presets"]["cup"]["open_mm"], 65.0)
        use("agx_gripper")
        c = GG.load_cfg()
        c["agx"]["tip_from_flange_mm"] = 150.0
        GG.save_cfg(c)
        self.assertAlmostEqual(AG.tip_drop_m(), 0.0257, places=4)
        c["agx"]["tip_from_flange_mm"] = 144.0
        GG.save_cfg(c)
        with open(os.path.join(HERE, "NERO_V365_Real_Cup_Task.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn('self.GG.is_agx() and scene == "sort_items"', src)
        with open(os.path.join(HERE, "NERO_V312_Console.py"), encoding="utf-8") as f:
            con = f.read()
        self.assertIn('start_job("一键启动", run_all_then_gripper)', con)

    def test_q9_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual(v, C.VERSION)
        self.assertEqual(v, RR.VERSION)
        self.assertIn(v, ("V4.8.1", "V4.8.2", "V4.8.3", "V4.8.4", "V4.8.5"))
        self.assertEqual(AG.MARKER, "V4.8.1 AGX GRIPPER GEOMETRY + COLLISION")


if __name__ == "__main__":
    unittest.main(verbosity=1)
