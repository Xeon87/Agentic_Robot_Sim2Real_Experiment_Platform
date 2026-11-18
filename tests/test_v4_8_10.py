#!/usr/bin/env python3
"""V4.8.10 tests: General Pick&Place in MuJoCo simulation."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import mujoco  # type: ignore
    HAS_MUJOCO = True
except Exception:
    HAS_MUJOCO = False


class _MockSimGeneralIO:
    """Kinematic mock of SimGeneralIO for offline verification of GeneralPickPlaceSkills in simulation."""
    name = "sim_general_mock"
    is_sim = True
    is_agx = True
    tip_drop_m = 0.030

    def __init__(self, log=None):
        import NERO_V362_Cup_Skills as CK
        import NERO_V343_Scenes as SC
        self.CK = CK
        self.SC = SC
        self.fast = CK.FastArm()
        self.cfg = {}
        self.stop_ev = threading.Event()
        self.motion_allowed = True
        self.home_q = [0.0] * 7
        self.q = [0.0] * 7
        self.gap_mm = 70.0
        self.hold_active = False
        self.held_body = None
        self.logs = [] if log is None else log
        self.poses = {
            "obj_egg_1": ([-0.43, -0.05, 0.022], [0.0, 0.0, 0.0, 1.0]),
            "obj_egg_2": ([-0.43, 0.05, 0.022], [0.0, 0.0, 0.0, 1.0]),
            "obj_cup_1": ([-0.34, -0.16, 0.1215], [0.0, 0.0, 0.0, 1.0]),
            "obj_cup_2": ([-0.34, 0.16, 0.1215], [0.0, 0.0, 0.0, 1.0]),
        }
        self.last_perception = None

    def emit(self, s):
        self.logs.append(s)

    def sleep(self, s):
        pass

    def gripper_profile(self):
        return self.CK.OFFICIAL_PROFILE

    def current_q(self):
        return list(self.q)

    def move_joints(self, label, q, slow=None):
        self.q = [float(v) for v in q]
        if self.held_body is not None:
            T = self.fast.fk(self.q)
            self.poses[self.held_body] = ([float(T[0, 3]), float(T[1, 3]), float(T[2, 3])], [0.0, 0.0, 0.0, 1.0])
        return {"ok": True, "label": label, "joint_error_deg": 0.1}

    def move_line(self, label, xyz, quat, speed=0.05, slow=None, strict=False):
        r = self.fast.ik([self.q], xyz, self.CK.quat_xyzw_to_R(quat), iters=30)
        if not r or r[0] is None:
            return {"ok": False, "label": label, "reason": "IK_FAIL"}
        self.q = list(r[0])
        if self.held_body is not None:
            self.poses[self.held_body] = ([float(xyz[0]), float(xyz[1]), float(xyz[2])], [0.0, 0.0, 0.0, 1.0])
        return {"ok": True, "label": label, "tcp_error_mm": 0.2}

    def go_home(self, label):
        self.q = list(self.home_q)
        return {"ok": True, "label": label}

    def gripper_open(self, mm):
        self.gap_mm = float(mm)
        if self.held_body is not None:
            # Drop held object onto target / table
            p = list(self.poses[self.held_body][0])
            self.poses[self.held_body] = ([p[0], p[1], max(0.022, p[2] - 0.03)], [0.0, 0.0, 0.0, 1.0])
            self.held_body = None
        self.hold_active = False
        return {"opening_mm": self.gap_mm, "gap_mm": self.gap_mm}

    def gripper_clamp(self, force_n, settle_s=2.2):
        T = self.fast.fk(self.q)
        tcp = T[:3, 3]
        best_b, best_d = None, 0.065
        for b, (bp, _bq) in self.poses.items():
            d = ((tcp[0] - bp[0]) ** 2 + (tcp[1] - bp[1]) ** 2) ** 0.5
            if "cup" in b:
                # Hollow cup (r_top ~ 47.5 mm): grasping the center (d < 25 mm) pinches empty air!
                if 0.030 <= d <= 0.062 and d < best_d:
                    best_b, best_d = b, d
            elif d < best_d:
                best_b, best_d = b, d
        if best_b is not None:
            self.held_body = best_b
            self.hold_active = True
            self.gap_mm = 42.0 if "egg" in best_b else 18.0
        else:
            self.hold_active = False
            self.gap_mm = 0.0
        return {"opening_mm": self.gap_mm, "gap_mm": self.gap_mm, "hold_active": self.hold_active}

    def gripper_state(self):
        return {"opening_mm": self.gap_mm, "gap_mm": self.gap_mm, "hold_active": self.hold_active}

    def perceive_general(self, tag, prompts=None, use_finetuned=True):
        import NERO_V489_General_Pick_Place as GP
        obs = GP._sim_general_perception(lambda b: self.poses.get(b, (None, None)), prompts=prompts)
        self.last_perception = obs
        return obs


class TestV4810GeneralPickPlaceSim(unittest.TestCase):
    def test_01_console_and_scenes_registered(self):
        import NERO_V312_Console as C
        import NERO_V343_Scenes as SC

        self.assertIn(C.VERSION, ("V4.8.10", "V4.8.11", "V4.9", "V4.9.1", "V4.9.2", "V4.9.3", "V4.10", "V4.10.1", "V4.10.2", "V4.10.3", "V4.11", "V4.11.1", "V4.11.2"))
        keys = [k for k, _lbl in C.SIM_SCENES]
        self.assertIn("general_pick_place", keys)
        self.assertIn("general_pick_place", SC.SCENES)

        old_a = os.environ.get("NERO_V343_ACTIVE_SCENE")
        old_s = os.environ.get("NERO_SCENE")
        try:
            os.environ["NERO_SCENE"] = "general_pick_place"
            os.environ["NERO_V343_ACTIVE_SCENE"] = "general_pick_place"
            self.assertEqual(SC.active_scene(), "sort_items")
            self.assertEqual(SC.selected_variant(), "general_pick_place")
            self.assertEqual(set(SC.tf_body_map().keys()), set(SC.GENERAL_OBJECT_NAMES))
            for n in SC.GENERAL_OBJECT_NAMES:
                self.assertIn(n, SC.hold_object_names())
            self.assertEqual(SC.planning_rotated_boxes(None, None), [])
        finally:
            if old_a is None:
                os.environ.pop("NERO_V343_ACTIVE_SCENE", None)
            else:
                os.environ["NERO_V343_ACTIVE_SCENE"] = old_a
            if old_s is None:
                os.environ.pop("NERO_SCENE", None)
            else:
                os.environ["NERO_SCENE"] = old_s

    def test_02_general_scene_xml_valid(self):
        import NERO_V343_Scenes as SC
        old = os.environ.get("NERO_V343_ACTIVE_SCENE")
        old_s = os.environ.get("NERO_SCENE")
        try:
            os.environ["NERO_SCENE"] = "general_pick_place"
            os.environ["NERO_V343_ACTIVE_SCENE"] = "general_pick_place"
            with tempfile.TemporaryDirectory() as td:
                src = Path(td) / "nero_motor.xml"
                src.write_text('<mujoco model="nero"><asset/><worldbody><body name="link7"/></worldbody></mujoco>',
                               encoding="utf-8")
                out = Path(SC.make_store_in_safe_xml(str(src)))
                self.assertTrue(out.exists())
                tree = ET.parse(out)
                bodies = {b.get("name") for b in tree.findall(".//body")}
                for n in SC.GENERAL_OBJECT_NAMES:
                    self.assertIn(n, bodies)
                eq_names = {e.get("name") for e in tree.findall(".//equality/weld")}   # V4.10.2: weld (rigid) holds
                for n in SC.GENERAL_OBJECT_NAMES:
                    self.assertIn(f"hold_{n}", eq_names)
        finally:
            if old is None:
                os.environ.pop("NERO_V343_ACTIVE_SCENE", None)
            else:
                os.environ["NERO_V343_ACTIVE_SCENE"] = old
            if old_s is None:
                os.environ.pop("NERO_SCENE", None)
            else:
                os.environ["NERO_SCENE"] = old_s

    def test_03_mock_sim_egg_into_cup_and_stack_cup(self):
        import NERO_V489_General_Pick_Place as GP

        # 1) Egg into cup
        io = _MockSimGeneralIO()
        sk = GP.GeneralPickPlaceSkills(io)
        obs = sk.perceive_objects(prompts=["egg", "cup"], use_finetuned=True)
        self.assertTrue(obs["ok"])
        self.assertTrue({"egg_1", "egg_2", "cup_1", "cup_2"}.issubset({o["object_id"] for o in obs["objects"]}))
        # V4.9: numeric parameters (no grasp_mode / place_mode); the cup-stack half of this test moved to
        # tests/test_v4_9.py (contact-based hold, measured geometry) - the box-only mock perception makes a cup solid
        self.assertTrue(sk.plan_grasp_chunk(object_id="egg_1", xy_offset_mm=[0.0, 0.0], grasp_z_offset_from_top_mm=-5.0,
                                            closing_yaw_deg=90.0, pitch_deg=0.0, open_mm=60.0, yaw_tolerance_deg=40.0,
                                            pregrasp_height_mm=70.0, lift_height_mm=120.0)["ok"])
        self.assertTrue(sk.execute_grasp_chunk(clamp_force_n=4.2)["ok"])
        self.assertTrue(sk.plan_place_chunk(target_id_or_xyz="cup_1", xy_offset_mm=[0.0, 0.0],
                                            held_bottom_above_target_top_mm=30.0, yaw_change_deg=0.0,
                                            approach_height_mm=60.0, release_open_mm=65.0)["ok"])
        self.assertTrue(sk.execute_place_chunk(retreat_up_mm=70.0)["ok"])
        self.assertTrue(sk.verify_and_park_chunk(action="clear_view_and_perceive", prompts=["egg", "cup"])["ok"])
        self.assertEqual(sk.stage, "VERIFIED")
        egg_p = io.poses["obj_egg_1"][0]
        cup_p = io.poses["obj_cup_1"][0]
        self.assertLess(((egg_p[0] - cup_p[0]) ** 2 + (egg_p[1] - cup_p[1]) ** 2) ** 0.5, 0.020)

    @unittest.skipUnless(HAS_MUJOCO, "mujoco package not installed in sandbox")
    def test_04_twin_egg_into_cup_e2e(self):
        import NERO_V489_General_Pick_Place as GP
        io = GP.TwinGeneralIO()
        sk = GP.GeneralPickPlaceSkills(io)
        io.skills = sk
        self.assertTrue(sk.perceive_objects(prompts=["egg", "cup"], use_finetuned=True)["ok"])
        # V4.9.3: V4.9 API (no grasp modes / hints; the old call put the fingertips 4.7 mm into the table)
        self.assertTrue(sk.plan_grasp_chunk(object_id="egg_1", xy_offset_mm=[0, 0], grasp_z_offset_from_top_mm=-9,
                                            closing_yaw_deg=90, pitch_deg=0, open_mm=60, yaw_tolerance_deg=30,
                                            pregrasp_height_mm=80, lift_height_mm=150)["ok"])
        self.assertTrue(sk.execute_grasp_chunk(clamp_force_n=8.0)["ok"])
        self.assertTrue(sk.plan_place_chunk(target_id_or_xyz="cup_1", xy_offset_mm=[0, 0],
                                            held_bottom_above_target_top_mm=15, yaw_change_deg=0,
                                            approach_height_mm=40, release_open_mm=60, carry_lift_mm=0)["ok"])
        self.assertTrue(sk.execute_place_chunk(retreat_up_mm=60)["ok"])
        self.assertTrue(sk.verify_and_park_chunk(action="clear_view_and_perceive", prompts=["egg", "cup"])["ok"])

    def test_05_gui_run_sim_task_contract(self):
        import NERO_V489_General_Pick_Place as GP

        logs = []
        bg_calls = []

        class _FakeCore:
            def emit(self, s):
                logs.append(s)

        class _FakeAgent:
            pass

        ctx = {
            "skill_core": _FakeCore(),
            "agent": _FakeAgent(),
            "cfg_fn": lambda: {"task": "pick up the egg and put it into the cup", "model": "gpt-5.6-luna",
                               "api_key": "k", "reasoning_effort": "low", "max_tool_steps": 30, "dry_run": True},
            "background": lambda name, fn, lock_motion=True: bg_calls.append((name, fn, lock_motion)),
            "log": logs.append,
            "safety_gate": lambda: True,
            "start_check": lambda *_: None,
            "shared": {},
        }
        GP.gui_run_sim_task(ctx)
        self.assertEqual(len(bg_calls), 1)
        self.assertEqual(bg_calls[0][0], "RUN TASK (general_pick_place)")



    def test_06_sim_general_io_tf_prefix_and_no_truth_bypass(self):
        import NERO_V489_General_Pick_Place as GP

        class _FakeTransform:
            def __init__(self, xyz):
                class _T:
                    x, y, z = xyz
                class _R:
                    x, y, z, w = 0.0, 0.0, 0.0, 1.0
                class _TR:
                    translation = _T()
                    rotation = _R()
                self.transform = _TR()

        class _RealBackendLikeCore:
            """Mimics NERO_V71_Hybrid_GPU_Planner_Core + _v329_object_pose in camera mode."""
            def __init__(self):
                self.tf_table = {
                    "sim_obj_egg_1": [-0.43, -0.05, 0.022],
                    "sim_obj_egg_2": [-0.43, 0.05, 0.022],
                    "sim_obj_cup_1": [-0.34, -0.16, 0.1215],
                    "sim_obj_cup_2": [-0.34, 0.16, 0.1215],
                }
                self.logs = []

            def emit(self, s):
                self.logs.append(s)

            def wait_for_tf(self, child, timeout_sec=0.35):
                if child not in self.tf_table:
                    raise RuntimeError(f"TF unavailable: base_link -> {child}")
                return _FakeTransform(self.tf_table[child])

            def object_pose(self, object_name):
                # Mimic _v329_object_pose blocking egg_* in camera mode AND V71 prefixing sim_obj_
                if str(object_name).startswith("egg"):
                    raise RuntimeError(f"V3.29 NO-TRUTH: {object_name} has no perceived pose yet")
                return self.wait_for_tf(f"sim_obj_{object_name}").transform.translation, [0, 0, 0, 1]

        core = _RealBackendLikeCore()
        io = GP.SimGeneralIO({"skill_core": core, "shared": {}})
        sk = GP.GeneralPickPlaceSkills(io)
        obs = sk.perceive_objects(prompts=["egg", "cup"], use_finetuned=True)
        self.assertTrue(obs["ok"])
        by_id = {o["object_id"]: o["label"] for o in obs["objects"]}
        self.assertEqual(by_id, {"egg_1": "egg", "egg_2": "egg", "cup_1": "cup", "cup_2": "cup"})



    def test_07_latch_recovery_and_smooth_descent(self):
        import NERO_V489_General_Pick_Place as GP

        io = _MockSimGeneralIO()
        latched = {"val": True, "cleared": 0}
        io.latch_pending = lambda: latched["val"]

        def _clear(lbl):
            latched["val"] = False
            latched["cleared"] += 1
            return True, "unlatched"

        io.clear_latch = _clear
        sk = GP.GeneralPickPlaceSkills(io)
        self.assertTrue(sk.perceive_objects(prompts=["egg", "cup"], use_finetuned=True)["ok"])
        self.assertTrue(sk.plan_grasp_chunk(object_id="egg_1", xy_offset_mm=[0.0, 0.0], grasp_z_offset_from_top_mm=-5.0,
                                            closing_yaw_deg=90.0, pitch_deg=0.0, open_mm=60.0, yaw_tolerance_deg=40.0,
                                            pregrasp_height_mm=70.0, lift_height_mm=120.0)["ok"])
        self.assertTrue(sk.execute_grasp_chunk(clamp_force_n=4.2)["ok"])
        self.assertGreaterEqual(latched["cleared"], 1)
        self.assertFalse(latched["val"])

        # 1) Simulate safety latch while STILL holding egg_1: verify_and_park_chunk clears latch and reports OBJECT_STILL_HELD
        latched["val"] = True
        r_vp = sk.verify_and_park_chunk(action="clear_view_and_perceive", prompts=["egg", "cup"])
        self.assertFalse(r_vp["ok"])
        self.assertEqual(r_vp["reason"], "OBJECT_STILL_HELD")
        self.assertTrue(r_vp["physical_verification"].get("latch_cleared"))
        self.assertFalse(latched["val"])

        # 2) Simulate safety stop that DROPPED egg_1 (hold_active=False) and knocked cup_1 over (90 deg tilt):
        #    _sync_held_state detects the drop so verify_and_park_chunk succeeds and reports cup_1 upright=False!
        io.hold_active = False
        io.held_body = None
        io.poses["obj_cup_1"] = ([-0.34, -0.16, 0.048], [0.0, 0.7071, 0.0, 0.7071])
        r_vp2 = sk.verify_and_park_chunk(action="clear_view_and_perceive", prompts=["egg", "cup"])
        self.assertTrue(r_vp2["ok"])
        cup1_obs = [o for o in r_vp2["physical_verification"]["visible_objects"] if o["object_id"] == "cup_1"][0]
        self.assertFalse(cup1_obs["upright"])
        self.assertGreater(cup1_obs["tilt_deg"], 60.0)


if __name__ == "__main__":
    unittest.main()
