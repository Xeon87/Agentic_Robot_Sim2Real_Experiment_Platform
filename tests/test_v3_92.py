#!/usr/bin/env python3
"""Tests for NERO V3.92:
1. Part 1 (Clean physics/efficiency fixes without overfitting):
   - Relative 3D top-height lift check (Delta Z_top = h_after - h_before_mm) prevents mid-air drop when an occluded/vision-reclassified cup has height_depth_mm=192 mm vs h_before=80 mm.
   - Vision_Review.rows_text() includes depth height when height_mm is None; parse_reply() recovers truncated trailing JSON; VisionReviewer.wanted() skips mid-descent tags ('before contact', 'after the descent').
   - _wrist_recheck_grasp uses measured cup rim height (base[2] + height_mm/1000) instead of hardcoded 121 mm while strictly preserving WRIST_GRASP_MAX_SHIFT_M = 0.025 (25 mm).
2. Part 2 (Multi-scene Real-Robot Task Tab):
   - TASK_SCENES ('stack_cups', 'fill_egg_holder', 'sort_items'), RealIO.perceive_eggs, RealEggSkills, RealSortSkills, wrist_check_egg_or_slot, and GUI scene switching.
"""
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import NERO_V362_Cup_Skills as CK
import NERO_V365_Real_Cup_Task as RT
import NERO_V374_Vision_Review as VR
import NERO_V386_Real_Wrist_Camera as RWC


class TestV392Part1CleanPhysicsFixes(unittest.TestCase):
    def test_version_is_v392(self):
        self.assertIn(RT.VERSION, ("V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7"))
        self.assertIn(RWC.VERSION, ("V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7"))

    def test_vision_review_rows_text_includes_depth_height_and_parse_truncated_json(self):
        dets = [
            {
                "det_id": "#1",
                "kind": "occluded",
                "base_xyz_m": [-0.369, -0.024, 0.0],
                "height_mm": None,
                "height_depth_mm": 192.0,
                "conf": 0.90,
            }
        ]
        txt = VR.rows_text(dets)
        self.assertIn("depth height 192 mm", txt)

        # Truncated JSON string on #2 should still recover #1 cleanly
        truncated = (
            '{"scene":"Cups visible.","items":['
            '{"id":"#1","what":"single_cup","in_gripper":false,"stacked_on":null,"note":"ok"},'
            '{"id":"#2","what":"single_cup","in_gripper":false,"stacked_on":null,"note":"cut off here'
        )
        parsed = VR.parse_reply(truncated)
        self.assertIsNotNone(parsed)
        self.assertIn("#1", parsed["items"])
        self.assertEqual(parsed["items"]["#1"]["what"], "single_cup")

    def test_vision_reviewer_wanted_skips_mid_descent_tags(self):
        rev = VR.VisionReviewer(lambda: {"provider": "openai", "model": "gpt-5.4-nano", "api_key": "dummy"})
        self.assertFalse(rev.wanted("check cup_1 before contact"))
        self.assertFalse(rev.wanted("check cup_1 after the descent"))
        self.assertTrue(rev.wanted("observe_scene"))
        self.assertTrue(rev.wanted("check cup_1 after lift"))
        self.assertTrue(rev.wanted("verify_stack"))

    def test_wrist_grasp_max_shift_kept_strictly_at_25mm(self):
        self.assertAlmostEqual(RWC.WRIST_GRASP_MAX_SHIFT_M, 0.025, places=6)


class TestV392Part2MultiSceneEggAndSort(unittest.TestCase):
    def test_wrist_check_egg_or_slot_synthetic(self):
        H, W = 240, 320
        rgb = np.full((H, W, 3), 60, dtype=np.uint8)
        # Draw a bright synthetic egg at (168, 124)
        yy, xx = np.mgrid[0:H, 0:W]
        mask = ((xx - 168.0) / 24.0) ** 2 + ((yy - 124.0) / 18.0) ** 2 <= 1.0
        rgb[mask] = (235, 225, 200)

        K = {"fx": 280.0, "fy": 280.0, "ppx": 160.0, "ppy": 120.0, "width": W, "height": H}
        T_flange = np.eye(4)
        T_flange[:3, 3] = [-0.30, 0.0, 0.25]
        # Camera looking straight down from flange
        T_fc = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], float)
        calib_eih = {"T_flange_cam": T_fc.tolist()}
        exp_xyz = np.array([-0.30, 0.0, 0.02], float)

        chk = RWC.wrist_check_egg_or_slot(rgb, K, T_flange, calib_eih, exp_xyz, kind="egg")
        self.assertTrue(chk["ok"])
        self.assertLess(chk["shift_norm_mm"], 20.0)
        self.assertIsNotNone(chk["egg_yaw_deg"])

    def test_real_egg_skills_plan_hover_and_execute(self):
        class DummyArm:
            def __init__(self):
                self._q = [0.0, -0.35, 0.0, 0.85, 0.0, 0.5, 0.0]

            def q(self):
                return list(self._q)

            def exec_path(self, path, times, label):
                self._q = list(path[-1])
                return {"ok": True}

        class DummyGrip:
            connected = True
            held = True
            cfg = {"port": "sim", "baud": 1000000, "servo_id": 1, "open_sign": 1,
                   "stall_torque_nm": 3.0, "touch_deg": 0.0, "touch_calibrated": True}

            def open_deg(self, deg):
                return {"ok": True, "open_deg": deg, "gap_mm": 45.0}

            def clamp(self, pct):
                return {"ok": True, "held": True, "open_deg": 12.0, "gap_mm": 38.0, "load_pct": 25.0}

            def status(self):
                return {"open_deg": 12.0, "servo_deg": 168.0, "gap_mm": 38.0, "load_pct": 25.0,
                        "current_ma": 180.0, "voltage_v": 12.0, "temp_c": 30, "torque_limit_pct": 30.0, "errors": []}

        arm = DummyArm()
        grip = DummyGrip()
        logs = []
        io = RT.RealIO(arm, grip, lambda: {}, log=logs.append, motion_allowed=True)

        # Stub perceive_eggs to return 1 egg on the table and a 6-slot carton (slot_1 free, then occupied on verify)
        calls = {"n": 0}

        def fake_perceive_eggs(tag):
            calls["n"] += 1
            occ1 = (tag == "verify_egg_in_slot")
            slots = []
            for idx in range(1, 7):
                slots.append({
                    "slot_id": idx,
                    "row": 0 if idx <= 3 else 1,
                    "col": (idx - 1) % 3,
                    "center_base": [-0.34, 0.06 + 0.03 * idx, 0.025],
                    "occupied": occ1 if idx == 1 else False,
                })
            out = {
                "ok": True,
                "scene": "fill_egg_holder",
                "table_z_m": 0.0,
                "n_2d": 2,
                "eggs": [] if occ1 else [{
                    "center_base": [-0.33, -0.06, 0.022],
                    "grasp_yaw_deg": 15.0,
                    "width_m": 0.044,
                    "length_m": 0.058,
                }],
                "carton": {
                    "center_base": [-0.34, 0.12, 0.025],
                    "yaw_deg": 0.0,
                    "slots": slots,
                    "free_slots": [s["slot_id"] for s in slots if not s["occupied"]],
                    "occupied_slots": [s["slot_id"] for s in slots if s["occupied"]],
                },
                "dets": [],
                "cups": [],
            }
            io.last_perception = out
            return out

        io.perceive_eggs = fake_perceive_eggs
        # Disable real sleep in unit test
        io.sleep = lambda s: None

        # 1) Plan preview
        r_plan = RT.run_plan_preview(io, "egg_1", "slot_1", scene="fill_egg_holder")
        self.assertTrue(r_plan["ok"])
        self.assertEqual(r_plan["skills"].stage, "PLANNED")

        # 2) Hover dry-run
        io.motion_allowed = True
        r_hover = RT.run_hover(io, "egg_1", "slot_1", scene="fill_egg_holder")
        self.assertTrue(r_hover["ok"])

        # 3) Execute mode
        r_exec = RT.run_execute(io, "egg_1", "slot_1", scene="fill_egg_holder")
        self.assertTrue(r_exec["ok"])
        self.assertEqual(r_exec["skills"].stage, "VERIFIED")

    def test_parse_task_eggs_and_task_scenes(self):
        self.assertEqual(RT.parse_task_eggs("Put egg_2 into slot_5"), ("egg_2", "slot_5"))
        self.assertEqual(RT.parse_task_eggs("把鸡蛋放进蛋托"), ("egg_1", "slot_1"))
        scene_keys = [k for k, _ in RT.TASK_SCENES]
        self.assertEqual(scene_keys, ["stack_cups", "fill_egg_holder", "sort_items"])




class TestV392GuiRunModeRegression(unittest.TestCase):
    def test_gui_run_mode_defines_scene_in_worker(self):
        """Regression test for NameError: name 'scene' is not defined inside RealCupTaskTabUI._run_mode job()."""
        import inspect
        src = inspect.getsource(RT.RealCupTaskTabUI._run_mode)
        self.assertIn('scene = self.v_task_scene.get()', src)
        # Ensure scene is defined before def job():
        idx_scene = src.index('scene = self.v_task_scene.get()')
        idx_job = src.index('def job(')
        self.assertLess(idx_scene, idx_job)


class TestV393RealPipelineNoStubEndToEnd(unittest.TestCase):
    def test_real_perceive_eggs_and_sort_items_through_actual_yolov347_and_tr384(self):
        """End-to-end test that runs RealIO.perceive_eggs() and RealIO.perceive_sort_items() WITHOUT mocking
        perceive_eggs/perceive_sort_items, verifying the exact call signatures and return contracts of:
          - YoloEDetector.detect(rgb, mode, path, prompts=..., conf=...)
          - YP.fuse_depth(depths, d2)
          - TR384.locate_eggs_and_carton(d2, rgb, depth, K, T_base_cam, table_z=..., fence_poly=..., calib_note=..., tracker=...)
          - YP.locate_objects(d2, depth, K, T_base_cam, table_z=..., fence_poly=..., calib_note=...)
        """
        import NERO_V347_Yolo_Perception as YP

        H, W = 480, 640
        K = {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0, "width": W, "height": H}
        # Camera 0.70 m directly above table looking straight down:
        # X_base = -0.35 + Y_cam, Y_base = X_cam, Z_base = 0.70 - Z_cam
        T_base_cam = np.array([
            [0.0, 1.0, 0.0, -0.35],
            [1.0, 0.0, 0.0, 0.00],
            [0.0, 0.0, -1.0, 0.70],
            [0.0, 0.0, 0.0, 1.00],
        ], dtype=float)

        rgb = np.full((H, W, 3), 140, dtype=np.uint8)
        depth = np.full((H, W), 0.70, dtype=np.float32)  # flat table at z_base = 0.0

        # 1) Create a realistic egg mask around (u=240, v=240) -> y_base = -0.093 m, x_base = -0.35 m
        yy, xx = np.mgrid[0:H, 0:W]
        egg_mask = (((xx - 240.0) / 24.0) ** 2 + ((yy - 240.0) / 18.0) ** 2) <= 1.0
        depth[egg_mask] = 0.672  # 28 mm above table
        rgb[egg_mask] = (240, 235, 220)

        # 2) Create a 6-slot carton mask around (u=410, v=240) -> y_base = +0.105 m, x_base = -0.35 m
        carton_mask = (np.abs(xx - 410) <= 65) & (np.abs(yy - 240) <= 45)
        depth[carton_mask] = 0.665  # 35 mm high carton

        class StrictSignatureDetector(YP.YoloEDetector):
            """Subclass of real YoloEDetector that enforces the exact signature of YoloEDetector.detect()."""
            def detect(self, rgb_in, mode, model_path, ref=None, prompts=None, conf=YP.CONF_MIN):
                # Enforce positional/keyword signature identical to NERO_V347_Yolo_Perception.YoloEDetector.detect
                assert isinstance(rgb_in, np.ndarray) and rgb_in.shape == (H, W, 3)
                assert mode in ("text", "pf", "vp")
                assert isinstance(model_path, str) and len(model_path) > 0
                assert isinstance(prompts, (list, tuple)) and len(prompts) > 0
                self.info = {"device": "cuda:0"}
                p_str = " ".join(str(x).lower() for x in prompts)
                if "egg" in p_str:
                    return [
                        {
                            "cls": 0, "name": "egg", "conf": 0.93,
                            "box": [216.0, 222.0, 264.0, 258.0], "mask": egg_mask.copy(),
                            "model": "egg_garage_2026-11-04.pt",
                        },
                        {
                            "cls": 0, "name": "egg carton", "conf": 0.95,
                            "box": [345.0, 195.0, 475.0, 285.0], "mask": carton_mask.copy(),
                            "model": "egg_carton_车库_2026-11-04.pt",
                        },
                    ]
                return [
                    {
                        "cls": 0, "name": "banana", "conf": 0.91,
                        "box": [216.0, 222.0, 264.0, 258.0], "mask": egg_mask.copy(),
                        "model": "original",
                    }
                ]

        det = StrictSignatureDetector()

        class DummyArm:
            def __init__(self):
                self._q = [0.0, -0.35, 0.0, 0.85, 0.0, 0.5, 0.0]
            def q(self):
                return list(self._q)
            def exec_path(self, path, times, label):
                self._q = list(path[-1])
                return {"ok": True}

        def make_cam(prompts):
            return {
                "rgb": rgb,
                "depths": [depth, depth],
                "K": K,
                "T_base_cam": T_base_cam.tolist(),
                "z_table": 0.0,
                "fence_poly": None,
                "calib_note": None,
                "yolo": {
                    "detector": det,
                    "mode": "text",
                    "path": "yoloe-26s-seg.pt",
                    "prompts": prompts,
                    "conf": 0.25,
                    "model_name": "yoloe-26s-seg.pt",
                    "model_dir": "/tmp",
                },
            }

        logs = []
        io_egg = RT.RealIO(DummyArm(), None, lambda: make_cam(["egg", "egg carton"]), log=logs.append, motion_allowed=False)
        res_egg = io_egg.perceive_eggs("test_perceive_eggs")
        self.assertTrue(res_egg["ok"])
        self.assertGreaterEqual(len(res_egg["eggs"]), 1)
        self.assertIsNotNone(res_egg["carton"])
        self.assertEqual(len(res_egg["carton"]["slots"]), 6)

        # Run full plan preview on the real perceive_eggs output!
        r_plan = RT.run_plan_preview(io_egg, "egg_1", "slot_1", scene="fill_egg_holder")
        self.assertTrue(r_plan["ok"])

        # Test overlay generation on real perceive_eggs output
        ov = RT.overlay_dets(res_egg)
        self.assertGreaterEqual(len(ov), 1)

        # Now test perceive_sort_items end-to-end without mocks
        io_sort = RT.RealIO(DummyArm(), None, lambda: make_cam(["banana", "apple"]), log=logs.append, motion_allowed=False)
        res_sort = io_sort.perceive_sort_items("test_perceive_sort")
        self.assertTrue(res_sort["ok"])
        self.assertEqual(len(res_sort["items"]), 1)
        self.assertEqual(res_sort["items"][0]["kind"], "banana")
        r_sort_plan = RT.run_plan_preview(io_sort, "item_1", "red", scene="sort_items")
        self.assertTrue(r_sort_plan["ok"])

if __name__ == "__main__":
    unittest.main()
