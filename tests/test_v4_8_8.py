#!/usr/bin/env python3
"""NERO V4.8.8 Agentic Real-Robot Pipeline Risk Fixes & Optimizations Test Suite."""
import os
import sys
import tempfile
import types
import unittest
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTS = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import NERO_V48_Gripper as GG
import NERO_V362_Cup_Skills as CK
import NERO_V365_Real_Cup_Task as RT
import NERO_V481_Agx_Geom as AG
import NERO_V483_Tidy as TD
import NERO_V4613_Tool_Collision as TC
import test_v4_6_13 as T13

USER_CAM = np.array([[1, 0, 0, 0.1221], [0, 1, 0, -0.0537], [0, 0, 1, -0.0250], [0, 0, 0, 1.0]])


class TestV488AgenticPipelineAuditFixes(unittest.TestCase):
    def setUp(self):
        self._orig_cfg_path = GG.CFG_PATH
        self._tmp_dir = tempfile.mkdtemp(prefix="nero_v488_")
        GG.CFG_PATH = os.path.join(self._tmp_dir, "gripper.json")
        self._orig_find_handeye = TC.find_handeye
        TC.find_handeye = lambda calib_dir=None: (USER_CAM.copy(), "user")

    def tearDown(self):
        GG.CFG_PATH = self._orig_cfg_path
        TC.find_handeye = self._orig_find_handeye

    def _set_gripper(self, kind="agx_gripper", fingers="stock"):
        cfg = GG.load_cfg()
        cfg["type"] = kind
        cfg.setdefault("agx", {})["fingers"] = fingers
        cfg["agx"]["tip_from_flange_mm"] = 144.0
        GG.save_cfg(cfg)

    def test_1_gripper_boxes_agx_stock_and_softfin_vs_so101(self):
        """Fix 1: RealIO.gripper_boxes returns AgxToolModel boxes for agx_gripper/Softfin and SO-101 box for so101."""
        arm = types.SimpleNamespace(q=lambda: [0.0, 0.4, 0.0, 1.2, 0.0, 0.3, 0.0],
                                    model={"joints": [f"joint{i}" for i in range(1, 8)], "limits": CK.JOINT_LIMITS})
        # 1) AgxGripper stock
        self._set_gripper("agx_gripper", "stock")
        io_agx = RT.RealIO(arm, None, lambda: {}, log=lambda s: None, motion_allowed=False)
        boxes_full = io_agx.gripper_boxes(tight=False)
        boxes_tight = io_agx.gripper_boxes(tight=True)
        self.assertGreaterEqual(len(boxes_full), 3)   # housing + finger_a + finger_b (+ bracket/cam)
        self.assertEqual(len(boxes_tight), 2)         # finger_a + finger_b
        # 2) AgxGripper softfin
        self._set_gripper("agx_gripper", "softfin")
        io_sf = RT.RealIO(arm, None, lambda: {}, log=lambda s: None, motion_allowed=False)
        boxes_sf_full = io_sf.gripper_boxes(tight=False)
        boxes_sf_tight = io_sf.gripper_boxes(tight=True)
        self.assertGreaterEqual(len(boxes_sf_full), 7)  # housing + finger_a/b + finger_root_a/b + adapter_a/b
        self.assertEqual(len(boxes_sf_tight), 2)
        # 3) SO-101 fallback
        self._set_gripper("so101", "stock")
        io_so = RT.RealIO(arm, None, lambda: {}, log=lambda s: None, motion_allowed=False)
        self.assertEqual(len(io_so.gripper_boxes(tight=False)), 1)
        self.assertEqual(len(io_so.gripper_boxes(tight=True)), 1)

    def test_2_egg_cfg_offcentre_thresholds_for_stock_and_softfin(self):
        """Fix 2: Softfin (8..46 mm) and stock AgxGripper (30..48 mm) do not false-trigger off-centre grasp."""
        arm = types.SimpleNamespace(q=lambda: [0.0, 0.4, 0.0, 1.2, 0.0, 0.3, 0.0],
                                    model={"joints": [f"joint{i}" for i in range(1, 8)], "limits": CK.JOINT_LIMITS})
        # Stock AgxGripper: held_gap_mm == 42.0, min_held_gap_mm == 30.0 -> offcentre_margin_mm == 12.0 (threshold 30.0 mm)
        self._set_gripper("agx_gripper", "stock")
        io_st = RT.RealIO(arm, None, lambda: {}, log=lambda s: None, motion_allowed=False)
        sk_st = RT.RealEggSkills(io_st)
        cfg_st = sk_st._egg_cfg()
        self.assertEqual(cfg_st["held_gap_mm"], 42.0)
        self.assertAlmostEqual(cfg_st["held_gap_mm"] - cfg_st["offcentre_margin_mm"], 30.0, places=1)
        self.assertFalse(32.0 < cfg_st["held_gap_mm"] - cfg_st["offcentre_margin_mm"])

        # Softfin AgxGripper: min_held_gap_mm == 8.0 -> threshold 8.0 mm
        self._set_gripper("agx_gripper", "softfin")
        io_sf = RT.RealIO(arm, None, lambda: {}, log=lambda s: None, motion_allowed=False)
        sk_sf = RT.RealEggSkills(io_sf)
        cfg_sf = sk_sf._egg_cfg()
        self.assertEqual(cfg_sf["held_gap_mm"], 42.0)
        self.assertAlmostEqual(cfg_sf["held_gap_mm"] - cfg_sf["offcentre_margin_mm"], 8.0, places=1)
        self.assertFalse(18.0 < cfg_sf["held_gap_mm"] - cfg_sf["offcentre_margin_mm"])

    def test_3_run_hover_executes_transit_and_egg_open_mm(self):
        """Fix 3: run_hover on fill_egg_holder opens to egg_open_mm() and executes plan['_transit'] steps."""
        self._set_gripper("agx_gripper", "stock")
        sk0, io = T13.user_scene()
        opened_mm = []
        moved_labels = []
        io.grip = types.SimpleNamespace(open_deg=lambda a: {"ok": True})
        io.gripper_open = lambda mm: (opened_mm.append(float(mm)), {"ok": True, "opening_mm": float(mm)})[1]
        orig_move_joints = io.move_joints

        def rec_move_joints(label, q, **kw):
            moved_labels.append(label)
            return orig_move_joints(label, q, **kw)

        io.move_joints = rec_move_joints
        io.sleep = lambda s: None
        orig_make_skills = RT.make_skills
        def wrapped_make_skills(*a, **kw):
            sk_ = orig_make_skills(*a, **kw)
            orig_plan = sk_.plan_egg_place
            def plan_with_transit(*pa, **pkw):
                res_p = orig_plan(*pa, **pkw)
                if sk_.plan and not sk_.plan.get("_transit"):
                    sk_.plan["_transit"] = [{"kind": "joints", "label": "carry high", "q": sk_.plan["_q"]["above_target"]}]
                return res_p
            sk_.plan_egg_place = plan_with_transit
            return sk_
        RT.make_skills = wrapped_make_skills
        try:
            out = RT.run_hover(io, "egg_1", "slot_1", scene="fill_egg_holder")
        finally:
            RT.make_skills = orig_make_skills
        self.assertTrue(out["ok"], out.get("steps"))
        self.assertTrue(opened_mm and abs(opened_mm[0] - RT.egg_open_mm()) < 1e-6)
        self.assertTrue(any("V4.8.8 hover" in lbl for lbl in moved_labels), moved_labels)

    def test_4_real_tidy_conservative_table_z_and_shifted_world_boxes(self):
        """Fix 4: RealTidySkills.ingest takes conservative_table_z (max of camera and fence) and shifts world_boxes."""
        self._set_gripper("agx_gripper", "stock")
        logs = []
        io = types.SimpleNamespace(
            emit=logs.append,
            current_q=lambda: [0.0, 0.4, 0.0, 1.2, 0.0, 0.3, 0.0],
            tcp_pose=lambda: ([-0.35, 0.0, 0.25], [0.0, 1.0, 0.0, 0.0]),
            gripper_state=lambda: {"gap_mm": 60.0, "opening_mm": 60.0},
            gripper_profile=lambda: CK.OFFICIAL_PROFILE,
            pose_mode=lambda: "camera",
            last_cam={"z_table": 0.005, "K": {"fx": 600, "fy": 600, "ppx": 320, "ppy": 240}, "T_base_cam": np.eye(4)},
            perceive_sort_items=lambda tag: {
                "ok": True, "scene": "sort_items",
                "table_z_m": -0.005,          # Camera says -5 mm
                "fence_z_table_m": 0.005,     # Safety fence says +5 mm (10 mm higher)
                "items": [{"kind": "banana", "conf": 0.92, "rim_xyz_m": [-0.38, -0.05, 0.015], "contour": []}],
            },
        )
        sk = TD.make_real_tidy_skills(io, log=logs.append)
        r = sk.observe_scene()
        self.assertTrue(r["ok"])
        self.assertAlmostEqual(sk.table_z, 0.005, places=4)
        banana_c = sk.cam["obj_banana"]["c"]
        import NERO_V343_Scenes as SC
        self.assertAlmostEqual(float(banana_c[2]), 0.005 + SC.SORT_ITEMS["obj_banana"]["cz"], places=4)

    def test_5_wrist_recheck_guard_rejects_fence_violating_shift(self):
        """Fix 5: _wrist_recheck_egg_grasp & _wrist_recheck_slot_place reject shifts that fail _check_path_fence."""
        self._set_gripper("agx_gripper", "stock")
        sk, io = T13.user_scene()
        r = sk.plan_egg_place(source_egg="egg_1", target_slot="slot_1")
        self.assertTrue(r["ok"])
        W = {k: (v[0].copy(), v[1].copy()) for k, v in sk.plan["_wps"].items()}
        Q = {k: list(v) for k, v in sk.plan["_q"].items()}
        p_pre_orig = W["pregrasp"][0].copy()
        io.wrist_snapshot = lambda: {"rgb": np.zeros((60, 60, 3), np.uint8), "K": {"fx": 500, "fy": 500, "ppx": 30, "ppy": 30},
                                     "calib_eih": np.eye(4)}
        import NERO_V386_Real_Wrist_Camera as RWC
        orig_chk = RWC.wrist_check_egg_or_slot
        orig_fence = sk._check_path_fence
        try:
            RWC.wrist_check_egg_or_slot = lambda *a, **kw: {
                "ok": True, "trusted": True, "delta_xy_m": [0.003, 0.002], "shift_norm_mm": 3.6
            }
            sk._check_path_fence = lambda qs, zt: False
            chk = sk._wrist_recheck_egg_grasp("egg_1", W, Q)
            self.assertTrue(chk.get("guard_rejected"))
            np.testing.assert_allclose(W["pregrasp"][0], p_pre_orig)
        finally:
            RWC.wrist_check_egg_or_slot = orig_chk
            sk._check_path_fence = orig_fence

    def test_6_cup_stack_path_validator_filters_unsafe_angles(self):
        """Fix 6: plan_cup_stack honors path_validator to filter out rim angles that fail tool/fence checks."""
        knobs = CK.cup_knobs()[0]
        calls = []

        def strict_validator(qs, zt):
            calls.append(len(qs))
            return len(calls) > 2

        res = CK.plan_cup_stack(
            knobs, [-0.42, 0.10, 0.1226], [1, 0, 0, 0], [-0.32, -0.12, 0.1226], [1, 0, 0, 0],
            q_start=[0.0, 0.4, 0.0, 1.2, 0.0, 0.3, 0.0],
            profile=CK.OFFICIAL_PROFILE,
            path_validator=strict_validator,
        )
        self.assertTrue(res.get("ok"))
        self.assertGreater(len(calls), 2)

    def test_7_fast_tl7_prescreen_skips_ik_when_tool_collides_with_table(self):
        """Fix 7: _slot_pass pre-screens T_l7 with _tool_model().clear_of_table before invoking IK."""
        self._set_gripper("agx_gripper", "stock")
        sk, io = T13.user_scene()
        ik_calls = []
        orig_ik = sk._fast.ik

        def counting_ik(*a, **kw):
            ik_calls.append(1)
            return orig_ik(*a, **kw)

        sk._fast.ik = counting_ik
        sk._tool_cache = types.SimpleNamespace(
            ok=True, note="mock tool",
            clear_of_table=lambda T_l7, zt: (False, 2.5, "camera_head")
        )
        why = {}

        def _rej(pitch, key, extra=None):
            d = why.setdefault("+" if pitch > 0 else "-", {"ik": 0, "joint": 0, "fence": 0, "tool": 0, "carton": 0, "tool_min": None})
            d[key] += 1
            if key == "tool" and extra is not None:
                d["tool_min"] = extra

        cands = []
        slot = sk.cam["slot_1"]
        sk._slot_pass(cands, (-90.0,), (180.0, 150.0), [[0.0]*7], np.array([-0.34, 0.12, 0.02]),
                      np.array([0.024, 0.0, 0.0]), 0.0275, [], [0.0]*7, 0.0, slot["rim"], _rej)
        self.assertEqual(len(cands), 0)
        self.assertEqual(len(ik_calls), 0)
        self.assertEqual(why["-"]["tool"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
