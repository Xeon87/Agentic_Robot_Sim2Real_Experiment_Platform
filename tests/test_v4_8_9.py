#!/usr/bin/env python3
"""NERO V4.8.9 offline tests: General pick&place (LLM chunk orchestration) + auto-close gripper zeroing."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
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
import NERO_V489_General_Pick_Place as GPP


class FakeGeneralIO:
    def __init__(self, logs=None):
        self.CK = CK
        self.fast = CK.FastArm()
        self.cfg = dict(RT.DEFAULT_CFG)
        self.logs = logs if logs is not None else []
        self.stop_ev = threading.Event()
        self.motion_allowed = True
        self.q_now = [0.0, 0.4, 0.0, 1.2, 0.0, 0.3, 0.0]
        self.home_q = list(self.q_now)
        self.moves = []
        self.opened = []
        self.clamped = []
        self.held_gap = 38.5
        self.held = False
        self.last_perception = None

    def emit(self, s):
        self.logs.append(str(s))

    def sleep(self, s):
        pass

    def current_q(self):
        return list(self.q_now)

    def gripper_profile(self):
        return CK.OFFICIAL_PROFILE

    def move_joints(self, label, q, **kw):
        self.q_now = list(q)
        self.moves.append(("joints", label))
        return {"ok": True, "label": label}

    def move_line(self, label, xyz, quat, speed=0.05, **kw):
        self.moves.append(("line", label, [round(float(v), 4) for v in xyz]))
        return {"ok": True, "label": label}

    def go_home(self, label="home"):
        self.q_now = list(self.home_q)
        self.moves.append(("home", label))
        return {"ok": True}

    def gripper_open(self, mm):
        self.opened.append(float(mm))
        self.held = False
        return {"ok": True, "opening_mm": float(mm), "gap_mm": float(mm)}

    def gripper_clamp(self, force_n):
        self.clamped.append(float(force_n))
        self.held = True
        return {"ok": True, "hold_active": True, "gap_mm": self.held_gap}

    def gripper_state(self):
        return {"hold_active": self.held, "gap_mm": self.held_gap if self.held else 60.0}

    def perceive_general(self, tag, prompts=None, use_finetuned=True):
        if not self.held and not any(m[1].startswith("V4.8.9 clear view") for m in self.moves):
            objs = [
                {"det_id": "egg_1", "kind": "egg", "center_xyz_m": [-0.36, -0.10, 0.021], "top_z_m": 0.042,
                 "height_mm": 42.0, "size_mm": [55.0, 42.0], "yaw_deg": -20.0, "conf": 0.92, "trusted": True},
                {"det_id": "cup_1", "kind": "cup", "center_xyz_m": [-0.32, 0.12, 0.055], "top_z_m": 0.110,
                 "height_mm": 110.0, "size_mm": [95.0, 95.0], "yaw_deg": 0.0, "conf": 0.94, "trusted": True},
            ]
        else:
            objs = [
                {"det_id": "cup_1", "kind": "cup", "center_xyz_m": [-0.32, 0.12, 0.055], "top_z_m": 0.110,
                 "height_mm": 110.0, "size_mm": [95.0, 95.0], "yaw_deg": 0.0, "conf": 0.94, "trusted": True},
            ]
        out = {"ok": True, "scene": "general_pick_place", "table_z_m": 0.0, "fence_z_table_m": 0.0,
               "objects": objs, "dets": objs, "cups": objs}
        self.last_perception = out
        return out


class FakeLLMAgent:
    """Simulates LLM designing each chunk dynamically for 'pick up the egg and put it into the cup'."""
    def __init__(self):
        self._lock = threading.Lock()
        self._running = False
        self._stop_event = threading.Event()
        self._last_response_id = None
        self.step = 0
        self.provider = "openai"
        self.seq = [
            ("perceive_objects", {"prompts": ["egg", "cup"], "use_finetuned": True}),
            # V4.9: numeric grasp / place parameters only (no grasp_mode / place_mode)
            ("plan_grasp_chunk", {"object_id": "egg_1", "xy_offset_mm": [0.0, 0.0], "grasp_z_offset_from_top_mm": -5.0,
                                  "closing_yaw_deg": 90.0, "pitch_deg": 0.0, "open_mm": 60.0, "yaw_tolerance_deg": 40.0,
                                  "pregrasp_height_mm": 70.0, "lift_height_mm": 120.0}),
            ("execute_grasp_chunk", {"clamp_force_n": 4.2}),
            ("plan_place_chunk", {"target_id_or_xyz": "cup_1", "xy_offset_mm": [0.0, 0.0],
                                  "held_bottom_above_target_top_mm": 30.0, "yaw_change_deg": 0.0,
                                  "approach_height_mm": 60.0, "release_open_mm": 60.0, "carry_lift_mm": 0.0}),
            ("execute_place_chunk", {"retreat_up_mm": 80.0}),
            ("verify_and_park_chunk", {"action": "clear_view_and_perceive", "prompts": ["egg", "cup"]}),
        ]

    def api_key_status(self, api_key, model):
        return True, "ok"

    def _post_responses(self, payload, api_key):
        if self.step < len(self.seq):
            name, args = self.seq[self.step]
            self.step += 1
            return {"id": f"resp_{self.step}", "calls": [{"name": name, "arguments": json.dumps(args), "call_id": f"c_{self.step}"}]}
        return {"id": "resp_done", "calls": [], "text": "Put egg_1 into cup_1 successfully."}

    def _log_usage(self, resp, model):
        pass

    def _function_calls(self, resp):
        return resp.get("calls", [])

    def _extract_text(self, resp):
        return resp.get("text", "")


class TestV489GeneralPickPlace(unittest.TestCase):
    def setUp(self):
        self._orig_cfg_path = GG.CFG_PATH
        self._tmp_dir = tempfile.mkdtemp(prefix="nero_v489_")
        GG.CFG_PATH = os.path.join(self._tmp_dir, "gripper.json")
        cfg = GG.load_cfg()
        cfg["type"] = "agx_gripper"
        GG.save_cfg(cfg)

    def tearDown(self):
        GG.CFG_PATH = self._orig_cfg_path

    def test_1_general_pick_place_scene_registered_and_old_scenes_untouched(self):
        keys = [k for k, _ in RT.TASK_SCENES]
        self.assertIn("stack_cups", keys)
        self.assertIn("fill_egg_holder", keys)
        self.assertIn("general_pick_place", keys)
        self.assertIn("general_pick_place", RT.SCENE_DEFAULT_PROMPTS)

    def test_2_llm_orchestrates_egg_into_cup_all_modes(self):
        # 1) Execute mode
        io = FakeGeneralIO()
        ag = FakeLLMAgent()
        out = GPP.run_general_pick_place(io, "pick up the egg and put it into the cup", mode="execute", agent=ag,
                                         llm_cfg={"model": "gpt-5.6-luna"})
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["skills"].stage, "VERIFIED")
        self.assertTrue(any("carry above cup_1" in m[1] for m in io.moves))
        # Release z must be above cup_1 top_z (0.110 + 0.030 = 0.140 m)
        lower_moves = [m for m in io.moves if m[0] == "line" and "lower to cup_1" in m[1]]
        self.assertTrue(lower_moves)
        self.assertGreaterEqual(lower_moves[0][2][2], 0.139)

        # 2) Plan mode (no physical arm motion)
        io_p = FakeGeneralIO()
        io_p.motion_allowed = False
        ag_p = FakeLLMAgent()
        out_p = GPP.run_general_pick_place(io_p, "pick up the egg and put it into the cup", mode="plan", agent=ag_p,
                                           llm_cfg={"model": "gpt-5.6-luna"})
        self.assertTrue(out_p["ok"])
        self.assertEqual(len(io_p.moves), 0)

        # 3) Hover mode (moves without clamping)
        io_h = FakeGeneralIO()
        ag_h = FakeLLMAgent()
        out_h = GPP.run_general_pick_place(io_h, "pick up the egg and put it into the cup", mode="hover", agent=ag_h,
                                           llm_cfg={"model": "gpt-5.6-luna"})
        self.assertTrue(out_h["ok"])
        self.assertEqual(len(io_h.clamped), 0)
        self.assertTrue(any("hover" in m[1] for m in io_h.moves))

    def test_3_realio_perceive_general_returns_egg_and_cup(self):
        class FakeDet:
            device = "cpu"
            info = {"device": "CPU"}
            def detect(self, rgb, mode, path, prompts=None, conf=0.25):
                m1 = np.zeros((480, 640), bool)
                m1[200:250, 200:260] = True
                m2 = np.zeros((480, 640), bool)
                m2[200:290, 400:490] = True
                return [
                    {"cls": 0, "name": "egg", "conf": 0.9, "box": [200, 200, 260, 250], "mask": m1},
                    {"cls": 0, "name": "cup", "conf": 0.92, "box": [400, 200, 490, 290], "mask": m2},
                ]
        K = {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0, "width": 640, "height": 480}
        T_cam = np.eye(4)
        T_cam[:3, :3] = np.diag([1.0, -1.0, -1.0])
        T_cam[:3, 3] = [-0.35, 0.0, 0.70]
        depth = np.full((480, 640), 0.70, np.float32)
        depth[200:250, 200:260] = 0.66
        depth[200:290, 400:490] = 0.60
        cam_fn = lambda: {
            "rgb": np.zeros((480, 640, 3), np.uint8),
            "depths": [depth],
            "K": K,
            "T_base_cam": T_cam,
            "z_table": 0.0,
            "fence_poly": None,
            "yolo": {"detector": FakeDet(), "mode": "text", "path": "yoloe-26s-seg.pt", "prompts": ["egg", "cup"], "conf": 0.25},
        }
        arm = types.SimpleNamespace(q=lambda: [0.0, 0.4, 0.0, 1.2, 0.0, 0.3, 0.0],
                                    model={"joints": [f"joint{i}" for i in range(1, 8)], "limits": CK.JOINT_LIMITS})
        io = RT.RealIO(arm, None, cam_fn, log=lambda s: None, motion_allowed=False)
        res = io.perceive_general("test", prompts=["egg", "cup"])
        self.assertTrue(res["ok"])
        ids = [o["det_id"] for o in res["objects"]]
        self.assertIn("egg_1", ids)
        self.assertIn("cup_1", ids)

    def test_4_unreachable_target_fails_fast_within_budget(self):
        io = FakeGeneralIO()
        sk = GPP.GeneralPickPlaceSkills(io)
        sk.perceive_objects(prompts=["egg", "cup"])
        # Put an unreachable point 2.5 m away
        sk.cam["far_1"] = {
            "rim": np.array([-2.5, 0.0, 0.05]),
            "center": np.array([-2.5, 0.0, 0.02]),
            "base": np.array([-2.5, 0.0, 0.0]),
            "top_z_m": 0.05, "label": "far", "yaw_deg": 0.0,
            "size_mm": [50.0, 50.0], "height_mm": 50.0, "conf": 0.9, "trusted": True,
        }
        t0 = time.perf_counter()
        r = sk.plan_grasp_chunk(object_id="far_1")
        dt = time.perf_counter() - t0
        self.assertFalse(r["ok"])
        self.assertLess(dt, GPP.PLAN_BUDGET_S + 1.0)

    def test_5_gripper_ui_auto_close_and_disable_on_zero_start(self):
        with open(os.path.join(HERE, "NERO_V48_Gripper_UI.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn("def _auto_close_and_relax():", src)
        self.assertIn("g.clamp(force=0.5", src)
        self.assertIn("g.relax()", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
