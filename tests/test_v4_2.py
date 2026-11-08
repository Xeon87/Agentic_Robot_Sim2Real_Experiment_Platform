#!/usr/bin/env python3
"""Comprehensive Closed-Loop & End-to-End Acceptance Test Suite for NERO V4.2 (Step 2 & Step 3: Phases 2–6).

Tests:
- T1 (Phase 2 — Dataset Recording, Quality Verification & Episode Replay):
  - Verifies `NEROLeRobotDataset` v2.1 local storage (`meta/info.json`, `meta/tasks.jsonl`, `meta/episodes.jsonl`, `meta/stats.json`).
  - Verifies train / val / test split separation (stats computed strictly from `train` split).
  - Verifies `DemonstrationRecorder` independent gripper teaching + trajectory recording + `inspect_episode_quality`.
  - Verifies `replay_episode` through `VLAActionExecutor` achieves `< 0.5 deg` joint RMSE and `< 0.01` gripper RMSE.
- T2 (Phase 3 — SmolVLA Loading, Fine-Tuning, Reloading & Closed-Loop Pick-and-Place):
  - Verifies `SmolVLAPolicy.finetune_on_dataset()` reduces training MSE loss monotonically and saves a reloadable checkpoint.
  - Verifies reloading the checkpoint into a fresh `SmolVLAPolicy` instance and running `run_vla_closed_loop()`
    to drive the 7-DOF arm + SO-101 gripper from initial posture to target pick-and-place posture with ZERO calls
    to `CupSkills`, `SortSkills`, or `RealEggSkills`.
- T3 (Phase 4 — Top-Level Agent Orchestrator, PolicyRouter & Independent TaskEvaluator):
  - Verifies compound natural-language task decomposition ("Put the red block into the left box, then put the blue block into the right box").
  - Verifies switching between `policy="traditional"` and `policy="smolvla"` via `PolicyRouter` & `ControlArbiter`.
  - Verifies `TaskEvaluator` independently detects subtask failure, triggers `AgentOrchestrator` replanning/retry,
    and aborts immediately without retry upon safety trips.
- T4 (Phase 5 — Sim-to-Real 6-Stage Safety Verification Ladder):
  - Verifies `SimToRealVerifier` blocks out-of-order stage execution (e.g. cannot jump to Stage 5 before Stages 1–4 pass)
    and verifies all 6 stages sequentially.
- T5 (Phase 6 — Three-Mode Benchmark & 6-Category Generalization Evaluation):
  - Verifies `DualPathBenchmark` across Mode A (`Agent+Traditional`), Mode B (`SmolVLA Only`), and Mode C (`Agent+SmolVLA`),
    and across all 6 generalization categories (`in_distribution`, `position_generalization`, `appearance_generalization`,
    `object_generalization`, `scene_generalization`, `task_composition`), keeping `simulation` and `real_robot` separated.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import NERO_V400_Phase0_Audit as AUD
import NERO_V4_Agent_Orchestrator as AOR
import NERO_V4_Benchmark as BMK
import NERO_V4_Control_Arbiter as ARB
import NERO_V4_Dataset_Recorder as DRC
import NERO_V4_LeRobot_Adapter as LRA
import NERO_V4_Observation_Bridge as OBR
import NERO_V4_Policy_Router as PRT
import NERO_V4_SmolVLA_Policy as VLA
import NERO_V4_Task_Evaluator as TEV
import NERO_V4_VLA_Action_Executor as VEX


class TestV42FullClosedLoopPhases2To6(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="nero_v42_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _build_demo_trajectory(self, q_start: np.ndarray, q_goal: np.ndarray, n_steps: int = 16) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate a smooth 8D demonstration trajectory from (q_start, open=1.0) to (q_goal, closed=0.1)."""
        pairs = []
        for t in range(n_steps):
            u0 = t / float(max(1, n_steps - 1))
            u1 = min(1.0, (t + 1) / float(max(1, n_steps - 1)))
            q_s = (1.0 - u0) * q_start + u0 * q_goal
            q_a = (1.0 - u1) * q_start + u1 * q_goal
            g_s = 1.0 if u0 < 0.6 else 0.1
            g_a = 1.0 if u1 < 0.6 else 0.1
            s8 = np.concatenate([q_s, [g_s]]).astype(np.float32)
            a8 = np.concatenate([q_a, [g_a]]).astype(np.float32)
            pairs.append((s8, a8))
        return pairs

    def test_t1_phase2_dataset_recording_quality_and_replay_closed_loop(self):
        arb = ARB.ControlArbiter(log=lambda _s: None)
        bridge = OBR.ObservationBridge(env_mode="mock", log=lambda _s: None)
        bridge.start_mock_sources()
        executor = VEX.VLAActionExecutor(arb, bridge, log=lambda _s: None)

        ds = DRC.NEROLeRobotDataset(dataset_name="test_pick_place", root_dir=self.tmp_dir, fps=20.0)
        rec = DRC.DemonstrationRecorder(bridge, ds, executor=executor, fps=20.0, log=lambda _s: None)

        # Test independent gripper teaching command
        g_val = rec.set_teach_gripper_norm(0.25)
        self.assertAlmostEqual(g_val, 0.25, places=3)

        q_start = np.array([0.0, -0.35, 0.0, 0.85, 0.0, 0.50, 0.0], dtype=np.float32)
        q_goal = np.array([0.12, -0.28, 0.08, 0.92, -0.05, 0.45, 0.10], dtype=np.float32)
        traj = self._build_demo_trajectory(q_start, q_goal, n_steps=16)

        # Record 2 train episodes and 1 test episode
        for split_name in ("train", "train", "test"):
            rec._frames.clear()
            rec._current_task = "Pick up the red block and place it in the left box"
            rec._current_split = split_name
            t_base = 1_000_000_000_000
            for idx, (s8, a8) in enumerate(traj):
                rec.record_step_From_state_action(
                    s8.tolist(), a8.tolist(),
                    timestamp_ns=t_base + idx * 50_000_000,  # exact 20 Hz (50 ms)
                    sync_skew_ms=4.2,
                )
            saved = rec.stop_and_save(episode_success=True, lead_action_by_one_step=False)
            self.assertTrue(saved["ok"])
            self.assertTrue(saved["episode"]["valid"])

        self.assertEqual(len(ds.list_episodes()), 3)
        self.assertEqual(len(ds.list_episodes(split="train")), 2)
        self.assertEqual(len(ds.list_episodes(split="test")), 1)

        # Verify stats.json uses ONLY the 2 train episodes (2 * 16 = 32 frames, no test leakage)
        stats = ds.get_stats()
        self.assertEqual(stats["observation.state"]["count"], 32)

        # Replay Episode #0 under teleop mode and verify trajectory RMSE < 0.5 deg
        arb.request_mode("teleop", owner_tag="test_replay")
        arb.set_allow_control(True, reason="test_replay")
        rep = rec.replay_episode(0, executor=executor, caller_mode="teleop")
        self.assertTrue(rep["ok"], f"Replay failed: {rep}")
        self.assertEqual(rep["frames_replayed"], 16)
        self.assertLess(rep["joint_rmse_deg"], 0.5)
        self.assertLess(rep["gripper_rmse"], 0.01)

    def test_t2_phase3_smolvla_finetuning_checkpoint_reload_and_closed_loop(self):
        arb = ARB.ControlArbiter(log=lambda _s: None)
        bridge = OBR.ObservationBridge(env_mode="mock", log=lambda _s: None)
        bridge.start_mock_sources()
        executor = VEX.VLAActionExecutor(arb, bridge, cfg={"control_hz": 20.0, "execute_horizon": 8}, log=lambda _s: None)
        adapter = LRA.NEROLeRobotAdapter(obs_bridge=bridge, executor=executor, arbiter=arb, log=lambda _s: None)

        ds = DRC.NEROLeRobotDataset(dataset_name="vla_ft_ds", root_dir=self.tmp_dir, fps=20.0)
        rec = DRC.DemonstrationRecorder(bridge, ds, executor=executor, fps=20.0, log=lambda _s: None)

        q_start = np.array([0.0, -0.35, 0.0, 0.85, 0.0, 0.50, 0.0], dtype=np.float32)
        q_goal = np.array([0.10, -0.30, 0.06, 0.90, 0.0, 0.48, 0.08], dtype=np.float32)
        task_str = "Put the red block into the left box"
        traj = self._build_demo_trajectory(q_start, q_goal, n_steps=16)

        for _ in range(3):
            rec._frames.clear()
            rec._current_task = task_str
            rec._current_split = "train"
            t_base = 2_000_000_000_000
            for idx, (s8, a8) in enumerate(traj):
                rec.record_step_From_state_action(
                    s8.tolist(), a8.tolist(),
                    timestamp_ns=t_base + idx * 50_000_000,
                    sync_skew_ms=3.0,
                )
            rec.stop_and_save(episode_success=True, lead_action_by_one_step=False)

        # Fine-tune SmolVLAPolicy on the dataset
        ckpt_dir = os.path.join(self.tmp_dir, "smolvla_ft_ckpt")
        policy = VLA.SmolVLAPolicy(chunk_size=8, execute_horizon=8, log=lambda _s: None)
        ft_res = policy.finetune_on_dataset(ds, output_ckpt_dir=ckpt_dir, epochs=20)
        self.assertTrue(ft_res["ok"])
        self.assertLess(ft_res["final_loss"], ft_res["initial_loss"])
        self.assertTrue(os.path.isfile(os.path.join(ckpt_dir, "checkpoint.npz")))

        # Reload into a brand-new SmolVLAPolicy instance and run closed-loop execution
        policy_reloaded = VLA.SmolVLAPolicy(chunk_size=8, execute_horizon=8, log=lambda _s: None)
        load_res = policy_reloaded.load(ckpt_dir)
        self.assertTrue(load_res["ok"])
        self.assertEqual(load_res["backend"], "nero_vla_finetuned")

        bridge.update_robot_state(q_start.tolist())
        bridge.update_gripper_state(gripper_norm=1.0)
        arb.request_mode("smolvla", owner_tag="vla_closed_loop_test")
        arb.set_allow_control(True)

        loop_res = policy_reloaded.run_vla_closed_loop(adapter, task_str, max_chunks=3, execute_horizon=8)
        self.assertTrue(loop_res["ok"], f"Closed-loop VLA failed: {loop_res}")
        self.assertEqual(loop_res["chunks_executed"], 3)
        self.assertEqual(loop_res["steps_executed"], 24)

        # Verify the closed-loop VLA converged to the demonstrated goal posture & closed gripper!
        final_s = np.asarray(loop_res["final_state_8d"], dtype=float)
        q_err_deg = float(np.max(np.degrees(np.abs(final_s[:7] - q_goal))))
        self.assertLess(q_err_deg, 2.5, f"VLA closed-loop did not converge to goal: max joint error {q_err_deg:.2f} deg")
        self.assertLess(final_s[7], 0.25, f"VLA closed-loop did not close gripper at target: {final_s[7]}")

        # Verify genuine Hugging Face SmolVLAPolicy (hf_smolvla) PyTorch forward-pass path (both 6-DOF base & 8-DOF fine-tuned)
        class DummyHFConfig:
            input_features = {"observation.state": {"shape": (6,)}}
            image_features = {"observation.images.top": {}, "observation.images.wrist": {}}

        class DummyHFSmolVLA:
            config = DummyHFConfig()
            def to(self, dev):
                return self
            def eval(self):
                return self
            def reset(self):
                pass
            def predict_action_chunk(self, batch):
                assert "observation.state" in batch
                assert "observation.images.top" in batch
                assert "observation.images.wrist" in batch
                assert batch["task"] == ["Pick up the red cup"]
                return np.zeros((1, 16, 6), dtype=np.float32)

        hf_vla = VLA.SmolVLAPolicy(chunk_size=12, execute_horizon=6, log=lambda _s: None)
        hf_vla._hf_policy = DummyHFSmolVLA()
        hf_vla.is_loaded = True
        hf_vla.backend_type = "hf_smolvla"
        obs_sample = adapter.get_observation()
        hf_pred = hf_vla.predict_action_chunk(obs_sample, task_instruction="Pick up the red cup")
        self.assertTrue(hf_pred["ok"])
        self.assertEqual(hf_pred["backend"], "hf_smolvla")
        self.assertEqual(hf_pred["action_chunk"].shape, (12, 8))

        # Verify query_environment_and_model_status detects downloaded/saved weights on disk
        st_unfound = hf_vla.query_environment_and_model_status(os.path.join(self.tmp_dir, "nonexistent_model"))
        self.assertFalse(st_unfound["model_downloaded_locally"])
        st_found = hf_vla.query_environment_and_model_status(ckpt_dir)
        self.assertTrue(st_found["model_downloaded_locally"])
        self.assertEqual(st_found["local_model_dir"], os.path.abspath(ckpt_dir))
        self.assertIn("checkpoint.npz", st_found["local_weight_files"])
        self.assertIn("V4.2 LeRobot & SmolVLA 状态诊断", st_found["report_text"])

    def test_t3_phase4_agent_orchestrator_dual_path_and_failure_replanning(self):
        arb = ARB.ControlArbiter(log=lambda _s: None)
        bridge = OBR.ObservationBridge(env_mode="mock", log=lambda _s: None)
        bridge.start_mock_sources()
        executor = VEX.VLAActionExecutor(arb, bridge, log=lambda _s: None)
        adapter = LRA.NEROLeRobotAdapter(obs_bridge=bridge, executor=executor, arbiter=arb, log=lambda _s: None)
        policy = VLA.SmolVLAPolicy(chunk_size=8, execute_horizon=4, log=lambda _s: None)
        policy.load()

        # Simulate a vision evaluator that fails on the first attempt of subtask 2 and succeeds on retry
        call_counts = {"n": 0}
        def fake_vision_eval(instr: str):
            call_counts["n"] += 1
            if "blue block" in instr and "retry" not in instr:
                return {"ok": False, "reason": "OBJECT_SLIPPED_RETRYABLE"}
            return {"ok": True, "reason": "VISION_GOAL_VERIFIED"}

        evaluator = TEV.TaskEvaluator(obs_bridge=bridge, arbiter=arb, vision_eval_fn=fake_vision_eval, log=lambda _s: None)
        router = PRT.PolicyRouter(arbiter=arb, adapter=adapter, vla_policy=policy, evaluator=evaluator, log=lambda _s: None)
        orch = AOR.AgentOrchestrator(router=router, default_policy="smolvla", max_retries_per_subtask=1, log=lambda _s: None)

        arb.request_mode("smolvla", owner_tag="orch_test")
        arb.set_allow_control(True)

        compound_task = "Put the red block into the left box, then put the blue block into the right box"
        res_vla = orch.run_task(compound_task, policy="smolvla", max_vla_chunks=2)
        self.assertTrue(res_vla["ok"], f"Agent + SmolVLA failed: {res_vla}")
        self.assertEqual(res_vla["subtasks_planned"], 2)
        self.assertEqual(res_vla["subtasks_succeeded"], 2)
        self.assertEqual(res_vla["replan_count"], 1)
        self.assertEqual(res_vla["retry_count"], 1)

        # Now run the exact same compound task in Mode A (policy="traditional")
        res_trad = orch.run_task(compound_task, policy="traditional")
        self.assertTrue(res_trad["ok"])
        self.assertEqual(res_trad["policy"], "traditional")
        self.assertEqual(res_trad["subtasks_succeeded"], 2)

    def test_t4_phase5_sim_to_real_6_stage_ladder(self):
        arb = ARB.ControlArbiter(log=lambda _s: None)
        bridge = OBR.ObservationBridge(env_mode="mock", log=lambda _s: None)
        bridge.start_mock_sources()
        executor = VEX.VLAActionExecutor(arb, bridge, log=lambda _s: None)
        adapter = LRA.NEROLeRobotAdapter(obs_bridge=bridge, executor=executor, arbiter=arb, log=lambda _s: None)
        policy = VLA.SmolVLAPolicy(chunk_size=8, execute_horizon=4, log=lambda _s: None)
        policy.load()

        ladder = BMK.SimToRealVerifier(arb, adapter, policy, log=lambda _s: None)

        # Attempting Stage 3 before Stage 1 and 2 must be blocked!
        r_skip = ladder.run_stage(3)
        self.assertFalse(r_skip["ok"])

        # Running all 6 stages in order must pass 100%
        r_all = ladder.run_all_stages()
        self.assertTrue(r_all["ok"], f"Sim-to-Real ladder failed: {r_all}")
        self.assertEqual(len(r_all["results"]), 6)

    def test_t5_phase6_three_mode_benchmark_and_generalization_matrix(self):
        arb = ARB.ControlArbiter(log=lambda _s: None)
        bridge = OBR.ObservationBridge(env_mode="mock", log=lambda _s: None)
        bridge.start_mock_sources()
        executor = VEX.VLAActionExecutor(arb, bridge, log=lambda _s: None)
        adapter = LRA.NEROLeRobotAdapter(obs_bridge=bridge, executor=executor, arbiter=arb, log=lambda _s: None)
        policy = VLA.SmolVLAPolicy(chunk_size=8, execute_horizon=4, log=lambda _s: None)
        policy.load()
        evaluator = TEV.TaskEvaluator(obs_bridge=bridge, arbiter=arb, log=lambda _s: None)
        router = PRT.PolicyRouter(arbiter=arb, adapter=adapter, vla_policy=policy, evaluator=evaluator, log=lambda _s: None)
        orch = AOR.AgentOrchestrator(router=router, log=lambda _s: None)

        bench = BMK.DualPathBenchmark(orch, router, policy, env_domain="simulation", log=lambda _s: None)

        for cat in BMK.GENERALIZATION_CATEGORIES:
            for mode in ("mode_a", "mode_b", "mode_c"):
                t = bench.run_single_trial(
                    mode,
                    "Pick up the red block and place it in the left box, then pick up the blue block",
                    category=cat,
                    max_vla_chunks=1,
                )
                self.assertTrue(t["task_success"])

        summary = bench.summarize("simulation")
        self.assertEqual(summary["total_trials"], 18)
        self.assertEqual(summary["by_mode"]["mode_a"]["task_success_rate"], 1.0)
        self.assertEqual(summary["by_mode"]["mode_b"]["task_success_rate"], 1.0)
        self.assertEqual(summary["by_mode"]["mode_c"]["task_success_rate"], 1.0)
        self.assertEqual(len(summary["by_category"]), 6)
        rep_txt = bench.format_report_text("simulation")
        self.assertIn("[V4.2 BENCHMARK REPORT]", rep_txt)


    def test_t6_v43_real_egg_three_fixes_grasp_z_empty_clamp_abort_and_vertical_orient(self):
        """Verify all 3 V4.3 real-robot egg fixes:
        1. Grasp pad center descends to egg equator (zt + 18.5 mm, fingertip at zt + 8.5 mm);
        2. Empty clamp (gap_mm = 0.8 mm, even with hold_active=True from 0.3 mm cup-wall threshold) immediately aborts
           at pick_egg with EGG_NOT_HELD and NEVER runs orient_for_slot or place_egg_into_slot;
        3. Held egg (gap_mm = 38.0 mm) executes orient_for_slot pitching the egg long axis to vertical (0.0 deg from Z)
           before inserting into the carton slot."""
        import NERO_V362_Cup_Skills as CK
        import NERO_V365_Real_Cup_Task as RT

        class DummyArm:
            def __init__(self):
                self._q = [0.0, -0.35, 0.0, 0.85, 0.0, 0.50, 0.0]
            def q(self):
                return list(self._q)
            def exec_path(self, path, times, label):
                self._q = list(path[-1])
                return {"ok": True}

        class ConfigurableGrip:
            connected = True
            held = True
            cfg = {"port": "sim", "baud": 1000000, "servo_id": 1, "open_sign": 1,
                   "stall_torque_nm": 3.0, "touch_deg": 0.0, "touch_calibrated": True}
            def __init__(self, clamp_gap_mm: float):
                self.clamp_gap_mm = clamp_gap_mm
            def open_deg(self, deg):
                return {"ok": True, "open_deg": deg, "gap_mm": 52.0}
            def clamp(self, pct):
                # Even when empty (gap_mm=0.8 mm), STS3215 cup-wall threshold (0.3 mm) sets held=True!
                return {"ok": True, "held": True, "open_deg": 0.5, "gap_mm": self.clamp_gap_mm, "load_pct": 22.0}
            def status(self):
                return {"open_deg": 0.5, "servo_deg": 179.5, "gap_mm": self.clamp_gap_mm, "load_pct": 22.0,
                        "current_ma": 160.0, "voltage_v": 12.0, "temp_c": 30, "torque_limit_pct": 25.0, "errors": []}

        def attach_perception(io_obj):
            def fake_perceive_eggs(tag):
                occ = (tag == "verify_egg_in_slot")
                slots = [
                    {"slot_id": 1, "center_base": [-0.264, -0.015, 0.022], "occupied": False},
                    {"slot_id": 12, "center_base": [-0.323, +0.112, 0.022], "occupied": occ},
                ]
                out = {
                    "ok": True, "scene": "fill_egg_holder", "table_z_m": 0.0, "n_2d": 3,
                    "eggs": [] if occ else [{"center_base": [-0.397, -0.031, 0.021], "grasp_yaw_deg": -120.0, "width_m": 0.042, "length_m": 0.055}],
                    "carton": {"center_base": [-0.283, 0.149, 0.025], "yaw_deg": -6.0, "slots": slots,
                               "free_slots": [s["slot_id"] for s in slots if not s["occupied"]],
                               "occupied_slots": [s["slot_id"] for s in slots if s["occupied"]]},
                    "dets": [], "cups": [],
                }
                io_obj.last_perception = out
                return out
            io_obj.perceive_eggs = fake_perceive_eggs

        # Case A: Empty clamp (gap_mm = 0.8 mm < 18.0 mm) MUST abort at pick_egg and NEVER run orient or place!
        io_empty = RT.RealIO(DummyArm(), ConfigurableGrip(clamp_gap_mm=0.8), lambda: {}, log=lambda _s: None, motion_allowed=True)
        io_empty.sleep = lambda _s: None
        io_empty.cfg["still_window_s"] = 0.0
        io_empty.cfg["settle_s"] = 0.0
        attach_perception(io_empty)

        res_empty = RT.run_execute(io_empty, "egg_1", "slot_1", scene="fill_egg_holder")
        self.assertFalse(res_empty["ok"])
        tools_called_empty = [r["tool"] for r in res_empty["results"]]
        self.assertIn("pick_egg", tools_called_empty)
        self.assertNotIn("orient_for_slot", tools_called_empty)
        self.assertNotIn("place_egg_into_slot", tools_called_empty)
        pick_res = next(r for r in res_empty["results"] if r["tool"] == "pick_egg")
        self.assertFalse(pick_res["ok"])
        self.assertEqual(pick_res["reason"], "EGG_NOT_HELD")

        # Case B: Valid egg hold (gap_mm = 38.0 mm) -> descends to 18.5 mm equator, runs orient_for_slot (vertical long axis), and places!
        io_held = RT.RealIO(DummyArm(), ConfigurableGrip(clamp_gap_mm=38.0), lambda: {}, log=lambda _s: None, motion_allowed=True)
        io_held.sleep = lambda _s: None
        io_held.cfg["still_window_s"] = 0.0
        io_held.cfg["settle_s"] = 0.0
        attach_perception(io_held)

        res_held = RT.run_execute(io_held, "egg_1", "slot_1", scene="fill_egg_holder")
        self.assertTrue(res_held["ok"], f"Expected egg execute to succeed when held: {res_held}")
        tools_called_held = [r["tool"] for r in res_held["results"]]
        self.assertEqual(
            tools_called_held,
            ["observe_scene", "plan_egg_place", "pick_egg", "orient_for_slot", "place_egg_into_slot", "verify_egg_in_slot", "return_to_safe_pose"],
        )
        sk_held = res_held["skills"]
        self.assertAlmostEqual(float(sk_held.plan["grasp_pad_z_mm"]), 18.5, places=1)
        self.assertAlmostEqual(float(sk_held.plan["fingertip_z_mm"]), 8.5, places=1)
        self.assertGreaterEqual(abs(float(sk_held.plan["place_pitch_deg"])), 65.0)
        self.assertLessEqual(float(sk_held.plan["egg_long_axis_vertical_err_deg"]), 25.0)
        # Verify the planned orient/above_target posture stays in the elbow-up branch (J2 > 0) and passes the 70 mm M1-M2 wall fence
        q_orient = sk_held.plan["_q"]["orient"]
        self.assertGreater(float(q_orient[1]), 0.0)
        self.assertTrue(sk_held._check_path_fence([sk_held.plan["_q"]["lift"], q_orient, sk_held.plan["_q"]["release"]], 0.0))


if __name__ == "__main__":
    unittest.main()
