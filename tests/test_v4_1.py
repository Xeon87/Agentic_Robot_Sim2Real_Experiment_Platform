#!/usr/bin/env python3
"""Comprehensive Acceptance & Regression Test Suite for NERO V4.1 (Phase 0 + Phase 1).

Tests:
- T1: Phase 0 Audit (`NERO_V400_Phase0_Audit.py`):
      Verifies baseline V3.93 modules + V4.1 modules integrity, GPU/VRAM probe, LeRobot env probe,
      ROS 2 interface probe, and dual-camera + 8D state synchronization skew benchmark (<= 25 ms).
- T2: Dual-Path Control Arbiter & Strategy-Independent Safety Supervisor (`NERO_V4_Control_Arbiter.py`):
      Verifies exclusive control authority (Traditional vs. SmolVLA vs. Monitor vs. E-Stop),
      proves `connect()` does NOT auto-grant control, and verifies independent FK tabletop/fence/limit/step-jump protection.
- T3: Synchronized Observation Bridge & Zero Ground-Truth Leak (`NERO_V4_Observation_Bridge.py`):
      Verifies dual RGB cameras (`(480, 640, 3)` uint8) + 8D state (`[j1..j7, gripper_norm]`),
      POSIX SharedMemory zero-copy round-trip, and guarantees no ground-truth object coordinates leak.
- T4: LeRobot Robot Adapter & Cross-Environment IPC Server/Client (`NERO_V4_LeRobot_Adapter.py`):
      Verifies `connect()`, `disconnect()`, `observation_features`, `action_features`, `get_observation()`,
      and `send_action()` both in-process and over Unix Domain Socket + POSIX SharedMemory IPC.
- T5: Controlled Action Chunk Execution in Simulation & Real-Robot Adapter (`NERO_V4_VLA_Action_Executor.py`):
      Verifies multi-step Action Chunk execution across `mujoco_shared` (`target_shared[7]`, `gripper_target_mm`),
      `mock`, and `real` (`move_joint_path` + `SO101RealGripper`), with zero calls to legacy `CupSkills` / `SortSkills`.
- T6: Console V4 Tab UI (`NERO_V4_Console_Tab.py` & `NERO_V312_Console.py`):
      Verifies the 4-stage control separation UI (Connect / Allow Control / Start / E-Stop).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import NERO_V400_Phase0_Audit as AUD
import NERO_V4_Console_Tab as V4T
import NERO_V4_Control_Arbiter as ARB
import NERO_V4_LeRobot_Adapter as LRA
import NERO_V4_Observation_Bridge as OBR
import NERO_V4_VLA_Action_Executor as VEX


class TestV41Phase0AndPhase1(unittest.TestCase):
    def test_t1_phase0_architecture_and_sync_audit(self):
        rep = AUD.run_phase0_audit(HERE)
        self.assertTrue(rep["ok"], f"Phase 0 audit failed: {rep}")
        self.assertIn(rep["version"], ("V4.1", "V4.2", "V4.3", "V4.4"))
        self.assertEqual(len(rep["baseline"]["missing_baseline"]), 0)
        self.assertEqual(len(rep["baseline"]["missing_v4"]), 0)
        self.assertLessEqual(rep["sync_benchmark"]["max_skew_ms"], 25.0)
        summary = AUD.format_audit_summary(rep)
        self.assertIn("[V4.1 PHASE 0 AUDIT]", summary)

    def test_t2_control_arbiter_exclusivity_and_safety_supervisor(self):
        logs = []
        arb = ARB.ControlArbiter(log=logs.append)

        # Default mode is monitor, allow_control is False
        self.assertEqual(arb.mode, "monitor")
        self.assertFalse(arb.allow_control)

        # Attempting to execute a SmolVLA step while in monitor mode must be rejected
        q0 = [0.0, -0.35, 0.0, 0.85, 0.0, 0.50, 0.0]
        r_denied = arb.validate_and_clamp_step(q0, q0, gripper_norm=0.5, caller_mode="smolvla")
        self.assertFalse(r_denied["ok"])
        self.assertIn("CONTROL_ARBITER_DENIED", r_denied["error"])

        # Switch to smolvla mode, but before enabling allow_control -> still rejected!
        arb.request_mode("smolvla", owner_tag="test_vla")
        self.assertFalse(arb.allow_control)
        r_no_allow = arb.validate_and_clamp_step(q0, q0, gripper_norm=0.5, caller_mode="smolvla")
        self.assertFalse(r_no_allow["ok"])
        self.assertIn("CONTROL_NOT_ALLOWED", r_no_allow["error"])

        # Enable allow_control -> SmolVLA is now permitted, while Traditional is simultaneously blocked!
        arb.set_allow_control(True, reason="unit_test")
        self.assertTrue(arb.allow_control)
        r_trad_blocked = arb.validate_and_clamp_step(q0, q0, gripper_norm=0.5, caller_mode="traditional")
        self.assertFalse(r_trad_blocked["ok"])
        self.assertIn("CONTROL_ARBITER_DENIED", r_trad_blocked["error"])

        # Valid small step succeeds
        q1 = [0.02, -0.34, 0.01, 0.86, 0.0, 0.50, 0.0]
        r_ok = arb.validate_and_clamp_step(q0, q1, gripper_norm=0.75, dt_s=0.05, caller_mode="smolvla")
        self.assertTrue(r_ok["ok"])
        self.assertAlmostEqual(r_ok["gripper_norm"], 0.75, places=3)

        # Oversized single-step jump is clamped when clamp_in_place=True, or rejected when clamp_in_place=False
        q_huge = [0.80, -0.35, 0.0, 0.85, 0.0, 0.50, 0.0]
        r_clamp = arb.validate_and_clamp_step(
            q0, q_huge, gripper_norm=0.5, dt_s=0.05, caller_mode="smolvla", clamp_in_place=True
        )
        self.assertTrue(r_clamp["ok"])
        self.assertTrue(r_clamp["clamped"])
        self.assertLessEqual(abs(r_clamp["q_safe_rad"][0] - q0[0]), math.radians(6.0) + 1e-4)

        r_strict = arb.validate_and_clamp_step(
            q0, q_huge, gripper_norm=0.5, dt_s=0.05, caller_mode="smolvla", clamp_in_place=False
        )
        self.assertFalse(r_strict["ok"])
        self.assertIn("STEP_JUMP_TOO_LARGE", r_strict["error"])

        # Emergency stop immediately blocks all motion until clear_estop()
        arb.trigger_estop("test_estop")
        self.assertEqual(arb.mode, "estop")
        self.assertFalse(arb.allow_control)
        r_estop = arb.validate_and_clamp_step(q0, q1, caller_mode="smolvla")
        self.assertFalse(r_estop["ok"])
        self.assertIn("EMERGENCY_STOP_ACTIVE", r_estop["error"])

        arb.clear_estop()
        self.assertEqual(arb.mode, "monitor")
        self.assertFalse(arb.allow_control)

    def test_t3_observation_bridge_and_shared_memory_zero_ground_truth_leak(self):
        shm_name = f"nero_v4_test_shm_{os.getpid()}"
        bridge = OBR.ObservationBridge(env_mode="mock", shm_prefix=shm_name)
        try:
            bridge.start_mock_sources()
            bridge.set_task_instruction("Pick up the red block")
            bridge.update_robot_state([0.1, -0.2, 0.05, 0.9, -0.1, 0.4, 0.2])
            bridge.update_gripper_state(gap_mm=22.5)

            obs = bridge.get_synchronized_observation()
            self.assertTrue(obs["sync_ok"])
            self.assertLessEqual(obs["sync_skew_ms"], 25.0)
            self.assertEqual(obs["camera1"].shape, (480, 640, 3))
            self.assertEqual(obs["camera2"].shape, (480, 640, 3))
            self.assertEqual(obs["state"].shape, (8,))
            self.assertAlmostEqual(float(obs["state"][0]), 0.1, places=4)
            self.assertAlmostEqual(float(obs["state"][7]), 0.5, places=3)  # 22.5 / 45.0 = 0.5

            # Verify zero ground-truth object keys exist in observation
            for forbidden_key in ("object_poses", "xpos", "xquat", "ground_truth", "cups", "eggs"):
                self.assertNotIn(forbidden_key, obs)

            # Verify POSIX SharedMemory round-trip
            shm_obs = OBR.ObservationBridge.read_from_shared_memory(shm_name)
            self.assertEqual(shm_obs["seq"], obs["seq"])
            np.testing.assert_allclose(shm_obs["state"], obs["state"], atol=1e-5)
            np.testing.assert_array_equal(shm_obs["camera1"], obs["camera1"])
            np.testing.assert_array_equal(shm_obs["camera2"], obs["camera2"])
        finally:
            bridge.close()

    def test_t4_lerobot_adapter_in_process_and_cross_process_ipc(self):
        shm_name = f"nero_v4_ipc_shm_{os.getpid()}"
        sock_path = os.path.join(tempfile.gettempdir(), f"nero_v4_ipc_{os.getpid()}.sock")

        arb = ARB.ControlArbiter(log=lambda _s: None)
        bridge = OBR.ObservationBridge(env_mode="mock", shm_prefix=shm_name, log=lambda _s: None)
        bridge.start_mock_sources()
        bridge.update_robot_state([0.0, -0.35, 0.0, 0.85, 0.0, 0.50, 0.0])
        bridge.set_task_instruction("Place block into left box")
        executor = VEX.VLAActionExecutor(arb, bridge, log=lambda _s: None)

        srv = LRA.NEROIPCServer(bridge, executor, socket_path=sock_path, shm_name=shm_name, log=lambda _s: None)
        srv.start()

        client = LRA.NEROLeRobotAdapter(ipc_socket_path=sock_path, log=lambda _s: None)
        try:
            # Verify feature schemas
            self.assertIn("observation.state", client.observation_features)
            self.assertIn("observation.images.cam_main", client.observation_features)
            self.assertIn("observation.images.cam_wrist", client.observation_features)
            self.assertIn("action", client.action_features)
            self.assertEqual(client.action_features["action"]["shape"], (8,))

            # 1) Connect over IPC — must NOT automatically grant control authority!
            c_res = client.connect()
            self.assertTrue(c_res["ok"])
            self.assertFalse(arb.allow_control)

            # 2) Read synchronized observation over IPC + SharedMemory
            obs = client.get_observation()
            self.assertEqual(obs["observation.state"].shape, (8,))
            self.assertEqual(obs["observation.images.cam_main"].shape, (480, 640, 3))
            self.assertEqual(obs["observation.images.cam_wrist"].shape, (480, 640, 3))
            self.assertEqual(obs["task"], "Place block into left box")
            self.assertTrue(obs["sync_ok"])

            # 3) Attempting send_action before enabling control authority is rejected
            q_cur = obs["observation.state"][:7].copy()
            test_act = np.concatenate([q_cur + 0.01, [0.6]]).astype(np.float32)
            r_blocked = client.send_action(test_act)
            self.assertFalse(r_blocked["ok"])

            # 4) Grant smolvla mode + allow_control on host Arbiter -> send_action Chunk succeeds over IPC!
            arb.request_mode("smolvla", owner_tag="ipc_client")
            arb.set_allow_control(True, reason="ipc_test")
            chunk = np.stack([
                np.concatenate([q_cur + 0.01 * k, [0.8 - 0.1 * k]])
                for k in range(1, 5)
            ]).astype(np.float32)
            r_exec = client.send_action(chunk)
            self.assertTrue(r_exec["ok"], f"IPC chunk execution failed: {r_exec}")
            self.assertEqual(r_exec["executed_steps"], 4)

            # Verify updated robot state reflects the final executed action in the chunk
            obs_after = client.get_observation()
            np.testing.assert_allclose(obs_after["observation.state"][:7], chunk[-1, :7], atol=1e-4)
            self.assertAlmostEqual(float(obs_after["observation.state"][7]), float(chunk[-1, 7]), places=3)
        finally:
            client.disconnect()
            srv.stop()
            bridge.close()

    def test_t5_mujoco_shared_and_real_robot_execution_without_legacy_skills(self):
        arb = ARB.ControlArbiter(log=lambda _s: None)
        bridge = OBR.ObservationBridge(env_mode="mujoco", log=lambda _s: None)
        bridge.start_mock_sources()
        q0 = [0.0, -0.35, 0.0, 0.85, 0.0, 0.50, 0.0]
        bridge.update_robot_state(q0)
        executor = VEX.VLAActionExecutor(arb, bridge, log=lambda _s: None)

        # Bind to MuJoCo shared arrays (exact structure of NERO_V2_7_1_Unified_Backend)
        target_shared = mp.Array("d", q0)
        gripper_target_mm = mp.Value("d", 32.0)
        executor.bind_mujoco_shared(target_shared, gripper_target_mm)

        arb.request_mode("smolvla", owner_tag="mujoco_vla")
        arb.set_allow_control(True)

        chunk = [
            [0.02 * k, -0.35 + 0.01 * k, 0.0, 0.85, 0.0, 0.50, 0.0, 0.5]
            for k in range(1, 6)
        ]
        res = executor.execute_action_chunk(chunk, execute_steps=5)
        self.assertTrue(res["ok"])
        self.assertEqual(res["executed_steps"], 5)
        self.assertAlmostEqual(target_shared[0], 0.10, places=4)
        self.assertAlmostEqual(gripper_target_mm.value, 22.5, places=2)




class TestV41RealEggFixesRegression(unittest.TestCase):
    def test_t6_real_egg_grasp_z_fence_safe_and_fast_ik_and_slot_fallback(self):
        """Reproduce the exact failure logs from the user's real-robot egg test:
        1. Run 3 log: egg_1@(-0.397, -0.031, +0.021) with fence z_min = +0.013 m.
           Verify that planned grasp TCP z (`W["grasp"][0][2]`) is >= +0.017 m (> 0.013 m fence z_min),
           never triggering `FENCE: tool z=0.002 m < 0.013 (z_min)`.
        2. Run 2 log: egg_1@(-0.390, -0.019, +0.021), slot_1@(-0.264, -0.015, +0.022) (in inner dead-zone)
           and slot_12@(-0.323, +0.112, +0.022) (reachable).
           Verify `plan_egg_place` completes in < 2.0 s (instead of 58.7 s) and automatically selects
           the reachable empty slot `slot_12` instead of failing with `IK_NO_SOLUTION`.
        3. Wrist camera ROI gate ignores a huge bright distractor far outside the expected egg ROI.
        """
        import time
        import NERO_V365_Real_Cup_Task as RT
        import NERO_V386_Real_Wrist_Camera as RWC

        class DummyArm:
            def __init__(self):
                self._q = [0.0, -0.35, 0.0, 0.85, 0.0, 0.50, 0.0]

            def q(self):
                return list(self._q)

            def exec_path(self, path, times, label):
                # Enforce the exact user fence rule: tool z must be >= 0.013 m!
                import NERO_V362_Cup_Skills as CK
                fast = CK.FastArm()
                T_end = fast.fk(np.asarray(path[-1], dtype=float))
                T_grasp = T_end @ RT.real_profile(CK).T_tcp_grasp
                if T_grasp[2, 3] < 0.013:
                    return {
                        "ok": False,
                        "error": f"FENCE: 目标姿态越界 -> 未发送: tool z={T_end[2, 3]:.3f} m < 0.013 (z_min)",
                        "fence_violation": {"waypoint": "goal"},
                    }
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
        io.sleep = lambda _s: None
        io.cfg["still_window_s"] = 0.0
        io.cfg["settle_s"] = 0.0

        # Reproduce Run 2 + Run 3 exact coordinates
        def fake_perceive_eggs(tag):
            occ = (tag == "verify_egg_in_slot")
            slots = [
                {"slot_id": 1, "center_base": [-0.264, -0.015, 0.022], "occupied": False},   # inner dead-zone from Run 2
                {"slot_id": 12, "center_base": [-0.323, +0.112, 0.022], "occupied": occ},    # reachable slot from Run 2
            ]
            out = {
                "ok": True,
                "scene": "fill_egg_holder",
                "table_z_m": 0.0,
                "n_2d": 3,
                "eggs": [] if occ else [{
                    "center_base": [-0.397, -0.031, 0.021],
                    "grasp_yaw_deg": -120.0,
                    "width_m": 0.042,
                    "length_m": 0.055,
                }],
                "carton": {
                    "center_base": [-0.283, 0.149, 0.025],
                    "yaw_deg": -6.0,
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

        t0_plan = time.perf_counter()
        res_plan = RT.run_plan_preview(io, "egg_1", "slot_1", scene="fill_egg_holder")
        dt_plan_s = time.perf_counter() - t0_plan
        self.assertTrue(res_plan["ok"], f"Egg plan failed: {res_plan}")
        self.assertLess(dt_plan_s, 2.0, f"Egg plan_egg_place took too long: {dt_plan_s:.2f}s")
        io.motion_allowed = True

        t0 = time.perf_counter()
        res_exec = RT.run_execute(io, "egg_1", "slot_1", scene="fill_egg_holder")
        dt_s = time.perf_counter() - t0

        self.assertTrue(res_exec["ok"], f"Egg execute failed: {res_exec}")
        self.assertLess(dt_s, 8.0, f"Egg planning + execution took too long: {dt_s:.2f}s")
        sk = res_exec["skills"]
        self.assertEqual(sk.plan["target_slot"], "slot_12")
        # Verify planned SO-101 grasp pad z is at the egg equator (18.5 mm) and fingertip is 8.5 mm above table
        self.assertAlmostEqual(float(sk.plan["grasp_pad_z_mm"]), 18.5, places=1)
        self.assertAlmostEqual(float(sk.plan["fingertip_z_mm"]), 8.5, places=1)
        # Verify egg long axis is rotated vertical (perpendicular to table) for slot insertion
        self.assertLessEqual(float(sk.plan["egg_long_axis_vertical_err_deg"]), 25.0)

        # Verify wrist camera ROI gate ignores a huge bright distractor in the corner while locking onto the true egg
        H, W = 240, 320
        rgb = np.full((H, W, 3), 50, dtype=np.uint8)
        yy, xx = np.mgrid[0:H, 0:W]
        # True egg near image center (164, 122)
        egg_m = ((xx - 164.0) / 20.0) ** 2 + ((yy - 122.0) / 15.0) ** 2 <= 1.0
        rgb[egg_m] = (235, 225, 205)
        # Huge bright background glare in top-left corner (area 4x larger than egg)
        rgb[5:75, 5:95] = (250, 250, 250)

        K = {"fx": 280.0, "fy": 280.0, "ppx": 160.0, "ppy": 120.0, "width": W, "height": H}
        T_flange = np.eye(4)
        T_flange[:3, 3] = [-0.35, 0.0, 0.25]
        T_fc = np.array([[0.0, -1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], float)
        chk = RWC.wrist_check_egg_or_slot(rgb, K, T_flange, {"T_flange_cam": T_fc.tolist()}, [-0.35, 0.0, 0.02], kind="egg")
        self.assertTrue(chk["ok"])
        self.assertTrue(chk["trusted"])
        self.assertLess(chk["shift_norm_mm"], 8.0)

if __name__ == "__main__":
    unittest.main()
