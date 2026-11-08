#!/usr/bin/env python3
"""NERO V4.1 — VLA Action Chunk Executor & Receding-Horizon Trajectory Dispatcher.

NERO_V4_VLA_Action_Executor.py

Implements Section 5.6 & Section 5.7 of the NERO V4.0 Architecture Specification:
1. Receives 8D Action Chunks (`[J1..J7 (rad), Gripper_norm (0..1)]`) or single 8D actions from SmolVLA / LeRobot.
2. Passes every action or chunk through `ControlArbiter` (exclusive authority check + strategy-independent
   joint limit, velocity, workspace fence, and tabletop collision checks).
3. Executes approved actions on the selected backend WITHOUT invoking MoveIt OMPL planning or any traditional skill:
   - MuJoCo Twin (`twin_io`): Directly drives the continuous joint path (`_apply_q` / `step`) and gripper opening.
   - MuJoCo Unified Backend (`shared_arrays`): Directly writes `target_shared[0..6]` and `gripper_target_mm`.
   - Real NERO (`real_pipe` + `real_gripper`): Sends pre-validated short joint trajectory chunks to `/execute_trajectory`
     (`move_joint_path`) and updates `SO101RealGripper.open_gap()`.
4. Supports Receding-Horizon Chunk Execution (`execute_horizon_steps` out of `chunk_length`), immediate cancellation,
   and timeout/heartbeat protection.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

VERSION = "V4.4"
MARKER = "V4.1 VLA ACTION CHUNK EXECUTOR"
HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from NERO_V4_Control_Arbiter import ControlArbiter
from NERO_V4_Observation_Bridge import ObservationBridge, gripper_norm_to_mm

DEFAULT_EXECUTOR_CFG = {
    "control_hz": 20.0,              # Default action playback rate (20 Hz -> dt = 50 ms)
    "chunk_horizon": 16,             # Default predicted chunk length H
    "execute_horizon": 8,            # Number of steps K <= H executed before next VLA re-inference
    "clamp_in_place": True,          # Smoothly clamp small step/soft-limit overshoots; hard fence violations abort
}


class VLAActionExecutor:
    """Executes SmolVLA / LeRobot 8D actions and Action Chunks across MuJoCo and Real NERO under ControlArbiter."""

    def __init__(
        self,
        arbiter: ControlArbiter,
        obs_bridge: ObservationBridge,
        cfg: Optional[Dict[str, Any]] = None,
        log: Callable[[str], None] = print,
    ):
        self.arbiter = arbiter
        self.obs_bridge = obs_bridge
        self.cfg = dict(DEFAULT_EXECUTOR_CFG, **(cfg or {}))
        self.log = log
        self.lock = threading.RLock()

        self._cancel_event = threading.Event()
        self._action_queue: List[List[float]] = []
        self._steps_executed: int = 0
        self._chunks_executed: int = 0
        self._last_action: Optional[List[float]] = None

        # Backend bindings
        self._backend_kind: str = "mock"
        self._twin_io = None
        self._sim_shared: Optional[Dict[str, Any]] = None
        self._real_pipe = None
        self._real_model_fn: Optional[Callable[[], Any]] = None
        self._real_gripper = None

        # Register stop callback on arbiter so mode switches or E-Stop immediately flush the queue
        self.arbiter.on_stop_callback = self.stop_and_flush

    # ------------------------------------------------------------------ Backend Bindings
    def bind_mujoco_twin(self, twin_io) -> None:
        """Bind to headless MuJoCo TwinIO (`NERO_V362_Cup_Skills.TwinIO` or `SortTwinIO`)."""
        with self.lock:
            self._backend_kind = "mujoco_twin"
            self._twin_io = twin_io
            self.obs_bridge.bind_mujoco_twin(twin_io)

    def bind_mujoco_shared(self, target_shared, gripper_target_mm, actual_shared=None, gripper_actual_mm=None) -> None:
        """Bind to `NERO_V2_7_1_Unified_Backend` multiprocessing shared arrays."""
        with self.lock:
            self._backend_kind = "mujoco_shared"
            self._sim_shared = {
                "target_shared": target_shared,
                "gripper_target_mm": gripper_target_mm,
                "actual_shared": actual_shared,
                "gripper_actual_mm": gripper_actual_mm,
            }

    def bind_real_robot(self, pipe, model_fn: Callable[[], Any], gripper=None, cam_tab=None, jpanel=None) -> None:
        """Bind to RealPipeline (`NERO_V313_Real_Robot`) and `SO101RealGripper` (`NERO_V365_STS3215_Gripper`)."""
        with self.lock:
            self._backend_kind = "real"
            self._real_pipe = pipe
            self._real_model_fn = model_fn
            self._real_gripper = gripper
            if jpanel is not None or cam_tab is not None:
                self.obs_bridge.bind_real_sources(pipe=pipe, jpanel=jpanel, cam_tab=cam_tab, gripper=gripper)

    def bind_mock(self) -> None:
        with self.lock:
            self._backend_kind = "mock"
            self.obs_bridge.start_mock_sources()

    # ------------------------------------------------------------------ Queue & Emergency Stop
    def stop_and_flush(self, reason: str = "stopped") -> None:
        """Cancel any active chunk playback and clear queued actions."""
        self._cancel_event.set()
        with self.lock:
            n_dropped = len(self._action_queue)
            self._action_queue.clear()
        if self._backend_kind == "real" and self._real_pipe is not None:
            try:
                if self._real_pipe.move_busy():
                    self._real_pipe.cancel_move()
            except Exception:
                pass
        if n_dropped > 0:
            self.log(f"[INFO] [V4.1 EXECUTOR] 已清空待执行动作队列 ({n_dropped} 步) · 原因: {reason}")

    def status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "backend_kind": self._backend_kind,
                "control_hz": float(self.cfg["control_hz"]),
                "queued_steps": len(self._action_queue),
                "steps_executed": self._steps_executed,
                "chunks_executed": self._chunks_executed,
                "last_action": self._last_action,
                "arbiter": self.arbiter.status(),
            }

    # ------------------------------------------------------------------ Single-Step & Chunk Execution
    def send_action(
        self,
        action_8d: List[float],
        caller_mode: str = "smolvla",
        dt_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Validate and execute a single 8D action `[j1..j7 (rad), gripper_norm (0..1)]`."""
        self._cancel_event.clear()
        dt = float(dt_s if dt_s is not None else (1.0 / max(1.0, float(self.cfg["control_hz"]))))
        obs = self.obs_bridge.get_synchronized_observation()
        q_cur = obs["robot_state"].tolist()

        arr = np.asarray(action_8d, dtype=float).flatten()
        if len(arr) != 8:
            return {"ok": False, "error": f"Expected 8D action (7 joints + 1 gripper), got {len(arr)}D"}

        val = self.arbiter.validate_and_clamp_step(
            q_cur,
            arr[:7].tolist(),
            gripper_norm=float(arr[7]),
            dt_s=dt,
            caller_mode=caller_mode,
            clamp_in_place=bool(self.cfg.get("clamp_in_place", True)),
        )
        if not val["ok"]:
            return val

        q_safe = val["q_safe_rad"]
        g_norm = val["gripper_norm"]
        disp = self._dispatch_single_step(q_safe, g_norm, dt)
        if not disp["ok"]:
            return disp

        with self.lock:
            self._steps_executed += 1
            self._last_action = list(q_safe) + [g_norm]

        return {
            "ok": True,
            "q_executed_rad": q_safe,
            "gripper_norm": g_norm,
            "clamped": val["clamped"],
            "tcp_xyz_m": val.get("tcp_xyz_m"),
            "backend": self._backend_kind,
        }

    def execute_action_chunk(
        self,
        chunk: List[List[float]],
        execute_steps: Optional[int] = None,
        caller_mode: str = "smolvla",
        dt_s: Optional[float] = None,
        sleep_between_steps: bool = False,
    ) -> Dict[str, Any]:
        """Validate and execute an Action Chunk of shape `[H, 8]` (Receding-Horizon execution).

        1. Slices the chunk to `execute_steps` (default `cfg['execute_horizon']`).
        2. Pre-validates all `K` steps through `ControlArbiter.validate_action_chunk()`.
        3. Dispatches the validated steps to MuJoCo or Real NERO, checking `_cancel_event` at every step.
        """
        self._cancel_event.clear()
        dt = float(dt_s if dt_s is not None else (1.0 / max(1.0, float(self.cfg["control_hz"]))))
        k_max = int(execute_steps if execute_steps is not None else self.cfg.get("execute_horizon", len(chunk)))
        sub_chunk = [list(map(float, r[:8])) for r in list(chunk)[: max(1, k_max)]]

        obs = self.obs_bridge.get_synchronized_observation()
        q_start = obs["robot_state"].tolist()

        val = self.arbiter.validate_action_chunk(
            q_start,
            sub_chunk,
            dt_s=dt,
            caller_mode=caller_mode,
            clamp_in_place=bool(self.cfg.get("clamp_in_place", True)),
        )
        if not val["ok"]:
            return val

        safe_chunk = val["safe_chunk"]
        with self.lock:
            self._action_queue = list(safe_chunk)

        if self._backend_kind == "real":
            res = self._dispatch_real_chunk(q_start, safe_chunk, dt)
            with self.lock:
                self._action_queue.clear()
                if res.get("ok"):
                    self._steps_executed += len(safe_chunk)
                    self._chunks_executed += 1
                    self._last_action = safe_chunk[-1]
            return res

        executed = 0
        for step_action in safe_chunk:
            if self._cancel_event.is_set() or self.arbiter.mode == "estop":
                with self.lock:
                    self._action_queue.clear()
                return {
                    "ok": False,
                    "aborted": True,
                    "executed_steps": executed,
                    "error": "CHUNK_CANCELLED_OR_ESTOP",
                }
            r_step = self._dispatch_single_step(step_action[:7], step_action[7], dt)
            if not r_step.get("ok"):
                with self.lock:
                    self._action_queue.clear()
                return r_step
            executed += 1
            with self.lock:
                if self._action_queue:
                    self._action_queue.pop(0)
                self._steps_executed += 1
                self._last_action = step_action
            if sleep_between_steps:
                time.sleep(dt)

        with self.lock:
            self._chunks_executed += 1

        return {
            "ok": True,
            "executed_steps": executed,
            "clamped": val["clamped"],
            "final_action": safe_chunk[-1],
            "backend": self._backend_kind,
        }

    # ------------------------------------------------------------------ Low-Level Hardware / Sim Dispatch
    def _dispatch_single_step(self, q_rad: List[float], gripper_norm: float, dt_s: float) -> Dict[str, Any]:
        gap_mm = gripper_norm_to_mm(gripper_norm)

        if self._backend_kind == "mujoco_twin" and self._twin_io is not None:
            try:
                # Drive TwinIO directly without invoking any CupSkills or SortSkills logic
                self._twin_io._apply_q(np.asarray(q_rad, dtype=float))
                self._twin_io.q = [float(v) for v in q_rad]
                self._twin_io.gripper_open(gap_mm)
                n_sub = max(2, int(round(dt_s / 0.005)))
                self._twin_io.step(n_sub)
                self.obs_bridge.update_robot_state(q_rad, ts_ns=time.monotonic_ns())
                self.obs_bridge.update_gripper_state(gap_mm=gap_mm, gripper_norm=gripper_norm, ts_ns=time.monotonic_ns())
                return {"ok": True}
            except Exception as exc:
                return {"ok": False, "error": f"MuJoCo Twin dispatch error: {exc}"}

        if self._backend_kind == "mujoco_shared" and self._sim_shared is not None:
            try:
                tgt = self._sim_shared["target_shared"]
                gtgt = self._sim_shared["gripper_target_mm"]
                for i in range(7):
                    tgt[i] = float(q_rad[i])
                gtgt.value = float(gap_mm)
                self.obs_bridge.update_robot_state(q_rad, ts_ns=time.monotonic_ns())
                self.obs_bridge.update_gripper_state(gap_mm=gap_mm, gripper_norm=gripper_norm, ts_ns=time.monotonic_ns())
                return {"ok": True}
            except Exception as exc:
                return {"ok": False, "error": f"MuJoCo shared-array dispatch error: {exc}"}

        if self._backend_kind == "real" and self._real_pipe is not None:
            q_cur = self.obs_bridge.get_synchronized_observation()["robot_state"].tolist()
            return self._dispatch_real_chunk(q_cur, [list(q_rad) + [gripper_norm]], dt_s)

        # Mock mode: update ObservationBridge directly
        now_ns = time.monotonic_ns()
        self.obs_bridge.update_robot_state(q_rad, ts_ns=now_ns)
        self.obs_bridge.update_gripper_state(gap_mm=gap_mm, gripper_norm=gripper_norm, ts_ns=now_ns)
        return {"ok": True}

    def _dispatch_real_chunk(
        self,
        q_start: List[float],
        safe_chunk: List[List[float]],
        dt_s: float,
    ) -> Dict[str, Any]:
        """Dispatch a pre-validated chunk of joint + gripper targets to the real NERO arm + STS3215 gripper."""
        if self._real_pipe is None or self._real_model_fn is None:
            return {"ok": False, "error": "RealPipeline or robot model not bound"}
        m = self._real_model_fn()
        if not m:
            return {"ok": False, "error": "Real robot model not ready (Step 5/6)"}

        path = [list(q_start)] + [list(row[:7]) for row in safe_chunk]
        times = [round(i * float(dt_s), 4) for i in range(len(path))]
        r = self._real_pipe.move_joint_path(m, path, times, label="V4.1 VLA chunk")
        if not r.get("ok"):
            return {"ok": False, "error": r.get("error") or "Real move_joint_path failed"}

        last_g_norm = float(safe_chunk[-1][7])
        gap_mm = gripper_norm_to_mm(last_g_norm)
        if self._real_gripper is not None and getattr(self._real_gripper, "connected", False):
            try:
                if last_g_norm <= 0.08:
                    self._real_gripper.clamp(25.0)
                else:
                    self._real_gripper.open_gap(gap_mm)
            except Exception as exc:
                self.log(f"[WARN] [V4.1 EXECUTOR] 夹爪动作异常: {exc}")

        now_ns = time.monotonic_ns()
        self.obs_bridge.update_robot_state(safe_chunk[-1][:7], ts_ns=now_ns)
        self.obs_bridge.update_gripper_state(gap_mm=gap_mm, gripper_norm=last_g_norm, ts_ns=now_ns)
        return {
            "ok": True,
            "executed_steps": len(safe_chunk),
            "final_action": safe_chunk[-1],
            "backend": "real",
        }
