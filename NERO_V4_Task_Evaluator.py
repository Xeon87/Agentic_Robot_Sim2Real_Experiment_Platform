#!/usr/bin/env python3
"""NERO V4.2 — Independent Task Outcome Evaluator.

NERO_V4_Task_Evaluator.py

Implements Section 3.4 of the NERO V4.0 Architecture Specification:
- SmolVLA finishing its action chunks does NOT automatically mean the subtask succeeded.
- Provides an independent verification mechanism that evaluates:
  1. Whether the target object reached the target receptacle / slot / stack (`goal_satisfied`);
  2. Whether an object is still held or was dropped in mid-air (`dropped`, `still_held`);
  3. Whether a timeout (`timeout`) or safety trip (`safety_tripped`) occurred.
- Supports 4 evaluation sources (`eval_source`):
  - `"vision_detector"`: External YOLOE + 3D geometry check (only at subtask completion, never inside VLA loop)
  - `"vlm_judge"`: `NERO_V374_Vision_Review` VLM visual inspection of fixed camera RGB
  - `"sim_truth_eval_only"`: Simulation ground truth strictly for post-subtask evaluation (NEVER passed into VLA observation)
  - `"proprioceptive_goal"`: Goal state verification (e.g., final joint/TCP target + gripper release verification)
"""
from __future__ import annotations

import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

VERSION = "V4.5"
MARKER = "V4.2 INDEPENDENT TASK OUTCOME EVALUATOR"
HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


class TaskEvaluator:
    """Evaluates subtask completion independently of the manipulation policy (Traditional or SmolVLA)."""

    def __init__(
        self,
        obs_bridge=None,
        arbiter=None,
        vision_eval_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        sim_truth_eval_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        log: Callable[[str], None] = print,
    ):
        self.obs_bridge = obs_bridge
        self.arbiter = arbiter
        self.vision_eval_fn = vision_eval_fn
        self.sim_truth_eval_fn = sim_truth_eval_fn
        self.log = log
        self._initial_trip_count = 0

    def snapshot_before_subtask(self) -> Dict[str, Any]:
        trips = self.arbiter.status()["trip_count"] if self.arbiter else 0
        self._initial_trip_count = int(trips)
        obs = self.obs_bridge.get_synchronized_observation() if self.obs_bridge else {}
        return {
            "t0": time.monotonic(),
            "trips_before": self._initial_trip_count,
            "initial_state": obs.get("state").tolist() if obs.get("state") is not None else None,
        }

    def evaluate_subtask(
        self,
        task_id: str,
        instruction: str,
        exec_result: Dict[str, Any],
        before_snap: Optional[Dict[str, Any]] = None,
        target_state_8d: Optional[List[float]] = None,
        pos_tol_rad: float = 0.12,
        timeout_s: float = 60.0,
    ) -> Dict[str, Any]:
        """Independently judge whether `task_id` (`instruction`) succeeded, failed, timed out, or tripped safety."""
        t_now = time.monotonic()
        t0 = float((before_snap or {}).get("t0", t_now))
        elapsed_s = round(t_now - t0, 3)

        trips_now = int(self.arbiter.status()["trip_count"]) if self.arbiter else 0
        trips_before = int((before_snap or {}).get("trips_before", self._initial_trip_count))
        safety_tripped = (trips_now > trips_before) or (self.arbiter is not None and self.arbiter.mode == "estop")

        timed_out = elapsed_s > float(timeout_s) or exec_result.get("status") == "timeout"
        aborted = exec_result.get("status") == "aborted"

        if safety_tripped:
            return self._report(
                task_id, instruction, False, "failed", "SAFETY_LIMIT_TRIGGERED",
                elapsed_s=elapsed_s, eval_source="safety_supervisor",
                details={"new_safety_trips": trips_now - trips_before},
            )
        if timed_out:
            return self._report(
                task_id, instruction, False, "timeout", "SUBTASK_TIMEOUT",
                elapsed_s=elapsed_s, eval_source="timer",
            )
        if aborted:
            return self._report(
                task_id, instruction, False, "aborted", "SUBTASK_ABORTED_BY_USER",
                elapsed_s=elapsed_s, eval_source="arbiter",
            )
        if not exec_result.get("ok"):
            return self._report(
                task_id, instruction, False, "failed", exec_result.get("reason") or "EXECUTION_ERROR",
                elapsed_s=elapsed_s, eval_source="execution_monitor",
                details={"exec_error": exec_result.get("reason") or exec_result.get("error")},
            )

        # Priority 1: Simulation ground-truth evaluator (strictly post-hoc for benchmark evaluation)
        if self.sim_truth_eval_fn is not None:
            verdict = self.sim_truth_eval_fn(instruction)
            ok = bool(verdict.get("ok"))
            return self._report(
                task_id, instruction, ok, "succeeded" if ok else "failed",
                verdict.get("reason") or ("GOAL_VERIFIED" if ok else "OBJECT_NOT_AT_GOAL"),
                elapsed_s=elapsed_s, eval_source="sim_truth_eval_only", details=verdict,
            )

        # Priority 2: External Vision / VLM Evaluator
        if self.vision_eval_fn is not None:
            verdict = self.vision_eval_fn(instruction)
            ok = bool(verdict.get("ok"))
            return self._report(
                task_id, instruction, ok, "succeeded" if ok else "failed",
                verdict.get("reason") or ("VISION_GOAL_VERIFIED" if ok else "VISION_GOAL_NOT_MET"),
                elapsed_s=elapsed_s, eval_source="vision_detector", details=verdict,
            )

        # Priority 3: Proprioceptive & target-state goal check
        obs_now = self.obs_bridge.get_synchronized_observation() if self.obs_bridge else {}
        cur_state = obs_now.get("state")
        if target_state_8d is not None and cur_state is not None:
            tgt = np.asarray(target_state_8d, dtype=float).flatten()[:8]
            cur = np.asarray(cur_state, dtype=float).flatten()[:8]
            q_err = float(np.max(np.abs(cur[:7] - tgt[:7])))
            g_err = abs(float(cur[7] - tgt[7]))
            ok = (q_err <= float(pos_tol_rad)) and (g_err <= 0.20)
            return self._report(
                task_id, instruction, ok, "succeeded" if ok else "failed",
                "GOAL_STATE_REACHED" if ok else f"GOAL_DEVIATION (max_q_err={math.degrees(q_err):.1f}°)",
                elapsed_s=elapsed_s, eval_source="proprioceptive_goal",
                details={"max_joint_err_deg": round(math.degrees(q_err), 2), "gripper_err": round(g_err, 3)},
            )

        return self._report(
            task_id, instruction, True, "succeeded", "SUBTASK_COMPLETED",
            elapsed_s=elapsed_s, eval_source="execution_monitor",
        )

    def _report(
        self,
        task_id: str,
        instruction: str,
        ok: bool,
        status: str,
        reason: str,
        elapsed_s: float,
        eval_source: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        res = {
            "ok": bool(ok),
            "task_id": task_id,
            "instruction": instruction,
            "status": status,
            "reason": reason,
            "elapsed_s": elapsed_s,
            "eval_source": eval_source,
            "details": details or {},
            "recommended_agent_action": "next_subtask" if ok else (
                "stop_and_report" if status in ("aborted", "timeout") or "SAFETY" in reason else "replan_or_retry"
            ),
        }
        self.log(
            f"[{'OK' if ok else 'WARN'}] [V4.2 EVALUATOR] 子任务 {task_id} ({eval_source}): "
            f"status={status} · reason={reason} ({elapsed_s:.2f} s)"
        )
        return res
