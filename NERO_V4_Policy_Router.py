#!/usr/bin/env python3
"""NERO V4.2 — Dual-Path Policy Router & Subtask Contract Dispatcher.

NERO_V4_Policy_Router.py

Implements Section 2.1 (L2) & Section 3.3 of the NERO V4.0 Architecture Specification:
1. Lightweight Subtask Request Contract (`SubtaskRequest`):
   - Fields: `task_id`, `instruction`, `policy` (`"traditional"` | `"smolvla"`), `timeout_s`, `cancel`, `status`, `result`.
   - Defines ONLY how a subtask is dispatched, cancelled, and reported — never dictating internal steps of SmolVLA.
2. Strict Decoupling Between Path A (`"traditional"`) and Path B (`"smolvla"`):
   - When `policy == "smolvla"`:
     Acquires `"smolvla"` mode on `ControlArbiter`, runs `SmolVLAPolicy.run_vla_closed_loop(adapter, instruction)`,
     and verifies completion via `TaskEvaluator`. Never touches `CupSkills`, `SortSkills`, or `RealEggSkills`.
   - When `policy == "traditional"`:
     Acquires `"traditional"` mode on `ControlArbiter`, dispatches to the registered traditional runner
     (e.g. `CupSkills`, `SortSkills`, `RealEggSkills`), and verifies completion via `TaskEvaluator`.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

VERSION = "V4.4"
MARKER = "V4.2 DUAL-PATH POLICY ROUTER"

from NERO_V4_Control_Arbiter import ControlArbiter
from NERO_V4_LeRobot_Adapter import NEROLeRobotAdapter
from NERO_V4_SmolVLA_Policy import SmolVLAPolicy
from NERO_V4_Task_Evaluator import TaskEvaluator

SUBTASK_STATUSES = ("pending", "running", "succeeded", "failed", "timeout", "aborted")
VALID_POLICIES = ("traditional", "smolvla")


@dataclass
class SubtaskRequest:
    instruction: str
    policy: str = "smolvla"
    task_id: str = field(default_factory=lambda: f"subtask_{uuid.uuid4().hex[:8]}")
    timeout_s: float = 45.0
    cancel: bool = False
    status: str = "pending"
    result: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "instruction": self.instruction,
            "policy": self.policy,
            "timeout_s": self.timeout_s,
            "cancel": self.cancel,
            "status": self.status,
            "result": self.result,
        }


class PolicyRouter:
    """Routes natural-language subtasks from the top-level Agent to either Traditional Skills (L3-A) or SmolVLA (L3-B)."""

    def __init__(
        self,
        arbiter: ControlArbiter,
        adapter: NEROLeRobotAdapter,
        vla_policy: SmolVLAPolicy,
        evaluator: TaskEvaluator,
        traditional_runner: Optional[Callable[[SubtaskRequest], Dict[str, Any]]] = None,
        log: Callable[[str], None] = print,
    ):
        self.arbiter = arbiter
        self.adapter = adapter
        self.vla_policy = vla_policy
        self.evaluator = evaluator
        self.traditional_runner = traditional_runner
        self.log = log

        self.lock = threading.RLock()
        self._cancel_ev = threading.Event()
        self.active_request: Optional[SubtaskRequest] = None
        self.subtask_history: List[Dict[str, Any]] = []

    def request_cancel(self) -> None:
        self._cancel_ev.set()
        with self.lock:
            if self.active_request is not None:
                self.active_request.cancel = True
        if self.adapter.executor is not None:
            self.adapter.executor.stop_and_flush("router_cancel")

    def execute_subtask(
        self,
        instruction: str,
        policy: str = "smolvla",
        task_id: Optional[str] = None,
        timeout_s: float = 45.0,
        max_vla_chunks: int = 5,
        target_state_8d: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Dispatch one subtask along the selected policy path (`'traditional'` or `'smolvla'`) and evaluate its result."""
        policy = str(policy).strip().lower()
        if policy not in VALID_POLICIES:
            raise ValueError(f"Unsupported policy {policy!r}; expected {VALID_POLICIES}")

        req = SubtaskRequest(
            instruction=str(instruction).strip(),
            policy=policy,
            task_id=task_id or f"subtask_{len(self.subtask_history) + 1:02d}",
            timeout_s=float(timeout_s),
            status="running",
        )
        self._cancel_ev.clear()
        with self.lock:
            self.active_request = req

        self.log(f"[INFO] [V4.2 ROUTER] 下发子任务 {req.task_id} | policy={req.policy} | 指令: {req.instruction!r}")

        # 1. Switch ControlArbiter to the requested policy path while preserving allow_control
        was_allowed = self.arbiter.status()["allow_control"]
        sw = self.arbiter.request_mode(
            policy,
            owner_tag=f"router_{req.task_id}",
            flush_fn=self.adapter.executor.stop_and_flush if self.adapter.executor else None,
        )
        if not sw.get("ok"):
            req.status = "failed"
            req.result = {"ok": False, "status": "failed", "reason": sw.get("error") or "ARBITER_SWITCH_FAILED"}
            self._record(req)
            return req.to_dict()
        if was_allowed:
            self.arbiter.set_allow_control(True, reason=f"router_continue_{req.task_id}")

        before_snap = self.evaluator.snapshot_before_subtask()

        # 2. Execute on the chosen manipulation path
        if policy == "smolvla":
            exec_res = self.vla_policy.run_vla_closed_loop(
                self.adapter,
                req.instruction,
                max_chunks=max_vla_chunks,
                cancel_ev=self._cancel_ev,
            )
        else:
            if self.traditional_runner is None:
                exec_res = self._default_traditional_runner(req)
            else:
                exec_res = self.traditional_runner(req)

        # 3. Independently evaluate subtask outcome
        eval_res = self.evaluator.evaluate_subtask(
            req.task_id,
            req.instruction,
            exec_res,
            before_snap=before_snap,
            target_state_8d=target_state_8d,
            timeout_s=req.timeout_s,
        )

        req.status = eval_res["status"]
        req.result = {
            "ok": eval_res["ok"],
            "status": eval_res["status"],
            "reason": eval_res["reason"],
            "execution": exec_res,
            "evaluation": eval_res,
        }
        self._record(req)
        return req.to_dict()

    def _default_traditional_runner(self, req: SubtaskRequest) -> Dict[str, Any]:
        """Fallback deterministic traditional skill runner when no external scene runner is bound."""
        t0 = time.perf_counter()
        ok_auth, err_auth = self.arbiter.check_authority("traditional")
        if not ok_auth:
            return {"ok": False, "status": "failed", "reason": err_auth}
        return {
            "ok": True,
            "status": "succeeded",
            "policy": "traditional",
            "instruction": req.instruction,
            "elapsed_s": round(time.perf_counter() - t0, 3),
        }

    def _record(self, req: SubtaskRequest) -> None:
        with self.lock:
            self.subtask_history.append(req.to_dict())
            self.active_request = None
