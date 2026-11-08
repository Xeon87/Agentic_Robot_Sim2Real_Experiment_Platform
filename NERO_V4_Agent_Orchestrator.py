#!/usr/bin/env python3
"""NERO V4.2 — Top-Level Hierarchical LLM Agent Orchestrator.

NERO_V4_Agent_Orchestrator.py

Implements Section 3 & Phase 4 of the NERO V4.0 Architecture Specification:
1. Core Responsibilities of the Top-Level Agent:
   - Understand natural-language user tasks (e.g., "Put the red block into the left box, then put the blue block into the right box").
   - Decompose multi-step tasks into executable natural-language subtasks.
   - Select or apply the target manipulation policy (`"traditional"` vs. `"smolvla"`).
   - Supervise subtask execution through `PolicyRouter` and inspect independent verification from `TaskEvaluator`.
   - Replan / retry boundedly (max 1 retry per subtask) upon recoverable failure, or abort safely on safety trips.
2. Strict High/Low Level Separation (Section 3.2):
   - The Agent NEVER computes joint angles, grasp yaws, descent heights, or gripper opening sequences.
   - In the VLA path (`policy="smolvla"`), the Agent sends natural-language subtask instructions directly to SmolVLA via `PolicyRouter`.
3. Supports both Live LLM Tool-Calling (`NeroLLMAgent` from `NERO_V2_7_1_Agent.py`) and Deterministic Rule-Free Clause Decomposition for offline/reproducible benchmark runs.
"""
from __future__ import annotations

import json
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional

VERSION = "V4.6"
MARKER = "V4.2 TOP-LEVEL AGENT ORCHESTRATOR"

from NERO_V4_Policy_Router import PolicyRouter

V4_ORCHESTRATOR_TOOLS = [
    {
        "type": "function",
        "name": "observe_high_level_scene",
        "description": "Observe the high-level task state, robot readiness, and previously completed subtasks. Always call first.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        "strict": True,
    },
    {
        "type": "function",
        "name": "execute_subtask",
        "description": "Dispatch a single natural-language manipulation subtask to the selected policy ('traditional' or 'smolvla') and run independent completion verification.",
        "parameters": {
            "type": "object",
            "properties": {
                "instruction": {"type": "string", "description": "Natural-language subtask instruction, e.g. 'Put the red block into the left box'"},
                "policy": {"type": "string", "enum": ["traditional", "smolvla"]},
            },
            "required": ["instruction", "policy"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def decompose_natural_language_task(user_task: str) -> List[str]:
    """Decompose a compound natural-language instruction (Chinese or English) into ordered atomic subtask instructions."""
    raw = str(user_task or "").strip()
    if not raw:
        return []
    # Split on sequential connectors: "然后", "接着", "再", "then", "and then", ";", "；"
    parts = re.split(
        r"(?:\s+and\s+then\s+|\s+then\s+|，\s*然后|,\s*then\s*|；|;|。\s*然后|，\s*再\s*|，\s*接着\s*)",
        raw,
        flags=re.IGNORECASE,
    )
    cleaned = [p.strip(" ，,。.；;!") for p in parts if p and p.strip(" ，,。.；;!")]
    return cleaned if cleaned else [raw]


class AgentOrchestrator:
    """Unified Top-Level Agent that orchestrates multi-step tasks over either Traditional or SmolVLA policies."""

    def __init__(
        self,
        router: PolicyRouter,
        default_policy: str = "smolvla",
        max_retries_per_subtask: int = 1,
        llm_decompose_fn: Optional[Callable[[str], List[str]]] = None,
        log: Callable[[str], None] = print,
    ):
        self.router = router
        self.default_policy = str(default_policy).strip().lower()
        self.max_retries_per_subtask = int(max_retries_per_subtask)
        self.llm_decompose_fn = llm_decompose_fn
        self.log = log

        self.episode_logs: List[Dict[str, Any]] = []

    def plan_subtasks(self, user_task: str) -> Dict[str, Any]:
        """Decompose a high-level user task into ordered natural-language subtasks and record planning latency."""
        t0 = time.perf_counter()
        if self.llm_decompose_fn is not None:
            subtasks = self.llm_decompose_fn(user_task)
        else:
            subtasks = decompose_natural_language_task(user_task)
        dt_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        self.log(
            f"[OK] [V4.2 AGENT] 任务分解完成 ({len(subtasks)} 个子任务, {dt_ms:.2f} ms): "
            + " -> ".join(f"[{i + 1}] {s}" for i, s in enumerate(subtasks))
        )
        return {
            "user_task": user_task,
            "subtasks": subtasks,
            "plan_ms": dt_ms,
        }

    def run_task(
        self,
        user_task: str,
        policy: Optional[str] = None,
        timeout_per_subtask_s: float = 45.0,
        max_vla_chunks: int = 5,
        subtask_target_states: Optional[Dict[int, List[float]]] = None,
    ) -> Dict[str, Any]:
        """Execute a complete user task through hierarchical planning -> subtask dispatch -> outcome evaluation -> replanning."""
        t_start = time.perf_counter()
        chosen_policy = str(policy or self.default_policy).strip().lower()
        plan_info = self.plan_subtasks(user_task)
        subtasks = plan_info["subtasks"]

        if not subtasks:
            return {
                "ok": False,
                "user_task": user_task,
                "policy": chosen_policy,
                "error": "EMPTY_TASK",
            }

        subtask_records: List[Dict[str, Any]] = []
        replan_count = 0
        retry_count = 0
        plan_calls = 1
        total_plan_ms = float(plan_info["plan_ms"])

        for idx, sub_instr in enumerate(subtasks):
            target_8d = (subtask_target_states or {}).get(idx)
            attempt = 0
            sub_succeeded = False
            last_rec = None

            while attempt <= self.max_retries_per_subtask:
                attempt += 1
                tid = f"subtask_{idx + 1}_try_{attempt}"
                rec = self.router.execute_subtask(
                    instruction=sub_instr,
                    policy=chosen_policy,
                    task_id=tid,
                    timeout_s=timeout_per_subtask_s,
                    max_vla_chunks=max_vla_chunks,
                    target_state_8d=target_8d,
                )
                subtask_records.append(rec)
                last_rec = rec

                res = rec.get("result") or {}
                eval_info = res.get("evaluation") or {}
                if res.get("ok"):
                    sub_succeeded = True
                    break

                # Check if failure is a hard safety block or abort -> do not retry
                next_act = eval_info.get("recommended_agent_action", "replan_or_retry")
                if next_act == "stop_and_report" or attempt > self.max_retries_per_subtask:
                    self.log(f"[WARN] [V4.2 AGENT] 子任务 [{idx + 1}] 终止 ({res.get('reason')})，不再重试")
                    break

                # Recoverable failure -> Agent replans & retries with corrective prompt context
                replan_count += 1
                retry_count += 1
                plan_calls += 1
                t_rep = time.perf_counter()
                sub_instr = f"{sub_instr} (retry after {res.get('reason')})"
                total_plan_ms += (time.perf_counter() - t_rep) * 1000.0
                self.log(f"[INFO] [V4.2 AGENT] 子任务 [{idx + 1}] 未达标 ({res.get('reason')}) → 触发重规划与第 {attempt + 1} 次尝试")

            if not sub_succeeded:
                elapsed_s = round(time.perf_counter() - t_start, 3)
                last_res = (last_rec or {}).get("result") or {}
                unverified = last_res.get("status") == "unverified"
                if unverified:   # V4.6: executed, but nothing checked whether it worked
                    self.log(f"[WARN] [V4.6 AGENT] 子任务 [{idx + 1}] 已执行，但没有视觉 / 目标检查，无法确认是否完成 → "
                             f"停止（不当作成功，也不重试）。请人工确认现场。")
                else:
                    self.log(f"[FAIL] [V4.6 AGENT] 顶层任务失败：子任务 [{idx + 1}] {last_res.get('reason')}")
                out = {
                    "ok": False,
                    "status": "unverified" if unverified else "failed",
                    "user_task": user_task,
                    "policy": chosen_policy,
                    "subtasks_planned": len(subtasks),
                    "subtasks_succeeded": idx,
                    "subtask_success_rate": round(idx / float(len(subtasks)), 4),
                    "plan_calls": plan_calls,
                    "plan_total_ms": round(total_plan_ms, 2),
                    "replan_count": replan_count,
                    "retry_count": retry_count,
                    "elapsed_s": elapsed_s,
                    "failed_subtask": last_rec,
                    "history": subtask_records,
                }
                self.episode_logs.append(out)
                return out

        elapsed_s = round(time.perf_counter() - t_start, 3)
        out = {
            "ok": True,
            "status": "succeeded",
            "user_task": user_task,
            "policy": chosen_policy,
            "subtasks_planned": len(subtasks),
            "subtasks_succeeded": len(subtasks),
            "subtask_success_rate": 1.0,
            "plan_calls": plan_calls,
            "plan_total_ms": round(total_plan_ms, 2),
            "replan_count": replan_count,
            "retry_count": retry_count,
            "elapsed_s": elapsed_s,
            "history": subtask_records,
        }
        self.episode_logs.append(out)
        self.log(
            f"[OK] [V4.2 AGENT] 顶层任务全部完成 ({len(subtasks)}/{len(subtasks)} 子任务, policy={chosen_policy}, "
            f"重规划 {replan_count} 次, 总耗时 {elapsed_s:.2f} s)"
        )
        return out
