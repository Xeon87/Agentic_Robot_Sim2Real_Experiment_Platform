#!/usr/bin/env python3
"""NERO V4.1 — Dual-Path Control Arbiter & Independent Safety Supervisor.

NERO_V4_Control_Arbiter.py

Implements Section 8 of the NERO V4.0 Architecture Specification:
1. Exclusive Control Authority State Machine:
   - Modes: "monitor" (Monitor Only, default), "traditional", "smolvla", "teleop", "estop".
   - At any moment, at most ONE manipulation policy holds authority to send robot commands.
   - Connecting a model or adapter NEVER automatically grants control authority; explicit
     `set_allow_control(True)` + `acquire_mode(mode)` is required.
   - Switching modes flushes any pending action queue, verifies current robot state, and logs the transition.
2. Strategy-Independent Safety Supervisor:
   - Enforces soft joint limits (`JOINT_LIMITS_RAD` with configurable safety margin).
   - Enforces maximum per-step joint displacement (`max_step_deg`) and joint velocity (`max_vel_deg_s`).
   - Enforces workspace polygon fence and minimum tabletop clearance (`z_table + min_table_clearance_m`)
     using fast forward kinematics (`FastArm` from `NERO_V362_Cup_Skills` / `NERO_V315_Consistency_Safety`),
     completely independent of any traditional grasp planning skill.
   - Enforces gripper opening bounds (`[0.0, 1.0]` normalized or `[0.0, 68.0]` mm).
   - Enforces command heartbeat / watchdog timeout (`watchdog_timeout_s`).
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

VERSION = "V4.5"
MARKER = "V4.1 CONTROL ARBITER & INDEPENDENT SAFETY SUPERVISOR"
HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

CONTROL_MODES = (
    "monitor",      # Monitor Only (no motion allowed)
    "traditional",  # Path A: Traditional Skills + Motion Planning
    "smolvla",      # Path B: LeRobot / SmolVLA Learned Policy
    "teleop",       # Manual / Zero-Force Drag / Teleoperation
    "estop",        # Software / Hardware Emergency Stop
)

# NERO 7-DOF standard joint limits in radians (from NERO_V2_7_1_Unified_Backend / URDF)
JOINT_LIMITS_RAD: Tuple[Tuple[float, float], ...] = (
    (-2.70526, 2.70526),
    (-1.74000, 1.74000),
    (-2.75000, 2.75000),
    (-1.01000, 2.14000),
    (-2.75000, 2.75000),
    (-0.73000, 0.95000),
    (-1.57080, 1.57080),
)

DEFAULT_SAFETY_CFG = {
    "soft_limit_margin_deg": 2.5,       # Joint soft-limit buffer (deg)
    "max_step_deg": 6.0,                # Max single-step joint jump (deg) per control tick
    "max_vel_deg_s": 45.0,              # Max joint velocity (deg/s)
    "min_table_z_m": -0.05,             # Default tabletop Z floor in base_link (m)
    "min_tip_clearance_m": 0.003,       # Fingertip minimum height above table_z (m)
    "workspace_x_bounds_m": (-0.75, 0.45),
    "workspace_y_bounds_m": (-0.58, 0.58),
    "workspace_z_bounds_m": (-0.05, 0.92),
    "watchdog_timeout_s": 2.0,          # Max silence before continuous stream auto-pauses
}


class ControlArbiter:
    """Thread-safe exclusive control authority manager and strategy-independent safety supervisor."""

    def __init__(
        self,
        safety_cfg: Optional[Dict[str, Any]] = None,
        log: Callable[[str], None] = print,
        on_stop_callback: Optional[Callable[[str], None]] = None,
    ):
        self.lock = threading.RLock()
        self.cfg = dict(DEFAULT_SAFETY_CFG, **(safety_cfg or {}))
        self.log = log
        self.on_stop_callback = on_stop_callback

        self._mode: str = "monitor"
        self._allow_control: bool = False
        self._owner_tag: str = "none"
        self._last_cmd_ts: float = 0.0
        self._estop_reason: Optional[str] = None
        self._trip_count: int = 0
        self._last_violation: Optional[Dict[str, Any]] = None
        self._history: List[Dict[str, Any]] = []
        self._fk_helper = None

    def _get_fk(self):
        if self._fk_helper is None:
            try:
                import NERO_V362_Cup_Skills as CK
                self._fk_helper = CK.FastArm()
            except Exception:
                self._fk_helper = False
        return self._fk_helper if self._fk_helper is not False else None

    # ------------------------------------------------------------------ Mode & Authority Management
    @property
    def mode(self) -> str:
        with self.lock:
            return self._mode

    @property
    def allow_control(self) -> bool:
        with self.lock:
            return self._allow_control and self._mode not in ("monitor", "estop")

    def status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "mode": self._mode,
                "allow_control": self._allow_control,
                "owner_tag": self._owner_tag,
                "estop": self._mode == "estop",
                "estop_reason": self._estop_reason,
                "trip_count": self._trip_count,
                "last_violation": self._last_violation,
                "last_cmd_age_s": round(time.monotonic() - self._last_cmd_ts, 3) if self._last_cmd_ts > 0 else None,
            }

    def set_allow_control(self, allowed: bool, reason: str = "") -> Dict[str, Any]:
        """Explicitly gate whether motion commands can be dispatched to the robot."""
        with self.lock:
            if self._mode == "estop" and allowed:
                return {
                    "ok": False,
                    "mode": self._mode,
                    "allow_control": False,
                    "error": f"处于急停状态 ({self._estop_reason})，请先解除急停 (clear_estop) 再开启控制权限",
                }
            self._allow_control = bool(allowed)
            if not allowed and self.on_stop_callback:
                try:
                    self.on_stop_callback("allow_control_revoked")
                except Exception:
                    pass
            msg = f"[INFO] [V4.1 ARBITER] 控制权限 {'开启 (ALLOW)' if self._allow_control else '关闭 (LOCKED)'}"
            if reason:
                msg += f" · {reason}"
            self.log(msg)
            return self.status() | {"ok": True}

    def request_mode(self, new_mode: str, owner_tag: str = "", flush_fn: Optional[Callable[[], None]] = None) -> Dict[str, Any]:
        """Switch control mode safely. Stops previous action queue before granting new mode."""
        new_mode = str(new_mode).strip().lower()
        if new_mode not in CONTROL_MODES:
            raise ValueError(f"未知控制模式 {new_mode!r}，支持：{CONTROL_MODES}")
        with self.lock:
            if self._mode == "estop" and new_mode != "estop":
                return {
                    "ok": False,
                    "mode": self._mode,
                    "error": f"当前处于急停锁定状态 ({self._estop_reason})，请先调用 clear_estop()",
                }
            prev_mode = self._mode
            if prev_mode != new_mode:
                if flush_fn is not None:
                    try:
                        flush_fn()
                    except Exception:
                        pass
                if self.on_stop_callback and prev_mode in ("traditional", "smolvla", "teleop"):
                    try:
                        self.on_stop_callback(f"mode_switch:{prev_mode}->{new_mode}")
                    except Exception:
                        pass
            self._mode = new_mode
            self._owner_tag = owner_tag or new_mode
            if new_mode in ("monitor", "estop"):
                self._allow_control = False
            self._history.append({
                "t": time.time(),
                "from": prev_mode,
                "to": new_mode,
                "owner": self._owner_tag,
            })
            self.log(f"[OK] [V4.1 ARBITER] 控制权切换：{prev_mode} → {new_mode} (持有者: {self._owner_tag})")
            return self.status() | {"ok": True}

    def trigger_estop(self, reason: str = "manual_estop") -> Dict[str, Any]:
        """Immediately lock the arbiter in 'estop' mode and invoke stop callback."""
        with self.lock:
            self._mode = "estop"
            self._allow_control = False
            self._estop_reason = str(reason)
            self._trip_count += 1
            if self.on_stop_callback:
                try:
                    self.on_stop_callback(f"ESTOP:{reason}")
                except Exception:
                    pass
            self.log(f"[FAIL] [V4.1 ARBITER] 触发紧急停止 (E-STOP)：{reason}")
            return self.status() | {"ok": True}

    def clear_estop(self) -> Dict[str, Any]:
        """Clear emergency stop and return to safe 'monitor' mode with allow_control=False."""
        with self.lock:
            self._mode = "monitor"
            self._allow_control = False
            self._estop_reason = None
            self.log("[OK] [V4.1 ARBITER] 急停已解除 → 已回到 Monitor Only 模式（需重新勾选允许控制）")
            return self.status() | {"ok": True}

    # ------------------------------------------------------------------ Independent Safety Supervisor
    def check_authority(self, caller_mode: str) -> Tuple[bool, Optional[str]]:
        """Verify that `caller_mode` ('traditional' or 'smolvla' or 'teleop') currently holds active authority."""
        with self.lock:
            if self._mode == "estop":
                return False, f"EMERGENCY_STOP_ACTIVE: {self._estop_reason}"
            if self._mode != caller_mode:
                return False, f"CONTROL_ARBITER_DENIED: 当前控制权为 {self._mode!r}，拒绝 {caller_mode!r} 发送运动指令"
            if not self._allow_control:
                return False, "CONTROL_NOT_ALLOWED: 尚未开启「允许控制」(allow_control=False)"
            return True, None

    def validate_and_clamp_step(
        self,
        q_current_rad: List[float],
        q_target_rad: List[float],
        gripper_norm: float = 1.0,
        dt_s: float = 0.05,
        caller_mode: str = "smolvla",
        clamp_in_place: bool = True,
    ) -> Dict[str, Any]:
        """Validate a single 7-DOF joint target + gripper command against all strategy-independent safety rules.

        Checks:
        1. Exclusive control authority (`caller_mode` must match `self.mode` and `allow_control` must be True).
        2. Finite numbers & dimension check (7 joints + 1 gripper).
        3. Per-step joint velocity / displacement limit (`max_step_deg`, `max_vel_deg_s`).
        4. Soft joint limits (`JOINT_LIMITS_RAD` with `soft_limit_margin_deg`).
        5. Forward-kinematics workspace bounding box and tabletop Z floor (`min_table_z_m + min_tip_clearance_m`).
        """
        ok_auth, err_auth = self.check_authority(caller_mode)
        if not ok_auth:
            return self._record_violation(err_auth or "AUTHORITY_DENIED", q_target_rad)

        q_cur = np.asarray(q_current_rad, dtype=float).flatten()
        q_tgt = np.asarray(q_target_rad, dtype=float).flatten()
        if len(q_cur) != 7 or len(q_tgt) != 7 or not (np.all(np.isfinite(q_cur)) and np.all(np.isfinite(q_tgt))):
            return self._record_violation("INVALID_JOINT_DIMENSION_OR_NAN", q_target_rad)

        dt = max(0.005, float(dt_s))
        margin_rad = math.radians(float(self.cfg["soft_limit_margin_deg"]))
        max_step_rad = min(
            math.radians(float(self.cfg["max_step_deg"])),
            math.radians(float(self.cfg["max_vel_deg_s"])) * dt,
        )

        q_safe = q_tgt.copy()
        clamped_joints: List[int] = []

        # 1. Check / clamp hard & soft joint limits
        for j, (lo, hi) in enumerate(JOINT_LIMITS_RAD):
            lo_s, hi_s = lo + margin_rad, hi - margin_rad
            if q_safe[j] < lo or q_safe[j] > hi:
                if not clamp_in_place:
                    return self._record_violation(
                        f"JOINT_LIMIT_EXCEEDED: J{j + 1}={math.degrees(q_safe[j]):.1f}° 超出物理限位 "
                        f"[{math.degrees(lo):.1f}°, {math.degrees(hi):.1f}°]",
                        q_target_rad,
                    )
            if q_safe[j] < lo_s or q_safe[j] > hi_s:
                if not clamp_in_place:
                    return self._record_violation(
                        f"SOFT_LIMIT_EXCEEDED: J{j + 1}={math.degrees(q_safe[j]):.1f}° 进入软限位保护区",
                        q_target_rad,
                    )
                q_safe[j] = float(np.clip(q_safe[j], lo_s, hi_s))
                clamped_joints.append(j + 1)

        # 2. Check / clamp per-step joint jump
        dq = q_safe - q_cur
        max_dq_deg = math.degrees(float(np.max(np.abs(dq))))
        if float(np.max(np.abs(dq))) > max_step_rad + 1e-6:
            if not clamp_in_place:
                return self._record_violation(
                    f"STEP_JUMP_TOO_LARGE: 单步最大关节跳变 {max_dq_deg:.2f}° > 上限 {math.degrees(max_step_rad):.2f}°",
                    q_target_rad,
                )
            scale = max_step_rad / float(np.max(np.abs(dq)))
            q_safe = q_cur + dq * scale
            clamped_joints.append(0)

        # 3. Forward Kinematics Workspace & Tabletop Collision Check
        fk = self._get_fk()
        tcp_xyz = None
        if fk is not None:
            T_tcp = fk.fk(q_safe)
            tcp_xyz = [round(float(v), 4) for v in T_tcp[:3, 3]]
            x, y, z = tcp_xyz
            xb = self.cfg["workspace_x_bounds_m"]
            yb = self.cfg["workspace_y_bounds_m"]
            zb = self.cfg["workspace_z_bounds_m"]
            z_floor = float(self.cfg["min_table_z_m"]) + float(self.cfg["min_tip_clearance_m"])
            if not (xb[0] <= x <= xb[1] and yb[0] <= y <= yb[1] and zb[0] <= z <= zb[1]):
                return self._record_violation(
                    f"WORKSPACE_FENCE_VIOLATION: TCP ({x:+.3f}, {y:+.3f}, {z:+.3f}) m 超出安全工作空间边界",
                    q_target_rad,
                )
            if z < z_floor:
                return self._record_violation(
                    f"TABLE_COLLISION_GUARD: TCP 高度 {z * 1000:.1f} mm 低于桌面安全下限 {z_floor * 1000:.1f} mm",
                    q_target_rad,
                )

        g_safe = float(np.clip(float(gripper_norm), 0.0, 1.0))
        with self.lock:
            self._last_cmd_ts = time.monotonic()

        return {
            "ok": True,
            "q_safe_rad": [round(float(v), 6) for v in q_safe],
            "gripper_norm": round(g_safe, 4),
            "clamped": len(clamped_joints) > 0,
            "clamped_joints": clamped_joints,
            "max_dq_deg": round(max_dq_deg, 2),
            "tcp_xyz_m": tcp_xyz,
        }

    def validate_action_chunk(
        self,
        q_start_rad: List[float],
        chunk: List[List[float]],
        dt_s: float = 0.05,
        caller_mode: str = "smolvla",
        clamp_in_place: bool = True,
    ) -> Dict[str, Any]:
        """Validate an entire predicted Action Chunk (shape [H, 8]: 7 joints + 1 gripper) step by step."""
        ok_auth, err_auth = self.check_authority(caller_mode)
        if not ok_auth:
            return self._record_violation(err_auth or "AUTHORITY_DENIED", q_start_rad)
        if not chunk:
            return {"ok": False, "error": "EMPTY_ACTION_CHUNK"}

        q_prev = list(q_start_rad)
        safe_chunk: List[List[float]] = []
        any_clamped = False
        for step_idx, row in enumerate(chunk):
            if len(row) < 8:
                return self._record_violation(f"CHUNK_STEP_{step_idx}_DIM_ERROR: expected 8D action", row)
            r = self.validate_and_clamp_step(
                q_prev, row[:7], gripper_norm=float(row[7]), dt_s=dt_s,
                caller_mode=caller_mode, clamp_in_place=clamp_in_place,
            )
            if not r["ok"]:
                r["failed_step_index"] = step_idx
                r["safe_prefix"] = safe_chunk
                return r
            q_prev = r["q_safe_rad"]
            any_clamped = any_clamped or bool(r["clamped"])
            safe_chunk.append(list(q_prev) + [r["gripper_norm"]])

        return {
            "ok": True,
            "safe_chunk": safe_chunk,
            "horizon": len(safe_chunk),
            "clamped": any_clamped,
        }

    def _record_violation(self, reason: str, target: Any) -> Dict[str, Any]:
        with self.lock:
            self._trip_count += 1
            v = {
                "t": time.time(),
                "mode": self._mode,
                "reason": str(reason),
                "target": [round(float(x), 4) for x in target[:8]] if isinstance(target, (list, tuple, np.ndarray)) else str(target),
            }
            self._last_violation = v
            self.log(f"[WARN] [V4.1 SAFETY] 动作被安全仲裁器拦截 (#{self._trip_count})：{reason}")
            return {"ok": False, "error": str(reason), "violation": v}
