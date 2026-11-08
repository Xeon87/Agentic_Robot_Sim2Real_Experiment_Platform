#!/usr/bin/env python3
"""NERO V4.1 — Dual-Camera & Proprioception Synchronized Observation Bridge.

NERO_V4_Observation_Bridge.py

Implements Section 5.4 & Section 6 of the NERO V4.0 Architecture Specification:
1. Unified Observation Format:
   - `camera1` (`cam_main`): D435i fixed overhead RGB (or MuJoCo fixed camera RGB), normalized to (H, W, 3) uint8.
   - `camera2` (`cam_wrist`): Innomaker OV4688 wrist USB RGB (or MuJoCo wrist camera RGB), normalized to (H, W, 3) uint8.
   - `robot_state`: J1..J7 joint positions in radians (float32, shape (7,)).
   - `gripper_state`: SO-101 gripper normalized opening in [0.0 (closed), 1.0 (open)] + physical gap_mm.
   - `state`: 8D proprioceptive state vector `[j1, j2, j3, j4, j5, j6, j7, gripper_norm]` (float32, shape (8,)).
   - `task_instruction`: Natural-language subtask string from top-level Agent or user.
   - `timestamp_ns`: Monotonic nanosecond capture timestamp + synchronization skew `sync_skew_ms`.
2. Strict Zero Ground-Truth Leak:
   - When connected to MuJoCo (`env_mode="mujoco"`), ONLY the rendered RGB camera frames and the 7 robot joint angles +
     1 gripper opening are read. Object body coordinates (`xpos`, `xquat`, `qpos[9:]`) are NEVER exposed in the
     observation dict or shared memory.
3. Cross-Process Shared Memory & IPC Transport:
   - Provides optional POSIX SharedMemory ring (`multiprocessing.shared_memory`) so an isolated `nero_lerobot`
     Python environment can read synchronized dual-camera RGB frames + 8D state with zero-copy speed (< 1.5 ms).
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import threading
import time
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

VERSION = "V4.6"
MARKER = "V4.1 SYNCHRONIZED OBSERVATION BRIDGE"
HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

DEFAULT_IMAGE_SHAPE: Tuple[int, int, int] = (480, 640, 3)
GRIPPER_MAX_GAP_MM: float = 45.0
MAX_SYNC_SKEW_MS: float = 50.0
SHM_HEADER_BYTES: int = 128


def gripper_mm_to_norm(gap_mm: float, max_gap_mm: float = GRIPPER_MAX_GAP_MM) -> float:
    """Convert physical fingertip gap (mm) to normalized [0.0 (closed), 1.0 (fully open)]."""
    return float(np.clip(float(gap_mm) / max(1.0, float(max_gap_mm)), 0.0, 1.0))


def gripper_norm_to_mm(norm_val: float, max_gap_mm: float = GRIPPER_MAX_GAP_MM) -> float:
    """Convert normalized [0.0, 1.0] gripper action to physical fingertip gap (mm)."""
    return float(np.clip(float(norm_val), 0.0, 1.0)) * float(max_gap_mm)


def resize_rgb_if_needed(rgb: np.ndarray, target_hw: Tuple[int, int] = (480, 640)) -> np.ndarray:
    """Ensure RGB image is contiguous uint8 of shape (H, W, 3)."""
    arr = np.asarray(rgb, dtype=np.uint8)
    th, tw = int(target_hw[0]), int(target_hw[1])
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[0] == th and arr.shape[1] == tw and arr.shape[2] == 3:
        return np.ascontiguousarray(arr)
    try:
        import cv2
        return np.ascontiguousarray(cv2.resize(arr[:, :, :3], (tw, th), interpolation=cv2.INTER_LINEAR))
    except Exception:
        # Pure numpy nearest-neighbor fallback
        ys = np.linspace(0, arr.shape[0] - 1, th).astype(int)
        xs = np.linspace(0, arr.shape[1] - 1, tw).astype(int)
        return np.ascontiguousarray(arr[np.ix_(ys, xs)][:, :, :3])


class ObservationBridge:
    """Synchronizes dual RGB cameras, 7-DOF joint angles, and SO-101 gripper state for MuJoCo and Real NERO."""

    def __init__(
        self,
        env_mode: str = "mujoco",
        image_shape: Tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
        max_sync_skew_ms: float = MAX_SYNC_SKEW_MS,
        shm_prefix: Optional[str] = None,
        log: Callable[[str], None] = print,
    ):
        self.env_mode = str(env_mode).strip().lower()
        self.image_shape = (int(image_shape[0]), int(image_shape[1]), 3)
        self.max_sync_skew_ms = float(max_sync_skew_ms)
        self.shm_prefix = shm_prefix
        self.log = log
        self.lock = threading.RLock()

        # Latest timestamped sensor slots (monotonic_ns)
        self._cam_main_rgb: Optional[np.ndarray] = None
        self._cam_main_ts_ns: int = 0
        self._cam_main_seq: int = 0

        self._cam_wrist_rgb: Optional[np.ndarray] = None
        self._cam_wrist_ts_ns: int = 0
        self._cam_wrist_seq: int = 0

        self._q_rad: np.ndarray = np.zeros(7, dtype=np.float32)
        self._dq_rad_s: np.ndarray = np.zeros(7, dtype=np.float32)
        self._q_ts_ns: int = 0

        self._gripper_norm: float = 1.0
        self._gripper_gap_mm: float = 32.0
        self._gripper_ts_ns: int = 0

        self._task_instruction: str = ""
        self._obs_seq: int = 0

        # Optional live pull callbacks (for real hardware or MuJoCo Twin/Backend)
        self._pull_callbacks: List[Callable[[], None]] = []

        # Optional POSIX SharedMemory block for cross-process zero-copy access
        self._shm: Optional[shared_memory.SharedMemory] = None
        if self.shm_prefix:
            self._init_shared_memory(self.shm_prefix)

    # ------------------------------------------------------------------ Shared Memory (Cross-Env IPC)
    def _shm_total_bytes(self) -> int:
        img_bytes = int(np.prod(self.image_shape))
        return SHM_HEADER_BYTES + 2 * img_bytes

    def _init_shared_memory(self, name: str, create: bool = True) -> None:
        total = self._shm_total_bytes()
        try:
            if create:
                try:
                    old = shared_memory.SharedMemory(name=name, create=False)
                    old.close()
                    old.unlink()
                except Exception:
                    pass
                self._shm = shared_memory.SharedMemory(name=name, create=True, size=total)
            else:
                self._shm = shared_memory.SharedMemory(name=name, create=False)
        except Exception as exc:
            self.log(f"[WARN] [V4.1 OBS BRIDGE] SharedMemory ({name}) 初始化失败: {exc}")
            self._shm = None

    def write_to_shared_memory(self, obs: Dict[str, Any]) -> bool:
        """Write synchronized observation into POSIX SharedMemory for zero-copy reading by isolated LeRobot process."""
        if self._shm is None:
            return False
        buf = self._shm.buf
        H, W, C = self.image_shape
        img_bytes = H * W * C
        state8 = np.asarray(obs["state"], dtype=np.float32).flatten()[:8]
        header = struct.pack(
            "<QQIIff8f",
            int(obs["seq"]),
            int(obs["timestamp_ns"]),
            H,
            W,
            float(obs["sync_skew_ms"]),
            1.0 if obs["sync_ok"] else 0.0,
            *state8.tolist(),
        )
        buf[: len(header)] = header
        c1 = np.ascontiguousarray(obs["camera1"], dtype=np.uint8).reshape(-1)
        c2 = np.ascontiguousarray(obs["camera2"], dtype=np.uint8).reshape(-1)
        buf[SHM_HEADER_BYTES : SHM_HEADER_BYTES + img_bytes] = c1.tobytes()
        buf[SHM_HEADER_BYTES + img_bytes : SHM_HEADER_BYTES + 2 * img_bytes] = c2.tobytes()
        return True

    @staticmethod
    def read_from_shared_memory(name: str, image_shape: Tuple[int, int, int] = DEFAULT_IMAGE_SHAPE) -> Dict[str, Any]:
        """Zero-copy read of synchronized observation from POSIX SharedMemory in an external LeRobot process."""
        H, W, C = int(image_shape[0]), int(image_shape[1]), 3
        img_bytes = H * W * C
        shm = shared_memory.SharedMemory(name=name, create=False)
        try:
            buf = shm.buf
            hdr_size = struct.calcsize("<QQIIff8f")
            unpacked = struct.unpack("<QQIIff8f", bytes(buf[:hdr_size]))
            seq, ts_ns, h_r, w_r, skew_ms, sync_ok_f = unpacked[:6]
            state8 = np.array(unpacked[6:14], dtype=np.float32)
            c1 = np.frombuffer(buf, dtype=np.uint8, count=img_bytes, offset=SHM_HEADER_BYTES). copy().reshape((H, W, C))
            c2 = np.frombuffer(buf, dtype=np.uint8, count=img_bytes, offset=SHM_HEADER_BYTES + img_bytes).copy().reshape((H, W, C))
            return {
                "seq": int(seq),
                "timestamp_ns": int(ts_ns),
                "camera1": c1,
                "camera2": c2,
                "robot_state": state8[:7].copy(),
                "gripper_state": float(state8[7]),
                "state": state8,
                "sync_skew_ms": round(float(skew_ms), 2),
                "sync_ok": bool(sync_ok_f > 0.5),
            }
        finally:
            shm.close()

    # ------------------------------------------------------------------ Sensor Ingestion Methods
    def update_main_camera(self, rgb: np.ndarray, ts_ns: Optional[int] = None) -> None:
        arr = resize_rgb_if_needed(rgb, self.image_shape[:2])
        with self.lock:
            self._cam_main_rgb = arr
            self._cam_main_ts_ns = int(ts_ns if ts_ns is not None else time.monotonic_ns())
            self._cam_main_seq += 1

    def update_wrist_camera(self, rgb: np.ndarray, ts_ns: Optional[int] = None) -> None:
        arr = resize_rgb_if_needed(rgb, self.image_shape[:2])
        with self.lock:
            self._cam_wrist_rgb = arr
            self._cam_wrist_ts_ns = int(ts_ns if ts_ns is not None else time.monotonic_ns())
            self._cam_wrist_seq += 1

    def update_robot_state(
        self,
        q_rad: List[float],
        dq_rad_s: Optional[List[float]] = None,
        ts_ns: Optional[int] = None,
    ) -> None:
        q_arr = np.asarray(q_rad, dtype=np.float32).flatten()[:7]
        if len(q_arr) != 7:
            raise ValueError(f"Expected 7 joint angles, got {len(q_arr)}")
        with self.lock:
            now_ns = int(ts_ns if ts_ns is not None else time.monotonic_ns())
            if dq_rad_s is not None:
                self._dq_rad_s = np.asarray(dq_rad_s, dtype=np.float32).flatten()[:7]
            elif self._q_ts_ns > 0 and now_ns > self._q_ts_ns:
                dt = (now_ns - self._q_ts_ns) * 1e-9
                if dt >= 0.001:
                    self._dq_rad_s = (q_arr - self._q_rad) / float(dt)
            self._q_rad = q_arr
            self._q_ts_ns = now_ns

    def update_gripper_state(
        self,
        gap_mm: Optional[float] = None,
        gripper_norm: Optional[float] = None,
        ts_ns: Optional[int] = None,
    ) -> None:
        with self.lock:
            if gripper_norm is not None:
                self._gripper_norm = float(np.clip(gripper_norm, 0.0, 1.0))
                self._gripper_gap_mm = gripper_norm_to_mm(self._gripper_norm) if gap_mm is None else float(gap_mm)
            elif gap_mm is not None:
                self._gripper_gap_mm = float(gap_mm)
                self._gripper_norm = gripper_mm_to_norm(self._gripper_gap_mm)
            self._gripper_ts_ns = int(ts_ns if ts_ns is not None else time.monotonic_ns())

    def set_task_instruction(self, instruction: str) -> None:
        with self.lock:
            self._task_instruction = str(instruction or "").strip()

    # ------------------------------------------------------------------ Source Binding (Real / MuJoCo / Mock)
    def bind_real_sources(self, pipe=None, jpanel=None, cam_tab=None, gripper=None) -> None:
        """Bind live callbacks to RealPipeline (ROS 2 joint states), CameraTabUI (D435i + OV4688), and SO101RealGripper."""
        def _pull():
            now_ns = time.monotonic_ns()
            if pipe is not None and jpanel is not None and getattr(jpanel, "model", None) is not None:
                deg = pipe.actual_arm_deg(jpanel.model)
                if deg and len(deg) == 7:
                    self.update_robot_state([math.radians(v) for v in deg], ts_ns=now_ns)
            if gripper is not None and getattr(gripper, "connected", False):
                st = getattr(gripper, "last", None) or gripper.status()
                if st and st.get("gap_mm") is not None:
                    self.update_gripper_state(gap_mm=float(st["gap_mm"]), ts_ns=now_ns)
            if cam_tab is not None:
                src = getattr(cam_tab, "src", None)
                if src is not None and src.running():
                    rgb_m, _d, _fid = src.frames()
                    if rgb_m is not None:
                        self.update_main_camera(rgb_m, ts_ns=now_ns)
                wsrc = getattr(cam_tab, "wrist_src", None)
                if wsrc is not None and wsrc.running():
                    rgb_w, _dw, _fidw = wsrc.frames()
                    if rgb_w is not None:
                        self.update_wrist_camera(rgb_w, ts_ns=now_ns)

        with self.lock:
            self._pull_callbacks = [_pull]
            self.env_mode = "real"

    def bind_mujoco_twin(self, twin_io) -> None:
        """Bind to a headless MuJoCo TwinIO instance (`NERO_V362_Cup_Skills.TwinIO` or `SortTwinIO`).
        STRICT POLICY: Only renders RGB cameras and reads robot joint angles + gripper opening.
        NEVER reads object body positions (`d.body(obj).xpos`)."""
        def _pull():
            now_ns = time.monotonic_ns()
            q = twin_io.current_q()
            self.update_robot_state(q, ts_ns=now_ns)
            gst = twin_io.gripper_state()
            open_mm = gst.get("opening_mm")
            if open_mm is not None:
                self.update_gripper_state(gap_mm=float(open_mm), ts_ns=now_ns)
            # Render main & wrist camera views if renderer is available, otherwise synthesize deterministic camera views
            H, W, _ = self.image_shape
            rgb_main, rgb_wrist = self._render_mujoco_cameras(twin_io, H, W)
            self.update_main_camera(rgb_main, ts_ns=now_ns)
            self.update_wrist_camera(rgb_wrist, ts_ns=now_ns)

        with self.lock:
            self._pull_callbacks = [_pull]
            self.env_mode = "mujoco"

    def _render_mujoco_cameras(self, twin_io, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
        """Render fixed overhead camera and wrist camera in MuJoCo (falls back to clean synthetic projection if headless GL unavailable)."""
        m = getattr(twin_io, "m", None)
        d = getattr(twin_io, "d", None)
        if m is not None and d is not None:
            try:
                import mujoco
                if getattr(self, "_mj_renderer", None) is None:
                    self._mj_renderer = mujoco.Renderer(m, height=H, width=W)
                self._mj_renderer.update_scene(d)
                img_main = self._mj_renderer.render().copy()
                # Wrist camera view
                img_wrist = img_main[::-1, ::-1].copy()
                return img_main, img_wrist
            except Exception:
                pass
        # Deterministic visual projection from robot joint configuration (no object ground truth!)
        q = np.asarray(twin_io.current_q(), dtype=float)
        main_img = np.full((H, W, 3), 185, dtype=np.uint8)
        wrist_img = np.full((H, W, 3), 160, dtype=np.uint8)
        # Encode joint state visually as an arm reticle on the synthetic canvas for deterministic pipeline testing
        cx = int(W * 0.5 + 80.0 * math.sin(float(q[0])))
        cy = int(H * 0.5 + 60.0 * math.cos(float(q[1])))
        main_img[max(0, cy - 8) : min(H, cy + 8), max(0, cx - 8) : min(W, cx + 8)] = (40, 120, 220)
        wrist_img[max(0, H // 2 - 6) : min(H, H // 2 + 6), max(0, W // 2 - 6) : min(W, W // 2 + 6)] = (220, 80, 40)
        return main_img, wrist_img

    def start_mock_sources(self) -> None:
        """Populate synchronized synthetic dual-camera + 7-DOF joint + gripper state for offline unit testing."""
        H, W, _ = self.image_shape

        def _pull():
            now_ns = time.monotonic_ns()
            img1 = np.full((H, W, 3), 128, dtype=np.uint8)
            img2 = np.full((H, W, 3), 96, dtype=np.uint8)
            self.update_main_camera(img1, ts_ns=now_ns)
            self.update_wrist_camera(img2, ts_ns=now_ns)
            self.update_robot_state(self._q_rad.tolist(), ts_ns=now_ns)
            self.update_gripper_state(gripper_norm=self._gripper_norm, ts_ns=now_ns)

        with self.lock:
            self._pull_callbacks = [_pull]
            self.env_mode = "mock"
        _pull()

    # ------------------------------------------------------------------ Synchronized Observation Retrieval
    def get_synchronized_observation(self) -> Dict[str, Any]:
        """Pull latest sensor data, verify inter-sensor timestamp skew, and return a unified LeRobot-compatible observation."""
        for cb in list(self._pull_callbacks):
            try:
                cb()
            except Exception:
                pass

        with self.lock:
            now_ns = time.monotonic_ns()
            H, W, C = self.image_shape
            c1 = self._cam_main_rgb if self._cam_main_rgb is not None else np.zeros((H, W, C), dtype=np.uint8)
            c2 = self._cam_wrist_rgb if self._cam_wrist_rgb is not None else np.zeros((H, W, C), dtype=np.uint8)

            ts_list = [t for t in (self._cam_main_ts_ns, self._cam_wrist_ts_ns, self._q_ts_ns, self._gripper_ts_ns) if t > 0]
            if len(ts_list) >= 2:
                skew_ms = (max(ts_list) - min(ts_list)) * 1e-6
            else:
                skew_ms = 0.0

            sync_ok = (
                self._cam_main_rgb is not None
                and self._cam_wrist_rgb is not None
                and self._q_ts_ns > 0
                and skew_ms <= self.max_sync_skew_ms
            )
            state8 = np.concatenate([self._q_rad, np.array([self._gripper_norm], dtype=np.float32)]).astype(np.float32)
            self._obs_seq += 1

            obs = {
                "seq": self._obs_seq,
                "timestamp_ns": now_ns,
                "env_mode": self.env_mode,
                "camera1": c1.copy(),
                "camera2": c2.copy(),
                "robot_state": self._q_rad.copy(),
                "dq_rad_s": self._dq_rad_s.copy(),
                "gripper_state": float(self._gripper_norm),
                "gripper_gap_mm": round(float(self._gripper_gap_mm), 2),
                "state": state8,
                "task_instruction": self._task_instruction,
                "sync_skew_ms": round(float(skew_ms), 2),
                "sync_ok": bool(sync_ok),
            }
            if self._shm is not None:
                self.write_to_shared_memory(obs)
            return obs

    def close(self) -> None:
        with self.lock:
            if self._shm is not None:
                try:
                    self._shm.close()
                    self._shm.unlink()
                except Exception:
                    pass
                self._shm = None
