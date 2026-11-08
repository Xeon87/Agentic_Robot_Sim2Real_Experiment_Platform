#!/usr/bin/env python3
"""NERO V4.1 — Hugging Face LeRobot Robot Adapter & Cross-Environment IPC Bridge.

NERO_V4_LeRobot_Adapter.py

Implements Section 5.3 & Section 6.3 of the NERO V4.0 Architecture Specification:
1. Standard Hugging Face LeRobot `Robot` Interface (`NEROLeRobotAdapter`):
   - `connect()`
   - `disconnect()`
   - `get_observation()` -> returns `{"observation.state": (8,), "observation.images.cam_main": (H,W,3), "observation.images.cam_wrist": (H,W,3), "task": str}`
   - `send_action(action)` -> submits 8D action or `[H, 8]` Action Chunk to `VLAActionExecutor` (never bypassing `ControlArbiter` or ROS 2 safety).
   - `observation_features` -> LeRobot-standard feature schema dict.
   - `action_features` -> LeRobot-standard action schema dict.
2. Dual Transport Support (Direct In-Process + Isolated Process IPC):
   - Direct Mode: Directly wraps `ObservationBridge` + `VLAActionExecutor` in the same Python process.
   - IPC Server / Client Mode (`NEROIPCServer` + `NEROLeRobotAdapter(ipc_socket_path=...)`):
     Allows LeRobot / SmolVLA running in a completely isolated conda/venv Python environment to exchange
     observations (via POSIX SharedMemory + Unix Domain Socket) and actions with the NERO ROS 2 / MuJoCo host.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

VERSION = "V4.5"
MARKER = "V4.1 LEROBOT ROBOT ADAPTER & IPC BRIDGE"
HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from NERO_V4_Control_Arbiter import ControlArbiter
from NERO_V4_Observation_Bridge import DEFAULT_IMAGE_SHAPE, ObservationBridge
from NERO_V4_VLA_Action_Executor import VLAActionExecutor

JOINT_NAMES_8D = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper")
DEFAULT_IPC_SOCKET = os.environ.get("NERO_V4_IPC_SOCKET", "/tmp/nero_v4_lerobot_ipc.sock")
DEFAULT_SHM_NAME = os.environ.get("NERO_V4_SHM_NAME", "nero_v4_obs_shm")


def lerobot_observation_features(image_shape: Tuple[int, int, int] = DEFAULT_IMAGE_SHAPE) -> Dict[str, Dict[str, Any]]:
    """Standard LeRobot observation_features schema for NERO 7-DOF + SO-101 Gripper + Dual RGB Cameras."""
    H, W, C = int(image_shape[0]), int(image_shape[1]), int(image_shape[2])
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (8,),
            "names": list(JOINT_NAMES_8D),
            "units": ["rad"] * 7 + ["normalized_0_1"],
        },
        "observation.images.cam_main": {
            "dtype": "video",
            "shape": (H, W, C),
            "names": ["height", "width", "channel"],
            "camera_role": "fixed_d435i_rgb",
        },
        "observation.images.cam_wrist": {
            "dtype": "video",
            "shape": (H, W, C),
            "names": ["height", "width", "channel"],
            "camera_role": "wrist_ov4688_rgb",
        },
    }


def lerobot_action_features() -> Dict[str, Dict[str, Any]]:
    """Standard LeRobot action_features schema for NERO 7-DOF + SO-101 Gripper."""
    return {
        "action": {
            "dtype": "float32",
            "shape": (8,),
            "names": list(JOINT_NAMES_8D),
            "units": ["rad"] * 7 + ["normalized_0_1"],
        }
    }


def _to_tensor_or_ndarray(arr: np.ndarray, use_torch: bool = False):
    if use_torch:
        try:
            import torch
            return torch.from_numpy(np.ascontiguousarray(arr))
        except Exception:
            pass
    return np.ascontiguousarray(arr)


def _from_tensor_or_ndarray(x: Any) -> np.ndarray:
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    if isinstance(x, dict) and "action" in x:
        return _from_tensor_or_ndarray(x["action"])
    return np.asarray(x, dtype=np.float32)


class NEROIPCServer:
    """Host-side Unix Domain Socket + SharedMemory IPC Server for isolated LeRobot Python environments."""

    def __init__(
        self,
        obs_bridge: ObservationBridge,
        executor: VLAActionExecutor,
        socket_path: str = DEFAULT_IPC_SOCKET,
        shm_name: str = DEFAULT_SHM_NAME,
        log: Callable[[str], None] = print,
    ):
        self.obs_bridge = obs_bridge
        self.executor = executor
        self.socket_path = socket_path
        self.shm_name = shm_name
        self.log = log
        self._stop_ev = threading.Event()
        self._srv_sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self.obs_bridge.shm_prefix != self.shm_name or self.obs_bridge._shm is None:
            self.obs_bridge.shm_prefix = self.shm_name
            self.obs_bridge._init_shared_memory(self.shm_name, create=True)
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass
        self._stop_ev.clear()
        self._srv_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv_sock.bind(self.socket_path)
        self._srv_sock.listen(4)
        self._srv_sock.settimeout(0.2)
        self._thread = threading.Thread(target=self._serve_loop, daemon=True, name="nero_v4_ipc_srv")
        self._thread.start()
        self.log(f"[OK] [V4.1 IPC] LeRobot 跨环境 IPC 服务已启动：{self.socket_path} (SHM={self.shm_name})")

    def _serve_loop(self) -> None:
        while not self._stop_ev.is_set():
            try:
                conn, _ = self._srv_sock.accept()  # type: ignore
            except socket.timeout:
                continue
            except Exception:
                break
            threading.Thread(target=self._handle_client, args=(conn,), daemon=True).start()

    def _handle_client(self, conn: socket.socket) -> None:
        with conn:
            f_in = conn.makefile("r", encoding="utf-8")
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                    resp = self._dispatch_req(req)
                except Exception as exc:
                    resp = {"ok": False, "error": str(exc)}
                conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))

    def _dispatch_req(self, req: Dict[str, Any]) -> Dict[str, Any]:
        cmd = req.get("cmd")
        if cmd == "ping":
            return {"ok": True, "version": VERSION, "shm_name": self.shm_name, "image_shape": list(self.obs_bridge.image_shape)}
        if cmd == "get_observation":
            obs = self.obs_bridge.get_synchronized_observation()
            return {
                "ok": True,
                "seq": obs["seq"],
                "timestamp_ns": obs["timestamp_ns"],
                "state": [float(v) for v in obs["state"]],
                "task_instruction": obs["task_instruction"],
                "sync_skew_ms": obs["sync_skew_ms"],
                "sync_ok": obs["sync_ok"],
                "shm_name": self.shm_name,
            }
        if cmd == "send_action":
            act = req.get("action")
            caller = req.get("caller_mode", "smolvla")
            arr = np.asarray(act, dtype=float)
            if arr.ndim == 2:
                return self.executor.execute_action_chunk(arr.tolist(), caller_mode=caller)
            return self.executor.send_action(arr.flatten().tolist(), caller_mode=caller)
        if cmd == "status":
            return {"ok": True, "executor": self.executor.status()}
        return {"ok": False, "error": f"Unknown IPC cmd: {cmd}"}

    def stop(self) -> None:
        self._stop_ev.set()
        if self._srv_sock is not None:
            try:
                self._srv_sock.close()
            except Exception:
                pass
            self._srv_sock = None
        if os.path.exists(self.socket_path):
            try:
                os.unlink(self.socket_path)
            except OSError:
                pass


class NEROLeRobotAdapter:
    """LeRobot-compatible `Robot` adapter for NERO 7-DOF + SO-101 Gripper (supports both MuJoCo and Real Robot).

    IMPORTANT INVARIANT (Section 11.3):
    Calling `connect()` only establishes the observation/communication link. It NEVER automatically grants
    real-robot or simulation motion authority (`arbiter.allow_control` remains unchanged until explicitly enabled).
    """
    robot_type = "nero_so101_7dof"

    def __init__(
        self,
        obs_bridge: Optional[ObservationBridge] = None,
        executor: Optional[VLAActionExecutor] = None,
        arbiter: Optional[ControlArbiter] = None,
        ipc_socket_path: Optional[str] = None,
        image_shape: Tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
        use_torch_tensors: bool = False,
        log: Callable[[str], None] = print,
    ):
        self.image_shape = (int(image_shape[0]), int(image_shape[1]), 3)
        self.use_torch_tensors = bool(use_torch_tensors)
        self.ipc_socket_path = ipc_socket_path
        self.log = log

        if ipc_socket_path is None:
            self.arbiter = arbiter or ControlArbiter(log=log)
            self.obs_bridge = obs_bridge or ObservationBridge(image_shape=self.image_shape, log=log)
            self.executor = executor or VLAActionExecutor(self.arbiter, self.obs_bridge, log=log)
        else:
            self.arbiter = arbiter  # type: ignore
            self.obs_bridge = obs_bridge  # type: ignore
            self.executor = executor  # type: ignore

        self.is_connected: bool = False
        self._ipc_sock: Optional[socket.socket] = None
        self._ipc_file = None
        self._shm_name: Optional[str] = None

    @property
    def observation_features(self) -> Dict[str, Dict[str, Any]]:
        return lerobot_observation_features(self.image_shape)

    @property
    def action_features(self) -> Dict[str, Dict[str, Any]]:
        return lerobot_action_features()

    def connect(self) -> Dict[str, Any]:
        """Connect to the NERO environment (in-process or via IPC socket).
        Does NOT automatically grant motion control authority (`allow_control` stays False unless explicitly enabled)."""
        if self.is_connected:
            return {"ok": True, "already_connected": True}
        if self.ipc_socket_path:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.connect(self.ipc_socket_path)
            self._ipc_sock = s
            self._ipc_file = s.makefile("r", encoding="utf-8")
            ping = self._ipc_call({"cmd": "ping"})
            if not ping.get("ok"):
                self.disconnect()
                raise RuntimeError(f"IPC handshake failed: {ping}")
            self._shm_name = ping.get("shm_name")
        self.is_connected = True
        self.log(f"[OK] [V4.1 LEROBOT ADAPTER] 已连接 NERO LeRobot 接口 (mode={'IPC' if self.ipc_socket_path else self.obs_bridge.env_mode})")
        return {"ok": True, "mode": "ipc" if self.ipc_socket_path else self.obs_bridge.env_mode}

    def disconnect(self) -> None:
        if self._ipc_file is not None:
            try:
                self._ipc_file.close()
            except Exception:
                pass
            self._ipc_file = None
        if self._ipc_sock is not None:
            try:
                self._ipc_sock.close()
            except Exception:
                pass
            self._ipc_sock = None
        self.is_connected = False
        self.log("[INFO] [V4.1 LEROBOT ADAPTER] 已断开连接")

    def _ipc_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._ipc_sock is None or self._ipc_file is None:
            raise RuntimeError("NEROLeRobotAdapter IPC not connected")
        self._ipc_sock.sendall((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))
        line = self._ipc_file.readline()
        if not line:
            raise RuntimeError("IPC server closed connection")
        return json.loads(line)

    def get_observation(self) -> Dict[str, Any]:
        """Return synchronized dual-camera RGB + 8D proprioceptive state in LeRobot format."""
        if not self.is_connected:
            raise RuntimeError("NEROLeRobotAdapter is not connected; call connect() first.")

        if self.ipc_socket_path:
            meta = self._ipc_call({"cmd": "get_observation"})
            if not meta.get("ok"):
                raise RuntimeError(f"IPC get_observation failed: {meta.get('error')}")
            shm_obs = ObservationBridge.read_from_shared_memory(meta["shm_name"], self.image_shape)
            c1 = shm_obs["camera1"]
            c2 = shm_obs["camera2"]
            state8 = shm_obs["state"]
            task_str = str(meta.get("task_instruction") or "")
            skew_ms = float(meta.get("sync_skew_ms", shm_obs["sync_skew_ms"]))
            sync_ok = bool(meta.get("sync_ok", shm_obs["sync_ok"]))
            ts_ns = int(meta.get("timestamp_ns", shm_obs["timestamp_ns"]))
        else:
            raw = self.obs_bridge.get_synchronized_observation()
            c1 = raw["camera1"]
            c2 = raw["camera2"]
            state8 = raw["state"]
            task_str = raw["task_instruction"]
            skew_ms = raw["sync_skew_ms"]
            sync_ok = raw["sync_ok"]
            ts_ns = raw["timestamp_ns"]

        return {
            "observation.state": _to_tensor_or_ndarray(state8, self.use_torch_tensors),
            "observation.images.cam_main": _to_tensor_or_ndarray(c1, self.use_torch_tensors),
            "observation.images.cam_wrist": _to_tensor_or_ndarray(c2, self.use_torch_tensors),
            "task": task_str,
            "timestamp_ns": ts_ns,
            "sync_skew_ms": skew_ms,
            "sync_ok": sync_ok,
        }

    def send_action(self, action: Union[np.ndarray, List[float], Dict[str, Any]]) -> Dict[str, Any]:
        """Submit an 8D single action or `[H, 8]` Action Chunk to the safety-gated `VLAActionExecutor`."""
        if not self.is_connected:
            raise RuntimeError("NEROLeRobotAdapter is not connected; call connect() first.")

        arr = _from_tensor_or_ndarray(action)
        if self.ipc_socket_path:
            return self._ipc_call({
                "cmd": "send_action",
                "action": arr.tolist(),
                "caller_mode": "smolvla",
            })

        if arr.ndim == 2:
            return self.executor.execute_action_chunk(arr.tolist(), caller_mode="smolvla")
        if arr.ndim == 1 and len(arr) == 8:
            return self.executor.send_action(arr.tolist(), caller_mode="smolvla")
        raise ValueError(f"Expected action shape (8,) or (H, 8), got {arr.shape}")
