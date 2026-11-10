#!/usr/bin/env python3
"""NERO V4.2 — LeRobot Demonstration Dataset Recorder, Quality Checker & Episode Replayer.

NERO_V4_Dataset_Recorder.py

Implements Section 7 & Phase 2 of the NERO V4.0 Architecture Specification:
1. Demonstration Recording (`DemonstrationRecorder`):
   - Supports NERO zero-force drag teach mode (`teach_mode`), SpaceMouse / teleop, or scripted demonstrations.
   - Dedicated independent gripper teaching channel (`trigger_gripper_open` / `trigger_gripper_close` / `set_gripper_target_norm`)
     because zero-force arm dragging only moves the 7 joints and does not move the STS3215 jaw.
   - Synchronizes dual cameras (`cam_main`, `cam_wrist`), 7-DOF joint positions, gripper state, and 8D target actions
     at a configurable recording frequency (`fps`, default 20 Hz).
2. LeRobotDataset v2.1 Local Storage Format (`NEROLeRobotDataset`):
   - Saves strictly locally (never auto-uploads to Hugging Face Hub):
     - `meta/info.json` (robot_type, fps, features, total_episodes, total_frames, calibration & recorder version)
     - `meta/tasks.jsonl` (`{"task_index": int, "task": str}`)
     - `meta/episodes.jsonl` (`{"episode_index": int, "tasks": [str], "length": int, "split": "train"|"val"|"test", "valid": bool, "quality": {...}}`)
     - `meta/stats.json` (per-channel mean, std, min, max for `observation.state` and `action` across training episodes)
     - `data/episode_<XXXXXX>.npz` (compressed arrays: `observation_state`, `action`, `timestamp_ns`, `sync_skew_ms`, `cam_main`, `cam_wrist`, plus metadata JSON).
   - Enforces strict train / val / test split separation to prevent evaluation data leakage.
3. Data Quality Inspection (`inspect_episode_quality`):
   - Checks timestamp synchronization skew (`max_skew_ms <= 50 ms`), frame interval regularity / drop rate (`< 5%`),
     joint trajectory continuity, and minimum episode length.
4. Episode Replay & Closed-Loop Verification (`replay_episode`):
   - Reloads a saved episode from disk, replays its recorded 8D actions through `VLAActionExecutor` (under `ControlArbiter`),
     and computes the trajectory tracking RMSE (in degrees and normalized gripper units) between recorded states and replayed states.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

VERSION = "V4.6"
MARKER = "V4.2 LEROBOT DATASET RECORDER & EPISODE REPLAYER"
HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from NERO_V4_LeRobot_Adapter import JOINT_NAMES_8D, lerobot_action_features, lerobot_observation_features
from NERO_V4_Observation_Bridge import DEFAULT_IMAGE_SHAPE, ObservationBridge
from NERO_V4_VLA_Action_Executor import VLAActionExecutor

import NERO_V44_Storage as ST

# V4.4: datasets default to <EMBODIED_AI>/nero_lerobot/datasets/nero (old internal folder when the drive is missing).
# Resolved at construction time; this constant is only the value at import time, kept for compatibility.
DEFAULT_DATASET_ROOT = ST.default_dataset_root()
VALID_SPLITS = ("train", "val", "test")


def inspect_episode_quality(
    frames: List[Dict[str, Any]],
    target_fps: float = 20.0,
    max_allowed_skew_ms: float = 50.0,
    max_drop_frac: float = 0.10,
) -> Dict[str, Any]:
    """Evaluate a recorded episode's synchronization quality, frame timing regularity, and action smoothness."""
    n = len(frames)
    if n < 3:
        return {
            "ok": False,
            "valid": False,
            "n_frames": n,
            "reasons": [f"帧数太少 ({n} < 3)"],
            "max_skew_ms": 0.0,
            "avg_skew_ms": 0.0,
            "drop_frac": 1.0,
            "max_joint_step_deg": 0.0,
        }

    skews = [float(f.get("sync_skew_ms", 0.0)) for f in frames]
    max_skew = round(float(max(skews)), 2)
    avg_skew = round(float(np.mean(skews)), 2)

    ts_ns = [int(f["timestamp_ns"]) for f in frames]
    dts_s = [(ts_ns[i] - ts_ns[i - 1]) * 1e-9 for i in range(1, n)]
    nom_dt = 1.0 / max(1.0, float(target_fps))
    dropped = sum(1 for dt in dts_s if dt > 1.8 * nom_dt)
    drop_frac = round(float(dropped) / max(1, len(dts_s)), 4)

    states = np.asarray([f["state"] for f in frames], dtype=np.float32)
    actions = np.asarray([f["action"] for f in frames], dtype=np.float32)
    dq_deg = np.degrees(np.abs( np.diff(states[:, :7], axis=0) ))
    max_step_deg = round(float(np.max(dq_deg)) if dq_deg.size else 0.0, 2)

    reasons: List[str] = []
    if max_skew > max_allowed_skew_ms:
        reasons.append(f"传感器时间同步超差 ({max_skew:.1f} ms > {max_allowed_skew_ms:.0f} ms)")
    if drop_frac > max_drop_frac:
        reasons.append(f"采样丢帧率过高 ({drop_frac * 100:.1f}% > {max_drop_frac * 100:.0f}%)")
    if not (np.all(np.isfinite(states)) and np.all(np.isfinite(actions))):
        reasons.append("状态或动作包含 NaN/Inf")

    ok = len(reasons) == 0
    return {
        "ok": ok,
        "valid": ok,
        "n_frames": n,
        "duration_s": round(float(sum(dts_s)), 3),
        "measured_fps": round((n - 1) / max(1e-6, float(sum(dts_s))), 2) if sum(dts_s) > 0 else target_fps,
        "max_skew_ms": max_skew,
        "avg_skew_ms": avg_skew,
        "drop_frac": drop_frac,
        "max_joint_step_deg": max_step_deg,
        "reasons": reasons,
    }


class NEROLeRobotDataset:
    """Local LeRobotDataset v2.1 manager with train/val/test split isolation, stats computation, and episode CRUD."""

    def __init__(
        self,
        dataset_name: str = "nero_v4_demo",
        root_dir: Optional[str] = None,
        fps: float = 20.0,
        image_shape: Tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
    ):
        self.dataset_name = str(dataset_name).strip() or "nero_v4_demo"
        self.root_dir = os.path.abspath(os.path.expanduser(root_dir or ST.default_dataset_root()))
        self.dataset_dir = os.path.join(self.root_dir, self.dataset_name)
        self.meta_dir = os.path.join(self.dataset_dir, "meta")
        self.data_dir = os.path.join(self.dataset_dir, "data")
        self.fps = float(fps)
        self.image_shape = (int(image_shape[0]), int(image_shape[1]), 3)
        self.joint_limits_rad = None            # V4.7.4: the arm's limits (rad); None = NERO URDF defaults
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        self._ensure_meta()

    def _info_path(self) -> str:
        return os.path.join(self.meta_dir, "info.json")

    def _episodes_path(self) -> str:
        return os.path.join(self.meta_dir, "episodes.jsonl")

    def _tasks_path(self) -> str:
        return os.path.join(self.meta_dir, "tasks.jsonl")

    def _stats_path(self) -> str:
        return os.path.join(self.meta_dir, "stats.json")

    def _episode_npz_path(self, ep_idx: int) -> str:
        return os.path.join(self.data_dir, f"episode_{int(ep_idx):06d}.npz")

    def _ensure_meta(self) -> None:
        if not os.path.isfile(self._info_path()):
            info = {
                "codebase_version": "v2.1",
                "nero_version": VERSION,
                "robot_type": "nero_so101_7dof",
                "fps": self.fps,
                "total_episodes": 0,
                "total_frames": 0,
                "features": {
                    **lerobot_observation_features(self.image_shape),
                    **lerobot_action_features(),
                },
            }
            with open(self._info_path(), "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
        for p in (self._episodes_path(), self._tasks_path()):
            if not os.path.isfile(p):
                with open(p, "w", encoding="utf-8") as f:
                    f.write("")

    def list_episodes(self, split: Optional[str] = None, valid_only: bool = False) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not os.path.isfile(self._episodes_path()):
            return out
        with open(self._episodes_path(), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if split and row.get("split") != split:
                    continue
                if valid_only and not row.get("valid", True):
                    continue
                out.append(row)
        return out

    def list_tasks(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if not os.path.isfile(self._tasks_path()):
            return out
        with open(self._tasks_path(), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def _register_task(self, task_instruction: str) -> int:
        task_instruction = str(task_instruction or "").strip() or "default_task"
        tasks = self.list_tasks()
        for t in tasks:
            if t.get("task") == task_instruction:
                return int(t["task_index"])
        new_idx = len(tasks)
        with open(self._tasks_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps({"task_index": new_idx, "task": task_instruction}, ensure_ascii=False) + "\n")
        return new_idx

    def save_episode(
        self,
        frames: List[Dict[str, Any]],
        task_instruction: str,
        split: str = "train",
        task_id: str = "",
        hardware_cfg: Optional[Dict[str, Any]] = None,
        calib_version: str = VERSION,
        episode_success: bool = True,
        thumbnail_stride: int = 1,
        images: Optional[Tuple[Any, Any]] = None,
    ) -> Dict[str, Any]:
        """Validate and save one recorded episode to local LeRobotDataset storage.
        V4.6.11: `images` = (cam_main, cam_wrist) arrays of shape (n, H, W, 3), e.g. read-only memmaps of the disk
        spool - written to the npz in chunks, never stacked in RAM (full resolution, lossless)."""
        split = str(split).strip().lower()
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

        qual = inspect_episode_quality(frames, target_fps=self.fps)
        # Convert frame list to next-step target action if raw action was not explicitly provided
        n = len(frames)
        states = np.asarray([f["state"] for f in frames], dtype=np.float32)
        actions = np.asarray([f["action"] for f in frames], dtype=np.float32)
        ts_ns = np.asarray([int(f["timestamp_ns"]) for f in frames], dtype=np.int64)
        skews = np.asarray([float(f.get("sync_skew_ms", 0.0)) for f in frames], dtype=np.float32)

        # Store compact RGB frames (subsampled or full, resized to (120, 160, 3) if thumbnail_stride > 1, or full)
        if images is not None:
            cam_main_arr, cam_wrist_arr = images
        else:
            c1_list = [np.asarray(f["camera1"], dtype=np.uint8) for f in frames[::max(1, int(thumbnail_stride))]]
            c2_list = [np.asarray(f["camera2"], dtype=np.uint8) for f in frames[::max(1, int(thumbnail_stride))]]
            cam_main_arr = np.stack(c1_list, axis=0) if c1_list else np.zeros((0, *self.image_shape), dtype=np.uint8)
            cam_wrist_arr = np.stack(c2_list, axis=0) if c2_list else np.zeros((0, *self.image_shape), dtype=np.uint8)

        existing = self.list_episodes()
        ep_idx = (max(e["episode_index"] for e in existing) + 1) if existing else 0
        task_idx = self._register_task(task_instruction)

        import NERO_V474_Episode_Tools as ET                  # V4.7.4: frames outside the commandable range
        lim_rep = ET.episode_limit_report(states, self.fps, self.joint_limits_rad)
        ep_meta = {
            "limit_ok": bool(lim_rep["ok"]),
            "limit_report": lim_rep,
            "episode_index": ep_idx,
            "task_id": task_id or f"task_{task_idx}",
            "task_index": task_idx,
            "tasks": [task_instruction],
            "length": n,
            "split": split,
            "episode_success": bool(episode_success),
            "valid": bool(qual["valid"] and episode_success),
            "fps": self.fps,
            "calib_version": calib_version,
            "recorder_version": VERSION,
            "hardware_cfg": hardware_cfg or {"arm": "NERO_7DOF", "gripper": "SO-101"},
            "quality": qual,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        npz_path = self._episode_npz_path(ep_idx)
        np.savez_compressed(
            npz_path,
            observation_state=states,
            action=actions,
            timestamp_ns=ts_ns,
            sync_skew_ms=skews,
            cam_main=cam_main_arr,
            cam_wrist=cam_wrist_arr,
            meta_json=np.array(json.dumps(ep_meta, ensure_ascii=False)),
        )

        ep_meta["file"] = npz_path                    # V4.6.11: where it is (on the removable drive when mounted)
        with open(self._episodes_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(ep_meta, ensure_ascii=False) + "\n")

        self._rebuild_info_and_stats()
        return ep_meta

    def next_episode_index(self) -> int:
        """V4.6.11: the index the next saved episode gets"""
        eps = self.list_episodes()
        return (max(int(e["episode_index"]) for e in eps) + 1) if eps else 0

    def load_episode(self, ep_idx: int) -> Dict[str, Any]:
        """Load a saved episode's states, actions, camera arrays, and metadata."""
        p = self._episode_npz_path(ep_idx)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Episode {ep_idx} not found at {p}")
        with np.load(p, allow_pickle=False) as data:
            meta = json.loads(str(data["meta_json"]))
            return {
                "meta": meta,
                "observation_state": data["observation_state"].copy(),
                "action": data["action"].copy(),
                "timestamp_ns": data["timestamp_ns"].copy(),
                "sync_skew_ms": data["sync_skew_ms"].copy(),
                "cam_main": data["cam_main"].copy(),
                "cam_wrist": data["cam_wrist"].copy(),
            }

    def delete_episode(self, ep_idx: int) -> bool:
        """Delete an episode by index and recompute dataset metadata & stats."""
        p = self._episode_npz_path(ep_idx)
        removed = False
        if os.path.isfile(p):
            os.unlink(p)
            removed = True
        eps = [e for e in self.list_episodes() if int(e["episode_index"]) != int(ep_idx)]
        with open(self._episodes_path(), "w", encoding="utf-8") as f:
            for e in eps:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        self._rebuild_info_and_stats()
        return removed

    def _rebuild_info_and_stats(self) -> Dict[str, Any]:
        """Recompute info.json and stats.json (using ONLY valid training split episodes to prevent test leakage)."""
        eps = self.list_episodes()
        total_frames = sum(int(e.get("length", 0)) for e in eps)
        with open(self._info_path(), "r", encoding="utf-8") as f:
            info = json.load(f)
        info["total_episodes"] = len(eps)
        info["total_frames"] = total_frames
        info["splits"] = {
            sp: len([e for e in eps if e.get("split") == sp]) for sp in VALID_SPLITS
        }
        with open(self._info_path(), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

        train_states, train_actions = [], []
        for e in eps:
            if e.get("split") == "train" and e.get("valid", True):
                try:
                    d = self.load_episode(int(e["episode_index"]))
                    train_states.append(d["observation_state"])
                    train_actions.append(d["action"])
                except Exception:
                    pass
        if not train_states:
            # Fallback to any valid episode if no train split yet
            for e in eps:
                if e.get("valid", True):
                    try:
                        d = self.load_episode(int(e["episode_index"]))
                        train_states.append(d["observation_state"])
                        train_actions.append(d["action"])
                    except Exception:
                        pass

        stats: Dict[str, Any] = {}
        if train_states:
            S = np.concatenate(train_states, axis=0)
            A = np.concatenate(train_actions, axis=0)
            stats["observation.state"] = {
                "mean": [round(float(x), 6) for x in np.mean(S, axis=0)],
                "std": [round(float(max(x, 1e-4)), 6) for x in np.std(S, axis=0)],
                "min": [round(float(x), 6) for x in np.min(S, axis=0)],
                "max": [round(float(x), 6) for x in np.max(S, axis=0)],
                "count": int(len(S)),
            }
            stats["action"] = {
                "mean": [round(float(x), 6) for x in np.mean(A, axis=0)],
                "std": [round(float(max(x, 1e-4)), 6) for x in np.std(A, axis=0)],
                "min": [round(float(x), 6) for x in np.min(A, axis=0)],
                "max": [round(float(x), 6) for x in np.max(A, axis=0)],
                "count": int(len(A)),
            }
        with open(self._stats_path(), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        return stats

    def get_stats(self) -> Dict[str, Any]:
        if os.path.isfile(self._stats_path()):
            try:
                with open(self._stats_path(), "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return self._rebuild_info_and_stats()


MARKER_V4611_REC = "V4.6.11 DISK SPOOL + BACKGROUND SAVE + DISCARD"
SPOOL_ROOT = os.path.expanduser(os.environ.get("NERO_REC_SPOOL_DIR", "~/.cache/nero_record_spool"))
SPOOL_QUEUE_FRAMES = 120                # ~ 6 s at 20 Hz of full-size images waiting for the disk


class ImageSpool:
    """V4.6.11: the camera images of a recording go to two raw files on the local disk (full 640x480, lossless) by a
    writer thread, instead of a Python list in RAM (V4.6.10: ~37 MB/s, 2.2 GB per minute -> the console got slower and
    slower).  The record loop only puts the arrays into a bounded queue."""

    def __init__(self, shape, root=None):
        import queue as _q
        root = root or SPOOL_ROOT
        self.shape = (int(shape[0]), int(shape[1]), 3)
        self.dir = os.path.join(root, time.strftime("%Y%m%d_%H%M%S") + f"_{os.getpid()}_{id(self) % 10000}")
        os.makedirs(self.dir, exist_ok=True)
        self.paths = (os.path.join(self.dir, "cam_main.raw"), os.path.join(self.dir, "cam_wrist.raw"))
        self._f = [open(p, "wb") for p in self.paths]
        self._q = _q.Queue(maxsize=SPOOL_QUEUE_FRAMES)
        self.n_written = 0
        self.error = None
        self._t = threading.Thread(target=self._writer, daemon=True, name="nero_v4611_spool")
        self._t.start()

    def _fit(self, img):
        a = np.asarray(img, dtype=np.uint8)
        if a.shape != self.shape:
            from NERO_V4_Observation_Bridge import resize_rgb_if_needed
            a = resize_rgb_if_needed(a, self.shape[:2])
        return np.ascontiguousarray(a)

    def put(self, c1, c2):
        self._q.put((self._fit(c1), self._fit(c2)))

    def _writer(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            try:
                self._f[0].write(item[0].tobytes())
                self._f[1].write(item[1].tobytes())
                self.n_written += 1
            except Exception as exc:          # disk full etc.: keep draining, report at the end
                self.error = str(exc)

    def close(self):
        if self._t.is_alive():
            self._q.put(None)
            self._t.join()
        for f in self._f:
            try:
                f.close()
            except Exception:
                pass

    def arrays(self, n):
        n = min(int(n), self.n_written)
        if n <= 0:
            return (np.zeros((0, *self.shape), np.uint8), np.zeros((0, *self.shape), np.uint8))
        return tuple(np.memmap(p, dtype=np.uint8, mode="r", shape=(n, *self.shape)) for p in self.paths)

    def delete(self):
        self.close()
        shutil.rmtree(self.dir, ignore_errors=True)


class DemonstrationRecorder:
    """Real-time demonstration recorder for NERO (supports zero-force drag + independent gripper control + replay)."""

    def __init__(
        self,
        obs_bridge: ObservationBridge,
        dataset: NEROLeRobotDataset,
        executor: Optional[VLAActionExecutor] = None,
        fps: float = 20.0,
        log: Callable[[str], None] = print,
    ):
        self.obs_bridge = obs_bridge
        self.dataset = dataset
        self.executor = executor
        self.fps = float(fps)
        self.log = log

        self.lock = threading.RLock()
        self._recording = False
        self._stop_ev = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frames: List[Dict[str, Any]] = []
        self._current_task: str = ""
        self._current_split: str = "train"
        # Independent gripper target command during zero-force drag teaching
        self._teach_gripper_cmd_norm: Optional[float] = None
        self.use_spool = True                       # V4.6.11: images to a disk spool, not RAM
        self._spool: Optional[ImageSpool] = None

    @property
    def is_recording(self) -> bool:
        with self.lock:
            return self._recording

    def set_teach_gripper_norm(self, norm_val: float, drive_gripper_fn: Optional[Callable[[float], None]] = None) -> float:
        """Independently command and record the SO-101 gripper opening during zero-force drag teaching."""
        v = float(np.clip(float(norm_val), 0.0, 1.0))
        with self.lock:
            self._teach_gripper_cmd_norm = v
        if drive_gripper_fn is not None:
            try:
                drive_gripper_fn(v)
            except Exception:
                pass
        else:
            self.obs_bridge.update_gripper_state(gripper_norm=v)
        self.log(f"[INFO] [V4.2 RECORDER] 示教夹爪独立指令: norm={v:.2f}")
        return v

    def start_recording(self, task_instruction: str, split: str = "train") -> Dict[str, Any]:
        with self.lock:
            if self._recording:
                return {"ok": False, "error": "已经在录制中"}
            self._frames.clear()
            self._spool = ImageSpool(self.dataset.image_shape) if self.use_spool else None    # V4.6.11
            self._current_task = str(task_instruction or "").strip() or "Demonstrate manipulation task"
            self._current_split = split if split in VALID_SPLITS else "train"
            self.obs_bridge.set_task_instruction(self._current_task)
            self._stop_ev.clear()
            self._recording = True
            self._thread = threading.Thread(target=self._record_loop, daemon=True, name="nero_v4_recorder")
            self._thread.start()
        self.log(f"[OK] [V4.2 RECORDER] 开始示教录制 ({self.fps:.0f} Hz, split={self._current_split}): {self._current_task!r}")
        return {"ok": True, "task": self._current_task, "split": self._current_split}

    def _record_loop(self) -> None:
        dt = 1.0 / max(1.0, self.fps)
        while not self._stop_ev.is_set():
            t0 = time.perf_counter()
            obs = self.obs_bridge.get_synchronized_observation()
            state8 = obs["state"].copy()
            with self.lock:
                g_cmd = self._teach_gripper_cmd_norm if self._teach_gripper_cmd_norm is not None else float(state8[7])
                frame = {
                    "timestamp_ns": int(obs["timestamp_ns"]),
                    "sync_skew_ms": float(obs["sync_skew_ms"]),
                    "state": state8,
                    "action": np.concatenate([state8[:7], np.array([g_cmd], dtype=np.float32)]).astype(np.float32),
                }
                sp = self._spool
                if sp is None:
                    frame["camera1"], frame["camera2"] = obs["camera1"], obs["camera2"]
            if sp is not None:
                sp.put(obs["camera1"], obs["camera2"])         # may wait for the disk (bounded queue), lock released
            with self.lock:
                self._frames.append(frame)
            elapsed = time.perf_counter() - t0
            sleep_s = max(0.0, dt - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)

    def record_step_From_state_action(
        self,
        state_8d: List[float],
        action_8d: List[float],
        cam_main: Optional[np.ndarray] = None,
        cam_wrist: Optional[np.ndarray] = None,
        timestamp_ns: Optional[int] = None,
        sync_skew_ms: float = 0.0,
    ) -> None:
        """Directly append a single synchronized step (useful for deterministic simulation demonstration generation)."""
        H, W, C = self.dataset.image_shape
        c1 = cam_main if cam_main is not None else np.full((H, W, C), 128, dtype=np.uint8)
        c2 = cam_wrist if cam_wrist is not None else np.full((H, W, C), 96, dtype=np.uint8)
        with self.lock:
            self._frames.append({
                "timestamp_ns": int(timestamp_ns if timestamp_ns is not None else time.monotonic_ns()),
                "sync_skew_ms": float(sync_skew_ms),
                "state": np.asarray(state_8d, dtype=np.float32).flatten()[:8],
                "action": np.asarray(action_8d, dtype=np.float32).flatten()[:8],
                "camera1": c1,
                "camera2": c2,
            })

    def stop_capture(self) -> Dict[str, Any]:
        """V4.6.11: stop the record loop at once (fast, Tk thread OK) -> the captured episode for save_captured /
        discard_captured (slow parts, a worker thread)"""
        self._stop_ev.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        with self.lock:
            self._recording = False
            cap = {"frames": list(self._frames), "spool": self._spool,
                   "task": self._current_task or "Demonstrate manipulation task", "split": self._current_split}
            self._frames = []
            self._spool = None
        return cap

    def discard_captured(self, cap: Dict[str, Any]) -> Dict[str, Any]:
        """V4.6.11: user decision - an abandoned recording is deleted, nothing is saved"""
        sp = cap.get("spool")
        n = len(cap.get("frames") or [])
        where = sp.dir if sp is not None else None
        if sp is not None:
            sp.delete()
        self.log(f"[INFO] [V4.6.11 RECORDER] 已放弃本次录制（{n} 帧，{n / max(1.0, self.fps):.1f} s）：没有保存"
                 + (f"，临时文件已删除（{where}）" if where else ""))
        return {"ok": True, "discarded": True, "frames": n}

    def stop_and_save(self, episode_success: bool = True, lead_action_by_one_step: bool = True) -> Dict[str, Any]:
        """Stop recording, align action[t] = state[t+1] (for kinesthetic teach), run quality checks, and save."""
        return self.save_captured(self.stop_capture(), episode_success, lead_action_by_one_step)

    def save_captured(self, cap: Dict[str, Any], episode_success: bool = True,
                      lead_action_by_one_step: bool = True) -> Dict[str, Any]:
        frames, task, split, sp = cap["frames"], cap["task"], cap["split"], cap.get("spool")
        images = None
        if sp is not None:
            t0 = time.perf_counter()
            sp.close()                                  # waits for the writer to finish the queue
            if sp.error:
                self.log(f"[WARN] [V4.6.11 RECORDER] 临时图像文件写入出错：{sp.error}")
            if sp.n_written < len(frames):
                frames = frames[:sp.n_written]
            images = sp.arrays(len(frames))
            self.log(f"[INFO] [V4.6.11 RECORDER] 正在保存 {len(frames)} 帧（640×480 原图，无损压缩）… "
                     f"写完临时文件用了 {time.perf_counter() - t0:.1f} s")

        if len(frames) >= 2 and lead_action_by_one_step:
            # In zero-force drag teaching, the target action at frame t is the achieved joint position at frame t+1
            for i in range(len(frames) - 1):
                next_q = frames[i + 1]["state"][:7]
                g_cmd = frames[i]["action"][7]
                frames[i]["action"] = np.concatenate([next_q, np.array([g_cmd], dtype=np.float32)]).astype(np.float32)

        try:
            meta = self.dataset.save_episode(
                frames,
                task_instruction=task,
                split=split,
                episode_success=episode_success,
                images=images,
            )
        finally:
            if sp is not None:
                images = None
                sp.delete()
        self.log(
            f"[{'OK' if meta['valid'] else 'WARN'}] [V4.2 RECORDER] 已保存 Episode #{meta['episode_index']} "
            f"({meta['length']} 帧, {meta['length'] / max(1.0, self.fps):.1f} s, split={split}, valid={meta['valid']}, "
            f"max_skew={meta['quality']['max_skew_ms']:.1f} ms) → {meta.get('file')}"
        )
        return {"ok": True, "episode": meta}

    def replay_episode(
        self,
        ep_idx: int,
        executor: Optional[VLAActionExecutor] = None,
        caller_mode: str = "teleop",
    ) -> Dict[str, Any]:
        """Reload a saved episode and replay its trajectory through `VLAActionExecutor`, measuring tracking RMSE."""
        ex = executor or self.executor
        if ex is None:
            return {"ok": False, "error": "No VLAActionExecutor provided for replay"}
        ep = self.dataset.load_episode(ep_idx)
        rec_states = ep["observation_state"]
        rec_actions = ep["action"]
        n = len(rec_actions)
        if n == 0:
            return {"ok": False, "error": "Empty episode"}

        # Initialize start state in ObservationBridge
        ex.obs_bridge.update_robot_state(rec_states[0, :7].tolist())
        ex.obs_bridge.update_gripper_state(gripper_norm=float(rec_states[0, 7]))

        replayed_states: List[np.ndarray] = []
        dt = 1.0 / max(1.0, float(ep["meta"].get("fps", self.fps)))
        for i in range(n):
            r = ex.send_action(rec_actions[i].tolist(), caller_mode=caller_mode, dt_s=dt)
            if not r.get("ok"):
                self.log(f"[FAIL] [V4.6.9 REPLAY] 回放 Episode #{ep_idx} 在第 {i}/{n} 帧中止：{r.get('error')}")
                return {
                    "ok": False,
                    "error": f"Replay aborted at frame {i}/{n}: {r.get('error')}",
                    "frames_replayed": i,
                }
            obs_now = ex.obs_bridge.get_synchronized_observation()
            replayed_states.append(obs_now["state"].copy())

        rep_arr = np.stack(replayed_states, axis=0)
        # Compare replayed states against recorded target actions (or next states)
        joint_err_deg = np.degrees(rep_arr[:, :7] - rec_actions[:, :7])
        rmse_deg = round(float(np.sqrt(np.mean(joint_err_deg ** 2))), 4)
        max_err_deg = round(float(np.max(np.abs(joint_err_deg))), 4)
        grip_rmse = round(float(np.sqrt(np.mean((rep_arr[:, 7] - rec_actions[:, 7]) ** 2))), 4)

        ok = rmse_deg < 2.0 and grip_rmse < 0.05
        self.log(
            f"[{'OK' if ok else 'WARN'}] [V4.2 REPLAY] 回放 Episode #{ep_idx} 完成 ({n} 帧) · "
            f"关节跟踪 RMSE={rmse_deg:.3f}° (最大误差 {max_err_deg:.3f}°) · 夹爪 RMSE={grip_rmse:.4f}"
        )
        return {
            "ok": ok,
            "episode_index": ep_idx,
            "frames_replayed": n,
            "joint_rmse_deg": rmse_deg,
            "joint_max_err_deg": max_err_deg,
            "gripper_rmse": grip_rmse,
        }


# ===================================================================== V4.6.9 REAL REPLAY (the real NERO really moves)
MARKER_V469_REPLAY = "V4.6.9 REAL REPLAY"
REPLAY_MIN_STEP_DEG = 0.5          # keep a recorded point when any joint moved at least this much since the last kept
REPLAY_MAX_GAP_S = 0.5             # ... or this much time passed (smooth timing of slow parts)
REPLAY_GRIP_THRESHOLD = 0.5        # gripper norm >= this = open, below = closed
REPLAY_OPEN_GAP_MM = 45.0
REPLAY_CLAMP_PCT = 25.0
REPLAY_START_SPEED = 0.10          # move to the recorded start pose at 10 %


def plan_real_replay(states, fps, speed=1.0, min_step_deg=REPLAY_MIN_STEP_DEG, max_gap_s=REPLAY_MAX_GAP_S):
    """V4.6.9: recorded 8-D states (7 joints rad + gripper norm) -> joint path segments for RealPipeline.move_joint_path.
    The trajectory is thinned (min_step_deg / max_gap_s), timed from the recording / speed, and cut where the gripper
    switches open <-> closed: [{"path": [q..], "t": [s..] (from 0 at the segment start), "grip_after": "open" /
    "close" / None}].  Each path starts at the end of the previous one (the first at the recorded start pose)."""
    st = np.asarray(states, dtype=float)
    if st.ndim != 2 or st.shape[0] < 2 or st.shape[1] < 8:
        return []
    dt = 1.0 / max(1.0, float(fps)) / max(0.1, float(speed))
    grip = np.where(st[:, 7] >= REPLAY_GRIP_THRESHOLD, 1, 0)
    segs, cur, t_cur, t_last = [], [0], [0.0], 0.0
    last_q = st[0, :7]
    for i in range(1, st.shape[0]):
        t_i = i * dt
        moved = float(np.max(np.abs(np.degrees(st[i, :7] - last_q))))
        switch = grip[i] != grip[i - 1]
        if moved >= min_step_deg or (t_i - t_last) >= max_gap_s or switch or i == st.shape[0] - 1:
            if moved > 1e-6 or switch or i == st.shape[0] - 1:
                cur.append(i)
                t_cur.append(t_i)
                last_q, t_last = st[i, :7], t_i
        if switch:
            segs.append((cur, t_cur, "open" if grip[i] == 1 else "close"))
            cur, t_cur = [i], [t_i]
    segs.append((cur, t_cur, None))
    out = []
    for idx, ts, g in segs:
        if len(idx) < 2:
            if g is not None:
                out.append({"path": [st[idx[0], :7].tolist()], "t": [0.0], "grip_after": g})
            continue
        t0 = ts[0]
        tt, last = [], -1.0
        for t in ts:
            v = round(t - t0, 4)
            if v <= last:
                v = round(last + 0.01, 4)
            tt.append(v)
            last = v
        out.append({"path": [st[k, :7].tolist() for k in idx], "t": tt, "grip_after": g})
    return out


def replay_episode_real(dataset, ep_idx, pipe, model, gripper=None, speed=0.5, log=print, stop_fn=None):
    """V4.6.9: replay a recorded episode on the REAL NERO (+ SO-101):  checks (not in teach mode, joint feedback) ->
    10 % move to the recorded start -> each segment as ONE timed joint path (no per-frame MoveIt calls) -> gripper
    open / close between segments -> measured joints after every segment vs the recording.  Every failure is logged.
    -> {"ok", "segments", "max_end_err_deg", "error"}"""
    import math as _m
    stop_fn = stop_fn or (lambda: False)
    if pipe is None or not model:
        log("[FAIL] [V4.6.9 REPLAY] 实机没有就绪（实机NERO调试 步骤 5/6）")
        return {"ok": False, "error": "real robot not ready"}
    if getattr(pipe, "teach_mode", False):
        log("[FAIL] [V4.6.9 REPLAY] NERO 还在示教模式：先「退出示教模式」再回放")
        return {"ok": False, "error": "teach mode"}
    ep = dataset.load_episode(int(ep_idx))
    states = ep["observation_state"]
    fps = float(ep["meta"].get("fps", 20.0))
    import NERO_V474_Episode_Tools as ET                      # V4.7.4 (C1): check BEFORE anything moves
    rep = ET.episode_limit_report(states, fps, model.get("limits") if isinstance(model, dict) else None, margin=0.0)
    if not rep["ok"]:
        log(f"[FAIL] [V4.7.4 REPLAY] Episode #{ep_idx} 有超出关节限位的帧，机械臂走不到，没有回放（机械臂没动）："
            f"{ET.report_text(rep)}。这是示教时手把关节带过了限位；这个 Episode 建议删除重录")
        return {"ok": False, "error": "outside joint limits", "limit_report": rep}
    segs = plan_real_replay(states, fps, speed=speed)
    if not segs:
        log(f"[FAIL] [V4.6.9 REPLAY] Episode #{ep_idx} 太短，没有可回放的轨迹")
        return {"ok": False, "error": "empty"}
    move = np.degrees(np.ptp(np.asarray(states)[:, :7], axis=0))
    if float(np.max(move)) < 1.0:
        log(f"[WARN] [V4.6.9 REPLAY] Episode #{ep_idx} 里机械臂几乎没动（最大关节变化 {float(np.max(move)):.2f}°）："
            "很可能是在「仿真替身」后端录的，回放时真机也不会动")
    dur = sum(sg["t"][-1] for sg in segs)
    log(f"[INFO] [V4.6.9 REPLAY] Episode #{ep_idx}：{len(states)} 帧 → {len(segs)} 段 · "
        f"{sum(len(sg['path']) for sg in segs)} 个路径点 · 速度 {speed:g}× · 约 {dur:.1f} s")
    q0 = list(segs[0]["path"][0])
    r = pipe.move_and_settle(model, q0, REPLAY_START_SPEED, label="V4.6.9 replay: go to the recorded start")
    if not r.get("ok"):
        log(f"[FAIL] [V4.6.9 REPLAY] 到录制起点失败：{r.get('error')}")
        return {"ok": False, "error": f"start: {r.get('error')}"}
    g0 = "open" if float(states[0][7]) >= REPLAY_GRIP_THRESHOLD else "close"
    _grip(gripper, g0, log)
    errs = []
    for k, sg in enumerate(segs):
        if stop_fn():
            log("[WARN] [V4.6.9 REPLAY] 已停止")
            return {"ok": False, "error": "stopped", "segments": k}
        if len(sg["path"]) >= 2:
            act = pipe.actual_arm_deg(model)
            path = [list(sg["path"][0]) if act is None else [_m.radians(v) for v in act]] + [list(p) for p in sg["path"][1:]]
            r = pipe.move_joint_path(model, path, sg["t"], label=f"V4.6.9 replay #{ep_idx} seg {k + 1}/{len(segs)}")
            if not r.get("ok"):
                log(f"[FAIL] [V4.6.9 REPLAY] 第 {k + 1}/{len(segs)} 段失败：{r.get('error')}")
                return {"ok": False, "error": r.get("error"), "segments": k}
            act = pipe.actual_arm_deg(model)
            if act is not None:
                e = max(abs(a - _m.degrees(b)) for a, b in zip(act, sg["path"][-1]))
                errs.append(e)
                log(f"[INFO] [V4.6.9 REPLAY] 第 {k + 1}/{len(segs)} 段完成：实测关节与录制位置最大差 {e:.2f}°")
        if sg["grip_after"]:
            _grip(gripper, sg["grip_after"], log)
    mx = round(max(errs), 2) if errs else None
    ok = mx is None or mx < 3.0
    log(f"[{'OK' if ok else 'WARN'}] [V4.6.9 REPLAY] 回放 Episode #{ep_idx} 完成：{len(segs)} 段 · 每段终点实测误差最大 "
        f"{'-' if mx is None else f'{mx:.2f}°'}")
    return {"ok": ok, "segments": len(segs), "max_end_err_deg": mx}


def _grip(gripper, what, log):
    if gripper is None or not getattr(gripper, "connected", False):
        log(f"[INFO] [V4.6.9 REPLAY] 夹爪{('张开' if what == 'open' else '夹紧')}（夹爪未连接，跳过）")
        return
    try:
        r = gripper.open_gap(REPLAY_OPEN_GAP_MM) if what == "open" else gripper.clamp(REPLAY_CLAMP_PCT)
        log(f"[INFO] [V4.6.9 REPLAY] 夹爪{('张开' if what == 'open' else '夹紧')}：{r}")
    except Exception as exc:
        log(f"[WARN] [V4.6.9 REPLAY] 夹爪动作失败：{exc}")

