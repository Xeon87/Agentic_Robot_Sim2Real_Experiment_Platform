#!/usr/bin/env python3
"""NERO V4.4 — SmolVLA worker that runs INSIDE the LeRobot venv on the EMBODIED_AI drive.

NERO_V44_SmolVLA_Worker.py   (V4.4 SMOLVLA DRIVE WORKER)

The NERO console runs on the system Python (tkinter, ROS 2, YOLOE ...). LeRobot + torch live in their own venv on the
removable drive, so the console never imports them: it starts this file with the venv's Python and talks to it.

Sub-commands (always started with `<drive>/nero_lerobot/envs/lerobot/bin/python -I NERO_V44_SmolVLA_Worker.py ...`):
    serve      stdin/stdout request loop used by NERO_V4_SmolVLA_Policy (backend "hf_smolvla_ext")
    download   --repo lerobot/smolvla_base --local-dir <drive>/.../models/smolvla_base  (+ its VLM backbone into HF_HOME)
    export     NERO recorder dataset (npz) -> real LeRobotDataset (parquet + video/png) for `lerobot-train`
    env        print torch / CUDA / lerobot versions as JSON

serve protocol (binary-safe, one request at a time):
    request : one UTF-8 JSON line {"cmd": ..., "blobs": [{"name", "dtype", "shape", "nbytes"}, ...]} + raw blob bytes
    response: one UTF-8 JSON line {"ok": bool, ...}
    cmds    : ping | load {path, device} | predict {state, task} + blobs cam_main / cam_wrist (uint8 HxWx3) | reset | quit
Everything LeRobot / torch prints goes to stderr (the worker log on the drive); stdout carries only the protocol.

Only numpy + the standard library are needed at import time; `--fake` (or NERO_V44_WORKER_FAKE=1) answers without
lerobot / torch so the protocol can be tested offline.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

VERSION = "V4.6"
MARKER = "V4.4 SMOLVLA DRIVE WORKER"
JOINT_NAMES_8D = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7", "gripper"]


def _err(msg: str) -> None:
    sys.stderr.write(msg.rstrip("\n") + "\n")
    sys.stderr.flush()


def _import_smolvla():
    for mod_path in ("lerobot.policies.smolvla.modeling_smolvla", "lerobot.common.policies.smolvla.modeling_smolvla"):
        try:
            mod = __import__(mod_path, fromlist=["SmolVLAPolicy"])
            cls = getattr(mod, "SmolVLAPolicy", None)
            if cls is not None:
                return cls, mod_path
        except Exception as exc:  # noqa: BLE001
            _err(f"[WARN] import {mod_path}: {exc!r}")
    raise ImportError("SmolVLAPolicy not importable (pip install 'lerobot[smolvla]')")


def env_info() -> Dict[str, Any]:
    o: Dict[str, Any] = {"python": sys.executable, "python_version": sys.version.split()[0],
                         "hf_home": os.environ.get("HF_HOME"), "worker_version": VERSION}
    try:
        import torch
        o["torch"] = torch.__version__
        o["cuda"] = bool(torch.cuda.is_available())
        if o["cuda"]:
            o["gpu"] = torch.cuda.get_device_name(0)
            o["vram_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 2 ** 20)
    except Exception as exc:  # noqa: BLE001
        o["torch_error"] = repr(exc)
    try:
        from importlib.metadata import version
        o["lerobot"] = version("lerobot")
    except Exception:  # noqa: BLE001
        o["lerobot"] = None
    return o


# ================================================================================================ serve
class SmolVLARuntime:
    def __init__(self, fake: bool = False):
        self.fake = fake
        self.policy = None
        self.pre = None
        self.post = None
        self.device = "cpu"
        self.cfg_summary: Dict[str, Any] = {}
        self.path = None

    def load(self, path: str, device: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if self.fake:
            self.path = path
            dims = [int(x) for x in os.environ.get("NERO_V44_FAKE_DIMS", "6,6").split(",")[:2]]
            self.cfg_summary = {"state_dim": dims[0], "action_dim": dims[1], "chunk_size": 50, "n_action_steps": 50,
                                "image_keys": ["observation.images.camera1", "observation.images.camera2"],
                                "dataset_repo_id": os.environ.get("NERO_V44_FAKE_REPO") or None}
            return {"ok": True, "device": "fake", "config": self.cfg_summary, "load_ms": 0.0}
        import torch
        cls, mod_path = _import_smolvla()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        pol = cls.from_pretrained(path)
        pol = pol.to(self.device)
        pol.eval()
        if hasattr(pol, "reset"):
            pol.reset()
        self.policy, self.path = pol, path
        # LeRobot >= 0.4 moved normalisation / tokenisation into processor pipelines saved beside the weights.
        self.pre = self.post = None
        try:
            from lerobot.policies.factory import make_pre_post_processors  # type: ignore
            try:
                self.pre, self.post = make_pre_post_processors(
                    pol.config, pretrained_path=path,
                    preprocessor_overrides={"device_processor": {"device": str(self.device)}})
            except TypeError:
                self.pre, self.post = make_pre_post_processors(pol.config, pretrained_path=path)
            _err("[OK] processors loaded (LeRobot >= 0.4 pipeline)")
        except Exception as exc:  # noqa: BLE001
            self.pre = self.post = None
            _err(f"[INFO] no processor pipeline ({exc!r}); assuming the policy normalises internally (LeRobot <= 0.3)")
        cfg = pol.config
        st_dim, act_dim = None, None
        try:
            st_dim = int(cfg.input_features["observation.state"].shape[0])
        except Exception:  # noqa: BLE001
            pass
        try:
            act_dim = int(cfg.output_features["action"].shape[0])
        except Exception:  # noqa: BLE001
            pass
        # A model fine-tuned from smolvla_base keeps the base config's state shape [6] while its normaliser was
        # fitted on the NERO dataset (8-D): the normaliser statistics decide what the processor pipeline accepts.
        st_norm = self._stat_dim("observation.state")
        if st_norm:
            st_dim = st_norm
        act_norm = self._stat_dim("action")          # V4.6: the action size the normaliser was fitted on
        if act_norm:
            act_dim = act_norm
        # V4.6: what was this checkpoint trained on? (lerobot-train writes train_config.json beside the weights)
        repo_id, robot_type, steps = None, None, None
        try:
            with open(os.path.join(path, "train_config.json"), "r", encoding="utf-8") as f:
                tc = json.load(f)
            repo_id = (tc.get("dataset") or {}).get("repo_id")
            steps = tc.get("steps")
        except Exception:  # noqa: BLE001
            pass
        try:
            with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
                robot_type = json.load(f).get("robot_type")
        except Exception:  # noqa: BLE001
            pass
        img_keys = list(getattr(cfg, "image_features", {}) or {})
        self.cfg_summary = {"state_dim": st_dim, "action_dim": act_dim, "chunk_size": getattr(cfg, "chunk_size", None),
                            "n_action_steps": getattr(cfg, "n_action_steps", None), "image_keys": img_keys,
                            "module": mod_path, "processors": self.pre is not None,
                            "dataset_repo_id": repo_id, "robot_type": robot_type, "train_steps": steps}
        return {"ok": True, "device": str(self.device), "config": self.cfg_summary,
                "load_ms": round((time.perf_counter() - t0) * 1000.0, 1)}

    def _stat_dim(self, key: str) -> Optional[int]:
        for step in list(getattr(self.pre, "steps", None) or []) + list(getattr(self.post, "steps", None) or []):
            st = getattr(step, "stats", None)
            if isinstance(st, dict) and isinstance(st.get(key), dict):
                for v in st[key].values():
                    try:
                        a = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
                        if a.ndim >= 1 and a.shape[-1] > 0:
                            return int(a.shape[-1])
                    except Exception:  # noqa: BLE001
                        continue
        return None

    def reset(self) -> Dict[str, Any]:
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()
        return {"ok": True}

    def predict(self, state: List[float], task: str, cam_main: np.ndarray, cam_wrist: np.ndarray) -> Dict[str, Any]:
        t0 = time.perf_counter()
        s = np.asarray(state, dtype=np.float32).flatten()
        if self.fake:
            H, D = int(self.cfg_summary["chunk_size"]), int(self.cfg_summary["action_dim"])
            base = np.resize(s, D) if s.size else np.zeros(D, np.float32)
            arr = np.stack([base + 0.001 * (h + 1) for h in range(H)]).astype(np.float32)
            arr[:, -1] = float(np.clip(cam_main.mean() / 255.0, 0, 1))  # proves the image bytes arrived
            return {"ok": True, "action": arr.tolist(), "shape": list(arr.shape), "infer_ms": 0.0}
        if self.policy is None:
            return {"ok": False, "error": "policy not loaded"}
        import torch
        st_dim = self.cfg_summary.get("state_dim") or len(s)
        s_in = s[:st_dim] if st_dim <= len(s) else np.pad(s, (0, st_dim - len(s)))
        dev = torch.device(self.device)
        state_t = torch.from_numpy(np.ascontiguousarray(s_in)).unsqueeze(0).to(dev)

        def _img(a: np.ndarray):
            return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(dev)

        c1, c2 = _img(cam_main), _img(cam_wrist)
        batch: Dict[str, Any] = {"observation.state": state_t, "task": [task]}
        keys = self.cfg_summary.get("image_keys") or ["observation.images.cam_main", "observation.images.cam_wrist"]
        for k in keys:
            kl = k.lower()
            batch[k] = c2 if any(t in kl for t in ("wrist", "hand", "camera2", "side")) else c1
        if self.pre is not None:
            batch = self.pre(batch)
        with torch.inference_mode():
            if hasattr(self.policy, "predict_action_chunk"):
                out = self.policy.predict_action_chunk(batch)
            else:
                out = self.policy.select_action(batch)
        if self.post is not None:
            out = self.post(out)
        if isinstance(out, dict):
            out = out.get("action", out.get("actions"))
        arr = out.detach().float().cpu().numpy() if hasattr(out, "detach") else np.asarray(out, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        elif arr.ndim == 1:
            arr = arr[None, :]
        return {"ok": True, "action": arr.astype(np.float32).tolist(), "shape": list(arr.shape),
                "infer_ms": round((time.perf_counter() - t0) * 1000.0, 2)}


def _read_exact(f, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            raise EOFError("stdin closed")
        buf.extend(chunk)
    return bytes(buf)


def serve(fake: bool = False) -> int:
    # Protocol on a private copy of fd 1; everything else that writes to stdout (lerobot, torch, tqdm) goes to stderr.
    proto_out = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    fin = sys.stdin.buffer
    rt = SmolVLARuntime(fake=fake)

    def reply(obj: Dict[str, Any]) -> None:
        proto_out.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))
        proto_out.flush()

    reply({"ok": True, "hello": True, "version": VERSION, "fake": fake, "env": env_info() if not fake else {"fake": True}})
    while True:
        line = fin.readline()
        if not line:
            return 0
        try:
            req = json.loads(line.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            reply({"ok": False, "error": f"bad request: {exc}"})
            continue
        arrays: Dict[str, np.ndarray] = {}
        try:
            for b in req.get("blobs") or []:
                raw = _read_exact(fin, int(b["nbytes"]))
                arrays[b["name"]] = np.frombuffer(raw, dtype=np.dtype(b["dtype"])).reshape(b["shape"])
        except EOFError:
            return 0
        cmd = req.get("cmd")
        try:
            if cmd == "ping":
                reply({"ok": True, "version": VERSION, "loaded": rt.policy is not None or (fake and rt.path is not None)})
            elif cmd == "env":
                reply({"ok": True, **env_info()})
            elif cmd == "load":
                reply(rt.load(str(req["path"]), req.get("device")))
            elif cmd == "reset":
                reply(rt.reset())
            elif cmd == "predict":
                blank = np.zeros((120, 160, 3), dtype=np.uint8)
                reply(rt.predict(req.get("state") or [], str(req.get("task") or ""),
                                 arrays.get("cam_main", blank), arrays.get("cam_wrist", blank)))
            elif cmd == "quit":
                reply({"ok": True, "bye": True})
                return 0
            else:
                reply({"ok": False, "error": f"unknown cmd {cmd}"})
        except Exception as exc:  # noqa: BLE001
            _err(traceback.format_exc())
            reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


# ================================================================================================ download
def download(repo: str, local_dir: str, with_backbone: bool = True) -> int:
    from huggingface_hub import snapshot_download  # type: ignore
    os.makedirs(local_dir, exist_ok=True)
    print(f"[V4.4 DOWNLOAD] {repo} -> {local_dir}  (HF_HOME={os.environ.get('HF_HOME')})", flush=True)
    t0 = time.time()
    path = snapshot_download(repo_id=repo, local_dir=local_dir)
    print(f"[OK] {repo} downloaded in {time.time() - t0:.0f} s -> {path}", flush=True)
    if with_backbone:
        try:
            with open(os.path.join(local_dir, "config.json"), "r", encoding="utf-8") as f:
                vlm = json.load(f).get("vlm_model_name")
        except Exception:  # noqa: BLE001
            vlm = None
        if vlm:
            print(f"[V4.4 DOWNLOAD] VLM backbone {vlm} -> HF cache {os.environ.get('HF_HUB_CACHE')}", flush=True)
            snapshot_download(repo_id=vlm)
            print(f"[OK] backbone {vlm} cached", flush=True)
    print("NERO_JOB_DONE ok", flush=True)
    return 0


# ================================================================================================ export
def read_nero_episodes(src_dir: str, splits: Tuple[str, ...] = ("train",), valid_only: bool = True) -> List[Dict[str, Any]]:
    """Episode metadata of a NERO V4.2 recorder dataset (meta/episodes.jsonl + data/episode_XXXXXX.npz)."""
    eps = []
    p = os.path.join(src_dir, "meta", "episodes.jsonl")
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if splits and str(e.get("split", "train")) not in splits:
                continue
            if valid_only and not e.get("valid", True):
                continue
            npz = os.path.join(src_dir, "data", f"episode_{int(e['episode_index']):06d}.npz")
            if os.path.isfile(npz):
                e["_npz"] = npz
                eps.append(e)
    return eps


def iter_episode_frames(ep: Dict[str, Any]):
    """Yield (frame dict with LeRobot keys, task) for every step of one NERO episode."""
    with np.load(ep["_npz"], allow_pickle=False) as d:
        S, A = d["observation_state"].astype(np.float32), d["action"].astype(np.float32)
        C1, C2 = d["cam_main"], d["cam_wrist"]
    task = (ep.get("tasks") or [""])[0] or "manipulation task"
    shape = None
    for C in (C1, C2):
        if C.ndim == 4 and len(C):
            shape = C.shape[1:]
            break
    shape = tuple(shape or (120, 160, 3))
    blank = np.zeros(shape, dtype=np.uint8)
    for t in range(len(S)):
        c1 = C1[min(t, len(C1) - 1)] if C1.ndim == 4 and len(C1) else blank
        c2 = C2[min(t, len(C2) - 1)] if C2.ndim == 4 and len(C2) else blank
        yield {"observation.state": S[t][:8], "action": A[t][:8],
               "observation.images.cam_main": np.ascontiguousarray(c1, dtype=np.uint8),
               "observation.images.cam_wrist": np.ascontiguousarray(c2, dtype=np.uint8)}, task


def lerobot_features(img_shape: Tuple[int, int, int], use_videos: bool) -> Dict[str, Any]:
    kind = "video" if use_videos else "image"
    return {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES_8D},
        "action": {"dtype": "float32", "shape": (8,), "names": JOINT_NAMES_8D},
        "observation.images.cam_main": {"dtype": kind, "shape": tuple(img_shape), "names": ["height", "width", "channels"]},
        "observation.images.cam_wrist": {"dtype": kind, "shape": tuple(img_shape), "names": ["height", "width", "channels"]},
    }


def _import_lerobot_dataset():
    for mp in ("lerobot.datasets.lerobot_dataset", "lerobot.common.datasets.lerobot_dataset"):
        try:
            return __import__(mp, fromlist=["LeRobotDataset"]).LeRobotDataset
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] import {mp}: {exc!r}", flush=True)
    raise ImportError("LeRobotDataset not importable")


def export(src_dir: str, repo_id: str, out_root: str, fps: float, splits: Tuple[str, ...], overwrite: bool,
           use_videos: bool = True) -> int:
    eps = read_nero_episodes(src_dir, splits=splits)
    if not eps:
        print(f"[FAIL] no valid episodes with split {splits} in {src_dir}", flush=True)
        return 2
    first = next(iter_episode_frames(eps[0]))[0]
    img_shape = first["observation.images.cam_main"].shape
    LeRobotDataset = _import_lerobot_dataset()

    def _run(videos: bool) -> None:
        if os.path.exists(out_root):
            if not overwrite:
                raise FileExistsError(f"{out_root} exists (tick overwrite)")
            shutil.rmtree(out_root)
        os.makedirs(os.path.dirname(out_root), exist_ok=True)
        ds = LeRobotDataset.create(repo_id=repo_id, fps=int(round(fps)), root=out_root, robot_type="nero_so101_7dof",
                                   features=lerobot_features(img_shape, videos), use_videos=videos)
        takes_task = "task" in inspect.signature(ds.add_frame).parameters
        n_frames = 0
        for i, ep in enumerate(eps):
            for frame, task in iter_episode_frames(ep):
                if takes_task:
                    ds.add_frame(frame, task=task)
                else:
                    frame["task"] = task
                    ds.add_frame(frame)
                n_frames += 1
            ds.save_episode()
            print(f"[OK] episode {i + 1}/{len(eps)} (NERO #{ep['episode_index']}, task={ep.get('tasks')}) saved", flush=True)
        if hasattr(ds, "finalize"):
            ds.finalize()
        print(f"[OK] LeRobotDataset {repo_id}: {len(eps)} episodes, {n_frames} frames -> {out_root} "
              f"({'video' if videos else 'png images'})", flush=True)

    print(f"[V4.4 EXPORT] {src_dir} -> {out_root}  repo_id={repo_id} fps={fps} splits={splits}", flush=True)
    try:
        _run(use_videos)
    except FileExistsError as exc:
        print(f"[FAIL] {exc}", flush=True)
        return 3
    except Exception as exc:  # noqa: BLE001
        if not use_videos:
            raise
        print(f"[WARN] video encoding failed ({exc!r}); retrying with PNG images (no ffmpeg needed)", flush=True)
        overwrite = True
        _run(False)
    print("NERO_JOB_DONE ok", flush=True)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=f"NERO {VERSION} SmolVLA worker (runs inside the drive venv)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--fake", action="store_true")
    d = sub.add_parser("download")
    d.add_argument("--repo", default="lerobot/smolvla_base")
    d.add_argument("--local-dir", required=True)
    d.add_argument("--no-backbone", action="store_true")
    e = sub.add_parser("export")
    e.add_argument("--src", required=True)
    e.add_argument("--repo-id", required=True)
    e.add_argument("--root", required=True)
    e.add_argument("--fps", type=float, default=20.0)
    e.add_argument("--splits", default="train")
    e.add_argument("--overwrite", action="store_true")
    e.add_argument("--images", action="store_true", help="store PNG images instead of mp4 video")
    sub.add_parser("env")
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        return serve(fake=a.fake or os.environ.get("NERO_V44_WORKER_FAKE") == "1")
    if a.cmd == "download":
        return download(a.repo, a.local_dir, with_backbone=not a.no_backbone)
    if a.cmd == "export":
        return export(a.src, a.repo_id, a.root, a.fps, tuple(x for x in a.splits.split(",") if x), a.overwrite,
                      use_videos=not a.images)
    if a.cmd == "env":
        print(json.dumps(env_info(), ensure_ascii=False))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
