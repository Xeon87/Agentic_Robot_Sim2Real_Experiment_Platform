#!/usr/bin/env python3
"""Tiny localhost JSON client for the optional NERO GPU motion-planner daemon.

This module deliberately has no CUDA/cuRobo/cuMotion imports.  It is safe to
install inside the existing ROS2 Humble runtime.  GPU dependencies live in a
separate virtual environment/process.
"""
from __future__ import annotations

import json
import os
import socket
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_HOST = os.environ.get("NERO_GPU_PLANNER_HOST", "127.0.0.1")
def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)

DEFAULT_PORT = _env_int("NERO_GPU_PLANNER_PORT", 8765)
DEFAULT_TIMEOUT_S = _env_float("NERO_GPU_PLANNER_TIMEOUT_S", 8.0)


class GPUPlannerClient:
    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 timeout_s: float = DEFAULT_TIMEOUT_S):
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)

    def _request(self, payload: Dict[str, Any], timeout_s: Optional[float] = None) -> Dict[str, Any]:
        raw = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        with socket.create_connection((self.host, self.port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(raw)
            buf = bytearray()
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf.extend(chunk)
                if b"\n" in chunk:
                    break
                if len(buf) > 32 * 1024 * 1024:
                    raise RuntimeError("GPU planner response exceeded 32 MiB")
        if not buf:
            raise RuntimeError("GPU planner returned an empty response")
        line = bytes(buf).split(b"\n", 1)[0]
        out = json.loads(line.decode("utf-8"))
        if not isinstance(out, dict):
            raise RuntimeError("GPU planner returned a non-object JSON response")
        return out

    def ping(self, timeout_s: float = 0.7) -> Dict[str, Any]:
        try:
            return self._request({"op": "status"}, timeout_s=timeout_s)
        except Exception as exc:
            return {"ok": False, "available": False, "error": str(exc)}

    def fk_batch(self, *, joint_names: Iterable[str], q_batch: Iterable[Iterable[float]],
                 timeout_s: Optional[float] = None) -> Dict[str, Any]:
        return self._request({
            "op": "fk_batch",
            "joint_names": [str(v) for v in joint_names],
            "q_batch": [[float(x) for x in q] for q in q_batch],
        }, timeout_s=timeout_s)

    def plan_pose(self, *, engine: str, joint_names: Iterable[str], q_start: Iterable[float],
                  position: Iterable[float], quat_xyzw: Iterable[float],
                  velocity_scale: float = 0.25, accel_scale: float = 0.25,
                  world: Optional[Dict[str, Any]] = None, max_attempts: int = 5,
                  retry_index: int = 0, random_seed: Optional[int] = None,
                  timeout_s: Optional[float] = None) -> Dict[str, Any]:
        return self._request({
            "op": "plan_pose",
            "engine": str(engine),
            "joint_names": [str(v) for v in joint_names],
            "q_start": [float(v) for v in q_start],
            "position": [float(v) for v in position],
            "quat_xyzw": [float(v) for v in quat_xyzw],
            "velocity_scale": float(velocity_scale),
            "accel_scale": float(accel_scale),
            "world": world,
            "max_attempts": int(max_attempts),
            "retry_index": int(retry_index),
            "random_seed": None if random_seed is None else int(random_seed),
        }, timeout_s=timeout_s)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", choices=["cumotion", "curobo", "auto"])
    ns = ap.parse_args()
    c = GPUPlannerClient()
    if ns.self_test:
        out = c._request({"op":"self_test", "engine":ns.self_test}, timeout_s=90.0)
    else:
        out = c.ping(timeout_s=2.0)
    print(json.dumps(out, indent=2, sort_keys=True))
    raise SystemExit(0 if out.get("ok") else 2)
