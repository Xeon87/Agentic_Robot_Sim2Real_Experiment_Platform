#!/usr/bin/env python3
"""NERO V2-24 isolated GPU motion-planner daemon.

cuRobo/cuMotion stay outside the ROS2 Humble Python process.  The daemon
accepts a tiny localhost JSON protocol and returns 7-DOF joint trajectories.

V2-22.2 fixes:
- warm cuRobo before the server advertises READY;
- handle MotionPlanner.plan_pose() returning None;
- flatten cuRobo's [batch, seed, horizon, dof] JointState safely to [N,7];
- use cuRobo's configured interpolation_dt instead of guessing JointState.dt;
- V2-23 disables runtime world-sync; MoveIt is the authoritative environment gate;
- add a true pose-planning self-test with no robot motion.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socketserver
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

SERVER_VERSION = "V2-24"
JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


def _as_float_list(x: Any) -> List[float]:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=float).reshape(-1)
    return [float(v) for v in a]


def _joint_rows(x: Any, dof: int = 7) -> tuple[List[List[float]], List[int]]:
    """Convert arbitrary cuRobo JointState tensor shape (..., horizon, dof) to [N,dof]."""
    if x is None:
        return [], []
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=float)
    raw_shape = [int(v) for v in a.shape]
    if a.ndim < 2 or a.shape[-1] != dof:
        raise RuntimeError(f"unexpected trajectory tensor shape {tuple(a.shape)}; expected (...,N,{dof})")
    rows = a.reshape(-1, dof)
    if rows.shape[0] < 1:
        return [], raw_shape
    if not np.all(np.isfinite(rows)):
        raise RuntimeError("trajectory contains non-finite values")
    return rows.astype(float).tolist(), raw_shape


def _path_waypoints(path: Any, max_samples: int = 120) -> List[List[float]]:
    if path is None:
        return []
    wp = getattr(path, "waypoints", None)
    if callable(wp):
        vals = wp()
        return [_as_float_list(v) for v in vals]
    dom_fn = getattr(path, "domain", None)
    eval_fn = getattr(path, "eval", None)
    if callable(dom_fn) and callable(eval_fn):
        dom = dom_fn()
        lo = float(dom.lower)
        hi = float(dom.upper)
        n = max(2, min(max_samples, int(math.ceil(max(1e-6, hi - lo) / 0.02)) + 1))
        return [_as_float_list(eval_fn(t)) for t in np.linspace(lo, hi, n)]
    arr = np.asarray(path, dtype=float)
    if arr.ndim == 2:
        return [[float(v) for v in row] for row in arr]
    return []


def _success_any(value: Any) -> bool:
    if value is None:
        return False
    if hasattr(value, "any"):
        v = value.any()
        if hasattr(v, "item"):
            return bool(v.item())
        return bool(v)
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


class PlannerHost:
    def __init__(self, config_dir: Path, preferred: str = "auto"):
        self.config_dir = config_dir.expanduser().resolve()
        self.preferred = preferred
        self.lock = threading.Lock()
        self._cumotion = None
        self._cumotion_state: Dict[str, Any] = {}
        self._curobo = None
        self._curobo_state: Dict[str, Any] = {}
        self.errors: Dict[str, str] = {}
        self._curobo_world_signature = None
        self._curobo_world_count = 0
        self._probe_imports()

    @property
    def urdf(self) -> Path:
        return self.config_dir / "nero.urdf"

    @property
    def xrdf(self) -> Path:
        return self.config_dir / "nero_cumotion.xrdf"

    @property
    def curobo_yml(self) -> Path:
        return self.config_dir / "nero_curobo.yml"

    def _probe_imports(self):
        try:
            import cumotion  # type: ignore
            self._cumotion = cumotion
        except Exception as exc:
            self.errors["cumotion"] = f"import failed: {exc}"
        try:
            import curobo  # type: ignore
            self._curobo = curobo
        except Exception as exc:
            self.errors["curobo"] = f"import failed: {exc}"

    def _ensure_cumotion(self):
        if self._cumotion is None:
            raise RuntimeError(self.errors.get("cumotion", "cuMotion is not installed"))
        if self._cumotion_state:
            return
        if not self.urdf.is_file() or not self.xrdf.is_file():
            raise RuntimeError(f"cuMotion robot files missing: {self.xrdf} + {self.urdf}")
        c = self._cumotion
        robot = c.load_robot_from_file(str(self.xrdf), str(self.urdf))
        names = [robot.cspace_coord_name(i) for i in range(robot.num_cspace_coords())]
        if names != JOINT_NAMES:
            raise RuntimeError(f"cuMotion joint order mismatch: {names} != {JOINT_NAMES}")
        world = c.create_world()
        if hasattr(world, "add_world_view"):
            view = world.add_world_view()
        elif hasattr(world, "create_world_view"):
            view = world.create_world_view()
        else:
            raise RuntimeError("cuMotion World has no world-view creation method")
        cfg = c.create_default_motion_planner_config(robot, "tcp_link", view)
        planner = c.create_motion_planner(cfg)
        kin = robot.kinematics()
        tg = c.create_cspace_trajectory_generator(kin)
        self._cumotion_state = {
            "robot": robot, "world": world, "view": view,
            "planner": planner, "kin": kin, "trajgen": tg,
            "joint_names": names,
        }

    def _ensure_curobo(self):
        if self._curobo is None:
            raise RuntimeError(self.errors.get("curobo", "cuRobo is not installed"))
        if self._curobo_state:
            return
        if not self.curobo_yml.is_file():
            raise RuntimeError(f"cuRobo robot config missing: {self.curobo_yml}")
        import torch
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

        # A collision cache creates a scene checker even before obstacles are loaded,
        # so MoveIt world snapshots can later be pushed with update_world().
        cfg = MotionPlannerCfg.create(
            robot=str(self.curobo_yml),
            collision_cache={"cuboid": 128},
            interpolation_dt=0.025,
        )
        planner = MotionPlanner(cfg)
        t0 = time.perf_counter()
        planner.warmup(enable_graph=True, num_warmup_iterations=5)
        warmup_ms = (time.perf_counter() - t0) * 1000.0
        names = [str(v) for v in planner.joint_names]
        if names != JOINT_NAMES:
            raise RuntimeError(f"cuRobo joint order mismatch: {names} != {JOINT_NAMES}")
        self._curobo_state = {
            "torch": torch,
            "planner": planner,
            "joint_names": names,
            "warmup_ms": warmup_ms,
        }

    def _clear_curobo_world(self) -> None:
        """V2-23: guarantee cuRobo has no stale environment world loaded."""
        self._ensure_curobo()
        if self._curobo_world_count == 0 and self._curobo_world_signature == tuple():
            return
        from curobo._src.geom.types import SceneCfg
        self._curobo_state["planner"].update_world(SceneCfg.create({}))
        self._curobo_world_signature = tuple()
        self._curobo_world_count = 0

    def _apply_curobo_world(self, world: Any) -> int:
        """Load a conservative OBB world snapshot sent by the ROS/MoveIt process."""
        if world is None:
            return self._curobo_world_count
        self._ensure_curobo()
        cuboids = world.get("cuboids", []) if isinstance(world, dict) else []
        if not isinstance(cuboids, list):
            raise RuntimeError("world.cuboids must be a list")
        normalized = []
        cfg_dict: Dict[str, Any] = {"cuboid": {}}
        for i, ob in enumerate(cuboids):
            if not isinstance(ob, dict):
                continue
            name = str(ob.get("name", f"obstacle_{i}"))
            pose = [float(v) for v in ob.get("pose", [])]
            dims = [float(v) for v in ob.get("dims", [])]
            if len(pose) != 7 or len(dims) != 3:
                continue
            if not all(math.isfinite(v) for v in pose + dims) or min(dims) <= 0.0:
                continue
            # Unique key even when one MoveIt object contains several primitives/meshes.
            key = f"{i:03d}_{name}".replace("/", "_")
            cfg_dict["cuboid"][key] = {"pose": pose, "dims": dims}
            normalized.append((key, tuple(round(v, 6) for v in pose), tuple(round(v, 6) for v in dims)))
        sig = tuple(normalized)
        if sig == self._curobo_world_signature:
            return self._curobo_world_count
        from curobo._src.geom.types import SceneCfg
        scene = SceneCfg.create(cfg_dict)
        self._curobo_state["planner"].update_world(scene)
        self._curobo_world_signature = sig
        self._curobo_world_count = len(normalized)
        return self._curobo_world_count

    def status(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "available": True,
            "server_version": SERVER_VERSION,
            "preferred": self.preferred,
            "config_dir": str(self.config_dir),
            "robot_files": {
                "urdf": self.urdf.is_file(),
                "xrdf": self.xrdf.is_file(),
                "curobo_yml": self.curobo_yml.is_file(),
            },
            "engines": {
                "cumotion": {"imported": self._cumotion is not None, "initialized": bool(self._cumotion_state)},
                "curobo": {
                    "imported": self._curobo is not None,
                    "initialized": bool(self._curobo_state),
                    "warmup_ms": self._curobo_state.get("warmup_ms"),
                    "world_cuboids": self._curobo_world_count,
                    "world_mode": "disabled_moveit_full_trajectory_gate",
                },
            },
            "errors": dict(self.errors),
        }

    def _plan_cumotion(self, req: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_cumotion()
        c = self._cumotion
        st = self._cumotion_state
        q = np.asarray(req["q_start"], dtype=np.float64)
        pos = np.asarray(req["position"], dtype=np.float64)
        x, y, z, w = [float(v) for v in req["quat_xyzw"]]
        pose = c.Pose3(c.Rotation3(w, x, y, z), pos)
        t0 = time.perf_counter()
        result = st["planner"].plan_to_pose_target(q, pose, generate_interpolated_path=True)
        planning_ms = (time.perf_counter() - t0) * 1000.0
        if not bool(result.path_found):
            return {"ok": False, "engine": "cumotion", "error": "path_found=false", "planning_ms": planning_ms}
        path = result.interpolated_path if getattr(result, "interpolated_path", None) is not None else result.path
        waypoints = _path_waypoints(path)
        if len(waypoints) < 2:
            return {"ok": False, "engine": "cumotion", "error": "planner returned <2 path waypoints", "planning_ms": planning_ms}
        traj = st["trajgen"].generate_trajectory([np.asarray(v, dtype=np.float64) for v in waypoints])
        dom = traj.domain()
        lo, hi = float(dom.lower), float(dom.upper)
        if not math.isfinite(hi) or hi <= lo:
            return {"ok": False, "engine": "cumotion", "error": "invalid timed trajectory domain", "planning_ms": planning_ms}
        times = list(np.arange(lo, hi, 0.02)) + [hi]
        positions = [_as_float_list(traj.eval(t, 0)) for t in times]
        velocities = [_as_float_list(traj.eval(t, 1)) for t in times]
        return {
            "ok": True, "engine": "cumotion", "planning_ms": planning_ms,
            "joint_names": st["joint_names"],
            "positions": positions, "velocities": velocities,
            "times": [float(t - lo) for t in times],
            "raw_waypoints": len(waypoints),
        }

    def _plan_curobo(self, req: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_curobo()
        # V2-23: never load the MoveIt scene into cuRobo.  Clear any stale world
        # from a previous daemon request; MoveIt will validate the full trajectory.
        self._clear_curobo_world()
        world_count = 0
        st = self._curobo_state
        torch = st["torch"]
        seed = int(req.get("random_seed") or (23000 + int(req.get("retry_index", 0))))
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        planner = st["planner"]
        from curobo.types import GoalToolPose, JointState

        q_list = [float(v) for v in req["q_start"]]
        q = torch.tensor([q_list], device="cuda", dtype=torch.float32)
        q_start = JointState.from_position(q, joint_names=st["joint_names"])
        x, y, z, w = [float(v) for v in req["quat_xyzw"]]
        p = torch.tensor(
            [[[[[float(req["position"][0]), float(req["position"][1]), float(req["position"][2])]]]]],
            device="cuda", dtype=torch.float32,
        )
        quat = torch.tensor([[[[[w, x, y, z]]]]], device="cuda", dtype=torch.float32)
        goal = GoalToolPose(tool_frames=planner.tool_frames, position=p, quaternion=quat)

        t0 = time.perf_counter()
        result = planner.plan_pose(goal, q_start, max_attempts=int(req.get("max_attempts", 5)))
        planning_ms = (time.perf_counter() - t0) * 1000.0
        if result is None:
            return {
                "ok": False, "engine": "curobo",
                "error": "plan_pose returned None",
                "planning_ms": planning_ms, "world_cuboids": world_count,
            }
        if not _success_any(getattr(result, "success", None)):
            return {
                "ok": False, "engine": "curobo",
                "error": str(getattr(result, "status", "success=false")),
                "planning_ms": planning_ms, "world_cuboids": world_count,
            }
        plan = result.get_interpolated_plan()
        if plan is None or getattr(plan, "position", None) is None:
            return {
                "ok": False, "engine": "curobo", "error": "no interpolated plan",
                "planning_ms": planning_ms, "world_cuboids": world_count,
            }

        positions, raw_shape = _joint_rows(plan.position, dof=7)
        if len(positions) < 2:
            return {
                "ok": False, "engine": "curobo", "error": "interpolated plan has <2 points",
                "planning_ms": planning_ms, "raw_position_shape": raw_shape,
                "world_cuboids": world_count,
            }
        velocities: List[List[float]] = []
        if getattr(plan, "velocity", None) is not None:
            velocities, _ = _joint_rows(plan.velocity, dof=7)
            if len(velocities) != len(positions):
                velocities = []

        # Official cuRobo examples use trajopt_solver.config.interpolation_dt for
        # get_interpolated_plan(); do not coerce JointState.dt tensors to scalar.
        dt = float(planner.trajopt_solver.config.interpolation_dt)
        times = [float(i * dt) for i in range(len(positions))]
        start_err = max(abs(float(positions[0][j]) - q_list[j]) for j in range(7))

        return {
            "ok": True,
            "engine": "curobo",
            "planning_ms": planning_ms,
            "joint_names": st["joint_names"],
            "positions": positions,
            "velocities": velocities,
            "times": times,
            "raw_position_shape": raw_shape,
            "flattened_shape": [len(positions), 7],
            "interpolation_dt": dt,
            "start_error_max_rad": float(start_err),
            "world_cuboids": world_count,
            "retry_index": int(req.get("retry_index", 0)),
            "random_seed": seed,
        }

    def fk_batch(self, req: Dict[str, Any]) -> Dict[str, Any]:
        """Batch forward-kinematics for NERO J1..J7 using cuRobo.

        This is planning-only.  It lets the ROS/backend score sparse wrist
        candidates and predict the held egg pose without physically probing the
        robot.  q_batch shape is [B,7].
        """
        self._ensure_curobo()
        q_batch = req.get("q_batch")
        names = req.get("joint_names")
        if names != JOINT_NAMES or not isinstance(q_batch, list) or not q_batch:
            return {"ok": False, "error": "invalid joint_names/q_batch"}
        rows = []
        for i, q in enumerate(q_batch):
            if not isinstance(q, list) or len(q) != 7:
                return {"ok": False, "error": f"q_batch[{i}] is not 7-DOF"}
            vals = [float(v) for v in q]
            if not all(math.isfinite(v) for v in vals):
                return {"ok": False, "error": f"q_batch[{i}] has non-finite value"}
            rows.append(vals)

        st = self._curobo_state
        torch = st["torch"]
        from curobo.types import JointState
        q_t = torch.tensor(rows, device="cuda", dtype=torch.float32)
        js = JointState.from_position(q_t, joint_names=st["joint_names"])
        kin = st["planner"].compute_kinematics(js)
        poses = kin.tool_poses
        p = poses.position.detach().cpu().numpy()
        q = poses.quaternion.detach().cpu().numpy()
        p = np.asarray(p, dtype=float).reshape(len(rows), -1, 3)[:, 0, :]
        q = np.asarray(q, dtype=float).reshape(len(rows), -1, 4)[:, 0, :]
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
            return {"ok": False, "error": "FK returned non-finite pose"}
        # cuRobo Pose quaternion order is wxyz; external NERO convention is xyzw.
        out = []
        for pi, qi in zip(p, q):
            out.append({
                "position": [float(v) for v in pi],
                "quat_xyzw": [float(qi[1]), float(qi[2]), float(qi[3]), float(qi[0])],
            })
        return {
            "ok": True,
            "engine": "curobo_fk",
            "joint_names": list(st["joint_names"]),
            "count": len(out),
            "poses": out,
        }

    def _self_test_curobo_pose(self) -> Dict[str, Any]:
        self._ensure_curobo()
        from curobo._src.geom.types import SceneCfg
        from curobo.types import JointState
        st = self._curobo_state
        planner = st["planner"]
        # Clear runtime world for a deterministic no-obstacle self-test.
        planner.update_world(SceneCfg.create({}))
        self._curobo_world_signature = tuple()
        self._curobo_world_count = 0
        q0 = planner.default_joint_state.position.unsqueeze(0).clone()
        q_start = JointState.from_position(q0, joint_names=st["joint_names"])
        q_goal = q_start.clone()
        q_goal.position[..., 0] += 0.03
        goal = planner.compute_kinematics(q_goal).tool_poses.as_goal()
        t0 = time.perf_counter()
        result = planner.plan_pose(goal, q_start, max_attempts=3)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if result is None:
            return {"ok": False, "error": "plan_pose returned None", "elapsed_ms": elapsed_ms}
        if not _success_any(getattr(result, "success", None)):
            return {"ok": False, "error": str(getattr(result, "status", "success=false")), "elapsed_ms": elapsed_ms}
        plan = result.get_interpolated_plan()
        if plan is None or getattr(plan, "position", None) is None:
            return {"ok": False, "error": "no interpolated plan", "elapsed_ms": elapsed_ms}
        positions, raw_shape = _joint_rows(plan.position, 7)
        start = q0.detach().cpu().numpy().reshape(-1).astype(float)
        start_err = max(abs(positions[0][j] - float(start[j])) for j in range(7)) if positions else float("inf")
        return {
            "ok": len(positions) >= 2 and start_err < 0.15,
            "path_points": len(positions),
            "raw_position_shape": raw_shape,
            "flattened_shape": [len(positions), 7],
            "start_error_max_rad": float(start_err),
            "elapsed_ms": elapsed_ms,
        }

    def self_test(self, engine: str) -> Dict[str, Any]:
        requested = str(engine or "auto").strip().lower()
        order = ["cumotion", "curobo"] if requested in ("auto", "gpu", "gpu auto") else [requested]
        results: Dict[str, Any] = {}
        any_ok = False
        with self.lock:
            for name in order:
                t0 = time.perf_counter()
                try:
                    if name == "curobo":
                        results[name] = self._self_test_curobo_pose()
                    elif name == "cumotion":
                        self._ensure_cumotion()
                        st = self._cumotion_state
                        robot = st["robot"]
                        q0 = np.asarray(robot.default_cspace_configuration(), dtype=np.float64).reshape(-1)
                        q1 = q0.copy(); q1[0] += 0.03
                        r = st["planner"].plan_to_cspace_target(q0, q1, generate_interpolated_path=True)
                        ok = bool(r.path_found)
                        n = len(_path_waypoints(r.interpolated_path if getattr(r, "interpolated_path", None) is not None else r.path)) if ok else 0
                        results[name] = {"ok": ok, "path_points": n, "elapsed_ms": (time.perf_counter()-t0)*1000.0}
                    else:
                        results[name] = {"ok": False, "error": f"unsupported engine {name}"}
                except Exception as exc:
                    self.errors[name] = f"{type(exc).__name__}: {exc}"
                    results[name] = {"ok": False, "error": self.errors[name], "elapsed_ms": (time.perf_counter()-t0)*1000.0}
                any_ok = any_ok or bool(results[name].get("ok"))
        return {"ok": any_ok, "self_test": results, "status": self.status()}

    def prewarm(self, engine: str) -> Dict[str, Any]:
        requested = str(engine or "none").strip().lower()
        if requested in ("", "none", "off"):
            return {"ok": True, "prewarm": "none"}
        out = self.self_test(requested)
        if not out.get("ok"):
            raise RuntimeError(f"GPU prewarm/self-test failed: {out}")
        return out

    def plan_pose(self, req: Dict[str, Any]) -> Dict[str, Any]:
        q = req.get("q_start"); p = req.get("position"); quat = req.get("quat_xyzw"); names = req.get("joint_names")
        if names != JOINT_NAMES or not isinstance(q, list) or len(q) != 7:
            return {"ok": False, "error": "invalid NERO joint_names/q_start"}
        if not isinstance(p, list) or len(p) != 3 or not isinstance(quat, list) or len(quat) != 4:
            return {"ok": False, "error": "invalid target pose"}
        requested = str(req.get("engine", "auto")).strip().lower()
        if requested in ("gpu auto", "auto", "gpu"):
            order = ["cumotion", "curobo"] if self.preferred != "curobo" else ["curobo", "cumotion"]
        elif requested in ("cumotion", "cu_motion"):
            order = ["cumotion"]
        elif requested in ("curobo", "cu_robo"):
            order = ["curobo"]
        else:
            return {"ok": False, "error": f"unsupported GPU engine: {requested}"}
        errs = []
        with self.lock:
            for engine in order:
                try:
                    out = self._plan_cumotion(req) if engine == "cumotion" else self._plan_curobo(req)
                    if out.get("ok"):
                        return out
                    errs.append(f"{engine}:{out.get('error', 'failed')}")
                except Exception as exc:
                    self.errors[engine] = f"{type(exc).__name__}: {exc}"
                    errs.append(f"{engine}:{self.errors[engine]}")
        return {"ok": False, "engine": requested, "error": " | ".join(errs)}


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            line = self.rfile.readline(32 * 1024 * 1024)
            if not line:
                return
            req = json.loads(line.decode("utf-8"))
            op = req.get("op")
            if op == "status":
                out = self.server.host.status()  # type: ignore[attr-defined]
            elif op == "self_test":
                out = self.server.host.self_test(req.get("engine", "auto"))  # type: ignore[attr-defined]
            elif op == "plan_pose":
                out = self.server.host.plan_pose(req)  # type: ignore[attr-defined]
            elif op == "fk_batch":
                with self.server.host.lock:  # type: ignore[attr-defined]
                    out = self.server.host.fk_batch(req)  # type: ignore[attr-defined]
            else:
                out = {"ok": False, "error": f"unknown op: {op}"}
        except Exception as exc:
            out = {"ok": False, "error": f"server exception: {type(exc).__name__}: {exc}"}
            if os.environ.get("NERO_GPU_DEBUG") == "1":
                out["traceback"] = traceback.format_exc()
        self.wfile.write((json.dumps(out, separators=(",", ":")) + "\n").encode("utf-8"))


class ThreadingTCP(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--config-dir", default=os.environ.get("NERO_GPU_CONFIG_DIR", "~/.nero_gpu_planner/config"))
    ap.add_argument("--prefer", choices=["auto", "cumotion", "curobo"], default="curobo")
    ap.add_argument("--prewarm", choices=["none", "cumotion", "curobo", "auto"], default="curobo")
    ns = ap.parse_args()
    host = PlannerHost(Path(ns.config_dir), preferred=ns.prefer)
    print(f"[GPU SERVER {SERVER_VERSION}] prewarming {ns.prewarm}...", flush=True)
    try:
        pre = host.prewarm(ns.prewarm)
        print("[GPU PREWARM PASS] " + json.dumps(pre, separators=(",", ":")), flush=True)
    except Exception as exc:
        print(f"[GPU PREWARM FAIL] {type(exc).__name__}: {exc}", flush=True)
        raise SystemExit(4)
    with ThreadingTCP((ns.host, ns.port), Handler) as server:
        server.host = host  # type: ignore[attr-defined]
        print(f"[GPU SERVER READY] version={SERVER_VERSION} {ns.host}:{ns.port} config={host.config_dir}", flush=True)
        print(json.dumps(host.status(), indent=2), flush=True)
        server.serve_forever(poll_interval=0.2)


if __name__ == "__main__":
    main()
