#!/usr/bin/env python3
"""NERO V4.8.7 — cuRobo GPU IK Manager, Server Lifecycle & FastIK (45-seed) Fallback Bridge.

V4.8.7 CUROBO REAL IK + SMOLVLA VRAM RELEASE:
1. Lifecycle management (`ensure_curobo_server`, `stop_curobo_server`, `curobo_status`):
   - Checks if `NERO_GPU_Planner_Server.py` (localhost:8765) is already running and responsive.
   - Starts `NERO_GPU_Planner_Server.py` on demand during real-robot bring-up (`run_all` / `step_env`) or when an
     Agentic real-robot task starts, using `~/.nero_gpu_planner/.venv/bin/python` and `~/.nero_gpu_planner/config`.
   - Stops `NERO_GPU_Planner_Server.py` cleanly before heavy GPU VRAM operations (`SmolVLA` model loading, closed-loop
     inference, `lerobot-train` fine-tuning, or `YOLOE` fine-tuning) so 1.5-2.2 GB of GPU VRAM is freed immediately.
2. Default IK solver with automatic 45-seed FastIK fallback (`solve_pose`, `solve_chain`):
   - Converts target `(xyz, R)` poses to `(xyz, quat_xyzw)` in the URDF `tcp_link` frame (`FastArm.tool`, identical to
     `nero_curobo.yml` tool frame).
   - Queries `GPUPlannerClient.plan_pose(engine="curobo")` seeded from `q_seed`.
   - Validates returned joint solutions against `FastArm.fk` accuracy (`tol_m`, `tol_deg`), joint limits, and J1 base
     rotation bounds (`<= 120 deg`).
   - Seamlessly falls back to the 45-seed `FastArm` solver (`fast_ik_45seeds`) whenever `cuRobo` is offline, stopped
     for SmolVLA VRAM, or rejects a pose.
"""
from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

VERSION = "V4.8.7"
MARKER = "V4.8.7 CUROBO REAL IK + SMOLVLA VRAM RELEASE"
HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import NERO_GPU_Planner_Client as GPC

JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
DEFAULT_GPU_ROOT = os.path.expanduser(os.environ.get("NERO_GPU_ROOT", "~/.nero_gpu_planner"))
DEFAULT_GPU_PY = os.path.expanduser(os.environ.get("NERO_GPU_PLANNER_PYTHON", os.path.join(DEFAULT_GPU_ROOT, ".venv", "bin", "python")))
DEFAULT_GPU_CFG = os.path.expanduser(os.environ.get("NERO_GPU_CONFIG_DIR", os.path.join(DEFAULT_GPU_ROOT, "config")))
SERVER_SCRIPT = os.path.join(HERE, "NERO_GPU_Planner_Server.py")

# Process-wide state & short-TTL availability cache so repeated inner-loop IK calls never block when server is offline
_LOCK = threading.RLock()
_STATE: Dict[str, Any] = {
    "proc": None,               # subprocess.Popen if started by this console process
    "pid": None,
    "vram_yielded": False,      # True while SmolVLA / training explicitly stopped cuRobo for VRAM
    "yield_reason": "",
    "last_ping_t": 0.0,
    "last_ping_ok": False,
    "last_status": {},
    "mock_solver": None,        # Optional callable(q_start, pos, quat_xyzw) -> q_goal or list[q] for unit tests
}


def mat_to_quat_xyzw(R: np.ndarray) -> List[float]:
    """Convert a 3x3 rotation matrix to unit quaternion [x, y, z, w]."""
    m = np.asarray(R, dtype=float).reshape(3, 3)
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    if w < 0.0:
        n = -n
    return [float(x / n), float(y / n), float(z / n), float(w / n)]


def set_mock_solver(fn: Optional[Callable]) -> None:
    """Install or clear an in-process mock cuRobo solver for unit tests."""
    with _LOCK:
        _STATE["mock_solver"] = fn
        _STATE["last_ping_t"] = time.monotonic()
        _STATE["last_ping_ok"] = fn is not None
        if fn is not None:
            _STATE["vram_yielded"] = False


def find_server_pids(proc_root: str = "/proc") -> List[int]:
    """Find running NERO_GPU_Planner_Server.py PIDs on the host."""
    pids: List[int] = []
    try:
        names = os.listdir(proc_root)
    except Exception:
        return pids
    me = os.getpid()
    for d in names:
        if not d.isdigit() or int(d) == me:
            continue
        try:
            with open(os.path.join(proc_root, d, "cmdline"), "rb") as f:
                raw = f.read()
            argv = [a.decode(errors="replace") for a in raw.split(b"\0") if a]
            if any(os.path.basename(a) == "NERO_GPU_Planner_Server.py" for a in argv[1:4]):
                pids.append(int(d))
        except Exception:
            continue
    return pids


def curobo_status(force_ping: bool = False, timeout_s: float = 0.35) -> Dict[str, Any]:
    """Return current cuRobo availability and status dict."""
    if os.environ.get("NERO_NO_CUROBO") == "1":
        return {"ok": False, "available": False, "reason": "NERO_NO_CUROBO=1", "solver": "fast_ik_45seeds"}
    with _LOCK:
        if _STATE["mock_solver"] is not None and not _STATE["vram_yielded"]:
            return {"ok": True, "available": True, "backend": "mock_curobo", "solver": "curobo"}
        if _STATE["vram_yielded"] and not force_ping:
            return {"ok": False, "available": False, "vram_yielded": True,
                    "reason": _STATE["yield_reason"] or "yielded_for_vram", "solver": "fast_ik_45seeds"}
        now = time.monotonic()
        if not force_ping and (now - float(_STATE["last_ping_t"])) < 2.0:
            ok = bool(_STATE["last_ping_ok"])
            return {"ok": ok, "available": ok, "cached": True,
                    "solver": "curobo" if ok else "fast_ik_45seeds", "status": _STATE["last_status"]}
    cli = GPC.GPUPlannerClient(timeout_s=timeout_s)
    st = cli.ping(timeout_s=timeout_s)
    eng = ((st.get("engines") or {}).get("curobo") or {}) if isinstance(st, dict) else {}
    ok = bool(st.get("ok") and (eng.get("initialized") or eng.get("imported") or st.get("available")))
    with _LOCK:
        _STATE["last_ping_t"] = time.monotonic()
        _STATE["last_ping_ok"] = ok
        _STATE["last_status"] = st if isinstance(st, dict) else {}
        if ok:
            _STATE["vram_yielded"] = False
    return {"ok": ok, "available": ok, "solver": "curobo" if ok else "fast_ik_45seeds", "status": st}


def ensure_curobo_server(log: Optional[Callable[[str], None]] = None, wait_s: float = 25.0) -> Dict[str, Any]:
    """Check if cuRobo server is running; if not, start it when the GPU venv & config are installed."""
    log = log or (lambda _s: None)
    if os.environ.get("NERO_NO_CUROBO") == "1":
        log("[INFO] [V4.8.7 CUROBO] NERO_NO_CUROBO=1：使用 FastIK (45 种子) 规划器")
        return {"ok": False, "started": False, "solver": "fast_ik_45seeds", "reason": "disabled"}

    with _LOCK:
        _STATE["vram_yielded"] = False
        _STATE["yield_reason"] = ""

    st = curobo_status(force_ping=True, timeout_s=0.5)
    if st.get("ok"):
        log("[OK] [V4.8.7 CUROBO] cuRobo GPU 规划服务已就绪（默认 IK 规划器：cuRobo，备选：FastIK 45 种子）")
        return {"ok": True, "started": False, "solver": "curobo"}

    gpu_py = os.path.expanduser(os.environ.get("NERO_GPU_PLANNER_PYTHON", DEFAULT_GPU_PY))
    gpu_cfg = os.path.expanduser(os.environ.get("NERO_GPU_CONFIG_DIR", DEFAULT_GPU_CFG))
    yml = os.path.join(gpu_cfg, "nero_curobo.yml")
    if not (os.path.isfile(gpu_py) and os.access(gpu_py, os.X_OK) and os.path.isfile(yml)):
        log(f"[INFO] [V4.8.7 CUROBO] 未检测到 cuRobo 运行环境（{gpu_py}），实机任务将自动使用备选 FastIK（45 种子）")
        return {"ok": False, "started": False, "solver": "fast_ik_45seeds", "reason": "env_missing"}

    log_dir = os.path.join(HERE, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "gpu_server.log")
    cmd = [gpu_py, SERVER_SCRIPT, "--config-dir", gpu_cfg, "--prefer", "curobo", "--prewarm", "curobo"]
    log(f"[INFO] [V4.8.7 CUROBO] 正在启动 cuRobo GPU 规划服务（{gpu_py}）…")
    try:
        fh = open(log_path, "ab")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        fh.close()
    except Exception as exc:
        log(f"[WARN] [V4.8.7 CUROBO] 启动 cuRobo 失败（{exc}）→ 自动切换为备选 FastIK（45 种子）")
        return {"ok": False, "started": False, "solver": "fast_ik_45seeds", "reason": str(exc)}

    with _LOCK:
        _STATE["proc"] = proc
        _STATE["pid"] = proc.pid

    t_deadline = time.monotonic() + float(wait_s)
    while time.monotonic() < t_deadline:
        if proc.poll() is not None:
            break
        s = curobo_status(force_ping=True, timeout_s=0.4)
        if s.get("ok"):
            log(f"[OK] [V4.8.7 CUROBO] cuRobo GPU 规划服务启动完成（PID={proc.pid}，默认 IK：cuRobo，备选：FastIK 45 种子）")
            return {"ok": True, "started": True, "pid": proc.pid, "solver": "curobo"}
        time.sleep(0.25)

    log("[WARN] [V4.8.7 CUROBO] cuRobo 服务未在限时内就绪（详见 logs/gpu_server.log）→ 当前使用备选 FastIK（45 种子）")
    return {"ok": False, "started": False, "pid": proc.pid, "solver": "fast_ik_45seeds", "reason": "timeout"}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{int(pid)}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except Exception:
        return True


def stop_curobo_server(log: Optional[Callable[[str], None]] = None, reason: str = "释放 GPU 显存") -> Dict[str, Any]:
    """Stop any running cuRobo GPU server daemon to free ~1.5-2.2 GB of GPU VRAM for SmolVLA / training."""
    log = log or (lambda _s: None)
    with _LOCK:
        _STATE["vram_yielded"] = True
        _STATE["yield_reason"] = reason
        _STATE["last_ping_ok"] = False
        _STATE["last_ping_t"] = time.monotonic()
        proc = _STATE.get("proc")
        _STATE["proc"] = None
        _STATE["pid"] = None

    pids = set(find_server_pids())
    if proc is not None and proc.poll() is None:
        pids.add(int(proc.pid))

    if not pids:
        return {"ok": True, "stopped": [], "reason": reason}

    log(f"[INFO] [V4.8.7 VRAM] 正在关闭 cuRobo 规划服务（PID={sorted(pids)}，原因：{reason}）以释放 GPU 显存…")
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGKILL):
        alive = []
        for pid in list(pids):
            try:
                os.kill(pid, sig)
                alive.append(pid)
            except ProcessLookupError:
                pids.discard(pid)
            except Exception:
                pass
        if not pids:
            break
        t_end = time.monotonic() + (1.2 if sig != signal.SIGKILL else 0.4)
        while time.monotonic() < t_end and pids:
            for pid in list(pids):
                if not _pid_alive(pid):
                    pids.discard(pid)
            if pids:
                time.sleep(0.08)
        if not pids:
            break

    log(f"[OK] [V4.8.7 VRAM] 已关闭 cuRobo 规划服务并释放显存（原因：{reason}；Agentic 任务期间如需 IK 将自动回退或重启）")
    return {"ok": len(pids) == 0, "stopped": sorted(alive), "left": sorted(pids), "reason": reason}


def solve_pose(fast_arm, q_seed: Iterable[float], xyz: Iterable[float], R_des: np.ndarray,
               tol_m: float = 1.5e-3, tol_deg: float = 2.0, timeout_s: float = 0.65) -> Tuple[Optional[List[float]], str]:
    """Solve 6-DoF IK for (xyz, R_des) in tcp_link frame using cuRobo first, returning (q_7dof, 'curobo') or (None, why)."""
    st = curobo_status(force_ping=False)
    if not st.get("ok"):
        return None, "curobo_unavailable"

    q_start = [float(v) for v in q_seed]
    pos = [float(v) for v in xyz]
    quat_xyzw = mat_to_quat_xyzw(np.asarray(R_des, float))

    with _LOCK:
        mock_fn = _STATE.get("mock_solver")

    if mock_fn is not None:
        try:
            q_out = mock_fn(q_start, pos, quat_xyzw, np.asarray(R_des, float))
            if q_out is None:
                return None, "mock_no_solution"
            q_sol = [float(v) for v in q_out]
        except Exception as exc:
            return None, f"mock_error:{exc}"
    else:
        cli = GPC.GPUPlannerClient(timeout_s=timeout_s)
        try:
            res = cli.plan_pose(
                engine="curobo",
                joint_names=JOINT_NAMES,
                q_start=q_start,
                position=pos,
                quat_xyzw=quat_xyzw,
                max_attempts=3,
                timeout_s=timeout_s,
            )
        except Exception as exc:
            with _LOCK:
                _STATE["last_ping_ok"] = False
                _STATE["last_ping_t"] = time.monotonic()
            return None, f"rpc_error:{exc}"
        if not res.get("ok") or not res.get("positions"):
            return None, str(res.get("error") or "plan_pose_failed")
        q_sol = [float(v) for v in res["positions"][-1]]

    if len(q_sol) != 7 or not all(math.isfinite(v) for v in q_sol):
        return None, "invalid_q"

    # Verify accuracy & joint limits using FastArm forward kinematics
    if fast_arm is not None:
        q_arr = np.asarray(q_sol, float)
        if np.any(q_arr < fast_arm.lo - 1e-3) or np.any(q_arr > fast_arm.hi + 1e-3):
            return None, "out_of_limits"
        q_arr = np.clip(q_arr, fast_arm.lo + 1e-4, fast_arm.hi - 1e-4)
        T = fast_arm.fk(q_arr)
        pe = float(np.linalg.norm(np.asarray(pos, float) - T[:3, 3]))
        Re = np.asarray(R_des, float) @ T[:3, :3].T
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(Re)) - 1.0) / 2.0))))
        if pe > tol_m or ang > tol_deg:
            # Polish with a few local analytic-Jacobian iterations seeded from cuRobo's solution
            q_pol, pe2, re2 = fast_arm.ik([q_arr.tolist()], pos, R_des, iters=20, tol_m=5e-4, tol_deg=1.0)
            if q_pol is not None:
                return [float(v) for v in q_pol], "curobo"
            return None, f"pose_err({pe*1000:.1f}mm,{ang:.1f}deg)"
        q_sol = [float(v) for v in q_arr]
    return q_sol, "curobo"


def solve_chain(fast_arm, check_joint_path_fn, waypoints: List[Tuple[str, Any, Any]], q_seed: Iterable[float],
                max_j1_deg: float = 120.0) -> Tuple[Optional[List[List[float]]], Optional[Dict[str, Any]]]:
    """Solve a multi-waypoint chain [(name, p, R), ...] with cuRobo first.
    Returns (qs, chk_with_solver_curobo) or (None, None) so caller can fall back to 45-seed FastIK."""
    st = curobo_status(force_ping=False)
    if not st.get("ok") or not waypoints:
        return None, None
    q_cur = [float(v) for v in q_seed]
    q0 = list(q_cur)
    qs: List[List[float]] = []
    for i, (_name, p, R) in enumerate(waypoints):
        if i == 0:
            q_i, _why = solve_pose(fast_arm, q_cur, p, R)
        else:
            # For subsequent closely-spaced waypoints (e.g. pre -> grasp -> lift), try local branch-preserving step
            # from cuRobo's first waypoint first, and query cuRobo if local step fails
            q_i = None
            if fast_arm is not None:
                r_loc = fast_arm.ik([q_cur], p, R, iters=30)
                if r_loc and r_loc[0] is not None:
                    q_i = list(r_loc[0])
            if q_i is None:
                q_i, _why = solve_pose(fast_arm, q_cur, p, R)
        if q_i is None:
            return None, None
        if i == 0 and abs(float(q_i[0]) - float(q0[0])) > math.radians(max_j1_deg):
            return None, None
        qs.append(q_i)
        q_cur = q_i
    chk = check_joint_path_fn(qs, jump_deg=1e9)
    if not chk.get("ok"):
        return None, None
    return qs, dict(chk, ik_solver="curobo")
