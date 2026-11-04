#!/usr/bin/env python3
"""NERO V3.65 REAL CUP TASK - the simulation's stack_cups_so101 skills and parameters on the real NERO.

One brain, two bodies: the SAME CupSkills class (NERO_V362_Cup_Skills: observe_scene / plan_cup_stack / pick_cup /
place_cup_into / verify_stack / return_to_safe_pose), the same skill knobs (active cup skill, CS by default), the
same SO-101 gripper profile (NERO_V364: grasp frame with the 7 mm ring and the 45 deg roll, fixed jaw outside, 2 mm
clearance, descent along the wall, low release + sidestep) and the same LLM tool contract.  Only the body differs:

  RealIO (this module)           simulation (BackendIO / TwinIO)
  -----------------------------  -------------------------------------------------------------
  eyes   fixed D435i (Camera tab stream + hand-eye calibration T_base_cam) -> V3.66: the Camera tab's YOLOE
         object pipeline + cup axis / silhouette height (NERO_V366_Real_Cup_Yolo; no colour fallback)
  arm    joint paths through MoveIt /execute_trajectory (NERO_V313 joint path, fence-checked):
         straight segments = strict-continuity IK points (like the twin), joint moves = linear joint interpolation
  hand   SO-101 jaw on an STS3215 bus servo (NERO_V365_STS3215_Gripper): position control + torque limit clamp

Run modes (safety ladder, AGENTS.md section 8): 1 plan preview (camera + plan, the arm does not move),
2 hover dry run (pregrasp above the source cup -> up -> above the target, gripper never closes, no descent),
3 execute (the full task; scripted sequence or the LLM agent).  The start posture is the safe pose the task
returns to.  Nothing here writes into the release folder (settings: NERO_CALIB_DIR/real_cup_task.json).
"""
import json
import math
import os
import queue
import threading
import time

import numpy as np

VERSION = "V3.74"
MARKER = "V3.65 NERO REAL CUP TASK"
CALIB_DIR = os.environ.get("NERO_CALIB_DIR") or os.path.expanduser("~/Downloads/cameracalib")
SETTINGS_NAME = "real_cup_task.json"
MODES = (("plan", "① 视觉规划预览（机械臂不动）"), ("hover", "② 悬停预演（不下探、不合夹爪）"),
         ("execute", "③ 实机执行（抓杯 → 放杯）"))
DEFAULT_CFG = {
    "joint_vmax_deg_s": 12.0,       # joint moves (linear joint interpolation)
    "line_vmax_mm_s": 20.0,         # straight segments near the cups (descent / lift / lowering)
    "path_step_mm": 3.0,            # IK point spacing of a straight segment
    "joint_step_deg": 2.0,          # point spacing of a joint move
    "settle_s": 0.4,
    "tcp_tol_mm": 6.0,              # a motion is OK when the measured TCP ends this close to the goal
    "hover_dwell_s": 3.0,
    "clamp_torque_pct": None,       # None = from the skill's clamp force (STS3215 torque limit)
    "lift_settle_s": 0.5,           # real gripper: no CONNECT hold swing -> short settle only
    # V3.67 REAL GRIPPER GEOMETRY + SAFETY (real arm only; the simulation keeps its SO-101 profile)
    "clearance_mm": 8.0,            # fixed jaw -> outer cup wall during the descent (simulation: 2 mm)
    "align_rotate_mm": 60.0,        # arrive this high above the pregrasp, rotate in place to the grasp angle
    "descent_stop_mm": 10.0,        # two-stage descent: stop with the fingertips this far above the rim
    "seat_mm": None, "ring_mm": None, "roll_deg": None,      # None = NERO_V364 MOUNT (estimate)
    "tip_corr_mm": [0.0, 0.0, 0.0],  # fingertip correction in the SO-101 gripper frame (4-pose touch calibration)
    # V3.71 ARRIVE AND SETTLE: a motion ends when the arm is really at the goal and still (not when the trajectory
    # time is over); the V3.70 real log: at 4X the arm was still 80 deg away when the next step started
    "arrive_timeout_min_s": 5.0,    # waiting time for the arrival: max(this, 3 x planned time + 2 s), <= 60 s
    "still_deg": 0.3,               # "still" = every joint moved less than this within still_window_s
    "still_window_s": 0.2,
    "safe_z_m": 0.22,               # V3.71 RECOVER: straight up to this TCP height after a failed motion / fence detour
    "nested_dz_mm": None,           # V3.71 STACK VERIFY: measured base-to-base height of a nested real pair (None = not measured)
}
# V3.69 SPEED MULTIPLIER: 1X = the base speeds of the fields; joint moves scale fully up to JOINT_CAP, straight
# segments near the cups (descent / lift / lowering) only up to LINE_CAP; settle / dwell times shrink
SPEED_CHOICES = ("1X", "2X", "3X", "4X", "5X", "10X")
JOINT_CAP_DEG_S = 72.0
LINE_CAP_MM_S = 60.0


def apply_speed(cfg, mult):
    """V3.69 SPEED MULTIPLIER -> new motion cfg (joint / line speeds, settle, hover dwell) and a description"""
    k = max(1.0, float(mult))
    c = dict(cfg)
    c["joint_vmax_deg_s"] = round(min(JOINT_CAP_DEG_S, float(cfg["joint_vmax_deg_s"]) * k), 1)
    c["line_vmax_mm_s"] = round(min(LINE_CAP_MM_S, float(cfg["line_vmax_mm_s"]) * k), 1)
    c["settle_s"] = round(max(0.15, float(cfg.get("settle_s", DEFAULT_CFG["settle_s"])) / k), 2)
    c["hover_dwell_s"] = round(max(1.0, float(cfg.get("hover_dwell_s", DEFAULT_CFG["hover_dwell_s"])) / k), 1)
    c["speed_mult"] = k
    capped = [n for n, v, b in (("关节", c["joint_vmax_deg_s"], float(cfg["joint_vmax_deg_s"]) * k),
                                ("直线", c["line_vmax_mm_s"], float(cfg["line_vmax_mm_s"]) * k)) if v < b - 1e-6]
    desc = (f"{k:g}X：关节 {c['joint_vmax_deg_s']} °/s · 直线 {c['line_vmax_mm_s']} mm/s · 停稳 {c['settle_s']} s"
            + (f"（{'/'.join(capped)}已到上限）" if capped else ""))
    return c, desc


MOUNT_LIMITS = {"seat_mm": (20.0, 60.0), "ring_mm": (0.0, 30.0), "roll_deg": (-180.0, 180.0)}
EXAMPLES = (("cup_1 → cup_2", "Stack cup_1 into cup_2"), ("cup_2 → cup_1", "Put cup_2 into cup_1"))


def load_settings(path=None):
    p = path or os.path.join(CALIB_DIR, SETTINGS_NAME)
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


MOUNT_REV = "V3.75"


def migrate_geometry_v375(geo, log=None):
    """V3.75 NEW ADAPTER RING: settings saved before V3.75 (no "mount_rev") describe the OLD ring with its 45 deg
    offset.  The new ring has none -> roll = old roll + 45 deg (old default -45 -> 0); the fingertip correction was
    measured on the old ring -> reset to 0 (old value kept under "pre_v375"); seat / ring thickness are kept (the
    ring thickness should be re-measured).  -> (new geometry dict, list of change lines)"""
    g = dict(geo or {})
    if not g or g.get("mount_rev") == MOUNT_REV:
        return g, []
    lines = []
    old = {k: g.get(k) for k in ("roll_deg", "tip_corr_mm", "ring_mm", "seat_mm")}
    if g.get("roll_deg") is not None:
        r = float(g["roll_deg"]) + 45.0
        g["roll_deg"] = round((r + 180.0) % 360.0 - 180.0, 2)
        lines.append(f"偏转 {float(old['roll_deg']):+.1f}° → {g['roll_deg']:+.1f}°（新转接环去掉了 45° 偏转）")
    tc = [float(v) for v in (g.get("tip_corr_mm") or [0.0, 0.0, 0.0])]
    if any(abs(v) > 1e-9 for v in tc):
        g["tip_corr_mm"] = [0.0, 0.0, 0.0]
        lines.append(f"指尖修正 {tc} mm 清零（是旧转接环上测的，请重新做 4 姿态碰点）")
    g["pre_v375"] = old
    g["mount_rev"] = MOUNT_REV
    if log and lines:
        log("[INFO] [V3.75 REAL] 转接环已换新设计：" + "；".join(lines) + "；转接环厚度仍按 "
            f"{old['ring_mm'] if old['ring_mm'] is not None else 7.0} mm，请实测后填写。")
    return g, lines


def save_settings(d, path=None):
    p = path or os.path.join(CALIB_DIR, SETTINGS_NAME)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def yolo_device(log=None):
    """V3.69 YOLOE DEVICE: 'cuda:0' when torch sees CUDA (forced - ultralytics was observed on the CPU: ~1 s per
    frame), else None (CPU) with the reason in the log"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            if log:
                log(f"[OK] [V3.69 REAL] YOLOE 用 GPU：cuda:0（{name}）")
            return "cuda:0"
        why = f"torch {torch.__version__} 看不到 CUDA（torch.cuda.is_available() = False）" + gpu_hint(torch)
    except Exception as exc:
        why = f"torch 不可用：{exc}"
    if log:
        log(f"[WARN] [V3.69 REAL] YOLOE 只能用 CPU（每帧约 1 s）：{why}")
    return None


def gpu_hint(torch=None):
    """V3.71 GPU REASON: what usually hides the GPU - CUDA_VISIBLE_DEVICES, no NVIDIA driver (nvidia-smi), driver
    too old for the torch CUDA build"""
    import shutil
    import subprocess
    parts = []
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        parts.append(f"环境变量 CUDA_VISIBLE_DEVICES='{cvd}'（控制台从 start_nero.sh 启动会清掉它）")
    cu = getattr(getattr(torch, "version", None), "cuda", None) if torch is not None else None
    if cu:
        parts.append(f"torch 编译的 CUDA {cu}")
    smi = shutil.which("nvidia-smi")
    if not smi:
        parts.append("找不到 nvidia-smi（NVIDIA 驱动没装好或没加载）")
    else:
        try:
            r = subprocess.run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True,
                               text=True, timeout=3.0)
            out = (r.stdout or r.stderr or "").strip().splitlines()
            parts.append(f"nvidia-smi：{out[0][:120]}" if out else f"nvidia-smi 无输出（rc={r.returncode}）")
        except Exception as exc:
            parts.append(f"nvidia-smi 失败：{str(exc)[:80]}")
    return ("；" + "；".join(parts) + "。下一步：在同一终端跑 nvidia-smi 和 python3 -c \"import torch;print(torch.cuda.is_available())\""
            "（驱动没加载就重启 / 检查 prime-select）") if parts else ""


def real_mount(cfg=None):
    """V3.67 REAL GRIPPER GEOMETRY: (T_l7_gripper, T_tcp_grasp, mount dict) of the real SO-101 - the NERO_V364 model
    with the measured seat / ring / roll and a fingertip correction (gripper frame) instead of the estimates"""
    import NERO_V364_SO101_Gripper as G
    c = dict(DEFAULT_CFG, **(cfg or {}))
    seat = G.MOUNT["seat_x_m"] * 1000.0 if c.get("seat_mm") is None else float(c["seat_mm"])
    ring = G.MOUNT["ring_m"] * 1000.0 if c.get("ring_mm") is None else float(c["ring_mm"])
    roll = G.MOUNT["roll_deg"] if c.get("roll_deg") is None else float(c["roll_deg"])
    R_nom = G._rx(-G.MOUNT["roll_deg"]) @ G.R_l7_gripper()
    T_l7g = np.eye(4)
    T_l7g[:3, :3] = G._rx(roll) @ R_nom
    T_l7g[:3, 3] = [(seat + ring) / 1000.0, 0.0, G.MOUNT["axis_z_m"]]
    T_gg = G.T_gripper_grasp().copy()
    tc = [float(v) for v in (c.get("tip_corr_mm") or [0.0, 0.0, 0.0])]
    T_gg[:3, 3] = T_gg[:3, 3] + np.asarray(tc) / 1000.0
    T_tg = np.linalg.inv(G.T_L7_TCP) @ T_l7g @ T_gg
    return T_l7g, T_tg, {"seat_mm": round(seat, 2), "ring_mm": round(ring, 2), "roll_deg": round(roll, 2),
                         "tip_corr_mm": [round(v, 2) for v in tc]}


def model_fixed_tip_l7(cfg=None, tip_corr=True):
    """fixed-jaw fingertip (pad end-face centre) in link7 for the current mount (for the 4-pose touch calibration)"""
    import NERO_V364_SO101_Gripper as G
    c = dict(DEFAULT_CFG, **(cfg or {}))
    if not tip_corr:
        c["tip_corr_mm"] = [0.0, 0.0, 0.0]
    T_l7g, _T, _m = real_mount(c)
    tip = np.array([G.PAD_FIXED_CENTRE[0], 0.0, G.PAD_FIXED_CENTRE[2] - G.PAD_HALF[2]])
    tip = tip + np.asarray(c.get("tip_corr_mm") or [0, 0, 0], float) / 1000.0
    return (T_l7g[:3, :3] @ tip + T_l7g[:3, 3]), T_l7g


def tip_corr_from_touch(tcp_in_flange_m, cfg=None, max_mm=25.0):
    """V3.67: the 4-pose touch calibration (Camera tab, done with the FIXED-jaw fingertip) gives the real tip in link7;
    the difference to the model tip (without correction), in the gripper frame = the fingertip correction"""
    p_model, T_l7g = model_fixed_tip_l7(cfg, tip_corr=False)
    d = T_l7g[:3, :3].T @ (np.asarray(tcp_in_flange_m, float) - p_model) * 1000.0
    if float(np.linalg.norm(d)) > max_mm:
        return None, [round(float(v), 1) for v in d]
    return [round(float(v), 2) for v in d], [round(float(v), 1) for v in d]


def real_profile(CK, lift_settle_s=0.5, cfg=None):
    """the simulation's SO-101 profile on the real gripper: short settle (no CONNECT hold), and V3.67 real geometry /
    safety: measured mount, clearance (default 8 mm), align-rotate-descend, two-stage descent, wide plan search"""
    c = dict(DEFAULT_CFG, **{k: v for k, v in (cfg or {}).items() if v is not None})
    p = CK.so101_profile()
    _T_l7g, T_tg, m = real_mount(c)
    return CK.GripperProfile("so101", T_tg, p.open_mm,
                             f"SO-101 gripper on a {m['ring_mm']:.0f} mm ring, roll {m['roll_deg']:+.0f} deg · real STS3215",
                             approach_clearance_m=float(c["clearance_mm"]) / 1000.0,
                             approach_along_wall=p.approach_along_wall, flips=p.flips,
                             release_gap_max_mm=p.release_gap_max_mm, lift_settle_s=lift_settle_s,
                             release_tilt_max_deg=p.release_tilt_max_deg,
                             align_rotate_m=float(c["align_rotate_mm"]) / 1000.0,
                             descent_stop_above_rim_m=(None if c.get("descent_stop_mm") in (None, "")
                                                       else float(c["descent_stop_mm"]) / 1000.0),
                             wide_search_below_deg=8.0,
                             # V3.68: lenient pick (clamp + lift, then judge), fast plan (30 deg wide step, 10 s
                             # budget, start from the last good rim angles)
                             lenient_pick=True, wide_step_deg=30, plan_time_budget_s=15.0, remember_angles=True,
                             # V3.69: clear view before verifying, the camera decides whether the cup is held
                             clear_view_before_verify=True, camera_decides_grip=True,
                             # V3.71: re-check the held cup above the target, stack verify by cluster, perception
                             # retry, recover after a failed motion
                             recheck_held_before_place=True, stack_by_cluster=True, perception_retries=2,
                             recover_after_motion_failure=True,
                             nested_dz_real_mm=(None if c.get("nested_dz_mm") in (None, "") else float(c["nested_dz_mm"])))


def time_joint_path(path, vmax_joint_rad, tcp_pts=None, vmax_lin=None, ramp_frac=0.2, ramp_gain=1.5):
    """time stamps (from 0) for a joint path: each segment as slow as its largest joint step / TCP step needs,
    softer start and stop (the first / last ramp_frac of the path up to (1 + ramp_gain) x slower)"""
    P = np.asarray(path, float)
    n = len(P)
    d = []
    for i in range(1, n):
        dt = float(np.max(np.abs(P[i] - P[i - 1]))) / max(vmax_joint_rad, 1e-6)
        if tcp_pts is not None and vmax_lin:
            dt = max(dt, float(np.linalg.norm(np.asarray(tcp_pts[i]) - np.asarray(tcp_pts[i - 1]))) / vmax_lin)
        d.append(max(dt, 0.02))
    tot = sum(d) or 1.0
    t, acc, out = 0.0, 0.0, [0.0]
    for di in d:
        s = (acc + di / 2.0) / tot
        f = 1.0 + ramp_gain * max(0.0, 1.0 - s / ramp_frac, 1.0 - (1.0 - s) / ramp_frac)
        acc += di
        t += di * f
        out.append(round(t, 4))
    return out


class StopRequested(Exception):
    pass


# ================================================================================================ real IO
class RealIO:
    """CupSkills IO on the real NERO.  arm: object with q() -> 7 joint angles (rad) or None, model (joint names /
    limits), exec_path(path, times, label) -> result dict; gripper: SO101RealGripper (connected) or None;
    camera_fn() -> dict(rgb, depth, K, T_base_cam, z_table, workspace)."""
    name = "real"

    def __init__(self, arm, gripper, camera_fn, log=print, cfg=None, motion_allowed=True):
        import NERO_V362_Cup_Skills as CK
        self.CK = CK
        self.arm, self.grip, self.camera_fn, self.log = arm, gripper, camera_fn, log
        self.cfg = dict(DEFAULT_CFG, **{k: v for k, v in (cfg or {}).items() if v is not None})
        self.fast = CK.FastArm()
        self.stop_ev = threading.Event()
        self.motion_allowed = bool(motion_allowed)
        self.home_q = None
        self.prof = real_profile(CK, float(self.cfg["lift_settle_s"]), self.cfg)
        self.T_l7g = real_mount(self.cfg)[0]
        self.motion_log = []
        self.perc_times = []                           # V3.68: perception durations (task summary)
        self.joint_cap_deg_s = None                    # V3.71 SPEED CAP: real joint speed learned from slow arrivals
        self.on_perception = None                      # V3.71 LIVE OVERLAY: called with every perception result
        self.vision = None                             # V3.74 LLM VISION REVIEW (NERO_V374_Vision_Review.VisionReviewer)
        self.table_offset_mm = None                    # V3.74 TABLE OFFSET CHECK: fitted table - fence z_table
        self._table_warned = False

    # -- basics
    def emit(self, s):
        self.log(s)

    def check_stop(self):
        if self.stop_ev.is_set():
            raise StopRequested("stopped by the user")

    def sleep(self, s):
        t_end = time.time() + max(0.0, float(s))
        while time.time() < t_end:
            if self.stop_ev.is_set():
                return
            time.sleep(min(0.05, max(0.0, t_end - time.time())))

    def pose_mode(self):
        return "camera"                         # the real arm has no simulator truth

    def gripper_profile(self):
        return self.prof

    def current_q(self):
        q = self.arm.q()
        if q is None:
            raise RuntimeError("no joint feedback from the real arm (step 5/6 not green?)")
        return [float(v) for v in q]

    def tcp_pose(self):
        T = self.fast.fk(np.asarray(self.current_q(), float))
        return [float(v) for v in T[:3, 3]], self.CK.R_to_quat_xyzw(T[:3, :3])

    def cup_pose(self, cup):
        raise RuntimeError("the real arm has no ground truth: cup poses come from the fixed D435i")

    def safety_trips(self):
        return 0

    def latch_pending(self):
        return False

    def clear_latch(self, label=""):
        return True, "no safety latch on the real arm (use the hardware e-stop)"

    # -- motion
    def _refuse(self, label, why, refused=True):
        return {"ok": False, "refused": refused, "label": label, "why": why, "error": why}

    def joint_vmax_deg_s(self):
        """V3.71 SPEED CAP: the speed the paths are timed with = speed setting, capped by the learned real speed"""
        v = float(self.cfg["joint_vmax_deg_s"])
        return min(v, self.joint_cap_deg_s) if self.joint_cap_deg_s else v

    def wait_still(self, timeout_s=3.0, goal=None, tol_mm=None):
        """V3.71 ARRIVE AND SETTLE: wait until the joints are still (and, with a goal, the TCP is within tol of it)
        -> (ok, q, why)"""
        win = float(self.cfg["still_window_s"])
        lim = math.radians(float(self.cfg["still_deg"]))
        tol = float(self.cfg["tcp_tol_mm"] if tol_mm is None else tol_mm) / 1000.0
        T_goal = None if goal is None else self.fast.fk(np.asarray(goal, float))
        t_end = time.time() + float(timeout_s)
        q_prev, t_prev, q = None, None, None
        why = "no joint feedback"
        while True:
            q = self.arm.q()
            now = time.time()
            if q is not None:
                at = True
                if T_goal is not None:
                    err = float(np.linalg.norm(self.fast.fk(np.asarray(q, float))[:3, 3] - T_goal[:3, 3]))
                    at = err <= tol
                    why = f"TCP {err * 1000:.1f} mm from the goal"
                if q_prev is None or now - t_prev >= win:
                    if q_prev is not None:
                        still = max(abs(a_ - b_) for a_, b_ in zip(q, q_prev)) < lim
                        if still and at:
                            return True, q, None
                        if not still:
                            why = (why + " · " if T_goal is not None else "") + "arm still moving"
                    q_prev, t_prev = q, now
            if now >= t_end or self.stop_ev.is_set():
                return False, q, why
            time.sleep(0.03)

    def _execute(self, label, path, tcp_pts, vmax_lin, slow):
        """time + execute a joint path; V3.71: start only with a still arm, then wait until the arm is REALLY at the
        path end and still (timeout from the planned time); a slow arrival teaches the real joint speed cap"""
        if not self.motion_allowed:
            return {"ok": False, "label": label, "error": "MOTION_NOT_ALLOWED in this run mode"}
        if self.stop_ev.is_set():
            return {"ok": False, "label": label, "error": "STOPPED_BY_USER"}
        ok0, q0, why0 = self.wait_still(3.0)
        if not ok0:
            return {"ok": False, "label": label, "error": f"ARM_NOT_STILL before {label}: {why0}"}
        if q0 is not None and max(abs(a_ - b_) for a_, b_ in zip(q0, path[0])) > math.radians(1.0):
            path = [list(q0)] + [list(p_) for p_ in path[1:]]     # start exactly where the arm is
        f = max(1.0, float(slow or 1.0))
        vj = self.joint_vmax_deg_s()
        times = time_joint_path(path, math.radians(vj) / f, tcp_pts, (vmax_lin / f) if vmax_lin else None)
        t0 = time.time()
        r = dict(self.arm.exec_path(path, times, label) or {"ok": False, "error": "no result"})
        rec = {"label": label, "points": len(path), "planned_s": times[-1], "elapsed_s": round(time.time() - t0, 2),
               "ok": bool(r.get("ok")), "error": r.get("error"), "joint_vmax_deg_s": round(vj / f, 1)}
        self.motion_log.append(rec)
        if not r.get("ok"):
            return {"ok": False, "label": label, "error": r.get("error"), "fence_violation": r.get("fence_violation")}
        wait = min(60.0, max(float(self.cfg["arrive_timeout_min_s"]), 3.0 * times[-1] + 2.0) - (time.time() - t0))
        arrived, q, why = self.wait_still(max(0.5, wait), goal=path[-1])
        self.sleep(float(self.cfg["settle_s"]))
        t_arr = time.time() - t0
        rec["arrived_s"] = round(t_arr, 2)
        self._learn_speed(label, times[-1], t_arr, vj / f, arrived)
        q = self.arm.q() or q
        if q is None:
            return {"ok": False, "label": label, "error": "no joint feedback after the motion"}
        T_goal = self.fast.fk(np.asarray(path[-1], float))
        T_act = self.fast.fk(np.asarray(q, float))
        err = float(np.linalg.norm(T_act[:3, 3] - T_goal[:3, 3]))
        jerr = max(abs(math.degrees(a - b)) for a, b in zip(q, path[-1]))
        ok = err * 1000.0 <= float(self.cfg["tcp_tol_mm"]) and arrived
        return {"ok": ok, "label": label, "tcp_error_mm": round(err * 1000, 1), "joint_error_deg": round(jerr, 2),
                "kind": r.get("kind", "path"), "arrived_s": round(t_arr, 2), "planned_s": round(times[-1], 2),
                "error": None if ok else (f"ARM_NOT_AT_GOAL: {why} {t_arr:.1f} s after the start (planned "
                                          f"{times[-1]:.1f} s) - the arm did not reach the goal")}

    def _learn_speed(self, label, planned_s, t_arr, v_used, arrived):
        """V3.71 SPEED CAP: the arm needed much longer than planned -> its real joint speed is lower than the speed
        setting (NERO driver speed_percent) -> time the following paths with the measured speed"""
        if planned_s < 1.0 or t_arr <= planned_s * 1.3 + 0.5:
            return
        ratio = t_arr / planned_s
        cap = round(max(2.0, v_used / ratio), 1)
        if self.joint_cap_deg_s is None or cap < self.joint_cap_deg_s - 0.5:
            self.joint_cap_deg_s = cap
            self.emit(f"[WARN] [V3.71 REAL] 机械臂实际比设定慢：{label} 计划 {planned_s:.1f} s，实际 {t_arr:.1f} s"
                      f"{'' if arrived else '（还没到位）'} → 之后的运动按实测关节速度 {cap:.1f} °/s 定时间。"
                      "原因：实机参数 speed_percent 限制了真实速度；想更快请先调高 speed_percent（重启驱动）再用高速度档")

    def move_up_to_safe(self, label, z=None):
        """V3.71 RECOVER: straight up (path-checked) to the safe TCP height; nothing to do when already there"""
        p, q = self.tcp_pose()
        zz = float(self.cfg["safe_z_m"] if z is None else z)
        if p[2] >= zz - 0.005:
            return {"ok": True, "skipped": True}
        return self.move_line(label, [p[0], p[1], zz], q, 0.05)

    def recover_after_failure(self, label):
        """V3.71 RECOVER: after a failed motion - wait for the arm to stop, then straight up to the safe height, so
        the next tool starts from a clean posture (V3.70: the next task started next to the cups and clipped a wall)"""
        ok, _q, why = self.wait_still(8.0)
        if not ok:
            self.emit(f"[WARN] [V3.71 REAL] {label}：机械臂 8 s 内没停稳（{why}）→ 不再自动移动，请检查")
            return {"ok": False, "error": why}
        r = self.move_up_to_safe(f"V3.71 recover after {label}")
        self.emit(f"[{'OK' if r.get('ok') else 'WARN'}] [V3.71 REAL] 运动失败后收尾：已停稳"
                  + ("，已在安全高度" if r.get("skipped") else
                     ("，竖直升到安全高度" if r.get("ok") else f"，竖直上升没执行：{r.get('error') or r.get('why')}")))
        return r

    def cartesian_path(self, xyz, quat):
        """straight TCP segment from the current pose: strict-continuity IK points (like TwinIO) -> (q list, tcp list)"""
        CK = self.CK
        q_now = np.asarray(self.current_q(), float)
        T0 = self.fast.fk(q_now)
        p0, p1 = T0[:3, 3].copy(), np.asarray(xyz, float)
        q0, q1 = np.array(CK.R_to_quat_xyzw(T0[:3, :3])), np.array(quat, float)
        if q0 @ q1 < 0:
            q1 = -q1
        ang = 2 * math.acos(min(1.0, abs(float(q0 @ q1))))
        step = float(self.cfg["path_step_mm"]) / 1000.0
        n = max(2, int(math.ceil(max(np.linalg.norm(p1 - p0) / step, ang / math.radians(1.0)))))
        qs, ps, qp = [list(q_now)], [p0], list(q_now)
        for i in range(1, n + 1):
            u = i / n
            qq = q0 * (1 - u) + q1 * u
            qq /= np.linalg.norm(qq)
            p = p0 + u * (p1 - p0)
            r = self.fast.ik([qp], p, CK.quat_xyzw_to_R(qq), iters=30)
            if r[0] is None:
                return None, None, f"Cartesian corridor only {100.0 * (i - 1) / n:.1f}% (IK continuity broke)"
            qp = list(r[0])
            qs.append(qp)
            ps.append(p)
        return qs, ps, ""

    def move_line(self, label, xyz, quat, speed=0.05, slow=None):
        try:
            qs, ps, why = self.cartesian_path(xyz, quat)
        except Exception as exc:
            return self._refuse(label, f"no joint feedback: {exc}")
        if qs is None:
            return self._refuse(label, why)
        chk = self.CK.check_joint_path(qs)
        if not chk["ok"]:
            return dict(self._refuse(label, chk["why"]), path_check=chk)
        vlin = min(float(speed), float(self.cfg["line_vmax_mm_s"]) / 1000.0)
        r = self._execute(label, qs, ps, vlin, slow)
        r["path_check"] = chk
        r["kind"] = "line"
        return r

    def move_joints(self, label, q, slow=None, speed=None, detour=False):
        try:
            q0 = np.asarray(self.current_q(), float)
        except Exception as exc:
            return {"ok": False, "label": label, "error": str(exc)}
        q1 = np.asarray(q, float)
        chk = self.CK.check_joint_path([list(q0), list(q1)], jump_deg=1e9)
        if not chk["ok"]:
            return {"ok": False, "label": label, "error": f"joint goal rejected: {chk['why']}"}
        r = self._joint_line(label, q0, q1, slow)
        if not r.get("ok") and (r.get("fence_violation") or "FENCE" in str(r.get("error") or "")) and not detour:
            # V3.71 FENCE DETOUR: the straight joint path clips the fence (V3.70: 69 mm < 70 mm from a wall) ->
            # straight up to the safe height first, then the joint path from there; still refused -> the error
            up = self.move_up_to_safe(f"{label} (V3.71 detour up)")
            if up.get("ok") and not up.get("skipped"):
                self.emit(f"[INFO] [V3.71 REAL] {label}：关节直线路径擦到围栏 → 先竖直升到安全高度再过去")
                r2 = self.move_joints(label, q, slow=slow, speed=speed, detour=True)
                r2["detour"] = "up first"
                return r2
        return r

    def _joint_line(self, label, q0, q1, slow):
        n = max(2, int(math.ceil(math.degrees(float(np.max(np.abs(q1 - q0)))) / float(self.cfg["joint_step_deg"]))))
        path = [list(q0 + (q1 - q0) * (i / n)) for i in range(n + 1)]
        r = self._execute(label, path, None, None, slow)
        r["kind"] = "joints"
        return r

    def move_pose(self, label, xyz, quat, speed=0.10, slow=None):
        q = self.CK.ik_from(self.current_q(), xyz, self.CK.quat_xyzw_to_R(quat))
        if q is None:
            return {"ok": False, "label": label, "error": "no IK"}
        return self.move_joints(label, q, slow=slow)

    def go_home(self, label):
        """back to the posture the task started from (known collision-free), first 80 mm up if low"""
        try:
            p, q = self.tcp_pose()
            if p[2] < 0.25:
                self.move_line(f"{label} up", [p[0], p[1], p[2] + 0.08], q, 0.05)
            if self.home_q is not None:
                r = self.move_joints(f"{label} start posture", self.home_q)
                return {"ok": bool(r.get("ok")), "error": r.get("error")}
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:300]}

    def clear_view(self, attempt, label):
        """V3.69 CLEAR VIEW VERIFY: attempt 0 = back to the start posture (out of the fixed camera's view of the cups);
        attempt 1 = additionally 100 mm straight up from there"""
        if attempt == 0:
            r = self.go_home(f"V3.69 clear view for {label}")
        else:
            p, q = self.tcp_pose()
            r = self.move_line(f"V3.69 clear view (higher) for {label}", [p[0], p[1], p[2] + 0.10], q, 0.05)
        self.emit(f"[INFO] [V3.69 REAL] 验证前让开视线（第 {attempt + 1} 次）：{'OK' if r.get('ok', True) else r.get('error')}")
        return r

    # -- gripper (virtual opening mm = simulation backend units, 2000 * q1)
    def gripper_open(self, mm):
        import NERO_V365_STS3215_Gripper as GR
        if not self.motion_allowed or self.grip is None:
            return {"opening_mm": round(float(mm), 1), "skipped": True}
        r = self.grip.open_deg(GR.open_deg_for_virtual_mm(mm))
        a = r.get("open_deg")
        return {"opening_mm": round(GR.virtual_mm_for_open_deg(a), 1) if a is not None else None,
                "gap_mm": r.get("gap_mm"), "ok": r.get("ok"), "error": r.get("error")}

    def clamp_pct(self, force_n):
        import NERO_V365_STS3215_Gripper as GR
        v = self.cfg.get("clamp_torque_pct")
        return float(v) if v not in (None, "") else GR.clamp_torque_pct(force_n)

    def gripper_clamp(self, force_n, settle_s=None):
        import NERO_V365_STS3215_Gripper as GR
        if not self.motion_allowed or self.grip is None:
            return {"contact_left": False, "contact_right": False, "left_N": 0.0, "right_N": 0.0, "hold_active": False,
                    "opening_mm": None, "error": "gripper not available in this run mode"}
        pct = self.clamp_pct(force_n)
        r = self.grip.clamp(pct)
        held = bool(r.get("held"))
        est = round(pct / 100.0 * self.grip.cfg["stall_torque_nm"] / GR.lever_m(), 1) if held else 0.0
        a = r.get("open_deg")
        return {"contact_left": held, "contact_right": held, "left_N": est, "right_N": est, "hold_active": held,
                "opening_mm": round(GR.virtual_mm_for_open_deg(a), 1) if a is not None else None,
                "gap_mm": r.get("gap_mm"), "torque_limit_pct": pct, "load_pct": r.get("load_pct"),
                "current_ma": r.get("current_ma"), "servo_errors": r.get("errors"), "force_note":
                "estimated from the torque limit (STS3215 has no force sensor)"}

    def gripper_state(self):
        import NERO_V365_STS3215_Gripper as GR
        st = self.grip.status() if (self.grip is not None and self.grip.connected) else None
        held = bool(self.grip.held) if self.grip is not None else False
        return {"opening_mm": round(GR.virtual_mm_for_open_deg(st["open_deg"]), 1) if st else None,
                "gap_mm": st["gap_mm"] if st else None, "hold_active": held, "contact_left": held,
                "contact_right": held, "force_mode": held, "connected": st is not None}

    def save_snapshot(self, tag, res=None):
        """V3.67 FAILURE SNAPSHOT of the last camera capture (NERO_CALIB_DIR/real_cup_logs) -> path or None"""
        import NERO_V366_Real_Cup_Yolo as YC
        cam = getattr(self, "last_cam", None)
        if not cam:
            return None
        try:
            path = YC.save_snapshot(cam, res if res is not None else getattr(self, "last_perception", None), tag)
            self.emit(f"[INFO] [V3.67 REAL CUP] 识别快照已保存：{path}")
            return path
        except Exception as exc:
            self.emit(f"[WARN] [V3.67 REAL CUP] 快照没保存：{exc}")
            return None

    # -- eyes
    def gripper_boxes(self, z_shift=0.0, tight=False):
        """oriented box around the SO-101 + ring (robot self-filter of the cup perception); V3.67 tight=True: the
        jaw outline only (down to the fingertips) for the image occlusion test"""
        import NERO_V364_SO101_Gripper as G
        try:
            T_tcp = self.fast.fk(np.asarray(self.current_q(), float))
        except Exception:
            return []
        T_l7 = T_tcp @ np.linalg.inv(self.CK.T_L7_TCP)
        Tg = T_l7 @ self.T_l7g                         # V3.67: the measured mount
        if tight:
            c = Tg[:3, :3] @ np.array([0.005, 0.0, -0.045]) + Tg[:3, 3]
            return [(c, Tg[:3, :3], np.array([0.040, 0.030, 0.060]))]
        c = Tg[:3, :3] @ np.array([0.012, 0.0, -0.05]) + Tg[:3, 3] - np.array([0.0, 0.0, z_shift])
        return [(c, Tg[:3, :3], np.array([0.055, 0.04, 0.075]))]

    def perceive(self, tag):
        """V3.66 REAL CUP YOLOE PERCEPTION: fixed D435i -> the Camera tab's YOLOE object pipeline (same detector, model,
        prompt resolution, depth fusion, locate_objects) + cup axis / silhouette height / kind (NERO_V366).
        No colour fallback: YOLOE unavailable -> the perception raises, the skill aborts (PERCEPTION_UNAVAILABLE)."""
        import NERO_V366_Real_Cup_Yolo as YC
        t_p = time.perf_counter()
        cam = self.camera_fn()
        try:
            res = YC.perceive_cups_yolo(cam, robot_boxes=self.gripper_boxes(),
                                        hull_boxes=self.gripper_boxes(tight=True))
        except YC.YoloCupUnavailable as exc:
            self.emit(f"[FAIL] [V3.66 REAL CUP YOLOE] YOLOE 不可用：{exc} → 任务终止（不改用颜色法）")
            raise
        self.last_perception, self.last_cam = res, cam
        self._table_check(res, cam)
        vis_line = self._vision_review(tag, res, cam)
        # V3.68 FAST PERCEPTION: snapshots only when a run fails (the task record keeps before / after images)
        dt = time.perf_counter() - t_p
        self.perc_times.append(dt)
        self.emit(f"[V3.66 REAL CUP YOLOE] {tag} | table z {res['table_z_m'] * 1000:+.0f} mm | YOLOE {res['n_2d']} 个 | "
                  + YC.summary(res) + f" | {dt:.2f} s（{YC.timing_text(res)}）"
                  + (f" · {res.get('device')}" if res.get("device") else ""))
        if vis_line:
            self.emit(vis_line)
        if self.on_perception is not None:            # V3.71 LIVE OVERLAY: the monitor shows what the task sees
            try:
                self.on_perception(res)
            except Exception:
                pass
        return res


    def _table_check(self, res, cam):
        """V3.74 TABLE OFFSET CHECK: the table plane fitted next to the cups vs the safety fence z_table.  A large
        offset means the hand-eye calibration (or the fence height) is off: grasp heights are off by the same amount"""
        zt = (cam or {}).get("z_table")
        if zt is None or res.get("table_z_m") is None:
            return
        off = (float(res["table_z_m"]) - float(zt)) * 1000.0
        self.table_offset_mm = round(off, 1)
        if abs(off) > 5.0 and not self._table_warned:
            self._table_warned = True
            self.emit(f"[WARN] [V3.74 REAL] 识别到的桌面比安全区设定的桌面 {'低' if off < 0 else '高'} {abs(off):.0f} mm："
                      "主相机手眼标定或安全区桌面高度可能偏了，抓取高度会偏同样多。下一步：Camera调试页复核手眼标定"
                      "（标记边长填实测 38 mm），或在安全区编辑器核对桌面高度" + ("；超过 10 mm 时先别跑 ③" if abs(off) > 10.0 else ""))

    def _vision_review(self, tag, res, cam):
        """V3.74 LLM VISION REVIEW at the key perceptions -> one log line (or None)"""
        if self.vision is None or not res.get("dets") or not self.vision.wanted(tag):
            return None
        import NERO_V374_Vision_Review as VR
        rv = self.vision.review(cam["rgb"], res["dets"], context=str(tag))
        if not rv.get("ok"):
            return None
        notes = VR.apply_review(res, rv)
        items = " ".join(f"{k} {v.get('what')}" + (" 在夹爪里" if v.get("in_gripper") else "")
                         for k, v in sorted((rv.get("items") or {}).items()))
        when = "（沿用上次）" if rv.get("cached") else f"{rv.get('elapsed_s', 0)} s"
        return (f"[V3.74 VISION] {tag} | {rv.get('model')} {when} | {items}"
                + (f" | 和几何不一致：{'；'.join(notes)}" if notes else "")
                + (f" | {rv.get('scene')}" if rv.get("scene") else ""))


# ================================================================================================ run modes
def make_skills(io, knobs=None, skill=None, log=None):
    import NERO_V362_Cup_Skills as CK
    if knobs is None or skill is None:
        knobs, skill = CK.cup_knobs()
    sk = CK.CupSkills(io, knobs, skill)
    io.skills = sk                                     # V3.71 LIVE OVERLAY: the monitor names cups like the skill
    return sk


def _record(sk, io, recorder):
    """V3.68 REAL TASK RECORD: before / after images + step lines for every tool call"""
    if recorder is not None:
        recorder.wrap(sk, perception_fn=lambda: getattr(io, "last_perception", None))
    return sk


def run_plan_preview(io, source, target, knobs=None, skill=None, recorder=None):
    """mode 1: camera + plan only"""
    io.motion_allowed = False
    sk = _record(make_skills(io, knobs, skill), io, recorder)
    out = [sk.run_tool("observe_scene", {})]
    if out[-1].get("ok"):
        out.append(sk.run_tool("plan_cup_stack", {"source_cup": source, "target_cup": target}))
    return {"ok": all(r.get("ok") for r in out), "results": out, "skills": sk}


def _hover_pregrasp(sk, io, source, W, Q):
    """V3.67 ALIGN-ROTATE-DESCEND in the hover dry run too (the same CupSkills step as pick_cup): above the cup with
    the current wrist yaw -> rotate in place to the grasp angle -> down to the pregrasp"""
    if sk.prof.align_rotate_m <= 0:
        return io.move_joints(f"V3.65 hover {source}", Q["pregrasp"])
    try:
        info = sk._align_rotate(source, W["pregrasp"], Q["pregrasp"])
    except sk._Abort as e:
        return {"ok": False, "error": f"{e.reason}: {json.dumps(e.info, ensure_ascii=False)[:200]}"}
    p = np.asarray(io.tcp_pose()[0], float)
    return {"ok": True, "tcp_error_mm": round(float(np.linalg.norm(p - W["pregrasp"][0])) * 1000.0, 1),
            "align": info}


def run_hover(io, source, target, knobs=None, skill=None, recorder=None):
    """mode 2: pregrasp hover above the source rim (jaws ~5 cm above the rim, gripper only opens) -> straight up to
    the lift height -> rotate -> above the target (the carry pose, without a cup) -> back to the start posture"""
    CK = io.CK
    sk = _record(make_skills(io, knobs, skill), io, recorder)
    io.home_q = io.current_q()
    res = [sk.run_tool("observe_scene", {})]
    if not res[-1].get("ok"):
        return {"ok": False, "results": res, "skills": sk}
    res.append(sk.run_tool("plan_cup_stack", {"source_cup": source, "target_cup": target}))
    if not res[-1].get("ok") or not sk.plan:
        return {"ok": False, "results": res, "skills": sk}
    Q, W = sk.plan["_q"], sk.plan["_wps"]
    steps = []
    try:
        if io.grip is not None:
            io.gripper_open(sk.prof.open_mm)
        for label, fn in (
                (f"hover above {source} (pregrasp)", lambda: _hover_pregrasp(sk, io, source, W, Q)),
                ("dwell", lambda: (io.sleep(float(io.cfg["hover_dwell_s"])), {"ok": True})[1]),
                ("check", lambda: (io.perceive(f"hover {source}"), {"ok": True})[1]),
                ("up to the lift height", lambda: io.move_line("V3.65 hover up", W["lift"][0],
                                                               CK.R_to_quat_xyzw(W["lift"][1]), 0.05)),
                ("rotate to the place orientation", lambda: io.move_joints("V3.65 hover rotate", Q["rotate"])),
                (f"above {target}", lambda: io.move_joints(f"V3.65 hover {target}", Q["above_target"])),
                ("dwell", lambda: (io.sleep(float(io.cfg["hover_dwell_s"])), {"ok": True})[1])):
            io.check_stop()
            r = fn()
            steps.append((label, bool(r.get("ok")), r.get("tcp_error_mm"), r.get("error") or r.get("why")))
            io.emit(f"[{'OK' if r.get('ok') else 'FAIL'}] [V3.65 HOVER] {label}"
                    + (f" | TCP error {r.get('tcp_error_mm')} mm" if r.get("tcp_error_mm") is not None else "")
                    + (f" | {r.get('error') or r.get('why')}" if not r.get("ok") else ""))
            if not r.get("ok"):
                break
    except StopRequested:
        steps.append(("stop", False, None, "stopped by the user"))
    finally:
        if not io.stop_ev.is_set():
            h = io.go_home("V3.65 hover back")
            steps.append(("back to the start posture", bool(h.get("ok")), None, h.get("error")))
    ok = all(s[1] for s in steps)
    return {"ok": ok, "results": res, "steps": steps, "skills": sk}


def run_execute(io, source, target, knobs=None, skill=None, agent=None, llm_cfg=None, recorder=None):
    """mode 3: the full task - scripted (observe, plan, pick, place, verify, back) or the LLM agent"""
    import NERO_V362_Cup_Skills as CK
    sk = _record(make_skills(io, knobs, skill), io, recorder)
    io.home_q = io.current_q()
    if agent is None:
        out = sk.scripted_episode(source, target)
        held = sk.held
        if not held:
            out.append(sk.run_tool("return_to_safe_pose", {}))
        ok = any(r.get("tool") == "verify_stack" and r.get("ok") for r in out)
        return {"ok": ok, "results": out, "skills": sk}
    c = dict(llm_cfg or {})
    final = CK.run_cup_task(agent, sk, c.get("task") or f"Stack {source} into {target}", c.get("model"),
                            c.get("api_key", ""), c.get("reasoning_effort", "low"),
                            max(10, min(60, int(c.get("max_tool_steps", 30)))), emit=io.emit, dry_run=False)
    return {"ok": sk.stage == "VERIFIED", "final": final, "skills": sk}


def parse_task_cups(task):
    """'Stack cup_1 into cup_2' / 'put cup_2 into cup_1' / '把 cup_2 放进 cup_1' -> (source, target)"""
    import re
    t = str(task or "").lower().replace(" ", "")
    m = re.findall(r"cup_?([12])", t)
    if len(m) >= 2 and m[0] != m[1]:
        return f"cup_{m[0]}", f"cup_{m[1]}"
    return "cup_1", "cup_2"


# ================================================================================================ arm adapter
class PipelineArm:
    """real arm through NERO_V313 RealPipeline + the joint panel's model"""

    def __init__(self, pipe, model_fn, log=print):
        self.pipe, self.model_fn, self.log = pipe, model_fn, log

    @property
    def model(self):
        return self.model_fn()

    def q(self):
        m = self.model
        if not m:
            return None
        a = self.pipe.actual_arm_deg(m)
        return [math.radians(v) for v in a] if a else None

    def exec_path(self, path, times, label):
        m = self.model
        if not m:
            return {"ok": False, "error": "no robot model (step 5/6)"}
        if hasattr(self.pipe, "move_joint_path") and self.pipe.supports_joint_path():
            r = self.pipe.move_joint_path(m, path, times, label=label)
            return dict(r, kind="path")
        # fallback (old monitor): short MoveIt joint goals along the same path (each <= ~3 deg / ~10 mm)
        self.log(f"[WARN] [V3.65 REAL] {label}: ROS monitor without joint-path support -> short MoveIt joint goals")
        keep = [path[0]]
        for p in path[1:]:
            if max(abs(a - b) for a, b in zip(p, keep[-1])) >= math.radians(3.0):
                keep.append(p)
        if keep[-1] is not path[-1]:
            keep.append(path[-1])
        r = {"ok": True}
        for k, p in enumerate(keep[1:]):
            r = self.pipe.move_and_settle(m, list(p), 0.08, label=f"{label} ({k + 1}/{len(keep) - 1})", settle_s=0.1)
            if not r.get("ok"):
                break
        return dict(r, kind="steps")


def camera_from_tab(cam_tab, yolo=None, n_frames=None, timeout_s=3.0):
    """V3.66 REAL CUP YOLOE PERCEPTION: D435i frames exactly like the Camera tab's object recognition
    (_grab_frames(YP.FUSE_FRAMES): colour of the last frame + all depth frames) + K + T_base_cam + fence (z_table,
    polygon) + calibration note + the Camera tab's YOLOE settings (yolo_cfg_from_tab, read in the Tk thread)"""
    import NERO_V347_Yolo_Perception as YP
    src = getattr(cam_tab, "src", None)
    if src is None or not src.running():
        raise RuntimeError("主相机 D435i 没有连接（本页「连接主相机」或 Camera调试 页）")
    calib = getattr(cam_tab, "result", None) or getattr(cam_tab, "calib", None)
    if not calib or not calib.get("T_base_cam"):
        raise RuntimeError("没有主相机手眼标定（Camera调试 → 标定）")
    if src.K is None:
        raise RuntimeError("相机内参还没读到")
    n = int(n_frames or YP.FUSE_FRAMES)
    t_grab = time.perf_counter()
    if hasattr(cam_tab, "_grab_frames"):
        rgb, depths = cam_tab._grab_frames(n, timeout=timeout_s)
    else:
        rgb, depths, last, t_end = None, [], -1, time.time() + timeout_s
        while len(depths) < n and time.time() < t_end:
            r_, d_, fid = src.frames()
            if r_ is not None and d_ is not None and fid != last:
                rgb, last = r_, fid
                depths.append(d_)
            else:
                time.sleep(0.02)
    if rgb is None or not depths:
        raise RuntimeError("没有拿到相机帧")
    fence = cam_tab._fence() if hasattr(cam_tab, "_fence") else None
    zt = (fence or {}).get("z_table")
    note = cam_tab._calib_note() if hasattr(cam_tab, "_calib_note") else None
    return {"rgb": rgb, "depths": list(depths), "K": dict(src.K), "T_base_cam": calib["T_base_cam"],
            "z_table": None if zt is None else float(zt), "fence_poly": (fence or {}).get("polygon"),
            "calib_note": note, "yolo": yolo, "grab_s": round(time.perf_counter() - t_grab, 3)}


KIND_TEXT = {"occluded": ("被夹爪挡住", "#b0bec5"), "cup": ("单杯", "#00e676"), "stack": ("已叠好", "#00e5ff"), "unknown": ("不像杯子", "#ff9100"),
             "lifted": ("在夹爪里", "#b0bec5"), "outside": ("围栏外", "#ff5252"), "edge": ("被画面切掉", "#ff9100"),
             "no_depth": ("没有深度", "#ff9100"), "rejected": ("剔除", "#9e9e9e"),
             "not_cup": ("看图：不是杯子", "#ff5252")}                          # V3.74 LLM VISION REVIEW
# V3.74 GEOMETRY MEASURES ONLY: a standing cup is "杯子" (single / nested is the vision review's answer)
KIND_TEXT["cup"] = ("杯子", "#00e676")


OVERLAY_OLD_S = 4.0                    # V3.71 LIVE OVERLAY: contours older than this are marked old (dashed)


def overlay_age_s(overlay, now=None):
    """V3.71 LIVE OVERLAY: seconds since the perception shown on the monitor (None = never)"""
    t = (overlay or {}).get("t")
    return None if t is None else max(0.0, (time.time() if now is None else now) - float(t))


def overlay_dets(res):
    """V3.66 REAL CUP YOLOE PERCEPTION: YOLOE contours + labels (image pixels) for the main monitor"""
    out = []
    for e in (res or {}).get("dets") or []:
        if not e.get("contour"):
            continue
        txt, col = KIND_TEXT.get(e.get("kind"), (str(e.get("kind")), "#ff9100"))
        h = e.get("height_mm")
        top = min(e["contour"], key=lambda p: p[1])
        out.append({"contour": e["contour"], "color": col, "anchor": top,
                    "label": f"{e.get('det_id', '')} {txt}" + (f" {h:.0f}mm" if h is not None else "")
                             + f" {e.get('conf', 0):.2f}"})
    return out


# ================================================================================================ GUI tab
class RealCupTaskTabUI:
    """「NERO实机任务」 tab (V3.65): live fixed / wrist camera monitors + SO-101 gripper + skill / parameters of the
    stack_cups_so101 simulation + LLM agent + run modes.  Worker threads never touch Tk (queue + after loop)."""
    DW, DH = 512, 288

    def __init__(self, tk, ttk, parent, pipe, jpanel, cam_tab, log=None, ask=None, visible_fn=lambda: True,
                 settings_path=None, gripper=None):
        import NERO_V365_STS3215_Gripper as GR
        self.tk, self.ttk, self.pipe, self.jpanel, self.cam_tab = tk, ttk, pipe, jpanel, cam_tab
        self.ext_log = log or (lambda s: None)
        self.ask = ask or (lambda t, q: True)
        self.visible_fn = visible_fn
        self.GR = GR
        self.settings_path = settings_path
        self.st = load_settings(settings_path)
        if self.st.get("geometry"):                                   # V3.75 NEW ADAPTER RING
            self.st["geometry"], self._v375_mount_lines = migrate_geometry_v375(self.st["geometry"])
        self.grip = gripper or GR.SO101RealGripper(self.st.get("gripper") or {}, log=self._log)
        self.q = queue.Queue()
        self.worker = None
        self.io = None
        self.agent = None
        self.skills = None
        self.overlay = {"cups": [], "plan": None, "dets": []}
        self.knobs, self.skill = None, None
        self.photo = self.photo_w = None
        self._shown = (-1, -1)
        self._build(parent)
        self.after_id = None
        if getattr(self, "_v375_mount_lines", None):                  # V3.75 NEW ADAPTER RING (once, at start)
            self._log("[INFO] [V3.75 REAL] 转接环已换新设计（无 45° 偏转，动指在上往上开）：" + "；".join(self._v375_mount_lines)
                      + "。转接环厚度仍按旧值，请实测后在“实机夹爪几何”里填写并保存。")
        self._schedule()

    # ---------------------------------------------------------------- helpers
    def _log(self, s):
        self.q.put(("log", str(s)))
        try:
            self.ext_log(str(s))
        except Exception:
            pass

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def _ui(self, fn):
        self.q.put(("fn", fn))

    # ---------------------------------------------------------------- build
    def _build(self, parent):
        tk, ttk = self.tk, self.ttk
        import NERO_V314_Joint_Control as JC
        self.v_status = tk.StringVar(value="")
        JC.fixed_label(tk, parent, 2, width=1180, textvariable=self.v_status, font=("TkDefaultFont", 9, "bold"),
                       wraplength=1170, pack={"fill": "x", "pady": (0, 4)})
        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))

        # -- monitors
        ttk.Label(left, text="主相机实时监控（固定 D435i · 识别到的杯子 / 抓取点）",
                  font=("TkDefaultFont", 9, "bold")).pack(anchor="w")
        self.cv = tk.Canvas(left, width=self.DW, height=self.DH, bg="#202020", highlightthickness=1,
                            highlightbackground="#999")
        self.cv.pack()
        ttk.Label(left, text="腕部相机实时监控（USB 广角）", font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(4, 0))
        self.cvw = tk.Canvas(left, width=self.DW, height=self.DH, bg="#181818", highlightthickness=1,
                             highlightbackground="#607d8b")
        self.cvw.pack()
        cb = ttk.Frame(left)
        cb.pack(fill="x", pady=2)
        ttk.Button(cb, text="▶ 连接主相机", command=self.on_cam_connect).pack(side="left")
        ttk.Button(cb, text="▶ 连接腕部相机", command=self.on_wrist_connect).pack(side="left", padx=4)
        ttk.Button(cb, text="识别杯子（不动）", command=self.on_perceive).pack(side="left", padx=4)
        self.v_video = tk.BooleanVar(value=bool(self.st.get("record_video", False)))      # V3.69 TASK VIDEO
        ttk.Checkbutton(cb, text="任务录像", variable=self.v_video).pack(side="left", padx=4)
        # V3.66 REAL CUP YOLOE PERCEPTION: the Camera tab's YOLOE object recognition (same model / confidence)
        yb = ttk.Frame(left)
        yb.pack(fill="x", pady=(0, 2))
        ttk.Label(yb, text="杯子识别 YOLOE 提示词").pack(side="left")
        self.v_cup_prompt = tk.StringVar(value=str(self.st.get("cup_prompt") or "cup"))
        ttk.Entry(yb, textvariable=self.v_cup_prompt, width=14).pack(side="left", padx=2)
        ttk.Label(yb, text="（模型 / 置信度 = Camera调试 → 视觉定位）").pack(side="left")
        self.v_yolo = tk.StringVar(value="")
        JC.fixed_label(tk, left, 2, width=self.DW, textvariable=self.v_yolo, font=("TkDefaultFont", 8),
                       wraplength=self.DW - 6, pack={"fill": "x"})

        # -- gripper
        gf = ttk.LabelFrame(right, text="SO-101 夹爪（STS3215 舵机 · 180° = 夹紧，正角度 = 张开）")
        gf.pack(fill="x")
        g = self.grip.cfg
        self.v_port = tk.StringVar(value=str(g["port"]))
        self.v_baud = tk.StringVar(value=str(g["baud"]))
        self.v_sid = tk.StringVar(value=str(g["servo_id"]))
        self.v_sign = tk.StringVar(value=str(g["open_sign"]))
        r0 = ttk.Frame(gf)
        r0.pack(fill="x", padx=4, pady=2)
        ttk.Label(r0, text="串口").pack(side="left")
        self.port_box = ttk.Combobox(r0, textvariable=self.v_port, values=self.GR.STSBus.list_ports() + [self.GR.SIM_PORT],
                                     width=16)
        self.port_box.pack(side="left", padx=2)
        ttk.Button(r0, text="刷新", command=self.on_ports).pack(side="left")
        ttk.Label(r0, text=" 波特率").pack(side="left")
        ttk.Combobox(r0, textvariable=self.v_baud, values=("1000000", "500000", "115200"), width=8,
                     state="readonly").pack(side="left")
        ttk.Label(r0, text=" ID").pack(side="left")
        ttk.Spinbox(r0, from_=0, to=253, textvariable=self.v_sid, width=4).pack(side="left")
        ttk.Label(r0, text=" 张开方向").pack(side="left")
        ttk.Combobox(r0, textvariable=self.v_sign, values=("1", "-1"), width=3, state="readonly").pack(side="left")
        ttk.Button(r0, text="连接", command=self.on_grip_connect).pack(side="left", padx=(6, 2))
        ttk.Button(r0, text="断开", command=self.on_grip_disconnect).pack(side="left")
        r1 = ttk.Frame(gf)
        r1.pack(fill="x", padx=4, pady=2)
        ttk.Button(r1, text="张开 10°（方向检查）", command=lambda: self.on_grip("deg", 10.0)).pack(side="left")
        ttk.Button(r1, text="张开 32 mm（预抓取）", command=lambda: self.on_grip("gap", 32.0)).pack(side="left", padx=2)
        ttk.Button(r1, text="夹紧测试", command=lambda: self.on_grip("clamp", None)).pack(side="left", padx=2)
        ttk.Button(r1, text="卸力", command=lambda: self.on_grip("relax", None)).pack(side="left", padx=2)
        ttk.Button(r1, text="空夹标定（夹爪里不要有东西）", command=lambda: self.on_grip("touch", None)).pack(side="left", padx=2)
        self.v_grip = tk.StringVar(value="夹爪未连接")
        JC.fixed_label(tk, gf, 2, width=640, textvariable=self.v_grip, font=("TkFixedFont", 9), wraplength=630,
                       pack={"fill": "x", "padx": 4})

        # -- skill + parameters
        sf = ttk.LabelFrame(right, text="技能与参数（= 仿真「叠杯子 · SO-101 夹爪」场景）")
        sf.pack(fill="x", pady=(6, 0))
        self.v_skill = tk.StringVar(value="")
        JC.fixed_label(tk, sf, 4, width=640, textvariable=self.v_skill, font=("TkFixedFont", 9), wraplength=630,
                       pack={"fill": "x", "padx": 4})
        r2 = ttk.Frame(sf)
        r2.pack(fill="x", padx=4, pady=2)
        cfg = dict(DEFAULT_CFG, **(self.st.get("motion") or {}))
        self.v_jv = tk.StringVar(value=str(cfg["joint_vmax_deg_s"]))
        self.v_lv = tk.StringVar(value=str(cfg["line_vmax_mm_s"]))
        self.v_tq = tk.StringVar(value="" if cfg.get("clamp_torque_pct") in (None, "") else str(cfg["clamp_torque_pct"]))
        for t_, v_, w_ in (("关节速度 °/s", self.v_jv, 5), (" 直线速度 mm/s", self.v_lv, 5),
                           (" 夹紧扭矩 %（空 = 按技能夹紧力）", self.v_tq, 5)):
            ttk.Label(r2, text=t_).pack(side="left")
            ttk.Entry(r2, textvariable=v_, width=w_).pack(side="left")
        ttk.Button(r2, text="重新读取技能", command=self.refresh_skill).pack(side="right")

        # -- V3.67 REAL GRIPPER GEOMETRY + SAFETY
        gf2 = ttk.LabelFrame(right, text="实机夹爪几何与下探安全（只影响实机；仿真不变）")
        gf2.pack(fill="x", pady=(6, 0))
        geo = dict(DEFAULT_CFG, **(self.st.get("geometry") or {}))
        import NERO_V364_SO101_Gripper as G_
        self.v_clear = tk.StringVar(value=str(geo["clearance_mm"]))
        self.v_align = tk.StringVar(value=str(geo["align_rotate_mm"]))
        self.v_stop = tk.StringVar(value="" if geo.get("descent_stop_mm") in (None, "") else str(geo["descent_stop_mm"]))
        self.v_seat = tk.StringVar(value=str(geo["seat_mm"] if geo.get("seat_mm") is not None else round(G_.MOUNT["seat_x_m"] * 1000, 1)))
        self.v_ring = tk.StringVar(value=str(geo["ring_mm"] if geo.get("ring_mm") is not None else round(G_.MOUNT["ring_m"] * 1000, 1)))
        self.v_roll = tk.StringVar(value=str(geo["roll_deg"] if geo.get("roll_deg") is not None else G_.MOUNT["roll_deg"]))
        self.tip_corr = [float(v) for v in (geo.get("tip_corr_mm") or [0.0, 0.0, 0.0])]
        # V3.71 STACK VERIFY: measured nested height (total height of the pair - height of one cup), empty = not measured
        self.v_nested = tk.StringVar(value="" if geo.get("nested_dz_mm") in (None, "") else str(geo["nested_dz_mm"]))
        g0 = ttk.Frame(gf2)
        g0.pack(fill="x", padx=4, pady=2)
        for t_, v_ in (("固定指离杯壁 mm", self.v_clear), (" 对准高度 mm", self.v_align),
                       (" 两段下探：指尖停在杯口上方 mm（空 = 不停）", self.v_stop)):
            ttk.Label(g0, text=t_).pack(side="left")
            ttk.Entry(g0, textvariable=v_, width=5).pack(side="left")
        g1 = ttk.Frame(gf2)
        g1.pack(fill="x", padx=4, pady=2)
        for t_, v_ in (("安装座面 mm", self.v_seat), (" 转接环 mm", self.v_ring), (" 偏转 °", self.v_roll)):
            ttk.Label(g1, text=t_).pack(side="left")
            ttk.Entry(g1, textvariable=v_, width=6).pack(side="left")
        ttk.Button(g1, text="读取碰点标定（固定指尖）", command=self.on_tip_from_touch).pack(side="left", padx=(8, 2))
        ttk.Button(g1, text="清除指尖修正", command=self.on_tip_clear).pack(side="left")
        g2 = ttk.Frame(gf2)
        g2.pack(fill="x", padx=4, pady=2)
        ttk.Label(g2, text="实测套叠增量 mm（两只套好总高 − 单杯高；空 = 未测）").pack(side="left")
        ttk.Entry(g2, textvariable=self.v_nested, width=6).pack(side="left")
        # V3.68 FINGERTIP OVERLAY + ROLL SWEEP
        g2 = ttk.Frame(gf2)
        g2.pack(fill="x", padx=4, pady=2)
        self.v_tips = tk.BooleanVar(value=bool(self.st.get("show_tips", True)))
        ttk.Checkbutton(g2, text="主相机画面画出模型指尖（对照真实指头校「偏转 °」）",
                        variable=self.v_tips).pack(side="left")
        ttk.Button(g2, text="偏转角扫描（只算不动）", command=self.on_roll_sweep).pack(side="right")
        self.v_geo = tk.StringVar(value="")
        JC.fixed_label(tk, gf2, 2, width=640, textvariable=self.v_geo, font=("TkFixedFont", 9), wraplength=630,
                       pack={"fill": "x", "padx": 4})

        # -- agent
        af = ttk.LabelFrame(right, text="Agent（与仿真 NERO SimAgent 同一套叠杯子工具契约）")
        af.pack(fill="x", pady=(6, 0))
        try:
            import NERO_V2_7_1_Agent as AG
            provs = {p: (AG.LLM_PROVIDERS.get(p) or {}).get("label", p) for p in ("openai", "deepseek", "kimi")}
            models = list((AG.LLM_PROVIDERS.get("openai") or {}).get("models") or [AG.DEFAULT_MODEL])
        except Exception:
            provs, models = {"openai": "OpenAI"}, ["gpt-6.1-sol"]
        self.provs = provs
        a0 = ttk.Frame(af)
        a0.pack(fill="x", padx=4, pady=2)
        self.v_prov = tk.StringVar(value=provs.get(self.st.get("provider", "openai"), "OpenAI"))
        self.v_model = tk.StringVar(value=self.st.get("model") or models[0])
        self.v_key = tk.StringVar(value="")
        self.v_effort = tk.StringVar(value=self.st.get("effort", "low"))
        self.v_steps = tk.StringVar(value=str(self.st.get("max_steps", 30)))
        ttk.Label(a0, text="Provider").pack(side="left")
        ttk.Combobox(a0, textvariable=self.v_prov, values=list(provs.values()), width=14, state="readonly").pack(side="left")
        ttk.Label(a0, text=" Model").pack(side="left")
        ttk.Combobox(a0, textvariable=self.v_model, values=models, width=18).pack(side="left")
        ttk.Label(a0, text=" Reasoning").pack(side="left")
        ttk.Combobox(a0, textvariable=self.v_effort, values=("none", "low", "medium", "high"), width=7,
                     state="readonly").pack(side="left")
        ttk.Label(a0, text=" Max steps").pack(side="left")
        ttk.Spinbox(a0, from_=10, to=60, textvariable=self.v_steps, width=4).pack(side="left")
        a1 = ttk.Frame(af)
        a1.pack(fill="x", padx=4, pady=2)
        ttk.Label(a1, text="API key（空 = 环境变量）").pack(side="left")
        ttk.Entry(a1, textvariable=self.v_key, show="*", width=40).pack(side="left", padx=2)
        a2 = ttk.Frame(af)
        a2.pack(fill="x", padx=4, pady=2)
        ttk.Label(a2, text="任务").pack(side="left")
        self.v_task = tk.StringVar(value=self.st.get("task") or EXAMPLES[0][1])
        ttk.Entry(a2, textvariable=self.v_task, width=46).pack(side="left", padx=2, fill="x", expand=True)
        for lab, txt in EXAMPLES:
            ttk.Button(a2, text=lab, command=lambda t=txt: self.v_task.set(t)).pack(side="left", padx=1)

        # -- run
        rf = ttk.LabelFrame(right, text="运行档位（先 ① → ② → 再 ③）")
        rf.pack(fill="x", pady=(6, 0))
        r3 = ttk.Frame(rf)
        r3.pack(fill="x", padx=4, pady=2)
        self.v_mode = tk.StringVar(value="plan")
        for k, lab in MODES:
            ttk.Radiobutton(r3, text=lab, value=k, variable=self.v_mode).pack(side="left", padx=(0, 6))
        r4 = ttk.Frame(rf)
        r4.pack(fill="x", padx=4, pady=2)
        self.v_use_llm = tk.BooleanVar(value=bool(self.st.get("use_llm", False)))
        ttk.Checkbutton(r4, text="③ 用 LLM Agent（不勾 = 固定流程：观察→规划→抓→放→验证→回起点）",
                        variable=self.v_use_llm).pack(side="left")
        # V3.74 LLM VISION REVIEW: the selected model looks at the main camera image (all run modes, also 「识别杯子」)
        self.v_vision = tk.BooleanVar(value=bool(self.st.get("vision_review", True)))
        ttk.Checkbutton(r4, text="LLM 看图复核", variable=self.v_vision).pack(side="left", padx=(8, 0))
        ttk.Button(r4, text="■ 停止", command=self.on_stop).pack(side="right")
        ttk.Button(r4, text="▶ 运行", command=self.on_run).pack(side="right", padx=4)
        # V3.69 SPEED MULTIPLIER (left of 「▶ 运行」)
        self.v_speed = tk.StringVar(value=str(self.st.get("speed_mult") or "1X"))
        ttk.Combobox(r4, textvariable=self.v_speed, values=SPEED_CHOICES, width=5, state="readonly").pack(side="right")
        ttk.Label(r4, text="速度").pack(side="right", padx=(8, 2))

        # -- log
        lf = ttk.Frame(right)
        lf.pack(fill="both", expand=True, pady=(6, 0))
        self.txt = tk.Text(lf, height=12, wrap="word", font=("TkFixedFont", 9))
        self.txt.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lf, command=self.txt.yview)
        sb.pack(side="right", fill="y")
        self.txt.configure(yscrollcommand=sb.set)
        bb = ttk.Frame(right)
        bb.pack(fill="x")
        ttk.Button(bb, text="Copy Log", command=self.copy_log).pack(side="right")
        self.refresh_skill()

    # ---------------------------------------------------------------- skill text
    def refresh_skill(self):
        import NERO_V362_Cup_Skills as CK
        try:
            knobs, skill = CK.cup_knobs()
        except Exception as exc:
            self.v_skill.set(f"技能读取失败：{exc}")
            return
        self.knobs, self.skill = knobs, skill
        try:
            geo = self.geometry_cfg() if hasattr(self, "v_clear") else {}
            geo_err = None
        except ValueError as exc:
            geo, geo_err = {}, str(exc)
        p = real_profile(CK, cfg=geo)
        _T, _Tg, m = real_mount(geo)
        tq = self.GR.clamp_torque_pct(knobs["cup_clamp_force_n"], self.grip.cfg)
        if hasattr(self, "v_geo"):
            self.v_geo.set(geo_err or (
                f"安装：座面 {m['seat_mm']:.1f} mm + 转接环 {m['ring_mm']:.1f} mm · 偏转 {m['roll_deg']:+.1f}° · 指尖修正 "
                f"{m['tip_corr_mm']} mm（夹爪坐标系）\n抓取前：先到预抓取点上方 {p.align_rotate_m * 1000:.0f} mm → 原地转腕到抓取角度"
                f"（两指连线沿杯子半径）→ 竖直降到预抓取点 → 沿杯壁下探"
                + (f"，指尖到杯口上方 {p.descent_stop_above_rim_m * 1000:.0f} mm 停一下复查杯子" if p.descent_stop_above_rim_m
                   else "") + f" · 固定指离杯壁 {p.approach_clearance_m * 1000:.0f} mm"))
        self.v_skill.set(
            f"技能 [{skill.get('code')}] {skill.get('short_name')}（{skill.get('source')}）· 捏深 "
            f"{knobs['cup_rim_pinch_depth_mm']:.0f} mm · 预抓取高 {knobs['cup_pregrasp_height_mm']:.0f} mm · 松手间隙 "
            f"{min(knobs['cup_release_gap_mm'], p.release_gap_max_mm or 1e9):.0f} mm · 对中容差 "
            f"{knobs['cup_release_xy_tol_mm']} mm · 夹紧力 {knobs['cup_clamp_force_n']} N（≈ 扭矩 {tq:.0f} %）\n"
            f"夹爪档案 SO-101：转接环 {m['ring_mm']:.0f} mm · 偏转 {m['roll_deg']:+.0f}° · "
            f"固定指在杯外 · 固定指离杯壁 {p.approach_clearance_m * 1000:.1f} mm · 沿杯壁下探 · 松手后外移 8 mm · "
            f"预抓取指尖缝 32 mm（舵机张开 {self.GR.open_deg_for_virtual_mm(p.open_mm):.1f}°）")

    # ---------------------------------------------------------------- cameras
    def on_cam_connect(self):
        try:
            self.cam_tab.on_connect()
        except Exception as exc:
            self._log(f"[FAIL] [V3.65 REAL] 主相机：{exc}")

    def on_wrist_connect(self):
        try:
            self.cam_tab.on_wrist_connect()
        except Exception as exc:
            self._log(f"[FAIL] [V3.65 REAL] 腕部相机：{exc}")

    def _schedule(self):
        try:
            self.after_id = self.cv.after(150, self._tick)
        except Exception:
            self.after_id = None

    def _tick(self):
        try:
            self.poll()
        except Exception as exc:
            if getattr(self, "_last_err", None) != str(exc):
                self._last_err = str(exc)
                self._log(f"[WARN] [V3.65 REAL] 刷新：{exc}")
        self._schedule()

    def poll(self):
        while True:
            try:
                kind, v = self.q.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self.txt.insert("end", v + "\n")
                self.txt.see("end")
            else:
                v()
        self.v_status.set(self.status_text())
        if not self.visible_fn():
            return
        self._draw_main()
        self._draw_wrist()

    def _photo(self, data):
        try:
            return self.tk.PhotoImage(data=data, format="PPM")
        except Exception:
            import NERO_V325_Camera as CAM
            return self.tk.PhotoImage(data=CAM.png_bytes_from_ppm(data), format="PNG")

    def _draw_main(self):
        src = getattr(self.cam_tab, "src", None)
        if src is None or not src.running():
            return
        _r, _d, fid = src.frames()
        if fid == self._shown[0]:
            return
        data = src.display_ppm("color", size=(self.DW, self.DH))
        if data is None:
            return
        self.photo = self._photo(data)
        c = self.cv
        c.delete("all")
        c.create_image(0, 0, image=self.photo, anchor="nw")
        self._shown = (fid, self._shown[1])
        self._draw_overlay(src)

    def _draw_overlay(self, src):
        calib = getattr(self.cam_tab, "result", None) or getattr(self.cam_tab, "calib", None)
        K = src.K
        if not calib or not K:
            return
        T = np.linalg.inv(np.asarray(calib["T_base_cam"], float))
        sx, sy = self.DW / float(K["width"]), self.DH / float(K["height"])

        def proj(p):
            pc = T[:3, :3] @ np.asarray(p, float) + T[:3, 3]
            if pc[2] <= 0.05:
                return None
            return ((K["fx"] * pc[0] / pc[2] + K["ppx"]) * sx, (K["fy"] * pc[1] / pc[2] + K["ppy"]) * sy)
        age = overlay_age_s(self.overlay)
        old = age is not None and age > OVERLAY_OLD_S       # V3.71 LIVE OVERLAY: older contours are drawn dashed
        for d in self.overlay.get("dets") or []:           # V3.66 REAL CUP YOLOE PERCEPTION: YOLOE masks
            pts = []
            for u, v in d["contour"]:
                pts += [u * sx, v * sy]
            if len(pts) >= 6:
                self.cv.create_polygon(*pts, outline=d["color"], fill="", width=1 if old else 2,
                                       dash=(4, 3) if old else None)
            if d.get("anchor"):
                self.cv.create_text(d["anchor"][0] * sx, d["anchor"][1] * sy - 2, text=d["label"], fill=d["color"],
                                    anchor="s", font=("TkDefaultFont", 8, "bold"))
        if age is not None and (self.overlay.get("dets") or self.overlay.get("cups")):
            self.cv.create_text(6, self.DH - 4, text=f"识别于 {age:.1f} s 前" + ("（旧）" if old else ""),
                                fill="#ffab40" if old else "#b2ff59", anchor="sw", font=("TkDefaultFont", 9, "bold"))
        for name, c in self.overlay.get("cups") or []:
            uv = proj(c["rim_xyz_m"])
            if uv:
                u, v = uv
                self.cv.create_oval(u - 6, v - 6, u + 6, v + 6, outline="#00e5ff", width=2)
                self.cv.create_text(u + 8, v - 8, text=f"{name} {c['kind']}", fill="#00e5ff", anchor="sw",
                                    font=("TkDefaultFont", 9, "bold"))
        self._draw_tips(proj)                          # V3.68 FINGERTIP OVERLAY
        pl = self.overlay.get("plan")
        if pl:
            for key, col in (("grasp", "#ffeb3b"), ("release", "#76ff03")):
                uv = proj(pl[key])
                if uv:
                    u, v = uv
                    self.cv.create_line(u - 7, v, u + 7, v, fill=col, width=2)
                    self.cv.create_line(u, v - 7, u, v + 7, fill=col, width=2)

    def _draw_tips(self, proj):
        """V3.68 FINGERTIP OVERLAY: model fixed (F) / moving (M) fingertips for the current arm pose and the mount in
        the fields; compare with the real jaws in the image and tune 「偏转 °」 until they coincide"""
        if not bool(self.v_tips.get()) or not getattr(self.jpanel, "model", None):
            return
        try:
            a = self.pipe.actual_arm_deg(self.jpanel.model)
            if not a:
                return
            import NERO_V368_Real_Task_Record as TR
            if getattr(self, "_fast", None) is None:
                import NERO_V362_Cup_Skills as CK
                self._fast = CK.FastArm()
            geo = self.geometry_cfg()
            st = self.grip.last if self.grip.connected else None
            fx, mv = TR.fingertip_points(self._fast.fk(np.radians(np.asarray(a, float))), geo,
                                         (st or {}).get("open_deg"))
        except Exception:
            return
        uf, um = proj(fx), proj(mv)
        if not uf or not um:
            return
        self.cv.create_line(uf[0], uf[1], um[0], um[1], fill="#ffeb3b", width=2)
        for (u, v), col, t in ((uf, "#ff9800", "F 固定指"), (um, "#00e5ff", "M 动指")):
            self.cv.create_oval(u - 5, v - 5, u + 5, v + 5, outline=col, width=2)
            self.cv.create_text(u + 7, v + 7, text=t, fill=col, anchor="nw", font=("TkDefaultFont", 8, "bold"))

    def on_roll_sweep(self):
        """V3.68 ROLL SWEEP: plan the current cup pair for mount rolls -90..+90 deg (arm does not move)"""
        if self.busy():
            self._log("[WARN] [V3.68 REAL] 正在运行任务")
            return
        ycfg = self.yolo_cfg()
        bad = [b for b in self.preflight("plan", ycfg) if "关节反馈" in b or "YOLOE" in b or "主相机" in b or "标定" in b]
        if bad:
            self._log("[FAIL] [V3.68 ROLL SWEEP] 不能开始：" + "；".join(bad))
            return
        try:
            cfg = self.motion_cfg()
        except ValueError as exc:
            self._log(f"[FAIL] [V3.68 ROLL SWEEP] {exc}")
            return
        task = self.v_task.get().strip()
        source, target = parse_task_cups(task)
        knobs = self.knobs
        arm = PipelineArm(self.pipe, lambda: self.jpanel.model, log=self._log)
        io = RealIO(arm, None, lambda: camera_from_tab(self.cam_tab, yolo=ycfg), log=self._log, cfg=cfg,
                    motion_allowed=False)
        self.io = io

        def job():
            import NERO_V362_Cup_Skills as CK
            import NERO_V368_Real_Task_Record as TR
            sk = make_skills(io, knobs, self.skill)
            r = sk.run_tool("observe_scene", {})
            if not r.get("ok"):
                self._log(f"[FAIL] [V3.68 ROLL SWEEP] 识别杯子失败：{r.get('reason')} {r.get('error') or ''}")
                return
            s_, t_ = sk._cup(source), sk._cup(target)
            self._log(f"[INFO] [V3.68 ROLL SWEEP] {source} → {target}，偏转 −90…+90°，每个角度最多约 5 s")
            rows = TR.roll_sweep(knobs, (s_["rim"], s_["quat"], t_["rim"], t_["quat"]), io.current_q(), cfg,
                                 log=self._log, stop_fn=lambda: io.stop_ev.is_set())
            okr = [r_ for r_ in rows if r_["ok"]]
            if okr:
                b = max(okr, key=lambda r_: r_["margin"])
                good = ", ".join("{:+.0f}°".format(r_["roll"]) for r_ in okr)
                self._log(f"[OK] [V3.68 ROLL SWEEP] 可规划的偏转角：{good}"
                          f"；余量最大 {b['roll']:+.0f}°（{b['margin']}°）。当前填的 {cfg['roll_deg']:+.0f}°")
            else:
                self._log("[FAIL] [V3.68 ROLL SWEEP] 这组杯子位置在所有偏转角下都规划不到")
        self._start("偏转角扫描", job)

    def _draw_wrist(self):
        w = getattr(self.cam_tab, "wrist_src", None)
        if w is None or not w.running():
            return
        _r, _d, fid = w.frames()
        if fid == self._shown[1]:
            return
        data = w.display_ppm("color", size=(self.DW, self.DH))
        if data is None:
            return
        self.photo_w = self._photo(data)
        self.cvw.delete("all")
        self.cvw.create_image(0, 0, image=self.photo_w, anchor="nw")
        self.cvw.create_line(self.DW / 2 - 10, self.DH / 2, self.DW / 2 + 10, self.DH / 2, fill="#ff5252")
        self.cvw.create_line(self.DW / 2, self.DH / 2 - 10, self.DW / 2, self.DH / 2 + 10, fill="#ff5252")
        self._shown = (self._shown[0], fid)

    def status_text(self):
        try:
            moveit = self.pipe.lamp("moveit") == "green" if hasattr(self.pipe, "lamp") else False
        except Exception:
            moveit = False
        src = getattr(self.cam_tab, "src", None)
        cam = bool(src is not None and src.running())
        calib = bool(getattr(self.cam_tab, "result", None) or getattr(self.cam_tab, "calib", None))
        w = getattr(self.cam_tab, "wrist_src", None)
        wrist = bool(w is not None and w.running())
        path = bool(getattr(self.pipe, "supports_joint_path", lambda: False)())
        ok = lambda b: "●" if b else "○"
        return (f"{ok(moveit)} 机械臂 MoveIt  {ok(path)} 关节路径执行  {ok(cam)} 主相机 D435i  {ok(calib)} 手眼标定  "
                f"{ok(wrist)} 腕部相机  {ok(self.grip.connected)} SO-101 夹爪"
                + ("   · 运行中…" if self.busy() else ""))

    # ---------------------------------------------------------------- gripper buttons
    def on_ports(self):
        self.port_box.configure(values=self.GR.STSBus.list_ports() + [self.GR.SIM_PORT])

    def _grip_cfg_from_ui(self):
        try:
            return {"port": self.v_port.get().strip(), "baud": int(self.v_baud.get()), "servo_id": int(self.v_sid.get()),
                    "open_sign": 1 if int(self.v_sign.get()) >= 0 else -1}
        except Exception as exc:
            raise ValueError(f"夹爪参数无效：{exc}")

    def on_grip_connect(self):
        if self.busy():
            return
        try:
            c = self._grip_cfg_from_ui()
        except ValueError as exc:
            self._log(f"[FAIL] [V3.65 GRIPPER] {exc}")
            return
        self.grip.cfg.update(c)

        d = self._collect_settings()               # Tk variables: read here, in the Tk thread

        def job():
            ok, msg = self.grip.connect(c["port"], c["baud"], c["servo_id"])
            self._log(f"{'[OK]' if ok else '[FAIL]'} [V3.65 GRIPPER] 连接 {msg}")
            if ok:
                self.st = d
                save_settings(d, self.settings_path)
                self._grip_text()
        self._start("夹爪连接", job)

    def on_grip_disconnect(self):
        self.grip.disconnect()
        self.v_grip.set("夹爪未连接")

    def _grip_text(self):
        st = self.grip.status() if self.grip.connected else None
        if st is None:
            self._ui(lambda: self.v_grip.set("夹爪无读数"))
            return
        t = (f"张开 {st['open_deg']:+.1f}°（舵机 {st['servo_deg']:.1f}°）· 指尖缝 {st['gap_mm']:.1f} mm · 负载 "
             f"{st['load_pct']:.0f}% · 电流 {st['current_ma']:.0f} mA · {st['voltage_v']:.1f} V · {st['temp_c']} °C · "
             f"扭矩上限 {st['torque_limit_pct']:.0f}%" + (f" · ⚠ {'/'.join(st['errors'])}" if st["errors"] else "")
             + (" · 夹持中" if self.grip.held else "")
             + (f" · 碰合角 {self.grip.cfg['touch_deg']:+.2f}°（已空夹标定）" if self.grip.cfg.get("touch_calibrated")
                else " · 未做空夹标定"))
        self._ui(lambda: self.v_grip.set(t))

    def on_grip(self, what, val):
        if self.busy():
            self._log("[WARN] [V3.65 GRIPPER] 正在运行任务")
            return
        if not self.grip.connected:
            self._log("[WARN] [V3.65 GRIPPER] 先连接夹爪")
            return
        tq = self.v_tq.get().strip()
        knobs = getattr(self, "knobs", None) or {"cup_clamp_force_n": 5.0}

        def job():
            if what == "deg":
                r = self.grip.open_deg(val)
            elif what == "gap":
                r = self.grip.open_gap(val)
            elif what == "clamp":
                pct = float(tq) if tq else self.GR.clamp_torque_pct(knobs["cup_clamp_force_n"], self.grip.cfg)
                r = self.grip.clamp(pct)
            elif what == "touch":                     # V3.67 EMPTY-CLAMP CALIBRATION
                r = self.grip.calibrate_touch()
                if r.get("ok"):
                    self.st = dict(self.st, gripper=dict(self.st.get("gripper") or {},
                                                        touch_deg=self.grip.cfg["touch_deg"], touch_calibrated=True))
                    save_settings(self.st, self.settings_path)
            else:
                r = {"ok": bool(self.grip.relax()[0])}
            self._log(f"{'[OK]' if r.get('ok') else '[WARN]'} [V3.65 GRIPPER] {what}: "
                      + json.dumps({k: v for k, v in r.items() if k != 'ok'}, ensure_ascii=False))
            self._grip_text()
        self._start(f"夹爪 {what}", job)

    # ---------------------------------------------------------------- run
    def _save(self):
        """Tk thread only (reads Tk variables)"""
        self.st = self._collect_settings()
        save_settings(self.st, self.settings_path)

    def _collect_settings(self):
        d = dict(self.st)
        d["gripper"] = {k: self.grip.cfg[k] for k in ("port", "baud", "servo_id", "open_sign", "touch_deg",
                                                      "touch_calibrated")}
        try:
            d["geometry"] = self.geometry_cfg()                       # V3.67
        except ValueError:
            pass
        try:
            d["motion"] = {"joint_vmax_deg_s": float(self.v_jv.get()), "line_vmax_mm_s": float(self.v_lv.get()),
                           "clamp_torque_pct": float(self.v_tq.get()) if self.v_tq.get().strip() else None}
        except ValueError:
            pass
        d.update(provider=next((k for k, v in self.provs.items() if v == self.v_prov.get()), "openai"),
                 model=self.v_model.get().strip(), effort=self.v_effort.get(), task=self.v_task.get().strip(),
                 use_llm=bool(self.v_use_llm.get()), cup_prompt=self.v_cup_prompt.get().strip() or "cup",
                 show_tips=bool(self.v_tips.get()), speed_mult=self.v_speed.get(),
                 record_video=bool(self.v_video.get()), vision_review=bool(self.v_vision.get()))
        try:
            d["max_steps"] = int(self.v_steps.get())
        except ValueError:
            pass
        return d

    def _start(self, name, fn):
        if self.busy():
            self._log(f"[WARN] [V3.65 REAL] 正在运行：{name} 未开始")
            return False

        def wrap():
            try:
                fn()
            except StopRequested:
                self._log("[WARN] [V3.65 REAL] 已停止")
            except Exception as exc:
                self._log(f"[FAIL] [V3.65 REAL] {name}: {exc}")
        self.worker = threading.Thread(target=wrap, daemon=True, name=f"v365_{name}")
        self.worker.start()
        return True

    def motion_cfg(self):
        try:
            jv, lv = float(self.v_jv.get()), float(self.v_lv.get())
            tq = float(self.v_tq.get()) if self.v_tq.get().strip() else None
        except ValueError:
            raise ValueError("速度 / 扭矩请填数字")
        if not (1.0 <= jv <= 30.0) or not (5.0 <= lv <= 50.0) or (tq is not None and not (5.0 <= tq <= 80.0)):
            raise ValueError("关节速度 1–30 °/s，直线速度 5–50 mm/s，扭矩 5–80 %")
        return dict({"joint_vmax_deg_s": jv, "line_vmax_mm_s": lv, "clamp_torque_pct": tq}, **self.geometry_cfg())

    def geometry_cfg(self):
        """V3.67 REAL GRIPPER GEOMETRY + SAFETY from the fields (Tk thread) -> dict, ValueError when out of range"""
        try:
            c = {"clearance_mm": float(self.v_clear.get()), "align_rotate_mm": float(self.v_align.get()),
                 "descent_stop_mm": float(self.v_stop.get()) if self.v_stop.get().strip() else None,
                 "seat_mm": float(self.v_seat.get()), "ring_mm": float(self.v_ring.get()),
                 "roll_deg": float(self.v_roll.get()), "tip_corr_mm": list(self.tip_corr), "mount_rev": MOUNT_REV,
                 "nested_dz_mm": (float(self.v_nested.get()) if hasattr(self, "v_nested") and self.v_nested.get().strip()
                                  else None)}
        except ValueError:
            raise ValueError("夹爪几何 / 安全参数请填数字")
        if not (2.0 <= c["clearance_mm"] <= 15.0):
            raise ValueError("固定指离杯壁 2–15 mm")
        if not (0.0 <= c["align_rotate_mm"] <= 120.0):
            raise ValueError("对准高度 0–120 mm（0 = 不单独转腕）")
        if c["descent_stop_mm"] is not None and not (3.0 <= c["descent_stop_mm"] <= 40.0):
            raise ValueError("两段下探停点 3–40 mm（或留空）")
        if c["nested_dz_mm"] is not None and not (3.0 <= c["nested_dz_mm"] <= 40.0):
            raise ValueError("实测套叠增量 3–40 mm（或留空）")
        for k_, (lo, hi) in MOUNT_LIMITS.items():
            if not (lo <= c[k_] <= hi):
                raise ValueError(f"{k_} 超出 {lo}–{hi}")
        return c

    def on_tip_from_touch(self):
        """V3.67: Camera tab 4-pose touch calibration done with the FIXED-jaw fingertip -> fingertip correction"""
        tc = getattr(self.cam_tab, "tcp_calib", None)
        p = (tc or {}).get("tcp_in_flange")
        if not p or len(p) != 3:
            self._log("[WARN] [V3.67 REAL] 没有碰点标定结果：Camera调试 → TCP 碰点标定，用 SO-101 固定指的指尖去碰同一个点（≥ 4 个姿态）")
            return
        try:
            cfg = self.geometry_cfg()
        except ValueError as exc:
            self._log(f"[FAIL] [V3.67 REAL] {exc}")
            return
        corr, raw = tip_corr_from_touch(p, cfg)
        if corr is None:
            self._log(f"[FAIL] [V3.67 REAL] 碰点结果与模型指尖差 {raw} mm（> 25 mm）：标定时碰的不是固定指尖？或安装参数填错")
            return
        self.tip_corr = corr
        self._log(f"[OK] [V3.67 REAL] 指尖修正（夹爪坐标系）= {corr} mm，来自碰点标定 {[round(v * 1000, 1) for v in p]} mm")
        self.refresh_skill()

    def on_tip_clear(self):
        self.tip_corr = [0.0, 0.0, 0.0]
        self.refresh_skill()

    def yolo_cfg(self):
        """V3.66 REAL CUP YOLOE PERCEPTION: the Camera tab's YOLOE settings (Tk thread)"""
        import NERO_V366_Real_Cup_Yolo as YC
        # V3.68 FAST PERCEPTION: the task's own detector instance (same class / model / confidence as the Camera tab)
        # so a different prompt used in the Camera tab does not force a model reload before each cup perception
        import NERO_V347_Yolo_Perception as YP
        cd = getattr(self.cam_tab, "det_yolo", None)
        if isinstance(cd, YP.YoloEDetector):
            if getattr(self, "_det_own", None) is None:
                self._det_own = YP.YoloEDetector(log=self._log)
                self._det_own.device = yolo_device(self._log)        # V3.69 YOLOE DEVICE
            det = self._det_own
        else:
            det = cd
        cfg = YC.yolo_cfg_from_tab(self.cam_tab, self.v_cup_prompt.get().strip() or YC.DEFAULT_PROMPT, detector=det)
        self.v_yolo.set(YC.describe_cfg(cfg))
        return cfg

    def preflight(self, mode, ycfg=None):
        """-> list of blocking problems for a run mode"""
        bad = []
        if ycfg is not None and ycfg.get("error"):
            bad.append(f"YOLOE 不可用：{ycfg['error']}")
        src = getattr(self.cam_tab, "src", None)
        if src is None or not src.running():
            bad.append("主相机 D435i 没有连接")
        if not (getattr(self.cam_tab, "result", None) or getattr(self.cam_tab, "calib", None)):
            bad.append("没有主相机手眼标定")
        if not getattr(self.jpanel, "model", None) or self.pipe.actual_arm_deg(self.jpanel.model) is None:
            bad.append("没有机械臂关节反馈（实机 NERO 调试 → 步骤 5/6）")
        if mode in ("hover", "execute"):
            if self.pipe.lamp("moveit") != "green":
                bad.append("MoveIt 没有就绪（步骤 6）")
            if getattr(self.pipe, "teach_mode", False):
                bad.append("NERO 在示教模式")
            if self.pipe.move_busy():
                bad.append("另一个运动正在执行")
            w = getattr(self.cam_tab, "worker", None)
            if w is not None and w.is_alive():
                bad.append("Camera调试 页正在执行操作")
        if mode == "execute" and not self.grip.connected:
            bad.append("SO-101 夹爪没有连接")
        return bad

    def on_perceive(self):
        self._run_mode("plan", perceive_only=True)

    def on_run(self):
        self._run_mode(self.v_mode.get())

    def on_stop(self):
        if self.io is not None:
            self.io.stop_ev.set()
        if self.agent is not None:
            try:
                self.agent._stop_event.set()
            except Exception:
                pass
        try:
            if self.pipe.move_busy():
                self.pipe.cancel_move()
        except Exception:
            pass
        self._log("[WARN] [V3.65 REAL] 停止：当前运动已取消，任务在下一步前结束（紧急情况请按急停）")

    def _run_mode(self, mode, perceive_only=False):
        if self.busy():
            self._log("[WARN] [V3.65 REAL] 上一个任务还在运行")
            return
        ycfg = self.yolo_cfg()
        bad = self.preflight(mode, ycfg)
        if bad:
            self._log("[FAIL] [V3.65 REAL] 不能开始：" + "；".join(bad))
            return
        try:
            mc = self.motion_cfg()
            steps = int(self.v_steps.get())
        except ValueError as exc:
            self._log(f"[FAIL] [V3.65 REAL] {exc}")
            return
        mult = float(str(self.v_speed.get() or "1X").rstrip("Xx") or 1)      # V3.69 SPEED MULTIPLIER
        mc, speed_desc = apply_speed(mc, mult)
        record_video = bool(self.v_video.get())
        self._save()
        self.refresh_skill()
        task = self.v_task.get().strip()
        source, target = parse_task_cups(task)
        use_llm = bool(self.v_use_llm.get()) and mode == "execute"
        llm = {"provider": next((k for k, v in self.provs.items() if v == self.v_prov.get()), "openai"),
               "model": self.v_model.get().strip(), "api_key": self.v_key.get().strip(),
               "reasoning_effort": self.v_effort.get(), "max_tool_steps": steps, "task": task}
        knobs, skill = self.knobs, self.skill
        ycfg_text = self.v_yolo.get()
        try:
            note = self.cam_tab._calib_note() if hasattr(self.cam_tab, "_calib_note") else None
        except Exception:
            note = None
        arm = PipelineArm(self.pipe, lambda: self.jpanel.model, log=self._log)
        io = RealIO(arm, self.grip if self.grip.connected else None, lambda: camera_from_tab(self.cam_tab, yolo=ycfg),
                    log=self._log, cfg=mc, motion_allowed=(mode != "plan"))
        self.io = io
        if bool(self.v_vision.get()):                 # V3.74 LLM VISION REVIEW with the tab's provider / model / key
            import NERO_V374_Vision_Review as VR
            vcfg = {"provider": llm["provider"], "model": llm["model"], "api_key": llm["api_key"]}
            io.vision = VR.VisionReviewer(lambda c=vcfg: c, log=self._log,
                                          query=self.v_cup_prompt.get().strip() or "cup")
        # V3.71 LIVE OVERLAY: every perception of the task (observe / pick / place / verify) goes to the monitor at once
        io.on_perception = lambda res, io_=io: self._ui(lambda: self._apply_overlay(res, getattr(io_, "skills", None)))
        touch_ok = bool(self.grip.cfg.get("touch_calibrated"))
        geo = {k: mc.get(k) for k in ("roll_deg", "clearance_mm", "seat_mm", "ring_mm")}
        setup = (f"skill [{skill.get('code')}] · SO-101 偏转 {geo['roll_deg']:+.0f}° · 离杯壁 {geo['clearance_mm']:.0f} mm · "
                 f"空夹标定 {'已做' if touch_ok else '未做'} · 关节 {mc['joint_vmax_deg_s']} °/s · 直线 "
                 f"{mc['line_vmax_mm_s']} mm/s（{mult:g}X）· " + (f"LLM {llm['model']}" if use_llm else "固定流程"))

        def snap():
            src = getattr(self.cam_tab, "src", None)
            w = getattr(self.cam_tab, "wrist_src", None)
            m = src.frames()[0] if src is not None and src.running() else None
            wr = w.frames()[0] if w is not None and w.running() else None
            return (m, wr)

        def job():
            import NERO_V368_Real_Task_Record as TR
            t0 = time.time()
            self._log(f"[INFO] [V3.66 REAL CUP YOLOE] {ycfg_text}")
            if note:
                self._log(f"[WARN] [V3.65 REAL] 手眼标定：{note}")
            self._log(f"[V3.65 REAL CUP TASK] mode={mode} | {source} -> {target} | skill [{skill.get('code')}] | "
                      f"gripper {io.prof.label} | joint {mc['joint_vmax_deg_s']} deg/s · line {mc['line_vmax_mm_s']} mm/s")
            if not perceive_only:
                self._log(f"[INFO] [V3.69 REAL] 速度 {speed_desc}")
            if perceive_only:
                res = io.perceive("识别杯子")
                self.last_table_offset_mm = io.table_offset_mm
                self._set_overlay(res, None)
                return
            if mode == "execute" and not touch_ok:
                self._log("[WARN] [V3.68 REAL] 还没做「空夹标定」：空夹会被当成夹住（停在碰合点也算夹住），请先标定")
            if mode != "plan":
                text = (f"档位 {'② 悬停预演' if mode == 'hover' else '③ 实机执行'}：{source} → {target}\n"
                        f"速度 {speed_desc}\n"
                        + ("⚠ 高速档：先在 ② 里用同样速度试过再跑 ③；急停在手边。\n" if mult >= 3 else "")
                        + ("机械臂到源杯上方约 5 cm、升高、转腕、到目标杯上方，再回起点；不下探、不合夹爪。"
                           if mode == "hover" else
                           "机械臂会下探捏住杯沿、抬起、放进另一只杯子（夹爪会夹紧）。请先跑过 ② 悬停预演。")
                        + (f"\n⚠ 上次识别的桌面和安全区设定差 {self.last_table_offset_mm:+.0f} mm：手眼标定可能偏了，"
                           "抓取高度会偏同样多（V3.74）" if abs(getattr(self, "last_table_offset_mm", 0) or 0) > 10 else "")
                        + "\n确认周围无人、急停在手边？")
                if not self.ask("NERO 实机叠杯子", text):
                    self._log("[INFO] [V3.65 REAL] 已取消")
                    return
            rec = None
            try:                                       # V3.68 REAL TASK RECORD
                rec = TR.TaskRecorder(task if mode == "execute" else f"{source} into {target}", mode, snap_fn=snap,
                                      log=self._log, meta={"setup": setup, "source": source, "target": target})
            except Exception as exc:
                self._log(f"[WARN] [V3.68 TASK RECORD] 记录文件夹建不了：{exc}")
            out, err = {"ok": False}, None
            vid = None
            if rec is not None and record_video:          # V3.69 TASK VIDEO
                try:
                    vid = TR.TaskVideo(rec, snap, perception_fn=lambda: getattr(io, "last_perception", None),
                                       log=self._log)
                    vid.start()
                except Exception as exc:
                    self._log(f"[WARN] [V3.69 TASK VIDEO] 录像没开始：{exc}")
                    vid = None
            try:
                if mode == "plan":
                    out = run_plan_preview(io, source, target, knobs, skill, recorder=rec)
                elif mode == "hover":
                    out = run_hover(io, source, target, knobs, skill, recorder=rec)
                else:
                    agent = None
                    if use_llm:
                        import NERO_V2_7_1_Agent as AG
                        agent = AG.NeroLLMAgent(emit=self._log, tool_runner=None)
                        agent.provider = llm["provider"]
                        self.agent = agent
                    out = run_execute(io, source, target, knobs, skill, agent=agent, llm_cfg=llm, recorder=rec)
            except Exception as exc:
                err = exc
            finally:
                if vid is not None:
                    vid.stop()
            failed = None
            for r in out.get("results") or []:
                if not r.get("ok") and r.get("reason") != "ALREADY_STACKED":
                    failed = r
                    self._log(f"[FAIL] [V3.65 REAL CUP TASK] {r.get('tool')}: {r.get('reason')} "
                              f"{(r.get('error') or json.dumps(r.get('physical_verification') or {}, ensure_ascii=False))[:300]}")
                    break
            if not out.get("ok"):                     # V3.67 FAILURE SNAPSHOT
                io.save_snapshot(f"failed_{(failed or {}).get('tool') or mode}_{(failed or {}).get('reason') or ''}")
            sk = out.get("skills")
            self.skills = sk
            if io.table_offset_mm is not None:            # V3.74 TABLE OFFSET CHECK (shown before the next ② / ③)
                self.last_table_offset_mm = io.table_offset_mm
            if sk is not None:
                self._set_overlay(getattr(io, "last_perception", None), sk.plan, sk)
            # V3.68 REAL TASK RECORD: end-of-task summary (egg-task style)
            per = getattr(io, "last_perception", None) or {}
            cups_txt = " · ".join(f"{e.get('det_id')} {e.get('kind')}@({e['base_xyz_m'][0]:+.3f},{e['base_xyz_m'][1]:+.3f})"
                                  + (f" h{e['height_mm']:.0f}" if e.get("height_mm") else "")
                                  for e in per.get("cups") or [])
            pt = list(getattr(io, "perc_times", []) or [])
            ml = list(getattr(io, "motion_log", []) or [])
            stats = {"n_perc": len(pt), "perc_avg_s": (sum(pt) / len(pt)) if pt else 0.0, "n_motion": len(ml),
                     "motion_s": sum(float(m.get("elapsed_s") or 0) for m in ml)}
            steps_ = list(rec.steps) if rec is not None else []
            if mode == "hover":
                for i, st_ in enumerate(out.get("steps") or []):
                    steps_.append({"n": len(steps_) + 1, "tool": f"hover: {st_[0]}", "args": {}, "ok": st_[1],
                                   "reason": st_[3], "dt_s": 0.0, "details": [] if st_[2] is None else
                                   [f"TCP error {st_[2]} mm"]})
            final = out.get("final")
            if err is not None:
                final = f"{type(err).__name__}: {err}"
            text_, ok_ = TR.task_summary(task if mode == "execute" else f"{source} into {target}",
                                         mode + (" (LLM)" if use_llm else ""), steps_, bool(out.get("ok")),
                                         time.time() - t0, setup=setup, cups=cups_txt, folder=rec.folder if rec else None,
                                         final_text=final if isinstance(final, str) else None, stats=stats)
            self._log(text_)
            if rec is not None:
                rec.finish(text_, ok_)
            self._log(f"{'[OK]' if out.get('ok') else '[FAIL]'} [V3.65 REAL CUP TASK] mode={mode} finished in "
                      f"{time.time() - t0:.0f} s")
            if err is not None:
                raise err
        self._start(f"实机叠杯子 {mode}", job)

    def _apply_overlay(self, res, sk):
        """V3.71 LIVE OVERLAY (Tk thread): contours + labels of this perception, cup names from the skill"""
        self._set_overlay(res, getattr(sk, "plan", None) if sk is not None else None, sk, now=True)

    def _set_overlay(self, res, plan, sk=None, now=False):
        cups = []
        named = {}
        if sk is not None and getattr(sk, "cam", None):          # the skill's own cup names (tracking)
            for n, e in sk.cam.items():
                named[n] = {"rim_xyz_m": [float(v) for v in e["rim"]], "kind": e.get("in_stack") or "cup"}
        if named:
            cups = sorted(named.items())
        elif res:
            for i, c in enumerate(sorted(res.get("cups") or [], key=lambda c_: c_["base_xyz_m"][1])):
                cups.append((f"cup_{i + 1}" if len(res.get("cups") or []) == 2 else f"#{i + 1}", c))
        pl = None
        if plan and plan.get("_wps"):
            pl = {"grasp": plan["_wps"]["grasp"][0], "release": plan["_wps"]["release"][0]}
        dets = overlay_dets(res)
        t_ = time.time()                               # V3.71 LIVE OVERLAY: age of the perception on the monitor
        if now:
            self.overlay.update(cups=cups, plan=pl, dets=dets, t=t_)
        else:
            self._ui(lambda: self.overlay.update(cups=cups, plan=pl, dets=dets, t=t_))

    def copy_log(self):
        try:
            t = self.txt.get("1.0", "end")
            self.txt.clipboard_clear()
            self.txt.clipboard_append(t)
        except Exception:
            pass

    def shutdown(self):
        self.on_stop()
        try:
            self.grip.disconnect()
        except Exception:
            pass
