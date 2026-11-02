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

VERSION = "V3.66"
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
}
EXAMPLES = (("cup_1 → cup_2", "Stack cup_1 into cup_2"), ("cup_2 → cup_1", "Put cup_2 into cup_1"))


def load_settings(path=None):
    p = path or os.path.join(CALIB_DIR, SETTINGS_NAME)
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def save_settings(d, path=None):
    p = path or os.path.join(CALIB_DIR, SETTINGS_NAME)
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def real_profile(CK, lift_settle_s=0.5):
    """the simulation's SO-101 profile, with the short real settle (the real gripper has no CONNECT hold)"""
    p = CK.so101_profile()
    return CK.GripperProfile("so101", p.T_tcp_grasp, p.open_mm, p.label + " · real STS3215",
                             approach_clearance_m=p.approach_clearance_m, approach_along_wall=p.approach_along_wall,
                             flips=p.flips, release_gap_max_mm=p.release_gap_max_mm, lift_settle_s=lift_settle_s,
                             release_tilt_max_deg=p.release_tilt_max_deg)


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
        self.prof = real_profile(CK, float(self.cfg["lift_settle_s"]))
        self.motion_log = []

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

    def _execute(self, label, path, tcp_pts, vmax_lin, slow):
        """time + execute a joint path; afterwards compare the measured TCP with the path end"""
        if not self.motion_allowed:
            return {"ok": False, "label": label, "error": "MOTION_NOT_ALLOWED in this run mode"}
        if self.stop_ev.is_set():
            return {"ok": False, "label": label, "error": "STOPPED_BY_USER"}
        f = max(1.0, float(slow or 1.0))
        times = time_joint_path(path, math.radians(float(self.cfg["joint_vmax_deg_s"])) / f, tcp_pts,
                                (vmax_lin / f) if vmax_lin else None)
        t0 = time.time()
        r = dict(self.arm.exec_path(path, times, label) or {"ok": False, "error": "no result"})
        self.motion_log.append({"label": label, "points": len(path), "planned_s": times[-1],
                                "elapsed_s": round(time.time() - t0, 2), "ok": bool(r.get("ok")), "error": r.get("error")})
        if not r.get("ok"):
            return {"ok": False, "label": label, "error": r.get("error"), "fence_violation": r.get("fence_violation")}
        self.sleep(float(self.cfg["settle_s"]))
        q = self.arm.q()
        T_goal = self.fast.fk(np.asarray(path[-1], float))
        if q is None:
            return {"ok": False, "label": label, "error": "no joint feedback after the motion"}
        T_act = self.fast.fk(np.asarray(q, float))
        err = float(np.linalg.norm(T_act[:3, 3] - T_goal[:3, 3]))
        jerr = max(abs(math.degrees(a - b)) for a, b in zip(q, path[-1]))
        return {"ok": err * 1000.0 <= float(self.cfg["tcp_tol_mm"]), "label": label, "tcp_error_mm": round(err * 1000, 1),
                "joint_error_deg": round(jerr, 2), "kind": r.get("kind", "path"),
                "error": None if err * 1000.0 <= float(self.cfg["tcp_tol_mm"]) else
                f"TCP {err * 1000:.1f} mm from the goal after the motion"}

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

    def move_joints(self, label, q, slow=None, speed=None):
        try:
            q0 = np.asarray(self.current_q(), float)
        except Exception as exc:
            return {"ok": False, "label": label, "error": str(exc)}
        q1 = np.asarray(q, float)
        chk = self.CK.check_joint_path([list(q0), list(q1)], jump_deg=1e9)
        if not chk["ok"]:
            return {"ok": False, "label": label, "error": f"joint goal rejected: {chk['why']}"}
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

    # -- eyes
    def gripper_boxes(self, z_shift=0.0):
        """oriented box around the SO-101 + ring (robot self-filter of the cup perception)"""
        import NERO_V364_SO101_Gripper as G
        try:
            T_tcp = self.fast.fk(np.asarray(self.current_q(), float))
        except Exception:
            return []
        T_l7 = T_tcp @ np.linalg.inv(self.CK.T_L7_TCP)
        Tg = T_l7 @ G.T_l7_gripper()
        c = Tg[:3, :3] @ np.array([0.012, 0.0, -0.05]) + Tg[:3, 3] - np.array([0.0, 0.0, z_shift])
        return [(c, Tg[:3, :3], np.array([0.055, 0.04, 0.075]))]

    def perceive(self, tag):
        """V3.66 REAL CUP YOLOE PERCEPTION: fixed D435i -> the Camera tab's YOLOE object pipeline (same detector, model,
        prompt resolution, depth fusion, locate_objects) + cup axis / silhouette height / kind (NERO_V366).
        No colour fallback: YOLOE unavailable -> the perception raises, the skill aborts (PERCEPTION_UNAVAILABLE)."""
        import NERO_V366_Real_Cup_Yolo as YC
        cam = self.camera_fn()
        try:
            res = YC.perceive_cups_yolo(cam, robot_boxes=self.gripper_boxes())
        except YC.YoloCupUnavailable as exc:
            self.emit(f"[FAIL] [V3.66 REAL CUP YOLOE] YOLOE 不可用：{exc} → 任务终止（不改用颜色法）")
            raise
        self.last_perception = res
        self.emit(f"[V3.66 REAL CUP YOLOE] {tag} | table z {res['table_z_m'] * 1000:+.0f} mm | YOLOE {res['n_2d']} 个 | "
                  + YC.summary(res))
        return res


# ================================================================================================ run modes
def make_skills(io, knobs=None, skill=None, log=None):
    import NERO_V362_Cup_Skills as CK
    if knobs is None or skill is None:
        knobs, skill = CK.cup_knobs()
    sk = CK.CupSkills(io, knobs, skill)
    return sk


def run_plan_preview(io, source, target, knobs=None, skill=None):
    """mode 1: camera + plan only"""
    io.motion_allowed = False
    sk = make_skills(io, knobs, skill)
    out = [sk.run_tool("observe_scene", {})]
    if out[-1].get("ok"):
        out.append(sk.run_tool("plan_cup_stack", {"source_cup": source, "target_cup": target}))
    return {"ok": all(r.get("ok") for r in out), "results": out, "skills": sk}


def run_hover(io, source, target, knobs=None, skill=None):
    """mode 2: pregrasp hover above the source rim (jaws ~5 cm above the rim, gripper only opens) -> straight up to
    the lift height -> rotate -> above the target (the carry pose, without a cup) -> back to the start posture"""
    CK = io.CK
    sk = make_skills(io, knobs, skill)
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
                (f"hover above {source} (pregrasp)", lambda: io.move_joints(f"V3.65 hover {source}", Q["pregrasp"])),
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


def run_execute(io, source, target, knobs=None, skill=None, agent=None, llm_cfg=None):
    """mode 3: the full task - scripted (observe, plan, pick, place, verify, back) or the LLM agent"""
    import NERO_V362_Cup_Skills as CK
    sk = make_skills(io, knobs, skill)
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
            "calib_note": note, "yolo": yolo}


KIND_TEXT = {"cup": ("单杯", "#00e676"), "stack": ("已叠好", "#00e5ff"), "unknown": ("不像杯子", "#ff9100"),
             "lifted": ("在夹爪里", "#b0bec5"), "outside": ("围栏外", "#ff5252"), "edge": ("被画面切掉", "#ff9100"),
             "no_depth": ("没有深度", "#ff9100"), "rejected": ("剔除", "#9e9e9e")}


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
        ttk.Button(r4, text="■ 停止", command=self.on_stop).pack(side="right")
        ttk.Button(r4, text="▶ 运行", command=self.on_run).pack(side="right", padx=4)

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
        p = real_profile(CK)
        import NERO_V364_SO101_Gripper as G
        tq = self.GR.clamp_torque_pct(knobs["cup_clamp_force_n"], self.grip.cfg)
        self.v_skill.set(
            f"技能 [{skill.get('code')}] {skill.get('short_name')}（{skill.get('source')}）· 捏深 "
            f"{knobs['cup_rim_pinch_depth_mm']:.0f} mm · 预抓取高 {knobs['cup_pregrasp_height_mm']:.0f} mm · 松手间隙 "
            f"{min(knobs['cup_release_gap_mm'], p.release_gap_max_mm or 1e9):.0f} mm · 对中容差 "
            f"{knobs['cup_release_xy_tol_mm']} mm · 夹紧力 {knobs['cup_clamp_force_n']} N（≈ 扭矩 {tq:.0f} %）\n"
            f"夹爪档案 SO-101：转接环 {G.MOUNT['ring_m'] * 1000:.0f} mm · 偏转 {G.MOUNT['roll_deg']:+.0f}° · "
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
        for d in self.overlay.get("dets") or []:           # V3.66 REAL CUP YOLOE PERCEPTION: YOLOE masks
            pts = []
            for u, v in d["contour"]:
                pts += [u * sx, v * sy]
            if len(pts) >= 6:
                self.cv.create_polygon(*pts, outline=d["color"], fill="", width=2)
            if d.get("anchor"):
                self.cv.create_text(d["anchor"][0] * sx, d["anchor"][1] * sy - 2, text=d["label"], fill=d["color"],
                                    anchor="s", font=("TkDefaultFont", 8, "bold"))
        for name, c in self.overlay.get("cups") or []:
            uv = proj(c["rim_xyz_m"])
            if uv:
                u, v = uv
                self.cv.create_oval(u - 6, v - 6, u + 6, v + 6, outline="#00e5ff", width=2)
                self.cv.create_text(u + 8, v - 8, text=f"{name} {c['kind']}", fill="#00e5ff", anchor="sw",
                                    font=("TkDefaultFont", 9, "bold"))
        pl = self.overlay.get("plan")
        if pl:
            for key, col in (("grasp", "#ffeb3b"), ("release", "#76ff03")):
                uv = proj(pl[key])
                if uv:
                    u, v = uv
                    self.cv.create_line(u - 7, v, u + 7, v, fill=col, width=2)
                    self.cv.create_line(u, v - 7, u, v + 7, fill=col, width=2)

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
             + (" · 夹持中" if self.grip.held else ""))
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
        d["gripper"] = {k: self.grip.cfg[k] for k in ("port", "baud", "servo_id", "open_sign")}
        try:
            d["motion"] = {"joint_vmax_deg_s": float(self.v_jv.get()), "line_vmax_mm_s": float(self.v_lv.get()),
                           "clamp_torque_pct": float(self.v_tq.get()) if self.v_tq.get().strip() else None}
        except ValueError:
            pass
        d.update(provider=next((k for k, v in self.provs.items() if v == self.v_prov.get()), "openai"),
                 model=self.v_model.get().strip(), effort=self.v_effort.get(), task=self.v_task.get().strip(),
                 use_llm=bool(self.v_use_llm.get()), cup_prompt=self.v_cup_prompt.get().strip() or "cup")
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
        return {"joint_vmax_deg_s": jv, "line_vmax_mm_s": lv, "clamp_torque_pct": tq}

    def yolo_cfg(self):
        """V3.66 REAL CUP YOLOE PERCEPTION: the Camera tab's YOLOE settings (Tk thread)"""
        import NERO_V366_Real_Cup_Yolo as YC
        cfg = YC.yolo_cfg_from_tab(self.cam_tab, self.v_cup_prompt.get().strip() or YC.DEFAULT_PROMPT)
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
        arm = PipelineArm(self.pipe, lambda: self.jpanel.model, log=self._log)
        io = RealIO(arm, self.grip if self.grip.connected else None, lambda: camera_from_tab(self.cam_tab, yolo=ycfg),
                    log=self._log, cfg=mc, motion_allowed=(mode != "plan"))
        self.io = io

        def job():
            t0 = time.time()
            note = None
            self._log(f"[INFO] [V3.66 REAL CUP YOLOE] {ycfg_text}")
            try:
                note = camera_from_tab(self.cam_tab, n_frames=1).get("calib_note")
            except Exception:
                pass
            if note:
                self._log(f"[WARN] [V3.65 REAL] 手眼标定：{note}")
            self._log(f"[V3.65 REAL CUP TASK] mode={mode} | {source} -> {target} | skill [{skill.get('code')}] | "
                      f"gripper {io.prof.label} | joint {mc['joint_vmax_deg_s']} deg/s · line {mc['line_vmax_mm_s']} mm/s")
            if perceive_only:
                res = io.perceive("识别杯子")
                self._set_overlay(res, None)
                return
            if mode == "plan":
                out = run_plan_preview(io, source, target, knobs, skill)
            else:
                text = (f"档位 {'② 悬停预演' if mode == 'hover' else '③ 实机执行'}：{source} → {target}\n"
                        f"关节 {mc['joint_vmax_deg_s']} °/s · 直线 {mc['line_vmax_mm_s']} mm/s\n"
                        + ("机械臂到源杯上方约 5 cm、升高、转腕、到目标杯上方，再回起点；不下探、不合夹爪。"
                           if mode == "hover" else
                           "机械臂会下探捏住杯沿、抬起、放进另一只杯子（夹爪会夹紧）。请先跑过 ② 悬停预演。")
                        + "\n确认周围无人、急停在手边？")
                if not self.ask("NERO 实机叠杯子", text):
                    self._log("[INFO] [V3.65 REAL] 已取消")
                    return
                if mode == "hover":
                    out = run_hover(io, source, target, knobs, skill)
                else:
                    agent = None
                    if use_llm:
                        import NERO_V2_7_1_Agent as AG
                        agent = AG.NeroLLMAgent(emit=self._log, tool_runner=None)
                        agent.provider = llm["provider"]
                        self.agent = agent
                    out = run_execute(io, source, target, knobs, skill, agent=agent, llm_cfg=llm)
            for r in out.get("results") or []:
                if not r.get("ok") and r.get("reason") != "ALREADY_STACKED":
                    self._log(f"[FAIL] [V3.65 REAL CUP TASK] {r.get('tool')}: {r.get('reason')} "
                              f"{(r.get('error') or json.dumps(r.get('physical_verification') or {}, ensure_ascii=False))[:300]}")
                    break
            sk = out.get("skills")
            self.skills = sk
            if sk is not None:
                self._set_overlay(getattr(io, "last_perception", None), sk.plan, sk)
            self._log(f"{'[OK]' if out.get('ok') else '[FAIL]'} [V3.65 REAL CUP TASK] mode={mode} finished in "
                      f"{time.time() - t0:.0f} s")
        self._start(f"实机叠杯子 {mode}", job)

    def _set_overlay(self, res, plan, sk=None):
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
        self._ui(lambda: self.overlay.update(cups=cups, plan=pl, dets=dets))

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
