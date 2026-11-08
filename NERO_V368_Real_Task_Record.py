#!/usr/bin/env python3
"""NERO V3.68 REAL TASK RECORD - debugging aids for the real cup task (「Agentic实机任务」, was 「NERO实机任务」).

V3.68 REAL TASK RECORD
  TaskRecorder   one folder per submitted task under <project>/task/ (env NERO_V368_TASK_DIR overrides): before and
                 after every tool call (LLM decision or scripted step) a labelled image (fixed D435i + wrist camera,
                 YOLOE contours, step / tool / arguments / result / time stamp), steps.jsonl and summary.txt
  step_line      one compact log line per tool call with the numbers that matter for debugging
  task_summary   the end-of-task block in the egg task's style ([... SUMMARY GREEN|YELLOW|RED], STATUS, TASK, SETUP,
                 CUPS, TOOLS with one line each, LAST FAILURE, TIME, RECORD)
V3.68 FINGERTIP OVERLAY
  fingertip_points  model fixed / moving fingertip positions (base frame) for the current arm pose, the measured mount
                    (roll, seat, ring, tip correction) and the jaw opening -> drawn on the main camera monitor so the
                    real jaws can be compared with the model (tune 「偏转 °」 until they coincide)
V3.68 ROLL SWEEP
  roll_sweep     plan the current cup pair for a range of mount rolls -> min joint margin per roll (which roll the
                 arm can work with, e.g. before re-mounting the gripper)
Images use ASCII labels (OpenCV fonts have no Chinese glyphs).  Nothing here moves the arm.
"""
import json
import math
import os
import sys
import time

VERSION = "V3.69"
MARKER = "V3.68 REAL TASK RECORD"
MARKER_V369 = "V3.69 TASK VIDEO"
HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np                      # noqa: E402

TASK_KEEP = 100                          # newest task folders kept


def task_root():
    return os.environ.get("NERO_V368_TASK_DIR") or os.path.join(HERE, "task")


def _cv2():
    import NERO_V325_Camera as CAM       # V3.46 Qt-variable guard
    return CAM._import_cv2()


def _slug(text, n=32):
    t = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text or "").strip())
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_")[:n] or "task"


def _args_text(args):
    return ", ".join(f"{k}={v}" for k, v in (args or {}).items())


# ================================================================================================ step line
def step_details(tool, res):
    """the numbers worth a log line for one tool result"""
    pv = (res or {}).get("physical_verification") or {}
    plan = (res or {}).get("plan") or {}
    d = []
    if tool == "plan_cup_stack" and plan:
        d.append(f"grasp {plan.get('grasp_rim_angle_rel_deg', 0):+.0f}° place {plan.get('place_rim_angle_rel_deg', 0):+.0f}° "
                 f"margin {plan.get('min_joint_margin_deg')}° ({plan.get('grasp_binding') if (plan.get('grasp_margin_deg') or 99) <= (plan.get('place_margin_deg') or 99) else plan.get('place_binding')})")
    if tool == "pick_cup":
        ar = pv.get("align_rotate") or {}
        if ar:
            d.append(f"转腕 {ar.get('rotated_deg', 0):+.0f}°")
        pc = pv.get("pre_contact_check") or {}
        if pc.get("cup_offset_mm") is not None:
            d.append(f"杯口上方复查偏差 {pc['cup_offset_mm']} mm")
        if "cup_moved_by_descent_mm" in pv:
            d.append(f"下探后位移 {pv['cup_moved_by_descent_mm']} mm" + (
                f"·倾斜变化 {pv['cup_tilt_change_deg']}°" if "cup_tilt_change_deg" in pv else ""))
        c = pv.get("clamp") or {}
        if c:
            d.append(f"夹紧停在缝 {c.get('gap_mm')} mm · 夹住={'是' if c.get('hold_active') else '否'}")
        if "lift_tcp_rise_mm" in pv:
            d.append(f"抬起 TCP +{pv['lift_tcp_rise_mm']} mm / 杯 +{pv.get('cup_rise_mm')} mm")
    if tool == "place_cup_into":
        for k in ("centre_error_mm", "xy_error_mm", "release_gap_mm"):
            if k in pv:
                d.append(f"{k} {pv[k]}")
    if tool == "verify_stack":
        d.append(f"套好={'是' if pv.get('nested_full') else '否'} · 杯底高差 {pv.get('base_dz_mm')} mm")
    if pv.get("detail"):
        d.append(str(pv["detail"])[:120])
    return d


def step_line(n, tool, args, res, dt):
    ok = bool((res or {}).get("ok"))
    reason = (res or {}).get("reason") or (res or {}).get("error")
    det = step_details(tool, res)
    return (f"[{'OK' if ok else 'FAIL'}] [V3.68 STEP {n}] {tool}({_args_text(args)}) → "
            f"{'OK' if ok else reason} | {dt:.1f} s" + (" | " + " · ".join(det) if det else ""))


# ================================================================================================ recorder
class TaskRecorder:
    """before / after images of every tool call of one task (see module docstring)"""

    def __init__(self, task, mode, snap_fn=None, log=None, meta=None, root=None):
        self.task, self.mode, self.snap_fn = task, mode, snap_fn
        self.log = log or (lambda s: None)
        self.meta = dict(meta or {})
        self.t0 = time.time()
        self.steps = []
        self.n = 0
        self.status = {"step": 0, "tool": "-", "phase": "start", "result": "", "t": time.time()}   # V3.69 TASK VIDEO
        root = root or task_root()
        self.folder = os.path.join(root, f"{time.strftime('%Y%m%d_%H%M%S')}_{mode}_{_slug(task)}")
        os.makedirs(self.folder, exist_ok=True)
        with open(os.path.join(self.folder, "task.json"), "w", encoding="utf-8") as f:
            json.dump({"task": task, "mode": mode, "start": time.strftime("%Y-%m-%d %H:%M:%S"), **self.meta}, f,
                      ensure_ascii=False, indent=1, default=str)
        self._prune(root)
        self.log(f"[INFO] [V3.68 TASK RECORD] 本次任务记录文件夹：{self.folder}")

    @staticmethod
    def _prune(root, keep=TASK_KEEP):
        import shutil
        try:
            subs = sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)))
            for x in subs[:-keep] if len(subs) > keep else []:
                shutil.rmtree(os.path.join(root, x), ignore_errors=True)
        except OSError:
            pass

    def wrap(self, skills, perception_fn=None):
        """record every skills.run_tool call (the LLM agent loop and the scripted episode both use it)"""
        orig = skills.run_tool

        def run_tool(name, args):
            self.n += 1
            n = self.n
            st0 = {"stage": getattr(skills, "stage", None), "held": getattr(skills, "held", None)}
            self.status = {"step": n, "tool": f"{name}({_args_text(args)})", "phase": "running", "result": "",
                           "stage": st0["stage"], "held": st0["held"], "t": time.time()}
            self.shot(n, "before", name, args, None, st0, perception_fn)
            t = time.perf_counter()
            res = orig(name, args)
            dt = time.perf_counter() - t
            st1 = {"stage": getattr(skills, "stage", None), "held": getattr(skills, "held", None)}
            self.status = {"step": n, "tool": f"{name}({_args_text(args)})", "phase": "done",
                           "result": "OK" if (res or {}).get("ok") else f"FAIL {(res or {}).get('reason')}",
                           "stage": st1["stage"], "held": st1["held"], "t": time.time()}
            img = self.shot(n, "after", name, args, res, st1, perception_fn, dt)
            rec = {"n": n, "tool": name, "args": args, "ok": bool((res or {}).get("ok")),
                   "reason": (res or {}).get("reason") or (res or {}).get("error"), "dt_s": round(dt, 2),
                   "time": time.strftime("%H:%M:%S"), "state_before": st0, "state_after": st1,
                   "details": step_details(name, res), "warnings": (res or {}).get("warnings") or [],
                   "physical_verification": (res or {}).get("physical_verification"), "image_after": img}
            self.steps.append(rec)
            try:
                with open(os.path.join(self.folder, "steps.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            except OSError:
                pass
            self.log(step_line(n, name, args, res, dt))
            return res
        skills.run_tool = run_tool
        return skills

    def shot(self, n, phase, tool, args, res, state, perception_fn=None, dt=None):
        """labelled image -> path (None when no camera frame)"""
        try:
            imgs = self.snap_fn() if self.snap_fn else None
        except Exception:
            imgs = None
        if not imgs or imgs[0] is None:
            return None
        main = np.ascontiguousarray(np.asarray(imgs[0])[..., ::-1])        # RGB -> BGR
        cv2 = _cv2()
        per = perception_fn() if perception_fn else None
        for e in (per or {}).get("dets") or []:
            c = e.get("contour") or []
            if len(c) >= 3:
                col = (0, 230, 118) if e.get("kind") in ("cup", "stack") else (0, 145, 255)
                cv2.polylines(main, [np.asarray(c, np.int32).reshape(-1, 1, 2)], True, col, 2)
                top = min(c, key=lambda p: p[1])
                cv2.putText(main, f"{e.get('det_id', '')} {e.get('kind')}", (int(top[0]) - 20, max(14, int(top[1]) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        h, w = main.shape[:2]
        wr = imgs[1] if len(imgs) > 1 else None
        if wr is not None:
            wr = np.ascontiguousarray(np.asarray(wr)[..., ::-1])
            ww = int(round(wr.shape[1] * h / float(wr.shape[0])))
            wr = cv2.resize(wr, (ww, h))
            main = np.hstack([main, wr])
        ok = None if res is None else bool(res.get("ok"))
        col = (90, 90, 90) if ok is None else ((40, 140, 40) if ok else (40, 40, 200))
        lines = [f"#{n} {phase.upper()}  {tool}({_args_text(args)})"[:110],
                 f"{time.strftime('%Y-%m-%d %H:%M:%S')}  mode={self.mode}  stage={state.get('stage')}  "
                 f"held={state.get('held')}"]
        if res is not None:
            lines.append((f"RESULT: {'OK' if ok else 'FAIL ' + str(res.get('reason') or res.get('error'))}"
                          + (f"  ({dt:.1f} s)" if dt is not None else ""))[:110])
            pv = res.get("physical_verification") or {}
            c = pv.get("clamp") or {}
            if c:
                lines.append(f"clamp gap {c.get('gap_mm')} mm held={c.get('hold_active')}  "
                             f"lift tcp {pv.get('lift_tcp_rise_mm')} mm cup {pv.get('cup_rise_mm')} mm")
        bar = np.zeros((26 * len(lines) + 10, main.shape[1], 3), np.uint8)
        bar[:] = col
        for i, t in enumerate(lines):
            cv2.putText(bar, t.encode("ascii", "replace").decode(), (10, 24 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (255, 255, 255), 2)
        img = np.vstack([bar, main])
        path = os.path.join(self.folder, f"{n:02d}_{phase}_{_slug(tool, 24)}.jpg")
        cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        return path

    def finish(self, summary_text, ok):
        try:
            with open(os.path.join(self.folder, "summary.txt"), "w", encoding="utf-8") as f:
                f.write(summary_text + "\n")
            with open(os.path.join(self.folder, "task.json"), "r+", encoding="utf-8") as f:
                d = json.load(f)
                d.update(end=time.strftime("%Y-%m-%d %H:%M:%S"), ok=bool(ok), duration_s=round(time.time() - self.t0, 1))
                f.seek(0)
                f.truncate()
                json.dump(d, f, ensure_ascii=False, indent=1, default=str)
        except OSError:
            pass


# ================================================================================================ video
class TaskVideo:
    """V3.69 TASK VIDEO: task_video.mp4 in the task record folder - fixed D435i (+ wrist) at `fps`, YOLOE contours of
    the last perception, header with time stamp, elapsed time, mode, current step / tool / phase / result, stage and
    held cup.  A daemon thread; stop() finishes the file."""

    def __init__(self, recorder, snap_fn, fps=5.0, width=1280, perception_fn=None, log=None):
        import threading
        self.rec, self.snap_fn, self.fps, self.width = recorder, snap_fn, float(fps), int(width)
        self.perception_fn = perception_fn
        self.log = log or (lambda s: None)
        self.path = os.path.join(recorder.folder, "task_video.mp4")
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True, name="v369_task_video")
        self.writer, self.frames = None, 0

    def start(self):
        self._th.start()
        self.log(f"[INFO] [V3.69 TASK VIDEO] 录像中（{self.fps:g} fps）→ {self.path}")

    def stop(self, timeout=5.0):
        self._stop.set()
        self._th.join(timeout)
        self.log(f"[INFO] [V3.69 TASK VIDEO] 录像结束：{self.frames} 帧，{self.frames / max(self.fps, 1e-6):.0f} s → "
                 f"{self.path}")

    def compose(self):
        cv2 = _cv2()
        imgs = self.snap_fn()
        if not imgs or imgs[0] is None:
            return None
        main = np.ascontiguousarray(np.asarray(imgs[0])[..., ::-1])
        per = self.perception_fn() if self.perception_fn else None
        for e in (per or {}).get("dets") or []:
            c = e.get("contour") or []
            if len(c) >= 3:
                col = (0, 230, 118) if e.get("kind") in ("cup", "stack") else (0, 145, 255)
                cv2.polylines(main, [np.asarray(c, np.int32).reshape(-1, 1, 2)], True, col, 2)
        h = main.shape[0]
        wr = imgs[1] if len(imgs) > 1 else None
        if wr is not None:
            wr = np.ascontiguousarray(np.asarray(wr)[..., ::-1])
            wr = cv2.resize(wr, (int(round(wr.shape[1] * h / float(wr.shape[0]))), h))
            main = np.hstack([main, wr])
        sc = self.width / float(main.shape[1])
        main = cv2.resize(main, (self.width, int(round(main.shape[0] * sc)) // 2 * 2))
        st = self.rec.status
        lines = [f"{time.strftime('%Y-%m-%d %H:%M:%S')}  T+{time.time() - self.rec.t0:6.1f} s  mode={self.rec.mode}",
                 f"step #{st.get('step')} {st.get('phase')}: {st.get('tool')} {st.get('result') or ''}"[:120],
                 f"stage={st.get('stage')}  held={st.get('held')}"]
        bar = np.zeros((26 * len(lines) + 8, self.width, 3), np.uint8)
        bar[:] = (40, 40, 40) if st.get("phase") != "done" else ((40, 120, 40) if str(st.get("result")).startswith("OK")
                                                                 else (40, 40, 170))
        for i, t in enumerate(lines):
            cv2.putText(bar, t.encode("ascii", "replace").decode(), (8, 22 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
        img = np.vstack([bar, main])
        return img[: img.shape[0] // 2 * 2]

    def _run(self):
        cv2 = _cv2()
        period = 1.0 / max(self.fps, 0.5)
        nxt = time.time()
        while not self._stop.is_set():
            try:
                img = self.compose()
                if img is not None:
                    if self.writer is None:
                        self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps,
                                                      (img.shape[1], img.shape[0]))
                        self.size = (img.shape[1], img.shape[0])
                    if (img.shape[1], img.shape[0]) != self.size:
                        img = cv2.resize(img, self.size)
                    self.writer.write(img)
                    self.frames += 1
            except Exception as exc:
                self.log(f"[WARN] [V3.69 TASK VIDEO] 一帧没录上：{exc}")
            nxt += period
            self._stop.wait(max(0.0, nxt - time.time()))
        if self.writer is not None:
            self.writer.release()


# ================================================================================================ summary
def task_summary(task, mode, steps, ok, elapsed_s, setup=None, cups=None, final_text=None, folder=None, stats=None):
    """end-of-task block (egg-task style).  steps: [{"n","tool","args","ok","reason","dt_s","details","warnings"}]"""
    failed = [s for s in steps if not s.get("ok") and s.get("reason") != "ALREADY_STACKED"]
    warns = sum(len(s.get("warnings") or []) for s in steps)
    if ok and not failed and not warns:
        color, status = "GREEN", "SUCCESS"
    elif ok:
        color, status = "YELLOW", "SUCCESS WITH WARNINGS"
    else:
        color, status = "RED", "FAILED"
    lines = [f"[REAL TASK SUMMARY {color}]", f"STATUS: {status}", f"TASK: {task} | MODE: {mode}"]
    if setup:
        lines.append(f"SETUP: {setup}")
    if cups:
        lines.append(f"CUPS: {cups}")
    if final_text:
        t = str(final_text).strip().replace("\n", " ")
        lines.append(f"OUTCOME: {t[:400]}")
    lines.append(f"TOOLS: {len(steps)} total | {len(failed)} failed | {warns} warnings")
    for s in steps:
        lines.append(f"  #{s['n']} {s['tool']}({_args_text(s.get('args'))}) "
                     f"{'OK' if s.get('ok') else 'FAIL ' + str(s.get('reason'))} {s.get('dt_s', 0):.1f} s"
                     + (" | " + " · ".join(s.get("details") or []) if s.get("details") else ""))
    lines.append("LAST FAILURE: " + (f"{failed[-1]['tool']}: {failed[-1]['reason']}"
                                     + (f" ({' · '.join(failed[-1].get('details') or [])})" if failed[-1].get("details") else "")
                                     if failed else "none"))
    st = stats or {}
    lines.append(f"TIME: {elapsed_s:.0f} s" + (
        f" | 识别 {st.get('n_perc', 0)} 次 平均 {st.get('perc_avg_s', 0):.1f} s · 运动 {st.get('n_motion', 0)} 段 "
        f"{st.get('motion_s', 0):.0f} s" if st else ""))
    if folder:
        lines.append(f"RECORD: {folder}")
    return "\n".join(lines), ok


# ================================================================================================ fingertips
def fingertip_points(T_base_tcp, cfg=None, open_deg=None):
    """V3.68 FINGERTIP OVERLAY: (fixed_tip, moving_tip) in the base frame for a TCP pose, the real mount (roll / seat /
    ring / tip correction from cfg) and the real jaw opening (deg from the touch point; None = closed)"""
    import NERO_V364_SO101_Gripper as G
    import NERO_V365_Real_Cup_Task as RT
    import NERO_V365_STS3215_Gripper as GR
    import NERO_V362_Cup_Skills as CK
    T_l7g = RT.real_mount(cfg or {})[0]
    T_bg = np.asarray(T_base_tcp, float) @ np.linalg.inv(CK.T_L7_TCP) @ T_l7g
    tc = np.asarray((cfg or {}).get("tip_corr_mm") or [0, 0, 0], float) / 1000.0
    z_tip = G.PAD_FIXED_CENTRE[2] - G.PAD_HALF[2]
    fixed = np.array([G.PAD_FIXED_CENTRE[0], 0.0, z_tip]) + tc
    th = math.radians(GR.theta_touch_deg() + float(open_deg or 0.0))
    h = np.asarray(G.HINGE_POS, float)
    r = np.array([-G.PAD_FIXED_CENTRE[0], 0.0, z_tip]) - h
    c, s = math.cos(th), math.sin(th)
    moving = np.array([h[0] + r[0] * c - r[2] * s, 0.0, h[2] + r[0] * s + r[2] * c]) + tc
    to_b = lambda p: (T_bg[:3, :3] @ p + T_bg[:3, 3])
    return to_b(fixed), to_b(moving)


# ================================================================================================ roll sweep
def roll_sweep(knobs, cups, q_start, cfg, rolls=None, log=None, budget_s=4.0, stop_fn=None):
    """V3.68 ROLL SWEEP: cups = (src_rim, src_quat, dst_rim, dst_quat).  For each mount roll plan the stack with the
    real profile (narrow + time-limited wide search) -> rows {roll, ok, margin, grasp, place, binding, s}"""
    import NERO_V362_Cup_Skills as CK
    import NERO_V365_Real_Cup_Task as RT
    rolls = list(rolls if rolls is not None else range(-90, 91, 15))
    rows = []
    for r_ in rolls:
        if stop_fn and stop_fn():
            break
        prof = RT.real_profile(CK, cfg=dict(cfg or {}, roll_deg=float(r_)))
        prof.plan_time_budget_s = float(budget_s)
        t = time.time()
        try:
            p = CK.plan_cup_stack(knobs, cups[0], cups[1], cups[2], cups[3], q_start=q_start, profile=prof)
        except Exception as exc:
            p = {"ok": False, "reason": f"error {exc}"}
        row = {"roll": float(r_), "ok": bool(p.get("ok")), "s": round(time.time() - t, 1)}
        if p.get("ok"):
            row.update(margin=p["min_joint_margin_deg"], grasp=p["grasp_rim_angle_rel_deg"],
                       place=p["place_rim_angle_rel_deg"],
                       binding=p["grasp_binding"] if p["grasp_margin_deg"] <= p["place_margin_deg"] else p["place_binding"])
        else:
            row["why"] = p.get("reason")
        rows.append(row)
        if log:
            log(f"[INFO] [V3.68 ROLL SWEEP] 偏转 {r_:+.0f}°: " + (
                f"可规划 · 最小关节余量 {row['margin']}° ({row['binding']}) · 抓 {row['grasp']:+.0f}° 放 {row['place']:+.0f}°"
                if row["ok"] else f"规划不到（{row.get('why')}）") + f" · {row['s']} s")
    return rows
