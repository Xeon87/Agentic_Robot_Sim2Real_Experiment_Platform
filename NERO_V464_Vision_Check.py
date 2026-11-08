#!/usr/bin/env python3
"""NERO V4.6.4 VISION ACCURACY CHECK (Camera调试 -> 视觉定位 tab).

Is the vision really accurate?  The SO-101 gripper is CLOSED and its fingertip is brought straight above the centre
of a target (egg or egg-carton slot) found by the very same perception as the egg task (RealEggSkills.observe_scene
-> RealIO.perceive_eggs, the Agentic tab's YOLOE models / settings).  The person judges by eye and writes the offset
(dx / dy) into the panel; every point is saved (CSV + main / wrist camera photos).

Three ways to get the target position:
  ① main   - main camera (fixed D435i + its hand-eye calibration) only;
  ② wrist  - wrist camera (eye-in-hand calibration) only: the main camera position is used just to bring the wrist
             camera above the target; the tip then goes to the WRIST position (needs two wrist reads that agree);
  ③ assist - main camera, the wrist camera corrects it when two reads agree and the correction is <= ASSIST_MAX_MM
             (else the main camera position is kept; the reason is logged).

Fingertip model = the egg task's: real_mount (seat / ring / roll / fingertip correction of the Agentic tab).  With the
jaws closed the two tips meet at the fixed pad's inner face, i.e. in the grasp frame G at (0, -WALL_HALF,
+TIP_BEYOND_GRASP_M) (10 mm beyond the grasp origin along the approach).  Hover height = fingertip - target TOP
(egg: centre + half its width; slot: the slot rim height the task uses), default 10 mm.
Motion: joint move to 60 mm above the top (fence + IK checked), straight line down to the hover height (strict
arrival, line speed <= 20 mm/s), the egg task's fingertip fence (lowest fingertip >= table + 3 mm).
Records: NERO_CALIB_DIR/vision_check/vision_check_<date>.csv + <date>/ photos (never the release folder).
Tk is only touched on the Tk main thread (V4.6.3 rule): Tk values are read before a job starts, results come back
through a queue.
"""
from __future__ import annotations

import csv
import json
import math
import os
import queue
import threading
import time

import numpy as np

MARKER_V464 = "V4.6.4 VISION ACCURACY CHECK"
MODES = (("main", "① 只用主相机"), ("wrist", "② 只用腕部相机"), ("assist", "③ 主相机 + 腕部辅助"))
MODE_LABEL = dict(MODES)
HOVER_DEFAULT_MM = 10.0
HOVER_RANGE_MM = (5.0, 100.0)
APPROACH_ABOVE_M = 0.06            # first stop: fingertip 60 mm above the target top
LIFT_ABOVE_M = 0.08                # 「抬起」: fingertip 80 mm above the target top
LINE_SPEED_M_S = 0.02              # straight descent to the hover height
ASSIST_MAX_MM = 15.0               # ③: wrist correction accepted up to this
WRIST_AGREE_MM = 4.0               # two wrist reads must agree within this (same as the egg task)
# closed tips: the yaw does not change the tip point -> any reachable yaw, the one nearest the preferred first
YAW_TRY_DEG = tuple(sorted(range(-180, 180, 15), key=lambda d: (abs(d), d)))
CSV_FIELDS = ("time", "mode", "target", "kind", "hover_mm", "main_x_mm", "main_y_mm", "wrist_x_mm", "wrist_y_mm",
              "wrist_reads", "used_x_mm", "used_y_mm", "used_from", "decision", "target_top_z_mm",
              "cmd_tip_x_mm", "cmd_tip_y_mm", "cmd_tip_z_mm", "act_tip_x_mm", "act_tip_y_mm", "act_tip_z_mm",
              "arrive_err_mm", "user_dx_mm", "user_dy_mm", "user_note", "photo_main", "photo_wrist")


def calib_dir():
    return os.environ.get("NERO_CALIB_DIR") or os.path.expanduser("~/Downloads/cameracalib")


def record_dir():
    return os.path.join(calib_dir(), "vision_check")


# ============================================================================ geometry (pure)
def closed_tip_in_grasp():
    """closed SO-101 fingertip in the grasp frame G (x_G along the jaw, y_G closing direction, z_G approach)"""
    import NERO_V362_Cup_Skills as CK
    import NERO_V364_SO101_Gripper as G
    return np.array([0.0, -float(G.WALL_HALF), float(CK.TIP_BEYOND_GRASP_M)])


def grasp_R(yaw_deg):
    import NERO_V372_Sort_Skills as SK72
    return SK72.grasp_R(math.radians(float(yaw_deg)))


def tcp_for_tip(prof, p_tip, R_G):
    """TCP pose that puts the CLOSED fingertip at p_tip with the grasp frame orientation R_G"""
    p_G = np.asarray(p_tip, float) - np.asarray(R_G, float) @ closed_tip_in_grasp()
    return prof.tcp_from_grasp(p_G, np.asarray(R_G, float))


def tip_from_tcp_T(prof, T_tcp):
    """closed fingertip position from a TCP pose (4x4)"""
    T_G = np.asarray(T_tcp, float) @ np.linalg.inv(prof.T_grasp_tcp)
    return T_G[:3, :3] @ closed_tip_in_grasp() + T_G[:3, 3]


def target_top_z(entry, kind):
    """egg lying on the table: centre + half its width; slot: the rim height the egg task uses"""
    c = np.asarray(entry["rim"], float)
    if kind == "egg":
        return float(c[2]) + 0.5 * float(entry.get("width_m", 0.042))
    return float(c[2])


def agree_pair(reads, agree_mm=WRIST_AGREE_MM):
    """-> mean of the last two reads when they agree, else None"""
    if len(reads) < 2:
        return None
    a, b = np.asarray(reads[-1], float), np.asarray(reads[-2], float)
    if float(np.linalg.norm(a - b)) * 1000.0 > agree_mm:
        return None
    return 0.5 * (a + b)


def decide(mode, main_xy, wrist_xy, max_mm=ASSIST_MAX_MM):
    """-> (xy to use, source 'main'/'wrist'/None, decision text)"""
    m = np.asarray(main_xy, float)
    if mode == "main":
        return m, "main", "只用主相机"
    if mode == "wrist":
        if wrist_xy is None:
            return None, None, "腕部相机没有得到两次一致的可信读数 → 不悬停"
        return np.asarray(wrist_xy, float), "wrist", "只用腕部相机"
    if wrist_xy is None:
        return m, "main", "腕部相机没有两次一致的可信读数 → 用主相机"
    d = (np.asarray(wrist_xy, float) - m) * 1000.0
    n = float(np.linalg.norm(d))
    if n > max_mm:
        return m, "main", f"腕部修正 {n:.1f} mm > {max_mm:.0f} mm 上限 → 不采用，用主相机"
    return np.asarray(wrist_xy, float), "wrist", f"腕部修正 ({d[0]:+.1f}, {d[1]:+.1f}) mm → 采用腕部相机"


def axis_hint(cam, p_xyz):
    """which way base +X / +Y point in the main camera image at p (for writing dx / dy) -> text or ''"""
    try:
        K, T = cam["K"], np.asarray(cam["T_base_cam"], float)
        Ti = np.linalg.inv(T)

        def uv(p):
            q = Ti[:3, :3] @ np.asarray(p, float) + Ti[:3, 3]
            return np.array([K["fx"] * q[0] / q[2] + K["ppx"], K["fy"] * q[1] / q[2] + K["ppy"]])
        p = np.asarray(p_xyz, float)
        o = uv(p)
        out = []
        for name, d in (("+X", [0.05, 0, 0]), ("+Y", [0, 0.05, 0])):
            v = uv(p + np.asarray(d)) - o
            if abs(v[0]) >= abs(v[1]):
                out.append(f"{name} = 画面{'右' if v[0] > 0 else '左'}")
            else:
                out.append(f"{name} = 画面{'下' if v[1] > 0 else '上'}")
        return "主相机画面里：" + "，".join(out)
    except Exception:
        return ""


def summarize(rows):
    """per mode: n, mean dx / dy, mean |d|, max |d| of the rows the person filled in -> list of text lines"""
    out = []
    for key, lab in MODES:
        ds = []
        for r in rows:
            if r.get("mode") != key:
                continue
            try:
                ds.append((float(r["user_dx_mm"]), float(r["user_dy_mm"])))
            except (TypeError, ValueError, KeyError):
                continue
        if not ds:
            continue
        a = np.asarray(ds, float)
        n = np.linalg.norm(a, axis=1)
        out.append(f"{lab}：{len(ds)} 点 · 平均 dx {a[:, 0].mean():+.1f} / dy {a[:, 1].mean():+.1f} mm · "
                   f"平均偏 {n.mean():.1f} mm · 最大 {n.max():.1f} mm")
    return out


class RecordBook:
    """CSV of one day (NERO_CALIB_DIR/vision_check/vision_check_<date>.csv) + photo folder"""

    def __init__(self, root=None, day=None):
        self.root = root or record_dir()
        self.day = day or time.strftime("%Y%m%d")
        self.csv_path = os.path.join(self.root, f"vision_check_{self.day}.csv")
        self.photo_dir = os.path.join(self.root, self.day)

    def append(self, row):
        os.makedirs(self.root, exist_ok=True)
        new = not os.path.isfile(self.csv_path)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in CSV_FIELDS})
        return self.csv_path

    def rows(self):
        try:
            with open(self.csv_path, newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except OSError:
            return []

    def save_photo(self, rgb, name):
        if rgb is None:
            return ""
        try:
            os.makedirs(self.photo_dir, exist_ok=True)
            path = os.path.join(self.photo_dir, name)
            img = np.asarray(rgb)
            try:
                import cv2
                cv2.imwrite(path, np.ascontiguousarray(img[..., ::-1]))
            except Exception:
                from PIL import Image
                Image.fromarray(img.astype(np.uint8)).save(path)
            return path
        except Exception:
            return ""


def _mm(v):
    return None if v is None else round(float(v) * 1000.0, 1)


# ============================================================================ the measurement (worker thread)
class VisionCheck:
    """one measurement session on the real arm.  io: NERO_V365.RealIO, sk: NERO_V365.RealEggSkills on that io."""

    def __init__(self, io, sk, log=print, photo_fn=None, book=None):
        self.io, self.sk, self.log = io, sk, log
        self.photo_fn = photo_fn or (lambda: (None, None))
        self.book = book or RecordBook()
        self.targets = {}
        self.cam = None

    def observe(self):
        """task perception -> {id: {kind, xyz, top_z, yaw}}"""
        r = self.sk.observe_scene()
        if not r.get("ok"):
            raise RuntimeError(f"识别失败：{r.get('reason')}")
        self.cam = getattr(self.io, "last_cam", None)
        out = {}
        for tid, e in self.sk.cam.items():
            kind = "egg" if tid.startswith("egg_") else "slot"
            if kind == "slot" and e.get("occupied"):
                continue
            out[tid] = {"kind": kind, "xyz": np.asarray(e["rim"], float).copy(), "top_z": target_top_z(e, kind),
                        "yaw": float(e.get("yaw_deg", 0.0))}
        self.targets = out
        return out

    # -- motion helpers
    def _zt(self):
        return float((self.sk.last_obs or {}).get("table_z_m", 0.0))

    def _ik_tip(self, p_tip, yaw0):
        """-> (q, R_G) of the first reachable yaw (IK + joint path + fence incl. the lowest fingertip)"""
        for dy in YAW_TRY_DEG:
            R_G = grasp_R(yaw0 + dy)
            p, R = tcp_for_tip(self.io.prof, p_tip, R_G)
            q = self.sk._ik_reachable(p, R, self._zt())
            if q is not None:
                return q, R_G
        return None, None

    def _move_tip_joint(self, label, p_tip, yaw0):
        q, R_G = self._ik_tip(p_tip, yaw0)
        if q is None:
            return {"ok": False, "error": f"{label}：IK / 安全区不可达"}, None
        r = self.io.move_joints(label, q)
        return r, R_G

    def _tip_now(self):
        q = np.asarray(self.io.current_q(), float)
        return tip_from_tcp_T(self.io.prof, self.io.fast.fk(q))

    def close_gripper(self):
        g = getattr(self.io, "grip", None)
        if g is None or not g.connected:
            raise RuntimeError("SO-101 夹爪没有连接（Agentic实机任务 页 → 连接）")
        if not g.cfg.get("touch_calibrated"):
            raise RuntimeError("还没做「空夹标定」：不知道两指刚好合拢的角度（Agentic实机任务 页 → 空夹标定）")
        r = g.open_deg(0.0)                            # 0 deg = the calibrated touch point: tips together
        self.log(f"[INFO] [V4.6.4 VISION CHECK] 夹爪合拢：{json.dumps({k: v for k, v in r.items()}, ensure_ascii=False)}")
        return r

    # -- wrist reading
    def wrist_measure(self, tid):
        """look from SLOT_LOOK_POSES above the target (closed gripper pointing down), two agreeing trusted reads
        -> (xy or None, reads list, notes)"""
        import NERO_V365_Real_Cup_Task as RT
        import NERO_V386_Real_Wrist_Camera as RWC
        t = self.targets[tid]
        exp = np.asarray(t["xyz"], float)
        reads, notes = [], []
        for dx, dy, h in RT.SLOT_LOOK_POSES:
            self.io.check_stop()
            p_tip = np.array([exp[0] + dx, exp[1] + dy, t["top_z"] + h])
            r, _R = self._move_tip_joint(f"V4.6.4 wrist look {tid}", p_tip, t["yaw"])
            if not r.get("ok"):
                notes.append(r.get("error") or "移动失败")
                continue
            self.io.sleep(0.4)
            fr = self.sk._wrist_frame()
            if fr is None:
                notes.append("腕部相机未就绪（未连接或没做 eye-in-hand 标定）")
                break
            try:
                if t["kind"] == "slot":
                    chk = RWC.wrist_find_slot_cavity(fr["rgb"], fr["K"], self.sk._T_flange_now(), fr["calib_eih"], exp,
                                                     slot_pitch_m=self.sk._carton_pitch_m(),
                                                     undistort_intr=fr.get("intr"),
                                                     max_shift_m=RT.SLOT_WRIST_MAX_MM / 1000.0)
                else:
                    chk = RWC.wrist_check_egg_or_slot(fr["rgb"], fr["K"], self.sk._T_flange_now(), fr["calib_eih"],
                                                      exp, kind="egg", undistort_intr=fr.get("intr"))
            except Exception as exc:
                notes.append(f"识别出错 {exc}")
                continue
            if not chk.get("ok") or not chk.get("trusted"):
                notes.append(chk.get("reason") or "不可信")
                continue
            m = np.asarray(chk["measured_xyz"][:2], float)
            reads.append(m)
            d = (m - exp[:2]) * 1000.0
            self.log(f"[INFO] [V4.6.4 VISION CHECK] 腕部第 {len(reads)} 次读数 {tid}：相对主相机 ({d[0]:+.1f}, {d[1]:+.1f}) mm")
            xy = agree_pair(reads)
            if xy is not None:
                return xy, reads, notes
        return None, reads, notes

    # -- one check
    def run(self, mode, tid, hover_mm):
        if tid not in self.targets:
            raise RuntimeError(f"没有目标 {tid}：先点「识别目标」")
        t = self.targets[tid]
        hover_m = min(max(float(hover_mm), HOVER_RANGE_MM[0]), HOVER_RANGE_MM[1]) / 1000.0
        self.close_gripper()
        main_xy = np.asarray(t["xyz"][:2], float)
        wrist_xy, reads, notes = None, [], []
        if mode in ("wrist", "assist"):
            wrist_xy, reads, notes = self.wrist_measure(tid)
            if wrist_xy is None:
                self.log(f"[WARN] [V4.6.4 VISION CHECK] 腕部相机：{len(reads)} 次可信读数，没有两次一致"
                         f"（{'；'.join(notes[-3:]) or '-'}）")
        xy, src, decision = decide(mode, main_xy, wrist_xy)
        row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": mode, "target": tid, "kind": t["kind"],
               "hover_mm": round(hover_m * 1000.0, 1), "main_x_mm": _mm(main_xy[0]), "main_y_mm": _mm(main_xy[1]),
               "wrist_x_mm": None if wrist_xy is None else _mm(wrist_xy[0]),
               "wrist_y_mm": None if wrist_xy is None else _mm(wrist_xy[1]), "wrist_reads": len(reads),
               "used_from": src or "", "decision": decision, "target_top_z_mm": _mm(t["top_z"])}
        self.log(f"[INFO] [V4.6.4 VISION CHECK] {MODE_LABEL[mode]} · {tid}：{decision}")
        if xy is None:
            row["user_note"] = "没有悬停"
            return {"ok": False, "row": row, "error": decision}
        row.update(used_x_mm=_mm(xy[0]), used_y_mm=_mm(xy[1]))
        self.io.check_stop()
        above = np.array([xy[0], xy[1], t["top_z"] + APPROACH_ABOVE_M])
        r, R_G = self._move_tip_joint(f"V4.6.4 above {tid}", above, t["yaw"])
        if not r.get("ok"):
            return {"ok": False, "row": row, "error": r.get("error") or "移动到目标上方失败"}
        self.io.check_stop()
        p_tip = np.array([xy[0], xy[1], t["top_z"] + hover_m])
        p, R = tcp_for_tip(self.io.prof, p_tip, R_G)
        r = self.io.move_line(f"V4.6.4 hover {tid}", p, self.io.CK.R_to_quat_xyzw(R), LINE_SPEED_M_S, strict=True)
        if not r.get("ok"):
            return {"ok": False, "row": row, "error": r.get("error") or "下降到悬停高度失败"}
        self.io.sleep(0.5)
        act = self._tip_now()
        row.update(cmd_tip_x_mm=_mm(p_tip[0]), cmd_tip_y_mm=_mm(p_tip[1]), cmd_tip_z_mm=_mm(p_tip[2]),
                   act_tip_x_mm=_mm(act[0]), act_tip_y_mm=_mm(act[1]), act_tip_z_mm=_mm(act[2]),
                   arrive_err_mm=round(float(np.linalg.norm(act - p_tip)) * 1000.0, 1))
        m_rgb, w_rgb = self.photo_fn()
        stamp = time.strftime("%H%M%S")
        row["photo_main"] = self.book.save_photo(m_rgb, f"{stamp}_{mode}_{tid}_main.jpg")
        row["photo_wrist"] = self.book.save_photo(w_rgb, f"{stamp}_{mode}_{tid}_wrist.jpg")
        hint = axis_hint(self.cam, p_tip) if self.cam else ""
        self.log(f"[OK] [V4.6.4 VISION CHECK] 爪尖已停在 {tid} 顶端上方 {hover_m * 1000:.0f} mm（{decision}；"
                 f"到位误差 {row['arrive_err_mm']} mm）。请看爪尖偏离目标中心多少，填 dx / dy 后点「记录偏差」。{hint}")
        return {"ok": True, "row": row, "hint": hint}

    def lift(self, tid=None):
        t = self.targets.get(tid) if tid else None
        p = self._tip_now()
        z = (t["top_z"] if t else p[2]) + LIFT_ABOVE_M
        p_tip = np.array([p[0], p[1], max(p[2], z)])
        T = self.io.fast.fk(np.asarray(self.io.current_q(), float))
        T_G = T @ np.linalg.inv(self.io.prof.T_grasp_tcp)
        pt, R = tcp_for_tip(self.io.prof, p_tip, T_G[:3, :3])
        return self.io.move_line("V4.6.4 lift", pt, self.io.CK.R_to_quat_xyzw(R), LINE_SPEED_M_S * 2)


# ============================================================================ Tk panel (main thread only)
class VisionCheckPanel:
    def __init__(self, tk, ttk, parent, rct_tab, cam_tab, pipe, jpanel, log=None):
        import NERO_V314_Joint_Control as JC
        self.tk, self.ttk, self.rct, self.cam_tab, self.pipe, self.jpanel = tk, ttk, rct_tab, cam_tab, pipe, jpanel
        self.ext_log = log or (lambda s: None)
        self.q = queue.Queue()
        self.worker = None
        self.vc = None
        self.pending = None
        self.book = RecordBook()
        ttk.Separator(parent).pack(fill="x", pady=(3, 1))
        fr = ttk.Frame(parent)                         # no frame title line: the default 1420x1000 window is tight
        fr.pack(fill="x")
        self.frame = fr
        r1 = ttk.Frame(fr)
        r1.pack(fill="x", padx=4, pady=1)
        ttk.Label(r1, text="视觉精度验证（合爪对准）：", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        self.v_mode = tk.StringVar(value="main")
        for k, lab in MODES:
            ttk.Radiobutton(r1, text=lab, value=k, variable=self.v_mode).pack(side="left", padx=(0, 6))
        ttk.Label(r1, text=" 悬停高度").pack(side="left")
        self.v_hover = tk.StringVar(value=f"{HOVER_DEFAULT_MM:.0f}")
        ttk.Entry(r1, textvariable=self.v_hover, width=4).pack(side="left")
        ttk.Label(r1, text="mm").pack(side="left")
        # V4.6.4 (fix): two rows + one status line only.  The left camera monitors (777 px) set the Camera page height;
        # a taller 视觉定位 page squeezed the Camera log below the page to nothing.  Details go to the Camera log.
        r2 = ttk.Frame(fr)
        r2.pack(fill="x", padx=4, pady=1)
        self.btn_obs = ttk.Button(r2, text="识别目标", command=self.on_observe)
        self.btn_obs.pack(side="left")
        self.v_target = tk.StringVar(value="")
        self.cb_target = ttk.Combobox(r2, textvariable=self.v_target, values=[], width=8, state="readonly")
        self.cb_target.pack(side="left", padx=2)
        self.btn_go = ttk.Button(r2, text="▶ 合爪并悬停", command=self.on_go)
        self.btn_go.pack(side="left", padx=(4, 2))
        ttk.Button(r2, text="↑ 抬起", command=self.on_lift).pack(side="left", padx=2)
        ttk.Button(r2, text="■ 停止", command=self.on_stop).pack(side="left", padx=2)
        ttk.Label(r2, text="  爪尖−目标 dx").pack(side="left")
        self.v_dx = tk.StringVar(value="")
        ttk.Entry(r2, textvariable=self.v_dx, width=5).pack(side="left")
        ttk.Label(r2, text="dy").pack(side="left")
        self.v_dy = tk.StringVar(value="")
        ttk.Entry(r2, textvariable=self.v_dy, width=5).pack(side="left")
        ttk.Label(r2, text="mm").pack(side="left")
        self.v_note = tk.StringVar(value="")
        self.btn_rec = ttk.Button(r2, text="记录偏差", command=self.on_record)
        self.btn_rec.pack(side="left", padx=4)
        self.v_status = tk.StringVar(value="悬停高度 = 爪尖 − 目标顶端；dx / dy = 爪尖 − 目标中心（基座坐标，悬停后提示 +X / +Y 在主相机画面里的方向）")
        lab = JC.fixed_label(tk, fr, 1, textvariable=self.v_status, font=("TkDefaultFont", 9), wraplength=2000,
                             pack={"fill": "x", "padx": 4})
        self.v_sum = tk.StringVar(value=self._summary_text())
        self._poll()

    # ------------------------------------------------------------ plumbing
    def log(self, s):
        try:
            self.ext_log(str(s))
        except Exception:
            pass

    def _ui(self, fn):
        self.q.put(fn)

    def _say(self, text):
        """status line (one line) + the same text into the Camera log (Tk thread)"""
        self.v_status.set(text)
        self.log(f"[INFO] [V4.6.4 VISION CHECK] {text}")

    def _poll(self):
        try:
            while True:
                fn = self.q.get_nowait()
                try:
                    fn()
                except Exception as exc:
                    self.log(f"[WARN] [V4.6.4 VISION CHECK] UI: {exc}")
        except queue.Empty:
            pass
        try:
            self.frame.after(200, self._poll)
        except Exception:
            pass

    def busy(self):
        return self.worker is not None and self.worker.is_alive()

    def _summary_text(self):
        lines = summarize(self.book.rows())
        return ("  ".join(lines) if lines else "还没有填了 dx / dy 的记录") + \
            f"  · 记录文件 vision_check/{os.path.basename(self.book.csv_path)}"

    def _blocked(self, motion=True):
        if self.busy():
            return "上一个验证还在进行"
        if self.rct is None:
            return "「Agentic实机任务」页不可用（夹爪 / 指尖模型 / 识别都来自那一页）"
        if self.rct.busy():
            return "「Agentic实机任务」页正在运行任务"
        try:
            ycfg = self.rct.yolo_cfg()
            bad = self.rct.preflight("hover" if motion else "plan", ycfg)
        except Exception as exc:
            return str(exc)
        return "；".join(bad)

    def _make(self):
        """build RealIO + RealEggSkills from the Agentic tab's settings (Tk thread)"""
        import NERO_V365_Real_Cup_Task as RT
        mc = self.rct.motion_cfg()
        mc, _desc = RT.apply_speed(mc, 1.0)
        ycfg = self.rct.yolo_cfg()
        cam_tab = self.cam_tab
        arm = RT.PipelineArm(self.pipe, lambda: self.jpanel.model, log=self.log)
        io = RT.RealIO(arm, self.rct.grip if self.rct.grip.connected else None,
                       lambda: RT.camera_from_tab(cam_tab, yolo=ycfg), log=self.log, cfg=mc, motion_allowed=True,
                       wrist_fn=lambda: RT.wrist_from_tab(cam_tab))
        sk = RT.RealEggSkills(io)
        io.skills = sk

        def photos():
            src, w = getattr(cam_tab, "src", None), getattr(cam_tab, "wrist_src", None)
            m = src.frames()[0] if src is not None and src.running() else None
            wr = w.frames()[0] if w is not None and w.running() else None
            return m, wr
        return VisionCheck(io, sk, log=self.log, photo_fn=photos, book=self.book)

    def _start(self, name, fn):
        def wrap():
            try:
                fn()
            except Exception as exc:
                self.log(f"[FAIL] [V4.6.4 VISION CHECK] {name}: {exc}")
                self._ui(lambda e=exc: self._say(f"[失败] {name}：{e}"))
        self.worker = threading.Thread(target=wrap, daemon=True, name=f"v464_{name}")
        self.worker.start()

    # ------------------------------------------------------------ buttons
    def on_observe(self):
        why = self._blocked(motion=False)
        if why:
            self._say(f"[不能识别] {why}")
            return
        try:
            vc = self._make()
        except Exception as exc:
            self._say(f"[不能识别] {exc}")
            return
        self.vc = vc
        self._say("识别中 ...（鸡蛋任务同一套识别）")

        def job():
            tg = vc.observe()
            ids = sorted(tg, key=lambda k: (tg[k]["kind"] != "egg", int(k.split("_")[-1])))
            txt = "；".join(f"{k} ({tg[k]['xyz'][0] * 1000:+.0f}, {tg[k]['xyz'][1] * 1000:+.0f}) mm" for k in ids[:8])
            self.log(f"[OK] [V4.6.4 VISION CHECK] 目标：{txt}")

            def ui():
                self.cb_target.configure(values=ids)
                if ids and self.v_target.get() not in ids:
                    self.v_target.set(ids[0])
                self._say(f"识别到 {len(ids)} 个目标（鸡蛋 + 空槽）。选一个目标和测量方式，点「合爪并悬停」。")
            self._ui(ui)
        self._start("识别目标", job)

    def on_go(self):
        why = self._blocked(motion=True)
        if why:
            self._say(f"[不能运动] {why}")
            return
        if self.vc is None or not self.vc.targets:
            self._say("先点「识别目标」")
            return
        tid, mode = self.v_target.get(), self.v_mode.get()
        try:
            hover = float(self.v_hover.get())
        except ValueError:
            self._say("悬停高度请填数字（mm）")
            return
        if not (HOVER_RANGE_MM[0] <= hover <= HOVER_RANGE_MM[1]):
            self._say(f"悬停高度 {HOVER_RANGE_MM[0]:.0f}–{HOVER_RANGE_MM[1]:.0f} mm")
            return
        if tid not in self.vc.targets:
            self._say("选一个目标")
            return
        # V4.6.4 (fix): the gripper is taken from the Agentic tab NOW (it may have been connected after 识别目标)
        g = getattr(self.rct, "grip", None)
        self.vc.io.grip = g if (g is not None and g.connected) else None
        if self.vc.io.grip is None:
            self._say("[不能运动] SO-101 夹爪没有连接（Agentic实机任务 页 → 夹爪「连接」）")
            return
        from tkinter import messagebox
        if not messagebox.askyesno("视觉精度验证", f"{MODE_LABEL[mode]}：夹爪会合拢，机械臂慢速移到 {tid} 上方，"
                                                f"爪尖停在目标顶端上方 {hover:.0f} mm。\n"
                                                + ("先在目标上方几个位置用腕部相机看一看。\n" if mode != "main" else "")
                                                + "确认周围无人、急停在手边？", parent=self.frame.winfo_toplevel()):
            return
        self._save_pending("未填（开始了下一次）")
        vc = self.vc
        vc.io.stop_ev.clear()
        self._say(f"{MODE_LABEL[mode]} · {tid}：运行中 ...")

        def job():
            res = vc.run(mode, tid, hover)

            def ui():
                self.pending = res["row"]
                if res.get("ok"):
                    self.v_dx.set("")
                    self.v_dy.set("")
                    self.v_note.set("")
                    self._say(f"{MODE_LABEL[mode]} · {tid}：{res['row']['decision']} · 到位误差 "
                                      f"{res['row'].get('arrive_err_mm')} mm。看爪尖偏离目标中心多少，填 dx / dy → "
                                      f"「记录偏差」。{res.get('hint', '')}")
                else:
                    self._say(f"{MODE_LABEL[mode]} · {tid}：没有完成 —— {res.get('error')}")
                    self._save_pending(res.get("error") or "没有完成")
            self._ui(ui)
        self._start("合爪并悬停", job)

    def on_lift(self):
        if self.busy() or self.vc is None:
            return
        vc, tid = self.vc, self.v_target.get()
        vc.io.stop_ev.clear()
        self._start("抬起", lambda: vc.lift(tid))

    def on_stop(self):
        if self.vc is not None:
            self.vc.io.stop_ev.set()
        try:
            if self.pipe.move_busy():
                self.pipe.cancel_move()
        except Exception:
            pass
        self.log("[WARN] [V4.6.4 VISION CHECK] 用户停止")

    def _save_pending(self, note):
        if self.pending is None:
            return
        row = dict(self.pending, user_note=self.pending.get("user_note") or note)
        self.pending = None
        try:
            self.book.append(row)
        except Exception as exc:
            self.log(f"[WARN] [V4.6.4 VISION CHECK] 记录没写进去：{exc}")
        self.v_sum.set(self._summary_text())

    def on_record(self):
        if self.pending is None:
            self._say("还没有可记录的悬停结果（先「合爪并悬停」）")
            return
        try:
            dx, dy = float(self.v_dx.get()), float(self.v_dy.get())
        except ValueError:
            self._say("dx / dy 请填数字（mm，对准了就填 0）")
            return
        row = dict(self.pending, user_dx_mm=round(dx, 1), user_dy_mm=round(dy, 1), user_note=self.v_note.get().strip())
        self.pending = None
        try:
            path = self.book.append(row)
        except Exception as exc:
            self._say(f"[没有记录] {exc}")
            return
        self.log(f"[OK] [V4.6.4 VISION CHECK] 记录：{MODE_LABEL[row['mode']]} {row['target']} 偏差 dx {dx:+.1f} / dy {dy:+.1f} mm "
                 f"（使用 {row.get('used_from')}；{row.get('decision')}）→ {path}")
        self.v_sum.set(self._summary_text())
        self._say(f"已记录 {row['target']}：dx {dx:+.1f} / dy {dy:+.1f} mm。可以「抬起」再测下一个。汇总：{self.v_sum.get()}")


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def attach(tk, ttk, cam_tab, rct_tab, pipe, jpanel, log=None):
    """put the panel at the bottom of the Camera调试 -> 视觉定位 page (NERO_V325_Camera.py stays untouched)"""
    nb = getattr(cam_tab, "nb", None)
    if nb is None:
        raise RuntimeError("Camera调试 页没有子页面")
    page = None
    for t in nb.tabs():
        if nb.tab(t, "text") == "视觉定位":
            page = nb.nametowidget(t)
            break
    if page is None:
        raise RuntimeError("找不到「视觉定位」页")
    # room for the panel in the default 1420x1000 window: the page's recognition table (tv_eggs, 6 rows) shows 3 rows
    # (it still scrolls); only the widget option is changed at run time - NERO_V325_Camera.py stays as it is (frozen)
    tv = getattr(cam_tab, "tv_eggs", None)
    if tv is not None:
        try:
            if int(tv.cget("height")) > 3:
                tv.configure(height=3)
        except Exception:
            pass
    return VisionCheckPanel(tk, ttk, page, rct_tab, cam_tab, pipe, jpanel, log=log)
