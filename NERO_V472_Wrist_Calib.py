#!/usr/bin/env python3
"""NERO V4.7.2 WRIST CALIBRATION CONSISTENCY - fixes for the whole wrist-camera calibration life cycle
(intrinsics -> eye-in-hand -> save -> restart -> load -> quick check).  NERO_V325_Camera.py is frozen (hash-checked),
so the CameraTabUI methods are replaced at CLASS level by patch_class() before the console builds the Camera tab
(the buttons are then bound to the fixed methods).

Problems found (2026-11-10 audit) and what is done here:
 P1 a good eye-in-hand result was only auto-saved when LOO max <= 5.00 mm (one run had 5.07 mm) and only a log line
    said so -> nothing was saved, every restart loaded an OLD file.  Now: graded good / usable / not usable; a usable
    result asks "保存？" in a dialog; the conclusion is one clear log line.
 P2 「清除内参」 deleted only the version folder file; at the next start the newest copy in ~/Downloads/cameracalib
    came back.  Now a marker file config/wrist_intrinsics.cleared makes the clear stick until the next save.
 P3 the board-based eye-in-hand stored the BOARD centre as "marker", and the quick check looked for the ArUco marker
    (ID 0) from the board poses -> "未测到标记" / huge errors.  Now the saved file records its target (board / ArUco)
    and the quick check measures that target (the other one as fallback).
 P4 the quick check compared against the stored target position: a re-stuck marker failed a perfectly valid
    calibration.  Now it checks the CONSISTENCY between the poses (the camera calibration itself) and, when only the
    target moved, says so and updates the stored position.
 P5 the marker-based solver "self-calibrated" fx / k1 and replaced the measured 8-parameter intrinsics (also on disk)
    -> disabled while measured intrinsics are loaded (NERO_V386 SELF_CAL_ALLOWED).
 P6 the intrinsics auto collection planned with a fixed 130 deg lens (fx = 0.38 w) -> now the loaded / last saved
    intrinsics (NERO_V349 PLANNING_K_FN).
 P7 a hand-eye result made with OTHER intrinsics was used silently -> the intrinsics used are stored with the hand-eye
    result; a mismatch at start / after a new intrinsics save is reported.
 P8 nothing told which files were loaded -> one line each at start (file, time, fx / RMS, link7->cam, target).
 P9 PLANAR POSE AMBIGUITY: a board / marker seen nearly face-on has two poses with almost the same reprojection
    error; solvePnP(IPPE) can pick the mirror one (tilt off by tens of degrees, position right).  With the default
    8-parameter intrinsics the board path had 7 of 9 views mirrored in the V3.58 test -> 51 / 173 mm (now 0.18 mm);
    for the single 23.8 mm marker it is rarer (~4 % of the views) and changes little.  Now every view keeps both
    solutions (NERO_V349 planar_pose_candidates; ArUco samples get them here from their corners) and the solver
    picks per view the one that agrees with the robot motion (NERO_V349 disambiguate_planar_samples).
 P10 the wide-angle 8-parameter intrinsics were cut to 5 parameters when a coefficient exceeded 8 (k4..k6 cancel
    k1..k3, so the cut is wrong: 4 - 15 mm board position errors) -> removed in NERO_V349.
"""
from __future__ import annotations

import glob
import json
import math
import os
import time

VERSION = "V4.7.2"
MARKER = "V4.7.2 WRIST CALIBRATION CONSISTENCY"
USABLE_MEAN_MM = 3.0             # eye-in-hand residual mean
USABLE_LOO_MEAN_MM = 4.0         # leave-one-out mean
USABLE_LOO_MAX_MM = 8.0          # leave-one-out max
CHECK_SPREAD_MM = 5.0            # quick check: the target seen from the check poses agrees within this
CHECK_MOVED_MM = 5.0             # ... and moved more than this from the stored position = the target was moved
INTR_FX_TOL = 0.015              # hand-eye made with intrinsics within 1.5 % fx ...
INTR_PP_TOL_PX = 15.0            # ... and 15 px principal point = the same intrinsics
_ORIG = {}


# ============================================================================ helpers (no Tk)
def is_measured_intr(d):
    """a real (board) calibration - not the nominal 4MP values and not the V3.89 self-calibrated guess"""
    try:
        import NERO_V349_Real_Wrist_Camera as W49
        return bool(d) and W49.intrinsics_valid(d) and not d.get("nominal") and not d.get("self_calibrated")
    except Exception:
        return False


def intr_fingerprint(d):
    if not d or not d.get("K"):
        return None
    K = d["K"]
    return {"fx": round(float(K["fx"]), 2), "fy": round(float(K["fy"]), 2), "ppx": round(float(K["ppx"]), 2),
            "ppy": round(float(K["ppy"]), 2), "width": int(K.get("width") or 0), "height": int(K.get("height") or 0),
            "rms_px": d.get("rms_px"), "saved": d.get("saved"), "measured": is_measured_intr(d)}


def fingerprint_match(fp, d):
    """-> (True / False / None (unknown), text)"""
    if not fp:
        return None, "旧文件没有记录用的是哪组内参"
    cur = intr_fingerprint(d)
    if not cur:
        return False, "现在没有腕部内参"
    sx = float(cur["width"] or 1) / float(fp.get("width") or cur["width"] or 1)
    dfx = abs(cur["fx"] - fp["fx"] * sx) / max(1.0, fp["fx"] * sx)
    dpp = max(abs(cur["ppx"] - fp["ppx"] * sx), abs(cur["ppy"] - fp["ppy"] * sx))
    ok = dfx <= INTR_FX_TOL and dpp <= INTR_PP_TOL_PX
    return ok, (f"标定时 fx={fp['fx']:.1f}，现在 fx={cur['fx']:.1f}（差 {dfx * 100:.1f}%），主点差 {dpp:.0f} px")


def grade_eih(r):
    """-> ("good" | "usable" | "bad", reasons)"""
    if not r or not r.get("ok"):
        return "bad", ["没有结果"]
    why = []
    if r.get("good"):
        return "good", []
    if float(r.get("mean_mm", 99)) > USABLE_MEAN_MM:
        why.append(f"残差均值 {r['mean_mm']:.1f} mm > {USABLE_MEAN_MM:.0f} mm")
    if float(r.get("loo_mean_mm", 99)) > USABLE_LOO_MEAN_MM:
        why.append(f"留一均值 {r['loo_mean_mm']:.1f} mm > {USABLE_LOO_MEAN_MM:.0f} mm")
    if float(r.get("loo_max_mm", 99)) > USABLE_LOO_MAX_MM:
        why.append(f"留一最大 {r['loo_max_mm']:.1f} mm > {USABLE_LOO_MAX_MM:.0f} mm")
    if why:
        return "bad", why
    soft = [f"留一最大 {r['loo_max_mm']:.2f} mm（自动保存门槛 5 mm）"] if float(r.get("loo_max_mm", 0)) > 5.0 else []
    return "usable", soft + list(r.get("warnings") or [])


def grade_intr(r):
    if not r or not r.get("ok"):
        return "bad", ["没有结果"]
    if r.get("good"):
        return "good", []
    why = []
    rms = float(r.get("rms_px", 99))
    if rms > 0.8:
        return "bad", [f"重投影 RMS {rms:.2f} px > 0.8 px"]
    if int(r.get("n_used", 0)) < 8:
        why.append(f"只有 {r.get('n_used')} 张有效图（建议 ≥ 8）")
    return "usable", why + list(r.get("warnings") or [])


def target_point_base(calib, T_base_flange, p_cam):
    """the target centre in base_link seen from one pose with the hand-eye result"""
    import NERO_V325_Camera as CAM
    import NERO_V349_Real_Wrist_Camera as W49
    T_bc = W49.wrist_T_base_cam(T_base_flange, calib)
    s_xy = float(calib.get("pnp_scale") or 1.0)
    s_z = float(calib.get("pnp_scale_z") or s_xy)
    return CAM.T_apply(T_bc, [float(p_cam[0]) * s_xy, float(p_cam[1]) * s_xy, float(p_cam[2]) * s_z])


def check_consistency(calib, points):
    """points (base_link, m) of the same target from the check poses -> dict(spread_mm, moved_mm, ok, moved)"""
    import NERO_V325_Camera as CAM
    n = len(points)
    if n == 0:
        return {"n": 0, "ok": False, "moved": False, "spread_mm": None, "moved_mm": None, "mean": None}
    mean = [sum(p[i] for p in points) / n for i in range(3)]
    spread = max(CAM.vnorm(CAM.vsub(p, mean)) for p in points) * 1000.0
    ref = calib.get("marker_in_base")
    moved = CAM.vnorm(CAM.vsub(mean, ref)) * 1000.0 if ref else None
    if n >= 2:
        ok = spread <= CHECK_SPREAD_MM
    else:                                   # one view: only the stored position can be compared
        ok = moved is not None and moved <= CHECK_MOVED_MM
    return {"n": n, "ok": bool(ok), "spread_mm": round(spread, 2), "moved_mm": None if moved is None else round(moved, 2),
            "moved": bool(ok and n >= 2 and moved is not None and moved > CHECK_MOVED_MM), "mean": mean}


def calib_dir():
    try:
        import NERO_V325_Camera as CAM
        return CAM.CALIB_DIR
    except Exception:
        return os.environ.get("NERO_CALIB_DIR") or os.path.expanduser("~/Downloads/cameracalib")


def tombstone_path(intr_path):
    return os.path.join(os.path.dirname(intr_path), "wrist_intrinsics.cleared")


def newest(pattern):
    c = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return c[0] if c else None


def _load_json(p):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_json(p, d):
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1, default=str)
    os.replace(tmp, p)


def loaded_sources(tab):
    """which files the (original) autoload picked: (intr_path or None, handeye_path or None)"""
    W49 = tab.W49
    ip, hp = tab._wrist_intr_path(), tab._wrist_handeye_path()
    cd = calib_dir()
    src_i = ip if W49.intrinsics_valid(_load_json(ip) or {}) else None
    if src_i is None and ip == W49.DEFAULT_WRIST_INTR_PATH:
        for f in sorted(glob.glob(os.path.join(cd, "wrist_intrinsics*.json")), key=os.path.getmtime, reverse=True):
            if W49.intrinsics_valid(_load_json(f) or {}):
                src_i = f
                break
    src_h = hp if W49.handeye_valid(_load_json(hp) or {}) else None
    if src_h is None and hp == W49.DEFAULT_WRIST_HANDEYE_PATH:
        for f in sorted(glob.glob(os.path.join(cd, "wrist_handeye*.json")), key=os.path.getmtime, reverse=True):
            if W49.handeye_valid(_load_json(f) or {}):
                src_h = f
                break
    return src_i, src_h


def planning_K(tab, w, h):
    """V4.7.2 P6: the best KNOWN intrinsics for planning (loaded measured -> last saved on disk -> None)"""
    W49 = tab.W49
    d = tab.wrist_intr if is_measured_intr(tab.wrist_intr) else None
    if d is None:
        for f in [tab._wrist_intr_path()] + sorted(glob.glob(os.path.join(calib_dir(), "wrist_intrinsics*.json")),
                                                   key=os.path.getmtime, reverse=True):
            c = _load_json(f) or {}
            if is_measured_intr(c):
                d = c
                break
    if d is None or not W49.intrinsics_valid(d):
        return None
    K = d["K"]
    sx, sy = float(w) / float(K.get("width") or w), float(h) / float(K.get("height") or h)
    return {"width": w, "height": h, "fx": float(K["fx"]) * sx, "fy": float(K["fy"]) * sy,
            "ppx": float(K["ppx"]) * sx, "ppy": float(K["ppy"]) * sy}


# ============================================================================ patched CameraTabUI methods
def _wrist_autoload(self):
    _ORIG["_wrist_autoload"](self)
    W49 = self.W49
    W49.PLANNING_K_FN = lambda w, h, _t=self: planning_K(_t, w, h)          # P6
    ip = self._wrist_intr_path()
    tomb = tombstone_path(ip)
    if os.path.isfile(tomb) and not os.path.isfile(ip):                     # P2
        self.wrist_intr = None
    try:
        report_loaded(self)
    except Exception as exc:
        self.log(f"[WARN] [V4.7.2 wrist] 读标定时出错：{exc}")


def report_loaded(self):
    """P8 + P7: one line per file in use"""
    ip = self._wrist_intr_path()
    src_i, src_h = loaded_sources(self)
    if os.path.isfile(tombstone_path(ip)) and not os.path.isfile(ip):
        self.log("[INFO] [V4.7.2 wrist] 腕部内参：已被「清除」（不再读 cameracalib 里的旧副本）→ 先做「内参标定」")
    elif self.wrist_intr:
        d = self.wrist_intr
        K = d["K"]
        self.log(f"[INFO] [V4.7.2 wrist] 腕部内参 ← {src_i or '?'}（保存于 {d.get('saved', '?')} · "
                 f"fx={float(K['fx']):.1f} · {float(K['fx']) / float(K.get('width') or 1920):.2f}×画面宽 · "
                 f"RMS {d.get('rms_px', '?')} px · {len(d.get('dist') or [])} 个畸变参数）")
    else:
        self.log("[WARN] [V4.7.2 wrist] 没有腕部内参：会用理论值，标定和定位都不准 → 先做「内参标定」")
    c = self.wrist_calib
    if c:
        t = c.get("cam_in_flange") or [row[3] for row in c["T_flange_cam"][:3]]
        tgt = {"charuco": "标定板", "aruco": "ArUco 标记"}.get(c.get("target_type"), "未记录（旧文件）")
        self.log(f"[INFO] [V4.7.2 wrist] 腕部手眼 ← {src_h or '?'}（保存于 {c.get('saved', '?')} · link7→cam "
                 f"({t[0] * 1000:+.1f}, {t[1] * 1000:+.1f}, {t[2] * 1000:+.1f}) mm · 参照 {tgt}）")
        ok, txt = fingerprint_match(c.get("intrinsics_used"), self.wrist_intr)
        if ok is False:
            self.log(f"[WARN] [V4.7.2 wrist] 这份手眼标定是用另一组内参算的（{txt}）→ 结果会偏，请重做腕部手眼标定")
        elif ok is None:
            self.log(f"[INFO] [V4.7.2 wrist] {txt}，确认不了它和现在的内参是否匹配 → 建议做一次「快速复核」")
    else:
        self.log("[WARN] [V4.7.2 wrist] 没有已保存的腕部手眼标定")


def on_intr_save(self, path=None):
    _ORIG["on_intr_save"](self, path)
    ip = self._wrist_intr_path()
    try:
        if os.path.isfile(tombstone_path(ip)):
            os.remove(tombstone_path(ip))
    except OSError:
        pass
    c = self.wrist_calib
    if c and self.wrist_intr:
        ok, txt = fingerprint_match(c.get("intrinsics_used"), self.wrist_intr)
        if ok is False:
            self.log(f"[WARN] [V4.7.2 wrist] 内参变了（{txt}）：已保存的腕部手眼标定是用旧内参算的 → 请重做并保存腕部手眼标定")


def on_intr_clear_calib(self):
    _ORIG["on_intr_clear_calib"](self)
    p = tombstone_path(self._wrist_intr_path())
    try:
        with open(p, "w", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
        self.log("[INFO] [V4.7.2 wrist] 下次启动也不会再读 cameracalib 里的旧内参（直到你保存新的内参）")
    except OSError:
        pass


def on_intr_compute(self):
    _ORIG["on_intr_compute"](self)
    r = self.intr_result
    if r and r.get("ok"):
        g, why = grade_intr(r)
        self.log({"good": "[OK] [V4.7.2 intr] 结论：内参良好",
                  "usable": "[OK] [V4.7.2 intr] 结论：内参可以用" + (f"（{'；'.join(why)}）" if why else ""),
                  "bad": f"[FAIL] [V4.7.2 intr] 结论：内参不能用（{'；'.join(why)}）→ 重新采集"}[g])


def _pose_alts_for_sample(self, s, size_mm):
    """both planar solutions for a captured hand-eye sample (P9): the board detection carries them; an ArUco
    sample gets them from its 4 corners with the camera's K / dist (as V3.25 detect_marker solved it)"""
    try:
        import numpy as np
        import NERO_V349_Real_Wrist_Camera as W49
        d = getattr(self, "wdet", None) or {}
        if s.get("source") == "charuco" or d.get("target_type") == "charuco":
            return list(d.get("pose_alts") or [])
        c = s.get("corners_px")
        ws = self.wrist_src
        K = getattr(ws, "K", None) if ws is not None else None
        if not c or len(c) != 4 or not K:
            return []
        cv2 = W49._cv2()
        Km = np.array([[K["fx"], 0.0, K["ppx"]], [0.0, K["fy"], K["ppy"]], [0.0, 0.0, 1.0]], dtype=np.float64)
        dl = [float(v) for v in (getattr(ws, "dist", None) or [0.0] * 5)]
        dc = np.array(dl if len(dl) in (4, 5, 8, 12, 14) else (dl + [0.0] * 8)[:8], dtype=np.float64)
        uv = np.asarray(c, dtype=np.float64).reshape(4, 1, 2)
        uv_u = cv2.undistortPoints(uv, Km, dc, P=Km).reshape(4, 2) if np.any(np.abs(dc) > 1e-9) else uv.reshape(4, 2)
        h = float(size_mm) / 2000.0
        obj = np.array([[-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)
        alts = W49.planar_pose_candidates(obj, uv_u, Km)
        return [[list(a[0]), list(a[1]), round(a[2], 3)] for a in alts] if len(alts) >= 2 else []
    except Exception:
        return []


def _capture_wrist_sample(self, *a, **k):
    s, why = _ORIG["_capture_wrist_sample"](self, *a, **k)
    if s is not None:
        m = {}
        try:
            m = self._wmarker_from_ui() or dict(self.wmarker)
        except Exception:
            pass
        alts = _pose_alts_for_sample(self, s, float(m.get("size_mm") or 24.0))
        if alts:
            s["pose_alts"] = alts
    return s, why


def on_wcalib_compute(self):
    import NERO_V386_Real_Wrist_Camera as W86
    W86.SELF_CAL_ALLOWED = not is_measured_intr(self.wrist_intr)          # P5
    try:
        _ORIG["on_wcalib_compute"](self)
    finally:
        W86.SELF_CAL_ALLOWED = True
    r = self.wcalib_result
    if not r or not r.get("ok"):
        return
    g, why = grade_eih(r)
    nf = int(r.get("planar_flips_fixed") or 0)
    if nf:
        self.log(f"[INFO] [V4.7.2 eih] {nf} 个样本的标定板/标记姿态是“镜像解”（近乎正对时的二义性），已按机械臂运动改成正确的那个")
    self.log({"good": "[OK] [V4.7.2 eih] 结论：腕部手眼标定良好",
              "usable": "[OK] [V4.7.2 eih] 结论：腕部手眼标定可以用" + (f"（{'；'.join(why)}）" if why else ""),
              "bad": f"[FAIL] [V4.7.2 eih] 结论：腕部手眼标定不能用（{'；'.join(why)}）→ 重新采集，别保存"}[g])
    if g == "usable":                                                         # P1
        self._v472_prompt_token = tok = object()
        try:
            self.cv.after(400, lambda: _prompt_save(self, r, tok))
        except Exception as e:
            self.log(f"[WARN] [V4.7.2 eih] 无法弹出保存提示：{e}；请手动点「保存腕部标定」")


def _prompt_save(self, r, tok):
    if getattr(self, "_v472_prompt_token", None) is not tok or self.wcalib_result is not r:
        return                                            # a newer result came (intrinsics + hand-eye both computed)
    if self.wrist_calib is not None and self.wrist_calib.get("loo_max_mm") == r.get("loo_max_mm") and \
            self.wrist_calib.get("mean_mm") == r.get("mean_mm"):
        return                                            # already saved
    t = r["cam_in_flange"]
    msg = (f"腕部手眼标定结果可以用：\n残差 {r['mean_mm']:.2f} / {r['max_mm']:.2f} mm · 留一 {r['loo_mean_mm']:.2f} / "
           f"{r['loo_max_mm']:.2f} mm\nlink7→cam ({t[0] * 1000:+.1f}, {t[1] * 1000:+.1f}, {t[2] * 1000:+.1f}) mm\n\n"
           "不保存的话，下次打开软件还会用旧的标定。现在保存吗？")
    try:
        from tkinter import messagebox
        yes = messagebox.askyesno("保存腕部手眼标定？", msg, parent=self.cv.winfo_toplevel())
    except Exception as e:
        yes = False
    if yes:
        self.on_wcalib_save()
    else:
        self.log("[INFO] [V4.7.2 eih] 没有保存这次的腕部手眼标定（以后要用可在「手眼标定」页点「保存腕部标定」）")


def on_wcalib_save(self, path=None):
    r_before = self.wcalib_result
    samples = list(self.wcalib_samples or [])
    _ORIG["on_wcalib_save"](self, path)
    if not r_before or not r_before.get("ok") or self.wcalib_result is not None or self.wrist_calib is None:
        return                                            # the original refused (nothing computed) -> touch nothing
    tgt = "charuco" if samples and all(s.get("source") == "charuco" for s in samples) else "aruco"
    extra = {"target_type": tgt, "intrinsics_used": intr_fingerprint(self.wrist_intr), "v472": True}
    if tgt == "charuco":
        try:
            extra["board_cfg"] = self._board_cfg()
        except Exception:
            pass
    else:
        extra["marker"] = dict(self.wmarker)
    p = path or self._wrist_handeye_path()
    files = [p]
    cp = newest(os.path.join(calib_dir(), "wrist_handeye_*.json"))
    if cp and time.time() - os.path.getmtime(cp) < 10.0:
        files.append(cp)
    for f in files:
        d = _load_json(f)
        if d:
            d.update(extra)
            _save_json(f, d)
    self.wrist_calib = dict(self.wrist_calib, **extra)
    self.log(f"[OK] [V4.7.2 eih] 已记录：参照 {'标定板' if tgt == 'charuco' else 'ArUco 标记'} · 用的内参 "
             f"fx={extra['intrinsics_used']['fx'] if extra['intrinsics_used'] else '?'}（下次启动会核对）")


def _measure_board(self, calib, board_cfg):
    det, why = self._capture_intr_board_settled(board_cfg, timeout=1.4)
    if det is None:
        return None, why
    ws = self.wrist_src
    intr = self.wrist_intr or {}
    K = dict(ws.K) if getattr(ws, "K", None) else (intr.get("K") if intr else None)
    if not K:
        return None, "腕部相机没有内参"
    ss = self.W49.eih_samples_from_intr_samples([det], K, list(getattr(ws, "dist", None) or intr.get("dist") or []),
                                                board_cfg=board_cfg, model_type=intr.get("model_type"))
    if not ss:
        return None, "标定板角点不够"
    return ss[0], None


def _measure_marker(self, calib):
    ws = self.wrist_src
    m = dict((calib or {}).get("marker") or self.wmarker, target_type="aruco")
    pts, T = [], None
    t_end = time.time() + 1.6
    while time.time() < t_end and len(pts) < 5:
        q = self.actual_q()
        d = ws.detect_marker(m, q) if q else None
        if d and d.get("p_cam") is not None:
            pts.append(d["p_cam"])
            T = self.flange_T(q)
        time.sleep(0.1)
    if len(pts) < 2 or T is None:
        return None, f"腕部相机只看到 {len(pts)} 帧 ArUco 标记 ID {m.get('id')}"
    mean = [sum(p[i] for p in pts) / len(pts) for i in range(3)]
    return {"p_cam": mean, "T_base_flange": T}, None


def on_wcalib_check(self):
    """P3 + P4: measure the SAME target as the calibration; check the camera calibration by the agreement between
    the poses; a target that was only moved is reported and its stored position updated"""
    c = self.wrist_calib
    poses = (c or {}).get("check_poses_deg") or []
    ws = self.wrist_src
    if not c or not poses or not ws or not getattr(ws, "K", None):
        self.log("[WARN] [V3.51 eih] 需要已保存含检验姿态的腕部标定且腕部相机已连接")
        return
    if getattr(self.pipe, "teach_mode", False) or not getattr(self.jpanel, "enabled", False):
        self.log("[WARN] [V3.51 eih] 需要退出示教模式且 MoveIt 就绪")
        return
    import NERO_V325_Camera as CAM
    speed = min(CAM.CALIB_MAX_SPEED, max(0.01, float(self.cfg_fn().get("speed_percent", 10)) / 100.0))
    tgt = c.get("target_type")
    board_cfg = c.get("board_cfg") or self._board_cfg()
    order = {"charuco": ("charuco", "aruco"), "aruco": ("aruco", "charuco")}.get(tgt, ("aruco", "charuco"))

    def job():
        model = self._model()
        q0 = self.actual_q()
        if not q0 or not self.ask("腕部标定快速复核", f"机械臂将以 {speed * 100:.0f}% 速度依次走到 {len(poses)} 个检验姿态，"
                                  "从不同角度测同一个目标，核对腕部手眼标定（各姿态相差 ≤ 5 mm 为通过），结束后回到当前姿态。继续？"):
            return
        q0r = [math.radians(v) for v in q0]
        self.wcheck_state = ("run", [])
        self._ui(self._update_eih_ui)
        got = {"charuco": [], "aruco": []}
        for k, qd in enumerate(poses):
            if self.stop_flag:
                break
            res = self.pipe.move_and_settle(model, [math.radians(v) for v in qd], speed,
                                            label=f"腕部快速复核 {k + 1}/{len(poses)}")
            if not res.get("ok"):
                self.log(f"[WARN] [V4.7.2 eih] 复核姿态 {k + 1}：运动失败 {res.get('error')}")
                continue
            time.sleep(0.18)
            self._wait_still(1.2, hold_s=0.18)
            for kind in order:
                s, why = (_measure_board(self, c, board_cfg) if kind == "charuco" else _measure_marker(self, c))
                if s is not None:
                    got[kind].append(target_point_base(c, s["T_base_flange"], s["p_cam"]))
                    break
            else:
                self.log(f"[WARN] [V4.7.2 eih] 复核姿态 {k + 1}：标定板和 ArUco 标记都没看到")
        self.pipe.move_and_settle(model, q0r, speed, label="腕部快速复核 回原位")
        kind = max(got, key=lambda k_: len(got[k_]))
        pts = got[kind]
        name = "标定板" if kind == "charuco" else "ArUco 标记"
        ref_ok = kind == (tgt or kind)
        cc = check_consistency(c if ref_ok else dict(c, marker_in_base=None), pts)
        if cc["n"] == 0:
            self.wcheck_state = ("fail", [99.0])
            self.log("[WARN] [V3.51 eih] 腕部快速复核未通过 ✗：所有复核姿态都没看到标定板或 ArUco 标记"
                     "（目标被拿走了？把标定时用的标定板 / 标记放回原处再试）")
        elif cc["ok"]:
            self.wcheck_state = ("ok", [cc["spread_mm"]])
            msg = (f"[OK] [V3.51 eih] 腕部快速复核通过 ✓：{cc['n']} 个姿态看{name}，彼此相差 ≤ {cc['spread_mm']:.1f} mm")
            if cc["moved"]:
                upd = {"marker_in_base": cc["mean"]}
                if c.get("T_base_marker"):
                    Tm = [list(r_) for r_ in c["T_base_marker"]]
                    for i in range(3):
                        Tm[i][3] = cc["mean"][i]
                    upd["T_base_marker"] = Tm
                self.wrist_calib = dict(c, **upd)
                try:
                    p = self._wrist_handeye_path()
                    d = _load_json(p)
                    if d:
                        d.update(upd)
                        _save_json(p, d)
                except Exception:
                    pass
                msg += (f"；{name}比标定时挪动了 {cc['moved_mm']:.0f} mm（手眼标定本身没问题，已更新{name}位置）")
            self.log(msg)
        else:
            self.wcheck_state = ("fail", [cc["spread_mm"] or 99.0])
            self.log(f"[WARN] [V3.51 eih] 腕部快速复核未通过 ✗：{cc['n']} 个姿态看{name}，彼此相差 "
                     f"{cc['spread_mm']} mm（> {CHECK_SPREAD_MM:.0f} mm）→ 腕部手眼标定或内参不对，需要重做")
        self._ui(self._update_eih_ui)
    self._run(job)


PATCHES = ("_wrist_autoload", "on_intr_save", "on_intr_clear_calib", "on_intr_compute", "on_wcalib_compute",
           "on_wcalib_save", "on_wcalib_check", "_capture_wrist_sample")


def patch_class(C):
    """install the V4.7.2 methods on CameraTabUI (idempotent); call BEFORE the Camera tab is built"""
    if getattr(C, "_v472_patched", False):
        return
    g = globals()
    for name in PATCHES:
        _ORIG[name] = getattr(C, name)
        setattr(C, name, g[name])
    C._v472_patched = True
