"""NERO V3.79 camera lab - the Camera tab's 「相机高级测试」 window (V3.79 CAMERA LAB).

Pages (all on the Camera tab's live D435i stream, no second pipeline):
  ① 相机参数  lock exposure / gain / white balance / anti-flicker; saved to NERO_CALIB_DIR/d435i_profile.json and
              applied on every Camera-tab connect while locked (apply_saved_profile; depth preset stays the tab's)
  ② 深度噪声  the V3.78 filter sets on a still flat target (valid %, temporal std, flatness)
  ③ 定位精度  ChArUco board = frame (z up, table z = 0), egg on an inner corner, the Camera tab's YOLOE + YP.locate
  ④ 采集标注  photos into <root>/<class>/ with YOLOE auto labels (YOLO segmentation txt) + box-level review:
              click = select, drag inside = move, drag a corner = resize, drag on empty = new box, Delete = remove;
              the outline inside an edited box is recomputed (the image's YOLOE outline clipped to the box, else
              GrabCut inside the box, else the box itself)

Dataset layout (V3.80 trains on it):
  <root>/classes.json                      {"classes": [...]}  global class ids, append-only
  <root>/<class>/images/{train,val}/<stem>.jpg
  <root>/<class>/labels/{train,val}/<stem>.txt   "<id> x1 y1 x2 y2 ..." normalised polygon per object
  <root>/<class>/meta/<stem>.json          boxes / outlines in pixels, conf, source, reviewed, YOLOE originals
  <root>/<class>/depth/<stem>.png          aligned depth, uint16 mm (optional)
  every 5th image of a class goes to val (about 8:2)

Writes only NERO_CALIB_DIR (profile, results, lab settings) and the chosen data folder; never the release folder.

V3.80 CAMERA LAB UPGRADE:
  ① one-click auto tune (AWB settle -> exposure on whole flicker periods, longest without clipping in the work ROI,
     then the lowest gain that reaches the target brightness) + named profiles (NERO_CALIB_DIR/d435i_profiles/) + the
     one applied on connect + a lighting check after connect (brightness vs the saved reference)
  ② rows no longer overlap (Camera-tab Treeview style) + a verdict (vs the D435i theory at that distance, flatness,
     best setting vs the Camera tab's, scaled to the work distance)
  ③ any object: egg / ball shape (ellipsoid fit) or general object (locate_objects), truth = corner + half its height
  ④ no default class: a class list (common household objects offered) with ticks, the current class = folder, every
     ticked class labelled, counts per class against the recommended number of images
"""
import json
import math
import os
import re
import shutil
import threading
import time

import numpy as np

import NERO_V378_D435i_Bench as B378

VERSION = "V3.81"
MARKER = "V3.79 CAMERA LAB"
MARKER_V380 = "V3.80 CAMERA LAB UPGRADE"
MARKER_V381 = "V3.81 PROFILE FILES + TRAIN PAGE"
CALIB_DIR = B378.CALIB_DIR
LAB_SETTINGS = os.path.join(CALIB_DIR, "camera_lab.json")
DEFAULT_ROOT = B378.DATA_DIR
VAL_EVERY = 5                       # every 5th image -> val
HANDLE_PX = 9                       # corner grab radius (display pixels)
MIN_BOX_PX = 6
POWER_CHOICES = [("50 Hz", 1), ("60 Hz", 2)]
# V3.80 CAMERA LAB UPGRADE: constants
PROFILES_DIR = os.path.join(CALIB_DIR, "d435i_profiles")
DEFAULT_ROI = [0.2, 0.2, 0.8, 0.8]          # work area of the auto tune (fractions of the image)
MEAN_LO, MEAN_HI, MEAN_AIM = 110.0, 150.0, 130.0
P995_MAX, SAT_MAX = 245.0, 0.5
EXP_UNIT_MS = 0.1                            # D435 RGB exposure unit (0.1 ms)
FRAME_MS = 33.0                              # 30 fps
LIGHT_CHANGE = 0.20                          # connect check: brightness change that asks for a re-tune
DEPTH_F_PX = 1280 / 2.0 / math.tan(math.radians(87.0 / 2.0))   # D435 depth, 1280 wide, HFOV 87 deg
BASELINE_M = 0.050
SUBPIXEL = 0.08                              # Intel's typical sub-pixel RMS
WORK_DIST_MM = 750.0                         # fixed camera to the table (user's set-up 0.71-0.80 m)
RECOMMENDED_IMAGES = 150                     # per class (100-300)
COMMON_CLASSES = ["cup", "mug", "bottle", "bowl", "plate", "spoon", "fork", "knife", "banana", "apple", "orange",
                  "egg", "egg carton", "box", "toy block", "ball", "mouse", "keyboard", "cell phone", "remote",
                  "marker", "pen", "scissors", "tape", "book", "can", "sponge", "towel"]


# ============================================================================================ settings
def load_settings(path=None):
    d = {"root": DEFAULT_ROOT, "conf": 0.25, "save_depth": True, "burst_n": 10, "burst_dt": 1.0,
         "trials": 10, "noise_frames": 30,
         # V3.80 CAMERA LAB UPGRADE: no egg defaults
         "class_list": [], "checked": [], "current": "", "roi": list(DEFAULT_ROI),
         "acc_prompt": "", "acc_shape": "general", "acc_L": "", "acc_W": "", "acc_H": ""}
    try:
        with open(path or os.environ.get("NERO_V379_LAB_SETTINGS") or LAB_SETTINGS, encoding="utf-8") as f:
            old = json.load(f)
        if "class_list" not in old:                  # V3.79 file: its egg defaults are dropped
            old.pop("acc_prompt", None)
            pr = [p for p in str(old.pop("prompts", "") or "").split(",") if p.strip() and p.strip() != "egg"]
            old["class_list"] = [p.strip() for p in pr]
            old["checked"] = list(old["class_list"])
        d.update(old)
    except Exception:
        pass
    return d


def save_settings(d, path=None):
    p = path or os.environ.get("NERO_V379_LAB_SETTINGS") or LAB_SETTINGS
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(d, version=VERSION), f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    return p


def read_profile_meta(path=None):
    """V3.80: the extra keys of the active profile file (name, ref brightness, roi) that B378.load_profile drops"""
    try:
        with open(path or B378.PROFILE_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return {k: d.get(k) for k in ("name", "ref", "roi")}
    except Exception:
        return {}


def apply_saved_profile(device, rs, log=print, path=None, src=None):
    """Camera-tab connect: apply the lab's profile when it is locked; the depth preset stays the tab's own choice.
    V3.80: with src, check the lighting a moment later against the brightness saved with the profile.
    -> "locked" | "none" """
    prof = B378.load_profile(path)
    if not prof.get("locked"):
        return "none"
    meta = read_profile_meta(path)
    p = json.loads(json.dumps(prof))
    p["depth"].pop("visual_preset", None)
    rows = B378.apply_profile(device, p, rs, log=lambda s: None)
    bad = [r for r in rows if r[3] is None]
    log(f"[{'OK' if not bad else 'WARN'}] [V3.79 camera lab] 已套用锁定的相机参数（曝光 "
        f"{prof['color'].get('exposure')} · 增益 {prof['color'].get('gain')} · 白平衡 {prof['color'].get('white_balance')}"
        f" · 防闪烁 {B378.POWER_LINE.get(int(prof['color'].get('power_line_frequency', 0)), '?')}）"
        + (f"；没套上：{', '.join(r[0] + '.' + r[1] for r in bad)}" if bad else "")
        + (f"（参数组「{meta['name']}」）" if meta.get("name") else ""))
    if src is not None and (meta.get("ref") or {}).get("mean"):
        threading.Thread(target=lighting_check, args=(src, meta, log), daemon=True, name="v380_light").start()
    return "locked"


# ============================================================================================ V3.80 auto tune
def roi_px(roi, wh):
    w, h = wh
    r = roi or DEFAULT_ROI
    x0, x1 = sorted((min(max(r[0], 0.0), 1.0), min(max(r[2], 0.0), 1.0)))
    y0, y1 = sorted((min(max(r[1], 0.0), 1.0), min(max(r[3], 0.0), 1.0)))
    if x1 - x0 < 0.05 or y1 - y0 < 0.05:
        x0, y0, x1, y1 = DEFAULT_ROI
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def roi_stats(rgb, roi=None):
    """brightness of the work area: mean luma, 99.5 % luma (the bright parts), % of pixels with a saturated channel"""
    x0, y0, x1, y1 = roi_px(roi, (rgb.shape[1], rgb.shape[0]))
    sub = rgb[y0:y1:2, x0:x1:2].astype(np.float32)
    y = sub @ np.array([0.299, 0.587, 0.114], np.float32)
    return {"mean": round(float(y.mean()), 1), "p995": round(float(np.percentile(y, 99.5)), 1),
            "sat_pct": round(float((sub >= 250).any(axis=2).mean() * 100.0), 2)}


def judge_brightness(st, flicker_safe=True, gain=None, gain_max=128):
    """-> list of problems (empty = good) in plain words"""
    out = []
    if st["p995"] >= P995_MAX or st["sat_pct"] >= SAT_MAX:
        out.append(f"亮部过曝（最亮 0.5 % 的亮度 {st['p995']:.0f}，过曝 {st['sat_pct']} %）")
    if st["mean"] < MEAN_LO and st["p995"] < 0.9 * P995_MAX:      # dark AND headroom left (else: high contrast)
        out.append(f"偏暗（平均 {st['mean']:.0f} < {MEAN_LO:.0f}）"
                   + ("：增益已到上限，需要加灯" if gain is not None and gain >= 0.95 * gain_max else ""))
    elif st["mean"] > MEAN_HI:
        out.append(f"偏亮（平均 {st['mean']:.0f} > {MEAN_HI:.0f}）")
    if not flicker_safe:
        out.append("灯光太亮，曝光短于一个闪烁周期，画面可能有条纹（可以调暗灯或接受）")
    return out


def flicker_exposures(power_hz):
    """exposures (0.1 ms units) that are whole periods of the light flicker (twice the mains) and fit in a frame"""
    per = 1000.0 / (2.0 * float(power_hz)) / EXP_UNIT_MS
    return [int(round(k * per)) for k in (1, 2, 3) if k * per * EXP_UNIT_MS <= FRAME_MS]


def autotune(set_eg, measure, power_hz=50, gain_range=(0, 128), exp_min=1, log=print):
    """set_eg(exposure, gain) then measure() -> roi_stats.  Longest flicker-safe exposure without clipping, then the
    lowest gain that brings the mean to the target.  -> (result, trace)"""
    trace = []

    def run(e, g):
        set_eg(e, g)
        st = measure()
        trace.append(dict(exposure=e, gain=g, **st))
        return st

    gmin, gmax = gain_range
    cands = flicker_exposures(power_hz)
    chosen, fallback, flicker_safe = None, None, True
    for e in sorted(cands, reverse=True):
        st = run(e, gmin)
        clip = st["p995"] >= P995_MAX or st["sat_pct"] >= SAT_MAX
        if not clip and fallback is None:
            fallback = (e, st)
        if not clip and st["mean"] <= MEAN_HI:
            chosen = (e, st)
            break
    chosen = chosen or fallback
    if chosen is None:                        # clipping even at one flicker period: shorter than a period
        flicker_safe = False
        e = cands[0]
        while e > exp_min:
            e = max(exp_min, int(e // 2))
            st = run(e, gmin)
            if st["p995"] < P995_MAX and st["sat_pct"] < SAT_MAX:
                break
        chosen = (e, st)
    e, st = chosen
    g = gmin
    if st["mean"] < MEAN_LO:                  # raise the gain (bisection) without clipping
        lo, hi, best = gmin, gmax, (gmin, st)
        for _ in range(7):
            mid = int(round((lo + hi) / 2.0))
            if mid in (lo, hi) and hi - lo <= 1:
                break
            s2 = run(e, mid)
            if s2["p995"] < P995_MAX and s2["sat_pct"] < SAT_MAX and s2["mean"] <= MEAN_HI:
                best, lo = (mid, s2), mid
                if s2["mean"] >= MEAN_AIM - 10:
                    break
            else:
                hi = mid
            if hi - lo <= 2:
                break
        g, st = best
    set_eg(e, g)
    issues = judge_brightness(st, flicker_safe, g, gmax)
    note = ""
    if not issues and st["mean"] < MEAN_LO:
        note = (f"场景反差大：亮部已到上限，平均亮度只能到 {st['mean']:.0f}（以白色物体不过曝为先）")
    res = {"exposure": e, "gain": g, "stats": st, "flicker_safe": flicker_safe, "ok": not issues, "issues": issues,
           "note": note, "exposure_ms": round(e * EXP_UNIT_MS, 1), "steps": len(trace)}
    log(f"[{'OK' if not issues else 'WARN'}] [V3.80 auto tune] 曝光 {res['exposure_ms']} ms · 增益 {g} · 平均 "
        f"{st['mean']:.0f} · 最亮 {st['p995']:.0f} · 过曝 {st['sat_pct']} %（{len(trace)} 步）"
        + (f"：{'；'.join(issues)}" if issues else "：合格") + (f"（{note}）" if note else ""))
    return res, trace


def lighting_check(src, meta, log=print, wait_s=6.0):
    """after a connect: compare the work-area brightness with the one saved with the profile"""
    t0 = time.time()
    while time.time() - t0 < wait_s:
        rgb, _d, fid = src.frames()
        if rgb is not None and fid >= 20:
            break
        time.sleep(0.1)
    else:
        return None
    ref = meta["ref"]
    st = roi_stats(rgb, meta.get("roi") or ref.get("roi"))
    ch = (st["mean"] - ref["mean"]) / max(1.0, ref["mean"])
    if abs(ch) > LIGHT_CHANGE or st["sat_pct"] >= SAT_MAX:
        log(f"[WARN] [V3.80 camera lab] 光照和保存参数时不同（工作区亮度 {ref['mean']:.0f} → {st['mean']:.0f}，"
            f"{ch * 100:+.0f} %，过曝 {st['sat_pct']} %）：到「相机信息 → 相机高级测试 → ① 相机参数」重新一键调参")
    else:
        log(f"[OK] [V3.80 camera lab] 光照检查：工作区亮度 {st['mean']:.0f}（保存时 {ref['mean']:.0f}），参数合适")
    return st


# ---------------------------------------------------------------- V3.80 named profiles
def _profile_file(name):
    return os.path.join(PROFILES_DIR, safe_class(name) + ".json")


def list_named():
    if not os.path.isdir(PROFILES_DIR):
        return []
    out = []
    for fn in sorted(os.listdir(PROFILES_DIR)):
        if fn.endswith(".json"):
            try:
                with open(os.path.join(PROFILES_DIR, fn), encoding="utf-8") as f:
                    out.append(str(json.load(f).get("name") or fn[:-5]))
            except Exception:
                pass
    return out


def save_named(name, prof):
    os.makedirs(PROFILES_DIR, exist_ok=True)
    p = _profile_file(name)
    d = dict(json.loads(json.dumps(prof)), name=str(name), version=VERSION, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    return p


def load_named(name):
    with open(_profile_file(name), encoding="utf-8") as f:
        return json.load(f)


def save_profile_file(path, prof):
    """V3.81: a profile anywhere the user chooses (same format as the named profiles)"""
    d = dict(json.loads(json.dumps(prof)), version=VERSION, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    d.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def load_profile_file(path):
    """V3.81: read + check a profile file (named profiles, d435i_profile.json and V3.78 files all work)"""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    col = d.get("color")
    if not isinstance(col, dict) or "exposure" not in col:
        raise ValueError("不是相机参数文件（没有 color.exposure）")
    for k in ("depth", "filters"):
        d.setdefault(k, {})
    d.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return d


def profile_summary(d):
    c = d.get("color", {})
    ref = d.get("ref") or {}
    return (f"「{d.get('name')}」曝光 {c.get('exposure')}（{float(c.get('exposure', 0)) * EXP_UNIT_MS:.1f} ms）· 增益 "
            f"{c.get('gain')} · 白平衡 {c.get('white_balance')} · 防闪烁 "
            f"{B378.POWER_LINE.get(int(c.get('power_line_frequency', 0)), '?')}"
            + (f" · 保存时工作区亮度 {ref.get('mean')}" if ref.get("mean") is not None else "")
            + (f" · {d.get('saved_at')}" if d.get("saved_at") else ""))


def activate_named(name):
    """make the named profile the one the Camera tab applies on connect"""
    d = load_named(name)
    d["locked"] = True
    return B378.save_profile(d)


# ============================================================================================ V3.80 noise verdict
def theory_std_mm(dist_mm):
    """D435 depth RMS from triangulation: z^2 * subpixel / (f * baseline)"""
    z = float(dist_mm) / 1000.0
    return z * z * SUBPIXEL / (DEPTH_F_PX * BASELINE_M) * 1000.0


def noise_verdict(rows, current="spatial+temporal", work_mm=WORK_DIST_MM, sets=None):
    """-> {"level": ok / warn / fail, "lines": [...], "best": setting}"""
    by = {r["setting"]: r for r in rows}
    med = {n: m for n, _c, m in (sets or B378.FILTER_SETS)}
    raw = by.get("raw") or rows[0]
    lines, level = [], "ok"
    if not raw.get("mean_dist_mm") or raw.get("temporal_std_mm") is None:
        return {"level": "fail", "lines": ["没有测到深度：测量区没有有效深度（太近 < 0.3 m、太暗、黑色 / 反光表面？）"],
                "best": None}
    dist = raw["mean_dist_mm"]
    th = theory_std_mm(dist)
    ratio = raw["temporal_std_mm"] / th
    if raw["valid_pct"] < 95 or ratio > 2.5:
        level = "fail"
        lines.append(f"判定：异常。原始深度抖动 {raw['temporal_std_mm']:.2f} mm，是这个距离（{dist:.0f} mm）理论值 "
                     f"{th:.2f} mm 的 {ratio:.1f} 倍，有效像素 {raw['valid_pct']} %。可能是测量区反光 / 太暗 / 有东西在动 / "
                     "红外投射被挡；换哑光平面再测")
    elif raw["valid_pct"] < 98 or ratio > 1.6:
        level = "warn"
        lines.append(f"判定：偏大。原始抖动 {raw['temporal_std_mm']:.2f} mm（理论 {th:.2f} mm，{ratio:.1f} 倍），有效像素 "
                     f"{raw['valid_pct']} %")
    else:
        lines.append(f"判定：正常。原始抖动 {raw['temporal_std_mm']:.2f} mm，和 D435i 在 {dist:.0f} mm 的理论值 "
                     f"{th:.2f} mm 相符（{ratio:.1f} 倍），有效像素 {raw['valid_pct']} %")
    ok_rows = [r for r in rows if r.get("temporal_std_mm") is not None and r.get("valid_pct", 0) >= 99.0] or \
        [r for r in rows if r.get("temporal_std_mm") is not None]
    best = min(ok_rows, key=lambda r: r["temporal_std_mm"])
    if best.get("flatness_rms_mm") is not None and best["flatness_rms_mm"] > max(3.0 * best["temporal_std_mm"], 3.0):
        level = "warn" if level == "ok" else level
        lines.append(f"平整度 {best['flatness_rms_mm']:.2f} mm 远大于抖动 {best['temporal_std_mm']:.2f} mm：测量框里的表面"
                     "可能不平、有物体或反光（框应全在哑光桌面上）")
    cur = by.get(current)
    if cur and cur.get("temporal_std_mm") is not None:
        if best["setting"] == current:
            lines.append(f"建议：Camera 页现在的「{current}」已经是最好的（抖动 {cur['temporal_std_mm']:.2f} mm），不用改")
        else:
            gain = (cur["temporal_std_mm"] - best["temporal_std_mm"]) / cur["temporal_std_mm"] * 100.0
            cost = med.get(best["setting"], 1) * FRAME_MS
            if gain < 10.0:
                lines.append(f"建议：「{best['setting']}」比现在的「{current}」只好 {gain:.0f} %（{cur['temporal_std_mm']:.2f} → "
                             f"{best['temporal_std_mm']:.2f} mm），不必改")
            else:
                lines.append(f"建议：「{best['setting']}」比现在的「{current}」好 {gain:.0f} %（{cur['temporal_std_mm']:.2f} → "
                             f"{best['temporal_std_mm']:.2f} mm），代价是每次定位约等 {cost / 1000.0:.2f} s 取帧；"
                             "静止物体抓取前的定位值得用")
    if abs(dist - work_mm) > 150.0:
        k = (work_mm / dist) ** 2
        lines.append(f"距离：这次测在 {dist:.0f} mm，实际工作距离约 {work_mm:.0f} mm；换算过去抖动约为现在的 {k:.2f} 倍"
                     f"（原始 ≈ {raw['temporal_std_mm'] * k:.2f} mm，最好的处理 ≈ {best['temporal_std_mm'] * k:.2f} mm）")
    return {"level": level, "lines": lines, "best": best["setting"], "theory_mm": round(th, 2)}


# ============================================================================================ geometry
def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def norm_box(b, wh=None):
    x0, x1 = sorted((float(b[0]), float(b[2])))
    y0, y1 = sorted((float(b[1]), float(b[3])))
    if wh:
        x0, x1 = max(0.0, x0), min(float(wh[0]), x1)
        y0, y1 = max(0.0, y0), min(float(wh[1]), y1)
    return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]


def box_poly(b):
    return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]


def mask_to_poly(mask, eps_px=1.5, max_pts=200):
    """largest outer contour of a bool mask -> [[x, y], ...] (pixels) or None"""
    cv2 = B378._cv2()
    m = np.ascontiguousarray(mask.astype(np.uint8))
    cs, _h = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 4:
        return None
    eps = eps_px
    a = cv2.approxPolyDP(c, eps, True)
    while len(a) > max_pts:
        eps *= 1.5
        a = cv2.approxPolyDP(c, eps, True)
    if len(a) < 3:
        return None
    return [[float(p[0][0]), float(p[0][1])] for p in a]


def poly_mask(poly, hw):
    cv2 = B378._cv2()
    m = np.zeros(hw, np.uint8)
    if poly and len(poly) >= 3:
        cv2.fillPoly(m, [np.round(np.asarray(poly, float)).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def clip_poly(poly, box, hw):
    m = poly_mask(poly, hw)
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    keep = np.zeros(hw, bool)
    keep[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    return mask_to_poly(m & keep)


def grabcut_poly(rgb, box, iters=4):
    cv2 = B378._cv2()
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in norm_box(box, (w, h))]
    if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
        return None
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(np.ascontiguousarray(rgb[..., ::-1]), mask, (x0, y0, x1 - x0, y1 - y0), bgd, fgd, iters,
                    cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None
    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    if fg.sum() < 0.10 * (x1 - x0) * (y1 - y0):
        return None
    return mask_to_poly(fg)


def poly_area(poly):
    if not poly or len(poly) < 3:
        return 0.0
    a = np.asarray(poly, float)
    return 0.5 * abs(float(np.dot(a[:, 0], np.roll(a[:, 1], 1)) - np.dot(a[:, 1], np.roll(a[:, 0], 1))))


def recompute_outline(rgb, box, originals):
    """outline for an edited / new box -> (poly, how): the image's YOLOE outline clipped to the box (box IoU >= 0.5),
    else GrabCut inside the box, else the box itself"""
    h, w = rgb.shape[:2]
    box = norm_box(box, (w, h))
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    best = max(originals or [], key=lambda o: box_iou(box, o["box"]), default=None)
    if best is not None and box_iou(box, best["box"]) >= 0.5 and best.get("poly"):
        p = clip_poly(best["poly"], box, (h, w))
        if p and poly_area(p) >= 0.15 * area:
            return p, "yoloe"
    p = grabcut_poly(rgb, box)
    if p:
        return p, "grabcut"
    return box_poly(box), "box"


# ============================================================================================ dataset
def safe_class(name):
    s = re.sub(r'[\\/:*?"<>|\s]+', "_", str(name or "").strip()).strip("._")
    return s or "object"


def dets_to_anns(dets, prompts):
    """YOLOE detections -> annotations with the user's class names (pf names mapped onto the matching prompt)"""
    import NERO_V347_Yolo_Perception as YP
    out = []
    for d in dets:
        name = str(d.get("name", ""))
        cls = name if name in prompts else next((p for p in prompts if YP.match_object_name(name, [p])), None)
        if cls is None:
            continue
        poly = mask_to_poly(d["mask"]) if d.get("mask") is not None else None
        out.append({"cls": cls, "box": norm_box(d["box"]), "poly": poly or box_poly(norm_box(d["box"])),
                    "conf": round(float(d.get("conf", 0.0)), 3), "source": "auto"})
    return out


def label_lines(anns, cid_of, wh):
    w, h = float(wh[0]), float(wh[1])
    lines = []
    for a in anns:
        poly = a.get("poly") if a.get("poly") and len(a["poly"]) >= 3 else box_poly(a["box"])
        pts = " ".join(f"{min(max(x / w, 0.0), 1.0):.6f} {min(max(y / h, 0.0), 1.0):.6f}" for x, y in poly)
        lines.append(f"{cid_of(a['cls'])} {pts}")
    return lines


class Dataset:
    def __init__(self, root):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- classes
    def classes(self):
        try:
            with open(os.path.join(self.root, "classes.json"), encoding="utf-8") as f:
                return list(json.load(f).get("classes", []))
        except Exception:
            return []

    def class_id(self, name, add=True):
        cl = self.classes()
        if name in cl:
            return cl.index(name)
        if not add:
            return None
        cl.append(name)
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, "classes.json"), "w", encoding="utf-8") as f:
            json.dump({"version": VERSION, "classes": cl}, f, ensure_ascii=False, indent=1)
        return len(cl) - 1

    def folders(self):
        if not os.path.isdir(self.root):
            return []
        return sorted(d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d, "meta")))

    def cdir(self, folder):
        return os.path.join(self.root, safe_class(folder))

    # ---------------------------------------------------------------- items
    def _meta_path(self, folder, stem):
        return os.path.join(self.cdir(folder), "meta", stem + ".json")

    def items(self, folder):
        md = os.path.join(self.cdir(folder), "meta")
        out = []
        if os.path.isdir(md):
            for fn in sorted(os.listdir(md)):
                if fn.endswith(".json"):
                    try:
                        with open(os.path.join(md, fn), encoding="utf-8") as f:
                            out.append(json.load(f))
                    except Exception:
                        pass
        return out

    def next_split(self, folder):
        return "val" if len(self.items(folder)) % VAL_EVERY == VAL_EVERY - 1 else "train"

    def add_image(self, folder, rgb, anns, depth=None, extra=None):
        cv2 = B378._cv2()
        with self.lock:
            folder = safe_class(folder)
            d = self.cdir(folder)
            split = self.next_split(folder)
            n = len(self.items(folder))
            stem = f"{folder}_{time.strftime('%Y%m%d_%H%M%S')}_{n:04d}"
            while os.path.exists(self._meta_path(folder, stem)):
                n += 1
                stem = f"{folder}_{time.strftime('%Y%m%d_%H%M%S')}_{n:04d}"
            for sub in (f"images/{split}", f"labels/{split}", "meta"):
                os.makedirs(os.path.join(d, sub), exist_ok=True)
            img = os.path.join(d, "images", split, stem + ".jpg")
            cv2.imwrite(img, np.ascontiguousarray(rgb[..., ::-1]), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            dep = None
            if depth is not None:
                os.makedirs(os.path.join(d, "depth"), exist_ok=True)
                dep = os.path.join("depth", stem + ".png")
                cv2.imwrite(os.path.join(d, dep), np.clip(np.asarray(depth) * 1000.0, 0, 65535).astype(np.uint16))
            meta = dict({"version": VERSION, "stem": stem, "folder": folder, "split": split,
                         "image": os.path.join("images", split, stem + ".jpg"), "depth": dep,
                         "size": [int(rgb.shape[1]), int(rgb.shape[0])], "taken": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "reviewed": False, "anns": anns, "auto": json.loads(json.dumps(anns))}, **(extra or {}))
            self._write(folder, meta)
            return stem

    def _write(self, folder, meta):
        d = self.cdir(folder)
        lab = os.path.join(d, "labels", meta["split"], meta["stem"] + ".txt")
        os.makedirs(os.path.dirname(lab), exist_ok=True)
        lines = label_lines(meta["anns"], lambda c: self.class_id(c), meta["size"])
        with open(lab, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))      # empty file = background image
        tmp = self._meta_path(folder, meta["stem"]) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self._meta_path(folder, meta["stem"]))

    def load(self, folder, stem):
        with open(self._meta_path(folder, stem), encoding="utf-8") as f:
            meta = json.load(f)
        cv2 = B378._cv2()
        bgr = cv2.imread(os.path.join(self.cdir(folder), meta["image"]))
        return (None if bgr is None else np.ascontiguousarray(bgr[..., ::-1])), meta

    def save_anns(self, folder, stem, anns, reviewed=None):
        with self.lock:
            with open(self._meta_path(folder, stem), encoding="utf-8") as f:
                meta = json.load(f)
            meta["anns"] = anns
            if reviewed is not None:
                meta["reviewed"] = bool(reviewed)
            meta["edited"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._write(folder, meta)
            return meta

    def delete_image(self, folder, stem):
        with self.lock:
            with open(self._meta_path(folder, stem), encoding="utf-8") as f:
                meta = json.load(f)
            d = self.cdir(folder)
            for rel in (meta.get("image"), meta.get("depth"), os.path.join("labels", meta["split"], stem + ".txt"),
                        os.path.join("meta", stem + ".json")):
                if rel and os.path.isfile(os.path.join(d, rel)):
                    os.remove(os.path.join(d, rel))

    def class_counts(self):
        """V3.80: per class over every folder -> {cls: {"images": images with it, "objects": boxes}}"""
        out = {}
        for fo in self.folders():
            for m in self.items(fo):
                seen = set()
                for a in m.get("anns", []):
                    c = out.setdefault(a["cls"], {"images": 0, "objects": 0})
                    c["objects"] += 1
                    if a["cls"] not in seen:
                        c["images"] += 1
                        seen.add(a["cls"])
        return out

    def stats(self):
        out = {}
        for fo in self.folders():
            it = self.items(fo)
            out[fo] = {"images": len(it), "train": sum(1 for m in it if m.get("split") == "train"),
                       "val": sum(1 for m in it if m.get("split") == "val"),
                       "reviewed": sum(1 for m in it if m.get("reviewed")),
                       "objects": sum(len(m.get("anns", [])) for m in it)}
        return out


# ============================================================================================ box editing
class BoxEditor:
    """box-level review in IMAGE pixels; the canvas converts display <-> image with scale (display px per image px).
    press / drag / release -> the edited annotation index (outline to recompute) or None"""

    def __init__(self, anns, wh, default_cls="object"):
        self.anns = anns
        self.wh = wh
        self.sel = None
        self.default_cls = default_cls
        self._op = None

    def hit(self, x, y, scale=1.0):
        r = HANDLE_PX / max(scale, 1e-6)
        if self.sel is not None and self.sel < len(self.anns):
            b = self.anns[self.sel]["box"]
            for k, (cx, cy) in enumerate(((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3]))):
                if abs(x - cx) <= r and abs(y - cy) <= r:
                    return self.sel, f"corner{k}"
        inside = [i for i, a in enumerate(self.anns)
                  if a["box"][0] <= x <= a["box"][2] and a["box"][1] <= y <= a["box"][3]]
        if inside:
            i = min(inside, key=lambda j: (self.anns[j]["box"][2] - self.anns[j]["box"][0]) *
                    (self.anns[j]["box"][3] - self.anns[j]["box"][1]))       # the smallest box under the cursor
            return i, "move"
        return None, "new"

    def press(self, x, y, scale=1.0):
        i, what = self.hit(x, y, scale)
        if what == "new":
            self.sel = None
            self._op = {"what": "new", "x0": x, "y0": y, "box": [x, y, x, y], "moved": False}
        else:
            self.sel = i
            self._op = {"what": what, "x0": x, "y0": y, "box0": list(self.anns[i]["box"]), "moved": False}
        return self.sel

    def drag(self, x, y):
        op = self._op
        if not op:
            return
        op["moved"] = op["moved"] or abs(x - op["x0"]) + abs(y - op["y0"]) > 1.0
        if op["what"] == "new":
            op["box"] = [op["x0"], op["y0"], x, y]
            return
        b0, a = op["box0"], self.anns[self.sel]
        if op["what"] == "move":
            dx = min(max(x - op["x0"], -b0[0]), self.wh[0] - b0[2])
            dy = min(max(y - op["y0"], -b0[1]), self.wh[1] - b0[3])
            a["box"] = [b0[0] + dx, b0[1] + dy, b0[2] + dx, b0[3] + dy]
        else:
            k = int(op["what"][-1])
            b = list(b0)
            b[0 if k in (0, 3) else 2] = x
            b[1 if k in (0, 1) else 3] = y
            a["box"] = b

    def release(self, x, y):
        if not self._op:
            return None
        self.drag(x, y)
        op, self._op = self._op, None
        if op["what"] == "new":
            b = norm_box(op["box"], self.wh)
            if b[2] - b[0] < MIN_BOX_PX or b[3] - b[1] < MIN_BOX_PX:
                return None
            self.anns.append({"cls": self.default_cls, "box": b, "poly": None, "conf": None, "source": "manual"})
            self.sel = len(self.anns) - 1
            return self.sel
        a = self.anns[self.sel]
        a["box"] = norm_box(a["box"], self.wh)
        if not op["moved"]:
            a["box"] = op["box0"]
            return None
        if a["box"][2] - a["box"][0] < MIN_BOX_PX or a["box"][3] - a["box"][1] < MIN_BOX_PX:
            a["box"] = op["box0"]
            return None
        a["source"] = "edited" if a.get("source") != "manual" else "manual"
        return self.sel

    def delete(self):
        if self.sel is None or self.sel >= len(self.anns):
            return False
        self.anns.pop(self.sel)
        self.sel = None
        return True

    def set_class(self, cls):
        if self.sel is None or self.sel >= len(self.anns):
            return False
        self.anns[self.sel]["cls"] = cls
        return True


# ============================================================================================ measurements
def measure_noise(src, K, frames=30, log=print, sets=None):
    """the V3.78 filter sets through the Camera tab's reader thread -> rows"""
    w, h = int(K["width"]), int(K["height"])
    roi = [w // 2 - 80, h // 2 - 80, w // 2 + 80, h // 2 + 80]
    rows = []
    for name, fcfg, med in sets or B378.FILTER_SETS:
        cfg = dict(B378.load_profile()["filters"], **fcfg)
        on = any(cfg.get(k) for k in ("spatial", "temporal", "hole_filling"))
        make = (lambda rs, c=cfg: B378.FilterChain(rs, c).process) if on else None
        depths = src.collect_depth(make, 10 + int(frames) * int(med))[10:]       # first 10 = temporal warm-up
        stack = [B378.median_stack(depths[i:i + med]) for i in range(0, len(depths), med)] if med > 1 else depths
        rep = B378.noise_report(stack, K, roi)
        rows.append(dict(setting=name, frames=len(stack), **rep))
        log(f"[INFO] [V3.79 noise] {name}: 有效 {rep['valid_pct']}% · 抖动 {rep['temporal_std_mm']} mm · "
            f"平整度 {rep['flatness_rms_mm']} mm · 距离 {rep['mean_dist_mm']} mm")
    return rows, roi


SHAPES = [("general", "一般物体（通用定位）"), ("egg", "蛋形 / 球形（椭球拟合）")]


def locate_fn(shape, size_lw_mm=None):
    """V3.80: the Camera tab's locating for the shape -> fn(dets, depth, K, T) -> results"""
    import NERO_V347_Yolo_Perception as YP
    if shape == "egg":
        size = tuple(size_lw_mm) if size_lw_mm else YP.EGG_SIZE_MM
        return lambda d, dm, K, T: YP.locate(d, dm, K, T, table_z=0.0, size_mm=size)
    return lambda d, dm, K, T: YP.locate_objects(d, dm, K, T, table_z=0.0)


def accuracy_verdict(summ):
    """V3.80: one plain sentence per depth setting"""
    out = []
    for name, s_ in summ.items():
        if not isinstance(s_, dict) or "err_3d" not in s_:
            continue
        if not s_["err_3d"].get("n"):
            out.append(f"{name}：没有检出（物体名 / 置信度？物体在画面里？）")
            continue
        e, xy = s_["err_3d"], s_["xy"]
        good = xy["mean"] <= 3.0 and e["max"] <= 6.0 and s_["detected_pct"] >= 90
        out.append(f"{name}：检出 {s_['detected_pct']} %，水平误差平均 {xy['mean']} mm（最大 {xy['max']}），3D 平均 {e['mean']} mm"
                   f" → {'可以用于抓取（抓取一般要求水平 ≤ 3 mm）' if good else '偏大：看哪个方向偏（CSV 里 dx / dy / dz）'}")
    return "\n".join(out)


def accuracy_trials(grab, detect, K, T, corners, exp_z, trials=10, settings=(("滤波 + 5 帧中值", 5), ("滤波 1 帧", 1)),
                    locate=None, height_mm=None):
    """grab(n) -> (rgb, [depth]); detect(rgb) -> dets -> (rows, summary); same maths as the V3.78 bench.
    V3.80: locate = locate_fn(shape) (default the egg fit); height_mm = the object's real height -> height error"""
    import NERO_V347_Yolo_Perception as YP
    if locate is None:
        locate = lambda d, dm, K_, T_: YP.locate(d, dm, K_, T_, table_z=0.0)
    rows = []
    for name, med in settings:
        for i in range(int(trials)):
            rgb, depths = grab(med)
            if rgb is None or not depths:
                rows.append({"setting": name, "trial": i, "detected": False, "why": "no frame"})
                continue
            dets = detect(rgb)
            d_m = B378.median_stack(depths) if med > 1 else depths[-1]
            res = locate(dets, d_m, K, T)
            res = res[0] if isinstance(res, tuple) else res
            ok = [r for r in res if r.get("xyz_m") is not None and r.get("status") != "rejected"]
            if not ok:
                rows.append({"setting": name, "trial": i, "detected": False, "why": f"{len(dets)} det"})
                continue
            best = min(ok, key=lambda r: B378.accuracy_row(r["xyz_m"], corners, exp_z)["xy_err_mm"])
            rows.append(dict({"setting": name, "trial": i, "detected": True, "status": best.get("status"),
                              "x_mm": round(best["xyz_m"][0] * 1000, 2), "y_mm": round(best["xyz_m"][1] * 1000, 2),
                              "z_mm": round(best["xyz_m"][2] * 1000, 2)},
                             **B378.accuracy_row(best["xyz_m"], corners, exp_z)))
            if height_mm and best.get("height_mm") is not None:
                rows[-1]["h_err_mm"] = round(float(best["height_mm"]) - float(height_mm), 2)
    summ = {}
    for name, _m in settings:
        rr = [r for r in rows if r["setting"] == name]
        summ[name] = {"detected_pct": round(100.0 * sum(1 for r in rr if r.get("detected")) / max(1, len(rr)), 1),
                      "err_3d": B378.summarize(rr), "xy": B378.summarize(rr, "xy_err_mm"),
                      "dz": B378.summarize(rr, "dz_mm"), "h": B378.summarize(rr, "h_err_mm")}
    return rows, summ


def corner_pixels(T, K, corners):
    Tc = np.linalg.inv(T)
    out = {}
    for i, (x, y) in corners.items():
        pc = Tc @ np.array([x, y, 0.0, 1.0])
        if pc[2] > 1e-6:
            out[i] = (K["fx"] * pc[0] / pc[2] + K["ppx"], K["fy"] * pc[1] / pc[2] + K["ppy"])
    return out


def draw_overlay(rgb, anns=None, sel=None, points=None, scale=1.0, colours=None):
    """display image (RGB, already scaled by `scale`) with boxes / outlines / numbered points"""
    cv2 = B378._cv2()
    img = np.ascontiguousarray(rgb.copy())
    pal = colours or [(255, 64, 64), (64, 160, 255), (60, 200, 60), (255, 170, 0), (200, 80, 255), (0, 200, 200)]
    names = []
    for k, a in enumerate(anns or []):
        if a["cls"] not in names:
            names.append(a["cls"])
        c = pal[names.index(a["cls"]) % len(pal)]
        b = [int(round(v * scale)) for v in a["box"]]
        if a.get("poly"):
            pts = np.round(np.asarray(a["poly"], float) * scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, c, 1)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 255, 255) if k == sel else c, 3 if k == sel else 1)
        if k == sel:
            for (x, y) in ((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])):
                cv2.rectangle(img, (x - 4, y - 4), (x + 4, y + 4), (255, 255, 255), -1)
        tag = f"#{k} {_ascii(a['cls'])}" + (f" {a['conf']:.2f}" if a.get("conf") is not None else
                                            f" {_ascii(a.get('source', ''))}")   # V3.80: cv2 cannot draw Chinese
        cv2.putText(img, tag, (b[0] + 2, max(12, b[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(img, tag, (b[0] + 2, max(12, b[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    for i, (u, v) in (points or {}).items():
        p = (int(round(u * scale)), int(round(v * scale)))
        cv2.circle(img, p, 4, (0, 90, 255), -1)
        cv2.putText(img, str(i), (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 90, 255), 1)
    return img


def _ascii(t):
    t = str(t or "")
    return t if t.isascii() else ""


def fit_scale(wh, box_wh):
    return min(box_wh[0] / float(wh[0]), box_wh[1] / float(wh[1]))


def resize(rgb, scale):
    cv2 = B378._cv2()
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


# ============================================================================================ window
_OPEN = {}


def open_lab(tab):
    """「相机高级测试」: one window per Camera tab (raise it when already open)"""
    lab = _OPEN.get(id(tab))
    if lab is not None and lab.alive:
        try:
            lab.win.lift()
        except Exception:
            pass
        return lab
    lab = CameraLab(tab)
    _OPEN[id(tab)] = lab
    return lab


class CameraLab:
    PW, PH = 640, 360            # live preview
    RW, RH = 800, 450            # review canvas

    def __init__(self, tab):
        self.tab, self.tk, self.ttk = tab, tab.tk, tab.ttk
        tk, ttk = self.tk, self.ttk
        self.log = tab.log
        self.alive = True
        self.busy = None
        self.st = load_settings()
        self.prof = B378.load_profile()
        self.meta = read_profile_meta()                  # V3.80 CAMERA LAB UPGRADE
        self.last_tune = None
        self.board_T, self.board_info = None, None
        self.ds = Dataset(self.st["root"])
        self.cur = None                  # (folder, stem)
        self.cur_rgb = None
        self.editor = None
        self.photos = {}
        self.win = tk.Toplevel(getattr(tab, "root", None) or tab.cv)
        self.win.title(f"NERO {VERSION} 相机高级测试")
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.nb = ttk.Notebook(self.win)
        self.nb.pack(fill="both", expand=True)
        self._build_profile()
        self._build_noise()
        self._build_accuracy()
        self._build_capture()
        self.train_page = None
        try:                                              # V3.81 PROFILE FILES + TRAIN PAGE: ⑤ 训练 / 评估
            import NERO_V381_Yolo_Train as TR381
            self.train_page = TR381.TrainPage(self)
        except Exception as exc:
            self.log(f"[FAIL] [V3.81 train] 「⑤ 训练 / 评估」打不开：{exc}")
        try:
            self.nb.bind("<<NotebookTabChanged>>", lambda _e: self._tab_changed())
        except Exception:
            pass
        self.log(f"[INFO] [{MARKER}] 打开相机高级测试（数据文件夹 {self.ds.root}）")
        self._tick()

    # ------------------------------------------------------------------ helpers
    def _src(self):
        s = getattr(self.tab, "src", None)
        return s if s is not None and s.running() else None

    def _need_src(self):
        s = self._src()
        if s is None:
            self.log("[WARN] [V3.79 camera lab] 先在 Camera 调试页「连接」D435i")
        return s

    def _run(self, name, fn):
        if self.busy:
            self.log(f"[WARN] [V3.79 camera lab] 「{self.busy}」还在进行，等它结束")
            return False
        self.busy = name

        def job():
            try:
                fn()
            except Exception as exc:
                self.tab._ui(lambda e=exc: self.log(f"[FAIL] [V3.79 camera lab] {name}：{e}"))
            finally:
                self.busy = None
        threading.Thread(target=job, daemon=True, name=f"v379_{name}").start()
        return True

    def _show(self, canvas, key, rgb):
        tk = self.tk
        import NERO_V325_Camera as CAM
        data = CAM.ppm_bytes(np.ascontiguousarray(rgb))
        try:
            ph = tk.PhotoImage(data=data, format="PPM")
        except Exception:
            ph = tk.PhotoImage(data=CAM.png_bytes_from_ppm(data), format="PNG")
        self.photos[key] = ph
        canvas.delete("all")
        canvas.create_image(0, 0, image=ph, anchor="nw")

    def _tree_style(self):
        """V3.80: the Camera tab's Treeview style (row height from the font) -> rows never overlap"""
        try:
            import NERO_V325_Camera as CAM
            return CAM.treeview_style(self.ttk, self.win)[0]
        except Exception:
            return "Treeview"

    def _var(self, value):
        return self.tk.StringVar(value=str(value))

    def _fget(self, var, lo, hi, name):
        try:
            v = float(var.get())
            assert lo <= v <= hi
            return v
        except Exception:
            raise ValueError(f"{name} 要在 {lo:g}–{hi:g}")

    def _save_st(self):
        try:
            save_settings(self.st)
        except Exception as exc:
            self.log(f"[WARN] [V3.79 camera lab] 设置没保存：{exc}")

    def close(self):
        self.alive = False
        self._save_st()
        _OPEN.pop(id(self.tab), None)
        try:
            self.win.destroy()
        except Exception:
            pass

    def _tab_changed(self):
        if self._page() == 4 and self.train_page is not None:      # V3.81: ⑤ shows the newest captures
            self._set_root(self.v_root.get())
            self.train_page.refresh()

    def _tick(self):
        if not self.alive:
            return
        if getattr(self, "train_page", None) is not None:
            try:
                self.train_page.poll()
            except Exception as exc:
                self.log(f"[WARN] [V3.81 train] 进度读取：{exc}")
        try:
            self._refresh()
        except Exception as exc:
            if getattr(self, "_last_err", None) != str(exc):
                self._last_err = str(exc)
                self.log(f"[WARN] [V3.79 camera lab] 画面刷新：{exc}")
        try:
            self.win.after(200, self._tick)
        except Exception:
            self.alive = False

    def _page(self):
        try:
            return int(self.nb.index(self.nb.select()))
        except Exception:
            return 0

    def _refresh(self):
        src = self._src()
        rgb = src.frames()[0] if src is not None else None
        page = self._page()
        if rgb is None:
            return
        canvas = {0: self.cv_prof, 1: self.cv_noise, 2: self.cv_acc, 3: self.cv_live}.get(page)
        if canvas is None:
            return
        s = fit_scale((rgb.shape[1], rgb.shape[0]), (self.PW, self.PH) if page != 3 else (480, 270))
        small = resize(rgb, s)
        if page == 0:                                   # V3.80: work-ROI brightness + the ROI drawn
            stt = roi_stats(rgb, self.st.get("roi"))
            bad = judge_brightness(stt)
            self.v_stats.set(f"工作区：平均 {stt['mean']:.0f}（110–150） · 最亮 {stt['p995']:.0f}（< 245） · 过曝 "
                             f"{stt['sat_pct']} %（< 0.5 %）  " + ("✓ 合适" if not bad else "← " + "；".join(bad)))
            self._prof_scale_wh = (small.shape[1], small.shape[0])
            x0, y0, x1, y1 = roi_px(self.st.get("roi"), (small.shape[1], small.shape[0]))
            small = draw_overlay(small, [{"cls": "work ROI", "box": [x0, y0, x1, y1], "poly": None, "conf": None,
                                          "source": ""}])
            op = getattr(self, "_roi_op", None)
            if op:
                small = draw_overlay(small, [{"cls": "new", "box": norm_box(op), "poly": None, "conf": None,
                                              "source": ""}])
        elif page == 1:
            w, h = small.shape[1], small.shape[0]
            r = 80 * s
            small = draw_overlay(small, [{"cls": "ROI", "box": [w / 2 - r, h / 2 - r, w / 2 + r, h / 2 + r],
                                          "poly": None, "conf": None, "source": "measure"}])
        elif page == 2 and self.board_T is not None and src.K:
            small = draw_overlay(small, points=corner_pixels(self.board_T, src.K, B378.inner_corners()), scale=s)
        self._show(canvas, f"p{page}", small)

    # ------------------------------------------------------------------ ① camera parameters
    def _build_profile(self):
        # V3.80 CAMERA LAB UPGRADE: one-click auto tune in a work ROI + named profiles
        tk, ttk = self.tk, self.ttk
        p = ttk.Frame(self.nb, padding=6)
        self.nb.add(p, text="① 相机参数")
        self.cv_prof = tk.Canvas(p, width=self.PW, height=self.PH, bg="#202020", highlightthickness=0)
        self.cv_prof.pack(anchor="w")
        self.cv_prof.bind("<ButtonPress-1>", self._roi_press)
        self.cv_prof.bind("<B1-Motion>", self._roi_drag)
        self.cv_prof.bind("<ButtonRelease-1>", self._roi_release)
        self.v_stats = self._var("（连接相机后显示工作区亮度；在画面上拖框 = 工作区）")
        self.lb_stats = ttk.Label(p, textvariable=self.v_stats)
        self.lb_stats.pack(anchor="w", pady=(4, 2))
        r0 = ttk.Frame(p)
        r0.pack(fill="x", pady=2)
        col = self.prof["color"]
        pl = int(col.get("power_line_frequency", 1))
        self.v_pl = self._var(next((n for n, v in POWER_CHOICES if v == pl), "50 Hz"))
        ttk.Label(r0, text="电网频率").pack(side="left")
        ttk.Combobox(r0, textvariable=self.v_pl, values=[n for n, _ in POWER_CHOICES], width=7,
                     state="readonly").pack(side="left", padx=(2, 8))
        ttk.Button(r0, text="▶ 一键自动调参", command=self.on_autotune).pack(side="left")
        ttk.Button(r0, text="工作区恢复默认", command=self.on_roi_default).pack(side="left", padx=6)
        self.v_tune = self._var("")
        self.lb_tune = ttk.Label(p, textvariable=self.v_tune, justify="left", wraplength=self.PW + 300)
        self.lb_tune.pack(anchor="w", pady=(2, 2))
        r = ttk.Frame(p)
        r.pack(fill="x", pady=2)
        self.v_exp, self.v_gain, self.v_wb = self._var(col["exposure"]), self._var(col["gain"]), self._var(col["white_balance"])
        ttk.Label(r, text="手动：曝光(0.1 ms)").pack(side="left")
        ttk.Entry(r, textvariable=self.v_exp, width=6).pack(side="left", padx=(2, 8))
        ttk.Label(r, text="增益").pack(side="left")
        ttk.Entry(r, textvariable=self.v_gain, width=5).pack(side="left", padx=(2, 8))
        ttk.Label(r, text="白平衡 K").pack(side="left")
        ttk.Entry(r, textvariable=self.v_wb, width=6).pack(side="left", padx=(2, 8))
        ttk.Button(r, text="应用", command=self.on_apply).pack(side="left")
        r2 = ttk.Frame(p)
        r2.pack(fill="x", pady=(6, 2))
        self.v_pname = self._var(self.meta.get("name") or "")
        ttk.Label(r2, text="参数组").pack(side="left")
        self.cb_pname = ttk.Combobox(r2, textvariable=self.v_pname, values=list_named(), width=18,
                                     postcommand=lambda: self.cb_pname.configure(values=list_named()))
        self.cb_pname.pack(side="left", padx=2)
        ttk.Button(r2, text="保存为这一组", command=self.on_save_profile).pack(side="left", padx=2)
        ttk.Button(r2, text="套用这一组", command=self.on_load_named).pack(side="left", padx=2)
        self.v_auto = self.tk.BooleanVar(value=True)
        ttk.Checkbutton(r2, text="设为连接时自动套用", variable=self.v_auto).pack(side="left", padx=6)
        r3 = ttk.Frame(p)                                 # V3.81 PROFILE FILES: any folder, any file name
        r3.pack(fill="x", pady=(2, 2))
        ttk.Label(r3, text="参数文件").pack(side="left")
        ttk.Button(r3, text="另存为文件…", command=self.on_save_file).pack(side="left", padx=2)
        ttk.Button(r3, text="打开文件…", command=self.on_open_file).pack(side="left", padx=2)
        ttk.Label(r3, text="（存到你选的文件夹、自己起文件名；打开后立即套用，勾选了上面的选项也会设为连接时自动套用）",
                  foreground="#555").pack(side="left", padx=4)
        self.v_prof = self._var(self._prof_text())
        ttk.Label(p, textvariable=self.v_prof, justify="left", foreground="#335").pack(anchor="w", pady=(4, 0))
        ttk.Label(p, text="目标：工作区平均亮度 110–150、最亮 0.5 % 的像素 < 245（白色物体不过曝）、增益尽量低、"
                          "曝光取灯光闪烁周期的整数倍（50 Hz 电网 = 10 ms）。\n"
                          "先在画面上拖框圈出桌面工作区（避开窗户 / 屏幕）→「一键自动调参」（约 10 s）→ 合格后起个名字"
                          "「保存为这一组」。\n参数组存在 cameracalib/d435i_profiles/；勾选「设为连接时自动套用」的那一组，"
                          "Camera 页每次连接都会套用并检查光照有没有变。",
                  foreground="#555", justify="left").pack(anchor="w", pady=(4, 0))

    def _prof_text(self, rb=None):
        c = self.prof["color"]
        meta = read_profile_meta()
        s = (f"连接时自动套用：{'「' + meta['name'] + '」' if meta.get('name') else '（未命名）'} · 曝光 "
             f"{c.get('exposure')} · 增益 {c.get('gain')} · 白平衡 {c.get('white_balance')} · 防闪烁 "
             f"{B378.POWER_LINE.get(int(c.get('power_line_frequency', 0)), '?')}"
             if self.prof.get("locked") else "连接时自动套用：没有（还没保存过参数组）")
        if rb:
            cc = rb.get("color", {})
            s += (f"\n相机当前：曝光 {cc.get('exposure')} · 增益 {cc.get('gain')} · 白平衡 {cc.get('white_balance')} · "
                  f"自动曝光 {cc.get('enable_auto_exposure')} · 自动白平衡 {cc.get('enable_auto_white_balance')} · 防闪烁 "
                  f"{B378.POWER_LINE.get(int(cc.get('power_line_frequency', -1)), '?')}")
        return s

    # ---- work ROI (drag on the preview)
    def _roi_press(self, ev):
        self._roi_op = [ev.x, ev.y, ev.x, ev.y]

    def _roi_drag(self, ev):
        if getattr(self, "_roi_op", None):
            self._roi_op[2:] = [ev.x, ev.y]

    def _roi_release(self, ev):
        op, self._roi_op = getattr(self, "_roi_op", None), None
        if not op:
            return
        op[2:] = [ev.x, ev.y]
        sc = getattr(self, "_prof_scale_wh", (self.PW, self.PH))
        x0, x1 = sorted((op[0] / sc[0], op[2] / sc[0]))
        y0, y1 = sorted((op[1] / sc[1], op[3] / sc[1]))
        if x1 - x0 < 0.05 or y1 - y0 < 0.05:
            return
        self.st["roi"] = [round(min(max(v, 0.0), 1.0), 3) for v in (x0, y0, x1, y1)]
        self.log(f"[INFO] [V3.80 camera lab] 工作区 = 画面 {self.st['roi']}（比例）")

    def on_roi_default(self):
        self.st["roi"] = list(DEFAULT_ROI)

    def _manual_colour(self):
        col = self.prof["color"]
        col["exposure"] = self._fget(self.v_exp, 1, 10000, "曝光")
        col["gain"] = self._fget(self.v_gain, 0, 128, "增益")
        col["white_balance"] = self._fget(self.v_wb, 2800, 6500, "白平衡")
        col["power_line_frequency"] = dict(POWER_CHOICES).get(self.v_pl.get(), 1)
        col.update(enable_auto_exposure=0, enable_auto_white_balance=0)

    def _apply_now(self, prof):
        src = self._need_src()
        if src is None:
            return None
        p = json.loads(json.dumps(prof))
        p.setdefault("depth", {}).pop("visual_preset", None)
        return src.run_on_device(lambda dev, rs: B378.apply_profile(dev, p, rs, log=lambda m: None))

    def on_apply(self):
        try:
            self._manual_colour()
        except ValueError as exc:
            self.log(f"[WARN] [V3.79 camera lab] {exc}")
            return
        self._run("应用相机参数", lambda: (self._apply_now(self.prof), self.tab._ui(self.on_read_back)))

    def _measure_roi(self, n=3, settle_s=0.35):
        """work-area brightness after new settings: wait for them to take effect, average a few new frames"""
        src = self._src()
        time.sleep(settle_s)
        sts, last, t0 = [], None, time.time()
        while len(sts) < n and time.time() - t0 < 3.0:
            rgb, _d, fid = src.frames()
            if rgb is not None and fid != last:
                sts.append(roi_stats(rgb, self.st.get("roi")))
                last = fid
            else:
                time.sleep(0.02)
        if not sts:
            raise RuntimeError("没有新画面")
        return {k: round(float(np.mean([s_[k] for s_ in sts])), 2) for k in sts[0]}

    def on_autotune(self):
        src = self._need_src()
        if src is None:
            return
        hz = 50 if self.v_pl.get().startswith("50") else 60
        col = self.prof["color"]

        def job():
            col["power_line_frequency"] = 1 if hz == 50 else 2
            self._apply_now(dict(self.prof, color=dict(col, enable_auto_exposure=1, enable_auto_white_balance=1)))
            time.sleep(3.0)                                       # auto white balance settles
            cur = src.run_on_device(lambda dev, rs: B378.read_back(dev, rs)).get("color", {})
            if "white_balance" in cur:
                col["white_balance"] = cur["white_balance"]
            col.update(enable_auto_exposure=0, enable_auto_white_balance=0)
            self._apply_now(self.prof)

            def set_eg(e, g):
                col["exposure"], col["gain"] = float(e), float(g)
                self._apply_now({"color": {"enable_auto_exposure": 0, "exposure": float(e), "gain": float(g)}})
            res, _trace = autotune(set_eg, self._measure_roi, hz,
                                   log=lambda m: self.tab._ui(lambda m_=m: self.log(m_)))
            self.last_tune = res

            def ui():
                self.v_exp.set(str(col["exposure"]))
                self.v_gain.set(str(col["gain"]))
                self.v_wb.set(str(col["white_balance"]))
                st = res["stats"]
                head = ("✓ 合格" + (f"（{res['note']}）" if res.get("note") else "")) if res["ok"] else \
                    "✗ " + "；".join(res["issues"])
                self.v_tune.set(f"一键调参结果：{head}\n曝光 {res['exposure_ms']} ms · 增益 {res['gain']:g} · 白平衡 "
                                f"{col['white_balance']:g} K · 工作区平均 {st['mean']:.0f} · 最亮 {st['p995']:.0f} · 过曝 "
                                f"{st['sat_pct']} %（{res['steps']} 步）"
                                + ("\n合格：起个名字（例如「办公室白天」）→「保存为这一组」" if res["ok"] else
                                   "\n不合格的原因见上；改善灯光后再点一次，或手动调整后「应用」"))
                try:
                    self.lb_tune.configure(foreground="#060" if res["ok"] else "#a00")
                except Exception:
                    pass
                self.on_read_back()
            self.tab._ui(ui)
        self.log("[INFO] [V3.80 auto tune] 开始一键调参（约 10 s，画面别动）")
        self._run("一键调参", job)

    def on_save_profile(self):
        name = self.v_pname.get().strip()
        if not name:
            self.log("[WARN] [V3.80 camera lab] 先给这组参数起个名字（例如「办公室白天」）")
            return
        try:
            self._manual_colour()
        except ValueError as exc:
            self.log(f"[WARN] [V3.79 camera lab] {exc}")
            return
        src = self._src()
        ref = None
        if src is not None and src.frames()[0] is not None:
            ref = dict(roi_stats(src.frames()[0], self.st.get("roi")), roi=list(self.st.get("roi") or DEFAULT_ROI))
        self.prof["locked"] = True
        d = dict(json.loads(json.dumps(self.prof)), ref=ref, roi=list(self.st.get("roi") or DEFAULT_ROI))
        p = save_named(name, d)
        msg = f"[OK] [V3.80 camera lab] 参数组「{name}」已保存 → {p}"
        if self.v_auto.get():
            activate_named(name)
            msg += "；设为 Camera 页连接时自动套用"
        self.prof = B378.load_profile()
        self.v_prof.set(self._prof_text())
        self.log(msg)
        if src is not None:
            self._run("应用相机参数", lambda: self._apply_now(self.prof))

    def _current_profile_dict(self):
        src = self._src()
        ref = None
        if src is not None and src.frames()[0] is not None:
            ref = dict(roi_stats(src.frames()[0], self.st.get("roi")), roi=list(self.st.get("roi") or DEFAULT_ROI))
        return dict(json.loads(json.dumps(self.prof)), locked=True, ref=ref, roi=list(self.st.get("roi") or DEFAULT_ROI))

    def on_save_file(self, path=None):
        """V3.81 PROFILE FILES: save-as dialog"""
        try:
            self._manual_colour()
        except ValueError as exc:
            self.log(f"[WARN] [V3.79 camera lab] {exc}")
            return None
        if path is None:
            try:
                from tkinter import filedialog
                path = filedialog.asksaveasfilename(
                    title="另存相机参数文件", defaultextension=".json", filetypes=[("相机参数", "*.json")],
                    initialdir=self.st.get("profile_dir") or PROFILES_DIR,
                    initialfile=safe_class(self.v_pname.get().strip() or "d435i_profile") + ".json")
            except Exception:
                path = None
        if not path:
            return None
        d = self._current_profile_dict()
        if self.v_pname.get().strip():
            d["name"] = self.v_pname.get().strip()
        else:
            d.pop("name", None)
        save_profile_file(path, d)
        self.st["profile_dir"] = os.path.dirname(os.path.abspath(path))
        self._save_st()
        self.log(f"[OK] [V3.81 camera lab] 相机参数已另存 → {path}")
        return path

    def on_open_file(self, path=None):
        """V3.81 PROFILE FILES: open dialog -> fields + camera (+ connect profile when ticked)"""
        if path is None:
            try:
                from tkinter import filedialog
                path = filedialog.askopenfilename(title="打开相机参数文件", filetypes=[("相机参数", "*.json")],
                                                  initialdir=self.st.get("profile_dir") or PROFILES_DIR)
            except Exception:
                path = None
        if not path:
            return None
        try:
            d = load_profile_file(path)
        except Exception as exc:
            self.log(f"[FAIL] [V3.81 camera lab] 打不开 {path}：{exc}")
            return None
        self.st["profile_dir"] = os.path.dirname(os.path.abspath(path))
        for k in ("color", "depth", "filters"):
            self.prof[k].update(d.get(k) or {})
        if d.get("roi"):
            self.st["roi"] = list(d["roi"])
        col = self.prof["color"]
        self.v_exp.set(str(col["exposure"]))
        self.v_gain.set(str(col["gain"]))
        self.v_wb.set(str(col["white_balance"]))
        self.v_pl.set(next((n for n, v in POWER_CHOICES if v == int(col.get("power_line_frequency", 1))), "50 Hz"))
        self.v_pname.set(str(d.get("name") or ""))
        msg = f"[OK] [V3.81 camera lab] 已打开 {path}：{profile_summary(d)}"
        if self.v_auto.get():
            B378.save_profile(dict(d, locked=True))
            self.prof = B378.load_profile()
            msg += "；设为连接时自动套用"
        self.v_prof.set(f"已打开：{profile_summary(d)}\n" + self._prof_text())
        self.log(msg)
        self._save_st()
        if self._src() is not None:
            self._run("应用相机参数", lambda: self._apply_now(self.prof))
        return d

    def on_load_named(self):
        name = self.v_pname.get().strip()
        if name not in list_named():
            self.log(f"[WARN] [V3.80 camera lab] 没有参数组「{name}」")
            return
        d = load_named(name)
        for k in ("color", "depth", "filters"):
            self.prof[k].update(d.get(k) or {})
        if d.get("roi"):
            self.st["roi"] = list(d["roi"])
        col = self.prof["color"]
        self.v_exp.set(str(col["exposure"]))
        self.v_gain.set(str(col["gain"]))
        self.v_wb.set(str(col["white_balance"]))
        self.v_pl.set(next((n for n, v in POWER_CHOICES if v == int(col.get("power_line_frequency", 1))), "50 Hz"))
        msg = f"[OK] [V3.80 camera lab] 套用参数组「{name}」"
        if self.v_auto.get():
            activate_named(name)
            self.prof = B378.load_profile()
            msg += "（并设为连接时自动套用）"
        self.v_prof.set(self._prof_text())
        self.log(msg)
        if self._src() is not None:
            self._run("应用相机参数", lambda: self._apply_now(self.prof))

    def on_read_back(self):
        src = self._src()
        if src is None:
            self.v_prof.set(self._prof_text())
            return
        try:
            rb = src.run_on_device(lambda dev, rs: B378.read_back(dev, rs), timeout=2.0)
        except Exception as exc:
            self.log(f"[WARN] [V3.79 camera lab] 读不到相机参数：{exc}")
            rb = None
        self.v_prof.set(self._prof_text(rb))

    # ------------------------------------------------------------------ ② depth noise
    def _build_noise(self):
        tk, ttk = self.tk, self.ttk
        p = ttk.Frame(self.nb, padding=6)
        self.nb.add(p, text="② 深度噪声")
        self.cv_noise = tk.Canvas(p, width=self.PW, height=self.PH, bg="#202020", highlightthickness=0)
        self.cv_noise.pack(anchor="w")
        r = ttk.Frame(p)
        r.pack(fill="x", pady=3)
        self.v_nframes = self._var(self.st.get("noise_frames", 30))
        ttk.Label(r, text="每种处理帧数").pack(side="left")
        ttk.Entry(r, textvariable=self.v_nframes, width=5).pack(side="left", padx=(2, 8))
        ttk.Button(r, text="▶ 开始测量（保持不动）", command=self.on_noise).pack(side="left")
        cols = ("setting", "valid", "std", "flat", "dist")
        self.tv_noise = ttk.Treeview(p, columns=cols, show="headings", height=6, style=self._tree_style())
        for c, t, w in zip(cols, ("深度处理", "有效像素 %", "抖动 mm", "平整度 mm", "距离 mm"), (220, 90, 90, 90, 90)):
            self.tv_noise.heading(c, text=t)
            self.tv_noise.column(c, width=w, anchor="w" if c == "setting" else "e")
        self.tv_noise.pack(fill="x", pady=3)
        self.v_nverdict = self._var("")                 # V3.80: verdict after a measurement
        self.lb_nverdict = ttk.Label(p, textvariable=self.v_nverdict, justify="left", wraplength=self.PW + 400)
        self.lb_nverdict.pack(anchor="w", pady=(2, 4))
        ttk.Label(p, text="相机对准平整、不反光的桌面或卡纸，画面中央方框是测量区，测量时人和物都不要动。\n"
                          "「抖动」= 同一像素逐帧变化（越小越稳）；「平整度」= 平均深度离最佳平面的偏差（越小越平）。\n"
                          "Camera 页现在用的是「spatial+temporal」。",
                  foreground="#555", justify="left").pack(anchor="w")

    def on_noise(self):
        src = self._need_src()
        if src is None:
            return
        try:
            n = int(self._fget(self.v_nframes, 5, 120, "帧数"))
        except ValueError as exc:
            self.log(f"[WARN] [V3.79 camera lab] {exc}")
            return
        self.st["noise_frames"] = n

        def job():
            rows, roi = measure_noise(src, dict(src.K), n, log=lambda m: self.tab._ui(lambda m_=m: self.log(m_)))
            verdict = noise_verdict(rows)
            stem = B378.write_results("noise", rows, {"roi": roi, "source": "camera lab", "verdict": verdict})

            def ui():
                self.tv_noise.delete(*self.tv_noise.get_children())
                for r_ in rows:
                    self.tv_noise.insert("", "end", values=(r_["setting"], r_["valid_pct"], r_["temporal_std_mm"],
                                                            r_["flatness_rms_mm"], r_["mean_dist_mm"]))
                self.v_nverdict.set("\n".join(verdict["lines"]))
                try:
                    self.lb_nverdict.configure(foreground={"ok": "#060", "warn": "#a60", "fail": "#a00"}[verdict["level"]])
                except Exception:
                    pass
                lv = {"ok": "OK", "warn": "WARN", "fail": "FAIL"}[verdict["level"]]
                self.log(f"[{lv}] [V3.80 noise] {verdict['lines'][0]}")
                self.log(f"[OK] [V3.79 noise] 结果已保存 → {stem}.csv")
            self.tab._ui(ui)
        self.log("[INFO] [V3.79 noise] 开始测量 5 种深度处理（约 30 s，保持不动）")
        self._run("深度噪声", job)

    # ------------------------------------------------------------------ ③ locating accuracy
    def _build_accuracy(self):
        # V3.80 CAMERA LAB UPGRADE: any object (egg / ball shape or general object)
        tk, ttk = self.tk, self.ttk
        p = ttk.Frame(self.nb, padding=6)
        self.nb.add(p, text="③ 定位精度")
        self.cv_acc = tk.Canvas(p, width=self.PW, height=self.PH, bg="#202020", highlightthickness=0)
        self.cv_acc.pack(anchor="w")
        r = ttk.Frame(p)
        r.pack(fill="x", pady=3)
        self.v_aprompt = self._var(self.st.get("acc_prompt", ""))
        self.v_ashape = self._var(dict(SHAPES).get(self.st.get("acc_shape"), SHAPES[0][1]))
        self.v_aL, self.v_aW, self.v_aH = (self._var(self.st.get(k, "")) for k in ("acc_L", "acc_W", "acc_H"))
        self.v_atrials = self._var(self.st.get("trials", 10))
        ttk.Label(r, text="物体名").pack(side="left")
        ttk.Entry(r, textvariable=self.v_aprompt, width=12).pack(side="left", padx=(2, 6))
        ttk.Combobox(r, textvariable=self.v_ashape, values=[t for _k, t in SHAPES], width=20,
                     state="readonly").pack(side="left")
        for lab, var in (("长", self.v_aL), ("宽", self.v_aW), ("高 mm", self.v_aH)):
            ttk.Label(r, text=lab).pack(side="left", padx=(6, 0))
            ttk.Entry(r, textvariable=var, width=5).pack(side="left", padx=2)
        ttk.Label(r, text="次数").pack(side="left", padx=(6, 0))
        ttk.Entry(r, textvariable=self.v_atrials, width=4).pack(side="left", padx=2)
        r2 = ttk.Frame(p)
        r2.pack(fill="x", pady=2)
        ttk.Button(r2, text="1 锁定标定板（先不放物体）", command=self.on_board).pack(side="left")
        ttk.Button(r2, text="2 ▶ 测量", command=self.on_accuracy).pack(side="left", padx=4)
        self.v_board = self._var("标定板：未锁定")
        ttk.Label(p, textvariable=self.v_board).pack(anchor="w")
        cols = ("setting", "det", "e3", "xy", "dz", "h")
        self.tv_acc = ttk.Treeview(p, columns=cols, show="headings", height=3, style=self._tree_style())
        for c, t, w in zip(cols, ("深度处理", "检出率 %", "3D 误差 均值/最大", "水平 均值/最大", "中心高度 均值/最大",
                                  "高度测量 均值/最大"), (140, 70, 140, 130, 140, 140)):
            self.tv_acc.heading(c, text=t)
            self.tv_acc.column(c, width=w, anchor="w")
        self.tv_acc.pack(fill="x", pady=3)
        self.v_averdict = self._var("")
        ttk.Label(p, textvariable=self.v_averdict, justify="left", wraplength=self.PW + 400).pack(anchor="w")
        ttk.Label(p, text="打印 7×5 ChArUco 板（32 mm 方格），去掉塑料袋，四边胶带贴平。① 不放物体「锁定标定板」→ ② 物体中心"
                          "放在画面上带编号的某个角点上 →「测量」（误差单位 mm）。\n"
                          "真值 = 角点正上方、物体高度的一半（「高」填物体放好后的高度；横放的鸡蛋 = 蛋宽 42）。"
                          "「蛋形 / 球形」用椭球拟合（填长、宽），「一般物体」用 Camera 页的通用定位，并比较它量出的高度。\n"
                          "建议先用规则物体（方块、圆柱）测，容易放正；误差里含手放的偏差（约 ±1–2 mm）。模型 / 置信度用"
                          " Camera 页「视觉定位」的设置。",
                  foreground="#555", justify="left").pack(anchor="w")

    def on_board(self):
        src = self._need_src()
        if src is None:
            return

        def job():
            rgbs = []
            last = None
            t0 = time.time()
            while len(rgbs) < 10 and time.time() - t0 < 4.0:
                rgb, _d, fid = src.frames()
                if rgb is not None and fid != last:
                    rgbs.append(rgb)
                    last = fid
                else:
                    time.sleep(0.02)
            T, info = B378.board_pose(rgbs, dict(src.K), list(src.dist or [0.0] * 5))

            def ui():
                self.board_T, self.board_info = T, info
                if T is None:
                    self.v_board.set(f"标定板：没找到（{info}）")
                    self.log(f"[WARN] [V3.79 accuracy] 标定板没找到：{info}")
                else:
                    self.v_board.set(f"标定板：已锁定 · {info['frames']} 帧 · 角点 {info['corners']} · 帧间 "
                                     f"{info['spread_mm']} mm · 相机离板 {info['cam_height_mm']} mm")
                    self.log(f"[OK] [V3.79 accuracy] 标定板已锁定：{info}")
            self.tab._ui(ui)
        self._run("锁定标定板", job)

    def _yolo(self):
        import NERO_V347_Yolo_Perception as YP
        tab = self.tab
        path = YP.find_model(tab.v_ydir.get(), tab.v_ymodel.get())
        if not path:
            raise RuntimeError(f"找不到 YOLOE 模型 {tab.v_ymodel.get()}（Camera 页「视觉定位」里选模型）")
        try:
            conf = float(tab.v_yconf.get())
        except Exception:
            conf = YP.CONF_MIN
        return YP, path, conf

    def on_accuracy(self):
        src = self._need_src()
        if src is None:
            return
        if self.board_T is None:
            self.log("[WARN] [V3.79 accuracy] 先不放鸡蛋点「锁定标定板」")
            return
        prompt = self.v_aprompt.get().strip()
        shape = next((k for k, t in SHAPES if t == self.v_ashape.get()), "general")
        try:
            if not prompt:
                raise ValueError("先填物体名（YOLOE 文字提示，例如 cup、box、egg）")
            trials = int(self._fget(self.v_atrials, 1, 50, "次数"))
            H = self._fget(self.v_aH, 5, 400, "高（物体放好后的高度 mm）")
            LW = None
            if shape == "egg":
                LW = (self._fget(self.v_aL, 5, 400, "长"), self._fget(self.v_aW, 5, 400, "宽"))
                if LW[1] > LW[0]:
                    raise ValueError("宽不能大于长")
            YP, path, conf = self._yolo()
        except Exception as exc:
            self.log(f"[WARN] [V3.80 accuracy] {exc}")
            return
        self.st.update(acc_prompt=prompt, acc_shape=shape, trials=trials, acc_H=self.v_aH.get(),
                       acc_L=self.v_aL.get(), acc_W=self.v_aW.get())
        T, K = np.array(self.board_T), dict(src.K)
        exp_z = H / 2000.0
        loc = locate_fn(shape, LW)

        def job():
            rows, summ = accuracy_trials(lambda n: self.tab._grab_frames(n),
                                         lambda rgb: self.tab.det_yolo.detect(rgb, "text", path, prompts=[prompt],
                                                                              conf=conf),
                                         K, T, B378.inner_corners(), exp_z, trials, locate=loc,
                                         height_mm=H if shape == "general" else None)
            summ["board"] = self.board_info
            summ["model"] = path
            summ["object"] = {"name": prompt, "shape": shape, "height_mm": H, "size_lw_mm": LW}
            stem = B378.write_results("accuracy", rows, summ)

            def ui():
                self.tv_acc.delete(*self.tv_acc.get_children())
                for name, s in summ.items():
                    if not isinstance(s, dict) or "err_3d" not in s:
                        continue
                    f = (lambda q: f"{q.get('mean', '-')} / {q.get('max', '-')}" if q.get("n") else "-")
                    self.tv_acc.insert("", "end", values=(name, s["detected_pct"], f(s["err_3d"]), f(s["xy"]),
                                                          f(s["dz"]), f(s.get("h") or {})))
                corner = next((r_.get("corner") for r_ in rows if r_.get("corner") is not None), "?")
                self.v_averdict.set(accuracy_verdict(summ))
                self.log(f"[OK] [V3.79 accuracy] 角点 {corner}：结果已保存 → {stem}.csv")
            self.tab._ui(ui)
        self.log(f"[INFO] [V3.80 accuracy] 开始测量（{prompt}，{dict(SHAPES)[shape]}，高 {H:g} mm，每种 {trials} 次）")
        self._run("定位精度", job)

    # ------------------------------------------------------------------ ④ capture + auto label + review
    def _build_capture(self):
        tk, ttk = self.tk, self.ttk
        p = ttk.Frame(self.nb, padding=6)
        self.nb.add(p, text="④ 采集标注")
        top = ttk.Frame(p)
        top.pack(fill="x")
        self.v_root = self._var(self.st["root"])
        ttk.Label(top, text="数据文件夹").pack(side="left")
        ttk.Entry(top, textvariable=self.v_root, width=46).pack(side="left", padx=2)
        ttk.Button(top, text="选择…", command=self.on_pick_root).pack(side="left")
        # V3.80 CAMERA LAB UPGRADE: class list (no default class) + current class + counts
        cl = ttk.Frame(p)
        cl.pack(fill="x", pady=(4, 0))
        self.lb_cls = tk.Listbox(cl, height=5, width=46, exportselection=False)
        self.lb_cls.pack(side="left")
        self.lb_cls.bind("<Double-Button-1>", lambda _e: self.on_toggle_class())
        cr = ttk.Frame(cl)
        cr.pack(side="left", fill="y", padx=6)
        r_a = ttk.Frame(cr)
        r_a.pack(anchor="w")
        self.v_newcls = self._var("")
        ttk.Combobox(r_a, textvariable=self.v_newcls, values=COMMON_CLASSES, width=16).pack(side="left")
        ttk.Button(r_a, text="加入清单", command=self.on_add_class).pack(side="left", padx=2)
        r_b = ttk.Frame(cr)
        r_b.pack(anchor="w", pady=2)
        ttk.Button(r_b, text="勾选 / 取消（双击）", command=self.on_toggle_class).pack(side="left")
        ttk.Button(r_b, text="移出清单", command=self.on_remove_class).pack(side="left", padx=2)
        r_c = ttk.Frame(cr)
        r_c.pack(anchor="w", pady=2)
        self.v_curcls = self._var(self.st.get("current", ""))
        ttk.Label(r_c, text="当前采集类别（= 子文件夹）").pack(side="left")
        self.cb_curcls = ttk.Combobox(r_c, textvariable=self.v_curcls, width=14, state="readonly",
                                      postcommand=lambda: self.cb_curcls.configure(values=self._checked()))
        self.cb_curcls.pack(side="left", padx=2)
        ttk.Label(cr, text="勾选的类别都会自动标注（同一张图里的其他物体也要标，否则训练时会被当成背景）；\n"
                           f"每类建议 ≥ {RECOMMENDED_IMAGES} 张，换位置 / 摆法 / 背景 / 光照。",
                  foreground="#555", justify="left").pack(anchor="w")
        r2 = ttk.Frame(p)
        r2.pack(fill="x", pady=3)
        self.v_cconf = self._var(self.st.get("conf", 0.25))
        self.v_cdepth = self.tk.BooleanVar(value=bool(self.st.get("save_depth", True)))
        self.v_bn, self.v_bdt = self._var(self.st.get("burst_n", 10)), self._var(self.st.get("burst_dt", 1.0))
        ttk.Label(r2, text="置信度").pack(side="left")
        ttk.Entry(r2, textvariable=self.v_cconf, width=5).pack(side="left", padx=(2, 8))
        ttk.Checkbutton(r2, text="同时存深度图", variable=self.v_cdepth).pack(side="left")
        ttk.Button(r2, text="📷 拍一张", command=self.on_shot).pack(side="left", padx=(10, 2))
        ttk.Label(r2, text="连拍").pack(side="left", padx=(8, 0))
        ttk.Entry(r2, textvariable=self.v_bn, width=4).pack(side="left", padx=2)
        ttk.Label(r2, text="张，间隔").pack(side="left")
        ttk.Entry(r2, textvariable=self.v_bdt, width=4).pack(side="left", padx=2)
        ttk.Label(r2, text="s").pack(side="left")
        ttk.Button(r2, text="▶ 连拍", command=self.on_burst).pack(side="left", padx=4)
        ttk.Button(r2, text="■ 停", command=self.on_stop_burst).pack(side="left")
        self.v_cstat = self._var("")
        ttk.Label(p, textvariable=self.v_cstat, foreground="#335").pack(anchor="w")
        body = ttk.Frame(p)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        self.cv_live = tk.Canvas(left, width=480, height=270, bg="#202020", highlightthickness=0)
        self.cv_live.pack(anchor="w")
        lr = ttk.Frame(left)
        lr.pack(fill="x", pady=2)
        self.v_folder = self._var("")
        ttk.Label(lr, text="复核").pack(side="left")
        self.cb_folder = ttk.Combobox(lr, textvariable=self.v_folder, width=16, state="readonly",
                                      postcommand=self._refresh_folders)
        self.cb_folder.pack(side="left", padx=2)
        self.cb_folder.bind("<<ComboboxSelected>>", lambda _e: self._fill_list())
        ttk.Button(lr, text="刷新", command=self._fill_list).pack(side="left")
        cols = ("stem", "split", "n", "rev")
        self.tv_items = ttk.Treeview(left, columns=cols, show="headings", height=9, style=self._tree_style())
        for c, t, w in zip(cols, ("图片", "集", "框", "已复核"), (250, 50, 40, 60)):
            self.tv_items.heading(c, text=t)
            self.tv_items.column(c, width=w, anchor="w")
        self.tv_items.pack(fill="both", expand=True)
        self.tv_items.bind("<<TreeviewSelect>>", lambda _e: self._on_pick_item())
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.cv_rev = tk.Canvas(right, width=self.RW, height=self.RH, bg="#303030", highlightthickness=0)
        self.cv_rev.pack(anchor="w")
        self.cv_rev.bind("<ButtonPress-1>", self._ev_press)
        self.cv_rev.bind("<B1-Motion>", self._ev_drag)
        self.cv_rev.bind("<ButtonRelease-1>", self._ev_release)
        self.win.bind("<Delete>", lambda _e: self.on_del_box())
        rr = ttk.Frame(right)
        rr.pack(fill="x", pady=2)
        self.v_bcls = self._var("")
        ttk.Label(rr, text="选中框的类别").pack(side="left")
        self.cb_bcls = ttk.Combobox(rr, textvariable=self.v_bcls, width=12, postcommand=self._refresh_cls)
        self.cb_bcls.pack(side="left", padx=2)
        self.cb_bcls.bind("<<ComboboxSelected>>", lambda _e: self.on_set_cls())
        ttk.Button(rr, text="删除框 (Del)", command=self.on_del_box).pack(side="left", padx=4)
        ttk.Button(rr, text="◀ 上一张", command=lambda: self._step(-1)).pack(side="left", padx=(10, 2))
        ttk.Button(rr, text="✓ 已复核，下一张", command=lambda: self._step(+1, reviewed=True)).pack(side="left", padx=2)
        ttk.Button(rr, text="下一张 ▶", command=lambda: self._step(+1)).pack(side="left", padx=2)
        ttk.Button(rr, text="删除这张图", command=self.on_del_image).pack(side="left", padx=(10, 0))
        self.v_rev = self._var("在左边列表选一张图。点框 = 选中；拖框内 = 移动；拖白色角 = 改大小；在空白处拖 = 新框"
                               "（类别 = 上面的框类别或当前采集类别）；Del = 删除。框改动后轮廓自动重算。")
        ttk.Label(right, textvariable=self.v_rev, foreground="#555", wraplength=self.RW, justify="left").pack(anchor="w")
        self._update_cstat()

    def _set_root(self, root):
        root = os.path.abspath(os.path.expanduser(root.strip() or DEFAULT_ROOT))
        if root != self.ds.root:
            self.ds = Dataset(root)
            self.cur = self.editor = self.cur_rgb = None
        self.st["root"] = root
        return root

    def on_pick_root(self):
        try:
            from tkinter import filedialog
            d = filedialog.askdirectory(initialdir=self.v_root.get() or DEFAULT_ROOT, title="选择数据文件夹")
        except Exception:
            d = None
        if d:
            self.v_root.set(d)
            self._set_root(d)
            self._update_cstat()
            self._save_st()

    # ---- V3.80 class list
    def _checked(self):
        return [c for c in self.st.get("class_list", []) if c in self.st.get("checked", [])]

    def _prompts(self):
        return self._checked()

    def _add_class(self, name):
        name = str(name or "").strip()
        if not name or "," in name:
            return False
        cl = self.st.setdefault("class_list", [])
        if name not in cl:
            cl.append(name)
        ck = self.st.setdefault("checked", [])
        if name not in ck:
            ck.append(name)
        if not self.st.get("current"):
            self.st["current"] = name
            self.v_curcls.set(name)
        return True

    def _toggle_class(self, name):
        ck = self.st.setdefault("checked", [])
        if name in ck:
            ck.remove(name)
            if self.st.get("current") == name:
                self.st["current"] = (self._checked() or [""])[0]
                self.v_curcls.set(self.st["current"])
        elif name in self.st.get("class_list", []):
            ck.append(name)

    def _sel_class(self):
        try:
            sel = [int(i) for i in self.lb_cls.curselection()]
        except Exception:
            sel = []
        cl = self.st.get("class_list", [])
        return [cl[i] for i in sel if 0 <= i < len(cl)]

    def on_add_class(self):
        if self._add_class(self.v_newcls.get()):
            self.v_newcls.set("")
            self._update_cstat()
            self._save_st()

    def on_toggle_class(self):
        for c in self._sel_class():
            self._toggle_class(c)
        self._update_cstat()

    def on_remove_class(self):
        for c in self._sel_class():
            self.st["class_list"].remove(c)
            if c in self.st.get("checked", []):
                self.st["checked"].remove(c)
            if self.st.get("current") == c:
                self.st["current"] = (self._checked() or [""])[0]
                self.v_curcls.set(self.st["current"])
        self._update_cstat()

    def _class_lines(self):
        cc = self.ds.class_counts()
        out = []
        for c in self.st.get("class_list", []):
            k = cc.get(c, {"images": 0, "objects": 0})
            need = max(0, RECOMMENDED_IMAGES - k["images"])
            out.append(f"{'☑' if c in self.st.get('checked', []) else '☐'} {c}   {k['images']} 张 · {k['objects']} 框"
                       + (f" · 还差 {need} 张" if need else " · 够了 ✓"))
        return out

    def _update_cstat(self):
        try:
            self.lb_cls.delete(0, "end")
            for line in self._class_lines():
                self.lb_cls.insert("end", line)
        except Exception:
            pass
        st = self.ds.stats()
        if not st:
            self.v_cstat.set(f"{self.ds.root}：还没有图片" + ("" if self.st.get("class_list") else
                                                          "；先在上面「加入清单」要采集的物体"))
            return
        self.v_cstat.set("文件夹：" + "  ·  ".join(f"{k}: {v['images']} 张（训练 {v['train']} / 验证 {v['val']}，已复核 "
                                                f"{v['reviewed']}）" for k, v in st.items()))

    def _capture_one(self, prompts, conf, depth_on, YP, path, folder=None):
        src = self._src()
        if src is None:
            raise RuntimeError("相机未连接")
        rgb, depths = self.tab._grab_frames(1)
        if rgb is None:
            raise RuntimeError("没有取到画面")
        dets = self.tab.det_yolo.detect(rgb, "text", path, prompts=prompts, conf=conf)
        anns = dets_to_anns(dets, prompts)
        folder = folder or prompts[0]
        stem = self.ds.add_image(folder, rgb, anns, depth=(depths[-1] if depth_on and depths else None),
                                 extra={"prompts": prompts, "current": folder, "conf_min": conf, "model": os.path.basename(path),
                                        "K": dict(src.K or {}), "profile_locked": bool(B378.load_profile().get("locked"))})
        return safe_class(folder), stem, len(anns)

    def _capture_args(self):
        prompts = self._prompts()
        if not prompts:
            raise ValueError("先在类别清单里「加入清单」并勾选要采集的物体")
        cur = self.v_curcls.get().strip()
        if cur not in prompts:
            cur = prompts[0]
            self.v_curcls.set(cur)
        self.st["current"] = cur
        conf = self._fget(self.v_cconf, 0.01, 0.99, "置信度")
        YP, path, _c = self._yolo()
        self._set_root(self.v_root.get())
        self.st.update(conf=conf, save_depth=bool(self.v_cdepth.get()))
        if not B378.load_profile().get("locked"):
            self.log("[WARN] [V3.79 capture] 相机参数还没锁定（① 相机参数 →「保存并锁定」）：现在拍的图和以后的可能对不上")
        return prompts, conf, bool(self.v_cdepth.get()), YP, path, cur

    def _after_shot(self, folder, stem, n):
        self._update_cstat()
        self.v_folder.set(folder)
        self._fill_list(select=stem)
        self.log(f"[OK] [V3.79 capture] {stem}：自动标注 {n} 个框 → {self.ds.cdir(folder)}")

    def on_shot(self):
        if self._need_src() is None:
            return
        try:
            args = self._capture_args()
        except Exception as exc:
            self.log(f"[WARN] [V3.79 capture] {exc}")
            return

        def job():
            folder, stem, n = self._capture_one(*args)
            self.tab._ui(lambda: self._after_shot(folder, stem, n))
        self._run("拍照", job)

    def on_burst(self):
        if self._need_src() is None:
            return
        try:
            args = self._capture_args()
            n = int(self._fget(self.v_bn, 1, 500, "张数"))
            dt = self._fget(self.v_bdt, 0.2, 60, "间隔")
        except Exception as exc:
            self.log(f"[WARN] [V3.79 capture] {exc}")
            return
        self.st.update(burst_n=n, burst_dt=dt)
        self._burst_stop = False

        def job():
            for i in range(n):
                if self._burst_stop or not self.alive:
                    break
                t0 = time.time()
                folder, stem, k = self._capture_one(*args)
                self.tab._ui(lambda f=folder, s=stem, k_=k: self._after_shot(f, s, k_))
                while time.time() - t0 < dt and not self._burst_stop:
                    time.sleep(0.05)
            self.tab._ui(lambda: self.log("[OK] [V3.79 capture] 连拍结束"))
        self.log(f"[INFO] [V3.79 capture] 连拍 {n} 张，间隔 {dt:g} s（拍的时候可以挪动物体 / 换摆法）")
        self._run("连拍", job)

    def on_stop_burst(self):
        self._burst_stop = True

    # ---- review
    def _refresh_folders(self):
        self.cb_folder.configure(values=self.ds.folders())

    def _refresh_cls(self):
        self.cb_bcls.configure(values=list(dict.fromkeys(self._prompts() + self.st.get("class_list", []) +
                                                         self.ds.classes())))

    def _fill_list(self, select=None):
        self._set_root(self.v_root.get())
        fo = self.v_folder.get()
        self.tv_items.delete(*self.tv_items.get_children())
        if not fo:
            return
        for m in self.ds.items(fo):
            self.tv_items.insert("", "end", iid=m["stem"], values=(m["stem"], m.get("split"), len(m.get("anns", [])),
                                                                   "✓" if m.get("reviewed") else ""))
        if select:
            try:
                self.tv_items.selection_set(select)
                self.tv_items.see(select)
            except Exception:
                pass
            self._open_item(fo, select)

    def _on_pick_item(self):
        sel = self.tv_items.selection()
        if sel and (self.cur is None or self.cur[1] != sel[0]):
            self._open_item(self.v_folder.get(), sel[0])

    def _open_item(self, folder, stem):
        self._commit()
        rgb, meta = self.ds.load(folder, stem)
        if rgb is None:
            self.log(f"[WARN] [V3.79 review] 读不到图片 {stem}")
            return
        self.cur, self.cur_rgb, self.cur_meta = (folder, stem), rgb, meta
        default = meta.get("current") or (meta.get("prompts") or self._prompts() or [folder])[0]
        self.editor = BoxEditor(json.loads(json.dumps(meta.get("anns", []))), meta["size"], default_cls=default)
        self._dirty = False
        self._redraw()

    def _scale(self):
        return fit_scale(self.cur_meta["size"], (self.RW, self.RH))

    def _redraw(self):
        if self.cur_rgb is None or self.editor is None:
            return
        s = self._scale()
        img = draw_overlay(resize(self.cur_rgb, s), self.editor.anns, self.editor.sel, scale=s)
        self._show(self.cv_rev, "rev", img)
        m = self.cur_meta
        sel = self.editor.sel
        self.v_rev.set(f"{m['stem']}（{m['split']}，{len(self.editor.anns)} 个框"
                       f"{'，已复核' if m.get('reviewed') else ''}）"
                       + (f" · 选中 #{sel} {self.editor.anns[sel]['cls']}" if sel is not None else ""))
        if sel is not None:
            self.v_bcls.set(self.editor.anns[sel]["cls"])

    def _img_xy(self, ev):
        s = self._scale()
        return ev.x / s, ev.y / s, s

    def _ev_press(self, ev):
        if self.editor is None:
            return
        x, y, s = self._img_xy(ev)
        if self.v_bcls.get().strip():
            self.editor.default_cls = self.v_bcls.get().strip()
        self.editor.press(x, y, s)
        self._redraw()

    def _ev_drag(self, ev):
        if self.editor is None:
            return
        x, y, _s = self._img_xy(ev)
        self.editor.drag(x, y)
        self._redraw()

    def _ev_release(self, ev):
        if self.editor is None:
            return
        x, y, _s = self._img_xy(ev)
        i = self.editor.release(x, y)
        if i is not None:
            a = self.editor.anns[i]
            a["poly"], how = recompute_outline(self.cur_rgb, a["box"], self.cur_meta.get("auto"))
            a["outline"] = how
            self._dirty = True
            self._commit()
        self._redraw()

    def on_del_box(self):
        if self.editor is not None and self.editor.delete():
            self._dirty = True
            self._commit()
            self._redraw()

    def on_set_cls(self):
        c = self.v_bcls.get().strip()
        if self.editor is not None and c and self.editor.set_class(c):
            self._dirty = True
            self._commit()
            self._redraw()

    def _commit(self, reviewed=None):
        if self.cur is None or self.editor is None:
            return
        if getattr(self, "_dirty", False) or reviewed is not None:
            self.cur_meta = self.ds.save_anns(self.cur[0], self.cur[1], self.editor.anns, reviewed=reviewed)
            self._dirty = False
            try:
                self.tv_items.item(self.cur[1], values=(self.cur[1], self.cur_meta.get("split"), len(self.editor.anns),
                                                        "✓" if self.cur_meta.get("reviewed") else ""))
            except Exception:
                pass

    def _step(self, d, reviewed=None):
        if self.cur is None:
            return
        self._commit(reviewed)
        self._update_cstat()
        ids = list(self.tv_items.get_children())
        if self.cur[1] not in ids:
            return
        j = ids.index(self.cur[1]) + d
        if 0 <= j < len(ids):
            self.tv_items.selection_set(ids[j])
            self.tv_items.see(ids[j])
            self._open_item(self.cur[0], ids[j])

    def on_del_image(self):
        if self.cur is None:
            return
        fo, stem = self.cur
        ids = list(self.tv_items.get_children())
        nxt = None
        if stem in ids:
            j = ids.index(stem)
            nxt = ids[j + 1] if j + 1 < len(ids) else (ids[j - 1] if j > 0 else None)
        self.ds.delete_image(fo, stem)
        self.log(f"[INFO] [V3.79 review] 已删除 {stem}")
        self.cur = self.editor = self.cur_rgb = None
        self.cv_rev.delete("all")
        self._update_cstat()
        self._fill_list(select=nxt)
