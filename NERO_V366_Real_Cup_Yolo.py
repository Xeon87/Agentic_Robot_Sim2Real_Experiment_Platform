#!/usr/bin/env python3
"""NERO V3.66 REAL CUP YOLOE PERCEPTION - the real cup task sees cups exactly the way the Camera tab does.

V3.66 REAL CUP YOLOE PERCEPTION

One pipeline, not two: 「Camera调试 → 视觉定位 → 物体识别」 (text prompt "cup") and 「NERO实机任务」 use the same
steps, the same code and the same settings:
  1. frames         Camera tab _grab_frames(YP.FUSE_FRAMES) (the same D435i stream, 5 frames)
  2. 2D detection   the Camera tab's YOLOE detector (det_yolo: same model file, model folder, confidence, GPU/CPU,
                    text prompt or *-pf vocabulary resolved exactly like the Camera tab's object recognition)
  3. depth fusion   YP.fuse_depth(depths, dets) (ROI temporal fusion)
  4. 3D             YP.locate_objects(...) (local table plane, fence, calibration note) = the Camera tab's result table
Then, per YOLOE instance mask, the cup-specific geometry the skills need (the YOLOE mask replaces the V3.63 red-colour
mask and the x-y clustering; nothing falls back to the colour method):
  5. cup axis       least-squares fit of the mask's wall points (depth) to the known cone r(z) (NERO_V363 fit_axis);
                    duplicate masks of one cup (axes < 5 cm apart) are merged
  6. rim height     from the SILHOUETTE, not from the depth: the rim circle of the cone model on the fitted axis is
                    projected into the image for every height; the height whose projected top edge matches the
                    mask's top edge is the measured height (the D435i loses depth on the thin, glossy rim - the depth
                    top read 99-108 mm for a 121 mm cup in the first real test)
  7. kind           single cup / nested pair / unknown from the silhouette height (single ~ H + rim bead, nested
                    ~ + 25 mm); lifted (held by the gripper), outside the fence, cut by the image edge -> not a cup
                    on the table
YOLOE unavailable (no ultralytics, no model file, no text encoder, Camera tab without a detector) -> YoloCupUnavailable:
the task stops with that message.
"""
import math
import os
import sys

VERSION = "V3.71"
MARKER = "V3.66 REAL CUP YOLOE PERCEPTION"
MARKER_V367 = "V3.67 PERCEPTION ROBUSTNESS"
HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np                          # noqa: E402
import NERO_V343_Scenes as SC               # noqa: E402

DEFAULT_PROMPT = "cup"
MERGE_M = 0.05                  # two masks whose cup axes are closer than this are one cup (or one nested pair)
LIFT_NEAR_M = 0.12             # V3.69 LIFTED NEAR GRIPPER: a lifted cup is within this of the gripper (x-y)
LIFT_M = 0.025                  # lowest mask points (1st percentile) this far above the table -> in the air (held)
TOL_LOW_MM = 18.0               # single cup: silhouette height >= H_top - 18 mm
TOL_HIGH_MM = 18.0              # nested pair: <= H_top + nested + 18 mm
MIN_PTS = 80
# V3.67 PERCEPTION ROBUSTNESS
OCCL_TOP_FRAC = 0.30            # >= this share of the mask's top-edge columns border the gripper -> top hidden
OCCL_PAD_PX = 10
TILT_SPLIT_MAX_DEG = 4.0        # two half-sample tilt fits must agree within this, else the tilt is not trusted
TILT_RMS_MAX_M = 0.005
TILT_MIN_BAND_PTS = 150
FIT_MAX_PTS = 1500              # V3.68 FAST PERCEPTION: wall points used per cone fit (evenly subsampled)
SNAP_KEEP = 40                  # failure snapshots kept in NERO_CALIB_DIR/real_cup_logs


class YoloCupUnavailable(RuntimeError):
    pass


def _yp():
    import NERO_V347_Yolo_Perception as YP
    return YP


# ================================================================================ settings (Tk thread)
def resolve_query(YP, model_dir, model_name, prompts):
    """(mode, path) for an object query - the Camera tab's _resolve_object_query with the task's prompts"""
    sel = YP.find_model(model_dir, model_name)
    pf_path = YP.find_model(model_dir, YP.DEFAULT_PF_MODEL)
    if sel and YP.is_pf_model(sel):
        return "pf", sel
    if YP.is_wildcard_prompt(prompts) and pf_path:
        return "pf", pf_path
    if sel:
        enc = os.path.join(os.path.dirname(sel), YP.text_encoder_name(sel))
        if not os.path.isfile(enc) and pf_path:
            return "pf", pf_path
    return "text", sel


def yolo_cfg_from_tab(cam_tab, prompt_text=DEFAULT_PROMPT, detector=None):
    """Camera tab YOLOE settings (Tk variables -> call in the Tk thread).  -> dict, with "error" when unusable.
    V3.68 FAST PERCEPTION: `detector` = the task's own YoloEDetector (same class, same model file / confidence / prompt
    rules) so that switching prompts in the Camera tab does not force a model reload before every cup perception"""
    YP = _yp()
    prompts = YP.parse_prompts(prompt_text) or [DEFAULT_PROMPT]
    det = detector if detector is not None else getattr(cam_tab, "det_yolo", None)
    if det is None or not all(hasattr(cam_tab, a) for a in ("v_ydir", "v_ymodel", "v_yconf")):
        return {"error": "Camera调试 页没有 YOLOE 检测器（视觉定位子页没有建立）", "prompts": prompts}
    try:
        ydir, yname = cam_tab.v_ydir.get().strip(), cam_tab.v_ymodel.get().strip()
        conf = float(cam_tab.v_yconf.get())
        assert 0.01 <= conf <= 0.99
    except Exception:
        return {"error": "Camera调试 → 视觉定位 的置信度门槛无效（0.01–0.99）", "prompts": prompts}
    mode, path = resolve_query(YP, ydir, yname, prompts)
    cfg = {"detector": det, "mode": mode, "path": path, "prompts": prompts, "conf": conf, "model_name": yname,
           "model_dir": ydir}
    if not path:
        cfg["error"] = f"找不到 YOLOE 模型 {yname}（找过：{', '.join(YP.model_dirs(ydir))}）"
    return cfg


def describe_cfg(cfg):
    if not cfg:
        return "YOLOE 未配置"
    if cfg.get("error"):
        return f"YOLOE 不可用：{cfg['error']}"
    q = "默认词库(-pf)" if cfg["mode"] == "pf" else f"文字提示({', '.join(cfg['prompts'][:4])})"
    return f"YOLOE {os.path.basename(cfg['path'])} · {q} · 置信度 ≥ {cfg['conf']:.2f}（= Camera调试 → 视觉定位）"


# ================================================================================ geometry
def _proj(T_cam_base, K, P):
    pc = P @ T_cam_base[:3, :3].T + T_cam_base[:3, 3]
    z = np.maximum(pc[:, 2], 1e-6)
    return K["fx"] * pc[:, 0] / z + K["ppx"], K["fy"] * pc[:, 1] / z + K["ppy"]


def mask_top_v(mask):
    """top edge of a mask: 2nd percentile of the per-column topmost rows (robust to a one-column spike)"""
    cols = np.flatnonzero(mask.any(axis=0))
    if not len(cols):
        return None, None
    tops = np.argmax(mask[:, cols], axis=0).astype(float)
    return float(np.percentile(tops, 2)), int(tops.min())


def silhouette_height(c_xy, z0, K, T_base_cam, v_obs, g, h_lo=0.03, h_hi=0.26, step=0.0005):
    """height h of the rim circle (radius r_top + bead, centre on the axis at z0 + h) whose projected top edge is at
    image row v_obs -> metres (None when v_obs lies outside the searched range)"""
    Tcb = np.linalg.inv(np.asarray(T_base_cam, float))
    R = g["r_top"] + g.get("rim_r", 0.0)
    ph = np.linspace(0.0, 2.0 * math.pi, 96, endpoint=False)
    hs = np.arange(h_lo, h_hi + 1e-9, step)
    ring = np.stack([c_xy[0] + R * np.cos(ph), c_xy[1] + R * np.sin(ph), np.zeros_like(ph)], axis=1)
    vt = np.empty(len(hs))
    for i, h in enumerate(hs):
        P = ring.copy()
        P[:, 2] = z0 + h
        _u, v = _proj(Tcb, K, P)
        vt[i] = v.min()
    if not (vt.min() - 2.0 <= v_obs <= vt.max() + 2.0):
        return None
    i = int(np.argmin(np.abs(vt - v_obs)))
    return float(hs[i])


def classify(h_mm, g, ndz_m):
    top = (g["h"] + g.get("rim_r", 0.0)) * 1000.0
    nd = ndz_m * 1000.0
    if h_mm is None:
        return "unknown"
    if top - TOL_LOW_MM <= h_mm < top + 0.5 * nd:
        return "cup"
    if top + 0.5 * nd <= h_mm <= top + nd + TOL_HIGH_MM:
        return "stack"
    return "unknown"


def _contour(mask, eps_px=2.0, max_pts=80):
    YP = _yp()
    cv2 = YP._cv2()
    cs, _h = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return []
    c = max(cs, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, eps_px, True).reshape(-1, 2)
    if len(c) > max_pts:
        c = c[np.linspace(0, len(c) - 1, max_pts).astype(int)]
    return [[int(u), int(v)] for u, v in c]


def gripper_hull_mask(robot_boxes, K, T_base_cam, shape, pad_px=OCCL_PAD_PX):
    """V3.67 PERCEPTION ROBUSTNESS: image region covered by the gripper boxes (projected corners, convex hull, padded)"""
    if not robot_boxes:
        return None
    YP = _yp()
    cv2 = YP._cv2()
    Tcb = np.linalg.inv(np.asarray(T_base_cam, float))
    m = np.zeros(shape[:2], np.uint8)
    for c, R, half in robot_boxes:
        cs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * np.asarray(half)
        P = cs @ np.asarray(R, float).T + np.asarray(c, float)
        pc = P @ Tcb[:3, :3].T + Tcb[:3, 3]
        if np.any(pc[:, 2] <= 0.05):
            continue
        u, v = _proj(Tcb, K, P)
        hull = cv2.convexHull(np.stack([u, v], axis=1).astype(np.float32)).astype(np.int32)
        cv2.fillConvexPoly(m, hull, 1)
    if not m.any():
        return None
    if pad_px:
        m = cv2.dilate(m, np.ones((2 * pad_px + 1, 2 * pad_px + 1), np.uint8))
    return m.astype(bool)


def _tilt(wall, z0, cam_xy, g, top_d):
    """tilt from a low and a high band (NERO_V363); V3.67: also from two half samples -> (tilt, reliable, why)"""
    import NERO_V363_Cup_Perception as CP
    zl, zh = z0 + 0.015, z0 + min(top_d, g["h"]) - 0.020

    def one(W):
        bl, bh = W[W[:, 2] < zl + 0.025], W[W[:, 2] > zh - 0.025]
        if len(bl) <= 40 or len(bh) <= 40 or zh - zl <= 0.05:
            return None, 0, 1e9
        cl, rl = CP.fit_axis(bl, z0, cam_xy, g)
        ch, rh = CP.fit_axis(bh, z0, cam_xy, g)
        t = math.degrees(math.atan2(float(np.linalg.norm(np.asarray(ch) - np.asarray(cl))),
                                    float(np.mean(bh[:, 2]) - np.mean(bl[:, 2]))))
        return t, min(len(bl), len(bh)), max(rl, rh)
    t, n, rms = one(wall)
    if t is None:
        return 0.0, False, "上下两段杯壁点太少"
    ta, _na, _ra = one(wall[0::2])
    tb, _nb, _rb = one(wall[1::2])
    if n < TILT_MIN_BAND_PTS:
        return t, False, f"杯壁点太少（{n}）"
    if rms > TILT_RMS_MAX_M:
        return t, False, f"拟合残差 {rms * 1000:.1f} mm"
    if ta is None or tb is None or abs(ta - tb) > TILT_SPLIT_MAX_DEG:
        return t, False, f"两半样本结果不一致（{ta if ta is None else round(ta, 1)}° / {tb if tb is None else round(tb, 1)}°）"
    return t, True, None


def held_geometry(P, T, g, robot_boxes=None):
    """V3.71 HELD CUP RECHECK: geometry of a cup in the jaws from its mask's depth points (gripper pixels already
    removed): bottom edge = 1st percentile height, cone axis fitted to the LOWER wall (the jaws hold the rim), tilt
    from a low and a high band when reliable -> {held_base_xyz_m, held_axis_rms_mm, held_tilt_deg, ...} or {}"""
    import NERO_V363_Cup_Perception as CP
    if P is None or len(P) < MIN_PTS:
        return {}
    if robot_boxes:
        keep = ~CP._inside_boxes(P, robot_boxes)
        if keep.sum() >= MIN_PTS:
            P = P[keep]
    zb = float(np.percentile(P[:, 2], 1))
    wall = P[(P[:, 2] > zb + 0.006) & (P[:, 2] < zb + g["h"] - 0.025)]
    if len(wall) < MIN_PTS:
        return {"held_note": f"杯壁点太少（{len(wall)}）"}
    if len(wall) > FIT_MAX_PTS:
        wall = wall[np.linspace(0, len(wall) - 1, FIT_MAX_PTS).astype(int)]
    cam_xy = np.asarray(T, float)[:2, 3]
    c, rms = CP.fit_axis(wall, zb, cam_xy, g)
    top_d = min(float(np.percentile(P[:, 2], 98)) - zb, g["h"])
    tilt, t_ok, t_why = _tilt(wall, zb, cam_xy, g, top_d)
    out = {"held_base_xyz_m": [round(float(c[0]), 4), round(float(c[1]), 4), round(zb, 4)],
           "held_axis_rms_mm": round(rms * 1000.0, 2), "held_n_wall": int(len(wall)),
           "held_tilt_deg": round(float(tilt), 1), "held_tilt_reliable": bool(t_ok and rms <= 0.008)}
    if not t_ok:
        out["held_tilt_note"] = t_why
    if rms > 0.008:
        out["held_note"] = f"锥面拟合残差 {rms * 1000:.1f} mm"
    return out


def cup_from_mask(mask, depth, K, T, z0, g, ndz_m, robot_boxes=None, fence_poly=None, hull=None):
    """cup geometry of one YOLOE instance mask -> dict (kind, base / rim, heights, tilt, reason)"""
    import NERO_V363_Cup_Perception as CP
    YP = _yp()
    cv2 = YP._cv2()
    T = np.asarray(T, float)
    out = {"kind": "unknown", "reason": None, "warnings": []}
    occ, top_hidden = 0.0, 0.0
    mask_full = mask
    if hull is not None and mask.any():
        occ = float((mask & hull).sum()) / float(mask.sum())
        cols = np.flatnonzero(mask.any(axis=0))
        tops = np.argmax(mask[:, cols], axis=0)
        # the silhouette top is hidden when the gripper borders it (the mask stops where the jaw starts)
        hid = [bool(hull[max(0, t - 6):t + 1, c].any()) for t, c in zip(tops, cols)]
        top_hidden = float(np.mean(hid)) if hid else 0.0
        out["occluded_frac"], out["top_hidden_frac"] = round(occ, 2), round(top_hidden, 2)
        if occ > 0:
            mask = mask & ~hull                    # V3.67: no gripper pixels in the cup's points
    er = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)   # no edge / flying pixels
    if er.sum() < MIN_PTS:
        er = mask
    P = CP.backproject_mask(depth, er, K, T)
    n_all = len(P)
    if n_all and float(np.percentile(P[:, 2], 1)) - z0 > LIFT_M:
        # the YOLOE mask is the cup itself (the gripper is not in it): its lowest points are the cup's bottom edge
        # V3.69 LIFTED NEAR GRIPPER: ... and a cup in the air is next to the gripper; a standing cup whose low depth
        # is missing (image edge, calibration board) is not "lifted"
        cxy = np.median(P[:, :2], axis=0)
        near = (not robot_boxes) or min(float(np.hypot(*(cxy - np.asarray(c, float)[:2]))) for c, _R, _h in robot_boxes) \
            <= LIFT_NEAR_M
        if near:
            out.update(kind="lifted", reason="杯子离开桌面（在夹爪里）", n_points=int(n_all))
            out.update(held_geometry(P, T, g, robot_boxes))
            return out
        out["warnings"].append("杯底附近没有深度（不在夹爪旁，不算被夹起）")
    P = P[(P[:, 2] > z0 + 0.003) & (P[:, 2] < z0 + g["h"] + ndz_m + 0.05)] if len(P) else P
    out["n_points"] = int(len(P))
    if len(P) < MIN_PTS:
        out.update(kind="no_depth", reason=f"轮廓里有效深度点太少（{len(P)} / {n_all}）")
        return out
    if robot_boxes:                       # mask edge pixels bleeding onto the jaw: drop them if enough points remain
        keep = ~CP._inside_boxes(P, robot_boxes)
        if keep.sum() >= MIN_PTS:
            P = P[keep]
        if keep.mean() < 0.5:
            out["warnings"].append(f"轮廓的 {100 * (1 - keep.mean()):.0f}% 在夹爪范围内")
    wall = P[(P[:, 2] > z0 + 0.008) & (P[:, 2] < z0 + g["h"] - 0.010)]
    if len(wall) < MIN_PTS:
        wall = P
    if len(wall) > FIT_MAX_PTS:                    # V3.68 FAST PERCEPTION: same result, a fraction of the time
        wall = wall[np.linspace(0, len(wall) - 1, FIT_MAX_PTS).astype(int)]
    cam_xy = T[:2, 3]
    if top_hidden >= OCCL_TOP_FRAC:                # gripper at the top of the cup: low wall points only
        low = wall[wall[:, 2] < z0 + g["h"] - 0.030]
        wall = low if len(low) >= MIN_PTS else wall
    c, rms = CP.fit_axis(wall, z0, cam_xy, g)
    c = np.asarray(c, float)
    out["fit_rms_mm"] = round(rms * 1000.0, 2)
    if rms > 0.008:
        out["warnings"].append(f"锥面拟合残差 {rms * 1000:.1f} mm")
    top_d = float(np.percentile(P[:, 2], 98)) - z0
    base = [round(float(c[0]), 4), round(float(c[1]), 4), round(float(z0), 4)]
    if top_hidden >= OCCL_TOP_FRAC:
        # V3.67: the top / silhouette are hidden by the gripper -> position only, not a cup the skill may re-classify
        out.update(kind="occluded", reason=f"杯口被夹爪挡住（只给位置）", base_xyz_m=base,
                   rim_xyz_m=[base[0], base[1], round(float(z0 + g["h"]), 4)], height_mm=None,
                   height_depth_mm=round(top_d * 1000.0, 1), tilt_deg=0.0, tilt_reliable=False)
        return out
    # tilt: separate fits of a low and a high band (as NERO_V363); V3.67: only when it is reliable
    tilt, t_ok, t_why = _tilt(wall, z0, cam_xy, g, top_d)
    out["tilt_measured_deg"] = round(tilt, 1)
    out["tilt_reliable"] = bool(t_ok)
    if not t_ok:
        out["warnings"].append(f"倾斜测不准（{t_why}；测得 {tilt:.1f}°）→ 按直立处理")
        tilt = 0.0
    v_obs, v_min = mask_top_v(mask_full)
    h_sil = silhouette_height(c, z0, K, T, v_obs, g) if v_obs is not None else None
    h_mm = None if h_sil is None else h_sil * 1000.0
    kind = classify(h_mm, g, ndz_m)
    if v_min is not None and v_min <= 2:
        kind, out["reason"] = "edge", "杯口被画面上边缘切掉"
    elif fence_poly and not YP.point_in_poly(float(c[0]), float(c[1]), fence_poly):
        kind, out["reason"] = "outside", "在安全区（围栏）外"
    elif kind == "unknown":
        out["reason"] = ("轮廓顶边对不上杯口" if h_mm is None else
                         f"轮廓量得杯高 {h_mm:.0f} mm，不像单杯（≈{(g['h'] + g.get('rim_r', 0)) * 1000:.0f}）"
                         f"也不像叠好的一对（≈{(g['h'] + g.get('rim_r', 0) + ndz_m) * 1000:.0f}）")
    rim_z = z0 + g["h"] + (ndz_m if kind == "stack" else 0.0)
    out.update(kind=kind, base_xyz_m=[round(float(c[0]), 4), round(float(c[1]), 4), round(float(z0), 4)],
               rim_xyz_m=[round(float(c[0]), 4), round(float(c[1]), 4), round(float(rim_z), 4)],
               height_mm=None if h_mm is None else round(h_mm, 1), height_depth_mm=round(top_d * 1000.0, 1),
               tilt_deg=round(tilt, 1), mask_top_v=v_obs)
    return out


# ================================================================================ main entry
def perceive_cups_yolo(cam, robot_boxes=None, cup=None, nested_dz=None, hull_boxes=None):
    """cam = {"rgb", "depths": [depth frames (m)], "K", "T_base_cam", "z_table", "fence_poly", "calib_note",
    "yolo": yolo_cfg_from_tab(...)} -> {"cups": [...], "dets": [...], "table_z_m", "diag", "yolo": text}"""
    YP = _yp()
    ycfg = cam.get("yolo")
    if not ycfg:
        raise YoloCupUnavailable("YOLOE 未配置（Camera调试 → 视觉定位）")
    if ycfg.get("error"):
        raise YoloCupUnavailable(ycfg["error"])
    rgb = np.asarray(cam["rgb"])
    depths = [np.asarray(d, np.float32) for d in (cam.get("depths") or [])]
    if not depths:
        raise RuntimeError("没有深度帧")
    K = dict(cam["K"])
    T = np.asarray(cam["T_base_cam"], float)
    det = ycfg["detector"]
    import time as _t
    tm = {"grab_s": cam.get("grab_s")}
    t0 = _t.perf_counter()
    try:
        d2 = det.detect(rgb, ycfg["mode"], ycfg["path"], prompts=ycfg["prompts"], conf=ycfg["conf"])
    except YP.YoloUnavailable as exc:
        raise YoloCupUnavailable(str(exc))
    t1 = _t.perf_counter()
    depth = YP.fuse_depth(depths, d2) if len(depths) > 1 else depths[0]
    zt = cam.get("z_table")
    fence = cam.get("fence_poly")
    t2 = _t.perf_counter()
    res, diag = YP.locate_objects(d2, depth, K, T, table_z=zt, fence_poly=fence, calib_note=cam.get("calib_note"))
    t3 = _t.perf_counter()
    tm.update(yolo_s=t1 - t0, fuse_s=t2 - t1, locate_s=t3 - t2)
    g = SC.cup_geometry(cup)
    ndz = float(g["nested_dz"] if nested_dz is None else nested_dz)
    # V3.67 PERCEPTION ROBUSTNESS: image region of the gripper (tight boxes when given, else the filter boxes)
    hull = gripper_hull_mask(hull_boxes if hull_boxes is not None else robot_boxes, K, T, rgb.shape)
    items = []
    for d in d2:
        if d.get("cls", 0) != 0:
            continue
        r = min(res, key=lambda r_: sum(abs(a - b) for a, b in zip(r_["yolo"]["box"], [round(v) for v in d["box"]])))
        items.append({"d": d, "r": r})
    # first pass: one cup per mask
    for it in items:
        r = it["r"]
        z0 = r.get("ground_z_m")
        if z0 is None:
            z0 = float(zt or 0.0)
        it["z0"] = float(z0)
        it["cup"] = (cup_from_mask(it["d"]["mask"], depth, K, T, it["z0"], g, ndz, robot_boxes, fence, hull)
                     if r["status"] != "rejected" else {"kind": "rejected", "reason": "；".join(r["reasons"])})
    # merge masks of the same cup / nested pair
    groups = []
    for it in items:
        b = it["cup"].get("base_xyz_m")
        for gr in groups:
            gb = gr[0]["cup"].get("base_xyz_m")
            if b and gb and math.hypot(b[0] - gb[0], b[1] - gb[1]) < MERGE_M:
                gr.append(it)
                break
        else:
            groups.append([it])
    cups, dets = [], []
    for gr in groups:
        best = max(gr, key=lambda it_: it_["d"]["conf"])
        mask = best["d"]["mask"]
        c = best["cup"]
        if len(gr) > 1:
            mask = np.logical_or.reduce([it_["d"]["mask"] for it_ in gr])
            c = cup_from_mask(mask, depth, K, T, best["z0"], g, ndz, robot_boxes, fence, hull)
            c["warnings"] = list(c.get("warnings") or []) + [f"{len(gr)} 个 YOLOE 框合并为一个"]
        r = best["r"]
        e = dict(c, conf=round(float(best["d"]["conf"]), 3), name=best["d"].get("name"),
                 box=[round(v) for v in best["d"]["box"]], contour=_contour(mask),
                 yolo_xyz_m=r.get("xyz_m"), yolo_height_mm=r.get("height_mm"), yolo_status=r.get("status"))
        if r.get("reasons") and r["status"] != "rejected":
            e["warnings"] = list(e.get("warnings") or []) + list(r["reasons"])
        dets.append(e)
        if c.get("base_xyz_m"):
            cups.append(e)
    cups.sort(key=lambda d_: d_["base_xyz_m"][1])
    for i, e in enumerate(dets):
        e["det_id"] = f"#{i + 1}"
    gz = diag.get("ground_z_m")
    tm["cup_s"] = _t.perf_counter() - t3                           # V3.68: timings for the log
    return {"timing": {k: (None if v is None else round(float(v), 3)) for k, v in tm.items()},
            "hull": hull, "cups": cups, "dets": dets, "table_z_m": round(float(gz if gz is not None else (zt or 0.0)), 4),
            "n_2d": len(d2), "diag": diag, "yolo": describe_cfg(ycfg),
            "device": (getattr(det, "info", {}) or {}).get("device")}


def needs_snapshot(res):
    """V3.67: a perception worth keeping for offline analysis (something was not understood)"""
    for e in (res or {}).get("dets") or []:
        if e.get("kind") in ("unknown", "edge", "no_depth") or any("倾斜测不准" in w for w in e.get("warnings") or []):
            return True
    return not (res or {}).get("dets")


def save_snapshot(cam, res, tag, folder=None, keep=SNAP_KEEP):
    """V3.67 FAILURE SNAPSHOT: colour (png), depth (uint16 mm png), YOLOE masks (npz), result + camera (json) into
    NERO_CALIB_DIR/real_cup_logs/<time>_<tag>/ (never into the release folder); keeps the newest `keep` -> path"""
    import json
    import shutil
    import time as _t
    YP = _yp()
    cv2 = YP._cv2()
    root = folder or os.path.join(os.environ.get("NERO_CALIB_DIR") or os.path.expanduser("~/Downloads/cameracalib"),
                                  "real_cup_logs")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(tag))[:40]
    d = os.path.join(root, f"{_t.strftime('%Y%m%d_%H%M%S')}_{int(_t.time() * 1000) % 1000:03d}_{safe}")
    os.makedirs(d, exist_ok=True)
    rgb = np.asarray(cam.get("rgb")) if cam and cam.get("rgb") is not None else None
    if rgb is not None:
        cv2.imwrite(os.path.join(d, "color.png"), np.ascontiguousarray(rgb[..., ::-1]))
    deps = (cam or {}).get("depths") or []
    if deps:
        dm = np.clip(np.asarray(deps[-1], float) * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(d, "depth_mm.png"), dm)
    masks = [e for e in (res or {}).get("dets") or [] if e.get("contour")]
    meta = {"tag": tag, "time": _t.strftime("%Y-%m-%d %H:%M:%S"), "K": (cam or {}).get("K"),
            "T_base_cam": np.asarray((cam or {}).get("T_base_cam")).tolist() if cam and cam.get("T_base_cam") is not None else None,
            "z_table": (cam or {}).get("z_table"), "fence_poly": (cam or {}).get("fence_poly"),
            "summary": summary(res) if res else None,
            "dets": [{k: v for k, v in e.items() if k != "contour"} | {"contour": e.get("contour")} for e in masks],
            "table_z_m": (res or {}).get("table_z_m")}
    with open(os.path.join(d, "result.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=str)
    try:
        subs = sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)))
        for x in subs[:-keep] if len(subs) > keep else []:
            shutil.rmtree(os.path.join(root, x), ignore_errors=True)
    except OSError:
        pass
    return d


def timing_text(res):
    """V3.68: 'grab 0.21 · YOLOE 0.08 · fuse 0.02 · 3D 0.31 · cup 0.12 s'"""
    t = (res or {}).get("timing") or {}
    parts = [(n, t.get(k)) for n, k in (("帧", "grab_s"), ("YOLOE", "yolo_s"), ("融合", "fuse_s"), ("3D", "locate_s"),
                                        ("杯", "cup_s"))]
    return " · ".join(f"{n} {v:.2f}" for n, v in parts if v is not None) + " s"


def summary(res):
    """one log line per perception"""
    parts = []
    for e in res.get("dets") or []:
        h = e.get("height_mm")
        hd = e.get("height_depth_mm")
        xy = e.get("base_xyz_m") or e.get("held_base_xyz_m")     # V3.71: the held cup's measured position too
        parts.append(f"{e['det_id']} {e['kind']}" + (f"@({xy[0]:+.3f},{xy[1]:+.3f})" if xy else "")
                     + (f" h={h:.0f}mm" if h is not None else "") + (f"(深度 {hd:.0f})" if hd is not None else "")
                     + f" conf {e.get('conf', 0):.2f}" + (f" [{e['reason']}]" if e.get("reason") else ""))
    return " ".join(parts) or "no cup（YOLOE 没有找到杯子）"
