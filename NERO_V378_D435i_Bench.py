#!/usr/bin/env python3
"""NERO V3.78 D435i BENCH - the fixed D435i on its own (no robot, no console): fixed camera parameters, depth filter
comparison, egg-locating accuracy against a printed board, training-image capture.

V3.78 D435I BENCH

Why (user, V4 preparation group 4 "perception accuracy", in the office with only the D435i): every later improvement
must be measured, and every training image must be taken under ONE fixed camera setting.

    python3 NERO_V378_D435i_Bench.py profile     lock exposure / gain / white balance / 60 Hz anti-flicker, save
    python3 NERO_V378_D435i_Bench.py noise       depth noise on a flat surface: raw vs filters vs multi-frame median
    python3 NERO_V378_D435i_Bench.py accuracy    egg centre vs a known board corner (ChArUco 7x5 32 mm on the desk)
    python3 NERO_V378_D435i_Bench.py capture     training images (+ aligned depth) under the saved profile

  * profile -> NERO_CALIB_DIR/d435i_profile.json (default ~/Downloads/cameracalib); every mode applies it on start
  * results  -> NERO_CALIB_DIR/d435i_bench/<date>/<mode>_<time>.csv (+ .json summary)
  * images   -> NERO_V378_DATA_DIR (default ~/Downloads/objectmodels/datasets)/<session>/
  * the release folder is never written (runtime read-only rule)
  * accuracy uses the Camera tab's pipeline: YOLOE (NERO_V347 YoloEDetector, text prompt "egg") + NERO_V347.locate
    (egg centre from the silhouette + the local table plane + the egg size 55 x 42 mm) with the BOARD as the base frame
    (board centre, z up, table = z 0); the expected centre = the chosen inner corner, at half the egg width (lying)
"""
import csv
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

VERSION = "V3.78"
MARKER = "V3.78 D435I BENCH"
CALIB_DIR = os.environ.get("NERO_CALIB_DIR") or os.path.join(os.path.expanduser("~"), "Downloads", "cameracalib")
PROFILE_PATH = os.path.join(CALIB_DIR, "d435i_profile.json")
BENCH_DIR = os.path.join(CALIB_DIR, "d435i_bench")
DATA_DIR = os.environ.get("NERO_V378_DATA_DIR") or os.path.join(os.path.expanduser("~"), "Downloads", "objectmodels",
                                                                "datasets")
COLOR_SIZE = (1280, 720)
DEPTH_SIZE = (1280, 720)                 # V3.53 default: 1280x720 High Accuracy
POWER_LINE = {0: "off", 1: "50 Hz", 2: "60 Hz", 3: "auto"}
DEFAULT_PROFILE = {
    "version": VERSION,
    "color": {"enable_auto_exposure": 0, "exposure": 156.0, "gain": 64.0, "enable_auto_white_balance": 0,
              "white_balance": 4600.0, "power_line_frequency": 2},          # 2 = 60 Hz (mains of the user's lab)
    "depth": {"visual_preset": "high_accuracy", "laser_power": 150.0, "enable_auto_exposure": 1},
    "filters": {"spatial": True, "spatial_magnitude": 2, "spatial_alpha": 0.5, "spatial_delta": 20,
                "temporal": True, "temporal_alpha": 0.4, "temporal_delta": 20,
                "hole_filling": False, "hole_filling_mode": 1, "median_frames": 5},
    "locked": False, "note": "",
}
EGG_HALF_WIDTH_M = 0.021                # lying egg: centre 21 mm above the table (width 42 mm, measured)
EGG_HALF_LENGTH_M = 0.0275              # standing egg


# ============================================================================================ profile
def load_profile(path=None):
    p = path or PROFILE_PATH
    prof = json.loads(json.dumps(DEFAULT_PROFILE))
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        for k in ("color", "depth", "filters"):
            prof[k].update(d.get(k) or {})
        prof["locked"] = bool(d.get("locked", False))
        prof["note"] = str(d.get("note", ""))
        prof["saved_at"] = d.get("saved_at")
    except Exception:
        pass
    return prof


def save_profile(prof, path=None):
    p = path or PROFILE_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    d = dict(prof, version=VERSION, saved_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, p)
    return p


def _sensors(device, rs):
    out = {}
    for s in device.query_sensors():
        try:
            name = s.get_info(rs.camera_info.name)
        except Exception:
            name = "?"
        if "RGB" in name or "Color" in name:
            out["color"] = s
        elif "Stereo" in name or "Depth" in name:
            out["depth"] = s
    return out


def apply_profile(device, prof, rs, log=print):
    """set every option the sensor supports (clamped to its range); auto modes off BEFORE the manual values.
    -> [(sensor, option, requested, applied or None, note)]"""
    rows = []
    sens = _sensors(device, rs)
    order = {"color": ["enable_auto_exposure", "enable_auto_white_balance", "exposure", "gain", "white_balance",
                       "power_line_frequency", "gamma"],                 # V3.82 GAMMA (only when the profile has it)
             "depth": ["visual_preset", "enable_auto_exposure", "laser_power"]}
    for sk, keys in order.items():
        s = sens.get(sk)
        if s is None:
            rows.append((sk, "*", None, None, "sensor not found"))
            continue
        for k in keys:
            if k not in prof.get(sk, {}):
                continue
            v = prof[sk][k]
            opt = getattr(rs.option, k, None)
            if opt is None or not s.supports(opt):
                rows.append((sk, k, v, None, "not supported"))
                continue
            try:
                if k == "visual_preset" and isinstance(v, str):
                    v = float(getattr(rs.rs400_visual_preset, v))
                r = s.get_option_range(opt)
                vv = min(max(float(v), r.min), r.max)
                if r.step > 0:
                    vv = r.min + round((vv - r.min) / r.step) * r.step
                s.set_option(opt, vv)
                rows.append((sk, k, v, round(float(s.get_option(opt)), 3), "" if abs(vv - float(v)) < 1e-6 else
                             f"clamped to {r.min:g}..{r.max:g}"))
            except Exception as exc:
                rows.append((sk, k, v, None, str(exc)[:80]))
    bad = [r for r in rows if r[3] is None]
    log(f"[{'OK' if not bad else 'WARN'}] [V3.78 D435i] profile applied: "
        + ", ".join(f"{r[0]}.{r[1]}={r[3]}" for r in rows if r[3] is not None)
        + (f" | not applied: {', '.join(f'{r[0]}.{r[1]} ({r[4]})' for r in bad)}" if bad else ""))
    return rows


def read_back(device, rs):
    out = {}
    for sk, s in _sensors(device, rs).items():
        out[sk] = {}
        for k in ("enable_auto_exposure", "exposure", "gain", "enable_auto_white_balance", "white_balance",
                  "power_line_frequency", "laser_power", "visual_preset", "gamma"):
            opt = getattr(rs.option, k, None)
            try:
                if opt is not None and s.supports(opt):
                    out[sk][k] = round(float(s.get_option(opt)), 3)
            except Exception:
                pass
    return out


# ============================================================================================ depth
class FilterChain:
    """RealSense post-processing in the order Intel recommends (no decimation: it would break the colour alignment)"""

    def __init__(self, rs, cfg):
        self.rs, self.cfg = rs, dict(cfg)
        self.spatial = self.temporal = self.holes = None
        if cfg.get("spatial"):
            self.spatial = rs.spatial_filter()
            self._set(self.spatial, rs.option.filter_magnitude, cfg.get("spatial_magnitude"))
            self._set(self.spatial, rs.option.filter_smooth_alpha, cfg.get("spatial_alpha"))
            self._set(self.spatial, rs.option.filter_smooth_delta, cfg.get("spatial_delta"))
        if cfg.get("temporal"):
            self.temporal = rs.temporal_filter()
            self._set(self.temporal, rs.option.filter_smooth_alpha, cfg.get("temporal_alpha"))
            self._set(self.temporal, rs.option.filter_smooth_delta, cfg.get("temporal_delta"))
        if cfg.get("hole_filling"):
            self.holes = rs.hole_filling_filter()
            self._set(self.holes, rs.option.holes_fill, cfg.get("hole_filling_mode"))

    @staticmethod
    def _set(f, opt, v):
        if v is None:
            return
        try:
            f.set_option(opt, float(v))
        except Exception:
            pass

    def process(self, d):
        for f in (self.spatial, self.temporal, self.holes):
            if f is not None:
                d = f.process(d)
        return d

    def label(self):
        on = [n for n, f in (("spatial", self.spatial), ("temporal", self.temporal), ("holes", self.holes)) if f]
        return "+".join(on) or "raw"


def median_stack(depths):
    """per-pixel median over frames, zeros (no depth) ignored; a pixel with no valid sample stays 0"""
    a = np.asarray(depths, dtype=np.float32)
    if a.ndim == 2:
        return a
    a = np.where(a > 0, a, np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmedian(a, axis=0)
    return np.nan_to_num(m, nan=0.0).astype(np.float32)


def deproject(depth_m, K, roi):
    x0, y0, x1, y1 = [int(v) for v in roi]
    sub = depth_m[y0:y1, x0:x1]
    vv, uu = np.mgrid[y0:y1, x0:x1]
    ok = sub > 0
    z = sub[ok]
    x = (uu[ok] - K["ppx"]) / K["fx"] * z
    y = (vv[ok] - K["ppy"]) / K["fy"] * z
    return np.column_stack([x, y, z])


def plane_rms_mm(P):
    """RMS distance (mm) of the points to their best-fit plane (SVD); flatness of a flat target"""
    if len(P) < 10:
        return None
    c = P.mean(axis=0)
    _u, _s, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[-1]
    return float(np.sqrt(np.mean(((P - c) @ n) ** 2)) * 1000.0)


def noise_report(stack, K, roi):
    """stack: frames (m) of a STILL flat target -> valid %, temporal std (median over pixels, mm), flatness (mm),
    mean distance (mm)"""
    a = np.asarray(stack, dtype=np.float32)
    x0, y0, x1, y1 = [int(v) for v in roi]
    sub = a[:, y0:y1, x0:x1]
    valid = sub > 0
    pct = float(valid.mean() * 100.0)
    with np.errstate(all="ignore"):
        s = np.where(valid, sub, np.nan)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            std = np.nanstd(s, axis=0)
            mean = np.nanmean(s, axis=0)
    tstd = float(np.nanmedian(std) * 1000.0) if np.isfinite(std).any() else None
    mean_full = np.zeros(a.shape[1:], np.float32)
    mean_full[y0:y1, x0:x1] = np.nan_to_num(mean, nan=0.0)
    flat = plane_rms_mm(deproject(mean_full, K, roi))
    md = float(np.nanmean(mean) * 1000.0) if np.isfinite(mean).any() else None
    return {"valid_pct": round(pct, 1), "temporal_std_mm": None if tstd is None else round(tstd, 3),
            "flatness_rms_mm": None if flat is None else round(flat, 3), "mean_dist_mm": None if md is None else round(md, 1)}


# ============================================================================================ board frame
def board_spec():
    import NERO_V349_Real_Wrist_Camera as W
    return dict(W.DEFAULT_BOARD)


def inner_corners(cfg=None):
    """ChArUco inner corners in the BENCH frame (board centre, x along the board columns, y = -board rows direction,
    z up): {id: (x, y)}; id = row * (cols - 1) + col like OpenCV"""
    cfg = dict(board_spec(), **(cfg or {}))
    cols, rows, sq = int(cfg["cols"]), int(cfg["rows"]), float(cfg["square_mm"]) / 1000.0
    cx, cy = 0.5 * cols * sq, 0.5 * rows * sq
    out = {}
    for r in range(rows - 1):
        for c in range(cols - 1):
            out[r * (cols - 1) + c] = ((c + 1) * sq - cx, -((r + 1) * sq - cy))
    return out


def T_bench_cam_from_pose(rvec, t_pnp):
    """board pose in the camera (OpenCV: z into the board) -> T_bench_cam with the bench frame z UP (180 deg about x)"""
    import NERO_V347_Yolo_Perception as YP
    cv2 = YP._cv2()
    R, _ = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))
    T_cam_board = np.eye(4)
    T_cam_board[:3, :3], T_cam_board[:3, 3] = R, np.asarray(t_pnp, float).reshape(3)
    flip = np.diag([1.0, -1.0, -1.0, 1.0])
    return flip @ np.linalg.inv(T_cam_board)


def board_pose(rgbs, K, dist=None):
    """average board pose over several frames (camera and board still) -> (T_bench_cam, info) or (None, why)"""
    import NERO_V349_Real_Wrist_Camera as W
    Ts, n = [], []
    for rgb in rgbs:
        det = W.detect_charuco_pose(rgb, K, dist)
        if det and det.get("rvec") is not None and det.get("p_pnp") is not None:
            Ts.append(T_bench_cam_from_pose(det["rvec"], det["p_pnp"]))
            n.append(int(det.get("n_corners", 0)))
    if not Ts:
        return None, "ChArUco board not found (whole board in view, flat, no glare?)"
    t = np.mean([T[:3, 3] for T in Ts], axis=0)
    U, _s, Vt = np.linalg.svd(np.sum([T[:3, :3] for T in Ts], axis=0))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = U @ Vt, t
    spread = float(max(np.linalg.norm(Ti[:3, 3] - t) for Ti in Ts) * 1000.0)
    return T, {"frames": len(Ts), "corners": int(np.median(n)), "spread_mm": round(spread, 2),
               "cam_height_mm": round(float(T[2, 3]) * 1000.0, 1)}     # camera above the board


def nearest_corner(xy, corners):
    k = min(corners, key=lambda i: math.hypot(xy[0] - corners[i][0], xy[1] - corners[i][1]))
    return k, corners[k]


def accuracy_row(xyz, corners, expected_z):
    """located egg centre (bench frame) -> errors against the nearest inner corner at the expected height"""
    k, (cx, cy) = nearest_corner(xyz, corners)
    dx, dy, dz = float(xyz[0] - cx) * 1000.0, float(xyz[1] - cy) * 1000.0, float(xyz[2] - expected_z) * 1000.0
    return {"corner": k, "dx_mm": round(dx, 2), "dy_mm": round(dy, 2), "dz_mm": round(dz, 2),
            "xy_err_mm": round(math.hypot(dx, dy), 2), "err_3d_mm": round(math.sqrt(dx * dx + dy * dy + dz * dz), 2)}


def summarize(rows, key="err_3d_mm"):
    v = [r[key] for r in rows if r.get(key) is not None]
    if not v:
        return {"n": 0}
    a = np.asarray(v, float)
    return {"n": len(v), "mean": round(float(a.mean()), 2), "std": round(float(a.std()), 2),
            "max": round(float(a.max()), 2), "p90": round(float(np.percentile(a, 90)), 2)}


def write_results(mode, rows, summary, folder=None):
    d = folder or os.path.join(BENCH_DIR, time.strftime("%Y%m%d"))
    os.makedirs(d, exist_ok=True)
    stem = os.path.join(d, f"{mode}_{time.strftime('%H%M%S')}")
    if rows:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        with open(stem + ".csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump({"version": VERSION, "mode": mode, "summary": summary, "profile": load_profile()}, f,
                  ensure_ascii=False, indent=1)
    return stem


# ============================================================================================ camera
class D435i:
    """aligned colour + depth (m) with the saved profile; filters switchable without a restart"""

    def __init__(self, prof=None, log=print):
        import pyrealsense2 as rs
        self.rs, self.log = rs, log
        self.prof = prof or load_profile()
        ctx = rs.context()
        devs = list(ctx.query_devices())
        if not devs:
            raise RuntimeError("没有找到 RealSense 相机（USB 线 / USB3 口？realsense-viewer 是否还开着？）")
        self.dev = devs[0]
        self.serial = self.dev.get_info(rs.camera_info.serial_number)
        self.pipe, cfg = rs.pipeline(ctx), rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, COLOR_SIZE[0], COLOR_SIZE[1], rs.format.rgb8, 30)
        cfg.enable_stream(rs.stream.depth, DEPTH_SIZE[0], DEPTH_SIZE[1], rs.format.z16, 30)
        prof_rs = self.pipe.start(cfg)
        self.applied = apply_profile(self.dev, self.prof, rs, log=log)
        self.scale = prof_rs.get_device().first_depth_sensor().get_depth_scale()
        intr = prof_rs.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx,
                  "ppy": intr.ppy}
        self.dist = [0.0] * 5
        self.align = rs.align(rs.stream.color)
        self.chain = FilterChain(rs, self.prof["filters"])
        for _ in range(15):                       # let auto depth exposure / the filters settle
            self.pipe.wait_for_frames(3000)

    def set_filters(self, cfg):
        self.chain = FilterChain(self.rs, cfg)

    def grab(self, filtered=True):
        fs = self.align.process(self.pipe.wait_for_frames(3000))
        c, d = fs.get_color_frame(), fs.get_depth_frame()
        if filtered:
            d = self.chain.process(d)
        return (np.asanyarray(c.get_data()).copy(),
                np.asanyarray(d.get_data()).astype(np.float32) * self.scale)

    def grab_n(self, n, filtered=True):
        rgbs, depths = [], []
        for _ in range(int(n)):
            r, d = self.grab(filtered)
            rgbs.append(r)
            depths.append(d)
        return rgbs, depths

    def close(self):
        try:
            self.pipe.stop()
        except Exception:
            pass


def image_stats(rgb):
    g = rgb.mean(axis=2)
    return {"mean": round(float(g.mean()), 1), "sat_pct": round(float((rgb >= 250).any(axis=2).mean() * 100.0), 2),
            "dark_pct": round(float((g <= 5).mean() * 100.0), 2)}


# ============================================================================================ modes (interactive)
def _cv2():
    import NERO_V347_Yolo_Perception as YP
    return YP._cv2()


def _show(cv2, rgb, lines, scale=0.6):
    img = np.ascontiguousarray(rgb[..., ::-1])
    img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
    for i, t in enumerate(lines):
        cv2.putText(img, t, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, t, (10, 22 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.imshow("NERO V3.78 D435i bench", img)
    return cv2.waitKey(30) & 0xFF


def mode_profile(cam):
    cv2, rs = _cv2(), cam.rs
    col = cam.prof["color"]
    print("[V3.78 profile] keys: a = auto settle 3 s then lock | e/E exposure -/+ | g/G gain -/+ | w/W white balance "
          "-/+ | f anti-flicker 50/60 Hz | s save | q quit")
    while True:
        rgb, _d = cam.grab(False)
        st = image_stats(rgb)
        cur = read_back(cam.dev, rs).get("color", {})
        k = _show(cv2, rgb, [f"exposure {cur.get('exposure')}  gain {cur.get('gain')}  WB {cur.get('white_balance')}  "
                             f"AE {cur.get('enable_auto_exposure')} AWB {cur.get('enable_auto_white_balance')}  "
                             f"flicker {POWER_LINE.get(int(cur.get('power_line_frequency', -1)), '?')}",
                             f"mean brightness {st['mean']} (aim 110-150)  saturated {st['sat_pct']}% (aim < 0.5%)",
                             "a auto-settle+lock | e/E g/G w/W | f 50/60Hz | s save | q quit"])
        if k == ord("q"):
            break
        if k == ord("a"):
            auto = dict(col, enable_auto_exposure=1, enable_auto_white_balance=1)
            apply_profile(cam.dev, dict(cam.prof, color=auto), rs, log=lambda s: None)
            t0 = time.time()
            while time.time() - t0 < 3.0:
                cam.grab(False)
            cur = read_back(cam.dev, rs).get("color", {})
            for k_ in ("exposure", "gain", "white_balance"):
                if k_ in cur:
                    col[k_] = cur[k_]
            col.update(enable_auto_exposure=0, enable_auto_white_balance=0)
            apply_profile(cam.dev, cam.prof, rs)
            print(f"[V3.78 profile] auto settled and locked: exposure {col['exposure']} gain {col['gain']} "
                  f"WB {col['white_balance']}")
        steps = {ord("e"): ("exposure", 0.9), ord("E"): ("exposure", 1.1), ord("g"): ("gain", 0.9),
                 ord("G"): ("gain", 1.1), ord("w"): ("white_balance", -100), ord("W"): ("white_balance", 100)}
        if k in steps:
            key, f = steps[k]
            col[key] = col[key] * f if key != "white_balance" else col[key] + f
            col.update(enable_auto_exposure=0, enable_auto_white_balance=0)
            apply_profile(cam.dev, cam.prof, rs, log=lambda s: None)
        if k == ord("f"):
            col["power_line_frequency"] = 1 if int(col.get("power_line_frequency", 2)) == 2 else 2
            apply_profile(cam.dev, cam.prof, rs, log=lambda s: None)
        if k == ord("s"):
            cam.prof["locked"] = not col.get("enable_auto_exposure") and not col.get("enable_auto_white_balance")
            p = save_profile(cam.prof)
            print(f"[OK] [V3.78 profile] saved {p} | {json.dumps(read_back(cam.dev, rs), ensure_ascii=False)}")


FILTER_SETS = [("raw", {"spatial": False, "temporal": False, "hole_filling": False}, 1),
               ("spatial+temporal", {"spatial": True, "temporal": True, "hole_filling": False}, 1),
               ("spatial+temporal+holes", {"spatial": True, "temporal": True, "hole_filling": True}, 1),
               ("spatial+temporal + median 5", {"spatial": True, "temporal": True, "hole_filling": False}, 5),
               ("raw + median 10", {"spatial": False, "temporal": False, "hole_filling": False}, 10)]


def mode_noise(cam, frames=30):
    cv2 = _cv2()
    w, h = cam.K["width"], cam.K["height"]
    roi = [w // 2 - 80, h // 2 - 80, w // 2 + 80, h // 2 + 80]
    print("[V3.78 noise] point the camera at a FLAT matte surface (desk / card), keep everything still; "
          "m = measure all filter settings | q quit")
    while True:
        rgb, _d = cam.grab()
        img = rgb.copy()
        img[roi[1]:roi[1] + 3, roi[0]:roi[2]] = 255
        img[roi[3] - 3:roi[3], roi[0]:roi[2]] = 255
        img[roi[1]:roi[3], roi[0]:roi[0] + 3] = 255
        img[roi[1]:roi[3], roi[2] - 3:roi[2]] = 255
        k = _show(cv2, img, ["m = measure (keep still)  q = quit", "ROI = white box"])
        if k == ord("q"):
            break
        if k != ord("m"):
            continue
        rows = []
        for name, fcfg, med in FILTER_SETS:
            cam.set_filters(dict(cam.prof["filters"], **fcfg))
            for _ in range(10):
                cam.grab()                          # temporal filter warm-up
            _r, depths = cam.grab_n(frames * med)
            stack = [median_stack(depths[i:i + med]) for i in range(0, len(depths), med)] if med > 1 else depths
            rep = noise_report(stack, cam.K, roi)
            rows.append(dict(setting=name, frames=len(stack), **rep))
            print(f"  {name:30s} valid {rep['valid_pct']:5.1f}%  temporal std {rep['temporal_std_mm']} mm  "
                  f"flatness {rep['flatness_rms_mm']} mm  distance {rep['mean_dist_mm']} mm")
        cam.set_filters(cam.prof["filters"])
        stem = write_results("noise", rows, {"roi": roi})
        print(f"[OK] [V3.78 noise] saved {stem}.csv")


def mode_accuracy(cam, model=None, prompt="egg", trials=10, standing=False):
    import NERO_V347_Yolo_Perception as YP
    cv2 = _cv2()
    det = YP.YoloEDetector(log=print)
    path = YP.find_model(None, model or YP.DEFAULT_MODEL)
    if not path:
        print(f"[FAIL] [V3.78 accuracy] YOLOE model {model or YP.DEFAULT_MODEL} not found in {YP.model_dirs(None)}")
        return
    corners = inner_corners()
    T = None
    exp_z = EGG_HALF_LENGTH_M if standing else EGG_HALF_WIDTH_M
    print("[V3.78 accuracy] 1) board flat on the desk, NO egg: b = lock the board pose  2) put the egg on an inner "
          "corner (marked), " + ("standing" if standing else "lying") + "  3) m = measure  |  t = toggle lying/standing"
          "  q = quit")
    rows_all = []
    while True:
        rgb, depth = cam.grab()
        lines = [f"board: {'LOCKED' if T is not None else 'b = lock (no egg on the board)'}   egg "
                 f"{'standing' if exp_z == EGG_HALF_LENGTH_M else 'lying'} (t)   m = measure   q = quit"]
        img = rgb.copy()
        if T is not None:
            Tc = np.linalg.inv(T)
            for i, (x, y) in corners.items():
                pc = Tc @ np.array([x, y, 0.0, 1.0])
                u = int(cam.K["fx"] * pc[0] / pc[2] + cam.K["ppx"])
                v = int(cam.K["fy"] * pc[1] / pc[2] + cam.K["ppy"])
                cv2.circle(img, (u, v), 5, (255, 0, 0), -1)
                cv2.putText(img, str(i), (u + 6, v - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        k = _show(cv2, img, lines)
        if k == ord("q"):
            break
        if k == ord("t"):
            exp_z = EGG_HALF_WIDTH_M if exp_z == EGG_HALF_LENGTH_M else EGG_HALF_LENGTH_M
        if k == ord("b"):
            rgbs, _d = cam.grab_n(10)
            T, info = board_pose(rgbs, cam.K, cam.dist)
            print(f"[{'OK' if T is not None else 'FAIL'}] [V3.78 accuracy] board pose: {info}")
        if k == ord("m"):
            if T is None:
                print("[WARN] [V3.78 accuracy] lock the board pose first (b, without the egg)")
                continue
            rows = []
            for name, med in (("filters + median 5", 5), ("filters, 1 frame", 1)):
                for i in range(int(trials)):
                    rgbs, depths = cam.grab_n(med)
                    dets = det.detect(rgbs[-1], "text", path, prompts=[prompt], conf=0.25)
                    d_m = median_stack(depths) if med > 1 else depths[-1]
                    res, _diag = YP.locate(dets, d_m, cam.K, T, table_z=0.0)
                    ok = [r for r in res if r.get("xyz_m") is not None and r.get("status") != "rejected"]
                    if not ok:
                        rows.append({"setting": name, "trial": i, "detected": False})
                        continue
                    best = min(ok, key=lambda r: accuracy_row(r["xyz_m"], corners, exp_z)["xy_err_mm"])
                    rows.append(dict({"setting": name, "trial": i, "detected": True,
                                      "x_mm": round(best["xyz_m"][0] * 1000, 2), "y_mm": round(best["xyz_m"][1] * 1000, 2),
                                      "z_mm": round(best["xyz_m"][2] * 1000, 2)},
                                     **accuracy_row(best["xyz_m"], corners, exp_z)))
            summ = {}
            for name in ("filters + median 5", "filters, 1 frame"):
                rr = [r for r in rows if r["setting"] == name]
                det_rate = sum(1 for r in rr if r.get("detected")) / max(1, len(rr))
                summ[name] = {"detected_pct": round(det_rate * 100, 1), "err_3d": summarize(rr),
                              "xy": summarize(rr, "xy_err_mm"), "dz": summarize(rr, "dz_mm")}
                print(f"  {name:22s} detected {summ[name]['detected_pct']}%  3D err {summ[name]['err_3d']}  "
                      f"xy {summ[name]['xy']}  dz {summ[name]['dz']}")
            rows_all += rows
            stem = write_results("accuracy", rows, summ)
            print(f"[OK] [V3.78 accuracy] saved {stem}.csv (corner {rows[0].get('corner') if rows else '?'}; "
                  "the error includes how exactly the egg sits on the corner - about +-1-2 mm by hand)")


def mode_capture(cam, session=None, with_depth=True):
    cv2 = _cv2()
    sess = session or time.strftime("%Y%m%d_%H%M%S")
    d = os.path.join(DATA_DIR, sess)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "camera.json"), "w", encoding="utf-8") as f:
        json.dump({"version": VERSION, "serial": cam.serial, "K": cam.K, "profile": cam.prof,
                   "read_back": read_back(cam.dev, cam.rs)}, f, ensure_ascii=False, indent=1)
    if not cam.prof.get("locked"):
        print("[WARN] [V3.78 capture] the camera profile is not locked (run `profile`, lock and save first) - images "
              "taken now may not match later ones")
    n = 0
    print(f"[V3.78 capture] session {d} | space = save | d = toggle depth | q = quit")
    while True:
        rgb, depth = cam.grab()
        k = _show(cv2, rgb, [f"session {sess}  saved {n}  depth {'on' if with_depth else 'off'}",
                             "space = save   d = depth on/off   q = quit"])
        if k == ord("q"):
            break
        if k == ord("d"):
            with_depth = not with_depth
        if k == ord(" "):
            stem = os.path.join(d, f"{sess}_{n:04d}")
            cv2.imwrite(stem + ".jpg", np.ascontiguousarray(rgb[..., ::-1]), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if with_depth:
                cv2.imwrite(stem + "_depth.png", np.clip(depth * 1000.0, 0, 65535).astype(np.uint16))
            n += 1


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="NERO V3.78 D435i bench (no robot needed)")
    ap.add_argument("mode", choices=["profile", "noise", "accuracy", "capture"])
    ap.add_argument("--model", default=None, help="YOLOE model file name (accuracy)")
    ap.add_argument("--prompt", default="egg", help="text prompt (accuracy)")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--standing", action="store_true")
    ap.add_argument("--session", default=None)
    a = ap.parse_args(argv)
    cam = D435i()
    try:
        if a.mode == "profile":
            mode_profile(cam)
        elif a.mode == "noise":
            mode_noise(cam, a.frames)
        elif a.mode == "accuracy":
            mode_accuracy(cam, a.model, a.prompt, a.trials, a.standing)
        else:
            mode_capture(cam, a.session)
    finally:
        cam.close()
        try:
            _cv2().destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
