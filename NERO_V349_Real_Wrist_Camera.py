#!/usr/bin/env python3
"""NERO V3.49 Real USB Wrist Camera + Wide-Angle Intrinsic Calibration + Eye-in-Hand Calibration + Zero-Prior Holder.

Standalone module for the real robot's wrist USB camera (innomaker 1080P USB2.0 130° wide-angle UVC RGB) and
zero-prior egg-holder slot localization.
Does NOT touch NERO_V326_Wrist_Camera.py (simulation wrist camera) or NERO_V308_Camera_Perception.py.
"""
from __future__ import annotations

import glob
import json
import math
import os
import threading
import time

import NERO_V325_Camera as CAM

# V3.49 REAL WRIST CAMERA + INTRINSICS + EYE-IN-HAND + ZERO-PRIOR HOLDER
# V3.50 DUAL MONITOR + WRIST QUALITY GATE + TCP CROSSHAIR + COARSE-TO-FINE
# V3.51 INDEPENDENT WRIST ARUCO + AUTO EYE-IN-HAND COLLECTION
# V3.52 FAST NON-BLOCKING EIH + 8-PARAM RATIONAL PNP FIX + POINT-DUALITY EIH SOLVER
# V3.56 FAST WRIST EYE-IN-HAND CALIB + LOOK-AT DOME PLANNER + FULL-RES ROI ARUCO + SCALE/LEVER-ARM SOLVER
# V3.57 AUTO TABLETOP CHARUCO INTRINSICS + CHARUCO EIH + 2D FOCAL/SCALE SELF-CALIB
# V3.58 KINEMATIC-ANCHORED WIDE-ANGLE INTRINSICS + FULL-QUADRANT TILT + ROBUST CHARUCO EIH
VERSION = "V4.3"
HERE = os.path.dirname(os.path.realpath(__file__))
WRIST_INFO_PATH = os.path.join(HERE, "config", "wrist_camera_info.json")
WRIST_INTR_PATH = os.path.join(HERE, "config", "wrist_intrinsics.json")
DEFAULT_WRIST_INTR_PATH = WRIST_INTR_PATH
WRIST_HANDEYE_PATH = os.path.join(HERE, "config", "wrist_handeye_calib.json")
DEFAULT_WRIST_HANDEYE_PATH = WRIST_HANDEYE_PATH
WRIST_INTR_SAMPLES_PATH = os.path.join(HERE, "config", "wrist_intrinsics_samples.json")
EIH_TARGET_MODES = ("ChArUco标定板/单码自适应 (推荐)", "ChArUco 标定板 (7×5 高精度)", "单张 ArUco 标记")

WRIST_RES_LIST = ("1920×1080", "1280×720")
BOARD_TYPES = ("ChArUco (推荐 130° 广角)", "棋盘格 (Chessboard)")
DIST_MODELS = ("广角 8 参数 (CALIB_RATIONAL_MODEL)", "针孔 5 参数 (Standard)", "鱼眼 4 参数 (Fisheye)")
DEFAULT_BOARD = {
    "type": BOARD_TYPES[0],
    "cols": 7,
    "rows": 5,
    "square_mm": 32.0,
    "marker_mm": 24.0,
    "dict": "DICT_4X4_50",
}
INTR_GOOD_RMS_PX = 0.8
INTR_MIN_SAMPLES = 6
HANDEYE_MIN_SAMPLES = 6
WRIST_MARKER_SIZES_MM = (24, 30, 40)
DEFAULT_WRIST_MARKER = {"dict": "DICT_4X4_50", "id": 0, "size_mm": 24.0}
EIH_VIEW_MAX_DEG = 42.0
EIH_Z_RANGE_M = (0.10, 0.58)
EIH_SPAN_TARGET_M = 0.14
EIH_ROT_MAX_DEG = 32.0
HOLDER_PROMPTS = "egg carton, egg holder, egg tray, tray, carton"
SLOT_PITCH_RANGE_MM = (36.0, 72.0)
WRIST_MOTION_MAX_DIFF = 4.5
WRIST_STILL_DEG = 0.15
WRIST_SHARP_MIN = 20.0
WRIST_OVEREXP_MAX = 0.25
WRIST_MAX_COARSE_DIFF_MM = 20.0
WRIST_ROI_FRAC = 0.78

_UNDISTORT_CACHE = {}


def _cv2():
    return CAM._import_cv2()


# =========================================================================== 1. USB Camera Discovery & UVC Source
def list_usb_cameras(sysfs_root="/sys/class/video4linux"):
    """Enumerate V4L2 video nodes, excluding Intel RealSense depth/RGB nodes, prioritizing innomaker / USB UVC."""
    devs = []
    if os.path.isdir(sysfs_root):
        for entry in sorted(os.listdir(sysfs_root)):
            if not entry.startswith("video"):
                continue
            try:
                idx = int(entry[5:])
            except ValueError:
                continue
            p = os.path.join(sysfs_root, entry)
            name = ""
            try:
                with open(os.path.join(p, "name"), "r", encoding="utf-8", errors="ignore") as f:
                    name = f.read().strip()
            except Exception:
                name = entry
            low = name.lower()
            if any(k in low for k in ("realsense", "d435", "d415", "d455", "d405")):
                continue
            # V4L2 creates 2 nodes per USB camera (index 0 = capture, index 1 = metadata)
            v_idx = 0
            idx_file = os.path.join(p, "index")
            if os.path.isfile(idx_file):
                try:
                    v_idx = int(open(idx_file).read().strip())
                except Exception:
                    v_idx = 0
            if v_idx > 0:
                continue
            devs.append({"device": f"/dev/{entry}", "index": idx, "name": name or entry})
    devs.sort(key=lambda d: (0 if any(k in d["name"].lower() for k in ("innomaker", "usb", "uvc", "1080")) else 1,
                             d["index"]))
    return devs


def apply_uvc_locks(cap, cv2, lock_exposure=True, lock_wb=True, lock_focus=True):
    """Configure MJPG / exposure / white-balance / focus locks on an OpenCV VideoCapture."""
    out = {}
    if lock_focus and hasattr(cv2, "CAP_PROP_AUTOFOCUS"):
        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            out["autofocus"] = "off"
        except Exception:
            out["autofocus"] = "unsupported"
    if lock_exposure and hasattr(cv2, "CAP_PROP_AUTO_EXPOSURE"):
        try:
            # V4L2: 1 = Manual Mode, 3 = Aperture Priority Mode
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            out["auto_exposure"] = "locked(manual)"
        except Exception:
            out["auto_exposure"] = "auto"
    if lock_wb and hasattr(cv2, "CAP_PROP_AUTO_WB"):
        try:
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            out["auto_wb"] = "locked"
        except Exception:
            out["auto_wb"] = "auto"
    return out


class UsbWristSource:
    """In-process UVC reader for the wrist RGB camera (innomaker 1080P 130° wide-angle)."""

    def __init__(self, device=None, res="1920×1080", lock_controls=True, intrinsics=None, log=print):
        self.device = device                    # e.g. '/dev/video4' or int index, None = auto-discover
        self.res = res if res in WRIST_RES_LIST else WRIST_RES_LIST[0]
        self.lock_controls = bool(lock_controls)
        self.log = log
        self.lock = threading.Lock()
        self.thread = None
        self.stop_ev = threading.Event()
        self.error = None
        self.info = {}
        self.K = None
        self.dist = [0.0] * 5
        self.model_type = DIST_MODELS[0]
        self.undistort_preview = False
        self.rgb = None
        self.frame_id = 0
        self.fps = 0.0
        if intrinsics and intrinsics_valid(intrinsics):
            self.set_intrinsics(intrinsics)

    def set_intrinsics(self, intr):
        with self.lock:
            if intr and intrinsics_valid(intr):
                self.K = dict(intr["K"])
                self.dist = [float(v) for v in intr["dist"]]
                self.model_type = intr.get("model_type") or DIST_MODELS[0]
                if self.info:
                    self.info["K"] = dict(self.K)
                    self.info["dist"] = list(self.dist)
                    self.info["intr_rms_px"] = intr.get("rms_px")
                    self.info["model_type"] = self.model_type
            else:
                self.K = None
                self.dist = [0.0] * 5

    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        if self.running():
            return
        self.error = None
        self.stop_ev.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name="v349_usb_wrist")
        self.thread.start()

    def stop(self, timeout=2.5):
        self.stop_ev.set()
        if self.thread is not None:
            self.thread.join(timeout)

    def _run(self):
        try:
            cv2 = _cv2()
            import numpy as np
        except Exception as exc:
            self.error = f"OpenCV / numpy 不可用：{exc}"
            return
        cap = None
        try:
            dev = self.device
            dev_name = str(dev or "")
            if not dev or dev == "auto":
                cands = list_usb_cameras()
                if not cands:
                    self.error = "没有找到 USB 腕部相机（请检查 USB 线，或运行 v4l2-ctl --list-devices）"
                    return
                dev = cands[0]["device"]
                dev_name = f"{cands[0]['name']} ({dev})"
            idx = int(dev[10:]) if isinstance(dev, str) and dev.startswith("/dev/video") and dev[10:].isdigit() else dev
            backend = getattr(cv2, "CAP_V4L2", 0)
            cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
            if not cap or not cap.isOpened():
                self.error = f"打不开腕部相机 {dev}（被其他程序占用或权限不足？）"
                return
            w, h = (1920, 1080) if self.res == WRIST_RES_LIST[0] else (1280, 720)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, 30)
            locks = apply_uvc_locks(cap, cv2, lock_exposure=self.lock_controls, lock_wb=self.lock_controls,
                                    lock_focus=self.lock_controls)
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or w)
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or h)
            info = {"name": dev_name or str(dev), "device": str(dev), "format": "MJPG",
                    "res": f"{aw}×{ah}", "width": aw, "height": ah, "controls": locks,
                    "K": dict(self.K) if self.K else None, "dist": list(self.dist) if self.K else None,
                    "model_type": self.model_type if self.K else None}
            with self.lock:
                self.info = info
            n, t0 = 0, time.time()
            while not self.stop_ev.is_set():
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    time.sleep(0.01)
                    continue
                rgb = np.ascontiguousarray(bgr[..., ::-1])
                n += 1
                if time.time() - t0 >= 1.0:
                    self.fps, n, t0 = n / (time.time() - t0), 0, time.time()
                with self.lock:
                    self.rgb = rgb
                    self.frame_id += 1
        except Exception as exc:
            self.error = f"腕部相机读取失败：{exc}"[:260]
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass

    def frames(self):
        with self.lock:
            rgb = self.rgb
            if rgb is not None and self.undistort_preview and self.K:
                rgb = undistort_rgb(rgb, {"K": self.K, "dist": self.dist, "model_type": self.model_type})
            return rgb, None, self.frame_id

    def raw_frame(self):
        with self.lock:
            return self.rgb, self.frame_id

    def display_ppm(self, view="color", size=(640, 360)):
        cv2 = _cv2()
        rgb, _d, _fid = self.frames()
        if rgb is None:
            return None
        tw, th = int(size[0]), int(size[1])
        if rgb.shape[1] != tw or rgb.shape[0] != th:
            img = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
        else:
            img = rgb
        return CAM.ppm_bytes(img)

    def detect_marker(self, marker, robot_q_deg=None):
        with self.lock:
            rgb, K, dist, mtype = self.rgb, (dict(self.K) if self.K else None), list(self.dist), self.model_type
        if rgb is None or not K:
            return None
        h_img, w_img = rgb.shape[:2]
        kw, kh = float(K.get("width") or w_img), float(K.get("height") or h_img)
        if abs(w_img - kw) > 1.0 or abs(h_img - kh) > 1.0:
            sx, sy = w_img / max(1.0, kw), h_img / max(1.0, kh)
            K = {
                "width": int(w_img),
                "height": int(h_img),
                "fx": float(K["fx"]) * sx,
                "fy": float(K["fy"]) * sy,
                "ppx": float(K["ppx"]) * sx,
                "ppy": float(K["ppy"]) * sy,
            }
        tmode = str((marker or {}).get("target_type") or getattr(self, "target_type", "auto") or "auto")
        bcfg = (marker or {}).get("board_cfg") or getattr(self, "board_cfg", None)
        if tmode != "单张 ArUco 标记" and tmode != "aruco":
            det_ch = detect_charuco_pose(rgb, K, dist, board_cfg=bcfg, model_type=mtype)
            if det_ch is not None:
                return det_ch
            if "ChArUco 标定板 (" in tmode or tmode == "charuco":
                return None
        if "鱼眼" in str(mtype) or "Fisheye" in str(mtype):
            rgb_u = undistort_rgb(rgb, {"K": K, "dist": dist, "model_type": mtype})
            return CAM.detect_aruco(rgb_u, K, [0.0] * 5, marker["dict"], marker["id"], marker["size_mm"] / 1000.0)
        # Pass full distortion vector (including 8-parameter CALIB_RATIONAL_MODEL k1..k6) to detect_aruco
        return CAM.detect_aruco(rgb, K, dist, marker["dict"], marker["id"], marker["size_mm"] / 1000.0)


# =========================================================================== 2. Wide-Angle Intrinsic Calibration
def generate_calib_board_png(board_cfg=None, out_path=None, dpi=300):
    """Generate an A4 300-DPI printable ChArUco or Chessboard calibration board PNG with a 100 mm verification ruler."""
    cv2 = _cv2()
    import numpy as np
    cfg = dict(DEFAULT_BOARD, **(board_cfg or {}))
    btype = str(cfg.get("type") or BOARD_TYPES[0])
    cols, rows = int(cfg.get("cols", 7)), int(cfg.get("rows", 5))
    sq_mm = float(cfg.get("square_mm", 32.0))
    mk_mm = float(cfg.get("marker_mm", round(sq_mm * 0.75, 1)))
    dict_name = str(cfg.get("dict") or "DICT_4X4_50")
    mm_px = float(dpi) / 25.4
    W, H = int(round(297.0 * mm_px)), int(round(210.0 * mm_px))
    img = np.full((H, W), 255, dtype=np.uint8)
    sq_px = max(20, int(round(sq_mm * mm_px)))

    if "ChArUco" in btype or "charuco" in btype.lower():
        aruco_dict = CAM._aruco_dict(cv2, dict_name)
        bw, bh = min(W - 200, cols * sq_px), min(H - 360, rows * sq_px)
        if hasattr(cv2.aruco, "CharucoBoard"):
            board = cv2.aruco.CharucoBoard((cols, rows), sq_mm / 1000.0, mk_mm / 1000.0, aruco_dict)
            bimg = board.generateImage((bw, bh))
        else:
            board = cv2.aruco.CharucoBoard_create(cols, rows, sq_mm / 1000.0, mk_mm / 1000.0, aruco_dict)
            bimg = board.draw((bw, bh))
        x0, y0 = (W - bw) // 2, (H - bh) // 2 - 20
        img[y0:y0 + bh, x0:x0 + bw] = bimg
        cv2.rectangle(img, (x0 - 4, y0 - 4), (x0 + bw + 4, y0 + bh + 4), 120, 2)
        titles = [
            f"NERO V3.49 Wrist Camera — ChArUco {cols}x{rows} Board ({dict_name})",
            f"Print on A4 at 100% (No Scaling). Square = {sq_mm:g} mm, Marker = {mk_mm:g} mm.",
        ]
        default_name = f"calib_charuco_{cols}x{rows}_{sq_mm:g}mm_A4.png"
    else:
        # cols x rows inner corners = (cols + 1) x (rows + 1) squares
        csq, rsq = cols + 1, rows + 1
        sq_px = min(sq_px, (W - 240) // csq, (H - 380) // rsq)
        bw, bh = csq * sq_px, rsq * sq_px
        x0, y0 = (W - bw) // 2, (H - bh) // 2 - 20
        for r in range(rsq):
            for c in range(csq):
                if (r + c) % 2 == 0:
                    img[y0 + r * sq_px:y0 + (r + 1) * sq_px, x0 + c * sq_px:x0 + (c + 1) * sq_px] = 0
        cv2.rectangle(img, (x0 - 4, y0 - 4), (x0 + bw + 4, y0 + bh + 4), 120, 2)
        titles = [
            f"NERO V3.49 Wrist Camera — Chessboard {cols}x{rows} Inner Corners ({csq}x{rsq} Squares)",
            f"Print on A4 at 100% (No Scaling). Square = {sq_mm:g} mm (measure with ruler after printing).",
        ]
        default_name = f"calib_chessboard_{cols}x{rows}_{sq_mm:g}mm_A4.png"

    # 100 mm verification ruler at bottom
    y_r = H - 120
    x_s = int(round((W - 100.0 * mm_px) / 2))
    x_e = x_s + int(round(100.0 * mm_px))
    cv2.line(img, (x_s, y_r), (x_e, y_r), 0, 3)
    for mm in range(0, 101, 10):
        x = x_s + int(round(mm * mm_px))
        th = 20 if mm % 50 == 0 else 10
        cv2.line(img, (x, y_r - th), (x, y_r + th), 0, 2)
        if mm % 50 == 0:
            cv2.putText(img, f"{mm}mm", (x - 26, y_r + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)
    for i, ln in enumerate(titles):
        cv2.putText(img, ln, (140, 75 + i * 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, 0, 2, cv2.LINE_AA)

    out_path = out_path or os.path.join(HERE, "config", default_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, img)
    return out_path


def detect_board_corners(rgb, board_cfg=None):
    """Detect ChArUco or Chessboard corners on an RGB frame.
    V3.58: Enables subpixel ArUco marker refinement + tryRefineMarkers + post-detection cornerSubPix + per-frame
    Homography/IRLS barrel-surface outlier corner rejection (eliminates 3..10 px corner pulls caused by gripper
    shadow or specular glare at the edge of the board).
    -> {'ok': bool, 'obj_pts': Nx3 (m), 'img_pts': Nx2 (px), 'n_corners': int, 'sharpness': float,
        'coverage': str, 'corners_px': [[u,v]...]} or {'ok': False, 'error': str}"""
    cv2 = _cv2()
    import numpy as np
    cfg = dict(DEFAULT_BOARD, **(board_cfg or {}))
    btype = str(cfg.get("type") or BOARD_TYPES[0])
    cols, rows = int(cfg.get("cols", 7)), int(cfg.get("rows", 5))
    sq_m = float(cfg.get("square_mm", 32.0)) / 1000.0
    mk_m = float(cfg.get("marker_mm", round(sq_m * 750.0, 1))) / 1000.0
    dict_name = str(cfg.get("dict") or "DICT_4X4_50")
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    h, w = gray.shape[:2]
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if "ChArUco" in btype or "charuco" in btype.lower():
        aruco_dict = CAM._aruco_dict(cv2, dict_name)
        if hasattr(cv2.aruco, "CharucoBoard"):
            board = cv2.aruco.CharucoBoard((cols, rows), sq_m, mk_m, aruco_dict)
            all_obj = np.asarray(board.getChessboardCorners(), dtype=np.float32).reshape(-1, 3)
        else:
            board = cv2.aruco.CharucoBoard_create(cols, rows, sq_m, mk_m, aruco_dict)
            all_obj = np.asarray(board.chessboardCorners, dtype=np.float32).reshape(-1, 3)
        if hasattr(cv2.aruco, "CharucoDetector"):
            dp = cv2.aruco.DetectorParameters()
            if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
                dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                dp.cornerRefinementWinSize = 5
                dp.cornerRefinementMaxIterations = 30
            cp = cv2.aruco.CharucoParameters()
            cp.tryRefineMarkers = True
            cdet = cv2.aruco.CharucoDetector(board, cp, dp)
            c_corners, c_ids, _m_corners, _m_ids = cdet.detectBoard(gray)
            if (c_corners is None or c_ids is None or len(c_ids) < 6):
                # Fallback with CLAHE contrast enhancement for glare/dim lighting
                g_eq = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
                c_corners, c_ids, _m_corners, _m_ids = cdet.detectBoard(g_eq)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
            if ids is None or len(ids) == 0:
                return {"ok": False, "error": "未检测到 ChArUco 标记"}
            _n_int, c_corners, c_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        if c_corners is None or c_ids is None or len(c_ids) < 6:
            n_seen = 0 if c_ids is None else len(c_ids)
            return {"ok": False, "error": f"ChArUco 角点太少（{n_seen}/≥6）：把标定板放进画面并正对相机"}
        ids_flat = [int(v) for v in np.asarray(c_ids).flatten()]
        img_pts = np.asarray(c_corners, dtype=np.float32).reshape(-1, 2)
        obj_pts = all_obj[ids_flat]
        # Subpixel saddle-point polish on interior corners
        in_b = (img_pts[:, 0] >= 8.0) & (img_pts[:, 0] <= w - 8.0) & (img_pts[:, 1] >= 8.0) & (img_pts[:, 1] <= h - 8.0)
        if np.any(in_b):
            try:
                sub_pts = img_pts[in_b].reshape(-1, 1, 2).copy()
                cv2.cornerSubPix(gray, sub_pts, (5, 5), (-1, -1), crit)
                sub_pts = sub_pts.reshape(-1, 2)
                # Accept subpixel refinement if shift is <= 2.2 px (avoids jumping onto nearby marker border)
                shift = np.linalg.norm(sub_pts - img_pts[in_b], axis=1)
                ok_sh = shift <= 2.2
                idx_in = np.where(in_b)[0]
                img_pts[idx_in[ok_sh]] = sub_pts[ok_sh]
            except Exception:
                pass
        # Per-frame Homography + IRLS barrel-surface outlier corner screen (strips gripper-shadow / glare corners)
        if len(img_pts) >= 9 and float(np.std(obj_pts[:, 0])) > 1e-3 and float(np.std(obj_pts[:, 1])) > 1e-3:
            try:
                H_f, _ = cv2.findHomography(obj_pts[:, :2].astype(np.float64), img_pts.astype(np.float64), cv2.RANSAC, 12.0)
                if H_f is not None:
                    pr_H = cv2.perspectiveTransform(obj_pts[:, :2].astype(np.float64).reshape(-1, 1, 2), H_f).reshape(-1, 2)
                    diff = img_pts.astype(np.float64) - pr_H
                    X_n = (obj_pts[:, 0].astype(np.float64) - float(obj_pts[:, 0].mean())) / max(0.02, float(obj_pts[:, 0].std()))
                    Y_n = (obj_pts[:, 1].astype(np.float64) - float(obj_pts[:, 1].mean())) / max(0.02, float(obj_pts[:, 1].std()))
                    Phi = np.column_stack([np.ones_like(X_n), X_n, Y_n, X_n * X_n, X_n * Y_n, Y_n * Y_n])
                    w_ir = np.ones(len(X_n), dtype=np.float64)
                    res_ir = np.zeros(len(X_n), dtype=np.float64)
                    for _ in range(5):
                        W_m = np.diag(w_ir)
                        coef, _, _, _ = np.linalg.lstsq(W_m @ Phi, W_m @ diff, rcond=None)
                        res_ir = np.linalg.norm(diff - (Phi @ coef), axis=1)
                        med_ir = max(0.25, float(np.median(res_ir)))
                        w_ir = np.where(res_ir <= 2.0 * med_ir, 1.0, (0.5 * med_ir / np.maximum(res_ir, 1e-6)) ** 2)
                    med_ir = max(0.25, float(np.median(res_ir)))
                    keep_c = res_ir <= max(2.5, 3.2 * med_ir)
                    if int(np.sum(keep_c)) >= 6 and int(np.sum(keep_c)) < len(img_pts):
                        img_pts = img_pts[keep_c]
                        obj_pts = obj_pts[keep_c]
            except Exception:
                pass
    else:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
        if not ok or corners is None:
            return {"ok": False, "error": f"未检测到 {cols}×{rows} 棋盘格内角点（检查格数设置与光线）"}
        corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), crit)
        img_pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        obj_pts = np.zeros((cols * rows, 3), dtype=np.float32)
        grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        obj_pts[:, 0] = grid[:, 0] * sq_m
        obj_pts[:, 1] = grid[:, 1] * sq_m

    us, vs = img_pts[:, 0] / max(1.0, float(w)), img_pts[:, 1] / max(1.0, float(h))
    cx, cy = float(us.mean()), float(vs.mean())
    span = max(float(us.max() - us.min()), float(vs.max() - vs.min()))
    tags = []
    if cx < 0.38:
        tags.append("左")
    elif cx > 0.62:
        tags.append("右")
    if cy < 0.38:
        tags.append("上")
    elif cy > 0.62:
        tags.append("下")
    if not tags:
        tags.append("中")
    cov = "".join(tags) + f" (跨度 {span * 100:.0f}%)"
    return {
        "ok": True,
        "obj_pts": obj_pts.tolist(),
        "img_pts": img_pts.tolist(),
        "n_corners": int(len(img_pts)),
        "sharpness": round(sharp, 1),
        "coverage": cov,
        "span_frac": round(span, 3),
        "center_frac": [round(cx, 3), round(cy, 3)],
        "corners_px": [[round(float(p[0]), 1), round(float(p[1]), 1)] for p in img_pts],
        "image_size": [int(w), int(h)],
        "board_cols": int(cols),
        "board_rows": int(rows),
        "square_mm": float(cfg.get("square_mm", 32.0)),
    }


def _check_rational_sanity(Km, dist_rat, Km_pin, dist_pin, w, h):
    """V3.57/V3.58 RATIONAL DISTORTION SANITY GATE:
    Checks whether an 8-parameter CALIB_RATIONAL_MODEL overfitted on partial board coverage:
      1. Focal length fx, fy must not drift > 15% from the 5-parameter pinhole model and must stay in [0.28*w, 0.85*w];
      2. Principal point (cx, cy) must stay within 14% of (w/2, h/2);
      3. Undistorting 8 test points across the outer ring (0.18..0.82 of w, h) must remain finite, monotonic, and within
         1.30x of the 5-parameter pinhole undistortion (prevents peripheral divergence and slow iterative undistortPoints).
    -> (ok: bool, reason: str)"""
    cv2 = _cv2()
    import numpy as np
    fx_r, fy_r, cx_r, cy_r = float(Km[0, 0]), float(Km[1, 1]), float(Km[0, 2]), float(Km[1, 2])
    fx_p, fy_p = float(Km_pin[0, 0]), float(Km_pin[1, 1])
    if not (0.28 * w <= fx_r <= 0.85 * w and 0.28 * w <= fy_r <= 0.85 * w):
        return False, "有理模型焦距超出广角物理范围"
    if abs(fx_r - fx_p) / max(1.0, fx_p) > 0.15 or abs(fy_r - fy_p) / max(1.0, fy_p) > 0.15:
        return False, f"8参数模型焦距漂移过大 ({fx_r:.0f} vs 5参数 {fx_p:.0f})"
    if abs(cx_r - 0.5 * w) > 0.14 * w or abs(cy_r - 0.5 * h) > 0.14 * h:
        return False, "8参数主点偏移过大"
    d_r = [float(v) for v in dist_rat]
    if any(abs(v) > 8.0 for v in d_r[:8]):
        return False, "8参数高阶畸变系数过大"
    test_uv = np.array([
        [0.20 * w, 0.20 * h], [0.80 * w, 0.20 * h], [0.80 * w, 0.80 * h], [0.20 * w, 0.80 * h],
        [0.50 * w, 0.16 * h], [0.50 * w, 0.84 * h], [0.16 * w, 0.50 * h], [0.84 * w, 0.50 * h],
    ], dtype=np.float64).reshape(-1, 1, 2)
    try:
        u_rat = cv2.undistortPoints(test_uv, Km, np.asarray(dist_rat, dtype=np.float64)).reshape(-1, 2)
        u_pin = cv2.undistortPoints(test_uv, Km_pin, np.asarray(dist_pin, dtype=np.float64)).reshape(-1, 2)
        if not np.all(np.isfinite(u_rat)):
            return False, "8参数边缘畸变发散"
        r_rat = np.linalg.norm(u_rat, axis=1)
        r_pin = np.maximum(1e-3, np.linalg.norm(u_pin, axis=1))
        ratio = r_rat / r_pin
        if float(np.max(ratio)) > 1.32 or float(np.min(ratio)) < 0.68:
            return False, f"8参数边缘拉伸异常 (×{float(np.max(ratio)):.2f})"
    except Exception:
        return False, "8参数校验异常"
    return True, "ok"


def calibrate_intrinsics(samples, image_size=(1920, 1080), model_type=DIST_MODELS[0], reject_px=1.8,
                         min_samples=INTR_MIN_SAMPLES):
    """Calibrate wide-angle camera intrinsics K and distortion coefficients dist from board samples.
    V3.58:
      1. Clean-subset barrel undistortion + planar RANSAC Homography per-view corner pruning (eliminates corrupted
         corners near the black gripper shadow or specular glare without losing valid views);
      2. Robot-Kinematic Metric Focal-Length Anchoring + Bounded Wide-Angle Profile Solver: prevents Zhang's planar
         homography singularity on low-tilt tabletop views from blowing fx up to 2147..5164 px (locks fx to the exact
         metric scale dictated by the robot arm's FK T_base_flange and physical wide-angle bounds [0.31*w, 0.78*w]);
      3. Iterative outlier view rejection when RMS > 0.8 px so blurred/glare views are pruned automatically."""
    cv2 = _cv2()
    import numpy as np
    idx = [i for i, s in enumerate(samples) if len(s.get("obj_pts") or []) >= 6]
    if len(idx) < min_samples:
        return {"ok": False, "error": f"至少需要 {min_samples} 张有效标定图（现在 {len(idx)} 张）"}
    w, h = int(image_size[0]), int(image_size[1])
    mtype = str(model_type or DIST_MODELS[0])
    is_fish = "鱼眼" in mtype or "Fisheye" in mtype
    is_rat = "8" in mtype or "RATIONAL" in mtype

    o_work = {i: np.asarray(samples[i]["obj_pts"], dtype=np.float32).reshape(-1, 3).copy() for i in idx}
    i_work = {i: np.asarray(samples[i]["img_pts"], dtype=np.float32).reshape(-1, 2).copy() for i in idx}

    s0 = samples[idx[0]]
    cols_b = int(s0.get("board_cols") or DEFAULT_BOARD["cols"])
    rows_b = int(s0.get("board_rows") or DEFAULT_BOARD["rows"])
    sq_m = float(s0.get("square_mm") or DEFAULT_BOARD["square_mm"]) / 1000.0
    board_center_3d = np.array([[0.5 * cols_b * sq_m], [0.5 * rows_b * sq_m], [0.0]], dtype=np.float64)

    pruned_corners = 0

    def _clean_outlier_corners(sub_idx):
        nonlocal pruned_corners
        if is_fish or len(sub_idx) < 6:
            return
        try:
            K_nom = np.array([[0.42 * w, 0.0, 0.5 * w], [0.0, 0.42 * w, 0.5 * h], [0.0, 0.0, 1.0]], dtype=np.float64)
            fl0 = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K3
            _, Km0, Dm0, rvs0, tvs0 = cv2.calibrateCamera([o_work[i] for i in sub_idx], [i_work[i] for i in sub_idx],
                                                          (w, h), K_nom.copy(), None, flags=fl0)
            v_rms0 = []
            for i_k, rv, tv in zip(sub_idx, rvs0, tvs0):
                pr, _ = cv2.projectPoints(o_work[i_k], rv, tv, Km0, Dm0)
                v_rms0.append((i_k, float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - i_work[i_k]) ** 2, axis=1))))))
            if max(v for _, v in v_rms0) > 0.72:
                v_rms0.sort(key=lambda x: x[1])
                c_idx = [x[0] for x in v_rms0[:max(5, int(len(sub_idx) * 0.55))]]
                _, Km_c, Dm_c, _, _ = cv2.calibrateCamera([o_work[i] for i in c_idx], [i_work[i] for i in c_idx],
                                                          (w, h), K_nom.copy(), None, flags=fl0)
                for i_k in sub_idx:
                    o_s = o_work[i_k].astype(np.float64)
                    im_s = i_work[i_k].astype(np.float64)
                    if len(o_s) < 8 or float(np.std(o_s[:, 0])) < 1e-4 or float(np.std(o_s[:, 1])) < 1e-4:
                        continue
                    pts_u = cv2.undistortPoints(im_s.reshape(-1, 1, 2), Km_c, Dm_c, P=Km_c).reshape(-1, 2)
                    H, _ = cv2.findHomography(o_s[:, :2], pts_u, cv2.RANSAC, 1.5)
                    if H is not None:
                        pr_u = cv2.perspectiveTransform(o_s[:, :2].reshape(-1, 1, 2), H).reshape(-1, 2)
                        err_u = np.linalg.norm(pr_u - pts_u, axis=1)
                        keep = err_u <= max(1.15, 2.2 * float(np.median(err_u)))
                        n_drop = len(o_s) - int(np.sum(keep))
                        if int(np.sum(keep)) >= 6 and 1 <= n_drop <= 3:
                            pruned_corners += n_drop
                            o_work[i_k] = o_work[i_k][keep]
                            i_work[i_k] = i_work[i_k][keep]
                            samples[i_k]["obj_pts"] = o_work[i_k].tolist()
                            samples[i_k]["img_pts"] = i_work[i_k].tolist()
                            samples[i_k]["n_corners"] = int(len(o_work[i_k]))
        except Exception:
            pass

    def _fk_eih_mm(sub_idx, rvecs, tvecs):
        fk_pairs = [(samples[i].get("T_base_flange"), rv, tv)
                    for i, rv, tv in zip(sub_idx, rvecs, tvecs)
                    if samples[i].get("T_base_flange") and len(o_work[i]) >= 8
                    and float(np.std(o_work[i][:, 0])) > 1e-4 and float(np.std(o_work[i][:, 1])) > 1e-4]
        if len(fk_pairs) < 5:
            return None
        A_sub = [p[0] for p in fk_pairs]
        if CAM.translation_span_m(A_sub) < 0.035:
            return None
        B_sub = []
        for _, rv, tv in fk_pairs:
            R_cm = CAM.rodrigues([float(v) for v in np.asarray(rv).flatten()])
            p_c = (np.array(R_cm, dtype=np.float64) @ board_center_3d + np.asarray(tv, dtype=np.float64).reshape(3, 1)).flatten().tolist()
            B_sub.append(CAM.T_from(R_cm, p_c))
        try:
            r_eih = solve_eye_in_hand(A_sub, B_sub, size_m=0.024, iters=120, max_offset_m=0.14)
            return float(r_eih["mean_mm"])
        except Exception:
            return None

    def _fit_pinhole(sub_idx, flags=0, km_lock=None):
        obj_list = [o_work[i] for i in sub_idx]
        img_list = [i_work[i] for i in sub_idx]
        if km_lock is not None:
            fl_r = flags | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT
            rms, Km, Dm, rvecs, tvecs = cv2.calibrateCamera(obj_list, img_list, (w, h), km_lock.copy(), None, flags=fl_r)
            errs = []
            for o, im, rv, tv in zip(obj_list, img_list, rvecs, tvecs):
                proj, _ = cv2.projectPoints(o, rv, tv, Km, Dm)
                errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - im.reshape(-1, 2)) ** 2, axis=1)))))
            return float(rms), Km, Dm.flatten().tolist(), errs, True

        K_init = np.array([[0.50 * w, 0.0, 0.5 * w], [0.0, 0.50 * w, 0.5 * h], [0.0, 0.0, 1.0]], dtype=np.float64)
        fl_free = flags | cv2.CALIB_USE_INTRINSIC_GUESS
        rms, Km, Dm, rvecs, tvecs = cv2.calibrateCamera(obj_list, img_list, (w, h), K_init.copy(), None, flags=fl_free)
        fx, fy, cx, cy = float(Km[0, 0]), float(Km[1, 1]), float(Km[0, 2]), float(Km[1, 2])
        d_flat = Dm.flatten().tolist()
        fk_err = _fk_eih_mm(sub_idx, rvecs, tvecs)
        sane_free = (
            0.30 * w <= fx <= 0.82 * w and 0.30 * w <= fy <= 0.82 * w
            and 0.90 <= fx / max(1.0, fy) <= 1.11
            and abs(cx - 0.5 * w) <= 0.14 * w and abs(cy - 0.5 * h) <= 0.14 * h
            and abs(d_flat[0]) <= 1.2 and abs(d_flat[1]) <= 1.5
            and (fk_err is None or fk_err <= 1.0)
            and float(rms) <= 0.78
        )
        used_kin_anchor = False
        if not sane_free:
            used_kin_anchor = True
            fl_fix = (flags & ~cv2.CALIB_RATIONAL_MODEL) | cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_K3

            def _eval_f(f_cand):
                K_c = np.array([[f_cand, 0.0, 0.5 * w], [0.0, f_cand, 0.5 * h], [0.0, 0.0, 1.0]], dtype=np.float64)
                r_c, Km_c, Dm_c, rvs_c, tvs_c = cv2.calibrateCamera(obj_list, img_list, (w, h), K_c, None, flags=fl_fix)
                e_fk = _fk_eih_mm(sub_idx, rvs_c, tvs_c)
                sc = float(r_c) + (0.08 * e_fk if e_fk is not None else 0.0)
                return sc, float(r_c), Km_c, Dm_c, rvs_c, tvs_c

            grid_f = [w * (0.33 + 0.042 * k) for k in range(11)]
            best_f = min(grid_f, key=lambda fc: _eval_f(fc)[0])
            lo_f, hi_f = max(0.31 * w, best_f - 0.045 * w), min(0.78 * w, best_f + 0.045 * w)
            for _ in range(4):
                m1, m2 = lo_f + 0.382 * (hi_f - lo_f), lo_f + 0.618 * (hi_f - lo_f)
                if _eval_f(m1)[0] < _eval_f(m2)[0]:
                    hi_f = m2
                else:
                    lo_f = m1
            f_opt = 0.5 * (lo_f + hi_f)
            _, rms, Km, Dm, rvecs, tvecs = _eval_f(f_opt)
            for extra_fl in (
                cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_K3,
                cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_PRINCIPAL_POINT,
            ):
                try:
                    r_p, Km_p, Dm_p, rvs_p, tvs_p = cv2.calibrateCamera(
                        obj_list, img_list, (w, h), Km.copy(), Dm.copy(), flags=(flags & ~cv2.CALIB_RATIONAL_MODEL) | extra_fl
                    )
                    dp_flat = Dm_p.flatten().tolist()
                    if (abs(float(Km_p[0, 2]) - 0.5 * w) <= 0.09 * w and abs(float(Km_p[1, 2]) - 0.5 * h) <= 0.09 * h
                            and abs(dp_flat[0]) <= 1.0 and abs(dp_flat[1]) <= 1.2 and abs(dp_flat[4] if len(dp_flat) >= 5 else 0.0) <= 0.6
                            and float(r_p) <= float(rms)):
                        rms, Km, Dm, rvecs, tvecs = float(r_p), Km_p, Dm_p, rvs_p, tvs_p
                except Exception:
                    pass
        errs = []
        for o, im, rv, tv in zip(obj_list, img_list, rvecs, tvecs):
            proj, _ = cv2.projectPoints(o, rv, tv, Km, Dm)
            errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - im.reshape(-1, 2)) ** 2, axis=1)))))
        return float(rms), Km, Dm.flatten().tolist(), errs, used_kin_anchor

    def _run_sub(sub_idx, fast_screen=False):
        obj_list = [o_work[i].reshape(-1, 1, 3) for i in sub_idx]
        img_list = [i_work[i].reshape(-1, 1, 2) for i in sub_idx]
        if is_fish:
            obj_f = [o.astype(np.float64) for o in obj_list]
            img_f = [im.astype(np.float64) for im in img_list]
            Km = np.eye(3, dtype=np.float64)
            Dm = np.zeros((4, 1), dtype=np.float64)
            flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
            rms, Km, Dm, rvecs, tvecs = cv2.fisheye.calibrate(obj_f, img_f, (w, h), Km, Dm, flags=flags)
            errs = []
            for o, im, rv, tv in zip(obj_f, img_f, rvecs, tvecs):
                proj, _ = cv2.fisheye.projectPoints(o, rv, tv, Km, Dm)
                errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - im.reshape(-1, 2)) ** 2, axis=1)))))
            return float(rms), Km, Dm.flatten().tolist()[:4], errs, None
        if fast_screen:
            K_nom = np.array([[0.45 * w, 0.0, 0.5 * w], [0.0, 0.45 * w, 0.5 * h], [0.0, 0.0, 1.0]], dtype=np.float64)
            rms_s, Km_s, d_s, errs_s, _ = _fit_pinhole(sub_idx, flags=cv2.CALIB_FIX_K3, km_lock=K_nom)
            return rms_s, Km_s, (d_s + [0.0] * 5)[:5], errs_s, None
        rms_pin, Km_pin, d_pin_full, errs_pin, kin_pin = _fit_pinhole(sub_idx, flags=0)
        d_pin5 = (d_pin_full + [0.0] * 5)[:5]
        if not is_rat:
            return rms_pin, Km_pin, d_pin5, errs_pin, None
        rms_rat, Km_rat, d_rat_full, errs_rat, _ = _fit_pinhole(
            sub_idx, flags=cv2.CALIB_RATIONAL_MODEL, km_lock=(Km_pin if kin_pin else None)
        )
        d_rat8 = (d_rat_full + [0.0] * 8)[:8]
        ok_san, why_san = _check_rational_sanity(Km_rat, d_rat8, Km_pin, d_pin5, w, h)
        if not ok_san:
            return rms_pin, Km_pin, d_pin5 + [0.0, 0.0, 0.0], errs_pin, f"已自动启用防过拟合正则化（{why_san}）"
        return rms_rat, Km_rat, d_rat8, errs_rat, None

    rejected = []
    reg_note = None
    try:
        # Step 1: Clean 1..3 isolated outlier corners (e.g. gripper shadow / glare); frames with >= 4 bad corners are left for whole-view rejection
        _clean_outlier_corners(idx)
        # Step 2: Fast whole-view rejection for motion-blurred or multi-corner corrupted views
        while len(idx) > min_samples:
            rms_s, _, _, errs_s, _ = _run_sub(idx, fast_screen=True)
            med_s = sorted(errs_s)[len(errs_s) // 2]
            k = max(range(len(errs_s)), key=lambda m: errs_s[m])
            if errs_s[k] > max(reject_px, 2.5 * med_s):
                rejected.append(idx.pop(k))
                continue
            break
        # Step 3: Full kinematic-coupled wide-angle solve + iterative worst-view pruning when RMS > 0.80 px
        while True:
            rms, Km, dist, errs, reg_note = _run_sub(idx, fast_screen=False)
            med = sorted(errs)[len(errs) // 2]
            k = max(range(len(errs)), key=lambda m: errs[m])
            if len(idx) > min_samples and (
                errs[k] > max(reject_px, 2.2 * med)
                or (rms > INTR_GOOD_RMS_PX and len(idx) > max(min_samples, 8) and errs[k] > max(0.82, 1.42 * med))
            ):
                rejected.append(idx.pop(k))
                continue
            break
    except Exception as exc:
        return {"ok": False, "error": f"内参标定计算失败：{exc}"}

    K = {
        "width": w,
        "height": h,
        "fx": round(float(Km[0, 0]), 3),
        "fy": round(float(Km[1, 1]), 3),
        "ppx": round(float(Km[0, 2]), 3),
        "ppy": round(float(Km[1, 2]), 3),
    }
    fov_x = math.degrees(2.0 * math.atan(w / (2.0 * max(1.0, K["fx"]))))
    fov_d = math.degrees(2.0 * math.atan(math.hypot(w, h) / (2.0 * max(1.0, 0.5 * (K["fx"] + K["fy"])))))
    centers = [samples[i].get("center_frac") or [0.5, 0.5] for i in idx]
    span_u = max(c[0] for c in centers) - min(c[0] for c in centers) if len(centers) > 1 else 0.0
    span_v = max(c[1] for c in centers) - min(c[1] for c in centers) if len(centers) > 1 else 0.0
    max_board_span = max([float(samples[i].get("span_frac") or 0.0) for i in idx] + [0.0])
    eff_span = max(span_u, span_v, 0.65 * max_board_span)
    warn = []
    if reg_note:
        warn.append(reg_note)
    if len(idx) < 8:
        warn.append(f"有效样本 {len(idx)} 张（建议 ≥ 8 张，覆盖画面中心与四角）")
    if eff_span < 0.20:
        warn.append("标定板位置太集中：建议覆盖画面中心与四周")
    if rms > INTR_GOOD_RMS_PX:
        warn.append(f"重投影 RMS {rms:.2f} px（> {INTR_GOOD_RMS_PX:.1f} px）")
    good = rms <= INTR_GOOD_RMS_PX and len(idx) >= 8 and eff_span >= 0.20
    return {
        "ok": True,
        "good": good,
        "K": K,
        "dist": [round(float(v), 6) for v in dist],
        "model_type": mtype,
        "rms_px": round(rms, 3),
        "max_px": round(max(errs), 3),
        "per_view_px": [round(e, 3) for e in errs],
        "used": idx,
        "rejected": rejected,
        "n_used": len(idx),
        "pruned_corners": int(pruned_corners),
        "fov_h_deg": round(fov_x, 1),
        "fov_diag_deg": round(fov_d, 1),
        "warnings": warn,
    }


def detect_charuco_pose(rgb, K, dist=None, board_cfg=None, model_type=None):
    """V3.57/V3.58 HIGH-PRECISION CHARUCO BOARD 6-DOF POSE DETECTOR:
    Detects up to 24 symmetric X-junction chessboard corners on the ChArUco (or Chessboard) calibration board,
    centers the 3D board frame at the board centre (so p_cam = [0, 0, z] when centered), undistorts the corners,
    prunes any residual outlier corners via planar RANSAC Homography, and solves the overdetermined 6-DoF pose
    via solvePnP + solvePnPRefineLM.
    -> None or marker-compatible dict {'id', 'target_type': 'charuco', 'n_corners', 'corners_px', 'center_px',
       'p_cam', 'p_pnp', 'source': 'charuco', 'rvec', 'normal_cam', 'side_px', 'reproj_px'}."""
    if rgb is None or not K:
        return None
    cv2 = _cv2()
    import numpy as np
    cfg = dict(DEFAULT_BOARD, **(board_cfg or {}))
    det_b = detect_board_corners(rgb, cfg)
    if not det_b.get("ok") or int(det_b.get("n_corners", 0)) < 4:
        return None
    h_img, w_img = rgb.shape[:2]
    kw, kh = float(K.get("width") or w_img), float(K.get("height") or h_img)
    sx, sy = w_img / max(1.0, kw), h_img / max(1.0, kh)
    Km = np.array([[float(K["fx"]) * sx, 0.0, float(K["ppx"]) * sx],
                   [0.0, float(K["fy"]) * sy, float(K["ppy"]) * sy],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    cols, rows = int(cfg.get("cols", 7)), int(cfg.get("rows", 5))
    sq_m = float(cfg.get("square_mm", 32.0)) / 1000.0
    btype = str(cfg.get("type") or BOARD_TYPES[0])
    if "ChArUco" in btype or "charuco" in btype.lower():
        cx_b, cy_b = 0.5 * cols * sq_m, 0.5 * rows * sq_m
    else:
        cx_b, cy_b = 0.5 * (cols - 1) * sq_m, 0.5 * (rows - 1) * sq_m
    obj_raw = np.asarray(det_b["obj_pts"], dtype=np.float64).reshape(-1, 3)
    obj_c = obj_raw.copy()
    obj_c[:, 0] -= cx_b
    obj_c[:, 1] -= cy_b
    if float(np.std(obj_c[:, 0])) < 1e-4 or float(np.std(obj_c[:, 1])) < 1e-4:
        return None
    img_pts = np.asarray(det_b["img_pts"], dtype=np.float64).reshape(-1, 2)
    d_list = [float(v) for v in (dist or [0.0] * 5)]
    mtype = str(model_type or DIST_MODELS[0])
    if "鱼眼" in mtype or "Fisheye" in mtype:
        Dm = np.asarray((d_list + [0.0] * 4)[:4], dtype=np.float64).reshape(4, 1)
        pts_u = cv2.fisheye.undistortPoints(img_pts.reshape(-1, 1, 2), Km, Dm, P=Km).reshape(-1, 2)
    else:
        if len(d_list) > 5 and any(abs(v) > 8.0 for v in d_list[:8]):
            d_list = d_list[:5]
        if len(d_list) in (4, 5, 8, 12, 14):
            dc = np.asarray(d_list, dtype=np.float64)
        elif len(d_list) > 5:
            dc = np.asarray((d_list + [0.0] * 8)[:8], dtype=np.float64)
        else:
            dc = np.asarray((d_list + [0.0] * 5)[:5], dtype=np.float64)
        if np.any(np.abs(dc) > 1e-9):
            pts_u = cv2.undistortPoints(img_pts.reshape(-1, 1, 2), Km, dc, P=Km).reshape(-1, 2)
        else:
            pts_u = img_pts

    if len(obj_c) >= 8:
        try:
            H_u, _ = cv2.findHomography(obj_c[:, :2], pts_u, cv2.RANSAC, 1.8)
            if H_u is not None:
                pr_u = cv2.perspectiveTransform(obj_c[:, :2].reshape(-1, 1, 2), H_u).reshape(-1, 2)
                err_u = np.linalg.norm(pr_u - pts_u, axis=1)
                keep = err_u <= max(1.5, 2.4 * float(np.median(err_u)))
                if int(np.sum(keep)) >= 6 and int(np.sum(keep)) < len(obj_c) and float(np.std(obj_c[keep, 0])) > 1e-4 and float(np.std(obj_c[keep, 1])) > 1e-4:
                    obj_c = obj_c[keep]
                    pts_u = pts_u[keep]
                    img_pts = img_pts[keep]
        except Exception:
            pass

    flag = getattr(cv2, "SOLVEPNP_IPPE", cv2.SOLVEPNP_ITERATIVE)
    ok, rvec, tvec = cv2.solvePnP(obj_c, pts_u, Km, np.zeros(5, dtype=np.float64), flags=flag)
    if not ok or float(np.asarray(tvec).flatten()[2]) <= 0.04:
        ok, rvec, tvec = cv2.solvePnP(obj_c, pts_u, Km, np.zeros(5, dtype=np.float64), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    if hasattr(cv2, "solvePnPRefineLM"):
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj_c, pts_u, Km, np.zeros(5, dtype=np.float64), rvec, tvec)
        except Exception:
            pass
    proj_u, _ = cv2.projectPoints(obj_c, rvec, tvec, Km, np.zeros(5, dtype=np.float64))
    reproj_px = float(np.sqrt(np.mean(np.sum((proj_u.reshape(-1, 2) - pts_u) ** 2, axis=1))))
    if reproj_px > 8.5:
        return None
    p = [float(v) for v in np.asarray(tvec).flatten()]
    rv = [float(v) for v in np.asarray(rvec).flatten()]
    R_cm = CAM.rodrigues(rv)
    normal_cam = [R_cm[0][2], R_cm[1][2], R_cm[2][2]]
    center_u = CAM.project(p, {"fx": Km[0, 0], "fy": Km[1, 1], "ppx": Km[0, 2], "ppy": Km[1, 2]}) or img_pts.mean(axis=0).tolist()
    u0, v0 = float(img_pts[:, 0].min()), float(img_pts[:, 1].min())
    u1, v1 = float(img_pts[:, 0].max()), float(img_pts[:, 1].max())
    box_corners = [[u0, v0], [u1, v0], [u1, v1], [u0, v1]]
    side_px = float(max(30.0, 0.5 * ((u1 - u0) + (v1 - v0))))
    return {
        "id": 0,
        "target_type": "charuco",
        "n_corners": int(len(img_pts)),
        "corners_px": box_corners,
        "center_px": [round(float(center_u[0]), 1), round(float(center_u[1]), 1)],
        "p_cam": list(p),
        "p_pnp": list(p),
        "p_depth": None,
        "source": "charuco",
        "rvec": rv,
        "normal_cam": normal_cam,
        "side_px": side_px,
        "reproj_px": round(reproj_px, 3),
    }


def plan_intr_auto_targets(fk_fn, q0_rad, limits_rad, s0_board=None, image_size=(1920, 1080), n=10,
                           board_cfg=None, fence_fn=lambda q: True, seed=0):
    """V3.57/V3.58 AUTOMATIC TABLETOP BOARD INTRINSIC COLLECTION PLANNER:
    Given the starting arm pose q0_rad looking down at a tabletop ChArUco/Chessboard board, generates n (default 10)
    safe, smooth joint poses that actively steer the board across the image centre + all 4 image quadrants
    (left-top, right-top, left-bottom, right-bottom), strong 18..28 deg perspective tilts, and height steps,
    accounting for the starting board's initial image offset (center_frac).
    -> (targets: [{'q': [rad...], 'T': 4x4, 'tag': str}, ...], stats_dict)"""
    import random
    rng = random.Random(seed)
    T0 = fk_fn(q0_rad)
    if T0 is None:
        return [], {"tries": 0, "pool": 0, "rejected": {"FK": 1}}
    w, h = int(image_size[0]), int(image_size[1])
    # Realistic 130-deg wide-angle focal prior (fx ≈ 0.38 * w = 730 px on 1920x1080)
    K_approx = {"width": w, "height": h, "fx": 0.38 * w, "fy": 0.38 * w, "ppx": 0.5 * w, "ppy": 0.5 * h}
    cfg = dict(DEFAULT_BOARD, **(board_cfg or {}))
    sq_m = float(cfg.get("square_mm", 32.0)) / 1000.0
    cols, rows = int(cfg.get("cols", 7)), int(cfg.get("rows", 5))

    p_cam0 = [0.0, 0.0, 0.25]
    uv0_frac = [0.5, 0.5]
    if isinstance(s0_board, dict):
        if s0_board.get("center_frac") and len(s0_board["center_frac"]) >= 2:
            uv0_frac = [float(s0_board["center_frac"][0]), float(s0_board["center_frac"][1])]
        if s0_board.get("obj_pts") and s0_board.get("img_pts"):
            try:
                cv2 = _cv2()
                import numpy as np
                Km = np.array([[K_approx["fx"], 0, K_approx["ppx"]], [0, K_approx["fy"], K_approx["ppy"]], [0, 0, 1]], dtype=np.float64)
                o_arr = np.asarray(s0_board["obj_pts"], dtype=np.float64).reshape(-1, 3).copy()
                o_arr[:, 0] -= 0.5 * cols * sq_m
                o_arr[:, 1] -= 0.5 * rows * sq_m
                im_arr = np.asarray(s0_board["img_pts"], dtype=np.float64).reshape(-1, 2)
                ok_p, rv0, tv0 = cv2.solvePnP(o_arr, im_arr, Km, np.zeros(5))
                if ok_p and 0.08 <= float(tv0.flatten()[2]) <= 0.65:
                    p_cam0 = [float(v) for v in tv0.flatten()]
            except Exception:
                pass
    s0_synth = {"T_base_flange": T0, "p_cam": p_cam0}
    prior = initial_eih_prior_from_start(s0_synth)
    X = prior["T_flange_cam"]
    Y = prior["T_base_marker"]
    p_m = list(CAM.T_t(Y))
    p0 = CAM.T_t(T0)
    R0 = CAM.T_R(T0)
    T_bc0 = CAM.mat_mul(T0, X)
    p_c0 = CAM.T_t(T_bc0)
    R_bc0 = CAM.T_R(T_bc0)
    v0 = CAM.vsub(p_m, p_c0)
    d0 = CAM.vnorm(v0) or 0.25
    u0 = [v / d0 for v in v0]
    margin_rad = math.radians(getattr(CAM, "SOFT_MARGIN_DEG", 3.0))
    z_floor = min(p_m[2] + 0.08, p0[2] - 0.055)

    def _rot_vec(Ra, Rb):
        R_ = [[sum(Rb[i][k] * Ra[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
        return [0.5 * (R_[2][1] - R_[1][2]), 0.5 * (R_[0][2] - R_[2][0]), 0.5 * (R_[1][0] - R_[0][1])]

    nj = len(q0_rad)
    h_j = 1e-5
    cols_j = []
    for j in range(nj):
        qp = list(q0_rad)
        qp[j] += h_j
        Tp = fk_fn(qp)
        if Tp is not None:
            cols_j.append([(Tp[i][3] - p0[i]) / h_j for i in range(3)] + [x / h_j for x in _rot_vec(R0, CAM.T_R(Tp))])
    J_pinv = None
    if len(cols_j) == nj:
        J0 = [[cols_j[j][r] for j in range(nj)] for r in range(6)]
        lam = 2.0e-3
        JJT = [[sum(J0[r][k] * J0[c][k] for k in range(nj)) + (lam if r == c else 0.0) for c in range(6)] for r in range(6)]
        aug = [JJT[r] + [1.0 if r == c else 0.0 for c in range(6)] for r in range(6)]
        ok_inv = True
        for c in range(6):
            pv = max(range(c, 6), key=lambda r: abs(aug[r][c]))
            aug[c], aug[pv] = aug[pv], aug[c]
            if abs(aug[c][c]) < 1e-12:
                ok_inv = False
                break
            for r in range(6):
                if r != c:
                    f = aug[r][c] / aug[c][c]
                    for k in range(c, 12):
                        aug[r][k] -= f * aug[c][k]
        if ok_inv:
            inv_JJT = [[aug[r][6 + c] / aug[r][r] for c in range(6)] for r in range(6)]
            J_pinv = [[sum(J0[r][j] * inv_JJT[r][c] for r in range(6)) for c in range(6)] for j in range(nj)]

    rej = {}
    pool = []
    tries = 0

    def try_add(q_cand, tag="auto"):
        nonlocal tries
        tries += 1
        if q_cand is None:
            return
        for j, (lo, hi) in enumerate(limits_rad):
            if not (lo + margin_rad - 1e-6 <= q_cand[j] <= hi - margin_rad + 1e-6):
                rej["软限位"] = rej.get("软限位", 0) + 1
                return
        if not fence_fn(q_cand):
            rej["安全区"] = rej.get("安全区", 0) + 1
            return
        T = fk_fn(q_cand)
        if T is None or T[2][3] < z_floor:
            rej["太低"] = rej.get("太低", 0) + 1
            return
        ok_v, why_v, vdeg = eih_pose_visible(T, X, Y, K_approx, size_m=0.04, max_tilt_deg=44.0,
                                             margin_px=40.0, max_off_axis=0.75)
        if not ok_v:
            rej[why_v] = rej.get(why_v, 0) + 1
            return
        if not CAM.check_auto_path(fence_fn, q0_rad, q_cand, steps=4):
            rej["路径安全区"] = rej.get("路径安全区", 0) + 1
            return
        T_bc = CAM.mat_mul(T, X)
        B_pred = CAM.mat_mul(CAM.T_inv(T_bc), Y)
        uv_c = CAM.project(CAM.T_t(B_pred), K_approx) or [0.5 * w, 0.5 * h]
        pool.append({
            "q": list(q_cand),
            "T": T,
            "pos": CAM.T_t(T),
            "uv_frac": [uv_c[0] / float(w), uv_c[1] / float(h)],
            "view_deg": vdeg,
            "tag": tag,
        })

    R_X_t = CAM.transpose(CAM.T_R(X))
    t_X = CAM.T_t(X)
    if J_pinv is not None:
        # 1) Structured 4-quadrant + center + strong perspective tilts (18..28 deg) + height steps
        recipes = [
            ("中心正视", (0.00, 0.00, 0.00), (0.00, 0.00), 0.0),
            ("左上区", (-0.055, 0.055, 0.01), (0.065, -0.045), 14.0),
            ("右上区", (0.055, -0.055, 0.01), (-0.065, -0.045), -14.0),
            ("左下区", (-0.055, -0.055, 0.00), (0.065, 0.045), -18.0),
            ("右下区", (0.055, 0.055, 0.00), (-0.065, 0.045), 18.0),
            ("前倾视", (-0.095, 0.00, 0.030), (0.015, 0.00), 22.0),
            ("后倾视", (0.095, 0.00, 0.030), (-0.015, 0.00), -22.0),
            ("左倾视", (0.00, 0.100, 0.025), (0.00, -0.015), 25.0),
            ("右倾视", (0.00, -0.100, 0.025), (0.00, 0.015), -25.0),
            ("对角倾视A", (-0.075, 0.075, 0.025), (0.030, -0.025), 20.0),
            ("对角倾视B", (0.075, -0.075, 0.025), (-0.030, 0.025), -20.0),
            ("近距大尺度", (0.01, -0.01, -0.035), (0.01, 0.01), 10.0),
            ("远距全貌", (-0.01, 0.01, 0.060), (-0.01, -0.01), -10.0),
        ]
        for tag, (dx, dy, dz), (ax_off, ay_off), roll_d in recipes:
            p_c = [p_c0[0] + dx, p_c0[1] + dy, p_c0[2] + dz]
            aim = [p_m[0] + ax_off, p_m[1] + ay_off, p_m[2]]
            v_los = CAM.vsub(aim, p_c)
            d_los = CAM.vnorm(v_los) or 0.25
            u_los = [v / d_los for v in v_los]
            cr = _vcross(u0, u_los)
            cn = CAM.vnorm(cr)
            dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(u0, u_los))))
            ang_deg = math.degrees(math.acos(dot))
            R_swing = CAM.axis_angle_R(cr, ang_deg) if cn > 1e-6 else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            R_roll = CAM.axis_angle_R(u_los, roll_d)
            R_bc = CAM.mat_mul(R_roll, CAM.mat_mul(R_swing, R_bc0))
            R_bf = CAM.mat_mul(R_bc, R_X_t)
            p_bf = CAM.vsub(p_c, CAM.mat_vec(R_bf, t_X))
            err6 = [p_bf[i] - p0[i] for i in range(3)] + _rot_vec(R0, R_bf)
            qc = [max(lo + margin_rad, min(hi - margin_rad, q0_rad[j] + sum(J_pinv[j][c] * err6[c] for c in range(6))))
                  for j, (lo, hi) in enumerate(limits_rad)]
            try_add(qc, tag=tag)
        # 2) Random Look-At Dome + 4-quadrant spread candidates with stronger tilt diversity
        for _ in range(32):
            dx = rng.uniform(-0.105, 0.105)
            dy = rng.uniform(-0.105, 0.105)
            dz = rng.uniform(-0.030, 0.065)
            p_c = [p_c0[0] + dx, p_c0[1] + dy, p_c0[2] + dz]
            aim = [p_m[0] + rng.uniform(-0.075, 0.075), p_m[1] + rng.uniform(-0.060, 0.060), p_m[2]]
            v_los = CAM.vsub(aim, p_c)
            d_los = CAM.vnorm(v_los) or 0.25
            u_los = [v / d_los for v in v_los]
            cr = _vcross(u0, u_los)
            cn = CAM.vnorm(cr)
            dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(u0, u_los))))
            ang_deg = math.degrees(math.acos(dot))
            R_swing = CAM.axis_angle_R(cr, ang_deg) if cn > 1e-6 else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            R_roll = CAM.axis_angle_R(u_los, rng.uniform(-26.0, 26.0))
            R_bc = CAM.mat_mul(R_roll, CAM.mat_mul(R_swing, R_bc0))
            R_bf = CAM.mat_mul(R_bc, R_X_t)
            p_bf = CAM.vsub(p_c, CAM.mat_vec(R_bf, t_X))
            err6 = [p_bf[i] - p0[i] for i in range(3)] + _rot_vec(R0, R_bf)
            qc = [max(lo + margin_rad, min(hi - margin_rad, q0_rad[j] + sum(J_pinv[j][c] * err6[c] for c in range(6))))
                  for j, (lo, hi) in enumerate(limits_rad)]
            try_add(qc, tag="穹顶采样")

    q0_deg = [math.degrees(v) for v in q0_rad]
    for i in range(24):
        scale = 0.50 if i < 15 else 0.80
        q_c = [max(lo + margin_rad, min(hi - margin_rad,
                                        math.radians(q0_deg[j] + rng.uniform(-(18.0 if j >= 4 else 11.0) * scale,
                                                                             (18.0 if j >= 4 else 11.0) * scale))))
               for j, (lo, hi) in enumerate(limits_rad)]
        try_add(q_c, tag="多角度采样")

    if not pool:
        return [], {"tries": tries, "pool": 0, "rejected": rej}

    # Farthest-point selection seeded from actual start uv0_frac, maximizing 2D quadrant coverage + 3D tilt/rot diversity
    selected = []
    used_uv = [list(uv0_frac)]
    used_T = [T0]
    rem = list(pool)
    while len(selected) < n and rem:
        best_i = -1
        best_sc = -1.0
        for idx, c in enumerate(rem):
            d_uv = min(math.hypot(c["uv_frac"][0] - u[0], c["uv_frac"][1] - u[1]) for u in used_uv)
            d_rot = min(CAM.rot_angle_deg(CAM.mat_mul(CAM.transpose(CAM.T_R(Ta)), CAM.T_R(c["T"]))) for Ta in used_T)
            d_pos = min(CAM.vnorm(CAM.vsub(c["pos"], CAM.T_t(Ta))) for Ta in used_T)
            sc = 1.15 * d_uv + 0.015 * d_rot + 1.2 * d_pos
            if sc > best_sc:
                best_sc = sc
                best_i = idx
        if best_i < 0:
            break
        pick = rem.pop(best_i)
        selected.append(pick)
        used_uv.append(pick["uv_frac"])
        used_T.append(pick["T"])

    # Smooth nearest-neighbor path ordering
    chosen = []
    prev_q = q0_rad
    rem_sel = list(selected)
    while rem_sel:
        rem_sel.sort(key=lambda c: sum((a - b) ** 2 for a, b in zip(prev_q, c["q"])))
        picked = None
        for idx, c in enumerate(rem_sel):
            if CAM.check_auto_path(fence_fn, prev_q, c["q"], steps=4):
                picked = rem_sel.pop(idx)
                break
        if picked is None:
            break
        chosen.append(picked)
        prev_q = picked["q"]

    return chosen, {"tries": tries, "pool": len(pool), "rejected": rej}


def eih_samples_from_intr_samples(intr_samples, K, dist=None, board_cfg=None, model_type=None):
    """V3.57/V3.58 ONE-PASS JOINT CALIBRATION BRIDGE:
    Convert captured intrinsic board samples that include robot pose ('q_deg' and 'T_base_flange') into
    high-precision Eye-in-Hand samples using the calibrated (K, dist), per-view planar RANSAC Homography corner
    inlier filtering, and solvePnP + solvePnPRefineLM."""
    cv2 = _cv2()
    import numpy as np
    if not intr_samples or not K:
        return []
    cfg = dict(DEFAULT_BOARD, **(board_cfg or {}))
    cols, rows = int(cfg.get("cols", 7)), int(cfg.get("rows", 5))
    sq_m = float(cfg.get("square_mm", 32.0)) / 1000.0
    btype = str(cfg.get("type") or BOARD_TYPES[0])
    cx_b = (0.5 * cols if ("ChArUco" in btype or "charuco" in btype.lower()) else 0.5 * (cols - 1)) * sq_m
    cy_b = (0.5 * rows if ("ChArUco" in btype or "charuco" in btype.lower()) else 0.5 * (rows - 1)) * sq_m
    Km = np.array([[float(K["fx"]), 0.0, float(K["ppx"])],
                   [0.0, float(K["fy"]), float(K["ppy"])],
                   [0.0, 0.0, 1.0]], dtype=np.float64)
    d_list = [float(v) for v in (dist or [0.0] * 5)]
    if len(d_list) > 5 and any(abs(v) > 8.0 for v in d_list[:8]):
        d_list = d_list[:5]
    mtype = str(model_type or DIST_MODELS[0])
    raw_out = []
    for s in intr_samples:
        if not s.get("T_base_flange") or not s.get("obj_pts") or not s.get("img_pts") or len(s["obj_pts"]) < 4:
            continue
        obj_c = np.asarray(s["obj_pts"], dtype=np.float64).reshape(-1, 3).copy()
        obj_c[:, 0] -= cx_b
        obj_c[:, 1] -= cy_b
        if float(np.std(obj_c[:, 0])) < 1e-4 or float(np.std(obj_c[:, 1])) < 1e-4:
            continue
        img_pts = np.asarray(s["img_pts"], dtype=np.float64).reshape(-1, 2)
        if "鱼眼" in mtype or "Fisheye" in mtype:
            Dm = np.asarray((d_list + [0.0] * 4)[:4], dtype=np.float64).reshape(4, 1)
            pts_u = cv2.fisheye.undistortPoints(img_pts.reshape(-1, 1, 2), Km, Dm, P=Km).reshape(-1, 2)
        else:
            dc = np.asarray((d_list + [0.0] * 8)[:8] if len(d_list) > 5 else (d_list + [0.0] * 5)[:5], dtype=np.float64)
            pts_u = cv2.undistortPoints(img_pts.reshape(-1, 1, 2), Km, dc, P=Km).reshape(-1, 2) if np.any(np.abs(dc) > 1e-9) else img_pts
        if len(obj_c) >= 8:
            try:
                H_u, _ = cv2.findHomography(obj_c[:, :2], pts_u, cv2.RANSAC, 1.6)
                if H_u is not None:
                    pr_u = cv2.perspectiveTransform(obj_c[:, :2].reshape(-1, 1, 2), H_u).reshape(-1, 2)
                    err_u = np.linalg.norm(pr_u - pts_u, axis=1)
                    keep = err_u <= max(1.4, 2.3 * float(np.median(err_u)))
                    if int(np.sum(keep)) >= 6 and int(np.sum(keep)) < len(obj_c) and float(np.std(obj_c[keep, 0])) > 1e-4 and float(np.std(obj_c[keep, 1])) > 1e-4:
                        obj_c = obj_c[keep]
                        pts_u = pts_u[keep]
            except Exception:
                pass
        flag = getattr(cv2, "SOLVEPNP_IPPE", cv2.SOLVEPNP_ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(obj_c, pts_u, Km, np.zeros(5, dtype=np.float64), flags=flag)
        if not ok or float(np.asarray(tvec).flatten()[2]) <= 0.04:
            ok, rvec, tvec = cv2.solvePnP(obj_c, pts_u, Km, np.zeros(5, dtype=np.float64), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        if hasattr(cv2, "solvePnPRefineLM"):
            try:
                rvec, tvec = cv2.solvePnPRefineLM(obj_c, pts_u, Km, np.zeros(5, dtype=np.float64), rvec, tvec)
            except Exception:
                pass
        proj_u, _ = cv2.projectPoints(obj_c, rvec, tvec, Km, np.zeros(5, dtype=np.float64))
        reproj_px = float(np.sqrt(np.mean(np.sum((proj_u.reshape(-1, 2) - pts_u) ** 2, axis=1))))
        p = [float(v) for v in np.asarray(tvec).flatten()]
        rv = [float(v) for v in np.asarray(rvec).flatten()]
        R_cm = CAM.rodrigues(rv)
        raw_out.append({
            "q_deg": list(s.get("q_deg") or [0.0] * 7),
            "p_cam": p,
            "rvec": rv,
            "T_cam_marker": CAM.T_from(R_cm, p),
            "T_base_flange": s["T_base_flange"],
            "spread_mm": round(max(0.2, reproj_px * 0.4), 2),
            "reproj_px": round(reproj_px, 3),
            "n_corners": int(len(obj_c)),
            "source": "charuco",
        })
    # If we have >= 9 samples, drop extreme PnP reprojection outliers (> max(1.8 px, 2.2 * median)) while keeping >= 8
    if len(raw_out) > 8:
        med_r = sorted(x["reproj_px"] for x in raw_out)[len(raw_out) // 2]
        filtered = [x for x in raw_out if x["reproj_px"] <= max(1.8, 2.3 * med_r)]
        if len(filtered) >= 8:
            raw_out = filtered
    return raw_out


def intrinsics_valid(d):
    try:
        K = d["K"]
        assert all(k in K for k in ("width", "height", "fx", "fy", "ppx", "ppy"))
        assert float(K["fx"]) > 50.0 and float(K["fy"]) > 50.0
        assert isinstance(d.get("dist"), list) and len(d["dist"]) >= 4
        return not d.get("load_error")
    except Exception:
        return False


def undistort_rgb(rgb, intr):
    """Fast cached undistortion of an RGB image using calibrated K, dist, and model_type."""
    cv2 = _cv2()
    import numpy as np
    if rgb is None or not intr or not intrinsics_valid(intr):
        return rgb
    h, w = rgb.shape[:2]
    K = intr["K"]
    dist = intr["dist"]
    mtype = str(intr.get("model_type") or DIST_MODELS[0])
    key = (w, h, round(K["fx"], 2), round(K["fy"], 2), round(K["ppx"], 2), round(K["ppy"], 2),
           tuple(round(v, 5) for v in dist), mtype)
    if key not in _UNDISTORT_CACHE:
        sx, sy = w / float(K.get("width") or w), h / float(K.get("height") or h)
        Km = np.array([[K["fx"] * sx, 0.0, K["ppx"] * sx],
                       [0.0, K["fy"] * sy, K["ppy"] * sy],
                       [0.0, 0.0, 1.0]], dtype=np.float64)
        if "鱼眼" in mtype or "Fisheye" in mtype:
            Dm = np.asarray(dist[:4], dtype=np.float64).reshape(4, 1)
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(Km, Dm, np.eye(3), Km, (w, h), cv2.CV_16SC2)
        else:
            Dm = np.asarray(dist, dtype=np.float64)
            m1, m2 = cv2.initUndistortRectifyMap(Km, Dm, None, Km, (w, h), cv2.CV_16SC2)
        if len(_UNDISTORT_CACHE) > 8:
            _UNDISTORT_CACHE.clear()
        _UNDISTORT_CACHE[key] = (m1, m2)
    m1, m2 = _UNDISTORT_CACHE[key]
    return cv2.remap(rgb, m1, m2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


# =========================================================================== 3. Eye-in-Hand Calibration (Wrist Camera)
MAX_EIH_OFFSET_M = 0.16
EIH_SWEET_SPOT_TAN = 0.55


def _vcross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _marker_basis_pts(size_m=0.024):
    h = min(0.015, max(0.005, float(size_m) / 2.0))
    return [[0.0, 0.0, 0.0], [-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0], [0.0, 0.0, h]]


def _scale_pair(scale):
    if isinstance(scale, (tuple, list)) and len(scale) >= 2:
        return float(scale[0]), float(scale[1])
    s_val = float(scale or 1.0)
    return s_val, s_val


def _sample_T_cam_marker(s, scale=1.0):
    """Build 4x4 T_cam_marker from a sample (supports scalar scale or 2D (scale_xy, scale_z) focal/marker scale)."""
    s_xy, s_z = _scale_pair(scale)
    raw = s["p_cam"]
    p = [float(raw[0]) * s_xy, float(raw[1]) * s_xy, float(raw[2]) * s_z]
    if s.get("T_cam_marker"):
        return CAM.T_from(CAM.T_R(s["T_cam_marker"]), p)
    rv = s.get("rvec")
    R = CAM.rodrigues(rv) if rv is not None else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    return CAM.T_from(R, p)


def _axial_vec(R):
    return [R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]]


def _solve_eih_point_bounded(Ai_list, A_list, p_list, y0=None, max_offset_m=MAX_EIH_OFFSET_M, iters=1500, tol=1e-11):
    """V3.56 LEVER-ARM-BOUNDED POINT-DUALITY SOLVER:
    Solves X p_cam_i ≈ A_i^-1 y_pos where X = T_flange_cam = (R_X, t_X) and y_pos = marker_in_base,
    subject to physical wrist-mount constraint ||t_X|| <= max_offset_m (default 0.16 m = 160 mm).
    Prevents noisy/peripheral wide-angle samples from pulling link7->cam out to +270 mm."""
    y = list(y0) if y0 is not None else [0.0, 0.0, 0.0]
    n = len(Ai_list)
    R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    t = [0.0, 0.0, 0.0]
    cp = [sum(p[i] for p in p_list) / n for i in range(3)]
    p_centered = [CAM.vsub(p, cp) for p in p_list]
    it = 0
    for it in range(1, iters + 1):
        tgt = [CAM.T_apply(Ai, y) for Ai in Ai_list]
        R, t = CAM.kabsch(p_list, tgt)
        tnorm = CAM.vnorm(t)
        if max_offset_m is not None and tnorm > max_offset_m:
            s_c = float(max_offset_m) / tnorm
            t = [v * s_c for v in t]
            # Re-solve optimal rotation R around clamped translation t: R p_i ≈ tgt_i - t
            q_shifted = [CAM.vsub(tg, t) for tg in tgt]
            R, _ = CAM.kabsch(p_centered, q_shifted)
        acc = [0.0, 0.0, 0.0]
        for A, p in zip(A_list, p_list):
            xp = CAM.mat_vec(R, p)
            xt = [xp[0] + t[0], xp[1] + t[1], xp[2] + t[2]]
            pred = CAM.T_apply(A, xt)
            acc = [acc[0] + pred[0] / n, acc[1] + pred[1] / n, acc[2] + pred[2] / n]
        step = CAM.vnorm(CAM.vsub(acc, y))
        y = acc
        if step < tol:
            break
    X = CAM.T_from(R, t)
    res = [CAM.vnorm(CAM.vsub(CAM.T_apply(CAM.mat_mul(A, X), p), y)) * 1000.0 for A, p in zip(A_list, p_list)]
    return {
        "T_flange_cam": X,
        "marker_in_base": y,
        "res_mm": res,
        "mean_mm": sum(res) / n,
        "max_mm": max(res),
        "iters": it,
    }


def solve_eye_in_hand(A_list, B_list, size_m=0.024, iters=1500, tol=1e-11, max_offset_m=MAX_EIH_OFFSET_M):
    """V3.52/V3.56 ROBUST EYE-IN-HAND SOLVER:
    A_i = T_base_flange (FK), B_i = T_cam_marker (tabletop marker in wrist camera frame).
    Unknowns: X = T_flange_cam (camera in link7), Y = T_base_marker (marker in base_link), with A_i X p_i ≈ y_pos.
    - Stage 1: Closed-form rotation initialization + 40-step small-basis Kabsch (works for 3-4 probe poses).
    - Stage 2 (when >= 4 poses): Polishes (X, y_pos) via lever-arm-bounded point-duality solver from both Stage 1 Y
      and a pure point-displacement physical initialization (||t_X|| <= 160 mm), immune to PnP rvec planar flips
      and immune to +270 mm lever-arm runaway.
    -> {'T_flange_cam', 'T_base_marker', 'marker_in_base', 'res_mm', 'mean_mm', 'max_mm', 'iters'}"""
    C_pts = _marker_basis_pts(size_m)
    Ai_list = [CAM.T_inv(A) for A in A_list]
    p_list = [CAM.T_t(B) for B in B_list]
    n = len(A_list)
    P_rot, Q_rot = [], []
    for i in range(n):
        RAi_t = CAM.transpose(CAM.T_R(A_list[i]))
        RBi = CAM.T_R(B_list[i])
        for j in range(i + 1, n):
            MA = CAM.mat_mul(RAi_t, CAM.T_R(A_list[j]))
            MB = CAM.mat_mul(RBi, CAM.transpose(CAM.T_R(B_list[j])))
            va, vb = _axial_vec(MA), _axial_vec(MB)
            if CAM.vnorm(va) > 1e-4 and CAM.vnorm(vb) > 1e-4:
                P_rot.extend([vb, [-v for v in vb]])
                Q_rot.extend([va, [-v for v in va]])
    if len(P_rot) >= 4:
        Rx0, _ = CAM.kabsch(P_rot, Q_rot)
    else:
        Rx0 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    RAs = [CAM.T_R(A) for A in A_list]
    d_list = [[u + v for u, v in zip(CAM.mat_vec(RA, CAM.mat_vec(Rx0, CAM.T_t(B))), CAM.T_t(A))]
              for A, RA, B in zip(A_list, RAs, B_list)]
    RA_mean = [[sum(RA[r][c] for RA in RAs) / n for c in range(3)] for r in range(3)]
    d_mean = [sum(d[i] for d in d_list) / n for i in range(3)]
    M_ls = [[0.0, 0.0, 0.0] for _ in range(3)]
    b_ls = [0.0, 0.0, 0.0]
    for RA, d in zip(RAs, d_list):
        D = [[RA[r][c] - RA_mean[r][c] for c in range(3)] for r in range(3)]
        rhs = [d_mean[i] - d[i] for i in range(3)]
        for r in range(3):
            for c in range(3):
                M_ls[r][c] += sum(D[k][r] * D[k][c] for k in range(3))
            b_ls[r] += sum(D[k][r] * rhs[k] for k in range(3))
    tx0 = CAM.solve_3x3(M_ls, b_ls) or [0.0, 0.0, 0.0]
    if CAM.vnorm(tx0) > max_offset_m:
        tx0 = [0.0, 0.0, 0.035]
    ty0 = [u + v for u, v in zip(CAM.mat_vec(RA_mean, tx0), d_mean)]
    Ry0 = CAM.mat_mul(RAs[0], CAM.mat_mul(Rx0, CAM.T_R(B_list[0])))
    X = CAM.T_from(Rx0, tx0)
    Y = CAM.T_from(Ry0, ty0)

    n_kabsch = 35 if n >= 5 else min(iters, 180)
    it = 0
    for it in range(1, n_kabsch + 1):
        P_cam, Q_flange = [], []
        for Ai, B in zip(Ai_list, B_list):
            AiY = CAM.mat_mul(Ai, Y)
            for c in C_pts:
                P_cam.append(CAM.T_apply(B, c))
                Q_flange.append(CAM.T_apply(AiY, c))
        Rx, tx = CAM.kabsch(P_cam, Q_flange)
        if max_offset_m is not None and CAM.vnorm(tx) > max_offset_m:
            s_c = float(max_offset_m) / CAM.vnorm(tx)
            tx = [v * s_c for v in tx]
        X_new = CAM.T_from(Rx, tx)
        P_m, Q_base = [], []
        for A, B in zip(A_list, B_list):
            AXB = CAM.mat_mul(A, CAM.mat_mul(X_new, B))
            for c in C_pts:
                P_m.append(c)
                Q_base.append(CAM.T_apply(AXB, c))
        Ry, ty = CAM.kabsch(P_m, Q_base)
        Y_new = CAM.T_from(Ry, ty)
        step = CAM.vnorm(CAM.vsub(tx, CAM.T_t(X))) + CAM.vnorm(CAM.vsub(ty, CAM.T_t(Y)))
        X, Y = X_new, Y_new
        if step < tol:
            break

    if n >= 4:
        # Candidate 1: point-duality starting from Stage 1 Y
        r_pt1 = _solve_eih_point_bounded(Ai_list, A_list, p_list, y0=CAM.T_t(Y),
                                         max_offset_m=max_offset_m, iters=iters, tol=tol)
        # Candidate 2: point-only physical initialization (immune to Stage 1 rvec flip corruption)
        R_bc_init, _ = CAM.kabsch(p_list, [[-v for v in CAM.T_t(A)] for A in A_list])
        Rx_init = CAM.mat_mul(CAM.transpose(RAs[0]), R_bc_init)
        X_init = CAM.T_from(Rx_init, [0.0, 0.0, 0.035])
        y0_phys = [sum(CAM.T_apply(CAM.mat_mul(A, X_init), p)[k] for A, p in zip(A_list, p_list)) / n for k in range(3)]
        r_pt2 = _solve_eih_point_bounded(Ai_list, A_list, p_list, y0=y0_phys,
                                         max_offset_m=max_offset_m, iters=iters, tol=tol)
        r_pt = r_pt2 if r_pt2["mean_mm"] + 1e-6 < r_pt1["mean_mm"] else r_pt1
        stage1_mean = sum(CAM.vnorm(CAM.vsub(CAM.T_apply(CAM.mat_mul(A, X), p), CAM.T_t(Y))) * 1000.0
                          for A, p in zip(A_list, p_list)) / n
        if r_pt["mean_mm"] <= stage1_mean + 0.5 or CAM.vnorm(CAM.T_t(X)) > max_offset_m:
            X = r_pt["T_flange_cam"]
            y_pos = r_pt["marker_in_base"]
            it = r_pt["iters"]
            P_m, Q_base = [], []
            for A, B in zip(A_list, B_list):
                AXB = CAM.mat_mul(A, CAM.mat_mul(X, B))
                for c in C_pts[1:]:
                    P_m.append(c)
                    Q_base.append(CAM.mat_vec(CAM.T_R(AXB), c))
            Ry, _ = CAM.kabsch(P_m, Q_base)
            Y = CAM.T_from(Ry, y_pos)

    y_pos = CAM.T_t(Y)
    res = [CAM.vnorm(CAM.vsub(CAM.T_apply(CAM.mat_mul(A, X), p), y_pos)) * 1000.0
           for A, p in zip(A_list, p_list)]
    return {
        "T_flange_cam": X,
        "T_base_marker": Y,
        "marker_in_base": y_pos,
        "res_mm": res,
        "mean_mm": sum(res) / len(res),
        "max_mm": max(res),
        "iters": it,
    }


def _calibrate_eih_at_scale(samples, scale=1.0, size_mm=24.0, reject_mm=5.0, min_samples=HANDEYE_MIN_SAMPLES):
    """Run residual outlier rejection + LOO outlier rejection + solve_eye_in_hand at scalar or (s_xy, s_z) scale."""
    s_xy, s_z = _scale_pair(scale)
    idx = list(range(len(samples)))
    size_m = float(size_mm) / 1000.0
    rejected = []
    while True:
        A = [samples[i]["T_base_flange"] for i in idx]
        B = [_sample_T_cam_marker(samples[i], scale=(s_xy, s_z)) for i in idx]
        r = solve_eye_in_hand(A, B, size_m=size_m * s_xy)
        res = r["res_mm"]
        med = sorted(res)[len(res) // 2]
        k = max(range(len(res)), key=lambda m: res[m])
        if len(idx) > min_samples and res[k] > max(reject_mm, 2.5 * med):
            rejected.append(idx.pop(k))
            continue
        break
    for _loo_pass in range(3):
        loo = []
        for j in range(len(idx)):
            A2 = [samples[i]["T_base_flange"] for m, i in enumerate(idx) if m != j]
            B2 = [_sample_T_cam_marker(samples[i], scale=(s_xy, s_z)) for m, i in enumerate(idx) if m != j]
            rj = solve_eye_in_hand(A2, B2, size_m=size_m * s_xy, iters=350)
            i = idx[j]
            Bi = _sample_T_cam_marker(samples[i], scale=(s_xy, s_z))
            p_pred = CAM.T_apply(CAM.mat_mul(samples[i]["T_base_flange"], rj["T_flange_cam"]), CAM.T_t(Bi))
            loo.append(CAM.vnorm(CAM.vsub(p_pred, rj["marker_in_base"])) * 1000.0)
        if len(idx) <= max(min_samples, 7) or max(loo) <= CAM.CALIB_GOOD_MAX_MM:
            break
        med_loo = sorted(loo)[len(loo) // 2]
        k_bad = max(range(len(loo)), key=lambda j: loo[j])
        if loo[k_bad] <= max(CAM.CALIB_GOOD_MAX_MM, 2.1 * med_loo):
            break
        cand_idx = [i for m, i in enumerate(idx) if m != k_bad]
        A_cand = [samples[i]["T_base_flange"] for i in cand_idx]
        if CAM.rotation_span_deg(A_cand) < 20.0 or CAM.translation_span_m(A_cand) < 0.12:
            break
        rejected.append(idx.pop(k_bad))
        A = [samples[i]["T_base_flange"] for i in idx]
        B = [_sample_T_cam_marker(samples[i], scale=(s_xy, s_z)) for i in idx]
        r = solve_eye_in_hand(A, B, size_m=size_m * s_xy)
    A = [samples[i]["T_base_flange"] for i in idx]
    rs, ts = CAM.rotation_span_deg(A), CAM.translation_span_m(A)
    is_charuco = any(samples[i].get("source") == "charuco" for i in idx)
    min_rs = 15.0 if is_charuco else 20.0
    min_ts = 0.08 if is_charuco else 0.12
    warn = []
    if rs < min_rs:
        warn.append(f"手腕朝向变化只有 {rs:.0f}°（建议 ≥ {min_rs:.0f}°）：多从不同倾角观察桌面标记")
    if ts < min_ts:
        warn.append(f"法兰位置范围只有 {ts * 100:.0f} cm（建议 ≥ {min_ts * 100:.0f} cm）")
    loo_mean, loo_max = sum(loo) / len(loo), max(loo)
    good = loo_mean <= CAM.CALIB_GOOD_MEAN_MM and loo_max <= CAM.CALIB_GOOD_MAX_MM and not warn
    X = r["T_flange_cam"]
    return dict(
        r,
        ok=True,
        used=idx,
        rejected=rejected,
        loo_mm=loo,
        loo_mean_mm=loo_mean,
        loo_max_mm=loo_max,
        rot_span_deg=rs,
        trans_span_m=ts,
        warnings=warn,
        good=good,
        cam_in_flange=CAM.T_t(X),
        n_used=len(idx),
        pnp_scale=round(float(s_xy), 5),
        pnp_scale_z=round(float(s_z), 5),
    )


def calibrate_eye_in_hand(samples, size_mm=40.0, reject_mm=5.0, min_samples=HANDEYE_MIN_SAMPLES):
    """V3.56/V3.57 Eye-in-hand calibration with lever-arm regularization, LOO outlier rejection, and 2D anisotropic
    (alpha_xy, alpha_z) focal-length + marker-scale self-calibration (handles both printed marker size offset AND
    wide-angle intrinsic fx/fy inflation)."""
    if len(samples) < min_samples:
        return {"ok": False, "error": f"至少需要 {min_samples} 个样本（现在 {len(samples)} 个）"}
    best = _calibrate_eih_at_scale(samples, scale=(1.0, 1.0), size_mm=size_mm, reject_mm=reject_mm, min_samples=min_samples)
    if not best["good"] and len(samples) >= min_samples:
        used0 = best["used"] if len(best["used"]) >= min_samples else list(range(len(samples)))
        A_sub = [samples[i]["T_base_flange"] for i in used0]
        Ai_sub = [CAM.T_inv(A) for A in A_sub]
        p_raw = [samples[i]["p_cam"] for i in used0]
        neg_tA = [[-v for v in CAM.T_t(A)] for A in A_sub]
        R0_t = CAM.transpose(CAM.T_R(A_sub[0]))

        def _eval_pair(axy, az):
            p_sc = [[p[0] * axy, p[1] * axy, p[2] * az] for p in p_raw]
            R_bc_i, _ = CAM.kabsch(p_sc, neg_tA)
            X_i = CAM.T_from(CAM.mat_mul(R0_t, R_bc_i), [0.0, 0.0, 0.035])
            y0_a = [sum(CAM.T_apply(CAM.mat_mul(A, X_i), p)[k] for A, p in zip(A_sub, p_sc)) / len(A_sub)
                    for k in range(3)]
            res_a = _solve_eih_point_bounded(Ai_sub, A_sub, p_sc, y0=y0_a, max_offset_m=0.14, iters=140)
            s_res = sorted(res_a["res_mm"])
            n_keep = max(min_samples, int(len(s_res) * 0.8))
            return sum(s_res[:n_keep]) / n_keep

        # Stage A: 1D uniform scale search (axy == az)
        grid_1d = [0.55 + 0.04 * k for k in range(23)]
        b1 = min(grid_1d, key=lambda a: _eval_pair(a, a))
        lo1, hi1 = max(0.45, b1 - 0.05), min(1.55, b1 + 0.05)
        for _ in range(7):
            m1, m2 = lo1 + 0.382 * (hi1 - lo1), lo1 + 0.618 * (hi1 - lo1)
            if _eval_pair(m1, m1) < _eval_pair(m2, m2):
                hi1 = m2
            else:
                lo1 = m1
        a_uni = 0.5 * (lo1 + hi1)
        if abs(a_uni - 1.0) >= 0.012:
            cand_uni = _calibrate_eih_at_scale(samples, scale=(a_uni, a_uni), size_mm=size_mm,
                                               reject_mm=reject_mm, min_samples=min_samples)
            if (cand_uni["good"] and not best["good"]) or (cand_uni["loo_max_mm"] + 0.25 < best["loo_max_mm"]):
                best = cand_uni

        # Stage B: 2D anisotropic (alpha_xy, alpha_z) search when uniform scale still leaves LOO > 5 mm (e.g. fx bias)
        if not best["good"]:
            axy, az = 1.0, a_uni
            for _round in range(2):
                best_az = min((0.14 + 0.052 * k for k in range(32)), key=lambda z_: _eval_pair(axy, z_))
                lo_z, hi_z = max(0.12, best_az - 0.060), min(1.85, best_az + 0.060)
                for _ in range(7):
                    m1, m2 = lo_z + 0.382 * (hi_z - lo_z), lo_z + 0.618 * (hi_z - lo_z)
                    if _eval_pair(axy, m1) < _eval_pair(axy, m2):
                        hi_z = m2
                    else:
                        lo_z = m1
                az = 0.5 * (lo_z + hi_z)
                best_axy = min((0.36 + 0.055 * k for k in range(24)), key=lambda x_: _eval_pair(x_, az))
                lo_x, hi_x = max(0.32, best_axy - 0.060), min(1.75, best_axy + 0.060)
                for _ in range(6):
                    m1, m2 = lo_x + 0.382 * (hi_x - lo_x), lo_x + 0.618 * (hi_x - lo_x)
                    if _eval_pair(m1, az) < _eval_pair(m2, az):
                        hi_x = m2
                    else:
                        lo_x = m1
                axy = 0.5 * (lo_x + hi_x)
            cand_2d = _calibrate_eih_at_scale(samples, scale=(axy, az), size_mm=size_mm,
                                              reject_mm=reject_mm, min_samples=min_samples)
            if (cand_2d["good"] and not best["good"]) or (cand_2d["loo_max_mm"] + 0.25 < best["loo_max_mm"]):
                best = cand_2d
    return best


def handeye_valid(d):
    try:
        X = d["T_flange_cam"]
        assert len(X) == 4 and all(len(r) == 4 for r in X)
        assert len(d["marker_in_base"]) == 3
        return not d.get("load_error")
    except Exception:
        return False


def wrist_T_base_cam(T_base_flange, calib_eih):
    """Compute T_base_cam = T_base_flange @ T_flange_cam for the wrist camera at the current arm pose."""
    return CAM.mat_mul(T_base_flange, calib_eih["T_flange_cam"])


def wrist_marker_error_mm(calib_eih, T_base_flange, p_cam):
    T_bc = wrist_T_base_cam(T_base_flange, calib_eih)
    s_xy = float(calib_eih.get("pnp_scale") or 1.0)
    s_z = float(calib_eih.get("pnp_scale_z") or s_xy)
    p_sc = [float(p_cam[0]) * s_xy, float(p_cam[1]) * s_xy, float(p_cam[2]) * s_z]
    return CAM.vnorm(CAM.vsub(CAM.T_apply(T_bc, p_sc), calib_eih["marker_in_base"])) * 1000.0


def eih_pose_visible(A, T_flange_cam, T_base_marker, K, size_m=0.024,
                     max_tilt_deg=EIH_VIEW_MAX_DEG, min_px=24.0, margin_px=64.0, z_range=EIH_Z_RANGE_M,
                     max_off_axis=None):
    """V3.51/V3.56 EIH VISIBILITY SCREEN: given candidate flange pose A = T_base_flange, X = T_flange_cam,
    and Y = T_base_marker, predict B = T_cam_marker = (A X)^-1 Y and check:
      1. Marker distance z_cam in z_range (0.10..0.58 m);
      2. Optional central sweet-spot check: hypot(x_cam, y_cam) / z_cam <= max_off_axis;
      3. Centre and 4 corners project inside [margin_px, W-margin_px] x [margin_px, H-margin_px];
      4. Apparent marker size in pixels >= min_px;
      5. Viewing angle between line-of-sight and marker normal <= max_tilt_deg.
    -> (ok: bool, reason: str, view_deg: float)"""
    if not A or not T_flange_cam or not T_base_marker or not K:
        return False, "缺少模型", 90.0
    T_bc = CAM.mat_mul(A, T_flange_cam)
    B = CAM.mat_mul(CAM.T_inv(T_bc), T_base_marker)
    p_cam = CAM.T_t(B)
    z = float(p_cam[2])
    if not (z_range[0] <= z <= z_range[1]):
        return False, "距离", 90.0
    if max_off_axis is not None and math.hypot(float(p_cam[0]), float(p_cam[1])) / max(0.05, z) > float(max_off_axis):
        return False, "出画面", 90.0
    w = float(K.get("width") or 1280.0)
    h = float(K.get("height") or 720.0)
    mx = min(float(margin_px), 0.22 * w)
    my = min(float(margin_px), 0.22 * h)
    half = 0.5 * float(size_m)
    for pt in ([0.0, 0.0, 0.0], [-half, half, 0.0], [h_ := half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]):
        uv = CAM.project(CAM.T_apply(B, pt), K)
        if not uv or not (mx <= uv[0] <= w - mx and my <= uv[1] <= h - my):
            return False, "出画面", 90.0
    px_side = 0.5 * (float(K["fx"]) + float(K["fy"])) * float(size_m) / z
    if px_side < min_px:
        return False, "标记太小", 90.0
    R_b = CAM.T_R(B)
    n_cam = [R_b[0][2], R_b[1][2], R_b[2][2]]
    dist = max(1e-6, CAM.vnorm(p_cam))
    ray = [p_cam[0] / dist, p_cam[1] / dist, p_cam[2] / dist]
    cos_v = abs(sum(ray[i] * n_cam[i] for i in range(3)))
    view_deg = math.degrees(math.acos(min(1.0, max(0.0, cos_v))))
    if view_deg > max_tilt_deg:
        return False, "倾角过大", view_deg
    return True, "ok", view_deg


def initial_eih_prior_from_start(s0):
    """V3.52/V3.56 START-POSE PRIOR: from a single starting sample s0 (T_base_flange A0 + T_cam_marker B0),
    construct a tabletop-consistent prior (T_flange_cam, T_base_marker) with horizontal tabletop normal +Z_base."""
    A0 = s0["T_base_flange"]
    B0 = _sample_T_cam_marker(s0)
    R_b0 = CAM.T_R(B0)
    R_flip = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
    R_bc0 = CAM.mat_mul(R_flip, CAM.transpose(R_b0))
    if R_bc0[2][2] > 0:
        R_bc0 = R_flip
    p_f0 = CAM.T_t(A0)
    T_bc0 = CAM.T_from(R_bc0, p_f0)
    X_prior = CAM.mat_mul(CAM.T_inv(A0), T_bc0)
    p_m0 = CAM.T_apply(T_bc0, CAM.T_t(B0))
    # Force Y_prior to have horizontal tabletop normal +Z_base = [0, 0, 1] and height below A0
    p_m0[2] = min(p_m0[2], p_f0[2] - max(0.14, float(CAM.T_t(B0)[2]) * 0.75))
    Y_prior = CAM.T_from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], p_m0)
    return {
        "T_flange_cam": X_prior,
        "T_base_marker": Y_prior,
        "marker_in_base": list(p_m0),
        "mean_mm": 0.0,
    }


def estimate_eih_from_probes(samples, size_mm=24.0):
    """V3.51/V3.57 EIH PROBE MODEL:
    - Evaluates both pure-translation Kabsch orientation solver (optimal when R_k ≈ R_0) AND lever-arm-bounded
      6-DoF point-duality solver (optimal when probes include >= 2.5 deg wrist rotation), selecting whichever
      achieves lower 3D prediction residual while enforcing tabletop height & horizontal normal priors."""
    if not samples:
        return None
    if len(samples) < 3:
        return initial_eih_prior_from_start(samples[0])
    A_list = [s["T_base_flange"] for s in samples]
    B_list = [_sample_T_cam_marker(s) for s in samples]
    p_list = [CAM.T_t(B) for B in B_list]
    A0 = A_list[0]
    R0 = CAM.T_R(A0)
    p_f0 = CAM.T_t(A0)
    rot_span = CAM.rotation_span_deg(A_list)
    if rot_span >= 12.0:
        r = solve_eye_in_hand(A_list, B_list, size_m=float(size_mm) / 1000.0, iters=350, max_offset_m=0.14)
        tx = CAM.T_t(r["T_flange_cam"])
        if CAM.vnorm(tx) <= MAX_EIH_OFFSET_M and r["mean_mm"] <= 6.0:
            return {
                "T_flange_cam": r["T_flange_cam"],
                "T_base_marker": r["T_base_marker"],
                "marker_in_base": r["marker_in_base"],
                "mean_mm": r["mean_mm"],
            }
    # Pure-translation Kabsch candidate (exact when R_k == R_0):
    Q_neg_t = [[-v for v in CAM.T_t(A)] for A in A_list]
    R_bc0, t_fit = CAM.kabsch(p_list, Q_neg_t)
    rms_tr_mm = math.sqrt(sum(CAM.vnorm(CAM.vsub(CAM.vsub(q, t_fit), CAM.mat_vec(R_bc0, p))) ** 2
                              for p, q in zip(p_list, Q_neg_t)) / len(p_list)) * 1000.0
    Rx_tr = CAM.mat_mul(CAM.transpose(R0), R_bc0)
    X_tr = CAM.T_from(Rx_tr, [0.0, 0.0, 0.035])

    # Rotated-probe candidate (when fallback probes rotated the wrist by 2..12 deg):
    best_X = X_tr
    best_mm = rms_tr_mm
    if rot_span >= 2.0:
        r_rot = solve_eye_in_hand(A_list, B_list, size_m=float(size_mm) / 1000.0, iters=300, max_offset_m=0.10)
        if r_rot["mean_mm"] < best_mm:
            best_X = r_rot["T_flange_cam"]
            best_mm = r_rot["mean_mm"]

    p_m_list = [CAM.T_apply(CAM.mat_mul(A, best_X), p) for A, p in zip(A_list, p_list)]
    p_m = [sum(pm[k] for pm in p_m_list) / len(p_m_list) for k in range(3)]
    if p_m[2] > p_f0[2] - 0.10:
        p_m[2] = p_f0[2] - max(0.16, min(0.32, float(p_list[0][2]) * 0.75))
    Y_reg = CAM.T_from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], p_m)
    return {
        "T_flange_cam": best_X,
        "T_base_marker": Y_reg,
        "marker_in_base": p_m,
        "mean_mm": best_mm,
    }


def plan_eih_auto_targets(fk_fn, ik_fn, q0_rad, limits_rad, eih_model, K, n=12, size_m=0.024,
                          fence_fn=lambda q: True, seed=0):
    """V3.56 LOOK-AT DOME + SWEET-SPOT EIH AUTO PLANNER:
    1. Stabilizes tabletop marker normal (+Z_base) and safe flange height floor so noisy initial probes never trigger
       '倾角过大 101' or '太低 140' mass rejections;
    2. Uses precomputed Fast Jacobian IK (CAM.build_fast_jacobian_ik) to generate Look-At Dome poses that translate the
       wrist across a 22 cm x 22 cm x 10 cm dome while automatically counter-tilting the wrist to keep the tabletop
       marker inside the central distortion-free sweet spot (off-axis <= 28 deg) of the 130° wide-angle lens;
    3. Supplements with multi-scale joint perturbations, selects n diverse poses maximizing 3D position + 3D rotation
       span, and orders them along a smooth nearest-neighbor path.
    -> (targets: [{'q': [rad...], 'T': 4x4, 'view_deg': float}, ...], stats_dict)"""
    import random
    rng = random.Random(seed)
    T0 = fk_fn(q0_rad)
    if T0 is None or not eih_model or not eih_model.get("T_flange_cam") or not eih_model.get("T_base_marker"):
        return [], {"tries": 0, "pool": 0, "rejected": {"缺少初值": 1}, "span_m": 0.0, "rot_max_deg": 0.0}
    X = eih_model["T_flange_cam"]
    Y_raw = eih_model["T_base_marker"]
    p_m = list(CAM.T_t(Y_raw))
    p0 = CAM.T_t(T0)
    R0 = CAM.T_R(T0)
    R0_t = CAM.transpose(R0)
    margin_rad = math.radians(getattr(CAM, "SOFT_MARGIN_DEG", 3.0))

    # Stabilize tabletop marker normal for visibility screening: if Y_raw's Z-normal is tilted > 25 deg from the
    # starting camera's line-of-sight, align Y_vis normal with +Z_base (or opposing starting line-of-sight).
    T_bc0 = CAM.mat_mul(T0, X)
    p_c0 = CAM.T_t(T_bc0)
    R_bc0 = CAM.T_R(T_bc0)
    v0 = CAM.vsub(p_m, p_c0)
    d0 = CAM.vnorm(v0) or 0.25
    u0 = [v / d0 for v in v0]
    n_y_raw = [CAM.T_R(Y_raw)[0][2], CAM.T_R(Y_raw)[1][2], CAM.T_R(Y_raw)[2][2]]
    cos_ny = abs(sum(u0[i] * n_y_raw[i] for i in range(3)))
    if cos_ny < math.cos(math.radians(25.0)):
        Y_vis = CAM.T_from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], p_m)
        ok0, _, _ = eih_pose_visible(T0, X, Y_vis, K, size_m=size_m)
        if not ok0:
            # Fallback: align marker normal directly opposite u0
            Y_vis = CAM.T_from(CAM.mat_mul([[1, 0, 0], [0, -1, 0], [0, 0, -1]], R_bc0), p_m)
    else:
        Y_vis = Y_raw

    # Safe flange height floor: never reject poses near T0[2][3] even if p_m[2] was slightly overestimated
    z_floor = min(p_m[2] + 0.08, p0[2] - 0.055)

    rej = {}
    pool_sweet = []
    pool_all = []
    tries = 0

    def try_add(q_cand):
        nonlocal tries
        tries += 1
        if q_cand is None:
            rej["IK"] = rej.get("IK", 0) + 1
            return
        for j, (lo, hi) in enumerate(limits_rad):
            if not (lo + margin_rad - 1e-6 <= q_cand[j] <= hi - margin_rad + 1e-6):
                rej["软限位"] = rej.get("软限位", 0) + 1
                return
        if not fence_fn(q_cand):
            rej["安全区"] = rej.get("安全区", 0) + 1
            return
        T = fk_fn(q_cand)
        if T is None:
            rej["FK"] = rej.get("FK", 0) + 1
            return
        if T[2][3] < z_floor:
            rej["太低"] = rej.get("太低", 0) + 1
            return
        rot_d = CAM.rot_angle_deg(CAM.mat_mul(R0_t, CAM.T_R(T)))
        if rot_d > EIH_ROT_MAX_DEG * 1.65:
            rej["转角过大"] = rej.get("转角过大", 0) + 1
            return
        ok_v, why_v, vdeg = eih_pose_visible(T, X, Y_vis, K, size_m=size_m)
        if not ok_v:
            rej[why_v] = rej.get(why_v, 0) + 1
            return
        if not CAM.check_auto_path(fence_fn, q0_rad, q_cand, steps=4):
            rej["路径安全区"] = rej.get("路径安全区", 0) + 1
            return
        item = {"q": list(q_cand), "T": T, "pos": CAM.T_t(T), "rot_deg": rot_d, "view_deg": vdeg}
        pool_all.append(item)
        # Check whether marker stays in the central sweet spot of the wide-angle lens (<= 28.8 deg off-axis)
        ok_sw, _, _ = eih_pose_visible(T, X, Y_vis, K, size_m=size_m,
                                       margin_px=max(80.0, 0.15 * float(K.get("width") or 1280.0)),
                                       max_off_axis=EIH_SWEET_SPOT_TAN)
        if ok_sw:
            pool_sweet.append(item)

    # Phase 1: 1-step Linearized Jacobian Look-At Dome candidates (~5 us per candidate, 0 extra fk_fn calls!)
    def _rot_vec(Ra, Rb):
        R_ = [[sum(Rb[i][k] * Ra[j][k] for k in range(3)) for j in range(3)] for i in range(3)]
        return [0.5 * (R_[2][1] - R_[1][2]), 0.5 * (R_[0][2] - R_[2][0]), 0.5 * (R_[1][0] - R_[0][1])]

    nj = len(q0_rad)
    h_j = 1e-5
    cols_j = []
    for j in range(nj):
        qp = list(q0_rad)
        qp[j] += h_j
        Tp = fk_fn(qp)
        if Tp is not None:
            cols_j.append([(Tp[i][3] - p0[i]) / h_j for i in range(3)] + [x / h_j for x in _rot_vec(R0, CAM.T_R(Tp))])
    J_pinv = None
    if len(cols_j) == nj:
        J0 = [[cols_j[j][r] for j in range(nj)] for r in range(6)]
        lam = 2.0e-3
        JJT = [[sum(J0[r][k] * J0[c][k] for k in range(nj)) + (lam if r == c else 0.0) for c in range(6)] for r in range(6)]
        aug = [JJT[r] + [1.0 if r == c else 0.0 for c in range(6)] for r in range(6)]
        ok_inv = True
        for c in range(6):
            pv = max(range(c, 6), key=lambda r: abs(aug[r][c]))
            aug[c], aug[pv] = aug[pv], aug[c]
            if abs(aug[c][c]) < 1e-12:
                ok_inv = False
                break
            for r in range(6):
                if r != c:
                    f = aug[r][c] / aug[c][c]
                    for k in range(c, 12):
                        aug[r][k] -= f * aug[c][k]
        if ok_inv:
            inv_JJT = [[aug[r][6 + c] / aug[r][r] for c in range(6)] for r in range(6)]
            J_pinv = [[sum(J0[r][j] * inv_JJT[r][c] for r in range(6)) for c in range(6)] for j in range(nj)]

    R_X = CAM.T_R(X)
    R_X_t = CAM.transpose(R_X)
    t_X = CAM.T_t(X)
    if J_pinv is not None:
        for _ in range(55):
            dx = rng.uniform(-0.11, 0.11)
            dy = rng.uniform(-0.11, 0.11)
            dz = rng.uniform(-0.035, 0.075)
            p_c = [p_c0[0] + dx, p_c0[1] + dy, p_c0[2] + dz]
            aim = [p_m[0] + rng.uniform(-0.022, 0.022),
                   p_m[1] + rng.uniform(-0.022, 0.022),
                   p_m[2]]
            v_los = CAM.vsub(aim, p_c)
            d_los = CAM.vnorm(v_los) or 0.25
            u_los = [v / d_los for v in v_los]
            cr = _vcross(u0, u_los)
            cn = CAM.vnorm(cr)
            dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(u0, u_los))))
            ang_deg = math.degrees(math.acos(dot))
            R_swing = CAM.axis_angle_R(cr, ang_deg) if cn > 1e-6 else [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            roll_deg = rng.uniform(-25.0, 25.0)
            R_roll = CAM.axis_angle_R(u_los, roll_deg)
            R_bc = CAM.mat_mul(R_roll, CAM.mat_mul(R_swing, R_bc0))
            R_bf = CAM.mat_mul(R_bc, R_X_t)
            p_bf = CAM.vsub(p_c, CAM.mat_vec(R_bf, t_X))
            err6 = [p_bf[i] - p0[i] for i in range(3)] + _rot_vec(R0, R_bf)
            qc = [max(lo + margin_rad, min(hi - margin_rad, q0_rad[j] + sum(J_pinv[j][c] * err6[c] for c in range(6))))
                  for j, (lo, hi) in enumerate(limits_rad)]
            try_add(qc)

    # Phase 2: Multi-scale joint perturbations (ensures broad exploration and full backward compatibility)
    q0_deg = [math.degrees(v) for v in q0_rad]
    for i in range(55):
        scale = 0.55 if i < 25 else (0.90 if i < 55 else 1.25)
        q_c = []
        for j, (lo, hi) in enumerate(limits_rad):
            span_deg = (22.0 if j >= 4 else 14.0) * scale
            v_rad = math.radians(q0_deg[j] + rng.uniform(-span_deg, span_deg))
            v_rad = max(lo + margin_rad, min(hi - margin_rad, v_rad))
            q_c.append(v_rad)
        try_add(q_c)

    pool = pool_sweet if len(pool_sweet) >= max(n, 6) else pool_all
    if not pool:
        return [], {"tries": tries, "pool": 0, "rejected": rej, "span_m": 0.0, "rot_max_deg": 0.0}

    # Farthest-point selection in 3D position + 3D rotation
    selected = []
    used_T = [T0]
    used_p = [p0]
    rem = list(pool)
    while len(selected) < n and rem:
        best_i = -1
        best_sc = -1.0
        for idx, c in enumerate(rem):
            d_pos = min(CAM.vnorm(CAM.vsub(c["pos"], p)) for p in used_p)
            d_rot = min(CAM.rot_angle_deg(CAM.mat_mul(CAM.transpose(CAM.T_R(Ta)), CAM.T_R(c["T"]))) for Ta in used_T)
            sc = d_pos + 0.0045 * d_rot
            if sc > best_sc:
                best_sc = sc
                best_i = idx
        if best_i < 0:
            break
        pick = rem.pop(best_i)
        selected.append(pick)
        used_T.append(pick["T"])
        used_p.append(pick["pos"])

    # Order selected poses along a smooth nearest-neighbor path in joint space (verifying path fence at each hop)
    chosen = []
    prev_q = q0_rad
    rem_sel = list(selected)
    while rem_sel:
        rem_sel.sort(key=lambda c: sum((a - b) ** 2 for a, b in zip(prev_q, c["q"])))
        picked = None
        for idx, c in enumerate(rem_sel):
            if CAM.check_auto_path(fence_fn, prev_q, c["q"], steps=4):
                picked = rem_sel.pop(idx)
                break
        if picked is None:
            break
        chosen.append(picked)
        prev_q = picked["q"]

    all_T = [T0] + [c["T"] for c in chosen]
    span_m = CAM.translation_span_m(all_T)
    rot_max = CAM.rotation_span_deg(all_T)
    return chosen, {
        "tries": tries,
        "pool": len(pool),
        "rejected": rej,
        "span_m": span_m,
        "rot_max_deg": rot_max,
    }


# =========================================================================== 4. Zero-Prior Egg Holder & Slot Inference
def infer_holder_grid(L_mm, W_mm, u_pts_mm=None, v_pts_mm=None, val_pts=None):
    """V3.49 ZERO-PRIOR HOLDER: autonomously infer (rows, cols) and slot pitches (pitch_l_mm, pitch_w_mm) from the
    measured holder dimensions (L_mm >= W_mm) and interior depth/intensity cavity profile, with NO prior user input.
    -> {'rows': R, 'cols': C, 'pitch_l_mm': float, 'pitch_w_mm': float, 'score': float, 'candidates': [...]}"""
    import numpy as np
    L_mm, W_mm = float(max(L_mm, W_mm)), float(min(L_mm, W_mm))
    lo_p, hi_p = SLOT_PITCH_RANGE_MM
    cands = []
    for R in range(1, 5):
        for C in range(R, 9):
            pl = L_mm / C
            pw = W_mm / R
            if not (lo_p <= pl <= hi_p and lo_p <= pw <= hi_p):
                continue
            ratio = pl / max(1.0, pw)
            if not (0.72 <= ratio <= 1.38):
                continue
            # Geometric scale + cell squareness score (standard egg-holder cells have pitch ~44-52 mm and ratio ~1.0)
            mid_p = 0.5 * (pl + pw)
            sc_geom = math.exp(-((mid_p - 48.0) / 14.0) ** 2 - 4.5 * (ratio - 1.0) ** 2)
            # Cavity profile periodicity score (when interior depth/intensity points are available)
            sc_prof = 0.0
            if u_pts_mm is not None and v_pts_mm is not None and val_pts is not None and len(val_pts) >= 40:
                u_arr = np.asarray(u_pts_mm, dtype=float)
                v_arr = np.asarray(v_pts_mm, dtype=float)
                z_arr = np.asarray(val_pts, dtype=float)
                # Normalized cell phase in [0, 1] within each candidate cell; centre is at (0.5, 0.5)
                pu = ((u_arr + 0.5 * L_mm) / pl) % 1.0
                pv = ((v_arr + 0.5 * W_mm) / pw) % 1.0
                dist_c = np.hypot(pu - 0.5, pv - 0.5)
                in_cup = dist_c <= 0.30
                on_wall = dist_c >= 0.42
                if in_cup.sum() >= 10 and on_wall.sum() >= 10:
                    diff = abs(float(np.median(z_arr[on_wall]) - np.median(z_arr[in_cup])))
                    sc_prof = min(0.35, diff / max(1e-4, float(np.std(z_arr)) + 1e-4) * 0.25)
            cands.append({
                "rows": R,
                "cols": C,
                "pitch_l_mm": round(pl, 1),
                "pitch_w_mm": round(pw, 1),
                "score": round(sc_geom + sc_prof, 4),
            })
    if not cands:
        R = max(1, int(round(W_mm / 48.0)))
        C = max(R, int(round(L_mm / 48.0)))
        cands.append({"rows": R, "cols": C, "pitch_l_mm": round(L_mm / C, 1), "pitch_w_mm": round(W_mm / R, 1),
                      "score": 0.2})
    cands.sort(key=lambda d: -d["score"])
    best = dict(cands[0], candidates=cands[:4])
    return best


def agent_judge_holder(rgb_crop, geom_info, agent_fn=None):
    """V3.49 AGENT JUDGMENT: optional VLM / Agent hook to inspect the holder crop and confirm/override (rows, cols)
    and occupied slot indices.  When agent_fn is None, returns geom_info with source='agent_geom'."""
    if callable(agent_fn):
        try:
            verdict = agent_fn(rgb_crop, dict(geom_info))
            if isinstance(verdict, dict) and verdict.get("rows") and verdict.get("cols"):
                out = dict(geom_info)
                out["rows"] = int(verdict["rows"])
                out["cols"] = int(verdict["cols"])
                if "occupied_slots" in verdict:
                    out["occupied_slots"] = list(verdict["occupied_slots"])
                out["judgment_source"] = verdict.get("source") or "agent_vlm"
                return out
        except Exception:
            pass
    out = dict(geom_info)
    out["judgment_source"] = "agent_geom"
    return out


# --------------------------------------------------------------------------- V4.5 SLOT CAVITY REFINEMENT
EGG_CARTON_CFG = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config", "egg_carton.json")


def egg_carton_config(path=None):
    """V4.5: optional measured carton geometry (config/egg_carton.json): {"slot_pitch_mm": [along, across] or null}.
    Env NERO_EGG_SLOT_PITCH_MM="47,47" overrides the file."""
    env = os.environ.get("NERO_EGG_SLOT_PITCH_MM", "").strip()
    if env:
        try:
            a, b = [float(x) for x in env.replace(" ", "").split(",")[:2]]
            return {"slot_pitch_mm": [a, b], "source": "NERO_EGG_SLOT_PITCH_MM"}
        except Exception:
            pass
    try:
        with open(path or EGG_CARTON_CFG, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        pm = cfg.get("slot_pitch_mm")
        if pm and len(pm) >= 2 and all(isinstance(v, (int, float)) and 20.0 <= float(v) <= 90.0 for v in pm[:2]):
            return {"slot_pitch_mm": [float(pm[0]), float(pm[1])], "source": os.path.basename(path or EGG_CARTON_CFG)}
    except Exception:
        pass
    return {"slot_pitch_mm": None, "source": None}


EGG_TASK_DEFAULTS = {                                   # V4.6.1 (values measured / chosen with the user)
    "held_gap_mm": 26.5, "offcentre_margin_mm": 6.0, "offcentre_regrasp": False,
    "wrist_slot_look_before_pick": False,
    "wrist_gripper_mask": [[0.0, 0.60], [1.0, 0.60], [1.0, 1.0], [0.0, 1.0]],
    "wrist_overexposed_frac": 0.35, "wrist_overexposed_level": 245,
}


def egg_task_config(path=None):
    """V4.6.1: egg-task settings from config/egg_carton.json (read only) over EGG_TASK_DEFAULTS; bad values -> default."""
    cfg = dict(EGG_TASK_DEFAULTS)
    try:
        with open(path or EGG_CARTON_CFG, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        raw = {}
    for k, dv in EGG_TASK_DEFAULTS.items():
        v = raw.get(k, dv)
        try:
            if isinstance(dv, bool):
                v = bool(v)
            elif isinstance(dv, (int, float)):
                v = float(v)
            elif k == "wrist_gripper_mask":
                v = [[float(a), float(b)] for a, b in v] if v else []
        except Exception:
            v = dv
        cfg[k] = v
    return cfg


MARKER_V466_TEMPLATE = "V4.6.6 FIXED CARTON TEMPLATE"
MARKER_V4612_TEMPLATE = "V4.6.12 CUPS INSIDE THE OUTLINE + OCCUPIED NEEDS RIM TOP"
CARTON_DIAG_KEEP = 30
TEMPLATE_SEARCH_MM = 60.0         # V4.6.12: template centre search +- this around the outer-box centre (was 40)
TEMPLATE_MIN_PTS = 150            # V4.6.9: depth points inside the (eroded) carton mask needed for the template fit
BOWL_MIN_CORE_FRAC = 0.04         # V4.6.9: bowl score needs this share of the cup-middle cells with depth (was 0.10)
BOWL_MIN_RING_FRAC = 0.10         # V4.6.9: ... and of the rim-ring cells (was 0.30) - sparse High Accuracy depth


def carton_template_config(path=None):
    """V4.6.6: fixed carton layout from config/egg_carton.json: "carton_grid": [rows, cols] (rows = the 2-hole
    direction, cols = the 3-hole direction of a 2x3 carton) + "slot_pitch_mm": [along the cols, along the rows].
    Env NERO_EGG_CARTON_GRID="2,3" overrides the grid; "none" switches the template off.  -> dict or None (then the V3.49 zero-prior guess is used)."""
    grid = None
    env = os.environ.get("NERO_EGG_CARTON_GRID", "").strip()
    if env.lower() in ("none", "off", "0"):
        return None                                   # template switched off -> V3.49 zero-prior guess
    if env:
        try:
            grid = [int(x) for x in env.replace(" ", "").split(",")[:2]]
        except Exception:
            grid = None
    if grid is None:
        try:
            with open(path or EGG_CARTON_CFG, "r", encoding="utf-8") as f:
                grid = json.load(f).get("carton_grid")
        except Exception:
            grid = None
    try:
        R, C = int(grid[0]), int(grid[1])
    except Exception:
        return None
    if not (1 <= R <= 6 and 1 <= C <= 8):
        return None
    pm = egg_carton_config(path).get("slot_pitch_mm")
    if not pm:
        return None
    return {"rows": R, "cols": C, "pitch_cols_mm": float(pm[0]), "pitch_rows_mm": float(pm[1]),
            "source": os.path.basename(path or EGG_CARTON_CFG)}


def _disk_kernels(pitch_mm, res_mm):
    import numpy as np
    r_core, r_in, r_out = 0.24 * pitch_mm, 0.36 * pitch_mm, 0.52 * pitch_mm
    n = int(math.ceil(r_out / res_mm))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    d = np.hypot(xx, yy) * res_mm
    return (d <= r_core).astype(np.float32), ((d >= r_in) & (d <= r_out)).astype(np.float32), (r_core, r_in, r_out)


def bowl_score_map(s_mm, t_mm, h_mm, pitch_mm, res_mm=1.0, margin_mm=35.0):
    """V4.6.6: height raster of the carton top (plane frame, mm) and the BOWL score per raster cell:
    mean height of a ring at the cup rim (0.36-0.52 x pitch) minus mean height of the cup middle (<= 0.24 x pitch).
    A real egg cup: rim high, middle low -> large positive score.  A drop-off at the carton edge (one side low), the
    bare table or a cone between cups -> small / negative.  Cells without enough points -> -2.
    -> dict(S, hmap, valid, x0, y0, res)"""
    import numpy as np
    cv2 = _cv2()
    s = np.asarray(s_mm, float)
    t = np.asarray(t_mm, float)
    h = np.asarray(h_mm, float)
    x0, y0 = float(s.min()) - margin_mm, float(t.min()) - margin_mm
    nx = int((float(s.max()) + margin_mm - x0) / res_mm) + 1
    ny = int((float(t.max()) + margin_mm - y0) / res_mm) + 1
    ix = np.clip(((s - x0) / res_mm).astype(int), 0, nx - 1)
    iy = np.clip(((t - y0) / res_mm).astype(int), 0, ny - 1)
    sm = np.zeros((ny, nx), np.float64)
    cn = np.zeros((ny, nx), np.float64)
    np.add.at(sm, (iy, ix), h)
    np.add.at(cn, (iy, ix), 1.0)
    valid = (cn > 0).astype(np.float32)
    hmap = np.where(cn > 0, sm / np.maximum(cn, 1.0), 0.0).astype(np.float32)
    kc, ka, radii = _disk_kernels(pitch_mm, res_mm)
    b = cv2.BORDER_CONSTANT
    s_c = cv2.filter2D(hmap * valid, -1, kc, borderType=b)
    n_c = cv2.filter2D(valid, -1, kc, borderType=b)
    s_a = cv2.filter2D(hmap * valid, -1, ka, borderType=b)
    n_a = cv2.filter2D(valid, -1, ka, borderType=b)
    ok = (n_c >= BOWL_MIN_CORE_FRAC * kc.sum()) & (n_a >= BOWL_MIN_RING_FRAC * ka.sum())
    S = np.where(ok, np.clip(s_a / np.maximum(n_a, 1e-6) - s_c / np.maximum(n_c, 1e-6), -5.0, 25.0), -2.0)
    return {"S": S.astype(np.float32), "hmap": hmap, "valid": valid, "x0": x0, "y0": y0, "res": res_mm,
            "radii": radii}


def template_offsets(R, C, pc_mm, pr_mm):
    """slot centres of the fixed template in its own frame (u = cols direction, v = rows direction), row-major"""
    return [((c_i + 0.5 - 0.5 * C) * pc_mm, (r_i + 0.5 - 0.5 * R) * pr_mm) for r_i in range(R) for c_i in range(C)]


def footprint_ok_map(bw, foot_s_mm, foot_t_mm, clear_mm):
    """V4.6.12: raster (same grid as the bowl map) of the cells that are at least clear_mm INSIDE the carton outline.
    foot_* = every carton-mask pixel projected onto the table plane (depth or not)."""
    import numpy as np
    cv2 = _cv2()
    S = bw["S"]
    ny, nx = S.shape
    res = bw["res"]
    ix = ((np.asarray(foot_s_mm, float) - bw["x0"]) / res).astype(int)
    iy = ((np.asarray(foot_t_mm, float) - bw["y0"]) / res).astype(int)
    g = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    fp = np.zeros((ny, nx), np.uint8)
    fp[iy[g], ix[g]] = 1
    fp = cv2.morphologyEx(fp, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))          # fill pixel gaps
    r = max(1, int(round(clear_mm / res)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(fp, k).astype(bool)


def fit_carton_template(s_mm, t_mm, h_mm, R, C, pc_mm, pr_mm, phi0_rad, centre0_mm, res_mm=1.0, search_mm=40.0,
                        cav_mm=4.0, foot_st_mm=None):
    """V4.6.6 FIXED CARTON TEMPLATE: rigid R x C grid with the MEASURED pitches; only its centre (search +-search_mm
    around centre0) and direction (phi0 or phi0 + 90 deg, +-10 deg) are searched; score = sum of the bowl scores at
    the slot centres.  The grid size never changes (a 2x3 carton is never read as 3x3 / 3x4).
    -> dict(centre_mm, phi_rad, score, per_slot, n_cavities, bowl)"""
    import numpy as np
    bw = bowl_score_map(s_mm, t_mm, h_mm, min(pc_mm, pr_mm), res_mm)
    S, x0, y0 = bw["S"], bw["x0"], bw["y0"]
    ny, nx = S.shape
    inside_ok = None
    if foot_st_mm is not None and len(foot_st_mm[0]) > 0:
        # V4.6.12 CUPS INSIDE THE CARTON: a cup centre is >= CUP_EDGE_CLEAR_FRAC x pitch inside the outline (V4.6.11
        # real frame: centres just OUTSIDE the edge scored as "bowls" - carton wall on one side, the low foot / paper
        # in the middle - and the whole grid slid one pitch onto the edge)
        inside_ok = footprint_ok_map(bw, foot_st_mm[0], foot_st_mm[1], CUP_EDGE_CLEAR_FRAC * min(pc_mm, pr_mm))
        S = np.where(inside_ok, S, -5.0).astype(np.float32)
    off = np.asarray(template_offsets(R, C, pc_mm, pr_mm), float)
    g = np.arange(-search_mm, search_mm + 1e-9, res_mm)
    TX, TY = np.meshgrid(centre0_mm[0] + g, centre0_mm[1] + g)
    tx, ty = TX.ravel(), TY.ravel()
    best = (-1e9, None, None, None)
    for base in (phi0_rad, phi0_rad + math.pi / 2):
        for dphi in np.radians(np.arange(-10.0, 10.0 + 1e-9, 2.0)):
            phi = base + float(dphi)
            c, s_ = math.cos(phi), math.sin(phi)
            dx = off[:, 0] * c - off[:, 1] * s_
            dy = off[:, 0] * s_ + off[:, 1] * c
            X = tx[:, None] + dx[None, :]
            Y = ty[:, None] + dy[None, :]
            ix = np.round((X - x0) / res_mm).astype(int)
            iy = np.round((Y - y0) / res_mm).astype(int)
            inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            vals = np.full(ix.shape, -5.0)
            vals[inside] = S[iy[inside], ix[inside]]
            sc = vals.sum(axis=1)
            k = int(np.argmax(sc))
            if sc[k] > best[0]:
                best = (float(sc[k]), (float(tx[k]), float(ty[k])), phi, vals[k].tolist())
    score, centre, phi, per = best
    phi = math.atan2(math.sin(phi), math.cos(phi))
    return {"centre_mm": centre, "phi_rad": phi, "score": round(score, 2),
            "per_slot": [round(v, 1) for v in per], "n_cavities": int(sum(v >= cav_mm for v in per)), "bowl": bw,
            "inside_checked": inside_ok is not None}


CUP_EDGE_CLEAR_FRAC = 0.30      # V4.6.12: a cup centre lies at least this x pitch inside the carton outline
SLOT_EGG_TOP_TOL_MM = 3.0       # V4.6.12: see rise_says_egg
SLOT_EGG_RISE_MM = 8.0          # V4.6.8: cup middle this much ABOVE the rim ring = an egg stands in the cup


def slot_rise(pts, centre_uv_mm, pitch_mm, min_core=12):
    """V4.6.8: (u, v, h) points in carton coordinates (mm) -> {core_top_mm: 90th percentile height of the cup middle
    (r <= 0.30 x pitch), rim_mm: median height of the rim ring (0.36-0.52 x pitch), rise_mm: core_top - rim, n_core}.
    Empty cup: rise strongly negative (bowl); an egg standing in it: its top is well above the rim (rise > 0)."""
    import numpy as np
    u, v, h = pts
    r = np.hypot(u - centre_uv_mm[0], v - centre_uv_mm[1])
    core = r <= 0.30 * pitch_mm
    ring = (r >= 0.36 * pitch_mm) & (r <= 0.52 * pitch_mm)
    n_core = int(core.sum())
    if n_core < min_core or int(ring.sum()) < min_core:
        return {"core_top_mm": None, "rim_mm": None, "rise_mm": None, "n_core": n_core}
    ct = float(np.percentile(h[core], 90))
    rm = float(np.median(h[ring]))
    # V4.6.12: the middle of the cup must also reach the TOP of its rim (an egg sticks out of the cup).  Real pulp
    # carton (V4.6.11 frame): the depth inside empty cups scatters by ~28 mm between cups, an empty cup had
    # rise +11 mm (-> "occupied") while its middle stayed 20 mm below the rim top.
    cm = float(np.median(h[core]))
    rt = float(np.percentile(h[ring], 90))
    return {"core_top_mm": round(ct, 1), "rim_mm": round(rm, 1), "rise_mm": round(ct - rm, 1), "n_core": n_core,
            "core_med_mm": round(cm, 1), "rim_top_mm": round(rt, 1)}


def rise_says_egg(rise):
    """V4.6.12: occupied by depth only when the middle rose above the rim (>= SLOT_EGG_RISE_MM) AND the middle's
    median reaches the rim top (- SLOT_EGG_TOP_TOL_MM)"""
    if not rise or rise.get("rise_mm") is None or rise["rise_mm"] < SLOT_EGG_RISE_MM:
        return False
    if rise.get("core_med_mm") is None or rise.get("rim_top_mm") is None:
        return True
    return rise["core_med_mm"] >= rise["rim_top_mm"] - SLOT_EGG_TOP_TOL_MM


def save_carton_diag(rgb_crop, mask_crop, mask_inner, slot_px, slot_scores, bowl, slot_st_mm, rect_st_mm, title,
                     folder=None, keep=CARTON_DIAG_KEEP, depth_crop=None):
    """V4.6.6: one PNG per carton fit (left: colour crop + YOLO mask outline (green) + eroded mask used (cyan) +
    slots (red, number, bowl score); right: height map + bowl score circles + outer box).  -> path or None"""
    try:
        import numpy as np
        import glob
        cv2 = _cv2()
        root = folder or os.path.join(os.environ.get("NERO_CALIB_DIR") or os.path.expanduser("~/Downloads/cameracalib"),
                                      "carton_diag")
        os.makedirs(root, exist_ok=True)
        left = np.ascontiguousarray(rgb_crop[..., ::-1]).copy() if rgb_crop is not None else None
        if left is not None:
            for mk, col in ((mask_crop, (0, 200, 0)), (mask_inner, (255, 255, 0))):
                if mk is not None:
                    cs, _hh = cv2.findContours(mk.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(left, cs, -1, col, 1)
            for k, (px, py) in enumerate(slot_px):
                if px is None:
                    continue
                cv2.circle(left, (int(px), int(py)), 5, (0, 0, 255), 2)
                cv2.putText(left, f"{k + 1}", (int(px) + 6, int(py) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        if bowl is None:                               # V4.6.9: no height map -> show where the depth exists
            right = None
            if depth_crop is not None:
                dv = np.asarray(depth_crop) > 0
                right = np.zeros(dv.shape + (3,), np.uint8)
                right[dv] = (60, 200, 60)                      # green = depth, black = no depth
                if mask_crop is not None:
                    cs, _hh = cv2.findContours(mask_crop.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(right, cs, -1, (255, 255, 255), 1)
            return _save_diag_panels(left, right, title, root, keep)
        hm, val, res = bowl["hmap"], bowl["valid"], bowl["res"]
        top = float(np.percentile(hm[val > 0], 95)) if (val > 0).any() else 1.0
        hm8 = np.clip(hm / max(top, 1.0) * 255.0, 0, 255).astype(np.uint8)
        right = cv2.applyColorMap(hm8, cv2.COLORMAP_JET)
        right[val == 0] = (40, 40, 40)
        r_in = bowl["radii"][1] / res

        def px(st):
            return int(round((st[0] - bowl["x0"]) / res)), int(round((st[1] - bowl["y0"]) / res))
        if rect_st_mm is not None:
            pts = np.array([px(q) for q in rect_st_mm], np.int32)
            cv2.polylines(right, [pts], True, (255, 255, 255), 1)
        for k, st in enumerate(slot_st_mm):
            cv2.circle(right, px(st), int(r_in), (255, 255, 255), 1)
        right = cv2.flip(right, 0)                             # plane frame y up
        hh = right.shape[0]
        for k, st in enumerate(slot_st_mm):                    # V4.6.12: text AFTER the flip (was mirrored)
            cv2.putText(right, f"{k + 1}:{slot_scores[k]:+.0f}", (px(st)[0] - 12, hh - 1 - px(st)[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return _save_diag_panels(left, right, title, root, keep)
    except Exception:
        return None


def _save_diag_panels(left, right, title, root, keep):
    try:
        import numpy as np
        import glob
        cv2 = _cv2()
        H = 360
        panels = []
        for im in (left, right):
            if im is None:
                continue
            f = H / float(im.shape[0])
            panels.append(cv2.resize(im, (max(1, int(im.shape[1] * f)), H), interpolation=cv2.INTER_NEAREST))
        out = np.hstack(panels) if len(panels) > 1 else panels[0]
        bar = np.full((22, out.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, title[:140], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        out = np.vstack([bar, out])
        path = os.path.join(root, time.strftime("carton_%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}.png")
        cv2.imwrite(path, out)
        old = sorted(glob.glob(os.path.join(root, "carton_*.png")))
        for f_ in old[:-keep]:
            try:
                os.remove(f_)
            except OSError:
                pass
        return path
    except Exception:
        return None


def refine_slot_grid(u_mm, v_mm, h_mm, R, C, pl_mm, pw_mm, h_top_mm=None, min_pts=8):
    """V4.5 SLOT CAVITY REFINEMENT: the nominal grid (outer box / columns) puts the slot centres wherever the outer box
    says; real cartons have flanges and sometimes a lid inside that box. Each EMPTY slot is a depression in the
    carton's top-surface depth -> its lowest points give the true slot centre. Pass 1 finds cavities near the nominal
    grid, then the grid (offset, and the pitch when >= 2 columns / rows were seen) is fitted to them; pass 2 repeats
    the search on the fitted grid. Occupied slots (egg higher than the top) follow the fitted grid.
    u_mm / v_mm / h_mm: carton points in holder coordinates (mm, height above the table).
    -> (centres [(u, v) mm per slot, row-major], info dict)"""
    import numpy as np
    u = np.asarray(u_mm, float)
    v = np.asarray(v_mm, float)
    hh = np.asarray(h_mm, float)

    def grid(a_u, b_u, a_v, b_v):
        return [(a_u + (c_i + 0.5 - 0.5 * C) * b_u, a_v + (r_i + 0.5 - 0.5 * R) * b_v) for r_i in range(R) for c_i in range(C)]

    def find(nominal, b_u, b_v, accept_frac):
        own = {}
        for k, (uc, vc) in enumerate(nominal):
            win = (np.abs(u - uc) <= 0.45 * b_u) & (np.abs(v - vc) <= 0.45 * b_v)
            if int(win.sum()) < 3 * min_pts:
                continue
            hw = hh[win]
            top = float(np.percentile(hw, 85))
            if h_top_mm is not None and top > float(h_top_mm) + 9.0:
                continue                                         # an egg sits here: no cavity to see
            hmin = float(np.percentile(hw, 2))
            if top - hmin < 6.0:
                continue                                         # no clear depression (flat / no depth)
            low = hw <= hmin + 0.4 * (top - hmin)                # bottom of the bowl only (not neighbouring rims)
            if int(low.sum()) < min_pts:
                continue                                         # no clear depression (flat / no depth)
            cu, cv = float(np.mean(u[win][low])), float(np.mean(v[win][low]))
            if math.hypot(cu - uc, cv - vc) <= accept_frac * min(b_u, b_v):
                own[k] = (cu, cv)
        return own

    def fit(own, b_u, b_v):
        cols = np.array([k % C for k in own], float)
        rows = np.array([k // C for k in own], float)
        cu = np.array([own[k][0] for k in own])
        cv = np.array([own[k][1] for k in own])
        xc, xr = cols + 0.5 - 0.5 * C, rows + 0.5 - 0.5 * R
        out = []
        for x, c, n, b0, nominal_b in ((xc, cu, len(set(cols.tolist())), b_u, pl_mm), (xr, cv, len(set(rows.tolist())), b_v, pw_mm)):
            a, b = float(np.median(c - x * b0)), b0
            if n >= 2:
                b_fit, a_fit = np.polyfit(x, c, 1)
                if 0.6 * nominal_b <= b_fit <= 1.4 * nominal_b and 30.0 <= b_fit <= 75.0:   # egg-carton pitch range
                    a, b = float(a_fit), float(b_fit)
            out += [a, b]
        a_u, b_u, a_v, b_v = out
        if len(own) == 1:
            a_u, a_v = float(np.clip(a_u, -8.0, 8.0)), float(np.clip(a_v, -8.0, 8.0))
        return a_u, b_u, a_v, b_v

    a_u, b_u, a_v, b_v = 0.0, float(pl_mm), 0.0, float(pw_mm)
    nominal0 = grid(a_u, b_u, a_v, b_v)
    own = find(nominal0, b_u, b_v, 0.45)
    info = {"n_cavities": len(own), "global_shift_mm": [0.0, 0.0], "pitch_mm": [round(b_u, 1), round(b_v, 1)],
            "per_slot_shift_mm": {}}
    if not own:
        info["note"] = "未找到槽穴凹陷，槽位按外框等分"
        return nominal0, info
    for _pass in range(5):                                       # fit -> re-detect until the grid settles
        a_u, b_u, a_v, b_v = fit(own, b_u, b_v)
        own2 = find(grid(a_u, b_u, a_v, b_v), b_u, b_v, 0.30)
        if not own2:
            break
        moved = max(math.hypot(own2[k][0] - own.get(k, own2[k])[0], own2[k][1] - own.get(k, own2[k])[1]) for k in own2)
        own = own2
        if moved < 0.2 and _pass >= 1:
            break
    a_u, b_u, a_v, b_v = fit(own, b_u, b_v)
    fitted = grid(a_u, b_u, a_v, b_v)
    out = []
    for k, (uc, vc) in enumerate(fitted):
        if k in own:
            uc += float(np.clip(own[k][0] - uc, -4.0, 4.0))
            vc += float(np.clip(own[k][1] - vc, -4.0, 4.0))
        out.append((uc, vc))
        info["per_slot_shift_mm"][k + 1] = [round(uc - nominal0[k][0], 1), round(vc - nominal0[k][1], 1)]
    info.update(n_cavities=len(own), global_shift_mm=[round(a_u, 1), round(a_v, 1)], pitch_mm=[round(b_u, 1), round(b_v, 1)])
    info["note"] = (f"槽位按 {len(own)} 个槽穴凹陷拟合：整体偏移 ({a_u:+.1f}, {a_v:+.1f}) mm，槽距 {b_u:.1f}×{b_v:.1f} mm")
    return out, info


def locate_egg_holder(dets2d, rgb, depth_m, K, T_base_cam, table_z=None, fence_poly=None, calib_note=None,
                      existing_eggs=None, agent_fn=None):
    """V3.49 ZERO-PRIOR HOLDER + SLOTS: given 2D YOLOE holder detections + RGB-D + T_base_cam:
    1. Fits local 3-6 cm table plane around the holder;
    2. Computes the 3D OBB of the holder (centre, L, W, H, yaw) with NO prior size input;
    3. Autonomously infers (rows x cols) and slot coordinates slot_1..slot_N + empty/occupied status;
    4. Optionally calls agent_judge_holder() for VLM/Agent visual confirmation.
    -> (holder_and_slot_rows, diag)"""
    import numpy as np
    import NERO_V347_Yolo_Perception as YP
    cv2 = _cv2()
    T = np.asarray(T_base_cam, dtype=float)
    h, w = depth_m.shape
    cands = [d for d in (dets2d or []) if d.get("cls", 0) == 0 and int(d["mask"].sum()) >= 120]
    if not cands:
        return [], {"n_2d": len(dets2d or []), "found": False, "reason": "未检测到蛋托轮廓"}
    cands.sort(key=lambda d: -float(d["conf"]) * math.sqrt(float(d["mask"].sum())))
    d = cands[0]
    m = d["mask"]
    bb = YP._mask_bbox(m)
    x0, y0, x1, y1 = d["box"]
    t_start = time.time()

    pad0 = 28
    cy0, cy1 = max(0, bb[1] - pad0), min(h, bb[3] + pad0 + 1)
    cx0, cx1 = max(0, bb[0] - pad0), min(w, bb[2] + pad0 + 1)
    mc = m[cy0:cy1, cx0:cx1]
    dc = depth_m[cy0:cy1, cx0:cx1]
    ring0 = cv2.dilate(mc.astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool) & ~mc & (dc > 0)
    zc = dc[ring0]
    z_in = dc[mc & (dc > 0)]
    z_cam = float(np.median(zc)) if zc.size >= 20 else (float(np.median(z_in)) if z_in.size >= 20 else 0.8)

    plane, n_band = YP.local_plane(depth_m, K, T, m, None, z_cam, bbox=bb)
    psrc = "局部桌面平面"
    if plane is None:
        if table_z is None:
            return [], {"n_2d": len(dets2d), "found": False, "reason": f"测不到蛋托周围桌面平面（有效点 {n_band}）"}
        plane, psrc = (np.array([0.0, 0.0, 1.0]), float(table_z), None, 0), "安全设置 z_table"

    n_vec, d_pl = np.asarray(plane[0], float), float(plane[1])
    ex = np.array([1.0, 0.0, 0.0]) - n_vec[0] * n_vec
    e1 = ex / np.linalg.norm(ex)
    e2 = np.cross(n_vec, e1)
    o_pl = n_vec * d_pl

    vv_m, uu_m = np.nonzero(mc)
    dz_m = dc[vv_m, uu_m]
    vv_m, uu_m = vv_m + cy0, uu_m + cx0
    cov = float((dz_m > 0).mean()) if len(dz_m) else 0.0

    if cov >= 0.35 and (dz_m > 0).sum() >= 60:
        ok = dz_m > 0
        P_all = YP._pix_to_base(uu_m[ok].astype(float), vv_m[ok].astype(float), dz_m[ok].astype(float), K, T)
        hs = P_all @ n_vec - d_pl
        h_top_m = max(0.015, min(0.090, float(np.percentile(hs, 90))))
        # Use upper-half points of the holder to avoid side-wall grazing stretch on oblique views
        h_cut = np.percentile(hs, 40)
        P_top = P_all[hs >= h_cut]
    else:
        h_top_m = 0.036
        # Ray-plane intersection at mid-height
        dc_rays = np.stack([(uu_m - K["ppx"]) / K["fx"], (vv_m - K["ppy"]) / K["fy"], np.ones(len(uu_m))], axis=1)
        D_rays = dc_rays @ T[:3, :3].T
        O_cam = T[:3, 3]
        lam = (d_pl + 0.5 * h_top_m - float(n_vec @ O_cam)) / (D_rays @ n_vec)
        P_all = O_cam + lam[:, None] * D_rays
        P_top = P_all
        hs = np.full(len(P_all), 0.5 * h_top_m)

    rel_top = P_top - o_pl
    st_top = np.stack([rel_top @ e1, rel_top @ e2], axis=1).astype(np.float32) * 1000.0
    rect = cv2.minAreaRect(st_top)
    (sc_mm, tc_mm), (w_r, h_r), ang_deg = rect
    if w_r >= h_r:
        L_mm, W_mm = float(w_r), float(h_r)
        yaw_rad = math.radians(ang_deg)
    else:
        L_mm, W_mm = float(h_r), float(w_r)
        yaw_rad = math.radians(ang_deg + 90.0)
    # Normalize yaw to [-pi/2, pi/2]
    while yaw_rad > math.pi / 2:
        yaw_rad -= math.pi
    while yaw_rad <= -math.pi / 2:
        yaw_rad += math.pi

    u_long = math.cos(yaw_rad) * e1 + math.sin(yaw_rad) * e2
    v_short = -math.sin(yaw_rad) * e1 + math.cos(yaw_rad) * e2
    centre_pl = o_pl + (sc_mm / 1000.0) * e1 + (tc_mm / 1000.0) * e2
    holder_xyz = centre_pl + (0.65 * h_top_m) * n_vec

    # Project all interior points to holder local (u, v) in mm for cavity analysis
    rel_all = P_all - centre_pl
    u_all_mm = (rel_all @ u_long) * 1000.0
    v_all_mm = (rel_all @ v_short) * 1000.0
    if rgb is not None and not (cov >= 0.35 and (dz_m > 0).sum() >= 60):
        gray_c = cv2.cvtColor(rgb[cy0:cy1, cx0:cx1], cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb[cy0:cy1, cx0:cx1]
        val_prof = gray_c[mc].astype(float)
    else:
        val_prof = hs * 1000.0

    crop_rgb = rgb[cy0:cy1, cx0:cx1] if rgb is not None else None
    # V4.6.6 FIXED CARTON TEMPLATE (config/egg_carton.json carton_grid + slot_pitch_mm): the grid is never guessed
    tcfg = carton_template_config()
    tmpl, tmpl_info = None, None
    if tcfg is not None:
        R, C = tcfg["rows"], tcfg["cols"]
        pc_mm, pr_mm = tcfg["pitch_cols_mm"], tcfg["pitch_rows_mm"]
        grid_info = {"rows": R, "cols": C, "score": 0.9, "judgment_source": "template"}
        mc_er = cv2.erode(mc.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        inner = np.zeros(0, bool)
        ok_d = dz_m > 0
        if int(ok_d.sum()) >= TEMPLATE_MIN_PTS:
            # V4.6.9: the template uses every pixel that HAS depth (even when < 35 % of the mask has some - High
            # Accuracy depth leaves holes; evenly spread points still show the cups)
            P_d = YP._pix_to_base(uu_m[ok_d].astype(float), vv_m[ok_d].astype(float), dz_m[ok_d].astype(float), K, T)
            hs_d = P_d @ n_vec - d_pl
            inner = mc_er[vv_m[ok_d] - cy0, uu_m[ok_d] - cx0]
            rel_t = P_d - o_pl
            s_pl, t_pl = (rel_t @ e1) * 1000.0, (rel_t @ e2) * 1000.0
            if int(inner.sum()) >= TEMPLATE_MIN_PTS:
                # V4.6.12: the carton outline on the table plane (every mask pixel, at half the carton height)
                rays = np.stack([(uu_m - K["ppx"]) / K["fx"], (vv_m - K["ppy"]) / K["fy"], np.ones(len(uu_m))], axis=1)
                Dr = rays @ T[:3, :3].T
                lam_f = (d_pl + 0.5 * h_top_m - float(n_vec @ T[:3, 3])) / (Dr @ n_vec)
                Pf = T[:3, 3] + lam_f[:, None] * Dr - o_pl
                foot = ((Pf @ e1) * 1000.0, (Pf @ e2) * 1000.0)
                tmpl = fit_carton_template(s_pl[inner], t_pl[inner], hs_d[inner] * 1000.0, R, C, pc_mm, pr_mm,
                                           yaw_rad, (float(sc_mm), float(tc_mm)), foot_st_mm=foot,
                                           search_mm=TEMPLATE_SEARCH_MM)
        if tmpl is not None and tmpl["n_cavities"] >= 2:
            phi = tmpl["phi_rad"]
            s0, t0 = tmpl["centre_mm"]
            how = f"槽穴拟合（{tmpl['n_cavities']}/{R * C} 个碗形槽穴，分数 {tmpl['score']:.0f}）"
        else:
            phi, s0, t0 = yaw_rad, float(sc_mm), float(tc_mm)
            how = (f"槽穴不清楚（蛋托轮廓深度覆盖 {cov * 100:.0f}%）→ 模板按外框中心 / 长边放置" if tmpl is not None else
                   f"深度点不够（蛋托轮廓里只有 {cov * 100:.0f}% 的像素有深度，内缩后 {int(inner.sum())} 点 < {TEMPLATE_MIN_PTS}）"
                   " → 模板按外框中心 / 长边放置")
        u_long = math.cos(phi) * e1 + math.sin(phi) * e2
        v_short = -math.sin(phi) * e1 + math.cos(phi) * e2
        centre_pl = o_pl + (s0 / 1000.0) * e1 + (t0 / 1000.0) * e2
        # stable numbering: columns (3-hole direction) increase away from the robot base
        if float(np.dot(u_long[:2], centre_pl[:2])) < 0.0:
            u_long, v_short = -u_long, -v_short
        yaw_rad = math.atan2(float(u_long @ e2), float(u_long @ e1))
        holder_xyz = centre_pl + (0.65 * h_top_m) * n_vec
        rel_all = P_all - centre_pl
        u_all_mm = (rel_all @ u_long) * 1000.0
        v_all_mm = (rel_all @ v_short) * 1000.0
        pl_m, pw_m = pc_mm / 1000.0, pr_mm / 1000.0
        tmpl_info = {"rows": R, "cols": C, "pitch_mm": [pc_mm, pr_mm], "how": how,
                     "depth_cov": round(float(cov), 3),                    # V4.6.9: mask pixels with depth
                     "inner_pts": int(inner.sum()),
                     "n_cavities": None if tmpl is None else tmpl["n_cavities"],
                     "score": None if tmpl is None else tmpl["score"],
                     "bowl_per_slot_mm": None if tmpl is None else tmpl["per_slot"],
                     "outer_box_mm": [round(L_mm, 1), round(W_mm, 1)],
                     "centre_vs_box_mm": [round(s0 - float(sc_mm), 1), round(t0 - float(tc_mm), 1)]}
    else:
        grid_info = infer_holder_grid(L_mm, W_mm, u_all_mm, v_all_mm, val_prof)
        grid_info = agent_judge_holder(crop_rgb, dict(grid_info, L_mm=round(L_mm, 1), W_mm=round(W_mm, 1),
                                                      H_mm=round(h_top_m * 1000.0, 1)), agent_fn=agent_fn)
        R, C = int(grid_info["rows"]), int(grid_info["cols"])
        pl_m = (L_mm / 1000.0) / C
        pw_m = (W_mm / 1000.0) / R
    # V4.5: measured slot pitch (config/egg_carton.json) replaces "outer box / columns" (flanges, lid)
    ccfg = egg_carton_config()
    pitch_src = "外框等分"
    lid_warn = None
    if tmpl_info is not None:
        slot_uv_mm = template_offsets(R, C, pl_m * 1000.0, pw_m * 1000.0)
        refine_info = {"n_cavities": tmpl_info["n_cavities"] or 0, "note": tmpl_info["how"], "template": tmpl_info}
        pitch_src = f"固定模板 {R}×{C}（{tcfg['source']}）"
    elif ccfg.get("slot_pitch_mm"):
        pl_m, pw_m = ccfg["slot_pitch_mm"][0] / 1000.0, ccfg["slot_pitch_mm"][1] / 1000.0
        pitch_src = f"实测槽距({ccfg['source']})"
        extra = L_mm - C * pl_m * 1000.0
        if extra > 40.0:
            lid_warn = f"外框比 槽距×列数 大 {extra:.0f} mm（盖子或边沿被算进外框？槽位改由槽穴凹陷定位）"
    if tmpl_info is not None:
        pass
    elif cov >= 0.35 and (dz_m > 0).sum() >= 60:
        slot_uv_mm, refine_info = refine_slot_grid(u_all_mm, v_all_mm, hs * 1000.0, R, C, pl_m * 1000.0, pw_m * 1000.0,
                                                   h_top_mm=h_top_m * 1000.0)
    else:
        slot_uv_mm = [((c_i + 0.5 - 0.5 * C) * pl_m * 1000.0, (r_i + 0.5 - 0.5 * R) * pw_m * 1000.0)
                      for r_i in range(R) for c_i in range(C)]
        refine_info = {"n_cavities": 0, "note": "蛋托深度覆盖不足，槽位按外框等分"}

    edge = x0 <= YP.EDGE_PX or y0 <= YP.EDGE_PX or x1 >= w - 1 - YP.EDGE_PX or y1 >= h - 1 - YP.EDGE_PX
    warn = []
    if edge:
        warn.append("蛋托贴画面边缘")
    if d["conf"] < 0.35:
        warn.append(f"YOLO 置信度低 {d['conf']:.2f}")
    if fence_poly and not YP.point_in_poly(float(holder_xyz[0]), float(holder_xyz[1]), fence_poly):
        warn.append("在安全区外")
    if psrc != "局部桌面平面":
        warn.append("周围桌面无深度，用安全设置 z_table")
    if calib_note:
        warn.append(calib_note)
    trusted = not warn

    jsrc_label = {"agent_vlm": "Agent+视觉", "template": "固定蛋托模板"}.get(grid_info.get("judgment_source"),
                                                                       "Agent几何自主推断")
    holder_row = {
        "det_id": "holder",
        "status": "object" if trusted else "untrusted",
        "trusted": trusted,
        "confidence": round(float(d["conf"]) * grid_info.get("score", 0.85), 3),
        "xyz_m": [round(float(v), 5) for v in holder_xyz],
        "axis_world": [round(float(v), 5) for v in u_long],
        "yaw_deg": round(math.degrees(yaw_rad), 1),
        "size_mm": [round(L_mm, 1), round(W_mm, 1)],
        "height_mm": round(h_top_m * 1000.0, 1),
        "rows": R,
        "cols": C,
        "pitch_mm": [round(pl_m * 1000.0, 1), round(pw_m * 1000.0, 1)],
        "yolo": {"conf": round(float(d["conf"]), 3), "box": [round(v) for v in d["box"]], "cls": 0, "name": "holder"},
        "reasons": list(warn),
        "notes": [f"{jsrc_label}：{R}×{C}（{R * C} 槽），实测外框 {L_mm:.0f}×{W_mm:.0f}×{h_top_m * 1000:.0f} mm，"
                  f"槽距 {pl_m * 1000:.0f}×{pw_m * 1000:.0f} mm（{pitch_src}），偏航 {math.degrees(yaw_rad):+.0f}°；"
                  f"{refine_info.get('note')}" + (f"；⚠ {lid_warn}" if lid_warn else "")],
        "slot_refine": refine_info,
    }

    # V4.6.8 SLOT OCCUPANCY BY RISE: all valid depth pixels of the crop (not only the carton mask - an egg
    # standing in a cup is often NOT part of the YOLO carton mask) -> per slot: top of the cup middle vs the rim ring
    crop_pts = None
    if tmpl_info is not None:
        try:
            vv_c, uu_c = np.nonzero(dc > 0)
            zz_c = dc[vv_c, uu_c]
            if len(zz_c) >= 60:
                P_c = YP._pix_to_base((uu_c + cx0).astype(float), (vv_c + cy0).astype(float), zz_c.astype(float), K, T)
                rel_c = P_c - centre_pl
                crop_pts = ((rel_c @ u_long) * 1000.0, (rel_c @ v_short) * 1000.0, (P_c @ n_vec - d_pl) * 1000.0)
        except Exception:
            crop_pts = None
    slots = []
    occ_override = set(grid_info.get("occupied_slots") or [])
    egg_pts = [e["xyz_m"][:2] for e in (existing_eggs or []) if e.get("xyz_m") and e.get("status") in ("egg", "untrusted")]
    slot_idx = 0
    for r_i in range(R):
        for c_i in range(C):
            slot_idx += 1
            sid = f"slot_{slot_idx}"
            uc_m = slot_uv_mm[slot_idx - 1][0] / 1000.0   # V4.5: cavity-refined slot centre
            vr_m = slot_uv_mm[slot_idx - 1][1] / 1000.0
            s_xyz = centre_pl + uc_m * u_long + vr_m * v_short + (0.60 * h_top_m) * n_vec
            # Determine occupied vs empty from height profile or known eggs or Agent override
            in_cell = (np.abs(u_all_mm - uc_m * 1000.0) <= 0.32 * pl_m * 1000.0) & \
                      (np.abs(v_all_mm - vr_m * 1000.0) <= 0.32 * pw_m * 1000.0)
            cell_h_mm = float(np.percentile(hs[in_cell], 85) * 1000.0) if (cov >= 0.35 and in_cell.sum() >= 8) else (h_top_m * 600.0)
            has_egg_near = any(math.hypot(s_xyz[0] - ex_, s_xyz[1] - ey_) < 0.42 * min(pl_m, pw_m) for ex_, ey_ in egg_pts)
            occ = (slot_idx in occ_override) or (sid in occ_override) or has_egg_near or (cell_h_mm > h_top_m * 1000.0 + 9.0)
            rise = slot_rise(crop_pts, slot_uv_mm[slot_idx - 1], min(pl_m, pw_m) * 1000.0) if crop_pts is not None else None
            if rise_says_egg(rise):
                occ = True
            uv = CAM.project(CAM.T_apply(CAM.T_inv(T.tolist()), s_xyz.tolist()), K)
            bx = [round(uv[0] - 14), round(uv[1] - 14), round(uv[0] + 14), round(uv[1] + 14)] if uv else [round(v) for v in d["box"]]
            slots.append({
                "det_id": sid,
                "slot_index": slot_idx,
                "row": r_i + 1,
                "col": c_i + 1,
                "occupied": bool(occ),
                "status": "egg" if trusted else "untrusted",
                "trusted": trusted,
                "confidence": holder_row["confidence"],
                "xyz_m": [round(float(v), 5) for v in s_xyz],
                "height_mm": round(cell_h_mm, 1),
                "rise": rise if crop_pts is not None else None,
                "yolo": {"conf": round(float(d["conf"]), 3), "box": bx, "cls": 0, "name": sid},
                "reasons": list(warn),
                "notes": [f"第 {r_i + 1} 行第 {c_i + 1} 列 · {'有蛋' if occ else '空槽'}"],
            })

    if tmpl_info is not None:                          # V4.6.6: diagnostic picture of every carton fit
        try:
            slot_px = []
            for s_ in slots:
                uv_ = CAM.project(CAM.T_apply(CAM.T_inv(T.tolist()), s_["xyz_m"]), K)
                slot_px.append((uv_[0] - cx0, uv_[1] - cy0) if uv_ else (None, None))
            st_mm = []
            for (uo, vo) in slot_uv_mm:
                p3 = centre_pl + (uo / 1000.0) * u_long + (vo / 1000.0) * v_short - o_pl
                st_mm.append((float(p3 @ e1) * 1000.0, float(p3 @ e2) * 1000.0))
            box = cv2.boxPoints(rect)
            bowl = tmpl["bowl"] if tmpl is not None else None
            # V4.6.9: a picture also when the template fell back to the outer box (right panel = where depth exists)
            tmpl_info["diag_png"] = save_carton_diag(
                crop_rgb, mc, mc_er, slot_px, tmpl["per_slot"] if tmpl is not None else [0.0] * (R * C), bowl, st_mm,
                [tuple(q) for q in box],
                f"V4.6.12 carton {R}x{C} pitch {pl_m * 1000:.0f}x{pw_m * 1000:.0f} mm | depth {cov * 100:.0f}% | "
                f"{tmpl_info['how']}", depth_crop=dc)
        except Exception as exc:
            tmpl_info["diag_error"] = str(exc)[:120]
        holder_row["template"] = tmpl_info
        holder_row["yaw_deg"] = round(math.degrees(yaw_rad), 1)
    fit_ms = round((time.time() - t_start) * 1000.0)
    diag = {
        "n_2d": len(dets2d),
        "found": True,
        "rows": R,
        "cols": C,
        "n_slots": len(slots),
        "n_empty": sum(1 for s_ in slots if not s_["occupied"]),
        "n_occupied": sum(1 for s_ in slots if s_["occupied"]),
        "size_mm": holder_row["size_mm"],
        "height_mm": holder_row["height_mm"],
        "pitch_mm": holder_row["pitch_mm"],
        "yaw_deg": holder_row["yaw_deg"],
        "judgment_source": grid_info.get("judgment_source", "agent_geom"),
        "fit_ms": fit_ms,
        "template": tmpl_info,
    }
    return [holder_row] + slots, diag


# =========================================================================== 5. V3.50 Wrist Quality Gate, TCP Crosshair & Coarse-to-Fine
def evaluate_wrist_frame(rgb, prev_rgb=None, q_deg=None, prev_q_deg=None,
                         motion_max=WRIST_MOTION_MAX_DIFF, still_deg=WRIST_STILL_DEG,
                         sharp_min=WRIST_SHARP_MIN, overexp_max=WRIST_OVEREXP_MAX):
    """V3.50 WRIST QUALITY GATE: evaluate rolling-shutter motion stability, joint velocity, Laplacian sharpness,
    and highlight saturation on a wrist UVC frame before triggering visual perception.
    -> {'settled': bool, 'status': 'ok'|'motion'|'overexp'|'blur'|'no_frame', 'badge': str,
        'motion_diff': float, 'joint_step_deg': float, 'sharpness': float, 'overexp_frac': float}"""
    if rgb is None:
        return {
            "settled": False,
            "status": "no_frame",
            "badge": "无画面",
            "motion_diff": 0.0,
            "joint_step_deg": 0.0,
            "sharpness": 0.0,
            "overexp_frac": 0.0,
        }
    cv2 = _cv2()
    import numpy as np
    arr = np.ascontiguousarray(rgb)
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY) if arr.ndim == 3 else arr
    # Central 2/3 ROI for sharpness and overexposure evaluation
    y0, y1 = h // 6, max(h // 6 + 1, (5 * h) // 6)
    x0, x1 = w // 6, max(w // 6 + 1, (5 * w) // 6)
    roi = gray[y0:y1, x0:x1]
    cy, cx = roi.shape[0] // 2, roi.shape[1] // 2
    s_patch = roi[max(0, cy - 120):cy + 120, max(0, cx - 180):cx + 180]
    sharp = float(cv2.Laplacian(s_patch, cv2.CV_64F).var())
    overexp = float((roi > 248).mean()) if roi.size else 0.0

    motion_diff = 0.0
    if prev_rgb is not None and getattr(prev_rgb, "shape", None) == arr.shape:
        p_arr = np.ascontiguousarray(prev_rgb)
        p_gray = cv2.cvtColor(p_arr, cv2.COLOR_RGB2GRAY) if p_arr.ndim == 3 else p_arr
        g1 = gray[::4, ::4].astype(np.float32)
        g0 = p_gray[::4, ::4].astype(np.float32)
        motion_diff = float(np.mean(np.abs(g1 - g0)))

    j_step = 0.0
    if q_deg is not None and prev_q_deg is not None and len(q_deg) == len(prev_q_deg):
        j_step = float(max((abs(float(a) - float(b)) for a, b in zip(q_deg, prev_q_deg)), default=0.0))

    if j_step > still_deg or motion_diff > motion_max:
        status = "motion"
        badge = f"运动模糊 ⚠ (ΔI={motion_diff:.1f}, Δq={j_step:.2f}°)"
    elif overexp > overexp_max:
        status = "overexp"
        badge = f"反光过曝 ⚠ ({overexp * 100:.0f}%)"
    elif sharp < sharp_min:
        status = "blur"
        badge = f"失焦模糊 ⚠ (清晰度 {sharp:.0f})"
    else:
        status = "ok"
        badge = f"静止可采 ✓ (清晰度 {sharp:.0f})"

    return {
        "settled": status == "ok",
        "status": status,
        "badge": badge,
        "motion_diff": round(motion_diff, 2),
        "joint_step_deg": round(j_step, 3),
        "sharpness": round(sharp, 1),
        "overexp_frac": round(overexp, 3),
    }


def project_tcp_wrist(K, T_flange_cam, tool_T, z_work_m=0.12, finger_half_gap_m=0.028,
                      finger_box_m=(0.022, 0.016), canvas_size=None):
    """V3.50 TCP & GRIPPER FOOTPRINT PROJECTION: project the active TCP crosshair, 21 mm egg-radius reference ring,
    gripper opening axis, and left/right fingertip drop-zone boxes into the wrist camera image at working depth z_work_m.
    -> dict with full-res pixel coords ('uv_tcp', 'radius_21mm_px', 'left_box_px', 'right_box_px', 'axis_line_px')
       and optional canvas-scaled coords when canvas_size=(cw, ch) is provided."""
    if not K or not T_flange_cam or not tool_T:
        return None
    try:
        T_cam_flange = CAM.T_inv(T_flange_cam)
        T_cam_tcp = CAM.mat_mul(T_cam_flange, tool_T)
        p_tcp_cam = CAM.T_t(T_cam_tcp)
        R_ct = CAM.T_R(T_cam_tcp)
        # Working plane depth in camera optical frame (z forward):
        # if the physical TCP is in front of the camera lens (>= 4 cm), use max(p_tcp_cam[2], z_work_m);
        # otherwise project the point directly under the TCP at working distance z_work_m.
        z_ref = max(0.06, float(z_work_m), float(p_tcp_cam[2]) if p_tcp_cam[2] >= 0.04 else 0.0)
        p_ref_cam = [float(p_tcp_cam[0]), float(p_tcp_cam[1]), z_ref]
        uv_tcp = CAM.project(p_ref_cam, K)
        if not uv_tcp:
            return None
        # Gripper finger opening direction in camera XY plane (try TCP Y axis first, then X axis)
        dx, dy = float(R_ct[0][1]), float(R_ct[1][1])
        norm_xy = math.hypot(dx, dy)
        if norm_xy < 0.25:
            dx, dy = float(R_ct[0][0]), float(R_ct[1][0])
            norm_xy = math.hypot(dx, dy)
        if norm_xy < 1e-6:
            dx, dy = 1.0, 0.0
        else:
            dx, dy = dx / norm_xy, dy / norm_xy

        px_per_m = 0.5 * (float(K["fx"]) + float(K["fy"])) / z_ref
        r_egg_px = 0.021 * px_per_m
        p_left = [p_ref_cam[0] - finger_half_gap_m * dx, p_ref_cam[1] - finger_half_gap_m * dy, z_ref]
        p_right = [p_ref_cam[0] + finger_half_gap_m * dx, p_ref_cam[1] + finger_half_gap_m * dy, z_ref]
        uv_l = CAM.project(p_left, K) or [uv_tcp[0] - finger_half_gap_m * px_per_m * dx,
                                          uv_tcp[1] - finger_half_gap_m * px_per_m * dy]
        uv_r = CAM.project(p_right, K) or [uv_tcp[0] + finger_half_gap_m * px_per_m * dx,
                                           uv_tcp[1] + finger_half_gap_m * px_per_m * dy]
        hw_px = 0.5 * float(finger_box_m[0]) * px_per_m
        hh_px = 0.5 * float(finger_box_m[1]) * px_per_m
        left_box = [uv_l[0] - hw_px, uv_l[1] - hh_px, uv_l[0] + hw_px, uv_l[1] + hh_px]
        right_box = [uv_r[0] - hw_px, uv_r[1] - hh_px, uv_r[0] + hw_px, uv_r[1] + hh_px]
        axis_line = [uv_l[0], uv_l[1], uv_r[0], uv_r[1]]

        out = {
            "ok": True,
            "uv_tcp": [round(uv_tcp[0], 1), round(uv_tcp[1], 1)],
            "z_ref_m": round(z_ref, 4),
            "px_per_mm": round(px_per_m / 1000.0, 3),
            "radius_21mm_px": round(r_egg_px, 1),
            "open_dir_2d": [round(dx, 4), round(dy, 4)],
            "left_box_px": [round(v, 1) for v in left_box],
            "right_box_px": [round(v, 1) for v in right_box],
            "axis_line_px": [round(v, 1) for v in axis_line],
        }
        if canvas_size:
            cw, ch = float(canvas_size[0]), float(canvas_size[1])
            sx = cw / max(1.0, float(K.get("width") or 1280.0))
            sy = ch / max(1.0, float(K.get("height") or 720.0))
            s_mean = 0.5 * (sx + sy)
            out["uv_tcp_cv"] = [round(uv_tcp[0] * sx, 1), round(uv_tcp[1] * sy, 1)]
            out["radius_21mm_cv"] = round(r_egg_px * s_mean, 1)
            out["left_box_cv"] = [round(left_box[0] * sx, 1), round(left_box[1] * sy, 1),
                                  round(left_box[2] * sx, 1), round(left_box[3] * sy, 1)]
            out["right_box_cv"] = [round(right_box[0] * sx, 1), round(right_box[1] * sy, 1),
                                   round(right_box[2] * sx, 1), round(right_box[3] * sy, 1)]
            out["axis_line_cv"] = [round(axis_line[0] * sx, 1), round(axis_line[1] * sy, 1),
                                   round(axis_line[2] * sx, 1), round(axis_line[3] * sy, 1)]
        return out
    except Exception:
        return None


def _segment_bright_contours(rgb, min_area=120):
    """Fallback 2D contour extractor when YOLOE is not passed (used for pure OpenCV unit tests or high-contrast tabletop)."""
    cv2 = _cv2()
    import numpy as np
    gray = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    med = float(np.median(gray))
    thr_val = max(60, min(230, int(med + 35)))
    _, bw = cv2.threshold(gray, thr_val, 255, cv2.THRESH_BINARY)
    cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dets = []
    h, w = gray.shape[:2]
    for c in cs:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        m = np.zeros((h, w), dtype=bool)
        cv2.drawContours(m.view(np.uint8), [c], -1, 1, -1)
        x, y, bw_, bh_ = cv2.boundingRect(c)
        dets.append({
            "box": [float(x), float(y), float(x + bw_), float(y + bh_)],
            "conf": 0.90,
            "cls": 0,
            "name": "egg",
            "mask": m,
            "contour": c.reshape(-1, 2).astype(float),
        })
    return dets


def wrist_refine_target(rgb, K, T_base_flange, calib_eih, tool_T, coarse_item,
                        dets2d=None, det_yolo=None, mode="pf", model_path=None, prompts=None,
                        conf=0.25, ref=None, table_z=0.0, egg_size_mm=(55.0, 42.0),
                        undistort_intr=None, max_coarse_diff_mm=WRIST_MAX_COARSE_DIFF_MM,
                        finger_half_gap_m=0.028, finger_box_m=(0.022, 0.016)):
    """V3.50 COARSE-TO-FINE MONOCULAR REFINEMENT:
    Given the wrist camera RGB frame at arm pose T_base_flange, calibrated wrist intrinsics K,
    eye-in-hand calibration calib_eih (T_flange_cam), active TCP tool_T, and the coarse target item
    from D435i (coarse_item with 'xyz_m' and 'det_id'):
      1. Undistorts the frame if undistort_intr is provided;
      2. Runs 2D instance segmentation (or uses dets2d);
      3. Projects 2D silhouette rays onto the horizontal target plane Z = z_target in base_link
         using T_base_wrist = T_base_flange @ T_flange_cam;
      4. Solves refined 3D centre fine_xyz_m, physical size_mm (corrected for perspective cone),
         and principal axis yaw fine_yaw_deg in base_link;
      5. Computes horizontal offset relative to current TCP delta_tcp_mm = (dx, dy) mm,
         yaw offset delta_yaw_deg, and discrepancy vs coarse pose delta_coarse_mm;
      6. Checks fingertip drop-zone clearance against neighbouring object masks.
    -> result dict with 'ok', 'trusted', 'fine_xyz_m', 'delta_tcp_mm', 'delta_coarse_mm',
       'fine_yaw_deg', 'delta_yaw_deg', 'clearance_ok', 'uv_fine', 'contour_px', 'warnings'."""
    cv2 = _cv2()
    import numpy as np
    if rgb is None:
        return {"ok": False, "error": "腕部相机无画面"}
    if not K or not calib_eih or not calib_eih.get("T_flange_cam"):
        return {"ok": False, "error": "需要先完成腕部相机内参标定与 Eye-in-Hand 手眼标定"}
    if not T_base_flange or not tool_T:
        return {"ok": False, "error": "缺少机械臂 FK 或 TCP 变换"}
    if not coarse_item or not coarse_item.get("xyz_m"):
        return {"ok": False, "error": "缺少粗定位目标坐标"}

    rgb_u = undistort_rgb(rgb, undistort_intr) if (undistort_intr and intrinsics_valid(undistort_intr)) else rgb
    h, w = rgb_u.shape[:2]

    if dets2d is None:
        if det_yolo is not None and model_path:
            try:
                dets2d = det_yolo.detect(rgb_u, mode, model_path, ref=ref if mode == "vp" else None,
                                         prompts=prompts, conf=conf)
            except Exception:
                dets2d = []
        if not dets2d:
            dets2d = _segment_bright_contours(rgb_u)

    cands = [d for d in (dets2d or []) if (d.get("cls", 0) == 0 or mode != "vp") and int(d["mask"].sum()) >= 60]
    if not cands:
        return {"ok": False, "error": "腕部相机画面中未检测到目标轮廓"}

    T_bw = np.asarray(wrist_T_base_cam(T_base_flange, calib_eih), dtype=float)
    T_wb = np.linalg.inv(T_bw)
    T_bt = np.asarray(CAM.mat_mul(T_base_flange, tool_T), dtype=float)
    O_cam = T_bw[:3, 3]
    R_bw = T_bw[:3, :3]

    coarse_xyz = [float(v) for v in coarse_item["xyz_m"]]
    det_id = str(coarse_item.get("det_id") or "egg_1")
    is_egg = det_id.startswith("egg_")
    b_egg_m = min(float(egg_size_mm[0]), float(egg_size_mm[1])) / 2000.0
    z_target = coarse_xyz[2] if len(coarse_xyz) >= 3 else (float(table_z) + (b_egg_m if is_egg else 0.018))

    fx, fy, ppx, ppy = float(K["fx"]), float(K["fy"]), float(K["ppx"]), float(K["ppy"])
    sx_k = float(K.get("width") or w) / float(w)
    sy_k = float(K.get("height") or h) / float(h)

    best = None
    best_idx = -1
    for idx, d in enumerate(cands):
        m = d["mask"]
        vv, uu = np.nonzero(m)
        if len(uu) < 60:
            continue
        u_mean, v_mean = float(uu.mean()) * sx_k, float(vv.mean()) * sy_k
        ray_c = R_bw @ np.array([(u_mean - ppx) / fx, (v_mean - ppy) / fy, 1.0], dtype=float)
        if abs(ray_c[2]) < 1e-4:
            continue
        lam_c = (z_target - float(O_cam[2])) / float(ray_c[2])
        if lam_c <= 0.02 or lam_c > 1.2:
            continue
        p_c = O_cam + lam_c * ray_c
        dist_coarse = math.hypot(float(p_c[0]) - coarse_xyz[0], float(p_c[1]) - coarse_xyz[1])
        if best is None or dist_coarse < best["dist_coarse"]:
            best = {"d": d, "dist_coarse": dist_coarse, "u_mean": u_mean, "v_mean": v_mean, "lam_c": lam_c}
            best_idx = idx

    if best is None:
        return {"ok": False, "error": "腕部相机检测到的轮廓无法投影到目标高度平面"}

    d = best["d"]
    m = d["mask"]
    cnt = d.get("contour")
    if cnt is None or len(cnt) < 8:
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float) if cs else np.column_stack(np.nonzero(m)[::-1]).astype(float)

    uu_c = cnt[:, 0] * sx_k
    vv_c = cnt[:, 1] * sy_k
    dc_rays = np.stack([(uu_c - ppx) / fx, (vv_c - ppy) / fy, np.ones(len(uu_c))], axis=1)
    D_rays = dc_rays @ R_bw.T
    lams = (z_target - float(O_cam[2])) / D_rays[:, 2]
    P_cnt = O_cam + lams[:, None] * D_rays
    pts_xy_mm = (P_cnt[:, :2] * 1000.0).astype(np.float32)

    # Use the area centroid + 2D PCA of all interior mask pixels on the target plane for sub-mm & sub-degree accuracy
    vv_all, uu_all = np.nonzero(m)
    if len(uu_all) > 6000:
        step_s = max(1, len(uu_all) // 6000)
        uu_all, vv_all = uu_all[::step_s], vv_all[::step_s]
    dc_all = np.stack([(uu_all * sx_k - ppx) / fx, (vv_all * sy_k - ppy) / fy, np.ones(len(uu_all))], axis=1)
    D_all = dc_all @ R_bw.T
    lams_all = (z_target - float(O_cam[2])) / D_all[:, 2]
    P_all = O_cam + lams_all[:, None] * D_all
    fine_x = float(np.mean(P_all[:, 0]))
    fine_y = float(np.mean(P_all[:, 1]))
    fine_xyz = [round(fine_x, 5), round(fine_y, 5), round(float(z_target), 5)]

    dx_all = P_all[:, 0] - fine_x
    dy_all = P_all[:, 1] - fine_y
    cov_xx = float(np.mean(dx_all * dx_all))
    cov_yy = float(np.mean(dy_all * dy_all))
    cov_xy = float(np.mean(dx_all * dy_all))
    yaw_rad = 0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy)
    while yaw_rad > math.pi / 2:
        yaw_rad -= math.pi
    while yaw_rad <= -math.pi / 2:
        yaw_rad += math.pi
    u_maj = np.array([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=float)
    v_min = np.array([-math.sin(yaw_rad), math.cos(yaw_rad)], dtype=float)
    rel_cnt_mm = (P_cnt[:, :2] - np.array([fine_x, fine_y])) * 1000.0
    proj_u = rel_cnt_mm @ u_maj
    proj_v = rel_cnt_mm @ v_min
    L_raw_mm = float(proj_u.max() - proj_u.min())
    W_raw_mm = float(proj_v.max() - proj_v.min())

    # Perspective cone correction for 3D ellipsoid / object thickness at close range
    dist_cam_m = float(np.linalg.norm(O_cam - np.array(fine_xyz)))
    cone_corr = math.sqrt(max(0.75, 1.0 - (b_egg_m / max(0.06, dist_cam_m)) ** 2)) if is_egg else 1.0
    L_mm = round(L_raw_mm * cone_corr, 1)
    W_mm = round(W_raw_mm * cone_corr, 1)

    fine_yaw_deg = round(math.degrees(yaw_rad), 1)
    coarse_yaw_deg = float(coarse_item.get("yaw_deg") or 0.0)
    delta_yaw_deg = fine_yaw_deg - coarse_yaw_deg
    while delta_yaw_deg > 90.0:
        delta_yaw_deg -= 180.0
    while delta_yaw_deg <= -90.0:
        delta_yaw_deg += 180.0
    delta_yaw_deg = round(delta_yaw_deg, 1)

    tcp_xyz = T_bt[:3, 3]
    dx_tcp_mm = round((fine_x - float(tcp_xyz[0])) * 1000.0, 2)
    dy_tcp_mm = round((fine_y - float(tcp_xyz[1])) * 1000.0, 2)
    delta_tcp_norm_mm = round(math.hypot(dx_tcp_mm, dy_tcp_mm), 2)
    delta_coarse_mm = round(math.hypot(fine_x - coarse_xyz[0], fine_y - coarse_xyz[1]) * 1000.0, 2)

    # Project refined centre and fingertip drop-zones into wrist camera pixel coordinates
    p_fine_cam = (T_wb[:3, :3] @ np.array(fine_xyz) + T_wb[:3, 3]).tolist()
    uv_fine = CAM.project(p_fine_cam, K) or [best["u_mean"], best["v_mean"]]
    # Grasp axis in base_link is perpendicular to the object's major axis
    u_grasp_base = np.array([-math.sin(yaw_rad), math.cos(yaw_rad), 0.0], dtype=float)
    p_fl_base = np.array(fine_xyz) - float(finger_half_gap_m) * u_grasp_base
    p_fr_base = np.array(fine_xyz) + float(finger_half_gap_m) * u_grasp_base
    uv_fl = CAM.project((T_wb[:3, :3] @ p_fl_base + T_wb[:3, 3]).tolist(), K) or [uv_fine[0] - 35.0, uv_fine[1]]
    uv_fr = CAM.project((T_wb[:3, :3] @ p_fr_base + T_wb[:3, 3]).tolist(), K) or [uv_fine[0] + 35.0, uv_fine[1]]
    px_per_m = 0.5 * (fx + fy) / max(0.06, float(p_fine_cam[2]))
    hw_px = 0.5 * float(finger_box_m[0]) * px_per_m / sx_k
    hh_px = 0.5 * float(finger_box_m[1]) * px_per_m / sy_k
    fl_box = [uv_fl[0] / sx_k - hw_px, uv_fl[1] / sy_k - hh_px, uv_fl[0] / sx_k + hw_px, uv_fl[1] / sy_k + hh_px]
    fr_box = [uv_fr[0] / sx_k - hw_px, uv_fr[1] / sy_k - hh_px, uv_fr[0] / sx_k + hw_px, uv_fr[1] / sy_k + hh_px]

    clearance_ok = True
    for j, other in enumerate(cands):
        if j == best_idx:
            continue
        om = other["mask"]
        for bx in (fl_box, fr_box):
            x0, y0 = max(0, int(round(bx[0]))), max(0, int(round(bx[1])))
            x1, y1 = min(w, int(round(bx[2]))), min(h, int(round(bx[3])))
            if x1 > x0 and y1 > y0 and bool(om[y0:y1, x0:x1].any()):
                clearance_ok = False
                break

    # Check outer border proximity (130° wide-angle edge distortion zone)
    u_frac = (uv_fine[0] / sx_k) / float(w)
    v_frac = (uv_fine[1] / sy_k) / float(h)
    margin = 0.5 * (1.0 - WRIST_ROI_FRAC)
    in_center_roi = (margin <= u_frac <= 1.0 - margin) and (margin <= v_frac <= 1.0 - margin)

    warn = []
    if delta_coarse_mm > max_coarse_diff_mm:
        warn.append(f"与主相机粗定位偏差 {delta_coarse_mm:.1f} mm (> {max_coarse_diff_mm:.0f} mm，可能匹配错目标或手眼未准)")
    if not clearance_ok:
        warn.append("指尖落点区有邻近物体遮挡/碰撞风险")
    if not in_center_roi:
        warn.append("目标偏离腕部画面中心区（建议先粗定位悬停到正上方）")

    trusted = (delta_coarse_mm <= max_coarse_diff_mm) and clearance_ok
    axis_world = [round(math.cos(yaw_rad), 5), round(math.sin(yaw_rad), 5), 0.0]
    p_ax1 = np.array(fine_xyz) - 0.025 * np.array(axis_world)
    p_ax2 = np.array(fine_xyz) + 0.025 * np.array(axis_world)
    uv_ax1 = CAM.project((T_wb[:3, :3] @ p_ax1 + T_wb[:3, 3]).tolist(), K) or uv_fine
    uv_ax2 = CAM.project((T_wb[:3, :3] @ p_ax2 + T_wb[:3, 3]).tolist(), K) or uv_fine

    return {
        "ok": True,
        "det_id": det_id,
        "trusted": trusted,
        "clearance_ok": clearance_ok,
        "in_center_roi": in_center_roi,
        "coarse_xyz_m": [round(v, 5) for v in coarse_xyz],
        "fine_xyz_m": fine_xyz,
        "tcp_xyz_m": [round(float(v), 5) for v in tcp_xyz],
        "delta_tcp_mm": [dx_tcp_mm, dy_tcp_mm],
        "delta_tcp_norm_mm": delta_tcp_norm_mm,
        "delta_coarse_mm": delta_coarse_mm,
        "fine_yaw_deg": fine_yaw_deg,
        "delta_yaw_deg": delta_yaw_deg,
        "axis_world": axis_world,
        "size_mm": [L_mm, W_mm],
        "dist_cam_mm": round(dist_cam_m * 1000.0, 1),
        "uv_fine": [round(uv_fine[0], 1), round(uv_fine[1], 1)],
        "axis_uv": [round(uv_ax1[0], 1), round(uv_ax1[1], 1), round(uv_ax2[0], 1), round(uv_ax2[1], 1)],
        "finger_boxes_px": [[round(v * sx_k, 1) for v in fl_box], [round(v * sx_k, 1) for v in fr_box]],
        "box_px": [round(float(v) * (sx_k if i % 2 == 0 else sy_k), 1) for i, v in enumerate(d["box"])],
        "conf": round(float(d.get("conf", 0.9)), 3),
        "warnings": warn,
    }

