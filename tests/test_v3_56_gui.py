#!/usr/bin/env python3
"""V3.56 GUI test for Wrist Camera Eye-in-Hand Calibration (needs tkinter + DISPLAY; skips otherwise).
Tests end-to-end on_wcalib_auto with pure-translation initial probes, Look-At Dome pose planning,
instant blind-pose skip (0 frames -> no retry), monocular scale bias (1.18x) + outlier self-recovery,
and on_wcalib_check verification.
"""
import math
import os
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    print("[GUI56] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI56] no DISPLAY -> skipped"); sys.exit(0)

import v347_fake as F                         # noqa: E402
import NERO_V315_Consistency_Safety as CS     # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49     # noqa: E402

TMP = tempfile.mkdtemp(prefix="v356gui_")
CFG = os.path.join(TMP, "config")
os.makedirs(CFG)
CAM.CALIB_PATH = CAM.DEFAULT_CALIB_PATH = os.path.join(CFG, "camera_calib.json")
CAM.LOCATE_PATH = CAM.DEFAULT_LOCATE_PATH = os.path.join(CFG, "locate_settings.json")
CAM.TCP_CALIB_PATH = CAM.DEFAULT_TCP_CALIB_PATH = os.path.join(CFG, "tcp_calib.json")
W49.WRIST_INFO_PATH = os.path.join(CFG, "wrist_camera_info.json")
W49.WRIST_INTR_PATH = W49.DEFAULT_WRIST_INTR_PATH = os.path.join(CFG, "wrist_intrinsics.json")
W49.WRIST_HANDEYE_PATH = W49.DEFAULT_WRIST_HANDEYE_PATH = os.path.join(CFG, "wrist_handeye_calib.json")
CAM.CALIB_DIR = os.path.join(TMP, "cameracalib")
os.makedirs(CAM.CALIB_DIR)
CAM.SETTLE_S = 0.0
CAM.FAST_SETTLE_S = 0.0
CAM.FAST_WRIST_EMPTY_S = 0.05
CAM.STILL_TOL_DEG = 999.0


class SyncThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self.t, self.a, self.k, self.alive = target, args, kwargs or {}, False

    def start(self):
        self.alive = True
        try:
            self.t(*self.a, **self.k)
        finally:
            self.alive = False

    def is_alive(self):
        return self.alive


CAM.threading = type("T", (), {"Thread": SyncThread, "Lock": threading.Lock, "Event": threading.Event})
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
LIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708)]
Q_WORK = [0.0, 30.0, 0.0, 80.0, 0.0, 40.0, 0.0]

K_WRIST = {"width": 1920, "height": 1080, "fx": 720.0, "fy": 720.0, "ppx": 960.0, "ppy": 540.0}
A0_WORK = [list(map(float, r)) for r in CS.fk(U, dict(zip(J, [math.radians(v) for v in Q_WORK])), "base_link", "link7")]
T_BC0 = [[1.0, 0.0, 0.0, -0.450], [0.0, -1.0, 0.0, 0.020], [0.0, 0.0, -1.0, 0.262], [0.0, 0.0, 0.0, 1.0]]
X_TRUE = CAM.mat_mul(CAM.T_inv(A0_WORK), T_BC0)
Y_TRUE = [[1.0, 0.0, 0.0, -0.436], [0.0, 1.0, 0.0, 0.112], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

TEX_RGB = np.full((720, 1280, 3), 40, dtype=np.uint8)


class FakeMain:
    def __init__(self):
        self.K = self.info = None
        self.dist, self.error, self.fps, self.fid, self.on = [0.0] * 5, None, 30.0, 0, False
        self.depth_res, self.depth_preset = CAM.DEPTH_RES[0], CAM.DEPTH_PRESETS[0]

    def running(self):
        return self.on

    def start(self):
        self.on, self.K = True, dict(F.K)
        self.info = {"name": "D435I (fake)", "serial": "261222075224", "usb": "3.2", "K": dict(F.K)}

    def stop(self, timeout=0):
        self.on = False

    @property
    def frame_id(self):
        self.fid += 1
        return self.fid

    def aruco_status(self):
        return True, "fake"

    def display_ppm(self, view="color", size=(560, 315)):
        return None

    def frames(self):
        self.fid += 1
        return TEX_RGB, None, self.fid

    def detect_marker(self, marker, q_deg=None):
        return None


class FakeWrist:
    def __init__(self):
        self.K = dict(K_WRIST)
        self.dist = [0.0] * 5
        self.model_type = W49.DIST_MODELS[0]
        self.undistort_preview = False
        self.info = {}
        self.error = None
        self.fps = 30.0
        self.fid = 0
        self.on = False
        self.pose_idx = 0

    def set_intrinsics(self, intr):
        if intr and W49.intrinsics_valid(intr):
            self.K = dict(intr["K"])
            self.dist = list(intr["dist"])

    def running(self):
        return self.on

    def start(self):
        self.on = True
        self.info = {"name": "Innomaker 1080P USB2.0 (fake)", "device": "/dev/video4", "format": "MJPG",
                     "res": "1920×1080", "width": 1920, "height": 1080, "controls": {}}

    def stop(self, timeout=0):
        self.on = False

    @property
    def frame_id(self):
        self.fid += 1
        return self.fid

    def raw_frame(self):
        return TEX_RGB, self.fid

    def frames(self):
        return TEX_RGB, None, self.fid

    def display_ppm(self, view="color", size=(560, 315)):
        return None

    def detect_marker(self, marker, q_deg=None):
        q = q_deg or pipe.q
        # Simulate 1 blind pose on auto pose #5 (0 frames -> should skip retry immediately)
        if pipe.move_count == 8:
            return None
        A = [list(map(float, r)) for r in CS.fk(U, dict(zip(J, [math.radians(v) for v in q])), "base_link", "link7")]
        B = CAM.mat_mul(CAM.T_inv(CAM.mat_mul(A, X_TRUE)), Y_TRUE)
        # Simulate 1.16x monocular scale bias + 1 outlier on move #6
        scale = 1.16
        p = [v * scale for v in CAM.T_t(B)]
        if pipe.move_count == 6:
            p[0] += 0.042
            p[2] -= 0.035
        rv, _ = cv2.Rodrigues(np.array(CAM.T_R(B), dtype=np.float64))
        uv = CAM.project(p, self.K) or [960.0, 540.0]
        return {"id": marker["id"], "p_cam": p, "rvec": rv.flatten().tolist(),
                "corners_px": [[uv[0] - 20, uv[1] - 20], [uv[0] + 20, uv[1] - 20],
                               [uv[0] + 20, uv[1] + 20], [uv[0] - 20, uv[1] + 20]],
                "center_px": uv, "side_px": 48.0}


class Pipe:
    def __init__(self):
        self.q = list(Q_WORK)
        self.teach_mode = False
        self.real_urdf = U
        self.move_count = 0
        self.state = {"moveit": ("green", "ok")}

    def actual_arm_deg(self, model):
        return list(self.q)

    def move_and_settle(self, model, target_rad, speed, label="", **kw):
        self.q = [math.degrees(v) for v in target_rad]
        self.move_count += 1
        return {"ok": True}

    def fence_goal_check(self, model, q):
        return True, "", []

    def cancel_move(self):
        pass


class Safe:
    sim = U
    dirty = False
    valid_ws = {"fence": {"polygon": [[-0.85, -0.45], [-0.10, -0.45], [-0.10, 0.45], [-0.85, 0.45]], "z_table": 0.0}}

    def arm_points(self, q):
        return []


root = tk.Tk()
root.geometry("1400x920")
frame = ttk.Frame(root)
frame.pack(fill="both", expand=True)
logs = []
pipe = Pipe()
safe = Safe()
jp = type("JP", (), {"enabled": True, "model": {"joints": J, "limits": [list(l) for l in LIM], "base_link": "base_link"}})()
cam = CAM.CameraTabUI(tk, ttk, frame, pipe, jp, safe, lambda: {"speed_percent": 10}, logs.append, lambda *a: True,
                      visible_fn=lambda: True, source_factory=FakeMain, wrist_source_factory=FakeWrist)


def tick(n=3):
    for _ in range(n):
        cam.poll()
        root.update()


tick(2)
cam.on_wrist_connect()
tick(2)
cam.wrist_intr = {"K": dict(K_WRIST), "dist": [0.0] * 5, "model_type": W49.DIST_MODELS[1], "rms_px": 0.25, "good": True}
cam.wrist_src.set_intrinsics(cam.wrist_intr)

t0 = time.time()
cam.v_wnauto.set("12")
cam.on_wcalib_auto()
tick(3)
dt_s = time.time() - t0

assert cam.wcalib_result and cam.wcalib_result["ok"] and cam.wcalib_result["good"], cam.wcalib_result
assert len(cam.wcalib_result["rejected"]) >= 1, cam.wcalib_result
assert abs(cam.wcalib_result["pnp_scale"] - (1.0 / 1.16)) < 0.03, cam.wcalib_result["pnp_scale"]
err_tx_mm = CAM.vnorm(CAM.vsub(cam.wcalib_result["cam_in_flange"], CAM.T_t(X_TRUE))) * 1000.0
assert err_tx_mm < 2.5 and cam.wcalib_result["loo_max_mm"] <= 3.0, (err_tx_mm, cam.wcalib_result)
# Verify blind pose #5 did NOT trigger '再测一次' retry
assert not any("只检测到 0 帧" in m and "再测一次" in m for m in logs), logs

cam.on_wcalib_save()
tick(2)
cam.on_wcalib_check()
tick(2)
assert cam.wcheck_state and cam.wcheck_state[0] == "ok", cam.wcheck_state

cam.shutdown()
root.destroy()
print(f"ALL V3.56 GUI TESTS PASSED (dt={dt_s:.2f}s, LOO max={cam.wcalib_result['loo_max_mm']:.2f} mm, pnp_scale={cam.wcalib_result['pnp_scale']:.4f})")
