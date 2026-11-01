#!/usr/bin/env python3
"""V3.57 GUI test for Automatic Tabletop Board Intrinsic Collection & ChArUco Eye-in-Hand Calibration
(needs tkinter + DISPLAY; skips otherwise).
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
    print("[GUI57] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI57] no DISPLAY -> skipped"); sys.exit(0)

import v347_fake as F                         # noqa: E402
import NERO_V315_Consistency_Safety as CS     # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49     # noqa: E402

TMP = tempfile.mkdtemp(prefix="v357gui_")
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

K_TRUE = {"width": 1280, "height": 720, "fx": 640.0, "fy": 640.0, "ppx": 640.0, "ppy": 360.0}
Km_TRUE = np.array([[640.0, 0.0, 640.0], [0.0, 640.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
A0_WORK = [list(map(float, r)) for r in CS.fk(U, dict(zip(J, [math.radians(v) for v in Q_WORK])), "base_link", "link7")]
T_BC0 = [[1.0, 0.0, 0.0, -0.450], [0.0, -1.0, 0.0, 0.020], [0.0, 0.0, -1.0, 0.250], [0.0, 0.0, 0.0, 1.0]]
X_TRUE = CAM.mat_mul(CAM.T_inv(A0_WORK), T_BC0)
Y_BOARD = [[1.0, 0.0, 0.0, -0.450], [0.0, 1.0, 0.0, 0.020], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

aruco_dict = CAM._aruco_dict(cv2, "DICT_4X4_50")
if hasattr(cv2.aruco, "CharucoBoard"):
    BOARD = cv2.aruco.CharucoBoard((7, 5), 0.032, 0.024, aruco_dict)
    BOARD_IMG = BOARD.generateImage((700, 500))
else:
    BOARD = cv2.aruco.CharucoBoard_create(7, 5, 0.032, 0.024, aruco_dict)
    BOARD_IMG = BOARD.draw((700, 500))

HW, HH = 0.5 * 7 * 0.032, 0.5 * 5 * 0.032
OUTER_3D = np.array([[-HW, -HH, 0.0], [HW, -HH, 0.0], [HW, HH, 0.0], [-HW, HH, 0.0]], dtype=np.float64)
SRC_OUTER = np.float32([[-0.5, -0.5], [699.5, -0.5], [699.5, 499.5], [-0.5, 499.5]])


def render_board_at_q(q_deg):
    A = [list(map(float, r)) for r in CS.fk(U, dict(zip(J, [math.radians(v) for v in q_deg])), "base_link", "link7")]
    B = CAM.mat_mul(CAM.T_inv(CAM.mat_mul(A, X_TRUE)), Y_BOARD)
    rv, _ = cv2.Rodrigues(np.array(CAM.T_R(B), dtype=np.float64))
    tv = np.array(CAM.T_t(B), dtype=np.float64)
    proj, _ = cv2.projectPoints(OUTER_3D, rv, tv, Km_TRUE, np.zeros(5))
    H_w = cv2.getPerspectiveTransform(SRC_OUTER, proj.reshape(4, 2).astype(np.float32))
    g = cv2.warpPerspective(BOARD_IMG, H_w, (1280, 720), flags=cv2.INTER_LINEAR, borderValue=235)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)


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
        return np.zeros((720, 1280, 3), dtype=np.uint8), None, self.fid

    def detect_marker(self, marker, q_deg=None):
        return None


class FakeWrist:
    def __init__(self):
        self.K = None
        self.dist = [0.0] * 5
        self.model_type = W49.DIST_MODELS[1]
        self.undistort_preview = False
        self.info = {}
        self.error = None
        self.fps = 30.0
        self.fid = 0
        self.on = False

    def set_intrinsics(self, intr):
        if intr and W49.intrinsics_valid(intr):
            self.K = dict(intr["K"])
            self.dist = list(intr["dist"])
            self.model_type = intr.get("model_type") or W49.DIST_MODELS[1]

    def running(self):
        return self.on

    def start(self):
        self.on = True
        self.info = {"name": "Innomaker 1080P USB2.0 (fake)", "device": "/dev/video4", "format": "MJPG",
                     "res": "1280×720", "width": 1280, "height": 720, "controls": {}}

    def stop(self, timeout=0):
        self.on = False

    @property
    def frame_id(self):
        self.fid += 1
        return self.fid

    def raw_frame(self):
        self.fid += 1
        return render_board_at_q(pipe.q), self.fid

    def frames(self):
        self.fid += 1
        return render_board_at_q(pipe.q), None, self.fid

    def display_ppm(self, view="color", size=(560, 315)):
        return None

    def detect_marker(self, marker, q_deg=None):
        rgb = render_board_at_q(q_deg or pipe.q)
        return W49.detect_charuco_pose(rgb, self.K or K_TRUE, self.dist)


class Pipe:
    def __init__(self):
        self.q = list(Q_WORK)
        self.teach_mode = False
        self.real_urdf = U
        self.moves = []
        self.state = {"moveit": ("green", "ok")}

    def actual_arm_deg(self, model):
        return list(self.q)

    def move_and_settle(self, model, target_rad, speed, label="", **kw):
        self.q = [math.degrees(v) for v in target_rad]
        self.moves.append((label, list(self.q), speed))
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
assert hasattr(cam, "b_intr_auto") and hasattr(cam, "v_inauto") and hasattr(cam, "v_wtarget")
assert cam.v_wtarget.get() == W49.EIH_TARGET_MODES[0]

cam.on_wrist_connect()
tick(2)
cam.v_imodel.set(W49.DIST_MODELS[1])
cam.v_inauto.set("10")

# Run one-pass automatic tabletop ChArUco intrinsic + Eye-in-Hand collection!
cam.on_intr_auto()
tick(4)

assert len(cam.intr_samples) >= 8, len(cam.intr_samples)
assert cam.wrist_intr and os.path.isfile(W49.WRIST_INTR_PATH), cam.wrist_intr
assert cam.wcalib_result and cam.wcalib_result["ok"] and cam.wcalib_result["good"], cam.wcalib_result
err_tx_mm = CAM.vnorm(CAM.vsub(cam.wcalib_result["cam_in_flange"], CAM.T_t(X_TRUE))) * 1000.0
assert err_tx_mm < 3.0 and cam.wcalib_result["loo_max_mm"] < 2.5, (err_tx_mm, cam.wcalib_result)

cam.shutdown()
root.destroy()
print(f"ALL V3.57 GUI TESTS PASSED (intr RMS={cam.wrist_intr['rms_px']:.3f} px, EIH LOO max={cam.wcalib_result['loo_max_mm']:.2f} mm, t_X err={err_tx_mm:.2f} mm)")
