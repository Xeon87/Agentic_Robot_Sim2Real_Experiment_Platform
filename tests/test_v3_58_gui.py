#!/usr/bin/env python3
"""V3.58 GUI test for Kinematic-Anchored Tabletop ChArUco Intrinsic + Auto-Saved Eye-in-Hand Calibration
(needs tkinter + DISPLAY; skips otherwise)."""
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import tkinter as tk
    from tkinter import ttk
    import cv2
    import numpy as np
    if not os.environ.get("DISPLAY"):
        print("[SKIP] test_v3_58_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v3_58_gui: {exc}")
    sys.exit(0)

import NERO_V315_Consistency_Safety as CS     # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49     # noqa: E402
import tempfile as _tf                         # noqa: E402
# V4.7.2: write calibrations to a temp folder - the test now runs to the end and used to overwrite config/ (the
# user's real wrist intrinsics / hand-eye) and ~/Downloads/cameracalib
_TMP = _tf.mkdtemp(prefix="nero_gui_calib_")
_CFG = os.path.join(_TMP, "config")
os.makedirs(_CFG)
CAM.CALIB_PATH = CAM.DEFAULT_CALIB_PATH = os.path.join(_CFG, "camera_calib.json")
CAM.LOCATE_PATH = CAM.DEFAULT_LOCATE_PATH = os.path.join(_CFG, "locate_settings.json")
CAM.TCP_CALIB_PATH = CAM.DEFAULT_TCP_CALIB_PATH = os.path.join(_CFG, "tcp_calib.json")
W49.WRIST_INFO_PATH = os.path.join(_CFG, "wrist_camera_info.json")
W49.WRIST_INTR_PATH = W49.DEFAULT_WRIST_INTR_PATH = os.path.join(_CFG, "wrist_intrinsics.json")
W49.WRIST_HANDEYE_PATH = W49.DEFAULT_WRIST_HANDEYE_PATH = os.path.join(_CFG, "wrist_handeye_calib.json")
CAM.CALIB_DIR = os.path.join(_TMP, "cameracalib")
os.makedirs(CAM.CALIB_DIR)
import threading                              # noqa: E402


# V4.7.2: run the camera tab's worker synchronously (as test_v3_57_gui) -- the asserts used to run before it finished
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
fk_fn = lambda qr: [list(map(float, r)) for r in CS.fk(U, dict(zip(J, qr)), "base_link", "link7")]

Q0_DEG = [0.0, 30.0, 0.0, 80.0, 0.0, 40.0, 0.0]
Q0_RAD = [math.radians(v) for v in Q0_DEG]
A0 = fk_fn(Q0_RAD)
T_BC0_TRUE = CAM.T_from([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], [-0.450, 0.020, 0.250])
X_TRUE = CAM.mat_mul(CAM.T_inv(A0), T_BC0_TRUE)
Y_BOARD_TRUE = CAM.T_from([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [-0.450, 0.020, 0.0])

K_TRUE = {"width": 1920, "height": 1080, "fx": 720.0, "fy": 720.0, "ppx": 960.0, "ppy": 540.0}
Km_TRUE = np.array([[720.0, 0.0, 960.0], [0.0, 720.0, 540.0], [0.0, 0.0, 1.0]], dtype=np.float64)

aruco_dict = CAM._aruco_dict(cv2, "DICT_4X4_50")
cols, rows, sq_m, mk_m = 7, 5, 0.032, 0.024
if hasattr(cv2.aruco, "CharucoBoard"):
    BOARD_OBJ = cv2.aruco.CharucoBoard((cols, rows), sq_m, mk_m, aruco_dict)
    BOARD_IMG = BOARD_OBJ.generateImage((700, 500))
else:
    BOARD_OBJ = cv2.aruco.CharucoBoard_create(cols, rows, sq_m, mk_m, aruco_dict)
    BOARD_IMG = BOARD_OBJ.draw((700, 500))

hw, hh = 0.5 * cols * sq_m, 0.5 * rows * sq_m
BOARD_OUTER_3D = np.array([[-hw, -hh, 0.0], [hw, -hh, 0.0], [hw, hh, 0.0], [-hw, hh, 0.0]], dtype=np.float64)
SRC_OUTER_2D = np.float32([[-0.5, -0.5], [699.5, -0.5], [699.5, 499.5], [-0.5, 499.5]])


def render_wrist_charuco(q_deg):
    qr = [math.radians(v) for v in q_deg]
    A = fk_fn(qr)
    B = CAM.mat_mul(CAM.T_inv(CAM.mat_mul(A, X_TRUE)), Y_BOARD_TRUE)
    R_b = np.array(CAM.T_R(B), dtype=np.float64) @ np.diag([1.0, -1.0, -1.0])
    rvec, _ = cv2.Rodrigues(R_b)
    tvec = np.array(CAM.T_t(B), dtype=np.float64)
    proj, _ = cv2.projectPoints(BOARD_OUTER_3D, rvec, tvec, Km_TRUE, np.zeros(5))
    H = cv2.getPerspectiveTransform(SRC_OUTER_2D, proj.reshape(4, 2).astype(np.float32))
    warped = cv2.warpPerspective(BOARD_IMG, H, (1920, 1080), flags=cv2.INTER_LINEAR, borderValue=235)
    return cv2.cvtColor(warped, cv2.COLOR_GRAY2RGB)


class FakeMain:
    def __init__(self):
        self.K = dict(CAM.SIM_K)
        self.dist = [0.0] * 5
        self.info = {"name": "Fake D435i", "serial": "SIM358", "usb": "3.2", "color": "1280×720",
                     "depth": "1280×720", "preset": "High Accuracy", "K": dict(self.K), "dist": self.dist}
        self.fps = 30.0
        self.error = None
        self.frame_id = 1
        self._run = True
        self.rgb = np.full((720, 1280, 3), 210, dtype=np.uint8)
        self.depth = np.full((720, 1280), 0.70, dtype=np.float32)

    def start(self):
        self._run = True

    def stop(self, *a, **k):
        self._run = False

    def running(self):
        return self._run

    def frames(self):
        return self.rgb, self.depth, self.frame_id

    def display_ppm(self, view="color", roi=None):
        return CAM.ppm_bytes(cv2.resize(self.rgb, (CAM.CameraTabUI.DW, CAM.CameraTabUI.DH)))

    def detect(self, marker):
        return None


class FakeWrist:
    def __init__(self, device="auto", res=W49.WRIST_RES_LIST[0], lock_controls=True, intrinsics=None):
        self.K = dict(intrinsics["K"]) if (intrinsics and W49.intrinsics_valid(intrinsics)) else dict(K_TRUE)
        self.dist = list(intrinsics["dist"]) if (intrinsics and W49.intrinsics_valid(intrinsics)) else [0.0] * 5
        self.model_type = (intrinsics or {}).get("model_type") or W49.DIST_MODELS[0]
        self.info = {"name": "Fake Wrist 130° UVC", "device": "/dev/video2", "format": "MJPG",
                     "res": "1920×1080", "width": 1920, "height": 1080,
                     "controls": {"auto_exposure": "locked", "auto_wb": "locked", "autofocus": "locked"},
                     "K": dict(self.K), "dist": list(self.dist), "model_type": self.model_type}
        self.fps = 30.0
        self.error = None
        self.frame_id = 1
        self._run = True
        self.undistort_preview = False
        self.target_type = W49.EIH_TARGET_MODES[0]
        self.board_cfg = dict(W49.DEFAULT_BOARD)
        self.rgb = render_wrist_charuco(Q0_DEG)

    def set_intrinsics(self, intr):
        if intr and W49.intrinsics_valid(intr):
            self.K = dict(intr["K"])
            self.dist = [float(v) for v in intr["dist"]]
            self.model_type = intr.get("model_type") or W49.DIST_MODELS[0]
            self.info["K"] = dict(self.K)
            self.info["dist"] = list(self.dist)

    def start(self):
        self._run = True

    def stop(self, *a, **k):
        self._run = False

    def running(self):
        return self._run

    def frames(self):
        self.rgb = render_wrist_charuco(pipe.q)
        self.frame_id += 1
        return self.rgb, None, self.frame_id

    def raw_frame(self):
        self.rgb = render_wrist_charuco(pipe.q)
        self.frame_id += 1
        return self.rgb, self.frame_id

    def display_ppm(self, view="color", size=(CAM.CameraTabUI.DW, CAM.CameraTabUI.DH)):
        self.rgb = render_wrist_charuco(pipe.q)
        return CAM.ppm_bytes(cv2.resize(self.rgb, (int(size[0]), int(size[1]))))

    def detect_marker(self, marker, robot_q_deg=None):
        q_use = robot_q_deg if robot_q_deg is not None else pipe.q
        self.rgb = render_wrist_charuco(q_use)
        self.frame_id += 1
        return W49.detect_charuco_pose(self.rgb, self.K, self.dist, board_cfg=self.board_cfg, model_type=self.model_type)


class Pipe:
    def __init__(self):
        self.q = list(Q0_DEG)
        self.teach_mode = False
        self.real_urdf = U
        self.moves = []

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
cam.on_wrist_connect()
tick(2)
# Test with the default 8-parameter rational model (DIST_MODELS[0])!
cam.v_imodel.set(W49.DIST_MODELS[0])
cam.v_inauto.set("10")

cam.on_intr_auto()
tick(4)

assert len(cam.intr_samples) >= 8, (len(cam.intr_samples), logs[-12:])
assert cam.wrist_intr and cam.wrist_intr["good"] and os.path.isfile(W49.WRIST_INTR_PATH), cam.wrist_intr
assert abs(cam.wrist_intr["K"]["fx"] - 720.0) < 25.0, cam.wrist_intr["K"]
# Verify both wrist_calib (saved to disk) and wcalib_result are populated and good!
assert cam.wrist_calib and cam.wrist_calib["good"] and os.path.isfile(W49.WRIST_HANDEYE_PATH), [l for l in logs if "eih" in l or "intr" in l][-8:]
assert cam.wcalib_result and cam.wcalib_result["good"], cam.wcalib_result
# Verify tv_intr still displays per_view_px values even after auto-save!
first_row_vals = cam.tv_intr.item(cam.tv_intr.get_children()[0], "values")
assert first_row_vals[-1] != "-", first_row_vals

err_tx_mm = CAM.vnorm(CAM.vsub(cam.wcalib_result["cam_in_flange"], CAM.T_t(X_TRUE))) * 1000.0
assert err_tx_mm < 3.0 and cam.wcalib_result["loo_max_mm"] < 2.5, (err_tx_mm, cam.wcalib_result)

cam.shutdown()
root.destroy()
print(f"ALL V3.58 GUI TESTS PASSED (intr fx={cam.wrist_intr['K']['fx']:.1f}, RMS={cam.wrist_intr['rms_px']:.3f} px, "
      f"EIH LOO max={cam.wcalib_result['loo_max_mm']:.2f} mm, auto-saved wrist_handeye_calib.json)")
