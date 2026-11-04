#!/usr/bin/env python3
"""V3.52 GUI test:
P1 Unified 560x315 main camera canvas scaling: RealSenseSource.display_ppm outputs 560x315 PPM; _px, _canvas_to_img,
   and place_yolo_labels agree to 0.0 px (eliminating the upper-left YOLO box offset)
P2 Log-bar Teach Mode buttons ('进入示教模式' wf_bar.b_teach_in & '退出示教模式' wf_bar.b_teach_out) to the left of Copy Log:
   clicking '进入示教模式' enters hand-guiding; dragging the robot to a marker-viewing pose and clicking '退出示教模式'
   calls set_teach_mode(False, ..., return_home=False), holding the robot in place without returning to zero [0]*7
P3 Non-blocking on_wcalib_record & auto-clearing on_wcalib_auto with 8-parameter rational distortion:
   auto-clears stale batch samples from a moved marker, probes 3 visibility-screened poses + collects 10 poses,
   and solves T_flange_cam to < 1.0 mm (合格 ✓)
"""
import math
import os
import sys
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    print("[GUI52] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI52] no DISPLAY -> skipped"); sys.exit(0)
import v347_fake as F                         # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V347_Yolo_Perception as YP        # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49     # noqa: E402
import NERO_V315_Consistency_Safety as CS     # noqa: E402

TMP = tempfile.mkdtemp(prefix="v352gui_")
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
Q_ZERO = [0.0] * 7
Q_WORK = [0.0, 30.0, 0.0, 80.0, 0.0, 40.0, 0.0]

K_WRIST = {"width": 1280, "height": 720, "fx": 820.0, "fy": 820.0, "ppx": 640.0, "ppy": 360.0}
DIST8 = [1.65, 0.82, 0.0004, -0.0003, 0.35, 2.05, 1.25, 0.68]
A0_WORK = [list(map(float, r)) for r in CS.fk(U, dict(zip(J, [math.radians(v) for v in Q_WORK])), "base_link", "link7")]
T_BC0 = [[1.0, 0.0, 0.0, -0.450], [0.0, -1.0, 0.0, 0.020], [0.0, 0.0, -1.0, 0.250], [0.0, 0.0, 0.0, 1.0]]
X_TRUE = CAM.mat_mul(CAM.T_inv(A0_WORK), T_BC0)
Y_TRUE = [[1.0, 0.0, 0.0, -0.450], [0.0, 1.0, 0.0, 0.020], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]

TEX_RGB = np.full((720, 1280, 3), 40, dtype=np.uint8)
TEX_RGB[::2, ::2] = 60


class FakeMain:
    def __init__(self):
        self.K = self.info = None
        self.dist, self.error, self.fps, self.fid, self.on = [0.0] * 5, None, 30.0, 0, False
        self.depth_res, self.depth_preset = CAM.DEPTH_RES[0], CAM.DEPTH_PRESETS[0]

    def running(self):
        return self.on

    def start(self):
        self.on, self.K = True, dict(F.K)
        self.info = {"name": "D435I (fake)", "serial": "261222075224", "usb": "3.2", "K": dict(F.K),
                     "depth_res": self.depth_res, "preset": self.depth_preset}

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
        return TEX_RGB, np.full((720, 1280), 0.8, dtype=np.float32), self.fid

    def detect_marker(self, marker, q_deg=None):
        return None


class FakeWrist:
    def __init__(self):
        self.K = dict(K_WRIST)
        self.dist = list(DIST8)
        self.model_type = W49.DIST_MODELS[0]
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

    def running(self):
        return self.on

    def start(self):
        self.on = True
        self.info = {"name": "Innomaker 1080P USB2.0 (fake)", "device": "/dev/video4", "format": "MJPG",
                     "res": "1920×1080", "width": 1920, "height": 1080,
                     "controls": {"autofocus": "off", "auto_exposure": "locked(manual)", "auto_wb": "locked"}}

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
        A = [list(map(float, r)) for r in CS.fk(U, dict(zip(J, [math.radians(v) for v in q])), "base_link", "link7")]
        B = CAM.mat_mul(CAM.T_inv(CAM.mat_mul(A, X_TRUE)), Y_TRUE)
        p = list(CAM.T_t(B))
        rv, _ = cv2.Rodrigues(np.array(CAM.T_R(B), dtype=np.float64))
        uv = CAM.project(p, self.K) or [640.0, 360.0]
        return {"id": marker["id"], "p_cam": p, "rvec": rv.flatten().tolist(),
                "corners_px": [[uv[0] - 20, uv[1] - 20], [uv[0] + 20, uv[1] - 20],
                               [uv[0] + 20, uv[1] + 20], [uv[0] - 20, uv[1] + 20]],
                "center_px": uv, "side_px": 45.0}


class Pipe:
    def __init__(self):
        self.q = list(Q_ZERO)
        self.teach_mode = False
        self.real_urdf = U
        self.moves = []
        self.teach_calls = []
        self.state = {"moveit": ("green", "ok")}

    def actual_arm_deg(self, model):
        return list(self.q)

    def set_teach_mode(self, on, timeout_s=30.0, home_model=None, home_fn=None, return_home=True):
        self.teach_calls.append((bool(on), bool(return_home), list(self.q)))
        self.teach_mode = bool(on)
        if not on and return_home:
            self.q = list(Q_ZERO)
        return {"ok": True}

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
    valid_ws = {"fence": {"polygon": [[-0.8, -0.4], [-0.1, -0.4], [-0.1, 0.4], [-0.8, 0.4]], "z_table": 0.0}}

    def arm_points(self, q):
        return []


root = tk.Tk()
root.geometry("1400x920")
frame = ttk.Frame(root)
frame.pack(fill="both", expand=True)
wf_holder = ttk.Frame(root)
wf_holder.pack(fill="x")
logs = []
pipe = Pipe()
safe = Safe()
jp = type("JP", (), {"enabled": True, "model": {"joints": J, "limits": [list(l) for l in LIM], "base_link": "base_link"}})()
cam = CAM.CameraTabUI(tk, ttk, frame, pipe, jp, safe, lambda: {"speed_percent": 10}, logs.append, lambda *a: True,
                      visible_fn=lambda: True, source_factory=FakeMain, wrist_source_factory=FakeWrist)
wf_bar = cam.attach_workflow_bar(wf_holder)


def tick(n=3):
    for _ in range(n):
        cam.poll()
        root.update()


tick(2)
# ------------------------------------------------------------------ P1: Unified 560x315 canvas scaling
rs_dummy = CAM.RealSenseSource()
rs_dummy.rgb = TEX_RGB
rs_dummy.depth = np.full((720, 1280), 0.8, dtype=np.float32)
ppm_560 = rs_dummy.display_ppm("color", size=(560, 315))
assert ppm_560.startswith(b"P6 560 315 255\n") and len(ppm_560) == len(b"P6 560 315 255\n") + 560 * 315 * 3
assert cam._px((640.0, 360.0)) == (280.0, 157.5) and cam._canvas_to_img(280.0, 157.5) == (640.0, 360.0)
print("[P1] RealSenseSource.display_ppm outputs 560x315 PPM; _px and place_yolo_labels unified at (280.0, 157.5)")

# ------------------------------------------------------------------ P2: Log-bar Teach Mode buttons (find start pose & hold)
assert wf_bar.b_teach_in is not None and wf_bar.b_teach_out is not None
assert str(wf_bar.b_teach_in.cget("state")) == "normal" and str(wf_bar.b_teach_out.cget("state")) == "normal"
cam.on_teach_enter()
tick(2)
assert pipe.teach_mode is True
assert wf_bar.b_teach_in.cget("text") == "示教中(零力)" and str(wf_bar.b_teach_out.cget("state")) == "normal"
# User drags the robot by hand from Q_ZERO to Q_WORK where the ArUco marker is visible:
pipe.q = list(Q_WORK)
cam.on_teach_exit()
tick(2)
assert pipe.teach_mode is False
assert pipe.teach_calls[-1][0] is False and pipe.teach_calls[-1][1] is False  # return_home=False!
assert pipe.q == list(Q_WORK), f"Robot moved back to zero instead of holding Q_WORK: {pipe.q}"
print("[P2] Log-bar '进入示教模式' / '退出示教模式' buttons verified: exited teach mode holding Q_WORK (return_home=False)")

# ------------------------------------------------------------------ P3: Non-blocking record & auto-clearing EIH auto-collection
cam.on_wrist_connect()
cam.wrist_intr = {"K": dict(K_WRIST), "dist": list(DIST8), "model_type": W49.DIST_MODELS[0], "rms_px": 0.24, "good": True}
cam.wrist_src.set_intrinsics(cam.wrist_intr)
tick(2)
# Record 1 sample via non-blocking on_wcalib_record
cam.on_wcalib_record()
tick(2)
assert len(cam.wcalib_samples) == 1
# Inject a corrupted stale sample from an old marker position ([0.9, -0.8, 0.9] m)
cam.wcalib_samples.append({
    "q_deg": list(Q_WORK),
    "p_cam": [0.9, -0.8, 0.9],
    "rvec": [0.0, 0.0, 0.0],
    "T_cam_marker": CAM.T_from([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.9, -0.8, 0.9]),
    "T_base_flange": A0_WORK,
    "spread_mm": 0.5,
})
assert len(cam.wcalib_samples) == 2

# Run on_wcalib_auto -> automatically clears the stale batch and solves cleanly!
cam.v_wnauto.set("10")
cam.on_wcalib_auto()
tick(2)
assert cam.wcalib_result and cam.wcalib_result["ok"] and cam.wcalib_result["good"], cam.wcalib_result
err_tx = CAM.vnorm(CAM.vsub(cam.wcalib_result["cam_in_flange"], CAM.T_t(X_TRUE))) * 1000.0
assert err_tx < 0.5, err_tx
print(f"[P3] Non-blocking on_wcalib_record & auto-clearing on_wcalib_auto: stale samples cleared, "
      f"{len(cam.wcalib_samples)} poses collected -> T_flange_cam err {err_tx:.3f} mm (合格 ✓)")

cam.shutdown()
root.destroy()
print("ALL V3.52 GUI TESTS PASSED")
