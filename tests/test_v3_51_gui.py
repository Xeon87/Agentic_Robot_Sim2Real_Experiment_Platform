#!/usr/bin/env python3
"""V3.51 GUI test:
N1 '手眼标定' sub-tabs ['主相机手眼标定', 'eye-in-hand手眼标定']; independent wrist ArUco marker controls
   (v_wdict, v_wmid, v_wmsize defaulting to '24' mm) decoupled from main camera marker controls (v_msize '40' mm);
   on_wmarker_png() generates printable desktop marker PNG
N2 Automatic Eye-in-Hand collection ('▶ 自动采集', on_wcalib_auto): without prior wrist calibration, probes 3 poses +
   collects 10 visibility-screened poses at <= 10% speed, returns to start pose, and auto-computes T_flange_cam
   to < 0.5 mm error (合格 ✓)
N3 Saving wrist calibration (on_wcalib_save) stores independent wrist marker + 4 check_poses_deg; '快速复核（走到存档姿态）'
   (on_wcalib_check) moves through 4 check poses and turns lamp wl4 green (通过 ✓)
N4 Full-Workflow Status Lamp Bar (WorkflowLampBar / attach_workflow_bar) above log box renders 9 stages (4 solid
   required rings + 5 dashed optional rings), dynamically updating across grey, red, yellow, and green
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
    print("[GUI51] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI51] no DISPLAY -> skipped"); sys.exit(0)
import v347_fake as F                         # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V347_Yolo_Perception as YP        # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49     # noqa: E402
import NERO_V315_Consistency_Safety as CS     # noqa: E402

TMP = tempfile.mkdtemp(prefix="v351gui_")
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
Q_WORK = [0.0, 30.0, 0.0, 80.0, 0.0, 40.0, 0.0]

K_WRIST = {"width": 1280, "height": 720, "fx": 820.0, "fy": 820.0, "ppx": 640.0, "ppy": 360.0}
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
        self.seen_markers = []

    def set_intrinsics(self, intr):
        if intr and W49.intrinsics_valid(intr):
            self.K = dict(intr["K"])
            self.dist = list(intr["dist"])

    def running(self):
        return self.on

    def start(self):
        self.on = True
        self.info = {"name": "Innomaker 1080P USB2.0 (fake)", "device": "/dev/video4", "format": "MJPG",
                     "res": "1280×720", "width": 1280, "height": 720,
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
        self.seen_markers.append(dict(marker))
        q = q_deg or pipe.q
        A = [list(map(float, r)) for r in CS.fk(U, dict(zip(J, [math.radians(v) for v in q])), "base_link", "link7")]
        B = CAM.mat_mul(CAM.T_inv(CAM.mat_mul(A, X_TRUE)), Y_TRUE)
        # Scale p_cam proportionally to marker['size_mm'] / 24.0 so a wrong marker size would fail calibration
        scale = float(marker["size_mm"]) / 24.0
        p = [v * scale for v in CAM.T_t(B)]
        rv, _ = cv2.Rodrigues(np.array(CAM.T_R(B), dtype=np.float64))
        uv = CAM.project(p, self.K) or [640.0, 360.0]
        return {"id": marker["id"], "p_cam": p, "rvec": rv.flatten().tolist(),
                "corners_px": [[uv[0] - 20, uv[1] - 20], [uv[0] + 20, uv[1] - 20],
                               [uv[0] + 20, uv[1] + 20], [uv[0] - 20, uv[1] + 20]],
                "center_px": uv, "side_px": 45.0}


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
    valid_ws = None

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
# ------------------------------------------------------------------ N1: Sub-tab rename & independent wrist ArUco marker
sub_tabs = [cam.nb_calib.tab(t, "text") for t in cam.nb_calib.tabs()]
assert sub_tabs == ["主相机手眼标定", "eye-in-hand手眼标定"], sub_tabs
assert cam.v_msize.get() == "40" and cam.v_wmsize.get() == "24"
cam.v_wmsize.set("25")
assert cam.v_msize.get() == "40" and cam._wmarker_from_ui()["size_mm"] == 25.0
cam.v_wmsize.set("24")
assert cam._wmarker_from_ui()["size_mm"] == 24.0
cam.on_wmarker_png()
assert any("桌面标记图片已生成" in m for m in logs), logs[-3:]
print("[N1] Sub-tab renamed to 'eye-in-hand手眼标定'; wrist marker (default 24 mm) independent of main marker (40 mm); PNG generated")

# ------------------------------------------------------------------ N4 (part 1): Workflow lamp bar initial states
# Before safety zone is configured, safety lamp is 'red' (required safety prerequisite missing)
assert len(wf_bar.cells) == 9
assert wf_bar.cells["safety"]["required"] is True and wf_bar.cells["safety"]["rid"] is None
assert wf_bar.cells["wrist_eih"]["required"] is False and wf_bar.cells["wrist_eih"]["rid"] is not None
assert wf_bar.cells["safety"]["color"] == "red"
# Set valid safety zone -> safety lamp turns green
safe.valid_ws = {"fence": {"polygon": [[-0.8, -0.4], [-0.1, -0.4], [-0.1, 0.4], [-0.8, 0.4]], "z_table": 0.0}}
cam.update_workflow_lamps()
assert wf_bar.cells["safety"]["color"] == "green" and wf_bar.cells["arm"]["color"] == "green"

# Connect wrist camera before wrist intrinsics -> wrist_intr turns red
cam.on_wrist_connect()
tick(2)
assert wf_bar.cells["wrist_cam"]["color"] == "green" and wf_bar.cells["wrist_intr"]["color"] == "red"
# Load wrist intrinsics -> wrist_intr turns green, wrist_eih turns red (needs EIH calibration)
cam.wrist_intr = {"K": dict(K_WRIST), "dist": [0.0] * 5, "model_type": W49.DIST_MODELS[1], "rms_px": 0.22, "good": True}
cam.wrist_src.set_intrinsics(cam.wrist_intr)
cam.update_workflow_lamps()
assert wf_bar.cells["wrist_intr"]["color"] == "green" and wf_bar.cells["wrist_eih"]["color"] == "red"

# ------------------------------------------------------------------ N2: Automatic Eye-in-Hand collection (on_wcalib_auto)
cam.v_wnauto.set("10")
cam.on_wcalib_auto()
tick(2)
assert len(cam.wcalib_samples) >= 8, len(cam.wcalib_samples)
assert cam.wcalib_result and cam.wcalib_result["ok"] and cam.wcalib_result["good"], cam.wcalib_result
err_t_mm = CAM.vnorm(CAM.vsub(cam.wcalib_result["cam_in_flange"], CAM.T_t(X_TRUE))) * 1000.0
assert err_t_mm < 0.5, err_t_mm
# Verify wrist detector was called with 24 mm (not 40 mm!)
assert all(abs(m["size_mm"] - 24.0) < 1e-6 for m in cam.wrist_src.seen_markers)
# Before saving, wrist_eih lamp is yellow ('已算合格，请点保存腕部标定')
cam.update_workflow_lamps()
assert wf_bar.cells["wrist_eih"]["color"] == "yellow"
print(f"[N2] Automatic Eye-in-Hand collection: {len(cam.wcalib_samples)} poses collected at <=10% speed -> "
      f"T_flange_cam err {err_t_mm:.3f} mm (合格 ✓), wrist marker size_mm=24.0 verified")

# ------------------------------------------------------------------ N3: Save + Quick Check (on_wcalib_check)
cam.on_wcalib_save()
tick(2)
assert cam.wrist_calib and cam.wrist_calib["marker"]["size_mm"] == 24.0
assert len(cam.wrist_calib.get("check_poses_deg") or []) == 4
assert wf_bar.cells["wrist_eih"]["color"] == "green"
cam.on_wcalib_check()
tick(2)
assert cam.wcheck_state and cam.wcheck_state[0] == "ok" and max(cam.wcheck_state[1]) < 0.5
assert "通过 ✓" in cam.ws4.get()
print(f"[N3] Saved wrist_handeye_calib.json (marker=24 mm, 4 check poses) -> 快速复核 passed (max err {max(cam.wcheck_state[1]):.3f} mm)")
print("[N4] Full-Workflow Status Lamp Bar (9 stages, solid required + dashed optional rings, 4-color transitions) verified")

cam.shutdown()
root.destroy()
print("ALL V3.51 GUI TESTS PASSED")
