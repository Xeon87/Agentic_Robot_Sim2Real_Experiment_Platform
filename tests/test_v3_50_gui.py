#!/usr/bin/env python3
"""V3.50 GUI test: dual 16:9 live monitor canvases (cam.cv 560x315 + cam.cv_wrist 560x315),
live wrist quality/motion gate badge, TCP crosshair & fingertip drop-zone overlay on cv_wrist,
and coarse-to-fine closed loop ('识别鸡蛋' -> '▶ 走到选中目标上方悬停' -> '◎ 腕部精定位' -> '✥ 一键微调对齐')
(needs tkinter + a display; skips otherwise).
All paths point to a temp folder (never the user's ~/Downloads/cameracalib).

M1 Left column has both cam.cv (560x315) and cam.cv_wrist (560x315) + cam.v_wrist_live; connecting both cameras
   updates both monitors and displays '静止可采 ✓' in v_wrist_live
M2 With wrist intrinsics and Eye-in-Hand calibration loaded, _redraw_wrist draws the TCP crosshair, 21 mm egg ring,
   fingertip clearance boxes, and quality badge on cv_wrist
M3 Motion gate blocks on_wrist_refine when the arm has not settled (joint_step_deg > 0.15 deg)
M4 Full coarse-to-fine closed loop: main camera locates egg_1 -> hover 100 mm above coarse egg_1 ->
   '◎ 腕部精定位' refines egg_1 to < 0.3 mm of true wrist-viewed egg -> '✥ 一键微调对齐' moves TCP horizontally
   by (Δx, Δy) to align within < 0.5 mm of the refined egg centre
"""
import math
import os
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    print("[GUI50] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI50] no DISPLAY -> skipped"); sys.exit(0)
import v347_fake as F                         # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V347_Yolo_Perception as YP        # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49     # noqa: E402
import NERO_V315_Consistency_Safety as CS     # noqa: E402

TMP = tempfile.mkdtemp(prefix="v350gui_")
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
YP.REF_JSON, YP.REF_IMG, YP.SAMPLES_DIR = os.path.join(CFG, "yolo_ref.json"), os.path.join(CFG, "yolo_ref.jpg"), \
    os.path.join(CFG, "samples")
MD = os.path.join(TMP, "objectmodels")
os.makedirs(MD)
Path(MD, "yoloe-26s-seg.pt").write_bytes(b"w")
Path(MD, "yoloe-26s-seg-pf.pt").write_bytes(b"pf")
Path(MD, "mobileclip2_b.ts").write_bytes(b"enc")
F.install()


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

# Coarse scene in D435i: egg_1 at (-0.445, 0.016, 0.021) m, yaw = 20 deg
OBS = [{"name": "E1", "label": "egg", "kind": "egg", "c": [-0.445, 0.016, F.B_EGG], "yaw": math.radians(20.0), "conf": 0.95}]
F.STATE.bleed_px = 0
RGB_M, DEPTH_M, IDS_M = F.render(OBS, noise_m=0.0003)
F.STATE.scene = (OBS, IDS_M)

# True physical egg for the close-up wrist camera: at (-0.450, 0.020, 0.021) m, yaw = 30 deg (5 mm X, 4 mm Y offset from coarse)
TRUE_EGG_XYZ = np.array([-0.450, 0.020, F.B_EGG], dtype=float)
TRUE_EGG_YAW = math.radians(30.0)
K_WRIST = {"width": 1280, "height": 720, "fx": 820.0, "fy": 820.0, "ppx": 640.0, "ppy": 360.0}
_tool_T0 = [list(map(float, r)) for r in CS.tool_transform_from_sim(U)]
_q_hov0, _ = CAM.plan_hover_target(U, J, LIM, [math.radians(v) for v in Q_WORK], _tool_T0, [-0.445, 0.016, F.B_EGG + 0.10])
_T_bf_hov0 = np.asarray(CS.fk(U, dict(zip(J, _q_hov0)), "base_link", "link7"), dtype=float)
_T_bw_hov0 = np.array([[1.0, 0.0, 0.0, -0.445], [0.0, -1.0, 0.0, 0.016], [0.0, 0.0, -1.0, 0.161], [0.0, 0.0, 0.0, 1.0]])
T_FLANGE_CAM = (np.linalg.inv(_T_bf_hov0) @ _T_bw_hov0).tolist()


def render_wrist_for_q(q_deg):
    """Render the synthetic wrist camera RGB frame + mask at arm joint pose q_deg."""
    T_bf = np.asarray(CS.fk(U, dict(zip(J, [math.radians(v) for v in q_deg])), "base_link", "link7"), dtype=float)
    T_bw = T_bf @ np.asarray(T_FLANGE_CAM, dtype=float)
    R_bw, t_bw = T_bw[:3, :3], T_bw[:3, 3]
    H, W = 720, 1280
    vv, uu = np.mgrid[0:H, 0:W].astype(float)
    dc = np.stack([(uu - K_WRIST["ppx"]) / K_WRIST["fx"], (vv - K_WRIST["ppy"]) / K_WRIST["fy"], np.ones_like(uu)], -1)
    D = dc @ R_bw.T
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = (TRUE_EGG_XYZ[2] - t_bw[2]) / D[..., 2]
    P = t_bw + lam[..., None] * D
    rel = P - TRUE_EGG_XYZ
    u_ax = np.array([math.cos(TRUE_EGG_YAW), math.sin(TRUE_EGG_YAW), 0.0])
    v_ax = np.array([-math.sin(TRUE_EGG_YAW), math.cos(TRUE_EGG_YAW), 0.0])
    in_egg = (lam > 0.02) & (lam < 1.0) & (((rel @ u_ax) / 0.0275) ** 2 + ((rel @ v_ax) / 0.021) ** 2 <= 1.0)
    # Add fine background texture so Laplacian sharpness > 25
    rgb = np.full((H, W, 3), 38, dtype=np.uint8)
    rgb[::2, ::2] = 52
    rgb[in_egg] = (238, 236, 230)
    ids = np.where(in_egg, 0, -1)
    return rgb, ids


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

    def depth_at(self, u, v):
        return float(DEPTH_M[int(v), int(u)])

    def roi_points(self, *a, **k):
        return [], 0

    def frames(self):
        self.fid += 1
        return RGB_M, DEPTH_M, self.fid

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
        self.cur_rgb, _ = render_wrist_for_q(Q_WORK)

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
        return self.cur_rgb, self.fid

    def frames(self):
        return self.cur_rgb, None, self.fid

    def display_ppm(self, view="color", size=(560, 315)):
        return None

    def detect_marker(self, marker, q_deg=None):
        return None


class Pipe:
    def __init__(self):
        self.q = list(Q_WORK)
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
    valid_ws = {"fence": {"polygon": [[-0.8, -0.4], [-0.1, -0.4], [-0.1, 0.4], [-0.8, 0.4]], "z_table": 0.0}}

    def arm_points(self, q):
        return []


root = tk.Tk()
root.geometry("1400x920")
frame = ttk.Frame(root)
frame.pack(fill="both", expand=True)
logs = []
pipe = Pipe()
jp = type("JP", (), {"enabled": True, "model": {"joints": J, "limits": [list(l) for l in LIM], "base_link": "base_link"}})()
cam = CAM.CameraTabUI(tk, ttk, frame, pipe, jp, Safe(), lambda: {"speed_percent": 10}, logs.append, lambda *a: True,
                      visible_fn=lambda: True, source_factory=FakeMain, wrist_source_factory=FakeWrist)
cam.calib = {"T_base_cam": F.T_BC.tolist(), "marker_in_flange": [0, 0, 0], "mean_mm": 1, "max_mm": 1,
             "loo_mean_mm": 1, "loo_max_mm": 1}
cam._set_vstate("ok", "刚标定")
cam.wrist_intr = {"K": dict(K_WRIST), "dist": [0.0] * 5, "model_type": W49.DIST_MODELS[1], "rms_px": 0.25, "good": True}
cam.wrist_calib = {"T_flange_cam": T_FLANGE_CAM, "marker_in_base": [-0.45, 0.0, 0.0],
                   "mean_mm": 0.6, "max_mm": 1.1, "loo_mean_mm": 0.8, "loo_max_mm": 1.3, "good": True}


def tick(n=3):
    for _ in range(n):
        cam.poll()
        root.update()


tick()
# ------------------------------------------------------------------ M1: Dual 16:9 monitor canvases (560x315 each)
assert hasattr(cam, "cv") and hasattr(cam, "cv_wrist") and hasattr(cam, "v_wrist_live")
assert int(cam.cv["width"]) == 560 and int(cam.cv["height"]) == 315
assert int(cam.cv_wrist["width"]) == 560 and int(cam.cv_wrist["height"]) == 315
assert "未连接" in cam.v_wrist_live.get()
cam.on_connect()
cam.on_wrist_connect()
tick(3)
assert "静止可采 ✓" in cam.v_wrist_live.get(), cam.v_wrist_live.get()
print(f"[M1] Dual 16:9 monitor canvases (cv={cam.cv['width']}x{cam.cv['height']}, cv_wrist={cam.cv_wrist['width']}x{cam.cv_wrist['height']}) active; v_wrist_live='{cam.v_wrist_live.get()}'")

# ------------------------------------------------------------------ M2: TCP crosshair & quality badge on cv_wrist
assert cam.wrist_quality and cam.wrist_quality["settled"] and cam.wrist_quality["status"] == "ok"
proj = W49.project_tcp_wrist(K_WRIST, T_FLANGE_CAM, cam._active_tcp_T(), z_work_m=0.10, canvas_size=(cam.DW, cam.DH))
assert proj and proj["ok"]
print(f"[M2] Wrist overlay active: TCP crosshair at canvas {proj['uv_tcp_cv']} px, 21 mm ring r={proj['radius_21mm_cv']} px")

# ------------------------------------------------------------------ M3: Motion gate blocks on_wrist_refine while arm is moving
cam.v_ydir.set(MD)
cam.v_ymodel.set("yoloe-26s-seg-pf.pt")
cam.v_method.set(CAM.LOCATE_METHODS[1])
F.STATE.scene = (OBS, IDS_M)
cam.on_eggs()
tick()
assert len(cam.eggs) == 1 and cam.eggs[0]["det_id"] == "egg_1"
# Simulate arm in motion (previous q differs by 1.2 deg from current q)
cam._prev_wrist_q = [pipe.q[0] + 1.2] + pipe.q[1:]
cam.on_wrist_refine()
assert any("腕部画面未停稳" in m for m in logs), logs[-3:]
assert cam.wrist_refine_res is None
print("[M3] Motion gate blocked on_wrist_refine while joint_step_deg > 0.15°")

# ------------------------------------------------------------------ M4: Coarse-to-Fine closed loop
# 1. Hover 100 mm above coarse egg_1
cam.v_hover_mm.set("100")
cam.on_hover_target()
tick(3)
assert len(pipe.moves) == 1
# Render wrist camera view at the hovered arm pose and update FakeYOLO scene for wrist resolution
w_rgb, w_ids = render_wrist_for_q(pipe.q)
cam.wrist_src.cur_rgb = w_rgb
F.STATE.scene = ([{"name": "E1", "label": "egg", "kind": "egg", "c": TRUE_EGG_XYZ.tolist(), "yaw": TRUE_EGG_YAW, "conf": 0.96}], w_ids)
tick(3)  # settles _prev_wrist_q == pipe.q and _prev_wrist_rgb == w_rgb

# 2. Run on_wrist_refine
cam.on_wrist_refine()
tick(2)
ref = cam.wrist_refine_res
assert ref and ref["ok"] and ref["trusted"] and ref["clearance_ok"], (ref, logs[-3:])
err_fine_mm = math.hypot(ref["fine_xyz_m"][0] - TRUE_EGG_XYZ[0], ref["fine_xyz_m"][1] - TRUE_EGG_XYZ[1]) * 1000.0
assert err_fine_mm < 0.3, (err_fine_mm, ref)
assert "腕部精定位完成" in cam.v_eggs.get() and "精定位 egg_1" in cam.v_wrist_live.get()
assert str(cam.b_walign.cget("state")) == "normal"

# 3. Run on_wrist_align ('✥ 一键微调对齐')
n_mv = len(pipe.moves)
cam.on_wrist_align()
tick(2)
assert len(pipe.moves) == n_mv + 1
assert any("已完成 egg_1 腕部精定位微调对齐" in m for m in logs), logs[-3:]
T_tcp_final = CS.tool_pose(U, J, [math.radians(v) for v in pipe.q], "base_link", cam._active_tcp_T())
tcp_final_xyz = CAM.T_t(T_tcp_final)
align_err_mm = math.hypot(tcp_final_xyz[0] - TRUE_EGG_XYZ[0], tcp_final_xyz[1] - TRUE_EGG_XYZ[1]) * 1000.0
assert align_err_mm < 0.5, (align_err_mm, tcp_final_xyz, TRUE_EGG_XYZ)
print(f"[M4] Coarse-to-Fine closed loop: coarse->hover->wrist refine (err {err_fine_mm:.3f} mm, Δx/Δy={ref['delta_tcp_mm']} mm)->fine align TCP (final xy err {align_err_mm:.3f} mm)")

cam.shutdown()
root.destroy()
print("ALL V3.50 GUI TESTS PASSED")
