#!/usr/bin/env python3
"""V3.49 GUI test: wrist camera info in 相机信息 tab, new 内参标定 tab after 深度检查, 手眼标定 split into
'主相机手眼标定' and 'eye-in-hand 手眼标定功能' sub-tabs, and zero-prior '识别蛋托（Agent 算槽位）' + hover over slot
(needs tkinter + a display; skips otherwise).
All paths point to a temp folder (never the user's ~/Downloads/cameracalib).

L1 Tabs order ['相机信息', '深度检查', '内参标定', '手眼标定', '视觉定位'];手眼标定 sub-tabs ['主相机手眼标定',
   'eye-in-hand 手眼标定功能']; wrist camera info below main camera info in 相机信息 tab
L2 内参标定 tab: 生成标定板打印图 (PNG), 拍一张 (corner detection), 计算内参 -> RMS < 0.5 px (合格), 保存内参 / 读取 / 清除内参
L3 eye-in-hand 手眼标定功能 sub-tab: 记录当前腕部姿态 (10 poses) -> 计算腕部手眼 (< 1.5 mm) -> 保存腕部标定 / 读取 / 清除
L4 视觉定位 '识别蛋托（Agent 算槽位）': autonomously infers 2x3 holder + slot_1..slot_6 without user size input; selecting
   slot_1 and clicking '▶ 走到选中目标上方悬停' moves the TCP 100 mm above slot_1 (< 1 mm error)
"""
import math
import os
import random
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
    print("[GUI49] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI49] no DISPLAY -> skipped"); sys.exit(0)
import v347_fake as F                         # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V347_Yolo_Perception as YP        # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49     # noqa: E402
import NERO_V315_Consistency_Safety as CS     # noqa: E402

TMP = tempfile.mkdtemp(prefix="v349gui_")
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

# Render a 2x3 egg holder scene (144 x 96 x 36 mm at (-0.46, 0.04), yaw=20 deg)
YAW_H = math.radians(20.0)
OBS_HOLDER = [{"name": "HOLDER", "label": "egg holder", "kind": "box", "c": [-0.46, 0.04, 0.018], "yaw": YAW_H,
               "half": [0.072, 0.048, 0.018], "conf": 0.95}]
F.STATE.bleed_px = 0
RGB_H, DEPTH_H, IDS_H = F.render(OBS_HOLDER, noise_m=0.0004)
F.STATE.scene = (OBS_HOLDER, IDS_H)

# Synthetic chessboard image for wrist camera intrinsic capture
sq_px = 50
bw, bh = 8 * sq_px, 6 * sq_px
b_img = np.full((bh + 80, bw + 80), 255, dtype=np.uint8)
for r in range(6):
    for c in range(8):
        if (r + c) % 2 == 0:
            b_img[40 + r * sq_px:40 + (r + 1) * sq_px, 40 + c * sq_px:40 + (c + 1) * sq_px] = 0
H_m = cv2.getPerspectiveTransform(
    np.float32([[40, 40], [40 + bw, 40], [40 + bw, 40 + bh], [40, 40 + bh]]),
    np.float32([[280, 140], [920, 160], [900, 560], [260, 540]])
)
WRIST_RGB = cv2.cvtColor(cv2.warpPerspective(b_img, H_m, (1280, 720), borderValue=200), cv2.COLOR_GRAY2RGB)
WRIST_DET = {"v": None}


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

    def display_ppm(self, view="color"):
        return None

    def depth_at(self, u, v):
        return float(DEPTH_H[int(v), int(u)])

    def roi_points(self, *a, **k):
        return [], 0

    def frames(self):
        self.fid += 1
        return RGB_H, DEPTH_H, self.fid

    def detect_marker(self, marker, q_deg=None):
        return None


class FakeWrist:
    def __init__(self):
        self.K = None
        self.dist = [0.0] * 5
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
            self.model_type = intr.get("model_type") or W49.DIST_MODELS[0]
        else:
            self.K = None

    def running(self):
        return self.on

    def start(self):
        self.on = True
        self.info = {"name": "Innomaker 1080P USB2.0 (fake)", "device": "/dev/video4", "format": "MJPG",
                     "res": "1920×1080", "width": 1920, "height": 1080,
                     "controls": {"autofocus": "off", "auto_exposure": "locked(manual)", "auto_wb": "locked"}}

    def stop(self, timeout=0):
        self.on = False

    def raw_frame(self):
        self.fid += 1
        return WRIST_RGB, self.fid

    def frames(self):
        self.fid += 1
        return WRIST_RGB, None, self.fid

    def display_ppm(self, view="color"):
        return None

    def detect_marker(self, marker, q_deg=None):
        return WRIST_DET["v"]


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
root.geometry("1400x900")
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


def tick(n=3):
    for _ in range(n):
        cam.poll()
        root.update()


tick()
# ------------------------------------------------------------------ L1
top_tabs = [cam.nb.tab(t, "text") for t in cam.nb.tabs()]
assert top_tabs == ["相机信息", "深度检查", "内参标定", "手眼标定", "视觉定位"], top_tabs
sub_tabs = [cam.nb_calib.tab(t, "text") for t in cam.nb_calib.tabs()]
assert sub_tabs in (["主相机手眼标定", "eye-in-hand 手眼标定功能"], ["主相机手眼标定", "eye-in-hand手眼标定"]), sub_tabs
assert "腕部相机未连接" in cam.v_wrist_info.get()
cam.on_wrist_connect()
tick()
assert "Innomaker 1080P" in cam.v_wrist_info.get() and "/dev/video4" in cam.v_wrist_info.get()
cam.on_save_wrist_info()
assert os.path.isfile(W49.WRIST_INFO_PATH)
print("[L1] Tabs ['相机信息', '深度检查', '内参标定', '手眼标定', '视觉定位']; 手眼标定 sub-tabs ['主相机手眼标定', 'eye-in-hand 手眼标定功能']; wrist info below main info")

# ------------------------------------------------------------------ L2
png_out = cam.on_intr_board_png(os.path.join(TMP, "test_board.png"))
assert png_out and os.path.isfile(png_out)
cam.v_ibtype.set(W49.BOARD_TYPES[1])
cam.v_icols.set("7")
cam.v_irows.set("5")
cam.v_isq.set("25")
cam.on_intr_capture()
assert len(cam.intr_samples) == 1 and cam.intr_samples[0]["n_corners"] == 35
# Populate 9 varied synthetic board views to run on_intr_compute -> on_intr_save
Km_t = np.array([[820.0, 0.0, 640.0], [0.0, 820.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
Dm_t = np.array([-0.16, 0.03, 0.0, 0.0, 0.0], dtype=np.float64)
g_obj = np.zeros((35, 3), dtype=np.float32)
g_uv = np.mgrid[0:7, 0:5].T.reshape(-1, 2)
g_obj[:, 0], g_obj[:, 1] = g_uv[:, 0] * 0.025, g_uv[:, 1] * 0.025
rng = np.random.default_rng(21)
cam.on_intr_clear()
for k in range(9):
    rv = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-0.2, 0.2)], dtype=np.float64)
    cx_f, cy_f = 0.22 + (k % 3) * 0.28, 0.22 + (k // 3) * 0.28
    tv = np.array([(cx_f - 0.5) * 0.30 - 0.08, (cy_f - 0.5) * 0.18 - 0.05, rng.uniform(0.32, 0.46)], dtype=np.float64)
    proj, _ = cv2.projectPoints(g_obj, rv, tv, Km_t, Dm_t)
    p2 = proj.reshape(-1, 2)
    cam.intr_samples.append({"ok": True, "obj_pts": g_obj.tolist(), "img_pts": p2.tolist(), "n_corners": 35,
                             "center_frac": [float(p2[:, 0].mean() / 1280.0), float(p2[:, 1].mean() / 720.0)],
                             "coverage": "中+四角", "sharpness": 400.0, "image_size": [1280, 720]})
cam.v_imodel.set(W49.DIST_MODELS[1])
cam.on_intr_compute()
assert cam.intr_result and cam.intr_result["good"] and cam.intr_result["rms_px"] < 0.3
cam.on_intr_save()
assert cam.wrist_intr and os.path.isfile(W49.WRIST_INTR_PATH) and cam.wrist_src.K is not None
print(f"[L2] 内参标定 tab: PNG board generated, 9 views -> RMS {cam.wrist_intr['rms_px']:.3f} px (合格) -> saved & applied to wrist camera")

# ------------------------------------------------------------------ L3
X_true = CAM.T_from(CAM.rodrigues([0.08, -0.18, 1.55]), [0.042, -0.015, 0.032])
Y_true = CAM.T_from(CAM.rodrigues([0.01, -0.02, 0.35]), [-0.44, 0.03, 0.0])
qs = CAM.sample_targets(Q_WORK, LIM, 10, seed=12)
for q in qs:
    pipe.q = list(q)
    A = [list(map(float, r)) for r in CS.fk(U, dict(zip(J, [math.radians(v) for v in q])), "base_link", "link7")]
    B = CAM.mat_mul(CAM.T_inv(CAM.mat_mul(A, X_true)), Y_true)
    rv_m, _ = cv2.Rodrigues(np.array(CAM.T_R(B), dtype=np.float64))
    WRIST_DET["v"] = {"id": 0, "p_cam": CAM.T_t(B), "rvec": rv_m.flatten().tolist(), "corners_px": [[0, 0]] * 4,
                      "center_px": [640, 360], "side_px": 60}
    cam.on_wcalib_record()
assert len(cam.wcalib_samples) == 10
cam.on_wcalib_compute()
assert cam.wcalib_result and cam.wcalib_result["good"]
err_w = CAM.vnorm(CAM.vsub(cam.wcalib_result["cam_in_flange"], CAM.T_t(X_true))) * 1000.0
assert err_w < 1.0, err_w
cam.on_wcalib_save()
assert cam.wrist_calib and os.path.isfile(W49.WRIST_HANDEYE_PATH)
cam.on_wcalib_clear_calib()
assert cam.wrist_calib is None and not os.path.isfile(W49.WRIST_HANDEYE_PATH)
print(f"[L3] eye-in-hand 手眼标定功能 sub-tab: 10 poses -> T_flange_cam err {err_w:.2f} mm (合格) -> saved -> cleared")

# ------------------------------------------------------------------ L4
pipe.q = list(Q_WORK)
cam.v_ydir.set(MD)
cam.v_ymodel.set("yoloe-26s-seg-pf.pt")
cam.on_connect()
tick(4)
cam.on_holder()
tick()
rows_h = [cam.tv_eggs.item(i, "values") for i in cam.tv_eggs.get_children()]
assert [r[0] for r in rows_h] == ["holder", "slot_1", "slot_2", "slot_3", "slot_4", "slot_5", "slot_6"], rows_h
assert "2×3（6 槽）" in rows_h[0][9] and "蛋托自主识别 · 2×3" in cam.v_eggs.get()
assert [v["label"] for v in cam.yolo_view] == ["蛋托", "s1", "s2", "s3", "s4", "s5", "s6"]
# Hover over slot_1 (first trusted item after holder or select slot_1)
slot1_item_id = cam.tv_eggs.get_children()[1]
cam.tv_eggs.selection = lambda: (slot1_item_id,)
cam.v_hover_mm.set("100")
n_mv0 = len(pipe.moves)
cam.on_hover_target()
tick()
assert len(pipe.moves) == n_mv0 + 1
assert any("[OK] [V3.48 hover] 已悬停在 slot_1 上方 100 mm" in m for m in logs), logs[-3:]
s1_xyz = cam.eggs[1]["xyz_m"]
T_tcp = CS.tool_pose(U, J, [math.radians(v) for v in pipe.q], "base_link", cam._active_tcp_T())
s1_err = CAM.vnorm(CAM.vsub(CAM.T_t(T_tcp), [s1_xyz[0], s1_xyz[1], s1_xyz[2] + 0.10])) * 1000.0
assert s1_err < 1.0, s1_err
print(f"[L4] 识别蛋托（Agent 算槽位）: zero-prior 2x3 holder + slot_1..slot_6 located; hovered 100 mm above slot_1 (err {s1_err:.2f} mm)")

cam.shutdown()
root.destroy()
print("ALL V3.49 GUI TESTS PASSED")
