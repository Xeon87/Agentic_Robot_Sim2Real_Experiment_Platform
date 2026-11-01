#!/usr/bin/env python3
"""V3.48 GUI test: 清除标定设定 button, 识别物体 (by name / -pf built-in vocabulary) alongside 识别鸡蛋, TCP calibration,
and hover verification (needs tkinter + a display; skips otherwise).
All paths point to a temp folder (never the user's ~/Downloads/cameracalib).

K1 手眼标定 '清除标定设定': clears active calibration + removes config/camera_calib.json (keeps ~/Downloads/cameracalib
   backups), resets lamps ③/④ to grey '还没有标定'
K2 视觉定位 '识别物体（按名称/词库）' + '识别物体名称': detects named objects ('cup, bottle') via YOLOE -pf / text prompt
   and fills tv_eggs ('obj_1', 'obj_2'); '识别鸡蛋' continues to work ('egg_1', 'egg_2')
K3 TCP 标定: recording 6 single-point touch poses -> '计算并保存 TCP' saves tcp_calib.json and updates status; '恢复默认 TCP'
   restores URDF default (+155.3, +0.0, -21.5 mm)
K4 悬停复核: '▶ 走到选中目标上方悬停' moves the robot TCP 100 mm above the selected target (< 1 mm error) and '↩ 回悬停前姿态'
   returns to the pre-hover pose
"""
import math
import os
import random
import sys
import tempfile
import threading
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    print("[GUI48] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI48] no DISPLAY -> skipped"); sys.exit(0)
import numpy as np                            # noqa: E402
import v347_fake as F                         # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V347_Yolo_Perception as YP        # noqa: E402
import NERO_V315_Consistency_Safety as CS     # noqa: E402

TMP = tempfile.mkdtemp(prefix="v348gui_")
CFG = os.path.join(TMP, "config")
os.makedirs(CFG)
CAM.CALIB_PATH = CAM.DEFAULT_CALIB_PATH = os.path.join(CFG, "camera_calib.json")
CAM.LOCATE_PATH = CAM.DEFAULT_LOCATE_PATH = os.path.join(CFG, "locate_settings.json")
CAM.TCP_CALIB_PATH = CAM.DEFAULT_TCP_CALIB_PATH = os.path.join(CFG, "tcp_calib.json")
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

OBS_EGGS = [o for o in F.default_objects() if o["name"] in ("E1", "E2")]
OBS_OBJS = [
    {"name": "CUP", "label": "cup", "kind": "box", "c": [-0.44, 0.05, 0.040], "yaw": 0.1,
     "half": [0.035, 0.035, 0.040], "conf": 0.92},
    {"name": "BOT", "label": "bottle", "kind": "box", "c": [-0.38, -0.09, 0.055], "yaw": -0.2,
     "half": [0.030, 0.030, 0.055], "conf": 0.89},
    {"name": "BOX", "label": "box", "kind": "box", "c": [-0.52, -0.06, 0.025], "yaw": 0.2,
     "half": [0.040, 0.030, 0.025], "conf": 0.85},
]
CUR = {"rgb": None, "depth": None}


def set_scene(obs):
    rgb, depth, ids = F.render(obs)
    CUR["rgb"], CUR["depth"] = rgb, depth
    F.STATE.scene = (obs, ids)


set_scene(OBS_OBJS)


class Fake:
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
        return float(CUR["depth"][int(v), int(u)])

    def roi_points(self, *a, **k):
        return [], 0

    def frames(self):
        self.fid += 1
        return CUR["rgb"], CUR["depth"], self.fid

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


CALIB = {"version": "V3.25", "saved": "2026-10-26 21:14:25", "camera_serial": "261222075224", "intrinsics": dict(F.K),
         "marker": {"dict": "DICT_4X4_50", "id": 0, "size_mm": 38.0}, "flange_link": "link7",
         "T_base_cam": F.T_BC.tolist(), "marker_in_flange": [0.0877, -0.0696, -0.0323], "mean_mm": 0.95, "max_mm": 1.8,
         "loo_mean_mm": 1.8, "loo_max_mm": 4.2, "n_used": 8, "check_poses_deg": [[0] * 7, [1] * 7], "samples": []}
backup_path = os.path.join(CAM.CALIB_DIR, "camera_calib_20261026_211425.json")
CAM.save_json(backup_path, CALIB)
CAM.save_json(CAM.CALIB_PATH, CALIB)

root = tk.Tk()
root.geometry("1400x900")
frame = ttk.Frame(root)
frame.pack(fill="both", expand=True)
logs = []
pipe = Pipe()
jp = type("JP", (), {"enabled": True, "model": {"joints": J, "limits": [list(l) for l in LIM], "base_link": "base_link"}})()
cam = CAM.CameraTabUI(tk, ttk, frame, pipe, jp, Safe(), lambda: {"speed_percent": 10}, logs.append, lambda *a: True,
                      visible_fn=lambda: True, source_factory=Fake)


def tick(n=3):
    for _ in range(n):
        cam.poll()
        root.update()


tick()
# ------------------------------------------------------------------ K1
assert cam.calib is not None and str(cam.b_clear_calib.cget("state")) == "normal"
cam.on_calib_clear()
tick()
assert cam.calib is None and cam.result is None and not os.path.isfile(CAM.CALIB_PATH)
assert os.path.isfile(backup_path), "backup in ~/Downloads/cameracalib must not be deleted"
assert getattr(cam.l3, "_v345_state") == "grey" and "还没有标定" in cam.s3.get()
assert str(cam.b_clear_calib.cget("state")) == "disabled"
assert any("已清除当前手眼标定设定" in m for m in logs)
cam.on_calib_load(backup_path)
for e in (1.0,) * 5:
    cam._verify_live(object(), e)
tick()
assert cam.calib is not None and cam.vstate["state"] == "ok"
print("[K1] 清除标定设定: clears active calib + removes config/camera_calib.json (keeps backup), lamps grey; reloaded & verified")

# ------------------------------------------------------------------ K2
cam.v_ydir.set(MD)
cam.on_connect()
tick(4)
cam.v_ymodel.set("yoloe-26s-seg-pf.pt")
cam.v_obj_names.set("cup, bottle")
cam.on_objects()
tick()
rows_obj = [cam.tv_eggs.item(i, "values") for i in cam.tv_eggs.get_children()]
assert [r[0] for r in rows_obj] == ["obj_1", "obj_2"] and all(r[8] == "✓" for r in rows_obj), rows_obj
assert "cup" in rows_obj[0][9] and "bottle" in rows_obj[1][9]
assert [v["label"] for v in cam.yolo_view] == ["o1:cup", "o2:bottle"]
set_scene(OBS_EGGS)
cam.v_ymodel.set("yoloe-26s-seg.pt")
cam.on_ref_toggle()
cam.ref_edit["boxes"], cam.ref_edit["cls"] = [[540.0, 380.0, 600.0, 440.0]], [0]
cam.on_ref_toggle()
cam.on_eggs()
tick()
rows_egg = [cam.tv_eggs.item(i, "values") for i in cam.tv_eggs.get_children()]
assert [r[0] for r in rows_egg] == ["egg_1", "egg_2"] and all(r[8] == "✓" for r in rows_egg), rows_egg
print("[K2] 识别物体 ('cup, bottle' via -pf vocabulary -> obj_1, obj_2) + 识别鸡蛋 (egg_1, egg_2) both work in 视觉定位 tab")

# ------------------------------------------------------------------ K3
assert "URDF 默认" in cam.v_tcp_status.get()
true_tcp = [0.153, -0.002, -0.020]
touch_pt = [-0.42, 0.03, 0.015]
T0 = CS.fk(U, dict(zip(J, [math.radians(v) for v in Q_WORK])), "base_link", "link7")
R0 = [list(map(float, r[:3])) for r in T0[:3]]
rnd = random.Random(9)
for degs in ((0, 0, 0), (20, 15, -25), (-20, -15, 25), (15, -22, 20), (-18, 20, -28), (24, 10, 32)):
    R = CAM.mat_mul(CAM.rodrigues([math.radians(d) for d in degs]), R0)
    rt = CAM.mat_vec(R, true_tcp)
    tf = [touch_pt[i] - rt[i] + rnd.gauss(0, 0.0002) for i in range(3)]
    cam.flange_T = lambda _q, T_fixed=CAM.T_from(R, tf): T_fixed
    cam.on_tcp_record()
del cam.flange_T
tick()
assert len(cam.tcp_samples) == 6 and str(cam.b_tcp_comp.cget("state")) == "normal"
cam.on_tcp_compute()
tick()
assert cam.tcp_calib is not None and os.path.isfile(CAM.TCP_CALIB_PATH) and "已标定" in cam.v_tcp_status.get()
assert CAM.vnorm(CAM.vsub(cam.tcp_calib["tcp_in_flange"], true_tcp)) * 1000.0 < 1.5
cam.on_tcp_reset()
tick()
assert cam.tcp_calib is None and not os.path.isfile(CAM.TCP_CALIB_PATH) and "URDF 默认" in cam.v_tcp_status.get()
print("[K3] TCP 标定: 6 touch poses -> 计算并保存 TCP (< 1.5 mm error) -> 恢复默认 TCP")

# ------------------------------------------------------------------ K4
pipe.q = list(Q_WORK)
n_mv0 = len(pipe.moves)
cam.v_hover_mm.set("100")
cam.on_hover_target()
tick()
assert len(pipe.moves) == n_mv0 + 1 and cam._pre_hover_q is not None
assert any("[OK] [V3.48 hover] 已悬停在 egg_1 上方 100 mm" in m for m in logs), logs[-3:]
T_tcp = CS.tool_pose(U, J, [math.radians(v) for v in pipe.q], "base_link", cam._active_tcp_T())
e1_xyz = cam.eggs[0]["xyz_m"]
hover_err = CAM.vnorm(CAM.vsub(CAM.T_t(T_tcp), [e1_xyz[0], e1_xyz[1], e1_xyz[2] + 0.10])) * 1000.0
assert hover_err < 1.0, hover_err
cam.on_hover_back()
tick()
assert len(pipe.moves) == n_mv0 + 2 and cam._pre_hover_q is None
assert max(abs(a - b) for a, b in zip(pipe.q, Q_WORK)) < 1e-4
print(f"[K4] 悬停复核: moved 100 mm above egg_1 (TCP err {hover_err:.2f} mm), then returned to pre-hover pose")

cam.shutdown()
root.destroy()
print("ALL V3.48 GUI TESTS PASSED")
