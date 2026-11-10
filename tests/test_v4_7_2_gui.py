#!/usr/bin/env python3
"""V4.7.2 END-TO-END wrist calibration life cycle (needs tkinter + DISPLAY; skips otherwise).  Simulated NERO + wrist
camera that renders the tabletop ChArUco board at the arm's FK pose (from test_v3_57_gui), CameraTabUI patched by
NERO_V472_Wrist_Calib exactly as the console does.

E1 「自动采集」 board intrinsics + eye-in-hand -> both saved; the hand-eye file records target = board and the
   intrinsics it was made with.
E2 RESTART (a new CameraTabUI on the same folders) -> the SAME files are loaded (one log line each, no mismatch).
E3 「快速复核」 measures the board (not the ArUco marker) -> passes.
E4 the board is moved 30 mm -> the check still passes (the poses agree), says the board moved, updates the file;
   restart -> the updated position is loaded.
E5 a new version folder (config/ empty) -> the newest copies in cameracalib are loaded.
E6 「清除内参」 -> restart -> stays cleared (no old copy comes back); saving new intrinsics lifts it.
E7 intrinsics replaced by different ones -> warning that the hand-eye was made with other intrinsics.
E8 a USABLE (not "good") result asks to save (dialog) and saves on yes; a bad result never asks.
E9 the intrinsics planner uses the measured K; self-calibration is off while measured intrinsics are loaded.
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
    print("[GUI472] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI472] no DISPLAY -> skipped"); sys.exit(0)

import v347_fake as F                         # noqa: E402
import NERO_V315_Consistency_Safety as CS     # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49     # noqa: E402
import NERO_V386_Real_Wrist_Camera as W86     # noqa: E402
import NERO_V472_Wrist_Calib as WC            # noqa: E402
WC.patch_class(CAM.CameraTabUI)               # as the console does, BEFORE the tab is built

TMP = tempfile.mkdtemp(prefix="v472gui_")
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
BOARD_SHIFT = [0.0, 0.0]

aruco_dict = CAM._aruco_dict(cv2, "DICT_4X4_50")
if hasattr(cv2.aruco, "CharucoBoard"):
    BOARD = cv2.aruco.CharucoBoard((7, 5), 0.032, 0.024, aruco_dict)
    BOARD_IMG = BOARD.generateImage((700, 500))
else:
    BOARD = cv2.aruco.CharucoBoard_create(7, 5, 0.032, 0.024, aruco_dict)
    BOARD_IMG = BOARD.draw((700, 500))

HW, HH = 0.5 * 7 * 0.032, 0.5 * 5 * 0.032
# the camera looks DOWN (camera y = -base y): image top = board +y, else the board is rendered mirrored and no
# ArUco code of it can be read (this is why test_v3_57_gui found 0 samples)
OUTER_3D = np.array([[-HW, HH, 0.0], [HW, HH, 0.0], [HW, -HH, 0.0], [-HW, -HH, 0.0]], dtype=np.float64)
SRC_OUTER = np.float32([[-0.5, -0.5], [699.5, -0.5], [699.5, 499.5], [-0.5, 499.5]])


def render_board_at_q(q_deg):
    A = [list(map(float, r)) for r in CS.fk(U, dict(zip(J, [math.radians(v) for v in q_deg])), "base_link", "link7")]
    Yb = [list(r) for r in Y_BOARD]
    Yb[0][3] += BOARD_SHIFT[0]
    Yb[1][3] += BOARD_SHIFT[1]
    B = CAM.mat_mul(CAM.T_inv(CAM.mat_mul(A, X_TRUE)), Yb)
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



import glob                                   # noqa: E402
import json                                   # noqa: E402
from tkinter import messagebox                # noqa: E402

root = tk.Tk()
root.geometry("1400x920")
frame = ttk.Frame(root)
frame.pack(fill="both", expand=True)
logs = []
pipe = Pipe()
safe = Safe()
jp = type("JP", (), {"enabled": True, "model": {"joints": J, "limits": [list(l) for l in LIM], "base_link": "base_link"}})()
cams = []


def make_cam():
    """(re)start: a fresh Camera tab on the same folders"""
    for c in cams:
        try:
            c.shutdown()
        except Exception:
            pass
    for w in frame.winfo_children():
        w.destroy()
    c = CAM.CameraTabUI(tk, ttk, frame, pipe, jp, safe, lambda: {"speed_percent": 10}, logs.append, lambda *a: True,
                        visible_fn=lambda: True, source_factory=FakeMain, wrist_source_factory=FakeWrist)
    cams.append(c)
    return c


def tick(c, n=3):
    for _ in range(n):
        c.poll()
        root.update()


def lines(tag, since=0):
    return [l for l in logs[since:] if tag in l]


def jload(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ---- E1 board intrinsics + eye-in-hand, saved
cam = make_cam()
tick(cam, 2)
cam.on_wrist_connect()
tick(cam, 2)
cam.v_imodel.set(W49.DIST_MODELS[1])
cam.v_inauto.set("10")
cam.on_intr_auto()
tick(cam, 4)
assert cam.wrist_intr and os.path.isfile(W49.WRIST_INTR_PATH), logs[-25:]
assert os.path.isfile(W49.WRIST_HANDEYE_PATH), lines("eih")[-4:]
he = jload(W49.WRIST_HANDEYE_PATH)
assert he.get("target_type") == "charuco" and he.get("intrinsics_used") and he.get("board_cfg"), list(he)
assert abs(he["intrinsics_used"]["fx"] - cam.wrist_intr["K"]["fx"]) < 0.01
assert lines("[V4.7.2 intr] 结论") and lines("[V4.7.2 eih] 结论：腕部手眼标定良好")
copies = sorted(glob.glob(os.path.join(CAM.CALIB_DIR, "wrist_handeye_*.json")))
assert copies and jload(copies[-1]).get("target_type") == "charuco"
fx_saved, X_saved = cam.wrist_intr["K"]["fx"], he["T_flange_cam"]
print(f"[E1] board intrinsics (fx={fx_saved:.1f}) + eye-in-hand saved; file records target=board + intrinsics used")

# ---- E2 restart -> the same files
n0 = len(logs)
cam = make_cam()
tick(cam, 2)
assert cam.wrist_intr and abs(cam.wrist_intr["K"]["fx"] - fx_saved) < 0.01
assert cam.wrist_calib and cam.wrist_calib["T_flange_cam"] == X_saved
li = lines("[V4.7.2 wrist] 腕部内参 ←", n0)
lh = lines("[V4.7.2 wrist] 腕部手眼 ←", n0)
assert li and W49.WRIST_INTR_PATH in li[0] and lh and "参照 标定板" in lh[0], (li, lh)
assert not lines("另一组内参", n0)
print("[E2] restart: the saved intrinsics + hand-eye are loaded (log names both files)")

# ---- E3 quick check measures the board
cam.on_wrist_connect()
tick(cam, 2)
n0 = len(logs)
cam.on_wcalib_check()
tick(cam, 3)
ok = lines("腕部快速复核通过", n0)
assert ok and "看标定板" in ok[0], logs[n0:]
print(f"[E3] quick check on the board: {ok[0][ok[0].index('：') + 1:]}")

# ---- E4 board moved 30 mm
BOARD_SHIFT[0] = 0.030
n0 = len(logs)
cam.on_wcalib_check()
tick(cam, 3)
ok = lines("腕部快速复核通过", n0)
assert ok and "挪动了" in ok[0], logs[n0:]
mb = jload(W49.WRIST_HANDEYE_PATH)["marker_in_base"]
assert abs(mb[0] - (-0.450 + 0.030)) < 0.004, mb
cam = make_cam()
tick(cam, 2)
assert abs(cam.wrist_calib["marker_in_base"][0] - mb[0]) < 1e-9
cam.on_wrist_connect()
tick(cam, 2)
n0 = len(logs)
cam.on_wcalib_check()
tick(cam, 3)
assert lines("腕部快速复核通过", n0) and not lines("挪动了", n0)
print("[E4] board moved 30 mm: check passes, says it moved, file updated; after restart no further move")

# ---- E5 a new version folder: config/ empty -> cameracalib copies
for p in (W49.WRIST_INTR_PATH, W49.WRIST_HANDEYE_PATH):
    os.remove(p)
n0 = len(logs)
cam = make_cam()
tick(cam, 2)
assert cam.wrist_intr and abs(cam.wrist_intr["K"]["fx"] - fx_saved) < 0.01
assert cam.wrist_calib and cam.wrist_calib.get("target_type") == "charuco"
assert CAM.CALIB_DIR in lines("[V4.7.2 wrist] 腕部内参 ←", n0)[0]
print("[E5] new version folder: the newest copies in cameracalib are loaded")

# ---- E6 clear sticks across a restart
cam.on_wrist_connect()
tick(cam, 2)
cam.on_intr_clear_calib()
n0 = len(logs)
cam = make_cam()
tick(cam, 2)
assert cam.wrist_intr is None, cam.wrist_intr
assert lines("已被「清除」", n0)
cam.on_wrist_connect()
tick(cam, 2)
cam.v_imodel.set(W49.DIST_MODELS[1])
cam.on_intr_auto()
tick(cam, 4)
assert not os.path.isfile(WC.tombstone_path(W49.WRIST_INTR_PATH))
cam = make_cam()
tick(cam, 2)
assert cam.wrist_intr is not None
print("[E6] 清除内参 stays cleared after a restart; the next saved intrinsics are loaded again")

# ---- E7 intrinsics changed -> mismatch warning
d = jload(W49.WRIST_INTR_PATH)
d["K"]["fx"] *= 1.10
d["K"]["fy"] *= 1.10
with open(W49.WRIST_INTR_PATH, "w", encoding="utf-8") as f:
    json.dump(d, f)
n0 = len(logs)
cam = make_cam()
tick(cam, 2)
assert lines("另一组内参", n0), logs[n0:]
print("[E7] intrinsics 10 % different -> warned that the hand-eye was made with other intrinsics")

# ---- E8 usable -> dialog -> saved; bad -> no dialog
asked = []
messagebox.askyesno = lambda *a, **k: asked.append(a) or True
orig_cal = W86.calibrate_eih_4mp
good_r = dict(cam.wrist_calib)
for kind, patch in (("usable", {"good": False, "mean_mm": 1.4, "max_mm": 2.0, "loo_mean_mm": 2.96, "loo_max_mm": 5.07,
                                "warnings": []}),
                    ("bad", {"good": False, "mean_mm": 40.8, "max_mm": 58.9, "loo_mean_mm": 77.4, "loo_max_mm": 109.8,
                             "warnings": []})):
    r_fake = dict(good_r, ok=True, n_used=7, used=list(range(7)), rejected=[], **patch)
    r_fake["cam_in_flange"] = [row[3] for row in good_r["T_flange_cam"][:3]]
    W86.calibrate_eih_4mp = lambda samples, size_mm=24.0, **k: dict(r_fake)
    cam.wcalib_samples = [dict(source="charuco", q_deg=list(Q_WORK), p_cam=[0.0, 0.0, 0.25], spread_mm=0.1,
                               T_base_flange=A0_WORK, rvec=[0.0, 0.0, 0.0], attempt=1)] * 7
    asked.clear()
    n0 = len(logs)
    cam.on_wcalib_compute()
    time.sleep(0.5)
    tick(cam, 3)
    if kind == "usable":
        assert len(asked) == 1 and lines("结论：腕部手眼标定可以用", n0), (asked, logs[n0:])
        assert jload(W49.WRIST_HANDEYE_PATH)["loo_max_mm"] == 5.07
    else:
        assert asked == [] and lines("结论：腕部手眼标定不能用", n0)
W86.calibrate_eih_4mp = orig_cal
print("[E8] usable result -> asked and saved (5.07 mm, the user's case); bad result -> not offered")

# ---- E9 planner K + no self-calibration with measured intrinsics
Kp = W49.PLANNING_K_FN(1280, 720)
assert Kp and abs(Kp["fx"] - cam.wrist_intr["K"]["fx"]) < 0.01, Kp
seen = []
W86.calibrate_eih_4mp = lambda samples, size_mm=24.0, **k: seen.append(W86.SELF_CAL_ALLOWED) or {"ok": False, "error": "x"}
cam.on_wcalib_compute()
W86.calibrate_eih_4mp = orig_cal
assert seen == [False] and W86.SELF_CAL_ALLOWED is True
print("[E9] planner uses the measured fx; self-calibration off while measured intrinsics are loaded")

for c in cams:
    try:
        c.shutdown()
    except Exception:
        pass
root.destroy()
print("ALL V4.7.2 GUI TESTS PASSED")
