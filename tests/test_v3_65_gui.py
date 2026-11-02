#!/usr/bin/env python3
"""V3.65 GUI test (needs tkinter + DISPLAY; skips otherwise):
R1 Real Tk: the 「NERO实机任务」 tab builds with fake pipeline / Camera tab; the status line shows the six readiness
   lamps; the skill box shows [CS] + the SO-101 profile.
R2 Gripper panel on the simulated STS3215: connect, 「张开 32 mm」 -> 32 mm tip gap, 「夹紧测试」 on a 2 mm wall ->
   held; the readout updates; worker threads never touch Tk (no "main thread is not in main loop" error).
R3 Both monitors draw the camera frames (fixed D435i + wrist) with the wrist cross hair; mode ① on frames without
   cups stops at observe_scene with CUP_NOT_PERCEIVED and no motion.
R4 Settings go to NERO_CALIB_DIR/real_cup_task.json; nothing inside the release folder changed.
"""
import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import tkinter as tk
    from tkinter import ttk
    import numpy as np
    if not os.environ.get("DISPLAY"):
        print("[SKIP] test_v3_65_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v3_65_gui: {exc}")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="nero_v365_gui_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
os.environ["NERO_V368_TASK_DIR"] = str(TMP / "task")          # V3.68: task records never in the release folder
H = lambda b: hashlib.sha256(b).hexdigest()
SNAP0 = {str(p.relative_to(ROOT)): H(p.read_bytes()) for p in ROOT.rglob("*")
         if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and "logs" not in p.parts}

import NERO_V325_Camera as CAM               # noqa: E402
import NERO_V365_Real_Cup_Task as RT         # noqa: E402
import NERO_V365_STS3215_Gripper as GR       # noqa: E402


class Src:
    K = {"width": 1280, "height": 720, "fx": 900.0, "fy": 900.0, "ppx": 640.0, "ppy": 360.0}

    def __init__(self, shade):
        self.fid, self.shade = 0, shade

    def running(self):
        return True

    def frames(self):
        self.fid += 1
        return np.zeros((720, 1280, 3), np.uint8) + 40, np.ones((720, 1280), np.float32), self.fid

    def display_ppm(self, view="color", size=(560, 315)):
        return CAM.ppm_bytes(np.zeros((size[1], size[0], 3), np.uint8) + self.shade)


class _Det:                                   # V3.66: the Camera tab's YOLOE detector (sees no cup here)
    info = {"device": "CPU"}

    def detect(self, rgb, mode, path, ref=None, prompts=None, conf=0.25):
        return []


class _V:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v


MODEL_DIR = TMP / "objectmodels"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
(MODEL_DIR / "yoloe-26s-seg.pt").write_bytes(b"fake")


class Cam:
    result, worker = None, None
    calib = {"T_base_cam": np.eye(4).tolist()}

    def __init__(self):
        self.src, self.wrist_src = Src(90), Src(60)
        self.det_yolo = _Det()
        self.v_ydir, self.v_ymodel, self.v_yconf = _V(str(MODEL_DIR)), _V("yoloe-26s-seg.pt"), _V("0.25")

    def _fence(self):
        return {"z_table": 0.0, "polygon": [[-0.8, -0.4], [-0.1, -0.4], [-0.1, 0.4], [-0.8, 0.4]]}

    def _calib_note(self):
        return None

    def on_connect(self):
        pass

    def on_wrist_connect(self):
        pass


class Pipe:
    teach_mode = False

    def lamp(self, k):
        return "green"

    def supports_joint_path(self):
        return True

    def actual_arm_deg(self, m):
        return [0.0] * 7

    def move_busy(self):
        return False

    def cancel_move(self):
        pass


class JP:
    model = {"joints": [f"joint{i}" for i in range(1, 8)], "limits": [(-3.0, 3.0)] * 7}


root = tk.Tk()
root.geometry("1500x980")
nb = ttk.Notebook(root)
nb.pack(fill="both", expand=True)
fr = ttk.Frame(nb, padding=6)
nb.add(fr, text="NERO实机任务")
sim_servo = GR.SimServo()
grip = GR.SO101RealGripper(log=lambda s: None)
_orig_connect = grip.connect
grip.connect = lambda port=None, baud=None, servo_id=None, sim_servo_=None: _orig_connect(port, baud, servo_id, sim_servo=sim_servo)
cam = Cam()
tab = RT.RealCupTaskTabUI(tk, ttk, fr, Pipe(), JP(), cam, log=lambda s: None, ask=lambda a, b: False, gripper=grip)


def pump(sec):
    t_end = time.time() + sec
    while time.time() < t_end:
        root.update()
        time.sleep(0.03)


def wait_idle(timeout=20):
    t_end = time.time() + timeout
    while tab.busy() and time.time() < t_end:
        pump(0.1)
    pump(0.3)


pump(0.5)
st = tab.v_status.get()
for k in ("机械臂 MoveIt", "关节路径执行", "主相机 D435i", "手眼标定", "腕部相机", "SO-101 夹爪"):
    assert k in st, st
assert "[CS]" in tab.v_skill.get() and "偏转 -45°" in tab.v_skill.get()
print("[R1] tab builds; readiness lamps + CS skill / SO-101 profile shown")

tab.v_port.set(GR.SIM_PORT)
tab.on_grip_connect()
wait_idle()
assert grip.connected and "指尖缝" in tab.v_grip.get(), tab.v_grip.get()
tab.on_grip("gap", 32.0)
wait_idle()
assert abs(grip.status()["gap_mm"] - 32.0) < 0.8 and "32" in tab.v_grip.get()
sim_servo.obstacle_gap_mm = 2.0
tab.on_grip("clamp", None)
wait_idle()
assert grip.held and "夹持中" in tab.v_grip.get(), tab.v_grip.get()
log = tab.txt.get("1.0", "end")
assert "main thread is not in main loop" not in log and "[FAIL]" not in log, log
print("[R2] gripper panel: connect, open 32 mm, clamp on a 2 mm wall -> held; no Tk access from workers")

pump(0.6)
assert tab.cv.find_all() and tab.cvw.find_all(), "monitors empty"
assert len(tab.cvw.find_all()) >= 3                     # image + cross hair
tab.v_mode.set("plan")
tab.on_run()
wait_idle(30)
log = tab.txt.get("1.0", "end")
assert "mode=plan" in log and "CUP_NOT_PERCEIVED" in log and "no cup" in log, log[-800:]
print("[R3] both monitors draw; mode ① without cups stops at observe_scene (CUP_NOT_PERCEIVED), no motion")

shot = os.environ.get("NERO_V365_GUI_SHOT")
if shot:
    os.system(f"import -window root {shot} >/dev/null 2>&1")
tab.shutdown()
root.destroy()
assert (TMP / "cameracalib" / "real_cup_task.json").is_file()
SNAP1 = {str(p.relative_to(ROOT)): H(p.read_bytes()) for p in ROOT.rglob("*")
         if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and "logs" not in p.parts}
assert SNAP1 == SNAP0, sorted(set(SNAP0.items()) ^ set(SNAP1.items()))[:6]
print("[R4] settings in NERO_CALIB_DIR; release folder unchanged")
print("ALL V3.65 GUI TESTS PASSED")
