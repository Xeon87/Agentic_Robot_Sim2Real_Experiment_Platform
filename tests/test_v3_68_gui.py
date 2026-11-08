#!/usr/bin/env python3
"""V3.68 GUI test (needs tkinter + DISPLAY; skips otherwise):
V1 Real Tk: the main monitor draws the model fingertips (「F 固定指」 / 「M 动指」 + jaw line) for the arm pose and the
   mount in the fields; changing 「偏转 °」 moves them.
V2 Mode ① run: the task record folder (NERO_V368_TASK_DIR) gets before / after images of observe_scene, task.json and
   summary.txt; the log has the "[V3.68 STEP 1]" line and the "[REAL TASK SUMMARY RED]" block (no cups); the
   perception log line carries the stage timings.
V3 Nothing inside the release folder changed.
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
        print("[SKIP] test_v3_68_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v3_68_gui: {exc}")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="nero_v368_gui_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
os.environ["NERO_V368_TASK_DIR"] = str(TMP / "task")
H = lambda b: hashlib.sha256(b).hexdigest()
SNAP0 = {str(p.relative_to(ROOT)): H(p.read_bytes()) for p in ROOT.rglob("*")
         if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and "logs" not in p.parts}

import NERO_V325_Camera as CAM               # noqa: E402
import NERO_V365_Real_Cup_Task as RT         # noqa: E402

KK = {"width": 1280, "height": 720, "fx": 900.0, "fy": 900.0, "ppx": 640.0, "ppy": 360.0}
T_BC = np.array([[0.0, -1.0, 0.0, -0.30], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.95], [0, 0, 0, 1.0]])


class Src:
    K = KK

    def __init__(self, shade):
        self.fid, self.shade = 0, shade

    def running(self):
        return True

    def frames(self):
        self.fid += 1
        return np.zeros((720, 1280, 3), np.uint8) + self.shade, np.full((720, 1280), 0.9, np.float32), self.fid

    def display_ppm(self, view="color", size=(560, 315)):
        return CAM.ppm_bytes(np.zeros((size[1], size[0], 3), np.uint8) + self.shade)


class Det:
    info = {"device": "CPU"}

    def detect(self, rgb, mode, path, ref=None, prompts=None, conf=0.25):
        return []


class _V:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v


MD = TMP / "objectmodels"
MD.mkdir(parents=True)
(MD / "yoloe-26s-seg.pt").write_bytes(b"fake")


class Cam:
    result, worker = None, None
    calib = {"T_base_cam": T_BC.tolist()}

    def __init__(self):
        self.src, self.wrist_src = Src(80), Src(50)
        self.det_yolo = Det()
        self.v_ydir, self.v_ymodel, self.v_yconf = _V(str(MD)), _V("yoloe-26s-seg.pt"), _V("0.25")

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
        return [0.0, 20.0, 0.0, 70.0, 0.0, 30.0, 0.0]

    def move_busy(self):
        return False

    def cancel_move(self):
        pass


class JP:
    model = {"joints": [f"joint{i}" for i in range(1, 8)], "limits": [(-3.0, 3.0)] * 7}


root = tk.Tk()
root.geometry("1500x1000")
nb = ttk.Notebook(root)
nb.pack(fill="both", expand=True)
fr = ttk.Frame(nb, padding=6)
nb.add(fr, text="NERO实机任务")
tab = RT.RealCupTaskTabUI(tk, ttk, fr, Pipe(), JP(), Cam(), log=lambda s: None, ask=lambda a, b: True)


def pump(sec):
    t_end = time.time() + sec
    while time.time() < t_end:
        root.update()
        time.sleep(0.03)


def wait_idle(timeout=30):
    t_end = time.time() + timeout
    while tab.busy() and time.time() < t_end:
        pump(0.1)
    pump(0.4)


def tip_xy():
    out = {}
    for i in tab.cv.find_all():
        if tab.cv.type(i) == "text" and tab.cv.itemcget(i, "text") in ("F 固定指", "M 动指"):
            out[tab.cv.itemcget(i, "text")] = tuple(tab.cv.coords(i))
    return out


pump(0.8)
a = tip_xy()
assert set(a) == {"F 固定指", "M 动指"}, a
assert any(tab.cv.type(i) == "line" for i in tab.cv.find_all())
tab.v_roll.set("45")
tab._shown = (-1, -1)
pump(0.6)
b = tip_xy()
assert set(b) == set(a) and b != a, (a, b)
print("[V1] fingertip overlay F / M + jaw line drawn; the roll field moves them")

tab.v_roll.set("-45")
tab.v_mode.set("plan")
tab.on_run()
wait_idle()
log = tab.txt.get("1.0", "end")
assert "[V3.68 STEP 1] observe_scene" in log and "[REAL TASK SUMMARY RED]" in log and "LAST FAILURE:" in log, log[-1500:]
assert "（YOLOE" in log or "（帧" in log, log[-1500:]
dirs = [d for d in (TMP / "task").iterdir() if d.is_dir()]
assert len(dirs) == 1, dirs
files = sorted(os.listdir(dirs[0]))
assert any("_before_observe_scene" in f for f in files) and any("_after_observe_scene" in f for f in files), files
assert "summary.txt" in files and "task.json" in files and "steps.jsonl" in files
print(f"[V2] mode ① run: task record {dirs[0].name} ({len(files)} files), STEP line, RED summary, timings")

tab.shutdown()
root.destroy()
SNAP1 = {str(p.relative_to(ROOT)): H(p.read_bytes()) for p in ROOT.rglob("*")
         if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and "logs" not in p.parts}
assert SNAP1 == SNAP0, sorted(set(SNAP0.items()) ^ set(SNAP1.items()))[:6]
print("[V3] release folder unchanged")
print("ALL V3.68 GUI TESTS PASSED")
