#!/usr/bin/env python3
"""V3.66 GUI test (needs tkinter + DISPLAY; skips otherwise):
S1 Real Tk: the 「NERO实机任务」 tab shows the YOLOE prompt field (default "cup") and, after 「识别杯子（不动）」, the
   Camera tab's YOLOE settings (model file, prompt, confidence) in the description line.
S2 「识别杯子（不动）」 runs the Camera tab's detector object on the Camera tab's frames (_grab_frames, FUSE_FRAMES)
   with the prompt from the field; the YOLOE instance is drawn on the main monitor (contour polygon + label);
   worker threads never touch Tk.
S3 Camera tab without a YOLOE model -> the run is refused with "YOLOE 不可用" (no colour fallback, nothing moves).
S4 Settings (prompt) go to NERO_CALIB_DIR/real_cup_task.json; nothing inside the release folder changed.
"""
import hashlib
import json
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
        print("[SKIP] test_v3_66_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v3_66_gui: {exc}")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="nero_v366_gui_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
H = lambda b: hashlib.sha256(b).hexdigest()
SNAP0 = {str(p.relative_to(ROOT)): H(p.read_bytes()) for p in ROOT.rglob("*")
         if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and "logs" not in p.parts}

import NERO_V325_Camera as CAM               # noqa: E402
import NERO_V347_Yolo_Perception as YP       # noqa: E402
import NERO_V365_Real_Cup_Task as RT         # noqa: E402

KK = {"width": 1280, "height": 720, "fx": 900.0, "fy": 900.0, "ppx": 640.0, "ppy": 360.0}
# camera 0.75 m above the table looking straight down (base frame: camera z = -base z)
T_BC = np.array([[1.0, 0, 0, -0.40], [0, -1.0, 0, 0.0], [0, 0, -1.0, 0.75], [0, 0, 0, 1.0]])


class Src:
    K = KK

    def __init__(self, shade):
        self.fid, self.shade = 0, shade

    def running(self):
        return True

    def frames(self):
        self.fid += 1
        return np.zeros((720, 1280, 3), np.uint8) + 40, np.full((720, 1280), 0.75, np.float32), self.fid

    def display_ppm(self, view="color", size=(560, 315)):
        return CAM.ppm_bytes(np.zeros((size[1], size[0], 3), np.uint8) + self.shade)


class Det:
    info = {"device": "CPU"}

    def __init__(self):
        self.calls = []

    def detect(self, rgb, mode, path, ref=None, prompts=None, conf=0.25):
        self.calls.append((mode, os.path.basename(path), list(prompts), conf))
        m = np.zeros(rgb.shape[:2], bool)
        m[300:420, 600:700] = True
        return [{"cls": 0, "name": prompts[0], "conf": 0.87, "box": [600.0, 300.0, 699.0, 419.0], "mask": m}]


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
        self.src, self.wrist_src = Src(90), Src(60)
        self.det_yolo = Det()
        self.v_ydir, self.v_ymodel, self.v_yconf = _V(str(MD)), _V("yoloe-26s-seg.pt"), _V("0.30")
        self.grabs = []

    def _grab_frames(self, n, timeout=2.0):
        self.grabs.append(n)
        rgb, d, _ = self.src.frames()
        return rgb, [d] * n

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
cam = Cam()
tab = RT.RealCupTaskTabUI(tk, ttk, fr, Pipe(), JP(), cam, log=lambda s: None, ask=lambda a, b: False)


def pump(sec):
    t_end = time.time() + sec
    while time.time() < t_end:
        root.update()
        time.sleep(0.03)


def wait_idle(timeout=20):
    t_end = time.time() + timeout
    while tab.busy() and time.time() < t_end:
        pump(0.1)
    pump(0.4)


pump(0.5)
assert tab.v_cup_prompt.get() == "cup"
tab.v_cup_prompt.set("cup, party cup")
tab.on_perceive()
wait_idle()
desc = tab.v_yolo.get()
assert "yoloe-26s-seg.pt" in desc and "cup" in desc and "0.30" in desc, desc
print("[S1] prompt field (default cup); the Camera tab's YOLOE settings shown")

assert cam.det_yolo.calls and cam.det_yolo.calls[-1] == ("text", "yoloe-26s-seg.pt", ["cup", "party cup"], 0.30), \
    cam.det_yolo.calls
assert YP.FUSE_FRAMES in cam.grabs
log = tab.txt.get("1.0", "end")
assert "[V3.66 REAL CUP YOLOE] 识别杯子 | " in log and "YOLOE 1 个" in log, log[-600:]
assert "main thread is not in main loop" not in log
pump(0.6)
polys = [i for i in tab.cv.find_all() if tab.cv.type(i) == "polygon"]
texts = [tab.cv.itemcget(i, "text") for i in tab.cv.find_all() if tab.cv.type(i) == "text"]
assert polys and any(t.startswith("#1 ") and "0.87" in t for t in texts), (polys, texts)
print(f"[S2] 识别杯子: Camera-tab detector + frames, prompt from the field; monitor shows the YOLOE contour {texts[:2]}")

cam.v_ymodel = _V("missing-model.pt")
tab.v_mode.set("plan")
tab.on_run()
wait_idle()
log = tab.txt.get("1.0", "end")
assert "不能开始" in log and "YOLOE 不可用" in log, log[-500:]
print("[S3] no YOLOE model -> run refused with YOLOE 不可用 (no colour fallback)")

tab.shutdown()
root.destroy()
saved = json.loads((TMP / "cameracalib" / "real_cup_task.json").read_text(encoding="utf-8"))
assert saved["cup_prompt"] == "cup, party cup"
SNAP1 = {str(p.relative_to(ROOT)): H(p.read_bytes()) for p in ROOT.rglob("*")
         if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and "logs" not in p.parts}
assert SNAP1 == SNAP0, sorted(set(SNAP0.items()) ^ set(SNAP1.items()))[:6]
print("[S4] settings in NERO_CALIB_DIR; release folder unchanged")
print("ALL V3.66 GUI TESTS PASSED")
