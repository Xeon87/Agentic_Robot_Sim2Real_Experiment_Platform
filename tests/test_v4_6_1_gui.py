#!/usr/bin/env python3
"""V4.6.1 GUI test (needs tkinter + DISPLAY; skips otherwise):
M1 the 「Smol VLA实机任务」 tab's camera monitors mirror the 「Agentic实机任务」 tab: same two headings, same frame size,
   and after a perception the same items on the main monitor (image, YOLOE contour polygon, label, "识别于 … s 前")
   and on the wrist monitor (image + red cross hair); the YOLOE description line is copied too.
M2 the VLA tab builds with the mirrored monitors and refreshes them from its own timer.
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
        print("[SKIP] test_v4_6_1_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v4_6_1_gui: {exc}")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="nero_v461_gui_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
os.environ["NERO_V368_TASK_DIR"] = str(TMP / "task")          # V3.68: task records never in the release folder
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
nb.add(fr, text="Agentic实机任务")
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
tab.on_perceive()
wait_idle()
fr2 = ttk.Frame(nb, padding=6)
nb.add(fr2, text="Smol VLA实机任务")
mirror = RT.CameraMonitorMirror(tk, ttk, fr2, tab)
nb.select(fr)
pump(0.6)
cam.src.fid += 1                                   # a new frame for both views
mirror.refresh()
tab._shown = (-1, -1)
tab.poll()


def items(cv):
    out = []
    for i in cv.find_all():
        t = cv.type(i)
        out.append((t, cv.itemcget(i, "text") if t == "text" else ""))
    return out


def labels(w):
    out = []
    for c in w.winfo_children():
        if c.winfo_class() in ("TLabel", "Label"):
            out.append(c.cget("text"))
        out += labels(c)
    return out


a, b = items(tab.cv), items(mirror.cv)
norm = lambda L: sorted((t, x if not x.startswith("识别于") else "识别于") for t, x in L)
assert norm(a) == norm(b) and any(t == "polygon" for t, _ in b), (a, b)
assert [t for t, _ in items(mirror.cvw)].count("line") == 2 and "image" in [t for t, _ in items(mirror.cvw)]
assert (int(mirror.cv.cget("width")), int(mirror.cv.cget("height"))) == (tab.DW, tab.DH)
lab_m = labels(fr2)
assert RT.MON_MAIN_TITLE in lab_m and RT.MON_WRIST_TITLE in lab_m, lab_m
assert RT.MON_MAIN_TITLE in labels(fr) and RT.MON_WRIST_TITLE in labels(fr)
assert mirror.v_yolo.get() == tab.v_yolo.get() and "yoloe-26s-seg.pt" in mirror.v_yolo.get()
print(f"[M1] mirrored monitors: same headings, size {tab.DW}x{tab.DH}, same overlay items {sorted(set(t for t, _ in b))}")

os.environ["NERO_DRIVE_ROOT"] = str(TMP / "drive")
os.makedirs(os.environ["NERO_DRIVE_ROOT"], exist_ok=True)
import NERO_V4_Console_Tab as V4T                       # noqa: E402
fr3 = ttk.Frame(nb, padding=6)
nb.add(fr3, text="Smol VLA实机任务 (2)")
vt = V4T.VLALeRobotTabUI(tk, ttk, fr3, pipe=Pipe(), jpanel=JP(), cam_tab=cam, rct_tab=tab, log=lambda s: None,
                         ask=lambda a_, b_: True, visible_fn=lambda: True)
assert vt.monitor is not None
cam.src.fid += 1
pump(1.0)
assert any(t == "image" for t, _ in items(vt.monitor.cv)), items(vt.monitor.cv)
print("[M2] Smol VLA tab builds with the mirrored monitors and refreshes them")
vt.shutdown()
tab.shutdown()
root.destroy()
print("ALL V4.6.1 GUI TESTS PASSED")
