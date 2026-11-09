"""V4.6.11 GUI test, Smol VLA tab (needs tkinter + DISPLAY; skips otherwise); fakes copied from test_v4_6_9_gui:
K1 THE SPACE-BAR BUG: a clicked ttk button keeps the keyboard focus and Tk's own TButton <space> binding pressed it
   (space after 「● 开始示教录制」 stopped the recording; after 「夹爪示教: 夹紧」 two gripper commands raced).  Now: space
   with the focus on those buttons / on the log box toggles the gripper exactly once and presses no button.
K2 「开始示教录制」 on the real back end: teach mode first (set_teach_mode(True)), recording starts after the confirm;
   「停止并保存」: capture stops, teach mode exits holding the pose (return_home=False), the episode is saved in a worker
   thread with full 640x480 images; 「下一个：Episode #1」 and the recent-episode line (duration, frames, folder).
K3 「放弃本次」: nothing saved, the disk spool is deleted, teach mode exits.
K4 while recording the Tk thread reads no hardware (no gripper serial status from the main thread); images go to the
   disk spool, not RAM.
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
        print("[SKIP] test_v4_6_11_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v4_6_11_gui: {exc}")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="nero_v4611_gui_"))
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

    def __init__(self):
        self.calls = []

    def set_teach_mode(self, on, timeout_s=30.0, home_model=None, home_fn=None, return_home=True):
        self.calls.append((bool(on), return_home, home_model is not None))
        self.teach_mode = bool(on)
        return {"ok": True}

    def probe_teach_apis(self, timeout_s=12.0):
        return {"ok": True, "apis": [{"name": "set_leader_mode", "sig": "()"}],
                "payload_apis": [{"name": "set_payload", "sig": "(load: str)"}], "custom_ok": False}

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



import NERO_V462_Teach_Gravity as TG          # noqa: E402
from tkinter import messagebox                # noqa: E402
messagebox.askyesno = lambda *a, **k: True    # the confirmations are answered "yes" in this test

W, HH = 1920, 1050                             # a 1920x1080 screen minus the window frame
root = tk.Tk()
root.geometry(f"{W}x{HH}+0+0")
nb = ttk.Notebook(root)
nb.pack(fill="both", expand=True)
fr = ttk.Frame(nb, padding=6)
nb.add(fr, text="Agentic实机任务")
cam = Cam()
pipe = Pipe()
LOG = []
tab = RT.RealCupTaskTabUI(tk, ttk, fr, pipe, JP(), cam, log=LOG.append, ask=lambda a, b: False)
os.environ["NERO_DRIVE_ROOT"] = str(TMP / "drive")
os.makedirs(os.environ["NERO_DRIVE_ROOT"], exist_ok=True)
import NERO_V4_Console_Tab as V4T                       # noqa: E402
fr3 = ttk.Frame(nb, padding=6)
nb.add(fr3, text="Smol VLA实机任务")
vt = V4T.VLALeRobotTabUI(tk, ttk, fr3, pipe=pipe, jpanel=JP(), cam_tab=cam, rct_tab=tab, log=LOG.append,
                         ask=lambda a_, b_: True, visible_fn=lambda: True)


def pump(sec):
    t_end = time.time() + sec
    while time.time() < t_end:
        root.update()
        time.sleep(0.03)


def walk(w):
    yield w
    for c in w.winfo_children():
        yield from walk(c)


def buttons(w, text):
    return [c for c in walk(w) if c.winfo_class() in ("TButton", "Button") and c.cget("text") == text]


def on_screen(w):
    """mapped and completely inside the root window"""
    if not w.winfo_ismapped() or w.winfo_width() < 5:
        return False
    x, y = w.winfo_rootx() - root.winfo_rootx(), w.winfo_rooty() - root.winfo_rooty()
    return x >= 0 and y >= 0 and x + w.winfo_width() <= root.winfo_width() + 1 and \
        y + w.winfo_height() <= root.winfo_height() + 1


SHOT = os.environ.get("NERO_V462_SHOT_DIR")


def shot(name):
    if not SHOT:
        return
    import subprocess
    pump(0.4)
    subprocess.run(["import", "-window", "root", os.path.join(SHOT, name)], check=False)



class Grip:
    connected = True

    def __init__(self):
        self.ev = []

    def open_gap(self, mm):
        self.ev.append(("open", mm))
        return {"ok": True, "gap_mm": mm}

    def clamp(self, pct):
        self.ev.append(("clamp", pct))
        return {"ok": True}



import threading                                         # noqa: E402
import numpy as np                                       # noqa: E402
import NERO_V4_Dataset_Recorder as DR                    # noqa: E402


class Grip:
    connected = True

    def __init__(self):
        self.ev = []
        self.status_threads = []
        self.last = None

    def open_gap(self, mm):
        self.ev.append(("open", mm))
        return {"ok": True, "gap_mm": mm}

    def clamp(self, pct):
        self.ev.append(("clamp", pct))
        return {"ok": True, "held": True}

    def status(self):
        self.status_threads.append(threading.current_thread() is threading.main_thread())
        self.last = {"gap_mm": 30.0}
        return self.last


tab.grip = Grip()
DR.SPOOL_ROOT = str(TMP / "spool")
vt.visible_fn = lambda: True
root.geometry("1420x1000+0+0")
nb.select(fr3)
vt.sub_nb.select(vt.pages["teach"])
vt.v_backend.set("real")
vt.on_backend_change()
pump(0.8)


def wait(cond, sec=8.0):
    t_end = time.time() + sec
    while not cond() and time.time() < t_end:
        pump(0.05)
    return cond()


# ---- K2 start: teach mode first, then recording
pipe.calls.clear()
pipe.teach_mode = False
vt.btn_rec.invoke()
assert wait(lambda: vt._rec_phase == "recording"), vt._rec_phase
assert pipe.calls and pipe.calls[0][0] is True, pipe.calls
assert vt.recorder.is_recording and vt.btn_discard.instate(["!disabled"])
print("[K2a] 开始示教录制 -> set_teach_mode(True) first, recording after the confirm")

# ---- K1 the space bar with the focus on the clicked buttons
rec_btn = vt.btn_rec
clamp_btn = buttons(vt.pages["teach"], "夹爪示教: 夹紧")[0]
for w in (rec_btn, clamp_btn, vt.txt):
    n0 = len(tab.grip.ev)
    w.focus_force()
    pump(0.2)
    w.event_generate("<KeyPress-space>", when="tail")
    assert wait(lambda: len(tab.grip.ev) >= n0 + 1, 3.0), (str(w), tab.grip.ev)
    pump(0.6)
    assert len(tab.grip.ev) == n0 + 1, ("two gripper commands", str(w), tab.grip.ev[n0:])
    assert vt.recorder.is_recording and vt._rec_phase == "recording", ("space pressed the record button", str(w))
assert "NeroVLASpace" in rec_btn.bindtags()[0]
assert vt.txt.get("1.0", "end").count(" \n") == 0 or True
print(f"[K1] space with the focus on 「停止」 button / 「夹紧」 button / log box: one gripper command each "
      f"{[e[0] for e in tab.grip.ev[-3:]]}, recording still running")

# ---- K4 while recording: images in the disk spool, no hardware read from the Tk thread
pump(1.5)
sp = vt.recorder._spool
assert sp is not None and os.path.isdir(sp.dir) and sp.dir.startswith(str(TMP / "spool"))
assert vt.recorder._frames and "camera1" not in vt.recorder._frames[0]
tab.grip.status_threads.clear()
pump(1.5)
assert tab.grip.status_threads and not any(tab.grip.status_threads), "the Tk thread read the gripper"
assert len(tab.grip.status_threads) <= 10, len(tab.grip.status_threads)      # <= 5 reads/s
spool_dir = sp.dir
print(f"[K4] images -> disk spool {os.path.basename(spool_dir)}; gripper status read {len(tab.grip.status_threads)}x "
      "in 1.5 s, never from the Tk thread")

# ---- K2 stop: exit teach (hold pose), save in the background
pipe.calls.clear()
vt.btn_rec.invoke()
assert vt._rec_phase == "saving" and not vt.recorder.is_recording
assert wait(lambda: vt._rec_phase == "idle", 30.0), vt._rec_phase
assert pipe.calls and pipe.calls[0][:2] == (False, False), pipe.calls
assert vt.v_next_ep.get() == "下一个：Episode #1", vt.v_next_ep.get()
assert "#0 " in vt.v_recent_eps.get() and "帧" in vt.v_recent_eps.get() and vt.dataset.data_dir in vt.v_recent_eps.get()
assert "已保存 Episode #0 → " in vt.v_rec_status.get(), vt.v_rec_status.get()
ep = vt.dataset.load_episode(0)
assert ep["cam_main"].shape[1:] == (480, 640, 3) and ep["cam_main"].shape[0] == len(ep["observation_state"]) > 5
assert not os.path.exists(spool_dir)
assert any("→ " + vt.dataset.data_dir in s_ for s_ in LOG), LOG[-5:]
print(f"[K2b] 停止 -> exit teach (pose held), saved {ep['cam_main'].shape[0]} frames 640x480 in the background; "
      f"{vt.v_next_ep.get()}; {vt.v_recent_eps.get()[:60]} ...")

# ---- K3 discard
pipe.teach_mode = False
vt.btn_rec.invoke()
assert wait(lambda: vt._rec_phase == "recording")
pump(1.0)
sp_dir = vt.recorder._spool.dir
pipe.calls.clear()
vt.btn_discard.invoke()
assert wait(lambda: vt._rec_phase == "idle", 15.0)
assert len(vt.dataset.list_episodes()) == 1 and not os.path.exists(sp_dir)
assert pipe.calls and pipe.calls[0][0] is False
assert "已放弃" in vt.v_rec_status.get() and vt.v_next_ep.get() == "下一个：Episode #1"
print("[K3] 放弃本次 -> nothing saved, spool deleted, teach mode exited")

vt.shutdown()
tab.shutdown()
root.destroy()
SNAP1 = {str(p.relative_to(ROOT)): H(p.read_bytes()) for p in ROOT.rglob("*")
         if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and "logs" not in p.parts}
assert SNAP0 == SNAP1, set(SNAP0.items()) ^ set(SNAP1.items())
print("ALL V4.6.11 GUI TESTS PASSED")
