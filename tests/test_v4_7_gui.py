"""V4.7 GUI test, Smol VLA tab (needs tkinter + DISPLAY; skips otherwise); fakes copied from test_v4_6_11_gui:
J1 「一键准备」 on the simulated back end without an environment / model: the steps run in order, NERO / cameras /
   gripper are skipped, the missing environment and model are yellow, the 「① 准备」 tab lamp turns YELLOW and the
   result line names the first problem.
J2 「一键准备」 on the real back end with everything there (environment, model, NERO lamps green): the cameras and
   the gripper that are not connected yet get connected by the button, SmolVLA is loaded -> every lamp green, the
   tab lamp GREEN; NERO not started -> yellow with "不会自动使能".
J3 Episode browser: two recorded episodes are listed newest first; a dataset made elsewhere appears in the dataset
   list and can be switched to (new recordings go there).
J4 「▶ 播放录像」: a player window with both cameras, play / step / slider, YOLOE labels (fake detector, run in the
   worker thread and cached per frame); a double click opens the same player.
J5 「删除 Episode」: asks first (no -> nothing deleted), yes -> the file is gone, the open player of that episode is
   closed, the list refreshes.
J6 everything fits 1420x1000 (log >= 100 px).
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
        print("[SKIP] test_v4_7_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v4_7_gui: {exc}")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="nero_v47_gui_"))
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




import threading                                         # noqa: E402
import types                                             # noqa: E402
import numpy as np                                       # noqa: E402
import NERO_V4_Dataset_Recorder as DR                    # noqa: E402
import NERO_V44_Storage as ST                            # noqa: E402
import NERO_V47_VLA_Tools as TL                          # noqa: E402

DR.SPOOL_ROOT = str(TMP / "spool")


class Grip:
    def __init__(self):
        self.connected = False
        import NERO_V365_STS3215_Gripper as GR
        self.cfg = dict(GR.DEFAULTS)
        self.ev = []
        self.last = {"gap_mm": 30.0}

    def connect(self, port, baud, sid):
        self.connected = True
        return True, "fake"

    def open_gap(self, mm):
        self.ev.append(("open", mm))
        return {"ok": True}

    def clamp(self, pct):
        self.ev.append(("clamp", pct))
        return {"ok": True, "held": True}

    def status(self):
        return self.last


tab.grip = Grip()
vt.visible_fn = lambda: True
root.geometry("1420x1000+0+0")
nb.select(fr3)
pump(0.8)


def wait(cond, sec=10.0):
    t_end = time.time() + sec
    while not cond() and time.time() < t_end:
        pump(0.05)
    return cond()


def lamp_states():
    return {k: l.state for k, l in vt.prep_lamps.items()}


def tab_image():
    return str(vt.sub_nb.tab(vt.pages["prep"], "image")[0]) if vt.sub_nb.tab(vt.pages["prep"], "image") else ""


# ---- J1 simulated back end, nothing installed
vt.sub_nb.select(vt.pages["prep"])
assert tab_image() == str(vt._tab_imgs["off"])
vt.v_backend.set("mock")
vt.btn_prepare.invoke()
assert wait(lambda: not vt._prep_busy, 60.0)
ls = lamp_states()
assert ls["drive"] == "ok" and ls["dirs"] == "ok", ls
assert ls["env"] == "warn" and ls["model"] == "warn", ls
assert ls["nero"] == ls["cam"] == ls["wrist"] == ls["grip"] == "off", ls
assert ls["vla"] == "off", ls
assert tab_image() == str(vt._tab_imgs["warn"]), tab_image()
assert vt.v_prep.get().startswith("⚠") and "环境" in vt.v_prep.get(), vt.v_prep.get()
shot("v47_prepare_warn.png")
print(f"[J1] simulated back end: {ls} -> tab lamp yellow; {vt.v_prep.get()[:60]}")

# ---- J2 real back end, everything available; cameras + gripper connected BY the button
ST_venv, ST_hw = ST.venv_status, ST.has_weights
ST.venv_status = lambda L=None, **k: {"ready": True, "cuda": True, "exists": True}
ST.has_weights = lambda d: True
loads = []
vt.vla_policy.load = lambda ck: loads.append(ck) or {"backend": "fake"}
vt.vla_policy.model_profile = lambda: {"loaded": True, "real_allowed": True, "reasons": [], "backend": "fake",
                                       "state_dim": 8, "action_dim": 8}
vt.adapter.connect = lambda: setattr(vt.adapter, "is_connected", True) if False else None
type(vt.adapter).is_connected = property(lambda self: True)


class Run:
    def __init__(self):
        self.on = False

    def running(self):
        return self.on

    def frames(self):
        return np.zeros((720, 1280, 3), np.uint8) + 40, np.full((720, 1280), 0.75, np.float32), 1

    def display_ppm(self, view="color", size=(560, 315)):
        return CAM.ppm_bytes(np.zeros((size[1], size[0], 3), np.uint8) + 50)


cam.src, cam.wrist_src = Run(), Run()
cam.on_connect = lambda: setattr(cam.src, "on", True)
cam.on_wrist_connect = lambda: setattr(cam.wrist_src, "on", True)
tab.grip.connected = False
vt.v_backend.set("real")
vt.btn_prepare.invoke()
assert wait(lambda: not vt._prep_busy, 60.0)
ls = lamp_states()
assert all(v == "ok" for v in ls.values()), (ls, [l_ for l_ in vt.txt.get("1.0", "end").splitlines() if "V4.6.11 UI" in l_ or "V3.65" in l_][-4:], [s_ for s_ in LOG if "V3.65" in s_][-4:])
assert cam.src.on and cam.wrist_src.on and tab.grip.connected
assert loads, "SmolVLA not loaded"
assert tab_image() == str(vt._tab_imgs["ok"]), tab_image()
assert vt.v_prep.get().startswith("✓"), vt.v_prep.get()
shot("v47_prepare_ok.png")
print(f"[J2] real back end: cameras + gripper connected by the button, SmolVLA loaded -> all green, tab lamp green")
pipe.lamp = lambda k: "grey" if k == "push" else "green"
vt.btn_prepare.invoke()
assert wait(lambda: not vt._prep_busy, 60.0)
assert vt.prep_lamps["nero"].state == "warn" and tab_image() == str(vt._tab_imgs["warn"])
assert any("不会自动使能" in s_ for s_ in LOG), [s_ for s_ in LOG if "PREPARE" in s_][-3:]
pipe.lamp = lambda k: "green"
ST.venv_status, ST.has_weights = ST_venv, ST_hw
print("[J2b] NERO not started -> yellow, the arm is not enabled by the button")

# ---- J3 two episodes (mock back end), the browser lists them newest first; dataset switch
vt.v_backend.set("mock")
vt.on_backend_change()
vt.sub_nb.select(vt.pages["teach"])
pump(0.3)
for _k in range(2):
    vt.btn_rec.invoke()
    assert wait(lambda: vt._rec_phase == "recording")
    pump(0.8)
    vt.btn_rec.invoke()
    assert wait(lambda: vt._rec_phase == "idle", 30.0)
ids = list(vt.ep_tree.get_children())
assert ids[:2] == ["1", "0"], ids
assert vt.ep_tree.selection() == ("1",), vt.ep_tree.selection()
assert "#1" in vt.v_browse_info.get()
other = DR.NEROLeRobotDataset(dataset_name="other_task", root_dir=vt.dataset.root_dir)
vt._fill_ds_list()
assert "other_task" in vt.cb_browse_ds.cget("values"), vt.cb_browse_ds.cget("values")
vt.v_browse_ds.set("other_task")
vt.on_browse_dataset()
pump(0.3)
assert vt.dataset.dataset_name == "other_task" and vt.v_ds_name.get() == "other_task"
assert vt.ep_tree.get_children() == ()
vt.v_browse_ds.set("nero_v4_demo")
vt.on_browse_dataset()
pump(0.3)
assert list(vt.ep_tree.get_children())[:2] == ["1", "0"]
shot("v47_browser.png")
print(f"[J3] browser: {ids} newest first, dataset switch both ways")

# ---- J4 player with YOLOE labels
calls = []
orig_detect = cam.det_yolo.detect


def det_wrap(rgb, mode, path, ref=None, prompts=None, conf=0.25):
    calls.append((threading.current_thread() is threading.main_thread(), rgb.shape, list(prompts or [])))
    return [{"box": [100, 100, 300, 300], "name": (prompts or ["egg"])[0], "conf": 0.9, "cls": 0}]


cam.det_yolo.detect = det_wrap
vt.ep_tree.selection_set("0")
pump(0.2)
vt.btn_play_video.invoke()
assert wait(lambda: len(vt.players) == 1, 5.0)
pl = vt.players[0]
assert pl.rd.n > 5
assert wait(lambda: all(pl.cv[k].find_all() for k in TL.CAM_KEYS), 5.0)
assert wait(lambda: ("cam_main", 0) in pl.dets and ("cam_wrist", 0) in pl.dets, 5.0)
assert calls and not any(c[0] for c in calls), "YOLOE must run in the worker thread"
assert calls[0][1] == (480, 640, 3)
pl.step(2)
assert wait(lambda: ("cam_main", 2) in pl.dets, 5.0)
pl.toggle()
assert pl.playing
assert wait(lambda: pl.i >= pl.rd.n - 1 or not pl.playing, 10.0)
pl.sl.set(1)
pump(0.3)
assert pl.i == 1
shot("v47_player.png")
pl.close()
pump(0.2)
assert vt.players == []
assert "on_play_episode_video" in vt.ep_tree.bind("<Double-1>") or vt.ep_tree.bind("<Double-1>")   # double click bound
vt.on_play_episode_video()                               # (Tk cannot synthesise a double click)
assert wait(lambda: len(vt.players) == 1, 5.0)
print(f"[J4] player: {pl.rd.n} frames, both cameras drawn, YOLOE labels from the worker thread ({len(calls)} calls), "
      "step / play / slider")

# ---- J5 delete asks first
vt.ep_tree.selection_set("0")
pump(0.2)
f0 = vt._selected_row()["file"]
vt.ask = lambda a, b: False
vt.on_delete_episode()
assert os.path.isfile(f0) and "0" in vt.ep_tree.get_children()
vt.ask = lambda a, b: True
vt.on_delete_episode()
pump(0.3)
assert not os.path.isfile(f0) and "0" not in vt.ep_tree.get_children()
assert vt.players == [], "the player of the deleted episode must close"
print("[J5] delete: asked, no -> kept; yes -> file + row gone, its player closed")

# ---- J6 layout
for pg_key in ("prep", "teach"):
    vt.sub_nb.select(vt.pages[pg_key])
    pump(0.5)
    assert vt.txt.winfo_height() >= 100 and on_screen(vt.txt) and on_screen(vt.btn_estop), (pg_key, vt.txt.winfo_height())
for w in (vt.btn_play_video, vt.ep_tree, vt.btn_rec):
    vt.sub_nb.select(vt.pages["teach"])
    pump(0.2)
    assert on_screen(w), w
vt.sub_nb.select(vt.pages["prep"])
pump(0.3)
assert on_screen(vt.btn_prepare)
print(f"[J6] 1420x1000: 一键准备 / Episode 浏览 / 录制 on screen, log {vt.txt.winfo_height()} px")

vt.shutdown()
tab.shutdown()
root.destroy()
SNAP1 = {str(p.relative_to(ROOT)): H(p.read_bytes()) for p in ROOT.rglob("*")
         if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and "logs" not in p.parts}
assert SNAP0 == SNAP1, set(SNAP0.items()) ^ set(SNAP1.items())
print("ALL V4.7 GUI TESTS PASSED")
