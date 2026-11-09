"""V4.6.9 GUI test, Smol VLA tab (needs tkinter + DISPLAY; skips otherwise); fakes copied from test_v4_6_2_gui:
H1 top bar (back end, policy, control authority, E-STOP) + 5 sub-pages ① 准备 ② 示教采集 ③ 训练 ④ 执行 ⑤ 评估 / 高级;
   in a 1420x1000 window every button of every page, E-STOP and Copy Log are on screen and the log keeps >= 100 px.
H2 gripper teach: buttons and the space bar drive the real SO-101 on the real back end (fake gripper), the space bar
   is ignored while a text field has the focus or the tab is hidden; mock back end only records the number.
H3 recording on the mock back end warns; replay on the real back end calls replay_episode_real (real NERO path).
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
        print("[SKIP] test_v4_6_9_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v4_6_9_gui: {exc}")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="nero_v469_gui_"))
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


tab.grip = Grip()
VIS = {"on": True}
vt.visible_fn = lambda: VIS["on"]
root.geometry("1420x1000+0+0")
nb.select(fr3)
pump(1.0)
titles = [vt.sub_nb.tab(i, "text") for i in range(vt.sub_nb.index("end"))]
assert titles == ["① 准备", "② 示教采集", "③ 训练", "④ 执行", "⑤ 评估 / 高级"], titles
for i, title in enumerate(titles):
    vt.sub_nb.select(i)
    pump(0.5)
    page = vt.sub_nb.nametowidget(vt.sub_nb.tabs()[i])
    bs = [c for c in walk(page) if c.winfo_class() in ("TButton", "Button") and c.winfo_manager()]
    assert bs, title
    for b in bs:
        if b.winfo_ismapped():
            assert on_screen(b), (title, b.cget("text"))
    assert on_screen(vt.btn_estop) and on_screen(buttons(fr3, "Copy Log")[0]), title
    assert vt.txt.winfo_height() >= 100 and on_screen(vt.txt), (title, vt.txt.winfo_height())
    shot(f"v469_vla_{i + 1}.png")
for t_ in ("● 开始示教录制", "夹爪示教: 张开", "夹爪示教: 夹紧", "回放 Episode", "删除 Episode", "进入示教模式"):
    assert len(buttons(vt.pages["teach"], t_)) == 1, t_
for t_ in ("③ 执行 SmolVLA 闭环", "④ 停止执行"):
    assert len(buttons(vt.pages["run"], t_)) == 1, t_
assert len(buttons(vt.pages["prep"], "① 加载 SmolVLA & 连接 Adapter")) == 1
assert len(buttons(vt.pages["adv"], "Phase 0 审查")) == 1
assert "控制权" in vt.v_arb_text.get()
print(f"[H1] 5 sub-pages {titles}; every page fits 1420x1000 with E-STOP + log ({vt.txt.winfo_height()} px)")


def wait_grip(n, sec=3.0):
    t_end = time.time() + sec
    while len(tab.grip.ev) < n and time.time() < t_end:
        pump(0.05)


vt.sub_nb.select(1)
vt.v_backend.set("real")
pump(0.3)
buttons(vt.pages["teach"], "夹爪示教: 夹紧")[0].invoke()
wait_grip(1)
assert tab.grip.ev == [("clamp", 25.0)], tab.grip.ev
assert vt.recorder._teach_gripper_cmd_norm == 0.0
root.focus_force()
fr3.focus_set()
pump(0.2)
root.event_generate("<KeyPress-space>", when="tail")
wait_grip(2)
assert tab.grip.ev[-1] == ("open", 45.0), tab.grip.ev
assert vt.recorder._teach_gripper_cmd_norm == 1.0
ent = [c for c in walk(fr3) if c.winfo_class() == "TEntry"][0]
import types                                                # Xvfb has no window manager: focus cannot move, so the
assert vt._on_space(types.SimpleNamespace(widget=ent)) is None   # key event is handed over as Tk would (widget = focus)
pump(0.5)
assert len(tab.grip.ev) == 2, tab.grip.ev                   # typing a space in a field does not toggle
fr3.focus_set()
VIS["on"] = False
root.event_generate("<KeyPress-space>", when="tail")
pump(0.5)
assert len(tab.grip.ev) == 2, tab.grip.ev                   # tab hidden -> ignored
VIS["on"] = True
vt.v_backend.set("mock")
n_log = len(LOG)
vt.on_teach_gripper(False)
pump(0.3)
assert len(tab.grip.ev) == 2 and any("只记录数值" in s for s in LOG[n_log:]), LOG[n_log:]
print("[H2] gripper teach buttons + space bar drive the SO-101 on the real back end; ignored in text fields / hidden tab")

n_log = len(LOG)
vt.on_toggle_record()
pump(0.5)
assert any("[V4.6.9 RECORDER]" in s and "虚拟机械臂" in s for s in LOG[n_log:]), LOG[n_log:]
if vt.recorder.is_recording:
    vt.on_toggle_record()
pump(0.5)
seen = {}
orig = V4T.replay_episode_real
V4T.replay_episode_real = lambda ds, idx, p, model, gripper=None, speed=0.5, log=print, stop_fn=None: \
    seen.update(idx=idx, pipe=p, grip=gripper, speed=speed) or {"ok": True}
try:
    vt.v_backend.set("real")
    vt.v_ep_idx.set("0")
    vt.v_replay_speed.set("0.25×")
    vt.on_replay_episode()
    t_end = time.time() + 3
    while not seen and time.time() < t_end:
        pump(0.05)
finally:
    V4T.replay_episode_real = orig
assert seen.get("pipe") is pipe and seen.get("grip") is tab.grip and seen.get("speed") == 0.25, seen
print("[H3] mock recording warns; real back end replay -> replay_episode_real(pipe, SO-101, 0.25x)")

vt.shutdown()
tab.shutdown()
root.destroy()
SNAP1 = {str(p.relative_to(ROOT)): H(p.read_bytes()) for p in ROOT.rglob("*")
         if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and "logs" not in p.parts}
assert SNAP0 == SNAP1, set(SNAP0.items()) ^ set(SNAP1.items())
print("ALL V4.6.9 GUI TESTS PASSED")
