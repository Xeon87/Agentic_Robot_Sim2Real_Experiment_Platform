#!/usr/bin/env python3
"""V4.6.2 GUI test (needs tkinter + DISPLAY; skips otherwise):
G1 both task tabs carry the same teach bar: 「进入示教模式」 「退出示教模式」 「高级」, visible on a 1920x1080 screen.
G2 「Smol VLA实机任务」: monitors in one row, the log BELOW them over the full tab width, 「Copy Log」 visible;
   「Agentic实机任务」: 「Copy Log」 visible (no longer pushed off screen by the expanding log).
G3 enter / exit call RealPipeline.set_teach_mode exactly like the 实机NERO调试 buttons (exit: hold pose, no move home).
G4 「高级」: custom mass is saved to NERO_CALIB_DIR/teach_gravity.json (not the release folder), used as --payload,
   「更轻」 nudges, 「保存并立即生效」 re-enters teach mode when NERO is in teach mode; bad input is not saved.
G5 Copy Log puts the log text on the clipboard (both tabs).
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
        print("[SKIP] test_v4_6_2_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v4_6_2_gui: {exc}")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="nero_v462_gui_"))
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


# ---- G1 / G2 layout
for frame, name in ((fr, "Agentic"), (fr3, "SmolVLA")):
    nb.select(frame)
    pump(1.0)
    for t_ in ("进入示教模式", "退出示教模式", "高级", "Copy Log"):
        bs = buttons(frame, t_)
        assert len(bs) == 1, (name, t_, len(bs))
        assert on_screen(bs[0]), (name, t_, bs[0].winfo_ismapped(), bs[0].winfo_rooty(), bs[0].winfo_height())
    shot(f"v462_{name}.png")
assert vt.teach_bar is not None and tab.teach_bar is not None
nb.select(fr3)
pump(0.5)
cv_bottom = vt.monitor.cvw.winfo_rooty() + vt.monitor.cvw.winfo_height()
assert vt.txt.winfo_rooty() >= cv_bottom, (vt.txt.winfo_rooty(), cv_bottom)          # log below the monitors
assert vt.txt.winfo_width() >= 0.9 * fr3.winfo_width(), (vt.txt.winfo_width(), fr3.winfo_width())
assert vt.txt.winfo_height() >= 80, vt.txt.winfo_height()
assert on_screen(vt.txt)
# the drive details fold (default folded, lamps stay); the default console window 1420x1000 still leaves the log room
assert not vt.drv_detail.winfo_manager() and vt.lamp_drive is not None
vt.toggle_drive_detail()
pump(0.3)
assert vt.drv_detail.winfo_ismapped() and "▴" in vt.btn_drv_fold.cget("text")
vt.toggle_drive_detail(False)
root.geometry("1420x1000+0+0")
pump(1.0)
assert vt.txt.winfo_height() >= 100 and on_screen(vt.txt) and on_screen(buttons(fr3, "Copy Log")[0]), \
    vt.txt.winfo_height()
h1420 = vt.txt.winfo_height()
shot("v462_SmolVLA_1420.png")
root.geometry(f"{W}x{HH}+0+0")
pump(0.6)
print(f"[G1] teach bar + Copy Log visible in both tabs; [G2] VLA log below the monitors, "
      f"{vt.txt.winfo_width()} of {fr3.winfo_width()} px wide, {vt.txt.winfo_height()} px high (1420x1000 window: {h1420} px)")

# ---- G3 enter / exit
for bar in (vt.teach_bar, tab.teach_bar):
    pipe.calls.clear()
    bar.btn_enter.invoke()
    t_end = time.time() + 5
    while bar.worker.is_alive() and time.time() < t_end:
        pump(0.05)
    bar.btn_exit.invoke()
    t_end = time.time() + 5
    while bar.worker.is_alive() and time.time() < t_end:
        pump(0.05)
    assert pipe.calls == [(True, True, False), (False, False, True)], pipe.calls
pump(0.7)
assert "正常控制" in vt.teach_bar.v_state.get() and "重力补偿" in vt.teach_bar.v_state.get()
print("[G3] enter -> set_teach_mode(True); exit -> set_teach_mode(False, return_home=False)")

# ---- G4 advanced dialog
pipe.calls.clear()
vt.teach_bar.btn_adv.invoke()
dlg = vt.teach_bar.dialog
assert dlg is not None
pump(0.3)
dlg.v_mode.set("custom")
dlg._mode_ui()
dlg.v_mass.set("abc")
dlg.on_save()
assert "没有保存" in dlg.v_info.get() and not os.path.exists(TG.settings_path())
dlg.v_mass.set("0.42")
dlg.v_cx.set("0"); dlg.v_cy.set("5"); dlg.v_cz.set("70")
dlg.nudge(-1)                                                  # 「更轻（回弹时）」
assert dlg.v_mass.get() == "0.37", dlg.v_mass.get()
dlg.on_save(apply_now=True)                                    # not in teach mode -> saved, no mode switch
assert pipe.calls == [] and "下次" in dlg.v_info.get()
s = TG.load()
assert s["mode"] == "custom" and s["mass_kg"] == 0.37 and s["com_mm"] == [0.0, 5.0, 70.0], s
assert TG.settings_path().startswith(str(TMP)) and not str(TG.settings_path()).startswith(str(ROOT))
assert json.loads(TG.payload_arg(TG.load()))["mass_kg"] == 0.37
pipe.teach_mode = True
dlg.on_save(apply_now=True)                                    # in teach mode -> re-enter teach with the new value
t_end = time.time() + 5
while vt.teach_bar.worker.is_alive() and time.time() < t_end:
    pump(0.05)
assert pipe.calls == [(True, True, False)], pipe.calls
dlg.v_mode.set("class")
dlg.v_class.set(TG.CLASS_LABELS["half"])
dlg.nudge(-1)
assert dlg.v_class.get() == TG.CLASS_LABELS["empty"]
dlg.nudge(-1)
assert dlg.v_class.get() == TG.CLASS_LABELS["empty"]                     # lightest class stays
dlg.on_save()
assert TG.payload_arg(TG.load()) == "empty"
dlg.on_probe()
pump(1.0)
assert "set_payload(load: str)" in dlg.v_info.get(), dlg.v_info.get()
shot("v462_advanced.png")
dlg.close()
pipe.teach_mode = False
print("[G4] 高级: custom mass saved outside the release folder, nudge, apply-now re-enters teach, probe listing")

# ---- G5 Copy Log
for t_obj in (vt, tab):
    t_obj.txt.insert("end", "V462-COPY-CHECK\n")
    t_obj.copy_log()
    assert "V462-COPY-CHECK" in root.clipboard_get()
print("[G5] Copy Log -> clipboard in both tabs")

vt.shutdown()
tab.shutdown()
root.destroy()
SNAP1 = {str(p.relative_to(ROOT)): H(p.read_bytes()) for p in ROOT.rglob("*")
         if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and "logs" not in p.parts}
assert SNAP0 == SNAP1, set(SNAP0.items()) ^ set(SNAP1.items())
print("ALL V4.6.2 GUI TESTS PASSED")
