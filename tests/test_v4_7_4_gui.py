"""V4.7.4 GUI test, Smol VLA tab (needs tkinter + DISPLAY; skips otherwise); fixture copied from test_v4_7_gui:
M1 the new buttons (修改任务描述 / 重命名数据集) and the live limit line are on screen at 1420x1000.
M2 recording with J7 outside the commandable range: a WARN log line + bell + red text while recording; after
   saving the user is asked: no -> kept, listed red as 超限 with the excursion; yes -> deleted.
M3 player: the YOLOE prompts come from the episode's task text (red block, box, blue block - not eggs); the text box
   + 应用 changes what YOLOE looks for (detector called with the new prompts, old labels dropped).
M4 修改任务描述: two selected episodes -> meta/episodes.jsonl + tasks.jsonl changed after a backup, the copy inside
   the episode files rewritten in the background (video unchanged), every change logged; then the whole dataset.
M5 重命名数据集: invalid / existing names refused; a valid name moves the folder, fixes the stored file paths,
   the tab follows (list, next recording goes there).
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
        print("[SKIP] test_v4_7_4_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v4_7_4_gui: {exc}")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="nero_v474_gui_"))
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



import shutil                                            # noqa: E402
import NERO_V474_Episode_Tools as ET                     # noqa: E402

bells = []
vt.parent.bell = lambda *a: bells.append(1)
vt.v_backend.set("mock")
vt.on_backend_change()
vt.sub_nb.select(vt.pages["teach"])
pump(0.3)

# ---- M1 layout
for w in (vt.lbl_limit_warn, vt.ep_tree):
    assert on_screen(w), w
assert buttons(vt.parent, "修改任务描述…") and buttons(vt.parent, "重命名数据集…")
print("[M1] 修改任务描述 / 重命名数据集 buttons + live limit line on screen")

# ---- M2 J7 outside the commandable range while recording (mock arm sits at 0; J7 range set to 30..90 deg)
JP.model["limits"] = [(-3.0, 3.0)] * 6 + [(0.52, 1.5708)]
vt.jpanel = JP()
asked = []
TASK = "Put the red block into the left box, then put the blue block into the right box"
vt.v_task.set(TASK)


def record(answer):
    vt.ask = lambda t, m: asked.append((t, m)) or answer
    n0 = len(LOG)
    vt.btn_rec.invoke()
    assert wait(lambda: vt._rec_phase == "recording")
    assert wait(lambda: "超出可执行范围" in vt.v_limit_warn.get(), 5.0), vt.v_limit_warn.get()
    assert str(vt.lbl_limit_warn.cget("fg")) == "#c62828"
    pump(0.8)
    vt.btn_rec.invoke()
    assert wait(lambda: vt._rec_phase == "idle", 30.0)
    pump(0.3)
    return LOG[n0:]


lg = record(False)
assert any("[V4.7.4 LIMIT] 录制中 J7" in s for s in lg), lg
assert len(bells) == 1, bells                    # once per excursion, not every tick
assert asked and "超出关节限位" in asked[-1][0] and "J7" in asked[-1][1], asked
ep0 = vt.dataset.list_episodes()[-1]
assert ep0["limit_ok"] is False and ep0["limit_report"]["violations"][0]["joint"] == 7
vt.refresh_episode_list()
iid = str(ep0["episode_index"])
assert vt.ep_tree.set(iid, "state") == "超限" and "bad" in vt.ep_tree.item(iid, "tags"), (vt.ep_tree.item(iid), ep0.get("valid"), ep0.get("quality", {}).get("reasons"))
n_before = len(vt.dataset.list_episodes())
lg = record(True)
assert len(vt.dataset.list_episodes()) == n_before, "yes -> the over-limit episode is deleted"
assert any("已删除超限的 Episode" in s for s in lg), lg
JP.model["limits"] = [(-3.0, 3.0)] * 7
vt.jpanel = JP()
vt.ask = lambda t, m: True
for _k in range(2):                              # two normal episodes for M3 - M5
    vt.btn_rec.invoke()
    assert wait(lambda: vt._rec_phase == "recording")
    pump(0.6)
    assert vt.v_limit_warn.get() == ""
    vt.btn_rec.invoke()
    assert wait(lambda: vt._rec_phase == "idle", 30.0)
eps = vt.dataset.list_episodes()
assert [e["limit_ok"] for e in eps] == [False, True, True], [e.get("limit_ok") for e in eps]
print(f"[M2] J7 outside the range: log + 1 bell + red text; kept -> 超限 red; deleted on yes ({len(eps)} episodes)")

# ---- M3 player prompts from the task text, editable
seen_prompts = []
orig_detect = cam.det_yolo.detect


def det_wrap(rgb, mode, path, ref=None, prompts=None, conf=0.25):
    seen_prompts.append(tuple(prompts or ()))
    return [{"box": [100, 100, 300, 300], "name": (prompts or ["x"])[0], "conf": 0.9, "cls": 0}]


cam.det_yolo.detect = det_wrap
vt.refresh_episode_list()
last = str(eps[-1]["episode_index"])
vt.ep_tree.selection_set(last)
pump(0.2)
vt.btn_play_video.invoke()
assert wait(lambda: len(vt.players) == 1, 5.0)
pl = vt.players[0]
assert wait(lambda: bool(seen_prompts), 5.0)
assert seen_prompts[0] == ("red block", "box", "blue block"), seen_prompts[0]
assert pl.v_prompts.get() == "red block, box, blue block", pl.v_prompts.get()
pl.v_prompts.set("cup, plate")
pl.apply_prompts()
assert wait(lambda: ("cup", "plate") in seen_prompts, 5.0), seen_prompts[-3:]
assert all(d and d[0]["name"] == "cup" for d in pl.dets.values()), pl.dets
assert "cup, plate" in pl.v_note.get(), (pl.v_note.get(), pl._note0)
pl.close()
pump(0.2)
cam.det_yolo.detect = orig_detect
print("[M3] player YOLOE prompts from the task text: red block, box, blue block; text box -> cup, plate")

# ---- M4 edit the task text: two selected episodes, then the whole dataset
ids = [str(e["episode_index"]) for e in eps]
vt.refresh_episode_list()
vt.ep_tree.selection_set(ids[1:])
pump(0.2)
vt.on_edit_task()
assert vt._task_dlg is not None
v_scope, v_new, ok = vt._task_dlg_vars
assert v_new.get() == TASK
NEW = "Stack the red cup on the blue cup"
v_new.set(NEW)
n0 = len(LOG)
ok()
assert wait(lambda: vt._ds_busy is None, 60.0)
lg = LOG[n0:]
rows = {int(e["episode_index"]): e for e in vt.dataset.list_episodes()}
for i in ids[1:]:
    assert rows[int(i)]["tasks"] == [NEW], rows[int(i)]
    assert any(f"Episode #{i}：「{TASK}」→「{NEW}」" in s for s in lg), lg
    rd = TL.NpzVideoReader(vt.dataset._episode_npz_path(int(i)))
    assert rd.meta["tasks"] == [NEW] and rd.n > 3
    assert rd.frame("cam_main", 2) is not None
    rd.close()
assert rows[int(ids[0])]["tasks"] == [TASK]
assert any(t["task"] == NEW for t in vt.dataset.list_tasks())
bk = [s for s in lg if "已备份 meta" in s][0].split("→ ")[1]
assert os.path.isfile(os.path.join(bk, "episodes.jsonl"))
with open(os.path.join(bk, "episodes.jsonl"), encoding="utf-8") as f:
    assert NEW not in f.read()
assert vt.ep_tree.set(ids[1], "task") == NEW
vt.on_edit_task()
v_scope, v_new, ok = vt._task_dlg_vars
v_scope.set("all")
v_new.set("Sort the blocks into the boxes")
ok()
assert wait(lambda: vt._ds_busy is None, 60.0)
assert all(e["tasks"] == ["Sort the blocks into the boxes"] for e in vt.dataset.list_episodes())
print(f"[M4] 修改任务描述: 2 selected, then all {len(ids)}; jsonl + tasks.jsonl + episode files, backup {os.path.basename(bk)}")

# ---- M5 rename the dataset
old_dir = vt.dataset.dataset_dir
n0 = len(LOG)
assert not vt.rename_dataset_to("bad name!")
os.makedirs(os.path.join(vt.dataset.root_dir, "taken"), exist_ok=True)
assert not vt.rename_dataset_to("taken")
assert any("只能用英文字母" in s for s in LOG[n0:]) and any("已经有同名" in s for s in LOG[n0:])
from tkinter import simpledialog                         # noqa: E402
simpledialog.askstring = lambda *a, **k: "Block_into_boxes_1"
vt.on_rename_dataset()
pump(0.3)
assert vt.dataset.dataset_name == "Block_into_boxes_1" and vt.v_ds_name.get() == "Block_into_boxes_1"
assert not os.path.exists(old_dir) and os.path.isdir(vt.dataset.dataset_dir)
assert all(e["file"].startswith(vt.dataset.dataset_dir) for e in vt.dataset.list_episodes())
assert len(vt.ep_tree.get_children()) == len(ids)
vt.btn_rec.invoke()
assert wait(lambda: vt._rec_phase == "recording")
pump(0.5)
vt.btn_rec.invoke()
assert wait(lambda: vt._rec_phase == "idle", 30.0)
assert os.path.dirname(os.path.dirname(vt.dataset.list_episodes()[-1]["file"])) == vt.dataset.dataset_dir
print("[M5] 重命名数据集: bad / taken names refused; folder renamed, paths fixed, next recording goes there")

vt.shutdown()
tab.shutdown()
root.destroy()
SNAP1 = {str(p.relative_to(ROOT)): H(p.read_bytes()) for p in ROOT.rglob("*")
         if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and "logs" not in p.parts}
assert SNAP0 == SNAP1, set(SNAP0.items()) ^ set(SNAP1.items())
print("ALL V4.7.4 GUI TESTS PASSED")
