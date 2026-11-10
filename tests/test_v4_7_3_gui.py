#!/usr/bin/env python3
"""V4.7.3 GUI test (needs tkinter + DISPLAY; skips otherwise).

K1 episode list row height: the 「Episode 浏览」 table uses a style whose row height comes from the font
   (linespace + 8) - the default fixed row height made the Chinese rows overlap on the user's scaled screen.
K2 player playback WITH a slow YOLOE (0.12 s per call, 2 calls per frame = ~5x slower than the 20 fps video):
   the first frame is on screen at once (not after the model), during playback many different frames are drawn
   (V4.7: none - black canvas, then one still image at the end), labels of the current / previous 2 frames are
   drawn on top, paused -> the exact frame's labels.
K3 「开始示教录制」 on the real back end: 是 -> moves to ready_to_work at 10 % (pose library first, built-in
   2026-11-09 pose otherwise), THEN teach mode, THEN recording; 否 -> no move, teach + record; 取消 -> nothing;
   a failed move -> no teach mode, no recording.
"""
import os
import sys
import tempfile
import threading
import time
import types

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    print("[SKIP] test_v4_7_3_gui: no tkinter"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[SKIP] test_v4_7_3_gui: no DISPLAY"); sys.exit(0)

import numpy as np                              # noqa: E402

TMP = tempfile.mkdtemp(prefix="nero_v473_gui_")
os.environ["NERO_CALIB_DIR"] = os.path.join(TMP, "calib")
os.environ["NERO_DRIVE_ROOT"] = os.path.join(TMP, "drive")
os.makedirs(os.environ["NERO_DRIVE_ROOT"], exist_ok=True)

import NERO_V4_Dataset_Recorder as DR           # noqa: E402
import NERO_V47_VLA_Tools as TL                 # noqa: E402
import NERO_V4_Console_Tab as VT                # noqa: E402
import NERO_V314_Joint_Control as JC            # noqa: E402

root = tk.Tk()
root.geometry("1200x700")


def pump(t):
    t_end = time.time() + t
    while time.time() < t_end:
        root.update()
        time.sleep(0.005)


import queue as _queue                          # noqa: E402
POSTQ = _queue.Queue()


def drain():
    while True:
        try:
            POSTQ.get_nowait()()
        except _queue.Empty:
            return


def wait(fn, t):
    t_end = time.time() + t
    while time.time() < t_end:
        root.update()
        drain()
        if fn():
            return True
        time.sleep(0.005)
    return fn()


# ---- K1 row height
src = open(os.path.join(ROOT, "NERO_V4_Console_Tab.py"), encoding="utf-8").read()
assert 'style=self._ep_style' in src and '"V473Episodes.Treeview"' in src
import NERO_V325_Camera as CAM                  # noqa: E402
fr = ttk.Frame(root)
name, rh = CAM.treeview_style(ttk, fr, "V473Episodes.Treeview")
ls = int(fr.tk.call("font", "metrics", "TkDefaultFont", "-linespace"))
assert int(ttk.Style(fr).lookup(name, "rowheight")) == ls + 8 == rh, (ttk.Style(fr).lookup(name, "rowheight"), ls)
print(f"[K1] episode table row height = font line height {ls} + 8 px")

# ---- K2 player with a slow YOLOE
ds = DR.NEROLeRobotDataset(dataset_name="v473", root_dir=os.path.join(TMP, "ds"))
frames = []
for k in range(60):
    st = np.zeros(8, np.float32)
    img = np.zeros((480, 640, 3), np.uint8)
    img[:, :, 0] = k * 4                                   # every frame different
    frames.append({"timestamp_ns": int(k * 5e7), "sync_skew_ms": 1.0, "state": st, "action": st.copy(),
                   "camera1": img, "camera2": img[:, ::-1].copy()})
meta = ds.save_episode(frames, task_instruction="demo")
calls = []


def slow_yolo(rgb):
    calls.append(int(rgb[0, 0, 0]) // 4 if rgb[0, 0, 0] else 0)
    time.sleep(0.12)
    return [{"box": [100, 100, 300, 300], "name": "egg", "conf": 0.9, "cls": 0}]


t0 = time.time()
pl = TL.EpisodePlayer(tk, ttk, root, ds._episode_npz_path(meta["episode_index"]), yolo_fn=slow_yolo, yolo_note="fake")
assert wait(lambda: pl.n_drawn >= 1, 1.0), "the first frame must not wait for YOLOE"
t_first = time.time() - t0
seen = set()
orig_redraw = pl._redraw


def spy():
    orig_redraw()
    seen.add(pl._cur[0])


pl._redraw = spy
pl.toggle()
assert pl.playing
wait(lambda: not pl.playing, 8.0)
assert not pl.playing and pl.i == pl.rd.n - 1
assert len(seen) >= 30, f"only {len(seen)} different frames drawn during playback"
assert wait(lambda: pl._cur[0] == pl.rd.n - 1, 1.0), pl._cur[0]
n_yolo_frames = len({k[1] for k in pl.dets})
assert 2 <= n_yolo_frames < pl.rd.n, n_yolo_frames            # YOLOE keeps up only partly, video does not wait
pl.step(-10)
assert wait(lambda: ("cam_main", pl.i) in pl.dets and ("cam_wrist", pl.i) in pl.dets, 3.0)
pump(0.1)
assert pl._cur[0] == pl.i
print(f"[K2] first frame after {t_first:.2f} s; playback drew {len(seen)} / {pl.rd.n} frames with a 0.24 s/frame "
      f"YOLOE (labels on {n_yolo_frames} frames); paused -> exact labels")
pl.close()

# ---- K3 ready_to_work before teach recording
JC.POSES_PATH = os.path.join(TMP, "real_poses.json")
q, src_ = VT.ready_to_work_deg(JC.POSES_PATH)
assert q == VT.READY_TO_WORK_DEFAULT_DEG == [-0.2, 21.6, -0.1, 61.2, 0.1, 0.2, 44.8] and "内置" in src_
JC.PoseLibrary(JC.POSES_PATH).save("ready_to_work", [1, 2, 3, 4, 5, 6, 7])
q, src_ = VT.ready_to_work_deg(JC.POSES_PATH)
assert q == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] and "姿态库" in src_
shipped = VT.ready_to_work_deg(os.path.join(ROOT, "config", "real_poses.json"))
assert shipped[0] == VT.READY_TO_WORK_DEFAULT_DEG, shipped


class Pipe:
    def __init__(self, ok=True):
        self.teach_mode, self.moves, self.ok, self.events = False, [], ok, []

    def set_teach_mode(self, on):
        self.teach_mode = on
        self.events.append("teach")
        return {"ok": True}

    def move_and_settle(self, model, target_rad, speed, label=""):
        time.sleep(0.05)
        self.moves.append(([round(np.degrees(v), 2) for v in target_rad], speed))
        self.events.append("move")
        return {"ok": True} if self.ok else {"ok": False, "error": "规划失败"}


class Bar:
    def __init__(self, pipe):
        self.pipe = pipe

    def enter_now(self, done=None):
        threading.Thread(target=lambda: done(self.pipe.set_teach_mode(True)), daemon=True).start()
        return True


class Arb:
    mode = "monitor"

    def request_mode(self, m, owner_tag=""):
        self.mode = m

    def set_allow_control(self, on, reason=""):
        self.allow = on


class Rec:
    fps = 20.0

    def __init__(self, pipe):
        self.pipe, self.started = pipe, []

    def start_recording(self, task, split="train"):
        self.pipe.events.append("record")
        self.started.append(task)
        return {"ok": True}


def make_tab(answer, ok=True):
    import queue
    t = VT.VLALeRobotTabUI.__new__(VT.VLALeRobotTabUI)
    t.q = queue.Queue()
    t.logs = []
    t._log = lambda m: t.logs.append(str(m))
    t.ext_log = lambda m: None
    t.pipe = Pipe(ok)
    t.teach_bar = Bar(t.pipe)
    t.arbiter = Arb()
    t.jpanel = types.SimpleNamespace(model={"joints": [f"joint{i}" for i in range(1, 8)]})
    t.v_backend, t.v_ds_name, t.v_task = tk.StringVar(value="real"), tk.StringVar(value="v473"), tk.StringVar(value="demo")
    t.v_mode, t.v_allow, t.v_rec_status = tk.StringVar(), tk.BooleanVar(), tk.StringVar()
    t.dataset = ds
    t.recorder = Rec(t.pipe)
    t.parent = root
    t.btn_rec, t.btn_discard = tk.Button(root), tk.Button(root)
    t._rec_phase = "idle"
    t.ask = lambda *a: True
    t.asked = []
    t.ask_ready_to_work = lambda qd, s: t.asked.append((list(qd), s)) or answer
    t._post = POSTQ.put                         # as the tab: worker -> queue -> Tk thread
    return t


for ans, ok, want in ((True, True, ["move", "teach", "record"]), (False, True, ["teach", "record"]),
                      (None, True, []), (True, False, ["move"])):
    t = make_tab(ans, ok)
    t.on_toggle_record()
    wait(lambda: t._rec_phase in ("recording", "idle") and len(t.pipe.events) >= len(want), 3.0)
    pump(0.2)
    assert t.pipe.events == want, (ans, ok, t.pipe.events, t.logs)
    if ans is True:
        assert t.pipe.moves[0][0] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0] and t.pipe.moves[0][1] == 0.10, t.pipe.moves
        assert len(t.asked) == 1
    assert t._rec_phase == ("recording" if want[-1:] == ["record"] else "idle"), t._rec_phase
    if not ok:
        assert any("没到 ready_to_work" in m for m in t.logs), t.logs
print("[K3] 是 -> ready_to_work (10 %) -> teach -> record; 否 -> teach -> record; 取消 -> nothing; failed move -> stop")

root.destroy()
print("ALL V4.7.3 GUI TESTS PASSED")
