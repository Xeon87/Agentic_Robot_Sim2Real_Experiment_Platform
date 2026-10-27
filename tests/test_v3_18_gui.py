#!/usr/bin/env python3
"""V3.18 GUI test (needs tkinter + a display, e.g. xvfb-run). Skips otherwise.

F1 target follow: the arm is moved by something else (homing after 一键启动) -> every slider follows to the actual
   angles without the user pressing 「同步为当前」; after the user edits a target, external motion no longer
   overwrites it; executing the edited target resumes following.
F2 side view: no explanatory text inside the plot, gutter labels never overlap — also with an enlarged font
   (the user's desktop renders Noto Sans CJK bigger than this test machine).
"""
import math, os, sys, tempfile, threading, time
HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
try:
    import tkinter as tk
    import tkinter.font as tkfont
    from tkinter import messagebox
except Exception:
    print("[GUI18] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI18] no DISPLAY -> skipped"); sys.exit(0)
import NERO_V312_Console as C
import NERO_V315_Consistency_Safety as CS
_v323_dw = CS.default_workspace


def _v323_legacy_default():          # V3.23: this test covers the V3.16-V3.22 layered rule (now rule B, 4 corners)
    w = _v323_dw()
    w["fence"].update(rule="B", margin=0.04, polygon=[[-0.15, 0.45], [-0.70, 0.45], [-0.70, -0.45], [-0.15, -0.45]])
    w["corners"] = w["corners"][:4]
    return w


CS.default_workspace = _v323_legacy_default
import NERO_V316_Safety_Editor as SE
import NERO_V313_Real_Robot as RR

TMP = tempfile.mkdtemp(prefix="v318gui_")
SE.STATE_PATH = os.path.join(TMP, "state.json")             # V3.19: never touch the real last-zone state
CS.WORKSPACE_PATH = os.path.join(TMP, "w.json")
CS.LOG_DIR = os.path.join(TMP, "logs")
messagebox.askyesno = lambda *a, **k: True
cap = {}
_i = SE.SafetyEditorUI.__init__


def init(self, *a, **k):
    _i(self, *a, **k)
    cap["ed"] = self


SE.SafetyEditorUI.__init__ = init
res = {}
_orig = tk.Tk.mainloop


def side_check(ed):
    c = ed.cv_side
    ed.redraw()
    c.update_idletasks()
    S, ox, oz = ed._side_tf()
    h0, h1 = ed._side_range()
    L, R = ed.w2s(h0, 0)[0], ed.w2s(h1, 0)[0]
    gutter, inside = [], []
    for it in c.find_all():
        if c.type(it) != "text":
            continue
        bb = c.bbox(it)
        if bb[0] > R:
            gutter.append(bb)
        elif bb[2] > L + 2:
            inside.append(c.itemcget(it, "text"))
    gutter.sort(key=lambda b: b[1])
    overl = [(a, b) for a, b in zip(gutter, gutter[1:]) if b[1] < a[3] - 1]
    return gutter, inside, overl


def driver(root):
    ed = cap["ed"]
    pipe, jp = ed.pipe, ed.jpanel

    def wait_for(cond, t=20):
        t_end = time.time() + t
        while time.time() < t_end and not cond():
            root.update()
            time.sleep(0.05)
        return cond()

    def settle(n=8):
        for _ in range(n):
            root.update()
            time.sleep(0.06)

    def move_external(q):
        th = threading.Thread(target=lambda: pipe.move_joints(jp.model, q, 0.5), daemon=True)
        th.start()
        while th.is_alive():
            root.update()
            time.sleep(0.05)
        settle()

    def s0():
        pipe.monitor.start([sys.executable, "-u", os.path.join(HERE, "tests", "v315_fake_monitor.py")],
                           env=dict(os.environ, FAKE_RAD_S="20", FAKE_Q0="0.3,0.6,0.2,-0.5,0.1,-0.3,0.2"), stdin_pipe=True)
        for k in RR.STEP_KEYS:
            pipe.state[k] = (RR.GREEN, "fake")
        pipe.safety_fn = None
        wait_for(lambda: jp.enabled)
        settle()
        res["start"] = [round(jp.cols[i]["var"].get(), 1) for i in range(7)]
        move_external([0.0] * 7)                          # "homing" done by the driver / RViz
        res["after_home"] = [round(jp.cols[i]["var"].get(), 1) for i in range(7)]
        res["follow_txt"] = jp.v_speed.get()
        jp.cols[1]["evar"].set("20")                      # user edits J2
        jp._on_entry(1)
        move_external([math.radians(5)] + [0.0] * 6)      # external move: must not overwrite the edit
        res["edited"] = [round(jp.cols[i]["var"].get(), 1) for i in range(7)]
        res["edited_txt"] = jp.v_speed.get()
        jp.on_execute()                                   # execute the edit -> follow again
        wait_for(lambda: not jp._busy() and not jp.targets_dirty, 30)
        settle(10)
        move_external([0.0] * 7)
        res["after_exec"] = [round(jp.cols[i]["var"].get(), 1) for i in range(7)]
        # F2 side view
        for side in ("XZ", "YZ"):
            ed.v_side.set(side)
            res[f"side_{side}"] = side_check(ed)
        f = tkfont.nametofont("TkDefaultFont")
        f.configure(size=14)                              # bigger font like the user's desktop
        ed.fence()["z_min_points"] = 0.012                # lines very close together
        ed.fence()["xy_free_height"] = 0.03
        res["side_big"] = side_check(ed)
        pipe.monitor.send({"quit": True})
        root.after(300, root.destroy)
    root.after(1200, s0)


def patched(self, n=0):
    driver(self)
    _orig(self, n)


tk.Tk.mainloop = patched
C.run_console("real")
st = res["start"]
assert abs(st[0] - 17.2) < 0.2 and abs(st[1] - 34.4) < 0.2, st
assert all(abs(v) < 0.15 for v in res["after_home"]), res["after_home"]
assert "跟随当前" in res["follow_txt"]
e = res["edited"]
assert abs(e[1] - 20.0) < 0.05 and abs(e[0]) < 0.15 and "已修改" in res["edited_txt"], (e, res["edited_txt"])
assert all(abs(v) < 0.15 for v in res["after_exec"]), res["after_exec"]
for k in ("side_XZ", "side_YZ", "side_big"):
    gutter, inside, overl = res[k]
    assert len(gutter) == 4 and not overl, (k, gutter, overl)
    assert all(t.strip().lstrip("-").isdigit() for t in inside), (k, inside)    # only axis numbers in the plot
print("[GUI18] sliders follow the arm (homing / external moves) until edited; edit kept; follow resumes after 执行; "
      "side view: no text in the plot, 4 gutter labels never overlap (XZ, YZ, 14 pt font + close lines)")
