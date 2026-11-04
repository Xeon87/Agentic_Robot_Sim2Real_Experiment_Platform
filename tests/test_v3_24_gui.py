#!/usr/bin/env python3
"""V3.24 GUI test (needs tkinter + a display). Skips otherwise.

Q1 top view: the corners behind the base can be dragged to 45 cm behind it (old view ended at 25 cm)
Q2 「待断电位置⚠」 (ops row, next to 回 Home): with the zone ending 12 cm behind the base MoveIt is refused and the log
   says how far back the zone must go; after dragging the zone back -> the arm reaches D (clamped to the soft limits)
Q3 「当前设为待断电位置」 stores the current angles in config/park_pose.json
"""
import json, math, os, sys, tempfile, time
HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
try:
    import tkinter as tk
    from tkinter import messagebox, ttk
except Exception:
    print("[GUI24] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI24] no DISPLAY -> skipped"); sys.exit(0)
import NERO_V312_Console as C
import NERO_V314_Joint_Control as JC
import NERO_V315_Consistency_Safety as CS
import NERO_V316_Safety_Editor as SE
import NERO_V313_Real_Robot as RR

TMP = tempfile.mkdtemp(prefix="v324gui_")
SE.STATE_PATH = os.path.join(TMP, "state.json")
CS.WORKSPACE_PATH = os.path.join(TMP, "ws.json")
CS.LOG_DIR = os.path.join(TMP, "logs")
JC.PARK_PATH = os.path.join(TMP, "park_pose.json")
w = CS.default_workspace()                       # rule A, 6 corners, 12 cm behind the base
CS.save_workspace(w, CS.WORKSPACE_PATH)
messagebox.askyesno = lambda *a, **k: True
LIM = JC.build_model(open(os.path.join(HERE, "config", "nero.urdf")).read(), None, None)["limits"]
D_T, _n = JC.park_target(JC.PARK_DEFAULT_DEG, LIM)
cap = {}
_i = SE.SafetyEditorUI.__init__


def init(self, *a, **k):
    _i(self, *a, **k)
    cap["ed"] = self


SE.SafetyEditorUI.__init__ = init
res = {}
_orig = tk.Tk.mainloop


def walk(w_):
    yield w_
    for c in w_.winfo_children():
        yield from walk(c)


def logs(root):
    return "\n".join(w_.get("1.0", "end") for w_ in walk(root) if w_.winfo_class() == "Text")


def driver(root):
    ed = cap["ed"]
    pipe = ed.pipe
    jp = ed.jpanel

    def wait(cond, t=30):
        t_end = time.time() + t
        while time.time() < t_end and not cond():
            root.update()
            time.sleep(0.05)
        return cond()

    def nb_select(text):
        for w_ in walk(root):
            if w_.winfo_class() == "TNotebook":
                for t in w_.tabs():
                    if w_.tab(t, "text") == text:
                        w_.select(t)

    def button(text):
        return next(w_ for w_ in walk(root) if w_.winfo_class() in ("TButton", "Button") and w_.cget("text") == text)

    def drag(cv, a, b, steps=8):
        cv.event_generate("<ButtonPress-1>", x=int(a[0]), y=int(a[1]))
        for k in range(1, steps + 1):
            cv.event_generate("<B1-Motion>", x=int(a[0] + (b[0] - a[0]) * k / steps), y=int(a[1] + (b[1] - a[1]) * k / steps))
        cv.event_generate("<ButtonRelease-1>", x=int(b[0]), y=int(b[1]))

    def go():
        pipe.monitor.start([sys.executable, "-u", os.path.join(HERE, "tests", "v315_fake_monitor.py")],
                           env=dict(os.environ, FAKE_Q0="0,0.5,0,1.0,0,0.3,0", FAKE_RAD_S="20"), stdin_pipe=True)
        for k in RR.STEP_KEYS:
            pipe.state[k] = (RR.GREEN, "fake")
        wait(lambda: jp.enabled)
        root.update()
        # Q2a: zone ends 12 cm behind the base -> refused + hint
        button("待断电位置⚠").invoke()
        wait(lambda: "[V3.24 park]" in logs(root) and "安全区后边界要到" in logs(root), 40)
        res["refused_log"] = logs(root)
        # Q1: drag the two corners behind the base to 45 cm behind
        nb_select("实机NERO调试")
        nb_select("工作区安全")
        root.update()
        time.sleep(0.3)
        root.update()
        poly = ed.fence()["polygon"]
        for k, (x, y) in enumerate(list(poly)):
            if x > 0.05:
                drag(ed.cv_top, ed.w2t(x, y), ed.w2t(0.45, y))
                root.update()
        res["poly"] = [list(p) for p in ed.fence()["polygon"]]
        ed.on_save()
        wait(lambda: "[V3.23] 围墙" in logs(root).split("已保存到")[-1], 20)
        # Q2b: now D is reachable
        n0 = logs(root).count("[OK] [V3.24 park]")
        button("待断电位置⚠").invoke()
        wait(lambda: logs(root).count("[OK] [V3.24 park]") > n0, 60)
        a = jp.actual_deg()
        res["a_end"] = a
        # Q3
        jp.on_park_record()
        res["park_file"] = json.load(open(JC.PARK_PATH))
        res["logs"] = logs(root)
        pipe.monitor.send({"quit": True})
        root.after(300, root.destroy)
    root.after(1200, go)


def patched(self, n=0):
    driver(self)
    _orig(self, n)


tk.Tk.mainloop = patched
C.run_console("real")
assert "安全区后边界要到 x ≥ +0.3" in res["refused_log"] or "安全区后边界要到 x ≥ +0.4" in res["refused_log"], \
    res["refused_log"][-1500:]
back = [p for p in res["poly"] if p[0] > 0.05]
assert len(back) == 2 and all(abs(p[0] - 0.45) < 0.01 for p in back), res["poly"]
assert res["a_end"] and max(abs(a - b) for a, b in zip(res["a_end"], D_T)) < 0.5, (res["a_end"], D_T)
assert "已到待断电位置 D" in res["logs"]
assert max(abs(a - b) for a, b in zip(res["park_file"]["deg"], res["a_end"])) < 0.05
print("[GUI24] zone corners dragged to 45 cm behind the base; 待断电位置: refused with a 'move the back wall to x ≥ …' "
      "hint while the zone ended 12 cm behind, reached D (soft-limit clamped) after; 当前设为待断电位置 saved")
