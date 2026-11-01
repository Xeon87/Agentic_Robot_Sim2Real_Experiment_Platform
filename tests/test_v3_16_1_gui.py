#!/usr/bin/env python3
"""V3.16.1 layout stability (needs tkinter + a display, e.g. xvfb-run). Skips otherwise.

The 实机 tab runs with the fake ROS monitor; then every live text (step messages, status rows, joint panel state,
column warnings, preview, TCP, consistency / safety banners, repeat summary...) is switched between empty, short and
very long multi-line values many times while the arm moves. The screen position and size of the widgets the user
clicks (step buttons, 执行, 停止运动, sub-tab contents, editor canvases, log) must never change.
"""
import os, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
try:
    import tkinter as tk
    from tkinter import messagebox
except Exception:
    print("[GUI161] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI161] no DISPLAY -> skipped"); sys.exit(0)
import tempfile
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

TMP = tempfile.mkdtemp(prefix="v3161gui_")
SE.STATE_PATH = os.path.join(TMP, "state.json")             # V3.19: never touch the real last-zone state
CS.WORKSPACE_PATH = os.path.join(TMP, "real_workspace.json")
CS.LOG_DIR = os.path.join(TMP, "logs")
messagebox.askyesno = lambda *a, **k: True
cap = {}
for cls, key in ((CS.ConsistencyTabUI, "cons"), (SE.SafetyEditorUI, "ed")):
    def mk(orig, key):
        def init(self, *a, **k):
            orig(self, *a, **k)
            cap[key] = self
        return init
    cls.__init__ = mk(cls.__init__, key)
res = {}
_orig = tk.Tk.mainloop
LONG = ("很长的状态文字 " * 30).strip()
MULTI = "第一行\n第二行比较长 " + "x" * 120 + "\n第三行\n第四行\n第五行\n第六行"
VALUES = ["", "短", LONG, MULTI, "ok"]


def walk(w):
    yield w
    for c in w.winfo_children():
        yield from walk(c)


def driver(root):
    cons, ed = cap["cons"], cap["ed"]
    pipe, jp = cons.pipe, cons.jpanel
    nbs = [w for w in walk(root) if w.winfo_class() == "TNotebook"]

    def select(text):
        for nb in nbs:
            for t in nb.tabs():
                if nb.tab(t, "text") == text:
                    nb.select(t)

    def probes():
        step_btns = [w for w in walk(root) if w.winfo_class() == "TButton" and str(w.cget("text")) == "执行"]
        return {"step1": step_btns[0], "step6": step_btns[5], "exec": jp.btn_exec, "cancel": jp.btn_cancel,
                "test": jp.btn_test, "dir": cons.btn_dir, "rep": cons.btn_rep, "top": ed.cv_top, "side": ed.cv_side,
                "log": [w for w in walk(root) if w.winfo_class() == "Text"][-1]}

    def geo(tab):
        select(tab)
        root.update()
        out = {}
        for k, w in probes().items():
            if w.winfo_ismapped():
                out[k] = (w.winfo_rootx(), w.winfo_rooty(), w.winfo_width(), w.winfo_height())
        return out

    def s0():
        pipe.monitor.start([sys.executable, "-u", os.path.join(HERE, "tests", "v315_fake_monitor.py")],
                           env=dict(os.environ, FAKE_RAD_S="2", FAKE_Q0="0,0.5,0,-0.6,0,-0.27,0"), stdin_pipe=True)
        for k in RR.STEP_KEYS:
            pipe.state[k] = (RR.GREEN, "fake")
        t_end = time.time() + 20
        while time.time() < t_end and not jp.enabled:
            root.update()
            time.sleep(0.05)
        for _ in range(10):
            root.update()
            time.sleep(0.05)
        base = {t: geo(t) for t in ("关节控制", "一致性测试", "工作区安全")}
        moved = []
        for it in range(25):
            v = VALUES[it % len(VALUES)]
            for k in RR.STEP_KEYS:
                pipe.state[k] = (RR.GREEN, v or "fake")
            m = dict(pipe.mon_status or {})
            m["services"] = {f"/svc_{i}_{v[:3]}": "x" for i in range(it % 9)}
            m["arm_status"] = {f"field_{i}": v[:20] for i in range(it % 12)}
            m["gripper_status"] = {"long": v} if it % 2 else None
            pipe.mon_status = m
            cons.v_rep.set(v)
            cons.v_st.set(v)
            ed.v_test.set(v)
            ed.v_teach_info.set(v)
            jp.cols[2]["warn"].configure(text=v)
            if it % 3 == 0:                            # real joint motion -> live values change too
                a = pipe.actual_arm_deg(jp.model)
                if a and not pipe.move_busy():
                    import math, threading
                    q = [math.radians(x) for x in a]
                    q[0] += math.radians(4 if it % 2 else -4)
                    threading.Thread(target=lambda q=q: pipe.move_joints(jp.model, q, 0.3), daemon=True).start()
            for _ in range(4):                          # let real_poll (200 ms) rewrite the texts
                root.update()
                time.sleep(0.06)
            for t in ("关节控制", "一致性测试", "工作区安全"):
                g = geo(t)
                for k, val in g.items():
                    if base[t].get(k) != val:
                        moved.append((it, t, k, base[t].get(k), val))
        res["moved"] = moved
        res["base"] = base
        pipe.monitor.send({"quit": True})
        root.after(300, root.destroy)
    root.after(1200, s0)


def patched(self, n=0):
    driver(self)
    _orig(self, n)


tk.Tk.mainloop = patched
C.run_console("real")
assert res.get("base") and all(res["base"][t] for t in res["base"]), res.get("base")
assert not res["moved"], res["moved"][:8]
n = sum(len(v) for v in res["base"].values())
print(f"[GUI161] layout stable: {n} clickable widgets never moved/resized over 25 rounds of changing live texts + motion")
