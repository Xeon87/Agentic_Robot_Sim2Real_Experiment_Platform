#!/usr/bin/env python3
"""GUI smoke test for the V3.13 实机 tab (needs tkinter + a display, e.g. xvfb-run). Skips otherwise.

Opens the console on the 实机 tab, runs step 1 (on a machine without ROS/CAN it must turn RED, never crash),
checks the step-2 button stays disabled, the e-stop without a driver logs the fallback hint, and the sim start button
is enabled while nothing real runs.  NERO_V313_SHOT=<png> saves a screenshot.
"""
import os, subprocess, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
try:
    import tkinter as tk
    from tkinter import messagebox
except Exception:
    print("[GUI13] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI13] no DISPLAY -> skipped"); sys.exit(0)
import NERO_V312_Console as C
messagebox.askyesno = lambda *a, **k: True
res = {}
_orig = tk.Tk.mainloop


def walk(w):
    yield w
    for c in w.winfo_children():
        yield from walk(c)


def btns(root, text):
    return [w for w in walk(root) if w.winfo_class() in ("TButton", "Button") and str(w.cget("text")) == text]


def text_of_logs(root):
    return "\n".join(w.get("1.0", "end") for w in walk(root) if w.winfo_class() == "Text")


def driver(root):
    t0 = [time.time()]

    def s1():
        res["step2_state_before"] = str(btns(root, "执行")[1].cget("state"))
        btns(root, "执行")[0].invoke()
        btns(root, "■ 急停")[0].invoke()
        root.after(200, s2)

    def s2():
        logs = text_of_logs(root)
        if "[V3.13 1 环境]" in logs and ("all checks passed" in logs or "failed:" in logs) or time.time() - t0[0] > 60:
            root.update()
            res["logs"] = logs
            res["step2_state_after"] = str(btns(root, "执行")[1].cget("state"))
            res["sim_btn"] = str(btns(root, "▶ 启动仿真")[0].cget("state"))
            shot = os.environ.get("NERO_V313_SHOT")
            if shot:
                subprocess.run(["import", "-window", "root", shot], check=False)
            root.after(100, root.destroy)
        else:
            root.after(200, s2)
    root.after(800, s1)


def patched(self, n=0):
    driver(self)
    _orig(self, n)


tk.Tk.mainloop = patched
C.run_console("real")
L = res["logs"]
assert res["step2_state_before"] == "disabled", res
assert "[V3.13 1 环境]" in L and "急停服务不可用" in L, L[-800:]
if "failed:" in L:
    assert res["step2_state_after"] == "disabled"
assert res["sim_btn"] == "normal"
print("[GUI13] 实机 tab: step 1 runs and reports, later steps locked until green, e-stop fallback hint, sim not locked")
