#!/usr/bin/env python3
"""V3.53 GUI test (needs tkinter + DISPLAY; skips otherwise)."""
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
try:
    import tkinter as tk
    from tkinter import messagebox
except Exception:
    print("[GUI53] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI53] no DISPLAY -> skipped"); sys.exit(0)

import NERO_V312_Console as C                 # noqa: E402
import NERO_V313_Real_Robot as RR             # noqa: E402
import NERO_V314_Joint_Control as JC          # noqa: E402
import NERO_V315_Consistency_Safety as CS     # noqa: E402
import NERO_V316_Safety_Editor as SE          # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402

TMP = tempfile.mkdtemp(prefix="v353gui_")
SE.STATE_PATH = os.path.join(TMP, "state.json")
CS.WORKSPACE_PATH = os.path.join(TMP, "ws.json")
CS.LOG_DIR = os.path.join(TMP, "logs")
JC.PARK_PATH = os.path.join(TMP, "park_pose.json")
CAM.LOCATE_PATH = CAM.DEFAULT_LOCATE_PATH = os.path.join(TMP, "locate_settings.json")
CAM.CALIB_DIR = os.path.join(TMP, "cameracalib")
os.makedirs(CAM.CALIB_DIR, exist_ok=True)

CAM.save_json(os.path.join(CAM.CALIB_DIR, "locate_settings.json"), {
    "version": "V3.47.1",
    "saved": "2026-10-30 09:00:00",
    "method": CAM.LOCATE_METHODS[0],
    "depth_res": "848×480",
    "depth_preset": "High Accuracy",
})

w = CS.default_workspace()
CS.save_workspace(w, CS.WORKSPACE_PATH)
messagebox.askyesno = lambda *a, **k: True
LIM = JC.build_model(open(os.path.join(HERE, "config", "nero.urdf"), encoding="utf-8").read(), None, None)["limits"]
D_T, _n = JC.park_target(JC.PARK_DEFAULT_DEG, LIM)

cap = {}
_orig_se_init = SE.SafetyEditorUI.__init__
_orig_cam_init = CAM.CameraTabUI.__init__


def se_init(self, *a, **k):
    _orig_se_init(self, *a, **k)
    cap["ed"] = self


def cam_init(self, *a, **k):
    _orig_cam_init(self, *a, **k)
    cap["cam"] = self


SE.SafetyEditorUI.__init__ = se_init
CAM.CameraTabUI.__init__ = cam_init
res = {}
_orig_mainloop = tk.Tk.mainloop


def walk(w_):
    yield w_
    for c in w_.winfo_children():
        yield from walk(c)


def logs(root):
    return "\n".join(w_.get("1.0", "end") for w_ in walk(root) if w_.winfo_class() == "Text")


def driver(root):
    ed = cap["ed"]
    cam = cap["cam"]
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

    def button_in(parent, text):
        return next(w_ for w_ in walk(parent) if w_.winfo_class() in ("TButton", "Button") and w_.cget("text") == text)

    def go():
        res["dres"] = cam.v_dres.get()
        nb_select("Camera调试")
        root.update_idletasks()
        root.update()
        cam_wf_bar = cam.workflow_bars[-1]
        b_in = cam_wf_bar.b_teach_in
        b_out = cam_wf_bar.b_teach_out
        b_copy = button_in(b_in.master, "Copy Log")
        cy_in = b_in.winfo_rooty() + b_in.winfo_height() / 2.0
        cy_out = b_out.winfo_rooty() + b_out.winfo_height() / 2.0
        cy_copy = b_copy.winfo_rooty() + b_copy.winfo_height() / 2.0
        res["align_diff_px"] = max(abs(cy_in - cy_copy), abs(cy_out - cy_copy))
        res["b_out_initial_state"] = str(b_out.cget("state"))

        teach_exit_calls = []
        orig_set_teach = pipe.set_teach_mode
        pipe.set_teach_mode = lambda on, **kw: (teach_exit_calls.append((on, kw.get("return_home"))), {"ok": True})[1]
        pipe.teach_mode = False
        b_out.invoke()
        wait(lambda: len(teach_exit_calls) >= 1, 5)
        res["cam_exit_when_flag_false"] = list(teach_exit_calls)

        nb_select("实机NERO调试")
        root.update()
        pipe.monitor.start([sys.executable, "-u", os.path.join(HERE, "tests", "v315_fake_monitor.py")],
                           env=dict(os.environ, FAKE_Q0="0,0.5,0,1.0,0,0.3,0", FAKE_RAD_S="20"), stdin_pipe=True)
        for k in RR.STEP_KEYS:
            pipe.state[k] = (RR.GREEN, "fake")
        wait(lambda: jp.enabled)
        root.update()

        pipe.teach_mode = True
        btn_real_exit = next(w_ for w_ in walk(root)
                             if w_.winfo_class() in ("TButton", "Button") and w_.cget("text") == "退出示教模式"
                             and w_ not in (wbar.b_teach_out for wbar in cam.workflow_bars))
        btn_real_exit.invoke()
        wait(lambda: len(teach_exit_calls) >= 2, 5)
        pipe.teach_mode = False
        res["real_exit_call"] = teach_exit_calls[-1]
        pipe.set_teach_mode = orig_set_teach

        btn_park = next(w_ for w_ in walk(root) if w_.winfo_class() in ("TButton", "Button") and w_.cget("text") == "待断电位置⚠")
        btn_park.invoke()
        wait(lambda: "[OK] [V3.24 park] 已到待断电位置 D" in logs(root), 40)
        res["park_deg"] = jp.actual_deg()
        res["logs"] = logs(root)
        pipe.monitor.send({"quit": True})
        root.after(200, root.destroy)

    root.after(800, go)


tk.Tk.mainloop = lambda self, n=0: (driver(self), _orig_mainloop(self, n))
C.run_console("real")

assert res["dres"] in ("1280×720", "848×480"), f"Expected 1280×720 (V3.53) or 848×480 (V4.6.10 start default), got {res['dres']}"
assert res["align_diff_px"] <= 1.0, f"Buttons not vertically aligned with Copy Log: diff={res['align_diff_px']} px"
assert res["b_out_initial_state"] == "normal", f"退出示教模式 was greyed out: {res['b_out_initial_state']}"
assert res["cam_exit_when_flag_false"] == [(False, False)], res["cam_exit_when_flag_false"]
assert res["real_exit_call"] == (False, False), res["real_exit_call"]
assert res["park_deg"] and max(abs(a - b) for a, b in zip(res["park_deg"], D_T)) < 0.5, (res["park_deg"], D_T)
print("ALL V3.53 GUI TESTS PASSED")
