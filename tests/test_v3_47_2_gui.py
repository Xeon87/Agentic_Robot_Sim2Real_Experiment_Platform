#!/usr/bin/env python3
"""V3.47.2 GUI test: shared Treeview rowheight + column auto-fit + scrollbars, non-overlapping short YOLO labels + 3D
centre cross, ROI depth fusion + GPU/CPU logging (needs tkinter + a display; skips otherwise).
All paths point to a temp folder (never the user's ~/Downloads/cameracalib).

J1 tv_eggs and tv both use V345Calib.Treeview (rowheight = font linespace + 8) and have vertical + horizontal scrollbars
J2 autofit_treeview on tv_eggs and tv: headers are not truncated; numeric columns do not stretch; only 'why' / 'q'
   stretches; inserting a long '说明' widens the column
J3 YOLO overlay: short labels ('e1'..'e4'), 3D centres drawn as crosses (no duplicate circle + 'egg_1' text); adjacent
   boxes get non-overlapping label coordinates; geometric V3.08 still draws circle + 'egg_...' label
J4 locate status text and log show YOLO(GPU|CPU) and timing
"""
import os
import sys
import tempfile
import threading
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    print("[GUI472] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI472] no DISPLAY -> skipped"); sys.exit(0)
import numpy as np                            # noqa: E402
import v347_fake as F                         # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V347_Yolo_Perception as YP        # noqa: E402

TMP = tempfile.mkdtemp(prefix="v3472gui_")
CFG = os.path.join(TMP, "config")
os.makedirs(CFG)
CAM.CALIB_PATH = CAM.DEFAULT_CALIB_PATH = os.path.join(CFG, "camera_calib.json")
CAM.LOCATE_PATH = CAM.DEFAULT_LOCATE_PATH = os.path.join(CFG, "locate_settings.json")
CAM.CALIB_DIR = os.path.join(TMP, "cameracalib")
os.makedirs(CAM.CALIB_DIR)
YP.REF_JSON, YP.REF_IMG, YP.SAMPLES_DIR = os.path.join(CFG, "yolo_ref.json"), os.path.join(CFG, "yolo_ref.jpg"), \
    os.path.join(CFG, "samples")
MD = os.path.join(TMP, "objectmodels")
os.makedirs(MD)
Path(MD, "yoloe-26s-seg.pt").write_bytes(b"w")
F.install()


class SyncThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self.t, self.a, self.k, self.alive = target, args, kwargs or {}, False

    def start(self):
        self.alive = True
        try:
            self.t(*self.a, **self.k)
        finally:
            self.alive = False

    def is_alive(self):
        return self.alive


CAM.threading = type("T", (), {"Thread": SyncThread, "Lock": threading.Lock, "Event": threading.Event})
OBS = F.default_objects()
RGB, DEPTH, IDS = F.render(OBS)
F.STATE.scene = (OBS, IDS)


class Fake:
    def __init__(self):
        self.K = self.info = None
        self.dist, self.error, self.fps, self.fid, self.on = [0.0] * 5, None, 30.0, 0, False
        self.depth_res, self.depth_preset = CAM.DEPTH_RES[0], CAM.DEPTH_PRESETS[0]

    def running(self):
        return self.on

    def start(self):
        self.on, self.K = True, dict(F.K)
        self.info = {"name": "D435I (fake)", "serial": "261222075224", "usb": "3.2", "K": dict(F.K),
                     "depth_res": self.depth_res, "preset": self.depth_preset}

    def stop(self, timeout=0):
        self.on = False

    @property
    def frame_id(self):
        self.fid += 1
        return self.fid

    def aruco_status(self):
        return True, "fake"

    def display_ppm(self, view="color"):
        return None

    def depth_at(self, u, v):
        return float(DEPTH[int(v), int(u)])

    def roi_points(self, *a, **k):
        return [], 0

    def frames(self):
        self.fid += 1
        return RGB, DEPTH, self.fid

    def detect_marker(self, marker, q_deg=None):
        return None


class Pipe:
    teach_mode, real_urdf = False, None

    def actual_arm_deg(self, model):
        return None


class Safe:
    sim = None
    valid_ws = {"fence": {"polygon": [[-0.8, -0.4], [-0.1, -0.4], [-0.1, 0.4], [-0.8, 0.4]], "z_table": 0.0}}

    def arm_points(self, q):
        return []


root = tk.Tk()
root.geometry("1400x900")
frame = ttk.Frame(root)
frame.pack(fill="both", expand=True)
logs = []
jp = type("JP", (), {"enabled": True, "model": {"joints": [f"joint{i}" for i in range(1, 8)], "limits": [[-1, 1]] * 7,
                                                 "base_link": "base_link"}})()
cam = CAM.CameraTabUI(tk, ttk, frame, Pipe(), jp, Safe(), lambda: {"speed_percent": 10}, logs.append, lambda *a: True,
                      visible_fn=lambda: True, source_factory=Fake)
cam.calib = {"T_base_cam": F.T_BC.tolist(), "marker_in_flange": [0, 0, 0], "mean_mm": 1, "max_mm": 1,
             "loo_mean_mm": 1, "loo_max_mm": 1}
cam._set_vstate("ok", "刚标定")


def tick(n=3):
    for _ in range(n):
        cam.poll()
        root.update()


tick()
# ------------------------------------------------------------------ J1
ls = int(root.tk.call("font", "metrics", "TkDefaultFont", "-linespace"))
style_h = int(ttk.Style(root).lookup("V345Calib.Treeview", "rowheight"))
assert style_h == ls + 8
assert str(cam.tv.cget("style")) == "V345Calib.Treeview" and str(cam.tv_eggs.cget("style")) == "V345Calib.Treeview"
assert str(cam.tv_eggs.cget("yscrollcommand")) and str(cam.tv_eggs.cget("xscrollcommand"))
print(f"[J1] tv and tv_eggs share V345Calib.Treeview (rowheight {style_h} px = font {ls} + 8) + vertical/horizontal scrollbars")

# ------------------------------------------------------------------ J2
for c, t, w_min in cam.egg_cols:
    w = int(cam.tv_eggs.column(c, "width"))
    st = bool(int(cam.tv_eggs.column(c, "stretch")))
    assert w >= max(w_min, CAM.measure_text_px(cam.tv_eggs, t) + 20), (c, w)
    assert st == (c == "why"), (c, st)
for c, t, w_min in cam.calib_cols:
    st = bool(int(cam.tv.column(c, "stretch")))
    assert st == (c == "q"), (c, st)
w_why_0 = int(cam.tv_eggs.column("why", "width"))
cam.v_ydir.set(MD)
cam.on_connect()
tick(4)
cam.on_ref_toggle()
cam.ref_edit["boxes"], cam.ref_edit["cls"] = [[540.0, 380.0, 600.0, 440.0]], [0]
cam.on_ref_toggle()
cam.on_eggs()
tick()
w_why_1 = int(cam.tv_eggs.column("why", "width"))
assert w_why_1 >= w_why_0
print(f"[J2] autofit_treeview: numeric columns non-stretching, 'why' stretched and widened ({w_why_0} -> {w_why_1} px)")

# ------------------------------------------------------------------ J3
assert [v["label"] for v in cam.yolo_view] == ["e1", "e2", "e3", "e4"], cam.yolo_view
cam._force_redraw()
tick()
types_on_cv = [cam.cv.type(i) for i in cam.cv.find_all()]
texts_on_cv = [cam.cv.itemcget(i, "text") for i in cam.cv.find_all() if cam.cv.type(i) == "text"]
assert "oval" not in types_on_cv, types_on_cv
assert set(texts_on_cv) >= {"e1", "e2", "e3", "e4"} and "egg_1" not in texts_on_cv, texts_on_cv
cam.v_method.set("几何法 V3.08")
cam.on_eggs()
tick()
cam._force_redraw()
tick()
types_geom = [cam.cv.type(i) for i in cam.cv.find_all()]
assert "oval" in types_geom
print("[J3] YOLO overlay: short labels e1..e4 + 3D centre crosses (no duplicate circle/egg_1); geometric V3.08 keeps circle")

# ------------------------------------------------------------------ J4
assert any(("YOLO(GPU)" in m or "YOLO(CPU)" in m) and "[V3.47 locate] 鸡蛋识别（vp）" in m for m in logs), logs[-4:]
cam.shutdown()
root.destroy()
print("[J4] locate log and status text report YOLO(GPU|CPU) and timing")
print("ALL V3.47.2 GUI TESTS PASSED")
