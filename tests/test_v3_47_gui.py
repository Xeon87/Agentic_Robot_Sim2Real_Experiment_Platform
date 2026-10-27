#!/usr/bin/env python3
"""V3.47 GUI test of the 视觉定位 page (needs tkinter + a display; skips otherwise).

Camera tab alone, fake D435i serving the ray-cast scene of v347_fake (two eggs, AirPods-like box, egg at the border),
fake ultralytics, worker jobs synchronous.
G1 page: method (default YOLOE 视觉提示), model folder / model, threshold, text prompt, egg size 55 x 42, reference row
G2 visual prompt without a reference -> clear message; 框选参考: drag on the image -> boxes (鸡蛋 / 非鸡蛋) -> 保存参考
G3 识别鸡蛋: table rows egg / egg / not trusted (border, AirPods-like silhouette); overlay drawn; z_table offset shown;
   sample saving on -> jpg + json
G4 text prompt and the old geometric method still run from the same button
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
    print("[GUI47] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI47] no DISPLAY -> skipped"); sys.exit(0)
import numpy as np                            # noqa: E402
import v347_fake as F                         # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V347_Yolo_Perception as YP        # noqa: E402

TMP = tempfile.mkdtemp(prefix="v347gui_")
CAM.CALIB_PATH = os.path.join(TMP, "camera_calib.json")
CAM.LOCATE_PATH = os.path.join(TMP, "locate_settings.json")      # V3.47.1: never the user's files
CAM.CALIB_DIR = os.path.join(TMP, "cameracalib")
YP.REF_JSON, YP.REF_IMG, YP.SAMPLES_DIR = os.path.join(TMP, "yolo_ref.json"), os.path.join(TMP, "yolo_ref.jpg"), \
    os.path.join(TMP, "samples")
MD = os.path.join(TMP, "objectmodels")
os.makedirs(MD)
Path(MD, "yoloe-26s-seg.pt").write_bytes(b"w")
Path(MD, "mobileclip2_b.ts").write_bytes(b"e")
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

    def running(self):
        return self.on

    def start(self):
        self.on, self.K = True, dict(F.K)
        self.info = {"name": "D435I (fake)", "serial": "1", "usb": "3.2", "K": dict(F.K)}

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
    valid_ws = {"fence": {"polygon": [[-0.8, -0.4], [-0.1, -0.4], [-0.1, 0.4], [-0.8, 0.4]], "z_table": -0.012}}

    def arm_points(self, q):
        return []


class Ev:
    def __init__(self, x, y):
        self.x, self.y = x, y


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


def tick(n=3):
    for _ in range(n):
        cam.poll()
        root.update()


tick()
# ------------------------------------------------------------------ G1
assert cam.v_method.get() == "YOLOE 视觉提示" and cam.v_egg_l.get() == "55" and cam.v_egg_w.get() == "42"
assert cam.v_ymodel.get() == "yoloe-26s-seg.pt" and "white round object" in cam.v_yprompt.get()
assert [cam.tv_eggs.heading(c, "text") for c in cam.tv_eggs["columns"]] == \
    ["编号", "x mm", "y mm", "z mm", "实测高", "YOLO", "轮廓重合", "置信度", "可信", "说明"]
assert "还没有参考" in cam.v_ref.get()
cam.v_ydir.set(MD)
print("[G1] 视觉定位: method YOLOE 视觉提示 (default), model folder / model / threshold / text prompt, egg 55 x 42 mm")
# ------------------------------------------------------------------ G2
cam.on_connect()
tick(4)
cam.on_eggs()
tick()
assert any("还没有参考图" in m for m in logs), logs[-3:]
cam.on_ref_toggle()
assert cam.ref_edit is not None and "框选中" in cam.v_ref.get()
e1 = OBS[0]
ys, xs = np.nonzero(IDS == 0)
x0, y0, x1, y1 = xs.min() / 2 - 4, ys.min() / 2 - 4, xs.max() / 2 + 4, ys.max() / 2 + 4
cam.on_press(Ev(x0, y0)); cam.on_drag(Ev(x1, y1)); cam.on_release(Ev(x1, y1))
cam.v_refcls.set("not")
ys, xs = np.nonzero(IDS == 2)
cam.on_press(Ev(xs.min() / 2, ys.min() / 2)); cam.on_drag(Ev(xs.max() / 2, ys.max() / 2)); cam.on_release(Ev(xs.max() / 2, ys.max() / 2))
assert cam.ref_edit["cls"] == [0, 1] and "2 个框" in cam.b_ref.cget("text") and cam.roi is None
cam._force_redraw(); tick()
cam.on_ref_toggle()
assert cam.ref_edit is None and cam.ref["cls"] == [0, 1] and os.path.isfile(YP.REF_IMG) and "1 个非鸡蛋" in cam.v_ref.get()
print("[G2] no reference -> message; 框选参考: two drags (鸡蛋 + 非鸡蛋) on the frozen frame -> 保存参考 (depth ROI untouched)")
# ------------------------------------------------------------------ G3
F.STATE.not_egg_names = set()                 # the fake still calls the AirPods "egg": geometry has to reject it
cam.v_ysave.set(True)
cam.on_eggs()
tick()
rows = [cam.tv_eggs.item(i, "values") for i in cam.tv_eggs.get_children()]
assert len(rows) == 4, (rows, logs[-4:])
st = [r[8] for r in rows]                     # V3.47.1: AirPods-like box -> silhouette does not match -> not trusted
assert [r[0] for r in rows] == ["egg_1", "egg_2", "egg_3", "egg_4"] and st[:2] == ["✓", "✓"] and st[2:] == ["-", "-"], rows
assert any("边缘" in r[9] for r in rows[2:]) and all(r[9] and not r[9].startswith("✓") for r in rows[2:]), rows
txt = cam.v_eggs.get()
assert "鸡蛋 2" in txt and "不可信 2" in txt and "z_table 高 12 mm" in txt and "建议在安全设置里更新 z_table" in txt, txt
assert "样本 →" in txt and len(os.listdir(YP.SAMPLES_DIR)) == 2
assert len(cam.yolo_view) == 4 and len(cam.eggs) == 4
cam._force_redraw(); tick()
assert not [m for m in logs if "叠加层" in m or "refresh" in m], [m for m in logs if "叠加层" in m or "refresh" in m]
assert any("[V3.47 locate] 鸡蛋识别（vp）" in m for m in logs)
print(f"[G3] 识别鸡蛋: {rows[0][0]} {rows[1][0]} trusted, border egg + AirPods-like not trusted ({rows[3][9][:24]}); "
      "overlay drawn; z_table offset shown; sample saved")
# ------------------------------------------------------------------ G4
cam.v_method.set("YOLOE 文字提示")
cam.on_eggs()
tick()
assert any("鸡蛋识别（text）" in m for m in logs) and F.STATE.set_classes_calls, logs[-3:]
cam.v_method.set("几何法 V3.08")
cam.on_eggs()
tick()
assert any("[V3.25 locate] 鸡蛋识别" in m or "[FAIL] [V3.25 locate]" in m for m in logs), logs[-3:]
assert not [m for m in logs if "V3.45 calib" in m and "FAIL" in m]
print("[G4] text prompt and the geometric method run from the same button")
cam.shutdown()
root.destroy()
print("ALL V3.47 GUI TESTS PASSED")
