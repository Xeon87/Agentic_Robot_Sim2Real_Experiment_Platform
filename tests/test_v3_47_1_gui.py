#!/usr/bin/env python3
"""V3.47.1 GUI test: calibration files + check after loading, 视觉定位 settings files, depth settings (needs tkinter +
a display; skips otherwise).  All paths point to a temp folder (never the user's ~/Downloads/cameracalib).

H1 new NERO folder (no config/camera_calib.json): newest calibration in the calib folder is loaded -> 'pending';
   locate results not trusted until checked; serial / intrinsics checked on connect; 5 live marker errors <= 5 mm
   -> verified; > 10 mm -> failed
H2 另存为 / 读取: saved copy loads back, becomes config/camera_calib.json, marker settings restored, pending again;
   invalid file rejected; 保存标定 writes a timestamped copy into the calib folder; serial mismatch -> failed
H3 视觉定位 设置: 保存设置 -> config + calib folder copy (with the reference image); a new page restores everything
   (method, model, size, depth settings, reference); 另存为 / 读取
H4 depth settings are handed to the camera source on connect; eggs with no depth on them are located (trusted after
   the check)
"""
import json
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
    print("[GUI471] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI471] no DISPLAY -> skipped"); sys.exit(0)
import numpy as np                            # noqa: E402
import v347_fake as F                         # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V347_Yolo_Perception as YP        # noqa: E402

TMP = tempfile.mkdtemp(prefix="v3471gui_")
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
OBS = [o for o in F.default_objects() if o["name"] in ("E1", "E2")]
RGB, DEPTH, IDS = F.render(OBS)
for k in range(len(OBS)):
    DEPTH[IDS == k] = 0                       # D435 sees nothing on the white eggs
F.STATE.scene = (OBS, IDS)
SERIAL = {"v": "261222075224"}
MADE = []


class Fake:
    def __init__(self):
        self.K = self.info = None
        self.dist, self.error, self.fps, self.fid, self.on = [0.0] * 5, None, 30.0, 0, False
        self.depth_res, self.depth_preset = "?", "?"
        MADE.append(self)

    def running(self):
        return self.on

    def start(self):
        self.on, self.K = True, dict(F.K)
        self.info = {"name": "D435I (fake)", "serial": SERIAL["v"], "usb": "3.2", "K": dict(F.K),
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


CALIB = {"version": "V3.25", "saved": "2026-10-26 21:14:25", "camera_serial": "261222075224", "intrinsics": dict(F.K),
         "marker": {"dict": "DICT_4X4_50", "id": 0, "size_mm": 38.0}, "flange_link": "link7",
         "T_base_cam": F.T_BC.tolist(), "marker_in_flange": [0.0877, -0.0696, -0.0323], "mean_mm": 0.95, "max_mm": 1.8,
         "loo_mean_mm": 1.8, "loo_max_mm": 4.2, "n_used": 8, "check_poses_deg": [[0] * 7, [1] * 7], "samples": []}
old = os.path.join(CAM.CALIB_DIR, "camera_calib_20261020_100000.json")
CAM.save_json(old, dict(CALIB, saved="2026-10-20 10:00:00", loo_mean_mm=9.0))
os.utime(old, (1e9, 1e9))
CAM.save_json(os.path.join(CAM.CALIB_DIR, "camera_calib_20261026_211425.json"), CALIB)
Path(CAM.CALIB_DIR, "notes.json").write_text("{}")             # not a calibration: ignored

root = tk.Tk()
logs = []
jp = type("JP", (), {"enabled": True, "model": {"joints": [f"joint{i}" for i in range(1, 8)], "limits": [[-1, 1]] * 7,
                                                 "base_link": "base_link"}})()


def page():
    fr = ttk.Frame(root)
    fr.pack()
    return CAM.CameraTabUI(tk, ttk, fr, Pipe(), jp, Safe(), lambda: {"speed_percent": 10}, logs.append,
                           lambda *a: True, visible_fn=lambda: True, source_factory=Fake)


def tick(cam, n=3):
    for _ in range(n):
        cam.poll()
        root.update()


def eggs(cam):
    cam.on_eggs()
    tick(cam)
    return [cam.tv_eggs.item(i, "values") for i in cam.tv_eggs.get_children()]


cam = page()
tick(cam)
# ------------------------------------------------------------------ H1
assert cam.calib and cam.calib["saved"] == "2026-10-26 21:14:25" and cam.calib_src.endswith("211425.json"), cam.calib_src
assert cam.vstate["state"] == "pending" and any("config 里没有标定" in m for m in logs)
assert getattr(cam.l4, "_v345_state") == "yellow" and "还没校验" in cam.s4.get()
cam.v_ydir.set(MD)
cam.on_connect()
tick(cam, 4)
assert any("序列号 / 内参和标定一致" in m for m in logs)
assert cam.ref is None
cam.on_ref_toggle()
cam.ref_edit["boxes"], cam.ref_edit["cls"] = [[540.0, 380.0, 600.0, 440.0]], [0]
cam.on_ref_toggle()
rows = eggs(cam)
assert len(rows) == 2 and all(r[8] == "-" and "还没校验" in r[9] for r in rows), rows
assert "深度不足" in rows[0][9]
for e in (1.2, 2.0, 1.4, 0.9, 1.6):
    cam._verify_live(object(), e)
assert cam.vstate["state"] == "ok" and "误差 1.4 mm" in cam.vstate["why"]
tick(cam)
assert getattr(cam.l4, "_v345_state") == "green"
rows = eggs(cam)
assert [r[8] for r in rows] == ["✓", "✓"], rows
print("[H1] new folder: newest calibration from the calib folder (pending, results not trusted); serial / intrinsics "
      "ok; 5 live marker errors <= 5 mm -> verified -> eggs trusted (no depth on them)")
# ------------------------------------------------------------------ H2
cam._set_vstate("pending")
for e in (12, 15, 13, 14, 16):
    cam._verify_live(object(), e)
assert cam.vstate["state"] == "fail" and "标定校验没通过" in cam._calib_note()
out = os.path.join(TMP, "export", "my_calib.json")
os.makedirs(os.path.dirname(out))
cam.on_calib_save_as(out)
assert json.load(open(out))["T_base_cam"] == CALIB["T_base_cam"]
cam.on_calib_load(os.path.join(TMP, "nope.json"))
assert any("不是有效的标定文件" in m for m in logs)
cam.v_msize.set("40")
cam.on_calib_load(out)
assert cam.calib_src == out and cam.vstate["state"] == "pending" and cam.v_msize.get() == "38"
assert json.load(open(CAM.CALIB_PATH))["saved"] == CALIB["saved"] and "exported" not in json.load(open(CAM.CALIB_PATH))
n0 = len(os.listdir(CAM.CALIB_DIR))
cam.result = {"ok": True, "good": True, "T_base_cam": CALIB["T_base_cam"], "marker_in_flange": CALIB["marker_in_flange"],
              "mean_mm": 1.0, "max_mm": 2.0, "loo_mean_mm": 1.5, "loo_max_mm": 3.0, "used": [], "rot_span_deg": 40,
              "trans_span_m": 0.25}
cam.on_save_calib()
assert len(os.listdir(CAM.CALIB_DIR)) == n0 + 1 and cam.vstate["state"] == "ok" and cam._calib_note() is None
SERIAL["v"] = "999"
cam.on_disconnect()
cam._set_vstate("pending")
cam.on_connect()
tick(cam, 3)
assert cam.vstate["state"] == "fail" and "序列号不同" in cam.vstate["why"], cam.vstate
SERIAL["v"] = "261222075224"
print("[H2] 另存为 / 读取 (invalid file rejected, becomes config/camera_calib.json, marker restored, pending); "
      "保存标定 copy in the calib folder; other camera serial -> failed")
# ------------------------------------------------------------------ H3
cam.v_method.set("YOLOE 文字提示")
cam.v_egg_l.set("56")
cam.v_egg_w.set("43")
cam.v_dres.set("1280×720")
cam.v_dpreset.set("Default")
cam.v_yprompt.set("white egg")
cam.on_locate_save()
assert os.path.isfile(CAM.LOCATE_PATH) and os.path.isfile(os.path.join(CAM.CALIB_DIR, "locate_settings.json"))
d = json.load(open(os.path.join(CAM.CALIB_DIR, "locate_settings.json")))
assert d["reference"]["image_jpg_b64"] and d["egg_mm"] == ["56", "43"] and d["depth_res"] == "1280×720"
os.remove(CAM.LOCATE_PATH)
os.remove(YP.REF_JSON)
os.remove(YP.REF_IMG)
cam2 = page()                                          # "next start" in a new folder
assert cam2.v_method.get() == "YOLOE 文字提示" and cam2.v_egg_l.get() == "56" and cam2.v_dres.get() == "1280×720"
assert cam2.v_dpreset.get() == "Default" and cam2.v_yprompt.get() == "white egg" and cam2.ref and cam2.ref["cls"] == [0]
assert os.path.isfile(YP.REF_IMG)
alt = os.path.join(TMP, "export", "locate_alt.json")
cam2.v_egg_w.set("41")
cam2.on_locate_save_as(alt)
cam2.v_egg_w.set("40")
cam2.on_locate_load(alt)
assert cam2.v_egg_w.get() == "41" and json.load(open(CAM.LOCATE_PATH))["egg_mm"][1] == "41"
cam2.on_locate_load(out)                               # a calibration file is not a settings file
assert any("不是视觉定位设置文件" in m for m in logs)
print("[H3] 保存设置 -> config + calib folder (with reference image); next start restores method / size / depth / "
      "prompt / reference; 另存为 / 读取")
# ------------------------------------------------------------------ H4
cam2.v_method.set("YOLOE 视觉提示")
cam2.on_connect()
tick(cam2, 3)
assert (MADE[-1].depth_res, MADE[-1].depth_preset) == ("1280×720", "Default")
assert any("深度 1280×720 · Default" in m for m in logs)
cam2.v_ydir.set(MD)
cam2.v_egg_l.set("55")
cam2.v_egg_w.set("42")
for e in (1.0,) * 5:
    cam2._verify_live(object(), e)
rows = eggs(cam2)
assert [r[8] for r in rows] == ["✓", "✓"] and all(r[4] == "-" for r in rows), rows
print("[H4] depth settings handed to the camera on connect; eggs without depth on them located and trusted")
for c in (cam, cam2):
    c.shutdown()
root.destroy()
print("ALL V3.47.1 GUI TESTS PASSED")
