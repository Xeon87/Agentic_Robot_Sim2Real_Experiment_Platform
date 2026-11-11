#!/usr/bin/env python3
"""V3.79 offline tests (camera lab; no camera / no YOLOE here: fake RealSense objects, fake detections, mocked Tk):
N1 Versions V3.79 + markers; frozen modules unchanged; the Camera tab has the 「相机高级测试」 button right after
   「保存相机信息」, the reader-thread hooks and the saved-profile hook; start_nero.sh checks the new module.
N2 apply_saved_profile: nothing is set while the profile is not locked; when locked the colour options are set
   (auto off first) and the depth preset is left to the Camera tab.
N3 RealSenseSource lab hooks: run_on_device runs inside the reader loop and returns its result (errors re-raised);
   collect_depth returns n frames through the given chain (None = raw) in metres.
N4 Outline maths: mask -> polygon; clip to a box; recompute_outline uses the image's YOLOE outline when the box
   matches it (IoU >= 0.5), GrabCut inside the box otherwise (synthetic object: mask IoU > 0.8), the box as last resort.
N5 Dataset: <root>/<class>/images|labels/{train,val}, meta, depth; classes.json ids shared over folders; every 5th
   image -> val; label lines = class id + normalised polygon; save_anns rewrites label + reviewed; delete removes
   every file of the image; stats.
N6 dets_to_anns: prompt names kept, YOLOE -pf names mapped onto the matching prompt, others dropped.
N7 BoxEditor: click selects, drag inside moves (clamped to the image), drag a corner resizes, drag on empty adds a
   box with the default class, a tiny box / a click without moving changes nothing, delete, set_class.
N8 measure_noise through a fake source (5 filter sets, warm-up frames dropped); accuracy_trials summary with a fake
   locate (detected %, errors against the corner).
N9 Lab window with mocked Tk: opens once per tab; capture writes an auto-labelled image into the chosen folder;
   review: open, new box -> outline recomputed + saved, delete box, 「已复核」; profile 「保存并锁定」 -> NERO_CALIB_DIR.
N10 Release folder unchanged; nothing written outside the temp folders.
"""
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest.mock as mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v379_"))
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
os.environ["NERO_V378_DATA_DIR"] = str(TMP / "datasets")
os.environ["NERO_V379_LAB_SETTINGS"] = str(TMP / "cameracalib" / "camera_lab.json")
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
H = lambda b: hashlib.sha256(b).hexdigest()


def snapshot():
    out = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rel = line.split(None, 1)[1].strip()
            out[rel] = H((ROOT / rel).read_bytes()) if (ROOT / rel).is_file() else None
    return out


SNAP0 = snapshot()
import numpy as np                          # noqa: E402
import NERO_V312_Console as CON             # noqa: E402
import NERO_V313_Real_Robot as RR           # noqa: E402
import NERO_V325_Camera as CAM              # noqa: E402
import NERO_V347_Yolo_Perception as YP      # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49   # noqa: E402
import NERO_V378_D435i_Bench as B           # noqa: E402
import NERO_V379_Camera_Lab as L            # noqa: E402

cv2 = YP._cv2()

# ------------------------------------------------------------------ N1
for m in (CON, RR, YP, W49, L):
    assert m.VERSION in ("V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2"), (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
for fn, h in (("NERO_V2_7_1_Unified_Backend.py", "5be2136f84ec4ccdfe6f8bc8d8998410ff9b3b827af4d571ea44373dc427527e"),
              ("NERO_V325_Camera.py", "82f2416d73671e7018d09d0cd72c17186154934a315b47c0d19fdb0427afd931"),
              ("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
s_cam = (ROOT / "NERO_V325_Camera.py").read_text(encoding="utf-8")
i_save = s_cam.index('text="保存相机信息"')
i_lab = s_cam.index('text="相机高级测试"')
assert 0 < i_lab - i_save < 200, "the lab button sits right after 「保存相机信息」"
for mk in ("V3.79 CAMERA LAB HOOKS", "V3.79 SAVED D435I PROFILE", "V3.79 CAMERA LAB", "def on_camera_lab"):
    assert mk in s_cam, mk
assert L.MARKER in (ROOT / "NERO_V379_Camera_Lab.py").read_text(encoding="utf-8")
s_sh = (ROOT / "start_nero.sh").read_text(encoding="utf-8")
assert "NERO_V379_Camera_Lab.py" in s_sh and s_sh.count("$LAB379") >= 2
print("[N1] V3.79; frozen modules unchanged; 「相机高级测试」 after 「保存相机信息」; hooks in V325; launcher checks it")

# ------------------------------------------------------------------ N2


class _Range:
    def __init__(self, lo, hi, step):
        self.min, self.max, self.step = lo, hi, step


class _Sensor:
    def __init__(self, name, opts):
        self.name, self.opts, self.calls = name, opts, []

    def get_info(self, _k):
        return self.name

    def supports(self, opt):
        return opt in self.opts

    def get_option_range(self, opt):
        return self.opts[opt][1]

    def set_option(self, opt, v):
        self.calls.append((opt, v))
        self.opts[opt][0] = v

    def get_option(self, opt):
        return self.opts[opt][0]


class _Dev:
    def __init__(self, sensors):
        self.s = sensors

    def query_sensors(self):
        return self.s


class _RS:
    class camera_info:
        name = "name"

    option = types.SimpleNamespace()
    rs400_visual_preset = type("P", (), {"high_accuracy": 3})


for k in ("enable_auto_exposure", "exposure", "gain", "enable_auto_white_balance", "white_balance",
          "power_line_frequency", "laser_power", "visual_preset"):
    setattr(_RS.option, k, k)


def fake_dev():
    col = _Sensor("RGB Camera", {"enable_auto_exposure": [1, _Range(0, 1, 1)], "exposure": [100, _Range(1, 10000, 1)],
                                 "gain": [64, _Range(0, 128, 1)], "enable_auto_white_balance": [1, _Range(0, 1, 1)],
                                 "white_balance": [4000, _Range(2800, 6500, 10)],
                                 "power_line_frequency": [3, _Range(0, 3, 1)]})
    dep = _Sensor("Stereo Module", {"visual_preset": [0, _Range(0, 5, 1)], "enable_auto_exposure": [1, _Range(0, 1, 1)]})
    return _Dev([col, dep]), col, dep


dev, col, dep = fake_dev()
logs = []
assert L.apply_saved_profile(dev, _RS, logs.append) == "none" and not col.calls and not dep.calls
p = B.load_profile()
p["color"].update(exposure=180.0, power_line_frequency=1)
p["locked"] = True
B.save_profile(p)
assert L.apply_saved_profile(dev, _RS, logs.append) == "locked"
names = [c[0] for c in col.calls]
assert names.index("enable_auto_exposure") < names.index("exposure")
assert col.opts["exposure"][0] == 180 and col.opts["power_line_frequency"][0] == 1
assert not any(c[0] == "visual_preset" for c in dep.calls) and dep.opts["visual_preset"][0] == 0
assert "已套用锁定的相机参数" in logs[-1] and "50 Hz" in logs[-1]
print("[N2] saved profile: not locked -> nothing set; locked -> colour options set (auto off first), depth preset "
      "left to the Camera tab")

# ------------------------------------------------------------------ N3


class _Frame:
    def __init__(self, a):
        self.a = a

    def get_data(self):
        return self.a


src = CAM.RealSenseSource(log=lambda s: None)
stop = threading.Event()
served = []


def loop():
    k = 0
    while not stop.is_set():
        k += 1
        if src._dev_cmds or src._depth_jobs:
            src._serve_lab("DEV", "RS", _Frame(np.full((4, 6), 1000 + k, np.uint16)), 0.001)
            served.append(k)
        time.sleep(0.005)


src.thread = threading.Thread(target=loop, daemon=True)
src.thread.start()
assert src.run_on_device(lambda d, r: (d, r)) == ("DEV", "RS")
try:
    src.run_on_device(lambda d, r: 1 / 0)
    raise AssertionError("error must be re-raised")
except ZeroDivisionError:
    pass
raw = src.collect_depth(None, 3)
assert len(raw) == 3 and raw[0].dtype == np.float32 and 1.0 < float(raw[0][0, 0]) < 1.2
half = src.collect_depth(lambda rs: (lambda f: _Frame(f.get_data() // 2)), 2)
assert len(half) == 2 and 0.5 < float(half[0][0, 0]) < 0.6 and not src._depth_jobs
stop.set()
src.thread.join(1.0)
try:
    src.run_on_device(lambda d, r: 0)
    raise AssertionError("not running must fail")
except RuntimeError:
    pass
print("[N3] reader-thread hooks: run_on_device result / error, collect_depth raw + chained frames in metres")

# ------------------------------------------------------------------ N4
hw = (360, 640)
m = np.zeros(hw, np.uint8)
cv2.circle(m, (320, 180), 60, 1, -1)
poly = L.mask_to_poly(m.astype(bool))
assert poly and abs(L.poly_area(poly) - np.pi * 60 ** 2) / (np.pi * 60 ** 2) < 0.05
cp = L.clip_poly(poly, [320, 0, 640, 360], hw)
assert cp and abs(L.poly_area(cp) - 0.5 * np.pi * 60 ** 2) / (0.5 * np.pi * 60 ** 2) < 0.08
rgb = np.full((360, 640, 3), 40, np.uint8)
rgb[:, :, 1] = 90
truth = np.zeros(hw, np.uint8)
cv2.ellipse(truth, (420, 200), (55, 40), 20, 0, 360, 1, -1)
rgb[truth > 0] = (235, 225, 200)
rng = np.random.default_rng(0)
rgb = np.clip(rgb.astype(int) + rng.integers(-8, 9, rgb.shape), 0, 255).astype(np.uint8)
orig = [{"box": [260, 120, 380, 240], "poly": poly}]
p1, how1 = L.recompute_outline(rgb, [262, 118, 384, 236], orig)
assert how1 == "yoloe" and abs(L.poly_area(p1) - L.poly_area(poly)) / L.poly_area(poly) < 0.1
p2, how2 = L.recompute_outline(rgb, [350, 145, 490, 255], orig)
assert how2 == "grabcut", how2
mk = L.poly_mask(p2, hw)
iou = (mk & (truth > 0)).sum() / float((mk | (truth > 0)).sum())
assert iou > 0.8, iou
p3, how3 = L.recompute_outline(rgb, [10, 10, 13, 13], [])
assert how3 == "box" and p3 == L.box_poly([10.0, 10.0, 13.0, 13.0])
print(f"[N4] outlines: circle polygon area within 5 %, clip to half, matching box -> YOLOE outline, new box -> "
      f"GrabCut (IoU {iou:.2f}), tiny box -> box")

# ------------------------------------------------------------------ N5
root = TMP / "mydata"
ds = L.Dataset(str(root))
img = np.zeros((720, 1280, 3), np.uint8)
ann = {"cls": "egg", "box": [100, 100, 200, 180], "poly": [[100, 100], [200, 100], [200, 180], [100, 180]],
       "conf": 0.8, "source": "auto"}
stems = [ds.add_image("egg", img, [dict(ann)], depth=np.full((720, 1280), 0.7, np.float32)) for _ in range(5)]
cup = ds.add_image("cup", img, [dict(ann, cls="cup"), dict(ann, cls="egg")])
assert ds.classes() == ["egg", "cup"]
it = ds.items("egg")
assert [m_["split"] for m_ in it] == ["train"] * 4 + ["val"]
assert (root / "egg" / "images" / "val" / (stems[4] + ".jpg")).is_file()
assert (root / "egg" / "depth" / (stems[0] + ".png")).is_file()
d16 = cv2.imread(str(root / "egg" / "depth" / (stems[0] + ".png")), cv2.IMREAD_UNCHANGED)
assert d16.dtype == np.uint16 and int(d16[0, 0]) == 700
lab = (root / "egg" / "labels" / "train" / (stems[0] + ".txt")).read_text().split()
assert lab[0] == "0" and abs(float(lab[1]) - 100 / 1280) < 1e-6 and abs(float(lab[2]) - 100 / 720) < 1e-6
lc = (root / "cup" / "labels" / "train" / (cup + ".txt")).read_text().splitlines()
assert [l_.split()[0] for l_ in lc] == ["1", "0"]
ds.save_anns("egg", stems[0], [], reviewed=True)
assert (root / "egg" / "labels" / "train" / (stems[0] + ".txt")).read_text() == ""
assert ds.load("egg", stems[0])[1]["reviewed"] and ds.load("egg", stems[0])[1]["auto"]
ds.delete_image("egg", stems[1])
assert not any(stems[1] in str(q) for q in root.rglob("*"))
st = ds.stats()
assert st["egg"] == {"images": 4, "train": 3, "val": 1, "reviewed": 1, "objects": 3} and st["cup"]["objects"] == 2
assert L.safe_class("red cup/2") == "red_cup_2"
print("[N5] dataset: class folders, shared class ids, 1 in 5 -> val, normalised polygon labels, depth mm, "
      "review rewrites the label, delete removes all files, stats")

# ------------------------------------------------------------------ N6
mk1 = np.zeros((100, 100), bool)
mk1[20:40, 20:50] = True
dets = [{"name": "egg", "conf": 0.9, "box": [20, 20, 50, 40], "mask": mk1},
        {"name": "chicken egg", "conf": 0.6, "box": [20, 20, 50, 40], "mask": mk1},
        {"name": "keyboard", "conf": 0.7, "box": [0, 0, 10, 10], "mask": None}]
an = L.dets_to_anns(dets, ["egg", "cup"])
assert [a["cls"] for a in an] == ["egg", "egg"] and an[0]["poly"] and an[0]["source"] == "auto"
print("[N6] detections -> annotations: prompt names kept, -pf names mapped, other classes dropped")

# ------------------------------------------------------------------ N7
anns = [{"cls": "egg", "box": [100, 100, 200, 200], "poly": None}, {"cls": "egg", "box": [400, 300, 450, 340],
                                                                     "poly": None}]
ed = L.BoxEditor(anns, (640, 360), default_cls="cup")
assert ed.press(150, 150) == 0 and ed.release(150, 150) is None and anns[0]["box"] == [100, 100, 200, 200]
ed.press(150, 150)
ed.drag(600, 150)
assert ed.release(600, 150) == 0 and anns[0]["box"] == [540.0, 100.0, 640.0, 200.0] and anns[0]["source"] == "edited"
ed.press(640, 200)                                   # bottom-right corner of the selected box
assert ed.release(620, 260) == 0 and anns[0]["box"] == [540.0, 100.0, 620.0, 260.0]
ed.press(10, 10)
assert ed.sel is None and ed.release(60, 50) == 2 and anns[2]["cls"] == "cup" and anns[2]["source"] == "manual"
ed.press(300, 20)
assert ed.release(302, 22) is None and len(anns) == 3
ed.sel = 1
assert ed.set_class("cup") and anns[1]["cls"] == "cup" and ed.delete() and len(anns) == 2 and ed.sel is None
print("[N7] box editor: select, move (clamped), resize at a corner, new box with the default class, tiny box / "
      "click ignored, set class, delete")

# ------------------------------------------------------------------ N8


class _NoiseSrc:
    def __init__(self):
        self.calls = []

    def collect_depth(self, make, n):
        self.calls.append((make is not None, n))
        r = np.random.default_rng(len(self.calls))
        return [np.full((720, 1280), 0.6, np.float32) + r.normal(0, 0.0015, (720, 1280)).astype(np.float32)
                for _ in range(n)]


ns = _NoiseSrc()
K = {"width": 1280, "height": 720, "fx": 640.0, "fy": 640.0, "ppx": 640.0, "ppy": 360.0}
rows, roi = L.measure_noise(ns, K, frames=6, log=lambda s: None)
assert len(rows) == 5 and [c[0] for c in ns.calls] == [False, True, True, True, False]
assert ns.calls[3][1] == 10 + 6 * 5 and rows[3]["frames"] == 6 and rows[0]["temporal_std_mm"] > rows[4]["temporal_std_mm"]
corners = B.inner_corners()
T = np.eye(4)
seq = iter([0.0, 0.002] * 20)


def fake_locate(dets_, d_m, K_, T_, table_z=0.0):
    if not dets_:
        return [], {}
    x, y = corners[7]
    return [{"xyz_m": [x + next(seq), y, 0.022], "status": "egg"}], {}


with mock.patch.object(YP, "locate", fake_locate):
    k_det = iter(range(100))
    rows2, summ = L.accuracy_trials(lambda n: (np.zeros((4, 4, 3), np.uint8), [np.zeros((4, 4))] * n),
                                    lambda rgb: [] if next(k_det) == 0 else [{"x": 1}], K, T, corners,
                                    B.EGG_HALF_WIDTH_M, trials=4)
a = summ["滤波 + 5 帧中值"]
assert a["detected_pct"] == 75.0 and summ["滤波 1 帧"]["detected_pct"] == 100.0
assert all(r_["corner"] == 7 for r_ in rows2 if r_.get("detected")) and abs(a["dz"]["mean"] - 1.0) < 1e-6
print("[N8] noise through the reader thread (5 sets, warm-up dropped, median lowers the jitter); accuracy summary")

# ------------------------------------------------------------------ N9


class FakeVar:
    def __init__(self, master=None, value=None, **_kw):
        self._v = value

    def get(self):
        return self._v

    def set(self, v):
        self._v = v


mtk, mttk = mock.MagicMock(), mock.MagicMock()
for nm in ("StringVar", "BooleanVar", "IntVar", "DoubleVar"):
    setattr(mtk, nm, FakeVar)
mttk.Treeview.return_value.get_children.return_value = ()
(TMP / "models").mkdir()
(TMP / "models" / "yoloe-26s-seg.pt").write_bytes(b"x")


class _LabSrc:
    K = dict(K)
    dist = [0.0] * 5

    def running(self):
        return True

    def frames(self):
        return rgb_lab, None, 1


rgb_lab = np.full((720, 1280, 3), 50, np.uint8)
cv2.ellipse(rgb_lab, (640, 360), (60, 45), 0, 0, 360, (240, 230, 210), -1)
mk_lab = np.zeros((720, 1280), bool)
mk_lab[315:406, 580:701] = True


class _Det:
    def detect(self, rgb, mode, path, prompts=None, conf=0.25):
        assert path == str(TMP / "models" / "yoloe-26s-seg.pt") and mode == "text"
        return [{"name": prompts[0], "conf": 0.88, "box": [580, 315, 700, 405], "mask": mk_lab}]


labs_log = []
uiq = []
tab = types.SimpleNamespace(tk=mtk, ttk=mttk, cv=mock.MagicMock(), log=labs_log.append, src=_LabSrc(), det_yolo=_Det(),
                            v_ydir=FakeVar(value=str(TMP / "models")), v_ymodel=FakeVar(value="yoloe-26s-seg.pt"),
                            v_yconf=FakeVar(value="0.25"), _ui=uiq.append,
                            _grab_frames=lambda n: (rgb_lab, [np.full((720, 1280), 0.7, np.float32)] * n))
lab = L.open_lab(tab)
lab._ldet = tab.det_yolo      # V3.82: the lab labels with its own detector (intended)
assert L.open_lab(tab) is lab
texts = [c_.kwargs.get("text") for c_ in mttk.Notebook.return_value.add.call_args_list]
assert texts[:4] == ["① 相机参数", "② 深度噪声", "③ 定位精度", "④ 采集标注"], texts   # V3.81 adds ⑤
lab.v_root.set(str(TMP / "shots"))
lab._add_class("egg")          # V3.80: the class list replaced the V3.79 prompt entry (intended)
lab._add_class("cup")
lab.v_curcls.set("egg")


def wait_idle():
    t0 = time.time()
    while lab.busy and time.time() - t0 < 10:
        time.sleep(0.02)
    while uiq:
        uiq.pop(0)()


lab.on_shot()
wait_idle()
ds2 = L.Dataset(str(TMP / "shots"))
it2 = ds2.items("egg")
assert len(it2) == 1 and it2[0]["anns"][0]["cls"] == "egg" and it2[0]["model"] == "yoloe-26s-seg.pt"
assert it2[0]["depth"] and not any("相机参数还没锁定" in l_ for l_ in labs_log)
stem = it2[0]["stem"]
lab._open_item("egg", stem)
s_ = lab._scale()
ev = lambda x, y: types.SimpleNamespace(x=x * s_, y=y * s_)
lab.v_bcls.set("cup")
lab._ev_press(ev(100, 100))
lab._ev_drag(ev(300, 250))
lab._ev_release(ev(300, 250))
m2 = ds2.load("egg", stem)[1]
assert len(m2["anns"]) == 2 and m2["anns"][1]["cls"] == "cup" and m2["anns"][1]["outline"] in ("grabcut", "box")
lab.editor.sel = 1
lab.on_del_box()
lab._step(+1, reviewed=True)
m3 = ds2.load("egg", stem)[1]
assert len(m3["anns"]) == 1 and m3["reviewed"] and ds2.classes() == ["egg", "cup"]
lab.v_exp.set("190")
lab.v_pname.set("office")       # V3.80: profiles are saved under a name (intended)
lab.on_save_profile()
assert B.load_profile()["locked"] and B.load_profile()["color"]["exposure"] == 190.0
lab.close()
assert json.loads((TMP / "cameracalib" / "camera_lab.json").read_text(encoding="utf-8"))["root"] == str(TMP / "shots")
assert L.open_lab(tab) is not lab
print("[N9] lab window: 4 pages, one per tab; capture -> auto-labelled image in the chosen folder; review new box "
      "(outline recomputed) / delete / reviewed; profile saved + locked; settings in NERO_CALIB_DIR")

# ------------------------------------------------------------------ N10
SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("[N10] release folder unchanged")
print("ALL V3.79 TESTS PASSED")
