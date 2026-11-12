#!/usr/bin/env python3
"""V3.80 offline tests (camera lab upgrade; no camera / no YOLOE / no Tk here: simulated sensor, fakes, mocked Tk):
P1 Versions V3.80 + markers; frozen modules unchanged; the Camera tab only passes itself to the saved-profile hook
   (lighting check); the lab module keeps its V3.79 functions.
P2 Auto tune on a simulated sensor (brightness = radiance x exposure x gain factor, clipped at 255):
   normal light -> exposure a whole flicker period (50 Hz: multiple of 10 ms, 60 Hz: of 8.3 ms), no clipping in the
   work ROI, mean 110-150, lowest gain that gets there; very bright -> shorter than one period + flicker warning;
   very dark -> gain at the top + "add light".
P3 Work-ROI brightness + judgement; named profiles (save / list / load / activate -> the connect profile keeps its
   name, reference brightness and ROI); lighting check after connect: same light -> OK, 35 % darker -> WARN.
P4 Noise verdict: the user's V3.79 numbers (1131 mm) -> normal (theory ~3.0 mm), best = filters + median 5 (17 %
   better than the Camera tab's spatial+temporal, cost stated), scaled to the 750 mm work distance; noisy -> fail;
   uneven surface -> warning.
P5 Locating any object: general shape -> locate_objects + height error; egg shape -> locate with the entered size;
   truth height = half the object height; verdict text.
P6 Lab window (mocked Tk): no default class; old V3.79 settings migrate (the egg default dropped); class list add /
   tick / remove / current class; capture labels every ticked class into the current class folder; class counts and
   "还差 n 张"; noise page shows the verdict; tables use the Camera tab's Treeview style; a profile needs a name.
P7 Release folder unchanged.
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
TMP = Path(tempfile.mkdtemp(prefix="nero_v380_"))
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
import NERO_V347_Yolo_Perception as YP      # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49   # noqa: E402
import NERO_V378_D435i_Bench as B           # noqa: E402
import NERO_V379_Camera_Lab as L            # noqa: E402

cv2 = YP._cv2()

# ------------------------------------------------------------------ P1
for m in (CON, RR, YP, W49, L):
    assert m.VERSION in ("V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3", "V4.8.4", "V4.8.5"), (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3", "V4.8.4", "V4.8.5")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
for fn, h in (("NERO_V2_7_1_Unified_Backend.py", "5be2136f84ec4ccdfe6f8bc8d8998410ff9b3b827af4d571ea44373dc427527e"),
              ("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
s_cam = (ROOT / "NERO_V325_Camera.py").read_text(encoding="utf-8")
assert "apply_saved_profile(prof.get_device(), rs, self.log, src=self)  # V3.80 light check" in s_cam
s_lab = (ROOT / "NERO_V379_Camera_Lab.py").read_text(encoding="utf-8")
assert L.MARKER in s_lab and L.MARKER_V380 in s_lab
for fn in ("Dataset", "BoxEditor", "recompute_outline", "measure_noise", "open_lab", "autotune", "noise_verdict",
           "save_named", "activate_named", "lighting_check", "locate_fn", "accuracy_verdict"):
    assert hasattr(L, fn), fn
print("[P1] V3.80; frozen modules unchanged; Camera tab passes itself for the lighting check; lab functions present")

# ------------------------------------------------------------------ P2
rng = np.random.default_rng(0)
RAD = np.full((360, 640), 0.35, np.float32) + rng.normal(0, 0.01, (360, 640)).astype(np.float32)   # table
RAD[150:210, 300:360] = 0.60                                                                         # white object
RAD[:, :60] = 3.0                                                                                    # window (outside ROI)


class Sim:
    def __init__(self, light):
        self.light, self.e, self.g, self.calls = light, 156, 64, 0

    def set_eg(self, e, g):
        self.e, self.g = e, g
        self.calls += 1

    def frame(self):
        y = np.clip(RAD * self.light * self.e * (1.0 + self.g / 32.0), 0, 255).astype(np.uint8)
        return np.repeat(y[..., None], 3, axis=2)

    def measure(self):
        return L.roi_stats(self.frame(), L.DEFAULT_ROI)


assert L.flicker_exposures(50) == [100, 200, 300] and L.flicker_exposures(60) == [83, 167, 250]
sim = Sim(light=1.2)
res, trace = L.autotune(sim.set_eg, sim.measure, 50, log=lambda m: None)
assert res["ok"] and res["flicker_safe"] and res["exposure"] % 100 == 0, res
st = res["stats"]
assert L.MEAN_LO <= st["mean"] <= L.MEAN_HI and st["p995"] < L.P995_MAX and (sim.e, sim.g) == (res["exposure"], res["gain"])
lower = [t for t in trace if t["exposure"] == res["exposure"] and t["gain"] < res["gain"] and t["mean"] >= L.MEAN_LO]
assert not lower, "a lower gain already reached the target"
RAD_SAVE = RAD.copy()
RAD[150:210, 300:360] = 0.95                                          # high contrast: white object 2.7x the table
hc = Sim(light=1.2)
rh, _t = L.autotune(hc.set_eg, hc.measure, 50, log=lambda m: None)
assert rh["ok"] and "反差大" in rh["note"] and rh["stats"]["p995"] < L.P995_MAX and rh["stats"]["mean"] < L.MEAN_LO, rh
RAD[:] = RAD_SAVE
s60 = Sim(1.2)
res60, _t = L.autotune(s60.set_eg, s60.measure, 60, log=lambda m: None)
assert res60["exposure"] in (83, 167, 250)
bright = Sim(light=30.0)
rb, _t = L.autotune(bright.set_eg, bright.measure, 50, log=lambda m: None)
assert not rb["flicker_safe"] and rb["exposure"] < 100 and any("闪烁" in i for i in rb["issues"])
assert rb["stats"]["p995"] < L.P995_MAX
dark = Sim(light=0.05)
rd, _t = L.autotune(dark.set_eg, dark.measure, 50, log=lambda m: None)
assert not rd["ok"] and rd["gain"] >= 120 and any("加灯" in i for i in rd["issues"]), rd
print(f"[P2] auto tune: normal light -> {res['exposure_ms']} ms / gain {res['gain']} / mean {st['mean']:.0f} "
      f"({res['steps']} steps); 60 Hz -> {res60['exposure_ms']} ms; very bright -> {rb['exposure_ms']} ms + flicker "
      f"warning; very dark -> gain {rd['gain']} + add light")

# ------------------------------------------------------------------ P3
img = sim.frame()
s1 = L.roi_stats(img, [0.0, 0.0, 1.0, 1.0])
s2 = L.roi_stats(img, L.DEFAULT_ROI)
assert s1["sat_pct"] > 5 and s2["sat_pct"] < 0.5, (s1, s2)          # the window only counts without the ROI
assert L.judge_brightness(s2) == [] and L.judge_brightness({"mean": 90, "p995": 200, "sat_pct": 0})
prof = B.load_profile()
prof["color"].update(exposure=200.0, gain=20.0)
prof["locked"] = True
p_named = L.save_named("办公室白天", dict(prof, ref=dict(s2, roi=L.DEFAULT_ROI), roi=L.DEFAULT_ROI))
assert p_named.startswith(str(TMP / "cameracalib" / "d435i_profiles")) and L.list_named() == ["办公室白天"]
L.activate_named("办公室白天")
assert B.load_profile()["locked"] and B.load_profile()["color"]["exposure"] == 200.0
meta = L.read_profile_meta()
assert meta["name"] == "办公室白天" and abs(meta["ref"]["mean"] - s2["mean"]) < 1e-6


class _Src:
    def __init__(self, img):
        self.img = img

    def frames(self):
        return self.img, None, 30


logs = []
L.lighting_check(_Src(img), meta, logs.append, wait_s=1.0)
assert logs[-1].startswith("[OK]") and "光照检查" in logs[-1]
L.lighting_check(_Src((img * 0.65).astype(np.uint8)), meta, logs.append, wait_s=1.0)
assert logs[-1].startswith("[WARN]") and "重新一键调参" in logs[-1]
print("[P3] work-ROI stats (window excluded); named profile saved / activated with name + reference; lighting check "
      "OK for the same light, WARN when 35 % darker")

# ------------------------------------------------------------------ P4
user_rows = [{"setting": "raw", "valid_pct": 98.2, "temporal_std_mm": 3.694, "flatness_rms_mm": 2.102, "mean_dist_mm": 1131.9},
             {"setting": "spatial+temporal", "valid_pct": 100.0, "temporal_std_mm": 1.52, "flatness_rms_mm": 1.805, "mean_dist_mm": 1135.2},
             {"setting": "spatial+temporal+holes", "valid_pct": 100.0, "temporal_std_mm": 1.474, "flatness_rms_mm": 1.812, "mean_dist_mm": 1135.2},
             {"setting": "spatial+temporal + median 5", "valid_pct": 100.0, "temporal_std_mm": 1.263, "flatness_rms_mm": 1.86, "mean_dist_mm": 1135.0},
             {"setting": "raw + median 10", "valid_pct": 100.0, "temporal_std_mm": 1.374, "flatness_rms_mm": 2.02, "mean_dist_mm": 1131.8}]
v = L.noise_verdict(user_rows)
assert v["level"] == "ok" and v["best"] == "spatial+temporal + median 5" and 2.8 < v["theory_mm"] < 3.3, v
txt = "\n".join(v["lines"])
assert "正常" in txt and "好 17 %" in txt and "0.17 s" in txt and "750 mm" in txt, txt
bad = [dict(r, temporal_std_mm=r["temporal_std_mm"] * 3) for r in user_rows]
assert L.noise_verdict(bad)["level"] == "fail"
uneven = [dict(r, flatness_rms_mm=8.0) for r in user_rows]
vu = L.noise_verdict(uneven)
assert vu["level"] == "warn" and any("不平" in l_ for l_ in vu["lines"])
print("[P4] noise verdict: user's run -> normal (theory %.2f mm), median 5 best (17 %%, 0.17 s), scaled to 750 mm; "
      "noisy -> fail; uneven -> warning" % v["theory_mm"])

# ------------------------------------------------------------------ P5
corners = B.inner_corners()
calls = {}


def fake_lo(d, dm, K, T, table_z=None):
    calls["objects"] = calls.get("objects", 0) + 1
    x, y = corners[3]
    return [{"xyz_m": [x + 0.001, y, 0.031], "status": "object", "height_mm": 63.5}], {}


def fake_lc(d, dm, K, T, table_z=None, size_mm=None):
    calls["egg_size"] = size_mm
    x, y = corners[3]
    return [{"xyz_m": [x, y, 0.021], "status": "egg"}], {}


K = {"width": 1280, "height": 720, "fx": 640.0, "fy": 640.0, "ppx": 640.0, "ppy": 360.0}
grab = lambda n: (np.zeros((4, 4, 3), np.uint8), [np.zeros((4, 4))] * n)
with mock.patch.object(YP, "locate_objects", fake_lo), mock.patch.object(YP, "locate", fake_lc):
    rows, summ = L.accuracy_trials(grab, lambda rgb: [{"x": 1}], K, np.eye(4), corners, 0.062 / 2, 3,
                                   locate=L.locate_fn("general"), height_mm=62.0)
    rows_e, summ_e = L.accuracy_trials(grab, lambda rgb: [{"x": 1}], K, np.eye(4), corners, 0.042 / 2, 2,
                                       locate=L.locate_fn("egg", (60.0, 44.0)))
a = summ["滤波 + 5 帧中值"]
assert calls["objects"] == 6 and calls["egg_size"] == (60.0, 44.0)
assert abs(a["xy"]["mean"] - 1.0) < 1e-6 and abs(a["dz"]["mean"]) < 1e-6 and abs(a["h"]["mean"] - 1.5) < 1e-6
assert summ_e["滤波 1 帧"]["err_3d"]["mean"] == 0.0 and summ_e["滤波 1 帧"]["h"] == {"n": 0}
vt = L.accuracy_verdict(summ)
assert "可以用于抓取" in vt and "检出 100.0 %" in vt
print("[P5] any object: general -> locate_objects + height error; egg -> ellipsoid with the entered size; "
      "truth = half the height; verdict")

# ------------------------------------------------------------------ P6


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
mtk.Listbox.return_value.curselection.return_value = ()
(TMP / "models").mkdir()
(TMP / "models" / "yoloe-26s-seg.pt").write_bytes(b"x")
# a V3.79 settings file: its egg default must not survive, the user's own extra class does
(TMP / "cameracalib").mkdir(exist_ok=True)
(TMP / "cameracalib" / "camera_lab.json").write_text(json.dumps({"root": str(TMP / "old"), "prompts": "egg, cup",
                                                                  "acc_prompt": "egg"}), encoding="utf-8")
st0 = L.load_settings()
assert st0["class_list"] == ["cup"] and st0["acc_prompt"] == "" and "prompts" not in st0
(TMP / "cameracalib" / "camera_lab.json").unlink()
assert L.load_settings()["class_list"] == [] and L.load_settings()["acc_prompt"] == ""

rgb_lab = np.full((720, 1280, 3), 60, np.uint8)
mk_a = np.zeros((720, 1280), bool)
mk_a[300:400, 500:600] = True
mk_b = np.zeros((720, 1280), bool)
mk_b[300:420, 800:880] = True


class _LabSrc:
    K = dict(K)
    dist = [0.0] * 5

    def running(self):
        return True

    def frames(self):
        return rgb_lab, None, 1

    def collect_depth(self, make, n):
        r = np.random.default_rng(n)
        return [np.full((720, 1280), 0.75, np.float32) + r.normal(0, 0.0012, (720, 1280)).astype(np.float32)
                for _ in range(n)]


seen_prompts = []


class _Det:
    def detect(self, rgb, mode, path, prompts=None, conf=0.25):
        seen_prompts.append(list(prompts))
        return [{"name": "mug", "conf": 0.9, "box": [500, 300, 600, 400], "mask": mk_a},
                {"name": "banana", "conf": 0.8, "box": [800, 300, 880, 420], "mask": mk_b}]


labs_log, uiq = [], []
tab = types.SimpleNamespace(tk=mtk, ttk=mttk, cv=mock.MagicMock(), log=labs_log.append, src=_LabSrc(), det_yolo=_Det(),
                            v_ydir=FakeVar(value=str(TMP / "models")), v_ymodel=FakeVar(value="yoloe-26s-seg.pt"),
                            v_yconf=FakeVar(value="0.25"), _ui=uiq.append,
                            _grab_frames=lambda n: (rgb_lab, [np.full((720, 1280), 0.7, np.float32)] * n))
lab = L.open_lab(tab)
lab._ldet = tab.det_yolo      # V3.82: the lab labels with its own detector (intended)
styles = [c_.kwargs.get("style") for c_ in mttk.Treeview.call_args_list]
assert len(styles) >= 3 and all(s_ == "V345Calib.Treeview" for s_ in styles), styles


def wait_idle():
    t0 = time.time()
    while lab.busy and time.time() - t0 < 120:
        time.sleep(0.02)
    while uiq:
        uiq.pop(0)()


assert lab._checked() == [] and lab.v_curcls.get() == "" and lab.v_aprompt.get() == ""
lab.v_root.set(str(TMP / "shots"))
lab.on_shot()
wait_idle()
assert any("先在类别清单" in l_ for l_ in labs_log) and not seen_prompts
for c in ("mug", "banana", "apple"):
    lab._add_class(c)
lab._toggle_class("apple")
assert lab._checked() == ["mug", "banana"] and lab.v_curcls.get() == "mug"
lab.v_curcls.set("banana")
lab.on_shot()
wait_idle()
assert seen_prompts[-1] == ["mug", "banana"]
ds = L.Dataset(str(TMP / "shots"))
assert ds.folders() == ["banana"]
m0 = ds.items("banana")[0]
assert sorted(a_["cls"] for a_ in m0["anns"]) == ["banana", "mug"] and m0["current"] == "banana"
lines = lab._class_lines()
assert lines[0].startswith("☑ mug") and "1 张 · 1 框" in lines[0] and f"还差 {L.RECOMMENDED_IMAGES - 1} 张" in lines[0]
assert lines[2].startswith("☐ apple")
mtk.Listbox.return_value.curselection.return_value = (2,)
lab.on_remove_class()
assert lab.st["class_list"] == ["mug", "banana"]
lab.v_nframes.set("5")
lab.on_noise()
wait_idle()
assert "判定" in lab.v_nverdict.get() and any("[V3.80 noise]" in l_ for l_ in labs_log)
lab.v_pname.set("")
lab.on_save_profile()
assert any("起个名字" in l_ for l_ in labs_log)
lab.v_pname.set("实验室晚上")
lab.on_save_profile()
assert "实验室晚上" in L.list_named() and L.read_profile_meta()["name"] == "实验室晚上"
lab.v_ashape.set(L.SHAPES[1][1])
lab.v_aprompt.set("egg")
lab.board_T = np.eye(4)
lab.on_accuracy()
assert any("高（物体放好后的高度 mm）" in l_ for l_ in labs_log)        # the height is required
lab.v_aH.set("42")
lab.on_accuracy()
assert any("长" in l_ and "V3.80 accuracy" in l_ for l_ in labs_log)       # egg shape needs length / width
lab.close()
saved = json.loads((TMP / "cameracalib" / "camera_lab.json").read_text(encoding="utf-8"))
assert saved["class_list"] == ["mug", "banana"] and saved["current"] == "banana"
print("[P6] lab window: no default class, V3.79 egg default dropped, class list add / tick / remove, current class "
      "folder, every ticked class labelled, counts + 还差; noise verdict; Treeview style; named profile; shape checks")

# ------------------------------------------------------------------ P7
SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("[P7] release folder unchanged")
print("ALL V3.80 TESTS PASSED")
