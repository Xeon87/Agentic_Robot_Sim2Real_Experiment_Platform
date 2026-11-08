#!/usr/bin/env python3
"""V3.82 offline tests (fine-tuned model in use + labelling assist + highlights / gamma; no camera, GPU, ultralytics or
Tk here: a fake ultralytics module, fake sensors, mocked Tk):
R1 Versions V3.82 + markers; frozen modules unchanged; the Camera tab (V325) and the real cup perception (V366) are
   NOT changed - a fine-tuned model reaches them through NERO_V347 only.
R2 NERO_V347: list_models adds finetuned/<name>/<file> after the originals; finetune_card; a fine-tuned model counts as
   a fixed-vocabulary (-pf like) model; trained classes + saved synonyms are "covered"; the original model is found.
R3 Detection with a fine-tuned model: never set_classes on it; results filtered by the object names; objects it was not
   trained on go to the original model (text prompts), overlapping boxes dropped; V366 resolve_query -> "pf".
R4 Training at 1280 (batch 4 quick / 2 full), evaluation at the Camera tab's size; synonyms saved in the model card;
   ⑤ page: apply a saved model to the Camera tab (relative name in its model folder) and restore the original.
R5 Exposure: specular spots counted on their own (not clipping), large clipping still caught; adjustable lower
   brightness bound; auto tune raises gamma when gain alone is not enough; spots do not shorten the exposure;
   apply_profile sets gamma only when the profile has it.
R6 Labelling assist (pure): synonyms map detections back to their class, low scores become candidates (not in the
   label file / training data until confirmed), visual-prompt class, re-labelling keeps a person's boxes, editor
   accept / accept all / drop / moving a candidate confirms it.
R7 Labelling assist in the lab window: capture -> 1 label + 1 candidate; double-click confirms; visual example from a
   reviewed image -> source switches, re-label this image / all unreviewed; 「✓ 已复核」 drops unconfirmed candidates;
   prompts per class saved; saved fine-tuned models offered as label source.
R8 Release folder unchanged.
"""
import hashlib
import json
import os
import sys
import tempfile
import time
import types
import unittest.mock as mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v382_"))
MODELS = TMP / "objectmodels"
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
os.environ["NERO_V378_DATA_DIR"] = str(TMP / "datasets")
os.environ["NERO_V379_LAB_SETTINGS"] = str(TMP / "cameracalib" / "camera_lab.json")
os.environ["NERO_V381_FINETUNE_DIR"] = str(MODELS / "finetuned")
os.environ["NERO_YOLO_DIR"] = str(MODELS)
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
import NERO_V366_Real_Cup_Yolo as YC        # noqa: E402
import NERO_V378_D435i_Bench as B           # noqa: E402
import NERO_V379_Camera_Lab as L            # noqa: E402
import NERO_V381_Yolo_Train as T            # noqa: E402

cv2 = YP._cv2()

# ------------------------------------------------------------------ R1
for m in (CON, RR, YP, W49, L, T):
    assert m.VERSION in ("V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6"), (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
for fn, h in (("NERO_V2_7_1_Unified_Backend.py", "5be2136f84ec4ccdfe6f8bc8d8998410ff9b3b827af4d571ea44373dc427527e"),
              ("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
assert H((ROOT / "NERO_V325_Camera.py").read_bytes()) in (ROOT / "tests" / "test_v3_79.py").read_text(encoding="utf-8")
assert "V3.82" not in (ROOT / "NERO_V366_Real_Cup_Yolo.py").read_text(encoding="utf-8")
assert "V3.82 FINE-TUNED MODELS" in (ROOT / "NERO_V347_Yolo_Perception.py").read_text(encoding="utf-8")
assert L.MARKER_V382 in (ROOT / "NERO_V379_Camera_Lab.py").read_text(encoding="utf-8")
assert T.MARKER_V382 in (ROOT / "NERO_V381_Yolo_Train.py").read_text(encoding="utf-8")
print("[R1] V3.82; frozen modules unchanged; Camera tab + V366 unchanged (fine-tuned models enter through V347 only)")

# ------------------------------------------------------------------ R2
MODELS.mkdir(parents=True)
(MODELS / "yoloe-26s-seg.pt").write_bytes(b"base")
(MODELS / "yoloe-26s-seg-pf.pt").write_bytes(b"pf")
(MODELS / "mobileclip2_b.ts").write_bytes(b"enc")
ftd = MODELS / "finetuned" / "eggs_v1"
ftd.mkdir(parents=True)
(ftd / "eggs_v1.pt").write_bytes(b"ft")
card = {"name": "eggs v1", "file": "eggs_v1.pt", "classes": ["egg"], "base_model": "yoloe-26s-seg.pt", "mode": "quick",
        "synonyms": {"egg": ["egg", "white egg", "white round object"]}, "verdict": "better", "created": "x",
        "overall_base": {"recall": 40.0, "ap50": 0.4}, "overall_new": {"recall": 95.0, "ap50": 0.93}}
(ftd / "model_card.json").write_text(json.dumps(card), encoding="utf-8")
(MODELS / "finetuned" / "broken").mkdir()
(MODELS / "finetuned" / "broken" / "model_card.json").write_text('{"file": "missing.pt"}', encoding="utf-8")
lm = YP.list_models(str(MODELS))
assert lm == ["yoloe-26s-seg-pf.pt", "yoloe-26s-seg.pt", "finetuned/eggs_v1/eggs_v1.pt"], lm
ftp = YP.find_model(str(MODELS), lm[-1])
assert ftp == str(ftd / "eggs_v1.pt") and YP.finetune_card(ftp)["classes"] == ["egg"]
assert YP.is_pf_model(ftp) and YP.is_pf_model(str(MODELS / "yoloe-26s-seg-pf.pt")) and not YP.is_pf_model(
    str(MODELS / "yoloe-26s-seg.pt"))
assert YP.finetune_covers(card, "egg") and YP.finetune_covers(card, "white round object")
assert YP.finetune_covers(card, "eggs") and not YP.finetune_covers(card, "cup")
assert YP.finetune_base_path(ftp, card) == str(MODELS / "yoloe-26s-seg.pt")
print("[R2] V347: fine-tuned models listed after the originals, card read, fixed vocabulary, covered names incl. "
      "synonyms, original model found")

# ------------------------------------------------------------------ R3 fake ultralytics
calls = []


class _Arr:
    def __init__(self, a):
        self.a = np.asarray(a, float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, rows):
        self.xyxy = _Arr([r[0] for r in rows])
        self.conf = _Arr([r[1] for r in rows])
        self.cls = _Arr([r[2] for r in rows])
        self.n = len(rows)

    def __len__(self):
        return self.n


class FakeYOLOE:
    def __init__(self, path):
        self.path = path
        self.ft = "finetuned" in path
        self.names = {0: "egg"} if self.ft else {}

    def set_classes(self, names, *a, **k):
        calls.append(("set_classes", os.path.basename(self.path), list(names)))
        self.names = {i: n for i, n in enumerate(names)}

    def predict(self, bgr, **kw):
        calls.append(("predict", os.path.basename(self.path)))
        if self.ft:
            rows = [([100, 100, 160, 150], 0.92, 0)]
        else:
            rows = []
            for i, n in self.names.items():
                if n == "cup":
                    rows.append(([300, 300, 380, 400], 0.7, i))
                    rows.append(([102, 101, 161, 151], 0.6, i))     # overlaps the fine-tuned egg -> dropped
        r = types.SimpleNamespace(boxes=_Boxes(rows) if rows else None, masks=None, names=self.names)
        return [r]


fake_ul = types.ModuleType("ultralytics")
fake_ul.YOLOE = FakeYOLOE
sys.modules["ultralytics"] = fake_ul
det = YP.YoloEDetector(log=lambda m: None)
rgb = np.zeros((720, 1280, 3), np.uint8)
d1 = det.detect(rgb, "text", ftp, prompts=["egg", "cup"], conf=0.25)
assert [(d["name"], d["model"]) for d in d1] in ([("egg", "finetuned"), ("cup", "original")], [("egg", "eggs v1"), ("cup", "original")]), d1
assert not any(c[0] == "set_classes" and c[1] == "eggs_v1.pt" for c in calls)
assert ("set_classes", "yoloe-26s-seg.pt", ["cup"]) in calls and det.info["routed"] == ["cup"]
calls.clear()
d2 = det.detect(rgb, "text", ftp, prompts=["white egg", "white round object"], conf=0.25)
assert [d["name"] for d in d2] == ["egg"] and not any(c[1] == "yoloe-26s-seg.pt" for c in calls)
assert YC.resolve_query(YP, str(MODELS), lm[-1], ["cup"]) == ("pf", ftp)
print("[R3] fine-tuned detection: no set_classes on it, untrained objects routed to the original model, overlaps "
      "dropped, synonyms need no routing; V366 treats it as fixed-vocabulary")
del sys.modules["ultralytics"]

# ------------------------------------------------------------------ R4
assert T.MODES["quick"]["imgsz"] == 1280 and T.MODES["quick"]["batch"] == 4 and T.MODES["full"]["batch"] == 2
src_t = (ROOT / "NERO_V381_Yolo_Train.py").read_text(encoding="utf-8")
assert '"eval_imgsz": int(YP.IMGSZ)' in src_t and 'job.get("eval_imgsz", job["imgsz"])' in src_t
run = TMP / "run1"
(run / "train" / "weights").mkdir(parents=True)
(run / "train" / "weights" / "best.pt").write_bytes(b"w")
ev = {"best": str(run / "train" / "weights" / "best.pt"), "verdict": {"level": "better"},
      "base": {"overall": {"recall": 50.0}}, "new": {"overall": {"recall": 90.0}},
      "job": {"mode": "quick", "classes": ["cup"], "base_model": str(MODELS / "yoloe-26s-seg.pt"), "epochs": 1,
              "batch": 4, "imgsz": 1280, "conf": 0.25, "eval_imgsz": 1280}}
(run / "eval.json").write_text(json.dumps(ev), encoding="utf-8")
dest = T.save_model(str(run), "cups", synonyms={"cup": ["cup", "red cup"], "egg": ["egg"]})
c2 = json.loads((Path(dest) / "model_card.json").read_text(encoding="utf-8"))
assert c2["synonyms"] == {"cup": ["cup", "red cup"]} and "finetuned/cups/cups.pt" in YP.list_models(str(MODELS))
print("[R4a] training 1280 / batch 4 + 2, evaluation at the Camera tab's size, synonyms in the card")

# ------------------------------------------------------------------ R5 exposure
img = np.full((360, 640, 3), 90, np.uint8)
for (x, y) in ((200, 150), (300, 160), (400, 170)):
    img[y:y + 4, x:x + 4] = 255                                          # specular spots
st = L.roi_stats(img, L.DEFAULT_ROI)
assert st["spots"] == 3 and st["sat_large_pct"] == 0.0 and st["spot_pct"] > 0 and st["p995_ex"] < 245
img2 = img.copy()
img2[100:200, 200:300] = 255                                              # large clipped area
st2 = L.roi_stats(img2, L.DEFAULT_ROI)
assert st2["sat_large_pct"] > 0.5 and L.judge_brightness(st2)
assert L.judge_brightness({"mean": 85.0, "p995": 200.0, "sat_pct": 0.0}, mean_lo=80) == []
assert L.judge_brightness({"mean": 85.0, "p995": 200.0, "sat_pct": 0.0})          # default 110 -> dark
rng = np.random.default_rng(0)
RAD = np.full((360, 640), 0.20, np.float32) + rng.normal(0, 0.005, (360, 640)).astype(np.float32)
RAD[150:200, 300:350] = 0.55


class Sim:
    def __init__(self, light, spots=False):
        self.light, self.e, self.g, self.gam, self.spots = light, 156, 64, 300.0, spots

    def set_eg(self, e, g):
        self.e, self.g = e, g

    def set_gamma(self, v):
        self.gam = float(v)

    def frame(self):
        lin = np.clip(RAD * self.light * self.e * (1.0 + self.g / 32.0), 0, 255) / 255.0
        y = (255.0 * lin ** (300.0 / self.gam)).astype(np.uint8)
        f = np.repeat(y[..., None], 3, axis=2)
        if self.spots:
            for (x, yy) in ((310, 160), (330, 170), (320, 185)):
                f[yy:yy + 3, x:x + 3] = 255
        return f

    def measure(self):
        return L.roi_stats(self.frame(), L.DEFAULT_ROI)


s1 = Sim(0.012)
r_no, _t = L.autotune(s1.set_eg, s1.measure, 50, log=lambda m: None)
s2 = Sim(0.012)
r_g, tr = L.autotune(s2.set_eg, s2.measure, 50, log=lambda m: None, set_gamma=s2.set_gamma,
                     gamma_range=(100.0, 500.0, 300.0))
assert r_no["stats"]["mean"] < 110 and r_g["gamma"] > 300 and r_g["stats"]["mean"] > r_no["stats"]["mean"] + 10, (
    r_no["stats"], r_g)
assert any("gamma" in t for t in tr) and s2.gam == r_g["gamma"]
s3 = Sim(0.012)
r_lo, _t = L.autotune(s3.set_eg, s3.measure, 50, log=lambda m: None, mean_lo=60)
assert r_lo["mean_lo"] == 60.0 and all("< 60" in i for i in r_lo["issues"] if "偏暗" in i)
s4, s5 = Sim(1.0), Sim(1.0, spots=True)
ra, _t = L.autotune(s4.set_eg, s4.measure, 50, log=lambda m: None)
rb, _t = L.autotune(s5.set_eg, s5.measure, 50, log=lambda m: None)
assert rb["exposure"] == ra["exposure"] and rb["stats"]["spots"] == 3 and "镜面亮斑" in rb["note"]


class _Rg:
    def __init__(self, lo, hi, st):
        self.min, self.max, self.step, self.default = lo, hi, st, 300


class _Sen:
    def __init__(self, name, opts):
        self.name, self.opts, self.calls = name, opts, []

    def get_info(self, _k):
        return self.name

    def supports(self, o):
        return o in self.opts

    def get_option_range(self, o):
        return self.opts[o][1]

    def set_option(self, o, v):
        self.calls.append(o)
        self.opts[o][0] = v

    def get_option(self, o):
        return self.opts[o][0]


RS = types.SimpleNamespace(camera_info=types.SimpleNamespace(name="n"), option=types.SimpleNamespace(),
                           rs400_visual_preset=types.SimpleNamespace(high_accuracy=3))
for k in ("enable_auto_exposure", "exposure", "gain", "enable_auto_white_balance", "white_balance",
          "power_line_frequency", "gamma", "visual_preset", "laser_power"):
    setattr(RS.option, k, k)
col = _Sen("RGB Camera", {"gamma": [300, _Rg(100, 500, 1)], "exposure": [100, _Rg(1, 10000, 1)]})
dev = types.SimpleNamespace(query_sensors=lambda: [col])
assert L.gamma_range(dev, RS) == (100.0, 500.0, 300)
B.apply_profile(dev, {"color": {"exposure": 200}}, RS, log=lambda m: None)
assert "gamma" not in col.calls
B.apply_profile(dev, {"color": {"exposure": 200, "gamma": 420}}, RS, log=lambda m: None)
assert col.opts["gamma"][0] == 420
print(f"[R5] spots apart from clipping; mean bound adjustable; gamma {r_g['gamma']:.0f} lifts the mean "
      f"{r_no['stats']['mean']:.0f} -> {r_g['stats']['mean']:.0f}; spots keep the exposure; gamma in the profile")

# ------------------------------------------------------------------ R6
mk = np.zeros((720, 1280), bool)
mk[100:150, 100:160] = True
dets = [{"name": "white egg", "conf": 0.8, "box": [100, 100, 160, 150], "mask": mk, "cls": 0},
        {"name": "egg", "conf": 0.15, "box": [400, 400, 450, 440], "mask": None, "cls": 0},
        {"name": "keyboard", "conf": 0.9, "box": [0, 0, 10, 10], "mask": None, "cls": 0}]
syn = {"egg": ["egg", "white egg", "white round object"]}
an = L.dets_to_anns(dets, ["egg"], synonyms=syn, accept_conf=0.30)
assert [(a["cls"], bool(a.get("candidate"))) for a in an] == [("egg", False), ("egg", True)]
vp = L.dets_to_anns([dict(dets[0], cls=0), dict(dets[2], cls=1)], ["cup"], accept_conf=0.3, force_cls="cup")
assert [a["cls"] for a in vp] == ["cup"]
assert len(L.label_lines(an, lambda c: 0, (1280, 720))) == 1
old = [{"cls": "egg", "box": [10, 10, 50, 50], "source": "manual"}, {"cls": "egg", "box": [100, 100, 160, 150],
                                                                     "source": "auto"},
       {"cls": "egg", "box": [600, 600, 650, 650], "source": "accepted"}]
new = [{"cls": "egg", "box": [11, 11, 50, 50], "source": "auto"}, {"cls": "egg", "box": [300, 300, 340, 340],
                                                                   "source": "auto"}]
mr = L.merge_relabel(old, new)
assert [a["source"] for a in mr] == ["manual", "accepted", "auto"] and mr[2]["box"] == [300, 300, 340, 340]
old.append({"cls": "cup", "box": [500, 50, 540, 90], "source": "auto"})
assert any(a["cls"] == "cup" for a in L.merge_relabel(old, new, ["egg"]))          # other classes are kept
anns = [{"cls": "egg", "box": [0, 0, 50, 50], "candidate": True, "source": "auto"},
        {"cls": "egg", "box": [100, 100, 150, 150], "candidate": True, "source": "auto"},
        {"cls": "egg", "box": [200, 200, 250, 250], "candidate": True, "source": "auto"}]
ed = L.BoxEditor(anns, (640, 360))
assert ed.accept(20, 20) == 0 and not anns[0].get("candidate") and anns[0]["source"] == "accepted"
ed.press(120, 120)
ed.drag(130, 130)
assert ed.release(130, 130) == 1 and not anns[1].get("candidate")
assert ed.accept(5, 300) is None and ed.drop_candidates() == 1 and len(anns) == 2
anns.append({"cls": "egg", "box": [300, 300, 340, 340], "candidate": True, "source": "auto"})
assert ed.accept_all() == 1
ds = L.Dataset(str(TMP / "ds_r6"))
st6 = ds.add_image("egg", np.zeros((360, 640, 3), np.uint8), an)
assert ds.class_counts()["egg"] == {"images": 1, "objects": 1, "candidates": 1}
assert len((TMP / "ds_r6" / "egg" / "labels" / "train" / (st6 + ".txt")).read_text().strip().splitlines()) == 1
assert len(T.scan(str(TMP / "ds_r6"))[0]["anns"]) == 1
print("[R6] synonyms -> class, candidates below the confirm threshold kept out of labels and training, visual-prompt "
      "class, re-label keeps a person's boxes, editor accept / move / drop / accept all")

# ------------------------------------------------------------------ R7 lab window


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
mttk.Treeview.return_value.get_children.side_effect = lambda: tuple(sorted(
    p.stem for p in (TMP / "shots" / "egg" / "meta").glob("*.json"))) if (TMP / "shots" / "egg" / "meta").is_dir() else ()
sel = {"v": ()}
mtk.Listbox.return_value.curselection.side_effect = lambda: sel["v"]
rgb_lab = np.full((720, 1280, 3), 70, np.uint8)
seen = []


class _Det:
    device = None

    def detect(self, rgb, mode, path, ref=None, prompts=None, conf=0.25):
        seen.append((mode, os.path.basename(path), list(prompts or []), conf, bool(ref)))
        m1 = np.zeros((720, 1280), bool)
        m1[200:260, 300:380] = True
        if mode == "vp":
            return [{"name": "鸡蛋", "cls": 0, "conf": 0.5, "box": [700, 200, 760, 260], "mask": None}]
        return [{"name": "white egg", "cls": 0, "conf": 0.85, "box": [300, 200, 380, 260], "mask": m1},
                {"name": "egg", "cls": 0, "conf": 0.18, "box": [900, 400, 950, 450], "mask": None}]


class _Src:
    K = {"width": 1280, "height": 720, "fx": 900.0, "fy": 900.0, "ppx": 640.0, "ppy": 360.0}
    dist = [0.0] * 5

    def running(self):
        return True

    def frames(self):
        return rgb_lab, None, 1


logs, uiq = [], []
tab = types.SimpleNamespace(tk=mtk, ttk=mttk, cv=mock.MagicMock(), log=logs.append, src=_Src(), det_yolo=_Det(),
                            v_ydir=FakeVar(value=str(MODELS)), v_ymodel=FakeVar(value="yoloe-26s-seg.pt"),
                            v_yconf=FakeVar(value="0.25"), _ui=uiq.append,
                            _grab_frames=lambda n: (rgb_lab, [np.full((720, 1280), 0.7, np.float32)] * n))
lab = L.open_lab(tab)
lab._ldet = _Det()


def idle():
    t0 = time.time()
    while lab.busy and time.time() - t0 < 30:
        time.sleep(0.02)
    while uiq:
        uiq.pop(0)()


lab.v_root.set(str(TMP / "shots"))
lab._add_class("egg")
assert lab._synonyms_of("egg") == ["egg", "white egg", "white round object"]
assert float(lab.v_cconf.get()) == L.AUTO_CONF and float(lab.v_aconf.get()) == L.ACCEPT_CONF
srcs = lab._label_sources()
assert srcs[:2] == [L.SRC_TEXT, L.SRC_VP] and L.SRC_FT + "eggs v1" in srcs and L.SRC_FT + "cups" in srcs
lab.on_shot()
idle()
assert seen[-1] == ("text", "yoloe-26s-seg.pt", ["egg", "white egg", "white round object"], 0.10, False), seen[-1]
dsl = L.Dataset(str(TMP / "shots"))
m0 = dsl.items("egg")[0]
assert [bool(a.get("candidate")) for a in m0["anns"]] == [False, True]
stem = m0["stem"]
lab._open_item("egg", stem)
s_ = lab._scale()
lab._ev_double(types.SimpleNamespace(x=920 * s_, y=420 * s_))
m1 = dsl.load("egg", stem)[1]
assert not any(a.get("candidate") for a in m1["anns"])
assert len((TMP / "shots" / "egg" / "labels" / "train" / (stem + ".txt")).read_text().strip().splitlines()) == 2
lab.v_curcls.set("egg")
lab.on_vp_ref()
assert lab.v_lsrc.get() == L.SRC_VP and lab.vp_ref["cls_name"] == "egg" and len(lab.vp_ref["boxes"]) == 2
lab.on_relabel_one()
idle()
assert seen[-1][0] == "vp" and seen[-1][4]
m2 = dsl.load("egg", stem)[1]
assert [a["source"] for a in m2["anns"]] == ["accepted", "auto"] and m2["anns"][-1]["box"] == [700.0, 200.0, 760.0, 260.0]
lab.v_lsrc.set(L.SRC_TEXT)
lab.on_shot()
idle()
lab.v_folder.set("egg")
n_before = len(seen)
lab.on_relabel_all()
idle()
assert len(seen) == n_before + 2                                  # both images unreviewed
stem2 = dsl.items("egg")[1]["stem"]
lab._open_item("egg", stem2)
assert any(a.get("candidate") for a in lab.editor.anns)
lab._step(+1, reviewed=True)
m3 = dsl.load("egg", stem2)[1]
assert m3["reviewed"] and not any(a.get("candidate") for a in m3["anns"])
assert any("丢掉" in l_ and "候选框" in l_ for l_ in logs)
sel["v"] = (0,)
lab.v_syn.set("egg, white egg, chicken egg")
lab.on_save_syn()
assert lab._synonyms_of("egg") == ["egg", "white egg", "chicken egg"]
lab.v_lsrc.set(L.SRC_FT + "eggs v1")
cfg = lab._label_cfg(["egg"])
assert cfg["source"] == "finetuned" and cfg["path"] == str(ftd / "eggs_v1.pt")
# ⑤ apply / restore
tp = lab.train_page
tp._refresh_saved(select="eggs v1")
tp.on_apply_model()
assert "finetuned/eggs_v1/eggs_v1.pt" in tab.v_ymodel.get() and "eggs_v1" in tp.v_active.get()
assert str(ftd / "eggs_v1.pt") in YP.find_model(tab.v_ydir.get(), tab.v_ymodel.get())
tp.on_restore_model()
assert tab.v_ymodel.get() == YP.DEFAULT_MODEL
lab.close()
print("[R7] lab: capture 1 label + 1 candidate, double-click confirms, visual example + re-label, re-label all "
      "unreviewed, reviewed drops candidates, prompts saved, fine-tuned label source; ⑤ apply / restore model")

# ------------------------------------------------------------------ R8
SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("[R8] release folder unchanged")
print("ALL V3.82 TESTS PASSED")
