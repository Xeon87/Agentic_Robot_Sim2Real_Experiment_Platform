#!/usr/bin/env python3
"""V3.81 offline tests (YOLOE training page; no camera / GPU / ultralytics / Tk here: fake YOLOE in a REAL child
process, mocked Tk):
Q1 Versions V3.81 + markers; frozen modules unchanged; the Camera tab (V325) is NOT changed by V3.81; the lab has
   the ⑤ page; start_nero.sh checks the new module.
Q2 Camera profile files: save-as to any folder / file name, open again (named profiles and V3.78 files too), a
   wrong file is refused; in the lab window: open -> fields + connect profile when ticked.
Q3 Training data: class table over the V3.79 dataset; plan_dataset (only reviewed by default, unreviewed counted,
   labels of other classes dropped + warned, errors without val / train); export_dataset: hard-linked images,
   labels remapped to the chosen classes, val kept as val, data.yaml + manifest.
Q4 Linear probing freeze list: everything before the head + every head child except the last layer of each *cv3
   branch (cv3 and one2one_cv3).
Q5 Evaluation maths: recall / precision / false detections per image at the working confidence, AP50 / AP50-95,
   outline IoU; verdict better / same / worse / unreliable.
Q6 The child process (TrainJob -> `NERO_V381_Yolo_Train.py job`): progress events per epoch, quick mode passes the
   freeze list + YOLOEPESegTrainer, full mode no freeze; evaluation of original vs new on the val images -> "better";
   STOP ends training early and still evaluates.
Q7 save_model -> NERO_V381_FINETUNE_DIR/<name>/ (weights + card + eval), duplicate name refused, list_saved;
   discard only inside the runs folder.
Q8 ⑤ page with mocked Tk: buttons follow the steps (train only after a good check, save only after the evaluation);
   check -> train (real child, fake YOLOE) -> table + verdict -> save; a training left running is picked up again.
Q9 Release folder unchanged.
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
TMP = Path(tempfile.mkdtemp(prefix="nero_v381_"))
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
os.environ["NERO_V378_DATA_DIR"] = str(TMP / "datasets")
os.environ["NERO_V379_LAB_SETTINGS"] = str(TMP / "cameracalib" / "camera_lab.json")
os.environ["NERO_V381_FINETUNE_DIR"] = str(TMP / "finetuned")
os.environ["NERO_V381_FAKE_ULTRALYTICS"] = str(ROOT / "tests" / "v381_fake_yolo.py")
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
import NERO_V381_Yolo_Train as T            # noqa: E402
import v381_fake_yolo as FY                 # noqa: E402

cv2 = YP._cv2()

# ------------------------------------------------------------------ Q1
for m in (CON, RR, YP, W49, L, T):
    assert m.VERSION in ("V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8"), (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
v325_hash = [l_ for l_ in (ROOT / "tests" / "test_v3_80.py").read_text(encoding="utf-8").splitlines()
             if "NERO_V325_Camera.py" in l_ and '", "' in l_]
for fn, h in (("NERO_V2_7_1_Unified_Backend.py", "5be2136f84ec4ccdfe6f8bc8d8998410ff9b3b827af4d571ea44373dc427527e"),
              ("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
h325 = H((ROOT / "NERO_V325_Camera.py").read_bytes())
assert h325 in (ROOT / "tests" / "test_v3_79.py").read_text(encoding="utf-8"), "V325 changed after V3.80"
assert T.MARKER in (ROOT / "NERO_V381_Yolo_Train.py").read_text(encoding="utf-8")
assert L.MARKER_V381 in (ROOT / "NERO_V379_Camera_Lab.py").read_text(encoding="utf-8")
s_sh = (ROOT / "start_nero.sh").read_text(encoding="utf-8")
assert "NERO_V381_Yolo_Train.py" in s_sh and s_sh.count("$TRAIN381") >= 2
print("[Q1] V3.81; frozen modules unchanged; Camera tab unchanged since V3.80; markers; launcher checks the module")

# ------------------------------------------------------------------ Q2
prof = B.load_profile()
prof["color"].update(exposure=200.0, gain=12.0, power_line_frequency=1)
p1 = L.save_profile_file(str(TMP / "my profiles" / "实验室A.json"), dict(prof, locked=True))
d1 = L.load_profile_file(p1)
assert d1["name"] == "实验室A" and d1["color"]["exposure"] == 200.0 and "50 Hz" in L.profile_summary(d1)
B.save_profile(dict(prof, locked=True))
assert L.load_profile_file(B.PROFILE_PATH)["color"]["gain"] == 12.0          # the connect profile opens too
(TMP / "bad.json").write_text('{"x": 1}', encoding="utf-8")
try:
    L.load_profile_file(str(TMP / "bad.json"))
    raise AssertionError("a wrong file must be refused")
except ValueError:
    pass
print("[Q2a] profile files: any folder / name, round trip, connect profile readable, wrong file refused")

# ------------------------------------------------------------------ Q3 dataset (red / blue blocks)
root = TMP / "data"
ds = L.Dataset(str(root))
rng = np.random.default_rng(1)


def scene(n_red, n_blue):
    img = np.full((360, 640, 3), 120, np.uint8)
    anns = []
    for nm, col, n, x_lo in (("red_block", (230, 20, 20), n_red, 40), ("blue_block", (20, 20, 230), n_blue, 340)):
        for _ in range(n):
            x, y = int(rng.integers(x_lo, x_lo + 200)), int(rng.integers(40, 260))
            w, h = int(rng.integers(30, 60)), int(rng.integers(30, 60))
            img[y:y + h, x:x + w] = col
            b = [float(x), float(y), float(x + w), float(y + h)]
            anns.append({"cls": nm, "box": b, "poly": [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]],
                         "conf": None, "source": "manual"})
    return img, anns


stems = []
for k in range(25):
    img, anns = scene(1, 1)
    if k == 3:
        anns.append({"cls": "cup", "box": [0, 0, 20, 20], "poly": None, "conf": None, "source": "manual"})
    stems.append(ds.add_image("blue_block", img, anns))
for s_ in stems[:-3]:
    ds.save_anns("blue_block", s_, ds.load("blue_block", s_)[1]["anns"], reviewed=True)
items = T.scan(str(root))
tab_ = T.class_table(items)
assert tab_["blue_block"]["images"] == 25 and tab_["blue_block"]["reviewed"] == 22 and tab_["cup"]["objects"] == 1
plan = T.plan_dataset(items, ["red_block", "blue_block"])
assert plan["ok"] and plan["skipped_unreviewed"] == 3 and plan["dropped"] == 1, plan["errors"]
assert plan["per_class"]["blue_block"]["val_images"] >= 4 and any("没复核" in w for w in plan["warnings"])
assert any("没勾选的类别" in w for w in plan["warnings"])
plan_all = T.plan_dataset(items, ["red_block", "blue_block"], reviewed_only=False)
assert len(plan_all["items"]) == 25 and plan_all["skipped_unreviewed"] == 0
bad = T.plan_dataset(items, ["green_block"])
assert not bad["ok"] and any("训练集没有图" in e for e in bad["errors"]) and any("验证集没有框" in e for e in bad["errors"])
assert not T.plan_dataset(items, [])["ok"]
out = TMP / "export"
yml = T.export_dataset(plan, str(out))
y = Path(yml).read_text(encoding="utf-8")
assert 'names:\n  0: "red_block"\n  1: "blue_block"' in y and "val: images/val" in y
val_imgs = sorted((out / "images" / "val").iterdir())
assert val_imgs and len(list((out / "images" / "train").iterdir())) + len(val_imgs) == 22
any_img = val_imgs[0]
assert os.stat(any_img).st_nlink >= 2                         # hard link, no copy
labs = (out / "labels" / "val" / (any_img.stem + ".txt")).read_text().split("\n")
assert sorted(l_.split()[0] for l_ in labs if l_) == ["0", "1"]
man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
assert man["classes"] == ["red_block", "blue_block"] and all(
    a["cls"] != "cup" for it in man["items"] for a in it["anns"])
print("[Q3] data: reviewed-only by default (3 skipped), other classes dropped + warned, errors without data; export "
      "hard-links images, remaps labels, keeps val, writes data.yaml + manifest")

# ------------------------------------------------------------------ Q4
fz, br = T.linear_probe_freeze(FY._Net().model)
assert br == ["cv3", "one2one_cv3"]
assert [str(i) for i in range(6)] == fz[:6] and "6.cv2" in fz and "6.dfl" in fz
assert "6.cv3.0.0" in fz and "6.cv3.0.1" in fz and "6.cv3.0.2" not in fz and "6.one2one_cv3.2.1" in fz
assert "6.one2one_cv3.2.2" not in fz and "6.cv3" not in fz
print("[Q4] linear probing: backbone + head frozen except the last layer of cv3 / one2one_cv3")

# ------------------------------------------------------------------ Q5
g = lambda c, b: {"cls": c, "box": b, "poly": None}
pr = lambda c, s, b: {"cls": c, "conf": s, "box": b, "poly": None}
samples = [{"gts": [dict(g("a", [0, 0, 10, 10]), poly=[[0, 0], [10, 0], [10, 10], [0, 10]]), g("b", [20, 20, 30, 30])],
            "preds": [pr("a", 0.9, [0, 0, 10, 10]), pr("b", 0.2, [20, 20, 30, 30]), pr("a", 0.6, [50, 50, 60, 60])],
            "hw": (100, 100), "ms": 10.0},
           {"gts": [g("a", [0, 0, 10, 10])], "preds": [pr("a", 0.8, [1, 1, 10, 10])], "hw": (100, 100), "ms": 12.0}]
ev = T.evaluate(samples, ["a", "b"], 0.25)
a, b_ = ev["per_class"]["a"], ev["per_class"]["b"]
assert a["recall"] == 100.0 and a["fp"] == 1 and a["precision"] == 66.7 and a["fp_per_img"] == 0.5
assert b_["recall"] == 0.0 and b_["ap50"] == 1.0                   # found only below the working confidence
assert a["ap50"] == 1.0 and a["ap50_95"] < 1.0 and a["mask_iou"] > 0.7
assert ev["overall"]["recall"] == 50.0 and ev["overall"]["ms_per_img"] == 11.0
mk = lambda r, ap, fp, n=20: {"per_class": {"a": {"recall": r, "n_gt": n}}, "overall": {
    "recall": r, "ap50": ap, "fp_per_img": fp, "ms_per_img": 9.0, "conf": 0.25}}
assert T.compare(mk(60, 0.6, 0.1), mk(90, 0.9, 0.05))["level"] == "better"
assert T.compare(mk(80, 0.8, 0.1), mk(81, 0.81, 0.1))["level"] == "same"
assert T.compare(mk(80, 0.8, 0.1), mk(60, 0.6, 0.1))["level"] == "worse"
c_ = T.compare(mk(60, 0.6, 0.1, 3), mk(90, 0.9, 0.05, 3))
assert c_["unreliable"] and any("验证集太小" in l_ for l_ in c_["lines"])
print("[Q5] evaluation: recall / precision / false detections at the working confidence, AP50 / AP50-95, outline "
      "IoU; verdicts better / same / worse / unreliable")

# ------------------------------------------------------------------ Q6 child process
models = TMP / "models"
models.mkdir()
(models / "yoloe-26s-seg.pt").write_bytes(b"x")


def job_for(run, mode, epochs):
    data = T.export_dataset(plan, os.path.join(run, "dataset"))
    return {"run_dir": run, "data": data, "manifest": os.path.join(run, "dataset", "manifest.json"),
            "base_model": str(models / "yoloe-26s-seg.pt"), "mode": mode, "epochs": epochs, "batch": 4, "imgsz": 640,
            "lr0": 1e-3, "classes": ["red_block", "blue_block"], "conf": 0.25, "patience": 15, "workers": 0,
            "device": "cpu"}


def wait(job, limit=120):
    t0 = time.time()
    while job.running() and time.time() - t0 < limit:
        job.poll()
        time.sleep(0.05)
    job.poll()
    return [e["stage"] for e in job.events]


runs = TMP / "data" / "_runs"
j1 = T.TrainJob.start(job_for(str(runs / "r1"), "quick", 5), env=dict(os.environ), cwd=str(models))
st1 = wait(j1)
assert st1[0] == "start" and st1.count("epoch") == 5 and st1[-1] == "done", (st1, (runs / "r1" / "train.log").read_text())
kw = json.loads((runs / "r1" / "train" / "train_kwargs.json").read_text())
assert kw["trainer"] == "YOLOEPESegTrainer" and "6.cv2" in kw["freeze"] and kw["optimizer"] == "AdamW"
ev1 = json.loads((runs / "r1" / "eval.json").read_text(encoding="utf-8"))
assert ev1["verdict"]["level"] == "better", ev1["verdict"]
assert ev1["new"]["per_class"]["blue_block"]["recall"] == 100.0 and ev1["base"]["per_class"]["blue_block"]["recall"] == 0.0
assert ev1["base"]["overall"]["fp_per_img"] > ev1["new"]["overall"]["fp_per_img"]
os.environ["V381_FAKE_EPOCH_S"] = "0.3"
j2 = T.TrainJob.start(job_for(str(runs / "r2"), "full", 200), env=dict(os.environ), cwd=str(models))
t0 = time.time()
while "epoch" not in [e["stage"] for e in j2.events] and time.time() - t0 < 30:
    j2.poll()
    time.sleep(0.05)
j2.stop()
st2 = wait(j2)
os.environ["V381_FAKE_EPOCH_S"] = "0.02"
assert "stopping" in st2 and st2[-1] == "done" and st2.count("epoch") < 20, st2
kw2 = json.loads((runs / "r2" / "train" / "train_kwargs.json").read_text())
assert "freeze" not in kw2
print(f"[Q6] child process: {st1.count('epoch')} epoch events, quick = freeze + PE trainer, full = no freeze; original "
      f"vs new on {ev1['new']['overall']['images']} val images -> better; STOP after {st2.count('epoch')} epochs still "
      "evaluates")

# ------------------------------------------------------------------ Q7
dest = T.save_model(str(runs / "r1"), "红蓝方块 v1")
card = json.loads((Path(dest) / "model_card.json").read_text(encoding="utf-8"))
assert dest == str(TMP / "finetuned" / "红蓝方块_v1") and (Path(dest) / "红蓝方块_v1.pt").is_file()
assert card["classes"] == ["red_block", "blue_block"] and card["verdict"] == "better" and (Path(dest) / "eval.json").is_file()
try:
    T.save_model(str(runs / "r1"), "红蓝方块 v1")
    raise AssertionError("duplicate name must be refused")
except FileExistsError:
    pass
assert [c["name"] for c in T.list_saved()] == ["红蓝方块 v1"]
assert not T.discard_run(str(TMP / "finetuned"), str(runs)) and (TMP / "finetuned").is_dir()
assert T.discard_run(str(runs / "r2"), str(runs)) and not (runs / "r2").exists()
assert not any(p.name.startswith("yoloe") for p in (TMP / "finetuned").rglob("*.pt"))   # never picked by the tab
print("[Q7] saved model + card + eval in the fine-tune folder, duplicate refused, listed; discard only inside _runs")

# ------------------------------------------------------------------ Q8 GUI


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
sel = {"v": ()}
mtk.Listbox.return_value.curselection.side_effect = lambda: sel["v"]
states = {}


def mk_button(*_a, **kw):
    b = mock.MagicMock()
    txt = kw.get("text")
    b.configure.side_effect = lambda **k: states.__setitem__(txt, k.get("state", states.get(txt)))
    return b


mttk.Button.side_effect = mk_button
logs, uiq = [], []


class _Src:
    K = {"width": 1280, "height": 720, "fx": 640.0, "fy": 640.0, "ppx": 640.0, "ppy": 360.0}

    def running(self):
        return False


tab = types.SimpleNamespace(tk=mtk, ttk=mttk, cv=mock.MagicMock(), log=logs.append, src=_Src(), det_yolo=None,
                            v_ydir=FakeVar(value=str(models)), v_ymodel=FakeVar(value="yoloe-26s-seg.pt"),
                            v_yconf=FakeVar(value="0.25"), _ui=uiq.append, _grab_frames=lambda n: (None, []))
(TMP / "cameracalib").mkdir(exist_ok=True)
(TMP / "cameracalib" / "camera_lab.json").write_text(json.dumps({"root": str(root)}), encoding="utf-8")
lab = L.open_lab(tab)
texts = [c_.kwargs.get("text") for c_ in mttk.Notebook.return_value.add.call_args_list]
assert texts == ["① 相机参数", "② 深度噪声", "③ 定位精度", "④ 采集标注", "⑤ 训练 / 评估"], texts
tp = lab.train_page
assert tp is not None and tp.cls_rows == ["blue_block", "cup", "red_block"]
assert states["▶ 开始训练"] == "disabled" and states["✓ 保存这个模型"] == "disabled" and states["■ 停止（用目前最好的结果）"] == "disabled"
sel["v"] = (0, 2)
tp._data_changed()
pl = tp.on_check()
assert pl["ok"] and pl["classes"] == ["blue_block", "red_block"] and "✓ 可以训练" in tp.v_check.get()
assert states["▶ 开始训练"] == "normal" and tp.v_name.get().startswith("blue_block_red_block_")
tp.v_mode.set("full")
tp._mode_changed()
assert tp.v_epochs.get() == "80" and tp.v_batch.get() == "2"   # V3.82: 1280, batch 2 (intended)
tp.v_mode.set("quick")
tp._mode_changed()
tp.v_epochs.set("4")
tp.on_train()
assert states["■ 停止（用目前最好的结果）"] == "normal" and states["▶ 开始训练"] == "disabled"
assert states["✓ 保存这个模型"] == "disabled"
t0 = time.time()
while (tp.result is None or tp.job.running()) and time.time() - t0 < 120:
    lab._tick()
    time.sleep(0.05)
lab._tick()
assert tp.result is not None and tp.result["verdict"]["level"] == "better", tp.v_prog.get()
assert "有实质提升" in tp.v_verdict.get() and states["✓ 保存这个模型"] == "normal"
rows = [c_.kwargs.get("values") or c_.args[2] for c_ in mttk.Treeview.return_value.insert.call_args_list]
assert any(r_[0] == "blue_block" and r_[2] == "0 → 100" for r_ in rows), rows
assert any(r_[0] == "（平均）" for r_ in rows)
tp.v_name.set("blocks gui")
tp.on_save()
assert (TMP / "finetuned" / "blocks_gui" / "blocks_gui.pt").is_file() and "已保存" in tp.v_saved.get()
assert tp.job is None and states["✓ 保存这个模型"] == "disabled"
# a run still going when the window opens is picked up again
os.environ["V381_FAKE_EPOCH_S"] = "0.2"
j3 = T.TrainJob.start(job_for(str(runs / "r3"), "quick", 50), env=dict(os.environ), cwd=str(models))
lab.close()
lab2 = L.open_lab(tab)
assert lab2.train_page.job is not None and lab2.train_page.job.run_dir == str(runs / "r3")
lab2.train_page.on_stop()
wait(lab2.train_page.job)
os.environ["V381_FAKE_EPOCH_S"] = "0.02"
lab2.close()
# profile file open through the window
d = lab2.on_open_file(p1)
assert d and lab2.v_exp.get() == "200.0" and B.load_profile()["color"]["exposure"] == 200.0
saved = lab2.on_save_file(str(TMP / "elsewhere" / "copy.json"))
assert saved and L.load_profile_file(saved)["color"]["gain"] == 12.0
print("[Q8] ⑤ page: steps gate the buttons; check -> train (child) -> table + verdict -> save; a running training "
      "is picked up again; profile files open / save through the window")

# ------------------------------------------------------------------ Q9
SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("[Q9] release folder unchanged")
print("ALL V3.81 TESTS PASSED")
