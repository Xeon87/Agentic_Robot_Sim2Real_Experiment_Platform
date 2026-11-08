#!/usr/bin/env python3
"""V3.83 tests:
S1  Version & markers across all relevant files; NERO_V308 untouched.
S2  GPU check & self-repair (NERO_V383_Gpu_Check.py).
S3  Strict matching bug fix (finetune_covers).
S4  Multi-model distribution & execution in YoloEDetector.detect (with fake ultralytics).
S5  CameraTabUI Menubutton multiselect UI & locate settings.
S6  TrainPage (⑤) & 采集标注 (④).
S7  Depth filtering (fuse_depth with method='median').
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import NERO_V347_Yolo_Perception as YP
import NERO_V383_Gpu_Check as GC383
import NERO_V378_D435i_Bench as B378

TMP = Path(tempfile.mkdtemp(prefix="nero_test_v383_"))
MODELS = TMP / "models"
CALIB = TMP / "calib"
os.environ["NERO_YOLO_DIR"] = str(MODELS)
os.environ["NERO_CALIB_DIR"] = str(CALIB)
os.environ["NERO_V381_FINETUNE_DIR"] = str(MODELS / "finetuned")
YP.DEFAULT_MODEL_DIR = str(MODELS)

# ------------------------------------------------------------------ S1 Version & markers
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6")
import NERO_V312_Console as CON
import NERO_V313_Real_Robot as RR
import NERO_V349_Real_Wrist_Camera as W49
import NERO_V381_Yolo_Train as TR381

for m in (CON, RR, YP, W49, TR381, GC383):
    assert m.VERSION in ("V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6"), (m.__name__, m.VERSION)

import NERO_V325_Camera as CAM
assert CAM.VERSION == "V3.25"

P08_SHA = "020cc6680f181872"
h08 = hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest()
assert h08.startswith(P08_SHA), f"V308 modified: {h08}"
assert "V3.83" not in (ROOT / "NERO_V366_Real_Cup_Yolo.py").read_text(encoding="utf-8")
print("[S1] Versions V3.83+ verified; V308 and V366 untouched")

# ------------------------------------------------------------------ S2 GPU check & self-repair
os.environ["NERO_TEST_TORCH_CUDA"] = "1"
ok, info = GC383.check_torch_cuda()
assert ok and info == "test_gpu"

logs = []
assert GC383.preflight_check(log=logs.append, interactive=False) is True
assert any("CUDA 可用" in l for l in logs)
logs.clear()

os.environ["NERO_TEST_TORCH_CUDA"] = "CUDA unknown error"
os.environ["NERO_TEST_NVIDIA_SMI"] = "failed to contact driver"
assert GC383.preflight_check(log=logs.append, interactive=False) is False
assert any("未检测到正常 NVIDIA 驱动" in l for l in logs)
logs.clear()

os.environ["NERO_TEST_TORCH_CUDA"] = "CUDA unknown error"
os.environ["NERO_TEST_NVIDIA_SMI"] = "1"


def fake_runner_ok(cmd, **kw):
    if "modprobe" in cmd:
        os.environ["NERO_TEST_TORCH_CUDA"] = "1"
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


assert GC383.preflight_check(log=logs.append, interactive=False, run_cmd_fn=fake_runner_ok) is True
assert any("修复成功" in l for l in logs)
logs.clear()

os.environ["NERO_TEST_TORCH_CUDA"] = "CUDA unknown error"


def fake_runner_fail(cmd, **kw):
    return types.SimpleNamespace(returncode=1, stdout="", stderr="Permission denied")


assert GC383.preflight_check(log=logs.append, interactive=False, run_cmd_fn=fake_runner_fail) is False
assert any("GPU 自修复未成功" in l for l in logs)
logs.clear()

os.environ["NERO_TEST_PKEXEC_SUCCESS"] = "1"
ok_pk, _ = GC383.repair_uvm_pkexec(log=logs.append)
assert ok_pk is True
assert any("重载 nvidia_uvm 成功" in l for l in logs)
logs.clear()

os.environ["NERO_TEST_PKEXEC_SUCCESS"] = "0"
fail_pk, _ = GC383.repair_uvm_pkexec(log=logs.append)
assert fail_pk is False

for k in ("NERO_TEST_TORCH_CUDA", "NERO_TEST_NVIDIA_SMI", "NERO_TEST_PKEXEC_SUCCESS"):
    os.environ.pop(k, None)

print("[S2] GPU preflight check & self-repair logic verified with mocks")

# ------------------------------------------------------------------ S3 Strict finetune_covers
card_egg = {"name": "egg model", "classes": ["egg"], "synonyms": {}}
card_carton = {"name": "carton model", "classes": ["egg carton"], "synonyms": {}}

assert not YP.finetune_covers(card_egg, "egg carton")
assert not YP.finetune_covers(card_egg, "egg cartons")
assert YP.finetune_covers(card_egg, "egg")
assert YP.finetune_covers(card_egg, "eggs")
assert YP.finetune_covers(card_egg, "white egg")
assert YP.finetune_covers(card_egg, "white round object")
assert YP.finetune_covers(card_carton, "egg carton")
assert YP.finetune_covers(card_carton, "egg cartons")
assert not YP.finetune_covers(card_carton, "egg")
assert not YP.finetune_covers(card_carton, "cup")
print("[S3] Strict finetune_covers: 'egg carton' strictly separated from 'egg'; plural & synonyms supported")

# ------------------------------------------------------------------ S4 Multi-model distribution in YoloEDetector
MODELS.mkdir(parents=True, exist_ok=True)
(MODELS / "yoloe-26s-seg.pt").write_bytes(b"base")
(MODELS / "yoloe-26s-seg-pf.pt").write_bytes(b"pf")
(MODELS / "mobileclip2_b.ts").write_bytes(b"enc")

ft_egg_dir = MODELS / "finetuned" / "eggs_v1"
ft_egg_dir.mkdir(parents=True, exist_ok=True)
(ft_egg_dir / "eggs_v1.pt").write_bytes(b"ft_egg")
card_egg_full = {
    "name": "eggs v1", "file": "eggs_v1.pt", "classes": ["egg"], "base_model": "yoloe-26s-seg.pt",
    "created": "2026-11-04 10:00:00", "overall_new": {"recall": 95.0, "ap50": 0.93}
}
(ft_egg_dir / "model_card.json").write_text(json.dumps(card_egg_full), encoding="utf-8")

ft_carton_dir = MODELS / "finetuned" / "cartons_v1"
ft_carton_dir.mkdir(parents=True, exist_ok=True)
(ft_carton_dir / "cartons_v1.pt").write_bytes(b"ft_carton")
card_carton_full = {
    "name": "cartons v1", "file": "cartons_v1.pt", "classes": ["egg carton"], "base_model": "yoloe-26s-seg.pt",
    "created": "2026-11-05 10:00:00", "overall_new": {"recall": 92.0, "ap50": 0.90}
}
(ft_carton_dir / "model_card.json").write_text(json.dumps(card_carton_full), encoding="utf-8")

combo_str = "yoloe-26s-seg.pt;finetuned/eggs_v1/eggs_v1.pt;finetuned/cartons_v1/cartons_v1.pt"
base_n, fts_n = YP.parse_model_set(combo_str)
assert base_n == "yoloe-26s-seg.pt" and len(fts_n) == 2

base_p, fts_p = YP.find_model_set(str(MODELS), combo_str)
assert base_p == str(MODELS / "yoloe-26s-seg.pt") and len(fts_p) == 2

found_combo = YP.find_model(str(MODELS), combo_str)
assert isinstance(found_combo, str) and os.path.dirname(found_combo) == str(MODELS)

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
        self.is_egg = "eggs_v1" in path
        self.is_carton = "cartons_v1" in path
        if self.is_egg:
            self.names = {0: "egg"}
        elif self.is_carton:
            self.names = {0: "egg carton"}
        else:
            self.names = {}

    def set_classes(self, names, *a, **k):
        calls.append(("set_classes", os.path.basename(self.path), list(names)))
        self.names = {i: n for i, n in enumerate(names)}

    def predict(self, bgr, **kw):
        calls.append(("predict", os.path.basename(self.path)))
        rows = []
        if self.is_egg:
            rows.append(([100, 100, 160, 150], 0.95, 0))
        elif self.is_carton:
            rows.append(([500, 200, 700, 400], 0.90, 0))
        else:
            for i, n in self.names.items():
                if n == "cup":
                    rows.append(([300, 300, 380, 400], 0.85, i))
                    rows.append(([101, 101, 159, 149], 0.60, i))
        r = types.SimpleNamespace(boxes=_Boxes(rows) if rows else None, masks=None, names=self.names)
        return [r]


fake_ul = types.ModuleType("ultralytics")
fake_ul.YOLOE = FakeYOLOE
fake_mod = types.ModuleType("ultralytics.models.yolo.yoloe")
fake_mod.YOLOEVPSegPredictor = object
sys.modules["ultralytics"] = fake_ul
sys.modules["ultralytics.models"] = types.ModuleType("ultralytics.models")
sys.modules["ultralytics.models.yolo"] = types.ModuleType("ultralytics.models.yolo")
sys.modules["ultralytics.models.yolo.yoloe"] = fake_mod

det = YP.YoloEDetector(log=lambda m: None)
rgb = np.zeros((720, 1280, 3), np.uint8)

dets = det.detect(rgb, "text", combo_str, prompts=["egg", "egg carton", "cup"], conf=0.25)
assert not any(c[0] == "set_classes" and c[1] in ("eggs_v1.pt", "cartons_v1.pt") for c in calls)
base_set_classes = [c for c in calls if c[0] == "set_classes" and c[1] == "yoloe-26s-seg.pt"]
assert len(base_set_classes) == 1 and base_set_classes[0][2] == ["cup"]

det_map = {d["name"]: d for d in dets}
assert "egg" in det_map and det_map["egg"]["model"] == "eggs v1"
assert "egg carton" in det_map and det_map["egg carton"]["model"] == "cartons v1"
assert "cup" in det_map and det_map["cup"]["model"] == "original"
assert len(dets) == 3

calls.clear()
dets_single = det.detect(rgb, "text", "yoloe-26s-seg.pt", prompts=["cup"], conf=0.25)
assert len(dets_single) == 2 and all(d["model"] == "original" for d in dets_single)

calls.clear()
dets_wild = det.detect(rgb, "text", combo_str, prompts=["全部"], conf=0.25)
assert any(d["name"] == "egg" for d in dets_wild) and any(d["name"] == "egg carton" for d in dets_wild)

calls.clear()
fake_ref = {"image": rgb, "boxes": [[10, 10, 50, 50]], "cls": [0]}
dets_vp = det.detect(rgb, "vp", combo_str, ref=fake_ref, conf=0.25)
predict_calls = [c[1] for c in calls if c[0] == "predict"]
assert "yoloe-26s-seg.pt" in predict_calls and not any(c in predict_calls for c in ("eggs_v1.pt", "cartons_v1.pt"))

del sys.modules["ultralytics"]
print("[S4] Multi-model distribution verified")

# ------------------------------------------------------------------ S5 CameraTabUI Menubutton multiselect UI & settings
class FakeVar:
    def __init__(self, val=""):
        self.val = val

    def get(self):
        return self.val

    def set(self, v):
        self.val = v if isinstance(v, bool) else str(v)


class FakeMenu:
    def __init__(self):
        self.items = []

    def delete(self, *a):
        self.items.clear()

    def add_checkbutton(self, label, **kw):
        self.items.append(("checkbutton", label, kw))

    def add_separator(self):
        self.items.append(("separator",))


class FakeMB:
    def __init__(self):
        self.text = ""

    def configure(self, **kw):
        if "text" in kw:
            self.text = kw["text"]


class FakeTab:
    def __init__(self):
        self.tk = types.SimpleNamespace(BooleanVar=lambda value=False: FakeVar(value))
        self.YP = YP
        self.v_ydir = FakeVar(str(MODELS))
        self.v_ymodel = FakeVar(YP.DEFAULT_MODEL)
        self.v_method = FakeVar("yolo_text")
        self.v_yconf = FakeVar("0.30")
        self.v_yprompt = FakeVar(YP.DEFAULT_TEXT_PROMPTS)
        self.v_obj_names = FakeVar("")
        self.v_hover_mm = FakeVar("60")
        self.v_dfilter = FakeVar("默认（单帧）")
        self.v_track_on = FakeVar(True)
        self.v_egg_l = FakeVar("55")
        self.v_egg_w = FakeVar("42")
        self.v_white = FakeVar(True)
        self.v_ysave = FakeVar(False)
        self.v_dres = FakeVar("1280x720")
        self.v_dpreset = FakeVar("High Accuracy")
        self.ref = None
        self.v_ref = FakeVar("")
        self.src = None
        self.mb_ymodel = FakeMB()
        self.menu_ymodel = FakeMenu()
        self._ymodel_vars = {}
        self.log_msgs = []

    def log(self, m):
        self.log_msgs.append(m)

    _hover_user_str = lambda self: self.v_hover_mm.get()
    _update_ymodel_menu = CAM.CameraTabUI._update_ymodel_menu
    _on_ymodel_toggle = CAM.CameraTabUI._on_ymodel_toggle
    _refresh_ymodel_summary = CAM.CameraTabUI._refresh_ymodel_summary
    on_repair_gpu = CAM.CameraTabUI.on_repair_gpu
    _locate_settings = CAM.CameraTabUI._locate_settings
    _locate_apply = CAM.CameraTabUI._locate_apply


tab = FakeTab()
tab._update_ymodel_menu()
assert "yoloe-26s-seg.pt" in tab.v_ymodel.get()
assert "eggs_v1.pt" in tab.v_ymodel.get() and "cartons_v1.pt" in tab.v_ymodel.get()
assert "原模型 + 2 精调模型" in tab.mb_ymodel.text

tab._on_ymodel_toggle("finetuned/cartons_v1/cartons_v1.pt", False)
assert "cartons_v1.pt" not in tab.v_ymodel.get() and "eggs_v1.pt" in tab.v_ymodel.get()
assert "原模型 + eggs_v1" in tab.mb_ymodel.text

st = tab._locate_settings()
assert st["filter"] == "默认（单帧）" and "eggs_v1.pt" in st["model"]

tab.v_dfilter.set("滤波 + 5 帧中值（推荐）")
st2 = tab._locate_settings()
assert st2["filter"] == "滤波 + 5 帧中值（推荐）"

tab2 = FakeTab()
tab2._locate_apply(st2, "fake_path")
assert tab2.v_dfilter.get() == "滤波 + 5 帧中值（推荐）" and "eggs_v1.pt" in tab2.v_ymodel.get()

os.environ["NERO_TEST_PKEXEC_SUCCESS"] = "1"
tab.on_repair_gpu()
assert any("nvidia_uvm 重载成功" in m for m in tab.log_msgs)
os.environ.pop("NERO_TEST_PKEXEC_SUCCESS", None)
print("[S5] CameraTabUI Menubutton multiselect UI & settings persistence verified")

# ------------------------------------------------------------------ S6 TrainPage & 采集标注
class FakeLab:
    def __init__(self, tab_):
        self.tab = tab_
        self.ds = types.SimpleNamespace(root=str(TMP / "dataset"))


class FakeTrainPage:
    def __init__(self, lab_):
        self.lab = lab_
        self.logs = []
        self.v_active = FakeVar()

    def log(self, m):
        self.logs.append(m)

    def _card(self):
        return card_carton_full

    def _model_ref(self, c):
        return f"finetuned/{c['file'].split('.')[0]}/{c['file']}"

    _show_active = TR381.TrainPage._show_active
    _set_tab_model = TR381.TrainPage._set_tab_model
    on_apply_model = TR381.TrainPage.on_apply_model
    on_restore_model = TR381.TrainPage.on_restore_model


lab = FakeLab(tab)
tp = FakeTrainPage(lab)
tp.on_apply_model()
assert "cartons_v1.pt" in tab.v_ymodel.get() and "eggs_v1.pt" in tab.v_ymodel.get()
assert any("Camera调试 页加入精调模型" in l for l in tp.logs)
tp.logs.clear()

tp.on_restore_model()
assert tab.v_ymodel.get() == YP.DEFAULT_MODEL
assert any("Camera调试 页恢复原模型" in l for l in tp.logs)

import NERO_V379_Camera_Lab as LAB379
class FakeCaptureTab:
    def _label_sources(self):
        return LAB379.CameraLab._label_sources(self)

cap = FakeCaptureTab()
assert "组合模型" in cap._label_sources()
print("[S6] TrainPage integration & 采集标注 verified")

# ------------------------------------------------------------------ S7 Depth filtering
depths = [
    np.array([[1.0, 1.2], [1.1, 0.0]], dtype=np.float32),
    np.array([[1.0, 1.4], [1.1, 1.5]], dtype=np.float32),
    np.array([[1.0, 1.0], [1.1, 1.3]], dtype=np.float32),
]
fused_med = YP.fuse_depth(depths, method="median")
assert np.isclose(fused_med[0, 1], 1.2) and np.isclose(fused_med[1, 1], 1.4)
print("[S7] Depth filtering (fuse_depth median_stack) verified")

shutil.rmtree(TMP, ignore_errors=True)
print("\nALL V3.83 TESTS PASSED")
