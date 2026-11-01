#!/usr/bin/env python3
"""V3.47.2: Camera调试 table row height / column auto-fit, non-overlapping short YOLO labels, ROI 3D speedup & GPU/CPU log.

Q1 treeview_style + autofit_treeview: rowheight = font linespace + 8; columns auto-widen to fit headers and cell text;
   only the designated column ('why' / 'q') stretches
Q2 short_egg_label ('egg_1' -> 'e1', '✗' -> '✗') and place_yolo_labels: adjacent / top-border boxes avoid label overlap
   and stay inside the 640x360 canvas
Q3 roi_boxes + fuse_depth(depths, dets2d): overlapping candidate ROIs merged, class-1 ignored; 3D locate results match
   full-frame depth fusion bit-for-bit
Q4 ROI local_plane + 1280x720 5-frame 3D speed: local_plane with bbox matches full-mask scan; 2-egg 5-frame fuse + locate
   completes well within the target budget
Q5 YoloEDetector.device_name + GPU/CPU logging: reports GPU vs CPU in model load and locate logs; CUDA RuntimeError
   falls back to CPU and updates info['device']
Q6 NERO_V308_Camera_Perception.py unchanged; sim / wrist / fixed camera modules untouched
"""
import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import v347_fake as F                                  # noqa: E402
import NERO_V347_Yolo_Perception as YP                 # noqa: E402
import NERO_V325_Camera as CAM                         # noqa: E402

assert YP.VERSION in ("V3.47.2", "V3.48", "V3.49", "V3.50", "V3.51", "V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58") and (ROOT / "VERSION").read_text().strip() in ("V3.47.2", "V3.48", "V3.49", "V3.50", "V3.51", "V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58")
TD = tempfile.mkdtemp(prefix="v3472_")
YP.REF_JSON, YP.REF_IMG = os.path.join(TD, "yolo_ref.json"), os.path.join(TD, "yolo_ref.jpg")
MP = os.path.join(TD, "yoloe-26s-seg.pt")
Path(MP).write_bytes(b"w")
FENCE = [[-0.8, -0.4], [-0.1, -0.4], [-0.1, 0.4], [-0.8, 0.4]]
F.install()


# ------------------------------------------------------------------ Q1
class _FakeStyle:
    cfg = {}

    def __init__(self, _w=None):
        pass

    def configure(self, name, **kw):
        self.cfg.setdefault(name, {}).update(kw)


class _FakeTk:
    def call(self, *args):
        if args[:3] == ("font", "metrics", "TkDefaultFont"):
            return 18
        if args[:3] == ("font", "measure", "TkDefaultFont"):
            s = args[3]
            return sum(14 if ord(ch) > 127 else 8 for ch in str(s))
        return 0


class _FakeTree:
    def __init__(self, rows):
        self.tk = _FakeTk()
        self._rows = list(rows)
        self.cols = {}

    def get_children(self):
        return [str(i) for i in range(len(self._rows))]

    def item(self, iid, opt=None):
        vals = self._rows[int(iid)]
        return vals if opt == "values" else {"values": vals}

    def column(self, c, **kw):
        self.cols[c] = kw


fake_ttk = type("TTK", (), {"Style": _FakeStyle})
w_dummy = type("W", (), {"tk": _FakeTk()})()
st_name, rh = CAM.treeview_style(fake_ttk, w_dummy)
assert st_name == "V345Calib.Treeview" and rh == 26 and _FakeStyle.cfg["V345Calib.Treeview"]["rowheight"] == 26
cols = (("id", "编号", 60), ("iou", "轮廓重合", 64), ("why", "说明", 260))
tv0 = _FakeTree([])
w0 = CAM.autofit_treeview(tv0, cols, stretch_col="why")
assert w0["iou"] >= 4 * 14 + 20 and not tv0.cols["id"]["stretch"] and not tv0.cols["iou"]["stretch"]
assert tv0.cols["why"]["stretch"] and tv0.cols["why"]["anchor"] == "w" and tv0.cols["id"]["anchor"] == "center"
tv1 = _FakeTree([("egg_1", "0.94", "✓ 平躺，轮廓对齐（鸡蛋上深度不足（0%），按轮廓 + 桌面定位）")])
w1 = CAM.autofit_treeview(tv1, cols, stretch_col="why")
assert w1["why"] > w0["why"] and w1["id"] >= 60
print(f"[Q1] treeview_style rowheight {rh} px (= linespace 18 + 8); autofit_treeview widens headers/cells ({w0['why']} -> {w1['why']} px)")

# ------------------------------------------------------------------ Q2
assert CAM.short_egg_label("egg_1") == "e1" and CAM.short_egg_label("egg_12") == "e12" and CAM.short_egg_label("✗") == "✗"
adj = [
    {"box": [500, 300, 560, 350], "status": "egg", "label": "e1"},
    {"box": [504, 302, 564, 352], "status": "untrusted", "label": "e2"},
    {"box": [200, 4, 260, 54], "status": "egg", "label": "e3"},        # near top edge: top label would clip y < 0
]
placed = CAM.place_yolo_labels(adj, canvas_w=640, canvas_h=360)
assert len(placed) == 3 and placed[0]["pos"] == "top" and placed[1]["pos"] in ("bottom", "right", "left")
assert placed[2]["pos"] in ("bottom", "right", "inside")
for i in range(len(placed)):
    r = placed[i]["label_rect"]
    assert 0 <= r[0] and 0 <= r[1] and r[2] <= 640 and r[3] <= 360, (i, r)
    for j in range(i + 1, len(placed)):
        assert not CAM._rects_overlap(r, placed[j]["label_rect"]), (r, placed[j]["label_rect"])
print(f"[Q2] short_egg_label ('egg_1'->'e1'); place_yolo_labels avoids overlap ({[p['pos'] for p in placed]})")

# ------------------------------------------------------------------ Q3
obs = F.default_objects()
rgb, depth, ids = F.render(obs)
F.STATE.scene = (obs, ids)
ref = YP.ref_save(rgb, [[540, 380, 600, 440]], [0])
logs = []
det = YP.YoloEDetector(log=logs.append)
d2 = det.detect(rgb, "vp", MP, ref=ref)
rois = YP.roi_boxes(d2 + [{"cls": 1, "box": [10, 10, 50, 50]}], depth.shape)
assert len(rois) >= 1 and all(0 <= x0 < x1 <= 1280 and 0 <= y0 < y1 <= 720 for x0, y0, x1, y1 in rois)
roi_px = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in rois)
assert roi_px < int(depth.size * 0.65), (roi_px, depth.size)
rng = np.random.default_rng(7)
depths = [np.where(depth > 0, depth + rng.normal(0, 0.0005, depth.shape), 0).astype(np.float32) for _ in range(5)]
df_full = YP.fuse_depth(depths)
df_roi = YP.fuse_depth(depths, d2)
res_full, diag_full = YP.locate(d2, df_full, F.K, F.T_BC, table_z=-0.012, fence_poly=FENCE)
res_roi, diag_roi = YP.locate(d2, df_roi, F.K, F.T_BC, table_z=-0.012, fence_poly=FENCE)
for a, b in zip(res_full, res_roi):
    assert {k: v for k, v in a.items() if k != "fit_ms"} == {k: v for k, v in b.items() if k != "fit_ms"}, (a, b)
assert diag_full == diag_roi
print(f"[Q3] ROI depth fusion ({roi_px * 100 // depth.size}% of 1280x720) -> 3D results identical to full-frame fusion")

# ------------------------------------------------------------------ Q4
eggs2 = [o for o in obs if o["name"] in ("E1", "E2")]
rgb2, depth2, ids2 = F.render(eggs2)
F.STATE.scene = (eggs2, ids2)
d2_eggs = det.detect(rgb2, "vp", MP, ref=ref)
depths2 = [depth2.copy() for _ in range(5)]
t0 = time.time()
df2 = YP.fuse_depth(depths2, d2_eggs)
res2, _ = YP.locate(d2_eggs, df2, F.K, F.T_BC, table_z=0.0, fence_poly=FENCE)
dt_ms = (time.time() - t0) * 1000.0
assert len(res2) == 2 and all(r["trusted"] for r in res2) and dt_ms < 500.0, (dt_ms, res2)
m0 = d2_eggs[0]["mask"]
p_no_bb, n_no_bb = YP.local_plane(df2, F.K, F.T_BC, m0, None, 0.8, bbox=None)
p_bb, n_bb = YP.local_plane(df2, F.K, F.T_BC, m0, None, 0.8, bbox=YP._mask_bbox(m0))
assert n_no_bb == n_bb and np.allclose(p_no_bb[0], p_bb[0]) and abs(p_no_bb[1] - p_bb[1]) < 1e-9
print(f"[Q4] 2-egg 1280x720 5-frame fuse + 3D locate in {dt_ms:.0f} ms; ROI local_plane matches full-mask scan")

# ------------------------------------------------------------------ Q5
assert det.info.get("device") in ("GPU", "CPU")
assert any(f"，vp，{det.info['device']}，加载" in m for m in logs), logs
det_gpu = YP.YoloEDetector(log=logs.append)
det_gpu.device = "cuda:0"
assert det_gpu.device_name() == "GPU"
det_cpu = YP.YoloEDetector(log=logs.append)
det_cpu.device = "cpu"
assert det_cpu.device_name() == "CPU"


class _CudaFailModel:
    predictor = type("P", (), {"device": "cuda:0"})()

    def __init__(self):
        self.calls = 0

    def predict(self, bgr, **kw):
        self.calls += 1
        if kw.get("device") != "cpu":
            raise RuntimeError("CUDA error: out of memory")
        self.predictor.device = "cpu"
        return ["ok"]


fm = _CudaFailModel()
det_fb = YP.YoloEDetector(log=logs.append)
det_fb.info["device"] = "GPU"
out_fb = det_fb._predict(fm, rgb2, 0.25)
assert out_fb == ["ok"] and det_fb.device == "cpu" and det_fb.info["device"] == "CPU" and det_fb.device_name(fm) == "CPU"
assert any("GPU 出错" in m and "改用 CPU" in m for m in logs)
print("[Q5] YoloEDetector device_name reports GPU/CPU in load & locate logs; CUDA failure falls back to CPU")

# ------------------------------------------------------------------ Q6
assert hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest().startswith("020cc6680f181872")
for f in ("NERO_V2_7_1_Unified_Backend.py", "NERO_V326_Wrist_Camera.py", "NERO_V331_Fixed_Camera.py", "NERO_V2_7_1_Agent.py"):
    assert "NERO_V347" not in (ROOT / f).read_text(), f
print("[Q6] NERO_V308_Camera_Perception.py unchanged; simulation / wrist / fixed camera modules untouched")
print("ALL V3.47.2 TESTS PASSED")
