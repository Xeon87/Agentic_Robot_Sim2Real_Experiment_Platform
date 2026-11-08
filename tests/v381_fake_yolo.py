"""V3.81 test helper: a fake ultralytics YOLOE + YOLOEPESegTrainer for the training child process.

Images: solid coloured rectangles on grey - red = class "red_block", blue = class "blue_block".
The ORIGINAL model (text prompts via set_classes) finds red blocks but misses blue ones (a poor zero-shot class);
a model TRAINED by this fake finds both - so the evaluation must report a real improvement.
train() calls the on_fit_epoch_end callbacks per epoch (honours trainer.stop) and writes <project>/<name>/weights/best.pt.
Environment: V381_FAKE_EPOCH_S = seconds per epoch (default 0.02)."""
import json
import os
import time

import numpy as np


class _Mod:
    def __init__(self, kids=None):
        self._kids = kids or []

    def named_children(self):
        return list(self._kids)

    def __len__(self):
        return len(self._kids)

    def __getitem__(self, i):
        return self._kids[i][1]


def _branch(n=3, k=3):
    return _Mod([(str(i), _Mod([(str(j), _Mod()) for j in range(k)])) for i in range(n)])


def _head():
    return _Mod([("cv2", _branch()), ("cv3", _branch()), ("one2one_cv3", _branch()), ("dfl", _Mod())])


class _Net:
    def __init__(self):
        self.model = [_Mod() for _ in range(6)] + [_head()]


class _T:
    def __init__(self, a):
        self.a = np.asarray(a)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _T(np.asarray(xyxy, float).reshape(-1, 4)), _T(conf), _T(cls)
        self.n = len(conf)

    def __len__(self):
        return self.n


class _Masks:
    def __init__(self, polys):
        self.xy = polys


class _Res:
    def __init__(self, dets, names):
        self.names = names
        self.boxes = _Boxes([d[0] for d in dets], [d[1] for d in dets], [d[2] for d in dets]) if dets else None
        self.masks = _Masks([np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], float)
                             for b, _c, _k in dets]) if dets else None


def _blobs(bgr, channel):
    """bounding boxes of saturated blobs in one BGR channel (rows / columns scan)"""
    m = (bgr[..., channel] > 180) & (bgr[..., (channel + 1) % 3] < 80) & (bgr[..., (channel + 2) % 3] < 80)
    out = []
    if not m.any():
        return out
    import NERO_V347_Yolo_Perception as YP
    cv2 = YP._cv2()
    n, lab, stats, _c = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if a > 30:
            out.append([float(x), float(y), float(x + w), float(y + h)])
    return out


class YOLOEPESegTrainer:
    pass


class YOLOE:
    def __init__(self, path):
        self.path = path
        self.model = _Net()
        self.callbacks = {}
        self.trained = False
        self.names = {0: "object"}
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    d = json.load(f)
                self.trained = bool(d.get("trained"))
                self.names = {int(k): v for k, v in d.get("names", {}).items()} or self.names
            except Exception:
                pass

    def set_classes(self, names, *_a, **_k):
        self.names = {i: n for i, n in enumerate(names)}

    def add_callback(self, ev, fn):
        self.callbacks.setdefault(ev, []).append(fn)

    def train(self, **kw):
        names = {}
        with open(kw["data"], encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if ":" in s and s.split(":")[0].isdigit():
                    names[int(s.split(":")[0])] = json.loads(s.split(":", 1)[1].strip())
        out = os.path.join(kw["project"], kw["name"])
        os.makedirs(os.path.join(out, "weights"), exist_ok=True)
        with open(os.path.join(out, "train_kwargs.json"), "w", encoding="utf-8") as f:
            json.dump({k: (v if k != "trainer" else getattr(v, "__name__", str(v))) for k, v in kw.items()}, f)
        tr = type("Trainer", (), {})()
        tr.epochs, tr.stop, tr.tloss = int(kw["epochs"]), False, 1.0
        dt = float(os.environ.get("V381_FAKE_EPOCH_S", "0.02"))
        for e in range(tr.epochs):
            time.sleep(dt)
            tr.epoch, tr.metrics, tr.tloss = e, {"metrics/mAP50(M)": 0.5 + 0.4 * (e + 1) / tr.epochs}, 1.0 / (e + 1)
            for fn in self.callbacks.get("on_fit_epoch_end", []):
                fn(tr)
            if tr.stop:
                break
        with open(os.path.join(out, "results.csv"), "w") as f:
            f.write("epoch,mAP50\n")
        with open(os.path.join(out, "weights", "best.pt"), "w", encoding="utf-8") as f:
            json.dump({"trained": True, "names": names, "epochs_done": e + 1}, f)

    def predict(self, bgr, conf=0.25, **_kw):
        inv = {v: k for k, v in self.names.items()}
        dets = []
        for nm, ch, sc in (("red_block", 2, 0.9 if self.trained else 0.6), ("blue_block", 0, 0.85 if self.trained else 0.0)):
            if nm not in inv or sc < conf:
                continue
            for b in _blobs(bgr, ch):
                dets.append((b, sc, inv[nm]))
        if not self.trained and "red_block" in inv:              # the original model also sees a false red block
            dets.append(([5.0, 5.0, 40.0, 40.0], 0.3, inv["red_block"]))
        return [_Res(dets, self.names)]
