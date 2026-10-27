#!/usr/bin/env python3
"""NERO V3.47 YOLO egg perception for the real D435i (Camera调试 → 视觉定位).

Pipeline (per capture)
----------------------
1. 2D: YOLOE-26 segmentation (ultralytics, the console's own python) on the colour image.
   * visual prompt (default): a reference image + boxes saved in config/ ("鸡蛋" boxes = class 0, optional
     "非鸡蛋" boxes = class 1, e.g. an AirPods case) -> every detection is labelled egg / not-egg;
   * text prompt: e.g. "white round object, white egg, egg" (needs mobileclip2_b.ts next to the model, no network).
2. 3D (V3.47.1, no depth on the egg needed): the table plane is fitted in a band 3-6 cm around each object (cables,
   other objects and invalid depth excluded); the egg (real size, default 55 x 42 mm) rests on that plane, lying
   (standing only when clearly better); its position and yaw are found by matching the projected silhouette with the
   YOLO mask (IoU).  The D435 flattens small white eggs at ~0.8 m (measured 17-28 mm instead of 42), so depth on the
   egg is used only for the displayed height and to reject flat things (< 12 mm).
3. checks: silhouette overlap (trusted >= 0.80, rejected < 0.60), image border, YOLO confidence, safety zone, plane
   found, hand-eye calibration verified.

Each result: det_id, status egg / untrusted / rejected, reasons, xyz_m, axis_world, quat_xyzw, fit_rms_mm, height_mm,
n_points, confidence, trusted, yolo{conf, box, cls}.  No 'location': the simulation's holder / basket
coordinates do not apply to the real bench.

NERO_V308_Camera_Perception.py is used (fit_spheroid, temporal_fuse, helpers) but not changed;
simulation, wrist camera and fixed camera modes do not import this module.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import threading
import time

VERSION = "V3.47.1"
HERE = os.path.dirname(os.path.realpath(__file__))
DEFAULT_MODEL_DIR = os.environ.get("NERO_YOLO_DIR") or os.path.expanduser("~/Downloads/objectmodels")
DEFAULT_MODEL = "yoloe-26s-seg.pt"
REF_JSON = os.path.join(HERE, "config", "yolo_ref.json")
REF_IMG = os.path.join(HERE, "config", "yolo_ref.jpg")
SAMPLES_DIR = os.path.join(HERE, "config", "yolo_samples")
DEFAULT_TEXT_PROMPTS = "white round object, white egg, egg"
EGG_SIZE_MM = (55.0, 42.0)                 # length, width of the real eggs (measured)
CONF_MIN = 0.25                             # YOLO confidence below this: not listed
IMGSZ = 1280
FUSE_FRAMES = 5
ERODE_FRAC, ERODE_MIN_PX = 0.08, 2          # mask erosion for the depth-on-egg check: 8 % of sqrt(area)
EDGE_PX = 3                                 # box this close to the image border -> cut off -> untrusted
# V3.47.1 POSE FROM SILHOUETTE: table plane in a band 3-6 cm around each object, egg pose from the YOLO silhouette
PLANE_RING_M = (0.03, 0.06)
PLANE_MIN_POINTS = 80
PLANE_MAX_TILT_DEG = 15.0
IOU_TRUST, IOU_REJECT = 0.80, 0.60          # silhouette overlap: trusted >= 0.80, rejected < 0.60
BLEED_MAX_PX = 8.0                          # YOLO mask this much larger / smaller than the egg -> not trusted
STANDING_MARGIN = 0.03                      # standing pose only if its overlap is this much better
STANDING_HINT_MM = 5.0                      # ... or measured height >= egg width + 5 mm (when depth exists)
DEPTH_COVER_MIN = 0.5                       # depth on the egg used (height display / flat check) only above 50 %
FLAT_MAX_MM = 12.0                          # measured height below this -> flat object (paper, label) -> rejected
TEXT_ENCODERS = {"26": "mobileclip2_b.ts", "": "mobileclip_blt.ts"}

_CWD_LOCK = threading.Lock()
_SHA_CACHE = {}


class YoloUnavailable(RuntimeError):
    """ultralytics / model file / text encoder missing: message is shown to the user as is."""


# =========================================================================== files
def model_dirs(primary=None):
    out = []
    for d in (primary, DEFAULT_MODEL_DIR, os.path.join(HERE, "config", "models"), os.path.expanduser("~")):
        if d and os.path.abspath(os.path.expanduser(d)) not in out:
            out.append(os.path.abspath(os.path.expanduser(d)))
    return out


def list_models(model_dir):
    d = os.path.abspath(os.path.expanduser(model_dir or DEFAULT_MODEL_DIR))
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "yoloe*.pt")))


def find_model(model_dir, name=DEFAULT_MODEL):
    """Absolute path of the model: the chosen folder first, then the fall-back folders."""
    if name and os.path.isabs(name) and os.path.isfile(name):
        return name
    for d in model_dirs(model_dir):
        p = os.path.join(d, name or DEFAULT_MODEL)
        if os.path.isfile(p):
            return p
    return None


def text_encoder_name(model_path):
    return TEXT_ENCODERS["26"] if "26" in os.path.basename(model_path or "") else TEXT_ENCODERS[""]


def file_sha256(path):
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 20), b""):
                h.update(blk)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def parse_prompts(text):
    return [s.strip() for s in str(text or "").replace("，", ",").split(",") if s.strip()]


def _cv2():
    import NERO_V325_Camera as CAM          # V3.46 Qt-variable guard
    return CAM._import_cv2()


# =========================================================================== reference (visual prompt)
def ref_load(path=None):
    """{'image': abs path, 'boxes': [[x0,y0,x1,y1]...], 'cls': [0|1...], 'saved': str} or None."""
    path = path or REF_JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            r = json.load(f)
        img = os.path.join(os.path.dirname(path), r.get("image") or os.path.basename(REF_IMG))
        boxes = [[float(v) for v in b] for b in r["boxes"]]
        cls = [int(c) for c in r["cls"]]
        if not os.path.isfile(img) or len(boxes) != len(cls) or 0 not in cls or not set(cls) <= {0, 1}:
            return None
        return {"image": img, "boxes": boxes, "cls": cls, "saved": r.get("saved", ""), "size": r.get("size")}
    except Exception:
        return None


def ref_save(rgb, boxes, cls, path=None, img_path=None):
    """Save the reference image (colour, RGB array) and its boxes. Needs >= 1 egg box (class 0)."""
    path, img_path = path or REF_JSON, img_path or REF_IMG
    cls = [int(c) for c in cls]
    if not boxes or len(boxes) != len(cls) or 0 not in cls or not set(cls) <= {0, 1}:
        raise ValueError("参考里至少要有 1 个「鸡蛋」框")
    cv2 = _cv2()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not cv2.imwrite(img_path, rgb[..., ::-1]):
        raise IOError(f"写不了 {img_path}")
    norm = []
    for b in boxes:
        x0, y0, x1, y1 = [float(v) for v in b]
        norm.append([round(min(x0, x1), 1), round(min(y0, y1), 1), round(max(x0, x1), 1), round(max(y0, y1), 1)])
    d = {"version": VERSION, "image": os.path.basename(img_path), "boxes": norm, "cls": cls,
         "names": ["egg", "not_egg"], "size": [int(rgb.shape[1]), int(rgb.shape[0])],
         "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return ref_load(path)


def ref_text(ref):
    if not ref:
        return "还没有参考：点「框选参考」，在画面上框 1–3 个鸡蛋（可再框干扰物为「非鸡蛋」），再点「保存参考」"
    ne = sum(1 for c in ref["cls"] if c == 0)
    nn = len(ref["cls"]) - ne
    return f"参考：{ne} 个鸡蛋" + (f" + {nn} 个非鸡蛋" if nn else "") + f"（{ref.get('saved', '')}）"


# =========================================================================== 2D detector
def _np(x):
    import numpy as np
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    return np.asarray(x)


class YoloEDetector:
    """Lazy YOLOE wrapper. detect() returns [{'cls': 0 egg | 1 not-egg, 'name', 'conf', 'box', 'mask'(HxW bool)}]."""

    def __init__(self, log=print):
        self.log = log
        self.lock = threading.Lock()
        self.model = None
        self.key = None
        self.device = None                  # None = ultralytics default (CUDA when available)
        self.info = {}

    def _load(self, path, mode, prompts):
        key = (path, mode, tuple(prompts) if mode == "text" else ())
        if self.model is not None and self.key == key:
            return self.model
        try:
            _cv2()
            from ultralytics import YOLOE
        except Exception as exc:
            raise YoloUnavailable(f"ultralytics 不可用：{exc}（控制台的 python3 里没有 ultralytics ≥ 8.4）")
        if not path or not os.path.isfile(path):
            raise YoloUnavailable(f"找不到模型文件 {path}")
        t0 = time.time()
        m = YOLOE(path)
        if mode == "text":
            enc_dir = os.path.dirname(path)
            enc = text_encoder_name(path)
            if not os.path.isfile(os.path.join(enc_dir, enc)):
                raise YoloUnavailable(f"文字提示需要 {enc}，请放进 {enc_dir}（否则会尝试联网下载）；视觉提示不需要它")
            with _CWD_LOCK:                 # ultralytics looks for the text encoder in the working directory
                old = os.getcwd()
                os.chdir(enc_dir)
                try:
                    m.set_classes(list(prompts))
                finally:
                    os.chdir(old)
        self.model, self.key = m, key
        self.info = {"path": path, "sha256": file_sha256(path), "mode": mode, "load_s": round(time.time() - t0, 2)}
        self.log(f"[OK] [V3.47 yolo] 模型 {path}（sha256 {self.info['sha256'][:12]}…，{mode}，加载 "
                 f"{self.info['load_s']:.1f} s）")
        return m

    def _predict(self, m, bgr, conf, **kw):
        try:
            return m.predict(bgr, conf=conf, imgsz=IMGSZ, verbose=False,
                             **({"device": self.device} if self.device else {}), **kw)
        except RuntimeError as exc:
            if "cuda" not in str(exc).lower() or self.device == "cpu":
                raise
            self.device = "cpu"
            self.log(f"[WARN] [V3.47 yolo] GPU 出错（{str(exc)[:120]}）→ 改用 CPU")
            return m.predict(bgr, conf=conf, imgsz=IMGSZ, verbose=False, device="cpu", **kw)

    def detect(self, rgb, mode, model_path, ref=None, prompts=None, conf=CONF_MIN):
        import numpy as np
        prompts = list(prompts or parse_prompts(DEFAULT_TEXT_PROMPTS))
        if mode == "vp" and not ref:
            raise YoloUnavailable("视觉提示还没有参考图：先「框选参考」并「保存参考」（或改用文字提示）")
        with self.lock:
            m = self._load(model_path, mode, prompts)
            bgr = np.ascontiguousarray(rgb[..., ::-1])       # D435i stream is RGB, ultralytics wants BGR
            if mode == "vp":
                from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
                vp = {"bboxes": np.array(ref["boxes"], dtype=float), "cls": np.array(ref["cls"], dtype=int)}
                res = self._predict(m, bgr, conf, refer_image=ref["image"], visual_prompts=vp,
                                    predictor=YOLOEVPSegPredictor)
            else:
                res = self._predict(m, bgr, conf)
        r = res[0]
        return parse_result(r, rgb.shape[:2], mode, prompts)


def parse_result(r, hw, mode, prompts):
    import numpy as np
    h, w = hw
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = _np(r.boxes.xyxy).reshape(-1, 4)
    confs = _np(r.boxes.conf).reshape(-1)
    clss = _np(r.boxes.cls).reshape(-1).astype(int)
    polys = list(r.masks.xy) if getattr(r, "masks", None) is not None and r.masks is not None else [None] * len(xyxy)
    cv2 = _cv2()
    out = []
    for k in range(len(xyxy)):
        b = [float(v) for v in xyxy[k]]
        mask = np.zeros((h, w), dtype=np.uint8)
        poly = polys[k] if k < len(polys) else None
        if poly is not None and len(poly) >= 3:
            cv2.fillPoly(mask, [np.round(np.asarray(poly, dtype=float)).astype(np.int32).reshape(-1, 1, 2)], 1)
        if not mask.any():                 # no polygon: ellipse inside the box
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            cv2.ellipse(mask, (int(round(cx)), int(round(cy))), (max(1, int((b[2] - b[0]) / 2)), max(1, int((b[3] - b[1]) / 2))),
                        0, 0, 360, 1, -1)
        c = int(clss[k])
        if mode == "vp":
            cls, name = (0, "鸡蛋") if c == 0 else (1, "非鸡蛋")
        else:
            cls, name = 0, (prompts[c] if 0 <= c < len(prompts) else str(c))
        out.append({"cls": cls, "name": name, "conf": float(confs[k]), "box": b, "mask": mask.astype(bool)})
    out.sort(key=lambda d: -d["conf"])
    return out


# =========================================================================== 3D
def point_in_poly(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-15) + x0:
            inside = not inside
    return inside


def _pix_to_base(uu, vv, z, K, T):
    import numpy as np
    pc = np.stack([(uu - K["ppx"]) * z / K["fx"], (vv - K["ppy"]) * z / K["fy"], z], axis=1)
    return pc @ T[:3, :3].T + T[:3, 3]


# --------------------------------------------------------------------------- V3.47.1 table plane
def fit_plane(P, iters=4):
    """Robust plane through base_link points: (n (up), d, rms_m, n_inliers) with n . p = d, or None."""
    import numpy as np
    Q = P
    n = d = None
    for _ in range(iters):
        if len(Q) < PLANE_MIN_POINTS:
            return None
        c = Q.mean(axis=0)
        _u, _s, vt = np.linalg.svd(Q - c, full_matrices=False)
        n = vt[2] if vt[2][2] >= 0 else -vt[2]
        d = float(n @ c)
        r = P @ n - d
        mad = float(np.median(np.abs(r - np.median(r)))) * 1.4826
        Q = P[np.abs(r) <= max(0.002, 3.0 * mad)]
    if n is None or len(Q) < PLANE_MIN_POINTS or n[2] < math.cos(math.radians(PLANE_MAX_TILT_DEG)):
        return None
    rms = float(np.sqrt(np.mean((Q @ n - d) ** 2)))
    return n, d, rms, int(len(Q))


def local_plane(depth_m, K, T, mask, other, z_cam):
    """Table plane in a band PLANE_RING_M (3-6 cm) around the object (other objects, invalid depth excluded)."""
    import numpy as np
    cv2 = _cv2()
    f = 0.5 * (K["fx"] + K["fy"])
    r_in = max(6, int(round(f * PLANE_RING_M[0] / z_cam)))
    r_out = max(r_in + 6, int(round(f * PLANE_RING_M[1] / z_cam)))
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    band = (dist >= r_in) & (dist <= r_out) & (depth_m > 0)
    if other is not None and other.any():
        k = 2 * r_in + 1
        band &= ~cv2.dilate(other.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
    vv, uu = np.nonzero(band)
    if len(vv) < PLANE_MIN_POINTS:
        return None, int(len(vv))
    if len(vv) > 6000:
        sel = np.random.default_rng(0).choice(len(vv), 6000, replace=False)
        vv, uu = vv[sel], uu[sel]
    P = _pix_to_base(uu.astype(float), vv.astype(float), depth_m[vv, uu].astype(float), K, T)
    return fit_plane(P), int(len(vv))


# --------------------------------------------------------------------------- V3.47.1 silhouette pose
_SPHERE = None


def _unit_sphere(n=700):
    import numpy as np
    global _SPHERE
    if _SPHERE is None or len(_SPHERE) != n:
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * i / n)
        th = math.pi * (1 + 5 ** 0.5) * i
        _SPHERE = np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], axis=1)
    return _SPHERE


class SilhouetteFit:
    """Egg (spheroid a = L/2 along the axis, b = W/2) resting on the plane n . p = d: pose (s, t, yaw) in the plane,
    standing or lying; score = IoU of the projected silhouette with the YOLO mask."""

    def __init__(self, mask, K, T_base_cam, plane, size_mm):
        import numpy as np
        self.np = np
        self.cv2 = _cv2()
        self.K = K
        self.Tcb = np.linalg.inv(np.asarray(T_base_cam, float))
        self.n, self.d = np.asarray(plane[0], float), float(plane[1])
        ex = np.array([1.0, 0.0, 0.0]) - self.n[0] * self.n
        self.e1 = ex / np.linalg.norm(ex)
        self.e2 = np.cross(self.n, self.e1)
        self.o = self.n * self.d
        self.a, self.b = size_mm[0] / 2000.0, size_mm[1] / 2000.0
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        pad = 60
        self.x0, self.y0 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
        self.x1, self.y1 = min(w, int(xs.max()) + pad + 1), min(h, int(ys.max()) + pad + 1)
        self.m = mask[self.y0:self.y1, self.x0:self.x1]
        self.m_area = float(self.m.sum())
        self.cxy = (float(xs.mean()), float(ys.mean()))
        self.evals = 0

    def centre(self, s, t, standing, scale=1.0):
        h = (self.a if standing else self.b) * scale
        return self.o + s * self.e1 + t * self.e2 + h * self.n

    def axis(self, yaw, standing):
        if standing:
            return self.n.copy()
        return math.cos(yaw) * self.e1 + math.sin(yaw) * self.e2

    def silhouette(self, s, t, yaw, standing, scale=1.0):
        np = self.np
        d = self.axis(yaw, standing)
        u = np.cross(d, self.n) if not standing else self.e1
        u = u / np.linalg.norm(u)
        v = np.cross(d, u)
        S = _unit_sphere()
        a, b = self.a * scale, self.b * scale
        P = (S[:, 0:1] * b) * u + (S[:, 1:2] * b) * v + (S[:, 2:3] * a) * d + self.centre(s, t, standing, scale)
        Pc = P @ self.Tcb[:3, :3].T + self.Tcb[:3, 3]
        if np.any(Pc[:, 2] <= 0.05):
            return None
        uv = np.stack([Pc[:, 0] / Pc[:, 2] * self.K["fx"] + self.K["ppx"] - self.x0,
                       Pc[:, 1] / Pc[:, 2] * self.K["fy"] + self.K["ppy"] - self.y0], axis=1)
        hull = self.cv2.convexHull(np.round(uv * 4).astype(np.int32))
        img = np.zeros(self.m.shape, np.uint8)
        self.cv2.fillConvexPoly(img, hull, 1, lineType=self.cv2.LINE_8, shift=2)
        return img.astype(bool)

    def iou(self, s, t, yaw, standing, scale=1.0):
        self.evals += 1
        sil = self.silhouette(s, t, yaw, standing, scale)
        if sil is None:
            return 0.0
        inter = float((sil & self.m).sum())
        return inter / max(1.0, float(sil.sum()) + self.m_area - inter)

    def init_st(self, standing):
        """Ray through the mask centroid hits the plane lifted to the egg centre height."""
        np = self.np
        u, v = self.cxy
        Tbc = np.linalg.inv(self.Tcb)
        dr = Tbc[:3, :3] @ np.array([(u - self.K["ppx"]) / self.K["fx"], (v - self.K["ppy"]) / self.K["fy"], 1.0])
        o = Tbc[:3, 3]
        h = self.d + (self.a if standing else self.b)
        lam = (h - self.n @ o) / (self.n @ dr)
        p = o + lam * dr - self.o
        return float(p @ self.e1), float(p @ self.e2)

    def _refine(self, standing):
        s, t = self.init_st(standing)
        yaws = [0.0] if standing else [math.radians(k) for k in range(0, 180, 15)]
        sc, yw = max((self.iou(s, t, y, standing), y) for y in yaws)
        step = [0.004, 0.004, math.radians(8.0)]
        while step[0] > 0.00025:
            moved = False
            for k in range(2 if standing else 3):
                for sg in (1, -1):
                    x = [s, t, yw]
                    x[k] += sg * step[k]
                    v = self.iou(x[0], x[1], x[2], standing)
                    if v > sc + 1e-9:
                        sc, s, t, yw = v, x[0], x[1], x[2]
                        moved = True
            if not moved:
                step = [step[0] / 2, step[1] / 2, step[2] / 2]
        return sc, s, t, yw

    def solve(self, standing_hint=False):
        """Lying first (an egg on a table lies); standing (egg in a holder) only if clearly better, or when the
        measured height says so (standing_hint) and its silhouette is not clearly worse."""
        lie = self._refine(False)
        std = self._refine(True)
        standing = std[0] > lie[0] + STANDING_MARGIN or (standing_hint and std[0] > lie[0] - 0.08)
        sc, s, t, yw = std if standing else lie
        scale = 1.0
        # centroid alignment: a YOLO mask that is uniformly too big / small keeps its centroid; matching IoU alone
        # would pull a fixed-size egg towards / away from the camera
        s, t = self._centroid_align(s, t, yw, standing, scale)
        shape, bleed = self.shape_iou(s, t, yw, standing, scale)
        return {"iou": shape, "iou_raw": self.iou(s, t, yw, standing, scale), "bleed_px": bleed, "s": s, "t": t,
                "yaw": yw, "standing": standing, "scale": scale,
                "centre": self.centre(s, t, standing, scale), "axis": self.axis(yw, standing)}

    def _centroid(self, s, t, yw, standing, scale):
        np = self.np
        sil = self.silhouette(s, t, yw, standing, scale)
        if sil is None or not sil.any():
            return None
        ys, xs = np.nonzero(sil)
        return np.array([xs.mean(), ys.mean()])

    def _centroid_align(self, s, t, yw, standing, scale, iters=4, h=0.001):
        np = self.np
        target = np.array([self.cxy[0] - self.x0, self.cxy[1] - self.y0])
        for _ in range(iters):
            c0 = self._centroid(s, t, yw, standing, scale)
            cs = self._centroid(s + h, t, yw, standing, scale)
            ct = self._centroid(s, t + h, yw, standing, scale)
            if c0 is None or cs is None or ct is None:
                break
            J = np.stack([(cs - c0) / h, (ct - c0) / h], axis=1)
            try:
                ds, dt = np.linalg.solve(J, target - c0)
            except np.linalg.LinAlgError:
                break
            ds, dt = float(np.clip(ds, -0.01, 0.01)), float(np.clip(dt, -0.01, 0.01))
            s, t = s + ds, t + dt
            if abs(ds) + abs(dt) < 5e-5:
                break
        return s, t

    def shape_iou(self, s, t, yw, standing, scale):
        """IoU after growing / shrinking the silhouette to the mask's area (k px): shape match independent of a
        uniformly loose / tight YOLO mask.  Returns (iou, k_px)."""
        np, cv2 = self.np, self.cv2
        sil = self.silhouette(s, t, yw, standing, scale)
        if sil is None or not sil.any():
            return 0.0, 0.0
        cs, _ = cv2.findContours(sil.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        per = max(1.0, float(sum(cv2.arcLength(c, True) for c in cs)))
        k = (self.m_area - float(sil.sum())) / per
        kk = int(round(abs(k)))
        if kk:
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kk + 1, 2 * kk + 1))
            sil = (cv2.dilate if k > 0 else cv2.erode)(sil.astype(np.uint8), ker).astype(bool)
        inter = float((sil & self.m).sum())
        return inter / max(1.0, float(sil.sum()) + self.m_area - inter), round(k, 1)


def _quat_z_to(d):
    import NERO_V308_Camera_Perception as P8
    return P8._quat_from_z_to(d)


def locate(dets2d, depth_m, K, T_base_cam, table_z=None, fence_poly=None, size_mm=EGG_SIZE_MM, calib_note=None):
    """V3.47.1: 2D detections + aligned depth (m) -> 3D results and diagnostics.
    The egg position does NOT use depth on the egg: table plane from a 3-6 cm band around the object + the YOLO
    silhouette + the egg size.  Depth on the egg (if any) only gives a displayed height and rejects flat things.
    calib_note: text when the hand-eye calibration is not verified (every result then not trusted)."""
    import numpy as np
    cv2 = _cv2()
    T = np.asarray(T_base_cam, dtype=float)
    h, w = depth_m.shape
    out, planes = [], []
    union = np.zeros((h, w), bool)
    for d in dets2d:
        union |= d["mask"]
    for k, d in enumerate(dets2d):
        x0, y0, x1, y1 = d["box"]
        t_start = time.time()
        res = {"yolo": {"conf": round(d["conf"], 3), "box": [round(v) for v in d["box"]], "cls": d["cls"],
                        "name": d["name"]}, "reasons": [], "notes": [], "status": "rejected", "trusted": False,
               "confidence": 0.0}
        out.append(res)
        if d["cls"] != 0:
            res["reasons"].append("参考里的非鸡蛋")
            continue
        m = d["mask"]
        area = int(m.sum())
        if area < 30:
            res["reasons"].append("轮廓太小")
            continue
        m8 = m.astype(np.uint8)
        r_er = max(ERODE_MIN_PX, int(round(ERODE_FRAC * math.sqrt(area))))
        er = cv2.erode(m8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_er + 1, 2 * r_er + 1))).astype(bool)
        # distance scale for the band: depth around the object (the egg itself may have none)
        ring0 = cv2.dilate(m8, np.ones((41, 41), np.uint8)).astype(bool) & ~m & (depth_m > 0)
        zc = depth_m[ring0]
        z_cam = float(np.median(zc)) if zc.size >= 20 else 0.8
        other = union & ~m
        plane, n_band = local_plane(depth_m, K, T, m, other, z_cam)
        psrc = "局部桌面平面"
        if plane is None:
            if table_z is None:
                res["reasons"].append(f"测不到桌面平面（周围 3–6 cm 有效深度点 {n_band}）")
                continue
            plane, psrc = (np.array([0.0, 0.0, 1.0]), float(table_z), None, 0), "安全设置 z_table"
        planes.append(plane)
        # depth on the egg: coverage + measured height above the plane (display / flat check / standing hint only)
        vv, uu = np.nonzero(er)
        dz = depth_m[vv, uu]
        cov = float((dz > 0).mean()) if len(dz) else 0.0
        hmeas = None
        if cov >= DEPTH_COVER_MIN and (dz > 0).sum() >= 20:
            ok = dz > 0
            P = _pix_to_base(uu[ok].astype(float), vv[ok].astype(float), dz[ok].astype(float), K, T)
            hmeas = float(np.percentile(P @ plane[0] - plane[1], 95)) * 1000.0
        hint = hmeas is not None and hmeas >= size_mm[1] + STANDING_HINT_MM
        fit = SilhouetteFit(m, K, T, plane, size_mm).solve(standing_hint=hint)
        c, ax = fit["centre"], fit["axis"]
        if ax[2] < 0:
            ax = -ax
        xyz = [float(v) for v in c]
        res["ground_z_m"] = round(float((plane[1] - plane[0][0] * xyz[0] - plane[0][1] * xyz[1]) / plane[0][2]), 4)
        edge = x0 <= EDGE_PX or y0 <= EDGE_PX or x1 >= w - 1 - EDGE_PX or y1 >= h - 1 - EDGE_PX
        res.update(xyz_m=[round(v, 5) for v in xyz], axis_world=[round(float(v), 5) for v in ax],
                   quat_xyzw=[round(v, 6) for v in _quat_z_to(ax)], iou=round(fit["iou"], 3),
                   iou_raw=round(fit["iou_raw"], 3), mask_bleed_px=fit["bleed_px"],
                   pose="竖立" if fit["standing"] else "平躺", size_mm=[round(size_mm[0], 1), round(size_mm[1], 1)],
                   height_mm=None if hmeas is None else round(hmeas, 1), depth_cover=round(cov, 2),
                   plane_src=psrc, plane_rms_mm=None if plane[2] is None else round(plane[2] * 1000.0, 2),
                   edge=bool(edge), fit_ms=round((time.time() - t_start) * 1000.0))
        if hmeas is not None and hmeas < FLAT_MAX_MM:
            res["reasons"].append(f"扁平：只高出桌面 {hmeas:.0f} mm（< {FLAT_MAX_MM:.0f}）")
            continue
        if fit["iou"] < IOU_REJECT:
            res["reasons"].append(f"轮廓不像鸡蛋（重合度 {fit['iou']:.2f}）")
            continue
        warn = []
        if fit["iou"] < IOU_TRUST:
            warn.append(f"轮廓和鸡蛋形状对不齐（重合度 {fit['iou']:.2f}）")
        if abs(fit["bleed_px"]) > BLEED_MAX_PX:
            warn.append(f"YOLO 轮廓比 {size_mm[0]:.0f}×{size_mm[1]:.0f} mm 的鸡蛋{'大' if fit['bleed_px'] > 0 else '小'} "
                        f"{abs(fit['bleed_px']):.0f} px（尺寸填对了吗？）")
        if edge:
            warn.append("贴画面边缘（被切掉一部分）")
        if d["conf"] < 0.5:
            warn.append(f"YOLO 置信度低 {d['conf']:.2f}")
        if fence_poly and not point_in_poly(xyz[0], xyz[1], fence_poly):
            warn.append("在安全区外")
        if psrc != "局部桌面平面":
            warn.append("周围桌面没有深度，用的是安全设置 z_table")
        if calib_note:
            warn.append(calib_note)
        if hmeas is None:
            res["notes"].append(f"鸡蛋上深度不足（{cov * 100:.0f}%），按轮廓 + 桌面定位")
        res["reasons"] = warn
        res["status"] = "untrusted" if warn else "egg"
        res["trusted"] = not warn
        res["confidence"] = round(float(d["conf"]) * fit["iou"] * (1.0 if not warn else 0.7), 3)
    order = {"egg": 0, "untrusted": 1, "rejected": 2}
    out.sort(key=lambda r: (order[r["status"]], -r["yolo"]["conf"]))
    n = 0
    for r in out:
        if r["status"] != "rejected":
            n += 1
            r["det_id"] = f"egg_{n}"
        else:
            r["det_id"] = "✗"
    diag = {"n_2d": len(dets2d), "n_egg": sum(1 for r in out if r["status"] == "egg"),
            "n_untrusted": sum(1 for r in out if r["status"] == "untrusted"),
            "n_rejected": sum(1 for r in out if r["status"] == "rejected")}
    gz = [r["ground_z_m"] for r in out if r.get("plane_src") == "局部桌面平面" and r.get("ground_z_m") is not None]
    if gz:
        diag["ground_z_m"] = round(float(np.median(gz)), 4)
        if table_z is not None:
            diag["ground_minus_z_table_mm"] = round((diag["ground_z_m"] - float(table_z)) * 1000.0, 1)
    loc = [p[2] for p in planes if p[2] is not None]
    if loc:
        diag["plane_rms_mm"] = round(float(np.median(loc)) * 1000.0, 2)
    return out, diag


def fuse_depth(depths):
    import NERO_V308_Camera_Perception as P8
    return P8.temporal_fuse(depths) if len(depths) > 1 else depths[0]


def save_sample(rgb, dets2d, results, meta, folder=None):
    """Optional: colour image + 2D boxes / labels + 3D results, for a later fine-tuning set."""
    folder = folder or SAMPLES_DIR
    cv2 = _cv2()
    os.makedirs(folder, exist_ok=True)
    stem = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    cv2.imwrite(os.path.join(folder, stem + ".jpg"), rgb[..., ::-1])
    rec = {"version": VERSION, "image": stem + ".jpg", "size": [int(rgb.shape[1]), int(rgb.shape[0])], "meta": meta,
           "objects": [{"box": r["yolo"]["box"], "yolo_conf": r["yolo"]["conf"], "yolo_cls": r["yolo"]["cls"],
                        "status": r["status"], "reasons": r["reasons"], "xyz_m": r.get("xyz_m"),
                        "height_mm": r.get("height_mm")} for r in results]}
    with open(os.path.join(folder, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    return os.path.join(folder, stem + ".jpg")
