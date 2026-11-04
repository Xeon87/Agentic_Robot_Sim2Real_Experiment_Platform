"""NERO V3.79 camera lab - the Camera tab's 「相机高级测试」 window (V3.79 CAMERA LAB).

Pages (all on the Camera tab's live D435i stream, no second pipeline):
  ① 相机参数  lock exposure / gain / white balance / anti-flicker; saved to NERO_CALIB_DIR/d435i_profile.json and
              applied on every Camera-tab connect while locked (apply_saved_profile; depth preset stays the tab's)
  ② 深度噪声  the V3.78 filter sets on a still flat target (valid %, temporal std, flatness)
  ③ 定位精度  ChArUco board = frame (z up, table z = 0), egg on an inner corner, the Camera tab's YOLOE + YP.locate
  ④ 采集标注  photos into <root>/<class>/ with YOLOE auto labels (YOLO segmentation txt) + box-level review:
              click = select, drag inside = move, drag a corner = resize, drag on empty = new box, Delete = remove;
              the outline inside an edited box is recomputed (the image's YOLOE outline clipped to the box, else
              GrabCut inside the box, else the box itself)

Dataset layout (V3.80 trains on it):
  <root>/classes.json                      {"classes": [...]}  global class ids, append-only
  <root>/<class>/images/{train,val}/<stem>.jpg
  <root>/<class>/labels/{train,val}/<stem>.txt   "<id> x1 y1 x2 y2 ..." normalised polygon per object
  <root>/<class>/meta/<stem>.json          boxes / outlines in pixels, conf, source, reviewed, YOLOE originals
  <root>/<class>/depth/<stem>.png          aligned depth, uint16 mm (optional)
  every 5th image of a class goes to val (about 8:2)

Writes only NERO_CALIB_DIR (profile, results, lab settings) and the chosen data folder; never the release folder.
"""
import json
import math
import os
import re
import shutil
import threading
import time

import numpy as np

import NERO_V378_D435i_Bench as B378

VERSION = "V3.79"
MARKER = "V3.79 CAMERA LAB"
CALIB_DIR = B378.CALIB_DIR
LAB_SETTINGS = os.path.join(CALIB_DIR, "camera_lab.json")
DEFAULT_ROOT = B378.DATA_DIR
VAL_EVERY = 5                       # every 5th image -> val
HANDLE_PX = 9                       # corner grab radius (display pixels)
MIN_BOX_PX = 6
POWER_CHOICES = [("50 Hz", 1), ("60 Hz", 2)]


# ============================================================================================ settings
def load_settings(path=None):
    d = {"root": DEFAULT_ROOT, "prompts": "egg", "conf": 0.25, "save_depth": True, "burst_n": 10, "burst_dt": 1.0,
         "acc_prompt": "egg", "standing": False, "trials": 10, "noise_frames": 30}
    try:
        with open(path or os.environ.get("NERO_V379_LAB_SETTINGS") or LAB_SETTINGS, encoding="utf-8") as f:
            d.update(json.load(f))
    except Exception:
        pass
    return d


def save_settings(d, path=None):
    p = path or os.environ.get("NERO_V379_LAB_SETTINGS") or LAB_SETTINGS
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(dict(d, version=VERSION), f, ensure_ascii=False, indent=1)
    os.replace(tmp, p)
    return p


def apply_saved_profile(device, rs, log=print, path=None):
    """Camera-tab connect: apply the lab's profile when it is locked; the depth preset stays the tab's own choice.
    -> "locked" | "none" """
    prof = B378.load_profile(path)
    if not prof.get("locked"):
        return "none"
    p = json.loads(json.dumps(prof))
    p["depth"].pop("visual_preset", None)
    rows = B378.apply_profile(device, p, rs, log=lambda s: None)
    bad = [r for r in rows if r[3] is None]
    log(f"[{'OK' if not bad else 'WARN'}] [V3.79 camera lab] 已套用锁定的相机参数（曝光 "
        f"{prof['color'].get('exposure')} · 增益 {prof['color'].get('gain')} · 白平衡 {prof['color'].get('white_balance')}"
        f" · 防闪烁 {B378.POWER_LINE.get(int(prof['color'].get('power_line_frequency', 0)), '?')}）"
        + (f"；没套上：{', '.join(r[0] + '.' + r[1] for r in bad)}" if bad else ""))
    return "locked"


# ============================================================================================ geometry
def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def norm_box(b, wh=None):
    x0, x1 = sorted((float(b[0]), float(b[2])))
    y0, y1 = sorted((float(b[1]), float(b[3])))
    if wh:
        x0, x1 = max(0.0, x0), min(float(wh[0]), x1)
        y0, y1 = max(0.0, y0), min(float(wh[1]), y1)
    return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]


def box_poly(b):
    return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]


def mask_to_poly(mask, eps_px=1.5, max_pts=200):
    """largest outer contour of a bool mask -> [[x, y], ...] (pixels) or None"""
    cv2 = B378._cv2()
    m = np.ascontiguousarray(mask.astype(np.uint8))
    cs, _h = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    if cv2.contourArea(c) < 4:
        return None
    eps = eps_px
    a = cv2.approxPolyDP(c, eps, True)
    while len(a) > max_pts:
        eps *= 1.5
        a = cv2.approxPolyDP(c, eps, True)
    if len(a) < 3:
        return None
    return [[float(p[0][0]), float(p[0][1])] for p in a]


def poly_mask(poly, hw):
    cv2 = B378._cv2()
    m = np.zeros(hw, np.uint8)
    if poly and len(poly) >= 3:
        cv2.fillPoly(m, [np.round(np.asarray(poly, float)).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def clip_poly(poly, box, hw):
    m = poly_mask(poly, hw)
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    keep = np.zeros(hw, bool)
    keep[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    return mask_to_poly(m & keep)


def grabcut_poly(rgb, box, iters=4):
    cv2 = B378._cv2()
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in norm_box(box, (w, h))]
    if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
        return None
    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(np.ascontiguousarray(rgb[..., ::-1]), mask, (x0, y0, x1 - x0, y1 - y0), bgd, fgd, iters,
                    cv2.GC_INIT_WITH_RECT)
    except Exception:
        return None
    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    if fg.sum() < 0.10 * (x1 - x0) * (y1 - y0):
        return None
    return mask_to_poly(fg)


def poly_area(poly):
    if not poly or len(poly) < 3:
        return 0.0
    a = np.asarray(poly, float)
    return 0.5 * abs(float(np.dot(a[:, 0], np.roll(a[:, 1], 1)) - np.dot(a[:, 1], np.roll(a[:, 0], 1))))


def recompute_outline(rgb, box, originals):
    """outline for an edited / new box -> (poly, how): the image's YOLOE outline clipped to the box (box IoU >= 0.5),
    else GrabCut inside the box, else the box itself"""
    h, w = rgb.shape[:2]
    box = norm_box(box, (w, h))
    area = max(1.0, (box[2] - box[0]) * (box[3] - box[1]))
    best = max(originals or [], key=lambda o: box_iou(box, o["box"]), default=None)
    if best is not None and box_iou(box, best["box"]) >= 0.5 and best.get("poly"):
        p = clip_poly(best["poly"], box, (h, w))
        if p and poly_area(p) >= 0.15 * area:
            return p, "yoloe"
    p = grabcut_poly(rgb, box)
    if p:
        return p, "grabcut"
    return box_poly(box), "box"


# ============================================================================================ dataset
def safe_class(name):
    s = re.sub(r'[\\/:*?"<>|\s]+', "_", str(name or "").strip()).strip("._")
    return s or "object"


def dets_to_anns(dets, prompts):
    """YOLOE detections -> annotations with the user's class names (pf names mapped onto the matching prompt)"""
    import NERO_V347_Yolo_Perception as YP
    out = []
    for d in dets:
        name = str(d.get("name", ""))
        cls = name if name in prompts else next((p for p in prompts if YP.match_object_name(name, [p])), None)
        if cls is None:
            continue
        poly = mask_to_poly(d["mask"]) if d.get("mask") is not None else None
        out.append({"cls": cls, "box": norm_box(d["box"]), "poly": poly or box_poly(norm_box(d["box"])),
                    "conf": round(float(d.get("conf", 0.0)), 3), "source": "auto"})
    return out


def label_lines(anns, cid_of, wh):
    w, h = float(wh[0]), float(wh[1])
    lines = []
    for a in anns:
        poly = a.get("poly") if a.get("poly") and len(a["poly"]) >= 3 else box_poly(a["box"])
        pts = " ".join(f"{min(max(x / w, 0.0), 1.0):.6f} {min(max(y / h, 0.0), 1.0):.6f}" for x, y in poly)
        lines.append(f"{cid_of(a['cls'])} {pts}")
    return lines


class Dataset:
    def __init__(self, root):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- classes
    def classes(self):
        try:
            with open(os.path.join(self.root, "classes.json"), encoding="utf-8") as f:
                return list(json.load(f).get("classes", []))
        except Exception:
            return []

    def class_id(self, name, add=True):
        cl = self.classes()
        if name in cl:
            return cl.index(name)
        if not add:
            return None
        cl.append(name)
        os.makedirs(self.root, exist_ok=True)
        with open(os.path.join(self.root, "classes.json"), "w", encoding="utf-8") as f:
            json.dump({"version": VERSION, "classes": cl}, f, ensure_ascii=False, indent=1)
        return len(cl) - 1

    def folders(self):
        if not os.path.isdir(self.root):
            return []
        return sorted(d for d in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, d, "meta")))

    def cdir(self, folder):
        return os.path.join(self.root, safe_class(folder))

    # ---------------------------------------------------------------- items
    def _meta_path(self, folder, stem):
        return os.path.join(self.cdir(folder), "meta", stem + ".json")

    def items(self, folder):
        md = os.path.join(self.cdir(folder), "meta")
        out = []
        if os.path.isdir(md):
            for fn in sorted(os.listdir(md)):
                if fn.endswith(".json"):
                    try:
                        with open(os.path.join(md, fn), encoding="utf-8") as f:
                            out.append(json.load(f))
                    except Exception:
                        pass
        return out

    def next_split(self, folder):
        return "val" if len(self.items(folder)) % VAL_EVERY == VAL_EVERY - 1 else "train"

    def add_image(self, folder, rgb, anns, depth=None, extra=None):
        cv2 = B378._cv2()
        with self.lock:
            folder = safe_class(folder)
            d = self.cdir(folder)
            split = self.next_split(folder)
            n = len(self.items(folder))
            stem = f"{folder}_{time.strftime('%Y%m%d_%H%M%S')}_{n:04d}"
            while os.path.exists(self._meta_path(folder, stem)):
                n += 1
                stem = f"{folder}_{time.strftime('%Y%m%d_%H%M%S')}_{n:04d}"
            for sub in (f"images/{split}", f"labels/{split}", "meta"):
                os.makedirs(os.path.join(d, sub), exist_ok=True)
            img = os.path.join(d, "images", split, stem + ".jpg")
            cv2.imwrite(img, np.ascontiguousarray(rgb[..., ::-1]), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            dep = None
            if depth is not None:
                os.makedirs(os.path.join(d, "depth"), exist_ok=True)
                dep = os.path.join("depth", stem + ".png")
                cv2.imwrite(os.path.join(d, dep), np.clip(np.asarray(depth) * 1000.0, 0, 65535).astype(np.uint16))
            meta = dict({"version": VERSION, "stem": stem, "folder": folder, "split": split,
                         "image": os.path.join("images", split, stem + ".jpg"), "depth": dep,
                         "size": [int(rgb.shape[1]), int(rgb.shape[0])], "taken": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "reviewed": False, "anns": anns, "auto": json.loads(json.dumps(anns))}, **(extra or {}))
            self._write(folder, meta)
            return stem

    def _write(self, folder, meta):
        d = self.cdir(folder)
        lab = os.path.join(d, "labels", meta["split"], meta["stem"] + ".txt")
        os.makedirs(os.path.dirname(lab), exist_ok=True)
        lines = label_lines(meta["anns"], lambda c: self.class_id(c), meta["size"])
        with open(lab, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))      # empty file = background image
        tmp = self._meta_path(folder, meta["stem"]) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self._meta_path(folder, meta["stem"]))

    def load(self, folder, stem):
        with open(self._meta_path(folder, stem), encoding="utf-8") as f:
            meta = json.load(f)
        cv2 = B378._cv2()
        bgr = cv2.imread(os.path.join(self.cdir(folder), meta["image"]))
        return (None if bgr is None else np.ascontiguousarray(bgr[..., ::-1])), meta

    def save_anns(self, folder, stem, anns, reviewed=None):
        with self.lock:
            with open(self._meta_path(folder, stem), encoding="utf-8") as f:
                meta = json.load(f)
            meta["anns"] = anns
            if reviewed is not None:
                meta["reviewed"] = bool(reviewed)
            meta["edited"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._write(folder, meta)
            return meta

    def delete_image(self, folder, stem):
        with self.lock:
            with open(self._meta_path(folder, stem), encoding="utf-8") as f:
                meta = json.load(f)
            d = self.cdir(folder)
            for rel in (meta.get("image"), meta.get("depth"), os.path.join("labels", meta["split"], stem + ".txt"),
                        os.path.join("meta", stem + ".json")):
                if rel and os.path.isfile(os.path.join(d, rel)):
                    os.remove(os.path.join(d, rel))

    def stats(self):
        out = {}
        for fo in self.folders():
            it = self.items(fo)
            out[fo] = {"images": len(it), "train": sum(1 for m in it if m.get("split") == "train"),
                       "val": sum(1 for m in it if m.get("split") == "val"),
                       "reviewed": sum(1 for m in it if m.get("reviewed")),
                       "objects": sum(len(m.get("anns", [])) for m in it)}
        return out


# ============================================================================================ box editing
class BoxEditor:
    """box-level review in IMAGE pixels; the canvas converts display <-> image with scale (display px per image px).
    press / drag / release -> the edited annotation index (outline to recompute) or None"""

    def __init__(self, anns, wh, default_cls="object"):
        self.anns = anns
        self.wh = wh
        self.sel = None
        self.default_cls = default_cls
        self._op = None

    def hit(self, x, y, scale=1.0):
        r = HANDLE_PX / max(scale, 1e-6)
        if self.sel is not None and self.sel < len(self.anns):
            b = self.anns[self.sel]["box"]
            for k, (cx, cy) in enumerate(((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3]))):
                if abs(x - cx) <= r and abs(y - cy) <= r:
                    return self.sel, f"corner{k}"
        inside = [i for i, a in enumerate(self.anns)
                  if a["box"][0] <= x <= a["box"][2] and a["box"][1] <= y <= a["box"][3]]
        if inside:
            i = min(inside, key=lambda j: (self.anns[j]["box"][2] - self.anns[j]["box"][0]) *
                    (self.anns[j]["box"][3] - self.anns[j]["box"][1]))       # the smallest box under the cursor
            return i, "move"
        return None, "new"

    def press(self, x, y, scale=1.0):
        i, what = self.hit(x, y, scale)
        if what == "new":
            self.sel = None
            self._op = {"what": "new", "x0": x, "y0": y, "box": [x, y, x, y], "moved": False}
        else:
            self.sel = i
            self._op = {"what": what, "x0": x, "y0": y, "box0": list(self.anns[i]["box"]), "moved": False}
        return self.sel

    def drag(self, x, y):
        op = self._op
        if not op:
            return
        op["moved"] = op["moved"] or abs(x - op["x0"]) + abs(y - op["y0"]) > 1.0
        if op["what"] == "new":
            op["box"] = [op["x0"], op["y0"], x, y]
            return
        b0, a = op["box0"], self.anns[self.sel]
        if op["what"] == "move":
            dx = min(max(x - op["x0"], -b0[0]), self.wh[0] - b0[2])
            dy = min(max(y - op["y0"], -b0[1]), self.wh[1] - b0[3])
            a["box"] = [b0[0] + dx, b0[1] + dy, b0[2] + dx, b0[3] + dy]
        else:
            k = int(op["what"][-1])
            b = list(b0)
            b[0 if k in (0, 3) else 2] = x
            b[1 if k in (0, 1) else 3] = y
            a["box"] = b

    def release(self, x, y):
        if not self._op:
            return None
        self.drag(x, y)
        op, self._op = self._op, None
        if op["what"] == "new":
            b = norm_box(op["box"], self.wh)
            if b[2] - b[0] < MIN_BOX_PX or b[3] - b[1] < MIN_BOX_PX:
                return None
            self.anns.append({"cls": self.default_cls, "box": b, "poly": None, "conf": None, "source": "manual"})
            self.sel = len(self.anns) - 1
            return self.sel
        a = self.anns[self.sel]
        a["box"] = norm_box(a["box"], self.wh)
        if not op["moved"]:
            a["box"] = op["box0"]
            return None
        if a["box"][2] - a["box"][0] < MIN_BOX_PX or a["box"][3] - a["box"][1] < MIN_BOX_PX:
            a["box"] = op["box0"]
            return None
        a["source"] = "edited" if a.get("source") != "manual" else "manual"
        return self.sel

    def delete(self):
        if self.sel is None or self.sel >= len(self.anns):
            return False
        self.anns.pop(self.sel)
        self.sel = None
        return True

    def set_class(self, cls):
        if self.sel is None or self.sel >= len(self.anns):
            return False
        self.anns[self.sel]["cls"] = cls
        return True


# ============================================================================================ measurements
def measure_noise(src, K, frames=30, log=print, sets=None):
    """the V3.78 filter sets through the Camera tab's reader thread -> rows"""
    w, h = int(K["width"]), int(K["height"])
    roi = [w // 2 - 80, h // 2 - 80, w // 2 + 80, h // 2 + 80]
    rows = []
    for name, fcfg, med in sets or B378.FILTER_SETS:
        cfg = dict(B378.load_profile()["filters"], **fcfg)
        on = any(cfg.get(k) for k in ("spatial", "temporal", "hole_filling"))
        make = (lambda rs, c=cfg: B378.FilterChain(rs, c).process) if on else None
        depths = src.collect_depth(make, 10 + int(frames) * int(med))[10:]       # first 10 = temporal warm-up
        stack = [B378.median_stack(depths[i:i + med]) for i in range(0, len(depths), med)] if med > 1 else depths
        rep = B378.noise_report(stack, K, roi)
        rows.append(dict(setting=name, frames=len(stack), **rep))
        log(f"[INFO] [V3.79 noise] {name}: 有效 {rep['valid_pct']}% · 抖动 {rep['temporal_std_mm']} mm · "
            f"平整度 {rep['flatness_rms_mm']} mm · 距离 {rep['mean_dist_mm']} mm")
    return rows, roi


def accuracy_trials(grab, detect, K, T, corners, exp_z, trials=10, settings=(("滤波 + 5 帧中值", 5), ("滤波 1 帧", 1))):
    """grab(n) -> (rgb, [depth]); detect(rgb) -> dets -> (rows, summary); same maths as the V3.78 bench"""
    import NERO_V347_Yolo_Perception as YP
    rows = []
    for name, med in settings:
        for i in range(int(trials)):
            rgb, depths = grab(med)
            if rgb is None or not depths:
                rows.append({"setting": name, "trial": i, "detected": False, "why": "no frame"})
                continue
            dets = detect(rgb)
            d_m = B378.median_stack(depths) if med > 1 else depths[-1]
            res, _diag = YP.locate(dets, d_m, K, T, table_z=0.0)
            ok = [r for r in res if r.get("xyz_m") is not None and r.get("status") != "rejected"]
            if not ok:
                rows.append({"setting": name, "trial": i, "detected": False, "why": f"{len(dets)} det"})
                continue
            best = min(ok, key=lambda r: B378.accuracy_row(r["xyz_m"], corners, exp_z)["xy_err_mm"])
            rows.append(dict({"setting": name, "trial": i, "detected": True, "status": best.get("status"),
                              "x_mm": round(best["xyz_m"][0] * 1000, 2), "y_mm": round(best["xyz_m"][1] * 1000, 2),
                              "z_mm": round(best["xyz_m"][2] * 1000, 2)},
                             **B378.accuracy_row(best["xyz_m"], corners, exp_z)))
    summ = {}
    for name, _m in settings:
        rr = [r for r in rows if r["setting"] == name]
        summ[name] = {"detected_pct": round(100.0 * sum(1 for r in rr if r.get("detected")) / max(1, len(rr)), 1),
                      "err_3d": B378.summarize(rr), "xy": B378.summarize(rr, "xy_err_mm"),
                      "dz": B378.summarize(rr, "dz_mm")}
    return rows, summ


def corner_pixels(T, K, corners):
    Tc = np.linalg.inv(T)
    out = {}
    for i, (x, y) in corners.items():
        pc = Tc @ np.array([x, y, 0.0, 1.0])
        if pc[2] > 1e-6:
            out[i] = (K["fx"] * pc[0] / pc[2] + K["ppx"], K["fy"] * pc[1] / pc[2] + K["ppy"])
    return out


def draw_overlay(rgb, anns=None, sel=None, points=None, scale=1.0, colours=None):
    """display image (RGB, already scaled by `scale`) with boxes / outlines / numbered points"""
    cv2 = B378._cv2()
    img = np.ascontiguousarray(rgb.copy())
    pal = colours or [(255, 64, 64), (64, 160, 255), (60, 200, 60), (255, 170, 0), (200, 80, 255), (0, 200, 200)]
    names = []
    for k, a in enumerate(anns or []):
        if a["cls"] not in names:
            names.append(a["cls"])
        c = pal[names.index(a["cls"]) % len(pal)]
        b = [int(round(v * scale)) for v in a["box"]]
        if a.get("poly"):
            pts = np.round(np.asarray(a["poly"], float) * scale).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, c, 1)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 255, 255) if k == sel else c, 3 if k == sel else 1)
        if k == sel:
            for (x, y) in ((b[0], b[1]), (b[2], b[1]), (b[2], b[3]), (b[0], b[3])):
                cv2.rectangle(img, (x - 4, y - 4), (x + 4, y + 4), (255, 255, 255), -1)
        tag = f"#{k} {a['cls']}" + (f" {a['conf']:.2f}" if a.get("conf") is not None else f" {a.get('source', '')}")
        cv2.putText(img, tag, (b[0] + 2, max(12, b[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(img, tag, (b[0] + 2, max(12, b[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    for i, (u, v) in (points or {}).items():
        p = (int(round(u * scale)), int(round(v * scale)))
        cv2.circle(img, p, 4, (0, 90, 255), -1)
        cv2.putText(img, str(i), (p[0] + 5, p[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 90, 255), 1)
    return img


def fit_scale(wh, box_wh):
    return min(box_wh[0] / float(wh[0]), box_wh[1] / float(wh[1]))


def resize(rgb, scale):
    cv2 = B378._cv2()
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


# ============================================================================================ window
_OPEN = {}


def open_lab(tab):
    """「相机高级测试」: one window per Camera tab (raise it when already open)"""
    lab = _OPEN.get(id(tab))
    if lab is not None and lab.alive:
        try:
            lab.win.lift()
        except Exception:
            pass
        return lab
    lab = CameraLab(tab)
    _OPEN[id(tab)] = lab
    return lab


class CameraLab:
    PW, PH = 640, 360            # live preview
    RW, RH = 800, 450            # review canvas

    def __init__(self, tab):
        self.tab, self.tk, self.ttk = tab, tab.tk, tab.ttk
        tk, ttk = self.tk, self.ttk
        self.log = tab.log
        self.alive = True
        self.busy = None
        self.st = load_settings()
        self.prof = B378.load_profile()
        self.board_T, self.board_info = None, None
        self.ds = Dataset(self.st["root"])
        self.cur = None                  # (folder, stem)
        self.cur_rgb = None
        self.editor = None
        self.photos = {}
        self.win = tk.Toplevel(getattr(tab, "root", None) or tab.cv)
        self.win.title(f"NERO {VERSION} 相机高级测试")
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.nb = ttk.Notebook(self.win)
        self.nb.pack(fill="both", expand=True)
        self._build_profile()
        self._build_noise()
        self._build_accuracy()
        self._build_capture()
        self.log(f"[INFO] [{MARKER}] 打开相机高级测试（数据文件夹 {self.ds.root}）")
        self._tick()

    # ------------------------------------------------------------------ helpers
    def _src(self):
        s = getattr(self.tab, "src", None)
        return s if s is not None and s.running() else None

    def _need_src(self):
        s = self._src()
        if s is None:
            self.log("[WARN] [V3.79 camera lab] 先在 Camera 调试页「连接」D435i")
        return s

    def _run(self, name, fn):
        if self.busy:
            self.log(f"[WARN] [V3.79 camera lab] 「{self.busy}」还在进行，等它结束")
            return False
        self.busy = name

        def job():
            try:
                fn()
            except Exception as exc:
                self.tab._ui(lambda e=exc: self.log(f"[FAIL] [V3.79 camera lab] {name}：{e}"))
            finally:
                self.busy = None
        threading.Thread(target=job, daemon=True, name=f"v379_{name}").start()
        return True

    def _show(self, canvas, key, rgb):
        tk = self.tk
        import NERO_V325_Camera as CAM
        data = CAM.ppm_bytes(np.ascontiguousarray(rgb))
        try:
            ph = tk.PhotoImage(data=data, format="PPM")
        except Exception:
            ph = tk.PhotoImage(data=CAM.png_bytes_from_ppm(data), format="PNG")
        self.photos[key] = ph
        canvas.delete("all")
        canvas.create_image(0, 0, image=ph, anchor="nw")

    def _var(self, value):
        return self.tk.StringVar(value=str(value))

    def _fget(self, var, lo, hi, name):
        try:
            v = float(var.get())
            assert lo <= v <= hi
            return v
        except Exception:
            raise ValueError(f"{name} 要在 {lo:g}–{hi:g}")

    def _save_st(self):
        try:
            save_settings(self.st)
        except Exception as exc:
            self.log(f"[WARN] [V3.79 camera lab] 设置没保存：{exc}")

    def close(self):
        self.alive = False
        self._save_st()
        _OPEN.pop(id(self.tab), None)
        try:
            self.win.destroy()
        except Exception:
            pass

    def _tick(self):
        if not self.alive:
            return
        try:
            self._refresh()
        except Exception as exc:
            if getattr(self, "_last_err", None) != str(exc):
                self._last_err = str(exc)
                self.log(f"[WARN] [V3.79 camera lab] 画面刷新：{exc}")
        try:
            self.win.after(200, self._tick)
        except Exception:
            self.alive = False

    def _page(self):
        try:
            return int(self.nb.index(self.nb.select()))
        except Exception:
            return 0

    def _refresh(self):
        src = self._src()
        rgb = src.frames()[0] if src is not None else None
        page = self._page()
        if rgb is None:
            return
        canvas = {0: self.cv_prof, 1: self.cv_noise, 2: self.cv_acc, 3: self.cv_live}.get(page)
        if canvas is None:
            return
        s = fit_scale((rgb.shape[1], rgb.shape[0]), (self.PW, self.PH) if page != 3 else (480, 270))
        small = resize(rgb, s)
        if page == 0:
            stt = B378.image_stats(rgb[::4, ::4])
            ok = 110 <= stt["mean"] <= 150 and stt["sat_pct"] < 0.5
            self.v_stats.set(f"平均亮度 {stt['mean']}（目标 110–150） · 过曝 {stt['sat_pct']} %（目标 < 0.5 %）"
                             f"  {'✓' if ok else '← 调曝光 / 增益'}")
        elif page == 1:
            w, h = small.shape[1], small.shape[0]
            r = 80 * s
            small = draw_overlay(small, [{"cls": "ROI", "box": [w / 2 - r, h / 2 - r, w / 2 + r, h / 2 + r],
                                          "poly": None, "conf": None, "source": "测量区"}])
        elif page == 2 and self.board_T is not None and src.K:
            small = draw_overlay(small, points=corner_pixels(self.board_T, src.K, B378.inner_corners()), scale=s)
        self._show(canvas, f"p{page}", small)

    # ------------------------------------------------------------------ ① camera parameters
    def _build_profile(self):
        tk, ttk = self.tk, self.ttk
        p = ttk.Frame(self.nb, padding=6)
        self.nb.add(p, text="① 相机参数")
        self.cv_prof = tk.Canvas(p, width=self.PW, height=self.PH, bg="#202020", highlightthickness=0)
        self.cv_prof.pack(anchor="w")
        self.v_stats = self._var("（连接相机后显示亮度）")
        ttk.Label(p, textvariable=self.v_stats).pack(anchor="w", pady=(4, 2))
        col = self.prof["color"]
        r = ttk.Frame(p)
        r.pack(fill="x", pady=2)
        self.v_exp, self.v_gain, self.v_wb = self._var(col["exposure"]), self._var(col["gain"]), self._var(col["white_balance"])
        pl = int(col.get("power_line_frequency", 2))
        self.v_pl = self._var(next((n for n, v in POWER_CHOICES if v == pl), "60 Hz"))
        for lab, var in (("曝光", self.v_exp), ("增益", self.v_gain), ("白平衡 K", self.v_wb)):
            ttk.Label(r, text=lab).pack(side="left")
            ttk.Entry(r, textvariable=var, width=7).pack(side="left", padx=(2, 8))
        ttk.Label(r, text="防闪烁").pack(side="left")
        ttk.Combobox(r, textvariable=self.v_pl, values=[n for n, _ in POWER_CHOICES], width=7,
                     state="readonly").pack(side="left", padx=(2, 8))
        r2 = ttk.Frame(p)
        r2.pack(fill="x", pady=2)
        ttk.Button(r2, text="自动调 3 秒后锁定", command=self.on_auto_lock).pack(side="left")
        ttk.Button(r2, text="应用（手动值）", command=self.on_apply).pack(side="left", padx=4)
        ttk.Button(r2, text="保存并锁定", command=self.on_save_profile).pack(side="left", padx=4)
        ttk.Button(r2, text="读回当前值", command=self.on_read_back).pack(side="left", padx=4)
        self.v_prof = self._var(self._prof_text())
        ttk.Label(p, textvariable=self.v_prof, justify="left", foreground="#335").pack(anchor="w", pady=(4, 0))
        ttk.Label(p, text="先把灯光开成平时干活的样子 →「自动调 3 秒后锁定」→ 亮度合适就「保存并锁定」。\n"
                          "锁定后 Camera 页每次连接相机都自动套用（深度预设仍按 Camera 页的选择）；之后的采图都在这组参数下。\n"
                          "防闪烁选当地电网频率（中国大陆 50 Hz，北美 60 Hz）：选错时画面会有横向滚动条纹。",
                  foreground="#555", justify="left").pack(anchor="w", pady=(4, 0))

    def _prof_text(self, rb=None):
        c = self.prof["color"]
        s = (f"保存的参数：曝光 {c.get('exposure')} · 增益 {c.get('gain')} · 白平衡 {c.get('white_balance')} · 防闪烁 "
             f"{B378.POWER_LINE.get(int(c.get('power_line_frequency', 0)), '?')} · "
             f"{'已锁定' if self.prof.get('locked') else '未锁定'}"
             + (f"（{self.prof.get('saved_at')}）" if self.prof.get("saved_at") else ""))
        if rb:
            cc = rb.get("color", {})
            s += (f"\n相机当前：曝光 {cc.get('exposure')} · 增益 {cc.get('gain')} · 白平衡 {cc.get('white_balance')} · "
                  f"自动曝光 {cc.get('enable_auto_exposure')} · 自动白平衡 {cc.get('enable_auto_white_balance')} · 防闪烁 "
                  f"{B378.POWER_LINE.get(int(cc.get('power_line_frequency', -1)), '?')}")
        return s

    def _manual_colour(self):
        col = self.prof["color"]
        col["exposure"] = self._fget(self.v_exp, 1, 10000, "曝光")
        col["gain"] = self._fget(self.v_gain, 0, 128, "增益")
        col["white_balance"] = self._fget(self.v_wb, 2800, 6500, "白平衡")
        col["power_line_frequency"] = dict(POWER_CHOICES).get(self.v_pl.get(), 2)
        col.update(enable_auto_exposure=0, enable_auto_white_balance=0)

    def _apply_now(self, prof):
        src = self._need_src()
        if src is None:
            return None
        p = json.loads(json.dumps(prof))
        p["depth"].pop("visual_preset", None)
        return src.run_on_device(lambda dev, rs: B378.apply_profile(dev, p, rs, log=self.log))

    def on_apply(self):
        try:
            self._manual_colour()
        except ValueError as exc:
            self.log(f"[WARN] [V3.79 camera lab] {exc}")
            return
        self._run("应用相机参数", lambda: (self._apply_now(self.prof), self.tab._ui(self.on_read_back)))

    def on_auto_lock(self):
        if self._need_src() is None:
            return

        def job():
            src = self._src()
            col = self.prof["color"]
            col["power_line_frequency"] = dict(POWER_CHOICES).get(self.v_pl.get(), 2)
            self._apply_now(dict(self.prof, color=dict(col, enable_auto_exposure=1, enable_auto_white_balance=1)))
            time.sleep(3.0)
            cur = src.run_on_device(lambda dev, rs: B378.read_back(dev, rs)).get("color", {})
            for k in ("exposure", "gain", "white_balance"):
                if k in cur:
                    col[k] = cur[k]
            col.update(enable_auto_exposure=0, enable_auto_white_balance=0)
            self._apply_now(self.prof)

            def ui():
                self.v_exp.set(str(col["exposure"]))
                self.v_gain.set(str(col["gain"]))
                self.v_wb.set(str(col["white_balance"]))
                self.on_read_back()
                self.log(f"[OK] [V3.79 camera lab] 自动调好并锁定：曝光 {col['exposure']} · 增益 {col['gain']} · "
                         f"白平衡 {col['white_balance']}（满意就点「保存并锁定」）")
            self.tab._ui(ui)
        self._run("自动调参", job)

    def on_save_profile(self):
        try:
            self._manual_colour()
        except ValueError as exc:
            self.log(f"[WARN] [V3.79 camera lab] {exc}")
            return
        self.prof["locked"] = True
        p = B378.save_profile(self.prof)
        self.prof = B378.load_profile()
        self.v_prof.set(self._prof_text())
        self.log(f"[OK] [V3.79 camera lab] 相机参数已保存并锁定 → {p}（Camera 页下次连接自动套用）")
        if self._src() is not None:
            self._run("应用相机参数", lambda: self._apply_now(self.prof))

    def on_read_back(self):
        src = self._src()
        if src is None:
            self.v_prof.set(self._prof_text())
            return
        try:
            rb = src.run_on_device(lambda dev, rs: B378.read_back(dev, rs), timeout=2.0)
        except Exception as exc:
            self.log(f"[WARN] [V3.79 camera lab] 读不到相机参数：{exc}")
            rb = None
        self.v_prof.set(self._prof_text(rb))

    # ------------------------------------------------------------------ ② depth noise
    def _build_noise(self):
        tk, ttk = self.tk, self.ttk
        p = ttk.Frame(self.nb, padding=6)
        self.nb.add(p, text="② 深度噪声")
        self.cv_noise = tk.Canvas(p, width=self.PW, height=self.PH, bg="#202020", highlightthickness=0)
        self.cv_noise.pack(anchor="w")
        r = ttk.Frame(p)
        r.pack(fill="x", pady=3)
        self.v_nframes = self._var(self.st.get("noise_frames", 30))
        ttk.Label(r, text="每种处理帧数").pack(side="left")
        ttk.Entry(r, textvariable=self.v_nframes, width=5).pack(side="left", padx=(2, 8))
        ttk.Button(r, text="▶ 开始测量（保持不动）", command=self.on_noise).pack(side="left")
        cols = ("setting", "valid", "std", "flat", "dist")
        self.tv_noise = ttk.Treeview(p, columns=cols, show="headings", height=6)
        for c, t, w in zip(cols, ("深度处理", "有效像素 %", "抖动 mm", "平整度 mm", "距离 mm"), (220, 90, 90, 90, 90)):
            self.tv_noise.heading(c, text=t)
            self.tv_noise.column(c, width=w, anchor="w" if c == "setting" else "e")
        self.tv_noise.pack(fill="x", pady=3)
        ttk.Label(p, text="相机对准平整、不反光的桌面或卡纸，画面中央方框是测量区，测量时人和物都不要动。\n"
                          "「抖动」= 同一像素逐帧变化（越小越稳）；「平整度」= 平均深度离最佳平面的偏差（越小越平）。\n"
                          "Camera 页现在用的是「spatial+temporal」。",
                  foreground="#555", justify="left").pack(anchor="w")

    def on_noise(self):
        src = self._need_src()
        if src is None:
            return
        try:
            n = int(self._fget(self.v_nframes, 5, 120, "帧数"))
        except ValueError as exc:
            self.log(f"[WARN] [V3.79 camera lab] {exc}")
            return
        self.st["noise_frames"] = n

        def job():
            rows, roi = measure_noise(src, dict(src.K), n, log=lambda m: self.tab._ui(lambda m_=m: self.log(m_)))
            stem = B378.write_results("noise", rows, {"roi": roi, "source": "camera lab"})

            def ui():
                self.tv_noise.delete(*self.tv_noise.get_children())
                for r_ in rows:
                    self.tv_noise.insert("", "end", values=(r_["setting"], r_["valid_pct"], r_["temporal_std_mm"],
                                                            r_["flatness_rms_mm"], r_["mean_dist_mm"]))
                self.log(f"[OK] [V3.79 noise] 结果已保存 → {stem}.csv")
            self.tab._ui(ui)
        self.log("[INFO] [V3.79 noise] 开始测量 5 种深度处理（约 30 s，保持不动）")
        self._run("深度噪声", job)

    # ------------------------------------------------------------------ ③ locating accuracy
    def _build_accuracy(self):
        tk, ttk = self.tk, self.ttk
        p = ttk.Frame(self.nb, padding=6)
        self.nb.add(p, text="③ 定位精度")
        self.cv_acc = tk.Canvas(p, width=self.PW, height=self.PH, bg="#202020", highlightthickness=0)
        self.cv_acc.pack(anchor="w")
        r = ttk.Frame(p)
        r.pack(fill="x", pady=3)
        self.v_aprompt = self._var(self.st.get("acc_prompt", "egg"))
        self.v_apose = self._var("竖放" if self.st.get("standing") else "横放")
        self.v_atrials = self._var(self.st.get("trials", 10))
        ttk.Label(r, text="物体").pack(side="left")
        ttk.Entry(r, textvariable=self.v_aprompt, width=10).pack(side="left", padx=(2, 8))
        ttk.Combobox(r, textvariable=self.v_apose, values=["横放", "竖放"], width=5, state="readonly").pack(side="left")
        ttk.Label(r, text="次数").pack(side="left", padx=(8, 0))
        ttk.Entry(r, textvariable=self.v_atrials, width=4).pack(side="left", padx=(2, 8))
        ttk.Button(r, text="1 锁定标定板（先不放鸡蛋）", command=self.on_board).pack(side="left")
        ttk.Button(r, text="2 ▶ 测量", command=self.on_accuracy).pack(side="left", padx=4)
        self.v_board = self._var("标定板：未锁定")
        ttk.Label(p, textvariable=self.v_board).pack(anchor="w")
        cols = ("setting", "det", "e3", "xy", "dz")
        self.tv_acc = ttk.Treeview(p, columns=cols, show="headings", height=3)
        for c, t, w in zip(cols, ("深度处理", "检出率 %", "3D 误差 均值/最大 mm", "水平 均值/最大 mm", "高度 均值/最大 mm"),
                           (150, 80, 160, 150, 150)):
            self.tv_acc.heading(c, text=t)
            self.tv_acc.column(c, width=w, anchor="w")
        self.tv_acc.pack(fill="x", pady=3)
        ttk.Label(p, text="打印 7×5 ChArUco 板（32 mm 方格），去掉塑料袋，四边胶带贴平。\n"
                          "① 不放鸡蛋点「锁定标定板」→ ② 鸡蛋放在画面上带编号的某个角点上 →「测量」。\n"
                          "真值 = 角点正上方蛋宽一半（横放 21 mm / 竖放 27.5 mm）；误差里含手放的偏差（约 ±1–2 mm）。"
                          "模型 / 置信度用 Camera 页「视觉定位」的设置。",
                  foreground="#555", justify="left").pack(anchor="w")

    def on_board(self):
        src = self._need_src()
        if src is None:
            return

        def job():
            rgbs = []
            last = None
            t0 = time.time()
            while len(rgbs) < 10 and time.time() - t0 < 4.0:
                rgb, _d, fid = src.frames()
                if rgb is not None and fid != last:
                    rgbs.append(rgb)
                    last = fid
                else:
                    time.sleep(0.02)
            T, info = B378.board_pose(rgbs, dict(src.K), list(src.dist or [0.0] * 5))

            def ui():
                self.board_T, self.board_info = T, info
                if T is None:
                    self.v_board.set(f"标定板：没找到（{info}）")
                    self.log(f"[WARN] [V3.79 accuracy] 标定板没找到：{info}")
                else:
                    self.v_board.set(f"标定板：已锁定 · {info['frames']} 帧 · 角点 {info['corners']} · 帧间 "
                                     f"{info['spread_mm']} mm · 相机离板 {info['cam_height_mm']} mm")
                    self.log(f"[OK] [V3.79 accuracy] 标定板已锁定：{info}")
            self.tab._ui(ui)
        self._run("锁定标定板", job)

    def _yolo(self):
        import NERO_V347_Yolo_Perception as YP
        tab = self.tab
        path = YP.find_model(tab.v_ydir.get(), tab.v_ymodel.get())
        if not path:
            raise RuntimeError(f"找不到 YOLOE 模型 {tab.v_ymodel.get()}（Camera 页「视觉定位」里选模型）")
        try:
            conf = float(tab.v_yconf.get())
        except Exception:
            conf = YP.CONF_MIN
        return YP, path, conf

    def on_accuracy(self):
        src = self._need_src()
        if src is None:
            return
        if self.board_T is None:
            self.log("[WARN] [V3.79 accuracy] 先不放鸡蛋点「锁定标定板」")
            return
        try:
            trials = int(self._fget(self.v_atrials, 1, 50, "次数"))
            YP, path, conf = self._yolo()
        except Exception as exc:
            self.log(f"[WARN] [V3.79 accuracy] {exc}")
            return
        prompt = self.v_aprompt.get().strip() or "egg"
        standing = self.v_apose.get() == "竖放"
        self.st.update(acc_prompt=prompt, standing=standing, trials=trials)
        T, K = np.array(self.board_T), dict(src.K)
        exp_z = B378.EGG_HALF_LENGTH_M if standing else B378.EGG_HALF_WIDTH_M

        def job():
            rows, summ = accuracy_trials(lambda n: self.tab._grab_frames(n),
                                         lambda rgb: self.tab.det_yolo.detect(rgb, "text", path, prompts=[prompt],
                                                                              conf=conf),
                                         K, T, B378.inner_corners(), exp_z, trials)
            summ["board"] = self.board_info
            summ["model"] = path
            stem = B378.write_results("accuracy", rows, summ)

            def ui():
                self.tv_acc.delete(*self.tv_acc.get_children())
                for name, s in summ.items():
                    if not isinstance(s, dict) or "err_3d" not in s:
                        continue
                    f = (lambda q: f"{q.get('mean', '-')} / {q.get('max', '-')}" if q.get("n") else "-")
                    self.tv_acc.insert("", "end", values=(name, s["detected_pct"], f(s["err_3d"]), f(s["xy"]),
                                                          f(s["dz"])))
                corner = next((r_.get("corner") for r_ in rows if r_.get("corner") is not None), "?")
                self.log(f"[OK] [V3.79 accuracy] 角点 {corner}：结果已保存 → {stem}.csv")
            self.tab._ui(ui)
        self.log(f"[INFO] [V3.79 accuracy] 开始测量（{prompt}，{'竖放' if standing else '横放'}，每种 {trials} 次）")
        self._run("定位精度", job)

    # ------------------------------------------------------------------ ④ capture + auto label + review
    def _build_capture(self):
        tk, ttk = self.tk, self.ttk
        p = ttk.Frame(self.nb, padding=6)
        self.nb.add(p, text="④ 采集标注")
        top = ttk.Frame(p)
        top.pack(fill="x")
        self.v_root = self._var(self.st["root"])
        ttk.Label(top, text="数据文件夹").pack(side="left")
        ttk.Entry(top, textvariable=self.v_root, width=46).pack(side="left", padx=2)
        ttk.Button(top, text="选择…", command=self.on_pick_root).pack(side="left")
        self.v_prompts = self._var(self.st.get("prompts", "egg"))
        ttk.Label(top, text="物体名（逗号分隔，第 1 个 = 子文件夹）").pack(side="left", padx=(10, 2))
        ttk.Entry(top, textvariable=self.v_prompts, width=22).pack(side="left")
        r2 = ttk.Frame(p)
        r2.pack(fill="x", pady=3)
        self.v_cconf = self._var(self.st.get("conf", 0.25))
        self.v_cdepth = self.tk.BooleanVar(value=bool(self.st.get("save_depth", True)))
        self.v_bn, self.v_bdt = self._var(self.st.get("burst_n", 10)), self._var(self.st.get("burst_dt", 1.0))
        ttk.Label(r2, text="置信度").pack(side="left")
        ttk.Entry(r2, textvariable=self.v_cconf, width=5).pack(side="left", padx=(2, 8))
        ttk.Checkbutton(r2, text="同时存深度图", variable=self.v_cdepth).pack(side="left")
        ttk.Button(r2, text="📷 拍一张", command=self.on_shot).pack(side="left", padx=(10, 2))
        ttk.Label(r2, text="连拍").pack(side="left", padx=(8, 0))
        ttk.Entry(r2, textvariable=self.v_bn, width=4).pack(side="left", padx=2)
        ttk.Label(r2, text="张，间隔").pack(side="left")
        ttk.Entry(r2, textvariable=self.v_bdt, width=4).pack(side="left", padx=2)
        ttk.Label(r2, text="s").pack(side="left")
        ttk.Button(r2, text="▶ 连拍", command=self.on_burst).pack(side="left", padx=4)
        ttk.Button(r2, text="■ 停", command=self.on_stop_burst).pack(side="left")
        self.v_cstat = self._var("")
        ttk.Label(p, textvariable=self.v_cstat, foreground="#335").pack(anchor="w")
        body = ttk.Frame(p)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body)
        left.pack(side="left", fill="y")
        self.cv_live = tk.Canvas(left, width=480, height=270, bg="#202020", highlightthickness=0)
        self.cv_live.pack(anchor="w")
        lr = ttk.Frame(left)
        lr.pack(fill="x", pady=2)
        self.v_folder = self._var("")
        ttk.Label(lr, text="复核").pack(side="left")
        self.cb_folder = ttk.Combobox(lr, textvariable=self.v_folder, width=16, state="readonly",
                                      postcommand=self._refresh_folders)
        self.cb_folder.pack(side="left", padx=2)
        self.cb_folder.bind("<<ComboboxSelected>>", lambda _e: self._fill_list())
        ttk.Button(lr, text="刷新", command=self._fill_list).pack(side="left")
        cols = ("stem", "split", "n", "rev")
        self.tv_items = ttk.Treeview(left, columns=cols, show="headings", height=9)
        for c, t, w in zip(cols, ("图片", "集", "框", "已复核"), (250, 50, 40, 60)):
            self.tv_items.heading(c, text=t)
            self.tv_items.column(c, width=w, anchor="w")
        self.tv_items.pack(fill="both", expand=True)
        self.tv_items.bind("<<TreeviewSelect>>", lambda _e: self._on_pick_item())
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self.cv_rev = tk.Canvas(right, width=self.RW, height=self.RH, bg="#303030", highlightthickness=0)
        self.cv_rev.pack(anchor="w")
        self.cv_rev.bind("<ButtonPress-1>", self._ev_press)
        self.cv_rev.bind("<B1-Motion>", self._ev_drag)
        self.cv_rev.bind("<ButtonRelease-1>", self._ev_release)
        self.win.bind("<Delete>", lambda _e: self.on_del_box())
        rr = ttk.Frame(right)
        rr.pack(fill="x", pady=2)
        self.v_bcls = self._var("")
        ttk.Label(rr, text="选中框的类别").pack(side="left")
        self.cb_bcls = ttk.Combobox(rr, textvariable=self.v_bcls, width=12, postcommand=self._refresh_cls)
        self.cb_bcls.pack(side="left", padx=2)
        self.cb_bcls.bind("<<ComboboxSelected>>", lambda _e: self.on_set_cls())
        ttk.Button(rr, text="删除框 (Del)", command=self.on_del_box).pack(side="left", padx=4)
        ttk.Button(rr, text="◀ 上一张", command=lambda: self._step(-1)).pack(side="left", padx=(10, 2))
        ttk.Button(rr, text="✓ 已复核，下一张", command=lambda: self._step(+1, reviewed=True)).pack(side="left", padx=2)
        ttk.Button(rr, text="下一张 ▶", command=lambda: self._step(+1)).pack(side="left", padx=2)
        ttk.Button(rr, text="删除这张图", command=self.on_del_image).pack(side="left", padx=(10, 0))
        self.v_rev = self._var("在左边列表选一张图。点框 = 选中；拖框内 = 移动；拖白色角 = 改大小；在空白处拖 = 新框"
                               "（类别 = 上面的框类别或物体名第 1 个）；Del = 删除。框改动后轮廓自动重算。")
        ttk.Label(right, textvariable=self.v_rev, foreground="#555", wraplength=self.RW, justify="left").pack(anchor="w")
        self._update_cstat()

    def _set_root(self, root):
        root = os.path.abspath(os.path.expanduser(root.strip() or DEFAULT_ROOT))
        if root != self.ds.root:
            self.ds = Dataset(root)
            self.cur = self.editor = self.cur_rgb = None
        self.st["root"] = root
        return root

    def on_pick_root(self):
        try:
            from tkinter import filedialog
            d = filedialog.askdirectory(initialdir=self.v_root.get() or DEFAULT_ROOT, title="选择数据文件夹")
        except Exception:
            d = None
        if d:
            self.v_root.set(d)
            self._set_root(d)
            self._update_cstat()
            self._save_st()

    def _prompts(self):
        import NERO_V347_Yolo_Perception as YP
        return [p for p in YP.parse_prompts(self.v_prompts.get()) if p.strip()]

    def _update_cstat(self):
        st = self.ds.stats()
        if not st:
            self.v_cstat.set(f"{self.ds.root}：还没有图片")
            return
        self.v_cstat.set("  ·  ".join(f"{k}: {v['images']} 张（训练 {v['train']} / 验证 {v['val']}，已复核 "
                                      f"{v['reviewed']}，{v['objects']} 个框）" for k, v in st.items()))

    def _capture_one(self, prompts, conf, depth_on, YP, path):
        src = self._src()
        if src is None:
            raise RuntimeError("相机未连接")
        rgb, depths = self.tab._grab_frames(1)
        if rgb is None:
            raise RuntimeError("没有取到画面")
        dets = self.tab.det_yolo.detect(rgb, "text", path, prompts=prompts, conf=conf)
        anns = dets_to_anns(dets, prompts)
        stem = self.ds.add_image(prompts[0], rgb, anns, depth=(depths[-1] if depth_on and depths else None),
                                 extra={"prompts": prompts, "conf_min": conf, "model": os.path.basename(path),
                                        "K": dict(src.K or {}), "profile_locked": bool(B378.load_profile().get("locked"))})
        return safe_class(prompts[0]), stem, len(anns)

    def _capture_args(self):
        prompts = self._prompts()
        if not prompts:
            raise ValueError("先填物体名（例如 egg 或 egg, cup）")
        conf = self._fget(self.v_cconf, 0.01, 0.99, "置信度")
        YP, path, _c = self._yolo()
        self._set_root(self.v_root.get())
        self.st.update(prompts=", ".join(prompts), conf=conf, save_depth=bool(self.v_cdepth.get()))
        if not B378.load_profile().get("locked"):
            self.log("[WARN] [V3.79 capture] 相机参数还没锁定（① 相机参数 →「保存并锁定」）：现在拍的图和以后的可能对不上")
        return prompts, conf, bool(self.v_cdepth.get()), YP, path

    def _after_shot(self, folder, stem, n):
        self._update_cstat()
        self.v_folder.set(folder)
        self._fill_list(select=stem)
        self.log(f"[OK] [V3.79 capture] {stem}：自动标注 {n} 个框 → {self.ds.cdir(folder)}")

    def on_shot(self):
        if self._need_src() is None:
            return
        try:
            args = self._capture_args()
        except Exception as exc:
            self.log(f"[WARN] [V3.79 capture] {exc}")
            return

        def job():
            folder, stem, n = self._capture_one(*args)
            self.tab._ui(lambda: self._after_shot(folder, stem, n))
        self._run("拍照", job)

    def on_burst(self):
        if self._need_src() is None:
            return
        try:
            args = self._capture_args()
            n = int(self._fget(self.v_bn, 1, 500, "张数"))
            dt = self._fget(self.v_bdt, 0.2, 60, "间隔")
        except Exception as exc:
            self.log(f"[WARN] [V3.79 capture] {exc}")
            return
        self.st.update(burst_n=n, burst_dt=dt)
        self._burst_stop = False

        def job():
            for i in range(n):
                if self._burst_stop or not self.alive:
                    break
                t0 = time.time()
                folder, stem, k = self._capture_one(*args)
                self.tab._ui(lambda f=folder, s=stem, k_=k: self._after_shot(f, s, k_))
                while time.time() - t0 < dt and not self._burst_stop:
                    time.sleep(0.05)
            self.tab._ui(lambda: self.log("[OK] [V3.79 capture] 连拍结束"))
        self.log(f"[INFO] [V3.79 capture] 连拍 {n} 张，间隔 {dt:g} s（拍的时候可以挪动物体 / 换摆法）")
        self._run("连拍", job)

    def on_stop_burst(self):
        self._burst_stop = True

    # ---- review
    def _refresh_folders(self):
        self.cb_folder.configure(values=self.ds.folders())

    def _refresh_cls(self):
        self.cb_bcls.configure(values=list(dict.fromkeys(self.ds.classes() + self._prompts())))

    def _fill_list(self, select=None):
        self._set_root(self.v_root.get())
        fo = self.v_folder.get()
        self.tv_items.delete(*self.tv_items.get_children())
        if not fo:
            return
        for m in self.ds.items(fo):
            self.tv_items.insert("", "end", iid=m["stem"], values=(m["stem"], m.get("split"), len(m.get("anns", [])),
                                                                   "✓" if m.get("reviewed") else ""))
        if select:
            try:
                self.tv_items.selection_set(select)
                self.tv_items.see(select)
            except Exception:
                pass
            self._open_item(fo, select)

    def _on_pick_item(self):
        sel = self.tv_items.selection()
        if sel and (self.cur is None or self.cur[1] != sel[0]):
            self._open_item(self.v_folder.get(), sel[0])

    def _open_item(self, folder, stem):
        self._commit()
        rgb, meta = self.ds.load(folder, stem)
        if rgb is None:
            self.log(f"[WARN] [V3.79 review] 读不到图片 {stem}")
            return
        self.cur, self.cur_rgb, self.cur_meta = (folder, stem), rgb, meta
        prompts = meta.get("prompts") or self._prompts() or [folder]
        self.editor = BoxEditor(json.loads(json.dumps(meta.get("anns", []))), meta["size"], default_cls=prompts[0])
        self._dirty = False
        self._redraw()

    def _scale(self):
        return fit_scale(self.cur_meta["size"], (self.RW, self.RH))

    def _redraw(self):
        if self.cur_rgb is None or self.editor is None:
            return
        s = self._scale()
        img = draw_overlay(resize(self.cur_rgb, s), self.editor.anns, self.editor.sel, scale=s)
        self._show(self.cv_rev, "rev", img)
        m = self.cur_meta
        sel = self.editor.sel
        self.v_rev.set(f"{m['stem']}（{m['split']}，{len(self.editor.anns)} 个框"
                       f"{'，已复核' if m.get('reviewed') else ''}）"
                       + (f" · 选中 #{sel} {self.editor.anns[sel]['cls']}" if sel is not None else ""))
        if sel is not None:
            self.v_bcls.set(self.editor.anns[sel]["cls"])

    def _img_xy(self, ev):
        s = self._scale()
        return ev.x / s, ev.y / s, s

    def _ev_press(self, ev):
        if self.editor is None:
            return
        x, y, s = self._img_xy(ev)
        if self.v_bcls.get().strip():
            self.editor.default_cls = self.v_bcls.get().strip()
        self.editor.press(x, y, s)
        self._redraw()

    def _ev_drag(self, ev):
        if self.editor is None:
            return
        x, y, _s = self._img_xy(ev)
        self.editor.drag(x, y)
        self._redraw()

    def _ev_release(self, ev):
        if self.editor is None:
            return
        x, y, _s = self._img_xy(ev)
        i = self.editor.release(x, y)
        if i is not None:
            a = self.editor.anns[i]
            a["poly"], how = recompute_outline(self.cur_rgb, a["box"], self.cur_meta.get("auto"))
            a["outline"] = how
            self._dirty = True
            self._commit()
        self._redraw()

    def on_del_box(self):
        if self.editor is not None and self.editor.delete():
            self._dirty = True
            self._commit()
            self._redraw()

    def on_set_cls(self):
        c = self.v_bcls.get().strip()
        if self.editor is not None and c and self.editor.set_class(c):
            self._dirty = True
            self._commit()
            self._redraw()

    def _commit(self, reviewed=None):
        if self.cur is None or self.editor is None:
            return
        if getattr(self, "_dirty", False) or reviewed is not None:
            self.cur_meta = self.ds.save_anns(self.cur[0], self.cur[1], self.editor.anns, reviewed=reviewed)
            self._dirty = False
            try:
                self.tv_items.item(self.cur[1], values=(self.cur[1], self.cur_meta.get("split"), len(self.editor.anns),
                                                        "✓" if self.cur_meta.get("reviewed") else ""))
            except Exception:
                pass

    def _step(self, d, reviewed=None):
        if self.cur is None:
            return
        self._commit(reviewed)
        self._update_cstat()
        ids = list(self.tv_items.get_children())
        if self.cur[1] not in ids:
            return
        j = ids.index(self.cur[1]) + d
        if 0 <= j < len(ids):
            self.tv_items.selection_set(ids[j])
            self.tv_items.see(ids[j])
            self._open_item(self.cur[0], ids[j])

    def on_del_image(self):
        if self.cur is None:
            return
        fo, stem = self.cur
        ids = list(self.tv_items.get_children())
        nxt = None
        if stem in ids:
            j = ids.index(stem)
            nxt = ids[j + 1] if j + 1 < len(ids) else (ids[j - 1] if j > 0 else None)
        self.ds.delete_image(fo, stem)
        self.log(f"[INFO] [V3.79 review] 已删除 {stem}")
        self.cur = self.editor = self.cur_rgb = None
        self.cv_rev.delete("all")
        self._update_cstat()
        self._fill_list(select=nxt)
