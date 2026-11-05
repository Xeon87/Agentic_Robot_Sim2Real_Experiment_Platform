"""NERO V3.81 YOLOE fine-tuning + evaluation (V3.81 YOLOE TRAIN) - page ⑤ of the camera lab.

Data   : the V3.79 dataset (<root>/classes.json, <root>/<class>/images|labels/{train,val}, meta/*.json).
         plan_dataset() picks the images (only reviewed ones by default), export_dataset() writes a training copy
         <root>/_runs/<run>/dataset (hard links + labels remapped to the chosen classes, val never trained on).
Train  : a child process (`python3 NERO_V381_Yolo_Train.py job <job.json>`, clean_child_env, own session) runs
         ultralytics' YOLOEPESegTrainer:
           quick = linear probing (everything frozen except the last layer of the class-embedding branches) - the
                   ultralytics-documented way for small datasets (default)
           full  = all layers trained
         progress -> <run>/progress.jsonl (one JSON line per event); stop = <run>/STOP (trainer.stop after the epoch),
         the parent kills the process group if it does not stop.
Eval   : our own matching code on the SAME fixed val images for the original model (text prompts = class names) and
         the new model (its own trained classes - never set_classes on it): per class recall (识别率) / precision /
         false detections per image at the Camera tab's confidence, AP50 / AP50-95, outline IoU, ms per image.
         compare() turns the numbers into a plain verdict.
Save   : save_model() copies the weights + card into ~/Downloads/objectmodels/finetuned/<name>/ (NERO_V381_FINETUNE_DIR);
         nothing uses it before V3.82 (the Camera tab only lists yoloe*.pt at the top of the model folder).
Never writes the release folder.
"""
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
VERSION = "V3.82"
MARKER = "V3.81 YOLOE TRAIN"
MARKER_V382 = "V3.82 APPLY FINE-TUNED + 1280 TRAINING"
# V3.82 APPLY FINE-TUNED + 1280 TRAINING: train at the size the Camera tab detects at (1280; small eggs keep their
# pixels), batch sized for an 8 GB laptop GPU; the evaluation always runs at the Camera tab's size
MODES = {
    "quick": {"label": "快速（只训分类头，推荐）", "epochs": 40, "batch": 4, "imgsz": 1280, "lr0": 1e-3},
    "full": {"label": "完整精调（数据多时更准，较慢）", "epochs": 80, "batch": 2, "imgsz": 1280, "lr0": 2e-4},
}
MIN_TRAIN_IMAGES = 20             # per class: below -> warning (best >= 100)
MIN_VAL_OBJECTS = 5               # per class: below -> the verdict is unreliable
EVAL_LOW_CONF = 0.05              # predictions kept for AP
IOU_MATCH = 0.5
BETTER_RECALL_PTS = 5.0           # verdict thresholds (percentage points / AP)
BETTER_AP = 0.05
WORST_DROP_PTS = 2.0
STOP_GRACE_S = 90.0


def finetune_dir():
    d = os.environ.get("NERO_V381_FINETUNE_DIR")
    if d:
        return d
    try:
        import NERO_V347_Yolo_Perception as YP
        base = YP.DEFAULT_MODEL_DIR
    except Exception:
        base = os.path.expanduser("~/Downloads/objectmodels")
    return os.path.join(base, "finetuned")


def safe_name(s):
    import re
    s = re.sub(r'[\\/:*?"<>|\s]+', "_", str(s or "").strip()).strip("._")
    return s or "model"


# ============================================================================================ dataset
def scan(root):
    """every image of the V3.79 dataset -> [{folder, stem, split, reviewed, anns, image, size}]"""
    out = []
    if not os.path.isdir(root):
        return out
    for fo in sorted(os.listdir(root)):
        md = os.path.join(root, fo, "meta")
        if fo.startswith("_") or not os.path.isdir(md):
            continue
        for fn in sorted(os.listdir(md)):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(md, fn), encoding="utf-8") as f:
                    m = json.load(f)
            except Exception:
                continue
            img = os.path.join(root, fo, m.get("image", ""))
            if not os.path.isfile(img):
                continue
            out.append({"folder": fo, "stem": m.get("stem", fn[:-5]), "split": m.get("split", "train"),
                        "reviewed": bool(m.get("reviewed")), "image": img,
                        "anns": [a for a in m.get("anns", []) if not a.get("candidate")],     # V3.82: confirmed only
                        "size": m.get("size")})
    return out


def class_table(items):
    """per class over all images: reviewed / total images (train / val) and boxes -> {cls: {...}}"""
    t = {}
    for it in items:
        seen = set()
        for a in it["anns"]:
            c = t.setdefault(a["cls"], {"images": 0, "reviewed": 0, "train": 0, "val": 0, "objects": 0})
            c["objects"] += 1
            if a["cls"] in seen:
                continue
            seen.add(a["cls"])
            c["images"] += 1
            c["reviewed"] += int(it["reviewed"])
            c[it["split"] if it["split"] in ("train", "val") else "train"] += 1
    return t


def plan_dataset(items, classes, reviewed_only=True):
    """-> {"items", "per_class", "background", "dropped", "skipped_unreviewed", "errors", "warnings", "ok"}"""
    classes = [c for c in classes if c]
    sel = set(classes)
    folders = {safe_name(c) for c in classes}
    per = {c: {"train_images": 0, "val_images": 0, "train_objects": 0, "val_objects": 0} for c in classes}
    chosen, background, dropped, skipped = [], 0, 0, 0
    for it in items:
        mine = [a for a in it["anns"] if a["cls"] in sel]
        if not mine and it["folder"] not in folders:
            continue
        if reviewed_only and not it["reviewed"]:
            skipped += 1
            continue
        split = "val" if it["split"] == "val" else "train"
        chosen.append(dict(it, split=split, anns=mine))
        dropped += len(it["anns"]) - len(mine)
        if not mine:
            background += 1
        seen = set()
        for a in mine:
            per[a["cls"]][f"{split}_objects"] += 1
            if a["cls"] not in seen:
                per[a["cls"]][f"{split}_images"] += 1
                seen.add(a["cls"])
    errors, warnings = [], []
    if not classes:
        errors.append("先勾选要训练的类别")
    for c in classes:
        p = per[c]
        if p["train_images"] == 0:
            errors.append(f"「{c}」训练集没有图" + ("（只用已复核的图：先在 ④ 复核）" if reviewed_only else ""))
        elif p["train_images"] < MIN_TRAIN_IMAGES:
            warnings.append(f"「{c}」训练集只有 {p['train_images']} 张（建议 ≥ {MIN_TRAIN_IMAGES}，最好 ≥ 100）")
        if p["val_objects"] == 0:
            errors.append(f"「{c}」验证集没有框：没法评估（每 5 张里第 5 张进验证集，再多拍几张）")
        elif p["val_objects"] < MIN_VAL_OBJECTS:
            warnings.append(f"「{c}」验证集只有 {p['val_objects']} 个框：评估结论不可靠（建议 ≥ {MIN_VAL_OBJECTS}，最好 ≥ 20）")
    if skipped:
        warnings.append(f"{skipped} 张还没复核，没用上（在 ④ 复核后点「✓ 已复核」）")
    if dropped:
        warnings.append(f"{dropped} 个框属于没勾选的类别，训练时去掉（这些物体在图里会被当成背景）")
    return {"items": chosen, "per_class": per, "background": background, "dropped": dropped,
            "skipped_unreviewed": skipped, "errors": errors, "warnings": warnings, "ok": not errors,
            "classes": classes, "reviewed_only": bool(reviewed_only)}


def _box_poly(b):
    return [[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]]


def export_dataset(plan, out_dir):
    """training copy: images hard-linked (copied across file systems), labels with the chosen class ids -> data.yaml"""
    classes = plan["classes"]
    cid = {c: i for i, c in enumerate(classes)}
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    manifest = []
    for it in plan["items"]:
        name = f"{it['folder']}__{it['stem']}"
        dst = os.path.join(out_dir, "images", it["split"], name + ".jpg")
        if not os.path.exists(dst):
            try:
                os.link(it["image"], dst)
            except OSError:
                shutil.copy2(it["image"], dst)
        w, h = (it.get("size") or [1280, 720])
        lines = []
        for a in it["anns"]:
            poly = a.get("poly") if a.get("poly") and len(a["poly"]) >= 3 else _box_poly(a["box"])
            pts = " ".join(f"{min(max(x / w, 0.0), 1.0):.6f} {min(max(y / h, 0.0), 1.0):.6f}" for x, y in poly)
            lines.append(f"{cid[a['cls']]} {pts}")
        with open(os.path.join(out_dir, "labels", it["split"], name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        manifest.append({"name": name, "split": it["split"], "image": dst, "size": [w, h],
                         "anns": [{"cls": a["cls"], "box": a["box"], "poly": a.get("poly")} for a in it["anns"]]})
    yml = os.path.join(out_dir, "data.yaml")
    with open(yml, "w", encoding="utf-8") as f:
        f.write(f"path: {json.dumps(os.path.abspath(out_dir))}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, c in enumerate(classes):
            f.write(f"  {i}: {json.dumps(c, ensure_ascii=False)}\n")
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"version": VERSION, "classes": classes, "items": manifest, "per_class": plan["per_class"],
                   "reviewed_only": plan["reviewed_only"]}, f, ensure_ascii=False, indent=1)
    return yml


# ============================================================================================ freezing
def linear_probe_freeze(seq):
    """ultralytics' documented YOLOE linear probing, generalised: freeze every layer before the head and, in the head,
    everything except the LAST layer of each class-embedding branch (children named *cv3, e.g. cv3 and the
    one2one_cv3 of NMS-free heads).  -> (freeze list for model.train(freeze=...), trainable branch names)"""
    head_index = len(seq) - 1
    freeze = [str(i) for i in range(head_index)]
    head = seq[head_index]
    kids = list(head.named_children())
    cls_branches = [n for n, _m in kids if n.endswith("cv3")]
    for n, m in kids:
        if n not in cls_branches:
            freeze.append(f"{head_index}.{n}")
            continue
        for i in range(len(m)):
            try:
                k = len(m[i])
            except TypeError:
                continue
            for j in range(k - 1):
                freeze.append(f"{head_index}.{n}.{i}.{j}")
    return freeze, cls_branches


# ============================================================================================ evaluation
def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _mask(poly, box, hw, s=0.5):
    """rasterised outline at half resolution (box when there is no outline)"""
    import NERO_V347_Yolo_Perception as YP
    cv2 = YP._cv2()
    h, w = int(hw[0] * s) + 1, int(hw[1] * s) + 1
    m = np.zeros((h, w), np.uint8)
    pts = poly if poly and len(poly) >= 3 else _box_poly(box)
    cv2.fillPoly(m, [np.round(np.asarray(pts, float) * s).astype(np.int32).reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def _ap(tp, conf, n_gt):
    """COCO 101-point AP from per-prediction TP flags"""
    if n_gt == 0:
        return None
    if not tp:
        return 0.0
    order = np.argsort(-np.asarray(conf))
    t = np.asarray(tp, float)[order]
    ctp, cfp = np.cumsum(t), np.cumsum(1.0 - t)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    ap = 0.0
    for r in np.linspace(0, 1, 101):
        k = np.searchsorted(rec, r, side="left")
        ap += prec[k] if k < len(prec) else 0.0
    return float(ap / 101.0)


def _match(preds, gts, thr):
    """greedy by confidence, same class, box IoU >= thr -> list of (pred index, gt index or None)"""
    used, out = set(), []
    for i in sorted(range(len(preds)), key=lambda k: -preds[k]["conf"]):
        p = preds[i]
        best, bj = thr, None
        for j, g in enumerate(gts):
            if j in used or g["cls"] != p["cls"]:
                continue
            v = box_iou(p["box"], g["box"])
            if v >= best:
                best, bj = v, j
        if bj is not None:
            used.add(bj)
        out.append((i, bj))
    return out


def evaluate(samples, classes, conf_thr):
    """samples: [{"gts": [{cls, box, poly}], "preds": [{cls, conf, box, poly}], "hw": (h, w), "ms": float}]
    -> {"per_class": {cls: {...}}, "overall": {...}}"""
    per = {c: {"n_gt": 0, "tp": 0, "fp": 0, "ap_rows": {t: ([], []) for t in range(10)}, "mask_iou": []}
           for c in classes}
    n_img = max(1, len(samples))
    ms = [s_["ms"] for s_ in samples if s_.get("ms") is not None]
    for s_ in samples:
        gts = [g for g in s_["gts"] if g["cls"] in per]
        preds = [p for p in s_["preds"] if p["cls"] in per]
        for g in gts:
            per[g["cls"]]["n_gt"] += 1
        for t in range(10):                                   # AP rows at IoU 0.50 ... 0.95 (all kept predictions)
            thr = 0.5 + 0.05 * t
            for i, j in _match(preds, gts, thr):
                rows = per[preds[i]["cls"]]["ap_rows"][t]
                rows[0].append(j is not None)
                rows[1].append(preds[i]["conf"])
        hi = [p for p in preds if p["conf"] >= conf_thr]          # detection at the working confidence
        for i, j in _match(hi, gts, IOU_MATCH):
            c = per[hi[i]["cls"]]
            if j is None:
                c["fp"] += 1
                continue
            c["tp"] += 1
            if hi[i].get("poly") or gts[j].get("poly"):
                a = _mask(hi[i].get("poly"), hi[i]["box"], s_["hw"])
                b = _mask(gts[j].get("poly"), gts[j]["box"], s_["hw"])
                u = float((a | b).sum())
                c["mask_iou"].append(float((a & b).sum()) / u if u else 0.0)
    out = {}
    for c, v in per.items():
        ap = [_ap(v["ap_rows"][t][0], v["ap_rows"][t][1], v["n_gt"]) for t in range(10)]
        out[c] = {"n_gt": v["n_gt"], "tp": v["tp"], "fp": v["fp"],
                  "recall": round(100.0 * v["tp"] / v["n_gt"], 1) if v["n_gt"] else None,
                  "precision": round(100.0 * v["tp"] / (v["tp"] + v["fp"]), 1) if v["tp"] + v["fp"] else None,
                  "fp_per_img": round(v["fp"] / n_img, 3),
                  "ap50": None if ap[0] is None else round(ap[0], 3),
                  "ap50_95": None if ap[0] is None else round(float(np.mean(ap)), 3),
                  "mask_iou": round(float(np.mean(v["mask_iou"])), 3) if v["mask_iou"] else None}
    have = [o for o in out.values() if o["n_gt"]]
    mean = lambda k: round(float(np.mean([o[k] for o in have if o[k] is not None])), 3) if any(
        o[k] is not None for o in have) else None
    overall = {"recall": mean("recall"), "precision": mean("precision"), "ap50": mean("ap50"),
               "ap50_95": mean("ap50_95"), "mask_iou": mean("mask_iou"),
               "fp_per_img": round(sum(o["fp"] for o in out.values()) / n_img, 3), "images": len(samples),
               "objects": sum(o["n_gt"] for o in out.values()),
               "ms_per_img": round(float(np.median(ms)), 1) if ms else None, "conf": conf_thr}
    return {"per_class": out, "overall": overall}


def compare(base, new):
    """-> {"level": better / same / worse, "unreliable": bool, "lines": [...]}"""
    ob, on = base["overall"], new["overall"]
    lines = []
    small = [c for c, v in new["per_class"].items() if v["n_gt"] < MIN_VAL_OBJECTS]
    d_rec = (on["recall"] or 0.0) - (ob["recall"] or 0.0)
    d_ap = (on["ap50"] or 0.0) - (ob["ap50"] or 0.0)
    drops = {c: (v["recall"] or 0.0) - (base["per_class"].get(c, {}).get("recall") or 0.0)
             for c, v in new["per_class"].items() if v["n_gt"]}
    worst_c = min(drops, key=drops.get) if drops else None
    worst = drops.get(worst_c, 0.0) if worst_c else 0.0
    d_fp = (on["fp_per_img"] or 0.0) - (ob["fp_per_img"] or 0.0)
    if (d_rec >= BETTER_RECALL_PTS or d_ap >= BETTER_AP) and worst >= -WORST_DROP_PTS and (d_fp <= 0.2 or d_rec >= 15):
        level = "better"
        lines.append(f"结论：有实质提升。平均识别率 {ob['recall']} % → {on['recall']} %（{d_rec:+.1f} 个百分点），"
                     f"mAP50 {ob['ap50']} → {on['ap50']}，误检每张 {ob['fp_per_img']} → {on['fp_per_img']}")
    elif d_rec <= -WORST_DROP_PTS or d_ap <= -0.03:
        level = "worse"
        lines.append(f"结论：变差了。平均识别率 {ob['recall']} % → {on['recall']} %（{d_rec:+.1f}），mAP50 {ob['ap50']} → "
                     f"{on['ap50']}：不要保存；多拍些图（换位置 / 背景 / 光照）、检查标注，或换「完整精调」再试")
    else:
        level = "same"
        lines.append(f"结论：没有实质提升（识别率 {d_rec:+.1f} 个百分点，mAP50 {d_ap:+.3f}）。原模型已经认得这些物体，"
                     "或者数据太少；可以不保存")
    if worst_c and worst < -WORST_DROP_PTS:
        lines.append(f"注意：「{worst_c}」的识别率下降 {worst:.1f} 个百分点")
    if d_fp > 0.2:
        lines.append(f"注意：误检变多（每张 {ob['fp_per_img']} → {on['fp_per_img']}）")
    if small:
        lines.append(f"验证集太小（{', '.join(small)} 少于 {MIN_VAL_OBJECTS} 个框）：结论只能参考，多拍几张再训一次更可靠")
    lines.append(f"速度：每张 {ob.get('ms_per_img')} ms → {on.get('ms_per_img')} ms；识别率 / 误检按 Camera 页的置信度 "
                 f"{on.get('conf')} 统计")
    return {"level": level, "unreliable": bool(small), "lines": lines, "d_recall": round(d_rec, 1),
            "d_ap50": round(d_ap, 3)}


def result_to_preds(r, names):
    """one ultralytics result -> [{cls, conf, box, poly}] with class NAMES"""
    import NERO_V347_Yolo_Perception as YP
    if r.boxes is None or len(r.boxes) == 0:
        return []
    xyxy = YP._np(r.boxes.xyxy).reshape(-1, 4)
    confs = YP._np(r.boxes.conf).reshape(-1)
    clss = YP._np(r.boxes.cls).reshape(-1).astype(int)
    polys = list(r.masks.xy) if getattr(r, "masks", None) is not None else [None] * len(xyxy)
    out = []
    for k in range(len(xyxy)):
        c = int(clss[k])
        nm = names.get(c, names.get(str(c), str(c))) if isinstance(names, dict) else (
            names[c] if 0 <= c < len(names) else str(c))
        p = polys[k] if k < len(polys) else None
        out.append({"cls": str(nm), "conf": float(confs[k]), "box": [float(v) for v in xyxy[k]],
                    "poly": [[float(x), float(y)] for x, y in np.asarray(p).reshape(-1, 2)] if p is not None and len(p) >= 3 else None})
    return out


def eval_model(predict, manifest, classes, conf_thr, log=print):
    """predict(bgr) -> preds; runs over the val images of the manifest"""
    import NERO_V347_Yolo_Perception as YP
    cv2 = YP._cv2()
    samples = []
    val = [m for m in manifest["items"] if m["split"] == "val"]
    for k, m in enumerate(val):
        bgr = cv2.imread(m["image"])
        if bgr is None:
            continue
        t0 = time.time()
        preds = predict(bgr)
        ms = (time.time() - t0) * 1000.0
        samples.append({"gts": m["anns"], "preds": preds, "hw": bgr.shape[:2], "ms": ms if k > 0 else None})
    return evaluate(samples, classes, conf_thr)


# ============================================================================================ child job
def _api():
    fake = os.environ.get("NERO_V381_FAKE_ULTRALYTICS")       # tests only: a file providing YOLOE + the trainer
    if fake:
        import importlib.util
        spec = importlib.util.spec_from_file_location("v381_fake_ultralytics", fake)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.YOLOE, mod.YOLOEPESegTrainer
    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer
    return YOLOE, YOLOEPESegTrainer


def _device():
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _num(x):
    try:
        if hasattr(x, "detach"):
            x = x.detach().cpu()
        if hasattr(x, "sum") and hasattr(x, "numel") and x.numel() > 1:
            x = x.sum()
        return round(float(x), 5)
    except Exception:
        return None


def run_job(job_path, api=None):
    """the child process: train -> evaluate original + new on the val set -> eval.json"""
    with open(job_path, encoding="utf-8") as f:
        job = json.load(f)
    run = job["run_dir"]
    prog_path = os.path.join(run, "progress.jsonl")

    def emit(**kw):
        kw["t"] = round(time.time(), 2)
        with open(prog_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kw, ensure_ascii=False) + "\n")

    try:
        YOLOE, Trainer = api or _api()
        device = job.get("device") or _device()
        emit(stage="start", device=device, mode=job["mode"], pid=os.getpid())
        model = YOLOE(job["base_model"])
        kw = dict(data=job["data"], epochs=int(job["epochs"]), batch=int(job["batch"]), imgsz=int(job["imgsz"]),
                  optimizer="AdamW", lr0=float(job["lr0"]), warmup_bias_lr=0.0, weight_decay=0.025, momentum=0.9,
                  close_mosaic=min(10, int(job["epochs"]) // 4), workers=int(job.get("workers", 2)), device=device,
                  project=run, name="train", exist_ok=True, patience=int(job.get("patience", 15)),
                  trainer=Trainer, plots=True, verbose=False)
        if job["mode"] == "quick":
            freeze, branches = linear_probe_freeze(model.model.model)
            kw["freeze"] = freeze
            emit(stage="freeze", branches=branches, frozen=len(freeze),
                 note="" if branches else "head has no *cv3 branch: only the backbone is frozen")
        stop_file = os.path.join(run, "STOP")

        def on_epoch(trainer):
            m = {k: _num(v) for k, v in (getattr(trainer, "metrics", None) or {}).items()}
            emit(stage="epoch", epoch=int(trainer.epoch) + 1, epochs=int(trainer.epochs), metrics=m,
                 loss=_num(getattr(trainer, "tloss", None)))
            if os.path.exists(stop_file):
                trainer.stop = True
                emit(stage="stopping")

        model.add_callback("on_fit_epoch_end", on_epoch)
        model.train(**kw)
        wdir = os.path.join(run, "train", "weights")
        best = next((os.path.join(wdir, n) for n in ("best.pt", "last.pt") if os.path.isfile(os.path.join(wdir, n))),
                    None)
        if not best:
            raise RuntimeError(f"训练结束但没有找到模型文件（{wdir}）")
        emit(stage="trained", best=best)
        with open(job["manifest"], encoding="utf-8") as f:
            manifest = json.load(f)
        classes, conf = job["classes"], float(job["conf"])
        emit(stage="eval", which="new")
        ft = YOLOE(best)
        names = getattr(ft, "names", None) or {i: c for i, c in enumerate(classes)}

        def pred_new(bgr):
            r = ft.predict(bgr, conf=EVAL_LOW_CONF, imgsz=int(job.get("eval_imgsz", job["imgsz"])), verbose=False,
                           device=device)
            return result_to_preds(r[0], names)
        ev_new = eval_model(pred_new, manifest, classes, conf)
        emit(stage="eval", which="base")
        base = YOLOE(job["base_model"])
        base.set_classes(list(classes))                    # original model = text prompts (as the Camera tab)

        def pred_base(bgr):
            r = base.predict(bgr, conf=EVAL_LOW_CONF, imgsz=int(job.get("eval_imgsz", job["imgsz"])), verbose=False,
                             device=device)
            return result_to_preds(r[0], {i: c for i, c in enumerate(classes)})
        ev_base = eval_model(pred_base, manifest, classes, conf)
        verdict = compare(ev_base, ev_new)
        res = {"version": VERSION, "best": best, "base": ev_base, "new": ev_new, "verdict": verdict,
               "job": dict({k: job[k] for k in ("mode", "epochs", "batch", "imgsz", "classes", "conf", "base_model")},
                           eval_imgsz=job.get("eval_imgsz", job["imgsz"]))}
        with open(os.path.join(run, "eval.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        emit(stage="done", verdict=verdict["level"])
        return res
    except Exception as exc:
        emit(stage="error", error=f"{type(exc).__name__}: {exc}"[:500], trace=traceback.format_exc()[-2000:])
        return None


# ============================================================================================ parent side
def pid_alive(pid):
    """alive and not a zombie (a finished child of this process is reaped here)"""
    if not pid:
        return False
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    except OSError:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return True


class TrainJob:
    """a running / finished child; events from progress.jsonl"""

    def __init__(self, run_dir, proc=None, pid=None):
        self.run_dir, self.proc, self.pid = run_dir, proc, pid or (proc.pid if proc else None)
        self._off = 0
        self.stop_t = None
        self.events = []

    @classmethod
    def start(cls, job, python=None, env=None, cwd=None):
        run = job["run_dir"]
        os.makedirs(run, exist_ok=True)
        jp = os.path.join(run, "job.json")
        with open(jp, "w", encoding="utf-8") as f:
            json.dump(job, f, ensure_ascii=False, indent=1)
        log = open(os.path.join(run, "train.log"), "ab")
        proc = subprocess.Popen([python or sys.executable, os.path.join(HERE, "NERO_V381_Yolo_Train.py"), "job", jp],
                                cwd=cwd or run, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        with open(os.path.join(run, "pid"), "w") as f:
            f.write(str(proc.pid))
        return cls(run, proc)

    def running(self):
        if self.proc is not None:
            return self.proc.poll() is None
        return pid_alive(self.pid)

    def poll(self):
        p = os.path.join(self.run_dir, "progress.jsonl")
        new = []
        try:
            with open(p, encoding="utf-8") as f:
                f.seek(self._off)
                for line in f:
                    if line.endswith("\n"):
                        try:
                            new.append(json.loads(line))
                        except Exception:
                            pass
                        self._off += len(line.encode("utf-8"))
        except FileNotFoundError:
            pass
        self.events += new
        if self.stop_t and self.running() and time.time() - self.stop_t > STOP_GRACE_S:
            self.kill()
        return new

    def stop(self):
        open(os.path.join(self.run_dir, "STOP"), "w").close()
        self.stop_t = time.time()

    def kill(self):
        try:
            os.killpg(os.getpgid(self.pid), signal.SIGTERM)
        except Exception:
            pass


def find_unfinished(runs_root):
    """a run whose child is still alive (the lab was closed while training) -> TrainJob or None"""
    if not os.path.isdir(runs_root):
        return None
    for d in sorted(os.listdir(runs_root), reverse=True):
        run = os.path.join(runs_root, d)
        try:
            pid = int(open(os.path.join(run, "pid")).read().strip())
        except Exception:
            continue
        j = TrainJob(run, pid=pid)
        if j.running():
            return j
    return None


def save_model(run_dir, name, note="", synonyms=None):
    """copy the trained model + its card into finetune_dir()/<name>/ -> folder"""
    with open(os.path.join(run_dir, "eval.json"), encoding="utf-8") as f:
        ev = json.load(f)
    dest = os.path.join(finetune_dir(), safe_name(name))
    if os.path.exists(dest):
        raise FileExistsError(f"「{safe_name(name)}」已经存在，换个名字")
    os.makedirs(dest)
    shutil.copy2(ev["best"], os.path.join(dest, safe_name(name) + ".pt"))
    for rel in ("eval.json", "dataset/data.yaml", "dataset/manifest.json", "train/results.csv", "train/results.png",
                "train/args.yaml"):
        src = os.path.join(run_dir, rel)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dest, os.path.basename(rel)))
    card = {"version": VERSION, "name": name, "file": safe_name(name) + ".pt", "classes": ev["job"]["classes"],
            "mode": ev["job"]["mode"], "base_model": os.path.basename(ev["job"]["base_model"]),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"), "verdict": ev["verdict"]["level"],
            "overall_base": ev["base"]["overall"], "overall_new": ev["new"]["overall"], "note": note,
            "synonyms": {c: list(v) for c, v in (synonyms or {}).items() if c in ev["job"]["classes"]},
            "use": "fixed classes: never call set_classes on this model (NERO_V347 treats it like a -pf model)"}
    with open(os.path.join(dest, "model_card.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, ensure_ascii=False, indent=1)
    return dest


def list_saved():
    d = finetune_dir()
    out = []
    if os.path.isdir(d):
        for n in sorted(os.listdir(d)):
            try:
                with open(os.path.join(d, n, "model_card.json"), encoding="utf-8") as f:
                    c = json.load(f)
            except Exception:
                continue
            if c.get("name") and c.get("file") and os.path.isfile(os.path.join(d, n, c["file"])):   # V3.82: valid only
                out.append(c)
    return out


def discard_run(run_dir, runs_root):
    if os.path.abspath(run_dir).startswith(os.path.abspath(runs_root) + os.sep) and os.path.isdir(run_dir):
        shutil.rmtree(run_dir, ignore_errors=True)
        return True
    return False


# ============================================================================================ GUI page ⑤
LEVEL_COLOUR = {"better": "#060", "same": "#a60", "worse": "#a00"}


class TrainPage:
    """⑤ 训练 / 评估 inside the camera lab: four steps top to bottom, each enabled only when the previous is done"""

    def __init__(self, lab):
        self.lab, self.tk, self.ttk, self.log = lab, lab.tk, lab.ttk, lab.log
        tk, ttk = self.tk, self.ttk
        self.plan = None
        self.job = None
        self.result = None
        self.t_start = None
        self.cls_rows = []
        p = ttk.Frame(lab.nb, padding=6)
        lab.nb.add(p, text="⑤ 训练 / 评估")
        # ---- 1 data
        f1 = ttk.LabelFrame(p, text="1 选择训练数据", padding=4)
        f1.pack(fill="x")
        self.v_root = lab._var("")
        ttk.Label(f1, textvariable=self.v_root, foreground="#555").pack(anchor="w")
        r = ttk.Frame(f1)
        r.pack(fill="x")
        self.lb = tk.Listbox(r, selectmode="multiple", height=6, width=64, exportselection=False)
        self.lb.pack(side="left")
        self.lb.bind("<<ListboxSelect>>", lambda _e: self._data_changed())
        rr = ttk.Frame(r)
        rr.pack(side="left", padx=8, anchor="n")
        self.v_reviewed = tk.BooleanVar(value=True)
        ttk.Checkbutton(rr, text="只用已复核的图（推荐）", variable=self.v_reviewed,
                        command=self._data_changed).pack(anchor="w")
        ttk.Button(rr, text="刷新列表", command=self.refresh).pack(anchor="w", pady=2)
        self.b_check = ttk.Button(rr, text="检查数据 →", command=self.on_check)
        self.b_check.pack(anchor="w", pady=2)
        self.v_check = lab._var("在列表里选要训练的类别（可多选），再点「检查数据」")
        self.lb_check = ttk.Label(f1, textvariable=self.v_check, justify="left", wraplength=900)
        self.lb_check.pack(anchor="w", pady=(2, 0))
        # ---- 2 train
        f2 = ttk.LabelFrame(p, text="2 训练", padding=4)
        f2.pack(fill="x", pady=(6, 0))
        r2 = ttk.Frame(f2)
        r2.pack(fill="x")
        self.v_mode = lab._var("quick")
        for k in ("quick", "full"):
            ttk.Radiobutton(r2, text=MODES[k]["label"], value=k, variable=self.v_mode,
                            command=self._mode_changed).pack(side="left", padx=(0, 10))
        r3 = ttk.Frame(f2)
        r3.pack(fill="x", pady=2)
        self.v_epochs, self.v_batch, self.v_imgsz = (lab._var(MODES["quick"][k]) for k in ("epochs", "batch", "imgsz"))
        for lab_t, var in (("轮数", self.v_epochs), ("批大小", self.v_batch), ("图像尺寸", self.v_imgsz)):
            ttk.Label(r3, text=lab_t).pack(side="left")
            ttk.Entry(r3, textvariable=var, width=5).pack(side="left", padx=(2, 10))
        ttk.Label(r3, text="（一般不用改；1280 = 和 Camera 页识别时一样，小物体更准；显存不够时把批大小减半）",
                  foreground="#555").pack(side="left")
        r4 = ttk.Frame(f2)
        r4.pack(fill="x", pady=2)
        self.b_train = ttk.Button(r4, text="▶ 开始训练", command=self.on_train)
        self.b_train.pack(side="left")
        self.b_stop = ttk.Button(r4, text="■ 停止（用目前最好的结果）", command=self.on_stop)
        self.b_stop.pack(side="left", padx=6)
        self.pb = ttk.Progressbar(f2, length=600, mode="determinate", maximum=100)
        self.pb.pack(anchor="w", pady=2)
        self.v_prog = lab._var("训练时显卡被占用，Camera 页的实时识别会变慢；关掉这个窗口训练也会在后台继续")
        ttk.Label(f2, textvariable=self.v_prog, justify="left").pack(anchor="w")
        # ---- 3 eval
        f3 = ttk.LabelFrame(p, text="3 评估：原模型 vs 新模型（同一份验证集，训练时没见过）", padding=4)
        f3.pack(fill="x", pady=(6, 0))
        cols = ("cls", "n", "rec", "fp", "ap", "iou")
        self.tv = ttk.Treeview(f3, columns=cols, show="headings", height=5, style=lab._tree_style())
        for c, t, w in zip(cols, ("类别", "验证框", "识别率 % 原→新", "误检/张 原→新", "mAP50 原→新", "轮廓重合 新"),
                           (150, 70, 150, 150, 150, 110)):
            self.tv.heading(c, text=t)
            self.tv.column(c, width=w, anchor="w")
        self.tv.pack(fill="x")
        self.v_verdict = lab._var("")
        self.lb_verdict = ttk.Label(f3, textvariable=self.v_verdict, justify="left", wraplength=900)
        self.lb_verdict.pack(anchor="w", pady=(2, 0))
        self.b_open = ttk.Button(f3, text="打开结果文件夹", command=self.on_open_folder)
        self.b_open.pack(anchor="w", pady=2)
        # ---- 4 save
        f4 = ttk.LabelFrame(p, text="4 保存", padding=4)
        f4.pack(fill="x", pady=(6, 0))
        r5 = ttk.Frame(f4)
        r5.pack(fill="x")
        self.v_name = lab._var("")
        ttk.Label(r5, text="模型名称").pack(side="left")
        ttk.Entry(r5, textvariable=self.v_name, width=28).pack(side="left", padx=4)
        self.b_save = ttk.Button(r5, text="✓ 保存这个模型", command=self.on_save)
        self.b_save.pack(side="left", padx=4)
        self.b_discard = ttk.Button(r5, text="✗ 丢弃这次训练", command=self.on_discard)
        self.b_discard.pack(side="left", padx=4)
        self.v_saved = lab._var("")
        ttk.Label(f4, textvariable=self.v_saved, justify="left", foreground="#335", wraplength=900).pack(anchor="w")
        # V3.82 APPLY FINE-TUNED + 1280 TRAINING: saved models -> Camera tab (and so the real task)
        f5 = ttk.LabelFrame(p, text="5 已保存的模型 → 用到 Camera 页和实机任务", padding=4)
        f5.pack(fill="x", pady=(6, 0))
        r6 = ttk.Frame(f5)
        r6.pack(fill="x")
        self.v_pick = lab._var("")
        self.cb_saved = ttk.Combobox(r6, textvariable=self.v_pick, width=34, state="readonly",
                                     postcommand=self._refresh_saved)
        self.cb_saved.pack(side="left")
        self.cb_saved.bind("<<ComboboxSelected>>", lambda _e: self._show_card())
        ttk.Button(r6, text="应用到 Camera 页", command=self.on_apply_model).pack(side="left", padx=6)
        ttk.Button(r6, text="恢复原模型", command=self.on_restore_model).pack(side="left")
        self.v_card = lab._var("")
        ttk.Label(f5, textvariable=self.v_card, justify="left", wraplength=900).pack(anchor="w")
        self.v_active = lab._var("")
        ttk.Label(f5, textvariable=self.v_active, justify="left", foreground="#335").pack(anchor="w")
        ttk.Label(f5, text="应用后 Camera 页的模型下拉框会切到它，实机任务的识别用的就是 Camera 页的设置。精调模型只认训练过的类别；"
                           "要找的物体不在里面时，会自动用原模型找（日志里看得到）。",
                  foreground="#555", justify="left", wraplength=900).pack(anchor="w")
        self._refresh_saved()
        self.refresh()
        j = find_unfinished(self.runs_root())
        if j is not None:
            self.job, self.t_start = j, time.time()
            self.log(f"[INFO] [V3.81 train] 发现后台还在训练：{os.path.basename(j.run_dir)}，接着显示进度")
        self._buttons()

    # ------------------------------------------------------------------ helpers
    def runs_root(self):
        return os.path.join(self.lab.ds.root, "_runs")

    def _set_state(self, w, on):
        try:
            w.configure(state="normal" if on else "disabled")
        except Exception:
            pass

    def _buttons(self):
        busy = self.job is not None and self.job.running()
        self._set_state(self.b_check, not busy)
        self._set_state(self.b_train, (not busy) and bool(self.plan and self.plan["ok"]))
        self._set_state(self.b_stop, busy)
        done = self.result is not None and not busy
        self._set_state(self.b_save, done)
        self._set_state(self.b_discard, (not busy) and self.job is not None)
        self._set_state(self.b_open, self.job is not None)

    def _colour(self, w, c):
        try:
            w.configure(foreground=c)
        except Exception:
            pass

    def refresh(self):
        root = self.lab.ds.root
        self.v_root.set(f"数据文件夹：{root}（在「④ 采集标注」里改）")
        self.items = scan(root)
        t = class_table(self.items)
        keep = set(self.selected())
        self.cls_rows = sorted(t)
        try:
            self.lb.delete(0, "end")
            for i, c in enumerate(self.cls_rows):
                v = t[c]
                self.lb.insert("end", f"{c}   已复核 {v['reviewed']}/{v['images']} 张 · 训练 {v['train']} / 验证 {v['val']} · "
                                      f"{v['objects']} 框")
                if c in keep:
                    self.lb.selection_set(i)
        except Exception:
            pass
        self._data_changed()

    def selected(self):
        try:
            return [self.cls_rows[int(i)] for i in self.lb.curselection() if int(i) < len(self.cls_rows)]
        except Exception:
            return []

    def _data_changed(self):
        self.plan = None
        self._buttons()

    def _mode_changed(self):
        m = MODES[self.v_mode.get()]
        self.v_epochs.set(str(m["epochs"]))
        self.v_batch.set(str(m["batch"]))
        self.v_imgsz.set(str(m["imgsz"]))

    # ------------------------------------------------------------------ 1
    def on_check(self, classes=None):
        classes = classes if classes is not None else self.selected()
        self.plan = plan_dataset(self.items, classes, bool(self.v_reviewed.get()))
        pl = self.plan
        lines = []
        if pl["ok"]:
            lines.append("✓ 可以训练：" + "；".join(f"{c} 训练 {v['train_images']} 张 / 验证 {v['val_objects']} 框"
                                                for c, v in pl["per_class"].items()))
        lines += ["✗ " + e for e in pl["errors"]] + ["• " + w for w in pl["warnings"]]
        self.v_check.set("\n".join(lines))
        self._colour(self.lb_check, "#060" if pl["ok"] and not pl["warnings"] else ("#a60" if pl["ok"] else "#a00"))
        if pl["ok"] and not self.v_name.get().strip():
            self.v_name.set(f"{'_'.join(safe_name(c) for c in classes)[:40]}_{time.strftime('%m%d')}")
        self._buttons()
        return pl

    # ------------------------------------------------------------------ 2
    def on_train(self):
        if not (self.plan and self.plan["ok"]):
            self.log("[WARN] [V3.81 train] 先「检查数据」")
            return
        try:
            ep = int(float(self.v_epochs.get()))
            bs = int(float(self.v_batch.get()))
            sz = int(float(self.v_imgsz.get()))
            assert 1 <= ep <= 500 and 1 <= bs <= 128 and 160 <= sz <= 1600 and sz % 32 == 0
        except Exception:
            self.log("[WARN] [V3.81 train] 轮数 1–500、批大小 1–128、图像尺寸 160–1600 且是 32 的倍数")
            return
        import NERO_V347_Yolo_Perception as YP
        tab = self.lab.tab
        base = YP.find_model(tab.v_ydir.get(), YP.DEFAULT_MODEL)
        if not base or YP.is_pf_model(base):
            self.log(f"[FAIL] [V3.81 train] 找不到原模型 {YP.DEFAULT_MODEL}（精调要用文字提示版 yoloe-26s-seg.pt）")
            return
        try:
            conf = float(tab.v_yconf.get())
        except Exception:
            conf = YP.CONF_MIN
        run = os.path.join(self.runs_root(), time.strftime("%Y%m%d_%H%M%S") + "_" + safe_name(self.v_name.get())[:30])
        mode = self.v_mode.get()
        try:
            data = export_dataset(self.plan, os.path.join(run, "dataset"))
        except Exception as exc:
            self.log(f"[FAIL] [V3.81 train] 生成训练集失败：{exc}")
            return
        job = {"version": VERSION, "run_dir": run, "data": data, "manifest": os.path.join(run, "dataset", "manifest.json"),
               "base_model": base, "mode": mode, "epochs": ep, "batch": bs, "imgsz": sz, "lr0": MODES[mode]["lr0"],
               "eval_imgsz": int(YP.IMGSZ),
               "classes": self.plan["classes"], "conf": conf, "patience": 15, "workers": 2}
        try:
            import NERO_V313_Real_Robot as RR
            env = RR.clean_child_env()
        except Exception:
            env = dict(os.environ)
        self.result = None
        self.tv.delete(*self.tv.get_children())
        self.v_verdict.set("")
        self.job = TrainJob.start(job, env=env, cwd=os.path.dirname(base))   # cwd: the text encoder lives there
        self.t_start = time.time()
        self.pb.configure(value=0)
        self.v_prog.set(f"准备中（加载模型、读数据）… 训练集 {os.path.join(run, 'dataset')}")
        self.log(f"[INFO] [V3.81 train] 开始训练 {', '.join(job['classes'])}（{MODES[mode]['label']}，{ep} 轮）→ {run}")
        self._buttons()

    def on_stop(self):
        if self.job is not None and self.job.running():
            self.job.stop()
            self.v_prog.set("正在停止：这一轮结束后停下，再用目前最好的结果做评估…")
            self.log("[INFO] [V3.81 train] 已请求停止（本轮结束后停）")

    def poll(self):
        if self.job is None:
            return
        for ev in self.job.poll():
            self._event(ev)
        if not self.job.running() and self.result is None and not getattr(self, "_ended", False):
            evs = [e["stage"] for e in self.job.events]
            if "done" not in evs and "error" not in evs:
                self._ended = True
                self.v_prog.set(f"训练进程意外结束：看 {os.path.join(self.job.run_dir, 'train.log')}")
                self.log(f"[FAIL] [V3.81 train] 训练进程意外结束（日志 {os.path.join(self.job.run_dir, 'train.log')}）")
        self._buttons()

    def _event(self, ev):
        st = ev.get("stage")
        if st == "start":
            self._ended = False
            self.v_prog.set(f"已启动（{'GPU' if ev.get('device') != 'cpu' else 'CPU（会很慢）'}）…")
        elif st == "freeze":
            if not ev.get("branches"):
                self.log(f"[WARN] [V3.81 train] {ev.get('note')}")
        elif st == "epoch":
            e, n = ev["epoch"], ev["epochs"]
            self.pb.configure(value=100.0 * e / max(1, n))
            el = time.time() - (self.t_start or time.time())
            left = el / max(1, e) * (n - e)
            m = ev.get("metrics") or {}
            map50 = m.get("metrics/mAP50(M)", m.get("metrics/mAP50(B)"))
            self.v_prog.set(f"第 {e}/{n} 轮 · 已用 {el / 60:.1f} min · 剩余约 {left / 60:.1f} min"
                            + (f" · 验证 mAP50 {map50:.3f}" if isinstance(map50, (int, float)) else ""))
        elif st == "stopping":
            self.v_prog.set("这一轮结束，正在停止…")
        elif st == "trained":
            self.pb.configure(value=100)
            self.v_prog.set("训练完成，正在评估原模型和新模型…")
        elif st == "eval":
            self.v_prog.set(f"评估{'新' if ev.get('which') == 'new' else '原'}模型…")
        elif st == "done":
            self.load_result(self.job.run_dir)
        elif st == "error":
            self.v_prog.set(f"出错：{ev.get('error')}")
            self.log(f"[FAIL] [V3.81 train] {ev.get('error')}（详细：{os.path.join(self.job.run_dir, 'train.log')}）")

    # ------------------------------------------------------------------ 3
    def load_result(self, run_dir):
        with open(os.path.join(run_dir, "eval.json"), encoding="utf-8") as f:
            res = json.load(f)
        self.result = res
        b, n = res["base"], res["new"]
        self.tv.delete(*self.tv.get_children())
        fm = lambda v: "-" if v is None else f"{v:g}"
        for c, v in n["per_class"].items():
            bv = b["per_class"].get(c, {})
            self.tv.insert("", "end", values=(c, v["n_gt"], f"{fm(bv.get('recall'))} → {fm(v['recall'])}",
                                              f"{fm(bv.get('fp_per_img'))} → {fm(v['fp_per_img'])}",
                                              f"{fm(bv.get('ap50'))} → {fm(v['ap50'])}", fm(v["mask_iou"])))
        ob, on = b["overall"], n["overall"]
        self.tv.insert("", "end", values=("（平均）", on["objects"], f"{fm(ob['recall'])} → {fm(on['recall'])}",
                                          f"{fm(ob['fp_per_img'])} → {fm(on['fp_per_img'])}",
                                          f"{fm(ob['ap50'])} → {fm(on['ap50'])}", fm(on["mask_iou"])))
        vd = res["verdict"]
        self.v_verdict.set("\n".join(vd["lines"]))
        self._colour(self.lb_verdict, LEVEL_COLOUR.get(vd["level"], "#333"))
        self.v_prog.set("完成：看下面的评估，满意就保存")
        lv = {"better": "OK", "same": "INFO", "worse": "WARN"}[vd["level"]]
        self.log(f"[{lv}] [V3.81 train] {vd['lines'][0]}")
        self._buttons()

    def on_open_folder(self):
        if self.job is None:
            return
        try:
            import NERO_V313_Real_Robot as RR
            subprocess.Popen(["xdg-open", self.job.run_dir], env=RR.clean_child_env(), stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except Exception as exc:
            self.log(f"[WARN] [V3.81 train] 打不开文件夹 {self.job.run_dir}：{exc}")

    # ------------------------------------------------------------------ 4
    def on_save(self):
        if self.result is None or self.job is None:
            return
        name = self.v_name.get().strip()
        if not name:
            self.log("[WARN] [V3.81 train] 先填模型名称")
            return
        if self.result["verdict"]["level"] == "worse":
            self.log("[WARN] [V3.81 train] 评估显示变差了，仍按你的要求保存")
        try:
            dest = save_model(self.job.run_dir, name, synonyms=self._synonyms())
        except Exception as exc:
            self.log(f"[FAIL] [V3.81 train] 保存失败：{exc}")
            return
        self.v_saved.set(f"已保存 → {dest}\n在下面「已保存的模型」里选它 →「应用到 Camera 页」，实机任务同时生效；原模型不受影响。")
        self._refresh_saved(select=name)
        self.log(f"[OK] [V3.81 train] 模型已保存 → {dest}")
        discard_run(self.job.run_dir, self.runs_root())
        self.job, self.result = None, None
        self._buttons()

    # ------------------------------------------------------------------ 5 (V3.82)
    def _synonyms(self):
        cp = self.lab.st.get("class_prompts") or {}
        import NERO_V347_Yolo_Perception as YP
        return {c: YP.parse_prompts(v) for c, v in cp.items()}

    def _refresh_saved(self, select=None):
        self.saved = list_saved()
        names = [c["name"] for c in self.saved]
        try:
            self.cb_saved.configure(values=names)
        except Exception:
            pass
        if select:
            self.v_pick.set(select)
        elif self.v_pick.get() not in names:
            self.v_pick.set(names[-1] if names else "")
        self._show_card()
        self._show_active()

    def _card(self):
        return next((c for c in getattr(self, "saved", []) if c["name"] == self.v_pick.get()), None)

    def _show_card(self):
        c = self._card()
        if not c:
            self.v_card.set("还没有保存的模型" if not getattr(self, "saved", None) else "")
            return
        ob, on = c.get("overall_base") or {}, c.get("overall_new") or {}
        self.v_card.set(f"类别 {', '.join(c['classes'])} · {MODES.get(c.get('mode'), {}).get('label', c.get('mode'))} · "
                        f"{c.get('created')} · 识别率 {ob.get('recall')} → {on.get('recall')} % · mAP50 {ob.get('ap50')} → "
                        f"{on.get('ap50')} · 结论 {dict(better='有实质提升', same='没有实质提升', worse='变差了').get(c.get('verdict'), '?')}")

    def _model_ref(self, card):
        """the Camera tab's model entry for a saved model: relative to its model folder when inside it, else absolute"""
        path = os.path.join(finetune_dir(), safe_name(card["name"]), card["file"])
        try:
            root = os.path.abspath(os.path.expanduser(self.lab.tab.v_ydir.get()))
            rel = os.path.relpath(path, root)
            if not rel.startswith(".."):
                return rel.replace(os.sep, "/")
        except Exception:
            pass
        return path

    def _set_tab_model(self, ref):
        tab = self.lab.tab
        tab.v_ymodel.set(ref)
        cb = getattr(tab, "cb_ymodel", None)
        if cb is not None:
            try:
                import NERO_V347_Yolo_Perception as YP
                vals = list(YP.list_models(tab.v_ydir.get()))
                cb.configure(values=vals + ([ref] if ref not in vals else []))
            except Exception:
                pass
        self._show_active()

    def _show_active(self):
        try:
            cur = self.lab.tab.v_ymodel.get()
        except Exception:
            cur = "?"
        self.v_active.set(f"Camera 页现在用的模型：{cur}")

    def on_apply_model(self):
        c = self._card()
        if not c:
            self.log("[WARN] [V3.82 train] 先在列表里选一个保存的模型")
            return
        if c.get("verdict") == "worse":
            self.log("[WARN] [V3.82 train] 这个模型评估时变差了，仍按你的选择应用")
        ref = self._model_ref(c)
        self._set_tab_model(ref)
        self.log(f"[OK] [V3.82 train] Camera 页改用精调模型「{c['name']}」（{ref}，类别 {', '.join(c['classes'])}）；"
                 "实机任务的识别同时生效；不认识的物体自动用原模型")

    def on_restore_model(self):
        import NERO_V347_Yolo_Perception as YP
        self._set_tab_model(YP.DEFAULT_MODEL)
        self.log(f"[OK] [V3.82 train] Camera 页恢复原模型 {YP.DEFAULT_MODEL}")

    def on_discard(self):
        if self.job is None or self.job.running():
            return
        run = self.job.run_dir
        if discard_run(run, self.runs_root()):
            self.log(f"[INFO] [V3.81 train] 已丢弃这次训练（{os.path.basename(run)}）")
        self.job, self.result = None, None
        self.tv.delete(*self.tv.get_children())
        self.v_verdict.set("")
        self.v_prog.set("")
        self.pb.configure(value=0)
        self._buttons()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "job":
        res = run_job(argv[1])
        return 0 if res is not None else 1
    print(f"NERO {VERSION} YOLOE train: used by the camera lab (Camera tab -> 相机信息 -> 相机高级测试 -> ⑤)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
