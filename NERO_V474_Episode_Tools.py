#!/usr/bin/env python3
"""NERO V4.7.4 EPISODE TOOLS - joint limits while recording / when saving / before replay, YOLOE prompts from the
task text, editing the task description of recorded episodes, renaming a dataset.

Why (user, 2026-11-10): an episode recorded in teach mode (zero-force drag, the controller does not stop the joint
at the URDF limit) had J7 at 96.2 deg; the replay failed half way with "joint path point 19: J7 96.2 deg outside
limit", and a policy trained on it would ask for an unreachable pose too.

Limits: the arm's URDF limits (the same as 实机NERO调试), "commandable" = limit - SOFT_MARGIN_DEG (3 deg, 「可设到」).
  recording: yellow within WARN_BAND_DEG of the commandable range, red (+ bell) outside it;
  saving: an episode with frames outside the commandable range gets limit_ok = false + the excursions; the list
          shows it red, the user is asked keep / delete, the LeRobot export skips it (unless told otherwise);
  replay: checked BEFORE anything moves; the log names joint, time span and amount.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

VERSION = "V4.7.4"
MARKER = "V4.7.4 JOINT LIMITS + TASK PROMPTS + EDIT TASK + RENAME DATASET"
# NERO URDF limits (rad) - the same as config/nero.urdf / NERO_V4_Control_Arbiter.JOINT_LIMITS_RAD
LIMITS_RAD: Tuple[Tuple[float, float], ...] = ((-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14),
                                               (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708))
SOFT_MARGIN_DEG = 3.0          # 「可设到」 = limit - 3 deg (NERO_V314 SOFT_MARGIN_DEG)
WARN_BAND_DEG = 7.0            # yellow this close to the commandable range (J7: from 80 deg)


# ============================================================================ joint limits
def soft_limits_deg(limits_rad: Optional[Sequence[Sequence[float]]] = None, margin: float = SOFT_MARGIN_DEG):
    lims = limits_rad or LIMITS_RAD
    return [(math.degrees(lo) + margin, math.degrees(hi) - margin) for lo, hi in list(lims)[:7]]


def check_q(q_rad: Sequence[float], limits_rad=None) -> List[Dict[str, Any]]:
    """live check of one joint vector -> [{joint (1-based), deg, level 'warn'|'over', lo, hi, over_deg}]"""
    out = []
    for j, ((lo, hi), q) in enumerate(zip(soft_limits_deg(limits_rad), list(q_rad)[:7])):
        d = math.degrees(float(q))
        over = max(lo - d, d - hi)
        if over > 0:
            out.append({"joint": j + 1, "deg": d, "level": "over", "lo": lo, "hi": hi, "over_deg": over})
        elif over > -WARN_BAND_DEG:
            out.append({"joint": j + 1, "deg": d, "level": "warn", "lo": lo, "hi": hi, "over_deg": over})
    return out


def live_text(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return ""
    worst = max(hits, key=lambda h: h["over_deg"])
    lim = worst["hi"] if worst["deg"] > 0 else worst["lo"]
    if worst["level"] == "over":
        return f"J{worst['joint']} {worst['deg']:+.1f}° 超出可执行范围（{lim:+.0f}°）{worst['over_deg']:.1f}° —— 往回带"
    return f"J{worst['joint']} {worst['deg']:+.1f}° 接近限位（可执行到 {lim:+.0f}°）"


def episode_limit_report(states, fps: float = 20.0, limits_rad=None, margin: float = SOFT_MARGIN_DEG) -> Dict[str, Any]:
    """states (n, >= 7) rad -> {"ok", "violations": [{joint, peak_deg, limit_deg, over_deg, t_from_s, t_to_s, frames}]}"""
    S = np.asarray(states, dtype=float)
    viol = []
    if S.ndim == 2 and S.shape[0] and S.shape[1] >= 7:
        D = np.degrees(S[:, :7])
        for j, (lo, hi) in enumerate(soft_limits_deg(limits_rad, margin)):
            bad = (D[:, j] > hi + 1e-6) | (D[:, j] < lo - 1e-6)
            if not bad.any():
                continue
            idx = np.nonzero(bad)[0]
            col = D[idx, j]
            k = int(np.argmax(np.maximum(col - hi, lo - col)))
            peak = float(col[k])
            lim = hi if peak > hi else lo
            viol.append({"joint": j + 1, "peak_deg": round(peak, 1), "limit_deg": round(lim, 1),
                         "over_deg": round(abs(peak - lim), 1), "t_from_s": round(float(idx[0]) / fps, 2),
                         "t_to_s": round(float(idx[-1]) / fps, 2), "frames": int(len(idx))})
    return {"ok": not viol, "violations": viol, "soft_margin_deg": float(margin)}


def report_text(rep: Dict[str, Any]) -> str:
    what = "可执行到" if float(rep.get("soft_margin_deg", SOFT_MARGIN_DEG)) > 0 else "限位"
    return "；".join(f"J{v['joint']} 最大 {v['peak_deg']:+.1f}°（{what} {v['limit_deg']:+.0f}°，超 {v['over_deg']:.1f}°）"
                    f"在 {v['t_from_s']:.1f}–{v['t_to_s']:.1f} s 共 {v['frames']} 帧" for v in rep.get("violations", []))


def backfill_limits(ds, limits_rad=None) -> List[int]:
    """episodes saved before V4.7.4 have no limit_ok: read only their joint array (not the video) and store the
    result in meta/episodes.jsonl -> indices that were checked"""
    rows = ds.list_episodes()
    todo = [r for r in rows if "limit_ok" not in r]
    if not todo:
        return []
    done = []
    for r in todo:
        p = ds._episode_npz_path(int(r["episode_index"]))
        try:
            with np.load(p, allow_pickle=False) as z:
                st = z["observation_state"]
            rep = episode_limit_report(st, float(r.get("fps") or ds.fps), limits_rad)
        except Exception:
            continue
        r["limit_ok"], r["limit_report"] = bool(rep["ok"]), rep
        done.append(int(r["episode_index"]))
    if done:
        _write_jsonl(ds._episodes_path(), rows)
    return done


# ============================================================================ YOLOE prompts from the task text
_DET = {"the", "a", "an", "each", "every", "all", "both", "another", "one", "two", "three"}
_STOP = {"into", "in", "on", "onto", "to", "from", "then", "and", "with", "of", "at", "inside", "next", "near",
         "by", "under", "over", "above", "below", "behind", "beside", "until", "after", "before", "using", "so",
         "that", "it", "them", "is", "are", "be", "while", "for", "out", "up", "down", "off", "back", "again",
         "away", "here", "there", "together", "aside", "carefully", "gently", "slowly", "quickly", "please"}
_POS = {"left", "right", "top", "bottom", "front", "middle", "center", "centre", "first", "second", "third",
        "last", "other", "same", "empty", "far", "near", "nearest", "closest", "upper", "lower"}
_ZH = [("红色", "red"), ("蓝色", "blue"), ("绿色", "green"), ("黄色", "yellow"), ("白色", "white"), ("黑色", "black"),
       ("鸡蛋托", "egg carton"), ("蛋托", "egg carton"), ("鸡蛋", "egg"), ("杯子", "cup"), ("方块", "block"),
       ("积木", "block"), ("盒子", "box"), ("碗", "bowl"), ("盘子", "plate"), ("瓶子", "bottle"), ("球", "ball")]


def _singular(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("xes", "ches", "shes", "sses")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us")):
        return w[:-1]
    return w


def task_prompts(task: str, fallback: Optional[List[str]] = None, max_n: int = 6) -> List[str]:
    """object names for YOLOE from the task text: 'Put the red block into the left box, then put the blue block
    into the right box' -> ['red block', 'blue block', 'box'] (position words dropped, colours kept)"""
    t = str(task or "")
    out: List[str] = []

    def add(p):
        p = p.strip()
        if p and p not in out:
            out.append(p)
    words = re.findall(r"[a-zA-Z]+", t.lower())
    i = 0
    while i < len(words):
        if words[i] in _DET:
            ph = []
            k = i + 1
            while k < len(words) and len(ph) < 3 and words[k] not in _STOP and words[k] not in _DET:
                ph.append(words[k])
                k += 1
            ph = [w for w in ph if w not in _POS]
            if ph:
                ph[-1] = _singular(ph[-1])
                add(" ".join(ph))
            i = k
        else:
            i += 1
    if not out:                                         # Chinese task text: colour + object words, in order
        cols = ("red", "blue", "green", "yellow", "white", "black")
        colours = [(m.start(), en) for cn, en in _ZH if en in cols for m in re.finditer(cn, t)]
        objs, taken = [], []
        for cn, en in _ZH:                              # longest names first in _ZH (蛋托 before 鸡蛋)
            if en in cols:
                continue
            for m in re.finditer(cn, t):
                if any(a <= m.start() < b for a, b in taken):
                    continue
                taken.append((m.start(), m.end()))
                pre = [c for p_, c in colours if 0 <= m.start() - p_ <= 3]
                objs.append((m.start(), (pre[-1] + " " + en) if pre else en))
        for _pos, name in sorted(objs):
            add(name)
    return out[:max_n] if out else list(fallback or [])


def parse_prompts(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"[,，;；、]", str(text or "")) if p.strip()]


# ============================================================================ edit task description
def _read_jsonl(p):
    out = []
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    out.append(json.loads(ln))
    return out


def _write_jsonl(p, rows):
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def backup_meta(ds, why: str = "edit") -> str:
    """copy meta/episodes.jsonl + meta/tasks.jsonl to meta/backup/<time>_<why>/ -> that folder"""
    d = os.path.join(ds.meta_dir, "backup", time.strftime("%Y%m%d_%H%M%S") + "_" + why)
    k = 1
    while os.path.exists(d):
        d = f"{d}_{k}"
        k += 1
    os.makedirs(d)
    for p in (ds._episodes_path(), ds._tasks_path(), ds._info_path()):
        if os.path.isfile(p):
            shutil.copy2(p, d)
    return d


def set_episode_task(ds, ep_indices: Sequence[int], new_task: str) -> Dict[str, Any]:
    """change the task text of the given episodes in meta/episodes.jsonl (+ tasks.jsonl) after a backup.
    -> {"ok", "backup", "changes": [(idx, old, new)], "npz": [paths to update in the background]}"""
    new_task = str(new_task or "").strip()
    if not new_task:
        return {"ok": False, "error": "新的任务描述是空的"}
    want = {int(i) for i in ep_indices}
    rows = ds.list_episodes()
    have = {int(r["episode_index"]) for r in rows}
    missing = sorted(want - have)
    if not want or missing:
        return {"ok": False, "error": f"没有这些 Episode：{missing}" if missing else "没有选 Episode"}
    bk = backup_meta(ds, "task")
    t_idx = ds._register_task(new_task)
    changes, npz = [], []
    for r in rows:
        i = int(r["episode_index"])
        if i not in want:
            continue
        old = (r.get("tasks") or [""])[0]
        r["tasks"] = [new_task]
        r["task_index"] = t_idx
        r["task_id"] = f"task_{t_idx}"
        r["task_edited"] = time.strftime("%Y-%m-%d %H:%M:%S")
        changes.append((i, old, new_task))
        npz.append(ds._episode_npz_path(i))
    _write_jsonl(ds._episodes_path(), rows)
    return {"ok": True, "backup": bk, "changes": changes, "npz": npz, "task_index": t_idx}


def update_npz_meta(npz_path: str, updates: Dict[str, Any], chunk: int = 1 << 22) -> Dict[str, Any]:
    """rewrite meta_json inside an episode .npz: every other member is streamed (decompress -> compress, never the
    whole array in memory) into a temp file that then replaces the original."""
    if not os.path.isfile(npz_path):
        return {"ok": False, "error": f"文件不在：{npz_path}"}
    tmp = npz_path + ".v474tmp"
    t0 = time.time()
    try:
        with zipfile.ZipFile(npz_path) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as dst:
            meta = {}
            if "meta_json.npy" in src.namelist():
                with src.open("meta_json.npy") as f:
                    meta = json.loads(str(np.lib.format.read_array(f)))
            meta.update(updates)
            for info in src.infolist():
                if info.filename == "meta_json.npy":
                    continue
                zi = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                zi.compress_type = zipfile.ZIP_DEFLATED
                with src.open(info) as fi, dst.open(zi, "w", force_zip64=True) as fo:
                    while True:
                        b = fi.read(chunk)
                        if not b:
                            break
                        fo.write(b)
            import io
            bio = io.BytesIO()
            np.lib.format.write_array(bio, np.array(json.dumps(meta, ensure_ascii=False)))
            dst.writestr(zipfile.ZipInfo("meta_json.npy", date_time=time.localtime()[:6]), bio.getvalue(),
                         compress_type=zipfile.ZIP_DEFLATED)
        os.replace(tmp, npz_path)
    except Exception as exc:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "seconds": round(time.time() - t0, 1)}


# ============================================================================ rename dataset
_NAME_OK = re.compile(r"^[A-Za-z0-9_.\-]+$")


def check_dataset_name(root: str, new: str) -> Optional[str]:
    """-> None when fine, else the reason"""
    new = str(new or "").strip()
    if not new:
        return "名字是空的"
    if not _NAME_OK.match(new) or new in (".", ".."):
        return "只能用英文字母、数字、_ - .（LeRobot 和训练脚本按文件夹名找数据集）"
    if os.path.exists(os.path.join(root, new)):
        return f"已经有同名的文件夹：{os.path.join(root, new)}"
    return None


def rename_dataset(root: str, old: str, new: str) -> Dict[str, Any]:
    """rename root/old -> root/new and fix the episode file paths stored in meta/episodes.jsonl"""
    why = check_dataset_name(root, new)
    if why:
        return {"ok": False, "error": why}
    src, dst = os.path.join(root, old), os.path.join(root, new)
    if not os.path.isdir(src):
        return {"ok": False, "error": f"数据集不在：{src}"}
    os.rename(src, dst)
    ep_p = os.path.join(dst, "meta", "episodes.jsonl")
    rows = _read_jsonl(ep_p)
    n_fix = 0
    for r in rows:
        f = r.get("file")
        if f:
            nf = os.path.join(dst, "data", os.path.basename(f))
            if nf != f:
                r["file"] = nf
                n_fix += 1
    if rows:
        _write_jsonl(ep_p, rows)
    return {"ok": True, "old": src, "new": dst, "paths_fixed": n_fix}
