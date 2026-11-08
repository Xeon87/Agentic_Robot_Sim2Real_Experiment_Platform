#!/usr/bin/env python3
"""NERO V3.84 Main-Camera Multi-Frame Tracker, Egg Carton 6-Slot Joint Perception & Table Plane Lock.

Provides:
1. MultiFrameTracker:
   - Associates 3D targets across frames by category and 3D distance (<= match_dist_mm) to maintain fixed IDs.
   - Exponential Moving Average (EMA) position/axis smoothing for static targets.
   - Instant reset when target jumps > jump_reset_mm (no lag when user moves an object).
   - Quantifies raw vs EMA jitter (mm) to prove stability gain.
   - Pre-grasp recheck gate: ok (<=3 mm) / adjust (3-15 mm) / abort (>15 mm or missing).

2. Joint Egg + Egg Carton 6-Slot Perception:
   - Runs combined model once with prompts for both egg and egg carton.
   - Splits detections into eggs and cartons; runs 3D ellipsoid fit on eggs and 2x3 OBB slot grid on carton.
   - Cross-checks 6 slot positions against 3D egg positions + local depth height to classify empty vs occupied slots.

3. Table Plane Reference Lock & Drift Monitor:
   - Saves/loads reference table plane (n, d, z_at_center) in NERO_CALIB_DIR/table_plane_lock.json.
   - Computes height drift (mm) and tilt angle (deg) against current measured plane and safety fence z_table.

4. Quantitative N-Frame A/B Benchmark:
   - Compares single-frame raw vs multi-frame EMA + depth median across N frames:
     detection rate, ID switches, raw 3D std (mm), tracked 3D std (mm), jitter reduction %, slot state consistency %.
"""

import json
import math
import os
import time
import numpy as np

# V3.84 OBJECT TRACKER + JOINT CARTON SLOTS + TABLE LOCK + BENCHMARK
VERSION = "V4.3"
MARKER_V384 = "V3.84 OBJECT TRACKER + JOINT CARTON SLOTS + TABLE LOCK"


def calib_dir():
    return os.path.expanduser(os.environ.get("NERO_CALIB_DIR") or "~/Downloads/cameracalib")


def table_lock_path():
    return os.path.join(calib_dir(), "table_plane_lock.json")


def _norm_cat(name):
    s = str(name or "").strip().lower()
    if any(k in s for k in ("carton", "holder", "tray", "box", "蛋托", "蛋盒")):
        return "carton"
    if any(k in s for k in ("egg", "white round", "鸡蛋")):
        return "egg"
    if s.startswith("slot_"):
        return "slot"
    return s or "object"


class MultiFrameTracker:
    """Multi-frame 3D object tracker in base_link frame."""

    def __init__(self, alpha=0.35, match_dist_mm=25.0, jump_reset_mm=15.0, max_missed=5, window=10):
        self.alpha = float(alpha)
        self.match_dist_m = float(match_dist_mm) / 1000.0
        self.jump_reset_m = float(jump_reset_mm) / 1000.0
        self.max_missed = int(max_missed)
        self.window = int(window)
        self.tracks = {}        # tid -> track dict
        self._counters = {}     # cat -> int
        self.frame_idx = 0

    def reset(self):
        self.tracks.clear()
        self._counters.clear()
        self.frame_idx = 0

    def _next_id(self, cat):
        c = self._counters.get(cat, 0) + 1
        self._counters[cat] = c
        return f"{cat}_{c}"

    @staticmethod
    def _std_3d_mm(pts):
        if len(pts) < 2:
            return 0.0
        a = np.asarray(pts, dtype=float)
        mean = a.mean(axis=0)
        dists = np.linalg.norm(a - mean, axis=1) * 1000.0
        return round(float(np.sqrt(np.mean(dists ** 2))), 2)

    def update(self, detections):
        """Update tracks with a list of 3D detection dicts (each having 'xyz_m', 'status', etc.).
        Returns a new list of detection dicts with fixed 'det_id', smoothed 'xyz_m', 'raw_xyz_m',
        'hits', 'raw_jitter_mm', 'jitter_mm', and 'jumped'."""
        self.frame_idx += 1
        valid_dets = []
        passthrough = []
        for d in (detections or []):
            if d.get("xyz_m") is not None and d.get("status") != "rejected" and not str(d.get("det_id", "")).startswith("slot_"):
                valid_dets.append(dict(d))
            else:
                passthrough.append(dict(d))

        # Group tracks and detections by category
        unmatched_tracks = set(self.tracks.keys())
        assigned = {}  # det_idx -> tid

        # Greedy nearest-neighbor matching within match_dist_m
        pairs = []
        for idx, d in enumerate(valid_dets):
            cat = _norm_cat(d.get("obj_name") or (d.get("yolo") or {}).get("name") or d.get("det_id") or "egg")
            pos = np.asarray(d["xyz_m"], dtype=float)
            for tid in unmatched_tracks:
                tr = self.tracks[tid]
                if tr["cat"] != cat:
                    continue
                dist = float(np.linalg.norm(pos - tr["xyz"]))
                # Allow matching up to max(match_dist_m, 2.5 * jump_reset_m) so a moved single object keeps its ID if unique,
                # or strictly match_dist_m when associating. Let's use match_dist_m (default 25mm, or up to 45mm if moved).
                if dist <= max(self.match_dist_m, 0.045):
                    pairs.append((dist, idx, tid))

        pairs.sort(key=lambda x: x[0])
        used_dets = set()
        for dist, idx, tid in pairs:
            if idx in used_dets or tid not in unmatched_tracks:
                continue
            assigned[idx] = tid
            used_dets.add(idx)
            unmatched_tracks.remove(tid)

        out = []
        for idx, d in enumerate(valid_dets):
            raw_xyz = np.asarray(d["xyz_m"], dtype=float)
            cat = _norm_cat(d.get("obj_name") or (d.get("yolo") or {}).get("name") or d.get("det_id") or "egg")

            if idx in assigned:
                tid = assigned[idx]
                tr = self.tracks[tid]
                step_m = float(np.linalg.norm(raw_xyz - tr["xyz"]))
                jumped = step_m > self.jump_reset_m
                if jumped:
                    # Object moved significantly: reset filter history immediately to avoid lag
                    tr["xyz"] = raw_xyz.copy()
                    tr["raw_hist"] = [raw_xyz.copy()]
                    tr["ema_hist"] = [raw_xyz.copy()]
                    tr["hits"] = 1
                else:
                    tr["xyz"] = self.alpha * raw_xyz + (1.0 - self.alpha) * tr["xyz"]
                    tr["raw_hist"].append(raw_xyz.copy())
                    tr["ema_hist"].append(tr["xyz"].copy())
                    if len(tr["raw_hist"]) > self.window:
                        tr["raw_hist"].pop(0)
                    if len(tr["ema_hist"]) > self.window:
                        tr["ema_hist"].pop(0)
                    tr["hits"] += 1
                tr["missed"] = 0
                tr["last_det"] = d
            else:
                tid = self._next_id(cat)
                jumped = False
                tr = {
                    "tid": tid,
                    "cat": cat,
                    "xyz": raw_xyz.copy(),
                    "raw_hist": [raw_xyz.copy()],
                    "ema_hist": [raw_xyz.copy()],
                    "hits": 1,
                    "missed": 0,
                    "last_det": d,
                }
                self.tracks[tid] = tr

            raw_std = self._std_3d_mm(tr["raw_hist"])
            ema_std = self._std_3d_mm(tr["ema_hist"])

            d_out = dict(d)
            d_out["det_id"] = tid
            d_out["raw_xyz_m"] = [round(float(v), 5) for v in raw_xyz]
            d_out["xyz_m"] = [round(float(v), 5) for v in tr["xyz"]]
            d_out["hits"] = tr["hits"]
            d_out["missed"] = 0
            d_out["raw_jitter_mm"] = raw_std
            d_out["jitter_mm"] = ema_std
            d_out["jumped"] = jumped
            notes = list(d_out.get("notes") or [])
            if jumped:
                notes.append("检测到位移→重置滤波")
            elif tr["hits"] >= 2:
                notes.append(f"跟踪 {tr['hits']} 帧 抖动 {ema_std:.1f}mm(原 {raw_std:.1f}mm)")
            d_out["notes"] = notes
            out.append(d_out)

        # Handle unmatched tracks
        for tid in list(unmatched_tracks):
            tr = self.tracks[tid]
            tr["missed"] += 1
            if tr["missed"] > self.max_missed:
                del self.tracks[tid]

        out.sort(key=lambda r: str(r.get("det_id", "")))
        return out + passthrough


def pregrasp_recheck(planned_xyz_m, current_detections, target_id=None, ok_tol_mm=3.0, adjust_max_mm=15.0):
    """Pre-grasp recheck gate at hover pose using main camera detections.
    Returns dict with:
      action: 'ok' (<=3mm) | 'adjust' (3..15mm) | 'abort' (>15mm or missing)
      shift_mm: 2D XY shift in mm
      shift_3d_mm: 3D shift in mm
      new_xyz_m: updated target xyz_m if matched
      reason: Chinese explanation
    """
    p_xyz = np.asarray(planned_xyz_m, dtype=float)
    cands = [d for d in (current_detections or [])
             if d.get("xyz_m") is not None and d.get("status") != "rejected"
             and not str(d.get("det_id", "")).startswith("slot_")]
    if not cands:
        return {
            "action": "abort",
            "shift_mm": None,
            "shift_3d_mm": None,
            "new_xyz_m": None,
            "reason": "抓前重检未识别到目标（可能被遮挡或滚出视野），中止下探并重规划",
        }

    # Prefer matching target_id if available and close, else nearest XY candidate
    best = None
    if target_id:
        for d in cands:
            if d.get("det_id") == target_id:
                best = d
                break
    if best is None:
        best = min(cands, key=lambda d: math.hypot(d["xyz_m"][0] - p_xyz[0], d["xyz_m"][1] - p_xyz[1]))

    c_xyz = np.asarray(best["xyz_m"], dtype=float)
    shift_xy_mm = round(float(math.hypot(c_xyz[0] - p_xyz[0], c_xyz[1] - p_xyz[1])) * 1000.0, 2)
    shift_3d_mm = round(float(np.linalg.norm(c_xyz - p_xyz)) * 1000.0, 2)
    new_xyz = [round(float(v), 5) for v in c_xyz]

    if shift_xy_mm <= ok_tol_mm:
        return {
            "action": "ok",
            "shift_mm": shift_xy_mm,
            "shift_3d_mm": shift_3d_mm,
            "new_xyz_m": new_xyz,
            "det_id": best.get("det_id"),
            "reason": f"抓前重检通过：水平偏差 {shift_xy_mm:.1f} mm ≤ {ok_tol_mm:.0f} mm，允许直接下探",
        }
    if shift_xy_mm <= adjust_max_mm:
        return {
            "action": "adjust",
            "shift_mm": shift_xy_mm,
            "shift_3d_mm": shift_3d_mm,
            "new_xyz_m": new_xyz,
            "det_id": best.get("det_id"),
            "reason": f"抓前重检发现微移：水平偏差 {shift_xy_mm:.1f} mm（{ok_tol_mm:.0f}–{adjust_max_mm:.0f} mm），先平移微调对准再下探",
        }
    return {
        "action": "abort",
        "shift_mm": shift_xy_mm,
        "shift_3d_mm": shift_3d_mm,
        "new_xyz_m": new_xyz,
        "det_id": best.get("det_id"),
        "reason": f"抓前重检偏差超限：水平偏移 {shift_xy_mm:.1f} mm > {adjust_max_mm:.0f} mm，中止下探并重新规划",
    }


def is_carton_det(d):
    name = str(d.get("name") or "").strip().lower()
    model = str(d.get("model") or "").strip().lower()
    if any(k in name for k in ("carton", "holder", "tray", "box", "container", "蛋托", "蛋盒")):
        return True
    if "carton" in model or "holder" in model:
        return True
    return False


def is_egg_det(d):
    if is_carton_det(d):
        return False
    name = str(d.get("name") or "").strip().lower()
    model = str(d.get("model") or "").strip().lower()
    if any(k in name for k in ("egg", "white round", "鸡蛋")) or "egg" in model:
        return True
    return d.get("cls", 0) == 0


def locate_eggs_and_carton(dets2d, rgb, depth_m, K, T_base_cam, table_z=None, fence_poly=None,
                           egg_size_mm=(55.0, 42.0), calib_note=None, tracker=None, agent_fn=None):
    """Jointly locate 3D eggs + egg carton 6 slots from combined YOLOE 2D detections.
    1. Splits dets2d into egg_dets and carton_dets;
    2. Runs YP.locate() on egg_dets -> 3D egg positions;
    3. Optionally runs MultiFrameTracker on eggs;
    4. Runs W49.locate_egg_holder() on carton_dets with existing_eggs=located_eggs -> holder + slot_1..slot_6;
    5. Returns (all_rows, combined_diag)."""
    import NERO_V347_Yolo_Perception as YP
    import NERO_V349_Real_Wrist_Camera as W49

    egg_dets = []
    carton_dets = []
    for d in (dets2d or []):
        if is_carton_det(d):
            dc = dict(d)
            dc["cls"] = 0
            carton_dets.append(dc)
        elif is_egg_det(d):
            de = dict(d)
            de["cls"] = 0
            egg_dets.append(de)

    egg_rows, egg_diag = YP.locate(
        egg_dets, depth_m, K, T_base_cam,
        table_z=table_z, fence_poly=fence_poly, size_mm=egg_size_mm, calib_note=calib_note
    )

    if tracker is not None:
        egg_rows = tracker.update(egg_rows)

    holder_rows, holder_diag = [], {"found": False}
    if carton_dets:
        holder_rows, holder_diag = W49.locate_egg_holder(
            carton_dets, rgb, depth_m, K, T_base_cam,
            table_z=table_z, fence_poly=fence_poly, calib_note=calib_note,
            existing_eggs=egg_rows, agent_fn=agent_fn
        )
        if tracker is not None and holder_rows:
            # Track the holder row itself while preserving slot rows
            h_only = [r for r in holder_rows if r.get("det_id") == "holder"]
            s_only = [r for r in holder_rows if r.get("det_id") != "holder"]
            if h_only:
                h_tracked = tracker.update(h_only)
                h_tracked[0]["det_id"] = "holder"
                holder_rows = h_tracked + s_only

    all_rows = list(egg_rows) + list(holder_rows)
    slots = [r for r in holder_rows if str(r.get("det_id", "")).startswith("slot_")]
    empty_slots = [s["det_id"] for s in slots if not s.get("occupied")]
    occupied_slots = [s["det_id"] for s in slots if s.get("occupied")]

    diag = {
        "n_2d": len(dets2d or []),
        "n_egg_2d": len(egg_dets),
        "n_carton_2d": len(carton_dets),
        "n_egg": egg_diag.get("n_egg", 0),
        "n_untrusted": egg_diag.get("n_untrusted", 0),
        "holder_found": bool(holder_diag.get("found")),
        "n_slots": len(slots),
        "n_empty": len(empty_slots),
        "n_occupied": len(occupied_slots),
        "empty_slots": empty_slots,
        "occupied_slots": occupied_slots,
        "yaw_deg": holder_diag.get("yaw_deg"),
        "ground_z_m": egg_diag.get("ground_z_m"),
        "ground_minus_z_table_mm": egg_diag.get("ground_minus_z_table_mm"),
        "egg_diag": egg_diag,
        "holder_diag": holder_diag,
    }
    return all_rows, diag


# ------------------------------------------------------------------ 3. Table Plane Lock & Drift Monitor
def fit_workspace_table_plane(depth_m, K, T_base_cam, z_table_hint=None):
    """Estimate current table plane (n, d, rms_mm, n_pts, z_center_m) from central workspace region."""
    import NERO_V347_Yolo_Perception as YP
    h, w = depth_m.shape
    T = np.asarray(T_base_cam, dtype=float)
    # Sample central 50% ROI of the image on a 4-px stride
    v0, v1 = int(h * 0.25), int(h * 0.85)
    u0, u1 = int(w * 0.20), int(w * 0.80)
    sub = depth_m[v0:v1:4, u0:u1:4]
    vv, uu = np.nonzero((sub > 0.25) & (sub < 1.50))
    if len(uu) < 60:
        return None
    zz = sub[vv, uu].astype(float)
    uu_f = (uu * 4 + u0).astype(float)
    vv_f = (vv * 4 + v0).astype(float)
    P = YP._pix_to_base(uu_f, vv_f, zz, K, T)
    if z_table_hint is not None:
        P = P[np.abs(P[:, 2] - float(z_table_hint)) < 0.08]
    if len(P) < 50:
        return None
    # Robust RANSAC / percentile trim near lowest surface (tabletop)
    z_med = float(np.percentile(P[:, 2], 35))
    inliers = P[np.abs(P[:, 2] - z_med) < 0.015]
    if len(inliers) < 40:
        inliers = P
    c = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - c, full_matrices=False)
    n = vt[-1]
    if n[2] < 0:
        n = -n
    d = float(n @ c)
    res = (inliers @ n) - d
    rms_mm = round(float(np.sqrt(np.mean(res ** 2))) * 1000.0, 2)
    # Height of plane at typical workspace center (x=0.35, y=0.0)
    z_center = float((d - n[0] * 0.35 - n[1] * 0.0) / n[2])
    return {
        "n": [round(float(x), 6) for x in n],
        "d": round(d, 6),
        "z_center_m": round(z_center, 5),
        "rms_mm": rms_mm,
        "n_pts": int(len(inliers)),
        "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": VERSION,
    }


def save_table_lock(plane_info, path=None):
    p = path or table_lock_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(plane_info, f, ensure_ascii=False, indent=2)
    return p


def load_table_lock(path=None):
    p = path or table_lock_path()
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if "n" in d and "d" in d and "z_center_m" in d:
            return d
    except Exception:
        pass
    return None


def check_table_drift(current_plane, locked_plane=None, z_table_fence=None, warn_mm=3.0):
    """Compare current measured table plane against locked reference plane and safety z_table."""
    if current_plane is None:
        return {"ok": False, "warn": True, "text": "桌面：未测到有效平面点云"}

    z_cur = float(current_plane["z_center_m"])
    n_cur = np.asarray(current_plane["n"], dtype=float)
    n_cur = n_cur / max(1e-9, np.linalg.norm(n_cur))

    parts = [f"当前桌面 z={z_cur * 1000:.1f} mm (RMS {current_plane.get('rms_mm', 0):.1f} mm)"]
    warn = False
    dz_lock_mm = None
    tilt_deg = None

    if locked_plane is not None:
        z_ref = float(locked_plane["z_center_m"])
        n_ref = np.asarray(locked_plane["n"], dtype=float)
        n_ref = n_ref / max(1e-9, np.linalg.norm(n_ref))
        dz_lock_mm = round((z_cur - z_ref) * 1000.0, 2)
        cos_ang = max(-1.0, min(1.0, float(np.dot(n_cur, n_ref))))
        tilt_deg = round(math.degrees(math.acos(cos_ang)), 2)
        parts.append(f"较锁定值 Δz={dz_lock_mm:+.1f} mm, 倾角差 {tilt_deg:.1f}°")
        if abs(dz_lock_mm) > warn_mm or tilt_deg > 2.0:
            warn = True

    dz_fence_mm = None
    if z_table_fence is not None:
        dz_fence_mm = round((z_cur - float(z_table_fence)) * 1000.0, 2)
        parts.append(f"较围栏 z_table 差 {dz_fence_mm:+.1f} mm")
        if abs(dz_fence_mm) > max(warn_mm, 5.0) and locked_plane is None:
            warn = True

    status = "⚠ 偏差超限（建议检查相机支架或重锁桌面）" if warn else "✓ 桌面基准正常"
    return {
        "ok": not warn,
        "warn": warn,
        "z_cur_m": z_cur,
        "dz_lock_mm": dz_lock_mm,
        "tilt_deg": tilt_deg,
        "dz_fence_mm": dz_fence_mm,
        "text": " · ".join(parts) + f" [{status}]",
    }


# ------------------------------------------------------------------ 4. Quantitative N-Frame A/B Benchmark
def run_ab_benchmark(frames_data, K, T_base_cam, table_z=None, fence_poly=None, egg_size_mm=(55.0, 42.0)):
    """Run quantitative A/B comparison across N frames.
    frames_data: list of dicts {'rgb', 'depths': [depth_1..depth_k], 'dets2d'}
    Compares:
      Baseline (A): single depth frame (depths[-1]), no tracker
      Upgraded (B): 5-frame median depth stack + MultiFrameTracker + joint carton slots
    Returns structured comparison dict and formatted summary text.
    """
    import NERO_V347_Yolo_Perception as YP
    import NERO_V378_D435i_Bench as B378

    tracker = MultiFrameTracker(alpha=0.35, match_dist_mm=25.0, jump_reset_mm=15.0)
    n_total = len(frames_data)
    if n_total == 0:
        return {"n_frames": 0, "summary_text": "无测试帧"}

    raw_positions_by_slot = []     # list of first-egg xyz in baseline
    ema_positions_by_id = {}       # tid -> list of ema xyz
    raw_positions_by_id = {}       # tid -> list of raw xyz
    baseline_id_list = []
    tracked_id_list = []
    baseline_egg_detected = 0
    upgraded_egg_detected = 0
    slot_occ_history = []          # list of tuple(occupied booleans)

    for item in frames_data:
        rgb = item.get("rgb")
        depths = item.get("depths") or []
        dets2d = item.get("dets2d") or []
        if not depths:
            continue

        # --- Baseline A: single frame depth, no tracker ---
        d_single = depths[-1]
        egg_dets = [dict(d, cls=0) for d in dets2d if is_egg_det(d)]
        res_a, _ = YP.locate(egg_dets, d_single, K, T_base_cam, table_z=table_z, fence_poly=fence_poly, size_mm=egg_size_mm)
        ok_a = [r for r in res_a if r.get("xyz_m") is not None and r.get("status") != "rejected"]
        if ok_a:
            baseline_egg_detected += 1
            raw_positions_by_slot.append(ok_a[0]["xyz_m"])
            baseline_id_list.append(ok_a[0]["det_id"])

        # --- Upgraded B: multi-frame median depth + tracker + joint carton ---
        d_med = B378.median_stack(depths) if len(depths) > 1 else depths[-1]
        rows_b, diag_b = locate_eggs_and_carton(
            dets2d, rgb, d_med, K, T_base_cam,
            table_z=table_z, fence_poly=fence_poly, egg_size_mm=egg_size_mm, tracker=tracker
        )
        ok_b = [r for r in rows_b if str(r.get("det_id", "")).startswith("egg_") and r.get("xyz_m") is not None]
        if ok_b:
            upgraded_egg_detected += 1
            tracked_id_list.append(ok_b[0]["det_id"])
            for r in ok_b:
                tid = r["det_id"]
                ema_positions_by_id.setdefault(tid, []).append(r["xyz_m"])
                raw_positions_by_id.setdefault(tid, []).append(r.get("raw_xyz_m") or r["xyz_m"])

        slots_b = [r for r in rows_b if str(r.get("det_id", "")).startswith("slot_")]
        if slots_b:
            slot_occ_history.append(tuple(bool(s.get("occupied")) for s in slots_b))

    # Compute metrics
    raw_std_mm = MultiFrameTracker._std_3d_mm(raw_positions_by_slot)
    ema_stds = [MultiFrameTracker._std_3d_mm(pts) for pts in ema_positions_by_id.values() if len(pts) >= 2]
    tracked_std_mm = round(float(np.mean(ema_stds)), 2) if ema_stds else 0.0

    reduction_pct = round((1.0 - tracked_std_mm / raw_std_mm) * 100.0, 1) if raw_std_mm > 1e-6 else 0.0

    # ID stability: count how many times the primary target changed its ID or ordering
    id_switches_b = sum(1 for i in range(1, len(tracked_id_list)) if tracked_id_list[i] != tracked_id_list[i - 1])

    # Slot consistency: fraction of frames matching the majority slot occupancy vector
    slot_consistency_pct = 100.0
    if len(slot_occ_history) >= 2:
        from collections import Counter
        most_common_occ, count = Counter(slot_occ_history).most_common(1)[0]
        slot_consistency_pct = round(100.0 * count / len(slot_occ_history), 1)

    summary = (
        f"[V3.84 对比测试 {n_total} 帧] "
        f"检出率：单帧 {baseline_egg_detected}/{n_total} vs 滤波跟踪 {upgraded_egg_detected}/{n_total} | "
        f"3D 位置抖动(std)：单帧原始 {raw_std_mm:.2f} mm → 多帧滤波+EMA {tracked_std_mm:.2f} mm（降低 {reduction_pct:+.1f}%）| "
        f"ID 跳变：{id_switches_b} 次 | "
        f"蛋托 6 孔状态一致率：{slot_consistency_pct:.0f}%"
    )

    return {
        "n_frames": n_total,
        "baseline_detected": baseline_egg_detected,
        "upgraded_detected": upgraded_egg_detected,
        "raw_std_mm": raw_std_mm,
        "tracked_std_mm": tracked_std_mm,
        "jitter_reduction_pct": reduction_pct,
        "id_switches": id_switches_b,
        "slot_consistency_pct": slot_consistency_pct,
        "summary_text": summary,
    }
