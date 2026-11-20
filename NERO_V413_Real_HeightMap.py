#!/usr/bin/env python3
"""NERO V4.13 REAL HEIGHT MAP - the measured scene geometry H1 needs, from the real fixed D435i.

Until V4.12 the real general pick & place handed H1 only YOLOE boxes (centre, height, mask size); H1 then treated every
object as a solid box (G49.HeightMap.from_boxes): no cups / bowls with an opening, no cross sections, no side profile.
This module builds the same height map the simulation's "real-like" camera gives (NERO_V49_Geometry.MjProbe.
camera_heightmap, V4.9.2) from real data:

  depth frames (fused) + intrinsics K + hand-eye T_base_cam  ->  points in base_link
  YOLOE masks (dilated a little, the smaller mask wins where masks overlap)  ->  which object each point belongs to
  points of the robot itself (arm link capsules + gripper boxes)  ->  removed
  the table plane fitted through the unmasked points: heights are measured from the local table (a hand-eye error
  of 0.3 deg tilts the table by ~3 mm over its width; the tilt is reported)
  3 mm grid: highest point per cell and its owner; table-level points mark the cell as seen
  small holes filled from the neighbours; cells nobody saw (behind an object) get the camera line grazing the object in
  front (hm.Hu = the most hidden material there can reach), exactly as in the simulation
  high points outside every mask  ->  "unidentified_<n>" obstacles (connected cells >= 6), so they still block paths

Then the descriptors come from the SAME code as in the simulation (G49.describe + side profile from the measured
points + G49.bottoms_in_openings).  Pure numpy + OpenCV; no robot or camera access here (callers pass the data).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

VERSION = "V4.13"
MARKER = "V4.13 REAL HEIGHT MAP"
RES_M = 0.003
STRIDE_PX = 2
OBJ_MIN_H_M = 0.003            # a masked point this far above the table belongs to the object
FREE_MIN_H_M = 0.012           # an unmasked point must be this high to count as an (unidentified) obstacle
UNID_MIN_CELLS = 6
MASK_DILATE_PX = 3
ROBOT_LINK_RADIUS_M = 0.075
ROBOT_BOX_MARGIN_M = 0.020
DEFAULT_BOUNDS = (-0.80, -0.10, -0.45, 0.45)


def bounds_with_fence(fence_poly, base=DEFAULT_BOUNDS, limit=1.0):
    """the grid covers the default table area and the workspace fence polygon (clipped to +-limit m)"""
    x0, x1, y0, y1 = base
    try:
        P = np.asarray(fence_poly, float).reshape(-1, 2)
        if len(P) >= 3:
            x0, x1 = max(-limit, min(x0, float(P[:, 0].min()))), min(limit, max(x1, float(P[:, 0].max())))
            y0, y1 = max(-limit, min(y0, float(P[:, 1].min()))), min(limit, max(y1, float(P[:, 1].max())))
    except Exception:
        pass
    return (x0, x1, y0, y1)


def _cv2():
    import cv2
    return cv2


def backproject(depth, K, T_base_cam, stride=STRIDE_PX):
    """-> (P (N, 3) in base_link, (v, u) pixel indices of each point)"""
    d = np.asarray(depth, np.float32)
    H, W = d.shape[:2]
    vv, uu = np.mgrid[0:H:stride, 0:W:stride]
    z = d[vv, uu]
    ok = np.isfinite(z) & (z > 0.05)
    vv, uu, z = vv[ok], uu[ok], z[ok].astype(float)
    pc = np.stack([(uu - float(K["ppx"])) * z / float(K["fx"]), (vv - float(K["ppy"])) * z / float(K["fy"]), z], axis=1)
    T = np.asarray(T_base_cam, float)
    P = pc @ T[:3, :3].T + T[:3, 3]
    return P, (vv, uu)


def label_image(masks: Sequence[np.ndarray], shape, dilate_px=MASK_DILATE_PX):
    """-> int image (-1 = no mask): mask k = its index; larger masks painted first so a small object standing in a big
    one (an egg in a cup) keeps its own pixels"""
    L = np.full(shape[:2], -1, int)
    order = sorted(range(len(masks)), key=lambda k: -int(np.asarray(masks[k]).sum()))
    cv2 = _cv2() if dilate_px > 0 else None
    for k in order:
        m = np.asarray(masks[k]).astype(np.uint8)
        if m.shape[:2] != tuple(shape[:2]):
            continue
        if cv2 is not None:
            m = cv2.dilate(m, np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8))
        L[m > 0] = k
    for k in order:                          # the undilated masks win over a neighbour's dilated rim
        m = np.asarray(masks[k]).astype(bool)
        if m.shape[:2] == tuple(shape[:2]):
            L[m] = k
    return L


def robot_keep_mask(P, joint_origins: Sequence[np.ndarray], boxes=(), link_r=ROBOT_LINK_RADIUS_M,
                    box_margin=ROBOT_BOX_MARGIN_M):
    """True = keep (not the robot): capsules along the arm (consecutive joint origins) + inflated gripper boxes
    [(centre, R, half)]"""
    P = np.asarray(P, float)
    keep = np.ones(len(P), bool)
    pts = [np.asarray(o, float) for o in joint_origins if o is not None]
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        t = np.clip(((P - a) @ ab) / max(L2, 1e-12), 0.0, 1.0)
        dist = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        keep &= dist > link_r
    for c, R, h in boxes or ():
        Q = (P - np.asarray(c, float)) @ np.asarray(R, float)
        keep &= ~np.all(np.abs(Q) <= np.asarray(h, float) + box_margin, axis=1)
    return keep


def _fill_holes(H, own, known):
    """two passes, like the simulation's depth hole filter (NERO_V49_Geometry.MjProbe.camera_heightmap)"""
    for _ in range(2):
        Hn, On, Kn = H.copy(), own.copy(), known.copy()
        for iy, ix in zip(*np.nonzero(~known[1:-1, 1:-1])):
            iy, ix = iy + 1, ix + 1
            o = own[iy - 1:iy + 2, ix - 1:ix + 2]
            k = known[iy - 1:iy + 2, ix - 1:ix + 2]
            if k.sum() >= 5:
                vals = o[k]
                top = int(np.bincount(vals + 1).argmax()) - 1
                Kn[iy, ix] = True
                if top >= 0:
                    On[iy, ix] = top
                    Hn[iy, ix] = float(np.median(H[iy - 1:iy + 2, ix - 1:ix + 2][k & (o == top)]))
        H, own, known = Hn, On, Kn
    return H, own, known


def _shadow_bound(H, own, known, x0, y0, res, cam, table_z):
    """cells nobody saw: the camera line grazing the object in front of them (as MjProbe.camera_heightmap);
    vectorised over the unseen cells (same result as _shadow_bound_loop)"""
    ny, nx = H.shape
    Hu = np.full((ny, nx), -1.0)
    iy, ix = np.nonzero(~known)
    if not len(iy):
        return Hu
    c = np.asarray(cam, float)
    cxy, zc = c[:2], float(c[2])
    P = np.column_stack([x0 + (ix + 0.5) * res, y0 + (iy + 0.5) * res])
    D = cxy[None, :] - P
    Dp = np.linalg.norm(D, axis=1)
    U = D / np.maximum(Dp, 1e-9)[:, None]
    bound = np.full(len(P), -np.inf)
    alive = np.ones(len(P), bool)
    for s_ in np.arange(res, 0.30, res):
        Q = P + U * s_
        jx = np.floor((Q[:, 0] - x0) / res).astype(int)
        jy = np.floor((Q[:, 1] - y0) / res).astype(int)
        alive &= (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
        if not alive.any():
            break
        jxa, jya = np.where(alive, jx, 0), np.where(alive, jy, 0)
        hit = alive & known[jya, jxa] & (own[jya, jxa] >= 0)
        if hit.any():
            Dq = Dp - s_
            zl = zc - (zc - H[jya, jxa]) * Dp / np.maximum(Dq, 1e-6)
            bound = np.where(hit, np.maximum(bound, zl), bound)
    ok = np.isfinite(bound) & (bound > table_z)
    Hu[iy[ok], ix[ok]] = bound[ok]
    return Hu


def _shadow_bound_loop(H, own, known, x0, y0, res, cam, table_z):
    """cells nobody saw: the camera line grazing the object in front of them (as MjProbe.camera_heightmap)"""
    ny, nx = H.shape
    Hu = np.full((ny, nx), -1.0)
    c = np.asarray(cam, float)
    cxy, zc = c[:2], float(c[2])
    for iy, ix in zip(*np.nonzero(~known)):
        p = np.array([x0 + (ix + 0.5) * res, y0 + (iy + 0.5) * res])
        Dp = float(np.linalg.norm(cxy - p))
        u = (cxy - p) / max(Dp, 1e-9)
        bound = None
        for s_ in np.arange(res, 0.30, res):
            q = p + u * s_
            jx, jy = int(math.floor((q[0] - x0) / res)), int(math.floor((q[1] - y0) / res))
            if not (0 <= jx < nx and 0 <= jy < ny):
                break
            if known[jy, jx] and own[jy, jx] >= 0:
                Dq = Dp - s_
                z_line = zc - (zc - float(H[jy, jx])) * Dp / max(Dq, 1e-6)
                bound = z_line if bound is None else max(bound, z_line)
        if bound is not None and bound > table_z:
            Hu[iy, ix] = bound
    return Hu


def build(depth, K, T_base_cam, masks: Sequence[np.ndarray], names: Sequence[str], table_z: float,
          bounds=DEFAULT_BOUNDS, res=RES_M, stride=STRIDE_PX, joint_origins=(), robot_boxes=(), shadows=True):
    """-> (G49.HeightMap with .known / .Hu / .noisy, {name: world points (N, 3)}, stats)"""
    import NERO_V49_Geometry as G49
    names = [str(n) for n in names]
    P, (vv, uu) = backproject(depth, K, T_base_cam, stride)
    n_all = len(P)
    keep = robot_keep_mask(P, joint_origins, robot_boxes) if (len(joint_origins) or len(robot_boxes)) else np.ones(n_all, bool)
    n_robot = int((~keep).sum())
    L = label_image(masks, np.asarray(depth).shape)
    lab = L[vv, uu]
    plane = fit_table_plane(P[keep & (lab < 0)], table_z)
    if plane is not None:                             # level the points: heights above the local table
        a_, b_, c_ = plane["abc"]
        P = P.copy()
        P[:, 2] = P[:, 2] - (a_ * P[:, 0] + b_ * P[:, 1] + c_) + float(table_z)
    x0, x1, y0, y1 = bounds
    nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
    ix = np.floor((P[:, 0] - x0) / res).astype(int)
    iy = np.floor((P[:, 1] - y0) / res).astype(int)
    inb = keep & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (P[:, 2] > table_z - 0.03) & (P[:, 2] < table_z + 0.6)
    P, lab, ix, iy = P[inb], lab[inb], ix[inb], iy[inb]
    h = P[:, 2] - table_z
    known = np.zeros((ny, nx), bool)
    known[iy, ix] = True
    obj = (lab >= 0) & (h > OBJ_MIN_H_M)
    free = (lab < 0) & (h > FREE_MIN_H_M)
    hi = obj | free
    H = np.full((ny, nx), float(table_z))
    own = np.full((ny, nx), -1, int)
    UNK = len(names)                                  # unmasked high point (resolved below)
    o_lab = np.where(obj, lab, UNK)
    order = np.argsort(P[hi, 2])                      # ascending: the highest point of a cell is written last
    sel = np.nonzero(hi)[0][order]
    H[iy[sel], ix[sel]] = P[sel, 2]
    own[iy[sel], ix[sel]] = o_lab[sel]
    H, own, known = _fill_holes(H, own, known)
    # unmasked high cells -> connected "unidentified" obstacles; tiny specks (flying pixels at edges) dropped
    names_all = list(names)
    unk = own == UNK
    if unk.any():
        cv2 = _cv2()
        n_c, comp = cv2.connectedComponents(unk.astype(np.uint8), connectivity=8)
        for c_ in range(1, n_c):
            m = comp == c_
            if int(m.sum()) >= UNID_MIN_CELLS:
                names_all.append(f"unidentified_{len(names_all) - len(names) + 1}")
                own[m] = len(names_all) - 1
            else:
                H[m], own[m] = float(table_z), -1
    hm = G49.HeightMap(x0, y0, res, H, own, names_all, table_z)
    hm.known = known
    hm.Hu = (_shadow_bound(H, own, known, x0, y0, res, np.asarray(T_base_cam, float)[:3, 3], table_z) if shadows
             else np.full((ny, nx), -1.0))
    hm.noisy = True
    pts = {n: P[obj & (lab == k)] for k, n in enumerate(names)}
    stats = {"table_plane": None if plane is None else {k: v for k, v in plane.items() if k != "abc"},
             "points": n_all, "robot_points_removed": n_robot, "cells_seen": int(known.sum()),
             "cells_unseen": int((~known).sum()), "unidentified": names_all[len(names):]}
    return hm, pts, stats


def fit_table_plane(Q, table_z, band_m=0.020, inlier_m=0.004, min_points=500):
    """least-squares plane z = a x + b y + c through the unmasked points near the table height (two refits with
    inliers); a hand-eye error of a few tenths of a degree tilts the table by mm over its width, and an object's
    mask rim would otherwise pick up table points.  -> {"abc", "tilt_deg", "offset_mm_at_center", "rms_mm",
    "points"} or None (too few points)"""
    Q = np.asarray(Q, float)
    if len(Q) < min_points:
        return None
    sel = np.abs(Q[:, 2] - float(table_z)) <= band_m
    abc = None
    for tol in (band_m, 0.008, inlier_m):
        if abc is not None:
            sel = np.abs(Q[:, 2] - (Q[:, 0] * abc[0] + Q[:, 1] * abc[1] + abc[2])) <= tol
        if int(sel.sum()) < min_points:
            return None
        A = np.column_stack([Q[sel, 0], Q[sel, 1], np.ones(int(sel.sum()))])
        abc, *_ = np.linalg.lstsq(A, Q[sel, 2], rcond=None)
    r = Q[sel, 2] - (Q[sel, 0] * abc[0] + Q[sel, 1] * abc[1] + abc[2])
    cx, cy = float(np.median(Q[sel, 0])), float(np.median(Q[sel, 1]))
    return {"abc": [float(v) for v in abc], "tilt_deg": round(math.degrees(math.atan(math.hypot(abc[0], abc[1]))), 3),
            "offset_mm_at_center": round((abc[0] * cx + abc[1] * cy + abc[2] - float(table_z)) * 1000.0, 2),
            "rms_mm": round(float(np.sqrt(np.mean(r * r))) * 1000.0, 2), "points": int(sel.sum())}


def side_fn_from_points(pts: Dict[str, np.ndarray], cam_xy, band_m=0.004, lateral_m=0.006):
    """the side profile from the measured points of each object, only for the sides the camera faces (outer) or sees
    through the opening (inner), like the simulation's real-like camera (ProbeVision._side_fn)"""
    cam_xy = np.asarray(cam_xy, float)

    def fn(oid, ctr, yaw, levels, reach, floor_z):
        P = pts.get(oid)
        a = math.radians(yaw)
        dirs = {"+a": (math.cos(a), math.sin(a)), "-a": (-math.cos(a), -math.sin(a)),
                "+b": (-math.sin(a), math.cos(a)), "-b": (math.sin(a), -math.cos(a))}
        to_cam = cam_xy - np.asarray(ctr, float)[:2]
        to_cam /= max(float(np.linalg.norm(to_cam)), 1e-9)
        rows = []
        for z in levels:
            row = {"z_above_table_mm": None, "outer_half_width_mm": {}, "inner_half_width_mm": {}}
            Q = None if P is None or not len(P) else P[np.abs(P[:, 2] - z) <= band_m]
            for k, (ux, uy) in dirs.items():
                outer = inner = None
                if Q is not None and len(Q):
                    rel = Q[:, :2] - np.asarray(ctr, float)[:2]
                    along = rel @ np.array([ux, uy])
                    perp = np.abs(rel @ np.array([-uy, ux]))
                    sel = along[(perp <= lateral_m) & (along > 0)]
                    facing = ux * to_cam[0] + uy * to_cam[1]
                    if len(sel) >= 2 and facing > 0.2:
                        outer = round(float(np.percentile(sel, 95)) * 1000.0, 1)
                    if len(sel) >= 2 and facing < -0.2 and floor_z is not None and z > floor_z + 0.003:
                        inner = round(float(np.percentile(sel, 10)) * 1000.0, 1)
                row["outer_half_width_mm"][k] = outer
                row["inner_half_width_mm"][k] = inner
            rows.append(row)
        return rows
    return fn


def paint_box(hm, name, center_xy, size_mm, yaw_deg, top_z):
    """an object the masks named but the depth did not show (too few cells): its YOLOE box as a solid block, so the
    collision checks still see it (the old from_boxes behaviour, per object)"""
    k = hm.index_of(name)
    if k < 0:
        hm.names.append(name)
        k = len(hm.names) - 1
    sx, sy = [float(v) / 2000.0 for v in (size_mm or [40.0, 40.0])[:2]]
    yaw = math.radians(float(yaw_deg or 0.0))
    ys, xs = np.mgrid[0:hm.ny, 0:hm.nx]
    X, Y = hm.x0 + (xs + 0.5) * hm.res, hm.y0 + (ys + 0.5) * hm.res
    u = (X - center_xy[0]) * math.cos(yaw) + (Y - center_xy[1]) * math.sin(yaw)
    v = -(X - center_xy[0]) * math.sin(yaw) + (Y - center_xy[1]) * math.cos(yaw)
    m = (np.abs(u) <= sx) & (np.abs(v) <= sy) & (float(top_z) > hm.H)
    hm.H[m], hm.owner[m] = float(top_z), k
    if hasattr(hm, "known"):
        hm.known[m] = True
    return int(m.sum())


def match_ids(new: List[Dict[str, Any]], prev: Dict[str, Dict[str, Any]]) -> Dict[int, str]:
    """V4.13 STABLE IDS: index in `new` -> object id.  Same label: nearest previous object first (greedy, any
    distance - a placed object keeps its id); new objects get the next free number of their label.
    new[i] needs "slug" and "xy"; prev: id -> {"slug", "xy"}."""
    out: Dict[int, str] = {}
    used = set()
    pairs = []
    for i, n in enumerate(new):
        for oid, p in prev.items():
            if p["slug"] == n["slug"]:
                pairs.append((float(np.linalg.norm(np.asarray(n["xy"]) - np.asarray(p["xy"]))), i, oid))
    for _d, i, oid in sorted(pairs):
        if i in out or oid in used:
            continue
        out[i], _ = oid, used.add(oid)
    taken = set(prev) | set(out.values())
    for i, n in enumerate(new):
        if i in out:
            continue
        k = 1
        while f"{n['slug']}_{k}" in taken:
            k += 1
        out[i] = f"{n['slug']}_{k}"
        taken.add(out[i])
    return out


def joint_origins(fast, q) -> List[np.ndarray]:
    """base and every joint origin of the NERO chain (FastArm's own URDF chain)"""
    T = np.eye(4)
    out = [np.zeros(3)]
    for Tf, ax, k in fast.steps:
        T = T @ Tf
        if ax is not None:
            out.append(T[:3, 3].copy())
            M = np.eye(4)
            M[:3, :3] = fast._rot(ax, q[k])
            T = T @ M
    return out                                       # the gripper itself: its boxes (robot_boxes), not a capsule
