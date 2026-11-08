#!/usr/bin/env python3
"""NERO V3.63 cup perception for the fixed stand D435i (stack_cups scene) - RGB-D only, no simulator truth.

V3.63 FIXED D435i CUP PERCEPTION

Same route as the egg perception of the fixed camera (NERO_V308 perceive_eggs / NERO_V331 process): the camera process
renders RGB + z-depth (D435i model: intrinsics, mm quantisation, optional noise, temporal fusion), the perception
works on the image only and returns object poses; the camera calibration (stand pose) is known, object poses are not.

perceive_cups(rgb, depth, K, T_world_opt, robot_boxes=None)
  1. red mask (party cup outside is red; inside / table are white)
  2. back-project the red pixels with the depth image -> 3D points; drop points inside the robot boxes (arm/gripper)
  3. table plane height from the non-red depth points around the cups (median, like the egg pipeline's locked plane)
  4. cluster the red points in x-y (2 cm grid connected components)
  5. per cluster: height of the top (98th percentile) above the table -> single cup (~121 mm) / nested pair
     (~121 + 25 mm) / lying or crushed (anything else); the cup axis = least-squares fit of the visible wall points to
     the known cone r(z) = r_bot + taper * (z - z_table) (only the side facing the camera is visible - the cone model
     puts the axis behind the visible surface); tilt from the axis offset between a low and a high band
  -> list of {"kind": "cup"|"stack"|"unknown", "base_xyz_m", "rim_xyz_m", "axis", "tilt_deg", "height_mm", ...}
The cup model (height, radii) is the scene's CUP dict - like EGG_PRIOR for eggs, perception is told the object type's
nominal size, not where the objects are.
"""
import math
import os
import sys

VERSION = "V3.63"
MARKER = "V3.63 FIXED D435i CUP PERCEPTION"
HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np                       # noqa: E402
import NERO_V343_Scenes as SC            # noqa: E402

WORKSPACE = {"x": (-0.80, -0.08), "y": (-0.42, 0.42)}


def red_mask(rgb):
    r = rgb[..., 0].astype(np.int32)
    g = rgb[..., 1].astype(np.int32)
    b = rgb[..., 2].astype(np.int32)
    # hue-ratio test, not brightness: the wall side away from the lights is dark red (r ~ 40), the shadow on the white
    # table is grey (r ~ g ~ b) and the cup inside is white
    return (r > 25) & (r * 10 > g * 18 + 30) & (r * 10 > b * 18 + 30)


def backproject_mask(depth, mask, K, T):
    vv, uu = np.nonzero(mask & (depth > 0))
    z = depth[vv, uu].astype(float)
    xc = (uu - K["ppx"]) * z / K["fx"]
    yc = (vv - K["ppy"]) * z / K["fy"]
    pc = np.stack([xc, yc, z], axis=1)
    return pc @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]


def _inside_boxes(P, boxes, margin=0.01):
    inside = np.zeros(len(P), dtype=bool)
    for c, R, half in boxes or []:
        q = (P - np.asarray(c)) @ np.asarray(R)
        inside |= np.all(np.abs(q) <= (np.asarray(half) + margin), axis=1)
    return inside


def table_height(depth, mask, K, T, stride=4):
    """median z of the non-red points that lie near the table (|z| < 3 cm) inside the workspace"""
    m = np.zeros_like(mask)
    m[::stride, ::stride] = True
    P = backproject_mask(depth, m & ~mask, K, T)
    if not len(P):
        return 0.0
    sel = (P[:, 0] > WORKSPACE["x"][0]) & (P[:, 0] < WORKSPACE["x"][1]) & (np.abs(P[:, 2]) < 0.03)
    return float(np.median(P[sel, 2])) if np.any(sel) else 0.0


def cluster_xy(P, cell=0.02, min_pts=150):
    if not len(P):
        return []
    ij = np.floor(P[:, :2] / cell).astype(int)
    cells = {}
    for k, key in enumerate(map(tuple, ij)):
        cells.setdefault(key, []).append(k)
    seen, out = set(), []
    for key in cells:
        if key in seen:
            continue
        stack, members = [key], []
        seen.add(key)
        while stack:
            c = stack.pop()
            members.extend(cells[c])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (c[0] + dx, c[1] + dy)
                    if n in cells and n not in seen:
                        seen.add(n)
                        stack.append(n)
        if len(members) >= min_pts:
            out.append(np.asarray(members))
    return out


def fit_axis(P, z0, cam_xy, cup=None, iters=30):
    """axis (cx, cy) of an upright cone wall r(z) = r_bot + taper (z - z0) seen from one side -> (c, rms_m)"""
    g = SC.cup_geometry(cup)
    taper = (g["r_top"] - g["r_bot"]) / g["h"]
    r = g["r_bot"] + taper * np.clip(P[:, 2] - z0, 0.0, g["h"])
    cen = P[:, :2].mean(axis=0)
    away = cen - np.asarray(cam_xy, float)
    away /= (np.linalg.norm(away) or 1.0)
    c = cen + away * float(np.mean(r)) * 0.7            # the visible wall is the camera side of the cone
    for _ in range(iters):
        d = P[:, :2] - c
        dist = np.linalg.norm(d, axis=1) + 1e-9
        res = dist - r
        J = -d / dist[:, None]                          # d res / d c
        step = np.linalg.lstsq(J, -res, rcond=None)[0]
        c = c + step
        if np.linalg.norm(step) < 1e-6:
            break
    res = np.linalg.norm(P[:, :2] - c, axis=1) - r
    return c, float(np.sqrt(np.mean(res ** 2)))


def perceive_cups(rgb, depth, K, T, robot_boxes=None, cup=None, nested_dz=None):
    """-> {"cups": [...], "table_z_m": z, "n_red_px": n, "clusters": n} (see module docstring)"""
    g = SC.cup_geometry(cup)
    H = g["h"]
    ndz = float(g["nested_dz"] if nested_dz is None else nested_dz)
    mask = red_mask(rgb)
    z0 = table_height(depth, mask, K, T)
    P = backproject_mask(depth, mask, K, T)
    if len(P) and robot_boxes:
        P = P[~_inside_boxes(P, robot_boxes)]
    sel = (P[:, 0] > WORKSPACE["x"][0]) & (P[:, 0] < WORKSPACE["x"][1]) & (P[:, 1] > WORKSPACE["y"][0]) & \
          (P[:, 1] < WORKSPACE["y"][1]) & (P[:, 2] > z0 + 0.003) & (P[:, 2] < z0 + H + 2 * ndz + 0.02) \
        if len(P) else np.zeros(0, bool)      # nothing on the table is taller than a nested pair (red finger pads are)
    P = P[sel] if len(P) else P
    cam_xy = np.asarray(T)[:2, 3]
    out = []
    for idx in cluster_xy(P):
        Q = P[idx]
        top = float(np.percentile(Q[:, 2], 98)) - z0
        low = Q[Q[:, 2] < z0 + min(top, H) - 0.008]       # wall points below the (top) rim
        if len(low) < 80:
            continue
        c, rms = fit_axis(low, z0, cam_xy, cup)
        # tilt: separate fits of a low and a high band
        tilt = 0.0
        zl, zh = z0 + 0.015, z0 + min(top, H) - 0.020
        bl, bh = low[low[:, 2] < zl + 0.025], low[low[:, 2] > zh - 0.025]
        if len(bl) > 40 and len(bh) > 40 and zh - zl > 0.05:
            cl, _ = fit_axis(bl, z0, cam_xy, cup)
            ch, _ = fit_axis(bh, z0, cam_xy, cup)
            tilt = math.degrees(math.atan2(float(np.linalg.norm(ch - cl)), float(np.mean(bh[:, 2]) - np.mean(bl[:, 2]))))
        if abs(top - H) <= 0.012:
            kind = "cup"
        elif abs(top - (H + ndz)) <= 0.012:
            kind = "stack"
        else:
            kind = "unknown"
        base = [float(c[0]), float(c[1]), float(z0)]
        rim_z = z0 + (H + ndz if kind == "stack" else H)
        out.append({"kind": kind, "base_xyz_m": [round(v, 4) for v in base],
                    "rim_xyz_m": [round(float(c[0]), 4), round(float(c[1]), 4), round(float(rim_z), 4)],
                    "height_mm": round(top * 1000.0, 1), "tilt_deg": round(tilt, 1), "fit_rms_mm": round(rms * 1000.0, 2),
                    "n_points": int(len(Q))})
    out.sort(key=lambda d: d["base_xyz_m"][1])
    return {"cups": out, "table_z_m": round(z0, 4), "n_red_px": int(mask.sum()), "clusters": len(out)}


def quat_from_tilt(tilt_deg, azimuth_rad=0.0):
    """wxyz quaternion of a cup axis tilted by tilt_deg towards azimuth (perception only knows the tilt angle)"""
    t = math.radians(float(tilt_deg)) / 2.0
    ax = (-math.sin(azimuth_rad), math.cos(azimuth_rad), 0.0)
    return [math.cos(t), ax[0] * math.sin(t), ax[1] * math.sin(t), 0.0]
