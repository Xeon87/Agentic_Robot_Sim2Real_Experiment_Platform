#!/usr/bin/env python3
"""
NERO V3.08 SIMULATED D435i CAMERA + RGB-D EGG PERCEPTION (record only)
======================================================================
(V3.10: temporal depth fusion, inlier-count model selection, free-space view check,
resting-egg prior, warm start, per-detection confidence / trusted flag.)
(V3.11: identity tracking + synchronous result channel so skills can use the perceived
poses; visual D435i body + floor stand drawn at the configured camera pose.)

What this module does
---------------------
* Injects one fixed (eye-to-hand) camera ``v308_d435i`` into the MuJoCo scene
  XML.  Intrinsics follow the Intel RealSense D435i COLOR stream (about
  69 x 42 deg field of view) with depth ALIGNED to color, i.e. RGB and depth
  share one pinhole model, exactly what ``rs.align(rs.stream.color)`` gives on
  the real camera.
* Runs in its OWN process (offscreen OpenGL: EGL -> OSMesa -> GLFW fallback),
  so rendering can never stall or crash the physics/viewer process.
* On every trigger (observe_scene / plan_grasp / verify_object_in_slot by
  default) it copies the live MuJoCo qpos, renders RGB + depth, runs a
  perception pipeline that uses ONLY the images + camera calibration, and
  compares the result with MuJoCo ground truth taken from the very same qpos.
* Everything is written to files under ``logs/perception/<run>/``.

What it does NOT do
-------------------
Skills still use ground truth.  Nothing here changes robot behaviour.

Import rule: this module must not import mujoco at module level.  The backend
imports it for configuration helpers inside processes that already chose their
own OpenGL backend (the viewer needs GLFW).  The camera process imports mujoco
only after the parent has set MUJOCO_GL for it.
"""
from __future__ import annotations

import csv
import json
import math
import os
import struct
import time
import traceback
import zlib
from pathlib import Path

import numpy as np

CAMERA_NAME = "v308_d435i"
VERSION = "V3.11"
EXIT_GL_FAILED = 3

# ----------------------------------------------------------------------------
# Configuration (environment overridable)
# ----------------------------------------------------------------------------

def _env_vec(name, default):
    raw = os.environ.get(name, default)
    vals = [float(v) for v in str(raw).replace(" ", "").split(",") if v != ""]
    if len(vals) != 3:
        raise ValueError(f"{name} needs 3 comma-separated numbers, got {raw!r}")
    return vals


def camera_config():
    """Single source of truth for the simulated camera."""
    w = int(os.environ.get("NERO_V308_WIDTH", "1280"))
    h = int(os.environ.get("NERO_V308_HEIGHT", "720"))
    # D435i color FOV is ~69 x 42 deg (H x V).  MuJoCo uses vertical fovy and
    # square pixels: f = (h/2)/tan(fovy/2) -> 1280x720 @ 42 deg gives
    # f ~ 938 px and HFOV ~ 68.6 deg (real D435i color 1280x720: fx ~ 900-930 px).
    fovy = float(os.environ.get("NERO_V308_FOVY_DEG", "42.0"))
    pos = _env_vec("NERO_V308_CAM_POS", "-0.95,0.0,0.72")
    target = _env_vec("NERO_V308_CAM_TARGET", "-0.42,0.0,0.03")
    return {
        "name": CAMERA_NAME,
        "width": w, "height": h, "fovy_deg": fovy,
        "pos": pos, "target": target,
        "min_range_m": float(os.environ.get("NERO_V308_MIN_RANGE_M", "0.28")),   # D435 min-Z at high res
        "max_range_m": float(os.environ.get("NERO_V308_MAX_RANGE_M", "3.0")),
        "depth_noise_k": float(os.environ.get("NERO_V308_DEPTH_NOISE_K", "0.0")),  # sigma_z = k * z^2 (0 = off)
        "depth_frames": max(1, int(os.environ.get("NERO_V310_DEPTH_FRAMES", "5"))),   # V3.10 temporal fusion
    }


def enabled():
    return os.environ.get("NERO_V308_CAMERA", "1") != "0"


def trigger_tools():
    raw = os.environ.get("NERO_V308_TRIGGER_TOOLS", "observe_scene,plan_grasp,verify_object_in_slot")
    return {t.strip() for t in raw.split(",") if t.strip()}


def gl_backends():
    raw = os.environ.get("NERO_V308_GL", "egl,osmesa,glfw")
    return [b.strip().lower() for b in raw.split(",") if b.strip()]


# ----------------------------------------------------------------------------
# Camera geometry
# ----------------------------------------------------------------------------

def _unit(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def look_at_axes(pos, target, up=(0.0, 0.0, 1.0)):
    """MuJoCo camera axes: x = image right, y = image up, camera looks along -z."""
    f = _unit(np.asarray(target, float) - np.asarray(pos, float))
    x = _unit(np.cross(f, np.asarray(up, float)))
    if float(np.linalg.norm(x)) < 1e-9:          # looking straight down
        x = np.array([1.0, 0.0, 0.0])
    y = _unit(np.cross(x, f))
    return x, y, f


def intrinsics(cfg):
    w, h = int(cfg["width"]), int(cfg["height"])
    f = (h / 2.0) / math.tan(math.radians(cfg["fovy_deg"]) / 2.0)
    # pixel-index convention: centre of pixel (0,0) is at (0,0); image centre at (w-1)/2
    return {"width": w, "height": h, "fx": f, "fy": f, "ppx": (w - 1) / 2.0, "ppy": (h - 1) / 2.0,
            "hfov_deg": math.degrees(2.0 * math.atan((w / 2.0) / f)), "vfov_deg": float(cfg["fovy_deg"])}


def extrinsics(cfg):
    """T_world_optical (OpenCV / RealSense optical frame: x right, y down, z forward)."""
    x, y, f = look_at_axes(cfg["pos"], cfg["target"])
    R_opt = np.stack([x, -y, f], axis=1)          # columns = optical axes in world
    T = np.eye(4)
    T[:3, :3] = R_opt
    T[:3, 3] = np.asarray(cfg["pos"], float)
    return T


def camera_xml(cfg):
    x, y, _ = look_at_axes(cfg["pos"], cfg["target"])
    p = cfg["pos"]
    return (f'<camera name="{CAMERA_NAME}" mode="fixed" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
            f'xyaxes="{x[0]:.6f} {x[1]:.6f} {x[2]:.6f} {y[0]:.6f} {y[1]:.6f} {y[2]:.6f}" '
            f'fovy="{cfg["fovy_deg"]:.4f}"/>')


def camera_rig_parts(cfg):
    """V3.11: visual-only D435i body + floor-standing stand at the configured camera pose.
    Returns a list of (kind, name, pos, size, mat3x3 or None, rgba).  Everything sits BEHIND the
    optical centre (the camera renders from cfg['pos'] looking along +forward), so nothing of the rig
    can appear in the rendered images; geoms are contype/conaffinity 0 (no physics)."""
    x, y, f = look_at_axes(cfg["pos"], cfg["target"])
    p = np.asarray(cfg["pos"], float)
    floor_z = float(os.environ.get("NERO_V311_FLOOR_Z_M", "-0.28"))
    R = np.stack([x, y, -f], axis=1)                       # local x right, y up, z backwards (out of the lens)
    parts = []
    depth_half = 0.0125
    body_c = p - f * (depth_half + 0.003)                  # front face 3 mm behind the optical centre
    parts.append(("box", "v311_d435i_body", body_c, [0.045, 0.0125, depth_half], R, [0.18, 0.18, 0.20, 1]))
    for k, (dx, rad, rgba) in enumerate(((-0.025, 0.0055, [0.02, 0.02, 0.03, 1]), (-0.0035, 0.0045, [0.02, 0.02, 0.03, 1]),
                                         (0.0115, 0.0045, [0.25, 0.05, 0.05, 1]), (0.032, 0.0055, [0.02, 0.02, 0.03, 1]))):
        lc = p - f * 0.0035 + x * dx
        parts.append(("cylinder", f"v311_d435i_lens{k}", lc, [rad, 0.0006], R, rgba))
    # mount: from the bottom of the body straight down to a ball head, then a vertical pole to the floor
    under = body_c - y * 0.0125 - np.array([0.0, 0.0, 0.0])
    head = np.array([under[0], under[1], under[2] - 0.035])
    parts.append(("capsule_ft", "v311_stand_neck", under, head, None, [0.12, 0.12, 0.13, 1]))
    parts.append(("sphere", "v311_stand_head", head, [0.014], None, [0.12, 0.12, 0.13, 1]))
    foot = np.array([head[0], head[1], floor_z + 0.012])
    parts.append(("capsule_ft", "v311_stand_pole", head, foot, None, [0.55, 0.56, 0.58, 1]))
    parts.append(("cylinder", "v311_stand_collar", np.array([head[0], head[1], floor_z + 0.45]), [0.017, 0.02], None, [0.12, 0.12, 0.13, 1]))
    for k in range(3):                                      # tripod legs on the floor
        ang = math.radians(90.0 + 120.0 * k)
        toe = np.array([head[0] + 0.28 * math.cos(ang), head[1] + 0.28 * math.sin(ang), floor_z + 0.008])
        knee = np.array([head[0], head[1], floor_z + 0.42])
        parts.append(("capsule_ft", f"v311_stand_leg{k}", knee, toe, None, [0.12, 0.12, 0.13, 1]))
    return parts


def camera_rig_xml(cfg):
    out = []
    for kind, name, a, b, R, rgba in camera_rig_parts(cfg):
        col = " ".join(f"{v:.3f}" for v in rgba)
        common = f'contype="0" conaffinity="0" group="0" rgba="{col}"'
        if kind == "capsule_ft":
            out.append(f'<geom name="{name}" type="capsule" fromto="{a[0]:.4f} {a[1]:.4f} {a[2]:.4f} {b[0]:.4f} {b[1]:.4f} {b[2]:.4f}" '
                       f'size="{0.009 if "pole" in name or "neck" in name else 0.007:.4f}" {common}/>')
            continue
        pos = f'pos="{a[0]:.4f} {a[1]:.4f} {a[2]:.4f}"'
        size = " ".join(f"{v:.4f}" for v in b)
        if R is not None:
            xa, ya = R[:, 0], R[:, 1]
            ori = f' xyaxes="{xa[0]:.6f} {xa[1]:.6f} {xa[2]:.6f} {ya[0]:.6f} {ya[1]:.6f} {ya[2]:.6f}"'
        else:
            ori = ""
        out.append(f'<geom name="{name}" type="{kind}" {pos}{ori} size="{size}" {common}/>')
    return "\n    ".join(out)


def inject_camera_into_mjcf(xml_path, cfg=None):
    """Return a new MJCF path (same folder, so relative mesh paths keep working).
    V3.11: also draws the D435i body and its floor stand (NERO_V311_DRAW_CAMERA=0 to disable)."""
    cfg = cfg or camera_config()
    src = Path(xml_path)
    text = src.read_text(encoding="utf-8")
    if f'name="{CAMERA_NAME}"' in text:
        return str(src)
    idx = text.rfind("</worldbody>")
    if idx < 0:
        raise RuntimeError("V3.08: </worldbody> not found in MJCF")
    rig = ""
    if os.environ.get("NERO_V311_DRAW_CAMERA", "1") != "0":
        rig = "    <!-- V3.11 D435i body + floor stand (visual only) -->\n    " + camera_rig_xml(cfg) + "\n"
    text = text[:idx] + "    " + camera_xml(cfg) + "\n" + rig + "  " + text[idx:]
    out = src.with_name("_nero_motor_v308_camera_tmp.xml")
    out.write_text(text, encoding="utf-8")
    return str(out)


def project(points_world, K, T_world_opt):
    """World points (N,3) -> pixel (N,2) and optical depth z (N,)."""
    R = T_world_opt[:3, :3]
    t = T_world_opt[:3, 3]
    pc = (np.asarray(points_world, float) - t) @ R          # world -> optical
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K["fx"] * pc[:, 0] / z + K["ppx"]
        v = K["fy"] * pc[:, 1] / z + K["ppy"]
    return np.stack([u, v], axis=1), z


def backproject(depth_m, K, T_world_opt, stride=1):
    """Aligned z-depth image (meters, 0 = invalid) -> world points and pixel indices."""
    h, w = depth_m.shape
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[::stride, ::stride]
    ok = z > 0
    uu, vv, z = uu[ok].astype(float), vv[ok].astype(float), z[ok].astype(float)
    xc = (uu - K["ppx"]) * z / K["fx"]
    yc = (vv - K["ppy"]) * z / K["fy"]
    pc = np.stack([xc, yc, z], axis=1)
    pw = pc @ T_world_opt[:3, :3].T + T_world_opt[:3, 3]
    return pw, vv.astype(int), uu.astype(int)


# ----------------------------------------------------------------------------
# Scene knowledge used by perception (NOT ground truth of object poses)
# ----------------------------------------------------------------------------
# Receptacle geometry is "registered", as it will be on the real bench with
# fiducial tags + CAD.  Egg poses are perceived from RGB-D only.
TABLE_Z_M = 0.0
BASKET = {"center_xy_m": [-0.425, -0.155], "radius_m": 0.088}
SLOTS = {
    "slot_1": (-0.510, 0.123), "slot_2": (-0.450, 0.123), "slot_3": (-0.390, 0.123), "slot_4": (-0.330, 0.123),
    "slot_5": (-0.510, 0.187), "slot_6": (-0.450, 0.187), "slot_7": (-0.390, 0.187), "slot_8": (-0.330, 0.187),
}
EGG_PRIOR = {"half_long_m": 0.0315, "half_short_m": 0.0219}   # mean of the scene eggs; perception is not told per-egg sizes
WORKSPACE = {"x": (-0.80, -0.08), "y": (-0.42, 0.42), "z": (0.004, 0.45)}


def classify_location(xyz):
    """Same rule as the backend's llm_classify_object_location()."""
    x, y, z = [float(v) for v in xyz]
    if z <= 0.095:
        best, bd = None, 1e9
        for s, (sx, sy) in SLOTS.items():
            d = math.hypot(x - sx, y - sy)
            if d < bd:
                best, bd = s, d
        if bd <= 0.026:
            return f"egg_holder.{best}"
    bx, by = BASKET["center_xy_m"]
    if math.hypot(x - bx, y - by) <= BASKET["radius_m"] and z <= 0.13:
        return "egg_basket"
    return "table_or_transit"


# ----------------------------------------------------------------------------
# Perception: RGB-D -> egg list (uses images + calibration only)
# ----------------------------------------------------------------------------

def egg_color_mask(rgb):
    """Brown-shell eggs: hue 15-45 deg, saturation 0.25-0.85, value > 0.25.
    Set NERO_V308_EGG_COLOR_FILTER=0 for white eggs (height-only segmentation)."""
    if os.environ.get("NERO_V308_EGG_COLOR_FILTER", "1") == "0":
        return np.ones(rgb.shape[:2], dtype=bool)
    c = rgb.astype(np.float32) / 255.0
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx = c.max(axis=2)
    mn = c.min(axis=2)
    d = mx - mn + 1e-6
    sat = d / (mx + 1e-6)
    hue = np.where(mx == r, ((g - b) / d) % 6.0, np.where(mx == g, (b - r) / d + 2.0, (r - g) / d + 4.0)) * 60.0
    return (hue >= 12.0) & (hue <= 48.0) & (sat >= 0.22) & (sat <= 0.90) & (mx >= 0.22)


def _axis_from(yaw, pitch):
    return np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])


def _fit_cost(params, pts, a, b):
    c = params[:3]
    d = _axis_from(params[3], params[4])
    q = pts - c
    along = q @ d
    perp = np.linalg.norm(q - np.outer(along, d), axis=1)
    s = np.sqrt((along / a) ** 2 + (perp / b) ** 2)
    return (s - 1.0) * b


def fit_spheroid(pts, init, a, b, iters=25, tilt_prior=None):
    """Damped Gauss-Newton (numeric Jacobian) with Huber weights. Returns params, rms_m.
    V3.10 tilt_prior=(sigma_noise_m, sigma_tilt_rad): soft prior that a resting egg lies horizontal
    (one extra residual row sqrt(N)*sigma_noise*sin(pitch)/sin(sigma_tilt)); rms excludes the prior row."""
    p = np.asarray(init, float).copy()
    lam = 1e-3
    hub = 0.002
    eps = np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-3])
    npts = len(pts)
    if tilt_prior is not None:
        pr_scale = math.sqrt(npts) * float(tilt_prior[0]) / max(1e-3, math.sin(float(tilt_prior[1])))
        def cost(q):
            return np.append(_fit_cost(q, pts, a, b), pr_scale * math.sin(q[4]))
    else:
        def cost(q):
            return _fit_cost(q, pts, a, b)
    def weights(r):
        w_ = np.where(np.abs(r) <= hub, 1.0, hub / (np.abs(r) + 1e-12))
        if tilt_prior is not None:
            w_[-1] = 1.0
        return w_
    for _ in range(iters):
        r = cost(p)
        wgt = weights(r)
        J = np.empty((len(r), 5))
        for k in range(5):
            dp = np.zeros(5)
            dp[k] = eps[k]
            J[:, k] = (cost(p + dp) - r) / eps[k]
        Jw = J * wgt[:, None]
        H = J.T @ Jw + lam * np.eye(5)
        g = Jw.T @ r
        try:
            step = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        pn = p + step
        if np.sum(wgt * cost(pn) ** 2) < np.sum(wgt * r ** 2):
            p, lam = pn, max(lam * 0.3, 1e-7)
            if np.linalg.norm(step[:3]) < 1e-6 and np.max(np.abs(step[3:])) < 1e-5:
                break
        else:
            lam *= 10.0
    r = _fit_cost(p, pts, a, b)
    return p, float(np.sqrt(np.mean(r ** 2)))


def _quat_from_z_to(d):
    """Quaternion (x,y,z,w) rotating local +Z onto unit vector d (egg long axis)."""
    z = np.array([0.0, 0.0, 1.0])
    d = _unit(d)
    c = float(np.dot(z, d))
    if c < -0.999999:
        return [1.0, 0.0, 0.0, 0.0]
    axis = np.cross(z, d)
    s = math.sqrt((1.0 + c) * 2.0)
    return [float(axis[0] / s), float(axis[1] / s), float(axis[2] / s), float(s / 2.0)]


def remove_small_blobs(mask, min_px):
    """V3.09: drop connected components smaller than min_px (specular specks on holder posts, label edges)."""
    if min_px <= 0:
        return mask
    try:
        import cv2
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        good = np.zeros(n, dtype=bool)
        good[1:] = stats[1:, cv2.CC_STAT_AREA] >= int(min_px)
        return good[lab]
    except Exception:
        pass
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
        if n == 0:
            return mask
        area = np.bincount(lab.ravel())
        good = area >= int(min_px)
        good[0] = False
        return good[lab]
    except Exception:
        return mask


def points_inside_boxes(P, boxes, margin):
    """Robot self-filter: True for points inside any oriented box (centre, R(3x3), half-extents) inflated by margin."""
    inside = np.zeros(len(P), dtype=bool)
    for c, R, half in boxes:
        q = (P - c) @ R
        inside |= np.all(np.abs(q) <= (np.asarray(half) + margin), axis=1)
    return inside


def temporal_fuse(frames, min_valid=None):
    """V3.10: fuse N aligned depth frames (RealSense-style temporal filter on a static scene).
    Per pixel: mean of the valid samples; pixel kept only if valid in >= min_valid frames.
    Output is re-quantised to z16 millimetres."""
    st = np.stack([np.asarray(f, dtype=np.float32) for f in frames], axis=0)
    if len(st) == 1:
        return st[0]
    if min_valid is None:
        min_valid = max(1, (len(st) + 1) // 2)
    ok = st > 0
    cnt = ok.sum(axis=0)
    s = np.where(ok, st, 0.0).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = np.where(cnt >= min_valid, s / np.maximum(cnt, 1), 0.0)
    return (np.round(m * 1000.0) / 1000.0).astype(np.float32)


def _egg_R(p):
    """Rotation whose 3rd column is the egg long axis (orthonormal frame for ray casting)."""
    d = _axis_from(p[3], p[4])
    t = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = _unit(np.cross(t, d))
    e2 = np.cross(d, e1)
    return np.stack([e1, e2, d], axis=1)


def predict_egg_depth(p, a, b, K, T_world_opt, shape, stride=1, inner_scale=0.8):
    """V3.10: ray-cast the fitted spheroid into the image.
    Returns (vv, uu, z_pred, inner) for every pixel (on the stride grid) whose ray hits the egg;
    inner = the ray also hits the egg shrunk to inner_scale (away from the grazing rim, where a
    1 mm pose error already changes the predicted depth by centimetres)."""
    h, w = shape
    c = np.asarray(p[:3], float)
    R = _egg_R(p)
    s = np.array([b, b, a])
    corners = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float) * s
    uv, z = project(c + corners @ R.T, K, T_world_opt)
    if np.any(z <= 0.05):
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0, bool)
    u0 = max(0, int(math.floor(uv[:, 0].min())) - 1); u1 = min(w - 1, int(math.ceil(uv[:, 0].max())) + 1)
    v0 = max(0, int(math.floor(uv[:, 1].min())) - 1); v1 = min(h - 1, int(math.ceil(uv[:, 1].max())) + 1)
    if u1 < u0 or v1 < v0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0, bool)
    u0 -= u0 % stride; v0 -= v0 % stride
    vv, uu = np.mgrid[v0:v1 + 1:stride, u0:u1 + 1:stride]
    vv = vv.ravel(); uu = uu.ravel()
    Dopt = np.stack([(uu - K["ppx"]) / K["fx"], (vv - K["ppy"]) / K["fy"], np.ones(len(uu))], axis=1)
    Dw = Dopt @ T_world_opt[:3, :3].T            # parametrised so t == optical z
    o = (R.T @ (T_world_opt[:3, 3] - c)) / s
    dd = (Dw @ R) / s
    A = np.sum(dd * dd, 1); B = 2.0 * dd @ o; C = float(o @ o) - 1.0
    disc = B * B - 4.0 * A * C
    hit = disc > 0
    t = (-B[hit] - np.sqrt(disc[hit])) / (2.0 * A[hit])
    good = t > 0
    si = inner_scale
    # same ray against the spheroid scaled by si (o, dd scale by 1/si)
    A2, B2, C2 = A[hit] / (si * si), B[hit] / (si * si), (C + 1.0) / (si * si) - 1.0
    inner = (B2 * B2 - 4.0 * A2 * C2) > 0
    return vv[hit][good], uu[hit][good], t[good], inner[good]


def egg_view_check(p, a, b, depth_m, K, T_world_opt, stride=1, tol_m=0.006):
    """V3.10 free-space / visibility test of one fitted egg against the observed depth image.
    consistent : observed depth matches the predicted surface (|dz| <= tol)
    occluded   : something is in front of the predicted surface (dz < -tol) -> not evidence against
    see_through: the camera sees FURTHER than the predicted surface (dz > +tol) -> the egg cannot be there
    missing    : no depth"""
    vv, uu, zp, inner = predict_egg_depth(p, a, b, K, T_world_opt, depth_m.shape, stride)
    n = len(zp)
    if n == 0:
        return {"silhouette_px": 0, "consistent": 0, "occluded": 0, "see_through": 0, "missing": 0,
                "coverage": 0.0, "see_through_ratio": 1.0, "match_ratio": 0.0}
    zo = depth_m[vv, uu].astype(float)
    miss = zo <= 0
    dz = zo - zp
    cons = (~miss) & (np.abs(dz) <= tol_m)
    occ = (~miss) & (dz < -tol_m)
    thr = (~miss) & (dz > tol_m) & inner
    visible = n - int(occ.sum()) - int(miss.sum())
    vis_inner = max(1, int(np.count_nonzero(inner & ~occ & ~miss)))
    return {"silhouette_px": int(n), "consistent": int(cons.sum()), "occluded": int(occ.sum()),
            "see_through": int(thr.sum()), "missing": int(miss.sum()),
            "coverage": round(visible / n, 3),
            "see_through_ratio": round(float(thr.sum()) / vis_inner, 3),
            "match_ratio": round(float(cons.sum()) / max(1, visible), 3)}


def _surface_res(p, pts, a, b):
    return np.abs(_fit_cost(p, pts, a, b))


def _is_resting(xyz, b):
    """V3.10: an egg whose centre is within ~8 mm of short-semi-axis height above the table, outside the
    holder, is resting on the table / basket floor -> physically it lies (near) horizontal."""
    if os.environ.get("NERO_V310_RESTING_PRIOR", "1") == "0":
        return False
    x, y, z = [float(v) for v in xyz[:3]]
    if classify_location([x, y, z]).startswith("egg_holder"):
        return False
    return z < TABLE_Z_M + b + 0.008


def _refine_on_neighbourhood(P, p, a, b, tol, rounds=2, cap=2500, seed=0, tilt=None):
    """Re-fit on the inliers of the current surface inside a 48 mm ball (not just the seed's Voronoi cell)."""
    rng = np.random.default_rng(seed)
    rms = None
    for _ in range(rounds):
        near = np.linalg.norm(P - p[:3], axis=1) < 0.048
        Q = P[near]
        if len(Q) < 40:
            break
        r = _surface_res(p, Q, a, b)
        Q = Q[r < 2.5 * tol]
        if len(Q) < 40:
            break
        if len(Q) > cap:
            Q = Q[rng.choice(len(Q), cap, replace=False)]
        p, rms = fit_spheroid(Q, p, a, b, tilt_prior=(tilt if (tilt and _is_resting(p, b)) else None))
    return p, rms


def perceive_eggs(rgb, depth_m, K, T_world_opt, stride=None, robot_boxes=None, prior=None, noise_k=None):
    """Return (detections, diagnostics).

    V3.10 pipeline (uses only RGB-D + calibration + robot joint state; NO object truth):
      1. back-project, egg colour mask, small-blob removal, workspace, robot self-filter    (as V3.09)
      2. dome-top seeds on a 3 mm height map                                               (as V3.09)
      3. candidates: multi-start spheroid fits per seed (+ previous-capture warm starts, + resting-egg starts),
         each refined on the surface inliers of a 48 mm neighbourhood
      4. candidate score = surface inliers x (1 - 3 x see-through ratio); greedy selection: the best
         candidate claims its inliers, the rest are re-scored on the unclaimed points only
         (V3.09 kept the LOWEST-RMS fit of two overlapping ones -> a half-egg fragment beat the full egg)
      5. final ownership + refit, then a per-detection confidence and a trusted flag
    Each detection: xyz_m, axis_world, quat_xyzw, rms_mm, n_points, location, confidence, trusted, quality{...}."""
    a, b = EGG_PRIOR["half_long_m"], EGG_PRIOR["half_short_m"]
    if stride is None:
        stride = int(os.environ.get("NERO_V308_STRIDE", "1"))
    if noise_k is None:
        noise_k = float(os.environ.get("NERO_V310_EXPECTED_NOISE_K", "0.0025"))
    pw, vv, uu = backproject(depth_m, K, T_world_opt, stride=stride)
    mask_img = egg_color_mask(rgb)
    mask_raw = int(mask_img.sum())
    min_blob = int(float(os.environ.get("NERO_V308_MIN_BLOB_PX", "150")) * (rgb.shape[0] * rgb.shape[1]) / (1280.0 * 720.0))
    mask_img = remove_small_blobs(mask_img, min_blob)
    col = mask_img[vv, uu]
    ws = WORKSPACE
    keep = (col & (pw[:, 0] > ws["x"][0]) & (pw[:, 0] < ws["x"][1]) & (pw[:, 1] > ws["y"][0]) & (pw[:, 1] < ws["y"][1])
            & (pw[:, 2] > TABLE_Z_M + ws["z"][0]) & (pw[:, 2] < ws["z"][1]))
    P = pw[keep]
    diag = {"pipeline": VERSION, "points_valid": int(len(pw)), "mask_px_raw": mask_raw,
            "mask_px_after_blob_filter": int(mask_img.sum()), "points_egg_mask": int(len(P))}
    if robot_boxes:
        rob = points_inside_boxes(P, robot_boxes, float(os.environ.get("NERO_V308_ROBOT_FILTER_MARGIN_M", "0.008")))
        diag["points_removed_robot_self_filter"] = int(rob.sum())
        P = P[~rob]
    if len(P) < 30:
        return [], diag
    # expected depth noise at the scene (z ~ 0.8-0.9 m): inlier tolerance adapts to it
    z_typ = float(np.median(depth_m[depth_m > 0])) if np.any(depth_m > 0) else 0.85
    sigma = max(0.0006, noise_k * z_typ * z_typ)
    tol = max(0.0015, 1.6 * sigma)
    diag["inlier_tol_mm"] = round(tol * 1000.0, 2)

    # --- seeds (unchanged from V3.09) ---
    cell = 0.003
    x0, y0 = ws["x"][0], ws["y"][0]
    ix = np.clip(((P[:, 0] - x0) / cell).astype(int), 0, int((ws["x"][1] - x0) / cell))
    iy = np.clip(((P[:, 1] - y0) / cell).astype(int), 0, int((ws["y"][1] - y0) / cell))
    Hm = np.full((ix.max() + 1, iy.max() + 1), -1.0)
    np.maximum.at(Hm, (ix, iy), P[:, 2])
    occ = np.argwhere(Hm > TABLE_Z_M + float(os.environ.get("NERO_V308_SEED_MIN_TOP_M", "0.030")))
    order = np.argsort(-Hm[occ[:, 0], occ[:, 1]])
    seeds = []
    min_sep = float(os.environ.get("NERO_V308_SEED_MIN_SEP_M", "0.036"))
    min_support = int(float(os.environ.get("NERO_V308_SEED_MIN_SUPPORT", "120")) * (rgb.shape[0] * rgb.shape[1]) / (1280.0 * 720.0) / (stride * stride))
    rejected_support = 0
    for k in order:
        i, j = occ[k]
        sx, sy = x0 + (i + 0.5) * cell, y0 + (j + 0.5) * cell
        if any(math.hypot(sx - s[0], sy - s[1]) < min_sep for s in seeds):
            continue
        support = int(np.count_nonzero((np.abs(P[:, 0] - sx) < 0.022) & (np.abs(P[:, 1] - sy) < 0.022)))
        if support < max(10, min_support):
            rejected_support += 1
            continue
        seeds.append((sx, sy, float(Hm[i, j])))
        if len(seeds) >= 40:
            break
    diag["seeds"] = len(seeds)
    diag["seeds_rejected_low_support"] = rejected_support
    diag["seed_xy_m"] = [[round(s[0], 4), round(s[1], 4), round(s[2], 4)] for s in seeds]
    if not seeds:
        return [], diag
    S = np.array([[s[0], s[1]] for s in seeds])
    dxy = np.linalg.norm(P[:, None, :2] - S[None, :, :], axis=2)
    lab = np.argmin(dxy, axis=1)
    near = dxy[np.arange(len(P)), lab] <= 0.045

    tilt = (sigma, math.radians(float(os.environ.get("NERO_V310_RESTING_TILT_SIGMA_DEG", "6"))))

    def _best_fit(Q, init_list):
        best = None
        for init in init_list:
            p, rms = fit_spheroid(Q, init, a, b)
            if _is_resting(p, b):
                p, rms = fit_spheroid(Q, p, a, b, tilt_prior=tilt)
            if best is None or rms < best[1]:
                best = (p, rms)
        return best

    # --- step 3: candidates ---
    cands = []
    for k, (sx, sy, top) in enumerate(seeds):
        Q = P[near & (lab == k)]
        if len(Q) < 40:
            continue
        if len(Q) > 2500:
            Q = Q[np.random.default_rng(k).choice(len(Q), 2500, replace=False)]
        xy = Q[:, :2] - Q[:, :2].mean(axis=0)
        ev, evec = np.linalg.eigh(np.cov(xy.T))
        yaw0 = math.atan2(evec[1, 1], evec[0, 1])
        inits = [[sx, sy, top - b, yaw0 + dy, 0.0] for dy in (0.0, math.pi / 4, -math.pi / 4, math.pi / 2)]
        inits.append([sx, sy, top - a, 0.0, math.pi / 2 - 1e-3])
        # resting-egg starts shifted half an egg along the PCA axis (the seed may sit on one end)
        for sgn in (1.0, -1.0):
            inits.append([sx + sgn * 0.5 * (a - b) * math.cos(yaw0), sy + sgn * 0.5 * (a - b) * math.sin(yaw0), top - b, yaw0, 0.0])
        p, _ = _best_fit(Q, inits)
        cands.append(("seed", p))
    for pr in (prior or []):
        try:
            c = np.asarray(pr["xyz_m"], float)
            ax = _unit(np.asarray(pr["axis_world"], float))
        except Exception:
            continue
        Q = P[np.linalg.norm(P - c, axis=1) < 0.045]
        if len(Q) < 40:
            continue
        yaw = math.atan2(ax[1], ax[0]); pitch = math.asin(max(-1.0, min(1.0, ax[2])))
        p, _ = _best_fit(Q[:2500], [[c[0], c[1], c[2], yaw, pitch]])
        cands.append(("prior", p))
    refined = []
    for n_, (src, p) in enumerate(cands):
        p2, rms2 = _refine_on_neighbourhood(P, p, a, b, tol, seed=n_, tilt=tilt)
        if rms2 is None:
            continue
        refined.append({"src": src, "p": p2, "rms": rms2})
    diag["candidates"] = len(refined)

    # --- step 4: greedy selection with view check ---
    claimed = np.zeros(len(P), dtype=bool)
    for c_ in refined:
        r = _surface_res(c_["p"], P, a, b)
        c_["inl"] = (r < tol) & (np.linalg.norm(P - c_["p"][:3], axis=1) < 0.05)
        c_["view"] = egg_view_check(c_["p"], a, b, depth_m, K, T_world_opt, stride=max(2, stride), tol_m=max(0.006, 3.0 * sigma))
    min_inl = int(float(os.environ.get("NERO_V310_MIN_INLIERS", "150")) * (rgb.shape[0] * rgb.shape[1]) / (1280.0 * 720.0) / (stride * stride))
    selected = []
    pool = list(refined)
    while pool:
        best, best_score = None, 0.0
        for c_ in pool:
            n_free = int(np.count_nonzero(c_["inl"] & ~claimed))
            st = c_["view"]["see_through_ratio"]
            score = n_free * max(0.0, 1.0 - 3.0 * st)
            c_["n_free"] = n_free
            if score > best_score:
                best, best_score = c_, score
        if best is None or best["n_free"] < max(40, min_inl) or best_score <= 0:
            break
        if any(np.linalg.norm(best["p"][:3] - s_["p"][:3]) < 0.030 for s_ in selected):
            pool = [c_ for c_ in pool if c_ is not best]
            continue
        selected.append(best)
        claimed |= best["inl"]
        pool = [c_ for c_ in pool if c_ is not best]
    diag["selected"] = len(selected)
    diag["rejected_candidates"] = len(refined) - len(selected)

    # --- step 5: ownership, refit, confidence ---
    dets = []
    # data-driven noise level for the quality terms: median fit residual of this frame's eggs
    # (works for any real sensor / filter setting; the config value only seeds the inlier tolerance)
    sigma_eff = max(0.0008, float(np.median([s_["rms"] for s_ in selected]))) if selected else sigma
    diag["sigma_eff_mm"] = round(sigma_eff * 1000.0, 2)
    if selected:
        Rm = np.stack([_surface_res(s_["p"], P, a, b) for s_ in selected], axis=1)
        own = np.argmin(Rm, axis=1)
        rmin = Rm[np.arange(len(P)), own]
        for k, s_ in enumerate(selected):
            Q = P[(own == k) & (rmin < 2.5 * tol) & (np.linalg.norm(P - s_["p"][:3], axis=1) < 0.05)]
            if len(Q) < 40:
                continue
            if len(Q) > 2500:
                Q = Q[np.random.default_rng(200 + k).choice(len(Q), 2500, replace=False)]
            pv = s_["p"].copy()
            p, rms = _best_fit(Q, [pv, [pv[0], pv[1], pv[2], pv[3] + math.pi / 2, pv[4]]])
            if np.linalg.norm(p[:3] - pv[:3]) > 0.015:       # refit wandered off: keep the selected pose
                p, rms = pv, s_["rms"]
            if rms > float(os.environ.get("NERO_V308_MAX_FIT_RMS_M", "0.006")):
                continue
            view = egg_view_check(p, a, b, depth_m, K, T_world_opt, stride=stride, tol_m=max(0.006, 3.0 * sigma_eff))
            n_inl = int(np.count_nonzero((own == k) & (rmin < tol)))
            n_exp = max(1, int((view["silhouette_px"] - view["occluded"] - view["missing"])))
            support = min(1.0, n_inl / n_exp)
            q = _confidence(view, support, rms, sigma_eff)
            d = _axis_from(p[3], p[4])
            if d[2] < 0:
                d = -d
            xyz = [float(v) for v in p[:3]]
            dets.append({
                "xyz_m": [round(v, 5) for v in xyz],
                "axis_world": [round(float(v), 5) for v in d],
                "quat_xyzw": [round(v, 6) for v in _quat_from_z_to(d)],
                "fit_rms_mm": round(rms * 1000.0, 2),
                "n_points": int(n_inl),
                "location": classify_location(xyz),
                "confidence": q["confidence"],
                "trusted": q["trusted"],
                "quality": dict(q["terms"], source=s_["src"], silhouette_px=view["silhouette_px"],
                                occluded_px=view["occluded"], see_through_px=view["see_through"]),
            })
    merged = []
    for dd in sorted(dets, key=lambda e: -e["confidence"]):
        if all(np.linalg.norm(np.subtract(dd["xyz_m"], m["xyz_m"])) > 0.030 for m in merged):
            merged.append(dd)
    merged.sort(key=lambda e: (e["xyz_m"][0], e["xyz_m"][1]))
    for i, dd in enumerate(merged, 1):
        dd["det_id"] = f"det_{i}"
    diag["trusted"] = sum(1 for d in merged if d["trusted"])
    return merged, diag


def _confidence(view, support, rms, sigma):
    """V3.10 per-detection confidence in [0,1] and trusted flag.
    coverage   : visible fraction of the predicted silhouette (low -> occluded, axis poorly observed)
    support    : surface inliers / visible silhouette pixels (low -> fragment or wrong pose)
    see_through: fraction of visible silhouette where the camera sees BEHIND the predicted surface
    rms        : fit residual relative to the expected depth noise"""
    cov = float(view["coverage"]); st = float(view["see_through_ratio"])
    rms_ratio = rms / max(1e-4, sigma)
    c_cov = min(1.0, max(0.0, (cov - 0.30) / 0.40))
    c_sup = min(1.0, max(0.0, (support - 0.30) / 0.40))
    c_st = min(1.0, max(0.0, 1.0 - st / 0.12))
    c_rms = min(1.0, max(0.0, (3.0 - rms_ratio) / 1.5))
    conf = round(float((c_cov * c_sup * c_st * c_rms) ** 0.25), 3) if min(c_cov, c_sup, c_st, c_rms) > 0 else 0.0
    thr = float(os.environ.get("NERO_V310_TRUST_MIN_CONF", "0.70"))
    trusted = bool(conf >= thr and cov >= float(os.environ.get("NERO_V310_TRUST_MIN_COVERAGE", "0.55"))
                   and st <= float(os.environ.get("NERO_V310_TRUST_MAX_SEE_THROUGH", "0.06")))
    return {"confidence": conf, "trusted": trusted,
            "terms": {"coverage": round(cov, 3), "support": round(float(support), 3), "see_through_ratio": round(st, 3),
                      "rms_over_sigma": round(float(rms_ratio), 2)}}


# ----------------------------------------------------------------------------
# Evaluation against ground truth
# ----------------------------------------------------------------------------

def _axis_from_quat_xyzw(q):
    x, y, z, w = q
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def evaluate(dets, truth, gate_m=None):
    """Greedy nearest matching (evaluation only). truth: {name: {xyz_m, quat_xyzw, location}}.
    V3.09: gate 25 mm (was 50) so a spurious blob near an egg counts as a miss + false positive."""
    if gate_m is None:
        gate_m = float(os.environ.get("NERO_V308_MATCH_GATE_M", "0.025"))
    pairs = []
    for tn, tv in truth.items():
        for dd in dets:
            pairs.append((float(np.linalg.norm(np.subtract(dd["xyz_m"], tv["xyz_m"]))), tn, dd["det_id"]))
    pairs.sort()
    used_t, used_d, match = set(), set(), {}
    for dist, tn, di in pairs:
        if dist <= gate_m and tn not in used_t and di not in used_d:
            used_t.add(tn); used_d.add(di); match[tn] = di
    byid = {d["det_id"]: d for d in dets}
    rows = []
    for tn, tv in truth.items():
        row = {"truth_name": tn, "truth_location": tv["location"], "detected": tn in match}
        if tn in match:
            d = byid[match[tn]]
            e = np.subtract(d["xyz_m"], tv["xyz_m"])
            at = _axis_from_quat_xyzw(tv["quat_xyzw"])
            ad = np.asarray(d["axis_world"])
            cosang = abs(float(np.dot(_unit(at), _unit(ad))))
            row.update({
                "det_id": d["det_id"], "err_3d_mm": round(float(np.linalg.norm(e)) * 1000, 2),
                "err_xy_mm": round(float(np.linalg.norm(e[:2])) * 1000, 2), "err_z_mm": round(float(e[2]) * 1000, 2),
                "axis_err_deg": round(math.degrees(math.acos(min(1.0, cosang))), 2),
                "fit_rms_mm": d["fit_rms_mm"], "n_points": d["n_points"], "det_location": d["location"],
                "location_match": d["location"] == tv["location"],
                "confidence": d.get("confidence"), "trusted": d.get("trusted"),
                "coverage": (d.get("quality") or {}).get("coverage"),
            })
        rows.append(row)
    fps = [d for d in dets if d["det_id"] not in used_d]
    return rows, fps, match


def summarize(rows, fps):
    det = [r for r in rows if r["detected"]]
    def stat(key, rs=None):
        v = [abs(r[key]) for r in (det if rs is None else rs)]
        return {"mean": round(float(np.mean(v)), 2), "max": round(float(np.max(v)), 2)} if v else None
    tr = [r for r in det if r.get("trusted") is not False]
    return {"truth": len(rows), "detected": len(det), "false_positives": len(fps),
            "err_xy_mm": stat("err_xy_mm"), "err_z_mm": stat("err_z_mm"), "err_3d_mm": stat("err_3d_mm"),
            "axis_err_deg": stat("axis_err_deg"),
            "location_match": sum(1 for r in det if r.get("location_match")),
            # V3.10: what a perception-driven skill would actually be allowed to use
            "trusted": {"detected": len(tr), "untrusted_detections": len(det) - len(tr),
                        "false_positives": sum(1 for f in fps if f.get("trusted") is not False),
                        "err_xy_mm": stat("err_xy_mm", tr), "axis_err_deg": stat("axis_err_deg", tr),
                        "location_match": sum(1 for r in tr if r.get("location_match"))}}


# ----------------------------------------------------------------------------
# Files
# ----------------------------------------------------------------------------

def _png_chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png_rgb8(path, rgb):
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[i].tobytes() for i in range(h))
    data = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) \
        + _png_chunk(b"IDAT", zlib.compress(raw, 6)) + _png_chunk(b"IEND", b"")
    Path(path).write_bytes(data)


def write_png_gray16(path, img_u16):
    img = np.ascontiguousarray(img_u16.astype(">u2"))
    h, w = img.shape
    raw = b"".join(b"\x00" + img[i].tobytes() for i in range(h))
    data = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 0, 0, 0, 0)) \
        + _png_chunk(b"IDAT", zlib.compress(raw, 6)) + _png_chunk(b"IEND", b"")
    Path(path).write_bytes(data)


def _draw_cross(img, u, v, color, r=9, t=1):
    h, w = img.shape[:2]
    u, v = int(round(u)), int(round(v))
    for k in range(-r, r + 1):
        for s in range(-t, t + 1):
            for (x, y) in ((u + k, v + s), (u + s, v + k)):
                if 0 <= x < w and 0 <= y < h:
                    img[y, x] = color


def _draw_line(img, p0, p1, color):
    h, w = img.shape[:2]
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    if not np.isfinite(n) or n > 5000:
        return
    for s in np.linspace(0, 1, n):
        x = int(round(p0[0] + s * (p1[0] - p0[0])))
        y = int(round(p0[1] + s * (p1[1] - p0[1])))
        if 0 <= x < w and 0 <= y < h:
            img[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = color


def overlay(rgb, dets, truth, K, T):
    """Green = trusted perceived centre + long axis; yellow = untrusted (V3.10); red = ground truth."""
    img = rgb.copy()
    for color, items in (((230, 40, 40), [(t["xyz_m"], _axis_from_quat_xyzw(t["quat_xyzw"])) for t in truth.values()]),
                         ((40, 220, 60), [(d["xyz_m"], np.asarray(d["axis_world"])) for d in dets if d.get("trusted", True)]),
                         ((250, 220, 0), [(d["xyz_m"], np.asarray(d["axis_world"])) for d in dets if not d.get("trusted", True)])):
        for xyz, ax in items:
            c = np.asarray(xyz, float)
            ends = np.stack([c - 0.031 * _unit(ax), c + 0.031 * _unit(ax), c])
            uv, z = project(ends, K, T)
            if np.all(z > 0):
                _draw_line(img, uv[0], uv[1], color)
                _draw_cross(img, uv[2, 0], uv[2, 1], color)
    return img


CSV_FIELDS = ["capture", "wall_time", "sim_time_s", "tag", "truth_name", "truth_location", "detected", "det_id",
              "err_3d_mm", "err_xy_mm", "err_z_mm", "axis_err_deg", "fit_rms_mm", "n_points", "det_location",
              "location_match", "confidence", "trusted", "coverage"]


class CaptureWriter:
    def __init__(self, root, cfg, K, T, save_images=True):
        self.run_dir = Path(root) / time.strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.save_images = bool(save_images)
        self.csv_path = self.run_dir / "errors.csv"
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
        self.summary = {"version": VERSION, "captures": []}
        info = {
            "version": VERSION, "camera": cfg["name"], "model": "Intel RealSense D435i (simulated, depth aligned to color)",
            "width": K["width"], "height": K["height"], "fx": K["fx"], "fy": K["fy"], "ppx": K["ppx"], "ppy": K["ppy"],
            "distortion_model": "none", "coeffs": [0.0] * 5, "hfov_deg": round(K["hfov_deg"], 3), "vfov_deg": K["vfov_deg"],
            "depth_scale_m_per_unit": 0.001, "depth_png": "16-bit, millimetres, 0 = no data (RealSense z16 style)",
            "min_range_m": cfg["min_range_m"], "max_range_m": cfg["max_range_m"], "depth_noise_k": cfg["depth_noise_k"],
            "pos_world_m": cfg["pos"], "target_world_m": cfg["target"],
            "T_world_camera_optical": [[round(float(v), 7) for v in row] for row in T],
            "optical_frame": "x right, y down, z forward (OpenCV / RealSense)",
            "perception_inputs": "RGB + aligned depth + camera calibration + registered receptacle geometry; NO object ground truth",
            "depth_temporal_frames": int(cfg.get("depth_frames", 1)),
            "depth_temporal_filter": "per-pixel mean of valid samples over N frames of a static scene (RealSense temporal-filter analogue)",
            "trusted_rule": {"min_confidence": float(os.environ.get("NERO_V310_TRUST_MIN_CONF", "0.70")),
                             "min_coverage": float(os.environ.get("NERO_V310_TRUST_MIN_COVERAGE", "0.55")),
                             "max_see_through": float(os.environ.get("NERO_V310_TRUST_MAX_SEE_THROUGH", "0.06"))},
        }
        (self.run_dir / "camera_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    def write(self, seq, tag, sim_time, rgb, depth_m, qpos, dets, truth, rows, fps, summ, diag, K, T, elapsed):
        safe = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in tag)[:60]
        cdir = self.run_dir / f"capture_{seq:04d}_{safe}"
        cdir.mkdir(exist_ok=True)
        if self.save_images:
            write_png_rgb8(cdir / "rgb.png", rgb)
            write_png_gray16(cdir / "depth_mm.png", np.clip(np.round(depth_m * 1000.0), 0, 65535).astype(np.uint16))
            write_png_rgb8(cdir / "overlay.png", overlay(rgb, dets, truth, K, T))
        np.save(cdir / "qpos.npy", np.asarray(qpos, dtype=np.float64))
        perceived = {"source": "rgbd_perception", "objects": {d["det_id"]: {
            "category": "egg", "xyz_m": d["xyz_m"], "quat_xyzw": d["quat_xyzw"], "location": d["location"],
            "fit_rms_mm": d["fit_rms_mm"], "n_points": d["n_points"],
            "confidence": d.get("confidence"), "trusted": d.get("trusted"), "quality": d.get("quality")} for d in dets}}
        (cdir / "perceived_scene.json").write_text(json.dumps(perceived, indent=2), encoding="utf-8")
        (cdir / "truth_scene.json").write_text(json.dumps({"source": "mujoco_ground_truth", "objects": truth}, indent=2), encoding="utf-8")
        errs = {"capture": seq, "tag": tag, "sim_time_s": sim_time, "elapsed_s": round(elapsed, 3),
                "summary": summ, "per_object": rows, "false_positives": fps, "diagnostics": diag}
        (cdir / "errors.json").write_text(json.dumps(errs, indent=2), encoding="utf-8")
        wall = time.strftime("%H:%M:%S")
        with open(self.csv_path, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            for r in rows:
                wr.writerow({**{k: r.get(k, "") for k in CSV_FIELDS}, "capture": seq, "wall_time": wall,
                             "sim_time_s": round(sim_time, 3), "tag": tag})
            for d in fps:
                wr.writerow({"capture": seq, "wall_time": wall, "sim_time_s": round(sim_time, 3), "tag": tag,
                             "truth_name": "", "detected": "false_positive", "det_id": d["det_id"],
                             "fit_rms_mm": d["fit_rms_mm"], "n_points": d["n_points"], "det_location": d["location"],
                             "confidence": d.get("confidence"), "trusted": d.get("trusted"),
                             "coverage": (d.get("quality") or {}).get("coverage")})
        self.summary["captures"].append({"capture": seq, "tag": tag, "dir": cdir.name, **summ})
        (self.run_dir / "summary.json").write_text(json.dumps(self._aggregate(), indent=2), encoding="utf-8")
        return cdir

    def _aggregate(self):
        caps = self.summary["captures"]
        out = dict(self.summary)
        tot_t = sum(c["truth"] for c in caps)
        tot_d = sum(c["detected"] for c in caps)
        xy = [c["err_xy_mm"]["mean"] for c in caps if c.get("err_xy_mm")]
        ax = [c["axis_err_deg"]["mean"] for c in caps if c.get("axis_err_deg")]
        xym = [c["err_xy_mm"]["max"] for c in caps if c.get("err_xy_mm")]
        axm = [c["axis_err_deg"]["max"] for c in caps if c.get("axis_err_deg")]
        trs = [c.get("trusted") or {} for c in caps]
        t_xy = [t["err_xy_mm"]["max"] for t in trs if t.get("err_xy_mm")]
        t_ax = [t["axis_err_deg"]["max"] for t in trs if t.get("axis_err_deg")]
        t_xy_mean = [t["err_xy_mm"]["mean"] for t in trs if t.get("err_xy_mm")]
        t_ax_mean = [t["axis_err_deg"]["mean"] for t in trs if t.get("axis_err_deg")]
        out["overall"] = {"captures": len(caps), "detection_rate": round(tot_d / tot_t, 3) if tot_t else None,
                          "false_positives": sum(c["false_positives"] for c in caps),
                          "err_xy_mm_mean_of_means": round(float(np.mean(xy)), 2) if xy else None,
                          "axis_err_deg_mean_of_means": round(float(np.mean(ax)), 2) if ax else None,
                          "err_xy_mm_max": round(float(np.max(xym)), 2) if xym else None,
                          "axis_err_deg_max": round(float(np.max(axm)), 2) if axm else None}
        t_det = sum(t.get("detected", 0) for t in trs)
        out["overall_trusted"] = {
            "detected": t_det, "of_truth": tot_t,
            "trusted_detection_rate": round(t_det / tot_t, 3) if tot_t else None,
            "untrusted_detections": sum(t.get("untrusted_detections", 0) for t in trs),
            "false_positives": sum(t.get("false_positives", 0) for t in trs),
            "err_xy_mm_mean_of_means": round(float(np.mean(t_xy_mean)), 2) if t_xy_mean else None,
            "axis_err_deg_mean_of_means": round(float(np.mean(t_ax_mean)), 2) if t_ax_mean else None,
            "err_xy_mm_max": round(float(np.max(t_xy)), 2) if t_xy else None,
            "axis_err_deg_max": round(float(np.max(t_ax)), 2) if t_ax else None,
            "note": "trusted = what a perception-driven skill may use; untrusted -> re-capture / look again"}
        return out


# ----------------------------------------------------------------------------
# Camera process
# ----------------------------------------------------------------------------

class PerceptionTracker:
    """V3.11: keep egg identities across captures using perception only.

    Identity seeding (SIM ONLY): the first time a name has no track, it is seeded from the evaluation
    matcher (truth bridge) so perceived names line up with the simulator bodies that the remaining
    in-hand / contact checks still read.  After seeding, names follow the detections:
      1. existing tracks <- detections within 30 mm (15 mm if untrusted), closest first   static eggs
      2. names never seen <- seeded from the matcher                             first sighting
      3. remaining tracks <- remaining TRUSTED detections, nearest first         an egg that was moved
    Poses handed to the skills always come from perception, never from truth."""

    def __init__(self, gate_static_m=0.030, gate_moved_m=0.60):
        self.tracks = {}
        self.gate_static = float(gate_static_m)
        self.gate_moved = float(gate_moved_m)

    def update(self, dets, seq, seed_match=None):
        byid = {d["det_id"]: d for d in dets}
        assigned, used = {}, set()
        pairs = []
        for name, t in self.tracks.items():
            for d in dets:
                pairs.append((float(np.linalg.norm(np.subtract(d["xyz_m"], t["xyz_m"]))), name, d["det_id"]))
        pairs.sort()
        for dist, name, did in pairs:
            gate = self.gate_static if byid[did].get("trusted") else 0.5 * self.gate_static   # untrusted: 15 mm
            if dist <= gate and name not in assigned and did not in used:
                assigned[name] = did; used.add(did)
        seeded = []
        for name, did in (seed_match or {}).items():
            if name not in self.tracks and name not in assigned and did not in used and did in byid:
                assigned[name] = did; used.add(did); seeded.append(name)
        pairs2 = sorted((dist, name, did) for dist, name, did in pairs
                        if name not in assigned and did not in used and byid[did].get("trusted"))
        moved = []
        for dist, name, did in pairs2:
            if dist <= self.gate_moved and name not in assigned and did not in used:
                assigned[name] = did; used.add(did); moved.append(name)
        for name, did in assigned.items():
            d = byid[did]
            t = self.tracks.setdefault(name, {"first_seq": seq})
            t.update({"xyz_m": list(d["xyz_m"]), "quat_xyzw": list(d["quat_xyzw"]), "axis_world": list(d["axis_world"]),
                      "location": d["location"], "confidence": d.get("confidence"), "trusted": bool(d.get("trusted")),
                      "coverage": (d.get("quality") or {}).get("coverage"), "det_id": did, "last_seq": seq})
        out = {}
        for name, t in sorted(self.tracks.items()):
            seen = name in assigned
            out[name] = {"xyz_m": t["xyz_m"], "quat_xyzw": t["quat_xyzw"], "axis_world": t["axis_world"],
                         "location": t["location"], "seen": seen, "trusted": bool(seen and t.get("trusted")),
                         "confidence": t.get("confidence") if seen else 0.0, "coverage": t.get("coverage") if seen else None,
                         "det_id": t.get("det_id") if seen else None, "captures_since_seen": int(seq - t.get("last_seq", seq))}
        unassigned = [did for did in byid if did not in used]
        return out, {"seeded": seeded, "reassociated_moved": moved, "unassigned_detections": unassigned}


def _identity_check(tracked, match):
    """Evaluation only: does each tracked name point at the detection the truth matcher gives that egg?"""
    ok, bad = 0, []
    for name, o in tracked.items():
        if not o.get("seen"):
            continue
        if match.get(name) == o.get("det_id"):
            ok += 1
        else:
            bad.append(name)
    return {"consistent": ok, "inconsistent": bad}


def _post_result(res_seq, res_buf, seq, payload):
    """V3.11: publish one capture result for a waiting skill (JSON in a shared byte buffer)."""
    if res_seq is None or res_buf is None:
        return
    b = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if len(b) >= len(res_buf):
        b = json.dumps({"seq": seq, "error": "result too large"}).encode("utf-8")
    with res_buf.get_lock():
        res_buf[:len(b)] = b
        res_buf[len(b)] = b"\0"
    with res_seq.get_lock():
        res_seq.value = int(seq)


def _temporal_annotate(dets, prev):
    """V3.10: record consistency with the previous capture (eggs legitimately move when grasped,
    so this is logged for analysis and used as a warm start, not as a trust veto)."""
    for d in dets:
        best = None
        for p in prev or []:
            dist = float(np.linalg.norm(np.subtract(d["xyz_m"], p["xyz_m"])))
            if best is None or dist < best[0]:
                best = (dist, p)
        q = d.setdefault("quality", {})
        if best is None or best[0] > 0.030:
            q["prev_capture"] = None
            continue
        cosang = abs(float(np.dot(_unit(d["axis_world"]), _unit(best[1]["axis_world"]))))
        q["prev_capture"] = {"delta_mm": round(best[0] * 1000.0, 2),
                             "delta_axis_deg": round(math.degrees(math.acos(min(1.0, cosang))), 2),
                             "prev_trusted": bool(best[1].get("trusted"))}


def _depth_postprocess(depth, cfg, rng):
    d = depth.astype(np.float32).copy()
    k = float(cfg["depth_noise_k"])
    if k > 0:
        d += rng.standard_normal(d.shape).astype(np.float32) * (k * d * d)
    d[(d < cfg["min_range_m"]) | (d > cfg["max_range_m"]) | ~np.isfinite(d)] = 0.0
    return np.round(d * 1000.0) / 1000.0              # z16 millimetre quantisation


def _truth_from_data(mujoco, model, data):
    truth = {}
    for i in range(1, 5):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obj_egg{i}")
        if bid < 0:
            continue
        p = [float(v) for v in data.xpos[bid]]
        w, x, y, z = [float(v) for v in data.xquat[bid]]
        truth[f"egg_{i}"] = {"xyz_m": [round(v, 5) for v in p], "quat_xyzw": [round(x, 6), round(y, 6), round(z, 6), round(w, 6)],
                             "location": classify_location(p)}
    return truth


def _robot_geom_ids(mujoco, model):
    """All geoms on the arm + gripper: descendants of the body that owns joint1."""
    try:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint1")
        root = int(model.jnt_bodyid[jid])
    except Exception:
        return []
    parent = np.asarray(model.body_parentid)
    def is_desc(b):
        while b > 0:
            if b == root:
                return True
            b = int(parent[b])
        return False
    bodies = {b for b in range(model.nbody) if is_desc(b)}
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in bodies]


def _robot_boxes(model, data, geom_ids):
    """Oriented bounding boxes of robot geoms at the current joint state."""
    boxes = []
    for g in geom_ids:
        aabb = np.asarray(model.geom_aabb[g], dtype=float)      # local centre(3) + half sizes(3)
        R = np.asarray(data.geom_xmat[g], dtype=float).reshape(3, 3)
        c = np.asarray(data.geom_xpos[g], dtype=float) + R @ aabb[:3]
        boxes.append((c, R, aabb[3:]))
    return boxes


def _self_check(truth, depth, K, T):
    """Projection-convention check: depth at a projected egg centre must be just in front of it."""
    ok = bad = 0
    for tv in truth.values():
        uv, z = project(np.asarray([tv["xyz_m"]]), K, T)
        u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
        if not (0 <= u < K["width"] and 0 <= v < K["height"]) or z[0] <= 0:
            continue
        dz = float(depth[v, u])
        if dz <= 0:
            continue
        if z[0] - 0.040 <= dz <= z[0] + 0.004:
            ok += 1
        else:
            bad += 1
    return ok, bad


def camera_process_main(qpos_arr, meta_arr, model_path_arr, trig_seq, trig_tag, status, stop_event, out_root, gl_name,
                        res_seq=None, res_buf=None):
    """Entry point of the spawned camera process. MUJOCO_GL was set by the parent before spawn."""
    prefix = f"[{VERSION} CAMERA]"
    try:
        import mujoco
    except Exception as exc:
        print(f"{prefix} mujoco import failed: {exc}", flush=True)
        status.value = 2
        return
    cfg = camera_config()
    t_wait = time.time()
    path = ""
    while not stop_event.is_set():
        path = bytes(model_path_arr[:]).split(b"\0", 1)[0].decode("utf-8", "ignore")
        if path and meta_arr[3] > 0.5:
            break
        if time.time() - t_wait > 180:
            print(f"{prefix} simulator never published its model; camera disabled", flush=True)
            status.value = 2
            return
        time.sleep(0.2)
    if stop_event.is_set():
        return
    try:
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME)
        if cam_id < 0:
            raise RuntimeError(f"camera {CAMERA_NAME} not in model {path}")
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(cfg["width"]))
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(cfg["height"]))
    except Exception as exc:
        print(f"{prefix} model load failed: {exc}", flush=True)
        status.value = 2
        return
    try:
        renderer = mujoco.Renderer(model, height=int(cfg["height"]), width=int(cfg["width"]))
    except Exception as exc:
        print(f"{prefix} OpenGL backend '{gl_name}' unavailable: {str(exc)[:200]}", flush=True)
        status.value = 2
        raise SystemExit(EXIT_GL_FAILED)

    K = intrinsics(cfg)
    T = extrinsics(cfg)
    writer = CaptureWriter(out_root, cfg, K, T, save_images=os.environ.get("NERO_V308_SAVE_IMAGES", "1") != "0")

    period = float(os.environ.get("NERO_V308_PERIOD_S", "0"))
    last_seq = int(trig_seq.value)
    last_periodic = time.time()
    capture_no = 0
    self_checked = False
    rng = np.random.default_rng(308)
    nq = int(model.nq)
    robot_geoms = _robot_geom_ids(mujoco, model)
    prev_dets = []
    tracker = PerceptionTracker()
    print(f"{prefix} robot self-filter: {len(robot_geoms)} robot geoms (from joint states, not object truth)", flush=True)
    # V3.10: trigger baseline is taken BEFORE announcing READY, so a trigger sent right after READY is never lost
    last_seq = int(trig_seq.value)
    status.value = 1
    print(f"{prefix} READY gl={gl_name} {K['width']}x{K['height']} fx={K['fx']:.1f} hfov={K['hfov_deg']:.1f}deg "
          f"vfov={K['vfov_deg']:.1f}deg pos={cfg['pos']} depth_frames={cfg['depth_frames'] if cfg['depth_noise_k'] > 0 else 1} "
          f"-> {writer.run_dir}", flush=True)
    while not stop_event.is_set():
        tag = None
        cur = int(trig_seq.value)
        if cur != last_seq:
            last_seq = cur
            tag = bytes(trig_tag[:]).split(b"\0", 1)[0].decode("utf-8", "ignore") or "trigger"
        elif period > 0 and time.time() - last_periodic >= period:
            tag = "periodic"
        if tag is None:
            time.sleep(0.05)
            continue
        last_periodic = time.time()
        t0 = time.perf_counter()
        done_seq = cur
        try:
            with qpos_arr.get_lock():
                n = int(meta_arr[0])
                q = np.frombuffer(qpos_arr.get_obj(), dtype=np.float64, count=n).copy()
                sim_time = float(meta_arr[1])
            if n != nq:
                raise RuntimeError(f"qpos size mismatch sim={n} camera_model={nq}")
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.disable_depth_rendering()
            renderer.update_scene(data, camera=cam_id)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=cam_id)
            raw_depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            # V3.10: N frames of the same static snapshot, each with its own sensor noise, fused per pixel
            n_fr = int(cfg["depth_frames"]) if float(cfg["depth_noise_k"]) > 0 else 1
            depth = temporal_fuse([_depth_postprocess(raw_depth, cfg, rng) for _ in range(n_fr)])
            truth = _truth_from_data(mujoco, model, data)
            if not self_checked:
                ok, bad = _self_check(truth, depth, K, T)
                self_checked = True
                verdict = "PASS" if bad == 0 and ok > 0 else ("FAIL" if bad > 0 else "INCONCLUSIVE")
                print(f"{prefix} SELF-CHECK {verdict} | projected egg centres consistent={ok} inconsistent={bad}", flush=True)
            boxes = _robot_boxes(model, data, robot_geoms)
            prior = [d for d in prev_dets if d.get("trusted")] if os.environ.get("NERO_V310_WARM_START", "1") != "0" else []
            dets, diag = perceive_eggs(rgb, depth, K, T, robot_boxes=boxes, prior=prior)
            diag["depth_frames_fused"] = n_fr
            _temporal_annotate(dets, prev_dets)
            prev_dets = dets
            rows, fps, match = evaluate(dets, truth)
            summ = summarize(rows, fps)
            tracked, track_diag = tracker.update(dets, capture_no + 1, seed_match=match)
            idc = _identity_check(tracked, match)
            diag["tracking"] = dict(track_diag, identity_check=idc)
            capture_no += 1
            elapsed = time.perf_counter() - t0
            cdir = writer.write(capture_no, tag, sim_time, rgb, depth, q, dets, truth, rows, fps, summ, diag, K, T, elapsed)
            try:
                (cdir / "tracked_scene.json").write_text(json.dumps({"source": "rgbd_perception_tracked", "objects": tracked},
                                                                     indent=2), encoding="utf-8")
            except Exception:
                pass
            _post_result(res_seq, res_buf, done_seq, {
                "seq": done_seq, "capture": capture_no, "tag": tag, "sim_time_s": round(sim_time, 3),
                "elapsed_s": round(elapsed, 3), "objects": tracked, "n_detections": len(dets),
                "unassigned_detections": track_diag["unassigned_detections"], "dir": cdir.name})
            xy, ax, zz = summ["err_xy_mm"], summ["axis_err_deg"], summ["err_z_mm"]
            tr = summ["trusted"]
            print(f"[{VERSION} PERCEPTION] #{capture_no} {tag} | detected {summ['detected']}/{summ['truth']} "
                  f"(+{summ['false_positives']} FP) | trusted {tr['detected']} untrusted {tr['untrusted_detections']} "
                  f"trusted-FP {tr['false_positives']} | xy mean {xy['mean'] if xy else '-'} max {xy['max'] if xy else '-'} mm | "
                  f"z mean {zz['mean'] if zz else '-'} mm | axis mean {ax['mean'] if ax else '-'} max {ax['max'] if ax else '-'} deg | "
                  f"frames {n_fr} | {elapsed:.2f}s -> {cdir.name}", flush=True)
            if idc["inconsistent"]:
                print(f"[{VERSION} TRACKING WARN] #{capture_no} identity differs from truth matcher for {idc['inconsistent']}", flush=True)
        except Exception as exc:
            print(f"[{VERSION} PERCEPTION ERROR] {tag}: {exc}", flush=True)
            traceback.print_exc()
            try:
                _post_result(res_seq, res_buf, done_seq, {"seq": done_seq, "tag": tag, "error": str(exc)[:300]})
            except Exception:
                pass
    try:
        renderer.close()
    except Exception:
        pass
    status.value = 0
