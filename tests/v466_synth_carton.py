"""V4.6.6 synthetic fixed-camera view of a 2x3 egg carton (for tests): oblique D435i-like pinhole camera, z-buffer
rendering of the carton height field on a table; YOLO-like mask that spills a few pixels onto the table."""
import math

import numpy as np

K = {"width": 1280, "height": 720, "fx": 900.0, "fy": 900.0, "ppx": 640.0, "ppy": 360.0}


def look_at(cam, tgt):
    z = np.asarray(tgt, float) - np.asarray(cam, float)
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = cam
    return T


def carton_height(u, v, pc=46.0, pr=47.0, R=2, C=3, flange=12.0, top=30.0, cup_r=19.0, depth=25.0, extra_mm=0.0):
    """height (mm) of the carton top surface at carton-local (u, v) mm; NaN outside the carton.  extra_mm: a flat
    strip at rim height along the +v side (wide edge / something next to the carton that the mask includes)"""
    Lh, Wh = 0.5 * C * pc + flange, 0.5 * R * pr + flange
    h = np.full(u.shape, np.nan)
    inside = (np.abs(u) <= Lh) & (np.abs(v) <= Wh)
    h[inside] = top
    strip = (np.abs(u) <= Lh) & (v > Wh) & (v <= Wh + extra_mm)
    h[strip] = top
    for r in range(R):
        for c in range(C):
            cu, cv = (c + 0.5 - 0.5 * C) * pc, (r + 0.5 - 0.5 * R) * pr
            rr = np.hypot(u - cu, v - cv)
            cup = inside & (rr < cup_r)
            h[cup] = top - depth * (1.0 - (rr[cup] / cup_r) ** 2)
    return h


def scene(centre=(-0.380, 0.060), yaw_deg=20.0, cam=(-0.10, -0.05, 0.78), spill_px=3, pc=46.0, pr=47.0, extra_mm=0.0,
          eggs_in=()):
    """eggs_in: template slot indices (0..5, row-major, carton frame) with an egg standing in the cup (V4.6.8); the
    egg pixels are NOT part of the carton mask (as YOLO often does)"""
    import cv2
    T = look_at(cam, [centre[0], centre[1], 0.0])
    yaw = math.radians(yaw_deg)
    g = np.arange(-0.16, 0.16, 0.0005)
    X, Y = np.meshgrid(centre[0] + g, centre[1] + g)
    dx, dy = X - centre[0], Y - centre[1]
    u = (dx * math.cos(yaw) + dy * math.sin(yaw)) * 1000.0
    v = (-dx * math.sin(yaw) + dy * math.cos(yaw)) * 1000.0
    h = carton_height(u, v, pc, pr, extra_mm=extra_mm)
    is_c = ~np.isnan(h)
    is_egg = np.zeros(h.shape, bool)
    for k in eggs_in:
        r_, c_ = divmod(int(k), 3)
        cu, cv = (c_ + 0.5 - 1.5) * pc, (r_ + 0.5 - 1.0) * pr
        rr = np.hypot(u - cu, v - cv)
        on = rr < 21.0
        top = 5.0 + 27.5 + 27.5 * np.sqrt(np.clip(1.0 - (rr / 21.0) ** 2, 0.0, 1.0))
        h = np.where(on, np.maximum(np.nan_to_num(h, nan=0.0), top), h)
        is_egg |= on
    is_c = is_c & ~is_egg
    Z = np.where(is_c | is_egg, np.nan_to_num(h) / 1000.0, 0.0)
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    Ti = np.linalg.inv(T)
    pc_ = P @ Ti[:3, :3].T + Ti[:3, 3]
    px = np.round(K["fx"] * pc_[:, 0] / pc_[:, 2] + K["ppx"]).astype(int)
    py = np.round(K["fy"] * pc_[:, 1] / pc_[:, 2] + K["ppy"]).astype(int)
    ok = (px >= 0) & (px < K["width"]) & (py >= 0) & (py < K["height"])
    depth = np.full((K["height"], K["width"]), np.inf)
    lab = np.zeros((K["height"], K["width"]), bool)
    order = np.argsort(-pc_[ok, 2])                        # far first, near last wins
    pxo, pyo, zo, co = px[ok][order], py[ok][order], pc_[ok, 2][order], is_c.ravel()[ok][order]
    depth[pyo, pxo] = zo
    lab[pyo, pxo] = co
    depth[~np.isfinite(depth)] = 0.0
    mask = cv2.dilate(lab.astype(np.uint8), np.ones((2 * spill_px + 1, 2 * spill_px + 1), np.uint8)).astype(bool)
    ys, xs = np.nonzero(mask)
    box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    truth = []
    for r in range(2):
        for c in range(3):
            cu, cv = (c + 0.5 - 1.5) * pc / 1000.0, (r + 0.5 - 1.0) * pr / 1000.0
            truth.append((centre[0] + cu * math.cos(yaw) - cv * math.sin(yaw),
                          centre[1] + cu * math.sin(yaw) + cv * math.cos(yaw)))
    rgb = np.full((K["height"], K["width"], 3), 90, np.uint8)
    return {"rgb": rgb, "depth": depth.astype(np.float32), "K": dict(K), "T": T,
            "dets": [{"cls": 0, "mask": mask, "conf": 0.9, "box": box, "name": "egg carton"}], "truth": truth}
