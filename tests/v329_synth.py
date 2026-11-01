"""V3.29 test helper: synthetic aligned depth of the egg scene (table + MJCF-like holder + eggs) by z-buffer
splatting of dense surface samples. Geometry copied from the backend MJCF builder (holder at (-0.420, 0.155))."""
import math
import numpy as np

SLOT_XS = (-0.510, -0.450, -0.390, -0.330)
SLOT_YS = (0.123, 0.187)


def _box(c, half, n=0.002):
    c = np.asarray(c, float); h = np.asarray(half, float)
    pts = []
    for ax in range(3):
        o = [a for a in range(3) if a != ax]
        g0 = np.arange(-h[o[0]], h[o[0]] + 1e-9, n); g1 = np.arange(-h[o[1]], h[o[1]] + 1e-9, n)
        A, Bm = np.meshgrid(g0, g1)
        for sgn in (-1, 1):
            P = np.zeros((A.size, 3)); P[:, o[0]] = A.ravel(); P[:, o[1]] = Bm.ravel(); P[:, ax] = sgn * h[ax]
            pts.append(P + c)
    return np.vstack(pts)


def _cyl(x, y, z0, z1, r, n=0.0015):
    zz = np.arange(z0, z1 + 1e-9, n)
    aa = np.arange(0, 2 * math.pi, n / r)
    A, Z = np.meshgrid(aa, zz)
    side = np.stack([x + r * np.cos(A.ravel()), y + r * np.sin(A.ravel()), Z.ravel()], axis=1)
    rr = np.arange(0, r, n)
    R_, A2 = np.meshgrid(rr, aa)
    top = np.stack([x + (R_ * np.cos(A2)).ravel(), y + (R_ * np.sin(A2)).ravel(), np.full(R_.size, z1 + r)], axis=1)
    return np.vstack([side, top])


def _ellipsoid(c, axes, R, n=0.0015):
    a = max(axes)
    m = int(2 * math.pi * a / n)
    th = np.linspace(0, math.pi, m // 2); ph = np.linspace(0, 2 * math.pi, m)
    T_, P_ = np.meshgrid(th, ph)
    loc = np.stack([axes[0] * np.sin(T_) * np.cos(P_), axes[1] * np.sin(T_) * np.sin(P_), axes[2] * np.cos(T_)], -1).reshape(-1, 3)
    return loc @ np.asarray(R).T + np.asarray(c)


def _carton_local(n=0.0015):
    """V3.33 moulded-pulp carton surface in the holder frame from the CAD (W.carton_height): top height field +
    the four vertical outer walls. -> points (N,3), normals (N,3)."""
    import NERO_V326_Wrist_Camera as W
    m = W.holder_model()
    hu, hv = m["outer"][0] / 2.0, m["outer"][1] / 2.0
    us, vs = np.arange(-hu, hu + 1e-9, n), np.arange(-hv, hv + 1e-9, n)
    U, V = np.meshgrid(us, vs, indexing="ij")
    Z = W.carton_height(U, V)
    gu, gv = np.gradient(Z, n, n)
    N = np.stack([-gu, -gv, np.ones_like(Z)], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    P = [np.stack([U.ravel(), V.ravel(), Z.ravel()], 1)]
    NN = [N.reshape(-1, 3)]
    top = m["plateau_z"]
    zz = np.arange(0.0, top + 1e-9, n)
    for fixed, along, normal in (("u", vs, (1, 0)), ("u", vs, (-1, 0)), ("v", us, (0, 1)), ("v", us, (0, -1))):
        A, Zw = np.meshgrid(along, zz)
        if fixed == "u":
            pts = np.stack([np.full(A.size, hu * normal[0]), A.ravel(), Zw.ravel()], 1)
        else:
            pts = np.stack([A.ravel(), np.full(A.size, hv * normal[1]), Zw.ravel()], 1)
        P.append(pts)
        NN.append(np.tile([normal[0], normal[1], 0.0], (len(pts), 1)))
    return np.vstack(P), np.vstack(NN)


def _pose(P, cx, cy, yaw, N=None):
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    Pw = P @ Rz.T + np.array([cx, cy, 0.0])
    return Pw if N is None else (Pw, N @ Rz.T)


def holder_points(cx=-0.420, cy=0.155, yaw=0.0):
    """V3.33: surface points of the moulded-pulp carton (what a depth camera sees)."""
    P, _N = _carton_local()
    return _pose(P, cx, cy, yaw)


def slot_truth(cx=-0.420, cy=0.155, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    out = {}
    k = 0
    for sy in SLOT_YS:
        for sx in SLOT_XS:
            k += 1
            du, dv = sx + 0.420, sy - 0.155
            out[f"slot_{k}"] = (cx + du * c - dv * s, cy + du * s + dv * c, 0.043)
    return out


def egg_points(c, horizontal=True, yaw=0.0, a=0.0315, b=0.0219):
    if horizontal:
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        R = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1]])
        return _ellipsoid(c, (a, b, b), R)
    return _ellipsoid(c, (b, b, a), np.eye(3))


def table_points(xr=(-0.72, -0.05), yr=(-0.38, 0.38), n=0.002):
    X, Y = np.meshgrid(np.arange(*xr, n), np.arange(*yr, n))
    return np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)


def render_depth(P, K, T, min_range=0.28, max_range=3.0):
    R = np.asarray(T)[:3, :3]; t = np.asarray(T)[:3, 3]
    pc = (P - t) @ R
    z = pc[:, 2]
    ok = (z > 0.05)
    u = np.round(K["fx"] * pc[ok, 0] / z[ok] + K["ppx"]).astype(int)
    v = np.round(K["fy"] * pc[ok, 1] / z[ok] + K["ppy"]).astype(int)
    zz = z[ok]
    m = (u >= 0) & (u < K["width"]) & (v >= 0) & (v < K["height"])
    u, v, zz = u[m], v[m], zz[m]
    D = np.full((K["height"], K["width"]), np.inf)
    np.minimum.at(D, (v, u), zz)
    D[~np.isfinite(D)] = 0.0
    D[(D < min_range) | (D > max_range)] = 0.0
    return D


def camera_T_for(look_cand, pc_tcp, fc_tcp):
    """Optical frame of the wrist camera for a look candidate (TCP pose from W.look_pose_candidates)."""
    import NERO_V326_Wrist_Camera as W
    R = np.array(W.matrix_from_quat(look_cand["quat_xyzw"]))
    cam = np.asarray(look_cand["xyz"]) + R @ np.asarray(pc_tcp)
    f = R @ np.asarray(fc_tcp)
    up = R @ np.array([1.0, 0.0, 0.0])                    # any roll (image axes not needed for geometry)
    x = np.cross(f, up); x /= np.linalg.norm(x); y = np.cross(f, x)
    T = np.eye(4); T[:3, :3] = np.stack([x, y, f], axis=1); T[:3, 3] = cam
    return T


# ---------------------------------------------------------------- V3.33 colour rendering (RGB-only camera tests)
# Same lights as the scene builder: headlight (ambient 0.25 + diffuse 0.30, ~view aligned), warm key light
# (castshadow), cool fill light. Colour = albedo x light, clipped - shadows keep headlight + fill only (cool grey).
KEY = (np.array([0.92, 0.84, 0.72]), -np.array([-0.30, 0.22, -1.0]) / np.linalg.norm([-0.30, 0.22, -1.0]))
FILL = (np.array([0.34, 0.40, 0.48]), -np.array([0.55, -0.25, -1.0]) / np.linalg.norm([0.55, -0.25, -1.0]))
HEAD = 0.25 + 0.27
ALB_TABLE = np.array([0.925, 0.925, 0.918])
ALB_PULP = np.array([0.41, 0.40, 0.38])
ALB_EGG = np.array([0.773, 0.459, 0.207])
ALB_FLOOR = np.array([0.40, 0.43, 0.50])
MARBLE = (236, 236, 234)        # legacy names kept for older tests
CARTON = (175, 168, 157)
WELL = (84, 81, 74)
EGG = (230, 150, 70)


def shade(albedo, normals, shadow=None):
    n = np.asarray(normals, float)
    lk = np.clip(n @ KEY[1], 0, None)[:, None] * KEY[0]
    lf = np.clip(n @ FILL[1], 0, None)[:, None] * FILL[0]
    if shadow is not None:
        lk = lk * (~np.asarray(shadow, bool))[:, None]
    return np.clip(np.asarray(albedo) * (HEAD + lk + lf), 0, 1) * 255.0


def floor_rgb():
    return tuple(shade(ALB_FLOOR, np.array([[0.0, 0.0, 1.0]]))[0])


def carton_rgb_part(cx=-0.420, cy=0.155, yaw=0.0):
    P, N = _carton_local()
    Pw, Nw = _pose(P, cx, cy, yaw, N)
    return Pw, shade(ALB_PULP, Nw)


def table_rgb_part(n=0.0015, shadows=()):
    """White table; shadows = [(x0, x1, y0, y1)] rectangles without key light (cool grey patches)."""
    P = table_points(n=n)
    sh = np.zeros(len(P), bool)
    for x0, x1, y0, y1 in shadows:
        sh |= (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
    return P, shade(ALB_TABLE, np.tile([0.0, 0.0, 1.0], (len(P), 1)), sh)


def egg_rgb_part(c, horizontal=True, yaw=0.0, a=0.0315, b=0.0219):
    if horizontal:
        cy_, sy_ = math.cos(yaw), math.sin(yaw)
        R = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1]]); axes = (a, b, b)
    else:
        R = np.eye(3); axes = (b, b, a)
    P = _ellipsoid(c, axes, R)
    loc = (P - np.asarray(c)) @ R
    N = (loc / np.asarray(axes) ** 2) @ R.T
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    return P, shade(ALB_EGG, N)


def well_points(cx=-0.420, cy=0.155, yaw=0.0, n=0.0012):
    """(legacy V3.30 helper, the V3.33 carton has no dark wells) -> empty."""
    return np.zeros((0, 3))


def render_rgb(parts, K, T, seed=0, background=None):
    """parts: [(points (N,3), rgb tuple or (N,3) colours)] -> uint8 image by z-buffer splatting (+ mild noise)."""
    R = np.asarray(T)[:3, :3]; t = np.asarray(T)[:3, 3]
    H, Wd = K["height"], K["width"]
    D = np.full((H, Wd), np.inf)
    img = np.zeros((H, Wd, 3), np.float32)
    for P, col in parts:
        if len(P) == 0:
            continue
        colv = np.asarray(col, np.float32)
        per_pt = colv.ndim == 2
        pc = (P - t) @ R
        z = pc[:, 2]
        ok = z > 0.05
        u0 = np.round(K["fx"] * pc[ok, 0] / z[ok] + K["ppx"]).astype(int)
        v0 = np.round(K["fy"] * pc[ok, 1] / z[ok] + K["ppy"]).astype(int)
        z0 = z[ok]
        c0 = colv[ok] if per_pt else None
        for du in (-1, 0, 1):                                      # 3x3 splat: no holes between samples
            for dv in (-1, 0, 1):
                u, v, zz = u0 + du, v0 + dv, z0 + 1e-4 * (abs(du) + abs(dv))
                m = (u >= 0) & (u < Wd) & (v >= 0) & (v < H)
                u, v, zz = u[m], v[m], zz[m]
                cc = c0[m] if per_pt else None
                order = np.argsort(-zz)                            # far first, near overwrites
                u, v, zz = u[order], v[order], zz[order]
                cc = cc[order] if per_pt else None
                closer = zz < D[v, u]
                u, v, zz = u[closer], v[closer], zz[closer]
                D[v, u] = zz
                img[v, u] = cc[closer] if per_pt else colv
    rng = np.random.default_rng(seed)
    img = img + rng.normal(0, 3.0, img.shape)
    img[~np.isfinite(D)] = floor_rgb() if background is None else background      # V3.33 grey floor far below
    return np.clip(img, 0, 255).astype(np.uint8)
