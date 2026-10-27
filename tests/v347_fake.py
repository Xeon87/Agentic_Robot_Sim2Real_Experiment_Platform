"""V3.47 test helpers: a synthetic D435i scene (table + eggs 55 x 42 mm + an AirPods-like box + an egg cut by the image
border) rendered by ray casting, and a fake `ultralytics` package (YOLOE + YOLOEVPSegPredictor) that "detects" the
rendered objects.  No real ultralytics / torch / GPU needed."""
import math
import sys
import types

import numpy as np

K = {"width": 1280, "height": 720, "fx": 908.7, "fy": 908.7, "ppx": 640.1, "ppy": 366.6}
A_EGG, B_EGG = 0.0275, 0.021                      # real eggs: 55 x 42 mm


def look_at(pos, tgt):
    """T_base_cam (optical: x right, y down, z forward)."""
    pos, tgt = np.asarray(pos, float), np.asarray(tgt, float)
    z = tgt - pos
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = pos
    return T


T_BC = look_at([-0.83, 0.26, 0.72], [-0.45, -0.05, 0.0])


def Rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def egg_R(yaw, pitch=0.0):
    """Local z = long axis, lying along yaw."""
    d = np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)])
    z = d
    x = np.cross([0.0, 0.0, 1.0], z)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def ground_point_for_pixel(u, v, z_plane):
    d = T_BC[:3, :3] @ np.array([(u - K["ppx"]) / K["fx"], (v - K["ppy"]) / K["fy"], 1.0])
    o = T_BC[:3, 3]
    t = (z_plane - o[2]) / d[2]
    return o + t * d


def default_objects():
    e3 = ground_point_for_pixel(560, 724, B_EGG)        # centre just below the image -> cut by the border
    return [
        {"name": "E1", "kind": "egg", "c": [-0.45, 0.02, B_EGG], "yaw": math.radians(30), "conf": 0.95},
        {"name": "E2", "kind": "egg", "c": [-0.38, -0.10, B_EGG], "yaw": math.radians(-60), "conf": 0.92},
        {"name": "AP", "kind": "box", "c": [-0.52, -0.08, 0.011], "yaw": math.radians(10),
         "half": [0.025, 0.0225, 0.011], "conf": 0.97},
        {"name": "E3", "kind": "egg", "c": [float(e3[0]), float(e3[1]), B_EGG], "yaw": math.radians(80), "conf": 0.51},
    ]


def render(objects, table_z=0.0, noise_m=0.0007, seed=0, shift_px=0):
    """-> rgb (H,W,3 uint8, RGB), depth (H,W float32 m, z-depth), ids (H,W int, -1 table)."""
    H, W = K["height"], K["width"]
    vv, uu = np.mgrid[0:H, 0:W].astype(float)
    dc = np.stack([(uu - K["ppx"]) / K["fx"], (vv - K["ppy"]) / K["fy"], np.ones_like(uu)], -1).reshape(-1, 3)
    D = dc @ T_BC[:3, :3].T
    O = T_BC[:3, 3]
    best = np.full(len(D), np.inf)
    ids = np.full(len(D), -1)
    t = (table_z - O[2]) / D[:, 2]
    ok = t > 0
    best[ok] = t[ok]
    for k, ob in enumerate(objects):
        c = np.asarray(ob["c"], float)
        if ob["kind"] == "egg":
            R = egg_R(ob["yaw"], ob.get("pitch", 0.0))
            sc = np.array([B_EGG, B_EGG, A_EGG]) if "size" not in ob else np.array([ob["size"][1], ob["size"][1], ob["size"][0]]) / 2
            q = (O - c) @ R / sc
            dd = D @ R / sc
            a_ = np.sum(dd * dd, 1)
            b_ = 2 * dd @ q
            c_ = q @ q - 1
            disc = b_ * b_ - 4 * a_ * c_
            hit = disc > 0
            th = np.full(len(D), np.inf)
            th[hit] = (-b_[hit] - np.sqrt(disc[hit])) / (2 * a_[hit])
        else:
            R = Rz(ob["yaw"])
            h = np.asarray(ob["half"], float)
            q = (O - c) @ R
            dd = D @ R
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (-h - q) / dd
                t2 = (h - q) / dd
            tmin = np.nanmax(np.minimum(t1, t2), 1)
            tmax = np.nanmin(np.maximum(t1, t2), 1)
            th = np.where((tmax >= tmin) & (tmin > 0), tmin, np.inf)
        closer = th < best
        best[closer] = th[closer]
        ids[closer] = k
    depth = best.reshape(H, W).astype(np.float32)
    depth[~np.isfinite(depth)] = 0
    rng = np.random.default_rng(seed)
    depth = np.where(depth > 0, depth + rng.normal(0, noise_m, depth.shape), 0).astype(np.float32)
    ids = ids.reshape(H, W)
    rgb = np.full((H, W, 3), 30, np.uint8)
    rgb[ids >= 0] = (235, 235, 230)
    rgb[..., 0] = np.clip(rgb[..., 0].astype(int) + (uu.astype(int) % 7), 0, 255).astype(np.uint8)
    return rgb, depth, ids


# =========================================================================== fake ultralytics
class _Arr:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def __len__(self):
        return len(self.a)


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _Arr(xyxy), _Arr(conf), _Arr(cls)

    def __len__(self):
        return len(self.xyxy)


class _Masks:
    def __init__(self, xy):
        self.xy = xy


class _Result:
    def __init__(self, boxes, masks):
        self.boxes, self.masks = boxes, masks


class FakeState:
    def __init__(self):
        self.scene = None            # (objects, ids)
        self.calls = []
        self.set_classes_calls = []
        self.not_egg_names = set()   # objects reported as class 1 when the reference has a class-1 box
        self.bleed_px = 2


STATE = FakeState()


class YOLOEVPSegPredictor:
    pass


class YOLOE:
    def __init__(self, path):
        import os
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        self.path = path
        self.classes = None

    def set_classes(self, names, *a):
        import os
        STATE.set_classes_calls.append((list(names), os.getcwd()))
        self.classes = list(names)

    def predict(self, source, conf=0.25, imgsz=640, verbose=True, refer_image=None, visual_prompts=None,
                predictor=None, device=None, **kw):
        import cv2
        STATE.calls.append({"source": source, "conf": conf, "refer_image": refer_image,
                            "visual_prompts": visual_prompts, "predictor": predictor, "device": device})
        objects, ids = STATE.scene
        has_not = visual_prompts is not None and 1 in set(np.asarray(visual_prompts["cls"]).tolist())
        xyxy, cf, cl, polys = [], [], [], []
        for k, ob in enumerate(objects):
            m = (ids == k).astype(np.uint8)
            if m.sum() < 30 or ob["conf"] < conf:
                continue
            if STATE.bleed_px:
                m = cv2.dilate(m, np.ones((2 * STATE.bleed_px + 1,) * 2, np.uint8))
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cs = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
            ys, xs = np.nonzero(m)
            xyxy.append([xs.min(), ys.min(), xs.max(), ys.max()])
            cf.append(ob["conf"])
            cl.append(1 if (has_not and ob["name"] in STATE.not_egg_names) else 0)
            polys.append(cs)
        return [_Result(_Boxes(np.array(xyxy, float).reshape(-1, 4), np.array(cf), np.array(cl)), _Masks(polys))]


def install():
    """Put the fake ultralytics into sys.modules (call before the first detect)."""
    u = types.ModuleType("ultralytics")
    u.YOLOE = YOLOE
    u.__version__ = "8.4.fake"
    models = types.ModuleType("ultralytics.models")
    yolo = types.ModuleType("ultralytics.models.yolo")
    yoloe = types.ModuleType("ultralytics.models.yolo.yoloe")
    yoloe.YOLOEVPSegPredictor = YOLOEVPSegPredictor
    u.models, models.yolo, yolo.yoloe = models, yolo, yoloe
    sys.modules.update({"ultralytics": u, "ultralytics.models": models, "ultralytics.models.yolo": yolo,
                        "ultralytics.models.yolo.yoloe": yoloe})


def uninstall():
    for k in [k for k in sys.modules if k == "ultralytics" or k.startswith("ultralytics.")]:
        del sys.modules[k]
