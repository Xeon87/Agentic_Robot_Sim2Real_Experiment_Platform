#!/usr/bin/env python3
"""
NERO V3.26 WRIST CAMERA (simulation) — a D435i mounted on the gripper, parallel to the V3.08-V3.11 fixed camera
==============================================================================================================

Pose sources for the skills (switch in the sim GUI, Agent tab):
  * camera  (default) — the fixed stand camera of V3.08-V3.11, unchanged (NERO_V308_Camera_Perception.py)
  * wrist   (V3.26)   — this module: a D435i on the gripper; the arm moves to "look poses" before each capture
  * truth             — MuJoCo ground truth (JSON)

What this module adds (nothing in the fixed-camera module is changed; its perception functions are reused):
* inject_wrist_camera_into_mjcf(): a <camera> inside body link7. V3.31.1: mounted ON the gripper base side face
  (V3.33: D435i and USB board both on the -y face, only the selected one is drawn; perpendicular to the finger
  travel along link7 z), front flush
  with the base front, on a bracket, tilted ~9 deg so the optical axis crosses the gripper axis 0.40 m ahead.
  Faces from gripper_base.stl (nominal values if not found). Visual models are massless and non-colliding.
* wrist_camera_process_main(): a separate render/perception process (same protocol as the V3.08 camera process:
  trigger seq/tag in, tracked scene JSON out). Extrinsics are taken PER CAPTURE from the rendered camera pose
  (= joint state x hand-eye mount), perception = V3.10 perceive_eggs on the wrist RGB-D, identities by its own
  PerceptionTracker (world frame, so eggs seen in different views keep their names).
* look_points() / merge_payloads(): which places to look at for a tool (observe_scene: basket + holder;
  plan_grasp: above the target egg; verify_object_in_slot: holder) and how the views of one survey are merged
  (each egg takes the latest view in which it was seen). The backend moves the arm (MoveIt, collision-checked)
  to TCP = look point + NERO_V326_LOOK_TCP_H_M (0.29 m) with the gripper pointing down, waits until settled,
  then triggers the capture. D435i min range 0.28 m -> it cannot see the last centimetres of an approach; the
  grasp is planned from the pose perceived at the look pose (same as with the fixed camera).
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

VERSION = "V3.47.1"
WRIST_CAMERA_NAME = "v326_wrist_d435i"
PARENT_BODY = "link7"


def _env_vec(name, default):
    raw = os.environ.get(name, default)
    vals = [float(v) for v in str(raw).replace(" ", "").split(",") if v != ""]
    if len(vals) != 3:
        raise ValueError(f"{name} needs 3 comma-separated numbers, got {raw!r}")
    return vals


def enabled():
    return os.environ.get("NERO_V326_WRIST", "1") != "0"


def look_tcp_height_m():
    return float(os.environ.get("NERO_V326_LOOK_TCP_H_M", "0.29"))


USB_CAMERA_NAME = "v330_wrist_usb130"
PROFILES = ("d435i", "usb130")
PROFILE_LABELS = {"d435i": "Intel RealSense D435i (RGB-D)",
                  "usb130": "InnoMaker U20CAM-1080P USB 2.0 UVC, 130 deg wide angle (RGB only, no depth)"}


# =========================================================================== V3.31.1 mount ON the gripper
# The gripper (backend _attach_official_gripper) sits on link7 with its axis along link7 +x at y = 0,
# z = GRIPPER_AXIS_Z; the fingers slide along link7 z.  The cameras are therefore mounted on the two SIDE faces of
# the gripper base (link7 +y: D435i, -y: USB board), perpendicular to the finger travel, front flush with the
# base front, on a small bracket.  The base faces come from gripper_base.stl when it can be found (same search as
# the backend), otherwise from nominal values.
GRIPPER_AXIS_Z = -0.0235
GRIPPER_BASE_POS = (0.0375, 0.0, -0.0235)
GRIPPER_BASE_QUAT = (0.499998, -0.5, 0.500002, -0.5)          # w x y z (backend gripper_base_visual)
NOMINAL_BASE = {"x_min": 0.032, "x_max": 0.080, "y_face_pos": 0.030, "y_face_neg": -0.030}
_BASE_CACHE = {}


def _gripper_stl_candidates():
    env = os.environ.get("NERO_V3311_GRIPPER_BASE_STL")
    if env:
        yield Path(env)
    dirs = []
    for key in ("NERO_MODEL_FILE",):
        if os.environ.get(key):
            dirs.append(Path(os.environ[key]).resolve().parent)
    if os.environ.get("NERO_ASSET_WS"):
        dirs.append(Path(os.environ["NERO_ASSET_WS"]))
    home = Path.home()
    out = []
    for d in dirs:
        out += [d / "assets", d / "agilex_nero" / "assets"]
    out += [home / "mujoco_ws" / "nero" / "assets", home / "agx_arm_sim" / "mujoco" / "agilex_arm" / "agilex_nero" / "assets",
            home / "agx_arm_ws" / "src" / "agx_arm_sim" / "mujoco" / "agilex_arm" / "agilex_nero" / "assets"]
    for d in out:
        yield d / "gripper_base.stl"
    for root in (home / "mujoco_ws", home / "agx_arm_ws", home / "agx_arm_sim"):   # same fallback as the backend
        try:
            if root.exists():
                yield from root.rglob("gripper_base.stl")
        except Exception:
            pass


def read_stl_vertices(path):
    """Vertices of a binary or ASCII STL (pure Python)."""
    import struct
    data = Path(path).read_bytes()
    verts = []
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            for i in range(n):
                o = 84 + 50 * i + 12
                v = struct.unpack("<9f", data[o:o + 36])
                verts += [v[0:3], v[3:6], v[6:9]]
            return verts
    for line in data.decode("utf-8", "ignore").splitlines():
        t = line.strip().split()
        if len(t) == 4 and t[0] == "vertex":
            verts.append(tuple(float(x) for x in t[1:]))
    return verts


def _quat_R(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def base_faces_from_vertices(verts_link7, z_half=0.045):
    """Gripper base extent in link7: x range, and the outermost +y / -y face within the camera's z band."""
    xs = [v[0] for v in verts_link7]
    x_min, x_max = min(xs), max(xs)
    band = [v for v in verts_link7 if abs(v[2] - GRIPPER_AXIS_Z) <= z_half]
    band = band or verts_link7
    return {"x_min": x_min, "x_max": x_max, "y_face_pos": max(v[1] for v in band),
            "y_face_neg": min(v[1] for v in band)}


def gripper_base_extent():
    """-> (extent dict in link7, source string). Cached per process."""
    if "ext" in _BASE_CACHE:
        return _BASE_CACHE["ext"]
    res = (dict(NOMINAL_BASE), "nominal (gripper_base.stl not found)")
    if os.environ.get("NERO_V3311_MOUNT_AUTO", "1") != "0":
        for path in _gripper_stl_candidates():
            try:
                if not path.is_file():
                    continue
                raw = read_stl_vertices(path)
                if not raw:
                    continue
                scale = float(os.environ.get("NERO_V3311_STL_SCALE", "1.0"))
                if max(abs(c) for v in raw[:3000] for c in v) > 5.0:     # mm files
                    scale = 0.001
                R = _quat_R(GRIPPER_BASE_QUAT)
                P = GRIPPER_BASE_POS
                vl = [[R[i][0] * v[0] * scale + R[i][1] * v[1] * scale + R[i][2] * v[2] * scale + P[i] for i in range(3)]
                      for v in raw]
                res = (base_faces_from_vertices(vl), f"gripper_base.stl ({path})")
                break
            except Exception as exc:                                    # unreadable -> try the next / nominal
                res = (dict(NOMINAL_BASE), f"nominal (stl error {str(exc)[:80]})")
    _BASE_CACHE["ext"] = res
    return res


def gripper_mount(profile):
    """Camera pose on the gripper base side face (link7 frame): mount_pos (optical centre), target, up, bracket."""
    ext, src = gripper_base_extent()
    zc = GRIPPER_AXIS_Z
    clear = float(os.environ.get("NERO_V3311_MOUNT_CLEARANCE_M", "0.004"))
    if profile == "usb130":
        side, half_t, depth = -1.0, 0.016, 0.0115          # 32 x 32 board (+ lens holder) facing forward
        face = abs(float(ext["y_face_neg"]))
    else:
        # V3.33: the D435i sits on the same (-y) side face as the USB board; only the selected one is drawn
        side = -1.0 if os.environ.get("NERO_V333_D435I_SIDE", "neg") != "pos" else 1.0
        half_t, depth = 0.0125, 0.026                      # D435i 90 x 25 x 25, long side along z (finger rail)
        face = abs(float(ext["y_face_neg"] if side < 0 else ext["y_face_pos"]))
    x_front = min(max(float(ext["x_max"]), 0.055), 0.140)
    tilt_margin = depth * math.sin(math.radians(12.0))     # rear corner swings towards the gripper when tilted
    y_c = side * (face + clear + half_t + tilt_margin)
    pos = [round(x_front, 4), round(y_c, 4), zc]
    target = [0.40, 0.0, zc]                                # optical axis crosses the gripper axis 0.40 m ahead
    up = [0.0, side, 0.0]                                   # image up = away from the gripper; image right = z
    # bracket: from just inside the base face out to (slightly into) the camera back -> visibly attached
    bracket = {"y0": side * max(face - 0.003, 0.0), "y1": side * (face + clear + 0.0075),
               "x0": x_front - depth - 0.010, "x1": x_front - 0.008, "z_half": 0.014}
    return {"mount_pos": pos, "mount_target": target, "mount_up": up, "bracket": bracket,
            "mount_source": src, "base_extent": ext}


def _hfov_to_fovy(hfov_deg, w, h):
    return math.degrees(2.0 * math.atan(math.tan(math.radians(hfov_deg) / 2.0) * float(h) / float(w)))


def _mount_cfg(profile, env_prefix):
    """V3.31.1: mount on the gripper side face; <prefix>_POS / _TARGET / _UP env values still override."""
    m = gripper_mount(profile)
    out = {"mount_pos": m["mount_pos"], "mount_target": m["mount_target"], "mount_up": m["mount_up"],
           "bracket": m["bracket"], "mount_source": m["mount_source"]}
    for k, key in (("mount_pos", "_POS"), ("mount_target", "_TARGET"), ("mount_up", "_UP")):
        if os.environ.get(env_prefix + key):
            out[k] = _env_vec(env_prefix + key, "0,0,0")
            out["mount_source"] = "env override"
            out["bracket"] = None
    return out


def wrist_config(profile="d435i"):
    """Wrist camera configuration (mount in the link7 frame: link7 +x = gripper approach, TCP at x = 0.155 m).
    d435i  : resolution / FOV / noise identical to the fixed camera (V3.08 camera_config), RGB-D.
    usb130 : V3.30 - InnoMaker U20CAM-1080P (1080p30 MJPG, M12 lens 130 deg, 32x32 mm board, RGB only). Simulated as
             the RECTIFIED pinhole image after fisheye undistortion (MuJoCo renders pinhole only): HFOV
             NERO_V330_USB_HFOV_DEG (100 deg, the usable crop of a 130 deg diagonal lens), 1280x720 by default
             (real 1920x1080 -> intrinsics scale). No depth -> monocular perception (ray / plane + models)."""
    profile = str(profile or "d435i")
    if profile == "usb130":
        w = int(os.environ.get("NERO_V330_USB_WIDTH", "1280"))
        h = int(os.environ.get("NERO_V330_USB_HEIGHT", "720"))
        hfov = float(os.environ.get("NERO_V330_USB_HFOV_DEG", "100"))
        return {"profile": "usb130", "name": USB_CAMERA_NAME, "parent_body": PARENT_BODY, "width": w, "height": h,
                "fovy_deg": _hfov_to_fovy(hfov, w, h), "hfov_deg": hfov, "depth": False,
                "min_range_m": 0.0, "max_range_m": float(os.environ.get("NERO_V330_USB_MAX_RANGE_M", "1.5")),
                "depth_noise_k": 0.0, "depth_frames": 1,
                **_mount_cfg("usb130", "NERO_V330_USB_MOUNT"),
                "pos": None, "target": None, "label": PROFILE_LABELS["usb130"]}
    try:
        import NERO_V308_Camera_Perception as P8
        base = P8.camera_config()
    except Exception:                                    # (tests without numpy) same defaults as V3.08
        base = {"width": int(os.environ.get("NERO_V308_WIDTH", "1280")), "height": int(os.environ.get("NERO_V308_HEIGHT", "720")),
                "fovy_deg": float(os.environ.get("NERO_V308_FOVY_DEG", "42.0")),
                "min_range_m": float(os.environ.get("NERO_V308_MIN_RANGE_M", "0.28")),
                "max_range_m": float(os.environ.get("NERO_V308_MAX_RANGE_M", "3.0")),
                "depth_noise_k": float(os.environ.get("NERO_V308_DEPTH_NOISE_K", "0.0")),
                "depth_frames": max(1, int(os.environ.get("NERO_V310_DEPTH_FRAMES", "5")))}
    cfg = dict(base)
    cfg.update({
        "profile": "d435i", "name": WRIST_CAMERA_NAME, "parent_body": PARENT_BODY, "depth": True,
        **_mount_cfg("d435i", "NERO_V326_MOUNT"),
        "pos": None, "target": None, "label": PROFILE_LABELS["d435i"],
    })
    return cfg


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 1e-12 else list(v)


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def mount_axes(cfg):
    """MuJoCo camera axes in the parent body frame: x = image right, y = image up, camera looks along -z."""
    p, t, up = cfg["mount_pos"], cfg["mount_target"], cfg["mount_up"]
    f = _unit([t[i] - p[i] for i in range(3)])
    x = _unit(_cross(f, up))
    y = _unit(_cross(x, f))
    return x, y, f


def camera_xml(cfg):
    x, y, _f = mount_axes(cfg)
    p = cfg["mount_pos"]
    return (f'<camera name="{WRIST_CAMERA_NAME}" mode="fixed" pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" '
            f'xyaxes="{x[0]:.6f} {x[1]:.6f} {x[2]:.6f} {y[0]:.6f} {y[1]:.6f} {y[2]:.6f}" fovy="{cfg["fovy_deg"]:.4f}"/>')


def _frame_attr(x, z):
    """MJCF xyaxes for a geom whose local x / z axes are x / z (y = z cross x)."""
    y = _cross(z, x)
    return f"{x[0]:.6f} {x[1]:.6f} {x[2]:.6f} {y[0]:.6f} {y[1]:.6f} {y[2]:.6f}"


def camera_model_geoms(cfg):
    """V3.30 visual camera model (massless, non-colliding geoms in link7), all BEHIND the optical centre so the
    camera never sees its own housing. Names start with v330_cam_<profile>_ (the simulator can hide them)."""
    x, _y, f = mount_axes(cfg)
    p = cfg["mount_pos"]
    pre = f"v330_cam_{cfg['profile']}_"
    at = lambda d: [p[i] - f[i] * d for i in range(3)]                    # noqa: E731  point d behind the optics
    common = {"contype": "0", "conaffinity": "0", "density": "0", "group": "0"}
    g = []
    br = cfg.get("bracket")
    if br:                                   # V3.31.1: bracket from the gripper base side face to the camera back
        cy = 0.5 * (br["y0"] + br["y1"]); hy = max(0.0008, 0.5 * abs(br["y1"] - br["y0"]))
        cx = 0.5 * (br["x0"] + br["x1"]); hx = max(0.002, 0.5 * (br["x1"] - br["x0"]))
        g.append(dict(common, name=pre + "bracket", type="box", size=f"{hx:.4f} {hy:.4f} {br['z_half']:.4f}",
                      pos=f"{cx:.4f} {cy:.4f} {GRIPPER_AXIS_Z:.4f}", rgba="0.55 0.56 0.58 1"))
    if cfg["profile"] == "usb130":
        # 32 x 32 x 1.6 mm PCB 9 mm behind the lens front, M12 lens barrel (dia 14 mm) up to the optical centre
        c = at(0.0095)
        g.append(dict(common, name=pre + "pcb", type="box", size="0.016 0.016 0.0008", pos="%.4f %.4f %.4f" % tuple(c),
                      xyaxes=_frame_attr(x, f), rgba="0.05 0.35 0.18 1"))
        c = at(0.0045)
        g.append(dict(common, name=pre + "lens_holder", type="box", size="0.0085 0.0085 0.0040",
                      pos="%.4f %.4f %.4f" % tuple(c), xyaxes=_frame_attr(x, f), rgba="0.08 0.08 0.08 1"))
        c = at(0.0012)
        g.append(dict(common, name=pre + "lens", type="cylinder", size="0.0070 0.0010", pos="%.4f %.4f %.4f" % tuple(c),
                      xyaxes=_frame_attr(x, f), rgba="0.02 0.02 0.03 1"))
    else:
        # D435i housing 90 x 25 x 25 mm (front glass at the optical centre), RGB lens + two IR lenses + projector
        c = at(0.0130)
        g.append(dict(common, name=pre + "housing", type="box", size="0.045 0.0125 0.0125",
                      pos="%.4f %.4f %.4f" % tuple(c), xyaxes=_frame_attr(x, f), rgba="0.16 0.16 0.18 1"))
        c = at(0.0006)
        g.append(dict(common, name=pre + "glass", type="box", size="0.043 0.0105 0.0004",
                      pos="%.4f %.4f %.4f" % tuple(c), xyaxes=_frame_attr(x, f), rgba="0.03 0.03 0.04 1"))
        for k, dx in enumerate((-0.0325, -0.0080, 0.0175, 0.0325)):          # IR-left, projector, RGB, IR-right
            cc = [at(0.0004)[i] + x[i] * dx for i in range(3)]
            g.append(dict(common, name=pre + f"lens_{k}", type="cylinder", size="0.0045 0.0003",
                          pos="%.4f %.4f %.4f" % tuple(cc), xyaxes=_frame_attr(x, f), rgba="0.30 0.32 0.40 1"))
    return g


def camera_model_geom_names(profile):
    return [g["name"] for g in camera_model_geoms(wrist_config(profile))]


def inject_wrist_camera_into_mjcf(xml_path, cfg=None, profiles=None):
    """Return a new MJCF path (same folder) with the wrist camera(s) inside body link7 and (V3.30, default on,
    NERO_V330_DRAW_WRIST_CAMERAS=0 to disable) their visual models. Physics unchanged (massless, no contacts)."""
    cfgs = [cfg] if cfg else [wrist_config(pf) for pf in (profiles or PROFILES)]
    src = Path(xml_path)
    text = src.read_text(encoding="utf-8")
    cfgs = [c for c in cfgs if f'name="{c["name"]}"' not in text]
    if not cfgs:
        return str(src)
    root = ET.fromstring(text)
    body = root.find(f".//body[@name='{PARENT_BODY}']")
    if body is None:
        raise RuntimeError(f"V3.26: <body name='{PARENT_BODY}'> not found in {xml_path}")
    draw = (os.environ.get("NERO_V330_DRAW_WRIST_CAMERAS", "1") != "0"
            or os.environ.get("NERO_V326_DRAW_WRIST_CAMERA", "0") == "1")
    for c in cfgs:
        x, y, f = mount_axes(c)
        p = c["mount_pos"]
        ET.SubElement(body, "camera", {
            "name": c["name"], "mode": "fixed", "pos": f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}",
            "xyaxes": f"{x[0]:.6f} {x[1]:.6f} {x[2]:.6f} {y[0]:.6f} {y[1]:.6f} {y[2]:.6f}",
            "fovy": f"{c['fovy_deg']:.4f}"})
        if draw:
            for g in camera_model_geoms(c):
                ET.SubElement(body, "geom", {k: str(v) for k, v in g.items()})
    out = src.with_name("_nero_motor_v326_wrist_tmp.xml")
    out.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    return str(out)


def optical_from_mujoco(cam_xpos, cam_xmat):
    """T_world_optical (x right, y down, z forward) from MuJoCo camera pose (xmat columns: x right, y up, z back)."""
    import numpy as np
    R = np.asarray(cam_xmat, dtype=float).reshape(3, 3)
    T = np.eye(4)
    T[:3, :3] = np.stack([R[:, 0], -R[:, 1], -R[:, 2]], axis=1)
    T[:3, 3] = np.asarray(cam_xpos, dtype=float)
    return T


# =========================================================================== survey planning (pure Python)
def look_points(tag, last_objects=None, basket_xy=(-0.425, -0.155), holder_xy=(-0.420, 0.155), z_ref=0.03):
    """-> [(name, [x, y, z_ref])] places to look at for this capture tag.
    observe_scene / unknown -> basket + holder; plan_grasp(:obj) / plan_grasp_recapture:obj -> above obj's last
    perceived position if it is near the basket, else the basket centre; verify* -> holder."""
    tag = str(tag or "")
    head, _, obj = tag.partition(":")
    last_objects = last_objects or {}
    basket = ("basket", [float(basket_xy[0]), float(basket_xy[1]), float(z_ref)])
    holder = ("holder", [float(holder_xy[0]), float(holder_xy[1]), float(z_ref)])
    if head.startswith("verify"):
        return [holder]
    if head.startswith("plan_grasp"):
        o = last_objects.get(obj) if obj else None
        if o and o.get("xyz_m"):
            x, y = float(o["xyz_m"][0]), float(o["xyz_m"][1])
            if math.hypot(x - basket_xy[0], y - basket_xy[1]) < 0.15:
                return [(f"above_{obj}", [x, y, float(z_ref)])]
        return [basket]
    return [basket, holder]


# =========================================================================== V3.27 LOOK-POSE FAMILY (pure Python)
# V3.26 asked cuRobo for ONE pose per view (TCP 0.29 m above the point, yaw 0) and cuRobo returned None 3x for
# basket and holder -> the arm never left q0 (camera facing the ceiling). A view does not need one exact pose:
# any TCP pose whose CAMERA looks at the point from a usable distance is equivalent. look_pose_candidates() returns
# that family (camera distance x tilt x roll about the optical axis); the backend screens it with BioIK
# (collision-aware) and plans a MoveIt joint goal for the first few solutions.
def _m_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _m_vec(A, v):
    return [sum(A[i][k] * v[k] for k in range(3)) for i in range(3)]


def _m_T(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]


def _rpy(r, p, y):
    return _m_mul(_rz(y), _m_mul(_ry(p), _rx(r)))


def quat_from_matrix(R):
    """xyzw quaternion of a 3x3 rotation matrix."""
    t = R[0][0] + R[1][1] + R[2][2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = [(R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s, 0.25 * s]
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0
        q = [0.25 * s, (R[0][1] + R[1][0]) / s, (R[0][2] + R[2][0]) / s, (R[2][1] - R[1][2]) / s]
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0
        q = [(R[0][1] + R[1][0]) / s, 0.25 * s, (R[1][2] + R[2][1]) / s, (R[0][2] - R[2][0]) / s]
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0
        q = [(R[0][2] + R[2][0]) / s, (R[1][2] + R[2][1]) / s, 0.25 * s, (R[1][0] - R[0][1]) / s]
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


def matrix_from_quat(q):
    x, y, z, w = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def tcp_in_link7():
    """(R, p) of tcp_link in link7/flange_link (URDF gpu_tcp_joint: xyz 0.1553 0 -0.0215, rpy -pi/2 0 -pi/2)."""
    xyz = _env_vec("NERO_V327_TCP_XYZ_IN_LINK7", "0.1553,0.0,-0.0215")
    rpy = _env_vec("NERO_V327_TCP_RPY_IN_LINK7", "-1.5708,0.0,-1.5708")
    return _rpy(*rpy), xyz


def camera_in_tcp(cfg=None):
    """(optical centre, forward unit vector) of the wrist camera in the TCP frame (TCP z = approach direction)."""
    cfg = cfg or wrist_config()
    R7t, p7t = tcp_in_link7()
    Rt7 = _m_T(R7t)
    p = cfg["mount_pos"]
    _x, _y, f = mount_axes(cfg)
    pc = _m_vec(Rt7, [p[i] - p7t[i] for i in range(3)])
    fc = _m_vec(Rt7, f)
    return pc, _unit(fc)


def _float_list(name, default):
    return [float(v) for v in str(os.environ.get(name, default)).replace(" ", "").split(",") if v != ""]


def look_pose_candidates(point, cfg=None, distances=None, tilts=None, rolls=None):
    """Ordered TCP poses whose wrist camera looks at `point` (world, m) along its optical axis.
    distances: camera -> point along the optical axis (m; D435 min range 0.28 m, eggs stick out ~3 cm);
    tilts: (deg from vertical, 'in'|'out') - 'in': the camera sits between the base and the point (shorter reach),
    'out': the camera sits beyond the point and looks back towards the base;
    rolls: rotation (deg) about the vertical before tilting = which side of the target the gripper body sits.
    Order: distance, then tilt, then roll (top-down first, every roll, before any tilted view)."""
    cfg = cfg or wrist_config()
    pc, fc = camera_in_tcp(cfg)
    distances = distances or _float_list("NERO_V327_LOOK_DIST_M", "0.36,0.40,0.33")
    if tilts is None:
        tilts = [(0.0, "out"), (15.0, "in"), (15.0, "out"), (30.0, "in"), (30.0, "out")]
    rolls = rolls or _float_list("NERO_V327_LOOK_ROLLS_DEG", "90,135,45,180,225,270,0,315")
    P = [float(v) for v in point]
    az0 = math.atan2(P[1], P[0])                           # direction base -> point
    down = _rx(math.pi)                                    # TCP z (approach) pointing down
    out, seen = [], set()
    for d in distances:
        for beta, side in tilts:
            for roll in rolls:
                az = az0 + (0.0 if side == "out" else math.pi)
                R = _m_mul(_rz(az), _m_mul(_ry(math.radians(beta)), _m_mul(_rz(math.radians(roll)), down)))
                f = _m_vec(R, fc)
                # tilt the whole TCP so that the OPTICAL axis (not the approach axis) has the requested tilt
                c = [P[i] - d * f[i] for i in range(3)]
                t = [c[i] - _m_vec(R, pc)[i] for i in range(3)]
                q = quat_from_matrix(R)
                key = (round(d, 3), tuple(round(v, 4) for v in q))
                if key in seen:
                    continue
                seen.add(key)
                out.append({"xyz": [round(v, 5) for v in t], "quat_xyzw": [round(v, 7) for v in q],
                            "cam_xyz": [round(v, 4) for v in c], "cam_fwd": [round(v, 4) for v in f],
                            "dist_m": d, "tilt_deg": beta if beta == 0 else (beta if side == "out" else -beta),
                            "roll_deg": roll,
                            "tag": f"d{d * 1000:.0f}/tilt{'' if beta == 0 else ('+' if side == 'out' else '-')}{beta:.0f}/roll{roll:.0f}"})
    return out


def look_pose_ok(c, min_tcp_z=None):
    """Cheap geometric filter: the TCP (finger tips) stays above min_tcp_z."""
    min_tcp_z = float(os.environ.get("NERO_V327_LOOK_MIN_TCP_Z_M", "0.20")) if min_tcp_z is None else float(min_tcp_z)
    return float(c["xyz"][2]) >= min_tcp_z


# =========================================================================== V3.28 VIEW-AWARE TRACKING
# V3.27 run: egg_1 was placed into slot_1; the holder view then gave that detection to egg_4 (the V3.11 tracker hands
# any unmatched detection to the nearest unmatched track within 0.6 m - written for a fixed camera that sees every
# egg every frame), and the next basket view gave egg_4's detection to egg_1 -> identities swapped.
# Now (wrist camera only, the fixed-camera tracker is unchanged):
#   * only tracks whose last position projects into the current image take part in the association; tracks out of
#     view keep name + pose untouched (seen=false, captures_since_seen grows);
#   * hints from the backend in the trigger tag: held=<egg> (not matched at all while in the gripper) and
#     placed=<egg>@x,y,z (the egg the arm released into a slot: its track is moved there before association).
HINT_SEP = "|"


def json_safe(o):
    """V3.38: numpy scalars / arrays / tuples / sets -> plain JSON types (V3.37 USB crash: a numpy bool in the carton
    height measurement made every capture fail with 'Object of type bool is not JSON serializable')."""
    try:
        import numpy as _np
    except Exception:                                    # pragma: no cover
        _np = None
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [json_safe(v) for v in o]
    if _np is not None:
        if isinstance(o, _np.bool_):
            return bool(o)
        if isinstance(o, _np.integer):
            return int(o)
        if isinstance(o, _np.floating):
            f = float(o)
            return f if math.isfinite(f) else None
        if isinstance(o, _np.ndarray):
            return json_safe(o.tolist())
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    return str(o)


def encode_hints(tag, held=None, placed=None, limit=255, cam=None, calib=None, heldp=None, live=False, lost=None):
    """'<tag>|cam=usb130|calib=0.30|live=1|held=egg_1|heldp=x,y,z|placed=egg_2@-0.510,0.123,0.054;...'
    (placed entries dropped if too long). V3.35: calib = calibration-error level, heldp = in-hand model position
    of the held egg (for the camera in-hand check), live = background refresh capture."""
    parts = [str(tag)]
    if cam and cam != "d435i":
        parts.append(f"cam={cam}")
    if calib is not None:
        parts.append(f"calib={float(calib):.3f}")
    if live:
        parts.append("live=1")
    if held:
        parts.append(f"held={held}")
        if heldp is not None:
            parts.append("heldp=" + ",".join(f"{float(v):.4f}" for v in heldp))
    if lost:
        parts.append("lost=" + ",".join(str(n) for n in lost))     # V3.37: forget these tracks (egg dropped)
    items = [f"{n}@{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}" for n, p in (placed or {}).items()]
    while items:
        s = HINT_SEP.join(parts + ["placed=" + ";".join(items)])
        if len(s.encode("utf-8")) <= limit:
            return s
        items.pop()
    return HINT_SEP.join(parts)


def decode_hints(raw):
    """-> (display tag, {'held': name|None, 'placed': {name: [x,y,z]}})"""
    bits = str(raw or "").split(HINT_SEP)
    hints = {"held": None, "placed": {}, "cam": "d435i", "heldp": None, "live": False, "lost": []}
    for b in bits[1:]:
        k, _, v = b.partition("=")
        if k == "cam" and v in PROFILES:
            hints["cam"] = v
        elif k == "calib":
            try:
                hints["calib"] = float(v)
            except ValueError:
                pass
        elif k == "live":
            hints["live"] = v == "1"
        elif k == "heldp":
            try:
                pp = [float(x) for x in v.split(",")]
                hints["heldp"] = pp if len(pp) == 3 else None
            except ValueError:
                pass
        elif k == "held" and v:
            hints["held"] = v
        elif k == "lost" and v:
            hints["lost"] = [n for n in v.split(",") if n]
        elif k == "placed":
            for it in v.split(";"):
                n, _, xyz = it.partition("@")
                try:
                    p = [float(x) for x in xyz.split(",")]
                except ValueError:
                    continue
                if n and len(p) == 3:
                    hints["placed"][n] = p
    return bits[0], hints


def in_view(xyz, K, T, margin_px=20.0, z_min=0.05, z_max=3.0):
    """Does world point xyz project into the image of camera (K, T_world_optical)? Pure Python."""
    R = [[float(T[i][j]) for j in range(3)] for i in range(3)]
    t = [float(T[i][3]) for i in range(3)]
    d = [float(xyz[i]) - t[i] for i in range(3)]
    pc = [sum(R[k][j] * d[k] for k in range(3)) for j in range(3)]       # R^T (p - t)
    if not (z_min < pc[2] < z_max):
        return False
    u = K["fx"] * pc[0] / pc[2] + K["ppx"]
    v = K["fy"] * pc[1] / pc[2] + K["ppy"]
    return -margin_px <= u <= K["width"] + margin_px and -margin_px <= v <= K["height"] + margin_px


class ViewTracker:
    """Wraps a V3.11 PerceptionTracker (unchanged): association restricted to the current view + backend hints."""

    def __init__(self, base, margin_px=None, names=None):
        self.base = base
        self.margin_px = float(os.environ.get("NERO_V328_VIEW_MARGIN_PX", "20")) if margin_px is None else float(margin_px)
        self.names = list(names or [])            # V3.30: name pool for eggs seen for the first time
        self.pending = []                         # V3.33: RGB-only candidates waiting for a second view
        self.confirm_views = max(1, int(os.environ.get("NERO_V333_CONFIRM_VIEWS", "2")))
        self.confirm_dist = float(os.environ.get("NERO_V333_CONFIRM_DIST_M", "0.025"))
        self._lost_since = {}                     # V3.41: name -> capture seq of the first "lost" hint

    def _confirmed(self, d, seq):
        """V3.33: an RGB-only detection (no depth to check it against) gets a name only after it was seen at the
        same world position (NERO_V333_CONFIRM_DIST_M) in NERO_V333_CONFIRM_VIEWS different captures - a colour
        blob that is not an egg rarely projects to the same table point from two camera poses."""
        if not (d.get("quality") or {}).get("rgb_only") or self.confirm_views <= 1:
            return True
        self.pending = [p for p in self.pending if seq - p["last"] <= 8]
        xy = d["xyz_m"][:2]
        for p in self.pending:
            if math.hypot(p["xy"][0] - xy[0], p["xy"][1] - xy[1]) <= self.confirm_dist:
                p["seqs"].add(seq)
                n = len(p["seqs"])
                p["xy"] = [(p["xy"][0] * (n - 1) + xy[0]) / n, (p["xy"][1] * (n - 1) + xy[1]) / n]
                p["last"] = seq
                if n >= self.confirm_views:
                    self.pending.remove(p)
                    return True
                return False
        self.pending.append({"xy": list(xy), "seqs": {seq}, "last": seq})
        return False

    def _predicted_assoc(self, out, dets, predict, hidden, held):
        """V3.39 CLOSE-VIEW ASSOCIATION: the V3.11 tracker gives an UNTRUSTED detection (cut by the gripper mask,
        partly occluded, coverage < 0.5) a name only within 15 mm of the track - and the track holds the old
        single-view plane position, 20-60 mm off with the USB camera -> 'seen []' in the close plan_grasp view, a
        recapture and two q0 round trips. Now an unclaimed, not border-clipped detection takes the name whose
        MEASURED position (triangulated / table plane) is nearest, if within NERO_V339_PREDICT_GATE_M and clearly
        nearer to it than to any other egg (ratio NERO_V339_PREDICT_RATIO)."""
        gate = float(os.environ.get("NERO_V339_PREDICT_GATE_M", "0.045"))
        ratio = float(os.environ.get("NERO_V339_PREDICT_RATIO", "1.4"))
        tracks = self.base.tracks
        used = {o.get("det_id") for o in out.values() if o.get("seen")}
        cand = [d for d in dets if d["det_id"] not in used and not (d.get("quality") or {}).get("border")
                and 0.30 <= float((d.get("quality") or {}).get("coverage") or 0.0) <= 1.6]
        preds = {}
        for n, t in tracks.items():
            if n in hidden or n == held:
                continue
            p = (predict or {}).get(n) or t.get("xyz_m")
            if p:
                preds[n] = [float(v) for v in p]
        done = []
        for name in sorted(preds):
            if (out.get(name) or {}).get("seen") or not cand:
                continue
            p = preds[name]
            dl = sorted((math.hypot(d["xyz_m"][0] - p[0], d["xyz_m"][1] - p[1]), d["det_id"]) for d in cand)
            dist, did = dl[0]
            if dist > gate:
                continue
            d = next(x for x in cand if x["det_id"] == did)
            other = min([math.hypot(d["xyz_m"][0] - q[0], d["xyz_m"][1] - q[1]) for m, q in preds.items() if m != name]
                        or [1e9])
            if other < ratio * dist:
                continue
            tracks[name].update({"xyz_m": list(d["xyz_m"]), "quat_xyzw": list(d["quat_xyzw"]),
                                 "axis_world": list(d["axis_world"]), "location": d["location"],
                                 "confidence": d.get("confidence"), "trusted": True,
                                 "coverage": (d.get("quality") or {}).get("coverage"), "det_id": did, "last_seq": None})
            out[name] = {"xyz_m": list(d["xyz_m"]), "quat_xyzw": list(d["quat_xyzw"]), "axis_world": list(d["axis_world"]),
                         "location": d["location"], "seen": True, "trusted": True, "confidence": d.get("confidence"),
                         "coverage": (d.get("quality") or {}).get("coverage"), "det_id": did,
                         "captures_since_seen": 0, "association": f"predicted ({1000 * dist:.0f} mm from the measured position)"}
            cand = [x for x in cand if x["det_id"] != did]
            done.append(name)
        return done

    def update(self, dets, seq, K, T, seed_match=None, hints=None, z_max=3.0, predict=None):
        hints = hints or {}
        tracks = self.base.tracks
        applied = []
        forgot = []
        lost341 = list(hints.get("lost") or [])
        self._lost_since = {n: s for n, s in self._lost_since.items() if n in lost341}
        for name in lost341:
            # V3.37: the backend says this egg was dropped - its old track is wrong; forget it so the next trusted
            # detection nobody claims gets this (now free) name again.
            # V3.41: forget only the track from BEFORE the drop. Captures already queued with the "lost" hint used to
            # forget the egg again right after it was re-found (V3.40 log: egg_1 found, forgotten, searched again).
            since = self._lost_since.setdefault(name, seq)
            t_ = tracks.get(name)
            if t_ is not None and name != hints.get("held"):
                fs = t_.get("first_seq")
                if fs is None or int(fs) < int(since) or os.environ.get("NERO_V341_LOST_ONCE", "1") == "0":
                    tracks.pop(name, None)
                    forgot.append(name)
        for name, p in (hints.get("placed") or {}).items():
            if name in tracks:
                tracks[name]["xyz_m"] = [float(v) for v in p]
                tracks[name]["location"] = "placed_hint"
                applied.append(name)
        held = hints.get("held")
        hidden = {}
        for name, t in list(tracks.items()):
            if name == held or not in_view(t["xyz_m"], K, T, self.margin_px, z_max=z_max):
                hidden[name] = tracks.pop(name)
        seed = {n: d for n, d in (seed_match or {}).items() if n not in hidden and n != held}
        try:
            out, diag = self.base.update(dets, seq, seed_match=seed)
            pred339 = []
            if predict is not None and os.environ.get("NERO_V339_PREDICT_ASSOC", "1") != "0":
                # V3.41: RGB-only (USB) wrist camera only - the D435i keeps the V3.38 association
                pred339 = self._predicted_assoc(out, dets, predict, hidden, held)
                for n_ in pred339:
                    self.base.tracks[n_]["last_seq"] = seq
                if pred339:
                    ids339 = {out[n_]["det_id"] for n_ in pred339}
                    diag = dict(diag, unassigned_detections=[i for i in diag.get("unassigned_detections", [])
                                                             if i not in ids339])
            named = []
            if self.names:
                # V3.30 GENERIC NAMING (no simulator identity seeding): a trusted detection no track claimed gets the
                # first free name; detections ordered by position (x, then y) so the naming is deterministic.
                byid = {d["det_id"]: d for d in dets}
                free = [n for n in self.names if n not in tracks and n not in hidden]
                todo = sorted((byid[i] for i in diag.get("unassigned_detections", []) if byid[i].get("trusted")),
                              key=lambda d: (round(float(d["xyz_m"][0]), 3), round(float(d["xyz_m"][1]), 3)))
                n_cand = len(todo)
                todo = [d for d in todo if self._confirmed(d, seq)]          # V3.33 multi-view confirmation
                for d, name in zip(todo, free):
                    tracks[name] = {"first_seq": seq, "xyz_m": list(d["xyz_m"]), "quat_xyzw": list(d["quat_xyzw"]),
                                    "axis_world": list(d["axis_world"]), "location": d["location"],
                                    "confidence": d.get("confidence"), "trusted": True,
                                    "coverage": (d.get("quality") or {}).get("coverage"), "det_id": d["det_id"],
                                    "last_seq": seq}
                    out[name] = {"xyz_m": list(d["xyz_m"]), "quat_xyzw": list(d["quat_xyzw"]),
                                 "axis_world": list(d["axis_world"]), "location": d["location"], "seen": True,
                                 "trusted": True, "confidence": d.get("confidence"),
                                 "coverage": (d.get("quality") or {}).get("coverage"), "det_id": d["det_id"],
                                 "captures_since_seen": 0}
                    named.append(name)
                diag = dict(diag, named=named, unnamed_trusted=max(0, len(todo) - len(named)),
                            awaiting_second_view=n_cand - len(todo),
                            unassigned_detections=[i for i in diag.get("unassigned_detections", [])
                                                   if i not in {out[n]["det_id"] for n in named}])
        finally:
            tracks.update(hidden)
        for name, t in hidden.items():
            out[name] = {"xyz_m": t["xyz_m"], "quat_xyzw": t["quat_xyzw"], "axis_world": t.get("axis_world"),
                         "location": t.get("location"), "seen": False, "trusted": False, "confidence": 0.0,
                         "coverage": None, "det_id": None, "captures_since_seen": int(seq - t.get("last_seq", seq)),
                         "out_of_view": name != held, "held": name == held}
        out = dict(sorted(out.items()))
        diag = dict(diag, out_of_view=sorted(n for n in hidden if n != held), held_hint=held, placed_hint=applied,
                    lost_forgotten=forgot, predicted_assoc=pred339 if "pred339" in locals() else [])
        return out, diag


# =========================================================================== V3.29 GENERIC SCAN + CONTAINER PERCEPTION
# No preset coordinates: the first survey of an episode sweeps a ring of table points around the base (directions
# only, NERO_V329_SCAN_AZ_DEG / _R_M), the egg holder is found in the depth image (its 12 mm base plate + rims
# stand out from the table plane; eggs and robot removed), its pose (centre, yaw) is fitted to the plate footprint,
# and every slot centre is refined from the six posts around it. What is known in advance is only the holder's
# MODEL (plate size, slot grid, post ring) - like a CAD file - never where it is.
# The scene's "basket" has no physical body in this MuJoCo model (V2-50); the source region is the perceived egg
# pile (backend: centroid of the eggs outside the holder).
def scan_points(z=0.02):
    """Points for the first (blind) survey: [(name, [x, y, z])].
    V3.31: a grid over the WORKSPACE (table area of the cell, P8.WORKSPACE / NERO_V331_SCAN_X_M, _Y_M) clipped to
    the arm's comfortable reach annulus (NERO_V331_SCAN_R_MIN/MAX_M), spacing ~ one camera footprint
    (NERO_V331_SCAN_SPACING_M), visited row by row (nearest row first, serpentine). Nothing about where objects are.
    NERO_V329_SCAN_AZ_DEG (if set) restores the V3.29 ring."""
    if os.environ.get("NERO_V329_SCAN_AZ_DEG"):
        r = float(os.environ.get("NERO_V329_SCAN_R_M", "0.42"))
        return [(f"scan_az{a:.0f}", [round(r * math.cos(math.radians(a)), 4), round(r * math.sin(math.radians(a)), 4),
                                     float(z)]) for a in _float_list("NERO_V329_SCAN_AZ_DEG", "")]
    try:
        import NERO_V308_Camera_Perception as P8
        wsx, wsy = P8.WORKSPACE["x"], P8.WORKSPACE["y"]
    except Exception:
        wsx, wsy = (-0.80, -0.08), (-0.42, 0.42)
    xr = _float_list("NERO_V331_SCAN_X_M", f"{wsx[0]},{wsx[1]}")
    yr = _float_list("NERO_V331_SCAN_Y_M", f"{wsy[0]},{wsy[1]}")
    sp = float(os.environ.get("NERO_V331_SCAN_SPACING_M", "0.26"))
    rmin = float(os.environ.get("NERO_V331_SCAN_R_MIN_M", "0.24"))   # V3.33: 0.20 -> 0.24 (r 0.23 m views were never reachable)
    rmax = float(os.environ.get("NERO_V331_SCAN_R_MAX_M", "0.62"))

    def axis(lo, hi):
        n = max(1, int(math.ceil((hi - lo) / sp - 1e-9)))
        step = (hi - lo) / n
        return [lo + step * (k + 0.5) for k in range(n)]
    xs = sorted(axis(min(xr), max(xr)), key=lambda v: abs(v))          # nearest row (to the base) first
    ys = sorted(axis(min(yr), max(yr)))
    out = []
    for i, x in enumerate(xs):
        row = ys if i % 2 == 0 else list(reversed(ys))
        for y in row:
            if rmin <= math.hypot(x, y) <= rmax:
                out.append((f"scan_{len(out) + 1}", [round(x, 4), round(y, 4), float(z)]))
    return out


def scan_distances():
    return _float_list("NERO_V329_SCAN_DIST_M", "0.45,0.40,0.36")


def holder_model():
    """Egg-holder model in its own frame (u = long axis, v = short axis), metres. Matches the MJCF carton."""
    return {
        "outer": (_float_list("NERO_V329_HOLDER_OUTER_M", "0.264,0.160")),        # rim outer extent (u, v)
        "plate_top": float(os.environ.get("NERO_V329_HOLDER_PLATE_TOP_M", "0.012")),
        "slot_u": _float_list("NERO_V329_HOLDER_SLOT_U_M", "-0.090,-0.030,0.030,0.090"),
        "slot_v": _float_list("NERO_V329_HOLDER_SLOT_V_M", "-0.032,0.032"),
        "post_r": (0.014, 0.034),                                                   # post ring (radial band)
        "slot_z_below_post_top": 0.005,
        "post_top": float(os.environ.get("NERO_V331_HOLDER_POST_TOP_M", "0.048")),   # capsule posts 11..44 mm + r 4 mm
        "ring_inner_r": float(os.environ.get("NERO_V331_HOLDER_RING_INNER_M", "0.0215")),  # gap between the posts
        "well_top": float(os.environ.get("NERO_V331_HOLDER_WELL_TOP_M", "0.0145")),
        "well_visible_z": float(os.environ.get("NERO_V330_WELL_TOP_M", "0.013")),
        # V3.33 supermarket moulded-pulp carton (VISUAL shape, the physics geometry is the hidden boxes/posts):
        # cells shaped round the egg, flat saddle plateau at the post-top height, three frusto-pyramidal posts
        # between the two rows, vertical outer walls. Rendered by the scene builder from carton_mesh_obj().
        "plateau_z": 0.048,
        "cell_rim_r": 0.0229,
        "cell_bottom_z": 0.0215,
        "carton_posts_u": (-0.060, 0.0, 0.060),
        "carton_post_top_z": 0.058,
        "carton_post_half": (0.014, 0.005),                                          # base / top half width
        "boxes": [("egg_holder_base", (0.0, 0.0, 0.006), (0.250, 0.144, 0.012)),
                  ("egg_holder_rim_xneg", (-0.128, 0.0, 0.025), (0.008, 0.152, 0.050)),
                  ("egg_holder_rim_xpos", (0.128, 0.0, 0.025), (0.008, 0.152, 0.050)),
                  ("egg_holder_rim_yneg", (0.0, -0.076, 0.025), (0.248, 0.008, 0.050)),
                  ("egg_holder_rim_ypos", (0.0, 0.076, 0.025), (0.248, 0.008, 0.050))],
    }


# ---- V3.33 carton CAD (shared by the scene builder, the synthetic tests and nothing else) ----------------------
_CARTON_EGG = (0.0315, 0.0219, 0.054)          # design egg (half long, half short, centre height) the cells hug


def carton_height(u, v):
    """Height of the carton top surface above the table at holder-frame points (u, v) (numpy arrays, metres).
    Outside the footprint -> nan."""
    import numpy as np
    m = holder_model()
    u = np.asarray(u, float); v = np.asarray(v, float)
    hu, hv = m["outer"][0] / 2.0, m["outer"][1] / 2.0
    a, b, zc = _CARTON_EGG
    z = np.full(np.broadcast(u, v).shape, m["plateau_z"])
    for su in m["slot_u"]:
        for sv in m["slot_v"]:
            r = np.hypot(u - su, v - sv)
            rho = np.clip((r - 0.001) / b, 0.0, 1.0)
            bowl = zc - a * np.sqrt(1.0 - rho ** 2) - 0.001
            bowl = np.clip(bowl, m["cell_bottom_z"], m["plateau_z"])
            z = np.where(r < m["cell_rim_r"], np.minimum(z, bowl), z)
    hb, ht = m["carton_post_half"]
    for pu in m["carton_posts_u"]:
        d = (np.abs(u - pu) + np.abs(v)) / math.sqrt(2.0)                  # square post turned 45 deg
        post = m["carton_post_top_z"] - (m["carton_post_top_z"] - m["plateau_z"]) * np.clip((d - ht) / (hb - ht), 0, 1)
        z = np.where(d < hb, np.maximum(z, post), z)
    return np.where((np.abs(u) <= hu + 1e-9) & (np.abs(v) <= hv + 1e-9), z, np.nan)


def carton_mesh_obj(step=0.002):
    """Closed (watertight) triangle mesh of the carton in its own frame as OBJ text: top height field, four
    vertical outer walls, flat bottom. Used only as a non-colliding visual geom."""
    import numpy as np
    m = holder_model()
    hu, hv = m["outer"][0] / 2.0, m["outer"][1] / 2.0
    us = np.linspace(-hu, hu, int(round(2 * hu / step)) + 1)
    vs = np.linspace(-hv, hv, int(round(2 * hv / step)) + 1)
    U, V = np.meshgrid(us, vs, indexing="ij")
    Z = carton_height(U, V)
    nu, nv = U.shape
    lines = ["# NERO V3.33 moulded-pulp egg carton (visual)"]
    for i in range(nu):
        for j in range(nv):
            lines.append(f"v {U[i, j]:.5f} {V[i, j]:.5f} {Z[i, j]:.5f}")
    for i in range(nu):
        for j in range(nv):
            lines.append(f"v {U[i, j]:.5f} {V[i, j]:.5f} 0.00000")
    top = lambda i, j: i * nv + j + 1                                       # noqa: E731
    bot = lambda i, j: nu * nv + i * nv + j + 1                             # noqa: E731
    for i in range(nu - 1):
        for j in range(nv - 1):
            a_, b_, c_, d_ = top(i, j), top(i + 1, j), top(i + 1, j + 1), top(i, j + 1)
            lines.append(f"f {a_} {b_} {c_}"); lines.append(f"f {a_} {c_} {d_}")         # up-facing
            a_, b_, c_, d_ = bot(i, j), bot(i + 1, j), bot(i + 1, j + 1), bot(i, j + 1)
            lines.append(f"f {a_} {c_} {b_}"); lines.append(f"f {a_} {d_} {c_}")         # down-facing
    ring = ([(i, 0) for i in range(nu)] + [(nu - 1, j) for j in range(1, nv)]
            + [(i, nv - 1) for i in range(nu - 2, -1, -1)] + [(0, j) for j in range(nv - 2, 0, -1)])
    for k in range(len(ring)):
        p, q = ring[k], ring[(k + 1) % len(ring)]
        # ring runs counter-clockwise seen from above -> outward walls: (bot p, bot q, top q), (bot p, top q, top p)
        lines.append(f"f {bot(*p)} {bot(*q)} {top(*q)}"); lines.append(f"f {bot(*p)} {top(*q)} {top(*p)}")
    return "\n".join(lines) + "\n"


def egg_center_in_slot_z(table_z=0.0, a=None, b=None):
    """V3.31: height of an egg's centre standing in a slot, from the holder MODEL and the egg CLASS size (no
    simulator values): an egg wider than the post gap rests on the post tops at the height where its section equals
    the gap; a narrower one drops onto the well."""
    if a is None or b is None:
        try:
            import NERO_V308_Camera_Perception as P8
            a, b = float(P8.EGG_PRIOR["half_long_m"]), float(P8.EGG_PRIOR["half_short_m"])
        except Exception:
            a, b = 0.0315, 0.0219
    m = holder_model()
    rho = m["ring_inner_r"]
    if b > rho:
        return float(table_z) + m["post_top"] + a * math.sqrt(max(0.0, 1.0 - (rho / b) ** 2))
    return float(table_z) + m["well_top"] + a


def holder_slot_name(iu, iv, n_u):
    return f"slot_{iv * n_u + iu + 1}"


def holder_boxes(holder):
    """MoveIt boxes [(id, centre_xyz, size_xyz, yaw_rad)] of the model placed at the perceived holder pose."""
    m = holder_model()
    c, yaw, zt = holder["center_xy_m"], float(holder["yaw_rad"]), float(holder.get("table_z_m", 0.0))
    cu, su = math.cos(yaw), math.sin(yaw)
    out = []
    for name, (du, dv, dz), size in m["boxes"]:
        out.append((name, [c[0] + du * cu - dv * su, c[1] + du * su + dv * cu, zt + dz], list(size), yaw))
    return out


def _components(cells):
    """8-connected components of a set of (i, j) grid cells."""
    cells = set(cells)
    comps = []
    while cells:
        seed = cells.pop()
        comp, stack = [seed], [seed]
        while stack:
            i, j = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (i + di, j + dj)
                    if nb in cells:
                        cells.remove(nb)
                        comp.append(nb)
                        stack.append(nb)
        comps.append(comp)
    return comps


def detect_holder(depth, K, T, dets=None, robot_boxes=None, stride=2):
    """Find the egg holder in one aligned depth image (world frame from T). -> dict or None.
    dict: center_xy_m, yaw_rad, extent_m (u, v), table_z_m, top_z_m, partial (touches the image border or is
    smaller than the model), slots {slot_k: {xyz_m, post_sectors, refined}}, complete, n_points."""
    import numpy as np
    import NERO_V308_Camera_Perception as P8
    m = holder_model()
    pw, vv, uu = P8.backproject(depth, K, T, stride=stride)
    if len(pw) < 100:
        return None
    ws = P8.WORKSPACE
    keep = ((pw[:, 0] > ws["x"][0]) & (pw[:, 0] < ws["x"][1]) & (pw[:, 1] > ws["y"][0]) & (pw[:, 1] < ws["y"][1])
            & (pw[:, 2] > -0.05) & (pw[:, 2] < 0.20))
    if robot_boxes:
        keep &= ~P8.points_inside_boxes(pw, robot_boxes, 0.010)
    pw, vv, uu = pw[keep], vv[keep], uu[keep]
    if len(pw) < 100:
        return None
    hist, edges = np.histogram(pw[:, 2], bins=np.arange(-0.05, 0.101, 0.002))
    table_z = float(edges[int(np.argmax(hist))] + 0.001)                       # dominant plane = table
    el = (pw[:, 2] > table_z + 0.007) & (pw[:, 2] < table_z + 0.075)
    a = float(P8.EGG_PRIOR["half_long_m"]) + 0.005
    b = float(P8.EGG_PRIOR["half_short_m"]) + 0.005
    for d in (dets or []):
        c = np.asarray(d["xyz_m"], float)
        ax = np.asarray(d.get("axis_world") or [1.0, 0.0, 0.0], float)
        ax = ax / max(1e-9, float(np.linalg.norm(ax)))
        rel = pw - c
        al = rel @ ax
        rad = np.linalg.norm(rel - np.outer(al, ax), axis=1)
        el &= ~((al / a) ** 2 + (rad / b) ** 2 <= 1.0)
    P, pv, pu = pw[el], vv[el], uu[el]
    if len(P) < 80:
        return None
    cell = 0.005
    ij = np.floor(P[:, :2] / cell).astype(int)
    cell_pts = {}
    for k, (i, j) in enumerate(map(tuple, ij)):
        cell_pts.setdefault((i, j), []).append(k)
    comps = _components(cell_pts.keys())
    best = None
    for comp in comps:
        idx = np.asarray([k for c_ in comp for k in cell_pts[c_]])
        if len(comp) * cell * cell < 0.25 * m["outer"][0] * m["outer"][1]:
            continue
        Q = P[idx, :2]
        mu = Q.mean(axis=0)
        w, V = np.linalg.eigh(np.cov((Q - mu).T))
        yaw0 = math.atan2(V[1, 1], V[0, 1])                                      # largest eigenvalue -> long axis
        Qs = (Q - mu)[:: max(1, len(Q) // 4000)]

        def _area(yy):
            u_ = np.array([math.cos(yy), math.sin(yy)]); v_ = np.array([-u_[1], u_[0]])
            a_, b_ = Qs @ u_, Qs @ v_
            return (np.percentile(a_, 99.5) - np.percentile(a_, 0.5)) * (np.percentile(b_, 99.5) - np.percentile(b_, 0.5))
        # minimum-area rectangle: coarse 1 deg over 90 deg (a truncated view can fool PCA), then 0.1 deg
        coarse = min((float(_area(yaw0 + math.radians(d))), yaw0 + math.radians(d)) for d in np.arange(-45.0, 45.0, 1.0))[1]
        yaw = min((float(_area(coarse + math.radians(d))), coarse + math.radians(d)) for d in np.arange(-1.0, 1.01, 0.1))[1]
        # long axis = the longer side of that rectangle
        u_t = np.array([math.cos(yaw), math.sin(yaw)])
        if np.ptp(Qs @ u_t) < np.ptp(Qs @ np.array([-u_t[1], u_t[0]])):
            yaw += math.pi / 2.0
        u = np.array([math.cos(yaw), math.sin(yaw)])
        if u[0] < -1e-6 or (abs(u[0]) <= 1e-6 and u[1] < 0):                     # canonical: long axis towards +x
            yaw += math.pi
            u = -u
        yaw = math.atan2(math.sin(yaw), math.cos(yaw))
        v = np.array([-u[1], u[0]])
        pu_, pv_ = (Q - mu) @ u, (Q - mu) @ v
        lo_u, hi_u, lo_v, hi_v = (np.percentile(pu_, 0.5), np.percentile(pu_, 99.5),
                                  np.percentile(pv_, 0.5), np.percentile(pv_, 99.5))
        ext = (float(hi_u - lo_u), float(hi_v - lo_v))
        if not (0.45 * m["outer"][0] <= ext[0] <= 1.15 * m["outer"][0] and 0.45 * m["outer"][1] <= ext[1] <= 1.20 * m["outer"][1]):
            continue
        oversize = ext[0] > m["outer"][0] + 0.010 or ext[1] > m["outer"][1] + 0.010
        center = mu + u * (lo_u + hi_u) / 2.0 + v * (lo_v + hi_v) / 2.0
        border = 4 * stride
        touches = bool(np.any((pv[idx] <= border) | (pu[idx] <= border) | (pv[idx] >= K["height"] - 1 - border)
                              | (pu[idx] >= K["width"] - 1 - border)))
        small = ext[0] < m["outer"][0] - 0.008 or ext[1] < m["outer"][1] - 0.008
        # cut by the image border only matters when the footprint is also shorter than the model
        cand = {"center_xy_m": [round(float(center[0]), 5), round(float(center[1]), 5)], "yaw_rad": round(float(yaw), 5),
                "extent_m": [round(ext[0], 4), round(ext[1], 4)], "table_z_m": round(table_z, 4),
                "top_z_m": round(float(np.percentile(P[idx, 2], 98)), 4), "partial": bool(small or oversize), "touches_border": touches,
                "n_points": int(len(idx)), "idx": idx}
        if best is None or cand["n_points"] > best["n_points"]:
            best = cand
    if best is None:
        return None
    idx = best.pop("idx")
    Ph = P[idx]
    c = np.asarray(best["center_xy_m"])
    yaw = best["yaw_rad"]
    u = np.array([math.cos(yaw), math.sin(yaw)]); v = np.array([-u[1], u[0]])
    zt = best["table_z_m"]
    # V3.33 moulded-pulp carton: every cell is surrounded by the flat saddle plateau (visible even with an egg in
    # the cell, the egg is narrower than the cell rim). Slot centres come from the fitted footprint + the CAD grid
    # (the depth points are true 3D, the footprint fit has no perspective bias); the plateau ring around each cell
    # tells whether that cell was really observed.
    r_in, r_out = m["cell_rim_r"] + 0.0015, m["cell_rim_r"] + 0.0075
    plat = Ph[(Ph[:, 2] > zt + m["plateau_z"] - 0.008) & (Ph[:, 2] < zt + m["plateau_z"] + 0.008)]
    slots, n_ok = {}, 0
    nu = len(m["slot_u"])
    for iv, sv in enumerate(m["slot_v"]):
        for iu, su in enumerate(m["slot_u"]):
            pred = c + u * su + v * sv
            r = np.linalg.norm(plat[:, :2] - pred, axis=1)
            ring = plat[(r >= r_in) & (r <= r_out)]
            sectors = 0
            zs = zt + m["plateau_z"] - m["slot_z_below_post_top"]
            if len(ring):
                ang = np.arctan2(ring[:, 1] - pred[1], ring[:, 0] - pred[0])
                bins = np.floor(((ang + math.pi / 6.0) % (2 * math.pi)) / (math.pi / 3.0)).astype(int)
                sectors = int(np.sum(np.bincount(bins, minlength=6) >= 3))
                if len(ring) >= 10:
                    zs = float(np.median(ring[:, 2])) - m["slot_z_below_post_top"]
            if sectors >= 3:
                n_ok += 1
            slots[holder_slot_name(iu, iv, nu)] = {"xyz_m": [round(float(pred[0]), 5), round(float(pred[1]), 5), round(zs, 4)],
                                                   "post_sectors": sectors, "refined": sectors >= 6}
    best["slots"] = slots
    best["slots_with_posts"] = n_ok
    best["complete"] = bool(not best["partial"] and n_ok >= len(slots) - 2)
    best["method"] = "depth_carton_footprint"
    return best


# =========================================================================== V3.30 RGB-ONLY WRIST CAMERA (usb130)
# No depth: every 3D quantity comes from the calibrated camera pose (joint state x mount) and a ray hitting a
# known plane - the table plane (robot mounting calibration, NERO_V330_TABLE_Z_M) lifted by a MODEL height:
#   * egg resting on the table : centre at table + egg short semi-axis (class prior)
#   * egg standing in a slot   : centre at table + NERO_V330_SLOT_EGG_Z_M (holder model)
#   * holder                   : the 8 dark slot wells (model: 4 x 2 grid, top at table + 13 mm) are found in the
#                                image, cast onto that plane and fitted with the grid model (2D rigid fit).
def table_z_m():
    return float(os.environ.get("NERO_V330_TABLE_Z_M", "0.0"))


def pixel_rays(K, T, u, v):
    """World ray origins / unit directions for pixel arrays (rectified pinhole)."""
    import numpy as np
    T = np.asarray(T, float)
    d_c = np.stack([(np.asarray(u, float) - K["ppx"]) / K["fx"], (np.asarray(v, float) - K["ppy"]) / K["fy"],
                    np.ones(np.shape(u))], axis=-1)
    d_w = d_c @ T[:3, :3].T
    d_w /= np.linalg.norm(d_w, axis=-1, keepdims=True)
    return T[:3, 3], d_w


def rays_to_plane(K, T, u, v, z_plane):
    """Pixels -> world points on the horizontal plane z = z_plane (nan where the ray does not hit it ahead)."""
    import numpy as np
    o, d = pixel_rays(K, T, u, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (float(z_plane) - o[2]) / d[..., 2]
    P = o + d * t[..., None]
    P[~(t > 0)] = np.nan
    return P


def _convex_hull(pts):
    pts = sorted(set(map(tuple, pts)))
    if len(pts) <= 2:
        return pts

    def cr(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cr(hi[-2], hi[-1], p) <= 0:
            hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def robot_pixel_mask(K, T, boxes, shape, margin_m=0.008, stride=2):
    """Model-based self mask: robot geom boxes (URDF/MJCF model at the current joint state) projected into the
    image (convex hull of the 8 inflated corners per box). -> bool mask at the given stride."""
    import numpy as np
    h, w = shape
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    mask = np.zeros(vv.shape, dtype=bool)
    T = np.asarray(T, float)
    R, t = T[:3, :3], T[:3, 3]
    for c, Rb, half in (boxes or []):
        hh = np.asarray(half, float) + margin_m
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * hh
        Pw = corners @ np.asarray(Rb).T + np.asarray(c)
        pc = (Pw - t) @ R
        zn = 0.02
        if np.all(pc[:, 2] <= zn):
            continue
        if np.any(pc[:, 2] < zn):
            # clip the box against the near plane: keep corners in front + edge / plane intersections
            keep = [p_ for p_ in pc if p_[2] >= zn]
            for i in range(8):
                for j in range(i + 1, 8):
                    if bin(i ^ j).count("1") != 1:
                        continue                                   # not an edge of the box
                    a_, b_ = pc[i], pc[j]
                    if (a_[2] - zn) * (b_[2] - zn) < 0:
                        tt = (zn - a_[2]) / (b_[2] - a_[2])
                        keep.append(a_ + (b_ - a_) * tt)
            pc = np.asarray(keep)
        u = K["fx"] * pc[:, 0] / pc[:, 2] + K["ppx"]
        v = K["fy"] * pc[:, 1] / pc[:, 2] + K["ppy"]
        hull = _convex_hull(np.stack([u, v], 1).tolist())
        if len(hull) < 3:
            continue
        hu = np.array(hull)
        inside = np.ones(vv.shape, dtype=bool)
        for i in range(len(hu)):
            a, b = hu[i], hu[(i + 1) % len(hu)]
            inside &= ((b[0] - a[0]) * (vv - a[1]) - (b[1] - a[1]) * (uu - a[0])) >= 0
        mask |= inside
    return mask


def _pixel_components(mask, min_px):
    """8-connected components of a boolean mask -> list of (rows, cols) index arrays."""
    import numpy as np
    ys, xs = np.nonzero(mask)
    comps = _components(zip(ys.tolist(), xs.tolist()))
    out = []
    for c in comps:
        if len(c) >= min_px:
            a = np.asarray(c)
            out.append((a[:, 0], a[:, 1]))
    return out


def carton_mask(rgb, ref=None):
    """V3.33 moulded-pulp carton pixels: low saturation, not bluish (table shadows lit by the cool fill light and
    the grey floor are bluish, the pulp is warm-to-neutral grey), clearly darker than the bright table."""
    import numpy as np
    c = rgb.astype(np.float32) / 255.0
    s = c.sum(axis=2) + 1e-6
    mx, mn = c.max(axis=2), c.min(axis=2)
    chroma = (c[..., 0] - c[..., 2]) / s
    sat = (mx - mn) / (mx + 1e-6)
    if ref is None:
        low = sat < 0.12
        ref = float(np.percentile(mx[low], 97)) if int(low.sum()) > 100 else 1.0
    dark_rel = float(os.environ.get("NERO_V333_CARTON_DARK_REL", "0.80"))
    return ((sat < float(os.environ.get("NERO_V333_CARTON_MAX_SAT", "0.21"))) & (chroma > -0.03) & (chroma < 0.12)
            & (mx < dark_rel * ref) & (mx > 0.12))


def _min_area_yaw(Q):
    """Yaw of the minimum-area rectangle of 2D points (coarse 1 deg / fine 0.1 deg; robust percentiles)."""
    import numpy as np
    mu = Q.mean(axis=0)
    Qs = (Q - mu)[:: max(1, len(Q) // 4000)]
    w, V = np.linalg.eigh(np.cov(Qs.T))
    yaw0 = math.atan2(V[1, 1], V[0, 1])

    def _area(yy):
        u_ = np.array([math.cos(yy), math.sin(yy)]); v_ = np.array([-u_[1], u_[0]])
        a_, b_ = Qs @ u_, Qs @ v_
        return (np.percentile(a_, 99.5) - np.percentile(a_, 0.5)) * (np.percentile(b_, 99.5) - np.percentile(b_, 0.5))
    coarse = min((float(_area(yaw0 + math.radians(d))), yaw0 + math.radians(d)) for d in np.arange(-45.0, 45.0, 1.0))[1]
    yaw = min((float(_area(coarse + math.radians(d))), coarse + math.radians(d)) for d in np.arange(-1.0, 1.01, 0.1))[1]
    u_t = np.array([math.cos(yaw), math.sin(yaw)])
    if np.ptp(Qs @ u_t) < np.ptp(Qs @ np.array([-u_t[1], u_t[0]])):
        yaw += math.pi / 2.0
    u = np.array([math.cos(yaw), math.sin(yaw)])
    if u[0] < -1e-6 or (abs(u[0]) <= 1e-6 and u[1] < 0):                         # canonical: long axis towards +x
        yaw += math.pi
    return math.atan2(math.sin(yaw), math.cos(yaw))


def detect_holder_rgb(rgb, K, T, robot_mask=None, egg_xy_in_slots=None, stride=2, table_z=None, z_top=None):
    """V3.33 egg carton from ONE RGB image (no depth): carton-coloured region -> rays onto the plane of the carton
    top (table + CAD plateau height) -> minimum-area rectangle -> pose from the FAR edges.
    Perspective: the near outer walls are seen from outside and their pixels land OUTSIDE the top rectangle when
    cast onto the top plane; the far walls are hidden behind the top. So the two edges facing away from the camera
    are exact, the near ones are pushed outwards - the centre is the far edge minus half the CAD size.
    complete = both far edges inside the image, footprint at least the CAD size, all 8 cells inside the image.
    Same output format as detect_holder."""
    import numpy as np
    m = holder_model()
    if z_top is not None:                       # V3.35: carton top height MEASURED by multi-view parallax
        z_top = float(z_top)
        zt = z_top - m["plateau_z"]
    else:
        zt = table_z_m() if table_z is None else float(table_z)
        z_top = zt + m["plateau_z"]
    L = np.asarray(m["outer"], float)
    mask = carton_mask(rgb)[::stride, ::stride]
    if robot_mask is not None:
        mask &= ~robot_mask
    hh, ww = mask.shape
    try:
        import NERO_V308_Camera_Perception as P8
        ws = P8.WORKSPACE
    except Exception:
        ws = {"x": (-0.80, -0.08), "y": (-0.42, 0.42)}
    best = None
    for rr, cc in _pixel_components(mask, 40):
        P = rays_to_plane(K, T, cc * stride, rr * stride, z_top)
        ok = np.isfinite(P[:, 0])
        ok &= (P[:, 0] > ws["x"][0] - 0.05) & (P[:, 0] < ws["x"][1] + 0.05) & (P[:, 1] > ws["y"][0] - 0.05) & (P[:, 1] < ws["y"][1] + 0.05)
        if ok.sum() < 40:
            continue
        rr_, cc_, Q = rr[ok], cc[ok], P[ok, :2]
        cells = {(int(math.floor(x / 0.004)), int(math.floor(y / 0.004))) for x, y in Q}
        area = len(cells) * 1.6e-5
        if area < 0.20 * L[0] * L[1] or area > 3.0 * L[0] * L[1]:
            continue
        if best is None or area > best[0]:
            best = (area, rr_, cc_, Q)
    if best is None:
        return None
    area, rr, cc, Q = best
    cam = np.asarray(T, float)[:2, 3]
    on_border = (rr <= 2) | (cc <= 2) | (rr >= hh - 3) | (cc >= ww - 3)
    g = _carton_geom(Q, cam, L, on_border)
    if g is None:
        return None
    center, yaw, ext, c0, far_ok, u, v = g["center"], g["yaw"], g["ext"], g["c0"], g["far_ok"], g["u"], g["v"]
    big_enough = ext[0] >= L[0] - 0.012 and ext[1] >= L[1] - 0.012
    not_huge = ext[0] <= L[0] + 0.09 and ext[1] <= L[1] + 0.09
    Tm = np.asarray(T, float)
    slots, inside_img = {}, 0
    z_slot = zt + m["post_top"] - m["slot_z_below_post_top"]
    su, sv = m["slot_u"], m["slot_v"]
    for iv, gv in enumerate(sv):
        for iu, gu in enumerate(su):
            xy = center + u * gu + v * gv
            pc = (np.array([xy[0], xy[1], z_slot]) - Tm[:3, 3]) @ Tm[:3, :3]
            seen = False
            if pc[2] > 0:
                uu_ = K["fx"] * pc[0] / pc[2] + K["ppx"]; vv_ = K["fy"] * pc[1] / pc[2] + K["ppy"]
                seen = bool(0 <= uu_ < K["width"] and 0 <= vv_ < K["height"])
            inside_img += int(seen)
            slots[holder_slot_name(iu, iv, len(su))] = {"xyz_m": [round(float(xy[0]), 5), round(float(xy[1]), 5),
                                                                   round(z_slot, 4)],
                                                        "post_sectors": 6 if seen else 0, "refined": seen}
    complete = bool(all(far_ok) and big_enough and not_huge and inside_img == len(slots))
    return {"center_xy_m": [round(float(center[0]), 5), round(float(center[1]), 5)], "yaw_rad": round(float(yaw), 5),
            "extent_m": [round(ext[0], 4), round(ext[1], 4)], "table_z_m": round(zt, 4), "top_z_m": round(z_top, 4),
            "partial": not complete, "touches_border": not all(far_ok), "n_points": int(len(Q)),
            "slots": slots, "slots_with_posts": inside_img, "complete": complete, "method": "rgb_carton_far_edges",
            "far_edges_in_view": far_ok, "area_m2": round(float(area), 4),
            # where to look next when this view is partial: the middle of what was actually seen
            "look_xy_m": [round(float(c0[0]), 5), round(float(c0[1]), 5)],
            # V3.35: pixels of the carton region (full-resolution u, v) for the multi-view height measurement;
            # removed by the camera process before the payload is published
            "_px_uv": (cc * stride, rr * stride, on_border)}


def fuse_partial_holders(parts, K=None, T=None):
    """V3.39: carton from PARTIAL views. In one view the carton centre along an axis is exact as soon as the FAR edge
    of that axis is inside the image (centre = far edge -/+ half the model size, see detect_holder_rgb), even if the
    near edge is cut by the image border. Two partial views that together have a valid far edge on BOTH axes, the
    same yaw (NERO_V339_FUSE_MAX_YAW_DEG) and agreeing centres where both measure the same axis
    (NERO_V339_FUSE_MAX_DIFF_M) give the full carton pose - V3.38 needed one view with the whole carton (log: two full
    observes before the carton was found). -> holder dict (complete) or None"""
    import numpy as np
    parts = [p for p in parts if p and p.get("far_edges_in_view") and p.get("center_xy_m") is not None]
    if len(parts) < 2:
        return None
    ymax = math.radians(float(os.environ.get("NERO_V339_FUSE_MAX_YAW_DEG", "4")))
    dmax = float(os.environ.get("NERO_V339_FUSE_MAX_DIFF_M", "0.008"))
    ref = parts[-1]
    y0 = float(ref["yaw_rad"])
    grp = []
    for p in parts:
        dy = math.atan2(math.sin(2.0 * (float(p["yaw_rad"]) - y0)), math.cos(2.0 * (float(p["yaw_rad"]) - y0))) / 2.0
        if abs(dy) <= ymax:
            grp.append((p, y0 + dy))
    if len(grp) < 2:
        return None
    yaw = math.atan2(sum(math.sin(y) for _p, y in grp), sum(math.cos(y) for _p, y in grp))
    u = np.array([math.cos(yaw), math.sin(yaw)]); v = np.array([-u[1], u[0]])
    au = [float(np.dot(np.asarray(p["center_xy_m"], float), u)) for p, _y in grp if p["far_edges_in_view"][0]]
    av = [float(np.dot(np.asarray(p["center_xy_m"], float), v)) for p, _y in grp if p["far_edges_in_view"][1]]
    if not au or not av:
        return None
    if (len(au) > 1 and max(au) - min(au) > dmax) or (len(av) > 1 and max(av) - min(av) > dmax):
        return None
    srcs = {id(p) for p, _y in grp if p["far_edges_in_view"][0]} | {id(p) for p, _y in grp if p["far_edges_in_view"][1]}
    if len(srcs) < 2:
        return None                                    # one view alone is handled by detect_holder_rgb
    center = u * float(np.median(au)) + v * float(np.median(av))
    m = holder_model()
    zt = float(ref.get("table_z_m", table_z_m()))
    z_slot = zt + m["post_top"] - m["slot_z_below_post_top"]
    slots = {}
    for iv, gv in enumerate(m["slot_v"]):
        for iu, gu in enumerate(m["slot_u"]):
            xy = center + u * gu + v * gv
            slots[holder_slot_name(iu, iv, len(m["slot_u"]))] = {
                "xyz_m": [round(float(xy[0]), 5), round(float(xy[1]), 5), round(z_slot, 4)],
                "post_sectors": 0, "refined": False}
    L = m["outer"]
    return {"center_xy_m": [round(float(center[0]), 5), round(float(center[1]), 5)], "yaw_rad": round(float(yaw), 5),
            "extent_m": [round(float(L[0]), 4), round(float(L[1]), 4)], "table_z_m": round(zt, 4),
            "top_z_m": ref.get("top_z_m"), "partial": False, "touches_border": False,
            "n_points": int(sum(int(p.get("n_points", 0)) for p, _y in grp)), "slots": slots,
            "slots_with_posts": len(slots), "complete": True, "method": "rgb_carton_far_edges_fused_partial_views",
            "fused_partial_views": len(grp), "far_edges_in_view": [True, True], "area_m2": ref.get("area_m2"),
            "look_xy_m": [round(float(center[0]), 5), round(float(center[1]), 5)],
            "fusion_spread_mm": [round(1000 * (max(au) - min(au)), 1), round(1000 * (max(av) - min(av)), 1)]}


def _carton_geom(Q, cam_xy, L, on_border, yaw=None):
    """Carton top rectangle from points cast onto a plane: min-area yaw (or the given one), extents, centre from the
    far edges (see detect_holder_rgb). -> dict or None."""
    import numpy as np
    if len(Q) < 40:
        return None
    yaw = _min_area_yaw(Q) if yaw is None else float(yaw)
    u = np.array([math.cos(yaw), math.sin(yaw)]); v = np.array([-u[1], u[0]])
    mu = Q.mean(axis=0)
    pu, pv = (Q - mu) @ u, (Q - mu) @ v
    lo_u, hi_u, lo_v, hi_v = (np.percentile(pu, 0.7), np.percentile(pu, 99.3), np.percentile(pv, 0.7), np.percentile(pv, 99.3))
    ext = (float(hi_u - lo_u), float(hi_v - lo_v))
    if ext[0] < 0.5 * L[0] or ext[1] < 0.5 * L[1]:
        return None
    c0 = mu + u * (lo_u + hi_u) / 2.0 + v * (lo_v + hi_v) / 2.0
    far_ok, cu_v = [], []
    for ax, lo, hi, p_, Lk in ((u, lo_u, hi_u, pu, L[0]), (v, lo_v, hi_v, pv, L[1])):
        away = float(np.dot(ax, c0 - np.asarray(cam_xy, float))) >= 0.0        # +axis side faces away from the camera
        edge = hi if away else lo
        cu_v.append(edge - Lk / 2.0 if away else edge + Lk / 2.0)
        near_edge = np.abs(p_ - edge) < 0.012
        far_ok.append(not bool(np.any(on_border & near_edge)))
    center = mu + u * cu_v[0] + v * cu_v[1]
    return {"center": center, "yaw": yaw, "ext": ext, "c0": c0, "far_ok": far_ok, "u": u, "v": v}


def _kmeans(X, k, iters=20):
    import numpy as np
    idx = [0]
    for _ in range(1, k):
        d = np.min(np.linalg.norm(X[:, None, :] - X[idx][None, :, :], axis=2), axis=1)
        idx.append(int(np.argmax(d)))
    Cc = X[idx].astype(float)
    for _ in range(iters):
        lab = np.argmin(np.linalg.norm(X[:, None, :] - Cc[None, :, :], axis=2), axis=1)
        Cn = np.array([X[lab == j].mean(axis=0) if np.any(lab == j) else Cc[j] for j in range(k)])
        if np.allclose(Cn, Cc):
            break
        Cc = Cn
    return lab


def _quat_from_axis_world(ax):
    """Egg body quaternion whose local z (long axis) points along ax."""
    import numpy as np
    z = np.array([0.0, 0.0, 1.0]); a = np.asarray(ax, float); a = a / np.linalg.norm(a)
    c = float(np.dot(z, a))
    if c > 1 - 1e-9:
        return [0.0, 0.0, 0.0, 1.0]
    if c < -1 + 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    w = np.cross(z, a); s = math.sqrt((1 + c) * 2.0)
    return [float(w[0] / s), float(w[1] / s), float(w[2] / s), float(s / 2.0)]


def egg_mask_v336(rgb):
    """V3.36 egg pixels: hue / saturation of the brown shell, but NO high brightness floor. The V3.08 mask needs
    value >= 0.22, which drops the egg's own shadowed lower half under the MuJoCo key light (the whole lower half
    was missing -> centroid and triangulated centre too high, width too small). Shading scales R, G, B together, so
    hue and saturation stay; the table / table shadows / carton / floor are low-saturation and stay out."""
    import numpy as np
    c = rgb.astype(np.float32) / 255.0
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx = c.max(axis=2)
    mn = c.min(axis=2)
    d = mx - mn + 1e-6
    sat = d / (mx + 1e-6)
    hue = np.where(mx == r, ((g - b) / d) % 6.0, np.where(mx == g, (b - r) / d + 2.0, (r - g) / d + 4.0)) * 60.0
    vmin = float(os.environ.get("NERO_V336_EGG_MIN_VALUE", "0.06"))
    return (hue >= 10.0) & (hue <= 50.0) & (sat >= 0.22) & (sat <= 0.95) & (mx >= vmin)


def _fill_rows_cols(rr, cc):
    """Fill a blob's holes (specular highlight, egg label) for a roughly convex object: row span AND column span."""
    import numpy as np
    r0, c0 = int(rr.min()), int(cc.min())
    H, Wd = int(rr.max()) - r0 + 1, int(cc.max()) - c0 + 1
    m = np.zeros((H, Wd), bool)
    m[rr - r0, cc - c0] = True
    rowf = np.zeros_like(m)
    for i in range(H):
        js = np.nonzero(m[i])[0]
        if len(js):
            rowf[i, js.min():js.max() + 1] = True
    colf = np.zeros_like(m)
    for j in range(Wd):
        is_ = np.nonzero(m[:, j])[0]
        if len(is_):
            colf[is_.min():is_.max() + 1, j] = True
    f = rowf & colf
    ys, xs = np.nonzero(f)
    return ys + r0, xs + c0


def _solidity(rr, cc):
    """Blob area / convex hull area (1 = convex egg; merged or partly hidden eggs are clearly lower)."""
    import numpy as np
    pts = np.stack([cc, rr], 1).astype(float)
    if len(pts) < 8:
        return 1.0
    sub = pts[:: max(1, len(pts) // 4000)]
    hull = _convex_hull(sub.tolist())
    if len(hull) < 3:
        return 1.0
    h = np.asarray(hull, float)
    area_h = 0.5 * abs(float(np.dot(h[:, 0], np.roll(h[:, 1], 1)) - np.dot(h[:, 1], np.roll(h[:, 0], 1))))
    return float(min(1.0, len(pts) / max(area_h + 0.5 * len(hull), 1.0)))


def perceive_eggs_rgb(rgb, K, T, robot_mask=None, holder=None, stride=2, table_z=None):
    """Eggs from ONE RGB image (V3.36): shell-colour mask incl. the shadowed lower half -> blobs (holes filled) ->
    split merged eggs (footprint + solidity) -> ELLIPSE (image moments) per egg: centre = image of the egg centre,
    axes = size, plane-projected footprint = yaw. The plane position (table + class half width) is only a guess for
    associating views; the reported position comes from multi-view triangulation (EggMeasure).
    -> dets in the V3.10 format + px_uv / px_cov / plane_yaw / elong / size_ok / solidity / occluded."""
    import numpy as np
    import NERO_V308_Camera_Perception as P8
    zt = table_z_m() if table_z is None else float(table_z)
    a = float(P8.EGG_PRIOR["half_long_m"]); b = float(P8.EGG_PRIOR["half_short_m"])
    z_rest = zt + b
    z_slot = (zt + float(os.environ["NERO_V330_SLOT_EGG_Z_M"])) if os.environ.get("NERO_V330_SLOT_EGG_Z_M") \
        else egg_center_in_slot_z(zt, a, b)                        # V3.31: from the holder model + egg class
    if os.environ.get("NERO_V336_EGG_MASK", "1") == "0":
        mask = P8.egg_color_mask(rgb)[::stride, ::stride]
    else:
        mask = egg_mask_v336(rgb)[::stride, ::stride]
    if robot_mask is not None:
        mask &= ~robot_mask
    hh, ww = mask.shape
    dets = []
    foot = math.pi * a * b
    min_sol = float(os.environ.get("NERO_V336_SIZE_MIN_SOLIDITY", "0.88"))
    for rr0, cc0 in _pixel_components(mask, 25):
        # how many eggs: from the UNFILLED pixels (filling would also fill the gaps between touching eggs)
        P0 = rays_to_plane(K, T, cc0 * stride, rr0 * stride, z_rest)
        ok0 = np.isfinite(P0[:, 0])
        cells0 = {(int(math.floor(x / 0.002)), int(math.floor(y / 0.002))) for x, y in P0[ok0, :2]}
        n0 = max(1, int(round(len(cells0) * 4e-6 / foot)))
        if n0 == 1:
            rr, cc = _fill_rows_cols(rr0, cc0)       # one egg: fill highlight / label holes
        else:
            rr, cc = rr0, cc0
        border = bool(rr.min() <= 1 or cc.min() <= 1 or rr.max() >= hh - 2 or cc.max() >= ww - 2)
        occluded = False
        if robot_mask is not None:
            for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
                r2 = np.clip(rr + dr, 0, hh - 1); c2 = np.clip(cc + dc, 0, ww - 1)
                if bool(robot_mask[r2, c2].any()):
                    occluded = True
                    break
        sol = _solidity(rr, cc)
        Pg = rays_to_plane(K, T, cc * stride, rr * stride, z_rest)
        ok = np.isfinite(Pg[:, 0])
        if ok.sum() < 20:
            continue
        rr, cc, Pg = rr[ok], cc[ok], Pg[ok]
        cells = {(int(math.floor(x / 0.002)), int(math.floor(y / 0.002))) for x, y in Pg[:, :2]}
        area = len(cells) * 4e-6
        n = n0
        if n == 1 and sol < float(os.environ.get("NERO_V336_SPLIT_SOLIDITY", "0.86")) and area > 1.15 * foot:
            n = 2                                     # two touching eggs seen as one blob
        if area < 0.3 * foot or n > int(os.environ.get("NERO_V330_MAX_EGGS_PER_BLOB", "4")):
            continue                                  # specks / large brown regions (e.g. the wooden floor)
        ws = P8.WORKSPACE
        cx0, cy0 = float(Pg[:, 0].mean()), float(Pg[:, 1].mean())
        if not (ws["x"][0] < cx0 < ws["x"][1] and ws["y"][0] < cy0 < ws["y"][1]):
            continue
        labels = _kmeans(Pg[:, :2], n) if n > 1 else np.zeros(len(Pg), dtype=int)
        for j in range(n):
            sel = labels == j
            if sel.sum() < 15:
                continue
            us_, vs_ = cc[sel].astype(float) * stride, rr[sel].astype(float) * stride
            uc, vc = float(us_.mean()), float(vs_.mean())             # ellipse (moment) centre of the filled blob
            ctr = rays_to_plane(K, T, np.array([uc]), np.array([vc]), z_rest)[0]
            in_slot = False
            if holder and holder.get("complete") and holder.get("slots"):
                ctr_s = rays_to_plane(K, T, np.array([uc]), np.array([vc]), z_slot)[0]
                sl = min(holder["slots"].values(), key=lambda s_: math.hypot(s_["xyz_m"][0] - ctr_s[0], s_["xyz_m"][1] - ctr_s[1]))
                if math.hypot(sl["xyz_m"][0] - ctr_s[0], sl["xyz_m"][1] - ctr_s[1]) < 0.030:
                    in_slot, ctr = True, ctr_s
            if not np.all(np.isfinite(ctr)):
                continue
            Q = Pg[sel, :2]
            cov = np.cov((Q - Q.mean(axis=0)).T) if len(Q) > 3 else np.eye(2)
            wv, V = np.linalg.eigh(cov)
            yaw = math.atan2(V[1, 1], V[0, 1])
            elong = float(math.sqrt(max(wv[1], 1e-12) / max(wv[0], 1e-12)))
            axis = [0.0, 0.0, 1.0] if in_slot else [math.cos(yaw), math.sin(yaw), 0.0]
            cells_j = {(int(math.floor(x / 0.002)), int(math.floor(y / 0.002))) for x, y in Q}
            cov_ratio = len(cells_j) * 4e-6 / foot              # projected footprint / class footprint (segmentation)
            trusted = (not border) and 0.5 <= cov_ratio <= float(os.environ.get("NERO_V336_MAX_COVERAGE", "2.2"))
            pcov = np.cov(np.stack([us_, vs_])) if len(us_) > 3 else np.eye(2)
            dets.append({"px_uv": [round(uc, 2), round(vc, 2)],
                         "px_cov": [[round(float(pcov[0, 0]), 2), round(float(pcov[0, 1]), 2)],
                                    [round(float(pcov[1, 0]), 2), round(float(pcov[1, 1]), 2)]],
                         "px_bbox": [int(us_.min()), int(vs_.min()), int(us_.max()) + stride, int(vs_.max()) + stride],
                         "plane_yaw": round(yaw, 4), "elong": round(elong, 3),
                         "size_ok": bool(n == 1 and not border and not occluded and sol >= min_sol),
                         "solidity": round(sol, 3), "occluded": occluded,
                         "det_id": len(dets), "xyz_m": [round(float(ctr[0]), 5), round(float(ctr[1]), 5),
                                                        round(float(ctr[2]), 5)],
                         "quat_xyzw": [round(v, 6) for v in _quat_from_axis_world(axis)], "axis_world": axis,
                         "location": "egg_holder_slot" if in_slot else "table_or_pile",
                         "trusted": bool(trusted), "confidence": round(float(cov_ratio), 3),
                         "fit_rms_mm": None, "n_points": int(sel.sum()),
                         "quality": {"coverage": round(float(cov_ratio), 3), "border": border, "rgb_only": True,
                                     "blob_split": n, "solidity": round(sol, 3), "occluded": occluded}})
    return dets


def holder_quality(h):
    """Ranking of holder detections: complete first, then posts found, then points."""
    if not h:
        return (-1, -1, -1)
    return (1 if h.get("complete") else 0, int(h.get("slots_with_posts", 0)), int(h.get("n_points", 0)))


def merge_payloads(payloads, views=None):
    """Merge the payloads of one survey: each object takes the latest payload in which it was SEEN, else its
    latest (unseen) entry. Keeps capture/tag of the last payload; adds views + wait_s."""
    payloads = [p for p in payloads if p]
    if not payloads:
        return None
    merged = {}
    for k, p in enumerate(payloads):
        for name, o in (p.get("objects") or {}).items():
            o = dict(o, view=(views[k] if views and k < len(views) else k))
            if o.get("seen") or name not in merged:          # latest sighting wins; an unseen entry never
                merged[name] = o                              # replaces a sighting from this survey
    last = dict(payloads[-1])
    last["objects"] = merged
    last["views"] = list(views or range(len(payloads)))
    last["captures"] = [p.get("capture") for p in payloads]
    last["wait_s"] = round(sum(float(p.get("wait_s") or 0.0) for p in payloads), 2)
    last["source"] = "wrist_d435i"
    return last


def capture_sync(shared, tag, timeout_s=None):
    """Request one capture from a camera process (V3.11 protocol) and wait for its payload (raises on failure)."""
    timeout_s = float(os.environ.get("NERO_V311_CAPTURE_TIMEOUT_S", "20")) if timeout_s is None else float(timeout_s)
    t_wait = time.time()
    while not (shared.get("status") is not None and int(shared["status"].value) == 1):
        if time.time() - t_wait > max(timeout_s, 60.0):
            raise RuntimeError("PERCEPTION_UNAVAILABLE: wrist camera process is not running")
        time.sleep(0.1)
    seq, arr = shared["trig_seq"], shared["trig_tag"]
    b = str(tag).encode("utf-8")[:255]
    with seq.get_lock():
        arr[: len(b)] = b
        arr[len(b)] = b"\0"
        seq.value += 1
        mine = int(seq.value)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if int(shared["res_seq"].value) >= mine:
            buf = shared["res_buf"]
            with buf.get_lock():
                raw = bytes(buf[:]).split(b"\0", 1)[0]
            payload = json.loads(raw.decode("utf-8"))
            if payload.get("error"):
                raise RuntimeError(f"PERCEPTION_ERROR: {payload['error']}")
            payload["wait_s"] = round(time.time() - t0, 2)
            return payload
        time.sleep(0.03)
    raise RuntimeError(f"PERCEPTION_TIMEOUT: no wrist capture result within {timeout_s:.0f}s")


# =========================================================================== V3.35 MEASUREMENT + CALIBRATION ERROR
# (1) Camera calibration error: a cheap USB camera is never calibrated exactly. Perception uses a BELIEVED camera
#     model (intrinsics fx/fy/ppx/ppy, residual lens distortion, hand-eye extrinsics) that differs from the real one
#     (= the rendered camera) by a fixed per-session error direction scaled with the GUI level (0 = perfect).
# (2) Measurement instead of priors (RGB-only): every egg position is TRIANGULATED from the rays of its image
#     centroid in >= 2 views taken from different camera poses (known from joint states x hand-eye); its size is
#     measured from the blob extent at the triangulated depth; the table height from the measured eggs; the carton
#     top height by multi-view parallax. The egg class prior is no longer the source of any number - plausibility
#     (is this a hen's egg?) is judged by the Agent from the measured values.
# (3) In-hand check: while an egg is held the wrist camera (rigid with the gripper) checks that the egg stays where
#     it was right after the grasp (slip / rotation in the fingers shows up as an image shift).
# (4) Live view for the GUI monitor: a downscaled image + the labels of what the perception recognised.
V335_CALIB_SCALE = {          # error at level 1.0 (one sigma-ish of a quick checkerboard + hand-eye calibration)
    "focal_rel": 0.020, "aspect_rel": 0.003, "pp_px_at_1280": 8.0, "k1": 0.030,
    "trans_m": 0.003, "rot_deg": 1.0}
V335_CALIB_MAX = 3.0


def calib_level_default():
    return float(os.environ.get("NERO_V335_CALIB_LEVEL", "0.3"))


def calib_error(level, seed=None, width=1280):
    """Per-session calibration error at the given level: fixed direction (seed), magnitude x level."""
    import random
    lvl = max(0.0, min(float(os.environ.get("NERO_V335_CALIB_MAX", str(V335_CALIB_MAX))), float(level or 0.0)))
    rnd = random.Random(int(os.environ.get("NERO_V335_CALIB_SEED", "335")) if seed is None else int(seed))
    u = []
    for _ in range(11):                        # each component 0.5..1 x its scale, random sign (fixed per seed)
        r_ = rnd.uniform(-1.0, 1.0)
        u.append(math.copysign(0.5 + 0.5 * abs(r_), r_))
    S = V335_CALIB_SCALE
    pp = S["pp_px_at_1280"] * float(width) / 1280.0
    return {"level": round(lvl, 3),
            "focal_rel": lvl * S["focal_rel"] * u[0], "aspect_rel": lvl * S["aspect_rel"] * u[1],
            "ppx_px": lvl * pp * u[2], "ppy_px": lvl * pp * u[3], "k1": lvl * S["k1"] * u[4],
            "t_m": [lvl * S["trans_m"] * u[5], lvl * S["trans_m"] * u[6], lvl * S["trans_m"] * u[7]],
            "r_deg": [lvl * S["rot_deg"] * u[8], lvl * S["rot_deg"] * u[9], lvl * S["rot_deg"] * u[10]]}


def calib_summary(err):
    return (f"f {100 * err['focal_rel']:+.2f}% | pp ({err['ppx_px']:+.1f},{err['ppy_px']:+.1f}) px | k1 {err['k1']:+.3f}"
            f" | hand-eye t ({', '.join(f'{1000 * v:+.1f}' for v in err['t_m'])}) mm"
            f" r ({', '.join(f'{v:+.2f}' for v in err['r_deg'])}) deg")


def calib_apply(K, T, err):
    """-> (K_believed, T_believed): the camera model perception uses (true K, T = the rendered camera)."""
    import numpy as np
    K2 = dict(K)
    if not err or float(err.get("level", 0.0)) <= 0.0:
        return K2, np.asarray(T, float)
    K2["fx"] = K["fx"] * (1.0 + err["focal_rel"])
    K2["fy"] = K["fy"] * (1.0 + err["focal_rel"] + err["aspect_rel"])
    K2["ppx"] = K["ppx"] + err["ppx_px"]
    K2["ppy"] = K["ppy"] + err["ppy_px"]
    rx, ry, rz = [math.radians(v) for v in err["r_deg"]]
    dR = np.asarray(_m_mul(_rz(rz), _m_mul(_ry(ry), _rx(rx))), float)
    dT = np.eye(4)
    dT[:3, :3] = dR
    dT[:3, 3] = err["t_m"]
    return K2, np.asarray(T, float) @ dT


def distort_residual(img, K, k1, nearest=False):
    """Image the REAL camera delivers when its lens has a residual radial distortion k1 the believed model does
    not contain: out(u_d) = render(u) with x_u = x_d / (1 + k1 r_d^2) (normalised with the true K)."""
    import numpy as np
    if abs(float(k1)) < 1e-6:
        return img
    h, w = img.shape[:2]
    vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
    xd = (uu - K["ppx"]) / K["fx"]; yd = (vv - K["ppy"]) / K["fy"]
    s = 1.0 / (1.0 + float(k1) * (xd * xd + yd * yd))
    us = xd * s * K["fx"] + K["ppx"]; vs = yd * s * K["fy"] + K["ppy"]
    if nearest:
        ui = np.clip(np.rint(us).astype(int), 0, w - 1); vi = np.clip(np.rint(vs).astype(int), 0, h - 1)
        return img[vi, ui]
    u0 = np.clip(np.floor(us).astype(int), 0, w - 2); v0 = np.clip(np.floor(vs).astype(int), 0, h - 2)
    fu = np.clip(us - u0, 0.0, 1.0)[..., None]; fv = np.clip(vs - v0, 0.0, 1.0)[..., None]
    I = img.astype(np.float32)
    if I.ndim == 2:
        I = I[..., None]
    out = (I[v0, u0] * (1 - fu) * (1 - fv) + I[v0, u0 + 1] * fu * (1 - fv) + I[v0 + 1, u0] * (1 - fu) * fv
           + I[v0 + 1, u0 + 1] * fu * fv)
    out = out[..., 0] if img.ndim == 2 else out
    return np.clip(np.rint(out), 0, 255).astype(img.dtype) if img.dtype == np.uint8 else out.astype(img.dtype)


def triangulate(origins, dirs):
    """Least-squares intersection of rays -> (point, rms distance to the rays m, max angle between rays deg)."""
    import numpy as np
    O = np.asarray(origins, float); D = np.asarray(dirs, float)
    D = D / np.linalg.norm(D, axis=1, keepdims=True)
    A = np.zeros((3, 3)); b = np.zeros(3)
    for o, d in zip(O, D):
        M = np.eye(3) - np.outer(d, d)
        A += M; b += M @ o
    if np.linalg.cond(A) > 1e8:
        return None, None, 0.0
    p = np.linalg.solve(A, b)
    res = [float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (p - o))) for o, d in zip(O, D)]
    ang = 0.0
    for i in range(len(D)):
        for j in range(i + 1, len(D)):
            ang = max(ang, math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(D[i], D[j])))))))
    return p, float(np.sqrt(np.mean(np.square(res)))), ang


class EggMeasure:
    """V3.35: per-name ray observations -> triangulated centre + measured size (RGB-only wrist camera)."""

    def __init__(self):
        self.obs = {}
        self.min_angle = float(os.environ.get("NERO_V335_TRI_MIN_ANGLE_DEG", "4.0"))
        self.max_rms = float(os.environ.get("NERO_V335_TRI_MAX_RMS_M", "0.012"))
        self.moved = float(os.environ.get("NERO_V335_TRI_MOVED_M", "0.020"))
        self.keep = int(os.environ.get("NERO_V335_TRI_KEEP", "10"))
        self.est = {}
        self.suspect = {}

    def reset(self, name=None):
        if name is None:
            self.obs.clear(); self.est.clear(); self.suspect.clear()
        else:
            self.obs.pop(name, None); self.est.pop(name, None); self.suspect.pop(name, None)

    def is_suspect(self, name):
        return int(self.suspect.get(name, 0)) > 0

    def add(self, name, det, K, T, seq):
        import numpy as np
        if not det or not det.get("px_uv"):
            return
        o, d = pixel_rays(K, T, np.array([det["px_uv"][0]]), np.array([det["px_uv"][1]]))
        d = d[0]
        e = self.est.get(name)
        if e is not None and e.get("ok"):
            p = np.asarray(e["xyz_m"], float)
            dist = float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (p - o)))
            if dist > max(self.moved, 3.0 * float(e.get("rms_m") or 0.0)):
                # this view disagrees with the triangulation: once = suspect (reported as possibly moved, the old
                # rays are kept), twice in a row = the egg WAS moved -> start a new measurement from this view
                self.suspect[name] = int(self.suspect.get(name, 0)) + 1
                if self.suspect[name] < 2:
                    return
                self.reset(name)
            else:
                self.suspect.pop(name, None)
        else:
            prev = self.obs.get(name) or []
            if prev and math.dist(prev[-1]["plane_xyz"], det["xyz_m"]) > 1.5 * self.moved + 0.010:
                self.reset(name)
        lst = self.obs.setdefault(name, [])
        lst.append({"o": [float(v) for v in o], "d": [float(v) for v in d], "seq": int(seq),
                    "cov": det.get("px_cov"), "size_ok": bool(det.get("size_ok")), "f": float(K["fx"]),
                    "border": bool((det.get("quality") or {}).get("border")),
                    "yaw": det.get("plane_yaw"), "elong": det.get("elong"),
                    "z_axis": [float(v) for v in np.asarray(T, float)[:3, 2]], "plane_xyz": list(det["xyz_m"])})
        del lst[:-self.keep]
        self._solve(name)

    def _solve(self, name):
        import numpy as np
        lst_all = self.obs.get(name) or []
        lst = [x for x in lst_all if not x.get("border")] or lst_all   # clipped blobs have biased centroids
        if len(lst) < 2:
            self.est.pop(name, None)
            return
        n_in339 = len(lst)
        lst = self._consensus(lst)                       # V3.39: largest set of mutually consistent rays
        n_out339 = n_in339 - len(lst)
        p, rms, ang = triangulate([x["o"] for x in lst], [x["d"] for x in lst])
        # robust: drop the worst ray while it is an outlier (partly occluded / clipped / mis-associated blob)
        while p is not None and len(lst) > 2:
            res = [float(np.linalg.norm((np.eye(3) - np.outer(x["d"], x["d"])) @ (p - np.asarray(x["o"])))) for x in lst]
            k = int(np.argmax(res))
            med = float(np.median(res))
            if res[k] <= max(0.004, 2.5 * med):
                break
            lst = lst[:k] + lst[k + 1:]
            p, rms, ang = triangulate([x["o"] for x in lst], [x["d"] for x in lst])
        if p is None or ang < self.min_angle or rms > self.max_rms:
            self.est[name] = {"ok": False, "n_views": len(lst), "max_angle_deg": round(ang, 1),
                              "rms_m": None if rms is None else round(rms, 4), "n_outlier_rays": n_out339,
                              "fail_reason": ("rays nearly parallel" if (p is None or ang < self.min_angle)
                                              else f"rays disagree (rms {1000 * rms:.1f} mm)"),
                              "xyz_m": [round(float(v), 5) for v in (p if p is not None else lst[-1]["plane_xyz"])]}
            return
        W, L = [], []
        for x in lst:
            if not x["size_ok"] or not x["cov"]:
                continue
            depth = float(np.dot(p - np.asarray(x["o"]), np.asarray(x["z_axis"])))
            ev = np.linalg.eigvalsh(np.asarray(x["cov"], float))
            if depth <= 0 or ev[0] <= 0:
                continue
            W.append(4.0 * math.sqrt(ev[0]) * depth / x["f"])
            L.append(4.0 * math.sqrt(ev[1]) * depth / x["f"])
        # V3.36 size: only isolated, complete, unoccluded blobs (size_ok) from >= 2 views that AGREE
        spread = float(os.environ.get("NERO_V336_SIZE_MAX_SPREAD", "1.25"))
        size_state = "measured"
        if len(W) < int(os.environ.get("NERO_V336_SIZE_MIN_VIEWS", "2")):
            size_state = "not enough clean views"
        elif max(W) / max(min(W), 1e-6) > spread or max(L) / max(min(L), 1e-6) > spread:
            size_state = "views disagree"
        # V3.36 yaw: axial mean of the plane footprints of all views, weighted by elongation
        sy = cy = 0.0
        for x in lst:
            if x.get("yaw") is None or not x.get("elong"):
                continue
            wgt = max(0.0, float(x["elong"]) - 1.05)
            sy += wgt * math.sin(2.0 * float(x["yaw"])); cy += wgt * math.cos(2.0 * float(x["yaw"]))
        yaw_f = 0.5 * math.atan2(sy, cy) if (sy or cy) else None
        self.est[name] = {"ok": True, "xyz_m": [round(float(v), 5) for v in p], "n_views": len(lst),
                          "n_rays_total": len(lst_all),
                          "max_angle_deg": round(ang, 1), "rms_m": round(rms, 4), "n_outlier_rays": n_out339,
                          "width_m": round(float(np.median(W)), 4) if (W and size_state == "measured") else None,
                          "length_m": round(float(np.median(L)), 4) if (L and size_state == "measured") else None,
                          "n_size": len(W), "size_state": size_state,
                          "width_views_mm": [round(1000 * w, 1) for w in W][-6:],
                          "yaw_rad": None if yaw_f is None else round(yaw_f, 4),
                          "_rays": [(x["o"], x["d"]) for x in lst]}

    def _consensus(self, lst):
        """V3.39: eggs touching in a pile -> some rays of a name belong to the NEIGHBOUR (mis-association). V3.38 only
        dropped the single worst ray while it was > 2.5 x the median residual, which fails when 3-5 of 10 rays are
        wrong (median itself large -> egg_2: 10 views, never triangulated). Now: every ray pair with enough angle is
        a hypothesis; the hypothesis with the most rays within NERO_V339_TRI_INLIER_M wins (ties: the one that keeps
        the latest view, then the smaller residual); only its rays are used."""
        import numpy as np
        if len(lst) <= 2 or os.environ.get("NERO_V339_TRI_CONSENSUS", "1") == "0":
            return lst
        thr = float(os.environ.get("NERO_V339_TRI_INLIER_M", "0.006"))
        O = np.asarray([x["o"] for x in lst], float); D = np.asarray([x["d"] for x in lst], float)
        cos_min = math.cos(math.radians(self.min_angle))
        best = None
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if float(np.dot(D[i], D[j])) > cos_min:
                    continue
                p, _r, _a = triangulate(O[[i, j]], D[[i, j]])
                if p is None:
                    continue
                dv = p[None, :] - O
                res = np.linalg.norm(dv - (dv * D).sum(axis=1, keepdims=True) * D, axis=1)
                inl = [k for k in range(len(lst)) if res[k] <= thr]
                score = (len(inl), int(len(lst) - 1 in inl), -float(np.sum(res[inl])))
                if best is None or score > best[0]:
                    best = (score, inl)
        if best is None or len(best[1]) < 2:
            return lst
        return [lst[k] for k in best[1]]

    def plane_estimate(self, name, z_plane, tol=None, min_cam_dist=None):
        """V3.39: rays of a name intersected with a known horizontal plane (lying egg: table + half width). The
        largest group of intersections within tol that comes from camera positions >= min_cam_dist apart is the
        estimate (at the wrong height the views would NOT agree, so two agreeing views confirm plane + xy).
        -> dict(xyz_m, n_views, spread_mm) or None"""
        import numpy as np
        tol = float(os.environ.get("NERO_V339_PLANE_TOL_M", "0.008")) if tol is None else tol
        min_cam_dist = (float(os.environ.get("NERO_V339_PLANE_MIN_CAM_DIST_M", "0.05")) if min_cam_dist is None
                        else min_cam_dist)
        lst = [x for x in (self.obs.get(name) or []) if not x.get("border")]
        pts, org = [], []
        for x in lst:
            o, d = x["o"], x["d"]
            if abs(d[2]) < 1e-6:
                continue
            t = (float(z_plane) - o[2]) / d[2]
            if t <= 0:
                continue
            pts.append([o[0] + d[0] * t, o[1] + d[1] * t]); org.append(o)
        if len(pts) < 2:
            return None
        P = np.asarray(pts)
        best = None
        for k in range(len(P)):
            grp = [m for m in range(len(P)) if float(np.hypot(*(P[m] - P[k]))) <= tol]
            spread_ok = any(math.dist(org[a], org[b]) >= min_cam_dist for a in grp for b in grp)
            if len(grp) >= 2 and spread_ok and (best is None or (len(grp), int(len(P) - 1 in grp)) > best[0]):
                best = ((len(grp), int(len(P) - 1 in grp)), grp)
        if best is None:
            return None
        Q = P[best[1]]
        c = np.median(Q, axis=0)
        return {"xyz_m": [round(float(c[0]), 5), round(float(c[1]), 5), round(float(z_plane), 5)],
                "n_views": len(best[1]), "n_rays": len(P),
                "spread_mm": round(1000.0 * float(np.max(np.hypot(Q[:, 0] - c[0], Q[:, 1] - c[1]))), 1)}

    def height_check(self, name, table_z, tol=None, half_width=None):
        """V3.36: a lying egg's centre must be at (measured table) + (measured half width). If the triangulated height
        disagrees by more than tol, the rays are re-intersected with that plane (xy from the measurement, height
        from the measured table + measured size). -> (xyz, z_source)
        V3.39: also without a measured width (class prior half width - V3.38 skipped the check, so egg_1 was planned
        at z 0.048 instead of ~0.022); the plane xy prefers the steep rays (least sensitive to a height error)."""
        import numpy as np
        e = self.est.get(name)
        if not e or not e.get("ok") or table_z is None:
            return (e or {}).get("xyz_m"), "triangulated"
        if e.get("width_m"):
            hw, hw_src = float(e["width_m"]) / 2.0, "measured half width"
        elif half_width is not None:
            hw, hw_src = float(half_width), "class half width"
        else:
            return e["xyz_m"], "triangulated"
        tol = float(os.environ.get("NERO_V336_HEIGHT_TOL_M", "0.008")) if tol is None else tol
        z_chk = float(table_z) + hw
        if abs(float(e["xyz_m"][2]) - z_chk) <= tol:
            return e["xyz_m"], f"triangulated (consistent with table + {hw_src})"
        rays = [(o, d) for o, d in (e.get("_rays") or []) if abs(d[2]) > 1e-6]
        steep = [(o, d) for o, d in rays if abs(d[2]) >= float(os.environ.get("NERO_V339_STEEP_RAY_DZ", "0.6"))]
        pts = []
        for o, d in (steep if len(steep) >= 2 else rays):
            t = (z_chk - o[2]) / d[2]
            if t > 0:
                pts.append([o[0] + d[0] * t, o[1] + d[1] * t])
        if not pts:
            return e["xyz_m"], "triangulated"
        P = np.asarray(pts)
        return [round(float(np.median(P[:, 0])), 5), round(float(np.median(P[:, 1])), 5), round(z_chk, 5)], \
            f"rays at table + {hw_src} (triangulated z {e['xyz_m'][2]:.4f} disagreed)"

    def get(self, name):
        return self.est.get(name)

    def table_z(self, locations):
        """Table height from measured eggs lying on it: centre z - half the measured width (median over eggs).
        V3.39: only well-conditioned triangulations (ray angle >= NERO_V339_TABLE_MIN_ANGLE_DEG, rms <=
        NERO_V339_TABLE_MAX_RMS_M) count - V3.38 also took eggs whose triangulated z was 25-50 mm too high."""
        amin = float(os.environ.get("NERO_V339_TABLE_MIN_ANGLE_DEG", "12"))
        rmax = float(os.environ.get("NERO_V339_TABLE_MAX_RMS_M", "0.004"))
        zs = [e["xyz_m"][2] - e["width_m"] / 2.0 for n, e in self.est.items()
              if e.get("ok") and e.get("width_m") and str(locations.get(n, "")).startswith("table")
              and float(e.get("max_angle_deg") or 0.0) >= amin and float(e.get("rms_m") or 1.0) <= rmax]
        self._table_n = len(zs)
        if len(zs) < 1:
            return None
        zs.sort()
        return round(zs[len(zs) // 2], 4)

    def plane(self, locations):
        """V3.40 TABLE PLANE (one value per episode): the calibrated plane (NERO_V330_TABLE_Z_M) until >= 2
        well-measured lying eggs agree with it within NERO_V340_TABLE_MAX_DEV_M (5 mm); then that measured value is
        LOCKED. V3.39 log: the measured plane drifted -0.006 -> +0.009 -> +0.012 m (all accepted within 15 mm) -> every
        egg z jumped ~15 mm -> false BYSTANDER_EGG_DISTURBED, implausible carton height. -> (z, source)"""
        if os.environ.get("NERO_V339_TABLE_FALLBACK", "1") == "0":
            m = self.table_z(locations)
            return (m, "measured (eggs)") if m is not None else (None, "not measured")
        lock = getattr(self, "table_lock", None)
        if lock is not None:
            return lock, "measured (eggs), locked"
        z_cal = table_z_m()
        m = self.table_z(locations)
        n = int(getattr(self, "_table_n", 0))
        dev = float(os.environ.get("NERO_V340_TABLE_MAX_DEV_M", "0.005"))
        if m is not None and n >= int(os.environ.get("NERO_V340_TABLE_MIN_EGGS", "2")) and abs(m - z_cal) <= dev:
            self.table_lock = float(m)
            return self.table_lock, "measured (eggs), locked"
        return z_cal, ("calibrated table plane" if m is None
                       else f"calibrated table plane (measured {m:.4f} from {n} egg(s) not used)")


def table_plane(measured=None):
    """V3.39: the table plane used to check lying eggs: the measured one (eggs) if it agrees with the robot-mounting
    calibration (NERO_V330_TABLE_Z_M) within NERO_V339_TABLE_MAX_DEV_M, else the calibrated plane. V3.38 used ONLY
    the measured plane, which needs measured egg sizes first -> no check at all during the first observes.
    -> (z, source)"""
    z_cal = table_z_m()
    if os.environ.get("NERO_V339_TABLE_FALLBACK", "1") == "0":
        return (measured, "measured (eggs)") if measured is not None else (None, "not measured")
    if measured is None:
        return z_cal, "calibrated table plane"
    if abs(float(measured) - z_cal) <= float(os.environ.get("NERO_V340_TABLE_MAX_DEV_M", "0.005")):
        return float(measured), "measured (eggs)"
    return z_cal, f"calibrated table plane (measured {float(measured):.4f} rejected)"


def apply_measurements(tracked, dets, meas, K, T, seq, held=None, require=True, placed=None):
    """Fold the current view into EggMeasure and replace the plane-assumption positions in the tracker output by
    the triangulated ones. require -> an egg without a valid triangulation is reported seen but NOT trusted.
    A held egg (and a placed one whose hint is far from its old measurement) starts a new measurement."""
    byid = {d["det_id"]: d for d in dets}
    for name, o in tracked.items():                       # fold this view in first, then judge
        if name != held and o.get("seen") and o.get("det_id") in byid:
            meas.add(name, byid[o["det_id"]], K, T, seq)
    table_z336, tz_src339 = meas.plane({n_: o_.get("location") for n_, o_ in tracked.items()})
    try:
        import NERO_V308_Camera_Perception as _P8
        hw339 = float(_P8.EGG_PRIOR["half_short_m"])
    except Exception:
        hw339 = 0.0219
    for name, p in (placed or {}).items():
        e = meas.get(name)
        if e and math.dist(e["xyz_m"], p) > meas.moved:
            meas.reset(name)
    for name, o in tracked.items():
        if name == held:
            meas.reset(name)
            continue
        e = meas.get(name)
        if o.get("seen") and o.get("det_id") in byid:
            o["this_view_xyz_m"] = list(byid[o["det_id"]]["xyz_m"])     # V3.39: single-view (plane) position
        if meas.is_suspect(name) and o.get("seen"):
            o["measurement"] = {"method": "this view disagrees with the triangulation (egg moved?)",
                                "n_views": 1, "previous_triangulated_xyz_m": (e or {}).get("xyz_m")}
            o["trusted"] = False
            o["untrusted_reason"] = "possibly moved - position from this single view"
            continue
        if e and e.get("ok"):
            xyz336, zsrc = list(e["xyz_m"]), "triangulated"
            if str(o.get("location", "")).startswith("table"):
                xyz336, zsrc = meas.height_check(name, table_z336, half_width=hw339)
                if e.get("yaw_rad") is not None:            # V3.36 multi-view yaw of a lying egg
                    ax = [math.cos(e["yaw_rad"]), math.sin(e["yaw_rad"]), 0.0]
                    o["axis_world"] = ax
                    o["quat_xyzw"] = [round(v, 6) for v in _quat_from_axis_world(ax)]
            o["xyz_m"] = list(xyz336)
            o["measurement"] = {"method": "multi_view_triangulation", "n_views": e["n_views"],
                                "max_ray_angle_deg": e["max_angle_deg"], "ray_rms_mm": round(1000 * e["rms_m"], 1),
                                "width_mm": None if e.get("width_m") is None else round(1000 * e["width_m"], 1),
                                "length_mm": None if e.get("length_m") is None else round(1000 * e["length_m"], 1),
                                "size_views": e.get("n_size", 0), "size_state": e.get("size_state"),
                                "height_source": zsrc, "yaw_views_fused": e.get("yaw_rad") is not None,
                                "outlier_rays": e.get("n_outlier_rays", 0), "table_plane": tz_src339}
            continue
        pe339 = None
        if (str(o.get("location", "")).startswith("table") and table_z336 is not None
                and os.environ.get("NERO_V339_PLANE_FALLBACK", "1") != "0"):
            pe339 = meas.plane_estimate(name, float(table_z336) + hw339)
        if pe339 is not None:
            # V3.39: triangulation not usable (rays nearly parallel / mixed with a neighbour) but >= 2 views from
            # different camera positions agree on the lying-egg plane -> that position is measured and trusted
            o["xyz_m"] = list(pe339["xyz_m"])
            o["measurement"] = {"method": "rays_at_table_plane", "n_views": pe339["n_views"],
                                "rays_total": pe339["n_rays"], "plane_spread_mm": pe339["spread_mm"],
                                "triangulation": (e or {}).get("fail_reason"),
                                "height_source": f"table + class half width ({tz_src339})", "table_plane": tz_src339}
        else:
            o["measurement"] = {"method": "single_view_plane_assumption (not measured yet)",
                                "n_views": (e or {}).get("n_views", 1 if o.get("seen") else 0),
                                "max_ray_angle_deg": (e or {}).get("max_angle_deg")}
            if require and o.get("seen"):
                o["trusted"] = False
                o["untrusted_reason"] = "position not triangulated yet (needs a 2nd view from another pose)"
    return tracked


class HolderHeight:
    """Carton top height by multi-view parallax (no CAD height): the far-edge rectangle of each view is cast onto a
    candidate plane z; only at the true height do the centres from different camera poses coincide.
    V3.36 (two views can almost always be made to agree at SOME height - V3.35 accepted 0.009 m):
      * >= 3 views, camera positions >= 0.10 m apart, view directions to the carton >= 15 deg apart
      * a sharp minimum: the centre spread at z0 +- 10 mm >= 3 x the spread at z0 (+ 2 mm)
      * spread at z0 <= 6 mm; plausible: 20..120 mm above the measured table (if known), far-edge rectangle
        not smaller than 85 % of the model outline
      * once accepted the height is LOCKED (later views do not re-solve) until the carton is confirmed moved"""

    def __init__(self):
        self.views = []
        self.est = None
        self.lock = None
        self.keep = int(os.environ.get("NERO_V335_HOLDER_VIEWS", "6"))
        self.n_add = 0

    def reset(self):
        self.views, self.est, self.lock = [], None, None

    def add(self, px, K, T, table_z=None):
        """One carton view. Cast onto horizontal planes the pixel set is a homothety about the camera's foot point
        (ratio (cam_z - z) / (cam_z - z_ref)), so each view is ray-cast ONCE and its yaw found once."""
        import numpy as np
        if self.lock is not None:
            return
        uu, vv, border = px
        step = max(1, len(uu) // 2500)
        T = np.asarray(T, float)
        z_ref = float(holder_model()["plateau_z"])
        P = rays_to_plane(K, T, uu[::step], vv[::step], z_ref)
        ok = np.isfinite(P[:, 0])
        if ok.sum() < 40 or T[2, 3] <= z_ref + 0.02:
            return
        Q0, b0 = P[ok, :2], border[::step][ok]
        try:
            yaw = _min_area_yaw(Q0)
        except Exception:
            return
        vw = {"Q0": Q0, "b": b0, "yaw": yaw, "T": T, "z_ref": z_ref, "c_ref": Q0.mean(axis=0)}
        if self.views and np.linalg.norm(self.views[-1]["T"][:3, 3] - T[:3, 3]) < 0.01:
            self.views[-1] = vw
        else:
            self.views.append(vw)
        del self.views[:-self.keep]
        self.table_z = table_z
        self.n_add += 1
        self._solve()

    def _centres(self, z, with_ext=False):
        import numpy as np
        L = np.asarray(holder_model()["outer"], float)
        out = []
        for vw in self.views:
            cz = float(vw["T"][2, 3])
            if cz <= z + 0.02:
                continue
            c = vw["T"][:2, 3]
            Qz = c + (vw["Q0"] - c) * ((cz - z) / (cz - vw["z_ref"]))
            g = _carton_geom(Qz, c, L, vw["b"], yaw=vw["yaw"])
            if g is not None and all(g["far_ok"]):
                out.append((g["center"], vw["T"][:3, 3], g["ext"]) if with_ext else (g["center"], vw["T"][:3, 3]))
        return out

    def _geometry_ok(self):
        """>= 3 views, baseline and view-direction spread."""
        import numpy as np
        n_min = int(os.environ.get("NERO_V336_HOLDER_MIN_VIEWS", "3"))
        if len(self.views) < n_min:
            return False, 0.0, 0.0
        pos = [v["T"][:3, 3] for v in self.views]
        base = max(float(np.linalg.norm(a - b)) for a in pos for b in pos)
        dirs = []
        for v in self.views:
            d = np.array([v["c_ref"][0], v["c_ref"][1], v["z_ref"]]) - v["T"][:3, 3]
            dirs.append(d / np.linalg.norm(d))
        ang = max(math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(a, b)))))) for a in dirs for b in dirs)
        ok = (base >= float(os.environ.get("NERO_V336_HOLDER_MIN_BASELINE_M", "0.10"))
              and ang >= float(os.environ.get("NERO_V336_HOLDER_MIN_VIEW_ANGLE_DEG", "15")))
        return ok, base, ang

    def _solve(self):
        import numpy as np
        geo_ok, base, ang = self._geometry_ok()
        if not geo_ok:
            self.est = {"ok": False, "reason": f"views {len(self.views)}, baseline {base:.2f} m, view angle {ang:.0f} deg"}
            return
        lo, hi = [float(v) for v in os.environ.get("NERO_V335_HOLDER_Z_RANGE_M", "-0.02,0.14").split(",")]

        def cost(z):
            c = self._centres(z)
            if len(c) < 3:
                return None, len(c)
            C = np.array([x[0] for x in c])
            return float(np.sqrt(np.mean(np.sum((C - C.mean(axis=0)) ** 2, axis=1)))), len(c)
        best = None
        for z in np.arange(lo, hi + 1e-9, 0.006):
            cz, n = cost(float(z))
            if cz is not None and (best is None or cz < best[0]):
                best = (cz, float(z), n)
        if best is None:
            self.est = {"ok": False, "reason": "fewer than 3 views with both far edges visible"}
            return
        z0 = best[1]
        for z in np.arange(z0 - 0.006, z0 + 0.0061, 0.001):
            cz, n = cost(float(z))
            if cz is not None and cz < best[0]:
                best = (cz, float(z), n)
        z0, c0 = best[1], best[0]
        side = [cost(z0 - 0.010)[0], cost(z0 + 0.010)[0]]
        sharp = all(s is not None and s >= 3.0 * c0 + 0.002 for s in side)
        L = np.asarray(holder_model()["outer"], float)
        ext = [x[2] for x in self._centres(z0, with_ext=True)]
        ext_ok = bool(bool(ext) and float(np.median([e[0] for e in ext])) >= 0.85 * float(L[0]) and
                      float(np.median([e[1] for e in ext])) >= 0.85 * float(L[1]))   # V3.38: plain bool (JSON)
        tz = getattr(self, "table_z", None)
        plaus = True if tz is None else (0.020 <= z0 - float(tz) <= 0.120)
        ok = bool(c0 <= float(os.environ.get("NERO_V336_HOLDER_MAX_SPREAD_M", "0.006")) and sharp and ext_ok and plaus)
        self.est = {"top_z_m": round(z0, 4), "centre_spread_mm": round(1000 * c0, 1), "n_views": best[2],
                    "baseline_m": round(float(base), 3), "view_angle_deg": round(float(ang), 1),
                    "sharp_minimum": bool(sharp), "extent_ok": bool(ext_ok), "plausible_vs_table": bool(plaus), "ok": bool(ok),
                    "locked": ok}
        if ok:
            self.lock = dict(self.est)

    def fused_center(self):
        """V3.38: carton centre fused over all stored views at the MEASURED (locked) top height - the far-edge
        rectangle of every view re-projected to that plane and averaged. None before the height is measured."""
        import numpy as np
        if not self.lock:
            return None
        c = self._centres(float(self.lock["top_z_m"]))
        if len(c) < 2:
            return None
        C = np.array([x[0] for x in c], float)
        return [float(C[:, 0].mean()), float(C[:, 1].mean())]

    def top_z(self):
        return self.lock["top_z_m"] if self.lock else None


class InHandCheck:
    """V3.35: the held egg's image position relative to where the in-hand model puts it, compared with the same
    offset right after the grasp (fingers occlude the same parts, so a change = the egg moved in the fingers)."""

    def __init__(self):
        self.ref = {}

    def check(self, rgb, K, T, held, pred_xyz, robot_mask, stride=2):
        import numpy as np
        import NERO_V308_Camera_Perception as P8
        if not held or pred_xyz is None:
            self.ref.clear()
            return None
        Tm = np.asarray(T, float)
        pc = (np.asarray(pred_xyz, float) - Tm[:3, 3]) @ Tm[:3, :3]
        if pc[2] <= 0.02:
            return {"name": held, "visible": False, "reason": "predicted egg behind the camera"}
        u0 = K["fx"] * pc[0] / pc[2] + K["ppx"]; v0 = K["fy"] * pc[1] / pc[2] + K["ppy"]
        mask = P8.egg_color_mask(rgb)[::stride, ::stride]
        if robot_mask is not None:
            mask &= ~robot_mask
        best = None
        for rr, cc in _pixel_components(mask, 25):
            uc, vc = float(cc.mean() * stride), float(rr.mean() * stride)
            dpx = math.hypot(uc - u0, vc - v0)
            if best is None or len(rr) / (1.0 + dpx / 40.0) > best[0]:
                best = (len(rr) / (1.0 + dpx / 40.0), uc, vc, len(rr))
        rad_px = float(os.environ.get("NERO_V335_INHAND_SEARCH_PX", "260"))
        if best is None or math.hypot(best[1] - u0, best[2] - v0) > rad_px:
            return {"name": held, "visible": False, "pred_px": [round(u0, 1), round(v0, 1)],
                    "reason": "no egg-coloured region near the gripper (occluded or lost)"}
        off = (best[1] - u0, best[2] - v0)
        if held not in self.ref:
            self.ref[held] = off
        r = self.ref[held]
        shift_px = math.hypot(off[0] - r[0], off[1] - r[1])
        return {"name": held, "visible": True, "pred_px": [round(u0, 1), round(v0, 1)],
                "blob_px": [round(best[1], 1), round(best[2], 1)], "blob_area_px": int(best[3] * stride * stride),
                "shift_px": round(shift_px, 1), "shift_mm": round(1000.0 * shift_px * float(pc[2]) / K["fx"], 1),
                "depth_m": round(float(pc[2]), 3)}


def live_view_write(out_dir, rgb, K, T, tracked, dets, holder, capture_no, tag, extra=None, scale=2):
    """V3.35 GUI monitor: <out_dir>/live_view.png (downscaled) + live_view.json (labels from the perception output
    only: tracked names <- det_id, unnamed / awaiting-confirmation detections, carton rectangle + slot names)."""
    import numpy as np
    import NERO_V308_Camera_Perception as P8
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = max(1, int(scale))
    img = np.ascontiguousarray(rgb[::s, ::s, :3])
    labels = []
    name_of = {o.get("det_id"): n for n, o in (tracked or {}).items() if o.get("seen") and o.get("det_id") is not None}
    for d in dets or []:
        bb = d.get("px_bbox")
        if not bb:
            continue
        n = name_of.get(d["det_id"])
        o = (tracked or {}).get(n) or {}
        meas = o.get("measurement") or {}
        if n:
            txt = n + ("" if o.get("trusted") else " (unconfirmed)")
            if meas.get("width_mm"):
                txt += f" {meas['width_mm']:.0f}x{meas.get('length_mm') or 0:.0f}mm"
        else:
            txt = "egg? (awaiting 2nd view)" if d.get("trusted") else "egg? (low quality)"
        labels.append({"kind": "egg", "name": n, "text": txt, "trusted": bool(o.get("trusted")) if n else False,
                       "bbox": [bb[0] / s, bb[1] / s, bb[2] / s, bb[3] / s]})
    if holder:
        Tm = np.asarray(T, float)

        def proj(p):
            pc = (np.asarray(p, float) - Tm[:3, 3]) @ Tm[:3, :3]
            if pc[2] <= 0.02:
                return None
            return [(K["fx"] * pc[0] / pc[2] + K["ppx"]) / s, (K["fy"] * pc[1] / pc[2] + K["ppy"]) / s]
        m = holder_model()
        c, yaw = holder["center_xy_m"], float(holder["yaw_rad"])
        zt = float(holder.get("top_z_m", m["plateau_z"]))
        cu, su = math.cos(yaw), math.sin(yaw)
        poly = []
        for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            du, dv = a * m["outer"][0] / 2.0, b * m["outer"][1] / 2.0
            poly.append(proj([c[0] + du * cu - dv * su, c[1] + du * su + dv * cu, zt]))
        if all(p is not None for p in poly):
            labels.append({"kind": "holder", "text": "egg_holder" + (" (complete)" if holder.get("complete") else " (partial)"),
                           "poly": poly, "trusted": bool(holder.get("complete"))})
        for sn, sl in (holder.get("slots") or {}).items():
            p = proj(sl["xyz_m"])
            if p is not None:
                labels.append({"kind": "slot", "text": sn, "pt": p, "trusted": bool(holder.get("complete"))})
    tmp = out_dir / "live_view.tmp.png"
    P8.write_png_rgb8(tmp, img)
    os.replace(tmp, out_dir / "live_view.png")
    meta = {"capture": capture_no, "tag": tag, "t": time.time(), "width": int(img.shape[1]), "height": int(img.shape[0]),
            "labels": labels, **(extra or {})}
    tmpj = out_dir / "live_view.tmp.json"
    tmpj.write_text(json.dumps(json_safe(meta)), encoding="utf-8")
    os.replace(tmpj, out_dir / "live_view.json")
    return meta


# =========================================================================== camera process
def wrist_camera_process_main(qpos_arr, meta_arr, model_path_arr, trig_seq, trig_tag, status, stop_event, out_root,
                              gl_name, res_seq=None, res_buf=None):
    """Spawned process: renders the wrist camera at the simulator's joint state and runs the V3.10 perception.
    Same protocol as NERO_V308_Camera_Perception.camera_process_main (that module is not modified)."""
    prefix = f"[{VERSION} WRIST CAMERA]"
    try:
        import mujoco
        import numpy as np
        import NERO_V308_Camera_Perception as P8
    except Exception as exc:
        print(f"{prefix} import failed: {exc}", flush=True)
        status.value = 2
        return
    cfgs = {pf: wrist_config(pf) for pf in PROFILES}
    t_wait = time.time()
    path = ""
    while not stop_event.is_set():
        path = bytes(model_path_arr[:]).split(b"\0", 1)[0].decode("utf-8", "ignore")
        if path and meta_arr[3] > 0.5:
            break
        if time.time() - t_wait > 180:
            print(f"{prefix} simulator never published its model; wrist camera disabled", flush=True)
            status.value = 2
            return
        time.sleep(0.2)
    if stop_event.is_set():
        return
    try:
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        cam_ids = {}
        for pf, c in cfgs.items():
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c["name"])
            if cid >= 0:
                cam_ids[pf] = cid
        if "d435i" not in cam_ids and "usb130" not in cam_ids:
            raise RuntimeError(f"no wrist camera in model {path} (NERO_V326_WRIST=0 at sim start?)")
        model.vis.global_.offwidth = max([int(model.vis.global_.offwidth)] + [int(c["width"]) for c in cfgs.values()])
        model.vis.global_.offheight = max([int(model.vis.global_.offheight)] + [int(c["height"]) for c in cfgs.values()])
    except Exception as exc:
        print(f"{prefix} model load failed: {exc}", flush=True)
        status.value = 2
        return
    renderers = {}
    try:
        for pf in cam_ids:
            key = (int(cfgs[pf]["height"]), int(cfgs[pf]["width"]))
            if key not in renderers:
                renderers[key] = mujoco.Renderer(model, height=key[0], width=key[1])
    except Exception as exc:
        print(f"{prefix} OpenGL backend '{gl_name}' unavailable: {str(exc)[:200]}", flush=True)
        status.value = 2
        raise SystemExit(P8.EXIT_GL_FAILED)
    Ks = {pf: P8.intrinsics(cfgs[pf]) for pf in cam_ids}
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    pf0 = "d435i" if "d435i" in cam_ids else "usb130"
    T0 = optical_from_mujoco(data.cam_xpos[cam_ids[pf0]], data.cam_xmat[cam_ids[pf0]])
    cfg_w = dict(cfgs[pf0], pos=[round(float(v), 4) for v in T0[:3, 3]], target="moves with link7")
    writer = P8.CaptureWriter(out_root, cfg_w, Ks[pf0], T0, save_images=os.environ.get("NERO_V308_SAVE_IMAGES", "1") != "0")
    rng = np.random.default_rng(326)
    nq = int(model.nq)
    robot_geoms = P8._robot_geom_ids(mujoco, model)
    names = [n for n in os.environ.get("NERO_V330_OBJECT_NAMES", "egg_1,egg_2,egg_3,egg_4").split(",") if n]
    trackers = {pf: ViewTracker(P8.PerceptionTracker(), names=names) for pf in cam_ids}   # V3.28/V3.30
    if os.environ.get("NERO_V329_NO_PRIOR", "1") != "0":
        # V3.29: this process forgets the registered receptacle layout of the fixed-camera module (process-local);
        # slots are filled in from the holder this camera perceives.
        P8.SLOTS = {}
        P8.BASKET = {"center_xy_m": [0.0, 0.0], "radius_m": 0.0}
    holder_best = {pf: None for pf in cam_ids}
    prev_dets = {pf: [] for pf in cam_ids}
    meas = {pf: EggMeasure() for pf in cam_ids}               # V3.35 triangulation per camera
    hh_meas = {pf: HolderHeight() for pf in cam_ids}          # V3.35 carton top height by parallax
    hjumps = {pf: [] for pf in cam_ids}                       # V3.36 unconfirmed carton jumps
    hparts = {pf: [] for pf in cam_ids}                       # V3.39 partial carton views (fusion)
    inhand = {pf: InHandCheck() for pf in cam_ids}            # V3.35 held-egg image check
    capture_no = 0
    last_seq = int(trig_seq.value)
    status.value = 1
    print(f"{prefix} READY gl={gl_name} cameras " + ", ".join(
        f"{pf}={cfgs[pf]['name']} {Ks[pf]['width']}x{Ks[pf]['height']} fx={Ks[pf]['fx']:.1f} mount={cfgs[pf]['mount_pos']}"
        for pf in cam_ids) + f" -> {writer.run_dir}", flush=True)
    while not stop_event.is_set():
        cur = int(trig_seq.value)
        if cur == last_seq:
            time.sleep(0.05)
            continue
        last_seq = cur
        raw_tag = bytes(trig_tag[:]).split(b"\0", 1)[0].decode("utf-8", "ignore") or "trigger"
        tag, hints = decode_hints(raw_tag)
        t0 = time.perf_counter()
        try:
            pf = hints.get("cam") or "d435i"
            if pf not in cam_ids:
                raise RuntimeError(f"wrist camera profile {pf} not in the model")
            cfg, K_true, cam_id = cfgs[pf], Ks[pf], cam_ids[pf]
            renderer = renderers[(int(cfg["height"]), int(cfg["width"]))]
            with qpos_arr.get_lock():
                n = int(meta_arr[0])
                q = np.frombuffer(qpos_arr.get_obj(), dtype=np.float64, count=n).copy()
                sim_time = float(meta_arr[1])
            if n != nq:
                raise RuntimeError(f"qpos size mismatch sim={n} camera_model={nq}")
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            T_true = optical_from_mujoco(data.cam_xpos[cam_id], data.cam_xmat[cam_id])     # moves with the arm
            # V3.35: perception works with the BELIEVED calibration (GUI error level); the real camera = rendered one
            calib = calib_error(hints.get("calib", calib_level_default()), width=int(cfg["width"]))
            K, T = calib_apply(K_true, T_true, calib)
            renderer.disable_depth_rendering()
            renderer.update_scene(data, camera=cam_id)
            rgb = distort_residual(renderer.render().copy(), K_true, calib["k1"])
            truth = P8._truth_from_data(mujoco, model, data)
            boxes = P8._robot_boxes(model, data, robot_geoms)
            holder = None
            diag = {"camera": f"wrist_{pf}", "calib": calib}
            rmask = None
            if cfg.get("depth", True):
                renderer.enable_depth_rendering()
                renderer.update_scene(data, camera=cam_id)
                raw_depth = distort_residual(renderer.render().copy(), K_true, calib["k1"], nearest=True)
                renderer.disable_depth_rendering()
                n_fr = int(cfg["depth_frames"]) if float(cfg["depth_noise_k"]) > 0 else 1
                depth = P8.temporal_fuse([P8._depth_postprocess(raw_depth, cfg, rng) for _ in range(n_fr)])
                prior = ([d for d in prev_dets[pf] if d.get("trusted")]
                         if os.environ.get("NERO_V310_WARM_START", "1") != "0" else [])
                dets, diag2 = P8.perceive_eggs(rgb, depth, K, T, robot_boxes=boxes, prior=prior)
                diag.update(diag2)
                diag["camera"] = f"wrist_{pf}"
                diag["depth_frames_fused"] = n_fr
                diag["valid_depth_ratio"] = round(float((depth > 0).mean()), 3)
                P8._temporal_annotate(dets, prev_dets[pf])
                try:
                    holder = detect_holder(depth, K, T, dets=dets, robot_boxes=boxes)
                except Exception as exc_h:
                    diag["holder_error"] = str(exc_h)[:200]
            else:
                # V3.30 RGB-ONLY: model-based self mask, carton -> holder, colour blobs -> eggs.
                # V3.35: table / carton-top heights MEASURED (eggs triangulated, carton by parallax) once available.
                depth = np.zeros(rgb.shape[:2], dtype=np.float32)
                rmask = robot_pixel_mask(K, T, boxes, rgb.shape[:2], stride=2)
                hb = holder_best[pf]
                zt_meas, zt_src340 = meas[pf].plane({n_: o_.get("location") for n_, o_ in trackers[pf].base.tracks.items()})
                diag["table_plane_source"] = zt_src340
                z_top_meas = hh_meas[pf].top_z()
                dets = perceive_eggs_rgb(rgb, K, T, robot_mask=rmask, holder=hb, table_z=zt_meas)
                in_slot_xy = [d["xyz_m"][:2] for d in dets if d["location"] == "egg_holder_slot"]
                try:
                    holder = detect_holder_rgb(rgb, K, T, robot_mask=rmask, egg_xy_in_slots=in_slot_xy,
                                               table_z=zt_meas, z_top=z_top_meas)
                except Exception as exc_h:
                    diag["holder_error"] = str(exc_h)[:200]
                if holder is not None:
                    px335 = holder.pop("_px_uv", None)
                    hb_c = hb if (hb and hb.get("complete")) else None
                    jump336 = False
                    if (holder.get("complete") and hb_c
                            and math.dist(holder["center_xy_m"], hb_c["center_xy_m"]) > float(os.environ.get("NERO_V335_HOLDER_MOVED_M", "0.03"))):
                        # V3.36: believe "moved" only from two views taken from camera positions >= 5 cm apart
                        cam336 = [float(v) for v in T[:3, 3]]
                        jl = [j for j in hjumps[pf] if math.dist(j["c"], holder["center_xy_m"]) <= 0.015]
                        if any(math.dist(j["cam"], cam336) >= float(os.environ.get("NERO_V336_MOVED_MIN_CAM_DIST_M", "0.05"))
                               for j in jl):
                            hh_meas[pf].reset()            # confirmed: unlock + measure the height again
                            hjumps[pf] = []
                        else:
                            hjumps[pf] = jl + [{"c": list(holder["center_xy_m"]), "cam": cam336}]
                            jump336 = True                 # not confirmed: do not let this view replace the carton
                    elif holder.get("complete"):
                        hjumps[pf] = []
                    if px335 is not None and not jump336:
                        hh_meas[pf].add(px335, K, T, table_z=zt_meas)
                    holder["moved_suspect"] = jump336
                    fc338 = hh_meas[pf].fused_center() if (holder.get("complete") and not jump336
                                                            and os.environ.get("NERO_V338_FUSED_CENTRE", "1") != "0") else None
                    if fc338 is not None:
                        dx338 = fc338[0] - float(holder["center_xy_m"][0])
                        dy338 = fc338[1] - float(holder["center_xy_m"][1])
                        if math.hypot(dx338, dy338) <= float(os.environ.get("NERO_V338_FUSED_MAX_SHIFT_M", "0.015")):
                            # V3.38: the measured height is known -> use the multi-view centre for this view's slots
                            holder["center_xy_m"] = [round(fc338[0], 5), round(fc338[1], 5)]
                            for s338 in (holder.get("slots") or {}).values():
                                s338["xyz_m"] = [round(float(s338["xyz_m"][0]) + dx338, 5),
                                                 round(float(s338["xyz_m"][1]) + dy338, 5), s338["xyz_m"][2]]
                            holder["fused_centre_shift_mm"] = round(1000.0 * math.hypot(dx338, dy338), 1)
                    holder["top_z_source"] = ("measured_multi_view_parallax" if z_top_meas is not None
                                              else "carton model on the table plane (not measured yet)")
                    holder["height_measurement"] = hh_meas[pf].est
                    holder["slot_layout_source"] = ("carton model: 2 x 4 cells relative to the measured rectangle "
                                                    "(cell centres are not measured)"
                                                    + ("; centre = multi-view fusion at the measured height (V3.38)"
                                                       if holder.get("fused_centre_shift_mm") is not None else ""))
                if (holder is not None and not holder.get("complete") and not (hb or {}).get("complete")
                        and os.environ.get("NERO_V339_PARTIAL_FUSE", "1") != "0"):
                    hparts[pf] = (hparts[pf] + [{k: v for k, v in holder.items() if k != "slots"}])[-6:]
                    fz339 = fuse_partial_holders(hparts[pf])
                    if fz339 is not None:
                        fz339["top_z_source"] = holder.get("top_z_source")
                        fz339["height_measurement"] = holder.get("height_measurement")
                        fz339["slot_layout_source"] = ("carton model relative to the far edges of "
                                                       f"{fz339['fused_partial_views']} partial views (V3.39)")
                        diag["holder_partial_fusion"] = {"views": fz339["fused_partial_views"],
                                                         "spread_mm": fz339["fusion_spread_mm"]}
                        holder = fz339
                elif holder is not None and holder.get("complete"):
                    hparts[pf] = []
                if holder is not None and hb is None and holder.get("complete"):
                    dets = perceive_eggs_rgb(rgb, K, T, robot_mask=rmask, holder=holder, table_z=zt_meas)
                diag["rgb_only"] = True
                diag["robot_mask_ratio"] = round(float(rmask.mean()), 3)
                diag["measured_table_z_m"] = zt_meas
                diag["measured_holder_top_z_m"] = z_top_meas
            prev_dets[pf] = dets
            if holder is not None and not holder.get("moved_suspect") and (
                    holder.get("complete") or holder_quality(holder) >= holder_quality(holder_best[pf])):
                holder_best[pf] = holder                  # V3.33: the newest COMPLETE view replaces older ones
                if os.environ.get("NERO_V329_NO_PRIOR", "1") != "0":
                    P8.SLOTS = {k: (v["xyz_m"][0], v["xyz_m"][1]) for k, v in holder["slots"].items()}
            diag["holder"] = {k: v for k, v in (holder or {}).items() if k != "slots"} if holder else None
            # truth restricted to eggs inside the image (evaluation of a partial view; LOG ONLY) - true camera
            vis = {}
            for name_t, tv in truth.items():
                uv, z = P8.project(np.asarray([tv["xyz_m"]]), K_true, T_true)
                if z[0] > 0 and 0 <= uv[0, 0] < K_true["width"] and 0 <= uv[0, 1] < K_true["height"]:
                    vis[name_t] = tv
            rows, fps, match = P8.evaluate(dets, vis)
            summ = P8.summarize(rows, fps)
            # V3.30: no identity seeding from the simulator - new eggs take free names in a fixed scan order
            pred339 = None
            if not cfg.get("depth", True):
                # V3.39: measured positions (triangulated, lying eggs checked against the table plane) for the
                # close-view association
                tz339, _src339 = meas[pf].plane({n_: o_.get("location") for n_, o_ in trackers[pf].base.tracks.items()})
                pred339 = {}
                for n_, t_ in trackers[pf].base.tracks.items():
                    e_ = meas[pf].get(n_)
                    if e_ and e_.get("ok"):
                        xyz_, _s = (meas[pf].height_check(n_, tz339, half_width=float(P8.EGG_PRIOR["half_short_m"]))
                                    if str(t_.get("location", "")).startswith("table") else (e_["xyz_m"], ""))
                        pred339[n_] = xyz_
            tracked, track_diag = trackers[pf].update(dets, capture_no + 1, K, T.tolist(), seed_match=None, hints=hints,
                                                      z_max=float(cfg["max_range_m"]), predict=pred339)
            if not cfg.get("depth", True):
                apply_measurements(tracked, dets, meas[pf], K, T, capture_no + 1, held=hints.get("held"),
                                   require=os.environ.get("NERO_V335_REQUIRE_MEASURED", "1") != "0",
                                   placed=hints.get("placed"))
            in_hand = None
            try:
                if rmask is None:
                    rmask = robot_pixel_mask(K, T, boxes, rgb.shape[:2], stride=2)
                in_hand = inhand[pf].check(rgb, K, T, hints.get("held"), hints.get("heldp"), rmask)
            except Exception as exc_ih:
                in_hand = {"name": hints.get("held"), "visible": False, "reason": f"check failed: {exc_ih}"[:160]}
            # LOG ONLY: error of the reported (measured) positions vs the nearest simulator egg
            meas_err = []
            for n_, o_ in tracked.items():
                if o_.get("seen") and (o_.get("measurement") or {}).get("method") == "multi_view_triangulation" and truth:
                    meas_err.append(min(math.dist(o_["xyz_m"], tv["xyz_m"]) for tv in truth.values()) * 1000.0)
            diag["tracking"] = dict(track_diag)
            capture_no += 1
            elapsed = time.perf_counter() - t0
            cdir = None
            if not hints.get("live") or os.environ.get("NERO_V335_SAVE_LIVE", "0") == "1":
                cdir = writer.write(capture_no, tag, sim_time, rgb, depth, q, dets, vis, rows, fps, summ, diag, K, T, elapsed)
            hold_view = holder or holder_best[pf]
            try:
                live_view_write(Path(out_root) / "live", rgb, K, T, tracked, dets, hold_view, capture_no, tag,
                                extra={"camera": pf, "calib": calib_summary(calib), "calib_level": calib["level"],
                                       "holder_from_this_view": holder is not None, "in_hand": in_hand,
                                       "live": bool(hints.get("live"))})
            except Exception as exc_lv:
                diag["live_view_error"] = str(exc_lv)[:160]
            P8._post_result(res_seq, res_buf, cur, json_safe({
                "seq": cur, "capture": capture_no, "tag": tag, "sim_time_s": round(sim_time, 3),
                "elapsed_s": round(elapsed, 3), "objects": tracked, "n_detections": len(dets),
                "unassigned_detections": track_diag["unassigned_detections"], "dir": cdir.name if cdir else None,
                "camera": f"wrist_{pf}", "cam_pos": [round(float(v), 4) for v in T[:3, 3]],
                "eggs_in_view": sorted(vis),
                "containers": {"egg_holder": holder},
                "calib": {"level": calib["level"], "summary": calib_summary(calib)},
                "measured": {"table_z_m": diag.get("measured_table_z_m"), "table_plane_source": diag.get("table_plane_source"),
                             "holder_top_z_m": diag.get("measured_holder_top_z_m"),
                             "holder_height": hh_meas[pf].est},
                "in_hand": in_hand, "live": bool(hints.get("live")),
                "tracking": {"out_of_view": track_diag.get("out_of_view", []), "held_hint": track_diag.get("held_hint"),
                             "placed_hint": track_diag.get("placed_hint", []), "named": track_diag.get("named", []), "unnamed_trusted": track_diag.get("unnamed_trusted", 0), "awaiting_second_view": track_diag.get("awaiting_second_view", 0),
                             "reassociated_moved": track_diag.get("reassociated_moved", []),
                             "predicted_assoc": track_diag.get("predicted_assoc", [])},
                "holder_partial_fusion": diag.get("holder_partial_fusion")}))
            xy, zz = summ["err_xy_mm"], summ["err_z_mm"]
            if not hints.get("live") or os.environ.get("NERO_V335_LOG_LIVE", "0") == "1":
                print(f"[{VERSION} WRIST PERCEPTION] #{capture_no} {pf} {tag} | in view {len(vis)} | detected "
                      f"{summ['detected']}/{summ['truth']} (+{summ['false_positives']} FP) | trusted {summ['trusted']['detected']} "
                      f"| xy mean {xy['mean'] if xy else '-'} max {xy['max'] if xy else '-'} mm | z mean "
                      f"{zz['mean'] if zz else '-'} mm | measured(triangulated) err mean "
                      f"{(sum(meas_err) / len(meas_err)) if meas_err else float('nan'):.1f} mm (log only) | holder "
                      f"{'complete' if (holder or {}).get('complete') else ('partial' if holder else '-')}"
                      f" | calib level {calib['level']} | cam {[round(float(v), 3) for v in T[:3, 3]]} | {elapsed:.2f}s"
                      f" -> {cdir.name if cdir else 'live (not saved)'}", flush=True)
        except Exception as exc:
            print(f"[{VERSION} WRIST PERCEPTION ERROR] {tag}: {exc}", flush=True)
            traceback.print_exc()
            try:
                P8._post_result(res_seq, res_buf, cur, {"seq": cur, "tag": tag, "error": str(exc)[:300]})
            except Exception:
                pass
    for r_ in renderers.values():
        try:
            r_.close()
        except Exception:
            pass
    status.value = 0


def supervisor(shared, stop_event, out_root):
    """Start the wrist camera process; on an OpenGL backend failure try the next one (as V3.08)."""
    import multiprocessing as mp
    import NERO_V308_Camera_Perception as P8
    for gl in P8.gl_backends():
        if stop_event.is_set():
            return
        saved = {k: os.environ.get(k) for k in ("MUJOCO_GL", "PYOPENGL_PLATFORM")}
        os.environ["MUJOCO_GL"] = gl
        if gl in ("egl", "osmesa"):
            os.environ["PYOPENGL_PLATFORM"] = gl
        else:
            os.environ.pop("PYOPENGL_PLATFORM", None)
        try:
            p = mp.Process(target=wrist_camera_process_main, name="nero_v326_wrist_camera", daemon=True, args=(
                shared["qpos"], shared["meta"], shared["model_path"], shared["trig_seq"], shared["trig_tag"],
                shared["status"], stop_event, out_root, gl, shared["res_seq"], shared["res_buf"]))
            p.start()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        shared["process"] = p
        ever_ready = False
        while p.is_alive():
            p.join(timeout=1.0)
            if int(shared["status"].value) == 1:
                ever_ready = True
        gl_problem = p.exitcode == P8.EXIT_GL_FAILED or (p.exitcode not in (0, None) and not ever_ready)
        if gl_problem and not stop_event.is_set():
            print(f"[{VERSION} WRIST CAMERA] OpenGL '{gl}' failed (exit {p.exitcode}); trying next backend", flush=True)
            continue
        return
    print(f"[{VERSION} WRIST CAMERA] no OpenGL backend worked; wrist perception disabled", flush=True)
