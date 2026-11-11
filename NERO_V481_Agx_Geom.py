#!/usr/bin/env python3
"""NERO V4.8.1 AGX GRIPPER GEOMETRY: the NERO AgxGripper (official AgileX two-finger parallel gripper) for the real
cup / egg planners - grasp frame, fingertip depth, collision boxes (housing, fingers, wrist-camera bracket, camera).

Sources: official URDF agx_arm_urdf/nero/urdf/nero_with_gripper_description.xacro + agx_gripper meshes (measured
here), the user's bracket STEP (01_NERO_USB_V7_1, mounted on the front-top of the housing, photo 2026-11-10).

Frames (all metres):
  link7: tool axis = link7 +x through (·, 0, -23.5 mm); flange face at link7 x = 31 mm; gripper_base 6 mm further.
  TCP (planners, URDF tool frame) = link7 x 155.3 mm: x_TCP = link7 -y, y_TCP = link7 -z (the fingers travel along
      link7 z = the closing axis), z_TCP = link7 +x (approach).  This IS the official gripper's tool frame.
  grasp frame G (NERO_V362 OFFICIAL_PROFILE) = TCP + 2 mm along y: the line where the pads close.  x_G = pad width
      (56 mm), y_G = closing, z_G = approach (towards the fingertips).
  gripper_base z = z_G + 118.3 mm (TCP at 155.3 - 31 - 6); fingertips at gripper_base z = 138 mm (z_G = +19.7 mm,
      configurable: config/gripper.json agx.tip_from_flange_mm, default 144 = 6 + 138).
  housing: gripper_base x -42..33, y -82..82, z 0..65 mm  (x_G = gripper x, y_G = gripper y)
  fingers (each): 56 mm wide (x_G), 26.5 mm thick (y_G, outside the opening), 76.5 mm long back from the tip.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

VERSION = "V4.8.1"
MARKER = "V4.8.1 AGX GRIPPER GEOMETRY + COLLISION"
T_L7_TCP = np.array([[0, 0, 1, 0.1553], [-1, 0, 0, 0], [0, -1, 0, -0.0215], [0, 0, 0, 1.0]])
CLAMP_CENTRE_Y_M = 0.002
TCP_BASE_Z_MM = 155.3 - 31.0 - 6.0                 # 118.3: TCP in gripper_base z
HOUSING_MM = ((-42.0, 33.0), (-82.0, 82.0), (0.0, 65.0))
FINGER_W_MM, FINGER_T_MM, FINGER_L_MM = 56.0, 26.5, 76.5
MARGIN_MM = 3.0
CAMERA_HEAD_MM = 30.0
# the egg is gripped by its upper end so the lower part sticks out below the pads into the carton cell
EGG_OFFSET_MM = 24.0                               # egg centre along x_G from the pad centre (overlap 31.5 mm, equator inside)
EGG_HELD_GAP_MM = (30.0, 48.0)                     # a clamped egg reads its width (40-44 mm)
EGG_LONG_AXIS_GAP_MM = 50.0
EGG_OPEN_MM = 60.0
CARTON_TOP_ABOVE_TABLE_M = 0.030                   # the user's carton: 30 mm high
PAD_ABOVE_CARTON_M = 0.002                         # + the 3 mm box margin: fingers >= 5 mm above the carton top


def tip_from_flange_mm() -> float:
    try:
        import NERO_V48_Gripper as GG
        v = GG.load_cfg()["agx"].get("tip_from_flange_mm")
        return float(v) if v not in (None, "") else 144.0
    except Exception:
        return 144.0


def tip_drop_m() -> float:
    """fingertips beyond the grasp origin along the approach axis z_G"""
    return (31.0 + tip_from_flange_mm() - 155.3) / 1000.0


def T_tcp_grasp() -> np.ndarray:
    T = np.eye(4)
    T[1, 3] = CLAMP_CENTRE_Y_M
    return T


def T_l7_grasp() -> np.ndarray:
    return T_L7_TCP @ T_tcp_grasp()


def T_l7_tip() -> np.ndarray:
    """link7 -> fingertip point (centre between the two tips) - the fence / table tool point"""
    S = np.eye(4)
    S[2, 3] = tip_drop_m()
    return T_l7_grasp() @ S


def _zg(base_z_mm):
    return (float(base_z_mm) - TCP_BASE_Z_MM) / 1000.0


def boxes_in_G(opening_mm: float = 60.0, cam_G: Optional[np.ndarray] = None, margin_mm: float = MARGIN_MM):
    """[(name, centre(3), half(3))] axis-aligned in the grasp frame (m)"""
    m = margin_mm / 1000.0
    tip_base = tip_from_flange_mm() - 6.0
    out = []
    (x0, x1), (y0, y1), (z0, z1) = HOUSING_MM
    out.append(("housing", np.array([(x0 + x1) / 2000.0, (y0 + y1) / 2000.0 - CLAMP_CENTRE_Y_M, (_zg(z0) + _zg(z1)) / 2]),
                np.array([(x1 - x0) / 2000.0 + m, (y1 - y0) / 2000.0 + m, (_zg(z1) - _zg(z0)) / 2 + m])))
    g = max(0.0, float(opening_mm)) / 1000.0
    zc = (_zg(tip_base - FINGER_L_MM) + _zg(tip_base)) / 2
    hz = FINGER_L_MM / 2000.0 + m
    for name, s in (("finger_a", -1.0), ("finger_b", 1.0)):
        yc = s * (g / 2 + FINGER_T_MM / 2000.0)
        out.append((name, np.array([0.0, yc, zc]), np.array([FINGER_W_MM / 2000.0 + m, FINGER_T_MM / 2000.0 + m, hz])))
    if cam_G is not None:
        c = np.asarray(cam_G, float)
        # bracket: from the housing front-top (mount plate) to the camera, ArUco plate sideways (both sides: unknown)
        top = x1 / 1000.0 if c[0] >= 0 else x0 / 1000.0
        xa, xb = sorted((top - 0.018 * (1 if c[0] >= 0 else -1), c[0] + 0.025 * (1 if c[0] >= 0 else -1)))
        za, zb = _zg(z1) - 0.002, c[2] + 0.020
        out.append(("bracket", np.array([(xa + xb) / 2, 0.0, (za + zb) / 2]),
                    np.array([(xb - xa) / 2 + m, 0.062 + m, abs(zb - za) / 2 + m])))
        hh = CAMERA_HEAD_MM / 2000.0 + m
        out.append(("camera_head", c.copy(), np.array([hh, hh, hh])))
    return out


class AgxToolModel:
    """same API as NERO_V4613_Tool_Collision.ToolModel: .ok .note .boxes [(name, centre, R, half)] in link7,
    lowest(T_base_l7), clear_of_table(T_base_l7, z_table) + lowest_of(T_base_l7, names)"""

    def __init__(self, opening_mm=60.0, T_flange_cam=None, clearance_mm=8.0, calib_dir=None):
        import NERO_V4613_Tool_Collision as TC
        self.clearance_mm = float(clearance_mm)
        src = None
        if T_flange_cam is None:
            T_flange_cam, src = TC.find_handeye(calib_dir)
        self.handeye_path = src
        T_l7G = T_l7_grasp()
        cam_G = None
        if T_flange_cam is not None:
            cam = np.asarray(T_flange_cam, float)[:3, 3]
            cam_G = T_l7G[:3, :3].T @ (cam - T_l7G[:3, 3])
        self.cam_G = cam_G
        self.boxes = []
        for name, c, h in boxes_in_G(opening_mm, cam_G):
            self.boxes.append((name, T_l7G[:3, :3] @ c + T_l7G[:3, 3], T_l7G[:3, :3].copy(), h))
        self.ok = True
        self.cfg = {"clearance_mm": self.clearance_mm}
        self.note = ("NERO 原装夹爪 + 腕部相机支架已装到 link7 上：指尖在法兰外 "
                     f"{tip_from_flange_mm():.0f} mm" + (f"，相机（腕部手眼）在夹爪坐标 ({cam_G[0] * 1000:+.0f}, "
                                                          f"{cam_G[1] * 1000:+.0f}, {cam_G[2] * 1000:+.0f}) mm"
                                                          if cam_G is not None else "，没有腕部手眼：不含支架")
                     + f"；规划时离桌面至少 {self.clearance_mm:.0f} mm")

    def corners_l7(self, names=None):
        sg = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        return [(n, c + (sg * h) @ R.T) for n, c, R, h in self.boxes if names is None or n in names]

    def lowest_of(self, T_base_l7, names=None) -> Tuple[float, Optional[str]]:
        T = np.asarray(T_base_l7, float)
        best = (1e9, None)
        for name, P in self.corners_l7(names):
            z = float(np.min(P @ T[:3, :3].T[:, 2] + T[2, 3]))
            if z < best[0]:
                best = (z, name)
        return best

    def lowest(self, T_base_l7):
        return self.lowest_of(T_base_l7)

    BODY = ("housing", "bracket", "camera_head")

    def clear_of_table(self, T_base_l7, z_table):
        """housing / bracket / camera >= clearance above the table (the fingertips have their own 5 mm rule, as SO-101)"""
        z, name = self.lowest_of(T_base_l7, self.BODY)
        dz = (z - float(z_table)) * 1000.0
        return dz >= self.clearance_mm, round(dz, 1), name


PART_CN = {"housing": "夹爪本体", "finger_a": "手指", "finger_b": "手指", "bracket": "相机支架", "camera_head": "腕部相机"}


def camera_side(cam_G=None) -> float:
    """+1 / -1: the side of the gripper (along x_G) where the wrist camera bracket sits (hand-eye; default +1 =
    the user's mount, 2026-11-10: camera at x_G = +54 mm)"""
    if cam_G is None:
        try:
            cam_G = AgxToolModel(opening_mm=44.0).cam_G
        except Exception:
            cam_G = None
    return 1.0 if (cam_G is None or float(cam_G[0]) >= 0.0) else -1.0


def egg_side(cam_G=None) -> float:
    """the egg sticks out on the side OPPOSITE the camera: turned upright the egg points down, the camera up"""
    return -camera_side(cam_G)


def egg_in_G_for_pick(cam_G=None) -> np.ndarray:
    """where the egg centre sits in the grasp frame after an end grasp (z: 0 = pad line; the planner adds the
    measured height difference)"""
    return np.array([egg_side(cam_G) * EGG_OFFSET_MM / 1000.0, 0.0, 0.0])
