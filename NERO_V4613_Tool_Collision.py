#!/usr/bin/env python3
"""NERO V4.6.13 TOOL COLLISION: the user's printed SO-101 <-> NERO adapter (assets/adapter/
SO101_NERO_adapter_v5_PRINT.stl = adapter ring + wrist camera cradle + ArUco plate) on the NERO link7, as oriented
boxes for the planners' table-clearance check.

Why: the place planner only kept the FINGERTIPS above the table; with the egg held vertically the wrist camera and
its cradle hung below the gripper and touched the table (user, 2026-11-09).

STL frame (mm, as printed): z = the adapter axis, z = 0 the face on the NERO seat, the SO-101 boss on top (z 15-20);
the ArUco plate runs to +x (x 23..73, z 5..12), the camera cradle to +y (y 31..67, z 12..32).
Mount in link7: the adapter axis = the SO-101 tool axis (link7 +x, through (0, 0, axis_z)), z = 0 at the seat
(link7 x = seat_x); the turn about the axis comes from the wrist hand-eye calibration (the cradle faces the
calibrated camera centre) or config/tool_collision.json "roll_deg".  The camera module itself sits inside the
cradle -> the cradle box grows by CAMERA_PAD_MM on every side.
"""
from __future__ import annotations

import glob
import json
import math
import os
import struct

import numpy as np

VERSION = "V4.6.13"
MARKER = "V4.6.13 ADAPTER + WRIST CAMERA COLLISION"
HERE = os.path.dirname(os.path.realpath(__file__))
STL_PATH = os.path.join(HERE, "assets", "adapter", "SO101_NERO_adapter_v5_PRINT.stl")
CFG_PATH = os.path.join(HERE, "config", "tool_collision.json")
DEFAULTS = {
    "enabled": True,
    "clearance_mm": 8.0,          # every box corner at least this high above the table
    "margin_mm": 3.0,             # boxes grow by this (print tolerance, model error)
    "camera_pad_mm": 6.0,         # the camera board + lens inside the cradle
    "camera_head_mm": 30.0,       # a cube this size around the calibrated camera centre (lens / board front)
    "roll_deg": None,             # None = from the wrist hand-eye calibration, else this turn about the tool axis
    "default_roll_deg": 180.0,    # no calibration: the cradle points to link7 -z (where the hand-eye put the camera)
    "seat_x_mm": None,            # None = the gripper model's seat (NERO_V364 MOUNT seat_x_m)
}
CRADLE_Y_MM = 25.0                # STL partition: y >= this = camera cradle
PLATE_X_MM = 23.0                 # x >= this (and not cradle) = ArUco plate


def load_cfg(path=None):
    c = dict(DEFAULTS)
    try:
        with open(path or CFG_PATH, encoding="utf-8") as f:
            d = json.load(f)
        c.update({k: v for k, v in d.items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    if os.environ.get("NERO_TOOL_CLEAR_MM"):
        try:
            c["clearance_mm"] = float(os.environ["NERO_TOOL_CLEAR_MM"])
        except ValueError:
            pass
    return c


def load_stl(path=STL_PATH):
    """binary or ASCII STL -> (n, 3, 3) float array (file units, mm)"""
    with open(path, "rb") as f:
        b = f.read()
    if len(b) >= 84:
        n = struct.unpack("<I", b[80:84])[0]
        if 84 + 50 * n == len(b):
            a = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
            return a["v"].reshape(-1, 3, 3).astype(float)
    vs = [list(map(float, ln.split()[1:4])) for ln in b.decode("ascii", "ignore").splitlines()
          if ln.strip().startswith("vertex")]
    return np.asarray(vs, float).reshape(-1, 3, 3)


def stl_parts(tris):
    """-> {name: (min_xyz, max_xyz)} in the STL frame (mm): ring, aruco_plate, camera_cradle"""
    v = np.asarray(tris, float).reshape(-1, 3)
    cradle = v[:, 1] >= CRADLE_Y_MM
    plate = (v[:, 0] >= PLATE_X_MM) & ~cradle
    ring = ~cradle & ~plate
    out = {}
    for name, m in (("ring", ring), ("aruco_plate", plate), ("camera_cradle", cradle)):
        if m.any():
            out[name] = (v[m].min(0), v[m].max(0))
    return out


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def R_l7_stl(roll_deg):
    """STL z -> link7 +x (tool axis), STL x -> link7 +y, STL y -> link7 +z, then the turn about link7 x"""
    base = np.column_stack([[0, 1.0, 0], [0, 0, 1.0], [1.0, 0, 0]])
    return _rx(math.radians(roll_deg)) @ base


def find_handeye(calib_dir=None):
    """the wrist hand-eye result in use: config/wrist_handeye_calib.json, else the newest in the calibration folder"""
    cands = [os.path.join(HERE, "config", "wrist_handeye_calib.json")]
    cd = calib_dir or os.environ.get("NERO_CALIB_DIR") or os.path.expanduser("~/Downloads/cameracalib")
    cands += sorted(glob.glob(os.path.join(cd, "wrist_handeye*.json")), key=os.path.getmtime, reverse=True)
    for p in cands:
        try:
            with open(p, encoding="utf-8") as f:
                d = json.load(f)
            T = np.asarray(d.get("T_flange_cam"), float)
            if T.shape == (4, 4):
                return T, p
        except (OSError, ValueError, TypeError):
            continue
    return None, None


def auto_roll_deg(cam_xyz_l7_m, axis_z_m):
    """turn about the tool axis that points the cradle (STL +y) at the calibrated camera centre"""
    ry, rz = float(cam_xyz_l7_m[1]), float(cam_xyz_l7_m[2]) - float(axis_z_m)
    return math.degrees(math.atan2(-ry, rz))


class ToolModel:
    """adapter + camera boxes in link7 (metres): .boxes = [(name, centre(3), R(3x3), half(3))]"""

    def __init__(self, cfg=None, stl_path=STL_PATH, T_flange_cam=None, calib_dir=None):
        import NERO_V364_SO101_Gripper as G
        self.cfg = load_cfg() if cfg is None else dict(DEFAULTS, **cfg)
        self.ok = False
        self.note = ""
        self.boxes = []
        if not self.cfg.get("enabled", True):
            self.note = "关闭（config/tool_collision.json enabled=false）"
            return
        if not os.path.isfile(stl_path):
            self.note = f"没有转接环模型 {stl_path}"
            return
        tris = load_stl(stl_path)
        parts = stl_parts(tris)
        self.axis_z = float(G.MOUNT["axis_z_m"])
        self.seat_x = (float(self.cfg["seat_x_mm"]) / 1000.0 if self.cfg.get("seat_x_mm") is not None
                       else float(G.MOUNT["seat_x_m"]))
        src_T = None
        if T_flange_cam is None and self.cfg.get("roll_deg") is None:
            T_flange_cam, src_T = find_handeye(calib_dir)
        cam = None if T_flange_cam is None else np.asarray(T_flange_cam, float)[:3, 3]
        if self.cfg.get("roll_deg") is not None:
            roll, how = float(self.cfg["roll_deg"]), "config/tool_collision.json"
        elif cam is not None:
            roll, how = auto_roll_deg(cam, self.axis_z), "腕部手眼标定"
        else:
            roll, how = float(self.cfg["default_roll_deg"]), "默认（没有腕部手眼标定）"
        self.roll_deg, self.roll_src, self.handeye_path = roll, how, src_T
        R = R_l7_stl(roll)
        o = np.array([self.seat_x, 0.0, self.axis_z])
        mg = float(self.cfg["margin_mm"])
        for name, (lo, hi) in parts.items():
            pad = mg + (float(self.cfg["camera_pad_mm"]) if name == "camera_cradle" else 0.0)
            c_stl = 0.5 * (lo + hi)
            half = 0.5 * (hi - lo) + pad
            self.boxes.append((name, o + R @ (c_stl / 1000.0), R, half / 1000.0))
        self.cam_xyz = cam
        self.cam_to_cradle_mm = None
        if cam is not None:
            _n, c, Rb, h = [b for b in self.boxes if b[0] == "camera_cradle"][0]
            loc = Rb.T @ (cam - c)
            out = np.maximum(np.abs(loc) - h, 0.0)
            self.cam_to_cradle_mm = round(float(np.linalg.norm(out)) * 1000.0, 1)
            # the lens / camera head where the hand-eye calibration put it (it may stick out of the cradle)
            hh = (float(self.cfg["camera_head_mm"]) / 2.0 + mg) / 1000.0
            self.boxes.append(("camera_head", cam.copy(), Rb, np.array([hh, hh, hh])))
        self.ok = True
        self.note = (f"转接环 + 相机支架 + ArUco 支架已装到 link7 上：绕工具轴 {roll:+.0f}°（{how}）"
                     + ("" if self.cam_to_cradle_mm is None else
                        f"，标定的相机光心离相机托架 {self.cam_to_cradle_mm:.0f} mm")
                     + f"；规划时离桌面至少 {float(self.cfg['clearance_mm']):.0f} mm")

    def corners_l7(self):
        """-> list of (name, (8, 3) corners in link7)"""
        out = []
        sg = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
        for name, c, R, h in self.boxes:
            out.append((name, c + (sg * h) @ R.T))
        return out

    def lowest(self, T_base_l7):
        """-> (lowest corner z in base_link, part name)"""
        T = np.asarray(T_base_l7, float)
        best = (1e9, None)
        for name, P in self.corners_l7():
            z = float(np.min(P @ T[:3, :3].T[:, 2] + T[2, 3]))
            if z < best[0]:
                best = (z, name)
        return best

    def clear_of_table(self, T_base_l7, z_table):
        """-> (ok, clearance_mm, part name)"""
        z, name = self.lowest(T_base_l7)
        dz = (z - float(z_table)) * 1000.0
        return dz >= float(self.cfg["clearance_mm"]), round(dz, 1), name


PART_CN = {"ring": "转接环", "aruco_plate": "ArUco 支架", "camera_cradle": "腕部相机 / 支架", "camera_head": "腕部相机镜头",
           "housing": "夹爪本体", "finger_a": "手指", "finger_b": "手指", "bracket": "相机支架"}


def make_tool_model(**kw):
    """V4.8.1: the model of the gripper in use - NERO AgxGripper + bracket, or the SO-101 printed adapter"""
    try:
        import NERO_V48_Gripper as GG
        agx = GG.is_agx()
    except Exception:
        agx = False
    if agx:
        import NERO_V481_Agx_Geom as AG
        return AG.AgxToolModel(clearance_mm=load_cfg()["clearance_mm"])
    return ToolModel(**kw)
