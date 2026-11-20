#!/usr/bin/env python3
"""NERO V4.10 HOUSEHOLD PICK & PLACE SCENES - everyday objects for the general pick & place (simulation only).

Objects (assets/household/, built by tools/dev/build_household_assets.py; sources + licences in each meta.json):
  remote       TV remote, 175 x 48 x 20 mm, built from primitives (no GSO / YCB model)          graspable
  apple        YCB 013_apple scan, collision = CoACD convex parts                                 graspable, rolls
  bottle       GSO glass bottle lying on its side, collision = CoACD convex parts                 graspable, rolls
  mug          GSO mug with handle, collision = primitive wall ring + floor + handle              graspable + container
  bowl         GSO bowl, collision = stepped primitive wall ring + floor                          graspable + container
  tray         GSO rectangular tray (75 %), collision = floor + rim wall segments                 container
  storage_box  GSO storage bin (60 %), collision = floor + wall segments                          container
The scan is the visual geom only (contype 0); perception and contact use the collision geoms.

Hold convention (unchanged backend / twin CONNECT hold = ball joint at the body origin): graspable objects have their
origin at the top centre with the mass below it (remote, mug, bowl; mug / bowl: the rim centre), round ones at their
centre (apple, lying bottle).  Containers that are never gripped (tray, storage box) have their origin at the floor
centre.

Scenes: FIXED (H1-H5, hand-placed) or random_scene(seed) (4-5 objects, footprints do not overlap, every graspable
object and every container is reachable top-down).  Each scene carries structured goals (source, relation, target)
for the success check (simulator truth, not the LLM's report).  Scene selection for the GUI simulation:
NERO_SCENE=household_pick_place + NERO_V410_HOUSEHOLD=fixed:H1 | seed:<n>.

Object reset (V4.10): reset_hook(model, data) restores every free object to the pose recorded when the scene had
settled, when the request file NERO_V410_RESET_FILE exists (the console writes it); called from the scene module's
planning_rotated_boxes, which the unchanged backend calls every 2 s inside the simulation loop.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

VERSION = "V4.10"
MARKER = "V4.10 HOUSEHOLD SCENES"
ASSET_DIR = os.path.join(HERE, "assets", "household")
SCENE_NAME = "household_pick_place"
ENV_SPEC = "NERO_V410_HOUSEHOLD"
ENV_RESET = "NERO_V410_RESET_FILE"
X_RANGE = (-0.45, -0.29)          # graspable objects (the official gripper's top-down band, as NERO_V49_Random_Scenes)
X_RANGE_CONT = (-0.50, -0.27)     # containers: only their opening has to be reachable (checked with IK)
Y_RANGE = (-0.21, 0.21)
GAP_M = 0.030                     # free space between footprints (room for the fingers)

GRASPABLE = ("remote", "apple", "bottle", "mug", "bowl", "tall_bottle", "book")
CONTAINERS = ("bowl", "mug", "tray", "storage_box", "mat")
MASS = {"remote": 0.12, "apple": 0.15, "bottle": 0.25, "mug": 0.30, "bowl": 0.30, "tray": 0.40, "storage_box": 0.35,
        "tall_bottle": 0.35, "book": 0.30, "mat": 0.80}
# V4.12 H1 V2 PRIMITIVE OBJECTS (no scan): a standing slim bottle (body / shoulder / neck / cap cylinders), a book (too
# wide for the 70 mm stroke and only 30 mm high) and a thin target mat
TALL_BOTTLE = {"r": 0.029, "body_h": 0.175, "shoulder_h": 0.205, "r_shoulder": 0.020, "r_neck": 0.0135, "h": 0.255,
               "r_cap": 0.0155, "cap_h": 0.018}
BOOK = {"hx": 0.075, "hy": 0.055, "h": 0.030}
MAT = {"hx": 0.090, "hy": 0.090, "h": 0.003, "bevel": 0.008}   # V4.12: bevelled edge (20 deg), a pushed book rides up
PRIMITIVE = ("remote", "tall_bottle", "book", "mat")
ROLL_FRICTION = "1.0 0.02 0.005"      # condim 6 (V4.9.3 egg lesson): rolling objects stop
FRICTION = "1.0 0.02 0.002"
CONTACT = 'solref="0.004 1" solimp="0.95 0.99 0.0005"'
RGBA_COLL = "0.55 0.55 0.55 1"        # collision geoms of scanned objects: opaque (mj_ray skips alpha 0) but in
COLL_GROUP = 3                        # geom group 3, which the viewer / renderers / cameras do not draw by default


def _f(*v):
    return " ".join(f"{float(x):.5f}" for x in v)


def _meta(kind) -> dict:
    cache = _meta.cache = getattr(_meta, "cache", {})
    if kind not in cache:
        p = os.path.join(ASSET_DIR, kind, "meta.json")
        cache[kind] = {}
        if os.path.exists(p):
            with open(p) as fh:
                cache[kind] = json.load(fh)
    return cache[kind]


def _row(kind, z):
    """outer half widths (xmin, xmax, ymin, ymax) of the scan at height z (nearest profile row)"""
    prof = _meta(kind)["profile_z_xmin_xmax_ymin_ymax"]
    r = min(prof, key=lambda t: abs(t[0] - z))
    return r[1], r[2], r[3], r[4]


def dims(kind) -> dict:
    """the sizes the scene / success check use (m); computed from the scans"""
    if kind == "remote":
        return {"hx": 0.0875, "hy": 0.024, "h": 0.020}
    if kind == "tall_bottle":
        return dict(TALL_BOTTLE)
    if kind == "book":
        return dict(BOOK)
    if kind == "mat":
        return dict(MAT, t=0.0, floor=MAT["h"], hx_bot=MAT["hx"], hy_bot=MAT["hy"])
    sx, sy, h = _meta(kind)["size_m"]
    if kind == "apple":
        return {"r": max(sx, sy) / 2.0, "h": h}
    if kind == "bottle":
        return {"hx": sx / 2.0, "r": h / 2.0, "h": h}
    if kind == "mug":
        x0, _x1, y0, y1 = _row("mug", h)
        r = (y1 - y0) / 2.0
        return {"r_top": r, "axis_x": x0 + r, "axis_y": (y0 + y1) / 2.0, "h": h, "t": 0.005, "floor": 0.008,
                "handle_x": _meta("mug")["size_m"][0] / 2.0}
    if kind == "bowl":
        return {"r_top": (sx + sy) / 4.0, "h": h, "t": 0.004, "floor": 0.006}
    if kind in ("tray", "storage_box"):
        x0, x1, y0, y1 = _row(kind, 0.0)
        X0, X1, Y0, Y1 = _row(kind, h)
        t = 0.005 if kind == "tray" else 0.004
        return {"hx_bot": (x1 - x0) / 2.0, "hy_bot": (y1 - y0) / 2.0, "hx": (X1 - X0) / 2.0, "hy": (Y1 - Y0) / 2.0,
                "h": h, "t": t, "floor": t}
    raise ValueError(kind)


def footprint(kind) -> Tuple[float, float, float, float]:
    """(centre x, centre y, half x, half y) of the footprint in the body frame"""
    d = dims(kind)
    if kind == "remote":
        return 0.0, 0.0, d["hx"], d["hy"]
    if kind == "apple":
        return 0.0, 0.0, d["r"], d["r"]
    if kind == "bottle":
        return 0.0, 0.0, d["hx"], d["r"]
    if kind == "tall_bottle":
        return 0.0, 0.0, d["r"], d["r"]
    if kind == "mug":
        sx, sy, _h = _meta("mug")["size_m"]
        return -d["axis_x"], -d["axis_y"], sx / 2.0, sy / 2.0
    if kind == "bowl":
        return 0.0, 0.0, d["r_top"], d["r_top"]
    return 0.0, 0.0, d["hx"], d["hy"]


# ------------------------------------------------------------------------------------------------ MJCF
def assets_xml(kinds, visual=True) -> str:
    """mesh / texture / material assets of these kinds (absolute paths: the backend writes its temp MJCF elsewhere)"""
    out = []
    for k in sorted(set(kinds)):
        if k == "mat":                                            # V4.12: bevelled plate (convex mesh, inline)
            hx, hy, h, bv = MAT["hx"], MAT["hy"], MAT["h"], MAT["bevel"]
            v = [(sx * hx, sy * hy, 0.0) for sx in (-1, 1) for sy in (-1, 1)]
            v += [(sx * (hx - bv), sy * (hy - bv), h) for sx in (-1, 1) for sy in (-1, 1)]
            out.append(f'<mesh name="hh_mat_plate" vertex="{" ".join(f"{c:.5f}" for p_ in v for c in p_)}"/>')
            continue
        if k in PRIMITIVE:
            continue
        d = os.path.join(ASSET_DIR, k)
        if visual:
            out.append(f'<mesh name="hh_{k}_vis" file="{os.path.join(d, "visual.obj")}"/>'
                       f'<texture name="hh_{k}_tex" type="2d" file="{os.path.join(d, "texture.png")}"/>'
                       f'<material name="hh_{k}_mat" texture="hh_{k}_tex" specular="0.2" shininess="0.2"/>')
        for i in range(int(_meta(k).get("hulls", 0))):
            out.append(f'<mesh name="hh_{k}_hull{i}" file="{os.path.join(d, f"hull_{i}.stl")}"/>')
    return "".join(out)


def _box(name, c, s, rgba=RGBA_COLL, yaw=0.0, extra=""):
    q = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    return (f'<geom name="{name}" type="box" pos="{_f(*c)}" size="{_f(*s)}" quat="{_f(*q)}" rgba="{rgba}" '
            f'friction="{FRICTION}" {CONTACT} mass="0" group="{COLL_GROUP}" {extra}/>')


def _ring(name, r_out, z0, z1, t, n=16, cx=0.0, cy=0.0):
    """vertical wall ring of n boxes between z0 and z1 (body frame), outer radius r_out"""
    rm = r_out - t / 2.0
    half_arc = math.pi * rm / n * 1.08
    return "".join(_box(f"{name}_{i}", (cx + rm * math.cos(2 * math.pi * i / n), cy + rm * math.sin(2 * math.pi * i / n),
                                        (z0 + z1) / 2.0), (t / 2.0, half_arc, (z1 - z0) / 2.0),
                        yaw=2 * math.pi * i / n) for i in range(n))


def _walls(name, hx, hy, h, t, z0, seg=0.040):
    """rectangular rim walls (outer half sizes hx, hy) cut into <= 2*seg long pieces (the backend's frozen V2-44 R2
    basket-rim rule thins thin boxes with half length >= 45 mm near the old basket; V4.10.3: seg 0.075 -> 0.040, the
    0.075 pieces of the storage box / tray were thinned to 14 mm in the GUI simulation)"""
    out = []
    for side, (L, fixed, along_x) in {"x0": (hy, -hx + t / 2, False), "x1": (hy, hx - t / 2, False),
                                     "y0": (hx - t, -hy + t / 2, True), "y1": (hx - t, hy - t / 2, True)}.items():
        n = max(1, int(math.ceil(L / seg)))
        for i in range(n):
            a = -L + (2 * i + 1) * L / n
            c = (a, fixed, (z0 + h) / 2.0) if along_x else (fixed, a, (z0 + h) / 2.0)
            s = (L / n, t / 2.0, (h - z0) / 2.0) if along_x else (t / 2.0, L / n, (h - z0) / 2.0)
            out.append(_box(f"{name}_w{side}_{i}", c, s))
    return "".join(out)


def _tiers(name, kind, h, floor, t, use_y_only=False, n=16):
    """stepped wall ring following the scan's outer profile (radius per height band), body origin at the rim"""
    rows = _meta(kind)["profile_z_xmin_xmax_ymin_ymax"]
    tiers = [r for r in rows if r[0] > floor + 0.004]
    out, z_prev = [], -h + floor
    for i, r in enumerate(tiers):
        z_top = 0.0 if i == len(tiers) - 1 else (r[0] + tiers[i + 1][0]) / 2.0 - h
        r_out = (r[4] - r[3]) / 2.0 if use_y_only else (r[2] - r[1] + r[4] - r[3]) / 4.0
        out.append(_ring(f"{name}_wall{i}", r_out, z_prev, z_top, t, n=n))
        z_prev = z_top
    r0 = rows[0]
    r_floor = ((r0[4] - r0[3]) / 2.0 if use_y_only else (r0[2] - r0[1] + r0[4] - r0[3]) / 4.0) + 0.004
    out.append(f'<geom name="{name}_floor" type="cylinder" pos="0 0 {-h + floor / 2:.5f}" size="{_f(r_floor, floor / 2)}" '
               f'rgba="{RGBA_COLL}" friction="{FRICTION}" {CONTACT} mass="0" group="{COLL_GROUP}"/>')
    return "".join(out)


def _visual(kind, name, pos, rgba=None):
    mat = "" if rgba else f'material="hh_{kind}_mat"'
    col = f'rgba="{rgba}"' if rgba else ""
    return (f'<geom name="{name}_vis" type="mesh" mesh="hh_{kind}_vis" pos="{_f(*pos)}" {mat} {col} '
            f'contype="0" conaffinity="0" group="1" mass="0"/>')


def body_xml(kind, oid, x, y, yaw_deg=0.0, visual=True) -> str:
    """one object: <body name="obj_<oid>"> with free joint, collision geoms, inertial and (visual) the scan"""
    name = f"obj_{oid}"
    yaw = math.radians(float(yaw_deg))
    q = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))
    d = dims(kind)
    m = MASS[kind]
    g = []
    if kind == "remote":
        z = d["h"]                                               # origin = top centre
        g.append(f'<geom name="{name}_body" type="box" pos="0 0 {-d["h"] / 2:.5f}" size="{_f(d["hx"], d["hy"], d["h"] / 2)}" '
                 f'rgba="0.12 0.12 0.13 1" friction="{FRICTION}" condim="4" {CONTACT} mass="0"/>')
        if visual:
            for i, (bx, by, col) in enumerate(((0.060, 0.0, "0.75 0.10 0.10 1"), (0.030, -0.010, "0.55 0.55 0.58 1"),
                                               (0.030, 0.010, "0.55 0.55 0.58 1"), (0.000, 0.0, "0.55 0.55 0.58 1"),
                                               (-0.030, -0.010, "0.55 0.55 0.58 1"), (-0.030, 0.010, "0.55 0.55 0.58 1"))):
                g.append(f'<geom name="{name}_btn{i}" type="cylinder" pos="{_f(bx, by, 0.0006)}" size="0.0045 0.0008" '
                         f'rgba="{col}" contype="0" conaffinity="0" group="1" mass="0"/>')
        inertial = f'<inertial pos="0 0 {-0.65 * d["h"]:.5f}" mass="{m}" diaginertia="{_f(1e-4, 4e-4, 4e-4)}"/>'
    elif kind in ("apple", "bottle"):
        z = d["h"] / 2.0                                         # round: origin = centre
        for i in range(int(_meta(kind).get("hulls", 0))):
            g.append(f'<geom name="{name}_hull{i}" type="mesh" mesh="hh_{kind}_hull{i}" pos="0 0 {-z:.5f}" '
                     f'rgba="{RGBA_COLL}" friction="{ROLL_FRICTION}" condim="6" {CONTACT} mass="0" group="{COLL_GROUP}"/>')
        if visual:
            g.append(_visual(kind, name, (0, 0, -z)))
        I = 2e-4 if kind == "apple" else 6e-4
        inertial = f'<inertial pos="0 0 0" mass="{m}" diaginertia="{_f(I * 0.5, I, I)}"/>'
    elif kind == "mug":
        z = d["h"]                                               # origin = rim centre on the body axis
        ax, ay = d["axis_x"], d["axis_y"]
        g.append(_tiers(name, "mug", d["h"], d["floor"], d["t"], use_y_only=True))
        hx0, hx1 = d["r_top"], d["handle_x"] - ax                 # handle along +x of the body frame
        hzb, hzt = -d["h"] + 0.030, -d["h"] + 0.090
        g.append(_box(f"{name}_handle_top", ((hx0 + hx1) / 2, 0, hzt), ((hx1 - hx0) / 2, 0.006, 0.005)))
        g.append(_box(f"{name}_handle_bot", ((hx0 + hx1) / 2, 0, hzb), ((hx1 - hx0) / 2, 0.006, 0.005)))
        g.append(_box(f"{name}_handle_out", (hx1 - 0.005, 0, (hzb + hzt) / 2), (0.005, 0.006, (hzt - hzb) / 2)))
        if visual:
            g.append(_visual(kind, name, (-ax, -ay, -d["h"])))
        inertial = f'<inertial pos="0 0 {-0.62 * d["h"]:.5f}" mass="{m}" diaginertia="{_f(6e-4, 6e-4, 5e-4)}"/>'
    elif kind == "bowl":
        z = d["h"]                                               # origin = rim centre
        g.append(_tiers(name, "bowl", d["h"], d["floor"], d["t"]))
        if visual:
            g.append(_visual(kind, name, (0, 0, -d["h"])))
        inertial = f'<inertial pos="0 0 {-0.7 * d["h"]:.5f}" mass="{m}" diaginertia="{_f(8e-4, 8e-4, 1.4e-3)}"/>'
    elif kind == "tall_bottle":                                  # V4.12: origin = bottom centre, standing
        z = 0.0
        tb = d
        glass = 'rgba="0.20 0.42 0.25 1"'
        segs = ((tb["r"], 0.0, tb["body_h"]), ((tb["r"] + tb["r_shoulder"]) / 2.0, tb["body_h"], (tb["body_h"] + tb["shoulder_h"]) / 2.0),
                (tb["r_shoulder"], (tb["body_h"] + tb["shoulder_h"]) / 2.0, tb["shoulder_h"]),
                (tb["r_neck"], tb["shoulder_h"], tb["h"] - tb["cap_h"]), (tb["r_cap"], tb["h"] - tb["cap_h"], tb["h"]))
        for i, (r_, z0, z1) in enumerate(segs):
            col = 'rgba="0.85 0.80 0.70 1"' if i == 4 else glass
            g.append(f'<geom name="{name}_part{i}" type="cylinder" pos="0 0 {(z0 + z1) / 2:.5f}" size="{_f(r_, (z1 - z0) / 2)}" '
                     f'{col} friction="{FRICTION}" condim="4" {CONTACT} mass="0"/>')
        if visual:
            g.append(f'<geom name="{name}_label" type="cylinder" pos="0 0 {0.55 * tb["body_h"]:.5f}" '
                     f'size="{_f(tb["r"] + 0.0004, 0.035)}" rgba="0.92 0.88 0.75 1" contype="0" conaffinity="0" group="1" mass="0"/>')
        inertial = (f'<inertial pos="0 0 {0.36 * tb["body_h"]:.5f}" mass="{m}" '
                    f'diaginertia="{_f(1.6e-3, 1.6e-3, 1.5e-4)}"/>')
    elif kind == "book":                                         # V4.12: origin = centre
        z = d["h"] / 2.0
        g.append(f'<geom name="{name}_body" type="box" size="{_f(d["hx"], d["hy"], d["h"] / 2)}" rgba="0.62 0.18 0.16 1" '
                 f'friction="{FRICTION}" condim="4" {CONTACT} mass="0"/>')
        if visual:
            g.append(f'<geom name="{name}_pages" type="box" pos="0.0015 0 0" size="{_f(d["hx"] - 0.001, d["hy"] + 0.0004, d["h"] / 2 - 0.003)}" '
                     f'rgba="0.95 0.93 0.86 1" contype="0" conaffinity="0" group="1" mass="0"/>')
        inertial = f'<inertial pos="0 0 0" mass="{m}" diaginertia="{_f(3.3e-4, 5.9e-4, 8.7e-4)}"/>'
    elif kind == "mat":                                          # V4.12: thin target mat, origin = bottom centre
        z = 0.0
        g.append(f'<geom name="{name}_floor" type="mesh" mesh="hh_mat_plate" '
                 f'rgba="0.16 0.33 0.62 1" friction="{FRICTION}" condim="4" {CONTACT} mass="0"/>')
        inertial = f'<inertial pos="0 0 {d["h"] / 2:.5f}" mass="{m}" diaginertia="{_f(2e-3, 2e-3, 4e-3)}"/>'
    else:                                                        # tray / storage box: origin = floor centre
        z = 0.0005
        hx, hy = (d["hx"] + d["hx_bot"]) / 2.0, (d["hy"] + d["hy_bot"]) / 2.0
        g.append(_box(f"{name}_floor", (0, 0, d["floor"] / 2), (d["hx_bot"], d["hy_bot"], d["floor"] / 2)))
        g.append(_walls(name, hx, hy, d["h"], d["t"], d["floor"]))
        if visual:
            g.append(_visual(kind, name, (0, 0, 0)))
        inertial = f'<inertial pos="0 0 {0.3 * d["h"]:.5f}" mass="{m}" diaginertia="{_f(2e-3, 2e-3, 3e-3)}"/>'
    return (f'<body name="{name}" pos="{_f(x, y, z + 0.0005)}" quat="{_f(*q)}"><freejoint/>{inertial}'
            + "".join(g) + "</body>")


# ------------------------------------------------------------------------------------------------ scenes
@dataclass
class Item:
    oid: str
    kind: str
    x: float
    y: float
    yaw_deg: float = 0.0


@dataclass
class HScene:
    sid: str                                  # "H1" .. or "seed:<n>"
    label: str
    items: List[Item]
    goals: List[Tuple[str, str, str]]         # (source oid, "into" / "onto", target oid)
    task: str
    seed: Optional[int] = None
    notes: str = ""

    def kinds(self):
        return [it.kind for it in self.items]

    def hold_names(self):
        return [f"obj_{it.oid}" for it in self.items if it.kind in GRASPABLE]

    def body_names(self):
        return [f"obj_{it.oid}" for it in self.items]

    def world_xml(self, visual=True) -> str:
        return "\n    ".join(body_xml(it.kind, it.oid, it.x, it.y, it.yaw_deg, visual=visual) for it in self.items)

    def assets(self, visual=True) -> str:
        return assets_xml(self.kinds(), visual=visual)

    def summary(self) -> str:
        return ", ".join(it.oid for it in self.items)


def _task_text(goals) -> str:
    parts = [f"put {s} upright on {t}" if r == "upright_on" else f"put {s} {r} {t}" for s, r, t in goals]
    s = "; then ".join(parts)
    return s[0].upper() + s[1:] + "."


def _rel(target_kind):
    return "onto" if target_kind == "tray" else "into"


# fixed scenes: objects + goals + the placement-search seed (positions follow from the seed; gap = free space)
FIXED = {
    "H1": {"label": "H1 基本：苹果放进碗，遥控器放进收纳盒",
           "objects": [("apple_1", "apple"), ("remote_1", "remote"), ("bowl_1", "bowl"), ("storage_box_1", "storage_box")],
           "goals": [("apple_1", "into", "bowl_1"), ("remote_1", "into", "storage_box_1")], "seed": 101, "gap": GAP_M},
    "H2": {"label": "H2 会滚的物品：躺倒的瓶子、苹果放到托盘上",
           "objects": [("bottle_1", "bottle"), ("apple_1", "apple"), ("tray_1", "tray")],
           "goals": [("bottle_1", "onto", "tray_1"), ("apple_1", "onto", "tray_1")], "seed": 202, "gap": GAP_M},
    "H3": {"label": "H3 空心物品：苹果放进马克杯，马克杯放进收纳盒",
           "objects": [("mug_1", "mug"), ("apple_1", "apple"), ("storage_box_1", "storage_box")],
           "goals": [("apple_1", "into", "mug_1"), ("mug_1", "into", "storage_box_1")], "seed": 303, "gap": GAP_M},
    "H4": {"label": "H4 杂乱：5 件物品挨得近，苹果放进碗",
           "objects": [("apple_1", "apple"), ("remote_1", "remote"), ("bottle_1", "bottle"), ("mug_1", "mug"),
                       ("bowl_1", "bowl")],
           "goals": [("apple_1", "into", "bowl_1")], "seed": 404, "gap": 0.012},
    "H5": {"label": "H5 关系：先把马克杯放到托盘上，再把苹果放进马克杯",
           "objects": [("mug_1", "mug"), ("apple_1", "apple"), ("tray_1", "tray")],
           "goals": [("mug_1", "onto", "tray_1"), ("apple_1", "into", "mug_1")], "seed": 505, "gap": GAP_M},
    # V4.12 H1 V2 scenes (side grasp / pushing are possible, nothing in the task says how)
    "H6": {"label": "H6 竖立的细长瓶子（高 255 mm）：放到托盘上并保持竖直",
           "objects": [("tall_bottle_1", "tall_bottle"), ("tray_1", "tray"), ("apple_1", "apple")],
           "goals": [("tall_bottle_1", "upright_on", "tray_1")], "seed": 606, "gap": 0.045,
           # V4.12: placed by hand - the far half of the table, where the arm reaches with the fingers horizontal
           # (checked with IK: grasp points x <= -0.46 m); the random placement uses the top-down band
           "placed": [("tall_bottle_1", "tall_bottle", -0.53, 0.10, 0.0), ("tray_1", "tray", -0.53, -0.15, 0.0),
                      ("apple_1", "apple", -0.36, 0.20, 30.0)]},
    "H7": {"label": "H7 书（150×110×30 mm）比夹爪行程宽：把书移到垫子上",
           "objects": [("book_1", "book"), ("mat_1", "mat"), ("mug_1", "mug")],
           "goals": [("book_1", "onto", "mat_1")], "seed": 707, "gap": 0.045},
}


def inner_size(kind) -> Tuple[float, float]:
    """(long, short) usable inner size of a container opening (m)"""
    d = dims(kind)
    if kind in ("bowl", "mug"):
        return 2 * (d["r_top"] - d["t"]), 2 * (d["r_top"] - d["t"])
    if kind == "mat":                      # V4.12: flat, no rim
        return 2 * MAT["hx"], 2 * MAT["hy"]
    if kind == "tray":                     # shallow sloped rim: an object lies on the floor and over the slope
        a, b = 2 * (d["hx"] - d["t"]), 2 * (d["hy"] - d["t"])
    else:
        a, b = 2 * (min(d["hx"], d["hx_bot"]) - d["t"]), 2 * (min(d["hy"], d["hy_bot"]) - d["t"])
    return max(a, b), min(a, b)


def outer_size(kind) -> Tuple[float, float]:
    """(long, short) footprint of an object (m)"""
    _cx, _cy, hx, hy = footprint(kind)
    if kind == "mug":
        return 2 * hx, 2 * hy
    return 2 * max(hx, hy), 2 * min(hx, hy)


def fits(src_kind, cont_kind, slack=0.010) -> bool:
    """the source's footprint fits the container's opening (both sides, with slack); a fact for the generator only"""
    if src_kind == cont_kind:
        return False
    L, S = inner_size(cont_kind)
    l, s = outer_size(src_kind)
    return l <= L - slack and s <= S - slack


def _corners(it: Item):
    cx, cy, hx, hy = footprint(it.kind)
    a = math.radians(it.yaw_deg)
    c, s = math.cos(a), math.sin(a)
    pts = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        lx, ly = cx + sx * hx, cy + sy * hy
        pts.append((it.x + c * lx - s * ly, it.y + s * lx + c * ly))
    return np.array(pts)


def overlap(a: Item, b: Item, gap=GAP_M) -> bool:
    """separating-axis test of the two footprint rectangles grown by gap / 2 each"""
    A, B = _corners(a), _corners(b)
    for P in (A, B):
        for i in range(4):
            e = P[(i + 1) % 4] - P[i]
            n = np.array([-e[1], e[0]]) / (np.linalg.norm(e) + 1e-12)
            pa, pb = A @ n, B @ n
            if pa.max() + gap / 2 < pb.min() - gap / 2 or pb.max() + gap / 2 < pa.min() - gap / 2:
                return False
    return True


def on_table(it: Item, margin=0.02) -> bool:
    P = _corners(it)
    return bool(P[:, 0].min() >= -0.72 + margin and P[:, 0].max() <= -0.10 and np.abs(P[:, 1]).max() <= 0.38 - margin)


def grasp_z(kind) -> float:
    d = dims(kind)
    return {"remote": 0.012, "apple": 0.036, "bottle": 0.033, "mug": d.get("h", 0.1) - 0.01,
            "bowl": d.get("h", 0.08) - 0.008, "tall_bottle": 0.10, "book": 0.015}.get(kind, 0.05)


def place_z(kind) -> float:
    return {"bowl": 0.15, "mug": 0.20, "tray": 0.10, "storage_box": 0.15, "mat": 0.06}.get(kind, 0.12)


def reachable(it: Item, as_target=False) -> bool:
    import NERO_V49_Random_Scenes as RS
    if as_target:
        return RS._reachable(it.x, it.y, place_z(it.kind), lift=0.04)
    gx, gy = it.x, it.y
    return RS._reachable(gx, gy, grasp_z(it.kind))


def check_layout(items: List[Item], goals, gap=GAP_M) -> List[str]:
    """problems of a layout (empty = fine): overlap, off the table, unreachable sources / targets, source too big"""
    bad = []
    for i, a in enumerate(items):
        if not on_table(a):
            bad.append(f"{a.oid} off the table")
        for b in items[i + 1:]:
            if overlap(a, b, gap):
                bad.append(f"{a.oid} overlaps {b.oid}")
    by = {it.oid: it for it in items}
    for s, _r, t in goals:
        if not reachable(by[s]):
            bad.append(f"{s} not reachable top-down")
        if not reachable(by[t], as_target=True):
            bad.append(f"{t} opening not reachable")
        if not fits(by[s].kind, by[t].kind):
            bad.append(f"{s} does not fit {t}")
    return bad


def _place(objects, rng, gap=GAP_M, tries=80) -> Optional[List[Item]]:
    items = []
    for oid, k in objects:
        xr = X_RANGE_CONT if k in ("tray", "storage_box", "mat") else X_RANGE
        for _t in range(tries):
            it = Item(oid, k, float(rng.uniform(*xr)), float(rng.uniform(*Y_RANGE)),
                      float(rng.choice([0, 90]) if k in ("tray", "storage_box", "mat") else rng.uniform(-180, 180)))
            if on_table(it) and not any(overlap(it, o, gap) for o in items):
                break
        else:
            return None
        items.append(it)
    return items


def _layout_ok(items, goals, gap) -> bool:
    for i, a in enumerate(items):
        for b in items[i + 1:]:
            if overlap(a, b, gap):
                return False
    return not [p for p in check_layout(items, goals, gap=gap) if "overlaps" not in p]


def fixed_scene(sid: str) -> HScene:
    """the fixed scene: the same placement search as random scenes, with the scene's own seed (deterministic)"""
    cache = fixed_scene.cache = getattr(fixed_scene, "cache", {})
    if sid not in cache:
        spec = FIXED[sid]
        rng = np.random.default_rng(int(spec["seed"]))
        if spec.get("placed"):                                    # V4.12: a hand-made layout (no search)
            items = [Item(*t) for t in spec["placed"]]
            bad = [i.oid for i in items if not on_table(i)] + [f"{a.oid}/{b.oid}" for n, a in enumerate(items)
                                                                for b in items[n + 1:] if overlap(a, b, spec["gap"])]
            if bad:
                raise RuntimeError(f"{sid}: placed layout invalid: {bad}")
            cache[sid] = HScene(sid, spec["label"], items, list(spec["goals"]), _task_text(spec["goals"]))
        for _attempt in range(0 if sid in cache else 400):
            items = _place(spec["objects"], rng, spec["gap"])
            if items and _layout_ok(items, spec["goals"], spec["gap"]):
                break
        else:
            if sid not in cache:
                raise RuntimeError(f"{sid}: no valid layout")
        if sid not in cache:
            cache[sid] = HScene(sid, spec["label"], items, list(spec["goals"]), _task_text(spec["goals"]))
    sc = cache[sid]
    return HScene(sc.sid, sc.label, [Item(i.oid, i.kind, i.x, i.y, i.yaw_deg) for i in sc.items], list(sc.goals),
                  sc.task)


def random_scene(seed: int, n_items: Optional[int] = None) -> HScene:
    """4-5 objects: 1-2 containers + graspable objects; 1-2 goals whose source fits the container; reproducible"""
    rng = np.random.default_rng(int(seed))
    for _attempt in range(300):
        n = int(n_items or rng.integers(4, 6))
        n_cont = int(rng.integers(1, 3))
        conts = [str(k) for k in rng.choice(["bowl", "tray", "storage_box", "mug"], size=n_cont, replace=False)]
        srcs = [k for k in ("remote", "apple", "bottle", "mug", "bowl") if k not in conts]
        rest = [str(k) for k in rng.choice(srcs, size=min(len(srcs), n - n_cont), replace=False)]
        objects = [(f"{k}_1", k) for k in conts + rest]
        pairs = [(s, t) for s in rest for t in conts if fits(s, t)]
        if not pairs:
            continue
        rng.shuffle(pairs)
        goals, used = [], set()
        for s, t in pairs:
            if s not in used and len(goals) < int(rng.integers(1, 3)):
                goals.append((f"{s}_1", _rel(t), f"{t}_1"))
                used.add(s)
        items = _place(objects, rng)
        if not items or not _layout_ok(items, goals, GAP_M):
            continue
        return HScene(f"seed:{int(seed)}", f"随机场景 种子 {int(seed)}", items, goals, _task_text(goals), seed=int(seed))
    raise RuntimeError(f"seed {seed}: no valid layout")


def scene_ids() -> List[str]:
    return list(FIXED)


def scene_from_spec(spec: str) -> HScene:
    """'fixed:H1' / 'H1' / 'seed:123'"""
    s = (spec or "").strip()
    if s.startswith("seed:"):
        return random_scene(int(s.split(":", 1)[1]))
    s = s.split(":", 1)[1] if s.startswith("fixed:") else s
    return fixed_scene(s if s in FIXED else "H1")


def scene_from_env() -> HScene:
    return scene_from_spec(os.environ.get(ENV_SPEC, "fixed:H1"))


def describe(scene: HScene) -> str:
    """multi-line text for the console's scene dialog"""
    lines = [scene.label, "物品: " + ", ".join(f"{it.oid} ({it.kind})" for it in scene.items),
             "目标: " + "; ".join(f"{s} {r} {t}" for s, r, t in scene.goals), "任务: " + scene.task]
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ truth check
def _bottom_z(m, d, body) -> float:
    bid = d.body(body).id
    zs = []
    for gi in range(m.ngeom):
        if int(m.geom_bodyid[gi]) != bid or (m.geom_contype[gi] == 0 and m.geom_conaffinity[gi] == 0):
            continue
        R = d.geom_xmat[gi].reshape(3, 3)
        a = m.geom_aabb[gi]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * a[3:] + a[:3]
        zs.append(float(np.min((R @ corners.T).T[:, 2] + d.geom_xpos[gi][2])))
    return min(zs) if zs else float(d.body(body).xpos[2])


def _com(d, body):
    return np.asarray(d.body(body).xipos, float)


def goal_status(m, d, scene: HScene, held=()) -> List[dict]:
    """per goal: ok + measured detail (simulator truth: body poses, collision geometry)"""
    kinds = {it.oid: it.kind for it in scene.items}
    out = []
    for s, rel, t in scene.goals:
        bs, bt = f"obj_{s}", f"obj_{t}"
        if bs in held:
            out.append({"goal": f"{s} {rel} {t}", "ok": False, "detail": f"{s} still in the gripper"})
            continue
        c = _com(d, bs)
        pt = np.asarray(d.body(bt).xpos, float)
        Rt = np.asarray(d.body(bt).xmat, float).reshape(3, 3)
        rl = Rt.T @ (c - pt)
        bot = _bottom_z(m, d, bs)
        k = kinds[t]
        dt = dims(k)
        if k in ("bowl", "mug"):
            okxy = math.hypot(rl[0], rl[1]) <= dt["r_top"] - 0.006
            okz = bot < pt[2] - 0.006
            det = f"{1000 * math.hypot(rl[0], rl[1]):.0f} mm from the {t} axis, bottom {1000 * (bot - pt[2]):+.0f} mm vs rim"
        elif k == "mat":                                          # V4.12: the source's centre over the mat, lying on it
            okxy = abs(rl[0]) <= dt["hx"] and abs(rl[1]) <= dt["hy"]
            okz = abs(bot - (pt[2] + dt["h"])) <= 0.006
            det = (f"rel ({1000 * rl[0]:.0f}, {1000 * rl[1]:.0f}) mm on {t} {2000 * dt['hx']:.0f}x{2000 * dt['hy']:.0f}, "
                   f"bottom {1000 * (bot - pt[2] - dt['h']):+.0f} mm vs its top")
        else:
            okxy = abs(rl[0]) <= dt["hx"] - dt["t"] and abs(rl[1]) <= dt["hy"] - dt["t"]
            okz = bot < pt[2] + dt["h"] + (0.004 if k == "tray" else -0.004) and bot > pt[2] - 0.01
            det = (f"rel ({1000 * rl[0]:.0f}, {1000 * rl[1]:.0f}) mm in {t} {2000 * dt['hx']:.0f}x{2000 * dt['hy']:.0f}, "
                   f"bottom {1000 * (bot - pt[2]):.0f} mm above its floor")
        ok = bool(okxy and okz)
        if rel == "upright_on":                                   # V4.12: and standing (body z axis near vertical)
            R = np.asarray(d.body(bs).xmat, float).reshape(3, 3)
            tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))
            ok = ok and tilt <= 15.0
            det += f", tilt {tilt:.0f} deg (upright <= 15)"
        out.append({"goal": f"{s} {rel} {t}", "ok": ok, "detail": det})
    return out


# ------------------------------------------------------------------------------------------------ twin + probe
def make_twin(scene: HScene, log=None, noise=None, rigid_hold=True):
    """the MuJoCo twin of a household scene; V4.10.1: rigid hold (weld) by default, so a held object turns with the
    gripper as on the real robot (the GUI simulation keeps the backend's ball-joint hold)"""
    import NERO_V489_General_Pick_Place as GP
    if noise is True:
        noise = dict(GP.REAL_LIKE_NOISE, seed=int(scene.seed or 0))
    import NERO_V362_Cup_Skills as CK
    extra = scene.assets(visual=False)
    orig = CK.twin_xml

    def _twin_xml(*a, **k):        # the cup-twin MJCF only lists the cup assets: add this scene's collision meshes
        return orig(*a, **k).replace("<asset>", "<asset>" + extra, 1)
    CK.twin_xml = _twin_xml
    try:
        return GP.TwinGeneralIO(log=log, world_xml=scene.world_xml(visual=False), hold_names=scene.hold_names(),
                                assets=extra, slots=None, noise=noise, rigid_hold=rigid_hold)
    finally:
        CK.twin_xml = orig


def probe_spec():
    """(world_xml, assets, slots) for the general pick & place perception (NERO_V489 SimGeneralIO) in the GUI run"""
    sc = scene_from_env()
    return sc.world_xml(visual=False), sc.assets(visual=False), None


# ------------------------------------------------------------------------------------------------ object reset
_RESET = {"home": None, "t0": None, "done": None}


def reset_file() -> str:
    return os.environ.get(ENV_RESET) or os.path.join(HERE, "logs", "v410_reset_request.json")


def reset_hook(model, data, settle_s=3.0) -> Optional[str]:
    """record the settled object poses once (settle_s after the first call), then restore them whenever the request
    file exists (the file is removed and a <file>.done answer written).  Free objects only; arm / gripper untouched."""
    import mujoco
    st = _RESET
    if st["t0"] is None:
        st["t0"] = time.time()
        return None
    if st["home"] is None:
        if time.time() - st["t0"] < settle_s:
            return None
        home = {}
        for bi in range(model.nbody):
            nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bi) or ""
            j = int(model.body_jntadr[bi])
            if nm.startswith("obj_") and j >= 0 and int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE):
                qa, va = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
                if float(data.qpos[qa + 2]) > -1.0:              # skip the hidden legacy bodies (z = -8 m)
                    home[nm] = (qa, va, data.qpos[qa:qa + 7].copy())
        st["home"] = home
        return None
    try:                                                          # V4.11 DOE: truth check of goals on request
        eval_hook(model, data)
    except Exception as exc:
        print(f"[V4.11 EVAL] failed: {exc}", flush=True)
    try:                                                          # V4.11.3 DOE: safety-stop moments
        safety_hook(model, data)
    except Exception as exc:
        print(f"[V4.11.3 SAFETY RECORD] failed: {exc}", flush=True)
    f = reset_file()
    if not os.path.exists(f):
        return None
    try:
        os.remove(f)
    except OSError:
        return None
    for i in range(model.neq):                                    # release any hold first
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) or ""
        if nm.startswith("hold_"):
            data.eq_active[i] = 0
    for nm, (qa, va, q) in st["home"].items():
        data.qpos[qa:qa + 7] = q
        data.qvel[va:va + 6] = 0.0
    mujoco.mj_forward(model, data)
    msg = f"[V4.10 RESET] {len(st['home'])} objects back to their start poses"
    try:
        with open(f + ".done", "w") as fh:
            json.dump({"t": time.time(), "objects": sorted(st["home"]), "message": msg}, fh)
    except OSError:
        pass
    print(msg, flush=True)
    st["done"] = time.time()
    return msg


# ------------------------------------------------------------------------------------------------ V4.11 DOE truth check
def eval_file() -> str:
    """V4.11 DOE: the goal check request next to the reset request (same directory, same process pair)"""
    f = reset_file()
    return (f[:-5] if f.endswith(".json") else f) + "_eval.json"


def parse_goals(text) -> List[Tuple[str, str, str]]:
    """'apple_1 into bowl_1; remote_1 onto tray_1' (also ',' / newline separated) -> [(src, rel, dst)];
    a bad item raises ValueError with the item"""
    out = []
    for part in [p.strip() for p in str(text or "").replace("\n", ";").replace("，", ";").replace("；", ";")
                 .replace(",", ";").split(";") if p.strip()]:
        w = part.split()
        if len(w) != 3 or w[1] not in ("into", "onto", "upright_on"):
            raise ValueError(f"goal {part!r}: expected '<object> into|onto|upright_on <target>'")
        out.append((w[0], w[1], w[2]))
    return out


def goals_text(goals) -> str:
    return "; ".join(f"{a} {r} {b}" for a, r, b in goals)


def held_bodies(model, data) -> List[str]:
    """bodies held by an active hold_<body> equality right now"""
    import mujoco
    out = []
    for i in range(model.neq):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, i) or ""
        if nm.startswith("hold_") and int(data.eq_active[i]):
            out.append(nm[5:])
    return out


def evaluate(model, data, scene: HScene, goals=None) -> dict:
    """V4.11 DOE: simulator truth after a task: per goal ok + detail, objects held / off the table / tilted"""
    sc = scene if goals is None else HScene(scene.sid, scene.label, scene.items, list(goals), scene.task, scene.seed)
    held = held_bodies(model, data)
    known = {it.oid for it in sc.items}
    bad = [g for g in sc.goals if g[0] not in known or g[2] not in known]
    res = goal_status(model, data, HScene(sc.sid, sc.label, sc.items, [g for g in sc.goals if g not in bad], sc.task,
                                          sc.seed), held=held)
    res += [{"goal": f"{a} {r} {b}", "ok": False, "detail": "object id not in this scene"} for a, r, b in bad]
    objs = {}
    for it in sc.items:
        b = f"obj_{it.oid}"
        try:
            p = np.asarray(data.body(b).xpos, float)
            R = np.asarray(data.body(b).xmat, float).reshape(3, 3)
        except Exception:
            continue
        objs[it.oid] = {"xyz_m": [round(float(v), 4) for v in p], "z_axis_tilt_deg": round(float(np.degrees(
            np.arccos(np.clip(R[2, 2], -1.0, 1.0)))), 1), "off_table": bool(p[2] < -0.03)}
    n_ok = sum(1 for g in res if g["ok"])
    status = "success" if res and n_ok == len(res) else ("partial" if n_ok else "fail")
    return {"status": status, "goals": res, "n_ok": n_ok, "n_goals": len(res), "held": held, "objects": objs,
            "scene": sc.sid}


def eval_hook(model, data):
    """physics side: answer an eval request file (<eval file>.done = evaluate(...)); scene from the environment"""
    f = eval_file()
    if not os.path.exists(f):
        return None
    try:
        with open(f) as fh:
            req = json.load(fh)
        os.remove(f)
    except Exception:
        return None
    goals = req.get("goals")
    out = evaluate(model, data, scene_from_env(), goals=[tuple(g) for g in goals] if goals else None)
    out.update(t=time.time(), id=req.get("id"))
    out.update(snapshot_state(model, data))                       # V4.11.2: for the DOE's final-state pictures
    with open(f + ".done", "w") as fh:
        json.dump(out, fh)
    print(f"[V4.11 EVAL] {req.get('id')}: {out['status']} ({out['n_ok']}/{out['n_goals']})", flush=True)
    return out


_SNAP_MODEL = {"path": None, "why": None}


def snapshot_model_path(model) -> Optional[str]:
    """V4.11.2: the physics process's own model saved once as MJCF next to the generated scene file (so the
    relative mesh / texture paths stay valid); a separate process renders the final state from it + qpos.  None when
    it cannot be saved (the reason is kept in _SNAP_MODEL['why'])."""
    if _SNAP_MODEL["path"] or _SNAP_MODEL["why"]:
        return _SNAP_MODEL["path"]
    try:
        import mujoco
        import NERO_V343_Scenes as SC
        base = SC.LAST_SCENE_XML.get("path")
        if not base:
            raise RuntimeError("the scene file of this process is unknown")
        p = os.path.join(os.path.dirname(base), "_nero_v411_snapshot_model.xml")
        mujoco.mj_saveLastXML(p, model)
        _SNAP_MODEL["path"] = p
    except Exception as exc:
        _SNAP_MODEL["why"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        print(f"[V4.11.2 SNAPSHOT] model not saved: {_SNAP_MODEL['why']}", flush=True)
    return _SNAP_MODEL["path"]


def snapshot_state(model, data) -> dict:
    """V4.11.2: what a renderer needs to show this moment: model file, qpos, simulation time"""
    return {"model_xml": snapshot_model_path(model), "model_why": _SNAP_MODEL["why"], "nq": int(model.nq),
            "qpos": [float(v) for v in data.qpos], "sim_time": float(data.time), **python_env()}


def python_env() -> dict:
    """V4.11.3: the interpreter + package paths of THIS (simulation) process, so the console can render with an
    environment that has mujoco (the console's own Python may not)"""
    paths = []
    for mod in ("mujoco", "numpy"):
        try:
            m = __import__(mod)
            d = os.path.dirname(os.path.dirname(os.path.realpath(m.__file__)))
            if d not in paths:
                paths.append(d)
        except Exception:
            pass
    return {"python": sys.executable, "python_paths": paths}


# ------------------------------------------------------------------------------------------------ V4.11.3 safety moment
def _safety_base() -> str:
    f = reset_file()
    return f[:-5] if f.endswith(".json") else f


def safety_log_file() -> str:
    return _safety_base() + "_safety_log.jsonl"


def request_safety_record(rid, info=None) -> str:
    """GUI side (harness, at a safety stop): ask the physics process to record this moment"""
    p = f"{_safety_base()}_safety_{rid}.json"
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p + ".tmp", "w") as f:
        json.dump({"id": rid, "t": time.time(), "info": info or {}}, f)
    os.replace(p + ".tmp", p)
    return p


def contacts_of_objects(model, data, limit=25) -> List[dict]:
    """contacts that involve a table object right now: the two geoms / bodies, penetration (mm; negative =
    overlap) and the normal force (N), strongest first"""
    import mujoco
    out = []
    f6 = np.zeros(6)
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1])) or ""
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g2])) or ""
        if not (b1.startswith("obj_") or b2.startswith("obj_")):
            continue
        mujoco.mj_contactForce(model, data, i, f6)
        out.append({"body1": b1, "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1) or str(g1),
                    "body2": b2, "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2) or str(g2),
                    "dist_mm": round(1000.0 * float(c.dist), 2), "normal_force_n": round(float(f6[0]), 2)})
    out.sort(key=lambda x: -abs(x["normal_force_n"]))
    return out[:limit]


def safety_hook(model, data):
    """physics side: answer every pending safety-moment request (appended to the safety log)"""
    import glob as _glob
    reqs = sorted(p for p in _glob.glob(f"{_safety_base()}_safety_*.json") if not p.endswith("_log.jsonl"))
    for p in reqs:
        try:
            with open(p) as f:
                req = json.load(f)
            os.remove(p)
        except Exception:
            continue
        held = held_bodies(model, data)
        rec = {"id": req.get("id"), "info": req.get("info"), "t_request": req.get("t"), "t": time.time(),
               "held": held, "contacts": contacts_of_objects(model, data),
               "max_abs_qvel": round(float(np.max(np.abs(data.qvel))) if model.nv else 0.0, 3)}
        rec.update(snapshot_state(model, data))
        with open(safety_log_file(), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[V4.11.3 SAFETY RECORD] {rec['id']}: held={held} contacts={len(rec['contacts'])}", flush=True)


def request_eval(goals=None, rid=None) -> str:
    """console side: ask the running simulation for the truth check (answer: <eval file>.done)"""
    f = eval_file()
    os.makedirs(os.path.dirname(f), exist_ok=True)
    try:
        os.remove(f + ".done")
    except OSError:
        pass
    tmp = f + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"t": time.time(), "id": rid, "goals": [list(g) for g in goals] if goals else None}, fh)
    os.replace(tmp, f)
    return f


def request_reset() -> str:
    """console side: ask the running simulation to put the objects back (answer: <file>.done within ~2-3 s)"""
    f = reset_file()
    os.makedirs(os.path.dirname(f), exist_ok=True)
    try:
        os.remove(f + ".done")
    except OSError:
        pass
    with open(f, "w") as fh:
        json.dump({"t": time.time()}, fh)
    return f
