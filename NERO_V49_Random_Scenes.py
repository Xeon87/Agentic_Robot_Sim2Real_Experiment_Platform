#!/usr/bin/env python3
"""NERO V4.9 RANDOM PICK & PLACE SCENES (MuJoCo twin): random objects, containers and tasks for testing the general
pick & place tools without any object-specific code.

  sample_scene(seed)  -> Scene: MJCF world (bodies named obj_<label>_<n>), graspable body names, the task sentence,
                         source / target ids and the relation to reach ("into" / "on")
  make_twin(scene)    -> NERO_V489_General_Pick_Place.TwinGeneralIO on that world
  check_success(io, scene) -> (ok, detail) from the simulator truth (body poses + the generator's dimensions)

Objects follow the backend's hold convention (NERO_V343_Scenes): the CONNECT hold is a ball joint at the body
origin, so a graspable object's origin is its top centre with the mass below it (it hangs the way it was gripped);
round objects (egg, ball) keep their centre.  Containers / plates are heavy free bodies (never in the hold list).
Positions are sampled inside the band the official gripper reaches top-down (checked with the same FastArm IK).
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

VERSION = "V4.9"
MARKER = "V4.9 RANDOM PICK PLACE SCENES"

X_RANGE = (-0.45, -0.29)
Y_RANGE = (-0.22, 0.22)
COLORS = {"red": "0.80 0.12 0.10 1", "blue": "0.15 0.30 0.80 1", "green": "0.15 0.62 0.25 1",
          "yellow": "0.92 0.80 0.15 1", "white": "0.92 0.92 0.90 1", "grey": "0.55 0.56 0.58 1"}


def _f(*v):
    return " ".join(f"{float(x):.5f}" for x in v)


@dataclass
class Obj:
    oid: str                       # object id seen by perception ("block_1")
    kind: str                      # block / can / ball / egg / cup / bin / bowl / plate
    xy: Tuple[float, float]
    dims: Dict[str, float]
    graspable: bool
    xml: str = ""
    radius: float = 0.03           # footprint radius for the layout


@dataclass
class Scene:
    seed: int
    objects: List[Obj]
    source: str
    target: str
    relation: str
    task: str
    world_xml: str = ""
    hold_names: List[str] = field(default_factory=list)


# ------------------------------------------------------------------------------------------------ object MJCF
def _block(name, xy, d, rgba, yaw):
    hx, hy, hz = d["hx"], d["hy"], d["hz"]
    q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    return (f'<body name="{name}" pos="{_f(xy[0], xy[1], 2 * hz + 0.0005)}" quat="{_f(*q)}"><freejoint/>'
            f'<geom name="{name}_box" type="box" pos="0 0 {-hz:.5f}" size="{_f(hx, hy, hz)}" mass="{d["mass"]:.4f}" '
            f'rgba="{rgba}" friction="1.0 0.02 0.002" condim="4" solref="0.004 1" solimp="0.95 0.99 0.0005"/>'
            f'<geom name="{name}_ballast" type="sphere" pos="0 0 {-1.6 * hz:.5f}" size="0.004" mass="{0.5 * d["mass"]:.4f}" '
            f'rgba="0 0 0 0" contype="0" conaffinity="0"/></body>')


def _can(name, xy, d, rgba):
    r, hz = d["r"], d["hz"]
    return (f'<body name="{name}" pos="{_f(xy[0], xy[1], 2 * hz + 0.0005)}"><freejoint/>'
            f'<geom name="{name}_body" type="cylinder" pos="0 0 {-hz:.5f}" size="{_f(r, hz)}" mass="{d["mass"]:.4f}" '
            f'rgba="{rgba}" friction="1.0 0.02 0.002" condim="4" solref="0.004 1" solimp="0.95 0.99 0.0005"/>'
            f'<geom name="{name}_ballast" type="sphere" pos="0 0 {-1.6 * hz:.5f}" size="0.004" mass="{0.5 * d["mass"]:.4f}" '
            f'rgba="0 0 0 0" contype="0" conaffinity="0"/></body>')


def _ball(name, xy, d, rgba):
    r = d["r"]
    return (f'<body name="{name}" pos="{_f(xy[0], xy[1], r + 0.0005)}"><freejoint/>'
            f'<geom name="{name}_ball" type="sphere" size="{r:.5f}" mass="{d["mass"]:.4f}" rgba="{rgba}" '
            f'friction="1.0 0.05 0.02" condim="6" solref="0.004 1" solimp="0.95 0.99 0.0005"/></body>')


def _egg(name, xy, d, rgba, yaw):
    q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    return (f'<body name="{name}" pos="{_f(xy[0], xy[1], 0.022)}" quat="{_f(*q)}"><freejoint/>'
            f'<geom name="{name}_shell" type="ellipsoid" size="0.029 0.022 0.022" mass="0.045" rgba="0.90 0.78 0.62 1" '
            f'friction="1.0 0.02 0.005" condim="6" solref="0.006 1" solimp="0.96 0.995 0.0005" margin="0.0003"/>'
            f'<geom name="{name}_ballast" type="sphere" pos="0 0 -0.012" size="0.006" mass="0.015" rgba="0 0 0 0" '
            f'contype="0" conaffinity="0"/></body>')


def _bin(name, xy, d, rgba, yaw):
    ix, iy, h, t = d["ix"], d["iy"], d["h"], 0.004
    q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    parts = [(f"{name}_floor", (0, 0, t / 2), (ix + t, iy + t, t / 2)),
             (f"{name}_wx0", (ix + t / 2, 0, h / 2), (t / 2, iy + t, h / 2)),
             (f"{name}_wx1", (-ix - t / 2, 0, h / 2), (t / 2, iy + t, h / 2)),
             (f"{name}_wy0", (0, iy + t / 2, h / 2), (ix, t / 2, h / 2)),
             (f"{name}_wy1", (0, -iy - t / 2, h / 2), (ix, t / 2, h / 2))]
    g = "".join(f'<geom name="{n}" type="box" pos="{_f(*c)}" size="{_f(*s)}" mass="0.08" rgba="{rgba}" '
                f'friction="1.0 0.02 0.002" solref="0.004 1" solimp="0.95 0.99 0.0005"/>' for n, c, s in parts)
    return f'<body name="{name}" pos="{_f(xy[0], xy[1], 0.0005)}" quat="{_f(*q)}"><freejoint/>{g}</body>'


def _plate(name, xy, d, rgba):
    r, hz = d["r"], d["hz"]
    return (f'<body name="{name}" pos="{_f(xy[0], xy[1], hz + 0.0005)}"><freejoint/>'
            f'<geom name="{name}_disc" type="cylinder" size="{_f(r, hz)}" mass="0.30" rgba="{rgba}" '
            f'friction="1.0 0.02 0.002" solref="0.004 1" solimp="0.95 0.99 0.0005"/></body>')


def _cup_like(name, xy, d):
    import NERO_V343_Scenes as SC
    c = dict(SC.CUP)
    c.update({k: d[k] for k in ("h", "r_top", "r_bot") if k in d})
    if "mass" in d:
        c["mass"] = d["mass"]
    return SC.cup_body_xml(name, xy, cup=c, origin="rim")


# ------------------------------------------------------------------------------------------------ sampling
def _reachable(x, y, z_grasp, lift=0.10) -> bool:
    import NERO_V362_Cup_Skills as CK
    import NERO_V49_Geometry as G
    A = _reachable.arm = getattr(_reachable, "arm", None) or CK.FastArm()
    prof = CK.OFFICIAL_PROFILE
    a1 = math.atan2(y, x)
    a1 = a1 + math.pi if a1 < 0 else a1 - math.pi
    for yaw in (0.0, 45.0, 90.0, 135.0):
        p, R = prof.tcp_from_grasp(np.array([x, y, z_grasp]), G.grasp_R(yaw))
        for sd in ([a1, 0.2, 0, 0.8, 0, 0.4, 0], [a1, 0.55, 0, 1.4, 0, 0.4, 0.9], [a1, 0.2, 0, 0.8, 0, -0.3, 0]):
            r = A.ik([sd], p, R, iters=40)
            if r and r[0] is not None:
                r2 = A.ik([r[0]], p + np.array([0, 0, lift]), R, iters=40)
                if r2 and r2[0] is not None and CK.check_joint_path([r[0], r2[0]], jump_deg=40).get("min_margin_deg", 0) >= 10:
                    return True
    return False


SOURCE_KINDS = ("block", "can", "ball", "egg", "cup")
TARGETS_FOR = {"block": ("bin", "bowl", "plate"), "can": ("bin", "plate"), "ball": ("bin", "bowl", "cup"),
               "egg": ("bin", "bowl", "cup"), "cup": ("cup", "bin")}


def _make_obj(kind, idx, rng, xy, color):
    oid = f"{kind}_{idx}"
    name = f"obj_{oid}"
    yaw = float(rng.uniform(-math.pi, math.pi))
    rgba = COLORS[color]
    if kind == "block":
        d = {"hx": float(rng.uniform(0.016, 0.026)), "hy": float(rng.uniform(0.016, 0.024)),
             "hz": float(rng.uniform(0.016, 0.030)), "mass": float(rng.uniform(0.03, 0.08))}
        return Obj(oid, kind, xy, d, True, _block(name, xy, d, rgba, yaw), math.hypot(d["hx"], d["hy"]))
    if kind == "can":
        d = {"r": float(rng.uniform(0.018, 0.026)), "hz": float(rng.uniform(0.025, 0.045)), "mass": float(rng.uniform(0.04, 0.10))}
        return Obj(oid, kind, xy, d, True, _can(name, xy, d, rgba), d["r"])
    if kind == "ball":
        d = {"r": float(rng.uniform(0.020, 0.026)), "mass": float(rng.uniform(0.02, 0.06))}
        return Obj(oid, kind, xy, d, True, _ball(name, xy, d, rgba), d["r"])
    if kind == "egg":
        d = {"a": 0.029, "b": 0.022}
        return Obj(oid, kind, xy, d, True, _egg(name, xy, d, rgba, yaw), 0.030)
    if kind == "cup":
        d = {"h": 0.121, "r_top": 0.0475, "r_bot": 0.030}
        return Obj(oid, kind, xy, d, True, _cup_like(name, xy, d), 0.05)
    if kind == "bowl":
        d = {"h": float(rng.uniform(0.045, 0.060)), "r_top": float(rng.uniform(0.062, 0.072)), "r_bot": 0.040, "mass": 0.25}
        return Obj(oid, kind, xy, d, False, _cup_like(name, xy, d), d["r_top"] + 0.004)
    if kind == "bin":
        d = {"ix": float(rng.uniform(0.045, 0.062)), "iy": float(rng.uniform(0.045, 0.060)), "h": float(rng.uniform(0.035, 0.055)),
             "yaw": yaw}
        return Obj(oid, kind, xy, d, False, _bin(name, xy, d, rgba, yaw), math.hypot(d["ix"], d["iy"]) + 0.006)
    if kind == "plate":
        d = {"r": float(rng.uniform(0.055, 0.068)), "hz": 0.006}
        return Obj(oid, kind, xy, d, False, _plate(name, xy, d, rgba), d["r"])
    raise ValueError(kind)


def sample_scene(seed: int, n_distractors: Optional[int] = None) -> Scene:
    rng = np.random.default_rng(int(seed))
    src_kind = str(rng.choice(SOURCE_KINDS))
    tgt_kind = str(rng.choice(TARGETS_FOR[src_kind]))
    rel = "on" if tgt_kind == "plate" else "into"
    kinds = [src_kind, tgt_kind]
    nd = int(rng.integers(0, 3)) if n_distractors is None else int(n_distractors)
    for _ in range(nd):
        kinds.append(str(rng.choice(("block", "can", "ball", "egg"))))
    colors = list(COLORS)
    for _attempt in range(400):
        objs, counts = [], {}
        ok = True
        for i, k in enumerate(kinds):
            counts[k] = counts.get(k, 0) + 1
            for _t in range(60):
                xy = (float(rng.uniform(*X_RANGE)), float(rng.uniform(*Y_RANGE)))
                o = _make_obj(k, counts[k], rng, xy, colors[int(rng.integers(0, len(colors)))])
                if all(math.hypot(xy[0] - p.xy[0], xy[1] - p.xy[1]) > o.radius + p.radius + 0.045 for p in objs):
                    break
            else:
                ok = False
                break
            objs.append(o)
        if not ok:
            continue
        src, tgt = objs[0], objs[1]
        z_src = {"block": 2 * src.dims.get("hz", 0.02) - 0.008, "can": 2 * src.dims.get("hz", 0.03) - 0.008,
                 "ball": 0.038, "egg": 0.036, "cup": 0.100}[src.kind]
        z_tgt = {"bin": 0.12, "bowl": 0.12, "plate": 0.09, "cup": 0.20}[tgt.kind]
        if _reachable(src.xy[0], src.xy[1], max(z_src, 0.036)) and _reachable(tgt.xy[0], tgt.xy[1], z_tgt, lift=0.04):
            break
    else:
        raise RuntimeError(f"seed {seed}: no reachable layout")
    phrase = {"into": "into", "on": "onto"}[rel]
    task = f"Put {src.oid} {phrase} {tgt.oid}."
    sc = Scene(int(seed), objs, src.oid, tgt.oid, rel, task)
    sc.world_xml = "\n    ".join(o.xml for o in objs)
    sc.hold_names = [f"obj_{o.oid}" for o in objs if o.graspable]
    return sc


def make_twin(scene: Scene, log=None, noise=None):
    """noise: None = simulator-true geometry; True / dict = V4.9.2 real-like perception (GP.REAL_LIKE_NOISE)"""
    import NERO_V343_Scenes as SC
    import NERO_V489_General_Pick_Place as GP
    if noise is True:
        noise = dict(GP.REAL_LIKE_NOISE, seed=int(scene.seed))
    return GP.TwinGeneralIO(log=log, world_xml=scene.world_xml, hold_names=scene.hold_names, assets=SC.CUP_ASSETS,
                            slots=None, noise=noise)


# ------------------------------------------------------------------------------------------------ truth check
def _body(io, oid):
    b = io.d.body(f"obj_{oid}")
    R = np.asarray(io.d.xmat[b.id]).reshape(3, 3)
    return np.asarray(b.xpos, float), R


def _bottom_z(io, oid) -> float:
    import mujoco
    zs = []
    bid = io.d.body(f"obj_{oid}").id
    for gi in range(io.m.ngeom):
        if int(io.m.geom_bodyid[gi]) != bid or (io.m.geom_contype[gi] == 0 and io.m.geom_conaffinity[gi] == 0):
            continue
        R = io.d.geom_xmat[gi].reshape(3, 3)
        a = io.m.geom_aabb[gi]
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * a[3:] + a[:3]
        zs.append(float(np.min((R @ corners.T).T[:, 2] + io.d.geom_xpos[gi][2])))
    del mujoco
    return min(zs)


def check_success(io, scene: Scene) -> Tuple[bool, str]:
    objs = {o.oid: o for o in scene.objects}
    src, tgt = objs[scene.source], objs[scene.target]
    ps, _ = _body(io, src.oid)
    pt, Rt = _body(io, tgt.oid)
    bot = _bottom_z(io, src.oid)
    held = bool(getattr(io, "hold", None))
    # source centre (its COM region): its body origin is its top centre for block / can; use the geom centre
    c_src = ps.copy()
    if src.kind in ("block", "can"):
        c_src = ps + (np.asarray(io.d.xmat[io.d.body(f'obj_{src.oid}').id]).reshape(3, 3) @ np.array([0, 0, -src.dims["hz"]]))
    rel = Rt.T @ (c_src - pt)
    if held:
        return False, "still in the gripper"
    if scene.relation == "on":
        top = pt[2] + tgt.dims["hz"]
        okz = top - 0.006 <= bot <= top + 0.012
        okxy = math.hypot(rel[0], rel[1]) <= tgt.dims["r"]
        return bool(okz and okxy), f"bottom {1000 * (bot - top):+.1f} mm vs plate top, xy {1000 * math.hypot(rel[0], rel[1]):.0f} mm from centre"
    if tgt.kind == "bin":
        okxy = abs(rel[0]) <= tgt.dims["ix"] and abs(rel[1]) <= tgt.dims["iy"]
        okz = bot < pt[2] + tgt.dims["h"] - 0.004
        return bool(okxy and okz), f"rel xy ({1000 * rel[0]:.0f}, {1000 * rel[1]:.0f}) mm in bin {1000 * tgt.dims['ix']:.0f}x{1000 * tgt.dims['iy']:.0f}, bottom {1000 * (bot - pt[2]):.0f} mm above floor"
    # cup / bowl: origin = rim centre
    r_top = tgt.dims["r_top"]
    okxy = math.hypot(rel[0], rel[1]) <= r_top - 0.004
    okz = bot < pt[2] - 0.004
    return bool(okxy and okz), f"{1000 * math.hypot(rel[0], rel[1]):.0f} mm from the axis, bottom {1000 * (bot - pt[2]):+.0f} mm vs rim"
