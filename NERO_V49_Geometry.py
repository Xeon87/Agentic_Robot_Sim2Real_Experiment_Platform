#!/usr/bin/env python3
"""NERO V4.9 GENERAL GEOMETRY: measured scene geometry for the LLM general pick & place (no object classes).

The general pick & place tools give the LLM facts, never strategies.  This module produces those facts:

  FingerModel   the gripper's collision boxes in the grasp frame (pads + palm) for a given opening
                (twin / MuJoCo: the backend's official gripper MJCF; real: NERO_V481_Agx_Geom.boxes_in_G)
  HeightMap     a top-down 2.5-D height map of the table (every cell: highest surface + which object it belongs to);
                a point is inside an object when it is below that object's surface in its cell
                (MuJoCo: ray cast, MjProbe; real fallback: the detected boxes, HeightMap.from_boxes)
  describe()    per object: centre, top, bottom, footprint axes, top-surface height profiles through the centre,
                side profile (outer / inner half widths at several heights, MuJoCo only) and an open cavity summary
  preview_grasp what the fingers would do at a grasp pose: collisions at the final pose (the approach is vertical,
                so in 2.5-D the final pose is the worst point of the descent), where each pad first touches while
                closing, how far the first contact pushes the object, the expected gap when both touch
  cloud_clearance clearance of a point cloud (held object + fingers) above the other objects / table
  relations()   which object is inside / on which (geometry only; the LLM decides what it means for the task)

Nothing here knows what an egg or a cup is; labels only pass through.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

VERSION = "V4.9"
MARKER = "V4.9 GENERAL GEOMETRY"


# ============================================================================================ gripper boxes
class FingerModel:
    """collision boxes of the gripper in the grasp frame G (x = along the jaw width, y = closing direction,
    z = approach, pointing DOWN; origin = the grasp point between the pads).  boxes(open_mm) ->
    [(name, lo(3), hi(3))] in metres; names starting with "pad" are the two contact pads (+y side first)."""

    def __init__(self, name, pad_w_m, pad_t_m, pad_z_m, face_off_m, palm_lo, palm_hi, max_open_mm, real_boxes=None):
        self.name = name
        self.pad_w, self.pad_t = float(pad_w_m), float(pad_t_m)
        self.pad_z = (float(pad_z_m[0]), float(pad_z_m[1]))
        self.face_off = float(face_off_m)
        self.palm_lo, self.palm_hi = np.asarray(palm_lo, float), np.asarray(palm_hi, float)
        self.max_open_mm = float(max_open_mm)
        self._real_boxes = real_boxes

    def face_y(self, open_mm: float) -> float:
        """distance of each pad's inner face from the grasp point"""
        return max(0.0, float(open_mm) / 2000.0 - self.face_off)

    def boxes(self, open_mm: float) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        if self._real_boxes is not None:
            return self._real_boxes(float(open_mm))
        f = self.face_y(open_mm)
        hw = self.pad_w / 2.0
        z0, z1 = self.pad_z
        return [("pad_pos_y", np.array([-hw, f, z0]), np.array([hw, f + self.pad_t, z1])),
                ("pad_neg_y", np.array([-hw, -f - self.pad_t, z0]), np.array([hw, -f, z1])),
                ("palm", self.palm_lo.copy(), self.palm_hi.copy())]

    def llm_spec(self) -> Dict[str, Any]:
        z0, z1 = self.pad_z
        return {
            "type": "parallel_two_finger",
            "frame": "grasp point = midpoint between the two pad faces; closing_yaw_deg = world direction of the "
                     "closing line (pad to pad); pitch_deg turns the gripper about that line (0 = fingers pointing "
                     "down, +-90 = horizontal); the approach runs along the fingers' direction",
            "max_opening_mm": self.max_open_mm,
            "pad_face_gap_mm": f"open_mm - {2000.0 * self.face_off:.0f}",
            "pad_width_mm": round(self.pad_w * 1000.0, 1),
            "pad_thickness_mm": round(self.pad_t * 1000.0, 1),
            "pad_top_above_grasp_point_mm": round(-z0 * 1000.0, 1),
            "fingertip_below_grasp_point_mm": round(z1 * 1000.0, 1),
            "palm_bottom_above_grasp_point_mm": round(-self.palm_hi[2] * 1000.0, 1),
            "palm_size_mm": {"along_closing": round((self.palm_hi[1] - self.palm_lo[1]) * 1000.0, 1),
                             "across_closing": round((self.palm_hi[0] - self.palm_lo[0]) * 1000.0, 1)},
        }


# the backend / twin official gripper MJCF (NERO_V362_Cup_Skills.twin_xml + the backend model): measured in the twin
# (tests/test_v4_9.py G1 re-measures it): pad 29 x 4 x 60 mm centred on the grasp point, inner face = open/2 - 4 mm,
# palm box x +-40, y +-32, 55 .. 120 mm above the grasp point
SIM_FINGERS = FingerModel("mujoco_official", 0.029, 0.004, (-0.030, 0.030), 0.004,
                          (-0.040, -0.032, -0.1203), (0.040, 0.032, -0.0553), 70.0)


def real_fingers(v=None) -> FingerModel:
    """the real AgxGripper (stock / Softfin) from NERO_V481_Agx_Geom.boxes_in_G (no margin); not tested on hardware"""
    import NERO_V481_Agx_Geom as AG

    def _boxes(open_mm):
        out = []
        for name, c, h in AG.boxes_in_G(open_mm, None, margin_mm=0.0, v=v):
            nm = ("pad_pos_y" if c[1] > 0 else "pad_neg_y") if name.startswith("finger") and "root" not in name else name
            out.append((nm, np.asarray(c) - np.asarray(h), np.asarray(c) + np.asarray(h)))
        return out
    pads = [b for b in _boxes(60.0) if b[0].startswith("pad")]
    lo, hi = pads[0][1], pads[0][2]
    house = [b for b in _boxes(60.0) if b[0] == "housing"]
    hlo, hhi = (house[0][1], house[0][2]) if house else (np.array([-0.04, -0.04, -0.2]), np.array([0.04, 0.04, -0.1]))
    return FingerModel("agx_real", hi[0] - lo[0], hi[1] - lo[1], (lo[2], hi[2]), 0.0, hlo, hhi,
                       float(getattr(AG, "STROKE_MM", 70.0)), real_boxes=_boxes)


def grasp_R(yaw_deg: float, pitch_deg: float = 0.0) -> np.ndarray:
    """columns = x_G, y_G (closing direction at yaw), z_G (down), then a pitch about x... same convention as the
    V4.8.9 planner: R_gr = grasp_R(yaw) @ Ry(pitch)"""
    phi = math.radians(float(yaw_deg))
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([math.cos(phi), math.sin(phi), 0.0])
    R = np.column_stack([np.cross(y, z), y, z])
    if abs(float(pitch_deg)) > 1e-6:
        a = math.radians(float(pitch_deg))
        ca, sa = math.cos(a), math.sin(a)
        R = R @ np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
    return R


def box_points(lo, hi, step=0.0025) -> np.ndarray:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    axes = [np.linspace(lo[k], hi[k], max(2, int(math.ceil((hi[k] - lo[k]) / step)) + 1)) for k in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    return g


# ============================================================================================ height map
class HeightMap:
    """H[iy, ix] = highest surface z (table_z where nothing); owner[iy, ix] = object index (-1 = table / nothing)"""

    def __init__(self, x0, y0, res, H, owner, names, table_z):
        self.x0, self.y0, self.res = float(x0), float(y0), float(res)
        self.H = np.asarray(H, float)
        self.owner = np.asarray(owner, int)
        self.names = list(names)
        self.table_z = float(table_z)
        self.ny, self.nx = self.H.shape

    def index_of(self, name: str) -> int:
        return self.names.index(name) if name in self.names else -2

    def at(self, xy: np.ndarray):
        xy = np.asarray(xy, float).reshape(-1, 2)
        ix = np.floor((xy[:, 0] - self.x0) / self.res).astype(int)
        iy = np.floor((xy[:, 1] - self.y0) / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        h = np.full(len(xy), self.table_z)
        o = np.full(len(xy), -1)
        h[ok] = self.H[iy[ok], ix[ok]]
        o[ok] = self.owner[iy[ok], ix[ok]]
        return h, o

    def cells_of(self, name: str) -> np.ndarray:
        k = self.index_of(name)
        iy, ix = np.nonzero(self.owner == k)
        return np.column_stack([self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res, self.H[iy, ix]])

    def without(self, names: Iterable[str]) -> "HeightMap":
        """a copy with these objects removed (their cells fall back to the table)"""
        H, own = self.H.copy(), self.owner.copy()
        for n in names:
            k = self.index_of(n)
            if k >= 0:
                m = own == k
                H[m], own[m] = self.table_z, -1
        return HeightMap(self.x0, self.y0, self.res, H, own, self.names, self.table_z)

    @classmethod
    def from_boxes(cls, objs: Sequence[Dict[str, Any]], table_z: float, bounds=(-0.80, -0.10, -0.45, 0.45),
                   res=0.004) -> "HeightMap":
        """fallback (no ray caster): every object = a solid box of its footprint up to top_z (hollow objects look
        solid - conservative)"""
        x0, x1, y0, y1 = bounds
        nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
        H = np.full((ny, nx), float(table_z))
        own = np.full((ny, nx), -1, int)
        xs = x0 + (np.arange(nx) + 0.5) * res
        ys = y0 + (np.arange(ny) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        names = []
        for o in objs:
            names.append(str(o["object_id"]))
            k = len(names) - 1
            c = o["center_xyz_m"]
            sx, sy = [float(v) / 2000.0 for v in (o.get("size_mm") or [40.0, 40.0])[:2]]
            yaw = math.radians(float(o.get("yaw_deg") or 0.0))
            u = (X - c[0]) * math.cos(yaw) + (Y - c[1]) * math.sin(yaw)
            v = -(X - c[0]) * math.sin(yaw) + (Y - c[1]) * math.cos(yaw)
            m = (np.abs(u) <= sx) & (np.abs(v) <= sy) & (float(o["top_z_m"]) > H)
            H[m], own[m] = float(o["top_z_m"]), k
        return cls(x0, y0, res, H, own, names, table_z)


# ============================================================================================ MuJoCo ray caster
def object_id_for(body_name: str, geom_name: str) -> Optional[str]:
    """object naming of the sim scenes: body obj_<label>_<n> -> "<label>_<n>"; static world geoms
    fix_<label>_<n>_<part> -> "<label>_<n>"; the V4.8.10 static carton general_carton_* -> "egg_carton_1"."""
    b, g = str(body_name or ""), str(geom_name or "")
    if b.startswith("obj_"):
        return b[4:]
    if g.startswith("fix_"):
        parts = g[4:].split("_")
        return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
    if g.startswith("general_carton"):
        return "egg_carton_1"
    return None


def label_of(object_id: str) -> str:
    parts = str(object_id).split("_")
    return "_".join(parts[:-1]) if len(parts) > 1 and parts[-1].isdigit() else str(object_id)


class MjProbe:
    """a static copy of the scene objects (no robot) for ray casting: poses are copied in before every look"""

    TABLE = ('<geom name="play_table" type="box" pos="-0.385 0 -0.03" size="0.335 0.38 0.03"/>')

    def __init__(self, world_xml: str, assets: str = ""):
        import mujoco
        self.mj = mujoco
        xml = (f'<mujoco model="nero_v49_probe"><asset>{assets}</asset><worldbody>{self.TABLE}{world_xml}'
               f'</worldbody></mujoco>')
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.geom_obj: List[Optional[str]] = []
        names = []
        for gi in range(self.m.ngeom):
            gname = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
            bname = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, int(self.m.geom_body_id[gi])
                                      if hasattr(self.m, "geom_body_id") else int(self.m.geom_bodyid[gi])) or ""
            oid = object_id_for(bname, gname)
            if self.m.geom_contype[gi] == 0 and self.m.geom_conaffinity[gi] == 0:
                oid = "__ghost__"                    # ballast / visual-only geoms are not seen
            self.geom_obj.append(oid)
            if oid and oid != "__ghost__" and oid not in names:
                names.append(oid)
        self.names = names
        self.free = {}
        for bi in range(self.m.nbody):
            jadr = int(self.m.body_jntadr[bi])
            if jadr >= 0 and int(self.m.jnt_type[jadr]) == int(mujoco.mjtJoint.mjJNT_FREE):
                self.free[mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, bi)] = int(self.m.jnt_qposadr[jadr])
        self._ghost = np.array([1 if o == "__ghost__" else 0 for o in self.geom_obj], np.uint8)
        mujoco.mj_kinematics(self.m, self.d)

    def set_poses(self, pose_fn) -> None:
        """pose_fn(body_name) -> (xyz, quat_wxyz) or (None, None)"""
        for b, a in self.free.items():
            p, q = pose_fn(b)
            if p is None:
                continue
            self.d.qpos[a:a + 3] = np.asarray(p, float)[:3]
            if q is not None:
                self.d.qpos[a + 3:a + 7] = np.asarray(q, float)[:4]
        self.mj.mj_kinematics(self.m, self.d)

    def _ray(self, p, v, skip_ghost=True):
        gid = np.zeros(1, np.int32)
        p = np.asarray(p, float).copy()
        v = np.asarray(v, float)
        travelled = 0.0
        for _ in range(6):
            dist = self.mj.mj_ray(self.m, self.d, p, v, None, 1, -1, gid)
            if dist < 0:
                return -1.0, -1
            g = int(gid[0])
            if skip_ghost and self.geom_obj[g] == "__ghost__":
                p = p + v * (dist + 1e-4)
                travelled += dist + 1e-4
                continue
            return travelled + dist, g
        return -1.0, -1

    def heightmap(self, bounds=(-0.80, -0.10, -0.45, 0.45), res=0.003, table_z=0.0) -> HeightMap:
        x0, x1, y0, y1 = bounds
        nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
        H = np.full((ny, nx), float(table_z))
        own = np.full((ny, nx), -1, int)
        top = 1.0
        down = np.array([0.0, 0.0, -1.0])
        for iy in range(ny):
            y = y0 + (iy + 0.5) * res
            for ix in range(nx):
                x = x0 + (ix + 0.5) * res
                dist, g = self._ray((x, y, top), down)
                if g < 0:
                    continue
                z = top - dist
                oid = self.geom_obj[g]
                if oid and z > table_z + 1e-4:
                    H[iy, ix] = z
                    own[iy, ix] = self.names.index(oid)
        return HeightMap(x0, y0, res, H, own, self.names, table_z)

    def camera_heightmap(self, cam_pos, bounds=(-0.80, -0.10, -0.45, 0.45), res=0.003, table_z=0.0, rng=None,
                         depth_k=0.0025, dropout=0.02) -> "HeightMap":
        """V4.9.2 SIMULATED DEPTH CAMERA: the height map a fixed RGB-D camera at cam_pos would give.  One ray per
        cell (aimed at the table point of that cell); the first hit lands in the cell under the hit point, with
        depth noise sigma = depth_k * d^2 along the ray and random dropouts.  Small holes inside an object are
        filled from its neighbours; cells nobody saw behind an object (its shadow) keep the table height in the
        object map and get hm.Hu = the camera line grazing the object in front (the most hidden material there can
        reach); unobserved_depth() reports gripper / held-object points below that line (facts, not a refusal)."""
        rng = rng if rng is not None else np.random.default_rng(0)
        x0, x1, y0, y1 = bounds
        nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
        H = np.full((ny, nx), float(table_z))
        own = np.full((ny, nx), -1, int)
        known = np.zeros((ny, nx), bool)
        c = np.asarray(cam_pos, float)
        for iy in range(ny):
            y = y0 + (iy + 0.5) * res
            for ix in range(nx):
                if rng.random() < dropout:
                    continue
                x = x0 + (ix + 0.5) * res
                v = np.array([x, y, table_z]) - c
                L = float(np.linalg.norm(v))
                v /= L
                dist, g = self._ray(c, v)
                if g < 0:
                    continue
                d = dist + float(rng.normal(0.0, depth_k * dist * dist))
                p = c + v * d
                jx, jy = int(math.floor((p[0] - x0) / res)), int(math.floor((p[1] - y0) / res))
                if not (0 <= jx < nx and 0 <= jy < ny):
                    continue
                known[jy, jx] = True
                oid = self.geom_obj[g]
                if oid and oid != "__ghost__" and p[2] > table_z + 1e-4 and p[2] > H[jy, jx]:
                    H[jy, jx], own[jy, jx] = p[2], self.names.index(oid)
        # fill small holes from the neighbours (two passes, like a depth hole filter)
        for _ in range(2):
            Hn, On, Kn = H.copy(), own.copy(), known.copy()
            for iy, ix in zip(*np.nonzero(~known[1:-1, 1:-1])):
                    iy, ix = iy + 1, ix + 1
                    o = own[iy - 1:iy + 2, ix - 1:ix + 2]
                    k = known[iy - 1:iy + 2, ix - 1:ix + 2]
                    if k.sum() >= 5:
                        vals = o[k]
                        top = int(np.bincount(vals + 1).argmax()) - 1
                        Kn[iy, ix] = True
                        if top >= 0:
                            On[iy, ix] = top
                            Hn[iy, ix] = float(np.median(H[iy - 1:iy + 2, ix - 1:ix + 2][k & (o == top)]))
            H, own, known = Hn, On, Kn
        # shadows (nobody saw them): the most anything hidden there can be is the camera line grazing the object
        # in front of it -> Hu (upper bound of hidden material); the cell itself stays "table" for the object map
        Hu = np.full((ny, nx), -1.0)
        cxy, zc = c[:2], float(c[2])
        for iy, ix in zip(*np.nonzero(~known)):
            p = np.array([x0 + (ix + 0.5) * res, y0 + (iy + 0.5) * res])
            Dp = float(np.linalg.norm(cxy - p))
            u = (cxy - p) / max(Dp, 1e-9)
            bound = None
            for s_ in np.arange(res, 0.30, res):
                q = p + u * s_
                jx, jy = int(math.floor((q[0] - x0) / res)), int(math.floor((q[1] - y0) / res))
                if not (0 <= jx < nx and 0 <= jy < ny):
                    break
                if known[jy, jx] and own[jy, jx] >= 0:
                    Dq = Dp - s_
                    z_line = zc - (zc - float(H[jy, jx])) * Dp / max(Dq, 1e-6)
                    bound = z_line if bound is None else max(bound, z_line)
            if bound is not None and bound > table_z:
                Hu[iy, ix] = bound
        names = list(self.names)
        hm = HeightMap(x0, y0, res, H, own, names, table_z)
        hm.known = known
        hm.Hu = Hu
        hm.noisy = True
        return hm

    def bottom_z(self, oid: str) -> Optional[float]:
        zs = []
        for gi, o in enumerate(self.geom_obj):
            if o != oid:
                continue
            R = self.d.geom_xmat[gi].reshape(3, 3)
            c = self.d.geom_xpos[gi]
            aabb = self.m.geom_aabb[gi]
            ctr, half = aabb[:3], aabb[3:]
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half + ctr
            zs.append(float(np.min((R @ corners.T).T[:, 2] + c[2])))
        return min(zs) if zs else None

    def side_profile(self, oid: str, center_xy, axis_deg: float, z_levels: Sequence[float],
                     reach_m: float, floor_z: Optional[float]) -> List[Dict[str, Any]]:
        """outer half width (ray from outside toward the axis) and, above the cavity floor, inner half width
        (ray from the axis outward) along +a, -a, +b, -b at each height"""
        out = []
        a = math.radians(axis_deg)
        dirs = {"+a": (math.cos(a), math.sin(a)), "-a": (-math.cos(a), -math.sin(a)),
                "+b": (-math.sin(a), math.cos(a)), "-b": (math.sin(a), -math.cos(a))}
        cx, cy = float(center_xy[0]), float(center_xy[1])
        for z in z_levels:
            row = {"z_above_table_mm": None, "outer_half_width_mm": {}, "inner_half_width_mm": {}}
            for k, (ux, uy) in dirs.items():
                start = np.array([cx + ux * reach_m, cy + uy * reach_m, z])
                v = np.array([-ux, -uy, 0.0])
                hw = None
                p = start.copy()
                done = 0.0
                for _ in range(6):
                    dist, g = self._ray(p, v)
                    if g < 0 or done + dist > reach_m + 1e-4:
                        break
                    if self.geom_obj[g] == oid:
                        hw = reach_m - (done + dist)
                        break
                    p = p + v * (dist + 0.002)
                    done += dist + 0.002
                row["outer_half_width_mm"][k] = None if hw is None else round(float(hw) * 1000.0, 1)
                inner = None
                if floor_z is not None and z > floor_z + 0.003:
                    dist, g = self._ray(np.array([cx, cy, z]), np.array([ux, uy, 0.0]))
                    if g >= 0 and self.geom_obj[g] == oid and (hw is None or dist < hw - 0.0005):
                        inner = dist
                row["inner_half_width_mm"][k] = None if inner is None else round(float(inner) * 1000.0, 1)
            out.append(row)
        return out


# ============================================================================================ object descriptors
def describe(hm: HeightMap, oid: str, bottom_z: Optional[float] = None, side_fn=None,
             profile_step_m: float = 0.005) -> Optional[Dict[str, Any]]:
    """measured geometry of one object from the height map (+ optional side profile)"""
    cells = hm.cells_of(oid)
    if len(cells) < 3:
        return None
    tz = hm.table_z
    xy = cells[:, :2]
    mu = xy.mean(axis=0)
    # V4.9.2 MIN-AREA RECTANGLE: the footprint axes = the rectangle of least area around the cells (PCA is
    # undefined for square / round footprints and could put a square block's axes on its diagonal)
    rel = xy - mu
    best = None
    for deg in np.arange(0.0, 90.0, 1.0):
        t = math.radians(float(deg))
        u = rel @ np.array([math.cos(t), math.sin(t)])
        v = rel @ np.array([-math.sin(t), math.cos(t)])
        ext_u, ext_v = float(u.max() - u.min()), float(v.max() - v.min())
        area = ext_u * ext_v
        if best is None or area < best[0] - 1e-9:
            best = (area, float(deg), ext_u, ext_v)
    yaw = best[1] if best[2] >= best[3] else best[1] + 90.0   # axis a = the longer side
    yaw = (yaw + 90.0) % 180.0 - 90.0                    # an axis, not a direction
    a = math.radians(yaw)
    ua, ub = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    pa, pb = (xy - mu) @ ua, (xy - mu) @ ub
    half = hm.res / 2.0
    lo_a, hi_a, lo_b, hi_b = pa.min() - half, pa.max() + half, pb.min() - half, pb.max() + half
    ctr = mu + ua * (lo_a + hi_a) / 2.0 + ub * (lo_b + hi_b) / 2.0
    ext_a, ext_b = float(hi_a - lo_a) / 2.0, float(hi_b - lo_b) / 2.0
    # V4.9.2: with a noisy depth map the maximum is biased upward by the noise -> a high percentile
    top = float(np.percentile(cells[:, 2], 97.0)) if getattr(hm, "noisy", False) and len(cells) > 40 \
        else float(cells[:, 2].max())
    bot = float(bottom_z) if bottom_z is not None else tz
    k = hm.index_of(oid)

    step = min(float(profile_step_m), hm.res)

    def line(u, ext):
        offs = np.arange(-(ext + 0.010), ext + 0.010 + 1e-9, step)
        pts = ctr[None, :] + offs[:, None] * u[None, :]
        h, o = hm.at(pts)
        return [(float(dd), (float(hh - tz) if oo == k else None)) for dd, hh, oo in zip(offs, h, o)]

    def segments(prof, tol=0.002):
        """run-length form: [from_mm, to_mm, height_above_table_mm or null]"""
        out = []
        for dd, hh in prof:
            if out and ((hh is None and out[-1][2] is None) or
                        (hh is not None and out[-1][2] is not None and abs(hh - out[-1][3]) <= tol)):
                out[-1][1] = dd
                continue
            out.append([dd, dd, hh, hh])
        return [[round(x0 * 1000.0, 1), round(x1 * 1000.0 + step * 1000.0, 1),
                 (None if h0 is None else round(h0 * 1000.0, 1))] for x0, x1, h0, _h in out]

    prof_a, prof_b = line(ua, ext_a), line(ub, ext_b)
    hc, oc = hm.at(ctr[None, :])
    center_h = float(hc[0]) if int(oc[0]) == k else tz
    cavity = None
    if center_h < top - 0.008:
        def first_high(prof, sign):
            pts = sorted([pp for pp in prof if pp[0] * sign >= 0], key=lambda pp: abs(pp[0]))
            hs = [hh for _dd, hh in pts if hh is not None]
            if not hs or max(hs) + tz < center_h + 0.008:
                return None
            rim = max(hs)
            for dd, hh in pts:
                if hh is not None and hh >= rim - 0.004:
                    return round(abs(dd) * 1000.0, 1)
            return None
        ha = [first_high(prof_a, 1), first_high(prof_a, -1)]
        hb = [first_high(prof_b, 1), first_high(prof_b, -1)]
        if None not in ha + hb:
            cavity = {"floor_z_above_table_mm": round((center_h - tz) * 1000.0, 1),
                      "rim_z_above_table_mm": round((top - tz) * 1000.0, 1),
                      "depth_mm": round((top - center_h) * 1000.0, 1),
                      "opening_half_width_mm": {"+a": ha[0], "-a": ha[1], "+b": hb[0], "-b": hb[1]},
                      "note": "measured from above: the surface inside the footprint is lower than its rim"}
    d = {
        "object_id": oid,
        "label": label_of(oid),
        "center_xyz_m": [round(float(ctr[0]), 4), round(float(ctr[1]), 4), round((top + bot) / 2.0, 4)],
        "top_z_m": round(top, 4),
        "bottom_z_m": round(bot, 4),
        "height_mm": round((top - bot) * 1000.0, 1),
        "axis_a_yaw_deg": round(yaw, 1),
        "size_mm": [round(float(ext_a) * 2000.0, 1), round(float(ext_b) * 2000.0, 1)],
        "size_axes": "[along axis a, along axis b]; axis b = axis a + 90 deg",
        "top_surface_segments_mm": {
            "along_a": segments(prof_a), "along_b": segments(prof_b),
            "format": "cross sections through center_xyz seen from above: [from_offset_mm, to_offset_mm, "
                      "surface height above table mm (null = not this object)]"},
        "open_cavity": cavity,
        # V4.10.1 MEASURED OUTLINE (internal, not sent to the LLM): the footprint's boundary cells in the object's
        # (a, b) frame, normalised by its half sizes - the held-object cloud uses this shape instead of a model
        "_outline_ab": _outline_ab(xy, ctr, ua, ub, ext_a, ext_b, hm.res),
        # footprint area / its rectangle (rectangle ~1.0, ellipse ~0.79): the fallback shape when no outline
        # (the rectangle without the half cell each side adds: a true rectangle gives ~1.0, not 0.93)
        "_footprint_fill": round(min(1.0, float(len(cells)) * hm.res * hm.res /
                                     max((2.0 * ext_a - hm.res) * (2.0 * ext_b - hm.res), 1e-9)), 3),
        # kept for the V4.8.x callers
        "yaw_deg": round(yaw, 1),
    }
    if side_fn is not None and top - bot > 0.004:
        n = 6
        levels = [bot + (top - bot) * (i + 0.5) / n for i in range(n)]
        floor_z = center_h if cavity else None
        rows = side_fn(oid, ctr, yaw, levels, max(ext_a, ext_b) + 0.05, floor_z)
        for z, r in zip(levels, rows):
            r["z_above_table_mm"] = round((z - tz) * 1000.0, 1)
            r["z_m"] = round(float(z), 4)
        d["side_profile"] = rows
    return d


def bottoms_in_openings(descs: List[Dict[str, Any]], table_z: float) -> None:
    """V4.9.2 (moved here in V4.13, used by the simulated and the real depth camera): a camera does not see an
    object's bottom; it is the floor of an opening the object stands in, else the table (descs changed in place)"""
    for d in descs:
        for e in descs:
            cav = e.get("open_cavity")
            if e is d or not cav:
                continue
            dxy = np.asarray(d["center_xyz_m"][:2]) - np.asarray(e["center_xyz_m"][:2])
            r_open = min(v for v in cav["opening_half_width_mm"].values()) / 1000.0
            if float(np.linalg.norm(dxy)) < r_open and float(d["top_z_m"]) < float(e["top_z_m"]) + 0.08:
                fz = table_z + float(cav["floor_z_above_table_mm"]) / 1000.0
                if fz < float(d["top_z_m"]) - 0.005:
                    d["bottom_z_m"] = round(fz, 4)
                    d["height_mm"] = round((float(d["top_z_m"]) - fz) * 1000.0, 1)
                    d["center_xyz_m"][2] = round((float(d["top_z_m"]) + fz) / 2.0, 4)


def relations(descs: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    """geometric relations between objects: inside (A's centre within B's measured opening and A's bottom below
    B's rim) / on (A's bottom at B's top within B's footprint)"""
    out = []
    for A in descs:
        for B in descs:
            if A is B:
                continue
            ca, cb = np.asarray(A["center_xyz_m"][:2]), np.asarray(B["center_xyz_m"][:2])
            a = math.radians(float(B.get("axis_a_yaw_deg") or 0.0))
            ua, ub = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
            da, db = float((ca - cb) @ ua), float((ca - cb) @ ub)
            cav = B.get("open_cavity")
            if cav:
                o = cav["opening_half_width_mm"]
                ra = (o["+a"] if da >= 0 else o["-a"]) / 1000.0
                rb = (o["+b"] if db >= 0 else o["-b"]) / 1000.0
                if (da / max(ra, 1e-3)) ** 2 + (db / max(rb, 1e-3)) ** 2 <= 1.0 and \
                        float(A["bottom_z_m"]) < float(B["top_z_m"]) - 0.005 and \
                        float(A["bottom_z_m"]) > float(B["bottom_z_m"]) + 0.002:
                    out.append({"object": A["object_id"], "relation": "inside", "of": B["object_id"]})
                    continue
            sa, sb = [float(v) / 2000.0 for v in B["size_mm"]]
            if not cav and abs(da) <= 0.8 * sa and abs(db) <= 0.8 * sb and \
                    float(A["bottom_z_m"]) < float(B["top_z_m"]) - 0.005 and \
                    float(A["bottom_z_m"]) > float(B["bottom_z_m"]) + 0.002 and float(A["top_z_m"]) > float(B["top_z_m"]):
                # V4.10.1: B's opening is not visible (A sticks out of it): A's bottom is below B's top inside B's
                # outline, which only a hollow B allows
                out.append({"object": A["object_id"], "relation": "inside", "of": B["object_id"],
                            "note": f"{B['object_id']}'s opening is covered; judged from its outline and heights"})
                continue
            if abs(da) <= sa and abs(db) <= sb and \
                    float(B["top_z_m"]) - 0.012 <= float(A["bottom_z_m"]) <= float(B["top_z_m"]) + 0.020:
                out.append({"object": A["object_id"], "relation": "on", "of": B["object_id"]})
    return out


# ============================================================================================ grasp preview
def _pts_world(p, R, lo, hi, step):
    g = box_points(lo, hi, step)
    return g, (R @ g.T).T + np.asarray(p, float)


def unobserved_depth(hm: HeightMap, P: np.ndarray) -> float:
    """V4.9.2: how far (m) the deepest point of P is below the upper bound of what could hide in unseen space
    (0 when the map has no unseen space or nothing is below it)"""
    Hu = getattr(hm, "Hu", None)
    if Hu is None:
        return 0.0
    xy = np.asarray(P, float)[:, :2]
    ix = np.floor((xy[:, 0] - hm.x0) / hm.res).astype(int)
    iy = np.floor((xy[:, 1] - hm.y0) / hm.res).astype(int)
    ok = (ix >= 0) & (ix < hm.nx) & (iy >= 0) & (iy < hm.ny)
    if not ok.any():
        return 0.0
    hu = np.full(len(xy), -1.0)
    hu[ok] = Hu[iy[ok], ix[ok]]
    d = np.where(hu > hm.table_z, hu - np.asarray(P, float)[:, 2], 0.0)
    return float(max(0.0, d.max()))


def penetration(hm: HeightMap, P: np.ndarray, ignore: Sequence[int] = ()) -> Tuple[np.ndarray, np.ndarray]:
    """per point: how far it is below the surface of an object (m, > 0 = inside), and that object index"""
    h, o = hm.at(P[:, :2])
    pen = np.where(o >= 0, h - P[:, 2], -1.0)
    for k in ignore:
        pen[o == k] = -1.0
    return pen, o


def preview_grasp(hm: HeightMap, fm: FingerModel, p_grasp, yaw_deg: float, pitch_deg: float, open_mm: float,
                  target: Optional[str], step_m: float = 0.0025, table_clear_m: float = 0.0,
                  margin_m: float = 0.0005) -> Dict[str, Any]:
    R = grasp_R(yaw_deg, pitch_deg)
    p = np.asarray(p_grasp, float)
    tk = hm.index_of(target) if target else -2
    boxes = fm.boxes(open_mm)
    coll, parts, unseen = [], {}, []
    min_table = 1e9
    for name, lo, hi in boxes:
        g, P = _pts_world(p, R, lo, hi, step_m if name.startswith("pad") else 2 * step_m)
        parts[name] = (g, P)
        pen, o = penetration(hm, P)
        for k in set(o[pen > margin_m].tolist()):
            dmax = float(pen[(o == k)].max())
            coll.append({"part": name, "object": hm.names[k], "depth_mm": round(dmax * 1000.0, 1)})
        min_table = min(min_table, float(P[:, 2].min()) - hm.table_z)
        du = unobserved_depth(hm, P)
        if du > margin_m:
            unseen.append({"part": name, "below_unseen_bound_mm": round(du * 1000.0, 1)})
    table_hit = min_table < table_clear_m - 1e-6
    # closing: each pad slides toward the other until it meets material (static map)
    pads = [(n, parts[n]) for n in parts if n.startswith("pad")]
    face = fm.face_y(open_mm)
    yG = R[:, 1]
    close = {}
    for name, (g, P) in pads:
        sgn = -1.0 if name.endswith("pos_y") else 1.0          # pos_y pad moves toward -y_G
        hit = None
        for s in np.arange(0.0, face + 1e-9, 0.0005):
            Ps = P + (sgn * s) * yG[None, :]
            pen, o = penetration(hm, Ps)
            m = pen > margin_m
            if m.any():
                ks = o[m]
                k = int(np.bincount(ks).argmax())
                zs = Ps[m & (o == k), 2]
                band = float(zs.max() - zs.min()) + step_m if len(zs) else 0.0
                hit = {"contact_after_travel_mm": round(float(s) * 1000.0, 1), "object": hm.names[k],
                       "contact_band_height_mm": round(band * 1000.0, 1)}
                break
        close[name] = hit
    a, b = close.get("pad_pos_y"), close.get("pad_neg_y")
    res: Dict[str, Any] = {
        "grasp_point_xyz_m": [round(float(v), 4) for v in p],
        "closing_yaw_deg": round(float(yaw_deg), 1),
        "pitch_deg": round(float(pitch_deg), 1),
        "open_mm": float(open_mm),
        "collisions_at_grasp_pose": coll,
        "lowest_gripper_point_above_table_mm": round(min_table * 1000.0, 1),
        "touches_table": bool(table_hit),
        "pad_pos_y": a, "pad_neg_y": b,
    }
    if unseen:
        res["enters_unobserved_space"] = {
            "parts": unseen, "note": "these parts go into space the camera could not see (behind another object), "
                                     "below the highest point anything hidden there could reach"}
    if a and b:
        sa, sb = a["contact_after_travel_mm"], b["contact_after_travel_mm"]
        res["first_contact_pushes_object_mm"] = round(abs(sa - sb), 1)
        res["expected_pad_gap_when_both_touch_mm"] = round(2.0 * face * 1000.0 - 2.0 * max(sa, sb), 1)
        res["closes_on"] = "both pads touch material"
        res["target_between_pads"] = bool(target and (a["object"] == target or b["object"] == target))
    elif a or b:
        res["closes_on"] = "only one pad touches material; the other closes on nothing"
        res["target_between_pads"] = bool(target and ((a or b)["object"] == target))
    else:
        res["closes_on"] = "no material between the pads"
        res["target_between_pads"] = False
    others = sorted({c["object"] for c in (a, b) if c and c["object"] != target})
    if others:
        res["pads_touch_other_objects"] = others
    return res


def sweep_collisions(hm: HeightMap, fm: FingerModel, p_grasp, yaw_deg: float, pitch_deg: float, open_mm: float,
                     dist_m: float, step_m: float = 0.005, margin_m: float = 0.0005,
                     may_touch: Sequence[str] = (), approach_dir=None) -> Dict[str, Any]:
    """V4.12 H1 V2 APPROACH SWEEP: the open gripper moving along its own approach axis (z_G) over the last dist_m
    before the grasp point (the grasp pose itself excluded - preview_grasp checks it): the first collision met on the
    way (part, object, depth, distance before the grasp pose) and the lowest point above the table.  A vertical
    approach can only meet what the grasp pose itself meets (the height map is a top surface), a tilted or horizontal
    one can hit the object or a neighbour on the way."""
    R = grasp_R(yaw_deg, pitch_deg)
    a = R[:, 2] if approach_dir is None else np.asarray(approach_dir, float) / max(1e-9, float(np.linalg.norm(approach_dir)))
    p = np.asarray(p_grasp, float)                    # V4.13: approach_dir = H1 V1's vertical approach (any pitch)
    may = {hm.index_of(n) for n in may_touch}
    first, min_table = None, 1e9
    n = max(1, int(math.ceil(float(dist_m) / step_m)))
    for i in range(n, 0, -1):                         # from the start of the approach toward the grasp pose
        s_ = float(dist_m) * i / n
        q = p - s_ * a
        for name, lo, hi in fm.boxes(open_mm):
            _g, P = _pts_world(q, R, lo, hi, 2 * 0.0025)
            min_table = min(min_table, float(P[:, 2].min()) - hm.table_z)
            pen, o = penetration(hm, P)
            hit = (pen > margin_m) & ~np.isin(o, list(may))
            if hit.any() and first is None:
                k = int(np.bincount(o[hit]).argmax())
                first = {"part": name, "object": hm.names[k], "depth_mm": round(float(pen[hit & (o == k)].max()) * 1000.0, 1),
                         "distance_before_grasp_pose_mm": round(s_ * 1000.0, 1)}
    return {"approach_direction_xyz": [round(float(v), 3) for v in a], "first_collision": first,
            "lowest_gripper_point_above_table_mm": round(min_table * 1000.0, 1)}


def stability(hm: HeightMap, oid: str, bottom_z: float) -> Optional[Dict[str, Any]]:
    """V4.12 H1 V2 STABILITY FACTS (estimates, stated as such): centre of mass of the object ASSUMING uniform density
    in the volume between its bottom and the surface seen from above (each height-map cell = a solid column), and the
    tip-over angle about the nearest edge of that footprint (atan(edge distance / centre-of-mass height))"""
    cells = hm.cells_of(oid)
    if len(cells) < 3:
        return None
    h = np.clip(cells[:, 2] - float(bottom_z), 0.0, None)
    if h.sum() <= 1e-9:
        return None
    w = h / h.sum()
    cx, cy = float((cells[:, 0] * w).sum()), float((cells[:, 1] * w).sum())
    cz = float(bottom_z) + float((h * h).sum() / (2.0 * h.sum()))
    xy = cells[:, :2] - np.array([cx, cy])
    best = 1e9                                         # nearest support edge: smallest half extent over directions
    for ang in np.radians(np.arange(0.0, 180.0, 10.0)):
        u = np.array([math.cos(ang), math.sin(ang)])
        proj = xy @ u
        best = min(best, float(proj.max()) + hm.res / 2.0, float(-proj.min()) + hm.res / 2.0)
    hcom = cz - float(bottom_z)
    return {"estimated_center_of_mass_xyz_m": [round(cx, 4), round(cy, 4), round(cz, 4)],
            "estimated_center_of_mass_height_mm": round(hcom * 1000.0, 1),
            "nearest_footprint_edge_from_com_mm": round(best * 1000.0, 1),
            "tip_over_angle_deg": round(math.degrees(math.atan2(best, max(hcom, 1e-6))), 1),
            "height_to_narrowest_width": round(float((cells[:, 2].max() - bottom_z) / max(2 * best, 1e-6)), 2)}


# ============================================================================================ held object model
def _outline_ab(xy, ctr, ua, ub, ext_a, ext_b, res, max_pts=240):
    """boundary cells of a footprint (cells with a missing 4-neighbour), pushed out by half a cell, as (a, b)
    normalised by the half sizes; at most max_pts points (evenly thinned)"""
    ij = np.round((np.asarray(xy) - np.asarray(xy).min(axis=0)) / res).astype(int)
    S = {(int(i), int(j)) for i, j in ij}
    edge = [k for k, (i, j) in enumerate(ij) if not all(((i + di), (j + dj)) in S
                                                       for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
    if not edge:
        return []
    P = np.asarray(xy)[edge] - np.asarray(ctr)[None, :]
    nrm = np.linalg.norm(P, axis=1, keepdims=True)
    P = P * (1.0 + (res / 2.0) / np.maximum(nrm, 1e-6))          # half a cell outward, along the radius
    pa, pb = P @ ua, P @ ub
    # V4.10.1: keep only what sticks out of the ellipse through the half sizes by more than one cell (a raster
    # circle bulges ~2-3 mm at its diagonals; the corners of a rectangle stick out by tens of mm): in metres,
    # radius = max(ellipse radius in that direction, measured radius - one cell)
    r = np.hypot(pa, pb)
    cs, sn = pa / np.maximum(r, 1e-9), pb / np.maximum(r, 1e-9)
    r_e = 1.0 / np.sqrt((cs / max(ext_a, 1e-6)) ** 2 + (sn / max(ext_b, 1e-6)) ** 2)
    r2 = np.maximum(r_e, r - res)
    out = np.column_stack([r2 * cs / max(ext_a, 1e-6), r2 * sn / max(ext_b, 1e-6)])
    if len(out) > max_pts:
        out = out[np.linspace(0, len(out) - 1, max_pts).astype(int)]
    return np.round(out, 3).tolist()


def superellipse_n(fill: float) -> float:
    """V4.10.1: exponent n of the superellipse whose area / rectangle area = fill (2 for fill <= pi/4, up to 12)"""
    def area(n):
        return math.gamma(1.0 + 1.0 / n) ** 2 / math.gamma(1.0 + 2.0 / n)
    if fill <= area(2.0):
        return 2.0
    if fill >= area(12.0):
        return 12.0
    lo, hi = 2.0, 12.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if area(mid) < fill else (lo, mid)
    return (lo + hi) / 2.0


def held_cloud_from_desc(desc: Dict[str, Any], step_m: float = 0.004) -> np.ndarray:
    """surface points of an object (world frame, at its perceived pose) from its descriptor: rings at the side
    profile heights (ellipse through the 4 measured outer half widths) + the bottom disc; without a side profile:
    its box"""
    c = np.asarray(desc["center_xyz_m"], float)
    a = math.radians(float(desc.get("axis_a_yaw_deg") or 0.0))
    ua, ub = np.array([math.cos(a), math.sin(a), 0.0]), np.array([-math.sin(a), math.cos(a), 0.0])
    bot, top = float(desc["bottom_z_m"]), float(desc["top_z_m"])
    pts = []
    rows = desc.get("side_profile") or []
    sa, sb = [float(v) / 2000.0 for v in desc["size_mm"]]
    levels = []
    for r in rows:
        vals = {k: (float(v) / 1000.0 if v is not None else None) for k, v in r["outer_half_width_mm"].items()}
        seen = [v for v in vals.values() if v is not None]
        if seen:                                  # V4.9.2: an unmeasured side = the largest measured one there
            vals = {k: (v if v is not None else max(seen)) for k, v in vals.items()}
        levels.append((float(r["z_m"]), vals))
    boxy = not levels
    # V4.10.1 OUTLINE FROM THE FOOTPRINT FILL: superellipse |x/a|^n + |y/b|^n = 1 whose area fills the rectangle as
    # much as the measured footprint does (n = 2: ellipse, ~0.785; n -> inf: rectangle).  The plain ellipse cut the
    # corners of a 175 x 48 mm remote (turned 41 deg it was 120 mm wide instead of 151 mm and "fitted" a 126 mm box)
    nexp = superellipse_n(float(desc.get("_footprint_fill") or 0.0))
    outline = [] if boxy else list(desc.get("_outline_ab") or [])
    rect = boxy
    if boxy:
        for z in np.linspace(bot, top, 6):
            levels.append((float(z), {"+a": sa, "-a": sa, "+b": sb, "-b": sb}))
    for z, v in levels:
        ra = [v.get("+a") or sa, v.get("-a") or sa]
        rb = [v.get("+b") or sb, v.get("-b") or sb]
        if outline:                                # V4.10.1: the measured outline, scaled to this level's widths
            for u_, w_ in outline:
                vec = (u_ * (ra[0] if u_ >= 0 else ra[1])) * ua + (w_ * (rb[0] if w_ >= 0 else rb[1])) * ub
                pts.append(np.array([c[0], c[1], z]) + vec)
            continue
        # V4.10.1: one point every <= 3 mm of the outline (24 fixed points left ~18 mm gaps on a 175 mm remote,
        # through which a 4 mm wall slipped; the height map has 3 mm cells)
        perim = 2.0 * (max(ra) + max(rb)) * (4.0 if rect else math.pi / 2.0 * 1.3)
        n = max(24, int(math.ceil(perim / min(step_m, 0.003))))
        for i in range(n):
            t = 2 * math.pi * i / n
            ca_, sb_ = math.cos(t), math.sin(t)
            rr_a = ra[0] if ca_ >= 0 else ra[1]
            rr_b = rb[0] if sb_ >= 0 else rb[1]
            if rect:
                s = max(abs(ca_) / max(rr_a, 1e-4), abs(sb_) / max(rr_b, 1e-4))
                vec = (ca_ / s) * ua + (sb_ / s) * ub
            else:
                e = 2.0 / nexp
                vec = (rr_a * math.copysign(abs(ca_) ** e, ca_)) * ua + (rr_b * math.copysign(abs(sb_) ** e, sb_)) * ub
            pts.append(np.array([c[0], c[1], z]) + vec)
    # bottom disc: the lowest measured level's half widths (the footprint seen from above can be wider)
    z_low, v_low = min(levels, key=lambda lv: lv[0])
    ba = max(v_low.get("+a") or sa, v_low.get("-a") or sa)
    bb = max(v_low.get("+b") or sb, v_low.get("-b") or sb)
    if outline:                                    # V4.10.1: the measured outline, shrunk in three rings
        for f in (1.0, 0.66, 0.33):
            for u_, w_ in outline:
                pts.append(np.array([c[0], c[1], bot]) + (f * u_ * ba) * ua + (f * w_ * bb) * ub)
        pts.append(np.array([c[0], c[1], bot]))
        return np.array(pts, float)
    for x in np.arange(-ba, ba + 1e-9, step_m * 2):
        for y in np.arange(-bb, bb + 1e-9, step_m * 2):
            if rect or abs(x / max(ba, 1e-4)) ** nexp + abs(y / max(bb, 1e-4)) ** nexp <= 1.0:
                pts.append(np.array([c[0], c[1], bot]) + x * ua + y * ub)
    return np.array(pts, float)


def cloud_clearance(hm: HeightMap, P: np.ndarray, ignore: Sequence[str] = (), margin_m: float = 0.0005) -> Dict[str, Any]:
    """vertical clearance of points above the height map (objects other than `ignore`, and the table)"""
    ign = [hm.index_of(n) for n in ignore]
    h, o = hm.at(P[:, :2])
    gap = P[:, 2] - h
    mask_obj = o >= 0
    for k in ign:
        mask_obj &= o != k
    out = {"min_clearance_above_table_mm": round(float((P[:, 2] - hm.table_z).min()) * 1000.0, 1)}
    du = unobserved_depth(hm, P)
    if du > margin_m:
        out["below_unobserved_space_bound_mm"] = round(du * 1000.0, 1)
    if mask_obj.any():
        i = int(np.argmin(np.where(mask_obj, gap, 1e9)))
        out["min_clearance_above_other_objects_mm"] = round(float(gap[i]) * 1000.0, 1)
        out["closest_object_below"] = hm.names[int(o[i])]
        pen = gap < margin_m
        hit = sorted({hm.names[int(k)] for k in o[pen & mask_obj]})
        out["penetrates"] = hit
    else:
        out["min_clearance_above_other_objects_mm"] = None
        out["penetrates"] = []
    return out
