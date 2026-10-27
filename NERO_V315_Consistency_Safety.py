#!/usr/bin/env python3
"""NERO V3.15 CONSISTENCY + SAFETY (fence: V3.16 polygon): sim/real consistency tests and workspace safety for the 实机 tab.

Logic (no Tk, no ROS at import):
- forward kinematics from a URDF (pure python 4x4), pose helpers
- static model comparison: sim URDF (config/nero.urdf) vs the driver's /robot_description
- live consistency: sim-URDF FK vs real-URDF FK vs TF for the same measured joint angles
- direction test evaluation (predicted vs measured TCP displacement), repeatability / stop statistics
- workspace config (config/real_workspace.json): collision boxes for MoveIt + a software fence for link/tool points
- fence check for one joint vector and for a planned trajectory (used by the ROS monitor before execution)
- CSV / JSON writers (logs/real_motion_log.csv, logs/v315_*.csv/json)
Tk: ConsistencyTabUI and SafetyTabUI (tkinter passed in by the console).
"""
import csv
import json
import math
import os
import queue
import threading
import time

import NERO_V314_Joint_Control as JC

HERE = os.path.dirname(os.path.realpath(__file__))
SIM_URDF_PATH = os.path.join(HERE, "config", "nero.urdf")
WORKSPACE_PATH = os.path.join(HERE, "config", "real_workspace.json")
LOG_DIR = os.path.join(HERE, "logs")
SCENE_PREFIX = "v315_"
SIM_REAL_TOL_MM = 1.0
URDF_TF_TOL_MM = 1.0
DIRECTION_COS_MIN = 0.95
DIRECTION_MAG_TOL = 0.10


# =========================================================================== kinematics
def _mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def _T(R, t):
    return [[R[0][0], R[0][1], R[0][2], t[0]], [R[1][0], R[1][1], R[1][2], t[1]],
            [R[2][0], R[2][1], R[2][2], t[2]], [0.0, 0.0, 0.0, 1.0]]


def _axis_angle(ax, q):
    n = math.sqrt(sum(a * a for a in ax)) or 1.0
    x, y, z = (a / n for a in ax)
    c, s, C = math.cos(q), math.sin(q), 1 - math.cos(q)
    return [[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C]]


I4 = [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]


def joint_transform(j, q):
    T = _T(_rpy_mat(*j["rpy"]), j["xyz"])
    if j["type"] in ("revolute", "continuous"):
        T = _mat_mul(T, _T(_axis_angle(j["axis"], q), (0, 0, 0)))
    elif j["type"] == "prismatic":
        n = math.sqrt(sum(a * a for a in j["axis"])) or 1.0
        T = _mat_mul(T, _T(_rpy_mat(0, 0, 0), [a / n * q for a in j["axis"]]))
    return T


def link_chain(urdf, base, tip):
    """Joint names (all types) from base to tip; [] if tip is not below base."""
    by_child = {d["child"]: n for n, d in urdf["joints"].items()}
    out, link, guard = [], tip, 0
    while link != base and guard < 200:
        guard += 1
        jn = by_child.get(link)
        if jn is None:
            return None
        out.append(jn)
        link = urdf["joints"][jn]["parent"]
    return list(reversed(out))


def fk(urdf, q_by_name, base, tip):
    """4x4 transform base->tip. q_by_name: {joint: value}; mimic joints follow their source. None if no chain."""
    chain = link_chain(urdf, base, tip)
    if chain is None:
        return None
    T = [row[:] for row in I4]
    for jn in chain:
        j = urdf["joints"][jn]
        q = q_by_name.get(jn, 0.0)
        if j.get("mimic"):
            q = q_by_name.get(j["mimic"], 0.0)
        T = _mat_mul(T, joint_transform(j, float(q)))
    return T


def fk_points(urdf, q_by_name, base, links):
    """{link: (x, y, z)} for the links that exist below base."""
    out = {}
    for ln in links:
        T = fk(urdf, q_by_name, base, ln)
        if T is not None:
            out[ln] = (T[0][3], T[1][3], T[2][3])
    return out


def pose_of(T):
    R = [row[:3] for row in T[:3]]
    sy = -R[2][0]
    pitch = math.asin(max(-1.0, min(1.0, sy)))
    roll = math.atan2(R[2][1], R[2][2])
    yaw = math.atan2(R[1][0], R[0][0])
    return {"xyz": [T[0][3], T[1][3], T[2][3]], "rpy_deg": [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]}


def dist_mm(a, b):
    return 1000.0 * math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def link_offset_transform(urdf, parent, child):
    """Fixed transform parent->child (used for the sim tool point relative to the flange)."""
    return fk(urdf, {}, parent, child)


# =========================================================================== static model comparison
def compare_models(sim, real, joints=None):
    """Joint-by-joint differences between two parsed URDFs. -> {"same": bool, "rows": [...], "missing": [...]}"""
    names = list(joints) if joints else [n for n in sim["order"] if sim["joints"][n]["type"] in JC.MOVABLE]
    rows, missing = [], []
    for n in names:
        a, b = sim["joints"].get(n), real["joints"].get(n)
        if a is None or b is None:
            missing.append(n)
            continue
        dxyz = dist_mm(a["xyz"], b["xyz"])
        drpy = max(abs(math.atan2(math.sin(x - y), math.cos(x - y))) for x, y in zip(a["rpy"], b["rpy"]))
        dax = max(abs(x - y) for x, y in zip(a["axis"], b["axis"]))
        dlo = abs((a["lower"] or 0) - (b["lower"] or 0))
        dhi = abs((a["upper"] or 0) - (b["upper"] or 0))
        issues = []
        if dxyz > 0.1:
            issues.append(f"origin xyz 差 {dxyz:.2f} mm")
        if drpy > 1e-3:
            issues.append(f"origin rpy 差 {math.degrees(drpy):.2f}°")
        if dax > 1e-6:
            issues.append(f"axis {b['axis']} ≠ {a['axis']}")
        if dlo > 1e-3 or dhi > 1e-3:
            issues.append(f"限位 {math.degrees(b['lower'] or 0):.1f}~{math.degrees(b['upper'] or 0):.1f}° ≠ "
                          f"{math.degrees(a['lower'] or 0):.1f}~{math.degrees(a['upper'] or 0):.1f}°")
        if a["type"] != b["type"]:
            issues.append(f"type {b['type']} ≠ {a['type']}")
        rows.append({"joint": n, "ok": not issues, "issues": issues})
    return {"same": not missing and all(r["ok"] for r in rows), "rows": rows, "missing": missing}


def live_consistency(sim, real, joints, q_rad, tf_pose, base="base_link", link="link7"):
    """Same measured angles through both URDFs (+ TF). -> dict with poses and differences in mm / deg."""
    qmap = dict(zip(joints, q_rad))
    Ts, Tr = fk(sim, qmap, base, link), fk(real, qmap, base, link)
    out = {"link": link, "sim": pose_of(Ts) if Ts else None, "real": pose_of(Tr) if Tr else None, "tf": tf_pose}
    if Ts and Tr:
        out["sim_real_mm"] = dist_mm(out["sim"]["xyz"], out["real"]["xyz"])
    if Tr and tf_pose and tf_pose.get("xyz"):
        out["urdf_tf_mm"] = dist_mm(out["real"]["xyz"], tf_pose["xyz"])
    return out


# =========================================================================== direction test / statistics
def predicted_displacement(urdf, joints, q_rad, j, delta_rad, base="base_link", link="link7"):
    q2 = list(q_rad)
    q2[j] += delta_rad
    a = fk(urdf, dict(zip(joints, q_rad)), base, link)
    b = fk(urdf, dict(zip(joints, q2)), base, link)
    if a is None or b is None:
        return None
    return [b[i][3] - a[i][3] for i in range(3)]


def evaluate_direction(pred, meas):
    """Predicted vs measured TCP displacement (m). -> (ok, cos, mag_ratio, text)"""
    if pred is None or meas is None:
        return None, None, None, "缺少预测或实测"
    np_, nm = math.sqrt(sum(x * x for x in pred)), math.sqrt(sum(x * x for x in meas))
    if np_ < 1e-4:
        return None, None, None, f"该关节在当前姿态几乎不移动 link（预测 {np_ * 1000:.1f} mm），只能看转向"
    if nm < 1e-5:
        return False, 0.0, 0.0, "实测几乎没有移动"
    cos = sum(a * b for a, b in zip(pred, meas)) / (np_ * nm)
    ratio = nm / np_
    ok = cos >= DIRECTION_COS_MIN and abs(ratio - 1) <= DIRECTION_MAG_TOL
    return ok, cos, ratio, f"方向余弦 {cos:+.3f} · 幅值比 {ratio:.2f}"


def stats(vals):
    v = [float(x) for x in vals]
    if not v:
        return {"n": 0, "mean": None, "std": None, "max": None}
    m = sum(v) / len(v)
    sd = math.sqrt(sum((x - m) ** 2 for x in v) / len(v))
    return {"n": len(v), "mean": m, "std": sd, "max": max(v)}


def stop_metrics(samples, j, t_cancel):
    """samples: [(t, [deg x7])]. -> angle at cancel, rest angle, overshoot after cancel, time until motion < 0.05°/0.1 s."""
    before = [s for s in samples if s[0] <= t_cancel]
    after = [s for s in samples if s[0] > t_cancel]
    if not before or len(after) < 2:
        return None
    a0 = before[-1][1][j]
    rest = after[-1][1][j]
    t_stop = None
    for k in range(1, len(after)):
        if abs(after[k][1][j] - after[k - 1][1][j]) < 0.05 and (after[k][0] - after[k - 1][0]) > 0:
            t_stop = after[k - 1][0] - t_cancel
            break
    return {"angle_at_cancel": a0, "rest_angle": rest, "overshoot_deg": abs(rest - a0),
            "stop_time_s": None if t_stop is None else max(0.0, t_stop)}


# =========================================================================== workspace config + fence (V3.16 polygon)
# V3.16 SAFETY ZONE: the fence is an XY polygon (taught from the table marks or drawn in the editor) united with a
# base disc (the arm's own column), applied to check points lower than xy_free_height above the table (higher
# points only have z_max), plus z_min (all check points; the tool may go lower only while teaching / inspecting). A point inside the polygon but closer than `margin` to an edge is a warning, not a violation.
# Old V3.15 files (x/y/z box) are migrated to a rectangle.
#
# V3.23 SAFETY RULES
#   A (default, "围墙"): the polygon extruded vertically (table .. z_max) is a wall. The whole arm incl. the gripper
#     must stay inside at ANY height: the kinematic chain link2..link7 -> tool is sampled every sample_step, every
#     sample is a sphere (link_radius; gripper_radius for link7 -> tool) that must keep `margin` from every wall.
#     The polygon must contain the base (else every pose is refused; editor: 「并入底座」). The walls are also sent
#     to MoveIt as thin boxes so it plans around them. Teach / inspect: the tool tip and gripper may touch the wall
#     line (radius 0) so the pointer can reach the corner marks; the arm keeps its clearance.
#   B ("分层", the V3.16-V3.22 rule): polygon ∪ base disc, only for check points lower than xy_free_height above the
#     table; higher points only have z_max. For open tables without walls nearby.
TOOL_AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0), "+z": (0, 0, 1), "-z": (0, 0, -1)}


def default_workspace():
    return {
        "version": "V3.17", "frame": "base_link", "measured": False,
        "note": "默认值（假设 NERO 底座装在桌面上）。请示教或拖动安全区后保存；measured=false 时界面标红。",
        "boxes": [
            {"name": "table", "size": [1.20, 1.20, 0.04], "pos": [-0.30, 0.0, -0.02]},     # V3.17: work side is −x
        ],
        "corners": [{"name": f"M{i + 1}", "source": "默认", "q": None} for i in range(6)],
        "fence": {"enabled": True, "path_check": True,
                  # in front (−x), V3.23: base included (rule A) -> 6 corners
                  "polygon": [[0.12, 0.12], [-0.15, 0.45], [-0.70, 0.45], [-0.70, -0.45], [-0.15, -0.45], [0.12, -0.12]],
                  "base_radius": 0.20, "z_table": 0.0, "z_min_points": 0.03, "z_max": 1.00, "margin": 0.02,
                  "xy_free_height": 0.15,
                  "rule": "A", "link_radius": 0.05, "gripper_radius": 0.05, "sample_step": 0.05,
                  "walls_to_moveit": True, "wall_thickness": 0.02,
                  "chain": ["link2", "link3", "link4", "link5", "link6", "link7"],
                  "points": ["link4", "link5", "link6", "link7", "tool"],
                  "tool": {"mode": "sim_tcp", "length": 0.10, "axis": "+x"}},
    }


def _migrate(d):
    """V3.15 box fence -> V3.16 polygon (rectangle from x/y), z -> z_max; tool_from_sim_tcp -> tool.mode."""
    f = d.get("fence") or {}
    if "polygon" not in f and "x" in f and "y" in f:
        (x0, x1), (y0, y1) = f["x"], f["y"]
        f["polygon"] = [[x0, y1], [x1, y1], [x1, y0], [x0, y0]]
        if "z" in f:
            f["z_max"] = f["z"][1]
        f["tool"] = {"mode": "sim_tcp" if f.get("tool_from_sim_tcp", True) else "none", "length": 0.10, "axis": "+x"}
        d["corners"] = [{"name": f"M{i + 1}", "source": "V3.15 迁移", "q": None} for i in range(4)]
        d["migrated_from"] = d.get("version", "V3.15")
    for k in ("x", "y", "z", "tool_from_sim_tcp"):
        f.pop(k, None)
    return d


def load_workspace(path=WORKSPACE_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            d = _migrate(json.load(f))
        base = default_workspace()
        base.update({k: v for k, v in d.items() if k != "fence"})
        base["fence"].update(d.get("fence") or {})
        return validate_workspace(base)
    except FileNotFoundError:
        return default_workspace()
    except Exception as exc:
        w = default_workspace()
        w["load_error"] = str(exc)
        return w


def validate_workspace(w):
    for b in w.get("boxes", []):
        if not str(b.get("name") or "").strip():
            raise ValueError("box without name")
        b["size"] = [float(x) for x in b["size"]]
        b["pos"] = [float(x) for x in b["pos"]]
        if len(b["size"]) != 3 or len(b["pos"]) != 3 or min(b["size"]) <= 0:
            raise ValueError(f"box {b['name']}: size/pos must be 3 numbers, size > 0")
    f = w["fence"]
    poly = [[float(p[0]), float(p[1])] for p in f.get("polygon") or []]
    if len(poly) < 3:
        raise ValueError("安全区至少需要 3 个角点")
    if polygon_self_intersects(poly):                       # V3.24: checked first (a crossed polygon has ~0 area)
        raise ValueError("安全区多边形自相交（角点顺序不对）")
    if abs(polygon_area(poly)) < 0.01:
        raise ValueError("安全区面积太小（< 0.01 m²）")
    f["polygon"] = poly
    for k, d in (("base_radius", 0.20), ("z_table", 0.0), ("z_min_points", 0.03), ("z_max", 1.00), ("margin", 0.04),
                 ("xy_free_height", 0.15)):
        f[k] = float(f.get(k, d))
    if f["z_min_points"] >= f["z_max"]:
        raise ValueError("z_min 必须小于 z 上限")
    if not (0.0 <= f["margin"] <= 0.2) or not (0.0 <= f["base_radius"] <= 0.5):
        raise ValueError("余量 0–0.2 m，底座区半径 0–0.5 m")
    if not (0.03 <= f["xy_free_height"] <= 2.0):
        raise ValueError("XY 限制高度 0.03–2.0 m（桌面以上）")
    f["rule"] = str(f.get("rule", "A")).upper()
    if f["rule"] not in ("A", "B"):
        raise ValueError("安全规则只能是 A（围墙）或 B（分层）")
    for k, d, lo, hi in (("link_radius", 0.05, 0.0, 0.2), ("gripper_radius", 0.05, 0.0, 0.2),
                         ("sample_step", 0.05, 0.01, 0.2), ("wall_thickness", 0.02, 0.005, 0.2)):
        f[k] = float(f.get(k, d))
        if not (lo <= f[k] <= hi):
            raise ValueError(f"{k} 超出范围 {lo}–{hi} m")
    f["walls_to_moveit"] = bool(f.get("walls_to_moveit", True))
    f.setdefault("chain", ["link2", "link3", "link4", "link5", "link6", "link7"])
    t = f.setdefault("tool", {"mode": "sim_tcp", "length": 0.10, "axis": "+x"})
    t["length"] = float(t.get("length", 0.10))
    if t.get("mode") not in ("sim_tcp", "pointer", "none") or t.get("axis", "+x") not in TOOL_AXES or not (0 <= t["length"] <= 0.5):
        raise ValueError("工具设置无效")
    cm = w.get("corners") or []
    w["corners"] = (cm + [{"name": f"M{i + 1}", "source": "编辑", "q": None} for i in range(len(cm), len(poly))])[:len(poly)]
    return w


def save_workspace(w, path=WORKSPACE_PATH):
    w = validate_workspace(json.loads(json.dumps(w)))
    w["version"] = "V3.23"
    w["measured"] = True
    w["saved"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(w, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return w


def tool_transform_from_sim(sim_urdf):
    """Sim TCP relative to link7 (flange + gpu_tcp_joint), as a 4x4 list; None if absent."""
    if not sim_urdf:
        return None
    return link_offset_transform(sim_urdf, "link7", "tcp_link")


def tool_transform(tool, sim_urdf):
    """V3.16: tool point relative to link7. sim_tcp = the sim TCP; pointer = `length` along `axis` of link7."""
    tool = tool or {}
    if tool.get("mode") == "none":
        return None
    if tool.get("mode") == "pointer":
        a = TOOL_AXES.get(tool.get("axis", "+x"), (1, 0, 0))
        L = float(tool.get("length", 0.1))
        return _T(_rpy_mat(0, 0, 0), [a[0] * L, a[1] * L, a[2] * L])
    return tool_transform_from_sim(sim_urdf)


def fence_points(urdf, joints, q_rad, base, fence, tool_T=None):
    if fence.get("rule", "B") == "A" and "polygon" in fence:
        return chain_points(urdf, joints, q_rad, base, fence, tool_T)
    qmap = dict(zip(joints, q_rad))
    pts = {}
    T7 = None
    for name in fence.get("points", []):
        if name == "tool":
            if tool_T is None:
                continue
            T7 = T7 or fk(urdf, qmap, base, "link7")
            if T7 is not None:
                T = _mat_mul(T7, tool_T)
                pts["tool"] = (T[0][3], T[1][3], T[2][3])
        else:
            T = fk(urdf, qmap, base, name)
            if T is not None:
                pts[name] = (T[0][3], T[1][3], T[2][3])
                if name == "link7":
                    T7 = T
    return pts


GRIP_TAG = "夹爪"


def chain_points(urdf, joints, q_rad, base, fence, tool_T=None):
    """V3.23 rule A: link origins of fence['chain'] + tool, plus samples every sample_step along the chain.
    Names: 'link5', 'link5~link6#1', '夹爪#1' (link7 -> tool), 'tool'. Zero-length segments are skipped."""
    qmap = dict(zip(joints, q_rad))
    nodes = []
    T7 = None
    for name in fence.get("chain") or []:
        T = fk(urdf, qmap, base, name)
        if T is None:
            continue
        nodes.append((name, (T[0][3], T[1][3], T[2][3])))
        if name == "link7":
            T7 = T
    if tool_T is not None:
        T7 = T7 or fk(urdf, qmap, base, "link7")
        if T7 is not None:
            T = _mat_mul(T7, tool_T)
            nodes.append(("tool", (T[0][3], T[1][3], T[2][3])))
    step = float(fence.get("sample_step", 0.05))
    pts = {}
    for i, (name, p) in enumerate(nodes):
        if i > 0:
            pn, pp = nodes[i - 1]
            L = math.dist(p, pp)
            k = int(L / step)
            tag = GRIP_TAG if name == "tool" else f"{pn}~{name}"
            for j in range(1, k + (1 if L - k * step > 1e-6 else 0)):
                t = j / (k + (1 if L - k * step > 1e-6 else 0))
                pts[f"{tag}#{j}"] = tuple(pp[a] + (p[a] - pp[a]) * t for a in range(3))
        if name not in pts and not any(math.dist(p, q) < 1e-6 for q in pts.values()):
            pts[name] = p
        elif name == "tool":
            pts[name] = p
    return pts


WRIST_PREFIXES = ("link5", "link6", "link7")


def relaxed_point(name, fence):
    """teach / inspect (rule A): wrist + gripper + tool tip may reach the wall line (pointer on a corner mark)."""
    return bool(fence.get("relax_tool")) and (name == "tool" or name.startswith(GRIP_TAG) or name.startswith(WRIST_PREFIXES))


def point_radius(name, fence):
    """sphere radius of a rule-A check point (0 for wrist / gripper / tool tip while teaching or inspecting)."""
    grip = name == "tool" or name.startswith(GRIP_TAG)
    if relaxed_point(name, fence):
        return 0.0
    if grip:
        return float(fence.get("gripper_radius", 0.05))
    return float(fence.get("link_radius", 0.05))


def base_clearance(fence):
    """rule A: distance (m) from the base axis (0, 0) to the nearest wall, negative if the base is outside."""
    poly = fence["polygon"]
    d, _e = edge_distance(0.0, 0.0, poly)
    return d if point_in_polygon(0.0, 0.0, poly) else -d


def base_inside(fence):
    return base_clearance(fence) >= float(fence.get("link_radius", 0.05)) + float(fence.get("margin", 0.0))


def _hull(points):
    pts = sorted(set((round(p[0], 6), round(p[1], 6)) for p in points))
    if len(pts) <= 2:
        return [list(p) for p in pts]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return [list(p) for p in lo[:-1] + up[:-1]]


def merge_base(w, half=None):
    """V3.23 「并入底座」: convex hull of the polygon and a square around the base (half = link_radius + margin
    + 3 cm, ≥ 12 cm). Existing corners keep their names / taught data; new ones are B1, B2, ... -> new workspace."""
    w = json.loads(json.dumps(w))
    f = w["fence"]
    h = half if half is not None else max(0.12, f.get("link_radius", 0.05) + f.get("margin", 0.04) + 0.03)
    old = [list(p) for p in f["polygon"]]
    hull = _hull(old + [[-h, -h], [-h, h], [h, h], [h, -h]])
    corners_old = w.get("corners") or []
    out_c, nb = [], 0
    for p in hull:
        k = next((i for i, q in enumerate(old) if abs(q[0] - p[0]) < 1e-5 and abs(q[1] - p[1]) < 1e-5), None)
        if k is not None and k < len(corners_old):
            out_c.append(corners_old[k])
        else:
            nb += 1
            out_c.append({"name": f"B{nb}", "source": "并入底座", "q": None})
    f["polygon"] = hull
    w["corners"] = out_c
    return validate_workspace(w)


def wall_boxes(w):
    """V3.23: rule A walls for MoveIt — one thin box per polygon edge, inner face on the edge, table .. z_max + 5 cm,
    extended by the thickness at both ends so the corners are closed. -> [{"id", "size", "pos", "yaw"}]"""
    f = w["fence"]
    poly = f["polygon"]
    t = float(f.get("wall_thickness", 0.02))
    z0, z1 = float(f.get("z_table", 0.0)), float(f.get("z_max", 1.0)) + 0.05
    ccw = polygon_area(poly) > 0
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy)
        if L < 1e-6:
            continue
        ux, uy = dx / L, dy / L
        nx, ny = (uy, -ux) if ccw else (-uy, ux)          # outward normal (interior is on the left for ccw)
        cx, cy = (a[0] + b[0]) / 2 + nx * t / 2, (a[1] + b[1]) / 2 + ny * t / 2
        out.append({"id": SCENE_PREFIX + f"wall_{i + 1}", "size": [L + 2 * t, t, z1 - z0],
                    "pos": [cx, cy, (z0 + z1) / 2], "yaw": math.atan2(uy, ux)})
    return out


# ---- polygon geometry (metres, XY)
def polygon_area(poly):
    return 0.5 * sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1]
                     for i in range(len(poly)))


def point_in_polygon(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xi = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xi > x:
                inside = not inside
    return inside


def seg_dist(px, py, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - a[0]) * dx + (py - a[1]) * dy) / L2))
    return math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy))


def edge_distance(x, y, poly):
    """-> (distance to the nearest edge, edge index i = edge M(i+1)-M(i+2))"""
    best = (float("inf"), -1)
    for i in range(len(poly)):
        d = seg_dist(x, y, poly[i], poly[(i + 1) % len(poly)])
        if d < best[0]:
            best = (d, i)
    return best


def _segs_cross(p1, p2, p3, p4):
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2, d3, d4 = orient(p3, p4, p1), orient(p3, p4, p2), orient(p1, p2, p3), orient(p1, p2, p4)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def polygon_self_intersects(poly):
    n = len(poly)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            if _segs_cross(poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n]):
                return True
    return False


def order_polygon(points):
    """Sort taught corners around their centroid (counter-clockwise) so the polygon never self-intersects."""
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    return sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def inset_polygon(poly, m):
    """Polygon offset inward by m (edge offset + intersection of neighbouring offset lines)."""
    n = len(poly)
    if n < 3 or m <= 0:
        return [list(p) for p in poly]
    s = 1.0 if polygon_area(poly) > 0 else -1.0           # ccw: interior on the left
    lines = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L * s, dx / L * s
        lines.append(((a[0] + nx * m, a[1] + ny * m), (dx, dy)))
    out = []
    for i in range(n):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-12:
            out.append([p2[0], p2[1]])
            continue
        t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
        out.append([p1[0] + t * d1[0], p1[1] + t * d1[1]])
    return out


def edge_name(i, n, corners=None):
    a, b = i, (i + 1) % n
    na = (corners[a]["name"] if corners and a < len(corners) else f"M{a + 1}")
    nb = (corners[b]["name"] if corners and b < len(corners) else f"M{b + 1}")
    return f"{na}–{nb}"


def classify_point(name, p, fence):
    """-> (violations, warnings) for one check point."""
    viol, warn = [], []
    zmin = fence["z_min_points"]
    if name == "tool" and fence.get("tool_z_min_override") is not None:
        zmin = float(fence["tool_z_min_override"])
    if p[2] < zmin:
        viol.append({"point": name, "axis": "z", "value": p[2], "bound": zmin, "kind": "z_min"})
    if p[2] > fence["z_max"]:
        viol.append({"point": name, "axis": "z", "value": p[2], "bound": fence["z_max"], "kind": "max"})
    xy_only = fence.get("xy_points_only")                  # teach / inspect: XY for these points only
    if "polygon" in fence and fence.get("rule", "B") == "A":
        poly = fence["polygon"]
        inside = point_in_polygon(p[0], p[1], poly)
        d, e = edge_distance(p[0], p[1], poly)
        c = point_radius(name, fence) + (0.0 if relaxed_point(name, fence) else float(fence.get("margin", 0.0)))
        tol = fence.get("edge_tol", 0.002)
        if (not inside and d > tol) or (inside and c > 0 and d < c):
            pen = (c + d) if not inside else (c - d)
            viol.append({"point": name, "axis": "xy", "value": pen, "bound": c, "kind": "wall", "edge": e,
                         "dist": d if inside else -d, "xyz": list(p)})
        elif inside and d < c + 0.02:
            warn.append({"point": name, "kind": "margin", "dist": d, "edge": e, "xyz": list(p)})
    elif "polygon" in fence and p[2] >= fence.get("z_table", 0.0) + fence.get("xy_free_height", 1e9):
        pass                                                # high above the table: XY is free (z_max still applies)
    elif "polygon" in fence and xy_only is not None and name not in xy_only:
        pass
    elif "polygon" in fence:
        poly = fence["polygon"]
        in_base = math.hypot(p[0], p[1]) <= fence.get("base_radius", 0.0)
        inside = point_in_polygon(p[0], p[1], poly)
        d, e = edge_distance(p[0], p[1], poly)
        if not inside and d <= fence.get("edge_tol", 0.002):
            inside = True                                   # on the boundary (e.g. hovering over a corner mark)
        if not inside and not in_base:
            viol.append({"point": name, "axis": "xy", "value": d, "bound": 0.0, "kind": "outside", "edge": e,
                         "xyz": list(p)})
        elif inside and d < fence.get("margin", 0.0):
            warn.append({"point": name, "kind": "margin", "dist": d, "edge": e, "xyz": list(p)})
    else:                                                   # V3.15 box (kept for old callers)
        for i, ax in enumerate("xy"):
            lo, hi = fence[ax]
            if p[i] < lo or p[i] > hi:
                viol.append({"point": name, "axis": ax, "value": p[i], "bound": lo if p[i] < lo else hi,
                             "kind": "min" if p[i] < lo else "max"})
    return viol, warn


def fence_violations(points, fence):
    out = []
    for name, p in points.items():
        out += classify_point(name, p, fence)[0]
    return out


def fence_warnings(points, fence):
    out = []
    for name, p in points.items():
        out += classify_point(name, p, fence)[1]
    return out


def fence_check(urdf, joints, q_rad, base, fence, tool_T=None):
    """-> (ok, violations, points)"""
    if not fence or not fence.get("enabled", True):
        return True, [], {}
    pts = fence_points(urdf, joints, q_rad, base, fence, tool_T)
    if not pts:      # wrong link names / base frame: never pass silently
        v = [{"point": "(none)", "axis": "-", "value": 0.0, "bound": 0.0, "kind": "missing",
              "text": f"检查点 {fence.get('points')} 在 URDF（base {base}）里一个都找不到"}]
        return False, v, pts
    v = fence_violations(pts, fence)
    return not v, v, pts


def path_fence_check(urdf, joints, traj_names, traj_points, base, fence, tool_T=None, allow_exit=True):
    """traj_points: [[q per traj_names]]. -> None or {"waypoint": k, "violations": [...]}
    V3.16 recovery rule: if the arm STARTS outside, the leading waypoints may stay outside (it is leaving), but once
    a waypoint is fully inside every later one must be inside; the last one must always be inside."""
    if not fence or not fence.get("enabled", True):
        return None
    idx = [traj_names.index(n) if n in traj_names else None for n in joints]
    leaving = allow_exit
    n = len(traj_points)
    worst0 = None
    for k, pt in enumerate(traj_points):
        q = [pt[i] if i is not None else 0.0 for i in idx]
        ok, v, _p = fence_check(urdf, joints, q, base, fence, tool_T)
        if ok:
            leaving = False
            continue
        if any(x["kind"] == "missing" for x in v):
            return {"waypoint": k, "of": n, "violations": v}
        sev = violation_severity(v)
        if worst0 is None:
            worst0 = sev
        if leaving and k < n - 1 and sev <= worst0 + 0.005:      # leaving, and not getting further out
            continue
        return {"waypoint": k, "of": n, "violations": v, "leaving": leaving}
    return None


def violation_severity(v):
    s = 0.0
    for x in v:
        if x["kind"] in ("outside", "wall"):
            s += x["value"]
        elif x["kind"] in ("z_min", "min"):
            s += max(0.0, x["bound"] - x["value"])
        elif x["kind"] == "max":
            s += max(0.0, x["value"] - x["bound"])
    return s


def violation_text(v, fence=None, corners=None):
    if not v:
        return ""
    n = len((fence or {}).get("polygon") or []) or 4

    def one(x):
        if x.get("text"):
            return x["text"]
        if x["kind"] == "outside":
            return f"{x['point']} 出界 {x['value'] * 1000:.0f} mm（{edge_name(x['edge'], n, corners)} 边）"
        if x["kind"] == "wall":
            dd = x.get("dist", 0.0)
            wall = edge_name(x["edge"], n, corners)
            if dd < 0:
                return f"{x['point']} 越墙 {abs(dd) * 1000:.0f} mm（{wall} 墙）"
            return f"{x['point']} 离墙 {dd * 1000:.0f} mm < {x['bound'] * 1000:.0f} mm（{wall} 墙）"
        return (f"{x['point']} {x['axis']}={x['value']:.3f} m {'<' if x['kind'] != 'max' else '>'} "
                f"{x['bound']:.3f}{' (z_min)' if x['kind'] == 'z_min' else ''}")
    v = sorted(v, key=lambda x: -(x.get("value", 0.0) if x["kind"] in ("wall", "outside") else 1.0))   # worst first
    return "; ".join(one(x) for x in v[:4]) + (f" … (+{len(v) - 4})" if len(v) > 4 else "")


def scene_objects(w, walls=None):
    """collision boxes (+ V3.23 rule-A walls unless walls=False or walls_to_moveit is off) for MoveIt."""
    out = [{"id": SCENE_PREFIX + b["name"], "size": list(b["size"]), "pos": list(b["pos"])} for b in w.get("boxes", [])]
    f = w.get("fence") or {}
    if walls is not False and f.get("rule", "B") == "A" and f.get("walls_to_moveit", True) and f.get("enabled", True):
        out += wall_boxes(w)
    return out


def find_violating_target(urdf, joints, q_rad, limits, base, fence, tool_T=None, step_deg=5.0, max_deg=120.0):
    """For the scene/fence test: the smallest J2 / J4 / J2+J4 change from q that puts a point under z_min.
    Pure computation; never moves anything. -> (q_target, violations) or (None, [])."""
    cands = []
    combos = [((1,), (1,)), ((1,), (-1,)), ((3,), (1,)), ((3,), (-1,))] + \
             [((1, 3), (a, b)) for a in (1, -1) for b in (1, -1)]
    for jset, sgns in combos:
        k = 1
        while k * step_deg <= max_deg:
            q = list(q_rad)
            inside = True
            for j, sgn in zip(jset, sgns):
                q[j] += math.radians(sgn * k * step_deg)
                lo, hi = limits[j]
                if not (lo + math.radians(3) <= q[j] <= hi - math.radians(3)):
                    inside = False
            if not inside:
                break
            pts = fence_points(urdf, joints, q, base, fence, tool_T)
            below = [p for p in pts.values() if p[2] < fence["z_min_points"] - 0.02]
            if below:
                cands.append((k * step_deg * len(jset), q, fence_violations(pts, fence)))
                break
            k += 1
    if not cands:
        return None, []
    cands.sort(key=lambda c: c[0])
    return cands[0][1], cands[0][2]


# ---- small IK (inspection hover poses): damped least squares on tool position + link7 orientation
def _solve(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-15:
            return None
        for r in range(n):
            if r != c:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _rot_err(Ra, Rb):
    """Rotation vector (approx) taking Ra to Rb."""
    R = [[sum(Rb[i][k] * Ra[j][k] for k in range(3)) for j in range(3)] for i in range(3)]   # Rb Ra^T
    return [0.5 * (R[2][1] - R[1][2]), 0.5 * (R[0][2] - R[2][0]), 0.5 * (R[1][0] - R[0][1])]


def tool_pose(urdf, joints, q, base, tool_T):
    T = fk(urdf, dict(zip(joints, q)), base, "link7")
    if T is None:
        return None
    if tool_T is not None:
        T = _mat_mul(T, tool_T)
    return T


def ik_tool(urdf, joints, q0, base, tool_T, target_xyz, limits, iters=250, tol=5e-4, keep_R=None):
    """Joint vector that puts the tool point at target_xyz, keeping the orientation of q0 (or keep_R) when
    possible; otherwise position only with a light orientation preference. -> (q, err_m) or (None, err)"""
    q, err = _ik_tool(urdf, joints, q0, base, tool_T, target_xyz, limits, iters, tol, keep_R, 0.3, True)
    if q is not None:
        return q, err
    best = err
    for w in (0.02, 0.0):                         # looser orientation, then position only
        q2, err2 = _ik_tool(urdf, joints, q0, base, tool_T, target_xyz, limits, iters * 2, tol, keep_R, w, False)
        if q2 is not None:
            return q2, err2
        best = min(best, err2)
    return None, best


def _ik_tool(urdf, joints, q0, base, tool_T, target_xyz, limits, iters, tol, keep_R, w_rot, strict):
    """Levenberg-Marquardt on [tool position; w_rot * link7 rotation error] with an active set for joint limits."""
    n = len(q0)
    q = [max(lo, min(hi, v)) for v, (lo, hi) in zip(q0, limits)]
    T0 = tool_pose(urdf, joints, q, base, tool_T)
    if T0 is None:
        return None, float("inf")
    Rt = keep_R or [row[:3] for row in T0[:3]]
    h = 1e-6

    def residual(qv):
        T = tool_pose(urdf, joints, qv, base, tool_T)
        ep = [target_xyz[i] - T[i][3] for i in range(3)]
        er = _rot_err([row[:3] for row in T[:3]], Rt)
        return T, ep, er, ep + [w_rot * x for x in er]

    T, ep, er, e = residual(q)
    cost = sum(x * x for x in e)
    lam = 1e-3
    for _ in range(iters):
        err = math.sqrt(sum(x * x for x in ep))
        if err < tol and (not strict or max(abs(x) for x in er) < 2e-3):
            return q, err
        Jc = []
        for j in range(n):
            qq = list(q)
            qq[j] += h
            T2 = tool_pose(urdf, joints, qq, base, tool_T)
            dp = [(T2[i][3] - T[i][3]) / h for i in range(3)]
            dr = _rot_err([row[:3] for row in T[:3]], [row[:3] for row in T2[:3]])
            Jc.append(dp + [w_rot * x / h for x in dr])
        improved = False
        for _try in range(12):
            locked, dq = set(), None
            for _k in range(n + 1):
                Jm = [[0.0 if j in locked else Jc[j][i] for j in range(n)] for i in range(6)]
                A = [[sum(Jm[i][k] * Jm[r][k] for k in range(n)) + (lam if i == r else 0.0) for r in range(6)] for i in range(6)]
                y = _solve(A, e)
                if y is None:
                    break
                dq = [sum(Jm[i][j] * y[i] for i in range(6)) for j in range(n)]
                push = {j for j in range(n) if j not in locked and
                        ((q[j] <= limits[j][0] + 1e-9 and dq[j] < 0) or (q[j] >= limits[j][1] - 1e-9 and dq[j] > 0))}
                if not push:
                    break
                locked |= push
            if dq is None:
                break
            mx = max(abs(x) for x in dq) or 1.0
            if mx > 0.2:
                dq = [x * 0.2 / mx for x in dq]
            qn = [max(lo, min(hi, a + d)) for a, d, (lo, hi) in zip(q, dq, limits)]
            Tn, epn, ern, en = residual(qn)
            cn = sum(x * x for x in en)
            if cn < cost:
                q, T, ep, er, e, cost = qn, Tn, epn, ern, en, cn
                lam = max(1e-7, lam * 0.3)
                improved = True
                break
            lam *= 5.0
        if not improved:
            break
    err = math.sqrt(sum(x * x for x in ep))
    ok = err < tol * 4 and (not strict or max(abs(x) for x in er) < 1e-2)
    return (q, err) if ok else (None, err)


# =========================================================================== logs
MOTION_LOG_HEADER = ["time", "label", "mode", "speed", "ok", "error", "error_code", "planning_time_s", "elapsed_s",
                     "final_error_deg", "final_error_joint"] + [f"target_J{i}" for i in range(1, 8)] + \
                    [f"actual_J{i}" for i in range(1, 8)]


class MotionLog:
    def __init__(self, path=None):
        self.path = path or os.path.join(LOG_DIR, "real_motion_log.csv")
        self.lock = threading.Lock()

    def add(self, label, mode, speed, res, target_deg, actual_deg):
        row = [time.strftime("%Y-%m-%d %H:%M:%S"), label, mode, f"{speed:.2f}", int(bool(res.get("ok"))),
               (res.get("error") or "")[:200], res.get("error_code", ""), res.get("planning_time", ""),
               res.get("elapsed_s", ""), "" if res.get("final_error_deg") is None else f"{res['final_error_deg']:.3f}",
               "" if res.get("final_error_joint") is None else res["final_error_joint"] + 1]
        row += [f"{v:.3f}" for v in target_deg] + ([f"{v:.3f}" for v in actual_deg] if actual_deg else [""] * 7)
        with self.lock:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            new = not os.path.exists(self.path)
            with open(self.path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(MOTION_LOG_HEADER)
                w.writerow(row)


def write_result(kind, data, rows=None, header=None):
    """logs/v315_<kind>_<ts>.json (+ .csv when rows are given). -> json path"""
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    jp = os.path.join(LOG_DIR, f"v315_{kind}_{ts}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, default=str)
    if rows is not None:
        with open(jp[:-5] + ".csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if header:
                w.writerow(header)
            w.writerows(rows)
    return jp


def load_sim_urdf(path=SIM_URDF_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            return JC.parse_urdf(f.read())
    except Exception:
        return None


# =========================================================================== Tk: shared worker helper
class _WorkerMixin:
    """Workers never touch Tk: they queue UI updates with _ui(); poll() (main thread) runs them."""

    def _ui(self, fn):
        self.ui_q.put(fn)

    def _drain_ui(self):
        while True:
            try:
                fn = self.ui_q.get_nowait()
            except queue.Empty:
                return
            try:
                fn()
            except Exception as exc:
                self.log(f"[WARN] [V3.15] UI update: {exc}")

    def _run(self, name, fn):
        if (self.job is not None and self.job.is_alive()) or self.pipe.move_busy() or \
                (self.jpanel.job is not None and self.jpanel.job.is_alive()):
            self.log(f"[WARN] [V3.15] busy — {name} not started")
            return False
        self.stop_flag = False

        def wrap():
            try:
                fn()
            except Exception as exc:
                self.pipe.log(f"[FAIL] [V3.15] {name}: {exc}")
        self.job = threading.Thread(target=wrap, daemon=True, name=f"v315_{name}")
        self.job.start()
        return True

    def _ready(self):
        if not self.jpanel.enabled:
            self.log("[WARN] [V3.15] 关节面板不可操作（第 6 步 / 关节反馈）：" + self.jpanel.v_state.get())
            return False
        return True

    def _speed(self, pct=None):
        s = float(pct if pct is not None else self.cfg_fn().get("speed_percent", 10))
        return max(0.01, min(1.0, s / 100.0))

    def _wait_status_after(self, t, timeout=3.0):
        t_end = time.time() + timeout
        while time.time() < t_end:
            m = self.pipe.mon_status or {}
            if float(m.get("t") or 0) > t:
                return m
            time.sleep(0.1)
        return self.pipe.mon_status or {}


# =========================================================================== Tk: consistency tab
class ConsistencyTabUI(_WorkerMixin):
    """V3.15 一致性测试: model comparison, live FK/TF comparison, direction wizard, repeatability, stop test."""

    def __init__(self, tk, ttk, parent, pipe, jpanel, cfg_fn, log, ask):
        self.tk, self.ttk, self.pipe, self.jpanel, self.cfg_fn, self.log, self.ask = tk, ttk, pipe, jpanel, cfg_fn, log, ask
        self.sim = load_sim_urdf()
        self.job = None
        self.ui_q = queue.Queue()
        self.stop_flag = False
        self.compared_for = None
        self.tf_watch_t = 0.0
        self.results = {"direction": [], "repeat": None, "stop": None}
        f = ttk.Frame(parent, padding=6)
        f.pack(fill="both", expand=True)
        # (1) model comparison + live
        top = ttk.LabelFrame(f, text="① 模型对比 + 实时一致性（同一组实测关节角 → 仿真 URDF / 驱动 URDF / 真机 TF，link7）")
        top.pack(fill="x", pady=3)
        r = ttk.Frame(top)
        r.pack(fill="x", padx=6, pady=2)
        self.v_model = tk.StringVar(value="等待驱动 URDF（第 5/6 步）")
        self.lbl_model = JC.fixed_label(tk, r, 2, textvariable=self.v_model, font=("TkDefaultFont", 9, "bold"), fg="#888888",
                                        wraplength=1100, pack={"side": "left", "fill": "x", "expand": True})
        ttk.Button(r, text="重新对比", command=lambda: self.compare(force=True)).pack(side="right")
        g = ttk.Frame(top)
        g.pack(anchor="w", padx=6, pady=2)
        heads = ["", "x mm", "y mm", "z mm", "roll°", "pitch°", "yaw°"]
        for c_, h in enumerate(heads):
            ttk.Label(g, text=h, width=12 if c_ == 0 else 9, anchor="e", font=("TkDefaultFont", 9, "bold")).grid(row=0, column=c_)
        self.live = {}
        for r_, (k, lab) in enumerate((("sim", "仿真 URDF FK"), ("real", "驱动 URDF FK"), ("tf", "真机 TF")), start=1):
            ttk.Label(g, text=lab, width=12, anchor="e").grid(row=r_, column=0)
            for c_ in range(1, 7):
                lb = tk.Label(g, text="-", width=9, anchor="e", font=("TkFixedFont", 9))
                lb.grid(row=r_, column=c_)
                self.live[(k, c_)] = lb
        self.v_live = tk.StringVar(value="")
        self.lbl_live = JC.fixed_label(tk, top, 1, textvariable=self.v_live, font=("TkDefaultFont", 9, "bold"), fg="#555555",
                                       pack={"padx": 6, "pady": (0, 4)})
        # (2) direction wizard
        dw = ttk.LabelFrame(f, text="② 关节方向 / 零位向导（每个关节转 +Δ → 比较预期与实测 link7 位移 → 你确认真机与 RViz 转向一致 → 回原位）")
        dw.pack(fill="x", pady=3)
        r = ttk.Frame(dw)
        r.pack(fill="x", padx=6, pady=2)
        ttk.Label(r, text="关节").pack(side="left")
        self.v_dir_j = tk.StringVar(value="全部 J1–J7")
        ttk.Combobox(r, textvariable=self.v_dir_j, values=["全部 J1–J7"] + [f"J{i}" for i in range(1, 8)], width=10,
                     state="readonly").pack(side="left", padx=4)
        ttk.Label(r, text="Δ°").pack(side="left")
        self.v_dir_d = tk.StringVar(value="10")
        ttk.Entry(r, textvariable=self.v_dir_d, width=5).pack(side="left", padx=4)
        self.btn_dir = ttk.Button(r, text="▶ 开始方向测试", command=self.on_direction)
        self.btn_dir.pack(side="left", padx=6)
        ttk.Button(r, text="■ 中止", command=self.on_stop).pack(side="left")
        ttk.Label(r, text="零位：在「关节控制」姿态库载入「零位」执行，对照 AgileX 零位姿态（手臂竖直、各关节刻线对齐）目视确认。",
                  foreground="#555555").pack(side="left", padx=10)
        self.tv_dir = ttk.Treeview(dw, columns=("j", "pred", "meas", "cos", "ratio", "auto", "human"), show="headings", height=7)
        for c_, (h, w) in zip(self.tv_dir["columns"], (("关节", 50), ("预期 Δ link7 (mm)", 210), ("实测 Δ (TF, mm)", 210),
                                                      ("方向余弦", 80), ("幅值比", 70), ("自动判定", 90), ("人工确认", 90))):
            self.tv_dir.heading(c_, text=h)
            self.tv_dir.column(c_, width=w, anchor="center")
        self.tv_dir.pack(fill="x", padx=6, pady=(0, 4))
        # (3) repeatability
        rp = ttk.LabelFrame(f, text="③ 重复定位精度（A = 当前姿态 ↔ B = A + 偏移，循环 N 次；每次回到 A 记录关节误差与 TF 位置）")
        rp.pack(fill="x", pady=3)
        r = ttk.Frame(rp)
        r.pack(fill="x", padx=6, pady=2)
        ttk.Label(r, text="B 偏移 J1..J7 (°)").pack(side="left")
        self.v_rep_off = tk.StringVar(value="10,5,0,10,0,5,0")
        ttk.Entry(r, textvariable=self.v_rep_off, width=18).pack(side="left", padx=4)
        ttk.Label(r, text="循环").pack(side="left")
        self.v_rep_n = tk.StringVar(value="10")
        ttk.Entry(r, textvariable=self.v_rep_n, width=4).pack(side="left", padx=4)
        ttk.Label(r, text="速度 % 列表").pack(side="left")
        self.v_rep_sp = tk.StringVar(value="10")
        ttk.Entry(r, textvariable=self.v_rep_sp, width=10).pack(side="left", padx=4)
        self.btn_rep = ttk.Button(r, text="▶ 开始", command=self.on_repeat)
        self.btn_rep.pack(side="left", padx=6)
        ttk.Button(r, text="■ 中止", command=self.on_stop).pack(side="left")
        self.v_rep = tk.StringVar(value="")
        JC.fixed_label(tk, rp, 4, textvariable=self.v_rep, font=("TkFixedFont", 9), pack={"padx": 6, "pady": (0, 4)})
        # (4) stop test
        sp = ttk.LabelFrame(f, text="④ 停止测试（慢速转动一个关节，t 秒后发送「停止运动」(cancel)，测停止距离和停止时间）")
        sp.pack(fill="x", pady=3)
        r = ttk.Frame(sp)
        r.pack(fill="x", padx=6, pady=2)
        ttk.Label(r, text="关节").pack(side="left")
        self.v_st_j = tk.StringVar(value="J1")
        ttk.Combobox(r, textvariable=self.v_st_j, values=[f"J{i}" for i in range(1, 8)], width=4, state="readonly").pack(side="left", padx=4)
        ttk.Label(r, text="幅度°").pack(side="left")
        self.v_st_a = tk.StringVar(value="20")
        ttk.Entry(r, textvariable=self.v_st_a, width=4).pack(side="left", padx=4)
        ttk.Label(r, text="速度%").pack(side="left")
        self.v_st_s = tk.StringVar(value="5")
        ttk.Entry(r, textvariable=self.v_st_s, width=4).pack(side="left", padx=4)
        ttk.Label(r, text="cancel 时刻 s").pack(side="left")
        self.v_st_t = tk.StringVar(value="1.0")
        ttk.Entry(r, textvariable=self.v_st_t, width=4).pack(side="left", padx=4)
        self.btn_st = ttk.Button(r, text="▶ 开始", command=self.on_stop_test)
        self.btn_st.pack(side="left", padx=6)
        ttk.Label(r, text="急停按钮的测试请手动做：运动中按 ■ 急停，记录现象和恢复步骤（使能 / 控制门）。",
                  foreground="#555555").pack(side="left", padx=10)
        self.v_st = tk.StringVar(value="")
        JC.fixed_label(tk, sp, 1, textvariable=self.v_st, font=("TkFixedFont", 9), pack={"padx": 6, "pady": (0, 4)})

    # ------------------------------------------------------------- live (main thread)
    def compare(self, force=False):
        real = self.pipe.real_urdf
        if real is None or self.sim is None:
            self.v_model.set("等待驱动 URDF（第 5/6 步）" if self.sim else f"仿真 URDF 读取失败: {SIM_URDF_PATH}")
            self.lbl_model.configure(fg="#888888")
            return None
        if not force and self.compared_for is real:
            return None
        self.compared_for = real
        cmp_ = compare_models(self.sim, real, self.jpanel.model["joints"])
        if cmp_["same"]:
            self.v_model.set("模型对比：驱动 URDF 与仿真 URDF（config/nero.urdf）的 7 个关节 origin / axis / 限位一致 ✓")
            self.lbl_model.configure(fg="#2e7d32")
            self.log("[OK] [V3.15 consistency] 驱动 URDF 与仿真 URDF 的 7 个关节参数一致")
        else:
            bad = [f"{r['joint']}: {', '.join(r['issues'])}" for r in cmp_["rows"] if not r["ok"]]
            if cmp_["missing"]:
                bad.append(f"缺少关节 {cmp_['missing']}")
            self.v_model.set("模型对比：不一致 ⚠  " + " | ".join(bad))
            self.lbl_model.configure(fg="#c62828")
            self.log("[WARN] [V3.15 consistency] 驱动 URDF ≠ 仿真 URDF: " + " | ".join(bad))
        return cmp_

    def poll(self):
        self._drain_ui()
        self.compare()
        if self.pipe.monitor.running() and time.time() - self.tf_watch_t > 5.0:
            self.tf_watch_t = time.time()
            self.pipe.set_tf_watch(["link7"])
        model = self.jpanel.model
        act = self.pipe.actual_arm_deg(model)
        tf_ = ((self.pipe.mon_status or {}).get("tf_links") or {}).get("link7")
        if act is None or self.sim is None:
            self.v_live.set("等待关节反馈")
            return
        lc = live_consistency(self.sim, self.pipe.real_urdf or self.sim, model["joints"], [math.radians(a) for a in act],
                              tf_ if tf_ and tf_.get("xyz") else None, base=model.get("base_link") or "base_link")
        for k in ("sim", "real", "tf"):
            p = lc.get(k)
            for c_ in range(1, 7):
                if not p:
                    txt = "-"
                elif c_ <= 3:
                    txt = f"{p['xyz'][c_ - 1] * 1000:.1f}"
                else:
                    txt = f"{p['rpy_deg'][c_ - 4]:.1f}"
                self.live[(k, c_)].configure(text=txt)
        parts, bad = [], False
        if lc.get("sim_real_mm") is not None and self.pipe.real_urdf is not None:
            d = lc["sim_real_mm"]
            bad |= d > SIM_REAL_TOL_MM
            parts.append(f"仿真 vs 驱动 URDF: {d:.2f} mm {'⚠' if d > SIM_REAL_TOL_MM else '✓'}")
        if lc.get("urdf_tf_mm") is not None:
            d = lc["urdf_tf_mm"]
            bad |= d > URDF_TF_TOL_MM + 2.0
            parts.append(f"驱动 URDF vs TF: {d:.2f} mm {'⚠' if d > URDF_TF_TOL_MM + 2.0 else '✓'}"
                         "（TF 每秒更新，运动中会有差值）")
        elif tf_ and tf_.get("error"):
            parts.append(tf_["error"])
        self.v_live.set("   ".join(parts) or "-")
        self.lbl_live.configure(fg="#c62828" if bad else "#2e7d32")

    def on_stop(self):
        self.stop_flag = True
        self.pipe.cancel_move()

    # ------------------------------------------------------------- direction wizard (worker)
    def on_direction(self):
        if not self._ready():
            return
        d = JC.parse_entry(self.v_dir_d.get())
        if d is None or not (1.0 <= abs(d) <= 30.0):
            self.log("[WARN] [V3.15 direction] Δ 需在 1–30°")
            return
        sel = self.v_dir_j.get()
        joints = list(range(7)) if sel.startswith("全部") else [int(sel[1:]) - 1]
        for it in self.tv_dir.get_children():
            self.tv_dir.delete(it)
        self._run("direction", lambda: self._direction(joints, d))

    def _tf_link7(self, after_t):
        m = self._wait_status_after(after_t, 3.0)
        p = (m.get("tf_links") or {}).get("link7")
        return p["xyz"] if p and p.get("xyz") else None

    def _direction(self, joints, d):
        model = self.jpanel.model
        speed = self._speed()
        if not self.ask("关节方向测试", f"将依次测试 {', '.join(f'J{j + 1}' for j in joints)}：每个关节转 {d:+.0f}°，"
                                     f"停下后请你对照 RViz 确认真机转向，再回到原位。\n速度 {speed * 100:.0f}%。确认周围无障碍、手在急停上？"):
            return
        rows = []
        for j in joints:
            if self.stop_flag:
                break
            a = self.pipe.actual_arm_deg(model)
            if a is None:
                self.pipe.log("[FAIL] [V3.15 direction] 没有关节反馈")
                break
            r = JC.step_test_targets(a, j, model["limits"], step=abs(d))
            if r is None:
                rows.append({"joint": j + 1, "auto": "无空间"})
                continue
            out, back, s = r
            pred = predicted_displacement(self.sim, model["joints"], [math.radians(x) for x in a], j, math.radians(s),
                                          base=model.get("base_link") or "base_link") if self.sim else None
            p0 = self._tf_link7(time.time())
            res = self.pipe.move_and_settle(model, [math.radians(x) for x in out], speed, label=f"方向测试 J{j + 1} {s:+.0f}°")
            if not res.get("ok"):
                self.pipe.log(f"[FAIL] [V3.15 direction] J{j + 1}: {res.get('error')}")
                break
            p1 = self._tf_link7(time.time())
            meas = [b - c for b, c in zip(p1, p0)] if (p0 and p1) else None
            ok, cos, ratio, txt = evaluate_direction(pred, meas)
            human = self.ask(f"J{j + 1} 转向确认", f"J{j + 1} 刚转了 {s:+.0f}°。\n真机的转动方向与 RViz 中模型的转动方向一致吗？\n"
                                                 f"（自动比较：{txt}）")
            row = {"joint": j + 1, "delta_deg": s, "pred_mm": [round(x * 1000, 1) for x in pred] if pred else None,
                   "meas_mm": [round(x * 1000, 1) for x in meas] if meas else None, "cos": cos, "ratio": ratio,
                   "auto": ("通过" if ok else "不符" if ok is False else "—"), "human": "一致" if human else "不一致",
                   "final_error_deg": res.get("final_error_deg")}
            rows.append(row)
            self.pipe.log(f"{'[OK]' if (ok is not False and human) else '[WARN]'} [V3.15 direction] J{j + 1} {s:+.0f}°: "
                          f"预期 {row['pred_mm']} mm · 实测 {row['meas_mm']} mm · {txt} · 人工: {row['human']}")
            self._ui(lambda row=row: self.tv_dir.insert("", "end", values=(
                f"J{row['joint']}", row["pred_mm"], row["meas_mm"], "-" if row["cos"] is None else f"{row['cos']:+.3f}",
                "-" if row["ratio"] is None else f"{row['ratio']:.2f}", row["auto"], row["human"])))
            res2 = self.pipe.move_and_settle(model, [math.radians(x) for x in back], speed, label=f"方向测试 J{j + 1} 回原位")
            if not res2.get("ok"):
                self.pipe.log(f"[FAIL] [V3.15 direction] J{j + 1} 回原位失败: {res2.get('error')}")
                break
        self.results["direction"] = rows
        allok = rows and all(r.get("auto") != "不符" and r.get("human") == "一致" for r in rows if "human" in r)
        path = write_result("direction", {"rows": rows, "speed": speed, "all_ok": bool(allok)})
        self.pipe.log(f"{'[OK]' if allok else '[WARN]'} [V3.15 direction] 完成 {len(rows)} 个关节 → {os.path.basename(path)}")

    # ------------------------------------------------------------- repeatability (worker)
    def on_repeat(self):
        if not self._ready():
            return
        try:
            off = [float(x) for x in self.v_rep_off.get().replace("，", ",").split(",")]
            n = int(self.v_rep_n.get())
            speeds = [float(x) for x in self.v_rep_sp.get().replace("，", ",").split(",") if x.strip()]
            assert len(off) == 7 and 1 <= n <= 100 and speeds and all(1 <= s <= 30 for s in speeds)
            assert max(abs(x) for x in off) <= 30 and max(abs(x) for x in off) > 0
        except Exception:
            self.log("[WARN] [V3.15 repeat] 偏移需 7 个数（|偏移| ≤ 30°）、循环 1–100、速度 1–30%")
            return
        self._run("repeat", lambda: self._repeat(off, n, speeds))

    def _repeat(self, off, n, speeds):
        model = self.jpanel.model
        a = self.pipe.actual_arm_deg(model)
        if a is None:
            return
        A = list(a)
        B = []
        for j, (x, o) in enumerate(zip(A, off)):
            lo, hi = JC.soft_range_deg(*model["limits"][j])
            B.append(max(lo, min(hi, x + o)))
        if not self.ask("重复定位精度", f"A = 当前姿态，B = A + ({', '.join(f'{o:+.0f}' for o in off)})°\n"
                                    f"每个速度 {', '.join(f'{s:.0f}%' for s in speeds)} 往返 {n} 次。确认周围无障碍、手在急停上？"):
            return
        rows, summary = [], []
        for sp in speeds:
            jerr, tfp, times = [], [], []
            for k in range(n):
                if self.stop_flag:
                    break
                t0 = time.time()
                r1 = self.pipe.move_and_settle(model, [math.radians(x) for x in B], self._speed(sp), label=f"重复 B {k + 1}/{n} @{sp:.0f}%")
                r2 = self.pipe.move_and_settle(model, [math.radians(x) for x in A], self._speed(sp), label=f"重复 A {k + 1}/{n} @{sp:.0f}%")
                if not (r1.get("ok") and r2.get("ok")):
                    self.pipe.log(f"[FAIL] [V3.15 repeat] 第 {k + 1} 次失败: {r1.get('error') or r2.get('error')}")
                    self.stop_flag = True
                    break
                times.append(time.time() - t0)
                p = self._tf_link7(time.time())
                jerr.append(r2.get("final_error_deg") or 0.0)
                if p:
                    tfp.append(p)
                rows.append([sp, k + 1, f"{times[-1]:.2f}", f"{jerr[-1]:.3f}"] + [f"{v:.3f}" for v in (r2.get("actual_deg") or [])]
                            + ([f"{v:.5f}" for v in p] if p else ["", "", ""]))
                self._ui(lambda t=f"@{sp:.0f}%: {k + 1}/{n}  A 关节误差 {jerr[-1]:.3f}°": self.v_rep.set(t))
            if not jerr:
                break
            spread = []
            if tfp:
                c = [sum(p[i] for p in tfp) / len(tfp) for i in range(3)]
                spread = [dist_mm(p, c) for p in tfp]
            sj, ss, st_ = stats(jerr), stats(spread), stats(times)
            summary.append({"speed": sp, "cycles": len(jerr), "joint_err_deg": sj, "tcp_spread_mm": ss, "cycle_s": st_})
            self.pipe.log(f"[OK] [V3.15 repeat] @{sp:.0f}% ×{len(jerr)}: 回 A 关节误差 mean {sj['mean']:.3f}° max {sj['max']:.3f}° · "
                          + (f"TF link7 离散 mean {ss['mean']:.2f} mm max {ss['max']:.2f} mm · " if ss["n"] else "无 TF · ")
                          + f"往返 {st_['mean']:.1f} s")
        self.results["repeat"] = summary
        txt = "\n".join(f"@{s['speed']:.0f}% ×{s['cycles']}: 关节误差 mean {s['joint_err_deg']['mean']:.3f}° / max "
                        f"{s['joint_err_deg']['max']:.3f}°   link7 离散 "
                        + (f"mean {s['tcp_spread_mm']['mean']:.2f} / max {s['tcp_spread_mm']['max']:.2f} mm" if s["tcp_spread_mm"]["n"] else "-")
                        + f"   往返 {s['cycle_s']['mean']:.1f} s" for s in summary)
        self._ui(lambda: self.v_rep.set(txt))
        path = write_result("repeat", {"A_deg": A, "B_deg": B, "summary": summary}, rows,
                            ["speed_pct", "cycle", "cycle_s", "joint_err_deg"] + [f"J{i}" for i in range(1, 8)] + ["tf_x", "tf_y", "tf_z"])
        self.pipe.log(f"[INFO] [V3.15 repeat] 结果 → {os.path.basename(path)} (+ .csv)")

    # ------------------------------------------------------------- stop test (worker)
    def on_stop_test(self):
        if not self._ready():
            return
        try:
            j = int(self.v_st_j.get()[1:]) - 1
            amp, sp, tc = float(self.v_st_a.get()), float(self.v_st_s.get()), float(self.v_st_t.get())
            assert 5 <= amp <= 45 and 1 <= sp <= 20 and 0.2 <= tc <= 10
        except Exception:
            self.log("[WARN] [V3.15 stop] 幅度 5–45°、速度 1–20%、cancel 时刻 0.2–10 s")
            return
        self._run("stop_test", lambda: self._stop_test(j, amp, sp, tc))

    def _stop_test(self, j, amp, sp, tc):
        model = self.jpanel.model
        a = self.pipe.actual_arm_deg(model)
        r = JC.step_test_targets(a, j, model["limits"], step=amp)
        if r is None:
            self.pipe.log(f"[WARN] [V3.15 stop] J{j + 1} 没有 {amp:.0f}° 空间")
            return
        out, back, s = r
        if not self.ask("停止测试", f"J{j + 1} 以 {sp:.0f}% 速度向 {s:+.0f}° 转动，{tc:.1f} s 后发送停止 (cancel)。\n"
                                 "之后会询问是否回到起点。确认周围无障碍、手在急停上？"):
            return
        t_start = time.time()
        box = {}
        th = threading.Thread(target=lambda: box.update(res=self.pipe.move_joints(
            model, [math.radians(x) for x in out], self._speed(sp), label=f"停止测试 J{j + 1}")), daemon=True)
        th.start()
        t_end = time.time() + 15
        sf = self.pipe.safety() or {}
        exec_phases = ("执行轨迹", "MONITOR") if (sf.get("fence") and sf.get("path_check")) else ("执行中", "执行轨迹", "MONITOR")
        while time.time() < t_end and self.pipe.move_state.get("phase") not in exec_phases and th.is_alive():
            time.sleep(0.02)
        time.sleep(tc)
        t_cancel = time.time()
        if th.is_alive():
            self.pipe.cancel_move()
        th.join(20)
        time.sleep(1.5)
        m = stop_metrics(self.pipe.js_samples(model, t_start), j, t_cancel)
        res = box.get("res") or {}
        if res.get("ok") and m and abs(m["rest_angle"] - out[j]) < 0.5:
            self.pipe.log(f"[WARN] [V3.15 stop] 运动在 cancel 前已经完成（幅度太小或 cancel 太晚），本次不算")
        if m:
            txt = (f"J{j + 1}: cancel 时 {m['angle_at_cancel']:+.2f}° → 停在 {m['rest_angle']:+.2f}° · 停止距离 "
                   f"{m['overshoot_deg']:.2f}° · 停止时间 " + ("-" if m["stop_time_s"] is None else f"{m['stop_time_s']:.2f} s")
                   + f" · MoveIt: {res.get('error') or ('完成' if res.get('ok') else '-')}")
            self._ui(lambda: self.v_st.set(txt))
            self.pipe.log("[OK] [V3.15 stop] " + txt)
        else:
            self.pipe.log("[WARN] [V3.15 stop] 关节采样不足，无法计算停止距离")
        self.results["stop"] = {"joint": j + 1, "speed_pct": sp, "cancel_after_s": tc, "metrics": m, "moveit": res}
        write_result("stop", self.results["stop"])
        if self.ask("停止测试", f"回到起点（J{j + 1} {back[j]:+.1f}°）？"):
            self.pipe.move_and_settle(model, [math.radians(x) for x in back], self._speed(sp), label=f"停止测试 J{j + 1} 回起点")

# V3.16: the V3.15 SafetyTabUI (number table) is replaced by NERO_V316_Safety_Editor.SafetyEditorUI
