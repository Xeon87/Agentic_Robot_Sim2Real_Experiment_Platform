#!/usr/bin/env python3
"""NERO V3.59 strategy profile: the "control knobs" shared by the MuJoCo simulation and the real robot.

V3.59 STRATEGY PROFILE + CONTROL KNOBS + SIM/REAL SHARED DECISION LAYER

  "One brain, two bodies": the decision layer below (target ranking, slot fill order, IK posture preference,
  gripper widths, hover height, held-check) is plain Python with no MuJoCo / ROS / camera dependency, so the
  simulation (via the Agent advice hook + environment variables) and the real robot (Camera tab) call exactly
  the same functions with exactly the same JSON profile.

  * KNOBS              4 groups (A ranking, B IK posture, C fine motion / gripper, D recovery): the 14 knob items
                       listed in AGENTS.md 8.1 (called "12 knobs" there) = 16 parameters, each with default, optimizer bounds [lo, hi], unit and Chinese description.
  * StrategyProfile    validated values (clamped to bounds), JSON save / load (config/strategies/*.json),
                       optimizer_space() for the V3.60 sim optimizer.
  * decision layer     rank_targets, order_slots, ik_preference_cost / pick_ik_solution, grip widths,
                       held_check, hover_z.
  * sim bridge         to_sim_env(): knob -> existing backend environment variables (the backend itself is not
                       modified: test_v3_44 byte-identity invariants hold); sim_scene_advice() / apply_to_next()
                       are used by NERO_V2_7_1_Agent.py when NERO_V359_STRATEGY_PROFILE points at a profile file.
  * StrategyPanelUI    console tab 「策略旋钮」: sliders, live ranking preview, save / load / reset presets.

Only the standard library is used (tkinter only inside StrategyPanelUI, passed in by the console).
"""
import json
import math
import os
import time

VERSION = "V3.59"
MARKER = "V3.59 STRATEGY PROFILE + CONTROL KNOBS + SIM/REAL SHARED DECISION LAYER"
HERE = os.path.dirname(os.path.realpath(__file__))
SCHEMA = "nero.strategy_profile/1"


def strategy_dir():
    """config/strategies (override with NERO_STRATEGY_DIR, used by the tests)."""
    return os.environ.get("NERO_STRATEGY_DIR") or os.path.join(HERE, "config", "strategies")


def active_snapshot_path():
    """runtime copy of the live profile that the simulation Agent re-reads (by mtime) on every tool call."""
    return os.environ.get("NERO_V359_ACTIVE_PATH") or os.path.join(HERE, "logs", "v359_active_strategy.json")


# =========================================================================== knob definitions
class Knob:
    __slots__ = ("name", "group", "kind", "default", "lo", "hi", "unit", "label", "desc", "choices", "family", "step")

    def __init__(self, name, group, kind, default, lo=None, hi=None, unit="", label="", desc="", choices=(),
                 family=None, step=None):
        self.name, self.group, self.kind, self.default = name, group, kind, default
        self.lo, self.hi, self.unit, self.label, self.desc = lo, hi, unit, label, desc
        self.choices, self.family = tuple(choices), family or name
        self.step = step if step is not None else (1 if kind == "int" else None)

    def clamp(self, v):
        """validate + clamp one value -> (value, warning or None)"""
        if self.kind == "bool":
            if isinstance(v, str):
                b = v.strip().lower() in ("1", "true", "yes", "on", "开", "是")
            else:
                b = bool(v)
            return b, None
        if self.kind == "choice":
            s = str(v)
            if s in self.choices:
                return s, None
            return self.default, f"{self.name}={v!r} 不是可选项 {list(self.choices)}，用默认 {self.default}"
        try:
            x = float(v)
            if not math.isfinite(x):
                raise ValueError(v)
        except (TypeError, ValueError):
            return self.default, f"{self.name}={v!r} 不是数字，用默认 {self.default}"
        warn = None
        if x < self.lo or x > self.hi:
            xc = min(self.hi, max(self.lo, x))
            warn = f"{self.name}={x:g} 超出 [{self.lo:g}, {self.hi:g}]，已截到 {xc:g}"
            x = xc
        if self.kind == "int":
            x = int(round(x))
        return x, warn


GROUPS = [("A", "A 目标选优与排序"), ("B", "B 机械臂 IK 与姿态偏好"), ("C", "C 精细动作与柔顺夹爪"), ("D", "D 容错与重抓恢复")]
FILL_ORDERS = ("numeric", "far_to_near", "left_to_right", "min_joint_move")
FILL_ORDER_LABELS = {"numeric": "numeric 按槽号（V3.04 原行为）", "far_to_near": "far_to_near 由远到近",
                     "left_to_right": "left_to_right 从左到右", "min_joint_move": "min_joint_move 关节移动最小"}

KNOBS = [
    # ---- A: target ranking
    Knob("w_dist", "A", "float", 0.45, 0.0, 1.0, "", "距离优先权重", "离当前 TCP 水平距离越近得分越高"),
    Knob("w_clearance", "A", "float", 0.35, 0.0, 1.0, "", "空旷度权重", "离邻近物体越远越优先（防夹爪碰邻蛋）"),
    Knob("w_yaw_align", "A", "float", 0.20, 0.0, 1.0, "", "长轴顺手度权重", "抓持所需腕部转角越小越优先（直立蛋不区分）"),
    Knob("slot_fill_order", "A", "choice", "numeric", unit="", label="蛋托装填顺序",
         desc="numeric=按槽号（默认，与 V3.04 一致）；far_to_near=先放离基座远的槽；left_to_right=+y（机器人左侧）先放；"
              "min_joint_move=离本次抓取点 J1 转角最小的槽", choices=FILL_ORDERS),
    # ---- B: IK posture preference
    Knob("w_base_joint_penalty", "B", "float", 1.0, 0.1, 2.0, "", "底座大关节移动惩罚", "J1–J3 的移动量乘这个权重（越大越不愿意大幅转底座/肩）"),
    Knob("w_elbow_lift", "B", "float", 0.5, 0.0, 1.0, "", "手肘抬高偏好", "IK 多解时偏好手肘（link4）更高的解，远离桌面与邻物"),
    Knob("w_limit_margin", "B", "float", 0.6, 0.1, 1.5, "", "远离关节限位权重", "离限位 < 20° 的关节受罚；仿真映射为限位裕量 8°×w/0.6"),
    # ---- C: fine motion + compliant gripper
    Knob("hover_height_mm", "C", "float", 60.0, 40.0, 90.0, "mm", "悬停高度", "预抓取 / 预放置时 TCP 在目标上方的高度"),
    Knob("wrist_refine_enable", "C", "bool", True, label="腕部精修开关", desc="悬停后是否用腕部相机做粗到精对准",
         family="wrist_refine"),
    Knob("wrist_refine_tol_mm", "C", "float", 1.5, 0.8, 3.0, "mm", "腕部精修收敛阈值", "水平残差小于此值即认为对准",
         family="wrist_refine"),
    Knob("grip_open_margin_mm", "C", "float", 14.0, 8.0, 22.0, "mm", "下探张开余量", "张开宽度 = 实测蛋宽 + 余量"),
    Knob("grip_close_squeeze_mm", "C", "float", 2.5, 1.5, 4.5, "mm", "闭合夹紧深度", "闭合目标 = 实测蛋宽 − 夹紧深度"),
    Knob("cruise_speed_ratio", "C", "float", 0.5, 0.2, 1.0, "", "空中巡航速度比", "相对实机速度上限（speed_percent）的比例",
         family="speed_ratio"),
    Knob("descent_speed_ratio", "C", "float", 0.286, 0.10, 0.60, "", "近桌面慢速下探比",
         "最后 2 cm 下探的速度比；仿真映射为轨迹放慢倍数 1/比值（0.286 ≈ V3.04 的 ×3.5）", family="speed_ratio"),
    # ---- D: recovery
    Knob("min_held_width_mm", "D", "float", 32.0, 28.0, 36.0, "mm", "抓持成功最小宽度", "闭合后编码器宽度低于此值 = 空抓"),
    Knob("max_retry_per_target", "D", "int", 1, 0, 2, "次", "单目标最大重抓次数", "同一目标原地重抓的上限"),
]
KNOB_BY_NAME = {k.name: k for k in KNOBS}
KNOB_FAMILIES = list(dict.fromkeys(k.family for k in KNOBS))          # 14 knob items (AGENTS.md 8.1 list)
assert len(KNOB_FAMILIES) == 14 and len(KNOBS) == 16


def default_values():
    return {k.name: k.default for k in KNOBS}


# =========================================================================== profile
class StrategyProfile:
    """validated knob values + metadata; JSON round trip."""

    def __init__(self, values=None, name="default", scene="fill_egg_holder", notes="", source="default"):
        self.values = default_values()
        self.name, self.scene, self.notes, self.source = str(name or "default"), str(scene or ""), str(notes or ""), source
        self.warnings = []
        if values:
            self.update(values)

    # ---- access
    def __getitem__(self, key):
        return self.values[key]

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        """-> warning string or None (unknown key -> KeyError)"""
        kn = KNOB_BY_NAME[key]
        v, warn = kn.clamp(value)
        self.values[key] = v
        if warn:
            self.warnings.append(warn)
        return warn

    def update(self, values):
        warns = []
        for k, v in dict(values).items():
            if k not in KNOB_BY_NAME:
                warns.append(f"未知旋钮 {k!r} 已忽略")
                continue
            w = self.set(k, v)
            if w:
                warns.append(w)
        self.warnings.extend(w for w in warns if w not in self.warnings)
        return warns

    def reset(self):
        self.values = default_values()
        self.warnings = []

    def copy(self, **meta):
        p = StrategyProfile(dict(self.values), self.name, self.scene, self.notes, self.source)
        for k, v in meta.items():
            setattr(p, k, v)
        return p

    def diff_from_default(self):
        d = default_values()
        return {k: v for k, v in self.values.items() if v != d[k]}

    # ---- serialisation
    def to_dict(self):
        return {"schema": SCHEMA, "version": VERSION, "name": self.name, "scene": self.scene, "notes": self.notes,
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "knobs": {k.name: self.values[k.name] for k in KNOBS},
                "bounds": {k.name: [k.lo, k.hi] for k in KNOBS if k.kind in ("float", "int")}}

    @classmethod
    def from_dict(cls, d, source="dict"):
        if not isinstance(d, dict):
            raise ValueError("策略文件内容不是 JSON 对象")
        knobs = d.get("knobs", d)              # also accept a flat {knob: value} dict
        if not isinstance(knobs, dict):
            raise ValueError("策略文件缺少 knobs 对象")
        knobs = {k: v for k, v in knobs.items() if k not in ("schema", "version", "name", "scene", "notes",
                                                             "saved_at", "bounds")}
        return cls(knobs, d.get("name", "unnamed"), d.get("scene", "fill_egg_holder"), d.get("notes", ""), source)

    def save(self, path=None):
        path = path or os.path.join(strategy_dir(), f"{safe_name(self.name)}.json")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f), source=os.path.abspath(path))

    def summary(self):
        return " ".join(f"{k}={_fmt(v)}" for k, v in self.values.items())


def safe_name(name):
    s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name).strip())
    return s.strip("_") or "profile"


def _fmt(v):
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def list_presets(directory=None):
    """-> sorted list of (name, path) for config/strategies/*.json"""
    d = directory or strategy_dir()
    try:
        names = sorted(fn for fn in os.listdir(d) if fn.endswith(".json") and not fn.startswith("."))
    except OSError:
        return []
    return [(fn[:-5], os.path.join(d, fn)) for fn in names]


def optimizer_space():
    """V3.60 hook: the numeric/categorical search space [(name, kind, lo, hi, choices)]"""
    out = []
    for k in KNOBS:
        if k.kind in ("float", "int"):
            out.append((k.name, k.kind, k.lo, k.hi, ()))
        elif k.kind == "choice":
            out.append((k.name, k.kind, None, None, k.choices))
        else:
            out.append((k.name, k.kind, None, None, (False, True)))
    return out


# =========================================================================== shared decision layer
DIST_SCALE_M = 0.30             # s_dist = exp(-d / 0.30 m)
CLEAR_FULL_MM = 60.0            # clearance >= 60 mm -> s_clear = 1
FINGER_PAD_MM = 8.0             # finger pad thickness used for the "tight" flag
DEFAULT_EGG_WIDTH_MM = 42.0     # user's measured egg: 55 x 42 mm
LIMIT_SOFT_RAD = math.radians(20.0)


def _xy(p):
    return float(p[0]), float(p[1])


def _wrap180(deg):
    return (float(deg) + 180.0) % 360.0 - 180.0


def long_axis_yaw_from_quat(q_xyzw, axis="z"):
    """yaw (deg) of the body's long axis projected on the table, or None when the axis is near vertical (an
    upright egg: every gripper yaw is equally good)."""
    x, y, z, w = [float(v) for v in q_xyzw]
    if axis == "z":
        ax = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    else:
        ax = (1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y))
    h = math.hypot(ax[0], ax[1])
    if h < 0.5:                                        # tilted < 30 deg from vertical: treat as upright
        return None
    return math.degrees(math.atan2(ax[1], ax[0]))


def _weights(profile):
    w = [max(0.0, float(profile["w_dist"])), max(0.0, float(profile["w_clearance"])), max(0.0, float(profile["w_yaw_align"]))]
    s = sum(w)
    return [1 / 3.0] * 3 if s <= 1e-9 else [v / s for v in w]


def rank_targets(objects, tcp_xyz, profile, base_xy=(0.0, 0.0), neighbors=None):
    """Target ranking (group A).  objects: [{"id", "xyz_m", optional "width_mm", "yaw_deg" (long axis, base frame)}]
    neighbors: extra obstacles [{"xyz_m", "width_mm"}] (default: the other objects).  -> list sorted best first:
    {"id", "rank", "score", "s_dist", "s_clear", "s_yaw", "dist_mm", "clearance_mm", "yaw_delta_deg", "tight"}"""
    wd, wc, wy = _weights(profile)
    items = [o for o in objects if o.get("xyz_m") is not None]
    tx, ty = _xy(tcp_xyz)
    need = 0.5 * float(profile["grip_open_margin_mm"]) + FINGER_PAD_MM
    out = []
    for o in items:
        ox, oy = _xy(o["xyz_m"])
        d = math.hypot(ox - tx, oy - ty)
        s_dist = math.exp(-d / DIST_SCALE_M)
        w_o = float(o.get("width_mm") or DEFAULT_EGG_WIDTH_MM)
        others = [n for n in items if n is not o] + list(neighbors or [])
        clear = None
        for n in others:
            nx, ny = _xy(n["xyz_m"])
            c = math.hypot(nx - ox, ny - oy) * 1000.0 - 0.5 * w_o - 0.5 * float(n.get("width_mm") or DEFAULT_EGG_WIDTH_MM)
            clear = c if clear is None else min(clear, c)
        s_clear = 1.0 if clear is None else min(1.0, max(0.0, clear / CLEAR_FULL_MM))
        yaw = o.get("yaw_deg")
        if yaw is None:
            s_yaw, dyaw = 1.0, None
        else:
            radial = math.degrees(math.atan2(oy - base_xy[1], ox - base_xy[0]))
            # two-finger gripper: fingers straddle the short axis, so the jaw line follows the long axis; the
            # natural (zero wrist-twist) jaw line is radial.  180 deg symmetric -> wrap to [-90, 90].
            dyaw = _wrap180(2.0 * _wrap180(float(yaw) - radial)) / 2.0
            s_yaw = math.cos(math.radians(dyaw)) ** 2
        score = wd * s_dist + wc * s_clear + wy * s_yaw
        out.append({"id": o.get("id"), "score": round(score, 4), "s_dist": round(s_dist, 3), "s_clear": round(s_clear, 3),
                    "s_yaw": round(s_yaw, 3), "dist_mm": round(d * 1000.0, 1),
                    "clearance_mm": (None if clear is None else round(clear, 1)),
                    "yaw_delta_deg": (None if dyaw is None else round(dyaw, 1)),
                    "tight": bool(clear is not None and clear < need)})
    out.sort(key=lambda r: (-r["score"], str(r["id"])))
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return out


def _slot_num(name):
    try:
        return int(str(name).split("_")[-1])
    except ValueError:
        return 10 ** 6


def order_slots(slots, profile, ref_xyz=None, base_xy=(0.0, 0.0)):
    """Slot fill order (group A).  slots: [{"name", "xyz_m", "occupied"}] -> empty slot names in fill order.
    ref_xyz: where the egg is picked (for min_joint_move; default: the base front)."""
    mode = profile["slot_fill_order"]
    empty = [s for s in slots if not s.get("occupied")]
    bx, by = float(base_xy[0]), float(base_xy[1])
    if mode == "numeric" or not empty or any(s.get("xyz_m") is None for s in empty):
        return [s["name"] for s in sorted(empty, key=lambda s: _slot_num(s["name"]))]
    if mode == "far_to_near":
        key = lambda s: (-math.hypot(s["xyz_m"][0] - bx, s["xyz_m"][1] - by), _slot_num(s["name"]))
    elif mode == "left_to_right":            # REP-103 base frame: +y is the robot's left
        key = lambda s: (-round(float(s["xyz_m"][1]), 3), -math.hypot(s["xyz_m"][0] - bx, s["xyz_m"][1] - by),
                         _slot_num(s["name"]))
    else:                                    # min_joint_move: smallest J1 swing from the pick point, then distance
        rx, ry = _xy(ref_xyz) if ref_xyz is not None else (bx + 0.4, by)
        a_ref = math.atan2(ry - by, rx - bx)

        def key(s):
            a = math.atan2(s["xyz_m"][1] - by, s["xyz_m"][0] - bx)
            return (round(abs(_wrap180(math.degrees(a - a_ref))), 1),
                    math.hypot(s["xyz_m"][0] - rx, s["xyz_m"][1] - ry), _slot_num(s["name"]))
    return [s["name"] for s in sorted(empty, key=key)]


def ik_preference_cost(q_rad, q_now_rad, limits_rad, profile, elbow_z_fn=None):
    """IK posture preference (group B): lower is better.
    cost = w_base * sum|dq J1-J3| + 0.3 * sum|dq J4-J7| + w_limit * sum((20deg - margin)+/20deg)^2 - w_elbow * z_elbow/0.1m"""
    dq = [abs(float(a) - float(b)) for a, b in zip(q_rad, q_now_rad)]
    base = sum(dq[:3])
    distal = sum(dq[3:])
    lim = 0.0
    for v, (lo, hi) in zip(q_rad, limits_rad):
        m = min(float(v) - float(lo), float(hi) - float(v))
        if m < LIMIT_SOFT_RAD:
            lim += ((LIMIT_SOFT_RAD - m) / LIMIT_SOFT_RAD) ** 2
    elbow = 0.0
    if elbow_z_fn is not None:
        try:
            elbow = float(elbow_z_fn(q_rad)) / 0.10
        except Exception:
            elbow = 0.0
    return (float(profile["w_base_joint_penalty"]) * base + 0.3 * distal + float(profile["w_limit_margin"]) * lim
            - float(profile["w_elbow_lift"]) * elbow)


def pick_ik_solution(candidates, q_now_rad, limits_rad, profile, elbow_z_fn=None):
    """-> (best_q or None, [costs]) among IK candidate joint vectors (rad)."""
    cands = [list(c) for c in candidates if c is not None]
    if not cands:
        return None, []
    costs = [ik_preference_cost(c, q_now_rad, limits_rad, profile, elbow_z_fn) for c in cands]
    k = min(range(len(cands)), key=lambda i: costs[i])
    return cands[k], costs


def grip_open_width_mm(egg_width_mm, profile, max_mm=100.0):
    return min(float(max_mm), float(egg_width_mm) + float(profile["grip_open_margin_mm"]))


def grip_close_width_mm(egg_width_mm, profile):
    return max(0.0, float(egg_width_mm) - float(profile["grip_close_squeeze_mm"]))


def held_check(measured_width_mm, egg_width_mm, profile):
    """V3.62 preview of the double grasp check (encoder part): ok if min_held <= w <= W_egg + 2 mm."""
    w = float(measured_width_mm)
    lo, hi = float(profile["min_held_width_mm"]), float(egg_width_mm) + 2.0
    if w < lo:
        return False, f"EMPTY_GRASP: 夹爪闭合到 {w:.1f} mm < {lo:.1f} mm（空抓）"
    if w > hi:
        return False, f"NOT_CLOSED: 夹爪停在 {w:.1f} mm > 蛋宽+2 = {hi:.1f} mm（未夹住）"
    return True, f"HELD: {w:.1f} mm ∈ [{lo:.1f}, {hi:.1f}]"


def hover_z(obj_z_m, z_table_m, profile, min_above_table_m=0.05):
    return max(float(obj_z_m) + float(profile["hover_height_mm"]) / 1000.0, float(z_table_m) + min_above_table_m)


def speed_scale(profile, phase="cruise"):
    return float(profile["descent_speed_ratio"] if phase == "descent" else profile["cruise_speed_ratio"])


# =========================================================================== sim bridge (non-invasive)
SIM_ENV_DOC = [
    ("NERO_V304_EGG_APERTURE_MARGIN_MM", "grip_open_margin_mm", "预抓张开 = 蛋宽 + 余量（后端默认 16）"),
    ("NERO_V304_FINAL_APPROACH_SLOW", "descent_speed_ratio", "最后 20 mm 下探轨迹放慢倍数 = 1/比值（后端默认 3.5）"),
    ("NERO_V303_HOVER_ABOVE_ENTRY_M", "hover_height_mm", "入槽前悬停高度（后端默认 0.10 m，内部下限 0.06 m）"),
    ("NERO_V341_LIMIT_MARGIN_DEG", "w_limit_margin", "关节限位裕量 = 8°×w/0.6，截到 [4, 15]°（后端默认 8°）"),
]


def to_sim_env(profile, snapshot_path=None):
    """knob -> environment variables the (unmodified) simulation backend already reads, plus the Agent hook."""
    env = {
        "NERO_V304_EGG_APERTURE_MARGIN_MM": f"{float(profile['grip_open_margin_mm']):.2f}",
        "NERO_V304_FINAL_APPROACH_SLOW": f"{1.0 / max(0.05, float(profile['descent_speed_ratio'])):.3f}",
        "NERO_V303_HOVER_ABOVE_ENTRY_M": f"{float(profile['hover_height_mm']) / 1000.0:.4f}",
        "NERO_V341_LIMIT_MARGIN_DEG": f"{min(15.0, max(4.0, 8.0 * float(profile['w_limit_margin']) / 0.6)):.2f}",
        "NERO_V359_STRATEGY_NAME": profile.name,
    }
    if snapshot_path:
        env["NERO_V359_STRATEGY_PROFILE"] = os.path.abspath(snapshot_path)
    return env


def write_active_snapshot(profile, path=None):
    """atomically write the live profile for the running sim Agent (it re-reads by mtime)."""
    return profile.save(path or active_snapshot_path())


_CACHE = {"path": None, "mtime": None, "profile": None}


def load_cached(path):
    """profile at path, re-read only when the file changed; None if missing / invalid."""
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return None
    if _CACHE["path"] != path or _CACHE["mtime"] != mt:
        try:
            _CACHE.update(path=path, mtime=mt, profile=StrategyProfile.load(path))
        except Exception:
            _CACHE.update(path=path, mtime=mt, profile=None)
    return _CACHE["profile"]


def _scene_objects(scene):
    objs, holder = [], []
    for name, ov in (scene.get("objects") or {}).items():
        if not isinstance(ov, dict) or ov.get("xyz_m") is None:
            continue
        loc = str(ov.get("location") or "")
        gg = ov.get("grasp_geometry") or {}
        ent = {"id": name, "xyz_m": ov["xyz_m"], "width_mm": gg.get("effective_width_mm")}
        if ov.get("quat_xyzw"):
            ent["yaw_deg"] = long_axis_yaw_from_quat(ov["quat_xyzw"])
        if loc.startswith("egg_holder") or loc == "in_gripper":
            holder.append(ent)
        else:
            objs.append(ent)
    return objs, holder


def sim_scene_advice(scene, profile):
    """observe_scene -> {"target_order", "next_target", "slot_order", "next_slot", "scores", ...} using the shared
    decision layer (the same functions the real Camera tab uses)."""
    objs, _held = _scene_objects(scene)
    tcp = ((scene.get("tcp") or {}).get("xyz_m")) or [0.0, 0.0, 0.0]
    ranking = rank_targets(objs, tcp, profile)
    slots = []
    for sn, sv in (((scene.get("receptacles") or {}).get("egg_holder") or {}).get("slots") or {}).items():
        slots.append({"name": sn, "xyz_m": sv.get("center_xyz_m"), "occupied": bool(sv.get("occupied"))})
    first = next((o for o in objs if ranking and o["id"] == ranking[0]["id"]), None)
    slot_order = order_slots(slots, profile, ref_xyz=(first or {}).get("xyz_m"))
    return {"profile": profile.name, "version": VERSION,
            "target_order": [r["id"] for r in ranking], "next_target": ranking[0]["id"] if ranking else None,
            "slot_fill_order": profile["slot_fill_order"], "slot_order": slot_order,
            "next_slot": slot_order[0] if slot_order else None,
            "scores": {r["id"]: {k: r[k] for k in ("score", "dist_mm", "clearance_mm", "tight")} for r in ranking},
            "max_retry_per_target": int(profile["max_retry_per_target"]),
            "rule": "pick eggs in target_order (start with next_target); fill slots in slot_order"}


def apply_to_scene(scene, advice):
    """copy of scene with v359_strategy added and (if the profile is not numeric) egg_holder_fill_order following
    the profile, so the existing V3.04 FILL ORDER prompt rule automatically uses the strategy."""
    s = dict(scene)
    s["v359_strategy"] = advice
    fo = s.get("egg_holder_fill_order")
    if isinstance(fo, dict) and advice.get("slot_fill_order") != "numeric":
        fo = dict(fo)
        fo["next_empty_slot"] = advice.get("next_slot")
        fo["empty_slots_in_order"] = list(advice.get("slot_order") or [])
        fo["rule"] = f"V3.59 strategy profile {advice.get('profile')}: slot_fill_order={advice.get('slot_fill_order')}; use next_empty_slot"
        s["egg_holder_fill_order"] = fo
    return s


def apply_to_next(next306, advice, profile):
    """verify_object_in_slot -> v306_next reordered with the last advice (target order + slot order)."""
    if not isinstance(next306, dict) or "error" in next306 or not advice:
        return next306
    n = dict(next306)
    order = list(advice.get("target_order") or [])
    rem = list(n.get("remaining_eggs_outside_holder") or [])
    rem_sorted = sorted(rem, key=lambda e: (order.index(e) if e in order else len(order), e))
    n["remaining_eggs_outside_holder"] = rem_sorted
    n["next_egg"] = rem_sorted[0] if rem_sorted else None
    if profile["slot_fill_order"] != "numeric":
        empty = set(n.get("empty_slots_in_order") or [])
        so = [s for s in (advice.get("slot_order") or []) if s in empty] + \
             sorted([s for s in empty if s not in (advice.get("slot_order") or [])], key=_slot_num)
        n["empty_slots_in_order"] = so
        n["next_empty_slot"] = so[0] if so else None
    n["v359_strategy"] = advice.get("profile")
    return n


AGENT_RULE = ("\n- V3.59 STRATEGY PROFILE: when observe_scene carries scene.v359_strategy, pick the eggs in "
              "scene.v359_strategy.target_order (start with next_target) instead of numeric egg order; "
              "scene.egg_holder_fill_order and v306_next already follow the profile's slot_fill_order and pick order. "
              "Retry one egg at most v359_strategy.max_retry_per_target extra times before moving to the next egg.\n")


# =========================================================================== demo scene for the preview
# hand-made illustrative table (sim base frame): 3 eggs packed close to the TCP + 2 isolated eggs farther away, so
# that w_dist vs w_clearance visibly changes the order; slots = the MuJoCo egg holder (V3.42 EGG_SLOT_POSES).
DEMO_SCENE = {
    "tcp": {"xyz_m": [-0.30, 0.0, 0.30]},
    "objects": {
        "egg_1": {"xyz_m": [-0.330, -0.080, 0.030], "location": "table", "grasp_geometry": {"effective_width_mm": 44.0}},
        "egg_2": {"xyz_m": [-0.330, -0.032, 0.031], "location": "table", "grasp_geometry": {"effective_width_mm": 45.6}},
        "egg_3": {"xyz_m": [-0.378, -0.058, 0.030], "location": "table", "grasp_geometry": {"effective_width_mm": 43.4}},
        "egg_4": {"xyz_m": [-0.460, -0.200, 0.031], "location": "table", "grasp_geometry": {"effective_width_mm": 46.2}},
        "egg_5": {"xyz_m": [-0.520, 0.020, 0.030], "location": "table", "grasp_geometry": {"effective_width_mm": 42.0}},
    },
    "receptacles": {"egg_holder": {"slots": {
        f"slot_{i + 1}": {"center_xyz_m": list(p), "occupied": False}
        for i, p in enumerate([(-0.510, 0.123, 0.043), (-0.450, 0.123, 0.043), (-0.390, 0.123, 0.043),
                               (-0.330, 0.123, 0.043), (-0.510, 0.187, 0.043), (-0.450, 0.187, 0.043),
                               (-0.390, 0.187, 0.043), (-0.330, 0.187, 0.043)])}}},
}
DEMO_LABEL = "示例场景（人工布置：3 蛋贴靠在 TCP 附近 + 2 蛋孤立在远处；8 槽 = 仿真蛋托；未连接实机识别结果）"


def real_scene_from_eggs(eggs, tcp_xyz=None, slots=None):
    """Camera-tab eggs ([{"det_id", "xyz_m", "status", "width_mm"?}]) -> scene dict for sim_scene_advice."""
    objs = {}
    for e in eggs or []:
        if not e.get("xyz_m") or e.get("status") == "rejected":
            continue
        w = e.get("width_mm") or e.get("short_mm") or e.get("b_mm")
        if w is None and e.get("size_mm"):               # V3.47 YOLO size [a, b] mm: the short side is the width
            try:
                w = float(min(e["size_mm"][:2]))
            except (TypeError, ValueError):
                w = None
        objs[str(e.get("det_id"))] = {"xyz_m": list(e["xyz_m"]), "location": "table",
                                      "grasp_geometry": {"effective_width_mm": w}}
    sc = {"tcp": {"xyz_m": list(tcp_xyz or [0.3, 0.0, 0.3])}, "objects": objs}
    if slots:
        sc["receptacles"] = {"egg_holder": {"slots": slots}}
    return sc


def format_advice(advice, profile):
    lines = [f"[V3.59 策略] 预设 {profile.name}: 抓取顺序 = {' → '.join(advice['target_order']) or '（无目标）'}"]
    for oid in advice["target_order"]:
        sc = advice["scores"][oid]
        cl = "-" if sc["clearance_mm"] is None else f"{sc['clearance_mm']:.0f} mm"
        lines.append(f"    {oid:<10} 得分 {sc['score']:.3f} | 距 TCP {sc['dist_mm']:.0f} mm | 邻物净空 {cl}"
                     + (" | ⚠ 最近邻净空 < 单侧张开需求（需选对抓取偏航方向）" if sc["tight"] else ""))
    if advice.get("slot_order"):
        lines.append(f"    装填顺序（{advice['slot_fill_order']}）: {' → '.join(advice['slot_order'])}")
    lines.append(f"    悬停 {profile['hover_height_mm']:.0f} mm | 张开 = 蛋宽 + {profile['grip_open_margin_mm']:.1f} mm | "
                 f"闭合 = 蛋宽 − {profile['grip_close_squeeze_mm']:.1f} mm | 巡航 {profile['cruise_speed_ratio']:.2f} | "
                 f"下探 {profile['descent_speed_ratio']:.3f}")
    return lines


# =========================================================================== console panel
class StrategyPanelUI:
    """V3.59 console tab 「策略旋钮」.  on_change(profile) is called (Tk thread) after every knob change.
    scene_fn() -> (scene_dict, source_label) or None for the ranking preview (default: DEMO_SCENE)."""

    def __init__(self, tk, ttk, parent, log=print, on_change=None, scene_fn=None, ask=None):
        self.tk, self.ttk, self.parent, self.log = tk, ttk, parent, log
        self.on_change, self.scene_fn, self.ask = on_change, scene_fn, ask
        self.profile = StrategyProfile()
        self.vars = {}
        self._loading = False
        self._snap_after = None
        self.v_sim_inject = tk.BooleanVar(value=False)
        self.v_real_follow = tk.BooleanVar(value=False)
        self.v_name = tk.StringVar(value="default")
        self.v_preset = tk.StringVar(value="")
        self.v_status = tk.StringVar(value="默认策略（未修改）")
        self._build()
        self.refresh_presets()
        self._sync_vars()

    # ---- layout
    def _build(self):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(self.parent)
        top.pack(fill="x")
        ttk.Label(top, text=f"{VERSION} 策略旋钮与预设管理 — 仿真与实机共读同一份 config/strategies/*.json",
                  font=("TkDefaultFont", 10, "bold")).pack(side="left")
        ttk.Label(top, textvariable=self.v_status, foreground="#555555").pack(side="right")

        bar = ttk.Frame(self.parent)
        bar.pack(fill="x", pady=4)
        ttk.Label(bar, text="预设:").pack(side="left")
        self.cb_preset = ttk.Combobox(bar, textvariable=self.v_preset, values=[], state="readonly", width=24)
        self.cb_preset.pack(side="left", padx=(2, 4))
        ttk.Button(bar, text="读取预设", command=self.on_load).pack(side="left")
        ttk.Label(bar, text="   名称:").pack(side="left")
        ttk.Entry(bar, textvariable=self.v_name, width=18).pack(side="left", padx=2)
        ttk.Button(bar, text="保存策略预设", command=self.on_save).pack(side="left", padx=2)
        ttk.Button(bar, text="恢复默认", command=self.on_reset).pack(side="left", padx=2)
        ttk.Button(bar, text="预览当前桌面目标评分排序", command=self.on_preview).pack(side="left", padx=8)

        opt = ttk.Frame(self.parent)
        opt.pack(fill="x", pady=(0, 4))
        ttk.Checkbutton(opt, text="仿真启动时注入此策略（张开余量 / 下探速度 / 入槽悬停 / 限位裕量 + Agent 抓取与装填顺序）",
                        variable=self.v_sim_inject, command=self._on_link_toggle).pack(side="left")
        ttk.Checkbutton(opt, text="实机 Camera 页跟随策略（悬停高度 + 目标排序 + IK 姿态偏好）",
                        variable=self.v_real_follow, command=self._on_link_toggle).pack(side="left", padx=12)

        body = ttk.Frame(self.parent)
        body.pack(fill="both", expand=True)
        grid = ttk.Frame(body)
        grid.pack(side="left", fill="y", anchor="n")
        r = 0
        for gk, glabel in GROUPS:
            ttk.Label(grid, text=glabel, font=("TkDefaultFont", 9, "bold")).grid(row=r, column=0, columnspan=4,
                                                                                sticky="w", pady=(6, 1))
            r += 1
            for kn in [k for k in KNOBS if k.group == gk]:
                self._knob_row(grid, r, kn)
                r += 1
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(10, 0))
        ttk.Label(right, text="评分排序预览 / 映射说明").pack(anchor="w")
        self.txt = tk.Text(right, height=18, width=70, wrap="word", font=("TkFixedFont", 9))
        self.txt.pack(fill="both", expand=True)
        self.txt.configure(state="disabled")

    def _knob_row(self, grid, r, kn):
        tk, ttk = self.tk, self.ttk
        ttk.Label(grid, text=f"{kn.label}").grid(row=r, column=0, sticky="w", padx=(8, 4))
        if kn.kind == "bool":
            v = tk.BooleanVar(value=bool(kn.default))
            ttk.Checkbutton(grid, variable=v, command=lambda n=kn.name: self._on_var(n)).grid(row=r, column=1, sticky="w")
        elif kn.kind == "choice":
            v = tk.StringVar(value=str(kn.default))
            cb = ttk.Combobox(grid, textvariable=v, values=list(kn.choices), state="readonly", width=16)
            cb.grid(row=r, column=1, sticky="w")
            cb.bind("<<ComboboxSelected>>", lambda _e, n=kn.name: self._on_var(n))
        else:
            v = tk.StringVar(value=_fmt(kn.default))
            sc = ttk.Scale(grid, from_=kn.lo, to=kn.hi, orient="horizontal", length=170,
                           command=lambda s, n=kn.name: self._on_scale(n, s))
            sc.grid(row=r, column=1, sticky="w")
            e = ttk.Entry(grid, textvariable=v, width=7)
            e.grid(row=r, column=2, sticky="w", padx=3)
            e.bind("<Return>", lambda _e, n=kn.name: self._on_var(n))
            e.bind("<FocusOut>", lambda _e, n=kn.name: self._on_var(n))
            self.vars[kn.name + "#scale"] = sc
        self.vars[kn.name] = v
        rng = f"[{kn.lo:g}, {kn.hi:g}] {kn.unit}" if kn.kind in ("float", "int") else ""
        ttk.Label(grid, text=f"{kn.name} {rng}", foreground="#666666").grid(row=r, column=3, sticky="w", padx=4)

    # ---- value plumbing
    def _sync_vars(self):
        self._loading = True
        try:
            for kn in KNOBS:
                v = self.profile[kn.name]
                var = self.vars.get(kn.name)
                if var is None:
                    continue
                var.set(v if kn.kind == "bool" else _fmt(v))
                sc = self.vars.get(kn.name + "#scale")
                if sc is not None:
                    try:
                        sc.set(float(v))
                    except Exception:
                        pass
            self.v_name.set(self.profile.name)
        finally:
            self._loading = False
        self._update_status()

    def _on_scale(self, name, s):
        if self._loading:
            return
        kn = KNOB_BY_NAME[name]
        try:
            x = float(s)
        except (TypeError, ValueError):
            return
        if kn.kind == "int":
            x = int(round(x))
        else:
            x = round(x, 3 if kn.hi <= 2.0 else 1)
        self.vars[name].set(_fmt(x))
        self._apply(name, x, sync_scale=False)

    def _on_var(self, name):
        if self._loading:
            return
        self._apply(name, self.vars[name].get())

    def _apply(self, name, value, sync_scale=True):
        old = self.profile[name]
        warn = self.profile.set(name, value)
        if warn:
            self.log(f"[WARN] [V3.59 策略] {warn}")
        new = self.profile[name]
        if sync_scale or warn:
            self._loading = True
            try:
                kn = KNOB_BY_NAME[name]
                self.vars[name].set(new if kn.kind == "bool" else _fmt(new))
                sc = self.vars.get(name + "#scale")
                if sc is not None:
                    sc.set(float(new))
            finally:
                self._loading = False
        if new != old:
            self.profile.source = "edited"
            self._changed(f"{name}: {_fmt(old)} → {_fmt(new)}")

    def _changed(self, what=None):
        self._update_status()
        if what:
            self.log(f"[INFO] [V3.59 策略] {what}")
        self._schedule_snapshot()
        if self.on_change is not None:
            try:
                self.on_change(self.profile)
            except Exception as exc:
                self.log(f"[WARN] [V3.59 策略] 应用到其他面板失败：{exc}")
        self.on_preview(quiet=True)

    def _update_status(self):
        d = self.profile.diff_from_default()
        self.v_status.set(f"预设 {self.profile.name} | 改动 {len(d)} 项" + (f": {', '.join(sorted(d))[:80]}" if d else "（与默认一致）"))

    def _schedule_snapshot(self):
        """debounced write of logs/v359_active_strategy.json so a running sim Agent picks up the change."""
        try:
            if self._snap_after is not None:
                self.parent.after_cancel(self._snap_after)
            self._snap_after = self.parent.after(300, self._write_snapshot)
        except Exception:
            self._write_snapshot()

    def _write_snapshot(self):
        self._snap_after = None
        try:
            write_active_snapshot(self.profile)
        except Exception as exc:
            self.log(f"[WARN] [V3.59 策略] 写运行快照失败：{exc}（原因：logs/ 不可写；下一步：检查文件夹权限）")

    def _on_link_toggle(self):
        self.log(f"[INFO] [V3.59 策略] 仿真注入={'开' if self.v_sim_inject.get() else '关'}，"
                 f"实机跟随={'开' if self.v_real_follow.get() else '关'}")
        if self.on_change is not None:
            try:
                self.on_change(self.profile)
            except Exception as exc:
                self.log(f"[WARN] [V3.59 策略] 应用到其他面板失败：{exc}")

    # ---- console hooks
    def sim_inject(self):
        return bool(self.v_sim_inject.get())

    def real_follow(self):
        return bool(self.v_real_follow.get())

    def sim_env(self):
        """env for start_nero_sim.sh ({} when injection is off -> sim behaves exactly as V3.58)."""
        if not self.sim_inject():
            return {}
        path = write_active_snapshot(self.profile)
        return to_sim_env(self.profile, snapshot_path=path)

    # ---- buttons
    def refresh_presets(self):
        names = [n for n, _p in list_presets()]
        try:
            self.cb_preset.configure(values=names)
        except Exception:
            pass
        if names and not self.v_preset.get():
            self.v_preset.set(names[0])
        return names

    def on_load(self, name=None):
        name = name or self.v_preset.get()
        path = dict(list_presets()).get(name)
        if not path:
            self.log(f"[WARN] [V3.59 策略] 没有预设 {name!r}（原因：config/strategies/ 里没有该文件；下一步：先「保存策略预设」）")
            return False
        try:
            p = StrategyProfile.load(path)
        except Exception as exc:
            self.log(f"[FAIL] [V3.59 策略] 读取 {path} 失败：{exc}（下一步：检查 JSON 格式，或「恢复默认」后重新保存）")
            return False
        self.profile = p
        self._sync_vars()
        for w in p.warnings:
            self.log(f"[WARN] [V3.59 策略] {w}")
        self.log(f"[OK] [V3.59 策略] 已读取预设 {p.name}（{path}），与默认不同 {len(p.diff_from_default())} 项")
        self._changed()
        return True

    def on_save(self):
        name = safe_name(self.v_name.get())
        path = os.path.join(strategy_dir(), f"{name}.json")
        if os.path.exists(path) and self.ask is not None and not self.ask("覆盖预设", f"预设 {name}.json 已存在，覆盖？"):
            return None
        self.profile.name = name
        try:
            out = self.profile.save(path)
        except Exception as exc:
            self.log(f"[FAIL] [V3.59 策略] 保存失败：{exc}（下一步：检查 config/strategies/ 写权限）")
            return None
        self.profile.source = out
        self.refresh_presets()
        self.v_preset.set(name)
        self._update_status()
        self._write_snapshot()
        self.log(f"[OK] [V3.59 策略] 已保存预设 {out}（仿真与实机都可读取）")
        return out

    def on_reset(self):
        self.profile = StrategyProfile(name="default")
        self._sync_vars()
        self.log("[OK] [V3.59 策略] 已恢复默认旋钮")
        self._changed()

    def on_preview(self, quiet=False):
        scene, src = DEMO_SCENE, DEMO_LABEL
        if self.scene_fn is not None:
            try:
                got = self.scene_fn()
                if got and got[0] and got[0].get("objects"):
                    scene, src = got
            except Exception as exc:
                self.log(f"[WARN] [V3.59 策略] 读取实机识别结果失败：{exc}，改用示例场景")
        advice = sim_scene_advice(scene, self.profile)
        lines = [f"数据来源：{src}"] + format_advice(advice, self.profile)
        lines.append("")
        lines.append("仿真注入映射（勾选「仿真启动时注入」后在下次启动仿真时生效；抓取/装填顺序在运行中也会即时生效）：")
        env = to_sim_env(self.profile)
        for k, knob, doc in SIM_ENV_DOC:
            lines.append(f"    {k}={env[k]}  ← {knob}  ({doc})")
        lines.append("实机：悬停高度、目标排序、IK 姿态偏好在勾选「实机 Camera 页跟随策略」后立即生效（不需要重启）。")
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("end", "\n".join(lines) + "\n")
        self.txt.configure(state="disabled")
        if not quiet:
            for ln in format_advice(advice, self.profile):
                self.log(("[OK] " if ln.startswith("[V3.59") else "") + ln)
        return advice
