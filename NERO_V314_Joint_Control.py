#!/usr/bin/env python3
"""NERO V3.14 JOINT CONTROL: logic for the 实机 tab joint panel (7 joints + gripper). No Tk, no ROS at import.

- robot model: joint limits from the URDF (/robot_description), planning groups + named states from the SRDF
  (/robot_description_semantic); fallback = the simulation limit table (same NERO URDF the sim uses)
- soft limits (3 deg inside the real limit), entry parsing / clamping, near-limit warnings
- move preview (per-joint delta, confirmation rule), single-joint small-step test targets
- panel / gripper enable rules (reason strings for the GUI)
- pose library: zero pose + SRDF named states + user poses in config/real_poses.json
- Tk panel (JointPanelUI) — tkinter is imported by the caller and passed in
Motion itself goes through MoveIt (MoveGroup action) in the ROS monitor helper of NERO_V313_Real_Robot.py.
"""
import json
import math
import os
import re
import threading
import time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.realpath(__file__))
POSES_PATH = os.path.join(HERE, "config", "real_poses.json")
PARK_PATH = os.path.join(HERE, "config", "park_pose.json")
# V3.24 PARK POSE D: the best pose before disabling / powering off (folded over the base, recorded by the user on the
# real NERO 2026-10-24). Some joints are beyond the URDF limits (reached by hand) -> MoveIt goes to the nearest
# reachable pose inside the soft limits (limit − 3°).
PARK_DEFAULT_DEG = [0.1, -100.6, 1.2, 126.0, -0.1, -0.2, 97.3]

ARM_JOINTS_DEFAULT = [f"joint{i}" for i in range(1, 8)]
# Same values as JOINT_LIMITS_RAD in the sim backend / config/nero.urdf (used only if the URDF cannot be read)
FALLBACK_LIMITS_RAD = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75),
                       (-0.73, 0.95), (-1.5707963, 1.5707963)]
FALLBACK_GRIPPER_MM = (0.0, 100.0)

SOFT_MARGIN_DEG = float(os.environ.get("NERO_V314_SOFT_MARGIN_DEG", "3.0"))
NEAR_LIMIT_DEG = 5.0
CONFIRM_MAX_DELTA_DEG = 30.0
CONFIRM_TOTAL_DELTA_DEG = 90.0
STEP_TEST_DEG = 5.0
JOG_MAX_STEP_DEG = 5.0
DEVIATION_WARN_DEG = 0.5
FEEDBACK_FRESH_S = 1.0
MOVE_TIMEOUT_S = 90.0


# =========================================================================== URDF / SRDF
def parse_urdf(xml_text):
    """-> {"joints": {name: {type, lower, upper, velocity, parent, child, mimic}}, "order": [...], "root": link}
    Raises ValueError on unparsable XML."""
    try:
        root = ET.fromstring(xml_text)
    except Exception as exc:
        raise ValueError(f"URDF parse error: {exc}")
    joints, order, children = {}, [], set()
    links = [l.get("name") for l in root.findall("link")]
    for j in root.findall("joint"):
        name, jt = j.get("name"), j.get("type")
        par = j.find("parent")
        chi = j.find("child")
        lim = j.find("limit")
        mim = j.find("mimic")
        org = j.find("origin")
        ax = j.find("axis")

        def _vec(el, key, default):
            try:
                v = [float(x) for x in (el.get(key) or "").split()] if el is not None else []
                return v if len(v) == 3 else list(default)
            except ValueError:
                return list(default)
        d = {"type": jt, "parent": par.get("link") if par is not None else None,
             "child": chi.get("link") if chi is not None else None, "mimic": mim.get("joint") if mim is not None else None,
             "lower": None, "upper": None, "velocity": None,
             "xyz": _vec(org, "xyz", (0, 0, 0)), "rpy": _vec(org, "rpy", (0, 0, 0)),     # V3.15: kinematics
             "axis": _vec(ax, "xyz", (1, 0, 0))}
        if lim is not None:
            for k in ("lower", "upper", "velocity"):
                if lim.get(k) is not None:
                    d[k] = float(lim.get(k))
        if jt == "continuous":
            d["lower"], d["upper"] = -math.pi, math.pi
        joints[name] = d
        order.append(name)
        if d["child"]:
            children.add(d["child"])
    roots = [l for l in links if l not in children]
    return {"joints": joints, "order": order, "root": roots[0] if roots else None}


def parse_srdf(xml_text):
    """-> {"groups": {name: {"joints": [...], "chains": [(base, tip)], "subgroups": [...]}},
           "states": {group: {state: {joint: value}}}, "end_effectors": [...]}"""
    try:
        root = ET.fromstring(xml_text)
    except Exception as exc:
        raise ValueError(f"SRDF parse error: {exc}")
    groups, states = {}, {}
    for g in root.findall("group"):
        groups[g.get("name")] = {
            "joints": [j.get("name") for j in g.findall("joint")],
            "chains": [(c.get("base_link"), c.get("tip_link")) for c in g.findall("chain")],
            "subgroups": [s.get("name") for s in g.findall("group")],
            "links": [l.get("name") for l in g.findall("link")],
        }
    for s in root.findall("group_state"):
        vals = {}
        for j in s.findall("joint"):
            try:
                vals[j.get("name")] = float(j.get("value"))
            except Exception:
                pass
        states.setdefault(s.get("group"), {})[s.get("name")] = vals
    ees = [{"name": e.get("name"), "group": e.get("group"), "parent_link": e.get("parent_link")}
           for e in root.findall("end_effector")]
    return {"groups": groups, "states": states, "end_effectors": ees}


MOVABLE = ("revolute", "continuous", "prismatic")


def chain_joints(urdf, base, tip):
    """Movable joints from base to tip (URDF order base->tip)."""
    by_child = {d["child"]: n for n, d in urdf["joints"].items()}
    out, link, guard = [], tip, 0
    while link and link != base and guard < 100:
        guard += 1
        jn = by_child.get(link)
        if jn is None:
            return []
        if urdf["joints"][jn]["type"] in MOVABLE:
            out.append(jn)
        link = urdf["joints"][jn]["parent"]
    return list(reversed(out)) if link == base else []


def group_joints(urdf, srdf, gname, depth=0):
    g = (srdf or {}).get("groups", {}).get(gname)
    if not g or depth > 5:
        return []
    out = [j for j in g["joints"] if urdf and j in urdf["joints"] and urdf["joints"][j]["type"] in MOVABLE]
    if urdf:
        for base, tip in g["chains"]:
            out += [j for j in chain_joints(urdf, base, tip) if j not in out]
    for sg in g["subgroups"]:
        out += [j for j in group_joints(urdf, srdf, sg, depth + 1) if j not in out]
    return out


def _tip_link(urdf, joints):
    if not urdf or not joints:
        return None
    return urdf["joints"][joints[-1]]["child"]


def build_model(urdf_xml=None, srdf_xml=None, feedback_names=None, source=""):
    """Everything the panel needs, JSON-able. Never raises: problems go to model["warnings"]."""
    warnings = []
    urdf = srdf = None
    if urdf_xml:
        try:
            urdf = parse_urdf(urdf_xml)
        except ValueError as exc:
            warnings.append(str(exc))
    if srdf_xml:
        try:
            srdf = parse_srdf(srdf_xml)
        except ValueError as exc:
            warnings.append(str(exc))
    fb = list(feedback_names or [])
    arm_group, arm = None, []
    if urdf and srdf:
        cands = []
        for gname in srdf["groups"]:
            js = group_joints(urdf, srdf, gname)
            rev = [j for j in js if urdf["joints"][j]["type"] in ("revolute", "continuous")]
            if len(rev) == 7 and len(js) == 7:
                score = (2 if gname.lower() in ("arm", "nero_arm", "manipulator") else 0) + (1 if "arm" in gname.lower() else 0)
                cands.append((score, gname, js))
        if cands:
            cands.sort(key=lambda c: -c[0])
            _s, arm_group, arm = cands[0]
        else:
            warnings.append("SRDF has no 7-joint revolute group")
    if not arm and urdf:
        rev = [n for n in urdf["order"] if urdf["joints"][n]["type"] in ("revolute", "continuous")
               and not urdf["joints"][n]["mimic"]]
        pref = [n for n in ARM_JOINTS_DEFAULT if n in rev]
        arm = pref if len(pref) == 7 else rev[:7]
    if len(arm) != 7:
        arm = [n for n in ARM_JOINTS_DEFAULT]
        if fb and all(n in fb for n in arm):
            pass
        elif len(fb) >= 7:
            arm = fb[:7]
    limits, vel = [], []
    lim_src = "URDF" if urdf else "仿真表"
    for i, n in enumerate(arm):
        d = urdf["joints"].get(n) if urdf else None
        if d and d["lower"] is not None and d["upper"] is not None:
            limits.append((d["lower"], d["upper"]))
            vel.append(d["velocity"])
        else:
            limits.append(FALLBACK_LIMITS_RAD[i] if i < 7 else (-math.pi, math.pi))
            vel.append(None)
            if urdf:
                lim_src = "URDF+仿真表"
    if not arm_group:
        arm_group = "arm"
        if srdf is None:
            warnings.append("SRDF not available -> planning group assumed 'arm'")
    # gripper: a non-arm group whose joints are prismatic
    grip = None
    if urdf and srdf:
        for gname in srdf["groups"]:
            if gname == arm_group:
                continue
            js = group_joints(urdf, srdf, gname)
            pri = [j for j in js if urdf["joints"][j]["type"] == "prismatic"]
            if pri and len(pri) == len(js):
                d = urdf["joints"][pri[0]]
                n_fingers = len([j for j in urdf["joints"].values() if j["type"] == "prismatic" and j["parent"] == d["parent"]])
                factor = 2.0 if n_fingers >= 2 or any(j["mimic"] == pri[0] for j in urdf["joints"].values()) else 1.0
                lo, hi = d["lower"] or 0.0, d["upper"] or 0.05
                grip = {"group": gname, "joints": [pri[0]], "factor": factor,
                        "range_mm": (round(lo * 1000 * factor, 1), round(hi * 1000 * factor, 1))}
                break
    states = {}
    if srdf:
        for sname, vals in srdf["states"].get(arm_group, {}).items():
            if all(n in vals for n in arm):
                states[sname] = [vals[n] for n in arm]
    tip = None
    base = urdf["root"] if urdf else None
    if urdf and srdf:
        for c in srdf["groups"].get(arm_group, {}).get("chains", []):
            base, tip = c[0] or base, c[1]
    if urdf and not tip:
        tip = _tip_link(urdf, arm)
        if tip and "tcp_link" in {d["child"] for d in urdf["joints"].values()}:
            tip = "tcp_link"
    return {"source": source or ("URDF" if urdf else "fallback"), "limit_source": lim_src, "arm_group": arm_group,
            "joints": arm, "limits": [list(l) for l in limits], "velocity": vel, "gripper": grip,
            "states": states, "base_link": base or "base_link", "tip_link": tip or "link7", "warnings": warnings,
            "has_urdf": urdf is not None, "has_srdf": srdf is not None}


def fallback_model():
    return build_model(None, None, ARM_JOINTS_DEFAULT, source="fallback")


# =========================================================================== numbers
def deg(v):
    return math.degrees(float(v))


def rad(v):
    return math.radians(float(v))


def soft_range_deg(lo_rad, hi_rad, margin=SOFT_MARGIN_DEG):
    lo, hi = deg(lo_rad), deg(hi_rad)
    m = min(margin, (hi - lo) / 4.0)
    return lo + m, hi - m


def load_park(path=None):
    """-> {"deg": [7], "source": str, "saved": str}"""
    try:
        with open(path or PARK_PATH, encoding="utf-8") as f:
            d = json.load(f)
        deg_ = [float(v) for v in d["deg"]]
        if len(deg_) != 7:
            raise ValueError("need 7 angles")
        return {"deg": deg_, "source": d.get("source", "用户"), "saved": d.get("saved", "")}
    except FileNotFoundError:
        return {"deg": list(PARK_DEFAULT_DEG), "source": "默认（2026-10-24 实机记录）", "saved": ""}
    except Exception as exc:
        return {"deg": list(PARK_DEFAULT_DEG), "source": f"默认（{path or PARK_PATH} 读取失败：{exc}）", "saved": ""}


def save_park(deg_, path=None, source="用户记录"):
    deg_ = [round(float(v), 2) for v in deg_]
    if len(deg_) != 7:
        raise ValueError("need 7 angles")
    path = path or PARK_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = {"deg": deg_, "source": source, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(path + ".tmp", path)
    return d


def park_target(deg_, limits_rad):
    """D clamped into the soft limits -> (target_deg, [(joint, recorded, used)])"""
    out, notes = [], []
    for j, (v, (lo, hi)) in enumerate(zip(deg_, limits_rad)):
        a, b = soft_range_deg(lo, hi)
        c = min(max(v, a), b)
        if abs(c - v) > 0.05:
            notes.append((j, v, c))
        out.append(min(max(round(c, 2), a), b))
    return out, notes


def parse_entry(text):
    """'12.5', ' -3', '+4°', '12,5' -> float; anything else -> None."""
    if text is None:
        return None
    s = str(text).strip().replace("°", "").replace("deg", "").replace(",", ".").replace("＋", "+").replace("－", "-")
    if not re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)", s):
        return None
    v = float(s)
    return v if math.isfinite(v) else None


def clamp_deg(v, lo_soft, hi_soft):
    """-> (clamped, was_out_of_range)"""
    if v < lo_soft:
        return lo_soft, True
    if v > hi_soft:
        return hi_soft, True
    return v, False


def near_limit(v_deg, lo_rad, hi_rad, within=NEAR_LIMIT_DEG):
    """-> None or 'upper'/'lower' with distance"""
    lo, hi = deg(lo_rad), deg(hi_rad)
    if hi - v_deg < within:
        return ("upper", hi - v_deg)
    if v_deg - lo < within:
        return ("lower", v_deg - lo)
    return None


def preview(actual_deg, target_deg):
    """-> {"delta": [...], "max": m, "max_joint": i, "total": t, "confirm": bool, "reason": str}"""
    d = [float(t) - float(a) for a, t in zip(actual_deg, target_deg)]
    ad = [abs(x) for x in d]
    m = max(ad) if ad else 0.0
    i = ad.index(m) if ad else -1
    tot = sum(ad)
    reasons = []
    if m > CONFIRM_MAX_DELTA_DEG:
        reasons.append(f"J{i + 1} 变化 {m:.1f}° > {CONFIRM_MAX_DELTA_DEG:.0f}°")
    if tot > CONFIRM_TOTAL_DELTA_DEG:
        reasons.append(f"总变化 {tot:.1f}° > {CONFIRM_TOTAL_DELTA_DEG:.0f}°")
    return {"delta": d, "max": m, "max_joint": i, "total": tot, "confirm": bool(reasons), "reason": "；".join(reasons)}


def step_test_targets(actual_deg, j, limits_rad, step=STEP_TEST_DEG):
    """Small-step test for joint j: move by +step (or -step when that would enter the soft margin), then back.
    -> (out_target, back_target, signed_step) or None when the joint has no room either way."""
    lo_s, hi_s = soft_range_deg(*limits_rad[j])
    a = float(actual_deg[j])
    for s in (step, -step):
        if lo_s <= a + s <= hi_s:
            out = list(actual_deg)
            out[j] = a + s
            return out, list(actual_deg), s
    return None


def final_error(target_deg, actual_deg):
    e = [abs(float(t) - float(a)) for t, a in zip(target_deg, actual_deg)]
    m = max(e) if e else 0.0
    return m, (e.index(m) if e else -1)


# =========================================================================== enable rules
def panel_enable(sim_running, moveit_lamp, model, js_age_s, moving, joints_present):
    """-> (enabled, reason)"""
    if sim_running:
        return False, "仿真正在运行（实机锁定）"
    if moveit_lamp != "green":
        return False, "第 6 步 MoveIt / RViz 未变绿"
    if not model or len(model.get("joints") or []) != 7:
        return False, "未读取到 7 个关节的机器人模型"
    if not joints_present:
        return False, f"/feedback/joint_states 中缺少关节 {model.get('joints')}"
    if js_age_s is None or js_age_s > FEEDBACK_FRESH_S:
        return False, "关节反馈超时（/feedback/joint_states 未更新）"
    if moving:
        return False, "正在执行运动"
    return True, ""


def gripper_enable(cfg, can_old_id, model, gripper_status, panel_ok):
    """-> (enabled, reason)"""
    if can_old_id:
        return False, f"夹爪 ID 仍为 {','.join(can_old_id)}（旧 ID）"
    if (cfg or {}).get("effector_type", "none") == "none":
        return False, "effector=none（驱动未加载夹爪）"
    if not (model or {}).get("gripper"):
        return False, "MoveIt 配置里没有夹爪 planning group"
    if gripper_status is None:
        return False, "没有 /feedback/gripper_status"
    if not panel_ok:
        return False, "关节面板未就绪"
    return True, ""


def gripper_mm_to_joint(mm, grip):
    return float(mm) / 1000.0 / float(grip.get("factor") or 1.0)


def gripper_joint_to_mm(q, grip):
    return float(q) * 1000.0 * float(grip.get("factor") or 1.0)


def extract_gripper_mm(gripper_status, joints_fb, grip):
    """Best effort: gripper joint in joint_states, else a width-like field in gripper_status (m or mm)."""
    if grip and joints_fb:
        names, pos = joints_fb.get("names") or [], joints_fb.get("position") or []
        for jn in grip.get("joints", []):
            if jn in names:
                return gripper_joint_to_mm(pos[names.index(jn)], grip)
    if isinstance(gripper_status, dict):
        for k, v in gripper_status.items():
            if isinstance(v, (int, float)) and any(w in k.lower() for w in ("width", "opening", "distance", "pos")):
                return float(v) * (1000.0 if abs(v) < 0.5 else 1.0)
    return None


# =========================================================================== pose library
class PoseLibrary:
    """Built-ins (zero pose, SRDF named states of the arm group) + user poses in config/real_poses.json."""

    def __init__(self, path=POSES_PATH):
        self.path = path
        self.user = {}
        self.builtin = {"零位 (全 0°)": [0.0] * 7}
        self.load()

    def load(self):
        try:
            with open(self.path, encoding="utf-8") as f:
                d = json.load(f)
            self.user = {str(k): [float(x) for x in v["joints_deg"]] for k, v in d.get("poses", {}).items()
                         if len(v.get("joints_deg", [])) == 7}
        except Exception:
            self.user = {}

    def set_model_states(self, model):
        self.builtin = {"零位 (全 0°)": [0.0] * 7}
        for name, vals in ((model or {}).get("states") or {}).items():
            self.builtin[f"{name} (SRDF)"] = [round(deg(v), 2) for v in vals]

    def names(self):
        return list(self.builtin) + [f"{n} (用户)" for n in self.user]

    def get(self, label):
        if label in self.builtin:
            return list(self.builtin[label])
        if label.endswith(" (用户)") and label[:-5] in self.user:
            return list(self.user[label[:-5]])
        return None

    def save(self, name, joints_deg):
        name = str(name).strip()
        if not name or len(joints_deg) != 7:
            raise ValueError("姿态名为空或关节数不是 7")
        self.user[name] = [round(float(x), 2) for x in joints_deg]
        self._write()

    def delete(self, label):
        if not label.endswith(" (用户)") or label[:-5] not in self.user:
            return False
        del self.user[label[:-5]]
        self._write()
        return True

    def _write(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": "V3.14", "unit": "deg",
                       "poses": {k: {"joints_deg": v, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}
                                 for k, v in self.user.items()}}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)


# =========================================================================== Tk helpers
_LINESPACE = {}


def fixed_label(tk, parent, lines=1, width=None, pack=None, font=("TkDefaultFont", 9), wraplength=None, **kw):
    """V3.16.1 FIXED LABEL: a label inside a holder of FIXED pixel height (lines x linespace) — live text can change
    its length / number of lines without moving anything below it. Extra lines are clipped.
    width (px) fixes the width too (for labels that are not packed with fill="x")."""
    # V3.16.2: query the line height WITHOUT creating a tkinter.font.Font object. A Font object is a named Tk font
    # that is deleted ("font delete") when Python garbage-collects it; on Tk 8.6 / Xft (Ubuntu 22.04) that left a
    # dangling font cache and the next widget creation segfaulted at startup.
    key = (str(font), parent.winfo_toplevel().winfo_id())
    ls = _LINESPACE.get(key)
    if ls is None:
        try:
            ls = int(parent.tk.call("font", "metrics", font, "-linespace"))
        except Exception:
            ls = 16
        _LINESPACE[key] = ls
    bg = kw.pop("bg", None)
    holder = tk.Frame(parent, height=lines * ls + 4, **({"width": width} if width else {}), **({"bg": bg} if bg else {}))
    holder.pack_propagate(False)
    holder.grid_propagate(False)
    if wraplength is None:
        wraplength = (width - 8) if width else 0
    lab = tk.Label(holder, font=font, anchor=kw.pop("anchor", "nw"), justify=kw.pop("justify", "left"), wraplength=wraplength,
                   **({"bg": bg} if bg else {}), **kw)
    lab.pack(fill="both", expand=True)
    pk = dict(pack or {})
    if width is None:
        pk.setdefault("fill", "x")
    holder.pack(**pk)
    lab.holder = holder
    return lab


# =========================================================================== Tk panel
class JointPanelUI:
    """V3.14 JOINT PANEL. Built inside the console's 实机 tab. `pipe` is the RealPipeline; `cfg_fn` the real cfg;
    `log` appends to the real-tab log; `ask(title, text)` -> bool (main thread)."""

    SLIDER_LEN = 230

    def __init__(self, tk, ttk, parent, pipe, cfg_fn, sim_running_fn, log, ask, ask_string, can_snapshot_fn):
        self.tk, self.ttk, self.pipe, self.cfg_fn = tk, ttk, pipe, cfg_fn
        self.sim_running_fn, self.log, self.ask, self.ask_string = sim_running_fn, log, ask, ask_string
        self.can_snapshot_fn = can_snapshot_fn
        self.model = fallback_model()
        self.model_key = None
        self.poses = PoseLibrary()
        self.job = None
        self.first_move_done = False
        self.jog_confirmed = False
        self.enabled = False
        self.grip_enabled = False
        self.targets_dirty = False
        self._prog_val = {}
        self.box = ttk.LabelFrame(parent, text="关节控制（V3.14）")
        self.box.pack(fill="x", pady=4)
        top = ttk.Frame(self.box)
        top.pack(fill="x", padx=6, pady=(2, 0))
        self.v_src = tk.StringVar(value="")
        self.v_state = tk.StringVar(value="")
        fixed_label(tk, top, 1, width=560, textvariable=self.v_src, font=("TkDefaultFont", 9, "bold"), pack={"side": "left"})
        self.lbl_state = fixed_label(tk, top, 1, textvariable=self.v_state, font=("TkDefaultFont", 9, "bold"), fg="#888888",
                                     pack={"side": "left", "padx": 12, "fill": "x", "expand": True})
        self.chans_frame = ttk.Frame(self.box)
        self.chans_frame.pack(fill="x", padx=4, pady=4)
        self.cols = []
        self._build_columns()
        self._build_bar()
        self._build_preview()
        self.refresh_model(force=True)

    # ------------------------------------------------------------- building
    def _build_columns(self):
        tk, ttk = self.tk, self.ttk
        for w in self.chans_frame.winfo_children():
            w.destroy()
        self.cols = []
        for i in range(8):
            self.chans_frame.columnconfigure(i, weight=1, uniform="ch")
            fr = tk.Frame(self.chans_frame, bd=1, relief="groove", bg="#f6f5f2")
            fr.grid(row=0, column=i, sticky="nsew", padx=2)
            c = {"frame": fr, "i": i}
            bg = "#f6f5f2"
            c["title"] = tk.Label(fr, text="", font=("TkDefaultFont", 11, "bold"), bg=bg)
            c["title"].pack()
            c["hi"] = tk.Label(fr, text="", fg="#b03a2e", font=("TkDefaultFont", 9, "bold"), bg=bg)
            c["hi"].pack()
            c["hi_soft"] = tk.Label(fr, text="", fg="#777777", font=("TkDefaultFont", 8), bg=bg)
            c["hi_soft"].pack()
            mid = tk.Frame(fr, bg=bg)
            mid.pack()
            c["canvas"] = tk.Canvas(mid, width=16, height=self.SLIDER_LEN, highlightthickness=0, bg=bg)
            c["canvas"].pack(side="left")
            c["var"] = tk.DoubleVar(value=0.0)
            c["scale"] = tk.Scale(mid, from_=1, to=-1, orient="vertical", variable=c["var"], resolution=0.1,
                                  length=self.SLIDER_LEN, showvalue=False, width=14, sliderlength=22, bg=bg,
                                  highlightthickness=0, command=lambda _v, i=i: self._on_scale(i))
            c["scale"].pack(side="left")
            c["lo_soft"] = tk.Label(fr, text="", fg="#777777", font=("TkDefaultFont", 8), bg=bg)
            c["lo_soft"].pack()
            c["lo"] = tk.Label(fr, text="", fg="#b03a2e", font=("TkDefaultFont", 9, "bold"), bg=bg)
            c["lo"].pack()
            c["evar"] = tk.StringVar(value="0.0")
            c["entry"] = tk.Entry(fr, textvariable=c["evar"], width=8, justify="right", font=("TkFixedFont", 10))
            c["entry"].pack(pady=2)
            c["entry"].bind("<Return>", lambda _e, i=i: self._on_entry(i))
            c["entry"].bind("<FocusOut>", lambda _e, i=i: self._on_entry(i, quiet=True))
            c["act"] = fixed_label(tk, fr, 1, text="实际 -", font=("TkFixedFont", 9), bg=bg, justify="center", anchor="n")
            c["dev"] = fixed_label(tk, fr, 1, text="偏差 -", font=("TkFixedFont", 9), bg=bg, justify="center", anchor="n")
            c["warn"] = fixed_label(tk, fr, 2, text="", fg="#b07d00", font=("TkDefaultFont", 8, "bold"), bg=bg,
                                    wraplength=120, justify="center", anchor="n")
            sb = tk.Frame(fr, bg=bg)
            sb.pack(pady=(1, 4))
            c["steps"] = []
            if i < 7:
                for s in (-5, -1, 1, 5):
                    b = ttk.Button(sb, text=f"{s:+d}", width=3, command=lambda i=i, s=s: self._on_step(i, s))
                    b.pack(side="left")
                    c["steps"].append(b)
            else:
                for t_, fn in (("张开", self._grip_open), ("闭合", self._grip_close)):
                    b = ttk.Button(sb, text=t_, width=4, command=fn)
                    b.pack(side="left")
                    c["steps"].append(b)
            c["actual"] = None
            self.cols.append(c)

    def _build_bar(self):
        tk, ttk = self.tk, self.ttk
        bar = ttk.Frame(self.box)
        bar.pack(fill="x", padx=6, pady=(0, 4))
        self.btn_exec = tk.Button(bar, text="▶ 执行（MoveIt）", bg="#2e7d32", fg="white", activebackground="#1b5e20",
                                  font=("TkDefaultFont", 10, "bold"), command=self.on_execute)
        self.btn_exec.pack(side="left")
        self.btn_plan = ttk.Button(bar, text="只规划（不执行）", command=self.on_plan_only)      # V3.15
        self.btn_plan.pack(side="left", padx=4)
        self.btn_cancel = ttk.Button(bar, text="■ 停止运动", command=self.on_cancel)
        self.btn_cancel.pack(side="left", padx=4)
        self.btn_sync = ttk.Button(bar, text="⟲ 同步为当前", command=self.sync_to_actual)
        self.btn_sync.pack(side="left", padx=4)
        self.v_jog = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="点动模式（±按钮直接小步执行）", variable=self.v_jog, command=self._on_jog).pack(side="left", padx=6)
        ttk.Label(bar, text="小步测试").pack(side="left", padx=(10, 2))
        self.v_test_j = tk.StringVar(value="J1")
        ttk.Combobox(bar, textvariable=self.v_test_j, values=[f"J{i}" for i in range(1, 8)], width=4,
                     state="readonly").pack(side="left")
        self.btn_test = ttk.Button(bar, text=f"±{STEP_TEST_DEG:.0f}° 往返", command=self.on_step_test)
        self.btn_test.pack(side="left", padx=2)
        pr = ttk.Frame(self.box)
        pr.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Label(pr, text="姿态库").pack(side="left")
        self.v_pose = tk.StringVar(value="")
        self.cb_pose = ttk.Combobox(pr, textvariable=self.v_pose, values=self.poses.names(), width=26, state="readonly")
        self.cb_pose.pack(side="left", padx=4)
        ttk.Button(pr, text="载入为目标", command=self.on_pose_load).pack(side="left", padx=2)
        ttk.Button(pr, text="保存当前姿态…", command=self.on_pose_save).pack(side="left", padx=2)
        ttk.Button(pr, text="删除", command=self.on_pose_delete).pack(side="left", padx=2)
        ttk.Button(pr, text="当前设为待断电位置", command=self.on_park_record).pack(side="left", padx=(10, 2))
        self.v_speed = tk.StringVar(value="")
        fixed_label(tk, pr, 1, textvariable=self.v_speed, fg="#555555", pack={"side": "left", "padx": 12, "fill": "x",
                                                                            "expand": True})

    def _build_preview(self):
        tk, ttk = self.tk, self.ttk
        row = ttk.Frame(self.box)
        row.pack(fill="x", padx=6, pady=(0, 6))
        pf = ttk.LabelFrame(row, text="执行预览")
        pf.pack(side="left", fill="both", expand=True)
        grid = ttk.Frame(pf)
        grid.pack(anchor="w", padx=4, pady=2)
        self.pv = {}
        for c_, h in enumerate([""] + [f"J{i}" for i in range(1, 8)]):
            ttk.Label(grid, text=h, font=("TkDefaultFont", 9, "bold"), width=8, anchor="e").grid(row=0, column=c_)
        for r_, h in enumerate(("实际", "目标", "Δ"), start=1):
            ttk.Label(grid, text=h, font=("TkDefaultFont", 9, "bold"), width=8, anchor="e").grid(row=r_, column=0)
            for c_ in range(1, 8):
                lb = tk.Label(grid, text="-", font=("TkFixedFont", 9), width=8, anchor="e")
                lb.grid(row=r_, column=c_)
                self.pv[(h, c_ - 1)] = lb
        self.v_pv = tk.StringVar(value="")
        fixed_label(tk, pf, 2, textvariable=self.v_pv, fg="#555555", wraplength=760, pack={"padx": 4, "pady": (0, 2)})
        tf = ttk.LabelFrame(row, text="TCP 位姿（TF，只读）")
        tf.pack(side="left", fill="both", padx=(6, 0))
        self.v_tcp = tk.StringVar(value="-")
        fixed_label(tk, tf, 3, width=360, textvariable=self.v_tcp, font=("TkFixedFont", 10), pack={"padx": 6, "pady": 4})

    # ------------------------------------------------------------- model / limits
    def refresh_model(self, force=False):
        m = self.pipe.model or None
        key = json.dumps(m, sort_keys=True, default=str) if m else None
        if not force and key == self.model_key:
            return
        self.model_key = key
        if m and len(m.get("joints") or []) == 7:
            self.model = m
            if m.get("warnings"):
                self.log(f"[WARN] [V3.14 joints] model warnings: {m['warnings']}")
            self.log(f"[OK] [V3.14 joints] 限位来源 {m['limit_source']} | group={m['arm_group']} | joints={m['joints']} | "
                     + " ".join(f"J{i + 1} {deg(lo):.1f}~{deg(hi):.1f}°" for i, (lo, hi) in enumerate(m["limits"]))
                     + f" | gripper={m.get('gripper')} | TCP {m['base_link']}→{m['tip_link']}")
        self.poses.set_model_states(self.model)
        self.cb_pose.configure(values=self.poses.names())
        src = self.model["limit_source"] if (m and m.get("has_urdf")) else "仿真表（驱动未运行，未读取 URDF）"
        self.v_src.set(f"限位来源：{src}    group：{self.model['arm_group']}    单位：度 / mm    "
                       f"软限位余量 {SOFT_MARGIN_DEG:.0f}°")
        for i, c in enumerate(self.cols):
            if i < 7:
                lo, hi = self.model["limits"][i]
                ls, hs = soft_range_deg(lo, hi)
                c["title"].configure(text=f"J{i + 1}")
                c["hi"].configure(text=f"上限 {deg(hi):+.1f}°")
                c["lo"].configure(text=f"下限 {deg(lo):+.1f}°")
                c["hi_soft"].configure(text=f"可设到 {hs:+.1f}°")
                c["lo_soft"].configure(text=f"可设到 {ls:+.1f}°")
                c["scale"].configure(from_=hs, to=ls)
                c["lims"] = (ls, hs)
            else:
                g = self.model.get("gripper")
                lo, hi = (g["range_mm"] if g else FALLBACK_GRIPPER_MM)
                c["title"].configure(text="夹爪")
                c["hi"].configure(text=f"张开 {hi:.0f} mm")
                c["lo"].configure(text=f"闭合 {lo:.0f} mm")
                c["hi_soft"].configure(text="（MoveIt 夹爪 group）" if g else "（未读取到夹爪配置）")
                c["lo_soft"].configure(text="")
                c["scale"].configure(from_=hi, to=lo)
                c["lims"] = (lo, hi)
            self._draw_marker(i)

    # ------------------------------------------------------------- per-channel events
    def targets_deg(self):
        return [float(self.cols[i]["var"].get()) for i in range(7)]

    def actual_deg(self):
        return [self.cols[i]["actual"] for i in range(7)]

    def _set_target(self, i, v):
        ls, hs = self.cols[i]["lims"]
        v2, out = clamp_deg(v, ls, hs)
        self._prog_val[i] = round(v2, 1)            # programmatic value: the scale callback must not count it as an edit
        self.cols[i]["var"].set(round(v2, 1))
        self.cols[i]["evar"].set(f"{v2:.1f}")
        return v2, out

    # V3.18 TARGET FOLLOW: until the user edits a target (slider, entry, ±, pose), all targets follow the actual
    # angles — homing after 一键启动, RViz moves, other programs. Editing stops following; a finished motion or
    # 「同步为当前」 resumes it.
    def _mark_edited(self, why=""):
        if not self.targets_dirty:
            self.targets_dirty = True

    def _entry_has_focus(self):
        try:
            w = self.box.focus_get()
        except Exception:
            return False
        return any(w is c["entry"] for c in self.cols)

    def _on_scale(self, i):
        if abs(float(self.cols[i]["var"].get()) - self._prog_val.get(i, 1e9)) > 0.05:
            self._mark_edited("scale")
        self.cols[i]["evar"].set(f"{float(self.cols[i]['var'].get()):.1f}")
        self.cols[i]["entry"].configure(bg="white")
        self._draw_marker(i)

    def _on_entry(self, i, quiet=False):
        c = self.cols[i]
        v = parse_entry(c["evar"].get())
        if v is None:
            c["entry"].configure(bg="#fde8e8")
            if not quiet:
                self.log(f"[WARN] [V3.14 joints] {c['title'].cget('text')}: 输入无效 '{c['evar'].get()}'")
            return
        if abs(v - float(c["var"].get())) > 0.05:
            self._mark_edited("entry")
        v2, out = self._set_target(i, v)
        c["entry"].configure(bg="#fde8e8" if out else "white")
        if out and not quiet:
            self.log(f"[WARN] [V3.14 joints] {c['title'].cget('text')} {v:.1f} 超出可设范围 -> 夹到 {v2:.1f}")
        self._draw_marker(i)

    def _on_step(self, i, s):
        if self.cols[i]["actual"] is None and self.v_jog.get():
            return
        base = float(self.cols[i]["var"].get())
        if self.v_jog.get() and self.cols[i]["actual"] is not None:
            base = self.cols[i]["actual"]
        else:
            self._mark_edited("step")
        self._set_target(i, base + s)
        self._draw_marker(i)
        if self.v_jog.get():
            if abs(s) > JOG_MAX_STEP_DEG:
                return
            self.on_execute(jog=True)

    def _on_jog(self):
        if self.v_jog.get() and not self.jog_confirmed:
            if not self.ask("点动模式", "点动模式下，每按一次 ±1° / ±5° 按钮机械臂就会立即运动一小步（不再逐次确认）。\n"
                                       "确认手放在急停上、周围无障碍？"):
                self.v_jog.set(False)
                return
            self.jog_confirmed = True
        if self.v_jog.get():
            self.sync_to_actual()

    def _draw_marker(self, i):
        c = self.cols[i]
        cv = c["canvas"]
        cv.delete("all")
        L = self.SLIDER_LEN
        pad = 11 + 2
        ls, hs = c.get("lims", (0, 1))
        span = (hs - ls) or 1.0

        def y(v):
            return pad + (hs - max(ls, min(hs, v))) / span * (L - 2 * pad)
        cv.create_rectangle(9, 0, 15, 4, fill="#e8a0a0", outline="")
        cv.create_rectangle(9, L - 4, 15, L, fill="#e8a0a0", outline="")
        if ls < 0 < hs:
            cv.create_line(6, y(0), 16, y(0), fill="#555555")
        a = c["actual"]
        if a is not None:
            yy = y(a)
            cv.create_polygon(1, yy - 6, 1, yy + 6, 12, yy, fill="#2e9e44", outline="#1b5e20")

    # ------------------------------------------------------------- periodic refresh (main thread)
    def poll(self):
        self.refresh_model()
        p = self.pipe
        m = p.mon_status or {}
        js_, js_t = p.feedback_js() if hasattr(p, "feedback_js") else (p.last_js, p.last_js_t)
        js = js_ or {}
        names, pos = js.get("names") or [], js.get("position") or []
        js_age = (time.time() - js_t) if js_t else None
        if len(pos) >= 7 and not all(n in names for n in self.model["joints"]) and getattr(p, "teach_mode", False):
            names = list(self.model["joints"]) + names[7:]       # leader topic with other joint names: by order
        present = all(n in names for n in self.model["joints"])
        for i in range(7):
            n = self.model["joints"][i]
            self.cols[i]["actual"] = deg(pos[names.index(n)]) if (present and js_age is not None and js_age < 5) else None
        moving = p.move_busy()
        snap = self.can_snapshot_fn() or {}
        ok, why = panel_enable(self.sim_running_fn(), p.lamp("moveit"), self.model, js_age, moving, present)
        gok, gwhy = gripper_enable(self.cfg_fn(), snap.get("gripper_old_id"), self.model, m.get("gripper_status"), ok)
        if ok and not self.enabled:
            self.sync_to_actual()                  # entering the panel: never start from stale targets
        if present and not self.targets_dirty and not moving and not self._entry_has_focus():
            for i in range(7):                     # V3.18: follow the actual angles until the user edits
                a = self.cols[i]["actual"]
                if a is not None and abs(float(self.cols[i]["var"].get()) - a) > 0.05:
                    self._set_target(i, a)
        self.enabled, self.grip_enabled = ok, gok
        if moving:
            self.v_state.set(f"● 执行中：{p.move_state.get('label', '')} {p.move_state.get('phase', '')}")
            self.lbl_state.configure(fg="#1565c0")
        elif ok:
            self.v_state.set("● 可操作（第 6 步绿灯 · 关节反馈正常 · 未在执行）")
            self.lbl_state.configure(fg="#2e7d32")
        else:
            self.v_state.set(f"● 不可操作：{why}")
            self.lbl_state.configure(fg="#888888")
        st = "normal" if ok else "disabled"
        for i, c in enumerate(self.cols):
            on = ok if i < 7 else gok
            for w in [c["scale"], c["entry"]] + c["steps"]:
                try:
                    w.configure(state="normal" if on else "disabled")
                except Exception:
                    pass
            bg = "#f6f5f2"
            warn = ""
            if i < 7:
                a = c["actual"]
                t = float(c["var"].get())
                c["act"].configure(text=f"实际 {a:+.1f}°" if a is not None else "实际 -")
                if a is not None:
                    d = t - a
                    c["dev"].configure(text=f"偏差 {d:+.1f}", fg="#b07d00" if abs(d) > DEVIATION_WARN_DEG else "#2e7d32")
                    lo, hi = self.model["limits"][i]
                    nl = near_limit(a, lo, hi) or near_limit(t, lo, hi)
                    if nl:
                        warn = f"距{'上' if nl[0] == 'upper' else '下'}限 {nl[1]:.1f}°"
                        bg = "#fff6d6"
                else:
                    c["dev"].configure(text="偏差 -", fg="#555555")
            else:
                mm = extract_gripper_mm(m.get("gripper_status"), js, self.model.get("gripper"))
                c["act"].configure(text=f"实际 {mm:.1f} mm" if mm is not None else "实际 -")
                c["dev"].configure(text="已锁定" if not gok else "", fg="#888888")
                warn = gwhy if not gok else ""
                bg = "#e4e2de" if not gok else "#f6f5f2"
            c["warn"].configure(text=warn)
            if c["frame"].cget("bg") != bg:
                for w in [c["frame"]] + [x for x in c["frame"].winfo_children()] + \
                        [x for h in c["frame"].winfo_children() for x in h.winfo_children()]:
                    try:
                        w.configure(bg=bg)
                    except Exception:
                        pass
            self._draw_marker(i)
        fence_txt = ""
        if ok and present:                            # V3.16: lock 执行 while the target is outside the safety zone
            try:
                fok, fence_txt, _v = p.fence_goal_check(self.model, [rad(v) for v in self.targets_deg()])
            except Exception as exc:
                fok, fence_txt = True, ""
            if not fok:
                st = "disabled"
                self.v_state.set("● 目标越界（见「工作区安全」页）：" + fence_txt.replace("FENCE: 目标姿态越界 -> 未发送: ", ""))
                self.lbl_state.configure(fg="#c62828")
        self.btn_exec.configure(state=st)
        self.btn_plan.configure(state="normal" if ok else "disabled")
        self.btn_test.configure(state=st)
        self.btn_sync.configure(state="normal" if present else "disabled")
        self.btn_cancel.configure(state="normal" if moving else "disabled")
        cfg = self.cfg_fn()
        self.v_speed.set(("目标：跟随当前角度" if not self.targets_dirty else "目标：已修改，未执行（「同步为当前」恢复跟随）")
                         + f" · 速度 {cfg.get('speed_percent', 10)}%（MoveIt 速度/加速度缩放同值）")
        self._update_preview()
        tcp = m.get("tcp")
        if tcp and tcp.get("xyz"):
            x, y_, z = tcp["xyz"]
            r, pch, yw = tcp["rpy_deg"]
            self.v_tcp.set(f"x {x:+.3f} m   y {y_:+.3f} m   z {z:+.3f} m\nroll {r:+.1f}°  pitch {pch:+.1f}°  yaw {yw:+.1f}°\n"
                           f"{tcp.get('base')} → {tcp.get('tip')}")
        else:
            self.v_tcp.set(f"-\n{(tcp or {}).get('error', '等待 TF')}")

    def _update_preview(self):
        a = self.actual_deg()
        t = self.targets_deg()
        for j in range(7):
            self.pv[("实际", j)].configure(text=f"{a[j]:.1f}" if a[j] is not None else "-")
            self.pv[("目标", j)].configure(text=f"{t[j]:.1f}")
        if any(x is None for x in a):
            for j in range(7):
                self.pv[("Δ", j)].configure(text="-", bg=self.pv[("目标", j)].cget("bg"))
            self.v_pv.set("等待关节反馈")
            return
        pv = preview(a, t)
        base_bg = self.pv[("目标", 0)].cget("bg")
        for j in range(7):
            d = pv["delta"][j]
            self.pv[("Δ", j)].configure(text=f"{d:+.1f}", bg="#ffe9c7" if abs(d) >= 10 else base_bg)
        self.v_pv.set(f"最大 Δ {pv['max']:.1f}° (J{pv['max_joint'] + 1}) · 总 Δ {pv['total']:.1f}°"
                      + (f" · 需确认：{pv['reason']}" if pv["confirm"] else "")
                      + f" · 规则：任一 Δ > {CONFIRM_MAX_DELTA_DEG:.0f}° 或总 Δ > {CONFIRM_TOTAL_DELTA_DEG:.0f}° 执行前确认")

    def sync_to_actual(self):
        self.targets_dirty = False
        for i in range(7):
            a = self.cols[i]["actual"]
            if a is not None:
                self._set_target(i, a)
                self.cols[i]["entry"].configure(bg="white")

    # ------------------------------------------------------------- actions
    def _busy(self):
        return (self.job is not None and self.job.is_alive()) or self.pipe.move_busy()

    def _run(self, name, fn):
        if self._busy():
            self.log(f"[WARN] [V3.14 joints] busy: {self.pipe.move_state.get('label') or name}")
            return
        def wrap():
            try:
                fn()
            except Exception as exc:
                self.pipe.log(f"[FAIL] [V3.14 joints] {name}: {exc}")
        self.job = threading.Thread(target=wrap, daemon=True, name=f"v314_{name}")
        self.job.start()

    def _speed(self):
        s = float(self.cfg_fn().get("speed_percent", 10))
        return max(0.01, min(1.0, s / 100.0))

    def on_execute(self, jog=False):
        if not self.enabled:
            self.log("[WARN] [V3.14 joints] 面板不可操作：" + self.v_state.get())
            return
        a = self.actual_deg()
        if any(x is None for x in a):
            return
        for i in range(7):
            self._on_entry(i, quiet=True)
        t = self.targets_deg()
        pv = preview(a, t)
        if pv["max"] < 0.05:
            self.log("[INFO] [V3.14 joints] 目标与当前一致，无需运动")
            return
        if jog and pv["max"] > JOG_MAX_STEP_DEG + 0.05:
            self.log(f"[WARN] [V3.14 joints] 点动只允许 ≤{JOG_MAX_STEP_DEG:.0f}° 小步（当前 {pv['max']:.1f}°），请先同步为当前")
            return
        msg_lines = [f"J{j + 1}: {a[j]:+.1f}° → {t[j]:+.1f}°  (Δ {pv['delta'][j]:+.1f}°)"
                     for j in range(7) if abs(pv["delta"][j]) >= 0.05]
        need = (pv["confirm"] or not self.first_move_done) and not jog
        speed = self._speed()

        def job():
            if need:
                head = "本次会话第一次实机运动。\n" if not self.first_move_done else ""
                if not self.ask("执行关节运动", head + "\n".join(msg_lines) + f"\n\n{pv['reason'] or ''}\n速度缩放 {speed * 100:.0f}%"
                                "\n确认周围无障碍、手放在急停上？"):
                    self.pipe.log("[INFO] [V3.14 joints] 用户取消")
                    return
            self.first_move_done = True
            self._move_and_report("点动" if jog else "执行", t, speed)
        self._run("execute", job)

    def _move_and_report(self, label, target_deg, speed, plan_only=False):
        """V3.15: goes through RealPipeline.move_and_settle (fence check, optional path check, motion CSV log)."""
        p = self.pipe
        p.log(f"[INFO] [V3.14 joints] {label}: 目标 " + " ".join(f"J{j + 1} {v:+.1f}" for j, v in enumerate(target_deg))
              + f" | 速度 {speed * 100:.0f}% | {'只规划' if plan_only else '规划中…'}")
        res = p.move_and_settle(self.model, [rad(v) for v in target_deg], speed, label=label, plan_only=plan_only)
        if not res or not res.get("ok"):
            p.log(f"[FAIL] [V3.14 joints] {label}: {(res or {}).get('error') or 'no result'}")
            return False
        if plan_only:
            p.log(f"[OK] [V3.14 joints] {label}: 只规划成功 · 规划 {res.get('planning_time', 0):.2f} s · 轨迹 "
                  f"{res.get('trajectory_s', 0):.1f} s / {res.get('n_points', '?')} 点"
                  + (" · 路径围栏通过" if res.get("fence_checked") else "") + "（未执行）")
            return True
        if res.get("final_error_deg") is None:
            p.log(f"[WARN] [V3.14 joints] {label}: 完成，但读不到关节反馈")
            return True
        e, j = res["final_error_deg"], res["final_error_joint"]
        self.targets_dirty = False                  # V3.18: motion done -> targets follow the arm again
        tag = "[OK]" if e <= 1.0 else "[WARN]"
        p.log(f"{tag} [V3.14 joints] {label}: 规划 {res.get('planning_time', 0):.2f} s · 执行 {res.get('elapsed_s', 0):.1f} s"
              f" · 最终误差 max {e:.2f}° (J{j + 1})" + (" · 路径围栏通过" if res.get("fence_checked") else ""))
        return e <= 1.0

    def on_cancel(self):
        self.pipe.cancel_move()

    # V3.19 HOME: MoveIt joint goal to all zeros (preview, confirmation, fence + path check, result in the log);
    # the driver's move_home service only as a fallback when MoveIt is not ready.
    HOME_DEG = [0.0] * 7

    def go_home(self):
        p = self.pipe
        if getattr(p, "teach_mode", False):
            self.log("[WARN] [V3.19 home] NERO 在示教模式：先「退出示教模式」")
            return
        if not self.enabled:
            if p.service_name("move_home") and p.monitor.running():
                if not self.ask("回 Home（驱动服务）", "MoveIt 未就绪，改用驱动的 move_home 服务（无安全区检查、无预览）。\n"
                                                    "确认周围无障碍、手在急停上？"):
                    return
                cid = p.call_service("move_home", timeout_s=60.0)
                self.log(f"[INFO] [V3.19 home] move_home 服务已调用（最长等 60 s，结果见日志）" if cid else
                         "[FAIL] [V3.19 home] move_home 服务不可用")
            else:
                self.log("[WARN] [V3.19 home] 面板不可操作且没有 move_home 服务：" + self.v_state.get())
            return
        a = self.actual_deg()
        if any(x is None for x in a):
            return
        pv = preview(a, self.HOME_DEG)
        if pv["max"] < 0.2:
            self.log("[INFO] [V3.19 home] 已在 Home（全部关节 0°）")
            return
        speed = self._speed()
        lines = [f"J{j + 1}: {a[j]:+.1f}° → 0.0°" for j in range(7) if abs(a[j]) >= 0.1]

        def job():
            if not self.ask("回 Home", "所有关节回到 0°（经 MoveIt 规划，带安全区检查）：\n" + "\n".join(lines)
                            + f"\n\n最大 Δ {pv['max']:.1f}° · 速度 {speed * 100:.0f}%\n确认周围无障碍、手在急停上？"):
                return
            self.first_move_done = True
            self.targets_dirty = True
            ok = self._move_and_report("回 Home", list(self.HOME_DEG), speed)
            if not ok:
                self.pipe.log("[INFO] [V3.19 home] 如果 MoveIt 拒绝（安全区 / 碰撞），先用关节控制把手臂移到开阔处再回 Home")
        self._run("home", job)

    def on_plan_only(self):
        """V3.15: MoveIt plan (+ fence path check) for the current targets; never executes."""
        if not self.enabled:
            return
        for i in range(7):
            self._on_entry(i, quiet=True)
        t = self.targets_deg()
        speed = self._speed()
        self._run("plan_only", lambda: self._move_and_report("只规划", t, speed, plan_only=True))

    def on_step_test(self):
        if not self.enabled:
            return
        j = int(self.v_test_j.get()[1:]) - 1
        a = self.actual_deg()
        r = step_test_targets(a, j, self.model["limits"])
        if r is None:
            self.log(f"[WARN] [V3.14 joints] J{j + 1} 两个方向都没有 {STEP_TEST_DEG:.0f}° 空间")
            return
        out, back, s = r
        speed = self._speed()

        def job():
            if not self.ask("单关节小步测试", f"J{j + 1} 将转动 {s:+.0f}°（{a[j]:+.1f}° → {out[j]:+.1f}°），然后回到原位。\n"
                                          f"速度缩放 {speed * 100:.0f}%。确认周围无障碍、手放在急停上？"):
                return
            self.first_move_done = True
            ok1 = self._move_and_report(f"小步测试 J{j + 1} {s:+.0f}°", out, speed)
            if not ok1 and not self.pipe.move_state.get("ok"):
                self.pipe.log(f"[FAIL] [V3.14 joints] 小步测试 J{j + 1}: 去程失败，不执行回程")
                return
            ok2 = self._move_and_report(f"小步测试 J{j + 1} 回原位", back, speed)
            self.pipe.log(f"{'[OK]' if ok1 and ok2 else '[WARN]'} [V3.14 joints] 小步测试 J{j + 1}: "
                          f"{'通过' if ok1 and ok2 else '有误差 > 1°，请看上面两行'}")
        self._run("step_test", job)

    def on_pose_load(self):
        v = self.poses.get(self.v_pose.get())
        if v is None:
            return
        for i in range(7):
            self._set_target(i, v[i])
        self._mark_edited("pose")
        self.log(f"[INFO] [V3.14 joints] 姿态「{self.v_pose.get()}」已载入为目标（未执行，点「执行」前请看预览）")

    # V3.24 PARK: MoveIt joint goal to the park pose D (before 去使能 / power off)
    def go_park(self):
        p = self.pipe
        if getattr(p, "teach_mode", False):
            self.log("[WARN] [V3.24 park] NERO 在示教模式：先「退出示教模式」")
            return
        if not self.enabled:
            self.log("[WARN] [V3.24 park] 关节面板不可操作（需要 MoveIt 就绪）：" + self.v_state.get())
            return
        a = self.actual_deg()
        if any(x is None for x in a):
            return
        pk = load_park()
        tgt, notes = park_target(pk["deg"], self.model["limits"])
        pv = preview(a, tgt)
        if pv["max"] < 0.3:
            self.log("[INFO] [V3.24 park] 已在待断电位置，可以「去使能」/ 断电")
            return
        speed = self._speed()
        lines = [f"J{j + 1}: {a[j]:+.1f}° → {tgt[j]:+.1f}°" for j in range(7) if abs(a[j] - tgt[j]) >= 0.1]
        clamp = "".join(f"\n· J{j + 1} 记录值 {v:+.1f}° 超出可设范围 → 用 {c:+.1f}°" for j, v, c in notes)

        def job():
            if not self.ask("待断电位置", "关节运动到去使能 / 断电前的最佳位置 D（经 MoveIt 规划，带安全区检查）：\n"
                            + "\n".join(lines) + clamp
                            + f"\n\n最大 Δ {pv['max']:.1f}° · 速度 {speed * 100:.0f}%\n确认周围无障碍、手在急停上？"):
                return
            self.first_move_done = True
            self.targets_dirty = True
            ok = self._move_and_report("待断电位置", tgt, speed)
            if ok:
                self.pipe.log("[OK] [V3.24 park] 已到待断电位置 D：现在可以「去使能」/ 断电"
                              + ("（部分关节按可设范围收了一点，与记录值的差见上）" if notes else ""))
            else:
                self.pipe.log("[INFO] [V3.24 park] " + self.park_zone_hint(tgt))
        self._run("park", job)

    def park_zone_hint(self, tgt_deg):
        """how far behind the base the zone must reach for D (rule A: point + its radius + margin)."""
        try:
            import NERO_V315_Consistency_Safety as CS
            sf = self.pipe.safety() or {}
            f = sf.get("fence")
            urdf = self.pipe.real_urdf or sf.get("sim_urdf")
            if not f or urdf is None:
                raise ValueError("no fence")
            pts = CS.fence_points(urdf, self.model["joints"], [rad(v) for v in tgt_deg], self.model.get("base_link") or
                                  "base_link", f, sf.get("tool_T"))
            name, pt = max(pts.items(), key=lambda kv: kv[1][0])
            need = pt[0] + (CS.point_radius(name, f) + float(f.get("margin", 0.0)) if f.get("rule") == "A" else 0.0)
            return (f"D 姿态最靠后的检查点 {name} 在 x = {pt[0]:+.2f} m（底座后方 {pt[0] * 100:.0f} cm）→ 安全区后边界要到 "
                    f"x ≥ {need:+.2f} m（「工作区安全」俯视图里把底座后方的角点往下拖），否则会被围栏 / MoveIt 围墙拒绝")
        except Exception:
            return "D 在底座上方偏后：安全区要包含底座后方，否则会被围栏 / MoveIt 围墙拒绝；也可先移到开阔处再试"

    def on_park_record(self):
        a = self.actual_deg()
        if any(x is None for x in a):
            self.log("[WARN] [V3.24 park] 没有关节反馈，无法记录")
            return
        old = load_park()
        if not self.ask("当前设为待断电位置", "把当前关节角记为新的待断电位置 D：\n"
                        + " ".join(f"J{j + 1} {v:+.1f}" for j, v in enumerate(a))
                        + "\n\n原来的 D：" + " ".join(f"J{j + 1} {v:+.1f}" for j, v in enumerate(old["deg"])) + "\n继续？"):
            return
        try:
            save_park(a)
        except Exception as exc:
            self.log(f"[FAIL] [V3.24 park] 保存失败：{exc}")
            return
        self.log("[OK] [V3.24 park] 待断电位置 D 已更新 → config/park_pose.json")

    def on_pose_save(self):
        a = self.actual_deg()
        if any(x is None for x in a):
            self.log("[WARN] [V3.14 joints] 没有关节反馈，无法保存当前姿态")
            return
        name = self.ask_string("保存当前姿态", "姿态名称：")
        if not name:
            return
        try:
            self.poses.save(name, a)
        except Exception as exc:
            self.log(f"[FAIL] [V3.14 joints] 保存失败：{exc}")
            return
        self.cb_pose.configure(values=self.poses.names())
        self.log(f"[OK] [V3.14 joints] 姿态「{name}」已保存到 config/real_poses.json")

    def on_pose_delete(self):
        lab = self.v_pose.get()
        if not lab.endswith(" (用户)"):
            self.log("[WARN] [V3.14 joints] 只能删除用户保存的姿态")
            return
        if self.ask("删除姿态", f"删除「{lab}」？") and self.poses.delete(lab):
            self.cb_pose.configure(values=self.poses.names())
            self.v_pose.set("")

    def _grip(self, mm):
        if not self.grip_enabled:
            return
        g = self.model.get("gripper")
        lo, hi = g["range_mm"]
        mm = max(lo, min(hi, float(mm)))
        self.cols[7]["var"].set(mm)
        speed = self._speed()

        def job():
            res = self.pipe.move_gripper(self.model, mm, speed)
            self.pipe.log(f"{'[OK]' if res and res.get('ok') else '[FAIL]'} [V3.14 gripper] {mm:.0f} mm: "
                          f"{(res or {}).get('error') or 'done'}")
        self._run("gripper", job)

    def _grip_open(self):
        self._grip(self.model["gripper"]["range_mm"][1] if self.model.get("gripper") else 0)

    def _grip_close(self):
        self._grip(self.model["gripper"]["range_mm"][0] if self.model.get("gripper") else 0)


# =========================================================================== MoveIt error codes
MOVEIT_ERROR_CODES = {
    1: "SUCCESS", 99999: "FAILURE", -1: "PLANNING_FAILED", -2: "INVALID_MOTION_PLAN",
    -3: "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE", -4: "CONTROL_FAILED", -5: "UNABLE_TO_AQUIRE_SENSOR_DATA",
    -6: "TIMED_OUT", -7: "PREEMPTED", -10: "START_STATE_IN_COLLISION", -11: "START_STATE_VIOLATES_PATH_CONSTRAINTS",
    -12: "GOAL_IN_COLLISION", -13: "GOAL_VIOLATES_PATH_CONSTRAINTS", -14: "GOAL_CONSTRAINTS_VIOLATED",
    -15: "INVALID_GROUP_NAME", -16: "INVALID_GOAL_CONSTRAINTS", -17: "INVALID_ROBOT_STATE", -18: "INVALID_LINK_NAME",
    -19: "INVALID_OBJECT_NAME", -21: "FRAME_TRANSFORM_FAILURE", -22: "COLLISION_CHECKING_UNAVAILABLE",
    -23: "ROBOT_STATE_STALE", -24: "SENSOR_INFO_STALE", -25: "COMMUNICATION_FAILURE", -26: "START_STATE_INVALID",
    -27: "GOAL_STATE_INVALID", -28: "UNRECOGNIZED_GOAL_TYPE", -29: "CRASH", -30: "ABORT", -31: "NO_IK_SOLUTION",
}
MOVEIT_HINTS = {
    -15: "planning group 名称不对：请把 `ros2 param get /move_group robot_description_semantic` 的输出发来",
    -10: "起始姿态被判定为碰撞：检查 RViz 里的碰撞体 / 当前姿态",
    -12: "目标姿态碰撞：换一个目标",
    -4: "控制器执行失败：检查控制门 / 使能 / arm_status",
    -6: "超时：降低目标变化量或检查控制器",
}


def moveit_error_text(code):
    name = MOVEIT_ERROR_CODES.get(int(code), f"code {code}")
    hint = MOVEIT_HINTS.get(int(code))
    return name + (f"（{hint}）" if hint else "")


def quat_to_rpy_deg(x, y, z, w):
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return [round(math.degrees(v), 2) for v in (roll, pitch, yaw)]
