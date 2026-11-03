#!/usr/bin/env python3
"""NERO V3.43 MUJOCO SCENES - scene registry + the RoboDojo-style ``store_in_safe`` scene.

Scenes
  fill_egg_holder  (default, "A1")  the existing egg scene, built by the unchanged backend code
  store_in_safe    ("A2")           open a small table safe, put a valuables box inside, close the door

Selection (V3.44): the scene is chosen BEFORE the simulation starts - the drop-down in the NERO console's
'仿真 (MuJoCo)' tab passes NERO_SCENE=<name> to start_nero_sim.sh (direct runs: set NERO_SCENE yourself; unset =
fill_egg_holder).  There is no switching while the simulation runs: stop it, choose another scene, start again.
The backend exports the scene once in main() as NERO_V343_ACTIVE_SCENE so the GUI process and the spawned MuJoCo
process always use the same scene.

Only the per-scene SETUP data lives here (MJCF, MoveIt boxes, TF / hold names, protected geoms).  The MuJoCo
per-step physics loop ("B") is not touched by any scene.

store_in_safe geometry (robot base at the origin, table top z = 0, the robot faces -x):
  safe body   x in [-0.60, -0.40], y in [0.04, 0.30], z in [0, 0.22], 15 mm steel walls, front open
  door        hinge axis z at (x=-0.40, y=0.30); opens toward the robot (+angle, 0..110 deg); damping + friction
              loss so it stays where it is left; door-vs-world contacts excluded (hinge holds it)
  handle      vertical bar (r 9 mm, 90 mm) 35 mm in front of the door, 40 mm from its free edge; own body
              'obj_safe_handle' so the existing CONNECT hold can grip it
  object      'obj_valuables_box' 70 x 50 x 45 mm, 0.12 kg, free joint, on the table at (-0.36, -0.17)
The task definition (RoboDojo store_in_safe, ARX X5 dual arm) is reproduced from its name / public summary;
the paper appendix with the exact objects was not available.  All sizes are parameters below.
"""
import math
import os
import struct
import zlib
from pathlib import Path

VERSION = "V3.44"
DEFAULT_SCENE = "fill_egg_holder"
SCENES = {
    "fill_egg_holder": {"label": "fill_egg_holder - 鸡蛋放入蛋托 (默认)",
                        "summary": "4 eggs -> 8-slot carton (existing scene)"},
    "store_in_safe": {"label": "store_in_safe - 物体放入保险箱 (RoboDojo)",
                      "summary": "open the safe door, put the valuables box inside, close the door"},
    # V3.61 STACK CUPS SCENE (RoboDojo-style): two red/white party cups on the same table, stack one into the other
    "stack_cups": {"label": "stack_cups - 叠杯子 (RoboDojo)",
                   "summary": "two red party cups upright on the table: put cup_1 into cup_2 (nested stack)"},
    # V3.64 SO-101 GRIPPER ON NERO: a VARIANT of stack_cups (same cups / table / task / skills, other gripper).
    # Scene code keeps seeing "stack_cups" (active_scene() returns the base scene, so the backend, the camera process
    # and every stack_cups branch run unchanged); only the MJCF builder and the cup skills read gripper_variant().
    "stack_cups_so101": {"label": "stack_cups_so101 - 叠杯子 · SO-101 夹爪 (7 mm 转接环, 偏转 45°)",
                         "summary": "stack_cups with the SO-101 gripper on a 7 mm adapter ring, rolled 45 deg",
                         "base": "stack_cups", "gripper": "so101"},
    # V3.72 SORT ITEMS SCENE: banana, apple, computer mouse and a thick marker on the table, an open red bin and an
    # open blue bin left / right; the agent sorts the items into the bins as the task prompt says (SO-101 gripper)
    "sort_items": {"label": "sort_items - 物品分类放入红框 / 蓝框 · SO-101 夹爪（香蕉 / 苹果 / 鼠标 / 马克笔）",
                   "summary": "banana, apple, mouse, marker on the table; sort them into the open red / blue bins by the prompt",
                   "gripper": "so101"},
}
SCENES_REV = "V3.72"       # VERSION above stays "V3.44" (tests pin it); stack_cups V3.61, stack_cups_so101 V3.64, sort_items V3.72
HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- geometry (store_in_safe)
SAFE = {
    "x_back": -0.60, "x_front": -0.40, "y0": 0.04, "y1": 0.30, "h": 0.22, "wall": 0.015,
    "door_t": 0.018, "door_max_deg": 110.0, "door_damping": 0.6, "door_frictionloss": 0.08, "door_mass": 1.2,
    "handle_from_edge": 0.04, "handle_standoff": 0.035, "handle_r": 0.009, "handle_z": 0.12, "handle_len": 0.09,
}
BOX = {"xy": (-0.36, -0.17), "half": (0.035, 0.025, 0.0225), "mass": 0.12}
LEGACY_HIDDEN = ("obj_red_cube", "obj_blue_cube", "obj_red_ball", "obj_blue_ball")


def scene_from_env():
    """NERO_SCENE if it names a known scene, else None."""
    s = (os.environ.get("NERO_SCENE") or "").strip()
    return s if s in SCENES else None


def base_scene(name):
    """V3.64: a scene variant (stack_cups_so101) -> its base scene (stack_cups); other names unchanged"""
    return SCENES.get(name, {}).get("base", name)


def selected_scene():
    return base_scene(scene_from_env() or DEFAULT_SCENE)


def active_scene():
    s = (os.environ.get("NERO_V343_ACTIVE_SCENE") or "").strip()
    return base_scene(s) if s in SCENES else selected_scene()


def selected_variant():
    """V3.64: the scene exactly as chosen in the console (NERO_SCENE, may be a variant such as stack_cups_so101)"""
    s = scene_from_env() or DEFAULT_SCENE
    return s if base_scene(s) == active_scene() else active_scene()


def gripper_variant():
    """V3.64 SO-101 GRIPPER ON NERO: "so101" for stack_cups_so101, else "official" (AgileX gripper)"""
    return SCENES.get(selected_variant(), {}).get("gripper", "official")


def scene_names():
    return list(SCENES)


# ----------------------------------------------------------------------------- run_sim setup data (A2)
def tf_body_map():
    """semantic name -> MuJoCo body; TF child frame = 'sim_' + semantic (object_pose('valuables_box') etc.)."""
    if active_scene() == "stack_cups":                  # V3.61 STACK CUPS SCENE
        return {n: n for n in CUP_NAMES}
    if active_scene() == "sort_items":                  # V3.72 SORT ITEMS SCENE
        return {n: n for n in SORT_ITEM_NAMES}
    return {"obj_valuables_box": "obj_valuables_box", "obj_safe_handle": "obj_safe_handle",
            "obj_safe_door": "safe_door"}


def hold_object_names():
    if active_scene() == "stack_cups":                  # V3.61 STACK CUPS SCENE
        return list(CUP_NAMES) + list(LEGACY_HIDDEN)
    if active_scene() == "sort_items":                  # V3.72 SORT ITEMS SCENE
        return list(SORT_ITEM_NAMES) + list(LEGACY_HIDDEN)
    return ["obj_valuables_box", "obj_safe_handle"] + list(LEGACY_HIDDEN)


PROTECTED_PREFIXES = ("safe_body_", "sort_bin_")     # V3.72: the bin walls / floors protect like the safe walls


def _derived():
    s = SAFE
    W = s["y1"] - s["y0"]
    return W, (s["y0"] + s["y1"]) / 2.0, (s["x_back"] + s["x_front"]) / 2.0, s["x_front"] - s["x_back"]


def static_boxes():
    """(name, centre, half-size) of the fixed safe body; the front is open (the door closes it)."""
    s = SAFE
    W, yc, xc, D = _derived()
    t, H = s["wall"], s["h"]
    xf_in = s["x_front"] - 0.001                       # 1 mm gap to the closed door
    D2 = xf_in - s["x_back"]
    xc2 = (xf_in + s["x_back"]) / 2.0
    return [
        ("safe_body_bottom", (xc2, yc, t / 2), (D2 / 2, W / 2, t / 2)),
        ("safe_body_top", (xc2, yc, H - t / 2), (D2 / 2, W / 2, t / 2)),
        ("safe_body_back", (s["x_back"] + t / 2, yc, H / 2), (t / 2, W / 2, H / 2)),
        ("safe_body_side_yneg", (xc2, s["y0"] + t / 2, H / 2), (D2 / 2, t / 2, H / 2)),
        ("safe_body_side_ypos", (xc2, s["y1"] - t / 2, H / 2), (D2 / 2, t / 2, H / 2)),
    ]


def planning_boxes(model=None, data=None):
    """MoveIt boxes for the fixed safe walls: (id, centre, full size) - same format as the egg-holder list."""
    if active_scene() == "stack_cups":                  # V3.61: the cups are movable objects, no fixed obstacles
        return []
    if active_scene() == "sort_items":                  # V3.72: the two open bins (walls + floor) are fixed obstacles
        return [(n, c, tuple(2.0 * v for v in h)) for n, c, h, _rgba in sort_bin_boxes()]
    return [(n, c, tuple(2.0 * v for v in h)) for n, c, h in static_boxes()]


def door_boxes(theta_rad):
    """Door panel + handle as yawed boxes for MoveIt at door angle theta: (id, centre, full size, yaw)."""
    s = SAFE
    W = s["y1"] - s["y0"]
    c, si = math.cos(theta_rad), math.sin(theta_rad)
    n = (c, si)                                        # outward door normal (+x when closed)
    u = (si, -c)                                       # hinge -> free edge
    hx, hy = s["x_front"], s["y1"]
    t = s["door_t"]
    panel = (hx + u[0] * W / 2 + n[0] * t / 2, hy + u[1] * W / 2 + n[1] * t / 2, s["h"] / 2)
    a = W - s["handle_from_edge"]
    off = t + s["handle_standoff"] / 2
    handle = (hx + u[0] * a + n[0] * off, hy + u[1] * a + n[1] * off, s["handle_z"])
    return [("safe_door", panel, (t, W, s["h"]), theta_rad),
            ("safe_door_handle", handle, (s["handle_standoff"] + 2 * s["handle_r"], 2 * s["handle_r"],
                                          s["handle_len"] + 2 * s["handle_r"]), theta_rad)]


def door_angle(model, data):
    import mujoco
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "safe_door_hinge")
    return float(data.qpos[int(model.jnt_qposadr[jid])]) if jid >= 0 else 0.0


def planning_rotated_boxes(model, data):
    if active_scene() in ("stack_cups", "sort_items"):  # V3.61 / V3.72: no hinged door in these scenes
        return []
    return door_boxes(door_angle(model, data))


def door_angle_from_quat_xyzw(q):
    x, y, z, w = q
    return math.degrees(2.0 * math.atan2(z, w))


def inside_safe(xyz, margin=0.005):
    s = SAFE
    t = s["wall"]
    return (s["x_back"] + t - margin <= xyz[0] <= s["x_front"] + margin
            and s["y0"] + t - margin <= xyz[1] <= s["y1"] - t + margin
            and t - margin <= xyz[2] <= s["h"] - t + margin)


def door_closed(angle_deg, tol_deg=3.0):
    return abs(angle_deg) <= tol_deg


def task_state(door_deg, box_xyz):
    """store_in_safe progress, RoboDojo-style sub-steps: door opened / object inside / door closed again."""
    inside = box_xyz is not None and inside_safe(box_xyz)
    return {"door_angle_deg": round(door_deg, 1), "door_open": door_deg >= 60.0, "object_inside": inside,
            "door_closed": door_closed(door_deg), "success": bool(inside and door_closed(door_deg))}


# ----------------------------------------------------------------------------- MJCF
def _png(path, rgb):
    import numpy as np
    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))

    def chunk(tag, payload):
        return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)

    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _f(*v):
    return " ".join(f"{float(x):.5f}" for x in v)


def compiler_angle_unit(xml_text):
    """MJCF default is degree; nero_motor.xml may set <compiler angle="radian">. Joint ranges must follow it."""
    import re
    m = re.search(r"<compiler\b[^>]*\bangle\s*=\s*[\"'](\w+)[\"']", xml_text)
    return "radian" if (m and m.group(1).lower().startswith("rad")) else "degree"


def scene_world_xml(angle_unit="radian"):
    s = SAFE
    rng_hi = (f"{math.radians(s['door_max_deg']):.5f}" if angle_unit == "radian" else f"{s['door_max_deg']:.3f}")
    W, yc, xc, D = _derived()
    t, H, td = s["wall"], s["h"], s["door_t"]
    parts = []
    for name, c, h in static_boxes():
        parts.append(f'<geom name="{name}" type="box" pos="{_f(*c)}" size="{_f(*h)}" material="v343_steel_mat" '
                     f'friction="1.0 0.01 0.001" contype="1" conaffinity="1"/>')
    # felt lining (part of the safe body, protected)
    parts.append(f'<geom name="safe_body_felt" type="box" pos="{_f(xc - 0.0005, yc, t + 0.002)}" '
                 f'size="{_f(D / 2 - t - 0.002, W / 2 - t - 0.002, 0.002)}" material="v343_felt_mat" '
                 f'friction="1.1 0.02 0.001" contype="1" conaffinity="1"/>')
    a = W - s["handle_from_edge"]
    so, r, L = s["handle_standoff"], s["handle_r"], s["handle_len"]
    door = f"""
    <body name="safe_door" pos="{_f(s['x_front'], s['y1'], 0.0)}">
        <joint name="safe_door_hinge" type="hinge" axis="0 0 1" limited="true"
               range="0 {rng_hi}" damping="{s['door_damping']}"
               frictionloss="{s['door_frictionloss']}" armature="0.005"/>
        <geom name="safe_door_panel" type="box" pos="{_f(td / 2, -W / 2, H / 2 + 0.001)}"
              size="{_f(td / 2, W / 2 - 0.001, H / 2 - 0.001)}" mass="{s['door_mass']}" material="v343_door_mat"
              friction="1.0 0.01 0.001" contype="1" conaffinity="1"/>
        <geom name="safe_door_dial" type="cylinder" fromto="{_f(td, -0.12, 0.14, td + 0.012, -0.12, 0.14)}"
              size="0.030" rgba="0.12 0.12 0.13 1" contype="0" conaffinity="0" density="0"/>
        <geom name="safe_door_dial_knob" type="cylinder" fromto="{_f(td + 0.012, -0.12, 0.14, td + 0.020, -0.12, 0.14)}"
              size="0.012" material="v343_metal_mat" contype="0" conaffinity="0" density="0"/>
        <geom name="safe_door_hinge_lo" type="cylinder" fromto="{_f(td / 2, 0.006, 0.025, td / 2, 0.006, 0.065)}"
              size="0.008" rgba="0.35 0.36 0.38 1" contype="0" conaffinity="0" density="0"/>
        <geom name="safe_door_hinge_hi" type="cylinder" fromto="{_f(td / 2, 0.006, 0.155, td / 2, 0.006, 0.195)}"
              size="0.008" rgba="0.35 0.36 0.38 1" contype="0" conaffinity="0" density="0"/>
        <body name="obj_safe_handle" pos="{_f(td + so, -a, s['handle_z'])}">
            <geom name="obj_safe_handle_bar" type="capsule" fromto="{_f(0, 0, -L / 2, 0, 0, L / 2)}" size="{r}"
                  mass="0.05" material="v343_metal_mat" friction="1.2 0.02 0.001" condim="4"
                  contype="1" conaffinity="1"/>
            <geom name="obj_safe_handle_post_lo" type="cylinder" fromto="{_f(-so, 0, -L / 2 + 0.010, 0, 0, -L / 2 + 0.010)}"
                  size="0.0065" mass="0.01" material="v343_metal_mat" contype="1" conaffinity="1"/>
            <geom name="obj_safe_handle_post_hi" type="cylinder" fromto="{_f(-so, 0, L / 2 - 0.010, 0, 0, L / 2 - 0.010)}"
                  size="0.0065" mass="0.01" material="v343_metal_mat" contype="1" conaffinity="1"/>
        </body>
    </body>"""
    bx, by = BOX["xy"]
    hx, hy, hz = BOX["half"]
    box = f"""
    <body name="obj_valuables_box" pos="{_f(bx, by, hz + 0.0005)}">
        <freejoint/>
        <geom name="obj_valuables_box_geom" type="box" size="{_f(hx, hy, hz)}" mass="{BOX['mass']}"
              material="v343_box_mat" friction="1.0 0.02 0.001" condim="4" solref="0.005 1"
              solimp="0.96 0.995 0.0008" margin="0.0003"/>
        <geom name="obj_valuables_box_lid" type="box" pos="0 0 {hz - 0.004:.5f}" size="{_f(hx + 0.0005, hy + 0.0005, 0.004)}"
              material="v343_lid_mat" contype="0" conaffinity="0" density="0"/>
    </body>"""
    legacy = "\n".join(
        f'    <body name="{n}" pos="{-8 - 0.1 * i:.1f} -8 -8"><freejoint/><geom name="{n}_geom" type="sphere" size="0.001" '
        f'mass="0.001" rgba="0 0 0 0" contype="0" conaffinity="0"/></body>' for i, n in enumerate(LEGACY_HIDDEN))
    return f"""
    <!-- ============================================================
         NERO V3.43 store_in_safe SCENE (RoboDojo task name) - table top z = 0, robot mount unchanged
         ============================================================ -->
    <geom name="v5_grid_floor" type="plane" pos="0 0 -0.28" size="6 6 0.1"
          material="v343_floor_mat" friction="1.2 0.01 0.001" contype="1" conaffinity="1"/>
    <geom name="play_table" type="box" pos="-0.385 0 -0.03" size="0.335 0.38 0.03" material="v343_table_mat"
          friction="1.2 0.01 0.001" solref="0.008 1" solimp="0.96 0.995 0.0005" margin="0.001"
          contype="1" conaffinity="1"/>
    <geom name="nero_support_top" type="box" pos="-0.025 0 -0.008" size="0.025 0.085 0.008"
          rgba="0.15 0.16 0.18 1" contype="0" conaffinity="0"/>
    <geom name="nero_support_vertical" type="box" pos="-0.055 0 -0.075" size="0.008 0.085 0.067"
          rgba="0.15 0.16 0.18 1" contype="0" conaffinity="0"/>
    <geom name="nero_support_lower_jaw" type="box" pos="-0.080 0 -0.134" size="0.030 0.085 0.008"
          rgba="0.15 0.16 0.18 1" contype="0" conaffinity="0"/>

    <!-- safe body (static) -->
    {chr(10).join('    ' + p for p in parts)}
    <!-- hinged door + handle -->
    {door}
    <!-- object to store -->
    {box}
    <!-- legacy hidden bodies (unchanged GUI/runtime compatibility) -->
{legacy}
    <light name="eggscene_key" pos="0.25 -0.65 2.20" dir="-0.30 0.22 -1" directional="true" castshadow="true"
           diffuse="0.92 0.84 0.72" specular="0.22 0.20 0.18"/>
    <light name="eggscene_fill" pos="-1.20 0.80 1.50" dir="0.55 -0.25 -1" directional="true" castshadow="false"
           diffuse="0.34 0.40 0.48" specular="0.05 0.06 0.08"/>
"""


def make_store_in_safe_xml(original_xml_path, visual_quality=None):
    """Same contract as the backend's make_grid_world_xml: read nero_motor.xml, add the scene, write a temp MJCF
    next to it (relative mesh paths stay valid) and return its path.
    V3.61: the backend calls this for every non-default scene (unchanged A2 block) -> stack_cups is dispatched here."""
    if active_scene() == "stack_cups":
        return make_stack_cups_xml(original_xml_path, visual_quality=visual_quality)
    if active_scene() == "sort_items":                  # V3.72 SORT ITEMS SCENE
        return make_sort_items_xml(original_xml_path, visual_quality=visual_quality)
    import numpy as np
    src = Path(original_xml_path).resolve()
    xml = src.read_text(encoding="utf-8")
    tex = src.parent / "_nero_v343_scene_textures"
    tex.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(34301)
    n = rng.normal(0.0, 1.0, (256, 256))
    n = (n + np.roll(n, 3, 0) + np.roll(n, 4, 1)) / 3.0
    floor = np.stack([102 + 6 * n, 110 + 6 * n, 127 + 6 * n], -1)          # same cool grey as V3.33
    table = np.stack([236 + 1.5 * n, 236 + 1.5 * n, 234 + 1.5 * n], -1)   # same plain white as V3.33
    _png(tex / "floor_grey.png", np.clip(floor, 0, 255))
    _png(tex / "table_white.png", np.clip(table, 0, 255))
    assets = f"""
    <!-- NERO V3.43 store_in_safe scene assets -->
    <texture name="v343_sky" type="skybox" builtin="gradient" rgb1="0.54 0.64 0.76" rgb2="0.92 0.90 0.84"
             width="512" height="3072"/>
    <texture name="v343_floor_tex" type="2d" file="{(tex / 'floor_grey.png').as_posix()}"/>
    <texture name="v343_table_tex" type="2d" file="{(tex / 'table_white.png').as_posix()}"/>
    <material name="v343_floor_mat" texture="v343_floor_tex" texrepeat="7 7" reflectance="0.02" specular="0.16"
              shininess="0.08"/>
    <material name="v343_table_mat" texture="v343_table_tex" texrepeat="2.2 1.7" reflectance="0.05"
              specular="0.25" shininess="0.2"/>
    <material name="v343_steel_mat" rgba="0.21 0.24 0.28 1" reflectance="0.05" specular="0.35" shininess="0.35"/>
    <material name="v343_door_mat" rgba="0.24 0.27 0.31 1" reflectance="0.06" specular="0.40" shininess="0.40"/>
    <material name="v343_metal_mat" rgba="0.80 0.81 0.83 1" reflectance="0.25" specular="0.8" shininess="0.8"/>
    <material name="v343_felt_mat" rgba="0.45 0.10 0.12 1" reflectance="0.0" specular="0.02" shininess="0.02"/>
    <material name="v343_box_mat" rgba="0.62 0.12 0.16 1" reflectance="0.02" specular="0.25" shininess="0.2"/>
    <material name="v343_lid_mat" rgba="0.80 0.62 0.22 1" reflectance="0.10" specular="0.5" shininess="0.5"/>
"""
    import re
    if re.search(r"<asset\b[^>]*/>", xml):
        xml = re.sub(r"<asset\b[^>]*/>", "<asset>\n" + assets + "\n</asset>", xml, count=1)
    elif "<asset" in xml:
        xml = re.sub(r"(<asset\b[^>]*>)", lambda m: m.group(1) + "\n" + assets, xml, count=1)
    else:
        xml = xml.replace("<worldbody>", "<asset>\n" + assets + "\n</asset>\n\n<worldbody>", 1)
    unit = compiler_angle_unit(xml)
    print(f"[V3.44 SCENE] store_in_safe: compiler angle unit = {unit} -> door range 0..{SAFE['door_max_deg']:.0f} deg", flush=True)
    xml = xml.replace("<worldbody>", "<worldbody>\n" + scene_world_xml(unit), 1)
    if visual_quality is not None:
        xml = visual_quality(xml)
    eq = "\n        ".join(
        f'<connect name="hold_{o}" body1="link7" body2="{o}" anchor="0 0 0" active="false" '
        f'solref="0.008 1" solimp="0.98 0.995 0.001"/>' for o in hold_object_names())
    if "</equality>" in xml:
        xml = xml.replace("</equality>", "        " + eq + "\n    </equality>", 1)
    else:
        xml = xml.replace("</mujoco>", "    <equality>\n        " + eq + "\n    </equality>\n</mujoco>", 1)
    # the hinge alone holds the door: no door-vs-table / door-vs-safe-wall contacts (no hinge friction from walls)
    xml = xml.replace("</mujoco>", '    <contact>\n        <exclude body1="world" body2="safe_door"/>\n'
                                   '        <exclude body1="world" body2="obj_safe_handle"/>\n    </contact>\n</mujoco>', 1)
    out = src.with_name("_nero_motor_v343_store_in_safe_tmp.xml")
    out.write_text(xml, encoding="utf-8")
    return str(out)


# ============================================================================= V3.61 STACK CUPS SCENE
# Two disposable US party cups (16 oz "red cup": red outside, white inside) standing upright on the same white table;
# the task (RoboDojo stack_cups style) is to put obj_cup_1 into obj_cup_2.  Sizes are the commonly quoted 16 oz
# party-cup dimensions (top 95 mm, bottom 60 mm, height 121 mm); not measured here - measure your real cups and edit
# CUP below.  The collision model is a thin-walled truncated cone: CUP["segments"] tilted wall boxes, a bottom disc
# and a rolled rim ring.  The real wall is ~0.3 mm; the collision wall is CUP["wall"] (2 mm) so a falling cup cannot
# tunnel through it at the backend's 1 ms time step.  Measured stack pitch of the model: ~25 mm (base-to-base).
CUP = {
    "h": 0.121, "r_top": 0.0475, "r_bot": 0.030, "wall": 0.002, "segments": 24, "rim_r": 0.0016,
    "mass": 0.010,                    # kg, one empty 16 oz cup ~ 10 g
    "friction": "0.45 0.01 0.001",    # polystyrene on polystyrene ~0.3-0.5 (assumed)
    "nested_dz": 0.0253,              # base-to-base height of a fully nested pair in this model (measured)
}
CUP_NAMES = ("obj_cup_1", "obj_cup_2")
CUP_ORIGIN = "rim"                   # V3.62 RIM-CENTRE CUP ORIGIN (scene bodies); the V3.61 study uses "base"
# cup_1 is moved into cup_2.  x = -0.35: carrying a cup (TCP ~96 mm above its base with a 25 mm rim pinch) over
# the other cup needs a top-down TCP at z ~0.24 m - the NERO reaches that only up to |x| ~0.36 (NERO_V361 study).
CUP_XY = {"obj_cup_1": (-0.35, -0.13), "obj_cup_2": (-0.35, 0.13)}


def cup_geometry(cup=None):
    c = dict(CUP if cup is None else cup)
    c["taper"] = (c["r_top"] - c["r_bot"]) / c["h"]
    c["slant_deg"] = math.degrees(math.atan2(c["r_top"] - c["r_bot"], c["h"]))
    c["pitch_m"] = c["wall"] / c["taper"]                      # wall/taper estimate only; measured: c["nested_dz"]
    return c


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)


def cup_body_xml(name, xy, cup=None, z0=0.0005, free=True, quat=None, origin="base"):
    """MJCF <body> of one cup.  origin="base": body origin = bottom centre (V3.61 study);
    origin="rim" (V3.62 scene): body origin = centre of the rim circle on the cup axis.
    V3.62 RIM-CENTRE CUP ORIGIN: the backend's grasp hold is a CONNECT point at the object's body origin; with the
    origin at the base the held cup is an inverted pendulum about its bottom (it swung to ~40 deg in the twin), with
    the origin on the axis ABOVE the centre of mass it hangs plumb from the gripper like a pendulum."""
    c = cup_geometry(cup)
    n, H, rb, rt, w = int(c["segments"]), c["h"], c["r_bot"], c["r_top"], c["wall"]
    oz = -H if origin == "rim" else 0.0                         # geom z offset in the body frame
    # V3.62 MASS-INDEPENDENT CUP CONTACT: MuJoCo scales a positive solref stiffness with the (10 g) cup's effective
    # mass -> the wall was so soft that the finger studs sank > 0.5 mm at ~1.7 N and the backend's gripper
    # penetration guard stopped the clamp.  Scene cups use a direct stiffness / damping (negative solref).
    sref = (os.environ.get("NERO_V362_CUP_SOLREF") or c.get("scene_solref", "-20000 -60")) if origin == "rim" else "0.004 1"
    beta = math.atan2(rt - rb, H)
    slant = math.hypot(H, rt - rb)
    half_t = (rt + w) * math.tan(math.pi / n)                  # wide enough to close the gaps at the rim
    r_mid = 0.5 * (rb + rt)
    m_wall = 0.75 * c["mass"] / n
    qy = (math.cos(beta / 2), 0.0, math.sin(beta / 2), 0.0)
    g = []
    for k in range(n):
        a = 2 * math.pi * k / n
        qz = (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))
        q = _quat_mul(qz, qy)
        cx, cy = r_mid * math.cos(a), r_mid * math.sin(a)
        g.append(f'<geom name="{name}_wall_{k:02d}" type="box" pos="{_f(cx, cy, H / 2 + oz)}" quat="{_f(*q)}" '
                 f'size="{_f(w / 2, half_t, slant / 2)}" mass="{m_wall:.6f}" material="v361_cup_red_mat" '
                 f'friction="{c["friction"]}" condim="4" solref="{sref}" solimp="0.95 0.99 0.0005" margin="0.0005"/>')
        ri = r_mid - 0.6 * w * math.cos(beta)                  # white inside (visual only)
        g.append(f'<geom name="{name}_inner_{k:02d}" type="box" pos="{_f(ri * math.cos(a), ri * math.sin(a), H / 2 + 0.6 * w * math.sin(beta) + oz)}" '
                 f'quat="{_f(*q)}" size="{_f(w / 4, half_t * 0.99, slant / 2 - 0.002)}" material="v361_cup_white_mat" '
                 f'contype="0" conaffinity="0" density="0"/>')
    g.append(f'<geom name="{name}_bottom" type="cylinder" pos="0 0 {w / 2 + oz:.5f}" size="{rb:.5f} {w / 2:.5f}" '
             f'mass="{0.15 * c["mass"]:.6f}" material="v361_cup_red_mat" friction="{c["friction"]}" condim="4" '
             f'solref="{sref}" solimp="0.95 0.99 0.0005"/>')
    g.append(f'<geom name="{name}_bottom_inner" type="cylinder" pos="0 0 {w + 0.0002 + oz:.5f}" size="{rb - w:.5f} 0.0002" '
             f'material="v361_cup_white_mat" contype="0" conaffinity="0" density="0"/>')
    rr = rt + 0.5 * w
    # V3.62 STIFF RIM CONTACT: in V3.61 the rim capsules had MuJoCo's default soft contact (timeconst 0.02 s, 5x
    # softer than the wall); finger studs sank 0.55 mm into the rim and the backend's gripper penetration guard
    # (> 0.55 mm -> no closing force) stopped the clamp at ~1 N.  Same stiffness / margin as the wall boxes now.
    rim_contact = (f' solref="{sref}" solimp="0.95 0.99 0.0005" margin="0.0005"' if origin == "rim" else "")
    for k in range(n):
        a0, a1 = 2 * math.pi * k / n, 2 * math.pi * (k + 1) / n
        g.append(f'<geom name="{name}_rim_{k:02d}" type="capsule" '
                 f'fromto="{_f(rr * math.cos(a0), rr * math.sin(a0), H + oz, rr * math.cos(a1), rr * math.sin(a1), H + oz)}" '
                 f'size="{c["rim_r"]:.5f}" mass="{0.10 * c["mass"] / n:.6f}" material="v361_cup_red_mat" '
                 f'friction="{c["friction"]}" condim="4"{rim_contact}/>')
    q_txt = f' quat="{_f(*quat)}"' if quat is not None else ""
    joint = "<freejoint/>" if free else ""
    body = "\n            ".join(g)
    return (f'<body name="{name}" pos="{_f(xy[0], xy[1], z0 - oz)}"{q_txt}>\n            {joint}\n            {body}\n'
            f'        </body>')


CUP_ASSETS = """
    <!-- NERO V3.61 stack_cups assets (party cup: red outside, white inside) -->
    <material name="v361_cup_red_mat" rgba="0.80 0.07 0.09 1" reflectance="0.08" specular="0.55" shininess="0.6"/>
    <material name="v361_cup_white_mat" rgba="0.96 0.96 0.95 1" reflectance="0.04" specular="0.30" shininess="0.4"/>
"""


def stack_cups_world_xml(cup_xy=None):
    xy = dict(CUP_XY if cup_xy is None else cup_xy)
    cups = "\n    ".join(cup_body_xml(n, xy[n], origin=CUP_ORIGIN) for n in CUP_NAMES)   # V3.62 rim origin
    legacy = "\n".join(
        f'    <body name="{n}" pos="{-8 - 0.1 * i:.1f} -8 -8"><freejoint/><geom name="{n}_geom" type="sphere" size="0.001" '
        f'mass="0.001" rgba="0 0 0 0" contype="0" conaffinity="0"/></body>' for i, n in enumerate(LEGACY_HIDDEN))
    return f"""
    <!-- ============================================================
         NERO V3.61 stack_cups SCENE (RoboDojo task name) - table top z = 0, robot mount unchanged
         ============================================================ -->
    <geom name="v5_grid_floor" type="plane" pos="0 0 -0.28" size="6 6 0.1"
          material="v343_floor_mat" friction="1.2 0.01 0.001" contype="1" conaffinity="1"/>
    <geom name="play_table" type="box" pos="-0.385 0 -0.03" size="0.335 0.38 0.03" material="v343_table_mat"
          friction="1.2 0.01 0.001" solref="0.008 1" solimp="0.96 0.995 0.0005" margin="0.001"
          contype="1" conaffinity="1"/>
    <geom name="nero_support_top" type="box" pos="-0.025 0 -0.008" size="0.025 0.085 0.008"
          rgba="0.15 0.16 0.18 1" contype="0" conaffinity="0"/>
    <geom name="nero_support_vertical" type="box" pos="-0.055 0 -0.075" size="0.008 0.085 0.067"
          rgba="0.15 0.16 0.18 1" contype="0" conaffinity="0"/>
    <geom name="nero_support_lower_jaw" type="box" pos="-0.080 0 -0.134" size="0.030 0.085 0.008"
          rgba="0.15 0.16 0.18 1" contype="0" conaffinity="0"/>
    <!-- two party cups, upright -->
    {cups}
    <!-- legacy hidden bodies (unchanged GUI/runtime compatibility) -->
{legacy}
    <light name="eggscene_key" pos="0.25 -0.65 2.20" dir="-0.30 0.22 -1" directional="true" castshadow="true"
           diffuse="0.92 0.84 0.72" specular="0.22 0.20 0.18"/>
    <light name="eggscene_fill" pos="-1.20 0.80 1.50" dir="0.55 -0.25 -1" directional="true" castshadow="false"
           diffuse="0.34 0.40 0.48" specular="0.05 0.06 0.08"/>
"""


def make_stack_cups_xml(original_xml_path, visual_quality=None):
    """V3.61: same contract as make_store_in_safe_xml for the stack_cups scene."""
    import re
    import numpy as np
    src = Path(original_xml_path).resolve()
    xml = src.read_text(encoding="utf-8")
    tex = src.parent / "_nero_v343_scene_textures"
    tex.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(34301)
    nz = rng.normal(0.0, 1.0, (256, 256))
    nz = (nz + np.roll(nz, 3, 0) + np.roll(nz, 4, 1)) / 3.0
    _png(tex / "floor_grey.png", np.clip(np.stack([102 + 6 * nz, 110 + 6 * nz, 127 + 6 * nz], -1), 0, 255))
    _png(tex / "table_white.png", np.clip(np.stack([236 + 1.5 * nz, 236 + 1.5 * nz, 234 + 1.5 * nz], -1), 0, 255))
    assets = f"""
    <texture name="v343_sky" type="skybox" builtin="gradient" rgb1="0.54 0.64 0.76" rgb2="0.92 0.90 0.84"
             width="512" height="3072"/>
    <texture name="v343_floor_tex" type="2d" file="{(tex / 'floor_grey.png').as_posix()}"/>
    <texture name="v343_table_tex" type="2d" file="{(tex / 'table_white.png').as_posix()}"/>
    <material name="v343_floor_mat" texture="v343_floor_tex" texrepeat="7 7" reflectance="0.02" specular="0.16"
              shininess="0.08"/>
    <material name="v343_table_mat" texture="v343_table_tex" texrepeat="2.2 1.7" reflectance="0.05"
              specular="0.25" shininess="0.2"/>
""" + CUP_ASSETS
    if re.search(r"<asset\b[^>]*/>", xml):
        xml = re.sub(r"<asset\b[^>]*/>", "<asset>\n" + assets + "\n</asset>", xml, count=1)
    elif "<asset" in xml:
        xml = re.sub(r"(<asset\b[^>]*>)", lambda m: m.group(1) + "\n" + assets, xml, count=1)
    else:
        xml = xml.replace("<worldbody>", "<asset>\n" + assets + "\n</asset>\n\n<worldbody>", 1)
    g = cup_geometry()
    print(f"[V3.61 SCENE] stack_cups: 2 cups h={g['h'] * 1000:.0f} mm top {2 * g['r_top'] * 1000:.0f} mm "
          f"bottom {2 * g['r_bot'] * 1000:.0f} mm, nested base-to-base {g['nested_dz'] * 1000:.1f} mm (model)", flush=True)
    xml = xml.replace("<worldbody>", "<worldbody>\n" + stack_cups_world_xml(), 1)
    if visual_quality is not None:
        xml = visual_quality(xml)
    so101 = gripper_variant() == "so101"
    if so101:                                           # V3.64 SO-101 GRIPPER ON NERO (backend then adds no gripper)
        import NERO_V364_SO101_Gripper as _G364
        xml = _G364.insert_into_model_xml(xml)
        print(f"[V3.64 SCENE] stack_cups_so101: SO-101 gripper mounted on link7 {_G364.describe()}", flush=True)
    eq = "\n        ".join(
        f'<connect name="hold_{o}" body1="link7" body2="{o}" anchor="0 0 0" active="false" '
        f'solref="0.008 1" solimp="0.98 0.995 0.001"/>' for o in hold_object_names())
    if "</equality>" in xml:
        xml = xml.replace("</equality>", "        " + eq + "\n    </equality>", 1)
    else:
        xml = xml.replace("</mujoco>", "    <equality>\n        " + eq + "\n    </equality>\n</mujoco>", 1)
    out = src.with_name("_nero_motor_v343_stack_cups_so101_tmp.xml" if so101 else "_nero_motor_v343_stack_cups_tmp.xml")
    out.write_text(xml, encoding="utf-8")
    return str(out)


def cup_axis_tilt_deg(quat_wxyz):
    """angle between a cup's axis (body z) and the world vertical"""
    w, x, y, z = [float(v) for v in quat_wxyz]
    zz = 1 - 2 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def cup_base_point(xyz, quat_wxyz, origin="base", cup=None):
    """V3.62: centre of a cup's bottom from its body pose (origin "base" or "rim")."""
    if origin != "rim":
        return [float(v) for v in xyz]
    H = cup_geometry(cup)["h"]
    w, x, y, z = [float(v) for v in quat_wxyz]
    ax = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))      # body z axis in the world
    return [float(xyz[i]) - H * ax[i] for i in range(3)]


def cup_task_state(cup1_xyz, cup1_quat_wxyz, cup2_xyz, cup2_quat_wxyz, cup=None, origin="base"):
    """stack_cups: stacked = cup_1 upright with its base inside cup_2 (success); nested_full = slid all the way down.
    A rigid model cup that is released off-centre jams part-way (stacked, not nested_full).
    origin: body-origin convention of the given poses ("base" = V3.61 study, "rim" = V3.62 scene)."""
    g = cup_geometry(cup)
    cup1_xyz = cup_base_point(cup1_xyz, cup1_quat_wxyz, origin, cup)
    cup2_xyz = cup_base_point(cup2_xyz, cup2_quat_wxyz, origin, cup)
    dxy = math.hypot(cup1_xyz[0] - cup2_xyz[0], cup1_xyz[1] - cup2_xyz[1])
    dz = cup1_xyz[2] - cup2_xyz[2]
    t1, t2 = cup_axis_tilt_deg(cup1_quat_wxyz), cup_axis_tilt_deg(cup2_quat_wxyz)
    # cup_1 standing on the rim -> dz ~ h; nested -> dz ~ 25 mm in this model (measured, see NERO_V361_Cup_Stack)
    stacked = dxy <= 0.5 * g["r_bot"] and t1 <= 8.0 and t2 <= 8.0 and 0.0 < dz <= 0.5 * g["h"]
    full = stacked and dz <= g["nested_dz"] + 0.005
    return {"xy_offset_mm": round(dxy * 1000, 1), "base_dz_mm": round(float(dz) * 1000, 1), "tilt_cup1_deg": round(t1, 1),
            "tilt_cup2_deg": round(t2, 1), "stacked": bool(stacked), "nested_full": bool(full),
            "nested": bool(stacked), "success": bool(stacked)}


# ----------------------------------------------------------------------------- V3.72 SORT ITEMS SCENE
# Items lie on the table (body origin = the item centre, = the grasp point of long items); x = long axis.
# category = ground truth for the test log only - the agent gets the detector label and decides by the prompt.
SORT_ITEMS = {
    "obj_banana": {"label": "banana", "category": "fruit", "xy": (-0.31, -0.085), "yaw_deg": 12.0, "cz": 0.017,
                   "width": 0.034, "length": 0.18, "height": 0.034, "shape": "long", "mass": 0.12},
    "obj_apple": {"label": "apple", "category": "fruit", "xy": (-0.45, 0.07), "yaw_deg": 0.0, "cz": 0.037,
                  "width": 0.074, "length": 0.074, "height": 0.074, "shape": "round", "mass": 0.18},
    "obj_mouse": {"label": "computer mouse", "category": "office", "xy": (-0.45, -0.06), "yaw_deg": 75.0, "cz": 0.019,
                  "width": 0.062, "length": 0.11, "height": 0.038, "shape": "long", "mass": 0.09,
                  "grasp_z": 0.012},      # V3.72: pads on the flat chassis sides (the shell above is narrower)
    "obj_marker": {"label": "marker pen", "category": "office", "xy": (-0.32, 0.08), "yaw_deg": -18.0, "cz": 0.0105,
                   "width": 0.021, "length": 0.13, "height": 0.021, "shape": "long", "mass": 0.025},
}
SORT_ITEM_NAMES = tuple(SORT_ITEMS)
# two open bins; the robot faces -x, so y < 0 is the robot's left
SORT_BINS = {
    "red": {"centre": (-0.37, -0.25), "inner": (0.24, 0.24), "wall_h": 0.06, "wall_t": 0.006, "floor_t": 0.004,
            "rgba": "0.82 0.12 0.10 1", "side": "left (robot view, y < 0)"},
    "blue": {"centre": (-0.37, 0.25), "inner": (0.24, 0.24), "wall_h": 0.06, "wall_t": 0.006, "floor_t": 0.004,
             "rgba": "0.12 0.30 0.85 1", "side": "right (robot view, y > 0)"},
}


def sort_item_poses(seed=None):
    """V3.72: item (x, y, yaw_deg); NERO_V372_ITEM_SEED (or seed) adds +-15 mm / +-15 deg jitter"""
    s = seed if seed is not None else os.environ.get("NERO_V372_ITEM_SEED")
    out = {n: (it["xy"][0], it["xy"][1], it["yaw_deg"]) for n, it in SORT_ITEMS.items()}
    if s in (None, ""):
        return out
    import numpy as np
    rng = np.random.default_rng(int(s))
    return {n: (x + rng.uniform(-0.015, 0.015), y + rng.uniform(-0.015, 0.015), yaw + rng.uniform(-15, 15))
            for n, (x, y, yaw) in out.items()}


def sort_bin_boxes():
    """(geom name, centre, half size, rgba) of every bin wall / floor"""
    out = []
    for col, b in SORT_BINS.items():
        cx, cy = b["centre"]
        ix, iy = b["inner"][0] / 2.0, b["inner"][1] / 2.0
        t, h, f = b["wall_t"], b["wall_h"], b["floor_t"]
        out += [(f"sort_bin_{col}_floor", (cx, cy, f / 2.0), (ix + t, iy + t, f / 2.0), b["rgba"]),
                (f"sort_bin_{col}_wall_xneg", (cx - ix - t / 2.0, cy, h / 2.0), (t / 2.0, iy + t, h / 2.0), b["rgba"]),
                (f"sort_bin_{col}_wall_xpos", (cx + ix + t / 2.0, cy, h / 2.0), (t / 2.0, iy + t, h / 2.0), b["rgba"]),
                (f"sort_bin_{col}_wall_yneg", (cx, cy - iy - t / 2.0, h / 2.0), (ix, t / 2.0, h / 2.0), b["rgba"]),
                (f"sort_bin_{col}_wall_ypos", (cx, cy + iy + t / 2.0, h / 2.0), (ix, t / 2.0, h / 2.0), b["rgba"])]
    return out


def _yaw_quat(deg):
    a = math.radians(deg) / 2.0
    return (math.cos(a), 0.0, 0.0, math.sin(a))


def _ballast(name, pos, mass):
    """V3.72 HANG BALLAST: an invisible, non-colliding mass inside the item so that its centre of mass lies straight
    below the body origin (= the grasp point).  The backend's CONNECT hold is a ball joint at the body origin: like the
    cups' rim origin (V3.62) this keeps a held item hanging the way it was gripped (without it the banana rolled ~70 deg
    about its long axis on the lift and was placed upside down)."""
    return (f'<geom name="{name}_ballast" type="sphere" pos="{_f(*pos)}" size="0.004" mass="{mass:.4f}" '
            'rgba="0 0 0 0" contype="0" conaffinity="0"/>')


def sort_item_body_xml(name, x, y, yaw_deg, free=True):
    """V3.72 TOP-CENTRE ITEM ORIGIN: the body origin is the item's top centre (like the cups' rim origin, V3.62); the
    backend's CONNECT hold pins this point, the ballast keeps the centre of mass well below it -> a held item hangs
    the way it was gripped.  oz = local z of the item's centre (= cz - height)."""
    it = SORT_ITEMS[name]
    q = _yaw_quat(yaw_deg)
    oz = it["cz"] - it["height"]
    fr = '<freejoint/>' if free else ''
    con = 'friction="1.0 0.02 0.002" condim="6" solref="0.004 1" solimp="0.95 0.99 0.001"'
    head = f'<body name="{name}" pos="{_f(x, y, it["height"] + 0.0005)}" quat="{_f(*q)}">{fr}'
    if name == "obj_banana":                            # 4 capsules on an arc (R 130 mm, 80 deg), bends towards +y
        R, n = 0.13, 4
        th = [math.radians(-40.0 + 80.0 * i / n) for i in range(n + 1)]
        pts = [(R * math.sin(a), R * (1.0 - math.cos(a))) for a in th]
        mb = 0.6 * it["mass"]                           # ballast also cancels the arc's sideways centre of mass
        m = (it["mass"] - mb) / n
        y_arc = sum((pts[i][1] + pts[i + 1][1]) / 2.0 for i in range(n)) / n
        g = "".join(f'<geom name="obj_banana_seg{i}" type="capsule" '
                    f'fromto="{_f(pts[i][0], pts[i][1], oz, pts[i + 1][0], pts[i + 1][1], oz)}" '
                    f'size="0.017" mass="{m:.4f}" rgba="0.97 0.84 0.18 1" {con}/>' for i in range(n))
        g += (f'<geom name="obj_banana_tip0" type="sphere" pos="{_f(pts[0][0], pts[0][1], oz)}" size="0.006" '
              f'rgba="0.30 0.22 0.08 1" contype="0" conaffinity="0" mass="0"/>'
              f'<geom name="obj_banana_tip1" type="sphere" pos="{_f(pts[-1][0], pts[-1][1], oz)}" size="0.006" '
              f'rgba="0.30 0.22 0.08 1" contype="0" conaffinity="0" mass="0"/>')
        g += _ballast(name, (0.0, -y_arc * (it["mass"] - mb) / mb, oz - 0.010), mb)
    elif name == "obj_apple":
        g = (f'<geom name="obj_apple_body" type="sphere" pos="{_f(0, 0, oz)}" size="0.037" mass="{0.55 * it["mass"]:.4f}" '
             f'rgba="0.78 0.10 0.10 1" {con}/>'
             f'<geom name="obj_apple_stem" type="cylinder" pos="{_f(0, 0, oz + 0.040)}" size="0.0025 0.007" '
             'rgba="0.35 0.22 0.10 1" contype="0" conaffinity="0" mass="0"/>'
             + _ballast(name, (0.0, 0.0, oz - 0.022), 0.45 * it["mass"]))
    elif name == "obj_mouse":                           # flat chassis + domed shell, 110 x 62 x 38 mm
        g = (f'<geom name="obj_mouse_base" type="box" pos="{_f(0, 0, oz - 0.009)}" size="0.055 0.031 0.010" '
             f'mass="{0.35 * it["mass"]:.4f}" rgba="0.22 0.23 0.25 1" {con}/>'
             f'<geom name="obj_mouse_shell" type="ellipsoid" pos="{_f(0, 0, oz + 0.001)}" size="0.050 0.028 0.018" '
             f'mass="{0.10 * it["mass"]:.4f}" rgba="0.30 0.31 0.34 1" {con}/>'
             f'<geom name="obj_mouse_wheel" type="cylinder" pos="{_f(0.030, 0, oz + 0.017)}" euler="90 0 0" '
             'size="0.004 0.003" rgba="0.1 0.1 0.1 1" contype="0" conaffinity="0" mass="0"/>'
             + _ballast(name, (0.0, 0.0, oz - 0.014), 0.55 * it["mass"]))
    else:                                               # thick marker, 130 mm x 21 mm, black barrel + orange cap
        # the cap is heavier per length: the ballast also moves the centre of mass back under the grasp point
        g = (f'<geom name="obj_marker_barrel" type="cylinder" fromto="{_f(-0.065, 0, oz, 0.040, 0, oz)}" size="0.0105" '
             f'mass="{0.4 * it["mass"]:.4f}" rgba="0.08 0.08 0.09 1" {con}/>'
             f'<geom name="obj_marker_cap" type="cylinder" fromto="{_f(0.040, 0, oz, 0.065, 0, oz)}" size="0.0110" '
             f'mass="{0.1 * it["mass"]:.4f}" rgba="0.95 0.45 0.05 1" {con}/>'
             + _ballast(name, (-0.002, 0.0, oz - 0.006), 0.5 * it["mass"]))
    return head + g + '</body>'


def sort_item_centre(name, pos, quat_wxyz):
    """V3.72: item centre from its body pose (origin = top centre, body z = up when the item lies normally)"""
    it = SORT_ITEMS[name]
    w, x, y, z = [float(v) for v in quat_wxyz]
    ax = (2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y))
    oz = it["cz"] - it["height"]
    return [float(pos[i]) + oz * ax[i] for i in range(3)]


def sort_world_xml(seed=None, bare=False, poses=None):
    """V3.72: worldbody part of the scene; bare=True = bins + items only (the skills' headless twin has its own table)"""
    poses = poses or sort_item_poses(seed)
    items = "\n    ".join(sort_item_body_xml(n, *poses[n]) for n in SORT_ITEM_NAMES)
    bins = "\n    ".join(f'<geom name="{n}" type="box" pos="{_f(*c)}" size="{_f(*h)}" rgba="{rgba}" '
                          'friction="1.0 0.02 0.002" solref="0.006 1" solimp="0.95 0.99 0.001" contype="1" conaffinity="1"/>'
                          for n, c, h, rgba in sort_bin_boxes())
    legacy = "\n".join(
        f'    <body name="{n}" pos="{-8 - 0.1 * i:.1f} -8 -8"><freejoint/><geom name="{n}_geom" type="sphere" size="0.001" '
        f'mass="0.001" rgba="0 0 0 0" contype="0" conaffinity="0"/></body>' for i, n in enumerate(LEGACY_HIDDEN))
    if bare:
        return bins + "\n    " + items
    return f"""
    <!-- ============================================================
         NERO V3.72 sort_items SCENE - table top z = 0, robot mount unchanged
         ============================================================ -->
    <geom name="v5_grid_floor" type="plane" pos="0 0 -0.28" size="6 6 0.1"
          material="v343_floor_mat" friction="1.2 0.01 0.001" contype="1" conaffinity="1"/>
    <geom name="play_table" type="box" pos="-0.385 0 -0.03" size="0.335 0.38 0.03" material="v343_table_mat"
          friction="1.2 0.01 0.001" solref="0.008 1" solimp="0.96 0.995 0.0005" margin="0.001"
          contype="1" conaffinity="1"/>
    <geom name="nero_support_top" type="box" pos="-0.025 0 -0.008" size="0.025 0.085 0.008"
          rgba="0.15 0.16 0.18 1" contype="0" conaffinity="0"/>
    <!-- open red bin (robot's left) and open blue bin (robot's right) -->
    {bins}
    <!-- banana, apple, computer mouse, thick marker -->
    {items}
{legacy}
    <light name="eggscene_key" pos="0.25 -0.65 2.20" dir="-0.30 0.22 -1" directional="true" castshadow="true"
           diffuse="0.92 0.84 0.72" specular="0.22 0.20 0.18"/>
    <light name="eggscene_fill" pos="-1.20 0.80 1.50" dir="0.55 -0.25 -1" directional="true" castshadow="false"
           diffuse="0.34 0.40 0.48" specular="0.05 0.06 0.08"/>
"""


def make_sort_items_xml(original_xml_path, visual_quality=None):
    """V3.72: same contract as make_store_in_safe_xml for the sort_items scene (SO-101 gripper always)"""
    import re
    import numpy as np
    src = Path(original_xml_path).resolve()
    xml = src.read_text(encoding="utf-8")
    tex = src.parent / "_nero_v343_scene_textures"
    tex.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(34301)
    nz = rng.normal(0.0, 1.0, (256, 256))
    nz = (nz + np.roll(nz, 3, 0) + np.roll(nz, 4, 1)) / 3.0
    _png(tex / "floor_grey.png", np.clip(np.stack([102 + 6 * nz, 110 + 6 * nz, 127 + 6 * nz], -1), 0, 255))
    _png(tex / "table_white.png", np.clip(np.stack([236 + 1.5 * nz, 236 + 1.5 * nz, 234 + 1.5 * nz], -1), 0, 255))
    assets = f"""
    <texture name="v343_sky" type="skybox" builtin="gradient" rgb1="0.54 0.64 0.76" rgb2="0.92 0.90 0.84"
             width="512" height="3072"/>
    <texture name="v343_floor_tex" type="2d" file="{(tex / 'floor_grey.png').as_posix()}"/>
    <texture name="v343_table_tex" type="2d" file="{(tex / 'table_white.png').as_posix()}"/>
    <material name="v343_floor_mat" texture="v343_floor_tex" texrepeat="7 7" reflectance="0.02" specular="0.16"
              shininess="0.08"/>
    <material name="v343_table_mat" texture="v343_table_tex" texrepeat="2.2 1.7" reflectance="0.05"
              specular="0.25" shininess="0.2"/>
"""
    if re.search(r"<asset\b[^>]*/>", xml):
        xml = re.sub(r"<asset\b[^>]*/>", "<asset>\n" + assets + "\n</asset>", xml, count=1)
    elif "<asset" in xml:
        xml = re.sub(r"(<asset\b[^>]*>)", lambda m: m.group(1) + "\n" + assets, xml, count=1)
    else:
        xml = xml.replace("<worldbody>", "<asset>\n" + assets + "\n</asset>\n\n<worldbody>", 1)
    print("[V3.72 SCENE] sort_items: banana / apple / computer mouse / marker pen; open red bin (left) + blue bin "
          "(right), SO-101 gripper", flush=True)
    xml = xml.replace("<worldbody>", "<worldbody>\n" + sort_world_xml(), 1)
    if visual_quality is not None:
        xml = visual_quality(xml)
    import NERO_V364_SO101_Gripper as _G364
    xml = _G364.insert_into_model_xml(xml)
    print(f"[V3.72 SCENE] sort_items: SO-101 gripper mounted on link7 {_G364.describe()}", flush=True)
    eq = "\n        ".join(
        f'<connect name="hold_{o}" body1="link7" body2="{o}" anchor="0 0 0" active="false" '
        f'solref="0.008 1" solimp="0.98 0.995 0.001"/>' for o in hold_object_names())
    if "</equality>" in xml:
        xml = xml.replace("</equality>", "        " + eq + "\n    </equality>", 1)
    else:
        xml = xml.replace("</mujoco>", "    <equality>\n        " + eq + "\n    </equality>\n</mujoco>", 1)
    out = src.with_name("_nero_motor_v343_sort_items_tmp.xml")
    out.write_text(xml, encoding="utf-8")
    return str(out)


def item_in_bin(xyz, margin=0.004):
    """V3.72: colour of the bin whose inner volume holds the point (item CENTRE, see sort_item_centre), else None"""
    for col, b in SORT_BINS.items():
        cx, cy = b["centre"]
        if (abs(xyz[0] - cx) <= b["inner"][0] / 2.0 + margin and abs(xyz[1] - cy) <= b["inner"][1] / 2.0 + margin
                and xyz[2] < b["wall_h"] + 0.02):
            return col
    return None


def sort_task_state(poses):
    """V3.72: {item: bin colour or None} from item centre positions"""
    return {n: item_in_bin(p) for n, p in poses.items()}
