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
}
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


def selected_scene():
    return scene_from_env() or DEFAULT_SCENE


def active_scene():
    s = (os.environ.get("NERO_V343_ACTIVE_SCENE") or "").strip()
    return s if s in SCENES else selected_scene()


def scene_names():
    return list(SCENES)


# ----------------------------------------------------------------------------- run_sim setup data (A2)
def tf_body_map():
    """semantic name -> MuJoCo body; TF child frame = 'sim_' + semantic (object_pose('valuables_box') etc.)."""
    return {"obj_valuables_box": "obj_valuables_box", "obj_safe_handle": "obj_safe_handle",
            "obj_safe_door": "safe_door"}


def hold_object_names():
    return ["obj_valuables_box", "obj_safe_handle"] + list(LEGACY_HIDDEN)


PROTECTED_PREFIXES = ("safe_body_",)


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
    next to it (relative mesh paths stay valid) and return its path."""
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
