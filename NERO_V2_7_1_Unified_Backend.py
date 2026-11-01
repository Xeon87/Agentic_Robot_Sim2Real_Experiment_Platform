#!/usr/bin/env python3
"""NERO V2-7 public-release unified MuJoCo/ROS2 manipulation backend.

The backend owns the deterministic side of the architecture: simulation state,
gripper control, MoveIt/BioIK/TRAC-IK motion skills, skill contracts, physical
postcondition checks, safety state, bounded recovery, and the Tk control panel.
The LLM is *not* trusted as a low-level controller; it only chooses validated
skills and arguments.

V2-7 is intentionally derived from the validated V2-5.2 stack.  The frozen V69
motion core remains separate so the public-release cleanup does not silently
change proven kinematics or trajectory behavior.
"""

# V2-44 R2 SAFE MULTI-MODEL EGG-BASKET RIM 14MM
def _v244r2_lower_egg_basket_rim_if_present(_v244_model):
    """Best-effort geometry-only adjustment. Never prevents MuJoCo/viewer startup."""
    try:
        import mujoco as _v244_mj
        _target=0.014000000
        _half=_target/2.0
        _changed=[]
        for _gid in range(int(_v244_model.ngeom)):
            try:
                _gname=_v244_mj.mj_id2name(_v244_model,_v244_mj.mjtObj.mjOBJ_GEOM,_gid) or ''
                _bid=int(_v244_model.geom_bodyid[_gid])
                _bname=_v244_mj.mj_id2name(_v244_model,_v244_mj.mjtObj.mjOBJ_BODY,_bid) or ''
                _gl=_gname.lower(); _bl=_bname.lower()
                if int(_v244_model.geom_type[_gid]) != int(_v244_mj.mjtGeom.mjGEOM_BOX):
                    continue
                _sx=float(_v244_model.geom_size[_gid][0]); _sy=float(_v244_model.geom_size[_gid][1]); _sz=float(_v244_model.geom_size[_gid][2])
                _thin=min(_sx,_sy) <= 0.020 and max(_sx,_sy) >= 0.045
                _named=('basket' in _gl or 'egg_basket' in _bl)
                _wallword=any(_k in _gl for _k in ('wall','rim','side','front','back','left','right'))
                _excluded=any(_k in _gl for _k in ('floor','bottom','base_plate','basket_base_floor'))
                _gx=float(_v244_model.geom_pos[_gid][0]); _gy=float(_v244_model.geom_pos[_gid][1])
                _bx=float(_v244_model.body_pos[_bid][0]) if _bid < len(_v244_model.body_pos) else 0.0
                _by=float(_v244_model.body_pos[_bid][1]) if _bid < len(_v244_model.body_pos) else 0.0
                _near=(abs((_gx+_bx)-(-0.425)) <= 0.14 and abs((_gy+_by)-(-0.155)) <= 0.14)
                _iswall=(not _excluded) and _thin and ((_named and (_wallword or 'egg_basket' in _bl)) or _near)
                if not _iswall or (2.0*_sz) <= (_target+0.0005):
                    continue
                _bottom=float(_v244_model.geom_pos[_gid][2])-_sz
                _v244_model.geom_size[_gid][2]=_half
                _v244_model.geom_pos[_gid][2]=_bottom+_half
                _changed.append((_gid,_gname,_bname,2.0*_sz,_target))
            except Exception:
                continue
        if _changed:
            print('[V2-44 R2 BASKET RIM] target_height={:.1f}mm changed={}'.format(_target*1000.0,len(_changed)),flush=True)
            for _r in _changed:
                print('[V2-44 R2 BASKET RIM GEOM] id={} geom={} body={} old={:.1f}mm new={:.1f}mm'.format(_r[0],_r[1],_r[2],_r[3]*1000.0,_r[4]*1000.0),flush=True)
        return len(_changed)
    except Exception as _v244_outer:
        # Geometry tuning must never suppress simulator/viewer creation.
        print('[V2-44 R2 GEOMETRY WARN] non-fatal: {}'.format(_v244_outer),flush=True)
        return 0

import sys
import os
import time
import math
import traceback
import threading
import json
import multiprocessing as mp
import tempfile
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mujoco

# ------------------------------------------------------------
# NERO V2-7 Public Release — multi-object, failure-aware manipulation on frozen V69 motion stack
# - Tkinter GUI: avoids Qt/PySide6 event-loop issues
# - multiprocessing "spawn": clean isolation from MuJoCo viewer
# - verified joint-name based mapping: joint1 ... joint7
# - per-joint speed / Wn / Zeta / accel limit
# - gravity ON/OFF, default OFF
# ------------------------------------------------------------

# NERO V3: single-folder release. The MuJoCo model + AgileX meshes stay in the
# established workspace; the launcher passes its exact path via NERO_MODEL_FILE.
MODEL_FILE = (
    os.path.abspath(os.path.expanduser(os.environ["NERO_MODEL_FILE"]))  # no symlink resolve: keep relative mesh paths
    if os.environ.get("NERO_MODEL_FILE")
    else str(Path(__file__).resolve().with_name("nero_motor.xml"))
)
# V3.43 A2 BEGIN
# V3.43 MUJOCO SCENES: the default egg scene ("A1") is built by the unchanged code; any other scene ("A2", e.g.
# store_in_safe) comes from NERO_V343_Scenes.py.  Only setup data differs - the per-step physics loop is shared.
try:
    import NERO_V343_Scenes as _V343
except Exception as _v343_import_exc:      # scene module missing -> always the default scene
    _V343 = None
    print(f"[V3.43 SCENES WARN] NERO_V343_Scenes unavailable ({_v343_import_exc}); default scene only", flush=True)


def _v343_scene_name():
    try:
        return _V343.active_scene() if _V343 is not None else "fill_egg_holder"
    except Exception:
        return "fill_egg_holder"
# V3.43 A2 END

JOINT_LIMITS_RAD = [
    (-2.70526,  2.70526),
    (-1.74,     1.74),
    (-2.75,     2.75),
    (-1.01,     2.14),
    (-2.75,     2.75),
    (-0.73,     0.95),
    (-1.5708,   1.5708),
]


def v341_limit_margins_deg(q):
    """V3.41: distance of each joint to its nearest limit (deg). -> (min_margin_deg, joint_index_1based, list)"""
    ms = []
    for i in range(min(7, len(q))):
        lo, hi = JOINT_LIMITS_RAD[i]
        ms.append(math.degrees(min(float(q[i]) - float(lo), float(hi) - float(q[i]))))
    if not ms:
        return 0.0, 0, []
    k = min(range(len(ms)), key=lambda i: ms[i])
    return round(ms[k], 2), k + 1, [round(v, 1) for v in ms]


def v341_margin_required_deg():
    return float(os.environ.get("NERO_V341_LIMIT_MARGIN_DEG", "8"))


def v341_margin_relaxed_deg():
    return float(os.environ.get("NERO_V341_LIMIT_MARGIN_MIN_DEG", "4"))


V341_STATE = {"insert_replans": {}, "skip_yaw": {}, "corr_margin": None}

# V2 reference speeds: defined as 20% in the new GUI.
V2_MAX_REF_VEL = np.array(
    [0.35, 0.35, 0.35, 0.35, 0.40, 0.40, 0.40],
    dtype=float,
)
FULL_SPEED_REF_VEL = V2_MAX_REF_VEL / 0.20

DEFAULT_WN = np.array([4.0] * 7, dtype=float)
DEFAULT_ZETA = np.array([1.25] * 7, dtype=float)
DEFAULT_MAX_QACC = np.array(
    [5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0],
    dtype=float,
)
DEFAULT_SPEED_PERCENT = np.array([20.0] * 7, dtype=float)


def shared_array(shared):
    return np.array(shared[:], dtype=float)




def _find_official_gripper_assets():
    """
    Locate AgileX official NERO gripper STL files.
    Search common local locations only; no download/network is required at runtime.
    """
    # NERO V3: search relative to the model file (old behavior: backend lived next to it).
    here = Path(os.path.dirname(MODEL_FILE))
    home = Path.home()

    candidate_dirs = [
        here / "assets",
        here / "agilex_nero" / "assets",
        home / "mujoco_ws" / "nero" / "assets",
        home / "agx_arm_sim" / "mujoco" / "agilex_arm" / "agilex_nero" / "assets",
        home / "agx_arm_ws" / "src" / "agx_arm_sim" / "mujoco" / "agilex_arm" / "agilex_nero" / "assets",
    ]

    required = [
        "gripper_flange.stl",
        "gripper_base.stl",
        "gripper_link1.stl",
        "gripper_link2.stl",
    ]

    for d in candidate_dirs:
        if all((d / name).exists() for name in required):
            return d

    # Fallback: bounded recursive searches in likely workspaces.
    roots = [
        home / "mujoco_ws",
        home / "agx_arm_ws",
        home / "agx_arm_sim",
    ]
    for root in roots:
        if not root.exists():
            continue
        for flange in root.rglob("gripper_flange.stl"):
            d = flange.parent
            if all((d / name).exists() for name in required):
                return d

    raise FileNotFoundError(
        "\nAgileX official NERO gripper mesh files were not found.\n"
        "Need these four files in one assets directory:\n"
        "  gripper_flange.stl\n"
        "  gripper_base.stl\n"
        "  gripper_link1.stl\n"
        "  gripper_link2.stl\n"
        "\nThey are in the official agx_arm_sim repository under:\n"
        "mujoco/agilex_arm/agilex_nero/assets/\n"
    )


def add_official_gripper_to_xml(xml_path):
    """
    Add the AgileX official NERO two-finger gripper to an existing arm-only MJCF.

    Geometry / transforms / joint ranges are taken from AgileX's official
    mujoco/agilex_arm/agilex_nero/nero.xml model.

    If the XML already contains gripper_joint1, nothing is changed.
    """
    xml_path = Path(xml_path).resolve()
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Already present -> leave it alone.
    if root.find(".//joint[@name='gripper_joint1']") is not None:
        return str(xml_path)

    assets_dir = _find_official_gripper_assets()

    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        worldbody = root.find("worldbody")
        insert_idx = list(root).index(worldbody) if worldbody is not None else 0
        root.insert(insert_idx, asset)

    existing_meshes = {
        m.get("name")
        for m in asset.findall("mesh")
        if m.get("name")
    }

    mesh_files = {
        "gripper_flange": "gripper_flange.stl",
        "gripper_base": "gripper_base.stl",
        "gripper_link1": "gripper_link1.stl",
        "gripper_link2": "gripper_link2.stl",
    }

    for mesh_name, file_name in mesh_files.items():
        if mesh_name not in existing_meshes:
            ET.SubElement(
                asset,
                "mesh",
                {
                    "name": mesh_name,
                    "file": str((assets_dir / file_name).resolve()),
                },
            )

    link7 = root.find(".//body[@name='link7']")
    if link7 is None:
        raise RuntimeError(
            "Could not find <body name='link7'> in nero_motor.xml; "
            "cannot attach the original gripper."
        )

    # Official flange visual + collision
    ET.SubElement(
        link7, "geom",
        {
            "name": "gripper_flange_visual",
            "pos": "0.032 0 -0.0235",
            "quat": "0.499998 -0.5 0.500002 -0.5",
            "type": "mesh",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
            "density": "0",
            "mesh": "gripper_flange",
        },
    )
    ET.SubElement(
        link7, "geom",
        {
            "name": "gripper_flange_collision",
            "pos": "0.032 0 -0.0235",
            "quat": "0.499998 -0.5 0.500002 -0.5",
            "type": "mesh",
            "mesh": "gripper_flange",
            "friction": "1.2 0.02 0.001",
            "solref": "0.008 1",
            "solimp": "0.95 0.99 0.001",
            "margin": "0.001",
        },
    )

    # Official gripper base visual + collision
    ET.SubElement(
        link7, "geom",
        {
            "name": "gripper_base_visual",
            "pos": "0.0375 -2.02026e-08 -0.0235",
            "quat": "0.499998 -0.5 0.500002 -0.5",
            "type": "mesh",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
            "density": "0",
            "mesh": "gripper_base",
        },
    )
    ET.SubElement(
        link7, "geom",
        {
            "name": "gripper_base_collision",
            "pos": "0.0375 -2.02026e-08 -0.0235",
            "quat": "0.499998 -0.5 0.500002 -0.5",
            "type": "mesh",
            "mesh": "gripper_base",
            "friction": "1.2 0.02 0.001",
            "solref": "0.008 1",
            "solimp": "0.95 0.99 0.001",
            "margin": "0.001",
        },
    )

    # Finger 1 - official transform.
    f1 = ET.SubElement(
        link7, "body",
        {
            "name": "gripper_link1",
            "pos": "0.1733 -5.19024e-07 -0.0235005",
            "quat": "-1.27026e-06 0.707108 0.707105 -1.30815e-06",
        },
    )
    ET.SubElement(
        f1, "inertial",
        {
            "pos": "0.000651232 -0.049193 0.00972259",
            "quat": "0.477798 0.572974 -0.518028 0.418398",
            "mass": "0.025",
            "diaginertia": "7.74531e-05 7.36783e-05 5.0886e-06",
        },
    )
    ET.SubElement(
        f1, "joint",
        {
            "name": "gripper_joint1",
            "pos": "0 0 0",
            "axis": "0 0 1",
            "type": "slide",
            "range": "0 0.05",
            "limited": "true",
        },
    )
    ET.SubElement(
        f1, "geom",
        {
            "type": "mesh",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
            "density": "0",
            "mesh": "gripper_link1",
        },
    )
    # The original finger mesh is visual only for grasping in V15.
    # Actual object contact is handled by the visible rubber pad below.
    ET.SubElement(
        f1, "geom",
        {
            "name": "gripper_finger1_collision_disabled",
            "type": "mesh",
            "mesh": "gripper_link1",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": "3",
            "rgba": "0 0 0 0",
        },
    )

    # --------------------------------------------------------------
    # V19 ATTACHED RUBBER PAD - FINGER 1
    #
    # Keep the pad on the original finger region.  Only the front edge
    # extends toward +Y.  This avoids the detached/floating V18 pad.
    #
    # Local-Y span: about -48 mm .. +12 mm
    # Full pad length: 60 mm
    # --------------------------------------------------------------
    ET.SubElement(
        f1, "geom",
        {
            "name": "finger1_rubber_pad",
            "type": "box",
            "pos": "0 -0.018 -0.0020",
            "size": "0.0145 0.030 0.0020",
            "rgba": "0.90 0.08 0.03 1",
            "density": "0",
            "friction": "10.0 0.30 0.05",
            "condim": "6",
            "solref": "0.003 1",
            "solimp": "0.97 0.997 0.0003",
            "margin": "0.00025",
        },
    )

    # 5 x 8 = 40 raised studs per finger.
    stud_x = (-0.0115, -0.0058, 0.0, 0.0058, 0.0115)
    stud_y = (-0.043, -0.035, -0.027, -0.019,
              -0.011, -0.003, 0.005, 0.010)
    stud_index = 0
    for sx in stud_x:
        for sy in stud_y:
            stud_index += 1
            ET.SubElement(
                f1, "geom",
                {
                    "name": f"finger1_stud_{stud_index:02d}",
                    "type": "sphere",
                    "pos": f"{sx:.4f} {sy:.4f} -0.0052",
                    "size": "0.0017",
                    "rgba": "1.00 0.65 0.00 1",
                    "density": "0",
                    "friction": "10.0 0.30 0.05",
                    "condim": "6",
                    "solref": "0.003 1",
                    "solimp": "0.97 0.997 0.0003",
                    "margin": "0.00020",
                },
            )

    # Finger 2 - official transform.
    f2 = ET.SubElement(
        link7, "body",
        {
            "name": "gripper_link2",
            "pos": "0.1733 -5.19024e-07 -0.0235005",
            "quat": "0.707105 -1.30815e-06 1.30815e-06 -0.707108",
        },
    )
    ET.SubElement(
        f2, "inertial",
        {
            "pos": "0.000651232 -0.049193 0.00972259",
            "quat": "0.477798 0.572974 -0.518028 0.418398",
            "mass": "0.025",
            "diaginertia": "7.74531e-05 7.36783e-05 5.0886e-06",
        },
    )
    ET.SubElement(
        f2, "joint",
        {
            "name": "gripper_joint2",
            "pos": "0 0 0",
            "axis": "0 0 -1",
            "type": "slide",
            "range": "-0.05 0",
            "limited": "true",
        },
    )
    ET.SubElement(
        f2, "geom",
        {
            "type": "mesh",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
            "density": "0",
            "mesh": "gripper_link2",
        },
    )
    # The original finger mesh is visual only for grasping in V15.
    ET.SubElement(
        f2, "geom",
        {
            "name": "gripper_finger2_collision_disabled",
            "type": "mesh",
            "mesh": "gripper_link2",
            "contype": "0",
            "conaffinity": "0",
            "density": "0",
            "group": "3",
            "rgba": "0 0 0 0",
        },
    )

    # --------------------------------------------------------------
    # V19 ATTACHED RUBBER PAD - FINGER 2
    # --------------------------------------------------------------
    ET.SubElement(
        f2, "geom",
        {
            "name": "finger2_rubber_pad",
            "type": "box",
            "pos": "0 -0.018 -0.0020",
            "size": "0.0145 0.030 0.0020",
            "rgba": "0.90 0.08 0.03 1",
            "density": "0",
            "friction": "10.0 0.30 0.05",
            "condim": "6",
            "solref": "0.003 1",
            "solimp": "0.97 0.997 0.0003",
            "margin": "0.00025",
        },
    )

    stud_index = 0
    for sx in stud_x:
        for sy in stud_y:
            stud_index += 1
            ET.SubElement(
                f2, "geom",
                {
                    "name": f"finger2_stud_{stud_index:02d}",
                    "type": "sphere",
                    "pos": f"{sx:.4f} {sy:.4f} -0.0052",
                    "size": "0.0017",
                    "rgba": "1.00 0.65 0.00 1",
                    "density": "0",
                    "friction": "10.0 0.30 0.05",
                    "condim": "6",
                    "solref": "0.003 1",
                    "solimp": "0.97 0.997 0.0003",
                    "margin": "0.00020",
                },
            )

    # Exclude direct finger-to-finger self-contact.
    # The pads can close through their nominal zero-gap pose without
    # generating a huge artificial internal collision impulse.
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")

    ET.SubElement(
        contact,
        "exclude",
        {
            "name": "gripper_finger_self_exclude",
            "body1": "gripper_link1",
            "body2": "gripper_link2",
        },
    )

    # Official coupling: q(gripper_joint2) = -q(gripper_joint1)
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")

    ET.SubElement(
        equality,
        "joint",
        {
            "name": "gripper_coupling",
            "joint1": "gripper_joint2",
            "joint2": "gripper_joint1",
            "polycoef": "0 -1 0 0 0",
        },
    )

    # Official AgileX gripper servo:
    # <position name="gripper" joint="gripper_joint1"
    #           kp="40" kv="5"; V12 simulation force limit = +/-20 N
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")

    if root.find(".//actuator/*[@name='gripper']") is None:
        ET.SubElement(
            actuator,
            "position",
            {
                "name": "gripper",
                "joint": "gripper_joint1",
                "kp": "40",
                "kv": "5",
                "forcerange": "-20 20",
                "forcelimited": "true",
                "ctrlrange": "0 0.05",
                "ctrllimited": "true",
            },
        )

    # V16 direct-force actuator.
    # q(gripper_joint1) increases when opening, so NEGATIVE force closes.
    # This actuator is separate from the position actuator above.
    if root.find(".//actuator/*[@name='gripper_force']") is None:
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": "gripper_force",
                "joint": "gripper_joint1",
                "gear": "1",
                "ctrlrange": "-50 50",
                "ctrllimited": "true",
                "forcerange": "-50 50",
                "forcelimited": "true",
            },
        )

    out = xml_path.with_name("_nero_motor_v10_stable_arm_safe_gripper_tmp.xml")
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return str(out)




def _v333_visual_quality(xml):
    """V3.33: sharp, softer shadows for the cameras - shadow map 8192 (the default 4096 gave blocky arm shadows on the
    table) and a brighter headlight ambient (shadows lighter grey, not near-black). Edits an existing <visual>
    block (attributes of <quality>/<headlight>) or adds one."""
    sh = os.environ.get("NERO_V333_SHADOWSIZE", "8192")
    amb = os.environ.get("NERO_V333_HEADLIGHT_AMBIENT", "0.25 0.25 0.25")
    dif = os.environ.get("NERO_V333_HEADLIGHT_DIFFUSE", "0.30 0.30 0.30")

    def _set_attr(tag_xml, attr, val):
        pat = rf"""\b{attr}\s*=\s*(["'])[^"']*\1"""
        if re.search(pat, tag_xml):
            return re.sub(pat, f'{attr}="{val}"', tag_xml, count=1)
        return tag_xml.replace("<" + tag_xml[1:].split()[0].rstrip("/>"), "<" + tag_xml[1:].split()[0].rstrip("/>") + f' {attr}="{val}"', 1)
    m = re.search(r"<visual\b[^>]*/>", xml)
    if m:
        xml = xml[:m.start()] + "<visual></visual>" + xml[m.end():]
    if "<visual" not in xml:
        block = f'<visual><quality shadowsize="{sh}"/><headlight ambient="{amb}" diffuse="{dif}" specular="0.1 0.1 0.1"/></visual>'
        k = xml.find("<asset")
        k = k if k >= 0 else xml.find("<worldbody")
        return xml[:k] + block + "\n" + xml[k:]
    vs = xml.index("<visual")
    ve = xml.index("</visual>", vs)
    inner = xml[vs:ve]
    for tag, attrs in (("quality", {"shadowsize": sh}), ("headlight", {"ambient": amb, "diffuse": dif})):
        mt = re.search(rf"<{tag}\b[^>]*/?>", inner)
        if mt:
            t = mt.group(0)
            for k_, v_ in attrs.items():
                t = _set_attr(t, k_, v_)
            inner = inner[:mt.start()] + t + inner[mt.end():]
        else:
            first = inner.index(">") + 1
            inner = inner[:first] + f"<{tag} " + " ".join(f'{k_}="{v_}"' for k_, v_ in attrs.items()) + "/>" + inner[first:]
    return xml[:vs] + inner + xml[ve:]


def make_grid_world_xml(original_xml_path):
    """Build the NERO V2-7 RoboDojo-aligned egg-handling tabletop scene.

    Scene-only V2-7 Egg Scene update.  The robot model, controller, safety
    thresholds, MoveIt/BioIK/TRAC-IK behavior, GUI, and skill implementations
    remain unchanged.  This function replaces the former red/blue sorting
    workcell with a household-style egg basket + 8-slot egg carton scene.

    The four eggs intentionally remain rigid in this version, matching the
    original RoboDojo ``fill_egg_holder`` benchmark assumption.  Fragility,
    tactile pressure, slip, and damage models are intentionally deferred to a
    later research step so visual/scene changes do not alter manipulation
    physics at the same time.
    """
    # V3.43 A2 BEGIN
    if _v343_scene_name() != "fill_egg_holder":
        return _V343.make_store_in_safe_xml(original_xml_path, visual_quality=_v333_visual_quality)
    # V3.43 A2 END
    src_path = Path(original_xml_path).resolve()
    xml = src_path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Runtime-generated texture maps.
    # Keep them next to nero_motor.xml so the existing public launcher
    # needs no asset-copy changes and all relative AgileX mesh paths remain
    # untouched.  PNG writing uses only Python stdlib + NumPy already used
    # by the backend; no Pillow/OpenCV dependency is introduced.
    # ------------------------------------------------------------------
    import struct
    import zlib

    tex_dir = src_path.parent / "_nero_egg_scene_textures"
    tex_dir.mkdir(parents=True, exist_ok=True)

    def _write_rgb_png(path, rgb):
        rgb = np.asarray(rgb, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("texture image must be HxWx3 uint8")
        h, w, _ = rgb.shape
        raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(h))

        def chunk(tag, payload):
            return (
                struct.pack(">I", len(payload))
                + tag
                + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
            )

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0),
        )
        png += chunk(b"IDAT", zlib.compress(raw, level=9))
        png += chunk(b"IEND", b"")
        path.write_bytes(png)

    rng = np.random.default_rng(69026)

    # Warm medium-oak texture: broad grain + fine pores + subtle plank bands.
    h = w = 512
    yy, xx = np.mgrid[0:h, 0:w]
    grain = (
        0.58 * np.sin(xx * 0.065 + 0.65 * np.sin(yy * 0.018))
        + 0.26 * np.sin(xx * 0.19 + yy * 0.012)
        + 0.16 * np.sin(xx * 0.43 + yy * 0.025)
    )
    plank = 0.10 * np.cos(yy * (2.0 * np.pi / 128.0))
    noise = rng.normal(0.0, 0.09, (h, w))
    g = np.clip((grain + plank + noise + 1.7) / 3.4, 0.0, 1.0)
    # V3.33: neutral, slightly cool grey floor (concrete-like). The former oak floor had the hue / saturation of
    # brown eggs and produced false egg detections for the RGB-only wrist camera. NERO_V333_FLOOR=wood restores it.
    if os.environ.get("NERO_V333_FLOOR", "grey") == "wood":
        wood = np.empty((h, w, 3), dtype=np.float32)
        wood[..., 0] = 112 + 78 * g
        wood[..., 1] = 62 + 58 * g
        wood[..., 2] = 32 + 36 * g
        wood_path = tex_dir / "wood_oak.png"
    else:
        fine_n = rng.normal(0.0, 1.0, (h, w))
        fine_n = (fine_n + np.roll(fine_n, 3, 0) + np.roll(fine_n, 4, 1)) / 3.0
        wood = np.empty((h, w, 3), dtype=np.float32)
        wood[..., 0] = 102 + 6 * fine_n
        wood[..., 1] = 110 + 6 * fine_n
        wood[..., 2] = 127 + 6 * fine_n
        wood_path = tex_dir / "floor_grey_v333.png"
    wood = np.clip(wood, 0, 255).astype(np.uint8)
    _write_rgb_png(wood_path, wood)

    # Light marble tabletop: warm off-white stone with subtle grey/taupe veins.
    # The pattern is intentionally low-contrast so objects remain easy to see.
    hm = wm = 512
    yym, xxm = np.mgrid[0:hm, 0:wm]
    base_noise = rng.normal(0.0, 1.0, (hm, wm))
    smooth = (base_noise + np.roll(base_noise, 5, 0) + np.roll(base_noise, 9, 1)
              + np.roll(base_noise, 17, 0) + np.roll(base_noise, 23, 1)) / 5.0
    vein_coord = xxm * 0.020 + 0.75 * np.sin(yym * 0.018) + 0.35 * np.sin((xxm+yym) * 0.010)
    veins = np.exp(-((np.sin(vein_coord) / 0.16) ** 2))
    fine = np.exp(-((np.sin(vein_coord * 2.3 + 1.2) / 0.10) ** 2))
    stone = np.empty((hm, wm, 3), dtype=np.float32)
    if os.environ.get("NERO_V333_TABLE", "white") == "marble":
        stone[..., 0] = 232 + 5*smooth - 24*veins - 10*fine
        stone[..., 1] = 229 + 5*smooth - 22*veins - 9*fine
        stone[..., 2] = 221 + 5*smooth - 19*veins - 8*fine
        marble_path = tex_dir / "marble_warm_white.png"
    else:
        # V3.33: plain matte white table (no veins: dark veins look like holes / shadows to an RGB camera)
        stone[..., 0] = 236 + 1.5*smooth
        stone[..., 1] = 236 + 1.5*smooth
        stone[..., 2] = 234 + 1.5*smooth
        marble_path = tex_dir / "table_white_v333.png"
    _write_rgb_png(marble_path, np.clip(stone, 0, 255).astype(np.uint8))

    # Woven basket texture: alternating light/dark reeds with low-frequency noise.
    h2 = w2 = 384
    yy2, xx2 = np.mgrid[0:h2, 0:w2]
    weave = (((xx2 // 12) + (yy2 // 7)) % 2).astype(float)
    weave += 0.22 * np.sin(xx2 * 0.11) + rng.normal(0, 0.07, (h2, w2))
    weave = np.clip(weave, 0.0, 1.3) / 1.3
    wicker = np.empty((h2, w2, 3), dtype=np.float32)
    wicker[..., 0] = 120 + 90 * weave
    wicker[..., 1] = 72 + 64 * weave
    wicker[..., 2] = 35 + 42 * weave
    wicker_path = tex_dir / "wicker.png"
    _write_rgb_png(wicker_path, np.clip(wicker, 0, 255).astype(np.uint8))

    # Egg shell: very mild speckle, intentionally not glossy/plastic looking.
    h3 = w3 = 256
    speckle = rng.normal(0, 4.5, (h3, w3))
    eggtex = np.empty((h3, w3, 3), dtype=np.float32)
    eggtex[..., 0] = 224 + speckle
    eggtex[..., 1] = 189 + 0.8 * speckle
    eggtex[..., 2] = 143 + 0.6 * speckle
    dots = rng.random((h3, w3)) < 0.008
    eggtex[dots] *= np.array([0.64, 0.58, 0.52], dtype=np.float32)
    egg_path = tex_dir / "egg_shell.png"
    _write_rgb_png(egg_path, np.clip(eggtex, 0, 255).astype(np.uint8))

    # Recycled fiber carton texture.
    h4 = w4 = 384
    fiber = rng.normal(0, 1, (h4, w4))
    fiber = (fiber + np.roll(fiber, 2, 0) + np.roll(fiber, 3, 1)) / 3.0
    carton = np.empty((h4, w4, 3), dtype=np.float32)
    carton[..., 0] = 160 + 13 * fiber
    carton[..., 1] = 153 + 12 * fiber
    carton[..., 2] = 137 + 10 * fiber
    carton_path = tex_dir / "carton_fiber.png"
    _write_rgb_png(carton_path, np.clip(carton, 0, 255).astype(np.uint8))

    # V3.33 supermarket moulded-pulp egg carton: visual mesh from the holder CAD (NERO_V326_Wrist_Camera)
    import NERO_V326_Wrist_Camera as _v333w
    _v333_carton_path = tex_dir / "egg_carton_pulp_v333.obj"
    _v333_carton_path.write_text(_v333w.carton_mesh_obj(), encoding="utf-8")
    _v333_carton_obj = _v333_carton_path.as_posix()
    _v333_pulp_rgba = os.environ.get("NERO_V333_PULP_RGBA", "0.41 0.40 0.38 1")

    # ---- Add / extend <asset> --------------------------------------------
    asset_block = f"""
    <!-- NERO V2-7 Egg Scene visual environment -->
    <texture name="eggscene_sky"
             type="skybox" builtin="gradient"
             rgb1="0.54 0.64 0.76" rgb2="0.92 0.90 0.84"
             width="512" height="3072"/>

    <texture name="eggscene_wood_tex" type="2d"
             file="{wood_path.as_posix()}"/>
    <texture name="eggscene_marble_tex" type="2d"
             file="{marble_path.as_posix()}"/>
    <texture name="eggscene_wicker_tex" type="2d"
             file="{wicker_path.as_posix()}"/>
    <texture name="eggscene_shell_tex" type="2d"
             file="{egg_path.as_posix()}"/>
    <texture name="eggscene_carton_tex" type="2d"
             file="{carton_path.as_posix()}"/>

    <material name="eggscene_floor_wood_mat" texture="eggscene_wood_tex"
              texrepeat="7.0 7.0" texuniform="false"
              reflectance="0.020" specular="0.16" shininess="0.08"
              rgba="1 1 1 1"/>
    <material name="eggscene_marble_mat" texture="eggscene_marble_tex"
              texrepeat="2.2 1.7" texuniform="false"
              reflectance="0.075" specular="0.32" shininess="0.22"
              rgba="1 1 1 1"/>
    <material name="eggscene_wicker_mat" texture="eggscene_wicker_tex"
              texrepeat="5 3" texuniform="true"
              reflectance="0.015" specular="0.08" shininess="0.04"
              rgba="1 1 1 1"/>
    <material name="eggscene_carton_mat" texture="eggscene_carton_tex"
              texrepeat="3 2" texuniform="true"
              reflectance="0.0" specular="0.035" shininess="0.02"
              rgba="0.94 0.94 0.94 1"/>
    <!-- V3.33 supermarket moulded-pulp carton (visual mesh written next to the textures) -->
    <material name="eggscene_pulp_mat" reflectance="0.0" specular="0.02" shininess="0.02"
              rgba="{_v333_pulp_rgba}"/>
    <mesh name="eggscene_carton_mesh" file="{_v333_carton_obj}"/>
    <material name="eggscene_egg1_mat" texture="eggscene_shell_tex"
              texrepeat="2 2" reflectance="0.015" specular="0.16"
              shininess="0.08" rgba="0.88 0.62 0.37 1"/>
    <material name="eggscene_egg2_mat" texture="eggscene_shell_tex"
              texrepeat="2 2" reflectance="0.015" specular="0.15"
              shininess="0.07" rgba="0.82 0.55 0.31 1"/>
    <material name="eggscene_egg3_mat" texture="eggscene_shell_tex"
              texrepeat="2 2" reflectance="0.015" specular="0.14"
              shininess="0.07" rgba="0.93 0.70 0.44 1"/>
    <material name="eggscene_egg4_mat" texture="eggscene_shell_tex"
              texrepeat="2 2" reflectance="0.015" specular="0.15"
              shininess="0.08" rgba="0.86 0.58 0.33 1"/>
"""

    if re.search(r"<asset\b[^>]*/>", xml):
        xml = re.sub(
            r"<asset\b[^>]*/>",
            "<asset>\n" + asset_block + "\n</asset>",
            xml,
            count=1,
        )
    elif "<asset" in xml:
        xml = re.sub(r"(<asset\b[^>]*>)", r"\1\n" + asset_block, xml, count=1)
    else:
        xml = xml.replace(
            "<worldbody>",
            "<asset>\n" + asset_block + "\n</asset>\n\n<worldbody>",
            1,
        )

    # V2-50 CLEAN V2-43 FORK / NO PHYSICAL BASKET
    # Physical egg basket removed. Historical center/radius retained only
    # because unchanged V2-43 semantic/scoring code may still read them.
    basket_cx, basket_cy = -0.425, -0.155
    basket_r = 0.092
    basket_xml = ""

    # Four brown eggs: small, realistic manufacturing/natural variation in
    # width, height, mass, shell tone and initial tilt.  They remain rigid.
    egg_specs = [
        ("obj_egg1", -0.451, -0.181, 0.030, 0.0220, 0.0212, 0.0310, 0.060, "eggscene_egg1_mat",  0.10, -0.06,  0.16),
        ("obj_egg2", -0.399, -0.181, 0.031, 0.0228, 0.0217, 0.0320, 0.063, "eggscene_egg2_mat", -0.08,  0.09, -0.12),
        ("obj_egg3", -0.452, -0.130, 0.030, 0.0217, 0.0209, 0.0305, 0.058, "eggscene_egg3_mat",  0.06,  0.12,  0.08),
        ("obj_egg4", -0.399, -0.129, 0.031, 0.0231, 0.0219, 0.0324, 0.064, "eggscene_egg4_mat", -0.11, -0.05, -0.15),
    ]
    egg_parts = []
    for name, x, y, z, sx, sy, sz, mass, material, er, ep, eyaw in egg_specs:
        egg_parts.append(f"""
    <body name="{name}" pos="{x:.4f} {y:.4f} {z:.4f}" euler="{er:.4f} {ep:.4f} {eyaw:.4f}">
        <freejoint/>
        <geom name="{name}_geom" type="ellipsoid"
              size="{sx:.4f} {sy:.4f} {sz:.4f}"
              material="{material}" mass="{mass:.4f}"
              friction="1.05 0.025 0.002" condim="6"
              solref="0.005 1" solimp="0.96 0.995 0.0008"
              margin="0.00035"/>
    </body>""")
    eggs_xml = "\n".join(egg_parts)

    # 8-slot carton: 4 columns x 2 rows.  Each pocket uses a low base plus
    # six short radial posts.  This gives an obvious visual/physical pocket
    # without introducing external meshes or changing the existing launcher.
    holder_cx, holder_cy = -0.420, 0.155
    slot_xs = (-0.510, -0.450, -0.390, -0.330)
    slot_ys = (0.123, 0.187)
    holder_parts = [
        '<geom name="egg_holder_base" type="box" pos="-0.420 0.155 0.006" '
        'size="0.125 0.072 0.006" material="eggscene_carton_mat" '
        'friction="1.1 0.02 0.001" contype="1" conaffinity="1"/>',
        '<geom name="egg_holder_rim_xneg" type="box" pos="-0.548 0.155 0.025" '
        'size="0.004 0.076 0.025" material="eggscene_carton_mat" contype="1" conaffinity="1"/>',
        '<geom name="egg_holder_rim_xpos" type="box" pos="-0.292 0.155 0.025" '
        'size="0.004 0.076 0.025" material="eggscene_carton_mat" contype="1" conaffinity="1"/>',
        '<geom name="egg_holder_rim_yneg" type="box" pos="-0.420 0.079 0.025" '
        'size="0.124 0.004 0.025" material="eggscene_carton_mat" contype="1" conaffinity="1"/>',
        '<geom name="egg_holder_rim_ypos" type="box" pos="-0.420 0.231 0.025" '
        'size="0.124 0.004 0.025" material="eggscene_carton_mat" contype="1" conaffinity="1"/>',
    ]
    slot_index = 0
    for sy in slot_ys:
        for sx in slot_xs:
            slot_index += 1
            holder_parts.append(
                f'<geom name="egg_holder_slot_{slot_index}_well" type="ellipsoid" '
                f'pos="{sx:.4f} {sy:.4f} 0.0105" size="0.024 0.021 0.004" '
                f'rgba="0.34 0.33 0.30 1" contype="0" conaffinity="0"/>'
            )
            for k in range(6):
                a = 2.0 * math.pi * k / 6.0
                px = sx + 0.027 * math.cos(a)
                py = sy + 0.024 * math.sin(a)
                holder_parts.append(
                    f'<geom name="egg_holder_slot_{slot_index}_post_{k}" type="capsule" '
                    f'fromto="{px:.4f} {py:.4f} 0.011 {px:.4f} {py:.4f} 0.044" '
                    f'size="0.0040" material="eggscene_carton_mat" '
                    f'friction="0.9 0.015 0.001" contype="1" conaffinity="1"/>'
                )
    # V2-52 MUJOCO EGG-HOLDER LID REMOVED
    # Lid panel + inner visual removed from MuJoCo only.
    # Holder base/rims/slot wells/posts and all manipulation logic are unchanged.
    # V3.33: the physical carton (base, rims, 6 posts per slot - unchanged, the whole placement pipeline is tuned
    # on it) becomes invisible (group 3, alpha 0); what cameras and the viewer see is the moulded-pulp carton mesh
    # whose cells hug the egg at the post-top height (the egg rests where it rested before).
    if os.environ.get("NERO_V333_PULP_CARTON", "1") != "0":
        _hidden = []
        for _g in holder_parts:
            if "_well\"" in _g:
                continue                                  # the old dark well discs are gone
            _g = _g.replace('material="eggscene_carton_mat"', 'rgba="0 0 0 0" group="3"')
            _hidden.append(_g)
        holder_parts = _hidden + [
            '<geom name="egg_holder_pulp_visual" type="mesh" mesh="eggscene_carton_mesh" '
            'pos="-0.420 0.155 0.0" material="eggscene_pulp_mat" contype="0" conaffinity="0" '
            'group="0" density="0"/>']
    holder_xml = "\n    ".join(holder_parts)

    # Invisible compatibility bodies preserve the validated V2-7 runtime
    # assumptions (TF publisher, hold equality lookup, legacy GUI selectors)
    # without leaving the old sorting objects visible in the household scene.
    legacy_xml = """
    <body name="obj_red_cube" pos="-8 -8 -8"><freejoint/><geom name="obj_red_cube_geom" type="sphere" size="0.001" mass="0.001" rgba="0 0 0 0" contype="0" conaffinity="0"/></body>
    <body name="obj_blue_cube" pos="-8.1 -8 -8"><freejoint/><geom name="obj_blue_cube_geom" type="sphere" size="0.001" mass="0.001" rgba="0 0 0 0" contype="0" conaffinity="0"/></body>
    <body name="obj_red_ball" pos="-8.2 -8 -8"><freejoint/><geom name="obj_red_ball_geom" type="sphere" size="0.001" mass="0.001" rgba="0 0 0 0" contype="0" conaffinity="0"/></body>
    <body name="obj_blue_ball" pos="-8.3 -8 -8"><freejoint/><geom name="obj_blue_ball_geom" type="sphere" size="0.001" mass="0.001" rgba="0 0 0 0" contype="0" conaffinity="0"/></body>
    """

    world_block = f"""
    <!-- ============================================================
         NERO V2-7 EGG HANDLING SCENE — RoboDojo-aligned task semantics
         Table top remains z=0.000 m exactly, preserving the proven V69
         robot mounting plane and all arm-controller assumptions.
         ============================================================ -->

    <geom name="v5_grid_floor" type="plane" pos="0 0 -0.28" size="6 6 0.1"
          material="eggscene_floor_wood_mat" friction="1.2 0.01 0.001"
          contype="1" conaffinity="1"/>

    <geom name="play_table" type="box" pos="-0.385 0 -0.03"
          size="0.335 0.38 0.03" material="eggscene_marble_mat"
          friction="1.2 0.01 0.001" solref="0.008 1"
          solimp="0.96 0.995 0.0005" margin="0.001"
          contype="1" conaffinity="1"/>

    <!-- Existing NERO clamp/support geometry retained, visual only. -->
    <geom name="nero_support_top" type="box" pos="-0.025 0 -0.008"
          size="0.025 0.085 0.008" rgba="0.15 0.16 0.18 1"
          contype="0" conaffinity="0"/>
    <geom name="nero_support_vertical" type="box" pos="-0.055 0 -0.075"
          size="0.008 0.085 0.067" rgba="0.15 0.16 0.18 1"
          contype="0" conaffinity="0"/>
    <geom name="nero_support_lower_jaw" type="box" pos="-0.080 0 -0.134"
          size="0.030 0.085 0.008" rgba="0.15 0.16 0.18 1"
          contype="0" conaffinity="0"/>

    <!-- Wicker basket with four brown eggs. -->
    {basket_xml}
    {eggs_xml}

    <!-- 8-slot recycled-fiber egg carton, open. -->
    {holder_xml}

    <!-- Legacy hidden bodies only for unchanged V2-7 runtime compatibility. -->
    {legacy_xml}

    <!-- Warm key + cool fill light for readable shape/contact shadows. -->
    <light name="eggscene_key" pos="0.25 -0.65 2.20" dir="-0.30 0.22 -1"
           directional="true" castshadow="true"
           diffuse="0.92 0.84 0.72" specular="0.22 0.20 0.18"/>
    <light name="eggscene_fill" pos="-1.20 0.80 1.50" dir="0.55 -0.25 -1"
           directional="true" castshadow="false"
           diffuse="0.34 0.40 0.48" specular="0.05 0.06 0.08"/>
"""

    xml = xml.replace("<worldbody>", "<worldbody>\n" + world_block, 1)
    xml = _v333_visual_quality(xml)

    # Preserve the existing V2-7 hold constraints for the hidden compatibility
    # objects.  The new eggs are not attached to the old grasp logic yet; this
    # revision intentionally changes only the scene.
    hold_object_names = [
        "obj_egg1", "obj_egg2", "obj_egg3", "obj_egg4",
        "obj_red_cube", "obj_blue_cube", "obj_red_ball", "obj_blue_ball",
    ]
    hold_eq_lines = []
    for obj_name in hold_object_names:
        hold_eq_lines.append(
            f'<connect name="hold_{obj_name}" body1="link7" body2="{obj_name}" '
            f'anchor="0 0 0" active="false" solref="0.008 1" '
            f'solimp="0.98 0.995 0.001"/>'
        )
    hold_eq_xml = "\n        ".join(hold_eq_lines)
    if "</equality>" in xml:
        xml = xml.replace(
            "</equality>", "        " + hold_eq_xml + "\n    </equality>", 1
        )
    else:
        xml = xml.replace(
            "</mujoco>",
            "    <equality>\n        " + hold_eq_xml + "\n    </equality>\n</mujoco>",
            1,
        )

    out_path = src_path.with_name("_nero_motor_v26_egg_scene_tmp.xml")
    out_path.write_text(xml, encoding="utf-8")
    return str(out_path)


def run_sim(
    target_shared,
    actual_shared,
    speed_shared,
    wn_shared,
    zeta_shared,
    qacc_shared,
    gravity_enabled,
    gripper_target_mm,
    gripper_actual_mm,
    gripper_force_mode,
    gripper_force_N,
    clamp_speed_mult_shared,
    anti_slip_boost_shared,
    anti_slip_slip_mm_s_shared,
    anti_slip_mu_shared,
    contact_left_n_shared,
    contact_right_n_shared,
    contact_tangent_n_shared,
    contact_motor_cmd_shared,
    hold_enabled_shared,
    hold_active_shared,
    hold_name_shared,
    hold_ready_shared,
    hold_now_request,
    hold_off_request,
    hold_status_code_shared,
    reset_zero_request,
    recover_request,
    safety_trip_count,
    safety_state,
    sim_ready,
    sim_error_code,
    ros_bridge_ready,
    ros_cmd_count,
    ros_feedback_count,
    ros_scene_count,
    ros_last_cmd_age,
    ros_last_cmd_joint_count,
    ros_controller_state_rx_shared,
    ros_fallback_joint_state_rx_shared,
    ros_last_command_source_shared,
    ros_arm_command_gate_shared,
    stop_event,
    v308_qpos=None,
    v308_meta=None,
    v308_model_path=None,
):
    """
    MuJoCo process.
    Deliberately imports mujoco.viewer INSIDE this spawned child,
    so Tk/Tcl and GLFW never share the same GUI process.
    """
    try:
        import mujoco.viewer

        visual_model_file = make_grid_world_xml(MODEL_FILE)
        visual_model_file = _v272_add_egg_number_labels_to_mjcf(visual_model_file)
        visual_model_file = add_official_gripper_to_xml(visual_model_file)
        # V3.08 SIMULATED D435i CAMERA: inject the fixed camera so the viewer and
        # the separate camera process share one model file. Physics unaffected.
        if v308_model_path is not None:
            try:
                import NERO_V308_Camera_Perception as _v308
                if _v308.enabled():
                    visual_model_file = _v308.inject_camera_into_mjcf(visual_model_file)
                    print(f"[V3.08 CAMERA] injected {_v308.CAMERA_NAME} into {visual_model_file}", flush=True)
            except Exception as _v308_exc:
                print(f"[V3.08 CAMERA WARN] camera injection failed, perception disabled: {_v308_exc}", flush=True)
            # V3.26 WRIST CAMERA: a <camera> element inside link7 (no geoms, physics + fixed camera unchanged)
            try:
                import NERO_V326_Wrist_Camera as _v326
                if _v326.enabled():
                    visual_model_file = _v326.inject_wrist_camera_into_mjcf(visual_model_file)
                    print(f"[V3.26/V3.30 WRIST CAMERA] injected {_v326.WRIST_CAMERA_NAME} + {_v326.USB_CAMERA_NAME} into link7"
                          f" (+ visual camera models) -> {visual_model_file}", flush=True)
                    for _pf3311 in _v326.PROFILES:
                        _c3311 = _v326.wrist_config(_pf3311)
                        print(f"[V3.31.1 WRIST MOUNT] {_pf3311}: optical centre {_c3311['mount_pos']} in link7 "
                              f"(on the gripper base side face) | faces from {_c3311.get('mount_source')}", flush=True)
            except Exception as _v326_exc:
                print(f"[V3.26 WRIST CAMERA WARN] wrist camera injection failed, wrist mode unavailable: {_v326_exc}", flush=True)
        model = mujoco.MjModel.from_xml_path(visual_model_file)
        try:
            _v244r2_lower_egg_basket_rim_if_present(model)
        except Exception as _v244r2_e:
            print('[V2-44 R2 GEOMETRY WARN] non-fatal call: {}'.format(_v244r2_e), flush=True)
        data = mujoco.MjData(model)

        model.opt.timestep = 0.001
        dt = float(model.opt.timestep)

        joint_names = [f"joint{i}" for i in range(1, 8)]
        qpos_addr = []
        dof_addr = []

        # IMPORTANT: keep the verified mapping by joint NAME,
        # never assume qpos[i] == joint i.
        for name in joint_names:
            jid = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            if jid < 0:
                raise RuntimeError(f"Joint not found in model: {name}")

            qpos_addr.append(int(model.jnt_qposadr[jid]))
            dof_addr.append(int(model.jnt_dofadr[jid]))

        # Original AgileX two-finger gripper.
        gripper_jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint1"
        )
        if gripper_jid < 0:
            raise RuntimeError("gripper_joint1 not found after gripper injection")

        gripper_qpos_addr = int(model.jnt_qposadr[gripper_jid])
        gripper_dof_addr = int(model.jnt_dofadr[gripper_jid])

        gripper2_jid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint2"
        )
        if gripper2_jid < 0:
            raise RuntimeError("gripper_joint2 not found after gripper injection")

        gripper2_qpos_addr = int(model.jnt_qposadr[gripper2_jid])
        gripper2_dof_addr = int(model.jnt_dofadr[gripper2_jid])


        # ==============================================================
        # V39 ROS2 / MOVEIT2 BRIDGE
        #
        # Official AgileX MoveIt2 publishes interpolated commands on:
        #   /control/joint_states
        #
        # This MuJoCo backend publishes actual state on:
        #   /feedback/joint_states
        #
        # With:
        #   demo.launch.py arm_type:=nero effector_type:=agx_gripper
        #                  follow:=true
        #
        # MoveIt/OMPL owns IK, planning, self/environment collision.
        # V31 remains the low-level MuJoCo execution controller.
        # ==============================================================
        ros_node = None
        ros_feedback_pub = None
        ros_scene_pub = None
        ros_joint_state_type = None
        ros_planning_scene_type = None
        ros_ready = False
        ros_last_spin_wall = 0.0
        ros_last_feedback_wall = 0.0
        ros_last_scene_wall = 0.0
        ros_last_command_wall = 0.0
        ros_cmd_seen = False
        # V41 safety latch:
        # if the arm penetrates a protected surface, stop accepting
        # further MoveIt arm commands until the user presses
        # STOP+RECOVER or HARD RESET.  This prevents the trajectory
        # controller from repeatedly driving back into the collision
        # after V14 rollback.
        ros_motion_fault_latched = False

        # Latest MoveIt command is consumed directly into V31 shared
        # targets. Callback runs in the SAME spawned MuJoCo process via
        # spin_once(), so no extra controller process touches MuJoCo.
        def _apply_arm_command(name_to_pos, source):
            nonlocal ros_last_command_wall, ros_cmd_seen
            nonlocal ros_motion_fault_latched

            updated = 0

            if (
                not ros_motion_fault_latched
                and bool(ros_arm_command_gate_shared.value)
            ):
                for i, jname in enumerate(joint_names):
                    if jname in name_to_pos:
                        qcmd = float(
                            np.clip(
                                name_to_pos[jname],
                                JOINT_LIMITS_RAD[i][0],
                                JOINT_LIMITS_RAD[i][1],
                            )
                        )
                        target_shared[i] = qcmd
                        updated += 1

            if updated > 0:
                ros_cmd_count.value += 1
                ros_last_cmd_joint_count.value = int(updated)
                ros_last_command_wall = time.perf_counter()
                ros_cmd_seen = True
                ros_last_command_source_shared.value = (
                    1 if source == "/arm_controller/controller_state" else 2
                )

            return updated

        def _moveit_joint_state_callback(msg):
            ros_fallback_joint_state_rx_shared.value += 1
            name_to_pos = {
                str(n): float(p)
                for n, p in zip(msg.name, msg.position)
            }
            _apply_arm_command(
                name_to_pos,
                "/control/joint_states",
            )

        def _controller_state_callback(msg):
            ros_controller_state_rx_shared.value += 1

            names = list(getattr(msg, "joint_names", []))
            point = None
            if hasattr(msg, "desired"):
                point = msg.desired
            elif hasattr(msg, "reference"):
                point = msg.reference

            if point is None:
                return

            positions = list(getattr(point, "positions", []))
            if not names or len(positions) < len(names):
                return

            name_to_pos = {
                str(n): float(p)
                for n, p in zip(names, positions)
            }
            _apply_arm_command(
                name_to_pos,
                "/arm_controller/controller_state",
            )

        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import (
                QoSProfile,
                ReliabilityPolicy,
                DurabilityPolicy,
            )
            from sensor_msgs.msg import JointState
            from control_msgs.msg import JointTrajectoryControllerState
            from moveit_msgs.msg import PlanningScene
            from moveit_msgs.msg import CollisionObject
            from shape_msgs.msg import SolidPrimitive
            from geometry_msgs.msg import Pose
            from geometry_msgs.msg import TransformStamped
            from tf2_ros import TransformBroadcaster

            if not rclpy.ok():
                rclpy.init(args=None)

            ros_node = Node("nero_mujoco_v62_backend")
            object_tf_broadcaster = TransformBroadcaster(
                ros_node
            )

            # Egg-task semantic TF registry.  Canonical task names are
            # egg_1..egg_4 while the MuJoCo bodies remain obj_egg1..obj_egg4.
            # Publishing sim_obj_egg_1 etc. lets the unchanged V69 object_pose()
            # helper work with the new semantic object names.
            object_tf_body_map = {
                "obj_egg_1": "obj_egg1",
                "obj_egg_2": "obj_egg2",
                "obj_egg_3": "obj_egg3",
                "obj_egg_4": "obj_egg4",
            }
            # V3.43 A2 BEGIN
            if _v343_scene_name() != "fill_egg_holder":
                object_tf_body_map = _V343.tf_body_map()
            # V3.43 A2 END
            object_tf_bids = {
                semantic_body: mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, mujoco_body
                )
                for semantic_body, mujoco_body in object_tf_body_map.items()
            }

            ros_feedback_pub = ros_node.create_publisher(
                JointState,
                "/feedback/joint_states",
                20,
            )

            ros_node.create_subscription(
                JointTrajectoryControllerState,
                "/arm_controller/controller_state",
                _controller_state_callback,
                50,
            )

            ros_node.create_subscription(
                JointState,
                "/control/joint_states",
                _moveit_joint_state_callback,
                20,
            )

            scene_qos = QoSProfile(depth=1)
            scene_qos.reliability = ReliabilityPolicy.RELIABLE
            scene_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

            ros_scene_pub = ros_node.create_publisher(
                PlanningScene,
                "/planning_scene",
                scene_qos,
            )

            ros_joint_state_type = JointState
            ros_planning_scene_type = PlanningScene

            def _make_box_collision(
                object_id,
                center_xyz,
                size_xyz,
            ):
                obj = CollisionObject()
                obj.header.frame_id = "base_link"
                obj.id = str(object_id)

                primitive = SolidPrimitive()
                primitive.type = SolidPrimitive.BOX
                primitive.dimensions = [
                    float(size_xyz[0]),
                    float(size_xyz[1]),
                    float(size_xyz[2]),
                ]

                pose = Pose()
                pose.position.x = float(center_xyz[0])
                pose.position.y = float(center_xyz[1])
                pose.position.z = float(center_xyz[2])
                pose.orientation.w = 1.0

                obj.primitives = [primitive]
                obj.primitive_poses = [pose]
                obj.operation = CollisionObject.ADD
                return obj

            def _publish_mujoco_scene():
                if ros_scene_pub is None:
                    return

                scene = PlanningScene()
                scene.is_diff = True

                # V69 contact-aware planning geometry.
                # IMPORTANT: tabletop grasp/approach is an intended near-contact
                # operation, so the table must use its REAL geometry.  The V58
                # global +30 mm table inflation intruded into the grasp workspace
                # and made the final ~5% of otherwise-valid approaches look like
                # collisions.  Keep runtime safety monitoring for model mismatch.
                table_obj = _make_box_collision(
                    "mujoco_play_table",
                    (-0.385, 0.0, -0.03),
                    (0.67, 0.76, 0.06),   # V69 exact table; near edge x=-0.05, top=0.000 m
                )

                static_objects = [table_obj]

                # V2-7 Egg Scene: approximate only the low static structures
                # needed by MoveIt.  The basket's thin wicker wall is omitted from
                # the planning scene so the gripper can reach into it; MuJoCo
                # contact + the existing runtime safety layer still enforce the
                # actual reed geometry.  The egg-holder base/rim/lid are included.
                egg_scene_boxes = [
                    
                    ("egg_holder_base",      (-0.420,  0.155, 0.006), (0.250, 0.144, 0.012)),
                    ("egg_holder_rim_xneg", (-0.548,  0.155, 0.025), (0.008, 0.152, 0.050)),
                    ("egg_holder_rim_xpos", (-0.292,  0.155, 0.025), (0.008, 0.152, 0.050)),
                    ("egg_holder_rim_yneg", (-0.420,  0.079, 0.025), (0.248, 0.008, 0.050)),
                    ("egg_holder_rim_ypos", (-0.420,  0.231, 0.025), (0.248, 0.008, 0.050)),
                ]
                # V3.43 A2 BEGIN
                if _v343_scene_name() != "fill_egg_holder":
                    egg_scene_boxes = _V343.planning_boxes(model, data)
                # V3.43 A2 END

                # V3.29 PERCEIVED HOLDER IN MOVEIT: in wrist no-prior mode the backend writes the holder boxes it
                # perceived (or none yet); the registered boxes are then not published.
                ovr329 = _v329_read_scene_override()
                skip329 = set(ovr329.get("ids") or []) if ovr329 else set()
                for object_id, center, size in egg_scene_boxes:
                    if object_id in skip329:
                        continue
                    static_objects.append(
                        _make_box_collision(object_id, center, size)
                    )
                if ovr329:
                    added329 = set()
                    for object_id, center, size, yaw in (ovr329.get("boxes") or []):
                        o329 = _make_box_collision(object_id, center, size)
                        o329.primitive_poses[0].orientation.z = math.sin(float(yaw) / 2.0)
                        o329.primitive_poses[0].orientation.w = math.cos(float(yaw) / 2.0)
                        static_objects.append(o329)
                        added329.add(object_id)
                    for object_id in skip329 - added329:
                        r329 = CollisionObject()
                        r329.header.frame_id = "base_link"
                        r329.id = str(object_id)
                        r329.operation = CollisionObject.REMOVE
                        static_objects.append(r329)

                # V3.43 A2 BEGIN
                if _v343_scene_name() != "fill_egg_holder":
                    for _id343, _c343, _s343, _yaw343 in _V343.planning_rotated_boxes(model, data):
                        _o343 = _make_box_collision(_id343, _c343, _s343)
                        _o343.primitive_poses[0].orientation.z = math.sin(float(_yaw343) / 2.0)
                        _o343.primitive_poses[0].orientation.w = math.cos(float(_yaw343) / 2.0)
                        static_objects.append(_o343)
                # V3.43 A2 END
                scene.world.collision_objects = static_objects

                ros_scene_pub.publish(scene)
                ros_scene_count.value += 1

            ros_ready = True
            ros_bridge_ready.value = True

            print(
                "V69 ROS2 bridge READY: "
                "/arm_controller/controller_state -> V31 (primary), "
                "/control/joint_states -> V31 (fallback), "
                "V31 -> /feedback/joint_states",
                flush=True,
            )

        except Exception as ros_exc:
            ros_bridge_ready.value = False
            ros_ready = False
            print(
                "V53 ROS2 bridge unavailable. "
                "Start this script after sourcing ROS2 + agx_arm_ws. "
                f"Reason: {ros_exc}",
                flush=True,
            )

        gripper_actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper"
        )
        if gripper_actuator_id < 0:
            raise RuntimeError("Official gripper position actuator not found")

        gripper_force_actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_force"
        )
        if gripper_force_actuator_id < 0:
            raise RuntimeError("V16 gripper force actuator not found")

        print("\n========== V10 VERIFIED ARM MAPPING ==========", flush=True)
        print("Slider | Joint  | qpos_addr | dof_addr", flush=True)
        for i, name in enumerate(joint_names):
            print(
                f"{i+1:>6} | {name:<6} | "
                f"{qpos_addr[i]:>9} | {dof_addr[i]:>8}",
                flush=True,
            )
        print("==============================================\n", flush=True)

        # --------------------------------------------------------------
        # V14 SAFETY GUARD SETUP
        # --------------------------------------------------------------
        protected_geom_ids = set()
        # V2-7 Egg Scene: protect the table plus basket/carton geometry from
        # robot penetration.  Eggs themselves remain manipulable and therefore
        # are intentionally not protected surfaces.
        for gid in range(model.ngeom):
            geom_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, gid
            ) or ""
            if (
                geom_name in ("play_table", "v5_grid_floor")
                or geom_name.startswith("egg_basket_")
                or geom_name.startswith("egg_holder_")
            ):
                protected_geom_ids.add(int(gid))
        # V3.43 A2 BEGIN
        if _v343_scene_name() != "fill_egg_holder":
            for gid in range(model.ngeom):
                geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
                if geom_name.startswith(_V343.PROTECTED_PREFIXES):
                    protected_geom_ids.add(int(gid))
        # V3.43 A2 END

        robot_body_ids = set()
        robot_body_names = (
            ["base_link"]
            + [f"link{i}" for i in range(1, 8)]
            + ["gripper_link1", "gripper_link2"]
        )
        for body_name in robot_body_names:
            bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if bid >= 0:
                robot_body_ids.add(int(bid))

        def is_robot_geom(geom_id):
            if geom_id < 0 or geom_id >= model.ngeom:
                return False
            return int(model.geom_bodyid[geom_id]) in robot_body_ids

        def protected_penetration_depth():
            """
            Return the most negative robot-vs-protected-surface contact distance.
            0.0 means no penetration of a protected surface.
            """
            worst = 0.0
            for ci in range(int(data.ncon)):
                c = data.contact[ci]
                g1 = int(c.geom1)
                g2 = int(c.geom2)
                robot_vs_surface = (
                    (g1 in protected_geom_ids and is_robot_geom(g2))
                    or
                    (g2 in protected_geom_ids and is_robot_geom(g1))
                )
                if robot_vs_surface:
                    worst = min(worst, float(c.dist))
            return worst

        object_geom_ids = set()
        for gi in range(model.ngeom):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi)
            if gname and gname.startswith("obj_"):
                object_geom_ids.add(int(gi))

        gripper_body_ids = set()
        for bname in ("gripper_link1", "gripper_link2"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid >= 0:
                gripper_body_ids.add(int(bid))

        def is_gripper_geom(geom_id):
            if geom_id < 0 or geom_id >= model.ngeom:
                return False
            return int(model.geom_bodyid[geom_id]) in gripper_body_ids

        def gripper_object_contact_info():
            has_contact = False
            worst = 0.0
            for ci in range(int(data.ncon)):
                c = data.contact[ci]
                g1 = int(c.geom1)
                g2 = int(c.geom2)
                pair = (
                    (is_gripper_geom(g1) and g2 in object_geom_ids)
                    or (is_gripper_geom(g2) and g1 in object_geom_ids)
                )
                if pair:
                    if float(c.dist) <= 0.0005:
                        has_contact = True
                    worst = min(worst, float(c.dist))
            return has_contact, worst

        def gripper_object_penetration_depth():
            return gripper_object_contact_info()[1]

        # --------------------------------------------------------------
        # V22 CONTACT-FORCE CLOSED-LOOP SETUP
        #
        # MuJoCo does NOT provide a separate static-friction coefficient.
        # Keep the V19 pad friction fixed at mu=10 and close the loop on
        # the REAL normal contact force measured by mj_contactForce().
        # --------------------------------------------------------------
        link7_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "link7"
        )
        if link7_bid < 0:
            raise RuntimeError("link7 body not found")

        finger1_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "gripper_link1"
        )
        finger2_bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "gripper_link2"
        )
        if finger1_bid < 0 or finger2_bid < 0:
            raise RuntimeError("gripper finger bodies not found")

        pad_geom_ids = set()
        for gi in range(model.ngeom):
            gname = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, gi
            )
            if gname and (
                gname.startswith("finger1_rubber_pad")
                or gname.startswith("finger2_rubber_pad")
                or gname.startswith("finger1_stud_")
                or gname.startswith("finger2_stud_")
            ):
                pad_geom_ids.add(int(gi))

        def contacted_object_body():
            """Return object body with the largest number of pad contacts."""
            counts = {}
            for ci in range(int(data.ncon)):
                c = data.contact[ci]
                g1 = int(c.geom1)
                g2 = int(c.geom2)

                obj_gid = -1
                if g1 in pad_geom_ids and g2 in object_geom_ids:
                    obj_gid = g2
                elif g2 in pad_geom_ids and g1 in object_geom_ids:
                    obj_gid = g1

                if obj_gid >= 0:
                    bid = int(model.geom_bodyid[obj_gid])
                    counts[bid] = counts.get(bid, 0) + 1

            if not counts:
                return -1
            return max(counts, key=counts.get)

        # --------------------------------------------------------------
        # V26 RUNTIME CONNECT HOLD SETUP
        # --------------------------------------------------------------
        hold_object_names = [
                "obj_egg1", "obj_egg2", "obj_egg3", "obj_egg4",
                "obj_red_cube", "obj_blue_cube", "obj_red_ball", "obj_blue_ball",
            ]
        # V3.43 A2 BEGIN
        if _v343_scene_name() != "fill_egg_holder":
            hold_object_names = _V343.hold_object_names()
        # V3.43 A2 END

        hold_eq_by_body = {}
        hold_name_by_body = {}

        for obj_name in hold_object_names:
            bid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, obj_name
            )
            eqid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY,
                f"hold_{obj_name}"
            )

            if bid >= 0 and eqid >= 0:
                hold_eq_by_body[int(bid)] = int(eqid)
                hold_name_by_body[int(bid)] = obj_name
                data.eq_active[eqid] = 0

        def deactivate_all_holds():
            # V2-7.2 HOLD-STATE CONTINUITY.  Snapshot the CURRENT displayed
            # world pose before eq_active goes to zero, then seed that pose back
            # into the free joint.  This blocks old-pose snapback/teleport.
            _snap = []
            for _bid, _eqid in hold_eq_by_body.items():
                try:
                    if int(data.eq_active[_eqid]) == 0:
                        continue
                    _jid = int(model.body_jntadr[int(_bid)])
                    if _jid < 0 or int(model.jnt_type[_jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
                        continue
                    _qadr = int(model.jnt_qposadr[_jid])
                    _dadr = int(model.jnt_dofadr[_jid])
                    _pos = np.array(data.xpos[int(_bid)], dtype=float).copy()
                    _quat = np.array(data.xquat[int(_bid)], dtype=float).copy()
                    _snap.append((int(_bid), _qadr, _dadr, _pos, _quat))
                except Exception:
                    pass
            for eqid in hold_eq_by_body.values():
                data.eq_active[eqid] = 0
            if _snap:
                for _bid, _qadr, _dadr, _pos, _quat in _snap:
                    data.qpos[_qadr:_qadr+3] = _pos
                    data.qpos[_qadr+3:_qadr+7] = _quat
                    data.qvel[_dadr:_dadr+6] = 0.0
                mujoco.mj_forward(model, data)
                _names = [hold_name_by_body.get(int(x[0]), f"body#{int(x[0])}") for x in _snap]
                print("[V2-7.2 HOLD CONTINUITY] released at CURRENT pose: " + str(_names), flush=True)

        def activate_connect_hold(obj_bid):
            """
            Activate a CONNECT equality at the object's CURRENT pose.

            For body-semantic connect equalities:
              eq_data[0:3] = anchor in body1 (link7) local frame
              eq_data[3:6] = anchor in body2 (object) local frame

            We choose body2 anchor = object origin/COM and compute the
            matching point in link7 local coordinates. Therefore position
            error is ~zero at activation: no snap-to-old-pose impulse.
            """
            eqid = hold_eq_by_body.get(int(obj_bid), -1)
            if eqid < 0:
                return False

            # Always ensure only one object is held.
            deactivate_all_holds()

            p_link = np.array(data.xpos[link7_bid], dtype=float)
            p_obj = np.array(data.xpos[obj_bid], dtype=float)

            # MuJoCo xmat is local->world rotation, row-major 3x3.
            R_link = np.array(
                data.xmat[link7_bid], dtype=float
            ).reshape(3, 3)

            anchor_link_local = R_link.T @ (p_obj - p_link)

            # CONNECT eq_data uses 6 anchor values.
            model.eq_data[eqid, 0:3] = anchor_link_local
            model.eq_data[eqid, 3:6] = 0.0

            # Activate. Constraint will be rebuilt by the next step1.
            data.eq_active[eqid] = 1
            return True

        def gripper_contact_sides():
            """
            Contact-presence and worst penetration based ONLY on the
            visible rubber pad/stud geoms, not arbitrary gripper meshes.
            """
            left_touch = False
            right_touch = False
            worst = 0.0

            for ci in range(int(data.ncon)):
                c = data.contact[ci]
                g1 = int(c.geom1)
                g2 = int(c.geom2)

                pad_gid = -1
                if g1 in pad_geom_ids and g2 in object_geom_ids:
                    pad_gid = g1
                elif g2 in pad_geom_ids and g1 in object_geom_ids:
                    pad_gid = g2
                else:
                    continue

                bid = int(model.geom_bodyid[pad_gid])
                if float(c.dist) <= 0.0005:
                    if bid == finger1_bid:
                        left_touch = True
                    elif bid == finger2_bid:
                        right_touch = True

                worst = min(worst, float(c.dist))

            return left_touch, right_touch, worst

        def measure_pad_contact_forces():
            """
            Sum contact forces on each visible fingertip pad.

            mj_contactForce() returns a 6-vector in the contact frame:
              [normal, tangent1, tangent2, torque1, torque2, torque3]

            Returns:
              left_normal_N
              right_normal_N
              total_tangential_N
            """
            left_n = 0.0
            right_n = 0.0
            tangential = 0.0
            cf = np.zeros(6, dtype=float)

            for ci in range(int(data.ncon)):
                c = data.contact[ci]
                g1 = int(c.geom1)
                g2 = int(c.geom2)

                pad_gid = -1
                if g1 in pad_geom_ids and g2 in object_geom_ids:
                    pad_gid = g1
                elif g2 in pad_geom_ids and g1 in object_geom_ids:
                    pad_gid = g2
                else:
                    continue

                cf[:] = 0.0
                mujoco.mj_contactForce(model, data, ci, cf)

                fn = abs(float(cf[0]))
                ft = float(np.hypot(cf[1], cf[2]))
                bid = int(model.geom_bodyid[pad_gid])

                if bid == finger1_bid:
                    left_n += fn
                elif bid == finger2_bid:
                    right_n += fn

                tangential += ft

            return left_n, right_n, tangential

        try:
            model.opt.iterations = max(int(model.opt.iterations), 80)
        except Exception:
            pass
        try:
            model.opt.ls_iterations = max(int(model.opt.ls_iterations), 20)
        except Exception:
            pass

        # V38 default gravity ON.
        model.opt.gravity[:] = (0.0, 0.0, -9.81)

        mujoco.mj_forward(model, data)

        q0 = np.array(
            [data.qpos[a] for a in qpos_addr],
            dtype=float,
        )

        q_ref = q0.copy()
        q_ref_prev = q0.copy()

        # Last known safe ARM pose. Updated only when the robot is not
        # penetrating the table/riser/floor and is not moving excessively.
        last_safe_q = q0.copy()
        last_safe_gripper_q = 0.0
        safety_cooldown_until = 0.0

        # V31 adjustable two-stage clamp:
        #
        # BASE_APPROACH_QVEL is the original stable V19/V30 speed (1x).
        # Default GUI command is 10x during FREE SPACE.
        #
        # As soon as EITHER fingertip touches an object, the speed is
        # automatically reduced to CONTACT_BRAKE_MULT (1.5x) so the
        # last closing motion remains gentle and does not launch objects.
        #
        # Once BOTH fingertips contact, the original V22 force closed
        # loop takes over exactly as before.
        BASE_APPROACH_QVEL = 0.0030   # m/s master coordinate = original 1x
        CONTACT_BRAKE_MULT = 1.5

        APPROACH_VEL_K = 25.0        # N/(m/s)
        # V39: more drive in free-space so the requested 15x approach
        # speed is actually reached.  First contact still automatically
        # brakes to CONTACT_BRAKE_MULT before force control takes over.
        APPROACH_FORCE_LIMIT = 0.60  # N
        CONTACT_FORCE_RAMP_SEC = 2.0

        clamp_mode_prev = False
        contact_latched = False
        contact_started_at = 0.0
        clamp_applied_force = 0.0

        # --------------------------------------------------------------
        # V22 contact-force loop state.
        #
        # GUI clamp-force value is now interpreted as:
        #     TARGET NORMAL FORCE PER FINGER [N]
        #
        # The actuator generalized force is allowed to differ from this
        # number.  The PI loop adjusts motor force until BOTH fingertips
        # actually measure the requested normal force.
        # --------------------------------------------------------------
        MAX_EXTRA_TARGET_N = 2.0

        # Closed-loop tuning deliberately conservative.
        FORCE_KP = 0.70
        FORCE_KI = 0.80
        MOTOR_FEEDFORWARD_GAIN = 1.20
        MOTOR_MAX_N = 30.0
        MOTOR_RISE_N_PER_S = 4.0
        MOTOR_FALL_N_PER_S = 10.0
        FORCE_INTEGRAL_LIMIT = 12.0

        anti_slip_boost = 0.0
        anti_slip_obj_bid = -1
        anti_slip_prev_rel_z = None
        anti_slip_vz_lp = 0.0
        anti_slip_persist = 0.0

        measured_left_n = 0.0
        measured_right_n = 0.0
        measured_tangent_n = 0.0

        force_integral = 0.0
        motor_clamp_cmd = 0.0

        # V30 AUTO CREEP HOLD state.
        #
        # Automatic HOLD:
        #   clamp mode
        #   + gripper/object contact exists
        #   + downward relative slip <= -0.05 mm/s
        #   + condition persists for >= 0.10 s
        #
        # HOLD OFF:
        #   any explicit release/opening action
        #
        # Manual HOLD ON / HOLD OFF buttons remain available for debug.
        AUTO_HOLD_SLIP_MM_S = -0.05
        AUTO_HOLD_PERSIST_SEC = 0.10

        hold_ready_time = 0.0
        hold_obj_bid = -1
        last_contact_obj_bid = -1
        hold_active = False
        auto_hold_timer = 0.0

        # status:
        # 0 = ready / monitoring
        # 1 = holding
        # 2 = manual HOLD ON requested but no object contact/candidate found
        hold_active_shared.value = False
        hold_name_shared.value = -1
        hold_ready_shared.value = 0.0
        hold_now_request.value = False
        hold_off_request.value = False
        hold_status_code_shared.value = 0

        anti_slip_boost_shared.value = 0.0
        anti_slip_slip_mm_s_shared.value = 0.0
        anti_slip_mu_shared.value = 10.0
        contact_left_n_shared.value = 0.0
        contact_right_n_shared.value = 0.0
        contact_tangent_n_shared.value = 0.0
        contact_motor_cmd_shared.value = 0.0

        gripper_q0 = float(data.qpos[gripper_qpos_addr])
        last_safe_gripper_q = gripper_q0
        gripper_target_mm.value = max(0.0, min(100.0, 2.0 * gripper_q0 * 1000.0))
        gripper_actual_mm.value = gripper_target_mm.value

        _v337_keep = {"t": -1.0e9}

        def _v337_held_body():
            for _bid, _eqid in hold_eq_by_body.items():
                try:
                    if int(data.eq_active[_eqid]) == 1:
                        return int(_bid)
                except Exception:
                    pass
            return -1

        def hard_restore_arm(q_restore, gripper_q=None, reason_code=1, count_trip=True, keep_hold=False):
            """
            Simulation-only safety recovery:
              - teleport arm back to a known safe joint pose
              - zero arm/gripper velocities and accelerations
              - synchronize GUI targets and internal references
              - clear applied forces before continuing

            V3.37 keep_hold (automatic RUNAWAY restore only): an egg held by the CONNECT hold is carried along with
            the gripper (same pose relative to link7), the gripper stays in force mode and the hold stays active, so
            the safety recovery itself no longer drops the egg.  A second keep-hold restore within
            NERO_V337_KEEP_HOLD_WINDOW_S falls back to the old behaviour (release).
            """
            nonlocal q_ref, q_ref_prev, next_step_wall, safety_cooldown_until

            q_restore = np.asarray(q_restore, dtype=float).copy()
            keep337 = False
            bid337 = -1
            if keep_hold and os.environ.get("NERO_V337_RESTORE_KEEP_HOLD", "1") != "0":
                bid337 = _v337_held_body()
                win337 = float(os.environ.get("NERO_V337_KEEP_HOLD_WINDOW_S", "3.0"))
                if bid337 >= 0 and time.perf_counter() - _v337_keep["t"] > win337:
                    keep337 = True
            if keep337:
                _v337_keep["t"] = time.perf_counter()
                R_l337 = np.array(data.xmat[link7_bid], dtype=float).reshape(3, 3)
                p_l337 = np.array(data.xpos[link7_bid], dtype=float)
                R_o337 = np.array(data.xmat[bid337], dtype=float).reshape(3, 3)
                rel_p337 = R_l337.T @ (np.array(data.xpos[bid337], dtype=float) - p_l337)
                rel_R337 = R_l337.T @ R_o337
                if not (np.all(np.isfinite(rel_p337)) and np.all(np.isfinite(rel_R337))
                        and np.isfinite(float(data.qpos[gripper_qpos_addr]))):
                    keep337 = False                                       # NaN state: old hard restore (release)
                else:
                    gripper_q = float(data.qpos[gripper_qpos_addr])      # fingers stay closed on the egg

            for i, qadr in enumerate(qpos_addr):
                lo, hi = JOINT_LIMITS_RAD[i]
                q_restore[i] = float(np.clip(q_restore[i], lo, hi))
                data.qpos[qadr] = q_restore[i]

            for dadr in dof_addr:
                data.qvel[dadr] = 0.0
                data.qacc[dadr] = 0.0

            if gripper_q is None:
                gripper_q = float(data.qpos[gripper_qpos_addr])

            gripper_q = float(np.clip(gripper_q, 0.0, 0.05))
            data.qpos[gripper_qpos_addr] = gripper_q
            data.qpos[gripper2_qpos_addr] = -gripper_q
            data.qvel[gripper_dof_addr] = 0.0
            data.qvel[gripper2_dof_addr] = 0.0
            data.qacc[gripper_dof_addr] = 0.0
            data.qacc[gripper2_dof_addr] = 0.0

            data.qfrc_applied[:] = 0.0
            data.xfrc_applied[:] = 0.0
            if model.nu > 0:
                data.ctrl[:] = 0.0

            q_ref[:] = q_restore
            q_ref_prev[:] = q_restore

            for i in range(7):
                target_shared[i] = float(q_restore[i])
                actual_shared[i] = float(q_restore[i])

            if keep337:
                mujoco.mj_forward(model, data)
                R_l337b = np.array(data.xmat[link7_bid], dtype=float).reshape(3, 3)
                p_obj337 = np.array(data.xpos[link7_bid], dtype=float) + R_l337b @ rel_p337
                q_obj337 = np.zeros(4)
                mujoco.mju_mat2Quat(q_obj337, (R_l337b @ rel_R337).reshape(-1))
                _jid337 = int(model.body_jntadr[bid337])
                if _jid337 >= 0 and int(model.jnt_type[_jid337]) == int(mujoco.mjtJoint.mjJNT_FREE):
                    _qa337 = int(model.jnt_qposadr[_jid337])
                    _da337 = int(model.jnt_dofadr[_jid337])
                    data.qpos[_qa337:_qa337 + 3] = p_obj337
                    data.qpos[_qa337 + 3:_qa337 + 7] = q_obj337
                    data.qvel[_da337:_da337 + 6] = 0.0
                mujoco.mj_forward(model, data)
                safety_state.value = int(reason_code)
                if bool(count_trip):
                    safety_trip_count.value += 1
                safety_cooldown_until = time.perf_counter() + (1.0 if not count_trip else 0.25)
                next_step_wall = time.perf_counter()
                print(f"[V3.37 RESTORE KEEP HOLD] arm restored to the last safe pose; held object "
                      f"{hold_name_by_body.get(bid337, bid337)} carried with the gripper, hold stays active", flush=True)
                return

            grip_mm = 2.0 * gripper_q * 1000.0
            gripper_target_mm.value = grip_mm
            gripper_actual_mm.value = grip_mm

            # Safety recovery always returns gripper to position mode.
            gripper_force_mode.value = False

            anti_slip_boost_shared.value = 0.0
            anti_slip_slip_mm_s_shared.value = 0.0
            anti_slip_mu_shared.value = 10.0
            contact_left_n_shared.value = 0.0
            contact_right_n_shared.value = 0.0
            contact_tangent_n_shared.value = 0.0
            contact_motor_cmd_shared.value = 0.0

            deactivate_all_holds()
            hold_ready_time = 0.0
            hold_obj_bid = -1
            hold_active = False
            hold_active_shared.value = False
            hold_name_shared.value = -1
            hold_ready_shared.value = 0.0
            hold_now_request.value = False
            hold_off_request.value = False
            hold_status_code_shared.value = 0
            auto_hold_timer = 0.0

            mujoco.mj_forward(model, data)

            safety_state.value = int(reason_code)
            if bool(count_trip):
                safety_trip_count.value += 1
            safety_cooldown_until = time.perf_counter() + (1.0 if not count_trip else 0.25)
            next_step_wall = time.perf_counter()

        def soft_freeze_arm(reason_code=2, count_trip=False):
            """Stop the arm at its *current* physical pose without teleporting.

            Critical V58 behavior: preserve gripper force/position mode, gripper
            opening, held-object equality state, and object pose.  This replaces
            the old automatic rollback that could jerk toward an old checkpoint
            and fling a grasped object.
            """
            nonlocal q_ref, q_ref_prev, next_step_wall, safety_cooldown_until

            q_now = np.array(
                [float(data.qpos[a]) for a in qpos_addr], dtype=float
            )
            for dadr in dof_addr:
                data.qvel[dadr] = 0.0
                data.qacc[dadr] = 0.0
                data.qfrc_applied[dadr] = 0.0

            q_ref[:] = q_now
            q_ref_prev[:] = q_now
            for i in range(7):
                target_shared[i] = float(q_now[i])
                actual_shared[i] = float(q_now[i])

            # Do NOT touch gripper_qpos, gripper target, force mode, anti-slip,
            # or object HOLD state. The object must remain held during replanning.
            mujoco.mj_forward(model, data)
            safety_state.value = int(reason_code)
            if bool(count_trip):
                safety_trip_count.value += 1
            safety_cooldown_until = time.perf_counter() + (0.30 if count_trip else 0.10)
            next_step_wall = time.perf_counter()

        for i in range(7):
            target_shared[i] = float(q0[i])
            actual_shared[i] = float(q0[i])

        sim_ready.value = True
        sim_error_code.value = 0

        # V3.08: hand the model file + live qpos to the camera process.
        _v308_last_qpos_write = 0.0
        _v308_nq = int(model.nq)
        if v308_model_path is not None and v308_qpos is not None and _v308_nq <= len(v308_qpos):
            _v308_bytes = str(visual_model_file).encode("utf-8")[: len(v308_model_path) - 1]
            v308_model_path[: len(_v308_bytes)] = _v308_bytes
            v308_model_path[len(_v308_bytes)] = b"\0"
            with v308_qpos.get_lock():
                v308_qpos[:_v308_nq] = [float(v) for v in data.qpos]
                v308_meta[0] = float(_v308_nq)
                v308_meta[1] = float(data.time)
                v308_meta[3] = 1.0
        elif v308_qpos is not None:
            print(f"[V3.08 CAMERA WARN] nq={_v308_nq} exceeds shared buffer; perception disabled", flush=True)
            v308_qpos = None

        viewer_period = 1.0 / 60.0
        last_viewer_sync = 0.0
        next_step_wall = time.perf_counter()

        # ------------------------------------------------------------
        # V3.01 MOTION DIAGNOSTICS (record only; never changes control)
        # While the ROS arm gate is ON (a trajectory is executing):
        #   * every physics step goes into a ring buffer (last ~1.5 s);
        #   * a 20 Hz sample is appended to diag_motion.csv.
        # On any automatic safety trip the ring buffer is dumped to
        # diag_trip_<n>_<reason>.csv, so the build-up to a RUNAWAY is visible:
        # commanded vs actual joint acceleration, joint velocity, applied torque.
        # ------------------------------------------------------------
        import collections as _v301_collections
        _v301_dir = os.environ.get("NERO_DIAG_DIR", "").strip()
        _v301_ring = _v301_collections.deque(maxlen=1500)
        _v301_state = {"csv": None, "last": 0.0, "trips": 0, "t0": time.perf_counter(), "failed": False}
        _v301_hdr = (
            ["t_s", "gate", "hold"]
            + [f"q{i+1}" for i in range(7)] + [f"qd{i+1}" for i in range(7)]
            + [f"qacc_cmd{i+1}" for i in range(7)] + [f"qacc_act{i+1}" for i in range(7)]
            + [f"tau{i+1}" for i in range(7)]
        )

        def _v301_row_text(row):
            return ",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in row)

        def _v301_record(row):
            if not _v301_dir or _v301_state["failed"]:
                return
            try:
                _v301_ring.append(row)
                now301 = time.perf_counter()
                if now301 - _v301_state["last"] >= 0.05:
                    _v301_state["last"] = now301
                    if _v301_state["csv"] is None:
                        os.makedirs(_v301_dir, exist_ok=True)
                        _v301_state["csv"] = open(os.path.join(_v301_dir, "diag_motion.csv"), "w", buffering=1)
                        _v301_state["csv"].write(",".join(_v301_hdr) + "\n")
                    _v301_state["csv"].write(_v301_row_text(row) + "\n")
            except Exception as exc301:
                _v301_state["failed"] = True
                print(f"[V3.01 DIAG] disabled after error: {exc301}", flush=True)

        def _v301_dump(reason):
            if not _v301_dir or not _v301_ring:
                return
            try:
                _v301_state["trips"] += 1
                os.makedirs(_v301_dir, exist_ok=True)
                fn301 = os.path.join(_v301_dir, f"diag_trip_{_v301_state['trips']}_{reason}.csv")
                with open(fn301, "w") as f301:
                    f301.write(",".join(_v301_hdr) + "\n")
                    for r301 in _v301_ring:
                        f301.write(_v301_row_text(r301) + "\n")
                print(f"[V3.01 DIAG] safety trip '{reason}': last {len(_v301_ring)} steps -> {fn301}", flush=True)
                _v301_ring.clear()
            except Exception as exc301:
                print(f"[V3.01 DIAG] dump failed: {exc301}", flush=True)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and not stop_event.is_set():

                if bool(gravity_enabled.value):
                    model.opt.gravity[:] = (0.0, 0.0, -9.81)
                else:
                    model.opt.gravity[:] = (0.0, 0.0, 0.0)

                # First half of MuJoCo step.
                mujoco.mj_step1(model, data)

                # ------------------------------------------------------
                # MANUAL SAFETY REQUESTS
                # ------------------------------------------------------
                if bool(reset_zero_request.value):
                    reset_zero_request.value = False
                    ros_motion_fault_latched = False

                    zero_pose = np.zeros(7, dtype=float)
                    # Open gripper to 80 mm during hard reset so a trapped
                    # object is released rather than pinched during teleport.
                    hard_restore_arm(
                        zero_pose,
                        gripper_q=0.040,
                        reason_code=3,
                        count_trip=False,
                    )
                    last_safe_q[:] = zero_pose
                    last_safe_gripper_q = 0.040
                    continue

                if bool(recover_request.value):
                    # EXPLICIT user/planner recovery only. Automatic safety
                    # handlers never set this flag in V69.
                    recover_request.value = False
                    ros_motion_fault_latched = False
                    soft_freeze_arm(
                        reason_code=2,
                        count_trip=False,
                    )
                    continue

                # ------------------------------------------------------
                # AUTOMATIC ANTI-PENETRATION GUARD
                #
                # Soft MuJoCo contacts normally allow tiny negative
                # distances. We only trip on > 2.0 mm penetration.
                # ------------------------------------------------------
                penetration = protected_penetration_depth()
                if (
                    penetration < -0.002
                    and time.perf_counter() >= safety_cooldown_until
                ):
                    print(
                        f"SAFETY GUARD: protected-surface penetration "
                        f"{-penetration*1000.0:.2f} mm -> FREEZE-IN-PLACE + "
                        "MOVEIT COMMAND LATCH",
                        flush=True,
                    )
                    ros_motion_fault_latched = True
                    _v301_dump("penetration")
                    soft_freeze_arm(
                        reason_code=1,
                        count_trip=True,
                    )
                    continue

                q = np.array(
                    [data.qpos[a] for a in qpos_addr],
                    dtype=float,
                )
                qd = np.array(
                    [data.qvel[a] for a in dof_addr],
                    dtype=float,
                )

                gripper_q = float(data.qpos[gripper_qpos_addr])
                gripper_qd = float(data.qvel[gripper_dof_addr])

                # ------------------------------------------------------
                # V39 ROS2 bridge heartbeat.
                # ------------------------------------------------------
                if ros_ready and ros_node is not None:
                    now_ros = time.perf_counter()

                    # Process MoveIt commands with low latency.
                    if now_ros - ros_last_spin_wall >= 0.002:
                        try:
                            rclpy.spin_once(
                                ros_node,
                                timeout_sec=0.0,
                            )
                        except Exception as exc:
                            print(
                                f"ROS spin error: {exc}",
                                flush=True,
                            )
                        ros_last_spin_wall = now_ros

                    # Publish actual MuJoCo state at 100 Hz.
                    if now_ros - ros_last_feedback_wall >= 0.010:
                        try:
                            msg = ros_joint_state_type()
                            msg.header.stamp = (
                                ros_node.get_clock().now().to_msg()
                            )
                            # Match the official AgileX MoveIt model:
                            # arm joints joint1..joint7 + active "gripper".
                            # "gripper" is FULL opening width in meters.
                            msg.name = joint_names + ["gripper"]
                            msg.position = (
                                [float(v) for v in q]
                                + [
                                    float(
                                        np.clip(
                                            2.0 * gripper_q,
                                            0.0,
                                            0.10,
                                        )
                                    )
                                ]
                            )
                            msg.velocity = (
                                [float(v) for v in qd]
                                + [
                                    float(2.0 * gripper_qd)
                                ]
                            )

                            ros_feedback_pub.publish(msg)
                            ros_feedback_count.value += 1
                        except Exception as exc:
                            print(
                                f"ROS feedback publish error: {exc}",
                                flush=True,
                            )

                        ros_last_feedback_wall = now_ros

                    # Publish MuJoCo object poses as TF frames at 20 Hz.
                    if (
                        not hasattr(run_sim, "_v39_last_tf_wall")
                        or now_ros
                        - getattr(run_sim, "_v39_last_tf_wall")
                        >= 0.050
                    ):
                        try:
                            tf_msgs = []

                            for (
                                obj_name,
                                obj_bid,
                            ) in object_tf_bids.items():
                                if obj_bid < 0:
                                    continue

                                tfm = TransformStamped()
                                tfm.header.stamp = (
                                    ros_node.get_clock()
                                    .now()
                                    .to_msg()
                                )
                                tfm.header.frame_id = "base_link"
                                tfm.child_frame_id = (
                                    "sim_" + obj_name
                                )

                                p = data.xpos[obj_bid]
                                qwxyz = data.xquat[obj_bid]

                                tfm.transform.translation.x = float(
                                    p[0]
                                )
                                tfm.transform.translation.y = float(
                                    p[1]
                                )
                                tfm.transform.translation.z = float(
                                    p[2]
                                )

                                tfm.transform.rotation.w = float(
                                    qwxyz[0]
                                )
                                tfm.transform.rotation.x = float(
                                    qwxyz[1]
                                )
                                tfm.transform.rotation.y = float(
                                    qwxyz[2]
                                )
                                tfm.transform.rotation.z = float(
                                    qwxyz[3]
                                )

                                tf_msgs.append(tfm)

                            if tf_msgs:
                                object_tf_broadcaster.sendTransform(
                                    tf_msgs
                                )
                        except Exception as exc:
                            print(
                                f"Object TF publish error: {exc}",
                                flush=True,
                            )

                        setattr(
                            run_sim,
                            "_v39_last_tf_wall",
                            now_ros,
                        )

                    # Re-publish PlanningScene periodically so MoveIt
                    # started after MuJoCo still receives the table + household egg scene.
                    if now_ros - ros_last_scene_wall >= 2.0:
                        try:
                            _publish_mujoco_scene()
                        except Exception as exc:
                            print(
                                f"PlanningScene publish error: {exc}",
                                flush=True,
                            )
                        ros_last_scene_wall = now_ros

                    if ros_cmd_seen:
                        ros_last_cmd_age.value = float(
                            max(
                                0.0,
                                now_ros - ros_last_command_wall,
                            )
                        )
                    else:
                        ros_last_cmd_age.value = -1.0

                target = shared_array(target_shared)

                # Per-joint reference speed.
                speed_pct = np.clip(
                    shared_array(speed_shared),
                    1.0,
                    100.0,
                )
                max_ref_vel = (
                    FULL_SPEED_REF_VEL * speed_pct / 100.0
                )

                max_step = max_ref_vel * dt
                delta = np.clip(
                    target - q_ref,
                    -max_step,
                    max_step,
                )
                q_ref += delta

                qd_ref = (q_ref - q_ref_prev) / dt
                q_ref_prev[:] = q_ref

                wn = np.clip(
                    shared_array(wn_shared),
                    0.05,
                    50.0,
                )
                zeta = np.clip(
                    shared_array(zeta_shared),
                    0.05,
                    5.0,
                )
                max_qacc = np.clip(
                    shared_array(qacc_shared),
                    0.05,
                    200.0,
                )

                # Same acceleration-space second-order controller
                # used by the stable version.
                qacc_cmd = (
                    (wn ** 2) * (q_ref - q)
                    + (2.0 * zeta * wn) * (qd_ref - qd)
                )
                qacc_cmd = np.clip(
                    qacc_cmd,
                    -max_qacc,
                    max_qacc,
                )

                # --------------------------------------------------------
                # V10 ARM CONTROL = EXACT STABLE V5/V6 APPROACH
                #
                # IMPORTANT:
                # Do not change the verified arm mapping/control path here.
                # J1..J7 are addressed ONLY through qpos_addr / dof_addr
                # obtained from the exact MuJoCo joint names.
                # --------------------------------------------------------

                # Inverse dynamics for the ARM ONLY, same as stable V5/V6.
                data.qacc[:] = 0.0
                for i, dadr in enumerate(dof_addr):
                    data.qacc[dadr] = qacc_cmd[i]

                mujoco.mj_inverse(model, data)
                required_force = data.qfrc_inverse.copy()

                # V13: DO NOT add qfrc_constraint here.
                # Keep the exact V10 stable arm force path.

                # Clear controls / arbitrary forces first.
                if model.nu > 0:
                    data.ctrl[:] = 0.0

                data.qfrc_applied[:] = 0.0
                data.xfrc_applied[:] = 0.0

                # Apply ONLY the seven verified arm generalized forces.
                for i, dadr in enumerate(dof_addr):
                    data.qfrc_applied[dadr] = required_force[dadr]

                # --------------------------------------------------------
                # V16 GRIPPER CONTROL
                #
                # POSITION MODE:
                #   slider -> position actuator
                #
                # FORCE MODE:
                #   "Clamp" button -> direct motor force.
                #   q1 increases while opening, so closing force is negative.
                #
                # The position actuator is commanded to CURRENT q during
                # force mode. That removes position spring error while keeping
                # its velocity damping term active for stable squeezing.
                # --------------------------------------------------------
                grip_target_q = float(np.clip(
                    float(gripper_target_mm.value) / 2000.0,
                    0.0,
                    0.05,
                ))

                force_mode_now = bool(gripper_force_mode.value)

                if force_mode_now:
                    user_target_per_finger = float(np.clip(
                        abs(float(gripper_force_N.value)),
                        0.5,
                        25.0,
                    ))

                    if not clamp_mode_prev:
                        contact_latched = False
                        contact_started_at = 0.0
                        clamp_applied_force = 0.0

                        deactivate_all_holds()
                        hold_ready_time = 0.0
                        hold_obj_bid = -1
                        hold_active = False
                        hold_active_shared.value = False
                        hold_name_shared.value = -1
                        hold_ready_shared.value = 0.0
                        hold_now_request.value = False
                        hold_off_request.value = False
                        hold_status_code_shared.value = 0
                        last_contact_obj_bid = -1
                        auto_hold_timer = 0.0

                        anti_slip_boost = 0.0
                        anti_slip_obj_bid = -1
                        anti_slip_prev_rel_z = None
                        anti_slip_vz_lp = 0.0
                        anti_slip_persist = 0.0

                        force_integral = 0.0
                        motor_clamp_cmd = 0.0

                    left_touch, right_touch, contact_dist = (
                        gripper_contact_sides()
                    )
                    both_touch = bool(left_touch and right_touch)

                    if both_touch and not contact_latched:
                        contact_latched = True
                        contact_started_at = time.perf_counter()
                        force_integral = 0.0

                        anti_slip_obj_bid = contacted_object_body()
                        anti_slip_prev_rel_z = None
                        anti_slip_vz_lp = 0.0
                        anti_slip_persist = 0.0

                    # If one side loses contact, go back to gentle approach.
                    if (
                        contact_latched
                        and not both_touch
                        and not hold_active
                    ):
                        contact_latched = False
                        force_integral = 0.0

                    # Neutralize position spring; retain built-in damping.
                    data.ctrl[gripper_actuator_id] = float(
                        np.clip(gripper_q, 0.0, 0.05)
                    )

                    if not contact_latched:
                        # ------------------------------------------------
                        # V31 STAGE A: FAST FREE-SPACE APPROACH
                        #
                        # User-selected multiplier controls only the
                        # free-space closing speed.
                        #
                        # The instant either fingertip contacts an object,
                        # automatically brake to 1.5x the original V30
                        # speed.  This preserves the fast one-click clamp
                        # without carrying 10x impact speed into the object.
                        # ------------------------------------------------
                        requested_mult = float(np.clip(
                            float(clamp_speed_mult_shared.value),
                            1.0,
                            20.0,
                        ))

                        any_touch = bool(left_touch or right_touch)

                        if any_touch:
                            active_mult = min(
                                requested_mult,
                                CONTACT_BRAKE_MULT,
                            )
                        else:
                            active_mult = requested_mult

                        active_approach_qvel = (
                            BASE_APPROACH_QVEL * active_mult
                        )
                        desired_qd = -active_approach_qvel

                        motor_force = APPROACH_VEL_K * (
                            desired_qd - gripper_qd
                        )
                        motor_force = float(np.clip(
                            motor_force,
                            -APPROACH_FORCE_LIMIT,
                            APPROACH_FORCE_LIMIT,
                        ))

                        if gripper_qd < -active_approach_qvel:
                            motor_force = max(0.0, motor_force)

                        motor_clamp_cmd = max(0.0, -motor_force)
                        clamp_applied_force = motor_clamp_cmd
                        data.ctrl[gripper_force_actuator_id] = motor_force

                        anti_slip_boost = 0.0
                        anti_slip_slip_mm_s_shared.value = 0.0
                        force_integral = 0.0

                    else:
                        # ------------------------------------------------
                        # MILD ANTI-SLIP:
                        # detect very slow downward creep relative to link7.
                        # Unlike V21, this does NOT alter friction.
                        # It only raises the NORMAL-FORCE TARGET slowly.
                        # ------------------------------------------------
                        current_obj_bid = contacted_object_body()
                        if current_obj_bid >= 0:
                            if current_obj_bid != anti_slip_obj_bid:
                                anti_slip_obj_bid = current_obj_bid
                                anti_slip_prev_rel_z = None
                                anti_slip_vz_lp = 0.0
                                anti_slip_persist = 0.0
                                anti_slip_boost = 0.0

                            rel_z = float(
                                data.xpos[anti_slip_obj_bid, 2]
                                - data.xpos[link7_bid, 2]
                            )

                            if anti_slip_prev_rel_z is not None:
                                raw_vz = (
                                    rel_z - anti_slip_prev_rel_z
                                ) / dt

                                # ~80 ms low-pass filter.
                                alpha = dt / (0.080 + dt)
                                anti_slip_vz_lp += alpha * (
                                    raw_vz - anti_slip_vz_lp
                                )

                            anti_slip_prev_rel_z = rel_z
                        else:
                            anti_slip_vz_lp *= 0.98

                        slip_down_mps = max(0.0, -anti_slip_vz_lp)
                        anti_slip_slip_mm_s_shared.value = (
                            1000.0 * anti_slip_vz_lp
                        )

                        contact_age = max(
                            0.0,
                            time.perf_counter() - contact_started_at,
                        )

                        # Detect much slower creep than V21:
                        # 0.15 mm/s instead of 1.5 mm/s.
                        if contact_age > 1.0:
                            if slip_down_mps > 0.00015:
                                anti_slip_persist += dt
                            else:
                                anti_slip_persist = max(
                                    0.0,
                                    anti_slip_persist - 2.0 * dt,
                                )

                            if anti_slip_persist > 0.25:
                                anti_slip_boost = min(
                                    MAX_EXTRA_TARGET_N,
                                    anti_slip_boost + 0.25 * dt,
                                )
                            elif slip_down_mps < 0.00005:
                                anti_slip_boost = max(
                                    0.0,
                                    anti_slip_boost - 0.10 * dt,
                                )

                        anti_slip_boost_shared.value = anti_slip_boost
                        anti_slip_mu_shared.value = 10.0

                        # ------------------------------------------------
                        # V22 TARGET CONTACT FORCE PER FINGER.
                        # Ramp target after bilateral contact to prevent
                        # impact, then close the loop on the weaker side.
                        # ------------------------------------------------
                        elapsed_contact = max(
                            0.0,
                            time.perf_counter() - contact_started_at,
                        )
                        target_ramp = min(
                            1.0,
                            elapsed_contact / CONTACT_FORCE_RAMP_SEC,
                        )

                        effective_target = (
                            user_target_per_finger + anti_slip_boost
                        ) * target_ramp

                        # Control the weaker fingertip so both sides must
                        # actually reach the requested normal force.
                        actual_control_force = min(
                            measured_left_n,
                            measured_right_n,
                        )

                        force_error = (
                            effective_target - actual_control_force
                        )

                        # Conditional integration / anti-windup.
                        if abs(force_error) < 8.0:
                            force_integral += force_error * dt
                            force_integral = float(np.clip(
                                force_integral,
                                -FORCE_INTEGRAL_LIMIT,
                                FORCE_INTEGRAL_LIMIT,
                            ))

                        desired_motor_cmd = (
                            MOTOR_FEEDFORWARD_GAIN * effective_target
                            + FORCE_KP * force_error
                            + FORCE_KI * force_integral
                        )
                        desired_motor_cmd = float(np.clip(
                            desired_motor_cmd,
                            0.0,
                            MOTOR_MAX_N,
                        ))

                        # Very gentle actuator-force slew.
                        if desired_motor_cmd > motor_clamp_cmd:
                            motor_clamp_cmd = min(
                                desired_motor_cmd,
                                motor_clamp_cmd
                                + MOTOR_RISE_N_PER_S * dt,
                            )
                        else:
                            motor_clamp_cmd = max(
                                desired_motor_cmd,
                                motor_clamp_cmd
                                - MOTOR_FALL_N_PER_S * dt,
                            )

                        closing_damping = -8.0 * gripper_qd
                        motor_force = (
                            -motor_clamp_cmd + closing_damping
                        )

                        penetration = max(
                            0.0,
                            -float(contact_dist),
                        )

                        # Preserve V19 penetration protection.
                        if penetration > 0.00025:
                            scale = 1.0 - (
                                (penetration - 0.00025) / 0.00030
                            )
                            scale = float(np.clip(scale, 0.0, 1.0))
                            motor_force = (
                                -motor_clamp_cmd * scale
                                + closing_damping
                            )

                        if penetration > 0.00055:
                            motor_force = max(
                                0.0,
                                closing_damping,
                            )

                        if penetration > 0.00080:
                            motor_force = 0.20

                        clamp_applied_force = motor_clamp_cmd
                        data.ctrl[gripper_force_actuator_id] = float(
                            np.clip(motor_force, -50.0, 50.0)
                        )

                else:
                    contact_latched = False
                    clamp_applied_force = 0.0
                    data.ctrl[gripper_force_actuator_id] = 0.0
                    data.ctrl[gripper_actuator_id] = grip_target_q

                    anti_slip_boost = 0.0
                    anti_slip_obj_bid = -1
                    anti_slip_prev_rel_z = None
                    anti_slip_vz_lp = 0.0
                    anti_slip_persist = 0.0

                    force_integral = 0.0
                    motor_clamp_cmd = 0.0

                    deactivate_all_holds()
                    hold_ready_time = 0.0
                    hold_obj_bid = -1
                    hold_active = False
                    hold_active_shared.value = False
                    hold_name_shared.value = -1
                    hold_ready_shared.value = 0.0
                    hold_now_request.value = False
                    hold_off_request.value = False
                    hold_status_code_shared.value = 0
                    last_contact_obj_bid = -1
                    auto_hold_timer = 0.0

                    anti_slip_boost_shared.value = 0.0
                    anti_slip_slip_mm_s_shared.value = 0.0
                    anti_slip_mu_shared.value = 10.0
                    contact_motor_cmd_shared.value = 0.0

                clamp_mode_prev = force_mode_now

                # ------------------------------------------------------
                # V30 AUTO CREEP HOLD
                #
                # Automatic trigger:
                #   force clamp mode
                #   + current/recent gripper-object contact
                #   + downward slip <= -0.05 mm/s
                #   + persists >= 0.10 s
                #
                # Release/opening disables HOLD immediately.
                # Manual HOLD ON / HOLD OFF remain available.
                # ------------------------------------------------------
                candidate_bid = contacted_object_body()

                if candidate_bid >= 0:
                    last_contact_obj_bid = int(candidate_bid)

                # ------------------------------------------------------
                # HOLD OFF: explicit user request.
                # ------------------------------------------------------
                if bool(hold_off_request.value):
                    hold_off_request.value = False
                    deactivate_all_holds()
                    hold_ready_time = 0.0
                    hold_obj_bid = -1
                    hold_active = False
                    auto_hold_timer = 0.0
                    hold_active_shared.value = False
                    hold_name_shared.value = -1
                    hold_ready_shared.value = 0.0
                    hold_status_code_shared.value = 0

                # ------------------------------------------------------
                # HOLD OFF: any opening / release state.
                #
                # force_mode_now=False means gripper has left CLAMP force
                # mode and is back in position/opening mode.
                # ------------------------------------------------------
                if (not force_mode_now) and hold_active:
                    deactivate_all_holds()
                    hold_ready_time = 0.0
                    hold_obj_bid = -1
                    hold_active = False
                    auto_hold_timer = 0.0
                    hold_active_shared.value = False
                    hold_name_shared.value = -1
                    hold_ready_shared.value = 0.0
                    hold_status_code_shared.value = 0

                # ------------------------------------------------------
                # Manual HOLD ON override.
                # ------------------------------------------------------
                if bool(hold_now_request.value):
                    hold_now_request.value = False

                    manual_bid = (
                        int(candidate_bid)
                        if candidate_bid >= 0
                        else int(last_contact_obj_bid)
                    )

                    if manual_bid >= 0:
                        hold_obj_bid = manual_bid

                        if activate_connect_hold(hold_obj_bid):
                            hold_active = True
                            auto_hold_timer = AUTO_HOLD_PERSIST_SEC
                            hold_active_shared.value = True
                            hold_name_shared.value = int(hold_obj_bid)
                            hold_ready_shared.value = AUTO_HOLD_PERSIST_SEC
                            hold_status_code_shared.value = 1
                        else:
                            hold_status_code_shared.value = 2
                    else:
                        hold_status_code_shared.value = 2

                # ------------------------------------------------------
                # Automatic HOLD ON from downward creep.
                # ------------------------------------------------------
                if (
                    bool(hold_enabled_shared.value)
                    and force_mode_now
                    and not hold_active
                ):
                    auto_bid = (
                        int(candidate_bid)
                        if candidate_bid >= 0
                        else int(last_contact_obj_bid)
                    )

                    contact_present = (auto_bid >= 0)

                    # anti_slip_vz_lp is object Z relative to link7 [m/s].
                    slip_mm_s = 1000.0 * float(anti_slip_vz_lp)

                    creep_condition = (
                        contact_present
                        and slip_mm_s <= AUTO_HOLD_SLIP_MM_S
                    )

                    if creep_condition:
                        auto_hold_timer = min(
                            AUTO_HOLD_PERSIST_SEC,
                            auto_hold_timer + dt,
                        )
                    else:
                        # Fast reset is okay here: HOLD should only trigger
                        # after a genuinely persistent downward creep.
                        auto_hold_timer = max(
                            0.0,
                            auto_hold_timer - 2.0 * dt,
                        )

                    hold_ready_shared.value = auto_hold_timer

                    if (
                        auto_hold_timer >= AUTO_HOLD_PERSIST_SEC
                        and auto_bid >= 0
                    ):
                        hold_obj_bid = int(auto_bid)

                        if activate_connect_hold(hold_obj_bid):
                            hold_active = True
                            hold_active_shared.value = True
                            hold_name_shared.value = int(hold_obj_bid)
                            hold_status_code_shared.value = 1

                # Checkbox OFF forces HOLD release and disables auto trigger.
                if not bool(hold_enabled_shared.value):
                    auto_hold_timer = 0.0
                    hold_ready_shared.value = 0.0

                    if hold_active:
                        deactivate_all_holds()
                        hold_obj_bid = -1
                        hold_active = False
                        hold_active_shared.value = False
                        hold_name_shared.value = -1
                        hold_status_code_shared.value = 0

                mujoco.mj_step2(model, data)

                # ------------------------------------------------------
                # V22 REAL CONTACT-FORCE MEASUREMENT
                # Use the solved contact forces from this physics step.
                # These values drive the NEXT 1-ms control iteration.
                # ------------------------------------------------------
                (
                    measured_left_n,
                    measured_right_n,
                    measured_tangent_n,
                ) = measure_pad_contact_forces()

                contact_left_n_shared.value = measured_left_n
                contact_right_n_shared.value = measured_right_n
                contact_tangent_n_shared.value = measured_tangent_n
                contact_motor_cmd_shared.value = motor_clamp_cmd

                # ------------------------------------------------------
                # POST-STEP RUNAWAY GUARD
                # ------------------------------------------------------
                arm_q_now = np.array(
                    [data.qpos[qadr] for qadr in qpos_addr],
                    dtype=float,
                )
                arm_qd_now = np.array(
                    [data.qvel[dadr] for dadr in dof_addr],
                    dtype=float,
                )

                finite_ok = (
                    np.all(np.isfinite(arm_q_now))
                    and np.all(np.isfinite(arm_qd_now))
                )
                speed_ok = np.all(np.abs(arm_qd_now) < 6.0)
                runaway_joint_idx = int(np.argmax(np.abs(arm_qd_now)))
                runaway_joint_speed = float(abs(arm_qd_now[runaway_joint_idx]))

                range_ok = True
                for i in range(7):
                    lo, hi = JOINT_LIMITS_RAD[i]
                    if (
                        arm_q_now[i] < lo - 0.05
                        or arm_q_now[i] > hi + 0.05
                    ):
                        range_ok = False
                        break

                if bool(ros_arm_command_gate_shared.value):
                    try:
                        _v301_record(
                            [round(time.perf_counter() - _v301_state["t0"], 4), 1, int(bool(hold_active))]
                            + [float(v) for v in arm_q_now] + [float(v) for v in arm_qd_now]
                            + [float(v) for v in qacc_cmd]
                            + [float(data.qacc[dadr]) for dadr in dof_addr]
                            + [float(required_force[dadr]) for dadr in dof_addr]
                        )
                    except Exception:
                        pass

                if not (finite_ok and speed_ok and range_ok):
                    ros_motion_fault_latched = True
                    _v301_dump(f"runaway_j{runaway_joint_idx+1}")
                    if finite_ok and range_ok:
                        # Excessive velocity but still a numerically valid pose:
                        # freeze in place and preserve the grasp.
                        print(
                            "SAFETY GUARD: high arm speed detected | "
                            f"max=joint{runaway_joint_idx+1}:{runaway_joint_speed:.3f} rad/s | "
                            f"qvel={np.round(arm_qd_now, 3)} -> FREEZE-IN-PLACE + "
                            "MOVEIT COMMAND LATCH",
                            flush=True,
                        )
                        soft_freeze_arm(reason_code=4, count_trip=True)
                    else:
                        # NaN/out-of-range state is a true numerical emergency;
                        # only here do we retain the old hard checkpoint restore.
                        print(
                            "SAFETY GUARD: invalid arm state -> emergency hard restore",
                            flush=True,
                        )
                        hard_restore_arm(
                            last_safe_q,
                            gripper_q=last_safe_gripper_q,
                            reason_code=4,
                            keep_hold=True,
                        )
                    continue

                # Update safe checkpoint only when there is no meaningful
                # protected-surface penetration and arm speed is moderate.
                penetration_after = protected_penetration_depth()
                if (
                    penetration_after > -0.0005
                    and np.all(np.abs(arm_qd_now) < 2.5)
                    and time.perf_counter() >= safety_cooldown_until
                ):
                    last_safe_q[:] = arm_q_now
                    last_safe_gripper_q = float(
                        np.clip(data.qpos[gripper_qpos_addr], 0.0, 0.05)
                    )
                    if int(safety_state.value) != 0:
                        safety_state.value = 0

                for i, qadr in enumerate(qpos_addr):
                    actual_shared[i] = float(data.qpos[qadr])

                gripper_actual_mm.value = float(
                    np.clip(2.0 * data.qpos[gripper_qpos_addr] * 1000.0, 0.0, 100.0)
                )

                now = time.perf_counter()
                if now - getattr(run_sim, "_v330_model_t", 0.0) >= 1.0:
                    setattr(run_sim, "_v330_model_t", now)
                    _v330_apply_wrist_model_visibility(model)
                if now - last_viewer_sync >= viewer_period:
                    viewer.sync()
                    last_viewer_sync = now

                if v308_qpos is not None and now - _v308_last_qpos_write >= 0.10:
                    _v308_last_qpos_write = now
                    with v308_qpos.get_lock():
                        v308_qpos[:_v308_nq] = data.qpos.tolist()
                        v308_meta[1] = float(data.time)
                        v308_meta[2] += 1.0

                next_step_wall += dt
                sleep_time = next_step_wall - time.perf_counter()

                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    next_step_wall = time.perf_counter()

    except Exception:
        sim_error_code.value = 1
        sim_ready.value = False

        print("\n========== NERO SIMULATION ERROR ==========", flush=True)
        traceback.print_exc()
        print("===========================================\n", flush=True)

        # Keep traceback visible in terminal for diagnosis.
        time.sleep(0.5)


def run_gui(
    sim_process,
    target_shared,
    actual_shared,
    speed_shared,
    wn_shared,
    zeta_shared,
    qacc_shared,
    gravity_enabled,
    gripper_target_mm,
    gripper_actual_mm,
    gripper_force_mode,
    gripper_force_N,
    clamp_speed_mult_shared,
    anti_slip_boost_shared,
    anti_slip_slip_mm_s_shared,
    anti_slip_mu_shared,
    contact_left_n_shared,
    contact_right_n_shared,
    contact_tangent_n_shared,
    contact_motor_cmd_shared,
    hold_enabled_shared,
    hold_active_shared,
    hold_name_shared,
    hold_ready_shared,
    hold_now_request,
    hold_off_request,
    hold_status_code_shared,
    reset_zero_request,
    recover_request,
    safety_trip_count,
    safety_state,
    sim_ready,
    sim_error_code,
    ros_bridge_ready,
    ros_cmd_count,
    ros_feedback_count,
    ros_scene_count,
    ros_last_cmd_age,
    ros_last_cmd_joint_count,
    ros_controller_state_rx_shared,
    ros_fallback_joint_state_rx_shared,
    ros_last_command_source_shared,
    ros_arm_command_gate_shared,
    stop_event,
):
    # Import GUI only in main process.
    import tkinter as tk
    from tkinter import ttk, messagebox

    root = tk.Tk()
    root.title("NERO V2-9 Skill-Contract Agent + V69 Motion Stack")
    root.geometry("1360x930")
    root.minsize(1120, 680)

    # V56: all tabs share one persistent log panel at the bottom.
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    root.rowconfigure(1, weight=0)

    notebook = ttk.Notebook(root)
    notebook.grid(row=0, column=0, sticky="nsew")

    global_log_frame = ttk.LabelFrame(root, text="Unified Runtime Log")
    global_log_frame.grid(
        row=1, column=0, sticky="nsew", padx=8, pady=(2, 8)
    )
    global_log_frame.columnconfigure(0, weight=1)
    global_log_frame.rowconfigure(0, weight=0)
    global_log_frame.rowconfigure(1, weight=1)

    global_log = tk.Text(
        global_log_frame, height=11, wrap="word", undo=False
    )
    global_log.grid(row=1, column=0, sticky="nsew")
    global_log_scroll = ttk.Scrollbar(
        global_log_frame, orient="vertical", command=global_log.yview
    )
    global_log_scroll.grid(row=1, column=1, sticky="ns")
    global_log.configure(yscrollcommand=global_log_scroll.set)
    global_log.insert(
        "end",
        "NERO V2-9 unified runtime log. V71 hybrid GPU/BioIK motion + TRAC reference + Skill Contract / Agent messages appear here.\n\n",
    )

    # V2-9 LOG STYLE — presentation only. No motion/startup/orchestration changes.
    global_log.tag_configure(
        "llm_message_bold", font=("Arial", 10, "bold")
    )
    global_log.tag_configure(
        "task_summary_success", foreground="#138A36", font=("Arial", 10, "bold")
    )
    global_log.tag_configure(
        "task_summary_warning", foreground="#A66B00", font=("Arial", 10, "bold")
    )
    global_log.tag_configure(
        "task_summary_failure", foreground="#C62828", font=("Arial", 10, "bold")
    )

    # V2-36 GUI TASK STATUS LAMP
    _v236_lamp={"canvas":None,"dot":None,"active":False,"terminal":None,"pulse":0,"parent":None}
    def _v236_lamp_ensure():
        # V3.11: the lamp lives in the log toolbar, right of "Copy Full Log", so it never
        # covers log text (V2-37 placed it inside the text widget).
        if _v236_lamp.get("canvas") is not None or _v236_lamp.get("parent") is None: return
        try:
            par=_v236_lamp["parent"]
            c=tk.Canvas(par,width=26,height=26,highlightthickness=0,bd=0)
            d=c.create_oval(3,3,23,23,fill="#7a7a7a",outline="#404040",width=2)
            _v236_lamp["canvas"],_v236_lamp["dot"]=c,d
            c.pack(side="left",padx=(2,6))
        except Exception: pass
    def _v236_lamp_set(color):
        _v236_lamp_ensure()
        try:
            if _v236_lamp.get("canvas") is not None:
                _v236_lamp["canvas"].itemconfigure(_v236_lamp["dot"],fill=color); _v236_lamp["canvas"].lift()
        except Exception: pass
    def _v236_lamp_pulse(color,ms=320):
        _v236_lamp["pulse"]+=1; token=_v236_lamp["pulse"]; _v236_lamp_set(color)
        def _settle():
            if token!=_v236_lamp.get("pulse") or _v236_lamp.get("terminal"): return
            _v236_lamp_set("#2e7d32" if _v236_lamp.get("active") else "#7a7a7a")
        try: root.after(ms,_settle)
        except Exception: pass
    def _v236_status_from_log(msg):
        u=str(msg).upper()
        if "[TASK SUMMARY GREEN]" in u or "STATUS: SUCCESS" in u or "END TASK WITH SUCCESS" in u:
            _v236_lamp.update({"active":False,"terminal":"success"}); _v236_lamp["pulse"]+=1; _v236_lamp_set("#1976d2"); return
        if "[TASK SUMMARY RED]" in u or "STATUS: FAILED" in u or "END TASK WITH FAILURE" in u:
            _v236_lamp.update({"active":False,"terminal":"failure"}); _v236_lamp["pulse"]+=1; _v236_lamp_set("#d32f2f"); return
        if "[LLM TASK START]" in u or "[TASK CONFIRMED BY USER]" in u:
            _v236_lamp.update({"active":True,"terminal":None}); _v236_lamp_pulse("#39d353",380); return
        if not _v236_lamp.get("active"): _v236_lamp_ensure(); return
        recovery=("NO_IK" in u or "NO IK" in u or "REPLAN" in u or "FALLBACK" in u or "RETREAT" in u or
                  "RECOVERY" in u or "REANCHOR" in u or "PUSHER FAILOVER" in u or "PLANNING FAILED" in u or
                  "STRICT FAIL" in u or "ORIENT REJECT" in u or "EDGE REPOSITION START" in u)
        if recovery: _v236_lamp_pulse("#f9a825",650)
        elif "[LLM TOOL RESULT #" in u or "[SKILL RESULT]" in u: _v236_lamp_pulse("#39d353",300)

    def append_gui_log(text):
        _v236_status_from_log(str(text))
        _msg = str(text).rstrip()
        _tag = None
        if _msg.startswith("[TASK SUMMARY GREEN]"):
            _tag = "task_summary_success"
        elif _msg.startswith("[TASK SUMMARY YELLOW]"):
            _tag = "task_summary_warning"
        elif _msg.startswith("[TASK SUMMARY RED]"):
            _tag = "task_summary_failure"
        elif _msg.startswith("[LLM"):
            _tag = "llm_message_bold"

        if _tag:
            global_log.insert("end", _msg + "\n", _tag)
        else:
            global_log.insert("end", _msg + "\n")
        global_log.see("end")


    def copy_full_log_to_clipboard():
        try:
            payload = global_log.get("1.0", "end-1c")
            root.clipboard_clear()
            root.clipboard_append(payload)
            root.update_idletasks()
            append_gui_log(f"[GUI] Copied full runtime log to clipboard ({len(payload)} chars)")
        except Exception as exc:
            messagebox.showerror("NERO V2-9", f"Could not copy log:\n{exc}")

    global_log_toolbar = ttk.Frame(global_log_frame)
    global_log_toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=(3, 3))
    ttk.Button(
        global_log_toolbar,
        text="Copy Full Log",
        command=copy_full_log_to_clipboard,
    ).pack(side="left", padx=(0, 6))
    # V3.11: task status lamp right of the button (was drawn over the log text).
    _v236_lamp["parent"] = global_log_toolbar
    _v236_lamp_ensure()

    basic_tab = ttk.Frame(notebook)
    skill_tab = ttk.Frame(notebook)
    bio_tab = ttk.Frame(notebook)
    trac_tab = ttk.Frame(notebook)
    llm_tab = ttk.Frame(notebook)

    notebook.add(basic_tab, text="Basic Controls")
    notebook.add(skill_tab, text="Skill Test")
    notebook.add(bio_tab, text="BioIK Primary")
    notebook.add(trac_tab, text="TRAC-IK Reference")
    notebook.add(llm_tab, text="NERO SimAgent")

    skill_core = None
    skill_core_error = ""
    try:
        from NERO_V71_Hybrid_GPU_Planner_Core import (
            NeroMoveItCore, OBJECTS, BINS, OBJECT_ZONE,
            GRASP_Z_OFFSET, DEFAULT_GRASP_Z_OFFSET, BIN_RELEASE_CLEARANCE,
        )

        def set_ros_arm_gate(enabled):
            ros_arm_command_gate_shared.value = bool(enabled)

        def get_safety_trip_count():
            return int(safety_trip_count.value)

        def request_soft_recover():
            # V69 compatibility callback. Automatic execution no longer calls
            # this on safety trips; only explicit recovery actions may unlatch.
            recover_request.value = True

        skill_core = NeroMoveItCore(
            set_arm_command_gate=set_ros_arm_gate,
            get_safety_trip_count=get_safety_trip_count,
            request_soft_recover=request_soft_recover,
        )
        skill_core.start()
        # Semantic task layer only: V69 module globals stay frozen.  The V2-7
        # backend exposes household egg objects to the Agent and GUI.
        OBJECTS = ["egg_1", "egg_2", "egg_3", "egg_4"]
    except Exception as exc:
        OBJECTS = ["egg_1", "egg_2", "egg_3", "egg_4"]
        skill_core_error = str(exc)

    def take_manual_arm_control(reason="manual arm control"):
        ros_arm_command_gate_shared.value = False

        if skill_core is not None:
            try:
                skill_core.force_recover(f"MANUAL ARM — {reason}")
            except Exception:
                try:
                    skill_core.cancel_active_motion()
                except Exception:
                    pass

    def take_manual_gripper_control(reason="manual gripper control"):
        if skill_core is not None:
            try:
                skill_core.emit(
                    f"[MODE] GRIPPER MANUAL — {reason}; "
                    "arm grasp branch preserved"
                )
            except Exception:
                pass

    # Make scrolling possible on smaller displays.
    outer = ttk.Frame(basic_tab)
    outer.pack(fill="both", expand=True)

    canvas = tk.Canvas(outer, highlightthickness=0)
    scrollbar = ttk.Scrollbar(
        outer,
        orient="vertical",
        command=canvas.yview,
    )
    content = ttk.Frame(canvas)

    content.bind(
        "<Configure>",
        lambda e: canvas.configure(
            scrollregion=canvas.bbox("all")
        ),
    )

    canvas.create_window(
        (0, 0),
        window=content,
        anchor="nw",
    )
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(
        side="left",
        fill="both",
        expand=True,
    )
    scrollbar.pack(
        side="right",
        fill="y",
    )

    title = ttk.Label(
        content,
        text="NERO V2-9 Skill-Contract Agent + V69 Motion Stack",
        font=("Arial", 18, "bold"),
    )
    title.pack(pady=(10, 4))

    status_var = tk.StringVar(value="Starting simulation...")
    status = ttk.Label(
        content,
        textvariable=status_var,
        font=("Arial", 11, "bold"),
    )
    status.pack(pady=(0, 8))

    # --------------------------------------------------------
    # V44 MoveIt2 bridge status
    # --------------------------------------------------------
    moveit_box = ttk.LabelFrame(
        content,
        text="MoveIt2 / ROS2 Bridge",
    )
    moveit_box.pack(
        fill="x",
        padx=12,
        pady=(2, 8),
    )

    moveit_status_var = tk.StringVar(
        value="ROS2 bridge: starting..."
    )
    ttk.Label(
        moveit_box,
        textvariable=moveit_status_var,
        font=("Arial", 10, "bold"),
    ).pack(
        anchor="w",
        padx=8,
        pady=(5, 2),
    )

    ttk.Label(
        moveit_box,
        text=(
            "MoveIt2 owns IK/path planning/collision avoidance. "
            "MuJoCo/V31 executes /arm_controller/controller_state (primary) or /control/joint_states (fallback), and publishes "
            "/feedback/joint_states. PlanningScene includes table + sorting bins."
        ),
    ).pack(
        anchor="w",
        padx=8,
        pady=(0, 5),
    )

    top = ttk.Frame(content)
    top.pack(fill="x", padx=12, pady=4)

    gravity_var = tk.BooleanVar(value=True)

    def gravity_callback():
        gravity_enabled.value = bool(gravity_var.get())

    gravity_check = ttk.Checkbutton(
        top,
        text="Gravity ON",
        variable=gravity_var,
        command=gravity_callback,
    )
    gravity_check.pack(side="left", padx=(0, 20))

    ttk.Label(
        top,
        text="Speed: 20% = original V2 speed | 100% = 5× V2 reference speed",
    ).pack(side="left")

    slider_vars = []
    target_labels = []
    actual_labels = []
    speed_vars = []
    wn_vars = []
    zeta_vars = []
    qacc_vars = []
    gain_labels = []

    joints_frame = ttk.Frame(content)
    joints_frame.pack(fill="both", expand=True, padx=12, pady=8)

    def update_gain_label(i):
        try:
            wn = float(wn_vars[i].get())
            zeta = float(zeta_vars[i].get())
            kp = wn * wn
            kd = 2.0 * zeta * wn
            gain_labels[i].configure(
                text=f"Kp≈{kp:.2f}  Kd≈{kd:.2f}"
            )
        except Exception:
            pass
    def slider_command(i, value):
        try:
            take_manual_arm_control(f"Joint {i+1} slider")
            deg = float(value)
            target_shared[i] = math.radians(deg)
            target_labels[i].configure(
                text=f"Target {deg:.1f}°"
            )
        except Exception:
            pass

    for i in range(7):
        lo, hi = JOINT_LIMITS_RAD[i]
        lo_deg = math.degrees(lo)
        hi_deg = math.degrees(hi)

        box = ttk.LabelFrame(
            joints_frame,
            text=f"Slider {i+1} → Joint {i+1}",
        )
        box.pack(fill="x", pady=5)

        box.columnconfigure(0, weight=1)

        s_var = tk.DoubleVar(value=0.0)
        slider = tk.Scale(
            box,
            from_=lo_deg,
            to=hi_deg,
            orient="horizontal",
            resolution=0.1,
            variable=s_var,
            command=lambda value, idx=i: slider_command(idx, value),
            length=520,
        )
        slider.grid(
            row=0,
            column=0,
            rowspan=2,
            sticky="ew",
            padx=(8, 10),
            pady=5,
        )

        t_label = ttk.Label(box, text="Target 0.0°", width=15)
        t_label.grid(row=0, column=1, padx=4)

        a_label = ttk.Label(box, text="Actual 0.0°", width=15)
        a_label.grid(row=1, column=1, padx=4)

        ttk.Label(box, text="Speed %").grid(row=0, column=2)
        speed_var = tk.StringVar(value="20")
        speed_spin = ttk.Spinbox(
            box,
            from_=1,
            to=100,
            increment=1,
            width=7,
            textvariable=speed_var,
        )
        speed_spin.grid(row=1, column=2, padx=4)

        ttk.Label(box, text="Wn").grid(row=0, column=3)
        wn_var = tk.StringVar(value="4.00")
        wn_spin = ttk.Spinbox(
            box,
            from_=0.05,
            to=50.0,
            increment=0.25,
            width=7,
            textvariable=wn_var,
        )
        wn_spin.grid(row=1, column=3, padx=4)

        ttk.Label(box, text="Zeta").grid(row=0, column=4)
        zeta_var = tk.StringVar(value="1.25")
        zeta_spin = ttk.Spinbox(
            box,
            from_=0.05,
            to=5.0,
            increment=0.05,
            width=7,
            textvariable=zeta_var,
        )
        zeta_spin.grid(row=1, column=4, padx=4)

        ttk.Label(box, text="Accel").grid(row=0, column=5)
        qacc_var = tk.StringVar(
            value=f"{DEFAULT_MAX_QACC[i]:.2f}"
        )
        qacc_spin = ttk.Spinbox(
            box,
            from_=0.05,
            to=200.0,
            increment=0.5,
            width=8,
            textvariable=qacc_var,
        )
        qacc_spin.grid(row=1, column=5, padx=4)

        gain_label = ttk.Label(
            box,
            text="Kp≈16.00  Kd≈10.00",
            width=23,
        )
        gain_label.grid(
            row=1,
            column=6,
            padx=(8, 8),
        )

        ttk.Label(
            box,
            text=f"Range {lo_deg:.1f}° … {hi_deg:.1f}°",
        ).grid(
            row=0,
            column=6,
            padx=(8, 8),
        )

        slider_vars.append(s_var)
        target_labels.append(t_label)
        actual_labels.append(a_label)
        speed_vars.append(speed_var)
        wn_vars.append(wn_var)
        zeta_vars.append(zeta_var)
        qacc_vars.append(qacc_var)
        gain_labels.append(gain_label)

    # --------------------------------------------------------
    # Original AgileX two-finger gripper
    # --------------------------------------------------------
    gripper_box = ttk.LabelFrame(
        content,
        text="Original AgileX Gripper",
    )
    gripper_box.pack(fill="x", padx=12, pady=(4, 8))

    gripper_box.columnconfigure(1, weight=1)

    ttk.Label(
        gripper_box,
        text="Opening",
        width=12,
    ).grid(row=0, column=0, padx=8, pady=8)

    gripper_var = tk.DoubleVar(value=0.0)

    def gripper_slider_command(value):
        try:
            take_manual_gripper_control("gripper slider")
            mm = max(0.0, min(100.0, float(value)))
            # Moving the manual slider always returns to position mode.
            gripper_force_mode.value = False
            gripper_target_mm.value = mm
            gripper_target_label.configure(text=f"Target {mm:.0f} mm")
            gripper_mode_label.configure(text="Mode: POSITION")
        except Exception:
            pass

    gripper_slider = tk.Scale(
        gripper_box,
        from_=0,
        to=100,
        orient="horizontal",
        resolution=1,
        variable=gripper_var,
        command=gripper_slider_command,
        length=650,
    )
    gripper_slider.grid(
        row=0,
        column=1,
        sticky="ew",
        padx=8,
        pady=8,
    )

    gripper_target_label = ttk.Label(
        gripper_box,
        text="Target 0 mm",
        width=16,
    )
    gripper_target_label.grid(row=0, column=2, padx=5)

    gripper_actual_label = ttk.Label(
        gripper_box,
        text="Actual 0 mm",
        width=16,
    )
    gripper_actual_label.grid(row=0, column=3, padx=5)

    ttk.Label(
        gripper_box,
        text="0 mm = closed    100 mm = fully open",
    ).grid(
        row=1,
        column=1,
        sticky="w",
        padx=8,
        pady=(0, 8),
    )

    # --------------------------------------------------------
    # V16 force-control panel
    # --------------------------------------------------------
    force_frame = ttk.Frame(gripper_box)
    force_frame.grid(
        row=2,
        column=0,
        columnspan=4,
        sticky="w",
        padx=8,
        pady=(4, 10),
    )

    ttk.Label(
        force_frame,
        text="Target normal force / finger (N):",
    ).pack(side="left", padx=(0, 6))

    gripper_force_var = tk.StringVar(value="5.0")
    gripper_force_spin = ttk.Spinbox(
        force_frame,
        from_=0.5,
        to=50.0,
        increment=0.5,
        width=8,
        textvariable=gripper_force_var,
    )
    gripper_force_spin.pack(side="left", padx=(0, 12))

    ttk.Label(
        force_frame,
        text="Clamp speed ×:",
    ).pack(side="left", padx=(0, 5))

    clamp_speed_var = tk.StringVar(value="15.0")
    clamp_speed_spin = ttk.Spinbox(
        force_frame,
        from_=1.0,
        to=20.0,
        increment=0.5,
        width=6,
        textvariable=clamp_speed_var,
    )
    clamp_speed_spin.pack(side="left", padx=(0, 12))

    gripper_mode_label = ttk.Label(
        force_frame,
        text="Mode: POSITION",
        width=38,
    )
    gripper_mode_label.pack(side="left", padx=(0, 12))

    def one_click_clamp():
        take_manual_gripper_control("Gripper FAST CLAMP")
        try:
            force_n = float(gripper_force_var.get())
        except Exception:
            force_n = 5.0
            gripper_force_var.set("5.0")

        force_n = max(0.5, min(50.0, force_n))
        gripper_force_N.value = force_n

        try:
            speed_mult = float(clamp_speed_var.get())
        except Exception:
            speed_mult = 15.0
            clamp_speed_var.set("15.0")

        speed_mult = max(1.0, min(20.0, speed_mult))
        clamp_speed_mult_shared.value = speed_mult

        gripper_force_mode.value = True

        gripper_mode_label.configure(
            text=(
                f"Mode: FORCE LOOP → {force_n:.1f} N/finger "
                f"| clamp {speed_mult:.1f}×"
            )
        )
        gripper_target_label.configure(
            text=f"Target {force_n:.1f} N/finger"
        )

    def one_click_release():
        take_manual_gripper_control("Gripper OPEN")
        # Release uses position mode and commands full opening.
        gripper_force_mode.value = False
        gripper_target_mm.value = 100.0
        gripper_var.set(100.0)

        gripper_mode_label.configure(text="Mode: RELEASE")
        gripper_target_label.configure(text="Target 100 mm")

    ttk.Button(
        force_frame,
        text="一键夹紧 / CLAMP",
        command=one_click_clamp,
    ).pack(side="left", padx=5)

    ttk.Button(
        force_frame,
        text="一键松开 / RELEASE",
        command=one_click_release,
    ).pack(side="left", padx=5)

    hold_var = tk.BooleanVar(value=True)

    def hold_toggle():
        hold_enabled_shared.value = bool(hold_var.get())

    ttk.Checkbutton(
        force_frame,
        text="Enable Auto HOLD",
        variable=hold_var,
        command=hold_toggle,
    ).pack(side="left", padx=(14, 5))

    def hold_now():
        hold_now_request.value = True

    ttk.Button(
        force_frame,
        text="HOLD ON",
        command=hold_now,
    ).pack(side="left", padx=5)

    def hold_off():
        hold_off_request.value = True

    ttk.Button(
        force_frame,
        text="HOLD OFF",
        command=hold_off,
    ).pack(side="left", padx=5)

    hold_enabled_shared.value = True

    ttk.Label(
        gripper_box,
        text=(
            "V31 clamp speed: selected multiplier applies in free space; "
            "first fingertip contact automatically brakes to 1.5× before "
            "the V22 force loop takes over."
        ),
    ).grid(
        row=3,
        column=0,
        columnspan=4,
        sticky="w",
        padx=8,
        pady=(0, 3),
    )

    ttk.Label(
        gripper_box,
        text="V22 closed loop: GUI target = measured normal contact force PER "
             "FINGER. Friction stays fixed at μ=10; no NoSlip/Newton tricks.",
    ).grid(
        row=4,
        column=0,
        columnspan=4,
        sticky="w",
        padx=8,
        pady=(0, 4),
    )

    contact_force_status_label = ttk.Label(
        gripper_box,
        text=(
            "Contact: L 0.00 N | R 0.00 N | Tangential 0.00 N | "
            "Motor 0.00 N"
        ),
    )
    contact_force_status_label.grid(
        row=5,
        column=0,
        columnspan=4,
        sticky="w",
        padx=8,
        pady=(0, 3),
    )

    anti_slip_status_label = ttk.Label(
        gripper_box,
        text="Slip: +0.00 mm/s | target boost +0.00 N | pad μ fixed 10.00",
    )
    anti_slip_status_label.grid(
        row=6,
        column=0,
        columnspan=4,
        sticky="w",
        padx=8,
        pady=(0, 8),
    )

    hold_status_label = ttk.Label(
        gripper_box,
        text="Auto HOLD: MONITORING creep | manual HOLD ON available",
    )
    hold_status_label.grid(
        row=7,
        column=0,
        columnspan=4,
        sticky="w",
        padx=8,
        pady=(0, 8),
    )

    bottom = ttk.Frame(content)
    bottom.pack(fill="x", padx=12, pady=(5, 15))

    def zero_all():
        take_manual_arm_control("ZERO ALL")
        for i in range(7):
            slider_vars[i].set(0.0)
            target_shared[i] = 0.0
            target_labels[i].configure(text="Target 0.0°")
        # V2-7 episode semantics: ZERO ALL is a fresh-task boundary when
        # no object is being held.  In V2-5.1 only the physical joint targets
        # were zeroed, leaving HOVER_REACHED / retry memory stale.
        if not bool(hold_active_shared.value):
            _v24_reset_skill_state("ZERO ALL", clear_adaptation=True)
        else:
            append_gui_log(
                "[V2-7 STATE RESET SKIPPED] ZERO ALL while hold_active=true; "
                "logical grasp state preserved"
            )

    ttk.Button(
        bottom,
        text="Zero All",
        command=zero_all,
    ).pack(side="left", padx=(0, 8))

    def emergency_recover():
        take_manual_arm_control("STOP + RECOVER")
        recover_request.value = True

    def hard_reset_zero():
        take_manual_arm_control("HARD RESET 0")
        gripper_force_mode.value = False
        reset_zero_request.value = True
        # HARD RESET is an explicit full episode reset.
        _v24_reset_skill_state("HARD RESET 0", clear_adaptation=True)
        # GUI is updated immediately; sim process then performs the
        # hard state reset and publishes the final actual values.
        for i in range(7):
            slider_vars[i].set(0.0)
            target_labels[i].configure(text="Target 0.0°")
        gripper_var.set(80.0)
        gripper_target_label.configure(text="Target 80 mm")

    ttk.Button(
        bottom,
        text="STOP + RECOVER",
        command=emergency_recover,
    ).pack(side="left", padx=8)

    ttk.Button(
        bottom,
        text="HARD RESET 0",
        command=hard_reset_zero,
    ).pack(side="left", padx=8)

    ttk.Button(
        bottom,
        text="Quit",
        command=root.destroy,
    ).pack(side="right")

    # ========================================================
    # V43 UNIFIED SKILL TEST TAB
    # ========================================================
    skill_outer = ttk.Frame(skill_tab)
    skill_outer.pack(fill="both", expand=True, padx=12, pady=12)

    skill_title = ttk.Label(
        skill_outer,
        text="MoveIt2 Skill Test",
        font=("Arial", 18, "bold"),
    )
    skill_title.pack(anchor="w", pady=(0, 5))

    skill_ready_var = tk.StringVar(
        value=(
            "Skill core starting..."
            if skill_core is not None
            else f"Skill core ERROR: {skill_core_error}"
        )
    )
    ttk.Label(
        skill_outer,
        textvariable=skill_ready_var,
        font=("Arial", 10, "bold"),
    ).pack(anchor="w", pady=(0, 3))

    control_mode_var = tk.StringVar(value="Arm input: MANUAL/IDLE")
    ttk.Label(
        skill_outer,
        textvariable=control_mode_var,
        font=("Arial", 10, "bold"),
    ).pack(anchor="w", pady=(0, 8))

    top_skill_row = ttk.Frame(skill_outer)
    top_skill_row.pack(fill="x", pady=(0, 8))

    ttk.Label(top_skill_row, text="Object:").pack(side="left")
    skill_object_var = tk.StringVar(value="egg_1")
    ttk.Combobox(
        top_skill_row,
        textvariable=skill_object_var,
        values=OBJECTS,
        state="readonly",
        width=22,
    ).pack(side="left", padx=(5, 12))

    # V2-22 V71 HYBRID GPU PLANNER A/B
    planner_row = ttk.Frame(skill_outer)
    planner_row.pack(fill="x", pady=(0, 8))
    ttk.Label(planner_row, text="Motion Planner:").pack(side="left")
    planner_mode_var = tk.StringVar(
        value=(skill_core.get_planner_mode() if skill_core is not None else "GPU Auto")
    )
    planner_combo = ttk.Combobox(
        planner_row,
        textvariable=planner_mode_var,
        values=("GPU Auto", "cuMotion", "cuMotion Strict", "cuRobo", "cuRobo Strict", "BioIK/MoveIt"),
        state="readonly",
        width=20,
    )
    planner_combo.pack(side="left", padx=(5, 8))
    gpu_status_var = tk.StringVar(value="GPU planner: not probed")
    ttk.Label(planner_row, textvariable=gpu_status_var).pack(side="left", padx=(4, 8))

    def _v222_refresh_gpu_status(log_it=False):
        if skill_core is None:
            gpu_status_var.set("GPU planner: skill core unavailable")
            return
        try:
            st = skill_core.gpu_planner_status(refresh=True)
            engines = st.get("engines", {}) if isinstance(st, dict) else {}
            cm = engines.get("cumotion", {}) if isinstance(engines, dict) else {}
            cr = engines.get("curobo", {}) if isinstance(engines, dict) else {}
            if st.get("ok"):
                txt = (
                    "GPU server ON | "
                    f"cuMotion={'ready' if cm.get('initialized') else ('installed' if cm.get('imported') else 'off')} | "
                    f"cuRobo={'ready' if cr.get('initialized') else ('installed' if cr.get('imported') else 'off')}"
                )
            else:
                txt = "GPU server OFF -> BioIK fallback"
            gpu_status_var.set(txt)
            if log_it:
                append_gui_log(f"[V2-22 GPU STATUS] {st}")
        except Exception as exc:
            gpu_status_var.set("GPU status error -> BioIK fallback")
            if log_it:
                append_gui_log(f"[V2-22 GPU STATUS ERROR] {exc}")

    def _v222_planner_changed(_event=None):
        if skill_core is None:
            return
        try:
            selected = skill_core.set_planner_mode(planner_mode_var.get())
            append_gui_log(f"[V2-22 PLANNER GUI] {selected}")
            _v222_refresh_gpu_status(log_it=False)
        except Exception as exc:
            append_gui_log(f"[V2-22 PLANNER GUI ERROR] {exc}")

    planner_combo.bind("<<ComboboxSelected>>", _v222_planner_changed)
    ttk.Button(
        planner_row, text="Refresh GPU", command=lambda: _v222_refresh_gpu_status(log_it=True)
    ).pack(side="left", padx=3)
    root.after(1200, lambda: _v222_refresh_gpu_status(log_it=False))

    MOTION_METHODS_REQUIRING_ARM_CONTROLLER = {
        "move_above", "approach", "lift", "move_to",
        "bio_move_above", "bio_approach", "bio_lift", "bio_move_above_bin",
        "trac_move_above", "trac_approach", "trac_lift", "trac_move_above_bin",
    }

    def submit_skill(method_name, *args):
        if skill_core is None:
            append_gui_log(f"[UI ERROR] Skill core unavailable: {skill_core_error}")
            return

        # V69: never plan a motion that cannot be executed.
        if method_name in MOTION_METHODS_REQUIRING_ARM_CONTROLLER:
            try:
                arm_ready_now = bool(skill_core.arm_controller_client.server_is_ready())
            except Exception:
                arm_ready_now = False
            if not arm_ready_now:
                append_gui_log(
                    "[UI BLOCKED] Root arm_controller is not ACTIVE: "
                    "/arm_controller/follow_joint_trajectory unavailable. "
                    "Motion was NOT planned or executed. Check launcher/controller status."
                )
                return

        append_gui_log(f"[UI] {method_name}{args}")
        try:
            method = getattr(skill_core, method_name)
            accepted = skill_core.submit(method, *args)
            if not accepted:
                append_gui_log("[UI] Skill request rejected because core is BUSY")
        except Exception as exc:
            append_gui_log(f"[UI ERROR] Could not submit {method_name}: {exc}")

    ttk.Button(
        top_skill_row,
        text="Observe Scene",
        command=lambda: submit_skill("observe_scene"),
    ).pack(side="left", padx=3)

    ttk.Button(
        top_skill_row,
        text="Move Above",
        command=lambda: submit_skill(
            "move_above", skill_object_var.get(), 0.12
        ),
    ).pack(side="left", padx=3)

    ttk.Button(
        top_skill_row,
        text="Approach",
        command=lambda: submit_skill(
            "approach", skill_object_var.get(), None
        ),
    ).pack(side="left", padx=3)

    ttk.Button(
        top_skill_row,
        text="Lift 100 mm",
        command=lambda: submit_skill("lift", 0.10),
    ).pack(side="left", padx=3)


    bin_skill_row = ttk.Frame(skill_outer)
    bin_skill_row.pack(fill="x", pady=(0, 8))
    ttk.Label(bin_skill_row, text="Release bin:").pack(side="left")
    skill_bin_var = tk.StringVar(value="red_bin")
    ttk.Combobox(
        bin_skill_row,
        textvariable=skill_bin_var,
        values=("red_bin", "blue_bin"),
        state="readonly",
        width=14,
    ).pack(side="left", padx=(5, 10))
    ttk.Button(
        bin_skill_row,
        text="Move Above Bin",
        command=lambda: submit_skill(
            "move_above_bin", skill_bin_var.get()
        ),
    ).pack(side="left", padx=3)
    ttk.Label(
        bin_skill_row,
        text="(positions gripper for release; does NOT open gripper)",
    ).pack(side="left", padx=(8, 0))

    # Requested local utility buttons beside Skill Test.
    util_box = ttk.LabelFrame(skill_outer, text="Local Robot Utilities")
    util_box.pack(fill="x", pady=(0, 8))

    ttk.Button(
        util_box,
        text="ZERO ALL",
        command=zero_all,
    ).pack(side="left", padx=6, pady=7)

    ttk.Button(
        util_box,
        text="Gripper OPEN",
        command=one_click_release,
    ).pack(side="left", padx=6, pady=7)

    ttk.Button(
        util_box,
        text="Gripper FAST CLAMP",
        command=one_click_clamp,
    ).pack(side="left", padx=6, pady=7)

    def unified_stop_recover():
        emergency_recover()

    def unified_hard_reset():
        hard_reset_zero()

    ttk.Button(
        util_box,
        text="STOP + RECOVER",
        command=unified_stop_recover,
    ).pack(side="left", padx=6, pady=7)

    ttk.Button(
        util_box,
        text="HARD RESET 0",
        command=unified_hard_reset,
    ).pack(side="left", padx=6, pady=7)

    xyz_box = ttk.LabelFrame(skill_outer, text="Move To World XYZ")
    xyz_box.pack(fill="x", pady=(0, 8))

    skill_x_var = tk.StringVar(value="-0.45")
    skill_y_var = tk.StringVar(value="0.00")
    skill_z_var = tk.StringVar(value="0.25")

    for lab, var in (("X", skill_x_var), ("Y", skill_y_var), ("Z", skill_z_var)):
        ttk.Label(xyz_box, text=lab).pack(side="left", padx=(8, 2), pady=8)
        ttk.Entry(xyz_box, textvariable=var, width=10).pack(side="left")

    def skill_move_to():
        try:
            xyz = (
                float(skill_x_var.get()),
                float(skill_y_var.get()),
                float(skill_z_var.get()),
            )
        except Exception:
            if skill_core is not None:
                skill_core.emit("Invalid XYZ")
            return
        submit_skill("move_to", *xyz)

    ttk.Button(
        xyz_box,
        text="MOVE TO",
        command=skill_move_to,
    ).pack(side="left", padx=12)

    # ========================================================
    # NERO V2-7 SKILL-CONTRACT TASK AGENT TAB
    # High-level agent only.  V69 remains the authoritative motion/safety layer.
    # ========================================================
    llm_outer = ttk.Frame(llm_tab)
    llm_outer.pack(fill="both", expand=True, padx=12, pady=12)
    # V3.43 A2 BEGIN
    append_gui_log(f"[V3.44 SCENE] MuJoCo scene = {_v343_scene_name()} (chosen in the NERO console before start)")
    # V3.43 A2 END

    ttk.Label(
        llm_outer,
        text="NERO SimAgent | Gripper-Aware Skill Contract Agent (simulation)",
        font=("Arial", 18, "bold"),
    ).pack(anchor="w", pady=(0, 5))

    ttk.Label(
        llm_outer,
        text=(
            "The LLM is the high-level task planner and never commands joints directly. "
            "V2-7 adds continuous gripper-aperture awareness and a lightweight manipulation prior: "
            "observe_scene reports actual opening plus approximate object grasp width/recommended range; "
            "the Agent must prepare aperture before approaching an object. V71 hybrid planner is active; BioIK/TRAC safety fallback remains available."
        ),
        wraplength=1260,
        justify="left",
    ).pack(anchor="w", pady=(0, 8))

    llm_status_var = tk.StringVar(value="LLM agent: initializing...")
    ttk.Label(
        llm_outer,
        textvariable=llm_status_var,
        font=("Arial", 10, "bold"),
    ).pack(anchor="w", pady=(0, 8))

    # V3.11 POSE SOURCE SWITCH: camera (D435i RGB-D perception) or JSON ground truth.
    v311_frame = ttk.LabelFrame(llm_outer, text="Object pose source for skills (V3.11)")
    v311_frame.pack(anchor="w", fill="x", pady=(0, 8))
    v311_var = tk.StringVar(value=V311_STATE.get("mode", "camera"))
    v311_info_var = tk.StringVar(value="")

    def _v311_on_switch():
        V311_STATE["mode"] = v311_var.get()
        if V311_STATE["mode"] in ("wrist", "wrist_rgb"):   # V3.26/V3.30: start the wrist camera process on first use
            _cam330 = v330_wrist_profile()
            v330_write_model_flag(_cam330)
            if not v326_start(stop_event):
                append_gui_log("[V3.35 POSE SOURCE ERROR] wrist camera unavailable (see console) -> RUN TASK is refused "
                               "and every skill stops with PERCEPTION_UNAVAILABLE (no fallback to simulator truth)")
            else:
                append_gui_log("[V3.26 POSE SOURCE] skills now use: WRIST camera "
                               + ("USB 130deg RGB-only (InnoMaker U20CAM-1080P, V3.30)" if _cam330 == "usb130"
                                  else "D435i RGB-D") + " on the gripper | the arm moves to look poses before each "
                               "capture | takes effect at the next tool call")
            _v311_refresh_info()
            return
        v330_write_model_flag(os.environ.get("NERO_V330_SHOW_WRIST_MODEL", "none"))   # V3.33: not a wrist source -> no wrist camera
        eff = v311_mode()
        if V311_STATE["mode"] == "camera" and not v311_camera_available():
            append_gui_log("[V3.35 POSE SOURCE] fixed camera selected but its process is not running (yet) - RUN TASK "
                           "is refused until it runs (no fallback to simulator truth; pick 'JSON ground truth' explicitly)")
        else:
            append_gui_log(f"[V3.11 POSE SOURCE] skills now use: "
                           f"{'camera D435i perception (RGB-D)' if eff == 'camera' else 'JSON ground truth (MuJoCo)'}"
                           " | takes effect at the next tool call")
        _v311_refresh_info()

    def _v311_refresh_info():
        try:
            eff = v311_mode()
            cam_ok = v311_camera_available()
            last = V311_STATE.get("last") or {}
            if V311_STATE.get("mode") in ("wrist", "wrist_rgb"):
                txt = (f"wrist {v330_wrist_profile()} camera process: {'running' if v326_available() else 'starting / unavailable'}"
                       + (f" | last capture #{last.get('capture')} ({last.get('tag')})" if last else ""))
            elif eff == "camera":
                txt = (f"camera process: {'running' if cam_ok else 'starting / unavailable'}"
                       + (f" | last capture #{last.get('capture')} ({last.get('tag')})" if last else ""))
            else:
                txt = "skills read MuJoCo ground truth; the camera (if on) still records perception errors"
            v311_info_var.set(txt)
        except Exception:
            pass

    def _v311_tick():
        _v311_refresh_info()
        try:
            root.after(2000, _v311_tick)
        except Exception:
            pass

    # V3.30: drop-down instead of radio buttons
    _V330_SOURCES = [("camera", "Fixed stand camera - D435i RGB-D (V3.08-V3.11, default)"),
                     ("wrist", "Wrist camera - D435i RGB-D on the gripper (V3.26-V3.29)"),
                     ("wrist_rgb", "Wrist camera - USB 130deg RGB only, InnoMaker U20CAM-1080P (V3.30)"),
                     ("truth", "JSON ground truth (MuJoCo)")]
    _v330_label_of = dict(_V330_SOURCES)
    _v330_value_of = {lbl: val for val, lbl in _V330_SOURCES}
    v330_combo_var = tk.StringVar(value=_v330_label_of.get(v311_var.get(), _V330_SOURCES[0][1]))

    def _v330_on_combo(_evt=None):
        v311_var.set(_v330_value_of.get(v330_combo_var.get(), "camera"))
        _v311_on_switch()

    ttk.Label(v311_frame, text="Pose source:").pack(side="left", padx=(8, 4), pady=4)
    v330_combo = ttk.Combobox(v311_frame, textvariable=v330_combo_var, state="readonly", width=64,
                              values=[lbl for _v, lbl in _V330_SOURCES])
    v330_combo.pack(side="left", padx=(0, 14), pady=4)
    v330_combo.bind("<<ComboboxSelected>>", _v330_on_combo)

    # V3.35 CAMERA CALIBRATION ERROR (continuous): 0 = perfect calibration; perception then uses a believed camera
    # model whose intrinsics / lens distortion / hand-eye pose differ from the real (rendered) camera.
    try:
        import NERO_V326_Wrist_Camera as _w335g
        _calib_max335 = float(os.environ.get("NERO_V335_CALIB_MAX", str(_w335g.V335_CALIB_MAX)))
    except Exception:
        _w335g, _calib_max335 = None, 3.0
    v335_calib_var = tk.DoubleVar(value=float(V335_STATE.get("calib_level", 0.3)))
    v335_calib_txt = tk.StringVar(value="")

    def _v335_on_calib(_v=None):
        lvl = round(float(v335_calib_var.get()), 2)
        V335_STATE["calib_level"] = lvl
        try:
            v335_calib_txt.set(f"{lvl:.2f}  ({_w335g.calib_summary(_w335g.calib_error(lvl))})" if _w335g else f"{lvl:.2f}")
        except Exception:
            v335_calib_txt.set(f"{lvl:.2f}")

    ttk.Label(v311_frame, text="Camera calib error:").pack(side="left", padx=(4, 2), pady=4)
    ttk.Scale(v311_frame, from_=0.0, to=_calib_max335, orient="horizontal", length=150, variable=v335_calib_var,
              command=_v335_on_calib).pack(side="left", padx=(0, 4), pady=4)
    ttk.Label(v311_frame, text=f"(0-{_calib_max335:g})").pack(side="left", padx=(0, 6))
    _v335_on_calib()
    ttk.Label(v311_frame, textvariable=v311_info_var).pack(side="left", padx=(6, 8))
    v335_calib_row = ttk.Frame(llm_outer)
    v335_calib_row.pack(anchor="w", fill="x", pady=(0, 6))
    ttk.Label(v335_calib_row, text="  believed vs real camera:").pack(side="left")
    ttk.Label(v335_calib_row, textvariable=v335_calib_txt).pack(side="left", padx=(4, 0))
    root.after(1500, _v311_tick)
    if V311_STATE.get("mode") in ("wrist", "wrist_rgb"):   # V3.26/V3.30: NERO_V311_POSE_SOURCE=wrist|wrist_rgb
        root.after(1200, _v311_on_switch)

    llm_agent = None
    llm_agent_error = ""
    try:
        from NERO_V2_7_1_Agent import NeroLLMAgent, DEFAULT_MODEL, TOOLS as LLM_TOOLS
        try:
            from NERO_V2_7_1_Agent import LLM_PROVIDERS, provider_for_model
        except Exception:
            LLM_PROVIDERS, provider_for_model = {}, (lambda m, e=None: "openai")
    except Exception as exc:
        DEFAULT_MODEL = os.environ.get("NERO_LLM_MODEL", "gpt-5.6-luna")
        LLM_TOOLS = []
        LLM_PROVIDERS, provider_for_model = {}, (lambda m, e=None: "openai")
        llm_agent_error = str(exc)
        append_gui_log(f"[LLM INIT ERROR] {exc}")

    # ----- Exact MuJoCo state exposed to LLM -----
    # Egg Task Semantic Integration Layer.  V69 remains unchanged; this layer
    # grounds MuJoCo bodies into task concepts used by the LLM/Skill Contracts.
    EGG_OBJECTS = ["egg_1", "egg_2", "egg_3", "egg_4"]
    EGG_BASKET = {"name": "egg_basket", "center_xy_m": [-0.425, -0.155], "radius_m": 0.088}
    EGG_HOLDER = {"name": "egg_holder", "center_xyz_m": [-0.420, 0.155, 0.006]}
    EGG_SLOT_POSES = {
        "slot_1": (-0.510, 0.123, 0.043),
        "slot_2": (-0.450, 0.123, 0.043),
        "slot_3": (-0.390, 0.123, 0.043),
        "slot_4": (-0.330, 0.123, 0.043),
        "slot_5": (-0.510, 0.187, 0.043),
        "slot_6": (-0.450, 0.187, 0.043),
        "slot_7": (-0.390, 0.187, 0.043),
        "slot_8": (-0.330, 0.187, 0.043),
    }
    EGG_SLOT_NAMES = list(EGG_SLOT_POSES)

    def _egg_slot_for_xyz(xyz):
        x, y, z = [float(v) for v in xyz]
        if z > 0.095:
            return None
        if V329_LAYOUT_STATE.get("perceived_required") and not V329_LAYOUT_STATE.get("holder_ok"):
            return None                       # V3.31: holder not perceived yet -> no slot can be named
        best = None
        best_d = 1e9
        for slot_name, (sx, sy, _sz) in EGG_SLOT_POSES.items():
            d = math.hypot(x - sx, y - sy)
            if d < best_d:
                best, best_d = slot_name, d
        return best if best_d <= 0.026 else None

    def llm_classify_object_location(xyz):
        x, y, z = [float(v) for v in xyz]
        slot = _egg_slot_for_xyz(xyz)
        if slot is not None:
            return f"egg_holder.{slot}"
        if V329_LAYOUT_STATE.get("perceived_required") and not V329_LAYOUT_STATE.get("pile_ok"):
            return "table_or_transit"         # V3.31: egg pile not perceived yet
        bx, by = EGG_BASKET["center_xy_m"]
        if math.hypot(x - bx, y - by) <= float(EGG_BASKET["radius_m"]) and z <= 0.13:
            return "egg_basket"
        return "table_or_transit"

    # --------------------------------------------------------
    # Egg object geometry + first manipulation prior
    # --------------------------------------------------------
    # Eggs are rigid ellipsoids in this stage, matching RoboDojo's original
    # fill_egg_holder physical assumption.  Fragility/tactile damage semantics
    # are intentionally deferred; only geometry/category/receptacle grounding is
    # added here.  Widths reflect the four slightly different scene eggs.
    V24_OBJECT_GEOMETRY = {
        "egg_1": {"category": "egg", "shape": "ellipsoid", "material_class": "rigid_shell_baseline", "fragile_model_enabled": False, "approx_size_mm": [44.0, 42.4, 62.0], "effective_width_mm": 44.0},
        "egg_2": {"category": "egg", "shape": "ellipsoid", "material_class": "rigid_shell_baseline", "fragile_model_enabled": False, "approx_size_mm": [45.6, 43.4, 64.0], "effective_width_mm": 45.6},
        "egg_3": {"category": "egg", "shape": "ellipsoid", "material_class": "rigid_shell_baseline", "fragile_model_enabled": False, "approx_size_mm": [43.4, 41.8, 61.0], "effective_width_mm": 43.4},
        "egg_4": {"category": "egg", "shape": "ellipsoid", "material_class": "rigid_shell_baseline", "fragile_model_enabled": False, "approx_size_mm": [46.2, 43.8, 64.8], "effective_width_mm": 46.2},
    }

    # V2-7 anti-poke adaptation memory.  This is intentionally session-local:
    # if a pre-grasp approach visibly disturbs an object, the manipulation prior
    # widens for that object on the next observation/retry instead of blindly
    # repeating the same aperture.
    V241_APERTURE_RETRY_BONUS_MM = {name: 0.0 for name in V24_OBJECT_GEOMETRY}
    # V2-7: one non-safety planning adaptation per object.  This is separate
    # from the physical safety-latch retry budget.
    V251_PLANNING_RETRY_USED = {name: 0 for name in V24_OBJECT_GEOMETRY}
    V251_GRASP_Z_RETRY_BONUS_M = {name: 0.0 for name in V24_OBJECT_GEOMETRY}

    # V2-7 reflective safety-retry memory.  A pre-grasp safety trip is not
    # automatically ignored or automatically unlatched.  Instead it becomes a
    # structured failure context for the Agent.  The Agent may explicitly
    # choose ONE recovery/retry per object.  This preserves V69's latched-safety
    # behavior while allowing reasoned recovery instead of immediate task abort.
    V242_SAFETY_RETRY_USED = {name: 0 for name in V24_OBJECT_GEOMETRY}
    V242_LAST_SAFETY_CONTEXT = {
        "active": False,
        "tool": None,
        "object_name": None,
        "trip_count": 0,
        "safety_state_code": 0,
        "likely_cause": None,
        "evidence": [],
        "suggested_strategy": None,
        "suggested_retry_opening_mm": None,
        "retry_allowed": False,
    }

    def _v242_set_safety_context(**kwargs):
        V242_LAST_SAFETY_CONTEXT.update({
            "active": False,
            "tool": None,
            "object_name": None,
            "trip_count": int(safety_trip_count.value),
            "safety_state_code": int(safety_state.value),
            "likely_cause": None,
            "evidence": [],
            "suggested_strategy": None,
            "suggested_retry_opening_mm": None,
            "retry_allowed": False,
        })
        V242_LAST_SAFETY_CONTEXT.update(kwargs)
        return dict(V242_LAST_SAFETY_CONTEXT)

    def _v242_clear_safety_context():
        return _v242_set_safety_context(active=False)

    # ------------------------------------------------------------------
    # Manipulation prior
    # ------------------------------------------------------------------
    # These margins encode simple, inspectable engineering experience: the
    # gripper should approach with meaningful lateral clearance rather than an
    # aperture that is merely equal to the nominal object width.  The prior is
    # intentionally expressed in millimeters so future experiments can replace
    # the constants with measured success statistics or a learned policy.
    #
    # IMPORTANT: this is NOT collision checking.  MoveIt still performs its own
    # geometric feasibility test; the prior only proposes a more robust starting
    # aperture for that check.
    def _v24_gripper_prior(object_name, actual_opening_mm=None):
        geom = dict(V24_OBJECT_GEOMETRY.get(str(object_name), {}))
        width = float(geom.get("effective_width_mm", 50.0))
        shape = str(geom.get("shape", "unknown"))

        # V2-7 shape-aware embodied manipulation prior.
        #
        # The V2-7 run showed that a 52 mm ball with a nominal 60 mm opening
        # still moved ~13 mm during the straight approach.  The joint opening
        # is therefore not equivalent to useful free clearance at the rubber
        # pad/stud collision envelope.  Spheres get a deliberately larger
        # pre-grasp aperture than box-like objects.
        #
        # These are empirical priors, not physical laws.  They are logged and
        # can later be replaced by a learned success model.
        if shape in ("sphere", "ellipsoid"):
            base_min_margin = 16.0      # 52 mm ball -> >=68 mm opening
            base_target_margin = 20.0   # preferred 72 mm
            base_max_margin = 28.0      # avoid needless full-open unless useful
            envelope_allowance = 8.0   # approximate pad/stud collision envelope
            policy = "round/ellipsoid anti-poke prior: +16/+20/+28 mm total aperture margin"
        else:
            # V2-7: 53 mm on a 45 mm cube left only ~2 mm useful free
            # clearance per side after the pad envelope.  Use a more human-like
            # manipulation prior: clearly wider than the object, but not full-open.
            base_min_margin = 13.0      # 45 mm cube -> >=58 mm opening
            base_target_margin = 15.0   # preferred 60 mm
            base_max_margin = 21.0      # robust range ends at 66 mm
            envelope_allowance = 4.0
            policy = "box anti-poke prior: +13/+15/+21 mm total aperture margin"

        retry_bonus = float(V241_APERTURE_RETRY_BONUS_MM.get(str(object_name), 0.0))
        rec_min = min(100.0, width + base_min_margin + retry_bonus)
        rec_target = min(100.0, width + base_target_margin + retry_bonus)
        rec_max = min(100.0, width + base_max_margin + retry_bonus)
        result = {
            "category": geom.get("category", "unknown"),
            "shape": shape,
            "material_class": geom.get("material_class"),
            "fragile_model_enabled": bool(geom.get("fragile_model_enabled", False)),
            "approx_size_mm": geom.get("approx_size_mm", [width, width, width]),
            "effective_width_mm": round(width, 2),
            "recommended_opening_range_mm": [round(rec_min, 2), round(rec_max, 2)],
            "recommended_target_opening_mm": round(rec_target, 2),
            "recommended_total_clearance_mm": round(rec_target - width, 2),
            "estimated_pad_envelope_allowance_total_mm": round(envelope_allowance, 2),
            "estimated_free_clearance_at_target_mm": round(max(0.0, rec_target - width - envelope_allowance), 2),
            "adaptive_retry_bonus_mm": round(retry_bonus, 2),
            "prior_basis": policy + "; +8 mm after detected poke; +15 mm after a pre-grasp safety-stop retry",
        }
        if actual_opening_mm is not None:
            opening = float(actual_opening_mm)
            result["current_opening_mm"] = round(opening, 2)
            result["geometric_clearance_total_mm"] = round(opening - width, 2)
            result["estimated_free_clearance_total_mm"] = round(opening - width - envelope_allowance, 2)
            result["estimated_free_clearance_per_side_mm"] = round((opening - width - envelope_allowance) / 2.0, 2)
            result["distance_to_recommended_target_mm"] = round(opening - rec_target, 2)
        return result

    # V2-18 POST-APPROACH REOBSERVE — egg aperture prior only.
    # User-requested experiment: pre-open to about 120% of the clamped-axis
    # width instead of the older large fixed clearance.  Keep a small robust
    # band around that target, but do not deliberately widen the egg aperture.
    _v218_base_gripper_prior = _v24_gripper_prior

    def _v24_gripper_prior(object_name, actual_opening_mm=None):
        result = _v218_base_gripper_prior(object_name, actual_opening_mm)
        if str(object_name).startswith("egg"):
            # V3.04 EGG APERTURE: width + fixed margin instead of 1.20 x width.
            # V3.03 log (2026-10-21 15:43-16:33): with 1.20x the pads had only
            # ~0.8 mm total free clearance; every approach pushed the egg
            # 10-14 mm and the fingers were spread from ~53 to 76-87 mm, i.e.
            # they landed ON the egg.  egg_3 ended 15.3/16.0 mm off-centre.
            width = float(result.get("effective_width_mm", 50.0))
            bonus = float(result.get("adaptive_retry_bonus_mm", 0.0))
            m_t = float(os.environ.get("NERO_V304_EGG_APERTURE_MARGIN_MM", "16"))
            rec_min = min(100.0, width + max(8.0, m_t - 4.0) + bonus)
            rec_target = min(100.0, width + m_t + bonus)
            rec_max = min(100.0, width + m_t + 6.0 + bonus)
            envelope = float(result.get("estimated_pad_envelope_allowance_total_mm", 0.0))
            result.update({
                "recommended_opening_range_mm": [round(rec_min, 2), round(rec_max, 2)],
                "recommended_target_opening_mm": round(rec_target, 2),
                "recommended_total_clearance_mm": round(rec_target-width, 2),
                "estimated_free_clearance_at_target_mm": round(max(0.0, rec_target-width-envelope), 2),
                "aperture_ratio_target": round(rec_target / max(width, 1e-6), 3),
                "prior_basis": f"V3.04 egg aperture prior: target = width + {m_t:.0f} mm (about {m_t/2:.0f} mm per side before pads); band = width+{max(8.0, m_t-4.0):.0f}..+{m_t+6.0:.0f} mm",
            })
            if actual_opening_mm is not None:
                opening = float(actual_opening_mm)
                result["distance_to_recommended_target_mm"] = round(opening-rec_target, 2)
        return result

    # ============================================================
    # V3.11 POSE SOURCE: camera perception (tracked) or JSON ground truth
    # ------------------------------------------------------------
    # Camera mode: observe_scene / plan_grasp (target + neighbours) / verify_object_in_slot
    # read the LAST synchronous capture.  In-hand, contact and motion-monitoring checks
    # (lift follow, bystander nudge, held-egg axis while carrying) still read the simulator:
    # on the real robot they become TCP x grasp transform + tactile.  Truth is logged for
    # comparison only and is never returned to the model in camera mode.
    # ============================================================
    V311_MONITOR_BASELINE = {}

    def _v311_obj_entry(name):
        last = V311_STATE.get("last") or {}
        return (last.get("objects") or {}).get(str(name))

    def _v311_obj_pose(name, require_trusted=True):
        o = _v311_obj_entry(name)
        if o is None:
            raise RuntimeError(f"PERCEPTION_TARGET_NOT_SEEN: {name} has never been perceived")
        if require_trusted and not (o.get("seen") and o.get("trusted")):
            raise RuntimeError(f"PERCEPTION_TARGET_UNTRUSTED: {name} seen={o.get('seen')} trusted={o.get('trusted')} "
                               f"confidence={o.get('confidence')} coverage={o.get('coverage')}")
        return [float(v) for v in o["xyz_m"]], [float(v) for v in o["quat_xyzw"]]

    def _v311_perception_tag(o):
        if not o:
            return {"source": "camera", "seen": False, "trusted": False}
        tag335 = {"source": "camera", "seen": bool(o.get("seen")), "trusted": bool(o.get("trusted")),
                  "confidence": o.get("confidence")}
        if o.get("measurement"):
            tag335["measurement"] = o["measurement"]           # V3.35: triangulated position + measured size
        if o.get("untrusted_reason"):
            tag335["untrusted_reason"] = o["untrusted_reason"]
        return tag335

    def _v311_log_vs_truth(label, names):
        """Log perceived vs simulator pose (comparison only; never returned to the model)."""
        parts = []
        for n in names:
            o = _v311_obj_entry(n)
            try:
                if _v329_no_prior() and o is not None:
                    _tn330, txyz = _v330_nearest_truth(o["xyz_m"])     # V3.30/V3.31: names are perception's own
                else:
                    txyz, _ = _v329_truth_pose(n)
            except Exception:
                txyz = None
            if o is None or txyz is None:
                parts.append(f"{n}=not_seen")
                continue
            err = math.dist([float(v) for v in o["xyz_m"]], [float(v) for v in txyz]) * 1000.0
            parts.append(f"{n} {'T' if o.get('trusted') else ('u' if o.get('seen') else 'stale')} err={err:.1f}mm")
        last = V311_STATE.get("last") or {}
        skill_core.emit(f"[V3.11 PERCEPTION] {label} | capture #{last.get('capture')} wait={last.get('wait_s')}s | "
                        + " | ".join(parts))

    def _v311_capture(label, names=None):
        if v326_wrist_selected():                  # V3.26: wrist camera
            return _v326_wrist_capture(label, names)
        if _v329_no_prior():
            # V3.31 FIXED CAMERA WITHOUT PRIORS: same scene ingestion as the wrist camera (holder, pile, generic
            # names, no simulator truth in the checks); the whole table is in view -> no scan motion
            _v329_install_no_truth()
            if not _V329_SCENE.get("active"):
                _v329_reset_scene("first fixed-camera capture")
            payload = v311_capture_sync(_v328_tag_with_hints(label))
            _v329_ingest(payload)
            _v329_apply()
            h331 = _V329_SCENE.get("holder") or {}
            skill_core.emit(f"[V3.31 FIXED VIEW] {label} | eggs seen "
                            f"{sorted(n for n, o in (payload.get('objects') or {}).items() if o.get('seen'))} | holder "
                            f"{'complete' if h331.get('complete') else ('partial' if h331 else 'not perceived')} | named "
                            f"{(payload.get('tracking') or {}).get('named')}")
            _v311_log_vs_truth(label, names or EGG_OBJECTS)
            return payload
        payload = v311_capture_sync(label)
        _v311_log_vs_truth(label, names or EGG_OBJECTS)
        return payload

    # ------------------------------------------------------------
    # V3.26 WRIST CAMERA SURVEY: move the arm to look poses (gripper down, TCP = look point + 0.29 m), wait until
    # settled, capture from the wrist D435i, merge the views (latest sighting per egg). Holding an egg -> no motion,
    # one capture from the current pose. A failed look move -> capture from where the arm is (logged).
    # ------------------------------------------------------------
    # V3.27 WRIST LOOK FAMILY: V3.26 sent ONE look pose (TCP 0.29 m above the point, yaw 0) to cuRobo, which
    # returned None 3x for basket and holder -> the arm never left q0 and every capture looked at the ceiling.
    # Now: a family of equivalent views (camera distance x tilt x roll about the vertical, see
    # NERO_V326_Wrist_Camera.look_pose_candidates), screened with collision-aware BioIK, ranked by joint travel,
    # then a MoveIt joint-goal plan (collision-checked, like the V2-39 q0 move); cuRobo pose plan as last resort.
    _V327_LOOK = {"at": None}

    def _v327_ik_solutions(cands, live, qnow, max_tries, want):
        sols, tried = [], 0
        for c in cands:
            if tried >= max_tries or len(sols) >= want:
                break
            tried += 1
            try:
                j, _e = skill_core._compute_ik(c["xyz"], c["quat_xyzw"],
                                               timeout_s=float(os.environ.get("NERO_V327_LOOK_IK_TIMEOUT_S", "0.08")),
                                               avoid_collisions=True, seed_joint_state=live)
            except Exception:
                j = None
            if j is None:
                continue
            jl = [float(v) for v in j]
            travel = max(abs(jl[i] - float(qnow[i])) for i in range(7))
            margin = min(min(jl[i] - _V327_JLIM[i][0], _V327_JLIM[i][1] - jl[i]) for i in range(7))
            if margin < math.radians(3.0):
                continue
            sols.append((travel - 0.5 * min(margin, math.radians(20.0)), jl, c))
        sols.sort(key=lambda s_: s_[0])
        return sols, tried

    _V327_JLIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14),
                  (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708)]

    def _v326_move_look(pt, label, distances=None, no_detour=False):
        import NERO_V326_Wrist_Camera as _v326w
        snap, _nums = _v307_posture_numbers()
        at = _V327_LOOK.get("at")
        at_look = at is not None and _v24_xyz_distance(snap["xyz"], at) <= 0.03
        if (snap["near_basket"] or snap["near_holder"] or float(snap["xyz"][2]) < _V239_LOW_Z_M) and not at_look \
                and not no_detour:
            try:
                _v307_go_q0(f"V3.26 WRIST LOOK {label}")
            except Exception as exc328q:
                # V3.28 LOW-POSE ESCAPE: no straight retreat and no q0 path -> plan the look pose directly from here
                # (MoveIt joint goal, collision-checked) instead of failing every later capture.
                skill_core.emit(f"[V3.28 WRIST LOOK DIRECT] {label} | q0 detour failed ({str(exc328q)[:160]}); "
                                "planning the look pose directly from the current pose")
        _V327_LOOK["at"] = None
        cands = [c for c in _v326w.look_pose_candidates(pt, cfg=_v326w.wrist_config(v330_wrist_profile()),
                                                         distances=distances) if _v326w.look_pose_ok(c)]
        live = skill_core._current_joint_state_copy()
        qnow = [float(v) for v in skill_core._arm_positions_from_joint_state(live)]
        sols, tried = _v327_ik_solutions(cands, live, qnow, int(os.environ.get("NERO_V327_LOOK_MAX_IK", "40")),
                                         int(os.environ.get("NERO_V327_LOOK_KEEP", "4")))
        skill_core.emit(f"[V3.27 WRIST LOOK SCREEN] {label} | point [{pt[0]:+.3f},{pt[1]:+.3f},{pt[2]:+.3f}] | "
                        f"{len(cands)} candidate views, BioIK tried {tried}, solutions {len(sols)} "
                        f"{[s_[2]['tag'] for s_ in sols]}")
        errs = []
        for _score, joints, c in sols[:3]:
            traj = None
            try:
                traj = skill_core._plan_joint_solution(joints, velocity_scale=0.20, accel_scale=0.15,
                                                       planning_attempts=6, planning_time=5.0)
                how = "MoveIt joint goal"
            except Exception as exc_j:
                errs.append(f"{c['tag']}: joint plan {exc_j}")
                try:
                    traj = skill_core._plan_pose(c["xyz"], c["quat_xyzw"], 0.20, 0.15)
                    how = "cuRobo pose"
                except Exception as exc_p:
                    errs.append(f"{c['tag']}: pose plan {str(exc_p)[:120]}")
                    continue
            safety0 = int(safety_trip_count.value)
            if traj is not None:
                _v218_slow_trajectory_in_place(traj, factor=1.45)
                skill_core._execute_trajectory(traj)
            if int(safety_trip_count.value) != safety0:
                raise RuntimeError("V3.26 wrist look move triggered safety")
            _v310_wait_tcp_settled(f"V3.26 wrist look {label}")
            xyz_now, _q_now = skill_core.current_tcp_pose()
            perr = _v24_xyz_distance(xyz_now, c["xyz"])
            if perr > 0.03:
                errs.append(f"{c['tag']}: reached {perr * 1000:.0f} mm off")
                continue
            _V327_LOOK["at"] = [float(v) for v in xyz_now]
            skill_core.emit(f"[V3.27 WRIST LOOK OK] {label} | view {c['tag']} via {how} | TCP {c['xyz']} | "
                            f"camera {c['cam_xyz']} -> {c['dist_m']:.2f} m to the point | error {perr * 1000:.0f} mm")
            return c
        raise RuntimeError("WRIST_LOOK_UNREACHABLE: "
                           + (f"no collision-free IK in {tried} candidate views" if not sols else "; ".join(errs)[:400]))

    # V3.28 VIEW-AWARE TRACKING HINTS: the wrist tracker gets the egg in the gripper (held) and the eggs this
    # episode released into slots (placed, at the commanded slot centre + NERO_V328_PLACED_DZ_M) with every trigger.
    def _v328_tag_with_hints(tag, live=False):
        import NERO_V326_Wrist_Camera as _v326w
        held = None
        try:
            held = _v3121_held_name()
        except Exception:
            held = None
        placed = {}
        for obj, rel in dict(V271_PLACEMENT_RUNTIME.get("released") or {}).items():
            c = EGG_SLOT_POSES.get(str((rel or {}).get("slot_name")))
            if c is not None and obj != held:
                placed[str(obj)] = [float(c[0]), float(c[1]), _v331_egg_in_slot_z(c)]
        heldp = None
        if held:
            try:                                            # V3.35: in-hand model position for the camera check
                tcp335, q335 = skill_core.current_tcp_pose()
                e335 = _v3121_held_pose_estimate(held, tcp335, q335)
                heldp = e335[0] if e335 is not None else None
            except Exception:
                heldp = None
        lost337 = sorted(str(n) for n in (_V329_EST.get("lost") or {}) if str(n) != str(held))
        return _v326w.encode_hints(tag, held=held, placed=placed, cam=v330_wrist_profile(),
                                   calib=V335_STATE.get("calib_level"), heldp=heldp, live=live, lost=lost337)

    # ------------------------------------------------------------
    # V3.35 LIVE PERCEPTION: the wrist camera keeps refreshing the scene in the background (NERO_V335_LIVE_HZ); before
    # every motion skill that relies on a perceived target a fresh capture is taken and the target compared with the
    # plan (egg moved -> re-plan the grasp, carton moved -> re-plan the placement, egg shifted in the fingers ->
    # warning / stop). Live payloads are queued by the capture thread and folded in by the skill thread (no races).
    # ------------------------------------------------------------
    _V335_LOCKS = {}

    def _v335_cap_lock():
        if "cap" not in _V335_LOCKS:
            import threading as _th335
            _V335_LOCKS["cap"] = _th335.Lock()
        return _V335_LOCKS["cap"]

    def _v335_wait_camera():
        """Wrist camera selected -> its process must run (V3.35: never fall back to simulator truth)."""
        if not v326_wrist_selected():
            return
        t335 = time.time()
        wait335 = float(os.environ.get("NERO_V335_CAMERA_WAIT_S", "20"))
        while not v326_available():
            if time.time() - t335 > wait335:
                raise RuntimeError("PERCEPTION_UNAVAILABLE: the wrist camera process is not running - task stopped "
                                   "(V3.35: no fallback to simulator truth)")
            time.sleep(0.2)

    def _v335_note_payload(p):
        if not p:
            return
        ih = p.get("in_hand")
        if ih is not None:
            V335_STATE["in_hand"] = dict(ih, t=time.time(), capture=p.get("capture"))
            if ih.get("visible") and float(ih.get("shift_mm") or 0.0) > float(os.environ.get("NERO_V335_INHAND_WARN_MM", "20")):
                V335_STATE["in_hand_bad"] = int(V335_STATE.get("in_hand_bad", 0)) + 1
            elif ih.get("visible"):
                V335_STATE["in_hand_bad"] = 0
        V335_STATE["last_calib"] = p.get("calib")
        V335_STATE["last_measured"] = p.get("measured")

    def _v335_capture(tag):
        """One wrist capture (skill or check): camera must be running; serialised with the live refresh."""
        import NERO_V326_Wrist_Camera as _v326w
        _v335_wait_camera()
        with _v335_cap_lock():
            p = _v326w.capture_sync(V326_SHARED, _v328_tag_with_hints(tag))
        _v335_note_payload(p)
        return p

    def _v335_ingest_live(p):
        """Fold one live / check capture into the scene: holder (filtered), names seen, pile, merged objects."""
        if not p or p.get("error"):
            return
        _v335_note_payload(p)
        if _v329_no_prior():
            _v329_ingest(p)
        last = V311_STATE.get("last")
        if not last:
            V311_STATE["last"] = p
            return
        objs = dict(last.get("objects") or {})
        for n, o in (p.get("objects") or {}).items():
            if o.get("seen") or n not in objs:
                objs[n] = dict(o, view=p.get("tag"))
        new = dict(last)
        new["objects"] = objs
        new["capture"] = max(int(last.get("capture") or 0), int(p.get("capture") or 0))
        new["live_tag"] = p.get("tag")
        V311_STATE["last"] = new
        if _v329_no_prior():
            _v329_apply()

    def _v335_drain():
        q335 = V335_STATE.get("pending") or []
        V335_STATE["pending"] = []
        for p in q335:
            try:
                _v335_ingest_live(p)
            except Exception as exc335:
                skill_core.emit(f"[V3.35 LIVE INGEST WARN] {exc335}")

    def _v335_live_loop():
        import NERO_V326_Wrist_Camera as _v326w
        while not stop_event.is_set():
            period = 1.0 / max(0.1, float(V335_STATE.get("live_hz") or 1.0))
            try:
                if V335_STATE.get("live_on") and v326_wrist_selected() and v326_available():
                    lk335 = _v335_cap_lock()
                    if lk335.acquire(blocking=False):
                        try:
                            p = _v326w.capture_sync(V326_SHARED, _v328_tag_with_hints("live", live=True), timeout_s=10.0)
                        finally:
                            lk335.release()
                        V335_STATE["live_count"] = int(V335_STATE.get("live_count", 0)) + 1
                        q335 = list(V335_STATE.get("pending") or [])[-2:]
                        q335.append(p)
                        V335_STATE["pending"] = q335
            except Exception as exc335:
                if time.time() - float(V335_STATE.get("live_err_t", 0.0)) > 30.0:
                    V335_STATE["live_err_t"] = time.time()
                    try:
                        skill_core.emit(f"[V3.35 LIVE WARN] {exc335}")
                    except Exception:
                        pass
            time.sleep(period)

    def _v335_holder_filter(h, tag=None, cam_pos=None):
        """V3.36 carton fusion with graded views:
        * SKILL views (observe_scene / pre-place look ...) that agree with the fused pose refine it;
          LIVE / precheck views never change the slots, they can only raise "possibly moved" (holder_suspect)
        * a view far from the fused pose (> max(15 mm, 3 x jitter)) is a JUMP; the carton counts as MOVED only when
          two consistent jumps come from camera positions >= 5 cm apart and at least one of them is a skill view
        * a skill view that agrees again clears the suspicion"""
        t = str(tag or "")
        skill_view = not t.startswith(("live", "precheck"))
        hist = list(V335_STATE.get("holder_hist") or [])
        if not hist or str(hist[-1].get("top_z_source")) != str(h.get("top_z_source")):
            if not skill_view and hist:
                return None                            # a live view never (re)starts the carton estimate
            V335_STATE["holder_hist"] = [dict(h, _cam=cam_pos, _skill=skill_view)]   # first view / height measured
            V335_STATE["holder_jump"] = []
            V335_STATE["holder_suspect"] = None
            return h

        def fuse(lst):
            n = len(lst)
            cx = sum(x["center_xy_m"][0] for x in lst) / n
            cy = sum(x["center_xy_m"][1] for x in lst) / n
            sy = sum(math.sin(float(x["yaw_rad"])) for x in lst); cyw = sum(math.cos(float(x["yaw_rad"])) for x in lst)
            out = {k: v for k, v in lst[-1].items() if not str(k).startswith("_")}
            out["center_xy_m"] = [round(cx, 5), round(cy, 5)]
            out["yaw_rad"] = round(math.atan2(sy, cyw), 5)
            slots = {}
            for k in (lst[-1].get("slots") or {}):
                pts = [x["slots"][k]["xyz_m"] for x in lst if k in (x.get("slots") or {})]
                slots[k] = dict(lst[-1]["slots"][k], xyz_m=[round(sum(p[i] for p in pts) / len(pts), 5) for i in range(3)])
            out["slots"] = slots
            out["fused_views"] = n
            return out
        fused = fuse(hist)
        jit = 0.0
        if len(hist) >= 2:
            jit = math.sqrt(sum(math.dist(x["center_xy_m"], fused["center_xy_m"]) ** 2 for x in hist) / len(hist))
        thr = max(float(os.environ.get("NERO_V335_HOLDER_MOVED_MM", "15")) / 1000.0, 3.0 * jit)
        d = math.dist(h["center_xy_m"], fused["center_xy_m"])
        if d <= thr:
            if skill_view:
                hist.append(dict(h, _cam=cam_pos, _skill=True))
                V335_STATE["holder_hist"] = hist[-int(os.environ.get("NERO_V335_HOLDER_FUSE", "5")):]
                V335_STATE["holder_jump"] = []
                if V335_STATE.get("holder_suspect"):
                    skill_core.emit(f"[V3.36 HOLDER OK] {t} | the carton is where it was - suspicion cleared")
                V335_STATE["holder_suspect"] = None
                return fuse(V335_STATE["holder_hist"])
            return fused
        if not skill_view and os.environ.get("NERO_V341_HOLDER_VIEW_GATE", "1") != "0":
            # V3.41: a live / precheck view may raise "possibly moved" only when it is a credible view of the carton:
            # complete, not cut by the image border, camera >= NERO_V341_HOLDER_MIN_CAM_DIST_M from the carton, and
            # never while a placement is planned (then only the dedicated plan_placement look counts). V3.40 log:
            # camera 6-18 cm above the slot -> carton 111-116 mm off -> HOLDER_POSSIBLY_MOVED loop.
            why341 = None
            if V335_STATE.get("frozen_place"):
                why341 = "placement in progress"
            elif not h.get("complete") or h.get("touches_border"):
                why341 = "carton not complete in this view"
            else:
                try:
                    ctr341 = [float(fused["center_xy_m"][0]), float(fused["center_xy_m"][1]),
                              float(fused.get("top_z_m") or 0.045)]
                    if cam_pos is None or math.dist([float(v) for v in cam_pos], ctr341) < float(
                            os.environ.get("NERO_V341_HOLDER_MIN_CAM_DIST_M", "0.28")):
                        why341 = "camera too close to the carton"
                except Exception:
                    why341 = "no camera position"
            if why341:
                if time.time() - float(V335_STATE.get("v341_ign_t", 0.0)) > 10.0:
                    V335_STATE["v341_ign_t"] = time.time()
                    skill_core.emit(f"[V3.41 HOLDER VIEW IGNORED] {t} | carton {1000 * d:.0f} mm off in this view - "
                                    f"not used for the moved check ({why341})")
                return fused
        jumps = [j for j in (V335_STATE.get("holder_jump") or []) if math.dist(j["center_xy_m"], h["center_xy_m"]) <= thr]
        jumps.append(dict(h, _cam=cam_pos, _skill=skill_view))
        min_cam = float(os.environ.get("NERO_V336_MOVED_MIN_CAM_DIST_M", "0.05"))
        far_apart = any(a.get("_cam") and b.get("_cam") and math.dist(a["_cam"], b["_cam"]) >= min_cam
                        for a in jumps for b in jumps)
        if len(jumps) >= 2 and far_apart and any(j.get("_skill") for j in jumps):
            V335_STATE["holder_hist"] = jumps
            V335_STATE["holder_jump"] = []
            V335_STATE["holder_suspect"] = None
            nf = fuse(jumps)
            V335_STATE["holder_moved_t"] = time.time()
            skill_core.emit(f"[V3.36 HOLDER MOVED] {t} | carton moved {1000 * d:.0f} mm (confirmed by {len(jumps)} views "
                            f"from different camera positions) -> slots updated: centre {nf['center_xy_m']}")
            return nf
        V335_STATE["holder_jump"] = jumps
        V335_STATE["holder_suspect"] = {"t": time.time(), "d_mm": round(1000 * d, 1), "tag": t}
        skill_core.emit(f"[V3.36 HOLDER SUSPECT] {t} | this view puts the carton {1000 * d:.0f} mm away (threshold "
                        f"{1000 * thr:.0f} mm) - not believed until a dedicated look from another position confirms it")
        return fused

    def _v336_perceived_slot(slot_name):
        """Latest PERCEIVED (fused) slot centre - also while the placement slots are frozen."""
        h = _V329_SCENE.get("holder") or {}
        s_ = (h.get("slots") or {}).get(str(slot_name))
        return s_["xyz_m"] if s_ else EGG_SLOT_POSES.get(str(slot_name))

    def _v335_measured_geometry():
        """Gripper geometry of each egg from its CAMERA-MEASURED size (no egg class prior). V3.36: an egg whose grasp
        is planned keeps the geometry it was planned with (frozen); an egg without a reliable size gets the class
        placeholder + a conservative margin (clearly labelled)."""
        last = (V311_STATE.get("last") or {}).get("objects") or {}
        frozen = V335_STATE.get("frozen_grasp") or {}
        for n, o in last.items():
            if n in frozen or n not in V24_OBJECT_GEOMETRY:
                continue
            m = o.get("measurement") or {}
            w, L = m.get("width_mm"), m.get("length_mm")
            g = dict(V24_OBJECT_GEOMETRY[n])
            if w:
                g.update({"approx_size_mm": [round(float(w), 1), round(float(w), 1), round(float(L or w), 1)],
                          "effective_width_mm": round(float(w) + float(os.environ.get("NERO_V330_EGG_WIDTH_MARGIN_MM", "1.5")), 1),
                          "size_source": f"measured by the camera ({m.get('size_views')} clean views)"})
            elif m.get("method") == "multi_view_triangulation":
                try:
                    import NERO_V308_Camera_Perception as _p8g
                    w0 = 2000.0 * _p8g.EGG_PRIOR["half_short_m"]; l0 = 2000.0 * _p8g.EGG_PRIOR["half_long_m"]
                except Exception:
                    w0, l0 = 43.8, 63.0
                mg = float(os.environ.get("NERO_V336_UNMEASURED_MARGIN_MM", "4"))
                g.update({"approx_size_mm": [round(w0, 1), round(w0, 1), round(l0, 1)],
                          "effective_width_mm": round(w0 + 1.5 + mg, 1),
                          "size_source": f"NOT measured ({m.get('size_state') or 'no clean view'}) - class placeholder "
                                         f"+ {mg:g} mm conservative margin"})
            else:
                continue
            V24_OBJECT_GEOMETRY[n] = g

    def _v336_unfreeze_place(reason=""):
        if V335_STATE.get("frozen_place"):
            V335_STATE["frozen_place"] = None
            try:
                _v329_apply()                                # apply carton updates that waited
            except Exception:
                pass

    def _v335_inhand_verdict():
        ih = V335_STATE.get("in_hand") or {}
        if not ih or time.time() - float(ih.get("t", 0.0)) > 5.0:
            return {"camera_check": "no recent in-hand view"}
        out = {"camera_check": "visible" if ih.get("visible") else "not visible", "shift_mm": ih.get("shift_mm"),
               "capture": ih.get("capture"), "reason": ih.get("reason")}
        return out

    def _v335_precheck(name, args):
        """-> None (go on) or a failure result: fresh capture, then target vs plan (plan data are frozen)."""
        if not (v326_wrist_selected() and _v329_no_prior()):
            return None
        tools_grasp = ("move_to_pregrasp", "approach_planned_grasp")
        tools_place = ("orient_for_slot", "move_to_slot_entry", "guarded_insert_and_release")
        if name not in tools_grasp + tools_place:
            return None
        obj = str(args.get("object_name") or v24_skill_state.get("active_object") or "")
        thr = float(os.environ.get("NERO_V335_TARGET_MOVED_MM", "15")) / 1000.0
        try:
            p = _v335_capture(f"precheck:{name}:{obj}")
            _v335_ingest_live(p)
        except Exception as exc:
            if "PERCEPTION_UNAVAILABLE" in str(exc):
                raise
            skill_core.emit(f"[V3.35 PRECHECK WARN] {name} | capture failed ({exc}) - continuing with the last scene")
            return None
        if name in tools_grasp:
            plan = ((V27_GRASP_RUNTIME.get("plans") or {}).get(obj)) or {}

            def _clear_view(o_):
                cov_ = o_.get("coverage")         # partly hidden by the fingers / image border -> no verdict
                return bool(o_.get("seen")) and (cov_ is None or 0.6 <= float(cov_) <= 2.2)
            o = (p.get("objects") or {}).get(obj) or {}
            if plan.get("object_xyz_at_plan_m") and _clear_view(o):
                d = math.dist(o["xyz_m"][:2], plan["object_xyz_at_plan_m"][:2])
                skill_core.emit(f"[V3.35 PRECHECK] {name}({obj}) | egg seen {1000 * d:.0f} mm from the planned position")
                if d > thr:
                    try:                          # confirm with a second capture before refusing the motion
                        p2 = _v335_capture(f"precheck2:{name}:{obj}")
                        _v335_ingest_live(p2)
                        o2 = (p2.get("objects") or {}).get(obj) or {}
                        if _clear_view(o2):
                            d = math.dist(o2["xyz_m"][:2], plan["object_xyz_at_plan_m"][:2])
                        skill_core.emit(f"[V3.35 PRECHECK] {name}({obj}) | 2nd view: {1000 * d:.0f} mm")
                    except Exception as exc2:
                        if "PERCEPTION_UNAVAILABLE" in str(exc2):
                            raise
                if d > thr:
                    (V335_STATE.get("frozen_grasp") or {}).pop(obj, None)
                    return {"ok": False, "tool": name, "reason": "TARGET_MOVED_SINCE_PLAN",
                            "detail": f"the camera sees {obj} {1000 * d:.0f} mm away from where plan_grasp planned it "
                                      "(someone / something moved it). No motion was made.",
                            "warnings": [], "recommended_next_actions": ["plan_grasp"]}
            else:
                skill_core.emit(f"[V3.35 PRECHECK] {name}({obj}) | egg not in this view - continuing with the plan")
            return None
        plan = ((V271_PLACEMENT_RUNTIME.get("plans") or {}).get(obj)) or {}
        sl = plan.get("slot_name")
        if sl and plan.get("slot_center_xyz_m") and name in ("orient_for_slot", "move_to_slot_entry"):
            _v338_refine_slot(obj, sl, plan, p, name)
        cur = _v336_perceived_slot(sl) if sl else None
        if cur is not None and plan.get("slot_center_xyz_m"):
            d = math.dist([float(cur[0]), float(cur[1])], plan["slot_center_xyz_m"][:2])
            skill_core.emit(f"[V3.35 PRECHECK] {name}({obj}->{sl}) | perceived slot {1000 * d:.0f} mm from the planned "
                            f"(frozen) slot | in hand: {_v335_inhand_verdict()}")
            if d > thr:
                return {"ok": False, "tool": name, "reason": "SLOT_MOVED_SINCE_PLAN",
                        "detail": f"the egg carton was moved (confirmed): {sl} is now {1000 * d:.0f} mm from where "
                                  "plan_placement planned it. No motion was made; the egg is still held.",
                        "warnings": [], "recommended_next_actions": ["plan_placement"]}
        sus = V335_STATE.get("holder_suspect") or {}
        if sus and float(sus.get("t", 0.0)) > float((V335_STATE.get("frozen_place") or {}).get("t", 0.0)):
            return {"ok": False, "tool": name, "reason": "HOLDER_POSSIBLY_MOVED",
                    "detail": f"a camera view puts the egg carton {sus.get('d_mm')} mm away from where it was planned; "
                              "plan_placement looks at the carton again (dedicated view) before placing. The egg is "
                              "still held.", "warnings": [], "recommended_next_actions": ["plan_placement"]}
        ih = V335_STATE.get("in_hand") or {}
        if ih.get("visible") and float(ih.get("shift_mm") or 0.0) > float(os.environ.get("NERO_V335_INHAND_STOP_MM", "35")) \
                and int(V335_STATE.get("in_hand_bad", 0)) >= 2:
            return {"ok": False, "tool": name, "reason": "IN_HAND_SHIFT_DETECTED",
                    "detail": f"the wrist camera sees the held egg {ih.get('shift_mm')} mm away from where it was right "
                              "after the grasp (it slipped / turned in the fingers). The placement plan is no longer valid.",
                    "warnings": [], "recommended_next_actions": ["stop_and_report"]}
        return None

    def _v338_view_slot(payload, sl):
        """Slot centre seen in THIS capture (complete carton in view), else None."""
        hv = ((payload or {}).get("containers") or {}).get("egg_holder") or {}
        if not hv.get("complete") or hv.get("moved_suspect"):
            return None
        s_ = (hv.get("slots") or {}).get(str(sl)) or {}
        return list(s_["xyz_m"]) if s_.get("xyz_m") else None

    def _v338_refine_slot(obj, sl, plan, p1, name):
        """V3.38 SLOT REFINE (before orient_for_slot / move_to_slot_entry): if the carton is completely in view and its
        slot is NERO_V338_REFINE_MIN_MM (4 mm) .. the moved threshold away from the planned (frozen) slot, take a second
        capture; when both views agree within 3 mm the target slot is corrected to their mean (logged). Bigger
        differences stay a 'moved carton' (re-plan)."""
        if os.environ.get("NERO_V338_SLOT_REFINE", "1") == "0":
            return None
        thr = float(os.environ.get("NERO_V335_TARGET_MOVED_MM", "15")) / 1000.0
        dmin = float(os.environ.get("NERO_V338_REFINE_MIN_MM", "4")) / 1000.0
        s1 = _v338_view_slot(p1, sl)
        if s1 is None:
            return None
        ref = plan["slot_center_xyz_m"]
        d1 = math.dist(s1[:2], ref[:2])
        if not (dmin < d1 <= thr):
            return None
        try:
            p2 = _v335_capture(f"refine:{name}:{obj}")
            _v335_ingest_live(p2)
        except Exception as exc:
            if "PERCEPTION_UNAVAILABLE" in str(exc):
                raise
            return None
        s2 = _v338_view_slot(p2, sl)
        if s2 is None or math.dist(s1[:2], s2[:2]) > 0.003:
            skill_core.emit(f"[V3.38 SLOT REFINE SKIP] {obj}->{sl} | second view "
                            f"{'without a complete carton' if s2 is None else 'disagrees'} - keeping the planned slot")
            return None
        new = [0.5 * (s1[0] + s2[0]), 0.5 * (s1[1] + s2[1]), float(ref[2])]
        plan["slot_center_xyz_m"] = list(new)
        V271_SLOT_CENTERS_M[str(sl)] = tuple(new)
        EGG_SLOT_POSES[str(sl)] = tuple(new)
        fp = V335_STATE.get("frozen_place") or {}
        if fp.get("slot") == sl:
            fp["xyz"] = list(new)
        skill_core.emit(f"[V3.38 SLOT REFINE] {obj}->{sl} | two views put the slot {1000 * d1:.1f} mm from the plan -> "
                        f"target corrected to {[round(v, 4) for v in new]}")
        return new

    def _v335_slot_moved_now(obj, slot_name, slot_xyz):
        """Insertion loop guard: carton moved (confirmed) / possibly moved (live view) / egg turned in the fingers."""
        _v335_drain()
        thr = float(os.environ.get("NERO_V335_TARGET_MOVED_MM", "15")) / 1000.0
        cur = _v336_perceived_slot(slot_name)
        if cur is not None and math.dist([float(cur[0]), float(cur[1])], [float(slot_xyz[0]), float(slot_xyz[1])]) > thr:
            return "SLOT_MOVED_DURING_INSERTION"
        sus = V335_STATE.get("holder_suspect") or {}
        if sus and float(sus.get("t", 0.0)) > float((V335_STATE.get("frozen_place") or {}).get("t", 0.0)):
            return "HOLDER_POSSIBLY_MOVED_DURING_INSERTION"
        ih = V335_STATE.get("in_hand") or {}
        if ih.get("visible") and float(ih.get("shift_mm") or 0.0) > float(os.environ.get("NERO_V335_INHAND_STOP_MM", "35")) \
                and int(V335_STATE.get("in_hand_bad", 0)) >= 2:
            return "IN_HAND_SHIFT_DETECTED_DURING_INSERTION"
        return None


    # ------------------------------------------------------------
    # V3.29 NO-PRIOR WRIST SCENE: generic scan + perceived holder/slots + egg pile; no simulator truth in the checks.
    # Wrist camera mode only (NERO_V329_NO_PRIOR=1, default). Fixed camera and JSON truth modes keep the registered
    # layout and the V3.11 behaviour. The registered values below are kept ONLY for the truth comparison logs and
    # to restore the layout when another pose source is used.
    # ------------------------------------------------------------
    _V329_TRUTH = {}

    def _v329_truth_layout():
        """Registered layout, captured once before anything is replaced (truth logs + restore for other modes)."""
        if not _V329_TRUTH:
            _V329_TRUTH.update({
                "slot_centers": {k: tuple(v) for k, v in V271_SLOT_CENTERS_M.items()},
                "egg_slots": {k: tuple(v) for k, v in EGG_SLOT_POSES.items()},
                "basket": {"center_xy_m": list(EGG_BASKET["center_xy_m"]), "radius_m": float(EGG_BASKET["radius_m"])},
                "holder_center": list(EGG_HOLDER["center_xyz_m"]),
                "v27_basket": tuple(V27_BASKET_CENTER_XY_M), "rim_top": float(V271_HOLDER_RIM_TOP_Z_M),
                "safe_xy": tuple(V271_SAFE_ORIENTATION_XY_M),
            })
        return _V329_TRUTH
    _V329_SCENE = {"holder": None, "pile": None, "applied": False, "active": False, "seen": set()}
    _V329_EST = {"released": {}, "held_last": None, "log_t": {}, "truth_fn": None}

    def _v329_no_prior():
        """V3.29 wrist / V3.31 fixed camera: no preset layout, no simulator truth in the checks."""
        if v326_wrist_selected():
            return os.environ.get("NERO_V329_NO_PRIOR", "1") != "0"
        return v311_mode() == "camera" and v331_fixed_no_prior()

    def _v329_restore_layout():
        global V27_BASKET_CENTER_XY_M, V271_HOLDER_RIM_TOP_Z_M, V271_SAFE_ORIENTATION_XY_M
        _v329_truth_layout()
        V271_SLOT_CENTERS_M.update(_V329_TRUTH["slot_centers"])
        EGG_SLOT_POSES.update(_V329_TRUTH["egg_slots"])
        EGG_BASKET["center_xy_m"] = list(_V329_TRUTH["basket"]["center_xy_m"])
        EGG_BASKET["radius_m"] = float(_V329_TRUTH["basket"]["radius_m"])
        EGG_HOLDER["center_xyz_m"] = list(_V329_TRUTH["holder_center"])
        V27_BASKET_CENTER_XY_M = tuple(_V329_TRUTH["v27_basket"])
        V271_HOLDER_RIM_TOP_Z_M = float(_V329_TRUTH["rim_top"])
        V271_SAFE_ORIENTATION_XY_M = tuple(_V329_TRUTH["safe_xy"])

    def _v329_write_moveit_override(boxes):
        import NERO_V326_Wrist_Camera as _w
        ids = [b[0] for b in _w.holder_model()["boxes"]]
        try:
            V329_SCENE_OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = V329_SCENE_OVERRIDE_PATH.with_suffix(".tmp")
            tmp.write_text(json.dumps({"ids": ids, "boxes": boxes, "t": time.time()}))
            os.replace(tmp, V329_SCENE_OVERRIDE_PATH)
        except Exception as exc:
            skill_core.emit(f"[V3.29 MOVEIT SCENE WARN] {exc}")

    def _v329_reset_scene(reason=""):
        """Episode start: forget the perceived scene. Wrist no-prior mode -> MoveIt holder removed until perceived."""
        _V329_SCENE.update({"holder": None, "pile": None, "applied": False, "seen": set(), "views_without_new": 0,
                            "unreachable": set(), "holder_miss": 0, "height_unmeasured_scans": 0,
                            "scan_count": 0, "grid_hist": {}})
        _V329_EST["released"].clear()
        _V329_EST["held_last"] = None
        _V329_EST.setdefault("lost", {}).clear()
        V335_STATE["holder_hist"] = []
        V335_STATE["holder_jump"] = []
        V335_STATE["holder_suspect"] = None
        V335_STATE["frozen_grasp"] = {}
        V335_STATE["frozen_place"] = None
        V335_STATE["in_hand_bad"] = 0
        _v329_restore_layout()
        _v330_apply_class_geometry(_v329_no_prior())
        V329_LAYOUT_STATE["perceived_required"] = _v329_no_prior()
        V329_LAYOUT_STATE["holder_ok"] = False
        V329_LAYOUT_STATE["pile_ok"] = False
        if _v329_no_prior():
            _V329_SCENE["active"] = True
            _v329_write_moveit_override([])
            skill_core.emit(f"[V3.29 SCENE RESET] {reason} | no preset layout: basket/holder/slots unknown until "
                            "perceived; MoveIt holder boxes removed")
        else:
            _V329_SCENE["active"] = False
            try:
                V329_SCENE_OVERRIDE_PATH.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass

    def _v329_in_holder_xy(xy, h, margin=0.0):
        import NERO_V326_Wrist_Camera as _w
        m = _w.holder_model()
        c, yaw = h["center_xy_m"], float(h["yaw_rad"])
        dx, dy = float(xy[0]) - c[0], float(xy[1]) - c[1]
        du = dx * math.cos(yaw) + dy * math.sin(yaw)
        dv = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return abs(du) <= m["outer"][0] / 2.0 + margin and abs(dv) <= m["outer"][1] / 2.0 + margin

    def _v329_ingest(payload):
        """Fold one wrist payload into the scene: best holder detection, egg pile from the tracked eggs."""
        import NERO_V326_Wrist_Camera as _w
        if not payload:
            return
        h = ((payload.get("containers") or {}).get("egg_holder"))
        if h and h.get("complete") and os.environ.get("NERO_V335_HOLDER_FILTER", "1") != "0":
            h = _v335_holder_filter(h, payload.get("tag"), payload.get("cam_pos"))   # V3.36: graded views
        if h and (h.get("complete") or _w.holder_quality(h) >= _w.holder_quality(_V329_SCENE["holder"])):
            _V329_SCENE["holder"] = h
        objs = payload.get("objects") or {}
        n_before = len(_V329_SCENE["seen"])
        for n, o in objs.items():
            if o.get("seen"):
                _V329_SCENE["seen"].add(str(n))
                _V329_EST["released"].pop(str(n), None)
        unnamed331 = int((payload.get("tracking") or {}).get("unnamed_trusted") or 0)
        if unnamed331 > 0:
            # V3.31: more eggs than names in the pool -> report it; the view still counts as "new" (scan goes on)
            skill_core.emit(f"[V3.31 UNNAMED EGGS] {payload.get('tag')} | {unnamed331} trusted detection(s) without a free "
                            "name (raise NERO_V330_OBJECT_NAMES); they are not planned for")
        _V329_SCENE["views_without_new"] = (0 if (len(_V329_SCENE["seen"]) > n_before or unnamed331 > 0)
                                            else int(_V329_SCENE.get("views_without_new", 0)) + 1)
        _v329_apply()

    def _v329_update_pile():
        last = (V311_STATE.get("last") or {}).get("objects") or {}
        h = _V329_SCENE["holder"] if (_V329_SCENE["holder"] or {}).get("complete") else None
        pts = []
        for n in EGG_OBJECTS:
            o = last.get(n)
            if not o or not o.get("xyz_m"):
                continue
            x, y, z = [float(v) for v in o["xyz_m"]]
            if z > 0.10 or (h and _v329_in_holder_xy((x, y), h, 0.01)) or o.get("held"):
                continue
            pts.append((x, y))
        if not pts:
            return
        cx = sum(p[0] for p in pts) / len(pts); cy = sum(p[1] for p in pts) / len(pts)
        r = max(math.hypot(p[0] - cx, p[1] - cy) for p in pts) + float(os.environ.get("NERO_V329_PILE_MARGIN_M", "0.040"))
        _V329_SCENE["pile"] = {"center_xy_m": [round(cx, 4), round(cy, 4)], "radius_m": round(max(0.06, r), 4),
                               "eggs": len(pts)}

    def _v329_apply():
        global V27_BASKET_CENTER_XY_M, V271_HOLDER_RIM_TOP_Z_M, V271_SAFE_ORIENTATION_XY_M
        import NERO_V326_Wrist_Camera as _w
        if not _V329_SCENE.get("active"):
            return
        h = _V329_SCENE["holder"]
        changed = []
        if h and h.get("complete") and not V335_STATE.get("frozen_place"):    # V3.36: slots frozen while placing
            for k, s in (h.get("slots") or {}).items():
                xyz = tuple(float(v) for v in s["xyz_m"])
                V271_SLOT_CENTERS_M[k] = xyz
                EGG_SLOT_POSES[k] = xyz
            zt = float(h.get("table_z_m", 0.0))
            EGG_HOLDER["center_xyz_m"] = [float(h["center_xy_m"][0]), float(h["center_xy_m"][1]), zt + 0.006]
            zs = [float(s["xyz_m"][2]) for s in h["slots"].values()]
            V271_HOLDER_RIM_TOP_Z_M = round(sum(zs) / len(zs) + 0.001, 4)
            boxes = [[b[0], [round(v, 5) for v in b[1]], b[2], round(b[3], 5)] for b in _w.holder_boxes(h)]
            if boxes != _V329_SCENE.get("boxes"):
                _v329_write_moveit_override(boxes)
                _V329_SCENE["boxes"] = boxes
                changed.append("holder")
            V329_LAYOUT_STATE["holder_ok"] = True
        _v329_update_pile()
        p = _V329_SCENE["pile"]
        if p:
            V329_LAYOUT_STATE["pile_ok"] = True
            EGG_BASKET["center_xy_m"] = list(p["center_xy_m"])
            EGG_BASKET["radius_m"] = float(p["radius_m"])
            V27_BASKET_CENTER_XY_M = tuple(p["center_xy_m"])
        if p and h and h.get("complete"):
            V271_SAFE_ORIENTATION_XY_M = ((p["center_xy_m"][0] + h["center_xy_m"][0]) / 2.0,
                                          (p["center_xy_m"][1] + h["center_xy_m"][1]) / 2.0)
        if changed or not _V329_SCENE["applied"]:
            if h and h.get("complete"):
                _V329_SCENE["applied"] = True
                _v329_log_layout_vs_truth()

    def _v329_log_layout_vs_truth():
        _v329_truth_layout()
        h, p = _V329_SCENE["holder"], _V329_SCENE["pile"]
        errs = []
        for k, t in _V329_TRUTH["slot_centers"].items():
            s = (h.get("slots") or {}).get(k)
            if s:
                errs.append(math.dist(s["xyz_m"][:2], t[:2]) * 1000.0)
        tc = _V329_TRUTH["holder_center"]
        skill_core.emit(
            f"[V3.29 SCENE PERCEPTION] holder centre {h['center_xy_m']} yaw {math.degrees(float(h['yaw_rad'])):.1f}deg "
            f"extent {h.get('extent_m')} | slots with posts {h.get('slots_with_posts')}/8 | egg pile "
            f"{(p or {}).get('center_xy_m')} r={(p or {}).get('radius_m')} | truth(log only): holder err "
            f"{math.dist(h['center_xy_m'], tc[:2]) * 1000.0:.1f} mm, slot err mean "
            f"{(sum(errs) / len(errs)) if errs else float('nan'):.1f} max {max(errs) if errs else float('nan'):.1f} mm")

    def _v329_scene_ready():
        """Holder complete + egg pile perceived + no new egg in the last NERO_V330_SCAN_STABLE_VIEWS views (V3.31: neither
        the number of eggs nor the size of the name pool ends the scan)."""
        h = _V329_SCENE["holder"]
        if not (h and h.get("complete") and _V329_SCENE["pile"]):
            return False
        if (os.environ.get("NERO_V335_REQUIRE_HOLDER_HEIGHT", "1") != "0" and h.get("top_z_source")
                and not str(h.get("top_z_source")).startswith("measured")
                and int(_V329_SCENE.get("height_unmeasured_scans", 0)) < 1):
            return False                                   # V3.35: one full scan to measure the carton height
        # V3.31: only the scan itself decides - no assumption on how many eggs there are
        return int(_V329_SCENE.get("views_without_new", 0)) >= int(os.environ.get("NERO_V330_SCAN_STABLE_VIEWS", "2"))

    def _v329_holder_tag():
        if not _v329_no_prior():
            return None
        h = _V329_SCENE["holder"]
        out335 = {"source": "wrist_scan", "perceived": bool(h and h.get("complete")),
                  "yaw_deg": round(math.degrees(float(h["yaw_rad"])), 1) if h else None}
        if h:
            out335.update({"measured_extent_mm": [round(1000 * float(v), 1) for v in (h.get("extent_m") or [])],
                           "top_z_m": h.get("top_z_m"), "top_z_source": h.get("top_z_source"),
                           "slot_layout_source": h.get("slot_layout_source"), "fused_views": h.get("fused_views", 1)})
        return out335

    def _v329_pile_tag():
        if not _v329_no_prior():
            return None
        p = _V329_SCENE["pile"]
        return {"source": "perceived_egg_pile", "perceived": bool(p), "radius_m": (p or {}).get("radius_m"),
                "note": "no physical basket in this scene; region = perceived eggs outside the holder"}

    def _v329_truth_pose(name):
        fn = _V329_EST.get("truth_fn") or skill_core.object_pose
        return fn(name)

    def _v329_object_pose(name):
        """Object pose WITHOUT simulator truth (wrist no-prior mode): held -> TCP x grasp-time transform (grasp from
        perception); released -> where it was let go until the camera sees it again; otherwise the perceived pose.
        Simulator truth is only logged for comparison."""
        truth_fn = _V329_EST["truth_fn"]
        if not _v329_no_prior() or not str(name).startswith("egg"):
            return truth_fn(name)                      # other modes / hidden legacy bodies: unchanged
        name = str(name)
        est, src = None, None
        held = bool(hold_active_shared.value) and str(v24_skill_state.get("active_object") or "") == name
        if held:
            try:
                tcp, tq = skill_core.current_tcp_pose()
                e = _v3121_held_pose_estimate(name, tcp, tq)
            except Exception:
                e = None
            if e is not None:
                est, src = ([float(v) for v in e[0]], [float(v) for v in e[1]]), "tcp_x_grasp"
                _V329_EST["held_last"] = (name, est)
        else:
            hl = _V329_EST.get("held_last")
            if hl and hl[0] == name:
                rel_xyz, rel_q = hl[1]
                # let go above a perceived slot -> it settles in that slot (rest height as the placed hint)
                sl = _egg_slot_for_xyz([rel_xyz[0], rel_xyz[1], 0.05])
                if sl is not None:
                    c = EGG_SLOT_POSES[sl]
                    rel_xyz = [float(c[0]), float(c[1]), _v331_egg_in_slot_z(c)]
                _V329_EST["released"][name] = (list(rel_xyz), list(rel_q))
                _V329_EST["held_last"] = None
            if name in _V329_EST["released"]:
                est, src = _V329_EST["released"][name], "released_at_tcp"
        fz336 = (V335_STATE.get("frozen_grasp") or {}).get(name) or {}
        if est is None and fz336.get("xyz"):
            est, src = (list(fz336["xyz"]), list(fz336.get("quat") or [0, 0, 0, 1])), "frozen_at_plan_grasp"
        if est is None:
            o = _v311_obj_entry(name)
            if o is not None and o.get("xyz_m"):
                est, src = ([float(v) for v in o["xyz_m"]], [float(v) for v in (o.get("quat_xyzw") or [0, 0, 0, 1])]), "perception"
        if est is None:
            raise RuntimeError(f"V3.29 NO-TRUTH: {name} has no perceived pose yet")
        now = time.time()
        if now - _V329_EST["log_t"].get(name, 0.0) > float(os.environ.get("NERO_V329_TRUTH_LOG_S", "5")):
            try:
                _tn330, txyz330 = _v330_nearest_truth(est[0])
                err = math.dist(est[0], [float(v) for v in txyz330]) * 1000.0
                if err > float(os.environ.get("NERO_V329_TRUTH_LOG_MM", "10")):
                    skill_core.emit(f"[V3.29 TRUTH LOG] {name} | estimate ({src}) vs nearest simulator egg {_tn330} "
                                    f"{err:.1f} mm (log only, not used)")
                _V329_EST["log_t"][name] = now
            except Exception:
                pass
        return est

    def _v331_egg_in_slot_z(slot_xyz):
        """Egg centre height in a slot: holder model + egg class (V3.31), relative to the slot's own table plane."""
        import NERO_V326_Wrist_Camera as _w
        if os.environ.get("NERO_V328_PLACED_DZ_M"):
            return float(slot_xyz[2]) + float(os.environ["NERO_V328_PLACED_DZ_M"])
        m = _w.holder_model()
        table_z = float(slot_xyz[2]) - (m["post_top"] - m["slot_z_below_post_top"])
        return _w.egg_center_in_slot_z(table_z)

    def _v330_nearest_truth(xyz):
        """LOG ONLY: the simulator egg nearest to a perceived position (perception names its eggs itself)."""
        fn = _V329_EST.get("truth_fn") or skill_core.object_pose
        best = None
        for n in EGG_OBJECTS:
            try:
                t = [float(v) for v in fn(n)[0]]
            except Exception:
                continue
            d = math.dist([float(v) for v in xyz], t)
            if best is None or d < best[0]:
                best = (d, n, t)
        if best is None:
            raise RuntimeError("no simulator egg")
        return best[1], best[2]

    def _v330_apply_class_geometry(on):
        """V3.30: wrist no-prior mode uses ONE egg class model for every egg (no per-egg simulator sizes)."""
        lay = _v329_truth_layout()
        if "geometry" not in lay:
            lay["geometry"] = {k: dict(v) for k, v in V24_OBJECT_GEOMETRY.items()}
        if not on:
            for k, v in lay["geometry"].items():
                V24_OBJECT_GEOMETRY[k] = dict(v)
            return
        try:
            import NERO_V308_Camera_Perception as _p8
            a_mm, b_mm = 2000.0 * _p8.EGG_PRIOR["half_long_m"], 2000.0 * _p8.EGG_PRIOR["half_short_m"]
        except Exception:
            a_mm, b_mm = 63.0, 43.8
        w = round(b_mm + float(os.environ.get("NERO_V330_EGG_WIDTH_MARGIN_MM", "1.5")), 1)
        for k in list(V24_OBJECT_GEOMETRY):
            g = dict(lay["geometry"][k])
            g.update({"approx_size_mm": [w, w, round(a_mm, 1)], "effective_width_mm": w,
                      "size_source": "egg class placeholder (V3.30) - NOT measured yet; replaced by the camera measurement (V3.35)"})
            V24_OBJECT_GEOMETRY[k] = g

    def _v329_install_no_truth():
        if (skill_core is not None and not getattr(skill_core, "_v341_exec_wrapped", False)
                and hasattr(skill_core, "_execute_trajectory")
                and os.environ.get("NERO_V341_HELD_REFRESH", "1") != "0"):
            # V3.41: refresh the held-egg pose after EVERY arm motion, so a release (return_to_pick_location,
            # insertion, ...) knows where the egg really was (V3.40 log: egg back in the basket, reported in slot_1
            # from the stale pose -> RETURN FAILED + EGG LOST)
            _orig_exec341 = skill_core._execute_trajectory

            def _v341_exec(traj, *a, **k):
                try:
                    return _orig_exec341(traj, *a, **k)
                finally:
                    try:
                        if _v329_no_prior() and bool(hold_active_shared.value):
                            act341 = str(v24_skill_state.get("active_object") or "")
                            if act341.startswith("egg"):
                                _v329_object_pose(act341)
                    except Exception:
                        pass
            skill_core._execute_trajectory = _v341_exec
            skill_core._v341_exec_wrapped = True
        if _V329_EST["truth_fn"] is None and skill_core is not None:
            _V329_EST["truth_fn"] = skill_core.object_pose
            skill_core.object_pose = _v329_object_pose
            skill_core.emit("[V3.29 NO-TRUTH] object poses in the skill checks now come from perception / TCP / force "
                            "in wrist mode; simulator truth only in [V3.29 TRUTH LOG]")

    def _v329_near_zone(x, y, z, z_max):
        """near_basket / near_holder for the posture checks, from the current (perceived or registered) layout.
        V3.31: a zone that has not been perceived yet does not exist (the checks fall back to height + joints)."""
        if V329_LAYOUT_STATE.get("perceived_required"):
            nb = nh = False
            if V329_LAYOUT_STATE.get("pile_ok"):
                bx, by = EGG_BASKET["center_xy_m"]
                nb = math.hypot(x - bx, y - by) < max(0.135, float(EGG_BASKET["radius_m"]) + 0.047) and z < z_max
            if V329_LAYOUT_STATE.get("holder_ok"):
                hx, hy = EGG_HOLDER["center_xyz_m"][0], EGG_HOLDER["center_xyz_m"][1]
                nh = math.hypot(x - hx, y - hy) < 0.145 and z < z_max
            return nb, nh
        bx, by = EGG_BASKET["center_xy_m"]
        hx, hy = EGG_HOLDER["center_xyz_m"][0], EGG_HOLDER["center_xyz_m"][1]
        rb = max(0.135, float(EGG_BASKET["radius_m"]) + 0.047)
        return (math.hypot(x - bx, y - by) < rb and z < z_max), (math.hypot(x - hx, y - hy) < 0.145 and z < z_max)

    def _v329_scan(label, payloads, views, look_failed):
        """Blind first survey: ring of table directions, then a closer look at a partially seen holder / the pile."""
        import NERO_V326_Wrist_Camera as _v326w
        skill_core.emit(f"[V3.29 SCAN START] {label} | no preset coordinates; workspace grid "
                        f"{[p_ for _n, p_ in _v326w.scan_points()]} (V3.31: table area x reach, no object knowledge)")

        unreach = _V329_SCENE.setdefault("unreachable", set())

        def one(vname, pt, dists):
            key333 = (round(float(pt[0]), 3), round(float(pt[1]), 3))
            if key333 in unreach:
                look_failed.append(vname)
                skill_core.emit(f"[V3.33 SCAN SKIP] {label}/{vname} | {list(key333)} was unreachable earlier in this task")
                return None
            try:
                _v326_move_look(pt, f"{label}/{vname}", distances=dists)
            except Exception as exc:
                look_failed.append(vname)
                if "UNREACHABLE" in str(exc):
                    unreach.add(key333)                   # V3.33: do not try this view again in this task
                skill_core.emit(f"[V3.26 WRIST LOOK WARN] {label}/{vname} | look move failed ({str(exc)[:200]})")
                return None
            p = _v335_capture(f"{label}/{vname}")
            payloads.append(p); views.append(vname)
            V311_STATE["last"] = _v326w.merge_payloads(payloads, views)
            _v329_ingest(p)
            h = (p.get("containers") or {}).get("egg_holder")
            skill_core.emit(f"[V3.29 SCAN VIEW] {label}/{vname} | eggs seen "
                            f"{sorted(n for n, o in (p.get('objects') or {}).items() if o.get('seen'))} | holder "
                            + (f"{'complete' if h.get('complete') else 'partial'} at {h.get('center_xy_m')} posts "
                               f"{h.get('slots_with_posts')}/8" if h else "not in view"))
            return p
        fast339 = os.environ.get("NERO_V339_SCAN_FAST", "1") != "0"
        n_scan339 = int(_V329_SCENE.get("scan_count", 0)) + 1
        _V329_SCENE["scan_count"] = n_scan339
        grid339 = _V329_SCENE.setdefault("grid_hist", {})
        done339 = {"refine": False, "height": False}

        def refine_holder():
            """V3.33 looks at a partly seen carton (usual distance, farther, closer) + V3.39: when those are
            unreachable, the same carton from points shifted towards the robot base and along the carton axis
            (V3.38 log: holder_refine2 unreachable -> HOLDER_NOT_PERCEIVED -> a second full observe)."""
            h_ = _V329_SCENE["holder"]
            if not h_ or h_.get("complete"):
                return
            zt_ = float(h_.get("table_z_m", 0.0))
            max_caps = int(os.environ.get("NERO_V339_HOLDER_REFINE_MAX", "5"))
            caps = 0
            plan = [("holder_refine", 0.0, 0.0, _v326w.scan_distances()), ("holder_refine2", 0.0, 0.0, [0.52, 0.48]),
                    ("holder_refine3", 0.0, 0.0, [0.40, 0.45, 0.36])]
            if fast339:
                plan += [("holder_refine_in", 0.07, 0.0, [0.45, 0.40, 0.52]),
                         ("holder_refine_a", 0.0, 0.07, [0.45, 0.40]), ("holder_refine_b", 0.0, -0.07, [0.45, 0.40])]
            for vname_, d_in, d_ax, dists_ in plan:
                h2 = _V329_SCENE["holder"]
                if (h2 or {}).get("complete") or caps >= max_caps:
                    break
                lxy = (h2 or h_).get("look_xy_m") or (h2 or h_)["center_xy_m"]
                r_ = math.hypot(lxy[0], lxy[1]) or 1.0
                yw_ = float((h2 or h_).get("yaw_rad", 0.0))
                x_ = lxy[0] - d_in * lxy[0] / r_ + d_ax * math.cos(yw_)
                y_ = lxy[1] - d_in * lxy[1] / r_ + d_ax * math.sin(yw_)
                if one(vname_, [x_, y_, zt_ + 0.043], dists_) is not None:
                    caps += 1
            if fast339 and not (_V329_SCENE["holder"] or {}).get("complete"):
                skill_core.emit(f"[V3.39 HOLDER REFINE] {label} | carton still partial after {caps} refine view(s)")

        def height_views():
            """V3.39: carton found but its height not measured yet -> two looks from camera positions ~0.2 m apart
            (centre +- 0.10 m along the carton axis) in THIS scan (V3.38: locked only in the third observe)."""
            h_ = _V329_SCENE["holder"]
            if not (h_ and h_.get("complete")) or str(h_.get("top_z_source", "measured")).startswith("measured"):
                return
            zt_ = float(h_.get("table_z_m", 0.0))
            c_, yw_ = h_["center_xy_m"], float(h_.get("yaw_rad", 0.0))
            off = float(os.environ.get("NERO_V339_HEIGHT_VIEW_OFFSET_M", "0.10"))
            for k_, s_ in enumerate((1.0, -1.0)):
                one(f"holder_height_{k_ + 1}", [c_[0] + s_ * off * math.cos(yw_), c_[1] + s_ * off * math.sin(yw_),
                                                 zt_ + 0.043], [0.45, 0.40])
                if str((_V329_SCENE["holder"] or {}).get("top_z_source", "")).startswith("measured"):
                    skill_core.emit(f"[V3.39 HOLDER HEIGHT] {label} | carton top measured after {k_ + 1} height view(s)")
                    break

        for vname, pt in _v326w.scan_points():
            gk339 = (round(float(pt[0]), 3), round(float(pt[1]), 3))
            if fast339 and n_scan339 > 1 and grid339.get(gk339) == "empty":
                skill_core.emit(f"[V3.39 SCAN REUSE] {label}/{vname} | nothing there in an earlier scan of this task - skipped")
                continue
            p339 = one(vname, pt, _v326w.scan_distances())
            if p339 is not None:
                seen339 = any(o_.get("seen") for o_ in (p339.get("objects") or {}).values())
                h339 = (p339.get("containers") or {}).get("egg_holder")
                grid339[gk339] = "content" if (seen339 or h339) else "empty"
            if fast339:
                hs339 = _V329_SCENE["holder"]
                if hs339 and not hs339.get("complete") and not done339["refine"]:
                    done339["refine"] = True
                    refine_holder()                  # V3.39: at once, while the arm is near the carton
                if (_V329_SCENE["holder"] or {}).get("complete") and not done339["height"]:
                    done339["height"] = True
                    height_views()
            if _v329_scene_ready():
                break
        h = _V329_SCENE["holder"]
        if h and not h.get("complete"):
            if not done339["refine"]:
                done339["refine"] = True
                refine_holder()
        elif not h:
            # V3.33: nothing like a carton in the whole grid -> a second, half-offset pass (at most 4 views)
            sp333 = float(os.environ.get("NERO_V331_SCAN_SPACING_M", "0.26")) / 2.0
            extra = [(f"holder_search_{k + 1}", [p_[0] + sp333 * (1 if k % 2 == 0 else -1), p_[1] + sp333, p_[2]])
                     for k, (_n, p_) in enumerate(_v326w.scan_points())][:int(os.environ.get("NERO_V333_HOLDER_SEARCH_VIEWS", "4"))]
            for vname, pt in extra:
                one(vname, pt, _v326w.scan_distances())
                if _V329_SCENE["holder"]:
                    break
            h = _V329_SCENE["holder"]
            if h and not h.get("complete") and not done339["refine"]:
                done339["refine"] = True
                refine_holder()
        if fast339 and not done339["height"] and (_V329_SCENE["holder"] or {}).get("complete"):
            done339["height"] = True
            height_views()
        if _V329_SCENE["pile"] and not _v329_scene_ready():
            pc = _V329_SCENE["pile"]["center_xy_m"]
            one("pile_refine", [pc[0], pc[1], 0.03], None)
        ready = _v329_scene_ready()
        skill_core.emit(f"[V3.29 SCAN {'DONE' if ready else 'INCOMPLETE'}] {label} | holder "
                        f"{'complete' if (_V329_SCENE['holder'] or {}).get('complete') else 'NOT perceived'} | eggs seen "
                        f"{sorted(_V329_SCENE['seen'])} | pile {(_V329_SCENE['pile'] or {}).get('center_xy_m')}")
        return ready

    def _v329_preplace_look(obj, slot_name):
        """Before placement (egg held, gripper still down): look at the holder, refresh the slots, return to the
        joint state the arm had (so orient_for_slot starts from the same posture as before)."""
        import NERO_V326_Wrist_Camera as _v326w
        h = _V329_SCENE["holder"]
        if not h:
            return False
        s = (h.get("slots") or {}).get(str(slot_name)) or {}
        pt = list(s.get("xyz_m") or [h["center_xy_m"][0], h["center_xy_m"][1], 0.043])
        live = skill_core._current_joint_state_copy()
        q_back = [float(v) for v in skill_core._arm_positions_from_joint_state(live)]
        # V3.38: with an egg in the gripper the near views (0.36 m) are often unreachable (V3.37 log: 0 solutions).
        # Try farther views one distance at a time (each IK screen only covers its first 40 candidates), then the
        # carton centre (a view of the whole carton refines every slot).
        try:
            dists338 = [float(v) for v in os.environ.get("NERO_V338_PREPLACE_DIST_M", "0.45,0.40,0.52,0.36").split(",")]
        except ValueError:
            dists338 = [0.45, 0.40, 0.52, 0.36]
        targets338 = [("slot", pt)] + [("carton_centre", [float(h["center_xy_m"][0]), float(h["center_xy_m"][1]), 0.043])]
        moved338, last_exc = False, None
        for tname338, tp338 in targets338:
            for d338 in dists338:
                try:
                    _v326_move_look(tp338, f"plan_placement:{obj}/holder", distances=[d338], no_detour=True)
                    moved338 = True
                    skill_core.emit(f"[V3.38 PRE-PLACE LOOK] {obj}->{slot_name} | view of the {tname338} at {d338:.2f} m")
                    break
                except Exception as exc:
                    last_exc = exc
                    if "safety" in str(exc).lower() or "TRACKING MISMATCH" in str(exc):
                        break
            if moved338 or (last_exc is not None and ("safety" in str(last_exc).lower()
                                                       or "TRACKING MISMATCH" in str(last_exc))):
                if last_exc is not None and "TRACKING MISMATCH" in str(last_exc) and not moved338:
                    # V3.40: the arm did not follow (V3.39 log: 6 x EXEC VERIFY FAILED 90-106 deg while holding an
                    # egg) -> no more look poses; place with the fused carton
                    skill_core.emit(f"[V3.40 PRE-PLACE LOOK STOP] {obj}->{slot_name} | the arm did not follow a look "
                                    "trajectory -> no further views; using the fused carton")
                break
        if not moved338:
            skill_core.emit(f"[V3.29 PRE-PLACE LOOK WARN] {obj}->{slot_name} | look move failed ({str(last_exc)[:200]}); "
                            "using the slots from the last holder view")
            return False
        try:
            p = _v335_capture(f"plan_placement:{obj}/holder")
            hn = (p.get("containers") or {}).get("egg_holder")
            if hn and hn.get("complete"):
                # V3.40: this view (egg in hand, often far / oblique, carton height maybe not measured) does NOT
                # replace the fused carton (V3.39 log: 13 mm off -> planned slot vs fused slot 16 mm -> false
                # SLOT_MOVED_SINCE_PLAN). It refines the fused pose only when it agrees with it; a disagreeing view is
                # ignored here (moved-carton detection stays with the V3.36 filter: two views, two camera positions).
                hf340 = _V329_SCENE.get("holder") if (_V329_SCENE.get("holder") or {}).get("complete") else None
                agree340 = float(os.environ.get("NERO_V340_PREPLACE_AGREE_MM", "6")) / 1000.0
                if hf340 is None or os.environ.get("NERO_V340_PREPLACE_FILTER", "1") == "0":
                    _V329_SCENE["holder"] = hn
                else:
                    d340 = math.dist(hn["center_xy_m"], hf340["center_xy_m"])
                    if d340 <= agree340:
                        hfil340 = _v335_holder_filter(hn, f"plan_placement:{obj}/holder", p.get("cam_pos"))
                        if hfil340:
                            _V329_SCENE["holder"] = hfil340
                        skill_core.emit(f"[V3.40 PRE-PLACE VIEW] {obj}->{slot_name} | agrees with the fused carton "
                                        f"({1000 * d340:.0f} mm) -> fused")
                    else:
                        skill_core.emit(f"[V3.40 PRE-PLACE VIEW IGNORED] {obj}->{slot_name} | this view puts the carton "
                                        f"{1000 * d340:.0f} mm from the fused pose (> {1000 * agree340:.0f} mm) -> "
                                        "keeping the fused carton")
            _v329_apply()
            s2 = ((_V329_SCENE["holder"] or {}).get("slots") or {}).get(str(slot_name)) or {}
            skill_core.emit(f"[V3.29 PRE-PLACE LOOK] {obj}->{slot_name} | holder "
                            f"{'complete' if (hn or {}).get('complete') else 'not complete in this view'} | {slot_name} at "
                            f"{s2.get('xyz_m')} (posts {s2.get('post_sectors')}/6) | truth(log only) "
                            f"{_v329_truth_layout()['slot_centers'].get(str(slot_name))}")
        finally:
            try:
                traj = skill_core._plan_joint_solution(q_back, velocity_scale=0.15, accel_scale=0.10,
                                                       planning_attempts=6, planning_time=5.0)
                if traj is not None:
                    _v218_slow_trajectory_in_place(traj, factor=1.6)
                    skill_core._execute_trajectory(traj)
                _v310_wait_tcp_settled(f"V3.29 pre-place return {obj}")
            except Exception as exc:
                skill_core.emit(f"[V3.29 PRE-PLACE RETURN WARN] {obj} | {str(exc)[:200]}")
        return True

    _V340_DIAG = {"t": 0.0}

    def _v340_curobo_diag(label):
        """V3.40: cuRobo failed every attempt within milliseconds (V3.39 log, held egg_2: all reorientation, recenter
        AND return_to_pick_location moves) -> the START state is the suspect, not the goal. Logs joint-limit margins
        and the MoveIt validity of the current state (at most every 5 s). -> dict"""
        out = {}
        try:
            live = skill_core._current_joint_state_copy()
            q = [float(v) for v in skill_core._arm_positions_from_joint_state(live)]
            marg = [min(q[i] - float(JOINT_LIMITS_RAD[i][0]), float(JOINT_LIMITS_RAD[i][1]) - q[i]) for i in range(7)]
            k = min(range(7), key=lambda i: marg[i])
            out = {"q_deg": [round(math.degrees(v), 1) for v in q], "min_limit_margin_deg": round(math.degrees(marg[k]), 1),
                   "joint": f"J{k + 1}"}
            try:
                ok_v, why_v = skill_core._v71_state_is_valid(q)
                out["moveit_state_valid"] = bool(ok_v)
                out["moveit_state"] = str(why_v)[:120]
            except Exception as exc_v:
                out["moveit_state"] = f"check failed: {str(exc_v)[:80]}"
        except Exception as exc:
            out = {"error": str(exc)[:160]}
        if time.time() - float(_V340_DIAG.get("t", 0.0)) > 5.0:
            _V340_DIAG["t"] = time.time()
            skill_core.emit(f"[V3.40 CUROBO DIAG] {label} | start state: {out}")
        _V340_DIAG["last"] = out
        return out

    def _v340_joint_fallback(label, exc, joints, vel=0.12, acc=0.10):
        """V3.40: cuRobo Strict failed -> MoveIt/OMPL joint goal to the already IK-screened (collision-aware BioIK)
        solution, if there is one. -> trajectory or None"""
        if os.environ.get("NERO_V340_MOVEIT_FALLBACK", "1") == "0" or "cuRobo" not in str(exc):
            return None
        _v340_curobo_diag(label)
        if joints is None:
            return None
        try:
            traj = skill_core._plan_joint_solution([float(v) for v in joints], velocity_scale=vel, accel_scale=acc,
                                                   planning_attempts=6, planning_time=5.0)
            if traj is not None:
                skill_core.emit(f"[V3.40 MOVEIT FALLBACK] {label} | cuRobo failed -> MoveIt joint goal OK")
            return traj
        except Exception as exc_j:
            skill_core.emit(f"[V3.40 MOVEIT FALLBACK FAIL] {label} | {str(exc_j)[:160]}")
            return None

    def _v340_pose_with_fallback(label, xyz, quat, vel=0.12, acc=0.12, pos_tol=0.010):
        """V3.40: plan a pose (cuRobo); on a cuRobo failure: collision-aware BioIK seeded at the current state ->
        MoveIt joint goal, then the V70 BioIK pose planner. Raises the original error if all fail."""
        try:
            return skill_core._plan_pose(xyz, quat, velocity_scale=vel, accel_scale=acc, position_tolerance=pos_tol)
        except Exception as exc:
            if os.environ.get("NERO_V340_MOVEIT_FALLBACK", "1") == "0" or "cuRobo" not in str(exc):
                raise
            j = None
            try:
                live = skill_core._current_joint_state_copy()
                j, _e = skill_core._compute_ik(list(xyz), list(quat), timeout_s=0.10, avoid_collisions=True,
                                               seed_joint_state=live)
            except Exception:
                j = None
            traj = _v340_joint_fallback(label, exc, j, vel, acc)
            if traj is not None:
                return traj
            try:
                traj = skill_core._plan_pose_bioik_v70(xyz, quat, velocity_scale=vel, accel_scale=acc,
                                                       position_tolerance=pos_tol)
                if traj is not None:
                    skill_core.emit(f"[V3.40 MOVEIT FALLBACK] {label} | cuRobo failed -> BioIK pose plan OK")
                    return traj
            except Exception as exc_b:
                skill_core.emit(f"[V3.40 MOVEIT FALLBACK FAIL] {label} | BioIK pose: {str(exc_b)[:160]}")
            raise exc

    def _v341_insert_ik_step(target_tcp, quat):
        """V3.41: one insertion level by collision-aware BioIK seeded at the current state (same TCP orientation),
        accepted only on the same branch (max joint step NERO_V341_INSERT_MAX_JOINT_STEP_DEG) and off the limits;
        planned as a MoveIt joint goal. -> (trajectory | None, why)"""
        try:
            live = skill_core._current_joint_state_copy()
            qcur = [float(v) for v in skill_core._arm_positions_from_joint_state(live)]
            j, _e = skill_core._compute_ik(list(target_tcp), list(quat), timeout_s=0.10, avoid_collisions=True,
                                           seed_joint_state=live)
        except Exception as exc:
            return None, f"IK exception {str(exc)[:80]}"
        if j is None:
            return None, "no collision-free IK for this level"
        step = max(abs(float(j[i]) - qcur[i]) for i in range(7))
        if math.degrees(step) > float(os.environ.get("NERO_V341_INSERT_MAX_JOINT_STEP_DEG", "12")):
            return None, f"only on another branch (joint step {math.degrees(step):.0f} deg)"
        m = v341_limit_margins_deg(j)
        if m[0] < 0.5:
            return None, f"J{m[1]} at its limit ({m[0]:.1f} deg)"
        try:
            traj = skill_core._plan_joint_solution([float(v) for v in j], velocity_scale=0.08, accel_scale=0.08,
                                                   planning_attempts=4, planning_time=3.0)
        except Exception as exc:
            return None, f"joint plan {str(exc)[:80]}"
        if traj is None:
            return None, "joint plan empty"
        return traj, f"joint step {math.degrees(step):.1f} deg, J{m[1]} {m[0]:.1f} deg from its limit"

    def _v339_rgb_only():
        import NERO_V326_Wrist_Camera as _v326w
        try:
            return v326_wrist_selected() and not _v326w.wrist_config(v330_wrist_profile()).get("depth", True)
        except Exception:
            return False

    def _v339_grasp_issue(o, payload=None):
        """V3.39 GRASP POSE CHECK (RGB-only wrist camera): why the perceived target is not good enough to plan a
        grasp from -> str, or None when it is. Lying egg: centre must be at table + half width (+-
        NERO_V339_GRASP_Z_TOL_M); the measured position must agree with the latest single view
        (NERO_V339_VIEW_DISAGREE_M)."""
        import NERO_V326_Wrist_Camera as _v326w
        import NERO_V308_Camera_Perception as _p8
        if not o:
            return "target never perceived"
        if not o.get("seen"):
            return "target not seen in the grasp view"
        if not o.get("trusted"):
            return f"target not trusted ({o.get('untrusted_reason') or 'detection'})"
        m = o.get("measurement") or {}
        if str(m.get("method", "")).startswith("single_view"):
            return "position not measured from two views yet"
        xyz = [float(v) for v in o.get("xyz_m") or [0, 0, 0]]
        tv = o.get("this_view_xyz_m")
        lim = float(os.environ.get("NERO_V339_VIEW_DISAGREE_M", "0.010"))
        if tv and math.hypot(xyz[0] - float(tv[0]), xyz[1] - float(tv[1])) > lim:
            return (f"measured position and this view differ by "
                    f"{1000 * math.hypot(xyz[0] - float(tv[0]), xyz[1] - float(tv[1])):.0f} mm")
        if str(o.get("location", "")).startswith("table"):
            tz, _src = _v326w.table_plane(((payload or {}).get("measured") or {}).get("table_z_m"))
            hw = (float(m["width_mm"]) / 2000.0 if m.get("width_mm") else float(_p8.EGG_PRIOR["half_short_m"]))
            ztol = float(os.environ.get("NERO_V339_GRASP_Z_TOL_M", "0.010"))
            if tz is not None and abs(xyz[2] - (tz + hw)) > ztol:
                return (f"lying egg centre z {xyz[2]:.3f} m is {1000 * (xyz[2] - tz - hw):+.0f} mm from table + half "
                        f"width ({tz + hw:.3f} m)")
        return None

    def _v326_wrist_capture(label, names=None):
        import NERO_V326_Wrist_Camera as _v326w
        _v329_install_no_truth()
        holding = bool(hold_active_shared.value)
        payloads, views, look_failed = [], [], []
        scanned = False
        if _v329_no_prior() and not _V329_SCENE.get("active"):
            _v329_reset_scene("first wrist capture")
        if _v329_no_prior() and not holding and not _v329_scene_ready():
            scanned = True
            _v329_scan(label, payloads, views, look_failed)
            h335 = _V329_SCENE.get("holder") or {}
            if h335.get("complete") and not str(h335.get("top_z_source", "measured")).startswith("measured"):
                _V329_SCENE["height_unmeasured_scans"] = int(_V329_SCENE.get("height_unmeasured_scans", 0)) + 1
                skill_core.emit("[V3.35 HOLDER HEIGHT] carton top height could not be measured by parallax in this "
                                "scan -> using the carton model on the table plane (flagged in the scene)")
        holder_xy = [sum(float(c[0]) for c in EGG_SLOT_POSES.values()) / len(EGG_SLOT_POSES),
                     sum(float(c[1]) for c in EGG_SLOT_POSES.values()) / len(EGG_SLOT_POSES)]
        z_ref = sum(float(c[2]) for c in EGG_SLOT_POSES.values()) / len(EGG_SLOT_POSES)
        last_objs = (V311_STATE.get("last") or {}).get("objects") or {}
        pts = _v326w.look_points(label, last_objs, EGG_BASKET["center_xy_m"], holder_xy, z_ref)
        if holding:
            pts = [("current_pose", None)]
        if scanned and not str(label).startswith(("plan_grasp", "verify")):
            pts = []                                   # V3.29: the scan itself was this survey
        if _v329_no_prior() and not V329_LAYOUT_STATE.get("holder_ok"):
            pts = [(v, p) for v, p in pts if v != "holder"]   # never look at a holder position nobody perceived
        if _v329_no_prior() and not _V329_SCENE.get("pile"):
            pts = [(v, p) for v, p in pts if v != "basket"]   # nor at an egg pile nobody perceived
        lost337 = dict(_V329_EST.get("lost") or {}) if (_v329_no_prior() and not holding) else {}
        if lost337 and str(label).startswith("observe_scene"):
            # V3.37 SEARCH: look at the drop point(s) first, then a ring around it, before the usual views
            r337 = float(os.environ.get("NERO_V337_SEARCH_R_M", "0.12"))
            nmax337 = int(os.environ.get("NERO_V337_SEARCH_MAX_VIEWS", "5"))
            extra337 = []
            for lname, info in lost337.items():
                base = list(info.get("search_xy") or [])
                ring = []
                if base:
                    cx, cy = base[0]
                    ring = [[cx + r337 * math.cos(a), cy + r337 * math.sin(a)] for a in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)]
                k = 0
                for xy in base + ring:
                    x = min(-0.15, max(-0.62, float(xy[0])))
                    y = min(0.33, max(-0.33, float(xy[1])))
                    if k >= nmax337:
                        break
                    extra337.append((f"lost_{lname}_{k}", [x, y, 0.02]))
                    k += 1
            skill_core.emit(f"[V3.37 SEARCH] {label} | lost {sorted(lost337)} | {len(extra337)} search view(s) first")
            pts = extra337 + list(pts)
        found337 = set()
        for vname, pt in pts:
            if vname.startswith("lost_"):
                ln337 = vname[len("lost_"):].rsplit("_", 1)[0]
                if ln337 in found337 or ln337 not in (_V329_EST.get("lost") or {}):
                    continue                               # already found -> skip the rest of its search views
            if pt is not None:
                try:
                    _v326_move_look(pt, f"{label}/{vname}")
                except Exception as exc326:
                    look_failed.append(vname)
                    skill_core.emit(f"[V3.26 WRIST LOOK WARN] {label}/{vname} | look move failed ({exc326}); "
                                    "capturing from the current pose")
            p = _v335_capture(f"{label}/{vname}")
            payloads.append(p)
            views.append(vname)
            if _v329_no_prior():
                _v329_ingest(p)
            for ln337 in list(lost337):
                o337 = (p.get("objects") or {}).get(ln337) or {}
                if o337.get("seen") and o337.get("trusted"):
                    found337.add(ln337)
                    skill_core.emit(f"[V3.37 SEARCH HIT] {label}/{vname} | {ln337} perceived at "
                                    f"{[round(float(v), 3) for v in o337.get('xyz_m') or []]}")
            skill_core.emit(f"[V3.26 WRIST VIEW] {label}/{vname} | capture #{p.get('capture')} | eggs in view "
                            f"{p.get('eggs_in_view')} | cam {p.get('cam_pos')} | wait {p.get('wait_s')}s")
            _tr328 = p.get("tracking") or {}
            skill_core.emit(f"[V3.28 WRIST TRACK] {label}/{vname} | seen "
                            f"{sorted(n for n, o in (p.get('objects') or {}).items() if o.get('seen'))} | out of view "
                            f"{_tr328.get('out_of_view')} | held hint {_tr328.get('held_hint')} | placed hint "
                            f"{_tr328.get('placed_hint')} | re-associated {_tr328.get('reassociated_moved')}")
        payload = _v326w.merge_payloads(payloads, views)
        obj339 = (list(names)[0] if names else str(label).partition(":")[2]) if str(label).startswith("plan_grasp") else ""
        if (payload is not None and obj339 and not holding and _v339_rgb_only()
                and os.environ.get("NERO_V339_GRASP_VIEWS", "1") != "0"):
            # V3.39: the grasp view is not good enough -> more close views RIGHT HERE (no q0 round trip, no separate
            # recapture skill step); each adds a ray for the triangulation
            nmax339 = int(os.environ.get("NERO_V339_GRASP_EXTRA_VIEWS", "2"))
            for k339 in range(nmax339):
                o339 = (payload.get("objects") or {}).get(obj339)
                why339 = _v339_grasp_issue(o339, payload)
                if why339 is None:
                    if k339 > 0:
                        skill_core.emit(f"[V3.39 GRASP VIEW OK] {label} | {obj339} measured + consistent after "
                                        f"{k339} extra close view(s)")
                    break
                if not o339 or not o339.get("xyz_m"):
                    break
                tx, ty = float(o339["xyz_m"][0]), float(o339["xyz_m"][1])
                if k339 % 2 == 1:                          # second extra view: shifted -> another ray direction
                    r339 = math.hypot(tx, ty) or 1.0
                    tx, ty = tx - 0.05 * tx / r339, ty - 0.05 * ty / r339
                skill_core.emit(f"[V3.39 GRASP VIEW] {label} | {obj339}: {why339} -> extra close view {k339 + 1}/{nmax339}")
                vn339 = f"close_{obj339}_{k339 + 1}"
                try:
                    _v326_move_look([tx, ty, 0.03], f"{label}/{vn339}",
                                    distances=([0.36, 0.40] if k339 % 2 == 0 else [0.40, 0.45]), no_detour=True)
                except Exception as exc339:
                    look_failed.append(vn339)
                    skill_core.emit(f"[V3.26 WRIST LOOK WARN] {label}/{vn339} | look move failed ({str(exc339)[:160]})")
                    continue
                p339 = _v335_capture(f"{label}/{vn339}")
                payloads.append(p339); views.append(vn339)
                if _v329_no_prior():
                    _v329_ingest(p339)
                o339b = (p339.get("objects") or {}).get(obj339) or {}
                skill_core.emit(f"[V3.28 WRIST TRACK] {label}/{vn339} | seen "
                                f"{sorted(n for n, o in (p339.get('objects') or {}).items() if o.get('seen'))} | "
                                f"{obj339} {(o339b.get('measurement') or {}).get('method')} "
                                f"{[round(float(v), 4) for v in o339b.get('xyz_m') or []]} | predicted assoc "
                                f"{(p339.get('tracking') or {}).get('predicted_assoc')}")
                payload = _v326w.merge_payloads(payloads, views)
            else:
                o339 = (payload.get("objects") or {}).get(obj339)
                why339 = _v339_grasp_issue(o339, payload)
                if why339:
                    skill_core.emit(f"[V3.39 GRASP VIEW WARN] {label} | {obj339}: {why339} after {nmax339} extra view(s)")
        if payload is not None and _v329_no_prior():
            V311_STATE["last"] = payload
            _v329_apply()                                  # egg pile from the merged view
        if payload is not None and look_failed:
            payload["look_failed"] = list(look_failed)        # V3.27: views captured from a non-look pose
        V311_STATE["last"] = payload
        _v311_log_vs_truth(f"{label} (wrist, views {views})", names or EGG_OBJECTS)
        if str(label).startswith("plan_grasp") and not holding:
            # grasp planning starts from SAFE_NEUTRAL q0 as with the fixed camera (keeps the V3.07 start-state logic valid)
            try:
                _v307_go_q0("V3.26 WRIST LOOK DONE")
            except Exception as exc326q:
                skill_core.emit(f"[V3.26 WRIST LOOK WARN] {label} | return to q0 failed ({exc326q}); planning from the look pose")
        return payload

    def _v332_known_eggs():
        """V3.32: the eggs of THIS episode. With priors (fixed V3.11 / truth): the registered list. Without priors
        (wrist / V3.31 fixed): only eggs perception has actually named in this episode (+ held / released ones) -
        a name in the pool that was never perceived (e.g. an egg that fell off the table) does not exist."""
        if not _v329_no_prior():
            return list(EGG_OBJECTS)
        known = set(str(n) for n in (_V329_SCENE.get("seen") or ()))
        known |= set(str(n) for n in (_V329_EST.get("released") or {}))
        known |= set(str(n) for n in (V271_PLACEMENT_RUNTIME.get("released") or {}))
        try:
            h332 = _v3121_held_name()
            if h332:
                known.add(str(h332))
        except Exception:
            pass
        pool = list(EGG_OBJECTS) + [n for n in os.environ.get("NERO_V330_OBJECT_NAMES", "").split(",")
                                    if n and n not in EGG_OBJECTS]
        return [n for n in pool if n in known] + sorted(n for n in known if n not in pool)

    def _v332_prune(obj, known):
        """V3.32: drop never-perceived egg names (keys / list items egg_N) from a tool result."""
        import re as _re
        pat = _re.compile(r"^egg_\d+$")
        if isinstance(obj, dict):
            return {k: _v332_prune(v, known) for k, v in obj.items()
                    if not (isinstance(k, str) and pat.match(k) and k not in known)}
        if isinstance(obj, list):
            return [_v332_prune(v, known) for v in obj
                    if not (isinstance(v, str) and pat.match(v) and v not in known)]
        return obj

    def llm_scene_snapshot():
        if skill_core is None:
            raise RuntimeError(f"V69 skill core unavailable: {skill_core_error}")
        tcp_xyz, tcp_q = skill_core.current_tcp_pose()
        objects = {}
        cam311 = v311_mode() == "camera"
        if v326_wrist_selected():
            try:
                _v335_measured_geometry()          # V3.35: aperture from the camera-measured egg size
            except Exception:
                pass
        for name in _v332_known_eggs():                 # V3.32: no-prior modes list perceived eggs only
            try:
                if cam311 and _v3121_held_name() == name:
                    # V3.12.1: the camera cannot see a held egg (and the tracker keeps its old basket pose).
                    est3121 = _v3121_held_pose_estimate(name, tcp_xyz, tcp_q)
                    if est3121 is not None:
                        objects[name] = {
                            "category": "egg",
                            "xyz_m": [round(float(v), 5) for v in est3121[0]],
                            "quat_xyzw": [round(float(v), 5) for v in est3121[1]],
                            "location": "in_gripper",
                            "grasp_geometry": _v24_gripper_prior(name, float(gripper_actual_mm.value)),
                            "perception": {"source": "tcp_x_grasp_transform", "seen": False, "trusted": True,
                                           "confidence": None, "held": True,
                                           "grasp_sensed_by_gripper": bool(hold_active_shared.value),
                                           "in_hand_camera_check": (_v335_inhand_verdict() if v326_wrist_selected() else None)},
                        }
                        continue
                if cam311:
                    o311 = _v311_obj_entry(name)
                    if o311 is None:
                        objects[name] = {"category": "egg", "location": "not_perceived",
                                         "perception": _v311_perception_tag(None)}
                        continue
                    xyz, quat = [float(v) for v in o311["xyz_m"]], [float(v) for v in o311["quat_xyzw"]]
                else:
                    xyz, quat = skill_core.object_pose(name)
                opening_now = float(gripper_actual_mm.value)
                objects[name] = {
                    "category": "egg",
                    "xyz_m": [round(float(v), 5) for v in xyz],
                    "quat_xyzw": [round(float(v), 5) for v in quat],
                    "location": llm_classify_object_location(xyz),
                    "grasp_geometry": _v24_gripper_prior(name, opening_now),
                }
                if cam311:
                    objects[name]["perception"] = _v311_perception_tag(o311)
            except Exception as exc:
                objects[name] = {"category": "egg", "error": str(exc), "location": "unknown"}

        slot_state = {}
        _unknown331 = bool(V329_LAYOUT_STATE.get("perceived_required") and not V329_LAYOUT_STATE.get("holder_ok"))
        for slot_name, center in ({} if _unknown331 else EGG_SLOT_POSES).items():
            occupied_by = None
            for obj_name, info in objects.items():
                if info.get("location") == f"egg_holder.{slot_name}":
                    occupied_by = obj_name
                    break
            slot_state[slot_name] = {
                "center_xyz_m": [round(float(v), 5) for v in center],
                "entry_axis": [0.0, 0.0, -1.0],
                "occupied": occupied_by is not None,
                "occupied_by": occupied_by,
            }

        return {
            "tcp": {
                "xyz_m": [round(float(v), 5) for v in tcp_xyz],
                "quat_xyzw": [round(float(v), 5) for v in tcp_q],
            },
            "semantic_scene": {
                "task_family": "fill_egg_holder",
                "source_container": "egg_basket",
                "manipulable_category": "egg",
                "destination_receptacle": "egg_holder",
                "note": "Eggs are rigid in this baseline; fragility/tactile damage is not enabled yet.",
            },
            "objects": objects,
            "objects_policy": ("perceived_only" if _v329_no_prior() else "registered"),
            "pose_source": (v335_pose_source_label() if cam311 else "mujoco_ground_truth"),
            **({"camera_v335": {
                "calibration_error_level": V335_STATE.get("calib_level"),
                "calibration": (V335_STATE.get("last_calib") or {}).get("summary"),
                "measured": V335_STATE.get("last_measured"),
                "live_refresh_hz": (V335_STATE.get("live_hz") if V335_STATE.get("live_on") else 0),
                "grasp_physics": "V3.34 sim grasp (pads mu 10 + auto CONNECT hold); V3.37: an automatic safety restore keeps the held egg in the gripper",
                "plausibility_note": ("egg sizes are camera measurements (measurement.width_mm / length_mm); "
                                      "judge yourself whether an object is a hen's egg"),
            }} if v326_wrist_selected() else {}),
            "containers": {
                "egg_basket": {
                    "category": "source_container",
                    "center_xy_m": (None if (V329_LAYOUT_STATE.get("perceived_required") and not V329_LAYOUT_STATE.get("pile_ok"))
                                    else [round(float(v), 5) for v in EGG_BASKET["center_xy_m"]]),
                    "contains": [n for n, info in objects.items() if info.get("location") == "egg_basket"],
                    **({"perception": _v329_pile_tag()} if _v329_pile_tag() else {}),
                }
            },
            "receptacles": {
                "egg_holder": {
                    "category": "slotted_receptacle",
                    "center_xyz_m": (None if _unknown331 else [round(float(v), 5) for v in EGG_HOLDER["center_xyz_m"]]),
                    "lid_state": "open_static_baseline",
                    "slot_count": 8,
                    "slots": slot_state,
                    **({"perception": _v329_holder_tag()} if _v329_holder_tag() else {}),
                }
            },
            "gripper": {
                "actual_opening_mm": round(float(gripper_actual_mm.value), 2),
                "position_target_opening_mm": round(float(gripper_target_mm.value), 2),
                "force_mode": bool(gripper_force_mode.value),
                "target_force_per_finger_N": round(float(gripper_force_N.value), 3),
                "contact_left_N": round(float(contact_left_n_shared.value), 3),
                "contact_right_N": round(float(contact_right_n_shared.value), 3),
                "hold_enabled": bool(hold_enabled_shared.value),
                "hold_active": bool(hold_active_shared.value),
            },
            "safety": {
                "trip_count": int(safety_trip_count.value),
                "state_code": int(safety_state.value),
            },
            "safety_recovery": {
                "last_context": dict(V242_LAST_SAFETY_CONTEXT),
                "retry_used_by_object": dict(V242_SAFETY_RETRY_USED),
                "planning_retry_used_by_object": dict(V251_PLANNING_RETRY_USED),
                "grasp_z_retry_bonus_mm_by_object": {k: round(float(v) * 1000.0, 1) for k, v in V251_GRASP_Z_RETRY_BONUS_M.items()},
                "policy": "one explicit safety recovery + one non-safety planning adaptation per object",
            },
        }

    def llm_require_arm_controller():
        try:
            ready = bool(skill_core.arm_controller_client.server_is_ready())
        except Exception:
            ready = False
        if not ready:
            raise RuntimeError("/arm_controller/follow_joint_trajectory unavailable")

    # ========================================================
    # NERO V2-7 SKILL CONTRACT + REFLECTIVE SAFETY RECOVERY
    # ========================================================
    # V69 remains byte-for-byte frozen. V2-7 wraps each V69 skill in a task-
    # semantic contract. The Agent sees the same semantics in its tool schema
    # BEFORE choosing an action. After execution, deterministic MuJoCo state
    # verifies the contract and returns a normalized result.

    V24_STAGE_IDLE = "IDLE"
    V24_STAGE_OBSERVED = "OBSERVED"
    V24_STAGE_HOVER = "HOVER_REACHED"
    V24_STAGE_GRASP_POSE = "GRASP_POSE_REACHED"
    V24_STAGE_GRASPED = "GRASP_ESTABLISHED"
    V24_STAGE_LIFTED = "LIFTED"
    V24_STAGE_BIN_HOVER = "RECEPTACLE_HOVER_REACHED"
    V24_STAGE_SLOT_PLACE = "SLOT_PLACE_REACHED"
    V24_STAGE_RELEASED = "RELEASED"
    V24_STAGE_VERIFIED = "PLACEMENT_VERIFIED"
    V24_STAGE_SAFETY = "SAFETY_STOPPED"

    v24_state_lock = threading.RLock()
    v24_skill_state = {
        "stage": V24_STAGE_IDLE,
        "active_object": None,
        "target_bin": None,  # legacy internal destination key
        "target_receptacle": None,
        "target_slot": None,
        "last_tool": None,
        "last_ok": None,
    }

    def _v24_state_snapshot():
        with v24_state_lock:
            return dict(v24_skill_state)

    def _v24_set_state(stage=None, active_object="__KEEP__", target_bin="__KEEP__", last_tool=None, last_ok=None):
        with v24_state_lock:
            if stage is not None:
                v24_skill_state["stage"] = str(stage)
            if active_object != "__KEEP__":
                v24_skill_state["active_object"] = active_object
            if target_bin != "__KEEP__":
                v24_skill_state["target_bin"] = target_bin
                if isinstance(target_bin, str) and target_bin.startswith("egg_holder."):
                    v24_skill_state["target_receptacle"] = "egg_holder"
                    v24_skill_state["target_slot"] = target_bin.split(".", 1)[1]
                elif target_bin is None:
                    v24_skill_state["target_receptacle"] = None
                    v24_skill_state["target_slot"] = None
            if last_tool is not None:
                v24_skill_state["last_tool"] = str(last_tool)
            if last_ok is not None:
                v24_skill_state["last_ok"] = bool(last_ok)
            return dict(v24_skill_state)

    def _v24_reset_skill_state(reason="manual reset", clear_adaptation=True):
        """Create a clean task episode boundary.

        Physical ZERO/HARD RESET and planner state must agree.  Without this
        reset, a robot that has physically returned home could still be labelled
        ``HOVER_REACHED`` from a previous task and the LLM might incorrectly call
        ``approach`` before a fresh ``move_above``.  Retry counters are therefore
        task-local and reset together with the semantic state.

        Reset the logical episode state so physical pose and planner state cannot diverge.

        V2-5.1 reset only the safety retry budget, which allowed stale
        HOVER_REACHED state and an already-consumed planning retry to leak into
        the next RUN TASK after ZERO ALL.  V2-7 treats these as task-local.
        """
        with v24_state_lock:
            v24_skill_state.update({
                "stage": V24_STAGE_IDLE,
                "active_object": None,
                "target_bin": None,
                "target_receptacle": None,
                "target_slot": None,
                "last_tool": None,
                "last_ok": None,
            })
        if clear_adaptation:
            try:                               # V3.12.1: return budget is per task episode
                V3121_RETURNS.clear()
            except NameError:
                pass

        for _obj in V242_SAFETY_RETRY_USED:
            V242_SAFETY_RETRY_USED[_obj] = 0
        for _obj in V251_PLANNING_RETRY_USED:
            V251_PLANNING_RETRY_USED[_obj] = 0
        for _obj in V251_GRASP_Z_RETRY_BONUS_M:
            V251_GRASP_Z_RETRY_BONUS_M[_obj] = 0.0
        if clear_adaptation:
            # Retry aperture widening is task-local.  A future learned prior
            # should live in a separate persistent memory, not in failure state.
            for _obj in V241_APERTURE_RETRY_BONUS_MM:
                V241_APERTURE_RETRY_BONUS_MM[_obj] = 0.0

        _v27_reset_grasp_runtime(OBJECTS)
        _v271_reset_placement_runtime(OBJECTS)
        _v242_clear_safety_context()
        V305_EXEC_FAULT.update({"consecutive": 0, "total": 0, "last": None, "in_recovery": False})
        V304_PREGRASP_CACHE.clear()
        try:
            V307_RESET_USED.clear()
        except NameError:
            pass
        try:
            if str(reason).startswith("new RUN TASK"):
                _v329_reset_scene(str(reason))          # V3.29: every task starts without a scene layout
        except Exception as exc329:
            skill_core.emit(f"[V3.29 SCENE RESET WARN] {exc329}")
        skill_core.emit(
            f"[V2-7 EPISODE RESET] {reason} -> IDLE | "
            "safety_retry=0 planning_retry=0 grasp_z_bonus=0 "
            + ("aperture_bonus=0" if clear_adaptation else "aperture_bonus=preserved")
        )
        return _v24_state_snapshot()

    def _v24_norm3(v):
        return math.sqrt(sum(float(x) * float(x) for x in v))

    def _v24_sub3(a, b):
        return [float(a[i]) - float(b[i]) for i in range(3)]

    def _v24_xyz_distance(a, b):
        return _v24_norm3(_v24_sub3(a, b))

    def _v24_xy_distance(a, b):
        return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))

    def _v24_contract(name, args):
        state = _v24_state_snapshot()
        obj = str(args.get("object_name", state.get("active_object") or "<active_object>"))
        bin_name = str(args.get("bin_name", state.get("target_bin") or "<target_bin>"))
        opening = float(gripper_actual_mm.value)
        prior = _v24_gripper_prior(obj, opening) if obj in V24_OBJECT_GEOMETRY else {}
        rr = prior.get("recommended_opening_range_mm", ["?", "?"])
        target = prior.get("recommended_target_opening_mm", "?")
        contracts = {
            "observe_scene": {
                "purpose": ("Perceive the scene with the selected camera (fixed D435i RGB-D / wrist D435i / wrist USB RGB-only) and report object poses, gripper opening, geometry/manipulation priors and state without moving anything."
                            if v311_mode() == "camera" else
                            "Read exact MuJoCo scene, continuous gripper opening, object geometry/manipulation priors, and V2-7 state without moving anything."),
                "preconditions": ["none"],
                "expected_effects": ["fresh scene/state snapshot"],
                "expected_non_effects": ["robot does not move", "gripper does not move", "objects do not move"],
                "success_criteria": ["scene snapshot returned"],
            },
            "recover_from_safety": {
                "purpose": "Explicitly clear a latched PRE-GRASP safety stop, retreat vertically away from the object, and prepare exactly one reasoned retry.",
                "preconditions": [
                    "state=SAFETY_STOPPED",
                    "no object is held",
                    "a structured safety context exists",
                    "this object has not already consumed its one safety retry",
                ],
                "expected_effects": [
                    "V69 safety latch is explicitly cleared",
                    "TCP retreats upward about 90 mm before any aperture narrowing",
                    "selected recovery strategy updates the manipulation prior when appropriate",
                ],
                "expected_non_effects": [
                    "no grasp is created",
                    "the failed approach is NOT blindly resumed from the contact pose",
                ],
                "success_criteria": [
                    "safety latch explicitly recovered",
                    "vertical retreat is achieved without a new safety trip",
                    "one-retry budget is consumed and a fresh re-observation is requested",
                ],
            },
            "set_gripper_opening": {
                "purpose": f"Set pre-grasp gripper aperture to {float(args.get('target_opening_mm', opening)):.1f} mm in position mode.",
                "preconditions": ["hold_active=false", "not transporting a grasped object", "normally called after observe_scene"],
                "expected_effects": ["actual gripper opening converges to requested millimeter target"],
                "expected_non_effects": ["arm TCP does not move", "objects do not move", "no grasp is created"],
                "success_criteria": ["actual opening within 3 mm of target", "hold_active=false", "no new safety trip"],
            },

            "reposition_egg_inward": {
                "purpose": f"Gently nudge edge-limited {obj} toward the egg-basket center in small feedback-controlled increments, then recover the arm to a high neutral/q0-safe posture before forcing a fresh observation/replan.",
                "preconditions": [
                    "state=OBSERVED",
                    "hold_active=false",
                    "latest plan_grasp failed primarily because every untried candidate violates basket-rim projected clearance",
                    "this egg has not already consumed its one V2-29 edge-reposition attempt",
                ],
                "expected_effects": [
                    "arm first retreats vertically to a safe transit height, then the gripper closes to a compact near-closed opening; the arm traverses at high Z above the target and descends vertically with the two fingers acting as one narrow radial pusher",
                    "egg is pushed inward in 8-10 mm then 5-8 mm feedback-controlled segments rather than one long sweep",
                    "egg XY/radius is re-read after every segment and pushing stops as soon as enough inward gain is achieved",
                    "arm retreats high after contact, then recovers toward a neutral joint posture; full q=0 is the validated slow fallback",
                    "old grasp candidates are invalidated and a fresh observe_scene is mandatory before replanning",
                ],
                "expected_non_effects": [
                    "no grasp/hold is created",
                    "no direct pickup is attempted inside this skill",
                    "no old grasp candidate is reused after the egg moves",
                    "V2-24 pickup/orient/place/release behavior is unchanged",
                ],
                "success_criteria": [
                    "feedback-measured inward gain is sufficient to recover the previous basket-clearance deficit, or egg enters the center-safe radius",
                    "no new safety trip",
                    "hold_active remains false",
                    "high-neutral or q0-safe posture recovery succeeds",
                    "fresh observe_scene is required before any new plan_grasp",
                ],
            },
            "plan_grasp": {
                "purpose": f"Generate and preflight bounded constraint-aware grasp candidates for {obj} without moving the robot.",
                "preconditions": [
                    "fresh scene observation available",
                    "no object is currently held",
                    f"gripper aperture is already prepared for {obj}",
                ],
                "expected_effects": [
                    "estimate egg major-axis orientation from quaternion",
                    "generate bounded wrist-yaw / small inward-offset candidates",
                    "analytically reject candidates whose projected fingers intersect the basket rim",
                    "MoveIt-preflight remaining pregrasp poses without execution",
                ],
                "expected_non_effects": ["robot does not move", "egg does not move", "gripper does not move"],
                "success_criteria": ["at least one untried candidate is geometry-feasible and MoveIt-pregrasp-feasible"],
            },
            "move_to_pregrasp": {
                "purpose": f"Execute one previously planned V2-7 pregrasp candidate for {obj}.",
                "preconditions": ["plan_grasp succeeded", "candidate_id belongs to latest fresh plan", "object pose has not become stale"],
                "expected_effects": ["TCP reaches candidate pregrasp XYZ with candidate wrist yaw"],
                "expected_non_effects": [f"{obj} should remain stationary", "no grasp/contact should occur"],
                "success_criteria": ["TCP reaches pregrasp tolerance", "no new safety trip", "object remains approximately stationary"],
            },
            "approach_planned_grasp": {
                "purpose": f"Collision-checked straight descent from a V2-7 pregrasp candidate to its grasp pose for {obj}.",
                "preconditions": ["same candidate reached pregrasp", "candidate remains current", "gripper aperture remains prepared"],
                "expected_effects": ["TCP descends with fixed candidate wrist yaw to the planned grasp pose"],
                "expected_non_effects": ["no clamp yet; small incidental target motion is tolerated and checked after approach"],
                "success_criteria": ["TCP reaches grasp target", "fresh post-approach observation confirms target remains between fingers OR robot retreats before replanning", "no safety trip"],
            },

            "plan_placement": {
                "purpose": f"Plan long-axis placement of held {obj} into an egg-holder slot without moving the robot.",
                "preconditions": ["state=LIFTED", "hold_active=true", "target egg-holder slot exists and is empty"],
                "expected_effects": [
                    "measure held egg long-axis orientation",
                    "compute the smaller rotation that aligns the egg long axis with the slot entry axis",
                    "choose a safe mid-workspace orientation pose",
                    "compute slot-entry and pre-release capture depths",
                ],
                "expected_non_effects": ["robot does not move", "egg does not move", "gripper does not open"],
                "success_criteria": ["valid slot and held egg", "safe orientation/entry/release targets produced"],
            },
            "orient_for_slot": {
                "purpose": f"Reorient held {obj} with sparse-wrist-first free-path planning, then move it above the slot without prescribing the intermediate egg-center path.",
                "preconditions": ["plan_placement succeeded", "hold_active=true", "state=LIFTED"],
                "expected_effects": [
                    "transport the held egg to a high obstacle-free waypoint",
                    "rotate at approximately fixed TCP XYZ in ~10-degree bounded orientation steps",
                    "use current-pose-seeded BioIK; reject excessive J1-J5 branches before execution; J6/J7 carry most reorientation when feasible",
                ],
                "expected_non_effects": ["egg must not be released", "orientation change is NOT performed at the slot mouth"],
                "success_criteria": ["egg long-axis error <= 10 deg", "hold remains active", "no safety trip"],
            },
            "move_to_slot_entry": {
                "purpose": f"Move vertically oriented held {obj} to a precise hover/capture entry above the selected slot.",
                "preconditions": ["state=PLACEMENT_ORIENTED", "hold_active=true", "egg long axis aligned to slot axis"],
                "expected_effects": ["egg center aligns to slot XY while orientation is preserved"],
                "expected_non_effects": ["no release", "no large wrist reorientation near the holder"],
                "success_criteria": ["egg center XY error <= 7 mm", "axis error <= 10 deg", "hold active", "no safety trip"],
            },
            "guarded_insert_and_release": {
                "purpose": f"Slowly insert held {obj} along its LONG axis to a safe capture depth, then gently release and retreat.",
                "preconditions": ["state=SLOT_ENTRY_ALIGNED", "hold_active=true", "entry axis error <= 8 deg", "entry XY error <= 8 mm"],
                "expected_effects": [
                    "slower, finer segmented Cartesian descent: 5 mm coarse then 2.5 mm fine steps",
                    "continuous XY/orientation/hold/safety/joint-tracking guards after every segment",
                    "stop above the final slot center so the fingers do not need to enter the holder walls",
                    "staged opening/release, then vertical gripper retreat",
                ],
                "expected_non_effects": ["no high-speed plunge", "no late large wrist rotation", "this skill does not replace final verify_object_in_slot"],
                "success_criteria": ["capture depth reached safely", "grasp released", "gripper retreats vertically", "final placement is left to verify_object_in_slot"],
            },
            "move_above": {
                "purpose": f"Move TCP to a top-down hover about 120 mm above {obj}.",
                "preconditions": [
                    "no safety trip",
                    "not currently transporting a held object",
                    f"continuous gripper opening should be within robust range {rr} mm for {obj}; preferred target={target} mm",
                ],
                "expected_effects": ["TCP moves to object XY and hover Z"],
                "expected_non_effects": [f"{obj} should remain stationary", "gripper stays unchanged", "no grasp is expected"],
                "success_criteria": ["TCP within 12 mm of hover target", "no new safety trip", "no large unexpected object displacement"],
            },
            "approach": {
                "purpose": f"Descend from hover to the grasp pose for {obj}.",
                "preconditions": [
                    "state=HOVER_REACHED", f"active_object={obj}",
                    f"actual gripper opening remains within robust range {rr} mm for {obj}",
                ],
                "expected_effects": ["TCP descends to object grasp pose without touching/pushing the object before clamp"],
                "expected_non_effects": [f"{obj} should remain approximately stationary during pre-grasp descent", "grasp/hold is not required yet"],
                "success_criteria": ["TCP within 10 mm of grasp target", "object displacement during approach <= 4 mm", "no new safety trip"],
            },
            "clamp_gripper": {
                "purpose": f"Close gripper and establish provisional contact/hold on {obj}.",
                "preconditions": ["state=GRASP_POSE_REACHED", "active object is known"],
                "expected_effects": ["gripper opening decreases", "two-finger contact or hold becomes established"],
                "expected_non_effects": [f"{obj} does not need to translate yet"],
                "success_criteria": ["two-finger contact OR hold_active"],
            },
            "lift": {
                "purpose": f"Lift {obj} vertically with the gripper and physically validate the grasp.",
                "preconditions": ["state=GRASP_ESTABLISHED", "active object is known"],
                "expected_effects": ["TCP rises", f"{obj} MUST rise with TCP", "object-to-TCP relation remains approximately stable"],
                "expected_non_effects": ["object must NOT stay on table"],
                "success_criteria": ["TCP reaches lift target", "object rises sufficiently", "follow drift within tolerance", "no new safety trip"],
            },
            "move_above_receptacle": {
                "purpose": f"Transport held {obj} above the selected egg-holder slot.",
                "preconditions": ["state=LIFTED", "active object is known", "target slot exists and is empty"],
                "expected_effects": ["TCP and object move together", "held egg is centered above the selected slot"],
                "expected_non_effects": ["grasp must not be lost", "egg is not released yet"],
                "success_criteria": ["TCP reaches a safe slot-hover pose", "object-to-TCP drift within tolerance", "no new safety trip"],
            },
            "place_into_receptacle": {
                "purpose": f"Lower held {obj} from slot hover to the selected egg-holder placement depth while maintaining the grasp.",
                "preconditions": ["state=RECEPTACLE_HOVER_REACHED", "active object and target slot are known"],
                "expected_effects": ["held egg descends into the selected slot", "gripper remains closed/holding"],
                "expected_non_effects": ["object is not released yet", "other eggs should remain undisturbed"],
                "success_criteria": ["held egg center reaches the selected slot placement region", "no new safety trip"],
            },
            "open_gripper": {
                "purpose": f"RELEASE held {obj} in the selected egg-holder slot; this is not the pre-grasp aperture tool.",
                "preconditions": ["state=SLOT_PLACE_REACHED", "active object and target slot are known"],
                "expected_effects": ["gripper opens", "hold becomes false", "egg detaches and settles in the pocket"],
                "expected_non_effects": ["this skill does NOT by itself prove final placement"],
                "success_criteria": ["opening >= 80 mm", "hold_active=false"],
            },
            "return_to_pick_location": {
                "purpose": f"V3.12.1 recovery: put held {obj} back where it was picked, open the gripper and retreat.",
                "preconditions": [f"{obj} is the held active object", "its pick location was recorded at clamp"],
                "expected_effects": ["TCP returns above the pick location", "egg is set down and released", "gripper retreats"],
                "expected_non_effects": ["egg is not dropped from height", "other eggs are not disturbed"],
                "success_criteria": ["hold_active=false", f"{obj} back in egg_basket"],
            },
            "verify_object_in_slot": {
                "purpose": (f"Verify final placement of {obj} in the requested egg-holder slot from a fresh camera capture."
                            if v311_mode() == "camera" else
                            f"Verify final placement of {obj} in the requested egg-holder slot from MuJoCo coordinates."),
                "preconditions": ["normally state=RELEASED"],
                "expected_effects": ["read-only final placement verdict and slot occupancy update"],
                "expected_non_effects": ["robot/object motion not required"],
                "success_criteria": ["actual_location equals requested egg_holder.slot_N", "hold_active=false"],
            },
        }
        return contracts.get(name, {"purpose": name})

    def _v24_log_contract(name, args, contract, state_before):
        skill_core.emit(f"[SKILL CONTRACT] {name} | state_before={state_before.get('stage')} | args={json.dumps(args, ensure_ascii=False)}")
        skill_core.emit("[SKILL CONTRACT PURPOSE] " + str(contract.get("purpose", "")))
        skill_core.emit("[SKILL CONTRACT PRE] " + "; ".join(contract.get("preconditions", [])))
        skill_core.emit("[SKILL CONTRACT EXPECTED EFFECT] " + "; ".join(contract.get("expected_effects", [])))
        skill_core.emit("[SKILL CONTRACT EXPECTED NON-EFFECT] " + "; ".join(contract.get("expected_non_effects", [])))
        skill_core.emit("[SKILL CONTRACT SUCCESS] " + "; ".join(contract.get("success_criteria", [])))

    def _v24_next_actions(stage, ok=True):
        if not ok:
            return ["observe_scene", "replan_or_stop_based_on_failure_reason"]
        return {
            V24_STAGE_IDLE: ["observe_scene"],
            V24_STAGE_OBSERVED: ["plan_grasp"],   # V3.06: plan_grasp prepares the aperture itself
            V24_STAGE_HOVER: ["approach"],
            V24_STAGE_GRASP_POSE: ["clamp_gripper"],
            V24_STAGE_GRASPED: ["lift"],
            V24_STAGE_LIFTED: ["plan_placement"],
            V24_STAGE_BIN_HOVER: ["place_into_receptacle"],
            V24_STAGE_SLOT_PLACE: ["open_gripper"],
            V24_STAGE_RELEASED: ["verify_object_in_slot"],
            V24_STAGE_VERIFIED: ["plan_grasp"],
            V24_STAGE_SAFETY: ["observe_scene", "recover_from_safety"],
        }.get(stage, ["observe_scene"])

    def _v24_precondition_failure(name, args, reason, contract, state_before, physical_verification=None, recommended_next_actions=None):
        skill_core.emit(f"[SKILL PRECONDITION FAILED] {name} | {reason}")
        return {
            "ok": False,
            "tool": name,
            "reason": "PRECONDITION_FAILED",
            "detail": str(reason),
            "state_before": state_before,
            "state_after": _v24_state_snapshot(),
            "contract": contract,
            "physical_verification": dict(physical_verification or {}),
            "warnings": [],
            "recommended_next_actions": list(recommended_next_actions or ["observe_scene", "follow_state_machine"]),
        }

    def _v24_result(name, ok, contract, state_before, verification, warnings=None, extra=None, next_stage=None, active_object="__KEEP__", target_bin="__KEEP__", recommended_next_actions=None):
        warnings = list(warnings or [])
        if ok and next_stage is not None:
            state_after = _v24_set_state(next_stage, active_object=active_object, target_bin=target_bin, last_tool=name, last_ok=True)
        else:
            state_after = _v24_set_state(last_tool=name, last_ok=bool(ok))
        result = {
            "ok": bool(ok),
            "tool": name,
            "state_before": state_before,
            "state_after": state_after,
            "contract": contract,
            "physical_verification": verification,
            "warnings": warnings,
            "recommended_next_actions": list(recommended_next_actions) if recommended_next_actions is not None else _v24_next_actions(state_after.get("stage"), bool(ok)),
        }
        if extra:
            result.update(extra)
        # V3.05: an execution-layer fault is not something another candidate/egg can fix.
        try:
            _ef305 = (verification or {}).get("execution_layer_fault") if isinstance(verification, dict) else None
        except Exception:
            _ef305 = None
        if (not ok) and _ef305:
            result["reason"] = "EXECUTION_LAYER_FAULT"
            result["execution_layer_unhealthy"] = bool(_ef305.get("unhealthy"))
            if _ef305.get("unhealthy"):
                result["recommended_next_actions"] = ["stop_and_report"]
                result["hint"] = ("Execution layer unhealthy (repeated controller/tracking fault). "
                                  "STOP the task now and report; do NOT try other candidates or other eggs.")
            else:
                result["recommended_next_actions"] = ["observe_scene", "retry_same_step_once"]
                result["hint"] = ("Controller/tracking fault, arm returned to SAFE_NEUTRAL q0. Observe, then redo the "
                                  "same step for the SAME egg once (plan_grasp again if it was a pregrasp move). "
                                  "Do not switch eggs and do not change the gripper opening because of this.")
        skill_core.emit("[PHYSICAL VERIFY] " + name + " | " + json.dumps(verification, ensure_ascii=False, separators=(",", ":")))
        if warnings:
            skill_core.emit("[SKILL WARNINGS] " + name + " | " + " | ".join(str(x) for x in warnings))
        skill_core.emit(
            f"[SKILL RESULT] {name} | {'SUCCESS' if ok else 'FAILED'} | "
            f"{state_before.get('stage')} -> {state_after.get('stage')} | next={','.join(result['recommended_next_actions'])}"
        )
        return result

    def _v24_aperture_check(object_name):
        opening = float(gripper_actual_mm.value)
        prior = _v24_gripper_prior(object_name, opening)
        rec_min, rec_max = [float(v) for v in prior["recommended_opening_range_mm"]]
        rec_target = float(prior["recommended_target_opening_mm"])
        width = float(prior["effective_width_mm"])
        robust = rec_min <= opening <= rec_max
        if opening < rec_min:
            status = "TOO_NARROW"
        elif opening > rec_max:
            status = "UNNECESSARILY_WIDE"
        else:
            status = "ROBUST_RANGE"
        return {
            "object_name": str(object_name),
            "actual_opening_mm": round(opening, 2),
            "effective_width_mm": round(width, 2),
            "geometric_clearance_total_mm": round(opening - width, 2),
            "estimated_free_clearance_total_mm": prior.get("estimated_free_clearance_total_mm"),
            "estimated_free_clearance_per_side_mm": prior.get("estimated_free_clearance_per_side_mm"),
            "recommended_opening_range_mm": [round(rec_min, 2), round(rec_max, 2)],
            "recommended_target_opening_mm": round(rec_target, 2),
            "adaptive_retry_bonus_mm": prior.get("adaptive_retry_bonus_mm", 0.0),
            "aperture_status": status,
            "robust_aperture_ok": bool(robust),
        }

    # V2-18 helpers.  Frozen V69 remains byte-for-byte unchanged.
    def _v218_slow_trajectory_in_place(traj, factor=2.0):
        """Stretch trajectory timing; factor=2 means about 50% execution speed."""
        factor = max(1.0, float(factor))
        points = list(getattr(getattr(traj, "joint_trajectory", None), "points", []) or [])
        for pt in points:
            old_t = float(pt.time_from_start.sec) + float(pt.time_from_start.nanosec) * 1e-9
            new_t = old_t * factor
            sec = int(new_t)
            nsec = int(round((new_t-sec)*1e9))
            if nsec >= 1_000_000_000:
                sec += 1; nsec -= 1_000_000_000
            pt.time_from_start.sec = sec
            pt.time_from_start.nanosec = nsec
            if getattr(pt, "velocities", None):
                pt.velocities = [float(v)/factor for v in pt.velocities]
            if getattr(pt, "accelerations", None):
                pt.accelerations = [float(a)/(factor*factor) for a in pt.accelerations]
        return traj

    def _v218_post_approach_observe(object_name, cand, plan):
        """Fresh deterministic scene read: is target still inside the finger capture zone?"""
        tcp_xyz, _ = skill_core.current_tcp_pose()
        obj_xyz, _ = skill_core.object_pose(object_name)
        opening_mm = float(gripper_actual_mm.value)
        dx = float(obj_xyz[0]) - float(tcp_xyz[0])
        dy = float(obj_xyz[1]) - float(tcp_xyz[1])
        gyaw = math.radians(float(cand.get("gripper_yaw_deg", 0.0)))
        cyaw = math.radians(float(cand.get("closing_axis_yaw_deg", float(cand.get("gripper_yaw_deg",0.0))+90.0)))
        lateral_mm = abs((dx*math.cos(cyaw) + dy*math.sin(cyaw))*1000.0)
        longitudinal_mm = abs((dx*math.cos(gyaw) + dy*math.sin(gyaw))*1000.0)
        expected_dz_mm = (float(cand["grasp_xyz_m"][2]) - float(plan["object_xyz_at_plan_m"][2]))*1000.0
        actual_dz_mm = (float(tcp_xyz[2]) - float(obj_xyz[2]))*1000.0
        # Center-zone check is intentionally stricter than merely being inside
        # the full finger opening.  It asks whether the egg is still plausibly
        # capturable without a new arm reposition.
        lateral_limit_mm = max(8.0, min(16.0, 0.30*opening_mm))
        longitudinal_limit_mm = 18.0
        vertical_error_mm = abs(actual_dz_mm - expected_dz_mm)
        between = bool(
            lateral_mm <= lateral_limit_mm
            and longitudinal_mm <= longitudinal_limit_mm
            and vertical_error_mm <= 12.0
        )
        return {
            "fresh_scene_read": True,
            "target_between_fingers": between,
            "opening_mm": round(opening_mm,2),
            "target_lateral_from_gripper_center_mm": round(lateral_mm,2),
            "lateral_limit_mm": round(lateral_limit_mm,2),
            "target_longitudinal_offset_mm": round(longitudinal_mm,2),
            "longitudinal_limit_mm": round(longitudinal_limit_mm,2),
            "actual_tcp_minus_object_z_mm": round(actual_dz_mm,2),
            "expected_tcp_minus_object_z_mm": round(expected_dz_mm,2),
            "vertical_relation_error_mm": round(vertical_error_mm,2),
        }

    # Keep V69 safety/joint checks, but interpret V69's legacy final-joint
    # mismatch at the TASK semantic layer using the actual MuJoCo TCP goal.
    def _v24_cartesian_postcheck(label, target_xyz, tol_m, safety_before, caught_exc=None):
        actual_xyz, actual_q = skill_core.current_tcp_pose()
        dist = _v24_xyz_distance(actual_xyz, target_xyz)
        safety_after = int(safety_trip_count.value)
        tracking_only = caught_exc is not None and "PHYSICAL TRACKING MISMATCH" in str(caught_exc)
        verification = {
            "target_xyz_m": [round(float(v), 5) for v in target_xyz],
            "actual_xyz_m": [round(float(v), 5) for v in actual_xyz],
            "tcp_error_mm": round(dist * 1000.0, 2),
            "tolerance_mm": round(float(tol_m) * 1000.0, 2),
            "safety_trip_before": int(safety_before),
            "safety_trip_after": int(safety_after),
        }
        if safety_after != int(safety_before):
            verification.update({
                "cartesian_ok": False,
                "reason": "SAFETY_TRIP_CHANGED",
                "safety_state_code": int(safety_state.value),
            })
            if caught_exc is not None:
                verification["motion_exception"] = str(caught_exc)
            return verification, actual_q, False, []
        if dist <= float(tol_m):
            warnings = []
            if tracking_only:
                warnings.append("Legacy V69 final-joint verifier reported PHYSICAL TRACKING MISMATCH, but the deterministic MuJoCo TCP postcondition is satisfied.")
                verification["legacy_joint_mismatch_overridden_by_task_postcondition"] = True
                skill_core.emit(
                    f"[V2-7 CARTESIAN ACCEPT] {label} | V69 joint mismatch is diagnostic only here; "
                    f"TCP error={dist*1000.0:.1f}mm <= {tol_m*1000.0:.1f}mm | safety unchanged"
                )
            verification.update({"cartesian_ok": True})
            return verification, actual_q, True, warnings
        verification.update({"cartesian_ok": False, "reason": "CARTESIAN_POSTCONDITION_FAILED"})
        return verification, actual_q, False, []

    def _v24_run_motion(label, fn, target_xyz, tol_m, preserve_grasp_quat=False):
        safety0 = int(safety_trip_count.value)
        caught = None
        raw_ok = False
        try:
            raw_ok = bool(fn())
        except Exception as exc:
            caught = exc
            _msg = str(exc)
            # V2-7 normalizes a physical safety stop into structured task
            # feedback so the Agent can reason about it.  The V69 latch remains
            # held; only recover_from_safety may explicitly clear it.
            if (
                "PHYSICAL TRACKING MISMATCH" not in _msg
                and "PHYSICAL_SAFETY_STOP" not in _msg
                and "CARTESIAN APPROACH BLOCKED" not in _msg
                and "EXECUTION_LAYER_FAULT" not in _msg
            ):
                raise
        verification, actual_q, cart_ok, warnings = _v24_cartesian_postcheck(label, target_xyz, tol_m, safety0, caught_exc=caught)

        # V2-7: controller completion can lag the first postcondition sample.
        # The log showed move_above reported ~35 mm error, then the very next
        # observation was already at the correct hover.  Recheck briefly before
        # turning a late settle into an Agent-visible planning failure.
        planning_blocked = caught is not None and "CARTESIAN APPROACH BLOCKED" in str(caught)
        if (not cart_ok) and (not planning_blocked) and int(safety_trip_count.value) == safety0:
            settle_deadline = time.time() + 1.8
            best = float(verification.get("tcp_error_mm", 1e9))
            while time.time() < settle_deadline:
                time.sleep(0.12)
                xyz2, q2 = skill_core.current_tcp_pose()
                d2 = _v24_xyz_distance(xyz2, target_xyz)
                best = min(best, d2 * 1000.0)
                if d2 <= float(tol_m):
                    actual_q = q2
                    cart_ok = True
                    verification.update({
                        "actual_xyz_m": [round(float(v), 5) for v in xyz2],
                        "tcp_error_mm": round(d2 * 1000.0, 2),
                        "cartesian_ok": True,
                        "late_settle_recheck": True,
                        "late_settle_best_error_mm": round(best, 2),
                    })
                    verification.pop("reason", None)
                    skill_core.emit(
                        f"[V2-7 LATE SETTLE ACCEPT] {label} | TCP reached tolerance on recheck: "
                        f"{d2*1000.0:.1f}mm <= {tol_m*1000.0:.1f}mm"
                    )
                    break

        if cart_ok and preserve_grasp_quat:
            skill_core.active_grasp_quat = [float(v) for v in actual_q]
            skill_core.emit(f"[V2-7 BRANCH PRESERVE] {label} | active_grasp_quat <- actual TCP quaternion")
        verification["legacy_skill_return"] = bool(raw_ok)
        if caught is not None:
            verification["legacy_joint_verify_error"] = str(caught)
        # V3.05 EXECUTION-LAYER FAULT -> safe-high recovery + bounded retry
        if (not cart_ok) and _v305_is_exec_fault(caught):
            _v305_handle_exec_fault(label, caught, verification)
        elif cart_ok and caught is None:
            V305_EXEC_FAULT["consecutive"] = 0
        return verification, cart_ok, warnings

    # ============================================================
    # V3.05 EXECUTION-LAYER FAULT HANDLING
    # ------------------------------------------------------------
    # A large final-joint mismatch (default >= 30 deg) right after the
    # controller "succeeded", a rejected goal, a non-SUCCEEDED goal status or
    # a duplicate controller is NOT a grasp/planning problem.  V3.04 let the
    # Agent keep switching candidates/eggs from random half-way postures.
    # Now: cancel, go back to the SAFE_NEUTRAL q0 posture (only when nothing
    # is held), and tell the Agent: retry the same step once; on a second
    # consecutive fault, stop and report.
    # ============================================================
    def _v305_is_exec_fault(exc):
        if exc is None:
            return False
        m = str(exc)
        if "EXECUTION_LAYER_FAULT" in m:
            return True
        if "PHYSICAL TRACKING MISMATCH" in m:
            mm = re.search(r"still ([0-9.]+)deg", m)
            thr = float(os.environ.get("NERO_V305_EXEC_FAULT_MISMATCH_DEG", "30"))
            return bool(mm) and float(mm.group(1)) >= thr
        return False

    def _v305_handle_exec_fault(label, exc, verification):
        V305_EXEC_FAULT["consecutive"] += 1
        V305_EXEC_FAULT["total"] += 1
        V305_EXEC_FAULT["last"] = f"{label}: {exc}"
        V304_PREGRASP_CACHE.clear()
        n = int(V305_EXEC_FAULT["consecutive"])
        max_retry = int(os.environ.get("NERO_V305_EXEC_FAULT_RETRIES", "1"))
        unhealthy = n > max_retry or "DUPLICATE_ARM_CONTROLLER" in str(exc)
        recovery = {"attempted": False}
        try:
            skill_core.cancel_active_motion()
        except Exception:
            pass
        try:
            skill_core._set_arm_command_gate(False)
        except Exception:
            pass
        holding = bool(hold_active_shared.value)
        if holding:
            recovery = {"attempted": False, "why": "object held; arm left in place"}
        elif "DUPLICATE_ARM_CONTROLLER" in str(exc):
            recovery = {"attempted": False, "why": "duplicate controller; no motion allowed"}
        elif V305_EXEC_FAULT.get("in_recovery"):
            recovery = {"attempted": False, "why": "already recovering"}
        elif os.environ.get("NERO_V305_EXEC_FAULT_RECOVERY", "1") != "0":
            V305_EXEC_FAULT["in_recovery"] = True
            try:
                time.sleep(0.5)
                skill_core.emit(f"[V3.05 EXEC FAULT RECOVERY] {label} -> SAFE_NEUTRAL q0")
                _v239_plan_exec_q0_neutral()
                recovery = {"attempted": True, "ok": True, "pose": "SAFE_NEUTRAL_q0"}
            except Exception as rexc:
                recovery = {"attempted": True, "ok": False, "error": str(rexc)[:240]}
                unhealthy = True
            finally:
                V305_EXEC_FAULT["in_recovery"] = False
        verification["reason"] = "EXECUTION_LAYER_FAULT"
        verification["execution_layer_fault"] = {
            "consecutive": n, "total": int(V305_EXEC_FAULT["total"]),
            "unhealthy": bool(unhealthy), "recovery": recovery,
            "error": str(exc)[:300],
        }
        skill_core.emit(
            f"[V3.05 EXECUTION-LAYER FAULT] {label} | consecutive={n} unhealthy={unhealthy} | "
            f"recovery={json.dumps(recovery, ensure_ascii=False)} | {str(exc)[:160]}"
        )

    # ============================================================
    # V3.07 START-STATE RESET + POST-PLACE POSTURE (plan_grasp supervisor)
    # ------------------------------------------------------------
    # V3.06 log: after egg_2 (J7 +80 deg wrist verticalization) every egg_3 /
    # egg_4 candidate failed with cuRobo success=false -- the start state, not
    # the goal, was unusable -- and nothing moved the arm back to a known
    # posture: plan_grasp is planning-only and the agent has no "go home" tool.
    #   (1) between eggs, check the released posture; poor joint margin or a
    #       wound-up distal wrist -> vertical retreat + SAFE_NEUTRAL q0 first;
    #   (2) if every untried geometry-feasible candidate still fails and nothing
    #       is held -> q0 once per object, then re-plan once;
    #   (3) the final failure says whether it was re-planned from q0, so the
    #       agent/user can tell start-state from goal-side failures.
    # Env: NERO_V307_POST_PLACE_POSTURE (1) | NERO_V307_POST_PLACE_MIN_MARGIN_DEG (20)
    #      NERO_V307_POST_PLACE_MAX_DISTAL_DEG (120) | NERO_V307_START_STATE_RESET (1)
    # ============================================================
    V307_RESET_USED = {}

    def _v307_posture_numbers():
        snap = _v239_health_snapshot()
        q = snap["q"]
        return snap, {
            "joints_deg": [round(math.degrees(float(v)), 1) for v in q],
            "joint_margin_deg": round(float(snap["joint_margin_deg"]), 1),
            "distal_wrist_deg": round(float(snap["distal_wrist_deg"]), 1),
            "tcp_z_mm": round(float(snap["xyz"][2]) * 1000.0, 0),
            "max_abs_joint_deg": round(max(abs(math.degrees(float(v))) for v in q), 1),
        }

    def _v310_wait_tcp_settled(label, timeout_s=None, tol_m=None, window_s=0.25):
        """V3.10: wait until the TCP has stopped moving (arm settling after a trajectory ends).
        Settled = TCP moved < tol over one window. Returns a small report dict (never raises)."""
        if os.environ.get("NERO_V310_SETTLE", "1") == "0":
            return {"enabled": False}
        try:
            timeout_s = float(os.environ.get("NERO_V310_SETTLE_TIMEOUT_S", "1.5")) if timeout_s is None else timeout_s
            tol_m = float(os.environ.get("NERO_V310_SETTLE_TOL_MM", "0.5")) / 1000.0 if tol_m is None else tol_m
            t_start = time.time()
            prev, _ = skill_core.current_tcp_pose()
            last_motion = None
            while time.time() - t_start < timeout_s:
                time.sleep(window_s)
                cur, _ = skill_core.current_tcp_pose()
                last_motion = _v24_xyz_distance(cur, prev)
                prev = cur
                if last_motion < tol_m:
                    rep = {"settled": True, "waited_s": round(time.time() - t_start, 2),
                           "last_window_motion_mm": round(last_motion * 1000.0, 2)}
                    if rep["waited_s"] > window_s + 0.05:
                        skill_core.emit(f"[V3.10 SETTLE] {label} | settled after {rep['waited_s']}s")
                    return rep
            rep = {"settled": False, "waited_s": round(time.time() - t_start, 2),
                   "last_window_motion_mm": round((last_motion or 0.0) * 1000.0, 2)}
            skill_core.emit(f"[V3.10 SETTLE] {label} | still moving {rep['last_window_motion_mm']}mm/{window_s}s after {rep['waited_s']}s")
            return rep
        except Exception as exc:
            return {"settled": None, "error": str(exc)[:160]}

    def _v307_go_q0(label):
        """Vertical retreat out of basket/holder zone (wrist preserved), then SAFE_NEUTRAL q0."""
        snap, _ = _v307_posture_numbers()
        if snap["near_basket"] or snap["near_holder"] or float(snap["xyz"][2]) < _V239_LOW_Z_M:
            xyz0 = list(snap["xyz"]); quat0 = list(snap["quat"])
            retreat = [xyz0[0], xyz0[1], max(_V239_CLEAR_Z_M, xyz0[2] + 0.120)]
            skill_core.emit(f"[V3.07 {label}] vertical retreat z={xyz0[2]*1000:.0f} -> {retreat[2]*1000:.0f}mm before q0")
            safety0 = int(safety_trip_count.value)
            # V3.12.1 RETREAT LADDER: a full retreat can be blocked (V3.12: 73 -> 340 mm only 40.7 %);
            # try shorter straight retreats, else go to q0 directly (OMPL joint plan is collision-checked).
            traj = None
            last3121 = None
            for dz3121 in sorted({round(retreat[2] - xyz0[2], 4), 0.120, 0.080, 0.050}, reverse=True):
                if dz3121 <= 0.0 or dz3121 > retreat[2] - xyz0[2] + 1e-6:
                    continue
                tgt3121 = [xyz0[0], xyz0[1], xyz0[2] + dz3121]
                try:
                    traj = skill_core._cartesian_approach_plan(tgt3121, quat0)
                    if dz3121 < retreat[2] - xyz0[2] - 1e-6:
                        skill_core.emit(f"[V3.12.1 RETREAT LADDER] {label} | full retreat blocked ({last3121}); "
                                        f"using +{dz3121*1000:.0f} mm")
                    break
                except Exception as exc3121:
                    last3121 = exc3121
            if traj is not None:
                _v218_slow_trajectory_in_place(traj, factor=1.45)
                skill_core._execute_trajectory(traj)
            else:
                skill_core.emit(f"[V3.12.1 RETREAT LADDER] {label} | no straight retreat plannable ({last3121}); "
                                f"collision-checked joint-space q0 directly")
            if int(safety_trip_count.value) != safety0:
                raise RuntimeError("V3.07 vertical retreat triggered safety")
        _v239_plan_exec_q0_neutral()
        _v310_wait_tcp_settled(f"after q0 ({label})")

    def _v307_post_place_posture(obj, stage):
        if os.environ.get("NERO_V307_POST_PLACE_POSTURE", "1") == "0":
            return None
        if stage not in (V24_STAGE_VERIFIED, V24_STAGE_RELEASED):
            return None
        if bool(hold_active_shared.value) or int(safety_state.value) != 0:
            return None
        min_m = float(os.environ.get("NERO_V307_POST_PLACE_MIN_MARGIN_DEG", "20"))
        max_d = float(os.environ.get("NERO_V307_POST_PLACE_MAX_DISTAL_DEG", "120"))
        _, nums = _v307_posture_numbers()
        why = []
        if nums["joint_margin_deg"] < min_m:
            why.append(f"joint_margin={nums['joint_margin_deg']}<{min_m:g}deg")
        if nums["distal_wrist_deg"] > max_d:
            why.append(f"distal_wrist={nums['distal_wrist_deg']}>{max_d:g}deg")
        rec = {"checked": True, "before": nums, "reasons": why, "q0_executed": False}
        skill_core.emit(
            f"[V3.07 POST-PLACE POSTURE] before {obj} | q_deg={nums['joints_deg']} | margin={nums['joint_margin_deg']}deg | "
            f"distal={nums['distal_wrist_deg']}deg | z={nums['tcp_z_mm']:.0f}mm | "
            + (f"-> SAFE_NEUTRAL q0 ({', '.join(why)})" if why else "OK, keep posture")
        )
        if why:
            try:
                _v307_go_q0("POST-PLACE POSTURE")
                rec["q0_executed"] = True
                rec["after"] = _v307_posture_numbers()[1]
            except Exception as exc:
                rec["error"] = str(exc)[:240]
                skill_core.emit(f"[V3.07 POST-PLACE POSTURE FAIL] {exc}")
        return rec

    def _v307_classify_failures(res):
        cands = ((res.get("grasp_plan") or {}).get("candidates") or [])
        live = [c for c in cands if not c.get("already_tried") and c.get("geometry_feasible")]
        start_cls = goal_cls = 0
        for c in live:
            e = str(c.get("moveit_pregrasp_error") or "")
            if "success=false" in e or "exhausted" in e:
                start_cls += 1
            elif e:
                goal_cls += 1
        return len(live), start_cls, goal_cls

    def _v307_plan_grasp_supervised(args):
        obj = str(args.get("object_name", ""))
        stage = str(_v24_state_snapshot().get("stage") or "")
        post = _v307_post_place_posture(obj, stage)
        res = llm_tool_runner("plan_grasp", dict(args, _v307_inner=True))
        if post is not None and isinstance(res, dict):
            res["v307_post_place_posture"] = post
        pv = (res.get("physical_verification") or {}) if isinstance(res, dict) else {}
        if (not isinstance(res, dict) or res.get("ok")
                or pv.get("reason") != "ALL_GRASP_CANDIDATES_EXHAUSTED_OR_INFEASIBLE"
                or pv.get("edge_reposition_available")):
            return res
        n_live, n_start, n_goal = _v307_classify_failures(res)
        info = {"attempted": False, "untried_feasible_candidates": n_live,
                "start_state_class_failures": n_start, "goal_class_failures": n_goal}
        if (post or {}).get("q0_executed"):
            info["already_at_q0_from_post_place"] = True
        _, nums = _v307_posture_numbers()
        info["start_posture"] = nums
        skip = None
        if os.environ.get("NERO_V307_START_STATE_RESET", "1") == "0":
            skip = "disabled by NERO_V307_START_STATE_RESET=0"
        elif n_live == 0:
            skip = "no untried geometry-feasible candidate (genuinely exhausted)"
        elif bool(hold_active_shared.value):
            skip = "hold_active=true"
        elif int(safety_state.value) != 0:
            skip = "safety state latched"
        elif int(V307_RESET_USED.get(obj, 0)) >= 1:
            skip = "reset budget for this object already used"
        elif nums["max_abs_joint_deg"] <= 2.0:
            skip = "start state already SAFE_NEUTRAL q0"
            info["replanned_from_q0"] = True
        if skip:
            info["skipped"] = skip
            res["v307_start_state_reset"] = info
            pv["v307_start_state_reset"] = info
            if info.get("replanned_from_q0") or info.get("already_at_q0_from_post_place"):
                res.setdefault("warnings", []).append(
                    "V3.07: planned from SAFE_NEUTRAL q0 and still infeasible -> object/goal-side failure, not start-state.")
            skill_core.emit(f"[V3.07 START-STATE RESET SKIPPED] {obj} | {skip}")
            return res
        V307_RESET_USED[obj] = int(V307_RESET_USED.get(obj, 0)) + 1
        skill_core.emit(
            f"[V3.07 START-STATE RESET] {obj} | {n_live} untried feasible candidates all failed "
            f"(start-class={n_start}, goal-class={n_goal}) | q_deg={nums['joints_deg']} | margin={nums['joint_margin_deg']}deg | "
            f"distal={nums['distal_wrist_deg']}deg -> SAFE_NEUTRAL q0, re-plan once"
        )
        info["attempted"] = True
        try:
            _v307_go_q0("START-STATE RESET")
            info["q0_ok"] = True
        except Exception as exc:
            info["q0_ok"] = False
            info["error"] = str(exc)[:240]
            skill_core.emit(f"[V3.07 START-STATE RESET FAIL] {obj} | {exc}")
            pv["reason"] = "START_STATE_PLANNING_FAILED"
            pv["v307_start_state_reset"] = info
            res["v307_start_state_reset"] = info
            res["warnings"] = ["V3.07: grasp planning failed from the current start state and the q0 reset itself failed. Stop and report; do not retry."]
            res["recommended_next_actions"] = ["finish_and_report"]
            return res
        res2 = llm_tool_runner("plan_grasp", dict(args, _v307_inner=True))
        info["replanned_from_q0"] = True
        info["replan_ok"] = bool(res2.get("ok"))
        res2["v307_start_state_reset"] = info
        if post is not None:
            res2["v307_post_place_posture"] = post
        pv2 = res2.get("physical_verification")
        if isinstance(pv2, dict):
            pv2["v307_start_state_reset"] = info
        if res2.get("ok"):
            skill_core.emit(f"[V3.07 START-STATE RESET OK] {obj} | re-plan from q0 found a candidate")
        else:
            res2.setdefault("warnings", []).append(
                "V3.07: re-planned from SAFE_NEUTRAL q0 and still infeasible -> object/goal-side failure, not start-state.")
            skill_core.emit(f"[V3.07 START-STATE RESET NO HELP] {obj} | still infeasible from q0 (goal-side)")
        return res2

    # ============================================================
    # V3.12.1 RECOVERY HELPERS
    # ------------------------------------------------------------
    # V3.12 run (camera mode) failed half-way although the backend was unchanged:
    #  * lift(0.13) -> "Cartesian corridor only 97.9%" raised as BACKEND_EXCEPTION;
    #  * the model then lifted only 40 mm and orient_for_slot gave up (held egg too low);
    #  * no tool could put the held egg back -> deadlock with 2 eggs left.
    # Helpers below: lift height ladder, pre-raise before reorientation, Cartesian
    # descent check from the planned pregrasp joint state, retreat after a blocked
    # descent, held-egg pose from TCP x grasp transform, return-to-pick-location.
    # ============================================================
    V3121_LIFT_LADDER_M = (0.108, 0.09, 0.07, 0.05)
    V3121_ORIENT_MIN_TCP_Z_M = float(os.environ.get("NERO_V3121_ORIENT_MIN_TCP_Z_M", "0.160"))
    V3121_GRASP_ORIGIN = {}      # obj -> where it was picked (TCP pose, aperture, perceived egg pose)
    V3121_HELD_REL = {}          # obj -> egg pose in the TCP frame at clamp time
    V3121_RETURNS = {}           # obj -> number of return_to_pick_location calls

    def _v3121_is_plan_block(exc):
        m = str(exc)
        return any(k in m for k in ("Cartesian corridor only", "CARTESIAN APPROACH BLOCKED",
                                    "Cartesian service failed", "CARTESIAN FAILED",
                                    "GetCartesianPath returned no response"))

    def _v3121_qrot(q, v):
        n = math.sqrt(sum(float(t) * float(t) for t in q)) or 1.0
        x, y, z, w = [float(t) / n for t in q]
        vx, vy, vz = [float(t) for t in v]
        tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
        return [vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx)]

    def _v3121_qconj(q):
        return [-float(q[0]), -float(q[1]), -float(q[2]), float(q[3])]

    def _v3121_qmul(a, b):
        ax, ay, az, aw = [float(t) for t in a]
        bx, by, bz, bw = [float(t) for t in b]
        return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]

    def _v3121_qangle_deg(a, b):
        d = abs(sum(float(x) * float(y) for x, y in zip(a, b)))
        return math.degrees(2.0 * math.acos(max(-1.0, min(1.0, d))))

    def _v3121_lift_ladder(dz, on_used=None, label="lift"):
        """Vertical lift; if the requested height's corridor is blocked, try lower heights.
        Only planning blocks are retried (nothing moved).  Returns the height actually lifted."""
        dz = float(dz)
        ladder = [dz] + [h for h in V3121_LIFT_LADDER_M if h < dz - 1e-6]
        tried, last = [], None
        for h in ladder:
            tried.append(round(h * 1000.0, 1))
            try:
                skill_core.bio_lift(h)
                if on_used:
                    on_used(h, tried)
                if h < dz - 1e-6:
                    skill_core.emit(f"[V3.12.1 LIFT FALLBACK] {label} | {dz*1000:.0f} mm blocked -> lifted {h*1000:.0f} mm "
                                    f"(tried {tried} mm)")
                return h
            except Exception as exc:
                if not _v3121_is_plan_block(exc):
                    raise
                last = exc
                skill_core.emit(f"[V3.12.1 LIFT FALLBACK] {label} | {h*1000:.0f} mm corridor blocked ({exc}); trying lower")
        if on_used:
            on_used(None, tried)
        raise RuntimeError(f"CARTESIAN APPROACH BLOCKED: {label} corridor blocked at every height {tried} mm; last: {last}")

    def _v3121_pre_raise_for_orient(obj):
        """orient_for_slot assumes the egg is at a normal post-lift height.  Raise it first if it is low."""
        tcp, _q = skill_core.current_tcp_pose()
        z = float(tcp[2])
        if z >= V3121_ORIENT_MIN_TCP_Z_M - 0.005:
            return None
        need = V3121_ORIENT_MIN_TCP_Z_M - z
        rep = {"tcp_z_before_mm": round(z * 1000.0, 1), "target_tcp_z_mm": round(V3121_ORIENT_MIN_TCP_Z_M * 1000.0, 1)}
        skill_core.emit(f"[V3.12.1 ORIENT PRE-RAISE] {obj} | TCP z={z*1000:.0f} mm < {V3121_ORIENT_MIN_TCP_Z_M*1000:.0f} mm "
                        f"-> vertical raise {need*1000:.0f} mm before reorientation")
        ladder = [need] + [need * f for f in (0.75, 0.5) if need * f >= 0.02]
        for h in ladder:
            try:
                skill_core.bio_lift(h)
                tcp2, _ = skill_core.current_tcp_pose()
                rep.update(ok=True, raised_mm=round(h * 1000.0, 1), tcp_z_after_mm=round(float(tcp2[2]) * 1000.0, 1))
                return rep
            except Exception as exc:
                if not _v3121_is_plan_block(exc):
                    rep.update(ok=False, error=str(exc)[:200])
                    return rep
                skill_core.emit(f"[V3.12.1 ORIENT PRE-RAISE] {h*1000:.0f} mm blocked; trying lower")
        rep.update(ok=False, error="pre-raise corridor blocked")
        return rep

    def _v3121_descent_check_from_traj(traj, cand):
        """Straight Cartesian descent pregrasp -> grasp, planned from the END joint state of the planned
        pregrasp trajectory (the state approach_planned_grasp will start from).  The V3.04 IK screen used
        its own BioIK chain and missed blocks on the real branch (egg_1 major_aligned__inward6: 25 %)."""
        if os.environ.get("NERO_V3121_DESCENT_PRECHECK", "1") == "0":
            return True, "disabled"
        core = sys.modules[type(skill_core).__module__]
        jt = getattr(traj, "joint_trajectory", None)
        if jt is None or not jt.points:
            return True, "no trajectory end state (not checked)"
        names = list(jt.joint_names)
        last = list(jt.points[-1].positions)
        try:
            arm = [float(last[names.index(n)]) for n in core.ARM_JOINT_NAMES]
        except ValueError:
            return True, "joint names differ (not checked)"
        req = core.GetCartesianPath.Request()
        req.header.frame_id = "base_link"
        req.group_name = "arm"
        req.link_name = "tcp_link"
        req.start_state = core.RobotState()
        req.start_state.joint_state = skill_core._v71_full_joint_state_for_arm(arm)
        req.start_state.is_diff = False
        q = [float(v) for v in cand["quat_xyzw"]]
        g = [float(v) for v in cand["grasp_xyz_m"]]
        fin = float(os.environ.get("NERO_V304_FINAL_APPROACH_M", "0.020"))
        poses = []
        for xyz in ([g[0], g[1], g[2] + fin], g):
            p = core.Pose()
            p.position.x, p.position.y, p.position.z = float(xyz[0]), float(xyz[1]), float(xyz[2])
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
            poses.append(p)
        req.waypoints = poses
        req.max_step = float(core.CARTESIAN_MAX_STEP)
        req.jump_threshold = 0.0
        for attr in ("prismatic_jump_threshold", "revolute_jump_threshold"):
            if hasattr(req, attr):
                setattr(req, attr, 0.0)
        req.avoid_collisions = True
        resp = skill_core.wait_future(skill_core.cartesian_client.call_async(req), 8.0, "V3.12.1 descent precheck")
        if resp is None:
            return False, "descent precheck: no response"
        if int(resp.error_code.val) != 1:
            return False, f"descent precheck: Cartesian error {int(resp.error_code.val)}"
        frac = float(resp.fraction)
        if frac < 0.98:
            return False, f"straight descent only {frac*100.0:.1f}% collision-free from the planned pregrasp"
        return True, f"ok {frac*100.0:.1f}%"

    def _v3121_retreat_to_z(z_target, label):
        """Straight vertical retreat at the current orientation (reverse of the path just executed)."""
        tcp, q = skill_core.current_tcp_pose()
        if float(tcp[2]) >= float(z_target) - 0.010:
            return {"needed": False}
        tgt = [float(tcp[0]), float(tcp[1]), float(z_target)]
        try:
            traj = skill_core._cartesian_approach_plan(tgt, q)
            _v218_slow_trajectory_in_place(traj, factor=1.45)
            skill_core._execute_trajectory(traj)
            tcp2, _ = skill_core.current_tcp_pose()
            skill_core.emit(f"[V3.12.1 RETREAT] {label} | z {float(tcp[2])*1000:.0f} -> {float(tcp2[2])*1000:.0f} mm")
            return {"needed": True, "ok": True, "z_after_mm": round(float(tcp2[2]) * 1000.0, 1)}
        except Exception as exc:
            skill_core.emit(f"[V3.12.1 RETREAT FAIL] {label} | {exc}")
            return {"needed": True, "ok": False, "error": str(exc)[:200]}

    def _v3121_record_grasp(obj, opening_before_mm):
        """At clamp: remember where the egg was picked and its pose in the TCP frame."""
        try:
            tcp, tq = skill_core.current_tcp_pose()
            plan = (V27_GRASP_RUNTIME.get("plans", {}) or {}).get(obj) or {}
            oxyz = plan.get("object_xyz_at_plan_m")
            oq = plan.get("object_quat_at_plan_xyzw")
            if oxyz is None or oq is None:
                oxyz, oq = skill_core.object_pose(obj)
            V3121_GRASP_ORIGIN[obj] = {
                "tcp_xyz": [float(v) for v in tcp], "tcp_quat": [float(v) for v in tq],
                "object_xyz": [float(v) for v in oxyz], "object_quat": [float(v) for v in oq],
                "opening_mm": float(opening_before_mm),
                "candidate_id": (V27_GRASP_RUNTIME.get("selected", {}) or {}).get(obj),
            }
            d = [float(oxyz[i]) - float(tcp[i]) for i in range(3)]
            V3121_HELD_REL[obj] = {"p": _v3121_qrot(_v3121_qconj(tq), d), "q": _v3121_qmul(_v3121_qconj(tq), oq)}
        except Exception as exc:
            skill_core.emit(f"[V3.12.1 GRASP RECORD WARN] {obj} | {exc}")

    def _v3121_held_name():
        if not bool(hold_active_shared.value):
            return None
        st = v24_skill_state
        if str(st.get("stage") or "") in (V24_STAGE_GRASPED, V24_STAGE_LIFTED, "PLACEMENT_ORIENTED",
                                          "SLOT_ENTRY_ALIGNED", V24_STAGE_BIN_HOVER, V24_STAGE_SLOT_PLACE):
            return st.get("active_object")
        return None

    def _v3121_held_pose_estimate(obj, tcp_xyz, tcp_q):
        rel = V3121_HELD_REL.get(obj)
        if not rel:
            return None
        p = _v3121_qrot(tcp_q, rel["p"])
        return [float(tcp_xyz[i]) + p[i] for i in range(3)], _v3121_qmul(tcp_q, rel["q"])

    def _v3121_pending_eggs():
        out = []
        for n in _v332_known_eggs():                     # V3.32
            try:
                if v311_mode() == "camera":
                    e = _v311_obj_entry(n)
                    xyz = e["xyz_m"] if e else None
                else:
                    xyz, _ = skill_core.object_pose(n)
                loc = llm_classify_object_location(xyz) if xyz is not None else "unknown"
            except Exception:
                loc = "unknown"
            if not str(loc).startswith("egg_holder."):
                out.append(n)
        return out

    def llm_tool_runner(name, args):
        """Synchronous V2-7 Gripper-Aware Skill Contract bridge used by Agent and manual debugger."""
        if skill_core is None:
            raise RuntimeError(f"V69 skill core unavailable: {skill_core_error}")
        name = str(name)
        args = dict(args or {})
        # V3.08: camera capture on selected tools (record only, non-blocking) in truth mode.
        # V3.11: in camera mode the skill WAITS for a fresh capture and uses it: observe_scene and
        # verify_object_in_slot at the start; plan_grasp in its inner call (after the V3.07 q0 move).
        _v311_cam = v311_mode() == "camera"
        if _v311_cam and (name in ("observe_scene", "verify_object_in_slot")
                          or (name == "plan_grasp" and args.get("_v307_inner"))):
            _v311_obj = args.get("object_name")
            _v311_tag = f"{name}:{_v311_obj}" if _v311_obj else name
            try:
                _v311_capture(_v311_tag)
            except Exception as _v311_exc:
                skill_core.emit(f"[V3.11 PERCEPTION FAIL] {_v311_tag} | {_v311_exc}")
                return {"ok": False, "tool": name, "reason": str(_v311_exc).split(":", 1)[0],
                        "detail": str(_v311_exc)[:300], "pose_source": v335_pose_source_label(),
                        "recommended_next_actions": ["observe_scene", "stop_and_report_if_repeated"]}
        elif not args.get("_v307_inner"):
            try:
                import NERO_V308_Camera_Perception as _v308c
                if name in _v308c.trigger_tools():
                    _v308_obj = args.get("object_name")
                    v308_trigger(f"{name}:{_v308_obj}" if _v308_obj else name)
            except Exception:
                pass
        if name == "plan_grasp":
            if not args.pop("_v307_inner", False):
                return _v307_plan_grasp_supervised(args)
        state_before = _v24_state_snapshot()
        # V2-7.2: recover stale carrying state BEFORE an older observe handler can return.
        if (
            name == "observe_scene"
            and not bool(hold_active_shared.value)
            and str(state_before.get("stage") or "") in (
                "LIFTED", "RECEPTACLE_HOVER_REACHED", "SLOT_PLACE_REACHED",
                "PLACEMENT_ORIENTED", "SLOT_ENTRY_ALIGNED",
            )
        ):
            _lost = str(state_before.get("active_object") or "")
            skill_core.emit(
                f"[V2-7.2 GRASP LOST RECOVERY] stage={state_before.get('stage')} "
                f"active_object={_lost or None} hold_active=false -> clear stale state"
            )
            try:
                if _lost:
                    for _bucket in ("plans", "oriented", "slot_entry", "released"):
                        V271_PLACEMENT_RUNTIME.get(_bucket, {}).pop(_lost, None)
            except Exception:
                pass
            _v24_reset_skill_state("V2-7.2 physical grasp loss", clear_adaptation=False)
            state_before = _v24_state_snapshot()
        contract = _v24_contract(name, args)
        skill_core.emit(f"[V2-7 TOOL START] {name} | args={json.dumps(args, ensure_ascii=False)}")
        _v24_log_contract(name, args, contract, state_before)
        t0 = time.perf_counter()

        if name == "observe_scene":
            if state_before.get("stage") == V24_STAGE_IDLE:
                _v24_set_state(V24_STAGE_OBSERVED, last_tool=name, last_ok=True)
            scene = llm_scene_snapshot()
            scene["v2_skill_state"] = _v24_state_snapshot()
            scene["constraint_aware_grasp"] = _v27_public_runtime_snapshot()
            scene["placement_aware"] = _v271_public_runtime_snapshot()
            # V3.04 FILL ORDER: holder slots are filled in numeric order.
            try:
                _sl304 = scene.get("receptacles", {}).get("egg_holder", {}).get("slots", {})
                _empty304 = sorted([k for k, v in _sl304.items() if not v.get("occupied")],
                                   key=lambda k: int(str(k).split("_")[-1]))
                scene["egg_holder_fill_order"] = {
                    "next_empty_slot": _empty304[0] if _empty304 else None,
                    "empty_slots_in_order": _empty304,
                    "rule": "fill slots in numeric order; use next_empty_slot unless the user names an explicit slot_N",
                }
            except Exception:
                pass
            skill_core.emit("[V2-7 SCENE] " + json.dumps(scene, ensure_ascii=False, separators=(",", ":")))
            opening_now = float(gripper_actual_mm.value)
            for obj_name in OBJECTS:
                pr = _v24_gripper_prior(obj_name, opening_now)
                skill_core.emit(
                    f"[GRIPPER PRIOR] {obj_name} | opening={opening_now:.1f}mm | "
                    f"effective_width={pr['effective_width_mm']:.1f}mm | "
                    f"recommended={pr['recommended_opening_range_mm']}mm | "
                    f"target={pr['recommended_target_opening_mm']:.1f}mm"
                )
            return _v24_result(
                name, True, contract, state_before,
                {"observation_only": True, "physical_motion_expected": False},
                extra={"scene": scene, "elapsed_s": round(time.perf_counter()-t0, 3)},
            )

        if name == "recover_from_safety":
            obj = str(args.get("object_name") or state_before.get("active_object") or "")
            strategy = str(args.get("strategy", "retreat_and_retry_once"))
            diagnosis = str(args.get("diagnosis", "")).strip()
            ctx = dict(V242_LAST_SAFETY_CONTEXT)

            if state_before.get("stage") != V24_STAGE_SAFETY:
                return _v24_precondition_failure(
                    name, args, f"requires SAFETY_STOPPED; current={state_before}",
                    contract, state_before, recommended_next_actions=["observe_scene"],
                )
            if not ctx.get("active") or ctx.get("object_name") != obj:
                return _v24_precondition_failure(
                    name, args, f"no matching active safety context for {obj}: {ctx}",
                    contract, state_before, recommended_next_actions=["observe_scene"],
                )
            if bool(hold_active_shared.value):
                return _v24_precondition_failure(
                    name, args, "hold_active=true; automatic pre-grasp retry is not allowed while holding an object",
                    contract, state_before, recommended_next_actions=["stop_and_preserve_grasp"],
                )
            used = int(V242_SAFETY_RETRY_USED.get(obj, 0))
            if used >= 1:
                return _v24_precondition_failure(
                    name, args, f"one safety retry already consumed for {obj}; do not loop",
                    contract, state_before,
                    physical_verification={"retry_used": used, "safety_context": ctx},
                    recommended_next_actions=["stop_and_explain_failure"],
                )

            # Consumes the one retry budget as soon as explicit recovery begins.
            V242_SAFETY_RETRY_USED[obj] = used + 1
            prior_before = _v24_gripper_prior(obj, float(gripper_actual_mm.value))
            old_bonus = float(V241_APERTURE_RETRY_BONUS_MM.get(obj, 0.0))
            new_bonus = old_bonus
            if strategy == "widen_aperture_and_retry_once":
                new_bonus = min(30.0, old_bonus + 15.0)
                V241_APERTURE_RETRY_BONUS_MM[obj] = new_bonus
            prior_after = _v24_gripper_prior(obj, float(gripper_actual_mm.value))

            skill_core.emit(
                f"[SAFETY REFLECTION] object={obj} | diagnosis={diagnosis or ctx.get('likely_cause')} | "
                f"strategy={strategy} | retry=1/1 | aperture_bonus={old_bonus:.1f}->{new_bonus:.1f}mm"
            )

            # Explicit planner recovery: this is the programmatic equivalent of
            # STOP + RECOVER, not an automatic bypass of V69 safety.
            trip_before = int(safety_trip_count.value)
            recover_request.value = True
            deadline = time.time() + 1.5
            while time.time() < deadline:
                # The MuJoCo loop converts recovery to reason_code=2 and clears
                # its internal ros_motion_fault_latched flag.
                if int(safety_state.value) == 2:
                    break
                time.sleep(0.03)
            time.sleep(0.12)

            retreat_start, _ = skill_core.current_tcp_pose()
            retreat_target = [float(retreat_start[0]), float(retreat_start[1]), float(retreat_start[2]) + 0.090]
            retreat_ok = False
            retreat_rise_mm = 0.0
            retreat_error = None
            retreat_motion_verification = {}
            try:
                llm_require_arm_controller()
                skill_core.emit("[SAFETY RECOVERY] explicit unlatch complete; vertical retreat +90 mm with current aperture preserved")
                retreat_motion_verification, retreat_cart_ok, retreat_warnings = _v24_run_motion(
                    f"safety_recovery_retreat({obj})",
                    lambda: skill_core.bio_lift(0.090),
                    retreat_target, 0.015, True,
                )
                retreat_end, _ = skill_core.current_tcp_pose()
                retreat_rise_mm = (float(retreat_end[2]) - float(retreat_start[2])) * 1000.0
                retreat_ok = bool(
                    retreat_cart_ok
                    and retreat_rise_mm > 70.0
                    and int(safety_trip_count.value) == trip_before
                )
                for _w in retreat_warnings:
                    skill_core.emit(f"[SAFETY RECOVERY WARN] {_w}")
            except Exception as exc:
                retreat_error = str(exc)

            verification = {
                "safety_context": ctx,
                "agent_diagnosis": diagnosis,
                "strategy": strategy,
                "retry_used": 1,
                "retry_limit": 1,
                "aperture_bonus_before_mm": round(old_bonus, 2),
                "aperture_bonus_after_mm": round(new_bonus, 2),
                "recommended_retry_opening_mm": prior_after.get("recommended_target_opening_mm"),
                "retreat_rise_mm": round(retreat_rise_mm, 2),
                "retreat_ok": bool(retreat_ok),
                "retreat_motion_verification": retreat_motion_verification,
                "new_safety_trip_during_recovery": int(safety_trip_count.value) != trip_before,
            }
            if retreat_error:
                verification["retreat_error"] = retreat_error

            if not retreat_ok:
                _v24_set_state(
                    V24_STAGE_SAFETY, active_object=obj, target_bin=None,
                    last_tool=name, last_ok=False,
                )
                return _v24_result(
                    name, False, contract, state_before, verification,
                    warnings=["Safety recovery retreat did not complete cleanly; do not attempt another automatic retry."],
                    extra={"object_name": obj, "elapsed_s": round(time.perf_counter()-t0, 3)},
                    recommended_next_actions=["stop_and_explain_failure"],
                )

            _v242_clear_safety_context()
            _v24_set_state(
                V24_STAGE_OBSERVED, active_object=None, target_bin=None,
                last_tool=name, last_ok=True,
            )
            skill_core.emit(
                f"[V2-7 RECOVERY OK] {obj} | retreat={retreat_rise_mm:.1f}mm | fresh re-observation required | "
                f"next recommended aperture={prior_after.get('recommended_target_opening_mm')}mm | retry budget now 1/1"
            )
            return _v24_result(
                name, True, contract, state_before, verification,
                warnings=[],
                extra={"object_name": obj, "elapsed_s": round(time.perf_counter()-t0, 3)},
                next_stage=V24_STAGE_OBSERVED, active_object=None, target_bin=None,
                recommended_next_actions=["set_gripper_opening", "observe_scene", "plan_grasp"],
            )

        if name == "set_gripper_opening":
            stage = state_before.get("stage")
            if stage not in (V24_STAGE_OBSERVED, V24_STAGE_HOVER, V24_STAGE_VERIFIED, V24_STAGE_RELEASED):
                return _v24_precondition_failure(
                    name, args,
                    f"pre-grasp aperture adjustment not allowed from state={stage}; observe scene first",
                    contract, state_before,
                    recommended_next_actions=["observe_scene"],
                )
            if bool(hold_active_shared.value):
                return _v24_precondition_failure(
                    name, args, "hold_active=true; do not use pre-grasp aperture adjustment while holding an object",
                    contract, state_before,
                    recommended_next_actions=["resolve_or_release_held_object"],
                )
            target_mm = max(5.0, min(100.0, float(args.get("target_opening_mm", 60.0))))
            # V3.10: V3.09 noisy run reported GRIPPER_APERTURE_TARGET_NOT_VERIFIED twice with tcp_motion
            # 3.2/4.5 mm right after the V3.07 q0 move -- the arm was still settling, not pushed by the fingers.
            settle = _v310_wait_tcp_settled("set_gripper_opening")
            opening_before = float(gripper_actual_mm.value)
            tcp_before, _ = skill_core.current_tcp_pose()
            object_before = {}
            for obj_name in OBJECTS:
                try:
                    object_before[obj_name] = skill_core.object_pose(obj_name)[0]
                except Exception:
                    pass
            safety_before = int(safety_trip_count.value)
            skill_core.emit(
                f"[V2-7 GRIPPER PREP] position mode | actual={opening_before:.1f}mm -> target={target_mm:.1f}mm"
            )
            hold_off_request.value = True
            gripper_force_mode.value = False
            gripper_target_mm.value = target_mm
            deadline = time.time() + 4.0
            last_report = 0.0
            while time.time() < deadline:
                actual = float(gripper_actual_mm.value)
                now = time.time()
                if now - last_report >= 0.50:
                    skill_core.emit(
                        f"[V2-7 GRIPPER PREP WAIT] target={target_mm:.1f}mm | actual={actual:.1f}mm | "
                        f"error={actual-target_mm:+.1f}mm"
                    )
                    last_report = now
                if abs(actual - target_mm) <= 2.0:
                    break
                time.sleep(0.05)
            time.sleep(0.08)
            opening_after = float(gripper_actual_mm.value)
            tcp_after, _ = skill_core.current_tcp_pose()
            tcp_motion = _v24_xyz_distance(tcp_after, tcp_before)
            max_obj_motion = 0.0
            max_obj_name = None
            for obj_name, before_xyz in object_before.items():
                try:
                    after_xyz, _ = skill_core.object_pose(obj_name)
                    d = _v24_xyz_distance(after_xyz, before_xyz)
                    if d > max_obj_motion:
                        max_obj_motion = d
                        max_obj_name = obj_name
                except Exception:
                    pass
            safety_after = int(safety_trip_count.value)
            error_mm = abs(opening_after - target_mm)
            ok = (
                error_mm <= 3.0
                and not bool(hold_active_shared.value)
                and safety_after == safety_before
                and tcp_motion <= 0.003
                and max_obj_motion <= 0.005
            )
            verification = {
                "target_opening_mm": round(target_mm, 2),
                "opening_before_mm": round(opening_before, 2),
                "opening_after_mm": round(opening_after, 2),
                "opening_error_mm": round(error_mm, 2),
                "hold_active": bool(hold_active_shared.value),
                "tcp_motion_mm": round(tcp_motion * 1000.0, 2),
                "max_object_motion_mm": round(max_obj_motion * 1000.0, 2),
                "max_object_motion_object": max_obj_name,
                "safety_trip_before": safety_before,
                "safety_trip_after": safety_after,
                "physical_motion_expected": "gripper fingers only",
                "v310_pre_settle": settle,
            }
            if not ok:
                verification["reason"] = "GRIPPER_APERTURE_TARGET_NOT_VERIFIED"
            return _v24_result(
                name, ok, contract, state_before, verification, [],
                {"elapsed_s": round(time.perf_counter()-t0, 3)},
                next_stage=None,
                recommended_next_actions=["plan_grasp"] if ok else ["observe_scene", "retry_or_stop_based_on_failure_reason"],
            )


        if name == "reposition_egg_inward":
            # V2-28 NEAR-CLOSED TWO-FINGER EDGE-EGG REPOSITION + V2-28 POSTURE RESET
            # ------------------------------------------------------------
            # Only this recovery skill differs from V2-26.  V2-24 motion,
            # held-object manipulation and release are intentionally untouched.
            obj = str(args.get("object_name", ""))
            stage = state_before.get("stage")
            if stage != V24_STAGE_OBSERVED:
                return _v24_precondition_failure(
                    name, args, f"requires OBSERVED state; current={stage}",
                    contract, state_before, recommended_next_actions=["observe_scene"],
                )
            if bool(hold_active_shared.value):
                return _v24_precondition_failure(
                    name, args, "hold_active=true; edge reposition is pre-grasp only",
                    contract, state_before, recommended_next_actions=["resolve_or_release_held_object"],
                )
            if obj not in ("egg_1", "egg_2", "egg_3", "egg_4"):
                return _v24_precondition_failure(
                    name, args, f"unsupported object={obj}",
                    contract, state_before, recommended_next_actions=["observe_scene"],
                )
            used27 = int(V226_EDGE_REPOSITION_USED.get(obj, 0))
            if used27 >= V226_EDGE_REPOSITION_MAX_PER_OBJECT:
                return _v24_precondition_failure(
                    name, args, f"V2-28 edge reposition budget already used for {obj}",
                    contract, state_before, recommended_next_actions=["observe_scene", "try_another_object_or_report"],
                )
            plan27 = V27_GRASP_RUNTIME.get("plans", {}).get(obj)
            if not plan27:
                return _v24_precondition_failure(
                    name, args, f"no latest failed grasp plan for {obj}",
                    contract, state_before, recommended_next_actions=["plan_grasp"],
                )
            cands27 = [c for c in plan27.get("candidates", []) if not c.get("already_tried")]
            if not cands27:
                return _v24_precondition_failure(
                    name, args, f"no current grasp candidates for edge diagnosis: {obj}",
                    contract, state_before, recommended_next_actions=["observe_scene", "plan_grasp"],
                )
            geom27 = [c for c in cands27 if bool(c.get("geometry_feasible"))]
            best_clear27 = max(float(c.get("basket_projected_clearance_mm", -999.0)) for c in cands27)
            if geom27 or best_clear27 >= 0.0:
                return _v24_precondition_failure(
                    name, args,
                    f"latest failure is not purely basket-edge limited; best_clearance={best_clear27:.1f}mm",
                    contract, state_before, recommended_next_actions=["plan_grasp", "try_another_object_or_report"],
                )

            safety0 = int(safety_trip_count.value)
            obj_before, objq_before = skill_core.object_pose(obj)
            bystanders_before27 = {}
            for other27 in ("egg_1", "egg_2", "egg_3", "egg_4"):
                if other27 == obj:
                    continue
                try:
                    bystanders_before27[other27] = list(skill_core.object_pose(other27)[0])
                except Exception:
                    pass

            ox27, oy27, oz27 = [float(v) for v in obj_before]
            bx27, by27 = [float(v) for v in V27_BASKET_CENTER_XY_M]
            dx27, dy27 = ox27 - bx27, oy27 - by27
            r027 = math.hypot(dx27, dy27)
            if r027 < 0.018:
                return _v24_precondition_failure(
                    name, args, f"{obj} is not radially edge-limited (r={r027*1000.0:.1f}mm)",
                    contract, state_before, recommended_next_actions=["observe_scene", "plan_grasp"],
                )
            V226_EDGE_REPOSITION_USED[obj] = used27 + 1
            outward27 = [dx27 / r027, dy27 / r027]
            inward27 = [-outward27[0], -outward27[1]]

            # Feedback target: recover the analytical deficit plus 3 mm margin.
            # Do NOT command that displacement in one sweep.  Physical contact can
            # amplify object motion, as observed in V2-26.
            required_gain27 = max(0.006, min(0.018, (-best_clear27 / 1000.0) + 0.003))
            center_safe_radius27 = 0.028
            opening_push27 = 12.0  # V2-28: near-closed compact two-finger pusher; avoid full 0 mm self-contact

            radial27 = math.atan2(dy27, dx27)
            tangent27 = _v27_axis_angle_pi(radial27 + math.pi / 2.0)
            qpush27 = _v27_top_down_quat(tangent27)
            prior27 = _v24_gripper_prior(obj, float(gripper_actual_mm.value))
            egg_half27 = max(0.018, float(prior27.get("effective_width_mm", 44.0)) / 2000.0)
            half_open27 = opening_push27 / 2000.0
            room27 = max(0.003, V27_BASKET_SAFE_INNER_RADIUS_M - r027 - V27_FINGER_HALF_WIDTH_M - 0.002)
            finger_center_out27 = min(egg_half27 + V27_FINGER_HALF_WIDTH_M + 0.003, room27)
            # V2-28: finger_center_out27 is now the center of the compact two-finger
            # pusher footprint, not the center of one widely separated outer finger.
            # Therefore do NOT subtract half the gripper opening from the TCP target.
            # The two fingers remain only +/-6 mm around this radial contact center.
            tcp_radial_offset27 = finger_center_out27
            sx27 = ox27 + outward27[0] * tcp_radial_offset27
            sy27 = oy27 + outward27[1] * tcp_radial_offset27
            contact_z27 = oz27 + V27_DEFAULT_GRASP_Z_OFFSET_M + 0.008
            hover_z27 = max(contact_z27 + 0.090, 0.170)
            recovery_z27 = max(0.300, hover_z27)
            hover27 = [sx27, sy27, hover_z27]
            contact27 = [sx27, sy27, contact_z27]

            warnings27 = []
            verification27 = {
                "v229_edge_reposition": True,
                "pusher_mode": "near_closed_two_finger",
                "object_name": obj,
                "attempt": int(V226_EDGE_REPOSITION_USED[obj]),
                "best_failed_basket_clearance_mm": round(best_clear27, 2),
                "required_inward_gain_mm": round(required_gain27 * 1000.0, 1),
                "basket_radius_before_mm": round(r027 * 1000.0, 2),
                "center_safe_radius_mm": round(center_safe_radius27 * 1000.0, 1),
                "pusher_opening_mm": opening_push27,
                "pusher_half_span_mm": round(half_open27 * 1000.0, 2),
                "pusher_yaw_deg": round(math.degrees(tangent27), 2),
                "finger_center_outward_offset_mm": round(finger_center_out27 * 1000.0, 2),
                "segmented_push": True,
            }
            skill_core.emit(
                f"[V2-29 EDGE REPOSITION START] {obj} | best_clearance={best_clear27:.1f}mm | "
                f"required_feedback_gain={required_gain27*1000.0:.1f}mm | segmented push"
            )

            # V2-29 entry corridor: NEVER cross from the previous work area to
            # the basket at low Z.  First preserve the current orientation and
            # retreat straight up; compact the gripper only after reaching the
            # safe height; then traverse horizontally at high Z and descend
            # vertically to the unchanged V2-28 pusher hover.
            llm_require_arm_controller()

            def _v229_cartesian_pose(target27, quat27, factor27):
                traj27 = skill_core._cartesian_approach_plan(target27, quat27)
                _v218_slow_trajectory_in_place(traj27, factor=float(factor27))
                skill_core._execute_trajectory(traj27)
                return True

            def _v227_cartesian(target27, factor27):
                return _v229_cartesian_pose(target27, qpush27, factor27)

            # V2-39 edge invariant: inspect the inherited configuration, but do
            # NOT force an invented TABLE_READY pose. High/clear neutral states
            # (including q0) are accepted. Low/fixture-near states retreat first;
            # only unhealthy joint/wrist configurations fall back to validated q0.
            skill_core.emit(f"[V2-39 EDGE SAFE-START CHECK] {obj} | SAFE_NEUTRAL policy; no forced TABLE_READY")
            _v239_task_start_posture_check(edge_mode=True)
            current_xyz29, current_q29 = skill_core.current_tcp_pose()
            # V2-35 FIX #1: the edge-reposition transit layer is an absolute
            # workspace layer, NOT max(current_z, 300 mm).  q0 is ~873 mm; using
            # that as the lateral-transit Z creates an impossible target above
            # the basket.  Always re-anchor the endpoint layer to 300 mm.
            safe_transit_z29 = 0.300
            safe_exit29 = [float(current_xyz29[0]), float(current_xyz29[1]), safe_transit_z29]
            high_hover29 = [float(sx27), float(sy27), safe_transit_z29]
            verification27["safe_transit_z_mm"] = 300.0
            verification27["safe_exit_start_xyz_m"] = [round(float(v), 5) for v in current_xyz29]
            verification27["safe_high_hover_xyz_m"] = [round(float(v), 5) for v in high_hover29]
            verification27["v235_transit_z_reanchored"] = True
            skill_core.emit(
                f"[V2-35 TRANSIT Z REANCHOR] {obj} | current_z={float(current_xyz29[2])*1000.0:.1f}mm "
                "-> transit_layer=300.0mm"
            )

            # A0. If below the 300 mm layer, keep V2-29's proven straight-up
            # vertical escape. If above it (notably q0), first TRY a planning-only
            # same-XY return to 300 mm. If that endpoint has no IK, do not restore
            # the bad high-Z behavior: defer the Z normalization to V2-34's
            # collision-checked position-first high transit, whose endpoint is
            # now forcibly on the 300 mm layer. Execution/safety failures never
            # fall through.
            dz_to_layer35 = safe_transit_z29 - float(current_xyz29[2])
            if dz_to_layer35 > 0.010:
                skill_core.emit(
                    f"[V2-29 SAFE RETREAT] {obj} | z={float(current_xyz29[2])*1000.0:.1f}mm "
                    f"-> {safe_transit_z29*1000.0:.1f}mm | vertical only"
                )
                ver_s29, ok_s29, warn_s29 = _v24_run_motion(
                    f"v229_edge_safe_retreat({obj})",
                    lambda: _v229_cartesian_pose(safe_exit29, current_q29, 2.2),
                    safe_exit29, 0.020, True,
                )
                warnings27.extend(warn_s29)
                if not ok_s29 or int(safety_trip_count.value) != safety0:
                    verification27.update(ver_s29); verification27["reason"] = "V229_SAFE_RETREAT_FAILED"
                    return _v24_result(name, False, contract, state_before, verification27, warnings27,
                        {"elapsed_s": round(time.perf_counter()-t0,3)}, next_stage=None,
                        recommended_next_actions=["observe_scene", "try_another_object_or_report"])
            elif dz_to_layer35 < -0.010:
                # V2-39: a high SAFE_NEUTRAL state such as q0 (~873 mm) is good.
                # Do NOT demand an often-unreachable same-XY vertical endpoint at
                # 300 mm. Defer both XY transfer and Z normalization to the strict
                # high-transit pose plan below, which is collision-gated and has
                # no BioIK/legacy whole-arm fallback.
                verification27["v239_high_start_reanchor"] = "defer_to_strict_high_transit"
                skill_core.emit(
                    f"[V2-39 EDGE HIGH-START DIRECT TRANSIT] {obj} | "
                    f"current_z={float(current_xyz29[2])*1000.0:.1f}mm -> high_hover_z=300.0mm | "
                    "no same-XY vertical reanchor"
                )
            else:
                verification27["safe_retreat_skipped_on_300mm_layer"] = True
                skill_core.emit(
                    f"[V2-35 TRANSIT LAYER READY] {obj} | z={float(current_xyz29[2])*1000.0:.1f}mm"
                )
            # V2-42: REUSE THE EXISTING plan_grasp TOP-DOWN MANIPULATION
            # BRANCH instead of inventing another pusher posture branch.
            #
            # The latest failed plan_grasp still contains the same top-down
            # candidate family used successfully by normal egg pickup. Basket-rim
            # geometry made those grasp footprints infeasible; their wrist
            # orientations are nevertheless valid manipulation-branch priors.
            #
            # Sequence:
            #   1) choose a plan_grasp candidate quaternion closest to the radial
            #      pusher tangent (theta or theta+180 symmetry),
            #   2) enter that branch at the familiar basket-center pregrasp layer,
            #   3) lift vertically to the fixed 300 mm transit layer,
            #   4) translate laterally to the pusher XY while preserving the
            #      ACTUAL verified top-down quaternion,
            #   5) inherit V2-34 vertical staging + V2-38 bounded J7 final trim.
            #
            # No V2-40 distal-wrist 180deg search. No V2-41 invented reorientation
            # zone. No BioIK/legacy whole-arm fallback in edge recovery.
            V242_BRANCH_ORIENT_LIMIT_DEG = 12.0
            V242_BRANCH_ORIENT_EPS_DEG = 0.10

            def _v242_quat_err(q1, q2):
                a=[float(v) for v in q1]; b=[float(v) for v in q2]
                na=math.sqrt(sum(v*v for v in a)); nb=math.sqrt(sum(v*v for v in b))
                if na < 1e-9 or nb < 1e-9: return 180.0
                d=abs(sum((a[i]/na)*(b[i]/nb) for i in range(4)))
                d=max(-1.0,min(1.0,d))
                return math.degrees(2.0*math.acos(d))

            def _v242_angle_dist(a,b):
                return abs(_v27_axis_angle_pi(float(a)-float(b)))

            # -----------------------------------------------------------------
            # A. Harvest the top-down branch family from the CURRENT plan_grasp
            # result. These candidates are allowed to be geometry-infeasible at
            # the egg itself; the branch anchor is at basket center / pregrasp Z,
            # well above contact. We deliberately do not invent a new quaternion.
            # -----------------------------------------------------------------
            branch_src42=[]
            seen_q42=set()
            for c42 in cands27:
                q42=c42.get("quat_xyzw")
                if not isinstance(q42,(list,tuple)) or len(q42)!=4:
                    continue
                try:
                    q42=[float(v) for v in q42]
                    yaw42=float(c42.get("gripper_yaw_rad"))
                except Exception:
                    try:
                        yaw42=math.radians(float(c42.get("gripper_yaw_deg")))
                    except Exception:
                        continue
                key42=tuple(round(v,6) for v in q42)
                if key42 in seen_q42: continue
                seen_q42.add(key42)
                dtheta42=min(
                    _v242_angle_dist(yaw42,float(tangent27)),
                    _v242_angle_dist(yaw42,float(tangent27)+math.pi),
                )
                try: score42=float(c42.get("score",0.0))
                except Exception: score42=0.0
                pre42=c42.get("pregrasp_xyz_m")
                try: zpre42=float(pre42[2]) if isinstance(pre42,(list,tuple)) and len(pre42)>=3 else 0.185
                except Exception: zpre42=0.185
                # Stay in the proven pregrasp-height family observed in plan_grasp.
                zpre42=max(0.175,min(0.220,zpre42))
                branch_src42.append((dtheta42,-score42,yaw42,q42,zpre42,str(c42.get("candidate_id","unknown"))))

            branch_src42.sort(key=lambda x:(x[0],x[1]))
            # More than a handful only repeats essentially identical plan_grasp
            # wrist families; keep the trial bounded and deterministic.
            branch_src42=branch_src42[:8]
            skill_core.emit(
                f"[V2-42 PLAN_GRASP BRANCH HARVEST] {obj} | candidates={len(branch_src42)} | "
                f"target_pusher_yaw={math.degrees(float(tangent27)):.1f}deg"
            )
            if not branch_src42:
                verification27["reason"]="V242_NO_PLAN_GRASP_TOPDOWN_BRANCH"
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])

            # -----------------------------------------------------------------
            # B. Enter one of the harvested branches at basket center. This is the
            # same workspace family repeatedly exercised by normal plan_grasp
            # preflight (roughly basket center, ~185 mm TCP height).
            # Strict cuRobo + MoveIt gate only.
            # -----------------------------------------------------------------
            chosen42=None
            for dtheta42,negscore42,yaw42,qbranch42,zpre42,cid42 in branch_src42:
                anchor42=[float(bx27),float(by27),float(zpre42)]
                skill_core.emit(
                    f"[V2-42 PLAN_GRASP BRANCH TRY] {obj} | source={cid42} | "
                    f"yaw={math.degrees(yaw42):.1f}deg | dpush={math.degrees(dtheta42):.1f}deg | "
                    f"anchor=[{anchor42[0]:+.3f},{anchor42[1]:+.3f},{anchor42[2]:+.3f}]"
                )
                try:
                    traj42=skill_core._plan_pose(
                        anchor42,qbranch42,
                        velocity_scale=0.14,accel_scale=0.10,
                        orientation_tolerance=0.14,position_tolerance=0.015,
                        planning_attempts=10,planning_time=10.0,
                    )
                except Exception as exc42:
                    skill_core.emit(
                        f"[V2-42 PLAN_GRASP BRANCH MISS] {obj} | source={cid42} | {exc42}"
                    )
                    continue
                ver_a42,ok_a42,warn_a42=_v24_run_motion(
                    f"v242_plan_grasp_branch_anchor({obj},{cid42})",
                    lambda traj42=traj42: skill_core._execute_trajectory(traj42),
                    anchor42,0.020,True,
                )
                warnings27.extend(warn_a42)
                if not ok_a42 or int(safety_trip_count.value)!=safety0:
                    # Execution/safety failure is not a cue to try a different
                    # whole-arm branch. Stop safely.
                    verification27.update(ver_a42)
                    verification27["reason"]="V242_BRANCH_ANCHOR_EXEC_FAILED"
                    return _v24_result(name,False,contract,state_before,verification27,warnings27,
                        {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                        recommended_next_actions=["observe_scene","try_another_object_or_report"])
                xyz_a42,q_a42=skill_core.current_tcp_pose()
                perr_a42=_v24_xyz_distance(xyz_a42,anchor42)
                oerr_a42=_v242_quat_err(q_a42,qbranch42)
                if perr_a42>0.020 or oerr_a42>(V242_BRANCH_ORIENT_LIMIT_DEG+V242_BRANCH_ORIENT_EPS_DEG):
                    verification27.update({
                        "v242_anchor_pos_error_mm":round(perr_a42*1000.0,2),
                        "v242_anchor_orient_error_deg":round(oerr_a42,3),
                        "reason":"V242_BRANCH_ANCHOR_VERIFY_FAILED",
                    })
                    return _v24_result(name,False,contract,state_before,verification27,warnings27,
                        {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                        recommended_next_actions=["observe_scene","try_another_object_or_report"])
                chosen42=(cid42,float(yaw42),list(qbranch42),list(xyz_a42),list(q_a42),float(zpre42),float(dtheta42))
                skill_core.emit(
                    f"[V2-42 PLAN_GRASP BRANCH REACHED] {obj} | source={cid42} | "
                    f"pos={perr_a42*1000.0:.1f}mm | orient={oerr_a42:.2f}deg"
                )
                break

            if chosen42 is None:
                verification27["reason"]="V242_NO_STRICT_PLAN_GRASP_BRANCH_ROUTE"
                skill_core.emit(
                    f"[V2-42 PLAN_GRASP BRANCH SAFE FAIL] {obj} | no harvested top-down branch reachable; no fallback"
                )
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])

            cid42,yaw_branch42,qnom_branch42,xyz_anchor_actual42,q_anchor_actual42,zpre42,dtheta42=chosen42

            # -----------------------------------------------------------------
            # C. V2-43: once the proven plan_grasp branch is physically reached,
            # make the gripper compact BEFORE edge transit, then move to the 300mm
            # layer with a FREE whole-arm cuRobo path.  The endpoint keeps the
            # ACTUAL verified plan_grasp quaternion, but the TCP is NOT forced to
            # follow a straight vertical Cartesian line (V2-42 log showed only
            # 26.1% of that line was collision-free).
            # -----------------------------------------------------------------
            hold_off_request.value=True
            gripper_force_mode.value=False
            gripper_target_mm.value=float(opening_push27)
            deadline43=time.time()+4.0
            while time.time()<deadline43 and abs(float(gripper_actual_mm.value)-float(opening_push27))>3.0:
                time.sleep(0.05)
            openerr43=abs(float(gripper_actual_mm.value)-float(opening_push27))
            skill_core.emit(
                f"[V2-43 COMPACT PUSHER] {obj} | target={opening_push27:.1f}mm | "
                f"actual={float(gripper_actual_mm.value):.1f}mm | err={openerr43:.1f}mm | hold={bool(hold_active_shared.value)}"
            )
            if openerr43>5.0 or bool(hold_active_shared.value):
                verification27.update({
                    "v243_pusher_target_opening_mm":round(float(opening_push27),2),
                    "v243_pusher_actual_opening_mm":round(float(gripper_actual_mm.value),2),
                    "v243_pusher_opening_error_mm":round(openerr43,2),
                    "hold_active":bool(hold_active_shared.value),
                    "reason":"V243_COMPACT_PUSHER_NOT_READY",
                })
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])

            center30042=[float(bx27),float(by27),float(safe_transit_z29)]
            skill_core.emit(
                f"[V2-43 BRANCH FREE-PATH LIFT] {obj} | {zpre42*1000.0:.0f}->{safe_transit_z29*1000.0:.0f}mm | "
                "same endpoint XY + actual plan_grasp quaternion | path=FREE"
            )
            try:
                traj_l43=skill_core._plan_pose(
                    center30042,q_anchor_actual42,
                    velocity_scale=0.14,accel_scale=0.10,
                    orientation_tolerance=0.14,position_tolerance=0.015,
                    planning_attempts=10,planning_time=10.0,
                )
            except Exception as exc43:
                verification27.update({
                    "v243_freepath_lift_error":str(exc43),
                    "reason":"V243_BRANCH_FREEPATH_300_NO_STRICT_PLAN",
                })
                skill_core.emit(f"[V2-43 BRANCH FREE-PATH SAFE FAIL] {obj} | {exc43}")
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])

            ver_l43,ok_l43,warn_l43=_v24_run_motion(
                f"v243_branch_freepath_300({obj})",
                lambda traj_l43=traj_l43: skill_core._execute_trajectory(traj_l43),
                center30042,0.020,True,
            )
            warnings27.extend(warn_l43)
            if not ok_l43 or int(safety_trip_count.value)!=safety0:
                verification27.update(ver_l43); verification27["reason"]="V243_BRANCH_FREEPATH_300_EXEC_FAILED"
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])
            xyz30042,q30042=skill_core.current_tcp_pose()
            poserr_l43=_v24_xyz_distance(xyz30042,center30042)
            oerr_l43=_v242_quat_err(q30042,q_anchor_actual42)
            if poserr_l43>0.020 or oerr_l43>(V242_BRANCH_ORIENT_LIMIT_DEG+V242_BRANCH_ORIENT_EPS_DEG):
                verification27.update({
                    "v243_freepath_pos_error_mm":round(poserr_l43*1000.0,2),
                    "v243_freepath_orient_error_deg":round(oerr_l43,3),
                    "reason":"V243_BRANCH_FREEPATH_300_VERIFY_FAILED",
                })
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])
            skill_core.emit(
                f"[V2-43 BRANCH FREE-PATH REACHED] {obj} | pos={poserr_l43*1000.0:.1f}mm | "
                f"orient={oerr_l43:.2f}deg | opening={float(gripper_actual_mm.value):.1f}mm"
            )

            # -----------------------------------------------------------------
            # D. Translate from basket-center 300 mm to the egg's outer pusher XY
            # while preserving the actual plan_grasp-branch quaternion. Because
            # the large posture change is already complete, this is a normal
            # top-down manipulation transit. Strict cuRobo + MoveIt only.
            # -----------------------------------------------------------------
            skill_core.emit(
                f"[V2-42 TOPDOWN BRANCH TRANSIT] {obj} | center300 -> target_xy=({sx27:+.3f},{sy27:+.3f}) | "
                f"source={cid42} | preserve verified plan_grasp branch"
            )
            try:
                traj_t42=skill_core._plan_pose(
                    high_hover29,q30042,
                    velocity_scale=0.14,accel_scale=0.10,
                    orientation_tolerance=0.14,position_tolerance=0.015,
                    planning_attempts=10,planning_time=10.0,
                )
            except Exception as exc42:
                verification27["reason"]="V242_TOPDOWN_BRANCH_TRANSIT_NO_STRICT_PLAN"
                verification27["v242_topdown_transit_error"]=str(exc42)
                skill_core.emit(f"[V2-42 TOPDOWN BRANCH TRANSIT SAFE FAIL] {obj} | {exc42}")
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])
            ver_t42,ok_t42,warn_t42=_v24_run_motion(
                f"v242_topdown_branch_transit({obj})",
                lambda traj_t42=traj_t42: skill_core._execute_trajectory(traj_t42),
                high_hover29,0.022,True,
            )
            warnings27.extend(warn_t42)
            if not ok_t42 or int(safety_trip_count.value)!=safety0:
                verification27.update(ver_t42); verification27["reason"]="V242_TOPDOWN_BRANCH_TRANSIT_EXEC_FAILED"
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])
            xyz_edge42,q_edge42=skill_core.current_tcp_pose()
            edge_poserr42=_v24_xyz_distance(xyz_edge42,high_hover29)
            branch_drift42=_v242_quat_err(q_edge42,q30042)
            if edge_poserr42>0.022 or branch_drift42>12.0:
                verification27.update({
                    "v242_edge_pos_error_mm":round(edge_poserr42*1000.0,2),
                    "v242_branch_orientation_drift_deg":round(branch_drift42,3),
                    "reason":"V242_TOPDOWN_BRANCH_TRANSIT_VERIFY_FAILED",
                })
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])

            # Inherited V2-34 staging descent now preserves this known-good
            # top-down manipulation branch. Leave qpush27/tangent27 as the IDEAL
            # pusher target so V2-38's bounded J7 trim can remove the remaining
            # small yaw error at staging.
            transit_q34=[float(v) for v in q_edge42]
            verification27.update({
                "v242_plan_grasp_branch_reuse":True,
                "v242_branch_source_candidate":cid42,
                "v242_branch_yaw_deg":round(math.degrees(yaw_branch42),2),
                "v242_branch_to_pusher_axis_delta_deg":round(math.degrees(dtheta42),2),
                "v242_anchor_z_mm":round(zpre42*1000.0,1),
                "v242_edge_branch_drift_deg":round(branch_drift42,3),
                "v234_high_transit_planner":"plan_grasp_branch_reuse_strict_curobo",
            })
            skill_core.emit(
                f"[V2-42 EDGE READY FOR STAGING] {obj} | plan_grasp branch={cid42} | "
                f"branch_yaw={math.degrees(yaw_branch42):.1f}deg | next=vertical staging + bounded J7 trim"
            )

            if safe_transit_z29 - staging_z34 > 0.010:
                skill_core.emit(
                    f"[V2-34 STAGING DESCENT] {obj} | z={safe_transit_z29*1000.0:.1f}mm "
                    f"-> {staging_z34*1000.0:.1f}mm | preserve transit orientation"
                )
                ver_sd34, ok_sd34, warn_sd34 = _v24_run_motion(
                    f"v234_edge_staging_descend({obj})",
                    lambda: _v229_cartesian_pose(staging34, transit_q34, 2.4),
                    staging34, 0.020, True,
                )
                warnings27.extend(warn_sd34)
                if not ok_sd34 or int(safety_trip_count.value) != safety0:
                    verification27.update(ver_sd34); verification27["reason"] = "V234_STAGING_DESCENT_FAILED"
                    return _v24_result(name, False, contract, state_before, verification27, warnings27,
                        {"elapsed_s": round(time.perf_counter()-t0,3)}, next_stage=None,
                        recommended_next_actions=["observe_scene", "try_another_object_or_report"])

            # V2-38: do NOT ask a general 6D pose planner to twist the whole
            # arm above the basket. Task-start normalization + V2-34 transit have
            # already delivered a top-down wrist family. At staging, pusher yaw
            # may change ONLY through J7. Search both theta/theta+180 symmetries
            # with FK + MoveIt state validation, then execute a bounded J7 path.
            V238_PUSHER_ORIENT_LIMIT_DEG = 10.0
            V238_PUSHER_ORIENT_EPS_DEG = 0.05

            def _v238_quat_err_edge(q1, q2):
                a=[float(v) for v in q1]; b=[float(v) for v in q2]
                na=math.sqrt(sum(v*v for v in a)); nb=math.sqrt(sum(v*v for v in b))
                if na < 1e-9 or nb < 1e-9: return 180.0
                d=abs(sum((a[i]/na)*(b[i]/nb) for i in range(4)))
                d=max(-1.0,min(1.0,d))
                return math.degrees(2.0*math.acos(d))

            live38 = skill_core._current_joint_state_copy()
            qcur38 = [float(v) for v in skill_core._arm_positions_from_joint_state(live38)]
            if len(qcur38) < 7:
                verification27["reason"]="V238_J7_SEARCH_NO_7DOF_STATE"
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])
            qcur38=qcur38[:7]
            yaw_targets38=[
                ("theta",float(tangent27),list(qpush27)),
                ("theta+180",float(tangent27)+math.pi,_v27_top_down_quat(float(tangent27)+math.pi)),
            ]
            qbatch38=[]; meta38=[]
            try:
                lo7=float(JOINT_LIMITS_RAD[6][0]); hi7=float(JOINT_LIMITS_RAD[6][1])
            except Exception:
                lo7=-math.pi; hi7=math.pi
            # Search a bounded distal-wrist neighborhood. No J1-J6 change.
            for ddeg38 in range(-180,181,5):
                q7=float(qcur38[6])+math.radians(float(ddeg38))
                if q7 < lo7+math.radians(2.0) or q7 > hi7-math.radians(2.0):
                    continue
                qv38=list(qcur38); qv38[6]=q7
                qbatch38.append(qv38); meta38.append(float(ddeg38))
            client38=getattr(skill_core,"_gpu_client",None)
            if client38 is None:
                verification27["reason"]="V238_J7_SEARCH_GPU_FK_UNAVAILABLE"
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])
            out38=client38.fk_batch(
                joint_names=[f"joint{i}" for i in range(1,8)],q_batch=qbatch38,timeout_s=20.0,
            )
            poses38=out38.get("poses") or [] if bool(out38.get("ok")) else []
            if len(poses38)!=len(qbatch38):
                verification27["reason"]="V238_J7_SEARCH_FK_FAILED"
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])

            candidates38=[]
            for qv38,pose38,ddeg38 in zip(qbatch38,poses38,meta38):
                okstate38,why38=skill_core._v71_state_is_valid(qv38)
                if not okstate38: continue
                pos38=[float(v) for v in pose38["position"]]
                if _v24_xyz_distance(pos38,staging34) > 0.020: continue
                pq38=[float(v) for v in pose38["quat_xyzw"]]
                for tag38,yaw38,qgoal38 in yaw_targets38:
                    oe38=_v238_quat_err_edge(pq38,qgoal38)
                    if oe38 <= (V238_PUSHER_ORIENT_LIMIT_DEG+V238_PUSHER_ORIENT_EPS_DEG):
                        candidates38.append((oe38,abs(ddeg38),tag38,yaw38,qgoal38,qv38,pose38,ddeg38))
            skill_core.emit(
                f"[V2-38 PUSHER J7 SEARCH] {obj} | samples={len(qbatch38)} | valid_candidates={len(candidates38)} | "
                f"yaw0={math.degrees(float(tangent27)):.1f}deg"
            )
            if not candidates38:
                verification27["reason"]="V238_PUSHER_J7_NO_SAFE_CANDIDATE"
                skill_core.emit(f"[V2-38 PUSHER SAFE FAIL] {obj} | no J7-only top-down pusher solution; no full-arm fallback")
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])
            candidates38.sort(key=lambda x:(x[0],x[1],0 if x[2]=="theta" else 1))
            oe38,_,tag38,yaw38,qgoal_quat38,qtarget38,pose_target38,ddeg38=candidates38[0]
            # Build a small, explicitly validated J7-only joint path.
            maxd38=abs(float(qtarget38[6])-float(qcur38[6]))
            n38=max(1,int(math.ceil(maxd38/math.radians(4.0))))
            qpath38=[]
            for k38 in range(n38+1):
                a38=float(k38)/float(n38)
                qv38=list(qcur38); qv38[6]=float(qcur38[6])+a38*(float(qtarget38[6])-float(qcur38[6]))
                qpath38.append(qv38)
            for i38,qv38 in enumerate(qpath38):
                ok38,why38=skill_core._v71_state_is_valid(qv38)
                if not ok38:
                    verification27["reason"]="V238_PUSHER_J7_PATH_REJECTED"
                    verification27["v238_j7_path_error"]=f"{i38}/{len(qpath38)-1}: {why38}"
                    return _v24_result(name,False,contract,state_before,verification27,warnings27,
                        {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                        recommended_next_actions=["observe_scene","try_another_object_or_report"])
            outp38=client38.fk_batch(
                joint_names=[f"joint{i}" for i in range(1,8)],q_batch=qpath38,timeout_s=20.0,
            )
            pathposes38=outp38.get("poses") or [] if bool(outp38.get("ok")) else []
            if len(pathposes38)!=len(qpath38):
                verification27["reason"]="V238_PUSHER_J7_PATH_FK_FAILED"
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])
            if min(float(p["position"][2]) for p in pathposes38) < 0.200:
                verification27["reason"]="V238_PUSHER_J7_PATH_Z_FLOOR"
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])
            if max(_v24_xyz_distance([float(v) for v in p["position"]],staging34) for p in pathposes38) > 0.030:
                verification27["reason"]="V238_PUSHER_J7_TCP_DRIFT"
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])
            xyzpath38=[[float(v) for v in p["position"]] for p in pathposes38]
            traj38=skill_core._make_incremental_joint_trajectory(qpath38,xyzpath38)
            if traj38 is None:
                verification27["reason"]="V238_PUSHER_J7_NO_TRAJECTORY"
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])
            _v218_slow_trajectory_in_place(traj38,factor=1.35)
            skill_core.emit(
                f"[V2-38 PUSHER J7 EXEC] {obj} | candidate={tag38} | delta_J7={ddeg38:+.1f}deg | "
                f"predicted_orient_error={oe38:.2f}deg | points={len(qpath38)}"
            )
            ver38,ok38,warn38=_v24_run_motion(
                f"v238_edge_pusher_j7({obj},{tag38})",
                lambda traj38=traj38: skill_core._execute_trajectory(traj38),
                xyzpath38[-1],0.020,True,
            )
            warnings27.extend(warn38)
            actual_xyz38,actual_q38=skill_core.current_tcp_pose()
            poserr38=_v24_xyz_distance(actual_xyz38,staging34)
            oerr38=_v238_quat_err_edge(actual_q38,qgoal_quat38)
            qact38=[float(v) for v in skill_core._arm_positions_from_joint_state(skill_core._current_joint_state_copy())][:7]
            jerr38=max(abs(math.degrees(float(qact38[i])-float(qtarget38[i]))) for i in range(7))
            strict38=bool(
                ok38 and poserr38<=0.020
                and oerr38<=(V238_PUSHER_ORIENT_LIMIT_DEG+V238_PUSHER_ORIENT_EPS_DEG)
                and jerr38<=4.0 and int(safety_trip_count.value)==safety0
                and not bool(hold_active_shared.value)
            )
            verification27.update({
                "v238_pusher_mode":"J7_only",
                "v238_pusher_candidate":tag38,
                "v238_pusher_delta_j7_deg":round(ddeg38,2),
                "v235_pusher_position_error_mm":round(poserr38*1000.0,2),
                "v235_pusher_orientation_error_deg":round(oerr38,3),
                "v235_pusher_max_joint_tracking_error_deg":round(jerr38,2),
                "v235_pusher_position_limit_mm":20.0,
                "v235_pusher_orientation_limit_deg":V238_PUSHER_ORIENT_LIMIT_DEG,
                "v235_pusher_joint_tracking_limit_deg":4.0,
                "v235_pusher_strict_pose_ok":strict38,
            })
            skill_core.emit(
                f"[V2-38 PUSHER STRICT VERIFY] {obj} | pos={poserr38*1000.0:.1f}/20.0mm | "
                f"orient={oerr38:.2f}/{V238_PUSHER_ORIENT_LIMIT_DEG:.1f}deg | joint={jerr38:.2f}/4.0deg | "
                f"strict_ok={strict38}"
            )
            if not strict38:
                verification27.update(ver38); verification27["reason"]="V238_PUSHER_J7_STRICT_VERIFY_FAILED"
                return _v24_result(name,False,contract,state_before,verification27,warnings27,
                    {"elapsed_s":round(time.perf_counter()-t0,3)},next_stage=None,
                    recommended_next_actions=["observe_scene","try_another_object_or_report"])

            # Critical: subsequent vertical approach/push preserves the ACTUAL
            # verified wrist quaternion. It must not reintroduce a hidden full-pose
            # correction toward an ideal theoretical quaternion during descent.
            tangent27=float(yaw38)
            qpush27=[float(v) for v in actual_q38]
            verification27["pusher_yaw_deg"]=round(math.degrees(tangent27),2)
            verification27["v234_pusher_orient_planner"]="bounded_J7_FK_MoveIt"
            skill_core.emit(
                f"[V2-38 PUSHER ORIENT VERIFIED] {obj} | J7-only | candidate={tag38} | "
                f"yaw={math.degrees(tangent27):.1f}deg | descent preserves actual quaternion"
            )
            warnings27.extend(warn_t29)
            if not ok_t29 or int(safety_trip_count.value) != safety0:
                verification27.update(ver_t29); verification27["reason"] = "V229_HIGH_TRANSIT_FAILED"
                return _v24_result(name, False, contract, state_before, verification27, warnings27,
                    {"elapsed_s": round(time.perf_counter()-t0,3)}, next_stage=None,
                    recommended_next_actions=["observe_scene", "try_another_object_or_report"])

            # A2. Straight vertical descent from high transit to the unchanged
            # V2-28 hover pose.  No free 3-D global path is used here.
            skill_core.emit(
                f"[V2-29 VERTICAL APPROACH] {obj} | z={safe_transit_z29*1000.0:.1f}mm "
                f"-> {hover_z27*1000.0:.1f}mm"
            )
            ver_h27, ok_h27, warn_h27 = _v24_run_motion(
                f"v229_edge_hover_descend({obj})",
                lambda: _v227_cartesian(hover27, 2.4),
                hover27, 0.018, True,
            )
            warnings27.extend(warn_h27)
            if not ok_h27 or int(safety_trip_count.value) != safety0:
                verification27.update(ver_h27); verification27["reason"] = "V229_EDGE_HOVER_FAILED"
                return _v24_result(name, False, contract, state_before, verification27, warnings27,
                    {"elapsed_s": round(time.perf_counter()-t0,3)}, next_stage=None,
                    recommended_next_actions=["observe_scene", "try_another_object_or_report"])

            ver_d27, ok_d27, warn_d27 = _v24_run_motion(
                f"v227_edge_descend({obj})",
                lambda: _v227_cartesian(contact27, 2.4),
                contact27, 0.016, True,
            )
            warnings27.extend(warn_d27)
            if not ok_d27 or int(safety_trip_count.value) != safety0:
                verification27.update(ver_d27); verification27["reason"] = "V229_EDGE_DESCEND_FAILED"
                return _v24_result(name, False, contract, state_before, verification27, warnings27,
                    {"elapsed_s": round(time.perf_counter()-t0,3)}, next_stage=None,
                    recommended_next_actions=["observe_scene", "try_another_object_or_report"])

            # B. Closed-loop segmented push.  First step 8 mm; later corrections
            # are 5-6 mm.  Re-read MuJoCo object pose after EVERY segment.
            commanded27 = 0.0
            push_steps27 = []
            cursor27 = list(contact27)
            push_ok27 = True
            reached27 = False
            step_schedule27 = [0.008, 0.006, 0.005]
            last_r27 = r027
            for idx27, nominal27 in enumerate(step_schedule27, start=1):
                obj_now27, _oq27 = skill_core.object_pose(obj)
                r_now27 = math.hypot(float(obj_now27[0])-bx27, float(obj_now27[1])-by27)
                gain_now27 = r027 - r_now27
                if gain_now27 >= required_gain27 or r_now27 <= center_safe_radius27:
                    reached27 = True
                    skill_core.emit(
                        f"[V2-29 EDGE FEEDBACK STOP] {obj} | before step {idx27} | "
                        f"gain={gain_now27*1000.0:.1f}mm | radius={r_now27*1000.0:.1f}mm"
                    )
                    break
                remaining27 = max(0.0, required_gain27 - gain_now27)
                if idx27 == 1:
                    step27 = min(0.010, max(0.008, remaining27 + 0.002))
                else:
                    step27 = min(nominal27, max(0.004, remaining27 + 0.001))
                # hard cap keeps the total commanded sweep below the old 19.6 mm case.
                step27 = min(step27, max(0.0, 0.019 - commanded27))
                if step27 < 0.0035:
                    break
                cursor27 = [
                    cursor27[0] + inward27[0] * step27,
                    cursor27[1] + inward27[1] * step27,
                    contact_z27,
                ]
                skill_core.emit(
                    f"[V2-29 EDGE PUSH STEP] {obj} | step={idx27} | command={step27*1000.0:.1f}mm | "
                    f"previous_gain={gain_now27*1000.0:.1f}mm"
                )
                ver_p27, ok_p27, warn_p27 = _v24_run_motion(
                    f"v227_edge_push_{idx27}({obj})",
                    lambda target27=list(cursor27): _v227_cartesian(target27, 3.2),
                    cursor27, 0.018, True,
                )
                warnings27.extend(warn_p27)
                commanded27 += step27
                time.sleep(0.18)
                obj_after_step27, _oqs27 = skill_core.object_pose(obj)
                r_after_step27 = math.hypot(float(obj_after_step27[0])-bx27, float(obj_after_step27[1])-by27)
                gain_after_step27 = r027 - r_after_step27
                push_steps27.append({
                    "step": idx27,
                    "command_mm": round(step27*1000.0, 2),
                    "cumulative_command_mm": round(commanded27*1000.0, 2),
                    "radius_mm": round(r_after_step27*1000.0, 2),
                    "inward_gain_mm": round(gain_after_step27*1000.0, 2),
                    "motion_ok": bool(ok_p27),
                })
                skill_core.emit(
                    f"[V2-29 EDGE OBSERVE] {obj} | step={idx27} | "
                    f"radius={r_after_step27*1000.0:.1f}mm | gain={gain_after_step27*1000.0:.1f}mm"
                )
                if not ok_p27 or int(safety_trip_count.value) != safety0:
                    push_ok27 = False
                    break
                # Wrong-way motion guard.
                if r_after_step27 > last_r27 + 0.002:
                    warnings27.append(
                        f"V2-29 pusher moved {obj} outward by more than 2 mm on step {idx27}; stopping contact motion."
                    )
                    push_ok27 = False
                    break
                last_r27 = r_after_step27
                if gain_after_step27 >= required_gain27 or r_after_step27 <= center_safe_radius27:
                    reached27 = True
                    skill_core.emit(
                        f"[V2-29 EDGE FEEDBACK STOP] {obj} | after step {idx27} | "
                        f"gain={gain_after_step27*1000.0:.1f}mm | radius={r_after_step27*1000.0:.1f}mm"
                    )
                    break

            verification27["push_steps"] = push_steps27
            verification27["total_commanded_push_mm"] = round(commanded27*1000.0, 2)

            # C. First get clear of the basket: vertical retreat to >=300 mm.
            retreat_high27 = [cursor27[0], cursor27[1], recovery_z27]
            retreat_ok27 = False
            if int(safety_trip_count.value) == safety0:
                ver_r27, retreat_ok27, warn_r27 = _v24_run_motion(
                    f"v227_edge_retreat_high({obj})",
                    lambda: _v227_cartesian(retreat_high27, 2.2),
                    retreat_high27, 0.022, True,
                )
                warnings27.extend(warn_r27)
            else:
                ver_r27 = {"reason": "SAFETY_TRIP_CHANGED_BEFORE_RETREAT"}

            # -------- posture recovery helpers (no held object) --------
            def _v227_interp_q(q0, q1, step_deg=4.0):
                md = max(abs(float(q1[i])-float(q0[i])) for i in range(7))
                n = max(1, int(math.ceil(md / math.radians(float(step_deg)))))
                return [
                    [float(q0[j]) + (float(q1[j])-float(q0[j]))*(float(k)/float(n)) for j in range(7)]
                    for k in range(n+1)
                ]

            def _v227_fk_batch(qbatch):
                client = getattr(skill_core, "_gpu_client", None)
                if client is None:
                    raise RuntimeError("GPU client unavailable for posture-recovery FK")
                out = client.fk_batch(
                    joint_names=[f"joint{i}" for i in range(1,8)], q_batch=qbatch, timeout_s=20.0,
                )
                if not bool(out.get("ok")):
                    raise RuntimeError("cuRobo FK failed: " + str(out.get("error")))
                poses = out.get("poses") or []
                if len(poses) != len(qbatch):
                    raise RuntimeError(f"FK count mismatch {len(poses)} != {len(qbatch)}")
                return poses

            def _v227_validate_qpath(qpath, min_tcp_z=0.145):
                for qi, qv in enumerate(qpath):
                    okq, whyq = skill_core._v71_state_is_valid(qv)
                    if not okq:
                        return False, f"MoveIt rejected point {qi}/{len(qpath)-1}: {whyq}", None
                poses = _v227_fk_batch(qpath)
                minz = min(float(p["position"][2]) for p in poses)
                if minz < float(min_tcp_z):
                    return False, f"TCP path dips to {minz*1000.0:.1f}mm", poses
                return True, "OK", poses

            def _v227_exec_qpath(label, qpath, poses, slow_factor):
                xyzpath = [[float(v) for v in p["position"]] for p in poses]
                traj = skill_core._make_incremental_joint_trajectory(qpath, xyzpath)
                if traj is None:
                    return False, {"reason":"NO_TRAJECTORY"}, []
                if slow_factor > 1.0:
                    for pt in traj.joint_trajectory.points:
                        ts = float(pt.time_from_start.sec) + float(pt.time_from_start.nanosec)*1e-9
                        ts *= float(slow_factor)
                        si = int(math.floor(ts))
                        pt.time_from_start.sec = si
                        pt.time_from_start.nanosec = int(round((ts-si)*1e9))
                return_tuple = _v24_run_motion(
                    label, lambda traj=traj: skill_core._execute_trajectory(traj),
                    xyzpath[-1], 0.040, True,
                )
                ver, ok, warn = return_tuple
                return bool(ok), ver, warn

            posture_ok27 = False
            posture_strategy27 = None
            posture_detail27 = None
            if retreat_ok27 and int(safety_trip_count.value) == safety0:
                try:
                    live27 = skill_core._current_joint_state_copy()
                    qcur27 = [float(v) for v in skill_core._arm_positions_from_joint_state(live27)]
                    # Prefer a partial move toward q=0 that is already high and neutral.
                    for alpha27 in (0.55, 0.70, 0.85):
                        qtarget27 = [(1.0-alpha27)*float(q) for q in qcur27]
                        qpath27 = _v227_interp_q(qcur27, qtarget27, step_deg=4.0)
                        valid27, why27, poses27 = _v227_validate_qpath(qpath27, min_tcp_z=0.145)
                        if not valid27:
                            continue
                        finalz27 = float(poses27[-1]["position"][2])
                        if finalz27 < 0.38:
                            continue
                        skill_core.emit(
                            f"[V2-28 POSTURE RECOVERY] high-neutral candidate alpha={alpha27:.2f} | "
                            f"points={len(qpath27)} | final_tcp_z={finalz27*1000.0:.0f}mm"
                        )
                        okq27, verq27, warnq27 = _v227_exec_qpath(
                            f"v227_high_neutral({obj})", qpath27, poses27, slow_factor=1.35,
                        )
                        warnings27.extend(warnq27)
                        if okq27 and int(safety_trip_count.value) == safety0:
                            posture_ok27 = True
                            posture_strategy27 = "high_neutral_partial_q0"
                            posture_detail27 = {"alpha_to_q0": alpha27, "final_tcp_z_m": finalz27}
                            skill_core.emit(
                                f"[V2-28 HIGH NEUTRAL REACHED] alpha={alpha27:.2f} | ready for fresh observe/replan"
                            )
                            break
                except Exception as exc27:
                    warnings27.append(f"V2-28 high-neutral recovery failed: {exc27}")

            # Full q=0 is the slower validated fallback, exactly in the spirit of
            # V2-24 Strategy 3 but with no object held.
            if retreat_ok27 and not posture_ok27 and int(safety_trip_count.value) == safety0:
                try:
                    live27 = skill_core._current_joint_state_copy()
                    qcur27 = [float(v) for v in skill_core._arm_positions_from_joint_state(live27)]
                    qzero27 = [0.0]*7
                    qpath27 = _v227_interp_q(qcur27, qzero27, step_deg=4.0)
                    valid27, why27, poses27 = _v227_validate_qpath(qpath27, min_tcp_z=0.145)
                    if not valid27:
                        raise RuntimeError("q=0 path rejected: " + why27)
                    skill_core.emit(
                        f"[V2-28 Q0 FALLBACK] validated {len(qpath27)} joint points | slow_factor=1.65"
                    )
                    okq27, verq27, warnq27 = _v227_exec_qpath(
                        f"v227_q0_recovery({obj})", qpath27, poses27, slow_factor=1.65,
                    )
                    warnings27.extend(warnq27)
                    if not okq27 or int(safety_trip_count.value) != safety0:
                        raise RuntimeError("q=0 execution failed")
                    posture_ok27 = True
                    posture_strategy27 = "full_q0"
                    posture_detail27 = {"final_tcp_z_m": float(poses27[-1]["position"][2])}
                    skill_core.emit("[V2-28 Q0 REACHED] q=[0,0,0,0,0,0,0] | ready for fresh observe/replan")
                except Exception as exc27:
                    warnings27.append(f"V2-28 q0 posture recovery failed: {exc27}")

            # Final measured object/bystander state after contact and posture recovery.
            obj_after27, objq_after27 = skill_core.object_pose(obj)
            r127 = math.hypot(float(obj_after27[0])-bx27, float(obj_after27[1])-by27)
            inward_gain27 = r027 - r127
            move27 = _v24_xyz_distance(obj_after27, obj_before)
            rot27 = _v27_quat_rotation_angle_deg(objq_after27, objq_before)
            max_bystander27 = 0.0
            max_bystander_name27 = None
            for other27, before27 in bystanders_before27.items():
                try:
                    after27 = skill_core.object_pose(other27)[0]
                    dm27 = _v24_xyz_distance(after27, before27)
                    if dm27 > max_bystander27:
                        max_bystander27 = dm27
                        max_bystander_name27 = other27
                except Exception:
                    pass
            if max_bystander27 > 0.008:
                warnings27.append(
                    f"V2-28 edge reposition moved bystander {max_bystander_name27} by {max_bystander27*1000.0:.1f}mm; fresh observe_scene will re-evaluate the basket."
                )
            verification27.update({
                "basket_radius_after_mm": round(r127*1000.0,2),
                "inward_gain_mm": round(inward_gain27*1000.0,2),
                "object_motion_mm": round(move27*1000.0,2),
                "object_rotation_deg": round(rot27,2),
                "max_bystander_motion_mm": round(max_bystander27*1000.0,2),
                "max_bystander_object": max_bystander_name27,
                "push_feedback_goal_reached": bool(reached27 or inward_gain27 >= required_gain27 or r127 <= center_safe_radius27),
                "retreat_high_ok": bool(retreat_ok27),
                "posture_recovery_ok": bool(posture_ok27),
                "posture_recovery_strategy": posture_strategy27,
                "posture_recovery_detail": posture_detail27,
                "safety_trip_before": safety0,
                "safety_trip_after": int(safety_trip_count.value),
                "hold_active": bool(hold_active_shared.value),
                "fresh_observation_required": True,
            })

            # Any contact invalidates all old grasp candidates.
            V27_GRASP_RUNTIME.get("plans", {}).pop(obj, None)
            V27_GRASP_RUNTIME.get("selected", {}).pop(obj, None)
            V27_GRASP_RUNTIME.setdefault("tried", {})[obj] = set()

            goal_reached27 = bool(reached27 or inward_gain27 >= required_gain27 or r127 <= center_safe_radius27)
            ok27 = bool(
                push_ok27 and goal_reached27 and retreat_ok27 and posture_ok27
                and max_bystander27 <= 0.020
                and int(safety_trip_count.value) == safety0
                and not bool(hold_active_shared.value)
            )
            if ok27:
                skill_core.emit(
                    f"[V2-29 EDGE REPOSITION SUCCESS] {obj} | commanded={commanded27*1000.0:.1f}mm | "
                    f"measured_gain={inward_gain27*1000.0:.1f}mm | posture={posture_strategy27} | "
                    "old grasp plan INVALIDATED | MUST observe_scene before replanning"
                )
                return _v24_result(
                    name, True, contract, state_before, verification27, warnings27,
                    {"object_name": obj, "elapsed_s": round(time.perf_counter()-t0,3)},
                    next_stage=V24_STAGE_IDLE, active_object=None, target_bin=None,
                    recommended_next_actions=["observe_scene"],
                )

            verification27["reason"] = "V229_EDGE_REPOSITION_OR_POSTURE_RECOVERY_NOT_VERIFIED"
            skill_core.emit(
                f"[V2-29 EDGE REPOSITION INCOMPLETE] {obj} | commanded={commanded27*1000.0:.1f}mm | "
                f"gain={inward_gain27*1000.0:.1f}mm | posture_ok={posture_ok27} | fresh observation required"
            )
            return _v24_result(
                name, False, contract, state_before, verification27, warnings27,
                {"object_name": obj, "elapsed_s": round(time.perf_counter()-t0,3)},
                next_stage=V24_STAGE_IDLE, active_object=None, target_bin=None,
                recommended_next_actions=["observe_scene", "try_another_object_or_report"],
            )

        # ------------------------------------------------------------
        # NERO V2-7 Constraint-Aware Grasp Planning
        # ------------------------------------------------------------
        if name == "plan_grasp":
            stage = state_before.get("stage")
            if stage not in (V24_STAGE_OBSERVED, V24_STAGE_HOVER, V24_STAGE_VERIFIED, V24_STAGE_RELEASED):
                return _v24_precondition_failure(
                    name, args,
                    f"plan_grasp not allowed from state={stage}; observe/recover first",
                    contract, state_before,
                    recommended_next_actions=["observe_scene"],
                )
            if bool(hold_active_shared.value):
                return _v24_precondition_failure(
                    name, args, "hold_active=true; cannot plan a new pickup while holding an object",
                    contract, state_before,
                    recommended_next_actions=["resolve_or_release_held_object"],
                )
            obj = str(args.get("object_name", ""))
            if obj not in OBJECTS:
                return _v24_precondition_failure(
                    name, args, f"unknown object {obj}; valid={list(OBJECTS)}",
                    contract, state_before, recommended_next_actions=["observe_scene"],
                )
            if not obj.startswith("egg"):
                return _v24_precondition_failure(
                    name, args, f"V2-7 plan_grasp currently supports egg objects only; got {obj}",
                    contract, state_before, recommended_next_actions=["use_legacy_skill_or_stop"],
                )
            aperture = _v24_aperture_check(obj)
            # ============================================================
            # V3.06 AUTO APERTURE IN PLAN_GRASP
            # ------------------------------------------------------------
            # V3.05 spent one Agent step per egg on set_gripper_opening (the
            # gripper is ~90 mm after every release).  plan_grasp now prepares
            # the recommended aperture itself (position mode, fingers only),
            # then re-checks.  Disable with NERO_V306_AUTO_APERTURE=0.
            # ============================================================
            auto_ap306 = None
            if (not aperture["robust_aperture_ok"]) and os.environ.get("NERO_V306_AUTO_APERTURE", "1") != "0":
                tgt306 = float(aperture.get("recommended_target_opening_mm") or 60.0)
                skill_core.emit(f"[V3.06 AUTO APERTURE] {obj} | {aperture['actual_opening_mm']:.1f}mm -> {tgt306:.1f}mm inside plan_grasp")
                try:
                    r306 = llm_tool_runner("set_gripper_opening", {"target_opening_mm": tgt306})
                    auto_ap306 = {"target_opening_mm": round(tgt306, 2), "ok": bool(r306.get("ok")),
                                  "opening_after_mm": (r306.get("physical_verification") or {}).get("opening_after_mm")}
                except Exception as exc306:
                    auto_ap306 = {"target_opening_mm": round(tgt306, 2), "ok": False, "error": str(exc306)[:200]}
                aperture = _v24_aperture_check(obj)
            if not aperture["robust_aperture_ok"]:
                return _v24_precondition_failure(
                    name, args,
                    f"prepare gripper aperture before grasp planning: {aperture}",
                    contract, state_before,
                    physical_verification={"gripper_aperture": aperture},
                    recommended_next_actions=["set_gripper_opening", "observe_scene", "plan_grasp"],
                )

            if v311_mode() == "camera":
                # V3.11: target pose from perception; untrusted -> clear the view (q0) and look again once.
                try:
                    obj_xyz, obj_q = _v311_obj_pose(obj, require_trusted=True)
                except Exception as exc311:
                    skill_core.emit(f"[V3.11 RECAPTURE] {obj} | {exc311}")
                    try:
                        _snap311, _n311 = _v307_posture_numbers()
                        if v326_wrist_selected():
                            pass        # V3.28: the wrist recapture moves to its own look pose (no q0 detour)
                        elif not bool(hold_active_shared.value) and _n311["max_abs_joint_deg"] > 2.0:
                            _v307_go_q0("PERCEPTION RECAPTURE")
                        else:
                            time.sleep(0.3)
                    except Exception as exc3121q:
                        # V3.12.1: report the real cause (could not reach a clear view), not a perception failure.
                        return _v24_precondition_failure(
                            name, args, f"could not move to a clear camera view (q0) to re-look at {obj}: {exc3121q}",
                            contract, state_before,
                            physical_verification={"pose_source": v335_pose_source_label(),
                                                   "reason": "RECAPTURE_CLEAR_VIEW_FAILED"},
                            recommended_next_actions=["observe_scene", "plan_grasp_another_egg", "stop_and_report_if_repeated"],
                        )
                    try:
                        _v311_capture(f"plan_grasp_recapture:{obj}", [obj])
                        obj_xyz, obj_q = _v311_obj_pose(obj, require_trusted=True)
                    except Exception as exc311b:
                        return _v24_precondition_failure(
                            name, args, f"V3.11 camera could not localise {obj} reliably: {exc311b}",
                            contract, state_before,
                            physical_verification={"pose_source": v335_pose_source_label(),
                                                   "reason": "PERCEPTION_TARGET_UNTRUSTED",
                                                   "perception": _v311_perception_tag(_v311_obj_entry(obj))},
                            recommended_next_actions=["observe_scene", "plan_grasp_another_egg", "stop_and_report_if_repeated"],
                        )
            else:
                obj_xyz, obj_q = skill_core.object_pose(obj)
            if (v311_mode() == "camera" and _v339_rgb_only()
                    and os.environ.get("NERO_V339_GRASP_Z_GUARD", "1") != "0"):
                o339g = _v311_obj_entry(obj) or {}
                why339g = _v339_grasp_issue(o339g, V311_STATE.get("last"))
                if why339g and ("from table + half width" in why339g):
                    # V3.39: never plan a grasp at an implausible height (V3.38 log: egg z 0.048 -> grasp 27 mm too
                    # high -> the fingers closed on the neighbour egg -> BYSTANDER_EGG_DISTURBED)
                    return _v24_precondition_failure(
                        name, args, f"V3.39 {obj}: {why339g} - not planning a grasp at this height",
                        contract, state_before,
                        physical_verification={"pose_source": v335_pose_source_label(),
                                               "reason": "GRASP_HEIGHT_IMPLAUSIBLE",
                                               "perception": _v311_perception_tag(o339g)},
                        recommended_next_actions=["observe_scene", "plan_grasp", "plan_grasp_another_egg"],
                    )
            V271_BYSTANDER_BASELINE[obj] = {}
            for _other in OBJECTS:
                if str(_other) == obj or not str(_other).startswith('egg'):
                    continue
                try:
                    _p_other, _q_other = skill_core.object_pose(_other)
                    V271_BYSTANDER_BASELINE[obj][str(_other)] = [float(v) for v in _p_other]
                except Exception:
                    pass
            opening = float(gripper_actual_mm.value)
            tried = V27_GRASP_RUNTIME.setdefault("tried", {}).setdefault(obj, set())
            # V3.04: neighbour eggs near the target (for finger-pad ranking).
            neighbors304 = []
            for _other in OBJECTS:
                if str(_other) == obj or not str(_other).startswith("egg"):
                    continue
                try:
                    if v311_mode() == "camera":
                        _po, _qo = _v311_obj_pose(_other, require_trusted=False)   # last perceived pose
                    else:
                        _po, _qo = skill_core.object_pose(_other)
                    if math.hypot(float(_po[0]) - float(obj_xyz[0]), float(_po[1]) - float(obj_xyz[1])) > 0.14:
                        continue
                    if abs(float(_po[2]) - float(obj_xyz[2])) > 0.03:
                        continue
                    _ax = _v27_egg_major_axis_from_quat(_qo)
                    _sz = (V24_OBJECT_GEOMETRY.get(str(_other), {}) or {}).get("approx_size_mm", [44.0, 42.0, 62.0])
                    neighbors304.append({
                        "name": str(_other), "xy": [float(_po[0]), float(_po[1])],
                        "yaw_rad": float(_ax.get("yaw_rad", 0.0)),
                        "semi_major_m": float(_sz[2]) / 2000.0, "semi_minor_m": float(_sz[0]) / 2000.0,
                    })
                except Exception:
                    pass
            axis_info, candidates = _v27_make_grasp_candidates(obj, obj_xyz, obj_q, opening, tried, neighbors=neighbors304)

            # ============================================================
            # V3.04 FAST + DESCENT-AWARE GRASP PREFLIGHT
            # ------------------------------------------------------------
            # V3.03: up to 8 full cuRobo+MoveIt plans (~1 s each) per call, only
            # one used; and the top candidate's straight DESCENT was never
            # checked (egg_1 major_aligned__inward6 hovered fine, then
            # "CARTESIAN APPROACH BLOCKED 87.5%", ~15 s wasted twice).
            # Now: every candidate gets a cheap chained-BioIK check of the
            # pregrasp AND the vertical descent to the grasp pose; only the
            # first NERO_V304_FULL_PREFLIGHT (default 2) survivors get a full
            # plan, whose trajectory is cached for move_to_pregrasp.
            # ============================================================
            full_need304 = max(1, int(os.environ.get("NERO_V304_FULL_PREFLIGHT", "2")))
            live304 = skill_core._current_joint_state_copy()
            try:
                qstart304 = [float(v) for v in skill_core._arm_positions_from_joint_state(live304)]
            except Exception:
                qstart304 = None
            # V3.05 IK SCREEN: V3.04 used 2 seeds x 0.08 s and rejected 14/14 egg_2
            # candidates as "no IK at pregrasp" although cuRobo planned the same
            # goal at once (and egg_1 rejects drifted 4->5->10 between calls).
            # Now 3 seeds x 0.20 s; sibling offsets of the same yaw reuse the
            # first solution as seed (or one quick try if that yaw failed).
            n_seed305 = max(1, int(os.environ.get("NERO_V305_IK_SCREEN_SEEDS", "3")))
            t_pre305 = float(os.environ.get("NERO_V305_IK_SCREEN_TIMEOUT_S", "0.20"))
            t_desc305 = float(os.environ.get("NERO_V305_IK_DESCENT_TIMEOUT_S", "0.15"))
            seeds304 = [live304]
            try:
                seeds304 += [sv for tg, sv in skill_core._v70_natural_seed_variants(live304, qstart304, reorientation=False)
                             if tg != "current"][:max(0, n_seed305 - 1)]
            except Exception:
                pass
            yaw_memo305 = {}

            def _v304_descent_ok(c304):
                jpre = None
                ykey = tuple(round(float(v), 4) for v in c304["quat_xyzw"])
                if ykey in yaw_memo305:
                    prior = yaw_memo305[ykey]
                    seeds = ([skill_core._seed_with_arm_positions(live304, prior)] if prior is not None else []) + seeds304[:1]
                    if prior is None:
                        seeds = seeds304[:1]
                else:
                    seeds = list(seeds304)
                for sd in seeds:
                    try:
                        jpre, _e = skill_core._compute_ik(c304["pregrasp_xyz_m"], c304["quat_xyzw"], timeout_s=t_pre305,
                                                          avoid_collisions=True, seed_joint_state=sd)
                    except Exception:
                        jpre = None
                    if jpre is not None:
                        break
                if ykey not in yaw_memo305 or (yaw_memo305[ykey] is None and jpre is not None):
                    yaw_memo305[ykey] = [float(v) for v in jpre] if jpre is not None else None
                if jpre is None:
                    return False, "no IK at pregrasp"
                prev = [float(v) for v in jpre]
                gz = float(c304["grasp_xyz_m"][2])
                for dz in (0.08, 0.04, 0.02, 0.01, 0.0):
                    xyz = [float(c304["grasp_xyz_m"][0]), float(c304["grasp_xyz_m"][1]), gz + dz]
                    try:
                        j, _e = skill_core._compute_ik(xyz, c304["quat_xyzw"], timeout_s=t_desc305, avoid_collisions=True,
                                                       seed_joint_state=skill_core._seed_with_arm_positions(live304, prev))
                    except Exception as exc:
                        return False, f"IK exception {dz*1000:.0f} mm above grasp: {exc}"
                    if j is None:
                        return False, f"descent blocked {dz*1000:.0f} mm above grasp (no collision-free IK)"
                    step = max(abs(float(j[i]) - prev[i]) for i in range(7))
                    if step > math.radians(15.0):
                        return False, f"descent branch jump {math.degrees(step):.0f} deg {dz*1000:.0f} mm above grasp"
                    prev = [float(v) for v in j]
                return True, "ok"

            preflighted = []
            full_ok304 = 0
            full_tries304 = 0
            ik_rejects304 = 0
            desc_rejects3121 = 0
            for cand in candidates:
                c = dict(cand)
                c["moveit_pregrasp_ok"] = False
                c["moveit_pregrasp_error"] = None
                if c["already_tried"]:
                    preflighted.append(c)
                    continue
                if not c["geometry_feasible"]:
                    c["moveit_pregrasp_error"] = "BASKET_RIM_PROJECTED_CLEARANCE_INSUFFICIENT"
                    preflighted.append(c)
                    continue
                ok304, why304 = _v304_descent_ok(c)
                c["v304_ik_descent_ok"] = bool(ok304)
                if not ok304:
                    ik_rejects304 += 1
                    c["moveit_pregrasp_error"] = "V304_IK_DESCENT_REJECT: " + why304
                    preflighted.append(c)
                    continue
                if full_ok304 >= full_need304 or full_tries304 >= full_need304 + 2 + min(2, desc_rejects3121):
                    c["moveit_pregrasp_error"] = "V304_IK_OK_NOT_YET_PLANNED (call plan_grasp again if the planned candidates fail)"
                    preflighted.append(c)
                    continue
                full_tries304 += 1
                try:
                    # Plan only; do NOT execute.
                    traj304 = skill_core._plan_pose(
                        c["pregrasp_xyz_m"], c["quat_xyzw"],
                        velocity_scale=0.18, accel_scale=0.18,
                        orientation_tolerance=0.12,
                        position_tolerance=0.010,
                        planning_attempts=3,
                        planning_time=2.5,
                    )
                    # V3.12.1 DESCENT PRECHECK from the planned pregrasp joint state.
                    try:
                        okd3121, whyd3121 = _v3121_descent_check_from_traj(traj304, c)
                    except Exception as exc3121d:
                        okd3121, whyd3121 = True, f"precheck error (not used): {exc3121d}"
                    c["v3121_descent_check"] = whyd3121
                    if not okd3121:
                        desc_rejects3121 += 1
                        c["moveit_pregrasp_error"] = "V3121_DESCENT_CARTESIAN_REJECT: " + whyd3121
                        skill_core.emit(f"[V3.12.1 DESCENT PRECHECK] {obj} {c['candidate_id']} rejected: {whyd3121}")
                        preflighted.append(c)
                        continue
                    c["moveit_pregrasp_ok"] = True
                    full_ok304 += 1
                    if qstart304 is not None:
                        V304_PREGRASP_CACHE[(obj, c["candidate_id"])] = {
                            "traj": traj304, "q_start": list(qstart304), "t": time.time(),
                        }
                except Exception as exc:
                    c["moveit_pregrasp_error"] = str(exc)
                preflighted.append(c)
            # Safety net: if the cheap screen rejected everything, fall back to
            # the V3.03 behaviour (full plans) for the best few candidates.
            if full_ok304 == 0 and ik_rejects304 > 0:
                skill_core.emit(f"[V3.04 GRASP PREFLIGHT] IK screen rejected all {ik_rejects304}; falling back to full planning")
                n_fb304 = 0
                for c in preflighted:
                    if c.get("already_tried") or not c.get("geometry_feasible") or n_fb304 >= 4:
                        continue
                    # V3.05: only candidates that failed the PREGRASP IK itself (possible
                    # BioIK false negative) are eligible.  A blocked/jumping DESCENT is a
                    # real geometric rejection and must stay rejected.
                    if str(c.get("moveit_pregrasp_error") or "") != "V304_IK_DESCENT_REJECT: no IK at pregrasp":
                        continue
                    n_fb304 += 1
                    full_tries304 += 1
                    try:
                        traj304 = skill_core._plan_pose(
                            c["pregrasp_xyz_m"], c["quat_xyzw"],
                            velocity_scale=0.18, accel_scale=0.18, orientation_tolerance=0.12,
                            position_tolerance=0.010, planning_attempts=3, planning_time=2.5,
                        )
                        try:
                            okd3121, whyd3121 = _v3121_descent_check_from_traj(traj304, c)
                        except Exception as exc3121d:
                            okd3121, whyd3121 = True, f"precheck error (not used): {exc3121d}"
                        c["v3121_descent_check"] = whyd3121
                        if not okd3121:
                            desc_rejects3121 += 1
                            c["moveit_pregrasp_error"] = "V3121_DESCENT_CARTESIAN_REJECT: " + whyd3121
                            skill_core.emit(f"[V3.12.1 DESCENT PRECHECK] {obj} {c['candidate_id']} rejected: {whyd3121}")
                            continue
                        c["moveit_pregrasp_ok"] = True
                        c["moveit_pregrasp_error"] = None
                        c["v304_ik_descent_ok"] = None          # unknown: pregrasp IK screen missed it
                        c["v305_screen_fallback"] = True        # descent not pre-verified
                        full_ok304 += 1
                        if qstart304 is not None:
                            V304_PREGRASP_CACHE[(obj, c["candidate_id"])] = {
                                "traj": traj304, "q_start": list(qstart304), "t": time.time(),
                            }
                        break
                    except Exception as exc:
                        c["moveit_pregrasp_error"] = str(exc)
            skill_core.emit(
                f"[V3.05 GRASP PREFLIGHT] {obj} | candidates={len(preflighted)} | ik/descent rejects={ik_rejects304} | "
                f"full plans={full_tries304} ok={full_ok304} | descent rejects={desc_rejects3121} | yaws screened={len(yaw_memo305)} | "
                f"neighbours={[n['name'] for n in neighbors304]}"
            )

            feasible = [
                c for c in preflighted
                if (not c["already_tried"])
                and c["geometry_feasible"]
                and c["moveit_pregrasp_ok"]
            ]
            feasible.sort(key=lambda c: -float(c["score"]))
            recommended = feasible[0] if feasible else None

            plan = {
                "object_name": obj,
                "object_xyz_at_plan_m": [round(float(v), 6) for v in obj_xyz],
                "object_quat_at_plan_xyzw": [round(float(v), 8) for v in obj_q],
                "egg_major_axis": axis_info,
                "gripper_opening_mm": round(opening, 2),
                "basket_center_xy_m": list(V27_BASKET_CENTER_XY_M),
                "basket_safe_inner_radius_mm": round(V27_BASKET_SAFE_INNER_RADIUS_M * 1000.0, 1),
                "candidate_count": len(preflighted),
                "tried_candidate_ids": sorted(str(x) for x in tried),
                "candidates": preflighted,
                "recommended_candidate_id": recommended["candidate_id"] if recommended else None,
                "reorientation_required_if_none": True,
                "pose_source": (v335_pose_source_label() if v311_mode() == "camera" else "mujoco_ground_truth"),
            }
            # V3.11: the "has the egg moved since planning?" monitor compares simulator pose with
            # simulator pose (perceived quats have an arbitrary roll about the long axis).  Kept
            # out of the plan so it never reaches the model.
            try:
                V311_MONITOR_BASELINE[obj] = skill_core.object_pose(obj) if v311_mode() == "camera" else None
            except Exception:
                V311_MONITOR_BASELINE[obj] = None
            V27_GRASP_RUNTIME["plans"][obj] = plan
            if recommended is not None:
                V27_GRASP_RUNTIME["selected"][obj] = recommended["candidate_id"]
                _v24_set_state(active_object=obj)
                verification = {
                    "planning_only": True,
                    "physical_motion_expected": False,
                    "recommended_candidate_id": recommended["candidate_id"],
                    "recommended_gripper_yaw_deg": recommended["gripper_yaw_deg"],
                    "recommended_basket_clearance_mm": recommended["basket_projected_clearance_mm"],
                }
                if auto_ap306 is not None:
                    verification["v306_auto_aperture"] = auto_ap306
                skill_core.emit(
                    f"[V2-7 GRASP PLAN] {obj} | major_axis_yaw={axis_info['yaw_deg']:.1f}deg | "
                    f"orientation={axis_info['orientation_case']} | selected={recommended['candidate_id']} | "
                    f"gripper_yaw={recommended['gripper_yaw_deg']:.1f}deg | "
                    f"basket_clearance={recommended['basket_projected_clearance_mm']:.1f}mm"
                )
                return _v24_result(
                    name, True, contract, state_before, verification, [],
                    {"grasp_plan": plan, "elapsed_s": round(time.perf_counter()-t0, 3)},
                    next_stage=None,
                    recommended_next_actions=["move_to_pregrasp"],
                )

            untried25 = [c for c in preflighted if not c.get("already_tried")]
            geom25 = [c for c in untried25 if bool(c.get("geometry_feasible"))]
            best_clear25 = max(
                [float(c.get("basket_projected_clearance_mm", -999.0)) for c in untried25] or [-999.0]
            )
            edge_limited25 = bool(untried25 and not geom25 and best_clear25 < 0.0)
            used25 = int(V226_EDGE_REPOSITION_USED.get(obj, 0))
            edge_available25 = bool(
                edge_limited25 and used25 < V226_EDGE_REPOSITION_MAX_PER_OBJECT
            )
            verification = {
                "planning_only": True,
                "physical_motion_expected": False,
                "reason": "ALL_GRASP_CANDIDATES_EXHAUSTED_OR_INFEASIBLE",
                "reorientation_required": True,
                "reorientation_skill_available": False,
                "edge_limited_by_basket_rim": edge_limited25,
                "best_basket_clearance_mm": round(best_clear25, 2),
                "edge_reposition_available": edge_available25,
                "edge_reposition_used": used25,
                "edge_reposition_budget": V226_EDGE_REPOSITION_MAX_PER_OBJECT,
            }
            if edge_available25:
                skill_core.emit(
                    f"[V2-27 EDGE RECOVERY AVAILABLE] {obj} | all grasp footprints blocked by basket rim | "
                    f"best_clearance={best_clear25:.1f}mm | next=reposition_egg_inward"
                )
                warnings25 = [
                    "Direct grasp is edge-limited. Do not end the object yet: use reposition_egg_inward once, then observe_scene and plan_grasp again."
                ]
                next25 = ["reposition_egg_inward"]
            else:
                skill_core.emit(
                    f"[V2-7 GRASP INFEASIBLE] {obj} | no untried candidate passed basket-clearance + MoveIt preflight | "
                    f"edge_reposition_available={edge_available25}"
                )
                warnings25 = [
                    "No bounded grasp candidate is executable and V2-27 edge-reposition is unavailable/consumed. Do not loop the same plan."
                ]
                next25 = ["try_another_object_or_report"]
            return _v24_result(
                name, False, contract, state_before, verification,
                warnings25,
                {"grasp_plan": plan, "elapsed_s": round(time.perf_counter()-t0, 3)},
                next_stage=None,
                recommended_next_actions=next25,
            )

        if name == "move_to_pregrasp":
            obj = str(args.get("object_name", ""))
            cid = str(args.get("candidate_id", ""))
            if bool(hold_active_shared.value):
                return _v24_precondition_failure(
                    name, args, "hold_active=true; cannot move to a new pregrasp",
                    contract, state_before, recommended_next_actions=["resolve_or_release_held_object"],
                )
            plan = V27_GRASP_RUNTIME.get("plans", {}).get(obj)
            if not plan:
                return _v24_precondition_failure(
                    name, args, f"no current V2-7 grasp plan for {obj}",
                    contract, state_before, recommended_next_actions=["observe_scene", "plan_grasp"],
                )
            cand_map = {c["candidate_id"]: c for c in plan.get("candidates", [])}
            cand = cand_map.get(cid)
            if cand is None:
                return _v24_precondition_failure(
                    name, args, f"candidate_id={cid} not found in latest plan",
                    contract, state_before, recommended_next_actions=["plan_grasp"],
                )
            if cand.get("already_tried") or cid in V27_GRASP_RUNTIME.setdefault("tried", {}).setdefault(obj, set()):
                return _v24_precondition_failure(
                    name, args, f"candidate {cid} was already tried; do not repeat it",
                    contract, state_before, recommended_next_actions=["plan_grasp"],
                )
            if not (cand.get("geometry_feasible") and cand.get("moveit_pregrasp_ok")):
                return _v24_precondition_failure(
                    name, args, f"candidate {cid} is not preflight-feasible: {cand}",
                    contract, state_before, recommended_next_actions=["plan_grasp"],
                )

            xyz_now, q_now = skill_core.object_pose(obj)
            _base311 = V311_MONITOR_BASELINE.get(obj)
            _bxyz311 = _base311[0] if _base311 else plan["object_xyz_at_plan_m"]
            _bq311 = _base311[1] if _base311 else plan["object_quat_at_plan_xyzw"]
            pos_stale = _v24_xyz_distance(xyz_now, _bxyz311)
            ang_stale = _v27_quat_rotation_angle_deg(q_now, _bq311)
            if pos_stale > V27_OBJECT_STALE_POS_M or ang_stale > V27_OBJECT_STALE_ANGLE_DEG:
                V27_GRASP_RUNTIME["plans"].pop(obj, None)
                return _v24_precondition_failure(
                    name, args,
                    f"grasp plan stale: object moved {pos_stale*1000.0:.1f}mm / rotated {ang_stale:.1f}deg",
                    contract, state_before,
                    physical_verification={"object_motion_since_plan_mm": round(pos_stale*1000.0,2), "object_rotation_since_plan_deg": round(ang_stale,2)},
                    recommended_next_actions=["observe_scene", "plan_grasp"],
                )

            llm_require_arm_controller()
            obj_before = list(xyz_now)
            target = [float(v) for v in cand["pregrasp_xyz_m"]]
            q = [float(v) for v in cand["quat_xyzw"]]
            # V3.04: reuse the trajectory planned (and MoveIt-validated) by
            # plan_grasp if the arm has not moved since; otherwise plan again.
            cache304 = V304_PREGRASP_CACHE.pop((obj, cid), None)
            use_cache304 = False
            # V3.05: cache reuse is OFF by default (every V3.04 cached execution
            # failed); NERO_V305_PREGRASP_CACHE=1 re-enables it for A/B tests.
            if os.environ.get("NERO_V305_PREGRASP_CACHE", "0") != "1":
                if cache304 is not None:
                    skill_core.emit(f"[V3.05 PREGRASP CACHE OFF] {obj} {cid} -> fresh move_to_pose (V3.03 path)")
                cache304 = None
            if cache304 is not None and (time.time() - float(cache304.get("t", 0.0))) < 120.0:
                try:
                    qnow304 = [float(v) for v in skill_core._arm_positions_from_joint_state(skill_core._current_joint_state_copy())]
                    dq304 = max(abs(qnow304[i] - float(cache304["q_start"][i])) for i in range(7))
                    use_cache304 = dq304 <= math.radians(0.6)
                    skill_core.emit(
                        f"[V3.04 PREGRASP CACHE] {obj} {cid} | arm moved {math.degrees(dq304):.2f} deg since plan -> "
                        f"{'REUSE planned trajectory' if use_cache304 else 'replan'}"
                    )
                except Exception:
                    use_cache304 = False
            if use_cache304:
                _traj304 = cache304["traj"]
                motion_fn304 = lambda: skill_core._execute_trajectory(_traj304)
            else:
                motion_fn304 = lambda: skill_core.move_to_pose(
                    target, q,
                    velocity_scale=0.18,
                    accel_scale=0.18,
                    orientation_tolerance=0.12,
                    position_tolerance=0.010,
                )
            verification, cart_ok, warnings = _v24_run_motion(
                f"move_to_pregrasp({obj},{cid})",
                motion_fn304,
                target, 0.012, True,
            )
            obj_after, _ = skill_core.object_pose(obj)
            object_move = _v24_xyz_distance(obj_after, obj_before)
            verification.update({
                "candidate_id": cid,
                "candidate_gripper_yaw_deg": cand["gripper_yaw_deg"],
                "basket_projected_clearance_mm": cand["basket_projected_clearance_mm"],
                "object_motion_mm": round(object_move*1000.0, 2),
                "object_motion_expected": False,
            })
            ok = bool(cart_ok and object_move <= 0.005)
            if not ok:
                V27_GRASP_RUNTIME.setdefault("tried", {}).setdefault(obj, set()).add(cid)
                verification.setdefault("reason", "PREGRASP_CANDIDATE_EXECUTION_FAILED")
                return _v24_result(
                    name, False, contract, state_before, verification, warnings,
                    {"object_name": obj, "candidate_id": cid, "elapsed_s": round(time.perf_counter()-t0,3)},
                    next_stage=None,
                    recommended_next_actions=["observe_scene", "plan_grasp"],
                )

            V27_GRASP_RUNTIME["selected"][obj] = cid
            skill_core.active_grasp_quat = list(q)
            return _v24_result(
                name, True, contract, state_before, verification, warnings,
                {"object_name": obj, "candidate_id": cid, "elapsed_s": round(time.perf_counter()-t0,3)},
                V24_STAGE_HOVER, obj,
                recommended_next_actions=["approach_planned_grasp"],
            )

        if name == "approach_planned_grasp":
            obj = str(args.get("object_name", ""))
            cid = str(args.get("candidate_id", ""))
            if state_before.get("stage") != V24_STAGE_HOVER or state_before.get("active_object") != obj:
                return _v24_precondition_failure(
                    name, args, f"requires HOVER_REACHED for same object; current={state_before}",
                    contract, state_before, recommended_next_actions=["plan_grasp", "move_to_pregrasp"],
                )
            plan = V27_GRASP_RUNTIME.get("plans", {}).get(obj)
            if not plan:
                return _v24_precondition_failure(
                    name, args, f"no current V2-7 grasp plan for {obj}",
                    contract, state_before, recommended_next_actions=["observe_scene", "plan_grasp"],
                )
            cand_map = {c["candidate_id"]: c for c in plan.get("candidates", [])}
            cand = cand_map.get(cid)
            if cand is None or V27_GRASP_RUNTIME.get("selected", {}).get(obj) != cid:
                return _v24_precondition_failure(
                    name, args, f"candidate {cid} is not the currently reached/selected pregrasp",
                    contract, state_before, recommended_next_actions=["plan_grasp", "move_to_pregrasp"],
                )
            aperture = _v24_aperture_check(obj)
            if not aperture["robust_aperture_ok"]:
                return _v24_precondition_failure(
                    name, args, f"aperture changed outside robust range: {aperture}",
                    contract, state_before,
                    physical_verification={"gripper_aperture": aperture},
                    recommended_next_actions=["set_gripper_opening", "observe_scene", "plan_grasp"],
                )

            xyz_now, q_now = skill_core.object_pose(obj)
            _base311 = V311_MONITOR_BASELINE.get(obj)
            _bxyz311 = _base311[0] if _base311 else plan["object_xyz_at_plan_m"]
            _bq311 = _base311[1] if _base311 else plan["object_quat_at_plan_xyzw"]
            pos_stale = _v24_xyz_distance(xyz_now, _bxyz311)
            ang_stale = _v27_quat_rotation_angle_deg(q_now, _bq311)
            if pos_stale > V27_OBJECT_STALE_POS_M or ang_stale > V27_OBJECT_STALE_ANGLE_DEG:
                V27_GRASP_RUNTIME.setdefault("tried", {}).setdefault(obj, set()).add(cid)
                return _v24_precondition_failure(
                    name, args,
                    f"object pose became stale before descent: moved {pos_stale*1000.0:.1f}mm / rotated {ang_stale:.1f}deg",
                    contract, state_before,
                    physical_verification={"object_motion_since_plan_mm": round(pos_stale*1000.0,2), "object_rotation_since_plan_deg": round(ang_stale,2)},
                    recommended_next_actions=["observe_scene", "plan_grasp"],
                )

            llm_require_arm_controller()
            obj_before = list(xyz_now)
            opening_before = float(gripper_actual_mm.value)
            target = [float(v) for v in cand["grasp_xyz_m"]]
            q = [float(v) for v in cand["quat_xyzw"]]

            def _do_v27_descent():
                # V3.04: two segments; the last NERO_V304_FINAL_APPROACH_M (20 mm)
                # at NERO_V304_FINAL_APPROACH_SLOW (x3.5) so the pads slide past
                # the egg instead of knocking it.
                fin304 = float(os.environ.get("NERO_V304_FINAL_APPROACH_M", "0.020"))
                slow304 = float(os.environ.get("NERO_V304_FINAL_APPROACH_SLOW", "3.5"))
                tcp_now304, _qn304 = skill_core.current_tcp_pose()
                if fin304 > 0.0 and float(tcp_now304[2]) - float(target[2]) > fin304 + 0.010:
                    mid304 = [float(target[0]), float(target[1]), float(target[2]) + fin304]
                    traj = skill_core._cartesian_approach_plan(mid304, q)
                    _v218_slow_trajectory_in_place(traj, factor=2.0)
                    skill_core._execute_trajectory(traj)
                    traj2 = skill_core._cartesian_approach_plan(target, q)
                    _v218_slow_trajectory_in_place(traj2, factor=slow304)
                    skill_core.emit(f"[V3.04 APPROACH SPEED] x2.0 to {fin304*1000:.0f} mm above grasp, then x{slow304:.1f}")
                    skill_core._execute_trajectory(traj2)
                    return True
                traj = skill_core._cartesian_approach_plan(target, q)
                _v218_slow_trajectory_in_place(traj, factor=2.0)
                skill_core.emit("[V2-18 APPROACH SPEED] trajectory timing x2.0 -> approximately 50% of previous approach speed")
                skill_core._execute_trajectory(traj)
                return True

            verification, cart_ok, warnings = _v24_run_motion(
                f"approach_planned_grasp({obj},{cid})",
                _do_v27_descent,
                target, 0.010, True,
            )
            obj_after, _ = skill_core.object_pose(obj)
            opening_after = float(gripper_actual_mm.value)
            object_move = _v24_xyz_distance(obj_after, obj_before)
            object_move_mm = object_move * 1000.0
            verification.update({
                "candidate_id": cid,
                "candidate_gripper_yaw_deg": cand["gripper_yaw_deg"],
                "basket_projected_clearance_mm": cand["basket_projected_clearance_mm"],
                "object_motion_mm": round(object_move_mm, 2),
                "object_motion_required": False,
                "gripper_opening_before_approach_mm": round(opening_before, 2),
                "gripper_opening_after_approach_mm": round(opening_after, 2),
            })

            motion_error = str(verification.get("legacy_joint_verify_error", ""))
            if (not cart_ok) and "CARTESIAN APPROACH BLOCKED" in motion_error:
                V27_GRASP_RUNTIME.setdefault("tried", {}).setdefault(obj, set()).add(cid)
                verification["reason"] = "GRASP_CANDIDATE_APPROACH_BLOCKED"
                verification["candidate_marked_tried"] = True
                warnings.append("This candidate is exhausted. Re-observe and request a different constraint-aware candidate; do not repeat it.")
                skill_core.emit(f"[V2-7 CANDIDATE BLOCKED] {obj} {cid} | candidate exhausted; replan another yaw/offset")
                # V3.12.1: go back up to the pregrasp height (reverse of the path just executed) so the next
                # plan / camera look does not start with the gripper parked low inside the basket.
                verification["v3121_retreat_to_pregrasp"] = _v3121_retreat_to_z(
                    float(cand["pregrasp_xyz_m"][2]), f"blocked descent {obj} {cid}")
                return _v24_result(
                    name, False, contract, state_before, verification, warnings,
                    {"object_name": obj, "candidate_id": cid, "elapsed_s": round(time.perf_counter()-t0,3)},
                    next_stage=None,
                    recommended_next_actions=["observe_scene", "plan_grasp"],
                )

            safety_stop = bool(
                verification.get("reason") == "SAFETY_TRIP_CHANGED"
                or "PHYSICAL_SAFETY_STOP" in str(verification.get("legacy_joint_verify_error", ""))
                or "PHYSICAL_SAFETY_STOP" in str(verification.get("motion_exception", ""))
            )
            if safety_stop:
                V27_GRASP_RUNTIME.setdefault("tried", {}).setdefault(obj, set()).add(cid)
                used = int(V242_SAFETY_RETRY_USED.get(obj, 0))
                retry_allowed = bool(used < 1 and not bool(hold_active_shared.value))
                ctx = _v242_set_safety_context(
                    active=True,
                    tool="approach_planned_grasp",
                    object_name=obj,
                    trip_count=int(safety_trip_count.value),
                    safety_state_code=int(safety_state.value),
                    likely_cause=(
                        "SELECTED_GRASP_CANDIDATE_CONTACTED_BASKET_OR_OBJECT; "
                        "candidate is exhausted and a different yaw/offset must be planned"
                    ),
                    confidence="high",
                    evidence=[
                        f"candidate={cid}",
                        f"predicted_basket_clearance={cand['basket_projected_clearance_mm']}mm",
                        f"object_motion_before_clamp={object_move_mm:.1f}mm",
                    ],
                    suggested_strategy="retreat_and_retry_once" if retry_allowed else "stop",
                    suggested_retry_opening_mm=float(aperture["recommended_target_opening_mm"]),
                    retry_allowed=retry_allowed,
                    retry_used=used,
                    retry_limit=1,
                )
                verification["reason"] = "PREGRASP_SAFETY_STOP_REQUIRES_ALTERNATE_CANDIDATE"
                verification["safety_reflection_context"] = ctx
                _v24_set_state(V24_STAGE_SAFETY, active_object=obj, target_bin=None, last_tool=name, last_ok=False)
                return _v24_result(
                    name, False, contract, state_before, verification,
                    warnings + ["Do not retry the same candidate after recovery."],
                    {"object_name": obj, "candidate_id": cid, "elapsed_s": round(time.perf_counter()-t0,3)},
                    next_stage=None,
                    recommended_next_actions=["observe_scene", "recover_from_safety"] if retry_allowed else ["stop_and_explain_failure"],
                )

            # V2-18: do not reject a grasp solely because the target egg moved a
            # few millimeters.  Re-observe the exact scene after approach and decide
            # geometrically whether the TARGET is still between the fingers.
            if cart_ok:
                post_obs = _v218_post_approach_observe(obj, cand, plan)
                verification["post_approach_observation"] = post_obs
                skill_core.emit(
                    f"[V2-18 POST-APPROACH OBSERVE] {obj} | between={post_obs['target_between_fingers']} | "
                    f"lateral={post_obs['target_lateral_from_gripper_center_mm']:.1f}/{post_obs['lateral_limit_mm']:.1f}mm | "
                    f"longitudinal={post_obs['target_longitudinal_offset_mm']:.1f}/{post_obs['longitudinal_limit_mm']:.1f}mm | "
                    f"zerr={post_obs['vertical_relation_error_mm']:.1f}mm | object_motion={object_move_mm:.1f}mm"
                )
                if bool(post_obs.get("target_between_fingers")):
                    if object_move_mm > 4.0:
                        warnings.append(
                            f"Target moved {object_move_mm:.1f}mm during approach, but the fresh post-approach observation confirms it remains inside the finger capture zone; proceed to clamp and let contact/hold + lift-follow verification decide success."
                        )
                    _, q_actual = skill_core.current_tcp_pose()
                    skill_core.active_grasp_quat = [float(v) for v in q_actual]
                    verification["reason"] = "TARGET_STILL_BETWEEN_FINGERS_TRY_CLAMP"
                    return _v24_result(
                        name, True, contract, state_before, verification, warnings,
                        {"object_name": obj, "candidate_id": cid, "elapsed_s": round(time.perf_counter()-t0,3)},
                        V24_STAGE_GRASP_POSE, obj,
                        recommended_next_actions=["clamp_gripper"],
                    )

                # Target is no longer in the capture zone.  Do NOT try to reopen
                # the gripper while buried in the basket.  Retreat vertically first,
                # exhaust this candidate, then observe/replan a new XY/yaw candidate.
                V27_GRASP_RUNTIME.setdefault("tried", {}).setdefault(obj, set()).add(cid)
                retreat_ok = False
                retreat_error = None
                try:
                    if int(safety_state.value) == 0:
                        skill_core.emit("[V2-18 REPOSITION] target left finger capture zone -> vertical retreat +90mm before replanning")
                        skill_core.bio_lift(0.090)
                        retreat_ok = True
                except Exception as exc:
                    retreat_error = str(exc)
                    skill_core.emit(f"[V2-18 REPOSITION WARN] retreat failed: {exc}")
                verification.update({
                    "reason": "TARGET_LEFT_GRIPPER_CAPTURE_ZONE",
                    "candidate_marked_tried": True,
                    "retreat_attempted": True,
                    "retreat_ok": bool(retreat_ok),
                })
                if retreat_error:
                    verification["retreat_error"] = retreat_error
                _v24_set_state(V24_STAGE_OBSERVED, active_object=None, target_bin=None, last_tool=name, last_ok=False)
                warnings.append(
                    "Fresh post-approach observation shows the target is no longer between the fingers. Retreat first, then re-observe and plan a new grasp position/yaw; do not reopen in-place inside the egg pile."
                )
                return _v24_result(
                    name, False, contract, state_before, verification, warnings,
                    {"object_name": obj, "candidate_id": cid, "elapsed_s": round(time.perf_counter()-t0,3)},
                    next_stage=None,
                    recommended_next_actions=["observe_scene", "set_gripper_opening", "plan_grasp"],
                )

            if not cart_ok:
                V27_GRASP_RUNTIME.setdefault("tried", {}).setdefault(obj, set()).add(cid)
                verification.setdefault("reason", "GRASP_CANDIDATE_APPROACH_FAILED")
                return _v24_result(
                    name, False, contract, state_before, verification, warnings,
                    {"object_name": obj, "candidate_id": cid, "elapsed_s": round(time.perf_counter()-t0,3)},
                    next_stage=None,
                    recommended_next_actions=["observe_scene", "plan_grasp"],
                )

            skill_core.active_grasp_quat = list(q)
            return _v24_result(
                name, True, contract, state_before, verification, warnings,
                {"object_name": obj, "candidate_id": cid, "elapsed_s": round(time.perf_counter()-t0,3)},
                V24_STAGE_GRASP_POSE, obj,
                recommended_next_actions=["clamp_gripper"],
            )


        # ------------------------------------------------------------
        # NERO V2-9 Placement-Aware Egg Handling
        # ------------------------------------------------------------

        # Deterministic routing guard: the old receptacle descent is exactly the
        # path that collided with the egg holder in V2-7.  Do not rely on prompt
        # obedience alone; refuse those old motions for an active egg.
        if name in ("move_above_receptacle", "place_into_receptacle"):
            _legacy_obj = str(state_before.get("active_object") or "")
            if _legacy_obj.startswith("egg"):
                return _v24_precondition_failure(
                    name, args,
                    "V2-7.1 blocks legacy egg placement; use long-axis placement planning instead",
                    contract, state_before,
                    recommended_next_actions=["plan_placement"] if state_before.get("stage") == V24_STAGE_LIFTED else ["observe_scene", "plan_placement"],
                )
        if name == "open_gripper":
            _legacy_obj = str(state_before.get("active_object") or "")
            if _legacy_obj.startswith("egg") and state_before.get("stage") not in (V24_STAGE_RELEASED, V24_STAGE_VERIFIED):
                return _v24_precondition_failure(
                    name, args,
                    "V2-7.1 blocks direct egg release before guarded_insert_and_release reaches capture depth",
                    contract, state_before,
                    recommended_next_actions=["plan_placement"] if state_before.get("stage") == V24_STAGE_LIFTED else ["observe_scene"],
                )

        # If an egg was unexpectedly dropped after lift/placement, the old
        # V2-7 state could remain RECEPTACLE_HOVER_REACHED forever even though
        # hold_active=false.  On the next observation create a clean semantic
        # boundary so a fresh pickup can begin.
        if name == "observe_scene":
            _stage_now = str(state_before.get("stage") or "")
            if (
                not bool(hold_active_shared.value)
                and _stage_now in (
                    "LIFTED",
                    "RECEPTACLE_HOVER_REACHED",
                    "SLOT_PLACE_REACHED",
                    "PLACEMENT_ORIENTED",
                    "SLOT_ENTRY_ALIGNED",
                )
            ):
                skill_core.emit(
                    f"[V2-7.2 HOLD-LOSS STATE RECOVERY] stage={_stage_now} but hold_active=false -> fresh OBSERVE episode"
                )
                _v24_reset_skill_state("V2-7.1 unexpected hold loss", clear_adaptation=False)
                state_before = _v24_state_snapshot()

        # V2-16 BYSTANDER TOLERANCE — ONLY grasp-policy change versus V2-15.
        # Dense basket pickup naturally nudges neighboring eggs by a few millimeters.
        # Keep the original 3 mm threshold as a WARNING threshold, but only abort
        # before clamp when a non-target egg moves more than 8 mm.  For 3–8 mm,
        # continue into the existing FAST CLAMP path; real grasp success is still
        # determined by two-finger contact / hold_active, and the existing lift()
        # postcondition must prove that the TARGET egg rises with the TCP.
        if name == "clamp_gripper":
            _obj_guard = str(state_before.get("active_object") or "")
            if _obj_guard.startswith("egg") and state_before.get("stage") == V24_STAGE_GRASP_POSE:
                _moved = _v271_bystander_moves(skill_core, _obj_guard)  # >3 mm warning set
                _severe = [
                    dict(item) for item in _moved
                    if float(item.get("motion_mm", 0.0)) > (V216_BYSTANDER_ABORT_TOL_M * 1000.0)
                ]
                if _severe:
                    _cid = V27_GRASP_RUNTIME.get("selected", {}).get(_obj_guard)
                    if _cid:
                        V27_GRASP_RUNTIME.setdefault("tried", {}).setdefault(_obj_guard, set()).add(str(_cid))
                    _retreat_ok = False
                    try:
                        if int(safety_state.value) == 0:
                            skill_core.bio_lift(0.090)
                            _retreat_ok = True
                    except Exception as _exc:
                        skill_core.emit(f"[V2-16 BYSTANDER RETREAT WARN] {_exc}")
                    _v24_set_state(V24_STAGE_OBSERVED, active_object=None, target_bin=None, last_tool=name, last_ok=False)
                    _verification = {
                        "reason": "BYSTANDER_EGG_DISTURBED",
                        "active_object": _obj_guard,
                        "moved_bystanders": _moved,
                        "severe_bystanders": _severe,
                        "warning_threshold_mm": round(V271_BYSTANDER_MOVE_TOL_M * 1000.0, 1),
                        "abort_threshold_mm": round(V216_BYSTANDER_ABORT_TOL_M * 1000.0, 1),
                        "candidate_marked_tried": bool(_cid),
                        "retreat_attempted": True,
                        "retreat_ok": bool(_retreat_ok),
                    }
                    return _v24_result(
                        name, False, contract, state_before, _verification,
                        ["A non-target egg moved more than 8 mm during approach. Clamp is blocked; choose another grasp candidate."],
                        {"elapsed_s": round(time.perf_counter()-t0, 3)},
                        next_stage=None,
                        recommended_next_actions=["observe_scene", "plan_grasp"],
                    )
                if _moved:
                    _summary = ", ".join(
                        f"{item.get('object_name')}={float(item.get('motion_mm', 0.0)):.1f}mm"
                        for item in _moved
                    )
                    skill_core.emit(
                        f"[V2-16 BYSTANDER WARN] {_obj_guard} approach nudged {_summary}; "
                        f"all <= {V216_BYSTANDER_ABORT_TOL_M*1000.0:.1f}mm -> continue FAST CLAMP. "
                        "Target grasp must still pass contact/hold and lift-follow verification."
                    )

        if name == "plan_placement":
            obj = str(args.get("object_name", state_before.get("active_object") or ""))
            receptacle = str(args.get("receptacle_name", "egg_holder"))
            slot_name = str(args.get("slot_name", ""))
            if (_v329_no_prior() and bool(hold_active_shared.value) and state_before.get("stage") == V24_STAGE_LIFTED
                    and os.environ.get("NERO_V329_PREPLACE_LOOK", "1") != "0"):
                if v326_wrist_selected():
                    _v329_preplace_look(obj, slot_name)   # V3.29: confirm the slot with the camera before placing
                else:
                    try:                                   # V3.31 fixed camera: a fresh capture, no motion
                        _v311_capture(f"plan_placement:{obj}/holder", [obj])
                    except Exception as exc331p:
                        skill_core.emit(f"[V3.31 PRE-PLACE VIEW WARN] {obj} | {str(exc331p)[:200]}")
            slot = _v271_slot(slot_name)
            if state_before.get("stage") != V24_STAGE_LIFTED or state_before.get("active_object") != obj:
                return _v24_precondition_failure(
                    name, args, f"requires LIFTED with same active object; current={state_before}",
                    contract, state_before, recommended_next_actions=["observe_scene"],
                )
            if not bool(hold_active_shared.value):
                return _v24_precondition_failure(
                    name, args, "hold_active=false; cannot plan placement for an unheld egg",
                    contract, state_before, recommended_next_actions=["observe_scene"],
                )
            if receptacle != "egg_holder" or slot is None:
                return _v24_precondition_failure(
                    name, args, f"invalid egg-holder destination: receptacle={receptacle} slot={slot_name}",
                    contract, state_before, recommended_next_actions=["observe_scene"],
                )
            try:
                _scene = llm_scene_snapshot()
                _slot_scene = _scene.get("receptacles", {}).get("egg_holder", {}).get("slots", {}).get(slot_name, {})
                if bool(_slot_scene.get("occupied")):
                    return _v24_precondition_failure(
                        name, args, f"{slot_name} is already occupied by {_slot_scene.get('occupied_by')}",
                        contract, state_before, recommended_next_actions=["observe_scene", "choose_empty_slot"],
                    )
            except Exception:
                pass

            obj_xyz, obj_q = skill_core.object_pose(obj)
            tcp_xyz, tcp_q = skill_core.current_tcp_pose()
            orient = _v271_target_tcp_quat_for_slot(obj_q, tcp_q)
            geom = dict(V24_OBJECT_GEOMETRY.get(obj, {}))
            half_len = _v271_egg_half_length_m(geom)
            release_obj_z = max(
                float(slot[2]) + 0.010,
                V271_HOLDER_RIM_TOP_Z_M + half_len - V271_CAPTURE_DEPTH_M,
            )
            entry_obj_z = release_obj_z + V271_ENTRY_ABOVE_RELEASE_M
            safe_z = max(
                V271_SAFE_ORIENTATION_MIN_Z_M,
                float(tcp_xyz[2]) + V271_SAFE_ORIENTATION_EXTRA_Z_M,
            )
            safe_z = min(0.340, safe_z)
            safe_xyz = [V271_SAFE_ORIENTATION_XY_M[0], V271_SAFE_ORIENTATION_XY_M[1], safe_z]

            plan = {
                "object_name": obj,
                "receptacle_name": receptacle,
                "slot_name": slot_name,
                "slot_center_xyz_m": [float(v) for v in slot],
                "slot_entry_axis": [float(v) for v in V271_SLOT_ENTRY_AXIS],
                "object_xyz_at_plan_m": [round(float(v), 6) for v in obj_xyz],
                "object_quat_at_plan_xyzw": [round(float(v), 8) for v in obj_q],
                "tcp_xyz_at_plan_m": [round(float(v), 6) for v in tcp_xyz],
                "egg_major_axis_world": [round(float(v), 6) for v in orient["current_axis"]],
                "axis_error_before_deg": round(float(orient["axis_error_before_deg"]), 2),
                "target_axis_world": [round(float(v), 6) for v in orient["target_axis"]],
                "safe_orientation_xyz_m": [round(float(v), 6) for v in safe_xyz],
                "entry_object_center_z_m": round(float(entry_obj_z), 6),
                "pre_release_object_center_z_m": round(float(release_obj_z), 6),
                "egg_half_length_mm": round(half_len * 1000.0, 2),
                "capture_depth_mm": round(V271_CAPTURE_DEPTH_M * 1000.0, 1),
                "policy": "short-axis grasp -> safe-height long-axis reorientation -> precise slot entry -> slow axial insertion -> staged release",
            }
            V271_PLACEMENT_RUNTIME["plans"][obj] = plan
            V271_PLACEMENT_RUNTIME["oriented"].pop(obj, None)
            V271_PLACEMENT_RUNTIME["slot_entry"].pop(obj, None)
            V271_PLACEMENT_RUNTIME["released"].pop(obj, None)
            skill_core.emit(
                f"[V2-7.2 PLACEMENT PLAN] {obj} -> {slot_name} | long-axis error={plan['axis_error_before_deg']:.1f}deg | "
                f"safe_orient_z={safe_z*1000.0:.0f}mm | entry_center_z={entry_obj_z*1000.0:.0f}mm | "
                f"pre_release_center_z={release_obj_z*1000.0:.0f}mm"
            )
            # V3.02 SLOT-ABOVE REACHABILITY PRECHECK (planning only, no motion)
            # A vertical held egg may take any yaw about its axis.  Check with
            # BioIK whether ANY of 12 yaws reaches the slot-above goal from the
            # current grasp.  Only if the requested slot has none AND another empty
            # slot has >=2, refuse and recommend that slot (V3.01 log: slot_1 at
            # x=-0.51 was unreachable while the arm held the egg off-centre).
            reach302 = None
            # V3.04: the precheck screens a minimal-rotation vertical pose from the
            # CURRENT grasp.  Right after lift the egg is ~90 deg off and that pose
            # is meaningless (V3.03 log: 0/12 for all 8 slots, 15.5 s, yet slot_1
            # succeeded).  Only run it when the egg is already near vertical
            # (typically after a slot change).
            _pre_tol304 = float(os.environ.get("NERO_V304_PRECHECK_MAX_AXIS_DEG", "30"))
            if float(plan.get("axis_error_before_deg", 90.0)) > _pre_tol304:
                skill_core.emit(
                    f"[V3.04 SLOT REACH SKIPPED] egg axis {float(plan.get('axis_error_before_deg', 90.0)):.0f} deg from vertical "
                    f"> {_pre_tol304:.0f} deg; orient_for_slot will choose a reachable yaw + descent corridor"
                )
                reach302 = {"requested_slot": slot_name, "skipped": "egg_not_yet_vertical"}
            elif os.environ.get("NERO_V302_SLOT_PRECHECK", "1") != "0":
                try:
                    def _v302_rot(q302, v302):
                        qv302 = _v27_quat_multiply(_v27_quat_multiply(q302, [float(v302[0]), float(v302[1]), float(v302[2]), 0.0]),
                                                   _v271_quat_conjugate(q302))
                        return qv302[:3]
                    axis302 = [float(v) for v in orient["current_axis"]]
                    r_world302 = [float(obj_xyz[i]) - float(tcp_xyz[i]) for i in range(3)]
                    r_local302 = _v302_rot(_v271_quat_conjugate(tcp_q), r_world302)
                    targ302 = [0.0, 0.0, 1.0 if axis302[2] >= 0.0 else -1.0]
                    qv302 = _v27_quat_normalize(_v27_quat_multiply(_v271_quat_from_two_vectors(axis302, targ302), tcp_q))
                    # V3.03: same hover height as orient_for_slot, and a yaw only
                    # counts if the SAME branch also reaches the slot-entry depth.
                    z302 = max(float(entry_obj_z) + max(0.06, float(os.environ.get("NERO_V303_HOVER_ABOVE_ENTRY_M", "0.10"))), 0.20)
                    live302 = skill_core._current_joint_state_copy()
                    seeds302 = [live302]
                    try:
                        _arm302 = [float(v) for v in skill_core._arm_positions_from_joint_state(live302)]
                        seeds302 += [sv for tg, sv in skill_core._v70_natural_seed_variants(live302, _arm302, reorientation=True)
                                     if tg != "current"][:2]
                    except Exception:
                        pass

                    def _v302_count(slot_xyz302):
                        n302 = 0
                        for yaw302 in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
                            h302 = 0.5 * math.radians(yaw302)
                            qt302 = _v27_quat_normalize(_v27_quat_multiply([0.0, 0.0, math.sin(h302), math.cos(h302)], qv302))
                            rw302 = _v302_rot(qt302, r_local302)
                            tcpt302 = [float(slot_xyz302[0]) - rw302[0], float(slot_xyz302[1]) - rw302[1], z302 - rw302[2]]
                            tcpe303 = [float(slot_xyz302[0]) - rw302[0], float(slot_xyz302[1]) - rw302[1], float(entry_obj_z) - rw302[2]]
                            j302 = None
                            for sd302 in seeds302:
                                try:
                                    j302, _e302 = skill_core._compute_ik(tcpt302, qt302, timeout_s=0.05,
                                                                        avoid_collisions=True, seed_joint_state=sd302)
                                except Exception:
                                    j302 = None
                                if j302 is not None:
                                    break
                            if j302 is None:
                                continue
                            try:
                                se303 = skill_core._seed_with_arm_positions(live302, list(j302))
                                je303, _ee303 = skill_core._compute_ik(tcpe303, qt302, timeout_s=0.05,
                                                                      avoid_collisions=True, seed_joint_state=se303)
                            except Exception:
                                je303 = None
                            if je303 is not None and max(abs(float(je303[i]) - float(j302[i])) for i in range(7)) <= math.radians(45.0):
                                n302 += 1
                        return n302

                    n_req302 = _v302_count(slot)
                    reach302 = {"requested_slot": slot_name, "requested_feasible_yaws": n_req302,
                                "object_center_z_m": round(z302, 3)}
                    if n_req302 == 0:
                        alts302 = []
                        try:
                            _slots302 = llm_scene_snapshot().get("receptacles", {}).get("egg_holder", {}).get("slots", {})
                        except Exception:
                            _slots302 = {}
                        for sn302, sd302 in sorted(_slots302.items()):
                            if sn302 == slot_name or bool(sd302.get("occupied")):
                                continue
                            sxyz302 = _v271_slot(sn302)
                            if sxyz302 is None:
                                continue
                            na302 = _v302_count(sxyz302)
                            alts302.append((sn302, na302))
                            if na302 >= 2:
                                break
                        reach302["alternatives_checked"] = [{"slot": a302, "feasible_yaws": b302} for a302, b302 in alts302]
                        good302 = [a302 for a302, b302 in alts302 if b302 >= 2]
                        if good302:
                            skill_core.emit(
                                f"[V3.02 SLOT REACH] {obj} -> {slot_name}: 0/12 yaws reachable | "
                                f"recommend {good302[0]} | plan refused"
                            )
                            V271_PLACEMENT_RUNTIME["plans"].pop(obj, None)
                            return _v24_result(
                                name, False, contract, state_before,
                                {"planning_only": True, "physical_motion_expected": False,
                                 "reason": "SLOT_ABOVE_UNREACHABLE_FROM_CURRENT_GRASP",
                                 "v302_slot_reachability": reach302,
                                 "recommended_slot": good302[0]},
                                [f"{slot_name} is not reachable with the current grasp; {good302[0]} is reachable. "
                                 f"Call plan_placement with slot_name={good302[0]}."],
                                {"elapsed_s": round(time.perf_counter()-t0, 3)},
                                next_stage=None,
                                recommended_next_actions=[f"plan_placement(slot_name={good302[0]})"],
                            )
                    skill_core.emit(
                        f"[V3.03 SLOT REACH] {obj} -> {slot_name}: {n_req302}/12 yaws reach hover+entry (BioIK screen)"
                    )
                except Exception as exc302:
                    skill_core.emit(f"[V3.02 SLOT REACH WARN] precheck skipped: {exc302}")
            verification = {
                "planning_only": True,
                "physical_motion_expected": False,
                "v302_slot_reachability": reach302,
                "axis_error_before_deg": plan["axis_error_before_deg"],
                "safe_orientation_xyz_m": plan["safe_orientation_xyz_m"],
                "entry_object_center_z_m": plan["entry_object_center_z_m"],
                "pre_release_object_center_z_m": plan["pre_release_object_center_z_m"],
                "wrist_dominant_intent": True,
                "hard_coded_joint6_joint7": False,
            }
            return _v24_result(
                name, True, contract, state_before, verification, [],
                {"placement_plan": plan, "elapsed_s": round(time.perf_counter()-t0, 3)},
                next_stage=None,
                recommended_next_actions=["orient_for_slot"],
            )

        if name == "orient_for_slot":
            # V2-24 HIERARCHICAL HELD-OBJECT MANIPULATION
            # ------------------------------------------------------------
            # Goal: stop treating held-object reorientation/transport as a
            # prescribed Cartesian curve.  Only the TASK endpoint matters:
            # egg long-axis vertical + safe held object + slot-above arrival.
            #
            # Strategy hierarchy:
            #   1) sparse wrist-dominant joint search (1 joint, then 2 joints),
            #      scored with cuRobo batch FK BEFORE any physical motion;
            #   2) unconstrained whole-arm cuRobo pose motion to a vertical egg;
            #   3) slow validated joint-space return to q=0 while maintaining
            #      clamp/hold, then cuRobo from q=0 to vertical slot-above.
            #
            # In ALL strategies the intermediate TCP/object-center trajectory
            # is free.  MoveIt remains the authoritative collision gate.
            obj = str(args.get("object_name") or state_before.get("active_object") or "")
            slot_name = str(args.get("slot_name") or state_before.get("target_slot") or "")
            warnings24 = []
            verification24 = {
                "v224_strategy_hierarchy": [
                    "sparse_wrist_1_or_2_joint",
                    "free_whole_arm_curobo",
                    "slow_zero_then_curobo",
                ],
                "center_path_constrained": False,
            }
            safety0 = int(safety_trip_count.value)
            plan24 = V271_PLACEMENT_RUNTIME.get("plans", {}).get(obj)
            slot24 = _v271_slot(slot_name)
            if not obj or plan24 is None or slot24 is None:
                verification24["reason"] = "V224_MISSING_OBJECT_OR_PLACEMENT_PLAN"
                return _v24_result(
                    name, False, contract, state_before, verification24, warnings24,
                    {"elapsed_s": round(time.perf_counter()-t0, 3)},
                    next_stage=None, recommended_next_actions=["observe_scene"],
                )
            if not bool(hold_active_shared.value):
                verification24["reason"] = "V224_GRASP_NOT_ACTIVE"
                return _v24_result(
                    name, False, contract, state_before, verification24, warnings24,
                    {"elapsed_s": round(time.perf_counter()-t0, 3)},
                    next_stage=None, recommended_next_actions=["observe_scene"],
                )
            # V3.12.1 ORIENT PRE-RAISE: the strategies below assume a normal post-lift height.
            try:
                _pr3121 = _v3121_pre_raise_for_orient(obj)
            except Exception as _e3121:
                _pr3121 = {"ok": False, "error": str(_e3121)[:200]}
            if _pr3121:
                verification24["v3121_pre_raise"] = _pr3121

            # ---------------- local math / FK helpers -----------------
            def _v224_vnorm(v):
                n = math.sqrt(sum(float(x)*float(x) for x in v))
                if n < 1e-12:
                    return [0.0, 0.0, 0.0]
                return [float(x)/n for x in v]

            def _v224_qnorm(q):
                n = math.sqrt(sum(float(x)*float(x) for x in q))
                if n < 1e-12:
                    return [0.0, 0.0, 0.0, 1.0]
                return [float(x)/n for x in q]

            def _v224_qconj(q):
                q = _v224_qnorm(q)
                return [-q[0], -q[1], -q[2], q[3]]

            def _v224_qmul(a, b):
                ax, ay, az, aw = _v224_qnorm(a)
                bx, by, bz, bw = _v224_qnorm(b)
                return _v224_qnorm([
                    aw*bx + ax*bw + ay*bz - az*by,
                    aw*by - ax*bz + ay*bw + az*bx,
                    aw*bz + ax*by - ay*bx + az*bw,
                    aw*bw - ax*bx - ay*by - az*bz,
                ])

            def _v224_qrot(q, v):
                x, y, z, w = _v224_qnorm(q)
                vx, vy, vz = [float(t) for t in v]
                # quaternion-vector rotation, expanded form
                tx = 2.0 * (y*vz - z*vy)
                ty = 2.0 * (z*vx - x*vz)
                tz = 2.0 * (x*vy - y*vx)
                return [
                    vx + w*tx + (y*tz - z*ty),
                    vy + w*ty + (z*tx - x*tz),
                    vz + w*tz + (x*ty - y*tx),
                ]

            def _v224_q_from_two_vectors(a, b):
                a = _v224_vnorm(a); b = _v224_vnorm(b)
                dot = max(-1.0, min(1.0, sum(a[i]*b[i] for i in range(3))))
                if dot > 0.999999:
                    return [0.0, 0.0, 0.0, 1.0]
                if dot < -0.999999:
                    # Pick a stable axis orthogonal to a.
                    ref = [1.0, 0.0, 0.0] if abs(a[0]) < 0.8 else [0.0, 1.0, 0.0]
                    axis = _v224_vnorm([
                        a[1]*ref[2] - a[2]*ref[1],
                        a[2]*ref[0] - a[0]*ref[2],
                        a[0]*ref[1] - a[1]*ref[0],
                    ])
                    return [axis[0], axis[1], axis[2], 0.0]
                cross = [
                    a[1]*b[2] - a[2]*b[1],
                    a[2]*b[0] - a[0]*b[2],
                    a[0]*b[1] - a[1]*b[0],
                ]
                return _v224_qnorm([cross[0], cross[1], cross[2], 1.0 + dot])

            def _v224_axis_err(axis):
                a = _v224_vnorm(axis)
                return math.degrees(math.acos(max(-1.0, min(1.0, abs(a[2])))))

            def _v224_fk_batch(q_batch):
                if getattr(skill_core, "_gpu_client", None) is None:
                    raise RuntimeError("V2-24 batch FK requires the GPU client")
                out = skill_core._gpu_client.fk_batch(
                    joint_names=[f"joint{i}" for i in range(1, 8)],
                    q_batch=q_batch,
                    timeout_s=20.0,
                )
                if not bool(out.get("ok")):
                    raise RuntimeError("cuRobo FK failed: " + str(out.get("error")))
                poses = out.get("poses") or []
                if len(poses) != len(q_batch):
                    raise RuntimeError(
                        f"cuRobo FK count mismatch: {len(poses)} != {len(q_batch)}"
                    )
                return poses

            def _v224_interp_q(q0, q1, step_deg=4.0):
                md = max(abs(float(q1[i])-float(q0[i])) for i in range(7))
                n = max(1, int(math.ceil(md / math.radians(float(step_deg)))))
                return [
                    [float(q0[j]) + (float(q1[j])-float(q0[j]))*(float(k)/float(n)) for j in range(7)]
                    for k in range(n+1)
                ]

            def _v224_rigid_local(obj_xyz, tcp_xyz, tcp_q, axis_world):
                qc = _v224_qconj(tcp_q)
                r_world = [float(obj_xyz[i])-float(tcp_xyz[i]) for i in range(3)]
                return _v224_qrot(qc, r_world), _v224_qrot(qc, axis_world)

            def _v224_predict(pose, r_local, axis_local):
                p = [float(v) for v in pose["position"]]
                q = [float(v) for v in pose["quat_xyzw"]]
                rw = _v224_qrot(q, r_local)
                aw = _v224_qrot(q, axis_local)
                return [p[i]+rw[i] for i in range(3)], aw

            def _v224_validate_joint_path(q_path, r_local, axis_local, min_obj_z=0.075):
                for i24, q24 in enumerate(q_path):
                    ok24, why24 = skill_core._v71_state_is_valid(q24)
                    if not ok24:
                        return False, f"MoveIt rejected sparse/zero point {i24}/{len(q_path)-1}: {why24}", None
                poses24 = _v224_fk_batch(q_path)
                minz24 = 1e9
                for pose24 in poses24:
                    objp24, _ = _v224_predict(pose24, r_local, axis_local)
                    minz24 = min(minz24, float(objp24[2]))
                if minz24 < float(min_obj_z):
                    return False, f"predicted held-object Z {minz24*1000.0:.1f}mm below safety floor", poses24
                return True, "OK", poses24

            def _v224_traj_object_clear(traj24, r_local, axis_local, min_obj_z=0.075):
                pts24 = list(traj24.joint_trajectory.points)
                if not pts24:
                    return False, "trajectory has no points"
                qs24 = [[float(v) for v in p.positions] for p in pts24]
                try:
                    poses24 = _v224_fk_batch(qs24)
                except Exception as exc24:
                    return False, f"FK validation failed: {exc24}"
                minz24 = 1e9
                for pose24 in poses24:
                    objp24, _ = _v224_predict(pose24, r_local, axis_local)
                    minz24 = min(minz24, float(objp24[2]))
                if minz24 < float(min_obj_z):
                    return False, f"planned held-object Z dips to {minz24*1000.0:.1f}mm"
                return True, f"held-object minZ={minz24*1000.0:.1f}mm"

            def _v224_exec_joint_path(label24, q_path24, xyz_path24, slow_factor=1.0):
                traj24 = skill_core._make_incremental_joint_trajectory(q_path24, xyz_path24)
                if traj24 is None:
                    return False, {"reason": "NO_TRAJECTORY"}, []
                if float(slow_factor) > 1.0:
                    for pt24 in traj24.joint_trajectory.points:
                        tsec24 = float(pt24.time_from_start.sec) + float(pt24.time_from_start.nanosec)*1e-9
                        tsec24 *= float(slow_factor)
                        si24 = int(math.floor(tsec24))
                        pt24.time_from_start.sec = si24
                        pt24.time_from_start.nanosec = int(round((tsec24-si24)*1e9))
                target24 = xyz_path24[-1]
                try:
                    ver24, ok24, warn24 = _v24_run_motion(
                        label24,
                        lambda traj24=traj24: skill_core._execute_trajectory(traj24),
                        target24, 0.035, True,
                    )
                except Exception as exc301:
                    # V3.01: _v24_run_motion re-raises an execution timeout, which
                    # skipped the V2-51 safe-high handoff entirely (dead code).
                    # Return it as a structured failure so the handoff can run.
                    msg301 = str(exc301)
                    if "arm_controller execution timeout after" not in msg301:
                        raise
                    skill_core.emit(f"[V3.01 TIMEOUT -> HANDOFF CHECK] {label24} | {msg301}")
                    return False, {"reason": "EXEC_TIMEOUT", "motion_exception": msg301}, [msg301]
                return bool(ok24), ver24, warn24

            def _v224_refresh_rigid():
                objp24, objq24 = skill_core.object_pose(obj)
                tcpp24, tcpq24 = skill_core.current_tcp_pose()
                axis24 = _v27_egg_major_axis_from_quat(objq24)["world_axis"]
                rloc24, aloc24 = _v224_rigid_local(objp24, tcpp24, tcpq24, axis24)
                return (
                    [float(v) for v in objp24], [float(v) for v in objq24],
                    [float(v) for v in tcpp24], [float(v) for v in tcpq24],
                    [float(v) for v in axis24], rloc24, aloc24,
                )

            def _v224_target_tcp_for_object(obj_target24, tcp_q_target24, r_local24):
                rw24 = _v224_qrot(tcp_q_target24, r_local24)
                return [float(obj_target24[i])-float(rw24[i]) for i in range(3)]

            def _v224_plan_score(traj24, q_start24):
                pts24 = list(traj24.joint_trajectory.points)
                if not pts24:
                    return (1e9, 1e9, 1e9)
                prev24 = [float(v) for v in q_start24]
                total24 = 0.0; maxseg24 = 0.0
                for p24 in pts24:
                    q24 = [float(v) for v in p24.positions]
                    ds24 = [abs(q24[j]-prev24[j]) for j in range(7)]
                    total24 += sum(ds24)
                    maxseg24 = max(maxseg24, max(ds24))
                    prev24 = q24
                qend24 = [float(v) for v in pts24[-1].positions]
                margin24 = min(
                    min(qend24[j]-float(JOINT_LIMITS_RAD[j][0]), float(JOINT_LIMITS_RAD[j][1])-qend24[j])
                    for j in range(7)
                )
                return (total24, maxseg24, -margin24)

            def _v224_execute_pose_plan(label24, traj24, target_tcp24, r_local24, axis_local24):
                clear24, why24 = _v224_traj_object_clear(traj24, r_local24, axis_local24)
                if not clear24:
                    skill_core.emit(f"[V2-24 HELD-OBJECT PATH REJECT] {label24} | {why24}")
                    return False, {"reason": why24}, []
                ver24, ok24, warn24 = _v24_run_motion(
                    label24,
                    lambda traj24=traj24: skill_core._execute_trajectory(traj24),
                    target_tcp24, 0.025, True,
                )
                return bool(ok24), ver24, warn24

            # Fresh relation after lift.  No path is prescribed from here.
            obj0, objq0, tcp0, tcpq0, axis0, rlocal0, alocal0 = _v224_refresh_rigid()
            err0 = _v224_axis_err(axis0)
            verification24["axis_error_before_deg"] = round(float(err0), 3)
            skill_core.emit(
                f"[V2-24 ORIENT START] {obj}->{slot_name} | axis_error={err0:.2f}deg | "
                f"center_path=FREE | hierarchy=wrist-sparse -> whole-arm -> q0"
            )

            # ============================================================
            # STRATEGY 1: 1-2 JOINT WRIST-DOMINANT SEARCH
            # ============================================================
            wrist_ok24 = False
            # V3.03: after a slot change the egg is usually still vertical; do
            # not wiggle the wrist, go straight to the yaw-free slot transport.
            already303 = float(err0) <= 0.6 * float(V271_AXIS_TOL_DEG)
            if already303:
                wrist_ok24 = True
                skill_core.emit(f"[V3.03 ALREADY VERTICAL] axis_error={err0:.2f}deg -> skip reorientation")
            try:
                if already303:
                    raise RuntimeError("already vertical")
                live24 = skill_core._current_joint_state_copy()
                qcur24 = [float(v) for v in skill_core._arm_positions_from_joint_state(live24)]
                candidates24 = []
                seen24 = set()

                def _v224_add_sparse(active24, deltas24, rank24):
                    q24 = list(qcur24)
                    for idx24, dd24 in zip(active24, deltas24):
                        q24[idx24] += math.radians(float(dd24))
                    for j24, lim24 in enumerate(JOINT_LIMITS_RAD):
                        if q24[j24] < float(lim24[0])+math.radians(2.0) or q24[j24] > float(lim24[1])-math.radians(2.0):
                            return
                    key24 = tuple(round(v, 5) for v in q24)
                    if key24 in seen24:
                        return
                    seen24.add(key24)
                    candidates24.append((rank24, tuple(active24), tuple(float(v) for v in deltas24), q24))

                single_vals24 = [-80,-70,-60,-50,-40,-30,-20,-10,10,20,30,40,50,60,70,80]
                for d24 in single_vals24:
                    _v224_add_sparse((6,), (d24,), 0)       # J7 first
                    _v224_add_sparse((4,), (d24,), 1)       # then J5
                pair_vals24 = [-60,-45,-30,-15,15,30,45,60]
                for a24 in pair_vals24:
                    for b24 in pair_vals24:
                        _v224_add_sparse((4,6), (a24,b24), 2)  # J5+J7 preferred pair
                        _v224_add_sparse((5,6), (a24,b24), 3)  # J6+J7 secondary
                        _v224_add_sparse((4,5), (a24,b24), 4)  # J5+J6 secondary

                skill_core.emit(
                    f"[V2-24 WRIST SEARCH] offline FK candidates={len(candidates24)} | "
                    f"single J7/J5 first; 2-joint distal pairs second"
                )
                poses24 = _v224_fk_batch([c24[3] for c24 in candidates24]) if candidates24 else []
                scored24 = []
                for c24, pose24 in zip(candidates24, poses24):
                    rank24, active24, deltas24, q24 = c24
                    objp24, ax24 = _v224_predict(pose24, rlocal0, alocal0)
                    e24 = _v224_axis_err(ax24)
                    if e24 > V271_AXIS_TOL_DEG:
                        continue
                    if float(objp24[2]) < 0.090:
                        continue
                    totald24 = sum(abs(v) for v in deltas24)
                    maxd24 = max(abs(v) for v in deltas24)
                    # Prefer fewer/more-distal joints first once endpoint is acceptable.
                    scored24.append(((len(active24), rank24, e24, totald24, maxd24), c24, pose24, objp24))
                scored24.sort(key=lambda x: x[0])

                verification24["wrist_fk_candidates"] = len(candidates24)
                verification24["wrist_vertical_candidates"] = len(scored24)
                for score24, c24, pose24, objp24 in scored24[:16]:
                    rank24, active24, deltas24, qgoal24 = c24
                    qpath24 = _v224_interp_q(qcur24, qgoal24, step_deg=4.0)
                    valid24, why24, pathposes24 = _v224_validate_joint_path(
                        qpath24, rlocal0, alocal0, min_obj_z=0.075,
                    )
                    if not valid24:
                        continue
                    xyzpath24 = [[float(v) for v in pp24["position"]] for pp24 in pathposes24]
                    active_names24 = "+".join(f"J{i+1}" for i in active24)
                    skill_core.emit(
                        f"[V2-24 WRIST CANDIDATE ACCEPT] joints={active_names24} | "
                        f"delta_deg={[round(v,1) for v in deltas24]} | predicted_axis={score24[2]:.2f}deg | "
                        f"center_path=FREE"
                    )
                    ok24, ver24, warn24 = _v224_exec_joint_path(
                        f"v224_sparse_{active_names24}({obj},{slot_name})",
                        qpath24, xyzpath24, slow_factor=1.0,
                    )
                    warnings24.extend(warn24)
                    if not ok24 or not bool(hold_active_shared.value) or int(safety_trip_count.value) != safety0:
                        break
                    _o24, _oq24, _t24, _tq24, axact24, _rl24, _al24 = _v224_refresh_rigid()
                    erract24 = _v224_axis_err(axact24)
                    verification24["wrist_active_joints"] = active_names24
                    verification24["wrist_delta_deg"] = [round(float(v),2) for v in deltas24]
                    verification24["wrist_axis_error_after_deg"] = round(float(erract24),3)
                    if erract24 <= V271_AXIS_TOL_DEG:
                        wrist_ok24 = True
                        skill_core.emit(
                            f"[V2-24 WRIST-ONLY SUCCESS] joints={active_names24} | "
                            f"axis_error={erract24:.2f}deg | hold=true"
                        )
                    break
            except Exception as exc24:
                if not already303:
                    warnings24.append(f"V2-24 wrist strategy unavailable: {exc24}")
                    skill_core.emit(f"[V2-24 WRIST STRATEGY FAIL] {exc24}")

            # ============================================================
            # V3.02 YAW-FREE VERTICAL TARGETS + NON-SINGULAR SAFE-HIGH HOLD
            # ------------------------------------------------------------
            # Evidence (V3.01 logs 2026-10-21):
            #  * Strategy 2 and q0->slot planned ONE fixed TCP orientation per
            #    position (minimal rotation that makes the egg vertical).  An egg
            #    is a body of revolution, so any rotation about the vertical egg
            #    axis is equally valid.  Ignoring that freedom made goals
            #    unreachable (cuRobo "plan_pose returned None" x10 per goal).
            #  * q=[0..0] is a double singularity (J1/J3/J5 collinear).  Two
            #    held-egg RUNAWAY drops happened at ~92-93% of the path into q0
            #    (J3 -6.1 rad/s vs J5 +3.6 rad/s = internal self-motion), and the
            #    arm kept diverging after freeze (J6 54.9 rad/s).
            # V3.02 therefore:
            #  1) searches a 12-member yaw family (every 30 deg about world Z) for
            #     every vertical-egg goal, screens it with BioIK, and plans only the
            #     best few with cuRobo;
            #  2) replaces the q0 fallback with a non-singular SAFE-HIGH hold pose
            #     (J2/J4 bent, approach horizontal, J7 chosen by batch FK so the egg
            #     is already vertical) - no collinear joint pair, min singular
            #     value ~0.14 vs 0.0 at q0;
            #  3) keeps the legacy q0 path only behind NERO_V302_ALLOW_Q0_FALLBACK=1.
            # ============================================================
            whole_ok24 = wrist_ok24
            selected_strategy24 = "wrist_sparse" if wrist_ok24 else None
            zero_used24 = False
            at_slot_above302 = False
            verification24["v302_yaw_family_deg"] = list(range(0, 360, 30))

            _V302_YAWS = [0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 180.0]

            def _v302_qz(deg302):
                h302 = 0.5 * math.radians(float(deg302))
                return [0.0, 0.0, math.sin(h302), math.cos(h302)]

            def _v302_vertical_family(tcpq302, axis302):
                targ302 = [0.0, 0.0, 1.0 if float(axis302[2]) >= 0.0 else -1.0]
                qv302 = _v224_qmul(_v224_q_from_two_vectors(axis302, targ302), tcpq302)
                return [(yaw302, _v224_qmul(_v302_qz(yaw302), qv302)) for yaw302 in _V302_YAWS]

            def _v302_ik_rank(cands302, label302):
                """cands302: list of (tag, obj_target, tcp_target, quat). BioIK screen + rank."""
                live302 = skill_core._current_joint_state_copy()
                cur302 = [float(v) for v in skill_core._arm_positions_from_joint_state(live302)]
                seeds302 = [("current", live302)]
                try:
                    extra302 = skill_core._v70_natural_seed_variants(live302, cur302, reorientation=True)
                    seeds302.extend([sv for sv in extra302 if sv[0] != "current"][:2])
                except Exception:
                    pass
                ranked302 = []
                for seed_i302, (seed_tag302, seed302) in enumerate(seeds302):
                    for c302 in cands302:
                        tag302, objt302, tcpt302, qt302 = c302
                        try:
                            j302, _err302 = skill_core._compute_ik(
                                tcpt302, qt302, timeout_s=0.05,
                                avoid_collisions=True, seed_joint_state=seed302,
                            )
                        except Exception:
                            j302 = None
                        if j302 is None:
                            continue
                        sc302, _m302 = skill_core._v70_endpoint_score(cur302, j302)
                        ranked302.append((tuple(float(v) for v in sc302), c302, j302, seed_tag302))
                    if ranked302:
                        break  # extra seeds only when the current seed finds nothing
                ranked302.sort(key=lambda x: x[0])
                skill_core.emit(
                    f"[V3.02 IK SCREEN] {label302} | feasible={len(ranked302)}/{len(cands302)} "
                    f"| seeds_tried={min(len(seeds302), seed_i302 + 1) if seeds302 else 0}"
                )
                if ranked302 and os.environ.get("NERO_V341_LIMIT_MARGIN", "1") != "0":
                    # V3.41 JOINT-LIMIT MARGIN: goals with every joint >= NERO_V341_LIMIT_MARGIN_DEG from its limit;
                    # if none, >= NERO_V341_LIMIT_MARGIN_MIN_DEG (logged). Larger margin first (4 deg buckets), then
                    # the V70 score. V3.40 log: slot entry with J6 0.6 deg from its limit -> no descent possible,
                    # cuRobo failing every start state within 1-3 deg of a limit.
                    mreq341, mmin341 = v341_margin_required_deg(), v341_margin_relaxed_deg()
                    marg341 = {id(r): v341_limit_margins_deg(r[2])[0] for r in ranked302}
                    strict341 = [r for r in ranked302 if marg341[id(r)] >= mreq341]
                    relax341 = [r for r in ranked302 if marg341[id(r)] >= mmin341]
                    keep341 = strict341 or relax341
                    tag341 = ("ok" if strict341 else (f"RELAXED to {mmin341:.0f} deg" if relax341 else "none - kept all"))
                    if keep341:
                        ranked302 = keep341
                    ranked302.sort(key=lambda r: (-int(min(marg341[id(r)], 20.0) // 4), r[0]))
                    skill_core.emit(
                        f"[V3.41 LIMIT MARGIN] {label302} | >= {mreq341:.0f} deg: {len(strict341)} | >= {mmin341:.0f} deg: "
                        f"{len(relax341)} | {tag341} | best margin {max(marg341.values()):.1f} deg")
                return ranked302

            def _v302_plan_ranked(ranked302, rl302, al302, label302, max_plan302=4, min_obj_z302=0.075):
                live302 = skill_core._current_joint_state_copy()
                qstart302 = [float(v) for v in skill_core._arm_positions_from_joint_state(live302)]
                plans302 = []
                for _sc302, c302, _j302, _st302 in ranked302[:max(1, int(max_plan302))]:
                    tag302, objt302, tcpt302, qt302 = c302
                    try:
                        traj302 = skill_core._plan_pose(
                            tcpt302, qt302,
                            velocity_scale=0.16, accel_scale=0.12,
                            orientation_tolerance=0.10, position_tolerance=0.012,
                            planning_attempts=6, planning_time=6.0,
                        )
                    except Exception as exc302:
                        skill_core.emit(f"[V3.02 PLAN FAIL] {label302} {tag302} | {exc302}")
                        traj302 = _v340_joint_fallback(f"{label302} {tag302}", exc302, _j302, 0.16, 0.12)
                        if traj302 is None:
                            continue
                    try:
                        if (traj302 is not None and _j302 is not None
                                and os.environ.get("NERO_V341_LIMIT_MARGIN", "1") != "0"):
                            pts341 = list(traj302.joint_trajectory.points)
                            end341 = [float(v) for v in pts341[-1].positions][:7] if pts341 else None
                            if end341 is not None:
                                me341 = v341_limit_margins_deg(end341)
                                ms341 = v341_limit_margins_deg(_j302)
                                if me341[0] < v341_margin_required_deg() and ms341[0] > me341[0] + 2.0:
                                    jt341 = skill_core._plan_joint_solution([float(v) for v in _j302], velocity_scale=0.16,
                                                                            accel_scale=0.12, planning_attempts=6,
                                                                            planning_time=5.0)
                                    if jt341 is not None:
                                        skill_core.emit(f"[V3.41 ENDPOINT MARGIN] {label302} {tag302} | cuRobo end J{me341[1]} "
                                                        f"{me341[0]:.1f} deg from its limit -> MoveIt joint goal to the "
                                                        f"screened IK (J{ms341[1]} {ms341[0]:.1f} deg)")
                                        traj302 = jt341
                    except Exception as exc341:
                        skill_core.emit(f"[V3.41 ENDPOINT MARGIN WARN] {label302} {tag302} | {str(exc341)[:120]}")
                    clear302, why302 = _v224_traj_object_clear(traj302, rl302, al302, min_obj_z=min_obj_z302)
                    if not clear302:
                        skill_core.emit(f"[V3.02 HELD-OBJECT REJECT] {label302} {tag302} | {why302}")
                        continue
                    plans302.append((_v224_plan_score(traj302, qstart302), traj302, tcpt302, objt302, qt302, tag302))
                    if len(plans302) >= 2:
                        break
                plans302.sort(key=lambda x: x[0])
                return plans302

            # ============================================================
            # V3.03 DESCENT-CORRIDOR GATE
            # ------------------------------------------------------------
            # V3.02 log 2026-10-21 (egg_2 -> slot_3): the yaw chosen at slot-above
            # (obj z=0.298) had NO IK solution at slot-entry depth (obj z=0.123);
            # move_to_slot_entry then got "plan_pose returned None" x10.  A
            # slot-above goal is now accepted only if the SAME TCP orientation has
            # a continuous chained-IK solution (seeded level by level, bounded
            # joint step) all the way down to the insertion end depth.  After
            # execution the corridor is re-checked from the ACTUAL joint branch.
            # ============================================================
            _V303_CORRIDOR = os.environ.get("NERO_V303_CORRIDOR", "1") != "0"
            _V303_LEVEL_M = 0.04
            _V303_STEP_RAD = math.radians(float(os.environ.get("NERO_V303_CORRIDOR_MAX_STEP_DEG", "20")))

            def _v303_corridor_levels(top_z303):
                entry303 = float(plan24.get("entry_object_center_z_m", 0.1215))
                end303 = entry303 - 0.020  # insertion stops at the capture-zone entry (~22 mm below in the 15:14 run)
                lv303 = []
                n303 = max(1, int(math.ceil((float(top_z303) - entry303) / _V303_LEVEL_M - 1e-9)))
                for k303 in range(1, n303 + 1):
                    lv303.append(float(top_z303) + (entry303 - float(top_z303)) * k303 / n303)
                if os.environ.get("NERO_V341_LIMIT_MARGIN", "1") != "0":
                    # V3.41: the insertion itself (10 mm levels down to NERO_V341_CORRIDOR_DEPTH_M below the entry)
                    depth341 = float(os.environ.get("NERO_V341_CORRIDOR_DEPTH_M", "0.030"))
                    k341 = 1
                    while 0.010 * k341 < depth341 - 1e-9:
                        lv303.append(entry303 - 0.010 * k341)
                        k341 += 1
                    end303 = entry303 - depth341
                lv303.append(end303)
                return lv303

            def _v303_corridor_check(j_top303, objt303, qt303, rl303):
                """Chained BioIK from j_top303 down the vertical corridor at fixed TCP orientation."""
                try:
                    seed303 = skill_core._seed_with_arm_positions(skill_core._current_joint_state_copy(), list(j_top303))
                except Exception:
                    seed303 = None
                prev303 = [float(v) for v in j_top303]
                worst303 = 0.0
                for z303 in _v303_corridor_levels(float(objt303[2])):
                    o303 = [float(objt303[0]), float(objt303[1]), float(z303)]
                    tcp303 = _v224_target_tcp_for_object(o303, qt303, rl303)
                    try:
                        j303, _e303 = skill_core._compute_ik(tcp303, qt303, timeout_s=0.05,
                                                            avoid_collisions=True, seed_joint_state=seed303)
                    except Exception as exc303:
                        return False, f"IK exception at obj_z={z303*1000:.0f}mm: {exc303}", worst303
                    if j303 is None:
                        return False, f"no IK at obj_z={z303*1000:.0f}mm", worst303
                    step303 = max(abs(float(j303[i]) - prev303[i]) for i in range(7))
                    if step303 > _V303_STEP_RAD:
                        return False, (f"branch jump {math.degrees(step303):.0f}deg at obj_z={z303*1000:.0f}mm"), worst303
                    if V341_STATE.get("corr_margin") is not None:
                        m341, jn341, _l341 = v341_limit_margins_deg(j303)
                        if m341 < float(V341_STATE["corr_margin"]):
                            return False, (f"J{jn341} {m341:.1f} deg from its limit at obj_z={z303*1000:.0f}mm "
                                           f"(< {float(V341_STATE['corr_margin']):.0f})"), worst303
                    worst303 = max(worst303, step303)
                    prev303 = [float(v) for v in j303]
                    try:
                        seed303 = skill_core._seed_with_arm_positions(seed303, prev303)
                    except Exception:
                        pass
                return True, f"ok max_step={math.degrees(worst303):.1f}deg", worst303

            class _V337Recenter(Exception):
                pass

            _V337_MIN_FEASIBLE = int(os.environ.get("NERO_V337_MIN_FEASIBLE", "3"))
            _V337_MAX_JOINT_DEG = float(os.environ.get("NERO_V337_MAX_JOINT_TRAVEL_DEG", "150"))

            def _v337_risk(ranked337):
                """V3.37: a whole-arm reorientation is risky when only a few yaw goals are reachable from here (the
                arm is at the edge of its workspace) or the best one swings a joint very far (the V3.36 egg_1 drop:
                1/60 feasible, 'p2/yaw+180', RUNAWAY 3 s into the motion)."""
                if not ranked337:
                    # V3.38: BioIK finding 0 goals is NOT a reason to move first - cuRobo often still plans one
                    # (V3.37 log: 0/60, recenter unreachable, then cuRobo found p0/yaw+30). Plan here first; the
                    # recenter is used only if that planning fails.
                    return None
                if len(ranked337) < _V337_MIN_FEASIBLE:
                    return f"only {len(ranked337)} feasible yaw goal(s) < {_V337_MIN_FEASIBLE}"
                try:
                    cur337 = [float(v) for v in skill_core._arm_positions_from_joint_state(
                        skill_core._current_joint_state_copy())]
                    j337 = ranked337[0][2]
                    if j337 is not None:
                        trav337 = max(abs(float(j337[i]) - cur337[i]) for i in range(7))
                        if math.degrees(trav337) > _V337_MAX_JOINT_DEG:
                            return (f"best goal {ranked337[0][1][0]} swings a joint {math.degrees(trav337):.0f}deg "
                                    f"> {_V337_MAX_JOINT_DEG:.0f}deg")
                except Exception:
                    pass
                return None

            def _v302_vertical_goal(obj_targets302, label302, max_plan302=4, corridor303=False, risk337=False):
                """Plan + execute ONE whole trajectory to a vertical-egg goal (yaw free).

                corridor303=True (slot-above goals, V3.03): only yaws whose descent
                corridor to the insertion depth is IK-continuous are planned, and
                the executed branch is re-checked; on failure one more yaw is tried.
                """
                use303 = bool(corridor303 and _V303_CORRIDOR)
                last303 = None
                for round303 in range(2 if use303 else 1):
                    objc302, objqc302, tcpc302, tcpqc302, axc302, rlc302, alc302 = _v224_refresh_rigid()
                    fam302 = _v302_vertical_family(tcpqc302, axc302)
                    cands302 = []
                    for k302, objt302 in enumerate(obj_targets302):
                        for yaw302, qt302 in fam302:
                            if round303 > 0 and abs(float(yaw302)) < 15.0:
                                continue  # the executed (current) yaw already failed the post-check
                            if (use303 and round303 == 0 and abs(float(yaw302)) < 15.0
                                    and V341_STATE["skip_yaw"].get(str(obj)) == str(slot_name)):
                                continue  # V3.41: the current yaw could not descend into this slot (insert blocked)
                            tcpt302 = _v224_target_tcp_for_object(objt302, qt302, rlc302)
                            cands302.append((f"p{k302}/yaw{yaw302:+.0f}", objt302, tcpt302, qt302))
                    ranked302 = _v302_ik_rank(cands302, label302)
                    if risk337 and not use303:
                        why337 = _v337_risk(ranked302)
                        if why337:
                            raise _V337Recenter(why337)
                    if use303:
                        kept303 = []
                        checked303 = 0
                        margins341 = ([v341_margin_required_deg(), v341_margin_relaxed_deg()]
                                      if os.environ.get("NERO_V341_LIMIT_MARGIN", "1") != "0" else [None])
                        for mg341 in margins341:
                            V341_STATE["corr_margin"] = mg341
                            for r303 in ranked302:
                                checked303 += 1
                                ok303, why303, _w303 = _v303_corridor_check(r303[2], r303[1][1], r303[1][3], rlc302)
                                if ok303:
                                    kept303.append(r303)
                                    if len(kept303) >= max(1, int(max_plan302)):
                                        break
                                else:
                                    skill_core.emit(f"[V3.03 CORRIDOR REJECT] {label302} {r303[1][0]} | {why303}")
                            if kept303:
                                if mg341 is not None:
                                    skill_core.emit(f"[V3.41 CORRIDOR MARGIN] {label302} | every level >= {mg341:.0f} deg "
                                                    f"from the joint limits ({len(kept303)} yaw(s))")
                                break
                        skill_core.emit(
                            f"[V3.03 CORRIDOR] {label302} | pass={len(kept303)}/{checked303} checked "
                            f"| levels_mm={[round(z*1000) for z in _v303_corridor_levels(float(obj_targets302[0][2]))]}"
                        )
                        if not kept303:
                            V341_STATE["corr_margin"] = None
                            raise RuntimeError(
                                f"DESCENT_CORRIDOR_UNREACHABLE: {label302}: no yaw reaches slot-above AND "
                                f"has a continuous vertical descent to insertion depth"
                            )
                        ranked302 = kept303
                    elif not ranked302:
                        # BioIK found nothing: still give cuRobo the first yaw candidates once.
                        ranked302 = [((0.0,), c302, None, "unscreened") for c302 in cands302[:3]]
                    plans302 = _v302_plan_ranked(ranked302, rlc302, alc302, label302, max_plan302=max_plan302)
                    if not plans302:
                        raise RuntimeError(f"{label302}: no yaw-family goal survived IK + cuRobo + held-object check")
                    _ps302, traj302, tcpt302, objt302, qt302, tag302 = plans302[0]
                    skill_core.emit(
                        f"[V3.02 VERTICAL GOAL PLAN] {label302} | choice={tag302} | "
                        f"target_obj={[round(v,3) for v in objt302]} | plans={len(plans302)}"
                    )
                    ok302, ver302, warn302 = _v224_execute_pose_plan(
                        f"v302_{label302}({obj},{slot_name})", traj302, tcpt302, rlc302, alc302,
                    )
                    warnings24.extend(warn302)
                    if not ok302 or not bool(hold_active_shared.value) or int(safety_trip_count.value) != safety0:
                        raise RuntimeError(f"{label302}: execution/hold/safety verification failed")
                    _o302, _oq302, _t302, _tq302, axa302, _r302, _a302 = _v224_refresh_rigid()
                    err302 = _v224_axis_err(axa302)
                    if err302 > V271_AXIS_TOL_DEG:
                        raise RuntimeError(f"{label302}: endpoint axis error {err302:.2f}deg")
                    if not use303:
                        return err302, tag302
                    # Post-arrival check on the branch cuRobo actually took.
                    if os.environ.get("NERO_V341_LIMIT_MARGIN", "1") != "0":
                        V341_STATE["corr_margin"] = v341_margin_relaxed_deg()
                    jact303 = [float(v) for v in skill_core._arm_positions_from_joint_state(
                        skill_core._current_joint_state_copy())]
                    okp303, whyp303, _wp303 = _v303_corridor_check(
                        jact303, [float(objt302[0]), float(objt302[1]), float(_o302[2])], _tq302, _r302)
                    V341_STATE["corr_margin"] = None
                    if okp303:
                        V341_STATE["skip_yaw"].pop(str(obj), None)
                        skill_core.emit(f"[V3.03 CORRIDOR POST-CHECK OK] {label302} {tag302} | {whyp303}")
                        verification24["v303_descent_corridor"] = whyp303
                        return err302, tag302
                    last303 = whyp303
                    skill_core.emit(
                        f"[V3.03 CORRIDOR POST-CHECK FAIL] {label302} {tag302} | executed branch: {whyp303} "
                        f"-> {'re-plan another yaw at hover' if round303 == 0 else 'give up'}"
                    )
                V341_STATE["corr_margin"] = None
                raise RuntimeError(f"DESCENT_CORRIDOR_UNREACHABLE: {label302}: executed branch has no descent corridor ({last303})")

            def _v302_slot_above_z(from_high302):
                # V3.03: hover a fixed height above the slot-entry depth instead of
                # staying at the (much higher) transport/orientation height.
                # V3.02 hovered at obj z=0.298 -> 177 mm blind descent.
                entry_z302 = float(plan24.get("entry_object_center_z_m", 0.1215))
                hover303 = float(os.environ.get("NERO_V303_HOVER_ABOVE_ENTRY_M", "0.10"))
                return max(entry_z302 + max(0.06, hover303), 0.20)

            # ------------------------------------------------------------
            # STRATEGY 2 (V3.02): whole-arm vertical goal, yaw family at 5 anchors
            # ------------------------------------------------------------
            def _v302_anchors():
                objc24, objqc24, tcpc24, tcpqc24, axc24, rlc24, alc24 = _v224_refresh_rigid()
                zbase24 = max(float(objc24[2]), 0.19)
                midx24 = 0.5*(float(objc24[0])+float(slot24[0]))
                midy24 = 0.5*(float(objc24[1])+float(slot24[1]))
                return [
                    [float(slot24[0]), float(slot24[1]), max(zbase24, 0.22)],
                    [midx24, midy24, max(zbase24, 0.22)],
                    [float(objc24[0]), float(objc24[1]), max(zbase24, 0.23)],
                    [-0.4225, 0.0, max(zbase24, 0.22)],
                    [float(objc24[0]), float(objc24[1]), zbase24],
                ]

            def _v337_recenter(why337):
                """V3.37/V3.38: carry the (still horizontal) egg towards the middle of the workspace with the grasp
                orientation unchanged, so the reorientation starts from a configuration with many reachable yaw goals.
                V3.38: several candidate points / heights (NERO_V338_RECENTER_POINTS, NERO_V338_RECENTER_Z_M); each is
                IK-screened with the CURRENT grasp orientation before it is planned (V3.37 log: the single fixed
                point had no IK for that orientation)."""
                objc, _oq, tcpc, tcpq, _ax, rl, al = _v224_refresh_rigid()
                pts338 = []
                for it in os.environ.get("NERO_V338_RECENTER_POINTS",
                                         os.environ.get("NERO_V337_RECENTER_XY", "-0.42,-0.02") +
                                         ";-0.40,-0.08;-0.45,0.00;-0.38,-0.12;-0.44,-0.10").split(";"):
                    try:
                        x_, y_ = [float(v) for v in it.split(",")[:2]]
                        pts338.append((x_, y_))
                    except Exception:
                        pass
                try:
                    zs338 = [float(v) for v in os.environ.get("NERO_V338_RECENTER_Z_M", "0.23,0.20,0.26").split(",")]
                except Exception:
                    zs338 = [0.23, 0.20, 0.26]
                live338 = skill_core._current_joint_state_copy()
                cands338 = []
                for mx, my in pts338:
                    if math.hypot(float(objc[0]) - mx, float(objc[1]) - my) < 0.05:
                        continue
                    for z_ in zs338:
                        mz = max(float(objc[2]), float(z_))
                        tcpt = [float(tcpc[0]) + mx - float(objc[0]), float(tcpc[1]) + my - float(objc[1]),
                                float(tcpc[2]) + mz - float(objc[2])]
                        try:
                            j338, _e338 = skill_core._compute_ik(tcpt, list(tcpq), timeout_s=0.05,
                                                                 avoid_collisions=True, seed_joint_state=live338)
                        except Exception:
                            j338 = None
                        if j338 is not None:
                            cands338.append(((mx, my, mz), tcpt, j338))
                            break                                       # one height per point is enough
                skill_core.emit(f"[V3.38 RECENTER SCREEN] {obj} | {len(cands338)}/{len(pts338)} centre points "
                                f"reachable with the current grasp orientation")
                if not cands338:
                    skill_core.emit(f"[V3.37 RECENTER SKIP] {obj} | no reachable centre point ({why337})")
                    return False
                for (mx, my, mz), tcpt, j340 in cands338[:3]:
                    skill_core.emit(f"[V3.37 RECENTER] {obj} | {why337} -> carry the egg (orientation unchanged) from "
                                    f"{[round(float(v), 3) for v in objc]} to {[round(mx, 3), round(my, 3), round(mz, 3)]} "
                                    "and re-plan the reorientation there")
                    try:
                        traj = skill_core._plan_pose(tcpt, list(tcpq), velocity_scale=0.12, accel_scale=0.10,
                                                     orientation_tolerance=0.05, position_tolerance=0.010,
                                                     planning_attempts=6, planning_time=6.0)
                    except Exception as exc337:
                        skill_core.emit(f"[V3.37 RECENTER FAIL] {obj} | plan: {str(exc337)[:160]}")
                        traj = _v340_joint_fallback(f"v337_recenter({obj})", exc337, j340, 0.12, 0.10)
                        if traj is None:
                            continue
                    if traj is None:
                        continue
                    ok337, _ver337, warn337 = _v224_execute_pose_plan(f"v337_recenter({obj},{slot_name})", traj, tcpt, rl, al)
                    warnings24.extend(warn337)
                    if not ok337 or not bool(hold_active_shared.value) or int(safety_trip_count.value) != safety0:
                        skill_core.emit(f"[V3.37 RECENTER FAIL] {obj} | execution/hold/safety check failed")
                        return False
                    verification24["v337_recentered"] = {"reason": why337, "to_xyz": [round(mx, 3), round(my, 3), round(mz, 3)]}
                    return True
                return False

            if not whole_ok24 and bool(hold_active_shared.value) and int(safety_trip_count.value) == safety0:
                try:
                    use337 = os.environ.get("NERO_V337_RECENTER", "1") != "0"
                    try:
                        err302, tag302 = _v302_vertical_goal(_v302_anchors(), "whole_arm_yaw_family", max_plan302=4,
                                                             risk337=use337)
                    except _V337Recenter as r337:
                        if not _v337_recenter(str(r337)):
                            if not bool(hold_active_shared.value) or int(safety_trip_count.value) != safety0:
                                raise RuntimeError("V3.37 recenter move failed (hold/safety)")
                            skill_core.emit(f"[V3.37 RECENTER] {obj} | could not recenter -> planning from here anyway")
                        err302, tag302 = _v302_vertical_goal(_v302_anchors(), "whole_arm_yaw_family", max_plan302=4)
                    except RuntimeError as e338:
                        # V3.38: nothing could be planned from here (no motion happened) -> recenter, then plan again
                        if not (use337 and "no yaw-family goal survived" in str(e338)
                                and bool(hold_active_shared.value) and int(safety_trip_count.value) == safety0):
                            raise
                        if not _v337_recenter("no reorientation plan from this pose"):
                            raise
                        err302, tag302 = _v302_vertical_goal(_v302_anchors(), "whole_arm_yaw_family", max_plan302=4)
                    whole_ok24 = True
                    selected_strategy24 = "v302_whole_arm_yaw_family"
                    verification24["whole_arm_axis_error_after_deg"] = round(float(err302), 3)
                    verification24["v302_whole_arm_choice"] = tag302
                    skill_core.emit(
                        f"[V3.02 WHOLE-ARM SUCCESS] axis_error={err302:.2f}deg | choice={tag302}"
                    )
                except Exception as exc24:
                    warnings24.append(f"V3.02 whole-arm yaw-family strategy failed: {exc24}")
                    skill_core.emit(f"[V3.02 WHOLE-ARM FAIL] {exc24}")

            # ------------------------------------------------------------
            # STRATEGY 3 (V3.02): NON-SINGULAR SAFE-HIGH HOLD POSE
            # Joint-space move (validated, slow) into a pose found by batch FK in
            # the family J2/J4 = (10/80, 15/75, 20/70) deg (approach horizontal),
            # J3=J5=0, J6 in {0,+-20}, J1 in {0,+-15}, J7 swept -90..90 (837 poses).
            # Offline check: egg axis = TCP x for major-aligned grasps; 54/837 poses vertical <= 6 deg.  Chosen so the held
            # egg is already vertical and high; no collinear joint pair.
            # ------------------------------------------------------------
            def _v302_safe_high_hold():
                objc302, objqc302, tcpc302, tcpqc302, axc302, rlc302, alc302 = _v224_refresh_rigid()
                live302 = skill_core._current_joint_state_copy()
                qcur302 = [float(v) for v in skill_core._arm_positions_from_joint_state(live302)]
                cands302 = []
                for j24302, j42302 in ((15.0, 75.0), (10.0, 80.0), (20.0, 70.0)):
                    for j1302 in (0.0, -15.0, 15.0):
                        for j6302 in (0.0, -20.0, 20.0):
                            for j7302 in range(-90, 91, 6):
                                q302 = [math.radians(j1302), math.radians(j24302), 0.0,
                                        math.radians(j42302), 0.0, math.radians(j6302),
                                        math.radians(float(j7302))]
                                okl302 = all(
                                    float(lim302[0]) + math.radians(3.0) <= q302[i302] <= float(lim302[1]) - math.radians(3.0)
                                    for i302, lim302 in enumerate(JOINT_LIMITS_RAD)
                                )
                                if okl302:
                                    cands302.append(q302)
                poses302 = _v224_fk_batch(cands302)
                scored302 = []
                for q302, pose302 in zip(cands302, poses302):
                    objp302, axp302 = _v224_predict(pose302, rlc302, alc302)
                    e302 = _v224_axis_err(axp302)
                    if e302 > 0.6 * float(V271_AXIS_TOL_DEG):
                        continue
                    if not (0.28 <= float(objp302[2]) <= 0.70):
                        continue
                    dxy302 = math.hypot(float(objp302[0]) - float(slot24[0]), float(objp302[1]) - float(slot24[1]))
                    travel302 = max(abs(q302[i] - qcur302[i]) for i in range(7))
                    scored302.append(((round(e302, 1), round(dxy302, 2), travel302), q302, objp302, e302))
                scored302.sort(key=lambda x: x[0])
                skill_core.emit(
                    f"[V3.02 SAFE-HIGH HOLD SEARCH] FK candidates={len(cands302)} | vertical+high={len(scored302)}"
                )
                last_why302 = "no vertical safe-high candidate"
                for _s302, qh302, objp302, e302 in scored302[:6]:
                    qpath302 = _v224_interp_q(qcur302, qh302, step_deg=4.0)
                    valid302, why302, pposes302 = _v224_validate_joint_path(qpath302, rlc302, alc302, min_obj_z=0.075)
                    if not valid302:
                        last_why302 = why302
                        continue
                    xyz302 = [[float(v) for v in pp302["position"]] for pp302 in pposes302]
                    skill_core.emit(
                        f"[V3.02 SAFE-HIGH HOLD] q_deg={[round(math.degrees(v),1) for v in qh302]} | "
                        f"predicted_axis={e302:.2f}deg | predicted_obj={[round(v,3) for v in objp302]} | "
                        f"points={len(qpath302)} | speed=SLOW | gripper force/hold PRESERVED"
                    )
                    ok302, ver302, warn302 = _v224_exec_joint_path(
                        f"v302_safe_high_hold({obj},{slot_name})", qpath302, xyz302, slow_factor=1.65,
                    )
                    warnings24.extend(warn302)
                    if not ok302 or not bool(hold_active_shared.value) or int(safety_trip_count.value) != safety0:
                        raise RuntimeError("safe-high hold execution/hold/safety verification failed")
                    _o302, _oq302, _t302, _tq302, axa302, _r302, _a302 = _v224_refresh_rigid()
                    erra302 = _v224_axis_err(axa302)
                    skill_core.emit(f"[V3.02 SAFE-HIGH HOLD REACHED] axis_error={erra302:.2f}deg | hold=true")
                    return erra302
                raise RuntimeError("safe-high hold: " + last_why302)

            if not whole_ok24 and bool(hold_active_shared.value) and int(safety_trip_count.value) == safety0:
                try:
                    erra302 = _v302_safe_high_hold()
                    verification24["v302_safe_high_axis_error_deg"] = round(float(erra302), 3)
                    # From the hold pose: one whole trajectory to vertical egg above
                    # the slot (yaw family, safe Z re-anchored like V2-31).
                    zt302 = _v302_slot_above_z(from_high302=True)
                    err302, tag302 = _v302_vertical_goal(
                        [[float(slot24[0]), float(slot24[1]), zt302]], "safe_high_to_slot_above", max_plan302=4, corridor303=True,
                    )
                    whole_ok24 = True
                    at_slot_above302 = True
                    selected_strategy24 = "v302_safe_high_hold_then_curobo"
                    verification24["v302_safe_high_choice"] = tag302
                    skill_core.emit(
                        f"[V3.02 SAFE-HIGH SUCCESS] slot-above reached | axis_error={err302:.2f}deg | choice={tag302}"
                    )
                except Exception as exc24:
                    warnings24.append(f"V3.02 safe-high hold strategy failed: {exc24}")
                    skill_core.emit(f"[V3.02 SAFE-HIGH FAIL] {exc24}")

            # ------------------------------------------------------------
            # STRATEGY 4 (legacy, OFF by default): slow q=0 then yaw-family
            # ------------------------------------------------------------
            if (not whole_ok24 and os.environ.get("NERO_V302_ALLOW_Q0_FALLBACK", "0") == "1"
                    and bool(hold_active_shared.value) and int(safety_trip_count.value) == safety0):
                zero_used24 = True
                try:
                    objc24, objqc24, tcpc24, tcpqc24, axc24, rlc24, alc24 = _v224_refresh_rigid()
                    live24 = skill_core._current_joint_state_copy()
                    qcur24 = [float(v) for v in skill_core._arm_positions_from_joint_state(live24)]
                    qpath24 = _v224_interp_q(qcur24, [0.0]*7, step_deg=4.0)
                    valid24, why24, poses24 = _v224_validate_joint_path(qpath24, rlc24, alc24, min_obj_z=0.075)
                    if not valid24:
                        raise RuntimeError("q=0 path rejected: " + why24)
                    xyz24 = [[float(v) for v in p24["position"]] for p24 in poses24]
                    skill_core.emit("[V3.02 LEGACY Q0 FALLBACK] enabled by NERO_V302_ALLOW_Q0_FALLBACK=1")
                    ok24, ver24, warn24 = _v224_exec_joint_path(
                        f"v224_slow_zero({obj},{slot_name})", qpath24, xyz24, slow_factor=1.65,
                    )
                    warnings24.extend(warn24)
                    if not ok24 or not bool(hold_active_shared.value) or int(safety_trip_count.value) != safety0:
                        raise RuntimeError("q=0 execution/hold verification failed")
                    zt302 = _v302_slot_above_z(from_high302=True)
                    err302, tag302 = _v302_vertical_goal(
                        [[float(slot24[0]), float(slot24[1]), zt302]], "q0_to_slot_above", max_plan302=4, corridor303=True,
                    )
                    whole_ok24 = True
                    at_slot_above302 = True
                    selected_strategy24 = "slow_zero_then_curobo"
                    skill_core.emit(f"[V2-24 ZERO FALLBACK SUCCESS] slot-above reached | axis_error={err302:.2f}deg")
                except Exception as exc24:
                    warnings24.append(f"V2-24 zero fallback failed: {exc24}")
                    skill_core.emit(f"[V2-24 ZERO FALLBACK FAIL] {exc24}")

            if not whole_ok24:
                verification24.update({
                    "reason": "V224_ALL_ORIENTATION_STRATEGIES_FAILED",
                    "selected_strategy": selected_strategy24,
                    "zero_fallback_used": bool(zero_used24),
                    "hold_active": bool(hold_active_shared.value),
                    "safety_trip_after": int(safety_trip_count.value),
                })
                return _v24_result(
                    name, False, contract, state_before, verification24, warnings24,
                    {"elapsed_s": round(time.perf_counter()-t0, 3),
                     "hint": ("V3.12.1: the held egg cannot be reoriented from this grasp. Do not try more slots; call "
                              "return_to_pick_location, then continue with the other eggs.")},
                    next_stage=None, recommended_next_actions=["return_to_pick_location"],
                )

            # ============================================================
            # SLOT-ABOVE TRANSPORT (V3.02): yaw-free vertical goal above the slot.
            # On failure, recover once through the non-singular safe-high hold.
            # ============================================================
            if not at_slot_above302:
                try:
                    zt302 = _v302_slot_above_z(from_high302=False)
                    err302, tag302 = _v302_vertical_goal(
                        [[float(slot24[0]), float(slot24[1]), zt302]], "slot_above_transport", max_plan302=4, corridor303=True,
                    )
                    at_slot_above302 = True
                    skill_core.emit(
                        f"[V2-24 SLOT-ABOVE WHOLE TRAJECTORY] strategy={selected_strategy24} | "
                        f"yaw-free choice={tag302} | axis={err302:.2f}deg"
                    )
                except Exception as exc24:
                    skill_core.emit(f"[V3.02 SLOT TRANSPORT FAIL -> SAFE-HIGH RECOVERY] {exc24}")
                    try:
                        if not bool(hold_active_shared.value) or int(safety_trip_count.value) != safety0:
                            raise RuntimeError("hold/safety changed; refusing recovery motion")
                        _v302_safe_high_hold()
                        zt302 = _v302_slot_above_z(from_high302=True)
                        err302, tag302 = _v302_vertical_goal(
                            [[float(slot24[0]), float(slot24[1]), zt302]], "recovery_safe_high_to_slot_above",
                            max_plan302=4, corridor303=True,
                        )
                        at_slot_above302 = True
                        selected_strategy24 = str(selected_strategy24) + "+v302_safe_high_recovery"
                    except Exception as excz24:
                        corridor303 = "DESCENT_CORRIDOR_UNREACHABLE" in (str(exc24) + str(excz24))
                        verification24["reason"] = ("V303_SLOT_DESCENT_UNREACHABLE" if corridor303
                                                    else "V224_SLOT_ABOVE_TRANSPORT_FAILED")
                        verification24["hold_active"] = bool(hold_active_shared.value)
                        warnings24.append(str(exc24))
                        warnings24.append(str(excz24))
                        if corridor303 and bool(hold_active_shared.value):
                            warnings24.append(
                                f"{slot_name} cannot be entered vertically with this grasp. The egg is still held "
                                f"(stage LIFTED). Call plan_placement ONCE with a different empty slot."
                            )
                        return _v24_result(
                            name, False, contract, state_before, verification24, warnings24,
                            {"elapsed_s": round(time.perf_counter()-t0, 3)},
                            next_stage=None,
                            recommended_next_actions=(["plan_placement(slot_name=<different empty slot>)"]
                                                      if corridor303 and bool(hold_active_shared.value)
                                                      else ["observe_scene"]),
                        )

            # Final physical verify at slot-above.
            objend24, objqend24 = skill_core.object_pose(obj)
            tcpend24, _tcpqend24 = skill_core.current_tcp_pose()
            axend24 = _v27_egg_major_axis_from_quat(objqend24)["world_axis"]
            axiserr24 = _v224_axis_err(axend24)
            xyerr24 = _v24_xy_distance(objend24, slot24)
            okend24 = bool(
                bool(hold_active_shared.value)
                and int(safety_trip_count.value) == safety0
                and axiserr24 <= V271_AXIS_TOL_DEG
                and xyerr24 <= 0.015
            )
            verification24.update({
                "selected_strategy": selected_strategy24,
                "zero_fallback_used": bool(zero_used24),
                "axis_error_after_deg": round(float(axiserr24),3),
                "object_xy_error_to_slot_mm": round(float(xyerr24*1000.0),2),
                "hold_active": bool(hold_active_shared.value),
                "safety_trip_before": safety0,
                "safety_trip_after": int(safety_trip_count.value),
            })
            if not okend24:
                verification24["reason"] = "V224_SLOT_ABOVE_END_VERIFY_FAILED"
                return _v24_result(
                    name, False, contract, state_before, verification24, warnings24,
                    {"elapsed_s": round(time.perf_counter()-t0, 3)},
                    next_stage=None, recommended_next_actions=["observe_scene"],
                )

            # ============================================================
            # V2-24 CAPTURE-ZONE RELEASE POLICY
            # ------------------------------------------------------------
            # Do NOT force the held egg to the old fixed pre-release center Z
            # (~66.5 mm).  Release becomes eligible as soon as the lower end of
            # the vertically aligned egg is inside the slot's capture funnel.
            # Geometry-derived upper capture boundary:
            #   slot center Z + egg half length + capture allowance.
            # For egg_3 this is ~97-99 mm, matching the observed manual-release
            # success region while remaining above the holder walls.
            half_m24 = float(plan24.get("egg_half_length_mm", 30.5))*0.001
            capture_allowance24 = float(os.environ.get("NERO_V24_CAPTURE_ALLOWANCE_M", "0.024"))
            capture_upper24 = float(slot24[2]) + half_m24 + capture_allowance24
            capture_lower24 = float(slot24[2]) + max(0.006, half_m24 - 0.015)
            old_release_z24 = float(plan24.get("pre_release_object_center_z_m", capture_upper24))
            plan24["v224_release_policy"] = "geometry_capture_zone"
            plan24["v224_capture_zone_lower_center_z_m"] = float(capture_lower24)
            plan24["v224_capture_zone_upper_center_z_m"] = float(capture_upper24)
            plan24["v224_original_pre_release_center_z_m"] = float(old_release_z24)
            # Existing guarded_insert_and_release already performs staged opening,
            # release, and retreat once its target is reached.  Retarget that end
            # condition to the ENTRY of the geometry capture zone, eliminating the
            # unnecessary blocked deeper plunge.
            plan24["pre_release_object_center_z_m"] = float(capture_upper24)
            plan24["policy"] = (
                "short-axis grasp -> hierarchical free-path verticalization -> "
                "whole-trajectory slot-above -> capture-zone insertion -> gravity-assisted release"
            )
            skill_core.emit(
                f"[V2-24 CAPTURE-ZONE RELEASE] old_target={old_release_z24*1000.0:.1f}mm -> "
                f"zone=[{capture_lower24*1000.0:.1f},{capture_upper24*1000.0:.1f}]mm | "
                f"release target=ZONE ENTRY, not fixed deep Z"
            )

            V271_PLACEMENT_RUNTIME["oriented"][obj] = {
                "slot_name": slot_name,
                "axis_error_deg": round(float(axiserr24),2),
                "tcp_xyz_m": [round(float(v),6) for v in tcpend24],
                "object_xy_error_to_slot_mm": round(float(xyerr24*1000.0),2),
                "v224_hierarchical_ready": True,
                "selected_strategy": selected_strategy24,
                "capture_zone_upper_center_z_m": round(float(capture_upper24),6),
            }
            skill_core.emit(
                f"[V2-24 ORIENT/TRANSPORT OK] {obj}->{slot_name} | strategy={selected_strategy24} | "
                f"axis={axiserr24:.2f}deg | XY={xyerr24*1000.0:.1f}mm | hold=true | "
                f"next=move_to_slot_entry"
            )
            return _v24_result(
                name, True, contract, state_before, verification24, warnings24,
                {"elapsed_s": round(time.perf_counter()-t0, 3)},
                "PLACEMENT_ORIENTED", obj,
                recommended_next_actions=["move_to_slot_entry"],
            )

        if name == "move_to_slot_entry":
            # V2-32 R2: snapshot safety-trip baseline before slot-entry motion.
            safety0 = int(safety_trip_count.value)
            obj = str(args.get("object_name", state_before.get("active_object") or ""))
            slot_name = str(args.get("slot_name", ""))
            plan = V271_PLACEMENT_RUNTIME.get("plans", {}).get(obj)
            slot = _v271_slot(slot_name)
            if not plan or plan.get("slot_name") != slot_name or slot is None:
                return _v24_precondition_failure(name, args, "no matching placement plan", contract, state_before, recommended_next_actions=["plan_placement"])
            if state_before.get("stage") != "PLACEMENT_ORIENTED" or state_before.get("active_object") != obj:
                return _v24_precondition_failure(
                    name, args, f"requires PLACEMENT_ORIENTED for {obj}; current={state_before}",
                    contract, state_before, recommended_next_actions=["orient_for_slot"],
                )
            if not bool(hold_active_shared.value):
                return _v24_precondition_failure(name, args, "hold_active=false", contract, state_before, recommended_next_actions=["observe_scene"])

            obj0, obj_q0 = skill_core.object_pose(obj)
            tcp0, tcp_q0 = skill_core.current_tcp_pose()
            axis0 = _v27_egg_major_axis_from_quat(obj_q0)["world_axis"]
            axis_error0 = _v271_axis_line_error_deg(axis0)
            if axis_error0 > V271_AXIS_TOL_DEG:
                return _v24_precondition_failure(
                    name, args, f"egg long-axis alignment degraded to {axis_error0:.1f}deg; reorient before approaching holder",
                    contract, state_before,
                    physical_verification={"axis_error_deg": round(axis_error0,2)},
                    recommended_next_actions=["plan_placement", "orient_for_slot"],
                )

            llm_require_arm_controller()
            rel0 = [float(obj0[i]) - float(tcp0[i]) for i in range(3)]
            desired_obj = [float(slot[0]), float(slot[1]), float(plan["entry_object_center_z_m"])]
            target_tcp = [desired_obj[i] - rel0[i] for i in range(3)]

            # ============================================================
            # V3.03 LAYERED SLOT-ENTRY DESCENT
            #   L1 straight Cartesian descent, fixed orientation (pure vertical)
            #   L2 cuRobo pose plan to the same entry pose (V3.02 behaviour)
            #   L3 rotate the vertical egg about its own axis AT HOVER (yaw
            #      family, BioIK-screened hover+entry on one branch), then L1
            #   all fail -> structured SLOT_ENTRY_UNREACHABLE, stage back to
            #      LIFTED so the Agent can choose another slot (egg still held).
            # Every layer is exception-safe (V3.02 leaked a raw exception here).
            # ============================================================
            attempts303 = []
            warnings = []
            verification1 = {}
            ok1 = False

            def _v303_motion(label303, fn303, tgt303):
                try:
                    v303, o303, w303 = _v24_run_motion(label303, fn303, tgt303, 0.012, True)
                except Exception as exc303:
                    attempts303.append(f"{label303}: {exc303}")
                    skill_core.emit(f"[V3.03 SLOT ENTRY LAYER FAIL] {label303} | {exc303}")
                    return {"reason": "MOTION_EXCEPTION", "motion_exception": str(exc303)}, False, []
                if not o303:
                    attempts303.append(f"{label303}: {v303.get('reason') or v303.get('legacy_joint_verify_error') or 'postcheck failed'}")
                    skill_core.emit(f"[V3.03 SLOT ENTRY LAYER FAIL] {label303} | {attempts303[-1]}")
                return v303, bool(o303), list(w303 or [])

            def _v303_cart_descent(tgt303, q303):
                traj303 = skill_core._cartesian_approach_plan(tgt303, q303)
                pts303 = list(traj303.joint_trajectory.points)
                prev303 = None
                for pt303 in pts303:
                    cur303 = [float(v) for v in pt303.positions][:7]
                    if prev303 is not None and max(abs(cur303[i] - prev303[i]) for i in range(7)) > math.radians(10.0):
                        raise RuntimeError("CARTESIAN APPROACH BLOCKED: joint jump > 10deg between waypoints (branch flip)")
                    prev303 = cur303
                _v218_slow_trajectory_in_place(traj303, factor=1.5)
                skill_core._execute_trajectory(traj303)
                return True

            def _v303_still_safe():
                return bool(hold_active_shared.value) and int(safety_trip_count.value) == safety0

            # L1
            verification1, ok1, w1 = _v303_motion(
                f"v303_slot_entry_cartesian({obj},{slot_name})",
                lambda: _v303_cart_descent(target_tcp, tcp_q0), target_tcp)
            warnings.extend(w1)
            layer303 = "cartesian" if ok1 else None
            # L2
            if not ok1 and _v303_still_safe():
                tcp_now, tcp_q_now = skill_core.current_tcp_pose()
                verification1, ok1, w2 = _v303_motion(
                    f"v271_slot_entry({obj},{slot_name})",
                    lambda: skill_core.move_to_pose(
                        target_tcp, tcp_q0,
                        velocity_scale=0.10, accel_scale=0.10,
                        orientation_tolerance=0.10, position_tolerance=0.008,
                    ),
                    target_tcp)
                warnings.extend(w2)
                layer303 = "curobo_pose" if ok1 else None
            # L3
            if not ok1 and _v303_still_safe():
                objh, _objqh = skill_core.object_pose(obj)
                tcph, tcpqh = skill_core.current_tcp_pose()
                relh = [float(objh[i]) - float(tcph[i]) for i in range(3)]
                rloc303 = _v27_quat_multiply(_v27_quat_multiply(_v271_quat_conjugate(tcpqh), [relh[0], relh[1], relh[2], 0.0]), tcpqh)[:3]
                live303 = skill_core._current_joint_state_copy()
                hover_z303 = max(float(objh[2]), float(desired_obj[2]) + 0.04)
                tries303 = 0
                for yaw303 in (30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 180.0):
                    if tries303 >= 3 or not _v303_still_safe():
                        break
                    h303 = 0.5 * math.radians(yaw303)
                    qy303 = _v27_quat_normalize(_v27_quat_multiply([0.0, 0.0, math.sin(h303), math.cos(h303)], tcpqh))
                    rw303 = _v27_quat_multiply(_v27_quat_multiply(qy303, [rloc303[0], rloc303[1], rloc303[2], 0.0]), _v271_quat_conjugate(qy303))[:3]
                    hov303 = [float(slot[0]) - rw303[0], float(slot[1]) - rw303[1], hover_z303 - rw303[2]]
                    ent303 = [float(slot[0]) - rw303[0], float(slot[1]) - rw303[1], float(desired_obj[2]) - rw303[2]]
                    try:
                        jh303, _eh = skill_core._compute_ik(hov303, qy303, timeout_s=0.05, avoid_collisions=True, seed_joint_state=live303)
                        je303 = None
                        if jh303 is not None:
                            je303, _ee = skill_core._compute_ik(
                                ent303, qy303, timeout_s=0.05, avoid_collisions=True,
                                seed_joint_state=skill_core._seed_with_arm_positions(live303, list(jh303)))
                    except Exception:
                        jh303, je303 = None, None
                    if jh303 is None or je303 is None or max(abs(float(je303[i]) - float(jh303[i])) for i in range(7)) > math.radians(45.0):
                        continue
                    tries303 += 1
                    skill_core.emit(f"[V3.03 SLOT ENTRY YAW RETRY] {obj}->{slot_name} | yaw{yaw303:+.0f} at hover z={hover_z303*1000:.0f}mm")
                    vh303, okh303, wh303 = _v303_motion(
                        f"v303_hover_yaw{yaw303:+.0f}({obj},{slot_name})",
                        lambda hov303=hov303, qy303=qy303: skill_core.move_to_pose(
                            hov303, qy303, velocity_scale=0.10, accel_scale=0.10,
                            orientation_tolerance=0.08, position_tolerance=0.008),
                        hov303)
                    warnings.extend(wh303)
                    if not okh303 or not _v303_still_safe():
                        continue
                    _ob, _obq = skill_core.object_pose(obj)
                    if _v271_axis_line_error_deg(_v27_egg_major_axis_from_quat(_obq)["world_axis"]) > V271_AXIS_TOL_DEG:
                        attempts303.append(f"yaw{yaw303:+.0f}: axis degraded at hover")
                        continue
                    tq303, tqq303 = skill_core.current_tcp_pose()
                    ob303, _ = skill_core.object_pose(obj)
                    rel303 = [float(ob303[i]) - float(tq303[i]) for i in range(3)]
                    tgt303 = [desired_obj[i] - rel303[i] for i in range(3)]
                    verification1, ok1, w3 = _v303_motion(
                        f"v303_slot_entry_cartesian_yaw{yaw303:+.0f}({obj},{slot_name})",
                        lambda tgt303=tgt303, tqq303=tqq303: _v303_cart_descent(tgt303, tqq303), tgt303)
                    warnings.extend(w3)
                    if ok1:
                        layer303 = f"yaw{yaw303:+.0f}+cartesian"
                        target_tcp = tgt303
                        break
            if ok1:
                skill_core.emit(f"[V3.03 SLOT ENTRY DESCENT OK] {obj}->{slot_name} | layer={layer303}")
            if (not ok1) or (not bool(hold_active_shared.value)):
                verification1 = dict(verification1 or {})
                verification1["hold_active"] = bool(hold_active_shared.value)
                verification1["v303_attempts"] = attempts303[-6:]
                held303 = _v303_still_safe()
                if held303:
                    # Egg is still held, vertical, above the holder: make a slot
                    # change possible (plan_placement requires LIFTED).
                    verification1["reason"] = "SLOT_ENTRY_UNREACHABLE"
                    _v24_set_state(V24_STAGE_LIFTED, active_object=obj)
                    V271_PLACEMENT_RUNTIME.get("oriented", {}).pop(obj, None)
                    warnings.append(
                        f"{slot_name} entry depth is not reachable with this grasp. Egg is still held "
                        f"(stage reset to LIFTED). Call plan_placement ONCE with a different empty slot."
                    )
                    nxt303 = ["plan_placement(slot_name=<different empty slot>)"]
                else:
                    verification1.setdefault("reason", "SLOT_ENTRY_TRANSPORT_FAILED")
                    nxt303 = ["observe_scene"]
                return _v24_result(
                    name, False, contract, state_before, verification1, warnings,
                    {"elapsed_s": round(time.perf_counter()-t0, 3)},
                    next_stage=None,
                    recommended_next_actions=nxt303,
                )

            # One deterministic micro-correction is allowed; do not make the LLM
            # oscillate around the slot mouth.
            correction_used = False
            obj1, obj_q1 = skill_core.object_pose(obj)
            tcp1, tcp_q1 = skill_core.current_tcp_pose()
            xy1 = _v24_xy_distance(obj1, slot)
            if V271_SLOT_XY_TOL_M < xy1 <= 0.025:
                correction_used = True
                rel1 = [float(obj1[i]) - float(tcp1[i]) for i in range(3)]
                desired_obj2 = [float(slot[0]), float(slot[1]), float(plan["entry_object_center_z_m"])]
                target_tcp2 = [desired_obj2[i] - rel1[i] for i in range(3)]
                verification2, ok2, warn2 = _v24_run_motion(
                    f"v271_slot_entry_correction({obj},{slot_name})",
                    lambda: skill_core.move_to_pose(
                        target_tcp2, tcp_q1,
                        velocity_scale=0.06,
                        accel_scale=0.06,
                        orientation_tolerance=0.10,
                        position_tolerance=0.006,
                    ),
                    target_tcp2, 0.010, True,
                )
                warnings.extend(warn2)
                if (not ok2) or (not bool(hold_active_shared.value)):
                    verification2["hold_active"] = bool(hold_active_shared.value)
                    verification2["correction_used"] = True
                    verification2.setdefault("reason", "SLOT_ENTRY_CORRECTION_FAILED")
                    return _v24_result(
                        name, False, contract, state_before, verification2, warnings,
                        {"elapsed_s": round(time.perf_counter()-t0, 3)},
                        next_stage=None,
                        recommended_next_actions=["observe_scene"],
                    )

            obj2, obj_q2 = skill_core.object_pose(obj)
            tcp2, _ = skill_core.current_tcp_pose()
            xy_error = _v24_xy_distance(obj2, slot)
            axis2 = _v27_egg_major_axis_from_quat(obj_q2)["world_axis"]
            axis_error = _v271_axis_line_error_deg(axis2)
            rel_dist0 = _v24_xyz_distance(obj0, tcp0)
            rel_dist2 = _v24_xyz_distance(obj2, tcp2)
            rel_drift = abs(rel_dist2 - rel_dist0)
            ok = bool(
                xy_error <= V271_SLOT_XY_TOL_M
                and axis_error <= V271_AXIS_TOL_DEG
                and bool(hold_active_shared.value)
                and rel_drift <= V271_RELATIVE_DISTANCE_DRIFT_TOL_M
            )
            verification = {
                "slot_name": slot_name,
                "target_slot_xy_m": [float(slot[0]), float(slot[1])],
                "object_xyz_m": [round(float(v),5) for v in obj2],
                "object_xy_error_mm": round(xy_error*1000.0, 2),
                "xy_tolerance_mm": round(V271_SLOT_XY_TOL_M*1000.0, 1),
                "axis_error_deg": round(axis_error, 2),
                "axis_tolerance_deg": V271_AXIS_TOL_DEG,
                "correction_used": bool(correction_used),
                "object_tcp_distance_drift_mm": round(rel_drift*1000.0, 2),
                "hold_active": bool(hold_active_shared.value),
            }
            if not ok:
                verification["reason"] = "SLOT_ENTRY_ALIGNMENT_NOT_PRECISE_ENOUGH"
                return _v24_result(
                    name, False, contract, state_before, verification, warnings,
                    {"elapsed_s": round(time.perf_counter()-t0, 3)},
                    next_stage=None,
                    recommended_next_actions=["observe_scene", "plan_placement"],
                )

            # V2-32 SLOT-ENTRY CONSISTENCY GATE
            # ------------------------------------------------------------
            # guarded_insert_and_release requires <=8 deg. Therefore
            # move_to_slot_entry may no longer report SUCCESS at 8..10 deg.
            # Do not duplicate reorientation code here: route that narrow band
            # back to the existing V2-24 sparse/bounded wrist correction, which
            # already prefers single distal joints (e.g. J7) before whole-arm.
            slot32 = _v271_slot(slot_name)
            obj32, objq32 = skill_core.object_pose(obj)
            axis32 = _v271_axis_line_error_deg(
                _v27_egg_major_axis_from_quat(objq32)["world_axis"]
            )
            xy32 = _v24_xy_distance(obj32, slot32) if slot32 is not None else 1e9
            safe32 = bool(
                slot32 is not None
                and xy32 <= 0.008
                and bool(hold_active_shared.value)
                and int(safety_trip_count.value) == safety0
            )
            if axis32 > 8.0:
                verification32 = {
                    "reason": "V232_SLOT_ENTRY_NEEDS_WRIST_CORRECTION" if axis32 <= V271_AXIS_TOL_DEG and safe32 else "V232_SLOT_ENTRY_8DEG_GATE_FAILED",
                    "slot_name": slot_name,
                    "object_xy_error_mm": round(float(xy32) * 1000.0, 2),
                    "xy_limit_mm": 8.0,
                    "axis_error_deg": round(float(axis32), 2),
                    "axis_success_limit_deg": 8.0,
                    "legacy_axis_limit_deg": float(V271_AXIS_TOL_DEG),
                    "hold_active": bool(hold_active_shared.value),
                    "safety_trip_before": safety0,
                    "safety_trip_after": int(safety_trip_count.value),
                }
                if axis32 <= V271_AXIS_TOL_DEG and safe32:
                    skill_core.emit(
                        f"[V2-32 SLOT ENTRY WRIST CORRECTION REQUIRED] {obj}->{slot_name} | "
                        f"axis={axis32:.2f}deg in (8,{V271_AXIS_TOL_DEG:.1f}] | "
                        f"xy={xy32*1000.0:.2f}mm | hold=true | route=orient_for_slot"
                    )
                    return _v24_result(
                        name, False, contract, state_before, verification32, [],
                        {"elapsed_s": round(time.perf_counter()-t0, 3)},
                        next_stage=None,
                        recommended_next_actions=["orient_for_slot"],
                    )
                skill_core.emit(
                    f"[V2-32 SLOT ENTRY BLOCK] {obj}->{slot_name} | "
                    f"axis={axis32:.2f}/8.00deg | xy={xy32*1000.0:.2f}/8.00mm | "
                    f"hold={bool(hold_active_shared.value)}"
                )
                return _v24_result(
                    name, False, contract, state_before, verification32, [],
                    {"elapsed_s": round(time.perf_counter()-t0, 3)},
                    next_stage=None,
                    recommended_next_actions=["observe_scene"],
                )

            V271_PLACEMENT_RUNTIME["slot_entry"][obj] = {
                "slot_name": slot_name,
                "xy_error_mm": round(xy_error*1000.0,2),
                "axis_error_deg": round(axis_error,2),
            }
            skill_core.emit(
                f"[V2-7.2 SLOT ENTRY OK] {obj}->{slot_name} | xy_error={xy_error*1000.0:.1f}mm | "
                f"axis_error={axis_error:.1f}deg | correction={correction_used}"
            )
            return _v24_result(
                name, True, contract, state_before, verification, warnings,
                {"elapsed_s": round(time.perf_counter()-t0, 3)},
                "SLOT_ENTRY_ALIGNED", obj, f"egg_holder.{slot_name}",
                recommended_next_actions=["guarded_insert_and_release"],
            )

        if name == "guarded_insert_and_release":
            obj = str(args.get("object_name", state_before.get("active_object") or ""))
            slot_name = str(args.get("slot_name", ""))
            plan = V271_PLACEMENT_RUNTIME.get("plans", {}).get(obj)
            slot = _v271_slot(slot_name)
            if not plan or plan.get("slot_name") != slot_name or slot is None:
                return _v24_precondition_failure(name, args, "no matching placement plan", contract, state_before, recommended_next_actions=["plan_placement"])
            if state_before.get("stage") != "SLOT_ENTRY_ALIGNED" or state_before.get("active_object") != obj:
                return _v24_precondition_failure(
                    name, args, f"requires SLOT_ENTRY_ALIGNED; current={state_before}",
                    contract, state_before, recommended_next_actions=["move_to_slot_entry"],
                )
            if not bool(hold_active_shared.value):
                return _v24_precondition_failure(name, args, "hold_active=false before insertion", contract, state_before, recommended_next_actions=["observe_scene"])

            # V2-30 HARD INSERTION ENTRY GATE.
            # Do not start descending merely because move_to_slot_entry used the
            # older V2-24/V2-29 10-degree acceptance rule.  Re-read the physical
            # scene now and require explicit margin before entering the holder.
            obj_entry30, obj_q_entry30 = skill_core.object_pose(obj)
            entry_xy30 = _v24_xy_distance(obj_entry30, slot)
            entry_axis30 = _v271_axis_line_error_deg(
                _v27_egg_major_axis_from_quat(obj_q_entry30)["world_axis"]
            )
            entry_hold30 = bool(hold_active_shared.value)
            if (not entry_hold30) or entry_axis30 > 8.0 or entry_xy30 > 0.008:
                verification30 = {
                    "reason": "V230_INSERT_ENTRY_GATE_FAILED",
                    "slot_name": slot_name,
                    "entry_axis_error_deg": round(float(entry_axis30), 2),
                    "entry_axis_limit_deg": 8.0,
                    "entry_xy_error_mm": round(float(entry_xy30)*1000.0, 2),
                    "entry_xy_limit_mm": 8.0,
                    "hold_active": entry_hold30,
                    "release_performed": False,
                }
                skill_core.emit(
                    f"[V2-30 INSERT ENTRY BLOCK] {obj}->{slot_name} | "
                    f"axis={entry_axis30:.2f}/8.00deg | xy={entry_xy30*1000.0:.2f}/8.00mm | "
                    f"hold={entry_hold30}"
                )
                return _v24_result(
                    name, False, contract, state_before, verification30, [],
                    {"elapsed_s": round(time.perf_counter()-t0, 3)},
                    next_stage=None,
                    recommended_next_actions=["observe_scene", "plan_placement"],
                )

            llm_require_arm_controller()
            safety0 = int(safety_trip_count.value)
            release_z = float(plan["pre_release_object_center_z_m"])
            records = []
            warnings = []
            insertion_ok = True
            abort_reason = None
            step_index = 0

            while True:
                obj_now, obj_q_now = skill_core.object_pose(obj)
                tcp_now, tcp_q_now = skill_core.current_tcp_pose()
                xy_error = _v24_xy_distance(obj_now, slot)
                axis_now = _v27_egg_major_axis_from_quat(obj_q_now)["world_axis"]
                axis_error = _v271_axis_line_error_deg(axis_now)
                remaining = float(obj_now[2]) - release_z

                reason335 = (_v335_slot_moved_now(obj, slot_name, slot)
                             if (v326_wrist_selected() and _v329_no_prior()) else None)
                if reason335:
                    # V3.35: the live camera says the carton moved (or the egg turned in the fingers) - stop the
                    # descent and lift back to the slot-entry height; nothing is released
                    insertion_ok = False
                    abort_reason = reason335
                    skill_core.emit(f"[V3.35 INSERT STOP] {obj}->{slot_name} | {reason335} | lifting back, egg kept")
                    try:
                        entry_z335 = float(plan.get("entry_object_center_z_m", float(obj_now[2]) + 0.010))
                        dz335 = max(0.0, entry_z335 - float(obj_now[2]))
                        if dz335 > 0.002 and int(safety_trip_count.value) == safety0:
                            tcp_r, q_r = skill_core.current_tcp_pose()
                            tgt_r = [float(tcp_r[0]), float(tcp_r[1]), float(tcp_r[2]) + dz335]

                            def _do_r335(tgt_r=tgt_r, q_r=q_r):
                                tr = skill_core._cartesian_approach_plan(tgt_r, q_r)
                                _v218_slow_trajectory_in_place(tr, factor=2.0)
                                skill_core._execute_trajectory(tr)
                                return True
                            _v24_run_motion(f"v335_insert_stop_retreat({obj},{slot_name})", _do_r335, tgt_r, 0.010, True)
                    except Exception as exc335:
                        warnings.append(f"V3.35 retreat after insert stop raised: {exc335}")
                    break

                if not bool(hold_active_shared.value):
                    insertion_ok = False
                    abort_reason = "GRASP_LOST_DURING_INSERTION"
                    break
                if int(safety_trip_count.value) != safety0:
                    insertion_ok = False
                    abort_reason = "SAFETY_TRIP_CHANGED_DURING_INSERTION"
                    break
                if xy_error > V271_SLOT_XY_ABORT_M:
                    insertion_ok = False
                    abort_reason = "LATERAL_SLOT_ERROR_DURING_INSERTION"
                    break
                if axis_error > V271_INSERT_AXIS_ABORT_DEG:
                    insertion_ok = False
                    abort_reason = "EGG_LONG_AXIS_DRIFT_DURING_INSERTION"
                    break
                if remaining <= 0.0025:
                    break

                # V2-30: 5 mm coarse / 2.5 mm fine insertion increments.
                dz = 0.005 if remaining > 0.012 else 0.0025
                target_obj_z = max(release_z, float(obj_now[2]) - dz)
                rel = [float(obj_now[i]) - float(tcp_now[i]) for i in range(3)]
                # Small deterministic XY correction keeps the egg centered while
                # the only intentional insertion direction is the slot/egg axis.
                target_obj = [float(slot[0]), float(slot[1]), target_obj_z]
                target_tcp = [target_obj[i] - rel[i] for i in range(3)]

                blk341 = {"blocked": False, "why": "", "fallback": False}

                def _do_insert_step(target_tcp=target_tcp, tcp_q_now=tcp_q_now, blk341=blk341):
                    try:
                        traj = skill_core._cartesian_approach_plan(target_tcp, tcp_q_now)
                    except Exception as exc341:
                        if ("CARTESIAN APPROACH BLOCKED" not in str(exc341)
                                or os.environ.get("NERO_V341_INSERT_FALLBACK", "1") == "0"):
                            raise
                        # V3.41: the straight descent is blocked (V3.40 log: 12 steps, 0 mm, reported as success).
                        # Same TCP orientation, one 5 mm level: collision-aware BioIK seeded here -> MoveIt joint goal.
                        blk341["why"] = str(exc341)[:120]
                        traj, why341 = _v341_insert_ik_step(target_tcp, tcp_q_now)
                        if traj is None:
                            blk341["blocked"] = True
                            blk341["why"] += f" | BioIK step: {why341}"
                            return False
                        blk341["fallback"] = True
                        skill_core.emit(f"[V3.41 INSERT IK STEP] {obj}->{slot_name} | straight descent blocked -> "
                                        f"BioIK + MoveIt joint goal ({why341})")
                    # V2-30: insertion is a contact-sensitive phase.  Stretch
                    # timing by 2x (~50% speed) while leaving geometry unchanged.
                    _v218_slow_trajectory_in_place(traj, factor=2.0)
                    skill_core._execute_trajectory(traj)
                    return True

                z_before341 = float(obj_now[2])
                verification_s, ok_s, warn_s = _v24_run_motion(
                    f"v271_insert_step_{step_index+1}({obj},{slot_name})",
                    _do_insert_step,
                    target_tcp, 0.008, True,
                )
                if not blk341["blocked"] and os.environ.get("NERO_V341_INSERT_FALLBACK", "1") != "0":
                    try:
                        _ob341, _q341 = skill_core.object_pose(obj)
                        if z_before341 - float(_ob341[2]) < 0.4 * dz:
                            blk341["blocked"] = True
                            blk341["why"] = (f"no descent: {1000 * (z_before341 - float(_ob341[2])):.1f} mm of "
                                             f"{1000 * dz:.1f} mm")
                    except Exception:
                        pass
                if blk341["blocked"]:
                    insertion_ok = False
                    abort_reason = "INSERT_BLOCKED"
                    m341 = v341_limit_margins_deg(skill_core._arm_positions_from_joint_state(
                        skill_core._current_joint_state_copy()))
                    records.append({"step": step_index + 1, "blocked": True, "why": blk341["why"],
                                    "min_joint_limit_margin_deg": m341[0], "joint": f"J{m341[1]}",
                                    "joint_limit_margins_deg": m341[2], "object_center_z_mm": round(1000 * z_before341, 1)})
                    skill_core.emit(f"[V3.41 INSERT BLOCKED] {obj}->{slot_name} | step {step_index + 1} | {blk341['why']} | "
                                    f"J{m341[1]} {m341[0]:.1f} deg from its limit")
                    break
                warnings.extend(warn_s)
                step_index += 1
                obj_after, obj_q_after = skill_core.object_pose(obj)
                xy_after = _v24_xy_distance(obj_after, slot)
                axis_after = _v271_axis_line_error_deg(_v27_egg_major_axis_from_quat(obj_q_after)["world_axis"])
                # V2-30 post-segment physical guards.  target_shared / actual_shared
                # are the same physical joint-state bridge used by the runtime.
                try:
                    track_max_deg30 = max(
                        abs(float(target_shared[i]) - float(actual_shared[i]))
                        for i in range(7)
                    ) * 180.0 / math.pi
                except Exception:
                    track_max_deg30 = 0.0
                records.append({
                    "step": step_index,
                    "requested_object_center_z_mm": round(target_obj_z*1000.0, 1),
                    "actual_object_center_z_mm": round(float(obj_after[2])*1000.0, 1),
                    "xy_error_mm": round(xy_after*1000.0, 2),
                    "axis_error_deg": round(axis_after, 2),
                    "joint_tracking_max_error_deg": round(float(track_max_deg30), 2),
                    "hold_active": bool(hold_active_shared.value),
                    "cartesian_ok": bool(ok_s),
                    "safety_trip_count": int(safety_trip_count.value),
                })
                skill_core.emit(
                    f"[V2-30 INSERT STEP] {obj}->{slot_name} | step={step_index} | "
                    f"z={float(obj_after[2])*1000.0:.1f}mm | xy={xy_after*1000.0:.1f}mm | "
                    f"axis={axis_after:.1f}deg | track={track_max_deg30:.2f}deg | "
                    f"hold={bool(hold_active_shared.value)}"
                )

                guard_reason30 = None
                # V2-32: 3..5 deg tracking residual is warning-only while task geometry is healthy.
                if 3.0 < float(track_max_deg30) <= 5.0 and (axis_after <= 8.0 and xy_after <= 0.008 and bool(hold_active_shared.value) and int(safety_trip_count.value) == safety0):
                    warnings.append(f"V2-32 insertion tracking WARN: {track_max_deg30:.2f}deg with geometry/hold/safety healthy")
                    skill_core.emit(
                        f"[V2-32 INSERT TRACK WARN] {obj}->{slot_name} | "
                        f"track={track_max_deg30:.2f}deg | axis={axis_after:.2f}deg | "
                        f"xy={xy_after*1000.0:.2f}mm | hold=true | continue"
                    )
                    skill_core.emit(
                        f"[V2-33 TRACK WARN TRUE CONTINUE] {obj}->{slot_name} | "
                        f"healthy 3-5deg residual bypasses legacy V2-30 abort"
                    )
                    continue
                if float(track_max_deg30) > 5.0 or (float(track_max_deg30) > 3.0 and not (axis_after <= 8.0 and xy_after <= 0.008 and bool(hold_active_shared.value) and int(safety_trip_count.value) == safety0)):
                    guard_reason30 = verification_s.get("reason", "INSERT_STEP_MOTION_FAILED")
                elif int(safety_trip_count.value) != safety0:
                    guard_reason30 = "V230_SAFETY_TRIP_AFTER_INSERT_STEP"
                elif not bool(hold_active_shared.value):
                    guard_reason30 = "V230_GRASP_LOST_AFTER_INSERT_STEP"
                elif axis_after > 8.0:
                    guard_reason30 = "V230_AXIS_GUARD_AFTER_INSERT_STEP"
                elif xy_after > 0.008:
                    guard_reason30 = "V230_XY_GUARD_AFTER_INSERT_STEP"
                elif track_max_deg30 > 3.0:
                    guard_reason30 = "V230_TRACKING_GUARD_AFTER_INSERT_STEP"

                if guard_reason30 is not None:
                    insertion_ok = False
                    abort_reason = guard_reason30
                    skill_core.emit(
                        f"[V2-30 INSERT GUARD STOP] {obj}->{slot_name} | reason={guard_reason30} | "
                        f"axis={axis_after:.2f}deg | xy={xy_after*1000.0:.2f}mm | "
                        f"track={track_max_deg30:.2f}deg | hold={bool(hold_active_shared.value)}"
                    )

                    # If the safety latch is still clean and the object is still held,
                    # withdraw vertically to the slot-entry object-center height before
                    # handing control back.  On a safety trip / grasp loss, do not issue
                    # further motion commands.
                    if int(safety_trip_count.value) == safety0 and bool(hold_active_shared.value):
                        try:
                            entry_z30 = float(plan.get("entry_object_center_z_m", float(obj_after[2]) + 0.010))
                            retreat_dz30 = max(0.0, entry_z30 - float(obj_after[2]))
                            if retreat_dz30 > 0.002:
                                tcp_ret30, q_ret30 = skill_core.current_tcp_pose()
                                target_ret30 = [float(tcp_ret30[0]), float(tcp_ret30[1]), float(tcp_ret30[2]) + retreat_dz30]
                                def _do_guard_retreat30(target_ret30=target_ret30, q_ret30=q_ret30):
                                    traj30 = skill_core._cartesian_approach_plan(target_ret30, q_ret30)
                                    _v218_slow_trajectory_in_place(traj30, factor=2.0)
                                    skill_core._execute_trajectory(traj30)
                                    return True
                                _vg30, retreat_ok30, _wg30 = _v24_run_motion(
                                    f"v230_insert_guard_retreat({obj},{slot_name})",
                                    _do_guard_retreat30, target_ret30, 0.010, True,
                                )
                                records[-1]["guard_retreat_attempted"] = True
                                records[-1]["guard_retreat_ok"] = bool(retreat_ok30)
                                skill_core.emit(
                                    f"[V2-30 INSERT GUARD RETREAT] {obj}->{slot_name} | "
                                    f"dz={retreat_dz30*1000.0:.1f}mm | ok={bool(retreat_ok30)}"
                                )
                        except Exception as exc30:
                            records[-1]["guard_retreat_attempted"] = True
                            records[-1]["guard_retreat_ok"] = False
                            records[-1]["guard_retreat_error"] = str(exc30)
                            warnings.append(f"V2-30 insertion guard retreat raised: {exc30}")
                    break

                time.sleep(0.10)
                if step_index >= 12:
                    insertion_ok = False
                    abort_reason = "INSERT_STEP_LIMIT_EXCEEDED"
                    break

            if not insertion_ok and abort_reason == "INSERT_BLOCKED":
                # V3.41: back to the entry height; once per egg+slot: re-plan the slot-above pose with another yaw
                # (orient_for_slot skips the current one), else INSERT_BLOCKED_JOINT_LIMIT
                try:
                    obj_b341, _qb341 = skill_core.object_pose(obj)
                    entry_b341 = float(plan.get("entry_object_center_z_m", float(obj_b341[2])))
                    dz_b341 = max(0.0, entry_b341 - float(obj_b341[2]))
                    if dz_b341 > 0.002 and int(safety_trip_count.value) == safety0 and bool(hold_active_shared.value):
                        tcp_b341, q_b341 = skill_core.current_tcp_pose()
                        tgt_b341 = [float(tcp_b341[0]), float(tcp_b341[1]), float(tcp_b341[2]) + dz_b341]

                        def _do_b341(tgt_b341=tgt_b341, q_b341=q_b341):
                            trb = skill_core._cartesian_approach_plan(tgt_b341, q_b341)
                            _v218_slow_trajectory_in_place(trb, factor=2.0)
                            skill_core._execute_trajectory(trb)
                            return True
                        _v24_run_motion(f"v341_insert_block_retreat({obj},{slot_name})", _do_b341, tgt_b341, 0.010, True)
                except Exception as exc_b341:
                    warnings.append(f"V3.41 retreat after insert block raised: {exc_b341}")
                key341 = f"{obj}|{slot_name}"
                n341 = int(V341_STATE["insert_replans"].get(key341, 0))
                last341 = records[-1] if records else {}
                held341 = bool(hold_active_shared.value)
                if n341 < int(os.environ.get("NERO_V341_INSERT_REPLANS", "1")) and held341:
                    V341_STATE["insert_replans"][key341] = n341 + 1
                    V341_STATE["skip_yaw"][str(obj)] = str(slot_name)
                    v24_skill_state["stage"] = V24_STAGE_LIFTED
                    reason341, next341 = "INSERT_BLOCKED_REPLAN_YAW", ["plan_placement", "orient_for_slot"]
                    hint341 = ("The descent into the slot is blocked at this wrist posture; the egg is still held at the "
                               "slot entry. Call plan_placement for the SAME egg and slot, then orient_for_slot: it re-plans "
                               "the slot-above pose with another yaw (the current one is skipped), then move_to_slot_entry "
                               "and guarded_insert_and_release again.")
                else:
                    reason341, next341 = "INSERT_BLOCKED_JOINT_LIMIT", ["return_to_pick_location"]
                    hint341 = ("The egg cannot be lowered into this slot from any reachable wrist posture. Call "
                               "return_to_pick_location, then continue with the other eggs (or try another slot later).")
                verification = {
                    "reason": reason341, "slot_name": slot_name, "insert_steps": records, "release_performed": False,
                    "hold_active": held341, "blocked_detail": last341.get("why"),
                    "min_joint_limit_margin_deg": last341.get("min_joint_limit_margin_deg"),
                    "joint_at_limit": last341.get("joint"), "joint_limit_margins_deg": last341.get("joint_limit_margins_deg"),
                    "required_margin_deg": v341_margin_required_deg(), "replans_used": int(V341_STATE["insert_replans"].get(key341, 0)),
                    "safety_trip_before": safety0, "safety_trip_after": int(safety_trip_count.value),
                }
                res341 = _v24_result(
                    name, False, contract, state_before, verification, warnings,
                    {"elapsed_s": round(time.perf_counter()-t0, 3), "hint": hint341},
                    next_stage=None, recommended_next_actions=next341,
                )
                if reason341 == "INSERT_BLOCKED_REPLAN_YAW":
                    v24_skill_state["stage"] = V24_STAGE_LIFTED
                return res341

            if not insertion_ok:
                if str(abort_reason).startswith(("SLOT_MOVED", "HOLDER_POSSIBLY")) and bool(hold_active_shared.value):
                    v24_skill_state["stage"] = V24_STAGE_LIFTED      # V3.35: egg still held -> plan the placement again
                verification = {
                    "reason": abort_reason,
                    "slot_name": slot_name,
                    "insert_steps": records,
                    "release_performed": False,
                    "hold_active": bool(hold_active_shared.value),
                    "safety_trip_before": safety0,
                    "safety_trip_after": int(safety_trip_count.value),
                }
                return _v24_result(
                    name, False, contract, state_before, verification, warnings,
                    {"elapsed_s": round(time.perf_counter()-t0, 3)},
                    next_stage=None,
                    recommended_next_actions=(["plan_placement"] if str(abort_reason).startswith(("SLOT_MOVED", "HOLDER_POSSIBLY"))
                                              else ["observe_scene"]),
                )

            # Capture-depth reached.  Stop first, then release in two stages.
            time.sleep(0.25)
            obj_capture, obj_q_capture = skill_core.object_pose(obj)
            xy_capture = _v24_xy_distance(obj_capture, slot)
            axis_capture = _v271_axis_line_error_deg(_v27_egg_major_axis_from_quat(obj_q_capture)["world_axis"])
            if xy_capture > V271_SLOT_XY_ABORT_M or axis_capture > V271_INSERT_AXIS_ABORT_DEG:
                verification = {
                    "reason": "CAPTURE_DEPTH_ALIGNMENT_DEGRADED",
                    "object_xy_error_mm": round(xy_capture*1000.0,2),
                    "axis_error_deg": round(axis_capture,2),
                    "insert_steps": records,
                    "release_performed": False,
                }
                return _v24_result(
                    name, False, contract, state_before, verification, warnings,
                    {"elapsed_s": round(time.perf_counter()-t0,3)},
                    next_stage=None,
                    recommended_next_actions=["observe_scene"],
                )

            opening0 = float(gripper_actual_mm.value)
            gripper_force_mode.value = False
            gripper_target_mm.value = min(100.0, opening0 + V271_RELEASE_STAGE_OPEN_MM)
            deadline = time.time() + 0.65
            while time.time() < deadline:
                if float(gripper_actual_mm.value) >= min(100.0, opening0 + 7.0):
                    break
                time.sleep(0.04)
            time.sleep(0.25)

            # At capture depth, only loosen enough to detach the egg.  Leaving
            # the fingers modestly open keeps their lateral swept volume small
            # while they are still close to the holder posts.
            hold_after_release = bool(hold_active_shared.value)
            opening_after_stage1 = float(gripper_actual_mm.value)
            obj_released, _ = skill_core.object_pose(obj)
            preliminary_location = llm_classify_object_location(obj_released)

            # Retreat the partially-open gripper straight upward BEFORE asking
            # for a wide opening.  This is the low-clearance phase.
            retreat_ok = False
            retreat_error = None
            try:
                tcp_r0, q_r0 = skill_core.current_tcp_pose()
                target_r = [float(tcp_r0[0]), float(tcp_r0[1]), float(tcp_r0[2]) + V271_RETREAT_M]
                def _do_retreat():
                    traj = skill_core._cartesian_approach_plan(target_r, q_r0)
                    skill_core._execute_trajectory(traj)
                    return True
                _vr, retreat_ok, _wr = _v24_run_motion(
                    f"v271_post_release_retreat({obj},{slot_name})",
                    _do_retreat,
                    target_r, 0.012, False,
                )
                warnings.extend(_wr)
            except Exception as _exc:
                retreat_error = str(_exc)
                warnings.append(f"Post-release vertical retreat raised: {_exc}")

            # Wide opening is safe only after the fingers have cleared the cup.
            if retreat_ok:
                gripper_target_mm.value = V271_RELEASE_FULL_OPEN_MM
                deadline = time.time() + 1.20
                while time.time() < deadline:
                    if float(gripper_actual_mm.value) >= 80.0:
                        break
                    time.sleep(0.04)
            time.sleep(0.35)
            opening_after = float(gripper_actual_mm.value)
            obj_final, _ = skill_core.object_pose(obj)
            final_location = llm_classify_object_location(obj_final)
            stage1_opened_enough = opening_after_stage1 >= min(100.0, opening0 + 5.0)
            release_ok = bool((not hold_after_release) and stage1_opened_enough and retreat_ok)
            verification = {
                "slot_name": slot_name,
                "insert_steps": records,
                "pre_release_object_center_z_mm": round(float(obj_capture[2])*1000.0, 2),
                "planned_pre_release_center_z_mm": round(release_z*1000.0, 2),
                "capture_xy_error_mm": round(xy_capture*1000.0, 2),
                "capture_axis_error_deg": round(axis_capture, 2),
                "opening_before_release_mm": round(opening0, 2),
                "opening_after_stage1_mm": round(opening_after_stage1, 2),
                "opening_after_release_mm": round(opening_after, 2),
                "wide_open_command_after_retreat": bool(retreat_ok),
                "hold_active_after_release": hold_after_release,
                "preliminary_location_after_release": preliminary_location,
                "location_after_retreat_settle": final_location,
                "post_release_retreat_ok": bool(retreat_ok),
                "post_release_retreat_error": retreat_error,
                "release_mechanically_ok": release_ok,
                "final_verification_required": True,
            }
            if not release_ok:
                verification["reason"] = "RELEASE_DID_NOT_COMPLETE"
                return _v24_result(
                    name, False, contract, state_before, verification, warnings,
                    {"elapsed_s": round(time.perf_counter()-t0,3)},
                    next_stage=None,
                    recommended_next_actions=["observe_scene"],
                )

            V271_PLACEMENT_RUNTIME["released"][obj] = {
                "slot_name": slot_name,
                "preliminary_location": final_location,
            }
            skill_core.emit(
                f"[V2-7.2 RELEASE COMPLETE] {obj}->{slot_name} | capture_z={float(obj_capture[2])*1000.0:.1f}mm | "
                f"open={opening_after:.1f}mm | hold=false | location={final_location} | verify next"
            )
            return _v24_result(
                name, True, contract, state_before, verification, warnings,
                {"slot_name": slot_name, "elapsed_s": round(time.perf_counter()-t0,3)},
                V24_STAGE_RELEASED, obj, f"egg_holder.{slot_name}",
                recommended_next_actions=["verify_object_in_slot"],
            )

        if name == "move_above":
            stage = state_before.get("stage")
            if stage not in (V24_STAGE_OBSERVED, V24_STAGE_VERIFIED, V24_STAGE_RELEASED):
                return _v24_precondition_failure(name, args, f"move_above not allowed from state={stage}", contract, state_before)
            if bool(hold_active_shared.value):
                return _v24_precondition_failure(name, args, "hold_active=true; release/resolve current object before selecting another", contract, state_before)
            obj = str(args["object_name"])
            aperture = _v24_aperture_check(obj)
            if not aperture["robust_aperture_ok"]:
                reason = (
                    f"gripper aperture {aperture['actual_opening_mm']:.1f}mm is {aperture['aperture_status']} for {obj}; "
                    f"effective width={aperture['effective_width_mm']:.1f}mm, "
                    f"recommended range={aperture['recommended_opening_range_mm']}mm, "
                    f"recommended target={aperture['recommended_target_opening_mm']:.1f}mm"
                )
                return _v24_precondition_failure(
                    name, args, reason, contract, state_before,
                    physical_verification={"gripper_aperture": aperture},
                    recommended_next_actions=["set_gripper_opening", "observe_scene", "move_above"],
                )
            llm_require_arm_controller()
            obj_before, _ = skill_core.object_pose(obj)
            target = [float(obj_before[0]), float(obj_before[1]), float(obj_before[2]) + 0.12]
            verification, cart_ok, warnings = _v24_run_motion(
                f"move_above({obj})", lambda: skill_core.bio_move_above(obj, 0.12), target, 0.012, True
            )
            obj_after, _ = skill_core.object_pose(obj)
            object_move = _v24_xyz_distance(obj_after, obj_before)
            verification.update({
                "object_xyz_before_m": [round(float(v),5) for v in obj_before],
                "object_xyz_after_m": [round(float(v),5) for v in obj_after],
                "object_motion_mm": round(object_move*1000.0, 2),
                "object_motion_expected": False,
            })
            if object_move > 0.005:
                warnings.append(f"Object moved {object_move*1000.0:.1f} mm during hover motion; object motion is not expected at this stage.")
            ok = bool(cart_ok and object_move <= 0.020)
            if cart_ok and not ok:
                verification["reason"] = "UNEXPECTED_OBJECT_MOTION_DURING_HOVER"
            return _v24_result(name, ok, contract, state_before, verification, warnings,
                               {"object_name": obj, "elapsed_s": round(time.perf_counter()-t0,3)},
                               V24_STAGE_HOVER if ok else None, obj if ok else "__KEEP__", None if ok else "__KEEP__")

        if name == "approach":
            obj = str(args["object_name"])
            if state_before.get("stage") != V24_STAGE_HOVER or state_before.get("active_object") != obj:
                return _v24_precondition_failure(name, args, f"requires HOVER_REACHED for same object; current={state_before}", contract, state_before)
            aperture = _v24_aperture_check(obj)
            if not aperture["robust_aperture_ok"]:
                reason = (
                    f"approach blocked: current aperture {aperture['actual_opening_mm']:.1f}mm is "
                    f"{aperture['aperture_status']} for {obj}; recommended target="
                    f"{aperture['recommended_target_opening_mm']:.1f}mm"
                )
                return _v24_precondition_failure(
                    name, args, reason, contract, state_before,
                    physical_verification={"gripper_aperture": aperture},
                    recommended_next_actions=["set_gripper_opening", "observe_scene", "approach"],
                )
            llm_require_arm_controller()
            obj_before, _ = skill_core.object_pose(obj)
            opening_before = float(gripper_actual_mm.value)
            base_zoff = float(GRASP_Z_OFFSET.get(obj, DEFAULT_GRASP_Z_OFFSET))
            z_bonus = float(V251_GRASP_Z_RETRY_BONUS_M.get(obj, 0.0))
            zoff = base_zoff + z_bonus
            target = [float(obj_before[0]), float(obj_before[1]), float(obj_before[2]) + zoff]
            if z_bonus > 0.0:
                skill_core.emit(
                    f"[V2-7 ADAPTED APPROACH] {obj} | base_z_offset={base_zoff*1000.0:.1f}mm "
                    f"+ retry_bonus={z_bonus*1000.0:.1f}mm"
                )
            verification, cart_ok, warnings = _v24_run_motion(
                f"approach({obj})", lambda: skill_core.approach(obj, zoff), target, 0.010, True
            )
            obj_after, _ = skill_core.object_pose(obj)
            opening_after = float(gripper_actual_mm.value)
            object_move = _v24_xyz_distance(obj_after, obj_before)
            object_move_mm = object_move * 1000.0
            opening_change_mm = opening_after - opening_before
            verification.update({
                "object_motion_mm": round(object_move_mm, 2),
                "object_motion_required": False,
                "gripper_opening_before_approach_mm": round(opening_before, 2),
                "gripper_opening_after_approach_mm": round(opening_after, 2),
                "gripper_opening_change_during_approach_mm": round(opening_change_mm, 2),
                "anti_poke_object_motion_limit_mm": 4.0,
            })

            # Planning failure recovery (pre-contact)
            # -------------------------------------------
            # A Cartesian collision-check refusal happens BEFORE execution.
            # It is therefore not a V69 safety trip and needs no unlatch.
            # Treat it as a recoverable planning failure once: widen the aperture
            # and raise the grasp target slightly, then re-observe and retry.
            motion_error = str(verification.get("legacy_joint_verify_error", ""))
            if "CARTESIAN APPROACH BLOCKED" in motion_error:
                m = re.search(r"only\s+([0-9.]+)%\s+collision-free", motion_error)
                fraction_pct = float(m.group(1)) if m else None
                used_plan = int(V251_PLANNING_RETRY_USED.get(obj, 0))
                retry_allowed = bool(used_plan < 1 and not bool(hold_active_shared.value))
                verification.update({
                    "reason": "PREGRASP_PATH_BLOCKED_REQUIRES_ADAPTATION",
                    "collision_free_fraction_pct": fraction_pct,
                    "planning_retry_allowed": retry_allowed,
                    "planning_retry_used": used_plan,
                    "planning_retry_limit": 1,
                })
                if retry_allowed:
                    V251_PLANNING_RETRY_USED[obj] = used_plan + 1
                    old_ap_bonus = float(V241_APERTURE_RETRY_BONUS_MM.get(obj, 0.0))
                    # A modest +6 mm opening change plus +8 mm grasp-height change
                    # is materially different without jumping to a full-open hand.
                    V241_APERTURE_RETRY_BONUS_MM[obj] = min(30.0, old_ap_bonus + 6.0)
                    V251_GRASP_Z_RETRY_BONUS_M[obj] = max(float(V251_GRASP_Z_RETRY_BONUS_M.get(obj, 0.0)), 0.008)
                    adapted = _v24_gripper_prior(obj, opening_after)
                    verification.update({
                        "diagnosis": "Cartesian collision checker predicts insufficient pre-grasp clearance; no physical contact occurred",
                        "retry_recommended_opening_mm": float(adapted["recommended_target_opening_mm"]),
                        "retry_grasp_z_bonus_mm": round(V251_GRASP_Z_RETRY_BONUS_M[obj] * 1000.0, 1),
                    })
                    warnings.append(
                        f"PRE-GRASP PATH BLOCKED at {fraction_pct if fraction_pct is not None else '?'}% collision-free. "
                        f"No physical collision occurred. Adapt once: opening -> {adapted['recommended_target_opening_mm']:.1f} mm; "
                        f"grasp target +{V251_GRASP_Z_RETRY_BONUS_M[obj]*1000.0:.0f} mm higher."
                    )
                    skill_core.emit(
                        f"[V2-7 PLANNING FAILURE DIAGNOSIS] approach({obj}) | "
                        f"collision_free={fraction_pct}% | no physical contact | "
                        f"opening_target={adapted['recommended_target_opening_mm']:.1f}mm | "
                        f"grasp_z_bonus={V251_GRASP_Z_RETRY_BONUS_M[obj]*1000.0:.0f}mm | retry=1/1"
                    )
                    return _v24_result(
                        name, False, contract, state_before, verification, warnings,
                        {"object_name": obj, "elapsed_s": round(time.perf_counter()-t0,3)},
                        None,
                        recommended_next_actions=["set_gripper_opening", "observe_scene", "approach"],
                    )
                warnings.append("Second Cartesian planning block for this object; no further planning retry allowed.")
                skill_core.emit(f"[V2-7 PLANNING FINAL ABORT] approach({obj}) | retry budget exhausted")
                return _v24_result(
                    name, False, contract, state_before, verification, warnings,
                    {"object_name": obj, "elapsed_s": round(time.perf_counter()-t0,3)},
                    None, recommended_next_actions=["stop_this_object_and_explain_failure"],
                )

            # Physical safety recovery (post-execution evidence)
            # ----------------------------------------------------
            # A latched safety stop during the pre-grasp descent is
            # converted into a structured failure for the Agent to reflect on.
            # We do NOT auto-unlatch here.  The Agent must observe, explain a
            # diagnosis, choose recover_from_safety, adjust strategy, and gets
            # at most one retry for this object.
            safety_stop = bool(
                verification.get("reason") == "SAFETY_TRIP_CHANGED"
                or "PHYSICAL_SAFETY_STOP" in str(verification.get("legacy_joint_verify_error", ""))
                or "PHYSICAL_SAFETY_STOP" in str(verification.get("motion_exception", ""))
            )
            if safety_stop:
                used = int(V242_SAFETY_RETRY_USED.get(obj, 0))
                current_prior = _v24_gripper_prior(obj, opening_after)
                proposed_bonus = min(30.0, float(V241_APERTURE_RETRY_BONUS_MM.get(obj, 0.0)) + 15.0)
                # Compute the proposed target without mutating the prior yet.
                old_bonus = float(V241_APERTURE_RETRY_BONUS_MM.get(obj, 0.0))
                V241_APERTURE_RETRY_BONUS_MM[obj] = proposed_bonus
                proposed_prior = _v24_gripper_prior(obj, opening_after)
                V241_APERTURE_RETRY_BONUS_MM[obj] = old_bonus

                evidence = [
                    f"safety trip occurred during PRE-GRASP approach",
                    f"opening before={opening_before:.1f}mm, after={opening_after:.1f}mm, change={opening_change_mm:+.1f}mm",
                    f"object motion before clamp={object_move_mm:.1f}mm",
                    f"effective object width={current_prior.get('effective_width_mm')}mm",
                ]
                if abs(opening_change_mm) >= 8.0:
                    likely = "OPENING_OR_COLLISION_ENVELOPE_TOO_TIGHT / possible lateral approach offset"
                    confidence = "high"
                elif object_move_mm > 2.0:
                    likely = "LATERAL_ALIGNMENT_OR_CLEARANCE_ERROR / premature object contact"
                    confidence = "moderate"
                else:
                    likely = "UNCERTAIN_PREGRASP_CONTACT_OR_TRAJECTORY_ERROR; fresh perception and replanning required"
                    confidence = "moderate"
                retry_allowed = bool(used < 1 and not bool(hold_active_shared.value))
                ctx = _v242_set_safety_context(
                    active=True, tool="approach", object_name=obj,
                    trip_count=int(safety_trip_count.value),
                    safety_state_code=int(safety_state.value),
                    likely_cause=likely, confidence=confidence, evidence=evidence,
                    suggested_strategy="widen_aperture_and_retry_once" if retry_allowed else "stop",
                    suggested_retry_opening_mm=float(proposed_prior.get("recommended_target_opening_mm", opening_after)),
                    retry_allowed=retry_allowed, retry_used=used, retry_limit=1,
                )
                verification["reason"] = "PREGRASP_SAFETY_STOP_REQUIRES_REFLECTION"
                verification["safety_reflection_context"] = ctx
                verification["proposed_retry_opening_mm"] = ctx.get("suggested_retry_opening_mm")
                warnings.append(
                    f"PRE-GRASP SAFETY STOP: likely cause={likely} ({confidence}). "
                    f"Do not blindly repeat. One reasoned recovery/retry is {'available' if retry_allowed else 'not available'}."
                )
                skill_core.emit(
                    f"[V2-7 FAILURE DIAGNOSIS REQUIRED] approach({obj}) | likely={likely} | "
                    f"opening={opening_before:.1f}->{opening_after:.1f}mm | object_motion={object_move_mm:.1f}mm | "
                    f"retry_allowed={retry_allowed} | proposed_opening={ctx.get('suggested_retry_opening_mm')}mm"
                )
                _v24_set_state(
                    V24_STAGE_SAFETY, active_object=obj, target_bin=None,
                    last_tool=name, last_ok=False,
                )
                return _v24_result(
                    name, False, contract, state_before, verification, warnings,
                    {"object_name": obj, "elapsed_s": round(time.perf_counter()-t0,3)},
                    None,
                    recommended_next_actions=["observe_scene", "recover_from_safety"] if retry_allowed else ["stop_and_explain_failure"],
                )

            # V2-5: pre-grasp descent should be non-contact.  A visible object
            # displacement before clamp is not "centering"; it is evidence that
            # the aperture/collision envelope was too tight or the approach was
            # laterally imperfect.  Treat it as a failed precondition and adapt.
            poke_detected = bool(object_move_mm > 4.0)
            if poke_detected:
                old_bonus = float(V241_APERTURE_RETRY_BONUS_MM.get(obj, 0.0))
                new_bonus = min(24.0, old_bonus + 8.0)
                V241_APERTURE_RETRY_BONUS_MM[obj] = new_bonus
                new_prior = _v24_gripper_prior(obj, opening_after)
                retry_target = float(new_prior["recommended_target_opening_mm"])
                verification.update({
                    "reason": "PREGRASP_CONTACT_POKE_DETECTED",
                    "adaptive_retry_bonus_before_mm": round(old_bonus, 2),
                    "adaptive_retry_bonus_after_mm": round(new_bonus, 2),
                    "retry_recommended_opening_mm": round(retry_target, 2),
                })
                warnings.append(
                    f"ANTI-POKE: {obj} moved {object_move_mm:.1f} mm during pre-grasp descent (>4 mm). "
                    f"Treating this as premature contact, not success. Next recommended aperture={retry_target:.1f} mm."
                )
                skill_core.emit(
                    f"[MANIPULATION PRIOR ADAPT] {obj} | pregrasp_motion={object_move_mm:.1f}mm | "
                    f"retry_bonus={old_bonus:.1f}->{new_bonus:.1f}mm | next_target={retry_target:.1f}mm"
                )

                # Recover vertically before asking the Agent to retry.  Widening
                # while still down around the object is less safe than first
                # separating the fingers from the contact region.
                retreat_ok = False
                retreat_rise_mm = 0.0
                try:
                    retreat_start, _ = skill_core.current_tcp_pose()
                    skill_core.emit("[ANTI-POKE RECOVERY] vertical retreat +90 mm before aperture retry")
                    skill_core.bio_lift(0.090)
                    retreat_end, _ = skill_core.current_tcp_pose()
                    retreat_rise_mm = (float(retreat_end[2]) - float(retreat_start[2])) * 1000.0
                    retreat_ok = retreat_rise_mm > 70.0
                except Exception as exc:
                    warnings.append(f"Anti-poke vertical retreat raised: {exc}")
                verification["anti_poke_retreat_ok"] = bool(retreat_ok)
                verification["anti_poke_retreat_rise_mm"] = round(retreat_rise_mm, 2)
                _v24_set_state(
                    V24_STAGE_OBSERVED,
                    active_object=None,
                    target_bin=None,
                    last_tool=name,
                    last_ok=False,
                )
                return _v24_result(
                    name, False, contract, state_before, verification, warnings,
                    {"object_name": obj, "elapsed_s": round(time.perf_counter()-t0,3)},
                    None,
                    recommended_next_actions=["set_gripper_opening", "observe_scene", "plan_grasp"],
                )

            ok = bool(cart_ok and object_move_mm <= 4.0)
            if not cart_ok:
                verification.setdefault("reason", "CARTESIAN_POSTCONDITION_FAILED")
            return _v24_result(name, ok, contract, state_before, verification, warnings,
                               {"object_name": obj, "elapsed_s": round(time.perf_counter()-t0,3)},
                               V24_STAGE_GRASP_POSE if ok else None)

        if name == "clamp_gripper":
            if state_before.get("stage") != V24_STAGE_GRASP_POSE or not state_before.get("active_object"):
                return _v24_precondition_failure(name, args, f"requires GRASP_POSE_REACHED with active object; current={state_before}", contract, state_before)
            obj = str(state_before["active_object"])
            obj_before, _ = skill_core.object_pose(obj)
            opening_before = float(gripper_actual_mm.value)
            force_n = max(0.5, min(15.0, float(args.get("force_n", 5.0))))
            skill_core.emit(f"[V2-7 GRIPPER] FAST CLAMP -> {force_n:.1f} N/finger")
            gripper_force_N.value = force_n
            clamp_speed_mult_shared.value = 15.0
            hold_enabled_shared.value = True
            gripper_force_mode.value = True
            deadline = time.time() + 3.2
            last_report = 0.0
            while time.time() < deadline:
                left = float(contact_left_n_shared.value); right = float(contact_right_n_shared.value); holding = bool(hold_active_shared.value)
                now = time.time()
                if now-last_report >= 0.55:
                    skill_core.emit(f"[V2-7 GRIPPER WAIT] L={left:.2f}N R={right:.2f}N opening={float(gripper_actual_mm.value):.1f}mm hold={holding}")
                    last_report = now
                if holding or (left >= 0.30 and right >= 0.30): break
                time.sleep(0.05)
            left = float(contact_left_n_shared.value); right = float(contact_right_n_shared.value); holding = bool(hold_active_shared.value)
            opening_after = float(gripper_actual_mm.value)
            obj_after, _ = skill_core.object_pose(obj)
            object_move = _v24_xyz_distance(obj_after, obj_before)
            contacted = left >= 0.30 and right >= 0.30
            ok = bool(contacted or holding)
            warnings = []
            if object_move > 0.015:
                warnings.append(f"Object shifted {object_move*1000.0:.1f} mm while clamping; some centering motion can be normal.")
            verification = {
                "active_object": obj, "contacted_two_finger": contacted, "hold_active": holding,
                "left_N": round(left,3), "right_N": round(right,3),
                "opening_before_mm": round(opening_before,2), "opening_after_mm": round(opening_after,2),
                "object_motion_mm": round(object_move*1000.0,2), "object_motion_required": False,
            }
            if not ok: verification["reason"] = "NO_RELIABLE_GRASP_CONTACT"
            if ok:
                _v3121_record_grasp(obj, opening_before)       # V3.12.1: pick location + in-hand transform
            return _v24_result(name, ok, contract, state_before, verification, warnings,
                               {"elapsed_s": round(time.perf_counter()-t0,3)},
                               V24_STAGE_GRASPED if ok else None)

        if name == "lift":
            if state_before.get("stage") != V24_STAGE_GRASPED or not state_before.get("active_object"):
                return _v24_precondition_failure(name, args, f"requires GRASP_ESTABLISHED; current={state_before}", contract, state_before)
            llm_require_arm_controller()
            obj = str(state_before["active_object"])
            dz = max(0.04, min(0.15, float(args.get("height_m", 0.10))))
            tcp_before, _ = skill_core.current_tcp_pose(); obj_before, _ = skill_core.object_pose(obj)
            target = [float(tcp_before[0]), float(tcp_before[1]), float(tcp_before[2]) + dz]
            # V3.12.1 LIFT FALLBACK: a blocked corridor at the requested height -> lower heights
            # (V3.12 run: lift(0.13) raised "Cartesian corridor only 97.9%" as BACKEND_EXCEPTION).
            lift3121 = {"dz": dz, "tried_mm": []}

            def _v3121_used(h, tried):
                lift3121["tried_mm"] = list(tried)
                if h is not None:
                    lift3121["dz"] = float(h)
                    target[2] = float(tcp_before[2]) + float(h)

            verification, cart_ok, warnings = _v24_run_motion(
                f"lift({dz:.3f}m)", lambda: _v3121_lift_ladder(dz, _v3121_used, f"lift {obj}"), target, 0.012, True
            )
            requested3121 = dz
            dz = float(lift3121["dz"])
            if len(lift3121["tried_mm"]) > 1:
                verification["v3121_lift_heights_tried_mm"] = lift3121["tried_mm"]
                verification["v3121_requested_lift_mm"] = round(requested3121 * 1000.0, 1)
            time.sleep(0.10)
            tcp_after, _ = skill_core.current_tcp_pose(); obj_after, _ = skill_core.object_pose(obj)
            tcp_dz = float(tcp_after[2])-float(tcp_before[2]); obj_dz = float(obj_after[2])-float(obj_before[2])
            rel_before = _v24_sub3(obj_before, tcp_before); rel_after = _v24_sub3(obj_after, tcp_after)
            follow_drift = _v24_xyz_distance(rel_after, rel_before)
            required_rise = max(0.020, min(0.060, 0.50*dz))
            object_follow_ok = obj_dz >= required_rise and follow_drift <= 0.040
            verification.update({
                "active_object": obj, "requested_lift_mm": round(dz*1000.0,1),
                "actual_tcp_rise_mm": round(tcp_dz*1000.0,2), "object_rise_mm": round(obj_dz*1000.0,2),
                "required_object_rise_mm": round(required_rise*1000.0,2), "object_tcp_follow_drift_mm": round(follow_drift*1000.0,2),
                "max_follow_drift_mm": 40.0, "hold_active": bool(hold_active_shared.value), "object_motion_required": True,
            })
            if not bool(hold_active_shared.value): warnings.append("hold_active=false after lift; object-follow verification is therefore especially important.")
            ok = bool(cart_ok and object_follow_ok)
            if cart_ok and not object_follow_ok: verification["reason"] = "OBJECT_DID_NOT_FOLLOW_LIFT"
            if not cart_ok and "corridor blocked at every height" in str(verification.get("legacy_joint_verify_error", "")):
                verification["reason"] = "V3121_LIFT_CORRIDOR_BLOCKED"
                return _v24_result(name, False, contract, state_before, verification, warnings,
                                   {"height_m": dz, "elapsed_s": round(time.perf_counter()-t0,3)}, None,
                                   recommended_next_actions=["return_to_pick_location"])
            return _v24_result(name, ok, contract, state_before, verification, warnings,
                               {"height_m": dz, "elapsed_s": round(time.perf_counter()-t0,3)},
                               V24_STAGE_LIFTED if ok else None)

        if name == "move_above_receptacle":
            if state_before.get("stage") != V24_STAGE_LIFTED or not state_before.get("active_object"):
                return _v24_precondition_failure(name, args, f"requires LIFTED with active object; current={state_before}", contract, state_before)
            llm_require_arm_controller()
            obj = str(state_before["active_object"])
            receptacle = str(args.get("receptacle_name", "egg_holder"))
            slot_name = str(args.get("slot_name", ""))
            if receptacle != "egg_holder" or slot_name not in EGG_SLOT_POSES:
                return _v24_precondition_failure(name, args, f"unknown receptacle/slot: {receptacle}.{slot_name}", contract, state_before)
            fresh_scene = llm_scene_snapshot()
            slot_info = fresh_scene["receptacles"]["egg_holder"]["slots"][slot_name]
            if bool(slot_info.get("occupied")) and slot_info.get("occupied_by") != obj:
                return _v24_precondition_failure(name, args, f"{slot_name} is already occupied by {slot_info.get('occupied_by')}", contract, state_before)
            tcp_before, tcp_q = skill_core.current_tcp_pose(); obj_before, _ = skill_core.object_pose(obj)
            rel_before = _v24_sub3(obj_before, tcp_before)
            sx, sy, _sz = EGG_SLOT_POSES[slot_name]
            desired_obj_hover = [sx, sy, 0.145]
            target = [desired_obj_hover[i] - rel_before[i] for i in range(3)]
            verification, cart_ok, warnings = _v24_run_motion(
                f"move_above_receptacle(egg_holder,{slot_name})",
                lambda: skill_core.move_to_pose(target, tcp_q, velocity_scale=0.16, accel_scale=0.16, position_tolerance=0.010),
                target, 0.035, True
            )
            time.sleep(0.10)
            tcp_after, _ = skill_core.current_tcp_pose(); obj_after, _ = skill_core.object_pose(obj)
            rel_after = _v24_sub3(obj_after, tcp_after)
            follow_drift = _v24_xyz_distance(rel_after, rel_before)
            object_slot_xy = _v24_xy_distance(obj_after, [sx, sy, 0.0])
            transport_ok = follow_drift <= 0.050 and object_slot_xy <= 0.040
            verification.update({
                "active_object": obj, "target_receptacle": receptacle, "target_slot": slot_name,
                "object_xy_error_to_slot_mm": round(object_slot_xy*1000.0,2),
                "object_tcp_follow_drift_mm": round(follow_drift*1000.0,2),
                "hold_active": bool(hold_active_shared.value), "object_motion_required": True,
            })
            ok = bool(cart_ok and transport_ok and bool(hold_active_shared.value))
            if cart_ok and not transport_ok: verification["reason"] = "OBJECT_DID_NOT_REMAIN_ALIGNED_WITH_SLOT"
            return _v24_result(name, ok, contract, state_before, verification, warnings,
                               {"receptacle_name": receptacle, "slot_name": slot_name, "elapsed_s": round(time.perf_counter()-t0,3)},
                               V24_STAGE_BIN_HOVER if ok else None, "__KEEP__", f"egg_holder.{slot_name}" if ok else "__KEEP__")

        if name == "place_into_receptacle":
            if state_before.get("stage") != V24_STAGE_BIN_HOVER or not state_before.get("active_object"):
                return _v24_precondition_failure(name, args, f"requires RECEPTACLE_HOVER_REACHED with active object; current={state_before}", contract, state_before)
            llm_require_arm_controller()
            obj = str(state_before["active_object"])
            slot_name = str(args.get("slot_name") or state_before.get("target_slot") or "")
            if slot_name not in EGG_SLOT_POSES:
                return _v24_precondition_failure(name, args, f"unknown target slot: {slot_name}", contract, state_before)
            tcp_before, tcp_q = skill_core.current_tcp_pose(); obj_before, _ = skill_core.object_pose(obj)
            rel_before = _v24_sub3(obj_before, tcp_before)
            sx, sy, sz = EGG_SLOT_POSES[slot_name]
            desired_obj = [sx, sy, sz]
            target = [desired_obj[i] - rel_before[i] for i in range(3)]
            verification, motion_ok, warnings = _v24_run_motion(
                f"place_into_receptacle({slot_name})",
                lambda: skill_core.move_to_pose(target, tcp_q, velocity_scale=0.10, accel_scale=0.10, position_tolerance=0.008),
                target, 0.018, True
            )
            time.sleep(0.10)
            obj_after, _ = skill_core.object_pose(obj)
            xy_err = _v24_xy_distance(obj_after, [sx, sy, sz])
            z_err = abs(float(obj_after[2]) - float(sz))
            ok = bool(motion_ok and xy_err <= 0.025 and z_err <= 0.035 and bool(hold_active_shared.value))
            verification.update({
                "active_object": obj, "target_receptacle": "egg_holder", "target_slot": slot_name,
                "object_xy_error_to_slot_mm": round(xy_err*1000.0,2),
                "object_z_error_to_place_target_mm": round(z_err*1000.0,2),
                "hold_active": bool(hold_active_shared.value),
            })
            if motion_ok and not ok: verification["reason"] = "HELD_EGG_NOT_AT_SLOT_PLACE_REGION"
            return _v24_result(name, ok, contract, state_before, verification, warnings,
                               {"slot_name": slot_name, "elapsed_s": round(time.perf_counter()-t0,3)},
                               V24_STAGE_SLOT_PLACE if ok else None)

        if name == "move_above_bin":
            if state_before.get("stage") != V24_STAGE_LIFTED or not state_before.get("active_object"):
                return _v24_precondition_failure(name, args, f"requires LIFTED with active object; current={state_before}", contract, state_before)
            llm_require_arm_controller()
            obj = str(state_before["active_object"]); bin_name = str(args["bin_name"])
            tcp_before, _ = skill_core.current_tcp_pose(); obj_before, _ = skill_core.object_pose(obj)
            rel_before = _v24_sub3(obj_before, tcp_before)
            bx, by, rim_z = [float(v) for v in BINS[bin_name]]
            target = [bx, by, rim_z + float(BIN_RELEASE_CLEARANCE)]
            verification, cart_ok, warnings = _v24_run_motion(
                f"move_above_bin({bin_name})", lambda: skill_core.bio_move_above_bin(bin_name), target, 0.035, True
            )
            time.sleep(0.10)
            tcp_after, _ = skill_core.current_tcp_pose(); obj_after, _ = skill_core.object_pose(obj)
            rel_after = _v24_sub3(obj_after, tcp_after)
            follow_drift = _v24_xyz_distance(rel_after, rel_before)
            object_bin_xy = _v24_xy_distance(obj_after, [bx,by,0.0])
            transport_ok = follow_drift <= 0.050 and object_bin_xy <= 0.060
            verification.update({
                "active_object": obj, "target_bin": bin_name, "object_xy_error_to_bin_mm": round(object_bin_xy*1000.0,2),
                "max_object_xy_error_mm": 60.0, "object_tcp_follow_drift_mm": round(follow_drift*1000.0,2),
                "max_follow_drift_mm": 50.0, "hold_active": bool(hold_active_shared.value), "object_motion_required": True,
            })
            if not bool(hold_active_shared.value): warnings.append("hold_active=false during/after bin transport; check object-follow metrics.")
            ok = bool(cart_ok and transport_ok)
            if cart_ok and not transport_ok: verification["reason"] = "OBJECT_DID_NOT_REMAIN_WITH_GRIPPER_TO_BIN"
            return _v24_result(name, ok, contract, state_before, verification, warnings,
                               {"bin_name": bin_name, "elapsed_s": round(time.perf_counter()-t0,3)},
                               V24_STAGE_BIN_HOVER if ok else None, "__KEEP__", bin_name if ok else "__KEEP__")

        if name == "open_gripper":
            if state_before.get("stage") not in (V24_STAGE_SLOT_PLACE, V24_STAGE_BIN_HOVER) or not state_before.get("active_object") or not state_before.get("target_bin"):
                return _v24_precondition_failure(name, args, f"requires SLOT_PLACE_REACHED (egg task) with active object/target; current={state_before}", contract, state_before)
            obj = str(state_before["active_object"]); bin_name = str(state_before["target_bin"])
            obj_before, _ = skill_core.object_pose(obj)
            skill_core.emit("[V2-7 GRIPPER] RELEASE -> detach HOLD + 100 mm opening")
            hold_off_request.value = True; gripper_force_mode.value = False; gripper_target_mm.value = 100.0
            deadline = time.time()+2.5
            while time.time()<deadline:
                if float(gripper_actual_mm.value)>=80.0 and not bool(hold_active_shared.value): break
                time.sleep(0.05)
            time.sleep(0.70)
            obj_after, _ = skill_core.object_pose(obj)
            opening = float(gripper_actual_mm.value); holding = bool(hold_active_shared.value)
            object_move = _v24_xyz_distance(obj_after, obj_before)
            ok = opening >= 80.0 and not holding
            verification = {
                "active_object": obj, "target_bin": bin_name, "opening_mm": round(opening,2),
                "hold_active": holding, "object_motion_after_release_mm": round(object_move*1000.0,2),
                "placement_verified": False,
            }
            if not ok: verification["reason"] = "RELEASE_NOT_CONFIRMED"
            return _v24_result(name, ok, contract, state_before, verification, [],
                               {"elapsed_s": round(time.perf_counter()-t0,3)},
                               V24_STAGE_RELEASED if ok else None)

        if name == "verify_object_in_slot":
            obj = str(args["object_name"]); slot_name = str(args["slot_name"])
            if slot_name not in EGG_SLOT_POSES:
                return _v24_precondition_failure(name, args, f"unknown target slot: {slot_name}", contract, state_before)
            cam311v = v311_mode() == "camera"
            if cam311v:
                o311v = _v311_obj_entry(obj)
                if not (o311v and o311v.get("seen") and o311v.get("trusted")):
                    skill_core.emit(f"[V3.11 RECAPTURE] verify {obj} | {_v311_perception_tag(o311v)}")
                    time.sleep(0.3)
                    try:
                        _v311_capture(f"verify_recapture:{obj}", [obj])
                    except Exception as exc311v:
                        skill_core.emit(f"[V3.11 PERCEPTION FAIL] verify {obj} | {exc311v}")
                    o311v = _v311_obj_entry(obj)
                if o311v and o311v.get("seen") and o311v.get("trusted"):
                    xyz = [float(v) for v in o311v["xyz_m"]]
                    location = llm_classify_object_location(xyz)
                else:
                    xyz = [float(v) for v in o311v["xyz_m"]] if o311v else [0.0, 0.0, 0.0]
                    location = "not_confirmed_by_camera"
                try:
                    _tloc311 = llm_classify_object_location(_v330_nearest_truth(xyz)[1] if _v329_no_prior() else _v329_truth_pose(obj)[0])
                    skill_core.emit(f"[V3.11 VERIFY] {obj} | camera={location} | truth(log only)={_tloc311}"
                                    + ("" if _tloc311 == location else " | MISMATCH"))
                except Exception:
                    pass
            else:
                xyz, _ = skill_core.object_pose(obj); location = llm_classify_object_location(xyz)
            requested_location = f"egg_holder.{slot_name}"
            same_active = state_before.get("active_object") in (None, obj)
            ok = bool(location == requested_location and not bool(hold_active_shared.value))
            warnings = []
            if not same_active:
                warnings.append(f"Verification object {obj} differs from state active_object={state_before.get('active_object')}")
            verification = {
                "object_name": obj, "requested_receptacle": "egg_holder", "requested_slot": slot_name,
                "actual_location": location, "xyz_m": [round(float(v),5) for v in xyz],
                "hold_active": bool(hold_active_shared.value), "placement_verified": bool(ok),
            }
            if cam311v:
                verification["pose_source"] = v335_pose_source_label()
                verification["perception"] = _v311_perception_tag(o311v)
            if not ok: verification["reason"] = ("PERCEPTION_COULD_NOT_CONFIRM" if location == "not_confirmed_by_camera"
                                                 else "OBJECT_NOT_CONFIRMED_IN_TARGET_SLOT")
            # V3.06 VERIFY CARRIES NEXT-STEP STATE: remaining eggs, next empty slot
            # and current aperture, so the Agent does not need observe_scene
            # between eggs (one Agent step saved per egg).
            next306 = None
            try:
                occ306, remaining306 = {}, []
                for _o in OBJECTS:
                    if not str(_o).startswith("egg"):
                        continue
                    if _v329_no_prior() and str(_o) not in _v332_known_eggs():
                        continue                         # V3.32: never perceived in this episode
                    if cam311v:
                        _o311 = _v311_obj_entry(_o)
                        _loc = llm_classify_object_location(_o311["xyz_m"]) if _o311 else "not_perceived"
                    else:
                        _loc = llm_classify_object_location(skill_core.object_pose(_o)[0])
                    if str(_loc).startswith("egg_holder.slot_"):
                        occ306[str(_loc).split(".", 1)[1]] = str(_o)
                    else:
                        remaining306.append(str(_o))
                empty306 = sorted([k for k in EGG_SLOT_POSES if k not in occ306], key=lambda k: int(str(k).split("_")[-1]))
                next306 = {
                    "remaining_eggs_outside_holder": remaining306,
                    "next_egg": remaining306[0] if remaining306 else None,
                    "next_empty_slot": empty306[0] if empty306 else None,
                    "empty_slots_in_order": empty306,
                    "occupied_slots": occ306,
                    "gripper_opening_mm": round(float(gripper_actual_mm.value), 1),
                    "note": "No observe_scene needed: call plan_grasp(next_egg) directly; plan_grasp sets the aperture itself.",
                }
            except Exception as exc306:
                next306 = {"error": str(exc306)[:160]}
            nxt306 = None
            if ok and next306 and "error" not in next306:
                nxt306 = ["plan_grasp"] if next306["remaining_eggs_outside_holder"] else ["finish_and_report"]
            return _v24_result(name, ok, contract, state_before, verification, warnings,
                               {"object_name": obj, "slot_name": slot_name, "elapsed_s": round(time.perf_counter()-t0,3),
                                "v306_next": next306},
                               V24_STAGE_VERIFIED if ok else None, obj if ok else "__KEEP__", requested_location if ok else "__KEEP__",
                               recommended_next_actions=nxt306)

        if name == "return_to_pick_location":
            # V3.12.1 RETURN TO PICK LOCATION: put a held egg back where it was picked, open, retreat.
            # Used when the held egg cannot be lifted/oriented/placed, so the task can continue with the
            # other eggs instead of dead-locking with an egg in the gripper.
            obj = str(args.get("object_name") or state_before.get("active_object") or "")
            origin = V3121_GRASP_ORIGIN.get(obj)
            stage = str(state_before.get("stage") or "")
            held_stages = (V24_STAGE_GRASPED, V24_STAGE_LIFTED, "PLACEMENT_ORIENTED", "SLOT_ENTRY_ALIGNED")
            if not obj or state_before.get("active_object") != obj or stage not in held_stages:
                return _v24_precondition_failure(
                    name, args, f"requires {obj or 'an egg'} to be the held active object (stage {held_stages}); current={state_before}",
                    contract, state_before, recommended_next_actions=["observe_scene"])
            if origin is None:
                return _v24_precondition_failure(name, args, f"no recorded pick location for {obj}", contract, state_before,
                                                 recommended_next_actions=["observe_scene", "stop_and_report"])
            if int(V3121_RETURNS.get(obj, 0)) >= 2:
                return _v24_precondition_failure(name, args, f"{obj} was already returned twice; stop and report",
                                                 contract, state_before, recommended_next_actions=["stop_and_report"])
            llm_require_arm_controller()
            o_tcp = [float(v) for v in origin["tcp_xyz"]]
            o_q = [float(v) for v in origin["tcp_quat"]]
            hover = [o_tcp[0], o_tcp[1], o_tcp[2] + 0.120]
            down = [o_tcp[0], o_tcp[1], o_tcp[2] + 0.003]
            steps = []
            warnings = []

            def _step(label, fn, target, tol):
                ver, ok_s, w = _v24_run_motion(label, fn, target, tol, False)
                warnings.extend(w)
                steps.append({"step": label, "ok": bool(ok_s), "tcp_error_mm": ver.get("tcp_error_mm"),
                              "error": ver.get("legacy_joint_verify_error")})
                return ok_s

            tcp0, q0_ = skill_core.current_tcp_pose()
            same_pose = (math.hypot(float(tcp0[0]) - o_tcp[0], float(tcp0[1]) - o_tcp[1]) <= 0.015
                         and _v3121_qangle_deg(q0_, o_q) <= 10.0)
            at_pick = same_pose and abs(float(tcp0[2]) - o_tcp[2]) <= 0.015
            ok_r = True
            # 1) still above the pick location with the grasp orientation (lift/orient failed early):
            #    no transit needed, just go straight down.  Otherwise transit to the hover first.
            if not same_pose:
                if float(tcp0[2]) < 0.15:
                    _v3121_retreat_to_z(0.18, f"return {obj}: clear first")
                def _go_hover():
                    try:
                        return skill_core.move_to_pose(hover, o_q, velocity_scale=0.12, accel_scale=0.12, position_tolerance=0.010)
                    except Exception as exc340:
                        # V3.40: cuRobo failure -> MoveIt joint goal / BioIK fallback (V3.39 log: BACKEND_EXCEPTION)
                        if "cuRobo" not in str(exc340) or os.environ.get("NERO_V340_MOVEIT_FALLBACK", "1") == "0":
                            raise
                        traj340 = _v340_pose_with_fallback(f"return {obj}: hover", hover, o_q, 0.12, 0.12, 0.010)
                        return skill_core._execute_trajectory(traj340)
                ok_r = _step(f"v3121_return_hover({obj})", _go_hover, hover, 0.015)
            # 2) straight descent to the pick pose (3 mm above, so the egg is set down, not pressed)
            if ok_r and not at_pick:
                def _go_down():
                    traj = skill_core._cartesian_approach_plan(down, o_q)
                    _v218_slow_trajectory_in_place(traj, factor=2.5)
                    skill_core._execute_trajectory(traj)
                    return True
                ok_r = _step(f"v3121_return_descent({obj})", _go_down, down, 0.010)
            released = False
            if ok_r:
                try:
                    skill_core.object_pose(obj)          # V3.41: held pose at the release point (no-truth estimator)
                except Exception:
                    pass
                # 3) release to the pre-grasp aperture (neighbours are close; do not open wide down here)
                opening0 = float(gripper_actual_mm.value)
                gripper_force_mode.value = False
                gripper_target_mm.value = min(100.0, max(float(origin.get("opening_mm") or 60.0), opening0 + 6.0))
                deadline = time.time() + 1.5
                while time.time() < deadline and bool(hold_active_shared.value):
                    time.sleep(0.05)
                time.sleep(0.25)
                released = not bool(hold_active_shared.value)
                # 4) straight retreat
                _rt = _v3121_retreat_to_z(hover[2], f"return {obj}: retreat")
                steps.append({"step": "v3121_return_retreat", "ok": bool(_rt.get("ok", True)), "detail": _rt})
            obj_after, _ = skill_core.object_pose(obj)          # sim monitor (as all in-hand checks)
            loc_after = llm_classify_object_location(obj_after)
            back_xy_mm = _v24_xy_distance(obj_after, origin["object_xyz"]) * 1000.0
            ok = bool(ok_r and released and loc_after == "egg_basket")
            verification = {
                "object_name": obj, "steps": steps, "released": released, "hold_active": bool(hold_active_shared.value),
                "location_after": loc_after, "distance_from_pick_location_mm": round(back_xy_mm, 1),
                "returns_used": int(V3121_RETURNS.get(obj, 0)) + (1 if ok else 0),
            }
            if ok:
                V3121_RETURNS[obj] = int(V3121_RETURNS.get(obj, 0)) + 1
                for _b in ("plans", "oriented", "slot_entry", "released"):
                    V271_PLACEMENT_RUNTIME.get(_b, {}).pop(obj, None)
                V27_GRASP_RUNTIME.get("plans", {}).pop(obj, None)
                V27_GRASP_RUNTIME.get("selected", {}).pop(obj, None)
                V3121_HELD_REL.pop(obj, None)
                V311_MONITOR_BASELINE.pop(obj, None)
                skill_core.emit(f"[V3.12.1 RETURNED] {obj} back in egg_basket ({back_xy_mm:.0f} mm from pick) | "
                                f"returns={V3121_RETURNS[obj]}")
            else:
                verification["reason"] = "RETURN_TO_PICK_LOCATION_FAILED"
            pending = _v3121_pending_eggs() if ok else []
            others = [e for e in pending if e != obj]
            return _v24_result(
                name, ok, contract, state_before, verification, warnings,
                {"object_name": obj, "elapsed_s": round(time.perf_counter() - t0, 3),
                 "v3121_next": {"pending_eggs": pending, "try_next": (others[0] if others else obj),
                                "note": ("Egg is back in the basket and the gripper is free. Continue with the other "
                                         "pending eggs first (plan_grasp), and retry this egg at the end.")} if ok else None},
                V24_STAGE_OBSERVED if ok else None, None if ok else "__KEEP__", None if ok else "__KEEP__",
                recommended_next_actions=(["plan_grasp"] if ok else ["observe_scene", "stop_and_report"]),
            )

        if name == "v306_safe_end":
            # ============================================================
            # V3.06 SAFE END (internal, not exposed to the model)
            # ------------------------------------------------------------
            # Called by the Agent supervisor when the tool-step budget is
            # exhausted.  V3.05 simply stopped with egg_4 held 13 cm above the
            # basket.  Now: holding -> keep the grasp and rise vertically to a
            # safe height; not holding -> SAFE_NEUTRAL q0.
            # ============================================================
            holding306 = bool(hold_active_shared.value)
            tcp306, _ = skill_core.current_tcp_pose()
            act306 = {"holding": holding306, "tcp_z_before_m": round(float(tcp306[2]), 4)}
            ok306 = True
            try:
                if holding306:
                    safe_z306 = float(os.environ.get("NERO_V306_SAFE_END_Z_M", "0.25"))
                    dz306 = min(0.15, max(0.0, safe_z306 - float(tcp306[2])))
                    if dz306 >= 0.01:
                        skill_core.lift(dz306)
                    act306["action"] = f"held egg kept; vertical rise {dz306*1000:.0f} mm"
                else:
                    _v239_plan_exec_q0_neutral()
                    act306["action"] = "SAFE_NEUTRAL q0"
            except Exception as exc306:
                ok306 = False
                act306["error"] = str(exc306)[:240]
            tcp306b, _ = skill_core.current_tcp_pose()
            act306["tcp_z_after_m"] = round(float(tcp306b[2]), 4)
            act306["active_object"] = state_before.get("active_object")
            act306["stage"] = state_before.get("stage")
            skill_core.emit(f"[V3.06 SAFE END] {json.dumps(act306, ensure_ascii=False)}")
            return _v24_result(name, ok306, contract, state_before, act306, [],
                               {"elapsed_s": round(time.perf_counter()-t0, 3)}, next_stage=None,
                               recommended_next_actions=["stop_and_report"])

        if name == "verify_object_in_bin":
            obj = str(args["object_name"]); bin_name = str(args["bin_name"])
            xyz, _ = skill_core.object_pose(obj); location = llm_classify_object_location(xyz)
            same_active = state_before.get("active_object") in (None, obj)
            same_bin = state_before.get("target_bin") in (None, bin_name)
            ok = bool(location == bin_name and not bool(hold_active_shared.value))
            warnings = []
            if not same_active: warnings.append(f"Verification object {obj} differs from state active_object={state_before.get('active_object')}")
            if not same_bin: warnings.append(f"Verification bin {bin_name} differs from state target_bin={state_before.get('target_bin')}")
            verification = {
                "object_name": obj, "requested_bin": bin_name, "actual_location": location,
                "xyz_m": [round(float(v),5) for v in xyz], "hold_active": bool(hold_active_shared.value),
                "placement_verified": bool(ok),
            }
            if not ok: verification["reason"] = "OBJECT_NOT_CONFIRMED_IN_TARGET_BIN"
            return _v24_result(name, ok, contract, state_before, verification, warnings,
                               {"object_name": obj, "requested_bin": bin_name, "elapsed_s": round(time.perf_counter()-t0,3)},
                               V24_STAGE_VERIFIED if ok else None, obj if ok else "__KEEP__", bin_name if ok else "__KEEP__")

        raise RuntimeError(f"Unknown V2-7 Agent tool: {name}")

    _llm_tool_runner_v331 = llm_tool_runner

    def llm_tool_runner(name, args):
        """V3.32 wrapper: in no-prior modes the egg list of the episode is what perception has seen. A never-perceived
        name is refused at once (no look motions), and such names are removed from every result."""
        args = args or {}
        if _v329_no_prior() and name == "plan_grasp" and not args.get("_v307_inner"):
            obj332 = str(args.get("object_name", ""))
            known332 = _v332_known_eggs()
            if obj332.startswith("egg") and obj332 not in known332:
                skill_core.emit(f"[V3.32 UNKNOWN OBJECT] plan_grasp({obj332}) refused | perceived eggs this episode: "
                                f"{known332} | no motion")
                return {"ok": False, "tool": name, "reason": "OBJECT_NOT_PERCEIVED_IN_THIS_EPISODE",
                        "detail": (f"{obj332} has not been perceived in this episode; the eggs that exist are "
                                   f"{known332}. Do not plan for {obj332}."),
                        "perceived_eggs": known332, "warnings": [],
                        "recommended_next_actions": (["plan_grasp"] if _v3121_pending_eggs() else ["finish_and_report"])}
        if name == "scene_snapshot":
            # V3.33: read-only state for the Agent's completion guard (no camera motion, no capture)
            snap333 = llm_scene_snapshot()
            if _v329_no_prior():
                snap333 = _v332_prune(snap333, set(_v332_known_eggs()))
            return {"ok": True, "tool": name, "scene": snap333}
        res = _llm_tool_runner_v331(name, args)
        if name == "observe_scene" and _v329_no_prior() and isinstance(res, dict) and res.get("ok"):
            if not V329_LAYOUT_STATE.get("holder_ok"):
                miss = int(_V329_SCENE.get("holder_miss", 0)) + 1
                _V329_SCENE["holder_miss"] = miss
                res.setdefault("warnings", []).append(
                    "V3.33 HOLDER_NOT_PERCEIVED: the egg carton was not found by the camera, so there is no destination "
                    "slot yet." + (" Call observe_scene once more (it searches again)." if miss < 2 else
                                   " It was searched twice: stop and report that the egg carton could not be found."))
                res["holder_not_perceived"] = True
                res["recommended_next_actions"] = ["observe_scene"] if miss < 2 else ["stop_and_report"]
                skill_core.emit(f"[V3.33 HOLDER NOT PERCEIVED] observe_scene #{miss} in this task | "
                                f"{'one more search allowed' if miss < 2 else 'stop and report'}")
            else:
                _V329_SCENE["holder_miss"] = 0
        if _v329_no_prior() and isinstance(res, dict):
            res = _v332_prune(res, set(_v332_known_eggs()))
        return res

    _llm_tool_runner_v332 = llm_tool_runner

    def llm_tool_runner(name, args):
        """V3.35 wrapper: (1) wrist camera selected but not running -> stop the task (no truth fallback);
        (2) fold in the live captures; (3) pre-motion check of the perceived target against the plan;
        (4) camera-measured egg geometry; (5) in-hand camera check reported with lift / transport results."""
        args = args or {}
        if v326_wrist_selected() and not v326_available():
            try:
                _v335_wait_camera()
            except Exception as exc335:
                skill_core.emit(f"[V3.35 PERCEPTION UNAVAILABLE] {name} | {exc335} -> task stopped")
                try:
                    if llm_agent is not None:
                        llm_agent.request_stop()
                except Exception:
                    pass
                return {"ok": False, "tool": name, "reason": "PERCEPTION_UNAVAILABLE",
                        "detail": str(exc335), "warnings": [], "recommended_next_actions": ["stop_and_report"]}
        V335_STATE["tool_running"] = True
        try:
            _v335_drain()
            if v326_wrist_selected():
                _v335_measured_geometry()
            fg336 = V335_STATE.setdefault("frozen_grasp", {})
            obj336 = str(args.get("object_name") or "")
            if name == "plan_grasp" and v326_wrist_selected():
                fg336.clear()                               # a new grasp plan: freeze THIS egg's geometry now
                if obj336 in V24_OBJECT_GEOMETRY:
                    fg336[obj336] = {"geom": dict(V24_OBJECT_GEOMETRY[obj336]), "t": time.time()}
            if name == "plan_placement":
                _v336_unfreeze_place("new plan_placement")
            pre = _v335_precheck(name, args)
            if pre is not None:
                if pre.get("reason") in ("SLOT_MOVED_SINCE_PLAN", "HOLDER_POSSIBLY_MOVED") and bool(hold_active_shared.value):
                    v24_skill_state["stage"] = V24_STAGE_LIFTED
                skill_core.emit(f"[V3.35 PRECHECK STOP] {name} | {pre['reason']} | {pre['detail']}")
                pre["state_after"] = dict(v24_skill_state)
                return pre
            res = _llm_tool_runner_v332(name, args)
            if v326_wrist_selected() and isinstance(res, dict):
                ok336 = bool(res.get("ok"))
                if name == "plan_grasp":
                    gp = res.get("grasp_plan") or {}
                    if ok336 and obj336 in fg336 and gp.get("object_xyz_at_plan_m"):
                        fg336[obj336].update({"xyz": list(gp["object_xyz_at_plan_m"]),
                                              "quat": list(gp.get("object_quat_at_plan_xyzw") or [0, 0, 0, 1])})
                        skill_core.emit(f"[V3.36 FREEZE] {obj336} | pose / size / aperture frozen for this grasp "
                                        f"(width {fg336[obj336]['geom'].get('effective_width_mm')} mm, "
                                        f"{fg336[obj336]['geom'].get('size_source')})")
                    elif not ok336:
                        fg336.pop(obj336, None)
                elif name == "clamp_gripper" and ok336:
                    fg336.clear()                            # held now: the in-hand transform takes over
                elif name == "plan_placement" and ok336:
                    pp = ((V271_PLACEMENT_RUNTIME.get("plans") or {}).get(str(args.get("object_name") or ""))) or {}
                    V335_STATE["frozen_place"] = {"obj": args.get("object_name"), "slot": pp.get("slot_name"),
                                                  "xyz": pp.get("slot_center_xyz_m"), "t": time.time()}
                    V335_STATE["holder_suspect"] = None
                    skill_core.emit(f"[V3.36 FREEZE] placement {args.get('object_name')} -> {pp.get('slot_name')} | slot "
                                    "frozen until released (live views only check for a moved carton)")
                elif name in ("guarded_insert_and_release", "verify_object_in_slot"):
                    if ok336 or str(res.get("reason") or (res.get("physical_verification") or {}).get("reason") or "").startswith(
                            ("SLOT_MOVED", "HOLDER_POSSIBLY")):
                        _v336_unfreeze_place(name)
            if v326_wrist_selected() and isinstance(res, dict) and name in ("lift", "orient_for_slot", "move_to_slot_entry",
                                                                             "guarded_insert_and_release"):
                pv = res.get("physical_verification")
                if isinstance(pv, dict):
                    pv["v335_in_hand_camera"] = _v335_inhand_verdict()
                    pv["v335_held_pose_source"] = ("in-hand transform from the grasp, checked by the wrist camera "
                                                   "and the gripper force sensing (not a free measurement)")
            return res
        finally:
            V335_STATE["tool_running"] = False

    # ------------------------------------------------------------
    # V3.37 (1) SAFETY LATCH before a recovery motion, (2) EGG LOST + SEARCH.
    # V3.36 log (egg_1): RUNAWAY during orient_for_slot -> the agent called return_to_pick_location with the MuJoCo
    # safety latch still held -> every motion was ignored (13.3 deg tracking mismatch); the dropped egg kept its old
    # pose ("egg_1 back in the basket", truth 130 mm away).
    # ------------------------------------------------------------
    V337_INTENDED_RELEASE = ("guarded_insert_and_release", "return_to_pick_location")

    def _v337_latch_pending():
        try:
            trip = int(safety_trip_count.value)
            return bool(skill_poll_state.get("safety_latch_held", False)) or \
                trip != int(skill_poll_state.get("last_recovered_trip", trip))
        except Exception:
            return False

    def _v337_clear_latch(label):
        """Same handshake as the V2-52.1 RUN TASK gate (STOP + RECOVER, MuJoCo ACK, no new trip, controller re-sync);
        the arm is frozen where it is and the gripper / HOLD are kept (soft freeze)."""
        trip0 = int(safety_trip_count.value)
        skill_core.emit(f"[V3.37 SAFETY LATCH] {label} | latch held (trips={trip0}) -> STOP + RECOVER before moving "
                        f"(hold_active={bool(hold_active_shared.value)})")
        try:
            try:
                skill_core.cancel_active_motion()
            except Exception:
                pass
            try:
                skill_core._set_arm_command_gate(False)
            except Exception:
                ros_arm_command_gate_shared.value = False
            recover_request.value = True
            dl = time.perf_counter() + 3.0
            while bool(recover_request.value) and time.perf_counter() < dl:
                time.sleep(0.02)
            if bool(recover_request.value):
                return False, "MuJoCo did not acknowledge STOP+RECOVER"
            t_ack = int(safety_trip_count.value)
            time.sleep(0.20)
            if int(safety_trip_count.value) != t_ack:
                return False, f"new safety trip during recovery {t_ack}->{int(safety_trip_count.value)}"
            skill_core._sync_arm_controller_to_feedback()
            skill_poll_state["last_recovered_trip"] = int(safety_trip_count.value)
            skill_poll_state["last_safety_trip"] = int(safety_trip_count.value)
            skill_poll_state["safety_latch_held"] = False
            skill_core.emit(f"[V3.37 SAFETY LATCH CLEARED] {label} | STOP+RECOVER ACK | ros2_control synced | "
                            f"hold_active={bool(hold_active_shared.value)}")
            return True, "ok"
        except Exception as exc337:
            skill_poll_state["safety_latch_held"] = True
            return False, str(exc337)

    def _v337_lost():
        return _V329_EST.setdefault("lost", {})

    def _v337_last_capture():
        try:
            return int((V311_STATE.get("last") or {}).get("capture") or 0)
        except Exception:
            return 0

    def _v337_mark_lost(obj, xys, why):
        obj = str(obj)
        pts = []
        for xy in xys:
            if xy is None:
                continue
            xy = [float(xy[0]), float(xy[1])]
            if all(math.hypot(xy[0] - q[0], xy[1] - q[1]) > 0.04 for q in pts):
                pts.append(xy)
        _v337_lost()[obj] = {"t": time.time(), "search_xy": pts, "why": why, "capture_at_loss": _v337_last_capture(),
                             "searches": 0}
        _V329_EST["released"].pop(obj, None)
        if (_V329_EST.get("held_last") or (None,))[0] == obj:
            _V329_EST["held_last"] = None
        for _b in ("plans", "oriented", "slot_entry", "released"):
            (V271_PLACEMENT_RUNTIME.get(_b) or {}).pop(obj, None)
        V27_GRASP_RUNTIME.get("plans", {}).pop(obj, None)
        V27_GRASP_RUNTIME.get("selected", {}).pop(obj, None)
        V3121_HELD_REL.pop(obj, None)
        (V335_STATE.get("frozen_grasp") or {}).pop(obj, None)
        V335_STATE["frozen_place"] = None
        skill_core.emit(f"[V3.37 EGG LOST] {obj} | {why} | pose UNKNOWN now (the old pose is not used any more); "
                        f"next observe_scene searches around {[[round(v, 3) for v in q] for q in pts]}")

    def _v337_update_lost():
        """A lost egg is found again when perception names it in a capture taken after the loss (the tracker had
        forgotten its old track, so this is a fresh detection)."""
        last = V311_STATE.get("last") or {}
        cap = _v337_last_capture()
        for obj, info in list(_v337_lost().items()):
            o = (last.get("objects") or {}).get(obj)
            if o and o.get("seen") and o.get("trusted") and cap > int(info.get("capture_at_loss") or 0):
                _v337_lost().pop(obj, None)
                skill_core.emit(f"[V3.37 EGG FOUND] {obj} | perceived again at "
                                f"{[round(float(v), 3) for v in o.get('xyz_m') or []]} (capture #{cap})")

    def _v337_held_xy():
        try:
            h = _v3121_held_name()
            if not h:
                return None, None
            tcp, tq = skill_core.current_tcp_pose()
            e = _v3121_held_pose_estimate(h, tcp, tq)
            return h, ([float(e[0][0]), float(e[0][1])] if e is not None else [float(tcp[0]), float(tcp[1])])
        except Exception:
            return None, None

    _llm_scene_snapshot_v336 = llm_scene_snapshot

    def llm_scene_snapshot():
        snap = _llm_scene_snapshot_v336()
        try:
            if _v329_no_prior():
                _v337_update_lost()
                for obj, info in _v337_lost().items():
                    (snap.setdefault("objects", {}))[obj] = {
                        "category": "egg", "location": "lost_unknown",
                        "perception": {"source": "camera", "seen": False, "trusted": False, "lost": True,
                                       "lost_reason": info.get("why"),
                                       "search_area_xy_m": [[round(v, 3) for v in q] for q in info.get("search_xy") or []],
                                       "searches_done": int(info.get("searches") or 0)},
                        "note": ("V3.37: this egg was dropped / lost; its position is unknown. observe_scene searches "
                                 "for it. Do not plan_grasp it until it is perceived again."),
                    }
        except Exception:
            pass
        return snap

    _llm_tool_runner_v335 = llm_tool_runner

    def llm_tool_runner(name, args):
        """V3.37 wrapper: safety latch cleared before return_to_pick_location; an egg that leaves the gripper outside
        a release skill is marked LOST (pose unknown) and searched for; a lost egg cannot be planned for."""
        args = args or {}
        if not _v329_no_prior():
            return _llm_tool_runner_v335(name, args)
        obj = str(args.get("object_name") or "")
        if name == "plan_grasp" and not args.get("_v307_inner"):
            _v337_update_lost()
            if obj in _v337_lost():
                info = _v337_lost()[obj]
                skill_core.emit(f"[V3.37 LOST EGG REFUSED] plan_grasp({obj}) | not perceived since it was lost | no motion")
                return {"ok": False, "tool": name, "reason": "EGG_LOST_NOT_RELOCATED",
                        "detail": (f"{obj} was lost ({info.get('why')}) and has not been perceived again. Call "
                                   "observe_scene (it searches around the drop point); if it is still not found after "
                                   "two searches, continue with the other eggs and report it."),
                        "searches_done": int(info.get("searches") or 0), "warnings": [],
                        "recommended_next_actions": ["observe_scene"] if int(info.get("searches") or 0) < 2
                        else (["plan_grasp"] if [e for e in _v3121_pending_eggs() if e != obj] else ["stop_and_report"])}
        if name == "return_to_pick_location" and _v337_latch_pending():
            ok_l, why_l = _v337_clear_latch(name)
            if not ok_l:
                skill_core.emit(f"[V3.37 SAFETY LATCH FAIL] {name} | {why_l}")
                return {"ok": False, "tool": name, "reason": "SAFETY_LATCH_NOT_CLEARED", "detail": why_l,
                        "warnings": [], "recommended_next_actions": ["stop_and_report"]}
        state0 = _v24_state_snapshot()
        held0 = bool(hold_active_shared.value)
        hname0, hxy0 = _v337_held_xy() if held0 else (None, None)
        if (name == "observe_scene" and not held0 and str(state0.get("active_object") or "")
                and str(state0.get("stage") or "") in ("GRASP_ESTABLISHED", "LIFTED", "RECEPTACLE_HOVER_REACHED",
                                                       "SLOT_PLACE_REACHED", "PLACEMENT_ORIENTED", "SLOT_ENTRY_ALIGNED")):
            lost0 = str(state0.get("active_object"))
            if lost0 not in _v337_lost():
                tcp0, _q0 = skill_core.current_tcp_pose()
                hl0 = _V329_EST.get("held_last")
                xy_hl = [float(hl0[1][0][0]), float(hl0[1][0][1])] if hl0 and hl0[0] == lost0 else None
                _v337_mark_lost(lost0, [xy_hl, [float(tcp0[0]), float(tcp0[1])]], "grasp lost between tools")
        res = _llm_tool_runner_v335(name, args)
        if held0 and hname0 and not bool(hold_active_shared.value) and name not in V337_INTENDED_RELEASE:
            try:
                tcp1, _q1 = skill_core.current_tcp_pose()
                _v337_mark_lost(hname0, [hxy0, [float(tcp1[0]), float(tcp1[1])]],
                                f"left the gripper during {name} (hold lost, not a release skill)")
                if isinstance(res, dict):
                    res.setdefault("warnings", []).append(
                        f"V3.37 EGG LOST: {hname0} left the gripper during {name}; its position is unknown. Call "
                        "observe_scene (it searches for it) before anything else.")
                    res["v337_egg_lost"] = {"object_name": hname0, "search_area_xy_m": _v337_lost()[hname0]["search_xy"]}
                    res["recommended_next_actions"] = ["observe_scene"]
            except Exception as exc337:
                skill_core.emit(f"[V3.37 EGG LOST WARN] {exc337}")
        if name == "observe_scene":
            _v337_update_lost()
            for obj_l, info_l in _v337_lost().items():
                info_l["searches"] = int(info_l.get("searches") or 0) + 1
                skill_core.emit(f"[V3.37 EGG STILL LOST] {obj_l} | search #{info_l['searches']} did not find it")
                if isinstance(res, dict):
                    res.setdefault("warnings", []).append(
                        f"V3.37 {obj_l} is still lost after {int(info_l.get('searches') or 0)} search(es)"
                        + ("; continue with the other eggs and report it." if int(info_l.get("searches") or 0) >= 2
                           else "; observe_scene once more searches again."))
        return res

    _v335_cap_lock()
    if v326_wrist_selected() or os.environ.get("NERO_V335_LIVE_THREAD", "1") != "0":
        threading.Thread(target=_v335_live_loop, daemon=True, name="v335_live_perception").start()

    if not llm_agent_error:
        try:
            llm_agent = NeroLLMAgent(
                emit=(skill_core.emit if skill_core is not None else append_gui_log),
                tool_runner=llm_tool_runner,
            )
            append_gui_log("[LLM INIT] NERO V2-9 Skill-Contract agent module loaded")
        except Exception as exc:
            llm_agent_error = str(exc)
            append_gui_log(f"[LLM INIT ERROR] {exc}")

    # ----- Configuration -----
    llm_cfg = ttk.LabelFrame(llm_outer, text="NERO V2-7 LLM planner / reasoning / API")
    llm_cfg.pack(fill="x", pady=(0, 8))

    # Public release: do not hard-code the planner to a particular branded
    # model family.  The field is editable; optional comma-separated presets can
    # be supplied with NERO_LLM_MODEL_PRESETS, and "Refresh API Models" can query
    # the IDs visible to the current API key.
    _preset_env = os.environ.get("NERO_LLM_MODEL_PRESETS", "")
    curated_models = [x.strip() for x in _preset_env.split(",") if x.strip()]
    if DEFAULT_MODEL not in curated_models:
        curated_models.insert(0, DEFAULT_MODEL)
    llm_model_var = tk.StringVar(value=os.environ.get("NERO_LLM_MODEL", os.environ.get("NERO_ASTRA_MODEL", DEFAULT_MODEL)))
    llm_effort_var = tk.StringVar(value="low")
    llm_max_steps_var = tk.StringVar(value=os.environ.get("NERO_V342_DEFAULT_STEPS", "60"))
    llm_dry_run_var = tk.BooleanVar(value=False)
    llm_api_key_var = tk.StringVar(value="")

    ttk.Label(llm_cfg, text="LLM model (editable)").grid(row=0, column=0, padx=(8, 3), pady=7, sticky="w")
    llm_model_combo = ttk.Combobox(
        llm_cfg, textvariable=llm_model_var, values=curated_models,
        state="normal", width=24,
    )
    llm_model_combo.grid(row=0, column=1, padx=(0, 6), pady=7, sticky="w")
    ttk.Button(
        llm_cfg, text="Refresh API Models",
        command=lambda: llm_refresh_models_cmd(),
    ).grid(row=0, column=2, padx=(0, 10), pady=7, sticky="w")

    ttk.Label(llm_cfg, text="Reasoning").grid(row=0, column=3, padx=(3, 3), pady=7)
    ttk.Combobox(
        llm_cfg, textvariable=llm_effort_var,
        values=("none", "low", "medium", "high", "xhigh", "max"),
        state="readonly", width=9,
    ).grid(row=0, column=4, padx=(0, 10), pady=7)

    ttk.Label(llm_cfg, text="Max tool steps").grid(row=0, column=5, padx=(3, 3), pady=7)
    ttk.Spinbox(
        llm_cfg, textvariable=llm_max_steps_var,
        from_=1, to=int(os.environ.get("NERO_V306_MAX_TOOL_STEPS_CAP", "100")), increment=1, width=6,
    ).grid(row=0, column=6, padx=(0, 10), pady=7)
    ttk.Checkbutton(
        llm_cfg, text="Dry run", variable=llm_dry_run_var,
    ).grid(row=0, column=7, padx=8, pady=7, sticky="w")

    ttk.Label(llm_cfg, text="API key (optional; blank = OPENAI_API_KEY)").grid(
        row=1, column=0, columnspan=2, padx=(8, 3), pady=(0, 7), sticky="w"
    )
    llm_key_entry = ttk.Entry(llm_cfg, textvariable=llm_api_key_var, show="*", width=44)
    llm_key_entry.grid(row=1, column=2, columnspan=3, padx=(0, 10), pady=(0, 7), sticky="w")
    ttk.Label(
        llm_cfg,
        text="Refresh lists every model ID visible to your API key. Unknown/specialized models may not support Responses function calling.",
        wraplength=620, justify="left",
    ).grid(row=1, column=5, columnspan=3, padx=(0, 8), pady=(0, 7), sticky="w")

    # V3.42 LLM PROVIDER: OpenAI (Responses) / DeepSeek / Kimi (Chat Completions). One key field; the key typed
    # for each provider is remembered while the GUI runs; blank = that provider's environment variable.
    _v342_labels = {p: (LLM_PROVIDERS.get(p) or {}).get("label", p) for p in ("openai", "deepseek", "kimi")}
    _v342_by_label = {v: k for k, v in _v342_labels.items()}
    llm_provider_var = tk.StringVar(value=_v342_labels[provider_for_model(llm_model_var.get())])
    _v342_keys = {}
    _v342_last_model = {}
    _v342_state = {"prov": provider_for_model(llm_model_var.get())}
    ttk.Label(llm_cfg, text="Provider").grid(row=2, column=0, padx=(8, 3), pady=(0, 7), sticky="w")
    llm_provider_combo = ttk.Combobox(llm_cfg, textvariable=llm_provider_var, values=list(_v342_labels.values()),
                                      state="readonly", width=22)
    llm_provider_combo.grid(row=2, column=1, padx=(0, 6), pady=(0, 7), sticky="w")
    llm_provider_info = ttk.Label(llm_cfg, text="", wraplength=760, justify="left")
    llm_provider_info.grid(row=2, column=2, columnspan=6, padx=(0, 8), pady=(0, 7), sticky="w")

    def _v342_provider_info(p):
        c = LLM_PROVIDERS.get(p) or {}
        llm_provider_info.configure(text=(f"API {c.get('api_base', '')} | key: field above or "
                                          f"{' / '.join(c.get('key_env', ()))} | "
                                          f"{'Responses API' if c.get('protocol') == 'responses' else 'Chat Completions (OpenAI-compatible)'}"))

    def _v342_on_provider(_e=None):
        new = _v342_by_label.get(llm_provider_var.get(), "openai")
        old = _v342_state["prov"]
        if new == old:
            return
        _v342_keys[old] = llm_api_key_var.get()
        _v342_last_model[old] = llm_model_var.get()
        llm_api_key_var.set(_v342_keys.get(new, ""))
        models = list((LLM_PROVIDERS.get(new) or {}).get("models") or [])
        if new == "openai":
            models = list(curated_models)
        llm_model_combo.configure(values=models)
        llm_model_var.set(_v342_last_model.get(new) or (models[0] if models else ""))
        _v342_state["prov"] = new
        _v342_provider_info(new)
        append_gui_log(f"[V3.42 PROVIDER] {(LLM_PROVIDERS.get(new) or {}).get('label', new)} | model={llm_model_var.get()}")
    llm_provider_combo.bind("<<ComboboxSelected>>", _v342_on_provider)
    _v342_provider_info(_v342_state["prov"])
    if _v342_state["prov"] != "openai":
        llm_model_combo.configure(values=list((LLM_PROVIDERS.get(_v342_state["prov"]) or {}).get("models") or []))

    # ----- Task prompt -----
    llm_task_box = ttk.LabelFrame(llm_outer, text="Natural-language task")
    llm_task_box.pack(fill="x", pady=(0, 8))
    llm_task_var = tk.StringVar(value="Pick up the four eggs and place them into empty egg-holder slots, one by one.")
    llm_task_entry = ttk.Entry(llm_task_box, textvariable=llm_task_var)
    llm_task_entry.pack(fill="x", padx=8, pady=(8, 5))

    llm_examples = ttk.Frame(llm_task_box)
    llm_examples.pack(fill="x", padx=8, pady=(0, 8))
    ttk.Label(llm_examples, text="Examples:").pack(side="left")
    for label, prompt in (
        ("Egg 1 → empty slot", "Pick up egg_1 and place it into the first empty egg-holder slot."),
        ("All 4 eggs → holder", "Pick up the four eggs and place them into empty egg-holder slots, one by one."),
    ):
        ttk.Button(
            llm_examples,
            text=label,
            command=lambda p=prompt: llm_task_var.set(p),
        ).pack(side="left", padx=4)

    llm_status_state = {"text": "LLM agent: IDLE"}
    llm_ui_worker_lock = threading.Lock()

    def llm_capture_config():
        try:
            steps = int(llm_max_steps_var.get())
        except Exception:
            steps = 60
            llm_max_steps_var.set("60")
        try:
            if llm_agent is not None:
                llm_agent.provider = _v342_by_label.get(llm_provider_var.get())   # V3.42
        except Exception:
            pass
        return {
            "provider": _v342_by_label.get(llm_provider_var.get(), "openai"),
            "model": llm_model_var.get().strip() or "gpt-6.1-sol",
            "api_key": llm_api_key_var.get().strip(),
            "reasoning_effort": llm_effort_var.get().strip() or "medium",
            "max_tool_steps": max(1, min(int(os.environ.get("NERO_V306_MAX_TOOL_STEPS_CAP", "100")), steps)),
            "dry_run": bool(llm_dry_run_var.get()),
            "task": llm_task_var.get().strip(),
        }

    def llm_background(label, fn, lock_motion=False):
        if llm_agent is None:
            append_gui_log(f"[LLM UI ERROR] Agent unavailable: {llm_agent_error}")
            return
        if not llm_ui_worker_lock.acquire(blocking=False):
            append_gui_log("[LLM UI] Another LLM/V2 debug job is already BUSY")
            return
        if llm_agent.running:
            llm_ui_worker_lock.release()
            append_gui_log("[LLM UI] Agent is already BUSY")
            return
        if lock_motion and skill_core is not None and skill_core.busy:
            llm_ui_worker_lock.release()
            append_gui_log("[LLM UI] V69 skill core is BUSY; wait or STOP + RECOVER")
            return

        llm_status_state["text"] = f"LLM agent: {label}..."

        def worker():
            if lock_motion and skill_core is not None:
                skill_core._set_busy(True)
            try:
                fn()
                llm_status_state["text"] = f"LLM agent: {label} complete"
            except Exception as exc:
                if skill_core is not None:
                    skill_core.emit(f"[LLM UI FAILED] {label}: {exc}")
                else:
                    append_gui_log(f"[LLM UI FAILED] {label}: {exc}")
                llm_status_state["text"] = f"LLM agent: FAILED — {exc}"
            finally:
                if lock_motion and skill_core is not None:
                    skill_core._set_busy(False)
                llm_ui_worker_lock.release()

        threading.Thread(target=worker, daemon=True, name=f"nero_v2_4_2_{label}").start()

    def llm_step_extension_request(info):
        """V3.42: called from the Agent thread when the tool-step budget is used up (or the model wants to stop
        with eggs pending and too few steps left). Shows a dialog in the GUI thread and waits for the answer.
        -> number of extra steps (0 = do not extend). No answer within NERO_V342_EXTEND_TIMEOUT_S = 0."""
        done = threading.Event()
        answer = {"n": 0}

        def show():
            try:
                win = tk.Toplevel(root)
                win.title("NERO V3.42 - 步数已用完 / step budget reached")
                win.transient(root)
                try:
                    win.grab_set()
                except Exception:
                    pass
                pend = info.get("pending_objects") or []
                txt = (f"Agent 已用 {info.get('steps_used')}/{info.get('steps_limit')} 步，任务还没完成。\n"
                       f"原因: {info.get('reason')}\n"
                       + (f"待处理: {', '.join(pend)}\n" if pend else "")
                       + (f"当前持蛋: {'是' if info.get('holding') else '否'} | stage={info.get('stage')}\n")
                       + (f"之前已增加: {info.get('extensions_so_far')}\n" if info.get("extensions_so_far") else "")
                       + "\n是否允许增加步数？选择增加多少步：")
                ttk.Label(win, text=txt, justify="left").pack(padx=14, pady=(12, 6), anchor="w")
                n_var = tk.StringVar(value=os.environ.get("NERO_V342_EXTEND_DEFAULT", "20"))
                row = ttk.Frame(win)
                row.pack(padx=14, pady=4, anchor="w")
                for k in (10, 20, 30, 50):
                    ttk.Button(row, text=f"+{k}", width=5, command=lambda k=k: n_var.set(str(k))).pack(side="left", padx=2)
                ttk.Label(row, text="  自定义:").pack(side="left")
                ttk.Spinbox(row, textvariable=n_var, from_=1, to=200, increment=5, width=6).pack(side="left", padx=4)
                btns = ttk.Frame(win)
                btns.pack(padx=14, pady=(8, 12), anchor="e")

                def finish(n):
                    answer["n"] = n
                    try:
                        win.grab_release()
                    except Exception:
                        pass
                    win.destroy()
                    done.set()

                def ok():
                    try:
                        n = max(0, min(500, int(float(n_var.get()))))
                    except Exception:
                        n = 0
                    finish(n)
                ttk.Button(btns, text="允许增加 / Allow", command=ok).pack(side="left", padx=4)
                ttk.Button(btns, text="不增加，结束任务 / End", command=lambda: finish(0)).pack(side="left", padx=4)
                win.protocol("WM_DELETE_WINDOW", lambda: finish(0))
                try:
                    win.bell()
                except Exception:
                    pass
            except Exception as exc:
                append_gui_log(f"[V3.42 STEP DIALOG ERROR] {exc}")
                done.set()

        root.after(0, show)
        if not done.wait(float(os.environ.get("NERO_V342_EXTEND_TIMEOUT_S", "600"))):
            append_gui_log("[V3.42 STEP DIALOG] no answer in time -> not extended")
            return 0
        return int(answer["n"])

    try:
        if llm_agent is not None:
            llm_agent.step_extension_cb = llm_step_extension_request
    except Exception:
        pass

    def llm_refresh_models_cmd():
        cfg = llm_capture_config()
        def run():
            ids = llm_agent.list_available_models(api_key=cfg["api_key"])
            # Keep exact API-visible list, but place common general-purpose models first.
            ordered = []
            for mid in curated_models:
                if mid in ids and mid not in ordered:
                    ordered.append(mid)
            for mid in ids:
                if mid not in ordered:
                    ordered.append(mid)
            def apply_models():
                llm_model_combo.configure(values=ordered)
                append_gui_log(
                    f"[MODEL DISCOVERY] GUI updated with {len(ordered)} API-visible model IDs; "
                    f"current={llm_model_var.get()}"
                )
            root.after(0, apply_models)
        llm_background("refresh API models", run, lock_motion=False)

    def llm_preflight_estimate(cfg, show_dialog=True):
        scene = llm_scene_snapshot()
        try:
            scene["v2_skill_state"] = _v24_state_snapshot()
        except Exception:
            pass
        estimate = llm_agent.estimate_task_usage(
            cfg["task"], scene, model=cfg["model"],
            reasoning_effort=cfg["reasoning_effort"],
            max_tool_steps=cfg["max_tool_steps"],
        )
        skill_core.emit("[PRE-FLIGHT ESTIMATE] " + json.dumps(estimate, ensure_ascii=False, separators=(",", ":")))
        likely = estimate.get("likely_cost_usd")
        upper = estimate.get("conservative_cost_usd")
        if estimate.get("pricing_known"):
            price_line = f"Estimated cost: about ${likely:.4f} likely; <= ${upper:.4f} conservative uncached estimate"
        else:
            price_line = "Estimated cost: unavailable for this model ID (pricing not in local map)"
        message = (
            f"Model: {estimate['model']}\n"
            f"Reasoning: {estimate['reasoning_effort']}\n"
            f"Estimated objects: {estimate['objects_estimated']}\n"
            f"Estimated tool steps: {estimate['tool_steps_estimated']}\n"
            f"Estimated API requests: {estimate['api_requests_estimated']}\n"
            f"Estimated input tokens: {estimate['input_tokens_estimated']:,}\n"
            f"Estimated output tokens: {estimate['output_tokens_estimated']:,}\n"
            f"{price_line}\n\n"
            "This is a local heuristic, not a billing quote. Retries and longer reasoning can increase usage."
        )
        if show_dialog:
            return estimate, bool(messagebox.askyesno("NERO V2-7 API Pre-flight", message + "\n\nExecute this task?"))
        return estimate, message

    def llm_estimate_cmd():
        cfg = llm_capture_config()
        if not cfg["task"]:
            append_gui_log("[PRE-FLIGHT ESTIMATE] Task is empty")
            return
        try:
            _estimate, message = llm_preflight_estimate(cfg, show_dialog=False)
            messagebox.showinfo("NERO V2-7 API Estimate", message)
        except Exception as exc:
            append_gui_log(f"[PRE-FLIGHT ESTIMATE FAILED] {exc}")
            messagebox.showerror("NERO V2-7", f"Could not prepare task estimate:\n{exc}")

    def llm_test_api_cmd():
        cfg = llm_capture_config()
        llm_background(
            "API test",
            lambda: llm_agent.test_connection(
                model=cfg["model"], api_key=cfg["api_key"], reasoning_effort="low"
            ),
            lock_motion=False,
        )

    def llm_observe_cmd():
        def run():
            result = llm_tool_runner("observe_scene", {})
            skill_core.emit("[LLM MANUAL OBSERVE RESULT] " + json.dumps(result, ensure_ascii=False))
        llm_background("observe scene", run, lock_motion=False)

    def llm_plan_only_cmd():
        cfg = llm_capture_config()
        def run():
            scene = llm_scene_snapshot()
            try:
                scene["v2_skill_state"] = _v24_state_snapshot()
            except Exception:
                pass
            llm_agent.plan_only(
                cfg["task"], scene,
                model=cfg["model"], api_key=cfg["api_key"],
                reasoning_effort=cfg["reasoning_effort"],
            )
        llm_background("plan only", run, lock_motion=False)

    # V2-39 SAFE_NEUTRAL_HIGH TASK-START NORMALIZATION
    _V239_LOW_Z_M = 0.255
    _V239_CLEAR_Z_M = 0.340
    _V239_NEUTRAL_HIGH_Z_M = 0.550
    _V239_JOINT_MARGIN_MIN_DEG = 12.0
    _V239_DISTAL_WRIST_MAX_DEG = 140.0

    def _v239_health_snapshot():
        q=_v238_live_q(); xyz,quat=skill_core.current_tcp_pose()
        xyz=[float(v) for v in xyz]; quat=[float(v) for v in quat]
        x,y,z=xyz
        near_basket, near_holder = _v329_near_zone(x, y, z, 0.315)      # V3.29: from the current layout
        margin=_v238_joint_margin_deg(q)
        topdown=_v238_topdown_error_deg(quat)  # diagnostic only; NOT a task-start reject condition
        distal=max(abs(math.degrees(q[5])),abs(math.degrees(q[6])))
        reasons=[]
        if z < _V239_LOW_Z_M: reasons.append(f"low_z={z*1000.0:.0f}mm")
        if near_basket: reasons.append("near_basket")
        if near_holder: reasons.append("near_holder")
        if margin < _V239_JOINT_MARGIN_MIN_DEG: reasons.append(f"joint_margin={margin:.1f}deg")
        if distal > _V239_DISTAL_WRIST_MAX_DEG: reasons.append(f"distal_wrist={distal:.1f}deg")
        safety_code=0
        try: safety_code=int(safety_state.value)
        except Exception: pass
        if safety_code != 0: reasons.append(f"safety_state={safety_code}")
        if bool(hold_active_shared.value): reasons.append("hold_active")
        if not reasons and z >= _V239_NEUTRAL_HIGH_Z_M and margin >= 20.0:
            classification="SAFE_NEUTRAL_HIGH"
        elif not reasons and z >= 0.315:
            classification="SAFE_WORKSPACE_CLEAR"
        elif not reasons:
            classification="SAFE_CURRENT"
        else:
            classification="NORMALIZE_REQUIRED"
        return {"q":q,"xyz":xyz,"quat":quat,"joint_margin_deg":margin,
                "topdown_error_deg":topdown,"distal_wrist_deg":distal,
                "near_basket":near_basket,"near_holder":near_holder,
                "safety_code":safety_code,"classification":classification,"reasons":reasons}

    def _v239_plan_exec_q0_neutral():
        qnow=_v238_live_q(); qzero=[0.0]*7
        if max(abs(math.degrees(float(qnow[i]))) for i in range(7)) <= 2.0:
            skill_core.emit("[V2-39 SAFE_NEUTRAL Q0 REACHED] already at q0")
            return True
        skill_core.emit("[V2-39 SAFE_NEUTRAL Q0 PLAN] MoveIt joint target q=[0,0,0,0,0,0,0] + dense collision revalidation")
        traj=skill_core._plan_joint_solution(
            qzero, velocity_scale=0.12, accel_scale=0.08,
            planning_attempts=10, planning_time=10.0,
        )
        if traj is None:
            raise RuntimeError("V2-39 q0 MoveIt joint planning returned no trajectory")
        jt=traj.joint_trajectory
        names=[str(n) for n in jt.joint_names]
        idx=[names.index(f"joint{i}") for i in range(1,8)]
        raw=[[float(p.positions[j]) for j in idx] for p in jt.points]
        if not raw: raise RuntimeError("V2-39 q0 trajectory has no arm points")
        qpath=[list(qnow)]
        for qgoal in raw:
            seg=_v238_interp_q(qpath[-1],qgoal,step_deg=2.0)
            qpath.extend(seg[1:])
        for i,qv in enumerate(qpath):
            ok,why=skill_core._v71_state_is_valid(qv)
            if not ok: raise RuntimeError(f"V2-39 q0 path rejected at {i}/{len(qpath)-1}: {why}")
        poses=_v238_fk_batch(qpath)
        minz=min(float(p["position"][2]) for p in poses)
        # V3.28 Q0 FLOOR FROM START: 180 mm, or the start height - 20 mm when the arm already starts lower
        # (V3.27 run: retreat blocked at z=143 mm -> every q0 path "violated" the fixed 180 mm floor -> deadlock).
        z0_328=float(poses[0]["position"][2])
        floor_328=min(0.180, z0_328-float(os.environ.get("NERO_V328_Q0_FLOOR_DROP_M","0.020")))
        if minz < floor_328:
            raise RuntimeError(f"V2-39 q0 path TCP floor violated: {minz*1000.0:.1f}mm < {floor_328*1000.0:.0f}mm")
        if floor_328 < 0.180:
            skill_core.emit(f"[V3.28 Q0 FLOOR FROM START] start z={z0_328*1000.0:.0f}mm -> floor {floor_328*1000.0:.0f}mm "
                            f"| path min {minz*1000.0:.0f}mm")
        if max(abs(math.degrees(float(qpath[-1][i]))) for i in range(7)) > 2.0:
            raise RuntimeError("V2-39 q0 planned endpoint misses neutral target")
        _v218_slow_trajectory_in_place(traj,factor=1.60)
        safety0=int(safety_trip_count.value)
        skill_core._execute_trajectory(traj)
        if int(safety_trip_count.value)!=safety0:
            raise RuntimeError("V2-39 q0 neutral execution triggered safety")
        qend=_v238_live_q(); xyzend,_=skill_core.current_tcp_pose()
        qerr=max(abs(math.degrees(float(qend[i]))) for i in range(7))
        if qerr > 3.0: raise RuntimeError(f"V2-39 q0 physical joint error {qerr:.1f}deg")
        skill_core.emit(
            f"[V2-39 SAFE_NEUTRAL Q0 REACHED] max_joint={qerr:.1f}deg | tcp_z={float(xyzend[2])*1000.0:.0f}mm"
        )
        return True

    def _v239_task_start_posture_check(edge_mode=False):
        snap=_v239_health_snapshot(); qdeg=[round(math.degrees(v),1) for v in snap["q"]]
        prefix="EDGE" if bool(edge_mode) else "TASK START"
        skill_core.emit(
            f"[V2-39 {prefix} POSTURE CHECK] classification={snap['classification']} | joints_deg={qdeg} | "
            f"tcp=[{snap['xyz'][0]:+.3f},{snap['xyz'][1]:+.3f},{snap['xyz'][2]:+.3f}] | "
            f"joint_margin={snap['joint_margin_deg']:.1f}deg | topdown_diag={snap['topdown_error_deg']:.1f}deg | "
            f"reasons={snap['reasons']}"
        )
        if bool(hold_active_shared.value):
            raise RuntimeError("V2-39 refuses automatic neutralization while hold_active=true")
        if snap["safety_code"] != 0:
            raise RuntimeError(f"V2-39 safety state latched/code={snap['safety_code']}; recover first")
        if not snap["reasons"]:
            if snap["classification"]=="SAFE_NEUTRAL_HIGH":
                skill_core.emit("[V2-39 SAFE_NEUTRAL_HIGH ACCEPT] high/clear posture accepted; top-down is not required at task start")
            else:
                skill_core.emit(f"[V2-39 POSTURE ACCEPT] {snap['classification']} | no normalization required")
            return True

        safety0=int(safety_trip_count.value)
        xyz0=list(snap["xyz"]); quat0=list(snap["quat"])
        if xyz0[2] < 0.315 or snap["near_basket"] or snap["near_holder"]:
            retreat=[xyz0[0],xyz0[1],max(_V239_CLEAR_Z_M,xyz0[2]+0.120)]
            skill_core.emit(
                f"[V2-39 SAFE RETREAT] z={xyz0[2]*1000.0:.0f} -> {retreat[2]*1000.0:.0f}mm | vertical only | preserve wrist"
            )
            traj=skill_core._cartesian_approach_plan(retreat,quat0)
            _v218_slow_trajectory_in_place(traj,factor=1.45)
            skill_core._execute_trajectory(traj)
            if int(safety_trip_count.value)!=safety0:
                raise RuntimeError("V2-39 vertical safe retreat triggered safety")
            snap=_v239_health_snapshot()
            if not snap["reasons"]:
                skill_core.emit("[V2-39 POST-RETREAT ACCEPT] workspace clear; q0 normalization not required")
                return True

        # Remaining reasons are configuration health (joint margin / distal wrist),
        # not Cartesian height or top-down style. Fall back only to the known q0
        # neutral, and only after MoveIt + dense collision/state validation.
        _v239_plan_exec_q0_neutral()
        snap2=_v239_health_snapshot()
        if snap2["reasons"]:
            raise RuntimeError(f"V2-39 q0 neutral reached but posture still unhealthy: {snap2['reasons']}")
        return True

    # V2-38 TASK START POSTURE CHECK + SESSION Q_TABLE_READY
    _v238_table_ready = {"q": None, "xyz": None, "quat": None, "source": None}
    _V238_READY_XYZ_CANDIDATES = ((-0.425, 0.000, 0.340), (-0.425, 0.000, 0.400))
    _V238_READY_POS_TOL_M = 0.025
    _V238_READY_ORIENT_TOL_DEG = 15.0
    _V238_LOW_Z_M = 0.255
    _V238_RETREAT_Z_M = 0.340

    def _v238_quat_error_deg(q1, q2):
        a=[float(v) for v in q1]; b=[float(v) for v in q2]
        na=math.sqrt(sum(v*v for v in a)); nb=math.sqrt(sum(v*v for v in b))
        if na < 1e-9 or nb < 1e-9: return 180.0
        d=abs(sum((a[i]/na)*(b[i]/nb) for i in range(4)))
        d=max(-1.0,min(1.0,d))
        return math.degrees(2.0*math.acos(d))

    def _v238_topdown_error_deg(q):
        # Yaw is intentionally free; only require the wrist to belong to the
        # top-down family. Sample equivalent top-down yaws and take the best.
        return min(_v238_quat_error_deg(q, _v27_top_down_quat(math.radians(d)))
                   for d in range(-180, 180, 30))

    def _v238_live_q():
        js=skill_core._current_joint_state_copy()
        q=[float(v) for v in skill_core._arm_positions_from_joint_state(js)]
        if len(q) < 7: raise RuntimeError("V2-38 requires 7 arm joints")
        return q[:7]

    def _v238_joint_margin_deg(q):
        try:
            return min(math.degrees(min(float(q[i])-float(JOINT_LIMITS_RAD[i][0]),
                                        float(JOINT_LIMITS_RAD[i][1])-float(q[i]))) for i in range(7))
        except Exception:
            return 999.0

    def _v238_health_snapshot():
        q=_v238_live_q(); xyz, quat=skill_core.current_tcp_pose()
        xyz=[float(v) for v in xyz]; quat=[float(v) for v in quat]
        x,y,z=xyz
        near_basket, near_holder = _v329_near_zone(x, y, z, 0.305)      # V3.29: from the current layout
        margin=_v238_joint_margin_deg(q)
        topdown=_v238_topdown_error_deg(quat)
        reasons=[]
        if z < _V238_LOW_Z_M: reasons.append(f"low_z={z*1000.0:.0f}mm")
        if z > 0.600: reasons.append(f"too_high_for_table={z*1000.0:.0f}mm")
        if near_basket: reasons.append("near_basket")
        if near_holder: reasons.append("near_holder")
        if margin < 12.0: reasons.append(f"joint_margin={margin:.1f}deg")
        if max(abs(math.degrees(q[5])), abs(math.degrees(q[6]))) > 135.0:
            reasons.append("distal_wrist_extreme")
        if topdown > 35.0: reasons.append(f"non_topdown={topdown:.1f}deg")
        safety_code=0
        try: safety_code=int(safety_state.value)
        except Exception: pass
        if safety_code != 0: reasons.append(f"safety_state={safety_code}")
        if bool(hold_active_shared.value): reasons.append("hold_active")
        return {"q":q,"xyz":xyz,"quat":quat,"joint_margin_deg":margin,
                "topdown_error_deg":topdown,"near_basket":near_basket,
                "near_holder":near_holder,"safety_code":safety_code,"reasons":reasons}

    def _v238_interp_q(q0, q1, step_deg=4.0):
        md=max(abs(float(q1[i])-float(q0[i])) for i in range(7))
        n=max(1,int(math.ceil(md/math.radians(float(step_deg)))))
        return [[float(q0[j])+(float(q1[j])-float(q0[j]))*(float(k)/float(n)) for j in range(7)]
                for k in range(n+1)]

    def _v238_fk_batch(qbatch):
        client=getattr(skill_core,"_gpu_client",None)
        if client is None: raise RuntimeError("GPU FK client unavailable")
        out=client.fk_batch(joint_names=[f"joint{i}" for i in range(1,8)], q_batch=qbatch, timeout_s=20.0)
        if not bool(out.get("ok")): raise RuntimeError("cuRobo FK failed: "+str(out.get("error")))
        poses=out.get("poses") or []
        if len(poses)!=len(qbatch): raise RuntimeError(f"FK count mismatch {len(poses)} != {len(qbatch)}")
        return poses

    def _v238_exec_cached_ready(qready):
        # Fixed endpoint, never blind joint interpolation. MoveIt plans to
        # the frozen 7-joint posture, then every state is revalidated.
        traj=skill_core._plan_joint_solution(
            qready, velocity_scale=0.14, accel_scale=0.10,
            planning_attempts=10, planning_time=8.0,
        )
        if traj is None:
            return list(_v238_table_ready.get("xyz") or skill_core.current_tcp_pose()[0])
        jt=traj.joint_trajectory
        names=[str(n) for n in jt.joint_names]
        idx=[names.index(f"joint{i}") for i in range(1,8)]
        raw=[]
        for p in jt.points:
            raw.append([float(p.positions[j]) for j in idx])
        if not raw:
            raise RuntimeError("cached TABLE_READY MoveIt path has no arm points")
        # Dense revalidation between MoveIt waypoints: <=2deg/joint step.
        qpath=[_v238_live_q()]
        for qgoal in raw:
            seg=_v238_interp_q(qpath[-1],qgoal,step_deg=2.0)
            qpath.extend(seg[1:])
        for qv in qpath:
            ok,why=skill_core._v71_state_is_valid(qv)
            if not ok: raise RuntimeError(f"cached TABLE_READY path rejected: {why}")
        if max(abs(math.degrees(float(qpath[-1][i])-float(qready[i]))) for i in range(7)) > 2.0:
            raise RuntimeError("cached TABLE_READY planned endpoint misses frozen joint target")
        poses=_v238_fk_batch(qpath)
        if min(float(p["position"][2]) for p in poses) < 0.180:
            raise RuntimeError("cached TABLE_READY path dips below 180mm TCP floor")
        _v218_slow_trajectory_in_place(traj,factor=1.35)
        skill_core._execute_trajectory(traj)
        return [float(v) for v in poses[-1]["position"]]

    def _v238_task_start_posture_check(force_ready=False):
        snap=_v238_health_snapshot()
        qdeg=[round(math.degrees(v),1) for v in snap["q"]]
        skill_core.emit(
            f"[V2-38 TASK START POSTURE CHECK] joints_deg={qdeg} | "
            f"tcp=[{snap['xyz'][0]:+.3f},{snap['xyz'][1]:+.3f},{snap['xyz'][2]:+.3f}] | "
            f"joint_margin={snap['joint_margin_deg']:.1f}deg | topdown={snap['topdown_error_deg']:.1f}deg | "
            f"reasons={snap['reasons']}"
        )
        if bool(force_ready):
            skill_core.emit("[V2-38 FORCE TABLE_READY] canonical posture required before edge manipulation")
        if bool(hold_active_shared.value):
            raise RuntimeError("V2-38 refuses automatic task-start reset while hold_active=true")
        if snap["safety_code"] != 0:
            raise RuntimeError(f"V2-38 task-start safety state is latched/code={snap['safety_code']}; recover first")
        if not snap["reasons"] and not bool(force_ready):
            skill_core.emit("[V2-38 TASK START POSTURE OK] current posture accepted; no reset needed")
            return True

        safety0=int(safety_trip_count.value)
        xyz0=list(snap["xyz"]); quat0=list(snap["quat"])
        # If low/near either workspace fixture, first escape straight upward
        # with the current wrist orientation. No lateral motion is allowed here.
        if xyz0[2] < 0.315 or snap["near_basket"] or snap["near_holder"]:
            retreat=[xyz0[0],xyz0[1],max(_V238_RETREAT_Z_M,xyz0[2]+0.120)]
            skill_core.emit(
                f"[V2-38 TASK START SAFE RETREAT] z={xyz0[2]*1000.0:.0f} -> {retreat[2]*1000.0:.0f}mm | vertical only"
            )
            traj=skill_core._cartesian_approach_plan(retreat,quat0)
            _v218_slow_trajectory_in_place(traj,factor=1.45)
            skill_core._execute_trajectory(traj)
            if int(safety_trip_count.value)!=safety0:
                raise RuntimeError("V2-38 task-start vertical retreat triggered safety")

        # Once established, Q_TABLE_READY is a fixed seven-joint session pose.
        if _v238_table_ready.get("q") is not None:
            try:
                qnow=_v238_live_q(); xyznow,_qnow=skill_core.current_tcp_pose()
                qdiff=max(abs(math.degrees(float(qnow[i])-float(_v238_table_ready["q"][i]))) for i in range(7))
                if qdiff <= 3.0 and _v24_xyz_distance(xyznow,_v238_table_ready["xyz"]) <= 0.030:
                    skill_core.emit("[V2-38 TABLE_READY REACHED] already at frozen session posture")
                    return True
                skill_core.emit("[V2-38 Q_TABLE_READY RETURN] using frozen session joint posture")
                _v238_exec_cached_ready(_v238_table_ready["q"])
                xyzr, qr=skill_core.current_tcp_pose()
                if _v24_xyz_distance(xyzr,_v238_table_ready["xyz"]) <= 0.030:
                    skill_core.emit("[V2-38 TABLE_READY REACHED] cached joint posture verified")
                    return True
                raise RuntimeError("cached TABLE_READY final TCP mismatch")
            except Exception as exc:
                skill_core.emit(f"[V2-38 Q_TABLE_READY CACHE REJECT] {exc} | strict canonical replan")
                _v238_table_ready.update({"q":None,"xyz":None,"quat":None,"source":None})

        # Resolve the canonical posture with cuRobo Strict + MoveIt full-trajectory gate ONLY.
        # No BioIK/legacy pose fallback is allowed for this task-entry normalization.
        qready_quat=_v27_top_down_quat(0.0)
        last_exc=None
        for target in _V238_READY_XYZ_CANDIDATES:
            target=[float(v) for v in target]
            try:
                skill_core.emit(f"[V2-38 TABLE_READY STRICT PLAN] target={target} | top-down | cuRobo+MoveIt only")
                traj=skill_core._plan_pose(
                    target,qready_quat,velocity_scale=0.14,accel_scale=0.10,
                    orientation_tolerance=0.14,position_tolerance=0.018,
                    planning_attempts=10,planning_time=10.0,
                )
                skill_core._execute_trajectory(traj)
                xyzr, qr=skill_core.current_tcp_pose()
                perr=_v24_xyz_distance(xyzr,target)
                oerr=_v238_quat_error_deg(qr,qready_quat)
                qactual=_v238_live_q(); margin=_v238_joint_margin_deg(qactual)
                if perr > _V238_READY_POS_TOL_M or oerr > _V238_READY_ORIENT_TOL_DEG or margin < 10.0:
                    raise RuntimeError(
                        f"TABLE_READY verify failed pos={perr*1000.0:.1f}mm orient={oerr:.1f}deg margin={margin:.1f}deg"
                    )
                _v238_table_ready.update({
                    "q":list(qactual),"xyz":[float(v) for v in xyzr],"quat":[float(v) for v in qr],
                    "source":"curobo_strict_canonical",
                })
                skill_core.emit(
                    f"[V2-38 Q_TABLE_READY FROZEN] joints_deg={[round(math.degrees(v),1) for v in qactual]} | "
                    f"tcp=[{xyzr[0]:+.3f},{xyzr[1]:+.3f},{xyzr[2]:+.3f}]"
                )
                skill_core.emit("[V2-38 TABLE_READY REACHED] canonical posture verified before LLM task")
                return True
            except Exception as exc:
                last_exc=exc
                skill_core.emit(f"[V2-38 TABLE_READY STRICT MISS] {exc}")
        raise RuntimeError(f"V2-38 could not establish collision-checked TABLE_READY: {last_exc}")

    def llm_run_task_cmd():
        # V3.43 A2 BEGIN
        if _v343_scene_name() != "fill_egg_holder":
            append_gui_log(f"[V3.43 SCENE] RUN TASK blocked: scene {_v343_scene_name()} has no Agent skills yet")
            messagebox.showinfo("NERO V3.44", f"场景 {_v343_scene_name()} 目前只提供仿真场景，还没有 Agent 技能。\n"
                                              "This scene has no Agent skills yet.")
            return
        # V3.43 A2 END
        cfg = llm_capture_config()
        if not cfg["task"]:
            append_gui_log("[LLM UI] Task is empty")
            return
        try:
            estimate, confirmed = llm_preflight_estimate(cfg, show_dialog=True)
        except Exception as exc:
            append_gui_log(f"[PRE-FLIGHT ESTIMATE FAILED] {exc}")
            messagebox.showerror("NERO V2-7", f"Could not prepare task estimate:\n{exc}")
            return
        if not confirmed:
            append_gui_log(
                f"[TASK CANCELLED BY USER] model={cfg['model']} | task={cfg['task']!r} | "
                f"preflight_estimate={json.dumps(estimate, ensure_ascii=False, separators=(',', ':'))}"
            )
            return
        # V2-7: every new user-confirmed task starts a fresh logical episode
        # when no object is physically held.  This prevents ZERO ALL or a prior
        # failed task from leaking HOVER_REACHED / planning retry memory into the
        # next command.
        if not bool(hold_active_shared.value):
            _v24_reset_skill_state("new RUN TASK", clear_adaptation=True)
        else:
            append_gui_log(
                "[V2-7 NEW TASK] hold_active=true; preserving current grasp/task "
                "state rather than erasing an object that is physically held"
            )
        append_gui_log(
            f"[TASK CONFIRMED BY USER] provider={cfg.get('provider')} | model={cfg['model']} | task={cfg['task']!r} | "
            "fresh_episode=true | safety_retry_budget=1/object | planning_retry_budget=1/object"
        )
        append_gui_log(f"[V3.11 POSE SOURCE] this task: "
                       f"{(('wrist camera USB 130deg RGB-only perception (V3.30)' if v330_wrist_profile() == 'usb130' else 'wrist camera D435i perception (V3.26/V3.27)') if v326_wrist_selected() else 'fixed camera D435i perception') if v311_mode() == 'camera' else 'JSON ground truth'}")
        def _v238_run_task_after_posture():
            if not bool(cfg.get("dry_run", False)):
                _v239_task_start_posture_check(edge_mode=False)
            return llm_agent.run_task(
                cfg["task"],
                model=cfg["model"],
                api_key=cfg["api_key"],
                reasoning_effort=cfg["reasoning_effort"],
                max_tool_steps=cfg["max_tool_steps"],
                dry_run=cfg["dry_run"],
            )
        if V311_STATE.get("mode") == "camera" and not v311_camera_available():
            append_gui_log("[V3.35 RUN TASK REFUSED] the fixed camera is the pose source but its process is not running "
                           "- no fallback to simulator truth (select 'JSON ground truth' explicitly if that is intended)")
            try:
                messagebox.showerror("NERO V3.35", "RUN TASK refused: the fixed camera process is not running.\n"
                                     "No fallback to simulator truth.")
            except Exception:
                pass
            return
        if v326_wrist_selected() and not v326_available():
            append_gui_log("[V3.35 RUN TASK REFUSED] a wrist camera is the pose source but its process is not running "
                           "(see console). V3.35 never falls back to simulator truth - fix the camera or pick another "
                           "pose source.")
            try:
                messagebox.showerror("NERO V3.35", "RUN TASK refused: the wrist camera process is not running.\n"
                                     "No fallback to simulator truth. See the console / runtime log.")
            except Exception:
                pass
            return
        if not _v2521_run_task_safety_gate():
            append_gui_log("[V2-52.1 RUN TASK BLOCKED] stale physical safety latch was not safely recovered")
            try:
                messagebox.showerror("NERO V2-52.1", "RUN TASK blocked: physical Safety-Latch recovery did not complete. Check runtime log.")
            except Exception:
                pass
            return
        llm_background(
            "RUN TASK",
            _v238_run_task_after_posture,
            lock_motion=not cfg["dry_run"],
        )

    def llm_stop_cmd():
        if llm_agent is not None:
            llm_agent.request_stop()
        if skill_core is not None:
            try:
                skill_core.force_recover("NERO V2-7 AGENT STOP")
            except Exception as exc:
                append_gui_log(f"[LLM STOP WARN] {exc}")
        recover_request.value = True
        try:
            _v24_reset_skill_state("STOP + ARM RECOVER")
        except Exception as exc:
            append_gui_log(f"[V2-7 STATE RESET WARN] {exc}")
        llm_status_state["text"] = "Agent: STOP requested / arm recovery requested / skill state reset"

    def llm_reset_context_cmd():
        if llm_agent is None:
            return
        try:
            llm_agent.reset_context()
            llm_status_state["text"] = "LLM agent: context reset"
        except Exception as exc:
            append_gui_log(f"[LLM CONTEXT ERROR] {exc}")

    llm_control = ttk.LabelFrame(llm_outer, text="LLM agent controls")
    llm_control.pack(fill="x", pady=(0, 8))
    ttk.Button(llm_control, text="Test LLM API", command=llm_test_api_cmd).pack(side="left", padx=5, pady=8)
    ttk.Button(llm_control, text="Observe Scene", command=llm_observe_cmd).pack(side="left", padx=5, pady=8)
    ttk.Button(llm_control, text="Plan Only", command=llm_plan_only_cmd).pack(side="left", padx=5, pady=8)
    ttk.Button(llm_control, text="Estimate Cost", command=llm_estimate_cmd).pack(side="left", padx=5, pady=8)
    ttk.Button(llm_control, text="RUN TASK", command=llm_run_task_cmd).pack(side="left", padx=5, pady=8)
    ttk.Button(llm_control, text="STOP AGENT + ARM RECOVER", command=llm_stop_cmd).pack(side="left", padx=12, pady=8)
    ttk.Button(llm_control, text="Reset Agent Context", command=llm_reset_context_cmd).pack(side="left", padx=5, pady=8)

    # ----- V3.35 WRIST CAMERA MONITOR: what the camera sees + the labels of the actual perception result -----
    v335_mon = ttk.LabelFrame(llm_outer, text="Wrist camera monitor (V3.35) - live image; labels = the perception "
                                             "result the skills use (names, measured sizes, carton, slots)")
    v335_mon.pack(fill="x", pady=(0, 8))
    _z337 = os.environ.get("NERO_V337_MONITOR_ZOOM") or {"320": "50", "640": "100"}.get(
        os.environ.get("NERO_V335_MONITOR_SIZE", "640"), "100")
    v335_mon_state = {"capture": None, "img": None, "zoom": f"{int(float(_z337))}%",
                      "win": None, "win_canvas": None, "t_mtime": 0.0, "win_size": None}
    v335_canvas = tk.Canvas(v335_mon, width=640, height=360, bg="#101010", highlightthickness=0)
    v335_canvas.pack(side="left", padx=6, pady=6)
    v335_side = ttk.Frame(v335_mon)
    v335_side.pack(side="left", fill="both", expand=True, padx=(4, 8), pady=6)
    v335_mon_info = tk.StringVar(value="waiting for the first wrist capture (select a wrist camera as pose source)")
    v335_live_var = tk.BooleanVar(value=bool(V335_STATE.get("live_on")))
    v335_hz_var = tk.StringVar(value=f"{float(V335_STATE.get('live_hz', 1.5)):g}")
    V337_ZOOMS = ("25%", "50%", "75%", "100%", "125%", "150%", "200%", "300%")
    v335_size_var = tk.StringVar(value=v335_mon_state["zoom"] if v335_mon_state["zoom"] in V337_ZOOMS else "100%")
    v335_ctrl = ttk.Frame(v335_side)
    v335_ctrl.pack(anchor="w", fill="x")

    def _v335_on_live():
        V335_STATE["live_on"] = bool(v335_live_var.get())
        try:
            V335_STATE["live_hz"] = max(0.1, min(5.0, float(v335_hz_var.get())))
        except Exception:
            pass

    ttk.Checkbutton(v335_ctrl, text="Live refresh (scene re-perceived continuously)", variable=v335_live_var,
                    command=_v335_on_live).pack(side="left")
    ttk.Label(v335_ctrl, text="Hz").pack(side="left", padx=(8, 2))
    ttk.Spinbox(v335_ctrl, textvariable=v335_hz_var, from_=0.2, to=5.0, increment=0.1, width=5,
                command=_v335_on_live).pack(side="left")
    ttk.Label(v335_ctrl, text="Zoom").pack(side="left", padx=(10, 2))
    ttk.Combobox(v335_ctrl, textvariable=v335_size_var, values=V337_ZOOMS, state="readonly",
                 width=6).pack(side="left")

    def _v337_zoom_step(d):
        try:
            i = V337_ZOOMS.index(v335_size_var.get())
        except ValueError:
            i = V337_ZOOMS.index("100%")
        v335_size_var.set(V337_ZOOMS[max(0, min(len(V337_ZOOMS) - 1, i + d))])

    ttk.Button(v335_ctrl, text="-", width=2, command=lambda: _v337_zoom_step(-1)).pack(side="left", padx=(2, 0))
    ttk.Button(v335_ctrl, text="+", width=2, command=lambda: _v337_zoom_step(+1)).pack(side="left")
    v335_canvas.bind("<Control-MouseWheel>", lambda e: _v337_zoom_step(1 if e.delta > 0 else -1))
    v335_canvas.bind("<Control-Button-4>", lambda e: _v337_zoom_step(+1))     # X11 wheel up
    v335_canvas.bind("<Control-Button-5>", lambda e: _v337_zoom_step(-1))     # X11 wheel down

    def _v335_popout():
        if v335_mon_state["win"] is not None:
            try:
                v335_mon_state["win"].lift()
                return
            except Exception:
                v335_mon_state["win"] = None
        w = tk.Toplevel(root)
        w.title("NERO SimAgent wrist camera monitor (resize the window to zoom)")
        c = tk.Canvas(w, width=640, height=360, bg="#101010", highlightthickness=0)
        c.pack(fill="both", expand=True)
        c.bind("<Configure>", lambda e: v335_mon_state.update(win_size=(int(e.width), int(e.height))))
        v335_mon_state["win"], v335_mon_state["win_canvas"] = w, c
        v335_mon_state["capture"] = None

        def _closed():
            v335_mon_state["win"], v335_mon_state["win_canvas"] = None, None
            w.destroy()
        w.protocol("WM_DELETE_WINDOW", _closed)

    ttk.Button(v335_ctrl, text="Pop out (resizable)", command=_v335_popout).pack(side="left", padx=(10, 0))
    ttk.Label(v335_side, textvariable=v335_mon_info, justify="left", wraplength=640).pack(anchor="w", pady=(6, 0))

    _V337_FRAC = {0.25: (1, 4), 0.5: (1, 2), 0.75: (3, 4), 1.0: (1, 1), 1.25: (5, 4), 1.5: (3, 2), 2.0: (2, 1), 3.0: (3, 1)}

    def _v337_scaled(png, W0, H0, k):
        """V3.37 monitor zoom: PIL resize when available (any factor), else Tk zoom/subsample with the nearest of
        the fixed factors. Returns (image, factor actually used)."""
        try:
            from PIL import Image as _PI, ImageTk as _PITk
            wk, hk = max(16, int(W0 * k)), max(9, int(H0 * k))
            return _PITk.PhotoImage(_PI.open(str(png)).convert("RGB").resize((wk, hk), _PI.BILINEAR)), float(k)
        except Exception:
            pass
        kq = min(_V337_FRAC, key=lambda f: abs(f - float(k)))
        a, b = _V337_FRAC[kq]
        img = tk.PhotoImage(file=str(png))
        if a > 1:
            img = img.zoom(a, a)
        if b > 1:
            img = img.subsample(b, b)
        return img, float(kq)

    def _v335_draw(canvas, meta, png, size):
        """size: zoom factor (1.0 = native camera resolution); "fit" = fit the canvas (pop-out window)."""
        W0, H0 = int(meta.get("width", 640)), int(meta.get("height", 360))
        if size == "fit":
            cw, ch = v335_mon_state.get("win_size") or (640, 360)
            size = max(0.1, min(cw / max(1, W0), ch / max(1, H0)))
        img, k = _v337_scaled(png, W0, H0, float(size))
        canvas.delete("all")
        if canvas is v335_canvas:
            canvas.configure(width=int(W0 * k), height=int(H0 * k))
        canvas.create_image(0, 0, image=img, anchor="nw")
        for lab in meta.get("labels") or []:
            col = "#35e06a" if lab.get("trusted") else "#ffb020"
            if lab.get("kind") == "egg":
                x0, y0, x1, y1 = [float(v) * k for v in lab["bbox"]]
                if not lab.get("name"):
                    col = "#b0b0b0"
                canvas.create_rectangle(x0, y0, x1, y1, outline=col, width=2)
                canvas.create_text(x0 + 2, max(8, y0 - 8), text=lab.get("text", ""), fill=col, anchor="w",
                                   font=("Arial", 9, "bold"))
            elif lab.get("kind") == "holder":
                pts = [float(v) * k for p in lab["poly"] for v in p]
                canvas.create_polygon(*pts, outline="#30c8ff" if lab.get("trusted") else "#ffb020", fill="", width=2)
                canvas.create_text(pts[0], pts[1] - 8, text=lab.get("text", ""), fill="#30c8ff", anchor="w",
                                   font=("Arial", 9, "bold"))
            elif lab.get("kind") == "slot":
                x, y = [float(v) * k for v in lab["pt"]]
                canvas.create_oval(x - 3, y - 3, x + 3, y + 3, outline="#30c8ff", width=2)
                canvas.create_text(x + 5, y, text=lab.get("text", "").replace("slot_", "s"), fill="#30c8ff",
                                   anchor="w", font=("Arial", 8))
        return img

    def _v335_tick():
        try:
            if not V335_STATE.get("tool_running"):
                _v335_drain()                   # idle: fold the live captures in here (skills do it themselves)
            d = v335_live_dir()
            png, js = d / "live_view.png", d / "live_view.json"
            if js.exists() and png.exists():
                meta = json.loads(js.read_text(encoding="utf-8"))
                try:
                    size = float(str(v335_size_var.get() or "100%").rstrip("%")) / 100.0
                except ValueError:
                    size = 1.0
                win_sz = v335_mon_state.get("win_size")
                if (meta.get("capture") != v335_mon_state["capture"] or size != v335_mon_state.get("drawn_size")
                        or win_sz != v335_mon_state.get("drawn_win")):
                    if v335_canvas.winfo_ismapped() or v335_mon_state.get("img") is None:
                        v335_mon_state["img"] = _v335_draw(v335_canvas, meta, png, size)
                    if v335_mon_state["win_canvas"] is not None:
                        v335_mon_state["img_win"] = _v335_draw(v335_mon_state["win_canvas"], meta, png, "fit")
                    v335_mon_state["capture"] = meta.get("capture")
                    v335_mon_state["drawn_size"] = size
                    v335_mon_state["drawn_win"] = win_sz
                    eggs = [l_["text"] for l_ in meta.get("labels") or [] if l_.get("kind") == "egg"]
                    ih = meta.get("in_hand") or {}
                    age = time.time() - float(meta.get("t", time.time()))
                    v335_mon_info.set(
                        f"capture #{meta.get('capture')} [{meta.get('tag')}] {age:.1f}s ago | camera {meta.get('camera')} | "
                        f"calib level {meta.get('calib_level')}: {meta.get('calib')}\n"
                        f"eggs: {', '.join(eggs) if eggs else 'none detected'}\n"
                        f"carton: {'from this view' if meta.get('holder_from_this_view') else 'last known (not in this view)'}"
                        + (f" | in hand {ih.get('name')}: {'visible, shift ' + str(ih.get('shift_mm')) + ' mm' if ih.get('visible') else ih.get('reason')}"
                           if ih.get("name") else "")
                        + f" | live captures {V335_STATE.get('live_count', 0)}")
        except Exception as exc_m:
            v335_mon_info.set(f"monitor: {str(exc_m)[:160]}")
        try:
            root.after(int(os.environ.get("NERO_V335_MONITOR_MS", "400")), _v335_tick)
        except Exception:
            pass

    root.after(1500, _v335_tick)

    # ----- Manual V2 tool debugger: executes exactly the same callbacks LLM sees -----
    llm_debug = ttk.LabelFrame(llm_outer, text="V2-7 tool debugger — Skill Contracts + Failure-Aware Recovery")
    llm_debug.pack(fill="x", pady=(0, 8))
    llm_debug_object_var = tk.StringVar(value="egg_1")
    llm_debug_bin_var = tk.StringVar(value="slot_1")
    ttk.Label(llm_debug, text="Object").pack(side="left", padx=(8, 3), pady=8)
    ttk.Combobox(
        llm_debug, textvariable=llm_debug_object_var,
        values=OBJECTS, state="readonly", width=13,
    ).pack(side="left", padx=(0, 8), pady=8)
    ttk.Label(llm_debug, text="Egg-holder slot").pack(side="left", padx=(3, 3), pady=8)
    ttk.Combobox(
        llm_debug, textvariable=llm_debug_bin_var,
        values=tuple(EGG_SLOT_NAMES), state="readonly", width=11,
    ).pack(side="left", padx=(0, 8), pady=8)

    def llm_manual_tool(name, args_fn, motion=True):
        def cmd():
            args = args_fn()
            def run():
                result = llm_tool_runner(name, args)
                skill_core.emit(
                    f"[V2 MANUAL TOOL RESULT] {name} | "
                    + json.dumps(result, ensure_ascii=False)
                )
            llm_background(f"manual {name}", run, lock_motion=motion)
        return cmd

    ttk.Button(
        llm_debug, text="Prep Aperture",
        command=llm_manual_tool(
            "set_gripper_opening",
            lambda: {"target_opening_mm": _v24_gripper_prior(llm_debug_object_var.get())["recommended_target_opening_mm"]},
        ),
    ).pack(side="left", padx=3, pady=8)
    ttk.Button(
        llm_debug, text="Move Above",
        command=llm_manual_tool("move_above", lambda: {"object_name": llm_debug_object_var.get()}),
    ).pack(side="left", padx=3, pady=8)
    ttk.Button(
        llm_debug, text="Approach",
        command=llm_manual_tool("approach", lambda: {"object_name": llm_debug_object_var.get()}),
    ).pack(side="left", padx=3, pady=8)
    ttk.Button(
        llm_debug, text="Safety Recover 1x",
        command=llm_manual_tool(
            "recover_from_safety",
            lambda: {
                "object_name": llm_debug_object_var.get(),
                "strategy": "widen_aperture_and_retry_once",
                "diagnosis": "Manual debug: pre-grasp contact/safety stop; widen aperture and retry once",
            },
        ),
    ).pack(side="left", padx=3, pady=8)
    ttk.Button(
        llm_debug, text="Clamp 5N",
        command=llm_manual_tool("clamp_gripper", lambda: {"force_n": 5.0}),
    ).pack(side="left", padx=3, pady=8)
    ttk.Button(
        llm_debug, text="Lift 100mm",
        command=llm_manual_tool("lift", lambda: {"height_m": 0.10}),
    ).pack(side="left", padx=3, pady=8)
    ttk.Button(
        llm_debug, text="Move Above Slot",
        command=llm_manual_tool("move_above_receptacle", lambda: {"receptacle_name": "egg_holder", "slot_name": llm_debug_bin_var.get()}),
    ).pack(side="left", padx=3, pady=8)
    ttk.Button(
        llm_debug, text="Place Into Slot",
        command=llm_manual_tool("place_into_receptacle", lambda: {"receptacle_name": "egg_holder", "slot_name": llm_debug_bin_var.get()}),
    ).pack(side="left", padx=3, pady=8)
    ttk.Button(
        llm_debug, text="Release",
        command=llm_manual_tool("open_gripper", lambda: {}),
    ).pack(side="left", padx=3, pady=8)
    ttk.Button(
        llm_debug, text="Verify",
        command=llm_manual_tool(
            "verify_object_in_slot",
            lambda: {"object_name": llm_debug_object_var.get(), "slot_name": llm_debug_bin_var.get()},
            motion=False,
        ),
    ).pack(side="left", padx=3, pady=8)

    llm_util = ttk.LabelFrame(llm_outer, text="V2-7 Skill Contract / Safety Reflection diagnostics")
    llm_util.pack(fill="x", pady=(0, 8))

    def llm_show_tools_cmd():
        if not LLM_TOOLS:
            append_gui_log("[LLM TOOLS] Tool schema unavailable")
            return
        summary = [f"{t.get('name')}: {t.get('description')}" for t in LLM_TOOLS]
        append_gui_log("[LLM TOOL SCHEMA]\n  - " + "\n  - ".join(summary))

    ttk.Button(llm_util, text="Show Skill Contracts", command=llm_show_tools_cmd).pack(side="left", padx=5, pady=7)

    def llm_show_skill_state_cmd():
        append_gui_log("[V2-7 SKILL STATE] " + json.dumps(_v24_state_snapshot(), ensure_ascii=False))

    def llm_reset_skill_state_cmd():
        if llm_agent is not None and llm_agent.running:
            append_gui_log("[V2-7 STATE RESET BLOCKED] Agent is BUSY")
            return
        _v24_reset_skill_state("GUI manual reset")

    ttk.Button(llm_util, text="Show Skill State", command=llm_show_skill_state_cmd).pack(side="left", padx=5, pady=7)
    ttk.Button(llm_util, text="Reset Skill State", command=llm_reset_skill_state_cmd).pack(side="left", padx=5, pady=7)
    ttk.Button(llm_util, text="ZERO ALL", command=zero_all).pack(side="left", padx=5, pady=7)
    ttk.Button(llm_util, text="STOP + RECOVER", command=emergency_recover).pack(side="left", padx=5, pady=7)
    ttk.Button(llm_util, text="HARD RESET 0", command=hard_reset_zero).pack(side="left", padx=5, pady=7)

    # ----- V3.37 COLLAPSIBLE SECTIONS: every section of the NERO SimAgent tab folds with a click on its title -----
    V337_FOLD = {}

    def _v337_fold(frame, collapse):
        st = V337_FOLD[frame]
        if collapse == st["collapsed"]:
            return
        if collapse:
            saved = []
            for ch in frame.pack_slaves():
                saved.append(("pack", ch, ch.pack_info()))
            for ch in frame.grid_slaves():
                saved.append(("grid", ch, None))
            for ch in frame.place_slaves():
                saved.append(("place", ch, ch.place_info()))
            for mgr, ch, info in saved:
                if mgr == "pack":
                    ch.pack_forget()
                elif mgr == "grid":
                    ch.grid_remove()
                else:
                    ch.place_forget()
            st["saved"] = saved
            st["ph"] = tk.Frame(frame, height=1, width=1)      # Tk keeps the old size of an empty master
            st["ph"].pack(anchor="w")
        else:
            if st.get("ph") is not None:
                try:
                    st["ph"].destroy()
                except Exception:
                    pass
                st["ph"] = None
            for mgr, ch, info in (st.get("saved") or []):
                try:
                    if mgr == "pack":
                        info = {k: v for k, v in dict(info).items() if k != "in"}
                        ch.pack(in_=frame, **info)
                    elif mgr == "grid":
                        ch.grid()
                    else:
                        info = {k: v for k, v in dict(info).items() if k != "in"}
                        ch.place(in_=frame, **info)
                except Exception:
                    pass
            st["saved"] = []
        st["collapsed"] = bool(collapse)
        st["label"].configure(text=("\u25b6 " if collapse else "\u25bc ") + st["title"])

    def _v337_make_collapsible(frame):
        try:
            title = str(frame.cget("text") or "")
        except Exception:
            return
        lab = ttk.Label(frame, text="\u25bc " + title, cursor="hand2", font=("Arial", 9, "bold"))
        frame.configure(labelwidget=lab)
        V337_FOLD[frame] = {"title": title, "label": lab, "collapsed": False, "saved": []}
        lab.bind("<Button-1>", lambda _e, f=frame: _v337_fold(f, not V337_FOLD[f]["collapsed"]))

    _v337_sections = [w for w in llm_outer.winfo_children() if isinstance(w, ttk.LabelFrame)]
    for _w337 in _v337_sections:
        _v337_make_collapsible(_w337)
    _v337_bar = ttk.Frame(llm_outer)
    if _v337_sections:
        _v337_bar.pack(before=_v337_sections[0], anchor="w", pady=(0, 4))
    ttk.Button(_v337_bar, text="Collapse all", command=lambda: [_v337_fold(f, True) for f in _v337_sections]).pack(side="left")
    ttk.Button(_v337_bar, text="Expand all", command=lambda: [_v337_fold(f, False) for f in _v337_sections]).pack(side="left", padx=(6, 0))
    ttk.Label(_v337_bar, text="(click a section title to fold / unfold it)").pack(side="left", padx=(8, 0))
    _v337_start = [t.strip().lower() for t in os.environ.get("NERO_V337_COLLAPSED", "").split(",") if t.strip()]
    for _w337 in _v337_sections:
        if any(k in V337_FOLD[_w337]["title"].lower() for k in _v337_start):
            _v337_fold(_w337, True)

    def poll_llm_status():
        try:
            if llm_agent is None:
                llm_status_var.set(f"LLM agent ERROR: {llm_agent_error}")
            else:
                try:
                    llm_agent.provider = _v342_by_label.get(llm_provider_var.get())
                    key_ok, key_text = llm_agent.api_key_status(llm_api_key_var.get(), llm_model_var.get().strip())
                except TypeError:
                    key_ok, key_text = llm_agent.api_key_status(llm_api_key_var.get())
                running_text = "BUSY" if llm_agent.running else "IDLE"
                extra = llm_status_state.get("text", "")
                llm_status_var.set(
                    f"Agent: {running_text} | model={llm_model_var.get().strip() or 'gpt-6.1-sol'} | "
                    f"API={'READY' if key_ok else 'NOT CONFIGURED'} | skill={_v24_state_snapshot().get('stage')} | {extra}"
                )
        except Exception as exc:
            llm_status_var.set(f"LLM status error: {exc}")
        finally:
            try:
                root.after(300, poll_llm_status)
            except Exception:
                pass

    root.after(300, poll_llm_status)

    # ========================================================
    # V69 BIOIK PRIMARY TAB
    # ========================================================
    bio_outer = ttk.Frame(bio_tab)
    bio_outer.pack(fill="both", expand=True, padx=12, pady=12)

    ttk.Label(
        bio_outer,
        text="BioIK Primary Planner",
        font=("Arial", 18, "bold"),
    ).pack(anchor="w", pady=(0, 5))

    ttk.Label(
        bio_outer,
        text=(
            "Primary solver: BioIK local optimizer (gd_c) with seed-near / joint-limit "
            "regularization. Move Above Bin uses BioIK incremental continuation first and "
            "the independent TRAC-IK service only as a bounded fallback."
        ),
        wraplength=1250,
        justify="left",
    ).pack(anchor="w", pady=(0, 5))

    bio_status_var = tk.StringVar(value="BioIK status: waiting...")
    ttk.Label(
        bio_outer, textvariable=bio_status_var, font=("Arial", 10, "bold")
    ).pack(anchor="w", pady=(0, 8))

    bio_params = ttk.LabelFrame(bio_outer, text="BioIK target / seed / request")
    bio_params.pack(fill="x", pady=(0, 8))

    bio_object_var = tk.StringVar(value="egg_1")
    bio_height_mm_var = tk.StringVar(value="120")
    bio_yaw_deg_var = tk.StringVar(value="0")
    bio_timeout_var = tk.StringVar(value="0.05")
    bio_runs_var = tk.StringVar(value="20")
    bio_seed_var = tk.StringVar(value="current")
    bio_collision_var = tk.BooleanVar(value=True)

    ttk.Label(bio_params, text="Object").grid(row=0, column=0, padx=(8,2), pady=8)
    ttk.Combobox(
        bio_params, textvariable=bio_object_var, values=OBJECTS,
        state="readonly", width=19,
    ).grid(row=0, column=1, padx=(0,10), pady=8)
    ttk.Label(bio_params, text="Above mm").grid(row=0, column=2, padx=(3,2), pady=8)
    ttk.Entry(bio_params, textvariable=bio_height_mm_var, width=8).grid(row=0, column=3, padx=(0,10), pady=8)
    ttk.Label(bio_params, text="Yaw deg").grid(row=0, column=4, padx=(3,2), pady=8)
    ttk.Entry(bio_params, textvariable=bio_yaw_deg_var, width=8).grid(row=0, column=5, padx=(0,10), pady=8)
    ttk.Label(bio_params, text="Seed").grid(row=0, column=6, padx=(3,2), pady=8)
    ttk.Combobox(
        bio_params, textvariable=bio_seed_var,
        values=("current", "zero", "bent"), state="readonly", width=10,
    ).grid(row=0, column=7, padx=(0,10), pady=8)
    ttk.Label(bio_params, text="Request timeout s").grid(row=0, column=8, padx=(3,2), pady=8)
    ttk.Entry(bio_params, textvariable=bio_timeout_var, width=8).grid(row=0, column=9, padx=(0,10), pady=8)
    ttk.Label(bio_params, text="Benchmark runs").grid(row=1, column=0, padx=(8,2), pady=(0,8))
    ttk.Entry(bio_params, textvariable=bio_runs_var, width=8).grid(row=1, column=1, padx=(0,10), pady=(0,8), sticky="w")
    ttk.Checkbutton(
        bio_params, text="avoid_collisions", variable=bio_collision_var,
    ).grid(row=1, column=2, columnspan=2, padx=(3,10), pady=(0,8), sticky="w")
    ttk.Label(
        bio_params,
        text="BioIK YAML: gd_c + minimal displacement + avoid joint limits + center joints",
    ).grid(row=1, column=4, columnspan=6, padx=(3,8), pady=(0,8), sticky="w")

    def bio_values():
        try:
            h = float(bio_height_mm_var.get()) / 1000.0
            yaw = float(bio_yaw_deg_var.get())
            timeout_s = float(bio_timeout_var.get())
            runs = int(bio_runs_var.get())
        except Exception as exc:
            if skill_core is not None:
                skill_core.emit(f"[BIO] invalid parameter: {exc}")
            return None
        if h <= 0.0 or timeout_s <= 0.0 or runs <= 0:
            if skill_core is not None:
                skill_core.emit("[BIO] height, timeout and runs must be > 0")
            return None
        return h, yaw, timeout_s, runs

    bio_diag = ttk.LabelFrame(bio_outer, text="BioIK diagnostics")
    bio_diag.pack(fill="x", pady=(0,8))

    ttk.Button(
        bio_diag, text="Observe Scene",
        command=lambda: submit_skill("observe_scene"),
    ).pack(side="left", padx=5, pady=8)

    def bio_self_test_cmd():
        vals = bio_values()
        if vals is None: return
        submit_skill(
            "bio_self_test", bio_seed_var.get(), bool(bio_collision_var.get()), vals[2]
        )

    def bio_probe_cmd():
        vals = bio_values()
        if vals is None: return
        submit_skill(
            "bio_object_probe", bio_object_var.get(), vals[0], vals[1],
            bio_seed_var.get(), bool(bio_collision_var.get()), vals[2]
        )

    def bio_bench_cmd():
        vals = bio_values()
        if vals is None: return
        submit_skill(
            "bio_benchmark", bio_object_var.get(), vals[0], vals[1],
            bio_seed_var.get(), bool(bio_collision_var.get()), vals[2], vals[3]
        )

    ttk.Button(bio_diag, text="Self-Test Current TCP", command=bio_self_test_cmd).pack(side="left", padx=5, pady=8)
    ttk.Button(bio_diag, text="Probe Object IK", command=bio_probe_cmd).pack(side="left", padx=5, pady=8)
    ttk.Button(bio_diag, text="Benchmark IK", command=bio_bench_cmd).pack(side="left", padx=5, pady=8)

    bio_skill = ttk.LabelFrame(bio_outer, text="BioIK manipulation")
    bio_skill.pack(fill="x", pady=(0,8))

    def bio_move_above_cmd():
        vals = bio_values()
        if vals is None: return
        submit_skill("bio_move_above", bio_object_var.get(), vals[0])

    ttk.Button(bio_skill, text="Move Above (BIO)", command=bio_move_above_cmd).pack(side="left", padx=5, pady=8)
    ttk.Button(
        bio_skill, text="Approach (BIO)",
        command=lambda: submit_skill("bio_approach", bio_object_var.get()),
    ).pack(side="left", padx=5, pady=8)
    ttk.Button(
        bio_skill, text="Gripper FAST CLAMP", command=one_click_clamp,
    ).pack(side="left", padx=5, pady=8)
    ttk.Button(
        bio_skill, text="Lift 100 mm", command=lambda: submit_skill("bio_lift", 0.10),
    ).pack(side="left", padx=5, pady=8)

    bio_bin_row = ttk.Frame(bio_outer)
    bio_bin_row.pack(fill="x", pady=(0,8))
    ttk.Label(bio_bin_row, text="Release bin:").pack(side="left")
    bio_bin_var = tk.StringVar(value="red_bin")
    ttk.Combobox(
        bio_bin_row, textvariable=bio_bin_var,
        values=("red_bin", "blue_bin"), state="readonly", width=14,
    ).pack(side="left", padx=(5,10))
    ttk.Button(
        bio_bin_row, text="Move Above Bin (BIO; stop at safe frontier if blocked)",
        command=lambda: submit_skill("bio_move_above_bin", bio_bin_var.get()),
    ).pack(side="left", padx=5, pady=5)
    ttk.Button(
        bio_bin_row, text="Gripper OPEN", command=one_click_release,
    ).pack(side="left", padx=5, pady=5)

    bio_util = ttk.LabelFrame(bio_outer, text="BioIK tab utilities")
    bio_util.pack(fill="x", pady=(0,8))
    ttk.Button(bio_util, text="ZERO ALL", command=zero_all).pack(side="left", padx=6, pady=7)
    ttk.Button(bio_util, text="STOP + RECOVER", command=emergency_recover).pack(side="left", padx=6, pady=7)
    ttk.Button(bio_util, text="HARD RESET 0", command=hard_reset_zero).pack(side="left", padx=6, pady=7)

    # ========================================================
    # V53 TRAC-IK LAB TAB
    # ========================================================
    trac_outer = ttk.Frame(trac_tab)
    trac_outer.pack(fill="both", expand=True, padx=12, pady=12)

    ttk.Label(
        trac_outer,
        text="TRAC-IK Solver Lab",
        font=("Arial", 18, "bold"),
    ).pack(anchor="w", pady=(0, 5))

    ttk.Label(
        trac_outer,
        text=(
            "TRAC-IK runs in an independent /trac_ref MoveIt instance. "
            "This keeps the primary BioIK stack untouched while providing a true "
            "solver baseline and bounded fallback."
        ),
        wraplength=1250,
        justify="left",
    ).pack(anchor="w", pady=(0, 5))

    trac_solver_name = os.environ.get("NERO_TRAC_IK_SOLVER", "TRAC-IK")
    trac_solve_type = os.environ.get("NERO_TRAC_IK_SOLVE_TYPE", "unknown")
    trac_config_timeout = os.environ.get("NERO_TRAC_IK_TIMEOUT", "unknown")

    trac_status_var = tk.StringVar(
        value=(
            f"Configured solver: {trac_solver_name} | "
            f"solve_type={trac_solve_type} | "
            f"config timeout={trac_config_timeout}s"
        )
    )
    ttk.Label(
        trac_outer,
        textvariable=trac_status_var,
        font=("Arial", 10, "bold"),
    ).pack(anchor="w", pady=(0, 8))

    trac_params = ttk.LabelFrame(
        trac_outer,
        text="IK test target / seed / request",
    )
    trac_params.pack(fill="x", pady=(0, 8))

    trac_object_var = tk.StringVar(value="egg_1")
    trac_height_mm_var = tk.StringVar(value="120")
    trac_yaw_deg_var = tk.StringVar(value="0")
    trac_timeout_var = tk.StringVar(value="0.10")
    trac_runs_var = tk.StringVar(value="20")
    trac_seed_var = tk.StringVar(value="current")
    trac_collision_var = tk.BooleanVar(value=True)

    ttk.Label(trac_params, text="Object").grid(
        row=0, column=0, padx=(8, 2), pady=8, sticky="w"
    )
    ttk.Combobox(
        trac_params,
        textvariable=trac_object_var,
        values=OBJECTS,
        state="readonly",
        width=19,
    ).grid(row=0, column=1, padx=(0, 10), pady=8)

    ttk.Label(trac_params, text="Above mm").grid(
        row=0, column=2, padx=(3, 2), pady=8
    )
    ttk.Entry(
        trac_params,
        textvariable=trac_height_mm_var,
        width=8,
    ).grid(row=0, column=3, padx=(0, 10), pady=8)

    ttk.Label(trac_params, text="Yaw deg").grid(
        row=0, column=4, padx=(3, 2), pady=8
    )
    ttk.Entry(
        trac_params,
        textvariable=trac_yaw_deg_var,
        width=8,
    ).grid(row=0, column=5, padx=(0, 10), pady=8)

    ttk.Label(trac_params, text="Seed").grid(
        row=0, column=6, padx=(3, 2), pady=8
    )
    ttk.Combobox(
        trac_params,
        textvariable=trac_seed_var,
        values=("current", "zero", "bent"),
        state="readonly",
        width=10,
    ).grid(row=0, column=7, padx=(0, 10), pady=8)

    ttk.Label(trac_params, text="Request timeout s").grid(
        row=0, column=8, padx=(3, 2), pady=8
    )
    ttk.Entry(
        trac_params,
        textvariable=trac_timeout_var,
        width=8,
    ).grid(row=0, column=9, padx=(0, 10), pady=8)

    ttk.Label(trac_params, text="Benchmark runs").grid(
        row=1, column=0, padx=(8, 2), pady=(0, 8), sticky="w"
    )
    ttk.Entry(
        trac_params,
        textvariable=trac_runs_var,
        width=8,
    ).grid(row=1, column=1, padx=(0, 10), pady=(0, 8), sticky="w")

    ttk.Checkbutton(
        trac_params,
        text="avoid_collisions",
        variable=trac_collision_var,
    ).grid(
        row=1, column=2, columnspan=2,
        padx=(3, 10), pady=(0, 8), sticky="w"
    )

    ttk.Label(
        trac_params,
        text=(
            "Seeds: current = live J1..J7; zero = all zero; "
            "bent = [0, 0.4, 0, 0.2, 0, 0, 0]"
        ),
    ).grid(
        row=1, column=4, columnspan=6,
        padx=(3, 8), pady=(0, 8), sticky="w"
    )

    def trac_values():
        try:
            h = float(trac_height_mm_var.get()) / 1000.0
            yaw = float(trac_yaw_deg_var.get())
            timeout_s = float(trac_timeout_var.get())
            runs = int(trac_runs_var.get())
        except Exception as exc:
            if skill_core is not None:
                skill_core.emit(f"[TRAC] invalid parameter: {exc}")
            return None

        if h <= 0.0 or timeout_s <= 0.0 or runs <= 0:
            if skill_core is not None:
                skill_core.emit(
                    "[TRAC] height, timeout and runs must be > 0"
                )
            return None

        return h, yaw, timeout_s, runs

    trac_actions = ttk.LabelFrame(
        trac_outer,
        text="TRAC-IK diagnostics",
    )
    trac_actions.pack(fill="x", pady=(0, 8))

    def trac_self_test_cmd():
        vals = trac_values()
        if vals is None:
            return
        _, _, timeout_s, _ = vals
        submit_skill(
            "trac_self_test",
            trac_seed_var.get(),
            bool(trac_collision_var.get()),
            timeout_s,
        )

    def trac_probe_cmd():
        vals = trac_values()
        if vals is None:
            return
        h, yaw, timeout_s, _ = vals
        submit_skill(
            "trac_object_probe",
            trac_object_var.get(),
            h,
            yaw,
            trac_seed_var.get(),
            bool(trac_collision_var.get()),
            timeout_s,
        )

    def trac_bench_cmd():
        vals = trac_values()
        if vals is None:
            return
        h, yaw, timeout_s, runs = vals
        submit_skill(
            "trac_benchmark",
            trac_object_var.get(),
            h,
            yaw,
            trac_seed_var.get(),
            bool(trac_collision_var.get()),
            timeout_s,
            runs,
        )

    ttk.Button(
        trac_actions,
        text="Observe Scene",
        command=lambda: submit_skill("observe_scene"),
    ).pack(side="left", padx=5, pady=8)

    ttk.Button(
        trac_actions,
        text="Self-Test Current TCP",
        command=trac_self_test_cmd,
    ).pack(side="left", padx=5, pady=8)

    ttk.Button(
        trac_actions,
        text="Probe Object IK",
        command=trac_probe_cmd,
    ).pack(side="left", padx=5, pady=8)

    ttk.Button(
        trac_actions,
        text="Benchmark IK",
        command=trac_bench_cmd,
    ).pack(side="left", padx=5, pady=8)

    def trac_move_above_cmd():
        vals = trac_values()
        if vals is None:
            return
        h = vals[0]
        submit_skill(
            "trac_move_above",
            trac_object_var.get(),
            h,
        )

    ttk.Button(
        trac_actions,
        text="Move Above (TRAC)",
        command=trac_move_above_cmd,
    ).pack(side="left", padx=5, pady=8)

    ttk.Button(
        trac_actions,
        text="Approach (TRAC)",
        command=lambda: submit_skill(
            "trac_approach",
            trac_object_var.get(),
        ),
    ).pack(side="left", padx=5, pady=8)

    ttk.Button(
        trac_actions,
        text="Lift 100 mm",
        command=lambda: submit_skill(
            "trac_lift", 0.10
        ),
    ).pack(side="left", padx=5, pady=8)


    trac_bin_row = ttk.Frame(trac_outer)
    trac_bin_row.pack(fill="x", pady=(0, 8))
    ttk.Label(trac_bin_row, text="Release bin:").pack(side="left")
    trac_bin_var = tk.StringVar(value="red_bin")
    ttk.Combobox(
        trac_bin_row,
        textvariable=trac_bin_var,
        values=("red_bin", "blue_bin"),
        state="readonly",
        width=14,
    ).pack(side="left", padx=(5, 10))
    ttk.Button(
        trac_bin_row,
        text="Move Above Bin (stop at safe frontier)",
        command=lambda: submit_skill(
            "trac_move_above_bin", trac_bin_var.get()
        ),
    ).pack(side="left", padx=5, pady=5)
    ttk.Label(
        trac_bin_row,
        text="If blocked, V69 moves only to the last validated state for inspection.",
    ).pack(side="left", padx=(8, 0))

    trac_util = ttk.LabelFrame(
        trac_outer,
        text="Existing robot utilities",
    )
    trac_util.pack(fill="x", pady=(0, 8))

    ttk.Button(
        trac_util,
        text="Gripper OPEN",
        command=one_click_release,
    ).pack(side="left", padx=6, pady=7)

    ttk.Button(
        trac_util,
        text="Gripper FAST CLAMP",
        command=one_click_clamp,
    ).pack(side="left", padx=6, pady=7)

    ttk.Button(
        trac_util,
        text="ZERO ALL",
        command=zero_all,
    ).pack(side="left", padx=6, pady=7)

    ttk.Button(
        trac_util,
        text="STOP + RECOVER",
        command=emergency_recover,
    ).pack(side="left", padx=6, pady=7)

    ttk.Button(
        trac_util,
        text="HARD RESET 0",
        command=hard_reset_zero,
    ).pack(side="left", padx=6, pady=7)

    ttk.Label(
        trac_outer,
        text=(
            "Best first diagnosis: Self-Test Current TCP with collision OFF, "
            "then ON. Next Probe Object IK. Only after probes succeed, run Move Above."
        ),
        wraplength=1250,
        justify="left",
    ).pack(anchor="w", pady=(0, 5))

    ttk.Label(
        trac_outer,
        text="All TRAC / skill / recovery logs are shown in the unified log panel below the tabs.",
        font=("Arial", 10, "bold"),
    ).pack(anchor="w", pady=(4, 8))

    ttk.Label(
        skill_outer,
        text=(
            "Execution path: MoveIt planning → official arm_controller "
            "FollowJointTrajectory → arm_controller/controller_state → V31 MuJoCo controller"
        ),
    ).pack(anchor="w", pady=(0, 5))

    ttk.Label(
        skill_outer,
        text="All runtime logs are shown in the unified log panel below the tabs.",
        font=("Arial", 10, "bold"),
    ).pack(anchor="w", pady=(4, 8))

    # Launcher writes every startup/process/interface check to this file.
    # The GUI tails it so the user never has to guess whether ROS2,
    # MoveIt, controllers, services or RViz actually started.
    startup_status_file = Path(
        os.environ.get(
            "NERO_STATUS_FILE",
            str(
                Path.home()
                / "mujoco_ws"
                / "nero"
                / "logs"
                / "v62_startup_status.log"
            ),
        )
    )
    startup_status_state = {
        "offset": 0,
        "missing_reported": False,
    }

    def tail_startup_status():
        try:
            if not startup_status_file.exists():
                if not startup_status_state["missing_reported"]:
                    append_gui_log(
                        "[WAIT] Launcher status file not created yet: "
                        + str(startup_status_file)
                    )
                    startup_status_state["missing_reported"] = True
                return

            with startup_status_file.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as f:
                f.seek(startup_status_state["offset"])
                chunk = f.read()
                startup_status_state["offset"] = f.tell()

            if chunk:
                for line in chunk.splitlines():
                    if line.strip():
                        append_gui_log(line)

        except Exception as exc:
            append_gui_log(
                "[WARN] Could not read launcher status log: "
                + str(exc)
            )

    skill_poll_state = {
        "last_safety_trip": int(safety_trip_count.value),
        "last_recovered_trip": int(safety_trip_count.value),
        "safety_latch_held": False,  # V2-52.1 physical-latch mirror
        "sim_ready_logged": False,
        "ros_bridge_logged": False,
        "move_ready_logged": False,
        "arm_ready_logged": False,
        "arm_was_ready": False,
        "cart_ready_logged": False,
        "all_skill_interfaces_logged": False,
    }

    # V2-52.1 RUN TASK SAFETY-LATCH GATE ONLY
    def _v2521_run_task_safety_gate():
        """Auto-clear any stale physical MuJoCo safety latch before RUN TASK.

        The true ros_motion_fault_latched flag is process-local inside MuJoCo.
        We therefore treat every unrecovered safety_trip_count increment as a
        held latch.  recover_request consumption is the MuJoCo-side ACK: in the
        simulator loop that request is consumed in the same recovery branch that
        clears ros_motion_fault_latched and freezes the arm at its current pose.

        No task/planning/manipulation behavior is changed here.  This gate only:
          1) blocks the new RUN TASK handoff,
          2) cancels stale arm motion and keeps the ROS arm gate OFF,
          3) requests STOP+RECOVER and waits for MuJoCo ACK,
          4) rejects recovery if a new safety trip appears,
          5) syncs ros2_control to the actual MuJoCo joints,
          6) releases this gate so the unchanged V2-39 task-start posture check
             runs next inside the normal Agent task-start path.
        """
        trip_now521 = int(safety_trip_count.value)
        recovered_trip521 = int(skill_poll_state.get("last_recovered_trip", trip_now521))
        pending521 = (
            bool(skill_poll_state.get("safety_latch_held", False))
            or trip_now521 != recovered_trip521
        )
        if not pending521:
            return True

        skill_poll_state["safety_latch_held"] = True
        append_gui_log(
            f"[V2-52.1 SAFETY-LATCH GATE] BLOCK new RUN TASK | trips={trip_now521} | "
            "auto STOP+RECOVER before unchanged task-start posture check"
        )
        try:
            # Stop/cancel anything left by the failed previous task.  Keep the
            # physical ROS command gate OFF throughout recovery and state sync.
            if skill_core is not None:
                try:
                    skill_core.cancel_active_motion()
                except Exception:
                    pass
                try:
                    skill_core.force_recover("V2-52.1 RUN TASK SAFETY GATE")
                except Exception as exc521:
                    append_gui_log(f"[V2-52.1 SAFETY-LATCH WARN] force_recover: {exc521}")
                try:
                    skill_core._set_arm_command_gate(False)
                except Exception:
                    ros_arm_command_gate_shared.value = False
            else:
                ros_arm_command_gate_shared.value = False

            # Actual MuJoCo latch-clear request.  Do not infer latch state from
            # safety_state==0; that state can decay to NORMAL while the internal
            # ros_motion_fault_latched flag is still held.  Consumption of this
            # request is the explicit recovery handshake.
            recover_request.value = True
            deadline521 = time.perf_counter() + 3.0
            while bool(recover_request.value) and time.perf_counter() < deadline521:
                time.sleep(0.02)
            if bool(recover_request.value):
                append_gui_log(
                    "[V2-52.1 SAFETY-LATCH GATE FAIL] MuJoCo did not acknowledge STOP+RECOVER"
                )
                return False

            # Allow the MuJoCo recovery branch to finish after it consumes the
            # shared request, then make sure recovery itself did not retrip.
            trip_after_ack521 = int(safety_trip_count.value)
            time.sleep(0.20)
            trip_after_settle521 = int(safety_trip_count.value)
            if trip_after_settle521 != trip_after_ack521:
                append_gui_log(
                    f"[V2-52.1 SAFETY-LATCH GATE FAIL] new safety trip during recovery "
                    f"{trip_after_ack521}->{trip_after_settle521}"
                )
                return False

            # Required controller re-anchor to the *actual* frozen MuJoCo pose.
            if skill_core is None:
                append_gui_log(
                    "[V2-52.1 SAFETY-LATCH GATE FAIL] skill core unavailable for joint sync"
                )
                return False
            skill_core._sync_arm_controller_to_feedback()

            # Mark this trip count as explicitly recovered.  The ROS arm gate is
            # intentionally left OFF here; unchanged V2-52 motion execution owns
            # re-enabling it when a verified trajectory is actually executed.
            skill_poll_state["last_recovered_trip"] = int(safety_trip_count.value)
            skill_poll_state["last_safety_trip"] = int(safety_trip_count.value)
            skill_poll_state["safety_latch_held"] = False
            append_gui_log(
                f"[V2-52.1 SAFETY-LATCH CLEARED] STOP+RECOVER ACK | "
                f"trips={int(safety_trip_count.value)} | ros2_control synced to actual MuJoCo joints | "
                "continuing to unchanged V2-39 task-start posture check"
            )
            return True
        except Exception as exc521:
            skill_poll_state["safety_latch_held"] = True
            append_gui_log(f"[V2-52.1 SAFETY-LATCH GATE FAIL] {exc521}")
            return False


    initial_synced = {"done": False}

    def drain_runtime_log_watchdog():
        """Independent log pump; survives failures in the main GUI telemetry poll."""
        try:
            if skill_core is not None:
                for line in skill_core.drain_logs(limit=500):
                    append_gui_log(line)
        except Exception as exc:
            append_gui_log(f"[LOG WATCHDOG ERROR] {exc}")
        finally:
            try:
                root.after(80, drain_runtime_log_watchdog)
            except Exception:
                pass

    def poll():
        # First mirror launcher/process/interface startup checks into GUI.
        tail_startup_status()

        # Log local MuJoCo/bridge readiness once.
        if bool(sim_ready.value) and not skill_poll_state["sim_ready_logged"]:
            append_gui_log(
                "[OK] MuJoCo simulator process READY "
                f"(PID {sim_process.pid})"
            )
            skill_poll_state["sim_ready_logged"] = True

        if (
            bool(ros_bridge_ready.value)
            and not skill_poll_state["ros_bridge_logged"]
        ):
            append_gui_log(
                "[OK] MuJoCo ↔ ROS2 bridge READY | "
                "/feedback/joint_states publisher active | "
                "/arm_controller/controller_state primary bridge active; /control/joint_states fallback active"
            )
            skill_poll_state["ros_bridge_logged"] = True

        control_mode_var.set(
            "Arm input: MOVEIT EXECUTING"
            if bool(ros_arm_command_gate_shared.value)
            else "Arm input: MANUAL/IDLE"
        )

        # Unified MoveIt skill status/log.
        if skill_core is not None:
            move_ready, arm_ready, cart_ready = skill_core.readiness()

            # V69 runtime watchdog: if the action server disappears after being
            # ready, surface it immediately and block new motion requests.
            if skill_poll_state["arm_was_ready"] and not arm_ready:
                append_gui_log(
                    "[CRITICAL] Root arm_controller action server disappeared: "
                    "/arm_controller/follow_joint_trajectory. Motion buttons are blocked "
                    "until the controller is restored/restarted."
                )
                skill_poll_state["arm_ready_logged"] = False
                skill_poll_state["all_skill_interfaces_logged"] = False
            skill_poll_state["arm_was_ready"] = bool(arm_ready)

            if move_ready and not skill_poll_state["move_ready_logged"]:
                append_gui_log(
                    "[OK] MoveIt action READY: /move_action"
                )
                skill_poll_state["move_ready_logged"] = True

            if arm_ready and not skill_poll_state["arm_ready_logged"]:
                append_gui_log(
                    "[OK] Arm controller action READY: "
                    "/arm_controller/follow_joint_trajectory"
                )
                skill_poll_state["arm_ready_logged"] = True

            if cart_ready and not skill_poll_state["cart_ready_logged"]:
                append_gui_log(
                    "[OK] Cartesian service READY: "
                    "/compute_cartesian_path"
                )
                skill_poll_state["cart_ready_logged"] = True

            if (
                move_ready
                and arm_ready
                and cart_ready
                and not skill_poll_state["all_skill_interfaces_logged"]
            ):
                append_gui_log(
                    "[READY] V69 skill interfaces are fully connected"
                )
                skill_poll_state[
                    "all_skill_interfaces_logged"
                ] = True
            skill_ready_var.set(
                "MoveIt %s | arm_controller %s | Cartesian %s | %s"
                % (
                    "READY" if move_ready else "WAIT",
                    "READY" if arm_ready else "WAIT",
                    "READY" if cart_ready else "WAIT",
                    "BUSY" if skill_core.busy else "IDLE",
                )
            )

            bio_ik_ready, bio_solver, bio_mode, bio_cfg_timeout, bio_weights = (
                skill_core.bio_readiness()
            )
            bio_status_var.set(
                "Configured solver: %s | mode=%s | config timeout=%ss | %s | /compute_ik %s | %s"
                % (
                    bio_solver, bio_mode, bio_cfg_timeout, bio_weights,
                    "READY" if bio_ik_ready else "WAIT",
                    "BUSY" if skill_core.busy else "IDLE",
                )
            )

            trac_ik_ready, trac_solver, trac_type, trac_cfg_timeout = (
                skill_core.trac_readiness()
            )
            trac_status_var.set(
                "Configured solver: %s | solve_type=%s | "
                "config timeout=%ss | /trac_ref/compute_ik %s | %s"
                % (
                    trac_solver,
                    trac_type,
                    trac_cfg_timeout,
                    "READY" if trac_ik_ready else "WAIT",
                    "BUSY" if skill_core.busy else "IDLE",
                )
            )

            # V69: automatic safety events are NON-DESTRUCTIVE AND LATCHED.
            # Cancel the active trajectory and gate commands OFF, but DO NOT
            # clear the MuJoCo safety latch automatically.
            trip_now = int(safety_trip_count.value)
            if trip_now != skill_poll_state["last_safety_trip"]:
                skill_poll_state["last_safety_trip"] = trip_now
                skill_poll_state["safety_latch_held"] = True  # V2-52.1; clear only after explicit MuJoCo recover ACK
                try:
                    skill_core.cancel_active_motion()
                except Exception:
                    pass
                # V69: DO NOT set recover_request here.  An automatic safety
                # event must remain latched until the user explicitly presses
                # STOP + RECOVER (or HARD RESET 0).  This prevents stale
                # FollowJointTrajectory commands from re-entering immediately.
                ros_arm_command_gate_shared.value = False
                state_code_now = int(safety_state.value)
                state_text_now = {
                    1: "PROTECTED-SURFACE PENETRATION",
                    4: "RUNAWAY",
                }.get(state_code_now, f"STATE {state_code_now}")
                skill_core.emit(
                    f"[SAFETY STOP #{trip_now}] {state_text_now}; "
                    "arm frozen in place; gripper/grasp preserved; SAFETY LATCH HELD — press STOP + RECOVER"
                )

        # V42 MoveIt2 bridge telemetry.
        if bool(ros_bridge_ready.value):
            age = float(ros_last_cmd_age.value)
            age_text = (
                "no command yet"
                if age < 0.0
                else f"last cmd {age:.2f}s ago"
            )
            moveit_status_var.set(
                "ROS2 bridge: READY | "
                f"commands={int(ros_cmd_count.value)} "
                f"({int(ros_last_cmd_joint_count.value)} joints last) | "
                f"ctrl_state_rx={int(ros_controller_state_rx_shared.value)} | "
                f"fallback_rx={int(ros_fallback_joint_state_rx_shared.value)} | "
                f"src={ {0: 'none', 1: '/arm_controller/controller_state', 2: '/control/joint_states'}.get(int(ros_last_command_source_shared.value), 'unknown') } | "
                f"feedback={int(ros_feedback_count.value)} | "
                f"scene pubs={int(ros_scene_count.value)} | "
                f"{age_text}"
            )
        else:
            moveit_status_var.set(
                "ROS2 bridge: NOT READY — source /opt/ros/humble/setup.bash "
                "and agx_arm_ws/install/setup.bash before starting V56"
            )

        # Update controller settings from spinboxes.
        for i in range(7):
            try:
                speed = float(speed_vars[i].get())
                speed_shared[i] = max(1.0, min(100.0, speed))
            except Exception:
                pass

            try:
                wn = float(wn_vars[i].get())
                wn_shared[i] = max(0.05, min(50.0, wn))
            except Exception:
                pass

            try:
                zeta = float(zeta_vars[i].get())
                zeta_shared[i] = max(0.05, min(5.0, zeta))
            except Exception:
                pass

            try:
                qa = float(qacc_vars[i].get())
                qacc_shared[i] = max(0.05, min(200.0, qa))
            except Exception:
                pass

            update_gain_label(i)

            deg = math.degrees(float(actual_shared[i]))
            actual_labels[i].configure(
                text=f"Actual {deg:.1f}°"
            )

        gripper_actual_label.configure(
            text=f"Actual {float(gripper_actual_mm.value):.0f} mm"
        )

        # Keep force command synchronized with the GUI entry.
        try:
            force_n = float(gripper_force_var.get())
            gripper_force_N.value = max(0.5, min(50.0, force_n))
        except Exception:
            pass

        # Keep clamp-speed multiplier synchronized with the GUI entry.
        try:
            speed_mult = float(clamp_speed_var.get())
            clamp_speed_mult_shared.value = max(
                1.0,
                min(20.0, speed_mult),
            )
        except Exception:
            pass

        if bool(gripper_force_mode.value):
            gripper_mode_label.configure(
                text=(
                    f"Mode: FORCE LOOP → "
                    f"{float(gripper_force_N.value):.1f} N/finger "
                    f"| clamp "
                    f"{float(clamp_speed_mult_shared.value):.1f}×"
                )
            )

        contact_force_status_label.configure(
            text=(
                f"Contact: L {float(contact_left_n_shared.value):.2f} N"
                f" | R {float(contact_right_n_shared.value):.2f} N"
                f" | Tangential "
                f"{float(contact_tangent_n_shared.value):.2f} N"
                f" | Motor {float(contact_motor_cmd_shared.value):.2f} N"
            )
        )

        anti_slip_status_label.configure(
            text=(
                f"Slip: "
                f"{float(anti_slip_slip_mm_s_shared.value):+.2f} mm/s"
                f" | target boost "
                f"+{float(anti_slip_boost_shared.value):.2f} N"
                f" | pad μ fixed 10.00"
            )
        )

        code = int(hold_status_code_shared.value)

        if not bool(hold_enabled_shared.value):
            hold_text = "DISABLED"
        elif bool(hold_active_shared.value):
            hold_text = "HOLDING (3-DOF connect)"
        elif code == 2:
            hold_text = "NO CONTACT OBJECT - clamp first, then HOLD ON"
        else:
            hold_text = (
                f"MONITORING "
                f"{float(hold_ready_shared.value):.2f}/0.10s "
                f"(trigger <= -0.05 mm/s)"
            )

        hold_status_label.configure(
            text=f"Auto HOLD: {hold_text}"
        )

        # Once sim publishes initial pose, mirror GUI sliders once.
        if bool(sim_ready.value) and not initial_synced["done"]:
            for i in range(7):
                deg = math.degrees(float(actual_shared[i]))
                slider_vars[i].set(round(deg, 1))
                target_shared[i] = math.radians(round(deg, 1))
                target_labels[i].configure(
                    text=f"Target {deg:.1f}°"
                )
            grip_mm = float(gripper_actual_mm.value)
            gripper_var.set(grip_mm)
            gripper_target_mm.value = grip_mm
            gripper_target_label.configure(text=f"Target {grip_mm:.0f} mm")
            initial_synced["done"] = True

        # Crucially: GUI stays alive if simulation dies.
        if bool(sim_ready.value) and sim_process.is_alive():
            grav = "ON" if bool(gravity_enabled.value) else "OFF"
            state_code = int(safety_state.value)
            state_text = {
                0: "NORMAL",
                1: "AUTO ROLLBACK",
                2: "MANUAL RECOVER",
                3: "HARD RESET",
                4: "RUNAWAY ROLLBACK",
            }.get(state_code, f"STATE {state_code}")

            status_var.set(
                f"SIM RUNNING | Gravity {grav} | "
                f"Safety {state_text} | Trips {int(safety_trip_count.value)}"
            )
        elif not sim_process.is_alive():
            if int(sim_error_code.value) != 0:
                status_var.set(
                    "SIM EXITED WITH ERROR — GUI kept open; check terminal traceback"
                )
            else:
                status_var.set(
                    "SIMULATION WINDOW CLOSED — GUI kept open"
                )
        else:
            status_var.set("Starting MuJoCo simulation...")

        root.after(50, poll)

    def on_close():
        if skill_core is not None:
            try:
                skill_core.shutdown()
            except Exception:
                pass
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    root.after(50, poll)
    root.after(80, drain_runtime_log_watchdog)

    try:
        root.mainloop()
    finally:
        if skill_core is not None:
            try:
                skill_core.shutdown()
            except Exception:
                pass
        stop_event.set()



# ============================================================
# NERO V2-7 — Constraint-Aware Egg Grasp Planning
# ============================================================
# IMPORTANT:
#   V69 motion core remains frozen.  V2-7 uses already-existing V69 public/
#   internal planning hooks (move_to_pose, _plan_pose, _cartesian_approach_plan)
#   from the semantic wrapper only; NERO_V69_...py is not edited.
#
# Current scope:
#   - rigid eggs only; no break/damage model
#   - top-down two-finger grasp family
#   - egg major-axis orientation awareness
#   - analytical basket-rim clearance pre-check
#   - bounded alternate grasp candidates
#   - explicit failure when reorientation is required
#
# Deliberately deferred:
#   - push/roll/reorient-object skill
#   - tactile feedback / slip / pressure / damage
#   - learned grasp proposal network
# ============================================================

V27_BASKET_CENTER_XY_M = (-0.425, -0.155)
# Physical reed center radius is ~92 mm; reed radius ~4.2 mm.  Keep a few
# additional millimeters as a conservative projected-finger safety margin.
V27_BASKET_SAFE_INNER_RADIUS_M = 0.083

# Conservative XY footprint of the current V69 finger pad + studs.
# The V69 pad is ~60 mm long and ~29 mm wide; these values add a small margin.
V27_FINGER_HALF_LENGTH_M = 0.033
V27_FINGER_HALF_WIDTH_M = 0.017

V27_DEFAULT_GRASP_Z_OFFSET_M = 0.028
V27_PREGRASP_HEIGHT_M = 0.120
V27_OBJECT_STALE_POS_M = 0.008
V27_OBJECT_STALE_ANGLE_DEG = 15.0
V27_MAX_CANDIDATES_TO_PREFLIGHT = 8
# V3.04: the arm settles ~2.5-2.8 mm BELOW the commanded grasp/pregrasp Z
# (V3.03 log: target 50.5 -> actual 47.8 mm; TCP-object 24.9 vs 28.0 mm).
# Command slightly higher so the real TCP lands on the intended 28 mm.
V304_GRASP_Z_COMP_M = float(os.environ.get("NERO_V304_GRASP_Z_COMP_M", "0.0025"))

V304_PREGRASP_CACHE = {}   # (object, candidate_id) -> {"traj", "q_start", "t"}
# V3.05: execution-layer fault bookkeeping (reset per episode).
V305_EXEC_FAULT = {"consecutive": 0, "total": 0, "last": None, "in_recovery": False}

V27_GRASP_RUNTIME = {
    "plans": {},               # object -> latest plan dict
    "tried": {},               # object -> set(candidate_id)
    "selected": {},            # object -> candidate_id
}

# V2-26 one-shot edge-object manipulation recovery budget.
V226_EDGE_REPOSITION_MAX_PER_OBJECT = 1
V226_EDGE_REPOSITION_USED = {}


def _v27_reset_grasp_runtime(object_names=()):
    V27_GRASP_RUNTIME["plans"].clear()
    V27_GRASP_RUNTIME["selected"].clear()
    V27_GRASP_RUNTIME["tried"] = {str(n): set() for n in object_names}
    V226_EDGE_REPOSITION_USED.clear()


def _v27_clamp(x, lo, hi):
    return max(float(lo), min(float(hi), float(x)))


def _v27_quat_normalize(q):
    x, y, z, w = [float(v) for v in q]
    n = (x*x + y*y + z*z + w*w) ** 0.5
    if n < 1e-12:
        return [0.0, 0.0, 0.0, 1.0]
    return [x/n, y/n, z/n, w/n]


def _v27_quat_multiply(a, b):
    ax, ay, az, aw = [float(v) for v in a]
    bx, by, bz, bw = [float(v) for v in b]
    return [
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ]


def _v27_top_down_quat(yaw_rad):
    # Exact same convention as V69 top_down_quat(yaw): q_yaw * Rx(pi).
    h = 0.5 * float(yaw_rad)
    q_yaw = [0.0, 0.0, math.sin(h), math.cos(h)]
    q_down = [1.0, 0.0, 0.0, 0.0]
    return _v27_quat_normalize(_v27_quat_multiply(q_yaw, q_down))


def _v27_axis_angle_pi(angle):
    """Normalize an undirected line angle to [-pi/2, pi/2)."""
    a = float(angle)
    while a >= math.pi / 2.0:
        a -= math.pi
    while a < -math.pi / 2.0:
        a += math.pi
    return a


def _v27_angle_diff_pi(a, b):
    return abs(_v27_axis_angle_pi(float(a) - float(b)))


def _v27_quat_rotation_angle_deg(a, b):
    qa = _v27_quat_normalize(a)
    qb = _v27_quat_normalize(b)
    d = abs(sum(float(qa[i]) * float(qb[i]) for i in range(4)))
    d = _v27_clamp(d, -1.0, 1.0)
    return math.degrees(2.0 * math.acos(d))


def _v27_egg_major_axis_from_quat(quat_xyzw):
    """Rotate local +Z (egg long axis) into world coordinates."""
    x, y, z, w = _v27_quat_normalize(quat_xyzw)
    # R(q) * [0,0,1]
    vx = 2.0 * (x*z + w*y)
    vy = 2.0 * (y*z - w*x)
    vz = 1.0 - 2.0 * (x*x + y*y)
    horizontal = math.hypot(vx, vy)
    if horizontal > 1e-9:
        yaw = _v27_axis_angle_pi(math.atan2(vy, vx))
    else:
        yaw = 0.0
    return {
        "world_axis": [vx, vy, vz],
        "horizontal_projection_strength": horizontal,
        "yaw_rad": yaw,
        "yaw_deg": math.degrees(yaw),
        "orientation_case": (
            "mostly_upright" if horizontal < 0.25
            else "tilted" if horizontal < 0.75
            else "mostly_horizontal"
        ),
    }


def _v27_gripper_footprint_clearance_m(grasp_xy, gripper_yaw_rad, opening_mm):
    """Projected finger-corner clearance to the circular wicker rim.

    Convention: gripper_yaw is the projected long-axis direction of each finger
    pad.  Finger closing direction is +90 deg from that axis.  Aligning the pad
    long axis with the egg major axis therefore pinches across the egg short axis.
    """
    gx, gy = [float(v) for v in grasp_xy]
    yaw = float(gripper_yaw_rad)
    long_dir = (math.cos(yaw), math.sin(yaw))
    close_dir = (-math.sin(yaw), math.cos(yaw))
    half_open = max(0.0, float(opening_mm) / 2000.0)

    points = []
    for finger_sign in (-1.0, +1.0):
        fcx = gx + finger_sign * half_open * close_dir[0]
        fcy = gy + finger_sign * half_open * close_dir[1]
        for ls in (-1.0, +1.0):
            for ws in (-1.0, +1.0):
                px = (
                    fcx
                    + ls * V27_FINGER_HALF_LENGTH_M * long_dir[0]
                    + ws * V27_FINGER_HALF_WIDTH_M * close_dir[0]
                )
                py = (
                    fcy
                    + ls * V27_FINGER_HALF_LENGTH_M * long_dir[1]
                    + ws * V27_FINGER_HALF_WIDTH_M * close_dir[1]
                )
                points.append((px, py))

    bx, by = V27_BASKET_CENTER_XY_M
    max_r = max(math.hypot(px - bx, py - by) for px, py in points)
    return V27_BASKET_SAFE_INNER_RADIUS_M - max_r


def _v304_neighbor_clearance_mm(grasp_xy, gripper_yaw_rad, opening_mm, neighbors):
    """V3.04: min clearance (mm, signed) between the two finger pads and neighbour eggs.

    Neighbour eggs are modelled as horizontal ellipses (semi-axes from their size,
    major axis from their quaternion).  Signed distance is approximated by
    (normalised_radius - 1) * semi_minor.  Used for RANKING only.
    """
    if not neighbors:
        return 999.0
    gx, gy = [float(v) for v in grasp_xy]
    yaw = float(gripper_yaw_rad)
    long_dir = (math.cos(yaw), math.sin(yaw))
    close_dir = (-math.sin(yaw), math.cos(yaw))
    half_open = max(0.0, float(opening_mm) / 2000.0)
    pts = []
    for fs in (-1.0, 1.0):
        fcx = gx + fs * (half_open + V27_FINGER_HALF_WIDTH_M * 0.5) * close_dir[0]
        fcy = gy + fs * (half_open + V27_FINGER_HALF_WIDTH_M * 0.5) * close_dir[1]
        for ls in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for ws in (-1.0, 0.0, 1.0):
                pts.append((fcx + ls * V27_FINGER_HALF_LENGTH_M * long_dir[0] + ws * 0.5 * V27_FINGER_HALF_WIDTH_M * close_dir[0],
                            fcy + ls * V27_FINGER_HALF_LENGTH_M * long_dir[1] + ws * 0.5 * V27_FINGER_HALF_WIDTH_M * close_dir[1]))
    best = 999.0
    for nb in neighbors:
        nx, ny = float(nb["xy"][0]), float(nb["xy"][1])
        a = max(0.005, float(nb.get("semi_major_m", 0.031)))
        b = max(0.005, float(nb.get("semi_minor_m", 0.022)))
        nyaw = float(nb.get("yaw_rad", 0.0))
        cu, su = math.cos(nyaw), math.sin(nyaw)
        for px, py in pts:
            dx, dy = px - nx, py - ny
            u = dx * cu + dy * su
            v = -dx * su + dy * cu
            r = math.sqrt((u / a) ** 2 + (v / b) ** 2)
            best = min(best, (r - 1.0) * b * 1000.0)
    return best


def _v27_make_grasp_candidates(object_name, obj_xyz, obj_quat, opening_mm, tried_ids=(), neighbors=None):
    obj_xyz = [float(v) for v in obj_xyz]
    obj_quat = [float(v) for v in obj_quat]
    opening_mm = float(opening_mm)
    tried = {str(x) for x in tried_ids}

    axis = _v27_egg_major_axis_from_quat(obj_quat)
    ox, oy, oz = obj_xyz
    bx, by = V27_BASKET_CENTER_XY_M
    radial = math.atan2(oy - by, ox - bx)
    tangent = _v27_axis_angle_pi(radial + math.pi / 2.0)

    if axis["horizontal_projection_strength"] >= 0.25:
        base = float(axis["yaw_rad"])
        base_source = "egg_major_axis_projection"
    else:
        # Near-upright egg: its horizontal long-axis projection is ill-defined.
        # Prefer fingers tangent to the basket rim to reduce radial footprint.
        base = tangent
        base_source = "basket_tangent_for_near_upright_egg"

    yaw_defs = [
        ("major_aligned",          base,                         0),
        ("major_plus15",           base + math.radians(15.0),   1),
        ("major_minus15",          base - math.radians(15.0),   1),
        ("major_plus30",           base + math.radians(30.0),   2),
        ("major_minus30",          base - math.radians(30.0),   2),
        ("orthogonal_fallback",    base + math.pi / 2.0,        3),
        ("basket_tangent",         tangent,                     2),
    ]

    # Remove duplicate line orientations after pi-periodic normalization.
    unique = []
    for label, yaw, penalty_rank in yaw_defs:
        y = _v27_axis_angle_pi(yaw)
        if any(_v27_angle_diff_pi(y, old[1]) < math.radians(2.0) for old in unique):
            continue
        unique.append((label, y, penalty_rank))

    # Small inward grasp-center offset is allowed because a basket-wall egg may
    # otherwise leave insufficient room for the outer finger.  Keep it modest;
    # V2-7 does not intentionally push/reorient the egg.
    to_center_x = bx - ox
    to_center_y = by - oy
    n = math.hypot(to_center_x, to_center_y)
    if n > 1e-9:
        inward = (to_center_x / n, to_center_y / n)
    else:
        inward = (0.0, 0.0)

    obj_radial_r = math.hypot(ox - bx, oy - by)
    wall_near = (V27_BASKET_SAFE_INNER_RADIUS_M - obj_radial_r) < 0.030
    offset_defs = [("center", 0.0)]
    if wall_near:
        offset_defs += [("inward4", 0.004), ("inward6", 0.006)]
    else:
        offset_defs += [("inward4", 0.004)]

    candidates = []
    for label, yaw, yaw_rank in unique:
        for off_label, off in offset_defs:
            cid = f"{label}__{off_label}"
            gx = ox + inward[0] * off
            gy = oy + inward[1] * off
            clearance = _v27_gripper_footprint_clearance_m((gx, gy), yaw, opening_mm)
            geom_ok = clearance >= 0.0
            # Favor physically clear, major-axis-aligned, small-offset candidates.
            score = (clearance * 1000.0) - yaw_rank * 4.0 - off * 1000.0 * 0.60
            # V3.04: the virtual basket rim is not a physical obstacle, the
            # neighbouring eggs are.  Penalise finger pads that intrude on them.
            nb_clear = _v304_neighbor_clearance_mm((gx, gy), yaw, opening_mm, neighbors)
            # Capped at -10 so it can only re-rank within the +-30 deg yaw family,
            # never promote an orthogonal (long-axis) clamp over a short-axis one.
            if nb_clear < 3.0:
                score += max(-10.0, 0.8 * (nb_clear - 3.0))
            candidates.append({
                "candidate_id": cid,
                "object_name": str(object_name),
                "strategy": label,
                "base_orientation_source": base_source,
                "gripper_yaw_rad": float(yaw),
                "gripper_yaw_deg": round(math.degrees(yaw), 2),
                "closing_axis_yaw_deg": round(math.degrees(_v27_axis_angle_pi(yaw + math.pi/2.0)), 2),
                "grasp_center_offset_mm": round(off * 1000.0, 2),
                "grasp_xy_m": [round(gx, 6), round(gy, 6)],
                "grasp_xyz_m": [round(gx, 6), round(gy, 6), round(oz + V27_DEFAULT_GRASP_Z_OFFSET_M + V304_GRASP_Z_COMP_M, 6)],
                "pregrasp_xyz_m": [round(gx, 6), round(gy, 6), round(oz + V27_DEFAULT_GRASP_Z_OFFSET_M + V304_GRASP_Z_COMP_M + V27_PREGRASP_HEIGHT_M, 6)],
                "neighbor_clearance_mm": round(nb_clear, 1) if nb_clear < 900 else None,
                "quat_xyzw": [round(float(v), 8) for v in _v27_top_down_quat(yaw)],
                "basket_projected_clearance_mm": round(clearance * 1000.0, 2),
                "geometry_feasible": True,
                "already_tried": cid in tried,
                "score": round(score, 3),
            })

    candidates.sort(
        key=lambda c: (
            c["already_tried"],
            not c["geometry_feasible"],
            -float(c["score"]),
        )
    )
    return axis, candidates


def _v27_public_runtime_snapshot():
    out = {"selected": dict(V27_GRASP_RUNTIME.get("selected", {})), "tried": {}}
    for obj, ids in V27_GRASP_RUNTIME.get("tried", {}).items():
        out["tried"][obj] = sorted(str(x) for x in ids)
    out["edge_reposition_used"] = {str(k): int(v) for k, v in V226_EDGE_REPOSITION_USED.items()}
    out["edge_reposition_budget_per_object"] = int(V226_EDGE_REPOSITION_MAX_PER_OBJECT)
    return out



# ============================================================
# NERO V2-9 — Placement-Aware Egg Handling
# ============================================================
# V69 remains frozen.  These helpers live only in the semantic/task wrapper.
#
# Physical idea:
#   1) grasp the egg across its SHORT axis (V2-7)
#   2) lift
#   3) move to a safe orientation workspace
#   4) rotate the HELD egg so its LONG axis aligns with the slot entry axis
#   5) move precisely above the slot
#   6) descend slowly along that long axis only to a capture depth
#   7) release before the fingers have to enter the egg-holder walls
#   8) retreat vertically and let verify_object_in_slot decide success
# ============================================================


# ============================================================
# NERO V2-9 — Hold Continuity + Bounded Wrist Reorientation
# ============================================================
V272_VERSION = "2.7.2"
V29_VERSION = "2.9"
V210_VERSION = "2.10"
V272_WRIST_STEP_DEG = 10.0
V272_WRIST_PLAN_ATTEMPTS = 4
V272_JOINT_STEP_LIMIT_DEG = {
    "joint1": 4.0, "joint2": 4.0, "joint3": 4.0,
    "joint4": 8.0, "joint5": 10.0,
    "joint6": 25.0, "joint7": 25.0,
}
V272_JOINT_SCORE_WEIGHT = {
    "joint1": 8.0, "joint2": 8.0, "joint3": 8.0,
    "joint4": 4.0, "joint5": 3.0,
    "joint6": 1.0, "joint7": 1.0,
}


def _v272_wrap_delta(v):
    x = float(v)
    while x > math.pi:
        x -= 2.0 * math.pi
    while x < -math.pi:
        x += 2.0 * math.pi
    return x


def _v272_plan_joint_deltas_deg(skill_core, traj):
    if traj is None:
        return {}
    jt = traj.joint_trajectory
    if not jt.points or not jt.joint_names:
        return {}
    js = skill_core._current_joint_state_copy(timeout_s=1.5)
    cur = {str(n): float(p) for n, p in zip(js.name, js.position)}
    final = {str(n): float(p) for n, p in zip(jt.joint_names, jt.points[-1].positions)}
    out = {}
    for name in [f"joint{i}" for i in range(1, 8)]:
        if name in cur and name in final:
            out[name] = math.degrees(abs(_v272_wrap_delta(final[name] - cur[name])))
    return out


def _v272_wrist_score(d):
    return sum(float(V272_JOINT_SCORE_WEIGHT.get(k, 1.0)) * float(v) for k, v in d.items())


def _v272_wrist_plan_ok(d):
    if not d:
        return False
    for name, lim in V272_JOINT_STEP_LIMIT_DEG.items():
        if float(d.get(name, 0.0)) > float(lim):
            return False
    proximal = sum(float(d.get(f"joint{i}", 0.0)) for i in range(1, 6))
    return proximal <= 22.0


def _v272_execute_wrist_bounded_pose(skill_core, xyz, quat_xyzw,
                                     velocity_scale=0.035,
                                     accel_scale=0.035,
                                     orientation_tolerance=0.07,
                                     position_tolerance=0.006):
    """Plan first; reject large shoulder/elbow branch changes before execution.

    V69 is untouched.  We use its existing _plan_pose/_execute_trajectory,
    inspect the final joint solution, and execute only a bounded branch.
    """
    accepted = []
    rejected = []
    for attempt in range(1, int(V272_WRIST_PLAN_ATTEMPTS) + 1):
        traj = skill_core._plan_pose(
            xyz, quat_xyzw,
            velocity_scale=velocity_scale,
            accel_scale=accel_scale,
            orientation_tolerance=orientation_tolerance,
            position_tolerance=position_tolerance,
            planning_attempts=12,
            planning_time=5.0,
        )
        d = _v272_plan_joint_deltas_deg(skill_core, traj)
        rec = {
            "attempt": attempt,
            "joint_delta_deg": {k: round(float(v), 2) for k, v in d.items()},
            "score": round(_v272_wrist_score(d), 2) if d else None,
            "bounded": bool(_v272_wrist_plan_ok(d)),
        }
        if rec["bounded"]:
            accepted.append((_v272_wrist_score(d), traj, rec))
        else:
            rejected.append(rec)
    if not accepted:
        skill_core.emit(
            "[V2-7.2 WRIST PREFLIGHT REJECT] no bounded branch | " +
            json.dumps(rejected, ensure_ascii=False)
        )
        raise RuntimeError(
            "WRIST_BOUNDED_PLAN_REJECTED: all IK/MoveIt branches require excessive joint motion"
        )
    accepted.sort(key=lambda x: x[0])
    _score, traj, chosen = accepted[0]
    skill_core.emit(
        "[V2-7.2 WRIST PREFLIGHT OK] joint_delta_deg=" +
        json.dumps(chosen["joint_delta_deg"], ensure_ascii=False) +
        f" | weighted_score={chosen['score']}"
    )
    return skill_core._execute_trajectory(traj)


def _v272_digit_segments(digit):
    yL, yR = -0.0048, 0.0048
    zB, zM, zT = -0.0065, 0.0, 0.0065
    if digit == 1:
        return [((0.0015, zT), (0.0015, zB)), ((-0.0015, zT-0.0015), (0.0015, zT))]
    if digit == 2:
        return [((yL, zT), (yR, zT)), ((yR, zT), (yL, zB)), ((yL, zB), (yR, zB))]
    if digit == 3:
        return [((yL, zT), (yR, zT)), ((yL+0.001, zM), (yR, zM)),
                ((yL, zB), (yR, zB)), ((yR, zT), (yR, zM)), ((yR, zM), (yR, zB))]
    if digit == 4:
        return [((yL, zT), (yL, zM)), ((yL, zM), (yR, zM)), ((yR, zT), (yR, zB))]
    return []


def _v272_add_egg_number_labels_to_mjcf(xml_path):
    """Attach non-colliding 3D numerals 1/2/3/4 to the actual four egg bodies."""
    # V3.43 A2 BEGIN
    if _v343_scene_name() != "fill_egg_holder":
        return str(xml_path)                       # no eggs in this scene: nothing to label
    # V3.43 A2 END
    path = Path(xml_path).resolve()
    tree = ET.parse(path)
    root = tree.getroot()
    all_egg_bodies = sorted(
        str(b.get('name')) for b in root.findall('.//body')
        if str(b.get('name', '')).startswith('obj_egg')
    )
    expected_egg_bodies = ['obj_egg1', 'obj_egg2', 'obj_egg3', 'obj_egg4']
    if all_egg_bodies != expected_egg_bodies:
        raise RuntimeError(
            f'V2-7.2 physical egg body registry mismatch: expected={expected_egg_bodies}, actual={all_egg_bodies}'
        )
    found = []
    for digit in range(1, 5):
        body_name = f"obj_egg{digit}"
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            continue
        found.append(body_name)
        for old in list(body.findall("geom")):
            if str(old.get("name", "")).startswith(f"v272_egg{digit}_label_"):
                body.remove(old)
        idx = 0
        for side in (-1.0, 1.0):
            x = side * 0.0235
            for (y0, z0), (y1, z1) in _v272_digit_segments(digit):
                idx += 1
                ET.SubElement(body, "geom", {
                    "name": f"v272_egg{digit}_label_{idx:02d}",
                    "type": "capsule",
                    "fromto": f"{x:.5f} {y0:.5f} {z0:.5f} {x:.5f} {y1:.5f} {z1:.5f}",
                    "size": "0.00085",
                    "rgba": "0.035 0.035 0.035 1",
                    "contype": "0", "conaffinity": "0", "density": "0", "group": "1",
                })
    tree.write(path, encoding="utf-8", xml_declaration=True)
    print(
        f"[V2-7.2 EGG IDENTITY] physical egg bodies={all_egg_bodies} | labelled={found} | count={len(all_egg_bodies)}",
        flush=True,
    )
    if found != expected_egg_bodies:
        raise RuntimeError(f"V2-7.2 egg label registry mismatch: {found}")
    return str(path)


V271_SLOT_CENTERS_M = {
    "slot_1": (-0.510, 0.123, 0.043),
    "slot_2": (-0.450, 0.123, 0.043),
    "slot_3": (-0.390, 0.123, 0.043),
    "slot_4": (-0.330, 0.123, 0.043),
    "slot_5": (-0.510, 0.187, 0.043),
    "slot_6": (-0.450, 0.187, 0.043),
    "slot_7": (-0.390, 0.187, 0.043),
    "slot_8": (-0.330, 0.187, 0.043),
}
V271_SLOT_ENTRY_AXIS = (0.0, 0.0, -1.0)
# V3.29: perceived-layout state (wrist no-prior mode) + MoveIt holder override shared with the simulator bridge
V329_LAYOUT_STATE = {"perceived_required": False, "holder_ok": False, "pile_ok": False}
V329_SCENE_OVERRIDE_PATH = Path(__file__).resolve().parent / "logs" / "v329_moveit_holder.json"
V271_HOLDER_RIM_TOP_Z_M = 0.044
V271_SAFE_ORIENTATION_XY_M = (-0.4225, 0.0000)
V271_SAFE_ORIENTATION_MIN_Z_M = 0.245
V271_SAFE_ORIENTATION_EXTRA_Z_M = 0.085
V271_ENTRY_ABOVE_RELEASE_M = 0.055
V271_CAPTURE_DEPTH_M = 0.008
V271_SLOT_XY_TOL_M = 0.007
V271_SLOT_XY_ABORT_M = 0.012
V271_AXIS_TOL_DEG = 10.0
V271_INSERT_AXIS_ABORT_DEG = 12.0
V271_BYSTANDER_MOVE_TOL_M = 0.003  # V2-16 warning threshold: small clutter motion is tolerated
V216_BYSTANDER_ABORT_TOL_M = 0.008  # >8 mm bystander motion still aborts before clamp
V271_RELATIVE_DISTANCE_DRIFT_TOL_M = 0.025
V271_INSERT_STEP_M = 0.010
V271_INSERT_FINE_STEP_M = 0.005
V271_RELEASE_STAGE_OPEN_MM = 10.0
V271_RELEASE_FULL_OPEN_MM = 90.0
V271_RETREAT_M = 0.080

V271_PLACEMENT_RUNTIME = {
    "plans": {},
    "oriented": {},
    "slot_entry": {},
    "released": {},
}
V271_BYSTANDER_BASELINE = {}


def _v271_reset_placement_runtime(object_names=()):
    V271_PLACEMENT_RUNTIME["plans"].clear()
    V271_PLACEMENT_RUNTIME["oriented"].clear()
    V271_PLACEMENT_RUNTIME["slot_entry"].clear()
    V271_PLACEMENT_RUNTIME["released"].clear()
    V271_BYSTANDER_BASELINE.clear()
    for _name in object_names:
        V271_BYSTANDER_BASELINE[str(_name)] = {}


def _v271_public_runtime_snapshot():
    def _safe(v):
        if isinstance(v, dict):
            return {str(k): _safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_safe(x) for x in v]
        if isinstance(v, set):
            return sorted(str(x) for x in v)
        return v
    return {
        "plans": _safe(V271_PLACEMENT_RUNTIME.get("plans", {})),
        "oriented": _safe(V271_PLACEMENT_RUNTIME.get("oriented", {})),
        "slot_entry": _safe(V271_PLACEMENT_RUNTIME.get("slot_entry", {})),
        "released": _safe(V271_PLACEMENT_RUNTIME.get("released", {})),
    }


def _v271_dot(a, b):
    return sum(float(a[i]) * float(b[i]) for i in range(3))


def _v271_cross(a, b):
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def _v271_norm(v):
    return math.sqrt(sum(float(x) * float(x) for x in v))


def _v271_unit(v):
    n = _v271_norm(v)
    if n < 1e-12:
        return [0.0, 0.0, 1.0]
    return [float(x) / n for x in v]


def _v271_quat_conjugate(q):
    x, y, z, w = _v27_quat_normalize(q)
    return [-x, -y, -z, w]


def _v271_quat_from_two_vectors(a, b):
    """World-frame quaternion that rotates vector a onto vector b."""
    va = _v271_unit(a)
    vb = _v271_unit(b)
    d = max(-1.0, min(1.0, _v271_dot(va, vb)))
    if d > 1.0 - 1e-10:
        return [0.0, 0.0, 0.0, 1.0]
    if d < -1.0 + 1e-10:
        # 180 deg: choose any stable axis perpendicular to a.
        ref = [1.0, 0.0, 0.0] if abs(va[0]) < 0.8 else [0.0, 1.0, 0.0]
        axis = _v271_unit(_v271_cross(va, ref))
        return [axis[0], axis[1], axis[2], 0.0]
    c = _v271_cross(va, vb)
    q = [c[0], c[1], c[2], 1.0 + d]
    return _v27_quat_normalize(q)


def _v271_slerp(q0, q1, t):
    qa = _v27_quat_normalize(q0)
    qb = _v27_quat_normalize(q1)
    dot = sum(float(qa[i]) * float(qb[i]) for i in range(4))
    if dot < 0.0:
        qb = [-float(x) for x in qb]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        q = [(1.0 - float(t)) * qa[i] + float(t) * qb[i] for i in range(4)]
        return _v27_quat_normalize(q)
    theta = math.acos(dot)
    s = math.sin(theta)
    a = math.sin((1.0 - float(t)) * theta) / s
    b = math.sin(float(t) * theta) / s
    return _v27_quat_normalize([a * qa[i] + b * qb[i] for i in range(4)])


def _v271_axis_line_error_deg(axis, target_axis=V271_SLOT_ENTRY_AXIS):
    a = _v271_unit(axis)
    b = _v271_unit(target_axis)
    # Egg long axis is an undirected line: +axis and -axis are equivalent.
    d = abs(max(-1.0, min(1.0, _v271_dot(a, b))))
    return math.degrees(math.acos(d))


def _v271_target_tcp_quat_for_slot(obj_quat, tcp_quat):
    axis_info = _v27_egg_major_axis_from_quat(obj_quat)
    current_axis = [float(v) for v in axis_info["world_axis"]]
    slot_axis = [float(v) for v in V271_SLOT_ENTRY_AXIS]
    # Choose +/- slot axis requiring the smaller rotation because the egg long
    # axis is a line, not a directed arrow.
    target_axis = slot_axis if _v271_dot(current_axis, slot_axis) >= 0.0 else [-v for v in slot_axis]
    q_delta = _v271_quat_from_two_vectors(current_axis, target_axis)
    target_tcp_q = _v27_quat_normalize(_v27_quat_multiply(q_delta, tcp_quat))
    return {
        "current_axis": current_axis,
        "target_axis": target_axis,
        "axis_error_before_deg": _v271_axis_line_error_deg(current_axis, slot_axis),
        "target_tcp_quat_xyzw": target_tcp_q,
    }


def _v330_apply_wrist_model_visibility(model):
    """Simulator side of V3.30: draw only the wrist camera model in use (flag file written by the GUI)."""
    try:
        import mujoco as _mj
        cache = getattr(_v330_apply_wrist_model_visibility, "_cache", None)
        if cache is None:
            cache = {"ids": {}, "rgba": {}, "last": None}
            for gid in range(int(model.ngeom)):
                nm = _mj.mj_id2name(model, _mj.mjtObj.mjOBJ_GEOM, gid) or ""
                for pf in ("d435i", "usb130"):
                    if nm.startswith(f"v330_cam_{pf}_"):
                        cache["ids"].setdefault(pf, []).append(gid)
                        cache["rgba"][gid] = [float(v) for v in model.geom_rgba[gid]]
            _v330_apply_wrist_model_visibility._cache = cache
        try:
            show = json.loads(V330_WRIST_MODEL_FLAG.read_text()).get("show", "none")
        except Exception:
            show = os.environ.get("NERO_V330_SHOW_WRIST_MODEL", "none")    # V3.33: both cameras share one face
        if show == cache["last"]:
            return
        cache["last"] = show
        for pf, gids in cache["ids"].items():
            on = show in (pf, "both")
            for gid in gids:
                rgba = list(cache["rgba"][gid])
                rgba[3] = rgba[3] if on else 0.0
                model.geom_rgba[gid] = rgba
    except Exception:
        pass


def _v329_read_scene_override():
    """Simulator bridge side of V3.29: perceived MoveIt holder boxes written by the backend (None = registered)."""
    try:
        return json.loads(V329_SCENE_OVERRIDE_PATH.read_text())
    except Exception:
        return None


def _v271_slot(slot_name):
    if V329_LAYOUT_STATE.get("perceived_required") and not V329_LAYOUT_STATE.get("holder_ok"):
        return None                       # V3.29: wrist no-prior mode and the holder has not been perceived yet
    return V271_SLOT_CENTERS_M.get(str(slot_name))


def _v271_egg_half_length_m(geom):
    try:
        size = geom.get("approx_size_mm", [44.0, 43.0, 62.0])
        return max(0.020, float(size[2]) / 2000.0)
    except Exception:
        return 0.031


def _v340_perceived_poses(skill_core):
    """V3.40: True when skill_core.object_pose returns PERCEIVED poses (wrist / fixed camera without priors)."""
    return (getattr(getattr(skill_core, "object_pose", None), "__name__", "") == "_v329_object_pose"
            and V311_STATE.get("mode") in ("camera", "wrist", "wrist_rgb"))


def _v271_bystander_moves(skill_core, active_obj):
    """V3.40: with perceived poses only the horizontal motion of a neighbour that a fresh view actually SEES counts,
    minus the perception noise (2 x its ray rms, at most NERO_V340_BYSTANDER_NOISE_MAX_MM). V3.39 log: the table-plane
    re-estimate moved every egg ~15 mm in z -> 3D 'motion' 10.7 / 11 mm -> false BYSTANDER_EGG_DISTURBED twice."""
    baseline = V271_BYSTANDER_BASELINE.get(str(active_obj), {})
    moved = []
    perc = _v340_perceived_poses(skill_core) and os.environ.get("NERO_V340_BYSTANDER_XY", "1") != "0"
    last_objs = ((V311_STATE.get("last") or {}).get("objects") or {}) if perc else {}
    nmax = float(os.environ.get("NERO_V340_BYSTANDER_NOISE_MAX_MM", "6")) / 1000.0
    for other, p0 in baseline.items():
        try:
            p1, _ = skill_core.object_pose(other)
            if perc:
                o = last_objs.get(str(other)) or {}
                if not o.get("seen"):
                    continue                     # not seen again since the plan: no evidence it moved
                m = o.get("measurement") or {}
                rms = float(m.get("ray_rms_mm") or m.get("plane_spread_mm") or 0.0) / 1000.0
                allow = min(nmax, 2.0 * rms)
                raw = math.hypot(float(p1[0]) - float(p0[0]), float(p1[1]) - float(p0[1]))
                d = max(0.0, raw - allow)
                if d > V271_BYSTANDER_MOVE_TOL_M:
                    moved.append({"object_name": other, "motion_mm": round(d * 1000.0, 2),
                                  "raw_xy_motion_mm": round(raw * 1000.0, 2),
                                  "perception_noise_allowance_mm": round(allow * 1000.0, 2),
                                  "basis": "perceived xy (V3.40), z ignored"})
                continue
            d = math.sqrt(sum((float(p1[i]) - float(p0[i])) ** 2 for i in range(3)))
            if d > V271_BYSTANDER_MOVE_TOL_M:
                moved.append({"object_name": other, "motion_mm": round(d * 1000.0, 2)})
        except Exception:
            pass
    moved.sort(key=lambda x: -float(x["motion_mm"]))
    return moved


# ============================================================
# V3.08 SIMULATED D435i CAMERA + PERCEPTION (record only; skills use truth)
# ============================================================
V308_SHARED = {}


def v308_trigger(tag):
    """Ask the camera process for one capture (non-blocking, never raises)."""
    try:
        seq = V308_SHARED.get("trig_seq")
        if seq is None:
            return
        b = str(tag).encode("utf-8")[:255]
        arr = V308_SHARED["trig_tag"]
        arr[: len(b)] = b
        arr[len(b)] = b"\0"
        with seq.get_lock():
            seq.value += 1
    except Exception:
        pass


# ============================================================
# V3.11 SKILLS USE PERCEPTION (camera) OR JSON GROUND TRUTH (switch in the Agent tab)
# ============================================================
V311_STATE = {"mode": ({"camera": "camera", "wrist": "wrist", "wrist_d435i": "wrist", "wrist_rgb": "wrist_rgb",
                         "wrist_usb130": "wrist_rgb", "usb130": "wrist_rgb"}.get(
                            os.environ.get("NERO_V311_POSE_SOURCE", "camera").strip().lower(), "truth")),
              "last": None}
# V3.26: wrist camera process (started on demand when "wrist" is selected; shares the V3.08 qpos/model buffers)
V326_SHARED = {}
# V3.35: calibration-error level, live perception refresh, friction grasp
V335_STATE = {
    "calib_level": float(os.environ.get("NERO_V335_CALIB_LEVEL", "0.3")),
    "live_on": os.environ.get("NERO_V335_LIVE", "1") != "0",
    "live_hz": float(os.environ.get("NERO_V335_LIVE_HZ", "1.5")),
    "pending": [], "in_hand": None, "in_hand_bad": 0, "holder_hist": [], "holder_jump": [],
    "tool_running": False, "live_count": 0, "live_err_t": 0.0, "events": [],
}


def v335_pose_source_label():
    """V3.35: which camera actually provides the poses (V3.11-V3.34 always said camera_d435i_perception)."""
    if V311_STATE.get("mode") == "wrist_rgb":
        return "camera_wrist_usb130_rgb_perception"
    if V311_STATE.get("mode") == "wrist":
        return "camera_wrist_d435i_perception"
    return "camera_fixed_d435i_perception"


def v335_live_dir():
    return Path(os.environ.get("NERO_V326_OUT_DIR") or str(Path(__file__).resolve().parent / "logs" / "perception_wrist")) / "live"


def v311_camera_available():
    try:
        st = V308_SHARED.get("status")
        return st is not None and V308_SHARED.get("res_seq") is not None and int(st.value) == 1
    except Exception:
        return False


def v311_mode():
    """'camera' only when selected AND the camera process is running; otherwise 'truth'.
    V3.26: the wrist camera is also a 'camera' source for the skills (perceived poses); v326_wrist_selected() tells
    which camera provides them."""
    if V311_STATE.get("mode") in ("wrist", "wrist_rgb"):
        return "camera"                    # V3.35: never "truth" when a wrist camera is selected
    # V3.35: the fixed camera too - only an explicit "truth" selection gives simulator truth
    return "camera" if V311_STATE.get("mode") == "camera" else "truth"


def v326_wrist_selected():
    """V3.35: the wrist camera is the pose source whenever it is SELECTED - if its process is not running the
    skills raise PERCEPTION_UNAVAILABLE and the task stops (no silent fallback to simulator truth)."""
    return V311_STATE.get("mode") in ("wrist", "wrist_rgb")


def v331_fixed_no_prior():
    """V3.31: fixed stand camera without priors (default on; NERO_V331_FIXED_NO_PRIOR=0 = V3.11 behaviour)."""
    return os.environ.get("NERO_V331_FIXED_NO_PRIOR", "1") != "0"


def v330_wrist_profile():
    """V3.30: which wrist camera provides the poses: 'd435i' (RGB-D) or 'usb130' (RGB-only 130 deg USB camera)."""
    return "usb130" if V311_STATE.get("mode") == "wrist_rgb" else "d435i"


V330_WRIST_MODEL_FLAG = Path(__file__).resolve().parent / "logs" / "v330_wrist_camera_model.json"


def v330_write_model_flag(show):
    """Tell the simulator which wrist camera model to draw ('d435i' / 'usb130' / 'both' / 'none')."""
    try:
        V330_WRIST_MODEL_FLAG.parent.mkdir(parents=True, exist_ok=True)
        V330_WRIST_MODEL_FLAG.write_text(json.dumps({"show": str(show), "t": time.time()}))
    except Exception:
        pass


def v326_available():
    try:
        st = V326_SHARED.get("status")
        return st is not None and int(st.value) == 1
    except Exception:
        return False


def v326_start(stop_event):
    """V3.26: create the wrist camera's shared state and start its process once (needs the V3.08 model buffers)."""
    if V326_SHARED.get("res_seq") is not None:
        return True
    if V308_SHARED.get("qpos") is None:
        print("[V3.26 WRIST CAMERA] V3.08 camera buffers missing (NERO_V308_CAMERA=0) -> wrist mode unavailable", flush=True)
        return False
    try:
        import NERO_V326_Wrist_Camera as _v326
        if not _v326.enabled():
            print("[V3.26 WRIST CAMERA] disabled (NERO_V326_WRIST=0)", flush=True)
            return False
    except Exception as exc:
        print(f"[V3.26 WRIST CAMERA] module unavailable: {exc}", flush=True)
        return False
    V326_SHARED.update({"qpos": V308_SHARED["qpos"], "meta": V308_SHARED["meta"], "model_path": V308_SHARED["model_path"],
                        "trig_seq": mp.Value("i", 0), "trig_tag": mp.Array("c", 256), "status": mp.Value("i", 0),
                        "res_buf": mp.Array("c", 65536)})
    V326_SHARED["res_seq"] = mp.Value("i", 0)
    out_root = os.environ.get("NERO_V326_OUT_DIR") or str(Path(__file__).resolve().parent / "logs" / "perception_wrist")
    threading.Thread(target=_v326.supervisor, args=(V326_SHARED, stop_event, out_root), daemon=True,
                     name="v326_wrist_camera_supervisor").start()
    print("[V3.26 WRIST CAMERA] starting wrist camera process", flush=True)
    return True


def v311_capture_sync(tag, timeout_s=None):
    """V3.11: request one capture and wait for its perceived scene. Returns the payload dict
    (objects: name -> pose/trusted/seen ...) or raises RuntimeError (never falls back to truth)."""
    timeout_s = float(os.environ.get("NERO_V311_CAPTURE_TIMEOUT_S", "20")) if timeout_s is None else float(timeout_s)
    t_wait = time.time()
    while not v311_camera_available():
        if time.time() - t_wait > timeout_s:
            raise RuntimeError("PERCEPTION_UNAVAILABLE: camera process is not running")
        time.sleep(0.1)
    seq = V308_SHARED["trig_seq"]
    arr = V308_SHARED["trig_tag"]
    b = str(tag).encode("utf-8")[:255]
    with seq.get_lock():
        arr[: len(b)] = b
        arr[len(b)] = b"\0"
        seq.value += 1
        mine = int(seq.value)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if int(V308_SHARED["res_seq"].value) >= mine:
            buf = V308_SHARED["res_buf"]
            with buf.get_lock():
                raw = bytes(buf[:]).split(b"\0", 1)[0]
            payload = json.loads(raw.decode("utf-8"))
            if payload.get("error"):
                raise RuntimeError(f"PERCEPTION_ERROR: {payload['error']}")
            payload["wait_s"] = round(time.time() - t0, 2)
            V311_STATE["last"] = payload
            return payload
        time.sleep(0.03)
    raise RuntimeError(f"PERCEPTION_TIMEOUT: no capture result within {timeout_s:.0f}s")


def _v308_camera_supervisor(stop_event):
    """Start the camera process; on an OpenGL backend failure try the next one."""
    import NERO_V308_Camera_Perception as cam
    out_root = os.environ.get("NERO_V308_OUT_DIR") or str(Path(__file__).resolve().parent / "logs" / "perception")
    for gl in cam.gl_backends():
        if stop_event.is_set():
            return
        saved = {k: os.environ.get(k) for k in ("MUJOCO_GL", "PYOPENGL_PLATFORM")}
        os.environ["MUJOCO_GL"] = gl
        if gl in ("egl", "osmesa"):
            os.environ["PYOPENGL_PLATFORM"] = gl
        else:
            os.environ.pop("PYOPENGL_PLATFORM", None)
        try:
            target331 = cam.camera_process_main
            if v331_fixed_no_prior():
                import NERO_V331_Fixed_Camera as _v331f      # V3.31: generic naming + perceived holder
                target331 = _v331f.fixed_camera_process_main
            p = mp.Process(target=target331, name="nero_v308_camera", daemon=True, args=(
                V308_SHARED["qpos"], V308_SHARED["meta"], V308_SHARED["model_path"], V308_SHARED["trig_seq"],
                V308_SHARED["trig_tag"], V308_SHARED["status"], stop_event, out_root, gl,
                V308_SHARED.get("res_seq"), V308_SHARED.get("res_buf")))
            p.start()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        V308_SHARED["process"] = p
        ever_ready = False
        while p.is_alive():
            p.join(timeout=1.0)
            if int(V308_SHARED["status"].value) == 1:
                ever_ready = True
        # exit 3 = Renderer could not start; any other crash before READY usually means
        # `import mujoco` failed for this MUJOCO_GL value -> try the next backend too.
        gl_problem = p.exitcode == cam.EXIT_GL_FAILED or (p.exitcode not in (0, None) and not ever_ready)
        if gl_problem and not stop_event.is_set():
            print(f"[V3.08 CAMERA] OpenGL '{gl}' failed (exit {p.exitcode}); trying next backend", flush=True)
            continue
        return
    print("[V3.08 CAMERA] no OpenGL backend worked (tried %s); perception disabled" % ",".join(cam.gl_backends()), flush=True)


def main():
    # V3.43 A2 BEGIN
    if _V343 is not None:
        os.environ["NERO_V343_ACTIVE_SCENE"] = _V343.selected_scene()      # GUI + spawned MuJoCo use one scene
        print(f"[V3.44 SCENE] active MuJoCo scene: {os.environ['NERO_V343_ACTIVE_SCENE']} "
              f"(NERO_SCENE={os.environ.get('NERO_SCENE', '') or 'unset'})", flush=True)
    # V3.43 A2 END
    try:
        V329_SCENE_OVERRIDE_PATH.unlink()        # V3.29: no stale perceived MoveIt holder from an earlier session
    except Exception:
        pass
    try:
        V330_WRIST_MODEL_FLAG.unlink()           # V3.30: the simulator draws both wrist camera models until a wrist source is chosen
    except Exception:
        pass
    target_shared = mp.Array("d", [0.0] * 7)
    actual_shared = mp.Array("d", [0.0] * 7)

    speed_shared = mp.Array(
        "d",
        DEFAULT_SPEED_PERCENT.tolist(),
    )
    wn_shared = mp.Array(
        "d",
        DEFAULT_WN.tolist(),
    )
    zeta_shared = mp.Array(
        "d",
        DEFAULT_ZETA.tolist(),
    )
    qacc_shared = mp.Array(
        "d",
        DEFAULT_MAX_QACC.tolist(),
    )

    gravity_enabled = mp.Value("b", True)
    gripper_target_mm = mp.Value("d", 0.0)
    gripper_actual_mm = mp.Value("d", 0.0)

    # V16 gripper force-control state.
    gripper_force_mode = mp.Value("b", False)
    gripper_force_N = mp.Value("d", 5.0)

    # V31 adjustable clamp approach speed.
    # 1x = original V30 speed; V42 default free-space speed = 15x.
    clamp_speed_mult_shared = mp.Value("d", 15.0)

    # V22 anti-slip + real contact-force telemetry.
    anti_slip_boost_shared = mp.Value("d", 0.0)
    anti_slip_slip_mm_s_shared = mp.Value("d", 0.0)
    anti_slip_mu_shared = mp.Value("d", 10.0)

    contact_left_n_shared = mp.Value("d", 0.0)
    contact_right_n_shared = mp.Value("d", 0.0)
    contact_tangent_n_shared = mp.Value("d", 0.0)
    contact_motor_cmd_shared = mp.Value("d", 0.0)

    # V30 auto-creep runtime equality HOLD.
    hold_enabled_shared = mp.Value("b", True)
    hold_active_shared = mp.Value("b", False)
    hold_name_shared = mp.Value("i", -1)
    hold_ready_shared = mp.Value("d", 0.0)  # retained for compatibility
    hold_now_request = mp.Value("b", False)
    hold_off_request = mp.Value("b", False)
    hold_status_code_shared = mp.Value("i", 0)

    # V14 safety control/status.
    reset_zero_request = mp.Value("b", False)
    recover_request = mp.Value("b", False)
    safety_trip_count = mp.Value("i", 0)
    safety_state = mp.Value("i", 0)

    sim_ready = mp.Value("b", False)
    sim_error_code = mp.Value("i", 0)

    # V43 ROS2 / MoveIt2 bridge telemetry.
    ros_bridge_ready = mp.Value("b", False)
    ros_cmd_count = mp.Value("i", 0)
    ros_feedback_count = mp.Value("i", 0)
    ros_scene_count = mp.Value("i", 0)
    ros_last_cmd_age = mp.Value("d", -1.0)
    ros_last_cmd_joint_count = mp.Value("i", 0)
    # V56: controller-state bridge telemetry must be shared with the Tk parent.
    ros_controller_state_rx_shared = mp.Value("i", 0)
    ros_fallback_joint_state_rx_shared = mp.Value("i", 0)
    # 0=none, 1=/arm_controller/controller_state, 2=/control/joint_states
    ros_last_command_source_shared = mp.Value("i", 0)

    # V45: OFF except during a skill trajectory execution.
    ros_arm_command_gate_shared = mp.Value("b", False)

    stop_event = mp.Event()

    # V3.08 SIMULATED D435i CAMERA shared state
    v308_on = False
    try:
        import NERO_V308_Camera_Perception as _v308
        v308_on = _v308.enabled()
    except Exception as _v308_exc:
        print(f"[V3.08 CAMERA WARN] module unavailable, perception disabled: {_v308_exc}", flush=True)
    v308_kwargs = {}
    if v308_on:
        V308_SHARED["qpos"] = mp.Array("d", 1024)
        V308_SHARED["meta"] = mp.Array("d", 4)          # nq, sim_time, write_count, ready
        V308_SHARED["model_path"] = mp.Array("c", 4096)
        V308_SHARED["trig_seq"] = mp.Value("i", 0)
        V308_SHARED["trig_tag"] = mp.Array("c", 256)
        V308_SHARED["status"] = mp.Value("i", 0)        # 0 starting, 1 running, 2 failed
        V308_SHARED["res_seq"] = mp.Value("i", 0)       # V3.11: trigger seq of the last finished capture
        V308_SHARED["res_buf"] = mp.Array("c", 65536)   # V3.11: its tracked scene (JSON)
        v308_kwargs = {"v308_qpos": V308_SHARED["qpos"], "v308_meta": V308_SHARED["meta"],
                       "v308_model_path": V308_SHARED["model_path"]}

    sim_process = mp.Process(
        target=run_sim,
        args=(
            target_shared,
            actual_shared,
            speed_shared,
            wn_shared,
            zeta_shared,
            qacc_shared,
            gravity_enabled,
            gripper_target_mm,
            gripper_actual_mm,
            gripper_force_mode,
            gripper_force_N,
            clamp_speed_mult_shared,
            anti_slip_boost_shared,
            anti_slip_slip_mm_s_shared,
            anti_slip_mu_shared,
            contact_left_n_shared,
            contact_right_n_shared,
            contact_tangent_n_shared,
            contact_motor_cmd_shared,
            hold_enabled_shared,
            hold_active_shared,
            hold_name_shared,
            hold_ready_shared,
            hold_now_request,
            hold_off_request,
            hold_status_code_shared,
            reset_zero_request,
            recover_request,
            safety_trip_count,
            safety_state,
            sim_ready,
            sim_error_code,
            ros_bridge_ready,
            ros_cmd_count,
            ros_feedback_count,
            ros_scene_count,
            ros_last_cmd_age,
            ros_last_cmd_joint_count,
            ros_controller_state_rx_shared,
            ros_fallback_joint_state_rx_shared,
            ros_last_command_source_shared,
            ros_arm_command_gate_shared,
            stop_event,
        ),
        kwargs=v308_kwargs,
        daemon=False,
        name="nero_mujoco_sim",
    )

    sim_process.start()
    if v308_on:
        threading.Thread(target=_v308_camera_supervisor, args=(stop_event,), daemon=True,
                         name="v308_camera_supervisor").start()

    try:
        run_gui(
            sim_process,
            target_shared,
            actual_shared,
            speed_shared,
            wn_shared,
            zeta_shared,
            qacc_shared,
            gravity_enabled,
            gripper_target_mm,
            gripper_actual_mm,
            gripper_force_mode,
            gripper_force_N,
            clamp_speed_mult_shared,
            anti_slip_boost_shared,
            anti_slip_slip_mm_s_shared,
            anti_slip_mu_shared,
            contact_left_n_shared,
            contact_right_n_shared,
            contact_tangent_n_shared,
            contact_motor_cmd_shared,
            hold_enabled_shared,
            hold_active_shared,
            hold_name_shared,
            hold_ready_shared,
            hold_now_request,
            hold_off_request,
            hold_status_code_shared,
            reset_zero_request,
            recover_request,
            safety_trip_count,
            safety_state,
            sim_ready,
            sim_error_code,
            ros_bridge_ready,
            ros_cmd_count,
            ros_feedback_count,
            ros_scene_count,
            ros_last_cmd_age,
            ros_last_cmd_joint_count,
            ros_controller_state_rx_shared,
            ros_fallback_joint_state_rx_shared,
            ros_last_command_source_shared,
            ros_arm_command_gate_shared,
            stop_event,
        )
    finally:
        stop_event.set()

        _v308_p = V308_SHARED.get("process")
        if _v308_p is not None:
            _v308_p.join(timeout=2.0)
            if _v308_p.is_alive():
                _v308_p.terminate()
        _v326_p = V326_SHARED.get("process")
        if _v326_p is not None:
            _v326_p.join(timeout=2.0)
            if _v326_p.is_alive():
                _v326_p.terminate()

        sim_process.join(timeout=2.0)

        if sim_process.is_alive():
            sim_process.terminate()
            sim_process.join(timeout=1.0)


if __name__ == "__main__":
    mp.freeze_support()

    # IMPORTANT:
    # spawn gives the MuJoCo viewer a fresh interpreter/process
    # and avoids inherited Qt/Tk/GLFW GUI state.
    mp.set_start_method("spawn", force=True)

    main()

# V2-22.2 CUROBO ADAPTER FIX

# V2-23 CUROBO MOVEIT REPLAN GATE

# V2-24 HIERARCHICAL HELD-OBJECT MANIPULATION

# V2-27 SEGMENTED EDGE REPOSITION + POSTURE RESET

# V2-28 NEAR-CLOSED TWO-FINGER EDGE REPOSITION

# V2-29 SAFE-HEIGHT TRANSIT EDGE REPOSITION

# V2-30 GUARDED INSERTION ROBUSTNESS

# V2-31 Q0 SAFE-HEIGHT REANCHOR

# V2-32 ENTRY-CONSISTENCY + TRACKING-TOLERANCE

# V2-32 R2 SAFETY0 SCOPE FIX

# V2-33 HIGH-TRANSIT FREE-PATH + TRACK-WARN TRUE-CONTINUE

# V2-34 STAGED EDGE-TRANSIT ORIENTATION DECOUPLING

# V2-35 EDGE-REPOSITION 300MM + STRICT PUSHER VERIFY + 180DEG CANDIDATE

# V2-36 PUSHER FAILOVER + 8DEG EPSILON + GUI STATUS LAMP

# V2-37 PUSHER 10DEG GATE + LOG-LAMP INSET

# V2-37 GUI LOG-INSET LARGE STATUS LAMP

# V2-38 EDGE FORCE TABLE_READY + VERTICAL-ONLY 300MM REANCHOR

# V2-38 EDGE REPOSITION STRICT TRANSIT + J7-ONLY PUSHER YAW

# V2-38 TASK-START TABLE-READY + SAFE WRIST-ONLY EDGE REPOSITION

# V2-38 R2 FAST PATCH INSTALLER

# V2-39 EDGE SAFE-START + HIGH-Z PUSHER POSTURE FIRST
# V2-39 SAFE-NEUTRAL TASK GATE + HIGH-Z PUSHER POSTURE FIRST

# V2-40 POSITION-FIRST + HIGH-Z WRIST-SPARSE EDGE PUSHER

# V2-41 HIGH-Z FREE-ZONE WHOLE-ARM POSTURE-FIRST EDGE PUSHER

# V2-42 PLAN-GRASP TOP-DOWN BRANCH REUSE EDGE PUSHER

# V2-43 PLAN-GRASP BRANCH + COMPACT PUSHER + FREE-PATH LIFT
