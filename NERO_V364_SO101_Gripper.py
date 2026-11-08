#!/usr/bin/env python3
"""NERO V3.64: the SO-101 open-source arm's gripper mounted on the NERO flange (7 mm adapter ring, 45 deg roll offset).

V3.64 SO-101 GRIPPER ON NERO

Real setup (user, V3.64): the SO-101 follower gripper (fixed jaw + STS3215 servo + rotating moving jaw) sits on the
NERO gripper mounting seat with a 7 mm adapter ring in between; because of the ring design the gripper is rolled by 45
deg about the ring normal (left-hand rule about the outward axis = -45 deg right-hand about link7 +x).

Geometry: SO-101 MJCF / meshes from TheRobotStudio/SO-ARM100 (Simulation/SO101, Apache-2.0; assets/so101/NOTICE.txt).
  * gripper body frame: z = the SO-101 wrist-roll axis (the jaws point along -z), the moving jaw hinge at
    (20.2, 18.8, -23.4) mm about -y, range -10 .. +100 deg; jaw inner faces measured from the meshes (stepped jaws,
    +-7.9 mm at the tips at 0 deg, i.e. the tips close to ~2.5 mm at -10 deg).
  * mount: seat face at link7 x = 37.5 mm (where the official gripper base starts), on the official tool axis
    (link7 y = 0, z = -23.5 mm); ring 7 mm; SO-101 interface face on the ring; nominal roll = closing axis along the
    official gripper's closing axis; plus the measured 45 deg offset.   ALL OF THIS IS CONFIGURABLE (MOUNT below, or
    NERO_V364_SO101_SEAT_X_MM / _RING_MM / _ROLL_DEG) - measure your adapter and correct it.
  * V3.75 NEW ADAPTER RING (user): the ring was redesigned - no 45 deg offset any more; the gripper sits like on the
    SO-101 arm (moving jaw on top, opening upwards) = the nominal roll -> MOUNT roll_deg default 0 (was -45).  The ring
    thickness was not re-measured: still 7 mm (NERO_V364_SO101_RING_MM / the real tab's field).
  * V3.76 SO-101 STANDARD ORIENTATION (user correction of V3.75): the SO-101 jaws open up / down - with the tool
    horizontal and J7 = 0 the moving jaw is on top and opens upwards.  The user's description ("from above, 90 deg
    clockwise") mapped to +90, the "moving jaw on top" to -90; rendered side by side (A -90 / B +90) the user chose
    A -> MOUNT roll_deg default -90 (= the real-validated old -45 ring turned 45 deg; V3.75: 0).  +90 would also make
    the cup stack unreachable with J7 limited to +-90 deg.
  * V3.76 SOFT PADS (sort_items only, soft_pads=True): the user's V3.75 run diverged (joint 5 / 7 at 6 -> 1168 rad/s)
    while an item was carried by the backend's CONNECT hold AND clamped by the stiff pads (solref 0.003); the sorting
    scene / twin use softer pad contacts (solref 0.006, solimp 0.95 0.99 0.001 - the items' own values).  The cup
    scene keeps the V3.64 pads.

Backend compatibility without touching the backend (its run_sim keeps the V3.42 bytes): the backend only adds the
official gripper if the model has no joint "gripper_joint1" (add_official_gripper_to_xml) and it drives / reads
  joint gripper_joint1 (actuators "gripper" position + "gripper_force" motor, opening = 2000 * q1 mm), joint
  gripper_joint2, bodies gripper_link1 / gripper_link2 and pad geoms finger1_* / finger2_*.
So the scene module writes the SO-101 gripper with those names:
  gripper_link1 = the SO-101 gripper body (fixed jaw + housing, no joint), pads finger1_rubber_pad*
  so101_driver  = a hidden 5 g slider carrying gripper_joint1 (range 0 .. 0.05 m) - the backend's force / position
                  control acts on it exactly as on the official finger slide
  gripper_link2 = the moving jaw, hinge so101_jaw_hinge = theta_min + k * q1 (joint equality "so101_jaw_coupling");
  gripper_joint2 is a massless slide = -gripper_joint1 exactly like the official gripper ("gripper_coupling"), so the
  backend's hard restore (qpos2 = -q1) stays exact
  -> the backend's "opening mm" is a virtual value 0 .. 100 (jaw -10 .. +100 deg); jaw_angle_deg() / tip_gap_mm() map it.
"""
import math
import os

VERSION = "V3.64"
MARKER = "V3.64 SO-101 GRIPPER ON NERO"
MARKER_V375 = "V3.75 NEW ADAPTER RING"
MARKER_V376 = "V3.76 SO-101 STANDARD ORIENTATION"
PAD_STIFF = ('solref="0.003 1" solimp="0.97 0.997 0.0003"')
PAD_SOFT = ('solref="0.006 1" solimp="0.95 0.99 0.001"')         # V3.76 SOFT PADS
HERE = os.path.dirname(os.path.realpath(__file__))
ASSET_DIR = os.path.join(HERE, "assets", "so101")
SCENE = "stack_cups_so101"

import numpy as np  # noqa: E402


def _env_f(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


MOUNT = {
    "seat_x_m": _env_f("NERO_V364_SO101_SEAT_X_MM", 37.5) / 1000.0,   # NERO gripper seat face along link7 x
    "axis_z_m": -0.0235,                                             # official tool axis (link7 z)
    "ring_m": _env_f("NERO_V364_SO101_RING_MM", 7.0) / 1000.0,         # adapter ring thickness
    # V3.76 SO-101 STANDARD ORIENTATION: -90 = moving jaw on top (user's choice A; V3.75: 0; V3.64-V3.74: -45)
    "roll_deg": _env_f("NERO_V364_SO101_ROLL_DEG", -90.0),
}
JAW = {"theta_min": math.radians(-10.0), "theta_max": math.radians(100.0), "q1_max": 0.05}
HINGE_POS = (0.0202, 0.0188, -0.0234)
HINGE_QUAT = (0.707107, 0.707107, -1.85362e-08, 1.85362e-08)      # wxyz, jaw z = gripper -y
# V3.64 TIP PADS: the bare SO-101 tips stay ~2.4 mm apart at the -10 deg hinge limit (4.7 mm with thin pads), so a
# cup wall cannot be pinched; like rubber / TPU tip pads on the real gripper, each tip gets a 4 mm pad (inner faces
# 5.0 mm from the jaw mid-plane at 0 deg) -> the jaws meet a 2 mm wall at about -6 deg, inside the hinge range.
PAD_HALF = (0.0020, 0.0055, 0.0100)                               # half thickness / width / length (m)
PAD_FIXED_CENTRE = (-0.0070, 0.0, -0.0920)                        # gripper frame; inner face x = -5.0 mm
PAD_FACE_X = -0.0050
WALL_HALF = 0.001                                                 # the grasp frame sits on the wall mid-surface
T_L7_TCP = np.array([[0, 0, 1, 0.1553], [-1, 0, 0, 0], [0, -1, 0, -0.0215], [0, 0, 0, 1.0]])


def _quat_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _R_quat(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s)
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = ((R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s)
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = ((R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = ((R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s)
    q = np.array(q)
    return q / np.linalg.norm(q)


def _rx(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def coupling_k():
    return (JAW["theta_max"] - JAW["theta_min"]) / JAW["q1_max"]


def jaw_angle_deg(opening_mm):
    """backend opening (virtual mm, 2000 * q1) -> moving jaw angle (deg, SO-101 MJCF convention)"""
    q1 = max(0.0, min(JAW["q1_max"], float(opening_mm) / 2000.0))
    return math.degrees(JAW["theta_min"] + coupling_k() * q1)


def opening_mm_for_angle(theta_deg):
    return 2000.0 * (math.radians(theta_deg) - JAW["theta_min"]) / coupling_k()


def tip_gap_mm(theta_deg, z_m=-0.100):
    """gap between the two pad faces at depth z (gripper frame) for a jaw angle - rotation of the moving pad face"""
    p = np.array([-PAD_FACE_X, 0.0, z_m])                         # moving pad face point at 0 deg (mirror of fixed)
    h = np.asarray(HINGE_POS)
    th = math.radians(theta_deg)
    # hinge axis = gripper -y: positive angle swings the jaw towards +x (opens)
    r = p - h
    c, s = math.cos(th), math.sin(th)
    x = h[0] + r[0] * c - r[2] * s
    return 1000.0 * (x - PAD_FACE_X)


def opening_mm_for_tip_gap(gap_mm, z_m=-0.100):
    lo, hi = math.degrees(JAW["theta_min"]), math.degrees(JAW["theta_max"])
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if tip_gap_mm(mid, z_m) < gap_mm:
            lo = mid
        else:
            hi = mid
    return opening_mm_for_angle(0.5 * (lo + hi))


def R_l7_gripper():
    """SO-101 gripper body in link7: z_g = -link7 x (jaws point along +x7), x_g (closing, towards the moving jaw) =
    -link7 z (nominal = the official closing axis), then the roll offset about link7 +x"""
    R_nom = np.column_stack([[0, 0, -1.0], [0, -1.0, 0], [-1.0, 0, 0]])
    return _rx(MOUNT["roll_deg"]) @ R_nom


def T_l7_gripper():
    T = np.eye(4)
    T[:3, :3] = R_l7_gripper()
    T[:3, 3] = [MOUNT["seat_x_m"] + MOUNT["ring_m"], 0.0, MOUNT["axis_z_m"]]
    return T


def T_gripper_grasp():
    """grasp frame G in the gripper frame: origin on the wall mid-surface next to the fixed-jaw pad face near the tips,
    z_G = approach (gripper -z), y_G = closing direction (fixed jaw -> moving jaw = gripper +x)"""
    T = np.eye(4)
    T[:3, :3] = np.column_stack([[0, 1.0, 0], [1.0, 0, 0], [0, 0, -1.0]])    # x_G = y_G x z_G (right-handed)
    T[:3, 3] = [PAD_FACE_X + WALL_HALF, 0.0, PAD_FIXED_CENTRE[2]]
    return T


def T_tcp_grasp():
    """official TCP (URDF tcp_link, used by MoveIt / IK) -> SO-101 grasp frame"""
    return np.linalg.inv(T_L7_TCP) @ T_l7_gripper() @ T_gripper_grasp()


def _f(*v):
    return " ".join(f"{float(x):.6g}" for x in v)


def gripper_parts(asset_dir=None, angle_unit="radian", soft_pads=False):
    """MJCF snippets for link7's children + asset / equality / actuator / contact sections (scene module + twin).
    angle_unit = the model's <compiler angle> (MJCF default: degree) - the hinge range is written in it; the coupling
    polycoef is joint data (radian per metre) and is not converted by MuJoCo."""
    ang = (lambda r: r) if angle_unit == "radian" else math.degrees
    a = os.path.abspath(asset_dir or ASSET_DIR)
    Tg = T_l7_gripper()
    qg = _R_quat(Tg[:3, :3])
    Rj = _quat_R(HINGE_QUAT)
    # V3.64 LINE-CONTACT MOVING PAD: parallel to the fixed pad at 0 deg, so at the clamp angle (~-6 deg) it touches the
    # wall along its tip edge.  A pad made parallel at the clamp angle gripped so firmly that a cup tipped by the
    # carry stayed tipped (8.7 deg) on the CONNECT hold; the line contact lets it hang upright again (twin).
    pad2_c = Rj.T @ (np.array([-PAD_FIXED_CENTRE[0], 0.0, PAD_FIXED_CENTRE[2]]) - np.asarray(HINGE_POS))
    fin2_c = Rj.T @ (np.array([0.0170, 0.0, -0.0720]) - np.asarray(HINGE_POS))
    pad = ('density="0" friction="10 0.3 0.05" condim="6" ' + (PAD_SOFT if soft_pads else PAD_STIFF) + ' '
           'margin="0.00025" rgba="0.15 0.15 0.15 1"')
    asset = (f'<mesh name="so101_follower" file="{a}/wrist_roll_follower_so101_v1.stl"/>'
             f'<mesh name="so101_servo" file="{a}/sts3215_03a_v1.stl"/>'
             f'<mesh name="so101_jaw" file="{a}/moving_jaw_so101_v1.stl"/>'
             '<material name="so101_yellow" rgba="1 0.82 0.12 1"/><material name="so101_black" rgba="0.1 0.1 0.1 1"/>'
             '<material name="v364_ring" rgba="0.55 0.57 0.60 1"/>')
    sx, rz = MOUNT["seat_x_m"], MOUNT["axis_z_m"]
    link7 = f"""
      <!-- V3.64 SO-101 GRIPPER ON NERO: seat (NERO gripper mounting seat, approximated) + 7 mm ring + SO-101 gripper -->
      <geom name="v364_seat" type="cylinder" fromto="{_f(0.004, 0, rz, sx, 0, rz)}" size="0.030" material="v364_ring"
            density="0" contype="0" conaffinity="0"/>
      <geom name="v364_ring" type="cylinder" fromto="{_f(sx, 0, rz, sx + MOUNT['ring_m'], 0, rz)}" size="0.026"
            material="v364_ring" density="0"/>
      <body name="gripper_link1" pos="{_f(*Tg[:3, 3])}" quat="{_f(*qg)}">
        <inertial pos="0.000213627 0.000245138 -0.025187" mass="0.087" diaginertia="2.75e-05 4.34e-05 3.45e-05"/>
        <geom type="mesh" mesh="so101_follower" pos="0 -0.000218214 0.000949706" quat="0 1 0 0" material="so101_yellow"
              contype="0" conaffinity="0" density="0" group="1"/>
        <geom type="mesh" mesh="so101_servo" pos="0.0077 0.0001 -0.0234" quat="0.707107 -0.707107 0 0" material="so101_black"
              contype="0" conaffinity="0" density="0" group="1"/>
        <geom name="so101_housing" type="box" pos="0 0 -0.021" size="0.031 0.024 0.019" rgba="0 0 0 0" density="0"/>
        <geom name="so101_fixed_finger" type="box" pos="-0.0160 0 -0.0700" size="0.0050 0.0080 0.0300" rgba="0 0 0 0"
              density="0"/>
        <geom name="finger1_rubber_pad" type="box" pos="{_f(*PAD_FIXED_CENTRE)}" size="{_f(*PAD_HALF)}" {pad}/>
        <body name="so101_driver" pos="0 0 -0.01">
          <inertial pos="0 0 0" mass="0.005" diaginertia="1e-07 1e-07 1e-07"/>
          <joint name="gripper_joint1" type="slide" axis="1 0 0" range="0 {JAW['q1_max']}" limited="true"/>
        </body>
        <!-- gripper_joint2 keeps the official meaning (= -gripper_joint1, written by the backend's hard restore) -->
        <body name="so101_driver2" pos="0 0 -0.01">
          <inertial pos="0 0 0" mass="0.005" diaginertia="1e-07 1e-07 1e-07"/>
          <joint name="gripper_joint2" type="slide" axis="1 0 0" range="-{JAW['q1_max']} 0" limited="true"/>
        </body>
        <body name="gripper_link2" pos="{_f(*HINGE_POS)}" quat="{_f(*HINGE_QUAT)}">
          <inertial pos="-0.00157495 -0.0300244 0.0192755" mass="0.012" diaginertia="6.6e-06 1.9e-06 5.3e-06"/>
          <joint name="so101_jaw_hinge" type="hinge" axis="0 0 1" range="{_f(ang(JAW['theta_min']), ang(JAW['theta_max']))}"
                 limited="true" damping="0.01" armature="0.0005"/>
          <geom type="mesh" mesh="so101_jaw" pos="0 0 0.0189" material="so101_yellow" contype="0" conaffinity="0"
                density="0" group="1"/>
          <geom name="so101_jaw_finger" type="box" pos="{_f(*fin2_c)}" size="0.0040 0.0300 0.0075" rgba="0 0 0 0" density="0"/>
          <geom name="finger2_rubber_pad" type="box" pos="{_f(*pad2_c)}" size="{_f(PAD_HALF[0], PAD_HALF[2], PAD_HALF[1])}"
                {pad}/>
        </body>
      </body>"""
    equality = ('<joint name="gripper_coupling" joint1="gripper_joint2" joint2="gripper_joint1" polycoef="0 -1 0 0 0"/>'
                f'<joint name="so101_jaw_coupling" joint1="so101_jaw_hinge" joint2="gripper_joint1" '
                f'polycoef="{_f(JAW["theta_min"], coupling_k(), 0, 0, 0)}" solref="0.004 1"/>')
    actuator = ('<position name="gripper" joint="gripper_joint1" kp="40" kv="5" forcerange="-20 20" forcelimited="true" '
                f'ctrlrange="0 {JAW["q1_max"]}" ctrllimited="true"/>'
                '<motor name="gripper_force" joint="gripper_joint1" gear="1" ctrlrange="-50 50" ctrllimited="true" '
                'forcerange="-50 50" forcelimited="true"/>')
    contact = '<exclude name="gripper_finger_self_exclude" body1="gripper_link1" body2="gripper_link2"/>'
    return {"asset": asset, "link7": link7, "equality": equality, "actuator": actuator, "contact": contact}


def angle_unit_of(xml):
    import re
    m = re.search(r"<compiler\b[^>]*\bangle\s*=\s*[\"']([a-zA-Z]+)", xml)
    return "radian" if (m and m.group(1).lower().startswith("rad")) else "degree"


def insert_into_model_xml(xml, soft_pads=False):
    """put the SO-101 gripper into a full NERO model XML string (link7 body, asset / equality / actuator / contact)"""
    import re
    p = gripper_parts(angle_unit=angle_unit_of(xml), soft_pads=soft_pads)
    m = re.search(r"<body\b[^>]*name=[\"']link7[\"'][^>]*>", xml)
    if not m:
        raise RuntimeError("link7 body not found - cannot mount the SO-101 gripper")
    if "gripper_joint1" in xml:
        raise RuntimeError("the model already has a gripper (gripper_joint1)")
    xml = xml[:m.end()] + p["link7"] + xml[m.end():]
    for tag, key in (("asset", "asset"), ("equality", "equality"), ("actuator", "actuator"), ("contact", "contact")):
        if re.search(rf"<{tag}\b[^>]*/>", xml):
            xml = re.sub(rf"<{tag}\b[^>]*/>", f"<{tag}>{p[key]}</{tag}>", xml, count=1)
        elif f"</{tag}>" in xml:
            xml = xml.replace(f"</{tag}>", p[key] + f"</{tag}>", 1)
        else:
            xml = xml.replace("</mujoco>", f"<{tag}>{p[key]}</{tag}>\n</mujoco>", 1)
    return xml


def describe():
    Tg = T_tcp_grasp()
    return {"seat_x_mm": round(MOUNT["seat_x_m"] * 1000, 1), "ring_mm": round(MOUNT["ring_m"] * 1000, 1),
            "roll_deg": MOUNT["roll_deg"], "grasp_point_in_tcp_mm": [round(float(v) * 1000, 1) for v in Tg[:3, 3]],
            "pregrasp_opening_virtual_mm": round(opening_mm_for_tip_gap(32.0), 1),
            "jaw_deg_at_tip_gap_2mm": round(jaw_angle_deg(opening_mm_for_tip_gap(2.0)), 1)}
