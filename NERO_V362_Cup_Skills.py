#!/usr/bin/env python3
"""NERO V3.62 stack_cups: Agent skills (skill contract + tools + agent loop) for the simulated NERO arm.

V3.62 STACK CUPS AGENT SKILLS

One skill logic, two "bodies" (the AGENTS.md principle):
  * BackendIO   the real simulation chain: NERO_V2_7_1_Unified_Backend (MuJoCo + ROS2/MoveIt/cuRobo).  The backend
                calls gui_run_task() from its existing `# V3.43 A2` block in llm_run_task_cmd (only change there);
                motion uses the backend's own planners (_v340_pose_with_fallback, MoveIt Cartesian paths), the
                gripper the backend's shared values (same force clamp + auto CONNECT hold as the egg skills).
  * TwinIO      a headless MuJoCo twin for offline tests / tuning: the official AgileX finger pads (copied 1:1 from
                add_official_gripper_to_xml) on a mocap link7 + the two scene cups; the backend's V22/V30/V39 gripper
                logic (velocity approach -> contact latch -> force PI with penetration guard, auto creep CONNECT
                hold) is re-implemented with the same constants.  The arm itself is NOT simulated in the twin;
                arm reachability is checked by plan_cup_stack() with strict 6-DoF URDF IK.

Skill CS "Cup Stack" (rim pinch, RoboDojo stack_cups style):
  observe_scene -> plan_cup_stack(source, target) -> pick_cup(source) -> place_cup_into(target) -> verify_stack
  pick_cup   : open -> pregrasp above the rim point -> straight descent (approach tilted parallel to the cup wall)
               -> force clamp (one finger inside, one outside the wall, pads below the rolled rim) -> straight lift
               until the cup's base is `transport_clearance` above the other cup's rim
  place_cup_into : carry over the target cup (rim angle chosen for reach, the cup turns about its own axis)
               -> closed-loop centring on the MEASURED cup poses -> lower to `release_gap` above the nested height
               -> open -> the cup slides down into the target -> retreat up

Why rim pinch (and not a side grasp around the body): when nested, only the top ~25 mm of the upper cup stays
above the lower cup's rim, and the rim (95 mm) is wider than the gripper opens (~90 mm between the pads) - a grasp
lower on the body would hit the target rim or catch the falling cup's rim.  Findings of the V3.62 twin study are in
CHANGELOG / AGENTS.md.

Honest limits: simulation only, ground-truth cup poses (JSON truth pose source; no camera perception for cups yet),
not run on hardware.
"""
import json
import math
import os
import sys
import threading
import time

VERSION = "V3.62"
MARKER = "V3.62 STACK CUPS AGENT SKILLS"
HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np                     # noqa: E402
import NERO_V343_Scenes as SC          # noqa: E402

CUPS = {"cup_1": "obj_cup_1", "cup_2": "obj_cup_2"}          # Agent name -> MuJoCo body (TF: sim_obj_cup_N)
TABLE_Z = 0.0                                                # table top
JOINT_LIMITS = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95),
                (-1.5708, 1.5708)]
CARRY_SPEED = 0.04             # m/s while a cup is held (V3.62 twin: 0.10 m/s + ~1 rad/s swung it out of the pinch)
CARRY_SLOW_FACTOR = 2.5        # backend: carry trajectory slowed by this factor (_v218_slow_trajectory_in_place)
PLACE_SPEED = 0.01             # m/s for centring + lowering into the target (twin: 0.03 m/s made the cup swing)
MIN_MARGIN_RELAXED_DEG = 4.0   # below this (V3.41 relaxed limit) the plan is refused, not just warned
MIN_MARGIN_DEG = 8.0                                         # V3.41 rule: every joint >= 8 deg from its limits
CLAMP_CENTRE_Y_M = 0.002       # the two pads close on TCP y = +2.0 mm (finger 1 / finger 2 stud faces, measured)
T_L7_TCP = np.array([[0, 0, 1, 0.1553], [-1, 0, 0, 0], [0, -1, 0, -0.0215], [0, 0, 0, 1.0]])   # URDF tool frame

CUP_KNOB_DEFAULTS = {          # fallbacks if NERO_V359_Strategy is unavailable (same values as its KNOB_SPECS)
    "cup_rim_pinch_depth_mm": 35.0, "cup_grasp_rim_angle_deg": 40.0, "cup_place_rim_angle_deg": 35.0,
    "cup_pregrasp_height_mm": 90.0, "cup_transport_clearance_mm": 20.0, "cup_release_gap_mm": 85.0,
    "cup_release_xy_tol_mm": 7.5, "cup_release_tilt_tol_deg": 3.0, "cup_nested_height_mm": 25.3,
    "cup_clamp_force_n": 5.0, "cup_wall_tilt_follow": True,
}


def cup_knobs(profile=None):
    """cup knobs of the active skill (logs/v360_active_strategy.json -> NERO_V359_Strategy) -> (knobs, skill)"""
    k = dict(CUP_KNOB_DEFAULTS)
    skill = {"code": "CS", "short_name": "Cup Stack", "skill_id": "cup_stack_rim_pinch", "source": "defaults"}
    try:
        import NERO_V359_Strategy as ST
        prof = profile or ST.get_active_strategy(reload=True)
        kn = ST.sanitize_knobs(prof.knobs)
        if profile is None and "stack_cups" not in (prof.skill_card or {}).get("scene_tags", []) and \
                all(kn.get(key) == ST.BASELINE_KNOBS.get(key) for key in k if key in ST.BASELINE_KNOBS):
            # the active skill is not a cup skill and has no own cup values -> the built-in cup skill CS
            cs = next(p for p in ST.builtin_strategy_presets() if p.code == "CS")
            note = f"built-in CS (active skill {prof.code} has no cup-specific values)"
            prof, kn = cs, ST.sanitize_knobs(cs.knobs)
            skill_src = note
        else:
            skill_src = "active skill"
        for key in k:
            if key in kn:
                k[key] = kn[key]
        skill = {"code": prof.code or "??", "short_name": prof.short_name or prof.skill_id,
                 "skill_id": prof.skill_id, "source": skill_src}
    except Exception:
        pass
    return k, skill


# =============================================================================================== geometry
def geom():
    g = SC.cup_geometry()
    g["beta"] = math.atan2(g["r_top"] - g["r_bot"], g["h"])           # wall half-angle (8.2 deg)
    return g


def quat_wxyz_to_R(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_quat_xyzw(R):
    R = np.asarray(R, float)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return [float(v) for v in q / np.linalg.norm(q)]


def quat_xyzw_to_R(q):
    x, y, z, w = [float(v) for v in q]
    return quat_wxyz_to_R((w, x, y, z))


def cup_axis_points(rim_xyz, quat_wxyz):
    """scene cup body origin = rim centre (V3.62) -> (rim centre, base centre, axis unit vector)"""
    g = geom()
    ax = quat_wxyz_to_R(quat_wxyz)[:, 2]
    rim = np.asarray(rim_xyz, float)
    return rim, rim - g["h"] * ax, ax


def pinch_frame(phi_rad, flip, tilt_follow=True):
    """TCP rotation for a rim pinch at rim angle phi (closing axis = TCP y radial, approach = TCP z down).
    flip = the symmetric gripper turned 180 deg about its approach axis.  With tilt_follow the approach leans
    inward by the wall half-angle so both pads are parallel to the conical wall (V3.62 twin: a vertical approach
    lets the pads touch only the rim / one point and the cup is twisted out)."""
    yaw = phi_rad + math.pi / 2 + (math.pi if flip else 0.0)
    z = np.array([0.0, 0.0, -1.0])
    x = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    y = np.cross(z, x)                                   # = outward radial (flip: inward)
    R = np.column_stack([x, y, z])
    t = geom()["beta"] * (1.0 if not flip else -1.0) if tilt_follow else 0.0
    c, s = math.cos(t), math.sin(t)
    return R @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def pinch_tcp(rim_centre, phi_rad, depth_m, flip, tilt_follow=True):
    """TCP position + rotation pinching the wall at rim angle phi, `depth` (along the wall) below the rim."""
    g = geom()
    u = np.array([math.cos(phi_rad), math.sin(phi_rad), 0.0])
    R = pinch_frame(phi_rad, flip, tilt_follow)
    top = np.asarray(rim_centre, float) + g["r_top"] * u              # wall mid-surface at the rim
    wall_down = np.array([-math.sin(g["beta"]) * u[0], -math.sin(g["beta"]) * u[1], -math.cos(g["beta"])])
    p = top + depth_m * (wall_down if tilt_follow else np.array([0.0, 0.0, -1.0]))
    p = p - CLAMP_CENTRE_Y_M * R[:, 1]                                 # pad clamp centre on the wall
    return p, R


def base_dir(xy):
    """direction from a cup axis to the robot base (rad)"""
    return math.atan2(-xy[1], -xy[0])


# =============================================================================================== planning (URDF IK)
def _ik_tools():
    import NERO_V315_Consistency_Safety as CS
    import NERO_V361_Cup_Stack as C61
    U = CS.load_sim_urdf()
    return U, [f"joint{i}" for i in range(1, 8)], CS.tool_transform_from_sim(U), C61.pose_ik


_IK = {}


def ik_chain(chain, q_start=None):
    """strict 6-DoF IK along [(name, xyz, R), ...]; -> dict(ok, why, min_margin_deg, binding, q)"""
    if "t" not in _IK:
        _IK["t"] = _ik_tools()
    U, J, T, pose_ik = _IK["t"]
    q_prev, worst, binding, qs = (list(q_start) if q_start else None), 1e9, "", []
    for name, p, R in chain:
        a1 = math.atan2(p[1], p[0])
        a1 = a1 + math.pi if a1 < 0 else a1 - math.pi
        seeds = ([q_prev] if q_prev else []) + [[a1, j2, 0.0, j4, 0.0, j6, j7] for j2 in (0.2, 0.6) for j4 in (0.8, 1.6)
                                                for j6 in (0.5, -0.2) for j7 in (0.0, 1.0, -1.0)]
        q, pe, re = pose_ik(U, J, JOINT_LIMITS, seeds, T, [float(v) for v in p], np.asarray(R).tolist())
        if q is None:
            return {"ok": False, "why": f"{name}: no 6-DoF IK (pos {pe * 1000:.1f} mm, rot {re:.1f} deg)",
                    "min_margin_deg": None, "binding": name, "q": qs}
        ms = [math.degrees(min(v - lo, hi - v)) for v, (lo, hi) in zip(q, JOINT_LIMITS)]
        k = min(range(7), key=lambda i: ms[i])
        if ms[k] < worst:
            worst, binding = ms[k], f"J{k + 1} at {name}"
        qs.append(q)
        q_prev = q
    return {"ok": True, "why": "", "min_margin_deg": round(worst, 1), "binding": binding, "q": qs}


def _angle_order(pref_deg, step=15, span=180):
    out = [0]
    for d in range(step, span + 1, step):
        out += [d, -d]
    return [math.radians(pref_deg + d) for d in out if d != -180]


def stack_heights(knobs, src_base_z, dst_base_z):
    """TCP-relevant heights: lift (base of the carried cup `clearance` above the target rim) and release
    (base at target base + nested + gap)."""
    g = geom()
    clear = float(knobs["cup_transport_clearance_mm"]) / 1000.0
    base_lift = max(src_base_z, dst_base_z) + g["h"] + clear
    base_release = dst_base_z + float(knobs["cup_nested_height_mm"]) / 1000.0 + float(knobs["cup_release_gap_mm"]) / 1000.0
    return base_lift, base_release


def plan_cup_stack(knobs, src_rim, src_quat, dst_rim, dst_quat, q_start=None, log=None):
    """rim pinch plan: grasp rim angle + flip (pregrasp / grasp / lift reachable), then place rim angle with the SAME
    flip (the cup axis offset in the TCP frame is fixed by the grasp; V3.61 chose the flips independently, which
    would have put the carried cup 2 x rim radius off the target).  Knob angles are relative to the cup -> base
    direction and are tried first, then +-15 deg steps.  -> dict"""
    g = geom()
    depth = float(knobs["cup_rim_pinch_depth_mm"]) / 1000.0
    pre = float(knobs["cup_pregrasp_height_mm"]) / 1000.0
    tilt = bool(knobs.get("cup_wall_tilt_follow", True))
    s_rim, s_base, _ = cup_axis_points(src_rim, src_quat)
    d_rim, d_base, _ = cup_axis_points(dst_rim, dst_quat)
    base_lift, base_release = stack_heights(knobs, s_base[2], d_base[2])
    dz_lift = base_lift - s_base[2]
    best_g = None
    tried = 0
    for phi in _angle_order(math.degrees(base_dir(s_rim)) + float(knobs["cup_grasp_rim_angle_deg"])):
        for flip in (False, True):
            p, R = pinch_tcp(s_rim, phi, depth, flip, tilt)
            ch = [("pregrasp", p + [0, 0, pre], R), ("grasp", p, R), ("lift", p + [0, 0, dz_lift], R)]
            r = ik_chain(ch, q_start)
            tried += 1
            if r["ok"] and (best_g is None or r["min_margin_deg"] > best_g["r"]["min_margin_deg"]):
                best_g = {"phi": phi, "flip": flip, "r": r, "tcp": p, "R": R}
        if best_g and best_g["r"]["min_margin_deg"] >= MIN_MARGIN_DEG:
            break
    if best_g is None:
        return {"ok": False, "reason": "GRASP_UNREACHABLE", "detail": f"no reachable rim pinch on the source cup ({tried} tried)"}
    flip = best_g["flip"]
    best_p = None
    lift_tcp = best_g["tcp"] + [0, 0, dz_lift]
    for phi in _angle_order(math.degrees(base_dir(d_rim)) + float(knobs["cup_place_rim_angle_deg"])):
        # rim centre of the carried cup when it is over the target at lift / release height
        rim_lift = np.array([d_base[0], d_base[1], base_lift + g["h"]])
        rim_rel = np.array([d_base[0], d_base[1], base_release + g["h"]])
        p1, R1 = pinch_tcp(rim_lift, phi, depth, flip, tilt)
        p2, _ = pinch_tcp(rim_rel, phi, depth, flip, tilt)
        ch = [("above_target", p1, R1), ("release", p2, R1), ("retreat", p2 + [0, 0, 0.08], R1)]
        r = ik_chain(ch, best_g["r"]["q"][-1] if best_g["r"]["q"] else None)
        if r["ok"] and (best_p is None or r["min_margin_deg"] > best_p["r"]["min_margin_deg"]):
            best_p = {"phi": phi, "r": r, "tcp_above": p1, "tcp_release": p2, "R": R1}
        if best_p and best_p["r"]["min_margin_deg"] >= MIN_MARGIN_DEG:
            break
    if best_p is None:
        return {"ok": False, "reason": "PLACE_UNREACHABLE",
                "detail": "the source cup can be grasped but no reachable carry/release pose over the target with the same grip"}
    wq = lambda R: R_to_quat_xyzw(R)
    wps = [("pregrasp", best_g["tcp"] + [0, 0, pre], best_g["R"]), ("grasp", best_g["tcp"], best_g["R"]),
           ("lift", lift_tcp, best_g["R"]), ("above_target", best_p["tcp_above"], best_p["R"]),
           ("release", best_p["tcp_release"], best_p["R"]), ("retreat", best_p["tcp_release"] + [0, 0, 0.08], best_p["R"])]
    margin = min(best_g["r"]["min_margin_deg"], best_p["r"]["min_margin_deg"])
    out = {"ok": True, "flip": bool(flip),
           "grasp_rim_angle_rel_deg": round((math.degrees(best_g["phi"] - base_dir(s_rim)) + 180) % 360 - 180, 1),
           "place_rim_angle_rel_deg": round((math.degrees(best_p["phi"] - base_dir(d_rim)) + 180) % 360 - 180, 1),
           "grasp_margin_deg": best_g["r"]["min_margin_deg"], "grasp_binding": best_g["r"]["binding"],
           "place_margin_deg": best_p["r"]["min_margin_deg"], "place_binding": best_p["r"]["binding"],
           "min_joint_margin_deg": margin, "margin_ok": bool(margin >= MIN_MARGIN_DEG),
           "base_lift_z_m": round(base_lift, 4), "base_release_z_m": round(base_release, 4),
           "waypoints": [{"name": n, "tcp_xyz_m": [round(float(v), 4) for v in p], "quat_xyzw": [round(v, 5) for v in wq(R)]}
                         for n, p, R in wps],
           "_wps": {n: (np.asarray(p, float), np.asarray(R, float)) for n, p, R in wps}}
    if log:
        log(f"[V3.62 CUP PLAN] grasp rim {out['grasp_rim_angle_rel_deg']:+.0f} deg (margin {out['grasp_margin_deg']} deg, "
            f"{out['grasp_binding']}) | place rim {out['place_rim_angle_rel_deg']:+.0f} deg (margin {out['place_margin_deg']} "
            f"deg, {out['place_binding']}) | flip={flip}")
    return out


# =============================================================================================== IO: headless MuJoCo twin
def _finger_xml(name, quat, axis, rng, pad):
    studs = []
    i = 0
    for sx in (-0.0115, -0.0058, 0.0, 0.0058, 0.0115):
        for sy in (-0.043, -0.035, -0.027, -0.019, -0.011, -0.003, 0.005, 0.010):
            i += 1
            studs.append(f'<geom name="{pad}_stud_{i:02d}" type="sphere" pos="{sx:.4f} {sy:.4f} -0.0052" size="0.0017" '
                         'rgba="1 .65 0 1" density="0" friction="10 0.3 0.05" condim="6" solref="0.003 1" '
                         'solimp="0.97 0.997 0.0003" margin="0.0002"/>')
    j = "gripper_joint1" if name.endswith("1") else "gripper_joint2"
    return (f'<body name="{name}" pos="0.1733 0 -0.0235" quat="{quat}">'
            '<inertial pos="0.000651 -0.049193 0.009723" quat="0.477798 0.572974 -0.518028 0.418398" mass="0.025" '
            'diaginertia="7.74531e-05 7.36783e-05 5.0886e-06"/>'
            f'<joint name="{j}" type="slide" axis="{axis}" range="{rng}" limited="true"/>'
            f'<geom name="{pad}_rubber_pad" type="box" pos="0 -0.018 -0.0020" size="0.0145 0.030 0.0020" rgba=".9 .08 .03 1" '
            'density="0" friction="10 0.30 0.05" condim="6" solref="0.003 1" solimp="0.97 0.997 0.0003" margin="0.00025"/>'
            '<geom type="box" pos="0 -0.075 0.006" size="0.010 0.032 0.006" rgba=".25 .25 .28 1" contype="0" conaffinity="0" density="0"/>'
            + "".join(studs) + "</body>")


def twin_xml(cup_xy=None):
    xy = dict(SC.CUP_XY if cup_xy is None else cup_xy)
    cups = "\n".join(SC.cup_body_xml(n, xy[n], z0=0.0005, origin=SC.CUP_ORIGIN) for n in SC.CUP_NAMES)
    holds = "".join(f'<connect name="hold_{n}" body1="link7" body2="{n}" anchor="0 0 0" active="false" '
                    'solref="0.008 1" solimp="0.98 0.995 0.001"/>' for n in SC.CUP_NAMES)
    return f"""<mujoco model="nero_v362_cup_twin">
  <option timestep="0.001" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="960"/></visual>
  <asset>{SC.CUP_ASSETS}</asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="play_table" type="box" pos="-0.385 0 -0.03" size="0.335 0.38 0.03" rgba=".93 .93 .91 1"
          friction="1.2 0.01 0.001" solref="0.008 1" solimp="0.96 0.995 0.0005" margin="0.001"/>
    <body name="link7" mocap="true" pos="-0.3 0 0.45">
      <!-- APPROXIMATE gripper housing (the official STL meshes are not in this repository) -->
      <geom name="gripper_base_box" type="box" pos="0.0675 0 -0.0235" size="0.0325 0.040 0.032" rgba=".3 .3 .35 .5"/>
      {_finger_xml("gripper_link1", "-1.27026e-06 0.707108 0.707105 -1.30815e-06", "0 0 1", "0 0.05", "finger1")}
      {_finger_xml("gripper_link2", "0.707105 -1.30815e-06 1.30815e-06 -0.707108", "0 0 -1", "-0.05 0", "finger2")}
    </body>
    {cups}
  </worldbody>
  <contact><exclude body1="gripper_link1" body2="gripper_link2"/></contact>
  <equality>
    <joint name="gripper_coupling" joint1="gripper_joint2" joint2="gripper_joint1" polycoef="0 -1 0 0 0"/>
    {holds}
  </equality>
  <actuator>
    <position name="gripper" joint="gripper_joint1" kp="40" kv="5" forcerange="-20 20" forcelimited="true"
              ctrlrange="0 0.05" ctrllimited="true"/>
    <motor name="gripper_force" joint="gripper_joint1" gear="1" ctrlrange="-50 50" ctrllimited="true"
           forcerange="-50 50" forcelimited="true"/>
  </actuator>
</mujoco>"""


class TwinIO:
    """Headless MuJoCo twin of the backend's gripper + cups (see module docstring)."""
    name = "twin"

    def __init__(self, log=None, cup_xy=None, start_tcp=(-0.30, 0.0, 0.40)):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_string(twin_xml(cup_xy))
        self.d = mujoco.MjData(self.m)
        self.log = log or (lambda s: None)
        self.mid = self.m.body("link7").mocapid[0]
        self.l7 = self.m.body("link7").id
        self.ga, self.fa = self.m.actuator("gripper").id, self.m.actuator("gripper_force").id
        self.j1 = self.d.joint("gripper_joint1")
        self.pad1, self.pad2, self.obj = set(), set(), set()
        for gi in range(self.m.ngeom):
            n = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
            if n.startswith(("finger1_rubber_pad", "finger1_stud_")):
                self.pad1.add(gi)
            elif n.startswith(("finger2_rubber_pad", "finger2_stud_")):
                self.pad2.add(gi)
            elif n.startswith("obj_"):
                self.obj.add(gi)
        self.force_mode, self.force_n, self.target_q = False, 5.0, 0.02
        self.hold, self.trips, self.t = None, 0, 0.0
        self._clamp_prev = False
        self.latched, self.fint, self.mcmd, self.t_contact = False, 0.0, 0.0, 0.0
        self.hold_t, self.prev_rel, self.vlp = 0.0, None, 0.0
        self.base_hits = 0
        self.max_ang_rate = 0.30          # rad/s, like the slowed backend carry trajectory
        R0 = pinch_frame(0.0, False, False)
        self._set_tcp(np.array(start_tcp, float), R0)
        mujoco.mj_forward(self.m, self.d)
        self.step(400)

    # -- primitives
    def _set_tcp(self, p, R):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        L = T @ np.linalg.inv(T_L7_TCP)
        q = np.zeros(4)
        self.mj.mju_mat2Quat(q, np.asarray(L[:3, :3], float).reshape(-1))
        self.d.mocap_pos[self.mid] = L[:3, 3]
        self.d.mocap_quat[self.mid] = q

    def _tcp_T(self):
        T = np.eye(4)
        T[:3, :3] = self.d.xmat[self.l7].reshape(3, 3)
        T[:3, 3] = self.d.xpos[self.l7]
        return T @ T_L7_TCP

    def _pad_info(self):
        lt = rt = False
        ln = rn = 0.0
        worst = 0.0
        cnt = {}
        cf = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if "gripper_base_box" in ((self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_GEOM, g1) or ""),
                                      (self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_GEOM, g2) or "")):
                if (g1 in self.obj or g2 in self.obj) and c.dist < 0:
                    self.base_hits += 1
                continue
            pad = g1 if g1 in self.pad1 | self.pad2 else (g2 if g2 in self.pad1 | self.pad2 else -1)
            other = g2 if pad == g1 else g1
            if pad < 0 or other not in self.obj:
                continue
            self.mj.mj_contactForce(self.m, self.d, i, cf)
            left = pad in self.pad1
            if c.dist <= 0.0005:
                lt, rt = (True, rt) if left else (lt, True)
            if left:
                ln += abs(cf[0])
            else:
                rn += abs(cf[0])
            worst = min(worst, float(c.dist))
            b = int(self.m.geom_bodyid[other])
            cnt[b] = cnt.get(b, 0) + 1
        return lt, rt, ln, rn, worst, (max(cnt, key=cnt.get) if cnt else -1)

    def _activate_hold(self, bid):
        name = self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_BODY, bid)
        try:
            eq = self.m.equality("hold_" + name).id
        except Exception:
            return
        R = self.d.xmat[self.l7].reshape(3, 3)
        self.m.eq_data[eq, 0:3] = R.T @ (self.d.xpos[bid] - self.d.xpos[self.l7])
        self.m.eq_data[eq, 3:6] = 0.0
        self.d.eq_active[eq] = 1
        self.hold = name

    def _release_hold(self):
        for n in SC.CUP_NAMES:
            self.d.eq_active[self.m.equality("hold_" + n).id] = 0
        self.hold, self.hold_t, self.prev_rel, self.vlp = None, 0.0, None, 0.0

    def step(self, n=1):
        """backend run_sim gripper logic (V22 force PI / V39 velocity approach / V30 auto creep CONNECT hold),
        constants copied from NERO_V2_7_1_Unified_Backend"""
        dt = 0.001
        for _ in range(int(n)):
            q, qd = float(self.j1.qpos[0]), float(self.j1.qvel[0])
            lt, rt, ln, rn, worst, bid = self._pad_info()
            self.L, self.Rn = ln, rn
            if self.force_mode:
                if not self._clamp_prev:
                    self.latched, self.fint, self.mcmd, self.t_contact = False, 0.0, 0.0, 0.0
                    self._release_hold()
                self._clamp_prev = True
                self.d.ctrl[self.ga] = min(max(q, 0.0), 0.05)
                if lt and rt and not self.latched:
                    self.latched, self.t_contact, self.fint = True, 0.0, 0.0
                if self.latched and not (lt and rt) and self.hold is None:
                    self.latched, self.fint = False, 0.0
                if not self.latched:
                    v = 0.0030 * (15.0 if not (lt or rt) else 1.5)
                    f = float(np.clip(25.0 * (-v - qd), -0.60, 0.60))
                    if qd < -v:
                        f = max(0.0, f)
                    self.mcmd = max(0.0, -f)
                    self.d.ctrl[self.fa] = f
                else:
                    self.t_contact += dt
                    tgt = self.force_n * min(1.0, self.t_contact / 2.0)
                    err = tgt - min(ln, rn)
                    if abs(err) < 8.0:
                        self.fint = float(np.clip(self.fint + err * dt, -12.0, 12.0))
                    des = float(np.clip(1.20 * tgt + 0.70 * err + 0.80 * self.fint, 0.0, 30.0))
                    self.mcmd = min(des, self.mcmd + 4.0 * dt) if des > self.mcmd else max(des, self.mcmd - 10.0 * dt)
                    mf = -self.mcmd - 8.0 * qd
                    pen = max(0.0, -worst)
                    if pen > 0.00025:
                        mf = -self.mcmd * float(np.clip(1.0 - (pen - 0.00025) / 0.00030, 0.0, 1.0)) - 8.0 * qd
                    if pen > 0.00055:
                        mf = max(0.0, -8.0 * qd)
                    if pen > 0.00080:
                        mf = 0.20
                    self.d.ctrl[self.fa] = float(np.clip(mf, -50.0, 50.0))
                if self.hold is None and bid >= 0:
                    rel = float(self.d.xpos[bid, 2] - self.d.xpos[self.l7, 2])
                    if self.prev_rel is not None:
                        self.vlp += dt / (0.080 + dt) * ((rel - self.prev_rel) / dt - self.vlp)
                    self.prev_rel = rel
                    self.hold_t = min(0.10, self.hold_t + dt) if self.vlp * 1000.0 <= -0.05 else max(0.0, self.hold_t - 2 * dt)
                    if self.hold_t >= 0.10:
                        self._activate_hold(bid)
            else:
                self._clamp_prev = False
                if self.hold is not None:
                    self._release_hold()
                self.latched = False
                self.d.ctrl[self.fa] = 0.0
                self.d.ctrl[self.ga] = self.target_q
            self.mj.mj_step(self.m, self.d)
            self.t += dt

    # -- IO interface
    def emit(self, s):
        self.log(s)

    def sleep(self, s):
        self.step(int(max(0.0, s) / 0.001))

    def tcp_pose(self):
        T = self._tcp_T()
        return [float(v) for v in T[:3, 3]], R_to_quat_xyzw(T[:3, :3])

    def cup_pose(self, cup):
        b = self.d.body(CUPS[cup])
        return [float(v) for v in b.xpos], [float(v) for v in b.xquat]

    def _move(self, label, xyz, quat, speed):
        T0 = self._tcp_T()
        p0, p1 = T0[:3, 3].copy(), np.asarray(xyz, float)
        q0, q1 = np.zeros(4), np.zeros(4)
        self.mj.mju_mat2Quat(q0, np.asarray(T0[:3, :3], float).reshape(-1))
        self.mj.mju_mat2Quat(q1, np.asarray(quat_xyzw_to_R(quat), float).reshape(-1))
        if np.dot(q0, q1) < 0:
            q1 = -q1
        ang = 2 * math.acos(min(1.0, abs(float(np.dot(q0, q1)))))
        n = max(2, int(max(np.linalg.norm(p1 - p0) / max(speed, 1e-3), ang / self.max_ang_rate) / 0.001))
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)
            q = q0 * (1 - s) + q1 * s
            q /= np.linalg.norm(q)
            R = np.zeros(9)
            self.mj.mju_quat2Mat(R, q)
            self._set_tcp(p0 + s * (p1 - p0), R.reshape(3, 3))
            self.step(1)
        self.step(250)
        err = float(np.linalg.norm(self._tcp_T()[:3, 3] - p1))
        return {"ok": err < 0.003, "tcp_error_mm": round(err * 1000, 2), "label": label}

    def move_pose(self, label, xyz, quat, speed=0.10, slow=None):
        return self._move(label, xyz, quat, speed)

    def move_line(self, label, xyz, quat, speed=0.05, slow=None):
        return self._move(label, xyz, quat, speed)

    def gripper_open(self, mm):
        self.force_mode = False
        self.target_q = float(np.clip(mm / 2000.0, 0.0, 0.05))
        self.step(900)
        return {"opening_mm": round(self.opening_mm(), 1)}

    def gripper_clamp(self, force_n, settle_s=2.2):
        self.force_n, self.force_mode = float(force_n), True
        t_end = self.t + 3.2
        while self.t < t_end and not self.latched:
            self.step(20)
        self.step(int(settle_s / 0.001))
        lt, rt, ln, rn, worst, bid = self._pad_info()
        return {"contact_left": bool(lt), "contact_right": bool(rt), "left_N": round(ln, 2), "right_N": round(rn, 2),
                "hold_active": self.hold is not None, "opening_mm": round(self.opening_mm(), 1)}

    def gripper_state(self):
        lt, rt, ln, rn, worst, bid = self._pad_info()
        return {"opening_mm": round(self.opening_mm(), 1), "hold_active": self.hold is not None,
                "contact_left": bool(lt), "contact_right": bool(rt), "force_mode": self.force_mode}

    def opening_mm(self):
        return 2000.0 * float(self.j1.qpos[0])

    def safety_trips(self):
        return self.trips

    def go_home(self, label):
        p, q = self.tcp_pose()
        self._move(label, [p[0], p[1], max(p[2], 0.30)], q, 0.15)
        return self._move(label, [-0.30, 0.0, 0.40], R_to_quat_xyzw(pinch_frame(0.0, False, False)), 0.15)

    def snapshot(self, path, lookat=None, azimuth=135, elevation=-25, distance=0.75):
        """PNG of the twin (needs an OpenGL context, e.g. MUJOCO_GL=glfw with a DISPLAY)"""
        from PIL import Image
        r = self.mj.Renderer(self.m, 480, 640)
        cam = self.mj.MjvCamera()
        cam.lookat[:] = lookat if lookat is not None else [-0.35, 0.0, 0.10]
        cam.azimuth, cam.elevation, cam.distance = azimuth, elevation, distance
        r.update_scene(self.d, cam)
        Image.fromarray(r.render()).save(path)
        return path


# =============================================================================================== IO: real backend
class BackendIO:
    """Adapter to NERO_V2_7_1_Unified_Backend (built from the ctx dict its A2 block passes in)."""
    name = "backend"

    def __init__(self, ctx):
        self.c = ctx
        self.core = ctx["skill_core"]
        self.sh = ctx["shared"]
        self._now = ctx.get("time_fn") or time.time            # tests inject the twin's simulation clock
        self._sleep = ctx.get("sleep_fn") or time.sleep

    def emit(self, s):
        self.core.emit(s)

    def sleep(self, s):
        self._sleep(max(0.0, s))

    def tcp_pose(self):
        p, q = self.core.current_tcp_pose()
        return [float(v) for v in p], [float(v) for v in q]

    def cup_pose(self, cup):
        p, q = self.core.object_pose(cup)               # TF sim_obj_cup_N (body origin = rim centre)
        return [float(v) for v in p], [float(q[3]), float(q[0]), float(q[1]), float(q[2])]

    def _post(self, label, xyz, s0):
        self._sleep(0.15)
        best = 1e9
        t_end = self._now() + 1.8
        while self._now() < t_end:
            p, _q = self.tcp_pose()
            best = min(best, float(np.linalg.norm(np.asarray(p) - np.asarray(xyz))))
            if best <= 0.006:
                break
            self._sleep(0.12)
        trips = int(self.sh["safety_trip_count"].value) - s0
        return {"ok": best <= 0.010 and trips == 0, "tcp_error_mm": round(best * 1000, 1), "safety_trips": trips,
                "label": label}

    def move_pose(self, label, xyz, quat, speed=0.10, slow=None):
        self.c["require_arm"]()
        s0 = int(self.sh["safety_trip_count"].value)
        try:
            traj = self.c["pose_plan"](label, list(xyz), list(quat), 0.12, 0.12)
            if slow and self.c.get("slow") is not None and traj is not None:
                self.c["slow"](traj, factor=float(slow))          # V3.62: gentle carry while a cup is held
            self.core._execute_trajectory(traj)
        except Exception as exc:
            return {"ok": False, "label": label, "error": str(exc)[:300], "tcp_error_mm": None}
        return self._post(label, xyz, s0)

    def move_line(self, label, xyz, quat, speed=0.05, fallback=True, slow=None):
        self.c["require_arm"]()
        s0 = int(self.sh["safety_trip_count"].value)
        try:
            traj = self.core._cartesian_waypoint_plan([list(xyz)], list(quat), label=label, max_step=0.004,
                                                      max_speed=float(speed))
            if traj is not None:
                if (slow or float(speed) <= 0.02) and self.c.get("slow") is not None:
                    # V3.62: max_cartesian_speed is not in every MoveIt release -> also stretch the gentle moves in time
                    self.c["slow"](traj, factor=float(slow or 2.5))
                self.core._execute_trajectory(traj)
        except Exception as exc:
            if not fallback or "PHYSICAL" in str(exc) or "SAFETY" in str(exc):
                return {"ok": False, "label": label, "error": str(exc)[:300], "tcp_error_mm": None}
            self.emit(f"[WARN] [V3.62 CUP] {label}: straight Cartesian path refused ({str(exc)[:120]}); "
                      "using the collision-checked pose planner instead")
            r = self.move_pose(label, xyz, quat, speed, slow=slow)
            r["cartesian_fallback"] = True
            return r
        return self._post(label, xyz, s0)

    def gripper_open(self, mm):
        sh = self.sh
        sh["hold_off_request"].value = True
        sh["gripper_force_mode"].value = False
        sh["gripper_target_mm"].value = float(max(5.0, min(100.0, mm)))
        t_end = self._now() + 4.0
        while self._now() < t_end and abs(float(sh["gripper_actual_mm"].value) - float(mm)) > 2.0:
            self._sleep(0.05)
        return {"opening_mm": round(float(sh["gripper_actual_mm"].value), 1)}

    def gripper_clamp(self, force_n, settle_s=2.2):
        sh = self.sh
        sh["gripper_force_N"].value = float(force_n)
        sh["clamp_speed_mult_shared"].value = 15.0
        sh["hold_enabled_shared"].value = True
        sh["gripper_force_mode"].value = True
        t_end = self._now() + 3.2
        while self._now() < t_end:
            if bool(sh["hold_active_shared"].value) or (float(sh["contact_left_n_shared"].value) >= 0.05
                                                        and float(sh["contact_right_n_shared"].value) >= 0.05):
                break
            self._sleep(0.05)
        self._sleep(settle_s)
        return {"contact_left": float(sh["contact_left_n_shared"].value) > 0.0,
                "contact_right": float(sh["contact_right_n_shared"].value) > 0.0,
                "left_N": round(float(sh["contact_left_n_shared"].value), 2),
                "right_N": round(float(sh["contact_right_n_shared"].value), 2),
                "hold_active": bool(sh["hold_active_shared"].value),
                "opening_mm": round(float(sh["gripper_actual_mm"].value), 1)}

    def gripper_state(self):
        sh = self.sh
        return {"opening_mm": round(float(sh["gripper_actual_mm"].value), 1),
                "hold_active": bool(sh["hold_active_shared"].value),
                "contact_left": float(sh["contact_left_n_shared"].value) > 0.0,
                "contact_right": float(sh["contact_right_n_shared"].value) > 0.0,
                "force_mode": bool(sh["gripper_force_mode"].value)}

    def safety_trips(self):
        return int(self.sh["safety_trip_count"].value)

    def go_home(self, label):
        try:
            self.c["go_q0"](label)
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)[:300]}


# =============================================================================================== skill contract
STAGES = ("IDLE", "OBSERVED", "PLANNED", "CUP_LIFTED", "RELEASED", "VERIFIED")


class CupSkills:
    """Deterministic skill layer: every tool checks its precondition, acts, measures its postcondition and returns
    ok / reason / physical_verification / recommended_next_actions (same contract style as the egg skills)."""

    def __init__(self, io, knobs=None, skill=None, urdf_check=True):
        self.io = io
        if knobs is None:
            knobs, skill = cup_knobs()
        self.k = dict(CUP_KNOB_DEFAULTS, **(knobs or {}))
        self.skill = skill or {"code": "CS"}
        self.stage = "IDLE"
        self.plan = None
        self.held = None
        self.urdf_check = urdf_check
        self.retries = {"pick": 0, "place": 0}

    # -- helpers
    def _cup(self, name):
        rim, q = self.io.cup_pose(name)
        r, b, ax = cup_axis_points(rim, q)
        return {"rim": r, "base": b, "axis": ax, "quat": q,
                "tilt_deg": round(SC.cup_axis_tilt_deg(q), 2)}

    def _res(self, tool, ok, verification=None, reason=None, nxt=None, warnings=None, **extra):
        v = dict(verification or {})
        if reason:
            v["reason"] = reason
        out = {"ok": bool(ok), "tool": tool, "state_after": {"stage": self.stage, "held_cup": self.held},
               "physical_verification": v, "warnings": list(warnings or []),
               "recommended_next_actions": list(nxt or [])}
        if reason:
            out["reason"] = reason
        out.update(extra)
        self.io.emit(f"[{'OK' if ok else 'FAIL'}] [V3.62 CUP] {tool}: " + (reason or "done") +
                     (f" | {json.dumps(v, ensure_ascii=False)[:300]}" if v else ""))
        return out

    def scene(self):
        cups = {n: self._cup(n) for n in CUPS}
        st = SC.cup_task_state(cups["cup_1"]["rim"], cups["cup_1"]["quat"], cups["cup_2"]["rim"], cups["cup_2"]["quat"],
                               origin="rim")
        st21 = SC.cup_task_state(cups["cup_2"]["rim"], cups["cup_2"]["quat"], cups["cup_1"]["rim"], cups["cup_1"]["quat"],
                                 origin="rim")
        g = self.io.gripper_state()
        tcp, _q = self.io.tcp_pose()
        objs = {}
        for n, c in cups.items():
            objs[n] = {"category": "cup", "base_center_xyz_m": [round(float(v), 4) for v in c["base"]],
                       "rim_center_xyz_m": [round(float(v), 4) for v in c["rim"]], "tilt_deg": c["tilt_deg"],
                       "upright": bool(c["tilt_deg"] <= 10.0), "on_table": bool(abs(c["base"][2]) < 0.006),
                       "held": self.held == n}
        stacked = None
        if st["stacked"]:
            stacked = {"upper": "cup_1", "lower": "cup_2", "fully_nested": st["nested_full"]}
        elif st21["stacked"]:
            stacked = {"upper": "cup_2", "lower": "cup_1", "fully_nested": st21["nested_full"]}
        return {"scene": "stack_cups", "pose_source": "simulator ground truth (TF)", "objects": objs,
                "stacked": stacked, "tcp_xyz_m": [round(v, 4) for v in tcp], "gripper": g,
                "cup_model": {"height_mm": 121, "rim_diameter_mm": 95, "base_diameter_mm": 60,
                              "nested_base_to_base_mm": float(self.k["cup_nested_height_mm"])},
                "skill": {"code": self.skill.get("code"), "name": self.skill.get("short_name"),
                          "knobs": {k: v for k, v in self.k.items()}}}

    # -- tools
    def observe_scene(self, **_):
        sc = self.scene()
        if self.stage == "IDLE" or (self.stage in ("RELEASED", "VERIFIED") and not self.held):
            self.stage = "OBSERVED"
        nxt = ["verify_stack"] if sc["stacked"] else (["place_cup_into"] if self.held else ["plan_cup_stack"])
        return self._res("observe_scene", True, {"stacked": sc["stacked"]}, nxt=nxt, scene=sc)

    def plan_cup_stack(self, source_cup="cup_1", target_cup="cup_2", **_):
        if source_cup not in CUPS or target_cup not in CUPS or source_cup == target_cup:
            return self._res("plan_cup_stack", False, reason="BAD_CUP_NAMES", nxt=["plan_cup_stack"])
        if self.held and self.held != source_cup:
            return self._res("plan_cup_stack", False, reason="ANOTHER_CUP_IS_HELD", nxt=["place_cup_into"])
        s, t = self._cup(source_cup), self._cup(target_cup)
        bad = [n for n, c in ((source_cup, s), (target_cup, t)) if c["tilt_deg"] > 10.0]
        if bad:
            return self._res("plan_cup_stack", False, {"tilted_cups": bad}, reason="CUP_NOT_UPRIGHT",
                             nxt=["stop_and_report"])
        st = SC.cup_task_state(s["rim"], s["quat"], t["rim"], t["quat"], origin="rim")
        if st["stacked"]:
            return self._res("plan_cup_stack", True, st, reason="ALREADY_STACKED", nxt=["verify_stack"])
        plan = plan_cup_stack(self.k, s["rim"], s["quat"], t["rim"], t["quat"], log=self.io.emit)
        if not plan["ok"]:
            return self._res("plan_cup_stack", False, {"detail": plan.get("detail")}, reason=plan["reason"],
                             nxt=["stop_and_report"])
        if plan["min_joint_margin_deg"] < MIN_MARGIN_RELAXED_DEG:
            return self._res("plan_cup_stack", False, {"min_joint_margin_deg": plan["min_joint_margin_deg"],
                                                       "binding": plan["grasp_binding"] if plan["grasp_margin_deg"] <=
                                                       plan["place_margin_deg"] else plan["place_binding"]},
                             reason="JOINT_MARGIN_TOO_SMALL", nxt=["stop_and_report"])
        self.plan = dict(plan, source=source_cup, target=target_cup)
        self.stage = "PLANNED"
        warn = [] if plan["margin_ok"] else [f"min joint margin {plan['min_joint_margin_deg']} deg < 8 deg (V3.41 rule): "
                                             "the motion may hit a joint limit"]
        pub = {k: v for k, v in plan.items() if not k.startswith("_")}
        return self._res("plan_cup_stack", True, {"min_joint_margin_deg": plan["min_joint_margin_deg"]},
                         nxt=["pick_cup"], warnings=warn, plan=pub)

    def pick_cup(self, source_cup="cup_1", **_):
        tool = "pick_cup"
        if self.held:
            return self._res(tool, False, reason="ALREADY_HOLDING", nxt=["place_cup_into"])
        if not self.plan or self.plan["source"] != source_cup or self.stage != "PLANNED":
            return self._res(tool, False, reason="PLAN_REQUIRED", nxt=["plan_cup_stack"])
        W = self.plan["_wps"]
        s0 = self._cup(source_cup)
        o = self.io.gripper_open(60.0)
        v = {"opening_before_mm": o["opening_mm"]}
        for name, fn, spd in (("pregrasp", self.io.move_pose, 0.12), ("grasp", self.io.move_line, 0.03)):
            p, R = W[name]
            r = fn(f"V3.62 {name} {source_cup}", p, R_to_quat_xyzw(R), spd)
            v[f"{name}_tcp_error_mm"] = r.get("tcp_error_mm")
            if not r.get("ok"):
                self._retreat_up()
                return self._res(tool, False, dict(v, error=r.get("error")), reason=f"{name.upper()}_MOTION_FAILED",
                                 nxt=["observe_scene", "plan_cup_stack"])
        s1 = self._cup(source_cup)
        moved = float(np.linalg.norm(s1["base"][:2] - s0["base"][:2])) * 1000
        v["cup_moved_by_descent_mm"] = round(moved, 1)
        if moved > 8.0 or s1["tilt_deg"] > 6.0:
            self._retreat_up()
            return self._res(tool, False, v, reason="CUP_PUSHED_DURING_DESCENT", nxt=["observe_scene", "plan_cup_stack"])
        c = self.io.gripper_clamp(float(self.k["cup_clamp_force_n"]))
        v.update({"clamp": c})
        if not (c["contact_left"] and c["contact_right"]) and not c["hold_active"]:
            self.io.gripper_open(60.0)
            self._retreat_up()
            return self._res(tool, False, v, reason="NO_TWO_FINGER_CONTACT", nxt=["observe_scene", "plan_cup_stack"])
        p, R = W["lift"]
        tcp0, _ = self.io.tcp_pose()
        r = self.io.move_line(f"V3.62 lift {source_cup}", p, R_to_quat_xyzw(R), 0.05)
        tcp1, _ = self.io.tcp_pose()
        s2 = self._cup(source_cup)
        rise_tcp = tcp1[2] - tcp0[2]
        rise_cup = float(s2["base"][2] - s1["base"][2])
        v.update({"lift_tcp_rise_mm": round(rise_tcp * 1000, 1), "cup_rise_mm": round(rise_cup * 1000, 1),
                  "cup_tilt_after_lift_deg": s2["tilt_deg"], "hold_active": self.io.gripper_state()["hold_active"]})
        if rise_cup < 0.7 * max(rise_tcp, 0.01):
            self.io.gripper_open(60.0)
            self._retreat_up()
            return self._res(tool, False, v, reason="CUP_DID_NOT_FOLLOW_LIFT", nxt=["observe_scene", "plan_cup_stack"])
        self.held = source_cup
        self.stage = "CUP_LIFTED"
        warn = []
        if s2["tilt_deg"] > float(self.k["cup_release_tilt_tol_deg"]):
            warn.append(f"held cup tilted {s2['tilt_deg']} deg (> {self.k['cup_release_tilt_tol_deg']} deg tolerance)")
        if not r.get("ok"):
            warn.append(f"lift ended {r.get('tcp_error_mm')} mm from its target")
        return self._res(tool, True, v, nxt=["place_cup_into"], warnings=warn)

    def _retreat_up(self, dz=0.08):
        try:
            p, q = self.io.tcp_pose()
            self.io.move_line("V3.62 retreat", [p[0], p[1], p[2] + dz], q, 0.05)
        except Exception:
            pass

    def _centre_error(self, src, dst):
        s, t = self._cup(src), self._cup(dst)
        d = (t["base"][:2] - s["base"][:2])
        return d, s, t

    def place_cup_into(self, target_cup="cup_2", **_):
        tool = "place_cup_into"
        if not self.held or self.stage != "CUP_LIFTED":
            return self._res(tool, False, reason="NO_CUP_HELD", nxt=["observe_scene"])
        if not self.plan or self.plan["target"] != target_cup:
            return self._res(tool, False, reason="TARGET_DIFFERS_FROM_PLAN", nxt=["plan_cup_stack"])
        src = self.held
        W = self.plan["_wps"]
        tol = float(self.k["cup_release_xy_tol_mm"]) / 1000.0
        v = {}
        p, R = W["above_target"]
        q = R_to_quat_xyzw(R)
        # level straight carry first (the cups are not in the MoveIt world, a free joint-space path could dip and
        # drag the held cup over the target rim); BackendIO falls back to the pose planner if it is refused
        r = self.io.move_line(f"V3.62 carry {src} over {target_cup}", p, q, CARRY_SPEED, slow=CARRY_SLOW_FACTOR)
        if not r.get("ok"):
            return self._res(tool, False, dict(v, error=r.get("error"), tcp_error_mm=r.get("tcp_error_mm")),
                             reason="CARRY_MOTION_FAILED", nxt=["observe_scene"])
        # closed-loop centring on the measured cup poses (ground truth)
        for it in range(3):
            d, s, t = self._centre_error(src, target_cup)
            if float(np.linalg.norm(d)) <= 0.0015:
                break
            tcp, q = self.io.tcp_pose()
            self.io.move_line(f"V3.62 centre #{it + 1}", [tcp[0] + d[0], tcp[1] + d[1], tcp[2]], q, PLACE_SPEED)
        d, s, t = self._centre_error(src, target_cup)
        v["centre_error_high_mm"] = round(float(np.linalg.norm(d)) * 1000, 1)
        # lower until the base is at release height
        _, base_rel = stack_heights(self.k, s["base"][2], t["base"][2])
        tcp, q = self.io.tcp_pose()
        dz = float(s["base"][2] - base_rel)
        r = self.io.move_line(f"V3.62 lower {src} into {target_cup}", [tcp[0] + d[0], tcp[1] + d[1], tcp[2] - dz], q, PLACE_SPEED)
        d, s, t = self._centre_error(src, target_cup)
        if float(np.linalg.norm(d)) > tol:
            tcp, q = self.io.tcp_pose()
            self.io.move_line("V3.62 centre at release height", [tcp[0] + d[0], tcp[1] + d[1], tcp[2]], q, PLACE_SPEED)
            d, s, t = self._centre_error(src, target_cup)
        v.update({"centre_error_mm": round(float(np.linalg.norm(d)) * 1000, 1),
                  "held_cup_tilt_deg": s["tilt_deg"], "target_cup_tilt_deg": t["tilt_deg"],
                  "base_above_target_base_mm": round(float(s["base"][2] - t["base"][2]) * 1000, 1)})
        if float(np.linalg.norm(d)) > tol:
            return self._res(tool, False, v, reason="NOT_CENTRED_OVER_TARGET", nxt=["place_cup_into", "stop_and_report"])
        if s["tilt_deg"] > float(self.k["cup_release_tilt_tol_deg"]) * 2.0:
            return self._res(tool, False, v, reason="HELD_CUP_TOO_TILTED", nxt=["stop_and_report"])
        warn = []
        if s["tilt_deg"] > float(self.k["cup_release_tilt_tol_deg"]):
            warn.append(f"held cup tilted {s['tilt_deg']} deg (> {self.k['cup_release_tilt_tol_deg']} deg); released anyway "
                        "(the V3.61 release sweep succeeded up to 6 deg)")
        self.io.gripper_open(60.0)
        self.io.sleep(1.5)
        self.held = None
        self.stage = "RELEASED"
        p2, R2 = W["retreat"]
        tcp, q = self.io.tcp_pose()
        self.io.move_line("V3.62 retreat after release", [tcp[0], tcp[1], tcp[2] + 0.08], q, 0.05)
        st = self._state(src, target_cup)
        v["after_release"] = st
        return self._res(tool, True, v, nxt=["verify_stack"], warnings=warn)

    def _state(self, upper, lower):
        a, b = self._cup(upper), self._cup(lower)
        return SC.cup_task_state(a["rim"], a["quat"], b["rim"], b["quat"], origin="rim")

    def verify_stack(self, upper_cup="cup_1", lower_cup="cup_2", **_):
        if upper_cup not in CUPS or lower_cup not in CUPS or upper_cup == lower_cup:
            return self._res("verify_stack", False, reason="BAD_CUP_NAMES")
        if self.held:
            return self._res("verify_stack", False, reason="CUP_STILL_HELD", nxt=["place_cup_into"])
        st = self._state(upper_cup, lower_cup)
        ok = bool(st["stacked"])
        if ok:
            self.stage = "VERIFIED"
        return self._res("verify_stack", ok, st, reason=None if ok else "NOT_STACKED",
                         nxt=["finish_and_report"] if ok else ["observe_scene", "plan_cup_stack"],
                         upper_cup=upper_cup, lower_cup=lower_cup)

    def return_to_safe_pose(self, **_):
        if self.held:
            return self._res("return_to_safe_pose", False, reason="CUP_HELD_PLACE_FIRST", nxt=["place_cup_into"])
        self.io.gripper_open(60.0)
        r = self.io.go_home("V3.62 cup task")
        if self.stage not in ("VERIFIED",):
            self.stage = "OBSERVED"
        return self._res("return_to_safe_pose", bool(r.get("ok", True)), {"detail": r.get("error")})

    def run_tool(self, name, args):
        fn = {"observe_scene": self.observe_scene, "plan_cup_stack": self.plan_cup_stack, "pick_cup": self.pick_cup,
              "place_cup_into": self.place_cup_into, "verify_stack": self.verify_stack,
              "return_to_safe_pose": self.return_to_safe_pose}.get(name)
        if fn is None:
            return {"ok": False, "tool": name, "reason": "UNKNOWN_TOOL", "recommended_next_actions": ["observe_scene"]}
        try:
            return fn(**(args or {}))
        except Exception as exc:
            return {"ok": False, "tool": name, "reason": "SKILL_EXCEPTION", "error": str(exc)[:400],
                    "recommended_next_actions": ["observe_scene", "stop_and_report"]}

    def scripted_episode(self, source="cup_1", target="cup_2"):
        """the nominal 5-tool sequence without an LLM (tests / twin tuning) -> list of results"""
        out = [self.run_tool("observe_scene", {})]
        for name, args in (("plan_cup_stack", {"source_cup": source, "target_cup": target}),
                           ("pick_cup", {"source_cup": source}), ("place_cup_into", {"target_cup": target}),
                           ("verify_stack", {"upper_cup": source, "lower_cup": target})):
            r = self.run_tool(name, args)
            out.append(r)
            if not r.get("ok") and not (name == "plan_cup_stack" and r.get("reason") == "ALREADY_STACKED"):
                break
        return out


# =============================================================================================== LLM contract
_CUPN = ["cup_1", "cup_2"]
CUP_TOOLS = [
    {"type": "function", "name": "observe_scene",
     "description": "Read the stack_cups scene: both cups (base/rim centre, tilt, upright, held), whether they are already "
                    "stacked, gripper opening / hold, TCP. Observation only. ALWAYS call first.",
     "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "plan_cup_stack",
     "description": "Plan putting source_cup INTO target_cup with the rim-pinch skill: chooses reachable grasp / place rim "
                    "angles (strict 6-DoF IK, joint margins). No motion.",
     "parameters": {"type": "object", "properties": {"source_cup": {"type": "string", "enum": _CUPN},
                                                     "target_cup": {"type": "string", "enum": _CUPN}},
                    "required": ["source_cup", "target_cup"], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "pick_cup",
     "description": "Execute the planned grasp: open, pregrasp above the rim, straight descent, pinch the rim (one finger "
                    "inside, one outside), lift until the cup clears the other cup. Verifies two-finger contact and that "
                    "the cup follows the lift.",
     "parameters": {"type": "object", "properties": {"source_cup": {"type": "string", "enum": _CUPN}},
                    "required": ["source_cup"], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "place_cup_into",
     "description": "Carry the held cup over target_cup, centre it on the measured cup axes, lower it to the release "
                    "height, open (the cup slides down into the target), retreat up. Does not itself prove success.",
     "parameters": {"type": "object", "properties": {"target_cup": {"type": "string", "enum": _CUPN}},
                    "required": ["target_cup"], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "verify_stack",
     "description": "Read-only final check that upper_cup is inside lower_cup (upright, centred, base inside).",
     "parameters": {"type": "object", "properties": {"upper_cup": {"type": "string", "enum": _CUPN},
                                                     "lower_cup": {"type": "string", "enum": _CUPN}},
                    "required": ["upper_cup", "lower_cup"], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "return_to_safe_pose",
     "description": "Open the empty gripper and return the arm to the safe neutral posture. Refused while a cup is held.",
     "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}, "strict": True},
]

CUP_SYSTEM_INSTRUCTIONS = """You are NERO V3.62, the high-level task planner for a MuJoCo simulated 7-DOF NERO robot arm with a
two-finger gripper. Scene stack_cups (RoboDojo): two identical red/white 16 oz party cups, cup_1 and cup_2, stand upright on
the table. "Stack the cups" means: put one cup INTO the other (default: cup_1 into cup_2 unless the user names another
order). You control the robot ONLY through the function tools. Never invent coordinates or joint angles.

Skill contract (skill CS "Cup Stack", rim pinch):
  observe_scene -> plan_cup_stack(source, target) -> pick_cup(source) -> place_cup_into(target) -> verify_stack(source, target)
  Stages: IDLE -> OBSERVED -> PLANNED -> CUP_LIFTED -> RELEASED -> VERIFIED.
Rules:
1. Call observe_scene first. If scene.stacked is already set for the requested order, call verify_stack and finish.
2. One physical tool at a time. After every tool read ok, reason, physical_verification, warnings and
   recommended_next_actions before continuing.
3. If pick_cup fails (CUP_PUSHED_DURING_DESCENT, NO_TWO_FINGER_CONTACT, CUP_DID_NOT_FOLLOW_LIFT, *_MOTION_FAILED), the
   gripper is empty again: call observe_scene, then plan_cup_stack once more and retry pick_cup ONCE. Never loop.
4. If place_cup_into fails with NOT_CENTRED_OVER_TARGET the cup is still held: call place_cup_into once more. Any other
   place failure or a second failure: stop and report.
5. Success only when verify_stack returns ok=true. Report fully_nested (nested_full) as well.
6. GRASP_UNREACHABLE / PLACE_UNREACHABLE / JOINT_MARGIN_TOO_SMALL / CUP_NOT_UPRIGHT: stop and report; do not try other
   motions.
7. Keep the final message short and factual (which cup into which, verified or not, the reason of any failure).
"""


def _score(history, elapsed_s, rounds, skill, source, target):
    """s = 1 when verify_stack confirmed a stack in this task (the pair it verified is reported; the task text only gives
    the default pair for the message - "put cup_2 into cup_1" in any language is judged by the verified result)"""
    good = [h["result"] for h in history if h["tool"] == "verify_stack" and isinstance(h["result"], dict)
            and h["result"].get("ok")]
    ok = bool(good)
    if good:
        source, target = good[-1].get("upper_cup", source), good[-1].get("lower_cup", target)
    s = 1.0 if ok else 0.0
    t_min = max(0.0, float(elapsed_s)) / 60.0
    z = int(rounds)
    y = s / (t_min * z) * 100.0 if t_min > 0 and z > 0 else 0.0
    line = (f"[V3.62 SKILL SCORE] skill={skill.get('code')} {skill.get('short_name')} | scene=stack_cups | "
            f"s={s:.2f} ({'stacked' if ok else 'not stacked'}: {source} into {target}) | t={t_min:.2f} min | "
            f"z={z} LLM rounds | Y = s/(t*z)*100 = {y:.3f}")
    return {"skill_code": skill.get("code"), "skill_name": skill.get("short_name"), "skill_id": skill.get("skill_id"),
            "s": s, "done": int(ok), "total": 1, "basis": "stack verified", "t_min": round(t_min, 4), "z": z,
            "Y": round(y, 4), "status": "SUCCESS" if ok else "FAILED", "line": line}


def _task_pair(task):
    t = str(task).lower().replace(" ", "")
    for a, b in (("cup_2", "cup_1"), ("cup2", "cup1")):
        if (a + "into" + b) in t or (a + "in" + b) in t:
            return "cup_2", "cup_1"
    return "cup_1", "cup_2"


def run_cup_task(agent, skills, task, model, api_key="", reasoning_effort="low", max_tool_steps=30, emit=None,
                 dry_run=False):
    """bounded tool-calling episode for stack_cups, reusing NeroLLMAgent's API transport (providers, retries,
    usage accounting).  -> final text"""
    emit = emit or agent.emit
    with agent._lock:
        if agent._running:
            raise RuntimeError("LLM agent is already running")
        agent._running = True
    agent._stop_event.clear()
    agent._last_response_id = None
    t0 = time.perf_counter()
    rounds, steps, history, final = 0, 0, [], ""
    status = "FAILED"
    try:
        ok, key_status = agent.api_key_status(api_key, model)
        if not ok:
            raise RuntimeError(key_status)
        emit(f"[V3.62 CUP TASK START] model={model} | effort={reasoning_effort} | max_tool_steps={max_tool_steps} | "
             f"skill={skills.skill.get('code')} | knobs={json.dumps(skills.k, ensure_ascii=False)}")
        emit(f"[LLM USER TASK] {task}")
        payload = {"model": str(model), "instructions": CUP_SYSTEM_INSTRUCTIONS, "input": str(task), "tools": CUP_TOOLS,
                   "tool_choice": "auto", "parallel_tool_calls": False, "reasoning": {"effort": str(reasoning_effort)},
                   "max_output_tokens": 1200, "store": True}
        while True:
            if agent._stop_event.is_set():
                raise RuntimeError("LLM task stopped by user")
            rounds += 1
            emit(f"[LLM REQUEST #{rounds}]")
            t_req = time.perf_counter()
            resp = agent._post_responses(payload, api_key)
            agent._log_usage(resp, model)
            agent._last_response_id = resp.get("id")
            calls = agent._function_calls(resp)
            text = agent._extract_text(resp)
            emit(f"[LLM RESPONSE #{rounds}] elapsed={time.perf_counter() - t_req:.2f}s | tool_calls={len(calls)}")
            if text:
                emit(f"[LLM TEXT] {text}")
            if not calls:
                final = text
                break
            outs = []
            for call in calls:
                if agent._stop_event.is_set():
                    raise RuntimeError("LLM task stopped by user")
                steps += 1
                name = str(call.get("name", ""))
                if steps > max_tool_steps:
                    raise RuntimeError(f"tool step budget {max_tool_steps} used up")
                try:
                    args = json.loads(call.get("arguments") or "{}")
                except Exception:
                    args = {}
                emit(f"[LLM CALL #{steps}] {name}(" + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")")
                t_tool = time.perf_counter()
                if dry_run and name not in ("observe_scene", "plan_cup_stack", "verify_stack"):
                    res = {"ok": True, "dry_run": True, "tool": name, "arguments": args}
                    emit(f"[LLM DRY RUN] skipped physical tool {name}")
                else:
                    res = skills.run_tool(name, args)
                emit(f"[LLM TOOL RESULT #{steps}] {name} | elapsed={time.perf_counter() - t_tool:.2f}s | "
                     + json.dumps({k: v for k, v in res.items() if k != "scene"}, ensure_ascii=False)[:1500])
                history.append({"tool": name, "result": res})
                outs.append({"type": "function_call_output", "call_id": str(call.get("call_id", "")),
                             "output": json.dumps(res, ensure_ascii=False, default=str)})
            payload = {"model": str(model), "instructions": CUP_SYSTEM_INSTRUCTIONS,
                       "previous_response_id": agent._last_response_id, "input": outs, "tools": CUP_TOOLS,
                       "tool_choice": "auto", "parallel_tool_calls": False,
                       "reasoning": {"effort": str(reasoning_effort)}, "max_output_tokens": 1200, "store": True}
        emit(f"[LLM FINAL] {final}")
        return final
    except Exception as exc:
        emit(f"[LLM TASK FAILED] {exc}")
        raise
    finally:
        src, dst = _task_pair(task)
        sc = _score(history, time.perf_counter() - t0, rounds, skills.skill, src, dst)
        status = sc["status"]
        emit(f"[TASK SUMMARY {'GREEN' if status == 'SUCCESS' else 'RED'}] STATUS: {status} | TASK: {task} | "
             f"tools {steps} | LLM rounds {rounds} | {time.perf_counter() - t0:.1f} s")
        emit(sc["line"])
        try:
            import NERO_V2_7_1_Agent as AG
            AG._v360_append_score(sc, task, model)
        except Exception:
            pass
        with agent._lock:
            agent._running = False


# =============================================================================================== backend entry (A2 block)
def gui_run_task(ctx):
    """called from NERO_V2_7_1_Unified_Backend.llm_run_task_cmd (its `# V3.43 A2` block) when NERO_SCENE=stack_cups"""
    log = ctx["log"]
    mb = ctx.get("messagebox")
    cfg = ctx["cfg_fn"]()
    if not cfg.get("task"):
        log("[LLM UI] Task is empty")
        return
    if ctx.get("pose_source_mode", lambda: "truth")() == "camera":
        log("[FAIL] [V3.62 CUP] stack_cups has no camera perception yet: select 'JSON ground truth' as the pose source")
        if mb:
            mb.showerror("NERO V3.62", "叠杯子场景目前只支持 JSON ground truth 位姿源（还没有杯子的相机感知）。")
        return
    if ctx["agent"] is None:
        log("[FAIL] [V3.62 CUP] the LLM agent module is not loaded")
        return
    if not ctx["safety_gate"]():
        log("[FAIL] [V3.62 CUP] RUN TASK blocked: stale physical safety latch was not recovered")
        return
    knobs, skill = cup_knobs()
    io = BackendIO(ctx)
    skills = CupSkills(io, knobs, skill)
    log(f"[V3.62 CUP TASK] scene=stack_cups | skill {skill['code']} {skill['short_name']} ({skill['source']}) | "
        f"model={cfg['model']} | task={cfg['task']!r}")

    def _run():
        if not bool(cfg.get("dry_run", False)):
            ctx["start_check"](False)
        return run_cup_task(ctx["agent"], skills, cfg["task"], cfg["model"], cfg["api_key"], cfg["reasoning_effort"],
                            max(10, min(60, int(cfg["max_tool_steps"]))), emit=io.emit,
                            dry_run=bool(cfg.get("dry_run", False)))

    ctx["background"]("RUN TASK (stack_cups)", _run, lock_motion=True)


def twin_episode(knobs=None, source="cup_1", target="cup_2", log=print, snapshot_dir=None, cup_xy=None):
    """offline: the scripted CS episode in the headless twin -> (results, final cup state, io)"""
    k = dict(CUP_KNOB_DEFAULTS, **(knobs or {}))
    io = TwinIO(log=log, cup_xy=cup_xy)
    sk = CupSkills(io, k, {"code": "CS", "short_name": "Cup Stack", "skill_id": "cup_stack_rim_pinch"})
    res = sk.scripted_episode(source, target)
    st = sk._state(source, target)
    if snapshot_dir:
        try:
            io.snapshot(os.path.join(snapshot_dir, "v362_twin_final.png"))
        except Exception as exc:
            log(f"[INFO] [V3.62 CUP] no snapshot ({exc})")
    return res, st, io


def _study_job(job):
    import random
    k, src, dst, seed, solref = job
    if solref:
        os.environ["NERO_V362_CUP_SOLREF"] = solref
    else:
        os.environ.pop("NERO_V362_CUP_SOLREF", None)          # pool workers are reused: never inherit a previous job's
    rnd = random.Random(seed)
    xy = ({n: (x + rnd.uniform(-0.01, 0.01), y + rnd.uniform(-0.01, 0.01)) for n, (x, y) in SC.CUP_XY.items()}
          if seed else None)
    res, st, io = twin_episode(k, src, dst, log=lambda s_: None, cup_xy=xy)
    pick = next((r for r in res if r["tool"] == "pick_cup"), {})
    return {"knobs": k, "source": src, "target": dst, "position_noise_seed": seed, "cup_solref": solref or "scene default",
            "stacked": st["stacked"], "nested_full": st["nested_full"], "base_dz_mm": st["base_dz_mm"],
            "tilt_after_lift_deg": (pick.get("physical_verification") or {}).get("cup_tilt_after_lift_deg"),
            "failed_at": next((f"{r['tool']}:{r.get('reason')}" for r in res if not r["ok"]), None),
            "sim_time_s": round(io.t, 1)}


def twin_study(processes=4, log=print, out_path=None):
    """V3.62 twin study (about 5 min on 4 cores): defaults both directions, +-10 mm cup position noise over pinch
    depth x clamp force, two cup contact stiffnesses, and the V3.61-style controls (25 mm / vertical approach)."""
    from multiprocessing import Pool
    jobs = [({}, "cup_1", "cup_2", 0, None), ({}, "cup_2", "cup_1", 0, None)]
    jobs += [({"cup_rim_pinch_depth_mm": d, "cup_clamp_force_n": f}, "cup_1", "cup_2", sd, None)
             for d in (30.0, 35.0, 40.0) for f in (3.0, 5.0) for sd in (1, 2)]
    jobs += [({}, "cup_1", "cup_2", 3, sr) for sr in ("-5000 -30", "-50000 -100")]
    jobs += [({"cup_rim_pinch_depth_mm": 25.0}, "cup_1", "cup_2", 0, None), ({"cup_wall_tilt_follow": False}, "cup_1", "cup_2", 0, None),
             ({"cup_rim_pinch_depth_mm": 25.0, "cup_wall_tilt_follow": False}, "cup_1", "cup_2", 0, None)]
    with Pool(processes) as pool:
        rows = []
        for r in pool.imap(_study_job, jobs):
            rows.append(r)
            log(f"[V3.62 twin study] {r['source']}->{r['target']} {r['knobs']} seed {r['position_noise_seed']} "
                f"{r['cup_solref']}: stacked={r['stacked']} nested_full={r['nested_full']} lift tilt {r['tilt_after_lift_deg']}"
                + (f" FAILED {r['failed_at']}" if r["failed_at"] else ""))
    knobs, _sk = cup_knobs()
    study = {"version": VERSION, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
             "what": ("headless MuJoCo twin: official AgileX finger pads on a mocap link7 + backend gripper logic "
                      "(force clamp, penetration guard, auto CONNECT hold) + the two scene cups; scripted CS episode "
                      "observe -> plan -> pick -> place -> verify. The arm is not simulated (reachability: strict 6-DoF "
                      "URDF IK in plan_cup_stack). Not measured on hardware."),
             "layout": {k_: list(v) for k_, v in SC.CUP_XY.items()}, "cup_origin": SC.CUP_ORIGIN,
             "defaults": knobs, "rows": rows,
             "summary": {"default_like_runs": sum(1 for r in rows if not ({"cup_wall_tilt_follow"} & set(r["knobs"]))
                                                  and r["knobs"].get("cup_rim_pinch_depth_mm", 35.0) >= 30.0),
                         "default_like_success": sum(1 for r in rows if r["nested_full"] and not ({"cup_wall_tilt_follow"} & set(r["knobs"]))
                                                     and r["knobs"].get("cup_rim_pinch_depth_mm", 35.0) >= 30.0)}}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(study, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return study


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--study":
        twin_study(out_path=sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        r, st, _io = twin_episode(log=print)
        print(json.dumps(st))
