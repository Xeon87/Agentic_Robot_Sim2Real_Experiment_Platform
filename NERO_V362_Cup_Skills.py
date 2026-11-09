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

VERSION = "V3.64"
MARKER = "V3.62 STACK CUPS AGENT SKILLS"
MARKER_V363 = "V3.63 JOINT-SPACE SAFE MOTION + FIXED D435i CUP PERCEPTION"
MARKER_V364 = "V3.64 SO-101 GRIPPER ON NERO"
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
# V3.63 JOINT-SPACE SAFE MOTION (user run V3.62: a 4-point MoveIt Cartesian centring move drove J7 past the MuJoCo
# limit -> 'invalid arm state' emergency restore; MoveIt / cuRobo may use wider joint limits than the MuJoCo model
# and pick another IK branch of the 7-DoF arm than the one the plan checked)
PLACE_J7_PREF_DEG = 15.0       # place poses: prefer J7 >= 15 deg from its limit (J7 turns the carried cup)
ALIGN_MAX_DEG = 90.0           # V3.67 ALIGN-ROTATE-DESCEND: largest in-place wrist rotation above the cup
ALIGN_MIN_DEG = 5.0            # smaller differences: arrive with the grasp angle directly
ALIGN_MAX_TCP_DEV_M = 0.05     # the joint-space rotation may move the TCP at most this much
TIP_BEYOND_GRASP_M = 0.010     # SO-101 pad: fingertip 10 mm beyond the grasp-frame origin along the approach
PRE_CONTACT_MOVE_MM = 6.0      # V3.67 TWO-STAGE DESCENT: cup estimate moved more than this -> abort before contact
LENIENT_ABORT_MM = 30.0        # V3.68 LENIENT REAL PICK: only a cup clearly pushed away (fresh view) stops the pick
LENIENT_TILT_DEG = 25.0        # ... or clearly knocked over (tilt change, fresh view)
_LAST_ANGLES = {}              # V3.68 FAST PLAN: profile name -> (grasp rel deg, place rel deg) of the last good plan
PATH_MARGIN_DEG = 4.0          # every point of an executed path keeps all joints >= 4 deg inside the MuJoCo limits
PATH_JUMP_DEG = 5.0            # max joint change between two consecutive path points (more = IK branch flip)
SMALL_CORRECTION_M = 0.003     # centring errors below this are not corrected (no tiny moves)
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


def quat_from_tilt_wxyz(tilt_deg, azimuth_rad=0.0):
    t = math.radians(float(tilt_deg)) / 2.0
    return [math.cos(t), -math.sin(azimuth_rad) * math.sin(t), math.cos(azimuth_rad) * math.sin(t), 0.0]


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


class GripperProfile:
    """V3.64 GRIPPER PROFILES: the skill plans the GRASP frame (origin on the wall mid-surface where the jaws pinch,
    z = approach, y = closing direction); a profile gives the fixed transform official TCP (URDF tcp_link: MoveIt, IK,
    TF) -> grasp frame and the gripper openings (backend units, opening = 2000 * gripper_joint1)."""

    def __init__(self, name, T_tcp_grasp, open_mm, label, approach_clearance_m=0.0, approach_along_wall=False,
                 flips=(False, True), release_gap_max_mm=None, lift_settle_s=0.0,
                 release_tilt_max_deg=0.0, align_rotate_m=0.0, descent_stop_above_rim_m=None,
                 wide_search_below_deg=None, lenient_pick=False, wide_step_deg=15, plan_time_budget_s=None,
                 remember_angles=False, clear_view_before_verify=False, camera_decides_grip=False,
                 recheck_held_before_place=False, stack_by_cluster=False, perception_retries=0,
                 recover_after_motion_failure=False, nested_dz_real_mm=None):
        self.name, self.T_tcp_grasp, self.open_mm, self.label = name, np.asarray(T_tcp_grasp, float), float(open_mm), label
        # V3.64: a gripper with a FIXED jaw (SO-101) descends with this clearance between the fixed pad and the wall
        # (grasp frame moved along -y); clamping then slides the 10 g cup onto the fixed jaw
        self.approach_clearance_m = float(approach_clearance_m)
        # V3.64: ... and descends ALONG the conical wall (grasp z axis) instead of vertically, so the pad that stays
        # outside the wall keeps that clearance past the rim bead (vertical: the cone narrows below the rim)
        self.approach_along_wall = bool(approach_along_wall)
        # V3.64 JAW SIDES: allowed pinch flips.  flip=True puts the grasp -y side (SO-101: the FIXED jaw) OUTSIDE the
        # wall and the moving jaw inside.  A falling cup's conical wall moves OUTWARD at a fixed height (8.2 deg:
        # +12 mm over the 85 mm release drop): with the moving jaw outside the cup caught it before it had opened, and
        # with the fixed jaw inside the cup tilted 18 deg in the grip (V3.64 twin).  Fixed jaw outside + the release
        # sidestep (TCP moved outward, fixed pad off the wall) is the combination that nests.
        self.flips = tuple(flips)
        # V3.64 LOW RELEASE: cap of the cup_release_gap_mm knob.  A one-side pinch holds less torque: the SO-101-held
        # cup leans 1-3 deg, and over an 85 mm drop its base caught the target's inner wall and it hung on the jaw
        self.release_gap_max_mm = release_gap_max_mm
        # V3.64 LIFT SETTLE: during the lift the backend's pad-penetration guard opens the jaws and the cup swings on
        # the CONNECT hold (official gripper too: up to 31 deg, upright again by the end of the lift).  The SO-101 jaw
        # opens less and the cup needs ~2 s more to hang upright and be re-clamped -> wait before carrying it.
        self.lift_settle_s = float(lift_settle_s)
        # V3.64 LOW RELEASE: a cup released only 35 mm above its nested height still nests when it leans a little more
        # (one-side pinch: pad friction keeps a carry lean of 6-9 deg); 0 = the knob rule (2 x cup_release_tilt_tol_deg)
        self.release_tilt_max_deg = float(release_tilt_max_deg)
        # V3.67 ALIGN-ROTATE-DESCEND (real arm; 0 / None = simulation behaviour, unchanged): arrive this high above the
        # pregrasp with the wrist yaw the arm already has, rotate in place to the grasp angle (jaw line along the cup
        # radius), then go straight down to the pregrasp
        self.align_rotate_m = float(align_rotate_m or 0.0)
        # V3.67 TWO-STAGE DESCENT: stop when the fingertips are this far above the rim, re-check the cup with the fixed
        # camera, then continue to the pinch
        self.descent_stop_above_rim_m = None if descent_stop_above_rim_m is None else float(descent_stop_above_rim_m)
        # V3.67 PLAN MARGIN: also run the wide rim-angle search when the best chain is below this margin
        self.wide_search_below_deg = None if wide_search_below_deg is None else float(wide_search_below_deg)
        # V3.68 LENIENT REAL PICK: no abort before the clamp on camera estimates taken while the gripper hides the cup;
        # clamp + lift always, then judge (clamp stop above the touch point AND the cup gone from the table)
        self.lenient_pick = bool(lenient_pick)
        # V3.68 FAST PLAN: wide rim-angle search step, time budget of the wide search, start from the last good angles
        self.wide_step_deg = int(wide_step_deg)
        self.plan_time_budget_s = None if plan_time_budget_s is None else float(plan_time_budget_s)
        self.remember_angles = bool(remember_angles)
        # V3.69 CLEAR VIEW VERIFY: move the arm out of the fixed camera's view of the cups before verifying
        self.clear_view_before_verify = bool(clear_view_before_verify)
        # V3.69 CAMERA DECIDES GRIP: after the lift the camera (cup gone from the table) decides; the servo angle only
        # warns (a 0.3-0.5 mm cup wall: clamped 2.62 mm vs empty 2.39 mm in the first real stack)
        self.camera_decides_grip = bool(camera_decides_grip)
        # V3.71 HELD CUP RECHECK (real): the cup turns / slides a little in the jaws after the clamp -> above the target
        # the fixed camera measures the HELD cup, the grasp offset is corrected, centred, then lowered
        self.recheck_held_before_place = bool(recheck_held_before_place)
        # V3.71 STACK VERIFY BY CLUSTER (real): one cup cluster on the target's place, taller than the target alone
        # (measured before) = stacked; the model's 25.3 mm nested height is not the real cups' (V3.70: judged NOT_STACKED)
        self.stack_by_cluster = bool(stack_by_cluster)
        self.nested_dz_real_mm = None if nested_dz_real_mm in (None, "") else float(nested_dz_real_mm)
        # V3.71 PERCEPTION RETRY (real): a cup YOLOE missed once (confidence near the threshold) -> look again
        self.perception_retries = int(perception_retries)
        # V3.71 RECOVER (real): after a failed motion the arm stops and goes straight up to a safe height
        self.recover_after_motion_failure = bool(recover_after_motion_failure)

        Rg = self.T_tcp_grasp[:3, :3]
        if abs(np.linalg.det(Rg) - 1.0) > 1e-6 or np.abs(Rg.T @ Rg - np.eye(3)).max() > 1e-6:
            raise ValueError(f"gripper profile {name}: TCP -> grasp transform is not a proper rotation")
        self.T_grasp_tcp = np.linalg.inv(self.T_tcp_grasp)

    def pre_offset(self, R, pre):
        """pregrasp offset from the grasp TCP (same height `pre` either way)"""
        if not self.approach_along_wall:
            return np.array([0.0, 0.0, pre])
        a = -np.asarray(R, float)[:, 2]
        return a * (pre / max(a[2], 0.5))

    def tcp_from_grasp(self, p, R):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, p
        T = T @ self.T_grasp_tcp
        return T[:3, 3].copy(), T[:3, :3].copy()


def _t(x, y, z):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


OFFICIAL_PROFILE = GripperProfile("official", _t(0.0, CLAMP_CENTRE_Y_M, 0.0), 60.0,
                                  "AgileX official two-finger gripper")


def so101_profile():
    import NERO_V364_SO101_Gripper as G
    return GripperProfile("so101", G.T_tcp_grasp(), G.opening_mm_for_tip_gap(32.0),
                          f"SO-101 gripper on a {G.MOUNT['ring_m'] * 1000:.0f} mm ring, roll {G.MOUNT['roll_deg']:+.0f} deg",
                          approach_clearance_m=float(os.environ.get("NERO_V364_SO101_CLEARANCE_MM", "2.0")) / 1000.0,
                          approach_along_wall=True, flips=(True,),
                          release_gap_max_mm=float(os.environ.get("NERO_V364_SO101_RELEASE_GAP_MM", "35")),
                          lift_settle_s=2.5, release_tilt_max_deg=float(os.environ.get("NERO_V364_SO101_RELEASE_TILT_DEG", "9")))


def profile_for_scene(scene=None):
    scene = scene or SC.active_scene()
    return so101_profile() if scene == "stack_cups_so101" else OFFICIAL_PROFILE


def pinch_tcp(rim_centre, phi_rad, depth_m, flip, tilt_follow=True, profile=None):
    """TCP position + rotation pinching the wall at rim angle phi, `depth` (along the wall) below the rim.
    V3.64: computed for the grasp frame of the gripper profile, returned as the official TCP pose."""
    g = geom()
    u = np.array([math.cos(phi_rad), math.sin(phi_rad), 0.0])
    R = pinch_frame(phi_rad, flip, tilt_follow)
    top = np.asarray(rim_centre, float) + g["r_top"] * u              # wall mid-surface at the rim
    wall_down = np.array([-math.sin(g["beta"]) * u[0], -math.sin(g["beta"]) * u[1], -math.cos(g["beta"])])
    p = top + depth_m * (wall_down if tilt_follow else np.array([0.0, 0.0, -1.0]))
    prof = profile or OFFICIAL_PROFILE
    p = p - prof.approach_clearance_m * R[:, 1]                      # fixed-jaw grippers: clearance to the wall
    return prof.tcp_from_grasp(p, R)                                 # official: pad clamp centre (+2 mm) on the wall


def base_dir(xy):
    """direction from a cup axis to the robot base (rad)"""
    return math.atan2(-xy[1], -xy[0])


# =============================================================================================== planning (URDF IK)
class FastArm:
    """V3.63 FAST IK: numpy forward kinematics + analytic geometric Jacobian of the URDF chain base_link -> link7 ->
    tool (same URDF / tool transform as NERO_V315; ~10x faster than the numeric-Jacobian pose_ik of V3.61)."""

    def __init__(self):
        import NERO_V315_Consistency_Safety as CS
        U = CS.load_sim_urdf()
        self.joints = [f"joint{i}" for i in range(1, 8)]
        self.steps = []                                   # (fixed 4x4, axis (unit) or None, arm joint index or None)
        for jn in CS.link_chain(U, "base_link", "link7"):
            j = U["joints"][jn]
            Tf = np.array(CS._T(CS._rpy_mat(*j["rpy"]), j["xyz"]), float)
            if j["type"] in ("revolute", "continuous") and jn in self.joints:
                ax = np.asarray(j["axis"], float)
                self.steps.append((Tf, ax / np.linalg.norm(ax), self.joints.index(jn)))
            else:
                self.steps.append((Tf, None, None))
        self.tool = np.array(CS.tool_transform_from_sim(U), float)
        self.lo = np.array([a for a, _ in JOINT_LIMITS])
        self.hi = np.array([b for _, b in JOINT_LIMITS])

    @staticmethod
    def _rot(ax, q):
        x, y, z = ax
        c, s_, C = math.cos(q), math.sin(q), 1 - math.cos(q)
        return np.array([[c + x * x * C, x * y * C - z * s_, x * z * C + y * s_],
                         [y * x * C + z * s_, c + y * y * C, y * z * C - x * s_],
                         [z * x * C - y * s_, z * y * C + x * s_, c + z * z * C]])

    def fk(self, q, jac=False):
        T = np.eye(4)
        origins, axes = [None] * 7, [None] * 7
        for Tf, ax, k in self.steps:
            T = T @ Tf
            if ax is not None:
                origins[k], axes[k] = T[:3, 3].copy(), T[:3, :3] @ ax
                M = np.eye(4)
                M[:3, :3] = self._rot(ax, q[k])
                T = T @ M
        T = T @ self.tool
        if not jac:
            return T
        Jm = np.zeros((6, 7))
        for k in range(7):
            Jm[:3, k] = np.cross(axes[k], T[:3, 3] - origins[k])
            Jm[3:, k] = axes[k]
        return T, Jm

    def ik(self, seeds, xyz, R_des, iters=80, tol_m=5e-4, tol_deg=1.0, w_rot=0.10):
        xyz, R_des = np.asarray(xyz, float), np.asarray(R_des, float)
        best = (None, 1e9, 1e9)
        for sd in seeds:
            q = np.clip(np.asarray(sd, float), self.lo + 1e-3, self.hi - 1e-3)
            lam = 1e-3
            for it in range(iters):
                T, Jm = self.fk(q, True)
                ep = xyz - T[:3, 3]
                Re = R_des @ T[:3, :3].T
                ang = math.acos(max(-1.0, min(1.0, (np.trace(Re) - 1) / 2)))
                if ang < 1e-9:
                    er = np.zeros(3)
                else:
                    er = ang / (2 * math.sin(ang)) * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]]) \
                        if ang < math.pi - 1e-6 else np.zeros(3) + ang
                pe, re = float(np.linalg.norm(ep)), math.degrees(float(np.linalg.norm(er)))
                if pe < tol_m and re < tol_deg:
                    return [float(v) for v in q], pe, re
                if (it >= 20 and pe > 0.02) or (it >= 40 and (pe > 0.003 or re > 5.0)):
                    break                                   # this seed is not converging (joint limit / unreachable)
                e = np.concatenate([ep, w_rot * er])
                Jw = np.vstack([Jm[:3], w_rot * Jm[3:]])
                dq = Jw.T @ np.linalg.solve(Jw @ Jw.T + lam * np.eye(6), e)
                qn = np.clip(q + dq, self.lo + 1e-4, self.hi - 1e-4)
                Tn = self.fk(qn)
                if np.linalg.norm(xyz - Tn[:3, 3]) + w_rot * np.linalg.norm(er) * 0 <= pe + 1e-12 or lam > 1e2:
                    q, lam = qn, max(1e-6, lam * 0.5)
                else:
                    q, lam = qn, lam * 4.0
            if pe + re / 1000.0 < best[1] + best[2] / 1000.0:
                best = (None, pe, re)
        return best


def _ik_tools():
    arm = FastArm()

    def pose_ik(U, J, limits, seeds, T, xyz, R_des, iters=80, **_):
        return arm.ik(seeds, xyz, R_des, iters=iters)
    return arm, arm.joints, arm.tool, pose_ik


_IK = {}


def ik_chain(chain, q_start=None, start_first=True):
    """strict 6-DoF IK along [(name, xyz, R), ...]; -> dict(ok, why, min_margin_deg, binding, q).
    start_first=False tries the generic seeds before q_start for the first pose (another IK branch of the 7-DoF arm)."""
    if "t" not in _IK:
        _IK["t"] = _ik_tools()
    U, J, T, pose_ik = _IK["t"]
    q_prev, worst, binding, qs = (list(q_start) if q_start else None), 1e9, "", []
    for name, p, R in chain:
        a1 = math.atan2(p[1], p[0])
        a1 = a1 + math.pi if a1 < 0 else a1 - math.pi
        generic = [[a1, j2, 0.0, j4, 0.0, j6, j7] for j2 in (0.2, 0.6) for j4 in (0.8, 1.6)
                   for j6 in (0.5, -0.2) for j7 in (0.0, 1.0, -1.0)]
        if q_prev and name.startswith("~"):
            seeds = [q_prev]                           # V3.64 straight-segment point: strict continuity, no branch change
        elif q_prev and (qs or start_first is None):
            seeds = [q_prev] + generic[::12]           # later poses of a chain: continue the branch (fast fail)
        elif q_prev and start_first:
            seeds = [q_prev] + generic
        else:
            seeds = generic + ([q_prev] if q_prev else [])
        q, pe, re = pose_ik(U, J, JOINT_LIMITS, seeds, T, [float(v) for v in p], np.asarray(R).tolist(), iters=60)
        if q is None:
            return {"ok": False, "why": f"{name}: no 6-DoF IK (pos {pe * 1000:.1f} mm, rot {re:.1f} deg)",
                    "min_margin_deg": None, "binding": name, "q": qs}
        if name.startswith("~") and q_prev and max(abs(a - b) for a, b in zip(q, q_prev)) > math.radians(12.0):
            return {"ok": False, "why": f"{name}: straight segment needs an IK branch change", "min_margin_deg": None,
                    "binding": name, "q": qs}
        ms = [math.degrees(min(v - lo, hi - v)) for v, (lo, hi) in zip(q, JOINT_LIMITS)]
        k = min(range(7), key=lambda i: ms[i])
        if ms[k] < worst:
            worst, binding = ms[k], f"J{k + 1} at {name}"
        qs.append(q)
        q_prev = q
    return {"ok": True, "why": "", "min_margin_deg": round(worst, 1), "binding": binding, "q": qs}


def segment_points(tag, p0, p1, R, n=5):
    """V3.64: intermediate poses of a straight TCP segment (names start with '~' -> strict IK continuity)"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    return [(f"~{tag}{i}", p0 + (p1 - p0) * i / n, R) for i in range(1, n)]


def joint_margins_deg(q):
    return [math.degrees(min(v - lo, hi - v)) for v, (lo, hi) in zip(q, JOINT_LIMITS)]


def check_joint_path(qs, margin_deg=PATH_MARGIN_DEG, jump_deg=PATH_JUMP_DEG):
    """V3.63: a joint path is executable only if every point keeps all joints >= margin_deg inside the (MuJoCo)
    limits and no joint changes by more than jump_deg between consecutive points.  -> dict(ok, why, ...)"""
    worst, wj, jump, jj = 1e9, -1, 0.0, -1
    for i, q in enumerate(qs):
        m = joint_margins_deg(q)
        k = min(range(7), key=lambda j: m[j])
        if m[k] < worst:
            worst, wj = m[k], k
        if i:
            d = [abs(math.degrees(a - b)) for a, b in zip(q, qs[i - 1])]
            k2 = max(range(7), key=lambda j: d[j])
            if d[k2] > jump:
                jump, jj = d[k2], k2
    ok = worst >= margin_deg and jump <= jump_deg
    why = "" if ok else (f"J{wj + 1} only {worst:.1f} deg from its limit" if worst < margin_deg
                         else f"J{jj + 1} jumps {jump:.1f} deg between two path points (IK branch flip)")
    return {"ok": bool(ok), "why": why, "min_margin_deg": round(worst, 1), "binding": f"J{wj + 1}",
            "max_jump_deg": round(jump, 1), "points": len(qs)}


def ik_from(q_now, xyz, R):
    """strict 6-DoF IK seeded from the current joints (stays on the current branch when possible) -> q or None"""
    r = ik_chain([("target", np.asarray(xyz, float), np.asarray(R, float))], q_now)
    return r["q"][-1] if r["ok"] and r["q"] else None


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


def plan_cup_stack(knobs, src_rim, src_quat, dst_rim, dst_quat, q_start=None, log=None, _wide=False, profile=None,
                   allow_wide=True, deadline=None):
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
    # V3.63: joint search over grasp candidates (rim angle x flip x IK branch) and the place rim angle, so that the
    # WHOLE chain (pregrasp ... retreat) keeps every joint >= 8 deg and J7 >= 15 deg from its limits during placing
    # (a grasp chain seeded from the real home posture can end on a branch where J7 runs into its limit while placing)
    rim_lift = np.array([d_base[0], d_base[1], base_lift + g["h"]])
    rim_rel = np.array([d_base[0], d_base[1], base_release + g["h"]])
    best = None
    tried = 0
    grasp_found = False
    # V3.68 FAST PLAN: the wide search may use a coarser step and a time budget (real profile; simulation unchanged)
    step = int(getattr(profile, "wide_step_deg", 15) or 15) if _wide else 15
    budget = getattr(profile, "plan_time_budget_s", None) if _wide else None
    t_end = time.time() + float(budget) if budget else None
    if deadline is not None:                       # V3.69 BEST-OF PLAN: one budget from the start of the planning
        over = lambda: time.time() > deadline
    else:
        over = lambda: bool(t_end and time.time() > t_end and best is not None
                            and best["key"][2] >= MIN_MARGIN_RELAXED_DEG)
    g_angles = _angle_order(math.degrees(base_dir(s_rim)) + float(knobs["cup_grasp_rim_angle_deg"]),
                            step=step, span=180 if _wide else 45)
    p_angles = _angle_order(math.degrees(base_dir(d_rim)) + float(knobs["cup_place_rim_angle_deg"]),
                            step=step, span=180 if _wide else 90)
    if not _wide and getattr(profile, "name", "") == "so101":
        for a_ in _angle_order(math.degrees(base_dir(s_rim)) - 30.0, step=15, span=30):
            if all(abs(a_ - b_) > 1e-4 for b_ in g_angles):
                g_angles.append(a_)
        for a_ in _angle_order(math.degrees(base_dir(d_rim)) - 35.0, step=15, span=45):
            if all(abs(a_ - b_) > 1e-4 for b_ in p_angles):
                p_angles.append(a_)
    for phi in g_angles:
        if over():
            break
        for flip in (profile or OFFICIAL_PROFILE).flips:
            p, R = pinch_tcp(s_rim, phi, depth, flip, tilt, profile)
            pp = p + (profile or OFFICIAL_PROFILE).pre_offset(R, pre)          # V3.64: official = [0, 0, pre]
            ch = ([("pregrasp", pp, R)] + segment_points("d", pp, p, R) + [("grasp", p, R)]
                  + segment_points("l", p, p + [0, 0, dz_lift], R) + [("lift", p + [0, 0, dz_lift], R)])
            for start_first in ((True, False) if q_start is not None else (True,)):
                r = ik_chain(ch, q_start, start_first=start_first)
                tried += 1
                if not r["ok"] or r["min_margin_deg"] < MIN_MARGIN_RELAXED_DEG:
                    continue
                grasp_found = True
                bg = {"phi": phi, "flip": flip, "r": r, "tcp": p, "R": R}
                lift_tcp = p + [0, 0, dz_lift]
                for phi_p in p_angles:
                    if over():
                        break
                    p1, R1 = pinch_tcp(rim_lift, phi_p, depth, flip, tilt, profile)
                    p2, _ = pinch_tcp(rim_rel, phi_p, depth, flip, tilt, profile)
                    chp = ([("rotate", lift_tcp, R1), ("above_target", p1, R1)] + segment_points("w", p1, p2, R1, 3)
                           + [("release", p2, R1), ("retreat", p2 + [0, 0, 0.08], R1)])
                    rp = ik_chain(chp, r["q"][-1], start_first=None)
                    if not rp["ok"]:
                        continue
                    # the cup is free at the retreat pose: 4 deg is enough there, the margin key uses the held poses
                    m_ret = min(joint_margins_deg(rp["q"][-1]))
                    if m_ret < MIN_MARGIN_RELAXED_DEG:
                        continue
                    rp["min_margin_deg"] = round(min(min(joint_margins_deg(q_)) for q_ in rp["q"][:-1]), 1)
                    j7 = min(math.degrees(min(q[6] - JOINT_LIMITS[6][0], JOINT_LIMITS[6][1] - q[6])) for q in rp["q"][:-1])
                    rp["j7_margin_deg"] = round(j7, 1)
                    mm = min(r["min_margin_deg"], rp["min_margin_deg"])
                    key = (mm >= MIN_MARGIN_DEG and j7 >= PLACE_J7_PREF_DEG, mm >= MIN_MARGIN_DEG, mm)
                    if best is None or key > best["key"]:
                        best = {"key": key, "g": bg, "p": {"phi": phi_p, "r": rp, "tcp_above": p1, "tcp_release": p2,
                                                           "R": R1}}
                    if key[0]:
                        break
                if best and best["key"][0]:
                    break
            if best and best["key"][0]:
                break
        if best and best["key"][0]:
            break
    if not grasp_found and not _wide and allow_wide:
        if log:
            log("[INFO] [V3.64 CUP PLAN] no reachable grasp within +-45 deg of the knob angle -> wide search")
        return plan_cup_stack(knobs, src_rim, src_quat, dst_rim, dst_quat, q_start=q_start, log=log, _wide=True,
                              profile=profile)
    if not grasp_found:
        return {"ok": False, "reason": "GRASP_UNREACHABLE", "detail": f"no reachable rim pinch on the source cup ({tried} tried)"}
    best_g, best_p = (best["g"], best["p"]) if best else (None, None)
    flip = best_g["flip"] if best_g else False
    lift_tcp = best_g["tcp"] + [0, 0, dz_lift] if best_g else None
    # V3.64: also search wide when the best chain would be refused anyway (margin < 4 deg; plans >= 4 deg unchanged)
    wide_below = max(MIN_MARGIN_RELAXED_DEG, float(getattr(profile, "wide_search_below_deg", None) or 0.0))
    if (best_p is None or best["key"][2] < wide_below) and not _wide and allow_wide:
        # V3.63: nothing in the usual window -> one wide search (all rim angles), slower but only on failure
        if log:
            log("[INFO] [V3.63 CUP PLAN] no reachable chain within +-45 / +-90 deg of the knob angles -> wide search")
        return plan_cup_stack(knobs, src_rim, src_quat, dst_rim, dst_quat, q_start=q_start, log=log, _wide=True,
                              profile=profile)
    if best_p is None:
        return {"ok": False, "reason": "PLACE_UNREACHABLE",
                "detail": "the source cup can be grasped but no reachable carry/release pose over the target with the same grip"}
    wq = lambda R: R_to_quat_xyzw(R)
    wps = [("pregrasp", best_g["tcp"] + (profile or OFFICIAL_PROFILE).pre_offset(best_g["R"], pre), best_g["R"]), ("grasp", best_g["tcp"], best_g["R"]),
           ("lift", lift_tcp, best_g["R"]), ("rotate", lift_tcp, best_p["R"]), ("above_target", best_p["tcp_above"], best_p["R"]),
           ("release", best_p["tcp_release"], best_p["R"]), ("retreat", best_p["tcp_release"] + [0, 0, 0.08], best_p["R"])]
    margin = min(best_g["r"]["min_margin_deg"], best_p["r"]["min_margin_deg"])
    out = {"ok": True, "flip": bool(flip),
           "grasp_phi_rad": float(best_g["phi"]), "place_phi_rad": float(best_p["phi"]),
           "grasp_rim_angle_rel_deg": round((math.degrees(best_g["phi"] - base_dir(s_rim)) + 180) % 360 - 180, 1),
           "place_rim_angle_rel_deg": round((math.degrees(best_p["phi"] - base_dir(d_rim)) + 180) % 360 - 180, 1),
           "grasp_margin_deg": best_g["r"]["min_margin_deg"], "grasp_binding": best_g["r"]["binding"],
           "place_margin_deg": best_p["r"]["min_margin_deg"], "place_binding": best_p["r"]["binding"],
           "min_joint_margin_deg": margin, "margin_ok": bool(margin >= MIN_MARGIN_DEG),
           "base_lift_z_m": round(base_lift, 4), "base_release_z_m": round(base_release, 4),
           "waypoints": [{"name": n, "tcp_xyz_m": [round(float(v), 4) for v in p], "quat_xyzw": [round(v, 5) for v in wq(R)]}
                         for n, p, R in wps],
           "_wps": {n: (np.asarray(p, float), np.asarray(R, float)) for n, p, R in wps},
           "place_j7_margin_deg": best_p["r"].get("j7_margin_deg"),
           "_q": dict(zip(["pregrasp", "grasp", "lift", "rotate", "above_target", "release", "retreat"],
                          [list(q) for q in best_g["r"]["q"][::5]] + [list(q) for q, n in zip(best_p["r"]["q"],
                           ["rotate", "above_target", "~w1", "~w2", "release", "retreat"]) if not n.startswith("~")])),
           # V3.64: joint solutions along the straight segments (descent / lift / lowering) - strict IK continuity
           "_qpath": {"grasp": [list(q) for q in best_g["r"]["q"][0:6]], "lift": [list(q) for q in best_g["r"]["q"][5:11]],
                      "release": [list(q) for q in best_p["r"]["q"][1:5]]}}
    if log:
        log(f"[V3.63 CUP PLAN] grasp rim {out['grasp_rim_angle_rel_deg']:+.0f} deg (margin {out['grasp_margin_deg']} deg, "
            f"{out['grasp_binding']}) | place rim {out['place_rim_angle_rel_deg']:+.0f} deg (margin {out['place_margin_deg']} "
            f"deg, {out['place_binding']}, J7 {out['place_j7_margin_deg']} deg) | flip={flip}")
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


def twin_xml(cup_xy=None, gripper="official", world_xml=None, hold_names=None, soft_pads=False):
    """V3.72 TWIN WORLD: world_xml / hold_names replace the two cups (sort_items twin); defaults = the cup twin"""
    xy = dict(SC.CUP_XY if cup_xy is None else cup_xy)
    if gripper == "so101":                   # V3.64: the same SO-101 gripper parts the scene module mounts
        import NERO_V364_SO101_Gripper as G
        gp = G.gripper_parts(angle_unit="degree", soft_pads=soft_pads)   # the twin XML has no <compiler angle>; V3.76 SOFT PADS
        g_asset, g_link7, g_eq, g_act, g_contact = gp["asset"], gp["link7"], gp["equality"], gp["actuator"], gp["contact"]
    else:
        g_asset = ""
        g_link7 = ('<geom name="gripper_base_box" type="box" pos="0.0675 0 -0.0235" size="0.0325 0.040 0.032" '
                   'rgba=".3 .3 .35 .5"/>'
                   + _finger_xml("gripper_link1", "-1.27026e-06 0.707108 0.707105 -1.30815e-06", "0 0 1", "0 0.05", "finger1")
                   + _finger_xml("gripper_link2", "0.707105 -1.30815e-06 1.30815e-06 -0.707108", "0 0 -1", "-0.05 0", "finger2"))
        g_eq = '<joint name="gripper_coupling" joint1="gripper_joint2" joint2="gripper_joint1" polycoef="0 -1 0 0 0"/>'
        g_act = ('<position name="gripper" joint="gripper_joint1" kp="40" kv="5" forcerange="-20 20" forcelimited="true" '
                 'ctrlrange="0 0.05" ctrllimited="true"/><motor name="gripper_force" joint="gripper_joint1" gear="1" '
                 'ctrlrange="-50 50" ctrllimited="true" forcerange="-50 50" forcelimited="true"/>')
        g_contact = '<exclude body1="gripper_link1" body2="gripper_link2"/>'
    cups = world_xml if world_xml is not None else \
        "\n".join(SC.cup_body_xml(n, xy[n], z0=0.0005, origin=SC.CUP_ORIGIN) for n in SC.CUP_NAMES)
    holds = "".join(f'<connect name="hold_{n}" body1="link7" body2="{n}" anchor="0 0 0" active="false" '
                    'solref="0.008 1" solimp="0.98 0.995 0.001"/>' for n in (hold_names or SC.CUP_NAMES))
    try:                                     # V3.63: the fixed stand D435i (same pose / fovy as the backend camera)
        import NERO_V308_Camera_Perception as P8
        cam = P8.camera_xml(P8.camera_config())
    except Exception:
        cam = ""
    return f"""<mujoco model="nero_v362_cup_twin">
  <option timestep="0.001" integrator="implicitfast"/>
  <visual><global offwidth="1280" offheight="960"/><headlight ambient="0.25 0.25 0.25" diffuse="0.30 0.30 0.30"/></visual>
  <asset>{SC.CUP_ASSETS}{g_asset}</asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <light name="eggscene_key" pos="0.25 -0.65 2.20" dir="-0.30 0.22 -1" directional="true" castshadow="true"
           diffuse="0.92 0.84 0.72" specular="0.22 0.20 0.18"/>
    <light name="eggscene_fill" pos="-1.20 0.80 1.50" dir="0.55 -0.25 -1" directional="true" castshadow="false"
           diffuse="0.34 0.40 0.48" specular="0.05 0.06 0.08"/>
    {cam}
    <geom name="play_table" type="box" pos="-0.385 0 -0.03" size="0.335 0.38 0.03" rgba=".93 .93 .91 1"
          friction="1.2 0.01 0.001" solref="0.008 1" solimp="0.96 0.995 0.0005" margin="0.001"/>
    <body name="link7" mocap="true" pos="-0.3 0 0.45">
      <!-- official: APPROXIMATE gripper housing (its STL meshes are not in this repository); so101: V3.64 parts -->
      {g_link7}
    </body>
    {cups}
  </worldbody>
  <contact>{g_contact}</contact>
  <equality>
    {g_eq}
    {holds}
  </equality>
  <actuator>{g_act}</actuator>
</mujoco>"""


class TwinIO:
    """Headless MuJoCo twin of the backend's gripper + cups (see module docstring)."""
    name = "twin"

    def __init__(self, log=None, cup_xy=None, start_tcp=None, pose_mode="truth", gripper="official", world_xml=None,
                 hold_names=None, soft_pads=False):
        import mujoco
        self._pose_mode = pose_mode
        self._gripper = gripper
        self.mj = mujoco
        self.hold_names = list(hold_names or SC.CUP_NAMES)          # V3.72 TWIN WORLD
        self.m = mujoco.MjModel.from_xml_string(twin_xml(cup_xy, gripper, world_xml, hold_names, soft_pads))
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
        # V3.63 TWIN ARM KINEMATICS: the twin now carries the 7 joint angles; every motion is executed in joint space
        # (MoveIt-like: Cartesian moves follow an IK-continuity path that is checked like the backend path) and the
        # gripper follows the URDF forward kinematics.  Still no arm dynamics / controller.
        self.arm = FastArm()
        self.q = [0.0] * 7 if start_tcp is None else None
        self.safety_latched, self.joint_speed = False, 0.35            # rad/s for joint moves
        if self.q is None:
            R0 = pinch_frame(0.0, False, False)
            self.q = ik_from([0.3, 0.3, 0.0, 1.2, 0.0, 0.4, 0.0], start_tcp, R0) or [0.0] * 7
        self._apply_q(self.q)
        mujoco.mj_forward(self.m, self.d)
        self.step(400)

    def _apply_q(self, q):
        T = self.arm.fk(np.asarray(q, float))
        self._set_tcp(T[:3, 3], T[:3, :3])

    def current_q(self):
        return list(self.q)

    def _run_q_path(self, qs, dur):
        """drive the gripper along a joint path (list of q) in `dur` seconds"""
        n = max(2, int(dur / 0.005))
        qs = [np.asarray(q, float) for q in qs]
        seg = len(qs) - 1
        for i in range(1, n + 1):
            u = i / n
            u = u * u * (3 - 2 * u)
            x = u * seg
            k = min(seg - 1, int(x))
            q = qs[k] + (x - k) * (qs[k + 1] - qs[k])
            self._apply_q(q)
            self.step(5)
        self.q = [float(v) for v in qs[-1]]
        self.step(250)

    def move_joints(self, label, q, slow=None, speed=None):
        if self.safety_latched:
            return {"ok": False, "label": label, "error": "SAFETY LATCH HELD", "safety": True}
        chk = check_joint_path([self.q, list(q)], jump_deg=1e9)
        if not chk["ok"]:
            return {"ok": False, "label": label, "error": f"joint goal rejected: {chk['why']}"}
        dq = max(abs(a - b) for a, b in zip(q, self.q))
        self._run_q_path([self.q, list(q)], max(0.3, dq / self.joint_speed) * float(slow or 1.0))
        return {"ok": True, "label": label, "joint_error_deg": 0.0, "tcp_error_mm": 0.0, "kind": "joints"}

    def latch_pending(self):
        return bool(self.safety_latched)

    def clear_latch(self, label):
        self.safety_latched = False
        return True, "ok"

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
            if {"gripper_base_box", "so101_housing"} & {(self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_GEOM, g1) or ""),
                                      (self.mj.mj_id2name(self.m, self.mj.mjtObj.mjOBJ_GEOM, g2) or "")}:
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
        for n in self.hold_names:
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

    def _cartesian_q_path(self, xyz, quat, step_m=0.002):
        """MoveIt-like Cartesian path: IK continuity from the current joints (None if it breaks)"""
        T0 = self.arm.fk(np.asarray(self.q, float))
        p0, p1 = T0[:3, 3].copy(), np.asarray(xyz, float)
        q0, q1 = np.array(R_to_quat_xyzw(T0[:3, :3])), np.array(quat, float)
        if q0 @ q1 < 0:
            q1 = -q1
        ang = 2 * math.acos(min(1.0, abs(float(q0 @ q1))))
        n = max(2, int(max(np.linalg.norm(p1 - p0) / step_m, ang / math.radians(1.0))))
        qs, qp = [list(self.q)], list(self.q)
        for i in range(1, n + 1):
            u = i / n
            qq = q0 * (1 - u) + q1 * u
            qq /= np.linalg.norm(qq)
            r = self.arm.ik([qp], p0 + u * (p1 - p0), quat_xyzw_to_R(qq), iters=30)
            if r[0] is None:
                return None, f"Cartesian corridor only {100.0 * (i - 1) / n:.1f}% (IK continuity broke)"
            qp = r[0]
            qs.append(qp)
        return qs, ""

    def move_line(self, label, xyz, quat, speed=0.05, slow=None):
        """V3.63: same contract as BackendIO.move_line - a path that fails check_joint_path is REFUSED (not run)"""
        if self.safety_latched:
            return {"ok": False, "label": label, "error": "SAFETY LATCH HELD", "safety": True}
        qs, why = self._cartesian_q_path(xyz, quat)
        if qs is None:
            return {"ok": False, "refused": True, "label": label, "why": why}
        chk = check_joint_path(qs)
        if not chk["ok"]:
            return {"ok": False, "refused": True, "label": label, "why": chk["why"], "path_check": chk}
        T0 = self.arm.fk(np.asarray(qs[0], float))
        dist = float(np.linalg.norm(np.asarray(xyz) - T0[:3, 3]))
        q0, q1 = np.array(R_to_quat_xyzw(T0[:3, :3])), np.array(quat, float)
        ang = 2 * math.acos(min(1.0, abs(float(q0 @ q1))))
        self._run_q_path(qs, max(dist / max(speed, 1e-3), ang / self.max_ang_rate, 0.2))
        err = float(np.linalg.norm(self._tcp_T()[:3, 3] - np.asarray(xyz)))
        return {"ok": err < 0.003, "tcp_error_mm": round(err * 1000, 2), "label": label, "kind": "line",
                "path_check": chk}

    def move_pose(self, label, xyz, quat, speed=0.10, slow=None):
        q = ik_from(self.q, xyz, quat_xyzw_to_R(quat))
        if q is None:
            return {"ok": False, "label": label, "error": "no IK"}
        return self.move_joints(label, q, slow=slow)

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
        if p[2] < 0.25:
            self.move_line(label, [p[0], p[1], 0.30], q, 0.10)
        r = self.move_joints(label, [0.0] * 7)
        return {"ok": bool(r.get("ok"))}

    def render_fixed_camera(self, frames=1, rng_seed=363):
        """V3.63: RGB + z-depth of the fixed stand D435i exactly as the backend camera process makes them (V3.08
        intrinsics, D435i range / mm quantisation, optional noise k, temporal fusion).  Needs an OpenGL context."""
        import NERO_V308_Camera_Perception as P8
        cfg = P8.camera_config()
        if getattr(self, "_renderer", None) is None:
            self.m.vis.global_.offwidth = max(int(self.m.vis.global_.offwidth), int(cfg["width"]))
            self.m.vis.global_.offheight = max(int(self.m.vis.global_.offheight), int(cfg["height"]))
            self._renderer = self.mj.Renderer(self.m, height=int(cfg["height"]), width=int(cfg["width"]))
            self._rng = np.random.default_rng(rng_seed)
        r = self._renderer
        r.disable_depth_rendering()
        r.update_scene(self.d, camera=P8.CAMERA_NAME)
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(self.d, camera=P8.CAMERA_NAME)
        raw = r.render().copy()
        r.disable_depth_rendering()
        n = int(cfg["depth_frames"]) if float(cfg["depth_noise_k"]) > 0 else 1
        depth = P8.temporal_fuse([P8._depth_postprocess(raw, cfg, self._rng) for _ in range(max(n, frames))])
        return rgb, depth, P8.intrinsics(cfg), P8.extrinsics(cfg)

    def pose_mode(self):
        return self._pose_mode

    def gripper_profile(self):
        return so101_profile() if self._gripper == "so101" else OFFICIAL_PROFILE

    def perceive(self, tag):
        """V3.63: fixed-camera capture of the twin -> NERO_V363_Cup_Perception.perceive_cups (needs OpenGL)"""
        import NERO_V363_Cup_Perception as CP
        rgb, depth, K, T = self.render_fixed_camera()
        res = CP.perceive_cups(rgb, depth, K, T, robot_boxes=self.gripper_boxes())
        self.log(f"[V3.63 FIXED CUP PERCEPTION] {tag} | " + " ".join(
            f"{c['kind']}@({c['base_xyz_m'][0]:+.3f},{c['base_xyz_m'][1]:+.3f}) h={c['height_mm']:.0f}mm" for c in res["cups"]))
        return res

    def gripper_boxes(self):
        """oriented boxes of the twin gripper (robot self-filter, like _robot_boxes in the camera process)"""
        boxes = []
        for gi in range(self.m.ngeom):
            if int(self.m.geom_bodyid[gi]) not in (self.l7, self.m.body("gripper_link1").id, self.m.body("gripper_link2").id):
                continue
            aabb = np.asarray(self.m.geom_aabb[gi], float)
            R = np.asarray(self.d.geom_xmat[gi], float).reshape(3, 3)
            boxes.append((np.asarray(self.d.geom_xpos[gi], float) + R @ aabb[:3], R, aabb[3:]))
        return boxes

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
                "safety": trips > 0, "label": label}

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

    def gripper_profile(self):
        """V3.64 SO-101 GRIPPER ON NERO: the gripper of the scene chosen in the console (stack_cups_so101 -> SO-101)"""
        return profile_for_scene(SC.selected_variant())

    def pose_mode(self):
        f = self.c.get("pose_source_mode")
        mode = f() if f else "truth"
        return "camera" if mode == "camera" and not self.c.get("wrist_selected", lambda: False)() else "truth"

    def perceive(self, tag):
        """V3.63: one capture of the fixed stand D435i (camera process, V3.31 protocol) -> its "cups" result"""
        payload = self.c["capture"](f"cups:{tag}")
        if not isinstance(payload, dict) or "cups" not in payload:
            raise RuntimeError("PERCEPTION_NO_CUP_RESULT: the camera process did not return cups (is it the V3.63 "
                               "NERO_V331_Fixed_Camera.py?)")
        return payload["cups"]

    # -- V3.63 JOINT-SPACE SAFE MOTION
    def current_q(self):
        return [float(v) for v in self.core._arm_positions_from_joint_state(self.core._current_joint_state_copy())]

    @staticmethod
    def traj_q(traj):
        jt = traj.joint_trajectory
        names = [str(n) for n in jt.joint_names]
        idx = [names.index(f"joint{i}") for i in range(1, 8)]
        return [[float(pt.positions[i]) for i in idx] for pt in jt.points]

    def _exec(self, label, traj, slow, s0):
        t0 = time.time()
        try:
            if slow and self.c.get("slow") is not None:
                self.c["slow"](traj, factor=float(slow))
            self.core._execute_trajectory(traj)
            return None
        except Exception as exc:
            m = str(exc)
            out = {"ok": False, "label": label, "error": m[:300], "tcp_error_mm": None,
                   "safety": int(self.sh["safety_trip_count"].value) != s0 or any(
                       k in m for k in ("PHYSICAL", "SAFETY", "RUNAWAY", "TRACKING")),
                   "safety_trips": int(self.sh["safety_trip_count"].value) - s0}
            if out["safety"]:
                out.update(self.v377_trip_info(t0))
            return out

    def v377_trip_info(self, t0):
        """V3.77 TRIP REASON IN THE RESULT: the backend writes diag_trip_<n>_<reason>.csv (NERO_DIAG_DIR) at every
        safety trip (runaway_j<k> = joint k faster than 6 rad/s or beyond its limit, penetration = a robot body in a
        protected surface); the newest reason since t0 + the joint closest to its limit now go into the tool result
        (user run V3.76: the log only said "STATE 2", the reason was in backend.log)"""
        out = {}
        d = os.environ.get("NERO_DIAG_DIR", "").strip()
        try:
            files = [(os.path.getmtime(os.path.join(d, f)), f) for f in os.listdir(d) if f.startswith("diag_trip_")]
            files = sorted(x for x in files if x[0] >= t0 - 1.0)
            if files:
                reasons = [f[:-4].split("_", 3)[-1] for _t, f in files]
                text = {"penetration": "a robot body pressed into a protected surface (table / bin / safe)"}
                last = reasons[-1]
                out["trip_reason"] = last
                out["trip_reasons"] = reasons[-4:]
                out["trip_meaning"] = text.get(last) or (
                    f"joint {last.split('_j')[-1]} faster than 6 rad/s or beyond its limit (RUNAWAY)"
                    if last.startswith("runaway") else last)
        except Exception:
            pass
        try:
            ms = joint_margins_deg(self.current_q())
            k = min(range(7), key=lambda i: ms[i])
            out["closest_joint_limit"] = {"joint": f"J{k + 1}", "margin_deg": round(ms[k], 1)}
        except Exception:
            pass
        if out:
            self.emit(f"[WARN] [V3.77 SAFETY] safety stop reason: {out.get('trip_reason', 'unknown')} "
                      f"({out.get('trip_meaning', 'no diag file')}); closest joint limit now "
                      f"{out.get('closest_joint_limit')}")
        return out

    def move_line(self, label, xyz, quat, speed=0.05, slow=None):
        """Cartesian path from MoveIt; every point is checked (all joints >= PATH_MARGIN_DEG inside the MuJoCo limits,
        <= PATH_JUMP_DEG per point) BEFORE it runs.  A failing path is REFUSED and not executed - the skill then moves
        in joint space to its own IK solution (V3.62 run: a 4-point centring path pushed J7 past its limit)."""
        self.c["require_arm"]()
        s0 = int(self.sh["safety_trip_count"].value)
        try:
            q0 = self.current_q()
            traj = self.core._cartesian_waypoint_plan([list(xyz)], list(quat), label=label, max_step=0.004,
                                                      max_speed=float(speed))
        except Exception as exc:
            r = {"ok": False, "refused": True, "label": label, "why": str(exc)[:200]}
            self._v375_report(label, None, xyz, quat, r)
            return r
        if traj is None:
            return self._post(label, xyz, s0)
        chk = check_joint_path([q0] + self.traj_q(traj))
        if not chk["ok"]:
            return {"ok": False, "refused": True, "label": label, "why": chk["why"], "path_check": chk}
        err = self._exec(label, traj, slow if slow else (2.5 if float(speed) <= 0.02 else None), s0)
        if err:
            return err
        r = self._post(label, xyz, s0)
        r["path_check"] = chk
        return r

    def move_joints(self, label, q, slow=None):
        """MoveIt joint goal to an IK solution that the skill checked (margins); the planned path is checked too"""
        self.c["require_arm"]()
        s0 = int(self.sh["safety_trip_count"].value)
        try:
            q0 = self.current_q()
            traj = self.core._plan_joint_solution([float(v) for v in q], velocity_scale=0.10, accel_scale=0.08,
                                                  planning_attempts=6, planning_time=5.0)
        except Exception as exc:
            r = {"ok": False, "label": label, "error": f"joint plan failed: {str(exc)[:200]}"}
            self._v375_report(label, q, None, None, r)
            return r
        if traj is None:
            return {"ok": False, "label": label, "error": "joint plan returned no trajectory"}
        chk = check_joint_path([q0] + self.traj_q(traj), jump_deg=30.0)
        if not chk["ok"]:
            return {"ok": False, "label": label, "error": f"joint path rejected: {chk['why']}", "path_check": chk}
        err = self._exec(label, traj, slow, s0)
        if err:
            return err
        self._sleep(0.15)
        qe = self.current_q()
        jerr = max(abs(math.degrees(a - b)) for a, b in zip(qe, q))
        # V3.64 JOINT SETTLE: the V3.63 run stopped on a 3.22 deg settling error right after the trajectory ended -
        # wait up to 1.2 s for the arm to settle, then accept <= 4 deg (the skill re-checks positions anyway)
        t_settle = 0.0
        while jerr > 3.0 and t_settle < 1.2 and int(self.sh["safety_trip_count"].value) == s0:
            self._sleep(0.2)
            t_settle += 0.2
            qe = self.current_q()
            jerr = max(abs(math.degrees(a - b)) for a, b in zip(qe, q))
        trips = int(self.sh["safety_trip_count"].value) - s0
        return {"ok": jerr <= 4.0 and trips == 0, "label": label, "joint_error_deg": round(jerr, 2),
                "safety_trips": trips, "safety": trips > 0, "kind": "joints"}

    def _v375_report(self, label, q, xyz, quat, r):
        """V3.75 COLLISION PRE-CHECK: MoveIt refused / failed -> log WHICH bodies collide at the target state (MoveIt
        /check_state_validity, every contact pair) and whether the mounted SO-101 itself (coarse boxes) touches the
        table / bins; the pairs also go into the result ("collision").  Logging only - never changes the motion."""
        try:
            import NERO_V375_Collision_Check as CC
            if q is None and xyz is not None:
                q = ik_from(self.current_q(), xyz, quat_xyzw_to_R(quat))
            if q is None:
                return None
            so101 = SC.gripper_variant() == "so101"
            opening = float(self.sh["gripper_actual_mm"].value) if "gripper_actual_mm" in self.sh else None
            a = CC.assess(self.core, q, opening, CC.world_boxes(SC.active_scene()))
            if not so101:                                  # the official gripper: MoveIt's model is the real one
                a["so101"] = []
        except Exception as exc:
            self.emit(f"[WARN] [V3.75 MOVEIT] {label}: collision report failed: {str(exc)[:160]}")
            return None
        r["collision"] = {k: a[k] for k in ("blocking", "model_only", "so101", "moveit_available")}
        if not a["moveit_available"]:
            self.emit(f"[INFO] [V3.75 MOVEIT] {label}: MoveIt state check not available - colliding bodies unknown"
                      + (f"; SO-101 boxes touch: {CC.pairs_text(a['so101'])}" if a["so101"] else ""))
        else:
            self.emit(f"[INFO] [V3.75 MOVEIT] {label}: target state "
                      + ("valid in MoveIt" if not (a["blocking"] or a["model_only"]) else
                         "collides in MoveIt: " + CC.pairs_text(a["blocking"] + a["model_only"], 6))
                      + (f" (MoveIt gripper model only: {CC.pairs_text(a['model_only'])})" if a["model_only"] else "")
                      + (f"; SO-101 boxes touch: {CC.pairs_text(a['so101'])}" if a["so101"] else "; SO-101 boxes clear"))
        return a

    def latch_pending(self):
        f = self.c.get("latch_pending")
        return bool(f()) if f else False

    def clear_latch(self, label):
        f = self.c.get("clear_latch")
        if not f:
            return False, "backend recovery hook not available"
        return f(label)

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
        self.safety_stops = {}
        self.motions = []
        self.grip_offset = None
        # V3.63 FIXED D435i CUP PERCEPTION: pose source "camera" = the fixed stand D435i (no simulator truth)
        self.camera = getattr(io, "pose_mode", lambda: "truth")() == "camera"
        self.prof = getattr(io, "gripper_profile", lambda: OFFICIAL_PROFILE)()     # V3.64 GRIPPER PROFILES
        cap = self.prof.release_gap_max_mm
        if cap is not None and float(self.k["cup_release_gap_mm"]) > cap:
            self.k = dict(self.k, cup_release_gap_mm=float(cap))
            io.emit(f"[INFO] [V3.64 CUP] {self.prof.name}: release gap capped at {cap:.0f} mm (one-side pinch, "
                    "the cup leans slightly in the grip)")
        self.cam = {}                  # name -> perceived cup entry
        self.cam_stack = None          # {"upper", "lower", "height_mm", "base_dz_mm", "nested_full"} when nested
        self.cam_single_h = None       # measured height of a single cup (rim bead included) for the nested check
        self.placed = None             # (upper, lower) of the last release
        self.single_h = {}             # V3.71 STACK VERIFY BY CLUSTER: measured height of each cup alone (before a release)
        self.placed_xy = None          # V3.71: target cup base x-y at the release

    # -- helpers
    def _entry(self, base_xyz, tilt_deg, extra=None):
        g = geom()
        q = [1.0, 0.0, 0.0, 0.0] if tilt_deg < 0.5 else quat_from_tilt_wxyz(tilt_deg)
        base = np.asarray(base_xyz, float)
        ax = quat_wxyz_to_R(q)[:, 2]
        e = {"rim": base + g["h"] * ax, "base": base, "axis": ax, "quat": q, "tilt_deg": round(float(tilt_deg), 2),
             "source": "camera"}
        e.update(extra or {})
        return e

    def _perceive(self, tag):
        """V3.63: one fixed-camera capture -> cup entries named cup_1 / cup_2 (first sight: by y, like the scene; then
        nearest-neighbour tracking); a nested pair is one cluster ("stack") -> upper / lower from what was placed"""
        try:
            res = self.io.perceive(tag)
        except Exception as exc:
            raise self._Abort("PERCEPTION_UNAVAILABLE", {"detail": str(exc)[:300]})
        g = geom()
        ndz = float(self.k["cup_nested_height_mm"]) / 1000.0
        found = [c for c in res.get("cups", []) if c.get("kind") in ("cup", "stack")]
        if str(tag).startswith(("observe_scene", "locate")) and self.placed is None and self.held is None and len(found) < 2:
            occs = [c for c in res.get("cups", []) if c.get("kind") == "occluded" and c.get("base_xyz_m")]
            if 0 < len(found) + len(occs) <= 2:
                for oc in occs:
                    fh = oc.get("height_mm") if oc.get("height_mm") is not None else oc.get("height_depth_mm")
                    found.append(dict(oc, kind="cup", height_mm=fh, tilt_deg=float(oc.get("tilt_deg") or 0.0)))
        singles = [c for c in found if c["kind"] == "cup"]
        h_vals = [float(c["height_mm"]) for c in singles if c.get("height_mm") is not None]
        if not h_vals:
            h_vals = [float(c["height_depth_mm"]) for c in singles if c.get("height_depth_mm") is not None]
        if h_vals and self.cam_single_h is None:
            self.cam_single_h = float(np.median(h_vals))
        new, stack = {}, None
        prev = {n: e for n, e in self.cam.items() if n != self.held}
        free_names = [n for n in CUPS if n != self.held]
        def _sort_key(c_):
            b_ = np.asarray(c_["base_xyz_m"][:2], float)
            if prev and len(found) > len(free_names):
                return (min((float(np.linalg.norm(prev[n_]["base"][:2] - b_)) for n_ in free_names if n_ in prev), default=9.0), float(b_[1]))
            return (0.0, float(b_[1]))
        for c in sorted(found, key=_sort_key):
            b = np.asarray(c["base_xyz_m"], float)
            t_deg = float(c.get("tilt_deg") or 0.0)
            c_h = c.get("height_mm") if c.get("height_mm") is not None else c.get("height_depth_mm")
            if c["kind"] == "stack":
                if self.placed:
                    upper, lower = self.placed
                elif prev:
                    lower = min(prev, key=lambda n: np.linalg.norm(prev[n]["base"][:2] - b[:2]))
                    upper = next(n for n in CUPS if n != lower)
                else:
                    upper, lower = "cup_1", "cup_2"
                h_ref = self.cam_single_h or (g["h"] * 1000.0 + 3.3)
                h_st = float(c_h) if c_h is not None else (h_ref + ndz * 1000.0)
                dz = h_st - h_ref
                stack = {"upper": upper, "lower": lower, "height_mm": c.get("height_mm") if c.get("height_mm") is not None else round(h_st, 1),
                         "base_dz_mm": round(dz, 1), "nested_full": bool(dz <= ndz * 1000.0 + 5.0), "xy": [float(b[0]), float(b[1])]}
                new[lower] = self._entry(b, t_deg, {"in_stack": "lower"})
                new[upper] = self._entry(b + [0, 0, ndz], t_deg, {"in_stack": "upper"})
                continue
            cand = [n for n in free_names if n not in new]
            if not cand:
                continue
            if prev:
                name = min(cand, key=lambda n: np.linalg.norm(prev[n]["base"][:2] - b[:2]) if n in prev else 9.0)
            else:
                name = cand[0]
            new[name] = self._entry(b, t_deg, {"height_mm": c_h, "fit_rms_mm": c.get("fit_rms_mm")})
            # V3.74: only while nothing is held - a cup carried above the target hides the target's rim and its
            # silhouette reads short (twin: 109.5 instead of 125 mm), which spoiled the "taller after the release" check
            if self.placed is None and self.held is None and c_h is not None and float(c_h) > 0:
                self.single_h[name] = float(c_h)
        for n, e in prev.items():                   # not seen now (occluded by the arm): keep the last estimate
            if n not in new:
                new[n] = dict(e, stale=True)
        if self.held and self.held in self.cam:
            new[self.held] = self.cam[self.held]
        self.cam, self.cam_stack = new, stack
        self.cam_raw = res
        return res

    def _cup(self, name):
        if not self.camera:
            rim, q = self.io.cup_pose(name)
            r, b, ax = cup_axis_points(rim, q)
            return {"rim": r, "base": b, "axis": ax, "quat": q,
                    "tilt_deg": round(SC.cup_axis_tilt_deg(q), 2)}
        if name == self.held and self.grip_offset is not None:
            # the held cup: TCP + the offset measured at the grasp (the backend hold is a rigid CONNECT point)
            tcp, qt = self.io.tcp_pose()
            rim = np.asarray(tcp, float) + quat_xyzw_to_R(qt) @ self.grip_offset
            g = geom()
            return {"rim": rim, "base": rim - [0, 0, g["h"]], "axis": np.array([0, 0, 1.0]), "quat": [1.0, 0, 0, 0],
                    "tilt_deg": 0.0, "source": "tcp + grasp offset"}
        if name not in self.cam:
            self._perceive(f"locate {name}")
        for k_ in range(self.prof.perception_retries):
            if name in self.cam:
                break
            # V3.71 PERCEPTION RETRY: YOLOE missed the cup in this frame (confidence near the threshold) -> look again
            self.io.emit(f"[INFO] [V3.71 CUP] {name} 这一帧没认出来 → 再识别一次（{k_ + 1}/{self.prof.perception_retries}）")
            self._perceive(f"locate {name} (retry {k_ + 1})")
        if name not in self.cam:
            raise self._Abort("CUP_NOT_PERCEIVED", {"cup": name, "detail": "the fixed camera does not see this cup"})
        return self.cam[name]

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
        if self.camera:
            self._perceive("observe_scene")
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
        if self.camera:
            if self.cam_stack:
                stacked = {"upper": self.cam_stack["upper"], "lower": self.cam_stack["lower"],
                           "fully_nested": self.cam_stack["nested_full"]}
        elif st["stacked"]:
            stacked = {"upper": "cup_1", "lower": "cup_2", "fully_nested": st["nested_full"]}
        elif st21["stacked"]:
            stacked = {"upper": "cup_2", "lower": "cup_1", "fully_nested": st21["nested_full"]}
        src = ("fixed stand D435i RGB-D perception (V3.63, no simulator truth)" if self.camera
               else "simulator ground truth (TF)")
        return {"scene": "stack_cups", "pose_source": src, "objects": objs,
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
        try:
            q_now = self.io.current_q()               # V3.63: plan from the robot's real joints (its IK branch)
        except Exception:
            q_now = None
        plan = None
        last = _LAST_ANGLES.get(self.prof.name) if self.prof.remember_angles else None
        if self.prof.plan_time_budget_s is not None:
            plan = self._plan_best_of(s, t, q_now, last)
            last = None
        if last is not None:
            # V3.68 FAST PLAN: start from the rim angles of the last good plan (narrow search only)
            k2 = dict(self.k, cup_grasp_rim_angle_deg=last[0], cup_place_rim_angle_deg=last[1])
            p_ = plan_cup_stack(k2, s["rim"], s["quat"], t["rim"], t["quat"], q_start=q_now, log=self.io.emit,
                                profile=self.prof, allow_wide=False)
            if p_.get("ok") and p_["min_joint_margin_deg"] >= max(MIN_MARGIN_RELAXED_DEG,
                                                                   self.prof.wide_search_below_deg or 0.0):
                plan = p_
                self.io.emit(f"[INFO] [V3.68 CUP PLAN] started from the last good rim angles "
                             f"(grasp {last[0]:+.0f} deg, place {last[1]:+.0f} deg)")
        if plan is None:
            plan = plan_cup_stack(self.k, s["rim"], s["quat"], t["rim"], t["quat"], q_start=q_now, log=self.io.emit,
                                  profile=self.prof)
        if plan.get("ok") and self.prof.remember_angles and plan["min_joint_margin_deg"] >= MIN_MARGIN_RELAXED_DEG:
            _LAST_ANGLES[self.prof.name] = (plan["grasp_rim_angle_rel_deg"], plan["place_rim_angle_rel_deg"])
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
        # V3.67 PLAN MARGIN: 4-8 deg is executable (every path is checked point by point before it runs) - say so,
        # an LLM read the old wording as a reason to stop
        warn = [] if plan["margin_ok"] else [f"min joint margin {plan['min_joint_margin_deg']} deg (preferred >= 8 deg, "
                                             f"hard minimum {MIN_MARGIN_RELAXED_DEG:.0f} deg): EXECUTABLE - every trajectory "
                                             "is checked point by point before it runs; continue with pick_cup"]
        pub = {k: v for k, v in plan.items() if not k.startswith("_")}
        pub["executable"] = True
        return self._res("plan_cup_stack", True, {"min_joint_margin_deg": plan["min_joint_margin_deg"]},
                         nxt=["pick_cup"], warnings=warn, plan=pub)

    def _plan_best_of(self, s, t, q_now, last):
        """V3.69 BEST-OF PLAN (real profile): the last good rim angles, the narrow window and - when the best is still
        below the preferred margin - the wide search, all within ONE time budget counted from here; the plan with the
        largest joint margin wins (V3.68 kept a 4.9 deg wide result although the narrow search had 6.1 deg)"""
        t0 = time.time()
        dead = t0 + float(self.prof.plan_time_budget_s)
        enough = max(MIN_MARGIN_RELAXED_DEG, self.prof.wide_search_below_deg or 0.0, MIN_MARGIN_DEG)
        cands = []

        def _good_enough():
            # V3.71 PLAN EARLY STOP: a candidate with the preferred margin is taken at once (V3.70: every plan used
            # the whole 15 s budget although the first candidate already had 8 deg)
            return any(c[1].get("ok") and c[1]["min_joint_margin_deg"] >= enough for c in cands)
        if last is not None:
            k2 = dict(self.k, cup_grasp_rim_angle_deg=last[0], cup_place_rim_angle_deg=last[1])
            cands.append(("last", plan_cup_stack(k2, s["rim"], s["quat"], t["rim"], t["quat"], q_start=q_now,
                                                 profile=self.prof, allow_wide=False, deadline=dead)))
        if not _good_enough():
            cands.append(("narrow", plan_cup_stack(self.k, s["rim"], s["quat"], t["rim"], t["quat"], q_start=q_now,
                                                   profile=self.prof, allow_wide=False, deadline=dead)))
        good = [c for c in cands if c[1].get("ok")]
        best_m = max((c[1]["min_joint_margin_deg"] for c in good), default=-1.0)
        if best_m < max(MIN_MARGIN_RELAXED_DEG, self.prof.wide_search_below_deg or 0.0) and time.time() < dead:
            cands.append(("wide", plan_cup_stack(self.k, s["rim"], s["quat"], t["rim"], t["quat"], q_start=q_now,
                                                 profile=self.prof, _wide=True, allow_wide=False, deadline=dead)))
        good = [c for c in cands if c[1].get("ok")]
        txt = " · ".join(f"{n} " + (f"{p['min_joint_margin_deg']}°" if p.get("ok") else str(p.get("reason")))
                         for n, p in cands)
        if not good:
            self.io.emit(f"[WARN] [V3.69 CUP PLAN] no plan ({txt}) in {time.time() - t0:.1f} s")
            return cands[-1][1]
        name, plan = max(good, key=lambda c: c[1]["min_joint_margin_deg"])
        self.io.emit(f"[INFO] [V3.69 CUP PLAN] {txt} -> {name} ({plan['min_joint_margin_deg']}°, grasp "
                     f"{plan['grasp_rim_angle_rel_deg']:+.0f}° place {plan['place_rim_angle_rel_deg']:+.0f}°) in "
                     f"{time.time() - t0:.1f} s (budget {self.prof.plan_time_budget_s:.0f} s)")
        return plan

    # ------------------------------------------------------------------ V3.63 JOINT-SPACE SAFE MOTION
    class _Abort(Exception):
        def __init__(self, reason, info):
            super().__init__(reason)
            self.reason, self.info = reason, dict(info or {})

    def _unlatched(self, label):
        if self.io.latch_pending():
            ok, why = self.io.clear_latch(f"V3.63 {label}")
            if not ok:
                raise self._Abort("SAFETY_LATCH_NOT_CLEARED", {"during": label, "detail": why})

    def _check(self, label, r):
        if r.get("safety") or r.get("safety_trips"):
            raise self._Abort("SAFETY_STOP", dict({"during": label, "error": r.get("error"),
                                                   "safety_trips": r.get("safety_trips")},
                                                  **{k_: r[k_] for k_ in ("trip_reason", "trip_meaning",
                                                                          "closest_joint_limit") if k_ in r}))
        if not r.get("ok"):
            raise self._Abort("MOTION_FAILED", {"during": label, "error": r.get("error"),
                                                "tcp_error_mm": r.get("tcp_error_mm"),
                                                "joint_error_deg": r.get("joint_error_deg")})
        return r

    def _go_joints(self, label, q, slow=None):
        self._unlatched(label)
        self.motions.append((label, "joints"))
        return self._check(label, self.io.move_joints(f"V3.63 {label}", q, slow=slow))

    def _go_line(self, label, xyz, R, speed, slow=None, q_hint=None, q_path=None, free_fallback=False):
        """straight move if the checked Cartesian path is safe.  If it is refused:
          * near the cups (descent / lift / lowering / retreat): follow a STRAIGHT joint path - the planned strict-
            continuity joint solutions along the segment (q_path) or ones solved now from the current joints - in
            short joint moves; if the segment cannot be followed continuously -> TARGET_UNREACHABLE (V3.64: a free
            joint-space descent swept the SO-101 jaw through the cup in the twin)
          * free_fallback (the level carry high above the cups): joint space to a checked IK solution, held-cup dip
            check (V3.63)"""
        self._unlatched(label)
        q_now = self.io.current_q()
        r = self.io.move_line(f"V3.63 {label}", xyz, R_to_quat_xyzw(R), speed, slow=slow)
        if not r.get("refused"):
            self.motions.append((label, "line"))
            return self._check(label, r)
        if not free_fallback:
            path = None
            if q_path and max(abs(a - b) for a, b in zip(q_path[0], q_now)) <= math.radians(3.0):
                path = [list(q) for q in q_path]
                T = self._arm().fk(np.asarray(path[-1], float))
                if np.linalg.norm(T[:3, 3] - np.asarray(xyz, float)) > 0.003:
                    path = None
            if path is None:
                p0 = np.asarray(self.io.tcp_pose()[0], float)
                ch = segment_points("s", p0, xyz, R, 6) + [("~end", np.asarray(xyz, float), R)]
                rr = ik_chain(ch, q_now)
                if not rr["ok"]:
                    raise self._Abort("TARGET_UNREACHABLE", {"during": label, "why": r.get("why"),
                                                             "detail": f"the straight segment cannot be followed: {rr['why']}"})
                path = [q_now] + rr["q"]
            chk = check_joint_path(path, jump_deg=12.0)
            if not chk["ok"]:
                raise self._Abort("TARGET_UNREACHABLE", {"during": label, "why": r.get("why"), "detail": chk["why"]})
            self.io.emit(f"[WARN] [V3.64 CUP] {label}: Cartesian path refused ({r.get('why')}) -> straight segment as "
                         f"{len(path) - 1} short joint moves along strict-continuity IK (min margin {chk['min_margin_deg']} deg)")
            self.motions.append((label, "joints (straight path)"))
            for k_, qk in enumerate(path[1:]):
                r = self.io.move_joints(f"V3.64 {label} (straight {k_ + 1}/{len(path) - 1})", qk, slow=slow or 2.5)
                if not r.get("ok"):
                    break
            return self._check(label, r)
        # V3.64 CARRY BRANCH: candidates = the planned (margin-checked) solution, IK seeded from it (the measured
        # held-cup offset moves the goal a few mm), IK seeded from the current joints; the first with margins wins
        # (V3.64 twin: seeding only from the current joints found a branch with J1 0.5 deg from its limit)
        q_t = None
        cands = []
        if q_hint is not None:
            T = FastArm.fk(self._arm(), np.asarray(q_hint, float))
            if np.linalg.norm(T[:3, 3] - np.asarray(xyz, float)) <= 0.002:
                cands.append(list(q_hint))
            cands.append(ik_from(q_hint, xyz, R))
        cands.append(ik_from(q_now, xyz, R))
        for c_ in cands:
            if c_ is not None and min(joint_margins_deg(c_)) >= PATH_MARGIN_DEG:
                q_t = c_
                break
        if q_t is None:
            raise self._Abort("TARGET_UNREACHABLE", {"during": label, "why": r.get("why"),
                                                     "detail": "no IK solution with joint margins for this pose"})
        self.io.emit(f"[WARN] [V3.63 CUP] {label}: Cartesian path refused ({r.get('why')}) -> joint-space move to "
                     f"a checked IK solution (min margin {min(joint_margins_deg(q_t)):.1f} deg)")
        self.motions.append((label, "joints (line refused)"))
        # a joint-space path is not straight in Cartesian space: while a cup is held the TCP must not dip more than
        # 10 mm below the lower end of the requested segment (split at the line midpoint if it would)
        path = [q_now, q_t]
        if self.held:
            z_floor = min(float(xyz[2]), float(self.io.tcp_pose()[0][2])) - 0.010
            if self._dip_z(q_now, q_t) < z_floor:
                p0 = np.asarray(self.io.tcp_pose()[0], float)
                q_mid = ik_from(q_now, 0.5 * (p0 + np.asarray(xyz, float)), R)
                if q_mid is None or self._dip_z(q_now, q_mid) < z_floor or self._dip_z(q_mid, q_t) < z_floor:
                    raise self._Abort("TARGET_UNREACHABLE", {"during": label, "why": r.get("why"),
                                                             "detail": "joint-space path would dip the held cup"})
                path = [q_now, q_mid, q_t]
        for k_, qk in enumerate(path[1:]):
            r = self.io.move_joints(f"V3.63 {label} (joint space{'' if len(path) == 2 else f' {k_ + 1}/2'})", qk,
                                    slow=slow or 2.5)
            if not r.get("ok"):
                break
        return self._check(label, r)

    def _dip_z(self, qa, qb, n=21):
        arm = self._arm()
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        return min(float(arm.fk(qa + u * (qb - qa))[2, 3]) for u in np.linspace(0.0, 1.0, n))

    def _arm(self):
        if "t" not in _IK:
            _IK["t"] = _ik_tools()
        return _IK["t"][0]

    def _abort_result(self, tool, e, v):
        """V3.63: a motion failure / safety stop ends the tool at once; after a safety stop the latch is cleared
        (backend STOP + RECOVER, hold kept) so that the Agent may retry the same tool ONCE from the current posture"""
        v = dict(v, **e.info)
        nxt = ["observe_scene", tool]
        if e.reason == "SAFETY_STOP":
            ok, why = self.io.clear_latch(f"V3.63 {tool} after safety stop")
            v.update({"latch_recovered": bool(ok), "recover_detail": why,
                      "hold_active": self.io.gripper_state().get("hold_active")})
            self.safety_stops[tool] = self.safety_stops.get(tool, 0) + 1
            if not ok or self.safety_stops[tool] >= 2:
                nxt = ["stop_and_report"]
        elif e.reason in ("SAFETY_LATCH_NOT_CLEARED", "TARGET_UNREACHABLE"):
            nxt = ["stop_and_report"]
        if (self.prof.recover_after_motion_failure and e.reason in ("MOTION_FAILED", "TARGET_UNREACHABLE")
                and hasattr(self.io, "recover_after_failure")):
            # V3.71 RECOVER: stop, then straight up to a safe height before the agent decides (V3.70: the next task
            # started next to the cups, its first joint path clipped a fence wall)
            try:
                rr = self.io.recover_after_failure(f"{tool}: {e.info.get('during') or e.reason}")
                v["recovered"] = bool(rr.get("ok"))
            except Exception as exc:
                v["recovered"] = False
                v["recover_error"] = str(exc)[:200]
        return self._res(tool, False, v, reason=e.reason, nxt=nxt)

    def pick_cup(self, source_cup="cup_1", **_):
        tool = "pick_cup"
        if self.held:
            return self._res(tool, False, reason="ALREADY_HOLDING", nxt=["place_cup_into"])
        if not self.plan or self.plan["source"] != source_cup or self.stage != "PLANNED":
            return self._res(tool, False, reason="PLAN_REQUIRED", nxt=["plan_cup_stack"])
        W, Q = self.plan["_wps"], self.plan["_q"]
        v = {}
        try:
            if self.camera:
                self._perceive(f"pick {source_cup}")
            s0 = self._cup(source_cup)
            s0_fixed = {"rim": np.asarray(s0["rim"], float).copy(), "base": np.asarray(s0["base"], float).copy(),
                        "tilt_deg": s0["tilt_deg"], "height_mm": s0.get("height_mm")}
            v["opening_before_mm"] = self.io.gripper_open(self.prof.open_mm)["opening_mm"]
            if self.prof.align_rotate_m > 0:
                v["align_rotate"] = self._align_rotate(source_cup, W["pregrasp"], Q["pregrasp"])
            else:
                # joint goal to the planned (margin-checked) IK branch - not a free pose goal
                self._go_joints(f"pregrasp {source_cup}", Q["pregrasp"])
            if self.camera and hasattr(self.io, "wrist_snapshot"):
                w_chk = self._wrist_recheck_grasp(source_cup, s0, W, Q)
                if w_chk is not None:
                    v["wrist_grasp_check"] = w_chk
            p, R = W["grasp"]
            if self.prof.descent_stop_above_rim_m is not None:
                stop = self._descent_stop(source_cup, s0_fixed, W)
                if stop is not None:
                    v["pre_contact_check"] = stop
                    if stop.get("abort"):
                        self._retreat_up()
                        return self._res(tool, False, v, reason="CUP_MOVED_BEFORE_CONTACT",
                                         nxt=["observe_scene", "plan_cup_stack"])
            self._go_line(f"grasp {source_cup}", p, R, 0.03, q_hint=Q["grasp"], q_path=self.plan["_qpath"]["grasp"])
            if self.camera:
                # the cup is still on the table here: its rim centre in the TCP frame = the grasp offset (V3.63)
                tcp_g, qg = self.io.tcp_pose()
                self.cam_grip_offset = quat_xyzw_to_R(qg).T @ (np.asarray(s0["rim"], float) - np.asarray(tcp_g, float))
                self._perceive(f"check {source_cup} after the descent")
            s1 = self._cup(source_cup)
            moved = 0.0 if s1.get("stale") else float(np.linalg.norm(s1["base"][:2] - s0_fixed["base"][:2])) * 1000
            v["cup_moved_by_descent_mm"] = round(moved, 1)
            if s1.get("stale"):
                v["descent_check"] = "cup hidden by the gripper in the fixed camera view - not checked"
            if self.prof.lenient_pick:
                # V3.68 LENIENT REAL PICK: only a FRESH view of a cup clearly pushed away / knocked over stops the pick;
                # the tilt is compared with the one before the descent (V3.67 compared a stale absolute tilt)
                d_tilt = 0.0 if s1.get("stale") else abs(float(s1["tilt_deg"]) - float(s0["tilt_deg"]))
                v["cup_tilt_change_deg"] = round(d_tilt, 1)
                if moved > LENIENT_ABORT_MM or d_tilt > LENIENT_TILT_DEG:
                    self._retreat_up()
                    why = (f"cup moved {moved:.0f} mm (> {LENIENT_ABORT_MM:.0f})" if moved > LENIENT_ABORT_MM else
                           f"cup tilt changed {d_tilt:.0f} deg (> {LENIENT_TILT_DEG:.0f})")
                    v["detail"] = why
                    return self._res(tool, False, v, reason="CUP_PUSHED_DURING_DESCENT",
                                     nxt=["observe_scene", "plan_cup_stack"])
            elif moved > 8.0 or s1["tilt_deg"] > 6.0:
                self._retreat_up()
                return self._res(tool, False, v, reason="CUP_PUSHED_DURING_DESCENT", nxt=["observe_scene", "plan_cup_stack"])
            c = self.io.gripper_clamp(float(self.k["cup_clamp_force_n"]))
            v["clamp"] = c
            if self.camera and self.prof.approach_clearance_m > 0:
                # the clamp slid the cup onto the fixed jaw: -clearance along the grasp y axis (TCP frame)
                self.cam_grip_offset = self.cam_grip_offset - self.prof.approach_clearance_m * self.prof.T_tcp_grasp[:3, 1]
            clamp_held = bool((c["contact_left"] and c["contact_right"]) or c["hold_active"])
            if not clamp_held and not self.prof.lenient_pick:
                self.io.gripper_open(self.prof.open_mm)
                self._retreat_up()
                return self._res(tool, False, v, reason="NO_TWO_FINGER_CONTACT", nxt=["observe_scene", "plan_cup_stack"])
            tcp0, _ = self.io.tcp_pose()
            self.grip_offset = None
            p, R = W["lift"]
            self._go_line(f"lift {source_cup}", p, R, 0.05, q_hint=Q["lift"], q_path=self.plan["_qpath"]["lift"])
            if self.prof.lift_settle_s > 0:
                self.io.sleep(self.prof.lift_settle_s)
        except self._Abort as e:
            return self._abort_result(tool, e, v)
        tcp1, q1 = self.io.tcp_pose()
        rise_tcp = tcp1[2] - tcp0[2]
        if self.camera:
            try:
                self._perceive(f"check {source_cup} lifted")
            except self._Abort as e:
                return self._abort_result(tool, e, v)
            still = self.cam.get(source_cup)
            h_before_mm = float(s0_fixed.get("height_mm") or s1.get("height_mm") or (geom()["h"] * 1000.0))
            min_dz_lift_mm = 0.5 * max(rise_tcp * 1000.0, 40.0)
            still_h_mm = float((still or {}).get("height_mm") or 0.0) if still else 0.0
            rose_by_depth = (still_h_mm - h_before_mm) >= min_dz_lift_mm
            on_table = bool(still and not still.get("stale") and not rose_by_depth
                            and np.linalg.norm(still["base"][:2] - s0_fixed["base"][:2]) < 0.02)
            if not on_table and not clamp_held and not rose_by_depth:
                raw_cups = [c_ for c_ in (getattr(self, "cam_raw", None) or {}).get("cups") or []
                            if c_.get("kind") == "cup" and c_.get("base_xyz_m")]
                has_lifted = any(c_.get("kind") == "lifted" for c_ in (getattr(self, "cam_raw", None) or {}).get("dets") or [])
                if len(raw_cups) >= 2 and not has_lifted:
                    c_pushed = min(raw_cups, key=lambda c_: float(np.linalg.norm(np.asarray(c_["base_xyz_m"][:2], float) - s0_fixed["base"][:2])))
                    c_h_after = float(c_pushed.get("height_mm") or c_pushed.get("height_depth_mm") or 0.0)
                    if (c_h_after - h_before_mm) < min_dz_lift_mm and float(np.linalg.norm(np.asarray(c_pushed["base_xyz_m"][:2], float) - s0_fixed["base"][:2])) < 0.12:
                        on_table = True
                        b_p = np.asarray(c_pushed["base_xyz_m"], float)
                        self.cam[source_cup] = self._entry(b_p, float(c_pushed.get("tilt_deg") or 0.0),
                                                           {"height_mm": c_pushed.get("height_mm") or c_pushed.get("height_depth_mm")})
            rise_cup = 0.0 if on_table else rise_tcp
            s2 = {"rim": np.asarray(tcp1, float) + quat_xyzw_to_R(q1) @ self.cam_grip_offset, "tilt_deg": 0.0}
            v["lift_check"] = "fixed camera: cup no longer at its table position" if not on_table else \
                "fixed camera: the cup is still standing at its table position"
        else:
            s2 = self._cup(source_cup)
            rise_cup = float(s2["base"][2] - s1["base"][2])
        v.update({"lift_tcp_rise_mm": round(rise_tcp * 1000, 1), "cup_rise_mm": round(rise_cup * 1000, 1),
                  "cup_tilt_after_lift_deg": s2["tilt_deg"], "hold_active": self.io.gripper_state()["hold_active"]})
        if rise_cup < 0.7 * max(rise_tcp, 0.01):
            self.io.gripper_open(self.prof.open_mm)
            self._retreat_up()
            return self._res(tool, False, v, reason="CUP_DID_NOT_FOLLOW_LIFT", nxt=["observe_scene", "plan_cup_stack"])
        no_hold = self.prof.lenient_pick and not v.get("clamp", {}).get("hold_active", True) \
            and not (v.get("clamp", {}).get("contact_left") and v.get("clamp", {}).get("contact_right"))
        if no_hold and self.camera and self.prof.camera_decides_grip:
            # V3.69 CAMERA DECIDES GRIP: the cup left the table with the gripper -> held; the servo angle only warns
            v["clamp_note"] = "servo stopped near the touch point (thin cup wall) - the camera confirms the cup is held"
            self.io.emit("[WARN] [V3.69 CUP] 夹爪停在碰合点附近（薄杯壁分不清），主相机确认杯子已离开桌面 → 以相机为准，算夹住")
            no_hold = False
        if no_hold:
            # V3.68 LENIENT REAL PICK: lifted although the clamp did not report a hold -> both checks must agree
            self.io.gripper_open(self.prof.open_mm)
            self._retreat_up()
            v["detail"] = "the cup left the table but the gripper did not stop on it (clamp closed to the touch point)"
            return self._res(tool, False, v, reason="NO_TWO_FINGER_CONTACT", nxt=["observe_scene", "plan_cup_stack"])
        # held cup rim centre in the TCP frame (the backend hold is a CONNECT at this point -> rigid while held)
        Rt = quat_xyzw_to_R(q1)
        self.grip_offset = (self.cam_grip_offset if self.camera else
                            Rt.T @ (np.asarray(s2["rim"], float) - np.asarray(tcp1, float)))
        self.held = source_cup
        self.stage = "CUP_LIFTED"
        warn = []
        if s2["tilt_deg"] > float(self.k["cup_release_tilt_tol_deg"]):
            warn.append(f"held cup tilted {s2['tilt_deg']} deg (> {self.k['cup_release_tilt_tol_deg']} deg tolerance)")
        return self._res(tool, True, v, nxt=["place_cup_into"], warnings=warn)

    # ------------------------------------------------------------------ V3.67 ALIGN-ROTATE-DESCEND
    def _align_rotate(self, cup, wp_pre, q_pre):
        """arrive above the pregrasp keeping the arm's current wrist yaw, rotate in place to the grasp angle (jaw line
        along the cup radius), go straight down to the pregrasp.  Falls back to a smaller / no rotation when the
        rotated pose has no IK solution with margins.  -> dict"""
        p_pre, R_g = np.asarray(wp_pre[0], float), np.asarray(wp_pre[1], float)
        p_al = p_pre + np.array([0.0, 0.0, self.prof.align_rotate_m])
        _p, qx = self.io.tcp_pose()
        M = quat_xyzw_to_R(qx) @ R_g.T
        d = math.atan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])            # best rotation about the vertical
        d = max(-math.radians(ALIGN_MAX_DEG), min(math.radians(ALIGN_MAX_DEG), d))
        tries = [d, 0.5 * d, 0.0] if abs(math.degrees(d)) >= ALIGN_MIN_DEG else [0.0]
        q_now = self.io.current_q()
        chosen = None
        for dd in tries:
            c, s_ = math.cos(dd), math.sin(dd)
            R_al = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]) @ R_g
            q_al = None
            for seed in (q_pre, q_now):
                cand = ik_from(seed, p_al, R_al)
                if cand is not None and min(joint_margins_deg(cand)) >= PATH_MARGIN_DEG:
                    q_al = cand
                    break
            if q_al is None:
                continue
            q_rot = q_al if dd == 0.0 else ik_from(q_al, p_al, R_g)
            if q_rot is None:
                continue
            n = 12
            path = [list(np.asarray(q_al) + (np.asarray(q_rot) - np.asarray(q_al)) * i / n) for i in range(n + 1)]
            chk = check_joint_path(path, jump_deg=12.0)
            dev = max(float(np.linalg.norm(self._arm().fk(np.asarray(q, float))[:3, 3] - p_al)) for q in path)
            if chk["ok"] and dev <= ALIGN_MAX_TCP_DEV_M:
                chosen = (dd, q_al, q_rot, dev)
                break
        if chosen is None:
            self.io.emit(f"[WARN] [V3.67 CUP] align above {cup}: no IK with margins for the rotated arrival -> straight to "
                         "the pregrasp")
            self._go_joints(f"pregrasp {cup}", q_pre)
            return {"rotated_deg": 0.0, "fallback": True}
        dd, q_al, q_rot, dev = chosen
        self._go_joints(f"align above {cup}", q_al)
        if dd != 0.0:
            self.io.emit(f"[INFO] [V3.67 CUP] rotate in place to the grasp angle: {math.degrees(dd):+.0f} deg about the "
                         f"vertical (TCP stays within {dev * 1000:.0f} mm)")
            self._go_joints(f"rotate to the grasp angle above {cup}", q_rot, slow=2.0)
        self._go_line(f"down to the pregrasp above {cup}", p_pre, R_g, 0.03, q_hint=q_pre)
        return {"rotated_deg": round(math.degrees(dd), 1), "align_height_mm": round(self.prof.align_rotate_m * 1000)}

    def _tip_z(self, tcp, R):
        """lowest fingertip height for a TCP pose: grasp-frame origin + 10 mm along the approach axis"""
        Tg = self.prof.T_tcp_grasp
        Rg = np.asarray(R, float) @ Tg[:3, :3]
        return float((np.asarray(tcp, float) + np.asarray(R, float) @ Tg[:3, 3] + Rg[:, 2] * TIP_BEYOND_GRASP_M)[2])

    def _descent_stop(self, cup, s0, W):
        """V3.67 TWO-STAGE DESCENT: stop with the fingertips descent_stop_above_rim_m above the rim, re-check the cup
        (fixed camera): unchanged -> continue; moved -> abort before any contact.  -> dict or None"""
        p_pre, R = np.asarray(W["pregrasp"][0], float), np.asarray(W["pregrasp"][1], float)
        p_g = np.asarray(W["grasp"][0], float)
        z_pre, z_g = self._tip_z(p_pre, R), self._tip_z(p_g, R)
        z_stop = float(np.asarray(s0["rim"], float)[2]) + self.prof.descent_stop_above_rim_m
        if not (z_g < z_stop < z_pre):
            return None
        f = (z_stop - z_g) / (z_pre - z_g)
        p_stop = p_g + f * (p_pre - p_g)
        self._go_line(f"stop above the rim of {cup}", p_stop, R, 0.03)
        out = {"tips_above_rim_mm": round(self.prof.descent_stop_above_rim_m * 1000)}
        if not self.camera:
            return out
        self._perceive(f"check {cup} before contact")
        s_ = self.cam.get(cup)
        base = None if (not s_ or s_.get("stale")) else np.asarray(s_["base"][:2], float)
        if base is None:
            # partly hidden by the gripper: a camera that reports the cup's position only (kind "occluded") still counts
            near = [np.asarray(c["base_xyz_m"][:2], float) for c in (getattr(self, "cam_raw", None) or {}).get("cups", [])
                    if c.get("kind") == "occluded" and c.get("base_xyz_m")
                    and np.linalg.norm(np.asarray(c["base_xyz_m"][:2]) - np.asarray(s0["base"][:2])) < 0.03]
            if not near:
                out["check"] = "cup hidden by the gripper in the fixed camera view - not checked"
                return out
            base = near[0]
            out["check"] = "cup partly hidden by the gripper: position from the lower wall"
        off = float(np.linalg.norm(base - np.asarray(s0["base"][:2]))) * 1000.0
        out["cup_offset_mm"] = round(off, 1)
        lim = LENIENT_ABORT_MM if self.prof.lenient_pick else PRE_CONTACT_MOVE_MM      # V3.68 LENIENT REAL PICK
        if off > lim:
            out["abort"] = True
            self.io.emit(f"[WARN] [V3.67 CUP] {cup} is {off:.1f} mm away from where the grasp was planned -> no descent "
                         "(re-observe and re-plan)")
        elif self.prof.lenient_pick and off > PRE_CONTACT_MOVE_MM:
            self.io.emit(f"[INFO] [V3.68 CUP] {cup} estimate {off:.1f} mm from the plan (camera view partly hidden) -> "
                         "continue, the lift decides")
        return out

    def _retreat_up(self, dz=0.08):
        try:
            p, q = self.io.tcp_pose()
            self._go_line("retreat", [p[0], p[1], p[2] + dz], quat_xyzw_to_R(q), 0.05)
        except Exception as exc:
            self.io.emit(f"[WARN] [V3.63 CUP] retreat failed: {exc}")

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
        try:
            if self.camera:
                self._perceive(f"place {src} into {target_cup}")      # fresh target pose (held cup: TCP + offset)
            R_p = W["above_target"][1]
            q_now = self.io.current_q()
            tcp, _q = self.io.tcp_pose()
            # 1. rotate in place (joint space, slow): lift position, place orientation; re-solved from the CURRENT
            #    joints so a retry after a safety stop starts from where the arm really is
            q_rot = ik_from(q_now, tcp, R_p)
            if q_rot is None or min(joint_margins_deg(q_rot)) < MIN_MARGIN_RELAXED_DEG:
                q_rot = self.plan["_q"]["rotate"]
            rot_done = False
            try:
                self._go_joints(f"rotate {src} to the place orientation", q_rot, slow=CARRY_SLOW_FACTOR)
                rot_done = True
            except self._Abort as exc_rot:
                if "FENCE" not in str(exc_rot.info.get("error") or "") and exc_rot.reason != "MOTION_FAILED":
                    raise
                self.io.emit(f"[WARN] [V3.71 CUP] 在 {src} 上方原地转腕擦到围栏 → 改为先平移向 {target_cup} 再转腕")
            # 2. level translation; the measured held-cup offset is folded into the target (no separate centring)
            d, s, t = self._centre_error(src, target_cup)
            tcp, _q = self.io.tcp_pose()
            above = np.array([tcp[0] + d[0], tcp[1] + d[1], tcp[2]])
            if not rot_done:
                q_ab = ik_from(self.io.current_q(), above, R_p)
                if q_ab is None or min(joint_margins_deg(q_ab)) < MIN_MARGIN_RELAXED_DEG:
                    q_ab = self.plan["_q"].get("above_target")
                self._go_joints(f"carry & rotate {src} over {target_cup}", q_ab, slow=CARRY_SLOW_FACTOR)
            else:
                self._go_line(f"carry {src} over {target_cup}", above, R_p, CARRY_SPEED, slow=CARRY_SLOW_FACTOR,
                              free_fallback=True, q_hint=self.plan["_q"].get("above_target"))
            if self.prof.lift_settle_s > 0:                # V3.64 LIFT SETTLE: the carry swings the cup on the hold too
                self.io.sleep(self.prof.lift_settle_s)
            # V4.6.9 TARGET PINNED: the target cup was measured while nothing blocked it; from here the held cup hangs
            # above it in the fixed camera's view - those views must not move the target (V4.6.8 run: the target was
            # re-associated to another blob 98 mm away and the place was "corrected" by ~20 mm)
            t_pin = {k_: np.asarray(self._cup(target_cup)[k_], float).copy() for k_ in ("base", "rim")}
            if self.camera and self.prof.recheck_held_before_place:
                v["held_recheck"] = self._recheck_held(src, target_cup, R_p)
                self._pin_target(target_cup, t_pin)
            if self.camera and hasattr(self.io, "wrist_snapshot"):
                w_pl = self._wrist_recheck_place(src, target_cup, R_p)
                if w_pl is not None:
                    v["wrist_place_check"] = w_pl
                self._pin_target(target_cup, t_pin)
            # 3. lower to the release height; the remaining error (if >= 3 mm) is folded into the same move
            d, s, t = self._centre_error(src, target_cup)
            v["centre_error_high_mm"] = round(float(np.linalg.norm(d)) * 1000, 1)
            if float(np.linalg.norm(d)) < SMALL_CORRECTION_M:
                d = np.zeros(2)
            _, base_rel = stack_heights(self.k, s["base"][2], t["base"][2])
            tcp, _q = self.io.tcp_pose()
            dz = float(s["base"][2] - base_rel)
            self._go_line(f"lower {src} into {target_cup}", [tcp[0] + d[0], tcp[1] + d[1], tcp[2] - dz], R_p, PLACE_SPEED)
            d, s, t = self._centre_error(src, target_cup)
            if float(np.linalg.norm(d)) > tol:            # at most one correction at the release height
                tcp, _q = self.io.tcp_pose()
                self._go_line("centre at release height", [tcp[0] + d[0], tcp[1] + d[1], tcp[2]], R_p, PLACE_SPEED)
                d, s, t = self._centre_error(src, target_cup)
        except self._Abort as e:
            return self._abort_result(tool, e, v)
        v.update({"centre_error_mm": round(float(np.linalg.norm(d)) * 1000, 1),
                  "held_cup_tilt_deg": s["tilt_deg"], "target_cup_tilt_deg": t["tilt_deg"],
                  "base_above_target_base_mm": round(float(s["base"][2] - t["base"][2]) * 1000, 1)})
        if float(np.linalg.norm(d)) > tol:
            return self._res(tool, False, v, reason="NOT_CENTRED_OVER_TARGET", nxt=["place_cup_into", "stop_and_report"])
        if s["tilt_deg"] > max(float(self.k["cup_release_tilt_tol_deg"]) * 2.0, self.prof.release_tilt_max_deg):
            return self._res(tool, False, v, reason="HELD_CUP_TOO_TILTED", nxt=["stop_and_report"])
        warn = []
        if s["tilt_deg"] > float(self.k["cup_release_tilt_tol_deg"]):
            warn.append(f"held cup tilted {s['tilt_deg']} deg (> {self.k['cup_release_tilt_tol_deg']} deg); released anyway "
                        "(the V3.61 release sweep succeeded up to 6 deg)")
        self.io.gripper_open(self.prof.open_mm)
        if self.prof.approach_clearance_m > 0:
            self._release_sidestep()
        self.io.sleep(1.5)
        self.held = None
        self.grip_offset = None
        self.placed_xy = [float(t["base"][0]), float(t["base"][1])]
        self.placed = (src, target_cup)
        self.stage = "RELEASED"
        self._retreat_up()
        st = self._state(src, target_cup)
        v["after_release"] = st
        return self._res(tool, True, v, nxt=["verify_stack"], warnings=warn)

    def _held_measured(self, src, tag):
        """V3.71 HELD CUP RECHECK: fixed-camera measurement of the cup in the jaws (NERO_V366 'lifted' entry with the
        cone axis of its lower wall) -> (base x-y, tilt or None, entry) or (None, None, None)"""
        self._perceive(tag)
        est = self._cup(src)["base"]
        best, dmin = None, 1e9
        for e in (getattr(self, "cam_raw", None) or {}).get("dets") or []:
            hb = e.get("held_base_xyz_m")
            if e.get("kind") != "lifted" or not hb:
                continue
            d = float(np.hypot(hb[0] - est[0], hb[1] - est[1]))
            if d < dmin:
                best, dmin = e, d
        if best is None:
            return None, None, None
        tilt = best.get("held_tilt_deg") if best.get("held_tilt_reliable") else None
        return np.asarray(best["held_base_xyz_m"][:2], float), tilt, best

    def _recheck_held(self, src, dst, R_p, max_corr_m=0.025, n_max=2):
        """V3.71 HELD CUP RECHECK: above the target (both cups visible, the held one clear of the target rim) measure
        the held cup, fold the difference into the grasp offset, centre over the target, measure once more"""
        out = {"checks": []}
        for k_ in range(n_max):
            meas, tilt, e = self._held_measured(src, f"recheck held {src} above {dst} ({k_ + 1})")
            if meas is None:
                out["note"] = "夹着的杯子没测到（被挡住 / 点太少）→ 沿用夹取时的偏移"
                self.io.emit(f"[WARN] [V3.71 CUP] 放置前复查：{out['note']}")
                break
            est = np.asarray(self._cup(src)["base"][:2], float)
            d = meas - est
            row = {"offset_mm": round(float(np.linalg.norm(d)) * 1000.0, 1), "tilt_deg": tilt,
                   "fit_rms_mm": e.get("held_axis_rms_mm"), "points": e.get("held_n_wall")}
            if float(np.linalg.norm(d)) > max_corr_m:
                row["applied"] = False
                out["checks"].append(row)
                self.io.emit(f"[WARN] [V3.71 CUP] 放置前复查：测得夹着的杯子偏 {row['offset_mm']} mm（> "
                             f"{max_corr_m * 1000:.0f} mm，不可信）→ 不修正，沿用夹取时的偏移")
                break
            tcp, qt = self.io.tcp_pose()
            Rt = quat_xyzw_to_R(qt)
            self.grip_offset = self.grip_offset + Rt.T @ np.array([d[0], d[1], 0.0])
            row["applied"] = True
            out["checks"].append(row)
            dd, _s, _t = self._centre_error(src, dst)
            self.io.emit(f"[INFO] [V3.71 CUP] 放置前复查 {k_ + 1}：夹着的 {src} 比夹取时的估计偏 {row['offset_mm']} mm"
                         + (f"，倾斜 {tilt:.1f}°" if tilt is not None else "") + f" → 偏移已修正；离 {dst} 中心 "
                         f"{float(np.linalg.norm(dd)) * 1000:.1f} mm")
            if tilt is not None:
                out["held_tilt_deg"] = round(float(tilt), 1)
            if float(np.linalg.norm(dd)) < SMALL_CORRECTION_M:
                break
            tcp, _q = self.io.tcp_pose()
            self._go_line(f"centre {src} over {dst} (V3.71 vision)", [tcp[0] + dd[0], tcp[1] + dd[1], tcp[2]], R_p,
                          PLACE_SPEED)
        return out

    def _wrist_recheck_grasp(self, cup, s0, W, Q):
        """V3.87 PHASE 3 PRE-GRASP WRIST VISUAL SERVOING: at pregrasp, use the wrist camera to verify the cup rim
        centre and the perpendicularity of the gripper closing line relative to the rim tangent, adjusting horizontal
        position and wrist angle before descending."""
        try:
            wsnap = self.io.wrist_snapshot()
        except Exception:
            wsnap = None
        if not wsnap or wsnap.get("rgb") is None or not wsnap.get("K") or not wsnap.get("calib_eih"):
            return None
        import NERO_V386_Real_Wrist_Camera as W86
        q_now = self.io.current_q()
        tcp_p, tcp_q = self.io.tcp_pose()
        T_bt = np.eye(4)
        T_bt[:3, :3], T_bt[:3, 3] = quat_xyzw_to_R(tcp_q), np.asarray(tcp_p, float)
        T_bf = T_bt @ np.linalg.inv(self._arm().tool)
        phi_plan = float(self.plan.get("grasp_phi_rad") or math.radians(
            math.degrees(base_dir(s0["rim"])) + float(self.plan.get("grasp_rim_angle_rel_deg") or 0.0)
        ))
        flip = bool(self.plan.get("flip", True))
        g = geom()
        h_m = (float(s0["height_mm"]) / 1000.0) if (s0.get("height_mm") and 50.0 <= float(s0["height_mm"]) <= 150.0) else float(s0["rim"][2] - s0["base"][2])
        exp_rim_opt = [float(s0["rim"][0]), float(s0["rim"][1]), float(s0["base"][2]) + h_m]
        res = W86.wrist_check_cup_grasp(
            wsnap["rgb"], wsnap["K"], T_bf.tolist(), wsnap["calib_eih"], self.prof.T_tcp_grasp, T_bt,
            exp_rim_opt, phi_plan, flip=flip, r_top_m=g["r_top"], undistort_intr=wsnap.get("intr")
        )
        if res.get("measured_rim_xyz"):
            res["measured_rim_xyz"][2] = round(float(s0["rim"][2]), 5)
        if not res.get("ok"):
            self.io.emit(f"[WARN] [V3.87 WRIST] 抓杯前腕部复查未生效：{res.get('reason')}")
            return res
        if not res.get("trusted"):
            self.io.emit(f"[WARN] [V3.87 WRIST] 抓杯前腕部复查偏差 {res['shift_norm_mm']:.1f} mm 超出可信阈值 → 沿用主相机规划")
            return res
        shift_mm = res["shift_norm_mm"]
        ang_err = res["angle_err_deg"]
        need_shift = shift_mm >= W86.WRIST_GRASP_MIN_SHIFT_M * 1000.0
        need_rot = abs(ang_err) >= W86.WRIST_GRASP_MIN_ROT_DEG and abs(ang_err) <= W86.WRIST_GRASP_MAX_ROT_DEG
        if not need_shift and not need_rot:
            self.io.emit(f"[OK] [V3.87 WRIST] 抓杯前腕部复查：{cup} 中心偏差 {shift_mm:.1f} mm · 夹爪垂直度误差 {ang_err:+.1f}°（已对齐 ✓）")
            res["applied"] = False
            return res

        new_rim = np.asarray(res["measured_rim_xyz"], float) if need_shift else np.asarray(s0["rim"], float)
        new_phi = float(res["recommended_phi_rad"]) if need_rot else phi_plan
        depth = float(self.k["cup_rim_pinch_depth_mm"]) / 1000.0
        pre = float(self.k["cup_pregrasp_height_mm"]) / 1000.0
        tilt = bool(self.k.get("cup_wall_tilt_follow", True))
        p_new, R_new = pinch_tcp(new_rim, new_phi, depth, flip, tilt, self.prof)
        pp_new = p_new + self.prof.pre_offset(R_new, pre)
        dz_lift = float(W["lift"][0][2] - W["grasp"][0][2])
        ch = ([("pregrasp", pp_new, R_new)] + segment_points("d", pp_new, p_new, R_new) + [("grasp", p_new, R_new)]
              + segment_points("l", p_new, p_new + [0, 0, dz_lift], R_new) + [("lift", p_new + [0, 0, dz_lift], R_new)])
        rk = ik_chain(ch, q_now, start_first=True)
        if not rk["ok"] or rk["min_margin_deg"] < MIN_MARGIN_RELAXED_DEG:
            self.io.emit(f"[WARN] [V3.87 WRIST] 抓杯前腕部微调姿态 IK 余量不足（{rk.get('min_margin_deg')}°）→ 保持原轨迹")
            res["applied"] = False
            return res
        W["pregrasp"] = (np.asarray(pp_new, float), np.asarray(R_new, float))
        W["grasp"] = (np.asarray(p_new, float), np.asarray(R_new, float))
        W["lift"] = (np.asarray(p_new + [0, 0, dz_lift], float), np.asarray(R_new, float))
        Q["pregrasp"] = list(rk["q"][0])
        Q["grasp"] = list(rk["q"][5])
        Q["lift"] = list(rk["q"][10])
        self.plan["_qpath"]["grasp"] = [list(q) for q in rk["q"][0:6]]
        self.plan["_qpath"]["lift"] = [list(q) for q in rk["q"][5:11]]
        s0["rim"] = new_rim.copy()
        s0["base"] = np.array([new_rim[0], new_rim[1], s0["base"][2]], dtype=float)
        try:
            self._go_joints(f"wrist align grasp angle & rim on {cup}", Q["pregrasp"], slow=1.8)
        except self._Abort as exc_w:
            self.io.emit(f"[WARN] [V3.87 WRIST] 抓杯前腕部微调运动未执行（{exc_w.reason}）→ 继续下探")
            res["applied"] = False
            return res
        res["applied"] = True
        self.io.emit(f"[OK] [V3.87 WRIST] 抓杯前腕部闭环微调：修正杯沿偏移 ({res['delta_rim_mm'][0]:+.1f}, {res['delta_rim_mm'][1]:+.1f}) mm · "
                     f"夹爪旋转 {ang_err:+.1f}°（开合线垂直于杯壁 ✓）")
        return res

    def _pin_target(self, dst, pin):
        """V4.6.9: restore the target cup position measured before the carry (unoccluded)"""
        c = self._cup(dst)
        moved = float(np.linalg.norm(np.asarray(c["base"][:2], float) - pin["base"][:2])) * 1000.0
        if moved > 0.5:
            self.io.emit(f"[INFO] [V4.6.9 CUP] 目标 {dst} 在放置前的画面里被挡住，读数漂了 {moved:.0f} mm → 沿用搬运前测得的位置")
        c["base"][:] = pin["base"]
        c["rim"][:] = pin["rim"]

    def _wrist_recheck_place(self, src, dst, R_p, n_max=2):
        """V3.87 PHASE 3 PRE-PLACE WRIST CONCENTRICITY SERVOING: above the target cup, use the downward-looking wrist
        camera to measure the concentricity offset between the target cup rim and the held cup bottom, centring before
        lowering into the target cup."""
        if not hasattr(self.io, "wrist_snapshot"):
            return None
        import NERO_V386_Real_Wrist_Camera as W86
        g = geom()
        out = {"checks": []}
        for k_ in range(n_max):
            try:
                wsnap = self.io.wrist_snapshot()
            except Exception:
                wsnap = None
            if not wsnap or wsnap.get("rgb") is None or not wsnap.get("K") or not wsnap.get("calib_eih"):
                break
            tcp_p, tcp_q = self.io.tcp_pose()
            T_bt = np.eye(4)
            T_bt[:3, :3], T_bt[:3, 3] = quat_xyzw_to_R(tcp_q), np.asarray(tcp_p, float)
            T_bf = T_bt @ np.linalg.inv(self._arm().tool)
            s_cup = self._cup(src)
            t_cup = self._cup(dst)
            ht_m = (float(t_cup["height_mm"]) / 1000.0) if (t_cup.get("height_mm") and 50.0 <= float(t_cup["height_mm"]) <= 150.0) else float(t_cup["rim"][2] - t_cup["base"][2])
            tgt_rim_opt = [float(t_cup["rim"][0]), float(t_cup["rim"][1]), float(t_cup["base"][2]) + ht_m]
            res = W86.wrist_check_cup_concentricity(
                wsnap["rgb"], wsnap["K"], T_bf.tolist(), wsnap["calib_eih"],
                tgt_rim_opt, s_cup["base"], r_top_m=g["r_top"], r_bot_m=g["r_bot"],
                undistort_intr=wsnap.get("intr")
            )
            if not res.get("ok"):
                out["note"] = res.get("reason")
                break
            if not res.get("trusted"):
                res["applied"] = False
                out["checks"].append(res)
                self.io.emit(f"[WARN] [V3.87 WRIST] 叠杯前腕部同心度复查：偏差 {res['concentric_err_mm']:.1f} mm 超出阈值 → 不修正")
                break
            # V4.6.9: the target position is NOT overwritten by the wrist reading (V4.6.8: two wrist reads 22 mm apart
            # moved the target back and forth); the reading only moves the TCP, within the caps below
            # If inner contour measured the held cup directly, fold difference into grip_offset
            if res.get("held_source") == "wrist_inner_contour" and self.grip_offset is not None:
                hx, hy = res["held_xy_m"]
                est_b = s_cup["base"]
                dh = np.array([hx - est_b[0], hy - est_b[1], 0.0], dtype=float)
                _tcp, qt = self.io.tcp_pose()
                Rt = quat_xyzw_to_R(qt)
                self.grip_offset = self.grip_offset + Rt.T @ dh
            dx_m, dy_m = res["delta_xy_m"]
            err_mm = res["concentric_err_mm"]
            step_mm = float(math.hypot(dx_m, dy_m)) * 1000.0
            applied_mm = sum(float(math.hypot(*c_["delta_xy_m"])) * 1000.0 for c_ in out["checks"] if c_.get("applied"))
            if step_mm > WRIST_PLACE_STEP_MAX_MM or applied_mm + step_mm > WRIST_PLACE_TOTAL_MAX_MM:
                res["applied"] = False
                out["checks"].append(res)
                self.io.emit(f"[WARN] [V4.6.9 WRIST] 叠杯前腕部同心度读数 {step_mm:.1f} mm（已修正 {applied_mm:.1f} mm）超过上限 "
                             f"{WRIST_PLACE_STEP_MAX_MM:.0f} / {WRIST_PLACE_TOTAL_MAX_MM:.0f} mm → 不修正，按搬运前的目标位置下放")
                break
            if err_mm < W86.WRIST_PLACE_TOL_M * 1000.0:
                res["applied"] = False
                out["checks"].append(res)
                self.io.emit(f"[OK] [V3.87 WRIST] 叠杯前腕部同心度复查 {k_ + 1}：上下双杯同心偏差 {err_mm:.1f} mm（已精确对准 ✓）")
                break
            tcp, _q = self.io.tcp_pose()
            try:
                self._go_line(f"wrist concentricity align {src} over {dst} ({k_ + 1})",
                              [tcp[0] + dx_m, tcp[1] + dy_m, tcp[2]], R_p, PLACE_SPEED)
            except self._Abort as exc_w:
                self.io.emit(f"[WARN] [V3.87 WRIST] 叠杯前腕部同心微调运动未执行（{exc_w.reason}）→ 保持当前位置下放")
                res["applied"] = False
                out["checks"].append(res)
                break
            res["applied"] = True
            out["checks"].append(res)
            self.io.emit(f"[OK] [V3.87 WRIST] 叠杯前腕部同心度微调 {k_ + 1}：补偿 (Δx={dx_m*1000:+.1f}, Δy={dy_m*1000:+.1f}) mm（总偏差 {err_mm:.1f} mm）")
        return out if out["checks"] or out.get("note") else None

    def _release_sidestep(self, dist_m=0.008):
        """V3.64 RELEASE SIDESTEP (fixed-jaw grippers): the released cup slides down along the FIXED pad and its
        widening conical wall wedges against it (V3.64 twin: hung tilted 13-18 deg on the SO-101 fixed jaw).  Move the
        TCP horizontally away from the wall (grasp -y; the moving jaw opened by 30 mm on the other side) so the cup
        drops free.  Path-checked straight move; refused -> warning only (the release already happened)."""
        p, q = self.io.tcp_pose()
        R = quat_xyzw_to_R(q)
        y = R @ self.prof.T_tcp_grasp[:3, 1]
        h = np.array([y[0], y[1], 0.0])
        if np.linalg.norm(h) < 1e-6:
            return
        tgt = np.asarray(p, float) - dist_m * h / np.linalg.norm(h)
        r = self.io.move_line("V3.64 release sidestep (fixed jaw off the wall)", [float(v) for v in tgt], q, 0.02)
        self.motions.append(("release sidestep", "line" if r.get("ok") else "skipped"))
        if not r.get("ok"):
            self.io.emit(f"[WARN] [V3.64 CUP] release sidestep not executed: {r.get('why') or r.get('error')}")

    def _state(self, upper, lower):
        if self.camera:
            self._perceive(f"check {upper} in {lower}")
            st = self.cam_stack
            if st and st["upper"] == upper and st["lower"] == lower:
                return {"xy_offset_mm": 0.0, "base_dz_mm": st["base_dz_mm"], "tilt_cup1_deg": self.cam[upper]["tilt_deg"],
                        "tilt_cup2_deg": self.cam[lower]["tilt_deg"], "stacked": True, "nested_full": st["nested_full"],
                        "nested": True, "success": True, "source": "fixed camera (one nested cluster)"}
            if self.prof.stack_by_cluster:
                byc = self._stack_by_cluster(upper, lower)
                if byc is not None:
                    return byc
            return {"stacked": False, "nested_full": False, "nested": False, "success": False,
                    "source": "fixed camera", "clusters": [(c["kind"], c["height_mm"]) for c in
                                                          (getattr(self, "cam_raw", {}) or {}).get("cups", [])]}
        a, b = self._cup(upper), self._cup(lower)
        return SC.cup_task_state(a["rim"], a["quat"], b["rim"], b["quat"], origin="rim")

    def _stack_by_cluster(self, upper, lower, near_m=0.035, min_dz_mm=4.0, max_dz_mm=40.0):
        """V3.71 STACK VERIFY BY CLUSTER: after the release exactly one cup cluster on the target's place, taller than
        the target cup alone (measured before the task moved anything) by 4..40 mm (or measured nested height + 8 mm)
        = stacked; the model's nested-height classes do not fit the real cups (V3.70: nested pair read as one cup)"""
        if self.placed != (upper, lower) or self.placed_xy is None:
            return None
        cups = [c for c in (getattr(self, "cam_raw", None) or {}).get("cups") or []
                if c.get("base_xyz_m") and c.get("kind") in ("cup", "stack", "unknown")]
        near = [c for c in cups if math.hypot(c["base_xyz_m"][0] - self.placed_xy[0],
                                              c["base_xyz_m"][1] - self.placed_xy[1]) <= near_m]
        h0 = self.single_h.get(lower)
        info = {"source": "fixed camera (V3.71 one cluster on the target's place)", "near_clusters": len(near),
                "target_alone_height_mm": h0}
        nh = near[0].get("height_mm") if near and near[0].get("height_mm") is not None else (near[0].get("height_depth_mm") if near else None)
        if len(near) != 1 or h0 is None or nh is None:
            return None
        nd = self.prof.nested_dz_real_mm
        dz = float(nh) - float(h0)
        hi = (nd + 8.0) if nd is not None else max_dz_mm
        others = [c for c in cups if c is not near[0] and c.get("kind") == "cup"]
        info.update(height_mm=near[0]["height_mm"], base_dz_mm=round(dz, 1),
                    nested_dz_real_mm=nd, other_single_cups=len(others))
        if not (min_dz_mm <= dz <= hi) or others:
            self.io.emit(f"[INFO] [V3.71 CUP] 叠好判断：目标位置的簇高 {near[0]['height_mm']} mm，比 {lower} 单独时 "
                         f"{h0:.0f} mm 高 {dz:+.1f} mm（要求 {min_dz_mm:.0f}–{hi:.0f} mm）"
                         + (f"，桌上还有 {len(others)} 只单杯" if others else "") + " → 不算叠好")
            return dict(info, stacked=False, nested_full=False, nested=False, success=False)
        full = None if nd is None else bool(dz <= nd + 5.0)
        self.io.emit(f"[OK] [V3.71 CUP] 叠好判断：目标位置只剩一个簇，高 {near[0]['height_mm']} mm，比 {lower} 单独时高 "
                     f"{dz:+.1f} mm → 已叠好" + ("" if nd is not None else "（实测套叠增量未填，是否完全套到底不判断）"))
        return dict(info, xy_offset_mm=0.0, tilt_cup1_deg=None, tilt_cup2_deg=None, stacked=True, nested_full=full,
                    nested=True, success=True)

    def verify_stack(self, upper_cup="cup_1", lower_cup="cup_2", **_):
        if upper_cup not in CUPS or lower_cup not in CUPS or upper_cup == lower_cup:
            return self._res("verify_stack", False, reason="BAD_CUP_NAMES")
        if self.held:
            return self._res("verify_stack", False, reason="CUP_STILL_HELD", nxt=["place_cup_into"])
        if self.camera and self.prof.clear_view_before_verify and hasattr(self.io, "clear_view"):
            # V3.69 CLEAR VIEW VERIFY: the gripper above the stack hid the rims (first real stack: NOT_STACKED although
            # nested) -> arm out of the view first; still hidden -> further away and look again
            self.io.clear_view(0, f"verify {upper_cup} in {lower_cup}")
            st = self._state(upper_cup, lower_cup)
            if not st["stacked"] and any(k == "occluded" for k, _h in st.get("clusters") or []):
                self.io.clear_view(1, f"verify {upper_cup} in {lower_cup}")
                st = self._state(upper_cup, lower_cup)
                st["view"] = "second clear-view retreat"
            else:
                st["view"] = "arm moved out of the camera view"
        else:
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
        if self.io.latch_pending():
            ok, why = self.io.clear_latch("V3.63 return_to_safe_pose")
            if not ok:
                return self._res("return_to_safe_pose", False, {"detail": why}, reason="SAFETY_LATCH_NOT_CLEARED",
                                 nxt=["stop_and_report"])
        self.io.gripper_open(self.prof.open_mm)
        r = self.io.go_home("V3.62 cup task")
        if self.stage not in ("VERIFIED",):
            self.stage = "OBSERVED"
        return self._res("return_to_safe_pose", bool(r.get("ok", True)), {"detail": r.get("error")})

    def run_tool(self, name, args):
        try:
            return self._run_tool(name, args)
        except self._Abort as e:                     # perception problems outside the motion wrappers
            return self._res(name, False, dict(e.info), reason=e.reason, nxt=["observe_scene", "stop_and_report"])

    def _run_tool(self, name, args):
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
6. GRASP_UNREACHABLE / PLACE_UNREACHABLE / JOINT_MARGIN_TOO_SMALL / CUP_NOT_UPRIGHT / TARGET_UNREACHABLE /
   SAFETY_LATCH_NOT_CLEARED: stop and report; do not try other motions.
6b. SAFETY_STOP: the simulator stopped the arm (e.g. a joint limit); the skill already ran STOP + RECOVER
   (physical_verification.latch_recovered) and a held cup stays in the gripper. If latch_recovered=true and the
   recommended actions contain the same tool: call observe_scene, then that tool ONCE more (it re-plans from the current
   posture). A second SAFETY_STOP of the same tool or latch_recovered=false: stop and report.
6c. plan_cup_stack ok=true is executable (executable=true): a joint-margin warning there is informational - continue
   with pick_cup.
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


# V4.6.9 TRY HARDER (LLM): overrides the "retry once / stop and report" rules; every failed tool result carries concrete
# alternatives; an early "I give up" is answered with a push (up to TRY_HARDER_PUSHES times) while steps remain
TRY_HARDER_PUSHES = 3
WRIST_PLACE_STEP_MAX_MM = 10.0     # V4.6.9: one wrist concentricity correction at most this
WRIST_PLACE_TOTAL_MAX_MM = 12.0    # V4.6.9: all wrist concentricity corrections together at most this
TRY_HARDER_INSTRUCTIONS = """
TRY HARDER MODE (the user enabled it; it OVERRIDES rules 3, 4 and 6 and every 'stop_and_report' recommendation):
- Do NOT give up after one failure. After a failed tool, read reason / physical_verification and the field
  try_harder.suggestions, then choose a DIFFERENT strategy and call a tool. Never answer with a final text while
  the task is not verified and tool steps remain, unless every suggestion has been tried.
- Strategies you can use with the existing tools: observe_scene again (fresh camera frames; the perception retries
  itself), return_to_safe_pose first when the arm may block the camera (only when no cup is held), plan_cup_stack
  again (the planner searches other rim angles each time) or with source and target swapped, pick_cup again after a
  new plan, place_cup_into again while the cup is held (it re-plans the place from the current pose).
- Keep a short plan in your text: what failed, why, what you try next. Stop only when the cup is verified or
  every strategy failed twice."""


def try_harder_suggestions(name, args, res):
    """V4.6.9: concrete next strategies for a failed cup tool (fed back to the LLM inside the tool result)"""
    reason = str(res.get("reason") or res.get("error") or "")
    held = bool((res.get("state_after") or {}).get("held_cup"))
    src = args.get("source_cup") or "cup_1"
    dst = args.get("target_cup") or "cup_2"
    if name == "observe_scene":
        out = ["observe_scene again (new frames)",
               "return_to_safe_pose, then observe_scene (the arm may block the fixed camera)" if not held else
               "observe_scene once more without moving (a cup is held)",
               "if one cup has no depth, still try plan_cup_stack with that cup as the TARGET (its position is used only "
               "for the place)"]
    elif name == "plan_cup_stack":
        out = [f"plan_cup_stack(source_cup='{dst}', target_cup='{src}') - the swapped order may be reachable",
               "return_to_safe_pose, observe_scene, then plan_cup_stack again (cup positions are re-measured, other rim "
               "angles are searched)"]
    elif name == "pick_cup":
        out = ["observe_scene, plan_cup_stack again, then pick_cup again",
               f"plan_cup_stack(source_cup='{dst}', target_cup='{src}') if the same cup keeps failing"]
    elif name == "place_cup_into":
        out = (["observe_scene (the held cup stays in the gripper), then place_cup_into again - it re-plans from the "
                "current pose", "place_cup_into once more if the reason was NOT_CENTRED_OVER_TARGET or a motion stop"]
               if held else ["the cup is no longer held: observe_scene, plan_cup_stack, pick_cup again"])
    elif name in ("verify_stack",):
        out = ["observe_scene, then verify_stack again (the first view may have been blocked)"]
    else:
        out = ["observe_scene and continue with the step that failed"]
    return {"failed_tool": name, "reason": reason, "suggestions": out,
            "do_not_give_up": "TRY HARDER is on: pick one of the suggestions now instead of reporting failure."}


def run_cup_task(agent, skills, task, model, api_key="", reasoning_effort="low", max_tool_steps=30, emit=None,
                 dry_run=False, try_harder=False):
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
        instr = CUP_SYSTEM_INSTRUCTIONS + (TRY_HARDER_INSTRUCTIONS if try_harder else "")
        pushes = 0
        payload = {"model": str(model), "instructions": instr, "input": str(task), "tools": CUP_TOOLS,
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
                verified = any(h["tool"] == "verify_stack" and h["result"].get("ok") for h in history)
                if try_harder and not verified and pushes < TRY_HARDER_PUSHES and steps < max_tool_steps - 1:
                    pushes += 1
                    last = next((h for h in reversed(history) if not h["result"].get("ok")), None)
                    why = f"{last['tool']}: {last['result'].get('reason') or last['result'].get('error')}" if last else "-"
                    sugg = try_harder_suggestions(last["tool"], {}, last["result"])["suggestions"] if last else \
                        ["observe_scene"]
                    emit(f"[V4.6.9 TRY HARDER] LLM 想结束但任务还没完成（{why}）→ 第 {pushes}/{TRY_HARDER_PUSHES} 次要求它换策略继续")
                    payload = {"model": str(model), "instructions": instr,
                               "previous_response_id": agent._last_response_id,
                               "input": [{"role": "user", "content":
                                          "TRY HARDER: the task is NOT finished (last failure " + why + "). Do not stop. "
                                          "Choose a different strategy now and call a tool. Options: " + "; ".join(sugg)}],
                               "tools": CUP_TOOLS, "tool_choice": "auto", "parallel_tool_calls": False,
                               "reasoning": {"effort": str(reasoning_effort)}, "max_output_tokens": 1200, "store": True}
                    continue
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
                if try_harder and not res.get("ok"):          # V4.6.9: no "stop", concrete alternatives instead
                    res = dict(res)
                    res["recommended_next_actions"] = [a for a in (res.get("recommended_next_actions") or [])
                                                       if a != "stop_and_report"] or ["observe_scene"]
                    res["try_harder"] = try_harder_suggestions(name, args, res)
                emit(f"[LLM TOOL RESULT #{steps}] {name} | elapsed={time.perf_counter() - t_tool:.2f}s | "
                     + json.dumps({k: v for k, v in res.items() if k != "scene"}, ensure_ascii=False)[:1500])
                history.append({"tool": name, "result": res})
                outs.append({"type": "function_call_output", "call_id": str(call.get("call_id", "")),
                             "output": json.dumps(res, ensure_ascii=False, default=str)})
            payload = {"model": str(model), "instructions": instr,
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
        if ctx.get("wrist_selected", lambda: False)():
            log("[FAIL] [V3.63 CUP] stack_cups: wrist-camera cup perception is not available - select the fixed stand "
                "camera (D435i) or 'JSON ground truth' as the pose source")
            if mb:
                mb.showerror("NERO V3.63", "叠杯子场景支持：固定主相机 D435i 或 JSON ground truth；腕部相机还不支持。")
            return
        if ctx.get("camera_available") is not None and not ctx["camera_available"]():
            log("[FAIL] [V3.63 CUP] the fixed camera process is not running - no fallback to simulator truth")
            if mb:
                mb.showerror("NERO V3.63", "固定主相机进程没有运行（不会退回仿真真值）。")
            return
        log("[INFO] [V3.63 CUP] pose source: fixed stand D435i RGB-D cup perception (no simulator truth)")
    if ctx["agent"] is None:
        log("[FAIL] [V3.62 CUP] the LLM agent module is not loaded")
        return
    if not ctx["safety_gate"]():
        log("[FAIL] [V3.62 CUP] RUN TASK blocked: stale physical safety latch was not recovered")
        return
    knobs, skill = cup_knobs()
    io = BackendIO(ctx)
    skills = CupSkills(io, knobs, skill)
    log(f"[V3.62 CUP TASK] scene={SC.selected_variant()} | skill {skill['code']} {skill['short_name']} ({skill['source']}) | "
        f"model={cfg['model']} | task={cfg['task']!r}")
    log(f"[INFO] [V3.64 CUP] gripper: {skills.prof.label}")                  # V3.64 SO-101 GRIPPER ON NERO

    def _run():
        if not bool(cfg.get("dry_run", False)):
            ctx["start_check"](False)
        return run_cup_task(ctx["agent"], skills, cfg["task"], cfg["model"], cfg["api_key"], cfg["reasoning_effort"],
                            max(10, min(60, int(cfg["max_tool_steps"]))), emit=io.emit,
                            dry_run=bool(cfg.get("dry_run", False)))

    ctx["background"]("RUN TASK (stack_cups)", _run, lock_motion=True)


def twin_episode(knobs=None, source="cup_1", target="cup_2", log=print, snapshot_dir=None, cup_xy=None,
                 pose_mode="truth", gripper="official"):
    """offline: the scripted CS episode in the headless twin -> (results, final cup state from simulator truth, io)"""
    k = dict(CUP_KNOB_DEFAULTS, **(knobs or {}))
    io = TwinIO(log=log, cup_xy=cup_xy, pose_mode=pose_mode, gripper=gripper)
    sk = CupSkills(io, k, {"code": "CS", "short_name": "Cup Stack", "skill_id": "cup_stack_rim_pinch"})
    res = sk.scripted_episode(source, target)
    io.skills = sk
    a, b = io.cup_pose(source), io.cup_pose(target)              # judged with simulator truth (also in camera mode)
    st = SC.cup_task_state(a[0], a[1], b[0], b[1], origin="rim")
    if snapshot_dir:
        try:
            io.snapshot(os.path.join(snapshot_dir, "v362_twin_final.png"))
        except Exception as exc:
            log(f"[INFO] [V3.62 CUP] no snapshot ({exc})")
    return res, st, io


def _study_job(job):
    import random
    k, src, dst, seed, solref = job[:5]
    pose_mode = job[5] if len(job) > 5 else "truth"
    gripper = job[6] if len(job) > 6 else "official"              # V3.64 SO-101 GRIPPER ON NERO
    if pose_mode == "camera":
        os.environ["NERO_V308_DEPTH_NOISE_K"] = "0.002"      # D435i-like depth noise sigma = k z^2 + temporal fusion
    else:
        os.environ.pop("NERO_V308_DEPTH_NOISE_K", None)
    if solref:
        os.environ["NERO_V362_CUP_SOLREF"] = solref
    else:
        os.environ.pop("NERO_V362_CUP_SOLREF", None)          # pool workers are reused: never inherit a previous job's
    rnd = random.Random(seed)
    xy = ({n: (x + rnd.uniform(-0.01, 0.01), y + rnd.uniform(-0.01, 0.01)) for n, (x, y) in SC.CUP_XY.items()}
          if seed else None)
    res, st, io = twin_episode(k, src, dst, log=lambda s_: None, cup_xy=xy, pose_mode=pose_mode, gripper=gripper)
    pick = next((r for r in res if r["tool"] == "pick_cup"), {})
    return {"knobs": k, "source": src, "target": dst, "position_noise_seed": seed, "cup_solref": solref or "scene default",
            "pose_mode": pose_mode, "gripper": gripper,
            "line_refused_to_joint_space": [lab for lab, kind in io.skills.motions if "refused" in kind],
            "stacked": st["stacked"], "nested_full": st["nested_full"], "base_dz_mm": st["base_dz_mm"],
            "tilt_after_lift_deg": (pick.get("physical_verification") or {}).get("cup_tilt_after_lift_deg"),
            "failed_at": next((f"{r['tool']}:{r.get('reason')}" for r in res if not r["ok"]), None),
            "sim_time_s": round(io.t, 1)}


def twin_study(processes=4, log=print, out_path=None, camera=False):
    """V3.62 twin study (about 5 min on 4 cores): defaults both directions, +-10 mm cup position noise over pinch
    depth x clamp force, two cup contact stiffnesses, and the V3.61-style controls (25 mm / vertical approach)."""
    from multiprocessing import Pool
    jobs = [({}, "cup_1", "cup_2", 0, None), ({}, "cup_2", "cup_1", 0, None)]
    jobs += [({"cup_rim_pinch_depth_mm": d, "cup_clamp_force_n": f}, "cup_1", "cup_2", sd, None)
             for d in (30.0, 35.0, 40.0) for f in (3.0, 5.0) for sd in (1, 2)]
    jobs += [({}, "cup_1", "cup_2", 3, sr) for sr in ("-5000 -30", "-50000 -100")]
    jobs += [({"cup_rim_pinch_depth_mm": 25.0}, "cup_1", "cup_2", 0, None), ({"cup_wall_tilt_follow": False}, "cup_1", "cup_2", 0, None),
             ({"cup_rim_pinch_depth_mm": 25.0, "cup_wall_tilt_follow": False}, "cup_1", "cup_2", 0, None)]
    if camera:            # V3.63: fixed D435i perception instead of simulator truth (needs OpenGL in the workers)
        jobs += [({}, "cup_1", "cup_2", 0, None, "camera"), ({}, "cup_2", "cup_1", 0, None, "camera"),
                 ({}, "cup_1", "cup_2", 4, None, "camera"), ({}, "cup_2", "cup_1", 5, None, "camera")]
    with Pool(processes) as pool:
        rows = []
        for r in pool.imap(_study_job, jobs):
            rows.append(r)
            log(f"[V3.63 twin study] {r['pose_mode']} {r['source']}->{r['target']} {r['knobs']} seed {r['position_noise_seed']} "
                f"{r['cup_solref']}: stacked={r['stacked']} nested_full={r['nested_full']} lift tilt {r['tilt_after_lift_deg']}"
                + (f" FAILED {r['failed_at']}" if r["failed_at"] else ""))
    knobs, _sk = cup_knobs()
    study = {"version": VERSION, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
             "what": ("headless MuJoCo twin: official AgileX finger pads driven by the URDF arm kinematics (V3.63: joint "
                      "space, Cartesian moves as checked IK-continuity paths) + backend gripper logic (force clamp, "
                      "penetration guard, auto CONNECT hold) + the two scene cups; scripted CS episode observe -> plan -> "
                      "pick -> place -> verify; camera rows: fixed stand D435i perception instead of simulator truth. No "
                      "arm dynamics / ros2_control. Not measured on hardware."),
             "layout": {k_: list(v) for k_, v in SC.CUP_XY.items()}, "cup_origin": SC.CUP_ORIGIN,
             "defaults": knobs, "rows": rows,
             "summary": {"camera_runs": sum(1 for r in rows if r["pose_mode"] == "camera"),
                         "camera_success": sum(1 for r in rows if r["pose_mode"] == "camera" and r["nested_full"]),
                         "default_like_runs": sum(1 for r in rows if not ({"cup_wall_tilt_follow"} & set(r["knobs"]))
                                                  and r["knobs"].get("cup_rim_pinch_depth_mm", 35.0) >= 30.0),
                         "default_like_success": sum(1 for r in rows if r["nested_full"] and not ({"cup_wall_tilt_follow"} & set(r["knobs"]))
                                                     and r["knobs"].get("cup_rim_pinch_depth_mm", 35.0) >= 30.0)}}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(study, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return study


def so101_study(processes=4, log=print, out_path=None, camera=True):
    """V3.64 SO-101 twin study: both directions, +-10 mm cup position noise (seeds 1-4), pinch depth 30 / 40 mm,
    and (camera=True, needs OpenGL) the fixed D435i as the pose source"""
    from multiprocessing import Pool
    jobs = [({}, a, b, sd, None, "truth", "so101") for a, b in (("cup_1", "cup_2"), ("cup_2", "cup_1")) for sd in (0, 1, 2, 3, 4)]
    jobs += [({"cup_rim_pinch_depth_mm": d}, "cup_1", "cup_2", 0, None, "truth", "so101") for d in (30.0, 40.0)]
    if camera:
        jobs += [({}, a, b, sd, None, "camera", "so101") for a, b in (("cup_1", "cup_2"), ("cup_2", "cup_1")) for sd in (0, 4, 5)]
    with Pool(processes) as pool:
        rows = []
        for r in pool.imap(_study_job, jobs):
            rows.append(r)
            log(f"[V3.64 SO-101 twin study] {r['pose_mode']} {r['source']}->{r['target']} {r['knobs']} seed "
                f"{r['position_noise_seed']}: nested_full={r['nested_full']}" + (f" FAILED {r['failed_at']}" if r["failed_at"] else ""))
    import NERO_V364_SO101_Gripper as G
    study = {"version": VERSION, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
             "what": ("headless MuJoCo twin with the SO-101 gripper (official SO-101 meshes, simplified collision boxes, "
                      "4 mm tip pads) on the assumed 7 mm ring / -45 deg roll mount; same arm kinematics, gripper logic "
                      "and cups as the V3.63 study. Not measured on hardware."),
             "mount": G.describe(), "rows": rows,
             "summary": {"runs": len(rows), "nested_full": sum(1 for r in rows if r["nested_full"]),
                         "camera_runs": sum(1 for r in rows if r["pose_mode"] == "camera"),
                         "camera_nested_full": sum(1 for r in rows if r["pose_mode"] == "camera" and r["nested_full"])}}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(study, f, ensure_ascii=False, indent=2)
            f.write("\n")
    return study


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--study-so101":
        so101_study(out_path=sys.argv[2] if len(sys.argv) > 2 else None)
    elif len(sys.argv) > 1 and sys.argv[1] in ("--study", "--study-camera"):
        twin_study(out_path=sys.argv[2] if len(sys.argv) > 2 else None, camera=sys.argv[1] == "--study-camera")
    else:
        r, st, _io = twin_episode(log=print)
        print(json.dumps(st))
