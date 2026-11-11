#!/usr/bin/env python3
"""NERO V3.75 COLLISION PRE-CHECK - which parts touch the table / bins in a candidate joint state, and why MoveIt refused.

V3.75 COLLISION PRE-CHECK

Why (user run V3.74, sort_items): pick_object(banana) "down to item_1" - the MoveIt Cartesian descent was only
12.5 % / 19 % collision-free, the straight joint-goal fallback failed with FAILURE (99999).  The twin had no arm /
MoveIt collision model, so it never saw it.  MoveIt's robot model is the OFFICIAL AgileX gripper (start_nero_sim.sh:
effector_type:=agx_gripper), not the SO-101 that the simulation really carries: its fingers reach ~19 mm below the
SO-101 grasp point (T_tcp_grasp z = -18.8 mm) and sit where the SO-101 is not -> MoveIt sees collisions with the table /
bin walls that the real SO-101 does not have.

This module (no ROS import at module level; MuJoCo not needed):
  * world_boxes(scene)        table (the planning-scene box "mujoco_play_table") + sort_items bin walls / floors
  * so101_parts(q, opening)   coarse oriented boxes of the SO-101 gripper (housing, fixed finger + pad, moving jaw +
                              pad at the jaw angle of the opening, seat + ring) and the link5-7 bodies, from the URDF
                              forward kinematics of the candidate joints (not the current state)
  * check_parts(parts, world, margin)   -> [(part, world box), ...] touching / closer than margin (SAT, OBB-OBB)
  * moveit_contacts(core, q)  MoveIt /check_state_validity at q with ALL contact pairs (NERO_V71 _v71_state_is_valid
                              only returns the first) -> {"available", "valid", "pairs"}
  * classify_pairs(pairs)     -> blocking (an arm link 0-7 / two world objects) vs model_only (a link of MoveIt's
                              gripper model - not the SO-101 that is really mounted)
  * assess(core, q, opening, world)   both checks together -> dict(ok, blocking, model_only, so101, moveit_available)
  * official_gripper_parts(q, opening)  an ESTIMATED envelope of the official gripper, only for the twin's MoveIt
                              emulation in the tests (the real MoveIt uses its own mesh)
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

VERSION = "V3.75"
MARKER = "V3.75 COLLISION PRE-CHECK"
ARM_LINKS = ("base_link",) + tuple(f"link{i}" for i in range(1, 8))
TABLE = ("mujoco_play_table", (-0.385, 0.0, -0.03), (0.335, 0.38, 0.03))     # = backend planning scene (V69)
MARGIN_M = 0.002


# ----------------------------------------------------------------------------------------------- geometry
def _box(name, centre, R, half):
    return (str(name), np.asarray(centre, float), np.asarray(R, float), np.asarray(half, float))


def world_boxes(scene=None):
    """table + (sort_items) the bins - axis-aligned boxes like the backend's planning scene"""
    out = [_box(TABLE[0], TABLE[1], np.eye(3), TABLE[2])]
    if scene == "sort_items":
        import NERO_V343_Scenes as SC
        out += [_box(n, c, np.eye(3), h) for n, c, h, _rgba in SC.sort_bin_boxes()]
    return out


def obb_overlap(a, b, margin=0.0):
    """separating-axis test of two oriented boxes (centre, R columns = axes, half sizes); margin inflates both"""
    _n, ca, Ra, ha = a
    _m, cb, Rb, hb = b
    ha, hb = ha + margin / 2.0, hb + margin / 2.0
    t = cb - ca
    axes = [Ra[:, i] for i in range(3)] + [Rb[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            c = np.cross(Ra[:, i], Rb[:, j])
            n = float(np.linalg.norm(c))
            if n > 1e-9:
                axes.append(c / n)
    for L in axes:
        ra = sum(ha[i] * abs(float(Ra[:, i] @ L)) for i in range(3))
        rb = sum(hb[i] * abs(float(Rb[:, i] @ L)) for i in range(3))
        if abs(float(t @ L)) > ra + rb:
            return False
    return True


def _arm():
    import NERO_V362_Cup_Skills as CK
    if "t" not in CK._IK:
        CK._IK["t"] = CK._ik_tools()
    return CK._IK["t"][0]


def link_frames(q):
    """world transforms of link1..link7 at q (URDF chain of NERO_V362 FastArm)"""
    arm = _arm()
    T = np.eye(4)
    out = {}
    for Tf, ax, k in arm.steps:
        T = T @ Tf
        if ax is not None:
            M = np.eye(4)
            M[:3, :3] = arm._rot(ax, float(q[k]))
            T = T @ M
            out[f"link{k + 1}"] = T.copy()
    return out


def _local(T, name, c_local, R_local, half):
    return _box(name, T[:3, :3] @ np.asarray(c_local, float) + T[:3, 3], T[:3, :3] @ np.asarray(R_local, float), half)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def so101_parts(q, opening_mm=None, arm_links=True):
    """coarse boxes of the mounted SO-101 (sim mount NERO_V364.MOUNT) + link5..link7 bodies at the joints q"""
    import NERO_V364_SO101_Gripper as G
    fr = link_frames(q)
    T7 = fr["link7"]
    Tg = T7 @ G.T_l7_gripper()
    parts = []
    sx, rz, ring = G.MOUNT["seat_x_m"], G.MOUNT["axis_z_m"], G.MOUNT["ring_m"]
    x0, x1 = 0.004, sx + ring
    parts.append(_local(T7, "so101_seat_ring", [(x0 + x1) / 2.0, 0.0, rz], np.eye(3), [(x1 - x0) / 2.0, 0.030, 0.030]))
    I3 = np.eye(3)
    parts.append(_local(Tg, "so101_housing", [0, 0, -0.021], I3, [0.031, 0.024, 0.019]))
    parts.append(_local(Tg, "so101_fixed_finger", [-0.016, 0, -0.070], I3, [0.005, 0.008, 0.030]))
    parts.append(_local(Tg, "so101_fixed_pad", G.PAD_FIXED_CENTRE, I3, G.PAD_HALF))
    th = math.radians(G.jaw_angle_deg(60.0 if opening_mm is None else float(opening_mm)))
    Th = np.eye(4)
    Th[:3, :3] = G._quat_R(G.HINGE_QUAT) @ _rz(th)
    Th[:3, 3] = G.HINGE_POS
    Tj = Tg @ Th
    Rj0 = G._quat_R(G.HINGE_QUAT)
    fin2 = Rj0.T @ (np.array([0.0170, 0.0, -0.0720]) - np.asarray(G.HINGE_POS))
    pad2 = Rj0.T @ (np.array([-G.PAD_FIXED_CENTRE[0], 0.0, G.PAD_FIXED_CENTRE[2]]) - np.asarray(G.HINGE_POS))
    parts.append(_local(Tj, "so101_moving_jaw", fin2, I3, [0.0040, 0.0300, 0.0075]))
    parts.append(_local(Tj, "so101_moving_pad", pad2, I3, [G.PAD_HALF[0], G.PAD_HALF[2], G.PAD_HALF[1]]))
    if arm_links:
        # NERO forearm / wrist bodies, coarse (link frame origins, ~45 mm radius)
        for ln, half in (("link5", 0.040), ("link6", 0.040), ("link7", 0.036)):
            parts.append(_local(fr[ln], ln, [0, 0, 0], I3, [half, half, half]))
    return parts


def official_gripper_parts(q, opening_mm=None):
    """ESTIMATED envelope of the official AgileX gripper of MoveIt's model (twin emulation in the tests only):
    palm from the seat to 40 mm before the official TCP, two fingers to 4 mm beyond it, sliding along the official
    closing axis (link7 -z) by +-opening/2"""
    fr = link_frames(q)
    T7 = fr["link7"]
    import NERO_V364_SO101_Gripper as G
    tcp_x, rz = float(G.T_L7_TCP[0, 3]), float(G.T_L7_TCP[2, 3])
    half_open = (40.0 if opening_mm is None else float(opening_mm)) / 2000.0
    I3 = np.eye(3)
    x0, x1 = 0.0375, tcp_x - 0.040
    parts = [_local(T7, "agx_gripper_base", [(x0 + x1) / 2.0, 0.0, rz], I3, [(x1 - x0) / 2.0, 0.030, 0.040])]
    f0, f1 = x1, tcp_x + 0.004
    for s, nm in ((1.0, "agx_gripper_finger_1"), (-1.0, "agx_gripper_finger_2")):
        parts.append(_local(T7, nm, [(f0 + f1) / 2.0, 0.0, rz - s * (half_open + 0.006)], I3,
                            [(f1 - f0) / 2.0, 0.010, 0.006]))
    return parts


SYM_GRIPPERS = ("official", "agx_gripper", "agx")
TWIN_OFFICIAL_TIP_FROM_FLANGE_MM = 154.3            # V4.8.3: the twin's official pads end 30 mm past the TCP


def agx_parts(q, opening_mm=None, tip_from_flange_mm=None, T_flange_cam=None, arm_links=True):
    """V4.8.3 TIDY DESK: coarse boxes of the NERO original gripper (NERO_V481_Agx_Geom: housing 75 x 164 x 65 mm, two
    56 x 26.5 x 76.5 mm fingers at +-opening/2, optional wrist-camera bracket from the hand-eye) + link5..link7"""
    import NERO_V481_Agx_Geom as AG
    fr = link_frames(q)
    T7 = fr["link7"]
    Tg = T7 @ AG.T_l7_grasp()
    cam_G = None
    if T_flange_cam is not None:
        cam = np.asarray(T_flange_cam, float)[:3, 3]
        Tl = AG.T_l7_grasp()
        cam_G = Tl[:3, :3].T @ (cam - Tl[:3, 3])
    opening = 60.0 if opening_mm is None else float(opening_mm)
    old = AG.tip_from_flange_mm
    if tip_from_flange_mm is not None:
        AG.tip_from_flange_mm = lambda: float(tip_from_flange_mm)
    try:
        boxes = AG.boxes_in_G(opening, cam_G, margin_mm=0.0)
    finally:
        AG.tip_from_flange_mm = old
    I3 = np.eye(3)
    parts = [_local(Tg, f"agx_{n}", c, I3, h) for n, c, h in boxes]
    if arm_links:
        for ln, half in (("link5", 0.040), ("link6", 0.040), ("link7", 0.036)):
            parts.append(_local(fr[ln], ln, [0, 0, 0], I3, [half, half, half]))
    return parts


def gripper_parts(q, opening_mm=None, gripper="so101", **kw):
    """V4.8.3: the coarse boxes of the mounted gripper (so101 / official / agx_gripper)"""
    if str(gripper) in SYM_GRIPPERS:
        tip = kw.get("tip_from_flange_mm")
        if tip is None and str(gripper) == "official":
            tip = TWIN_OFFICIAL_TIP_FROM_FLANGE_MM
        return agx_parts(q, opening_mm, tip, kw.get("T_flange_cam"))
    return so101_parts(q, opening_mm)


def check_parts(parts, world, margin=MARGIN_M):
    out = []
    for p in parts:
        for w in world:
            if obb_overlap(p, w, margin):
                out.append((p[0], w[0]))
    return out


def item_box(centre_xy, z_bottom, yaw_deg, spec):
    """V3.75: footprint box of an item lying flat (centre, long-axis yaw) - NERO_V343 sort_item_footprint"""
    import NERO_V343_Scenes as SC
    c2, half2, a = SC.sort_item_footprint(spec, centre_xy, yaw_deg)
    R = _rz(a)
    return _box("held_item", [c2[0], c2[1], z_bottom + spec["height"] / 2.0], R, [half2[0], half2[1], spec["height"] / 2.0])


# ----------------------------------------------------------------------------------------------- MoveIt
def moveit_contacts(core, q, timeout_s=1.0):
    """MoveIt /check_state_validity at the arm joints q (the gripper joints from the live joint state), every
    contact pair -> {"available": bool, "valid": bool, "pairs": [(body1, body2), ...], "why": str}"""
    fn = getattr(core, "v375_state_validity", None)          # test doubles (MoveIt emulation)
    if fn is not None:
        return fn(q)
    cli = getattr(core, "state_validity_client", None)
    if cli is None:
        return {"available": False, "valid": None, "pairs": [], "why": "no /check_state_validity client"}
    try:
        if not cli.wait_for_service(timeout_sec=0.2):
            return {"available": False, "valid": None, "pairs": [], "why": "MoveIt /check_state_validity unavailable"}
        from moveit_msgs.msg import RobotState
        from moveit_msgs.srv import GetStateValidity
        req = GetStateValidity.Request()
        req.robot_state = RobotState()
        req.robot_state.joint_state = core._v71_full_joint_state_for_arm([float(v) for v in q])
        req.robot_state.is_diff = False
        req.group_name = "arm"
        resp = core.wait_future(cli.call_async(req), float(timeout_s), "V3.75 state validity")
    except Exception as exc:
        return {"available": False, "valid": None, "pairs": [], "why": f"state-validity exception: {str(exc)[:160]}"}
    if resp is None:
        return {"available": False, "valid": None, "pairs": [], "why": "state-validity returned no response"}
    pairs = []
    for c in list(getattr(resp, "contacts", []) or []):
        pr = (str(getattr(c, "contact_body_1", "?")), str(getattr(c, "contact_body_2", "?")))
        if pr not in pairs and (pr[1], pr[0]) not in pairs:
            pairs.append(pr)
    return {"available": True, "valid": bool(resp.valid), "pairs": pairs,
            "why": "" if resp.valid else ("collision" if pairs else "invalid (constraint / no contact reported)")}


def is_world(name, world=None):
    n = str(name)
    names = {w[0] for w in (world or [])}
    return n in names or n.startswith(("mujoco_play_table", "sort_bin_", "egg_holder", "v343", "sim_", "safe_"))


def classify_pairs(pairs, world=None):
    """-> (blocking, model_only): a pair is model_only when one body is a link of MoveIt's gripper model (a robot link
    that is not base_link / link1..link7) - that gripper is the official one, not the mounted SO-101"""
    blocking, model_only = [], []
    for a, b in pairs:
        grip = [n for n in (a, b) if n not in ARM_LINKS and not is_world(n, world)]
        (model_only if grip else blocking).append((a, b))
    return blocking, model_only


def assess(core, q, opening_mm=None, world=None, margin=MARGIN_M, extra_parts=None, gripper="so101", **kw):
    """SO-101 coarse geometry + MoveIt at q -> dict(ok, blocking, model_only, so101, moveit_available, moveit_valid)
    V4.8.3: gripper "official" / "agx_gripper" -> the original gripper's boxes (key "so101" kept: the mounted
    gripper's own boxes); MoveIt's model IS that gripper then, so its gripper-link contacts block too"""
    world = world if world is not None else world_boxes()
    so = check_parts(gripper_parts(q, opening_mm, gripper, **kw) + list(extra_parts or []), world, margin)
    mv = moveit_contacts(core, q) if core is not None else {"available": False, "valid": None, "pairs": []}
    blocking, model_only = classify_pairs(mv.get("pairs") or [], world)
    if str(gripper) in SYM_GRIPPERS:
        blocking, model_only = blocking + model_only, []
    if mv.get("available") and mv.get("valid") is False and not mv.get("pairs"):
        blocking.append(("moveit", mv.get("why") or "invalid"))
    return {"ok": not so and not blocking, "so101": so, "blocking": blocking, "model_only": model_only,
            "moveit_available": bool(mv.get("available")), "moveit_valid": mv.get("valid")}


def pairs_text(pairs, n=4):
    return ", ".join(f"{a}<->{b}" for a, b in list(pairs)[:n]) + (f" (+{len(pairs) - n})" if len(pairs) > n else "")


# ----------------------------------------------------------------------------------------------- 2-D footprints
def rect_corners(centre, half, ang):
    """corners of a rectangle (centre xy, (half along, half across), long-axis angle rad)"""
    c, s = math.cos(ang), math.sin(ang)
    u, v = np.array([c, s]) * half[0], np.array([-s, c]) * half[1]
    p = np.asarray(centre, float)
    return [p + u + v, p - u + v, p - u - v, p + u - v]


def _seg_dist(a, b, c, d):
    def pt(p, a_, b_):
        ab = b_ - a_
        t = float(np.clip((p - a_) @ ab / max(float(ab @ ab), 1e-12), 0.0, 1.0))
        return float(np.linalg.norm(p - (a_ + t * ab)))
    return min(pt(a, c, d), pt(b, c, d), pt(c, a, b), pt(d, a, b))


def rect_dist(r1, r2):
    """clearance between two footprints (centre, half, ang) - 0 when they overlap"""
    P, Q = rect_corners(*r1), rect_corners(*r2)
    # overlap: separating axis test on the 4 edge normals
    sep = False
    for poly in (P, Q):
        for i in range(4):
            e = poly[(i + 1) % 4] - poly[i]
            n = np.array([-e[1], e[0]])
            pa = [float(x @ n) for x in P]
            pb = [float(x @ n) for x in Q]
            if max(pa) < min(pb) or max(pb) < min(pa):
                sep = True
                break
        if sep:
            break
    if not sep:
        return 0.0
    return min(_seg_dist(P[i], P[(i + 1) % 4], Q[j], Q[(j + 1) % 4]) for i in range(4) for j in range(4))
