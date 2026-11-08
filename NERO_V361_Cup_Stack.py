#!/usr/bin/env python3
"""NERO V3.61 stack_cups: offline skill study (MuJoCo physics + URDF IK), no ROS / MoveIt / robot needed.

V3.61 STACK CUPS SCENE + OFFLINE SKILL STUDY

Skill being studied ("CS Cup Stack", RoboDojo stack_cups style, two-finger gripper):
  1. rim pinch      top-down, closing axis radial: one finger inside cup_1's wall, one outside, pads `pinch_depth`
                    below the rim (the cup is 95 mm wide at the rim, the gripper opens < 100 mm, so a top-down grasp
                    around the cup is impossible; the bottom is narrower than the rim, so fingers coming down outside
                    the cup would hit the rim first)
  2. lift + carry   upright, the TCP keeps the offset to the cup axis (rim radius)
  3. align          cup_1 axis over cup_2 axis (xy error, tilt)
  4. lower + release  cup_1 is lowered to `release_gap` above its final nested height and released; it slides in.
                    The outer finger is above cup_2's rim (same rim radius), so cup_1 can only be lowered until
                    its rim is pinch_depth + clearance above cup_2's rim - the nested height leaves ~25 mm.

What the study does
  * release_trial()      one MuJoCo trial: cup_1 held by a mocap "hand" (weld) at a commanded xy error / tilt / gap
                         above its nested height, released, simulated, judged with NERO_V343_Scenes.cup_task_state
  * release_sweep()      grid over xy error x gap x tilt x error direction -> success map
  * choose_release_params()  robust parameters from the map (largest 100 %-success region, safety factor)
  * grasp_reachability() URDF IK for pregrasp / grasp / lift / place / release poses around the rim -> best rim
                         angle (joint-limit margin) for the scene layout
Limits (honest): the fingers are not simulated here (only the cups); the real robot pinch is friction, the sim
hold is a CONNECT point; results are physics-model results, not measured on hardware.
"""
import math
import os
import sys

VERSION = "V3.61"
MARKER = "V3.61 STACK CUPS SCENE + OFFLINE SKILL STUDY"
HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import NERO_V343_Scenes as SC  # noqa: E402

TABLE_Z0 = 0.0005                        # cup body origin above the table top (scene z0)


def mujoco_available():
    try:
        import mujoco  # noqa: F401
        return True
    except Exception:
        return False


def study_xml(target_xy=(-0.40, 0.14), cup=None):
    """table + cup_2 (target, free) + cup_1 (carried, free) welded to a mocap hand."""
    c2 = SC.cup_body_xml("obj_cup_2", target_xy, cup=cup, z0=TABLE_Z0)
    c1 = SC.cup_body_xml("obj_cup_1", (target_xy[0], target_xy[1] + 0.25), cup=cup, z0=TABLE_Z0)
    return f"""<mujoco model="nero_v361_cup_study">
  <option timestep="0.001" integrator="implicitfast"/>
  <asset>{SC.CUP_ASSETS}</asset>
  <worldbody>
    <geom name="play_table" type="box" pos="-0.385 0 -0.03" size="0.335 0.38 0.03" friction="1.2 0.01 0.001"
          solref="0.008 1" solimp="0.96 0.995 0.0005" margin="0.001"/>
    <body name="hand" mocap="true" pos="{target_xy[0]} {target_xy[1] + 0.25} {TABLE_Z0}">
      <geom type="sphere" size="0.004" rgba="0 0.6 1 0.4" contype="0" conaffinity="0"/>
    </body>
    {c2}
    {c1}
  </worldbody>
  <equality>
    <weld name="hold" body1="hand" body2="obj_cup_1" relpose="0 0 0 1 0 0 0" solref="0.004 1"/>
  </equality>
</mujoco>"""


_MODEL_CACHE = {}


def _model(target_xy=(-0.40, 0.14)):
    import mujoco
    key = tuple(target_xy)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = mujoco.MjModel.from_xml_string(study_xml(target_xy))
    return _MODEL_CACHE[key]


def nested_height(target_xy=(-0.40, 0.14), settle_s=1.5):
    """base-to-base height of a nested cup_1 (dropped centred from just above), measured in the model."""
    import mujoco
    m = _model(target_xy)
    d = mujoco.MjData(m)
    _place(m, d, target_xy, (0.0, 0.0), 0.0, 0.0, z_base=TABLE_Z0 + 0.040)
    d.eq_active[0] = 0
    for _ in range(int(settle_s / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return float(d.body("obj_cup_1").xpos[2] - d.body("obj_cup_2").xpos[2])


def _quat_tilt(tilt_deg, axis_deg):
    """rotation about a horizontal axis at angle axis_deg (wxyz)"""
    a = math.radians(tilt_deg) / 2
    ax = math.radians(axis_deg)
    return (math.cos(a), math.sin(a) * math.cos(ax), math.sin(a) * math.sin(ax), 0.0)


def _place(m, d, target_xy, dxy, tilt_deg, tilt_axis_deg, z_base):
    import mujoco
    mujoco.mj_resetData(m, d)
    q = _quat_tilt(tilt_deg, tilt_axis_deg)
    pos = (target_xy[0] + dxy[0], target_xy[1] + dxy[1], z_base)
    jid = m.joint(m.body("obj_cup_1").jntadr[0]).id
    adr = m.jnt_qposadr[jid]
    d.qpos[adr:adr + 3] = pos
    d.qpos[adr + 3:adr + 7] = q
    mid = m.body_mocapid[m.body("hand").id]
    d.mocap_pos[mid] = pos
    d.mocap_quat[mid] = q
    d.eq_active[0] = 1
    mujoco.mj_forward(m, d)


def release_trial(xy_err_mm=0.0, err_dir_deg=0.0, gap_mm=10.0, tilt_deg=0.0, tilt_axis_deg=0.0,
                  target_xy=(-0.40, 0.14), z_nested=None, hold_s=0.3, settle_s=1.5):
    """cup_1 held `gap_mm` above its nested height with the given xy error / tilt, then released.
    -> dict(success, nested, xy_offset_mm, tilt_cup1_deg, cup2_moved_mm, ...)"""
    import mujoco
    m = _model(target_xy)
    d = mujoco.MjData(m)
    zn = nested_height(target_xy) if z_nested is None else z_nested
    e = xy_err_mm / 1000.0
    dxy = (e * math.cos(math.radians(err_dir_deg)), e * math.sin(math.radians(err_dir_deg)))
    _place(m, d, target_xy, dxy, tilt_deg, tilt_axis_deg, z_base=TABLE_Z0 + zn + gap_mm / 1000.0)
    p2_0 = d.body("obj_cup_2").xpos.copy()
    for _ in range(int(hold_s / m.opt.timestep)):          # hand still: lets contacts settle (no release yet)
        mujoco.mj_step(m, d)
    d.eq_active[0] = 0
    for _ in range(int(settle_s / m.opt.timestep)):
        mujoco.mj_step(m, d)
    b1, b2 = d.body("obj_cup_1"), d.body("obj_cup_2")
    st = SC.cup_task_state(b1.xpos, b1.xquat, b2.xpos, b2.xquat)
    moved = math.hypot(*(b2.xpos[:2] - p2_0[:2])) * 1000.0
    st.update({"cup2_moved_mm": round(moved, 1), "xy_err_mm": xy_err_mm, "gap_mm": gap_mm, "tilt_deg": tilt_deg,
               "success": bool(st["stacked"] and moved <= 5.0),
               "full_success": bool(st["nested_full"] and moved <= 5.0)})
    return st


def release_sweep(xy_errs=(0, 4, 8, 12, 16, 20), gaps=(0, 10, 20, 35, 50), tilts=(0, 5), dirs=(0, 90, 180, 270),
                  target_xy=(-0.40, 0.14), log=None):
    zn = nested_height(target_xy)
    rows = []
    for gap in gaps:
        for tilt in tilts:
            for e in xy_errs:
                ok = full = 0
                n = 0
                for dd in (dirs if e > 0 else (0,)):
                    r = release_trial(e, dd, gap, tilt, tilt_axis_deg=dd + 90, target_xy=target_xy, z_nested=zn)
                    ok += int(r["success"])
                    full += int(r["full_success"])
                    n += 1
                rows.append({"gap_mm": gap, "tilt_deg": tilt, "xy_err_mm": e, "success_rate": ok / n,
                             "full_rate": full / n, "n": n})
                if log:
                    log(f"[V3.61 cup study] gap {gap:>3} mm tilt {tilt:>2} deg xy {e:>2} mm -> stacked {ok}/{n}, "
                        f"fully nested {full}/{n}")
    return {"z_nested_mm": round(zn * 1000.0, 1), "rows": rows}


def choose_release_params(sweep, safety=0.5):
    """gap with the widest xy tolerance at 100 % success for every tilt in the sweep; knob = tolerance x safety."""
    rows = sweep["rows"]
    gaps = sorted({r["gap_mm"] for r in rows})
    best = None
    for g in gaps:
        tol = None
        for t in sorted({r["tilt_deg"] for r in rows}):
            ok_e = [r["xy_err_mm"] for r in rows if r["gap_mm"] == g and r["tilt_deg"] == t]
            ok_e = sorted(ok_e)
            lim = -1
            for e in ok_e:
                r = next(x for x in rows if x["gap_mm"] == g and x["tilt_deg"] == t and x["xy_err_mm"] == e)
                if r["success_rate"] < 1.0:
                    break
                lim = e
            tol = lim if tol is None else min(tol, lim)
        if tol is not None and tol >= 0 and (best is None or tol > best[1] or (tol == best[1] and g < best[0])):
            best = (g, tol)
    if best is None:
        return None
    return {"release_gap_mm": float(best[0]), "xy_tolerance_found_mm": float(best[1]),
            "release_xy_tol_mm": round(best[1] * safety, 1), "z_nested_mm": sweep["z_nested_mm"],
            "tilt_tested_deg": max(r["tilt_deg"] for r in rows)}


# --------------------------------------------------------------------------- grasp / place reachability (URDF IK)
def tcp_quat_for_yaw(yaw_rad):
    """top-down TCP orientation for gripper yaw (backend convention: TCP x = yaw direction, closing axis = TCP y)."""
    return (math.cos(yaw_rad / 2), math.sin(yaw_rad / 2), 0.0, 0.0)          # xyzw: 180 deg about a horizontal axis


def _rot_xyzw(q):
    x, y, z, w = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


def _rotvec(R):
    """rotation vector of a 3x3 rotation matrix"""
    tr = R[0][0] + R[1][1] + R[2][2]
    ang = math.acos(max(-1.0, min(1.0, (tr - 1) / 2)))
    if ang < 1e-9:
        return [0.0, 0.0, 0.0]
    if ang > math.pi - 1e-6:                       # 180 deg: axis from the diagonal
        ax = [math.sqrt(max(0.0, (R[i][i] + 1) / 2)) for i in range(3)]
        return [a * ang for a in ax]
    k = ang / (2 * math.sin(ang))
    return [k * (R[2][1] - R[1][2]), k * (R[0][2] - R[2][0]), k * (R[1][0] - R[0][1])]


def pose_ik(urdf, joints, limits, seeds, tool_T, xyz, R_des, base="base_link", w_rot=0.10, iters=120,
            tol_m=5e-4, tol_deg=1.0):
    """strict 6-DoF IK (damped least squares, numeric Jacobian, several seeds); -> (q, pos_err_m, rot_err_deg) or
    (None, ...) - unlike CS.ik_tool it never silently drops the orientation."""
    import NERO_V315_Consistency_Safety as CS

    def resid(q):
        T = CS.tool_pose(urdf, joints, q, base, tool_T)
        Ra = [row[:3] for row in T[:3]]
        dR = [[sum(R_des[k][i] * Ra[k][j] for k in range(3)) for j in range(3)] for i in range(3)]   # R_des^T Ra
        rv = _rotvec(dR)
        rw = [sum(R_des[i][k] * rv[k] for k in range(3)) for i in range(3)]                        # world frame
        return [T[0][3] - xyz[0], T[1][3] - xyz[1], T[2][3] - xyz[2]] + [w_rot * v for v in rw]

    best = (None, 1e9, 1e9)
    for sd in seeds:
        q = [max(lo + 1e-3, min(hi - 1e-3, v)) for v, (lo, hi) in zip(sd, limits)]
        lam = 1e-3
        r = resid(q)
        for _ in range(iters):
            e2 = sum(v * v for v in r)
            pe = math.sqrt(sum(v * v for v in r[:3]))
            re = math.degrees(math.sqrt(sum(v * v for v in r[3:])) / w_rot)
            if pe < tol_m and re < tol_deg:
                break
            Jm = []
            for j in range(len(q)):
                qd = list(q)
                qd[j] += 1e-6
                rd = resid(qd)
                Jm.append([(a - b) / 1e-6 for a, b in zip(rd, r)])
            # (J^T J + lam I) dq = -J^T r   (J stored column-wise in Jm)
            n = len(q)
            A = [[sum(Jm[i][k] * Jm[j][k] for k in range(6)) + (lam if i == j else 0.0) for j in range(n)] for i in range(n)]
            g = [-sum(Jm[i][k] * r[k] for k in range(6)) for i in range(n)]
            for c in range(n):                                   # Gauss-Jordan
                piv = max(range(c, n), key=lambda rr: abs(A[rr][c]))
                A[c], A[piv], g[c], g[piv] = A[piv], A[c], g[piv], g[c]
                if abs(A[c][c]) < 1e-15:
                    continue
                for rr in range(n):
                    if rr != c:
                        f = A[rr][c] / A[c][c]
                        A[rr] = [a - f * b for a, b in zip(A[rr], A[c])]
                        g[rr] -= f * g[c]
            dq = [g[i] / A[i][i] if abs(A[i][i]) > 1e-15 else 0.0 for i in range(n)]
            qn = [max(lo + 1e-4, min(hi - 1e-4, v + d)) for v, d, (lo, hi) in zip(q, dq, limits)]
            rn = resid(qn)
            if sum(v * v for v in rn) < e2:
                q, r, lam = qn, rn, max(1e-6, lam * 0.3)
            else:
                lam *= 10.0
                if lam > 1e4:
                    break
        pe = math.sqrt(sum(v * v for v in r[:3]))
        re = math.degrees(math.sqrt(sum(v * v for v in r[3:])) / w_rot)
        if pe < tol_m and re < tol_deg:
            return q, pe, re
        if pe + re / 1000.0 < best[1] + best[2] / 1000.0:
            best = (None, pe, re)
    return best


def rim_pinch_pose(cup_xy, phi_deg, pinch_depth_mm, cup=None, cup_base_z=TABLE_Z0):
    """TCP position + yaw for a rim pinch at rim angle phi (closing axis radial -> yaw = phi + 90 deg)."""
    g = SC.cup_geometry(cup)
    ph = math.radians(phi_deg)
    r = g["r_top"]
    xyz = (cup_xy[0] + r * math.cos(ph), cup_xy[1] + r * math.sin(ph), cup_base_z + g["h"] - pinch_depth_mm / 1000.0)
    return xyz, ph + math.pi / 2


def grasp_reachability(knobs=None, cup_from=None, cup_to=None, phis=range(0, 360, 15), urdf=None, log=None):
    """URDF IK study of the CS skill.  The cup is symmetric about its axis, so the rim angle used to GRASP cup_1
    (pregrasp -> grasp -> lift) and the rim angle used to PLACE it (place_above -> release over cup_2) are chosen
    independently (the gripper may turn the cup about its own axis while carrying it).  For every rim angle both
    gripper yaws (yaw, yaw+180: symmetric two-finger gripper) are tried; strict 6-DoF IK (orientation checked).
    -> {"grasp": [...], "place": [...]} each sorted by min joint-limit margin (deg), best first."""
    import NERO_V315_Consistency_Safety as CS
    k = dict(knobs or {})
    depth = float(k.get("cup_rim_pinch_depth_mm", 25.0))
    pre = float(k.get("cup_pregrasp_height_mm", 60.0)) / 1000.0
    clear = float(k.get("cup_transport_clearance_mm", 20.0)) / 1000.0
    gap = float(k.get("cup_release_gap_mm", 85.0)) / 1000.0
    zn = float(k.get("cup_nested_height_mm", 25.3)) / 1000.0
    g = SC.cup_geometry()
    cf = cup_from or SC.CUP_XY["obj_cup_1"]
    ct = cup_to or SC.CUP_XY["obj_cup_2"]
    U = urdf or CS.load_sim_urdf()
    J = [f"joint{i}" for i in range(1, 8)]
    T_tool = CS.tool_transform_from_sim(U)
    rel_z = g["h"] - depth / 1000.0                       # TCP height above the carried cup's base
    z_travel = TABLE_Z0 + g["h"] + clear + rel_z          # carried cup's base `clear` above the other cup's rim
    out = {"grasp": [], "place": []}
    for phi in phis:
        for stage, cup_xy in (("grasp", cf), ("place", ct)):
            xyz0, yaw0 = rim_pinch_pose(cup_xy, phi, depth)
            if stage == "grasp":
                chain = [("pregrasp", (xyz0[0], xyz0[1], xyz0[2] + pre)), ("grasp", xyz0),
                         ("lift", (xyz0[0], xyz0[1], z_travel))]
            else:
                chain = [("place_above", (xyz0[0], xyz0[1], z_travel)),
                         ("release", (xyz0[0], xyz0[1], TABLE_Z0 + zn + gap + rel_z))]
            best = None
            for flip in (0.0, math.pi):
                r_ = _reach_chain(U, J, T_tool, chain, yaw0 + flip)
                r_.update({"phi_deg": phi, "flip": bool(flip)})
                if best is None or (r_["ok"] and (not best["ok"] or r_["min_margin_deg"] > best["min_margin_deg"])):
                    best = r_
            out[stage].append(best)
            if log:
                log(f"[V3.61 cup study] {stage:<5} rim {phi:>3} deg -> " + (
                    f"OK margin {best['min_margin_deg']:.1f} deg (yaw {best['yaw_deg']:.0f}, {best['binding']})"
                    if best["ok"] else best["why"]))
    for stage in out:
        out[stage].sort(key=lambda r: (not r["ok"], -(r["min_margin_deg"] if r["min_margin_deg"] is not None else -1e9)))
    return out


JOINT_LIMITS = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95),
                (-1.5708, 1.5708)]


def _reach_chain(U, J, T_tool, chain, yaw):
    """strict IK along a chain of TCP positions with one top-down yaw; seeds chained from the previous solution."""
    R_tcp = _rot_xyzw(tcp_quat_for_yaw(yaw))
    q_prev, worst, ok, why, binding = None, None, True, "", ""
    for name, xyz in chain:
        a1 = math.atan2(xyz[1], xyz[0])
        a1 = a1 + math.pi if a1 < 0 else a1 - math.pi
        seeds = ([q_prev] if q_prev else []) + [[a1, j2, 0.0, j4, 0.0, 0.5, j7] for j2 in (0.2, 0.6)
                                                for j4 in (0.8, 1.6) for j7 in (0.0, 1.0, -1.0)]
        q, pe, re = pose_ik(U, J, JOINT_LIMITS, seeds, T_tool, xyz, R_tcp)
        if q is None:
            ok, why = False, f"{name}: no 6-DoF IK (pos {pe * 1000:.1f} mm, rot {re:.1f} deg)"
            break
        ms = [min(v - lo, hi - v) for v, (lo, hi) in zip(q, JOINT_LIMITS)]
        kj = min(range(7), key=lambda i: ms[i])
        if worst is None or ms[kj] < worst:
            worst, binding = ms[kj], f"J{kj + 1} at {name}"
        q_prev = q
    return {"yaw_deg": round(math.degrees(yaw) % 360, 1), "ok": ok, "why": why, "binding": binding,
            "min_margin_deg": round(math.degrees(worst), 1) if ok else None}


def base_dir_deg(cup_xy):
    """direction from a cup's axis to the robot base (deg)"""
    return math.degrees(math.atan2(-cup_xy[1], -cup_xy[0]))


def rim_angle_abs(rel_deg, cup_xy):
    """knob rim angle (relative to the cup -> base direction) -> absolute rim angle (deg, 0..360)"""
    return (base_dir_deg(cup_xy) + float(rel_deg)) % 360.0


def plan_cup_stack(knobs=None, cup_from=None, cup_to=None, urdf=None):
    """CS skill plan from the knobs: TCP waypoints + yaw for pregrasp / grasp / lift / place_above / release, each
    checked with strict 6-DoF IK (both symmetric gripper yaws).  -> dict(ok, waypoints, grasp, place, ...)"""
    import NERO_V315_Consistency_Safety as CS
    k = dict(knobs or {})
    cf = cup_from or SC.CUP_XY["obj_cup_1"]
    ct = cup_to or SC.CUP_XY["obj_cup_2"]
    phi_g = rim_angle_abs(k.get("cup_grasp_rim_angle_deg", 40.0), cf)
    phi_p = rim_angle_abs(k.get("cup_place_rim_angle_deg", -55.0), ct)
    U = urdf or CS.load_sim_urdf()
    r = grasp_reachability(k, cf, ct, phis=(phi_g,), urdf=U)
    rp = grasp_reachability(k, cf, ct, phis=(phi_p,), urdf=U)
    g, p_ = r["grasp"][0], rp["place"][0]
    g_geo = SC.cup_geometry()
    depth = float(k.get("cup_rim_pinch_depth_mm", 25.0))
    rel_z = g_geo["h"] - depth / 1000.0
    z_travel = TABLE_Z0 + g_geo["h"] + float(k.get("cup_transport_clearance_mm", 20.0)) / 1000.0 + rel_z
    gx, _ = rim_pinch_pose(cf, phi_g, depth)
    px, _ = rim_pinch_pose(ct, phi_p, depth)
    z_rel = TABLE_Z0 + float(k.get("cup_nested_height_mm", 25.3)) / 1000.0 + float(k.get("cup_release_gap_mm", 85.0)) / 1000.0 + rel_z
    wps = [("pregrasp", (gx[0], gx[1], gx[2] + float(k.get("cup_pregrasp_height_mm", 60.0)) / 1000.0), g["yaw_deg"]),
           ("grasp", gx, g["yaw_deg"]), ("lift", (gx[0], gx[1], z_travel), g["yaw_deg"]),
           ("place_above", (px[0], px[1], z_travel), p_["yaw_deg"]), ("release", (px[0], px[1], z_rel), p_["yaw_deg"])]
    return {"ok": bool(g["ok"] and p_["ok"]), "grasp": g, "place": p_, "grasp_rim_abs_deg": round(phi_g, 1),
            "place_rim_abs_deg": round(phi_p, 1),
            "waypoints": [{"name": n, "tcp_xyz_m": [round(v, 4) for v in xyz], "yaw_deg": y} for n, xyz, y in wps],
            "release_checks": {"xy_tol_mm": float(k.get("cup_release_xy_tol_mm", 7.5)),
                               "tilt_tol_deg": float(k.get("cup_release_tilt_tol_deg", 3.0))}}


def run_study(log=print):
    """Full V3.61 study (a few minutes): release sweep + parameter choice + grasp reachability."""
    if not mujoco_available():
        log("[WARN] [V3.61 cup study] MuJoCo 不可用（pip install mujoco）：跳过物理试验")
        return None
    sw = release_sweep(log=log)
    par = choose_release_params(sw)
    log(f"[OK] [V3.61 cup study] nested height {sw['z_nested_mm']} mm; chosen {par}")
    reach = grasp_reachability({"cup_release_gap_mm": par["release_gap_mm"], "cup_nested_height_mm": sw["z_nested_mm"]},
                               log=log)
    for stage in ("grasp", "place"):
        good = [r for r in reach[stage] if r["ok"]]
        log(f"[OK] [V3.61 cup study] {stage}: reachable rim angles {len(good)}/{len(reach[stage])}; best {good[:2]}")
    return {"sweep": sw, "params": par, "reach": reach}


if __name__ == "__main__":
    import json
    res = run_study()
    if res:
        print(json.dumps({"params": res["params"], "best_grasp": res["reach"]["grasp"][:3],
                          "best_place": res["reach"]["place"][:3]}, indent=2))
