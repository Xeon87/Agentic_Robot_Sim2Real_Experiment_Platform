#!/usr/bin/env python3
"""Offline harness for V3.02 orient_for_slot strategy code.

Runs the REAL backend source (V2-24 helpers + V3.02 block, extracted verbatim)
against a mocked skill core whose FK is computed from config/nero_7dof_with_tcp.urdf
(the cuRobo URDF).  No ROS / MuJoCo / GPU needed.

Scenarios
  A  whole-arm: only non-zero yaws are IK/plan-feasible  -> must succeed via yaw family
  B  whole-arm infeasible everywhere -> safe-high hold (batch-FK search) -> slot-above
  C  nothing feasible -> clean failure, NO q0 motion (legacy q0 is opt-in only)
"""
import ast, math, os, sys, textwrap, time, types
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()

# ---------------- URDF FK (same chain cuRobo uses) ----------------
_urdf = ET.parse(ROOT / "config" / "nero_7dof_with_tcp.urdf").getroot()
_child = {j.find("parent").get("link"): j for j in _urdf.findall("joint")}
_chain, _link = [], "base_link"
while _link in _child:
    _j = _child[_link]; _chain.append(_j); _link = _j.find("child").get("link")

def _rpy(a, b, c):
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return np.array([[cc*cb, cc*sb*sa-sc*ca, cc*sb*ca+sc*sa], [sc*cb, sc*sb*sa+cc*ca, sc*sb*ca-cc*sa], [-sb, cb*sa, cb*ca]])

def fk(q):
    T = np.eye(4); k = 0
    for j in _chain:
        o = j.find("origin"); xyz = [float(v) for v in o.get("xyz").split()]
        a = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
        A = np.eye(4); A[:3, :3] = _rpy(*a); A[:3, 3] = xyz; T = T @ A
        if j.get("type") == "revolute":
            c, s = np.cos(q[k]), np.sin(q[k]); R = np.eye(4); R[:2, :2] = [[c, -s], [s, c]]; T = T @ R; k += 1
    return T

def mat2q(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2; w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return [x, y, z, w]

def q2mat(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

# ---------------- extract real backend code ----------------
h0 = SRC.index("            def _v224_vnorm(v):")
h1 = SRC.index("            # STRATEGY 1: 1-2 JOINT WRIST-DOMINANT SEARCH")
h1 = SRC.rindex("            # ====", h0, h1)
b0 = SRC.index("            # ============================================================\n            # V3.02 YAW-FREE VERTICAL TARGETS")
b1 = SRC.index("            # Final physical verify at slot-above.")
body = SRC[h0:h1] + SRC[b0:b1] + "            return ('FALLTHROUGH', selected_strategy24, at_slot_above302, zero_used24)\n"
FUNC = "def _orient(env):\n" + "    globals().update(env)\n" + textwrap.indent(textwrap.dedent(body), "    ")
JOINT_LIMITS_RAD = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708)]  # = backend

class Pt:
    def __init__(self, q, t):
        self.positions = list(q); self.time_from_start = types.SimpleNamespace(sec=int(t), nanosec=int((t-int(t))*1e9))

class Traj:
    def __init__(self, qs, target=None):
        self.joint_trajectory = types.SimpleNamespace(points=[Pt(q, 0.5*i) for i, q in enumerate(qs)])
        self.target = target

class Core:
    def __init__(self, q_start, r_local, feasible):
        self.q = list(q_start); T = fk(self.q)
        self.tcp = T[:3, 3].tolist(); self.tcpq = mat2q(T[:3, :3])
        self.r_local = r_local; self.feasible = feasible; self.log = []; self.motions = []
        self._gpu_client = self
    def emit(self, m): self.log.append(m)
    def object_pose(self, name):
        R = q2mat(self.tcpq); p = (np.array(self.tcp) + R @ np.array(self.r_local)).tolist()
        return p, list(self.tcpq)           # egg frame == TCP frame; egg axis = local x
    def current_tcp_pose(self): return list(self.tcp), list(self.tcpq)
    def _current_joint_state_copy(self): return {"q": list(self.q)}
    def _arm_positions_from_joint_state(self, js): return list(js["q"])
    def _v70_natural_seed_variants(self, live, arm, reorientation=False): return [("current", live), ("wrist_center", live)]
    def _v70_endpoint_score(self, a, b): return ((0.0, 0.0, 0.0, 0.0, 0.0), {})
    def _seed_with_arm_positions(self, base, arm): return {"q": list(arm)}
    def _compute_ik(self, xyz, q, **kw):
        # V3.03 harness: a feasible IK returns the seed branch (continuous), so the
        # chained descent-corridor check sees zero joint steps unless feasible() says no.
        seed = kw.get("seed_joint_state") or {"q": list(self.q)}
        return (list(seed["q"]), "SUCCESS") if self.feasible(self, xyz, q) else (None, "NO_IK")
    def _plan_pose(self, xyz, q, **kw):
        if not self.feasible(self, xyz, q):
            raise RuntimeError("cuRobo exhausted 10 trajectory attempts (mock)")
        return Traj([list(self.q), [0.0, 0.3, 0.0, 1.2, 0.0, 0.3, 0.0]], target=(list(xyz), list(q)))
    def fk_batch(self, joint_names, q_batch, timeout_s=20.0):
        out = []
        for q in q_batch:
            T = fk(q); out.append({"position": T[:3, 3].tolist(), "quat_xyzw": mat2q(T[:3, :3])})
        return {"ok": True, "poses": out}
    def _v71_state_is_valid(self, q): return True, "OK"
    def _make_incremental_joint_trajectory(self, qs, xyz): return Traj(qs)
    def _execute_trajectory(self, traj):
        if traj.target is not None:
            self.tcp, self.tcpq = traj.target; self.motions.append(("pose", traj.target))
        else:
            qend = traj.joint_trajectory.points[-1].positions; self.q = list(qend)
            T = fk(self.q); self.tcp = T[:3, 3].tolist(); self.tcpq = mat2q(T[:3, :3])
            self.motions.append(("joint", [round(math.degrees(v), 1) for v in qend]))
        return True

def run(feasible, q0_flag="0"):
    os.environ["NERO_V302_ALLOW_Q0_FALLBACK"] = q0_flag
    # lifted egg_1 pose from the real 14:52 log, egg ~ under TCP, axis along tool x
    q_start = [math.radians(v) for v in [-17.2, 35.9, -4.6, 105.7, -15.2, -15.2, -53.6]]
    # Use a top-down lift pose instead: solve by taking a known grasp-like config
    q_start = [math.radians(v) for v in [-25.0, 30.0, 0.0, 95.0, 0.0, 55.0 - 3.5, 0.0]]
    core = Core(q_start, r_local=[0.004, 0.012, 0.028], feasible=feasible)
    assert core.tcp[2] < 0.38, core.tcp
    hold = types.SimpleNamespace(value=True); trips = types.SimpleNamespace(value=0)
    def run_motion(label, fn, target, tol, preserve=False):
        return {"cartesian_ok": True}, bool(fn()), []
    def result(name, ok, *a, **k):
        return ("RESULT", ok, a[2] if len(a) > 2 else None)
    def egg_axis(q):
        return {"world_axis": (q2mat(q) @ np.array([1.0, 0.0, 0.0])).tolist()}
    env = dict(
        math=math, os=os, time=time, skill_core=core, obj="egg_1", slot_name="slot_1",
        slot24=[-0.51, 0.123, 0.043], plan24={"safe_orientation_xyz_m": [-0.4225, 0.0, 0.245], "entry_object_center_z_m": 0.122},
        safety0=0, warnings24=[], verification24={}, wrist_ok24=False, t0=time.perf_counter(), name="orient_for_slot",
        contract={}, state_before={}, hold_active_shared=hold, safety_trip_count=trips, V271_AXIS_TOL_DEG=10.0,
        JOINT_LIMITS_RAD=JOINT_LIMITS_RAD, _v24_result=result, _v24_run_motion=run_motion,
        _v27_egg_major_axis_from_quat=egg_axis,
    )
    # V3.41 module-level joint-limit margin helpers (real backend source)
    m341 = SRC[SRC.index("def v341_limit_margins_deg(q):"):SRC.index("# V2 reference speeds")]
    g341 = {"math": math, "os": os, "JOINT_LIMITS_RAD": JOINT_LIMITS_RAD}
    exec(m341, g341)
    env.update({k: v for k, v in g341.items() if k.startswith(("v341_", "V341_"))})
    g = {}
    exec(FUNC, g)
    out = g["_orient"](env)
    return out, core, env

def yaw_of(q):  # TCP x-axis heading as a crude yaw tag
    R = q2mat(q); return math.degrees(math.atan2(R[1, 2], R[0, 2]))

# A: feasible only for candidates whose approach (TCP z) points to -y side (excludes yaw 0 goal)
first_goal = {}
def feas_A(core, xyz, q):
    R = q2mat(q)
    return R[1, 2] < -0.3
os.environ["NERO_V341_LIMIT_MARGIN"] = "0"          # V3.40 selection (no joint-limit margin)
out, core, env = run(feas_A)
os.environ.pop("NERO_V341_LIMIT_MARGIN", None)
assert out[0] == "FALLTHROUGH" and out[1] == "v302_whole_arm_yaw_family" and out[2], out
assert any("[V3.02 VERTICAL GOAL PLAN] whole_arm_yaw_family" in m for m in core.log)
assert not any(m[0] == "joint" for m in core.motions), core.motions
print("[OK] A: whole-arm succeeds only through a non-zero yaw; no joint-space fallback;",
      [m for m in core.log if "choice=" in m][0].split("|")[1].strip())

# A2 (V3.41): same, with the joint-limit margin: the slot-above poses reachable from this branch keep J6 only
# 2.9 deg from its limit -> rejected; the safe-high recovery gives a branch >= 8 deg from every limit
out, core, env = run(feas_A)
assert out[0] == "FALLTHROUGH" and out[1] == "v302_whole_arm_yaw_family+v302_safe_high_recovery" and out[2], out
assert any("[V3.41 LIMIT MARGIN] slot_above_transport" in m and "none - kept all" in m for m in core.log)
assert any("J6 2.9 deg from its limit" in m for m in core.log)
assert any("[V3.41 CORRIDOR MARGIN] recovery_safe_high_to_slot_above | every level >= 8 deg" in m for m in core.log)
print("[OK] A2 (V3.41): slot-above branch with J6 2.9 deg margin rejected -> safe-high recovery, >= 8 deg everywhere")

# B: nothing reachable from the low lifted start; everything reachable once the arm is high
def feas_B(core, xyz, q):
    # V3.03: once the arm has been high (hold pose), goals stay reachable (IK does not
    # depend on where the arm currently is; only the low lifted start was infeasible).
    if core.tcp[2] > 0.38:
        core.been_high = True
    return bool(getattr(core, "been_high", False))
out, core, env = run(feas_B)
assert out[0] == "FALLTHROUGH" and out[1] == "v302_safe_high_hold_then_curobo" and out[2], (out, core.log[-6:])
hold_q = [m for m in core.motions if m[0] == "joint"][0][1]
assert abs(hold_q[3]) > 60 and abs(hold_q[1]) >= 10, hold_q   # bent elbow, J2 != 0 -> not q0
assert not any("LEGACY Q0" in m for m in core.log)
print("[OK] B: safe-high hold used, q_deg =", hold_q, "| then slot-above via yaw family")
print("     ", [m for m in core.log if "SAFE-HIGH HOLD SEARCH" in m][0])

# C: nothing feasible -> clean failure, no motion at all (q0 is opt-in)
out, core, env = run(lambda c, xyz, q: False)
assert out[0] == "RESULT" and out[1] is False, out
assert core.motions == [] or all(m[0] == "joint" for m in core.motions)
assert not any("LEGACY Q0" in m for m in core.log)
print("[OK] C: all strategies fail -> structured failure; legacy q0 not used; motions =", core.motions)

# C2: same with legacy flag -> q0 attempted
out, core, env = run(lambda c, xyz, q: False, q0_flag="1")
assert any("LEGACY Q0" in m for m in core.log)
print("[OK] C2: NERO_V302_ALLOW_Q0_FALLBACK=1 re-enables the legacy q0 path")
