#!/usr/bin/env python3
"""Offline tests for V3.03 (no ROS / MuJoCo / GPU).

D   orient_for_slot: a slot-above yaw that is reachable at hover height but NOT at
    slot-entry depth must be rejected by the descent-corridor gate (V3.02 bug,
    log 2026-10-21 egg_2 -> slot_3).  D0 shows V3.02 behaviour (gate off) picks it.
E1  move_to_slot_entry: Cartesian descent works -> layer=cartesian.
E2  Cartesian + cuRobo both fail for the current yaw -> rotate at hover (yaw family)
    -> Cartesian descent succeeds.
E3  nothing reaches the entry depth -> structured SLOT_ENTRY_UNREACHABLE, stage reset
    to LIFTED, no raw exception (V3.02 leaked one).
Real backend source is extracted and executed against mocks.
"""
import ast, math, os, sys, textwrap, time, types
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
H = (HERE / "test_v3_02_orient_harness.py").read_text().split("# A: feasible only")[0]
G = {"__file__": str(HERE / "test_v3_02_orient_harness.py")}
exec(compile(H, "harness", "exec"), G)
q2mat, run = G["q2mat"], G["run"]
SRC = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()


def heading(q):
    """Compass heading (deg) of the horizontal TCP approach axis (tool z)."""
    R = q2mat(q)
    return math.degrees(math.atan2(R[1, 2], R[0, 2]))


# ---------------------------------------------------------------- D
# High goals: any yaw.  Low goals (TCP z < 0.20): only approach headings in [-60, 0] deg.
def feas_D(core, xyz, q):
    if xyz[2] >= 0.20:
        return True
    return -60.0 <= heading(q) <= 0.0

os.environ["NERO_V303_CORRIDOR"] = "0"
out, core, env = run(feas_D)
choice0 = [m for m in core.log if "slot_above_transport | choice=" in m]
pose_goals0 = [m[1] for m in core.motions if m[0] == "pose"]
h0 = heading(pose_goals0[-1][1])
print(f"[OK] D0 (gate OFF = V3.02): slot-above choice heading={h0:.0f}deg ->",
      "would FAIL at entry" if not (-60 <= h0 <= 0) else "happens to pass")

os.environ["NERO_V303_CORRIDOR"] = "1"
out, core, env = run(feas_D)
assert out[0] == "FALLTHROUGH" and out[2], (out, core.log[-8:])
assert any("[V3.03 CORRIDOR REJECT] slot_above_transport" in m for m in core.log), core.log
assert any("[V3.03 CORRIDOR POST-CHECK OK]" in m for m in core.log), core.log
h1 = heading([m[1] for m in core.motions if m[0] == "pose"][-1][1])
assert -60.0 <= h1 <= 0.0, h1
print(f"[OK] D  (gate ON): rejected yaws without a descent corridor; chosen heading={h1:.0f}deg;",
      [m for m in core.log if m.startswith("[V3.03 CORRIDOR] slot_above")][0].split("|")[1].strip())
del os.environ["NERO_V303_CORRIDOR"]

# ---------------------------------------------------------------- E
body0 = SRC.index("            llm_require_arm_controller()\n            rel0 = [float(obj0[i]) - float(tcp0[i]) for i in range(3)]")
body1 = SRC.index("            # One deterministic micro-correction is allowed")
BODY = SRC[body0:body1] + "            return ('ENTRY_OK', layer303, target_tcp)\n"
ENTRY = "def _entry(env):\n    globals().update(env)\n" + textwrap.indent(textwrap.dedent(BODY), "    ")

mod = ast.parse(SRC)
want = {"_v27_quat_normalize", "_v27_quat_multiply", "_v271_quat_conjugate", "_v271_axis_line_error_deg",
        "_v271_unit", "_v271_dot", "_v271_norm"}
HELP = "import math\nV271_SLOT_ENTRY_AXIS=(0.0,0.0,-1.0)\n" + "\n".join(
    ast.get_source_segment(SRC, n) for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in want)
HG = {}
exec(HELP, HG)

SLOT3 = [-0.39, 0.123, 0.043]
ENTRY_Z = 0.123
# vertical-egg TCP orientation from the real log (TCP x = egg axis = world z)
Q_VERT = [-0.09054, -0.70575, -0.09273, 0.69651]
R_LOCAL = (q2mat(Q_VERT).T @ np.array([-0.027, 0.009, -0.004])).tolist()


class EntryCore:
    def __init__(self, feas):
        self.feas = feas; self.log = []; self.motions = []
        self.tcpq = list(Q_VERT)
        obj = np.array([SLOT3[0], SLOT3[1], 0.223])
        self.tcp = (obj - q2mat(self.tcpq) @ np.array(R_LOCAL)).tolist()
    def emit(self, m): self.log.append(m)
    def object_pose(self, name):
        return (np.array(self.tcp) + q2mat(self.tcpq) @ np.array(R_LOCAL)).tolist(), list(self.tcpq)
    def current_tcp_pose(self): return list(self.tcp), list(self.tcpq)
    def _current_joint_state_copy(self): return {"q": [0.0]*7}
    def _seed_with_arm_positions(self, base, arm): return {"q": list(arm)}
    def _compute_ik(self, xyz, q, **kw):
        return ([0.1]*7, "SUCCESS") if self.feas(xyz, q) else (None, "NO_IK")
    def _cartesian_approach_plan(self, xyz, q):
        if not self.feas(xyz, q):
            raise RuntimeError("CARTESIAN APPROACH BLOCKED: only 40.0% collision-free")
        pts = [types.SimpleNamespace(positions=[0.1]*7), types.SimpleNamespace(positions=[0.12]*7)]
        return types.SimpleNamespace(joint_trajectory=types.SimpleNamespace(points=pts), target=(list(xyz), list(q)))
    def _execute_trajectory(self, traj):
        self.tcp, self.tcpq = traj.target; self.motions.append(("cart", heading(self.tcpq), round(self.tcp[2], 3)))
        return True
    def move_to_pose(self, xyz, q, **kw):
        if not self.feas(xyz, q):
            raise RuntimeError("GOAL_IK_UNREACHABLE: cuRobo plan_pose returned None 3x in a row (mock)")
        self.tcp, self.tcpq = list(xyz), list(q); self.motions.append(("pose", heading(q), round(xyz[2], 3)))
        return True


def run_entry(feas):
    core = EntryCore(feas)
    hold = types.SimpleNamespace(value=True); trips = types.SimpleNamespace(value=0)
    state = {"stage": "PLACEMENT_ORIENTED"}

    def run_motion(label, fn, target, tol, preserve=False):   # mirrors _v24_run_motion re-raise policy
        try:
            ok = bool(fn())
        except Exception as exc:
            if "CARTESIAN APPROACH BLOCKED" not in str(exc):
                raise
            return {"reason": "CARTESIAN_BLOCKED", "legacy_joint_verify_error": str(exc)}, False, []
        d = math.dist(core.tcp, target)
        return {"tcp_error_mm": d*1000}, ok and d <= tol, []

    def set_state(stage=None, **kw):
        if stage: state["stage"] = stage

    def result(name, ok, contract, sb, ver, warn=None, extra=None, next_stage=None, recommended_next_actions=None, **k):
        return ("RESULT", ok, ver, recommended_next_actions)

    obj0, oq0 = core.object_pose("egg_2"); tcp0, tq0 = core.current_tcp_pose()
    env = dict(HG)
    env.update(math=math, time=time, skill_core=core, obj="egg_2", slot_name="slot_3", slot=SLOT3,
               plan={"entry_object_center_z_m": ENTRY_Z}, obj0=obj0, tcp0=tcp0, tcp_q0=tq0, safety0=0,
               hold_active_shared=hold, safety_trip_count=trips, V271_AXIS_TOL_DEG=10.0,
               _v27_egg_major_axis_from_quat=lambda q: {"world_axis": (q2mat(q) @ np.array([1.0, 0, 0])).tolist()},
               llm_require_arm_controller=lambda: None, _v24_run_motion=run_motion,
               _v218_slow_trajectory_in_place=lambda traj, factor=2.0: None, _v24_set_state=set_state,
               V24_STAGE_LIFTED="LIFTED", V271_PLACEMENT_RUNTIME={"oriented": {"egg_2": {}}}, _v24_result=result,
               contract={}, state_before={}, name="move_to_slot_entry", t0=time.perf_counter())
    g = {}
    exec(ENTRY, g)
    return g["_entry"](env), core, state


H0 = heading(Q_VERT)
# E1: everything feasible
out, core, st = run_entry(lambda xyz, q: True)
assert out[0] == "ENTRY_OK" and out[1] == "cartesian", out
assert core.motions[0][0] == "cart" and abs(core.object_pose("e")[0][2] - ENTRY_Z) < 1e-6
print("[OK] E1: straight Cartesian descent is the first layer:", core.motions)

# E2: low poses only reachable for headings rotated >= 50 deg away from the current one
def feas_E2(xyz, q):
    if xyz[2] >= 0.20:
        return True
    d = (heading(q) - H0 + 180) % 360 - 180
    return abs(d) >= 50
out, core, st = run_entry(feas_E2)
assert out[0] == "ENTRY_OK" and out[1].startswith("yaw") and out[1].endswith("+cartesian"), (out, core.log)
assert core.motions[0][0] == "pose" and core.motions[0][2] > 0.2, core.motions   # rotate at HOVER first
assert core.motions[-1][0] == "cart"
print(f"[OK] E2: cart + cuRobo failed -> rotate at hover -> descend: layer={out[1]} motions={core.motions}")

# E3: nothing low is reachable
out, core, st = run_entry(lambda xyz, q: xyz[2] >= 0.20)
assert out[0] == "RESULT" and out[1] is False, out
assert out[2]["reason"] == "SLOT_ENTRY_UNREACHABLE" and st["stage"] == "LIFTED", (out, st)
assert out[3] and out[3][0].startswith("plan_placement"), out[3]
assert all(m[2] > 0.2 for m in core.motions), core.motions        # never went low
print("[OK] E3: structured SLOT_ENTRY_UNREACHABLE, stage=LIFTED, next=", out[3], "| attempts:", len(out[2]["v303_attempts"]))
