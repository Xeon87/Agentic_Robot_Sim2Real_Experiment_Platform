#!/usr/bin/env python3
"""Offline tests for V3.12.1 held-egg recovery (no ROS / MuJoCo / GPU).

The V3.12 camera run: lift(0.13) corridor 97.9 % -> BACKEND_EXCEPTION; lift(0.04) left the egg low; orient_for_slot
failed every strategy; no tool to put the egg back -> 2 eggs never attempted.

H1  lift ladder: blocked 130 mm -> 108 mm used; all blocked -> CARTESIAN APPROACH BLOCKED (structured, not raised as
    BACKEND_EXCEPTION); execution errors are NOT retried
H2  lift handler: requested 130 mm blocked -> reports the 108 mm actually lifted; all blocked -> reason
    V3121_LIFT_CORRIDOR_BLOCKED + next return_to_pick_location; 108 mm unblocked -> unchanged behaviour
H3  orient pre-raise: TCP 92 mm -> raised to 160 mm before the strategies; already high -> nothing
H4  descent precheck from the planned pregrasp END state: 25 % -> reject; 100 % -> ok; start state = trajectory end
H5  V3.07 q0 retreat ladder: 73 -> 340 mm blocked -> shorter retreat; nothing plannable -> q0 directly
H6  held egg pose from TCP x grasp transform follows TCP translation + rotation
H7  return_to_pick_location: lifted egg straight down at the pick pose, released, retreat, state OBSERVED + next egg;
    egg not held -> precondition failure; failed release -> RETURN_TO_PICK_LOCATION_FAILED
H8  wiring: preflight uses the precheck, approach-block retreat, recapture reports RECAPTURE_CLEAR_VIEW_FAILED,
    camera scene reports in_gripper, agent tool + rule, contract entry
"""
import math, os, sys, textwrap, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
AG = (ROOT / "NERO_V2_7_1_Agent.py").read_text()


def block(start, end):
    a = SRC.index(start)
    b = SRC.index(end, a + len(start))
    return SRC[a:b]


HELPERS = textwrap.dedent(block("    # V3.12.1 RECOVERY HELPERS", "    def llm_tool_runner(name, args):"))
NS = lambda o: types.SimpleNamespace(value=o)


class Core:
    def __init__(self, z=0.0525, blocked=(), exec_error=None):
        self.z, self.xy, self.q = z, [-0.454, -0.220], [0.84395, 0.53643, 0.0, 0.0]
        self.blocked, self.exec_error, self.logs, self.calls = set(blocked), exec_error, [], []

    def emit(self, m):
        self.logs.append(m)

    def current_tcp_pose(self):
        return [self.xy[0], self.xy[1], self.z], list(self.q)

    def bio_lift(self, dz):
        self.calls.append(("lift", round(dz, 3)))
        if round(dz, 3) in self.blocked:
            raise RuntimeError("Cartesian corridor only 97.9% collision-free")
        if self.exec_error:
            raise RuntimeError(self.exec_error)
        self.z += dz
        return True


def helpers(core, **extra):
    g = dict(math=math, os=os, sys=sys, skill_core=core, hold_active_shared=NS(True),
             V27_GRASP_RUNTIME={"plans": {}, "selected": {}}, _v218_slow_trajectory_in_place=lambda t, factor=1.0: None)
    g.update(extra)
    exec(HELPERS, g)
    return g


# ---------------------------------------------------------------- H1 ladder
c = Core(blocked={0.13})
g = helpers(c)
used = {}
h = g["_v3121_lift_ladder"](0.13, lambda hh, tr: used.update(h=hh, tr=tr), "lift egg_1")
assert h == 0.108 and used["tr"] == [130.0, 108.0] and abs(c.z - (0.0525 + 0.108)) < 1e-9, (h, used)
assert any("[V3.12.1 LIFT FALLBACK]" in m and "lifted 108 mm" in m for m in c.logs)
c = Core(blocked={0.13, 0.108, 0.09, 0.07, 0.05})
g = helpers(c)
try:
    g["_v3121_lift_ladder"](0.13, None, "lift egg_1")
    raise AssertionError("expected block")
except RuntimeError as e:
    assert "CARTESIAN APPROACH BLOCKED" in str(e) and "every height" in str(e)
assert c.z == 0.0525
c = Core(exec_error="PHYSICAL TRACKING MISMATCH joint4")
g = helpers(c)
try:
    g["_v3121_lift_ladder"](0.108)
    raise AssertionError("expected raise")
except RuntimeError as e:
    assert "TRACKING" in str(e) and len(c.calls) == 1            # execution faults are never retried
print("[H1] lift ladder: 130 blocked -> 108 used; all blocked -> structured block; exec faults not retried")

# ---------------------------------------------------------------- H2 lift handler
LIFT = block('        if name == "lift":\n', '        if name == "move_above_receptacle":')
LIFT_FN = "def _lift(name, args, state_before, contract, t0):\n" + textwrap.indent(textwrap.dedent(LIFT), "    ") + "    return None\n"


def run_lift(blocked, height):
    core = Core(blocked=blocked)
    obj = {"z": 0.022}
    core.object_pose = lambda n: ([-0.455, -0.224, obj["z"]], [0, 0, 0, 1])
    _orig = core.bio_lift

    def lift(dz):
        r = _orig(dz)
        obj["z"] += dz
        return r
    core.bio_lift = lift

    def run_motion(label, fn, target, tol, keep):
        try:
            fn()
            return {"tcp_error_mm": 0.5}, True, []
        except RuntimeError as e:
            if "CARTESIAN APPROACH BLOCKED" not in str(e):
                raise
            return {"legacy_joint_verify_error": str(e)}, False, []

    res = {}

    def result(name, ok, contract, sb, ver, warnings=None, extra=None, next_stage=None, recommended_next_actions=None, **k):
        res.update(ok=ok, ver=ver, extra=extra, next_stage=next_stage, next=recommended_next_actions)
        return res

    g = helpers(core)
    g.update(time=types.SimpleNamespace(sleep=lambda s: None, perf_counter=lambda: 0.0), llm_require_arm_controller=lambda: None,
             _v24_run_motion=run_motion, _v24_result=result, _v24_sub3=lambda a, b: [a[i] - b[i] for i in range(3)],
             _v24_xyz_distance=lambda a, b: math.dist(a, b), V24_STAGE_GRASPED="GRASP_ESTABLISHED", V24_STAGE_LIFTED="LIFTED",
             _v24_precondition_failure=lambda *a, **k: {"ok": False, "pre": True})
    exec(LIFT_FN, g)
    g["_lift"]("lift", {"height_m": height}, {"stage": "GRASP_ESTABLISHED", "active_object": "egg_1"}, {}, 0.0)
    return res, core


r, core = run_lift({0.13}, 0.13)
assert r["ok"] and r["next_stage"] == "LIFTED" and r["ver"]["requested_lift_mm"] == 108.0, r
assert r["ver"]["v3121_lift_heights_tried_mm"] == [130.0, 108.0] and r["ver"]["v3121_requested_lift_mm"] == 130.0
r, core = run_lift({0.13, 0.108, 0.09, 0.07, 0.05}, 0.13)
assert (not r["ok"]) and r["ver"]["reason"] == "V3121_LIFT_CORRIDOR_BLOCKED" and r["next"] == ["return_to_pick_location"], r
r, core = run_lift(set(), 0.108)
assert r["ok"] and r["ver"]["requested_lift_mm"] == 108.0 and "v3121_lift_heights_tried_mm" not in r["ver"]
print("[H2] lift: 130 blocked -> reports 108 lifted; all blocked -> V3121_LIFT_CORRIDOR_BLOCKED + return; normal unchanged")

# ---------------------------------------------------------------- H3 orient pre-raise
c = Core(z=0.092)
g = helpers(c)
rep = g["_v3121_pre_raise_for_orient"]("egg_1")
assert rep["ok"] and abs(c.z - 0.160) < 1e-6 and rep["raised_mm"] == 68.0, rep
c = Core(z=0.092, blocked={0.068})
g = helpers(c)
rep = g["_v3121_pre_raise_for_orient"]("egg_1")
assert rep["ok"] and rep["raised_mm"] == 51.0, rep                        # 75 % of the need
c = Core(z=0.183)
assert helpers(c)["_v3121_pre_raise_for_orient"]("egg_2") is None and not c.calls
ORIENT = block('        if name == "orient_for_slot":', "            # ---------------- local math / FK helpers")
assert "_v3121_pre_raise_for_orient(obj)" in ORIENT and ORIENT.index("V224_GRASP_NOT_ACTIVE") < ORIENT.index("_v3121_pre_raise")
print("[H3] orient pre-raise: 92 -> 160 mm (or partial when blocked), high egg untouched, wired after the hold check")

# ---------------------------------------------------------------- H4 descent precheck
fake = types.ModuleType("fake_core_v3121")


class _Obj(types.SimpleNamespace):
    pass


class _Req:
    def __init__(self):
        self.header = _Obj(frame_id="")


fake.GetCartesianPath = types.SimpleNamespace(Request=_Req)
fake.RobotState = lambda: _Obj()
fake.Pose = lambda: _Obj(position=_Obj(), orientation=_Obj())
fake.ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
fake.CARTESIAN_MAX_STEP = 0.005
sys.modules["fake_core_v3121"] = fake
FakeCoreCls = type("FakeCore", (), {"__module__": "fake_core_v3121"})
fc = FakeCoreCls()
seen = {}
frac = {"v": 0.25}


def call_async(req):
    seen["req"] = req
    return _Obj(error_code=_Obj(val=1), fraction=frac["v"])


fc.cartesian_client = _Obj(call_async=call_async)
fc.wait_future = lambda fut, t, label: fut
fc._v71_full_joint_state_for_arm = lambda arm: _Obj(arm=list(arm))
g = helpers(fc)
traj = _Obj(joint_trajectory=_Obj(joint_names=list(reversed(fake.ARM_JOINT_NAMES)),
                                  points=[_Obj(positions=[0.0] * 7), _Obj(positions=[7, 6, 5, 4, 3, 2, 1])]))
cand = {"quat_xyzw": [0.84394797, 0.53642505, 0.0, 0.0], "grasp_xyz_m": [-0.453167, -0.21818, 0.05245]}
ok, why = g["_v3121_descent_check_from_traj"](traj, cand)
assert not ok and "25.0%" in why, why
req = seen["req"]
assert req.start_state.joint_state.arm == [1, 2, 3, 4, 5, 6, 7] and req.start_state.is_diff is False  # trajectory END, by name
assert [round(p.position.z, 5) for p in req.waypoints] == [0.07245, 0.05245] and req.avoid_collisions
frac["v"] = 1.0
assert g["_v3121_descent_check_from_traj"](traj, cand)[0]
os.environ["NERO_V3121_DESCENT_PRECHECK"] = "0"
assert g["_v3121_descent_check_from_traj"](traj, cand) == (True, "disabled")
os.environ.pop("NERO_V3121_DESCENT_PRECHECK")
PRE = block("            preflighted = []\n            full_ok304 = 0", "            feasible = [")
assert PRE.count("_v3121_descent_check_from_traj(traj304, c)") == 2 and "V3121_DESCENT_CARTESIAN_REJECT" in PRE
assert PRE.index("_v3121_descent_check_from_traj") < PRE.index('c["moveit_pregrasp_ok"] = True')
print("[H4] descent precheck: from planned pregrasp END state; 25 % rejected, 100 % ok; used in both preflight paths")

# ---------------------------------------------------------------- H5 q0 retreat ladder
B307 = textwrap.dedent(block("    def _v307_go_q0(label):", "    def _v307_post_place_posture("))


def go_q0(max_ok_dz):
    calls, logs = [], []

    def plan(xyz, q):
        dz = round(xyz[2] - 0.073, 3)
        calls.append(("plan", dz))
        if dz > max_ok_dz:
            raise RuntimeError(f"CARTESIAN APPROACH BLOCKED: only 40.7% collision-free")
        return object()
    core = types.SimpleNamespace(emit=logs.append, _cartesian_approach_plan=plan,
                                 _execute_trajectory=lambda t: calls.append("exec"))
    g = dict(skill_core=core, safety_trip_count=NS(0), _V239_LOW_Z_M=0.255, _V239_CLEAR_Z_M=0.340,
             _v218_slow_trajectory_in_place=lambda t, factor=1.0: None,
             _v307_posture_numbers=lambda: ({"near_basket": True, "near_holder": False, "xyz": [-0.453, -0.218, 0.073],
                                             "quat": [1, 0, 0, 0]}, {}),
             _v239_plan_exec_q0_neutral=lambda: calls.append("q0"), _v310_wait_tcp_settled=lambda l: None)
    exec(B307, g)
    g["_v307_go_q0"]("PERCEPTION RECAPTURE")
    return calls, logs


calls, logs = go_q0(1.0)
assert calls == [("plan", 0.267), "exec", "q0"], calls                   # unchanged when the full retreat works
calls, logs = go_q0(0.10)
assert calls == [("plan", 0.267), ("plan", 0.12), ("plan", 0.08), "exec", "q0"], calls
assert any("RETREAT LADDER" in m and "+80 mm" in m for m in logs)
calls, logs = go_q0(0.0)
assert calls[-1] == "q0" and "exec" not in calls and any("joint-space q0 directly" in m for m in logs), calls
print("[H5] q0 retreat ladder: full retreat unchanged; blocked -> +80 mm; none -> collision-checked q0 directly")

# ---------------------------------------------------------------- H6 held pose estimate
c = Core(z=0.0525)
g = helpers(c)
g["V27_GRASP_RUNTIME"]["plans"]["egg_1"] = {"object_xyz_at_plan_m": [-0.4556, -0.2237, 0.0220],
                                             "object_quat_at_plan_xyzw": [0.64022, -0.30016, 0.0, 0.70712]}
g["_v3121_record_grasp"]("egg_1", 60.0)
o = g["V3121_GRASP_ORIGIN"]["egg_1"]
assert o["opening_mm"] == 60.0 and abs(o["tcp_xyz"][2] - 0.0525) < 1e-9
tcp, q = c.current_tcp_pose()
p, qq = g["_v3121_held_pose_estimate"]("egg_1", tcp, q)
assert math.dist(p, [-0.4556, -0.2237, 0.0220]) < 1e-6
p2, _ = g["_v3121_held_pose_estimate"]("egg_1", [tcp[0], tcp[1], tcp[2] + 0.108], q)
assert abs(p2[2] - (0.022 + 0.108)) < 1e-6
qz = [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)]          # rotate TCP 90 deg about z
qn = g["_v3121_qmul"](qz, q)
p3, _ = g["_v3121_held_pose_estimate"]("egg_1", tcp, qn)
d0 = [-0.4556 - tcp[0], -0.2237 - tcp[1]]
assert abs((p3[0] - tcp[0]) - (-d0[1])) < 1e-6 and abs((p3[1] - tcp[1]) - d0[0]) < 1e-6
g["v24_skill_state"] = {"stage": "LIFTED", "active_object": "egg_1"}
g.update(V24_STAGE_GRASPED="GRASP_ESTABLISHED", V24_STAGE_LIFTED="LIFTED", V24_STAGE_BIN_HOVER="RH", V24_STAGE_SLOT_PLACE="SP")
assert g["_v3121_held_name"]() == "egg_1"
g["hold_active_shared"].value = False
assert g["_v3121_held_name"]() is None
print("[H6] held egg pose = TCP x grasp transform (translation + rotation); held only while hold_active")

# ---------------------------------------------------------------- H7 return_to_pick_location
RET = block('        if name == "return_to_pick_location":', '        if name == "v306_safe_end":')
RET_FN = "def _ret(name, args, state_before, contract, t0):\n" + textwrap.indent(textwrap.dedent(RET), "    ") + "    return None\n"


def run_return(stage="LIFTED", z=0.0925, release_works=True, same=True):
    core = Core(z=z)
    if not same:
        core.xy = [-0.30, 0.05]
    egg = {"xyz": [-0.4556, -0.2237, 0.022 + (z - 0.0525)]}
    if not same:
        egg["xyz"] = [-0.30 - 0.0016, 0.05 - 0.0037, 0.022 + (z - 0.0525)]
    hold = NS(True)
    gtarget, gforce = NS(55.0), NS(True)
    core.object_pose = lambda n: (list(egg["xyz"]), [0, 0, 0, 1])
    motions = []

    def cart(tgt, q):
        motions.append(("cart", round(tgt[2], 4)))
        return {"tgt": tgt}

    def execute(t):
        dz = t["tgt"][2] - core.z
        core.xy = list(t["tgt"][:2])
        core.z = t["tgt"][2]
        if hold.value:
            egg["xyz"][2] += dz

    def move_to_pose(t, q, **k):
        motions.append(("move", round(t[2], 4)))
        if hold.value:                                   # held egg travels with the TCP
            egg["xyz"] = [egg["xyz"][0] + t[0] - core.xy[0], egg["xyz"][1] + t[1] - core.xy[1], egg["xyz"][2] + t[2] - core.z]
        core.xy, core.z = list(t[:2]), t[2]
        return True
    core._cartesian_approach_plan, core._execute_trajectory, core.move_to_pose = cart, execute, move_to_pose

    def run_motion(label, fn, target, tol, keep):
        fn()
        return {"tcp_error_mm": 0.4}, True, []
    res = {}

    def result(name, ok, contract, sb, ver, warnings=None, extra=None, next_stage=None, active_object="__KEEP__",
               target_bin="__KEEP__", recommended_next_actions=None):
        res.update(ok=ok, ver=ver, extra=extra, next_stage=next_stage, active=active_object, next=recommended_next_actions)
        return res

    class _T:
        t = 0.0

        @staticmethod
        def time():
            _T.t += 0.1
            return _T.t

        @staticmethod
        def sleep(s):
            if gforce.value is False and release_works:
                hold.value = False

        @staticmethod
        def perf_counter():
            return 0.0
    g = helpers(core, hold_active_shared=hold)
    g["V27_GRASP_RUNTIME"]["plans"]["egg_1"] = {"object_xyz_at_plan_m": [-0.4556, -0.2237, 0.022],
                                                 "object_quat_at_plan_xyzw": [0, 0, 0, 1]}
    core_z_saved, core.z = core.z, 0.0525
    xy_saved = core.xy
    core.xy = [-0.454, -0.220]
    g["_v3121_record_grasp"]("egg_1", 60.0)
    core.z, core.xy = core_z_saved, xy_saved
    plans = {"egg_1": {}}
    g.update(time=_T, llm_require_arm_controller=lambda: None, _v24_run_motion=run_motion, _v24_result=result,
             _v24_precondition_failure=lambda n, a, reason, *x, **k: {"ok": False, "reason": "PRECONDITION_FAILED", "detail": reason},
             V24_STAGE_GRASPED="GRASP_ESTABLISHED", V24_STAGE_LIFTED="LIFTED", V24_STAGE_OBSERVED="OBSERVED",
             gripper_actual_mm=NS(55.0), gripper_force_mode=gforce, gripper_target_mm=gtarget,
             llm_classify_object_location=lambda xyz: "egg_basket" if xyz[2] < 0.03 else "air",
             _v24_xy_distance=lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]),
             V271_PLACEMENT_RUNTIME={"plans": {"egg_1": {}}, "oriented": {}, "slot_entry": {}, "released": {}},
             V311_MONITOR_BASELINE={"egg_1": 1}, EGG_OBJECTS=["egg_1", "egg_2", "egg_3", "egg_4"], _v332_known_eggs=lambda: ["egg_1", "egg_2", "egg_3", "egg_4"],
             v311_mode=lambda: "truth")
    g["V27_GRASP_RUNTIME"]["plans"].update(plans)
    exec(RET_FN, g)
    out = g["_ret"]("return_to_pick_location", {"object_name": "egg_1"},
                    {"stage": stage, "active_object": "egg_1"}, {}, 0.0)
    return out, res, motions, core, egg, gtarget, g


out, res, motions, core, egg, gt, g = run_return()
assert res["ok"] and res["next_stage"] == "OBSERVED" and res["active"] is None and res["next"] == ["plan_grasp"], res
assert motions[0] == ("cart", 0.0555) and motions[-1] == ("cart", 0.1725), motions      # straight down, then retreat
assert abs(egg["xyz"][2] - 0.025) < 1e-6 and gt.value == 61.0                             # set down; opened to >= pick aperture
assert res["extra"]["v3121_next"]["try_next"] == "egg_2" and g["V3121_RETURNS"]["egg_1"] == 1
assert "egg_1" not in g["V27_GRASP_RUNTIME"]["plans"] and "egg_1" not in g["V311_MONITOR_BASELINE"]
out, res, motions, core, egg, gt, g = run_return(same=False, z=0.30)
assert res["ok"] and motions[0] == ("move", 0.1725) and motions[1] == ("cart", 0.0555), motions
out, res, motions, *_ = run_return(stage="GRASP_ESTABLISHED", z=0.0525)
assert res["ok"] and motions == [("cart", 0.1725)], motions                             # at pick pose: release + retreat only
out, *_ = run_return(stage="OBSERVED")
assert out["reason"] == "PRECONDITION_FAILED"
out, res, *_ = run_return(release_works=False)
assert (not res["ok"]) and res["ver"]["reason"] == "RETURN_TO_PICK_LOCATION_FAILED" and res["next_stage"] is None
print("[H7] return_to_pick_location: down at pick pose -> release -> retreat -> OBSERVED + next egg; transit when moved;"
      " GRASPED releases in place; preconditions; failed release reported")

# ---------------------------------------------------------------- H8 wiring
APP = block('        if name == "approach_planned_grasp":', '            safety_stop = bool(')
assert "_v3121_retreat_to_z(" in APP and APP.index("CANDIDATE BLOCKED") < APP.index("_v3121_retreat_to_z(")
REC = block("                    obj_xyz, obj_q = _v311_obj_pose(obj, require_trusted=True)\n                except", "            else:\n                obj_xyz, obj_q = skill_core.object_pose(obj)")
assert "RECAPTURE_CLEAR_VIEW_FAILED" in REC and REC.index("RECAPTURE_CLEAR_VIEW_FAILED") < REC.index("_v311_capture(")
SCN = block("    def llm_scene_snapshot():", "    def llm_require_arm_controller():")
assert '"location": "in_gripper"' in SCN and SCN.index("_v3121_held_name()") < SCN.index("o311 = _v311_obj_entry(name)")
CL = block('        if name == "clamp_gripper":', '        if name == "lift":\n')
assert "_v3121_record_grasp(obj, opening_before)" in CL
assert '"return_to_pick_location": {' in SRC and "V224_ALL_ORIENTATION_STRATEGIES_FAILED" in SRC
assert 'recommended_next_actions=["return_to_pick_location"]' in block("            if not whole_ok24:", "            # SLOT-ABOVE TRANSPORT (V3.02)")
assert "V3121_RETURNS.clear()" in block("    def _v24_reset_skill_state(", "        for _obj in V242_SAFETY_RETRY_USED:")
sys.path.insert(0, str(ROOT))
import NERO_V2_7_1_Agent as A  # noqa: E402
names = [t["name"] for t in A.TOOLS]
assert "return_to_pick_location" in names and "V3.12.1 HELD-EGG RECOVERY" in A.SYSTEM_INSTRUCTIONS
assert "0.10-0.11" in [t for t in A.TOOLS if t["name"] == "lift"][0]["description"]
assert '"return_to_pick_location")' in AG                                                  # allowed in the finish grace
print("[H8] wiring: preflight precheck, approach retreat, recapture reason, in_gripper scene, clamp record, agent tool + rule")
print("ALL V3.12.1 TESTS PASS")
