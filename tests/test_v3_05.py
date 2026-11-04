#!/usr/bin/env python3
"""Offline tests for V3.05 (no ROS / MuJoCo / GPU).

X1  executor: a goal that ends CANCELED/ABORTED with error_code=0 is a fault
    (V3.04 treated it as success); SUCCEEDED passes.
X2  executor: two publishers on the arm_controller action status -> refuse.
X3  backend: large tracking mismatch / rejected goal -> EXECUTION_LAYER_FAULT,
    q0 recovery when nothing is held, retry_same_step_once; 2nd consecutive
    fault -> unhealthy + stop_and_report.  Small (contact-size) mismatch is
    NOT an execution fault.  Holding an egg -> no recovery motion.
X4  agent: repeated EXECUTION_LAYER_FAULT halts the episode deterministically
    (no egg switching, no multi-object-guard nudges).
S1  IK screen: a pregrasp only found from the 3rd seed is accepted (V3.04 = 2 seeds).
S2  fallback never re-admits a descent-BLOCKED candidate.
S3  pregrasp cache is OFF by default.
"""
import ast, json, math, os, re, sys, textwrap, time, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
VSRC = (ROOT / "NERO_V71_Hybrid_GPU_Planner_Core.py").read_text()

# ---------------- X1 / X2 executor helpers ----------------
vmod = ast.parse(VSRC)
cls = [n for n in vmod.body if isinstance(n, ast.ClassDef) and any(
    isinstance(b, ast.FunctionDef) and b.name == "_v305_require_succeeded" for b in n.body)][0]
parts = [ast.get_source_segment(VSRC, b) for b in cls.body
         if (isinstance(b, ast.FunctionDef) and b.name.startswith("_v305_"))
         or (isinstance(b, ast.Assign) and any(getattr(t, "id", "") == "V305_GOAL_STATUS_NAMES" for t in b.targets))]
code = "class Exec:\n" + textwrap.indent("\n".join(textwrap.dedent(p) for p in parts), "    ")
g = {"os": os}
exec(code, g)


class Node:
    def __init__(self, n): self.n = n
    def count_publishers(self, topic): return self.n


def mk(n):
    e = g["Exec"](); e.node = Node(n); e.log = []; e.emit = e.log.append
    return e


wr = lambda st: types.SimpleNamespace(status=st, result=types.SimpleNamespace(error_code=0, error_string="cancelled"))
e = mk(1)
e._v305_require_succeeded(wr(4), "trajectory")
for st in (5, 6):
    try:
        e._v305_require_succeeded(wr(st), "trajectory"); raise AssertionError("not raised")
    except RuntimeError as exc:
        assert "EXECUTION_LAYER_FAULT" in str(exc)
print("[OK] X1 status!=SUCCEEDED with error_code=0 -> EXECUTION_LAYER_FAULT; SUCCEEDED passes")
mk(1)._v305_check_single_arm_controller()
try:
    mk(2)._v305_check_single_arm_controller(); raise AssertionError("not raised")
except RuntimeError as exc:
    assert "DUPLICATE_ARM_CONTROLLER" in str(exc)
print("[OK] X2 duplicate /arm_controller detected -> refuse to execute")

# ---------------- X3 backend fault handling ----------------
a = SRC.index("    def _v305_is_exec_fault(exc):")
b = SRC.index("    def llm_tool_runner(name, args):")
r0 = SRC.index("    def _v24_result(")
r1 = SRC.index("    def _v24_aperture_check(")
FAULT = textwrap.dedent(SRC[a:b]) + "\n" + textwrap.dedent(SRC[r0:r1])


class Core:
    def __init__(self): self.log = []; self.gate = []
    def emit(self, m): self.log.append(m)
    def cancel_active_motion(self): self.log.append("CANCEL")
    def _set_arm_command_gate(self, v): self.gate.append(v)


def env(holding=False, recover_ok=True):
    core = Core(); calls = []

    def q0():
        calls.append("q0")
        if not recover_ok:
            raise RuntimeError("PHYSICAL TRACKING MISMATCH: still 90.0deg")
        return True
    G = dict(re=re, os=os, json=json, time=types.SimpleNamespace(sleep=lambda s: None), skill_core=core,
             V305_EXEC_FAULT={"consecutive": 0, "total": 0, "last": None, "in_recovery": False},
             V304_PREGRASP_CACHE={("egg_1", "c"): 1}, hold_active_shared=types.SimpleNamespace(value=holding),
             _v239_plan_exec_q0_neutral=q0,
             _v24_set_state=lambda *a, **k: {"stage": "OBSERVED"},
             _v24_next_actions=lambda st, ok: ["observe_scene", "plan_grasp"])
    exec(FAULT, G)
    return G, core, calls


G, core, calls = env()
big = RuntimeError("PHYSICAL TRACKING MISMATCH: ros2_control reported success but MuJoCo is still 132.1deg from the planned joint goal")
small = RuntimeError("PHYSICAL TRACKING MISMATCH: ros2_control reported success but MuJoCo is still 7.5deg from the planned joint goal")
assert G["_v305_is_exec_fault"](big) and not G["_v305_is_exec_fault"](small)
assert G["_v305_is_exec_fault"](RuntimeError("EXECUTION_LAYER_FAULT: arm_controller rejected trajectory"))
ver = {"reason": "CARTESIAN_POSTCONDITION_FAILED"}
G["_v305_handle_exec_fault"]("move_to_pregrasp(egg_1,c)", big, ver)
res = G["_v24_result"]("move_to_pregrasp", False, {}, {}, ver)
assert calls == ["q0"] and not G["V304_PREGRASP_CACHE"], calls
assert res["reason"] == "EXECUTION_LAYER_FAULT" and res["recommended_next_actions"] == ["observe_scene", "retry_same_step_once"], res
assert res["execution_layer_unhealthy"] is False
ver2 = {}
G["_v305_handle_exec_fault"]("move_to_pregrasp(egg_1,c)", RuntimeError("EXECUTION_LAYER_FAULT: arm_controller rejected trajectory"), ver2)
res2 = G["_v24_result"]("move_to_pregrasp", False, {}, {}, ver2)
assert res2["execution_layer_unhealthy"] is True and res2["recommended_next_actions"] == ["stop_and_report"], res2
print("[OK] X3 fault #1 -> q0 recovery + retry_same_step_once; fault #2 -> unhealthy + stop_and_report; 7.5 deg contact mismatch is not a fault")
G, core, calls = env(holding=True)
v = {}; G["_v305_handle_exec_fault"]("lift", big, v)
assert calls == [] and v["execution_layer_fault"]["recovery"]["attempted"] is False
G, core, calls = env(recover_ok=False)
v = {}; G["_v305_handle_exec_fault"]("move_to_pregrasp", big, v)
assert v["execution_layer_fault"]["unhealthy"] is True
print("[OK] X3 holding egg -> no recovery motion; failed recovery -> unhealthy immediately")

# ---------------- X4 agent supervisor ----------------
sys.path.insert(0, str(ROOT))
import NERO_V2_7_1_Agent as AG

logs = []
tools_called = []


def runner(name, args):
    tools_called.append(name)
    if name == "observe_scene":
        return {"ok": True, "tool": name, "scene": {"objects": {f"egg_{i}": {"location": "egg_basket"} for i in range(1, 5)}}}
    if name == "move_to_pregrasp":
        return {"ok": False, "tool": name, "reason": "EXECUTION_LAYER_FAULT", "execution_layer_unhealthy": False,
                "recommended_next_actions": ["observe_scene", "retry_same_step_once"]}
    return {"ok": True, "tool": name}


script = iter(["observe_scene", "plan_grasp", "move_to_pregrasp", "observe_scene", "plan_grasp", "move_to_pregrasp",
               "plan_grasp", "move_to_pregrasp", "plan_grasp"])


def fake_post(self, payload, api_key, timeout_s=180.0):
    n = next(script)
    args = {} if n == "observe_scene" else {"object_name": "egg_1", "candidate_id": "c"}
    return {"id": f"r{len(tools_called)}", "output": [{"type": "function_call", "name": n, "call_id": "x",
                                                      "arguments": json.dumps(args)}]}


AG.NeroLLMAgent._post_responses = fake_post
AG.NeroLLMAgent._log_usage = lambda self, r, m: None
agent = AG.NeroLLMAgent(logs.append, runner)
agent.api_key_status = lambda k="": (True, "ok")
txt = agent.run_task("Pick up the four eggs and place them into empty egg-holder slots, one by one.", api_key="x")
assert "TASK HALTED" in txt, txt
assert tools_called.count("move_to_pregrasp") == 2, tools_called
assert any("[V3.05 EXECUTION HALT]" in m for m in logs)
assert not any("MULTI-OBJECT GUARD]" in m for m in logs)
print("[OK] X4 agent halts after the 2nd consecutive EXECUTION_LAYER_FAULT; tools =", tools_called)

# ---------------- S1/S2 IK screen ----------------
b0 = SRC.index("            full_need304 = max(1, int(os.environ.get(\"NERO_V304_FULL_PREFLIGHT\", \"2\")))")
b1 = SRC.index("            feasible = [\n                c for c in preflighted", b0)
BODY = SRC[b0:b1] + "            return preflighted\n"
FUNC = "def _pre(env):\n    globals().update(env)\n" + textwrap.indent(textwrap.dedent(BODY), "    ")


class SCore:
    def __init__(self, mode): self.mode = mode; self.log = []; self.plans = []
    def emit(self, m): self.log.append(m)
    def _current_joint_state_copy(self): return {"q": [0.0] * 7, "tag": "live"}
    def _arm_positions_from_joint_state(self, js): return list(js["q"])
    def _v70_natural_seed_variants(self, live, arm, reorientation=False):
        return [("current", live), ("a", {"q": [0.1] * 7, "tag": "a"}), ("b", {"q": [0.2] * 7, "tag": "b"})]
    def _seed_with_arm_positions(self, base, arm): return {"q": list(arm), "tag": "chain"}
    def _compute_ik(self, xyz, q, **kw):
        sd = kw["seed_joint_state"]
        if xyz[2] > 0.15:                                    # pregrasp
            if self.mode == "third_seed" and sd["tag"] not in ("b", "chain"):
                return None, "NO_IK"
            return list(sd["q"]), "SUCCESS"
        if self.mode == "descent_blocked" and xyz[2] < 0.065:
            return None, "COLLISION"
        return list(sd["q"]), "SUCCESS"
    def _plan_pose(self, xyz, q, **kw):
        self.plans.append(tuple(q)); return types.SimpleNamespace()


def cand(cid, score, yaw):
    return {"candidate_id": cid, "already_tried": False, "geometry_feasible": True, "score": score,
            "pregrasp_xyz_m": [0, 0, 0.17], "grasp_xyz_m": [0, 0, 0.05], "quat_xyzw": [1, 0, 0, yaw]}


def run_s(mode):
    core = SCore(mode)
    cands = [cand(f"c{i}", -i, round(i / 2) * 0.1) for i in range(6)]
    genv = dict(os=os, math=math, time=time, skill_core=core, candidates=cands, obj="egg_2",
                neighbors304=[], V304_PREGRASP_CACHE={})
    gg = {}; exec(FUNC, gg)
    return gg["_pre"](genv), core


out, core = run_s("third_seed")
ok = [c["candidate_id"] for c in out if c["moveit_pregrasp_ok"]]
assert len(ok) == 2 and all(c.get("v304_ik_descent_ok") for c in out if c["moveit_pregrasp_ok"]), [(c["candidate_id"], c["moveit_pregrasp_error"]) for c in out]
print("[OK] S1 pregrasp IK only found from the 3rd seed -> accepted; planned", ok)
out, core = run_s("descent_blocked")
assert core.plans == [] and not any(c["moveit_pregrasp_ok"] for c in out)
assert all("descent blocked" in str(c["moveit_pregrasp_error"]) for c in out)
print("[OK] S2 descent-blocked candidates are NOT re-admitted by the fallback (0 full plans)")

# ---------------- S3 cache off ----------------
assert 'os.environ.get("NERO_V305_PREGRASP_CACHE", "0") != "1"' in SRC
print("[OK] S3 pregrasp trajectory cache OFF by default (NERO_V305_PREGRASP_CACHE=1 to re-enable)")
