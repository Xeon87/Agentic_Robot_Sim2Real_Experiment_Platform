#!/usr/bin/env python3
"""Offline tests for V3.06 (no ROS / MuJoCo / GPU).

B1  step budget = max(user, 14 x eggs + 8), cap 100 (4 eggs, user 48 -> 64).
B2  budget exhausted with an egg in hand: placement-only tools get a finish
    grace; a new-egg tool triggers v306_safe_end and the episode stops.
A1  plan_grasp auto-prepares the aperture (one internal set_gripper_opening).
V1  verify_object_in_slot returns v306_next (remaining eggs, next empty slot).
S1  v306_safe_end: holding -> vertical rise to safe Z keeping the grasp;
    not holding -> SAFE_NEUTRAL q0.
"""
import json, os, re, sys, textwrap, time, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
sys.path.insert(0, str(ROOT))
import NERO_V2_7_1_Agent as AG

AG.NeroLLMAgent._log_usage = lambda self, r, m: None


def make_agent(script, runner, logs):
    it = iter(script)

    def fake_post(self, payload, api_key, timeout_s=180.0):
        try:
            n, args = next(it)
        except StopIteration:
            return {"id": "end", "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}]}
        return {"id": "r", "output": [{"type": "function_call", "name": n, "call_id": "x", "arguments": json.dumps(args)}]}
    AG.NeroLLMAgent._post_responses = fake_post
    a = AG.NeroLLMAgent(logs.append, runner)
    a.api_key_status = lambda k="": (True, "ok")
    return a


os.environ["NERO_V342_AUTO_BUDGET"] = "1"   # V3.42: the per-egg auto budget is opt-in now
# ---------------- B1 ----------------
logs = []
def r_all_done(name, args):
    return {"ok": True, "tool": name, "scene": {"objects": {f"egg_{i}": {"location": f"egg_holder.slot_{i}"} for i in range(1, 5)}}}
make_agent([("observe_scene", {})], r_all_done, logs).run_task(
    "Pick up the four eggs and place them into empty egg-holder slots, one by one.", api_key="x", max_tool_steps=48)
b = [m for m in logs if "[V3.06 STEP BUDGET]" in m]
assert b and b[0].endswith("-> 64"), b
print("[OK] B1", b[0])

# ---------------- B2 ----------------
os.environ["NERO_V306_STEPS_PER_OBJECT"] = "1"; os.environ["NERO_V306_STEPS_MARGIN"] = "0"
logs, called = [], []
def r_hold(name, args):
    called.append(name)
    if name == "lift":
        return {"ok": True, "tool": name, "state_after": {"stage": "LIFTED"}, "physical_verification": {"hold_active": True}}
    if name == "v306_safe_end":
        return {"ok": True, "physical_verification": {"holding": True, "action": "held egg kept; vertical rise 100 mm"}}
    return {"ok": True, "tool": name}
script = [("observe_scene", {}), ("plan_grasp", {"object_name": "egg_1"}), ("lift", {"height_m": 0.08}),
          ("plan_placement", {"object_name": "egg_1", "slot_name": "slot_1", "receptacle_name": "egg_holder"}),
          ("orient_for_slot", {"object_name": "egg_1", "slot_name": "slot_1"}),
          ("plan_grasp", {"object_name": "egg_2"})]
try:
    make_agent(script, r_hold, logs).run_task("pick up egg_1", api_key="x", max_tool_steps=3)
    raise AssertionError("should stop")
except RuntimeError as exc:
    assert "safe end" in str(exc), exc
assert called == ["observe_scene", "plan_grasp", "lift", "plan_placement", "orient_for_slot", "v306_safe_end"], called
assert sum("[V3.06 FINISH GRACE]" in m for m in logs) == 2
print("[OK] B2 budget hit with egg in hand -> grace for plan_placement/orient_for_slot; new-egg plan_grasp -> safe end:", called)
del os.environ["NERO_V306_STEPS_PER_OBJECT"], os.environ["NERO_V306_STEPS_MARGIN"]

# ---------------- A1 ----------------
a0 = SRC.index("            auto_ap306 = None")
a1 = SRC.index('            if not aperture["robust_aperture_ok"]:\n                return _v24_precondition_failure(', a0)
blk = "def _a(env):\n    globals().update(env)\n    aperture = env['aperture']\n" + textwrap.indent(textwrap.dedent(SRC[a0:a1]), "    ") + "    return aperture, auto_ap306\n"
state = {"open": 90.0}; sub = []
def check(obj):
    return {"robust_aperture_ok": 56 <= state["open"] <= 66, "actual_opening_mm": state["open"], "recommended_target_opening_mm": 60.0}
def runner(n, a):
    sub.append((n, a)); state["open"] = a["target_opening_mm"]
    return {"ok": True, "physical_verification": {"opening_after_mm": state["open"]}}
g = {}; exec(blk, g)
ap, auto = g["_a"](dict(os=os, obj="egg_1", aperture=check("egg_1"), _v24_aperture_check=check, llm_tool_runner=runner,
                        skill_core=types.SimpleNamespace(emit=lambda m: None)))
assert ap["robust_aperture_ok"] and sub == [("set_gripper_opening", {"target_opening_mm": 60.0})] and auto["ok"]
print("[OK] A1 plan_grasp auto aperture 90 -> 60 mm via one internal set_gripper_opening")

# ---------------- V1 ----------------
v0 = SRC.index("            next306 = None")
v1 = SRC.index("            nxt306 = None", v0)
blk = "def _v(env):\n    globals().update(env)\n" + textwrap.indent(textwrap.dedent(SRC[v0:v1]), "    ") + "    return next306\n"
locs = {"egg_1": "egg_holder.slot_1", "egg_2": "egg_holder.slot_2", "egg_3": "egg_holder.slot_3", "egg_4": "egg_basket"}
g = {}; exec(blk, g)
nx = g["_v"](dict(OBJECTS=list(locs), llm_classify_object_location=lambda xyz: locs[xyz],
                  skill_core=types.SimpleNamespace(object_pose=lambda o: (o, None)),
                  EGG_SLOT_POSES={f"slot_{i}": None for i in (8, 1, 2, 3, 4, 5, 6, 7)},
                  gripper_actual_mm=types.SimpleNamespace(value=89.4), cam311v=False,   # V3.11: truth mode
                  _v329_no_prior=lambda: False, _v332_known_eggs=lambda: list(locs)))   # V3.32
assert nx["next_egg"] == "egg_4" and nx["next_empty_slot"] == "slot_4" and nx["remaining_eggs_outside_holder"] == ["egg_4"], nx
print("[OK] V1 verify -> v306_next:", {k: nx[k] for k in ("next_egg", "next_empty_slot")})

# ---------------- S1 ----------------
s0 = SRC.index('        if name == "v306_safe_end":')
s1 = SRC.index("            return _v24_result(name, ok306", s0)
body = textwrap.dedent(SRC[s0:s1]).split("\n", 1)[1]
blk = "def _s(env):\n    globals().update(env)\n" + textwrap.indent(textwrap.dedent(body), "    ") + "    return ok306, act306\n"
def run_s(holding):
    calls = []; tcp = {"z": 0.133}
    def lift(dz): calls.append(("lift", round(dz, 3))); tcp["z"] += dz
    core = types.SimpleNamespace(current_tcp_pose=lambda: ([0, 0, tcp["z"]], [1, 0, 0, 0]), lift=lift, emit=lambda m: None)
    g = {}; exec(blk, g)
    return g["_s"](dict(os=os, json=json, time=time, hold_active_shared=types.SimpleNamespace(value=holding), skill_core=core,
                        _v239_plan_exec_q0_neutral=lambda: calls.append(("q0",)), state_before={"stage": "LIFTED", "active_object": "egg_4"},
                        t0=time.perf_counter())), calls
(ok, act), calls = run_s(True)
assert ok and calls == [("lift", 0.117)] and act["tcp_z_after_m"] >= 0.249, (act, calls)
(ok, act), calls = run_s(False)
assert ok and calls == [("q0",)]
print("[OK] S1 safe end: holding -> rise to 0.25 m keeping grasp; empty -> q0")
