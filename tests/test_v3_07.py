#!/usr/bin/env python3
"""Offline tests for V3.07 (no ROS / MuJoCo / GPU).

R1  healthy post-place posture; all candidates fail from current start
    (cuRobo success=false) -> vertical retreat + q0 -> re-plan once -> OK.
R2  post-place posture wound up (distal 150 deg) -> q0 BEFORE planning; plan
    still fails from q0 -> no second reset, failure marked goal-side.
R3  re-plan from q0 still fails -> warning says goal-side; reset budget is
    once per object (second call skips).
R4  q0 reset itself fails -> reason START_STATE_PLANNING_FAILED, stop/report.
R5  genuinely exhausted (no untried feasible candidate) -> no reset.
G1  completion guard does not nudge for eggs already re-planned from q0.
"""
import json, math, os, sys, textwrap, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
a = SRC.index("    # V3.07 START-STATE RESET + POST-PLACE POSTURE")
b = SRC.index("    def llm_tool_runner(name, args):", a)
BLOCK = textwrap.dedent(SRC[a:b])
assert 'if not args.pop("_v307_inner", False):' in SRC


def build(q_deg, margin, distal, plans, q0_fails=False, stage="PLACEMENT_VERIFIED", z=0.181):
    st = {"q": [math.radians(v) for v in q_deg], "margin": margin, "distal": distal, "z": z}
    calls, logs = [], []

    def snap():
        return {"q": st["q"], "xyz": [-0.425, 0.14, st["z"]], "quat": [1, 0, 0, 0],
                "joint_margin_deg": st["margin"], "distal_wrist_deg": st["distal"],
                "near_basket": False, "near_holder": st["z"] < 0.3}

    def q0():
        calls.append("q0")
        if q0_fails:
            raise RuntimeError("V2-39 q0 MoveIt joint planning returned no trajectory")
        st.update(q=[0.0] * 7, margin=60.0, distal=0.0, z=0.873)

    def runner(name, args):
        assert name == "plan_grasp" and args.get("_v307_inner") is True
        calls.append("plan")
        r = plans.pop(0)
        return json.loads(json.dumps(r))

    core = types.SimpleNamespace(
        emit=logs.append,
        _cartesian_approach_plan=lambda xyz, q: (calls.append(("retreat", round(xyz[2], 3))) or object()),
        _execute_trajectory=lambda t: (st.update(z=0.34) if st["z"] < 0.34 else None),
    )
    g = dict(math=math, os=os, skill_core=core, _v239_health_snapshot=snap, _v239_plan_exec_q0_neutral=q0,
             hold_active_shared=types.SimpleNamespace(value=False), safety_state=types.SimpleNamespace(value=0),
             safety_trip_count=types.SimpleNamespace(value=0), V24_STAGE_VERIFIED="PLACEMENT_VERIFIED",
             V24_STAGE_RELEASED="RELEASED", _V239_LOW_Z_M=0.255, _V239_CLEAR_Z_M=0.340,
             _v218_slow_trajectory_in_place=lambda t, factor=1.0: None,
             _v24_state_snapshot=lambda: {"stage": stage}, llm_tool_runner=runner)
    exec(BLOCK, g)
    return g, calls, logs


def fail(errs):
    return {"ok": False, "physical_verification": {"reason": "ALL_GRASP_CANDIDATES_EXHAUSTED_OR_INFEASIBLE",
                                                   "edge_reposition_available": False},
            "warnings": ["No bounded grasp candidate is executable"],
            "grasp_plan": {"candidates": [{"candidate_id": f"c{i}", "geometry_feasible": True, "already_tried": False,
                                           "moveit_pregrasp_error": e} for i, e in enumerate(errs)]}}


START = "cuRobo exhausted 10 trajectory attempts; tail=[#10:curobo:success=false]"
GOAL = "V304_IK_DESCENT_REJECT: no IK at pregrasp"
OK = {"ok": True, "physical_verification": {"recommended_candidate_id": "c0"}}
Q_WOUND = [10, 20, -5, 80, 30, 40, 150]

# R1
g, calls, logs = build(Q_WOUND, margin=30, distal=50, plans=[fail([START, START, GOAL]), OK])
r = g["_v307_plan_grasp_supervised"]({"object_name": "egg_3"})
assert r["ok"] and calls == ["plan", ("retreat", 0.34), "q0", "plan"], calls
i = r["v307_start_state_reset"]
assert i["attempted"] and i["q0_ok"] and i["replanned_from_q0"] and i["start_state_class_failures"] == 2, i
assert any("[V3.07 START-STATE RESET]" in m for m in logs) and any("RESET OK" in m for m in logs)
print("[OK] R1 start-state failure -> retreat + q0 -> re-plan OK:", calls)

# R2
g, calls, logs = build(Q_WOUND, margin=30, distal=150, plans=[fail([START, GOAL])])
r = g["_v307_plan_grasp_supervised"]({"object_name": "egg_3"})
assert calls == [("retreat", 0.34), "q0", "plan"], calls
assert r["v307_post_place_posture"]["q0_executed"] and r["v307_start_state_reset"]["skipped"].startswith("start state already")
assert any("goal-side" in w for w in r["warnings"]), r["warnings"]
print("[OK] R2 wound-up wrist -> q0 before planning; still infeasible -> goal-side, no second reset:", calls)

# R3
g, calls, logs = build(Q_WOUND, margin=30, distal=50, plans=[fail([START]), fail([START]), fail([START])], stage="HOVER_REACHED")
r = g["_v307_plan_grasp_supervised"]({"object_name": "egg_4"})
assert not r["ok"] and r["v307_start_state_reset"]["replanned_from_q0"] and any("goal-side" in w for w in r["warnings"])
assert any("NO HELP" in m for m in logs)
# second call for same egg from a non-q0 posture: budget already used
g["_v239_health_snapshot"]  # same closure state
r2 = g["_v307_plan_grasp_supervised"]({"object_name": "egg_4"})
assert r2["v307_start_state_reset"]["skipped"] in ("reset budget for this object already used", "start state already SAFE_NEUTRAL q0")
assert calls.count("q0") == 1, calls
print("[OK] R3 still infeasible from q0 -> goal-side warning; reset budget once per egg:", calls)

# R4
g, calls, logs = build(Q_WOUND, margin=30, distal=50, plans=[fail([START])], q0_fails=True)
r = g["_v307_plan_grasp_supervised"]({"object_name": "egg_3"})
assert r["physical_verification"]["reason"] == "START_STATE_PLANNING_FAILED"
assert r["recommended_next_actions"] == ["finish_and_report"] and calls == ["plan", ("retreat", 0.34), "q0"]
print("[OK] R4 q0 reset fails -> START_STATE_PLANNING_FAILED, stop and report")

# R5
f = fail([START]); f["grasp_plan"]["candidates"][0]["already_tried"] = True
g, calls, logs = build(Q_WOUND, margin=30, distal=50, plans=[f])
r = g["_v307_plan_grasp_supervised"]({"object_name": "egg_3"})
assert calls == ["plan"] and "genuinely exhausted" in r["v307_start_state_reset"]["skipped"]
print("[OK] R5 genuinely exhausted -> no reset")

# G1 agent guard
sys.path.insert(0, str(ROOT))
import NERO_V2_7_1_Agent as AG
AG.NeroLLMAgent._log_usage = lambda self, r, m: None
script = iter([("plan_grasp", {"object_name": "egg_3"}), ("plan_grasp", {"object_name": "egg_4"})])
def fake_post(self, payload, api_key, timeout_s=180.0):
    try:
        n, args = next(script)
    except StopIteration:
        return {"id": "end", "output": [{"type": "message", "content": [{"type": "output_text", "text": "egg_3/egg_4 infeasible"}]}]}
    return {"id": "r", "output": [{"type": "function_call", "name": n, "call_id": "x", "arguments": json.dumps(args)}]}
AG.NeroLLMAgent._post_responses = fake_post
SCENE = {"objects": {"egg_1": {"location": "egg_holder.slot_1"}, "egg_2": {"location": "egg_holder.slot_2"},
                     "egg_3": {"location": "egg_basket"}, "egg_4": {"location": "egg_basket"}}}
def tr(name, args):
    if name == "plan_grasp":
        x = fail([START]); x["v307_start_state_reset"] = {"attempted": True, "replanned_from_q0": True}
        return x
    return {"ok": True, "tool": name, "scene": SCENE}
glogs = []
ag = AG.NeroLLMAgent(glogs.append, tr); ag.api_key_status = lambda k="": (True, "ok")
ag.run_task("Pick up the four eggs and place them into empty egg-holder slots, one by one.", api_key="x", max_tool_steps=48)
assert any("[V3.07 GUARD]" in m for m in glogs), [m for m in glogs if "GUARD" in m]
assert not any("[MULTI-OBJECT GUARD] model attempted" in m for m in glogs)
print("[OK] G1 guard does not nudge for eggs proven infeasible from q0")
