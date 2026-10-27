#!/usr/bin/env python3
"""Offline tests for V3.04 (no ROS / MuJoCo / GPU).

P1  egg aperture prior: width + 16 mm (44 mm egg -> 60 mm), band +12..+22 mm.
P2  neighbour clearance: fingers over a neighbour egg -> negative; far -> positive.
P3  grasp preflight: candidate whose descent is blocked is rejected by the cheap
    IK screen (no full plan); only 2 full plans; trajectory cached.
P4  preflight safety net: IK screen rejects everything -> fall back to full plans.
P5  model-facing compaction: contract/state_before dropped, 3 candidates shown,
    much smaller JSON; full result unchanged.
"""
import ast, json, math, os, sys, textwrap, time, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
mod = ast.parse(SRC)


def module_funcs(*names):
    return "\n".join(ast.get_source_segment(SRC, n) for n in mod.body
                     if isinstance(n, ast.FunctionDef) and n.name in names)

# ---------------- P1 aperture prior ----------------
i0 = SRC.index("    _v218_base_gripper_prior = _v24_gripper_prior")
i1 = SRC.index("    def llm_scene_snapshot():")
prior_src = textwrap.dedent(SRC[i0:i1])
g = {"os": os, "math": math}
g["_v24_gripper_prior"] = lambda name, op=None: {"effective_width_mm": 44.0, "estimated_pad_envelope_allowance_total_mm": 8.0,
                                                 "adaptive_retry_bonus_mm": 0.0}
exec(prior_src, g)
r = g["_v24_gripper_prior"]("egg_1", 52.8)
assert r["recommended_target_opening_mm"] == 60.0, r
assert r["recommended_opening_range_mm"] == [56.0, 66.0], r
assert r["estimated_free_clearance_at_target_mm"] == 8.0, r
print("[OK] P1 egg_1 (44 mm): target", r["recommended_target_opening_mm"], "band", r["recommended_opening_range_mm"],
      "free clearance", r["estimated_free_clearance_at_target_mm"], "mm (V3.03: 52.8 / 0.8)")

# ---------------- P2 neighbour clearance ----------------
g2 = {"math": math, "V27_FINGER_HALF_LENGTH_M": 0.033, "V27_FINGER_HALF_WIDTH_M": 0.017}
exec(module_funcs("_v304_neighbor_clearance_mm"), g2)
f = g2["_v304_neighbor_clearance_mm"]
# gripper yaw 0 -> fingers at +-y.  Neighbour 55 mm away along +y (closing axis) vs along +x.
near = f((0.0, 0.0), 0.0, 60.0, [{"xy": [0.0, 0.055], "yaw_rad": 0.0, "semi_major_m": 0.032, "semi_minor_m": 0.023}])
far = f((0.0, 0.0), 0.0, 60.0, [{"xy": [0.12, 0.0], "yaw_rad": 0.0, "semi_major_m": 0.032, "semi_minor_m": 0.023}])
assert near < 0 < far, (near, far)
print(f"[OK] P2 neighbour on closing axis -> {near:.1f} mm (intrusion), far neighbour -> {far:.1f} mm")

# ---------------- P3/P4 grasp preflight block ----------------
b0 = SRC.index("            full_need304 = max(1, int(os.environ.get(\"NERO_V304_FULL_PREFLIGHT\", \"2\")))")
b1 = SRC.index("            feasible = [\n                c for c in preflighted", b0)
BODY = SRC[b0:b1] + "            return preflighted\n"
FUNC = "def _pre(env):\n    globals().update(env)\n" + textwrap.indent(textwrap.dedent(BODY), "    ")


class Core:
    def __init__(self, blocked_ids, all_ik_fail=False):
        self.log = []; self.plans = []; self.blocked = blocked_ids; self.all_fail = all_ik_fail
    def emit(self, m): self.log.append(m)
    def _current_joint_state_copy(self): return {"q": [0.1] * 7}
    def _arm_positions_from_joint_state(self, js): return list(js["q"])
    def _v70_natural_seed_variants(self, live, arm, reorientation=False): return [("current", live)]
    def _seed_with_arm_positions(self, base, arm): return {"q": list(arm)}
    def _compute_ik(self, xyz, q, **kw):
        if self.all_fail:
            return None, "NO_IK"
        cid = self.cur
        if cid in self.blocked and xyz[2] < self.gz + 0.015:      # blocked in the last 15 mm
            return None, "COLLISION"
        return list(kw["seed_joint_state"]["q"]), "SUCCESS"
    def _plan_pose(self, xyz, q, **kw):
        self.plans.append(self.cur)
        return types.SimpleNamespace(tag=self.cur)


def cand(cid, score):
    return {"candidate_id": cid, "already_tried": False, "geometry_feasible": True, "score": score,
            "pregrasp_xyz_m": [0, 0, 0.17], "grasp_xyz_m": [0, 0, 0.05], "quat_xyzw": [1, 0, 0, 0]}


def run_pre(blocked, all_fail=False):
    core = Core(blocked, all_fail); core.gz = 0.05
    cands = [cand(f"c{i}", -i) for i in range(6)]
    # make the core know which candidate is being evaluated: wrap dict access order
    class Tracker(list):
        def __iter__(self_inner):
            for c in list.__iter__(self_inner):
                core.cur = c["candidate_id"]; yield c
    cache = {}
    env = dict(os=os, math=math, time=time, skill_core=core, candidates=Tracker(cands), obj="egg_1",
               neighbors304=[], V304_PREGRASP_CACHE=cache)
    gg = {}; exec(FUNC, gg)
    out = gg["_pre"](env)
    return out, core, cache

out, core, cache = run_pre(blocked={"c0"})
st = {c["candidate_id"]: (c["moveit_pregrasp_ok"], str(c["moveit_pregrasp_error"])[:28]) for c in out}
assert st["c0"][0] is False and st["c0"][1].startswith("V304_IK_DESCENT_REJECT"), st
assert core.plans == ["c1", "c2"], core.plans
assert set(k[1] for k in cache) == {"c1", "c2"}
assert all(not c["moveit_pregrasp_ok"] for c in out if c["candidate_id"] in ("c3", "c4", "c5"))
print("[OK] P3 blocked-descent candidate rejected without planning; full plans =", core.plans, "| cached =", sorted(k[1] for k in cache))

# P4: safety net (the fallback needs the real tracker order too)
out, core, cache = run_pre(blocked=set(), all_fail=True)
assert any("falling back to full planning" in m for m in core.log)
assert sum(1 for c in out if c["moveit_pregrasp_ok"]) == 1, [c["moveit_pregrasp_ok"] for c in out]
print("[OK] P4 IK screen rejected all -> fallback full plan; ready =", [c["candidate_id"] for c in out if c["moveit_pregrasp_ok"]])

# ---------------- P5 compaction ----------------
ASRC = (ROOT / "NERO_V2_7_1_Agent.py").read_text()
amod = ast.parse(ASRC)
seg = "\n".join(ast.get_source_segment(ASRC, n) for n in amod.body
                if (isinstance(n, ast.FunctionDef) and n.name == "_v304_compact_for_model")
                or (isinstance(n, ast.Assign) and any(getattr(t, "id", "").startswith("_V304_") for t in n.targets)))
ga = {"os": os}; exec(seg, ga)
full = {"ok": True, "tool": "plan_grasp", "contract": {"purpose": "x" * 400}, "state_before": {"stage": "OBSERVED"},
        "state_after": {"stage": "OBSERVED"}, "recommended_next_actions": ["move_to_pregrasp"],
        "grasp_plan": {"object_name": "egg_1", "recommended_candidate_id": "c1",
                       "candidates": [dict(cand(f"c{i}", -i), moveit_pregrasp_ok=(i in (1, 2)), moveit_pregrasp_error=None,
                                           strategy="s", gripper_yaw_deg=1.0, basket_projected_clearance_mm=-3.0,
                                           closing_axis_yaw_deg=91.0, grasp_xy_m=[0, 0]) for i in range(18)]}}
before = json.dumps(full)
small = ga["_v304_compact_for_model"]("plan_grasp", full)
assert "contract" not in small and "state_before" not in small
assert [c["candidate_id"] for c in small["grasp_plan"]["candidates"]] == ["c1", "c2", "c0"]
assert json.dumps(full) == before          # original untouched
print(f"[OK] P5 compact plan_grasp output {len(before)} -> {len(json.dumps(small))} chars; shown =",
      [c["candidate_id"] for c in small["grasp_plan"]["candidates"]])
