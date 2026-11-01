#!/usr/bin/env python3
"""V3.32: (1) launchers executable in the release; (2) no-prior modes do not assume the number of eggs.

E1 start_nero.sh / start_nero_sim.sh / tests/check_release.sh carry the executable bit (V3.31 / V3.31.1 zips lost it)
E2 backend _v332_known_eggs(): with priors -> registered list; without priors -> only eggs perceived in this episode
   (+ released / held), new episode -> empty; _v332_prune() drops never-perceived egg_N keys / list items
E3 Agent completion guard with objects_policy=perceived_only: 3 perceived eggs -> egg_4 is never pending; all 3 in
   the holder -> nothing pending (no nudges, no extra observe_scene); registered policy unchanged (old behaviour)
E4 wiring: snapshot lists known eggs + objects_policy, verify next-step + pending lists use known eggs, plan_grasp
   of a never-perceived name is refused without motion, LLM rule present
"""
import os, re, sys, textwrap, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ------------------------------------------------------------------ E1
for f in ("start_nero.sh", "start_nero_sim.sh", "tests/check_release.sh"):
    assert os.access(ROOT / f, os.X_OK), f"{f} is not executable"
print("[E1] launchers + check_release.sh executable")

# ------------------------------------------------------------------ E2
B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
a = B.index("    def _v332_known_eggs():")
b = B.index("    def llm_scene_snapshot():")
code = textwrap.dedent(B[a:b])
mode = {"np": True, "held": None}
env = {"os": os, "EGG_OBJECTS": ["egg_1", "egg_2", "egg_3", "egg_4"],
       "_v329_no_prior": lambda: mode["np"], "_V329_SCENE": {"seen": set()}, "_V329_EST": {"released": {}},
       "V271_PLACEMENT_RUNTIME": {"released": {}}, "_v3121_held_name": lambda: mode["held"]}
exec(code, env)
K = env["_v332_known_eggs"]
assert K() == []                                                    # episode start: nothing is known
env["_V329_SCENE"]["seen"] = {"egg_3", "egg_1", "egg_2"}
assert K() == ["egg_1", "egg_2", "egg_3"]
env["_V329_SCENE"]["seen"] = {"egg_1"}; env["V271_PLACEMENT_RUNTIME"]["released"] = {"egg_2": {}}; mode["held"] = "egg_3"
assert K() == ["egg_1", "egg_2", "egg_3"]
mode["np"] = False
assert K() == ["egg_1", "egg_2", "egg_3", "egg_4"]                 # fixed V3.11 / truth: registered list
P = env["_v332_prune"]
r = P({"objects": {"egg_1": {"x": 1}, "egg_4": {}}, "remaining": ["egg_4", "egg_1"], "egg_holder": {"egg_4": 0},
       "retry": {"egg_1": 0, "egg_4": 0}}, {"egg_1"})
assert r == {"objects": {"egg_1": {"x": 1}}, "remaining": ["egg_1"], "egg_holder": {}, "retry": {"egg_1": 0}}, r
print("[E2] known eggs = perceived in this episode (+held/released); registered list with priors; prune removes "
      "never-perceived names")

# ------------------------------------------------------------------ E3
A = (ROOT / "NERO_V2_7_1_Agent.py").read_text()
m = re.search(r"\n    @staticmethod\n    def _task_object_names\(task, scene_snapshot=None\):.*?\n        return requested\n", A, re.S)
assert m, "_task_object_names not found"
ns = {}
exec(textwrap.dedent(m.group(0).replace("@staticmethod", "")), ns)
T = ns["_task_object_names"]
task = "Pick up the eggs on the table and place them into empty egg-holder slots, one by one."
scene = {"objects_policy": "perceived_only", "objects": {
    "egg_1": {"location": "egg_holder.slot_1"}, "egg_2": {"location": "egg_basket"}, "egg_3": {"location": "egg_basket"}}}
assert T(task, scene) == ["egg_2", "egg_3"]
scene["objects"]["egg_2"]["location"] = "egg_holder.slot_2"; scene["objects"]["egg_3"]["location"] = "egg_holder.slot_3"
assert T(task, scene) == []                                         # done - egg_4 never perceived is not pending
scene["objects"]["egg_4"] = {"location": "not_perceived"}
assert T(task, scene) == []
assert T("pick up egg_4", scene) == []                              # named but never perceived -> not pending
old = {"objects": {"egg_1": {"location": "egg_holder.slot_1"}, "egg_2": {"location": "egg_basket"},
                   "egg_3": {"location": "egg_basket"}, "egg_4": {"location": "egg_basket"}}}
assert T(task, old) == ["egg_2", "egg_3", "egg_4"]                  # registered policy unchanged
assert T(task, None) == ["egg_1", "egg_2", "egg_3", "egg_4"]
print("[E3] completion guard: perceived eggs only (3 eggs on the table -> done after 3); registered policy unchanged")

# ------------------------------------------------------------------ E4
for mk in ("for name in _v332_known_eggs():", '"objects_policy": ("perceived_only" if _v329_no_prior() else "registered")',
           "if _v329_no_prior() and str(_o) not in _v332_known_eggs():", "for n in _v332_known_eggs():                     # V3.32",
           "_llm_tool_runner_v331 = llm_tool_runner", "OBJECT_NOT_PERCEIVED_IN_THIS_EPISODE", "[V3.32 UNKNOWN OBJECT]",
           "res = _v332_prune(res, set(_v332_known_eggs()))"):
    assert mk in B, mk
assert "V3.32 PERCEIVED OBJECTS ONLY" in A
w = B.index("    def llm_tool_runner(name, args):\n        \"\"\"V3.32 wrapper")
assert B.index("_llm_tool_runner_v331 = llm_tool_runner") < w < B.index("tool_runner=llm_tool_runner,")
print("[E4] wiring: snapshot / verify / pending use perceived eggs; unknown name refused without motion; LLM rule")
print("ALL V3.32 TESTS PASS")
