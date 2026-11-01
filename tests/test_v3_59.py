#!/usr/bin/env python3
"""V3.59 offline unit tests (strategy profile / control knobs, phase 1 of the V3.59-V3.62 Sim2Real plan):
Z1 Version V3.59 & markers (Console, Camera, Strategy module, Agent, launchers); NERO_V308 untouched; simulation backend
   and sim wrist camera byte-identical to V3.58; real wrist camera module differs from V3.58 only by VERSION.
Z2 Knob table: 4 groups / 14 knob items / 16 parameters with the AGENTS.md 8.1 defaults and optimizer bounds;
   clamping + type validation; JSON save / load round trip in a temp dir; unknown keys warned; shipped presets valid.
Z3 Shared decision layer: w_dist vs w_clearance flips the pick order; yaw alignment; 4 slot fill orders;
   IK posture preference (base-joint penalty, limit margin, elbow lift); grip widths; held check; hover z.
Z4 Sim bridge (non-invasive): to_sim_env reproduces the backend defaults (limit margin 8 deg, descent x3.5); Agent hook
   is the identity without NERO_V359_STRATEGY_PROFILE; with it observe_scene carries v359_strategy, the fill order and
   v306_next follow the profile, a rewritten profile file applies on the next tool call (no restart); the extra
   prompt rule appears only when the hook is active; SimRunner passes the knob env to the child process.
Z5 Real side: plan_hover_target prefer_fn picks the lowest-cost IK; CameraTabUI.set_strategy links hover height /
   best-ranked target / cruise speed and unlinks back to the old hover height.
Z6 Mocked-Tkinter console: run_console('sim') / run_console('real') build the 「策略旋钮」 tab; changing w_clearance /
   hover_height_mm in the panel immediately changes the ranking and the Camera tab hover height; save -> reset -> load
   restores the preset; sim_env() is {} until 「仿真启动时注入」 is ticked.
"""
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

TMP = tempfile.mkdtemp(prefix="nero_v359_")
os.environ["NERO_STRATEGY_DIR"] = os.path.join(TMP, "strategies")       # never touch config/strategies
os.environ["NERO_V359_ACTIVE_PATH"] = os.path.join(TMP, "active.json")
os.environ.pop("NERO_V359_STRATEGY_PROFILE", None)

import NERO_Strategy_Profile as SP            # noqa: E402
import NERO_V312_Console as CON               # noqa: E402
import NERO_V313_Real_Robot as RR             # noqa: E402
import NERO_V315_Consistency_Safety as CS     # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V347_Yolo_Perception as YP        # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49     # noqa: E402

H = lambda b: hashlib.sha256(b).hexdigest()

# ------------------------------------------------------------------ Z1: versions, markers, frozen files
for m in (CON, RR, YP, W49, SP):
    assert m.VERSION == "V3.59", (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() == "V3.59"
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
assert H((ROOT / "NERO_V2_7_1_Unified_Backend.py").read_bytes()) == \
    "357bbc5bc88a41b6c715d9cef78607b49273fbbdcb64d416ab505a274e389f03", "simulation backend must stay as V3.58"
assert H((ROOT / "NERO_V326_Wrist_Camera.py").read_bytes()) == \
    "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a", "sim wrist camera must stay as V3.58"
w49 = (ROOT / "NERO_V349_Real_Wrist_Camera.py").read_text(encoding="utf-8").replace('VERSION = "V3.59"', 'VERSION = "V3.58"', 1)
assert H(w49.encode("utf-8")) == "d5955ce5c9c8acf8d654faa163da49e1b54ad21775a5ec060ac353a13c76a622", "USB wrist module changed"
marker = SP.MARKER
assert marker == "V3.59 STRATEGY PROFILE + CONTROL KNOBS + SIM/REAL SHARED DECISION LAYER"
for fn in ("NERO_Strategy_Profile.py", "NERO_V312_Console.py", "NERO_V325_Camera.py"):
    assert marker in (ROOT / fn).read_text(encoding="utf-8"), fn
assert "V3.59 STRATEGY PROFILE ADVICE" in (ROOT / "NERO_V2_7_1_Agent.py").read_text(encoding="utf-8")
assert "NERO_Strategy_Profile.py" in (ROOT / "start_nero.sh").read_text() and \
    'python3 -m py_compile "$STRAT359"' in (ROOT / "start_nero_sim.sh").read_text()
print("[Z1] V3.59 versions & markers; NERO_V308 / backend / sim wrist camera untouched; USB wrist module = V3.58 + VERSION")

# ------------------------------------------------------------------ Z2: knob table + profile JSON
EXPECT = {  # name: (group, default, lo, hi)  -- AGENTS.md 8.1
    "w_dist": ("A", 0.45, 0.0, 1.0), "w_clearance": ("A", 0.35, 0.0, 1.0), "w_yaw_align": ("A", 0.20, 0.0, 1.0),
    "w_base_joint_penalty": ("B", 1.0, 0.1, 2.0), "w_elbow_lift": ("B", 0.5, 0.0, 1.0), "w_limit_margin": ("B", 0.6, 0.1, 1.5),
    "hover_height_mm": ("C", 60.0, 40.0, 90.0), "wrist_refine_tol_mm": ("C", 1.5, 0.8, 3.0),
    "grip_open_margin_mm": ("C", 14.0, 8.0, 22.0), "grip_close_squeeze_mm": ("C", 2.5, 1.5, 4.5),
    "cruise_speed_ratio": ("C", 0.5, 0.2, 1.0), "descent_speed_ratio": ("C", 0.286, 0.10, 0.60),
    "min_held_width_mm": ("D", 32.0, 28.0, 36.0), "max_retry_per_target": ("D", 1, 0, 2),
}
assert len(SP.KNOBS) == 16 and len(SP.KNOB_FAMILIES) == 14 and [g for g, _ in SP.GROUPS] == ["A", "B", "C", "D"]
for n, (g, d, lo, hi) in EXPECT.items():
    k = SP.KNOB_BY_NAME[n]
    assert (k.group, k.default, k.lo, k.hi) == (g, d, lo, hi), (n, k.group, k.default, k.lo, k.hi)
assert SP.KNOB_BY_NAME["slot_fill_order"].choices == ("numeric", "far_to_near", "left_to_right", "min_joint_move")
assert SP.KNOB_BY_NAME["slot_fill_order"].default == "numeric"           # = V3.04 behaviour
assert SP.KNOB_BY_NAME["wrist_refine_enable"].default is True

p = SP.StrategyProfile()
assert p.set("hover_height_mm", 150) and p["hover_height_mm"] == 90.0
assert p.set("max_retry_per_target", "7") and p["max_retry_per_target"] == 2
assert p.set("w_dist", "abc") and p["w_dist"] == 0.45
assert p.set("slot_fill_order", "random") and p["slot_fill_order"] == "numeric"
assert p.set("wrist_refine_enable", "false") is None and p["wrist_refine_enable"] is False
assert p.set("max_retry_per_target", 0.6) is None and p["max_retry_per_target"] == 1
try:
    p.set("no_such_knob", 1)
    raise AssertionError("unknown knob accepted")
except KeyError:
    pass
p = SP.StrategyProfile({"w_clearance": 0.9, "slot_fill_order": "far_to_near", "bogus": 3}, name="dense test")
assert any("bogus" in w for w in p.warnings) and p.diff_from_default() == {"w_clearance": 0.9, "slot_fill_order": "far_to_near"}
path = p.save()
assert path == os.path.join(TMP, "strategies", "dense_test.json") and os.path.isfile(path)
d = json.load(open(path, encoding="utf-8"))
assert d["schema"] == SP.SCHEMA and d["version"] == "V3.59" and d["bounds"]["hover_height_mm"] == [40.0, 90.0]
q = SP.StrategyProfile.load(path)
assert q.values == p.values and q.name == "dense test"
flat = SP.StrategyProfile.from_dict({"hover_height_mm": 70})                   # flat dict also accepted
assert flat["hover_height_mm"] == 70.0
assert [n for n, _ in SP.list_presets()] == ["dense_test"]
space = SP.optimizer_space()
assert len(space) == 16 and ("hover_height_mm", "float", 40.0, 90.0, ()) in space
shipped = SP.list_presets(str(ROOT / "config" / "strategies"))
assert [n for n, _ in shipped] == ["default", "dense_safe", "speed_first"], shipped
for n, fp in shipped:
    sp = SP.StrategyProfile.load(fp)
    assert not sp.warnings, (n, sp.warnings)
assert SP.StrategyProfile.load(dict(shipped)["default"]).values == SP.default_values()
print(f"[Z2] 16 knobs (4 groups, 14 items) with doc defaults/bounds; clamping; JSON round trip; "
      f"{len(shipped)} shipped presets valid")

# ------------------------------------------------------------------ Z3: shared decision layer
scene = SP.DEMO_SCENE
objs = [{"id": k, "xyz_m": v["xyz_m"], "width_mm": v["grasp_geometry"]["effective_width_mm"]} for k, v in scene["objects"].items()]
tcp = scene["tcp"]["xyz_m"]
near = SP.rank_targets(objs, tcp, SP.StrategyProfile({"w_dist": 1.0, "w_clearance": 0.0, "w_yaw_align": 0.0}))
clear = SP.rank_targets(objs, tcp, SP.StrategyProfile({"w_dist": 0.1, "w_clearance": 1.0, "w_yaw_align": 0.0}))
assert near[0]["id"] == "egg_2" and near[0]["dist_mm"] < 50, near[0]
assert {clear[0]["id"], clear[1]["id"]} == {"egg_4", "egg_5"} and clear[0]["clearance_mm"] > 100
assert near[0]["tight"] and not clear[0]["tight"] and [r["rank"] for r in near] == [1, 2, 3, 4, 5]
# yaw: lying egg whose long axis is radial needs no wrist twist; one rotated 90 deg is the worst
yaw_objs = [{"id": "radial", "xyz_m": [0.40, 0.0, 0.03], "yaw_deg": 0.0}, {"id": "cross", "xyz_m": [0.40, 0.0, 0.03], "yaw_deg": 90.0}]
r = {x["id"]: x for x in SP.rank_targets(yaw_objs, [0.4, 0.0, 0.3], SP.StrategyProfile({"w_dist": 0, "w_clearance": 0, "w_yaw_align": 1}))}
assert r["radial"]["s_yaw"] == 1.0 and r["cross"]["s_yaw"] < 1e-6 and abs(abs(r["cross"]["yaw_delta_deg"]) - 90) < 1e-6
assert SP.long_axis_yaw_from_quat([0, 0, 0, 1]) is None                                 # upright egg
s45 = math.sin(math.radians(45))
assert abs(SP.long_axis_yaw_from_quat([0.0, s45, 0.0, s45]) - 0.0) < 1e-6               # z axis rotated onto +x
slots = [{"name": k, "xyz_m": v["center_xyz_m"], "occupied": k == "slot_1"} for k, v in scene["receptacles"]["egg_holder"]["slots"].items()]
o = lambda mode, ref=None: SP.order_slots(slots, SP.StrategyProfile({"slot_fill_order": mode}), ref_xyz=ref)
assert o("numeric") == ["slot_2", "slot_3", "slot_4", "slot_5", "slot_6", "slot_7", "slot_8"]
assert o("far_to_near")[0] == "slot_5" and o("far_to_near")[-1] == "slot_4"
assert o("left_to_right")[:4] == ["slot_5", "slot_6", "slot_7", "slot_8"]
assert o("min_joint_move", ref=[-0.33, 0.30, 0.03])[0] == "slot_8"
assert "slot_1" not in o("far_to_near")
# IK preference
lim = [(-2.7, 2.7)] * 7
q_now = [0.0] * 7
pr = SP.StrategyProfile()
q_base = [0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]          # swings J1
q_wrist = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6]         # same angle on J7
assert SP.ik_preference_cost(q_base, q_now, lim, pr) > SP.ik_preference_cost(q_wrist, q_now, lim, pr)
best, costs = SP.pick_ik_solution([q_base, q_wrist], q_now, lim, pr)
assert best == q_wrist and len(costs) == 2
q_edge = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.65]         # 3 deg from the limit
q_mid = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0]
hi_lim = SP.StrategyProfile({"w_limit_margin": 1.5})
assert SP.pick_ik_solution([q_edge, q_mid], q_now, lim, hi_lim)[0] == q_mid
ez = lambda q: 0.30 if q[3] > 0 else 0.10
lift = SP.StrategyProfile({"w_elbow_lift": 1.0})
assert SP.pick_ik_solution([[0, 0, 0, -0.2, 0, 0, 0], [0, 0, 0, 0.2, 0, 0, 0]], q_now, lim, lift, ez)[0][3] == 0.2
assert SP.grip_open_width_mm(42.0, pr) == 56.0 and SP.grip_close_width_mm(42.0, pr) == 39.5
assert SP.grip_open_width_mm(95.0, pr) == 100.0
assert SP.held_check(40.0, 42.0, pr)[0] and not SP.held_check(25.0, 42.0, pr)[0] and not SP.held_check(46.0, 42.0, pr)[0]
assert "EMPTY_GRASP" in SP.held_check(25.0, 42.0, pr)[1]
assert abs(SP.hover_z(0.03, 0.0, pr) - 0.09) < 1e-12 and abs(SP.hover_z(-0.2, 0.0, pr) - 0.05) < 1e-12
assert SP.speed_scale(pr, "descent") == 0.286 and SP.speed_scale(pr) == 0.5
print(f"[Z3] ranking near-first {near[0]['id']} vs clear-first {clear[0]['id']}; yaw / 4 fill orders / IK preference / "
      "grip widths / held check / hover z verified")

# ------------------------------------------------------------------ Z4: sim bridge (Agent hook + env)
env = SP.to_sim_env(pr, snapshot_path=os.path.join(TMP, "active.json"))
assert env["NERO_V341_LIMIT_MARGIN_DEG"] == "8.00" and abs(float(env["NERO_V304_FINAL_APPROACH_SLOW"]) - 3.5) < 0.01
assert env["NERO_V304_EGG_APERTURE_MARGIN_MM"] == "14.00" and env["NERO_V303_HOVER_ABOVE_ENTRY_M"] == "0.0600"
assert env["NERO_V359_STRATEGY_PROFILE"] == os.path.join(TMP, "active.json")
BACK = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text(encoding="utf-8")
for k, _knob, _doc in SP.SIM_ENV_DOC:                     # every injected variable is really read by the backend
    assert f'os.environ.get("{k}"' in BACK, k
assert SP.to_sim_env(SP.StrategyProfile({"w_limit_margin": 1.5}))["NERO_V341_LIMIT_MARGIN_DEG"] == "15.00"

ASRC = (ROOT / "NERO_V2_7_1_Agent.py").read_text(encoding="utf-8")
amod = ast.parse(ASRC)
seg = "\n".join(ast.get_source_segment(ASRC, n) for n in amod.body
                if (isinstance(n, ast.FunctionDef) and n.name in ("_v359_strategy_advise", "_v304_compact_for_model"))
                or (isinstance(n, ast.Assign) and any(getattr(t, "id", "").startswith(("_V304_", "_V359_")) for t in n.targets)))
ga = {"os": os}
exec(seg, ga)
adv_fn, compact = ga["_v359_strategy_advise"], ga["_v304_compact_for_model"]


def obs_result():
    objs_ = {k: dict(v, quat_xyzw=[0, 0, 0, 1], location="egg_basket") for k, v in scene["objects"].items()}
    sl = {k: dict(v, occupied=False, occupied_by=None, entry_axis=[0, 0, -1]) for k, v in scene["receptacles"]["egg_holder"]["slots"].items()}
    return {"ok": True, "tool": "observe_scene", "contract": {"x": 1}, "state_before": {},
            "scene": {"tcp": {"xyz_m": tcp, "quat_xyzw": [0, 0, 0, 1]}, "objects": objs_,
                      "receptacles": {"egg_holder": {"slots": sl}},
                      "egg_holder_fill_order": {"next_empty_slot": "slot_1", "empty_slots_in_order": sorted(sl), "rule": "numeric"}}}


res = obs_result()
assert adv_fn("observe_scene", res) is res                                  # hook off -> identity
apath = os.path.join(TMP, "agent_profile.json")
SP.StrategyProfile({"w_dist": 1.0, "w_clearance": 0.0, "w_yaw_align": 0.0}, name="near").save(apath)
os.environ["NERO_V359_STRATEGY_PROFILE"] = apath
before = json.dumps(res, sort_keys=True)
out = adv_fn("observe_scene", res)
assert json.dumps(res, sort_keys=True) == before                           # original untouched
assert out["scene"]["v359_strategy"]["next_target"] == "egg_2" and out["scene"]["egg_holder_fill_order"]["next_empty_slot"] == "slot_1"
small = compact("observe_scene", out)
assert small["scene"]["v359_strategy"]["target_order"][0] == "egg_2" and "quat_xyzw" not in small["scene"]["objects"]["egg_2"]
time.sleep(0.02)
SP.StrategyProfile({"w_dist": 0.1, "w_clearance": 1.0, "slot_fill_order": "far_to_near"}, name="clear").save(apath)
os.utime(apath, (time.time() + 5, time.time() + 5))                        # make sure the mtime differs
out2 = adv_fn("observe_scene", obs_result())
assert out2["scene"]["v359_strategy"]["profile"] == "clear" and out2["scene"]["v359_strategy"]["next_target"] in ("egg_4", "egg_5")
assert out2["scene"]["egg_holder_fill_order"]["next_empty_slot"] == "slot_5"
ver = {"ok": True, "tool": "verify_object_in_slot", "v306_next": {
    "remaining_eggs_outside_holder": ["egg_1", "egg_2", "egg_3", "egg_4"], "next_egg": "egg_1", "next_empty_slot": "slot_2",
    "empty_slots_in_order": ["slot_2", "slot_3", "slot_4", "slot_6", "slot_7", "slot_8"], "occupied_slots": {"slot_5": "egg_5"}}}
nxt = adv_fn("verify_object_in_slot", ver)["v306_next"]
assert nxt["next_egg"] == "egg_4" and nxt["remaining_eggs_outside_holder"][0] == "egg_4" and nxt["next_empty_slot"] == "slot_6"
assert ver["v306_next"]["next_egg"] == "egg_1"
assert adv_fn("plan_grasp", {"ok": True}) == {"ok": True}
os.environ.pop("NERO_V359_STRATEGY_PROFILE")
code = "import NERO_V2_7_1_Agent as A; print('V3.59 STRATEGY PROFILE' in A.SYSTEM_INSTRUCTIONS)"
e_off = {k: v for k, v in os.environ.items() if k != "NERO_V359_STRATEGY_PROFILE"}
r_off = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=e_off, capture_output=True, text=True)
r_on = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=dict(e_off, NERO_V359_STRATEGY_PROFILE=apath),
                      capture_output=True, text=True)
assert r_off.stdout.strip() == "False" and r_on.stdout.strip() == "True", (r_off.stdout, r_off.stderr[-400:], r_on.stderr[-400:])
script = os.path.join(TMP, "fake_sim.sh")
with open(script, "w") as f:
    f.write('echo "MARGIN=$NERO_V304_EGG_APERTURE_MARGIN_MM SCENE=$NERO_SCENE PROF=$NERO_V359_STRATEGY_NAME"\n')
runner = CON.SimRunner(script)
runner.strategy_env = SP.to_sim_env(SP.StrategyProfile({"grip_open_margin_mm": 18}, name="m18"))
runner.start(extra_env=CON.scene_env("fill_egg_holder"))
lines = []
t_end = time.time() + 10
while time.time() < t_end:
    ln = runner.lines.get(timeout=10)
    if ln is None:
        break
    lines.append(ln)
assert any("MARGIN=18.00 SCENE=fill_egg_holder PROF=m18" in ln for ln in lines), lines
r0 = CON.SimRunner(script)
r0.start(extra_env=CON.scene_env("fill_egg_holder"))
l0 = [r0.lines.get(timeout=10)]
assert "MARGIN= " in l0[0], l0                                               # default: nothing injected
print("[Z4] sim env mapping reproduces backend defaults; Agent hook identity when off, live re-read when on; "
      "prompt rule only with hook; SimRunner passes knob env to start_nero_sim.sh")

# ------------------------------------------------------------------ Z5: real side
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
LIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708)]
tool_T = [[0.0, 0.0, 1.0, CAM.DEFAULT_TCP_IN_FLANGE[0]], [-1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, CAM.DEFAULT_TCP_IN_FLANGE[2]],
          [0.0, 0.0, 0.0, 1.0]]
q0r = [math.radians(v) for v in (0.0, 30.0, 0.0, 80.0, 0.0, 40.0, 0.0)]
T0 = CS.tool_pose(U, J, q0r, "base_link", tool_T)
tgt = [T0[0][3] + 0.03, T0[1][3] - 0.04, T0[2][3] - 0.02]
qa, ia = CAM.plan_hover_target(U, J, LIM, q0r, tool_T, tgt)
assert qa is not None and "v359_cost" not in ia
cost_fn = lambda q: SP.ik_preference_cost(q, q0r, LIM, pr)
qb, ib = CAM.plan_hover_target(U, J, LIM, q0r, tool_T, tgt, prefer_fn=cost_fn)
assert qb is not None and ib["err_mm"] < 2.0 and ib["v359_cost"] <= cost_fn(qa) + 1e-9
import unittest.mock as mock  # noqa: E402


class FakeVar:
    def __init__(self, master=None, value=None, **_kw):
        self._v = value

    def get(self):
        return self._v

    def set(self, v):
        self._v = v

    def trace_add(self, *_a, **_k):
        return "t"

    trace = trace_add


mtk = mock.MagicMock()
for nm in ("StringVar", "BooleanVar", "IntVar", "DoubleVar"):
    setattr(mtk, nm, FakeVar)
cam = CAM.CameraTabUI(mtk, mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), mock.MagicMock(),
                      mock.MagicMock(), lambda: {"speed_percent": 10}, lambda *a: None, lambda *a: True)
cam.tv_eggs = mock.MagicMock()
cam.tv_eggs.selection.return_value = ()
cam.actual_q = lambda: None
cam.eggs = [dict(det_id=f"e{i}", xyz_m=v["xyz_m"], status="ok", trusted=True, size_mm=[55.0, 42.0])
            for i, v in enumerate(scene["objects"].values(), 1)]
assert cam.strategy is None and cam._selected_hover_target()["det_id"] == "e1"     # V3.58 behaviour
old_hover = cam.v_hover_mm.get()
assert old_hover == "100"
cam.set_strategy(SP.StrategyProfile({"w_dist": 0.1, "w_clearance": 1.0}))
assert cam.v_hover_mm.get() == "60" and cam._selected_hover_target()["det_id"] in ("e4", "e5")
cam.set_strategy(SP.StrategyProfile({"hover_height_mm": 85}))
assert cam.v_hover_mm.get() == "85"
sc_, src_ = cam.strategy_scene()
assert len(sc_["objects"]) == 5 and sc_["objects"]["e1"]["grasp_geometry"]["effective_width_mm"] == 42.0 and "实机" in src_
cam.set_strategy(None)
assert cam.strategy is None and cam.v_hover_mm.get() == old_hover and cam._selected_hover_target()["det_id"] == "e1"
print(f"[Z5] plan_hover_target prefer_fn cost {ib['v359_cost']:.3f} <= default {cost_fn(qa):.3f}; Camera tab "
      "hover/ranking linked and unlinked")

# ------------------------------------------------------------------ Z6: mocked-Tk console + strategy panel
mock_tk = mock.MagicMock()
mock_ttk = mock.MagicMock()
for nm in ("StringVar", "BooleanVar", "IntVar", "DoubleVar"):
    setattr(mock_tk, nm, FakeVar)
sys.modules["tkinter"] = mock_tk
sys.modules["tkinter.ttk"] = mock_ttk
sys.modules["tkinter.font"] = mock.MagicMock()
sys.modules["tkinter.messagebox"] = mock.MagicMock()
sys.modules["tkinter.simpledialog"] = mock.MagicMock()
mock_tk.ttk = mock_ttk
mock_tk.Text.return_value.yview.return_value = (0.0, 1.0)      # log_append() reads the scroll position
mock_tk.Text.return_value.index.return_value = "1.0"
root_mock = mock.MagicMock()
root_mock.winfo_screenwidth.return_value = 1920
root_mock.winfo_screenheight.return_value = 1080
mock_tk.Tk.return_value = root_mock
root_mock.mainloop.side_effect = lambda: None

panels, cams = [], []
_P, _C = SP.StrategyPanelUI, CAM.CameraTabUI


class RecPanel(_P):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        panels.append(self)


class RecCam(_C):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        cams.append(self)


SP.StrategyPanelUI, CAM.CameraTabUI = RecPanel, RecCam
CON.run_console("sim")
CON.run_console("real")
SP.StrategyPanelUI, CAM.CameraTabUI = _P, _C
assert len(panels) == 2 and len(cams) == 2
tabs = [c.kwargs.get("text") for c in mock_ttk.Notebook.return_value.add.call_args_list]
assert "策略旋钮" in tabs and "Camera调试" in tabs
panel, ctab = panels[-1], cams[-1]
assert panel.sim_env() == {} and not panel.real_follow()
assert ctab.strategy is None and ctab.v_hover_mm.get() == "100"
panel.v_real_follow.set(True)
panel._on_link_toggle()
assert ctab.strategy is panel.profile and ctab.v_hover_mm.get() == "60"
panel.vars["hover_height_mm"].set("75")
panel._on_var("hover_height_mm")
assert panel.profile["hover_height_mm"] == 75.0 and ctab.v_hover_mm.get() == "75"
adv0 = panel.on_preview(quiet=True)
panel.vars["w_clearance"].set("1.0")
panel._on_var("w_clearance")
panel.vars["w_dist"].set("0.05")
panel._on_var("w_dist")
adv1 = panel.on_preview(quiet=True)
assert adv0["target_order"] != adv1["target_order"] or adv0["scores"] != adv1["scores"]
assert adv1["next_target"] in ("egg_4", "egg_5")
panel.vars["w_dist"].set("1.0")
panel._on_var("w_dist")
panel.vars["w_clearance"].set("0")
panel._on_var("w_clearance")
assert panel.on_preview(quiet=True)["next_target"] == "egg_2"           # order flips back immediately
panel._on_scale("grip_open_margin_mm", "17.04")
assert panel.profile["grip_open_margin_mm"] == 17.0
panel.vars["hover_height_mm"].set("500")
panel._on_var("hover_height_mm")
assert panel.profile["hover_height_mm"] == 90.0 and panel.vars["hover_height_mm"].get() == "90"
panel.v_name.set("my preset")
saved = panel.on_save()
assert saved == os.path.join(TMP, "strategies", "my_preset.json") and os.path.isfile(saved)
panel.on_reset()
assert panel.profile.values == SP.default_values() and ctab.v_hover_mm.get() == "60"
panel.v_preset.set("my_preset")
assert panel.on_load()
assert panel.profile["hover_height_mm"] == 90.0 and panel.profile["w_dist"] == 1.0 and ctab.v_hover_mm.get() == "90"
assert ctab.strategy is panel.profile
assert not panel.on_load("does_not_exist")
panel.v_sim_inject.set(True)
e2 = panel.sim_env()
assert e2["NERO_V303_HOVER_ABOVE_ENTRY_M"] == "0.0900" and os.path.isfile(e2["NERO_V359_STRATEGY_PROFILE"])
assert SP.StrategyProfile.load(e2["NERO_V359_STRATEGY_PROFILE"]).values == panel.profile.values
panel.v_real_follow.set(False)
panel._on_link_toggle()
assert ctab.strategy is None and ctab.v_hover_mm.get() == "100"
assert not os.path.exists(str(ROOT / "config" / "strategies" / "my_preset.json"))
print("[Z6] console 「策略旋钮」 tab: knob change -> ranking + Camera hover immediately; save/reset/load; "
      "sim injection opt-in")

print("ALL V3.59 TESTS PASSED")
