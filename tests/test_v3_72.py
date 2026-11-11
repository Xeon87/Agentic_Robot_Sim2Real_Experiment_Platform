#!/usr/bin/env python3
"""V3.72 offline tests (5th simulation scene sort_items, SO-101 gripper):
D1 Versions V3.72 + markers; NERO_V308 / V326 / Agent / Camera tab frozen; backend = V3.71 + one sort_items block inside
   the RUN TASK A2 hook (the hook region of test_v3_62 P1, so "backend minus the hook" is still V3.60).
D2 Scene data: sort_items is a base scene with the SO-101 gripper; TF / hold names = the 4 items; MoveIt boxes = the 10
   bin walls / floors; bin geoms protected (sort_bin_); cup / safe scenes unchanged.
D3 Scene MJCF (stub robot model): SO-101 names the backend needs, hold_obj_* equalities, obj_* item geoms, bins; the items
   lie still on the table (1 s physics).
D4 observe_scene gives labels / sizes / bins but no category (the agent decides by the prompt); tool contract: 5 tools,
   enums, refusals (unknown item / box, place without an item, safe pose while holding).
D5 Twin (scripted, fruit -> red, others -> blue): default layout + 3 jittered layouts, every item in its bin.
D6 Backend path: gui_run_task -> SortBackendIO -> fake backend core on the twin, agent loop with a scripted model that
   assigns by label: 4/4 sorted, s = 1, result table + score logged; camera pose source refused.
D7 Console scene list, start_nero_sim.sh, start_nero.sh know sort_items / the new module; release folder unchanged.
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v372_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
os.environ["NERO_V368_TASK_DIR"] = str(TMP / "task")
for k in ("NERO_SCENE", "NERO_V343_ACTIVE_SCENE", "NERO_V362_CUP_SOLREF", "NERO_V308_DEPTH_NOISE_K", "NERO_V372_ITEM_SEED"):
    os.environ.pop(k, None)
H = lambda b: hashlib.sha256(b).hexdigest()


def snapshot():
    out = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rel = line.split(None, 1)[1].strip()
            out[rel] = H((ROOT / rel).read_bytes()) if (ROOT / rel).is_file() else None
    return out


SNAP0 = snapshot()
import numpy as np                          # noqa: E402
import NERO_V312_Console as CON             # noqa: E402
import NERO_V313_Real_Robot as RR           # noqa: E402
import NERO_V343_Scenes as SC               # noqa: E402
import NERO_V347_Yolo_Perception as YP      # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49   # noqa: E402
import NERO_V362_Cup_Skills as CK           # noqa: E402
import NERO_V372_Sort_Skills as S           # noqa: E402

try:
    import mujoco
    HAVE_MJ = True
except Exception:
    HAVE_MJ = False

# ------------------------------------------------------------------ D1
for m in (CON, RR, YP, W49, S):
    assert m.VERSION in ("V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3"), (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3") and SC.SCENES_REV in ("V3.72", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
for fn, h in (("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a"),
              ("NERO_V2_7_1_Agent.py", "3e52d05ea84d94c8359d5875fa0fee8c2cc42d612b52e59109e1166912439696"),
              ("NERO_V325_Camera.py", "82f2416d73671e7018d09d0cd72c17186154934a315b47c0d19fdb0427afd931")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
BK = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text(encoding="utf-8")
a = BK.index('        if _v343_scene_name() == "stack_cups":   # V3.62 STACK CUPS AGENT')
b = BK.index('        if _v343_scene_name() != "fill_egg_holder":\n            append_gui_log(f"[V3.43 SCENE] RUN TASK blocked')
hook = BK[a:b]
assert H((BK[:a] + BK[b:]).encode()) == "ffbd630d70edf57a49929363134ceaae3b95fe4675e0887509d3dc5d7b08f1ed"
assert 'if _v343_scene_name() == "sort_items":   # V3.72 SORT ITEMS AGENT' in hook and "_V372.gui_run_task(" in hook
assert BK.count("# V3.43 A2 BEGIN") == BK.count("# V3.43 A2 END") == 11
for fn, mk in (("NERO_V372_Sort_Skills.py", S.MARKER), ("NERO_V343_Scenes.py", "V3.72 SORT ITEMS SCENE"),
               ("NERO_V362_Cup_Skills.py", "V3.72 TWIN WORLD"), ("NERO_V312_Console.py", "V3.72 SORT ITEMS SCENE")):
    assert mk in (ROOT / fn).read_text(encoding="utf-8"), (fn, mk)
print("[D1] V3.72; frozen modules unchanged; backend = V3.71 + the sort_items block inside the RUN TASK A2 hook")

# ------------------------------------------------------------------ D2
assert list(SC.SCENES)[:5] == ["fill_egg_holder", "store_in_safe", "stack_cups", "stack_cups_so101", "sort_items"]   # V4.8.3: + tidy_desk
os.environ["NERO_SCENE"], os.environ["NERO_V343_ACTIVE_SCENE"] = "sort_items", "sort_items"
try:
    assert SC.active_scene() == "sort_items" and SC.gripper_variant() == "so101" and SC.selected_variant() == "sort_items"
    assert sorted(SC.tf_body_map()) == sorted(SC.SORT_ITEM_NAMES)
    assert SC.hold_object_names()[:4] == list(SC.SORT_ITEM_NAMES)
    pb = SC.planning_boxes()
    assert len(pb) == 10 and all(n.startswith("sort_bin_") for n, _c, _s in pb) and SC.planning_rotated_boxes(None, None) == []
finally:
    os.environ.pop("NERO_SCENE"), os.environ.pop("NERO_V343_ACTIVE_SCENE")
assert "sort_bin_" in SC.PROTECTED_PREFIXES and "safe_body_" in SC.PROTECTED_PREFIXES
assert SC.active_scene() == "fill_egg_holder" and SC.gripper_variant() == "official"
assert {SC.SORT_ITEMS[n]["label"] for n in SC.SORT_ITEM_NAMES} == {"banana", "apple", "computer mouse", "marker pen"}
assert SC.SORT_BINS["red"]["centre"][1] < 0 < SC.SORT_BINS["blue"]["centre"][1]
print("[D2] sort_items: SO-101 gripper, 4 items (TF / hold), 10 bin boxes for MoveIt, bins protected; other scenes unchanged")

# ------------------------------------------------------------------ D3
os.environ["NERO_SCENE"], os.environ["NERO_V343_ACTIVE_SCENE"] = "sort_items", "sort_items"
try:
    td = Path(tempfile.mkdtemp(dir=TMP))
    pth = td / "nero_motor.xml"
    pth.write_text("<mujoco model='t'><compiler angle='radian'/><asset></asset><worldbody>"
                   "<body name='link7' pos='0 0 0.8'><freejoint/><geom type='sphere' size='0.01' contype='0' "
                   "conaffinity='0'/></body></worldbody></mujoco>")
    out = Path(SC.make_store_in_safe_xml(str(pth)))
finally:
    os.environ.pop("NERO_SCENE"), os.environ.pop("NERO_V343_ACTIVE_SCENE")
xml = out.read_text()
assert out.name == "_nero_motor_v343_sort_items_tmp.xml" and xml.count('name="gripper_joint1"') == 1
for n in SC.SORT_ITEM_NAMES:
    assert f'name="hold_{n}"' in xml and f'<body name="{n}"' in xml
if HAVE_MJ:
    m = mujoco.MjModel.from_xml_path(str(out))
    d = mujoco.MjData(m)
    for kind, n in ((mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint1"), (mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_force"),
                    (mujoco.mjtObj.mjOBJ_GEOM, "finger1_rubber_pad"), (mujoco.mjtObj.mjOBJ_GEOM, "sort_bin_red_floor"),
                    (mujoco.mjtObj.mjOBJ_GEOM, "obj_apple_body"), (mujoco.mjtObj.mjOBJ_EQUALITY, "hold_obj_marker")):
        assert mujoco.mj_name2id(m, kind, n) >= 0, n
    mujoco.mj_forward(m, d)
    p0 = {n: d.body(n).xpos.copy() for n in SC.SORT_ITEM_NAMES}
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    drift = {n: float(np.linalg.norm(d.body(n).xpos - p0[n])) * 1000 for n in SC.SORT_ITEM_NAMES}
    assert max(drift.values()) < 3.0, drift
    print(f"[D3] scene MJCF loads; items still after 1 s (max drift {max(drift.values()):.2f} mm)")
else:
    print("[D3] scene MJCF built (MuJoCo not installed: load / physics skipped)")

# ------------------------------------------------------------------ D4 + D5 twin
if HAVE_MJ:
    io = S.SortTwinIO()
    sk = S.SortSkills(io)
    obs = sk.run_tool("observe_scene", {})
    sc = obs["scene"]
    assert obs["ok"] and len(sc["items"]) == 4 and [b_["box"] for b_ in sc["boxes"]] == ["red", "blue"]
    assert all("category" not in it for it in sc["items"]) and "category" not in json.dumps(sc)
    assert {it["label"] for it in sc["items"]} == {"banana", "apple", "computer mouse", "marker pen"}
    assert sk.run_tool("pick_object", {"item_id": "item_9"})["reason"] == "UNKNOWN_ITEM"
    assert sk.run_tool("place_in_box", {"box": "red"})["reason"] == "NO_ITEM_HELD"
    assert sk.run_tool("place_in_box", {"box": "green"})["reason"] == "UNKNOWN_BOX"
    assert [t_["name"] for t_ in S.SORT_TOOLS] == ["observe_scene", "pick_object", "place_in_box", "check_bins",
                                                   "return_to_safe_pose"]
    assert "decide its category and its target bin" in S.SORT_SYSTEM_INSTRUCTIONS
    r = sk.run_tool("pick_object", {"item_id": "item_4"})
    assert r["ok"] and sk.held == "item_4", r
    assert sk.run_tool("return_to_safe_pose", {})["reason"] == "ITEM_HELD_PLACE_FIRST"
    print("[D4] observe: labels / sizes / bins, no category; refusals; pick holds the marker")
    rows = []
    for seed in (None, 1, 2, 3):
        res, state, _io, sk_ = S.twin_episode(log=lambda s_: None, seed=seed)
        want = {iid: ("red" if SC.SORT_ITEMS[b_]["category"] == "fruit" else "blue") for iid, b_ in S.ITEM_IDS.items()}
        fails = [(r_["tool"], r_.get("reason"), (r_.get("physical_verification") or {}).get("note")) for r_ in res
                 if not r_["ok"]]
        if seed == 1 and getattr(S, "HELD_MARGIN_DEG", 0.0) >= 10.0:
            # V3.77 HELD MARGIN (intended change): in this jittered layout the banana has no grasp + place with every
            # joint >= 10 deg from its limit while held (best 8.4 deg, -90 deg SO-101 mount, J6 -42..+54 deg) ->
            # GRASP_UNREACHABLE with the margin named, the other three items are sorted
            assert state["item_1"] is None and all(state[k_] == want[k_] for k_ in ("item_2", "item_3", "item_4")), (
                seed, state, fails)
            assert fails and fails[0][1] == "GRASP_UNREACHABLE" and "needs >= 10 deg" in (fails[0][2] or ""), fails
        else:
            assert state == want, (seed, state, fails)
        rows.append(seed)
    print(f"[D5] twin: layouts {rows} - banana / apple -> red, mouse / marker -> blue (seed 1: the banana is refused by "
          "the V3.77 10 deg held margin, the rest sorted)")
    import v372_backend_twin as SB  # noqa: E402
    r6 = SB.run_backend_episode()
    assert r6["started"].get("label") == "RUN TASK (sort_items)" and not r6["started"].get("error"), r6["started"]
    assert r6["state"] == {"item_1": "red", "item_2": "red", "item_3": "blue", "item_4": "blue"}, r6["state"]
    assert r6["scores"] and r6["scores"][0]["s"] == 1.0
    assert any(l.startswith("[V3.72 SORT RESULT]") for l in r6["log"]) and any("[V3.72 SKILL SCORE]" in l for l in r6["log"])
    r6c = SB.run_backend_episode(pose_mode="camera")
    assert not r6c["started"] and any("camera item recognition is not built" in l for l in r6c["log"])
    print("[D6] backend path (SortBackendIO + agent loop, scripted model): 4/4 sorted, s=1; camera source refused")
else:
    print("[D4-D6] skipped: MuJoCo not installed")

# ------------------------------------------------------------------ D7
assert ("sort_items", SC.SCENES["sort_items"]["label"]) in CON.SIM_SCENES and [k for k, _ in CON.SIM_SCENES] == list(SC.SCENES)
sim_sh = (ROOT / "start_nero_sim.sh").read_text(encoding="utf-8")
assert "  sort_items)" in sim_sh and "NERO_V372_Sort_Skills.py" in sim_sh
assert "NERO_V372_Sort_Skills.py" in (ROOT / "start_nero.sh").read_text(encoding="utf-8")
SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("[D7] console / launchers know sort_items; release folder unchanged")
print("ALL V3.72 TESTS PASSED")
