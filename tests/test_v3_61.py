#!/usr/bin/env python3
"""V3.61 offline unit tests (stack_cups scene, offline cup study, cup skill knobs, collapsible skill tab):
M1  Versions V3.61 + markers; NERO_V308 / NERO_V326 frozen; backend, Agent and Camera tab byte-identical to V3.60;
    USB wrist module = V3.58 except VERSION.
M2  Scene registry: 3 scenes (fill_egg_holder default, store_in_safe, stack_cups); console drop-down = registry;
    scene_env('stack_cups'); the launcher's case statement accepts stack_cups and still refuses unknown names.
M3  store_in_safe unchanged: generated XML byte-identical to V3.60 (sha256 ad6dfbcb...); its TF / hold / MoveIt data
    unchanged; with no scene set the module still answers with the safe data (test_v3_44 C5 contract).
M4  stack_cups XML: 2 cups (24 tilted wall boxes + bottom + rim, red outside / white inside visuals), hold CONNECTs
    for both cups, no MoveIt boxes, no hinge; compiles in MuJoCo and both cups stand upright (skipped w/o MuJoCo).
M5  Physics (MuJoCo): nested height ~25.3 mm; with the chosen knobs (release 85 mm above the nested height, xy error
    = tolerance 7.5 mm, tilt 3 deg) cup_1 nests fully in 4 error directions; a 50 mm miss fails; a low release (10 mm)
    with 9 mm error jams part-way (stacked, not fully nested).
M6  Study file config/cup_stack_study_v361.json == knob defaults (85 mm, 7.5 mm, 25.3 mm, rim angles +40 / -55 deg).
M7  URDF strict 6-DoF IK: pose_ik round trip; plan_cup_stack(defaults) reachable with every joint >= 8 deg from its
    limits (V3.41 rule); a cup at x = -0.55 is not reachable at the transport height.
M8  Knobs / skills: group 6 'cup_stacking' (9 knobs, wired 'study'), built-in CS 'Cup Stack', codes unique, shipped
    JSON == definitions, skill_display_name = 'name [XX]'.
M9  Mocked Tk: every section of the skill tab folds / unfolds on its own, collapse_all / expand_all; sim mode hides
    the dry-run section; preset labels and the status banner carry the codes; console builds (sim + real); nothing
    in the release folder is written.
"""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="nero_v361_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
for k in ("NERO_SCENE", "NERO_V343_ACTIVE_SCENE"):
    os.environ.pop(k, None)
H = lambda b: hashlib.sha256(b).hexdigest()


def snapshot():
    out = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rel = line.split(None, 1)[1].strip()
            out[rel] = H((ROOT / rel).read_bytes()) if (ROOT / rel).is_file() else None
    out["__config__"] = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "config").rglob("*") if p.is_file())
    return out


SNAP0 = snapshot()
import NERO_V312_Console as CON            # noqa: E402
import NERO_V313_Real_Robot as RR          # noqa: E402
import NERO_V325_Camera as CAM             # noqa: E402
import NERO_V343_Scenes as SC              # noqa: E402
import NERO_V347_Yolo_Perception as YP     # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49  # noqa: E402
import NERO_V359_Strategy as ST            # noqa: E402
import NERO_V361_Cup_Stack as CUP          # noqa: E402

CAM.CALIB_PATH = str(TMP / "camera_calib.json")
CAM.LOCATE_PATH = str(TMP / "locate_settings.json")
CAM.CALIB_DIR = str(TMP / "cameracalib")
HAVE_MJ = CUP.mujoco_available()

# ------------------------------------------------------------------ M1
V362 = ST.VERSION >= "V3.62"           # V3.62 intentionally changed: backend A2 block, cup origin, cup knobs (CHANGELOG)
for m in (CON, RR, YP, W49, ST):
    assert m.VERSION in ("V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10"), (m.__name__, m.VERSION)
assert CUP.VERSION == "V3.61"
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10")
assert SC.VERSION == "V3.44" and SC.SCENES_REV in ("V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
assert H((ROOT / "NERO_V326_Wrist_Camera.py").read_bytes()) == "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a"
FROZEN_V360 = {"NERO_V2_7_1_Unified_Backend.py": "ffbd630d70edf57a49929363134ceaae3b95fe4675e0887509d3dc5d7b08f1ed",
               "NERO_V2_7_1_Agent.py": "3e52d05ea84d94c8359d5875fa0fee8c2cc42d612b52e59109e1166912439696",
               "NERO_V325_Camera.py": "82f2416d73671e7018d09d0cd72c17186154934a315b47c0d19fdb0427afd931"}
for fn, h in FROZEN_V360.items():
    if V362 and fn == "NERO_V2_7_1_Unified_Backend.py":
        continue                         # V3.62: one A2 block gained the stack_cups RUN TASK hook (test_v3_62 pins it)
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed in V3.61"
w49 = re.sub(r'VERSION = "V3\.\d+"', 'VERSION = "V3.58"', (ROOT / "NERO_V349_Real_Wrist_Camera.py").read_text(encoding="utf-8"), count=1)
assert H(w49.encode()) == "d5955ce5c9c8acf8d654faa163da49e1b54ad21775a5ec060ac353a13c76a622"
for fn, mk in (("NERO_V343_Scenes.py", "V3.61 STACK CUPS SCENE"), ("NERO_V361_Cup_Stack.py", CUP.MARKER),
               ("NERO_V359_Strategy.py", "V3.61 COLLAPSIBLE SKILL TAB + CODES AFTER SKILL NAMES"),
               ("NERO_V312_Console.py", "V3.61 STACK CUPS SCENE"), ("start_nero_sim.sh", "V3.61 STACK CUPS SCENE")):
    assert mk in (ROOT / fn).read_text(encoding="utf-8"), (fn, mk)
print("[M1] V3.61 versions/markers; V308, V326, backend, Agent, Camera tab unchanged; V349 = V3.58 + VERSION")

# ------------------------------------------------------------------ M2
assert list(SC.SCENES)[:3] == ["fill_egg_holder", "store_in_safe", "stack_cups"]   # V3.64 adds the stack_cups_so101 variant
assert list(SC.SCENES)[3:] in ([], ["stack_cups_so101"], ["stack_cups_so101", "sort_items"])  # V3.72: + sort_items and SC.DEFAULT_SCENE == "fill_egg_holder"
assert [k for k, _ in CON.SIM_SCENES] == list(SC.SCENES) and CON.scene_env("stack_cups") == {"NERO_SCENE": "stack_cups"}
L = (ROOT / "start_nero_sim.sh").read_text()
case = L[L.index('SCENE344="${NERO_SCENE:-fill_egg_holder}"'):L.index('export NERO_SCENE="$SCENE344"')]
for val, ok in (("stack_cups", True), ("store_in_safe", True), ("", True), ("cups", False)):
    r = subprocess.run(["bash", "-c", 'fatal(){ echo FATAL; exit 3; }\n' + case + 'echo "OK $SCENE344"'],
                       env=dict(os.environ, NERO_SCENE=val), capture_output=True, text=True)
    assert (r.returncode == 0) == ok, (val, r.stdout, r.stderr)
os.environ["NERO_SCENE"] = "stack_cups"
assert SC.selected_scene() == SC.active_scene() == "stack_cups"
os.environ.pop("NERO_SCENE")
print("[M2] 3 scenes; console drop-down + launcher accept stack_cups")


# ------------------------------------------------------------------ M3 + M4
def build(scene, body="<body name='link7'/>"):
    os.environ["NERO_SCENE"] = os.environ["NERO_V343_ACTIVE_SCENE"] = scene
    try:
        td = Path(tempfile.mkdtemp(dir=TMP))
        p = td / "nero_motor.xml"
        p.write_text(f"<mujoco model='t'><asset></asset><worldbody>{body}</worldbody></mujoco>")
        out = Path(SC.make_store_in_safe_xml(str(p)))
        return out, out.read_text().replace(str(td.resolve()), "<TMP>"), SC.tf_body_map(), SC.hold_object_names(), \
            SC.planning_boxes(), SC.planning_rotated_boxes(None, None) if scene == "stack_cups" else None
    finally:
        os.environ.pop("NERO_SCENE")
        os.environ.pop("NERO_V343_ACTIVE_SCENE")


out, xml, tf, hold, pb, _r = build("store_in_safe")
assert out.name == "_nero_motor_v343_store_in_safe_tmp.xml"
assert H(xml.encode()) == "ad6dfbcb83f83f6ba9bb3882763414efc7cd2e6525b4980e461aef55b90f2476", "store_in_safe XML changed"
assert tf == {"obj_valuables_box": "obj_valuables_box", "obj_safe_handle": "obj_safe_handle", "obj_safe_door": "safe_door"}
assert hold[:2] == ["obj_valuables_box", "obj_safe_handle"] and len(pb) == 5
assert len(SC.planning_boxes()) == 5 and SC.tf_body_map()["obj_safe_door"] == "safe_door"      # no scene set
print("[M3] store_in_safe XML byte-identical to V3.60 (sha256 ad6dfbcb...), TF/hold/MoveIt data unchanged")

out, xml, tf, hold, pb, rot = build("stack_cups", "<body name='link7' pos='0 0 0.8'><geom type='sphere' size='0.01' "
                                                  "contype='0' conaffinity='0'/></body>")
assert out.name == "_nero_motor_v343_stack_cups_tmp.xml"
assert tf == {"obj_cup_1": "obj_cup_1", "obj_cup_2": "obj_cup_2"} and hold[:2] == ["obj_cup_1", "obj_cup_2"]
assert pb == [] and rot == []
import xml.etree.ElementTree as ET  # noqa: E402
r = ET.fromstring(Path(out).read_text())
bodies = {b.get("name"): b for b in r.iter("body")}
for c in SC.CUP_NAMES:
    gs = [g for g in bodies[c].iter("geom")]
    walls = [g for g in gs if "_wall_" in g.get("name")]
    assert len(walls) == SC.CUP["segments"] == 24 and all(g.get("material") == "v361_cup_red_mat" for g in walls)
    inner = [g for g in gs if "_inner_" in g.get("name")]
    assert len(inner) == 24 and all(g.get("contype") == "0" and g.get("material") == "v361_cup_white_mat" for g in inner)
    assert bodies[c].find("freejoint") is not None
    assert abs(sum(float(g.get("mass") or 0) for g in gs) - SC.CUP["mass"]) < 1e-4   # 6-decimal rounding
eq = {e.get("name") for e in r.iter("connect")}
assert {"hold_obj_cup_1", "hold_obj_cup_2"} <= eq and "safe_door" not in bodies and not any(n and n.startswith("egg") for n in bodies)
if HAVE_MJ:
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(out))
    d = mujoco.MjData(m)
    for _ in range(1500):
        mujoco.mj_step(m, d)
    for c in SC.CUP_NAMES:
        b = d.body(c)
        bz = SC.cup_base_point(b.xpos, b.xquat, getattr(SC, "CUP_ORIGIN", "base"))[2]       # V3.62: rim-centre origin
        assert SC.cup_axis_tilt_deg(b.xquat) < 0.5 and abs(b.xpos[0] - SC.CUP_XY[c][0]) < 0.002 and 0 < bz < 0.004
    print(f"[M4] stack_cups XML: 2 cups x ({len(walls)} walls + bottom + rim), hold CONNECTs, compiles and both cups stand")
else:
    print("[M4] stack_cups XML structure OK (MuJoCo not installed here: compile/settle check skipped)")

# ------------------------------------------------------------------ M5
K = ST.BASELINE_KNOBS
if HAVE_MJ:
    zn = CUP.nested_height()
    assert abs(zn * 1000 - K["cup_nested_height_mm"]) < 1.5, zn
    for dd in (0, 90, 180, 270):
        res = CUP.release_trial(K["cup_release_xy_tol_mm"], dd, K["cup_release_gap_mm"], K["cup_release_tilt_tol_deg"],
                                tilt_axis_deg=dd + 90, z_nested=zn)
        assert res["full_success"] and res["cup2_moved_mm"] < 5.0, (dd, res)
    bad = CUP.release_trial(50.0, 0, K["cup_release_gap_mm"], 0.0, z_nested=zn)
    assert not bad["full_success"]
    jam = CUP.release_trial(9.0, 0, 10.0, 0.0, z_nested=zn)
    assert not jam["full_success"] and jam["base_dz_mm"] > zn * 1000 + 10
    print(f"[M5] nested {zn * 1000:.1f} mm; defaults nest fully in 4 directions; 50 mm misses; low release jams "
          f"({jam['base_dz_mm']} mm)")
else:
    print("[M5] skipped: MuJoCo not installed (pip install mujoco)")

# ------------------------------------------------------------------ M6
study = json.loads((ROOT / "config" / "cup_stack_study_v361.json").read_text(encoding="utf-8"))
par = study["release_params"]
assert par["release_gap_mm"] == K["cup_release_gap_mm"] == 85.0 and par["release_xy_tol_mm"] == K["cup_release_xy_tol_mm"]
assert abs(par["z_nested_mm"] - K["cup_nested_height_mm"]) < 0.05
assert round(study["grasp_rim_angle_rel_base_deg"]) == K["cup_grasp_rim_angle_deg"] == 40.0
assert round(study["place_rim_angle_rel_base_deg"]) == -55.0 and K["cup_place_rim_angle_deg"] == (35.0 if V362 else -55.0)
assert study["layout"] == {k: list(v) for k, v in SC.CUP_XY.items()}
assert CUP.choose_release_params(study["release_sweep"]) == par
print("[M6] study file == knob defaults (release 85 mm / tol 7.5 mm / nested 25.3 mm / rim +40, -55 deg)")

# ------------------------------------------------------------------ M7
import NERO_V315_Consistency_Safety as CS  # noqa: E402
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
T = CS.tool_transform_from_sim(U)
q0 = [0.2, 0.5, 0.1, 1.3, 0.1, 0.6, 0.3]
P = CS.tool_pose(U, J, q0, "base_link", T)
q, pe, re_ = CUP.pose_ik(U, J, CUP.JOINT_LIMITS, [[0.0] * 7], T, [P[0][3], P[1][3], P[2][3]], [r_[:3] for r_ in P[:3]])
assert q is not None and pe < 5e-4 and re_ < 1.0
K61 = dict(K, cup_rim_pinch_depth_mm=25.0, cup_pregrasp_height_mm=60.0, cup_place_rim_angle_deg=-55.0)   # V3.61 values
plan = CUP.plan_cup_stack(K61, urdf=U)
assert plan["ok"] and plan["grasp"]["min_margin_deg"] >= 8.0 and plan["place"]["min_margin_deg"] >= 8.0, plan
assert [w["name"] for w in plan["waypoints"]] == ["pregrasp", "grasp", "lift", "place_above", "release"]
wz = {w["name"]: w["tcp_xyz_m"][2] for w in plan["waypoints"]}
g = SC.cup_geometry()
assert abs(wz["lift"] - (CUP.TABLE_Z0 + g["h"] + 0.020 + g["h"] - 0.025)) < 1e-4
assert abs(wz["release"] - (CUP.TABLE_Z0 + 0.0253 + 0.085 + g["h"] - 0.025)) < 1e-4
far = CUP.grasp_reachability(K61, cup_from=(-0.55, -0.13), phis=(0, 90, 180, 270), urdf=U)
assert not any(r_["ok"] for r_ in far["grasp"])
print(f"[M7] CS plan reachable: grasp margin {plan['grasp']['min_margin_deg']} deg, place {plan['place']['min_margin_deg']} "
      f"deg (>= 8); cup at x=-0.55 unreachable")

# ------------------------------------------------------------------ M8
cup_specs = [s for s in ST.KNOB_SPECS if s["group"] == "cup_stacking"]
assert [g_ for g_, _ in ST.KNOB_GROUPS][-1] == "cup_stacking" and len(cup_specs) == (11 if V362 else 9)
assert all(s["wired"] == ("sim_skill" if V362 else "study") and (s["type"] == "bool" or s["min"] <= s["default"] <= s["max"])
           for s in cup_specs)
assert len(ST.KNOB_SPECS) == (56 if V362 else 54) and ST.WIRING_LABELS["study"] == "离线试验✓"
pres = ST.builtin_strategy_presets()
assert [p.code for p in pres] == ["EB", "DP", "HD", "QH", "CS"] and len({p.code for p in pres}) == 5
cs = pres[-1]
assert cs.skill_id == "cup_stack_rim_pinch" and cs.short_name == "Cup Stack" and cs.skill_card["sim_benchmark"]["success_rate"] is None
for p in pres:
    d = json.loads((ROOT / "config" / "strategies" / f"{p.skill_id}.json").read_text(encoding="utf-8"))
    e = p.to_dict()
    d.pop("saved"), e.pop("saved")
    assert d == e, p.skill_id
assert ST.skill_display_name(cs) == "叠杯子·捏杯沿技能 (Cup Stack, rim pinch) [CS]"
assert ST.skill_display_name(pres[0]).endswith(" [EB]")
assert json.loads((ROOT / "config/strategies/default.json").read_text()) == \
    json.loads((ROOT / "config/strategies/egg_grasp_v345_baseline.json").read_text())
print("[M8] 9 cup knobs (离线试验✓), skills EB/DP/HD/QH/CS, shipped JSON == definitions, 'name [XX]' display")

# ------------------------------------------------------------------ M9
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


mtk, mttk = mock.MagicMock(), mock.MagicMock()
for nm in ("StringVar", "BooleanVar", "IntVar", "DoubleVar"):
    setattr(mtk, nm, FakeVar)
mtk.ttk = mttk
mtk.Text.return_value.yview.return_value = (0.0, 1.0)
mtk.Text.return_value.index.return_value = "1.0"
sys.modules["tkinter"] = mtk
sys.modules["tkinter.ttk"] = mttk
for sub in ("font", "messagebox", "simpledialog", "filedialog"):
    sys.modules[f"tkinter.{sub}"] = mock.MagicMock()


class W:                                   # minimal widget double that records pack state + text
    def __init__(self, *a, **k):
        self.packed = False
        self.text = k.get("text")

    def pack(self, *a, **k):
        self.packed = True

    def pack_forget(self):
        self.packed = False

    def configure(self, **k):
        if "text" in k:
            self.text = k["text"]

    def __getattr__(self, n):
        return mock.MagicMock()


wttk = mock.MagicMock()
wttk.Frame = wttk.LabelFrame = wttk.Button = W
ui = ST.SkillStrategyTabUI(mtk, wttk, mock.MagicMock(), log=lambda *a: None)
keys = list(ui._sections)
assert keys[:2] == ["skill_card", "dry_run"] and keys[2:] == [g_ for g_, _ in ST.KNOB_GROUPS], keys
assert all(v["open"] and v["body"].packed for v in ui._sections.values())
assert ui.toggle_section("ik_posture") is False and not ui._sections["ik_posture"]["body"].packed
assert ui._sections["ik_posture"]["btn"].text.startswith("▸ ") and ui._sections["gripper_approach"]["body"].packed
ui.collapse_all()
assert not any(v["open"] or v["body"].packed for v in ui._sections.values())
assert all(v["btn"].text.startswith("▸ ") for v in ui._sections.values())
ui.expand_all()
assert all(v["open"] and v["body"].packed and v["btn"].text.startswith("▾ ") for v in ui._sections.values())
assert ui.toggle_section("cup_stacking", True) is True                       # already open: no change
assert ui._sections["dry_run"]["box"].packed and ui._sections["skill_card"]["box"].packed
labels = list(ui._preset_map)
for p in pres:
    assert any(lab.startswith(ST.skill_display_name(p)) for lab in labels), (p.code, labels)
assert "[EB]" in ui.v_status.get()
ui_sim = ST.SkillStrategyTabUI(mtk, wttk, mock.MagicMock(), log=lambda *a: None, mode="sim")
assert not ui_sim._sections["dry_run"]["box"].packed and ui_sim._sections["cup_stacking"]["box"].packed
assert "dry_run" not in ui_sim.section_states() and len(ui_sim.section_states()) == len(ST.KNOB_GROUPS) + 1
lab_cs = next(lab for lab, (sid, _f, _p) in ui._preset_map.items() if sid == "cup_stack_rim_pinch")
ui.v_preset.set(lab_cs)
ui.on_select_preset()
assert ui.profile.code == "CS" and "[CS]" in ui.v_status.get() and "技能按场景区分" in ui.v_status.get()
root = mock.MagicMock()
root.winfo_screenwidth.return_value = 1920
root.winfo_screenheight.return_value = 1080
mtk.Tk.return_value = root
root.mainloop.side_effect = lambda: None
CON.run_console("sim")
CON.run_console("real")
SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k for k in SNAP0 if SNAP0[k] != SNAP1.get(k)]
print(f"[M9] {len(keys)} collapsible sections (fold one / all / unfold all), sim mode hides dry-run, codes in labels; "
      "console OK; release folder unchanged")
print("ALL V3.61 TESTS PASSED")
