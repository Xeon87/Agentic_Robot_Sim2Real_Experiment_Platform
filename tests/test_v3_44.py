#!/usr/bin/env python3
"""V3.44: the MuJoCo scene is chosen in the NERO console before the simulation starts (NERO_SCENE); no switching /
restart logic and no scene GUI in the simulation window.  Scenes: fill_egg_holder (default, "A1") and the
RoboDojo-style store_in_safe ("A2"); the per-step physics loop ("B") is unchanged.

C1 B = run_sim from 'with mujoco.viewer.launch_passive(...)' to its end is byte-identical to V3.42
C2 the whole backend with the '# V3.43 A2 BEGIN/END' blocks removed is byte-identical to V3.42 (so A1 is untouched)
C3 default scene: the three run_sim lists equal V3.42, every A2 override is guarded; default-scene XML == V3.42
C4 store_in_safe XML (safe body, hinged door, handle body, free box, hold equalities, contact exclusions)
C5 scene geometry helpers (MoveIt boxes, yawed door boxes, inside / closed / success)
C6 selection = NERO_SCENE only; console drop-down -> NERO_SCENE; launcher validates it; no restart / selector left
C7 door range follows <compiler angle>
"""
import ast
import hashlib
import math
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("NERO_SCENE", None)
os.environ.pop("NERO_V343_ACTIVE_SCENE", None)
import NERO_V343_Scenes as S          # noqa: E402
import NERO_V326_Wrist_Camera as W    # noqa: E402

assert S.VERSION == "V3.44" and W.VERSION in ("V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1") and (ROOT / "VERSION").read_text().strip() in ("V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
assert hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest().startswith("020cc6680f181872")

B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
A2 = re.compile(r"\n[ \t]*# V3\.43 A2 BEGIN\n.*?\n[ \t]*# V3\.43 A2 END(?=\n)", re.S)
H = lambda s: hashlib.sha256(s.encode()).hexdigest()
tree = ast.parse(B)
FN = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
run_sim = ast.get_source_segment(B, FN["run_sim"])

# ------------------------------------------------------------------ C1
i = run_sim.index("        with mujoco.viewer.launch_passive(model, data) as viewer:")
assert H(run_sim[i:]) == "2b1f124e838087f65a4e93e65ed1ceba169ea3c725bbea99ff87998ad8cd7252", "B (physics loop) changed"
assert "V3.43" not in run_sim[i:]
print("[C1] B: run_sim per-step physics loop byte-identical to V3.42 (sha256 2b1f124e...)")

# ------------------------------------------------------------------ C2
assert H(A2.sub("", B)) == "5e83581fb69bd7d8afd8278b26a2ebe72224f67e73b1b83abe12a533b2700405", "outside A2 changed"
assert H(A2.sub("", run_sim)) == "4006e52c36dc4aff598de653d2b231bf06fabc64defdac33b8eddbfdfbf03036"
n_a2 = B.count("# V3.43 A2 BEGIN")
assert n_a2 == B.count("# V3.43 A2 END") == 11, n_a2
print(f"[C2] backend minus the {n_a2} A2 blocks == V3.42 byte for byte (A1 egg scene code untouched)")

# ------------------------------------------------------------------ C3
V342_LISTS = {
    "object_tf_body_map": {"obj_egg_1": "obj_egg1", "obj_egg_2": "obj_egg2", "obj_egg_3": "obj_egg3", "obj_egg_4": "obj_egg4"},
    "egg_scene_boxes": [("egg_holder_base", (-0.42, 0.155, 0.006), (0.25, 0.144, 0.012)),
                        ("egg_holder_rim_xneg", (-0.548, 0.155, 0.025), (0.008, 0.152, 0.05)),
                        ("egg_holder_rim_xpos", (-0.292, 0.155, 0.025), (0.008, 0.152, 0.05)),
                        ("egg_holder_rim_yneg", (-0.42, 0.079, 0.025), (0.248, 0.008, 0.05)),
                        ("egg_holder_rim_ypos", (-0.42, 0.231, 0.025), (0.248, 0.008, 0.05))],
    "hold_object_names": ["obj_egg1", "obj_egg2", "obj_egg3", "obj_egg4",
                          "obj_red_cube", "obj_blue_cube", "obj_red_ball", "obj_blue_ball"],
}
found, overrides = {}, {}
for node in ast.walk(FN["run_sim"]):
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in V342_LISTS:
        try:
            found.setdefault(node.targets[0].id, (node.lineno, ast.literal_eval(node.value)))
            if node.lineno < found[node.targets[0].id][0]:
                found[node.targets[0].id] = (node.lineno, ast.literal_eval(node.value))
        except ValueError:
            overrides[node.targets[0].id] = node.lineno
for k, v in V342_LISTS.items():
    assert found[k][1] == v, k
assert set(overrides) == set(V342_LISTS)
for node in ast.walk(FN["run_sim"]):
    if isinstance(node, ast.If) and "_v343_scene_name()" in ast.unparse(node.test):
        assert ast.unparse(node.test) == "_v343_scene_name() != 'fill_egg_holder'", ast.unparse(node.test)
guarded = {n.targets[0].id for g in ast.walk(FN["run_sim"]) if isinstance(g, ast.If)
           and ast.unparse(g.test) == "_v343_scene_name() != 'fill_egg_holder'"
           for n in g.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
assert set(V342_LISTS) <= guarded
assert S.selected_scene() == S.active_scene() == "fill_egg_holder" == S.DEFAULT_SCENE

g = {"np": np, "re": re, "math": math, "Path": Path, "os": os, "_V343": S}
for name in ("make_grid_world_xml", "_v333_visual_quality", "_v343_scene_name"):
    exec(compile(ast.Module([FN[name]], []), "backend", "exec"), g)
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "nero_motor.xml"
    p.write_text("<mujoco model='t'><asset></asset><worldbody></worldbody></mujoco>")
    out = Path(g["make_grid_world_xml"](str(p))).read_text().replace(str(Path(td).resolve()), "<TMP>")
    assert H(out) == "31c52c06965826a06612c970d1b522a05396d9da9b5c9d0919c8824c832a3ef2", "default-scene XML changed"
print("[C3] default scene: TF map / MoveIt holder boxes / hold list == V3.42, overrides only under "
      "`!= fill_egg_holder`; generated egg-scene XML == V3.42 (sha256 31c52c06...)")

# ------------------------------------------------------------------ C4
os.environ["NERO_SCENE"] = "store_in_safe"
assert S.active_scene() == "store_in_safe" and g["_v343_scene_name"]() == "store_in_safe"
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "nero_motor.xml"
    p.write_text("<mujoco model='t'><asset></asset><worldbody><body name='link7'/></worldbody></mujoco>")
    path = g["make_grid_world_xml"](str(p))
    assert Path(path).name == "_nero_motor_v343_store_in_safe_tmp.xml"
    r = ET.parse(path).getroot()
    bodies = {b.get("name"): b for b in r.iter("body")}
    geoms = [x.get("name") for x in r.iter("geom") if x.get("name")]
    assert len(geoms) == len(set(geoms)) and len(bodies) == len(list(r.iter("body")))
    assert not [n for n in geoms if n.startswith("egg_")] and not [n for n in bodies if n.startswith("obj_egg")]
    walls = [n for n in geoms if n.startswith("safe_body_")]
    assert len(walls) == 6 and "play_table" in geoms and "v5_grid_floor" in geoms
    door = bodies["safe_door"]
    j = door.find("joint")
    assert j.get("name") == "safe_door_hinge" and j.get("type") == "hinge" and j.get("axis") == "0 0 1"
    lo, hi = (float(v) for v in j.get("range").split())
    assert lo == 0.0 and abs(hi - 110.0) < 0.01       # V3.43.1: no <compiler angle> -> MJCF default degree
    assert float(j.get("frictionloss")) > 0 and float(j.get("damping")) > 0
    hb = door.find("body")
    assert hb.get("name") == "obj_safe_handle" and all(x.get("name").startswith("obj_safe_handle_") for x in hb.iter("geom"))
    box = bodies["obj_valuables_box"]
    assert box.find("freejoint") is not None
    z0 = float(box.get("pos").split()[2])
    assert abs(z0 - (S.BOX["half"][2] + 0.0005)) < 1e-6
    eq = {c.get("name"): c for c in r.iter("connect")}
    assert set(eq) == {f"hold_{o}" for o in S.hold_object_names()} and all(c.get("active") == "false" for c in eq.values())
    for o in S.hold_object_names():
        assert o in bodies, o
    ex = [(e.get("body1"), e.get("body2")) for e in r.iter("exclude")]
    assert ("world", "safe_door") in ex and ("world", "obj_safe_handle") in ex
    tex = {t.get("name"): t.get("file") for t in r.iter("texture") if t.get("file")}
    assert all(Path(f).is_file() for f in tex.values()) and len(tex) == 2
    assert r.find("visual/quality").get("shadowsize") == "8192"
    for sem, body in S.tf_body_map().items():
        assert body in bodies and sem.startswith("obj_")
del os.environ["NERO_SCENE"]
print(f"[C4] store_in_safe XML: {len(walls)} protected safe_body_* geoms, door hinge 0-110 deg (damping + friction), "
      f"handle body obj_safe_handle, free valuables box, {len(eq)} hold equalities, door/world contacts excluded")

# ------------------------------------------------------------------ C5
pb = S.planning_boxes()
assert [n for n, _, _ in pb] == ["safe_body_bottom", "safe_body_top", "safe_body_back", "safe_body_side_yneg", "safe_body_side_ypos"]
back = dict((n, (c, s)) for n, c, s in pb)["safe_body_back"]
assert abs(back[0][0] - (-0.5925)) < 1e-9 and abs(back[1][0] - 0.015) < 1e-9 and abs(back[1][2] - 0.22) < 1e-9
d0 = dict((n, (c, s, y)) for n, c, s, y in S.door_boxes(0.0))
assert abs(d0["safe_door"][0][0] - (-0.391)) < 1e-9 and abs(d0["safe_door"][0][1] - 0.17) < 1e-9
assert abs(d0["safe_door_handle"][0][1] - 0.08) < 1e-9 and d0["safe_door_handle"][0][0] > -0.40
d90 = dict((n, (c, s, y)) for n, c, s, y in S.door_boxes(math.pi / 2))
assert abs(d90["safe_door"][0][0] - (-0.40 + 0.13)) < 1e-9 and abs(d90["safe_door"][0][1] - (0.30 + 0.009)) < 1e-9
assert abs(S.door_angle_from_quat_xyzw((0, 0, math.sin(math.radians(45)), math.cos(math.radians(45)))) - 90.0) < 1e-9
assert S.inside_safe((-0.50, 0.17, 0.04)) and not S.inside_safe((-0.36, -0.17, 0.0225))
st = S.task_state(1.0, (-0.50, 0.17, 0.04))
assert st["success"] and st["object_inside"] and st["door_closed"]
st = S.task_state(95.0, (-0.50, 0.17, 0.04))
assert st["door_open"] and not st["success"]
print("[C5] MoveIt: 5 wall boxes + yawed door/handle boxes follow the hinge angle; inside / closed / success state")

# ------------------------------------------------------------------ C6
assert S.selected_scene() == S.active_scene() == "fill_egg_holder"
os.environ["NERO_SCENE"] = "store_in_safe"
assert S.selected_scene() == "store_in_safe"
os.environ["NERO_SCENE"] = "bogus"
assert S.selected_scene() == "fill_egg_holder"
del os.environ["NERO_SCENE"]
os.environ["NERO_V343_ACTIVE_SCENE"] = "store_in_safe"
assert S.active_scene() == "store_in_safe"
del os.environ["NERO_V343_ACTIVE_SCENE"]
for gone in ("save_selection", "selection_file", "request_restart", "RESTART_EXIT_CODE", "forced_scene"):
    assert not hasattr(S, gone), gone
for gone in ("_V343_RESTART", "os._exit(", "WM_DELETE_WINDOW\"))", "v343_combo", "SCENE SELECTOR", "v343_restart"):
    assert gone not in B, gone
for mk in ('os.environ["NERO_V343_ACTIVE_SCENE"] = _V343.selected_scene()', "[V3.43 SCENE] RUN TASK blocked",
           '[V3.44 SCENE] MuJoCo scene = '):
    assert mk in B, mk
L = (ROOT / "start_nero_sim.sh").read_text()
for gone in ("BACKEND_RC", "exec bash", "REQ343", "v343_scene.txt", "v343_before_switch"):
    assert gone not in L, gone
assert L.rstrip().endswith('wait "$BACKEND_PID"')
for mk in ('SCENES343="$HERE/NERO_V343_Scenes.py"', 'SCENE344="${NERO_SCENE:-fill_egg_holder}"',
           'fill_egg_holder|store_in_safe) ;;', 'fatal "Unknown NERO_SCENE=', 'export NERO_SCENE="$SCENE344"'):
    assert mk in L, mk
import subprocess  # noqa: E402
case = L[L.index('SCENE344="${NERO_SCENE:-fill_egg_holder}"'):L.index('export NERO_SCENE="$SCENE344"')]
for val, ok in (("", True), ("store_in_safe", True), ("nope", False)):
    r = subprocess.run(["bash", "-c", 'fatal(){ echo FATAL; exit 3; }\n' + case + 'echo "OK $SCENE344"'],
                       env=dict(os.environ, NERO_SCENE=val), capture_output=True, text=True)
    assert (r.returncode == 0) == ok and (("OK " in r.stdout) == ok), (val, r.stdout)
sys.modules.pop("tkinter", None)
import NERO_V312_Console as C  # noqa: E402
assert C.VERSION in ("V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1") and C.DEFAULT_SIM_SCENE == "fill_egg_holder" and [k for k, _ in C.SIM_SCENES] == list(S.SCENES)
assert C.scene_env("store_in_safe") == {"NERO_SCENE": "store_in_safe"}
try:
    C.scene_env("nope")
    raise AssertionError("unknown scene accepted")
except ValueError:
    pass
CS = (ROOT / "NERO_V312_Console.py").read_text()
for mk in ('sim_scene_var = tk.StringVar(value=sim_scene_labels[DEFAULT_SIM_SCENE])', "pid = runner.start(extra_env=scene_env(scene))",
           'sim_scene_box.configure(state="disabled" if runner.running() else "readonly")'):
    assert mk in CS, mk
with tempfile.TemporaryDirectory() as td:                       # SimRunner really passes NERO_SCENE to the launcher
    fake = Path(td) / "fake_sim.sh"
    fake.write_text('#!/usr/bin/env bash\necho "SCENE=${NERO_SCENE:-unset}"\n')
    r = C.SimRunner(str(fake))
    r.start(extra_env=C.scene_env("store_in_safe"))
    got = []
    while True:
        line = r.lines.get(timeout=10)
        if line is None:
            break
        got.append(line)
    assert got == ["SCENE=store_in_safe"], got
print("[C6] scene = NERO_SCENE only (default fill_egg_holder); console drop-down -> runner.start(NERO_SCENE) -> launcher "
      "validates + exports it; drop-down locked while running; no restart / exit-75 / selector code left")
# ------------------------------------------------------------------ C7
assert S.compiler_angle_unit("<mujoco><compiler angle='radian' meshdir='m'/></mujoco>") == "radian"
assert S.compiler_angle_unit('<mujoco><compiler meshdir="m" angle="degree"/></mujoco>') == "degree"
assert S.compiler_angle_unit("<mujoco><worldbody/></mujoco>") == "degree"
assert 'range="0 1.91986"' in S.scene_world_xml("radian") and 'range="0 110.000"' in S.scene_world_xml("degree")
print("[C7] door range follows <compiler angle> (radian, or MJCF default degree)")
print("ALL V3.44 TESTS PASSED")
