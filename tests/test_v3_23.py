#!/usr/bin/env python3
"""V3.23 safety rule A (default): the polygon extruded vertically is a wall for the whole arm + gripper.

A1 defaults: rule A, base inside the default zone, margin 2 cm; a pre-V3.23 file without "rule" loads as A
A2 premise 1: a zone that misses the base refuses every pose; 并入底座 (merge_base) keeps the taught corners + names
A3 any height: the arm high above the table with the tool outside the zone -> refused by A, allowed by B (old rule)
A4 clearance: link spheres keep link_radius + margin, gripper gripper_radius + margin; teach / inspect relax only
   wrist + gripper + tool tip (pointer on a corner mark), the upper arm keeps its clearance
A5 path check: a sweep that crosses a wall is refused at the first crossing waypoint
A6 MoveIt walls: one thin box per edge, inner face on the edge, yaw along the edge, table .. z_max + 5 cm; only for
   rule A + walls_to_moveit; walls=False leaves them out (inspect); monitor sends the yaw as a quaternion
A7 editor wiring: rule selector, 并入底座, auto-merge after teaching, inspect retracts / restores the walls, save /
   open apply the scene, hand-guiding alarm
"""
import json, math, os, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V315_Consistency_Safety as CS  # noqa: E402
import NERO_V314_Joint_Control as JC  # noqa: E402
import NERO_V313_Real_Robot as RR  # noqa: E402

assert RR.VERSION.startswith(("V3.2", "V3.3", "V3.4"))
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
LIM = JC.build_model((ROOT / "config" / "nero.urdf").read_text(), None, None)["limits"]
w0 = CS.default_workspace()
f0 = w0["fence"]
assert f0["rule"] == "A" and f0["margin"] == 0.02 and CS.base_inside(f0) and len(w0["corners"]) == len(f0["polygon"])
tmp = Path(tempfile.mkdtemp(prefix="v323_"))
USER = [[-0.60, -0.10], [-0.60, 0.28], [-0.08, 0.28], [-0.17, -0.11]]      # like the user's zone (base outside)
old = {"version": "V3.17", "fence": {"polygon": USER, "margin": 0.04, "z_table": 0.027, "z_min_points": 0.029},
       "corners": [{"name": f"M{i + 1}", "source": "示教", "q": None} for i in range(4)], "boxes": w0["boxes"]}
(tmp / "old.json").write_text(json.dumps(old))
wu = CS.load_workspace(str(tmp / "old.json"))
assert wu["fence"]["rule"] == "A" and not CS.base_inside(wu["fence"]) and wu["fence"]["margin"] == 0.04
print("[A1] default rule A (base inside, margin 2 cm); old files without a rule load as A (their margin kept)")

T = CS.tool_transform(wu["fence"]["tool"], U)
Q_IN = [0, 1.6, 0, 1.6, 0, -0.1, 0]                    # gripper pointing down, in front of the base
ok, v, _p = CS.fence_check(U, J, Q_IN, "base_link", wu["fence"], T)
assert not ok and any(x["point"] == "link2" and x["kind"] == "wall" for x in v), v
wm = CS.merge_base(wu)
fm = wm["fence"]
assert CS.base_inside(fm) and len(fm["polygon"]) == 6 and CS.polygon_area(fm["polygon"]) > CS.polygon_area(USER)
names = [c["name"] for c in wm["corners"]]
assert {"M1", "M2", "M3"} <= set(names) and "B1" in names and all(c["source"] == "示教" for c in wm["corners"] if c["name"][0] == "M")
assert all(CS.point_in_polygon(x * 0.999 + (-0.3) * 0.001, y * 0.999 + 0.1 * 0.001, fm["polygon"]) or
           CS.edge_distance(x, y, fm["polygon"])[0] < 1e-6 for x, y in USER if [x, y] != [-0.17, -0.11])
print(f"[A2] zone without the base refuses every pose (link2 越墙); 并入底座 -> {len(fm['polygon'])} corners {names}")

R = [r[:3] for r in CS.tool_pose(U, J, Q_IN, "base_link", T)[:3]]
q_mid, e = CS.ik_tool(U, J, Q_IN, "base_link", T, [-0.40, 0.09, 0.06], LIM, keep_R=R)
assert q_mid and CS.fence_check(U, J, q_mid, "base_link", fm, T)[0]
Q_HIGH = [-0.6, 0.3, 0, 1.2, 0, 0, 0]                  # arm high, tool at y = 0.29 (outside), z = 0.46
pt = CS.fence_points(U, J, Q_HIGH, "base_link", fm, T)["tool"]
assert pt[2] > 0.4 and not CS.point_in_polygon(pt[0], pt[1], fm["polygon"])
okA, vA, _ = CS.fence_check(U, J, Q_HIGH, "base_link", fm, T)
okB, _vB, _ = CS.fence_check(U, J, Q_HIGH, "base_link", dict(fm, rule="B"), T)
assert not okA and okB and any(x["point"] == "tool" and x["dist"] < 0 for x in vA), vA
print("[A3] tool 460 mm above the table but outside the zone: rule A refuses (越墙), rule B allowed it")

pts = CS.fence_points(U, J, q_mid, "base_link", fm, T)
assert sum(1 for n in pts if "#" in n) >= 10 and "tool" in pts and any(n.startswith(CS.GRIP_TAG) for n in pts)
assert all(math.dist(a, b) <= fm["sample_step"] + 0.03 for a, b in zip(list(pts.values()), list(pts.values())[1:]))
f2 = dict(fm, polygon=[[-0.6, -0.3], [-0.6, 0.3], [0.12, 0.3], [0.12, -0.3]])
c_link = fm["link_radius"] + fm["margin"]
c_grip = fm["gripper_radius"] + fm["margin"]
assert CS.classify_point("link3", (-0.3, 0.3 - c_link + 0.005, 0.5), f2)[0][0]["kind"] == "wall"
assert not CS.classify_point("link3", (-0.3, 0.3 - c_link - 0.005, 0.5), f2)[0]
assert CS.classify_point("tool", (-0.3, 0.3 - c_grip + 0.005, 0.05), f2)[0]
relax = dict(f2, relax_tool=True)
assert not CS.classify_point("tool", (-0.6, 0.3, 0.05), relax)[0]                    # tip on the corner mark
assert not CS.classify_point("link7", (-0.6, 0.3, 0.2), relax)[0]                    # wrist above it
assert CS.classify_point("tool", (-0.61, 0.3, 0.05), relax)[0]                       # beyond the wall line: no
assert CS.classify_point("link2~link3#4", (-0.3, 0.3 - 0.02, 0.3), relax)[0]          # upper arm keeps clearance
print(f"[A4] spheres: links {c_link * 1000:.0f} mm, gripper {c_grip * 1000:.0f} mm from every wall; teach / inspect: "
      "wrist + gripper + tip may touch the wall line, the upper arm may not")

traj = [[a + (b - a) * k / 20 for a, b in zip(q_mid, Q_HIGH)] for k in range(21)]
viol = CS.path_fence_check(U, J, J, traj, "base_link", fm, T)
assert viol and 0 < viol["waypoint"] < 20, viol
assert CS.path_fence_check(U, J, J, [q_mid, q_mid], "base_link", fm, T) is None
print(f"[A5] path to the outside pose refused at waypoint {viol['waypoint'] + 1}/21")

walls = CS.wall_boxes(wm)
assert len(walls) == len(fm["polygon"])
th = fm["wall_thickness"]
ccw = CS.polygon_area(fm["polygon"]) > 0
for i, wb in enumerate(walls):
    a, b = fm["polygon"][i], fm["polygon"][(i + 1) % len(fm["polygon"])]
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    yaw = wb["yaw"]
    assert abs(yaw - math.atan2(b[1] - a[1], b[0] - a[0])) < 1e-9
    nx, ny = math.sin(yaw), -math.cos(yaw)
    if not ccw:
        nx, ny = -nx, -ny
    inner = (wb["pos"][0] - nx * th / 2, wb["pos"][1] - ny * th / 2)
    assert math.dist(inner, (mx, my)) < 1e-9 and not CS.point_in_polygon(wb["pos"][0], wb["pos"][1], fm["polygon"])
    assert abs(wb["size"][0] - (math.dist(a, b) + 2 * th)) < 1e-9 and wb["size"][1] == th
    assert abs(wb["pos"][2] - wb["size"][2] / 2 - fm["z_table"]) < 1e-9 and abs(wb["pos"][2] + wb["size"][2] / 2 - (fm["z_max"] + 0.05)) < 1e-9
objs = CS.scene_objects(wm)
assert sum("wall_" in o["id"] for o in objs) == len(walls) and any(o["id"] == "v315_table" for o in objs)
assert not any("wall_" in o["id"] for o in CS.scene_objects(wm, walls=False))
wb_ = json.loads(json.dumps(wm))
wb_["fence"]["rule"] = "B"
assert not any("wall_" in o["id"] for o in CS.scene_objects(wb_))
wb_["fence"].update(rule="A", walls_to_moveit=False)
assert not any("wall_" in o["id"] for o in CS.scene_objects(wb_))
RRs = (ROOT / "NERO_V313_Real_Robot.py").read_text()
assert 'pose.orientation.z, pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)' in RRs
print(f"[A6] {len(walls)} MoveIt walls: inner face on each edge, yaw along it, table .. z_max + 5 cm; only for rule A")

E = (ROOT / "NERO_V316_Safety_Editor.py").read_text()
for m in ('value="A", variable=self.v_rule', 'def on_merge_base(', 'CS.merge_base(self.ws)', 'self.on_apply(walls=False, wait=True)',
          'f["relax_tool"] = True', '手动拖动越界', '规则 A：安全区没有包住 NERO 底座', 'self.on_apply()'):
    assert m in E, m
assert E.count("CS.merge_base(self.ws)") == 2           # button + auto-merge after teaching
print("[A7] editor: rule A/B selector, 并入底座 + auto-merge after teaching, inspect retracts/restores the walls, "
      "save/open sync to MoveIt, hand-guiding alarm")
print("ALL V3.23 TESTS PASS")
