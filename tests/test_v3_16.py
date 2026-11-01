#!/usr/bin/env python3
"""V3.16 safety-zone rules (no ROS, no Tk). Polygon basics / migration are in test_v3_15.py K4-K5.

S1 XY free height: low points must be in the zone, high points only below z_max
S2 teach / inspect relaxations: tool-only XY, tool z floor, 2 mm boundary tolerance
S3 IK: hover poses above table marks reachable near joint limits (active set + LM), position < 2 mm
S4 recovery path rule + editor module wiring
"""
import math, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V314_Joint_Control as JC  # noqa: E402
import NERO_V315_Consistency_Safety as CS  # noqa: E402
_v323_dw = CS.default_workspace


def _v323_legacy_default():          # V3.23: this test covers the V3.16-V3.22 layered rule (now rule B, 4 corners)
    w = _v323_dw()
    w["fence"].update(rule="B", margin=0.04, polygon=[[-0.15, 0.45], [-0.70, 0.45], [-0.70, -0.45], [-0.15, -0.45]])
    w["corners"] = w["corners"][:4]
    return w


CS.default_workspace = _v323_legacy_default

X = (ROOT / "config" / "nero.urdf").read_text()
U, J = JC.parse_urdf(X), [f"joint{i}" for i in range(1, 8)]
LIM = JC.build_model(X, None, None)["limits"]
f = CS.default_workspace()["fence"]
assert f["xy_free_height"] == 0.15 and f["z_max"] == 1.0
# S1
assert CS.classify_point("link5", (0.30, 0.60, 0.10), f)[0][0]["kind"] == "outside"
assert not CS.classify_point("link5", (0.30, 0.60, 0.20), f)[0]
ft = dict(f, z_table=0.05)
assert CS.classify_point("link5", (0.30, 0.60, 0.19), ft)[0]                  # free height is relative to the table
assert not CS.classify_point("link5", (0.30, 0.60, 0.21), ft)[0]
print("[S1] XY zone applies below z_table + 150 mm; above only z_max")
# S2
fi = dict(f, xy_points_only=["tool"], tool_z_min_override=0.01)
assert not CS.classify_point("link6", (0.30, 0.60, 0.10), fi)[0]
assert CS.classify_point("tool", (0.30, 0.60, 0.10), fi)[0][0]["kind"] == "outside"
assert not CS.classify_point("tool", (-0.40, 0.0, 0.012), fi)[0] and CS.classify_point("link6", (-0.40, 0.0, 0.012), fi)[0]
assert not CS.classify_point("tool", (-0.40, 0.4515, 0.05), f)[0]             # 1.5 mm outside the edge: tolerance
assert CS.classify_point("tool", (-0.40, 0.4530, 0.05), f)[0]
print("[S2] teach/inspect: XY for the tool only, tool z floor; 2 mm boundary tolerance")
# S3
tool = CS.tool_transform({"mode": "pointer", "length": 0.10, "axis": "+x"}, U)
seed = [0, -1.3, 0, -0.9, 0, 0.0, 0]
for mx, my in ((0.40, 0.20), (0.55, 0.20), (0.55, -0.20), (0.40, -0.20), (0.5, 0.0)):
    for z in (0.002, 0.052):
        q, err = CS.ik_tool(U, J, seed, "base_link", tool, [mx, my, z], LIM)
        assert q is not None and err < 2e-3, (mx, my, z, err)
        assert all(lo - 1e-9 <= v <= hi + 1e-9 for v, (lo, hi) in zip(q, LIM))
        T = CS.tool_pose(U, J, q, "base_link", tool)
        assert CS.dist_mm([T[i][3] for i in range(3)], [mx, my, z]) < 2.0
print("[S3] IK: 5 table points at 2 mm / 52 mm from a pose at the J4 limit, all within limits, < 2 mm")
# S4
q_in = [0, -0.5, 0, 0.6, 0, 0.27, 0]
q_out = [math.radians(90), -0.5 - math.radians(50), 0, 0.6 - math.radians(60), 0, 0.27, 0]
assert CS.fence_check(U, J, q_in, "base_link", f, tool)[0] and not CS.fence_check(U, J, q_out, "base_link", f, tool)[0]
path = [[a + k / 10 * (b - a) for a, b in zip(q_out, q_in)] for k in range(11)]
assert CS.path_fence_check(U, J, J, path, "base_link", f, tool) is None                     # coming back in
assert CS.path_fence_check(U, J, J, list(reversed(path)), "base_link", f, tool)              # going out
E = (ROOT / "NERO_V316_Safety_Editor.py").read_text()
for m in ("V3.16 SAFETY EDITOR", "class SafetyEditorUI", "def on_teach_generate", "def _inspect", "def on_top_double",
          "xy_points_only", "tool_z_min_override", "pipe.safety_fn = self.safety"):
    assert m in E, m
C = (ROOT / "NERO_V312_Console.py").read_text()
assert 'VERSION = "V3.' in C and "SE.SafetyEditorUI(" in C
print("[S4] recovery: back into the zone allowed, out of it blocked; editor wired into the console")
print("ALL V3.16 TESTS PASS")
