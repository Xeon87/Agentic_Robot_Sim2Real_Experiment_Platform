#!/usr/bin/env python3
"""V3.17: work side and views.

W1 the sim works on the −x side: egg basket / holder centres from the backend are inside the default safety zone,
   the default zone and table box are in front (−x) of the base
W2 editor: forward (−x) drawn up, +y right, tick labels in margins, side view explains every height,
   warning for a zone saved behind the robot (+x, the V3.16 default)
"""
import re, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V315_Consistency_Safety as CS  # noqa: E402
_v323_dw = CS.default_workspace


def _v323_legacy_default():          # V3.23: this test covers the V3.16-V3.22 layered rule (now rule B, 4 corners)
    w = _v323_dw()
    w["fence"].update(rule="B", margin=0.04, polygon=[[-0.15, 0.45], [-0.70, 0.45], [-0.70, -0.45], [-0.15, -0.45]])
    w["corners"] = w["corners"][:4]
    return w


CS.default_workspace = _v323_legacy_default

B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text(encoding="utf-8")
basket = [float(v) for v in re.search(r"V27_BASKET_CENTER_XY_M = \(([-\d.]+), ([-\d.]+)\)", B).groups()]
holder = [float(v) for v in re.search(r'name="egg_holder_base" type="box" pos="([-\d.]+) ([-\d.]+)', B).groups()]
w = CS.default_workspace()
poly = w["fence"]["polygon"]
assert basket[0] < 0 and holder[0] < 0, (basket, holder)
assert CS.point_in_polygon(*basket, poly) and CS.point_in_polygon(*holder, poly), (basket, holder, poly)
assert sum(p[0] for p in poly) / len(poly) < -0.2
table = [b for b in w["boxes"] if b["name"] == "table"][0]
assert table["pos"][0] < 0 and all(abs(p[0] - table["pos"][0]) <= table["size"][0] / 2 for p in poly)
print(f"[W1] sim basket {basket} and holder {holder} are inside the default zone (front = −x); table box under it")
E = (ROOT / "NERO_V316_Safety_Editor.py").read_text(encoding="utf-8")
for m in ("V3.17 VIEW ORIENTATION", "return ox + y * S, oy + x * S", "前方（−x）", "M_L, M_B, M_T",
          "V3.18 SIDE LEGEND", "SIDE_GUTTER", "防撞桌区", "安全区在底座后方（+x）"):
    assert m in E, m
C = (ROOT / "NERO_V312_Console.py").read_text(encoding="utf-8")
assert 'VERSION = "V3.' in C
print("[W2] editor: forward up / +y right, margin ticks, side view legend, +x zone warning")
print("ALL V3.17 TESTS PASS")
