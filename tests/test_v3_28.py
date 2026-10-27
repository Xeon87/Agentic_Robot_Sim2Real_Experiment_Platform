#!/usr/bin/env python3
"""V3.28 wrist camera: view-aware tracking + low-pose escape.

T1 hint protocol in the trigger tag (held / placed), length limit, decode
T2 in_view(): projection of a world point into the wrist image
T3 (numpy) replay of the V3.27 failure with the unchanged V3.11 PerceptionTracker:
   basket view -> egg_1 held -> egg_1 placed in slot_1 -> holder view -> basket view.
   Plain tracker: identities swap (egg_1 <-> egg_4) as in the log; ViewTracker + hints: no swap.
T4 backend: hints sent with every wrist trigger, q0 detour failure -> direct look plan, wrist recapture without
   q0, q0 floor relative to the start height; fixed-camera recapture path and V3.11 tracker unchanged
"""
import hashlib, math, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V326_Wrist_Camera as W  # noqa: E402

assert W.VERSION in ("V3.28", "V3.29", "V3.30", "V3.31", "V3.31.1", "V3.33", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
# ------------------------------------------------------------------ T1
s = W.encode_hints("observe_scene/holder", held="egg_2", placed={"egg_1": [-0.51, 0.123, 0.054]})
assert s == "observe_scene/holder|held=egg_2|placed=egg_1@-0.510,0.123,0.054", s
tag, h = W.decode_hints(s)
assert tag == "observe_scene/holder" and h["held"] == "egg_2" and h["placed"] == {"egg_1": [-0.51, 0.123, 0.054]}
assert W.decode_hints("plan_grasp:egg_1/above_egg_1") == ("plan_grasp:egg_1/above_egg_1", {"held": None, "placed": {}, "cam": "d435i", "heldp": None, "live": False, "lost": []})
many = {f"egg_{i}": [-0.5, 0.1, 0.05] for i in range(20)}
s2 = W.encode_hints("x" * 60, placed=many)
assert len(s2.encode()) <= 255 and len(W.decode_hints(s2)[1]["placed"]) >= 5
print("[T1] hint protocol: tag|held=..|placed=name@x,y,z;..  (<= 255 bytes)")

# ------------------------------------------------------------------ T2
K = {"width": 1280, "height": 720, "fx": 935.0, "fy": 935.0, "ppx": 639.5, "ppy": 359.5}


def T_down(x, y, z):
    """optical frame looking straight down from (x, y, z): x_opt = world x, y_opt = -world y, z_opt = -world z"""
    return [[1.0, 0.0, 0.0, x], [0.0, -1.0, 0.0, y], [0.0, 0.0, -1.0, z], [0.0, 0.0, 0.0, 1.0]]


T_basket, T_holder = T_down(-0.425, -0.155, 0.40), T_down(-0.420, 0.155, 0.40)
assert W.in_view([-0.425, -0.155, 0.03], K, T_basket) and W.in_view([-0.38, -0.09, 0.03], K, T_basket)
assert not W.in_view([-0.51, 0.123, 0.05], K, T_basket) and W.in_view([-0.51, 0.123, 0.05], K, T_holder)
assert not W.in_view([-0.425, -0.155, 0.60], K, T_basket)                       # behind the camera
print("[T2] in_view: basket eggs in the basket view, slot_1 only in the holder view, points behind -> out")

# ------------------------------------------------------------------ T3
try:
    import numpy as np  # noqa: F401
    import NERO_V308_Camera_Perception as P8
    have_np = True
except Exception:
    have_np = False
if not have_np:
    print("[T3] numpy missing -> tracker replay skipped")
else:
    E = {"egg_1": [-0.456, -0.224, 0.022], "egg_2": [-0.381, -0.141, 0.023],
         "egg_3": [-0.429, -0.174, 0.021], "egg_4": [-0.395, -0.087, 0.023]}
    SLOT1 = [-0.510, 0.123, 0.054]

    def det(i, xyz):
        return {"det_id": i, "xyz_m": list(xyz), "quat_xyzw": [0, 0, 0, 1], "axis_world": [1, 0, 0],
                "location": "egg_basket", "trusted": True, "confidence": 1.0, "quality": {"coverage": 0.9}}

    def run(view_aware):
        base = P8.PerceptionTracker()
        tr = W.ViewTracker(base) if view_aware else None

        def upd(dets, seq, T, match=None, hints=None):
            if tr:
                return tr.update(dets, seq, K, T, seed_match=match, hints=hints)[0]
            return base.update(dets, seq, seed_match=match)[0]
        d = [det(k, E[n]) for k, n in enumerate(sorted(E))]
        upd(d, 1, T_basket, match={n: k for k, n in enumerate(sorted(E))})        # first sighting (sim seeding)
        upd([det(k, E[n]) for k, n in enumerate(["egg_2", "egg_3", "egg_4"])], 2, T_basket,
            hints={"held": "egg_1"})                                               # egg_1 in the gripper
        out_h = upd([det(0, SLOT1)], 3, T_holder, hints={"placed": {"egg_1": SLOT1}} if view_aware else None)
        out_b = upd([det(k, E[n]) for k, n in enumerate(["egg_2", "egg_3", "egg_4"])], 4, T_basket,
                    hints={"placed": {"egg_1": SLOT1}} if view_aware else None)
        return out_h, out_b
    oh, ob = run(False)
    assert oh["egg_4"]["seen"] and math.dist(oh["egg_4"]["xyz_m"], SLOT1) < 1e-6        # V3.27 bug reproduced
    assert ob["egg_1"]["seen"] and math.dist(ob["egg_1"]["xyz_m"], E["egg_4"]) < 1e-6
    oh, ob = run(True)
    assert oh["egg_1"]["seen"] and math.dist(oh["egg_1"]["xyz_m"], SLOT1) < 1e-6
    assert not oh["egg_4"]["seen"] and oh["egg_4"]["out_of_view"] and math.dist(oh["egg_4"]["xyz_m"], E["egg_4"]) < 1e-6
    for n in ("egg_2", "egg_3", "egg_4"):
        assert ob[n]["seen"] and math.dist(ob[n]["xyz_m"], E[n]) < 1e-6, n
    assert not ob["egg_1"]["seen"] and math.dist(ob["egg_1"]["xyz_m"], SLOT1) < 1e-6
    print("[T3] V3.27 replay: plain tracker swaps egg_1<->egg_4 (bug reproduced); ViewTracker + hints keeps "
          "egg_1 in slot_1 and egg_4 in the basket")

# ------------------------------------------------------------------ T4
B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
for mk in ("_v335_capture(", "def _v328_tag_with_hints(tag",
           "[V3.28 WRIST TRACK]", "[V3.28 WRIST LOOK DIRECT]", "pass        # V3.28: the wrist recapture",
           "floor_328=min(0.180, z0_328-", "V3.28 Q0 FLOOR FROM START", 'V271_PLACEMENT_RUNTIME.get("released")'):
    assert mk in B, mk
assert '_v307_go_q0("PERCEPTION RECAPTURE")' in B                                     # fixed camera path kept
WM = (ROOT / "NERO_V326_Wrist_Camera.py").read_text()
assert "ViewTracker(P8.PerceptionTracker()" in WM and "tag, hints = decode_hints(raw_tag)" in WM
h = hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest()
assert h.startswith("020cc6680f181872"), h
print("[T4] backend: hints on every wrist trigger, direct look plan when q0 fails, wrist recapture without q0, "
      "q0 floor = min(180 mm, start - 20 mm); fixed-camera module + recapture unchanged")
print("ALL V3.28 TESTS PASS")
