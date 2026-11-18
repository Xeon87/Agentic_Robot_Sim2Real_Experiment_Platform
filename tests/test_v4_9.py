#!/usr/bin/env python3
"""NERO V4.9 offline tests (user run 2026-11-15, general_pick_place: "Pick up cup_1 and stack it into cup_2" failed:
two empty clamps passed as "held" (gap >= 4 mm rule) and were reported as OBJECT_SLIPPED_ON_LIFT, the third grasp put
the fingers into the cup wall -> MuJoCo diverged (backend.log qvel 2349 rad/s, QACC NaN)).

G1 gripper geometry: pad faces at open/2 - 4 mm, 29 x 4 x 60 mm pads centred on the grasp point, palm 55 mm above;
   with mujoco: the same numbers measured in the twin's MJCF.
G2 height map + descriptor of a hollow object (synthetic ring): open cavity, opening half widths, cross sections;
   relations: inside (and not "inside" for the outer of two nested objects), on.
G3 preview_grasp: a pinch 10 mm off the wall: only one pad meets the wall (it would push the object); a centred
   pinch of a ring closes on nothing; a pad inside the wall is a collision; a fingertip below the table is reported.
G4 hold in simulation = hold_active or both pad contacts (never the opening): an empty clamp with a 30 mm reading is
   OBJECT_NOT_HELD at the clamp (V4.8.11 lifted it and reported OBJECT_SLIPPED_ON_LIFT); the clamp facts are returned.
G5 no strategy hints: no grasp_mode / place_mode enums, no example values in the tool descriptions or the prompt,
   no min_held_gap_mm; the tools list has preview_grasp.
G6 place needs a held object (V4.8.11 executed a place after LIFT_FAILED); plan checks: the held object inside the
   target is PLACE_COLLISION with the clearance numbers.
G7 the general carton walls are short enough for the backend's frozen basket-rim rule.
G8 versions / markers.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTS = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("NERO_CALIB_DIR", tempfile.mkdtemp(prefix="nero_v49_"))

import NERO_V49_Geometry as G      # noqa: E402

try:
    import mujoco  # noqa: F401
    HAS_MJ = True
except Exception:
    HAS_MJ = False


def ring_map(r_in=0.044, r_out=0.047, h=0.12, floor=0.003, c=(-0.35, 0.0), res=0.002, name="ring_1"):
    x0, y0 = c[0] - 0.12, c[1] - 0.12
    n = int(0.24 / res)
    xs = x0 + (np.arange(n) + 0.5) * res
    ys = y0 + (np.arange(n) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    R = np.hypot(X - c[0], Y - c[1])
    H = np.zeros_like(R)
    own = np.full(R.shape, -1)
    H[R <= r_out] = floor
    H[(R >= r_in) & (R <= r_out)] = h
    own[R <= r_out] = 0
    return G.HeightMap(x0, y0, res, H, own, [name], 0.0)


class TestV49(unittest.TestCase):
    def test_g1_finger_model(self):
        fm = G.SIM_FINGERS
        b = {n: (lo, hi) for n, lo, hi in fm.boxes(60.0)}
        self.assertAlmostEqual(b["pad_pos_y"][0][1], 0.026, places=6)
        self.assertAlmostEqual(b["pad_neg_y"][1][1], -0.026, places=6)
        self.assertAlmostEqual(b["pad_pos_y"][1][2] - b["pad_pos_y"][0][2], 0.060, places=6)
        spec = fm.llm_spec()
        self.assertEqual(spec["fingertip_below_grasp_point_mm"], 30.0)
        self.assertEqual(spec["pad_face_gap_mm"], "open_mm - 8")
        if HAS_MJ:
            import NERO_V489_General_Pick_Place as GP
            import NERO_V362_Cup_Skills as CK
            io = GP.TwinGeneralIO()
            io.gripper_open(60.0)
            Tg = io._tcp_T() @ CK.OFFICIAL_PROFILE.T_tcp_grasp
            gi = io.m.geom("finger1_rubber_pad").id
            c = Tg[:3, :3].T @ (io.d.geom_xpos[gi] - Tg[:3, 3])
            self.assertAlmostEqual(abs(c[1]), 0.028, delta=0.0015)       # centre of a 4 mm pad whose face is at 26
            self.assertAlmostEqual(c[2], 0.0, delta=0.002)

    def test_g2_descriptor_and_relations(self):
        hm = ring_map()
        d = G.describe(hm, "ring_1", 0.0)
        cav = d["open_cavity"]
        self.assertIsNotNone(cav)
        self.assertGreater(cav["depth_mm"], 100.0)
        for v in cav["opening_half_width_mm"].values():
            self.assertAlmostEqual(v, 44.0, delta=3.0)
        self.assertIn("along_a", d["top_surface_segments_mm"])
        inner = {"object_id": "ball_1", "center_xyz_m": [-0.35, 0.0, 0.03], "top_z_m": 0.05, "bottom_z_m": 0.01,
                 "size_mm": [40.0, 40.0], "axis_a_yaw_deg": 0.0}
        rel = G.relations([inner, d])
        self.assertIn({"object": "ball_1", "relation": "inside", "of": "ring_1"}, rel)
        outer = dict(d, object_id="outer", bottom_z_m=-0.001)
        nested = dict(d, object_id="nested", bottom_z_m=0.025)
        rel2 = G.relations([outer, nested])
        self.assertNotIn({"object": "outer", "relation": "inside", "of": "nested"}, rel2)
        plate = {"object_id": "plate_1", "center_xyz_m": [-0.35, 0.0, 0.005], "top_z_m": 0.012, "bottom_z_m": 0.0,
                 "size_mm": [120.0, 120.0], "axis_a_yaw_deg": 0.0}
        blk = {"object_id": "block_1", "center_xyz_m": [-0.34, 0.0, 0.03], "top_z_m": 0.05, "bottom_z_m": 0.013,
               "size_mm": [40.0, 40.0], "axis_a_yaw_deg": 0.0}
        self.assertIn({"object": "block_1", "relation": "on", "of": "plate_1"}, G.relations([blk, plate]))

    def test_g3_preview(self):
        hm = ring_map()
        fm = G.SIM_FINGERS
        top = 0.12
        # 10 mm inside the wall centre (the V4.8.11 35 mm offset case): only the outer pad meets the (vertical)
        # wall; it would push the object and the inner pad meets nothing
        pv = G.preview_grasp(hm, fm, (-0.35, 0.0355, top - 0.018), 90.0, 0.0, 40.0, "ring_1")
        self.assertEqual(pv["collisions_at_grasp_pose"], [])
        self.assertEqual(pv["closes_on"], "only one pad touches material; the other closes on nothing")
        self.assertIsNotNone(pv["pad_pos_y"])
        self.assertIsNone(pv["pad_neg_y"])
        pv2 = G.preview_grasp(hm, fm, (-0.35, 0.0455, top - 0.018), 90.0, 0.0, 40.0, "ring_1")
        self.assertLessEqual(pv2["first_contact_pushes_object_mm"], 3.0)      # 2 mm map cells
        pv3 = G.preview_grasp(hm, fm, (-0.35, 0.0, top - 0.018), 90.0, 0.0, 60.0, "ring_1")
        self.assertEqual(pv3["closes_on"], "no material between the pads")
        pv4 = G.preview_grasp(hm, fm, (-0.35, 0.0, top - 0.062), 90.0, 0.0, 68.0, "ring_1")    # pads in the wall
        self.assertTrue(pv4["collisions_at_grasp_pose"])
        pv5 = G.preview_grasp(hm, fm, (-0.35, 0.2, 0.02), 90.0, 0.0, 40.0, None, table_clear_m=0.005)
        self.assertTrue(pv5["touches_table"])

    def _mock_skills(self):
        import NERO_V489_General_Pick_Place as GP
        import test_v4_8_10 as T10
        io = T10._MockSimGeneralIO()
        return GP, io, GP.GeneralPickPlaceSkills(io)

    def test_g4_contact_hold(self):
        GP, io, sk = self._mock_skills()
        sk.perceive_objects(prompts=["egg", "cup"])
        r = sk.plan_grasp_chunk(object_id="egg_1", xy_offset_mm=[0, 0], grasp_z_offset_from_top_mm=-5.0, closing_yaw_deg=90.0,
                                pitch_deg=0.0, open_mm=60.0, yaw_tolerance_deg=40.0, pregrasp_height_mm=70.0, lift_height_mm=120.0)
        self.assertTrue(r["ok"], r)
        io.gripper_clamp = lambda f, settle_s=2.2: {"opening_mm": 30.0, "gap_mm": 30.0, "hold_active": False,
                                                    "contact_left": False, "contact_right": False}
        lifts = []
        orig = io.move_line
        io.move_line = lambda label, *a, **k: (lifts.append(label), orig(label, *a, **k))[1]
        r = sk.execute_grasp_chunk(clamp_force_n=8.0)
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "OBJECT_NOT_HELD")
        self.assertFalse(any("lift" in x for x in lifts))
        self.assertEqual(r["physical_verification"]["clamp"]["contact_left"], False)
        self.assertIsNone(sk.held)
        self.assertTrue(sk._held_now({"contact_left": True, "contact_right": True, "hold_active": False}))
        self.assertFalse(sk._held_now({"contact_left": True, "contact_right": False, "gap_mm": 40.0}))

    def test_g5_no_hints(self):
        import NERO_V489_General_Pick_Place as GP
        txt = json.dumps(GP.GENERAL_TOOLS) + GP.GENERAL_SYSTEM_INSTRUCTIONS
        for bad in ("grasp_mode", "place_mode", "equator_pinch", "rim_pinch", "end_offset", "radius_mm", "e.g.",
                    "-20 to -25", "min_held_gap", "drop_into"):
            self.assertNotIn(bad, txt)
        self.assertFalse(any(ch.isdigit() for ch in GP.GENERAL_SYSTEM_INSTRUCTIONS.replace("7-DOF", "")))
        names = [t["name"] for t in GP.GENERAL_TOOLS]
        self.assertIn("preview_grasp", names)
        for t in GP.GENERAL_TOOLS:                      # strict schemas: every property required
            self.assertEqual(sorted(t["parameters"]["required"]), sorted(t["parameters"]["properties"]))

    def test_g6_place_needs_hold(self):
        GP, io, sk = self._mock_skills()
        sk.perceive_objects(prompts=["egg", "cup"])
        r = sk.execute_place_chunk(retreat_up_mm=50.0)
        self.assertEqual(r["reason"], "PLACE_NOT_PLANNED")
        sk.place_plan = {"target": "cup_1", "_wps": {}, "_q": {}, "release_open_mm": 60.0}
        r = sk.execute_place_chunk(retreat_up_mm=50.0)
        self.assertEqual(r["reason"], "NOT_HOLDING")
        self.assertTrue(sk.plan_grasp_chunk(object_id="egg_1", xy_offset_mm=[0, 0], grasp_z_offset_from_top_mm=-5.0,
                                            closing_yaw_deg=90.0, pitch_deg=0.0, open_mm=60.0, yaw_tolerance_deg=40.0,
                                            pregrasp_height_mm=70.0, lift_height_mm=120.0)["ok"])
        self.assertTrue(sk.execute_grasp_chunk(clamp_force_n=4.2)["ok"])
        r = sk.plan_place_chunk(target_id_or_xyz="cup_1", xy_offset_mm=[0, 0], held_bottom_above_target_top_mm=-60.0,
                                yaw_change_deg=0.0, approach_height_mm=40.0, release_open_mm=60.0)
        self.assertFalse(r["ok"])                       # the box-only mock cup is solid: the egg would be inside it
        self.assertEqual(r["reason"], "PLACE_COLLISION")
        self.assertLess(r["physical_verification"]["held_object_clearance"]["min_clearance_above_other_objects_mm"], 0)

    def test_g7_carton_walls(self):
        import re
        import NERO_V343_Scenes as SC
        geoms = [(m.group(1), [float(v) for v in m.group(2).split()], [float(v) for v in m.group(3).split()], m.group(4))
                 for m in re.finditer(r'<geom name="(general_carton[^"]*)" type="box" pos="([^"]+)" size="([^"]+)" '
                                      r'rgba="([^"]+)"', SC.general_carton_geoms_xml())]
        self.assertGreaterEqual(len(geoms), 9)
        self.assertEqual(SC.legacy_rim_hits(geoms), [])

    def test_g8_versions(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        import NERO_V489_General_Pick_Place as GP
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual((v, RR.VERSION, GP.VERSION), (C.VERSION, C.VERSION, C.VERSION))
        self.assertIn(v, ("V4.9", "V4.9.1", "V4.9.2", "V4.9.3", "V4.10", "V4.10.1", "V4.10.2", "V4.10.3", "V4.11", "V4.11.1"))
        self.assertEqual(G.MARKER, "V4.9 GENERAL GEOMETRY")


if __name__ == "__main__":
    unittest.main(verbosity=1)
