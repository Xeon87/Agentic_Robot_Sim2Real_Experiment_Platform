#!/usr/bin/env python3
"""NERO V4.12 = H1 V2 offline tests (user 11-20: 「开发H1 V2：通用运动片段；任意接近方向的抓取和预演；稳定性相关事实；
抬起试探；不抓的推动；V1 工具改为在片段上实现；新场景：竖立酒瓶、推到指定位置 … 直接在通用pick and place场景里面增加。」).

K1 schema: plan_motion_chunk / execute_motion_chunk (strict), approach_distance_mm replaces pregrasp_height_mm (the
   old name is still accepted), execute_place_chunk has retreat_back_mm; legend has stability_estimate; versions.
K2 geometry: sweep_collisions finds an object on the approach line (not at the grasp pose); stability of a tall slim
   box vs a flat one; grasp_R approach axis (pitch 0 = down, pitch 90 = horizontal).
K3 scenes: H6 (hand-placed far half), H7 (bevelled mat mesh), upright_on parses and is judged with the tilt.
K4 (mujoco) H6: side grasp of the standing bottle's neck (pitch 90, horizontal approach), centring step after a
   one-pad contact, lift probe facts, upright onto the tray with retreat_back -> goal ok (simulator truth).
K5 (mujoco) H7: push the book onto the mat with a motion chunk: predicted contact, a path through an object that is
   not in may_touch is refused, execute -> goal ok; motion refused while holding.
"""
from __future__ import annotations

import json
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
os.environ.setdefault("NERO_CALIB_DIR", tempfile.mkdtemp(prefix="nero_v412_"))

try:
    import mujoco  # noqa: F401
    HAS_MJ = True
except Exception:
    HAS_MJ = False

PERCEIVE_H6 = {"prompts": ["bottle", "tray", "apple"], "use_finetuned": True}
GRASP_H6 = {"object_id": "tall_bottle_1", "xy_offset_mm": [0, 0], "grasp_z_offset_from_top_mm": -35,
            "closing_yaw_deg": 90, "pitch_deg": 90, "open_mm": 50, "yaw_tolerance_deg": 30,
            "approach_distance_mm": 80, "lift_height_mm": 100}
PLACE_H6 = {"target_id_or_xyz": "tray_1", "xy_offset_mm": [0, 0], "held_bottom_above_target_top_mm": -17,
            "yaw_change_deg": 0, "approach_height_mm": 60, "release_open_mm": 50, "carry_lift_mm": 0,
            "pitch_change_deg": 0}


def _push_path(cb, cm, yaw_b, z=0.04):
    """a straight push of a 150 x 113 mm book (centre cb, yaw yaw_b) so that its centre ends at cm"""
    import math
    u = np.array([cm[0] - cb[0], cm[1] - cb[1]])
    u /= np.linalg.norm(u)
    ang = math.degrees(math.atan2(u[1], u[0]))
    ha = 75 * abs(math.cos(math.radians(ang - yaw_b))) + 56.6 * abs(math.sin(math.radians(ang - yaw_b)))
    back = (ha + 4.0) / 1000.0
    st = np.array(cb[:2]) - u * (back + 0.03)
    en = np.array(cm[:2]) - u * back
    ret = en - u * 0.03
    path = [[st[0], st[1], 0.12], [st[0], st[1], z], [en[0], en[1], z], [ret[0], ret[1], z], [ret[0], ret[1], 0.12]]
    return [[round(float(c), 4) for c in p] for p in path], round(ang, 1)


class V412(unittest.TestCase):
    def test_k1_schema(self):
        import NERO_V489_General_Pick_Place as GP
        tools = {t["name"]: t for t in GP.GENERAL_TOOLS}
        for n in ("plan_motion_chunk", "execute_motion_chunk"):
            self.assertIn(n, tools)

        def strict(sch):
            if sch.get("type") == "object":
                self.assertFalse(sch.get("additionalProperties", True))
                self.assertEqual(set(sch.get("required", [])), set(sch.get("properties", {})))
                for v in sch.get("properties", {}).values():
                    strict(v)
            if sch.get("type") == "array":
                strict(sch["items"])
        for t in GP.GENERAL_TOOLS:
            strict(t["parameters"])
        g = tools["plan_grasp_chunk"]["parameters"]["properties"]
        self.assertIn("approach_distance_mm", g)
        self.assertNotIn("pregrasp_height_mm", g)
        self.assertIn("retreat_back_mm", tools["execute_place_chunk"]["parameters"]["required"])
        self.assertIn("may_touch", tools["plan_motion_chunk"]["parameters"]["required"])
        self.assertIn("stability_estimate", GP.DESC_LEGEND)
        self.assertIn("plan_motion_chunk", GP.GENERAL_SYSTEM_INSTRUCTIONS)
        self.assertIn("H1 V2", GP.MARKER_V412)
        import NERO_V49_Geometry as G49
        self.assertIn("along the fingers' direction", G49.SIM_FINGERS.llm_spec()["frame"])
        for t in GP.GENERAL_TOOLS:                                   # facts only: no advice words in descriptions
            self.assertNotRegex(t["description"].lower(), r"\b(should|recommend|try to|prefer)\b", t["name"])

    def test_k1_versions(self):
        import NERO_V312_Console as C
        import NERO_V411_Sim_DOE as D
        import NERO_V489_General_Pick_Place as GP
        for v in (C.VERSION, D.VERSION, GP.VERSION):
            self.assertEqual(v, "V4.12")
        with open(os.path.join(HERE, "VERSION")) as f:
            self.assertEqual(f.read().strip(), "V4.12")

    def test_k2_geometry(self):
        import NERO_V49_Geometry as G49
        R0, R90 = G49.grasp_R(90.0, 0.0), G49.grasp_R(90.0, 90.0)
        self.assertTrue(np.allclose(R0[:, 2], [0, 0, -1], atol=1e-6))
        self.assertAlmostEqual(float(R90[2, 2]), 0.0, places=6)              # horizontal approach
        objs = [{"object_id": "post_1", "center_xyz_m": [-0.40, 0.0, 0.10], "size_mm": [40, 40], "top_z_m": 0.20},
                {"object_id": "slab_1", "center_xyz_m": [-0.40, 0.20, 0.01], "size_mm": [150, 110], "top_z_m": 0.02},
                {"object_id": "wall_1", "center_xyz_m": [-0.33, 0.0, 0.10], "size_mm": [10, 120], "top_z_m": 0.20}]
        hm = G49.HeightMap.from_boxes(objs, 0.0)
        # the gripper reaches post_1 horizontally from +x: wall_1 is on that line 70 mm before the grasp pose
        p = np.array([-0.40, 0.0, 0.10])
        sw = G49.sweep_collisions(hm, G49.SIM_FINGERS, p, 90.0, 90.0, 60.0, 0.12, margin_m=0.0)
        self.assertAlmostEqual(sw["approach_direction_xyz"][0], -1.0, places=3)          # from +x (the robot side)
        self.assertIsNotNone(sw["first_collision"])
        self.assertEqual(sw["first_collision"]["object"], "wall_1")
        self.assertGreater(sw["first_collision"]["distance_before_grasp_pose_mm"], 20.0)
        sw2 = G49.sweep_collisions(hm, G49.SIM_FINGERS, p + [0, 0, 0.3], 90.0, 0.0, 60.0, 0.08, margin_m=0.0)
        self.assertIsNone(sw2["first_collision"])                                 # from above, high up
        st_post, st_slab = G49.stability(hm, "post_1", 0.0), G49.stability(hm, "slab_1", 0.0)
        self.assertLess(st_post["tip_over_angle_deg"], 15.0)
        self.assertGreater(st_slab["tip_over_angle_deg"], 70.0)
        self.assertGreater(st_post["height_to_narrowest_width"], 4.0)
        self.assertAlmostEqual(st_post["estimated_center_of_mass_height_mm"], 100.0, delta=5.0)

    def test_k3_scenes(self):
        import NERO_V410_Household_Scenes as HS
        h6, h7 = HS.fixed_scene("H6"), HS.fixed_scene("H7")
        self.assertEqual(h6.goals, [("tall_bottle_1", "upright_on", "tray_1")])
        self.assertEqual({i.oid: (round(i.x, 2), round(i.y, 2)) for i in h6.items}["tall_bottle_1"], (-0.53, 0.10))
        self.assertEqual(h7.goals, [("book_1", "onto", "mat_1")])
        xml = h7.world_xml(visual=False) if hasattr(h7, "world_xml") else ""
        self.assertIn('mesh="hh_mat_plate"', xml)
        self.assertIn("hh_mat_plate", HS.assets_xml(["mat"]))
        self.assertEqual(HS.parse_goals("tall_bottle_1 upright_on tray_1"), [("tall_bottle_1", "upright_on", "tray_1")])
        self.assertIn("tall_bottle", HS.GRASPABLE)
        self.assertIn("mat", HS.CONTAINERS)

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_k4_h6_side_grasp_upright(self):
        import NERO_V410_Household_Scenes as HS
        import NERO_V489_General_Pick_Place as GP
        sc = HS.fixed_scene("H6")
        io = HS.make_twin(sc, log=lambda s: None)
        sk = GP.GeneralPickPlaceSkills(io)
        r = sk.run_tool("perceive_objects", PERCEIVE_H6)
        bt = [o for o in r["objects"] if isinstance(o, dict) and o.get("object_id") == "tall_bottle_1"][0]
        self.assertLess(bt["stability_estimate"]["tip_over_angle_deg"], 20.0)
        pv = sk.run_tool("preview_grasp", {k: GRASP_H6[k] for k in ("object_id", "xy_offset_mm",
                                            "grasp_z_offset_from_top_mm", "closing_yaw_deg", "pitch_deg", "open_mm")})
        a = pv["physical_verification"]["approach_last_80mm"]["approach_direction_xyz"]
        self.assertAlmostEqual(a[2], 0.0, places=2)                                    # horizontal approach
        p = sk.run_tool("plan_grasp_chunk", GRASP_H6)
        self.assertTrue(p["ok"], json.dumps(p)[:600])
        self.assertAlmostEqual(p["physical_verification"]["approach_direction_xyz"][2], 0.0, places=2)
        e = sk.run_tool("execute_grasp_chunk", {"clamp_force_n": 15})
        self.assertTrue(e["ok"], json.dumps(e)[:800])
        f = e["physical_verification"]
        self.assertIn("object_in_gripper_during_lift", f)                              # lift probe
        self.assertLess(f["object_in_gripper_during_lift"]["slid_down_mm"], 2.0)
        self.assertGreater(f["object_rose_mm"], 80.0)
        pl = sk.run_tool("plan_place_chunk", PLACE_H6)
        self.assertTrue(pl["ok"], json.dumps(pl)[:600])
        self.assertGreaterEqual(pl["physical_verification"]["held_object_bottom_clearance_mm"], 3.0)
        ep = sk.run_tool("execute_place_chunk", {"retreat_up_mm": 60, "retreat_back_mm": 70})
        self.assertTrue(ep["ok"], json.dumps(ep)[:600])
        self.assertTrue(ep["physical_verification"]["retreat_back"]["ok"])
        g = HS.goal_status(io.m, io.d, sc)
        self.assertTrue(g[0]["ok"], g)
        self.assertIn("tilt", g[0]["detail"])

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_k5_h7_push(self):
        import NERO_V410_Household_Scenes as HS
        import NERO_V489_General_Pick_Place as GP
        sc = HS.fixed_scene("H7")
        io = HS.make_twin(sc, log=lambda s: None)
        sk = GP.GeneralPickPlaceSkills(io)
        r = sk.run_tool("perceive_objects", {"prompts": ["book", "mat", "mug"], "use_finetuned": True})
        ob = {o["object_id"]: o for o in r["objects"] if isinstance(o, dict) and "object_id" in o}
        path, ang = _push_path(ob["book_1"]["center_xyz_m"], ob["mat_1"]["center_xyz_m"], ob["book_1"]["axis_a_yaw_deg"])
        args = {"path_xyz_m": path, "closing_yaw_deg": ang, "pitch_deg": 0, "gripper_open_mm": 0, "speed_mm_s": 30,
                "may_touch": ["mat_1"]}
        bad = sk.run_tool("plan_motion_chunk", args)                       # the book is not allowed to be touched
        self.assertFalse(bad["ok"])
        self.assertEqual(bad.get("reason") or bad["physical_verification"].get("reason"), "MOTION_COLLISION")
        self.assertFalse(sk.run_tool("execute_motion_chunk", {})["ok"])
        m = sk.run_tool("plan_motion_chunk", dict(args, may_touch=["book_1", "mat_1"]))
        self.assertTrue(m["ok"], json.dumps(m)[:800])
        segs = m["physical_verification"]["segments"]
        pc = [s["predicted_contact"] for s in segs if s.get("predicted_contact")]
        self.assertEqual(pc[0]["object"], "book_1")
        e = sk.run_tool("execute_motion_chunk", {})
        self.assertTrue(e["ok"], json.dumps(e)[:800])
        self.assertEqual(e["physical_verification"]["segments_done"], len(segs))
        mv = e["physical_verification"]["objects_listed_in_may_touch"]["book_1"]
        self.assertGreater(float(np.hypot(*mv["moved_xy_mm"])), 150.0)
        g = HS.goal_status(io.m, io.d, sc)
        self.assertTrue(g[0]["ok"], g)
        # holding: no motion chunk
        sk.held = "book_1"
        self.assertEqual(sk.run_tool("plan_motion_chunk", args).get("reason")
                         or sk.run_tool("plan_motion_chunk", args)["physical_verification"].get("reason"), "HOLDING")


if __name__ == "__main__":
    unittest.main(verbosity=2)
