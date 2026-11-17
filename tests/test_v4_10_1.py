#!/usr/bin/env python3
"""NERO V4.10.1 offline tests (user GUI run V4.10: 'insert the remote into the mug' was refused - the tools could not
turn a held object, and the simulator's ball-joint hold would not have turned it anyway).

P1 plan_place_chunk has pitch_change_deg (strict schema, required); the gripper facts say whether a held object turns
   with the gripper (held_object_turns_with_gripper).
P2 (mujoco) ball-joint twin (= the backend simulation's hold): a pitch change is refused with that fact.
P3 (mujoco) rigid-hold twin (weld): grasp the remote near one end, turn it 90 deg during the carry, lower it into the
   mug: it ends up standing in the mug (simulator truth), and the final look reports remote_1 inside mug_1 although
   the remote covers the mug's opening.
P4 relations: 'inside' from the outline when B's opening is covered; not for an object resting on B or beside it.
P5 bridge: attach_from_notebook schedules with a fixed delay (V4.10 raised NameError delay_ms); versions / marker.
"""
from __future__ import annotations

import inspect
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
os.environ.setdefault("NERO_CALIB_DIR", tempfile.mkdtemp(prefix="nero_v4101_"))

try:
    import mujoco  # noqa: F401
    HAS_MJ = True
except Exception:
    HAS_MJ = False


def _scene():
    import NERO_V410_Household_Scenes as HS
    return HS.HScene("t", "t", [HS.Item("remote_1", "remote", -0.40, 0.10, 0.0),
                               HS.Item("mug_1", "mug", -0.36, -0.10, 180.0)], [("remote_1", "into", "mug_1")], "")


GRASP = {"object_id": "remote_1", "xy_offset_mm": [50, 0], "grasp_z_offset_from_top_mm": 15, "closing_yaw_deg": 90,
         "pitch_deg": 0, "open_mm": 60, "yaw_tolerance_deg": 180, "pregrasp_height_mm": 80, "lift_height_mm": 180}
PLACE = {"target_id_or_xyz": "mug_1", "xy_offset_mm": [0, 0], "held_bottom_above_target_top_mm": -10,
         "yaw_change_deg": 315, "approach_height_mm": 60, "release_open_mm": 60, "carry_lift_mm": 0, "pitch_change_deg": 90}


class V4101(unittest.TestCase):
    def test_p1_schema(self):
        import NERO_V489_General_Pick_Place as GP
        t = [x for x in GP.GENERAL_TOOLS if x["name"] == "plan_place_chunk"][0]
        self.assertIn("pitch_change_deg", t["parameters"]["properties"])
        self.assertIn("pitch_change_deg", t["parameters"]["required"])
        self.assertTrue(hasattr(GP.SimGeneralIO, "hold_kind"))            # V4.10.2: rigid in the general scenes

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_p2_ball_hold_refuses(self):
        import NERO_V410_Household_Scenes as HS
        import NERO_V489_General_Pick_Place as GP
        io = HS.make_twin(_scene(), rigid_hold=False)
        sk = GP.GeneralPickPlaceSkills(io)
        r = sk.run_tool("perceive_objects", {"prompts": ["remote", "mug"], "use_finetuned": True})
        self.assertFalse(r["gripper"]["held_object_turns_with_gripper"])
        self.assertTrue(sk.run_tool("plan_grasp_chunk", GRASP)["ok"])
        p = sk.run_tool("plan_place_chunk", PLACE)
        self.assertFalse(p["ok"])
        self.assertIn("ball joint", str(p["physical_verification"]["problems"]))

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_p3_rigid_hold_insert(self):
        import NERO_V410_Household_Scenes as HS
        import NERO_V489_General_Pick_Place as GP
        sc = _scene()
        io = HS.make_twin(sc)
        sk = GP.GeneralPickPlaceSkills(io)
        r = sk.run_tool("perceive_objects", {"prompts": ["remote", "mug"], "use_finetuned": True})
        self.assertTrue(r["gripper"]["held_object_turns_with_gripper"])
        g = None
        for off in ([60, 0], [60, 1.5], [60, -1.5], [50, 0], [70, 0], [45, 1]):   # the sim gripper's weak push
            if not sk.run_tool("plan_grasp_chunk", dict(GRASP, xy_offset_mm=off))["ok"]:
                continue
            g = sk.run_tool("execute_grasp_chunk", {"clamp_force_n": 10})
            if g["ok"]:
                break
            sk.run_tool("perceive_objects", {"prompts": ["remote", "mug"], "use_finetuned": True})
        self.assertTrue(g and g["ok"], g)
        p = sk.run_tool("plan_place_chunk", PLACE)
        self.assertTrue(p["ok"], p)
        self.assertEqual(p["physical_verification"]["pitch_change_deg"], 90.0)
        x = sk.run_tool("execute_place_chunk", {"retreat_up_mm": 80})
        self.assertTrue(x["ok"], x)
        io.step(2000)
        R = np.asarray(io.d.body("obj_remote_1").xmat).reshape(3, 3)
        self.assertGreater(abs(float(R[2, 0])), 0.85)                   # long axis (x of the body) is vertical
        self.assertTrue(all(g_["ok"] for g_ in HS.goal_status(io.m, io.d, sc)))
        v = sk.run_tool("verify_and_park_chunk", {"action": "clear_view_and_perceive", "prompts": ["remote", "mug"]})
        rel = v["physical_verification"]["relations"]
        self.assertTrue(any(r_["object"] == "remote_1" and r_["relation"] == "inside" and r_["of"] == "mug_1"
                            for r_ in rel), rel)

    def test_p4_relations(self):
        import NERO_V49_Geometry as G
        mug = {"object_id": "mug_1", "center_xyz_m": [-0.36, -0.1, 0.05], "top_z_m": 0.105, "bottom_z_m": 0.0,
               "size_mm": [120, 100], "axis_a_yaw_deg": 0.0, "open_cavity": None}
        inside = {"object_id": "remote_1", "center_xyz_m": [-0.37, -0.1, 0.09], "top_z_m": 0.18, "bottom_z_m": 0.01,
                  "size_mm": [60, 40], "axis_a_yaw_deg": 0.0}
        on = dict(inside, bottom_z_m=0.106, top_z_m=0.126)
        beside = dict(inside, center_xyz_m=[-0.36, 0.05, 0.02], bottom_z_m=0.0, top_z_m=0.02)
        self.assertEqual(G.relations([inside, mug])[0]["relation"], "inside")
        self.assertEqual([r["relation"] for r in G.relations([on, mug])], ["on"])
        self.assertEqual(G.relations([beside, mug]), [])

    def test_p5_bridge_and_versions(self):
        import NERO_V312_Console as C
        import NERO_V410_Sim_Bridge as B
        import NERO_V489_General_Pick_Place as GP
        src = inspect.getsource(B.attach_from_notebook)
        self.assertNotIn("delay_ms)", src.split("attach_gui(")[0] + src.split("attach_gui(")[1].split("\n", 1)[1])
        self.assertIn("root.after(1500, later)", src)
        with open(os.path.join(HERE, "VERSION")) as f:
            self.assertEqual(f.read().strip(), C.VERSION)
        self.assertEqual((C.VERSION, GP.VERSION), (C.VERSION, C.VERSION))
        self.assertEqual(GP.MARKER_V4101, "V4.10.1 PITCH CHANGE + RIGID TWIN HOLD + BRIDGE FIX")
        old = os.environ.get("NERO_SCENE")
        os.environ["NERO_SCENE"] = "household_pick_place"
        try:
            self.assertEqual(GP._scene_name(), "household_pick_place")
        finally:
            if old is None:
                os.environ.pop("NERO_SCENE", None)
            else:
                os.environ["NERO_SCENE"] = old


if __name__ == "__main__":
    unittest.main(verbosity=1)
