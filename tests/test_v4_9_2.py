#!/usr/bin/env python3
"""NERO V4.9.2 offline tests (before the general pick & place goes to the real robot: real-like perception in the
twin, a real-LLM batch evaluation, smaller tool outputs).

N1 footprint axes = minimum-area rectangle: a square block turned 30 deg measures its sides, not its diagonal
   (V4.9 PCA could put a square's axes on the diagonal: a 43 x 45 mm block read 67 x 66 mm with noise).
N2 compact outputs: the gripper geometry + legend only in the first perceive_objects of a task; an object that has
   not moved is a one-line "unchanged" entry; no per-object format / note strings.
N3 collision margin: 0.5 mm with simulator-true geometry, 4 mm with the real-like twin perception, 6 mm on the
   real robot; the gripper facts tell the LLM.
N4 (mujoco) real-like perception: simulated fixed depth camera -> unseen space behind objects (upper bound = the
   grazing camera line, reported, not refused), sides the camera
   does not see are missing, tops within 8 mm, bottoms not measured (table / floor of the opening it stands in).
N5 (mujoco) the real-LLM evaluation plumbing (tests/v49_llm_eval.run_one) with a scripted agent: egg_1 into cup_1.
N6 versions / markers.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import threading
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTS = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("NERO_CALIB_DIR", tempfile.mkdtemp(prefix="nero_v492_"))

import NERO_V49_Geometry as G      # noqa: E402

try:
    import mujoco  # noqa: F401
    HAS_MJ = True
except Exception:
    HAS_MJ = False


class ScriptedAgent:
    """the Responses-API surface run_general_pick_place uses, replaying a fixed tool sequence"""

    def __init__(self, seq):
        self._lock, self._running, self._stop_event = threading.Lock(), False, threading.Event()
        self._last_response_id, self.seq, self.i = None, list(seq), 0
        self._usage_totals = {"input_tokens": 0, "output_tokens": 0}

    def api_key_status(self, k, m):
        return True, "ok"

    def _post_responses(self, payload, key):
        self._usage_totals["input_tokens"] += len(json.dumps(payload, default=str)) // 4
        if self.i < len(self.seq):
            n, a = self.seq[self.i]
            self.i += 1
            return {"id": f"r{self.i}", "calls": [{"name": n, "arguments": json.dumps(a), "call_id": f"c{self.i}"}]}
        return {"id": "done", "calls": [], "text": "done"}

    def _log_usage(self, r, m):
        pass

    def _function_calls(self, r):
        return r.get("calls", [])

    def _extract_text(self, r):
        return r.get("text", "")


class TestV492(unittest.TestCase):
    def test_n1_min_area_rectangle(self):
        res = 0.002
        n = 120
        x0, y0 = -0.40, -0.12
        xs = x0 + (np.arange(n) + 0.5) * res
        ys = y0 + (np.arange(n) + 0.5) * res
        X, Y = np.meshgrid(xs, ys)
        a = math.radians(30.0)
        u = (X + 0.28) * math.cos(a) + (Y + 0.0) * math.sin(a)
        v = -(X + 0.28) * math.sin(a) + (Y + 0.0) * math.cos(a)
        inside = (np.abs(u) <= 0.0215) & (np.abs(v) <= 0.0225)
        H = np.where(inside, 0.04, 0.0)
        own = np.where(inside, 0, -1)
        hm = G.HeightMap(x0, y0, res, H, own, ["block_1"], 0.0)
        d = G.describe(hm, "block_1", 0.0)
        self.assertLess(max(d["size_mm"]), 50.0, d["size_mm"])          # diagonal would be ~62
        self.assertAlmostEqual(sorted(d["size_mm"])[0], 43.0, delta=3.0)

    def _mock(self):
        import NERO_V489_General_Pick_Place as GP
        import test_v4_8_10 as T10
        io = T10._MockSimGeneralIO()
        return GP, io, GP.GeneralPickPlaceSkills(io)

    def test_n2_compact(self):
        GP, io, sk = self._mock()
        a = sk.perceive_objects(prompts=["egg", "cup"])
        self.assertIsInstance(a["gripper"], dict)
        self.assertIn("legend", a)
        self.assertNotIn("gripper_specs", a)
        txt = json.dumps(a, default=str)
        for k in ('"format"', '"size_axes"', '"note": "measured from above'):
            self.assertNotIn(k, txt)
        b = sk.perceive_objects(prompts=["egg", "cup"])
        self.assertIsInstance(b["gripper"], str)
        self.assertNotIn("legend", b)
        self.assertTrue(all(o.get("unchanged") for o in b["objects"]), b["objects"])
        self.assertLess(len(json.dumps(b, default=str)), 0.6 * len(txt))
        io.poses["obj_egg_1"] = ([-0.40, -0.05, 0.022], [0.0, 0.0, 0.0, 1.0])
        c = sk.perceive_objects(prompts=["egg", "cup"])
        e1 = [o for o in c["objects"] if o["object_id"] == "egg_1"][0]
        self.assertNotIn("unchanged", e1)

    def test_n3_margins(self):
        GP, io, sk = self._mock()
        self.assertAlmostEqual(sk._meas_margin_m(), 0.0005)
        io.perception_noise = dict(GP.REAL_LIKE_NOISE)
        self.assertAlmostEqual(sk._meas_margin_m(), 0.004)
        io.perception_noise = None
        io.is_sim = False
        sk2 = GP.GeneralPickPlaceSkills(io)
        self.assertAlmostEqual(sk2._meas_margin_m(), 0.006)
        io.is_sim = True
        r = sk.perceive_objects(prompts=["egg"])
        self.assertIn("collision_margin_mm", r["gripper"])

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_n4_real_like_perception(self):
        import NERO_V489_General_Pick_Place as GP
        io = GP.TwinGeneralIO(noise=True)
        sk = GP.GeneralPickPlaceSkills(io)
        r = sk.run_tool("perceive_objects", {"prompts": ["object"], "use_finetuned": True})
        self.assertGreater(int((sk.hm.Hu > 0.0).sum()), 100)              # shadows behind the objects
        far = np.array([[float(io.d.body("obj_cup_1").xpos[0]) + 0.07, float(io.d.body("obj_cup_1").xpos[1]), 0.01]])
        self.assertGreaterEqual(G.unobserved_depth(sk.hm, far), 0.0)
        cup = sk.desc["cup_1"]
        self.assertAlmostEqual(float(cup["top_z_m"]), float(io.d.body("obj_cup_1").xpos[2]), delta=0.008)
        self.assertEqual(float(cup["bottom_z_m"]), 0.0)
        missing = sum(1 for row in cup["side_profile"] for v in row["outer_half_width_mm"].values() if v is None)
        self.assertGreater(missing, 6)                                  # the far sides are not seen
        self.assertTrue(any(o.get("measured_by") for o in r["objects"]))

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_n5_llm_eval_plumbing(self):
        import NERO_V489_General_Pick_Place as GP
        import v49_llm_eval as E
        seq = [("perceive_objects", {"prompts": ["egg", "cup"], "use_finetuned": True}),
               ("plan_grasp_chunk", {"object_id": "egg_1", "xy_offset_mm": [0, 0], "grasp_z_offset_from_top_mm": -9,
                                     "closing_yaw_deg": 90, "pitch_deg": 0, "open_mm": 60, "yaw_tolerance_deg": 30,
                                     "pregrasp_height_mm": 80, "lift_height_mm": 150}),
               ("execute_grasp_chunk", {"clamp_force_n": 8}),
               ("plan_place_chunk", {"target_id_or_xyz": "cup_1", "xy_offset_mm": [0, 0],
                                     "held_bottom_above_target_top_mm": 15, "yaw_change_deg": 0,
                                     "approach_height_mm": 40, "release_open_mm": 60, "carry_lift_mm": 0}),
               ("execute_place_chunk", {"retreat_up_mm": 60}),
               ("verify_and_park_chunk", {"action": "clear_view_and_perceive", "prompts": ["egg", "cup"]})]
        io = GP.TwinGeneralIO()
        r = E.run_one(io, "Put egg_1 into cup_1.", ScriptedAgent(seq), {"model": "scripted", "max_tool_steps": 20},
                      lambda: E.user_success(io, "egg_1", "cup_1"))
        self.assertTrue(r["success"], r)
        self.assertEqual(r["tool_calls"], 6)
        self.assertGreater(r["input_tokens"], 0)
        self.assertEqual(r["failed_calls"], [])

    def test_n6_versions(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        import NERO_V489_General_Pick_Place as GP
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual((v, RR.VERSION, GP.VERSION), (C.VERSION, C.VERSION, C.VERSION))
        self.assertIn(v, ("V4.9.2", "V4.9.3", "V4.10"))
        self.assertEqual(GP.MARKER_V492, "V4.9.2 REAL-LIKE PERCEPTION + LLM EVAL + COMPACT OUTPUTS")


if __name__ == "__main__":
    unittest.main(verbosity=1)
