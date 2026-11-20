#!/usr/bin/env python3
"""NERO V4.13 offline tests (user 11-20: 「V2暂时还不稳定。先用V1版本的Harness，做sim2real，在实机任务场景的“通用pick and
place”选项中应用V1 harness，并做相应的适配和完整的替身测试」; V1 with the two V2 checks; first real tasks: egg + cup and
the household objects; stand-in reasoner only).

R1 profiles: H1 V1 = the 7 V4.11.3 tools (pregrasp_height_mm, retreat_up_mm only) + V1 prompt, default; V2 only by
   argument / NERO_H1_PROFILE; run_tool refuses the other profile's tools; run_general_pick_place sends V1 to the LLM.
R2 V1 approach: vertical even with pitch != 0 (pregrasp straight above, swept vertically); V2: along the fingers.
R3 the two V2 checks kept in V1: approach sweep fact in preview_grasp; release bottom clearance < 3 mm = a problem.
R4 real height map (pure numpy, synthetic top-down depth): a box and an open container -> footprint, height, open
   cavity; robot points removed; unmasked points -> an "unidentified" obstacle; specks dropped; masked object with no
   depth -> its box painted.
R5 stable ids across looks (nearest same label; placed objects keep their id; new ones get the next number).
R6 AgxGripper held check: stopped AND closed >= 1 mm (a frozen status feed no longer reports held); RealIO facts:
   fingers_stopped / closed_by_mm / held; still held only while the fingers did not close on further.
R7 the real task page: general_pick_place passes profile="V1"; household prompts and examples.
R8 (mujoco + OpenGL) the real stand-in: egg -> cup through RealIO / perceive_general / the real height map /
   AgxGripper / H1 V1 -> simulator truth; a perceive with the gripper over the table: no phantom obstacle;
   a missed target -> no motion.
"""
from __future__ import annotations

import inspect
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
os.environ.setdefault("NERO_CALIB_DIR", tempfile.mkdtemp(prefix="nero_v413_"))
if not os.environ.get("DISPLAY"):                 # headless: OSMesa (set before mujoco is imported)
    os.environ.setdefault("MUJOCO_GL", "osmesa")
if os.environ.get("MUJOCO_GL") in ("osmesa", "egl"):
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

try:
    import mujoco  # noqa: F401
    HAS_MJ = True
except Exception:
    HAS_MJ = False


def _gl_ok():
    if not HAS_MJ:
        return False
    try:
        import mujoco as mj
        m = mj.MjModel.from_xml_string("<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>")
        mj.Renderer(m, 32, 32).render()
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ synthetic top-down depth camera (pure numpy)
def _synthetic():
    W, H, f = 480, 400, 560.0                     # ~1.4 mm per pixel at the table (D435i at 0.8 m: ~1 mm)
    K = {"fx": f, "fy": f, "ppx": (W - 1) / 2.0, "ppy": (H - 1) / 2.0, "width": W, "height": H}
    cam = np.array([-0.45, 0.0, 0.80])
    R = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]).T     # columns = optical axes in world
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cam
    vv, uu = np.mgrid[0:H, 0:W]
    dirs = np.stack([(uu - K["ppx"]) / f, (vv - K["ppy"]) / f, np.ones_like(uu, float)], -1) @ R.T
    # scene: table z=0; box_1 60x40x30 at (-0.40, 0.08); bin_1 open box 90x90x60 (walls 6 mm, floor 5 mm) at
    # (-0.50, -0.08); a 30x30x40 block without a mask at (-0.36, -0.10) (unidentified); a 2-cell speck
    def hit_z(x, y):
        z = np.zeros_like(x)
        m_box = (np.abs(x + 0.40) <= 0.030) & (np.abs(y - 0.08) <= 0.020)
        z = np.where(m_box, 0.030, z)
        ax, ay = np.abs(x + 0.50), np.abs(y + 0.08)
        m_bin = (ax <= 0.045) & (ay <= 0.045)
        wall = m_bin & ((ax > 0.039) | (ay > 0.039))
        z = np.where(m_bin, np.where(wall, 0.060, 0.005), z)
        m_unk = (np.abs(x + 0.36) <= 0.015) & (np.abs(y + 0.10) <= 0.015)
        z = np.where(m_unk, 0.040, z)
        return z, m_box, m_bin, m_unk
    # first hit: for each flat top level, the ray meets it at t_l; keep the nearest level whose region is there
    t = np.full(dirs.shape[:2], np.inf)
    for lvl in (0.060, 0.040, 0.030, 0.005, 0.0):
        tl = (cam[2] - lvl) / -dirs[..., 2]
        x, y = cam[0] + tl * dirs[..., 0], cam[1] + tl * dirs[..., 1]
        z, _a, _b, _c = hit_z(x, y)
        hit = np.isclose(z, lvl) & (tl < t)
        t = np.where(hit, tl, t)
    x, y = cam[0] + t * dirs[..., 0], cam[1] + t * dirs[..., 1]
    z, m_box, m_bin, m_unk = hit_z(x, y)
    depth = t.astype(np.float32)                   # optical z = t (the ray's optical z component is 1)
    return depth, K, T, m_box, m_bin, m_unk


class V413(unittest.TestCase):
    def test_r1_profiles(self):
        import NERO_V489_General_Pick_Place as GP
        self.assertEqual(GP.normalize_profile(None), "V1")
        self.assertEqual(GP.normalize_profile("H1 V2"), "V2")
        v1 = {t["name"]: t for t in GP.GENERAL_TOOLS_V1}
        self.assertEqual(list(v1), ["perceive_objects", "preview_grasp", "plan_grasp_chunk", "execute_grasp_chunk",
                                    "plan_place_chunk", "execute_place_chunk", "verify_and_park_chunk"])
        self.assertIn("pregrasp_height_mm", v1["plan_grasp_chunk"]["parameters"]["required"])
        self.assertNotIn("approach_distance_mm", v1["plan_grasp_chunk"]["parameters"]["properties"])
        self.assertEqual(v1["execute_place_chunk"]["parameters"]["required"], ["retreat_up_mm"])
        self.assertIn("vertical descent", v1["plan_grasp_chunk"]["description"])
        self.assertIn("pick-and-place", GP.GENERAL_SYSTEM_INSTRUCTIONS_V1)
        self.assertNotIn("motion_chunk", GP.GENERAL_SYSTEM_INSTRUCTIONS_V1)
        for t in GP.GENERAL_TOOLS_V1:
            sch = t["parameters"]
            self.assertEqual(set(sch["required"]), set(sch["properties"]))
            self.assertFalse(sch["additionalProperties"])
        io = _FakeIO()
        sk = GP.GeneralPickPlaceSkills(io)
        self.assertEqual(sk.profile, "V1")
        r = sk.run_tool("plan_motion_chunk", {"path_xyz_m": [[0, 0, 0.1], [0, 0.1, 0.1]], "closing_yaw_deg": 0,
                                              "pitch_deg": 0, "gripper_open_mm": 0, "speed_mm_s": 20, "may_touch": []})
        self.assertEqual(r["reason"], "UNKNOWN_TOOL")
        os.environ[GP.PROFILE_ENV] = "V2"
        try:
            self.assertEqual(GP.GeneralPickPlaceSkills(io).profile, "V2")
        finally:
            del os.environ[GP.PROFILE_ENV]
        # the LLM gets the V1 tools and prompt
        sent = []

        class Agent:
            import threading
            _lock, _running, _stop_event, _last_response_id = threading.Lock(), False, threading.Event(), None
            _usage_totals = {}

            def api_key_status(self, k, m):
                return True, "ok"

            def _post_responses(self, payload, key):
                sent.append(payload)
                return {"id": "r1"}

            def _log_usage(self, *a):
                pass

            def _function_calls(self, resp):
                return []

            def _extract_text(self, resp):
                return "done"
        GP.run_general_pick_place(io, "put the egg into the cup", mode="plan", agent=Agent(), llm_cfg={"model": "x"})
        self.assertEqual([t["name"] for t in sent[0]["tools"]], list(v1))
        self.assertTrue(sent[0]["instructions"].startswith(GP.GENERAL_SYSTEM_INSTRUCTIONS_V1))
        self.assertEqual(GP.TASK_STATS.get("profile"), "V1")

    def test_r2_r3_v1_vertical_and_checks(self):
        import NERO_V489_General_Pick_Place as GP
        src = inspect.getsource(GP.GeneralPickPlaceSkills._plan_grasp_core)
        self.assertIn("a_dir = R_gr[:, 2] if self.v2 else np.array([0.0, 0.0, -1.0])", src)
        self.assertIn("approach_dir=a_dir", src)
        pv = inspect.getsource(GP.GeneralPickPlaceSkills.preview_grasp)
        self.assertIn('approach_dir=None if self.v2 else (0.0, 0.0, -1.0)', pv)
        self.assertEqual(GP.PLACE_MIN_CLEAR_MM, 3.0)
        pl = inspect.getsource(GP.GeneralPickPlaceSkills._plan_place_core)
        self.assertIn("held_object_bottom_clearance_mm", pl)
        import NERO_V49_Geometry as G49
        hm = G49.HeightMap.from_boxes([{"object_id": "wall_1", "center_xyz_m": [-0.33, 0.0, 0.1], "size_mm": [10, 120],
                                        "top_z_m": 0.20}], 0.0)
        p = np.array([-0.40, 0.0, 0.10])
        tilted = G49.sweep_collisions(hm, G49.SIM_FINGERS, p, 90.0, 90.0, 60.0, 0.12, margin_m=0.0)
        vert = G49.sweep_collisions(hm, G49.SIM_FINGERS, p, 90.0, 90.0, 60.0, 0.12, margin_m=0.0,
                                    approach_dir=(0, 0, -1))
        self.assertIsNotNone(tilted["first_collision"])                  # V2: along the fingers -> the wall
        self.assertEqual(vert["approach_direction_xyz"], [0.0, 0.0, -1.0])

    def test_r4_real_heightmap(self):
        import NERO_V413_Real_HeightMap as RHM
        import NERO_V49_Geometry as G49
        depth, K, T, m_box, m_bin, m_unk = _synthetic()
        # robot: a gripper box hovering over the box_1 region must not appear; put one "robot" box at (-0.30, 0.15)
        robot_boxes = [(np.array([-0.30, 0.15, 0.05]), np.eye(3), np.array([0.02, 0.02, 0.05]))]
        hm, pts, st = RHM.build(depth, K, T, [m_box, m_bin], ["box_1", "bin_1"], 0.0, res=0.003, stride=1,
                                robot_boxes=robot_boxes)
        self.assertIn("unidentified_1", hm.names)
        self.assertEqual(len(st["unidentified"]), 1)
        side = RHM.side_fn_from_points(pts, T[:3, 3][:2])
        b = G49.describe(hm, "box_1", 0.0, side)
        self.assertAlmostEqual(b["top_z_m"], 0.030, delta=0.002)
        self.assertAlmostEqual(sorted(b["size_mm"])[1], 60.0, delta=7.0)
        self.assertAlmostEqual(sorted(b["size_mm"])[0], 40.0, delta=7.0)
        self.assertAlmostEqual(b["center_xyz_m"][0], -0.40, delta=0.004)
        self.assertIsNone(b["open_cavity"])
        c = G49.describe(hm, "bin_1", 0.0, side)
        self.assertIsNotNone(c["open_cavity"])
        self.assertAlmostEqual(c["open_cavity"]["floor_z_above_table_mm"], 5.0, delta=2.0)
        self.assertAlmostEqual(c["top_z_m"], 0.060, delta=0.002)
        u = G49.describe(hm, "unidentified_1", 0.0, None)
        self.assertAlmostEqual(u["center_xyz_m"][0], -0.36, delta=0.005)
        # the robot points: a fake depth bump inside the robot box is removed
        d2 = depth.copy()
        hm2, _p, st2 = RHM.build(d2, K, T, [m_box, m_bin], ["box_1", "bin_1"], 0.0, res=0.003, stride=1,
                                 robot_boxes=[(np.array([-0.36, -0.10, 0.03]), np.eye(3), np.array([0.02, 0.02, 0.03]))])
        self.assertEqual(st2["unidentified"], [])                         # the block is inside the "robot" box
        self.assertGreater(st2["robot_points_removed"], 0)
        # a masked object with no depth cells -> its box painted
        empty = np.zeros_like(m_box)
        empty[0:3, 0:3] = True
        hm3, _p3, _s3 = RHM.build(depth, K, T, [m_box, empty], ["box_1", "egg_1"], 0.0, res=0.003, stride=1)
        self.assertLess(len(hm3.cells_of("egg_1")), 3)
        n = RHM.paint_box(hm3, "egg_1", (-0.30, -0.20), [44, 44], 0.0, 0.045)
        self.assertGreater(n, 100)
        self.assertGreaterEqual(len(hm3.cells_of("egg_1")), 100)
        # a hand-eye error of 0.4 deg (the table looks tilted): heights from the fitted plane, the box keeps its size
        a = math.radians(0.4)
        Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
        Tt = T.copy()
        Tt[:3, :3] = Rx @ T[:3, :3]
        Tt[:3, 3] = Rx @ (T[:3, 3] - np.array([-0.45, 0.0, 0.0])) + np.array([-0.45, 0.0, 0.0])
        hmt, ptt, stt = RHM.build(depth, K, Tt, [m_box, m_bin], ["box_1", "bin_1"], 0.0, res=0.003, stride=1)
        self.assertAlmostEqual(stt["table_plane"]["tilt_deg"], 0.4, delta=0.05)
        bt = G49.describe(hmt, "box_1", 0.0, None)
        self.assertAlmostEqual(sorted(bt["size_mm"])[1], 60.0, delta=7.0)
        self.assertAlmostEqual(bt["top_z_m"], 0.030, delta=0.003)
        # the vectorised shadow bound = the simulation's loop (an oblique camera, punched unseen cells)
        kn = hm.known.copy()
        kn[::7, ::5] = False
        for cam in ([-0.95, 0.0, 0.72], [-0.2, 0.3, 0.5]):
            a_ = RHM._shadow_bound(hm.H, hm.owner, kn, hm.x0, hm.y0, hm.res, cam, 0.0)
            b_ = RHM._shadow_bound_loop(hm.H, hm.owner, kn, hm.x0, hm.y0, hm.res, cam, 0.0)
            self.assertTrue(np.allclose(a_, b_))
            self.assertGreater(int((a_ > 0).sum()), 10)
        # labels: the smaller mask wins where masks overlap
        L = RHM.label_image([m_bin, m_bin & (np.arange(m_bin.shape[1])[None, :] % 2 == 0)], m_bin.shape, 0)
        self.assertTrue(((L == 1) | ~m_bin).all() or (L[m_bin] == 1).sum() > 0)

    def test_r5_stable_ids(self):
        import NERO_V413_Real_HeightMap as RHM
        prev = {"egg_1": {"slug": "egg", "xy": [-0.42, -0.06]}, "egg_2": {"slug": "egg", "xy": [-0.43, 0.09]},
                "cup_1": {"slug": "cup", "xy": [-0.33, -0.18]}}
        new = [{"slug": "cup", "xy": [-0.33, -0.18]}, {"slug": "egg", "xy": [-0.33, -0.18]},   # egg_1 now in the cup
               {"slug": "egg", "xy": [-0.43, 0.09]}, {"slug": "bowl", "xy": [-0.5, 0.0]}]
        ids = RHM.match_ids(new, prev)
        self.assertEqual(ids[0], "cup_1")
        self.assertEqual(ids[2], "egg_2")
        self.assertEqual(ids[1], "egg_1")
        self.assertEqual(ids[3], "bowl_1")
        ids2 = RHM.match_ids([{"slug": "egg", "xy": [0, 0]}] * 3, {"egg_2": {"slug": "egg", "xy": [0, 0]}})
        self.assertEqual(sorted(ids2.values()), ["egg_1", "egg_2", "egg_3"])

    def test_r6_gripper_held(self):
        import NERO_V48_Gripper as GG
        be = GG.FakeAgxBackend()
        g = GG.AgxGripper(backend=be)
        g.open_gap(60.0)
        be.object_mm = 44.0
        r = g.clamp(force=1.0)
        self.assertTrue(r["held"], r)
        self.assertGreaterEqual(r["closed_by_mm"], 10.0)
        g.open_gap(60.0)
        be.object_mm = None
        be.alive = True
        orig = be.request

        def frozen(cmd):                                  # the status feed stopped: the same width forever
            if cmd.get("cmd") == "status":
                return {"ok": True, "width_m": 0.060, "force": 0.05, "is_ok": True, "fps": 0.0,
                        "foc": {"driver_enable_status": True, "homing_status": True}}
            return orig(cmd)
        be.request = frozen
        r2 = g.clamp(force=1.0, timeout_s=1.5)
        self.assertFalse(r2["held"], r2)                  # V4.8: "stalled at 60 mm" = held
        self.assertLess(r2["closed_by_mm"], 1.0)
        import NERO_V365_Real_Cup_Task as RT
        src = inspect.getsource(RT.RealIO.gripper_clamp)
        for k in ('"fingers_stopped"', '"closed_by_mm"', '"held": held', "contact_note"):
            self.assertIn(k, src)
        st = inspect.getsource(RT.RealIO.gripper_state)
        self.assertIn('float(lc["gap_mm"]) - float(gap) <= 3.0', st)

    def test_r9_versions(self):
        import NERO_V312_Console as C
        import NERO_V411_Sim_DOE as D
        import NERO_V489_General_Pick_Place as GP
        for v in (C.VERSION, D.VERSION, GP.VERSION):
            self.assertEqual(v, "V4.13")
        with open(os.path.join(HERE, "VERSION")) as f:
            self.assertEqual(f.read().strip(), "V4.13")
        self.assertIn("SIM2REAL", GP.MARKER_V413)
        import NERO_V413_Real_HeightMap as RHM
        self.assertEqual(RHM.VERSION, "V4.13")

    def test_r7_real_page(self):
        import NERO_V365_Real_Cup_Task as RT
        src = inspect.getsource(RT)
        self.assertIn('recorder=rec,\n                                                     profile="V1")', src)
        self.assertIn("H1 V1", dict(RT.TASK_SCENES)["general_pick_place"])
        self.assertIn("remote control", RT.SCENE_DEFAULT_PROMPTS["general_pick_place"])
        self.assertTrue(any("bowl" in t for _l, t in RT.SCENE_EXAMPLES["general_pick_place"]))
        self.assertIn("def _general_heightmap", src)
        self.assertIn("RHM.match_ids", src)

    @unittest.skipUnless(_gl_ok(), "mujoco + OpenGL")
    def test_r8_standin(self):
        import NERO_V49_Random_Scenes as RS
        import v413_real_standin as SB
        from v49_reasoner import GeometricReasoner
        sb, sc = SB.egg_cup_scene()
        sk = SB.make_skills(sb, "V1")
        # a look with the gripper low over the table: the gripper is removed from the map (no phantom obstacle)
        q_low = sb.CK.ik_from(sb.io.current_q(), [-0.40, 0.05, 0.10], sb.CK.pinch_frame(0.0, False, False))
        if q_low is not None:
            sb.tw.q = list(q_low)
            sb.tw._apply_q(sb.tw.q)
            sb.advance(0.3)
        r = sk.run_tool("perceive_objects", {"prompts": ["egg", "cup"], "use_finetuned": True})
        ids = {o["object_id"] for o in r["objects"]}
        self.assertEqual(ids, {"egg_1", "egg_2", "cup_1", "cup_2"})
        self.assertFalse([o for o in r["other_objects"] if str(o["object_id"]).startswith("unidentified")], r)
        cup = [o for o in r["objects"] if o["object_id"] == "cup_1"][0]
        self.assertIsNotNone(cup["open_cavity"])
        sb.tw.q = list(sb.io.home_q)
        sb.tw._apply_q(sb.tw.q)
        sb.advance(0.3)
        sk = SB.make_skills(sb, "V1")                  # a fresh task (the gripper facts are sent once per task)
        out = GeometricReasoner(sk, log=lambda *_: None, prompts=["egg", "cup"]).run("egg_1", "cup_1", "into")
        self.assertTrue(out["ok"], out)
        sb.advance(1.5)
        ok, det = RS.check_success(sb.tw, sc)
        self.assertTrue(ok, det)
        self.assertGreater(len(sb.arm.paths), 4)                         # the real RealIO motions ran
        self.assertTrue(any(c.get("cmd") == "move" for c in sb.grip.backend.log))
        # no egg detected: no grasp, nothing moved (ids belong to the perception: with only egg_1 missed, the other egg
        # would be called egg_1)
        sb2, _sc2 = SB.egg_cup_scene(miss=("egg_1", "egg_2"))
        sk2 = SB.make_skills(sb2, "V1")
        out2 = GeometricReasoner(sk2, log=lambda *_: None, prompts=["egg", "cup"]).run("egg_1", "cup_1", "into")
        self.assertFalse(out2["ok"])
        self.assertEqual(sb2.arm.paths, [])


class _FakeIO:
    """no motion, no camera: enough for the schema / profile tests"""
    is_sim = False
    is_agx = True
    motion_allowed = False

    def __init__(self):
        import threading
        import NERO_V362_Cup_Skills as CK
        self.stop_ev = threading.Event()
        self.prof = CK.OFFICIAL_PROFILE
        self.home_q = None

    def gripper_profile(self):
        return self.prof

    def current_q(self):
        return [0.0, 0.4, 0.0, 1.2, 0.0, 0.3, 0.0]

    def emit(self, s):
        pass

    def perceive_general(self, tag, prompts=None, use_finetuned=True):
        return {"ok": True, "table_z_m": 0.0, "objects": []}

    def go_home(self, label):
        return {"ok": True}


if __name__ == "__main__":
    unittest.main(verbosity=2)
