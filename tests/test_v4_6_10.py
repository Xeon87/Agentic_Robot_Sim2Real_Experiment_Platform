#!/usr/bin/env python3
"""NERO V4.6.10 offline tests.

T1 the V4.6.9 run: after the place the carton fit turned ~37 deg, the slot numbers moved and the egg stood in the new
   "slot_5" -> verify by the RELEASE POINT finds it (V4.6.9 looked at the new "slot_1" and failed).
T2 a verify look whose straight raise fails is raised in joint space (V4.6.9: 'Cartesian corridor only 0 %').
T3 verify_egg_try_harder: wrist camera sees the egg -> verified; nothing seen, egg gone from the table, nothing near
   the release point -> UNCONFIRMED (ok + warning); an egg lying near the release point -> EGG_FELL_OUT; the egg still
   where it was picked -> NOT_PLACED.
T4 fill the carton with eggs the camera never confirms: every egg is UNCONFIRMED, the task goes on, each slot is used
   once (placed eggs keep their slot occupied in later observations), run_execute ok + unconfirmed.
T5 task wording: 'Put the eggs into the empty slots, one by one.' fills the carton.
T6 task summary: an unconfirmed step -> YELLOW 'UNCONFIRMED', not SUCCESS; egg tasks are labelled 固定流程.
T7 main camera depth default 848x480 + Default at start (NERO_DEPTH_KEEP_SAVED=1 keeps the saved values).
T8 versions / markers.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTS = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["NERO_CALIB_DIR"] = tempfile.mkdtemp(prefix="nero_v4610_calib_")
os.environ.pop("NERO_EGG_CARTON_GRID", None)

import NERO_V365_Real_Cup_Task as RT         # noqa: E402
import test_v4_6_8 as T8                     # noqa: E402


def src(f):
    with open(os.path.join(HERE, f), encoding="utf-8") as fh:
        return fh.read()


def rotated_view(scene, eggs_extra, occ_at):
    """a perception whose carton fit is rotated by 37 deg about the carton centre with shifted numbering"""
    c = np.array([-0.346, 0.143])
    a = math.radians(37.0)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    slots = []
    pts = [c + R @ (np.array(xy) - c) for xy in T8.SLOTS]
    order = [2, 4, 0, 5, 1, 3]                   # new numbering
    for new_id, k in enumerate(order, start=1):
        xy = pts[k]
        occ = any(float(np.linalg.norm(xy - np.asarray(o))) < 0.02 for o in occ_at)
        core = 40.0 if occ else 8.0
        slots.append({"slot_id": new_id, "center_base": [float(xy[0]), float(xy[1]), 0.022], "occupied": occ,
                      "rise": {"core_top_mm": core, "rim_mm": 22.0, "rise_mm": core - 22.0}})
    eggs = [list(e) for e in scene.eggs] + [list(e) for e in eggs_extra]
    return {"ok": True, "scene": "fill_egg_holder", "table_z_m": 0.0, "n_2d": 3,
            "eggs": [{"center_base": e, "grasp_yaw_deg": -120.0, "width_m": 0.042, "length_m": 0.055} for e in eggs],
            "carton": {"center_base": [-0.346, 0.143, 0.025], "yaw_deg": 37.0, "pitch_mm": [46.0, 47.0], "rows": 2,
                       "cols": 3, "slots": slots, "free_slots": [], "occupied_slots": []}, "dets": [], "cups": []}


class TestV4610(unittest.TestCase):
    def test_t1_verify_by_release_point(self):
        sc = T8.Scene()
        sk, io = T8.make(sc)
        sk.observe_scene()
        p3 = np.asarray(sk.cam["slot_3"]["rim"][:2], float)
        rel = p3 + np.array([0.003, -0.004])             # where the egg was let go (measured joints)
        sk.last_place = {"slot": "slot_3", "xy": list(rel), "source_xy": [-0.40, -0.03]}
        egg_in = [float(rel[0]) - 0.008, float(rel[1]) + 0.010, 0.04]
        # the rotated fit: the egg is no longer at the new "slot_3"; it is in the new slot whose centre is near p3
        io.perceive_eggs = lambda tag: rotated_view(sc, [egg_in], [p3])
        new3 = [s for s in rotated_view(sc, [], [])["carton"]["slots"] if s["slot_id"] == 3][0]
        self.assertGreater(float(np.linalg.norm(np.asarray(new3["center_base"][:2]) - p3)), 0.03)   # numbers moved
        r = sk.verify_egg_in_slot("slot_3", looks=1)
        self.assertTrue(r["ok"], io.logs[-3:])
        lk = r["physical_verification"]["looks"][0]
        self.assertEqual(lk["point_source"], "松开点")
        self.assertTrue(any("[V4.6.10 VERIFY]" in s and "松开点旁" in s for s in io.logs), io.logs[-3:])
        # nothing near the release point (the egg is in another slot) -> not verified
        sk2, io2 = T8.make(T8.Scene())
        sk2.observe_scene()
        sk2.last_place = {"slot": "slot_3", "xy": list(p3), "source_xy": None}
        far = np.asarray(sk2.cam["slot_1"]["rim"][:2], float)
        io2.perceive_eggs = lambda tag: rotated_view(sc, [[float(far[0]), float(far[1]), 0.04]], [far])
        self.assertFalse(sk2.verify_egg_in_slot("slot_3", looks=1)["ok"])

    def test_t2_joint_raise(self):
        sc = T8.Scene()
        sk, io = T8.make(sc)
        sk.observe_scene()
        moves = []
        io.move_line = lambda *a, **k: {"ok": False, "error": "Cartesian corridor only 0.0% (IK continuity broke)"}
        io.go_home = lambda label: {"ok": True}
        io.move_joints = lambda label, q, **k: moves.append((label, list(q))) or {"ok": True}
        sk._fast = types.SimpleNamespace(ik=lambda seeds, xyz, R, iters=80: ([0.1] * 7, 0.0, 0.0), fk=sk._fast.fk)
        sc.verify_plan = [{}, {}, {}]
        sk.verify_egg_in_slot("slot_3", looks=3)
        self.assertEqual([m[0] for m in moves], ["V4.6.10 verify look +100 mm (joint)", "V4.6.10 verify look +200 mm (joint)"])
        self.assertTrue(any("关节空间抬高 100 mm" in s for s in io.logs))

    def _th(self, wrist_ok=False, table_eggs=(), source=(-0.40, -0.03)):
        sc = T8.Scene()
        sk, io = T8.make(sc)
        sk.observe_scene()
        p3 = np.asarray(sk.cam["slot_3"]["rim"][:2], float)
        sk.last_place = {"slot": "slot_3", "xy": list(p3), "source_xy": list(source)}
        sk.plan = {"target_slot": "slot_3"}
        sk.held = None
        sk._wrist_look_at = lambda xy, z: ({"ok": True, "trusted": True, "shift_norm_mm": 4.0,
                                             "measured_xyz": [xy[0], xy[1], z]} if wrist_ok
                                            else {"ok": False, "reason": "腕部相机局部 ROI 内未检测到鸡蛋轮廓"})
        sc.eggs = [list(e) for e in table_eggs]
        return sk, io, p3

    def test_t3_try_harder_verify(self):
        sk, io, _ = self._th(wrist_ok=True)
        r = sk.run_tool("verify_egg_try_harder", {"target_slot": "slot_3"})
        self.assertTrue(r["ok"] and not r.get("unconfirmed"))
        self.assertEqual(r["physical_verification"]["how"], "wrist_camera")
        sk, io, _ = self._th(table_eggs=[[-0.43, -0.05, 0.021]])          # another egg far away; source spot empty
        r = sk.run_tool("verify_egg_try_harder", {"target_slot": "slot_3"})
        self.assertTrue(r["ok"] and r["unconfirmed"], r)
        self.assertTrue(r["warnings"])
        self.assertTrue(any("大概率已放入（未确认）" in s for s in io.logs))
        sk, io, p3 = self._th()
        sk.io.perceive_eggs.__self__.eggs = [[float(p3[0]) - 0.07, float(p3[1]) - 0.0, 0.021]]   # lies 70 mm away
        r = sk.run_tool("verify_egg_try_harder", {"target_slot": "slot_3"})
        self.assertEqual(r.get("reason"), "EGG_FELL_OUT")
        sk, io, _ = self._th(table_eggs=[[-0.40, -0.03, 0.021]])          # still where it was picked
        r = sk.run_tool("verify_egg_try_harder", {"target_slot": "slot_3"})
        self.assertEqual(r.get("reason"), "NOT_PLACED")

    def test_t4_fill_unconfirmed(self):
        sc = T8.Scene()
        sk, io = T8.make(sc)
        orig = RT.RealEggSkills.place_egg_into_slot.__get__(sk)

        def place(target_slot=None, **k):              # the egg leaves the table; the camera never sees it again
            e = np.asarray(sk.cam[sk.plan["source_egg"]]["rim"], float).copy()
            r = orig(target_slot=target_slot, **k)
            if r.get("ok"):
                sc.eggs = [x for x in sc.eggs if math.hypot(x[0] - e[0], x[1] - e[1]) > 0.005]
            return r
        sk.place_egg_into_slot = place
        sk._wrist_look_at = lambda xy, z: {"ok": False, "reason": "腕部相机未就绪"}
        out = sk.fill_carton_episode(try_harder=True)
        fr = sk.fill_result
        self.assertEqual(fr["fails"], 0, io.logs[-6:])
        self.assertEqual(len(fr["unconfirmed"]), 3)
        self.assertEqual(len(set(fr["unconfirmed"])), 3)                 # every slot used once
        self.assertEqual(fr["placed"], [])
        self.assertEqual(len(sk.placed_xy), 3)
        self.assertTrue(any("记为未确认，继续下一个鸡蛋" in s for s in io.logs))
        self.assertTrue(any(r.get("unconfirmed") for r in out))
        # run_execute: ok + unconfirmed
        sc = T8.Scene()
        sk2, io2 = T8.make(sc)
        orig_make = RT.make_skills
        RT.make_skills = lambda *a, **k: sk2
        sk2._wrist_look_at = lambda xy, z: {"ok": False, "reason": "-"}
        sk2.verify_egg_in_slot = lambda **k: sk2._res("verify_egg_in_slot", False, reason="SLOT_EMPTY")
        try:
            r = RT.run_execute(io2, "egg_1", "slot_1", scene="fill_egg_holder", try_harder=True, fill_all=False)
        finally:
            RT.make_skills = orig_make
        self.assertTrue(r["ok"] and r["unconfirmed"], [x.get("tool") for x in r["results"]])
        self.assertEqual([x["tool"] for x in r["results"]][-3:],
                         ["verify_egg_in_slot", "verify_egg_try_harder", "return_to_safe_pose"])

    def test_t5_wording(self):
        for t in ("Put the eggs into the empty slots, one by one.", "Put all eggs in the carton", "把鸡蛋逐个放进蛋托",
                  "依次把鸡蛋放进空槽", "Place the remaining eggs"):
            self.assertTrue(RT.is_fill_all_task(t), t)
        for t in ("Put egg_1 into slot_1", "Put the egg into slot_2", "把 egg_1 放进 slot_1"):
            self.assertFalse(RT.is_fill_all_task(t), t)

    def test_t6_summary(self):
        import NERO_V368_Real_Task_Record as TR
        steps = [{"n": 1, "tool": "verify_egg_in_slot", "ok": False, "reason": "SLOT_EMPTY"},
                 {"n": 2, "tool": "verify_egg_try_harder", "ok": True, "unconfirmed": True, "warnings": ["x"]}]
        txt, ok = TR.task_summary("Put the eggs", "execute (固定流程 + Try harder)", steps, True, 10.0)
        self.assertIn("[REAL TASK SUMMARY YELLOW]", txt)
        self.assertIn("STATUS: UNCONFIRMED", txt)
        self.assertNotIn("STATUS: SUCCESS", txt)
        s = src("NERO_V365_Real_Cup_Task.py")
        self.assertIn('llm_drives = use_llm and scene not in ("fill_egg_holder", "sort_items")', s)
        self.assertIn("mode + mode_tag", s)
        self.assertNotIn('mode + (" (LLM)" if use_llm else "")', s)
        self.assertIn('"unconfirmed": bool((res or {}).get("unconfirmed"))', src("NERO_V368_Real_Task_Record.py"))

    def test_t7_depth_default(self):
        import NERO_V312_Console as C

        class V:
            def __init__(self, v):
                self.v = v

            def get(self):
                return self.v

            def set(self, v):
                self.v = v
        logs = []
        cam = types.SimpleNamespace(v_dres=V("1280×720"), v_dpreset=V("High Accuracy"))
        old = C.v4610_depth_defaults(cam, logs.append)
        self.assertEqual(old, ("1280×720", "High Accuracy"))
        self.assertEqual((cam.v_dres.get(), cam.v_dpreset.get()), ("848×480", "Default"))
        self.assertTrue(logs and "848×480 · Default" in logs[0])
        import NERO_V325_Camera as CAM
        self.assertIn(C.V4610_DEPTH_RES, CAM.DEPTH_RES)
        self.assertIn(C.V4610_DEPTH_PRESET, CAM.DEPTH_PRESETS)
        os.environ["NERO_DEPTH_KEEP_SAVED"] = "1"
        try:
            cam = types.SimpleNamespace(v_dres=V("1280×720"), v_dpreset=V("High Accuracy"))
            self.assertIsNone(C.v4610_depth_defaults(cam))
            self.assertEqual(cam.v_dres.get(), "1280×720")
        finally:
            os.environ.pop("NERO_DEPTH_KEEP_SAVED")
        s = src("NERO_V312_Console.py")
        self.assertLess(s.index("v4610_depth_defaults(cam_tab, cam_log_line)"), s.index("cam_tab.attach_workflow_bar(real_wf_bar)"))

    def test_t8_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual((C.VERSION, RR.VERSION), (v, v))
        self.assertIn(v, ("V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13"))
        self.assertEqual(RT.MARKER_V4610, "V4.6.10 VERIFY BY RELEASE POSITION + TRY HARDER VERIFY + UNCONFIRMED CONTINUE")
        self.assertEqual(C.MARKER_V4610, "V4.6.10 DEPTH DEFAULT 848x480 + DEFAULT")


if __name__ == "__main__":
    unittest.main(verbosity=1)
