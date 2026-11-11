#!/usr/bin/env python3
"""NERO V4.6.8 offline tests.

F1 slot occupancy by rise (NERO_V349.slot_rise): synthetic oblique 2x3 carton, eggs standing in 0 / 1 / 2 / 6 cups and
   NOT part of the carton mask -> every slot's occupied flag is right (empty cup: middle below the rim; egg: above).
F2 verify_egg_in_slot: placed when (a) the slot is occupied, (b) an egg lies within 0.5 x pitch of the slot centre or
   (c) the cup middle rose >= 15 mm since before the place - on look 1, else look 2 from higher; an egg 30 mm away /
   no change on all looks -> SLOT_EMPTY with every look logged.
F3 observe_scene: eggs standing in the carton are not source eggs.
F4 jaw clearance: a jaw pointing at an egg standing next to the slot is refused; the place plan next to an occupied
   slot keeps both jaws clear.
F5 fill the carton: 3 eggs on the table, 6 empty slots -> 3 rounds (one recognition each), farthest free slot first,
   every egg verified, the task ends when the table is empty; a refused place without try harder stops after 1 failure.
F6 try harder: a pick that fails once is retried (open, back, look again, re-plan) and the egg is placed; without try
   harder the episode stops; a failed insert is re-planned to the same / another slot (replan_place).
F7 run_execute: 「全部入托」 -> fill mode; try harder for the LLM agent = instructions + twice the tool steps.
F8 short monitor labels: slots = number, eggs = e1.., carton = 蛋托 2×3; no second label for slots / eggs.
F9 GUI wiring: 「Try harder」 checkbox left of the speed box (saved), Camera tab in a scrolling canvas with the
   mouse wheel, vision check recognizes again before every hover by default; versions.
"""
from __future__ import annotations

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
os.environ["NERO_CALIB_DIR"] = tempfile.mkdtemp(prefix="nero_v468_calib_")
os.environ.pop("NERO_EGG_CARTON_GRID", None)

import NERO_V349_Real_Wrist_Camera as W      # noqa: E402
import NERO_V365_Real_Cup_Task as RT         # noqa: E402
import test_v4_5 as T45                      # noqa: E402
import v466_synth_carton as SY               # noqa: E402

SLOTS = [(-0.300, 0.120), (-0.346, 0.120), (-0.392, 0.120), (-0.300, 0.167), (-0.346, 0.167), (-0.392, 0.167)]


class Scene:
    """fake main-camera scene that changes when an egg is placed (eggs on the table, 2x3 carton)"""

    def __init__(self, eggs=None, occ=(), egg_near=None):
        self.eggs = [list(e) for e in (eggs or [[-0.40, -0.03, 0.021], [-0.38, -0.08, 0.021], [-0.43, -0.05, 0.021]])]
        self.occ = set(occ)
        self.extra_eggs = []
        self.looks = []
        self.verify_plan = None                  # list of per-look overrides for verify perceptions

    def perceive(self, tag):
        occ = set(self.occ)
        eggs = [list(e) for e in self.eggs] + [list(e) for e in self.extra_eggs]
        if tag == "verify_egg_in_slot" and self.verify_plan:
            v = self.verify_plan.pop(0)
            occ |= set(v.get("occ", ()))
            eggs += [list(e) for e in v.get("eggs", ())]
            rise_override = v.get("core_top")
        else:
            rise_override = None
        self.looks.append(tag)
        slots = []
        for k, (x, y) in enumerate(SLOTS, start=1):
            core = 40.0 if k in occ else 8.0
            if rise_override is not None and k in rise_override:
                core = rise_override[k]
            slots.append({"slot_id": k, "center_base": [x, y, 0.022], "occupied": k in occ,
                          "rise": {"core_top_mm": core, "rim_mm": 22.0, "rise_mm": core - 22.0}})
        return {"ok": True, "scene": "fill_egg_holder", "table_z_m": 0.0, "fence_z_table_m": None, "n_2d": 3,
                "eggs": [{"center_base": e, "grasp_yaw_deg": -120.0, "width_m": 0.042, "length_m": 0.055} for e in eggs],
                "carton": {"center_base": [-0.346, 0.143, 0.025], "yaw_deg": 0.0, "pitch_mm": [46.0, 47.0],
                           "rows": 2, "cols": 3, "slots": slots,
                           "free_slots": [k for k in range(1, 7) if k not in occ], "occupied_slots": sorted(occ)},
                "dets": [], "cups": []}


def make(scene, grip=None):
    io = RT.RealIO(T45.DummyArm(), grip or T45.Grip(), lambda: {}, log=lambda s: None, motion_allowed=True)
    io.sleep = lambda s: None
    io.cfg["still_window_s"] = 0.0
    io.cfg["settle_s"] = 0.0
    io.logs = []
    io.log = io.logs.append
    io.perceive_eggs = scene.perceive
    sk = RT.RealEggSkills(io)
    orig = sk.place_egg_into_slot

    def place(target_slot=None, **k):
        src = sk.plan["source_egg"]
        e = np.asarray(sk.cam[src]["rim"], float).copy()
        r = orig(target_slot=target_slot, **k)
        if r.get("ok"):
            scene.eggs = [x for x in scene.eggs if math.hypot(x[0] - e[0], x[1] - e[1]) > 0.005]
            scene.occ.add(int(sk.plan["target_slot"].split("_")[1]))
        return r
    sk.place_egg_into_slot = place
    return sk, io


class TestV468(unittest.TestCase):
    def test_f1_rise(self):
        for eggs in ((), (0,), (0, 4), (0, 1, 2, 3, 4, 5)):
            sc = SY.scene(yaw_deg=20.0, eggs_in=eggs)
            rows, _d = W.locate_egg_holder(sc["dets"], sc["rgb"], sc["depth"], sc["K"], sc["T"], table_z=0.0)
            sl = [r for r in rows if r["det_id"].startswith("slot_")]
            self.assertEqual(len(sl), 6)
            for k, (tx, ty) in enumerate(sc["truth"]):
                s = min(sl, key=lambda r: math.hypot(r["xyz_m"][0] - tx, r["xyz_m"][1] - ty))
                self.assertEqual(s["occupied"], k in eggs, (eggs, k, s["rise"]))
                self.assertEqual(s["rise"]["rise_mm"] > 0, k in eggs)

    def _verify(self, plan, looks=2):
        scene = Scene()
        sk, io = make(scene)
        sk.observe_scene()
        scene.verify_plan = plan
        r = sk.verify_egg_in_slot("slot_3", looks=looks)
        return r, io

    def test_f2_verify(self):
        r, io = self._verify([{"occ": {3}}])
        self.assertTrue(r["ok"])
        self.assertTrue(any("槽被占" in s for s in io.logs))
        r, io = self._verify([{"core_top": {3: 26.0}}])                     # middle rose 18 mm, still below +8
        self.assertTrue(r["ok"], r)
        self.assertTrue(any("比放蛋前高了 18" in s for s in io.logs))
        r, io = self._verify([{"eggs": [[-0.392, 0.140, 0.04]]}])           # egg 20 mm from the slot centre
        self.assertTrue(r["ok"])
        r, io = self._verify([{"eggs": [[-0.392, 0.150, 0.04]]}, {"occ": {3}}])   # 30 mm away, then occupied
        self.assertTrue(r["ok"])
        self.assertEqual(len(r["physical_verification"]["looks"]), 2)
        r, io = self._verify([{}, {}, {}], looks=3)
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "SLOT_EMPTY")
        self.assertEqual(len(r["physical_verification"]["looks"]), 3)
        self.assertEqual(sum(("[V4.6.8 VERIFY]" in s or "[V4.6.10 VERIFY]" in s) and "次看" in s for s in io.logs), 3)

    def test_f3_eggs_in_carton(self):
        scene = Scene()
        scene.extra_eggs = [[-0.392, 0.167, 0.045]]                         # standing in slot_6
        sk, io = make(scene)
        r = sk.observe_scene()
        self.assertEqual(r["physical_verification"]["eggs"], 3)
        self.assertEqual(r["physical_verification"]["eggs_in_carton"], 1)
        self.assertEqual(len(sk._occupied_xy()), 1)

    def test_f4_jaws(self):
        scene = Scene(occ={6})
        sk, io = make(scene)
        sk.observe_scene()
        occ = sk._occupied_xy(exclude=sk.cam["slot_3"])
        self.assertEqual(len(occ), 1)
        egg_in_G = np.array([0.0, 0.020, 0.0])
        slot3 = np.array([-0.392, 0.120, 0.05])
        toward = math.degrees(math.atan2(occ[0][1] - slot3[1], occ[0][0] - slot3[0]))
        blocked = clear = 0
        for yaw in range(-180, 180, 15):
            R = sk._slot_vertical_grasp_R(float(yaw), -90.0)
            p_a = slot3 - R @ egg_in_G
            ok = sk._jaws_clear(p_a, R, egg_in_G, occ)
            y = R[:2, 1] / np.linalg.norm(R[:2, 1])
            ang = math.degrees(math.atan2(y[1], y[0]))
            d = min(abs((ang - toward + 180) % 360 - 180), abs((ang - toward) % 360 - 180))
            if d < 20:
                self.assertFalse(ok, yaw)
                blocked += 1
            if d > 60:
                self.assertTrue(ok, yaw)
                clear += 1
        self.assertGreater(blocked, 0)
        self.assertGreater(clear, 0)
        r = sk.plan_egg_place("egg_1", "slot_3")
        self.assertTrue(r["ok"], r)
        R = sk.plan["_wps"]["release"][1]
        R_g = R @ np.linalg.inv(sk.prof.T_grasp_tcp[:3, :3])
        self.assertEqual(sk.plan["target_slot"], "slot_3")

    def test_f5_fill(self):
        scene = Scene()
        sk, io = make(scene)
        out = sk.fill_carton_episode(try_harder=False)
        self.assertEqual({k: sk.fill_result[k] for k in ("placed", "fails")}, {"placed": ["slot_6", "slot_3", "slot_5"], "fails": 0})
        self.assertEqual(sum(1 for r in out if r["tool"] == "observe_scene"), 4)     # 3 rounds + the empty table
        self.assertEqual(sum(1 for r in out if r["tool"] == "verify_egg_in_slot" and r["ok"]), 3)
        self.assertEqual(scene.eggs, [])
        self.assertTrue(RT.is_fill_all_task("Place all eggs into empty carton slots"))
        self.assertTrue(RT.is_fill_all_task("把所有鸡蛋放进蛋托"))
        self.assertFalse(RT.is_fill_all_task("Put egg_1 into slot_1"))
        # every place refused: without try harder the fill stops after the first egg
        scene = Scene()
        sk, io = make(scene)
        sk._plan_slot_candidates = lambda *a, **k: []
        sk.fill_carton_episode(try_harder=False)
        self.assertEqual({k: sk.fill_result[k] for k in ("placed", "fails")}, {"placed": [], "fails": 1})
        scene = Scene()
        sk, io = make(scene)
        sk._plan_slot_candidates = lambda *a, **k: []
        sk.fill_carton_episode(try_harder=True)
        self.assertEqual(sk.fill_result["fails"], RT.FILL_MAX_FAILS)

    def test_f6_try_harder(self):
        for th in (False, True):
            scene = Scene()
            sk, io = make(scene)
            orig_pick = sk.pick_egg
            n = {"k": 0}

            def pick(source_egg="egg_1", **k):
                n["k"] += 1
                if n["k"] == 1:
                    sk.held = None
                    return sk._res("pick_egg", False, reason="EGG_NOT_HELD", nxt=["observe_scene"])
                return orig_pick(source_egg=source_egg, **k)
            sk.pick_egg = pick
            out = sk.scripted_episode("egg_1", "slot_6", try_harder=th)
            ok = any(r["tool"] == "verify_egg_in_slot" and r["ok"] for r in out)
            self.assertEqual(ok, th, th)
            if th:
                self.assertTrue(any("夹取失败" in s and "第 2 次" in s for s in io.logs))
        # replan_place: the held egg gets a new place plan for another slot
        scene = Scene()
        sk, io = make(scene)
        sk.observe_scene()
        self.assertTrue(sk.plan_egg_place("egg_1", "slot_6")["ok"])
        sk.held, sk.stage = "egg_1", "ORIENTED"
        self.assertTrue(sk.replan_place("slot_3"))
        self.assertEqual((sk.plan["target_slot"], sk.stage), ("slot_3", "HOLDING"))
        self.assertTrue(any("[V4.6.8 TRY HARDER] 重新规划放蛋 → slot_3" in s for s in io.logs))

    def test_f7_run_execute(self):
        import NERO_V362_Cup_Skills as CK
        seen = {}

        def fake_run(agent, sk, task, model, key, effort, steps, emit=None, dry_run=False, try_harder=False):
            seen.update(task=task, steps=steps, try_harder=try_harder)
            return {}
        orig = CK.run_cup_task
        CK.run_cup_task = fake_run
        try:
            io = T45.make_io()
            RT.run_execute(io, "cup_1", "cup_2", agent=object(), llm_cfg={"task": "Stack cup_1 into cup_2",
                                                                         "max_tool_steps": 30}, try_harder=True)
        finally:
            CK.run_cup_task = orig
        self.assertTrue(seen["try_harder"])                       # V4.6.9: passed as a flag, not appended to the task
        self.assertEqual(seen["steps"], 60)
        scene = Scene()
        sk, io = make(scene)
        io.home_q = None
        import types
        RT_make = RT.make_skills
        RT.make_skills = lambda io_, k=None, s=None, log=None, scene="stack_cups": sk
        try:
            res = RT.run_execute(io, "egg_1", "slot_1", scene="fill_egg_holder", fill_all=True)
        finally:
            RT.make_skills = RT_make
        self.assertEqual(res["fill"]["placed"], ["slot_6", "slot_3", "slot_5"])
        self.assertTrue(res["ok"])

    def test_f8_labels(self):
        res = {"dets": [
            {"det_id": "slot_3", "kind": "slot_free", "contour": [[1, 1], [5, 1], [5, 5]], "height_mm": 19.0, "conf": 0.9},
            {"det_id": "egg_2", "kind": "egg", "contour": [[1, 1], [5, 1], [5, 5]], "height_mm": 42.0, "conf": 0.8},
            {"det_id": "carton", "kind": "carton", "rows": 2, "cols": 3, "contour": [[1, 1], [5, 1], [5, 5]],
             "conf": 0.9}]}
        labs = [(d["label"], d["small"]) for d in RT.overlay_dets(res)]
        self.assertEqual(labs, [("3", True), ("e2", True), ("蛋托 2×3", False)])
        src = open(os.path.join(HERE, "NERO_V365_Real_Cup_Task.py"), encoding="utf-8").read()
        self.assertIn('if str(name).startswith(("slot_", "egg_")):', src)

    def test_f9_wiring(self):
        src = open(os.path.join(HERE, "NERO_V365_Real_Cup_Task.py"), encoding="utf-8").read()
        i_lab = src.index('ttk.Label(r4, text="速度").pack(side="right", padx=(8, 2))')
        i_chk = src.index('ttk.Checkbutton(r4, text="Try harder", variable=self.v_try_harder).pack(side="right"')
        self.assertLess(i_lab, i_chk)                          # packed after "速度" on the right = left of it
        self.assertIn('try_harder=bool(self.v_try_harder.get()) if hasattr(self, "v_try_harder") else False', src)
        con = open(os.path.join(HERE, "NERO_V312_Console.py"), encoding="utf-8").read()
        self.assertIn("cam_canvas = tk.Canvas(cam_outer, highlightthickness=0)", con)
        self.assertIn("cam_main = ttk.Frame(cam_page)", con)
        self.assertIn('cam_canvas.bind_all(seq, lambda e: _cam_wheel(e) if str(e.widget).startswith(str(cam_outer))', con)
        vc = open(os.path.join(HERE, "NERO_V464_Vision_Check.py"), encoding="utf-8").read()
        self.assertIn("self.v_refresh = tk.BooleanVar(value=True)", vc)
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        v = open(os.path.join(HERE, "VERSION")).read().strip()
        self.assertEqual((C.VERSION, RR.VERSION), (v, v))
        self.assertIn(v, ("V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3"))
        self.assertEqual(RT.MARKER_V468, "V4.6.8 FILL CARTON + VERIFY BY RISE + TRY HARDER")


if __name__ == "__main__":
    unittest.main(verbosity=1)
