#!/usr/bin/env python3
"""NERO V4.8.3 offline tests: tidy desk ("听懂弦外之音的桌面整理") on the sort skills with the NERO original gripper.

T1 layout: tidy_desk = a variant of sort_items (official gripper); set_sort_layout switches the item / container
   tables in place and back (the V3.72 sort layout unchanged); 7 items <= 62 mm across, 4 containers on the table,
   footprints apart, every container inside the table; the ids / tool enums follow (sync_layout).
T2 collision boxes of the original gripper (housing / fingers / bracket) in the sort pre-check; SO-101 default kept.
T3 symmetric grasp planner (no MuJoCo, a pose-only fake IO): every item to tidy has a grasp from the home posture
   (fingers clear of its neighbours, gap <= 70 mm stroke, fingertips >= 4 mm above the table); too wide -> refused.
T4 intents: guests / work / kids / snack / trash / leave / general from everyday sentences, the rule plan per
   intent (what moves, what stays and why, what is asked), explicit words win ("别动饼干", "把饼干放进水果盘").
T5 validation: in-use mouse stays, capacity overflow -> question, unknown container -> question, already there -> keep.
T6 preferences: answers are remembered (per intent + general) and answer the next run's questions.
T7 LLM intent: a fake provider (JSON in a code fence) drives the plan, a forgotten item is filled by the rules,
   a broken reply falls back to the rules.
T8 runner: a blocked grasp is retried after the others (from the home posture), a refused place puts the item back
   on the table; the report names what went where / stayed / failed.
T9 real adapter: container positions file (template written / overrides read), mask yaw on the table plane,
   detections mapped to the items (aliases), the gripper width decides "held"; real tab knows scene tidy_desk.
T10 versions / markers.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ["NERO_CALIB_DIR"] = tempfile.mkdtemp(prefix="nero_v483_calib_")
os.environ.pop("NERO_GRIPPER", None)
os.environ.pop("NERO_SORT_LAYOUT", None)

import NERO_V343_Scenes as SC                  # noqa: E402
import NERO_V362_Cup_Skills as CK              # noqa: E402
import NERO_V372_Sort_Skills as SK             # noqa: E402
import NERO_V375_Collision_Check as CC         # noqa: E402
import NERO_V483_Tidy as TD                    # noqa: E402


def quat_yaw(deg):
    a = math.radians(deg) / 2.0
    return [math.cos(a), 0.0, 0.0, math.sin(a)]


class PoseIO:
    """poses only (the layout positions), the official gripper profile - enough for the planners"""
    name = "pose-only"

    def __init__(self, prof=None):
        self.prof = prof or CK.OFFICIAL_PROFILE
        self.logs = []
        self.poses = {n: (it["xy"][0], it["xy"][1], it["yaw_deg"]) for n, it in SC.SORT_ITEMS.items()}

    def gripper_profile(self):
        return self.prof

    def emit(self, s):
        self.logs.append(s)

    def object_pose(self, body):
        x, y, yaw = self.poses[body]
        return [x, y, SC.SORT_ITEMS[body]["height"] + 0.0005], quat_yaw(yaw)

    def gripper_state(self):
        return {"opening_mm": 60.0, "hold_active": False, "contact_left": False, "contact_right": False}

    def current_q(self):
        return [0.0] * 7

    def tcp_pose(self):
        T = CK.FastArm().fk(np.zeros(7))
        return [float(v) for v in T[:3, 3]], CK.R_to_quat_xyzw(T[:3, :3])

    def state_check(self, q, opening_mm=None, extra_parts=None):
        return CC.assess(None, q, opening_mm, CC.world_boxes("sort_items"), extra_parts=extra_parts,
                         gripper="official")


def tidy():
    SC.set_sort_layout("tidy")
    SK.sync_layout()


def desk():
    tidy()
    sk = SK.SortSkills(PoseIO())
    return TD.desk_from_scene(sk.scene(), stroke_mm=sk.stroke_mm), sk


def plan_for(text, prefs=None, llm=None, post=None):
    d, _sk = desk()
    p = TD.understand(text, d, prefs or {"rules": {}, "log": []}, llm, post=post, log=lambda s: None)
    return p, d, {st["label"]: st for st in p["steps"]}


class TestV483(unittest.TestCase):
    def tearDown(self):
        SC.set_sort_layout("sort")
        SK.sync_layout()

    def test_t1_layout(self):
        self.assertEqual(SC.SCENES["tidy_desk"]["base"], "sort_items")
        self.assertNotIn("gripper", SC.SCENES["tidy_desk"])               # -> the official gripper
        old = {k: dict(v) for k, v in SC.SORT_ITEMS.items()}
        self.assertEqual(SC.sort_layout(), "sort")
        self.assertEqual(SC.set_sort_layout("tidy"), "sort")
        SK.sync_layout()
        self.assertEqual(len(SC.SORT_ITEMS), 7)
        self.assertEqual(set(SC.SORT_BINS), {"fruit_bowl", "trash_bin", "pen_box", "toy_box"})
        self.assertEqual(len(SK.ITEM_IDS), 7)
        self.assertEqual(SK.SORT_TOOLS[2]["parameters"]["properties"]["box"]["enum"], list(SC.SORT_BINS))
        os.environ["NERO_V343_ACTIVE_SCENE"] = "tidy_desk"
        try:
            self.assertEqual(SC.active_scene(), "sort_items")                # the backend's sort_items branches run
            self.assertEqual(SC.hold_object_names()[:7], list(SC.SORT_ITEM_NAMES))
            self.assertEqual(SC._layout_from_env(), "tidy")
        finally:
            os.environ.pop("NERO_V343_ACTIVE_SCENE", None)
        foot = []
        for n, it in SC.SORT_ITEMS.items():
            self.assertLessEqual(it["width"], 0.0625, n)
            self.assertTrue(it.get("cn"))
            foot.append((n, SC.sort_item_footprint(n, it["xy"], it["yaw_deg"])))
        for i, (a, fa) in enumerate(foot):
            for b, fb in foot[i + 1:]:
                self.assertGreater(CC.rect_dist(fa, fb), 0.005, (a, b))
            for bn, c, h, _r in SC.sort_bin_boxes():
                self.assertGreater(CC.rect_dist(fa, (c[:2], h[:2], 0.0)), 0.005, (a, bn))
        boxes = [(k, (b["centre"], (b["inner"][0] / 2 + b["wall_t"], b["inner"][1] / 2 + b["wall_t"]), 0.0))
                 for k, b in SC.SORT_BINS.items()]
        for i, (a, ra) in enumerate(boxes):
            self.assertLess(abs(ra[0][1]) + ra[1][1], 0.38)                   # on the table (y +-0.38)
            for b, rb in boxes[i + 1:]:
                self.assertGreater(CC.rect_dist(ra, rb), 0.005, (a, b))
        xml = SC.sort_world_xml(bare=True)
        for n in SC.SORT_ITEM_NAMES:
            self.assertIn(f'<body name="{n}"', xml)
        self.assertEqual(SK.default_box(SC.SORT_ITEMS["obj_paper"]), "trash_bin")
        self.assertIsNone(SK.default_box(SC.SORT_ITEMS["obj_mouse"]))
        SC.set_sort_layout("sort")
        SK.sync_layout()
        self.assertEqual(set(SC.SORT_BINS), {"red", "blue"})
        self.assertEqual(SC.SORT_ITEMS["obj_apple"]["width"], 0.074)
        self.assertEqual({k: dict(v) for k, v in SC.SORT_ITEMS.items()}, old)
        self.assertEqual(SK.BOXES, ("red", "blue"))
        import NERO_V312_Console as C
        self.assertIn("tidy_desk", [k for k, _ in C.SIM_SCENES])
        with open(os.path.join(HERE, "start_nero_sim.sh"), encoding="utf-8") as f:
            self.assertIn("tidy_desk)", f.read())

    def test_t2_collision_boxes(self):
        q = [0.0, 0.4, 0.0, 1.2, 0.0, 0.3, 0.0]
        names = [p[0] for p in CC.gripper_parts(q, 50.0, "official")]
        for n in ("agx_housing", "agx_finger_a", "agx_finger_b", "link7"):
            self.assertIn(n, names)
        so = [p[0] for p in CC.gripper_parts(q, 50.0)]
        self.assertIn("so101_fixed_pad", so)
        cam = np.eye(4)
        cam[:3, 3] = [0.1221, -0.0537, -0.0250]
        self.assertIn("agx_bracket", [p[0] for p in CC.gripper_parts(q, 50.0, "agx_gripper", T_flange_cam=cam)])
        a = CC.assess(None, q, 50.0, CC.world_boxes("sort_items"), gripper="official")
        self.assertIn("ok", a)
        # MoveIt's gripper links block with the original gripper (they ARE the mounted one)
        core = types.SimpleNamespace(v375_state_validity=lambda q_: {"available": True, "valid": False,
                                                                       "pairs": [("gripper_link1", "mujoco_play_table")]})
        self.assertFalse(CC.assess(core, q, 50.0, gripper="official")["ok"])
        self.assertTrue(CC.assess(core, [0.0] * 7, 50.0)["model_only"])

    def test_t3_grasp_planner(self):
        d, sk = desk()
        self.assertTrue(sk.sym)
        self.assertEqual(sk.stroke_mm, 70.0)
        q0 = [0.0] * 7
        for body in SC.SORT_ITEM_NAMES:
            sp = SC.SORT_ITEMS[body]
            box = SK.default_box(sp) or ("fruit_bowl" if sp["category"] == "snack" else None)
            if box is None:
                continue
            iid = SK.BODY_ID[body]
            pl = sk._plan_pick(sk._item(iid), q0, box)
            self.assertIsNotNone(pl, (body, sk.v375_rejects[:3]))
            self.assertLessEqual(pl["gap_mm"], 70.0)
            self.assertGreaterEqual(pl["jaw_clear_mm"], SK.SYM_JAW_CLEAR_M * 1000 - 1e-6)
            self.assertGreaterEqual(pl["zg"] - sk.tip_drop_m, 0.004 - 1e-9)
            self.assertGreaterEqual(pl["margin"], CK.MIN_MARGIN_RELAXED_DEG)
        self.assertEqual(sk._cmd(80.0), 70.0)
        self.assertEqual(sk._cmd(44.0), 44.0)
        big = dict(SC.SORT_ITEMS["obj_apple"], width=0.068)
        SC.SORT_ITEMS["obj_apple"] = big
        try:
            r = sk.pick_object(item_id=SK.BODY_ID["obj_apple"], box="fruit_bowl")
        finally:
            SC.set_sort_layout("sort")
            SC.TIDY_ITEMS["obj_apple"]["width"] = 0.062
            tidy()
        self.assertEqual(r["reason"], "ITEM_TOO_WIDE")

    def test_t4_intents(self):
        self.assertEqual(TD.detect_intent("我妈妈一会儿要来"), "guests")
        self.assertEqual(TD.detect_intent("等下要开会，帮我腾个地方"), "work")
        self.assertEqual(TD.detect_intent("孩子一会儿来玩"), "kids")
        self.assertEqual(TD.detect_intent("有点饿了"), "snack")
        self.assertEqual(TD.detect_intent("把垃圾扔了，别的别动"), "trash")
        self.assertEqual(TD.detect_intent("我下班了"), "leave")
        self.assertEqual(TD.detect_intent("桌子好乱"), "general")
        p, d, by = plan_for("我妈妈一会儿要来")
        self.assertEqual(by["banana"]["box"], "fruit_bowl")
        self.assertEqual(by["marker pen"]["box"], "pen_box")
        self.assertEqual(by["toy block"]["box"], "toy_box")
        self.assertEqual(by["computer mouse"]["action"], "keep")
        self.assertEqual(by["cookie pack"]["action"], "ask")
        self.assertEqual(by["crumpled paper"]["action"], "ask")
        self.assertIn("trash_bin", by["crumpled paper"]["options"])
        p, d, by = plan_for("等下要开会，帮我腾个地方")
        self.assertEqual(by["marker pen"]["action"], "keep")                  # pens may be needed
        self.assertEqual(by["cookie pack"]["action"], "keep")
        p, d, by = plan_for("孩子一会儿来玩")
        self.assertEqual(by["toy block"]["action"], "keep")                   # left out to play with
        self.assertEqual(by["marker pen"]["box"], "pen_box")
        self.assertIn("乱画", by["marker pen"]["reason"])
        p, d, by = plan_for("有点饿了")
        self.assertEqual((by["banana"]["action"], by["apple"]["action"]), ("keep", "keep"))
        p, d, by = plan_for("把垃圾扔了，别的别动")
        self.assertEqual(by["crumpled paper"]["box"], "trash_bin")
        self.assertTrue(all(st["action"] == "keep" for k, st in by.items() if k != "crumpled paper"))
        p, d, by = plan_for("我妈妈要来了，饼干别动，纸团扔了")
        self.assertEqual(by["cookie pack"]["action"], "keep")
        self.assertTrue(by["cookie pack"]["explicit"])
        self.assertEqual(by["crumpled paper"]["box"], "trash_bin")
        p, d, by = plan_for("收拾一下，把饼干放进水果盘")
        self.assertEqual(by["cookie pack"]["box"], "fruit_bowl")
        self.assertEqual(by["apple"]["action"], "ask")                        # the bowl holds 2: your cookies first
        p, d, by = plan_for("收拾一下，马克笔留着")
        self.assertEqual(by["marker pen"]["action"], "keep")
        self.assertIn("你说了", by["marker pen"]["reason"])
        txt = TD.plan_text(p, d)
        self.assertIn("我理解", txt)

    def test_t5_validation(self):
        d, _sk = desk()
        base = TD.rule_plan("收拾一下", d)
        for st in base["steps"]:
            if st["label"] == "computer mouse":
                st.update(action="move", box="toy_box")
            if st["label"] == "toy block":
                st.update(action="move", box="drawer")
        d["boxes"] = [dict(b, capacity=1) if b["box"] == "fruit_bowl" else b for b in d["boxes"]]
        v = TD.validate(base, d, {"rules": {}})
        by = {st["label"]: st for st in v["steps"]}
        self.assertEqual(by["computer mouse"]["action"], "keep")
        self.assertEqual(by["toy block"]["action"], "ask")
        self.assertEqual(by["banana"]["box"], "fruit_bowl")
        self.assertEqual(by["apple"]["action"], "ask")                        # bowl full after the banana
        self.assertIn("最多放 1 件", by["apple"]["question"])
        self.assertTrue(any("drawer" in w for w in v["warnings"]))
        d2, _ = desk()
        d2["items"] = [dict(it, now="fruit_bowl") if it["label"] == "banana" else it for it in d2["items"]]
        v2 = TD.validate(TD.rule_plan("收拾一下", d2), d2, {"rules": {}})
        self.assertIn("已经在", {st["label"]: st for st in v2["steps"]}["banana"]["reason"])

    def test_t6_preferences(self):
        path = os.path.join(os.environ["NERO_CALIB_DIR"], "prefs_t6.json")
        prefs = TD.load_prefs(path)
        p, d, by = plan_for("我妈妈一会儿要来", prefs=prefs)
        dec = TD.initial_decisions(p)
        self.assertIsNone(dec[by["cookie pack"]["item_id"]])
        dec[by["cookie pack"]["item_id"]] = "keep"
        dec[by["crumpled paper"]["item_id"]] = "trash_bin"
        stored = TD.remember(prefs, p, dec)
        self.assertEqual(sorted(stored), [("cookie pack", "keep"), ("crumpled paper", "trash_bin")])
        TD.save_prefs(prefs, path)
        prefs2 = TD.load_prefs(path)
        p2, d2, by2 = plan_for("我妈妈一会儿要来", prefs=prefs2)
        self.assertEqual(by2["cookie pack"]["action"], "keep")
        self.assertEqual(by2["cookie pack"]["pref"], "intent")
        self.assertIn("习惯", by2["cookie pack"]["reason"])
        self.assertEqual(by2["crumpled paper"]["box"], "trash_bin")
        p3, d3, by3 = plan_for("桌子好乱", prefs=prefs2)                      # another intent: the general rule answers
        self.assertEqual(by3["crumpled paper"]["box"], "trash_bin")
        self.assertEqual(by3["crumpled paper"]["pref"], "general")
        self.assertTrue(TD.prefs_path().startswith(os.environ["NERO_CALIB_DIR"]))

    def test_t7_llm(self):
        d, _sk = desk()
        ids = {it["label"]: it["item_id"] for it in d["items"]}
        reply = {"intent": "guests", "understanding": "妈妈要来，桌面要体面", "steps": [
            {"item_id": ids["banana"], "action": "move", "box": "fruit_bowl", "reason": "水果进果盘"},
            {"item_id": ids["computer mouse"], "action": "move", "box": "toy_box", "reason": "x"},
            {"item_id": ids["cookie pack"], "action": "ask", "question": "饼干要招待妈妈吗？",
             "options": ["fruit_bowl", "keep"]}]}
        calls = []

        def post(prov, model, key, system, user):
            calls.append(json.loads(user))
            return "```json\n" + json.dumps(reply, ensure_ascii=False) + "\n```"
        p, d, by = plan_for("我妈妈一会儿要来", llm={"model": "fake-model"}, post=post)
        self.assertEqual(p["source"], "llm")
        self.assertEqual(p["understanding"], "妈妈要来，桌面要体面")
        self.assertEqual(by["computer mouse"]["action"], "keep")              # in use: not moved by the LLM
        self.assertEqual(by["cookie pack"]["question"], "饼干要招待妈妈吗？")
        self.assertIn("按规则补上", by["marker pen"]["reason"])               # forgotten by the LLM
        self.assertEqual(len(calls[0]["desk_items"]), 7)
        p2, _d, _by = plan_for("我妈妈一会儿要来", llm={"model": "m"}, post=lambda *a: "sorry, no JSON")
        self.assertEqual(p2["source"], "rules")

    def test_t8_runner_report(self):
        d, _sk = desk()
        p = TD.validate(TD.rule_plan("我妈妈一会儿要来", d), d, {"rules": {}})
        dec = {st["item_id"]: (st["box"] if st["action"] == "move" else "keep") for st in p["steps"]}
        ids = {st["label"]: st["item_id"] for st in p["steps"]}
        where = {iid: None for iid in dec}

        class FakeSk:
            held = None
            calls = []
            blocked = {ids["marker pen"]: 1}

            def run_tool(self, name, args):
                self.calls.append((name, dict(args)))
                if name == "pick_object":
                    iid = args["item_id"]
                    if self.blocked.get(iid):
                        self.blocked[iid] -= 1
                        return {"ok": False, "reason": "GRASP_UNREACHABLE"}
                    self.held = iid
                    return {"ok": True}
                if name == "place_in_box":
                    iid, self.held = self.held, None
                    if iid == ids["toy block"]:
                        self.held = iid
                        return {"ok": False, "reason": "PLACE_UNREACHABLE"}
                    where[iid] = args["box"]
                    return {"ok": True}
                if name == "return_item_to_table":
                    self.held = None
                    return {"ok": True}
                if name == "check_bins":
                    return {"ok": True, "physical_verification": {"items": {k: {"in_bin": v} for k, v in where.items()}}}
                return {"ok": True}
        sk = FakeSk()
        rep = TD.TidyRunner(sk, log=lambda s: None).run(p, dec)
        names = [c[0] for c in sk.calls]
        self.assertIn("return_item_to_table", names)
        i_safe = names.index("return_to_safe_pose")
        self.assertEqual(sk.calls[i_safe + 2], ("pick_object", {"item_id": ids["marker pen"], "box": "pen_box"}))
        self.assertIn(ids["marker pen"], rep["done"])
        self.assertEqual([f["item"] for f in rep["failed"]], [ids["toy block"]])
        self.assertFalse(rep["ok"])
        for key in ("水果盘：香蕉、苹果", "留在原处：鼠标", "没放好：积木", "容器里找不到", "桌上现在"):
            self.assertIn(key, rep["text"])
        self.assertTrue(TD.save_report(rep, p, dec))

    def test_t9_real_adapter(self):
        tidy()
        path = TD.real_layout_path()
        if os.path.exists(path):
            os.remove(path)
        note = TD.apply_real_layout(log=lambda s: None)
        self.assertIn("模板", note)
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["containers"]["toy_box"]["centre"] = [-0.40, 0.30]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        TD.apply_real_layout(log=lambda s: None)
        self.assertEqual(SC.SORT_BINS["toy_box"]["centre"], (-0.40, 0.30))
        SC.set_sort_layout("sort")
        SC.TIDY_BINS["toy_box"] = dict(SC.TIDY_BINS["toy_box"], centre=(-0.39, 0.29))
        os.remove(path)
        # mask yaw: a 130 x 21 mm marker at yaw 30 deg seen from straight above
        K = {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0}
        T = np.eye(4)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = [-0.40, 0.0, 0.80]
        pts = []
        a = math.radians(30.0)
        for s_ in np.linspace(-0.065, 0.065, 9):
            for t in (-0.0105, 0.0105):
                x = -0.40 + s_ * math.cos(a) - t * math.sin(a)
                y = 0.0 + s_ * math.sin(a) + t * math.cos(a)
                pc = np.linalg.inv(T) @ np.array([x, y, 0.021, 1.0])
                pts.append([pc[0] / pc[2] * K["fx"] + K["ppx"], pc[1] / pc[2] * K["fy"] + K["ppy"]])
        yaw = TD.yaw_from_contour(pts, K, T, 0.021)
        self.assertAlmostEqual((yaw - 30.0 + 90.0) % 180.0 - 90.0, 0.0, delta=1.0)
        self.assertIsNone(TD.yaw_from_contour([[300, 200], [340, 200], [340, 240], [300, 240], [320, 220]], K, T, 0.0))
        # detections -> items, gripper width -> held
        import NERO_V48_Gripper as GG
        GG.CFG_PATH = os.path.join(os.environ["NERO_CALIB_DIR"], "gripper.json")
        c = GG.load_cfg()
        c["type"] = "agx_gripper"
        GG.save_cfg(c)
        import NERO_V365_Real_Cup_Task as RT

        class RIO(PoseIO):
            def __init__(self):
                super().__init__(RT.real_profile(CK))
                self.gap = 40.0

            def perceive_sort_items(self, tag):
                return {"table_z_m": -0.009, "items": [
                    {"kind": "banana", "rim_xyz_m": [-0.43, -0.095, 0.02], "conf": 0.8},
                    {"kind": "mouse", "rim_xyz_m": [-0.57, 0.0, 0.02], "conf": 0.7},
                    {"kind": "paper ball", "rim_xyz_m": [-0.46, 0.0, 0.02], "conf": 0.6},
                    {"kind": "chair", "rim_xyz_m": [-0.3, 0.0, 0.02], "conf": 0.9}]}

            def gripper_state(self):
                return {"gap_mm": self.gap, "opening_mm": self.gap}
        io = RIO()
        sk = TD.make_real_tidy_skills(io, log=lambda s: None)
        self.assertEqual(sk.prof.name, "agx_gripper")
        r = sk.observe_scene()
        self.assertTrue(r["ok"])
        labels = sorted(i["label"] for i in r["scene"]["items"])
        self.assertEqual(labels, ["banana", "computer mouse", "crumpled paper"])
        self.assertTrue(any("没看到" in w for w in r.get("warnings", [])))
        self.assertAlmostEqual(float(sk._item(SK.BODY_ID["obj_banana"])["c"][2]), -0.009 + 0.017, places=4)
        self.assertIsNotNone(io.state_check)
        m = sk._measure_after_lift(SK.BODY_ID["obj_paper"], 0.0, {})
        self.assertTrue(m["held"])
        io.gap = 2.0
        self.assertFalse(sk._measure_after_lift(SK.BODY_ID["obj_paper"], 0.0, {})["held"])
        self.assertIn(("tidy_desk", "④ 桌面整理 tidy_desk（听懂弦外之音 · 原装夹爪）"), RT.TASK_SCENES)
        self.assertIn("crumpled paper", RT.SCENE_DEFAULT_PROMPTS["tidy_desk"])
        with open(os.path.join(HERE, "NERO_V365_Real_Cup_Task.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn("out = self._run_tidy(io, task, mode, llm if tidy_llm else None)", src)
        self.assertIn('scene == "tidy_desk" and not self.GG.is_agx()', src)
        c["type"] = "so101"
        GG.save_cfg(c)

    def test_t10_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual((v, C.VERSION, RR.VERSION), ("V4.8.3", "V4.8.3", "V4.8.3"))
        self.assertEqual(TD.MARKER, "V4.8.3 TIDY DESK INTENT LAYER")


if __name__ == "__main__":
    unittest.main(verbosity=1)
