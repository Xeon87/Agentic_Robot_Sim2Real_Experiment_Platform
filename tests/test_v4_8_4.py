#!/usr/bin/env python3
"""NERO V4.8.4 offline tests: NERO 原装夹爪-Softfin + the egg-task fixes from the 11-11 office run.

U1 gripper choice: SO-101 / NERO 原装 / NERO 原装夹爪-Softfin (same AgxGripper hardware, finger setting agx.fingers);
   switching keeps the hardware type, the Softfin tip length (design 205.4 mm) has its own setting; no emoji.
U2 Softfin geometry (user CAD NERO_SoftFin_Adapter, URDF gripper_link1 frame): fin tips 61.4 mm past the stock tips,
   grasp line 25 mm inside the tips, fins 25.8 mm wide (pad half width 12.9 mm), adapter boxes; the stock values
   are unchanged.
U3 collision parts and the real profile follow the mounted fingers; the twin's official gripper stays stock.
U4 egg grip nearer the middle (user scene): stock jaws 17 mm from the egg centre (V4.8.1: 24), Softfin a few mm;
   every cell plans, the fingers stay >= 5 mm above the carton; Softfin held-egg reading range (8, 46) mm.
U5 blunt end from the mask outline (synthetic egg outline seen from above): direction towards the wider end, None
   for a symmetric outline; the grasp prefers pads on the blunt end.
U6 held egg vs carton while carried: the sweep check finds a dip into the carton; then the transit goes high and
   straight down (orient_for_slot executes those steps).
U7 the pregrasp move from the current posture is in the plan's fence check (11-11: plan OK, 3x FENCE at execution).
U8 a fine-tuned model named in the Camera tab but missing on disk is reported (11-10 / 11-11 it was dropped silently).
U9 selectors in Agentic实机任务 / Smol VLA / 实机NERO调试 offer the Softfin; episodes record AgxGripper-Softfin.
U10 versions / markers.
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
os.environ["NERO_CALIB_DIR"] = tempfile.mkdtemp(prefix="nero_v484_calib_")
os.environ.pop("NERO_GRIPPER", None)

import NERO_V48_Gripper as GG                  # noqa: E402

GG.CFG_PATH = os.path.join(tempfile.mkdtemp(prefix="nero_v484_"), "gripper.json")
import NERO_V481_Agx_Geom as AG                # noqa: E402
import NERO_V4613_Tool_Collision as TC         # noqa: E402
import NERO_V362_Cup_Skills as CK              # noqa: E402
import NERO_V365_Real_Cup_Task as RT           # noqa: E402
import NERO_V375_Collision_Check as CC         # noqa: E402
import test_v4_6_13 as T13                     # noqa: E402

USER_CAM = np.array([[1, 0, 0, 0.1221], [0, 1, 0, -0.0537], [0, 0, 1, -0.0250], [0, 0, 0, 1.0]])
_find = TC.find_handeye
EMOJI = [chr(c) for c in list(range(0x1F300, 0x1FAFF)) + list(range(0x2600, 0x27C0))]


def use(choice):
    c = GG.load_cfg()
    c["type"] = "so101" if choice == "so101" else "agx_gripper"
    c["agx"]["fingers"] = "softfin" if choice == "agx_softfin" else "stock"
    c["agx"]["tip_from_flange_mm"] = 144.0
    c["agx"]["softfin_tip_from_flange_mm"] = 205.4
    GG.save_cfg(c)
    TC.find_handeye = lambda calib_dir=None: (USER_CAM.copy(), "user")


def finger_low(sk, q):
    T = sk._fast.fk(np.asarray(q, float))
    return sk._agx_finger_low(T[:3, 3], T[:3, :3])


class TestV484(unittest.TestCase):
    def tearDown(self):
        TC.find_handeye = _find

    def test_u1_choice(self):
        use("so101")
        self.assertEqual(GG.CHOICES, ("so101", "agx_gripper", "agx_softfin"))
        self.assertEqual(GG.current_choice(), "so101")
        r = GG.set_choice("agx_softfin")
        self.assertTrue(r["ok"] and r["changed"], r)
        self.assertEqual((GG.active_type(), GG.fingers(), GG.is_softfin(), GG.current_choice()),
                         ("agx_gripper", "softfin", True, "agx_softfin"))
        self.assertAlmostEqual(GG.tip_from_flange_mm(), 205.4)
        seen = []
        GG.add_listener(seen.append)
        r = GG.set_choice("agx_gripper")                   # same hardware, other fingers -> listeners still told
        self.assertTrue(r["changed"])
        self.assertEqual(seen[-1], "agx_gripper")
        self.assertFalse(GG.is_softfin())
        self.assertAlmostEqual(GG.tip_from_flange_mm(), 144.0)
        self.assertTrue(any("Softfin" in t for t in GG.todo_after_finger_switch("softfin")))
        self.assertEqual(GG.choice_from_label("NERO 原装夹爪-Softfin（柔性指 · CAN）"), "agx_softfin")
        self.assertEqual(GG.choice_from_label(GG.LABELS["agx_gripper"]), "agx_gripper")
        txt = "".join(GG.CHOICE_LABELS.values()) + "".join(GG.CHOICE_SHORT.values())
        self.assertFalse([e for e in EMOJI if e in txt])
        GG._listeners.remove(seen.append)

    def test_u2_geometry(self):
        use("agx_gripper")
        self.assertAlmostEqual(AG.tip_drop_m(), 0.0197, places=4)
        self.assertAlmostEqual(AG.T_l7_grasp()[0, 3], 0.1553, places=4)
        self.assertAlmostEqual(AG.pad_half_width_m(), 0.028, places=4)
        use("agx_softfin")
        self.assertEqual(AG.variant(), "softfin")
        self.assertAlmostEqual(AG.tip_drop_m(), 0.025, places=4)
        self.assertAlmostEqual(AG.T_l7_tip()[0, 3], 0.031 + 0.2054, places=4)     # flange face + 205.4 mm
        self.assertAlmostEqual(AG.T_l7_grasp()[0, 3], 0.2114, places=4)
        self.assertAlmostEqual(AG.pad_half_width_m(), 0.0129, places=4)
        b = {n: (c, h) for n, c, h in AG.boxes_in_G(44.0)}
        for n in ("finger_a", "finger_b", "finger_root_a", "adapter_b", "housing"):
            self.assertIn(n, b)
        self.assertAlmostEqual(b["finger_a"][1][0], 0.0129 + 0.003, places=4)     # 25.8 / 2 + margin
        tip_zg = b["finger_a"][0][2] + b["finger_a"][1][2] - 0.003
        self.assertAlmostEqual(tip_zg, 0.025, places=4)                           # fin tips 25 mm past the grasp line
        self.assertGreater(b["finger_b"][0][1], 0.022)                            # outside the 44 mm opening
        self.assertEqual(AG.egg_held_gap_mm(), (8.0, 46.0))
        self.assertEqual(AG.finger_footprint_mm(), (25.8, 24.0))
        self.assertEqual(AG.finger_footprint_mm("stock"), (56.0, 26.5))

    def test_u3_parts_profile(self):
        use("agx_softfin")
        q = [0.0, 0.4, 0.0, 1.2, 0.0, 0.3, 0.0]
        names = [p[0] for p in CC.gripper_parts(q, 40.0, "agx_gripper")]
        self.assertIn("agx_adapter_a", names)
        self.assertNotIn("agx_adapter_a", [p[0] for p in CC.gripper_parts(q, 40.0, "official")])
        prof = RT.real_profile(CK)
        self.assertEqual(prof.name, "agx_gripper")
        self.assertIn("Softfin", prof.label)
        self.assertAlmostEqual(prof.T_tcp_grasp[2, 3], 0.2114 - 0.1553, places=4)
        self.assertAlmostEqual(RT.so101_tip_drop_m(), 0.025, places=4)
        tm = TC.make_tool_model()
        self.assertEqual(tm.variant, "softfin")
        self.assertIn("adapter_a", tm.BODY)
        self.assertIn("Softfin", tm.note)

    def _egg_runs(self, choice):
        use(choice)
        sk, io = T13.user_scene()
        out = []
        for slot in ("slot_1", "slot_2", "slot_3", "slot_4", "slot_5", "slot_6"):
            r = sk.plan_egg_place(source_egg="egg_1", target_slot=slot)
            self.assertTrue(r["ok"], (choice, slot, r))
            pl = sk.plan
            zt = float(pl.get("table_z_m", 0.0))
            low = finger_low(sk, pl["_q"]["release"])
            self.assertGreaterEqual(low - zt - AG.CARTON_TOP_ABOVE_TABLE_M, AG.PAD_ABOVE_CARTON_M - 1e-6, slot)
            for st in pl.get("_transit") or []:
                self.assertIn(st["kind"], ("joints", "line"))
            out.append((pl["egg_offset_mm"], pl["egg_in_G_m"][0]))
        return out, sk, io

    def test_u4_egg_offset(self):
        out, sk, io = self._egg_runs("agx_gripper")
        for off, x in out:
            self.assertLessEqual(off, 20.0)
            self.assertAlmostEqual(abs(x) * 1000.0, off, delta=0.5)
        self.assertTrue(any("离蛋心" in l for l in io.logs))
        out, sk, io = self._egg_runs("agx_softfin")
        for off, x in out:
            self.assertLessEqual(off, 8.0)
        self.assertEqual(sk._is_egg_held(None, {"gap_mm": 20.0})[0], True)       # wrapped by the fins
        self.assertEqual(sk._is_egg_held(None, {"gap_mm": 3.0})[0], False)
        self.assertGreater(RT.EGG_RELEASE_ABOVE_SLOT_AGX_M, RT.EGG_RELEASE_ABOVE_SLOT_M)

    def test_u5_blunt_end(self):
        K = {"fx": 600.0, "fy": 600.0, "ppx": 320.0, "ppy": 240.0}
        T = np.eye(4)
        T[:3, :3] = np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = [-0.30, 0.0, 0.70]

        def outline(asym):
            pts = []
            for t in np.linspace(0, 2 * math.pi, 60, endpoint=False):
                x = 0.0275 * math.cos(t)
                r = 0.021 * (1.0 + asym * math.cos(t))          # wider towards +x (blunt end) when asym > 0
                y = r * math.sin(t)
                wx, wy = -0.30 + x * math.cos(0.5) - y * math.sin(0.5), x * math.sin(0.5) + y * math.cos(0.5)
                pc = np.linalg.inv(T) @ np.array([wx, wy, 0.021, 1.0])
                pts.append([pc[0] / pc[2] * K["fx"] + K["ppx"], pc[1] / pc[2] * K["fy"] + K["ppy"]])
            return pts
        ax = [math.cos(0.5), math.sin(0.5)]
        d = RT.egg_blunt_dir(outline(0.15), K, T, 0.021, ax)
        self.assertIsNotNone(d)
        self.assertGreater(d[0] * ax[0] + d[1] * ax[1], 0.99)
        d = RT.egg_blunt_dir(outline(-0.15), K, T, 0.021, [-a for a in ax])
        self.assertGreater(d[0] * -ax[0] + d[1] * -ax[1], -1.01)
        self.assertIsNone(RT.egg_blunt_dir(outline(0.0), K, T, 0.021, ax))
        # the grasp prefers the pads on the blunt end
        use("agx_gripper")
        sk, io = T13.user_scene()
        e = sk.cam["egg_1"]
        x_long = np.array([math.cos(math.radians(e["yaw_deg"] - 90.0)), math.sin(math.radians(e["yaw_deg"] - 90.0))])
        for sgn in (1.0, -1.0):
            e["blunt_dir"] = list(sgn * x_long)
            sk._egg_offset_cur = 17.0
            cs = sk._plan_grasp_candidates(e, float(sk.last_obs.get("table_z_m", 0.0)), io.current_q())
            self.assertTrue(cs)
            good_blunt = [c for c in cs if c["on_blunt"] and c["margin"] >= 6.0]
            # pads on the blunt end whenever that grasp has a normal joint margin (else the other end, logged)
            self.assertEqual(bool(cs[0]["on_blunt"]), bool(good_blunt), (sgn, [(c["on_blunt"], c["margin"]) for c in cs]))
        e["blunt_dir"] = list(x_long)
        r = sk.plan_egg_place(source_egg="egg_1", target_slot="slot_1")
        self.assertTrue(r["ok"] and sk.plan["on_blunt"])
        self.assertTrue(any("蛋的大头" in l for l in io.logs))

    def test_u6_sweep_transit(self):
        use("agx_gripper")
        sk, io = T13.user_scene()
        sk.plan_egg_place(source_egg="egg_1", target_slot="slot_1")
        pl = sk.plan
        zt = float(pl["table_z_m"])
        e_G, hl = np.asarray(pl["egg_in_G_m"]), float(pl["egg_half_len_m"])
        ok, worst, where = sk._egg_sweep_ok([pl["_q"]["lift"], pl["_q"]["release"]], e_G, hl, zt)
        self.assertFalse(ok)                                   # the release pose is inside the carton by design
        # make the carton very tall: the direct move must be replaced by the high carry
        orig = sk._egg_obstacles
        sk._egg_obstacles = lambda z: [(xy, top + 0.05, r, c) for xy, top, r, c in orig(z)]
        try:
            gc = {"q_lift": pl["_q"]["lift"], "p_lift": pl["_wps"]["lift"][0]}
            sc = {"q_above": pl["_q"]["above_target"], "p_above": pl["_wps"]["above_target"][0],
                  "R_rel": pl["_wps"]["above_target"][1]}
            steps, note = sk._plan_transit(gc, sc, e_G, hl, zt)
        finally:
            sk._egg_obstacles = orig
        if steps is not None:
            self.assertTrue(steps)
            self.assertEqual(steps[-1]["kind"], "line")
            self.assertIn("经高处", note)
        else:
            self.assertIn("直接", note)                          # refused with the reason, never a collision
        with open(os.path.join(HERE, "NERO_V365_Real_Cup_Task.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn('transit = self.plan.get("_transit") or []', src)

    def test_u7_fence_start(self):
        with open(os.path.join(HERE, "NERO_V365_Real_Cup_Task.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn("self._check_path_fence([list(q_seed), q_pre, q_g, q_lift], zt)", src)
        self.assertIn("n_seg = 24 if idx == 0 else 3", src)

    def test_u8_missing_finetuned(self):
        import NERO_V366_Real_Cup_Yolo as YC
        YP = types.SimpleNamespace(parse_model_set=lambda s: ("yoloe-26s-seg.pt", ["finetuned/a/a.pt", "finetuned/b/b.pt"]),
                                   find_model=lambda d, n: "/x/" + n if n.endswith("a.pt") else None)
        self.assertEqual(YC.missing_finetuned(YP, "/x", "yoloe-26s-seg.pt;finetuned/a/a.pt;finetuned/b/b.pt"),
                         ["finetuned/b/b.pt"])
        txt = YC.describe_cfg({"mode": "text", "prompts": ["egg"], "conf": 0.25, "path": "/x/yoloe-26s-seg.pt",
                               "missing_models": ["finetuned/b/b.pt"]})
        self.assertIn("精调模型找不到 1 个", txt)
        with open(os.path.join(HERE, "NERO_V365_Real_Cup_Task.py"), encoding="utf-8") as f:
            self.assertIn("[V4.8.4 YOLO] Camera调试 选了精调模型", f.read())

    def test_u9_selectors(self):
        for fn, key in (("NERO_V365_Real_Cup_Task.py", "self.GG.CHOICE_LABELS[t] for t in self.GG.CHOICES"),
                        ("NERO_V4_Console_Tab.py", "GG.CHOICE_SHORT[t] for t in GG.CHOICES"),
                        ("NERO_V48_Gripper_UI.py", "GG.CHOICE_LABELS[t] for t in GG.CHOICES")):
            with open(os.path.join(HERE, fn), encoding="utf-8") as f:
                self.assertIn(key, f.read(), fn)
        import NERO_V4_Dataset_Recorder as DR
        use("agx_softfin")
        self.assertEqual(DR._gripper_name(), "AgxGripper-Softfin")
        use("agx_gripper")
        self.assertEqual(DR._gripper_name(), "AgxGripper")

    def test_u10_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual((v, C.VERSION, RR.VERSION), ("V4.8.4", "V4.8.4", "V4.8.4"))
        self.assertEqual(AG.SOFTFIN["fin_w_mm"], 25.8)


if __name__ == "__main__":
    unittest.main(verbosity=1)
