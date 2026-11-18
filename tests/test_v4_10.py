#!/usr/bin/env python3
"""NERO V4.10 offline tests (household scenes + console LLM settings + scene dialog + simulation bridge).

H1 assets: every household item has meta.json (source, licence, size), its texture (<= 1024 px) and as many convex
   parts as meta.json says; sizes fit the official gripper where they are meant to be gripped (<= 62 mm pad gap).
H2 scenes: the fixed scenes H1-H5 and random seeds are valid (no overlap, on the table, sources reachable top-down,
   target openings reachable, every source fits its container) and reproducible from the seed.
H3 (mujoco) physics + perception: objects settle (< 3 mm in 2 s); collision geoms are opaque in group 3 (mj_ray skips
   alpha 0); the general pick & place perception sees every object of the twin.
H4 (mujoco) object reset: settled poses recorded, a moved object goes back on a request, the answer file is written.
H5 scene module: household variant (TF / hold names / no fixed MoveIt boxes / XML compiles with the scans); the
   original general_pick_place world XML is byte-identical to V4.9.3.
H6 console logic: LLM environment (key only in the provider's variable), scene environment, scene list, keyring
   wrapper degrades without a keyring.
H7 bridge: console settings parsed from the environment, panel filled + locked, task file channel (inbox -> RUN TASK
   -> outbox); attached from NERO_V359_Strategy.add_sim_strategy_tab, the backend is unchanged.
H8 versions / markers.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTS = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("NERO_CALIB_DIR", tempfile.mkdtemp(prefix="nero_v410_"))

try:
    import mujoco  # noqa: F401
    HAS_MJ = True
except Exception:
    HAS_MJ = False

import NERO_V410_Household_Scenes as HS  # noqa: E402


class _Var:
    def __init__(self, v=None):
        self.v = v

    def set(self, v):
        self.v = v

    def get(self):
        return self.v


class _W:
    def __init__(self):
        self.state = "normal"

    def configure(self, **k):
        self.state = k.get("state", self.state)


class _Frame:
    def __init__(self):
        self.kids = [_W(), _W(), _W()]

    def winfo_children(self):
        return self.kids


class V410(unittest.TestCase):
    def test_h1_assets(self):
        for k in ("apple", "bottle", "mug", "bowl", "tray", "storage_box"):
            d = os.path.join(HS.ASSET_DIR, k)
            m = json.load(open(os.path.join(d, "meta.json")))
            self.assertIn(m["source"], ("GSO", "YCB"))
            self.assertIn("CC BY 4.0", m["license"])
            self.assertTrue(os.path.exists(os.path.join(d, "visual.obj")))
            hulls = sorted(f for f in os.listdir(d) if f.startswith("hull_"))
            self.assertEqual(len(hulls), m["hulls"], k)
            with open(os.path.join(d, "texture.png"), "rb") as f:
                head = f.read(24)
            w, h = int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")
            self.assertLessEqual(max(w, h), 1024)
        self.assertLessEqual(2 * HS.dims("apple")["r"], 0.062)        # the official gripper's pad gap (70 mm stroke)
        self.assertLessEqual(2 * HS.dims("bottle")["r"], 0.062)
        self.assertLessEqual(2 * HS.dims("remote")["hy"], 0.062)

    def test_h2_scenes(self):
        for sid in HS.scene_ids():
            sc = HS.fixed_scene(sid)
            gap = HS.FIXED[sid]["gap"]
            self.assertEqual(HS.check_layout(sc.items, sc.goals, gap=gap), [], sid)
            self.assertTrue(sc.goals and sc.task)
        for seed in (1, 2, 3, 7, 11):
            a, b = HS.random_scene(seed), HS.random_scene(seed)
            self.assertEqual([(i.oid, i.x, i.y, i.yaw_deg) for i in a.items], [(i.oid, i.x, i.y, i.yaw_deg) for i in b.items])
            self.assertEqual(HS.check_layout(a.items, a.goals), [], seed)
            self.assertTrue(4 <= len(a.items) <= 5)
        self.assertFalse(HS.fits("bottle", "mug"))
        self.assertFalse(HS.fits("remote", "bowl"))
        self.assertTrue(HS.fits("apple", "bowl"))
        self.assertEqual(HS.scene_from_spec("fixed:H3").sid, "H3")
        self.assertEqual(HS.scene_from_spec("seed:5").seed, 5)

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_h3_physics_perception(self):
        import mujoco
        import NERO_V489_General_Pick_Place as GP
        sc = HS.fixed_scene("H4")
        xml = (f'<mujoco><option timestep="0.001"/><asset>{sc.assets()}</asset><worldbody>'
               f'<geom type="box" pos="-0.385 0 -0.03" size="0.335 0.38 0.03"/>{sc.world_xml()}</worldbody></mujoco>')
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        p0 = {n: d.body(n).xpos.copy() for n in sc.body_names()}
        for _ in range(2000):
            mujoco.mj_step(m, d)
        for n, p in p0.items():
            self.assertLess(float(np.linalg.norm(d.body(n).xpos - p)), 0.003, n)
        for g in range(m.ngeom):
            if m.geom_contype[g] and m.geom_bodyid[g] > 0 and m.geom_rgba[g][3] == 0:
                self.fail("a colliding geom is fully transparent (mj_ray would not see it)")
        io = HS.make_twin(sc)
        sk = GP.GeneralPickPlaceSkills(io)
        r = sk.perceive_objects(prompts=["object"], use_finetuned=True)
        self.assertEqual(sorted(o["object_id"] for o in r["objects"]), sorted(i.oid for i in sc.items))

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_h4_reset(self):
        import mujoco
        sc = HS.fixed_scene("H2")
        m = mujoco.MjModel.from_xml_string(f'<mujoco><asset>{sc.assets(visual=False)}</asset><worldbody>'
                                           f'<geom type="box" pos="-0.385 0 -0.03" size="0.335 0.38 0.03"/>'
                                           f'{sc.world_xml(visual=False)}</worldbody></mujoco>')
        d = mujoco.MjData(m)
        td = tempfile.mkdtemp()
        os.environ[HS.ENV_RESET] = os.path.join(td, "reset.json")
        try:
            HS._RESET.update(home=None, t0=None, done=None)
            HS.reset_hook(m, d)
            HS._RESET["t0"] -= 10
            for _ in range(500):
                mujoco.mj_step(m, d)
            HS.reset_hook(m, d)
            self.assertEqual(sorted(HS._RESET["home"]), sorted(sc.body_names()))
            n = sc.body_names()[0]
            home = d.body(n).xpos.copy()
            qa = m.jnt_qposadr[m.body_jntadr[d.body(n).id]]
            d.qpos[qa:qa + 2] += 0.08
            mujoco.mj_forward(m, d)
            self.assertIsNone(HS.reset_hook(m, d))                  # no request -> nothing
            HS.request_reset()
            self.assertIn("RESET", HS.reset_hook(m, d))
            self.assertLess(float(np.linalg.norm(d.body(n).xpos - home)), 1e-6)
            self.assertTrue(os.path.exists(HS.reset_file() + ".done"))
            self.assertFalse(os.path.exists(HS.reset_file()))
        finally:
            os.environ.pop(HS.ENV_RESET, None)
            HS._RESET.update(home=None, t0=None, done=None)

    def test_h5_scene_module(self):
        import NERO_V343_Scenes as SC
        old = {k: os.environ.get(k) for k in ("NERO_SCENE", "NERO_V343_ACTIVE_SCENE", HS.ENV_SPEC)}
        try:
            os.environ["NERO_SCENE"] = os.environ["NERO_V343_ACTIVE_SCENE"] = "general_pick_place"
            self.assertEqual(hashlib.sha256(SC.general_world_xml().encode()).hexdigest()[:12], "61f7db223038")
            os.environ["NERO_SCENE"] = os.environ["NERO_V343_ACTIVE_SCENE"] = "household_pick_place"
            os.environ[HS.ENV_SPEC] = "fixed:H1"
            self.assertEqual(SC.active_scene(), "sort_items")
            self.assertEqual(SC.selected_variant(), "household_pick_place")
            sc = HS.fixed_scene("H1")
            self.assertEqual(sorted(SC.tf_body_map()), sorted(sc.body_names()))
            self.assertTrue(set(sc.hold_names()) <= set(SC.hold_object_names()))
            self.assertNotIn("obj_storage_box_1", SC.hold_object_names())
            self.assertEqual(SC.planning_boxes(), [])
            wx, assets, slots = SC.general_probe_spec()
            self.assertIn("obj_apple_1", wx)
            self.assertIsNone(slots)
            td = tempfile.mkdtemp()
            src = os.path.join(td, "nero_motor.xml")
            with open(src, "w") as f:
                f.write("<mujoco model='t'><compiler angle='radian'/><asset></asset><worldbody><body name='link7' "
                        "pos='0 0 0.8'><geom type='sphere' size='0.01' contype='0' conaffinity='0'/></body>"
                        "</worldbody></mujoco>")
            out = SC.make_store_in_safe_xml(src)
            self.assertTrue(out.endswith("_nero_motor_v343_household_pick_place_tmp.xml"))
            txt = open(out).read()
            self.assertIn('name="hold_obj_apple_1"', txt)
            self.assertIn("hh_apple_vis", txt)
            if HAS_MJ:
                import mujoco
                mujoco.MjModel.from_xml_path(out)
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_h6_console_logic(self):
        import NERO_V312_Console as C
        import NERO_V343_Scenes as SC
        import NERO_V410_Console_Sim as CS
        self.assertEqual([k for k, _ in C.SIM_SCENES], list(SC.SCENES))
        self.assertIn("household_pick_place", [k for k, _ in C.SIM_SCENES])
        e = CS.llm_env("deepseek", "deepseek-v4-pro", "high", 40, True, "sk-x")
        self.assertEqual(e["DEEPSEEK_API_KEY"], "sk-x")
        self.assertNotIn("OPENAI_API_KEY", e)
        self.assertEqual(sum(1 for v in e.values() if v == "sk-x"), 1)
        self.assertEqual((e["NERO_LLM_MODEL"], e["NERO_V410_LLM_EFFORT"], e["NERO_V342_DEFAULT_STEPS"],
                          e["NERO_V410_LLM_DRY_RUN"]), ("deepseek-v4-pro", "high", "40", "1"))
        self.assertNotIn("OPENAI_API_KEY", CS.llm_env("openai", "m", "low", 60, False, ""))
        self.assertEqual(CS.scene_extra_env("household_pick_place", "seed:9", "/r")["NERO_V410_HOUSEHOLD"], "seed:9")
        self.assertNotIn("NERO_V410_HOUSEHOLD", CS.scene_extra_env("general_pick_place", None, "/r"))
        ks = CS.KeyStore()
        if not ks.available():
            self.assertEqual(ks.get("openai"), "")
            self.assertFalse(ks.set("openai", "x"))
        src = open(os.path.join(HERE, "NERO_V312_Console.py")).read()
        self.assertIn("env410.update(llm410.env())            # never logged", src)
        self.assertIn('env.update(getattr(self, "v410_env", None) or {})', src)
        sim = open(os.path.join(HERE, "NERO_V410_Console_Sim.py")).read()
        self.assertNotRegex(sim, "[\U0001F300-\U0001FAFF]")                  # no emoji in Tk text

    def test_h7_bridge(self):
        import NERO_V410_Sim_Bridge as B
        self.assertEqual(B.console_settings({}), {})
        cs = B.console_settings({"NERO_V410_LLM_FROM_CONSOLE": "1", "NERO_V410_LLM_PROVIDER": "kimi",
                                 "NERO_LLM_MODEL": "kimi-k3", "NERO_V410_LLM_EFFORT": "medium",
                                 "NERO_V342_DEFAULT_STEPS": "45", "NERO_V410_LLM_DRY_RUN": "1"})
        self.assertEqual(cs, {"provider": "kimi", "model": "kimi-k3", "effort": "medium", "steps": 45, "dry_run": True})
        V = {k: _Var() for k in ("provider", "model", "effort", "steps", "dry", "key", "task")}
        V["key"].set("typed")
        frame = _Frame()
        called = []
        g = {"vars": lambda: V, "labels": lambda: {"kimi": "Kimi (Moonshot)"}, "on_provider": lambda: (lambda: called.append(1)),
             "cfg_frame": lambda: frame}
        env = {k: os.environ.get(k) for k in ("NERO_V410_LLM_FROM_CONSOLE", "NERO_LLM_MODEL", "NERO_V410_LLM_PROVIDER")}
        os.environ.update({"NERO_V410_LLM_FROM_CONSOLE": "1", "NERO_V410_LLM_PROVIDER": "kimi", "NERO_LLM_MODEL": "kimi-k3"})
        try:
            self.assertTrue(B.apply_console_settings(g, lambda s: None))
        finally:
            for k, v in env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        self.assertEqual((V["provider"].get(), V["model"].get(), V["key"].get()), ("Kimi (Moonshot)", "kimi-k3", ""))
        self.assertEqual(called, [1])
        self.assertTrue(all(w.state == "disabled" for w in frame.kids))
        # task channel
        td = tempfile.mkdtemp()
        os.environ["NERO_V410_TASK_DIR"] = td
        try:
            class Agent:
                running = False
            ag, status, runs = Agent(), {"text": "LLM agent: IDLE"}, []

            def run():
                runs.append(V["task"].get())
                ag.running = True
                status["text"] = "LLM agent: RUN TASK (general_pick_place)..."

            class Root:
                def after(self, ms, fn):
                    pass
            ch = B.TaskChannel(Root(), {"vars": lambda: V, "status": lambda: status, "agent": lambda: ag,
                                        "run": lambda: run}, lambda s: None)
            with open(os.path.join(td, "inbox", "t001.json"), "w") as f:
                json.dump({"id": "t001", "task": "Put apple_1 into bowl_1."}, f)
            ch._step()
            self.assertEqual(runs, ["Put apple_1 into bowl_1."])
            ch._step()                                   # busy
            ag.running = False
            status["text"] = "LLM agent: RUN TASK (general_pick_place) complete"
            ch._step()
            out = json.load(open(os.path.join(td, "outbox", "t001.json")))
            self.assertEqual(out["status"], "complete")
            self.assertFalse(os.listdir(os.path.join(td, "inbox")))
        finally:
            os.environ.pop("NERO_V410_TASK_DIR", None)
        with open(os.path.join(HERE, "NERO_V2_7_1_Unified_Backend.py")) as f:
            self.assertNotIn("V410", f.read())          # the backend stays byte-identical (hash-pinned by V3.6x tests)
        with open(os.path.join(HERE, "NERO_V359_Strategy.py")) as f:
            self.assertIn("_B410.attach_from_notebook(notebook,", f.read())

    def test_h8_versions(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        import NERO_V489_General_Pick_Place as GP
        import NERO_V410_Console_Sim as CS
        import NERO_V410_Sim_Bridge as B
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual((v, RR.VERSION, GP.VERSION), (C.VERSION, C.VERSION, C.VERSION))
        self.assertIn(v, ("V4.10", "V4.10.1", "V4.10.2", "V4.10.3", "V4.11", "V4.11.1"))
        self.assertEqual((HS.MARKER, CS.MARKER, B.MARKER), ("V4.10 HOUSEHOLD SCENES",
                                                            "V4.10 CONSOLE LLM SETTINGS + SCENE DIALOG", "V4.10 SIM BRIDGE"))


if __name__ == "__main__":
    unittest.main(verbosity=1)
