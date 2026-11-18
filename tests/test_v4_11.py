#!/usr/bin/env python3
"""NERO V4.11 offline tests (user 11-18: 「做 Sim DOE 页」 + 「清掉残留的建议性文字」).

D1 facts only: the general pick & place tool results carry no recommended_next_actions; the advice sentences
   ("Call perceive_objects ...", "look before doing anything else", restart advice, "use the measurements to decide
   the next step") are gone; tool results in the twin have none of them.
D2 truth check (mujoco): parse_goals; evaluate() on the household MJCF: nothing done -> fail; apple put into the bowl
   -> that goal ok (partial / success); unknown object ids reported; eval request -> physics hook -> .done answer.
D3 task stats: run_general_pick_place with a scripted agent in the twin fills TASK_STATS (tool calls, reasons,
   usage, error on an exhausted step budget).
D4 sim bridge (no Tk): an inbox message with llm + reset_objects sets the panel and the harness's object reset; the
   outbox carries llm + stats; stop_<id>.json is handled while a task runs; ready.json.
D5 DOE logic: expand / grouping / shuffle, check_row, parse_model, classify, Wilson interval, report, csv.
D6 DOE runner against a fake simulation: one simulation start per scene, every run judged, a time-out, a crash and a
   blow-up restart the simulation, abort, unique file ids.
D7 console wiring (source) + versions; backend unchanged since V4.10.2.
"""
from __future__ import annotations

import hashlib
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
os.environ.setdefault("NERO_CALIB_DIR", tempfile.mkdtemp(prefix="nero_v411_"))

try:
    import mujoco  # noqa: F401
    HAS_MJ = True
except Exception:
    HAS_MJ = False


def _src(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return f.read()


class _Var:
    def __init__(self, v=""):
        self.v = v

    def get(self):
        return self.v

    def set(self, v):
        self.v = v


class V411(unittest.TestCase):
    # ------------------------------------------------------------------ D1
    def test_d1_facts_only(self):
        import NERO_V489_General_Pick_Place as GP
        s = _src("NERO_V489_General_Pick_Place.py")
        for bad in ("Call perceive_objects to inspect", "look before doing anything else", "or a restart if motions",
                    "use the measurements to decide the next step", "Use them to check parameters"):
            self.assertNotIn(bad, s)
        self.assertNotIn('"recommended_next_actions": list(', s)
        sk = GP.GeneralPickPlaceSkills.__new__(GP.GeneralPickPlaceSkills)
        sk.stage, sk.held, sk._probing = "IDLE", None, True
        out = sk._res("plan_grasp_chunk", False, {"x": 1}, reason="OBSERVE_FIRST", nxt=["perceive_objects"])
        self.assertNotIn("recommended_next_actions", out)
        self.assertEqual(out["reason"], "OBSERVE_FIRST")

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_d1_twin_outputs(self):
        import NERO_V410_Household_Scenes as HS
        import NERO_V489_General_Pick_Place as GP
        io = HS.make_twin(HS.fixed_scene("H1"))
        sk = GP.GeneralPickPlaceSkills(io)
        outs = [sk.run_tool("perceive_objects", {"prompts": ["apple", "bowl"], "use_finetuned": True}),
                sk.run_tool("plan_place_chunk", {"target_id_or_xyz": "bowl_1", "xy_offset_mm": [0, 0],
                                                 "held_bottom_above_target_top_mm": 0, "yaw_change_deg": 0,
                                                 "approach_height_mm": 100, "release_open_mm": 60,
                                                 "carry_lift_mm": 0, "pitch_change_deg": 0}),
                sk.run_tool("execute_grasp_chunk", {"clamp_force_n": 10})]
        txt = json.dumps(outs, default=str)
        self.assertNotIn("recommended_next_actions", txt)
        self.assertNotIn("Call perceive", txt)

    # ------------------------------------------------------------------ D2
    def test_d2_parse_goals(self):
        import NERO_V410_Household_Scenes as HS
        self.assertEqual(HS.parse_goals("apple_1 into bowl_1; remote_1 onto tray_1"),
                         [("apple_1", "into", "bowl_1"), ("remote_1", "onto", "tray_1")])
        self.assertEqual(HS.parse_goals("a into b，c onto d\ne into f"), [("a", "into", "b"), ("c", "onto", "d"),
                                                                        ("e", "into", "f")])
        with self.assertRaises(ValueError):
            HS.parse_goals("apple_1 in bowl_1")
        self.assertEqual(HS.goals_text([("a", "into", "b")]), "a into b")

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_d2_evaluate(self):
        import mujoco
        import NERO_V410_Household_Scenes as HS
        sc = HS.fixed_scene("H1")
        m = mujoco.MjModel.from_xml_string(f'<mujoco><asset>{sc.assets(visual=False)}</asset><worldbody>'
                                           f'<geom type="box" pos="-0.385 0 -0.03" size="0.335 0.38 0.03"/>'
                                           f'{sc.world_xml(visual=False)}</worldbody></mujoco>')
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        ev = HS.evaluate(m, d, sc)
        self.assertEqual(ev["status"], "fail")
        self.assertEqual(ev["n_goals"], 2)
        self.assertEqual(set(ev["objects"]), {it.oid for it in sc.items})
        bowl = d.body("obj_bowl_1").xpos.copy()
        qa = m.jnt_qposadr[m.body_jntadr[d.body("obj_apple_1").id]]
        d.qpos[qa:qa + 3] = bowl + np.array([0.0, 0.0, 0.02])
        d.qvel[:] = 0
        for _ in range(1500):
            mujoco.mj_step(m, d)
        ev = HS.evaluate(m, d, sc)
        self.assertEqual(ev["status"], "partial", ev["goals"])
        self.assertTrue(ev["goals"][0]["ok"])
        ev1 = HS.evaluate(m, d, sc, goals=[("apple_1", "into", "bowl_1")])
        self.assertEqual(ev1["status"], "success")
        ev2 = HS.evaluate(m, d, sc, goals=[("apple_1", "into", "mug_9")])
        self.assertEqual(ev2["status"], "fail")
        self.assertIn("not in this scene", ev2["goals"][0]["detail"])
        td = tempfile.mkdtemp()
        old = {k: os.environ.get(k) for k in (HS.ENV_RESET, HS.ENV_SPEC)}
        os.environ[HS.ENV_RESET] = os.path.join(td, "reset.json")
        os.environ[HS.ENV_SPEC] = "fixed:H1"
        try:
            self.assertIsNone(HS.eval_hook(m, d))
            f = HS.request_eval([("apple_1", "into", "bowl_1")], rid="x1")
            self.assertTrue(f.startswith(td) and f.endswith("_eval.json"))
            HS._RESET.update(home={}, t0=0.0, done=None)
            HS.reset_hook(m, d)                                  # the physics hook answers the eval request
            with open(f + ".done") as fh:
                ans = json.load(fh)
            self.assertEqual((ans["id"], ans["status"]), ("x1", "success"))
            self.assertFalse(os.path.exists(f))
        finally:
            HS._RESET.update(home=None, t0=None, done=None)
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    # ------------------------------------------------------------------ D3
    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_d3_task_stats(self):
        import threading
        import NERO_V410_Household_Scenes as HS
        import NERO_V489_General_Pick_Place as GP

        class Agent:
            def __init__(self, script):
                self.script = list(script)
                self._lock = threading.Lock()
                self._running = False
                self._stop_event = threading.Event()
                self._last_response_id = None
                self._usage_totals = {"requests": 0, "input_tokens": 0, "output_tokens": 0}

            def api_key_status(self, k, m):
                return True, "ok"

            def _post_responses(self, payload, key):
                return self.script.pop(0)

            def _log_usage(self, resp, model):
                self._usage_totals["requests"] += 1
                self._usage_totals["input_tokens"] += 100
                self._usage_totals["output_tokens"] += 10

            def _function_calls(self, resp):
                return resp.get("calls", [])

            def _extract_text(self, resp):
                return resp.get("text", "")

        io = HS.make_twin(HS.fixed_scene("H1"))
        call = {"name": "perceive_objects", "arguments": json.dumps({"prompts": ["apple"], "use_finetuned": True}),
                "call_id": "c1"}
        bad = {"name": "execute_place_chunk", "arguments": json.dumps({"retreat_up_mm": 50}), "call_id": "c2"}
        ag = Agent([{"id": "1", "calls": [call]}, {"id": "2", "calls": [bad]}, {"id": "3", "text": "done"}])
        r = GP.run_general_pick_place(io, "put the apple into the bowl", agent=ag,
                                      llm_cfg={"model": "m1", "api_key": "k", "max_tool_steps": 30})
        st = GP.TASK_STATS
        self.assertFalse(r["ok"])
        self.assertEqual([t["tool"] for t in st["tools"]], ["perceive_objects", "execute_place_chunk"])
        self.assertEqual([t["ok"] for t in st["tools"]], [True, False])
        self.assertEqual(st["tools"][1]["reason"], "PLACE_NOT_PLANNED")
        self.assertEqual((st["rounds"], st["steps"], st["model"], st["error"]), (3, 2, "m1", None))
        self.assertEqual(st["usage"]["requests"], 3)
        self.assertEqual(st["final"], "done")
        ag2 = Agent([{"id": str(k), "calls": [call]} for k in range(12)])
        with self.assertRaises(RuntimeError):
            GP.run_general_pick_place(io, "t", agent=ag2, llm_cfg={"model": "m1", "api_key": "k", "max_tool_steps": 10})
        self.assertIn("step budget", GP.TASK_STATS["error"])
        self.assertEqual(GP.TASK_STATS["steps"], 11)

    # ------------------------------------------------------------------ D4
    def test_d4_bridge(self):
        import time
        import NERO_V410_Sim_Bridge as B
        import NERO_V489_General_Pick_Place as GP
        td = tempfile.mkdtemp()
        os.environ["NERO_V410_TASK_DIR"] = td
        try:
            v = {k: _Var(x) for k, x in (("provider", "OpenAI"), ("model", "gpt-x"), ("effort", "low"), ("steps", "60"),
                                         ("dry", False), ("key", ""), ("task", ""))}
            status = {"text": "Agent: IDLE"}
            calls = {"run": 0, "stop": 0, "prov": []}

            def run():
                calls["run"] += 1
                status["text"] = "Agent: BUSY | LLM agent: RUN TASK"
            g = {"vars": lambda: v, "labels": lambda: {"openai": "OpenAI", "deepseek": "DeepSeek"},
                 "on_provider": lambda: (lambda: calls["prov"].append(v["provider"].get())),
                 "run": lambda: run, "stop": lambda: (lambda: calls.__setitem__("stop", calls["stop"] + 1)),
                 "status": lambda: dict(status),
                 "agent": lambda: type("A", (), {"running": property(lambda s: status["text"].startswith("Agent: BUSY"))})()}
            logs = []
            ch = B.TaskChannel(None, g, logs.append)
            GP._V4103_BLOWUP.update(pending=False, why="")
            with open(os.path.join(td, "inbox", "d1.json"), "w") as f:
                json.dump({"id": "d1", "task": "apple into bowl", "reset_objects": True,
                           "llm": {"provider": "deepseek", "model": "deepseek-v4-pro", "effort": "max", "steps": 40,
                                   "dry_run": False}}, f)
            v["dry"].set(True)
            ch._step()
            self.assertEqual(calls["run"], 1)
            self.assertEqual((v["provider"].get(), v["model"].get(), v["effort"].get(), v["steps"].get()),
                             ("DeepSeek", "deepseek-v4-pro", "max", "40"))
            self.assertFalse(v["dry"].get())
            self.assertEqual(calls["prov"], ["DeepSeek"])
            self.assertTrue(GP._V4103_BLOWUP["pending"])
            GP._V4103_BLOWUP.update(pending=False, why="")
            GP.TASK_STATS.clear()
            GP.TASK_STATS.update(t_start=time.time(), tools=[{"tool": "perceive_objects", "ok": True}], safety_stops=0)
            with open(os.path.join(td, "inbox", "stop_d1.json"), "w") as f:
                json.dump({"id": "d1", "cmd": "stop"}, f)
            ch._step()                                           # busy: the stop is still handled
            self.assertEqual(calls["stop"], 1)
            status["text"] = "Agent: IDLE | LLM agent: RUN TASK (general_pick_place) complete"
            ch._step()
            with open(os.path.join(td, "outbox", "d1.json")) as f:
                out = json.load(f)
            self.assertEqual(out["status"], "complete")
            self.assertEqual(out["llm"]["provider"], "deepseek")
            self.assertEqual(out["stats"]["tools"][0]["tool"], "perceive_objects")
            B.write_ready()
            with open(os.path.join(td, "ready.json")) as f:
                self.assertIn("pid", json.load(f))
        finally:
            os.environ.pop("NERO_V410_TASK_DIR", None)

    # ------------------------------------------------------------------ D5
    def test_d5_logic(self):
        import NERO_V411_Sim_DOE as D
        rows = D.default_rows(["fixed:H1", "fixed:H3"]) + D.default_rows(["seed:3"])
        self.assertEqual([D.check_row(r) for r in rows], [[], [], []])
        self.assertTrue(D.check_row({"scene": "fixed:H1", "task": "x", "goals": "apple_1 into mug_7"}))
        self.assertTrue(D.check_row({"scene": "fixed:H1", "task": "x", "goals": ""}))
        self.assertTrue(D.check_row({"scene": "fixed:H9", "task": "x", "goals": "a into b"}))
        self.assertEqual(D.parse_model("deepseek:deepseek-v4-pro@max"),
                         {"provider": "deepseek", "model": "deepseek-v4-pro", "effort": "max"})
        self.assertEqual(D.parse_model("kimi-k3")["provider"], "kimi")
        rows.append(dict(rows[0], task="the apple goes in the bowl, the remote in the box"))   # same scene, new wording
        plan = {"rows": rows, "models": [D.parse_model("a:m1"), D.parse_model("b:m2@low")], "replicates": 2,
                "shuffle": True, "order_seed": 7}
        runs = D.expand(plan)
        self.assertEqual(len(runs), 4 * 2 * 2)
        sc_order = [r["scene"] for r in runs]
        self.assertEqual([s for k, s in enumerate(sc_order) if k == 0 or s != sc_order[k - 1]],
                         ["fixed:H1", "fixed:H3", "seed:3"])                     # one block per scene
        self.assertEqual([r["id"] for r in runs], [r["id"] for r in D.expand(plan)])
        self.assertEqual([(r["row"], r["model"], r["rep"]) for r in runs],
                         [(r["row"], r["model"], r["rep"]) for r in D.expand(plan)])   # same order seed, same order
        self.assertNotEqual([(r["row"], r["model"], r["rep"]) for r in runs],
                            [(r["row"], r["model"], r["rep"]) for r in D.expand(dict(plan, shuffle=False))])
        ok_ev = {"status": "success", "n_ok": 2, "n_goals": 2, "objects": {}}
        part = {"status": "partial", "n_ok": 1, "n_goals": 2, "objects": {}}
        fail = {"status": "fail", "n_ok": 0, "n_goals": 2, "objects": {}}
        out = {"status": "complete", "stats": {"tools": [{"tool": "x", "ok": True, "safety_stops": 0}]}}
        C = D.classify
        self.assertEqual(C(out, ok_ev), ("success", "success"))
        self.assertEqual(C(out, part), ("partial", "partial"))
        self.assertEqual(C(out, fail), ("fail", "ended_not_done"))
        self.assertEqual(C(dict(out, stats={"ok": True, "stage": "VERIFIED"}), fail), ("fail", "claimed_but_not_done"))
        self.assertEqual(C(dict(out, stats={"tools": [{"safety_stops": 7}], "safety_stops": 7}), fail),
                         ("invalid", "blowup"))
        self.assertEqual(C(out, dict(fail, objects={"a": {"off_table": True}})), ("invalid", "blowup"))
        self.assertEqual(C(dict(out, stats={"safety_stops": 2, "tools": [{"safety_stops": 2}]}), fail),
                         ("fail", "safety_stops"))
        self.assertEqual(C(dict(out, stats={"error": "RuntimeError: tool step budget 60 used up"}), fail),
                         ("fail", "step_budget"))
        self.assertEqual(C(dict(out, stats={"error": "HTTPError: 500"}), fail), ("invalid", "llm_error"))
        self.assertEqual(C({"status": "not_started"}, None), ("invalid", "not_started"))
        self.assertEqual(C(out, None, timed_out=True), ("invalid", "timeout"))
        self.assertEqual(C(None, None, crashed=True), ("invalid", "sim_crash"))
        self.assertEqual(C(out, None), ("invalid", "no_eval"))
        lo, hi = D.wilson(8, 10)
        self.assertAlmostEqual(lo, 0.490, places=2)
        self.assertAlmostEqual(hi, 0.943, places=2)
        self.assertEqual(D.wilson(0, 0), (0.0, 0.0))
        for k, r in enumerate(runs):
            r["status"] = ("success", "fail", "partial", "invalid")[k % 4]
            r["klass"] = {"success": "success", "fail": "ended_not_done", "partial": "partial",
                          "invalid": "timeout"}[r["status"]]
            r["stats"] = {"tools": [{"tool": "plan_place_chunk", "ok": False, "reason": "PLACE_COLLISION"}],
                          "safety_stops": 1, "usage": {"input_tokens": 100, "output_tokens": 5}}
            r["last_fail"] = D.last_fail_reason(r["stats"])
            r["seconds"] = 60
        tot = D.summary_rows(runs, lambda r: "all")[0]
        self.assertEqual((tot["runs"], tot["judged"], tot["success"], tot["invalid"]), (16, 12, 4, 4))
        self.assertAlmostEqual(tot["rate"], 4 / 12)
        md = D.report_md(dict(plan, name="t"), runs)
        for k in ("按 LLM 设置", "按任务", "结果分类", "plan_place_chunk: PLACE_COLLISION", "a:m1@high", "b:m2@low"):
            self.assertIn(k, md)
        csv_txt = D.runs_csv(runs)
        self.assertEqual(len(csv_txt.strip().splitlines()), 17)
        self.assertGreater(D.estimate_s(plan), 16 * 200)

    # ------------------------------------------------------------------ D6
    def test_d6_runner(self):
        import NERO_V410_Household_Scenes as HS
        import NERO_V411_Sim_DOE as D
        td, od = tempfile.mkdtemp(), tempfile.mkdtemp()
        old = os.environ.get(HS.ENV_RESET)
        os.environ[HS.ENV_RESET] = os.path.join(td, "reset.json")
        clock = [1000.0]
        sim = {"up": False, "starts": [], "stops": 0, "crash_on": None, "hang_on": None, "blow_on": None}

        def start(spec, env):
            sim["up"] = True
            sim["starts"].append(spec)
            sim["t_ready"] = clock[0] + 30
            return True, "started"

        def stop():
            sim["up"] = False
            sim["stops"] += 1
        api = {"start": start, "stop": stop, "running": lambda: sim["up"]}

        def fake_sim_side():
            """what the simulation does: ready file, task channel, truth check"""
            if not sim["up"]:
                if os.path.exists(os.path.join(td, "ready.json.mark")):
                    os.remove(os.path.join(td, "ready.json.mark"))
                return
            if clock[0] >= sim.get("t_ready", 1e18) and not os.path.exists(os.path.join(td, "ready.json.mark")):
                with open(os.path.join(td, "ready.json"), "w") as f:
                    json.dump({"t": clock[0]}, f)
                os.utime(os.path.join(td, "ready.json"), (clock[0], clock[0]))
                open(os.path.join(td, "ready.json.mark"), "w").close()
            inbox = os.path.join(td, "inbox")
            for n in sorted(os.listdir(inbox)):
                if not n.endswith(".json"):
                    continue
                with open(os.path.join(inbox, n)) as f:
                    msg = json.load(f)
                os.remove(os.path.join(inbox, n))
                if n.startswith("stop_"):
                    continue
                rid = msg["id"].split("_")[-1]
                self.assertTrue(msg["id"].startswith("doe"))
                self.assertTrue(msg["reset_objects"])
                if rid == sim["crash_on"]:
                    sim["up"] = False
                    continue
                if rid == sim["hang_on"]:
                    continue
                st = {"tools": [{"tool": "perceive_objects", "ok": True, "safety_stops": 0}], "safety_stops": 0}
                if rid == sim["blow_on"]:
                    st = {"tools": [{"tool": "execute_place_chunk", "ok": False, "reason": "SAFETY_STOP",
                                     "safety_stops": 9}], "safety_stops": 9}
                with open(os.path.join(td, "outbox", msg["id"] + ".json"), "w") as f:
                    json.dump({"id": msg["id"], "status": "complete", "stats": st, "llm": msg["llm"]}, f)
            ef = HS.eval_file()
            if os.path.exists(ef):
                with open(ef) as f:
                    req = json.load(f)
                os.remove(ef)
                ok = int(req["id"].split("_r")[-1]) % 2 == 1
                with open(ef + ".done", "w") as f:
                    json.dump({"id": req["id"], "status": "success" if ok else "fail", "n_ok": int(ok), "n_goals": 1,
                               "objects": {}}, f)

        def drive(rn, max_ticks=20000):
            for _ in range(max_ticks):
                if rn.finished:
                    return
                fake_sim_side()
                rn.tick()
                clock[0] += 1.0
            self.fail(f"runner did not finish (state {rn.state})")

        try:
            import NERO_V411_Sim_DOE as D
            rows = D.default_rows(["fixed:H1", "fixed:H2"])
            plan = {"rows": rows, "models": [D.parse_model("a:m1"), D.parse_model("b:m2")], "replicates": 2,
                    "run_timeout_min": 2}
            rn = D.DOERunner(plan, api, od, td, now=lambda: clock[0], timeouts={"settle_s": 3, "after_stop_s": 2,
                                                                               "stop_grace_s": 20})
            rn.start()
            drive(rn)
            self.assertEqual(sim["starts"], ["fixed:H1", "fixed:H2"])
            self.assertEqual([r["status"] for r in rn.runs], ["success", "fail"] * 4)
            self.assertFalse(sim["up"])
            for f in ("plan.json", "runs.jsonl", "runs.csv", "report.md", "doe.log"):
                self.assertTrue(os.path.exists(os.path.join(od, f)), f)
            plan2, runs2 = D.load_runs(od)
            self.assertEqual(len(runs2), 8)
            # time-out (no answer after STOP -> restart), crash, blow-up -> fresh simulation for the next run
            od2 = tempfile.mkdtemp()
            sim.update(starts=[], crash_on="r002", hang_on="r004", blow_on="r006")
            plan3 = dict(plan, rows=rows[:1], replicates=4, models=plan["models"][:2])
            rn = D.DOERunner(plan3, api, od2, td, now=lambda: clock[0],
                             timeouts={"settle_s": 3, "after_stop_s": 2, "stop_grace_s": 20})
            rn.start()
            drive(rn)
            k = {r["id"]: (r["status"], r["klass"]) for r in rn.runs}
            self.assertEqual(k["r001"], ("success", "success"))
            self.assertEqual(k["r002"], ("invalid", "sim_crash"))
            self.assertEqual(k["r004"], ("invalid", "timeout"))
            self.assertEqual(k["r006"], ("invalid", "blowup"))
            self.assertEqual(k["r007"], ("success", "success"))
            self.assertEqual(len(sim["starts"]), 4)            # first + after the crash, the time-out, the blow-up
            # abort
            od3 = tempfile.mkdtemp()
            sim.update(starts=[], crash_on=None, hang_on="r001", blow_on=None)
            rn = D.DOERunner(plan3, api, od3, td, now=lambda: clock[0], timeouts={"settle_s": 3})
            rn.start()
            for _ in range(80):
                fake_sim_side()
                rn.tick()
                clock[0] += 1.0
            self.assertEqual(rn.state, "running")
            rn.abort()
            drive(rn)
            self.assertTrue(all(r["status"] in ("skipped", "invalid") for r in rn.runs))
            self.assertFalse(sim["up"])
        finally:
            if old is None:
                os.environ.pop(HS.ENV_RESET, None)
            else:
                os.environ[HS.ENV_RESET] = old

    # ------------------------------------------------------------------ D7
    def test_d7_wiring_versions(self):
        import NERO_V312_Console as C
        import NERO_V411_Sim_DOE as D
        import NERO_V489_General_Pick_Place as GP
        cs = _src("NERO_V312_Console.py")
        for k in ('def on_sim_start(auto_spec=None, extra_env=None):', 'nb.insert(nb.index(sim_tab) + 1, doe_tab.frame, '
                  'text="仿真 DOE")', '"key_env": llm410.keys_env', "def on_sim_stop(ask=True):",
                  'if not doe411.get("auto") and not messagebox.askyesno("启动前检查（V3.77）"',
                  "env410.update(extra_env or {})", "pid = runner.start(extra_env=scene_env(scene))"):
            self.assertIn(k, cs)
        self.assertEqual((C.VERSION, GP.VERSION, D.VERSION), ("V4.11", "V4.11", "V4.11"))
        self.assertEqual(_src("VERSION").strip(), "V4.11")
        self.assertIn("SIM DOE", GP.MARKER_V411)
        h = hashlib.sha256(open(os.path.join(HERE, "NERO_V2_7_1_Unified_Backend.py"), "rb").read()).hexdigest()
        self.assertEqual(h, "1a1da2c8c8a9836e335b42444f95b112b0839be9277af62871ba9f19cd24f2fa")
        for bad in ("\U0001F600", "✅", "❌"):
            self.assertNotIn(bad, _src("NERO_V411_Sim_DOE.py"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
