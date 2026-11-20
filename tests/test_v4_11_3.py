#!/usr/bin/env python3
"""NERO V4.11.3 offline tests (user 11-18: 「按你上一条说的第一组4项内容去修改吧。不用录像。」).

U1 pictures with the simulation's Python: the truth check / safety record carry the simulation's interpreter and
   package paths; the DOE renders with them.  Reproduces the 11-17 failure: a console Python without mujoco
   ("No module named 'mujoco'") renders fine once the simulation's paths are used (needs an OpenGL backend).
U2 safety-stop moment: the harness asks once per tool call (at most 3 per run, only in a DOE run of the GUI
   simulation); the physics hook answers with held objects, contacts (penetration, normal force), qpos + model.
U3 DOE runner: a task that does not start is retried once (the first outbox removed); the safety records of the run
   are attached; their pictures are rendered next to the final ones; runs.html shows contacts + pictures.
U4 report: end-to-end success rate (+ Wilson interval), tokens per run / per success; time estimate 5.5 min.
U5 bridge: 45 s before "not started"; the run id reaches the harness; versions / markers.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTS = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("NERO_CALIB_DIR", tempfile.mkdtemp(prefix="nero_v4113_"))

try:
    import mujoco  # noqa: F401
    HAS_MJ = True
except Exception:
    HAS_MJ = False

NO_MJ_PY = "/usr/bin/python3"


def _no_mujoco_python():
    import subprocess
    if not os.path.exists(NO_MJ_PY):
        return None
    r = subprocess.run([NO_MJ_PY, "-c", "import mujoco"], capture_output=True)
    return NO_MJ_PY if r.returncode != 0 else None


class _Reset:
    def __init__(self):
        import NERO_V410_Household_Scenes as HS
        self.HS = HS
        self.td = tempfile.mkdtemp()

    def __enter__(self):
        HS = self.HS
        self.old = {k: os.environ.get(k) for k in (HS.ENV_RESET, HS.ENV_SPEC)}
        os.environ[HS.ENV_RESET] = os.path.join(self.td, "reset.json")
        os.environ[HS.ENV_SPEC] = "fixed:H1"
        return self.td

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class V4113(unittest.TestCase):
    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_u1_sim_python(self):
        import test_v4_11_2 as T2
        import NERO_V411_Sim_DOE as D
        st = T2._final_state()
        self.assertEqual(st["python"], sys.executable)
        self.assertTrue(any(os.path.exists(os.path.join(p, "mujoco")) for p in st["python_paths"]), st["python_paths"])
        if not T2._gl_ok():
            self.skipTest("no OpenGL backend here")
        bare = _no_mujoco_python()
        if not bare:
            self.skipTest("no Python without mujoco to reproduce the 11-17 failure")
        od = tempfile.mkdtemp()
        rn = D.DOERunner({"rows": [], "models": []}, {"start": None, "stop": None, "running": lambda: False}, od,
                         tempfile.mkdtemp())
        keep = D.sys.executable
        try:
            D.sys.executable = bare                           # the console's Python has no mujoco (11-17)
            for label, stt, ok in (("as 11-17", dict(st, python=None, python_paths=[]), False),
                                   ("sim paths", dict(st, python="/no/such/python"), True),
                                   ("sim python", dict(st), True)):
                r = {"id": label.replace(" ", "_"), "status": "success"}
                rn._safety_states = {}
                self.assertTrue(rn._start_snapshots(r, stt))
                rn.state = "snap"
                for _ in range(2400):
                    rn._snap_tick()
                    if rn.state == "next":
                        break
                    time.sleep(0.05)
                self.assertEqual(bool(r["snapshots"]), ok, (label, r.get("snapshot_why")))
                if not ok:
                    self.assertIn("mujoco", r["snapshot_why"])
        finally:
            D.sys.executable = keep

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_u2_safety_moment(self):
        import mujoco
        import numpy as np
        import NERO_V410_Household_Scenes as HS
        import NERO_V489_General_Pick_Place as GP
        import test_v4_11_2 as T2
        T2._final_state()
        m, d = T2._STATE["md"]
        with _Reset():

            class IO:
                msgs = []

                def reset_objects(self):
                    pass

                def emit(self, s):
                    self.msgs.append(s)
            io = IO()
            GP.CURRENT_RUN["id"] = None
            self.assertIsNone(GP._record_safety_moment(io, "execute_place_chunk", "lower", "apple_1"))
            GP.CURRENT_RUN["id"] = "doeX_r001"
            GP.TASK_STATS.clear()
            ks = [GP._record_safety_moment(io, "execute_place_chunk", "lower", "apple_1") for _ in range(5)]
            self.assertEqual(ks, [1, 2, 3, None, None])
            self.assertIsNone(GP._record_safety_moment(object(), "x", "y", None))     # not the GUI simulation
            self.assertEqual(len(GP.TASK_STATS["safety_records"]), 3)
            for _ in range(200):
                mujoco.mj_step(m, d)
            HS.safety_hook(m, d)
            with open(HS.safety_log_file()) as f:
                recs = [json.loads(x) for x in f if x.strip()]
            self.assertEqual([r["id"] for r in recs], ["doeX_r001_stop1", "doeX_r001_stop2", "doeX_r001_stop3"])
            r0 = recs[0]
            self.assertEqual(r0["info"]["tool"], "execute_place_chunk")
            self.assertTrue(r0["contacts"])                    # the objects rest on the table
            c = r0["contacts"][0]
            self.assertTrue(c["body1"].startswith("obj_") or c["body2"].startswith("obj_"))
            self.assertGreater(c["normal_force_n"], 0.0)
            self.assertEqual(len(r0["qpos"]), m.nq)
            self.assertTrue(r0["model_xml"] and r0["python"])
            self.assertFalse([p for p in os.listdir(os.path.dirname(HS.safety_log_file()))
                              if "_safety_doeX" in p])        # requests consumed
            self.assertTrue(np.isfinite(r0["max_abs_qvel"]))
        GP.CURRENT_RUN["id"] = None

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_u3_runner(self):
        import NERO_V410_Household_Scenes as HS
        import NERO_V411_Sim_DOE as D
        import test_v4_11_2 as T2
        st = T2._final_state()
        gl = T2._gl_ok()
        with _Reset() as td:
            od = tempfile.mkdtemp()
            clock = [1000.0]
            sim = {"up": False, "first": True}

            def start(spec, env):
                sim.update(up=True, t_ready=clock[0] + 3)
                return True, "ok"
            api = {"start": start, "stop": lambda: sim.update(up=False), "running": lambda: sim["up"]}

            def fake():
                if not sim["up"]:
                    return
                rp = os.path.join(td, "ready.json")
                if clock[0] >= sim["t_ready"] and not os.path.exists(rp):
                    with open(rp, "w") as f:
                        json.dump({}, f)
                    os.utime(rp, (clock[0], clock[0]))
                ib = os.path.join(td, "inbox")
                for n in sorted(os.listdir(ib)):
                    with open(os.path.join(ib, n)) as f:
                        msg = json.load(f)
                    os.remove(os.path.join(ib, n))
                    if sim["first"]:                    # the first attempt does not start (11-17 r011)
                        sim["first"] = False
                        out = {"id": msg["id"], "status": "not_started", "status_text": "Agent: BUSY | previous"}
                    else:
                        with open(HS.safety_log_file(), "a") as f:
                            f.write(json.dumps({"id": msg["id"] + "_stop1", "info": {"tool": "execute_place_chunk",
                                                                                     "step": "lower", "tool_call": 7},
                                                "t_request": 1.0, "t": 2.5, "held": ["obj_apple_1"],
                                                "contacts": [{"body1": "obj_apple_1", "geom1": "g", "body2":
                                                              "obj_bowl_1", "geom2": "obj_bowl_1_wall3",
                                                              "dist_mm": -2.4, "normal_force_n": 85.0}],
                                                "max_abs_qvel": 41.0, **{k: st[k] for k in (
                                                    "qpos", "model_xml", "python", "python_paths", "sim_time")}})
                                    + "\n")
                        out = {"id": msg["id"], "status": "complete", "stats": {
                            "tools": [{"tool": "execute_place_chunk", "ok": False, "reason": "SAFETY_STOP",
                                       "safety_stops": 2}], "safety_stops": 2,
                            "usage": {"input_tokens": 9000, "output_tokens": 1000}}}
                    with open(os.path.join(td, "outbox", msg["id"] + ".json"), "w") as f:
                        json.dump(out, f)
                ef = HS.eval_file()
                if os.path.exists(ef):
                    with open(ef) as f:
                        req = json.load(f)
                    os.remove(ef)
                    with open(ef + ".done", "w") as f:
                        json.dump(dict({"id": req["id"], "status": "fail", "n_ok": 0, "n_goals": 1, "objects": {},
                                        "goals": [{"goal": "apple_1 into bowl_1", "ok": False, "detail": "x"}]},
                                       **{k: st[k] for k in ("qpos", "model_xml", "python", "python_paths",
                                                             "sim_time")}), f)
            plan = {"rows": D.default_rows(["fixed:H1"]), "models": [D.parse_model("openai:m1")], "replicates": 1}
            rn = D.DOERunner(plan, api, od, td, now=lambda: clock[0], timeouts={"settle_s": 1, "retry_s": 3})
            rn.start()
            for _ in range(20000):
                if rn.finished:
                    break
                fake()
                rn.tick()
                clock[0] += 1.0
                if rn.state == "snap":
                    time.sleep(0.05)
            self.assertTrue(rn.finished)
            r = rn.runs[0]
            self.assertEqual((r["status"], r["klass"], r["retries"]), ("fail", "safety_stops", 1))
            self.assertIn("previous", r["first_attempt"]["status_text"])
            self.assertEqual(len(r["safety_records"]), 1)
            sr = r["safety_records"][0]
            self.assertEqual((sr["k"], sr["held"], sr["delay_s"]), ("1", ["obj_apple_1"], 1.5))
            self.assertNotIn("qpos", sr)                      # the state is only used for the picture
            if gl:
                self.assertEqual(len(r["snapshots"]), 2)
                self.assertEqual(sorted(os.path.basename(x) for x in r["stop_snapshots"]["stop1"]),
                                 ["r001_stop1_overview.png", "r001_stop1_top.png"])
            html = open(os.path.join(od, "runs.html"), encoding="utf-8").read()
            for k in ("安全停止 #1", "obj_bowl_1_wall3", "-2.4", "85.0"):
                self.assertIn(k, html)
            if gl:
                self.assertIn("snapshots/r001_stop1_overview.png", html)

    def test_u4_report(self):
        import NERO_V411_Sim_DOE as D
        runs = []
        for k, (st, tok) in enumerate((("success", 100000), ("success", 300000), ("fail", 200000),
                                       ("invalid", 400000))):
            runs.append({"id": f"r{k}", "row": 0, "scene": "fixed:H1", "rep": 1, "task": "t", "goals": "a into b",
                         "llm": {"provider": "openai", "model": "m", "effort": "high"}, "status": st, "klass": st,
                         "stats": {"usage": {"input_tokens": tok, "output_tokens": 0}}})
        g = D.summary_rows(runs, lambda r: "all")[0]
        self.assertAlmostEqual(g["rate"], 2 / 3)
        self.assertAlmostEqual(g["e2e_rate"], 2 / 4)
        self.assertAlmostEqual(g["tokens_per_run"], 250000)
        self.assertAlmostEqual(g["tokens_per_success"], 500000)
        md = D.report_md({"rows": [], "models": []}, runs)
        self.assertIn("端到端成功率", md)
        self.assertIn("| 67% |", md)
        self.assertIn("| 50% |", md)
        self.assertIn("250k | 500k |", md)
        self.assertAlmostEqual(D.estimate_s({"rows": D.default_rows(["fixed:H1"]),
                                             "models": [D.parse_model("a:m")], "replicates": 2}), 2 * 330 + 60)

    def test_u5_bridge_versions(self):
        import NERO_V410_Sim_Bridge as B
        import NERO_V411_Sim_DOE as D
        import NERO_V489_General_Pick_Place as GP
        self.assertEqual(B.NOT_STARTED_S, 45.0)
        self.assertIn("GP.CURRENT_RUN[\"id\"] = tid", open(os.path.join(HERE, "NERO_V410_Sim_Bridge.py")).read())
        self.assertIn("SAFETY-STOP RECORD", D.MARKER_V4113)
        self.assertIn("SAFETY-STOP RECORD", GP.MARKER_V4113)
        with open(os.path.join(HERE, "VERSION")) as f:
            self.assertIn(f.read().strip(), ("V4.11.3", "V4.12"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
