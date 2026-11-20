#!/usr/bin/env python3
"""NERO V4.11.2 offline tests (user 11-18: 「每个任务有最终判定失败或者成功时刻的截图吗？我需要后面追溯」).

S1 (mujoco) the physics side: the truth check answer carries the final state (model saved once next to the scene
   file with mj_saveLastXML, qpos, simulation time); a model that cannot be saved is reported, not fatal.
S2 (mujoco + an OpenGL backend; skipped without one) the renderer: separate process, overview / top pictures (+ the
   D435i camera when the model has it) as valid PNG files of the requested size; a wrong qpos length is refused.
S3 the DOE runner with a fake simulation whose truth check carries a real final state: every run gets its pictures
   (or the reason why not), runs.html shows pictures + goal check + tool calls (arguments) + LLM text, report.md
   links the pictures; without a final state the run is still judged.
S4 TASK_STATS keeps the tool arguments, a result excerpt and the LLM's text per round.
S5 versions / markers.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTS = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("NERO_CALIB_DIR", tempfile.mkdtemp(prefix="nero_v4112_"))

try:
    import mujoco  # noqa: F401
    HAS_MJ = True
except Exception:
    HAS_MJ = False

_STATE = {}


def _final_state():
    """household H1 model (test robot stand-in) after 0.3 s -> the truth check's snapshot fields"""
    if "st" in _STATE:
        return _STATE["st"]
    import mujoco
    import NERO_V410_Household_Scenes as HS
    from test_v4_10_2 import _Env, _xml
    HS._SNAP_MODEL.update(path=None, why=None)
    with _Env("household_pick_place", "fixed:H1"):
        m = mujoco.MjModel.from_xml_path(_xml(tempfile.mkdtemp()))
        d = mujoco.MjData(m)
        for _ in range(300):
            mujoco.mj_step(m, d)
        _STATE["st"] = HS.snapshot_state(m, d)
        _STATE["md"] = (m, d)
    return _STATE["st"]


def _gl_ok():
    """an OpenGL backend for the renderer in this environment (tried once)"""
    if "gl" not in _STATE:
        import NERO_V411_Snapshot as SN
        st = _final_state()
        out = tempfile.mkdtemp()
        res = SN.snapshot(st["model_xml"], st["qpos"], os.path.join(out, "probe"), size="320x240", timeout_s=120)
        _STATE["gl"] = res
    return bool(_STATE["gl"].get("ok"))


def _png_size(p):
    with open(p, "rb") as f:
        b = f.read(24)
    assert b[:8] == b"\x89PNG\r\n\x1a\n", p
    return int.from_bytes(b[16:20], "big"), int.from_bytes(b[20:24], "big")


class V4112(unittest.TestCase):
    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_s1_final_state(self):
        import mujoco
        import NERO_V410_Household_Scenes as HS
        st = _final_state()
        self.assertTrue(st["model_xml"] and os.path.exists(st["model_xml"]), st)
        self.assertEqual(len(st["qpos"]), st["nq"])
        self.assertGreater(st["sim_time"], 0.29)
        m2 = mujoco.MjModel.from_xml_path(st["model_xml"])          # the saved model loads, same layout
        self.assertEqual(m2.nq, st["nq"])
        m, d = _STATE["md"]
        old = {k: os.environ.get(k) for k in (HS.ENV_RESET, HS.ENV_SPEC)}
        td = tempfile.mkdtemp()
        os.environ[HS.ENV_RESET] = os.path.join(td, "reset.json")
        os.environ[HS.ENV_SPEC] = "fixed:H1"
        try:
            f = HS.request_eval([("apple_1", "into", "bowl_1")], rid="s1")
            HS.eval_hook(m, d)
            with open(f + ".done") as fh:
                ans = json.load(fh)
            self.assertEqual((ans["id"], ans["nq"], len(ans["qpos"])), ("s1", m.nq, m.nq))
            self.assertEqual(ans["model_xml"], st["model_xml"])
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        saved = dict(HS._SNAP_MODEL)
        try:
            HS._SNAP_MODEL.update(path=None, why=None)
            import NERO_V343_Scenes as SC
            keep = SC.LAST_SCENE_XML["path"]
            SC.LAST_SCENE_XML["path"] = None
            st2 = HS.snapshot_state(m, d)
            SC.LAST_SCENE_XML["path"] = keep
            self.assertIsNone(st2["model_xml"])
            self.assertIn("unknown", st2["model_why"])
        finally:
            HS._SNAP_MODEL.update(saved)

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_s2_renderer(self):
        import NERO_V411_Snapshot as SN
        if not _gl_ok():
            self.skipTest(f"no OpenGL backend here: {_STATE['gl'].get('why')}")
        st = _final_state()
        out = tempfile.mkdtemp()
        res = SN.snapshot(st["model_xml"], st["qpos"], os.path.join(out, "r001_success"), size="480x360")
        self.assertTrue(res["ok"], res)
        names = sorted(os.path.basename(x) for x in res["files"])
        self.assertEqual(names, ["r001_success_overview.png", "r001_success_top.png"])
        for x in res["files"]:
            self.assertEqual(_png_size(x), (480, 360))
            self.assertGreater(os.path.getsize(x), 5000)
        bad = SN.snapshot(st["model_xml"], st["qpos"][:-1], os.path.join(out, "bad"), size="160x120")
        self.assertFalse(bad["ok"])
        self.assertIn("qpos", bad["why"])

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_s3_runner_trace(self):
        import NERO_V410_Household_Scenes as HS
        import NERO_V411_Sim_DOE as D
        st = _final_state()
        gl = _gl_ok()
        td, od = tempfile.mkdtemp(), tempfile.mkdtemp()
        old = os.environ.get(HS.ENV_RESET)
        os.environ[HS.ENV_RESET] = os.path.join(td, "reset.json")
        clock = [1000.0]
        sim = {"up": False}

        def start(spec, env):
            sim.update(up=True, t_ready=clock[0] + 5)
            return True, "started"
        api = {"start": start, "stop": lambda: sim.update(up=False), "running": lambda: sim["up"]}

        def fake_sim():
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
                    m = json.load(f)
                os.remove(os.path.join(ib, n))
                if n.startswith("stop_"):
                    continue
                stats = {"tools": [{"step": 1, "tool": "perceive_objects", "ok": True, "safety_stops": 0,
                                    "args": {"prompts": ["apple"]}, "result": '{"count": 4}'},
                                   {"step": 2, "tool": "plan_place_chunk", "ok": False, "reason": "PLACE_COLLISION",
                                    "safety_stops": 0, "args": {"target_id_or_xyz": "bowl_1"}, "result": "{}"}],
                         "texts": [{"round": 1, "text": "I will grasp the apple <first>."}], "final": "done",
                         "safety_stops": 0}
                with open(os.path.join(td, "outbox", m["id"] + ".json"), "w") as f:
                    json.dump({"id": m["id"], "status": "complete", "stats": stats, "llm": m["llm"]}, f)
            ef = HS.eval_file()
            if os.path.exists(ef):
                with open(ef) as f:
                    req = json.load(f)
                os.remove(ef)
                second = req["id"].endswith("r002")
                ans = {"id": req["id"], "status": "fail" if second else "success", "n_ok": 0 if second else 1,
                       "n_goals": 1, "objects": {}, "held": [],
                       "goals": [{"goal": "apple_1 into bowl_1", "ok": not second, "detail": "12 mm from the axis"}]}
                if not second:                    # the second run has no final state (model not saved)
                    ans.update({k: st[k] for k in ("model_xml", "qpos", "nq", "sim_time")})
                with open(ef + ".done", "w") as f:
                    json.dump(ans, f)
        try:
            plan = {"rows": D.default_rows(["fixed:H1"]), "models": [D.parse_model("openai:m1")], "replicates": 2}
            rn = D.DOERunner(plan, api, od, td, now=lambda: clock[0], timeouts={"settle_s": 1, "snap_s": 300})
            rn.start()
            import time
            for _ in range(20000):
                if rn.finished:
                    break
                fake_sim()
                rn.tick()
                clock[0] += 1.0
                if rn.state == "snap":
                    time.sleep(0.05)               # the renderer is a real process
            self.assertTrue(rn.finished, rn.state)
            r1, r2 = rn.runs
            self.assertEqual((r1["status"], r2["status"]), ("success", "fail"))
            if gl:
                self.assertEqual(sorted(os.path.basename(x) for x in r1["snapshots"]),
                                 ["r001_success_overview.png", "r001_success_top.png"])
                for x in r1["snapshots"]:
                    self.assertTrue(os.path.exists(os.path.join(od, x)))
            else:
                self.assertEqual(r1["snapshots"], [])
                self.assertTrue(r1["snapshot_why"])
            self.assertEqual(r2["snapshots"], [])
            self.assertIn("no final state", r2["snapshot_why"])
            html = open(os.path.join(od, "runs.html"), encoding="utf-8").read()
            for k in ("r001", "r002", "apple_1 into bowl_1", "PLACE_COLLISION", "target_id_or_xyz",
                      "I will grasp the apple &lt;first&gt;.", "12 mm from the axis", "没有截图"):
                self.assertIn(k, html)
            if gl:
                self.assertIn("<img src='snapshots/r001_success_overview.png'", html)
                md = open(os.path.join(od, "report.md"), encoding="utf-8").read()
                self.assertIn("[overview](snapshots/r001_success_overview.png)", md)
            p2, runs2 = D.load_runs(od)
            self.assertEqual(runs2[0]["snapshots"], r1["snapshots"])
        finally:
            if old is None:
                os.environ.pop(HS.ENV_RESET, None)
            else:
                os.environ[HS.ENV_RESET] = old

    @unittest.skipUnless(HAS_MJ, "mujoco")
    def test_s4_task_stats_trace(self):
        import NERO_V410_Household_Scenes as HS
        import NERO_V489_General_Pick_Place as GP

        class Agent:
            def __init__(self, script):
                self.script = list(script)
                self._lock = threading.Lock()
                self._running = False
                self._stop_event = threading.Event()
                self._last_response_id = None
                self._usage_totals = {}

            def api_key_status(self, k, m):
                return True, "ok"

            def _post_responses(self, payload, key):
                return self.script.pop(0)

            def _log_usage(self, resp, model):
                pass

            def _function_calls(self, resp):
                return resp.get("calls", [])

            def _extract_text(self, resp):
                return resp.get("text", "")

        io = HS.make_twin(HS.fixed_scene("H1"))
        call = {"name": "perceive_objects", "arguments": json.dumps({"prompts": ["apple"], "use_finetuned": True}),
                "call_id": "c1"}
        ag = Agent([{"id": "1", "text": "Looking first.", "calls": [call]}, {"id": "2", "text": "Done."}])
        GP.run_general_pick_place(io, "t", agent=ag, llm_cfg={"model": "m", "api_key": "k"})
        t = GP.TASK_STATS["tools"][0]
        self.assertEqual(t["args"], {"prompts": ["apple"], "use_finetuned": True})
        self.assertTrue(t["result"].startswith("{") and len(t["result"]) <= 600)
        self.assertEqual([x["text"] for x in GP.TASK_STATS["texts"]], ["Looking first.", "Done."])

    def test_s5_versions(self):
        import NERO_V411_Sim_DOE as D
        import NERO_V411_Snapshot as SN
        import NERO_V489_General_Pick_Place as GP
        self.assertIn("SNAPSHOTS", D.MARKER_V4112)
        self.assertIn("SNAPSHOTS", GP.MARKER_V4112)
        self.assertIn(SN.VERSION, ("V4.11.2", "V4.11.3", "V4.12"))
        with open(os.path.join(HERE, "VERSION")) as f:
            self.assertIn(f.read().strip(), ("V4.11.2", "V4.11.3", "V4.12"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
