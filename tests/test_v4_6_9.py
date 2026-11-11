#!/usr/bin/env python3
"""NERO V4.6.9 offline tests.

R1 plan_real_replay: a recording (7 joints + gripper norm) -> timed joint-path segments cut at gripper switches.
R2 replay_episode_real on a fake RealPipeline + fake SO-101: refused in teach mode; 10 % move to the recorded start;
   one move_joint_path per segment starting at the MEASURED pose; gripper open / close between segments; a failed
   segment is logged and stops the replay; an almost-still (mock) recording gets a warning.
L1 Try harder (LLM): extra instructions, failed tool results lose "stop_and_report" and carry suggestions, an early
   final text is answered with a push (max TRY_HARDER_PUSHES), no push without try harder.
L2 cup place: the target position measured before the carry is restored after the occluded re-checks; the wrist
   concentricity correction is capped (source checks + _pin_target).
E1 egg verify: one more egg inside the carton whose nearest slot is the target counts; an egg nearer a neighbour
   slot does not.
E2 egg pick: a clamp gap wider than 36 mm (long axis) is warned.
D1 sparse depth: the carton template still finds 6 slots within 3 mm at 10 % depth coverage; at 2 % it falls back
   to the outer box and STILL saves a diagnostic PNG; depth coverage is in the template info / logs.
V1 Smol VLA tab: 5 sub-pages, E-STOP in the top bar, gripper teach drives the real gripper, space key wiring.
V2 versions / markers.
"""
from __future__ import annotations

import json
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
os.environ["NERO_CALIB_DIR"] = tempfile.mkdtemp(prefix="nero_v469_calib_")
os.environ.pop("NERO_EGG_CARTON_GRID", None)

import NERO_V4_Dataset_Recorder as DR        # noqa: E402
import NERO_V349_Real_Wrist_Camera as W      # noqa: E402
import v466_synth_carton as SY               # noqa: E402


def src(f):
    with open(os.path.join(HERE, f), encoding="utf-8") as fh:
        return fh.read()


def recording(n=60, grip_switch=(20, 40)):
    st = np.zeros((n, 8))
    for i in range(n):
        st[i, :7] = np.radians([10 + i * 0.5, -5 + i * 0.3, 0, 60 - i * 0.2, 0, 10, i * 0.1])
        st[i, 7] = 1.0 if (i < grip_switch[0] or i >= grip_switch[1]) else 0.0
    return st


class FakePipe:
    def __init__(self, fail_seg=None, teach=False):
        self.teach_mode = teach
        self.calls = []
        self.q_deg = [0.0] * 7
        self.fail_seg = fail_seg

    def move_and_settle(self, model, q, speed, label=""):
        self.calls.append(("settle", list(q), speed))
        self.q_deg = list(np.degrees(q))
        return {"ok": True}

    def move_joint_path(self, model, path, times, label=""):
        self.calls.append(("path", [list(p) for p in path], list(times)))
        n = sum(1 for c in self.calls if c[0] == "path")
        if self.fail_seg == n:
            return {"ok": False, "error": "trajectory aborted"}
        self.q_deg = list(np.degrees(path[-1]))
        return {"ok": True}

    def actual_arm_deg(self, model):
        return list(self.q_deg)


class FakeGrip:
    connected = True

    def __init__(self):
        self.ev = []

    def open_gap(self, mm):
        self.ev.append(("open", mm))
        return {"ok": True}

    def clamp(self, pct):
        self.ev.append(("clamp", pct))
        return {"ok": True}


class FakeDS:
    def __init__(self, st, fps=20.0):
        self.st, self.fps = st, fps

    def load_episode(self, idx):
        return {"observation_state": self.st, "meta": {"fps": self.fps}}


class TestV469(unittest.TestCase):
    # ------------------------------------------------------------------ replay
    def test_r1_plan(self):
        st = recording()
        segs = DR.plan_real_replay(st, 20.0, speed=1.0)
        self.assertEqual([s["grip_after"] for s in segs], ["close", "open", None])
        for s in segs:
            self.assertEqual(s["t"][0], 0.0)
            self.assertTrue(all(b > a for a, b in zip(s["t"], s["t"][1:])), s["t"])
            self.assertEqual(len(s["path"]), len(s["t"]))
        np.testing.assert_allclose(segs[0]["path"][0], st[0, :7])
        np.testing.assert_allclose(segs[-1]["path"][-1], st[-1, :7])
        slow = DR.plan_real_replay(st, 20.0, speed=0.5)
        self.assertAlmostEqual(slow[0]["t"][-1], 2 * segs[0]["t"][-1], places=2)
        self.assertEqual(DR.plan_real_replay(st[:1], 20.0), [])

    def test_r2_replay_real(self):
        st = recording()
        logs = []
        r = DR.replay_episode_real(FakeDS(st), 0, FakePipe(teach=True), "nero", log=logs.append)
        self.assertFalse(r["ok"])
        self.assertIn("示教模式", logs[-1])
        p, g, logs = FakePipe(), FakeGrip(), []
        r = DR.replay_episode_real(FakeDS(st), 0, p, "nero", gripper=g, speed=1.0, log=logs.append)
        self.assertTrue(r["ok"], logs)
        self.assertEqual(p.calls[0][0], "settle")
        self.assertEqual(p.calls[0][2], DR.REPLAY_START_SPEED)
        paths = [c for c in p.calls if c[0] == "path"]
        self.assertEqual(len(paths), 3)
        for c in paths:                                  # each path starts at the measured pose
            self.assertEqual(c[2][0], 0.0)
        self.assertEqual([e[0] for e in g.ev], ["open", "clamp", "open"])
        self.assertEqual(r["segments"], 3)
        self.assertLess(r["max_end_err_deg"], 0.01)
        # a failed segment stops and is logged
        p, logs = FakePipe(fail_seg=2), []
        r = DR.replay_episode_real(FakeDS(st), 0, p, "nero", gripper=FakeGrip(), log=logs.append)
        self.assertFalse(r["ok"])
        self.assertTrue(any("[FAIL] [V4.6.9 REPLAY] 第 2/3 段失败" in s for s in logs), logs)
        # a mock recording (arm still) -> warning
        still = np.zeros((40, 8))
        still[:, 7] = 1.0
        still[:, 0] = np.radians(np.linspace(0, 0.3, 40))
        logs = []
        DR.replay_episode_real(FakeDS(still), 0, FakePipe(), "nero", log=logs.append)
        self.assertTrue(any("几乎没动" in s for s in logs), logs)
        logs = []
        self.assertFalse(DR.replay_episode_real(FakeDS(st), 0, None, None, log=logs.append)["ok"])

    # ------------------------------------------------------------------ try harder (LLM)
    def _agent(self, responses):
        import threading
        seen = []

        class A:
            _lock = threading.Lock()
            _running = False
            _stop_event = threading.Event()
            _last_response_id = None

            def emit(self, s):
                pass

            def api_key_status(self, k, m):
                return True, "ok"

            def _post_responses(self, payload, key):
                seen.append(payload)
                return responses.pop(0) if responses else {"id": "x", "text": "done", "calls": []}

            def _log_usage(self, r, m):
                pass

            def _function_calls(self, r):
                return r.get("calls", [])

            def _extract_text(self, r):
                return r.get("text", "")
        return A(), seen

    def _skills(self):
        class S:
            skill = {"code": "t"}
            k = {}
            stage = ""

            def run_tool(self, name, args):
                if name == "pick_cup":
                    return {"ok": False, "tool": name, "reason": "GRASP_SLIPPED",
                            "recommended_next_actions": ["stop_and_report"]}
                return {"ok": True, "tool": name}
        return S()

    def test_l1_try_harder(self):
        import NERO_V362_Cup_Skills as CK
        call = lambda n, i: {"id": f"r{i}", "calls": [{"name": n, "arguments": "{}", "call_id": f"c{i}"}]}
        for th in (False, True):
            agent, seen = self._agent([call("pick_cup", 1), {"id": "r2", "text": "I give up", "calls": []},
                                       {"id": "r3", "text": "still no", "calls": []}])
            logs = []
            CK.run_cup_task(agent, self._skills(), "Stack cup_1 into cup_2", "m", "k", "low", 20, emit=logs.append,
                            try_harder=th)
            self.assertEqual("TRY HARDER MODE" in seen[0]["instructions"], th)
            out = json.loads(seen[1]["input"][0]["output"])
            if th:
                self.assertNotIn("stop_and_report", out["recommended_next_actions"])
                self.assertTrue(out["try_harder"]["suggestions"])
                self.assertIn("pick_cup", out["try_harder"]["failed_tool"])
                pushes = [s for s in logs if "[V4.6.9 TRY HARDER]" in s]
                self.assertEqual(len(pushes), CK.TRY_HARDER_PUSHES, logs)
                self.assertIn("TRY HARDER: the task is NOT finished", seen[2]["input"][0]["content"])
            else:
                self.assertEqual(out["recommended_next_actions"], ["stop_and_report"])
                self.assertEqual(len(seen), 2)
        s = CK.try_harder_suggestions("plan_cup_stack", {"source_cup": "cup_1", "target_cup": "cup_2"}, {})
        self.assertIn("source_cup='cup_2', target_cup='cup_1'", s["suggestions"][0])

    def test_l2_cup_place_pin(self):
        import NERO_V362_Cup_Skills as CK
        logs = []
        cups = {"cup_2": {"base": np.array([-0.40, 0.10, 0.0]), "rim": np.array([-0.40, 0.10, 0.09])}}
        fake = types.SimpleNamespace(_cup=lambda n: cups[n], io=types.SimpleNamespace(emit=logs.append))
        pin = {k: v.copy() for k, v in cups["cup_2"].items()}
        cups["cup_2"]["base"][:] = [-0.38, 0.12, 0.0]                 # re-check drifted 28 mm (occluded)
        CK.CupSkills._pin_target(fake, "cup_2", pin)
        np.testing.assert_allclose(cups["cup_2"]["base"], pin["base"])
        self.assertTrue(any("沿用搬运前测得的位置" in s for s in logs))
        s = src("NERO_V362_Cup_Skills.py")
        self.assertEqual(s.count("self._pin_target(target_cup, t_pin)"), 2)
        self.assertIn("step_mm > WRIST_PLACE_STEP_MAX_MM or applied_mm + step_mm > WRIST_PLACE_TOTAL_MAX_MM", s)
        self.assertEqual((CK.WRIST_PLACE_STEP_MAX_MM, CK.WRIST_PLACE_TOTAL_MAX_MM), (10.0, 12.0))

    # ------------------------------------------------------------------ eggs
    def test_e1_verify_by_count(self):
        import test_v4_6_8 as T8
        sc = T8.Scene()
        sk, io = T8.make(sc)
        sk.observe_scene()
        sc.verify_plan = [{"eggs": [[-0.392, 0.142, 0.04]]}]          # 22 mm from slot_3 -> nearest slot = slot_3
        r = sk.verify_egg_in_slot("slot_3", looks=1)
        self.assertTrue(r["ok"])
        self.assertTrue(any("蛋托里的鸡蛋从 0 个变成 1 个" in s for s in io.logs), io.logs)
        self.assertEqual(r["physical_verification"]["looks"][0]["eggs_in_carton"], 1)
        sk, io = T8.make(T8.Scene())
        sk.observe_scene()
        sk.io.perceive_eggs.__self__.verify_plan = [{"eggs": [[-0.392, 0.150, 0.04]]}]   # nearer slot_6
        r = sk.verify_egg_in_slot("slot_3", looks=1)
        self.assertFalse(r["ok"])
        self.assertFalse(any("变成" in s for s in io.logs))

    def test_e2_long_axis(self):
        import NERO_V365_Real_Cup_Task as RT
        self.assertEqual(RT.EGG_LONG_AXIS_GAP_MM, 36.0)
        s = src("NERO_V365_Real_Cup_Task.py")
        self.assertTrue("float(gap_clamp) > EGG_LONG_AXIS_GAP_MM" in s
                        or "float(gap_clamp) > egg_long_axis_gap_mm()" in s)   # V4.8.1: per gripper
        self.assertIn("[WARN] [V4.6.9 EGG PICK]", s)

    # ------------------------------------------------------------------ sparse depth
    def _sparse(self, sc, keep, seed=1):
        d = np.asarray(sc["depth"]).copy()
        rng = np.random.default_rng(seed)
        d[rng.random(d.shape) > keep] = 0
        return d

    def test_d1_sparse_depth(self):
        sc = SY.scene(yaw_deg=20.0)
        rows, diag = W.locate_egg_holder(sc["dets"], sc["rgb"], self._sparse(sc, 0.10), sc["K"], sc["T"], table_z=0.0)
        sl = [r for r in rows if str(r["det_id"]).startswith("slot_")]
        self.assertEqual(len(sl), 6)
        e = [min(np.hypot(s["xyz_m"][0] - tx, s["xyz_m"][1] - ty) for s in sl) * 1000 for tx, ty in sc["truth"]]
        self.assertLess(max(e), 3.0, e)                                  # random 10 % of the pixels
        t = diag["template"]
        self.assertLess(t["depth_cov"], 0.2)
        self.assertGreater(t["inner_pts"], W.TEMPLATE_MIN_PTS)
        rows, diag = W.locate_egg_holder(sc["dets"], sc["rgb"], self._sparse(sc, 0.02), sc["K"], sc["T"], table_z=0.0)
        t = diag["template"]
        self.assertIn("外框中心", t["how"])
        self.assertIn("%", t["how"])
        self.assertTrue(t.get("diag_png") and os.path.isfile(t["diag_png"]), t)       # V4.6.9: fallback picture too
        self.assertIn("蛋托轮廓深度覆盖", src("NERO_V365_Real_Cup_Task.py"))
        self.assertIn('out["depth_cov"]', src("NERO_V366_Real_Cup_Yolo.py"))
        self.assertIn("深度覆盖", src("NERO_V366_Real_Cup_Yolo.py"))

    # ------------------------------------------------------------------ Smol VLA tab
    def test_v1_vla_layout(self):
        s = src("NERO_V4_Console_Tab.py")
        for t in ('"① 准备"', '"② 示教采集"', '"③ 训练"', '"④ 执行"', '"⑤ 评估 / 高级"'):
            self.assertIn(t, s)
        i_es, i_nb = s.index('text="■ 紧急急停 (E-STOP)"'), s.index("self.sub_nb = ttk.Notebook(parent)")
        self.assertLess(i_es, i_nb)                                        # E-STOP above the sub-pages
        self.assertIn('bind_all("<KeyPress-space>", self._on_space', s)
        self.assertIn("replay_episode_real(self.dataset, idx, self.pipe, model", s)
        import NERO_V4_Console_Tab as VT
        g = FakeGrip()
        rec = types.SimpleNamespace(v=[], set_teach_gripper_norm=None)

        def stn(v, drive_gripper_fn=None):
            rec.v.append(v)
            if drive_gripper_fn:
                drive_gripper_fn(v)
        rec.set_teach_gripper_norm = stn
        var = lambda x: types.SimpleNamespace(get=lambda: x, set=lambda *_: None)
        logs = []
        fake = types.SimpleNamespace(v_backend=var("real"), rct_tab=types.SimpleNamespace(grip=g), recorder=rec,
                                     v_grip_teach=var(""), _log=logs.append)
        fake._real_gripper = lambda: VT.VLALeRobotTabUI._real_gripper(fake)
        import threading
        orig = threading.Thread

        class Now:
            def __init__(self, target=None, **k):
                self.t = target

            def start(self):
                self.t()
        VT.threading.Thread = Now
        try:
            VT.VLALeRobotTabUI.on_teach_gripper(fake, False)
            VT.VLALeRobotTabUI.on_teach_gripper(fake, True)
            fake.v_backend = var("mock")
            VT.VLALeRobotTabUI.on_teach_gripper(fake, False)
        finally:
            VT.threading.Thread = orig
        self.assertEqual(g.ev, [("clamp", DR.REPLAY_CLAMP_PCT), ("open", DR.REPLAY_OPEN_GAP_MM)])
        self.assertEqual(rec.v, [0.0, 1.0, 0.0])
        self.assertTrue(any("只记录数值" in s_ for s_ in logs), logs)

    def test_v2_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        import NERO_V365_Real_Cup_Task as RT
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual((C.VERSION, RR.VERSION), (v, v))
        self.assertIn(v, ("V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3", "V4.8.4"))
        self.assertEqual(RT.MARKER_V469, "V4.6.9 VERIFY BY EGG COUNT + LONG-AXIS WARNING + DEPTH COVERAGE")
        self.assertEqual(DR.MARKER_V469_REPLAY, "V4.6.9 REAL REPLAY")


if __name__ == "__main__":
    unittest.main(verbosity=1)
