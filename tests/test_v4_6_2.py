#!/usr/bin/env python3
"""NERO V4.6.2 offline tests (teach bar in both task tabs, 「高级」 gravity compensation, full-width log + Copy Log).

P1 settings: NERO_CALIB_DIR/teach_gravity.json, defaults, bad values fall back, payload_arg (class / custom JSON).
P2 apply_teach_payload with a custom mass: sent only to an end-load call with NAMED mass parameters (kg + com in m);
   a class-only call such as Piper set_payload(load: str) is never fed a number -> the fallback class is sent and the
   log line says the custom mass was not sent; no load API -> [WARN] as in V3.70; plain classes unchanged.
P3 the mode helper gets --payload from the console's real cfg (class or custom JSON).
P4 「检测驱动接口」: cmd_probe lists load / payload calls with their parameters (fake pyAgxArm), probe_teach_apis parses it.
P5 wiring: both tabs build the same TeachBar; the console's 「示教负载」 box and 「高级」 share the one setting;
   VLA tab log below the monitors with Copy Log (layout checked with real Tk in tests/test_v4_6_2_gui.py).
P6 versions / markers.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ["NERO_CALIB_DIR"] = tempfile.mkdtemp(prefix="nero_v462_calib_")

import NERO_V313_Real_Robot as RR          # noqa: E402
import NERO_V462_Teach_Gravity as TG       # noqa: E402


class PiperLike:
    def __init__(self):
        self.calls = []

    def set_payload(self, load: str):
        self.calls.append(("set_payload", load))
        if load not in ("empty", "half", "full"):
            raise ValueError(load)
        return True


class MassArm:
    def __init__(self):
        self.calls = []

    def set_end_load(self, mass_kg, com_x=0.0, com_y=0.0, com_z=0.0):
        self.calls.append(("set_end_load", mass_kg, com_x, com_y, com_z))
        return True


class ComListArm:
    def __init__(self):
        self.calls = []

    def set_tcp_load(self, weight, cog):
        self.calls.append(("set_tcp_load", weight, list(cog)))
        return None


class NoApi:
    pass


class TestV462(unittest.TestCase):
    def setUp(self):
        try:
            os.remove(TG.settings_path())
        except OSError:
            pass

    def test_p1_settings(self):
        self.assertTrue(TG.settings_path().startswith(os.environ["NERO_CALIB_DIR"]))
        s = TG.load()
        self.assertEqual((s["mode"], s["payload"]), ("class", "empty"))
        self.assertEqual(TG.payload_arg(s), "empty")
        ok, where = TG.save({"mode": "custom", "payload": "half", "mass_kg": 0.42, "com_mm": [0, 5, 70]})
        self.assertTrue(ok, where)
        s = TG.load()
        d = json.loads(TG.payload_arg(s))
        self.assertEqual(d, {"mass_kg": 0.42, "com_mm": [0.0, 5.0, 70.0], "fallback": "half"})
        self.assertIn("0.420 kg", TG.describe(s))
        bad = TG.normalize({"mode": "x", "payload": "heavy", "mass_kg": 99, "com_mm": [1, 2]})
        self.assertEqual((bad["mode"], bad["payload"], bad["mass_kg"], bad["com_mm"]),
                         ("class", "empty", TG.DEFAULTS["mass_kg"], TG.DEFAULTS["com_mm"]))
        self.assertEqual(TG.normalize({"payload": "不设置"})["payload"], "none")
        self.assertEqual(TG.payload_arg({"payload": "none"}), "none")
        seen = []
        TG.add_listener(seen.append)
        TG.save({"payload": "full"})
        self.assertEqual(seen[-1]["payload"], "full")
        TG._listeners.remove(seen.append)
        open(TG.settings_path(), "w").write("{broken")
        self.assertEqual(TG.load()["payload"], "empty")

    def test_p2_custom_payload(self):
        js = json.dumps({"mass_kg": 0.35, "com_mm": [0, 0, 60], "fallback": "empty"})
        # piper-like class call: never fed a number -> fallback class + note
        r = PiperLike()
        rep = RR.apply_teach_payload(r, js)
        self.assertEqual(r.calls, [("set_payload", "empty")])
        self.assertTrue(rep["ok"])
        self.assertIn("没有发送", rep["custom_note"])
        line = RR.teach_payload_line(rep)
        self.assertTrue(line.startswith("[OK]") and "改发档位 empty" in line, line)
        # named mass + x/y/z parameters -> kg and metres
        m = MassArm()
        rep = RR.apply_teach_payload(m, js)
        self.assertEqual(m.calls, [("set_end_load", 0.35, 0.0, 0.0, 0.06)])
        self.assertTrue(rep["ok"] and rep["custom"] and rep["api"] == "set_end_load")
        self.assertIn("0.350 kg", RR.teach_payload_line(rep))
        c = ComListArm()
        rep = RR.apply_teach_payload(c, js)
        self.assertEqual(c.calls, [("set_tcp_load", 0.35, [0.0, 0.0, 0.06])])
        self.assertTrue(rep["ok"])
        # no API at all -> WARN as in V3.70
        rep = RR.apply_teach_payload(NoApi(), js)
        self.assertFalse(rep["ok"])
        self.assertTrue(RR.teach_payload_line(rep).startswith("[WARN]"))
        # classes unchanged
        r = PiperLike()
        self.assertEqual(RR.apply_teach_payload(r, "half")["api"], "set_payload")
        self.assertEqual(r.calls, [("set_payload", "half")])
        self.assertIsNone(RR.apply_teach_payload(r, "none")["payload"])
        self.assertFalse(RR.apply_teach_payload(r, "{bad json")["ok"])

    def test_p3_helper_argv(self):
        seen = {}

        class P:
            stdout, stderr, returncode = '{"event": "mode_result", "ok": true}\n', "", 0

        def fake_run(argv, **k):
            seen["argv"] = argv
            return P()
        import subprocess as sp
        orig = sp.run
        sp.run = fake_run
        try:
            fake = types.SimpleNamespace(log=lambda s: None, EXIT_SPEED=0.1, _fw_cfg=lambda: "v121")
            for pay in ("half", TG.payload_arg({"mode": "custom", "mass_kg": 0.3})):
                fake.cfg_fn = lambda pay=pay: {"can_port": "can0", "teach_payload": pay}
                ev = RR.RealPipeline._run_mode_helper(fake, True, 5.0)
                self.assertTrue(ev["ok"])
                a = seen["argv"]
                self.assertEqual(a[a.index("--payload") + 1], pay)
        finally:
            sp.run = orig

    def test_p4_probe(self):
        class Arm(MassArm, PiperLike):
            def __init__(self):
                self.calls = []

            def connect(self):
                pass

            def set_leader_mode(self):
                """enter leader mode"""

        fake = types.ModuleType("pyAgxArm")
        fake.create_agx_arm_config = lambda **k: k
        fake.AgxArmFactory = types.SimpleNamespace(create_arm=lambda cfg: Arm())
        fake.ArmModel = types.SimpleNamespace(NERO="nero")
        fake.NeroFW = types.SimpleNamespace(V121="v121", DEFAULT="d")
        sys.modules["pyAgxArm"] = fake
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = RR.cmd_probe("can0", "v121")
        finally:
            del sys.modules["pyAgxArm"]
        ev = [json.loads(l) for l in buf.getvalue().splitlines() if l.startswith("{")]
        pr = [e for e in ev if e.get("event") == "probe"]
        self.assertEqual(rc, 0, ev)
        names = [a["name"] for a in pr[0]["payload_apis"]]
        self.assertEqual(sorted(names), ["set_end_load", "set_payload"])
        self.assertTrue(pr[0]["custom_ok"])
        self.assertIn("mass_kg", [a for a in pr[0]["payload_apis"] if a["name"] == "set_end_load"][0]["sig"])
        txt = TG.probe_text(pr[0])
        self.assertIn("「自定义质量」可用", txt)
        self.assertIn("没有负载", TG.probe_text({"ok": True, "apis": [], "payload_apis": []}))
        self.assertIn("检测失败", TG.probe_text({"ok": False, "error": "x"}))

        class P:
            stdout = json.dumps(pr[0]) + "\n"
            stderr, returncode = "", 0
        import subprocess as sp
        orig = sp.run
        sp.run = lambda argv, **k: P()
        try:
            fake_pipe = types.SimpleNamespace(cfg_fn=lambda: {"can_port": "can0"}, _fw_cfg=lambda: "v121")
            r = RR.RealPipeline.probe_teach_apis(fake_pipe)
        finally:
            sp.run = orig
        self.assertTrue(r["ok"] and r["custom_ok"])

    def test_p5_wiring(self):
        src = lambda f: open(os.path.join(HERE, f), encoding="utf-8").read()
        rct, vla, con = src("NERO_V365_Real_Cup_Task.py"), src("NERO_V4_Console_Tab.py"), src("NERO_V312_Console.py")
        for s in (rct, vla):
            self.assertIn("TG.TeachBar(", s)
            self.assertIn('text="Copy Log", command=self.copy_log', s)
            self.assertIn("safe_tab_fn=lambda: self.safe_tab", s)
        self.assertIn('bb.pack(side="bottom", fill="x")', rct)
        self.assertIn('logbar.pack(side="bottom", fill="x")', vla)
        self.assertNotIn("logcol", vla)
        self.assertIn('"teach_payload": TG.payload_arg(TG.load())}', con)
        self.assertIn("TG.add_listener(_payload_from_dialog)", con)
        self.assertEqual(con.count("safe_tab=safe_tab"), 2)
        tg = src("NERO_V462_Teach_Gravity.py")
        for t_ in ('text="进入示教模式"', 'text="退出示教模式"', 'text="高级"', "保存并立即生效", "检测驱动接口"):
            self.assertIn(t_, tg)
        self.assertIn("return_home=False", tg)                  # exit holds the pose like the 实机NERO调试 button
        import re
        self.assertFalse(re.search("[\U0001F300-\U0001FAFF]", tg))   # no emoji (Tk / Xft crash, V4.5.1)

    def test_p6_markers(self):
        import NERO_V312_Console as C
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertIn(v, ("V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2"))
        self.assertEqual(C.VERSION, v)
        self.assertEqual(RR.VERSION, v)
        self.assertEqual(TG.MARKER_V462, "V4.6.2 TEACH BAR + GRAVITY COMP")
        import NERO_V365_Real_Cup_Task as RT
        self.assertEqual(RT.MARKER_V462, "V4.6.2 TEACH BAR + GRAVITY COMP + FULL-WIDTH LOG")


if __name__ == "__main__":
    unittest.main(verbosity=1)
