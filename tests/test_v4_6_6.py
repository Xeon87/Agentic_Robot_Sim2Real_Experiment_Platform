#!/usr/bin/env python3
"""NERO V4.6.6 offline tests.

C1 config/egg_carton.json: the user's 2x3 carton, pitch 46 mm (3-hole direction) x 47 mm (2-hole direction);
   NERO_EGG_CARTON_GRID overrides ("none" = old zero-prior guess).
C2 the V4.6.5 failure reproduced on a synthetic oblique view: a 2x3 carton whose mask carries a 30 mm strip of extra
   material (outer box 162 x 148 mm, like the real 150-167 x 128-149 mm) -> the old zero-prior guess makes a 3x3
   grid (9 slots, 3 of them on the strip); the fixed template finds 6 slots within 2 mm at three carton directions.
C3 slot numbering is stable: columns (3-hole direction) count away from the robot base; turning the carton by 180
   degrees gives the same slot positions for the same numbers.
C4 bowl score: a cup centre scores high, the carton edge / the bare table / the flat strip do not.
C5 one diagnostic PNG per fit under NERO_CALIB_DIR/carton_diag (newest 30 kept), path in the template info.
C6 native crash trace (tools/nero_crashtrace.c, built with gcc): a segfault writes the C stack to the log file;
   the console reports it ON and removes LD_PRELOAD for child processes.  (skipped without gcc)
C7 Tk input method OFF by default (NERO_TK_IM=1 keeps it).  (skipped without DISPLAY)
C8 vision-check direction hint uses 8 directions (V4.6.5 printed "+X = 画面左, +Y = 画面左").
C9 versions / markers.
"""
from __future__ import annotations

import glob
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTS = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["NERO_CALIB_DIR"] = tempfile.mkdtemp(prefix="nero_v466_calib_")
os.environ.pop("NERO_EGG_CARTON_GRID", None)

import NERO_V349_Real_Wrist_Camera as W      # noqa: E402
import v466_synth_carton as SY               # noqa: E402


def slots_of(rows):
    return [r for r in rows if str(r["det_id"]).startswith("slot_")]


def errs(rows, truth):
    sl = slots_of(rows)
    return np.array([min(math.hypot(s["xyz_m"][0] - tx, s["xyz_m"][1] - ty) for s in sl) * 1000.0 for tx, ty in truth])


class TestV466(unittest.TestCase):
    def test_c1_config(self):
        c = W.carton_template_config()
        self.assertEqual((c["rows"], c["cols"], c["pitch_cols_mm"], c["pitch_rows_mm"]), (2, 3, 46.0, 47.0))
        os.environ["NERO_EGG_CARTON_GRID"] = "none"
        try:
            self.assertIsNone(W.carton_template_config())
        finally:
            os.environ.pop("NERO_EGG_CARTON_GRID")
        self.assertEqual(W.MARKER_V466_TEMPLATE, "V4.6.6 FIXED CARTON TEMPLATE")

    def test_c2_template_vs_zero_prior(self):
        for yaw in (20.0, 87.0, -9.0):
            sc = SY.scene(yaw_deg=yaw, extra_mm=30.0)
            rows, diag = W.locate_egg_holder(sc["dets"], sc["rgb"], sc["depth"], sc["K"], sc["T"], table_z=0.0)
            self.assertEqual(len(slots_of(rows)), 6, yaw)
            e = errs(rows, sc["truth"])
            self.assertLess(float(e.max()), 2.0, (yaw, e))
            t = diag["template"]
            self.assertEqual(t["n_cavities"], 6)
            self.assertGreater(t["outer_box_mm"][1], 140.0)                       # the extra strip is in the box
            os.environ["NERO_EGG_CARTON_GRID"] = "none"
            try:
                rows2, d2 = W.locate_egg_holder(sc["dets"], sc["rgb"], sc["depth"], sc["K"], sc["T"], table_z=0.0)
            finally:
                os.environ.pop("NERO_EGG_CARTON_GRID")
            self.assertEqual((d2["rows"], d2["cols"]), (3, 3), yaw)                # the V4.6.5 behaviour
            self.assertEqual(len(slots_of(rows2)), 9)

    def test_c3_numbering(self):
        pos = {}
        for yaw in (20.0, 200.0):
            sc = SY.scene(yaw_deg=yaw)
            rows, _d = W.locate_egg_holder(sc["dets"], sc["rgb"], sc["depth"], sc["K"], sc["T"], table_z=0.0)
            pos[yaw] = {s["det_id"]: np.asarray(s["xyz_m"][:2]) for s in slots_of(rows)}
            d = {k: float(np.linalg.norm(v)) for k, v in pos[yaw].items()}
            self.assertGreater(d["slot_3"], d["slot_1"])                           # columns count away from the base
            self.assertGreater(d["slot_6"], d["slot_4"])
        for k in pos[20.0]:
            self.assertLess(float(np.linalg.norm(pos[20.0][k] - pos[200.0][k])) * 1000.0, 2.5, k)

    def test_c4_bowl_score(self):
        u, v = np.meshgrid(np.arange(-110.0, 110.0, 1.0), np.arange(-90.0, 90.0, 1.0))
        h = SY.carton_height(u, v, extra_mm=30.0)
        ok = ~np.isnan(h)
        bw = W.bowl_score_map(u[ok], v[ok], h[ok], 46.0)

        def at(x, y):
            return float(bw["S"][int(round((y - bw["y0"]) / bw["res"])), int(round((x - bw["x0"]) / bw["res"]))])
        cup = at(-46.0, -23.5)
        self.assertGreater(cup, 15.0)
        for x, y in ((-81.0, -59.0), (0.0, -59.0), (0.0, 70.0), (-46.0, 0.0)):   # corner, edge, strip, between cups
            self.assertLess(at(x, y), 4.0, (x, y, at(x, y)))

    def test_c5_diag_png(self):
        sc = SY.scene(yaw_deg=20.0)
        rows, diag = W.locate_egg_holder(sc["dets"], sc["rgb"], sc["depth"], sc["K"], sc["T"], table_z=0.0)
        p = diag["template"]["diag_png"]
        self.assertTrue(p and os.path.isfile(p), diag["template"])
        self.assertTrue(p.startswith(os.path.join(os.environ["NERO_CALIB_DIR"], "carton_diag")))
        self.assertFalse(p.startswith(HERE))
        for _ in range(3):
            W.locate_egg_holder(sc["dets"], sc["rgb"], sc["depth"], sc["K"], sc["T"], table_z=0.0)
        self.assertLessEqual(len(glob.glob(os.path.join(os.path.dirname(p), "carton_*.png"))), W.CARTON_DIAG_KEEP)

    @unittest.skipUnless(shutil.which("gcc"), "gcc not available")
    def test_c6_native_trace(self):
        d = tempfile.mkdtemp(prefix="nero_v466_ct_")
        so = os.path.join(d, "nero_crashtrace_test.so")
        subprocess.run(["gcc", "-shared", "-fPIC", "-O1", "-o", so, os.path.join(HERE, "tools", "nero_crashtrace.c")],
                       check=True)
        log = os.path.join(d, "native_crash.log")
        env = dict(os.environ, LD_PRELOAD=so, NERO_NATIVE_CRASH_LOG=log)
        code = (f"import sys, os; sys.path.insert(0, {HERE!r}); import NERO_V312_Console as C\n"
                "print(C.v466_native_trace_status()); print('PRELOAD' in os.environ.get('LD_PRELOAD', 'x') or "
                "'nero_crashtrace' in os.environ.get('LD_PRELOAD', ''))\n"
                "import ctypes; ctypes.string_at(0)\n")
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("ON -> " + log, r.stdout)
        self.assertIn("False", r.stdout)                         # LD_PRELOAD removed for child processes
        txt = open(log).read()
        self.assertIn("NERO native crash: signal 11", txt)
        self.assertIn("libc.so", txt)
        sh = open(os.path.join(HERE, "start_nero.sh"), encoding="utf-8").read()
        self.assertIn('export NERO_NATIVE_CRASH_LOG="$HERE/logs/native_crash.log"', sh)
        self.assertIn("NERO_NO_NATIVE_TRACE", sh)

    @unittest.skipUnless(os.environ.get("DISPLAY"), "no DISPLAY")
    def test_c7_tk_input_methods(self):
        import tkinter
        import NERO_V312_Console as C
        r = tkinter.Tk()
        try:
            os.environ.pop("NERO_TK_IM", None)
            self.assertEqual(C.v466_tk_input_methods(r), "OFF")
            self.assertEqual(int(r.tk.call("tk", "useinputmethods", "-displayof", r._w)), 0)
            os.environ["NERO_TK_IM"] = "1"
            self.assertTrue(C.v466_tk_input_methods(r).startswith("ON"))
        finally:
            os.environ.pop("NERO_TK_IM", None)
            r.destroy()

    def test_c8_axis_hint(self):
        import NERO_V464_Vision_Check as VC
        sc = SY.scene(yaw_deg=20.0)
        cam = {"K": sc["K"], "T_base_cam": sc["T"].tolist()}
        h = VC.axis_hint(cam, [-0.38, 0.06, 0.02])
        a, b = [x.split("画面")[1] for x in h.split("：")[1].split("，")]
        self.assertNotEqual(a, b, h)
        top = {"K": {"fx": 900.0, "fy": 900.0, "ppx": 640.0, "ppy": 360.0},
               "T_base_cam": [[1.0, 0, 0, -0.40], [0, -1.0, 0, 0.0], [0, 0, -1.0, 0.75], [0, 0, 0, 1.0]]}
        self.assertEqual(VC.axis_hint(top, [-0.40, 0.0, 0.04]), "主相机画面里：+X = 画面右，+Y = 画面上")

    def test_c9_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        import NERO_V365_Real_Cup_Task as RT
        v = open(os.path.join(HERE, "VERSION")).read().strip()
        self.assertIn(v, ("V4.6.6", "V4.6.8"))
        self.assertEqual((C.VERSION, RR.VERSION), (v, v))
        self.assertEqual(C.MARKER_V466, "V4.6.6 NATIVE CRASH TRACE + TK INPUT METHOD OFF")
        self.assertEqual(RT.MARKER_V466, "V4.6.6 FIXED CARTON TEMPLATE LOG")


if __name__ == "__main__":
    unittest.main(verbosity=1)
