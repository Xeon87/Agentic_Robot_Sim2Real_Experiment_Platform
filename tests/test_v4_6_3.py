#!/usr/bin/env python3
"""NERO V4.6.3 offline tests (native crash guard after the V4.6.2 abort inside Tk during an egg task).

H1 v463_xinitthreads: 'ok' with libX11, 'off' with NERO_NO_XINITTHREADS=1; called before tk.Tk() in run_console.
H2 start_nero.sh: stderr copied to logs/stderr.log (tee, survives an abort of the console process), glibc fatal
   messages to stderr (LIBC_FATAL_STDERR_=1), size rotation; the exact exec line keeps a simulated abort's message.
H3 real_cfg reads Tk variables on the main thread only (snapshot for worker threads; refreshed in real_poll and
   before every job) - checked with a running console in tests/test_v4_6_3_gui.py.
H4 versions / markers.
H5 an X error no longer ends the console: written to crash.log, the process keeps running.
"""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)


class TestV463(unittest.TestCase):
    def test_h1_xinitthreads(self):
        import NERO_V312_Console as C
        old = os.environ.pop("NERO_NO_XINITTHREADS", None)
        try:
            r = C.v463_xinitthreads()
            self.assertTrue(r == "ok" or r.startswith("unavailable"), r)
            os.environ["NERO_NO_XINITTHREADS"] = "1"
            self.assertEqual(C.v463_xinitthreads(), "off")
        finally:
            os.environ.pop("NERO_NO_XINITTHREADS", None)
            if old is not None:
                os.environ["NERO_NO_XINITTHREADS"] = old
        src = open(os.path.join(HERE, "NERO_V312_Console.py"), encoding="utf-8").read()
        self.assertLess(src.index("xinit = v463_xinitthreads()"), src.index("root = tk.Tk()"))

    def test_h5_x_error_survived(self):
        """an X error (here: BadPixmap from freeing a pixmap that does not exist) is written to crash.log and the
        process keeps running (Xlib's default handler would exit without any trace)"""
        if not os.environ.get("DISPLAY"):
            self.skipTest("no DISPLAY")
        d = tempfile.mkdtemp(prefix="nero_v463_x_")
        code = (f"import sys; sys.path.insert(0, {HERE!r})\n"
                "import ctypes, ctypes.util, NERO_V312_Console as C\n"
                f"C.install_crash_log({d!r})\n"
                "print(C.v463_x_error_handlers())\n"
                "x = ctypes.CDLL(ctypes.util.find_library('X11'))\n"
                "x.XOpenDisplay.restype = ctypes.c_void_p\n"
                "dpy = ctypes.c_void_p(x.XOpenDisplay(None))\n"
                "x.XFreePixmap(dpy, ctypes.c_ulong(0x1234567)); x.XSync(dpy, 0)\n"
                "print('ALIVE')\n")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        self.assertIn("ALIVE", r.stdout, r.stdout + r.stderr)
        txt = open(os.path.join(d, "crash.log"), encoding="utf-8").read()
        self.assertIn("V4.6.3 X error #1 IGNORED", txt)
        self.assertIn("BadPixmap", txt)

    def test_h2_stderr_capture(self):
        sh = open(os.path.join(HERE, "start_nero.sh"), encoding="utf-8").read()
        self.assertIn("export LIBC_FATAL_STDERR_=1", sh)
        line = [l for l in sh.splitlines() if l.startswith('exec "$PY" "$CONSOLE"')][0]
        self.assertIn('2> >(tee -a "$HERE/logs/stderr.log" >&2)', line)
        self.assertIn("stderr.prev.log", sh)
        d = tempfile.mkdtemp(prefix="nero_v463_")
        os.makedirs(os.path.join(d, "logs"))
        fake = os.path.join(d, "fake_console.py")
        open(fake, "w").write("import os, sys\nsys.stderr.write('Tcl_Panic simulated: alloc: invalid block\\n')\n"
                              "sys.stderr.flush()\nos.abort()\n")
        script = os.path.join(d, "run.sh")
        open(script, "w").write(f'#!/bin/bash\nHERE="{d}"\nPY="{sys.executable}"\nCONSOLE="{fake}"\nARGS=()\n'
                                + line + "\n")
        subprocess.run(["bash", script], stderr=subprocess.DEVNULL, timeout=30)
        import time
        for _ in range(50):
            p = os.path.join(d, "logs", "stderr.log")
            if os.path.exists(p) and "Tcl_Panic simulated" in open(p).read():
                break
            time.sleep(0.1)
        self.assertIn("Tcl_Panic simulated", open(os.path.join(d, "logs", "stderr.log")).read())

    def test_h3_cfg_snapshot_source(self):
        src = open(os.path.join(HERE, "NERO_V312_Console.py"), encoding="utf-8").read()
        self.assertIn("if threading.current_thread() is not threading.main_thread():", src)
        self.assertIn("real_cfg()                                # V4.6.3: refresh the snapshot", src)
        self.assertIn("real_cfg()                                    # V4.6.3: fresh snapshot", src)
        self.assertIn('"teach_payload": v_payload.get()', src)        # V3.70 source assertion kept

    def test_h4_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        v = open(os.path.join(HERE, "VERSION")).read().strip()
        self.assertEqual(v, "V4.6.3")
        self.assertEqual((C.VERSION, RR.VERSION), (v, v))
        self.assertEqual(C.MARKER_V463, "V4.6.3 NATIVE CRASH GUARD")


if __name__ == "__main__":
    unittest.main(verbosity=1)
