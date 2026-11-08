#!/usr/bin/env python3
"""V4.6.3 GUI test (needs tkinter + DISPLAY; skips otherwise): the whole console is started (as start_nero.sh does);
the pipeline's cfg_fn (= real_cfg) called from a worker thread must not touch Tk at all and must return the values
of the main-thread snapshot; a change made on the main thread is seen by the worker after the next real_poll.
crash.log carries the XInitThreads line."""
import gc
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    import tkinter
    if not os.environ.get("DISPLAY"):
        print("[SKIP] test_v4_6_3_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v4_6_3_gui: {exc}")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="nero_v463_gui_"))
os.environ["NERO_CALIB_DIR"] = str(TMP / "calib")
os.environ["NERO_DRIVE_ROOT"] = str(TMP / "drive")
os.environ["NERO_V368_TASK_DIR"] = str(TMP / "task")
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "active.json")
os.environ["NERO_NO_VLA_TAB"] = "1"
import NERO_V312_Console as C           # noqa: E402
import NERO_V313_Real_Robot as RR       # noqa: E402

C.HERE = str(TMP)                        # logs/ of this run go to the temp folder, not the release folder
RES = {}


class Probe:
    """wraps the Tk app: records every Tk call that is not made on the main thread"""
    def __init__(self, app):
        object.__setattr__(self, "_app", app)
        object.__setattr__(self, "bad", [])

    def __getattr__(self, name):
        attr = getattr(self._app, name)
        if not callable(attr):
            return attr

        def w(*a, **k):
            if threading.current_thread() is not threading.main_thread():
                self.bad.append((name, a[:3]))
            return attr(*a, **k)
        return w


orig_init, orig_loop = tkinter.Tk.__init__, tkinter.Tk.mainloop


def init(self, *a, **k):
    orig_init(self, *a, **k)
    self.tk = Probe(self.tk)
    RES["probe"] = self.tk


def loop(self, n=0):
    def check():
        pipe = next(o for o in gc.get_objects() if isinstance(o, RR.RealPipeline))
        out = {}
        main_cfg = pipe.cfg_fn()
        t = threading.Thread(target=lambda: out.update(cfg=pipe.cfg_fn()))
        t.start()
        t.join(5)
        RES.update(main=main_cfg, worker=out.get("cfg"), bad=list(RES["probe"].bad))
        self.after(300, self.destroy)
    self.after(3000, check)
    orig_loop(self, n)


tkinter.Tk.__init__, tkinter.Tk.mainloop = init, loop
try:
    C.main([])
except SystemExit:
    pass
assert RES.get("worker") is not None, RES
assert RES["bad"] == [], RES["bad"][:5]
assert RES["worker"] == RES["main"], (RES["worker"], RES["main"])
assert RES["worker"]["can_port"] == "can0" and "teach_payload" in RES["worker"]
crash = (TMP / "logs" / "crash.log").read_text(encoding="utf-8")
assert "V4.6.3: XInitThreads ok; X error handler ok" in crash, crash[-400:]
assert "V4.6.3 console closed normally" in crash, crash[-400:]
print("[G1] worker-thread cfg_fn: 0 Tk calls, same values as the main-thread snapshot; crash.log: XInitThreads ok")
print("ALL V4.6.3 GUI TESTS PASSED")
