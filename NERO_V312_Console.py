#!/usr/bin/env python3
"""NERO V3.12 console: the first window ./start_nero.sh opens.

V3.12 CONSOLE: nothing starts automatically any more.
  * "仿真 (MuJoCo)" tab  -> button starts start_nero_sim.sh, which is the unchanged V3.11 launch chain
    (GPU server -> backend/MuJoCo + sim GUI -> BioIK MoveIt + RViz -> TRAC-IK). Step lamps follow its
    status lines. The sim GUI (Agent tab etc.) still opens as its own window, exactly as before.
  * "实机NERO调试" tab  -> V3.13: six-step real-robot pipeline (NERO_V313_Real_Robot.py), live CAN/ROS status, e-stop.
    (V3.25: renamed from "实机 NERO")
  * "Camera调试" tab    -> V3.25: D435i live view, depth check, eye-to-hand calibration, visual locating
    (NERO_V325_Camera.py)
  * Sim and real robot are mutually exclusive (both would publish /feedback/joint_states).

Only the standard library is used (tkinter, subprocess, threading). The step tracker and the process
runner have no Tk dependency so tests can drive them headless.
"""
import math
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time

VERSION = "V4.6.4"
# V3.59 SKILL STRATEGY KNOBS TAB + SKILL CARD + MULTI-POINT HOVER DRY-RUN
# V3.60 SIM ENV INJECTION OF THE APPLIED SKILL + SKILL TAB SYNC BETWEEN CONSOLE AND SIM GUI
# V3.61 STACK CUPS SCENE + COLLAPSIBLE SKILL TAB
# V3.51 WORKFLOW STATUS LAMP BAR ABOVE LOG
# V3.52 TEACH MODE BUTTONS LEFT OF COPY LOG
# V3.53 TEACH EXIT & PARK FIX + ALIGNED TEACH BUTTONS + DEFAULT 1280x720 DEPTH
# V3.54 FIRMWARE V1.2.1 (v121 DEFAULT) + FOLLOWER MODE TEACH EXIT + STEP 3 PUSH RECOVERY
# V3.55 FAST MAIN CAMERA HAND-EYE CALIB + UNIFIED PNP/DEPTH SOURCE + LOO OUTLIER REJECTION
# V3.56 FAST WRIST EYE-IN-HAND CALIB + LOOK-AT DOME PLANNER + FULL-RES ROI ARUCO + SCALE/LEVER-ARM SOLVER
# V3.57 AUTO TABLETOP CHARUCO INTRINSICS + CHARUCO EIH + 2D FOCAL/SCALE SELF-CALIB
# V3.58 KINEMATIC-ANCHORED WIDE-ANGLE INTRINSICS + FULL-QUADRANT TILT + ROBUST CHARUCO EIH
HERE = os.path.dirname(os.path.realpath(__file__))
SIM_SCRIPT = os.environ.get("NERO_V312_SIM_SCRIPT", os.path.join(HERE, "start_nero_sim.sh"))

# V3.44 SCENE SELECT: the MuJoCo scene is chosen here before the simulation starts and passed to
# start_nero_sim.sh as NERO_SCENE.  Every console start defaults to the egg scene.
SIM_SCENES = [("fill_egg_holder", "fill_egg_holder - 鸡蛋放入蛋托 (默认)"),
              ("store_in_safe", "store_in_safe - 物体放入保险箱 (RoboDojo)"),
              ("stack_cups", "stack_cups - 叠杯子 (RoboDojo)"),          # V3.61 STACK CUPS SCENE
              ("stack_cups_so101", "stack_cups_so101 - 叠杯子 · SO-101 夹爪 (7 mm 转接环, 偏转 45°)"),  # V3.64 SO-101 GRIPPER ON NERO
              ("sort_items", "sort_items - 物品分类放入红框 / 蓝框 · SO-101 夹爪（香蕉 / 苹果 / 鼠标 / 马克笔）")]  # V3.72 SORT ITEMS SCENE
DEFAULT_SIM_SCENE = "fill_egg_holder"


def clean_child_env(env=None):
    """V3.46 QT ENV GUARD (same as NERO_V313_Real_Robot.clean_child_env): no cv2/qt entries in a child's Qt
    variables - pip opencv-python sets them on import and rviz2 then aborts."""
    e = dict(os.environ if env is None else env)
    for k in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        v = e.get(k)
        if v and "cv2" in v:
            keep = [x for x in v.split(os.pathsep) if x and "cv2" not in x]
            if keep:
                e[k] = os.pathsep.join(keep)
            else:
                e.pop(k, None)
    return e


def scene_env(scene):
    """Environment additions for one simulation start (pure logic, tested)."""
    names = [k for k, _ in SIM_SCENES]
    if scene not in names:
        raise ValueError(f"unknown scene {scene!r}; known: {names}")
    return {"NERO_SCENE": scene}
STOP_GRACE_S = float(os.environ.get("NERO_V312_STOP_GRACE_S", "25"))

# --------------------------------------------------------------------------- sim step tracker
GREY, YELLOW, GREEN, RED, AMBER = "grey", "yellow", "green", "red", "amber"

# (key, label, regex that turns the step green, regex that turns it amber = optional/warn)
SIM_STEPS = [
    ("integrity", "1 完整性/编译", r"\[OK\] py_compile PASS", None),
    ("stale", "2 残留进程", r"V3\.07 pre-launch check: no stale|V3\.07 stale processes stopped", None),
    ("gpu", "3 GPU / cuRobo", r"cuRobo pose self-test PASS", None),
    ("modules", "4 模块解析", r"All NERO modules resolve inside this folder", None),
    ("backend", "5 后端 + MuJoCo", r"ROS topic READY: /feedback/joint_states", None),
    ("bioik", "6 BioIK MoveIt + RViz", r"Primary /move_group IK solver verified: BioIK", None),
    ("trac", "7 TRAC-IK 参考(可选)", r"Reference /trac_ref/move_group IK solver verified", r"continuing BioIK-only"),
    ("ready", "8 SYSTEM READY", r"\[READY\] SYSTEM READY", None),
]
FAIL_RE = re.compile(r"\[FAIL\]|START FAILED")


class SimStepTracker:
    """Turns start_nero_sim.sh output lines into per-step lamp states. Pure logic, no Tk."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.state = {k: GREY for k, *_ in SIM_STEPS}
        self.ready = False
        self.failed = False
        self.fail_line = ""

    def start(self):
        self.reset()
        self.state[SIM_STEPS[0][0]] = YELLOW

    def current(self):
        for k, *_ in SIM_STEPS:
            if self.state[k] == YELLOW:
                return k
        return None

    def _advance_from(self, idx):
        for k, *_ in SIM_STEPS[idx + 1:]:
            if self.state[k] == GREY:
                self.state[k] = YELLOW
                return

    def feed(self, line):
        for i, (k, _label, ok_re, warn_re) in enumerate(SIM_STEPS):
            if self.state[k] in (GREEN, AMBER):
                continue
            if re.search(ok_re, line):
                # anything still pending before a later milestone was evidently passed
                # (a non-fatal [FAIL] line followed by recovery is cleared the same way)
                for kk, *_ in SIM_STEPS[:i]:
                    if self.state[kk] in (GREY, YELLOW, RED):
                        self.state[kk] = AMBER if kk == "trac" else GREEN
                self.failed = False
                self.state[k] = GREEN
                if k == "ready":
                    self.ready = True
                else:
                    self._advance_from(i)
                return
            if warn_re and re.search(warn_re, line) and self.state[k] == YELLOW:
                self.state[k] = AMBER
                self._advance_from(i)
                return
        if FAIL_RE.search(line) and not self.failed:
            self.failed = True
            self.fail_line = line.strip()
            cur = self.current()
            if cur:
                self.state[cur] = RED

    def process_exited(self, rc):
        cur = self.current()
        if cur and not self.ready:
            self.state[cur] = RED
            self.failed = True


# --------------------------------------------------------------------------- process runner
class SimRunner:
    """Runs start_nero_sim.sh in its own session; lines go to a queue. Pure logic, no Tk."""

    def __init__(self, script=SIM_SCRIPT, log_path=None):
        self.script = script
        self.proc = None
        self.lines = queue.Queue()
        self.log_path = log_path
        self._stop_requested = False
        self.exit_code = None
        self.strategy_env = {}               # V3.60 SIM ENV INJECTION: set by the console before start()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, extra_env=None):
        if self.running():
            raise RuntimeError("simulation already running")
        env = clean_child_env()
        env["NERO_NO_FAIL_PAUSE"] = "1"      # never block on "Press Enter" when started from the console
        env["PYTHONUNBUFFERED"] = "1"
        env.update(extra_env or {})
        env.update(getattr(self, "strategy_env", None) or {})     # V3.60: applied skill -> backend env (only diffs)
        self._stop_requested = False
        self.exit_code = None
        self.proc = subprocess.Popen(
            ["bash", self.script], cwd=os.path.dirname(os.path.realpath(self.script)),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, start_new_session=True, bufsize=1, text=True, errors="replace")
        reader = threading.Thread(target=self._reader, args=(self.proc,), daemon=True, name="v312_sim_reader")
        reader.start()
        threading.Thread(target=self._watcher, args=(self.proc, reader), daemon=True, name="v312_sim_watch").start()
        return self.proc.pid

    def _reader(self, proc):
        logf = None
        if self.log_path:
            try:
                logf = open(self.log_path, "a", encoding="utf-8")
            except Exception:
                logf = None
        try:
            for line in proc.stdout:
                self.lines.put(line.rstrip("\n"))
                if logf:
                    logf.write(line)
                    logf.flush()
        except Exception:
            pass
        if logf:
            logf.close()

    def _watcher(self, proc, reader):
        # The launcher's exit is the end of the run.  A leftover grandchild may still hold the stdout pipe
        # open, so do not wait for EOF: give the reader 2 s to flush the last lines, then report the exit.
        rc = proc.wait()
        reader.join(timeout=2.0)
        self.exit_code = rc
        self.lines.put(None)                   # sentinel: launcher finished

    def stop(self):
        """SIGINT to the launcher only -> its trap runs cleanup (stops MoveIt, backend, GPU server and
        restores kinematics.yaml). Escalates to SIGKILL of the whole session after STOP_GRACE_S."""
        if not self.running():
            return
        self._stop_requested = True
        try:
            os.kill(self.proc.pid, signal.SIGINT)
        except ProcessLookupError:
            return
        threading.Thread(target=self._escalate, args=(self.proc,), daemon=True).start()

    def _escalate(self, proc):
        t_end = time.time() + STOP_GRACE_S
        while time.time() < t_end:
            if proc.poll() is not None:
                return
            time.sleep(0.2)
        self.lines.put(f"[V3.12 CONSOLE] launcher still alive after {STOP_GRACE_S:.0f}s -> SIGKILL session")
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass

    def stop_requested(self):
        return self._stop_requested


# --------------------------------------------------------------------------- GUI
LAMP_COLORS = {GREY: "#9e9e9e", YELLOW: "#f2c037", GREEN: "#2e7d32", RED: "#c62828", AMBER: "#ef8f00"}



CJK_FAMILIES = ("Noto Sans CJK SC", "Noto Sans CJK", "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
                "Droid Sans Fallback", "AR PL UMing CN", "Source Han Sans SC")


def _v312_pick_cjk_font(root):
    """Use a Chinese-capable family for the Tk default fonts when one is installed (labels are Chinese)."""
    try:
        import tkinter.font as tkfont
        fams = set(tkfont.families(root))
        fam = next((f for f in CJK_FAMILIES if f in fams), None)
        if fam:
            for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
                tkfont.nametofont(name, root).configure(family=fam)
    except Exception:
        pass


# V3.70 CRASH LOG: a hard crash (segfault in Tk / X11 / a C extension) used to kill the console together with its
# terminal and left nothing behind.  faulthandler writes the Python stack of every thread into logs/crash.log at the
# moment of the crash; uncaught Python exceptions (main thread, worker threads, Tk callbacks) are appended there too.
_V370_CRASH = {"fh": None, "path": None}


def v370_crash_write(title, exc_type, exc, tb):
    import traceback
    path = _V370_CRASH.get("path")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n===== [V3.70 CRASH LOG] {time.strftime('%Y-%m-%d %H:%M:%S')} {VERSION} pid {os.getpid()} "
                    f"· {title} =====\n")
            f.write("".join(traceback.format_exception(exc_type, exc, tb)))
    except Exception:
        pass


def v370_crash_note(text):
    """V4.6.3: one info line into the open crash.log (below the start line)."""
    fh = _V370_CRASH.get("fh")
    if fh is None:
        return
    try:
        fh.write(f"[info] {text}\n")
        fh.flush()
    except Exception:
        pass


def install_crash_log(log_dir, root=None):
    import faulthandler
    import threading as _th
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "crash.log")
    if _V370_CRASH["fh"] is None or _V370_CRASH["path"] != path:
        fh = open(path, "a", encoding="utf-8")         # kept open for the whole run: faulthandler needs a live fd
        fh.write(f"----- [V3.70 CRASH LOG] console start {time.strftime('%Y-%m-%d %H:%M:%S')} {VERSION} "
                 f"pid {os.getpid()} (nothing below this line = no crash) -----\n")
        fh.flush()
        faulthandler.enable(file=fh, all_threads=True)
        _V370_CRASH.update(fh=fh, path=path)
        prev_sys = sys.excepthook

        def _sys_hook(t, e, tb):
            v370_crash_write("uncaught exception (main thread)", t, e, tb)
            prev_sys(t, e, tb)
        sys.excepthook = _sys_hook
        prev_th = _th.excepthook

        def _th_hook(a):
            v370_crash_write(f"uncaught exception in thread {getattr(a.thread, 'name', '?')}",
                             a.exc_type, a.exc_value, a.exc_traceback)
            prev_th(a)
        _th.excepthook = _th_hook
    if root is not None:
        def _tk_hook(t, e, tb):
            v370_crash_write("exception in a Tk callback", t, e, tb)
            import traceback
            traceback.print_exception(t, e, tb)
        root.report_callback_exception = _tk_hook
    return path


# V4.6.3 NATIVE CRASH GUARD: the V4.6.2 console aborted inside Tk (C level, main thread) while an egg task ran.
# (1) Xlib is made thread-safe BEFORE Tk opens the display (XInitThreads; libX11 >= 1.8 does this by itself, Ubuntu
#     22.04 ships 1.7.5) - an Xlib call from any second thread then cannot corrupt the connection and abort the
#     process ("[xcb] Unknown sequence number ... Assertion failed").  NERO_NO_XINITTHREADS=1 switches it off.
# (2) start_nero.sh copies stderr into logs/stderr.log (with LIBC_FATAL_STDERR_=1): the one-line C reason that is
#     printed right before such an abort (Tcl / Xlib / glibc) was so far only on the terminal.
MARKER_V463 = "V4.6.3 NATIVE CRASH GUARD"


def v463_xinitthreads():
    """-> 'ok' / 'off' / reason string; must run before the first Xlib call of the process (before tk.Tk())."""
    if os.environ.get("NERO_NO_XINITTHREADS", "0") == "1":
        return "off"
    try:
        import ctypes
        import ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")
        return "ok" if lib.XInitThreads() else "XInitThreads returned 0"
    except Exception as exc:
        return f"unavailable: {exc}"


_V463_X = {"lib": None, "handlers": None, "n": 0}
X_ERROR_LOG_MAX = 50


def v463_x_error_handlers():
    """V4.6.3: Xlib's default X error handler EXITS the process (no signal -> faulthandler writes nothing: the
    V4.6.2 console vanished without a trace when the Smol VLA tab was opened).  Installed BEFORE tk.Tk(), Tk chains
    unhandled X errors to this handler: the error is written to crash.log and the console keeps running.
    A lost display connection (X IO error) cannot be survived; it is written to crash.log before Xlib exits.
    -> 'ok' / 'off' / reason"""
    if os.environ.get("NERO_NO_X_ERROR_HANDLER", "0") == "1":
        return "off"
    try:
        import ctypes
        import ctypes.util
        lib = ctypes.CDLL(ctypes.util.find_library("X11") or "libX11.so.6")

        class XErrorEvent(ctypes.Structure):
            _fields_ = [("type", ctypes.c_int), ("display", ctypes.c_void_p), ("resourceid", ctypes.c_ulong),
                        ("serial", ctypes.c_ulong), ("error_code", ctypes.c_ubyte), ("request_code", ctypes.c_ubyte),
                        ("minor_code", ctypes.c_ubyte)]
        ERR = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(XErrorEvent))
        IOERR = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)

        def on_error(_dpy, ev):
            try:
                _V463_X["n"] += 1
                if _V463_X["n"] <= X_ERROR_LOG_MAX:
                    e = ev.contents
                    names = {1: "BadRequest", 2: "BadValue", 3: "BadWindow", 4: "BadPixmap", 5: "BadAtom",
                             8: "BadMatch", 9: "BadDrawable", 11: "BadAlloc", 13: "BadGC", 16: "BadLength"}
                    v370_crash_note(f"V4.6.3 X error #{_V463_X['n']} IGNORED (console keeps running): "
                                    f"{names.get(e.error_code, e.error_code)} request {e.request_code}.{e.minor_code} "
                                    f"serial {e.serial} resource 0x{e.resourceid:x}"
                                    + (" - request 139 = RENDER: font / glyph drawing (Xft)" if e.request_code == 139 else ""))
            except Exception:
                pass
            return 0

        def on_io_error(_dpy):
            v370_crash_note("V4.6.3 X IO error: connection to the display lost -> Xlib ends the process")
            return 0
        h = (ERR(on_error), IOERR(on_io_error))
        lib.XSetErrorHandler.argtypes, lib.XSetErrorHandler.restype = [ERR], ctypes.c_void_p
        lib.XSetIOErrorHandler.argtypes, lib.XSetIOErrorHandler.restype = [IOERR], ctypes.c_void_p
        lib.XSetErrorHandler(h[0])
        lib.XSetIOErrorHandler(h[1])
        _V463_X.update(lib=lib, handlers=h)                     # keep the callbacks alive for the whole run
        return "ok"
    except Exception as exc:
        return f"unavailable: {exc}"


LOG_KEEP = 10


def v46_rotate_console_log(log_dir, console_log, keep=LOG_KEEP):
    """V4.6 LOG ROTATION: the previous run's console.log becomes logs/console_<its start time>.log (newest `keep`
    kept) and is also copied to console.prev.log. V4.5 kept only one previous log - two restarts lost a run."""
    import shutil
    try:
        if os.path.isfile(console_log) and os.path.getsize(console_log) > 0:
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(os.path.getmtime(console_log)))
            dst = os.path.join(log_dir, f"console_{ts}.log")
            n = 1
            while os.path.exists(dst):
                dst = os.path.join(log_dir, f"console_{ts}_{n}.log")
                n += 1
            os.replace(console_log, dst)
            shutil.copy2(dst, os.path.join(log_dir, "console.prev.log"))
        olds = sorted(f for f in os.listdir(log_dir) if f.startswith("console_") and f.endswith(".log"))
        for f in olds[:-keep] if len(olds) > keep else []:
            os.unlink(os.path.join(log_dir, f))
    except OSError:
        pass


# V4.6.1 TAB NAMES (user's wording)
TAB_AGENTIC = "Agentic实机任务"      # was 「NERO实机任务」 (V3.65): LLM agent + traditional skills on the real arm
TAB_SMOLVLA = "Smol VLA实机任务"     # was 「VLA / LeRobot (V4)」


def v370_mirror_lines(path, lines):
    """V3.70 CONSOLE LOG MIRROR: real-page [V3.xx ...] lines (real NERO, Camera, skills, real task tabs) also go into
    logs/console.log so one file holds the whole story (the subprocess output is already written there)."""
    if not path or not lines:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            for ln in lines:
                f.write(f"[{time.strftime('%H:%M:%S')} REAL] {ln}\n")
    except Exception:
        pass


def run_console(initial_tab="sim"):
    import tkinter as tk
    from tkinter import ttk, messagebox

    log_dir = os.path.join(HERE, "logs")
    os.makedirs(log_dir, exist_ok=True)
    console_log = os.path.join(log_dir, "console.log")
    v46_rotate_console_log(log_dir, console_log)          # V4.6 LOG ROTATION
    open(console_log, "w").close()

    xinit = v463_xinitthreads()                       # V4.6.3 NATIVE CRASH GUARD (before the display is opened)
    install_crash_log(log_dir)                        # V4.6.3: open crash.log before Tk (X error notes go there)
    xerr = v463_x_error_handlers()
    root = tk.Tk()
    install_crash_log(log_dir, root)                  # V3.70 CRASH LOG
    v370_crash_note(f"V4.6.3: XInitThreads {xinit}; X error handler {xerr}; C-level messages (Tcl / Xlib / glibc) "
                    f"-> logs/stderr.log when started with start_nero.sh")
    _v312_pick_cjk_font(root)
    root.title(f"NERO {VERSION} 控制台")
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{min(1420, sw - 40)}x{min(1000, sh - 80)}")      # V3.14: joint panel needs room
    root.minsize(900, 600)

    hdr = ttk.Frame(root, padding=(10, 8, 10, 4))
    hdr.pack(fill="x")
    ttk.Label(hdr, text=f"NERO {VERSION}", font=("TkDefaultFont", 14, "bold")).pack(side="left")
    ttk.Label(hdr, text=f"   文件夹: {HERE}").pack(side="left")
    mode_var = tk.StringVar(value="当前模式: 空闲")
    import NERO_V314_Joint_Control as _JCF          # V3.16.1: live labels in fixed-size holders (no layout jumps)
    _JCF.fixed_label(tk, hdr, 1, width=420, textvariable=mode_var, font=("TkDefaultFont", 11, "bold"), anchor="ne",
                     justify="right", pack={"side": "right"})

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=8, pady=4)

    # ------------------------------------------------------------- shared helpers
    def make_lamp_row(parent, labels):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(6, 4))
        lamps = []
        for i, lab in enumerate(labels):
            cell = ttk.Frame(row)
            cell.pack(side="left", padx=4)
            c = tk.Canvas(cell, width=22, height=22, highlightthickness=0)
            oid = c.create_oval(2, 2, 20, 20, fill=LAMP_COLORS[GREY], outline="#555555")
            c.pack(side="left")
            ttk.Label(cell, text=lab).pack(side="left", padx=(3, 0))
            if i < len(labels) - 1:
                ttk.Label(row, text="→").pack(side="left")
            lamps.append((c, oid))
        return lamps

    def set_lamp(lamp, state):
        c, oid = lamp
        c.itemconfigure(oid, fill=LAMP_COLORS[state])

    def make_log(parent):
        frm = ttk.Frame(parent)
        frm.pack(fill="both", expand=True, pady=(6, 0))
        txt = tk.Text(frm, height=20, wrap="none", font=("TkFixedFont", 9))
        sy = ttk.Scrollbar(frm, orient="vertical", command=txt.yview)
        sx = ttk.Scrollbar(frm, orient="horizontal", command=txt.xview)
        txt.configure(yscrollcommand=sy.set, xscrollcommand=sx.set, state="disabled")
        txt.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        frm.rowconfigure(0, weight=1)
        frm.columnconfigure(0, weight=1)
        txt.tag_configure("fail", foreground="#c62828")
        txt.tag_configure("ok", foreground="#2e7d32")
        txt.tag_configure("warn", foreground="#b26a00")
        return txt

    def log_append(txt, line):
        tag = None
        if "[FAIL]" in line or "FAILED" in line:
            tag = "fail"
        elif "[OK]" in line or "[READY]" in line:
            tag = "ok"
        elif "[WARN]" in line:
            tag = "warn"
        at_end = txt.yview()[1] > 0.995
        txt.configure(state="normal")
        txt.insert("end", line + "\n", (tag,) if tag else ())
        n = int(txt.index("end-1c").split(".")[0])
        if n > 6000:
            txt.delete("1.0", f"{n - 5000}.0")
        txt.configure(state="disabled")
        if at_end:
            txt.see("end")

    def copy_text(txt):
        root.clipboard_clear()
        root.clipboard_append(txt.get("1.0", "end-1c"))

    # ------------------------------------------------------------- 仿真 tab
    sim_tab = ttk.Frame(nb, padding=10)
    nb.add(sim_tab, text="仿真 (MuJoCo)")
    ttk.Label(sim_tab, text="启动流程（与 V3.11 的 start_nero.sh 完全相同，只是改为按按钮启动）",
              font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    sim_lamps = make_lamp_row(sim_tab, [lab for _k, lab, *_ in SIM_STEPS])

    sim_bar = ttk.Frame(sim_tab)
    sim_bar.pack(fill="x", pady=4)
    ttk.Label(sim_bar, text="场景:").pack(side="left")
    sim_scene_labels = {k: lab for k, lab in SIM_SCENES}
    sim_scene_by_label = {lab: k for k, lab in SIM_SCENES}
    sim_scene_var = tk.StringVar(value=sim_scene_labels[DEFAULT_SIM_SCENE])
    sim_scene_box = ttk.Combobox(sim_bar, textvariable=sim_scene_var, values=[lab for _k, lab in SIM_SCENES],
                                 state="readonly", width=38)
    sim_scene_box.pack(side="left", padx=(4, 12))
    btn_sim_start = ttk.Button(sim_bar, text="▶ 启动仿真")
    btn_sim_stop = ttk.Button(sim_bar, text="■ 停止仿真", state="disabled")
    btn_sim_copy = ttk.Button(sim_bar, text="Copy Log")
    btn_sim_start.pack(side="left")
    btn_sim_stop.pack(side="left", padx=6)
    btn_sim_copy.pack(side="left", padx=6)
    sim_status = tk.StringVar(value="未运行")
    ttk.Label(sim_bar, textvariable=sim_status, font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=16)
    ttk.Label(sim_tab, text="仿真 GUI（Agent / Skill Test 等）在后端启动后作为单独窗口打开；关闭那个窗口或点"
                            "「停止仿真」都会停止全部仿真进程。完整日志: logs/launcher.log",
              foreground="#555555", wraplength=1000, justify="left").pack(anchor="w")
    sim_log = make_log(sim_tab)
    btn_sim_copy.configure(command=lambda: copy_text(sim_log))

    tracker = SimStepTracker()
    runner = SimRunner(SIM_SCRIPT, log_path=console_log)

    def refresh_sim_lamps():
        for lamp, (k, *_r) in zip(sim_lamps, SIM_STEPS):
            set_lamp(lamp, tracker.state[k])

    # ------------------------------------------------------------- 实机 tab (V3.13 REAL TAB)
    import NERO_V313_Real_Robot as RR
    # V3.14: the real tab scrolls (lamps + params + status + joint panel + log do not fit small screens)
    real_outer = ttk.Frame(nb)
    nb.add(real_outer, text="实机NERO调试")
    real_canvas = tk.Canvas(real_outer, highlightthickness=0)
    real_sb = ttk.Scrollbar(real_outer, orient="vertical", command=real_canvas.yview)
    real_canvas.configure(yscrollcommand=real_sb.set)
    real_sb.pack(side="right", fill="y")
    real_canvas.pack(side="left", fill="both", expand=True)
    real_tab = ttk.Frame(real_canvas, padding=10)
    real_win = real_canvas.create_window(0, 0, window=real_tab, anchor="nw")
    real_tab.bind("<Configure>", lambda _e: real_canvas.configure(scrollregion=real_canvas.bbox("all")))
    real_canvas.bind("<Configure>", lambda e: real_canvas.itemconfigure(real_win, width=e.width))

    def _real_wheel(e):
        if e.widget.winfo_class() in ("Scale", "Text", "TCombobox"):
            return
        real_canvas.yview_scroll(-1 if (getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0) else 1, "units")
    for seq in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
        real_canvas.bind_all(seq, lambda e: _real_wheel(e) if str(e.widget).startswith(str(real_outer)) else None, add="+")
    ttk.Label(real_tab, text="实机启动流程：每一步变绿后才能执行下一步；「一键启动」按顺序执行，遇到红灯停止",
              font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

    steps_row = ttk.Frame(real_tab)
    steps_row.pack(fill="x", pady=(6, 2))
    real_lamps, real_step_btns, real_step_msgs = [], [], []
    for i, lab in enumerate(RR.STEP_LABELS):
        cell = ttk.Frame(steps_row)
        cell.grid(row=0, column=2 * i, sticky="n", padx=3)
        top = ttk.Frame(cell)
        top.pack()
        c = tk.Canvas(top, width=22, height=22, highlightthickness=0)
        oid = c.create_oval(2, 2, 20, 20, fill=LAMP_COLORS[GREY], outline="#555555")
        c.pack(side="left")
        ttk.Label(top, text=lab).pack(side="left", padx=(3, 0))
        b_ = ttk.Button(cell, text="执行", width=6)
        b_.pack(pady=(2, 0))
        mv = tk.StringVar(value="")
        _JCF.fixed_label(tk, cell, 3, width=160, textvariable=mv, fg="#555555", wraplength=154, justify="center", anchor="n")
        real_lamps.append((c, oid))
        real_step_btns.append(b_)
        real_step_msgs.append(mv)
        if i < len(RR.STEP_LABELS) - 1:
            ttk.Label(steps_row, text="→").grid(row=0, column=2 * i + 1, sticky="n", pady=3)

    real_bar = ttk.Frame(real_tab)
    real_bar.pack(fill="x", pady=4)
    btn_real_all = ttk.Button(real_bar, text="▶ 一键启动")
    btn_real_stop = ttk.Button(real_bar, text="■ 全部停止")
    btn_real_stale = ttk.Button(real_bar, text="清理残留进程")
    btn_real_all.pack(side="left")
    btn_real_stop.pack(side="left", padx=6)
    btn_real_stale.pack(side="left", padx=6)
    btn_estop = tk.Button(real_bar, text="■ 急停", bg="#c62828", fg="white", activebackground="#8e0000",
                          font=("TkDefaultFont", 11, "bold"))
    btn_estop.pack(side="right")
    real_status = tk.StringVar(value="空闲")
    _JCF.fixed_label(tk, real_bar, 1, width=420, textvariable=real_status, font=("TkDefaultFont", 10, "bold"),
                     pack={"side": "left", "padx": 16})
    btn_params = ttk.Button(real_bar, text="参数 ▴")
    btn_params.pack(side="right", padx=8)

    # ---- parameters
    pf_holder = ttk.Frame(real_tab)
    pf_holder.pack(fill="x")
    pf = ttk.LabelFrame(pf_holder, text="启动参数（命令由界面拼接；arm_type 固定 nero）")
    pf.pack(fill="x", pady=4)

    def toggle_params():
        if pf.winfo_ismapped():
            pf.pack_forget()
            pf_holder.configure(height=1)          # Tk keeps an emptied master's old size otherwise
            btn_params.configure(text="参数 ▾")
        else:
            pf.pack(fill="x", pady=4)
            btn_params.configure(text="参数 ▴")
    btn_params.configure(command=toggle_params)
    v_port = tk.StringVar(value=RR.CAN_PORT_DEFAULT)
    v_eff = tk.StringVar(value="none")
    v_fw = tk.StringVar(value=RR.DEFAULT_FW_VERSION)
    v_speed = tk.IntVar(value=10)
    v_unlock = tk.BooleanVar(value=False)
    v_gate = tk.BooleanVar(value=False)
    v_follow = tk.BooleanVar(value=True)
    v_rviz = tk.BooleanVar(value=True)
    v_domain = tk.StringVar(value=str(RR.DEFAULT_DOMAIN_ID))
    import NERO_V462_Teach_Gravity as TG                              # V4.6.2: one setting for this box and 「高级」
    v_payload = tk.StringVar(value=RR.DEFAULT_TEACH_PAYLOAD)      # V3.70 TEACH PAYLOAD
    _tg0 = TG.load()["payload"]                                       # V4.6.2: the saved class (if any)
    v_payload.set("不设置" if _tg0 == "none" else _tg0)

    def _payload_box_changed(*_a):
        cur = TG.load()
        val = "none" if v_payload.get() == "不设置" else v_payload.get()
        if cur["payload"] != val:
            cur.update(payload=val, mode="class")
            TG.save(cur)
    v_payload.trace_add("write", _payload_box_changed)

    def _payload_from_dialog(st_):
        want = "不设置" if st_["payload"] == "none" else st_["payload"]
        if v_payload.get() != want:
            v_payload.set(want)
    TG.add_listener(_payload_from_dialog)
    r0 = ttk.Frame(pf)
    r0.pack(fill="x", padx=6, pady=3)
    ttk.Label(r0, text="can_port").pack(side="left")
    ttk.Entry(r0, textvariable=v_port, width=7).pack(side="left", padx=(2, 10))
    ttk.Label(r0, text="arm_type nero（固定）").pack(side="left", padx=(0, 10))
    ttk.Label(r0, text="effector").pack(side="left")
    cb_eff = ttk.Combobox(r0, textvariable=v_eff, values=["none", "agx_gripper"], width=11, state="readonly")
    cb_eff.pack(side="left", padx=(2, 10))
    ttk.Label(r0, text="fw_version").pack(side="left")
    ttk.Combobox(r0, textvariable=v_fw, values=["v121", "v120", "v112", "v111", "default", ""], width=7, state="readonly").pack(side="left", padx=(2, 10))
    ttk.Label(r0, text="ROS_DOMAIN_ID").pack(side="left")
    ttk.Entry(r0, textvariable=v_domain, width=4).pack(side="left", padx=(2, 10))
    ttk.Label(r0, text="示教负载").pack(side="left")             # V3.70: gravity compensation class for teach mode
    ttk.Combobox(r0, textvariable=v_payload, values=list(RR.PAYLOAD_CHOICES) + ["不设置"], width=6,
                 state="readonly").pack(side="left", padx=(2, 10))
    r1 = ttk.Frame(pf)
    r1.pack(fill="x", padx=6, pady=3)
    ttk.Label(r1, text="speed_percent").pack(side="left")
    sc_speed = tk.Scale(r1, from_=1, to=30, orient="horizontal", variable=v_speed, length=180, showvalue=True)
    sc_speed.pack(side="left", padx=(2, 4))

    def on_unlock():
        sc_speed.configure(to=100 if v_unlock.get() else 30)
        if not v_unlock.get() and v_speed.get() > 30:
            v_speed.set(30)
        refresh_preview()
    ttk.Checkbutton(r1, text="解锁 >30%", variable=v_unlock, command=on_unlock).pack(side="left", padx=(0, 12))
    for t_, var_ in (("auto_control_gate", v_gate), ("follow", v_follow), ("RViz", v_rviz)):
        ttk.Checkbutton(r1, text=t_, variable=var_, command=lambda: refresh_preview()).pack(side="left", padx=4)
    r2 = ttk.Frame(pf)
    r2.pack(fill="x", padx=6, pady=(0, 4))
    preview = tk.StringVar(value="")
    _JCF.fixed_label(tk, r2, 2, textvariable=preview, font=("TkFixedFont", 9), wraplength=900,
                     pack={"side": "left", "fill": "x", "expand": True})

    _cfg_snap = {"v": None}

    def real_cfg():
        """V4.6.3: the Tk variables are read on the Tk main thread only; a worker thread (pipeline, teach, joint jobs)
        gets the last snapshot taken on the main thread (refreshed by real_poll and before every job start)."""
        if threading.current_thread() is not threading.main_thread():
            snap = _cfg_snap["v"]
            if snap is not None:
                return dict(snap)
            return {"can_port": RR.CAN_PORT_DEFAULT, "effector_type": "none", "fw_version": RR.DEFAULT_FW_VERSION,
                    "push_fw": RR.DEFAULT_FW_VERSION, "speed_percent": 10, "speed_unlocked": False,
                    "auto_control_gate": False, "follow": True, "use_rviz": True,
                    "domain_id": int(RR.DEFAULT_DOMAIN_ID), "teach_payload": TG.payload_arg(TG.load())}
        snap = _real_cfg_read()
        _cfg_snap["v"] = snap
        return dict(snap)

    def _real_cfg_read():
        try:
            dom = int(v_domain.get())
        except Exception:
            dom = int(RR.DEFAULT_DOMAIN_ID)
        return {"can_port": v_port.get().strip() or "can0", "effector_type": v_eff.get(), "fw_version": v_fw.get(),
                "push_fw": v_fw.get() or RR.DEFAULT_FW_VERSION, "speed_percent": int(v_speed.get()), "speed_unlocked": bool(v_unlock.get()),
                "auto_control_gate": bool(v_gate.get()), "follow": bool(v_follow.get()), "use_rviz": bool(v_rviz.get()),
                "domain_id": dom, "teach_payload": v_payload.get()} | {
                    "teach_payload": TG.payload_arg(TG.load())}      # V4.6.2: the saved class or custom mass wins

    def refresh_preview(*_a):
        preview.set(f"ROS_DOMAIN_ID={real_cfg()['domain_id']}  " + RR.launch_command(real_cfg()))
    for var_ in (v_port, v_eff, v_fw, v_speed, v_domain):
        var_.trace_add("write", refresh_preview)

    def copy_cmd():
        root.clipboard_clear()
        root.clipboard_append(RR.ros_env_prefix(real_cfg()["domain_id"]) + RR.launch_command(real_cfg()))

    def copy_env():
        root.clipboard_clear()
        root.clipboard_append(RR.ros_env_prefix(real_cfg()["domain_id"]).strip())
    ttk.Button(r2, text="复制命令", command=copy_cmd).pack(side="right", padx=3)
    ttk.Button(r2, text="复制终端环境", command=copy_env).pack(side="right", padx=3)

    # ---- live status + operations
    lf = ttk.LabelFrame(real_tab, text="实时状态")
    lf.pack(fill="x", pady=4)
    st_vars = {k: tk.StringVar(value="-") for k in ("can", "ros", "joints", "arm", "grip")}
    for k, lab in (("can", "CAN"), ("ros", "ROS"), ("joints", "关节"), ("arm", "arm_status"), ("grip", "夹爪")):
        row = ttk.Frame(lf)
        row.pack(fill="x", padx=6, pady=1)
        ttk.Label(row, text=f"{lab}:", width=10).pack(side="left")
        _JCF.fixed_label(tk, row, 2 if k in ("ros", "arm", "grip") else 1, textvariable=st_vars[k], font=("TkFixedFont", 9),
                         wraplength=1000, pack={"side": "left", "fill": "x", "expand": True})
    ob = ttk.Frame(lf)
    ob.pack(fill="x", padx=6, pady=4)
    op_btns = []
    for t_, suffix, val, warn in (("控制门 开", "control_enable", True, None), ("控制门 关", "control_enable", False, None),
                                  ("使能", "enable_agx_arm", True, None),
                                  ("去使能⚠", "enable_agx_arm", False, "去使能后电机失去力矩，手臂可能下垂。确认手臂在低位或已扶住？"),
                                  ("回 Home⚠", "move_home", None, None),
                                  ("待断电位置⚠", "park", None, None),
                                  ("退出示教模式", "exit_teach_mode", None, None)):
        def _op(suffix=suffix, val=val, warn=warn, t_=t_):
            if suffix == "move_home":                      # V3.19: MoveIt to all-zero (service only as fallback)
                jpanel.go_home()
                return
            if suffix == "park":                           # V3.24: park pose D before 去使能 / power off
                jpanel.go_park()
                return
            if suffix == "exit_teach_mode":                # V3.19/V3.53: service + pyAgxArm normal mode + enable (hold current pose)
                if not messagebox.askyesno("退出示教模式", "NERO 将退出示教模式并切回正常控制（直接保持并锁在当前姿态，不强制回零位；如需回零可点旁边「回 Home⚠」）。\n请扶稳机械臂、手在急停上。继续？"):
                    return
                if safe_tab.mode == "teach":
                    safe_tab._teach_exiting = True
                    safe_tab.v_mode.set("edit")
                    safe_tab.on_mode()
                    safe_tab._teach_exiting = False
                try:
                    cam_tab.stop_flag = True
                except Exception:
                    pass
                start_job("退出示教模式", lambda: pipe.set_teach_mode(False, home_model=getattr(jpanel, "model", None),
                                                                 home_fn=safe_tab.exit_home, return_home=False), force=True)
                return
            if warn and not messagebox.askyesno(t_, warn):
                return
            if suffix == "enable_agx_arm" and val is True and (pipe.teach_mode or pipe.leader_active() or not pipe.monitor.running()):
                start_job("使能（退出示教）", lambda: pipe.set_teach_mode(False, home_model=getattr(jpanel, "model", None),
                                                                      home_fn=safe_tab.exit_home, return_home=False), force=True)
                return
            pipe.call_service(suffix, val)
        b_ = ttk.Button(ob, text=t_, command=_op)
        b_.pack(side="left", padx=3)
        op_btns.append(b_)
    # V3.15 REAL SUB-TABS: 关节控制 | 一致性测试 | 工作区安全 (built after the pipeline)
    real_nb = ttk.Notebook(real_tab)
    real_nb.pack(fill="x", pady=4)
    jp_holder = ttk.Frame(real_nb)            # V3.14 JOINT PANEL goes here
    cons_holder = ttk.Frame(real_nb)
    safe_holder = ttk.Frame(real_nb)
    real_nb.add(jp_holder, text="关节控制")
    real_nb.add(cons_holder, text="一致性测试")
    real_nb.add(safe_holder, text="工作区安全")
    real_wf_bar = ttk.Frame(real_tab)
    real_wf_bar.pack(fill="x", pady=(4, 0))
    real_log = make_log(real_tab)
    real_log.configure(height=12)
    ttk.Button(ob, text="Copy Log", command=lambda: copy_text(real_log)).pack(side="right")
    ttk.Button(real_wf_bar, text="Copy Log", command=lambda: copy_text(real_log)).pack(side="right")

    # ---- controller
    ask_q = queue.Queue()      # V3.14: worker threads never touch Tk; real_poll (main thread) shows the dialogs

    def confirm_from_worker(title, text):
        if threading.current_thread() is threading.main_thread():
            return bool(messagebox.askyesno(title, text))
        ev, ans = threading.Event(), {}
        ask_q.put((title, text, ev, ans))
        ev.wait()
        return bool(ans.get("v"))

    def serve_asks():
        while True:
            try:
                title, text, ev, ans = ask_q.get_nowait()
            except queue.Empty:
                return
            try:
                ans["v"] = messagebox.askyesno(title, text)
            finally:
                ev.set()

    pipe = RR.RealPipeline(real_cfg, sim_active_fn=lambda: runner.running(), confirm_fn=confirm_from_worker,
                           log_path=console_log)
    real_job = {"t": None, "name": None}
    ui_flags = {}

    # ---- V3.14 JOINT PANEL (7 joints + gripper, MoveIt joint goals)
    import NERO_V314_Joint_Control as JC
    from tkinter import simpledialog
    jpanel = JC.JointPanelUI(tk, ttk, jp_holder, pipe, real_cfg, sim_running_fn=lambda: runner.running(),
                             log=lambda line: log_append(real_log, line), ask=confirm_from_worker,
                             ask_string=lambda t, q: simpledialog.askstring(t, q, parent=root),
                             can_snapshot_fn=lambda: (pipe.can.snapshot() if (pipe.can.running() or pipe.can.total) else None))
    import NERO_V315_Consistency_Safety as CS
    cons_tab = CS.ConsistencyTabUI(tk, ttk, cons_holder, pipe, jpanel, real_cfg,
                                   log=lambda line: log_append(real_log, line), ask=confirm_from_worker)
    import NERO_V316_Safety_Editor as SE        # V3.16 SAFETY EDITOR replaces the V3.15 number table
    safe_tab = SE.SafetyEditorUI(tk, ttk, safe_holder, pipe, jpanel, real_cfg,
                                 log=lambda line: log_append(real_log, line), ask=confirm_from_worker)

    # ---- V3.25 CAMERA TAB (D435i: live view, depth check, hand-eye calibration, visual locating)
    import NERO_V325_Camera as CAM
    cam_outer = ttk.Frame(nb, padding=6)
    nb.add(cam_outer, text="Camera调试")
    cam_main = ttk.Frame(cam_outer)
    cam_main.pack(fill="both", expand=True)
    cam_bar = ttk.Frame(cam_outer)
    cam_bar.pack(fill="x")
    cam_log = make_log(cam_outer)
    cam_log.configure(height=8)
    ttk.Button(cam_bar, text="Copy Log", command=lambda: copy_text(cam_log)).pack(side="right")

    cam_q = queue.Queue()

    def cam_log_line(line):                         # any thread; drained by real_poll (Tk main thread)
        pipe.log(line)                              # also into the real log + console.log
        cam_q.put(line)
    cam_tab = CAM.CameraTabUI(tk, ttk, cam_main, pipe, jpanel, safe_tab, real_cfg, log=cam_log_line,
                              ask=confirm_from_worker, visible_fn=lambda: nb.select() == str(cam_outer))
    cam_tab.attach_workflow_bar(real_wf_bar)
    cam_tab.attach_workflow_bar(cam_bar)

    # ---- V3.59 SKILL STRATEGY KNOBS & HOVER DRY-RUN TAB
    import NERO_V359_Strategy as ST59
    strat_outer = ttk.Frame(nb, padding=6)
    nb.add(strat_outer, text="技能参数旋钮")
    strat_tab = ST59.SkillStrategyTabUI(
        tk, ttk, strat_outer, cam_tab=cam_tab, pipe=pipe, jpanel=jpanel,
        log=cam_log_line, ask=confirm_from_worker,
    )
    cam_tab.strategy_tab = strat_tab
    # V3.60: a skill applied in the simulation GUI tab shows up here when this tab is opened
    nb.bind("<<NotebookTabChanged>>",
            lambda _e: strat_tab.reload_active_if_changed() if nb.select() == str(strat_outer) else None, add="+")

    # ---- V3.65 NERO REAL CUP TASK TAB: the stack_cups_so101 simulation skills + parameters on the real NERO
    #      (SO-101 gripper on an STS3215 servo, fixed D435i + wrist camera monitors, LLM agent, run modes 1-3)
    rct_tab = None
    try:
        import NERO_V365_Real_Cup_Task as RCT
        rct_outer = ttk.Frame(nb, padding=6)
        nb.add(rct_outer, text=TAB_AGENTIC)
        rct_tab = RCT.RealCupTaskTabUI(tk, ttk, rct_outer, pipe, jpanel, cam_tab, log=pipe.log, ask=confirm_from_worker,
                                       visible_fn=lambda: nb.select() == str(rct_outer), safe_tab=safe_tab)
        nb.bind("<<NotebookTabChanged>>",
                lambda _e: rct_tab.refresh_skill() if nb.select() == str(rct_outer) else None, add="+")
    except Exception as exc:
        log_append(real_log, f"[FAIL] [V3.65 REAL CUP TASK] {TAB_AGENTIC} tab not available: {exc}")

    # ---- V4.6.4 VISION ACCURACY CHECK: panel at the bottom of Camera调试 -> 视觉定位 (closed SO-101 fingertip above
    #      the target; main camera only / wrist camera only / main + wrist assist; the person writes dx / dy)
    vision_check = None
    if rct_tab is not None:
        try:
            import NERO_V464_Vision_Check as VC
            vision_check = VC.attach(tk, ttk, cam_tab, rct_tab, pipe, jpanel, log=cam_log_line)
        except Exception as exc:
            try:
                pipe.log(f"[FAIL] [V4.6.4 VISION CHECK] 视觉精度验证面板不可用：{exc}")
            except Exception:
                pass

    # ---- V4.1 VLA / LeRobot TAB: Dual-Path Control Arbiter, Observation Bridge, LeRobot Adapter & Action Executor
    # V4.5.1 LAZY VLA TAB: the tab is built the first time it is OPENED, not at console start. A V4.5 console crashed
    # (segfault inside Tk while this tab created its widgets at start-up); now the console always starts like V4.3,
    # and a problem in this tab can no longer stop the cup / egg tabs. NERO_NO_VLA_TAB=1 (./start_nero.sh --no-vla)
    # leaves the tab out completely.
    vla_holder = {"tab": None, "building": False}
    if os.environ.get("NERO_NO_VLA_TAB", "0") != "1":
        try:
            vla_outer = ttk.Frame(nb, padding=6)
            nb.add(vla_outer, text=TAB_SMOLVLA)
            vla_hint = ttk.Label(vla_outer, text=f"正在加载 {TAB_SMOLVLA} 页 ...（第一次打开这一页时才加载）")
            vla_hint.pack(anchor="w", pady=8)

            def _build_vla_tab():
                if vla_holder["tab"] is not None or vla_holder["building"]:
                    return
                vla_holder["building"] = True
                log_append(real_log, "[INFO] [V4.5.1 VLA TAB] 正在创建 VLA / LeRobot 页 ...")
                try:
                    import NERO_V4_Console_Tab as V4T
                    vla_hint.destroy()
                    vla_holder["tab"] = V4T.VLALeRobotTabUI(
                        tk, ttk, vla_outer, pipe=pipe, jpanel=jpanel, cam_tab=cam_tab, rct_tab=rct_tab,
                        log=pipe.log, ask=confirm_from_worker, visible_fn=lambda: nb.select() == str(vla_outer),
                        safe_tab=safe_tab
                    )
                    log_append(real_log, "[OK] [V4.5.1 VLA TAB] VLA / LeRobot 页已创建")
                except Exception as exc:
                    log_append(real_log, f"[FAIL] [V4.1 VLA TAB] VLA / LeRobot tab not available: {exc}")
                    try:
                        ttk.Label(vla_outer, text=f"VLA / LeRobot 页加载失败：{exc}").pack(anchor="w", pady=8)
                    except Exception:
                        pass

            nb.bind("<<NotebookTabChanged>>",
                    lambda _e: root.after(50, _build_vla_tab) if nb.select() == str(vla_outer) else None, add="+")
        except Exception as exc:
            log_append(real_log, f"[FAIL] [V4.1 VLA TAB] VLA / LeRobot tab not available: {exc}")
    else:
        log_append(real_log, "[INFO] [V4.5.1 VLA TAB] NERO_NO_VLA_TAB=1：不加载 VLA / LeRobot 页")

    def start_job(name, fn, force=False):
        real_cfg()                                    # V4.6.3: fresh snapshot before the worker starts
        if not force and real_job["t"] is not None and real_job["t"].is_alive():
            log_append(real_log, f"[WARN] busy: {real_job['name']} still running")
            return
        if runner.running():
            log_append(real_log, "[FAIL] 仿真正在运行：实机流程被互斥锁定（两者都会发布 /feedback/joint_states）")
            return

        def wrap():
            try:
                fn()
            except Exception as exc:
                pipe.log(f"[FAIL] [V3.13] {name}: {exc}")
        real_job["name"] = name
        real_job["t"] = threading.Thread(target=wrap, daemon=True, name=f"v313_{name}")
        real_job["t"].start()

    for i, key in enumerate(RR.STEP_KEYS):
        real_step_btns[i].configure(command=lambda key=key: start_job(key, lambda: pipe.run_step(key)))
    btn_real_all.configure(command=lambda: start_job("一键启动", pipe.run_all))

    def on_real_stop():
        if pipe.active() and not messagebox.askyesno("全部停止", "停止 RViz / MoveIt / 驱动 / push？\n"
                                                              "停止驱动后电机状态取决于固件，请确认手臂在低位。"):
            return
        threading.Thread(target=pipe.stop_all, daemon=True).start()
    btn_real_stop.configure(command=on_real_stop)
    btn_real_stale.configure(command=lambda: start_job("清理残留", lambda: pipe.step_env(kill_stale=True)))

    def on_estop():
        if pipe.call_service("emergency_stop") is None:
            log_append(real_log, "[FAIL] 急停服务不可用（驱动未运行）：请用 Web 界面急停或直接断开 24 V 电源")
    btn_estop.configure(command=on_estop)

    def real_poll():
        serve_asks()
        try:
            real_cfg()                                # V4.6.3: refresh the snapshot the worker threads read
        except Exception:
            pass
        mirrored = []                                 # V3.70 CONSOLE LOG MIRROR
        try:
            while True:
                ln = pipe.events.get_nowait()
                mirrored.append(ln)
                log_append(real_log, ln)
        except queue.Empty:
            pass
        v370_mirror_lines(console_log, mirrored)
        try:
            while True:
                log_append(cam_log, cam_q.get_nowait())
        except queue.Empty:
            pass
        busy = real_job["t"] is not None and real_job["t"].is_alive()
        for i, key in enumerate(RR.STEP_KEYS):
            lamp, msg = pipe.state[key]
            if busy and real_job["name"] in (key, "一键启动") and lamp == GREY:
                pass
            set_lamp(real_lamps[i], lamp)
            real_step_msgs[i].set(msg[:80])
            real_step_btns[i].configure(state="disabled" if (busy or runner.running() or not pipe.prev_ok(key)) else "normal")
        btn_real_all.configure(state="disabled" if (busy or runner.running()) else "normal")
        snap = pipe.can.snapshot() if pipe.can.running() or pipe.can.total else None
        if snap:
            g = " ".join(f"{k.split('_', 1)[1]}{'✓' if v else '✗'}" for k, v in snap["groups"].items())
            grip = (" | 夹爪旧ID " + ",".join(snap["gripper_old_id"]) + " ⚠") if snap["gripper_old_id"] else ""
            st_vars["can"].set(f"{snap['fps']:.0f} fps | 总帧 {snap['total']} | 错误帧 {snap['error_frames']} | {g}{grip}"
                               + (f" | {snap['error']}" if snap.get("error") else ""))
            if snap["gripper_old_id"]:
                v_eff.set("none")
                cb_eff.configure(state="disabled")
        m = pipe.mon_status
        if m:
            st_vars["ros"].set(f"joint_states {m.get('js_hz')} Hz | nodes {len(m.get('nodes') or [])} | "
                               f"move_group {'✓' if m.get('has_move_group') else '✗'} | MoveGroup action "
                               f"{'✓' if m.get('has_move_action') else '✗'} | services {sorted((m.get('services') or {}).keys())}")
            js = m.get("joints") or {}
            if js.get("names"):
                st_vars["joints"].set("  ".join(f"{n}={math.degrees(p):.1f}°" for n, p in zip(js["names"], js["position"])))
            st_vars["arm"].set(RR.compact_status(m.get("arm_status")) or "-")
            st_vars["grip"].set(RR.compact_status(m.get("gripper_status")) or "-")
        elif pipe.push_info.get("joints") is not None:
            st_vars["joints"].set(f"(pyAgxArm) {pipe.push_info.get('joints')}")
            st_vars["arm"].set(f"(pyAgxArm) {str(pipe.push_info.get('status'))[:300]}")
        if busy:
            real_status.set(f"执行中: {real_job['name']}")
        elif pipe.lamp("moveit") == GREEN:
            real_status.set("实机就绪：驱动 + MoveIt + RViz")
        elif pipe.active():
            real_status.set("实机进程运行中")
        else:
            real_status.set("空闲")
        btn_sim_start.configure(state="disabled" if (pipe.active() or runner.running()) else "normal")
        # V3.16.1: no automatic collapse of the parameter box (it moved everything below); use 「参数 ▴」
        for name_, ui_ in (("joints", jpanel), ("consistency", cons_tab), ("safety", safe_tab)):
            try:
                ui_.poll()
            except Exception as exc:
                key_ = f"poll_err_{name_}"
                if ui_flags.get(key_) != str(exc):
                    ui_flags[key_] = str(exc)
                    log_append(real_log, f"[FAIL] [V3.15] {name_} refresh: {exc}")
        cam_tab.update_workflow_lamps()
        update_mode()
        root.after(200, real_poll)

    refresh_preview()

    # ------------------------------------------------------------- mode / interlock
    def update_mode():
        try:
            sim_scene_box.configure(state="disabled" if runner.running() else "readonly")   # V3.44: fixed while running
        except Exception:
            pass
        if runner.running():
            mode_var.set("当前模式: 仿真运行中（实机锁定）")
        elif pipe.active():
            mode_var.set("当前模式: 实机运行中（仿真锁定）")
        else:
            mode_var.set("当前模式: 空闲")

    def on_sim_start():
        if runner.running():
            return
        if pipe.active():
            log_append(sim_log, "[FAIL] 实机进程正在运行：仿真被互斥锁定（两者都会发布 /feedback/joint_states）")
            return
        if not v377_clear_stale():
            return
        tracker.start()
        refresh_sim_lamps()
        sim_log.configure(state="normal")
        sim_log.delete("1.0", "end")
        sim_log.configure(state="disabled")
        scene = sim_scene_by_label.get(sim_scene_var.get(), DEFAULT_SIM_SCENE)
        log_append(sim_log, f"[V3.12 CONSOLE] starting {SIM_SCRIPT} | V3.44 scene={scene}")
        runner.strategy_env = {}
        try:                                        # V3.60 SIM ENV INJECTION (applied skill; baseline EB injects nothing)
            prof360 = ST59.read_active_strategy_file() or strat_tab.profile
            env360, notes360 = ST59.strategy_sim_env(prof360, base_env=clean_child_env())
            runner.strategy_env = env360
            n_inj = sum(1 for k_ in env360 if not k_.startswith("NERO_V360_SKILL_"))
            log_append(sim_log, f"[OK] [V3.60 策略] 本次仿真技能 {prof360.label()} ({prof360.skill_id})："
                                + (f"注入 {n_inj} 个后端参数" if n_inj else "与基准 EB 相同，后端参数保持默认"))
            for n_ in notes360:
                log_append(sim_log, n_)
        except Exception as exc:
            runner.strategy_env = {}
            log_append(sim_log, f"[WARN] [V3.60 策略] 技能注入失败，按后端默认参数启动：{exc}")
        try:
            pid = runner.start(extra_env=scene_env(scene))
        except Exception as exc:
            log_append(sim_log, f"[FAIL] [V3.12 CONSOLE] could not start: {exc}")
            tracker.feed("[FAIL]")
            refresh_sim_lamps()
            return
        log_append(sim_log, f"[V3.12 CONSOLE] launcher PID={pid}")
        sim_status.set(f"启动中…（场景 {scene}）")
        btn_sim_start.configure(state="disabled")
        btn_sim_stop.configure(state="normal")
        update_mode()

    def v377_clear_stale():
        """V3.77 STALE PROCESS CHECK: leftovers of an earlier run (ROS / MoveIt / RViz / real driver / NERO scripts of
        any release folder) are listed before the simulation starts; the operator decides; stopped INT -> TERM -> KILL
        (the launcher's V3.07 check uses the same detector as a second line)"""
        try:
            import NERO_V377_Stale_Procs as SP
            me = [os.getpid()]
            try:
                me += [int(p_) for p_ in (pipe.own_pids() or [])]
            except Exception:
                pass
            hits = SP.stale_procs(me=me)
        except Exception as exc:
            log_append(sim_log, f"[WARN] [V3.77 STALE] 残余进程检查失败：{exc}（继续启动；启动脚本还会再查一次）")
            return True
        if not hits:
            log_append(sim_log, "[OK] [V3.77 STALE] 启动前检查：没有残余的 ROS / MoveIt / RViz / NERO 进程")
            return True
        lines = "\n".join(f"  {pid}  {kind}" for pid, kind, _cmd in hits[:15]) + (
            f"\n  …共 {len(hits)} 个" if len(hits) > 15 else "")
        for pid, kind, cmd in hits:
            log_append(sim_log, f"[WARN] [V3.77 STALE] 残余进程 {pid}: {cmd}")
        if not messagebox.askyesno("启动前检查（V3.77）", f"发现上次运行留下的进程：\n{lines}\n\n关闭它们后再启动仿真？"
                                   "\n（选“否”= 不启动；它们会和新仿真争用 ROS 话题）"):
            log_append(sim_log, "[INFO] [V3.77 STALE] 没有关闭残余进程，仿真未启动")
            return False
        sim_status.set(f"正在关闭 {len(hits)} 个残余进程…")
        done = {}
        th = threading.Thread(target=lambda: done.update(SP.stop_procs([h[0] for h in hits], log=lambda m: done.setdefault("log", m))),
                              daemon=True)
        th.start()
        t_end = time.time() + 15.0
        while th.is_alive() and time.time() < t_end:
            root.update()
            time.sleep(0.05)
        log_append(sim_log, done.get("log") or "[WARN] [V3.77 STALE] 关闭残余进程超时")
        if "stopped" not in done or done.get("left"):
            sim_status.set("残余进程没关掉 — 见日志")
            return False
        return True

    def on_sim_stop():
        if not runner.running():
            return
        if not messagebox.askyesno("停止仿真", "停止仿真？MoveIt、RViz、MuJoCo 和仿真 GUI 都会关闭。"):
            return
        log_append(sim_log, "[V3.12 CONSOLE] stop requested -> SIGINT to launcher (cleanup + restore kinematics.yaml)")
        sim_status.set("正在停止…")
        btn_sim_stop.configure(state="disabled")
        stop_t0[0] = time.time()                    # V3.76 STOP COUNTDOWN
        runner.stop()

    btn_sim_start.configure(command=on_sim_start)
    btn_sim_stop.configure(command=on_sim_stop)
    stop_t0 = [None]

    def stop_countdown():
        """V3.76 STOP COUNTDOWN: the launcher stops every process within ~8 s (INT -> TERM -> KILL); the console's
        SIGKILL of the whole session follows after STOP_GRACE_S at the latest"""
        if stop_t0[0] is None or not runner.running() or not runner.stop_requested():
            return
        left = max(0, int(round(STOP_GRACE_S - (time.time() - stop_t0[0]))))
        sim_status.set(f"正在停止…（MoveIt / RViz / MuJoCo / 仿真 GUI，最多再等 {left} s）")

    def poll():
        changed = False
        try:
            while True:
                line = runner.lines.get_nowait()
                if line is None:
                    rc = runner.exit_code
                    if not runner.stop_requested():
                        tracker.process_exited(rc)
                    changed = True
                    if runner.stop_requested():
                        sim_status.set(f"已停止 (exit {rc})")
                    elif tracker.ready:
                        sim_status.set(f"仿真已结束 (exit {rc})")
                    else:
                        sim_status.set(f"启动失败 (exit {rc}) — 见日志")
                    log_append(sim_log, f"[V3.12 CONSOLE] launcher exited rc={rc}"
                               + (f" | V3.76 stop took {time.time() - stop_t0[0]:.1f} s" if stop_t0[0] else ""))
                    stop_t0[0] = None
                    btn_sim_start.configure(state="normal")
                    btn_sim_stop.configure(state="disabled")
                    update_mode()
                    continue
                log_append(sim_log, line)
                before = dict(tracker.state)
                tracker.feed(line)
                if tracker.state != before:
                    changed = True
                    if tracker.ready:
                        sim_status.set("SYSTEM READY — 仿真运行中")
                    elif tracker.failed:
                        sim_status.set("启动失败 — 见日志")
        except queue.Empty:
            pass
        if changed:
            refresh_sim_lamps()
        stop_countdown()
        root.after(100, poll)

    def on_close():
        try:
            cam_tab.shutdown()
        except Exception:
            pass
        try:
            if rct_tab is not None:
                rct_tab.shutdown()              # V3.65: stop a running real task, close the gripper serial port
        except Exception:
            pass
        try:
            if vla_holder["tab"] is not None:
                vla_holder["tab"].shutdown()    # V4.1: stop IPC server, adapter & shared memory
        except Exception:
            pass
        if pipe.active():
            if not messagebox.askyesno("退出", "实机进程（驱动 / MoveIt / push）正在运行。关闭控制台会先全部停止，确定？"):
                return
            th = threading.Thread(target=pipe.stop_all, daemon=True)
            th.start()
            t_end = time.time() + 30
            while th.is_alive() and time.time() < t_end:
                root.update()
                time.sleep(0.1)
        pipe.shutdown()
        if runner.running():
            if not messagebox.askyesno("退出", "仿真正在运行。关闭控制台会先停止仿真，确定？"):
                return
            runner.stop()
            t_end = time.time() + STOP_GRACE_S + 3
            while runner.running() and time.time() < t_end:
                root.update()
                time.sleep(0.1)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    nb.select(real_outer if initial_tab == "real" else sim_tab)
    update_mode()
    root.after(100, poll)
    root.after(300, real_poll)
    try:
        root.mainloop()
    except BaseException as exc:                      # V4.6.3: e.g. SystemExit raised inside a Tk callback
        v370_crash_note(f"V4.6.3 console main loop left by {type(exc).__name__}: {exc}")
        raise
    v370_crash_note(f"V4.6.3 console closed normally (X errors ignored during the run: {_V463_X['n']})")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    tab = "real" if "--real" in argv else "sim"
    run_console(tab)


if __name__ == "__main__":
    main()
