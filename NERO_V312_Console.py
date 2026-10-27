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

VERSION = "V3.47.1"
HERE = os.path.dirname(os.path.realpath(__file__))
SIM_SCRIPT = os.environ.get("NERO_V312_SIM_SCRIPT", os.path.join(HERE, "start_nero_sim.sh"))

# V3.44 SCENE SELECT: the MuJoCo scene is chosen here before the simulation starts and passed to
# start_nero_sim.sh as NERO_SCENE.  Every console start defaults to the egg scene.
SIM_SCENES = [("fill_egg_holder", "fill_egg_holder - 鸡蛋放入蛋托 (默认)"),
              ("store_in_safe", "store_in_safe - 物体放入保险箱 (RoboDojo)")]
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

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, extra_env=None):
        if self.running():
            raise RuntimeError("simulation already running")
        env = clean_child_env()
        env["NERO_NO_FAIL_PAUSE"] = "1"      # never block on "Press Enter" when started from the console
        env["PYTHONUNBUFFERED"] = "1"
        env.update(extra_env or {})
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


def run_console(initial_tab="sim"):
    import tkinter as tk
    from tkinter import ttk, messagebox

    log_dir = os.path.join(HERE, "logs")
    os.makedirs(log_dir, exist_ok=True)
    console_log = os.path.join(log_dir, "console.log")
    open(console_log, "w").close()

    root = tk.Tk()
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
    v_fw = tk.StringVar(value="v111")
    v_speed = tk.IntVar(value=10)
    v_unlock = tk.BooleanVar(value=False)
    v_gate = tk.BooleanVar(value=False)
    v_follow = tk.BooleanVar(value=True)
    v_rviz = tk.BooleanVar(value=True)
    v_domain = tk.StringVar(value=str(RR.DEFAULT_DOMAIN_ID))
    r0 = ttk.Frame(pf)
    r0.pack(fill="x", padx=6, pady=3)
    ttk.Label(r0, text="can_port").pack(side="left")
    ttk.Entry(r0, textvariable=v_port, width=7).pack(side="left", padx=(2, 10))
    ttk.Label(r0, text="arm_type nero（固定）").pack(side="left", padx=(0, 10))
    ttk.Label(r0, text="effector").pack(side="left")
    cb_eff = ttk.Combobox(r0, textvariable=v_eff, values=["none", "agx_gripper"], width=11, state="readonly")
    cb_eff.pack(side="left", padx=(2, 10))
    ttk.Label(r0, text="fw_version").pack(side="left")
    ttk.Combobox(r0, textvariable=v_fw, values=["v111", ""], width=6, state="readonly").pack(side="left", padx=(2, 10))
    ttk.Label(r0, text="ROS_DOMAIN_ID").pack(side="left")
    ttk.Entry(r0, textvariable=v_domain, width=4).pack(side="left", padx=(2, 10))
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

    def real_cfg():
        try:
            dom = int(v_domain.get())
        except Exception:
            dom = int(RR.DEFAULT_DOMAIN_ID)
        return {"can_port": v_port.get().strip() or "can0", "effector_type": v_eff.get(), "fw_version": v_fw.get(),
                "push_fw": v_fw.get() or "v111", "speed_percent": int(v_speed.get()), "speed_unlocked": bool(v_unlock.get()),
                "auto_control_gate": bool(v_gate.get()), "follow": bool(v_follow.get()), "use_rviz": bool(v_rviz.get()),
                "domain_id": dom}

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
            if suffix == "exit_teach_mode":                # V3.19: service + pyAgxArm normal mode + enable
                if not messagebox.askyesno("退出示教模式", "NERO 切回正常控制（先保持当前位置），然后以 10% 速度经 MoveIt "
                                                       "回零位。\n请扶稳机械臂、手在急停上。继续？"):
                    return
                if safe_tab.mode == "teach":
                    safe_tab.v_mode.set("edit")
                    safe_tab.on_mode()
                start_job("退出示教模式", lambda: pipe.set_teach_mode(False, home_model=getattr(jpanel, "model", None),
                                                                 home_fn=safe_tab.exit_home))
                return
            if warn and not messagebox.askyesno(t_, warn):
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
    real_log = make_log(real_tab)
    real_log.configure(height=12)
    ttk.Button(ob, text="Copy Log", command=lambda: copy_text(real_log)).pack(side="right")

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

    def start_job(name, fn):
        if real_job["t"] is not None and real_job["t"].is_alive():
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
            while True:
                log_append(real_log, pipe.events.get_nowait())
        except queue.Empty:
            pass
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
        tracker.start()
        refresh_sim_lamps()
        sim_log.configure(state="normal")
        sim_log.delete("1.0", "end")
        sim_log.configure(state="disabled")
        scene = sim_scene_by_label.get(sim_scene_var.get(), DEFAULT_SIM_SCENE)
        log_append(sim_log, f"[V3.12 CONSOLE] starting {SIM_SCRIPT} | V3.44 scene={scene}")
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

    def on_sim_stop():
        if not runner.running():
            return
        if not messagebox.askyesno("停止仿真", "停止仿真？MoveIt、RViz、MuJoCo 和仿真 GUI 都会关闭。"):
            return
        log_append(sim_log, "[V3.12 CONSOLE] stop requested -> SIGINT to launcher (cleanup + restore kinematics.yaml)")
        sim_status.set("正在停止…")
        btn_sim_stop.configure(state="disabled")
        runner.stop()

    btn_sim_start.configure(command=on_sim_start)
    btn_sim_stop.configure(command=on_sim_stop)

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
                    log_append(sim_log, f"[V3.12 CONSOLE] launcher exited rc={rc}")
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
        root.after(100, poll)

    def on_close():
        try:
            cam_tab.shutdown()
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
    root.mainloop()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    tab = "real" if "--real" in argv else "sim"
    run_console(tab)


if __name__ == "__main__":
    main()
