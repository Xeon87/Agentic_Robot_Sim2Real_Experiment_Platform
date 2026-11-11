#!/usr/bin/env python3
"""NERO V4.8 GRIPPER PANEL: 「实机NERO调试 → 夹爪」.

Top bar: the global gripper choice (SO-101 / NERO AgxGripper), connection lamp, CAN port, connect / disconnect.
Three columns:  状态 (big width read-out, force, enable / homed, six fault lamps)  |  操作 (width slider, force
slider with 轻 / 中 / 重, open fully / close / disable / clear faults, object presets)  |  标定与测试 (notebook:
① 校正归零 step by step, ② 力控标定: empty baseline / gauge table / object presets, ③ 示教模式测试, ④ J7 控制器限位).
Every gripper call runs in a worker thread; widgets are touched in the Tk thread only (queue + after()).
No emoji in widget text (Ubuntu 22.04 libXft crash, V4.5.1).
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Dict, Optional

import NERO_V48_Gripper as GG

MARKER = "V4.8 GRIPPER PANEL: ZERO + FORCE CALIBRATION + TEACH TEST + J7 LIMIT"
C_OK, C_WARN, C_BAD, C_OFF, C_TXT2 = "#2e7d32", "#ef6c00", "#c62828", "#9e9e9e", "#607d8b"
FORCE_STEPS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
ZERO_STEPS = ("检查连接", "失能（手指可以用手推动）", "用手把两指合拢到底", "归零（发送归零指令，等夹爪确认）",
              "全行程检查（合拢 → 张到最大 → 合拢）")


class Lamp:
    def __init__(self, tk, parent, text, size=12):
        self.tk = tk
        self.f = tk.Frame(parent)
        self.c = tk.Canvas(self.f, width=size, height=size, highlightthickness=0)
        self.c.pack(side="left")
        self.dot = self.c.create_oval(2, 2, size - 2, size - 2, fill=C_OFF, outline="")
        self.l = tk.Label(self.f, text=text, anchor="w")
        self.l.pack(side="left", padx=(3, 0))
        self.state = "off"

    def set(self, state, text=None):
        self.state = state
        self.c.itemconfigure(self.dot, fill={"ok": C_OK, "warn": C_WARN, "bad": C_BAD}.get(state, C_OFF))
        if text is not None:
            self.l.configure(text=text)


class GripperPanelUI:
    def __init__(self, tk, ttk, parent, grip: "GG.GripperProxy", log: Callable[[str], None] = print,
                 ask: Callable[[str, str], bool] = lambda a, b: True, visible_fn=lambda: True, teach_fn=None):
        self.tk, self.ttk, self.parent, self.grip = tk, ttk, parent, grip
        self.log, self.ask, self.visible_fn = log, ask, visible_fn
        self.teach_fn = teach_fn or (lambda: None)           # -> True / False / None: is the arm in teach mode
        self.q: "queue.Queue" = queue.Queue()
        self.busy: Optional[str] = None
        self.alive = True
        self.zero_step = 0
        self.j7_orig = None
        self.j7_test = {"running": False, "max_abs": 0.0, "t_end": 0.0}
        self.cfg = GG.load_cfg()
        self._build()
        GG.add_listener(lambda t: self.q.put(("call", self._on_switched)))
        GG.add_blocker(lambda: f"夹爪面板正在{self.busy}" if self.busy else None)
        self._last_poll = 0.0
        self.parent.after(200, self._tick)

    # ================================================================== layout
    def _build(self):
        tk, ttk = self.tk, self.ttk
        top = ttk.Frame(self.parent, padding=(6, 4))
        top.pack(fill="x")
        ttk.Label(top, text="当前夹爪", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        self.v_type = tk.StringVar(value=GG.LABELS[GG.active_type()])
        self.cb_type = ttk.Combobox(top, textvariable=self.v_type, values=[GG.LABELS[t] for t in GG.TYPES], width=28,
                                    state="readonly")
        self.cb_type.pack(side="left", padx=(6, 12))
        self.cb_type.bind("<<ComboboxSelected>>", lambda _e: self.on_type())
        self.lamp_conn = Lamp(tk, top, "未连接", 14)
        self.lamp_conn.f.pack(side="left", padx=(0, 12))
        ttk.Label(top, text="CAN").pack(side="left")
        self.v_chan = tk.StringVar(value=self.cfg["agx"].get("channel", "can0"))
        self.ent_chan = ttk.Entry(top, textvariable=self.v_chan, width=7)
        self.ent_chan.pack(side="left", padx=(3, 8))
        self.btn_conn = ttk.Button(top, text="连接", command=self.on_connect, width=8)
        self.btn_conn.pack(side="left")
        ttk.Button(top, text="断开", command=self.on_disconnect, width=8).pack(side="left", padx=4)
        ttk.Label(top, text="指尖到法兰").pack(side="left", padx=(14, 2))     # V4.8.1: measured, used by the planners
        self.v_tip = tk.StringVar(value=f"{float(self.cfg['agx'].get('tip_from_flange_mm') or 144.0):.0f}")
        ttk.Entry(top, textvariable=self.v_tip, width=5).pack(side="left")
        ttk.Label(top, text="mm").pack(side="left", padx=(2, 4))
        ttk.Button(top, text="保存", command=self.on_tip_save, width=5).pack(side="left")
        self.v_note = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.v_note, foreground=C_TXT2).pack(side="left", padx=10)

        self.body = ttk.Frame(self.parent, padding=(6, 0))
        self.body.pack(fill="both", expand=True)
        self.so101_hint = ttk.Label(self.parent, foreground=C_TXT2, padding=(12, 8), justify="left", text=(
            "当前是 SO-101 夹爪：连接、空夹标定和测试在「Agentic实机任务」页的夹爪区。\n"
            "这一页的状态 / 操作 / 校正归零 / 力控标定 是给 NERO 原装 AgxGripper 用的：在上面切换夹爪。"))
        self.body.columnconfigure(2, weight=1)

        # -- column 1: status
        c1 = ttk.LabelFrame(self.body, text="状态", padding=8)
        c1.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.v_width = tk.StringVar(value="-- mm")
        tk.Label(c1, textvariable=self.v_width, font=("TkDefaultFont", 22, "bold"), fg="#1565c0", width=9,
                 anchor="w").grid(row=0, column=0, columnspan=2, sticky="w")
        self.v_width_sub = tk.StringVar(value="开口（行程 0–100 mm）")
        ttk.Label(c1, textvariable=self.v_width_sub, foreground=C_TXT2).grid(row=1, column=0, columnspan=2, sticky="w")
        self.cv_bar = tk.Canvas(c1, width=230, height=14, bg="#eceff1", highlightthickness=0)
        self.cv_bar.grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 8))
        self.bar = self.cv_bar.create_rectangle(0, 0, 0, 14, fill="#64b5f6", outline="")
        self.v_force = tk.StringVar(value="力 --")
        ttk.Label(c1, textvariable=self.v_force, font=("TkDefaultFont", 11)).grid(row=3, column=0, columnspan=2, sticky="w")
        self.lamp_en = Lamp(tk, c1, "使能")
        self.lamp_en.f.grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.lamp_home = Lamp(tk, c1, "已归零")
        self.lamp_home.f.grid(row=4, column=1, sticky="w", pady=(6, 0))
        ttk.Separator(c1).grid(row=5, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Label(c1, text="故障", foreground=C_TXT2).grid(row=6, column=0, columnspan=2, sticky="w")
        self.fault_lamps = {}
        for i, (k, cn) in enumerate(GG.FAULTS):
            lp = Lamp(tk, c1, cn)
            lp.f.grid(row=7 + i // 2, column=i % 2, sticky="w", pady=1, padx=(0, 10))
            self.fault_lamps[k] = lp
        self.v_fps = tk.StringVar(value="")
        ttk.Label(c1, textvariable=self.v_fps, foreground=C_TXT2).grid(row=11, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # -- column 2: operation
        c2 = ttk.LabelFrame(self.body, text="操作", padding=8)
        c2.grid(row=0, column=1, sticky="nsew", padx=(0, 6))
        ttk.Label(c2, text="目标开口 (mm)").grid(row=0, column=0, sticky="w")
        self.v_goal = tk.DoubleVar(value=50.0)
        self.sc_goal = ttk.Scale(c2, from_=0.0, to=100.0, variable=self.v_goal, length=200,
                                 command=lambda v: self.v_goal_txt.set(f"{float(v):.0f}"))
        self.sc_goal.grid(row=1, column=0, sticky="w")
        self.v_goal_txt = tk.StringVar(value="50")
        e = ttk.Entry(c2, textvariable=self.v_goal_txt, width=6)
        e.grid(row=1, column=1, padx=4)
        e.bind("<Return>", lambda _e: self._goal_from_text())
        ttk.Button(c2, text="到这个开口", command=self.on_goto, width=11).grid(row=1, column=2)
        ttk.Label(c2, text="力度（0.5–3.0，SDK 设定值）").grid(row=2, column=0, sticky="w", pady=(10, 0))
        f0 = float(self.cfg["agx"].get("default_force", 1.0))
        self.v_force_set = tk.DoubleVar(value=f0)
        ttk.Scale(c2, from_=0.5, to=3.0, variable=self.v_force_set, length=200,
                  command=lambda v: self.v_force_txt.set(f"{float(v):.2f}")).grid(row=3, column=0, sticky="w")
        self.v_force_txt = tk.StringVar(value=f"{f0:.2f}")
        ttk.Label(c2, textvariable=self.v_force_txt, width=6).grid(row=3, column=1)
        fb = ttk.Frame(c2)
        fb.grid(row=4, column=0, columnspan=3, sticky="w", pady=(2, 0))
        for t, v in (("轻 0.5", 0.5), ("中 1.0", 1.0), ("重 2.0", 2.0)):
            ttk.Button(fb, text=t, width=7, command=lambda v_=v: self._set_force(v_)).pack(side="left", padx=(0, 4))
        ob = ttk.Frame(c2)
        ob.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        for i, (t, fn) in enumerate((("张到最大", lambda: self.on_goto(stroke=True)), ("夹紧（用上面的力度）", self.on_clamp),
                                     ("失能", self.on_disable), ("清除故障", self.on_reset))):
            ttk.Button(ob, text=t, command=fn).grid(row=i // 2, column=i % 2, sticky="ew", padx=(0, 4), pady=2)
        ob.columnconfigure(0, weight=1)
        ob.columnconfigure(1, weight=1)
        ttk.Separator(c2).grid(row=6, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Label(c2, text="物体预设（张开到预设开口；夹紧用预设力度）", foreground=C_TXT2).grid(row=7, column=0, columnspan=3, sticky="w")
        pb = ttk.Frame(c2)
        pb.grid(row=8, column=0, columnspan=3, sticky="w", pady=(2, 0))
        for k, cn in GG.PRESET_KEYS:
            ttk.Button(pb, text=cn, width=11, command=lambda k_=k: self.on_preset(k_)).pack(side="left", padx=(0, 4))
        self.v_last = tk.StringVar(value="")
        ttk.Label(c2, textvariable=self.v_last, wraplength=330, foreground=C_TXT2, justify="left").grid(
            row=9, column=0, columnspan=3, sticky="w", pady=(10, 0))

        # -- column 3: calibration / tests
        c3 = ttk.LabelFrame(self.body, text="标定与测试", padding=6)
        c3.grid(row=0, column=2, sticky="nsew")
        self.nb = ttk.Notebook(c3)
        self.nb.pack(fill="both", expand=True)
        self._build_zero(ttk.Frame(self.nb, padding=8))
        self._build_force(ttk.Frame(self.nb, padding=8))
        self._build_teach(ttk.Frame(self.nb, padding=8))
        self._build_j7(ttk.Frame(self.nb, padding=8))
        self._on_switched()

    def _build_zero(self, f):
        ttk = self.ttk
        self.nb.add(f, text="① 校正归零")
        ttk.Label(f, text="归零 = 把两指合拢到底的位置记为 0 mm。\n第一次装上夹爪、或开口读数不准时做一次。",
                  justify="left").pack(anchor="w")
        self.zero_rows = []
        box = ttk.Frame(f)
        box.pack(fill="x", pady=(8, 4))
        for i, t in enumerate(ZERO_STEPS):
            lp = Lamp(self.tk, box, f"{i + 1}. {t}")
            lp.f.grid(row=i, column=0, sticky="w", pady=2)
            v = self.tk.StringVar(value="")
            self.ttk.Label(box, textvariable=v, foreground=C_TXT2).grid(row=i, column=1, sticky="w", padx=8)
            self.zero_rows.append((lp, v))
        bb = ttk.Frame(f)
        bb.pack(fill="x", pady=(6, 0))
        self.btn_zero = ttk.Button(bb, text="开始归零", command=self.on_zero_next, width=16)
        self.btn_zero.pack(side="left")
        ttk.Button(bb, text="取消", command=self.on_zero_cancel, width=8).pack(side="left", padx=6)
        self.v_zero = self.tk.StringVar(value=self._zero_saved_text())
        ttk.Label(f, textvariable=self.v_zero, foreground=C_TXT2, wraplength=430, justify="left").pack(anchor="w", pady=(8, 0))

    def _build_force(self, f):
        ttk = self.ttk
        self.nb.add(f, text="② 力控标定")
        lf1 = ttk.LabelFrame(f, text="A. 空载基线（不需要设备；夹爪里不要放东西）", padding=6)
        lf1.pack(fill="x")
        ttk.Button(lf1, text="开始空载基线", command=self.on_baseline).pack(side="left")
        self.v_base = self.tk.StringVar(value=self._baseline_text())
        ttk.Label(lf1, textvariable=self.v_base, foreground=C_TXT2, wraplength=330).pack(side="left", padx=8)
        lf2 = ttk.LabelFrame(f, text="B. 测力计对照（可选：设定值 → 指尖实际力 N）", padding=6)
        lf2.pack(fill="x", pady=(6, 0))
        r = ttk.Frame(lf2)
        r.pack(fill="x")
        ttk.Label(r, text="设定值").pack(side="left")
        self.v_fs = self.tk.StringVar(value="1.0")
        ttk.Combobox(r, textvariable=self.v_fs, values=[f"{v:.1f}" for v in FORCE_STEPS], width=5,
                     state="readonly").pack(side="left", padx=4)
        ttk.Button(r, text="夹紧测力计", command=self.on_gauge_clamp).pack(side="left", padx=4)
        ttk.Label(r, text="读数 N").pack(side="left", padx=(8, 0))
        self.v_gauge = self.tk.StringVar(value="")
        ttk.Entry(r, textvariable=self.v_gauge, width=7).pack(side="left", padx=4)
        ttk.Button(r, text="记录", command=self.on_gauge_record).pack(side="left")
        ttk.Button(r, text="清空", command=self.on_gauge_clear).pack(side="left", padx=4)
        self.tv_gauge = ttk.Treeview(lf2, columns=("s", "n"), show="headings", height=4)
        for c, t in (("s", "设定值"), ("n", "测得 N")):
            self.tv_gauge.heading(c, text=t)
            self.tv_gauge.column(c, width=90, anchor="center")
        self.tv_gauge.pack(anchor="w", pady=(4, 0))
        self.v_gauge_fit = self.tk.StringVar(value="")
        ttk.Label(lf2, textvariable=self.v_gauge_fit, foreground=C_TXT2).pack(anchor="w")
        lf3 = ttk.LabelFrame(f, text="C. 物体预设（在真实物体上试，合适就保存）", padding=6)
        lf3.pack(fill="x", pady=(6, 0))
        self.preset_vars = {}
        ttk.Label(lf3, text="物体").grid(row=0, column=0, sticky="w")
        ttk.Label(lf3, text="张开 mm").grid(row=0, column=1)
        ttk.Label(lf3, text="力度").grid(row=0, column=2)
        for i, (k, cn) in enumerate(GG.PRESET_KEYS):
            p = self.cfg["presets"].get(k, {})
            vo = self.tk.StringVar(value=f"{float(p.get('open_mm', 60)):.0f}")
            vf = self.tk.StringVar(value=f"{float(p.get('force', 1.0)):.2f}")
            ttk.Label(lf3, text=cn).grid(row=i + 1, column=0, sticky="w", pady=1)
            ttk.Entry(lf3, textvariable=vo, width=6).grid(row=i + 1, column=1, padx=4)
            ttk.Entry(lf3, textvariable=vf, width=6).grid(row=i + 1, column=2, padx=4)
            ttk.Button(lf3, text="试夹", width=6, command=lambda k_=k: self.on_preset_try(k_)).grid(row=i + 1, column=3, padx=2)
            self.preset_vars[k] = (vo, vf)
        ttk.Button(lf3, text="保存预设", command=self.on_preset_save).grid(row=4, column=0, columnspan=4, sticky="w", pady=(4, 0))
        self._fill_gauge()

    def _build_teach(self, f):
        ttk = self.ttk
        self.nb.add(f, text="③ 示教模式测试")
        self.ttk.Label(f, wraplength=430, justify="left", text=(
            "检查：机械臂在示教（零力拖动）模式时，夹爪还能不能开合。官方文档没写，要在真机上试一次。\n\n"
            "1. 在「关节控制」页进入示教模式（用手扶住机械臂）\n2. 点下面的按钮：夹爪会动 10 mm 再回来\n"
            "3. 结果会保存；Smol VLA 示教录制时的空格键就靠它")).pack(anchor="w")
        self.ttk.Button(f, text="测试：示教模式下夹爪能动吗", command=self.on_teach_test).pack(anchor="w", pady=(10, 0))
        self.v_teach = self.tk.StringVar(value=self._teach_saved_text())
        self.ttk.Label(f, textvariable=self.v_teach, foreground=C_TXT2, wraplength=430).pack(anchor="w", pady=(8, 0))

    def _build_j7(self, f):
        ttk = self.ttk
        self.nb.add(f, text="④ J7 控制器限位")
        ttk = self.ttk
        ttk.Label(f, wraplength=430, justify="left", text=(
            "把 J7 的角度限位写进 NERO 控制器（pyAgxArm set_joint_angle_vel_limits），再在示教模式下拖动 J7，"
            "看拖动时控制器会不会把它挡住。写入前会先读出原来的限位，测完可以恢复。")).pack(anchor="w")
        r = ttk.Frame(f)
        r.pack(fill="x", pady=(8, 0))
        ttk.Label(r, text="限位").pack(side="left")
        self.v_j7lo = self.tk.StringVar(value="-87")
        self.v_j7hi = self.tk.StringVar(value="87")
        ttk.Entry(r, textvariable=self.v_j7lo, width=6).pack(side="left", padx=3)
        ttk.Label(r, text="~").pack(side="left")
        ttk.Entry(r, textvariable=self.v_j7hi, width=6).pack(side="left", padx=3)
        ttk.Label(r, text="°").pack(side="left")
        r2 = ttk.Frame(f)
        r2.pack(fill="x", pady=(6, 0))
        for t, fn in (("读取", self.on_j7_read), ("写入", self.on_j7_write), ("恢复原来的", self.on_j7_restore)):
            ttk.Button(r2, text=t, command=fn, width=11).pack(side="left", padx=(0, 4))
        r3 = ttk.Frame(f)
        r3.pack(fill="x", pady=(10, 0))
        ttk.Button(r3, text="开始拖动测试（30 s）", command=self.on_j7_test).pack(side="left")
        ttk.Button(r3, text="结束", command=self.on_j7_stop, width=8).pack(side="left", padx=6)
        self.v_j7 = self.tk.StringVar(value=self._j7_saved_text())
        ttk.Label(f, textvariable=self.v_j7, foreground=C_TXT2, wraplength=430, justify="left").pack(anchor="w", pady=(8, 0))

    # ================================================================== helpers
    def agx(self):
        return self.grip.devices["agx_gripper"]

    def _run(self, name, fn, done=None):
        if self.busy:
            self.log(f"[WARN] [V4.8 GRIPPER] 请稍等：正在{self.busy}")
            return False

        def job():
            try:
                r = fn()
            except Exception as exc:
                r = {"ok": False, "error": str(exc)[:200]}
            self.q.put(("done", (name, r, done)))
        self.busy = name
        threading.Thread(target=job, daemon=True, name=f"v48_grip_{name}").start()
        return True

    def _need_agx(self):
        if not GG.is_agx():
            self.log("[WARN] [V4.8 GRIPPER] 当前是 SO-101 夹爪：在最上面切换到 NERO 原装夹爪")
            return False
        if not self.agx().connected:
            self.log("[WARN] [V4.8 GRIPPER] 先「连接」NERO 原装夹爪")
            return False
        return True

    def _force(self):
        return round(float(self.v_force_set.get()), 2)

    def _set_force(self, v):
        self.v_force_set.set(v)
        self.v_force_txt.set(f"{v:.2f}")

    def _goal_from_text(self):
        try:
            self.v_goal.set(float(self.v_goal_txt.get()))
        except ValueError:
            pass

    def _report(self, what, r):
        ok = bool(r.get("ok", True)) and not r.get("error")
        txt = f"{what}：" + ("完成" if ok else f"没成功（{r.get('error') or r.get('how')}）")
        if r.get("gap_mm") is not None:
            txt += f" · 开口 {r['gap_mm']:.1f} mm"
        if r.get("held") is not None:
            txt += " · 夹住了东西" if r["held"] else " · 没夹到东西"
        self.v_last.set(txt)
        self.log(f"[{'OK' if ok else 'WARN'}] [V4.8 GRIPPER] {txt}")

    # ================================================================== top bar
    def on_type(self):
        want = next((t for t in GG.TYPES if GG.LABELS[t] == self.v_type.get()), None)
        if want is None or want == GG.active_type():
            return
        r = GG.set_active(want)
        if not r.get("ok"):
            self.log(f"[WARN] [V4.8 GRIPPER] {r.get('error')}")
            self.v_type.set(GG.LABELS[GG.active_type()])
            return
        self.log(f"[OK] [V4.8 GRIPPER] 夹爪切换为 {GG.LABELS[want]}。接下来：" + "；".join(
            f"{i + 1}. {t}" for i, t in enumerate(r.get("todo") or [])))

    def _on_switched(self):
        agx = GG.is_agx()
        self.v_type.set(GG.LABELS[GG.active_type()])
        if agx:
            self.so101_hint.pack_forget()
            self.body.pack(fill="both", expand=True)
        else:
            self.body.pack_forget()
            self.so101_hint.pack(fill="x", anchor="w")
        for w in (self.ent_chan, self.btn_conn):
            w.configure(state="normal" if agx else "disabled")
        self._show_conn()

    def _show_conn(self):
        g = self.grip.dev
        if g is not None and g.connected:
            self.lamp_conn.set("ok", "已连接")
        else:
            self.lamp_conn.set("off", "未连接")

    def on_connect(self):
        if not GG.is_agx():
            return
        ch = self.v_chan.get().strip() or "can0"
        c = GG.load_cfg()
        c["agx"]["channel"] = ch
        GG.save_cfg(c)
        self.agx().cfg["channel"] = ch
        self.lamp_conn.set("warn", "连接中…")

        def fn():
            ok, msg = self.agx().connect(channel=ch)
            return {"ok": ok, "error": None if ok else msg, "msg": msg}

        def done(r):
            self.log(f"[{'OK' if r.get('ok') else 'FAIL'}] [V4.8 GRIPPER] 连接：{r.get('msg') or r.get('error')}")
            self.v_note.set((f"行程 0–{self.agx().stroke_mm:.0f} mm · 夹爪 ID 10" if r.get("ok") else r.get("error")) or "")
            self._show_conn()
            if r.get("ok"):
                self.sc_goal.configure(to=self.agx().stroke_mm)
        self._run("连接", fn, done)

    def on_tip_save(self):
        """V4.8.1: flange face -> fingertips (mm), measured with a ruler; the cup / egg planners use it"""
        try:
            v = float(self.v_tip.get())
        except ValueError:
            self.log("[WARN] [V4.8.1 GRIPPER] 指尖到法兰要是数字（mm）")
            return
        if not 120.0 <= v <= 170.0:
            self.log(f"[WARN] [V4.8.1 GRIPPER] {v:.0f} mm 不合理（官方模型 144 mm，允许 120–170 mm）")
            return
        c = GG.load_cfg()
        c["agx"]["tip_from_flange_mm"] = v
        GG.save_cfg(c)
        self.log(f"[OK] [V4.8.1 GRIPPER] 指尖到法兰 = {v:.0f} mm（官方模型 144 mm）→ 叠杯子 / 鸡蛋任务下次规划起生效")

    def on_disconnect(self):
        if self.busy:
            return
        if self.grip.dev is not None:
            self.grip.dev.disconnect()
        self._show_conn()
        self.log("[INFO] [V4.8 GRIPPER] 已断开")

    # ================================================================== operation
    def on_goto(self, stroke=False):
        if not self._need_agx():
            return
        self._goal_from_text()
        goal = self.agx().stroke_mm if stroke else float(self.v_goal.get())
        f = self._force()
        self._run("移动", lambda: self.agx().open_gap(goal, force=f), lambda r: self._report(f"张开到 {goal:.0f} mm", r))

    def on_clamp(self):
        if not self._need_agx():
            return
        f = self._force()
        self._run("夹紧", lambda: self.agx().clamp(force=f), lambda r: self._report(f"夹紧（力度 {f:.2f}）", r))

    def on_disable(self):
        if not self._need_agx():
            return
        self._run("失能", lambda: dict(zip(("ok", "msg"), self.agx().relax())), lambda r: self._report("失能", r))

    def on_reset(self):
        if not self._need_agx():
            return
        self._run("清除故障", lambda: self.agx().reset_faults(), lambda r: self._report("清除故障", r))

    def on_preset(self, k):
        if not self._need_agx():
            return
        p = GG.load_cfg()["presets"].get(k, {})
        mm, f = float(p.get("open_mm", 60.0)), float(p.get("force", 1.0))
        stroke = self.agx().stroke_mm
        if mm > stroke:                                   # V4.8.1: the 70 mm version cannot open to 95 mm
            self.log(f"[WARN] [V4.8.1 GRIPPER] {dict(GG.PRESET_KEYS)[k]}预设 {mm:.0f} mm 超过这个夹爪的行程 "
                     f"{stroke:.0f} mm → 按 {stroke:.0f} mm；请在 ② 力控标定 → C 改预设")
            mm = stroke
        self._set_force(f)
        self.v_goal.set(mm)
        self.v_goal_txt.set(f"{mm:.0f}")
        self._run("移动", lambda: self.agx().open_gap(mm, force=f),
                  lambda r: self._report(f"{dict(GG.PRESET_KEYS)[k]}预设：张开 {mm:.0f} mm（夹紧力度 {f:.2f}）", r))

    # ================================================================== ① zero
    def _zero_saved_text(self):
        z = GG.load_cfg().get("zero")
        if not z:
            return "还没有做过归零。"
        return (f"上次归零：{z.get('saved')} · 全行程检查 合拢 {z.get('closed_mm', 0):.1f} mm / 最大 "
                f"{z.get('open_mm', 0):.1f} mm（行程 {z.get('stroke_mm', 0):.0f} mm）")

    def _zero_mark(self, i, state, text=""):
        lp, v = self.zero_rows[i]
        lp.set(state)
        v.set(text)

    def on_zero_cancel(self):
        if self.busy and self.busy.startswith("归零"):
            return
        self.zero_step = 0
        for i in range(len(ZERO_STEPS)):
            self._zero_mark(i, "off")
        self.btn_zero.configure(text="开始归零")
        self.log("[INFO] [V4.8 GRIPPER] 归零已取消")

    def on_zero_next(self):
        """one button walks the five steps; step 3 waits for the user ("已合拢")"""
        if not self._need_agx():
            return
        i = self.zero_step
        g = self.agx()
        if i == 0:
            for k in range(len(ZERO_STEPS)):
                self._zero_mark(k, "off")
            st = g.status()
            if st is None:
                self._zero_mark(0, "bad", "没有读数")
                return
            self._zero_mark(0, "ok", f"开口读数 {st['gap_mm']:.1f} mm")
            self.zero_step = 1
            self.btn_zero.configure(text="下一步：失能")
        elif i == 1:
            def done(r):
                if r.get("ok"):
                    self._zero_mark(1, "ok", "已失能")
                    self._zero_mark(2, "warn", "现在用手把两指合拢到底（碰到为止），然后点「已合拢」")
                    self.zero_step = 2
                    self.btn_zero.configure(text="已合拢")
                else:
                    self._zero_mark(1, "bad", r.get("error") or r.get("msg") or "失能失败")
            self._run("归零 · 失能", lambda: dict(zip(("ok", "msg"), g.relax())), done)
        elif i == 2:
            if not self.ask("校正归零", "两指已经用手合拢到底了吗？\n（归零后，这个位置就是 0 mm）"):
                return
            self._zero_mark(2, "ok", "已合拢")

            def done(r):
                if r.get("ok"):
                    self._zero_mark(3, "ok", "夹爪已确认")
                    self.zero_step = 4
                    self.btn_zero.configure(text="下一步：全行程检查")
                else:
                    self._zero_mark(3, "bad", r.get("error"))
            self._run("归零 · 发送", g.calibrate_zero, done)
        elif i == 3:
            self.zero_step = 2
            self.on_zero_next()
        else:
            if not self.ask("全行程检查", "夹爪会合拢 → 张到最大 → 再合拢。两指之间不要有东西、不要放手指。继续？"):
                return

            def done(r):
                if r.get("ok"):
                    self._zero_mark(4, "ok", f"合拢 {r['closed_mm']:.1f} mm · 最大 {r['open_mm']:.1f} mm")
                    c = GG.load_cfg()
                    c["zero"] = {"saved": time.strftime("%Y-%m-%d %H:%M:%S"), "stroke_mm": r["stroke_mm"],
                                 "closed_mm": r["closed_mm"], "open_mm": r["open_mm"]}
                    GG.save_cfg(c)
                    self.v_zero.set(self._zero_saved_text())
                    self.log(f"[OK] [V4.8 GRIPPER] 校正归零完成：合拢 {r['closed_mm']:.1f} mm，最大 {r['open_mm']:.1f} mm")
                else:
                    self._zero_mark(4, "bad", r.get("error"))
                    self.log(f"[WARN] [V4.8 GRIPPER] 全行程检查没通过：{r.get('error')}")
                self.zero_step = 0
                self.btn_zero.configure(text="重新归零")
            self._run("归零 · 全行程检查", g.stroke_check, done)

    # ================================================================== ② force calibration
    def _baseline_text(self):
        b = GG.load_cfg().get("baseline")
        if not b:
            return "还没有做（夹爪自己开合时的力读数 = 摩擦，用来判断「碰到东西」）"
        return f"{b.get('saved')}：空载最大 {b['max_force']:.3f} · 判定阈值 {b['threshold']:.3f}"

    def on_baseline(self):
        if not self._need_agx():
            return
        if not self.ask("空载基线", "夹爪会全行程开合 2 次。两指之间不要有东西。继续？"):
            return

        def done(r):
            if r.get("ok"):
                c = GG.load_cfg()
                c["baseline"] = dict(r, saved=time.strftime("%Y-%m-%d %H:%M:%S"))
                GG.save_cfg(c)
                self.v_base.set(self._baseline_text())
                self.log(f"[OK] [V4.8 GRIPPER] 空载基线：最大 {r['max_force']:.3f}，阈值 {r['threshold']:.3f}（{r['n']} 个读数）")
            else:
                self.v_base.set(f"没成功：{r.get('error')}")
        self._run("空载基线", lambda: self.agx().force_baseline(2), done)

    def on_gauge_clamp(self):
        if not self._need_agx():
            return
        f = float(self.v_fs.get())
        if not self.ask("测力计对照", f"把测力计放在两指之间，夹爪会用力度 {f:.1f} 夹紧。继续？"):
            return
        self._run("夹紧测力计", lambda: self.agx().clamp(force=f),
                  lambda r: self._report(f"夹紧测力计（设定 {f:.1f}）— 读出测力计的数，填进「读数 N」点「记录」", r))

    def on_gauge_record(self):
        try:
            n = float(self.v_gauge.get())
            s = float(self.v_fs.get())
        except ValueError:
            self.log("[WARN] [V4.8 GRIPPER] 读数要是数字（N）")
            return
        c = GG.load_cfg()
        tab = [p for p in (c.get("force_table") or []) if abs(float(p[0]) - s) > 1e-6] + [[s, n]]
        c["force_table"] = sorted(tab)
        GG.save_cfg(c)
        self.v_gauge.set("")
        self._fill_gauge()
        self.log(f"[OK] [V4.8 GRIPPER] 力控对照：设定 {s:.1f} → 实测 {n:.1f} N")

    def on_gauge_clear(self):
        if not self.ask("清空对照表", "清空测力计对照表？以后按线性估算（3.0 ≈ 额定 40 N）。"):
            return
        c = GG.load_cfg()
        c["force_table"] = []
        GG.save_cfg(c)
        self._fill_gauge()

    def _fill_gauge(self):
        tab = GG.load_cfg().get("force_table") or []
        self.tv_gauge.delete(*self.tv_gauge.get_children())
        for s, n in tab:
            self.tv_gauge.insert("", "end", values=(f"{float(s):.1f}", f"{float(n):.1f}"))
        if len(tab) >= 2:
            self.v_gauge_fit.set(f"已标定：要 10 N 时设定 {GG.force_level(10.0, tab):.2f}，20 N 时 {GG.force_level(20.0, tab):.2f}")
        else:
            self.v_gauge_fit.set("未标定（至少 2 个点）：按线性估算，3.0 ≈ 额定 40 N")

    def on_preset_try(self, k):
        if not self._need_agx():
            return
        vo, vf = self.preset_vars[k]
        try:
            mm, f = float(vo.get()), float(vf.get())
        except ValueError:
            self.log("[WARN] [V4.8 GRIPPER] 预设要是数字")
            return

        def fn():
            r1 = self.agx().open_gap(mm, force=f)
            if not r1.get("ok"):
                return r1
            time.sleep(0.8)
            return self.agx().clamp(force=f)
        if not self.ask("试夹", f"夹爪先张开 {mm:.0f} mm，0.8 秒后用力度 {f:.2f} 夹紧：把{dict(GG.PRESET_KEYS)[k]}放到两指之间。继续？"):
            return
        self._run("试夹", fn, lambda r: self._report(f"试夹{dict(GG.PRESET_KEYS)[k]}（{mm:.0f} mm / {f:.2f}）", r))

    def on_preset_save(self):
        c = GG.load_cfg()
        try:
            for k, (vo, vf) in self.preset_vars.items():
                mm, f = float(vo.get()), float(vf.get())
                if not (0.0 <= mm <= 100.0 and GG.AGX["force_min"] <= f <= GG.AGX["force_max"]):
                    raise ValueError(f"{dict(GG.PRESET_KEYS)[k]}：开口 0–100 mm、力度 0.5–3.0")
                c["presets"][k] = {"open_mm": mm, "force": f}
        except ValueError as exc:
            self.log(f"[WARN] [V4.8 GRIPPER] 预设没保存：{exc}")
            return
        GG.save_cfg(c)
        self.log("[OK] [V4.8 GRIPPER] 物体预设已保存：" + "；".join(
            f"{dict(GG.PRESET_KEYS)[k]} {v['open_mm']:.0f} mm / {v['force']:.2f}" for k, v in c["presets"].items()))

    # ================================================================== ③ teach-mode test
    def _teach_saved_text(self):
        t = GG.load_cfg().get("teach_test")
        if not t:
            return "还没有测试。"
        return (f"{t['saved']}：示教模式下夹爪 {'能' if t['follows'] else '不能'} 开合（动了 {t['moved_mm']:.1f} mm）"
                + ("" if t["follows"] else " —— 示教录制时空格键只记录数值，夹爪不动；录完再在回放时动夹爪"))

    def on_teach_test(self):
        if not self._need_agx():
            return
        tm = self.teach_fn()
        if tm is False and not self.ask("示教模式测试", "机械臂现在不在示教模式。仍然测试（结果只说明「非示教模式」下能动）？"):
            return

        def done(r):
            if r.get("ok"):
                c = GG.load_cfg()
                c["teach_test"] = {"saved": time.strftime("%Y-%m-%d %H:%M:%S"), "follows": bool(r["follows"]),
                                   "moved_mm": r["moved_mm"], "in_teach_mode": tm}
                GG.save_cfg(c)
                self.v_teach.set(self._teach_saved_text())
                self.log(f"[{'OK' if r['follows'] else 'WARN'}] [V4.8 GRIPPER] 示教模式测试："
                         f"{'夹爪能动' if r['follows'] else '夹爪不动'}（{r['moved_mm']:.1f} mm）"
                         + ("" if tm else "（机械臂不在示教模式）"))
            else:
                self.v_teach.set(f"没成功：{r.get('error')}")
        self._run("示教模式测试", lambda: self.agx().teach_mode_test(10.0), done)

    # ================================================================== ④ J7 controller limit
    def _j7_saved_text(self):
        t = GG.load_cfg().get("j7_test")
        if not t:
            return "还没有测试。"
        return (f"{t['saved']}：控制器限位 {t['lo']:+.0f}° ~ {t['hi']:+.0f}°，拖动时 J7 最大 {t['max_abs']:.1f}° → "
                + ("控制器在拖动时会挡住 J7" if t["holds"] else "拖动时控制器不挡（只限制主动运动）"))

    def on_j7_read(self):
        if not self._need_agx():
            return

        def done(r):
            if r.get("ok"):
                if self.j7_orig is None:
                    self.j7_orig = (r["min_deg"], r["max_deg"])
                self.v_j7.set(f"控制器现在的 J7 限位：{r['min_deg']:+.1f}° ~ {r['max_deg']:+.1f}°")
            else:
                self.v_j7.set(f"读不到：{r.get('error')}")
        self._run("读取 J7 限位", lambda: self.agx().get_joint_limits(7), done)

    def on_j7_write(self):
        if not self._need_agx():
            return
        try:
            lo, hi = float(self.v_j7lo.get()), float(self.v_j7hi.get())
        except ValueError:
            self.log("[WARN] [V4.8 GRIPPER] 限位要是数字")
            return
        if not self.ask("写入 J7 控制器限位", f"把 NERO 控制器的 J7 限位写成 {lo:+.0f}° ~ {hi:+.0f}°？\n"
                                             "（机械臂不要在运动；原来的限位会先读出来，可以恢复）"):
            return
        g = self.agx()

        def fn():
            if self.j7_orig is None:
                r0 = g.get_joint_limits(7)
                if r0.get("ok"):
                    self.j7_orig = (r0["min_deg"], r0["max_deg"])
            r = g.set_joint_limits(lo, hi, 7)
            r2 = g.get_joint_limits(7)
            return dict(r, readback=r2)

        def done(r):
            rb = r.get("readback") or {}
            ok = r.get("ok") and rb.get("ok") and abs(rb["min_deg"] - lo) < 0.5 and abs(rb["max_deg"] - hi) < 0.5
            self.v_j7.set((f"已写入并读回：{rb['min_deg']:+.1f}° ~ {rb['max_deg']:+.1f}°" if ok else
                           f"没写成功：{r.get('error') or rb.get('error') or '读回的值不对'}")
                          + (f" · 原来 {self.j7_orig[0]:+.1f}° ~ {self.j7_orig[1]:+.1f}°" if self.j7_orig else ""))
            self.log(f"[{'OK' if ok else 'WARN'}] [V4.8 J7] {self.v_j7.get()}")
        self._run("写入 J7 限位", fn, done)

    def on_j7_restore(self):
        if not self._need_agx():
            return
        lo, hi = self.j7_orig if self.j7_orig else (-90.0, 90.0)
        if not self.ask("恢复 J7 限位", f"把 J7 控制器限位恢复成 {lo:+.1f}° ~ {hi:+.1f}°？"):
            return
        self._run("恢复 J7 限位", lambda: self.agx().set_joint_limits(lo, hi, 7),
                  lambda r: (self.v_j7.set(f"已恢复 {lo:+.1f}° ~ {hi:+.1f}°" if r.get("ok") else f"没恢复：{r.get('error')}"),
                             self.log(f"[INFO] [V4.8 J7] {self.v_j7.get()}")))

    def on_j7_test(self, seconds=30.0):
        if not self._need_agx():
            return
        try:
            lo, hi = float(self.v_j7lo.get()), float(self.v_j7hi.get())
        except ValueError:
            return
        if self.teach_fn() is False and not self.ask("拖动测试", "机械臂不在示教模式：拖不动 J7。仍然开始？"):
            return
        self.j7_test = {"running": True, "max_abs": 0.0, "t_end": time.time() + seconds, "lo": lo, "hi": hi, "n": 0}
        self.log(f"[INFO] [V4.8 J7] 拖动测试开始（{seconds:.0f} s）：用手把 J7 往一个方向转到底，看能不能超过 {hi:+.0f}°")
        threading.Thread(target=self._j7_watch, daemon=True, name="v48_j7_watch").start()

    def on_j7_stop(self):
        self.j7_test["t_end"] = 0.0

    def _j7_watch(self):
        t = self.j7_test
        while t["running"] and time.time() < t["t_end"] and self.alive:
            v = self.agx().joint7_deg()
            if v is not None:
                t["max_abs"] = max(t["max_abs"], abs(v))
                t["n"] += 1
                self.q.put(("j7", (v, t["max_abs"])))
            time.sleep(0.05)
        t["running"] = False
        self.q.put(("j7_done", dict(t)))

    def _j7_done(self, t):
        lim = max(abs(t["lo"]), abs(t["hi"]))
        holds = t["max_abs"] <= lim + 1.0
        if t["n"] == 0:
            self.v_j7.set("没有读到 J7 角度（机械臂反馈？）")
            return
        c = GG.load_cfg()
        c["j7_test"] = {"saved": time.strftime("%Y-%m-%d %H:%M:%S"), "lo": t["lo"], "hi": t["hi"],
                        "max_abs": round(t["max_abs"], 1), "holds": holds}
        GG.save_cfg(c)
        self.v_j7.set(self._j7_saved_text())
        self.log(f"[{'OK' if holds else 'WARN'}] [V4.8 J7] {self.v_j7.get()}")

    # ================================================================== status loop
    def _show_status(self, st):
        if st is None:
            self.v_width.set("-- mm")
            self.v_force.set("力 --")
            return
        stroke = float(st.get("stroke_mm") or 100.0)
        self.v_width.set(f"{st['gap_mm']:.1f} mm")
        self.v_width_sub.set(f"开口（行程 0–{stroke:.0f} mm）")
        self.cv_bar.coords(self.bar, 0, 0, 230 * max(0.0, min(1.0, st["gap_mm"] / stroke)), 14)
        self.v_force.set(f"力 {st['force']:.2f}（设定值单位）")
        self.lamp_en.set("ok" if st["enabled"] else "off", "已使能" if st["enabled"] else "未使能")
        self.lamp_home.set("ok" if st["homed"] else "warn", "已归零" if st["homed"] else "未归零")
        bad = set(st.get("faults") or [])
        for k, cn in GG.FAULTS:
            self.fault_lamps[k].set("bad" if cn in bad else "ok")
        self.v_fps.set(f"反馈 {st.get('fps') or 0:.0f} Hz")

    def _poll(self):
        st = self.agx().status() if (GG.is_agx() and self.agx().connected) else None
        self.q.put(("status", st))

    def _tick(self):
        if not self.alive:
            return
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "status":
                    self._polling = False
                    self._show_status(val)
                elif kind == "done":
                    name, r, done = val
                    self.busy = None
                    if done:
                        done(r)
                    self._show_conn()
                elif kind == "call":
                    val()
                elif kind == "j7":
                    v, mx = val
                    self.v_j7.set(f"J7 现在 {v:+.1f}° · 拖动中最大 {mx:.1f}°")
                elif kind == "j7_done":
                    self._j7_done(val)
        except queue.Empty:
            pass
        now = time.time()
        if (GG.is_agx() and self.agx().connected and now - self._last_poll > 0.25 and not getattr(self, "_polling", False)
                and self.visible_fn()):
            self._last_poll, self._polling = now, True
            threading.Thread(target=self._poll, daemon=True, name="v48_grip_poll").start()
        try:
            self.parent.after(100, self._tick)
        except Exception:
            pass

    def shutdown(self):
        self.alive = False
        self.j7_test["running"] = False
