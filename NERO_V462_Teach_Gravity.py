#!/usr/bin/env python3
"""NERO V4.6.2 TEACH BAR + GRAVITY COMPENSATION (「高级」).

* TeachBar: the buttons 「进入示教模式」 / 「退出示教模式」 / 「高级」 + a status line.  The same bar is placed in the
  「Agentic实机任务」 tab and in the 「Smol VLA实机任务」 tab.  Enter / exit use the very same pipeline calls as the
  existing buttons of the 实机NERO调试 tab (RealPipeline.set_teach_mode; exit holds the current pose, no move home).
* 「高级」 opens a dialog for the end load the controller uses for its gravity compensation in teach mode
  (SO-101 gripper + small bracket + USB camera):
    - load class empty / half / full / 不设置 (as V3.70, sent with set_payload & co. when the driver offers it);
    - optional custom mass (kg) + centre of mass (mm, flange frame).  Only sent when the installed pyAgxArm driver has
      an end-load call with mass parameters; otherwise the class is sent and the log says so;
    - 「检测驱动接口」 lists the load / payload / gravity calls of the installed driver with their parameters;
    - 「保存并立即生效」 re-enters teach mode when NERO is in teach mode, so the new value is used right away.
  Rule of thumb in the dialog: the arm springs back UP after you press it down = compensation too large -> lighter class
  / smaller mass; the arm sinks DOWN after you let go = compensation too small -> heavier class / larger mass.
* Settings: NERO_CALIB_DIR/teach_gravity.json (never into the release folder).  The console's 「示教负载」 box in the
  实机NERO调试 parameters and this dialog edit the same setting.
Tk is only touched on the Tk main thread: worker threads hand results back through a queue polled with after().
"""
from __future__ import annotations

import json
import math
import os
import queue
import threading

MARKER_V462 = "V4.6.2 TEACH BAR + GRAVITY COMP"
CLASSES = ("empty", "half", "full", "none")
CLASS_LABELS = {"empty": "empty（空载 / 最轻）", "half": "half（半载）", "full": "full（满载）",
                "none": "不设置（沿用控制器当前值）"}
DEFAULTS = {"mode": "class", "payload": "empty", "mass_kg": 0.35, "com_mm": [0.0, 0.0, 60.0]}
MASS_RANGE_KG = (0.0, 3.0)
COM_RANGE_MM = 500.0
_listeners = []


def settings_path():
    root = os.environ.get("NERO_CALIB_DIR") or os.path.expanduser("~/Downloads/cameracalib")
    return os.path.join(root, "teach_gravity.json")


def _norm_class(v):
    v = str(v or "").strip().lower()
    if v in ("", "不设置", "off", "skip"):
        return "none"
    return v if v in CLASSES else "empty"


def normalize(raw):
    """-> a clean settings dict (bad values fall back to the defaults, never raises)."""
    s = dict(DEFAULTS)
    s["com_mm"] = list(DEFAULTS["com_mm"])
    if not isinstance(raw, dict):
        return s
    s["mode"] = "custom" if str(raw.get("mode", "")).lower() == "custom" else "class"
    s["payload"] = _norm_class(raw.get("payload", s["payload"]))
    try:
        m = float(raw.get("mass_kg", s["mass_kg"]))
        if math.isfinite(m) and MASS_RANGE_KG[0] <= m <= MASS_RANGE_KG[1]:
            s["mass_kg"] = round(m, 3)
    except (TypeError, ValueError):
        pass
    try:
        c = [float(v) for v in raw.get("com_mm")]
        if len(c) == 3 and all(math.isfinite(v) and abs(v) <= COM_RANGE_MM for v in c):
            s["com_mm"] = [round(v, 1) for v in c]
    except (TypeError, ValueError):
        pass
    return s


def load(path=None):
    p = path or settings_path()
    try:
        with open(p, encoding="utf-8") as f:
            return normalize(json.load(f))
    except Exception:
        return normalize(None)


def save(s, path=None):
    """-> (ok, path or error).  Notifies the listeners (Tk main thread callers only)."""
    s = normalize(s)
    p = path or settings_path()
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        os.replace(tmp, p)
    except Exception as exc:
        return False, str(exc)
    for fn in list(_listeners):
        try:
            fn(s)
        except Exception:
            pass
    return True, p


def add_listener(fn):
    _listeners.append(fn)


def payload_arg(s):
    """settings -> the --payload value of the mode helper: class name, 'none', or JSON for a custom mass."""
    s = normalize(s)
    if s["mode"] == "custom":
        return json.dumps({"mass_kg": s["mass_kg"], "com_mm": s["com_mm"], "fallback": s["payload"]},
                          separators=(",", ":"))
    return s["payload"]


def describe(s):
    s = normalize(s)
    if s["mode"] == "custom":
        c = s["com_mm"]
        return (f"自定义 {s['mass_kg']:.3f} kg，质心 ({c[0]:+.0f}, {c[1]:+.0f}, {c[2]:+.0f}) mm"
                f"（驱动不支持时改发档位 {s['payload']}）")
    return f"档位 {CLASS_LABELS.get(s['payload'], s['payload'])}"


ADVICE = ("怎么调：\n"
          "  · 往下压机械臂、松手后往上回弹 = 重力补偿偏大 → 选更轻的档位 / 减小质量；\n"
          "  · 松手后机械臂往下掉 = 重力补偿偏小 → 选更重的档位 / 增大质量；\n"
          "  · 每改一次点「保存并立即生效」（在示教模式时会重新进入示教，用新值），再用手试一次。\n"
          "注意：设置是否被控制器接受看日志里「示教负载」一行；[WARN] = 当前 pyAgxArm 驱动没有对应接口，"
          "点「检测驱动接口」看驱动能提供什么。")


# ============================================================================ Tk parts
class TeachBar:
    """「进入示教模式」 「退出示教模式」 「高级」 + status.  `busy_fn()` -> reason string or '' (a task is running)."""

    def __init__(self, tk, ttk, parent, pipe, jpanel=None, log=None, busy_fn=None, safe_tab_fn=None, cam_tab=None,
                 pack=None):
        self.tk, self.ttk, self.pipe, self.jpanel = tk, ttk, pipe, jpanel
        self.log = log or (lambda s: None)
        self.busy_fn = busy_fn or (lambda: "")
        self.safe_tab_fn = safe_tab_fn or (lambda: None)
        self.cam_tab = cam_tab
        self.q = queue.Queue()
        self.worker = None
        self.dialog = None
        fr = ttk.Frame(parent)
        fr.pack(**(pack or {"fill": "x", "padx": 4, "pady": 2}))
        self.frame = fr
        ttk.Label(fr, text="示教：", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        self.btn_enter = tk.Button(fr, text="进入示教模式", bg="#1565c0", fg="white",
                                   font=("TkDefaultFont", 9, "bold"), command=self.on_enter)
        self.btn_enter.pack(side="left")
        self.btn_exit = tk.Button(fr, text="退出示教模式", font=("TkDefaultFont", 9, "bold"), command=self.on_exit)
        self.btn_exit.pack(side="left", padx=4)
        self.btn_adv = ttk.Button(fr, text="高级", command=self.on_advanced)
        self.btn_adv.pack(side="left")
        self.v_state = tk.StringVar(value="")
        ttk.Label(fr, textvariable=self.v_state, font=("TkDefaultFont", 9)).pack(side="left", padx=6)
        self.refresh()
        self._poll()

    # ------------------------------------------------------------ status
    def state_text(self):
        on = bool(getattr(self.pipe, "teach_mode", False))
        run = self.worker is not None and self.worker.is_alive()
        st = "切换中 ..." if run else ("NERO：示教模式（可用手带动）" if on else "NERO：正常控制")
        return f"{st} · 重力补偿：{describe(load())}"

    def refresh(self):
        try:
            self.v_state.set(self.state_text())
        except Exception:
            pass

    def _poll(self):
        try:
            while True:
                fn = self.q.get_nowait()
                try:
                    fn()
                except Exception:
                    pass
        except queue.Empty:
            pass
        self.refresh()
        try:
            self.frame.after(500, self._poll)
        except Exception:
            pass

    def _run(self, name, fn):
        if self.worker is not None and self.worker.is_alive():
            self.log(f"[WARN] [V4.6.2 teach] {name}：上一个示教切换还没结束")
            return False

        def wrap():
            try:
                fn()
            except Exception as exc:
                self.log(f"[FAIL] [V4.6.2 teach] {name}: {exc}")
        self.worker = threading.Thread(target=wrap, daemon=True, name=f"v462_{name}")
        self.worker.start()
        return True

    def _blocked(self):
        why = ""
        try:
            why = self.busy_fn() or ""
        except Exception:
            why = ""
        if not why:
            try:
                if self.pipe.move_busy():
                    why = "机械臂正在运动"
            except Exception:
                pass
        return why

    def _ask(self, title, text):
        from tkinter import messagebox
        return messagebox.askyesno(title, text, parent=self.frame.winfo_toplevel())

    # ------------------------------------------------------------ buttons
    def on_enter(self):
        why = self._blocked()
        if why:
            self.log(f"[WARN] [V4.6.2 teach] 现在不能进入示教模式：{why}（先停止任务）")
            return
        if not self._ask("进入示教模式", "NERO 将进入示教（主导臂 / 联动示教输入）模式，之后可以用手带动机械臂。\n"
                                       f"驱动的控制门会先关闭。重力补偿：{describe(load())}。\n"
                                       "请先用手扶住机械臂（尤其是前臂和末端），确认周围无人。继续？"):
            return
        self.log(f"[INFO] [V4.6.2 teach] 进入示教模式 · 重力补偿：{describe(load())}")
        self._run("进入示教模式", lambda: self.pipe.set_teach_mode(True))

    def on_exit(self):
        why = self._blocked()
        if why:
            self.log(f"[WARN] [V4.6.2 teach] 现在不能退出示教模式：{why}")
            return
        if not self._ask("退出示教模式", "NERO 将退出示教模式并切回正常控制（直接保持并锁在当前姿态，不强制回零位）。\n"
                                       "请扶稳机械臂、手在急停上。继续？"):
            return
        st = None
        try:
            st = self.safe_tab_fn()
        except Exception:
            st = None
        if st is not None and getattr(st, "mode", None) == "teach":    # same as the 实机NERO调试 exit button
            try:
                st._teach_exiting = True
                st.v_mode.set("edit")
                st.on_mode()
                st._teach_exiting = False
            except Exception:
                pass
        if self.cam_tab is not None:
            try:
                self.cam_tab.stop_flag = True
            except Exception:
                pass
        model = getattr(self.jpanel, "model", None)
        home_fn = getattr(st, "exit_home", None) if st is not None else None
        self._run("退出示教模式", lambda: self.pipe.set_teach_mode(False, home_model=model, home_fn=home_fn,
                                                                 return_home=False))

    def on_advanced(self):
        if self.dialog is not None:
            try:
                self.dialog.win.lift()
                return
            except Exception:
                self.dialog = None
        self.dialog = GravityDialog(self)


class GravityDialog:
    """「高级」：teach-mode gravity compensation (end load) of the SO-101 gripper + bracket + USB camera."""

    def __init__(self, bar):
        tk, ttk = bar.tk, bar.ttk
        self.bar = bar
        s = load()
        w = tk.Toplevel(bar.frame.winfo_toplevel())
        w.title("高级：示教模式重力补偿（末端负载）")
        w.protocol("WM_DELETE_WINDOW", self.close)
        self.win = w
        body = ttk.Frame(w, padding=10)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text="末端：SO-101 夹爪 + 小支架 + USB 相机。示教模式下由控制器按这里的负载做重力补偿。",
                  wraplength=560, justify="left").pack(anchor="w")
        self.v_mode = tk.StringVar(value=s["mode"])
        f1 = ttk.LabelFrame(body, text="方式")
        f1.pack(fill="x", pady=(8, 0))
        r = ttk.Frame(f1)
        r.pack(fill="x", padx=6, pady=3)
        ttk.Radiobutton(r, text="负载档位", value="class", variable=self.v_mode, command=self._mode_ui).pack(side="left")
        self.v_class = tk.StringVar(value=CLASS_LABELS[s["payload"]])
        self.cb_class = ttk.Combobox(r, textvariable=self.v_class, values=[CLASS_LABELS[c] for c in CLASSES],
                                     width=28, state="readonly")
        self.cb_class.pack(side="left", padx=6)
        r2 = ttk.Frame(f1)
        r2.pack(fill="x", padx=6, pady=3)
        ttk.Radiobutton(r2, text="自定义质量", value="custom", variable=self.v_mode, command=self._mode_ui).pack(side="left")
        self.v_mass = tk.StringVar(value=f"{s['mass_kg']:g}")
        self.v_cx, self.v_cy, self.v_cz = (tk.StringVar(value=f"{v:g}") for v in s["com_mm"])
        self.ents = []
        for t_, v_, wd in (("质量 kg", self.v_mass, 6), ("  质心 x mm", self.v_cx, 5), (" y mm", self.v_cy, 5),
                           (" z mm", self.v_cz, 5)):
            ttk.Label(r2, text=t_).pack(side="left")
            e = ttk.Entry(r2, textvariable=v_, width=wd)
            e.pack(side="left")
            self.ents.append(e)
        ttk.Label(f1, text="质心在法兰坐标系（z = 沿末端向外）。自定义质量只有驱动提供带质量参数的负载接口时才会发送，"
                           "否则改发上面选中的档位（日志会写明）。", wraplength=560, justify="left").pack(anchor="w", padx=6)
        r3 = ttk.Frame(body)
        r3.pack(fill="x", pady=(6, 0))
        ttk.Label(r3, text="微调：").pack(side="left")
        ttk.Button(r3, text="更轻（回弹时）", command=lambda: self.nudge(-1)).pack(side="left", padx=2)
        ttk.Button(r3, text="更重（下坠时）", command=lambda: self.nudge(+1)).pack(side="left", padx=2)
        ttk.Label(body, text=ADVICE, wraplength=560, justify="left", foreground="#37474f").pack(anchor="w", pady=(6, 0))
        self.v_info = tk.StringVar(value="")
        ttk.Label(body, textvariable=self.v_info, wraplength=560, justify="left",
                  font=("TkFixedFont", 9)).pack(anchor="w", pady=(6, 0))
        bb = ttk.Frame(body)
        bb.pack(fill="x", pady=(8, 0))
        ttk.Button(bb, text="关闭", command=self.close).pack(side="right")
        ttk.Button(bb, text="保存并立即生效", command=lambda: self.on_save(apply_now=True)).pack(side="right", padx=4)
        ttk.Button(bb, text="保存", command=lambda: self.on_save(apply_now=False)).pack(side="right")
        ttk.Button(bb, text="检测驱动接口", command=self.on_probe).pack(side="left")
        self._mode_ui()
        self.v_info.set(f"当前：{describe(s)}\n设置文件：{settings_path()}")

    def _class_key(self):
        lab = self.v_class.get()
        for k, v in CLASS_LABELS.items():
            if v == lab:
                return k
        return _norm_class(lab)

    def _mode_ui(self):
        custom = self.v_mode.get() == "custom"
        for e in self.ents:
            e.configure(state="normal" if custom else "disabled")

    def read(self):
        """-> (settings, error)"""
        raw = {"mode": self.v_mode.get(), "payload": self._class_key()}
        if raw["mode"] == "custom":
            try:
                m = float(self.v_mass.get())
                c = [float(self.v_cx.get()), float(self.v_cy.get()), float(self.v_cz.get())]
            except ValueError:
                return None, "质量 / 质心必须是数字"
            if not (MASS_RANGE_KG[0] <= m <= MASS_RANGE_KG[1]):
                return None, f"质量应在 {MASS_RANGE_KG[0]}–{MASS_RANGE_KG[1]} kg"
            if any(abs(v) > COM_RANGE_MM for v in c):
                return None, f"质心每个分量应在 ±{COM_RANGE_MM:.0f} mm 以内"
            raw.update(mass_kg=m, com_mm=c)
        else:
            cur = load()
            raw.update(mass_kg=cur["mass_kg"], com_mm=cur["com_mm"])
        return normalize(raw), None

    def nudge(self, sign):
        if self.v_mode.get() == "custom":
            try:
                m = float(self.v_mass.get())
            except ValueError:
                m = DEFAULTS["mass_kg"]
            m = min(max(m + 0.05 * sign, MASS_RANGE_KG[0]), MASS_RANGE_KG[1])
            self.v_mass.set(f"{m:.2f}")
            return
        order = ["none", "empty", "half", "full"]
        k = self._class_key()
        if k == "none":
            k = "empty"
        i = order.index(k) + sign
        self.v_class.set(CLASS_LABELS[order[min(max(i, 1), 3)]])

    def on_save(self, apply_now=False):
        s, err = self.read()
        if err:
            self.v_info.set(f"[没有保存] {err}")
            return
        ok, where = save(s)
        if not ok:
            self.v_info.set(f"[没有保存] {where}")
            return
        self.bar.log(f"[OK] [V4.6.2 teach] 示教重力补偿已保存：{describe(s)}（{where}）")
        msg = f"已保存：{describe(s)}"
        if apply_now:
            if getattr(self.bar.pipe, "teach_mode", False):
                why = self.bar._blocked()
                if why:
                    msg += f"\n没有立即生效：{why}"
                else:
                    self.bar.log("[INFO] [V4.6.2 teach] 重新进入示教模式，使用新的重力补偿")
                    if self.bar._run("重新进入示教（新重力补偿）", lambda: self.bar.pipe.set_teach_mode(True)):
                        msg += "\n正在重新进入示教模式 ... 结果看日志「示教负载」一行"
            else:
                msg += "\nNERO 不在示教模式：下次「进入示教模式」时生效"
        self.v_info.set(msg)
        self.bar.refresh()

    def on_probe(self):
        fn = getattr(self.bar.pipe, "probe_teach_apis", None)
        if not callable(fn):
            self.v_info.set("这个版本的实机管线没有接口检测")
            return
        self.v_info.set("正在检测驱动接口（连接 CAN，只读，不切换模式）...")

        def work():
            r = fn()
            self.bar.q.put(lambda: self._show_probe(r))
        threading.Thread(target=work, daemon=True, name="v462_probe").start()

    def _show_probe(self, r):
        txt = probe_text(r)
        self.bar.log("[INFO] [V4.6.2 teach] 驱动接口检测：" + txt.replace("\n", " | "))
        try:
            self.v_info.set(txt)
        except Exception:
            pass

    def close(self):
        try:
            self.win.destroy()
        except Exception:
            pass
        self.bar.dialog = None


def probe_text(r):
    r = r or {}
    if not r.get("ok"):
        return f"检测失败：{r.get('error') or '没有结果'}"
    lines = []
    pay = r.get("payload_apis") or []
    if pay:
        lines.append("负载 / 重力补偿接口：")
        lines += [f"  {a.get('name')}{a.get('sig') or ''}" for a in pay[:8]]
    else:
        lines.append("当前 pyAgxArm 驱动没有负载 / 重力补偿接口：GUI 里的设置无法发给控制器。"
                     "可用 AgileX 上位机设置末端负载，或在夹爪上加 / 减配重。")
    if r.get("custom_ok"):
        lines.append("其中有带质量参数的接口：「自定义质量」可用")
    pay_names = {a.get("name") for a in pay}
    other = [a.get("name") for a in (r.get("apis") or []) if a.get("name") not in pay_names][:12]
    if other:
        lines.append("其他相关接口：" + ", ".join(str(n) for n in other))
    return "\n".join(lines)
