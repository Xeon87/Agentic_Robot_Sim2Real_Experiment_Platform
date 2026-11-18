#!/usr/bin/env python3
"""NERO V4.10 CONSOLE SIM SETTINGS - the LLM settings and the scene choice of the simulation live in the console.

  LLMSettingsPanel  in the console's '仿真 (MuJoCo)' tab: provider, model (+ refresh from the API), reasoning effort,
                    max tool steps, dry run, API key (hidden, show / hide, check, save in / delete from the system
                    keyring).  Passed to the simulation through environment variables at start (never a file, a
                    command line or a log): the backend's NERO SimAgent panel shows them read-only (A2 hook).
  ask_scene()       scene dialog for the general pick & place: the original scene (eggs / carton / cups), the fixed
                    household scenes H1-H5, or a random household scene from a seed; top-view preview of the layout.
  request_reset()   put the objects of the running general / household scene back where they started.
Tk rules of this project: no emoji in widget text, no tkinter.font.Font, worker threads never touch Tk variables.
"""
from __future__ import annotations

import math
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

VERSION = "V4.10"
MARKER = "V4.10 CONSOLE LLM SETTINGS + SCENE DIALOG"
KEYRING_SERVICE = "NERO-console-llm"
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")   # same list as the SimAgent panel
GENERAL_SCENES = ("general_pick_place", "household_pick_place")
ORIGINAL = "G0"


def providers():
    import NERO_V2_7_1_Agent as A
    return A.LLM_PROVIDERS


# ------------------------------------------------------------------------------------------------ keyring
class KeyStore:
    """system keyring (Secret Service / KWallet / macOS Keychain) through the `keyring` package; unavailable ->
    every call is a no-op that says why"""

    def __init__(self):
        self.why = ""
        try:
            import keyring
            from keyring.backends import fail
            kr = keyring.get_keyring()
            if isinstance(kr, fail.Keyring) or "fail" in type(kr).__module__:
                self.kr, self.why = None, "没有可用的系统钥匙串后端"
            else:
                self.kr = keyring
        except Exception as exc:
            self.kr, self.why = None, f"python3-keyring 不可用（{type(exc).__name__}）"

    def available(self):
        return self.kr is not None

    def get(self, provider):
        if not self.kr:
            return ""
        try:
            return self.kr.get_password(KEYRING_SERVICE, provider) or ""
        except Exception as exc:
            self.why = f"读取钥匙串失败：{type(exc).__name__}"
            return ""

    def set(self, provider, key):
        if not self.kr:
            return False
        try:
            self.kr.set_password(KEYRING_SERVICE, provider, key)
            return True
        except Exception as exc:
            self.why = f"写入钥匙串失败：{type(exc).__name__}"
            return False

    def delete(self, provider):
        if not self.kr:
            return False
        try:
            self.kr.delete_password(KEYRING_SERVICE, provider)
            return True
        except Exception as exc:
            self.why = f"钥匙串里没有这一项或删除失败：{type(exc).__name__}"
            return False


def llm_env(provider, model, effort, steps, dry_run, key) -> dict:
    """environment for the simulation (pure logic, tested): the backend reads NERO_LLM_MODEL / NERO_V342_DEFAULT_STEPS
    itself; the V4.10 A2 hook reads the rest; the key goes into the provider's own variable (the agent's fallback)"""
    env = {"NERO_V410_LLM_FROM_CONSOLE": "1", "NERO_V410_LLM_PROVIDER": provider, "NERO_LLM_MODEL": model,
           "NERO_V410_LLM_EFFORT": effort, "NERO_V342_DEFAULT_STEPS": str(int(steps)),
           "NERO_V410_LLM_DRY_RUN": "1" if dry_run else "0"}
    if key:
        names = providers().get(provider, {}).get("key_env") or ("OPENAI_API_KEY",)
        env[names[0]] = key
    return env


# ------------------------------------------------------------------------------------------------ LLM panel
class LLMSettingsPanel:
    def __init__(self, tk, ttk, parent, root, log=None, messagebox=None):
        self.tk, self.ttk, self.root, self.mb = tk, ttk, root, messagebox
        self.log = log or (lambda s: None)
        self.store = KeyStore()
        P = providers()
        self.labels = {k: v["label"] for k, v in P.items()}
        self.by_label = {v: k for k, v in self.labels.items()}
        self.keys = {}                                  # provider -> key typed / loaded in this console run
        self.models = {k: list(v.get("models") or []) for k, v in P.items()}
        try:
            import NERO_V2_7_1_Agent as A
            dm = A.DEFAULT_MODEL
        except Exception:
            dm = "gpt-6.1-sol"
        if dm not in self.models.get("openai", []):
            self.models.setdefault("openai", []).insert(0, dm)
        self.v_prov = tk.StringVar(value=self.labels.get("openai", "OpenAI"))
        self.v_model = tk.StringVar(value=os.environ.get("NERO_LLM_MODEL", dm))
        self.v_effort = tk.StringVar(value="high")
        self.v_steps = tk.StringVar(value="60")
        self.v_dry = tk.BooleanVar(value=False)
        self.v_key = tk.StringVar(value="")
        self.v_show = tk.BooleanVar(value=False)
        self.v_status = tk.StringVar(value="")
        self.prov = "openai"

        f = self.frame = ttk.LabelFrame(parent, text="LLM 设置（仿真用；启动仿真时传给仿真窗口，运行中不可修改）", padding=(10, 6))
        for c in (1, 3):
            f.columnconfigure(c, weight=0)
        f.columnconfigure(9, weight=1)
        ttk.Label(f, text="提供方").grid(row=0, column=0, sticky="w", padx=(0, 4), pady=3)
        self.c_prov = ttk.Combobox(f, textvariable=self.v_prov, values=list(self.labels.values()), state="readonly",
                                   width=16)
        self.c_prov.grid(row=0, column=1, sticky="w", pady=3)
        self.c_prov.bind("<<ComboboxSelected>>", self._on_provider)
        ttk.Label(f, text="模型").grid(row=0, column=2, sticky="w", padx=(14, 4))
        self.c_model = ttk.Combobox(f, textvariable=self.v_model, values=self.models.get("openai", []), width=26)
        self.c_model.grid(row=0, column=3, sticky="w")
        self.b_refresh = ttk.Button(f, text="刷新模型列表", command=self.refresh_models)
        self.b_refresh.grid(row=0, column=4, sticky="w", padx=(6, 0))
        ttk.Label(f, text="思考档位").grid(row=0, column=5, sticky="w", padx=(14, 4))
        self.c_eff = ttk.Combobox(f, textvariable=self.v_effort, values=EFFORTS, state="readonly", width=8)
        self.c_eff.grid(row=0, column=6, sticky="w")
        ttk.Label(f, text="最大工具步数").grid(row=0, column=7, sticky="w", padx=(14, 4))
        self.s_steps = ttk.Spinbox(f, from_=10, to=100, increment=5, textvariable=self.v_steps, width=5)
        self.s_steps.grid(row=0, column=8, sticky="w")
        self.c_dry = ttk.Checkbutton(f, text="Dry run（只规划不动）", variable=self.v_dry)
        self.c_dry.grid(row=0, column=9, sticky="w", padx=(14, 0))

        ttk.Label(f, text="API key").grid(row=1, column=0, sticky="w", padx=(0, 4), pady=3)
        self.e_key = ttk.Entry(f, textvariable=self.v_key, show="*", width=46)
        self.e_key.grid(row=1, column=1, columnspan=3, sticky="we", pady=3)
        self.c_show = ttk.Checkbutton(f, text="显示", variable=self.v_show, command=self._on_show)
        self.c_show.grid(row=1, column=4, sticky="w", padx=(6, 0))
        bar = ttk.Frame(f)
        bar.grid(row=1, column=5, columnspan=5, sticky="w", padx=(14, 0))
        self.b_check = ttk.Button(bar, text="检测 key", command=self.check_key)
        self.b_save = ttk.Button(bar, text="保存到钥匙串", command=self.save_key)
        self.b_del = ttk.Button(bar, text="删除已保存的 key", command=self.delete_key)
        for b in (self.b_check, self.b_save, self.b_del):
            b.pack(side="left", padx=(0, 6))
        ttk.Label(f, textvariable=self.v_status, foreground="#555555").grid(row=2, column=0, columnspan=10, sticky="w")
        self.v_key.trace_add("write", lambda *_: self._update_status())
        self.widgets = [self.c_prov, self.c_model, self.b_refresh, self.c_eff, self.s_steps, self.c_dry, self.e_key,
                        self.c_show, self.b_check, self.b_save, self.b_del]
        self._load_key("openai")

    # ---------------------------------------------------------------- state
    def _source(self):
        k = self.v_key.get().strip()
        if k:
            return "已填写" + ("（来自钥匙串）" if k == self.store.get(self.prov) and k else "")
        envs = providers().get(self.prov, {}).get("key_env") or ()
        if any(os.environ.get(e) for e in envs):
            return f"未填写，将使用环境变量 {' / '.join(envs)}"
        return "未设置"

    def _update_status(self):
        kr = "系统钥匙串可用" if self.store.available() else f"系统钥匙串不可用（{self.store.why}），key 只在本次控制台运行期间有效"
        self.v_status.set(f"API key：{self._source()}  |  {kr}")

    def _load_key(self, prov):
        k = self.keys.get(prov)
        if k is None:
            k = self.store.get(prov)
            self.keys[prov] = k
        self.v_key.set(k)
        self._update_status()

    def _on_provider(self, _e=None):
        new = self.by_label.get(self.v_prov.get(), "openai")
        if new == self.prov:
            return
        self.keys[self.prov] = self.v_key.get().strip()
        self.prov = new
        vals = self.models.get(new) or []
        self.c_model.configure(values=vals)
        self.v_model.set(vals[0] if vals else "")
        self._load_key(new)

    def _on_show(self):
        self.e_key.configure(show="" if self.v_show.get() else "*")

    def set_locked(self, locked):
        for w in self.widgets:
            try:
                if w is self.c_prov or w is self.c_eff:
                    w.configure(state="disabled" if locked else "readonly")
                else:
                    w.configure(state="disabled" if locked else "normal")
            except Exception:
                pass

    # ---------------------------------------------------------------- actions
    def save_key(self):
        k = self.v_key.get().strip()
        if not k:
            self.v_status.set("没有可保存的 key：先在输入框里填写")
            return
        if self.store.set(self.prov, k):
            self.keys[self.prov] = k
            self.log(f"[OK] [V4.10 LLM] {self.labels[self.prov]} API key 已保存到系统钥匙串（值不显示）")
        else:
            self.log(f"[WARN] [V4.10 LLM] 没有保存：{self.store.why}")
        self._update_status()

    def delete_key(self):
        if self.mb is not None and not self.mb.askyesno("删除 API key", f"从系统钥匙串删除已保存的 {self.labels[self.prov]} API key？"):
            return
        ok = self.store.delete(self.prov)
        self.keys[self.prov] = ""
        self.v_key.set("")
        self.log(f"[{'OK' if ok else 'WARN'}] [V4.10 LLM] {self.labels[self.prov]} API key "
                 + ("已从系统钥匙串删除" if ok else f"未删除：{self.store.why}"))
        self._update_status()

    def _agent(self):
        import NERO_V2_7_1_Agent as A
        ag = A.NeroLLMAgent(emit=lambda s: None, tool_runner=None)
        ag.provider = self.prov
        return ag

    def check_key(self):
        """real check: list the models the key can see (GET /models) - in a worker thread"""
        key, prov, label = self.v_key.get().strip(), self.prov, self.labels[self.prov]
        self.v_status.set(f"正在检测 {label} API key…")

        def work():
            try:
                ids = self._agent().list_available_models(api_key=key)
                msg, ok = f"{label} API key 有效：可用模型 {len(ids)} 个", True
            except Exception as exc:
                msg, ok = f"{label} API key 检测失败：{str(exc)[:160]}", False
            self.root.after(0, lambda: (self.v_status.set(msg), self.log(f"[{'OK' if ok else 'WARN'}] [V4.10 LLM] {msg}")))
        threading.Thread(target=work, daemon=True).start()

    def refresh_models(self):
        key, prov = self.v_key.get().strip(), self.prov
        self.v_status.set("正在读取模型列表…")

        def work():
            try:
                ids = self._agent().list_available_models(api_key=key)
                cur = self.models.get(prov) or []
                ordered = [m for m in cur if m in ids] + [m for m in ids if m not in cur]

                def apply():
                    self.models[prov] = ordered
                    if prov == self.prov:
                        self.c_model.configure(values=ordered)
                    self.v_status.set(f"模型列表已更新：{len(ordered)} 个")
                self.root.after(0, apply)
            except Exception as exc:
                self.root.after(0, lambda: self.v_status.set(f"读取模型列表失败：{str(exc)[:160]}"))
        threading.Thread(target=work, daemon=True).start()

    def env(self) -> dict:
        try:
            steps = max(10, min(100, int(float(self.v_steps.get()))))
        except ValueError:
            steps = 60
        return llm_env(self.prov, self.v_model.get().strip(), self.v_effort.get(), steps, bool(self.v_dry.get()),
                       self.v_key.get().strip())

    def keys_env(self, provs) -> dict:
        """V4.11 SIM DOE: the API key of each provider (typed in this console run, else from the keyring) in its own
        environment variable; providers without a key are left out (their environment variable may still be set)"""
        self.keys[self.prov] = self.v_key.get().strip()
        env = {}
        for p in provs:
            k = self.keys.get(p)
            if not k:
                k = self.store.get(p)
            names = providers().get(p, {}).get("key_env") or ()
            if k and names:
                env[names[0]] = k
        return env

    def summary(self) -> str:
        e = self.env()
        return (f"提供方 {self.labels[self.prov]} | 模型 {e['NERO_LLM_MODEL']} | 思考档位 {e['NERO_V410_LLM_EFFORT']} | "
                f"最大工具步数 {e['NERO_V342_DEFAULT_STEPS']} | Dry run {'是' if e['NERO_V410_LLM_DRY_RUN'] == '1' else '否'} | "
                f"API key {self._source()}")


# ------------------------------------------------------------------------------------------------ scene dialog
KIND_COLOR = {"remote": "#5f6368", "apple": "#e57373", "bottle": "#a1887f", "mug": "#ffd54f", "bowl": "#80cbc4",
              "tray": "#cfd8dc", "storage_box": "#90a4ae"}
ROUND = ("apple", "bowl")


def _draw_layout(canvas, scene, W, H):
    """top view (robot at the bottom): the table, the top-down grasp band, each footprint; task objects bold"""
    import NERO_V410_Household_Scenes as HS
    canvas.delete("all")
    sc = 455.0                                         # px per m: the whole table (0.67 x 0.76 m) fits
    ox, oy = W / 2.0, H - 16.0                         # world (0, 0) = robot base

    def px(x, y):
        return ox - y * sc, oy + x * sc                # world -x points up the canvas, +y to the left
    x0, y0 = px(-0.72, 0.38)
    x1, y1 = px(-0.05, -0.38)
    canvas.create_rectangle(x0, y0, x1, y1, fill="#fafaf8", outline="#b0b0b0")
    a0, b0 = px(HS.X_RANGE[1], HS.Y_RANGE[0])
    a1, b1 = px(HS.X_RANGE[0], HS.Y_RANGE[1])
    canvas.create_rectangle(a0, b0, a1, b1, outline="#64b5f6", dash=(4, 3))
    rx, ry = px(0.0, 0.0)
    canvas.create_rectangle(rx - 22, ry - 9, rx + 22, ry + 9, fill="#37474f", outline="")
    canvas.create_text(rx, ry, text="NERO", fill="white")
    canvas.create_text(8, 6, anchor="nw", fill="#555555", width=W - 16,
                       text="俯视图（机器人在下方）\n蓝色虚线 = 俯抓可达带；粗边 = 任务涉及的物体")
    if scene is None:
        canvas.create_text(W / 2, H / 2, text="原通用场景：2 个鸡蛋、4 格蛋托、2 个纸杯", fill="#333333")
        return
    involved = {s for s, _r, _t in scene.goals} | {t for _s, _r, t in scene.goals}
    for it in sorted(scene.items, key=lambda i: i.kind not in ("tray", "storage_box")):   # containers first
        col = KIND_COLOR.get(it.kind, "#bdbdbd")
        width = 3 if it.oid in involved else 1
        cx, cy = px(it.x, it.y)
        if it.kind in ROUND:
            _cx, _cy, hx, _hy = HS.footprint(it.kind)
            r = hx * sc
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=col, outline="#263238", width=width)
        else:
            pts = [px(x, y) for x, y in HS._corners(it)]
            canvas.create_polygon(*[c for p in pts for c in p], fill=col, outline="#263238", width=width)
        canvas.create_text(cx, cy, text=it.oid, fill="#111111")


def ask_scene(tk, ttk, root, current=None):
    """modal dialog -> ("general_pick_place", None) | ("household_pick_place", "fixed:H1" / "seed:12") | None"""
    import NERO_V410_Household_Scenes as HS
    res = {"v": None}
    dlg = tk.Toplevel(root)
    dlg.title("选择通用 Pick&Place 场景")
    dlg.transient(root)
    dlg.resizable(False, False)
    body = ttk.Frame(dlg, padding=12)
    body.pack(fill="both", expand=True)
    left = ttk.Frame(body)
    left.grid(row=0, column=0, sticky="nsw")
    right = ttk.Frame(body)
    right.grid(row=0, column=1, sticky="nsew", padx=(14, 0))
    v_mode = tk.StringVar(value="fixed")
    v_fixed = tk.StringVar(value=ORIGINAL)
    v_seed = tk.StringVar(value=str(int(time.time()) % 10000))
    if current and str(current).startswith("seed:"):
        v_mode.set("random")
        v_seed.set(str(current)[5:])
    elif current and str(current).startswith("fixed:"):
        v_fixed.set(str(current)[6:])

    ttk.Radiobutton(left, text="固定场景", variable=v_mode, value="fixed", command=lambda: update()).pack(anchor="w")
    lf = ttk.Frame(left, padding=(18, 2, 0, 6))
    lf.pack(anchor="w")
    opts = [(ORIGINAL, "G0 原通用场景：鸡蛋 / 蛋托 / 纸杯")] + [(k, HS.FIXED[k]["label"]) for k in HS.scene_ids()]
    for k, lab in opts:
        ttk.Radiobutton(lf, text=lab, variable=v_fixed, value=k, command=lambda: (v_mode.set("fixed"), update())).pack(anchor="w")
    ttk.Radiobutton(left, text="随机生成家居场景（同一个种子总是同一个场景）", variable=v_mode, value="random",
                    command=lambda: update()).pack(anchor="w", pady=(6, 0))
    rf = ttk.Frame(left, padding=(18, 2, 0, 6))
    rf.pack(anchor="w")
    ttk.Label(rf, text="种子").pack(side="left")
    sp = ttk.Spinbox(rf, from_=0, to=999999, textvariable=v_seed, width=8, command=lambda: (v_mode.set("random"), update()))
    sp.pack(side="left", padx=(4, 6))
    sp.bind("<Return>", lambda _e: (v_mode.set("random"), update()))
    ttk.Button(rf, text="换一个", command=lambda: (v_seed.set(str((int(v_seed.get() or 0) * 7919 + 13) % 100000)),
                                                v_mode.set("random"), update())).pack(side="left")
    W, H = 440, 400
    cv = tk.Canvas(right, width=W, height=H, background="white", highlightthickness=1, highlightbackground="#cccccc")
    cv.pack()
    info = tk.Text(right, width=60, height=6, wrap="word", relief="flat", background="#fafafa")
    info.pack(fill="x", pady=(8, 0))
    btns = ttk.Frame(body)
    btns.grid(row=1, column=0, columnspan=2, sticky="e", pady=(12, 0))
    state = {"spec": None}

    def update():
        try:
            if v_mode.get() == "fixed" and v_fixed.get() == ORIGINAL:
                sc, spec, text = None, None, "原通用场景（V4.8.10）：2 个鸡蛋、4 格蛋托、2 个纸杯。"
            elif v_mode.get() == "fixed":
                spec = "fixed:" + v_fixed.get()
                sc = HS.scene_from_spec(spec)
                text = HS.describe(sc)
            else:
                spec = "seed:" + str(int(float(v_seed.get() or 0)))
                sc = HS.scene_from_spec(spec)
                text = HS.describe(sc)
            state["spec"] = spec
            _draw_layout(cv, sc, W, H)
            ok_btn.configure(state="normal")
        except Exception as exc:
            text = f"这个种子生成不出合格的布局：{exc}。请换一个。"
            ok_btn.configure(state="disabled")
        info.configure(state="normal")
        info.delete("1.0", "end")
        info.insert("end", text + "\n\n任务句只是建议，启动后可以在仿真窗口里改。")
        info.configure(state="disabled")

    def ok():
        spec = state["spec"]
        res["v"] = ("general_pick_place", None) if spec is None else ("household_pick_place", spec)
        dlg.destroy()

    ok_btn = ttk.Button(btns, text="确定，启动仿真", command=ok)
    ttk.Button(btns, text="取消", command=dlg.destroy).pack(side="right")
    ok_btn.pack(side="right", padx=(0, 8))
    update()
    dlg.grab_set()
    root.wait_window(dlg)
    return res["v"]


# ------------------------------------------------------------------------------------------------ reset
def request_reset(log, root, timeout_s=8.0):
    """write the reset request for the running simulation, then report its answer (polled in the Tk thread)"""
    import NERO_V410_Household_Scenes as HS
    f = HS.request_reset()
    log("[V4.10 RESET] 请求已发出：等待仿真把物体放回初始位置（仿真每 2 秒检查一次）")
    t0 = time.time()

    def poll():
        if os.path.exists(f + ".done"):
            try:
                import json
                with open(f + ".done") as fh:
                    msg = json.load(fh).get("message", "done")
            except Exception:
                msg = "done"
            log(f"[OK] {msg}")
            return
        if time.time() - t0 > timeout_s:
            log("[WARN] [V4.10 RESET] 仿真没有回应（场景不是通用 / 家居场景，或仿真尚未就绪）")
            try:
                os.remove(f)
            except OSError:
                pass
            return
        root.after(300, poll)
    root.after(300, poll)


def scene_extra_env(scene, spec, reset_file):
    """environment for the chosen scene (pure logic, tested)"""
    env = {"NERO_V410_RESET_FILE": reset_file}
    if scene == "household_pick_place":
        env["NERO_V410_HOUSEHOLD"] = spec or "fixed:H1"
    return env
