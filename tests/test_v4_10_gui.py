#!/usr/bin/env python3
"""NERO V4.10 GUI test (needs a display; tools/dev/run_tests.sh <dir> gui uses xvfb-run): the console's LLM settings
panel (build, provider switch keeps one key per provider, lock while running, env) and the scene dialog (fixed
household scene, random seed, original scene, cancel)."""
import os
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
import tkinter as tk                     # noqa: E402
from tkinter import ttk                  # noqa: E402

import NERO_V410_Console_Sim as CS       # noqa: E402

root = tk.Tk()
root.geometry("1300x300")
logs = []
p = CS.LLMSettingsPanel(tk, ttk, root, root, log=logs.append)
p.frame.pack(fill="x")
root.update()
p.v_key.set("sk-openai")
p.v_prov.set(p.labels["deepseek"])
p._on_provider()
assert p.prov == "deepseek" and p.v_key.get() == "" and p.v_model.get().startswith("deepseek"), (p.prov, p.v_model.get())
p.v_key.set("sk-ds")
p.v_prov.set(p.labels["openai"])
p._on_provider()
assert p.v_key.get() == "sk-openai"
e = p.env()
assert e["OPENAI_API_KEY"] == "sk-openai" and "DEEPSEEK_API_KEY" not in e
assert "sk-openai" not in p.summary()
p.set_locked(True)
assert str(p.e_key.cget("state")) == "disabled" and str(p.c_prov.cget("state")) == "disabled"
p.set_locked(False)
assert str(p.c_prov.cget("state")) == "readonly"
print("[OK] LLM panel: one key per provider, key only in its provider variable and never in the summary, lock / unlock")


def answer(action):
    def go():
        root.update()
        dlg = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][-1]
        action(dlg)
    root.after(700, go)


def find(w, cls, text=None):
    out = []
    for c in w.winfo_children():
        if isinstance(c, cls) and (text is None or str(c.cget("text")).startswith(text)):
            out.append(c)
        out += find(c, cls, text)
    return out


answer(lambda d: find(d, ttk.Button, "确定")[0].invoke())
assert CS.ask_scene(tk, ttk, root, current="fixed:H2") == ("household_pick_place", "fixed:H2")
answer(lambda d: find(d, ttk.Button, "确定")[0].invoke())
assert CS.ask_scene(tk, ttk, root, current="seed:42") == ("household_pick_place", "seed:42")


def pick_original(d):
    find(d, ttk.Radiobutton, "G0")[0].invoke()
    find(d, ttk.Button, "确定")[0].invoke()
answer(pick_original)
assert CS.ask_scene(tk, ttk, root) == ("general_pick_place", None)
answer(lambda d: find(d, ttk.Button, "取消")[0].invoke())
assert CS.ask_scene(tk, ttk, root) is None
print("[OK] scene dialog: fixed / random seed / original / cancel")

# ---------------------------------------------------------------- bridge on a SimAgent-like panel (backend layout)
import json, tempfile                    # noqa: E402
import NERO_V410_Sim_Bridge as B         # noqa: E402
win = tk.Toplevel(root)
nb = ttk.Notebook(win)
nb.pack(fill="both", expand=True)
outer = ttk.Frame(nb)
nb.add(outer, text="NERO SimAgent")
st = tk.StringVar(win, value="Agent: IDLE | model=x | API=READY | skill=IDLE | LLM agent: IDLE")
ttk.Label(outer, textvariable=st).pack()
cfg = ttk.LabelFrame(outer, text=B.CFG_TITLE)
cfg.pack()
mv, ev, sv, dv, kv = (tk.StringVar(win, "gpt-5.6-luna"), tk.StringVar(win, "low"), tk.StringVar(win, "60"),
                      tk.BooleanVar(win, False), tk.StringVar(win, "typed-key"))
pv = tk.StringVar(win, "OpenAI")
ttk.Label(cfg, text="LLM model (editable)").grid(row=0, column=0)
ttk.Combobox(cfg, textvariable=mv).grid(row=0, column=1)
ttk.Button(cfg, text="Refresh API Models").grid(row=0, column=2)
ttk.Combobox(cfg, textvariable=ev, state="readonly").grid(row=0, column=4)
ttk.Spinbox(cfg, textvariable=sv, from_=1, to=100).grid(row=0, column=6)
ttk.Checkbutton(cfg, text="Dry run", variable=dv).grid(row=0, column=7)
ttk.Entry(cfg, textvariable=kv, show="*").grid(row=1, column=2)
pc = ttk.Combobox(cfg, textvariable=pv, state="readonly")
pc.grid(row=2, column=1)
switched = []
pc.bind("<<ComboboxSelected>>", lambda e: switched.append(pv.get()))
tb = ttk.LabelFrame(outer, text=B.TASK_TITLE)
tb.pack()
tv = tk.StringVar(win, "")
ttk.Entry(tb, textvariable=tv).pack()
runs = []


def run_cmd():
    runs.append(tv.get())
    st.set("Agent: BUSY | model=x | API=READY | skill=IDLE | LLM agent: RUN TASK (general_pick_place)...")


ctl = ttk.LabelFrame(outer, text="LLM agent controls")
ctl.pack()
ttk.Button(ctl, text="RUN TASK", command=run_cmd).pack()
root.update()
W = B.find_widgets(root)
assert all(k in W for k in ("cfg_frame", "model", "effort", "steps", "dry", "key", "provider", "task", "run", "status")), sorted(W)
g = B.getters_from_widgets(root, W)
os.environ.update({"NERO_V410_LLM_FROM_CONSOLE": "1", "NERO_V410_LLM_PROVIDER": "deepseek", "NERO_LLM_MODEL": "deepseek-v4-pro",
                   "NERO_V410_LLM_EFFORT": "max", "NERO_V342_DEFAULT_STEPS": "45", "NERO_V410_LLM_DRY_RUN": "1"})
assert B.apply_console_settings(g, print)
root.update()
assert (pv.get(), mv.get(), ev.get(), sv.get(), bool(dv.get()), kv.get()) == ("DeepSeek", "deepseek-v4-pro", "max", "45", True, ""), \
    (pv.get(), mv.get(), ev.get(), sv.get(), dv.get(), kv.get())
assert switched == ["DeepSeek"]
assert all(str(c.cget("state")) == "disabled" for c in cfg.winfo_children() if c.winfo_class() != "TLabel")
td = tempfile.mkdtemp()
os.environ["NERO_V410_TASK_DIR"] = td
ch = B.TaskChannel(root, g, print)
with open(os.path.join(td, "inbox", "a1.json"), "w") as f:
    json.dump({"id": "a1", "task": "Put apple_1 into bowl_1."}, f)
ch._step()
assert runs == ["Put apple_1 into bowl_1."] and tv.get() == "Put apple_1 into bowl_1."
ch._step()
st.set("Agent: IDLE | model=x | API=READY | skill=IDLE | LLM agent: RUN TASK (general_pick_place) complete")
ch._step()
with open(os.path.join(td, "outbox", "a1.json")) as f:
    assert json.load(f)["status"] == "complete"
print("[OK] bridge: SimAgent widgets found by structure, console settings applied + locked, task file -> RUN TASK -> outbox")
# V4.10.1: the real entry point (attach_from_notebook -> find -> apply) on a fresh panel
mv.set("gpt-5.6-luna")
ev.set("low")
for c in cfg.winfo_children():
    try:
        c.configure(state="normal")
    except Exception:
        pass
B.attach_from_notebook(nb, print)
t_end = time.time() + 4.0
while time.time() < t_end and mv.get() != "deepseek-v4-pro":
    root.update()
    time.sleep(0.05)
assert (mv.get(), ev.get()) == ("deepseek-v4-pro", "max"), (mv.get(), ev.get())
print("[OK] attach_from_notebook: settings applied through the real entry point")
root.destroy()
print("ALL V4.10 GUI TESTS PASSED")
