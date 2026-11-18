#!/usr/bin/env python3
"""NERO V4.11 GUI test (needs a display; xvfb-run): the console's 「仿真 DOE」 tab - build, task rows (H1-H5, random
seeds, edit dialog with the goal check), LLM settings list, plan summary, a short DOE against a fake simulation
(lamps, live statistics, report window), and the console's LLM panel giving the keys of several providers."""
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
import tkinter as tk                     # noqa: E402
from tkinter import ttk                  # noqa: E402

td = tempfile.mkdtemp()
os.environ["NERO_V410_TASK_DIR"] = os.path.join(td, "tasks")
os.environ["NERO_V410_RESET_FILE"] = os.path.join(td, "reset.json")
os.environ["NERO_V411_DOE_DIR"] = os.path.join(td, "doe")
import NERO_V410_Console_Sim as CS       # noqa: E402
import NERO_V410_Household_Scenes as HS  # noqa: E402
import NERO_V411_Sim_DOE as D            # noqa: E402

root = tk.Tk()
root.geometry("1400x900")
if os.environ.get("NERO_TEST_TK_SCALING"):        # e.g. 2.0: large fonts as on a HiDPI screen
    root.tk.call("tk", "scaling", float(os.environ["NERO_TEST_TK_SCALING"]))
    root.geometry("2000x1150")
nb = ttk.Notebook(root)
nb.pack(fill="both", expand=True)
panel = CS.LLMSettingsPanel(tk, ttk, root, root, log=print)
panel.v_key.set("sk-o")
panel.keys["deepseek"] = "sk-d"
env = panel.keys_env(["openai", "deepseek", "kimi"])
assert env == {"OPENAI_API_KEY": "sk-o", "DEEPSEEK_API_KEY": "sk-d"}, env
print("[OK] LLM panel: keys of several providers for the DOE (typed / keyring), each in its own variable")

sim = {"up": False, "starts": [], "env": None}


def start(spec, e):
    sim.update(up=True, t0=time.time(), env=e)
    sim["starts"].append(spec)
    return True, "started"


def stop():
    sim["up"] = False


api = {"start": start, "stop": stop, "running": lambda: sim["up"], "busy": lambda: None,
       "llm_default": lambda: {"provider": "openai", "model": "gpt-6.1-sol", "effort": "high"},
       "key_env": lambda provs: {"OPENAI_API_KEY": "sk-o"}}
tab = D.DOETab(tk, ttk, nb, root, api)
nb.add(tab.frame, text="仿真 DOE")
root.update()
tab.add_fixed()
assert [r["scene"] for r in tab.rows] == [f"fixed:H{k}" for k in range(1, 6)]
tab.v_s0.set("3")
tab.v_s1.set("4")
tab.add_seeds()
assert [r["scene"] for r in tab.rows][-2:] == ["seed:3", "seed:4"]
tab.add_current_model()
tab.v_pprov.set(tab.prov_info["deepseek"]["label"])            # V4.11.1 picker: provider -> its model list
tab._on_pick_provider()
assert tab.v_pmodel.get() in tab.prov_info["deepseek"]["models"] and tab.c_pmodel.cget("values")
tab.v_pmodel.set("deepseek-v4-pro")
tab.v_peff.set("max")
tab.add_picked()
tab.add_picked()                                                # no duplicate
assert "DeepSeek 没有 key" in tab.v_keys.get() and "OpenAI 有" in tab.v_keys.get(), tab.v_keys.get()
assert any("deepseek" in x and "API key" in x for x in [f"提供方 {p} 没有 API key" for p in tab.key_problems()])
assert [D.model_label(m) for m in tab.models] == ["openai:gpt-6.1-sol@high", "deepseek:deepseek-v4-pro@max"]
tab.v_rep.set("2")
tab.update_summary()
assert "7 个任务 × 2 组 LLM × 2 次 = 28 次运行" in tab.v_sum.get(), tab.v_sum.get()
lsp = int(root.tk.call("font", "metrics", "TkDefaultFont", "-linespace"))
assert int(ttk.Style(root).lookup("DOE.Treeview", "rowheight")) >= lsp, "rows must be at least one text line high"
assert tab.tv_sum.column("ci", "width") >= tab.px("95% 区间") - 1
print("[OK] DOE tab: H1-H5 + random seeds as task rows, model picker (provider -> models, effort), key status, "
      "LLM list without duplicates, plan summary, row height from the font")


def find(w, cls, text=None):
    out = []
    for c in w.winfo_children():
        if isinstance(c, cls) and (text is None or str(c.cget("text")).startswith(text)):
            out.append(c)
        out += find(c, cls, text)
    return out


tab.tv_rows.selection_set("0")
tab.edit_row()
root.update()
dlg = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][-1]
ent = [e for e in find(dlg, ttk.Entry) if not isinstance(e, ttk.Combobox)][0]
ent.delete(0, "end")
ent.insert(0, "apple_1 into mug_9")
find(dlg, ttk.Button, "确定")[0].invoke()
root.update()
msg = [lab for lab in find(dlg, ttk.Label) if "mug_9" in str(lab.cget("text"))]
assert msg, "the goal check should name mug_9"
ent.delete(0, "end")
ent.insert(0, "apple_1 into bowl_1")
txt = find(dlg, tk.Text)[0]
txt.delete("1.0", "end")
txt.insert("1.0", "Put the apple in the bowl.")
find(dlg, ttk.Button, "确定")[0].invoke()
root.update()
assert tab.rows[0] == {"scene": "fixed:H1", "task": "Put the apple in the bowl.", "goals": "apple_1 into bowl_1"}
if os.environ.get("NERO_V411_SHOT"):
    nb.select(tab.frame)
    root.update()
    import subprocess
    subprocess.run(["import", "-window", "root", os.environ["NERO_V411_SHOT"]], check=False)
print("[OK] edit dialog: a goal naming an object that is not in the scene is refused, an edited row is kept")

# a short DOE on a fake simulation: 2 rows (H1, H2) x 1 model x 2 replicates
tab.rows = tab.rows[:2]
tab.models = tab.models[:1]
tab._refresh_rows()
tab._refresh_models()
tdir = os.environ["NERO_V410_TASK_DIR"]


def fake_sim():
    if sim["up"]:
        rp = os.path.join(tdir, "ready.json")
        if time.time() - sim["t0"] > 1.0 and (not os.path.exists(rp) or os.path.getmtime(rp) < sim["t0"]):
            with open(rp, "w") as f:
                json.dump({"t": time.time()}, f)
        ib = os.path.join(tdir, "inbox")
        for n in sorted(os.listdir(ib)) if os.path.isdir(ib) else []:
            if n.endswith(".json"):
                with open(os.path.join(ib, n)) as f:
                    m = json.load(f)
                os.remove(os.path.join(ib, n))
                with open(os.path.join(tdir, "outbox", m["id"] + ".json"), "w") as f:
                    json.dump({"id": m["id"], "status": "complete", "llm": m["llm"],
                               "stats": {"tools": [{"tool": "perceive_objects", "ok": True, "safety_stops": 0}]}}, f)
        ef = HS.eval_file()
        if os.path.exists(ef):
            with open(ef) as f:
                req = json.load(f)
            os.remove(ef)
            ok = req["id"].endswith(("r001", "r003"))
            with open(ef + ".done", "w") as f:
                json.dump({"id": req["id"], "status": "success" if ok else "partial", "n_ok": 1 + ok, "n_goals": 2,
                           "objects": {}}, f)
    root.after(200, fake_sim)


fake_sim()
D.DOERunner.__init__.__defaults__  # noqa: B018 (defaults untouched; settle time shortened below)
_orig_init = D.DOERunner.__init__


def _fast_init(self, *a, **k):
    k["timeouts"] = {"settle_s": 1.0, "after_stop_s": 1.0}
    _orig_init(self, *a, **k)


D.DOERunner.__init__ = _fast_init
tab.start()
t_end = time.time() + 90
while time.time() < t_end and not (tab.runner and tab.runner.finished):
    root.update()
    time.sleep(0.03)
assert tab.runner.finished, tab.runner.state
st = [r["status"] for r in tab.runner.runs]
assert sorted(st) == ["partial", "partial", "success", "success"], st
assert sim["starts"] == ["fixed:H1", "fixed:H2"] and sim["env"] == {"OPENAI_API_KEY": "sk-o"}
assert len(tab.lamp_items) == 4
fills = sorted(tab.cv.itemcget(o, "fill") for o in tab.lamp_items)
assert fills == sorted([D.STATUS_COLOR["success"]] * 2 + [D.STATUS_COLOR["partial"]] * 2), fills
rows = [tab.tv_sum.item(i, "values") for i in tab.tv_sum.get_children()]
assert rows[0][0] == "全部" and rows[0][1] in ("4", 4) and rows[0][6] == "50%", rows[0]
reports = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel) and w.title() == "DOE 报告"]
assert reports
out = tab.runner.out
assert os.path.exists(os.path.join(out, "report.md")) and os.path.exists(os.path.join(out, "runs.csv"))
print(f"[OK] DOE on a fake simulation: 4 runs, one start per scene, lamps / statistics / report ({out})")
root.update()
root.destroy()
print("ALL V4.11 GUI TESTS PASSED")
