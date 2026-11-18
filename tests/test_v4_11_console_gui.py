#!/usr/bin/env python3
"""NERO V4.11 console GUI test (needs tkinter + a display, e.g. xvfb-run; skips otherwise): the real console window
with a fake launcher that plays the simulation's side of the DOE (ready file, task channel, truth check).

Checks: the 「仿真 DOE」 tab sits right after 「仿真 (MuJoCo)」; a DOE of 2 scenes x 1 LLM setting x 2 replicates starts
the simulation once per scene through the console's own start path (household layout from the DOE, no dialog, the
DOE's API key in the launcher's environment and never in the console log), stops it between scenes, and judges every
run.  Optional: NERO_V411_SHOT=<png> saves a screenshot of the DOE tab (ImageMagick `import`).
"""
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
try:
    import tkinter as tk
    from tkinter import messagebox
except Exception:
    print("[GUI] tkinter missing -> skipped")
    sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI] no DISPLAY -> skipped")
    sys.exit(0)

tmp = tempfile.mkdtemp(prefix="v411gui_")
os.environ["NERO_V410_TASK_DIR"] = os.path.join(tmp, "tasks")
os.environ["NERO_V410_RESET_FILE"] = os.path.join(tmp, "reset.json")
os.environ["NERO_V411_DOE_DIR"] = os.path.join(tmp, "doe")
os.environ["OPENAI_API_KEY"] = ""
fake_py = os.path.join(tmp, "fake_sim.py")
with open(fake_py, "w") as f:
    f.write(r'''
import json, os, signal, sys, time
signal.signal(signal.SIGINT, lambda *a: sys.exit(130))
td = os.environ["NERO_V410_TASK_DIR"]
rf = os.environ["NERO_V410_RESET_FILE"]
ef = rf[:-5] + "_eval.json"
for sub in ("inbox", "outbox", "done"):
    os.makedirs(os.path.join(td, sub), exist_ok=True)
with open(os.path.join(tmp_log := os.path.join(td, "launches.jsonl")), "a") as f:
    f.write(json.dumps({"scene": os.environ.get("NERO_SCENE"), "household": os.environ.get("NERO_V410_HOUSEHOLD"),
                        "key": os.environ.get("OPENAI_API_KEY")}) + "\n")
print("[OK] Release integrity: all files match SHA256SUMS.txt", flush=True)
print("[READY] SYSTEM READY: NERO V4.11 fake", flush=True)
time.sleep(1.0)
with open(os.path.join(td, "ready.json"), "w") as f:
    json.dump({"t": time.time()}, f)
n = 0
while True:
    for nm in sorted(os.listdir(os.path.join(td, "inbox"))):
        if not nm.endswith(".json"):
            continue
        with open(os.path.join(td, "inbox", nm)) as f:
            m = json.load(f)
        os.remove(os.path.join(td, "inbox", nm))
        time.sleep(0.5)
        with open(os.path.join(td, "outbox", m["id"] + ".json"), "w") as f:
            json.dump({"id": m["id"], "status": "complete", "llm": m.get("llm"),
                       "stats": {"tools": [{"tool": "perceive_objects", "ok": True, "safety_stops": 0}]}}, f)
    if os.path.exists(ef):
        with open(ef) as f:
            req = json.load(f)
        os.remove(ef)
        n += 1
        with open(ef + ".done", "w") as f:
            json.dump({"id": req["id"], "status": "success" if n % 2 else "fail", "n_ok": n % 2, "n_goals": 1,
                       "objects": {}}, f)
    time.sleep(0.2)
''')
fake = os.path.join(tmp, "fake_sim.sh")
with open(fake, "w") as f:
    f.write("#!/usr/bin/env bash\nexec env -u PYTHONHOME -u PYTHONPATH -u LD_LIBRARY_PATH /usr/bin/python3 "
            f"{fake_py}\n")
os.environ["NERO_V312_SIM_SCRIPT"] = fake
import NERO_V312_Console as C  # noqa: E402
import NERO_V411_Sim_DOE as D  # noqa: E402

messagebox.askyesno = lambda *a, **k: True
messagebox.showerror = lambda *a, **k: None
_orig_init = D.DOERunner.__init__


def _fast_init(self, *a, **k):
    k["timeouts"] = {"settle_s": 1.0, "after_stop_s": 1.0}
    _orig_init(self, *a, **k)


D.DOERunner.__init__ = _fast_init
D.DOETab.show_text = lambda self, title, text: setattr(self, "_shown", title)
result = {}
_orig_mainloop = tk.Tk.mainloop


def walk(w):
    yield w
    for c in w.winfo_children():
        yield from walk(c)


def driver(root):
    nb = [w for w in walk(root) if w.winfo_class() == "TNotebook"][0]
    tabs = [nb.tab(t, "text") for t in nb.tabs()]
    result["tabs"] = tabs
    tab = None
    for t in nb.tabs():
        if nb.tab(t, "text") == "仿真 DOE":
            nb.select(t)
            tab = DOE_TABS[0] if DOE_TABS else None
    result["doe_found"] = tab is not None
    if tab is None:
        root.destroy()
        return
    tab.rows = D.default_rows(["fixed:H1", "fixed:H2"])
    tab.models = [D.parse_model("openai:gpt-6.1-sol@high")]
    tab.v_rep.set("2")
    tab._refresh_rows()
    tab._refresh_models()
    for prov, k in (("openai", "sk-doe-test"),):
        tab.api["key_env"].__self__.keys[prov] = k
        tab.api["key_env"].__self__.v_key.set(k)
    root.update()
    tab.start()
    t0 = time.time()

    def wait():
        if tab.runner and tab.runner.finished or time.time() - t0 > 150:
            result["status"] = [r["status"] for r in tab.runner.runs] if tab.runner else None
            result["running_after"] = C_RUNNER[0].running() if C_RUNNER else None
            shot = os.environ.get("NERO_V411_SHOT")
            if shot:
                root.update()
                subprocess.run(["import", "-window", "root", shot], check=False)
            root.destroy()
        else:
            root.after(200, wait)
    root.after(200, wait)


DOE_TABS = []
_orig_tab_init = D.DOETab.__init__


def _tab_init(self, *a, **k):
    _orig_tab_init(self, *a, **k)
    DOE_TABS.append(self)


D.DOETab.__init__ = _tab_init
C_RUNNER = []
_orig_runner_init = C.SimRunner.__init__


def _runner_init(self, *a, **k):
    _orig_runner_init(self, *a, **k)
    C_RUNNER.append(self)


C.SimRunner.__init__ = _runner_init


def patched_mainloop(self, n=0):
    self.after(800, lambda: driver(self))
    _orig_mainloop(self, n)


tk.Tk.mainloop = patched_mainloop
C.run_console("sim")
print({k: v for k, v in result.items() if k != "tabs"})
tabs = result["tabs"]
assert tabs[tabs.index("仿真 (MuJoCo)") + 1] == "仿真 DOE", tabs
assert result["doe_found"]
assert result["status"] == ["success", "fail", "success", "fail"], result
assert result["running_after"] is False
with open(os.path.join(os.environ["NERO_V410_TASK_DIR"], "launches.jsonl")) as f:
    launches = [json.loads(x) for x in f if x.strip()]
assert [(x["scene"], x["household"]) for x in launches] == [("household_pick_place", "fixed:H1"),
                                                            ("household_pick_place", "fixed:H2")], launches
assert all(x["key"] == "sk-doe-test" for x in launches), launches
logs = open(os.path.join(HERE, "logs", "console.log"), encoding="utf-8", errors="replace").read() \
    if os.path.exists(os.path.join(HERE, "logs", "console.log")) else ""
assert "sk-doe-test" not in logs
print("[GUI] console DOE: tab after 仿真 (MuJoCo); one start per scene through the console's start path, key in the "
      "launcher environment only; 4 runs judged; simulation stopped at the end")
