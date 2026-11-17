#!/usr/bin/env python3
"""NERO V4.10 SIM BRIDGE - console -> simulation GUI, without touching the backend (its file is hash-pinned by the
V3.6x-V3.8x tests): attached from NERO_V359_Strategy.add_sim_strategy_tab(), which the backend already calls while it
builds the GUI; the SimAgent widgets are found in the Tk widget tree (find_widgets) and driven through their Tk
variables / buttons, exactly as a person would.

1. LLM settings from the console: when the simulation was started by the console (NERO_V410_LLM_FROM_CONSOLE=1)
   the NERO SimAgent 'LLM planner / reasoning / API' panel is filled from the environment (provider, model, effort,
   max tool steps, dry run; the API key stays in the provider's environment variable, which the agent already uses
   when the key field is blank) and all its widgets are disabled, with a note that the console owns them.
2. Task file channel (for the next version's Sim DOE): <task dir>/inbox/<id>.json = {"id", "task"} -> the task is
   typed into the task field and RUN TASK is pressed; when the run ends
   <task dir>/outbox/<id>.json = {"id", "status": "complete" | "failed" | "not_started", "status_text", "t_start",
   "t_end"}.  task dir = NERO_V410_TASK_DIR or <release>/logs/sim_tasks.  One task at a time, in file-name order.
"""
from __future__ import annotations

import json
import os
import time

VERSION = "V4.10"
MARKER = "V4.10 SIM BRIDGE"
HERE = os.path.dirname(os.path.realpath(__file__))


def task_dir() -> str:
    return os.environ.get("NERO_V410_TASK_DIR") or os.path.join(HERE, "logs", "sim_tasks")


def console_settings(env=None) -> dict:
    """the console's LLM settings from the environment (pure logic, tested); {} = not started by the console"""
    e = os.environ if env is None else env
    if e.get("NERO_V410_LLM_FROM_CONSOLE") != "1":
        return {}
    try:
        steps = int(e.get("NERO_V342_DEFAULT_STEPS", "60"))
    except ValueError:
        steps = 60
    return {"provider": e.get("NERO_V410_LLM_PROVIDER", "openai"), "model": e.get("NERO_LLM_MODEL", ""),
            "effort": e.get("NERO_V410_LLM_EFFORT", "high"), "steps": steps,
            "dry_run": e.get("NERO_V410_LLM_DRY_RUN", "0") == "1"}


def apply_console_settings(g, log):
    """fill + lock the SimAgent LLM panel; g = getter dict (getters_from_widgets)"""
    cs = console_settings()
    if not cs:
        return False
    v = g["vars"]()
    labels = g["labels"]()
    lab = labels.get(cs["provider"])
    if lab:
        v["provider"].set(lab)
        try:
            g["on_provider"]()()
        except Exception:
            pass
    if cs["model"]:
        v["model"].set(cs["model"])
    v["effort"].set(cs["effort"])
    v["steps"].set(str(cs["steps"]))
    v["dry"].set(bool(cs["dry_run"]))
    v["key"].set("")                                   # blank = the provider's environment variable (from the console)
    frame = g["cfg_frame"]()
    for w in frame.winfo_children():
        try:
            w.configure(state="disabled")
        except Exception:
            pass
    try:
        import tkinter.ttk as ttk
        ttk.Label(frame, text="以上 LLM 设置由 NERO 控制台「仿真 (MuJoCo)」页传入（V4.10），这里只读；"
                              "要修改请停止仿真，在控制台里改后重新启动。", foreground="#a15c00").grid(
            row=3, column=0, columnspan=8, padx=8, pady=(0, 6), sticky="w")
    except Exception:
        pass
    log(f"[V4.10 LLM] settings from the console: provider={cs['provider']} model={cs['model']} effort={cs['effort']} "
        f"steps={cs['steps']} dry_run={cs['dry_run']} (API key from the environment, value hidden); panel read-only")
    return True


class TaskChannel:
    """file inbox -> RUN TASK -> outbox (polled from the Tk thread with root.after)"""

    def __init__(self, root, g, log, period_ms=1000):
        self.root, self.g, self.log, self.period = root, g, log, int(period_ms)
        self.dir = task_dir()
        self.cur = None
        for sub in ("inbox", "outbox", "done"):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)

    def start(self):
        self.root.after(self.period, self._tick)

    def _busy(self):
        try:
            ag = self.g["agent"]()
            return bool(ag is not None and ag.running)
        except Exception:
            return False

    def _tick(self):
        try:
            self._step()
        except Exception as exc:
            self.log(f"[WARN] [V4.10 TASK CHANNEL] {exc}")
        self.root.after(self.period, self._tick)

    def _step(self):
        st = self.g["status"]()
        if self.cur is not None:
            txt = str(st.get("text", ""))
            started = self.cur.get("seen_busy") or self._busy()
            self.cur["seen_busy"] = started
            if started and not self._busy() and ("complete" in txt or "FAILED" in txt):
                out = {"id": self.cur["id"], "status": "failed" if "FAILED" in txt else "complete",
                       "status_text": txt, "t_start": self.cur["t_start"], "t_end": time.time()}
                with open(os.path.join(self.dir, "outbox", f"{self.cur['id']}.json"), "w") as f:
                    json.dump(out, f)
                self.log(f"[V4.10 TASK CHANNEL] task {self.cur['id']} ended: {out['status']}")
                self.cur = None
            elif time.time() - self.cur["t_start"] > 15 and not started:
                out = {"id": self.cur["id"], "status": "not_started", "status_text": txt,
                       "t_start": self.cur["t_start"], "t_end": time.time()}
                with open(os.path.join(self.dir, "outbox", f"{self.cur['id']}.json"), "w") as f:
                    json.dump(out, f)
                self.log(f"[WARN] [V4.10 TASK CHANNEL] task {self.cur['id']} did not start: {txt}")
                self.cur = None
            return
        if self._busy():
            return
        inbox = os.path.join(self.dir, "inbox")
        names = sorted(n for n in os.listdir(inbox) if n.endswith(".json"))
        if not names:
            return
        p = os.path.join(inbox, names[0])
        try:
            with open(p) as fh:
                msg = json.load(fh)
        except Exception:
            return                                      # half-written: next tick
        os.replace(p, os.path.join(self.dir, "done", names[0]))
        tid = str(msg.get("id") or names[0][:-5])
        task = str(msg.get("task") or "").strip()
        if not task:
            self.log(f"[WARN] [V4.10 TASK CHANNEL] {tid}: empty task ignored")
            return
        self.g["vars"]()["task"].set(task)
        self.cur = {"id": tid, "t_start": time.time(), "seen_busy": False}
        self.log(f"[V4.10 TASK CHANNEL] task {tid}: {task!r} -> RUN TASK")
        self.g["run"]()()


CFG_TITLE = "NERO V2-7 LLM planner / reasoning / API"
TASK_TITLE = "Natural-language task"
GRID = {"model": (0, 1), "effort": (0, 4), "steps": (0, 6), "dry": (0, 7), "key": (1, 2), "provider": (2, 1)}


def _walk(w):
    yield w
    for c in w.winfo_children():
        yield from _walk(c)


def _text(w):
    try:
        return str(w.cget("text"))
    except Exception:
        return ""


class _TkVar:
    """a Tk variable by its Tcl name (the backend keeps its StringVar objects in run_gui's locals)"""

    def __init__(self, root, name):
        self.root, self.name = root, str(name)

    def get(self):
        return self.root.getvar(self.name)

    def set(self, v):
        self.root.setvar(self.name, v)


def find_widgets(root) -> dict:
    """the SimAgent widgets by structure (LabelFrame titles, grid cells, button text, status variable)"""
    found = {}
    for w in _walk(root):
        cls = w.winfo_class()
        if cls == "TLabelframe" and _text(w) == CFG_TITLE:
            found["cfg_frame"] = w
            for c in w.winfo_children():
                gi = c.grid_info() if c.winfo_manager() == "grid" else {}
                rc = (int(gi.get("row", -1)), int(gi.get("column", -1))) if gi else None
                for k, pos in GRID.items():
                    if rc == pos and c.winfo_class() in ("TCombobox", "TSpinbox", "TCheckbutton", "TEntry"):
                        found[k] = c
        elif cls == "TLabelframe" and _text(w) == TASK_TITLE:
            for c in _walk(w):
                if c.winfo_class() == "TEntry":
                    found.setdefault("task", c)
        elif cls == "TButton" and _text(w) == "RUN TASK":
            found["run"] = w
        elif cls == "TLabel":
            try:
                tv = str(w.cget("textvariable"))
            except Exception:
                tv = ""
            if tv and str(root.getvar(tv)).startswith(("Agent:", "LLM agent")):
                found["status"] = w
    return found


def getters_from_widgets(root, W) -> dict:
    """the getter dict apply_console_settings / TaskChannel use, built on the found widgets"""
    def var(key):
        w = W[key]
        name = w.cget("variable") if key == "dry" else w.cget("textvariable")
        return _TkVar(root, name)

    def status_text():
        try:
            return str(root.getvar(str(W["status"].cget("textvariable"))))
        except Exception:
            return ""

    class _Agent:
        @property
        def running(self):
            return status_text().startswith("Agent: BUSY")

    labels = {}
    try:
        import NERO_V2_7_1_Agent as A
        labels = {k: v.get("label", k) for k, v in A.LLM_PROVIDERS.items()}
    except Exception:
        pass
    return {"vars": lambda: {k: var(k) for k in ("provider", "model", "effort", "steps", "dry", "key", "task")},
            "labels": lambda: labels,
            "on_provider": lambda: (lambda: W["provider"].event_generate("<<ComboboxSelected>>")),
            "cfg_frame": lambda: W["cfg_frame"], "run": lambda: W["run"].invoke,
            "status": lambda: {"text": status_text()}, "agent": lambda: _Agent()}


def attach_from_notebook(notebook, log):
    """called from NERO_V359_Strategy.add_sim_strategy_tab (simulation GUI only)"""
    root = notebook.winfo_toplevel()

    def later(tries=[0]):
        W = find_widgets(root)
        need = ("cfg_frame", "model", "effort", "steps", "dry", "key", "provider", "task", "run", "status")
        missing = [k for k in need if k not in W]
        if missing and tries[0] < 10:
            tries[0] += 1
            root.after(1000, later)
            return
        if missing:
            log(f"[WARN] [V4.10 SIM BRIDGE] SimAgent widgets not found: {missing}; console settings / task channel off")
            return
        attach_gui(root, getters_from_widgets(root, W), log, delay_ms=0)
    root.after(1500, later)


def attach_gui(root, g, log, delay_ms=1500):
    """apply the console settings and start the task channel (after the GUI exists)"""
    def later():
        try:
            apply_console_settings(g, log)
        except Exception as exc:
            log(f"[WARN] [V4.10 LLM] console settings not applied: {exc}")
        try:
            TaskChannel(root, g, log).start()
            log(f"[V4.10 TASK CHANNEL] watching {os.path.join(task_dir(), 'inbox')}")
        except Exception as exc:
            log(f"[WARN] [V4.10 TASK CHANNEL] not started: {exc}")
    root.after(int(delay_ms), later)
