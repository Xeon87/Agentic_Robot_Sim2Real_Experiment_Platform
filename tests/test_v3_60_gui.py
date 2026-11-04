#!/usr/bin/env python3
"""V3.60 GUI test (needs tkinter + DISPLAY; skips otherwise):
H1 Real Tk: CameraTabUI + 「技能参数旋钮」 (real mode) built; start-up keeps the user's hover value; the dry-run panel is
   packed in real mode; 6 knobs carry the [仿真✓] tag.
H2 Selecting QH writes 60 mm into the hover field, wrist USB functions keep the user's 85 mm; reset -> EB 100 mm.
H3 Dry-run planned from the UI with real IK uses the hover field (60 mm with QH, 100 mm with EB); executing it moves
   the (fake) arm through hover A -> transit -> hover B -> return home on the worker thread.
H4 Simulation GUI: add_sim_strategy_tab adds its own notebook tab; the dry-run panel is not packed in sim mode.
H5 The whole console (run_console('real')) builds with real Tk and has the 「技能参数旋钮」 tab.
H6 Nothing inside the release folder changed.
"""
import hashlib
import math
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import tkinter as tk
    from tkinter import ttk
    import numpy  # noqa: F401
    if not os.environ.get("DISPLAY"):
        print("[SKIP] test_v3_60_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v3_60_gui: {exc}")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="nero_v360_gui_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")


def snapshot():
    out = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rel = line.split(None, 1)[1].strip()
            out[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    out["__config__"] = sorted(str(p) for p in (ROOT / "config").rglob("*"))
    return out


SNAP0 = snapshot()
import NERO_V315_Consistency_Safety as CS     # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V359_Strategy as ST               # noqa: E402

CAM.CALIB_PATH = str(TMP / "camera_calib.json")
CAM.LOCATE_PATH = str(TMP / "locate_settings.json")
CAM.CALIB_DIR = str(TMP / "cameracalib")
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
LIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708)]
Q0_DEG = [0.0, 30.0, 0.0, 80.0, 0.0, 40.0, 0.0]


class Pipe:
    def __init__(self):
        self.q = list(Q0_DEG)
        self.teach_mode = False
        self.real_urdf = U
        self.moves = []

    def actual_arm_deg(self, model):
        return list(self.q)

    def move_and_settle(self, model, target_rad, speed, label="", **kw):
        self.q = [math.degrees(v) for v in target_rad]
        self.moves.append((label, speed))
        return {"ok": True}

    def fence_goal_check(self, model, q):
        return True, "", []

    def cancel_move(self):
        pass


class Safe:
    sim = U
    dirty = False
    valid_ws = {"fence": {"polygon": [[-0.85, -0.45], [-0.10, -0.45], [-0.10, 0.45], [-0.85, 0.45]], "z_table": 0.0}}

    def arm_points(self, q):
        return []


root = tk.Tk()
root.geometry("1400x920")
nb = ttk.Notebook(root)
nb.pack(fill="both", expand=True)
cam_frame, strat_frame = ttk.Frame(nb), ttk.Frame(nb)
nb.add(cam_frame, text="Camera调试")
nb.add(strat_frame, text="技能参数旋钮")
logs = []
pipe = Pipe()
jp = type("JP", (), {"enabled": True, "model": {"joints": J, "limits": [list(l) for l in LIM], "base_link": "base_link"}})()
cam = CAM.CameraTabUI(tk, ttk, cam_frame, pipe, jp, Safe(), lambda: {"speed_percent": 10}, logs.append, lambda *a: True,
                      visible_fn=lambda: True)
cam.v_hover_mm.set("85")
tab = ST.SkillStrategyTabUI(tk, ttk, strat_frame, cam_tab=cam, pipe=pipe, jpanel=jp, log=logs.append, ask=lambda *a: True)
root.update()


def tick(n=3):
    for _ in range(n):
        try:
            cam.poll()
        except Exception:
            pass
        root.update()


assert tab.profile.code == "EB" and cam.v_hover_mm.get() == "85" and not Path(os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"]).exists()
assert tab.b_dry_check.master.master.winfo_manager() == "pack"


def all_labels(w):
    out = []
    for c in w.winfo_children():
        if c.winfo_class() in ("TLabel", "Label"):
            out.append(str(c.cget("text")))
        out += all_labels(c)
    return out


labels = all_labels(strat_frame)
tags = [t for t in labels if "(默认:" in t]
assert len(tags) == len(ST.KNOB_SPECS) and sum("[仿真✓]" in t for t in tags) == 6 and sum("[实机✓]" in t for t in tags) == 8 \
    and sum("[未接入]" in t for t in tags) == 31, tags[:5]                      # V3.61: + 9 cup knobs [离线试验✓]
print("[H1] real Tk: start-up keeps hover 85 mm, dry-run panel packed, 6 [仿真✓] + 8 [实机✓] tags")

lab_qh = next(lab for lab, (sid, _f, _p) in tab._preset_map.items() if sid == "quick_hover_verify")
tab.v_preset.set(lab_qh)
tab.on_select_preset()
tick()
assert cam.v_hover_mm.get() == "60" and cam._hover_user_str() == "85" and cam._locate_settings()["hover_mm"] == "85"
print("[H2] QH: hover field 60 mm, wrist USB / saved settings keep 85 mm")

cam.eggs = [{"det_id": "egg_a", "xyz_m": [-0.45, -0.05, 0.03], "trusted": True, "confidence": 0.9, "status": "ok"},
            {"det_id": "egg_b", "xyz_m": [-0.40, 0.08, 0.03], "trusted": True, "confidence": 0.8, "status": "ok"}]
plan = tab.on_dry_run_plan_only()
assert plan and plan["ok"], plan
assert plan["hover_height_mm"] == 60.0 and [s["kind"] for s in plan["steps"]][-1] == "return_home"
tab.on_reset_baseline()
tick()
assert tab.profile.code == "EB" and cam.v_hover_mm.get() == "100" and cam._hover_user_str() == "85"
plan = tab.on_dry_run_plan_only()
assert plan["ok"] and plan["hover_height_mm"] == 100.0, plan
hz = [s["tcp_xyz_m"][2] for s in plan["steps"] if s["kind"] == "hover"]
assert all(abs(z - 0.13) < 1e-9 for z in hz), hz
n0 = len(pipe.moves)
tab.on_dry_run_execute()
t_end = time.time() + 60
while time.time() < t_end and (cam.worker is not None and cam.worker.is_alive()):
    tick(1)
    time.sleep(0.05)
tick(3)
kinds = [m[0] for m in pipe.moves[n0:]]
assert len(kinds) == len(plan["steps"]) and kinds[0] == "Dry-Run egg_a" and kinds[-1] == "Dry-Run 回起点姿态", kinds
assert all(abs(m[1] - 0.10) < 1e-9 for m in pipe.moves[n0:])
assert any("多点悬停预演完成" in str(l) for l in logs)
print(f"[H3] dry-run planned with real IK (QH 60 mm, EB 100 mm) and executed: {len(kinds)} moves at 10 %")

sim_nb = ttk.Notebook(root)
sim_nb.pack()
n_tabs = len(sim_nb.tabs())
ui = ST.add_sim_strategy_tab(tk, ttk, sim_nb, log=logs.append)
root.update()
assert len(sim_nb.tabs()) == n_tabs + 1 and sim_nb.tab(sim_nb.tabs()[-1], "text") == "技能参数旋钮"
assert ui.mode == "sim" and (ui.b_dry_check.master.master.winfo_manager() == ""
                             or ui._sections["dry_run"]["box"].winfo_manager() == "")   # V3.61 collapsible section
print("[H4] sim GUI tab added; dry-run panel hidden in sim mode")

cam.shutdown()
root.destroy()

import NERO_V312_Console as CON   # noqa: E402
built = {}
_orig = tk.Tk.mainloop


def fake_mainloop(self, n=0):
    for _ in range(5):
        self.update()
    nbs = [w for w in self.winfo_children() if w.winfo_class() == "TNotebook"]
    built["tabs"] = [nbs[0].tab(t, "text") for t in nbs[0].tabs()] if nbs else []
    self.destroy()


tk.Tk.mainloop = fake_mainloop
try:
    CON.run_console("real")
finally:
    tk.Tk.mainloop = _orig
assert "技能参数旋钮" in built.get("tabs", []) and "Camera调试" in built["tabs"], built
print(f"[H5] run_console('real') with real Tk: tabs {built['tabs']}")

snap1 = snapshot()
assert snap1 == SNAP0, [k for k in SNAP0 if SNAP0[k] != snap1.get(k)]
print("[H6] release folder unchanged")
print("ALL V3.60 GUI TESTS PASSED")
