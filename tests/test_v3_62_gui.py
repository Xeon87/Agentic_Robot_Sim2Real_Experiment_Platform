#!/usr/bin/env python3
"""V3.62 GUI test (needs tkinter + DISPLAY; skips otherwise):
Q1 Real Tk, console mode: the 「技能参数旋钮」 tab group ⑥ shows 11 cup knobs, each tagged [仿真技能✓]; the bool knob
   「沿杯壁倾斜下探」 is a check box; typed values round-trip (clamped) through collect_profile_from_ui.
Q2 Simulation GUI (add_sim_strategy_tab): the same 11 cup knobs; selecting CS shows [CS] in the banner.
Q3 Nothing inside the release folder changed.
"""
import hashlib
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import tkinter as tk
    from tkinter import ttk
    import numpy  # noqa: F401
    if not os.environ.get("DISPLAY"):
        print("[SKIP] test_v3_62_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v3_62_gui: {exc}")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="nero_v362_gui_"))
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
import NERO_V359_Strategy as ST   # noqa: E402


def labels(w):
    out = []
    for c in w.winfo_children():
        if c.winfo_class() in ("TLabel", "Label"):
            out.append(str(c.cget("text")))
        out += labels(c)
    return out


root = tk.Tk()
root.geometry("1400x920")
f = ttk.Frame(root)
f.pack(fill="both", expand=True)
tab = ST.SkillStrategyTabUI(tk, ttk, f, log=lambda *a: None)
root.update()
cup_box = tab._sections["cup_stacking"]["body"]
tags = [t for t in labels(cup_box) if "(默认:" in t]
assert len(tags) == 11 and all("[仿真技能✓]" in t for t in tags), tags
assert sum("[仿真技能✓]" in t for t in labels(f)) == 11
assert isinstance(tab.knob_vars["cup_wall_tilt_follow"], tk.BooleanVar)
tab.knob_vars["cup_rim_pinch_depth_mm"].set("80")
tab.knob_vars["cup_clamp_force_n"].set("3")
tab.knob_vars["cup_wall_tilt_follow"].set(False)
p = tab.collect_profile_from_ui()
assert p.knobs["cup_rim_pinch_depth_mm"] == 50.0 and p.knobs["cup_clamp_force_n"] == 3.0
assert p.knobs["cup_wall_tilt_follow"] is False
print("[Q1] console tab: group 6 = 11 cup knobs [仿真技能✓], bool check box, typed values clamped")

nb = ttk.Notebook(root)
nb.pack()
ui = ST.add_sim_strategy_tab(tk, ttk, nb, log=lambda *a: None)
root.update()
assert len([k for k in ui.knob_vars if k.startswith("cup_")]) == 11
lab_cs = next(lab for lab, (sid, _f, _p) in ui._preset_map.items() if sid == "cup_stack_rim_pinch")
ui.v_preset.set(lab_cs)
ui.on_select_preset()
root.update()
assert ui.profile.code == "CS" and "[CS]" in ui.v_status.get()
assert ui.knob_vars["cup_rim_pinch_depth_mm"].get() == "35"
print("[Q2] sim GUI: 11 cup knobs, CS selected -> [CS], depth 35 mm")
root.destroy()
snap1 = snapshot()
assert snap1 == SNAP0, [k for k in SNAP0 if SNAP0[k] != snap1.get(k)]
print("[Q3] release folder unchanged")
print("ALL V3.62 GUI TESTS PASSED")
