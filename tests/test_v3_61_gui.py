#!/usr/bin/env python3
"""V3.61 GUI test (needs tkinter + DISPLAY; skips otherwise):
N1 Real Tk, real mode: the 「技能参数旋钮」 tab has 8 collapsible sections (skill card, dry-run, 6 knob groups); each
   one folds / unfolds on its own (body unmapped / mapped), 「⊟ 全部折叠」 / 「⊞ 全部展开」 work; values typed in a folded
   group are kept.
N2 The preset drop-down shows every skill as 'name [XX] · skill_id'; selecting CS shows [CS] in the banner.
N3 Simulation GUI (add_sim_strategy_tab): same sections and buttons, dry-run section hidden, fold/unfold work.
N4 The console's simulation scene drop-down offers stack_cups.
N5 Nothing inside the release folder changed.
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
        print("[SKIP] test_v3_61_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v3_61_gui: {exc}")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="nero_v361_gui_"))
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

root = tk.Tk()
root.geometry("1400x920")
f_real = ttk.Frame(root)
f_real.pack(fill="both", expand=True)
tab = ST.SkillStrategyTabUI(tk, ttk, f_real, log=lambda *a: None)
root.update()
GROUPS = [g for g, _ in ST.KNOB_GROUPS]


def mapped(sec):
    return sec["body"].winfo_manager() == "pack"


assert list(tab._sections) == ["skill_card", "dry_run"] + GROUPS
assert all(mapped(s) for s in tab._sections.values())
tab.knob_vars["slot_xy_tol_mm"].set("6.5")
tab._sections["placement_recovery"]["btn"].invoke()
root.update()
assert not mapped(tab._sections["placement_recovery"]) and mapped(tab._sections["ik_posture"])
assert tab._sections["placement_recovery"]["btn"].cget("text").startswith("▸ ")
tab.b_collapse_all.invoke()
root.update()
assert not any(mapped(s) for s in tab._sections.values())
tab.b_expand_all.invoke()
root.update()
assert all(mapped(s) for s in tab._sections.values()) and tab.knob_vars["slot_xy_tol_mm"].get() == "6.5"
assert tab.collect_profile_from_ui().knobs["slot_xy_tol_mm"] == 6.5
print(f"[N1] real mode: {len(tab._sections)} sections fold/unfold individually + 全部折叠/全部展开; typed value kept")

vals = list(tab.cb_preset.cget("values")) if hasattr(tab, "cb_preset") else list(tab._preset_map)
for p in ST.builtin_strategy_presets():
    assert any(v.startswith(f"{ST.skill_display_name(p)} · {p.skill_id}") for v in vals), (p.code, vals)
lab_cs = next(lab for lab, (sid, _f, _p) in tab._preset_map.items() if sid == "cup_stack_rim_pinch")
tab.v_preset.set(lab_cs)
tab.on_select_preset()
root.update()
assert tab.profile.code == "CS" and "[CS]" in tab.v_status.get()
print("[N2] drop-down labels 'name [XX] · id'; CS selected -> banner shows [CS]")

nb = ttk.Notebook(root)
nb.pack()
ui = ST.add_sim_strategy_tab(tk, ttk, nb, log=lambda *a: None)
root.update()
assert list(ui._sections) == list(tab._sections) and ui._sections["dry_run"]["box"].winfo_manager() == ""
assert ui.b_collapse_all.winfo_class() == "TButton" and ui.b_expand_all.winfo_class() == "TButton"
ui._sections["cup_stacking"]["btn"].invoke()
root.update()
assert not mapped(ui._sections["cup_stacking"])
ui.b_collapse_all.invoke()
ui.b_expand_all.invoke()
root.update()
assert all(mapped(ui._sections[k]) for k in ["skill_card"] + GROUPS)
print("[N3] sim GUI: same sections/buttons, dry-run hidden, fold/unfold OK")
root.destroy()

import NERO_V312_Console as CON   # noqa: E402
assert ("stack_cups", "stack_cups - 叠杯子 (RoboDojo)") in CON.SIM_SCENES
print("[N4] console scene list offers stack_cups")
snap1 = snapshot()
assert snap1 == SNAP0, [k for k in SNAP0 if SNAP0[k] != snap1.get(k)]
print("[N5] release folder unchanged")
print("ALL V3.61 GUI TESTS PASSED")
