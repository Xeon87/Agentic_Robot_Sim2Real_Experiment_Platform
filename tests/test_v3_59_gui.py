#!/usr/bin/env python3
"""V3.59 GUI test for the 「策略旋钮」 panel (needs tkinter + DISPLAY; skips otherwise).
G1 StrategyPanelUI builds all 16 knob rows in a real Tk window (no tkinter.font objects).
G2 typing a value + <Return> clamps, updates the profile, calls on_change and refreshes the preview text.
G3 save -> reset -> load in a temp NERO_STRATEGY_DIR (never config/strategies).
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import tkinter as tk
    from tkinter import ttk
    if not os.environ.get("DISPLAY"):
        print("[SKIP] test_v3_59_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v3_59_gui: {exc}")
    sys.exit(0)

TMP = tempfile.mkdtemp(prefix="nero_v359_gui_")
os.environ["NERO_STRATEGY_DIR"] = os.path.join(TMP, "strategies")
os.environ["NERO_V359_ACTIVE_PATH"] = os.path.join(TMP, "active.json")
import NERO_Strategy_Profile as SP  # noqa: E402

root = tk.Tk()
root.withdraw()
frm = ttk.Frame(root)
frm.pack(fill="both", expand=True)
logs, changes = [], []
panel = SP.StrategyPanelUI(tk, ttk, frm, log=logs.append, on_change=lambda p: changes.append(dict(p.values)),
                           ask=lambda *_a: True)
root.update()
assert all(k.name in panel.vars for k in SP.KNOBS)
print("[G1] 16 knob rows built")

panel.vars["hover_height_mm"].set("120")
panel._on_var("hover_height_mm")
root.update()
assert panel.profile["hover_height_mm"] == 90.0 and panel.vars["hover_height_mm"].get() == "90"
assert changes and changes[-1]["hover_height_mm"] == 90.0
assert "悬停 90 mm" in panel.txt.get("1.0", "end")
print("[G2] value clamped to 90 mm, on_change called, preview refreshed")

panel.v_name.set("gui_test")
out = panel.on_save()
assert out and os.path.isfile(out) and out.startswith(TMP)
panel.on_reset()
assert panel.profile["hover_height_mm"] == 60.0
panel.v_preset.set("gui_test")
assert panel.on_load() and panel.profile["hover_height_mm"] == 90.0
root.update()
print("[G3] save / reset / load round trip")
root.destroy()
print("ALL V3.59 GUI TESTS PASSED")
