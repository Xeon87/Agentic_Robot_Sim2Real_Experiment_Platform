#!/usr/bin/env python3
"""V3.59 GUI test (written in V3.60: the V3.59 release referenced it but did not ship it; needs tkinter + DISPLAY):
K1 Camera tab: the V3.59 「▶ 多点悬停预演」 row exists (button, route combobox, manual target entry).
K2 「技能参数旋钮」 tab: 45 knob widgets, skill-card fields, dry-run route modes; values typed in the UI round-trip
   through collect_profile_from_ui (clamped) and the automatic summary names the changed knob.
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
    import numpy  # noqa: F401
    if not os.environ.get("DISPLAY"):
        print("[SKIP] test_v3_59_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v3_59_gui: {exc}")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="nero_v359_gui_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
import NERO_V325_Camera as CAM     # noqa: E402
import NERO_V359_Strategy as ST    # noqa: E402

CAM.CALIB_PATH = str(TMP / "camera_calib.json")
CAM.LOCATE_PATH = str(TMP / "locate_settings.json")
CAM.CALIB_DIR = str(TMP / "cameracalib")


class Pipe:
    teach_mode = False
    real_urdf = None

    def actual_arm_deg(self, model):
        return None


root = tk.Tk()
f1, f2 = ttk.Frame(root), ttk.Frame(root)
f1.pack()
f2.pack()
jp = type("JP", (), {"enabled": False, "model": {"joints": [f"joint{i}" for i in range(1, 8)], "limits": [[-3, 3]] * 7}})()
safe = type("S", (), {"sim": None, "dirty": False, "valid_ws": None, "arm_points": lambda self, q: []})()
cam = CAM.CameraTabUI(tk, ttk, f1, Pipe(), jp, safe, lambda: {}, lambda *a: None, lambda *a: True, visible_fn=lambda: True)
root.update()
assert cam.b_dry_run.winfo_class() == "TButton" and cam.cb_dry_dest.cget("values")
assert cam.v_dry_dest_xyz.get() == "-450, 123, 43"
print("[K1] Camera tab dry-run row present")

tab = ST.SkillStrategyTabUI(tk, ttk, f2, cam_tab=cam, log=lambda *a: None)
root.update()
assert len(tab.knob_vars) == 45 and len(tab.dry_modes) == 4
tab.knob_vars["egg_open_margin_mm"].set("99")
tab.knob_vars["slot_fill_order"].set("far_to_near")
p = tab.collect_profile_from_ui()
assert p.knobs["egg_open_margin_mm"] == 26.0 and p.knobs["slot_fill_order"] == "far_to_near"
tab.on_auto_summary()
assert "鸡蛋预张开总裕量" in tab.v_behavior.get()
print("[K2] skill tab: 45 knobs, typed values clamped, automatic summary")
cam.shutdown()
root.destroy()
print("ALL V3.59 GUI TESTS PASSED")
