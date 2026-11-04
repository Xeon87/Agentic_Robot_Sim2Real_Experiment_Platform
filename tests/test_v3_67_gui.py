#!/usr/bin/env python3
"""V3.67 GUI test (needs tkinter + DISPLAY; skips otherwise):
T1 Real Tk: the 「NERO实机任务」 tab shows the new 「实机夹爪几何与下探安全」 panel (clearance 8 mm, align 60 mm, stop
   10 mm, seat / ring / roll) and its description line (align -> rotate in place -> down, fixed jaw 8 mm).
T2 「空夹标定」 on a simulated STS3215 whose tips touch at +1.85 deg: the touch angle is found, shown in the gripper
   readout, and a following 「夹紧测试」 with nothing in the jaws is NOT held; worker threads never touch Tk.
T3 Changing a field updates the description after 「重新读取技能」; settings (geometry + touch point) go to
   NERO_CALIB_DIR/real_cup_task.json; nothing inside the release folder changed.
"""
import hashlib
import json
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
    import numpy as np
    if not os.environ.get("DISPLAY"):
        print("[SKIP] test_v3_67_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v3_67_gui: {exc}")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="nero_v367_gui_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
os.environ["NERO_V368_TASK_DIR"] = str(TMP / "task")          # V3.68: task records never in the release folder
H = lambda b: hashlib.sha256(b).hexdigest()
SNAP0 = {str(p.relative_to(ROOT)): H(p.read_bytes()) for p in ROOT.rglob("*")
         if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and "logs" not in p.parts}

import NERO_V365_Real_Cup_Task as RT         # noqa: E402
import NERO_V365_STS3215_Gripper as GR       # noqa: E402


class Cam:
    src = wrist_src = result = calib = worker = None


class Pipe:
    teach_mode = False

    def lamp(self, k):
        return "grey"

    def supports_joint_path(self):
        return False

    def actual_arm_deg(self, m):
        return None

    def move_busy(self):
        return False

    def cancel_move(self):
        pass


class JP:
    model = None


root = tk.Tk()
root.geometry("1500x1000")
nb = ttk.Notebook(root)
nb.pack(fill="both", expand=True)
fr = ttk.Frame(nb, padding=6)
nb.add(fr, text="NERO实机任务")
sim = GR.SimServo()
sim.obstacle_gap_mm = GR.gap_mm_for_open_deg(1.85)
grip = GR.SO101RealGripper(log=lambda s: None)
_orig = grip.connect
grip.connect = lambda port=None, baud=None, servo_id=None, sim_servo_=None: _orig(port, baud, servo_id, sim_servo=sim)
tab = RT.RealCupTaskTabUI(tk, ttk, fr, Pipe(), JP(), Cam(), log=lambda s: None, ask=lambda a, b: False, gripper=grip)


def pump(sec):
    t_end = time.time() + sec
    while time.time() < t_end:
        root.update()
        time.sleep(0.03)


def wait_idle(timeout=20):
    t_end = time.time() + timeout
    while tab.busy() and time.time() < t_end:
        pump(0.1)
    pump(0.3)


pump(0.4)
assert (tab.v_clear.get(), tab.v_align.get(), tab.v_stop.get()) == ("8.0", "60.0", "10.0")
assert (tab.v_seat.get(), tab.v_ring.get(), tab.v_roll.get()) == ("37.5", "7.0", "90.0")   # V3.76 (V3.75: 0, before: -45)
geo = tab.v_geo.get()
assert "原地转腕" in geo and "固定指离杯壁 8 mm" in geo and "杯口上方 10 mm" in geo, geo
print("[T1] geometry / safety panel with defaults and the align -> rotate -> descend description")

tab.v_port.set(GR.SIM_PORT)
tab.on_grip_connect()
wait_idle()
assert grip.connected
tab.on_grip("touch", None)
wait_idle()
assert grip.cfg["touch_calibrated"] and abs(grip.cfg["touch_deg"] - 1.85) < 0.35, grip.cfg
assert "碰合角" in tab.v_grip.get(), tab.v_grip.get()
tab.on_grip("clamp", None)
wait_idle()
assert not grip.held and "夹持中" not in tab.v_grip.get(), tab.v_grip.get()
log = tab.txt.get("1.0", "end")
assert "main thread is not in main loop" not in log and "[FAIL]" not in log, log[-600:]
print(f"[T2] 空夹标定: touch {grip.cfg['touch_deg']:+.2f} deg shown; empty clamp afterwards not held")

tab.v_clear.set("10")
tab.refresh_skill()
assert "固定指离杯壁 10 mm" in tab.v_geo.get()
tab._save()
tab.shutdown()
root.destroy()
saved = json.loads((TMP / "cameracalib" / "real_cup_task.json").read_text(encoding="utf-8"))
assert saved["geometry"]["clearance_mm"] == 10.0 and saved["gripper"]["touch_calibrated"]
SNAP1 = {str(p.relative_to(ROOT)): H(p.read_bytes()) for p in ROOT.rglob("*")
         if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts and "logs" not in p.parts}
assert SNAP1 == SNAP0, sorted(set(SNAP0.items()) ^ set(SNAP1.items()))[:6]
print("[T3] description follows the fields; settings in NERO_CALIB_DIR; release folder unchanged")
print("ALL V3.67 GUI TESTS PASSED")
