#!/usr/bin/env python3
"""V4.8 GUI test: 实机NERO调试 -> 夹爪 panel with the simulated AgxGripper (needs tkinter + DISPLAY; skips otherwise).
G1 switch SO-101 -> NERO AgxGripper in the panel: the AgxGripper columns appear, the global setting is saved; a
   blocker (task running) refuses the switch.
G2 connect: lamp green, width / force / enable / homed shown, a fault lamp turns red and 「清除故障」 clears it.
G3 operation: 到这个开口 30 mm, 夹紧 on a 45 mm object -> held, presets open to their width.
G4 校正归零: the five steps with one button (失能 -> 用手合拢 -> 归零 -> 全行程检查), result saved.
G5 力控标定: empty baseline saved; gauge table with two points -> the fitted text; presets validated + saved.
G6 示教模式测试: firmware ignores commands in teach mode -> "不能"; accepts -> "能".
G7 J7 控制器限位: read -90..90, write -87..87 (read back), drag test sees 95 deg -> "不挡", restore.
G8 layout: every column on screen at 1400x900; no emoji in any widget text.
"""
import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    print("[SKIP] test_v4_8_gui: no tkinter"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[SKIP] test_v4_8_gui: no DISPLAY"); sys.exit(0)
os.environ["NERO_V48_FAKE_GRIPPER"] = "1"
os.environ.pop("NERO_GRIPPER", None)
TMP = tempfile.mkdtemp(prefix="nero_v48_gui_")
import NERO_V48_Gripper as GG                 # noqa: E402
GG.CFG_PATH = os.path.join(TMP, "gripper.json")
import NERO_V48_Gripper_UI as GUI             # noqa: E402


class FakeSO101:
    connected = False
    held = False

    def disconnect(self):
        pass


root = tk.Tk()
root.geometry("1400x900+0+0")
LOG = []
asks = []
proxy = GG.GripperProxy(FakeSO101())
state = {"teach": True}
panel = GUI.GripperPanelUI(tk, ttk, root, proxy, log=LOG.append, ask=lambda t, m: asks.append(t) or True,
                           teach_fn=lambda: state["teach"])


def pump(t):
    t_end = time.time() + t
    while time.time() < t_end:
        root.update()
        time.sleep(0.01)


def wait(fn, t=8.0):
    t_end = time.time() + t
    while time.time() < t_end:
        root.update()
        if fn():
            return True
        time.sleep(0.01)
    return fn()


def logs(tag, n0=0):
    return [s for s in LOG[n0:] if tag in s]


# ---- G1 switch
pump(0.3)
assert not panel.body.winfo_ismapped() and panel.so101_hint.winfo_ismapped()
GG.add_blocker(lambda: "测试：任务运行中" if state.get("block") else None)
state["block"] = True
panel.v_type.set(GG.LABELS["agx_gripper"])
panel.on_type()
assert GG.active_type() == "so101" and logs("现在不能切换夹爪"), LOG[-2:]
assert panel.v_type.get() == GG.LABELS["so101"]
state["block"] = False
panel.v_type.set(GG.LABELS["agx_gripper"])
panel.on_type()
pump(0.4)
assert GG.active_type() == "agx_gripper" and GG.load_cfg()["type"] == "agx_gripper"
assert panel.body.winfo_ismapped() and not panel.so101_hint.winfo_ismapped()
assert logs("接下来") and "effector_type:=none" in logs("接下来")[-1]
print("[G1] switch to the NERO AgxGripper: columns shown, saved, blocked while a task runs")

# ---- G2 connect + status + faults
panel.on_connect()
assert wait(lambda: panel.lamp_conn.state == "ok")
fake = panel.agx().backend
assert wait(lambda: panel.v_width.get().startswith("50.0"))
assert panel.lamp_home.state == "ok" and "Hz" in panel.v_fps.get()
fake.faults["driver_overcurrent"] = True
assert wait(lambda: panel.fault_lamps["driver_overcurrent"].state == "bad")
panel.on_reset()
assert wait(lambda: panel.fault_lamps["driver_overcurrent"].state == "ok")
print("[G2] connected; width / homed / feedback rate shown; overcurrent lamp red, 清除故障 clears it")

# ---- G3 operation
panel.v_goal_txt.set("30")
panel.on_goto()
assert wait(lambda: panel.busy is None and "张开到 30 mm" in panel.v_last.get())
assert abs(fake.width - 30.0) <= 1.0, (fake.width, fake.target, panel.v_last.get(), LOG[-3:])
fake.object_mm = 22.0
panel._set_force(0.5)
panel.on_clamp()
assert wait(lambda: panel.busy is None and "夹住了东西" in panel.v_last.get()), panel.v_last.get()
assert abs(fake.force_set - 0.5) < 1e-6
fake.object_mm = None
panel.on_preset("cup")
assert wait(lambda: panel.busy is None and abs(fake.width - 95.0) <= 1.0)
print("[G3] goto 30 mm, clamp (0.5) holds a 22 mm object, cup preset opens to 95 mm")

# ---- G4 zero wizard
fake.zero_offset = 0.0
fake.width = 3.0                                         # the user closes the fingers by hand to the true zero
panel.on_zero_next()                                     # 1 check
assert panel.zero_rows[0][0].state == "ok"
panel.on_zero_next()                                     # 2 disable
assert wait(lambda: panel.zero_step == 2)
assert not fake.enabled and panel.zero_rows[2][0].state == "warn"
fake.width = 0.0
panel.on_zero_next()                                     # 3 closed by hand -> 4 zero
assert wait(lambda: panel.zero_step == 4)
assert panel.zero_rows[3][0].state == "ok"
panel.on_zero_next()                                     # 5 stroke check
assert wait(lambda: panel.zero_step == 0 and panel.busy is None, 15.0)
assert panel.zero_rows[4][0].state == "ok", panel.zero_rows[4][1].get()
z = GG.load_cfg()["zero"]
assert z and abs(z["open_mm"] - 100.0) < 1.6 and abs(z["closed_mm"]) <= 1.0
print(f"[G4] 校正归零 5 steps: closed {z['closed_mm']:.1f} mm, full {z['open_mm']:.1f} mm, saved")

# ---- G5 force calibration
panel.on_baseline()
assert wait(lambda: panel.busy is None and GG.load_cfg().get("baseline"), 20.0)
assert "阈值" in panel.v_base.get()
for s_, n_ in (("0.5", "6.0"), ("3.0", "44.0")):
    panel.v_fs.set(s_)
    panel.v_gauge.set(n_)
    panel.on_gauge_record()
assert GG.load_cfg()["force_table"] == [[0.5, 6.0], [3.0, 44.0]]
assert "已标定" in panel.v_gauge_fit.get() and len(panel.tv_gauge.get_children()) == 2
assert abs(GG.force_level(25.0) - (0.5 + 19.0 * 2.5 / 38.0)) < 0.02
panel.preset_vars["egg"][1].set("9")
panel.on_preset_save()
assert logs("预设没保存")
panel.preset_vars["egg"][1].set("0.6")
panel.preset_vars["egg"][0].set("58")
panel.on_preset_save()
assert GG.load_cfg()["presets"]["egg"] == {"open_mm": 58.0, "force": 0.6}
print("[G5] baseline saved, gauge table 2 points -> force_level interpolates, presets validated + saved")

# ---- G6 teach-mode test
fake.teach_mode, fake.teach_accepts = True, False
panel.on_teach_test()
assert wait(lambda: panel.busy is None and "不能" in panel.v_teach.get(), 10.0), panel.v_teach.get()
fake.teach_accepts = True
panel.on_teach_test()
assert wait(lambda: panel.busy is None and "能 开合" in panel.v_teach.get(), 10.0), panel.v_teach.get()
fake.teach_mode = False
print("[G6] teach-mode test: ignored -> 不能, followed -> 能 (saved)")

# ---- G7 J7 controller limit
panel.on_j7_read()
assert wait(lambda: panel.busy is None and "-90.0" in panel.v_j7.get())
panel.on_j7_write()
assert wait(lambda: panel.busy is None and "已写入并读回" in panel.v_j7.get()), panel.v_j7.get()
assert abs(fake.j_limits[7][1] - 1.5184) < 1e-3
fake.j7_now = 1.66                                       # 95 deg: the controller did not hold it
panel.on_j7_test(seconds=0.6)
assert wait(lambda: not panel.j7_test["running"] and GG.load_cfg().get("j7_test"), 5.0)
pump(0.2)
t = GG.load_cfg()["j7_test"]
assert t["holds"] is False and t["max_abs"] > 94.0, t
assert "不挡" in panel.v_j7.get()
panel.on_j7_restore()
assert wait(lambda: panel.busy is None and "已恢复" in panel.v_j7.get())
assert abs(fake.j_limits[7][1] - 1.5708) < 1e-3
print("[G7] J7: read -90..90, write -87..87 (read back), drag test 95 deg -> not held, restored")

# ---- G8 layout + no emoji
root.update()


def walk(w):
    yield w
    for c in w.winfo_children():
        yield from walk(c)


bad = []
for w in walk(root):
    for opt in ("text",):
        try:
            v = str(w.cget(opt))
        except Exception:
            continue
        if any(ord(c) >= 0x1F000 or 0x2600 <= ord(c) <= 0x27BF for c in v):
            bad.append(v)
assert not bad, bad
for w in (panel.btn_conn, panel.sc_goal, panel.nb, panel.btn_zero):
    x, y = w.winfo_rootx() + w.winfo_width(), w.winfo_rooty() + w.winfo_height()
    assert w.winfo_ismapped() and x <= 1400 + root.winfo_rootx() and y <= 900 + root.winfo_rooty(), (w, x, y)
for k in range(4):
    panel.nb.select(k)
    pump(0.1)
print("[G8] all controls on screen at 1400x900, no emoji")
panel.shutdown()
root.destroy()
print("ALL V4.8 GUI TESTS PASSED")
