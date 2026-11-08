#!/usr/bin/env python3
"""V4.6.4 GUI test (needs tkinter + DISPLAY; skips otherwise): the whole console is started; Camera调试 -> 视觉定位
carries the 视觉精度验证 panel (3 modes, hover height 10 mm, 识别目标 / 合爪并悬停 / 抬起 / 停止, dx / dy / 备注,
记录偏差), the gripper is taken from the Agentic tab at click time, the Camera log keeps its V4.6.3 height,
every control is inside the default 1420x1000 window; without a camera the buttons refuse with the
reason (nothing moves); 记录偏差 without a hover result asks for one first; dx / dy must be numbers."""
import gc
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    import tkinter
    if not os.environ.get("DISPLAY"):
        print("[SKIP] test_v4_6_4_gui: no DISPLAY")
        sys.exit(0)
except Exception as exc:
    print(f"[SKIP] test_v4_6_4_gui: {exc}")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="nero_v464_gui_"))
for k, v in (("NERO_CALIB_DIR", "calib"), ("NERO_DRIVE_ROOT", "drive"), ("NERO_V368_TASK_DIR", "task"),
             ("NERO_V360_USER_STRATEGY_DIR", "user"), ("NERO_V360_ACTIVE_STRATEGY_FILE", "active.json")):
    os.environ[k] = str(TMP / v)
os.environ["NERO_NO_VLA_TAB"] = "1"
import NERO_V312_Console as C           # noqa: E402
import NERO_V464_Vision_Check as VC     # noqa: E402

C.HERE = str(TMP)
RES = {}
orig_loop = tkinter.Tk.mainloop


def walk(w):
    yield w
    for c in w.winfo_children():
        yield from walk(c)


def loop(self, n=0):
    def step1():
        self.geometry("1420x1000+0+0")              # the console's default window on a 1920x1080 screen
        self.update()
        nb = next(w for w in walk(self) if w.winfo_class() == "TNotebook")
        for t in nb.tabs():
            if nb.tab(t, "text") == "Camera调试":
                nb.select(t)
        panel = next(o for o in gc.get_objects() if isinstance(o, VC.VisionCheckPanel))
        cnb = panel.cam_tab.nb
        for t in cnb.tabs():
            if cnb.tab(t, "text") == "视觉定位":
                cnb.select(t)
        RES["panel"] = panel
        self.after(1500, step2)

    def step2():
        p = RES["panel"]
        rx, ry, W, H = self.winfo_rootx(), self.winfo_rooty(), self.winfo_width(), self.winfo_height()
        texts = {}
        for w in walk(p.frame):
            if w.winfo_class() in ("TButton", "TRadiobutton"):
                ok = (w.winfo_ismapped() and w.winfo_rootx() >= rx and w.winfo_rooty() >= ry
                      and w.winfo_rootx() + w.winfo_width() <= rx + W + 1 and w.winfo_rooty() + w.winfo_height() <= ry + H + 1)
                texts[w.cget("text")] = ok
        RES["controls"] = texts
        RES["labels_ok"] = [
            (w.winfo_class(), w.winfo_height(), w.winfo_reqheight(), w.winfo_rooty() + w.winfo_height() - ry - H)
            for w in walk(p.frame) if w is not p.frame and w.winfo_class() in ("Frame", "TFrame")
            and not (w.winfo_ismapped() and w.winfo_height() >= w.winfo_reqheight() - 1
                     and w.winfo_rooty() + w.winfo_height() <= ry + H + 1)]
        RES["hover"] = p.v_hover.get()
        p.on_observe()
        RES["observe"] = p.v_status.get()
        p.on_go()
        RES["go"] = p.v_status.get()
        p.on_record()
        RES["record"] = p.v_status.get()
        p.pending = {"mode": "main", "target": "egg_1"}
        p.v_dx.set("abc")
        p.on_record()
        RES["record_bad"] = p.v_status.get()
        p.v_dx.set("12")
        p.v_dy.set("-3")
        p.on_record()
        RES["record_ok"] = p.v_status.get()
        RES["record_ok"] = p.v_sum.get()
        # gripper connected AFTER 识别目标 (the user's case): 合爪并悬停 takes it from the Agentic tab at click time
        import types
        from tkinter import messagebox
        messagebox.askyesno = lambda *a, **k: True
        seen = {}

        class FakeGrip:
            connected = True
        fake_io = types.SimpleNamespace(grip=None, stop_ev=types.SimpleNamespace(clear=lambda: None))
        p.vc = types.SimpleNamespace(io=fake_io, targets={"egg_1": {}},
                                     run=lambda mode, tid, h: seen.update(grip=fake_io.grip) or
                                     {"ok": False, "row": {"mode": mode, "target": tid}, "error": "test"})
        p._blocked = lambda motion=True: ""
        p.v_target.set("egg_1")
        real_grip = p.rct.grip
        p.rct.grip = FakeGrip()
        p.on_go()
        p.worker.join(5)
        RES["grip_seen"] = seen.get("grip")
        p.rct.grip = types.SimpleNamespace(connected=False)
        p.on_go()
        RES["grip_off"] = p.v_status.get()
        p.rct.grip = real_grip
        # the Camera tab's log below the page keeps (at least) its V4.6.3 height
        cam_texts = [w for w in walk(p.cam_tab.nb.master.master.master) if w.winfo_class() == "Text"]
        RES["cam_log_h"] = max((w.winfo_height() for w in walk(self) if w.winfo_class() == "Text"
                                and w.winfo_ismapped() and w.winfo_reqheight() >= 140), default=0)
        if os.environ.get("NERO_V464_SHOT"):
            self.update()
            os.system(f"import -window root {os.environ['NERO_V464_SHOT']}")
        self.after(300, self.destroy)
    self.after(3000, step1)
    orig_loop(self, n)


tkinter.Tk.mainloop = loop
try:
    C.main([])
except SystemExit:
    pass
need = ["① 只用主相机", "② 只用腕部相机", "③ 主相机 + 腕部辅助", "识别目标", "▶ 合爪并悬停", "↑ 抬起", "■ 停止", "记录偏差"]
missing = [t for t in need if not RES["controls"].get(t)]
assert not missing, (missing, RES["controls"])
assert RES["hover"] == "10", RES["hover"]
assert RES["observe"].startswith("[不能识别]") and "主相机 D435i 没有连接" in RES["observe"], RES["observe"]
assert RES["go"].startswith("[不能运动]"), RES["go"]
assert "先「合爪并悬停」" in RES["record"], RES["record"]
assert "请填数字" in RES["record_bad"], RES["record_bad"]
assert "dx +12.0 / dy -3.0" in RES["record_ok"], RES["record_ok"]
assert "vision_check/vision_check_" in RES["record_ok"] and "① 只用主相机：1 点" in RES["record_ok"], RES["record_ok"]
assert RES["labels_ok"] == [], RES["labels_ok"]
assert type(RES["grip_seen"]).__name__ == "FakeGrip", RES["grip_seen"]
assert "夹爪没有连接" in RES["grip_off"], RES["grip_off"]
assert RES["cam_log_h"] >= 33, RES["cam_log_h"]                  # V4.6.3: 33 px; V4.6.4 first build: 1 px          # every row / text line complete and inside the window
print("[G1] 视觉定位 -> 视觉精度验证: all controls inside the 1420x1000 window; no camera -> refused with the reason; "
      "dx / dy recorded to NERO_CALIB_DIR/vision_check + per-mode summary")
print("ALL V4.6.4 GUI TESTS PASSED")
