#!/usr/bin/env python3
"""GUI smoke test for the V3.12 console (needs tkinter + a display, e.g. xvfb-run). Skips otherwise.

Drives the real window: start button -> fake launcher replays a recorded V3.11 startup -> all 8 lamps green +
"SYSTEM READY"; stop button -> launcher gets SIGINT -> status "已停止"; start button enabled again.
Optional: NERO_V312_SHOT=<png> saves a screenshot (ImageMagick `import`).
"""
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

tmp = tempfile.mkdtemp(prefix="v312gui_")
fake = os.path.join(tmp, "fake_sim.sh")
log_lines = [
    "[OK] Release integrity: all files match SHA256SUMS.txt", "[OK] py_compile PASS",
    "[OK] V3.07 pre-launch check: no stale ROS/NERO processes", "[OK] GPU server READY PID=1",
    "[OK] cuRobo pose self-test PASS (no robot motion)", "[OK] All NERO modules resolve inside this folder",
    "[OK] ROS topic READY: /feedback/joint_states", "[OK] Primary /move_group IK solver verified: BioIK",
    "[OK] Reference /trac_ref/move_group IK solver verified: TRAC-IK",
    "[READY] SYSTEM READY: NERO V3.12 | BioIK primary + cuRobo GPU + MuJoCo | TRAC optional | single arm_controller",
]
body = "\n".join(f'echo "[$(date +%H:%M:%S)] {ln}"; sleep 0.15' for ln in log_lines)
open(fake, "w").write("#!/usr/bin/env bash\ntrap 'echo \"[INFO] Shutdown: stopping NERO processes\"' EXIT\n"
                      "trap 'echo \"[INFO] Stop signal received\"; exit 130' INT TERM\n"
                      f"{body}\nsleep 300 & wait $!\n")
os.environ["NERO_V312_SIM_SCRIPT"] = fake
import NERO_V312_Console as C  # noqa: E402

messagebox.askyesno = lambda *a, **k: True
result = {}
_orig_mainloop = tk.Tk.mainloop


def walk(w):
    yield w
    for c in w.winfo_children():
        yield from walk(c)


def find_button(root, text):
    for w in walk(root):
        try:
            if w.winfo_class() == "TButton" and w.cget("text") == text:
                return w
        except Exception:
            pass
    raise AssertionError(f"button not found: {text}")


def status_text(root):
    for w in walk(root):
        try:
            v = str(w.cget("textvariable"))
        except Exception:
            continue
        if v:
            s = root.globalgetvar(v)
            if any(x in s for x in ("未运行", "启动中", "SYSTEM READY", "已停止", "失败", "正在停止")):
                return s
    return ""


def lamp_colors(root):
    out = []
    for w in walk(root):
        if w.winfo_class() == "Canvas" and w.winfo_ismapped():
            for item in w.find_all():
                out.append(w.itemcget(item, "fill"))
    return out


def driver(root):
    def step(fn, delay):
        root.after(delay, fn)

    def s1():
        find_button(root, "▶ 启动仿真").invoke()
        step(s2, 100)

    def s2(t0=[time.time()]):
        if "SYSTEM READY" in status_text(root) or time.time() - t0[0] > 15:
            root.update()
            result["ready_status"] = status_text(root)
            result["ready_lamps"] = lamp_colors(root)
            result["start_state_ready"] = str(find_button(root, "▶ 启动仿真").cget("state"))
            shot = os.environ.get("NERO_V312_SHOT")
            if shot:
                subprocess.run(["import", "-window", "root", shot], check=False)
            find_button(root, "■ 停止仿真").invoke()
            step(s3, 100)
        else:
            step(s2, 100)

    def s3(t0=[time.time()]):
        st = status_text(root)
        if "已停止" in st or time.time() - t0[0] > 15:
            result["stop_status"] = st
            result["start_state_after"] = str(find_button(root, "▶ 启动仿真").cget("state"))
            root.destroy()
        else:
            step(s3, 100)

    step(s1, 300)


def patched_mainloop(self, n=0):
    driver(self)
    _orig_mainloop(self, n)


tk.Tk.mainloop = patched_mainloop
C.run_console("sim")
print(result)
assert "SYSTEM READY" in result["ready_status"], result
sim_lamps = result["ready_lamps"][:8]
assert sim_lamps == [C.LAMP_COLORS[C.GREEN]] * 8, sim_lamps
assert result["start_state_ready"] == "disabled"
assert "已停止" in result["stop_status"] and "130" in result["stop_status"], result
assert result["start_state_after"] == "normal"
print("[GUI] console: start -> 8 green lamps + SYSTEM READY; stop -> 已停止 (exit 130); start re-enabled")
