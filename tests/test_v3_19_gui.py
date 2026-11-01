#!/usr/bin/env python3
"""V3.19 GUI test (needs tkinter + a display, e.g. xvfb-run). Skips otherwise.

G1 teach panel: 进入示教模式 (fake pyAgxArm set_master_mode) -> label shows teach mode, Home refused;
   退出示教模式 -> normal + enabled
G2 回 Home: MoveIt joint goal to all zeros from a non-zero pose (preview + confirm), arm ends at 0
G3 zone files: 另存为… (chosen name/folder) -> file + remembered; edit; 打开… restores it; next start opens it
"""
import json, os, sys, tempfile, textwrap, time
HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
try:
    import tkinter as tk
    from tkinter import messagebox, filedialog
except Exception:
    print("[GUI19] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI19] no DISPLAY -> skipped"); sys.exit(0)
import NERO_V312_Console as C
import NERO_V315_Consistency_Safety as CS
_v323_dw = CS.default_workspace


def _v323_legacy_default():          # V3.23: this test covers the V3.16-V3.22 layered rule (now rule B, 4 corners)
    w = _v323_dw()
    w["fence"].update(rule="B", margin=0.04, polygon=[[-0.15, 0.45], [-0.70, 0.45], [-0.70, -0.45], [-0.15, -0.45]])
    w["corners"] = w["corners"][:4]
    return w


CS.default_workspace = _v323_legacy_default
import NERO_V316_Safety_Editor as SE
import NERO_V313_Real_Robot as RR

TMP = tempfile.mkdtemp(prefix="v319gui_")
SE.STATE_PATH = os.path.join(TMP, "state.json")
CS.WORKSPACE_PATH = os.path.join(TMP, "default_ws.json")
CS.LOG_DIR = os.path.join(TMP, "logs")
FAKE = os.path.join(TMP, "fake")
os.makedirs(os.path.join(FAKE, "pyAgxArm"))
open(os.path.join(FAKE, "pyAgxArm", "__init__.py"), "w").write(textwrap.dedent("""
    class ArmModel: NERO = "NERO"
    class NeroFW: V111 = "V111"
    def create_agx_arm_config(**kw): return kw
    class _Arm:
        cm = 1
        def connect(self): pass
        def enable(self): return True
        def set_normal_mode(self): _Arm.cm = 1
        def set_leader_mode(self): _Arm.cm = 6
        def get_arm_status(self): return {"ctrl_mode": _Arm.cm}
    class AgxArmFactory:
        @staticmethod
        def create_arm(cfg): return _Arm()
"""))
os.environ["PYTHONPATH"] = FAKE + os.pathsep + os.environ.get("PYTHONPATH", "")
asked = []
messagebox.askyesno = lambda t, x, **k: (asked.append(t) or True)
ZONE = os.path.join(TMP, "my_zones", "鸡蛋工位.json")
os.makedirs(os.path.dirname(ZONE))
filedialog.asksaveasfilename = lambda **k: ZONE
filedialog.askopenfilename = lambda **k: ZONE
cap = {}
_i = SE.SafetyEditorUI.__init__


def init(self, *a, **k):
    _i(self, *a, **k)
    cap["ed"] = self


SE.SafetyEditorUI.__init__ = init
res = {}
_orig = tk.Tk.mainloop


def logs(root):
    def walk(w):
        yield w
        for c in w.winfo_children():
            yield from walk(c)
    return "\n".join(w.get("1.0", "end") for w in walk(root) if w.winfo_class() == "Text")


def run_once(driver):
    def patched(self, n=0):
        driver(self)
        _orig(self, n)
    tk.Tk.mainloop = patched
    C.run_console("real")


def first(root):
    ed = cap["ed"]
    pipe, jp = ed.pipe, ed.jpanel

    def wait(cond, t=40):
        t_end = time.time() + t
        while time.time() < t_end and not cond():
            root.update()
            time.sleep(0.05)
        return cond()

    def go():
        pipe.monitor.start([sys.executable, "-u", os.path.join(HERE, "tests", "v315_fake_monitor.py")],
                           env=dict(os.environ, FAKE_RAD_S="20", FAKE_Q0="0.2,0.5,0.1,-0.6,0.1,-0.27,0.3"), stdin_pipe=True)
        for k in RR.STEP_KEYS:
            pipe.state[k] = (RR.GREEN, "fake")
        wait(lambda: jp.enabled)
        # G1
        ed.on_teach_enter()
        wait(lambda: pipe.teach_mode and "手动拖动" in ed.v_teach_mode.get())
        res["teach_label"] = ed.v_teach_mode.get()
        res["editor_mode"] = ed.mode
        jp.go_home()
        root.update()
        res["home_refused"] = "先「退出示教模式」" in logs(root)
        ed.on_teach_exit()
        wait(lambda: not pipe.teach_mode and "退出示教模式:" in logs(root) and "正常" in ed.v_teach_mode.get())
        res["normal_label"] = ed.v_teach_mode.get()
        ed.v_mode.set("edit")
        ed.on_mode()
        # G2
        a0 = jp.actual_deg()
        res["a0"] = a0
        jp.go_home()
        wait(lambda: "回 Home: 规划" in logs(root) or "[FAIL] [V3.14 joints] 回 Home" in logs(root), 40)
        time.sleep(0.5)
        root.update()
        res["after_home"] = jp.actual_deg()
        # G3
        ed.on_save_as()
        res["saved_exists"] = os.path.isfile(ZONE)
        res["state"] = json.load(open(SE.STATE_PATH))
        poly0 = json.loads(json.dumps(ed.fence()["polygon"]))
        ed._select(("v", 0))
        ed.sel_vars["x"].set("-0.300")
        ed.on_sel_apply()
        res["edited"] = list(ed.fence()["polygon"][0])
        ed.on_open()
        res["reopened"] = ed.fence()["polygon"] == poly0
        res["head"] = ed.v_head.get()
        res["asked"] = list(asked)
        res["logs"] = logs(root)
        pipe.monitor.send({"quit": True})
        root.after(300, root.destroy)
    root.after(1200, go)


def second(root):
    ed = cap["ed"]

    def go():
        res["second_path"] = ed.path
        res["second_logs"] = logs(root)
        root.after(100, root.destroy)
    root.after(800, go)


run_once(first)
run_once(second)
L = res["logs"]
assert "示教模式（手动拖动）" in res["teach_label"] and res["editor_mode"] == "teach" and res["home_refused"], res
assert "正常控制" in res["normal_label"] and "[OK] [V3.22 teach] 进入示教模式" in L and "[OK] [V3.22 teach] 退出示教模式" in L
assert any(abs(x) > 5 for x in res["a0"]) and all(abs(x) < 0.2 for x in res["after_home"]), (res["a0"], res["after_home"])
assert "回 Home" in res["asked"] and "[OK] [V3.14 joints] 回 Home: 规划" in L
assert res["saved_exists"] and res["state"]["last_zone_file"] == os.path.abspath(ZONE)
assert json.load(open(ZONE, encoding="utf-8"))["name"] == "鸡蛋工位"
assert res["edited"][0] == -0.3 and res["reopened"] and "鸡蛋工位.json" in res["head"]
assert res["second_path"] == os.path.abspath(ZONE) and "已打开上次使用的安全区文件" in res["second_logs"]
print("[GUI19] teach mode enter/exit (label, Home refused while teaching); Home via MoveIt to all zeros; zone 另存为 / "
      "打开 / auto-open at the next start")
