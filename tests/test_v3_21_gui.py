#!/usr/bin/env python3
"""V3.21 GUI test (needs tkinter + a display). Skips otherwise.

Hand guiding as on the real NERO (fw V111): after 进入示教模式 /feedback/joint_states stops; the hand-guided angles
only come from pyAgxArm get_leader_joint_angles (fake SDK reads them from a file = "the arm moved by hand").
记录角点 x3 must record the corners (tool tip = FK of the leader angles), 生成安全区 builds the zone; with no source
at all the log explains which feed is missing.
"""
import json, math, os, sys, tempfile, textwrap, time
HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
try:
    import tkinter as tk
    from tkinter import messagebox
except Exception:
    print("[GUI21] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI21] no DISPLAY -> skipped"); sys.exit(0)
import NERO_V312_Console as C
import NERO_V314_Joint_Control as JC
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

TMP = tempfile.mkdtemp(prefix="v321gui_")
SE.STATE_PATH = os.path.join(TMP, "state.json")
CS.WORKSPACE_PATH = os.path.join(TMP, "ws.json")
CS.LOG_DIR = os.path.join(TMP, "logs")
LEADER = os.path.join(TMP, "leader.json")
PAUSE = os.path.join(TMP, "pause_js")
FAKE = os.path.join(TMP, "fake")
os.makedirs(os.path.join(FAKE, "pyAgxArm"))
open(os.path.join(FAKE, "pyAgxArm", "__init__.py"), "w").write(textwrap.dedent(f"""
    import json, os
    class ArmModel: NERO = "NERO"
    class NeroFW: V111 = "V111"
    def create_agx_arm_config(**kw): return kw
    class _Arm:
        cm = 1
        def connect(self): pass
        def enable(self): return True
        def set_normal_mode(self): _Arm.cm = 1
        def set_leader_mode(self): _Arm.cm = 6
        def get_arm_status(self): return {{"ctrl_mode": _Arm.cm}}
        def get_leader_joint_angles(self):
            try:
                return json.load(open({LEADER!r}))
            except Exception:
                return None
    class AgxArmFactory:
        @staticmethod
        def create_arm(cfg): return _Arm()
"""))
os.environ["PYTHONPATH"] = FAKE + os.pathsep + os.environ.get("PYTHONPATH", "")
messagebox.askyesno = lambda *a, **k: True
U = JC.parse_urdf(open(os.path.join(HERE, "config", "nero.urdf")).read())
J = [f"joint{i}" for i in range(1, 8)]
LIM = JC.build_model(open(os.path.join(HERE, "config", "nero.urdf")).read(), None, None)["limits"]
TOOL = CS.tool_transform({"mode": "sim_tcp"}, U)
SEED = [0, 1.3, 0, 0.9, 0, 0.0, 0]
MARKS = [(-0.40, 0.20), (-0.55, 0.20), (-0.55, -0.20)]
POSES = [CS.ik_tool(U, J, SEED, "base_link", TOOL, [x, y, 0.002], LIM)[0] for x, y in MARKS]
assert all(POSES)
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


def driver(root):
    ed = cap["ed"]
    pipe = ed.pipe

    def wait(cond, t=30):
        t_end = time.time() + t
        while time.time() < t_end and not cond():
            root.update()
            time.sleep(0.05)
        return cond()

    def go():
        pipe.monitor.start([sys.executable, "-u", os.path.join(HERE, "tests", "v315_fake_monitor.py")],
                           env=dict(os.environ, FAKE_JS_PAUSE_FILE=PAUSE, FAKE_Q0="0,0.5,0,-0.6,0,-0.27,0"), stdin_pipe=True)
        for k in RR.STEP_KEYS:
            pipe.state[k] = (RR.GREEN, "fake")
        wait(lambda: ed.jpanel.enabled)
        ed.on_teach_enter()
        wait(lambda: pipe.teach_mode)
        open(PAUSE, "w").close()                     # V111 leader mode: joint_states stop
        time.sleep(2.5)
        root.update()
        ed.on_teach_record()                         # no leader angles yet -> explained, nothing recorded
        res["empty_log"] = logs(root)
        res["n0"] = len(ed.teach_pts)
        for q in POSES:                              # "move the arm by hand" to each mark
            json.dump({"_msg": q}, open(LEADER, "w"))
            wait(lambda q=q: pipe.last_pleader and max(abs(a - b) for a, b in zip(pipe.last_pleader, q)) < 1e-4)
            time.sleep(0.3)
            root.update()
            ed.on_teach_record()
        res["pts"] = [tp["xyz"] for tp in ed.teach_pts]
        root.update()
        time.sleep(0.4)
        root.update()
        res["tip"] = ed.v_tip.get()
        ed.on_teach_generate()
        res["poly"] = ed.fence()["polygon"]
        res["logs"] = logs(root)
        os.remove(PAUSE)
        ed.on_teach_exit()
        wait(lambda: not pipe.teach_mode)
        pipe.monitor.send({"quit": True})
        root.after(300, root.destroy)
    root.after(1200, go)


def patched(self, n=0):
    driver(self)
    _orig(self, n)


tk.Tk.mainloop = patched
C.run_console("real")
assert res["n0"] == 0 and "没有新鲜的关节角，角点未记录" in res["empty_log"] and "pyAgxArm 主导臂 无" in res["empty_log"]
assert len(res["pts"]) == 3, res["pts"]
for (mx, my), p in zip(MARKS, res["pts"]):
    assert math.hypot(p[0] - mx, p[1] - my) < 0.002 and abs(p[2] - 0.002) < 0.002, (mx, my, p)
assert "pyAgxArm" in res["tip"] and "leader" in res["tip"], res["tip"]
assert len(res["poly"]) == 3 and "[OK] [V3.16 teach] 安全区生成：3 个角点" in res["logs"]
print("[GUI21] hand guiding with silent joint_states: corners recorded from pyAgxArm leader angles (3/3, < 2 mm), "
      "zone generated; missing feeds explained")
