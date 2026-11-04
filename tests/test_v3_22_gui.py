#!/usr/bin/env python3
"""V3.22 GUI test (from V3.21) (needs tkinter + a display). Skips otherwise.

As seen on the real NERO (V111): set_leader_mode makes the arm limp, ctrl_mode keeps reporting 0x01, pyAgxArm shows no
leader angles, /feedback/joint_states stops and the driver streams /feedback/leader_joint_states. Teach mode must be
accepted, 记录角点 x3 records the corners from the leader topic, and 退出示教模式 holds the pose (ROS controller sync,
speed 10 % + move_j before enable) and then returns to zero via MoveIt at 10 %.
"""
import json, math, os, sys, tempfile, textwrap, time
HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
try:
    import tkinter as tk
    from tkinter import messagebox
except Exception:
    print("[GUI22] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI22] no DISPLAY -> skipped"); sys.exit(0)
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

TMP = tempfile.mkdtemp(prefix="v322gui_")
SE.STATE_PATH = os.path.join(TMP, "state.json")
CS.WORKSPACE_PATH = os.path.join(TMP, "ws.json")
CS.LOG_DIR = os.path.join(TMP, "logs")
LJS = os.path.join(TMP, "ljs.json")
PAUSE = os.path.join(TMP, "pause_js")
CALLS = os.path.join(TMP, "calls.txt")
HOLDS = os.path.join(TMP, "holds.txt")
FAKE = os.path.join(TMP, "fake")
os.makedirs(os.path.join(FAKE, "pyAgxArm"))
open(os.path.join(FAKE, "pyAgxArm", "__init__.py"), "w").write(textwrap.dedent(f"""
    import json, os
    def rec(x):
        open({CALLS!r}, "a").write(x + "\\n")
    class ArmModel: NERO = "NERO"
    class NeroFW: V111 = "V111"
    def create_agx_arm_config(**kw): return kw
    class _Arm:
        def connect(self): pass
        def enable(self): rec("enable"); return True
        def set_normal_mode(self):
            rec("set_normal_mode")
            for f in ({PAUSE!r}, {LJS!r}):           # the arm leaves leader mode: joint_states back, leader topic gone
                try:
                    os.remove(f)
                except OSError:
                    pass
        def set_leader_mode(self): rec("set_leader_mode")
        def get_arm_status(self): return {{"ctrl_mode": 1}}
        def get_leader_joint_angles(self): return None
        def set_speed_percent(self, p): rec(f"speed {{p}}")
        def move_j(self, q): rec("move_j")
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
                           env=dict(os.environ, FAKE_JS_PAUSE_FILE=PAUSE, FAKE_LJS_FILE=LJS, FAKE_HOLD_LOG=HOLDS,
                                    FAKE_Q0="0,0.5,0,-0.6,0,-0.27,0", FAKE_RAD_S="3.0"), stdin_pipe=True)
        for k in RR.STEP_KEYS:
            pipe.state[k] = (RR.GREEN, "fake")
        wait(lambda: ed.jpanel.enabled)
        q_now = list(pipe.last_js["position"])
        json.dump(q_now, open(LJS, "w"))           # set_leader_mode: driver starts the leader topic ...
        open(PAUSE, "w").close()                     # ... and joint_states stop
        ed.on_teach_enter()
        wait(lambda: pipe.teach_mode, 20)
        res["entered"] = pipe.teach_mode
        for q in POSES:                              # "move the arm by hand" to each mark
            json.dump(q, open(LJS, "w"))
            wait(lambda q=q: pipe.last_ljs and max(abs(a - b) for a, b in zip(pipe.last_ljs["position"], q)) < 1e-4)
            time.sleep(0.2)
            root.update()
            ed.on_teach_record()
        res["pts"] = [tp["xyz"] for tp in ed.teach_pts]
        root.update()
        time.sleep(0.4)
        root.update()
        res["tip"] = ed.v_tip.get()
        res["label"] = ed.v_teach_mode.get()
        ed.on_teach_generate()
        res["poly"] = ed.fence()["polygon"]
        ed.on_teach_exit()
        wait(lambda: not pipe.teach_mode, 30)
        wait(lambda: "[V3.22 teach] 回零位" in logs(root), 60)
        root.update()
        res["logs"] = logs(root)
        res["q_end"] = list(pipe.last_js["position"])
        pipe.monitor.send({"quit": True})
        root.after(300, root.destroy)
    root.after(1200, go)


def patched(self, n=0):
    driver(self)
    _orig(self, n)


tk.Tk.mainloop = patched
C.run_console("real")
assert res["entered"], res["logs"][-3000:]
assert "依据=/feedback/leader_joint_states 在更新" in res["logs"]
assert len(res["pts"]) == 3, res["pts"]
for (mx, my), p in zip(MARKS, res["pts"]):
    assert math.hypot(p[0] - mx, p[1] - my) < 0.002 and abs(p[2] - 0.002) < 0.002, (mx, my, p)
assert "leader_joint_states" in res["tip"], res["tip"]
assert "示教模式" in res["label"], res["label"]
assert len(res["poly"]) == 3
c = open(CALLS).read().split("\n")
i_en = c.index("enable")
assert "speed 10" in c[:i_en] and "move_j" in c[:i_en], c
h = [json.loads(l) for l in open(HOLDS)]
assert len(h) == 2 and max(abs(a - b) for a, b in zip(h[0]["positions"], POSES[-1])) < 1e-3, h
assert "[OK] [V3.22 teach] 回零位: 完成" in res["logs"], "\n".join(l for l in res["logs"].split("\n") if "V3.2" in l or "FAIL" in l)[-3000:]
assert "先竖直抬起工具点" in res["logs"] and "以 10% 速度经 MoveIt 回零位" in res["logs"]
assert max(abs(v) for v in res["q_end"]) < 0.01, res["q_end"]
import csv
rows = [r for r in csv.DictReader(open(os.path.join(CS.LOG_DIR, "real_motion_log.csv"))) if "退出示教" in r["label"]]
assert [r["label"] for r in rows] == ["退出示教 抬起", "退出示教回零"] and all(r["speed"] == "0.10" for r in rows), rows
print("[GUI22] ctrl_mode stuck at 0x01 + leader topic: teach accepted, 3/3 corners (< 2 mm) from the leader topic, zone "
      "generated; exit: ROS controller synced, speed 10 % + move_j before enable, lift tool straight up, MoveIt home at 10 % -> zero")
