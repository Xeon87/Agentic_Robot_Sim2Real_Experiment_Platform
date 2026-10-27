#!/usr/bin/env python3
"""V3.23 GUI test (needs tkinter + a display). Skips otherwise.

The user's zone (4 taught corners, base outside, saved by V3.22 without a rule) opens as rule A: red header, every
motion refused. 并入底座 + 保存 -> walls synced to MoveIt (6 wall boxes with yaw). A pose high above the table but
outside the zone is refused (越墙), a pose in the middle executes. 巡检 retracts the MoveIt walls and restores them.
Hand guiding outside the zone -> alarm in the log + warning line.
"""
import json, math, os, sys, tempfile, time
HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
try:
    import tkinter as tk
    from tkinter import messagebox
except Exception:
    print("[GUI23] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI23] no DISPLAY -> skipped"); sys.exit(0)
import NERO_V312_Console as C
import NERO_V314_Joint_Control as JC
import NERO_V315_Consistency_Safety as CS
import NERO_V316_Safety_Editor as SE
import NERO_V313_Real_Robot as RR

TMP = tempfile.mkdtemp(prefix="v323gui_")
SE.STATE_PATH = os.path.join(TMP, "state.json")
CS.WORKSPACE_PATH = os.path.join(TMP, "ws.json")
CS.LOG_DIR = os.path.join(TMP, "logs")
USER = [[-0.60, -0.10], [-0.60, 0.28], [-0.08, 0.28], [-0.17, -0.11]]
w = CS.default_workspace()
json.dump({"version": "V3.17", "measured": True, "boxes": w["boxes"],
           "fence": {"polygon": USER, "margin": 0.02, "z_table": 0.0, "z_min_points": 0.03,
                     "tool": {"mode": "sim_tcp", "length": 0.1, "axis": "+x"}},
           "corners": [{"name": f"M{i + 1}", "source": "示教", "q": None} for i in range(4)]},
          open(CS.WORKSPACE_PATH, "w"))
messagebox.askyesno = lambda *a, **k: True
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
LIM = JC.build_model(open(os.path.join(HERE, "config", "nero.urdf")).read(), None, None)["limits"]
TOOL = CS.tool_transform({"mode": "sim_tcp"}, U)
Q_IN = [0, 1.6, 0, 1.6, 0, -0.1, 0]
R = [r[:3] for r in CS.tool_pose(U, J, Q_IN, "base_link", TOOL)[:3]]
Q_MID = CS.ik_tool(U, J, Q_IN, "base_link", TOOL, [-0.40, 0.09, 0.08], LIM, keep_R=R)[0]
Q_HIGH = [-0.6, 0.3, 0, 1.2, 0, 0, 0]
cap = {}
_i = SE.SafetyEditorUI.__init__


def init(self, *a, **k):
    _i(self, *a, **k)
    cap["ed"] = self


SE.SafetyEditorUI.__init__ = init
res = {}
_orig = tk.Tk.mainloop


def logs(root):
    def walk(w_):
        yield w_
        for c in w_.winfo_children():
            yield from walk(c)
    return "\n".join(w_.get("1.0", "end") for w_ in walk(root) if w_.winfo_class() == "Text")


def driver(root):
    ed = cap["ed"]
    pipe = ed.pipe
    scenes = []
    _ap = pipe.apply_scene

    def ap(objs, **k):
        scenes.append([o["id"] for o in objs])
        res["yaws"] = [o.get("yaw") for o in objs if "wall_" in o["id"]]
        return _ap(objs, **k)
    pipe.apply_scene = ap

    def wait(cond, t=30):
        t_end = time.time() + t
        while time.time() < t_end and not cond():
            root.update()
            time.sleep(0.05)
        return cond()

    def go():
        pipe.monitor.start([sys.executable, "-u", os.path.join(HERE, "tests", "v315_fake_monitor.py")],
                           env=dict(os.environ, FAKE_Q0=",".join(str(v) for v in Q_IN), FAKE_RAD_S="6.0"), stdin_pipe=True)
        for k in RR.STEP_KEYS:
            pipe.state[k] = (RR.GREEN, "fake")
        wait(lambda: ed.jpanel.enabled)
        root.update()
        res["head0"] = ed.v_head.get()
        res["rule0"] = ed.v_rule.get()
        model = ed.jpanel.model
        res["r_before"] = pipe.move_joints(model, Q_MID, 0.5, label="t")
        ed.on_merge_base()
        ed.on_save()
        wait(lambda: "[V3.23] 围墙" in logs(root), 20)
        res["head1"] = ed.v_head.get()
        res["scene_after_save"] = scenes[-1] if scenes else []
        res["yaws_save"] = list(res.get("yaws") or [])
        out = {}

        def job():
            out["high"] = pipe.move_and_settle(model, Q_HIGH, 0.5, label="越墙目标")
            out["mid"] = pipe.move_and_settle(model, Q_MID, 0.5, label="区内目标")
        import threading
        th = threading.Thread(target=job, daemon=True)
        th.start()
        wait(lambda: not th.is_alive(), 60)
        res.update(out)
        n0 = len(scenes)
        ed.v_mode.set("inspect")
        ed.on_mode()
        root.update()
        ed.stop_flag = False
        th = threading.Thread(target=lambda: ed._inspect(0.05), daemon=True)
        th.start()
        wait(lambda: not th.is_alive(), 120)
        res["inspect_scenes"] = scenes[n0:]
        ed.v_mode.set("edit")
        ed.on_mode()
        # hand guiding outside the zone
        pipe.teach_mode = True
        res["jump"] = pipe.monitor.send({"move": {"id": 999, "group": "arm", "joints": J, "positions": Q_HIGH,
                                                  "vel": 1.0, "acc": 1.0}})
        wait(lambda: "手动拖动越界" in logs(root), 20)
        root.update()
        res["now"] = ed.v_now.get()
        pipe.teach_mode = False
        res["logs"] = logs(root)
        pipe.monitor.send({"quit": True})
        root.after(300, root.destroy)
    root.after(1200, go)


def patched(self, n=0):
    driver(self)
    _orig(self, n)


tk.Tk.mainloop = patched
C.run_console("real")
L = res["logs"]
assert res["rule0"] == "A" and "没有包住 NERO 底座" in res["head0"], res["head0"]
assert not res["r_before"]["ok"] and "越墙" in res["r_before"]["error"] and "link2" in res["r_before"]["error"], res["r_before"]
assert "没有包住" not in res["head1"] and "已把底座并入安全区" in L
walls = [i for i in res["scene_after_save"] if "wall_" in i]
assert len(walls) == 6 and "v315_table" in res["scene_after_save"] and len(res["yaws_save"]) == 6 and all(y is not None for y in res["yaws_save"]), res["scene_after_save"]
assert not res["high"]["ok"] and "越墙" in res["high"]["error"], res["high"]
assert res["mid"]["ok"], res["mid"]
sc = res["inspect_scenes"]
assert len(sc) >= 2 and not any("wall_" in i for i in sc[0]) and sum("wall_" in i for i in sc[-1]) == 6, sc
assert "巡检期间 MoveIt 围墙暂时撤下" in L and "MoveIt 围墙已恢复" in L
assert "手动拖动越界" in L and "手动拖动越界" in res["now"], res["now"][:300]
print("[GUI23] V3.22 zone opens as rule A (red header, motion refused: link2 越墙); 并入底座 + 保存 -> 6 walls with yaw "
      "synced to MoveIt; high pose outside refused (越墙), middle pose executes; inspect retracts + restores walls; "
      "hand guiding outside -> alarm")
