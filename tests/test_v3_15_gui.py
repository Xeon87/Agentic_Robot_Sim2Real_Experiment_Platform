#!/usr/bin/env python3
"""GUI test for V3.15 一致性测试 / 工作区安全 (needs tkinter + a display, e.g. xvfb-run). Skips otherwise.

The console's 实机 tab gets tests/v315_fake_monitor.py as its ROS monitor (TF = FK with J3 turning the wrong way) and
all six steps green. Checks: URDF comparison, saved scene auto-applied at step 6, live FK/TF row, direction wizard
(J3 flagged), repeatability at two speeds, stop test, plan-only button, fence + MoveIt auto test, workspace save.
NERO_V315_SHOT=<prefix> saves <prefix>_cons.png / <prefix>_safe.png.
"""
import json, os, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog
except Exception:
    print("[GUI15] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI15] no DISPLAY -> skipped"); sys.exit(0)
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

TMP = tempfile.mkdtemp(prefix="v315gui_")
SE.STATE_PATH = os.path.join(TMP, "state.json")             # V3.19: never touch the real last-zone state
CS.WORKSPACE_PATH = os.path.join(TMP, "real_workspace.json")
CS.LOG_DIR = os.path.join(TMP, "logs")
asked = []
messagebox.askyesno = lambda t, x, **k: (asked.append((t, x)) or True)
cap = {}
for cls in (CS.ConsistencyTabUI, SE.SafetyEditorUI):
    def mk(orig, key):
        def init(self, *a, **k):
            orig(self, *a, **k)
            cap[key] = self
        return init
    cls.__init__ = mk(cls.__init__, cls.__name__)
res = {}
_orig = tk.Tk.mainloop


def logs(root):
    def walk(w):
        yield w
        for c in w.winfo_children():
            yield from walk(c)
    return "\n".join(w.get("1.0", "end") for w in walk(root) if w.winfo_class() == "Text")


def driver(root):
    cons, safe = cap["ConsistencyTabUI"], cap["SafetyEditorUI"]
    pipe, jp = cons.pipe, cons.jpanel

    def wait(cond, then, t_max=40):
        t0 = time.time()

        def chk():
            if cond():
                then()
            elif time.time() - t0 > t_max:
                res["fail"] = f"timeout waiting before {then.__name__}\n" + logs(root)[-2500:]
                root.destroy()
            else:
                root.after(100, chk)
        root.after(100, chk)

    def nb_select(text):
        def walk(w):
            yield w
            for c in w.winfo_children():
                yield from walk(c)
        for w in walk(root):
            if w.winfo_class() == "TNotebook":
                for t in w.tabs():
                    if w.tab(t, "text") == text:
                        w.select(t)

    def shot(name):
        pref = os.environ.get("NERO_V315_SHOT")
        if pref:
            root.update()
            time.sleep(0.4)
            root.update()
            subprocess.run(["import", "-window", "root", f"{pref}_{name}.png"], check=False)

    def s0():
        res["head0"] = safe.v_head.get()
        pipe.monitor.start([sys.executable, "-u", os.path.join(HERE, "tests", "v315_fake_monitor.py")],
                           env=dict(os.environ, FAKE_FLIP_JOINT="3", FAKE_RAD_S="20", FAKE_Q0="0,0.5,0,-0.6,0,-0.27,0"),
                           stdin_pipe=True)
        for k in RR.STEP_KEYS:
            pipe.state[k] = (RR.GREEN, "fake")
        wait(lambda: jp.enabled and "参数一致" in logs(root) and "规划场景: 添加 ['v315_table']" in logs(root)
             and "驱动 URDF vs TF" in cons.v_live.get(), s1)

    def s1():
        res["live"] = cons.v_live.get()
        res["model_txt"] = cons.v_model.get()
        cons.v_dir_j.set("全部 J1–J7")
        cons.on_direction()
        wait(lambda: "[V3.15 direction] 完成" in logs(root), s2, 90)

    def s2():
        root.update()
        res["dir_rows"] = [cons.tv_dir.item(i, "values") for i in cons.tv_dir.get_children()]
        cons.v_rep_n.set("2")
        cons.v_rep_sp.set("10,20")
        cons.on_repeat()
        wait(lambda: "[V3.15 repeat] 结果" in logs(root), s3, 90)

    def s3():
        nb_select("一致性测试")
        shot("cons")
        cons.v_st_j.set("J1")
        cons.v_st_s.set("5")
        cons.v_st_t.set("0.2")
        cons.on_stop_test()
        wait(lambda: "[V3.15 stop]" in logs(root) and not (cons.job and cons.job.is_alive()), s4, 60)

    def s4():
        jp.sync_to_actual()
        jp.on_plan_only()
        wait(lambda: "只规划成功" in logs(root), s5)

    def s5():
        safe.on_test_auto()
        wait(lambda: "[V3.16 safety] 自动越界测试" in logs(root), s6, 60)

    def s6():
        safe.on_save()
        root.update()
        res["head1"] = safe.v_head.get()
        res["now"] = safe.v_now.get()
        nb_select("工作区安全")
        shot("safe")
        res["logs"] = logs(root)
        res["asked"] = list(asked)
        pipe.monitor.send({"quit": True})
        root.after(300, root.destroy)
    root.after(1200, s0)


def patched(self, n=0):
    driver(self)
    _orig(self, n)


tk.Tk.mainloop = patched
C.run_console("real")
assert "fail" not in res, res.get("fail")
if os.environ.get("GUI15_DEBUG"):
    print("\n".join(l for l in res["logs"].splitlines() if "direction" in l or "FENCE" in l or "FAIL" in l)[:4000])
L = res["logs"]
assert "未测量" in res["head0"] and "已保存" in res["head1"], (res["head0"], res["head1"])
assert os.path.exists(CS.WORKSPACE_PATH) and json.load(open(CS.WORKSPACE_PATH))["measured"]
assert "一致 ✓" in res["model_txt"] and "仿真 vs 驱动 URDF: 0.00 mm ✓" in res["live"], res["live"]
rows = {r[0]: r for r in res["dir_rows"]}
assert len(rows) == 7 and rows["J3"][5] == "不符" and rows["J1"][5] == "通过" and rows["J2"][5] == "通过", rows
assert rows["J7"][5] == "—"
assert "[WARN] [V3.15 direction] J3" in L and "[WARN] [V3.15 direction] 完成 7 个关节" in L
assert L.count("[OK] [V3.15 repeat] @") == 2 and "@20% ×2" in L
assert "[OK] [V3.15 stop] J1: cancel 时" in L and "PREEMPTED" in L, L[-3000:]
assert "只规划成功" in L and "路径围栏通过" in L
assert "(a) 围栏：拒绝 ✓" in L and "(b) MoveIt 场景（只规划）：拒绝 ✓ GOAL_IN_COLLISION" in L, L[-2000:]
assert "当前姿态在安全区内" in res["now"], res["now"]
assert any(t == "关节方向测试" for t, _x in res["asked"]) and any(t == "J3 转向确认" for t, _x in res["asked"])
logs_dir = os.listdir(CS.LOG_DIR)
assert any(f.startswith("v315_direction_") for f in logs_dir) and any(f.startswith("v315_repeat_") and f.endswith(".csv") for f in logs_dir)
assert any(f.startswith("v315_stop_") for f in logs_dir) and any(f.startswith("v315_safety_test_") for f in logs_dir)
mlog = os.path.join(CS.LOG_DIR, "real_motion_log.csv")
assert os.path.exists(mlog) and "方向测试 J3" in open(mlog, encoding="utf-8").read()
print("[GUI15] URDF compare ✓, scene auto-applied, live FK/TF, direction wizard flags flipped J3, repeatability @10/20%, "
      "stop test, plan-only, fence + MoveIt auto test refuse, workspace saved (measured)")
