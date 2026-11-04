#!/usr/bin/env python3
"""GUI test for the V3.14 joint panel (needs tkinter + a display, e.g. xvfb-run). Skips otherwise.

Console on the 实机 tab; a fake ROS monitor process (joint_states + MoveGroup results) is attached to the pipeline and
steps are marked green. Checks: panel locked before / enabled after, targets synced to actual, out-of-range entry
clamped, 执行 -> confirm -> move -> final error logged, small-step test (out + back), jog mode, pose save, gripper
locked while CAN shows the old id 517.  NERO_V314_SHOT=<png> saves a screenshot.
"""
import json, math, os, subprocess, sys, tempfile, textwrap, time
HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
try:
    import tkinter as tk
    from tkinter import messagebox, simpledialog
except Exception:
    print("[GUI14] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI14] no DISPLAY -> skipped"); sys.exit(0)
import NERO_V312_Console as C
import NERO_V316_Safety_Editor as SE
import NERO_V314_Joint_Control as JC
import NERO_V313_Real_Robot as RR

TMP = tempfile.mkdtemp(prefix="v314gui_")
SE.STATE_PATH = os.path.join(TMP, "state.json")             # V3.19: never touch the real last-zone state
JC.POSES_PATH = os.path.join(TMP, "real_poses.json")
JC.PoseLibrary.__init__.__defaults__ = (JC.POSES_PATH,)
asked = []
messagebox.askyesno = lambda t, x, **k: (asked.append((t, x)) or True)
simpledialog.askstring = lambda *a, **k: "测试姿态"
FAKE_MON = os.path.join(TMP, "fake_monitor.py")
open(FAKE_MON, "w").write(textwrap.dedent("""
    import json, sys, threading, time
    stop = threading.Event()
    pos = [0.2, -0.6, 0.0, 1.5, 0.03, 0.35, -0.09]
    def rd():
        for line in sys.stdin:
            c = json.loads(line)
            if c.get("quit"): stop.set(); return
            if c.get("move"):
                mv = c["move"]
                print(json.dumps({"event": "move_accepted", "id": mv["id"]}), flush=True)
                time.sleep(0.4)
                for n, p in zip(mv["joints"], mv["positions"]):
                    if n.startswith("joint") and int(n[5:]) <= 7:
                        pos[int(n[5:]) - 1] = p
                print(json.dumps({"event": "move_result", "id": mv["id"], "ok": True, "error_code": 1,
                                  "planning_time": 0.08, "elapsed_s": 0.4}), flush=True)
    threading.Thread(target=rd, daemon=True).start()
    n = 0
    while not stop.is_set():
        print(json.dumps({"event": "js", "joints": {"names": ["joint%d" % i for i in range(1, 8)], "position": pos}}),
              flush=True)
        n += 1
        if n % 5 == 0:
            print(json.dumps({"event": "status", "js_hz": 200.0, "joints": None, "has_move_group": True,
                              "has_move_action": True, "services": {"/emergency_stop": "std_srvs/srv/Trigger"},
                              "gripper_status": None, "arm_status": {"ctrl_mode": 1},
                              "tcp": {"xyz": [0.412, 0.087, 0.265], "rpy_deg": [179.2, 2.1, 11.8],
                                      "base": "base_link", "tip": "link7"}}), flush=True)
        stop.wait(0.1)
"""))
MODEL = JC.build_model(open(os.path.join(HERE, "config", "nero.urdf")).read(),
                       '<robot name="n"><group name="arm"><chain base_link="base_link" tip_link="link7"/></group>'
                       '<group_state name="home" group="arm">' + "".join(f'<joint name="joint{i}" value="0"/>' for i in range(1, 8))
                       + '</group_state></robot>', None, source="test")

cap = {}
_init = JC.JointPanelUI.__init__


def init(self, *a, **k):
    _init(self, *a, **k)
    cap["p"] = self


JC.JointPanelUI.__init__ = init
res = {}
_orig = tk.Tk.mainloop


def logs(root):
    def walk(w):
        yield w
        for c in w.winfo_children():
            yield from walk(c)
    return "\n".join(w.get("1.0", "end") for w in walk(root) if w.winfo_class() == "Text")


def driver(root):
    P = cap["p"]
    pipe = P.pipe
    seq = []

    def wait(cond, then, t_max=20):
        t0 = time.time()

        def chk():
            if cond():
                then()
            elif time.time() - t0 > t_max:
                res["fail"] = f"timeout waiting: {then.__name__} job={P.job} alive={P.job and P.job.is_alive()} ms={pipe.move_state} asked={asked} en={P.enabled} st={P.v_state.get()}\n" + logs(root)[-1500:]
                root.destroy()
            else:
                root.after(100, chk)
        root.after(100, chk)

    def s0():
        res["locked_before"] = (not P.enabled, P.v_state.get())
        pipe.can.feed = pipe.can.feed
        pipe.can.total = 1
        pipe.can.snapshot = lambda now=None: {"fps": 1000, "total": 1, "error_frames": 0, "groups": {},
                                              "gripper_old_id": ["517"]}
        pipe.safety_fn = None                  # V3.16: this test is about the panel, not the fence (see test_v3_16_gui)
        pipe.monitor.start([sys.executable, "-u", FAKE_MON], stdin_pipe=True)
        for k in RR.STEP_KEYS:
            pipe.state[k] = (RR.GREEN, "fake")
        pipe.model = MODEL
        wait(lambda: P.enabled, s1)

    def s1():
        res["synced"] = [round(P.cols[i]["var"].get(), 1) for i in range(7)]
        P.cols[5]["evar"].set("80")                      # J6 upper 54.4 -> clamp to 51.4
        P._on_entry(5)
        res["clamped"] = P.cols[5]["var"].get()
        P.cols[3]["evar"].set("100")
        P._on_entry(3)
        P.cols[5]["evar"].set("25")
        P._on_entry(5)
        root.update()
        res["preview"] = P.v_pv.get()
        P.on_execute()
        wait(lambda: "[V3.14 joints] 执行: 规划" in logs(root), s2)

    def s2():
        res["asked_exec"] = list(asked)
        P.v_test_j.set("J3")
        P.on_step_test()
        wait(lambda: "小步测试 J3: 通过" in logs(root) or "小步测试 J3: 有误差" in logs(root), s3)

    def s3():
        asked.clear()
        P.v_jog.set(True)
        P._on_jog()
        P._on_step(0, 1)
        wait(lambda: "点动: 规划" in logs(root), s4)

    def s4():
        res["jog_asked"] = list(asked)
        P.v_jog.set(False)
        P.on_pose_save()
        res["poses"] = P.poses.names()
        root.update()
        res["logs"] = logs(root)
        res["grip_enabled"] = P.grip_enabled
        res["grip_warn"] = P.cols[7]["warn"].cget("text")
        res["tcp"] = P.v_tcp.get()
        shot = os.environ.get("NERO_V314_SHOT")
        if shot:
            P.cols[3]["evar"].set("110.5")
            P._on_entry(3)
            root.update()
            time.sleep(0.5)
            root.update()
            subprocess.run(["import", "-window", "root", shot], check=False)
        pipe.monitor.send({"quit": True})
        root.after(300, root.destroy)
    root.after(1200, s0)


def patched(self, n=0):
    driver(self)
    _orig(self, n)


tk.Tk.mainloop = patched
C.run_console("real")
assert "fail" not in res, res.get("fail")
L = res["logs"]
assert res["locked_before"][0] and "不可操作" in res["locked_before"][1], res["locked_before"]
assert res["synced"] == [round(math.degrees(v), 1) for v in (0.2, -0.6, 0.0, 1.5, 0.03, 0.35, -0.09)], res["synced"]
hs6 = JC.soft_range_deg(-0.73, 0.95)[1]
assert abs(res["clamped"] - round(hs6, 1)) < 0.06 and "超出可设范围" in L
assert "最大 Δ" in res["preview"]
assert any("第一次实机运动" in x for _t, x in res["asked_exec"]), res["asked_exec"]
assert "[OK] [V3.14 joints] 执行: 规划 0.08 s" in L and "最终误差 max" in L, L[-2000:]
assert "小步测试 J3 +5°: 规划" in L and "小步测试 J3 回原位: 规划" in L and "小步测试 J3: 通过" in L
assert any(t == "点动模式" for t, _x in res["jog_asked"]) and "点动: 规划" in L
assert "测试姿态 (用户)" in res["poses"] and "home (SRDF)" in res["poses"]
assert not res["grip_enabled"] and "517" in res["grip_warn"]
assert "0.412" in res["tcp"] and "179.2" in res["tcp"]
assert "[OK] [V3.14 joints] 限位来源 URDF" in L
print("[GUI14] joint panel: locked -> enabled, synced to actual, clamp, first-move confirm, execute + final error, "
      "J3 step test out/back, jog, pose save, gripper locked on 517, TCP")
