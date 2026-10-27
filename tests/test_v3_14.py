#!/usr/bin/env python3
"""V3.14 joint panel tests (no ROS, no Tk, no robot).

J1  robot model: URDF limits + SRDF arm group (chain) + gripper group + named states; URDF-only; fallback
J2  numbers: soft limits, entry parsing, clamping, near-limit, preview confirmation rule, step-test targets, final error
J3  enable rules for the joint panel and the gripper (old CAN id 517 -> locked)
J4  pose library: built-ins + SRDF states + user poses saved / reloaded / deleted (config/real_poses.json format)
J5  ROS monitor helper against a fake rclpy + fake moveit_msgs / tf2_ros: model event, 10 Hz js events, MoveGroup
    goal content (group, joint constraints, scaling), success, INVALID_GROUP_NAME hint, cancel, TCP from TF
J6  RealPipeline.move_joints via a fake monitor process: ok, limit refusal, not-green refusal, busy refusal,
    timeout -> cancel, gripper without group refused, stop_all releases a waiting move
J7  console wiring
"""
import json, math, os, shutil, signal, subprocess, sys, tempfile, textwrap, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V314_Joint_Control as JC  # noqa: E402
import NERO_V313_Real_Robot as RR  # noqa: E402

assert RR.VERSION.startswith(("V3.14", "V3.15", "V3.16", "V3.17", "V3.18", "V3.19", "V3.20", "V3.21", "V3.22", "V3.23", "V3.24", "V3.25", "V3.26", "V3.27", "V3.28", "V3.29", "V3.30", "V3.31", "V3.31.1", "V3.32", "V3.33", "V3.34", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")) and "tkinter" not in sys.modules and "rclpy" not in sys.modules
TMP = Path(tempfile.mkdtemp(prefix="v314_"))

# ---------------------------------------------------------------- J1
URDF = (ROOT / "config" / "nero.urdf").read_text()
GRIP_URDF = URDF.replace("</robot>", """
<link name="gripper_base"/><joint name="grip_mount" type="fixed"><parent link="link7"/><child link="gripper_base"/></joint>
<link name="finger1"/><joint name="joint8" type="prismatic"><parent link="gripper_base"/><child link="finger1"/>
<limit lower="0" upper="0.05" effort="10" velocity="1"/></joint>
<link name="finger2"/><joint name="joint9" type="prismatic"><parent link="gripper_base"/><child link="finger2"/>
<limit lower="-0.05" upper="0" effort="10" velocity="1"/><mimic joint="joint8" multiplier="-1"/></joint>
</robot>""")
SRDF = """<robot name="nero">
<group name="arm"><chain base_link="base_link" tip_link="link7"/></group>
<group name="gripper"><joint name="joint8"/><joint name="joint9"/></group>
<group_state name="home" group="arm"><joint name="joint1" value="0"/><joint name="joint2" value="0.3"/>
<joint name="joint3" value="0"/><joint name="joint4" value="1.2"/><joint name="joint5" value="0"/>
<joint name="joint6" value="0.2"/><joint name="joint7" value="0"/></group_state>
<group_state name="open" group="gripper"><joint name="joint8" value="0.05"/></group_state>
</robot>"""
u = JC.parse_urdf(URDF)
assert u["root"] == "world" and u["joints"]["joint4"]["lower"] == -1.01 and u["joints"]["joint4"]["upper"] == 2.14
assert JC.chain_joints(u, "base_link", "link7") == [f"joint{i}" for i in range(1, 8)]
m = JC.build_model(GRIP_URDF, SRDF, [f"joint{i}" for i in range(1, 10)], source="test")
assert m["arm_group"] == "arm" and m["joints"] == [f"joint{i}" for i in range(1, 8)] and m["limit_source"] == "URDF"
assert m["limits"][3] == [-1.01, 2.14] and m["limits"][6] == [-1.5707963, 1.5707963]
assert m["gripper"] == {"group": "gripper", "joints": ["joint8"], "factor": 2.0, "range_mm": (0.0, 100.0)}, m["gripper"]
assert m["states"] == {"home": [0.0, 0.3, 0.0, 1.2, 0.0, 0.2, 0.0]} and m["base_link"] == "base_link" and m["tip_link"] == "link7"
m2 = JC.build_model(URDF, None, None)
assert m2["joints"] == [f"joint{i}" for i in range(1, 8)] and m2["arm_group"] == "arm" and m2["gripper"] is None
assert any("SRDF not available" in w for w in m2["warnings"]) and m2["tip_link"] == "tcp_link"
fb = JC.fallback_model()
assert fb["limit_source"] == "仿真表" and fb["limits"][5] == [-0.73, 0.95] and not fb["has_urdf"]
bad = JC.build_model("<robot", None, None)
assert bad["warnings"] and len(bad["joints"]) == 7
json.dumps(m)
print("[J1] model: URDF limits + SRDF arm chain + gripper group (2 fingers -> width) + home state; URDF-only; fallback")

# ---------------------------------------------------------------- J2
ls, hs = JC.soft_range_deg(-2.70526, 2.70526)
assert abs(hs - (math.degrees(2.70526) - 3.0)) < 1e-9 and abs(ls + hs) < 1e-9
assert [JC.parse_entry(x) for x in ("12.5", " -3", "+4°", "12,5", "abc", "", "1e3", "nan", "－7")] == \
       [12.5, -3.0, 4.0, 12.5, None, None, None, None, -7.0]
assert JC.clamp_deg(200, ls, hs) == (hs, True) and JC.clamp_deg(10, ls, hs) == (10, False)
assert JC.near_limit(52.0, -0.73, 0.95)[0] == "upper" and JC.near_limit(0.0, -0.73, 0.95) is None
pv = JC.preview([12, -35, 0, 88.6, 2, 20, -5], [12, -20, 0, 110.5, 2, 48, -5])
assert abs(pv["max"] - 28.0) < 1e-9 and pv["max_joint"] == 5 and not pv["confirm"] and abs(pv["total"] - 64.9) < 1e-9
pv = JC.preview([0] * 7, [31, 0, 0, 0, 0, 0, 0])
assert pv["confirm"] and "J1" in pv["reason"]
pv = JC.preview([0] * 7, [25, 25, 25, 25, 0, 0, 0])
assert pv["confirm"] and "总变化" in pv["reason"]
lims = fb["limits"]
out, back, s = JC.step_test_targets([0] * 7, 2, lims)
assert s == 5.0 and out[2] == 5.0 and back == [0] * 7
hi6 = JC.soft_range_deg(*lims[5])[1]
out, back, s = JC.step_test_targets([0, 0, 0, 0, 0, hi6 - 1, 0], 5, lims)
assert s == -5.0 and abs(out[5] - (hi6 - 6)) < 1e-9                 # near the upper limit -> goes down instead
assert JC.final_error([1, 2, 3], [1, 2.5, 3]) == (0.5, 1)
assert JC.moveit_error_text(-15).startswith("INVALID_GROUP_NAME（") and JC.moveit_error_text(1) == "SUCCESS"
r, p_, y = JC.quat_to_rpy_deg(0, 0, math.sin(math.pi / 8), math.cos(math.pi / 8))
assert abs(y - 45) < 1e-6 and abs(r) < 1e-9
print("[J2] soft limits 3 deg, entry parse/clamp, near-limit, confirm rule (>30 deg / total >90 deg), step test flips at limit")

# ---------------------------------------------------------------- J3
ok = JC.panel_enable(False, "green", m, 0.1, False, True)
assert ok == (True, "")
assert JC.panel_enable(True, "green", m, 0.1, False, True)[1].startswith("仿真")
assert "第 6 步" in JC.panel_enable(False, "red", m, 0.1, False, True)[1]
assert "超时" in JC.panel_enable(False, "green", m, 3.0, False, True)[1]
assert "超时" in JC.panel_enable(False, "green", m, None, False, True)[1]
assert "执行" in JC.panel_enable(False, "green", m, 0.1, True, True)[1]
assert "缺少" in JC.panel_enable(False, "green", m, 0.1, False, False)[1]
cfg = {"effector_type": "agx_gripper"}
assert JC.gripper_enable(cfg, ["517"], m, {}, True) == (False, "夹爪 ID 仍为 517（旧 ID）")
assert "effector=none" in JC.gripper_enable({"effector_type": "none"}, [], m, {}, True)[1]
assert "planning group" in JC.gripper_enable(cfg, [], m2, {}, True)[1]
assert "gripper_status" in JC.gripper_enable(cfg, [], m, None, True)[1]
assert JC.gripper_enable(cfg, [], m, {"width": 0.03}, True) == (True, "")
assert abs(JC.extract_gripper_mm({"width": 0.03}, None, None) - 30.0) < 1e-9
assert abs(JC.extract_gripper_mm(None, {"names": ["joint8"], "position": [0.02]}, m["gripper"]) - 40.0) < 1e-9
assert abs(JC.gripper_mm_to_joint(60, m["gripper"]) - 0.03) < 1e-12
print("[J3] panel enable rules (sim / step 6 / feedback age / moving / joints); gripper locked on old id 517 / none / no group")

# ---------------------------------------------------------------- J4
pp = TMP / "cfg" / "real_poses.json"
lib = JC.PoseLibrary(str(pp))
assert lib.names() == ["零位 (全 0°)"] and lib.get("零位 (全 0°)") == [0.0] * 7
lib.set_model_states(m)
assert "home (SRDF)" in lib.names() and abs(lib.get("home (SRDF)")[3] - math.degrees(1.2)) < 0.01
lib.save("取蛋上方", [1, 2, 3, 4, 5, 6, 7.123])
d = json.loads(pp.read_text())
assert d["unit"] == "deg" and d["poses"]["取蛋上方"]["joints_deg"][6] == 7.12
lib2 = JC.PoseLibrary(str(pp))
assert lib2.get("取蛋上方 (用户)") == [1, 2, 3, 4, 5, 6, 7.12]
assert not lib2.delete("零位 (全 0°)") and lib2.delete("取蛋上方 (用户)") and lib2.names() == ["零位 (全 0°)"]
try:
    lib2.save(" ", [0] * 7)
    raise AssertionError("empty name accepted")
except ValueError:
    pass
print("[J4] pose library: zero + SRDF states + user poses saved/reloaded/deleted; built-ins protected")

# ---------------------------------------------------------------- J5 monitor helper with fake ROS
FR = TMP / "fakeros"
for d_ in ("rclpy", "rosidl_runtime_py", "std_msgs", "moveit_msgs", "tf2_ros"):
    (FR / d_).mkdir(parents=True, exist_ok=True)
GOALS = TMP / "goals.jsonl"
(FR / "rclpy" / "__init__.py").write_text(textwrap.dedent("""
    import time as _time
    _ok = [True]
    def init(): pass
    def ok(): return _ok[0]
    def shutdown(): _ok[0] = False
    def spin_once(node, timeout_sec=0.0):
        node._tick(); _time.sleep(timeout_sec)
    def spin_until_future_complete(node, fut, timeout_sec=0.0): fut._done = True
"""))
(FR / "rclpy" / "time.py").write_text("class Time:\n    pass\n")
(FR / "rclpy" / "qos.py").write_text(textwrap.dedent("""
    class QoSProfile:
        def __init__(self, **kw): self.kw = kw
    class ReliabilityPolicy: BEST_EFFORT = 1; RELIABLE = 2
    class HistoryPolicy: KEEP_LAST = 1
    class DurabilityPolicy: TRANSIENT_LOCAL = 1
"""))
(FR / "rclpy" / "node.py").write_text(textwrap.dedent("""
    import os
    class _Msg:
        def __init__(self, **kw): self.__dict__.update(kw)
    class Node:
        def __init__(self, name): self.subs = []
        def create_subscription(self, mt, topic, cb, qos): self.subs.append((topic, cb)); return object()
        def get_topic_names_and_types(self):
            return [("/feedback/joint_states", ["sensor_msgs/msg/JointState"]), ("/feedback/arm_status", ["agx/msg/ArmStatus"])]
        def get_node_names_and_namespaces(self): return [("move_group", "/")]
        def get_service_names_and_types(self): return [("/emergency_stop", ["std_srvs/srv/Trigger"])]
        def create_client(self, st, name): raise RuntimeError("no clients in this fake")
        def destroy_client(self, c): pass
        def destroy_node(self): pass
        def _tick(self):
            for f in list(getattr(self, "_pending", [])):
                f._fire()
            for topic, cb in self.subs:
                if topic.endswith("joint_states"):
                    cb(_Msg(name=["joint%d" % i for i in range(1, 8)], position=[0.1] * 7))
                elif topic == "/robot_description":
                    cb(_Msg(data=open(os.environ["FAKE_URDF"]).read()))
                elif topic == "/robot_description_semantic":
                    cb(_Msg(data=open(os.environ["FAKE_SRDF"]).read()))
                else:
                    cb(_Msg(ctrl_mode=1))
"""))
(FR / "rclpy" / "action.py").write_text(textwrap.dedent("""
    import json, os
    class _Fut:
        def __init__(self, node, value, hold=False):
            self.node, self.value, self.hold, self.cbs = node, value, hold, []
            node._pending = getattr(node, "_pending", []) + [self]
        def add_done_callback(self, cb): self.cbs.append(cb)
        def result(self): return self.value
        def _fire(self):
            if self.hold: return
            self.node._pending.remove(self)
            for cb in self.cbs: cb(self)
    class _Obj:
        def __init__(self, **kw): self.__dict__.update(kw)
    class _GH:
        def __init__(self, node, code, hold):
            self.node, self.code, self.accepted, self.hold = node, code, True, hold
        def get_result_async(self):
            self.rf = _Fut(self.node, None, hold=self.hold)
            self.rf.value = _Obj(status=4, result=_Obj(error_code=_Obj(val=self.code), planning_time=0.12,
                       planned_trajectory=_Obj(joint_trajectory=_Obj(points=[_Obj(time_from_start=_Obj(sec=2, nanosec=5e8))]))))
            return self.rf
        def cancel_goal_async(self):
            self.rf.value.result.error_code.val = -7
            self.rf.hold = False
    class ActionClient:
        def __init__(self, node, t, name): self.node, self.name = node, name
        def wait_for_server(self, timeout_sec=0): return True
        def send_goal_async(self, goal, feedback_callback=None):
            r = goal.request
            with open(os.environ["FAKE_GOALS"], "a") as f:
                f.write(json.dumps({"action": self.name, "group": r.group_name, "vel": r.max_velocity_scaling_factor,
                    "acc": r.max_acceleration_scaling_factor, "diff": r.start_state.is_diff, "plan_only": goal.planning_options.plan_only,
                    "jc": [[c.joint_name, c.position, c.tolerance_above, c.tolerance_below, c.weight]
                           for c in r.goal_constraints[0].joint_constraints]}) + "\\n")
            if feedback_callback:
                feedback_callback(_Obj(feedback=_Obj(state="PLANNING")))
            code = int(os.environ.get("FAKE_MOVE_CODE", "1"))
            return _Fut(self.node, _GH(self.node, code, hold=os.environ.get("FAKE_MOVE_HOLD") == "1"))
    def get_action_names_and_types(node): return [("/move_action", ["moveit_msgs/action/MoveGroup"])]
"""))
(FR / "std_msgs" / "__init__.py").write_text("")
(FR / "std_msgs" / "msg.py").write_text("class String:\n    pass\n")
(FR / "moveit_msgs" / "__init__.py").write_text("")
(FR / "moveit_msgs" / "msg.py").write_text(textwrap.dedent("""
    class JointConstraint:
        joint_name = ""; position = 0.0; tolerance_above = 0.0; tolerance_below = 0.0; weight = 0.0
    class Constraints:
        def __init__(self): self.joint_constraints = []
"""))
(FR / "moveit_msgs" / "action.py").write_text(textwrap.dedent("""
    class _O:
        pass
    class MoveGroup:
        class Goal:
            def __init__(self):
                self.request = _O(); self.request.start_state = _O(); self.request.start_state.is_diff = False
                self.request.goal_constraints = []
                self.planning_options = _O()
"""))
(FR / "tf2_ros" / "__init__.py").write_text(textwrap.dedent("""
    class _O:
        def __init__(self, **kw): self.__dict__.update(kw)
    class Buffer:
        def lookup_transform(self, base, tip, t):
            return _O(transform=_O(translation=_O(x=0.412, y=0.087, z=0.265), rotation=_O(x=0.0, y=0.0, z=0.0, w=1.0)))
    class TransformListener:
        def __init__(self, buf, node): pass
"""))
(FR / "rosidl_runtime_py" / "__init__.py").write_text("")
(FR / "rosidl_runtime_py" / "utilities.py").write_text("def get_message(t): return t\ndef get_service(t): return t\n")
(FR / "rosidl_runtime_py" / "convert.py").write_text("def message_to_ordereddict(m): return dict(m.__dict__)\n")
(TMP / "u.urdf").write_text(GRIP_URDF)
(TMP / "s.srdf").write_text(SRDF)


def run_monitor(script, env_extra=None, timeout=15):
    env = dict(os.environ, PYTHONPATH=str(FR), FAKE_URDF=str(TMP / "u.urdf"), FAKE_SRDF=str(TMP / "s.srdf"),
               FAKE_GOALS=str(GOALS), **(env_extra or {}))
    p = subprocess.Popen(RR.helper_argv("monitor", "--period", "0.3"), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, env=env)
    got, t_end = [], time.time() + timeout
    while time.time() < t_end:
        ln = p.stdout.readline()
        if not ln:
            break
        ev = json.loads(ln)
        got.append(ev)
        for cmd in script(ev, got) or []:
            p.stdin.write(json.dumps(cmd) + "\n")
            p.stdin.flush()
        if ev["event"] == "monitor_stopped":
            break
    p.wait(timeout=5)
    return got


def scr_ok(ev, got):
    if ev["event"] == "model" and not any(e["event"] == "move_result" for e in got) and \
            sum(e["event"] == "model" for e in got) == 1:
        return [{"move": {"id": 5, "group": "arm", "joints": [f"joint{i}" for i in range(1, 8)],
                          "positions": [0.1, 0.2, 0, 0, 0, 0, 0], "vel": 0.1, "acc": 0.1}}]
    if ev["event"] == "move_result" and ev["id"] == 5:
        return [{"move": {"id": 6, "group": "bad_group", "joints": ["joint1"], "positions": [0.0]}}]
    if ev["event"] == "move_result" and ev["id"] == 6:
        return [{"quit": True}]
    return []


GOALS.write_text("")
got = run_monitor(scr_ok)
models = [e for e in got if e["event"] == "model"]
assert models and models[0]["model"]["arm_group"] == "arm" and models[0]["model"]["gripper"]["group"] == "gripper"
assert sum(e["event"] == "js" for e in got) >= 5
res = {e["id"]: e for e in got if e["event"] == "move_result"}
assert res[5]["ok"] and res[5]["error_code"] == 1 and res[5]["trajectory_s"] == 2.5 and res[5]["planning_time"] == 0.12
os.environ.pop("FAKE_MOVE_CODE", None)
assert any(e["event"] == "move_feedback" and e["phase"] == "PLANNING" for e in got)
goals = [json.loads(x) for x in GOALS.read_text().splitlines()]
g0 = goals[0]
assert g0["action"] == "/move_action" and g0["group"] == "arm" and g0["vel"] == 0.1 and g0["diff"] and not g0["plan_only"]
assert [c[0] for c in g0["jc"]] == [f"joint{i}" for i in range(1, 8)] and g0["jc"][1][1] == 0.2 and g0["jc"][0][2] == 0.0017
stat = [e for e in got if e["event"] == "status" and e.get("tcp")][-1]
assert stat["tcp"]["xyz"] == [0.412, 0.087, 0.265] and stat["tcp"]["rpy_deg"] == [0.0, 0.0, 0.0] and stat["tcp"]["tip"] == "link7"
assert stat["has_urdf"] and stat["has_srdf"] and stat["model_known"]
got = run_monitor(lambda ev, g: ([{"move": {"id": 1, "group": "armx", "joints": ["joint1"], "positions": [0.0]}}]
                                 if ev["event"] == "model" and sum(e["event"] == "model" for e in g) == 1 else
                                 [{"quit": True}] if ev["event"] == "move_result" else []),
                  {"FAKE_MOVE_CODE": "-15"})
r15 = [e for e in got if e["event"] == "move_result"][0]
assert not r15["ok"] and r15["error_code"] == -15 and "INVALID_GROUP_NAME" in r15["error"] and "ros2 param get" in r15["error"]


def scr_cancel(ev, g):
    if ev["event"] == "model" and sum(e["event"] == "model" for e in g) == 1:
        return [{"move": {"id": 2, "group": "arm", "joints": ["joint1"], "positions": [0.0]}}]
    if ev["event"] == "move_accepted":
        return [{"move": {"id": 3, "group": "arm", "joints": ["joint1"], "positions": [0.0]}}, {"cancel": True}]
    if ev["event"] == "move_result" and ev["id"] == 2:
        return [{"cancel": True}, {"quit": True}]
    return []


got = run_monitor(scr_cancel, {"FAKE_MOVE_HOLD": "1"})
rs = {e["id"]: e for e in got if e["event"] == "move_result"}
assert not rs[3]["ok"] and "in progress" in rs[3]["error"]
assert not rs[2]["ok"] and rs[2]["error_code"] == -7 and "PREEMPTED" in rs[2]["error"]
cr = [e for e in got if e["event"] == "cancel_result"]
assert cr[0]["ok"] and not cr[-1]["ok"]                           # second cancel: no active goal
print("[J5] monitor: model event, 10 Hz js, MoveGroup goal (group/joints/tol/scaling/is_diff), success, -15 hint, "
      "busy refusal, cancel -> PREEMPTED, TCP from TF")

# ---------------------------------------------------------------- J6 pipeline with fake monitor process
FAKE_MON = TMP / "fake_monitor.py"
FAKE_MON.write_text(textwrap.dedent("""
    import json, os, sys, threading, time
    stop = threading.Event()
    pos = [0.0] * 7
    pending = {}
    def rd():
        for line in sys.stdin:
            c = json.loads(line)
            if c.get("quit"): stop.set(); return
            if c.get("move"):
                mv = c["move"]
                print(json.dumps({"event": "move_accepted", "id": mv["id"]}), flush=True)
                if os.environ.get("FAKE_HANG") == "1":
                    pending["id"] = mv["id"]; continue
                time.sleep(0.3)
                if mv["group"] == "arm":
                    for n, p in zip(mv["joints"], mv["positions"]):
                        pos[int(n[5:]) - 1] = p
                print(json.dumps({"event": "move_result", "id": mv["id"], "ok": True, "error_code": 1,
                                  "planning_time": 0.05, "elapsed_s": 0.3, "vel": mv["vel"]}), flush=True)
            if c.get("cancel"):
                print(json.dumps({"event": "cancel_result", "ok": True}), flush=True)
                if pending.get("id"):
                    print(json.dumps({"event": "move_result", "id": pending.pop("id"), "ok": False, "error_code": -7,
                                      "error": "PREEMPTED"}), flush=True)
    threading.Thread(target=rd, daemon=True).start()
    while not stop.is_set():
        print(json.dumps({"event": "js", "joints": {"names": ["joint%d" % i for i in range(1, 8)], "position": pos}}),
              flush=True)
        stop.wait(0.1)
"""))


def make_pipe(hang=False):
    pipe = RR.RealPipeline(lambda: {"can_port": "can0", "speed_percent": 10, "domain_id": 42},
                           can_opener=lambda p: (_ for _ in ()).throw(OSError("no can")))
    env = dict(os.environ, FAKE_HANG="1" if hang else "0")
    pipe.monitor.start([sys.executable, "-u", str(FAKE_MON)], env=env, stdin_pipe=True)
    return pipe


pipe = make_pipe()
time.sleep(0.4)
assert pipe.move_joints(m, [0.1] * 7, 0.1)["error"].startswith("step 6")         # not green -> refused
for k in RR.STEP_KEYS:
    pipe.state[k] = (RR.GREEN, "test")
assert "outside limit" in pipe.move_joints(m, [0, 0, 0, 0, 0, 1.2, 0], 0.1)["error"]  # J6 upper 0.95
res = pipe.move_joints(m, [0.1, 0.2, 0, 0.5, 0, 0, 0], 0.1, label="t1")
assert res["ok"] and res["vel"] == 0.1 and not pipe.move_busy(), res
time.sleep(0.3)
act = pipe.actual_arm_deg(m)
assert abs(act[3] - math.degrees(0.5)) < 1e-6 and abs(act[1] - math.degrees(0.2)) < 1e-6
th = threading.Thread(target=lambda: pipe.move_joints(m, [0.0] * 7, 0.2, label="t2"))
th.start()
time.sleep(0.05)
assert pipe.move_busy() and "in progress" in pipe.move_joints(m, [0.0] * 7, 0.2)["error"]
th.join(5)
assert pipe.move_state["ok"] and pipe.move_state["label"] == "t2"
assert "no gripper" in pipe.move_gripper(m2, 50, 0.1)["error"]
res = pipe.move_gripper(m, 60, 0.1)
assert res["ok"]
pipe.stop_all()
pipe.shutdown()
pipe = make_pipe(hang=True)
for k in RR.STEP_KEYS:
    pipe.state[k] = (RR.GREEN, "test")
time.sleep(0.3)
res = pipe._send_move("hang", "arm", m["joints"], [0.0] * 7, 0.1, timeout_s=1.0)
assert not res["ok"] and "cancel sent" in res["error"] and not pipe.move_busy()
th = threading.Thread(target=lambda: pipe.move_joints(m, [0.0] * 7, 0.1, timeout_s=30))
th.start()
time.sleep(0.3)
assert pipe.cancel_move()
th.join(5)
assert not th.is_alive() and pipe.move_state["result"]["error_code"] == -7
th = threading.Thread(target=lambda: pipe.move_joints(m, [0.0] * 7, 0.1, timeout_s=30))
th.start()
time.sleep(0.3)
pipe.stop_all()
th.join(5)
assert not th.is_alive() and not pipe.move_busy() and pipe.model is None
pipe.shutdown()
print("[J6] pipeline: move ok + feedback, limit / not-green / busy refused, gripper group check, timeout -> cancel, "
      "cancel -> PREEMPTED, stop_all releases waiting move")

# ---------------------------------------------------------------- J7 console wiring
C = (ROOT / "NERO_V312_Console.py").read_text()
assert "V3.14 JOINT PANEL" in C and "JC.JointPanelUI(" in C and ("jpanel.poll()" in C or ("ui_.poll()" in C and "(\"joints\", jpanel)" in C)) and any(f'VERSION = "{v}' in C for v in ("V3.14", "V3.15", "V3.16", "V3.17", "V3.18", "V3.19", "V3.20", "V3.21", "V3.22", "V3.23", "V3.24", "V3.25", "V3.26", "V3.27", "V3.28", "V3.29", "V3.30", "V3.31", "V3.32", "V3.33", "V3.34", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1"))
assert "threading.main_thread()" in C and "serve_asks()" in C and "real_canvas" in C and "toggle_params" in C
S = (ROOT / "start_nero.sh").read_text()
assert "NERO_V314_Joint_Control.py" in S
J = (ROOT / "NERO_V314_Joint_Control.py").read_text()
assert "V3.14 JOINT CONTROL" in J and "class JointPanelUI" in J and "self.pipe.log(" in J
print("[J7] console: scrollable real tab, collapsible params, joint panel wired, thread-safe confirm; launcher checks module")
shutil.rmtree(TMP, ignore_errors=True)
print("ALL V3.14 TESTS PASS")
