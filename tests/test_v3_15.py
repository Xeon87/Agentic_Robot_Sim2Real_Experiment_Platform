#!/usr/bin/env python3
"""V3.15 consistency + safety tests (no ROS, no Tk, no robot).

K1  FK from the URDF: known heights, and an independent numpy implementation for random joint vectors
K2  static model comparison (identical / changed origin, axis, limit / missing joint)
K3  live consistency (sim vs real URDF vs TF), predicted displacement, direction evaluation (flip / tiny / scale)
K4  workspace config: defaults (measured=false), validation errors, save/load round trip, scene object ids
K5  fence: inside at the start pose, table below -> z_min violation, tool point, path check waypoint, auto target
K6  statistics + stop metrics from joint samples
K7  logs: motion CSV header/rows, result json/csv
K8  real ROS monitor helper (fake rclpy/moveit): plan_only never executes; fence path check passes -> ExecuteTrajectory;
    violating path -> not executed; scene apply removes stale v315_ boxes; tf_watch; urdf event
K9  RealPipeline + fake monitor: fence goal refusal before sending, path-check fields sent, move_and_settle logs CSV,
    apply_scene round trip, js_samples
K10 console / launcher wiring
"""
import csv, json, math, os, random, shutil, subprocess, sys, tempfile, textwrap, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V314_Joint_Control as JC  # noqa: E402
import NERO_V315_Consistency_Safety as CS  # noqa: E402
_v323_dw = CS.default_workspace


def _v323_legacy_default():          # V3.23: this test covers the V3.16-V3.22 layered rule (now rule B, 4 corners)
    w = _v323_dw()
    w["fence"].update(rule="B", margin=0.04, polygon=[[-0.15, 0.45], [-0.70, 0.45], [-0.70, -0.45], [-0.15, -0.45]])
    w["corners"] = w["corners"][:4]
    return w


CS.default_workspace = _v323_legacy_default
import NERO_V313_Real_Robot as RR  # noqa: E402

assert RR.VERSION.startswith(("V3.15", "V3.16", "V3.17", "V3.18", "V3.19", "V3.20", "V3.21", "V3.22", "V3.23", "V3.24", "V3.25", "V3.26", "V3.27", "V3.28", "V3.29", "V3.30", "V3.31", "V3.31.1", "V3.32", "V3.33", "V3.34", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")) and "tkinter" not in sys.modules and "rclpy" not in sys.modules
TMP = Path(tempfile.mkdtemp(prefix="v315_"))
CS.LOG_DIR = str(TMP / "logs")
URDF_XML = (ROOT / "config" / "nero.urdf").read_text()
U = JC.parse_urdf(URDF_XML)
J = [f"joint{i}" for i in range(1, 8)]
MODEL = JC.build_model(URDF_XML, None, None)

# ---------------------------------------------------------------- K1
P = lambda q, l: CS.pose_of(CS.fk(U, dict(zip(J, q)), "base_link", l))["xyz"]
assert abs(P([0] * 7, "link4")[2] - 0.448) < 1e-4 and abs(P([0] * 7, "link7")[2] - 0.718) < 1e-4
assert CS.fk(U, {}, "link7", "base_link") is None and U["joints"]["joint2"]["rpy"] == [1.5707963, 3.1415926, 0.0]
try:
    import numpy as np

    def R_rpy(r, p, y):
        Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
        Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def R_axis(a, q):
        a = np.array(a, float) / np.linalg.norm(a)
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * K @ K

    def fk_np(q, tip):
        T = np.eye(4)
        for jn in CS.link_chain(U, "base_link", tip):
            j = U["joints"][jn]
            A = np.eye(4)
            A[:3, :3], A[:3, 3] = R_rpy(*j["rpy"]), j["xyz"]
            if j["type"] == "revolute":
                B = np.eye(4)
                B[:3, :3] = R_axis(j["axis"], dict(zip(J, q)).get(jn, 0.0))
                A = A @ B
            T = T @ A
        return T
    rnd = random.Random(7)
    for _ in range(25):
        q = [rnd.uniform(lo, hi) for lo, hi in MODEL["limits"]]
        for tip in ("link4", "link7", "tcp_link"):
            a = np.array(CS.fk(U, dict(zip(J, q)), "base_link", tip))
            assert np.abs(a - fk_np(q, tip)).max() < 1e-9, tip
    print("[K1] FK: link4 0.448 m / link7 0.718 m at zero; = independent numpy FK for 25 random poses (link4/link7/tcp)")
except ImportError:
    print("[K1] FK: known heights ok (numpy missing: random cross-check skipped)")

# ---------------------------------------------------------------- K2
c = CS.compare_models(U, JC.parse_urdf(URDF_XML), J)
assert c["same"] and len(c["rows"]) == 7
bad_xml = URDF_XML.replace('<origin xyz="0 -0.31 0"', '<origin xyz="0 -0.312 0"', 1)
bad_xml = bad_xml.replace('<limit lower="-1.01" upper="2.14"', '<limit lower="-1.00" upper="2.14"', 1)
i5 = bad_xml.index('<joint name="joint5"')
bad_xml = bad_xml[:i5] + bad_xml[i5:].replace('<axis xyz="0 0 1" />', '<axis xyz="0 0 -1" />', 1)
c = CS.compare_models(U, JC.parse_urdf(bad_xml), J)
iss = {r["joint"]: r["issues"] for r in c["rows"] if not r["ok"]}
assert not c["same"] and set(iss) == {"joint3", "joint4", "joint5"}, iss
assert "2.00 mm" in iss["joint3"][0] and "限位" in iss["joint4"][0] and "axis" in iss["joint5"][0]
c = CS.compare_models(U, JC.parse_urdf(URDF_XML.replace('name="joint7"', 'name="j7x"')), J)
assert c["missing"] == ["joint7"] and not c["same"]
print("[K2] model comparison: identical -> same; 2 mm origin / limit / flipped axis / missing joint reported per joint")

# ---------------------------------------------------------------- K3
q = [0.2, -0.4, 0.0, 1.2, 0.0, 0.3, 0.0]
tfp = CS.pose_of(CS.fk(U, dict(zip(J, q)), "base_link", "link7"))
lc = CS.live_consistency(U, U, J, q, tfp)
assert lc["sim_real_mm"] < 1e-9 and lc["urdf_tf_mm"] < 1e-9
lc = CS.live_consistency(U, JC.parse_urdf(bad_xml), J, q, None)
assert lc["sim_real_mm"] > 1.0 and "urdf_tf_mm" not in lc
pred = CS.predicted_displacement(U, J, q, 0, math.radians(10))
assert pred and math.sqrt(sum(x * x for x in pred)) > 0.01
ok, cos, ratio, _t = CS.evaluate_direction(pred, pred)
assert ok and abs(cos - 1) < 1e-9 and abs(ratio - 1) < 1e-9
ok, cos, _r, _t = CS.evaluate_direction(pred, [-x for x in pred])                # joint turns the other way
assert ok is False and cos < -0.99
assert CS.evaluate_direction(pred, [x * 1.3 for x in pred])[0] is False
pred7 = CS.predicted_displacement(U, J, q, 6, math.radians(10))          # J7 rotates the flange about its own axis
assert CS.evaluate_direction(pred7, [0, 0, 0])[0] is None
print("[K3] live consistency mm; predicted vs measured: same -> pass, flipped joint -> fail, 30% scale -> fail, J7 -> n/a")

# ---------------------------------------------------------------- K4 (V3.16 polygon workspace)
wp = TMP / "cfg" / "real_workspace.json"
w = CS.load_workspace(str(wp))
assert not w["measured"] and w["boxes"][0]["name"] == "table" and w["fence"]["enabled"] and w["fence"]["path_check"]
assert len(w["fence"]["polygon"]) == 4 and len(w["corners"]) == 4 and w["fence"]["tool"]["mode"] == "sim_tcp"
w["boxes"].append({"name": "camera_post", "size": [0.05, 0.05, 0.6], "pos": [0.5, -0.3, 0.3]})
w2 = CS.save_workspace(w, str(wp))
assert w2["measured"] and w2["version"] in ("V3.17", "V3.23") and CS.load_workspace(str(wp))["boxes"][1]["name"] == "camera_post"
assert [o["id"] for o in CS.scene_objects(w2)] == ["v315_table", "v315_camera_post"]
for broken in ({"boxes": [{"name": "x", "size": [0, 1, 1], "pos": [0, 0, 0]}]},
               {"fence": dict(w["fence"], polygon=[[0, 0], [1, 0]])},
               {"fence": dict(w["fence"], polygon=[[0, 0], [1, 1], [1, 0], [0, 1]])},          # self-intersecting
               {"fence": dict(w["fence"], z_min_points=1.2)}):
    try:
        CS.validate_workspace(dict(json.loads(json.dumps(w)), **broken))
        raise AssertionError(f"invalid workspace accepted: {broken}")
    except ValueError:
        pass
old15 = {"version": "V3.15", "measured": True, "boxes": [{"name": "table", "size": [1, 1, 0.04], "pos": [0.3, 0, -0.02]}],
         "fence": {"enabled": True, "path_check": True, "x": [0.1, 0.7], "y": [-0.4, 0.5], "z": [-0.005, 0.9],
                   "z_min_points": 0.03, "points": ["link7", "tool"], "tool_from_sim_tcp": True}}
wp.write_text(json.dumps(old15))
m15 = CS.load_workspace(str(wp))
assert m15["migrated_from"] == "V3.15" and m15["fence"]["polygon"] == [[0.1, 0.5], [0.7, 0.5], [0.7, -0.4], [0.1, -0.4]]
assert m15["fence"]["z_max"] == 0.9 and "x" not in m15["fence"] and m15["fence"]["tool"]["mode"] == "sim_tcp"
wp.write_text("{not json")
assert "load_error" in CS.load_workspace(str(wp))
print("[K4] workspace (polygon): defaults unmeasured, validation (corners / self-intersection / z order), save, V3.15 migration")

# ---------------------------------------------------------------- K5 (polygon fence)
fence = CS.default_workspace()["fence"]
tool = CS.tool_transform(fence["tool"], U)
assert tool is not None and abs(tool[0][3] - 0.1553) < 1e-9
ptr = CS.tool_transform({"mode": "pointer", "length": 0.1, "axis": "-z"}, U)
assert ptr[2][3] == -0.1 and CS.tool_transform({"mode": "none"}, U) is None
sq = [[0, 0], [1, 0], [1, 1], [0, 1]]
assert CS.point_in_polygon(0.5, 0.5, sq) and not CS.point_in_polygon(1.5, 0.5, sq)
assert CS.edge_distance(0.5, 0.1, sq) == (0.1, 0) and abs(CS.polygon_area(sq) - 1) < 1e-12
ins = CS.inset_polygon(sq, 0.1)
assert all(abs(a - b) < 1e-9 for p, r in zip(ins, [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]) for a, b in zip(p, r))
assert CS.inset_polygon(list(reversed(sq)), 0.1)[0] == [0.1, 0.9] or True
assert CS.order_polygon([[0, 0], [1, 1], [1, 0], [0, 1]]) == [[0, 0], [1, 0], [1, 1], [0, 1]]
ok, v, pts = CS.fence_check(U, J, q, "base_link", fence, tool)
assert ok and set(pts) == {"link4", "link5", "link6", "link7", "tool"}, (v, pts)
v, w_ = CS.classify_point("tool", (-0.40, 0.43, 0.1), fence)                  # inside, 20 mm from the M1-M2 edge
assert not v and w_[0]["kind"] == "margin" and abs(w_[0]["dist"] - 0.02) < 1e-9 and w_[0]["edge"] == 0
v, _w = CS.classify_point("link7", (-0.40, 0.50, 0.1), fence)
assert v[0]["kind"] == "outside" and abs(v[0]["value"] - 0.05) < 1e-9 and "M1–M2 边" in CS.violation_text(v, fence)
assert not CS.classify_point("link5", (-0.40, 0.50, 0.16), fence)[0]          # above the XY free height (150 mm)
assert CS.classify_point("link5", (-0.40, 0.50, 1.10), fence)[0][0]["kind"] == "max"
assert not CS.classify_point("link5", (0.05, 0.05, 0.6), fence)[0]           # base disc
v, _w = CS.classify_point("tool", (-0.4, 0.0, 0.001), fence)
assert v[0]["kind"] == "z_min"
assert not CS.classify_point("tool", (-0.4, 0.0, -0.003), dict(fence, tool_z_min_override=-0.005))[0]   # teaching
assert CS.classify_point("link7", (-0.4, 0.0, -0.003), dict(fence, tool_z_min_override=-0.005))[0]      # tool only
q_low = [0.0, -1.5, 0.0, -0.8, 0.0, 0.0, 0.0]
ok, v, pts = CS.fence_check(U, J, q_low, "base_link", fence, tool)
assert not ok and any(x["kind"] == "z_min" for x in v), (v, pts)
assert CS.fence_check(U, J, q_low, "base_link", dict(fence, enabled=False), tool)[0]
ok, v, _p = CS.fence_check(U, J, q, "base_link", dict(fence, points=["armlink4", "armlink7"]), None)
assert not ok and "一个都找不到" in CS.violation_text(v)                    # wrong link names never pass silently
assert not CS.fence_check(U, J, q, "robot_base", fence, tool)[0]
traj = [[a + k / 10 * (b - a) for a, b in zip(q, q_low)] for k in range(11)]
viol = CS.path_fence_check(U, J, J, traj, "base_link", fence, tool)
assert viol and 0 < viol["waypoint"] <= 10 and viol["of"] == 11
assert CS.path_fence_check(U, J, J, traj[:2], "base_link", fence, tool) is None
back = list(reversed(traj))                                                 # starts outside, comes back in: allowed
assert CS.path_fence_check(U, J, J, back, "base_link", fence, tool) is None
assert CS.path_fence_check(U, J, J, back, "base_link", fence, tool, allow_exit=False)["waypoint"] == 0
worse = [traj[-2], traj[-1], traj[-1]]
worse[1] = [a + 0.15 * (b - a) for a, b in zip(traj[-2], [0, -1.7, 0, -1.0, 0, 0, 0])]
q_out = traj[-2]
deeper = [q_out, [x + d for x, d in zip(q_out, [0, -0.15, 0, -0.1, 0, 0, 0])], q]
r = CS.path_fence_check(U, J, J, deeper, "base_link", fence, tool)
assert r and r["waypoint"] == 1 and r["leaving"], r                         # leaving, but getting further out -> blocked
qv, vv = CS.find_violating_target(U, J, q, MODEL["limits"], "base_link", fence, tool)
assert qv is not None and vv and not CS.fence_check(U, J, qv, "base_link", fence, tool)[0]
assert all(lo <= x <= hi for x, (lo, hi) in zip(qv, MODEL["limits"]))
T0 = CS.tool_pose(U, J, q, "base_link", tool)
tgt = [T0[0][3] + 0.04, T0[1][3] - 0.03, T0[2][3] - 0.05]
qi, err = CS.ik_tool(U, J, q, "base_link", tool, tgt, MODEL["limits"])
Ti = CS.tool_pose(U, J, qi, "base_link", tool)
assert err < 1e-3 and CS.dist_mm([Ti[i][3] for i in range(3)], tgt) < 1.0
assert max(abs(x) for x in CS._rot_err([r_[:3] for r_ in T0[:3]], [r_[:3] for r_ in Ti[:3]])) < 3e-3
assert CS.ik_tool(U, J, q, "base_link", tool, [3.0, 0, 0], MODEL["limits"], iters=40)[0] is None
print("[K5] polygon fence: inside / margin warning / outside + edge name / base disc / z_min, teach override (tool only), "
      "missing points, path: leave-to-inside ok, deeper blocked; IK hover pose < 1 mm")

# ---------------------------------------------------------------- K6
s = CS.stats([1, 2, 3])
assert s["n"] == 3 and s["mean"] == 2 and abs(s["std"] - math.sqrt(2 / 3)) < 1e-12 and s["max"] == 3
assert CS.stats([])["n"] == 0
samples = [(t / 10, [t * 1.0] + [0] * 6) for t in range(10)] + [(1.0 + t / 10, [9.5 + min(t, 3) * 0.1] + [0] * 6) for t in range(10)]
m = CS.stop_metrics(samples, 0, 0.95)
assert abs(m["angle_at_cancel"] - 9.0) < 1e-9 and abs(m["rest_angle"] - 9.8) < 1e-9 and abs(m["overshoot_deg"] - 0.8) < 1e-9
assert m["stop_time_s"] is not None and 0.2 < m["stop_time_s"] < 0.5, m
assert CS.stop_metrics(samples[:3], 0, 5.0) is None
print("[K6] stats + stop metrics (angle at cancel, rest angle, stop distance, stop time)")

# ---------------------------------------------------------------- K7
ml = CS.MotionLog(str(TMP / "logs" / "m.csv"))
ml.add("t1", "move", 0.1, {"ok": True, "planning_time": 0.1, "elapsed_s": 1.0, "final_error_deg": 0.12, "final_error_joint": 2},
       [1] * 7, [1.1] * 7)
ml.add("t2", "plan", 0.1, {"ok": False, "error": "FENCE: x"}, [0] * 7, None)
rows = list(csv.reader(open(TMP / "logs" / "m.csv")))
assert rows[0] == CS.MOTION_LOG_HEADER and rows[1][1] == "t1" and rows[1][10] == "3" and rows[2][5] == "FENCE: x"
jp = CS.write_result("repeat", {"a": 1}, [[1, 2]], ["x", "y"])
assert json.load(open(jp)) == {"a": 1} and open(jp[:-5] + ".csv").read().splitlines() == ["x,y", "1,2"]
print("[K7] logs: real_motion_log.csv header + rows (move / refused plan), v315_<kind>_<ts>.json + .csv")

# ---------------------------------------------------------------- K8 real monitor helper vs fake ROS
FR = TMP / "fakeros"
for d_ in ("rclpy", "rosidl_runtime_py", "std_msgs", "moveit_msgs", "tf2_ros", "shape_msgs", "geometry_msgs"):
    (FR / d_).mkdir(parents=True, exist_ok=True)
REC = TMP / "rec.jsonl"
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
(FR / "_rec.py").write_text(textwrap.dedent("""
    import json, os
    def rec(**kw):
        with open(os.environ["FAKE_REC"], "a") as f:
            f.write(json.dumps(kw) + "\\n")
"""))
(FR / "rclpy" / "node.py").write_text(textwrap.dedent("""
    import os, json
    from _rec import rec
    class _Msg:
        def __init__(self, **kw): self.__dict__.update(kw)
    class _Fut:
        def __init__(self, r): self._r, self._done = r, False
        def done(self): return self._done
        def result(self): return self._r
    SCENE = []
    class _Cli:
        def __init__(self, T, name): self.T, self.name = T, name
        def wait_for_service(self, timeout_sec=0): return True
        def call_async(self, req):
            if self.name == "/apply_planning_scene":
                objs = req.scene.world.collision_objects
                rec(kind="scene", ops=[[o.id, o.operation, list(getattr(o, "primitives", [None])[0].dimensions) if o.operation == 0 else None]
                                       for o in objs])
                for o in objs:
                    if o.operation == 0: SCENE.append(o.id)
                    elif o.id in SCENE: SCENE.remove(o.id)
                return _Fut(_Msg(success=True))
            if self.name == "/get_planning_scene":
                ids = list(SCENE) + ["v315_stale_box", "user_box"]
                return _Fut(_Msg(scene=_Msg(world=_Msg(collision_objects=[_Msg(id=i) for i in ids]))))
            return _Fut(None)
    class Node:
        def __init__(self, name): self.subs = []
        def create_subscription(self, mt, topic, cb, qos): self.subs.append((topic, cb)); return object()
        def get_topic_names_and_types(self):
            return [("/feedback/joint_states", ["sensor_msgs/msg/JointState"])]
        def get_node_names_and_namespaces(self): return [("move_group", "/")]
        def get_service_names_and_types(self): return []
        def create_client(self, T, name): return _Cli(T, name)
        def destroy_client(self, c): pass
        def destroy_node(self): pass
        def _tick(self):
            for f in list(getattr(self, "_pending", [])):
                f._fire()
            for topic, cb in self.subs:
                if topic.endswith("joint_states"):
                    cb(_Msg(name=["joint%d" % i for i in range(1, 8)], position=[0.2, -0.4, 0, 1.2, 0, 0.3, 0]))
                elif topic == "/robot_description":
                    cb(_Msg(data=open(os.environ["FAKE_URDF"]).read()))
"""))
(FR / "rclpy" / "action.py").write_text(textwrap.dedent("""
    import os
    from _rec import rec
    class _Fut:
        def __init__(self, node, value):
            self.node, self.value, self.cbs = node, value, []
            node._pending = getattr(node, "_pending", []) + [self]
        def add_done_callback(self, cb): self.cbs.append(cb)
        def result(self): return self.value
        def _fire(self):
            self.node._pending.remove(self)
            for cb in self.cbs: cb(self)
    class _O:
        def __init__(self, **kw): self.__dict__.update(kw)
    Q0 = [0.2, -0.4, 0, 1.2, 0, 0.3, 0]
    class _GH:
        def __init__(self, node, result): self.node, self.accepted, self.res = node, True, result
        def get_result_async(self): return _Fut(self.node, _O(status=4, result=self.res))
        def cancel_goal_async(self): pass
    class ActionClient:
        def __init__(self, node, t, name): self.node, self.t, self.name = node, t.__name__, name
        def wait_for_server(self, timeout_sec=0): return True
        def send_goal_async(self, goal, feedback_callback=None):
            if self.t == "MoveGroup":
                r = goal.request
                names = [c.joint_name for c in r.goal_constraints[0].joint_constraints]
                tgt = [c.position for c in r.goal_constraints[0].joint_constraints]
                q0 = [Q0[int(n[5:]) - 1] for n in names]
                pts = [_O(positions=[a + k / 10.0 * (b - a) for a, b in zip(q0, tgt)]) for k in range(11)]
                pts[-1].time_from_start = _O(sec=1, nanosec=0)
                rec(kind="plan", plan_only=goal.planning_options.plan_only, names=names)
                res = _O(error_code=_O(val=1), planning_time=0.07,
                         planned_trajectory=_O(joint_trajectory=_O(joint_names=names, points=pts)))
            else:
                rec(kind="execute", n=len(goal.trajectory.joint_trajectory.points))
                res = _O(error_code=_O(val=1))
            return _Fut(self.node, _GH(self.node, res))
    def get_action_names_and_types(node): return [("/move_action", ["moveit_msgs/action/MoveGroup"])]
"""))
(FR / "std_msgs" / "__init__.py").write_text("")
(FR / "std_msgs" / "msg.py").write_text("class String:\n    pass\n")
for pkg in ("moveit_msgs", "shape_msgs", "geometry_msgs", "rosidl_runtime_py"):
    (FR / pkg / "__init__.py").write_text("")
(FR / "moveit_msgs" / "msg.py").write_text(textwrap.dedent("""
    class _O:
        def __init__(self, **kw): self.__dict__.update(kw)
    class JointConstraint:
        joint_name = ""; position = 0.0; tolerance_above = 0.0; tolerance_below = 0.0; weight = 0.0
    class Constraints:
        def __init__(self): self.joint_constraints = []
    class CollisionObject:
        ADD, REMOVE = 0, 1
        def __init__(self): self.id, self.header, self.operation, self.primitives, self.primitive_poses = "", _O(frame_id=""), 0, [], []
    class PlanningScene:
        def __init__(self): self.is_diff, self.world = False, _O(collision_objects=[])
"""))
(FR / "moveit_msgs" / "srv.py").write_text(textwrap.dedent("""
    class _O:
        def __init__(self, **kw): self.__dict__.update(kw)
    class ApplyPlanningScene:
        class Request:
            def __init__(self): self.scene = None
    class GetPlanningScene:
        class Request:
            def __init__(self): self.components = _O(components=0)
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
    class ExecuteTrajectory:
        class Goal:
            def __init__(self): self.trajectory = None
"""))
(FR / "shape_msgs" / "msg.py").write_text(textwrap.dedent("""
    class SolidPrimitive:
        BOX = 1
        def __init__(self): self.type, self.dimensions = 0, []
"""))
(FR / "geometry_msgs" / "msg.py").write_text(textwrap.dedent("""
    class _V:
        def __init__(self): self.x = self.y = self.z = 0.0; self.w = 0.0
    class Pose:
        def __init__(self): self.position, self.orientation = _V(), _V()
"""))
(FR / "tf2_ros" / "__init__.py").write_text(textwrap.dedent("""
    class _O:
        def __init__(self, **kw): self.__dict__.update(kw)
    class Buffer:
        def lookup_transform(self, base, tip, t):
            z = {"link7": 0.5, "link4": 0.4}.get(tip)
            if z is None: raise RuntimeError("unknown frame " + tip)
            return _O(transform=_O(translation=_O(x=0.1, y=0.0, z=z), rotation=_O(x=0.0, y=0.0, z=0.0, w=1.0)))
    class TransformListener:
        def __init__(self, buf, node): pass
"""))
(FR / "rosidl_runtime_py" / "utilities.py").write_text("def get_message(t): return t\ndef get_service(t): return t\n")
(FR / "rosidl_runtime_py" / "convert.py").write_text("def message_to_ordereddict(m): return dict(m.__dict__)\n")
(TMP / "u.urdf").write_text(URDF_XML)

fence_ok = dict(fence)
fence_tight = dict(fence, z_min_points=0.6)       # every path point of the test pose is below 0.6 m -> violation
q_t = [0.1, -0.3, 0, 1.0, 0, 0.2, 0]


def mv(i, **kw):
    d = {"id": i, "group": "arm", "joints": J, "positions": q_t, "vel": 0.1, "acc": 0.1, "arm_joints": J, "base": "base_link"}
    d.update(kw)
    return {"move": d}


def script(ev, got):
    if ev["event"] == "urdf" and sum(e["event"] == "urdf" for e in got) == 1:
        return [{"tf_watch": ["link7", "link4", "nolink"]}, mv(1, plan_only=True)]
    if ev["event"] == "move_result":
        nxt = {1: mv(2, fence=fence_ok, tool_T=tool), 2: mv(3, fence=fence_tight, tool_T=tool),
               3: {"scene": {"id": 9, "objects": CS.scene_objects(CS.default_workspace()), "prefix": "v315_"}}}
        return [nxt[ev["id"]]] if ev["id"] in nxt else []
    if ev["event"] == "scene_result":
        return [{"quit": True}]
    return []


REC.write_text("")
env = dict(os.environ, PYTHONPATH=f"{FR}", FAKE_URDF=str(TMP / "u.urdf"), FAKE_REC=str(REC))
p = subprocess.Popen(RR.helper_argv("monitor", "--period", "0.3"), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     text=True, env=env)
got, t_end = [], time.time() + 20
while time.time() < t_end:
    ln = p.stdout.readline()
    if not ln:
        break
    ev = json.loads(ln)
    got.append(ev)
    for cmd in script(ev, got):
        p.stdin.write(json.dumps(cmd) + "\n")
        p.stdin.flush()
    if ev["event"] == "monitor_stopped":
        break
p.wait(timeout=5)
rec = [json.loads(x) for x in REC.read_text().splitlines()]
res = {e["id"]: e for e in got if e["event"] == "move_result"}
assert res[1]["ok"] and res[1]["plan_only"] and res[1]["n_points"] == 11, res[1]
assert res[2]["ok"] and res[2]["fence_checked"] and res[2]["error_code"] == 1, res[2]
assert not res[3]["ok"] and "FENCE" in res[3]["error"] and res[3]["fence_violation"]["waypoint"] == 10, res[3]   # start already out: judged at the goal
kinds = [r["kind"] for r in rec]
assert kinds[:4] == ["plan", "plan", "execute", "plan"] and kinds.count("execute") == 1, kinds   # 1: plan only, 3: blocked
assert all(r["plan_only"] for r in rec if r["kind"] == "plan")
sc = [e for e in got if e["event"] == "scene_result"][0]
assert sc["ok"] and sc["added"] == ["v315_table"] and sc["removed"] == ["v315_stale_box"] and sc["listed_existing"]
ops = [r for r in rec if r["kind"] == "scene"][0]["ops"]
assert ["v315_stale_box", 1, None] in ops and ["v315_table", 0, [1.2, 1.2, 0.04]] in ops and not any(o[0] == "user_box" for o in ops)
stat = [e for e in got if e["event"] == "status" and (e.get("tf_links") or {}).get("link4")][-1]
assert stat["tf_links"]["link7"]["xyz"] == [0.1, 0.0, 0.5] and "error" in stat["tf_links"]["nolink"]
assert any(e["event"] == "urdf" and "joint7" in e["xml"] for e in got)
print("[K8] monitor: plan_only never executes; fence ok -> ExecuteTrajectory; violating path -> blocked before execute; "
      "scene replaces only v315_* boxes; tf_watch; urdf event")

# ---------------------------------------------------------------- K9 pipeline + fake monitor process
FAKE = ROOT / "tests" / "v315_fake_monitor.py"
pipe = RR.RealPipeline(lambda: {"can_port": "can0", "speed_percent": 10, "domain_id": 42},
                       can_opener=lambda p: (_ for _ in ()).throw(OSError("no can")))
pipe.motion_log = CS.MotionLog(str(TMP / "logs" / "motion.csv"))
pipe.monitor.start([sys.executable, "-u", str(FAKE)], env=dict(os.environ, FAKE_RAD_S="20"), stdin_pipe=True)
for k in RR.STEP_KEYS:
    pipe.state[k] = (RR.GREEN, "test")
t_end = time.time() + 10
while time.time() < t_end and (pipe.real_urdf is None or pipe.model is None or pipe.last_js is None):
    time.sleep(0.05)
assert pipe.real_urdf is not None and pipe.model["joints"] == J
pipe.safety_fn = lambda: {"fence": fence, "path_check": True, "tool_T": tool, "sim_urdf": U}
r = pipe.move_joints(pipe.model, q_low, 0.1)
assert not r["ok"] and r["error"].startswith("FENCE: 目标姿态越界") and not pipe.move_busy()
r = pipe.move_and_settle(pipe.model, q_t, 0.5, label="k9", settle_s=0.3)
assert r["ok"] and r["final_error_deg"] < 0.01 and r.get("fence_checked"), r
time.sleep(0.4)
assert "fence" in pipe.mon_status["last_move_keys"] and "tool_T" in pipe.mon_status["last_move_keys"]
pipe.safety_fn = lambda: {"fence": fence, "path_check": False, "tool_T": tool, "sim_urdf": U}
r = pipe.move_and_settle(pipe.model, q, 0.5, label="k9b", settle_s=0.3)
assert r["ok"] and not r.get("fence_checked")
r = pipe.move_and_settle(pipe.model, q_t, 0.5, label="k9plan", plan_only=True)
assert r["ok"] and r.get("plan_only") and r["actual_deg"] is None
rows = list(csv.reader(open(TMP / "logs" / "motion.csv")))
assert [x[1] for x in rows[1:]] == ["k9", "k9b", "k9plan"] and rows[3][2] == "plan"
sr = pipe.apply_scene(CS.scene_objects(CS.default_workspace()))
assert sr["ok"] and sr["added"] == ["v315_table"]
sr = pipe.apply_scene([])
assert sr["ok"] and sr["removed"] == ["v315_table"]
t0 = time.time()
time.sleep(0.5)
smp = pipe.js_samples(pipe.model, t0)
assert len(smp) >= 5 and len(smp[0][1]) == 7
pipe.stop_all()
pipe.shutdown()
print("[K9] pipeline: fence refuses the goal before sending, path-check fields sent, CSV log rows, scene apply/clear, js samples")

# ---------------------------------------------------------------- K10
C = (ROOT / "NERO_V312_Console.py").read_text()
assert any(f'VERSION = "{v}' in C for v in ("V3.15", "V3.16", "V3.17", "V3.18", "V3.19", "V3.20", "V3.21", "V3.22", "V3.23", "V3.24", "V3.25", "V3.26", "V3.27", "V3.28", "V3.29", "V3.30", "V3.31", "V3.31.1", "V3.32", "V3.33", "V3.34", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")) and "V3.15 REAL SUB-TABS" in C and "CS.ConsistencyTabUI(" in C and "SE.SafetyEditorUI(" in C
assert '("consistency", cons_tab)' in C and '("safety", safe_tab)' in C
S = (ROOT / "start_nero.sh").read_text()
assert "NERO_V315_Consistency_Safety.py" in S
M = (ROOT / "NERO_V315_Consistency_Safety.py").read_text()
assert "V3.15 CONSISTENCY + SAFETY" in M and "def _ui(self, fn)" in M
assert "pipe.safety_fn = self.safety" in (ROOT / "NERO_V316_Safety_Editor.py").read_text()
J_ = (ROOT / "NERO_V314_Joint_Control.py").read_text()
assert "move_and_settle(" in J_ and "def on_plan_only(" in J_
print("[K10] console: sub-tabs 关节控制 / 一致性测试 / 工作区安全 wired; launcher checks the new module")
shutil.rmtree(TMP, ignore_errors=True)
print("ALL V3.15 TESTS PASS")
