#!/usr/bin/env python3
"""V3.19 tests (no ROS, no Tk, no robot).

M1 teach-mode helper (fake pyAgxArm): teach -> connect + set_master_mode (1 s pauses); normal -> set_normal_mode +
   enable retries; enable never succeeds -> error; pyAgxArm missing -> error
M2 RealPipeline.set_teach_mode: teach flag, joint motion refused while teaching, exit restores; uses the driver's
   exit_teach_mode service when offered
M3 async service calls in the ROS monitor (fake rclpy): a slow service (move_home) does not stop the joint feedback;
   result arrives later with elapsed time; timeout reported; immediate error for a missing service
M4 wiring: Home goes through MoveIt (go_home), zone files (open / save as / last file), teach buttons
"""
import json, os, subprocess, sys, tempfile, textwrap, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V313_Real_Robot as RR  # noqa: E402

assert RR.VERSION.startswith(("V3.19", "V3.20", "V3.21", "V3.22", "V3.23", "V3.24", "V3.25", "V3.26", "V3.27", "V3.28", "V3.29", "V3.30", "V3.31", "V3.32", "V3.33", "V3.34", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1"))
TMP = Path(tempfile.mkdtemp(prefix="v319_"))
FAKE = TMP / "fakepkgs"
(FAKE / "pyAgxArm").mkdir(parents=True)
(FAKE / "pyAgxArm" / "__init__.py").write_text(textwrap.dedent("""
    import os, time
    class ArmModel: NERO = "NERO"
    class NeroFW: V111 = "V111"
    def create_agx_arm_config(**kw):
        assert kw["robot"] == "NERO" and kw["interface"] == "socketcan"
        return kw
    def _m(x): open(os.environ["FAKE_MARK"], "a").write(f"{time.time():.2f} {x}\\n")
    class _Arm:
        def __init__(self, cfg): self.n = 0; self.need = int(os.environ.get("FAKE_ENABLE_AFTER", "2"))
        def connect(self): _m("connect")
        def enable(self):
            self.n += 1
            ok = self.need >= 0 and self.n > self.need
            if ok: _m("enabled")
            return ok
        def set_normal_mode(self): _m("normal")
        def set_master_mode(self): _m("master")
        def get_arm_status(self): return {"ctrl_mode": 2 if "master" in open(os.environ["FAKE_MARK"]).read().split()[-1:] else 1}
    class AgxArmFactory:
        @staticmethod
        def create_arm(cfg): return _Arm(cfg)
"""))
mark = TMP / "mark.txt"
os.environ["FAKE_MARK"] = str(mark)
os.environ["PYTHONPATH"] = str(FAKE) + os.pathsep + os.environ.get("PYTHONPATH", "")


def helper(mode, env=None):
    r = subprocess.run(RR.helper_argv("mode", "--channel", "can0", "--fw", "v111", "--mode", mode), capture_output=True,
                       text=True, timeout=30, env=env)
    return r.returncode, json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1])


# ---------------------------------------------------------------- M1
mark.write_text("")
rc, ev = helper("teach")
log = [l.split() for l in mark.read_text().splitlines()]
assert rc == 0 and ev["ok"] and ev["mode"] == "teach" and [x[1] for x in log] == ["connect", "master"], (ev, log)
assert float(log[1][0]) - float(log[0][0]) >= 0.95                      # AgileX: 1 s pause before the switch
mark.write_text("")
rc, ev = helper("normal")
log = [l.split()[1] for l in mark.read_text().splitlines()]
assert rc == 0 and ev["ok"] and ev["enabled"] and log[0] == "connect" and log[-1] == "enabled" and set(log[1:-1]) == {"normal"}, (ev, log)
rc, ev = helper("normal", env=dict(os.environ, FAKE_ENABLE_AFTER="-1"))
assert rc == 2 and not ev["ok"] and "enable" in ev["error"]
rc, ev = helper("teach", env=dict(os.environ, PYTHONPATH="", PYTHONNOUSERSITE="1"))
assert rc == 1 and "pyAgxArm import failed" in ev["error"]
print("[M1] mode helper: teach = set_master_mode after 1 s; normal = set_normal_mode + enable; failures reported")

# ---------------------------------------------------------------- M2
import NERO_V314_Joint_Control as JC  # noqa: E402
X = (ROOT / "config" / "nero.urdf").read_text()
MODEL = JC.build_model(X, None, None)
pipe = RR.RealPipeline(lambda: {"can_port": "can0", "push_fw": "v111", "speed_percent": 10},
                       can_opener=lambda p: (_ for _ in ()).throw(OSError("no can")))
for k in RR.STEP_KEYS:
    pipe.state[k] = (RR.GREEN, "test")
mark.write_text("")
ev = pipe.set_teach_mode(True)
assert ev["ok"] and pipe.teach_mode
r = pipe.move_joints(MODEL, [0.0] * 7, 0.1)
assert not r["ok"] and "示教模式" in r["error"]
calls = []
pipe.mon_status = {"services": {"/exit_teach_mode": "std_srvs/srv/Trigger"}}
pipe.monitor.running = lambda: True
pipe.monitor.send = lambda obj: (calls.append(obj), pipe.call_results.__setitem__(obj["id"], {"ok": True, "response": {"success": True}}))
ev = pipe.set_teach_mode(False)
assert ev["ok"] and not pipe.teach_mode and calls and calls[0]["call"] == "/exit_teach_mode" and calls[0]["timeout"] == 10.0
lg = [l.split()[1] for l in mark.read_text().splitlines()]
assert lg[-1] == "enabled" and "normal" in lg
pipe.shutdown()
print("[M2] set_teach_mode: flag on, motion refused while teaching; exit = exit_teach_mode service + normal + enable")

# ---------------------------------------------------------------- M3 async services in the monitor (fake rclpy)
FR = TMP / "fakeros"
for d_ in ("rclpy", "rosidl_runtime_py"):
    (FR / d_).mkdir(parents=True, exist_ok=True)
(FR / "rclpy" / "__init__.py").write_text(textwrap.dedent("""
    import time as _time
    _ok = [True]
    def init(): pass
    def ok(): return _ok[0]
    def shutdown(): _ok[0] = False
    def spin_once(node, timeout_sec=0.0):
        node._tick(); _time.sleep(timeout_sec)
    def spin_until_future_complete(node, fut, timeout_sec=0.0): raise AssertionError("V3.19 must not block on services")
"""))
(FR / "rclpy" / "qos.py").write_text(textwrap.dedent("""
    class QoSProfile:
        def __init__(self, **kw): self.kw = kw
    class ReliabilityPolicy: BEST_EFFORT = 1; RELIABLE = 2
    class HistoryPolicy: KEEP_LAST = 1
"""))
(FR / "rclpy" / "node.py").write_text(textwrap.dedent("""
    import time
    class _Msg:
        def __init__(self, **kw): self.__dict__.update(kw)
    class _Fut:
        def __init__(self, r, delay): self._r, self._t = r, time.time() + delay
        def done(self): return time.time() >= self._t
        def result(self): return self._r
    class _Cli:
        def __init__(self, name): self.name = name
        def wait_for_service(self, timeout_sec=0): return True
        def call_async(self, req):
            d = {"/move_home": 2.0, "/hang": 99.0}.get(self.name, 0.0)
            return _Fut(_Msg(success=True, message="done " + self.name), d)
    class Node:
        def __init__(self, name): self.subs = []; self.destroyed = 0
        def create_subscription(self, mt, topic, cb, qos): self.subs.append((topic, cb)); return object()
        def get_topic_names_and_types(self): return [("/feedback/joint_states", ["sensor_msgs/msg/JointState"])]
        def get_node_names_and_namespaces(self): return [("move_group", "/")]
        def get_service_names_and_types(self):
            return [("/move_home", ["std_srvs/srv/Trigger"]), ("/hang", ["std_srvs/srv/Trigger"])]
        def create_client(self, st, name): return _Cli(name)
        def destroy_client(self, c): self.destroyed += 1
        def destroy_node(self): pass
        def _tick(self):
            for topic, cb in self.subs:
                cb(_Msg(name=["joint%d" % i for i in range(1, 8)], position=[0.1] * 7))
"""))
(FR / "rosidl_runtime_py" / "__init__.py").write_text("")
(FR / "rosidl_runtime_py" / "utilities.py").write_text(textwrap.dedent("""
    class _Req:
        data = None
    class _Srv:
        Request = _Req
    def get_message(t): return t
    def get_service(t): return _Srv
"""))
(FR / "rosidl_runtime_py" / "convert.py").write_text("def message_to_ordereddict(m): return dict(m.__dict__)\n")
p = subprocess.Popen(RR.helper_argv("monitor", "--period", "0.3"), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                     env=dict(os.environ, PYTHONPATH=str(FR)))
got, sent, t_end = [], False, time.time() + 20
while time.time() < t_end:
    ln = p.stdout.readline()
    if not ln:
        break
    ev = json.loads(ln)
    ev["_rx"] = time.time()
    got.append(ev)
    if ev["event"] == "js" and not sent:
        for c in ({"call": "/move_home", "id": 1, "timeout": 60}, {"call": "/hang", "id": 2, "timeout": 1.0},
                  {"call": "/nope", "id": 3}):
            p.stdin.write(json.dumps(c) + "\n")
        p.stdin.flush()
        sent = time.time()
    if sum(e["event"] == "call_result" for e in got) == 3:
        p.stdin.write(json.dumps({"quit": True}) + "\n")
        p.stdin.flush()
    if ev["event"] == "monitor_stopped":
        break
p.wait(timeout=5)
res = {e["id"]: e for e in got if e["event"] == "call_result"}
assert res[3]["ok"] is False and "not available" in res[3]["error"]
assert res[2]["ok"] is False and "timeout 1 s" in res[2]["error"]
assert res[1]["ok"] and res[1]["response"]["message"] == "done /move_home" and 1.8 <= res[1]["elapsed_s"] <= 3.5, res[1]
js_during = [e for e in got if e["event"] == "js" and sent < e["_rx"] < sent + 1.8]
assert len(js_during) >= 8, len(js_during)                 # joint feedback kept flowing during the 2 s service call
print("[M3] async services: slow move_home result after ~2 s while joint feedback keeps flowing; timeout; missing service")

# ---------------------------------------------------------------- M4 wiring
J = (ROOT / "NERO_V314_Joint_Control.py").read_text(encoding="utf-8")
assert "V3.19 HOME" in J and "def go_home(" in J and 'self._move_and_report("回 Home"' in J and 'timeout_s=60.0' in J
C = (ROOT / "NERO_V312_Console.py").read_text(encoding="utf-8")
assert "jpanel.go_home()" in C and "pipe.set_teach_mode(False" in C and 'VERSION = "V3.' in C
E = (ROOT / "NERO_V316_Safety_Editor.py").read_text(encoding="utf-8")
for m in ("V3.19 ZONE FILES", "def on_open(", "def on_save_as(", "last_zone_file", "def on_teach_enter(", "def on_teach_exit(",
          "进入示教模式（可用手带动）"):
    assert m in E, m
print("[M4] Home via MoveIt (service only as fallback), teach buttons in the teach panel, zone open / save as / last file")
print("ALL V3.19 TESTS PASS")
