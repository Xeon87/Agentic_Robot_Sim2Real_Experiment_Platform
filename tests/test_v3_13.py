#!/usr/bin/env python3
"""Offline tests for V3.13 real-robot pipeline (no ROS / CAN hardware / robot needed).

R1  CAN port: `ip -json` parsing, UP / ERROR-ACTIVE / 1 Mbit rule, ONE pkexec command for down/bitrate/up
R2  CAN flow monitor: SocketCAN frame parsing, 1 s rate, id groups; no push -> yellow; old gripper id -> amber
R3  launch command: arm_type nero fixed, speed clamp 30 unless unlocked, effector whitelist, fw, ROS env + domain
R4  stale/sim process detection (own helpers excluded)
R5  push helper (= nero_start.py) against a fake pyAgxArm: enable after retries, state reports, SIGINT stop;
    enable timeout; pyAgxArm missing
R6  pipeline end-to-end with fakes (CAN frames only while a bus master runs, fake driver + ROS monitor):
    one-click -> 6 green; push-release test picks keep_until_driver and stops push once the driver has feedback;
    "released" arm keeps pushing -> driver started without push; driver "Failed to get firmware version" -> red;
    sim running -> locked; stop_all stops every helper
R7  service calls go to the monitor by suffix; e-stop without driver is refused
R8  console wiring: real tab uses the pipeline, sim/real interlock both ways, e-stop button
"""
import json, os, shutil, signal, socket, struct, subprocess, sys, tempfile, textwrap, threading, time
from pathlib import Path

os.environ["NERO_V313_FLOW_OK_FPS"] = "150"          # the fake bus below runs at ~400-900 fps
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V313_Real_Robot as RR  # noqa: E402

assert RR.VERSION.startswith(("V3.13", "V3.14", "V3.15", "V3.16", "V3.17", "V3.18", "V3.19", "V3.20", "V3.21", "V3.22", "V3.23", "V3.24", "V3.25", "V3.26", "V3.27", "V3.28", "V3.29", "V3.30", "V3.31", "V3.31.1", "V3.32", "V3.33", "V3.34", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")) and "tkinter" not in sys.modules
TMP = Path(tempfile.mkdtemp(prefix="v313_"))

# ---------------------------------------------------------------- R1
IPJ = json.dumps([{"ifindex": 3, "ifname": "can0", "flags": ["NOARP", "UP", "LOWER_UP", "ECHO"], "operstate": "UP",
                   "linkinfo": {"info_kind": "can", "info_data": {"state": "ERROR-ACTIVE", "restart_ms": 100,
                                                                  "berr_counter": {"tx": 0, "rx": 0},
                                                                  "bittiming": {"bitrate": 1000000, "sample_point": "0.750"}}},
                   "stats64": {"rx": {"packets": 20834, "errors": 0}, "tx": {"packets": 1, "errors": 0}}}])
st = RR.parse_ip_details(IPJ)
assert st["up"] and st["bitrate"] == 1000000 and st["state"] == "ERROR-ACTIVE" and st["rx_packets"] == 20834
assert RR.evaluate_can_port(st)[0] == RR.GREEN
down = dict(st, up=False, operstate="DOWN")
assert RR.evaluate_can_port(down)[0] == RR.RED and RR.evaluate_can_port(dict(st, bitrate=500000))[0] == RR.RED
assert RR.evaluate_can_port(dict(st, state="BUS-OFF"))[0] == RR.RED and RR.evaluate_can_port(None)[0] == RR.RED
assert RR.parse_ip_details("garbage") is None
cmd = RR.can_setup_command("can0")
assert cmd[:3] == ["pkexec", "/bin/sh", "-c"] and cmd[3].count("link set can0") == 4 and "bitrate 1000000" in cmd[3]
print("[R1] CAN port parse/evaluate; one pkexec prompt for down + bitrate + up")
# R1b (V3.13.1): gs_usb rejects restart-ms -> the script falls back to bitrate only and still brings can0 UP
import tempfile, stat as _stat
_d = tempfile.mkdtemp()
_ip = os.path.join(_d, "ip"); _log = os.path.join(_d, "calls")
for _support in (False, True):
    open(_log, "w").close()
    with open(_ip, "w") as f:
        f.write("#!/bin/sh\necho \"$*\" >> %s\n" % _log)
        if not _support:
            f.write('case "$*" in *restart-ms*) echo "Error: Device doesn\'t support restart from Bus Off." >&2; exit 2;; esac\n')
        f.write("exit 0\n")
    os.chmod(_ip, 0o755)
    _orig_which = RR.which
    RR.which = lambda n: _ip if n == "ip" else _orig_which(n)
    try:
        _cmd = RR.can_setup_command("can0")
    finally:
        RR.which = _orig_which
    _r = subprocess.run(_cmd[1:], capture_output=True, text=True)
    _calls = open(_log).read().splitlines()
    assert _r.returncode == 0, (_support, _r.stderr)
    assert _calls[-1] == "link set can0 up", _calls
    assert any(c == "link set can0 type can bitrate 1000000" for c in _calls) != _support, _calls
print("[R1b] gs_usb without restart-ms support: fallback to bitrate only, can0 UP (with support: restart-ms kept)")

# ---------------------------------------------------------------- R2
ARM_IDS = list(range(0x251, 0x258)) + list(range(0x261, 0x268)) + list(range(0x2A1, 0x2AA))


def frame(cid, data=b"\x00" * 8, flags=0):
    return struct.pack(RR.CAN_FRAME_FMT, cid | flags, len(data), data.ljust(8, b"\0"))


assert RR.parse_can_frame(frame(0x2A8, b"\x01\x02")) == (0x2A8, 2, b"\x01\x02")
m = RR.CanMonitor("can0")
t0 = 1000.0
for i in range(600):
    m.feed(frame(ARM_IDS[i % len(ARM_IDS)]), now=t0 + i / 600.0)
s = m.snapshot(now=t0 + 1.0)
assert 590 <= s["fps"] <= 600 and all(s["groups"].values()), s
assert RR.evaluate_flow(s)[0] == RR.GREEN
m.feed(frame(0x517), now=t0 + 1.0)
m.feed(frame(0x7FF, flags=RR.CAN_ERR_FLAG), now=t0 + 1.0)
s = m.snapshot(now=t0 + 1.0)
assert s["gripper_old_id"] == ["0x517"] and s["error_frames"] == 1 and RR.evaluate_flow(s)[0] == RR.AMBER
m2 = RR.CanMonitor("can0")
for i in range(3):
    m2.feed(frame(0x2A1), now=t0 + i * 0.3)
lamp, msg = RR.evaluate_flow(m2.snapshot(now=t0 + 1.0))
assert lamp == RR.YELLOW and "step 3" in msg
assert RR.evaluate_flow(RR.CanMonitor("can0").snapshot())[0] == RR.RED
assert RR.push_release_decision(800, 790) == "released" and RR.push_release_decision(800, 3) == "keep_until_driver"
print("[R2] CAN monitor: rate + groups -> green; sporadic -> yellow (push needed); 0x517 -> amber; error frames counted")

# ---------------------------------------------------------------- R3
cfg = {"can_port": "can0", "effector_type": "agx_gripper", "fw_version": "v111", "speed_percent": 80, "domain_id": 42}
a = RR.launch_args(cfg)
assert "arm_type:=nero" in a and "speed_percent:=30" in a and "fw_version:=v111" in a and "effector_type:=agx_gripper" in a
assert "speed_percent:=80" in RR.launch_args(dict(cfg, speed_unlocked=True))
assert "effector_type:=none" in RR.launch_args(dict(cfg, effector_type="piper"))
assert not any(x.startswith("fw_version") for x in RR.launch_args(dict(cfg, fw_version="")))
assert "auto_control_gate:=false" in a and "follow:=true" in a and "use_rviz:=true" in a
sh = RR.launch_shell(cfg)
assert "source /opt/ros/" in sh and "agx_arm_ws/install/setup.bash" in sh and "export ROS_DOMAIN_ID=42;" in sh
assert sh.endswith(RR.launch_command(cfg)) and "exec ros2 launch agx_arm_ctrl start_single_agx_arm_moveit.launch.py" in sh
print("[R3] launch: nero fixed, speed<=30 unless unlocked, effector whitelist, fw optional, ROS env + domain")

# ---------------------------------------------------------------- R4
procs = [(10, ["/opt/ros/humble/lib/moveit_ros_move_group/move_group", "--ros-args"]),
         (11, ["rviz2", "-d", "x.rviz"]), (12, ["/usr/bin/python3", "/opt/ros/humble/bin/ros2", "launch", "agx_arm_ctrl", "x"]),
         (13, ["python3", "/home/q/nero_tools/nero_start.py"]), (14, ["python3", "-u", "NERO_V313_Real_Robot.py", "push"]),
         (15, ["vim", "nero_start.py"]), (16, ["python3", "-u", "/x/NERO_V2_7_1_Unified_Backend.py"])]
stale = {p for p, _c in RR.find_stale(procs, me={14})}
assert stale == {10, 11, 12, 13, 16}, stale
assert RR.sim_running(procs) and not RR.sim_running(procs[:5])
print("[R4] leftovers detected (move_group, rviz2, ros2 launch, nero_start.py, sim backend); own helper + editor ignored")

# ---------------------------------------------------------------- R5 push helper vs fake pyAgxArm
FAKE = TMP / "fakepkgs"
(FAKE / "pyAgxArm").mkdir(parents=True)
(FAKE / "pyAgxArm" / "__init__.py").write_text(textwrap.dedent("""
    import os
    class ArmModel: NERO = "NERO"
    class NeroFW: V111 = "V111"
    def create_agx_arm_config(**kw):
        assert kw["robot"] == "NERO" and kw["firmeware_version"] == "V111" and kw["interface"] == "socketcan"
        return kw
    class _Arm:
        def __init__(self, cfg): self.n = 0; self.need = int(os.environ.get("FAKE_ENABLE_AFTER", "3"))
        def connect(self): open(os.environ["FAKE_MARK"], "a").write("connect\\n")
        def enable(self):
            self.n += 1
            return self.need >= 0 and self.n > self.need
        def set_normal_mode(self): open(os.environ["FAKE_MARK"], "a").write("normal\\n")
        def get_joint_angles(self): return [0.0, 0.1, -0.2, 0.3, 0.0, 0.0, 0.05]
        def get_arm_status(self): return {"ctrl_mode": 1, "err": 0}
    class AgxArmFactory:
        @staticmethod
        def create_arm(cfg): return _Arm(cfg)
"""))
mark = TMP / "mark.txt"
os.environ["FAKE_MARK"] = str(mark)
os.environ["PYTHONPATH"] = str(FAKE) + os.pathsep + os.environ.get("PYTHONPATH", "")
p = subprocess.Popen(RR.helper_argv("push", "--channel", "can0", "--fw", "v111", "--timeout", "5", "--period", "0.1"),
                     stdout=subprocess.PIPE, text=True, start_new_session=True)
evs = []
t_end = time.time() + 10
while time.time() < t_end:
    ln = p.stdout.readline()
    if not ln:
        break
    evs.append(json.loads(ln))
    if sum(e["event"] == "state" for e in evs) >= 2:
        break
os.killpg(p.pid, signal.SIGINT)
rest = p.communicate(timeout=5)[0]
evs += [json.loads(x) for x in rest.splitlines() if x.strip()]
names = [e["event"] for e in evs]
assert names[:3] == ["connecting", "connected", "enabled"] and names[-1] == "stopped" and p.returncode == 0, names
assert evs[2]["attempts"] == 4 and mark.read_text().count("normal") == 3
st_ev = [e for e in evs if e["event"] == "state"][0]
assert st_ev["joints"][2] == -0.2 and st_ev["status"]["ctrl_mode"] == 1
os.environ["FAKE_ENABLE_AFTER"] = "-1"                  # never enables
r = subprocess.run(RR.helper_argv("push", "--timeout", "0.4"), capture_output=True, text=True, timeout=10)
assert r.returncode == 2 and json.loads(r.stdout.splitlines()[-1])["event"] == "timeout"
os.environ["FAKE_ENABLE_AFTER"] = "2"
env_np = dict(os.environ, PYTHONPATH="", PYTHONNOUSERSITE="1")
r = subprocess.run(RR.helper_argv("push", "--timeout", "0.4"), capture_output=True, text=True, timeout=10, env=env_np)
assert r.returncode == 1 and "pyAgxArm import failed" in r.stdout
print("[R5] push helper: connect -> set_normal_mode x3 -> enabled, joints/status reports, SIGINT stop; timeout rc2; missing rc1")

# ---------------------------------------------------------------- R6 pipeline end-to-end with fakes
FAKE_DRIVER = TMP / "fake_driver.sh"
FAKE_DRIVER.write_text('#!/bin/bash\ntrap "echo driver-stopped; exit 0" INT TERM\n'
                       'echo "[INFO] [launch]: All log files can be found below /tmp"\n'
                       'if [ -n "$FAKE_DRIVER_FAIL" ]; then echo "[ERROR] [agx]: Failed to get firmware version"; '
                       'echo "[ERROR] [agx_arm_ctrl_single-1]: process has died"; exit 1; fi\n'
                       'while true; do sleep 0.1; done\n')
FAKE_MON = TMP / "fake_monitor.py"
FAKE_MON.write_text(textwrap.dedent("""
    import json, sys, threading, time, signal
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *a: stop.set())
    def rd():
        for line in sys.stdin:
            c = json.loads(line)
            if c.get("quit"): stop.set(); return
            if c.get("call"):
                print(json.dumps({"event": "call_result", "id": c["id"], "call": c["call"], "value": c.get("value"),
                                  "ok": True, "type": "std_srvs/srv/SetBool", "response": {"success": True}}), flush=True)
    threading.Thread(target=rd, daemon=True).start()
    print(json.dumps({"event": "monitor_ready"}), flush=True)
    t0 = time.time()
    while not stop.is_set():
        up = time.time() - t0
        print(json.dumps({"event": "status", "js_hz": 200.0 if up > 0.5 else 0.0,
                          "joints": {"names": ["joint%d" % i for i in range(1, 8)], "position": [0.0] * 7},
                          "arm_status": {"header": {"stamp": 1}, "ctrl_mode": 1, "err_code": 0},
                          "nodes": ["/agx_arm_ctrl_single_node", "/move_group"], "has_move_group": up > 1.0,
                          "has_move_action": up > 1.2,
                          "services": {"/emergency_stop": "std_srvs/srv/Trigger", "/control_enable": "std_srvs/srv/SetBool"}}),
              flush=True)
        stop.wait(0.2)
    print(json.dumps({"event": "monitor_stopped"}), flush=True)
"""))
RR.launch_shell = lambda cfg: f"exec bash {FAKE_DRIVER}"
RR.monitor_shell = lambda dom: f"exec {sys.executable} -u {FAKE_MON}"
RR.rviz_alive = lambda procs=None: True


class FakeSock:
    """~400-900 frames/s of NERO arm ids while source() is true, else silence (timeouts)."""
    def __init__(self, source):
        self.source, self.i = source, 0

    def recv(self, n):
        if not self.source():
            time.sleep(0.05)
            raise socket.timeout()
        time.sleep(0.001)
        self.i += 1
        return frame(ARM_IDS[self.i % len(ARM_IDS)])

    def close(self):
        pass


def make_pipe(source_mode, sim=False):
    box = {}
    port = {"up": False, "setup_calls": 0}

    def port_status(p):
        return (dict(st) if port["up"] else dict(st, up=False, operstate="DOWN")), ""

    def setup_runner(cmd, timeout=0):
        assert cmd[0] == "pkexec"
        port["setup_calls"] += 1
        port["up"] = True
        return 0, "", ""

    def source():
        pipe = box["pipe"]
        if source_mode == "keep":
            return pipe.push.running() or pipe.driver.running()
        if source_mode == "released":
            box.setdefault("pushed_once", False)
            if pipe.push_info.get("enabled"):
                box["pushed_once"] = True
            return box["pushed_once"]
        return False

    pipe = RR.RealPipeline(lambda: {"can_port": "can0", "effector_type": "agx_gripper", "fw_version": "v111",
                                    "push_fw": "v111", "speed_percent": 10, "domain_id": 42, "use_rviz": True},
                           sim_active_fn=lambda: sim, confirm_fn=lambda t, x: True,
                           can_opener=lambda p: FakeSock(source),
                           env_checker=lambda sim_runner_active=False, me=None: (RR.GREEN, [("fake", True, "ok")], []),
                           port_status_fn=port_status, setup_runner=setup_runner)
    box["pipe"] = pipe
    return pipe, port


def logs(pipe):
    out = []
    while not pipe.events.empty():
        out.append(pipe.events.get())
    return out


os.environ["FAKE_ENABLE_AFTER"] = "2"
pipe, port = make_pipe("keep")
assert pipe.run_step("push") is None                          # previous steps not green -> refused
ok = pipe.run_all()
L = logs(pipe)
assert ok, [x for x in L if "FAIL" in x][-5:]
assert all(pipe.lamp(k) == RR.GREEN for k in RR.STEP_KEYS), pipe.state
assert port["setup_calls"] == 1 and pipe.push_mode == "keep_until_driver"
assert not pipe.push.running() and pipe.driver.running() and pipe.monitor.running()
assert any("=> keep_until_driver" in x for x in L) and any("stopping push helper" in x for x in L)
assert any("ROS_DOMAIN_ID=42" in x and "arm_type:=nero" in x for x in L)
assert pipe.run_all() and port["setup_calls"] == 1             # already up: no second password prompt
# R7 services
cid = pipe.call_service("control_enable", True)
time.sleep(0.6)
assert cid and pipe.call_results.get(cid, {}).get("ok"), pipe.call_results
assert pipe.call_service("move_home") is None                  # service not offered by this driver
assert pipe.stop_all() and not pipe.active()
assert all(pipe.lamp(k) == RR.GREY for k in ("push", "driver", "moveit"))
assert pipe.call_service("emergency_stop") is None             # no driver -> refused (GUI tells: Web e-stop / power)
pipe.shutdown()
print("[R6] one-click 6 green; release test -> keep_until_driver, push stopped after driver feedback; stop_all clean")
print("[R7] services via monitor by suffix; missing service / no driver refused")

pipe, port = make_pipe("released")
assert pipe.run_all(), [x for x in logs(pipe) if "FAIL" in x][-5:]
assert pipe.push_mode == "released" and not pipe.push.running() and pipe.lamp("moveit") == RR.GREEN
pipe.stop_all()
pipe.shutdown()

os.environ["FAKE_DRIVER_FAIL"] = "1"
pipe, port = make_pipe("keep")
assert not pipe.run_all()
assert pipe.lamp("driver") == RR.RED and "firmware" in pipe.state["driver"][1], pipe.state["driver"]
pipe.stop_all()
pipe.shutdown()
os.environ.pop("FAKE_DRIVER_FAIL")

pipe, port = make_pipe("keep", sim=True)
assert pipe.run_step("env") == RR.RED and "simulation is running" in pipe.state["env"][1]
pipe.shutdown()
print("[R6] 'released' arm -> driver without push; firmware-version failure -> red; sim running -> locked")

# ---------------------------------------------------------------- R8 console wiring
C = (ROOT / "NERO_V312_Console.py").read_text()
assert "import NERO_V313_Real_Robot as RR" in C and "RR.RealPipeline(" in C and "V3.13 REAL TAB" in C
assert "sim_active_fn=lambda: runner.running()" in C and "if pipe.active():" in C and 'pipe.call_service("emergency_stop")' in C
assert '"去使能⚠"' in C and "pipe.stop_all" in C and any(f'VERSION = "{v}' in C for v in ("V3.13", "V3.14", "V3.15", "V3.16", "V3.17", "V3.18", "V3.19", "V3.20", "V3.21", "V3.22", "V3.23", "V3.24", "V3.25", "V3.26", "V3.27", "V3.28", "V3.29", "V3.30", "V3.31", "V3.32", "V3.33", "V3.34", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1"))
E = (ROOT / "start_nero.sh").read_text()
assert "NERO_V313_Real_Robot.py" in E
assert RR.compact_status({"header": {"stamp": 1}, "ctrl_mode": 1, "err": {"a": 0}, "q": [1, 2]}) == "ctrl_mode=1  err.a=0  q=[1, 2]"
print("[R8] console: real tab -> pipeline, sim/real interlock both ways, e-stop, confirmations; entry checks helper file")
# ---------------------------------------------------------------- R9 monitor helper against a fake rclpy
FR = TMP / "fakeros"
for d in ("rclpy", "rosidl_runtime_py"):
    (FR / d).mkdir(parents=True, exist_ok=True)
(FR / "rclpy" / "__init__.py").write_text(textwrap.dedent("""
    import time
    _ok = [True]
    def init(): pass
    def ok(): return _ok[0]
    def shutdown(): _ok[0] = False
    def spin_once(node, timeout_sec=0.0):
        node._tick(); time.sleep(timeout_sec)
    def spin_until_future_complete(node, fut, timeout_sec=0.0): fut._done = True
"""))
(FR / "rclpy" / "qos.py").write_text(textwrap.dedent("""
    class QoSProfile:
        def __init__(self, **kw): self.kw = kw
    class ReliabilityPolicy: BEST_EFFORT = 1
    class HistoryPolicy: KEEP_LAST = 1
"""))
(FR / "rclpy" / "node.py").write_text(textwrap.dedent("""
    import time
    class _Msg:
        def __init__(self, **kw): self.__dict__.update(kw)
    class _Fut:
        def __init__(self, r): self._r, self._done = r, True        # V3.19: calls are polled, not spun
        def done(self): return self._done
        def result(self): return self._r
    class _Cli:
        def __init__(self, name): self.name = name
        def wait_for_service(self, timeout_sec=0): return True
        def call_async(self, req): return _Fut(_Msg(success=True, message="ok:" + str(getattr(req, "data", None))))
    class Node:
        def __init__(self, name): self.subs = []
        def create_subscription(self, mt, topic, cb, qos): self.subs.append((topic, cb)); return object()
        def get_topic_names_and_types(self):
            return [("/feedback/joint_states", ["sensor_msgs/msg/JointState"]), ("/feedback/arm_status", ["agx/msg/ArmStatus"])]
        def get_node_names_and_namespaces(self): return [("move_group", "/"), ("agx_arm_ctrl_single_node", "/")]
        def get_service_names_and_types(self):
            return [("/emergency_stop", ["std_srvs/srv/Trigger"]), ("/control_enable", ["std_srvs/srv/SetBool"]),
                    ("/move_action/_action/send_goal", ["moveit_msgs/action/MoveGroup_SendGoal"])]
        def create_client(self, st, name): return _Cli(name)
        def destroy_client(self, c): pass
        def destroy_node(self): pass
        def _tick(self):
            for topic, cb in self.subs:
                if topic.endswith("joint_states"):
                    cb(_Msg(name=["joint%d" % i for i in range(1, 8)], position=[0.1] * 7))
                else:
                    cb(_Msg(ctrl_mode=1, err_code=0))
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
(FR / "rosidl_runtime_py" / "convert.py").write_text(textwrap.dedent("""
    def message_to_ordereddict(m): return dict(m.__dict__)
"""))
envm = dict(os.environ, PYTHONPATH=str(FR))
p = subprocess.Popen(RR.helper_argv("monitor", "--period", "0.3"), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     text=True, env=envm)
got = []
t_end = time.time() + 10
sent = False
while time.time() < t_end:
    ln = p.stdout.readline()
    if not ln:
        break
    ev = json.loads(ln)
    got.append(ev)
    if ev["event"] == "status" and not sent and ev.get("js_hz"):
        p.stdin.write(json.dumps({"call": "/control_enable", "value": True, "id": 7}) + "\n")
        p.stdin.flush()
        sent = True
    if ev["event"] == "call_result":
        p.stdin.write(json.dumps({"quit": True}) + "\n")
        p.stdin.flush()
    if ev["event"] == "monitor_stopped":
        break
p.wait(timeout=5)
stat = [e for e in got if e["event"] == "status" and e.get("js_hz")][-1]
assert stat["has_move_group"] and stat["has_move_action"] and stat["joints"]["names"][0] == "joint1", stat
assert stat["arm_status"] == {"ctrl_mode": 1, "err_code": 0} and set(stat["services"]) == {"/emergency_stop", "/control_enable"}
cr = [e for e in got if e["event"] == "call_result"][0]
assert cr["ok"] and cr["id"] == 7 and cr["type"] == "std_srvs/srv/SetBool" and cr["response"]["message"] == "ok:True", cr
assert got[-1]["event"] == "monitor_stopped" and p.returncode == 0
r = subprocess.run(RR.helper_argv("monitor"), capture_output=True, text=True, timeout=10,
                   env=dict(os.environ, PYTHONPATH="", PYTHONNOUSERSITE="1"))
assert r.returncode == 1 and "rclpy unavailable" in r.stdout
print("[R9] ROS monitor helper (fake rclpy): rates/joints/arm_status/move_group/action/services; SetBool call; quit; no-ROS error")
shutil.rmtree(TMP, ignore_errors=True)
print("ALL V3.13 TESTS PASS")
