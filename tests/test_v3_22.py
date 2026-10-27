#!/usr/bin/env python3
"""V3.22: hand guiding on NERO V111 where ctrl_mode keeps reporting 0x01 (seen on the real arm), slow exit.

L1 mode helper (fake pyAgxArm, ctrl_mode stuck at 1): entering is confirmed by leader angles that start updating;
   no leader angles at all -> still a failure
L2 mode helper normal with --hold: speed 10 % and move_j(current pose) are sent BEFORE enable() (no jump to zero)
L3 pipeline: helper says "failed" but /feedback/leader_joint_states streams and joint_states is silent -> teach mode
   accepted, the gate stays closed; without the teach flag the leader topic is still used (corner recording works)
L4 exit: gate closed, ROS controller synced to the pose, helper gets --hold, gate reopened, MoveIt home at 10 %
"""
import json, os, subprocess, sys, tempfile, textwrap, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V313_Real_Robot as RR  # noqa: E402
import NERO_V314_Joint_Control as JC  # noqa: E402

assert RR.VERSION.startswith(("V3.2", "V3.3", "V3.4"))
TMP = Path(tempfile.mkdtemp(prefix="v322_"))
CALLS = TMP / "calls.txt"
FAKE = TMP / "fake"
(FAKE / "pyAgxArm").mkdir(parents=True)
(FAKE / "pyAgxArm" / "__init__.py").write_text(textwrap.dedent(f"""
    import os, time
    LOG = {str(CALLS)!r}
    def rec(s):
        open(LOG, "a").write(s + "\\n")
    class ArmModel: NERO = "NERO"
    class NeroFW: V111 = "V111"
    def create_agx_arm_config(**kw): return kw
    class _Msg:
        def __init__(self, v): self._msg = v; self.timestamp = time.time()
    class _Arm:
        leader = False
        def connect(self): pass
        def enable(self): rec("enable"); return True
        def set_normal_mode(self): rec("set_normal_mode"); _Arm.leader = False
        def set_leader_mode(self): rec("set_leader_mode"); _Arm.leader = os.environ.get("FAKE_NO_LEADER") != "1"
        def get_arm_status(self): return {{"ctrl_mode": 1}}             # V111: never changes
        def get_leader_joint_angles(self): return _Msg([0.1] * 7) if _Arm.leader else None
        def set_speed_percent(self, p): rec(f"speed {{p}}")
        def move_j(self, q): rec("move_j " + ",".join(f"{{v:.3f}}" for v in q))
    class AgxArmFactory:
        @staticmethod
        def create_arm(cfg): return _Arm()
"""))
os.environ["PYTHONPATH"] = str(FAKE) + os.pathsep + os.environ.get("PYTHONPATH", "")


def helper(*a, env=None):
    r = subprocess.run(RR.helper_argv("mode", *a), capture_output=True, text=True, timeout=30,
                       env=dict(os.environ, **(env or {})))
    return [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]


ev = helper("--mode", "teach")[-1]
assert ev["ok"] and ev["ctrl_mode"] == 1 and "主导臂角度" in ev["evidence"], ev
ev = helper("--mode", "teach", env={"FAKE_NO_LEADER": "1"})[-1]
assert not ev["ok"], ev
print("[L1] ctrl_mode stuck at 0x01: entry confirmed by streaming leader angles; nothing streaming -> failure")

CALLS.write_text("")
evs = helper("--mode", "normal", "--hold", json.dumps([0.16, 1.2, 0, 0.37, 0, 0.03, 1.5]), "--speed", "10")
c = CALLS.read_text().split("\n")
i_en = c.index("enable")
assert "speed 10" in c[:i_en] and any(x.startswith("move_j 0.160,1.200") for x in c[:i_en]), c
assert evs[-1]["ok"] and evs[-1]["movej_api"] == "move_j" and evs[-1]["speed_api"] == "set_speed_percent"
print("[L2] exit: speed 10 % + move_j(current pose) sent before enable() -> firmware holds the pose")

M = JC.build_model((ROOT / "config" / "nero.urdf").read_text(), None, None)
logs = []
pipe = RR.RealPipeline(lambda: {"can_port": "can0", "push_fw": "v111"},
                       can_opener=lambda p: (_ for _ in ()).throw(OSError("x")))
pipe.log = logs.append
sent, gates = [], []
pipe.service_name = lambda sfx: "/" + sfx if sfx == "control_enable" else None
pipe.monitor.running = lambda: True


def fake_send(msg):
    sent.append(msg)
    if msg.get("call"):
        gates.append(msg["value"])
        pipe.call_results[msg["id"]] = {"ok": True, "response": {"success": True}}
    if msg.get("hold"):
        pipe.call_results[("hold", msg["hold"]["id"])] = {"ok": True, "topic": "/arm_controller/joint_trajectory"}


pipe.monitor.send = fake_send


def fake_call(sfx, value=None, timeout_s=10.0):
    pipe._call_id += 1
    fake_send({"call": "/" + sfx, "value": value, "id": pipe._call_id})
    return pipe._call_id


pipe.call_service = fake_call
QL = [0.159, 1.2064, -0.0001, 0.3673, 0.0025, 0.0293, 1.6555]
pipe.last_js, pipe.last_js_t = {"names": M["joints"], "position": [0.0] * 7}, time.time() - 30   # joint_states stopped
stop = {"v": False}
import threading


def ljs_pub():                                      # driver keeps publishing /feedback/leader_joint_states
    while not stop["v"]:
        pipe.last_ljs, pipe.last_ljs_t = {"names": M["joints"], "position": list(QL)}, time.time()
        time.sleep(0.05)


os.environ["FAKE_NO_LEADER"] = "1"                 # pyAgxArm sees nothing -> only the ROS topic proves it
threading.Thread(target=ljs_pub, daemon=True).start()
ev = pipe.set_teach_mode(True)
assert ev["ok"] and pipe.teach_mode and "leader_joint_states" in ev["evidence"], ev
assert gates == [False], gates                      # closed, never reopened
a = pipe.actual_arm_deg(M)
assert a and abs(a[1] - 69.122) < 0.01, a
pipe.teach_mode = False                             # as in the user's log: console thinks "normal"
assert pipe.leader_active() and "检测到手动拖动" in pipe.feedback_source()[0] and pipe.actual_arm_deg(M)
pipe.teach_mode = True
print("[L3] helper failed but leader topic streams + joint_states silent -> teach accepted, gate stays closed; "
      "leader topic also used without the teach flag")

moves = []
pipe.move_and_settle = lambda model, tgt, speed, label="move", **k: moves.append((tgt, speed, label)) or {"ok": True}
_orig_run = pipe._run_mode_helper
hold_seen = {}


def run_helper(on, timeout_s, hold=None):
    hold_seen["q"] = hold
    stop["v"] = True
    time.sleep(0.1)
    pipe.last_js, pipe.last_js_t = {"names": M["joints"], "position": list(hold)}, time.time()
    threading.Thread(target=lambda: [setattr(pipe, "last_js_t", time.time()) or time.sleep(0.05) for _ in range(200)],
                     daemon=True).start()
    return _orig_run(on, timeout_s, hold=hold)


pipe._run_mode_helper = run_helper
os.environ.pop("FAKE_NO_LEADER")
ev = pipe.set_teach_mode(False, home_model=M)
assert ev["ok"] and not pipe.teach_mode, ev
q = hold_seen["q"]
assert abs(q[1] - 1.2064) < 1e-6 and abs(q[6] - (M["limits"][6][1] - 0.01)) < 1e-6, q        # J7 clamped into limit
holds = [m["hold"]["positions"] for m in sent if m.get("hold")]
assert len(holds) == 2 and abs(holds[0][1] - 1.2064) < 1e-6, holds
assert gates == [False, False, True], gates
assert moves == [([0.0] * 7, 0.10, "退出示教回零")], moves
print("[L4] exit: gate closed -> ROS controller synced -> helper --hold (J7 clamped into its limit) -> gate open -> "
      "MoveIt home at 10 %")
print("ALL V3.22 TESTS PASS")
