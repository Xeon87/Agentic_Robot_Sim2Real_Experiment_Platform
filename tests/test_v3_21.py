#!/usr/bin/env python3
"""V3.21: recording teach corners while hand guiding (no ROS, no Tk, no robot).

L1 leader helper (fake pyAgxArm): streams get_leader_joint_angles() (message object with ._msg) + get_joint_angles()
L2 pipeline: entering teach mode starts the leader reader; with /feedback/joint_states silent (V111 leader mode),
   actual angles come from the pyAgxArm leader angles; priority ROS leader topic > pyAgxArm leader > joint_states;
   leaving teach mode stops the reader
L3 diagnostics text names the age of every source (shown when a corner cannot be recorded)
"""
import json, os, subprocess, sys, tempfile, textwrap, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V313_Real_Robot as RR  # noqa: E402
import NERO_V314_Joint_Control as JC  # noqa: E402

assert RR.VERSION.startswith(("V3.2", "V3.3", "V3.4"))
TMP = Path(tempfile.mkdtemp(prefix="v321_"))
FAKE = TMP / "fake"
(FAKE / "pyAgxArm").mkdir(parents=True)
(FAKE / "pyAgxArm" / "__init__.py").write_text(textwrap.dedent("""
    class ArmModel: NERO = "NERO"
    class NeroFW: V111 = "V111"
    def create_agx_arm_config(**kw): return kw
    class _Msg:
        def __init__(self, v): self._msg = v; self._msg_type = "ArmMsgFeedbackJointStates"
    class _Arm:
        cm = 1
        def connect(self): pass
        def enable(self): return True
        def set_normal_mode(self): _Arm.cm = 1
        def set_leader_mode(self): _Arm.cm = 6
        def get_arm_status(self): return {"ctrl_mode": _Arm.cm}
        def get_leader_joint_angles(self): return _Msg([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        def get_joint_angles(self): return _Msg([0.0] * 7)
    class AgxArmFactory:
        @staticmethod
        def create_arm(cfg): return _Arm()
"""))
os.environ["PYTHONPATH"] = str(FAKE) + os.pathsep + os.environ.get("PYTHONPATH", "")
r = subprocess.run(RR.helper_argv("leader", "--period", "0.02", "--max-reports", "3"), capture_output=True, text=True, timeout=20)
evs = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
assert evs[0]["event"] == "leader_ready" and evs[0]["has_leader_api"]
assert evs[1]["leader"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] and evs[1]["joints"] == [0.0] * 7 and len(evs) == 4
assert RR._angles7({"_msg": [1, 2, 3, 4, 5, 6, 7]}) == [1, 2, 3, 4, 5, 6, 7] and RR._angles7([1, 2]) is None
print("[L1] leader helper streams pyAgxArm leader angles (message objects unwrapped) + follower angles")

M = JC.build_model((ROOT / "config" / "nero.urdf").read_text(), None, None)
pipe = RR.RealPipeline(lambda: {"can_port": "can0", "push_fw": "v111"}, can_opener=lambda p: (_ for _ in ()).throw(OSError("x")))
pipe.last_js, pipe.last_js_t = {"names": M["joints"], "position": [0.0] * 7}, time.time() - 30   # joint_states silent
assert pipe.actual_arm_deg(M) is None
ev = pipe.set_teach_mode(True)
assert ev["ok"] and pipe.teach_mode and pipe.leader.running()
t_end = time.time() + 10
while time.time() < t_end and pipe.last_pleader is None:
    time.sleep(0.05)
a = pipe.actual_arm_deg(M)
assert a and abs(a[0] - 5.7296) < 1e-3 and abs(a[6] - 40.107) < 1e-2, a
assert pipe.feedback_source()[0] == "pyAgxArm get_leader_joint_angles"
pipe.last_ljs, pipe.last_ljs_t = {"names": M["joints"], "position": [0.05] * 7}, time.time()
assert pipe.feedback_source()[0].startswith("ROS /feedback/leader")
pipe.last_ljs_t = time.time() - 10
d = pipe.feedback_diag()
assert "joint_states 3" in d and "leader_joint_states 10" in d and "pyAgxArm 主导臂 0." in d, d
ev = pipe.set_teach_mode(False)
time.sleep(0.5)
assert ev["ok"] and not pipe.teach_mode and not pipe.leader.running()
assert pipe.feedback_source()[0] == "ROS /feedback/joint_states"
pipe.shutdown()
print("[L2] teach mode starts the pyAgxArm leader reader; silent joint_states -> leader angles; ROS leader topic wins; "
      "exit stops the reader")
print("[L3] diagnostics: " + d)
print("ALL V3.21 TESTS PASS")
