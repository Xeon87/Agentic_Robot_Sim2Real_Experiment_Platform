#!/usr/bin/env python3
"""V3.20: hand-guiding (teach) mode done right (no ROS, no Tk, no robot).

T1 helper prefers set_leader_mode (pyAgxArm nero_api) over set_master_mode, prints the available mode APIs, and
   VERIFIES ctrl_mode (0x06 / 0x02): unchanged ctrl_mode -> failure with the value (V3.19 reported success)
T2 ctrl_mode parsing: int, '0x06', enum-like strings, nested status dicts
T3 pipeline: driver control gate closed before entering, reopened after leaving and after a failed entry
T4 while teaching, joint angles come from /feedback/leader_joint_states (monitor subscribes it)
"""
import json, os, subprocess, sys, tempfile, textwrap, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V313_Real_Robot as RR  # noqa: E402

assert RR.VERSION.startswith(("V3.20", "V3.21", "V3.22", "V3.23", "V3.24", "V3.25", "V3.26", "V3.27", "V3.28", "V3.29", "V3.30", "V3.31", "V3.32", "V3.33", "V3.34", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1"))
TMP = Path(tempfile.mkdtemp(prefix="v320_"))
FAKE = TMP / "fake"
(FAKE / "pyAgxArm").mkdir(parents=True)
(FAKE / "pyAgxArm" / "__init__.py").write_text(textwrap.dedent("""
    import os
    class ArmModel: NERO = "NERO"
    class NeroFW: V111 = "V111"
    def create_agx_arm_config(**kw): return kw
    def _m(x): open(os.environ["FAKE_MARK"], "a").write(x + "\\n")
    class _Status:
        def __init__(self, cm): self.ctrl_mode = cm; self.arm_status = 0
    class _Arm:
        cm = 1
        def connect(self): _m("connect")
        def enable(self): _m("enable"); return True
        def set_normal_mode(self): _m("normal"); _Arm.cm = 1
        def set_master_mode(self): _m("master")
        def set_follower_mode(self): _m("follower")
        def get_arm_status(self):
            return _Status(os.environ.get("FAKE_CM_FMT", "{}").format(_Arm.cm) if os.environ.get("FAKE_CM_FMT") else _Arm.cm)
    if os.environ.get("FAKE_HAS_LEADER", "1") == "1":
        def _leader(self):
            _m("leader")
            if os.environ.get("FAKE_LEADER_WORKS", "1") == "1": _Arm.cm = 6
        _Arm.set_leader_mode = _leader
    class AgxArmFactory:
        @staticmethod
        def create_arm(cfg): return _Arm()
"""))
mark = TMP / "mark.txt"
os.environ["FAKE_MARK"] = str(mark)
os.environ["PYTHONPATH"] = str(FAKE) + os.pathsep + os.environ.get("PYTHONPATH", "")


def helper(mode, **env):
    mark.write_text("")
    r = subprocess.run(RR.helper_argv("mode", "--mode", mode), capture_output=True, text=True, timeout=40,
                       env=dict(os.environ, **env))
    evs = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
    return r.returncode, {e["event"]: e for e in evs}, mark.read_text().split()


rc, ev, calls = helper("teach")
assert rc == 0 and ev["mode_result"]["ok"] and ev["mode_result"]["api"] == "set_leader_mode" and ev["mode_result"]["ctrl_mode"] == 6
assert "leader" in calls and "master" not in calls and "set_leader_mode" in ev["diag"]["apis"] and ev["diag"]["ctrl_mode_before"] == 1
rc, ev, calls = helper("teach", FAKE_LEADER_WORKS="0")
assert rc == 2 and not ev["mode_result"]["ok"] and "ctrl_mode = 1" in ev["mode_result"]["error"], ev
rc, ev, calls = helper("teach", FAKE_HAS_LEADER="0")
assert ev["mode_result"]["api"] == "set_master_mode" and not ev["mode_result"]["ok"]           # old API, no effect
rc, ev, calls = helper("normal")
assert rc == 0 and ev["mode_result"]["ok"] and ev["mode_result"]["api"] == "set_normal_mode" and calls[-1] == "enable"
print("[T1] helper: set_leader_mode preferred, diag APIs, ctrl_mode verified (no change -> failure), normal + enable")

assert RR.ctrl_mode_of({"ctrl_mode": 6}) == 6 and RR.ctrl_mode_of({"a": {"ctrl_mode": "0x02"}}) == 2
assert RR.ctrl_mode_of({"ctrl_mode": "CtrlMode.LINKAGE_TEACHING_INPUT_MODE"}) == 6
assert RR.ctrl_mode_of({"ctrl_mode": "TEACHING_MODE"}) == 2 and RR.ctrl_mode_of({"x": 1}) is None
assert RR.is_teach_ctrl_mode(6) and RR.is_teach_ctrl_mode(2) and not RR.is_teach_ctrl_mode(1)
rc, ev, _c = helper("teach", FAKE_CM_FMT="CtrlMode(0x0{})")
assert ev["mode_result"]["ok"] and ev["mode_result"]["ctrl_mode"] == 6
print("[T2] ctrl_mode parsing: int / hex string / enum names / nested")

pipe = RR.RealPipeline(lambda: {"can_port": "can0", "push_fw": "v111"}, can_opener=lambda p: (_ for _ in ()).throw(OSError("x")))
sent = []
pipe.mon_status = {"services": {"/control_enable": "std_srvs/srv/SetBool", "/exit_teach_mode": "std_srvs/srv/Empty"}}
pipe.monitor.running = lambda: True
pipe.monitor.send = lambda o: (sent.append((o.get("call"), o.get("value"))), pipe.call_results.__setitem__(o["id"], {"ok": True, "response": {}}))
mark.write_text("")
ev = pipe.set_teach_mode(True)
assert ev["ok"] and pipe.teach_mode and sent == [("/control_enable", False)], sent
sent.clear()
ev = pipe.set_teach_mode(False)
assert ev["ok"] and not pipe.teach_mode and sent == [("/control_enable", False), ("/exit_teach_mode", None), ("/control_enable", True)], sent   # V3.22: gate closed first
sent.clear()
os.environ["FAKE_LEADER_WORKS"] = "0"
ev = pipe.set_teach_mode(True)
os.environ.pop("FAKE_LEADER_WORKS")
assert not ev["ok"] and not pipe.teach_mode and sent == [("/control_enable", False), ("/control_enable", True)], sent
print("[T3] control gate: closed before entering, reopened after leaving and after a failed entry")

import NERO_V314_Joint_Control as JC  # noqa: E402
M = JC.build_model((ROOT / "config" / "nero.urdf").read_text(), None, None)
now = time.time()
pipe.last_js, pipe.last_js_t = {"names": M["joints"], "position": [0.0] * 7}, now
pipe.last_ljs, pipe.last_ljs_t = {"names": [f"l{i}" for i in range(7)], "position": [0.1] * 7}, now
assert pipe.actual_arm_deg(M)[0] == 0.0                       # not teaching: follower feedback
pipe.teach_mode = True
assert abs(pipe.actual_arm_deg(M)[0] - 5.7296) < 1e-3 and pipe.feedback_js()[0] is pipe.last_ljs
pipe.last_ljs_t = now - 5                                      # stale leader topic -> back to joint_states
assert pipe.actual_arm_deg(M)[0] == 0.0
pipe.shutdown()
S = (ROOT / "NERO_V313_Real_Robot.py").read_text(encoding="utf-8")
assert '"/feedback/leader_joint_states", "ljs"' in S and '_emit(event="ljs"' in S
print("[T4] teaching: angles from /feedback/leader_joint_states (fresh), else joint_states")
print("ALL V3.20 TESTS PASS")
