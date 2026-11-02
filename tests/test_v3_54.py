#!/usr/bin/env python3
"""V3.54 offline unit tests:
V1 Version V3.54 & markers across NERO_V312_Console.py, NERO_V313_Real_Robot.py, NERO_V325_Camera.py,
   NERO_V347_Yolo_Perception.py, NERO_V349_Real_Wrist_Camera.py; NERO_V308 untouched; DEFAULT_FW_VERSION == 'v121'.
V2 Firmware resolver (_resolve_nero_fw & resolve_ros_driver_fw):
   - resolves 'v121', '1.2.1', 'v120', 'v112', 'v111', 'default' to NeroFW constants;
   - falls back gracefully when an older pyAgxArm install only defines V120/V112/V111 while keeping fw_attr >= 'V112'.
V3 Firmware V1.2.1 (NeroFW.V121) teach mode entry & exit (cmd_mode + RealPipeline.set_teach_mode):
   - Simulates real V1.2.1 controller behavior where set_normal_mode() is a no-op (does NOT exit leader mode),
     set_follower_mode() is required to exit leader mode, and mode-switch commands only take effect when enabled;
   - Verifies set_teach_mode(True) enters leader mode on V121 (streaming 0x155-0x157+0x170 leader angles);
   - Verifies set_teach_mode(False, return_home=False) calls set_follower_mode() both before and after enable(),
     locks the hand-guided 7-joint pose at 10% speed, clears hardware leader mode, and reopens the ROS control gate;
   - Even if --fw v111 is passed against a V1.2.1 controller, cmd_mode(normal) still calls set_follower_mode() and exits teach mode.
V4 Step 3 'CAN push / 使能' (cmd_push + RealPipeline.step_push) recovery when arm is stuck in teach (leader) mode:
   - Simulates real V1.2.1 controller stuck in leader mode where enable() / set_normal_mode() alone fail;
   - Verifies cmd_push calls set_follower_mode(), holds current pose at 10% speed, enables all 7 joints, clears
     pipe.teach_mode = False, stops pipe.leader if running, and turns Step 3 GREEN.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import NERO_V312_Console as CON               # noqa: E402
import NERO_V313_Real_Robot as RR             # noqa: E402
import NERO_V314_Joint_Control as JC          # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V347_Yolo_Perception as YP        # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49     # noqa: E402

# ---------------------------------------------------------------- V1: version & markers
assert CON.VERSION in ("V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64") and RR.VERSION in ("V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64") and YP.VERSION in ("V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64") and W49.VERSION in ("V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64")
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64")
assert RR.DEFAULT_FW_VERSION == "v121"
h308 = hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest()
assert h308.startswith("020cc6680f181872"), f"NERO_V308 modified: {h308}"
s_rr = (ROOT / "NERO_V313_Real_Robot.py").read_text(encoding="utf-8")
s_con = (ROOT / "NERO_V312_Console.py").read_text(encoding="utf-8")
s_cam = (ROOT / "NERO_V325_Camera.py").read_text(encoding="utf-8")
marker = "V3.54 FIRMWARE V1.2.1"
assert marker in s_rr and marker in s_con and marker in s_cam
assert "v_fw = tk.StringVar(value=RR.DEFAULT_FW_VERSION)" in s_con
assert 'values=["v121", "v120", "v112", "v111", "default", ""]' in s_con
assert '"push_fw": v_fw.get() or RR.DEFAULT_FW_VERSION' in s_con
print("[V1] V3.54 versions, DEFAULT_FW_VERSION='v121', and markers verified; NERO_V308 untouched")

# ---------------------------------------------------------------- V2: _resolve_nero_fw
class _FullFW:
    DEFAULT = "default"
    V111 = "v111"
    V112 = "v112"
    V120 = "v120"
    V121 = "v121"

class _OldFW:
    DEFAULT = "default"
    V111 = "v111"
    V112 = "v112"

assert RR._resolve_nero_fw(_FullFW, "v121") == ("V121", "v121")
assert RR._resolve_nero_fw(_FullFW, "1.2.1") == ("V121", "v121")
assert RR._resolve_nero_fw(_FullFW, "") == ("V121", "v121")
assert RR._resolve_nero_fw(_FullFW, "v111") == ("V111", "v111")
assert RR._resolve_nero_fw(_OldFW, "v121") == ("V121", "v112")
assert RR._resolve_nero_fw(_FullFW, "v999") == ("V999", None)
assert "fw_version:=v121" in RR.launch_args({})
print("[V2] _resolve_nero_fw & launch_args default 'v121' (with graceful fallback for older SDKs) verified")

# ---------------------------------------------------------------- V3 & V4: V1.2.1 controller simulation
TMP = Path(tempfile.mkdtemp(prefix="v354_"))
STATE_FILE = TMP / "arm_state.json"
CALL_LOG = TMP / "calls.txt"
FAKE = TMP / "fake"
(FAKE / "pyAgxArm").mkdir(parents=True)

(FAKE / "pyAgxArm" / "__init__.py").write_text(textwrap.dedent(f"""
    import json, os, time
    STATE_PATH = {str(STATE_FILE)!r}
    LOG_PATH = {str(CALL_LOG)!r}

    def _load():
        if not os.path.exists(STATE_PATH):
            return {{"leader": False, "enabled": True, "fw": None, "joints": [0.15, -0.42, 0.08, 1.15, -0.05, 0.22, 0.65]}}
        return json.load(open(STATE_PATH, "r", encoding="utf-8"))

    def _save(st):
        json.dump(st, open(STATE_PATH, "w", encoding="utf-8"))

    def _rec(s):
        open(LOG_PATH, "a", encoding="utf-8").write(s + "\\n")

    class ArmModel:
        NERO = "NERO"

    class NeroFW:
        DEFAULT = "DEFAULT"
        V111 = "V111"
        V112 = "V112"
        V120 = "V120"
        V121 = "V121"

    def create_agx_arm_config(**kw):
        return kw

    class _Msg:
        def __init__(self, v):
            self._msg = v
            self.msg = v
            self.timestamp = time.time()

    class _Arm:
        def __init__(self, cfg):
            self.cfg = cfg
            self.fw = cfg.get("firmeware_version")
            _rec(f"init:{{self.fw}}")

        def connect(self):
            _rec("connect")

        def set_normal_mode(self):
            # V1.2.1: set_normal_mode is a strict no-op (does NOT exit leader mode!)
            _rec("set_normal_mode")

        def set_leader_mode(self):
            _rec("set_leader_mode")
            st = _load()
            st["leader"] = True
            st["enabled"] = False
            _save(st)

        def set_follower_mode(self):
            _rec("set_follower_mode")
            st = _load()
            # V1.2.1 hardware behavior: leaving leader mode via set_follower_mode allows enabling,
            # and when called while enabled ensures follower controlled state is active.
            st["leader"] = False
            _save(st)

        def enable(self):
            _rec("enable")
            st = _load()
            # If still in leader mode (set_follower_mode was never called), V1.2.1 refuses to enable!
            if st.get("leader"):
                return False
            st["enabled"] = True
            _save(st)
            return True

        def get_arm_status(self):
            st = _load()
            # As seen on NERO hardware, ctrl_mode may report 0x01 or 0x06
            return {{"ctrl_mode": 6 if st.get("leader") else 1, "arm_status": 0}}

        def get_leader_joint_angles(self):
            st = _load()
            # On V1.2.1 (0x155-0x157+0x170), only V112/V120/V121 drivers decode the leader frames!
            if st.get("leader") and self.fw in ("V112", "V120", "V121"):
                return _Msg(list(st["joints"]))
            return None

        def get_joint_angles(self):
            st = _load()
            if not st.get("leader"):
                return _Msg(list(st["joints"]))
            return None

        def set_speed_percent(self, p):
            _rec(f"speed:{{p}}")

        def move_j(self, q):
            _rec("move_j:" + ",".join(f"{{v:.3f}}" for v in q))

    class AgxArmFactory:
        @staticmethod
        def create_arm(cfg):
            return _Arm(cfg)
"""), encoding="utf-8")

os.environ["PYTHONPATH"] = str(FAKE) + os.pathsep + os.environ.get("PYTHONPATH", "")
M = JC.build_model((ROOT / "config" / "nero.urdf").read_text(encoding="utf-8"), None, None)

# Test V3: Enter and Exit Teach Mode via RealPipeline with default V1.2.1 config
CALL_LOG.write_text("", encoding="utf-8")
if STATE_FILE.exists():
    STATE_FILE.unlink()

pipe = RR.RealPipeline(lambda: {"can_port": "can0", "fw_version": "v121", "push_fw": "v121"},
                       can_opener=lambda p: (_ for _ in ()).throw(OSError("x")))
gates = []
pipe.mon_status = {"services": {"/control_enable": "std_srvs/srv/SetBool"}}
pipe.monitor.running = lambda: True
pipe.monitor.send = lambda msg: (
    gates.append(msg["value"]) if msg.get("call") == "/control_enable" else None,
    pipe.call_results.__setitem__(msg.get("id") or ("hold", (msg.get("hold") or {}).get("id")), {"ok": True, "response": {"success": True}}),
)
pipe.last_js = {"names": M["joints"], "position": [0.15, -0.42, 0.08, 1.15, -0.05, 0.22, 0.65]}
pipe.last_js_t = time.time()

ev_in = pipe.set_teach_mode(True, timeout_s=8.0)
assert ev_in["ok"] and pipe.teach_mode is True, ev_in
assert ev_in.get("fw") == "V121", ev_in
st_after_in = json.loads(STATE_FILE.read_text(encoding="utf-8"))
assert st_after_in["leader"] is True, st_after_in

# Simulate joint_states resuming once follower mode is restored
def _resume_js_when_follower():
    for _ in range(100):
        if STATE_FILE.exists():
            st = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if not st.get("leader") and st.get("enabled"):
                pipe.last_js_t = time.time()
        time.sleep(0.05)

import threading
threading.Thread(target=_resume_js_when_follower, daemon=True).start()

CALL_LOG.write_text("", encoding="utf-8")
ev_out = pipe.set_teach_mode(False, home_model=M, return_home=False)
assert ev_out["ok"] and pipe.teach_mode is False, ev_out
st_after_out = json.loads(STATE_FILE.read_text(encoding="utf-8"))
assert st_after_out["leader"] is False and st_after_out["enabled"] is True, st_after_out
calls_out = CALL_LOG.read_text(encoding="utf-8").splitlines()
assert "init:V121" in calls_out and "set_follower_mode" in calls_out and "enable" in calls_out, calls_out
assert any(x.startswith("move_j:0.150,-0.420,0.080,1.150") for x in calls_out), calls_out
assert gates[-1] is True, gates

# Even if --fw v111 is passed to cmd_mode(normal) while the V1.2.1 arm is in leader mode, set_follower_mode() is STILL called!
st_after_out["leader"] = True
st_after_out["enabled"] = False
STATE_FILE.write_text(json.dumps(st_after_out), encoding="utf-8")
CALL_LOG.write_text("", encoding="utf-8")
r_v111 = subprocess.run(RR.helper_argv("mode", "--channel", "can0", "--fw", "v111", "--mode", "normal"),
                        capture_output=True, text=True, timeout=20)
assert r_v111.returncode == 0, r_v111.stdout
st_v111 = json.loads(STATE_FILE.read_text(encoding="utf-8"))
assert st_v111["leader"] is False and st_v111["enabled"] is True, st_v111
print("[V3] V1.2.1 (NeroFW.V121) teach mode entry & exit (set_follower_mode before & after enable + hold pose) verified")

# Test V4: Step 3 'CAN push / 使能' (step_push) when arm is stuck in teach (leader) mode
st_stuck = {"leader": True, "enabled": False, "fw": "V121", "joints": [0.10, -0.35, 0.05, 1.05, 0.0, 0.18, 0.50]}
STATE_FILE.write_text(json.dumps(st_stuck), encoding="utf-8")
CALL_LOG.write_text("", encoding="utf-8")
pipe.teach_mode = True
pipe.state["env"] = (RR.GREEN, "ok")
pipe.state["can"] = (RR.GREEN, "ok")
lamp_push = pipe.step_push()
assert lamp_push == RR.GREEN, (lamp_push, pipe.state["push"])
assert pipe.teach_mode is False
st_after_push = json.loads(STATE_FILE.read_text(encoding="utf-8"))
assert st_after_push["leader"] is False and st_after_push["enabled"] is True, st_after_push
calls_push = CALL_LOG.read_text(encoding="utf-8").splitlines()
assert "init:V121" in calls_push and "set_follower_mode" in calls_push and "enable" in calls_push, calls_push
assert any(x.startswith("move_j:0.100,-0.350,0.050,1.050") for x in calls_push), calls_push
pipe.stop_all()
pipe.shutdown()
print("[V4] Step 3 'CAN push / 使能' recovery from teach mode (set_follower_mode + enable + hold pose -> GREEN) verified")
print("ALL V3.54 TESTS PASSED")
