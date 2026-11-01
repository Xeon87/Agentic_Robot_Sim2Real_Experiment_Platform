#!/usr/bin/env python3
"""V3.53 offline unit tests:
V1 Version V3.53 & markers across NERO_V312_Console.py, NERO_V313_Real_Robot.py, NERO_V314_Joint_Control.py,
   NERO_V325_Camera.py, NERO_V349_Real_Wrist_Camera.py; NERO_V308 untouched.
V2 Default depth resolution in RealSenseSource & CameraTabUI._build_locate is 1280×720 (DEPTH_RES[1]).
V3 Teach mode on V111 (where ctrl_mode stays 0x01 and /feedback/joint_states pauses):
   - set_teach_mode(True) detects entry when joint_states was fresh before set_leader_mode and pauses after;
   - set_teach_mode(False) recovers hand pose even after a >2s confirmation dialog pause, clamps joints >11.5° past
     URDF limits, and reopens the driver control gate;
   - cmd_mode(normal) without --hold reads current angles directly from CAN get_leader_joint_angles/get_joint_angles;
   - WorkflowLampBar packs b_teach_in & b_teach_out directly into `parent` (side="right") alongside Copy Log, keeps
     b_teach_out always enabled ('normal') when not busy, and CameraTabUI.on_teach_exit works even when teach_mode=False.
V4 fence_goal_check, move_joints, and JointPanelUI.go_park ('待断电位置⚠'):
   - supports ignore_xy_fence=True (for Park Pose D folding over the base at x=+0.31 m) while enforcing Z table safety;
   - go_park() succeeds on the first click even with the default 12 cm back wall and even when invoked while in teach mode.
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
import NERO_V315_Consistency_Safety as CS     # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V347_Yolo_Perception as YP        # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49     # noqa: E402

assert CON.VERSION in ("V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61") and RR.VERSION in ("V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61") and YP.VERSION in ("V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61") and W49.VERSION in ("V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61")
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61")
h308 = hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest()
assert h308.startswith("020cc6680f181872"), f"NERO_V308 modified: {h308}"
s_cam = (ROOT / "NERO_V325_Camera.py").read_text(encoding="utf-8")
s_con = (ROOT / "NERO_V312_Console.py").read_text(encoding="utf-8")
assert "V3.53 TEACH EXIT & PARK FIX + ALIGNED TEACH BUTTONS + DEFAULT 1280x720 DEPTH" in s_cam
assert "V3.53 TEACH EXIT & PARK FIX + ALIGNED TEACH BUTTONS + DEFAULT 1280x720 DEPTH" in s_con
print("[V1] V3.53 versions & markers verified; NERO_V308 untouched")

rs_src = CAM.RealSenseSource(log=lambda *a: None)
assert rs_src.depth_res == "1280×720" and rs_src.depth_res == CAM.DEPTH_RES[1], rs_src.depth_res
assert "self.v_dres = tk.StringVar(value=DEPTH_RES[1])" in s_cam
print(f"[V2] RealSenseSource & 视觉定位 default depth_res = {rs_src.depth_res} (High Accuracy)")

TMP = Path(tempfile.mkdtemp(prefix="v353_"))
CALLS = TMP / "calls.txt"
FAKE = TMP / "fake"
(FAKE / "pyAgxArm").mkdir(parents=True)
(FAKE / "pyAgxArm" / "__init__.py").write_text(textwrap.dedent(f"""\
    import time
    LOG = {str(CALLS)!r}
    def rec(s):
        open(LOG, "a").write(s + "\\n")
    class ArmModel: NERO = "NERO"
    class NeroFW: V111 = "V111"
    def create_agx_arm_config(**kw): return kw
    class _Arm:
        def connect(self): pass
        def enable(self): rec("enable"); return True
        def set_normal_mode(self): rec("set_normal_mode")
        def set_leader_mode(self): rec("set_leader_mode")
        def get_arm_status(self): return {{"ctrl_mode": 1}}
        def get_leader_joint_angles(self): return None
        def get_joint_angles(self): return [0.12, -0.45, 0.05, 1.10, -0.02, 0.25, 0.80]
        def set_speed_percent(self, p): rec(f"speed {{p}}")
        def move_j(self, q): rec("move_j " + ",".join(f"{{v:.3f}}" for v in q))
    class AgxArmFactory:
        @staticmethod
        def create_arm(cfg): return _Arm()
"""), encoding="utf-8")
env = dict(os.environ, PYTHONPATH=str(FAKE) + os.pathsep + os.environ.get("PYTHONPATH", ""))

CALLS.write_text("", encoding="utf-8")
r = subprocess.run(RR.helper_argv("mode", "--mode", "normal", "--speed", "10"),
                   capture_output=True, text=True, timeout=30, env=env)
evs = [json.loads(l) for l in r.stdout.splitlines() if l.startswith("{")]
c_lines = CALLS.read_text(encoding="utf-8").splitlines()
i_en = c_lines.index("enable")
assert any(x.startswith("move_j 0.120,-0.450,0.050,1.100") for x in c_lines[:i_en]), c_lines
assert evs[-1]["ok"] and evs[-1]["movej_api"] == "move_j"

os.environ["PYTHONPATH"] = env["PYTHONPATH"]
M = JC.build_model((ROOT / "config" / "nero.urdf").read_text(encoding="utf-8"), None, None)
pipe = RR.RealPipeline(lambda: {"can_port": "can0", "push_fw": "v111"},
                       can_opener=lambda p: (_ for _ in ()).throw(OSError("x")))
gates = []
pipe.mon_status = {"services": {"/control_enable": "std_srvs/srv/SetBool"}}
pipe.monitor.running = lambda: True
pipe.monitor.send = lambda msg: (
    gates.append(msg["value"]) if msg.get("call") == "/control_enable" else None,
    pipe.call_results.__setitem__(msg.get("id") or ("hold", (msg.get("hold") or {}).get("id")), {"ok": True, "response": {"success": True}}),
)
pipe.last_js = {"names": M["joints"], "position": [0.10, -0.50, 0.0, 1.05, 0.0, 0.20, 1.95]}
pipe.last_js_t = time.time()
ev_in = pipe.set_teach_mode(True, timeout_s=6.0)
assert ev_in["ok"] and pipe.teach_mode is True, ev_in
assert gates == [False], gates

pipe.last_js_t = time.time() - 5.0
ev_out = pipe.set_teach_mode(False, home_model=M, return_home=False)
assert ev_out["ok"] and pipe.teach_mode is False, ev_out
assert gates[-1] is True, gates
c_lines2 = CALLS.read_text(encoding="utf-8").splitlines()
mj_lines = [x for x in c_lines2 if x.startswith("move_j ")]
assert mj_lines and "1.561" in mj_lines[-1], mj_lines
pipe.shutdown()

class _W:
    def __init__(self, master=None, **kw):
        self.master = master
        self.cfg = dict(kw)
        self.pack_info = None
    def pack(self, **kw): self.pack_info = dict(kw)
    def configure(self, **kw): self.cfg.update(kw)
    def bind(self, *a, **k): pass
    def create_oval(self, *a, **k): return 1
    def itemconfigure(self, *a, **k): pass

class _Var:
    def __init__(self, value=None): self.v = value
    def set(self, v): self.v = v
    def get(self): return self.v

fake_tk = type("tk", (), {"Canvas": _W, "StringVar": _Var})
fake_ttk = type("ttk", (), {"Frame": _W, "Button": _W, "Label": _W})
parent_bar = _W()
wf = CAM.WorkflowLampBar(fake_tk, fake_ttk, parent_bar, on_teach_enter=lambda: None, on_teach_exit=lambda: None)
assert wf.b_teach_in.master is parent_bar and wf.b_teach_out.master is parent_bar
assert wf.b_teach_in.pack_info["side"] == "right" and wf.b_teach_out.pack_info["side"] == "right"
wf.update_from([], teach_mode=False, busy=False)
assert wf.b_teach_in.cfg["state"] == "normal" and wf.b_teach_out.cfg["state"] == "normal"
wf.update_from([], teach_mode=True, busy=False)
assert wf.b_teach_in.cfg["state"] == "disabled" and wf.b_teach_out.cfg["state"] == "normal"
print("[V3] V111 teach entry/exit, WorkflowLampBar parent-level right-packing (aligned with Copy Log), and always-enabled b_teach_out verified")

U = CS.load_sim_urdf()
ws = CS.load_workspace()
pipe2 = RR.RealPipeline(lambda: {"can_port": "can0", "push_fw": "v111"},
                        can_opener=lambda p: (_ for _ in ()).throw(OSError("x")))
pipe2.real_urdf = U
pipe2.safety_fn = lambda: {"fence": ws["fence"], "path_check": True, "tool_T": None, "sim_urdf": U, "corners": ws.get("corners")}
pk = JC.load_park()
tgt_deg, _notes = JC.park_target(pk["deg"], M["limits"])
tgt_rad = [JC.rad(v) for v in tgt_deg]
ok_strict, txt_strict, v_strict = pipe2.fence_goal_check(M, tgt_rad, ignore_xy_fence=False)
ok_park, txt_park, v_park = pipe2.fence_goal_check(M, tgt_rad, ignore_xy_fence=True)
assert not ok_strict and any(v["axis"] == "xy" for v in v_strict), v_strict
assert ok_park and v_park == [], (ok_park, txt_park, v_park)

walls_calls = []
moves_seen = []
teach_exits = []
pipe2.teach_mode = True
pipe2.state["moveit"] = ("green", "ok")
pipe2.apply_walls_fn = lambda walls=None, wait=True: walls_calls.append(walls)
pipe2.set_teach_mode = lambda on, **kw: (teach_exits.append((on, kw.get("return_home"))), setattr(pipe2, "teach_mode", False))

dummy_jp = type("JP", (), {
    "pipe": pipe2,
    "model": M,
    "enabled": False,
    "v_state": _Var("示教中"),
    "first_move_done": False,
    "targets_dirty": False,
    "log": lambda self, msg: None,
    "actual_deg": lambda self: [0.0] * 7,
    "_speed": lambda self: 0.10,
    "ask": lambda self, t, q: True,
    "park_zone_hint": lambda self, t: "安全区后边界要到 x ≥ +0.38 m",
    "_move_and_report": lambda self, label, tgt, speed, ignore_xy_fence=False: (moves_seen.append((label, list(tgt), ignore_xy_fence)) or True),
    "_run": lambda self, name, fn: fn(),
})()
JC.JointPanelUI.go_park(dummy_jp)
assert teach_exits == [(False, False)], teach_exits
assert walls_calls == [False, None], walls_calls
assert len(moves_seen) == 1 and moves_seen[0][0] == "待断电位置" and moves_seen[0][2] is True, moves_seen
assert max(abs(a - b) for a, b in zip(moves_seen[0][1], tgt_deg)) < 0.01
pipe2.shutdown()
print("[V4] Park Pose D fence check & JointPanelUI.go_park (auto-exit teach mode + temporary wall removal + ignore_xy_fence=True) verified")
print("ALL V3.53 TESTS PASSED")
