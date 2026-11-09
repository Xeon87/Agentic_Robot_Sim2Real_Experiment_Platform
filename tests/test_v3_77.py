#!/usr/bin/env python3
"""V3.77 offline tests (stale processes before a simulation; sort_items held-item safety after the V3.76 run):
K1 Versions V3.77 + markers; frozen modules unchanged; the backend is not changed by V3.77.
K2 Stale detector (NERO_V377_Stale_Procs.classify): rviz2 / move_group / ros2_control_node, `ros2 launch agx_arm_ctrl`
   (real driver) and `agx_arm_moveit`, python-launched ros2 launch, a NERO backend of ANOTHER release folder, the GPU
   server -> stale; the console NERO_V312_Console.py, an editor / tail mentioning a NERO file, an unrelated python ->
   not stale.
K3 Real processes: a fake leftover "NERO_V2_7_1_Unified_Backend.py" from another folder that ignores SIGINT / SIGTERM
   (with a child) is found by stale_procs and stopped by stop_procs (INT -> TERM -> KILL) within ~3 steps.
K4 Launcher: list_stale / the stop use NERO_V377_Stale_Procs.py (--list / --stop with the launcher's pid); console:
   on_sim_start runs v377_clear_stale() before anything starts (operator dialog, refuses to start when not cleared).
K5 Own straight path like the plan: turns the wrist along the line (_turning_segment) and is used when MoveIt refused
   by the joint-path check (no gripper-model contact) as long as no arm link collides - backend path with a MoveIt
   emulation whose carry paths are refused "J6 only 0.0 deg from its limit": 4/4 sorted via the own path.
K6 Held margin: with NERO_V377_HELD_MARGIN_DEG = 40 the place is refused (PLACE_UNREACHABLE) with margin_rejects and a
   note naming the best margin; the default 10 deg keeps the twin 4/4 (test_v3_76 I10 / test_v3_72 D5).
K7 Trip reason: after a safety stop the adapter reads the newest diag_trip_<n>_<reason>.csv (NERO_DIAG_DIR) and adds
   trip_reason / trip_meaning / closest_joint_limit to the result; CupSkills._check passes them into SAFETY_STOP.
K8 Release folder unchanged.
"""
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v377_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
os.environ["NERO_V368_TASK_DIR"] = str(TMP / "task")
for k in ("NERO_SCENE", "NERO_V343_ACTIVE_SCENE", "NERO_V372_ITEM_SEED", "NERO_V364_SO101_ROLL_DEG",
          "NERO_V375_SO101_OWN_PATH", "NERO_V377_HELD_MARGIN_DEG", "NERO_DIAG_DIR"):
    os.environ.pop(k, None)
H = lambda b: hashlib.sha256(b).hexdigest()


def snapshot():
    out = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rel = line.split(None, 1)[1].strip()
            out[rel] = H((ROOT / rel).read_bytes()) if (ROOT / rel).is_file() else None
    return out


SNAP0 = snapshot()
import NERO_V312_Console as CON             # noqa: E402
import NERO_V313_Real_Robot as RR           # noqa: E402
import NERO_V347_Yolo_Perception as YP      # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49   # noqa: E402
import NERO_V362_Cup_Skills as CK           # noqa: E402
import NERO_V372_Sort_Skills as S           # noqa: E402
import NERO_V377_Stale_Procs as SP          # noqa: E402

try:
    import mujoco  # noqa: F401
    HAVE_MJ = True
except Exception:
    HAVE_MJ = False

# ------------------------------------------------------------------ K1
for m in (CON, RR, YP, W49, S, SP):
    assert m.VERSION in ("V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10"), (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
for fn, h in (("NERO_V2_7_1_Unified_Backend.py", "5be2136f84ec4ccdfe6f8bc8d8998410ff9b3b827af4d571ea44373dc427527e"),
              ("NERO_V2_7_1_Agent.py", "3e52d05ea84d94c8359d5875fa0fee8c2cc42d612b52e59109e1166912439696"),
              ("NERO_V325_Camera.py", "82f2416d73671e7018d09d0cd72c17186154934a315b47c0d19fdb0427afd931"),
              ("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
for fn, mk in (("NERO_V377_Stale_Procs.py", "V3.77 STALE PROCESS CHECK"), ("NERO_V312_Console.py", "V3.77 STALE PROCESS CHECK"),
               ("start_nero_sim.sh", "V3.77 STALE PROCESS CHECK"), ("NERO_V372_Sort_Skills.py", "V3.77 OWN PATH LIKE THE PLAN"),
               ("NERO_V372_Sort_Skills.py", "V3.77 HELD MARGIN"), ("NERO_V362_Cup_Skills.py", "V3.77 TRIP REASON IN THE RESULT")):
    assert mk in (ROOT / fn).read_text(encoding="utf-8"), (fn, mk)
assert "NERO_V377_Stale_Procs.py" in (ROOT / "start_nero.sh").read_text(encoding="utf-8")
print("[K1] V3.77; frozen modules and the backend unchanged; markers present")

# ------------------------------------------------------------------ K2
stale = [["/opt/ros/humble/lib/rviz2/rviz2", "-d", "x.rviz"], ["/opt/ros/humble/lib/moveit_ros_move_group/move_group"],
         ["/opt/ros/humble/lib/controller_manager/ros2_control_node", "--ros-args"],
         ["ros2", "launch", "agx_arm_ctrl", "start_single_agx_arm.launch.py"],
         ["/usr/bin/python3", "/opt/ros/humble/bin/ros2", "launch", "agx_arm_moveit", "demo.launch.py"],
         ["python3", "-u", "/home/u/Downloads/NERO_V3_74/NERO_V2_7_1_Unified_Backend.py"],
         ["/venv/bin/python", "/home/u/NERO_V3_77/NERO_GPU_Planner_Server.py", "--prefer", "curobo"],
         ["python3", "/home/u/NERO_V3_70/NERO_V372_Sort_Skills.py", "--twin"]]
fine = [["python3", "/home/u/NERO_V3_77/NERO_V312_Console.py", "--sim"], ["vim", "NERO_V2_7_1_Unified_Backend.py"],
        ["tail", "-f", "logs/backend.log"], ["python3", "tests/test_v3_77.py"], ["python3", "-c", "print(1)"],
        ["bash", "start_nero_sim.sh"]]
assert all(SP.classify(a) for a in stale), [a for a in stale if not SP.classify(a)]
assert not any(SP.classify(a) for a in fine), [a for a in fine if SP.classify(a)]
print(f"[K2] detector: {len(stale)} leftover kinds found (incl. agx_arm_ctrl, a V3.74 backend), console / editor / tail ignored")

# ------------------------------------------------------------------ K3
other = TMP / "NERO_V3_00"
other.mkdir()
(other / "NERO_V2_7_1_Unified_Backend.py").write_text(
    "import signal, subprocess, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "c = subprocess.Popen(['sleep', '307'])\ntime.sleep(300)\n", encoding="utf-8")
pr = subprocess.Popen([sys.executable, str(other / "NERO_V2_7_1_Unified_Backend.py")], start_new_session=True)
time.sleep(1.0)
hits = SP.stale_procs()
assert any(h[0] == pr.pid for h in hits), hits
logs = []
t0 = time.time()
r3 = SP.stop_procs([pr.pid], step_s=0.6, log=logs.append)
dt = time.time() - t0
pr.wait(timeout=5)
time.sleep(0.3)
kids = subprocess.run(["pgrep", "-f", "sleep 307"], capture_output=True, text=True).stdout.split()
assert r3["stopped"] == [pr.pid] and not r3["left"] and dt < 3.0 and not kids, (r3, dt, kids, logs)
assert not any(h[0] == pr.pid for h in SP.stale_procs())
print(f"[K3] leftover backend of another folder (ignores INT / TERM) + child stopped in {dt:.1f} s; log: {logs[0]}")

# ------------------------------------------------------------------ K4
sh = (ROOT / "start_nero_sim.sh").read_text(encoding="utf-8")
assert 'list_stale(){ python3 "$HERE/NERO_V377_Stale_Procs.py" --list "$$" "$PPID"; }' in sh
assert 'NERO_V377_Stale_Procs.py" --stop "$$" "$PPID"' in sh and "PYSTALE" not in sh
r = subprocess.run(["bash", "-n", str(ROOT / "start_nero_sim.sh")], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
cs = (ROOT / "NERO_V312_Console.py").read_text(encoding="utf-8")
i_start = cs.index("    def on_sim_start():")
seg = cs[i_start:cs.index("tracker.start()", i_start)]
assert "if not v377_clear_stale():" in seg and "def v377_clear_stale():" in cs and 'messagebox.askyesno("启动前检查（V3.77）"' in cs
out = subprocess.run([sys.executable, str(ROOT / "NERO_V377_Stale_Procs.py"), "--list", str(os.getpid())],
                     capture_output=True, text=True)
assert out.returncode == 0
print("[K4] launcher list / stop through the module (pid of the launcher excluded); console checks before starting")

if HAVE_MJ:
    # -------------------------------------------------------------- K5
    import v375_moveit_twin as MT  # noqa: E402

    class JointRefusingCore(MT.MoveItSortCore):
        def _cartesian_waypoint_plan(self, waypoints_xyz, target_quat, label="", max_step=0.006, max_speed=0.07):
            if "carry" in label:
                raise RuntimeError("J6 only 0.0 deg from its limit")            # the user's V3.76 refusal
            return super()._cartesian_waypoint_plan(waypoints_xyz, target_quat, label, max_step, max_speed)

    import v372_backend_twin as SB  # noqa: E402
    orig = SB.SortCore
    SB.SortCore = JointRefusingCore
    try:
        r5 = SB.run_backend_episode()
    finally:
        SB.SortCore = orig
    own = [l for l in r5["log"] if "own straight path" in l and "carry" in l]
    assert r5["state"] == {"item_1": "red", "item_2": "red", "item_3": "blue", "item_4": "blue"}, r5["state"]
    assert own and all("no arm link collides" in l and "wrist turn" in l for l in own), own[:2]
    print(f"[K5] MoveIt refused every carry by the joint-path check -> own turning path ({len(own)} carries), 4/4 sorted")

    # -------------------------------------------------------------- K6
    S.HELD_MARGIN_DEG = 40.0
    try:
        tw = S.SortTwinIO()
        sk = S.SortSkills(tw)
        r6 = sk.run_tool("pick_object", {"item_id": "item_4", "box": "blue"})
        assert not r6["ok"] and r6["reason"] == "GRASP_UNREACHABLE", r6
        pv = r6["physical_verification"]
        assert pv.get("margin_rejects") and "needs >= 40 deg" in pv.get("note", ""), pv
    finally:
        S.HELD_MARGIN_DEG = 10.0
    print(f"[K6] held margin 40 deg -> GRASP_UNREACHABLE with margin_rejects ('{pv['note']}'); default 10 deg")

# ------------------------------------------------------------------ K7
diag = TMP / "diag"
diag.mkdir()
os.environ["NERO_DIAG_DIR"] = str(diag)
t0 = time.time()
(diag / "diag_trip_1_runaway_j6.csv").write_text("t\n", encoding="utf-8")


class _Core:
    def emit(self, s):
        logs7.append(s)

    def _current_joint_state_copy(self):
        return [0.0, 0.3, 0.0, 1.2, 0.0, 0.90, 0.0]

    @staticmethod
    def _arm_positions_from_joint_state(js):
        return list(js)


class _V:
    def __init__(self, v):
        self.value = v


logs7 = []
io = CK.BackendIO({"skill_core": _Core(), "shared": {"safety_trip_count": _V(0)}})
info = io.v377_trip_info(t0)
assert info["trip_reason"] == "runaway_j6" and "joint 6" in info["trip_meaning"], info
assert info["closest_joint_limit"]["joint"] == "J6" and info["closest_joint_limit"]["margin_deg"] < 3.0, info
assert any("[V3.77 SAFETY]" in l and "runaway_j6" in l for l in logs7)
sk7 = CK.CupSkills.__new__(CK.CupSkills)
try:
    sk7._check("carry", dict({"ok": False, "safety": True, "safety_trips": 1, "error": "PHYSICAL_SAFETY_STOP"}, **info))
    raise AssertionError("must abort")
except CK.CupSkills._Abort as e:
    assert e.reason == "SAFETY_STOP" and e.info["trip_reason"] == "runaway_j6" and e.info["closest_joint_limit"]["joint"] == "J6"
os.environ.pop("NERO_DIAG_DIR")
print(f"[K7] safety stop result: trip_reason runaway_j6 ({info['trip_meaning']}), closest {info['closest_joint_limit']}")

SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("[K8] release folder unchanged")
print("ALL V3.77 TESTS PASSED")
