#!/usr/bin/env python3
"""Offline tests for V3.12 (console entry; sim chain started by a button). No ROS / GPU / MuJoCo / display.

L1  launchers: start_nero.sh only opens the console (--sim / --real); start_nero_sim.sh = V3.11 chain + trap fix
L2  sim stack untouched: backend / agent / planner core / GPU client+server / camera module byte-identical to V3.11
L3  SimStepTracker on a recorded real V3.11 startup log, a failed start, a TRAC-less start, an early exit
L4  SimRunner stop semantics on a fake launcher with the V3.12 trap (cleanup once, no further startup steps)
L5  the REAL start_nero_sim.sh: missing deps -> fails fast without waiting for Enter;
    SIGINT while it waits for the GPU server -> exits 130, cleanup runs, fake GPU server killed
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
import NERO_V312_Console as C  # noqa: E402  (must import without tkinter)

assert C.VERSION.startswith(("V3.12", "V3.13", "V3.14", "V3.15", "V3.16", "V3.17", "V3.18", "V3.19", "V3.20", "V3.21", "V3.22", "V3.23", "V3.24", "V3.25", "V3.26", "V3.27", "V3.28", "V3.29", "V3.30", "V3.31", "V3.32", "V3.33", "V3.34", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1"))
assert "tkinter" not in sys.modules, "console must not import tkinter at module import"


def rd(p):
    return open(os.path.join(HERE, p), encoding="utf-8").read()


def drain(runner, timeout=15.0):
    out, t_end = [], time.time() + timeout
    while time.time() < t_end:
        try:
            ln = runner.lines.get(timeout=0.2)
        except Exception:
            continue
        if ln is None:
            return out, True
        out.append(ln)
    return out, False


# ---------------------------------------------------------------- L1 launchers
ENTRY, SIM = rd("start_nero.sh"), rd("start_nero_sim.sh")
for f in ("start_nero.sh", "start_nero_sim.sh"):
    subprocess.run(["bash", "-n", os.path.join(HERE, f)], check=True)
assert "V3.12 CONSOLE ENTRY" in ENTRY and 'exec bash "$SIM"' in ENTRY and "--real" in ENTRY
assert "NERO_V312_Console.py" in ENTRY
for forbidden in ("NERO_V2_7_1_Unified_Backend.py", "ros2 launch", "GPU_Planner_Server"):
    assert forbidden not in ENTRY, f"entry script must not start the sim itself: {forbidden}"
assert "V3.12 SIM LAUNCH CHAIN" in SIM
assert "trap cleanup EXIT\n" in SIM and "exit 130' INT TERM" in SIM and "trap cleanup EXIT INT TERM" not in SIM
for m in ("V3.07 PRE-LAUNCH STALE-PROCESS CLEANUP", "V3.07 DUPLICATE ARM CONTROLLER", "NERO_V307_PRELAUNCH_CLEANUP",
          "NERO_V308_Camera_Perception", 'python3 -u "$BACKEND"', "SYSTEM READY", "NERO_NO_FAIL_PAUSE"):
    assert m in SIM, m
print("[L1] start_nero.sh opens only the console; start_nero_sim.sh keeps the V3.11 chain, trap fixed")

# ---------------------------------------------------------------- L2 sim stack untouched
V311 = {
    # V3.12.1: backend + agent changed on purpose (held-egg recovery); see tests/test_v3_12_1.py
    "NERO_V71_Hybrid_GPU_Planner_Core.py": "973ba5d05636a394832e2e31d3b7379833ead9e48211900ab7fd3614768b9a2c",
    "NERO_GPU_Planner_Client.py": "3956f7a596251149dfb0abc2b89f27334d8f9279a8fcf9e325eb5877d7605438",
    "NERO_GPU_Planner_Server.py": "2728c2f191c8cd24c88276d8b39efffdfdb395e7fcdee5359a5727861e23a4bc",
    "NERO_V308_Camera_Perception.py": "020cc6680f18187214c6c9d33fe24735bba9d4adb2cc8d9a9466e02e62b96efd",
}
for f, h in V311.items():
    got = hashlib.sha256(open(os.path.join(HERE, f), "rb").read()).hexdigest()
    assert got == h, f"{f} changed vs V3.11"
print("[L2] planner / GPU / camera files byte-identical to V3.11")

# ---------------------------------------------------------------- L3 tracker
REAL_LOG = """\
[14:17:00] [OK] Release integrity: all files match SHA256SUMS.txt
[14:17:00] [OK] py_compile PASS
[14:17:00] [INFO] MuJoCo model (external): /home/qiuren/mujoco_ws/nero/nero_motor.xml sha256=73492680c22cc0d2
[14:17:00] [OK] V3.07 pre-launch check: no stale ROS/NERO processes
[14:17:00] [INFO] CUDA self-check with GPU planner venv...
[14:17:03] [INFO] Starting cuRobo GPU server: /x/NERO_GPU_Planner_Server.py
[14:17:08] [OK] GPU server READY PID=11696
[14:17:08] [OK] cuRobo pose self-test PASS (no robot motion)
[MODULE OK ] NERO_V2_7_1_Agent -> /x/NERO_V2_7_1_Agent.py sha256=9debb186043c66d2
[14:17:09] [OK] All NERO modules resolve inside this folder
[14:17:09] [OK] V3.08 simulated D435i camera ON (offscreen GL order: egl,osmesa,glfw); captures -> /x/logs/perception
[14:17:09] [OK] BioIK package available: bio_ik
[14:17:09] [OK] TRAC-IK package available (optional reference)
[14:17:09] [INFO] Starting backend: /x/NERO_V2_7_1_Unified_Backend.py
[14:17:09] [OK] Backend started PID=11853 (log: /x/logs/backend.log)
[14:17:10] [OK] ROS topic READY: /feedback/joint_states
[14:17:10] [OK] Temporarily configured root MoveIt for BioIK (mode=gd_c timeout=0.05s)
[14:17:10] [OK] Primary BioIK MoveIt launch PID=12012
[14:17:13] [OK] Primary /move_group IK solver verified: BioIK
[14:17:13] [OK] TRAC reference MoveIt launch PID=12516
[14:17:37] [OK] Reference /trac_ref/move_group IK solver verified: TRAC-IK
[14:17:38] [INFO] Restored original AgileX kinematics.yaml
[14:17:39] [INFO] V3.07 arm_controller instances: action_servers=1 controller_state_publishers=1
[14:17:39] [READY] SYSTEM READY: NERO V3.11 | BioIK primary + cuRobo GPU + MuJoCo | TRAC optional | single arm_controller
"""
keys = [k for k, *_ in C.SIM_STEPS]
t = C.SimStepTracker()
t.start()
seen = []
for ln in REAL_LOG.splitlines():
    t.feed(ln)
    seen.append(t.current())
assert all(t.state[k] == C.GREEN for k in keys), t.state
assert t.ready and not t.failed
order = [k for k in dict.fromkeys(x for x in seen if x)]
assert order == keys[1:-1] or order == keys[:-1] or order[0] in keys[:2], order
t.process_exited(0)
assert all(t.state[k] == C.GREEN for k in keys)        # normal shutdown after READY stays green

t.start()
for ln in REAL_LOG.splitlines()[:6]:
    t.feed(ln)
assert t.current() == "gpu"
t.feed("[14:17:30] [FAIL] cuRobo GPU server did not become READY")
t.feed(" NERO V3.12 START FAILED — see /x/logs/launcher.log")
assert t.state["gpu"] == C.RED and t.failed and t.state["integrity"] == C.GREEN and t.state["modules"] == C.GREY
t.process_exited(1)
assert t.state["gpu"] == C.RED

t.start()
for ln in REAL_LOG.splitlines():
    if "trac_ref" in ln:
        ln = "[14:17:37] [WARN] Optional TRAC reference MoveIt not verified; continuing BioIK-only"
    t.feed(ln)
assert t.state["trac"] == C.AMBER and t.ready and t.state["ready"] == C.GREEN

t.start()                                               # early TRAC plugin warning must not touch the TRAC lamp
for ln in REAL_LOG.splitlines()[:13]:
    t.feed(ln.replace("TRAC-IK package available (optional reference)", "TRAC-IK plugin unavailable; BioIK-only"))
assert t.state["trac"] == C.GREY and t.current() == "backend"
t.process_exited(1)                                     # backend died without a [FAIL] line
assert t.state["backend"] == C.RED and t.failed
print("[L3] step tracker: recorded V3.11 startup all green; fail / TRAC-optional / silent-exit cases correct")

# ---------------------------------------------------------------- L4 runner + V3.12 trap on a fake launcher
tmp = tempfile.mkdtemp(prefix="v312_")
fake = os.path.join(tmp, "fake_sim.sh")
marker = os.path.join(tmp, "after_stop")
open(fake, "w").write(f"""#!/usr/bin/env bash
status(){{ echo "[$(date +%H:%M:%S)] [$1] $2"; }}
cleanup(){{ status INFO "Shutdown: stopping NERO processes"; [[ -n "$BK" ]] && kill $BK 2>/dev/null; wait $BK 2>/dev/null; echo CLEANUP_DONE; }}
trap cleanup EXIT
trap 'status INFO "Stop signal received"; exit 130' INT TERM
status OK "py_compile PASS"
status OK "V3.07 pre-launch check: no stale ROS/NERO processes"
for i in $(seq 1 40); do sleep 0.1; done
status OK "cuRobo pose self-test PASS (no robot motion)"
sleep 300 & BK=$!
echo "BK=$BK"
wait $BK
touch {marker}
""")
r = C.SimRunner(fake)
r.start()
lines = []
t_end = time.time() + 10
while time.time() < t_end and not any("pose self-test" in x for x in lines):
    try:
        ln = r.lines.get(timeout=0.2)
    except Exception:
        continue
    lines.append(ln)
while not any(x.startswith("BK=") for x in lines):
    lines.append(r.lines.get(timeout=5))
bk = int([x for x in lines if x.startswith("BK=")][0][3:])
t0 = time.time()
r.stop()
rest, done = drain(r, 10)
assert done and r.exit_code == 130, (r.exit_code, rest)
assert time.time() - t0 < 5
assert rest.count("CLEANUP_DONE") == 1 and any("Stop signal received" in x for x in rest)
assert not os.path.exists(marker), "launcher continued after stop"
assert subprocess.run(["kill", "-0", str(bk)], capture_output=True).returncode != 0, "child survived"

r = C.SimRunner(fake)                                    # stop during the startup loop: must not continue
r.start()
time.sleep(0.5)
r.stop()
rest, done = drain(r, 10)
assert done and r.exit_code == 130 and not any("pose self-test" in x for x in rest), rest
print("[L4] SimRunner.stop(): exit 130, cleanup exactly once, no further startup steps, children gone")

# ---------------------------------------------------------------- L5 the real start_nero_sim.sh
rel = os.path.join(tmp, "rel")
shutil.copytree(HERE, rel, ignore=shutil.ignore_patterns("logs", "__pycache__", "tests"))
os.makedirs(os.path.join(rel, "tests"))
shutil.copy(os.path.join(HERE, "tests", "check_release.sh"), os.path.join(rel, "tests"))
for f in os.listdir(os.path.join(HERE, "tests")):
    if f.endswith(".py"):
        shutil.copy(os.path.join(HERE, "tests", f), os.path.join(rel, "tests"))
home = os.path.join(tmp, "home")
os.makedirs(home)
env_base = {"HOME": home, "PATH": os.environ["PATH"], "NERO_GPU_PLANNER_PORT": "65011"}

r = C.SimRunner(os.path.join(rel, "start_nero_sim.sh"))   # 1) deps missing -> FAIL fast, no Enter prompt
os.environ.update(env_base)
r.start()
out, done = drain(r, 20)
assert done and r.exit_code == 1, (r.exit_code, out[-5:])
assert any("External dependency missing" in x for x in out), out[-8:]
t = C.SimStepTracker()
t.start()
for ln in out:
    t.feed(ln)
t.process_exited(1)
assert t.state["integrity"] == C.RED and t.failed

fr = os.path.join(tmp, "fakeroot")                      # 2) fake deps -> reaches the GPU wait loop -> SIGINT
for d in ("ros/setup.bash", "agx/install/setup.bash", "venv/bin/activate", "asset/nero_motor.xml",
          "gpu/config/nero.urdf", "gpu/config/nero_curobo.yml"):
    p = os.path.join(fr, d)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").write("# fake\n")
gpu_py = os.path.join(fr, "gpu", "python")
open(gpu_py, "w").write('#!/usr/bin/env bash\nif [[ "$1" == "-" ]]; then cat >/dev/null; '
                        'echo cuda_available=True; exit 0; fi\nexec -a FAKE_GPU_SERVER sleep 300\n')
os.chmod(gpu_py, 0o755)
if not os.path.isdir("/opt/ros"):
    print("[L5] /opt/ros missing on this machine -> SIGINT part skipped")
    print("ALL V3.12 TESTS PASS")
    sys.exit(0)
os.environ.update({"ROS_DISTRO": "../.." + fr + "/ros",
                   "AGX_ARM_WS": os.path.join(fr, "agx"), "VENV": os.path.join(fr, "venv"),
                   "NERO_ASSET_WS": os.path.join(fr, "asset"), "NERO_GPU_ROOT": os.path.join(fr, "gpu"),
                   "NERO_GPU_PLANNER_PYTHON": gpu_py})
assert os.path.isfile(f"/opt/ros/{os.environ['ROS_DISTRO']}/setup.bash")
r = C.SimRunner(os.path.join(rel, "start_nero_sim.sh"))
r.start()
lines, t_end = [], time.time() + 30
while time.time() < t_end and not any("Starting cuRobo GPU server" in x for x in lines):
    try:
        ln = r.lines.get(timeout=0.2)
    except Exception:
        continue
    assert ln is not None, lines[-10:]
    lines.append(ln)
assert any("py_compile PASS" in x for x in lines) and any("pre-launch check" in x for x in lines), lines
time.sleep(1.0)
pg = subprocess.run(["pgrep", "-f", "FAKE_GPU_SERVER"], capture_output=True, text=True).stdout.split()
t0 = time.time()
r.stop()
rest, done = drain(r, 15)
assert done and r.exit_code == 130, (r.exit_code, rest[-8:])
assert time.time() - t0 < 6, time.time() - t0
assert any("Stop signal received" in x for x in rest) and sum("Shutdown: stopping NERO processes" in x for x in rest) == 1
assert not any("GPU server READY" in x or "did not become READY" in x for x in rest), rest
def alive(pid):   # a killed child may linger briefly as a zombie until it is reaped
    try:
        st = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0]
    except Exception:
        return False
    return st != "Z"
t_end = time.time() + 3
while time.time() < t_end and any(alive(p) for p in pg):
    time.sleep(0.1)
still = [p for p in pg if alive(p)]
assert not still, f"fake GPU server survived: {still}"
for k in ("ROS_DISTRO", "AGX_ARM_WS", "VENV", "NERO_ASSET_WS", "NERO_GPU_ROOT", "NERO_GPU_PLANNER_PYTHON"):
    os.environ.pop(k, None)
shutil.rmtree(tmp, ignore_errors=True)
print("[L5] real start_nero_sim.sh: fails fast without Enter prompt; SIGINT in GPU wait -> exit 130, cleanup once, GPU server killed")
print("ALL V3.12 TESTS PASS")
