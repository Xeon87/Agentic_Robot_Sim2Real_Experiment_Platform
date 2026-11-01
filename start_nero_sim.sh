#!/usr/bin/env bash
# ============================================================
# NERO V3 single-folder launcher — SIMULATION chain
#
# V3.12 SIM LAUNCH CHAIN: this is the V3.11 start_nero.sh, unchanged except the
# signal trap.  It is started by the "仿真 (MuJoCo)" tab of NERO_V312_Console.py
# (or directly: ./start_nero.sh --sim).  ./start_nero.sh itself now only opens
# the console; nothing starts automatically.
#
# Every NERO Python file that runs comes from THIS folder.  Nothing is copied,
# patched or rebuilt at startup.  External, non-NERO dependencies (ROS2,
# agx_arm_ws/MoveIt, BioIK/TRAC-IK, the MuJoCo venv, the cuRobo GPU venv and
# the MuJoCo model nero_motor.xml + AgileX meshes) are referenced by path only.
#
# Startup order is identical to the previous V2-52.3 chain:
#   preflight -> CUDA check -> cuRobo GPU server + self-test
#   -> backend (MuJoCo + Tk GUI) -> BioIK MoveIt -> TRAC-IK reference MoveIt
# ============================================================
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE="$(cat "$HERE/VERSION" 2>/dev/null || echo unknown)"

# ---------------- external dependencies (override via env) ----------------
ROS_DISTRO="${ROS_DISTRO:-humble}"
AGX_ARM_WS="${AGX_ARM_WS:-$HOME/agx_arm_ws}"
VENV="${VENV:-$HOME/mujoco_ws/venv}"
BIO_IK_WS="${BIO_IK_WS:-$HOME/bio_ik_ws}"
TRAC_IK_WS="${TRAC_IK_WS:-$HOME/trac_ik_ws}"
NERO_ASSET_WS="${NERO_ASSET_WS:-$HOME/mujoco_ws/nero}"      # nero_motor.xml + meshes
GPU_ROOT="${NERO_GPU_ROOT:-$HOME/.nero_gpu_planner}"
GPU_PY="${NERO_GPU_PLANNER_PYTHON:-$GPU_ROOT/.venv/bin/python}"
GPU_CONFIG="${NERO_GPU_CONFIG_DIR:-$GPU_ROOT/config}"
USE_RVIZ="${USE_RVIZ:-true}"

BIO_IK_MODE="${BIO_IK_MODE:-gd_c}"
BIO_IK_TIMEOUT="${BIO_IK_TIMEOUT:-0.05}"
BIO_MIN_DISP_W="${BIO_MIN_DISP_W:-1.0}"
BIO_AVOID_LIMIT_W="${BIO_AVOID_LIMIT_W:-0.4}"
BIO_CENTER_W="${BIO_CENTER_W:-0.1}"
TRAC_IK_SOLVE_TYPE="${TRAC_IK_SOLVE_TYPE:-Distance}"
TRAC_IK_TIMEOUT="${TRAC_IK_TIMEOUT:-0.10}"

ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
AGX_SETUP="${AGX_ARM_WS}/install/setup.bash"
BIO_SETUP="${BIO_IK_WS}/install/setup.bash"
TRAC_SETUP="${TRAC_IK_WS}/install/setup.bash"
VENV_ACTIVATE="${VENV}/bin/activate"
MODEL_FILE="${NERO_ASSET_WS}/nero_motor.xml"

# ---------------- NERO files: ONLY from this folder ----------------
BACKEND="$HERE/NERO_V2_7_1_Unified_Backend.py"
AGENT="$HERE/NERO_V2_7_1_Agent.py"
CORE="$HERE/NERO_V71_Hybrid_GPU_Planner_Core.py"
CLIENT="$HERE/NERO_GPU_Planner_Client.py"
GPU_SERVER="$HERE/NERO_GPU_Planner_Server.py"
CAMERA="$HERE/NERO_V308_Camera_Perception.py"
WRIST="$HERE/NERO_V326_Wrist_Camera.py"
FIXED331="$HERE/NERO_V331_Fixed_Camera.py"
SCENES343="$HERE/NERO_V343_Scenes.py"
NERO_MODULES="NERO_V2_7_1_Agent NERO_V71_Hybrid_GPU_Planner_Core NERO_GPU_Planner_Client NERO_V308_Camera_Perception"

LOG_DIR="$HERE/logs"
mkdir -p "$LOG_DIR"
LAUNCH_LOG="$LOG_DIR/launcher.log"
STATUS_LOG="$LOG_DIR/startup_status.log"
BACKEND_LOG="$LOG_DIR/backend.log"
GPU_LOG="$LOG_DIR/gpu_server.log"
BIO_MOVEIT_LOG="$LOG_DIR/bio_moveit.log"
TRAC_MOVEIT_LOG="$LOG_DIR/trac_ref_moveit.log"
KIN_BACKUP="$LOG_DIR/kinematics_backup.yaml"
: > "$LAUNCH_LOG"; : > "$STATUS_LOG"; : > "$GPU_LOG"
exec > >(tee -a "$LAUNCH_LOG") 2>&1
export NERO_STATUS_FILE="$STATUS_LOG"

GPU_PID=""; BACKEND_PID=""; BIO_MOVEIT_PID=""; TRAC_MOVEIT_PID=""
KIN_YAML=""; KIN_BACKED_UP=0

stamp(){ date '+%H:%M:%S'; }
status(){ local level="$1"; shift; local line="[$(stamp)] [${level}] $*"; echo "$line"; echo "$line" >> "$STATUS_LOG"; }
pause_on_fail(){
  if [[ "${NERO_NO_FAIL_PAUSE:-0}" != "1" && -r /dev/tty ]]; then
    read -r -p 'Press Enter to close this window...' _ </dev/tty || true
  fi
}
fatal(){
  status FAIL "$1"
  echo "============================================================"
  echo " NERO ${RELEASE} START FAILED — see $LAUNCH_LOG"
  echo "============================================================"
  [[ -s "$GPU_LOG" ]] && { echo "--- GPU server tail ---"; tail -n 60 "$GPU_LOG"; }
  [[ -s "$BACKEND_LOG" ]] && { echo "--- backend tail ---"; tail -n 40 "$BACKEND_LOG"; }
  pause_on_fail
  exit 1
}
restore_kinematics(){
  set +e
  if [[ "$KIN_BACKED_UP" -eq 1 && -n "$KIN_YAML" && -f "$KIN_BACKUP" ]]; then
    cp -f "$KIN_BACKUP" "$KIN_YAML" 2>/dev/null || true
    status INFO "Restored original AgileX kinematics.yaml"
    KIN_BACKED_UP=0
  fi
}
cleanup(){
  set +e
  status INFO "Shutdown: stopping NERO processes"
  [[ -n "$TRAC_MOVEIT_PID" ]] && { kill -- "-$TRAC_MOVEIT_PID" 2>/dev/null || kill "$TRAC_MOVEIT_PID" 2>/dev/null; }
  [[ -n "$BIO_MOVEIT_PID" ]] && { kill -- "-$BIO_MOVEIT_PID" 2>/dev/null || kill "$BIO_MOVEIT_PID" 2>/dev/null; }
  [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null
  [[ -n "$GPU_PID" ]] && kill "$GPU_PID" 2>/dev/null
  for p in "$TRAC_MOVEIT_PID" "$BIO_MOVEIT_PID" "$BACKEND_PID" "$GPU_PID"; do
    [[ -n "$p" ]] && wait "$p" 2>/dev/null
  done
  restore_kinematics
}
# V3.12: INT/TERM exit, EXIT runs cleanup once.  (The console stops the sim by
# sending SIGINT to this script only; with the old "trap cleanup INT" bash ran
# cleanup and then carried on with the next startup step.)
trap cleanup EXIT
trap 'status INFO "Stop signal received"; exit 130' INT TERM

echo "============================================================"
echo " NERO ${RELEASE} — single-folder release"
echo " Folder: $HERE"
echo "============================================================"

# ============================================================
# 1. PREFLIGHT — files, integrity, compile
# ============================================================
for f in "$BACKEND" "$AGENT" "$CORE" "$CLIENT" "$GPU_SERVER" "$CAMERA" "$WRIST" "$FIXED331" "$SCENES343" "$HERE/SHA256SUMS.txt"; do
  [[ -f "$f" ]] || fatal "NERO file missing from this folder: $f"
done
for f in "$ROS_SETUP" "$AGX_SETUP" "$VENV_ACTIVATE" "$MODEL_FILE"; do
  [[ -f "$f" ]] || fatal "External dependency missing: $f"
done
[[ -x "$GPU_PY" ]] || fatal "GPU planner Python missing: $GPU_PY"
[[ -f "$GPU_CONFIG/nero.urdf" && -f "$GPU_CONFIG/nero_curobo.yml" ]] || fatal "NERO GPU config missing in $GPU_CONFIG"

if ( cd "$HERE" && sha256sum --quiet -c SHA256SUMS.txt ) >/dev/null 2>&1; then
  status OK "Release integrity: all files match SHA256SUMS.txt"
else
  status WARN "Release integrity: files differ from SHA256SUMS.txt (edited locally?):"
  ( cd "$HERE" && sha256sum -c SHA256SUMS.txt 2>/dev/null | grep -v ': OK$' ) | sed 's/^/          /'
  if [[ "${NERO_ALLOW_MODIFIED:-0}" != "1" ]]; then
    fatal "Refusing to start a modified release. Set NERO_ALLOW_MODIFIED=1 to run anyway."
  fi
fi
python3 -m py_compile "$BACKEND" "$AGENT" "$CORE" "$CLIENT" "$GPU_SERVER" "$CAMERA" "$WRIST" "$FIXED331" "$SCENES343" || fatal "py_compile failed"
status OK "py_compile PASS"
status INFO "MuJoCo model (external): $MODEL_FILE sha256=$(sha256sum "$MODEL_FILE" | cut -c1-16)"
# V3.44 MUJOCO SCENE: chosen before start (NERO console drop-down -> NERO_SCENE); unset = fill_egg_holder
SCENE344="${NERO_SCENE:-fill_egg_holder}"
case "$SCENE344" in
  fill_egg_holder|store_in_safe) ;;
  *) fatal "Unknown NERO_SCENE='$SCENE344' (known: fill_egg_holder, store_in_safe)";;
esac
export NERO_SCENE="$SCENE344"
status INFO "V3.44 MuJoCo scene: $SCENE344 (chosen in the NERO console before start; default fill_egg_holder)"

# ============================================================
# 1b. V3.07 PRE-LAUNCH STALE-PROCESS CLEANUP
#     V3.06 run halted on "[V3.05 DUPLICATE ARM CONTROLLER] publishers=2":
#     a previous run left ros2_control_node / move_group behind.  Stop every
#     leftover NERO / AgileX-MoveIt / ros2_control process before starting.
#     NERO_V307_PRELAUNCH_CLEANUP=0 -> do not kill, refuse to start instead.
# ============================================================
# Match by argv[0] (and the script argv for python/ros2 launch) only, so an
# editor or `tail -f` that merely mentions a NERO file name is never killed.
list_stale(){ python3 - "$$" <<'PYSTALE'
import os, sys
me = {int(sys.argv[1]), os.getpid(), os.getppid()}
bins = {"ros2_control_node", "move_group", "spawner", "rviz2", "robot_state_publisher"}
scripts = {"NERO_V2_7_1_Unified_Backend.py", "NERO_GPU_Planner_Server.py", "spawner"}
for d in os.listdir("/proc"):
    if not d.isdigit() or int(d) in me:
        continue
    try:
        argv = [a.decode(errors="replace") for a in open(f"/proc/{d}/cmdline", "rb").read().split(b"\0") if a]
    except Exception:
        continue
    if not argv:
        continue
    b0 = os.path.basename(argv[0])
    hit = b0 in bins
    if not hit and b0.startswith("python"):
        rest = [os.path.basename(a) for a in argv[1:3] if not a.startswith("-")]
        hit = any(r in scripts for r in rest) or (
            len(argv) >= 4 and os.path.basename(argv[1]) == "ros2" and argv[2] == "launch" and argv[3] == "agx_arm_moveit")
    if not hit and b0 == "ros2" and argv[1:3] == ["launch", "agx_arm_moveit"]:
        hit = True
    if hit:
        print(d, " ".join(argv)[:200])
PYSTALE
}
STALE="$(list_stale)"
if [[ -n "$STALE" ]]; then
  status WARN "V3.07 stale processes from a previous run:"
  echo "$STALE" | sed 's/^/          /' | tee -a "$STATUS_LOG"
  if [[ "${NERO_V307_PRELAUNCH_CLEANUP:-1}" == "0" ]]; then
    fatal "Stale ROS/NERO processes present and NERO_V307_PRELAUNCH_CLEANUP=0; stop them manually"
  fi
  PIDS="$(echo "$STALE" | awk '{print $1}')"
  kill $PIDS 2>/dev/null || true
  for _ in $(seq 1 20); do [[ -z "$(list_stale)" ]] && break; sleep 0.25; done
  LEFT="$(list_stale)"
  if [[ -n "$LEFT" ]]; then
    status WARN "V3.07 SIGKILL for processes that ignored SIGTERM"
    kill -9 $(echo "$LEFT" | awk '{print $1}') 2>/dev/null || true
    sleep 0.5
  fi
  [[ -z "$(list_stale)" ]] || fatal "V3.07 could not stop stale processes: $(list_stale | tr '\n' ';')"
  V307_RESTART_DAEMON=1
  status OK "V3.07 stale processes stopped"
else
  V307_RESTART_DAEMON=0
  status OK "V3.07 pre-launch check: no stale ROS/NERO processes"
fi

# ============================================================
# 2. GPU — CUDA self-check, cuRobo server, pose self-test (from V2-43/V2-50.1)
# ============================================================
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  status INFO "Clearing inherited CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}'"
fi
unset CUDA_VISIBLE_DEVICES

# ---- V3.34 CUDA AUTOFIX BEGIN ----------------------------------------------
# Typical cause of "CUDA is not available" while nvidia-smi still works: the
# nvidia_uvm kernel module is left in a bad state (most often after suspend /
# resume, sometimes after a crashed CUDA process).  A reboot reloads it; here
# the launcher does the same without a reboot:
#   1. retry once (transient)          2. stop leftover NERO GPU processes
#   3. find who still holds /dev/nvidia-uvm (non-NERO holders are only listed)
#   4. reload nvidia_uvm with ONE admin prompt (sudo -n if allowed, else pkexec
#      password dialog, else sudo in the terminal)   5. check CUDA again
# NERO_V334_CUDA_AUTOFIX=0 disables it; NERO_V334_CUDA_PRIV=auto|sudo|pkexec|none
cuda_check(){
  CUDA_CHECK_OUT="$("$GPU_PY" - <<'PYCUDA' 2>&1
import sys
try:
    import torch
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is not available to the GPU planner venv")
    print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"CUDA_CHECK_ERROR: {type(e).__name__}: {e}")
    sys.exit(1)
PYCUDA
)"
}
uvm_holders(){ python3 - <<'PYUVM'
import os
for d in os.listdir("/proc"):
    if not d.isdigit():
        continue
    try:
        fds = os.listdir(f"/proc/{d}/fd")
    except Exception:
        continue
    for fd in fds:
        try:
            if os.readlink(f"/proc/{d}/fd/{fd}") == "/dev/nvidia-uvm":
                argv = open(f"/proc/{d}/cmdline", "rb").read().split(b"\0")
                print(d, b" ".join(a for a in argv if a).decode(errors="replace")[:160])
                break
        except Exception:
            pass
PYUVM
}
run_privileged(){   # $1 = shell command run as root; returns its status
  local cmd="$1" mode="${NERO_V334_CUDA_PRIV:-auto}"
  if [[ "$mode" == "none" ]]; then return 99; fi
  if [[ "$mode" == "auto" || "$mode" == "sudo" ]] && command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    status INFO "V3.34 CUDA AUTOFIX: running as root via sudo (no password needed)"
    sudo -n /bin/sh -c "$cmd"; return $?
  fi
  if [[ "$mode" == "auto" || "$mode" == "pkexec" ]] && command -v pkexec >/dev/null 2>&1 && [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    status INFO "V3.34 CUDA AUTOFIX: a password dialog asks for admin rights to reload the NVIDIA UVM driver"
    pkexec /bin/sh -c "$cmd"; return $?
  fi
  if [[ "$mode" == "auto" || "$mode" == "sudo" ]] && command -v sudo >/dev/null 2>&1 && [[ -r /dev/tty ]]; then
    status INFO "V3.34 CUDA AUTOFIX: enter your password in this terminal to reload the NVIDIA UVM driver"
    sudo /bin/sh -c "$cmd" </dev/tty; return $?
  fi
  return 98
}
cuda_autofix(){
  [[ "${NERO_V334_CUDA_AUTOFIX:-1}" == "0" ]] && return 1
  status WARN "V3.34 CUDA AUTOFIX: CUDA not available -> trying to repair without a reboot"
  sleep "${NERO_V334_CUDA_RETRY_S:-2}"
  if cuda_check; then status OK "V3.34 CUDA AUTOFIX: CUDA back after a short wait (transient)"; return 0; fi
  pkill -f '[N]ERO_GPU_Planner_Server.py' 2>/dev/null || true
  sleep 0.5
  local holders nero_h other_h
  holders="$(uvm_holders)"
  nero_h="$(echo "$holders" | grep -E 'NERO_|curobo|cuRobo' || true)"
  if [[ -n "$nero_h" ]]; then
    status WARN "V3.34 CUDA AUTOFIX: stopping leftover NERO GPU processes:"; echo "$nero_h" | sed 's/^/          /'
    kill $(echo "$nero_h" | awk '{print $1}') 2>/dev/null || true; sleep 1
    kill -9 $(echo "$nero_h" | awk '{print $1}') 2>/dev/null || true; sleep 0.5
    if cuda_check; then status OK "V3.34 CUDA AUTOFIX: CUDA back after stopping the leftover processes"; return 0; fi
  fi
  other_h="$(uvm_holders)"
  if [[ -n "$other_h" ]]; then
    status WARN "V3.34 CUDA AUTOFIX: other programs still use the NVIDIA UVM driver (not stopped automatically):"
    echo "$other_h" | sed 's/^/          /' | tee -a "$STATUS_LOG"
    status WARN "V3.34 CUDA AUTOFIX: close them (or reboot) and start NERO again"
    return 1
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
    status WARN "V3.34 CUDA AUTOFIX: nvidia-smi does not see the GPU either -> the driver itself is down (reboot needed)"
    return 1
  fi
  local rc=0
  run_privileged "${NERO_V334_UVM_RELOAD_CMD:-modprobe -r nvidia_uvm && modprobe nvidia_uvm}" || rc=$?
  if [[ $rc -ne 0 ]]; then
    case $rc in
      98) status WARN "V3.34 CUDA AUTOFIX: no way to get admin rights here (no sudo/pkexec/terminal)";;
      99) status WARN "V3.34 CUDA AUTOFIX: admin step disabled (NERO_V334_CUDA_PRIV=none)";;
      *)  status WARN "V3.34 CUDA AUTOFIX: reloading nvidia_uvm failed or was cancelled (rc=$rc)";;
    esac
    status WARN "V3.34 manual fix:  sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm"
    return 1
  fi
  sleep 1
  if cuda_check; then status OK "V3.34 CUDA AUTOFIX: nvidia_uvm reloaded -> CUDA available again (no reboot needed)"; return 0; fi
  status WARN "V3.34 CUDA AUTOFIX: CUDA still unavailable after reloading nvidia_uvm"
  return 1
}
# ---- V3.34 CUDA AUTOFIX END ------------------------------------------------

status INFO "CUDA self-check with GPU planner venv..."
if ! cuda_check; then
  echo "$CUDA_CHECK_OUT"
  cuda_autofix || fatal "CUDA self-check failed and the automatic repair did not help. Reboot (or: sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm). To avoid it after suspend: enable nvidia-suspend/nvidia-resume services or disable suspend."
fi
echo "$CUDA_CHECK_OUT"

pkill -f '[N]ERO_GPU_Planner_Server.py' 2>/dev/null || true
sleep 0.4
status INFO "Starting cuRobo GPU server: $GPU_SERVER"
nohup "$GPU_PY" "$GPU_SERVER" --config-dir "$GPU_CONFIG" --prefer curobo --prewarm curobo >>"$GPU_LOG" 2>&1 &
GPU_PID=$!
GPU_STATUS="$LOG_DIR/gpu_status.json"
READY=0
for _ in $(seq 1 360); do
  kill -0 "$GPU_PID" 2>/dev/null || break
  if python3 "$CLIENT" >"$GPU_STATUS" 2>/dev/null && grep -q '"server_version": "V2-24"' "$GPU_STATUS"; then
    READY=1; break
  fi
  sleep 0.25
done
[[ "$READY" -eq 1 ]] || fatal "cuRobo GPU server did not become READY"
status OK "GPU server READY PID=$GPU_PID"
cat "$GPU_STATUS"

GPU_SELFTEST="$LOG_DIR/gpu_selftest.json"
python3 "$CLIENT" --self-test curobo >"$GPU_SELFTEST" 2>&1 || { cat "$GPU_SELFTEST"; fatal "cuRobo self-test failed"; }
grep -q '"ok": true' "$GPU_SELFTEST" && grep -q '"flattened_shape"' "$GPU_SELFTEST" \
  || { cat "$GPU_SELFTEST"; fatal "cuRobo pose self-test did not prove a valid trajectory"; }
status OK "cuRobo pose self-test PASS (no robot motion)"

# ============================================================
# 3. ROS2 / MoveIt environment (from V2-7.2)
# ============================================================
set +u
source "$ROS_SETUP"
[[ -f "$BIO_SETUP" ]] && source "$BIO_SETUP" || status WARN "BioIK workspace setup missing: $BIO_SETUP"
[[ -f "$TRAC_SETUP" ]] && source "$TRAC_SETUP" || status WARN "TRAC-IK workspace setup missing: $TRAC_SETUP"
source "$AGX_SETUP"
source "$VENV_ACTIVATE"
if [[ "${V307_RESTART_DAEMON:-0}" == "1" ]]; then
  timeout 8 ros2 daemon stop >/dev/null 2>&1 || true
  status OK "V3.07 ros2 daemon restarted (stale graph cache cleared)"
fi

# NERO modules must resolve to THIS folder. Our folder goes first on sys.path.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export NERO_MODEL_FILE="$MODEL_FILE"
# V3.01 motion diagnostics (record only). Fresh folder each launch.
export NERO_DIAG_DIR="$LOG_DIR/diag"
rm -rf "$NERO_DIAG_DIR" && mkdir -p "$NERO_DIAG_DIR"
export NERO_PLANNER_MODE="${NERO_PLANNER_MODE:-cuRobo Strict}"
export NERO_V23_CUROBO_REPLAN_ATTEMPTS="${NERO_V23_CUROBO_REPLAN_ATTEMPTS:-10}"
export NERO_V24_CAPTURE_ALLOWANCE_M="${NERO_V24_CAPTURE_ALLOWANCE_M:-0.024}"

RESOLVE_OUT="$(cd "$HERE" && python3 - "$HERE" $NERO_MODULES <<'PYRES' 2>&1
import importlib.util, hashlib, os, sys
here = os.path.realpath(sys.argv[1]); bad = 0
for name in sys.argv[2:]:
    spec = importlib.util.find_spec(name)
    origin = os.path.realpath(spec.origin) if spec and spec.origin else None
    ok = bool(origin) and os.path.dirname(origin) == here
    digest = hashlib.sha256(open(origin, "rb").read()).hexdigest()[:16] if origin else "-"
    print(f"[MODULE {'OK ' if ok else 'BAD'}] {name} -> {origin} sha256={digest}")
    bad += 0 if ok else 1
sys.exit(1 if bad else 0)
PYRES
)"
RESOLVE_RC=$?
echo "$RESOLVE_OUT"
echo "$RESOLVE_OUT" >> "$STATUS_LOG"
N_OK="$(echo "$RESOLVE_OUT" | grep -c '^\[MODULE OK \]' || true)"
if [[ "$RESOLVE_RC" -ne 0 || "$N_OK" -ne 4 ]]; then
  fatal "A NERO module does not resolve inside $HERE (ok=$N_OK/4)"
fi
status OK "All NERO modules resolve inside this folder"
if [[ "${NERO_V308_CAMERA:-1}" != "0" ]]; then
  if python3 -c "import OpenGL" >/dev/null 2>&1; then
    status OK "V3.08 simulated D435i camera ON (offscreen GL order: ${NERO_V308_GL:-egl,osmesa,glfw}); captures -> $LOG_DIR/perception"
    status INFO "V3.31 no priors: wrist + fixed camera perceive holder / pile / eggs (generic names, no simulator truth); NERO_V331_FIXED_NO_PRIOR=0 = old fixed path"
    status INFO "V3.36: V3.34 grasp physics (auto HOLD); wrist camera = live re-perception (frozen during a grasp / placement) + calibration error (GUI slider) + measured sizes/heights; no fallback to simulator truth"
    status INFO "V3.37: risky reorientation -> egg carried to the table centre first; safety restore keeps the held egg; latch cleared before return_to_pick_location; dropped egg = LOST + searched; GUI tab 'NERO SimAgent' (foldable sections, monitor zoom)"
    status INFO "V3.38: USB camera JSON fix (numpy-safe payloads); LLM API retries transient network errors; recenter only when needed (IK-screened centre points); carton centre fused at the measured height + slot refine + farther pre-place views"
    status INFO "V3.39: USB wrist camera - faster scan (partial carton views fused, refine/height views at once, empty views reused), lying eggs always checked against the table plane, consensus triangulation, close-view association by measured position, extra close views + height guard in plan_grasp"
    status INFO "V3.40: one locked table plane per episode; bystander check = perceived xy minus noise; pre-place view only refines the fused carton; cuRobo failure -> start-state diagnosis + MoveIt joint-goal fallback; pre-place looks stop after a tracking mismatch"
    status INFO "V3.41: every slot-above / descent-corridor / reorientation goal keeps all 7 joints >= 8 deg from their limits (4 deg fallback); a blocked insertion is a failure (BioIK step -> one yaw re-plan -> INSERT_BLOCKED_JOINT_LIMIT); close/partial live views never flag the carton as moved; held egg pose refreshed after every motion; a re-found lost egg stays found"
    status INFO "V3.44: MuJoCo scene chosen in the NERO console before start (fill_egg_holder default / store_in_safe = RoboDojo safe: hinged door + handle + valuables box); no switching while running; per-step physics loop unchanged"
    status INFO "V3.42: end-of-task statistics (time per skill / object, tokens) + one LLM failure analysis; LLM providers OpenAI / DeepSeek / Kimi; 60 tool steps by default, the operator is asked for more when they run out"
    status INFO "V3.33 scene: grey floor, plain white table, supermarket pulp egg carton (visual; physics unchanged); NERO_V333_FLOOR=wood / NERO_V333_TABLE=marble / NERO_V333_PULP_CARTON=0 restore the old look"
    status INFO "V3.26-V3.30 wrist cameras (D435i / USB 130deg RGB): select 'Wrist camera' in the Agent tab (pose source); captures -> $LOG_DIR/perception_wrist"
  else
    status WARN "V3.08 camera ON but PyOpenGL missing in the MuJoCo venv: EGL/OSMesa unavailable, GLFW fallback (pip install PyOpenGL)"
  fi
else
  status INFO "V3.08 simulated camera OFF (NERO_V308_CAMERA=0)"
fi

command -v ros2 >/dev/null 2>&1 || fatal "ros2 CLI unavailable"
ros2 pkg prefix agx_arm_moveit >/dev/null 2>&1 || fatal "agx_arm_moveit package unavailable"
ros2 pkg prefix bio_ik >/dev/null 2>&1 || fatal "BioIK package 'bio_ik' unavailable"
status OK "BioIK package available: bio_ik"
if ros2 pkg prefix trac_ik_kinematics_plugin >/dev/null 2>&1 || ros2 pkg prefix trac_ik_kinematics >/dev/null 2>&1; then
  status OK "TRAC-IK package available (optional reference)"
else
  status WARN "TRAC-IK plugin unavailable; BioIK-only"
fi

AGX_MOVEIT_PREFIX="$(ros2 pkg prefix agx_arm_moveit)"
KIN_YAML="${AGX_MOVEIT_PREFIX}/share/agx_arm_moveit/config/kinematics.yaml"
[[ -f "$KIN_YAML" ]] || fatal "AgileX kinematics.yaml not found: $KIN_YAML"
[[ -w "$KIN_YAML" ]] || fatal "kinematics.yaml is not writable: $KIN_YAML"
if grep -Eq "NERO_V69_(BIOIK|TRAC_REF)_TEMP" "$KIN_YAML" 2>/dev/null; then
  if [[ -f "$KIN_BACKUP" ]]; then
    cp -f "$KIN_BACKUP" "$KIN_YAML" || fatal "Could not recover stale V69 kinematics.yaml"
    status WARN "Recovered kinematics.yaml left patched by an interrupted run"
  else
    fatal "Stale V69 kinematics marker found but no backup exists"
  fi
fi
cp -f "$KIN_YAML" "$KIN_BACKUP" || fatal "Could not back up kinematics.yaml"
KIN_BACKED_UP=1

write_bio_config(){
cat > "$KIN_YAML" <<EOF
arm:
  kinematics_solver: bio_ik/BioIKKinematicsPlugin
  kinematics_solver_search_resolution: 0.005
  kinematics_solver_timeout: ${BIO_IK_TIMEOUT}
  kinematics_solver_attempts: 1
  mode: ${BIO_IK_MODE}
  minimal_displacement_weight: ${BIO_MIN_DISP_W}
  avoid_joint_limits_weight: ${BIO_AVOID_LIMIT_W}
  center_joints_weight: ${BIO_CENTER_W}
EOF
}
write_trac_config(){
cat > "$KIN_YAML" <<EOF
arm:
  kinematics_solver: trac_ik_kinematics_plugin/TRAC_IKKinematicsPlugin
  kinematics_solver_timeout: ${TRAC_IK_TIMEOUT}
  solve_type: ${TRAC_IK_SOLVE_TYPE}
EOF
}
export NERO_BIO_IK_SOLVER="BioIK" NERO_BIO_IK_MODE="$BIO_IK_MODE" NERO_BIO_IK_TIMEOUT="$BIO_IK_TIMEOUT"
export NERO_BIO_IK_WEIGHTS="minDisp=${BIO_MIN_DISP_W}, avoidLimit=${BIO_AVOID_LIMIT_W}, center=${BIO_CENTER_W}"
export NERO_TRAC_IK_SOLVER="TRAC-IK" NERO_TRAC_IK_SOLVE_TYPE="$TRAC_IK_SOLVE_TYPE" NERO_TRAC_IK_TIMEOUT="$TRAC_IK_TIMEOUT"

root_arm_action_ready(){ timeout 2 ros2 action list 2>/dev/null | grep -qx "/arm_controller/follow_joint_trajectory"; }
root_controller_manager_ready(){ timeout 2 ros2 service list 2>/dev/null | grep -qx "/controller_manager/list_controllers"; }
ensure_root_arm_controller(){
  for _ in $(seq 1 30); do root_arm_action_ready && return 0; sleep 0.3; done
  status WARN "Root arm_controller action missing; inspecting controller lifecycle"
  root_controller_manager_ready || { status FAIL "Root /controller_manager unavailable"; return 1; }
  CONTROLLERS="$(timeout 4 ros2 control list_controllers -c /controller_manager 2>/dev/null || true)"
  echo "$CONTROLLERS" | sed 's/^/[CONTROLLER] /' >> "$STATUS_LOG"
  if echo "$CONTROLLERS" | grep -Eq '^arm_controller[[:space:]]'; then
    if ! echo "$CONTROLLERS" | grep -Eq '^arm_controller[[:space:]].*[[:space:]]active([[:space:]]|$)'; then
      status WARN "arm_controller exists but is not active; activating it"
      timeout 8 ros2 control set_controller_state arm_controller active -c /controller_manager >>"$STATUS_LOG" 2>&1 || true
    fi
  else
    status WARN "arm_controller is not loaded; loading + activating it"
    timeout 10 ros2 control load_controller arm_controller --set-state active -c /controller_manager >>"$STATUS_LOG" 2>&1 || true
  fi
  for _ in $(seq 1 30); do
    root_arm_action_ready && { status OK "Root arm_controller FollowJointTrajectory action READY"; return 0; }
    sleep 0.3
  done
  status FAIL "Could not restore /arm_controller/follow_joint_trajectory"
  return 1
}

# ============================================================
# 4. Backend (MuJoCo + Tk GUI + Agent), run IN PLACE from this folder
# ============================================================
cd "$HERE"
status INFO "Starting backend: $BACKEND"
python3 -u "$BACKEND" >"$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
status OK "Backend started PID=$BACKEND_PID (log: $BACKEND_LOG)"
for i in $(seq 1 100); do
  kill -0 "$BACKEND_PID" 2>/dev/null || { tail -n 40 "$BACKEND_LOG"; fatal "Backend exited during startup"; }
  if timeout 1 ros2 topic list 2>/dev/null | grep -qx "/feedback/joint_states"; then
    status OK "ROS topic READY: /feedback/joint_states"; break
  fi
  [[ "$i" -eq 100 ]] && fatal "Backend bridge timeout: /feedback/joint_states"
  sleep 0.4
done

# ============================================================
# 5. Primary BioIK MoveIt
# ============================================================
write_bio_config
status OK "Temporarily configured root MoveIt for BioIK (mode=$BIO_IK_MODE timeout=${BIO_IK_TIMEOUT}s)"
setsid ros2 launch agx_arm_moveit demo.launch.py \
    arm_type:=nero effector_type:=agx_gripper follow:=true \
    feedback_topic:=feedback/joint_states control_topic:=control/joint_states \
    tcp_offset:='[0.1553, 0.0, -0.0215, -1.5708, 0.0, -1.5708]' \
    use_rviz:="${USE_RVIZ}" >"$BIO_MOVEIT_LOG" 2>&1 &
BIO_MOVEIT_PID=$!
status OK "Primary BioIK MoveIt launch PID=$BIO_MOVEIT_PID"
BIO_READY=0
for i in $(seq 1 150); do
  kill -0 "$BIO_MOVEIT_PID" 2>/dev/null || fatal "Primary BioIK MoveIt exited"
  NODES="$(timeout 1 ros2 node list 2>/dev/null || true)"
  SERVICES="$(timeout 1 ros2 service list 2>/dev/null || true)"
  if echo "$NODES" | grep -qx "/move_group" && echo "$SERVICES" | grep -qx "/compute_ik"; then
    SOLVER="$(timeout 2 ros2 param get /move_group robot_description_kinematics.arm.kinematics_solver 2>/dev/null || true)"
    if echo "$SOLVER" | grep -q "bio_ik/BioIKKinematicsPlugin"; then
      status OK "Primary /move_group IK solver verified: BioIK"; BIO_READY=1; break
    fi
  fi
  sleep 0.4
done
[[ "$BIO_READY" -eq 1 ]] || fatal "BioIK MoveIt did not become ready/verified"
ensure_root_arm_controller || fatal "Primary MoveIt is up, but root arm_controller could not be made ACTIVE"

# ============================================================
# 6. Optional TRAC-IK reference MoveIt (namespace trac_ref)
# ============================================================
write_trac_config
setsid ros2 launch agx_arm_moveit demo.launch.py \
    namespace:=trac_ref arm_type:=nero effector_type:=agx_gripper follow:=true \
    feedback_topic:=/feedback/joint_states control_topic:=control/joint_states \
    tcp_offset:='[0.1553, 0.0, -0.0215, -1.5708, 0.0, -1.5708]' \
    use_rviz:=false >"$TRAC_MOVEIT_LOG" 2>&1 &
TRAC_MOVEIT_PID=$!
status OK "TRAC reference MoveIt launch PID=$TRAC_MOVEIT_PID"
TRAC_READY=0
for i in $(seq 1 150); do
  if ! kill -0 "$TRAC_MOVEIT_PID" 2>/dev/null; then
    status WARN "Optional TRAC reference MoveIt exited; continuing BioIK-only"; TRAC_MOVEIT_PID=""; break
  fi
  NODES="$(timeout 1 ros2 node list 2>/dev/null || true)"
  SERVICES="$(timeout 1 ros2 service list 2>/dev/null || true)"
  if echo "$NODES" | grep -qx "/trac_ref/move_group" && echo "$SERVICES" | grep -qx "/trac_ref/compute_ik"; then
    SOLVER="$(timeout 2 ros2 param get /trac_ref/move_group robot_description_kinematics.arm.kinematics_solver 2>/dev/null || true)"
    if echo "$SOLVER" | grep -q "TRAC_IKKinematicsPlugin"; then
      status OK "Reference /trac_ref/move_group IK solver verified: TRAC-IK"; TRAC_READY=1; break
    fi
  fi
  sleep 0.4
done
[[ "$TRAC_READY" -eq 1 ]] || status WARN "Optional TRAC reference MoveIt not verified; continuing BioIK-only"
if ! ensure_root_arm_controller; then
  status WARN "TRAC startup disturbed root arm_controller; stopping TRAC and retrying"
  if [[ -n "$TRAC_MOVEIT_PID" ]]; then
    kill -- "-$TRAC_MOVEIT_PID" 2>/dev/null || kill "$TRAC_MOVEIT_PID" 2>/dev/null || true
    wait "$TRAC_MOVEIT_PID" 2>/dev/null || true
    TRAC_MOVEIT_PID=""
  fi
  ensure_root_arm_controller || fatal "Mandatory root arm_controller could not be recovered"
fi
restore_kinematics

# ============================================================
# 7. Final readiness
# ============================================================
READY=0
for i in $(seq 1 150); do
  ACTIONS="$(timeout 1 ros2 action list 2>/dev/null || true)"
  SERVICES="$(timeout 1 ros2 service list 2>/dev/null || true)"
  TOPICS="$(timeout 1 ros2 topic list 2>/dev/null || true)"
  if echo "$ACTIONS" | grep -qx "/move_action" \
     && echo "$ACTIONS" | grep -qx "/arm_controller/follow_joint_trajectory" \
     && echo "$SERVICES" | grep -qx "/compute_cartesian_path" \
     && echo "$SERVICES" | grep -qx "/compute_ik" \
     && echo "$TOPICS" | grep -qx "/planning_scene"; then
    READY=1
    break
  fi
  sleep 0.4
done
# V3.07 single-arm-controller check: refuse SYSTEM READY with duplicates.
if [[ "$READY" -eq 1 ]]; then
  N_AS="$(timeout 4 ros2 action info /arm_controller/follow_joint_trajectory 2>/dev/null | sed -n 's/^Action servers: *\([0-9]\+\).*/\1/p' | head -n1)"
  N_CS="$(timeout 4 ros2 topic info /arm_controller/controller_state 2>/dev/null | sed -n 's/^Publisher count: *\([0-9]\+\).*/\1/p' | head -n1)"
  status INFO "V3.07 arm_controller instances: action_servers=${N_AS:-?} controller_state_publishers=${N_CS:-?}"
  if [[ "${N_AS:-1}" -gt 1 || "${N_CS:-1}" -gt 1 ]]; then
    pgrep -af "ros2_control_node|move_group" | sed 's/^/[DUPLICATE] /' | tee -a "$STATUS_LOG"
    fatal "V3.07 DUPLICATE ARM CONTROLLER at startup (action_servers=${N_AS} controller_state=${N_CS}); close everything and restart"
  fi
fi
if [[ "$READY" -ne 1 ]]; then
  tail -n 50 "$BIO_MOVEIT_LOG" | sed 's/^/[BIO MOVEIT] /' >> "$STATUS_LOG" || true
  fatal "Final readiness failed; GUI execution is intentionally blocked"
fi
status READY "SYSTEM READY: NERO ${RELEASE} | BioIK primary + cuRobo GPU + MuJoCo | TRAC optional | single arm_controller"

status INFO "Close the NERO GUI to stop everything."
wait "$BACKEND_PID"
