#!/usr/bin/env bash
# ============================================================
# NERO V3.16 entry point — V3.12 CONSOLE ENTRY
#
#   ./start_nero.sh          -> opens the NERO console (nothing else starts)
#                               仿真 tab: button runs start_nero_sim.sh
#                               实机 tab: V3.13 six-step real-robot bring-up + V3.14 joint panel + V3.15 consistency / safety + V3.16 safety editor
#   ./start_nero.sh --sim    -> old behaviour: run the full simulation chain
#                               directly (start_nero_sim.sh), no console
#   ./start_nero.sh --real   -> console with the 实机NERO调试 tab selected
#   ./start_nero.sh --no-vla -> V4.5.1: console without the Smol VLA实机任务 tab (emergency switch)
#
# Every NERO file used comes from THIS folder.
# ============================================================
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE="$(cat "$HERE/VERSION" 2>/dev/null || echo unknown)"
SIM="$HERE/start_nero_sim.sh"
CONSOLE="$HERE/NERO_V312_Console.py"
REAL="$HERE/NERO_V313_Real_Robot.py"
JOINT="$HERE/NERO_V314_Joint_Control.py"
CONS="$HERE/NERO_V315_Consistency_Safety.py"
SAFE="$HERE/NERO_V316_Safety_Editor.py"
CAM="$HERE/NERO_V325_Camera.py"
YOLO347="$HERE/NERO_V347_Yolo_Perception.py"   # V3.47 YOLOE egg perception (Camera调试 → 视觉定位)
WRIST349="$HERE/NERO_V349_Real_Wrist_Camera.py" # V3.49 real USB wrist camera + intrinsics + eye-in-hand + holder
STRAT359="$HERE/NERO_V359_Strategy.py"         # V3.59 skill strategy knobs + skill card + hover dry-run
REALCUP365="$HERE/NERO_V365_Real_Cup_Task.py"  # V3.65 Agentic实机任务 tab (was NERO实机任务) (stack_cups_so101 skills on the real arm)
GRIP365="$HERE/NERO_V365_STS3215_Gripper.py"   # V3.65 SO-101 jaw on the STS3215 bus servo
CUPYOLO366="$HERE/NERO_V366_Real_Cup_Yolo.py"   # V3.66 real cup perception = the Camera tab's YOLOE pipeline
VISION374="$HERE/NERO_V374_Vision_Review.py"    # V3.74 LLM vision review of the real cup perception
TASKREC368="$HERE/NERO_V368_Real_Task_Record.py"  # V3.68 task records (before / after images) + summary
BENCH378="$HERE/NERO_V378_D435i_Bench.py"    # V3.78 D435I BENCH (standalone: profile / noise / accuracy / capture)
LAB379="$HERE/NERO_V379_Camera_Lab.py"      # V3.79 CAMERA LAB (Camera tab 「相机高级测试」 window)
TRAIN381="$HERE/NERO_V381_Yolo_Train.py"   # V3.81 YOLOE TRAIN (camera lab page 5 + training child)
GPU383="$HERE/NERO_V383_Gpu_Check.py"      # V3.83 GPU PREFLIGHT CHECK & SELF-REPAIR
TRACK384="$HERE/NERO_V384_Object_Tracker.py" # V3.84 MULTI-FRAME OBJECT TRACKER
WRIST386="$HERE/NERO_V386_Real_Wrist_Camera.py" # V3.86 INNOMAKER 4MP OV4688 WRIST CAMERA + WRIST LAB
STALE377="$HERE/NERO_V377_Stale_Procs.py"     # V3.77 STALE PROCESS CHECK (console: before a simulation starts)
COLL375="$HERE/NERO_V375_Collision_Check.py"   # V3.75 COLLISION PRE-CHECK (sort / cup skills, MoveIt contact pairs)
SORT372="$HERE/NERO_V372_Sort_Skills.py"      # V3.72 SORT ITEMS AGENT (sort_items scene skills, imported by the backend)
STORE44="$HERE/NERO_V44_Storage.py"           # V4.4 EMBODIED_AI DRIVE STORAGE (LeRobot venv / models / datasets / training on the drive)
WORKER44="$HERE/NERO_V44_SmolVLA_Worker.py"    # V4.4 SMOLVLA DRIVE WORKER (runs inside the drive venv)
VLATAB4="$HERE/NERO_V4_Console_Tab.py"         # V4.x VLA / LeRobot console tab

if [[ "${1:-}" == "--sim" ]]; then
  shift
  exec bash "$SIM" "$@"
fi

die(){
  echo "[FAIL] $*"
  if [[ "${NERO_NO_FAIL_PAUSE:-0}" != "1" && -r /dev/tty ]]; then
    read -r -p 'Press Enter to close this window...' _ </dev/tty || true
  fi
  exit 1
}

echo "============================================================"
echo " NERO ${RELEASE} — console"
echo " Folder: $HERE"
echo "============================================================"
for f in "$SIM" "$CONSOLE" "$REAL" "$JOINT" "$CONS" "$SAFE" "$CAM" "$YOLO347" "$WRIST349" "$STRAT359" "$REALCUP365" "$GRIP365" "$CUPYOLO366" "$TASKREC368" "$SORT372" "$VISION374" "$COLL375" "$STALE377" "$BENCH378" "$LAB379" "$TRAIN381" "$STORE44" "$WORKER44" "$VLATAB4" "$HERE/SHA256SUMS.txt"; do
  [[ -f "$f" ]] || die "NERO file missing from this folder: $f"
done
if ( cd "$HERE" && sha256sum --quiet -c SHA256SUMS.txt ) >/dev/null 2>&1; then
  echo "[OK] Release integrity: all files match SHA256SUMS.txt"
else
  echo "[WARN] Release integrity: files differ from SHA256SUMS.txt:"
  ( cd "$HERE" && sha256sum -c SHA256SUMS.txt 2>/dev/null | grep -v ': OK$' ) | sed 's/^/          /'
  [[ "${NERO_ALLOW_MODIFIED:-0}" == "1" ]] || die "Refusing to start a modified release. Set NERO_ALLOW_MODIFIED=1 to run anyway."
fi

# The console uses only the Python standard library + tkinter.
PY="${NERO_CONSOLE_PYTHON:-/usr/bin/python3}"
[[ -x "$PY" ]] || PY="$(command -v python3)"
"$PY" -m py_compile "$CONSOLE" "$REAL" "$JOINT" "$CONS" "$SAFE" "$STRAT359" "$REALCUP365" "$GRIP365" "$CUPYOLO366" "$TASKREC368" "$VISION374" "$COLL375" "$STALE377" "$BENCH378" "$LAB379" "$TRAIN381" "$STORE44" "$WORKER44" "$VLATAB4" || die "py_compile failed: $CONSOLE / $REAL / $JOINT / $CONS / $SAFE"
"$PY" -c "import tkinter" 2>/dev/null || die "tkinter missing for $PY (sudo apt install python3-tk), or run the simulation directly: ./start_nero.sh --sim"
bash -n "$SIM" || die "syntax error in $SIM"

# V3.71 GPU ENV: an inherited CUDA_VISIBLE_DEVICES (e.g. empty) hides the GPU from YOLOE in the console - the
# simulation launcher already clears it the same way
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  echo "[INFO] V3.71: clearing inherited CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' (YOLOE GPU)"
  unset CUDA_VISIBLE_DEVICES
fi

# V3.83 GPU PREFLIGHT: test CUDA in child process, auto-repair nvidia_uvm if needed
"$PY" "$GPU383" --preflight || true

# V4.4 EMBODIED_AI DRIVE: LeRobot + SmolVLA, models, datasets and training runs live on the removable drive
"$PY" "$STORE44" --brief || true

ARGS=()
for a in "$@"; do
  case "$a" in
    --real) ARGS+=(--real) ;;
    --no-vla) export NERO_NO_VLA_TAB=1; echo "[INFO] V4.5.1: --no-vla -> console without the Smol VLA实机任务 tab" ;;
  esac
done
cd "$HERE"
# V4.6.3 NATIVE CRASH GUARD: keep the C-level reason of a crash (Tcl / Xlib / glibc print it to stderr right before
# an abort) in logs/stderr.log as well as on the terminal; glibc writes its fatal messages to stderr, not /dev/tty.
mkdir -p "$HERE/logs"
if [[ -f "$HERE/logs/stderr.log" ]] && (( $(stat -c %s "$HERE/logs/stderr.log" 2>/dev/null || echo 0) > 5000000 )); then
  mv -f "$HERE/logs/stderr.log" "$HERE/logs/stderr.prev.log"
fi
echo "----- console start $(date '+%F %T') (stderr of the console process) -----" >> "$HERE/logs/stderr.log"
export LIBC_FATAL_STDERR_=1
# V4.6.6 NATIVE CRASH TRACE: a tiny signal handler (tools/nero_crashtrace.c, built once with gcc into ~/.cache/nero)
# writes the C-level stack (function + library) of a crash to logs/native_crash.log.  No gcc -> skipped (noted in
# crash.log).  NERO_NO_NATIVE_TRACE=1 switches it off.
if [[ "${NERO_NO_NATIVE_TRACE:-0}" != "1" ]] && command -v gcc >/dev/null 2>&1; then
  _ct_src="$HERE/tools/nero_crashtrace.c"
  _ct_dir="${XDG_CACHE_HOME:-$HOME/.cache}/nero"
  _ct_so="$_ct_dir/nero_crashtrace_$(sha256sum "$_ct_src" | cut -c1-12).so"
  if [[ ! -f "$_ct_so" ]]; then
    mkdir -p "$_ct_dir" && gcc -shared -fPIC -O1 -o "$_ct_so" "$_ct_src" 2>>"$HERE/logs/stderr.log" || _ct_so=""
  fi
  if [[ -n "$_ct_so" && -f "$_ct_so" ]]; then
    export NERO_NATIVE_CRASH_LOG="$HERE/logs/native_crash.log"
    export LD_PRELOAD="$_ct_so${LD_PRELOAD:+:$LD_PRELOAD}"
  fi
fi
exec "$PY" "$CONSOLE" "${ARGS[@]}" 2> >(tee -a "$HERE/logs/stderr.log" >&2)
