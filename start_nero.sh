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
REALCUP365="$HERE/NERO_V365_Real_Cup_Task.py"  # V3.65 NERO实机任务 tab (stack_cups_so101 skills on the real arm)
GRIP365="$HERE/NERO_V365_STS3215_Gripper.py"   # V3.65 SO-101 jaw on the STS3215 bus servo
CUPYOLO366="$HERE/NERO_V366_Real_Cup_Yolo.py"   # V3.66 real cup perception = the Camera tab's YOLOE pipeline
VISION374="$HERE/NERO_V374_Vision_Review.py"    # V3.74 LLM vision review of the real cup perception
TASKREC368="$HERE/NERO_V368_Real_Task_Record.py"  # V3.68 task records (before / after images) + summary
STALE377="$HERE/NERO_V377_Stale_Procs.py"     # V3.77 STALE PROCESS CHECK (console: before a simulation starts)
COLL375="$HERE/NERO_V375_Collision_Check.py"   # V3.75 COLLISION PRE-CHECK (sort / cup skills, MoveIt contact pairs)
SORT372="$HERE/NERO_V372_Sort_Skills.py"      # V3.72 SORT ITEMS AGENT (sort_items scene skills, imported by the backend)

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
for f in "$SIM" "$CONSOLE" "$REAL" "$JOINT" "$CONS" "$SAFE" "$CAM" "$YOLO347" "$WRIST349" "$STRAT359" "$REALCUP365" "$GRIP365" "$CUPYOLO366" "$TASKREC368" "$SORT372" "$VISION374" "$COLL375" "$STALE377" "$HERE/SHA256SUMS.txt"; do
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
"$PY" -m py_compile "$CONSOLE" "$REAL" "$JOINT" "$CONS" "$SAFE" "$STRAT359" "$REALCUP365" "$GRIP365" "$CUPYOLO366" "$TASKREC368" "$VISION374" "$COLL375" "$STALE377" || die "py_compile failed: $CONSOLE / $REAL / $JOINT / $CONS / $SAFE"
"$PY" -c "import tkinter" 2>/dev/null || die "tkinter missing for $PY (sudo apt install python3-tk), or run the simulation directly: ./start_nero.sh --sim"
bash -n "$SIM" || die "syntax error in $SIM"

# V3.71 GPU ENV: an inherited CUDA_VISIBLE_DEVICES (e.g. empty) hides the GPU from YOLOE in the console - the
# simulation launcher already clears it the same way
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  echo "[INFO] V3.71: clearing inherited CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES}' (YOLOE GPU)"
  unset CUDA_VISIBLE_DEVICES
fi

ARGS=()
[[ "${1:-}" == "--real" ]] && ARGS+=(--real)
cd "$HERE"
exec "$PY" "$CONSOLE" "${ARGS[@]}"
