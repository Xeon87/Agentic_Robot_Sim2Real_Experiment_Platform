#!/bin/bash
# Dev helper (not used at runtime): run every test file and print its last lines, one row per file.
#   bash tools/dev/run_tests.sh <release folder>        non-GUI tests (system python3)
#   bash tools/dev/run_tests.sh <release folder> gui    GUI tests (needs DISPLAY; uses xvfb-run when DISPLAY is unset)
# Save the output and diff it against the previous release's output: no new failures before delivery.
cd "${1:-.}" || exit 1
PY=${PY:-python3}
if [[ "${2:-}" == "gui" ]]; then
  for t in tests/test_*_gui.py; do
    if [[ -n "$DISPLAY" ]]; then r=$(timeout 400 $PY "$t" 2>&1 | tail -2 | tr '\n' ' ')
    else r=$(timeout 400 xvfb-run -a -s "-screen 0 1600x1000x24" $PY "$t" 2>&1 | tail -2 | tr '\n' ' '); fi
    echo "$t :: $r"
  done
else
  for t in tests/test_v3_*.py tests/test_v4_*.py; do
    case "$t" in *_gui.py) continue;; esac
    r=$(timeout 900 $PY "$t" 2>&1 | tail -3 | tr '\n' ' ')
    echo "$t :: $r"
  done
fi
