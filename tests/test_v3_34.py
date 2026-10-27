#!/usr/bin/env python3
"""V3.34: launcher repairs "CUDA is not available" without a reboot.

The real V3.34 CUDA AUTOFIX block of start_nero_sim.sh is run in bash with stub programs on PATH
(GPU python, sudo, modprobe, nvidia-smi, pkill):
U1 transient failure -> OK after the short retry, no admin step
U2 broken nvidia_uvm -> ONE privileged 'modprobe -r nvidia_uvm && modprobe nvidia_uvm' -> CUDA OK
U3 another (non-NERO) program holds /dev/nvidia-uvm -> nothing killed, no reload, clear message, fail
U4 NERO_V334_CUDA_AUTOFIX=0 -> no repair attempt
U5 no admin rights possible -> manual command printed, fail
U6 the launcher calls cuda_autofix before fatal; executable bits kept
"""
import os, re, subprocess, tempfile, textwrap
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
L = (ROOT / "start_nero_sim.sh").read_text()
a = L.index("# ---- V3.34 CUDA AUTOFIX BEGIN")
b = L.index("# ---- V3.34 CUDA AUTOFIX END")
BLOCK = L[a:b]


def run(scenario, env_extra=None, holders=""):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td); bin_ = td / "bin"; bin_.mkdir()
        state = td / "state"; state.mkdir()
        (state / "fails").write_text({"transient": "1", "uvm": "99", "other": "99", "off": "99", "nopriv": "99"}[scenario])

        def stub(name, body):
            p = bin_ / name
            p.write_text("#!/bin/bash\n" + textwrap.dedent(body))
            p.chmod(0o755)
        # GPU python: fails while state/fails > 0 and the module was not reloaded
        stub("gpupy", f"""
            cat >/dev/null
            n=$(cat {state}/fails)
            if [[ -f {state}/reloaded ]]; then echo "cuda_available=True"; exit 0; fi
            if [[ "$n" -gt 0 ]]; then echo $((n-1)) > {state}/fails; echo "CUDA_CHECK_ERROR: RuntimeError: CUDA is not available"; exit 1; fi
            echo "cuda_available=True"; exit 0
        """)
        stub("sudo", f"""
            if [[ "$1" == "-n" && "$2" == "true" ]]; then [[ -f {state}/nosudo ]] && exit 1; exit 0; fi
            [[ "$1" == "-n" ]] && shift
            echo "$@" >> {state}/sudo_calls
            "$@"
        """)
        stub("modprobe", f"""echo "modprobe $@" >> {state}/modprobe_calls; [[ "$1" != "-r" ]] && touch {state}/reloaded; exit 0""")
        stub("nvidia-smi", "echo 'GPU 0: NVIDIA RTX'; exit 0")
        stub("pkill", "exit 0")
        if scenario == "nopriv":
            (state / "nosudo").write_text("1")
        script = td / "t.sh"
        script.write_text(f"""#!/bin/bash
set -o pipefail
STATUS_LOG={td}/status.log
status(){{ echo "[$1] ${{@:2}}"; echo "[$1] ${{@:2}}" >> "$STATUS_LOG"; }}
GPU_PY={bin_}/gpupy
{BLOCK}
uvm_holders(){{ printf '%s' "{holders}"; }}
if cuda_check; then echo RESULT_OK_FIRST; exit 0; fi
if cuda_autofix; then echo RESULT_FIXED; else echo RESULT_FAILED; fi
""")
        env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", NERO_V334_CUDA_RETRY_S="0")
        env.pop("DISPLAY", None); env.pop("WAYLAND_DISPLAY", None)
        env.update(env_extra or {})
        out = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, stdin=subprocess.DEVNULL,
                             timeout=60).stdout
        calls = (state / "modprobe_calls").read_text() if (state / "modprobe_calls").exists() else ""
        return out, calls


out, calls = run("transient")
assert "RESULT_FIXED" in out and "transient" in out and calls == "", out
print("[U1] transient failure -> fixed by the short retry, no admin step")
out, calls = run("uvm")
assert "RESULT_FIXED" in out and "reloaded" in out and calls.splitlines() == ["modprobe -r nvidia_uvm", "modprobe nvidia_uvm"], (out, calls)
print("[U2] broken nvidia_uvm -> one privileged reload (sudo) -> CUDA available, no reboot")
out, calls = run("other", holders="4242 /usr/bin/python3 some_other_cuda_app.py")
assert "RESULT_FAILED" in out and calls == "" and "some_other_cuda_app" in out and "not stopped automatically" in out, out
print("[U3] a non-NERO program holds /dev/nvidia-uvm -> listed, not killed, no reload")
out, calls = run("off", {"NERO_V334_CUDA_AUTOFIX": "0"})
assert "RESULT_FAILED" in out and calls == "" and "AUTOFIX" not in out, out
print("[U4] NERO_V334_CUDA_AUTOFIX=0 -> no repair attempt")
out, calls = run("nopriv")
assert "RESULT_FAILED" in out and calls == "" and "sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm" in out, out
print("[U5] no admin rights available (no passwordless sudo, no display, no terminal) -> manual command printed")
assert re.search(r"if ! cuda_check; then\n  echo \"\$CUDA_CHECK_OUT\"\n  cuda_autofix \|\| fatal", L)
assert 'if [[ "$mode" == "auto" || "$mode" == "pkexec" ]]' in L and "pkexec /bin/sh -c" in L
r = subprocess.run(["bash", "-n", str(ROOT / "start_nero_sim.sh")]); assert r.returncode == 0
for f in ("start_nero.sh", "start_nero_sim.sh", "tests/check_release.sh"):
    assert os.access(ROOT / f, os.X_OK), f
print("[U6] launcher: CUDA failure -> automatic repair before giving up (password dialog via pkexec on the desktop); "
      "launchers executable")
print("ALL V3.34 TESTS PASS")
