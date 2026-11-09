#!/usr/bin/env python3
"""V3.70 offline unit tests:
A1 Versions V3.70, markers present; NERO_V308 untouched; test_v3_44 backend invariants still hold (backend unchanged).
A2 Crash fix 1: Camera tab 「退出示教模式」 - the safety editor's mode switch + redraw (on_mode) runs in the Tk main
   thread (queued with _ui), not in the worker thread that talks to the arm.
A3 Crash fix 2: install_crash_log -> logs/crash.log: faulthandler enabled; uncaught exceptions of worker threads and
   Tk callbacks are written there; a real segfault in a child process leaves the faulting Python stack (file + line).
A4 Crash fix 3: real-page lines ([V3.xx ...] from the real NERO / Camera / skills / real task tabs) are mirrored into
   logs/console.log (v370_mirror_lines, wired in real_poll).
A5 Teach payload: apply_teach_payload uses set_payload('empty') when the SDK offers it, WARNs when not ('不设置'
   skips); cmd_mode(teach) sends it BEFORE set_leader_mode; diag lists load / payload APIs; log lines.
A6 Teach payload plumbing: console 「示教负载」 (default empty) -> real_cfg teach_payload -> _run_mode_helper --payload
   (only on teach entry) -> CLI parser; payload report logged by the pipeline.
"""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="nero_v370_"))

import NERO_V312_Console as CON               # noqa: E402
import NERO_V313_Real_Robot as RR             # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V347_Yolo_Perception as YP        # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49     # noqa: E402
import NERO_V365_Real_Cup_Task as RCT         # noqa: E402

H = lambda b: hashlib.sha256(b).hexdigest()

# ---------------------------------------------------------------- A1
for m in (CON, RR, YP, W49, RCT):
    assert m.VERSION in ("V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11"), (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
s_con = (ROOT / "NERO_V312_Console.py").read_text(encoding="utf-8")
s_rr = (ROOT / "NERO_V313_Real_Robot.py").read_text(encoding="utf-8")
s_cam = (ROOT / "NERO_V325_Camera.py").read_text(encoding="utf-8")
assert "V3.70 CRASH LOG" in s_con and "V3.70 CONSOLE LOG MIRROR" in s_con and "V3.70 TEACH PAYLOAD" in s_con
assert "V3.70 TEACH PAYLOAD" in s_rr and "V3.70 TK MAIN THREAD" in s_cam
r = subprocess.run([sys.executable, str(ROOT / "tests" / "test_v3_44.py")], capture_output=True, text=True, timeout=600)
assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
print("[A1] V3.70 versions + markers; NERO_V308 untouched; test_v3_44 backend invariants pass")

# ---------------------------------------------------------------- A2
calls = {}


class _V:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v

    def set(self, v):
        self.v = v


class SafeTab:
    mode = "teach"

    def __init__(self):
        self.v_mode = _V("teach")
        self._teach_exiting = False

    def on_mode(self):
        calls["on_mode_thread"] = threading.current_thread()
        calls["exiting"] = self._teach_exiting
        self.mode = self.v_mode.get()


class Pipe:
    def set_teach_mode(self, on, home_model=None, return_home=True):
        calls["teach"] = (on, return_home, threading.current_thread())
        return {"ok": True}


fake = types.SimpleNamespace(_ui_q=[], safe_tab=SafeTab(), pipe=Pipe(), logs=[])
fake._busy = lambda: False
fake.ask = lambda a, b: True
fake.log = fake.logs.append
fake._model = lambda: {"limits": []}
fake._update_calib_ui = lambda: None
fake._ui = lambda fn: fake._ui_q.append(fn)
workers = []


def _run(job):
    t = threading.Thread(target=job, name="v370_worker")
    workers.append(t)
    t.start()


fake._run = _run
CAM.CameraTabUI.on_teach_exit(fake)
workers[0].join(5)
assert "on_mode_thread" not in calls, "safety editor redraw ran in the worker thread"
assert calls["teach"][0] is False and calls["teach"][1] is False and calls["teach"][2].name == "v370_worker"
while fake._ui_q:                                         # = CameraTabUI._poll in the Tk main thread
    fake._ui_q.pop(0)()
assert calls["on_mode_thread"] is threading.main_thread() and calls["exiting"] is True
assert fake.safe_tab.mode == "edit" and fake.safe_tab._teach_exiting is False
assert any("已退出示教模式" in s for s in fake.logs)
print("[A2] 退出示教模式: arm call in the worker, safety-editor switch + redraw in the Tk main thread")

# ---------------------------------------------------------------- A3
logd = TMP / "logs"
path = CON.install_crash_log(str(logd))
import faulthandler                                       # noqa: E402
assert path == str(logd / "crash.log") and faulthandler.is_enabled()


def _boom():
    raise RuntimeError("v370 worker boom")


t = threading.Thread(target=_boom, name="v370_boom")
with contextlib.redirect_stderr(io.StringIO()):
    t.start()
    t.join()


class Root:
    pass


root = Root()
CON.install_crash_log(str(logd), root)
try:
    raise ValueError("v370 tk callback boom")
except ValueError as exc:
    with contextlib.redirect_stderr(io.StringIO()):
        root.report_callback_exception(type(exc), exc, exc.__traceback__)
txt = (logd / "crash.log").read_text(encoding="utf-8")
assert "console start" in txt and "v370 worker boom" in txt and "thread v370_boom" in txt, txt
assert "v370 tk callback boom" in txt and "Tk callback" in txt, txt
faulthandler.disable()
child = TMP / "child"
code = (f"import sys; sys.path.insert(0, {str(ROOT)!r})\n"
        "import NERO_V312_Console as CON, faulthandler\n"
        f"CON.install_crash_log({str(child)!r})\n"
        "def segv_here():\n"
        "    faulthandler._sigsegv()\n"
        "segv_here()\n")
cf = TMP / "crash_child.py"
cf.write_text(code, encoding="utf-8")
r = subprocess.run([sys.executable, str(cf)], capture_output=True, text=True, timeout=60)
assert r.returncode != 0
ct = (child / "crash.log").read_text(encoding="utf-8")
assert "Segmentation fault" in ct and "crash_child.py" in ct and "segv_here" in ct, ct
print("[A3] logs/crash.log: thread + Tk callback exceptions recorded; child segfault left its stack:",
      [ln.strip() for ln in ct.splitlines() if "crash_child.py" in ln][:1])

# ---------------------------------------------------------------- A4
cl = TMP / "console.log"
cl.write_text("[sim] earlier line\n", encoding="utf-8")
CON.v370_mirror_lines(str(cl), ["[OK] [V3.22 teach] 进入示教模式", "[INFO] [V3.55 CALIB] 自动采集 1/6"])
CON.v370_mirror_lines(str(cl), [])
lines = cl.read_text(encoding="utf-8").splitlines()
assert lines[0] == "[sim] earlier line" and len(lines) == 3
assert lines[1].endswith("REAL] [OK] [V3.22 teach] 进入示教模式") and "[V3.55 CALIB]" in lines[2]
assert "v370_mirror_lines(console_log, mirrored)" in s_con and "mirrored.append(ln)" in s_con
assert "install_crash_log(log_dir, root)" in s_con
print("[A4] real-page lines mirrored into console.log (appended, sim output kept)")

# ---------------------------------------------------------------- A5


class ArmWithPayload:
    def __init__(self):
        self.seq = []

    def set_payload(self, load, timeout=1.0):
        self.seq.append(("set_payload", load))
        return True


class ArmNoPayload:
    pass


a = ArmWithPayload()
rep = RR.apply_teach_payload(a, "empty")
assert rep["ok"] is True and rep["api"] == "set_payload" and a.seq == [("set_payload", "empty")]
assert "[OK] [V3.70 teach] 示教负载 = empty" in RR.teach_payload_line(rep)
rep = RR.apply_teach_payload(ArmNoPayload(), "empty")
assert rep["ok"] is False and rep["api"] is None
ln = RR.teach_payload_line(rep)
assert ln.startswith("[WARN] [V3.70 teach]") and "原因" in ln and "下一步" in ln and "配重" in ln
rep = RR.apply_teach_payload(ArmNoPayload(), "不设置")
assert rep["ok"] is None and RR.teach_payload_line(rep).startswith("[INFO]")
assert RR.apply_teach_payload(a, "heavy")["ok"] is False
assert RR.DEFAULT_TEACH_PAYLOAD == "empty" and RR.PAYLOAD_CHOICES == ("empty", "half", "full")

seq = []


class FakeNeroArm:
    def __init__(self, with_payload):
        self.leader = False
        if with_payload:
            self.set_payload = lambda load, timeout=1.0: seq.append(("set_payload", load)) or True

    def connect(self):
        seq.append(("connect",))

    def get_arm_status(self):
        return {"ctrl_mode": 0x06 if self.leader else 0x01}

    def set_leader_mode(self):
        seq.append(("set_leader_mode",))
        self.leader = True

    def set_follower_mode(self):
        seq.append(("set_follower_mode",))

    def enable(self):
        return True


def fake_sdk(with_payload):
    mod = types.ModuleType("pyAgxArm")

    class NeroFW:
        DEFAULT, V111, V112, V120, V121 = "DEFAULT", "V111", "V112", "V120", "V121"

    mod.NeroFW = NeroFW
    mod.ArmModel = types.SimpleNamespace(NERO="NERO")
    mod.create_agx_arm_config = lambda **kw: kw
    mod.AgxArmFactory = types.SimpleNamespace(create_arm=lambda cfg: FakeNeroArm(with_payload))
    return mod


def run_mode(with_payload, payload):
    seq.clear()
    sys.modules["pyAgxArm"] = fake_sdk(with_payload)
    out = io.StringIO()
    sleep0 = RR.time.sleep
    RR.time.sleep = lambda s: None
    try:
        with contextlib.redirect_stdout(out):
            rc = RR.cmd_mode("can0", "v121", "teach", verify_s=0.3, payload=payload)
    finally:
        RR.time.sleep = sleep0
        sys.modules.pop("pyAgxArm", None)
    evs = [json.loads(x) for x in out.getvalue().splitlines() if x.startswith("{")]
    return rc, evs


rc, evs = run_mode(True, "empty")
pl = next(e for e in evs if e.get("event") == "payload")
assert rc == 0 and pl["ok"] is True and pl["api"] == "set_payload"
assert seq.index(("set_payload", "empty")) < seq.index(("set_leader_mode",)), seq
diag = next(e for e in evs if e.get("event") == "diag")
assert "set_payload" in diag["apis"] and "set_leader_mode" in diag["apis"], diag
rc, evs = run_mode(False, "empty")
pl = next(e for e in evs if e.get("event") == "payload")
assert rc == 0 and pl["ok"] is False and ("set_leader_mode",) in seq      # teach entry still works without the API
rc, evs = run_mode(False, None)
assert next(e for e in evs if e.get("event") == "payload")["payload"] is None and rc == 0
print("[A5] payload sent before set_leader_mode when the SDK has set_payload; WARN + teach still entered when not")

# ---------------------------------------------------------------- A6
assert "v_payload = tk.StringVar(value=RR.DEFAULT_TEACH_PAYLOAD)" in s_con
assert '"teach_payload": v_payload.get()' in s_con and 'text="示教负载"' in s_con
seen = {}


class P:
    pass


pipe = types.SimpleNamespace(cfg_fn=lambda: {"can_port": "can0", "push_fw": "v121", "teach_payload": "empty"},
                             EXIT_SPEED=0.10, logs=[])
pipe.log = pipe.logs.append
pipe._fw_cfg = lambda: RR.RealPipeline._fw_cfg(pipe)


def fake_run(argv, capture_output=True, text=True, timeout=None):
    seen.setdefault("argv", []).append(list(argv))
    lines = [json.dumps({"event": "diag", "apis": ["set_leader_mode"], "ctrl_mode_before": 1, "fw": "V121"}),
             json.dumps({"event": "payload", "payload": "empty", "api": None, "ok": False,
                         "note": "当前 pyAgxArm 的 NERO 驱动没有负载设置接口（set_payload 等）"}),
             json.dumps({"event": "mode_result", "ok": True, "mode": "teach", "api": "set_leader_mode"})]
    return types.SimpleNamespace(stdout="\n".join(lines), stderr="", returncode=0)


run0 = RR.subprocess.run
RR.subprocess.run = fake_run
try:
    ev = RR.RealPipeline._run_mode_helper(pipe, True, 5.0)
    RR.RealPipeline._run_mode_helper(pipe, False, 5.0, hold=[0.0] * 7)
finally:
    RR.subprocess.run = run0
a_on, a_off = seen["argv"]
assert ev["ok"] and a_on[a_on.index("--payload") + 1] == "empty" and "--payload" not in a_off
assert any(s.startswith("[WARN] [V3.70 teach] 示教负载 empty") for s in pipe.logs), pipe.logs
ap_out = io.StringIO()
cm0 = RR.cmd_mode
got = {}
RR.cmd_mode = lambda *a, **k: got.update(a=a, k=k) or 0
try:
    RR.main(["mode", "--mode", "teach", "--fw", "v121", "--payload", "half"])
finally:
    RR.cmd_mode = cm0
assert got["k"]["payload"] == "half", got
print("[A6] 示教负载 (default empty) -> real_cfg -> --payload on teach entry only -> CLI -> logged")

print("ALL V3.70 TESTS PASSED")
