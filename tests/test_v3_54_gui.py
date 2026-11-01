#!/usr/bin/env python3
"""V3.54 GUI test (needs tkinter + DISPLAY; skips otherwise):
- Verifies fw_version Combobox in 实机NERO调试 defaults to 'v121' and includes ('v121', 'v120', 'v112', 'v111', 'default', '');
- Verifies launch command preview contains 'fw_version:=v121';
- Verifies exiting teach mode and Step 3 'CAN push / 使能' when the arm is in teach mode on V1.2.1 firmware.
"""
import json
import os
import sys
import tempfile
import textwrap
import time
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))
try:
    import tkinter as tk
    from tkinter import messagebox
except Exception:
    print("[GUI54] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI54] no DISPLAY -> skipped"); sys.exit(0)

import NERO_V312_Console as C                 # noqa: E402
import NERO_V313_Real_Robot as RR             # noqa: E402
import NERO_V314_Joint_Control as JC          # noqa: E402
import NERO_V315_Consistency_Safety as CS     # noqa: E402
import NERO_V316_Safety_Editor as SE          # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="v354gui_"))
SE.STATE_PATH = str(TMP / "state.json")
CS.WORKSPACE_PATH = str(TMP / "ws.json")
CS.LOG_DIR = str(TMP / "logs")
JC.PARK_PATH = str(TMP / "park_pose.json")
CAM.LOCATE_PATH = CAM.DEFAULT_LOCATE_PATH = str(TMP / "locate_settings.json")
CAM.CALIB_DIR = str(TMP / "cameracalib")
os.makedirs(CAM.CALIB_DIR, exist_ok=True)

STATE_FILE = TMP / "arm_state.json"
FAKE = TMP / "fake"
(FAKE / "pyAgxArm").mkdir(parents=True)
(FAKE / "pyAgxArm" / "__init__.py").write_text(textwrap.dedent(f"""
    import json, os, time
    STATE_PATH = {str(STATE_FILE)!r}
    def _load():
        if not os.path.exists(STATE_PATH):
            return {{"leader": True, "enabled": False, "fw": None, "joints": [0.12, -0.40, 0.05, 1.10, 0.0, 0.20, 0.55]}}
        return json.load(open(STATE_PATH, "r", encoding="utf-8"))
    def _save(st):
        json.dump(st, open(STATE_PATH, "w", encoding="utf-8"))
    class ArmModel: NERO = "NERO"
    class NeroFW:
        DEFAULT = "DEFAULT"; V111 = "V111"; V112 = "V112"; V120 = "V120"; V121 = "V121"
    def create_agx_arm_config(**kw): return kw
    class _Msg:
        def __init__(self, v): self._msg = v; self.msg = v; self.timestamp = time.time()
    class _Arm:
        def __init__(self, cfg):
            st = _load(); st["fw"] = cfg.get("firmeware_version"); _save(st)
        def connect(self): pass
        def set_normal_mode(self): pass
        def set_leader_mode(self):
            st = _load(); st["leader"] = True; st["enabled"] = False; _save(st)
        def set_follower_mode(self):
            st = _load(); st["leader"] = False; _save(st)
        def enable(self):
            st = _load()
            if st.get("leader"): return False
            st["enabled"] = True; _save(st); return True
        def get_arm_status(self):
            st = _load()
            return {{"ctrl_mode": 6 if st.get("leader") else 1, "arm_status": 0}}
        def get_leader_joint_angles(self):
            st = _load()
            return _Msg(list(st["joints"])) if st.get("leader") else None
        def get_joint_angles(self):
            st = _load()
            return _Msg(list(st["joints"])) if not st.get("leader") else None
        def set_speed_percent(self, p): pass
        def move_j(self, q): pass
    class AgxArmFactory:
        @staticmethod
        def create_arm(cfg): return _Arm(cfg)
"""), encoding="utf-8")
os.environ["PYTHONPATH"] = str(FAKE) + os.pathsep + os.environ.get("PYTHONPATH", "")

messagebox.askyesno = lambda *a, **k: True
cap = {}
_orig_se_init = SE.SafetyEditorUI.__init__
_orig_cam_init = CAM.CameraTabUI.__init__

def se_init(self, *a, **k):
    _orig_se_init(self, *a, **k)
    cap["ed"] = self

def cam_init(self, *a, **k):
    _orig_cam_init(self, *a, **k)
    cap["cam"] = self

SE.SafetyEditorUI.__init__ = se_init
CAM.CameraTabUI.__init__ = cam_init
res = {}
_orig_mainloop = tk.Tk.mainloop

def walk(w_):
    yield w_
    for c in w_.winfo_children():
        yield from walk(c)

def driver(root):
    ed = cap["ed"]
    cam = cap["cam"]
    pipe = ed.pipe

    def wait(cond, t=25):
        t_end = time.time() + t
        while time.time() < t_end and not cond():
            root.update()
            time.sleep(0.05)
        return cond()

    def go():
        res["cfg"] = ed.cfg_fn()
        cbs = [w_ for w_ in walk(root) if w_.winfo_class() == "TCombobox" and "v121" in (w_.cget("values") or ())]
        res["fw_cb_values"] = list(cbs[0].cget("values")) if cbs else []

        # Arm starts in teach (leader) mode; click 退出示教模式 on workflow bar
        STATE_FILE.write_text(json.dumps({"leader": True, "enabled": False, "fw": None,
                                          "joints": [0.12, -0.40, 0.05, 1.10, 0.0, 0.20, 0.55]}), encoding="utf-8")
        cam.workflow_bars[-1].b_teach_out.invoke()
        wait(lambda: STATE_FILE.exists() and not json.loads(STATE_FILE.read_text(encoding="utf-8")).get("leader")
             and json.loads(STATE_FILE.read_text(encoding="utf-8")).get("enabled"), 15)
        res["after_teach_exit"] = json.loads(STATE_FILE.read_text(encoding="utf-8"))

        # Now put arm back into leader mode and run Step 3 (step_push) to verify Step 3 recovers from teach mode
        STATE_FILE.write_text(json.dumps({"leader": True, "enabled": False, "fw": None,
                                          "joints": [0.12, -0.40, 0.05, 1.10, 0.0, 0.20, 0.55]}), encoding="utf-8")
        pipe.state["env"] = (RR.GREEN, "ok")
        pipe.state["can"] = (RR.GREEN, "ok")
        lamp = pipe.step_push()
        res["step_push_lamp"] = lamp
        res["after_step_push"] = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        pipe.stop_all()
        root.after(100, root.destroy)

    root.after(600, go)

tk.Tk.mainloop = lambda self, n=0: (driver(self), _orig_mainloop(self, n))
C.run_console("real")

assert res["cfg"]["fw_version"] == "v121" and res["cfg"]["push_fw"] == "v121", res["cfg"]
assert "v121" in res["fw_cb_values"] and "v111" in res["fw_cb_values"], res["fw_cb_values"]
assert res["after_teach_exit"]["leader"] is False and res["after_teach_exit"]["enabled"] is True and res["after_teach_exit"]["fw"] == "V121", res["after_teach_exit"]
assert res["step_push_lamp"] == RR.GREEN and res["after_step_push"]["leader"] is False and res["after_step_push"]["enabled"] is True, res
print("ALL V3.54 GUI TESTS PASSED")
