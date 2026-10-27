#!/usr/bin/env python3
"""V3.46: RViz no longer killed by pip opencv-python's Qt plugin path; an RViz exit is a warning, not a driver failure.

E1 clean_child_env (console + real-robot module): cv2/qt entries removed from the Qt variables, other entries kept
E2 children really get the cleaned environment: ProcRunner (driver / MoveIt / helpers), run(), console SimRunner
E3 the camera module imports cv2 through a guard that restores the Qt variables (fake cv2 that sets them on import);
   importing the module / building the page does not import cv2 (ArUco checked only after a camera connects)
E4 'process has died' of rviz2 -> not a driver failure; MoveIt step AMBER (run_all continues); other deaths stay RED
"""
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V313_Real_Robot as RR      # noqa: E402
import NERO_V312_Console as C          # noqa: E402

assert RR.VERSION in ("V3.46", "V3.47", "V3.47.1") and C.VERSION == RR.VERSION and (ROOT / "VERSION").read_text().strip() == RR.VERSION
CV = "/home/u/.local/lib/python3.10/site-packages/cv2/qt/plugins"
FD = "/home/u/.local/lib/python3.10/site-packages/cv2/qt/fonts"

# ------------------------------------------------------------------ E1
for fn in (RR.clean_child_env, C.clean_child_env):
    e = fn({"QT_QPA_PLATFORM_PLUGIN_PATH": CV, "QT_QPA_FONTDIR": FD, "QT_PLUGIN_PATH": f"/opt/qt/plugins:{CV}",
            "PATH": "/usr/bin", "ROS_DOMAIN_ID": "42"})
    assert "QT_QPA_PLATFORM_PLUGIN_PATH" not in e and "QT_QPA_FONTDIR" not in e, e
    assert e["QT_PLUGIN_PATH"] == "/opt/qt/plugins" and e["PATH"] == "/usr/bin" and e["ROS_DOMAIN_ID"] == "42"
    e2 = fn({"QT_QPA_PLATFORM_PLUGIN_PATH": "/opt/ros/humble/qt"})
    assert e2["QT_QPA_PLATFORM_PLUGIN_PATH"] == "/opt/ros/humble/qt"          # a non-cv2 setting is kept
print("[E1] cv2/qt entries removed from QT_QPA_PLATFORM_PLUGIN_PATH / QT_PLUGIN_PATH / QT_QPA_FONTDIR; others kept")

# ------------------------------------------------------------------ E2
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = CV
try:
    pr = RR.ProcRunner("t")
    pr.start(["/bin/bash", "-c", 'echo "QT=${QT_QPA_PLATFORM_PLUGIN_PATH:-unset}"'])
    got = []
    while True:
        ln = pr.lines.get(timeout=10)
        if ln is None:
            break
        got.append(ln)
    assert got == ["QT=unset"], got
    pr2 = RR.ProcRunner("t2")
    pr2.start(["/bin/bash", "-c", 'echo "QT=${QT_QPA_PLATFORM_PLUGIN_PATH:-unset} D=$ROS_DOMAIN_ID"'],
              env=dict(os.environ, ROS_DOMAIN_ID="42"))
    got2 = []
    while True:
        ln = pr2.lines.get(timeout=10)
        if ln is None:
            break
        got2.append(ln)
    assert got2 == ["QT=unset D=42"], got2
    rc, out, _err = RR.run('echo "${QT_QPA_PLATFORM_PLUGIN_PATH:-unset}"')
    assert rc == 0 and out.strip() == "unset", out
    with tempfile.TemporaryDirectory() as td:
        fake = Path(td) / "fake_sim.sh"
        fake.write_text('#!/usr/bin/env bash\necho "QT=${QT_QPA_PLATFORM_PLUGIN_PATH:-unset}"\n')
        sr = C.SimRunner(str(fake))
        sr.start(extra_env=C.scene_env("fill_egg_holder"))
        got3 = []
        while True:
            ln = sr.lines.get(timeout=10)
            if ln is None:
                break
            got3.append(ln)
        assert got3 == ["QT=unset"], got3
finally:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
print("[E2] with cv2's path in the console environment: ProcRunner (driver / MoveIt / helpers), run() and the sim "
      "launcher all start without it")

# ------------------------------------------------------------------ E3
with tempfile.TemporaryDirectory() as td:
    pkg = Path(td) / "cv2"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(textwrap.dedent(f"""
        import os
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = {CV!r}
        os.environ["QT_QPA_FONTDIR"] = {FD!r}
        __version__ = "fake"
        class aruco: pass
    """))
    code = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {td!r}); sys.path.insert(1, {str(ROOT)!r})
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None); os.environ.pop("QT_QPA_FONTDIR", None)
        import NERO_V325_Camera as CAM
        assert "cv2" not in sys.modules, "module import pulled in cv2"
        ok, msg = CAM.aruco_status()
        assert ok and "fake" in msg and "cv2" in sys.modules
        assert "QT_QPA_PLATFORM_PLUGIN_PATH" not in os.environ and "QT_QPA_FONTDIR" not in os.environ, dict(os.environ)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/opt/keep"
        del sys.modules["cv2"]
        CAM._import_cv2()
        assert os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] == "/opt/keep"
        print("ok")
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), (r.stdout, r.stderr)
S = (ROOT / "NERO_V325_Camera.py").read_text()
assert "if aru_ok is None and cam:" in S and S.count("\n    import cv2\n") == 1      # only inside _import_cv2
assert "_import_cv2()                        # V3.46" in S
print("[E3] cv2 imported through the guard (Qt variables restored, an earlier user value kept); importing the "
      "camera module / building the page does not import cv2")

# ------------------------------------------------------------------ E4
rviz = ("[ERROR] [rviz2-4]: process has died [pid 8302, exit code -6, cmd '/opt/ros/humble/lib/rviz2/rviz2 -d "
        "/tmp/moveit.rviz --ros-args']")
mg = "[ERROR] [move_group-3]: process has died [pid 8300, exit code -11, cmd '/opt/ros/humble/lib/moveit_ros_move_group/move_group']"
assert RR.is_rviz_death(rviz) and RR.driver_log_failure(rviz) is None
assert not RR.is_rviz_death(mg) and RR.driver_log_failure(mg) == "process has died"
mon = {"has_move_group": True, "has_move_action": True}
lamp, msg = RR.evaluate_moveit(mon, False, True, "exit -6")
assert lamp == RR.AMBER and "RViz exited" in msg, (lamp, msg)
assert RR.evaluate_moveit(mon, False, True)[0] == RR.YELLOW and RR.evaluate_moveit(mon, True, True)[0] == RR.GREEN
assert RR.evaluate_moveit(mon, False, False)[0] == RR.GREEN
R = (ROOT / "NERO_V313_Real_Robot.py").read_text()
assert "return (lamp in (GREEN, AMBER) or not self.driver.running()) and True" in R
assert "if r not in (GREEN, AMBER):" in R and "self.rviz_died = None" in R
print("[E4] rviz2 'process has died' -> warning only, MoveIt step AMBER and 一键启动 continues; move_group death "
      "is still a driver failure")
print("ALL V3.46 TESTS PASSED")
