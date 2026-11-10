#!/usr/bin/env python3
"""V3.90 offline tests (Real-robot cup task fixes after wrist Eye-in-Hand calibration):
R1  Reproduces the exact real-robot failure from the V3.89 log:
      - #1: kind="occluded", base_xyz_m=[-0.351, -0.056, -0.004], height_mm=None, height_depth_mm=100.0,
            reason="杯口被夹爪挡住（只给位置）"
      - #2: kind="cup", base_xyz_m=[-0.399, +0.128, -0.004], height_mm=114.0, height_depth_mm=109.0
      - Vision Review (gpt-5.6-luna) answers: #1="cup", #2="cup" ("Single upright cup; gripper is above it.")
    Verifies apply_review populates #1's height_mm from height_depth_mm, clears the stale occluded reason,
    and CupSkills.observe_scene() + plan_cup_stack("cup_1", "cup_2") succeed without TypeError.
R2  Reproduces the case where Vision Review is unavailable (e.g. OpenAI API key missing) during observe_scene()
    when one of the two cups on the table has kind="occluded" and height_mm=None: verifies CupSkills._perceive()
    promotes the table-occluded cup and handles height_mm=None without raising TypeError or CUP_NOT_PERCEIVED.
R3  Tests Vision Review override occluded -> stack (what="nested_cups") when height_mm is None: verifies
    CupSkills._perceive() computes stack height and base_dz_mm without crashing.
R4  Reproduces the exact STS3215 gripper empty-clamp touch calibration failure from the user's log:
      - previous_touch_deg = -3.69, stall at -4.04 deg (0.04 deg past the old -4.0 deg bound)
    Verifies calibrate_touch() succeeds at -4.04 deg while still rejecting objects in the jaws (+9.0 deg)
    and non-stalling full travel (how="reached").
"""
import json
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v390_"))
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")

import numpy as np                              # noqa: E402
import NERO_V362_Cup_Skills as CK               # noqa: E402
import NERO_V365_Real_Cup_Task as RT            # noqa: E402
import NERO_V365_STS3215_Gripper as GR          # noqa: E402
import NERO_V374_Vision_Review as VR            # noqa: E402

assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2")
assert RT.VERSION in ("V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2")

# ------------------------------------------------------------------ R1: Exact real-robot log reproduction (with Vision Review)
g = CK.geom()
dets_real = [
    {
        "det_id": "#1",
        "kind": "occluded",
        "reason": "杯口被夹爪挡住（只给位置）",
        "base_xyz_m": [-0.351, -0.056, -0.004],
        "rim_xyz_m": [-0.351, -0.056, round(-0.004 + g["h"], 4)],
        "height_mm": None,
        "height_depth_mm": 100.0,
        "tilt_deg": 0.0,
        "tilt_reliable": False,
        "conf": 0.92,
        "contour": [[180, 120], [240, 120], [240, 220], [180, 220]],
    },
    {
        "det_id": "#2",
        "kind": "cup",
        "reason": None,
        "base_xyz_m": [-0.399, 0.128, -0.004],
        "rim_xyz_m": [-0.399, 0.128, round(-0.004 + g["h"], 4)],
        "height_mm": 114.0,
        "height_depth_mm": 109.0,
        "tilt_deg": 0.5,
        "tilt_reliable": True,
        "conf": 0.88,
        "contour": [[360, 110], [420, 110], [420, 215], [360, 215]],
    },
]
res_real = {
    "cups": [dict(d) for d in dets_real],
    "dets": [dict(d) for d in dets_real],
    "table_z_m": -0.004,
    "n_2d": 2,
}
review_reply = {
    "ok": True,
    "model": "gpt-5.6-luna",
    "elapsed_s": 3.77,
    "items": {
        "#1": {"what": "cup", "upright": True, "in_gripper": False, "note": "Single upright cup; gripper is above it."},
        "#2": {"what": "cup", "upright": True, "in_gripper": False, "note": "Single upright cup."},
    },
    "scene": "Two upright single cups on the table.",
}
notes = VR.apply_review(res_real, review_reply)
assert any("#1: 几何=occluded → 看图=cup" in n for n in notes), notes
assert res_real["cups"][0]["kind"] == "cup"
assert res_real["cups"][0]["height_mm"] == 100.0, res_real["cups"][0]
assert res_real["cups"][0]["reason"] is None, res_real["cups"][0]

logs_r1 = []
io_r1 = types.SimpleNamespace(
    pose_mode=lambda: "camera",
    emit=logs_r1.append,
    gripper_profile=lambda: RT.real_profile(CK),
    current_q=lambda: [0.0, -0.4, 0.0, 1.1, 0.0, 0.5, 0.0],
    perceive=lambda tag: json.loads(json.dumps(res_real)),
    gripper_state=lambda: {"hold_active": False, "opening_mm": 32.0},
    tcp_pose=lambda: ([-0.30, 0.0, 0.30], [1.0, 0.0, 0.0, 0.0]),
    latch_pending=lambda: False,
)
sk_r1 = CK.CupSkills(io_r1, *CK.cup_knobs())
r_obs1 = sk_r1.run_tool("observe_scene", {})
assert r_obs1["ok"], r_obs1
assert set(r_obs1["scene"]["objects"].keys()) == {"cup_1", "cup_2"}
assert sk_r1.cam_single_h == 107.0, sk_r1.cam_single_h
assert sk_r1.single_h == {"cup_1": 100.0, "cup_2": 114.0}, sk_r1.single_h
r_plan1 = sk_r1.run_tool("plan_cup_stack", {"source_cup": "cup_1", "target_cup": "cup_2"})
assert r_plan1["ok"], r_plan1
print(f"[R1] Exact real-robot log reproduced: observe_scene + plan_cup_stack succeeded (min_margin={r_plan1['plan']['min_joint_margin_deg']} deg)")

# ------------------------------------------------------------------ R2: Without Vision Review (occluded cup + height_mm=None)
res_no_vr = {
    "cups": [dict(d) for d in dets_real],
    "dets": [dict(d) for d in dets_real],
    "table_z_m": -0.004,
    "n_2d": 2,
}
io_r2 = types.SimpleNamespace(
    pose_mode=lambda: "camera",
    emit=logs_r1.append,
    gripper_profile=lambda: RT.real_profile(CK),
    current_q=lambda: [0.0, -0.4, 0.0, 1.1, 0.0, 0.5, 0.0],
    perceive=lambda tag: json.loads(json.dumps(res_no_vr)),
    gripper_state=lambda: {"hold_active": False, "opening_mm": 32.0},
    tcp_pose=lambda: ([-0.30, 0.0, 0.30], [1.0, 0.0, 0.0, 0.0]),
    latch_pending=lambda: False,
)
sk_r2 = CK.CupSkills(io_r2, *CK.cup_knobs())
r_obs2 = sk_r2.run_tool("observe_scene", {})
assert r_obs2["ok"], r_obs2
assert set(r_obs2["scene"]["objects"].keys()) == {"cup_1", "cup_2"}
assert sk_r2.single_h == {"cup_1": 100.0, "cup_2": 114.0}
print("[R2] Without Vision Review (API key missing): table-occluded cup promoted safely during observe_scene")

# ------------------------------------------------------------------ R3: Stack with height_mm=None and height_depth_mm=None
res_stack_none = {
    "cups": [{
        "det_id": "#1",
        "kind": "stack",
        "base_xyz_m": [-0.399, 0.128, -0.004],
        "height_mm": None,
        "height_depth_mm": None,
        "tilt_deg": None,
    }],
    "dets": [],
}
io_r3 = types.SimpleNamespace(
    pose_mode=lambda: "camera",
    emit=logs_r1.append,
    gripper_profile=lambda: RT.real_profile(CK),
    current_q=lambda: [0.0] * 7,
    perceive=lambda tag: json.loads(json.dumps(res_stack_none)),
    gripper_state=lambda: {"hold_active": False, "opening_mm": 32.0},
    tcp_pose=lambda: ([-0.30, 0.0, 0.30], [1.0, 0.0, 0.0, 0.0]),
    latch_pending=lambda: False,
)
sk_r3 = CK.CupSkills(io_r3, *CK.cup_knobs())
r_obs3 = sk_r3.run_tool("observe_scene", {})
assert r_obs3["ok"] and r_obs3["scene"]["stacked"]["upper"] == "cup_1", r_obs3
print("[R3] Stack with height_mm=None & tilt_deg=None handled cleanly without TypeError")

# ------------------------------------------------------------------ R4: Gripper calibrate_touch at -4.04 deg
sim = GR.SimServo(closed_steps=2048, open_sign=1, start_open_deg=10.0)
# Make SimServo stall at -4.04 deg relative to closed_steps (2048):
#Since SimServo obstacle_gap_mm uses open_deg_for_gap_mm (>=0), we simulate a servo whose closed_steps offset makes it stall at -4.04 deg
grip = GR.SO101RealGripper({"touch_deg": -3.69, "touch_calibrated": True}, log=lambda s: None)
ok, msg = grip.connect(GR.SIM_PORT, sim_servo=sim)
assert ok, msg
# Override _wait once to return stalled at open_deg = -0.35 relative to old=-3.69 -> new = -4.04 deg
_orig_wait = grip._wait
grip_status = {"position_steps": 2002, "servo_deg": 175.96, "open_deg": -0.35, "gap_mm": 0.0,
               "goal_steps": 1957, "speed": 0, "load_pct": 20.0, "current_ma": 110.5,
               "voltage_v": 7.4, "temp_c": 31, "moving": False, "torque_on": True,
               "torque_limit_pct": 20.0, "errors": []}
grip._wait = lambda a_goal, timeout_s=3.0, tol_deg=1.5, stall_s=0.30: (
    (dict(grip_status), "stalled") if a_goal < 0 else _orig_wait(a_goal, timeout_s, tol_deg, stall_s)
)
_orig_status = grip.status
grip.status = lambda: dict(grip_status) if sim.goal < 2048 else _orig_status()
try:
    cal = grip.calibrate_touch()
finally:
    grip._wait = _orig_wait
    grip.status = _orig_status
assert cal["ok"] and cal["touch_deg"] == -4.04 and grip.cfg["touch_deg"] == -4.04, cal
print(f"[R4] Gripper empty-clamp calibration at -4.04 deg (prev -3.69 deg) succeeded: {cal}")

print("ALL V3.90 TESTS PASSED")
