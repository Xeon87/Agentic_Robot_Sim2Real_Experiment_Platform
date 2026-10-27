#!/usr/bin/env python3
"""V3.40: no false alarms from perception noise, a way out when cuRobo refuses the start state.

V1 table plane: calibrated until >= 2 well-measured eggs agree within 5 mm, then LOCKED (V3.39 log: -6 -> +9 -> +12 mm
   drift, every egg z jumped)
V2 bystander check with perceived poses: horizontal motion of a re-seen neighbour minus 2 x its ray rms; z ignored;
   unseen neighbours not judged; truth-backed poses unchanged (3D)
V3 pre-place view: refines the fused carton only when it agrees (<= 6 mm), else ignored (V3.39 log: 13 mm off ->
   false SLOT_MOVED_SINCE_PLAN); a tracking mismatch stops the look poses
V4 cuRobo failure: start-state diagnosis + MoveIt joint goal to the IK-screened solution / BioIK pose fallback in
   reorientation, recenter and return_to_pick_location
"""
import math, os, sys, textwrap, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import NERO_V326_Wrist_Camera as W       # noqa: E402
import hashlib                           # noqa: E402

assert W.VERSION in ("V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1") and (ROOT / "VERSION").read_text().strip() in ("V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
assert hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest().startswith("020cc6680f181872")
B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()


def block(start, end, dedent=True):
    i = B.index(start); s = B[i:B.index(end, i)]
    return textwrap.dedent(s) if dedent else s


# ------------------------------------------------------------------ V1
def meas_with(zs):
    M = W.EggMeasure()
    for k, z in enumerate(zs):
        M.est[f"e{k}"] = {"ok": True, "xyz_m": [0, 0, z + 0.022], "width_m": 0.044, "max_angle_deg": 25.0, "rms_m": 0.002}
    return M, {f"e{k}": "table_or_pile" for k in range(len(zs))}


M, loc = meas_with([-0.0057, -0.0060])                     # 6 mm below the calibrated plane: not used
z, src = M.plane(loc)
assert z == W.table_z_m() and "not used" in src and getattr(M, "table_lock", None) is None
M, loc = meas_with([0.003])                                # one egg only: not enough
assert M.plane(loc)[0] == W.table_z_m()
M, loc = meas_with([0.003, 0.0034])
z, src = M.plane(loc)
assert abs(z - 0.0034) < 1e-9 and "locked" in src
for k in M.est:                                            # later estimates drift +12 mm: the plane stays
    M.est[k]["xyz_m"][2] += 0.012
assert abs(M.plane(loc)[0] - 0.0034) < 1e-9
WM = (ROOT / "NERO_V326_Wrist_Camera.py").read_text()
for mk in ('zt_meas, zt_src340 = meas[pf].plane(', 'table_z336, tz_src339 = meas.plane(', '"table_plane_source"'):
    assert mk in WM, mk
print("[V1] table plane: 6 mm off / one egg -> calibrated; 2 eggs within 5 mm -> locked; later +12 mm drift ignored; "
      "the same plane for eggs, carton detection and carton height")

# ------------------------------------------------------------------ V2
src2 = block("def _v340_perceived_poses(skill_core):", "# ============================================================\n# V3.08")
g = {"os": os, "math": math, "V271_BYSTANDER_MOVE_TOL_M": 0.003, "V311_STATE": {"mode": "wrist_rgb", "last": {}},
     "V271_BYSTANDER_BASELINE": {}}
exec(src2, g)
poses = {}


def _v329_object_pose(name):
    return poses[name], [0, 0, 0, 1]


def truth_pose(name):
    return poses[name], [0, 0, 0, 1]


sc = types.SimpleNamespace(object_pose=_v329_object_pose)
g["V271_BYSTANDER_BASELINE"]["egg_1"] = {"egg_2": [-0.431, -0.173, 0.0146], "egg_3": [-0.384, -0.136, 0.0168],
                                         "egg_4": [-0.398, -0.080, 0.0146]}
poses.update({"egg_2": [-0.4284, -0.1728, 0.0297], "egg_3": [-0.3831, -0.1400, 0.0352], "egg_4": [-0.398, -0.068, 0.0146]})
g["V311_STATE"]["last"] = {"objects": {
    "egg_2": {"seen": True, "measurement": {"ray_rms_mm": 0.6}},
    "egg_3": {"seen": True, "measurement": {"ray_rms_mm": 1.1}},
    "egg_4": {"seen": False}}}
mv = g["_v271_bystander_moves"](sc, "egg_1")
assert all(m["motion_mm"] < 8.0 for m in mv), mv                       # V3.39 log case: 10.7 / 11 mm 3D -> no abort
assert "egg_4" not in [m["object_name"] for m in mv]                  # not re-seen: not judged
poses["egg_2"] = [-0.419, -0.173, 0.0146]                              # really pushed 12 mm sideways
mv = g["_v271_bystander_moves"](sc, "egg_1")
e2 = [m for m in mv if m["object_name"] == "egg_2"][0]
assert e2["motion_mm"] > 8.0 and abs(e2["raw_xy_motion_mm"] - 12.0) < 0.1 and e2["perception_noise_allowance_mm"] == 1.2
sc_t = types.SimpleNamespace(object_pose=truth_pose)                   # truth-backed poses: unchanged 3D check
poses["egg_2"] = [-0.431, -0.173, 0.0296]
mv = g["_v271_bystander_moves"](sc_t, "egg_1")
assert [m for m in mv if m["object_name"] == "egg_2"][0]["motion_mm"] == 15.0
print("[V2] bystander (perceived): V3.39 case (xy 3-4 mm, z +15 mm) -> no abort; real 12 mm push -> abort; unseen "
      "neighbour not judged; truth-backed check unchanged")

# ------------------------------------------------------------------ V3
logs = []
fused = {"complete": True, "center_xy_m": [-0.42341, 0.15378], "yaw_rad": 0.01,
         "slots": {"slot_1": {"xyz_m": [-0.513, 0.121, 0.039]}}}
scene = {"holder": dict(fused)}
filt = []
views = []


def make_env(view_holder, fail_exec=False):
    env = {"os": os, "math": math, "_V329_SCENE": scene, "skill_core": None, "V311_STATE": {}}
    moves = []

    def move_look(pt, label, distances=None, no_detour=False):
        moves.append(distances)
        if fail_exec:
            raise RuntimeError("PHYSICAL TRACKING MISMATCH: ros2_control reported success but MuJoCo is still 101.0deg")
    env["_v326_move_look"] = move_look
    env["_v335_capture"] = lambda tag: {"containers": {"egg_holder": view_holder}, "cam_pos": [0, 0, 0.4]}
    env["_v335_holder_filter"] = lambda h, tag=None, cam=None: (filt.append(h), dict(fused, center_xy_m=[-0.4232, 0.1537]))[1]
    env["_v329_apply"] = lambda: None
    env["_v329_truth_layout"] = lambda: {"slot_centers": {}}
    env["_v218_slow_trajectory_in_place"] = lambda t, factor=1.0: t
    env["_v310_wait_tcp_settled"] = lambda *a: None
    env["skill_core"] = types.SimpleNamespace(
        emit=logs.append, _current_joint_state_copy=lambda: None, _arm_positions_from_joint_state=lambda j: [0.0] * 7,
        _plan_joint_solution=lambda *a, **k: None, _execute_trajectory=lambda t: None)
    exec(block("    def _v329_preplace_look(obj, slot_name):", "    def _v339_rgb_only():"), env)
    return env, moves


off13 = dict(fused, center_xy_m=[-0.41012, 0.14663])
env, _m = make_env(off13)
env["_v329_preplace_look"]("egg_2", "slot_1")
assert scene["holder"]["center_xy_m"] == fused["center_xy_m"] and not filt
assert any("[V3.40 PRE-PLACE VIEW IGNORED]" in l for l in logs)
near = dict(fused, center_xy_m=[-0.4214, 0.1525])
env, _m = make_env(near)
env["_v329_preplace_look"]("egg_2", "slot_1")
assert len(filt) == 1 and scene["holder"]["center_xy_m"] == [-0.4232, 0.1537]
assert any("[V3.40 PRE-PLACE VIEW] egg_2->slot_1 | agrees" in l for l in logs)
logs.clear()
env, moves = make_env(near, fail_exec=True)
assert env["_v329_preplace_look"]("egg_2", "slot_1") is False
assert len(moves) == 1 and any("[V3.40 PRE-PLACE LOOK STOP]" in l for l in logs)        # V3.39: 6 failed executions
print("[V3] pre-place view 13 mm off the fused carton -> ignored (no false SLOT_MOVED); 1.6 mm -> through the carton "
      "filter; a tracking mismatch stops the look poses after the first")

# ------------------------------------------------------------------ V4
logs4 = []
calls = []


class SC:
    emit = logs4.append

    def _plan_pose(self, xyz, quat, **k):
        raise RuntimeError("cuRobo exhausted 10 trajectory attempts; none passed MoveIt full-trajectory validation.")

    def _current_joint_state_copy(self):
        return "live"

    def _arm_positions_from_joint_state(self, live):
        return [0.0, 0.5, 0.0, 1.0, 0.0, 0.9, 1.5]

    def _v71_state_is_valid(self, q):
        return False, "collision:egg_2<->gripper_flange"

    def _compute_ik(self, xyz, quat, **k):
        calls.append("ik")
        return [0.1] * 7, None

    def _plan_joint_solution(self, j, **k):
        calls.append(("joint", tuple(j)))
        return "JOINT_TRAJ"

    def _plan_pose_bioik_v70(self, *a, **k):
        calls.append("bioik")
        return "BIOIK_TRAJ"


g4 = {"os": os, "math": math, "time": __import__("time"), "skill_core": SC(),
      "JOINT_LIMITS_RAD": [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75),
                           (-0.73, 0.95), (-1.5708, 1.5708)]}
exec(block("    _V340_DIAG = {\"t\": 0.0}", "    def _v339_rgb_only():"), g4)
assert g4["_v340_pose_with_fallback"]("return egg_2: hover", [-0.43, -0.17, 0.30], [1, 0, 0, 0]) == "JOINT_TRAJ"
d4 = g4["_V340_DIAG"]["last"]
assert d4["joint"] == "J6" and d4["moveit_state_valid"] is False and "collision" in d4["moveit_state"]
assert any("[V3.40 CUROBO DIAG]" in l for l in logs4) and any("[V3.40 MOVEIT FALLBACK]" in l for l in logs4)
g4["skill_core"]._compute_ik = lambda *a, **k: (None, "no ik")
assert g4["_v340_pose_with_fallback"]("x", [0, 0, 0.3], [1, 0, 0, 0]) == "BIOIK_TRAJ"
assert g4["_v340_joint_fallback"]("x", RuntimeError("some other error"), [0] * 7) is None    # only cuRobo failures
for mk in ('traj302 = _v340_joint_fallback(f"{label302} {tag302}", exc302, _j302, 0.16, 0.12)',
           'traj = _v340_joint_fallback(f"v337_recenter({obj})", exc337, j340, 0.12, 0.10)',
           'traj340 = _v340_pose_with_fallback(f"return {obj}: hover", hover, o_q, 0.12, 0.12, 0.010)',
           'except Exception as exc340:'):
    assert mk in B, mk
print(f"[V4] cuRobo refuses -> start-state diagnosis ({d4['joint']} margin {d4['min_limit_margin_deg']} deg, MoveIt: "
      f"{d4['moveit_state']}) + MoveIt joint goal to the screened IK / BioIK pose plan; used by the reorientation "
      "yaw family, recenter and return_to_pick_location")
print("ALL V3.40 TESTS PASSED")
