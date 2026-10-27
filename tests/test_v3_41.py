#!/usr/bin/env python3
"""V3.41: joint-limit margin, honest insertion block, credible carton views only, real release pose, lost egg re-found.

M1 joint-limit margin: every slot-above / corridor level / reorientation IK >= 8 deg from all 7 limits (4 deg
   fallback), larger margin preferred (V3.40 log: slot entry with J6 0.6 deg from its limit)
M2 insertion: CARTESIAN APPROACH BLOCKED -> BioIK 5 mm step + MoveIt joint goal; still blocked -> back to the entry,
   one re-plan with another yaw (INSERT_BLOCKED_REPLAN_YAW), then INSERT_BLOCKED_JOINT_LIMIT with the margins
M3 carton suspicion: live / precheck views closer than 0.28 m, incomplete or border-cut, or during a placement never
   raise "possibly moved"
M4 the held egg pose is refreshed after every arm motion -> the release position is the real one
M5 a lost egg that is found again is not forgotten again by queued "lost" captures; USB-only predicted association
"""
import math, os, sys, textwrap, time, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import NERO_V326_Wrist_Camera as W       # noqa: E402
import hashlib                           # noqa: E402

assert W.VERSION in ("V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1") and (ROOT / "VERSION").read_text().strip() in ("V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
assert hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest().startswith("020cc6680f181872")
B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
A = (ROOT / "NERO_V2_7_1_Agent.py").read_text()
WM = (ROOT / "NERO_V326_Wrist_Camera.py").read_text()
LIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708)]


def block(start, end, dedent=True):
    i = B.index(start); s = B[i:B.index(end, i + len(start))]
    return textwrap.dedent(s) if dedent else s


G = {"math": math, "os": os, "JOINT_LIMITS_RAD": LIM}
exec(block("def v341_limit_margins_deg(q):", "# V2 reference speeds"), G)
marg = G["v341_limit_margins_deg"]

# ------------------------------------------------------------------ M1
q340 = [0.2, 0.6, -0.1, 1.2, 0.3, 0.95 - math.radians(0.6), 0.4]       # V3.40 log: J6 0.6 deg from its limit
m, j, _l = marg(q340)
assert j == 6 and abs(m - 0.6) < 0.01 and m < G["v341_margin_required_deg"]() == 8.0
assert G["v341_margin_relaxed_deg"]() == 4.0 and marg([0.0] * 7)[0] > 40
for mk in ('[V3.41 LIMIT MARGIN] {label302}', "ranked302.sort(key=lambda r: (-int(min(marg341[id(r)], 20.0) // 4), r[0]))",
           'V341_STATE["corr_margin"] = mg341', 'from its limit at obj_z=', '[V3.41 ENDPOINT MARGIN]',
           'NERO_V341_CORRIDOR_DEPTH_M', 'V341_STATE["corr_margin"] = v341_margin_relaxed_deg()'):
    assert mk in B, mk
print("[M1] margin helper: V3.40 slot-entry pose -> J6 0.6 deg (< 8); IK screen + every corridor level (now 30 mm "
      "below the entry) need >= 8 deg (4 deg fallback), larger margin first; cuRobo end near a limit -> screened IK "
      "(orientation harness A2 covers the full selection)")

# ------------------------------------------------------------------ M2
logs = []


class SC:
    emit = logs.append
    ik = [0.21, 0.6, -0.1, 1.25, 0.3, 0.5, 0.4]

    def _current_joint_state_copy(self):
        return "live"

    cur = [0.2, 0.6, -0.1, 1.2, 0.3, 0.48, 0.4]

    def _arm_positions_from_joint_state(self, live):
        return list(self.cur)

    def _compute_ik(self, xyz, quat, **k):
        assert k["avoid_collisions"] and k["seed_joint_state"] == "live"
        return (list(self.ik) if self.ik else None), None

    def _plan_joint_solution(self, j, **k):
        return ("JOINT", tuple(j))


g2 = dict(G, skill_core=SC())
exec(block("    def _v341_insert_ik_step(target_tcp, quat):", "    def _v339_rgb_only():"), g2)
step = g2["_v341_insert_ik_step"]
tr, why = step([0, 0, 0.2], [1, 0, 0, 0])
assert tr[0] == "JOINT" and "joint step" in why
SC.ik = [0.2, 0.6, -0.1, 1.2, 0.3 + math.radians(40), 0.48, 0.4]
assert step([0, 0, 0.2], [1, 0, 0, 0])[0] is None and "another branch" in step([0, 0, 0.2], [1, 0, 0, 0])[1]
SC.cur = [0.2, 0.6, -0.1, 1.2, 0.3, 0.93, 0.4]                       # V3.40 case: J6 already at its limit
SC.ik = [0.2, 0.6, -0.1, 1.2, 0.3, 0.95 - math.radians(0.2), 0.4]
assert "at its limit" in step([0, 0, 0.2], [1, 0, 0, 0])[1]
SC.ik = None
assert step([0, 0, 0.2], [1, 0, 0, 0]) == (None, "no collision-free IK for this level")
for mk in ('"CARTESIAN APPROACH BLOCKED" not in str(exc341)', 'traj, why341 = _v341_insert_ik_step(target_tcp, tcp_q_now)',
           'abort_reason = "INSERT_BLOCKED"', 'if z_before341 - float(_ob341[2]) < 0.4 * dz:',
           '"INSERT_BLOCKED_REPLAN_YAW", ["plan_placement", "orient_for_slot"]',
           '"INSERT_BLOCKED_JOINT_LIMIT", ["return_to_pick_location"]', '"min_joint_limit_margin_deg"',
           'v341_insert_block_retreat', 'V341_STATE["skip_yaw"][str(obj)] = str(slot_name)',
           'and V341_STATE["skip_yaw"].get(str(obj)) == str(slot_name)):'):
    assert mk in B, mk
assert "V3.41 INSERT BLOCKED" in A and "INSERT_BLOCKED_JOINT_LIMIT: call return_to_pick_location" in A
print("[M2] blocked straight descent -> collision-aware BioIK step on the same branch (12 deg max, off the limits) + "
      "MoveIt joint goal; another branch / at a limit / no IK -> INSERT_BLOCKED (no more silent 0 mm steps); retreat "
      "to the entry, one yaw re-plan (current yaw skipped), then INSERT_BLOCKED_JOINT_LIMIT with the margins")

# ------------------------------------------------------------------ M3
logs3 = []
ST = {}
g3 = {"os": os, "math": math, "time": time, "V335_STATE": ST, "skill_core": types.SimpleNamespace(emit=logs3.append)}
exec(block("    def _v335_holder_filter(h, tag=None, cam_pos=None):", "\n    def "), g3)
filt = g3["_v335_holder_filter"]


def hold(cx, cy, complete=True, border=False):
    return {"center_xy_m": [cx, cy], "yaw_rad": 0.0, "complete": complete, "touches_border": border,
            "top_z_source": "measured", "top_z_m": 0.045, "slots": {"slot_1": {"xyz_m": [cx - 0.09, cy - 0.03, 0.039]}}}


def reset():
    ST.clear()
    filt(hold(-0.42, 0.155), "observe_scene/scan_1", [-0.42, 0.0, 0.45])


reset()
filt(hold(-0.31, 0.16), "live:insert", [-0.50, 0.12, 0.17])            # V3.40 log: camera ~0.13 m from the carton
assert not ST.get("holder_suspect") and not ST.get("holder_jump") and any("V3.41 HOLDER VIEW IGNORED" in l for l in logs3)
reset()
filt(hold(-0.31, 0.16, complete=False), "precheck:x", [-0.42, 0.0, 0.50])
assert not ST.get("holder_suspect") and not ST.get("holder_jump")
reset()
ST["frozen_place"] = {"obj": "egg_1", "slot": "slot_1", "t": time.time()}
filt(hold(-0.38, 0.155), "live:orient", [-0.42, 0.0, 0.50])
assert not ST.get("holder_jump")
reset()
filt(hold(-0.38, 0.155), "live:observe", [-0.42, 0.0, 0.50])            # credible far view, 40 mm off: still judged
assert ST.get("holder_jump") or ST.get("holder_suspect")
print("[M3] carton 'possibly moved' from live/precheck views: camera 0.13 m away / carton incomplete / placement "
      "in progress -> ignored; a complete view from >= 0.28 m still counts")

# ------------------------------------------------------------------ M4
EST = {"released": {}, "held_last": None, "log_t": {}, "truth_fn": None}
tcp = {"z": 0.20}
calls4 = []


def obj_pose(name):
    EST["held_last"] = (name, ([-0.43, -0.17, tcp["z"] - 0.1], [0, 0, 0, 1]))
    calls4.append(name)
    return EST["held_last"][1]


core4 = types.SimpleNamespace(emit=lambda *a: None, object_pose=lambda n: None,
                              _execute_trajectory=lambda t: tcp.update(z=float(t)))
g4 = {"os": os, "_V329_EST": EST, "_v329_no_prior": lambda: True, "hold_active_shared": types.SimpleNamespace(value=True),
      "v24_skill_state": {"active_object": "egg_2"}, "_v329_object_pose": obj_pose, "skill_core": core4}
exec(block("    def _v329_install_no_truth():", "    def _v329_near_zone("), g4)
g4["_v329_install_no_truth"](); g4["_v329_install_no_truth"]()
core4._execute_trajectory(0.05)                                          # e.g. return_to_pick descent into the basket
assert EST["held_last"][1][0][2] == 0.05 - 0.1 and calls4 == ["egg_2"]   # wrapped once, refreshed after the motion
g4["hold_active_shared"].value = False
core4._execute_trajectory(0.30)
assert calls4 == ["egg_2"]
assert "skill_core.object_pose(obj)          # V3.41: held pose at the release point" in B
print("[M4] held egg pose refreshed after every executed trajectory while holding (and right before the "
      "return_to_pick release) -> released where it really was, no false RETURN FAILED / LOST")

# ------------------------------------------------------------------ M5
class Base:
    def __init__(self):
        self.tracks = {}

    def update(self, dets, seq, seed_match=None):
        return {}, {"unassigned_detections": [d["det_id"] for d in dets]}


K = {"fx": 300.0, "fy": 300.0, "ppx": 320.0, "ppy": 240.0, "width": 640, "height": 480}
T = [[1, 0, 0, -0.42], [0, -1, 0, -0.15], [0, 0, -1, 0.40], [0, 0, 0, 1]]
os.environ["NERO_V333_CONFIRM_VIEWS"] = "1"
tr = W.ViewTracker(Base(), names=["egg_1", "egg_2"])
tr.base.tracks["egg_1"] = {"first_seq": 1, "xyz_m": [-0.40, -0.15, 0.02], "quat_xyzw": [0, 0, 0, 1], "location": "x"}
hints = {"lost": ["egg_1"]}
o, d = tr.update([], 5, K, T, hints=hints)
assert d["lost_forgotten"] == ["egg_1"] and "egg_1" not in tr.base.tracks
det = [{"det_id": 0, "xyz_m": [-0.45, -0.16, 0.02], "quat_xyzw": [0, 0, 0, 1], "axis_world": [1, 0, 0],
        "location": "table", "trusted": True, "quality": {}}]
o, d = tr.update(det, 6, K, T, hints=hints)                              # found again (capture still carries the hint)
assert d["named"] == ["egg_1"]
o, d = tr.update([], 7, K, T, hints=hints)                               # a queued capture with the old hint
assert "egg_1" in tr.base.tracks and d["lost_forgotten"] == []
o, d = tr.update([], 8, K, T, hints={})
o, d = tr.update([], 9, K, T, hints={"lost": ["egg_1"]})                 # a NEW loss later: forgotten again
assert d["lost_forgotten"] == ["egg_1"]
assert "if predict is not None and os.environ.get(\"NERO_V339_PREDICT_ASSOC\"" in WM
print("[M5] lost egg forgotten once; re-found egg_1 stays named although later captures still carry the lost hint; "
      "a new loss forgets it again; V3.39 predicted association only with the RGB-only (USB) camera")
print("ALL V3.41 TESTS PASSED")
