#!/usr/bin/env python3
"""V3.27 wrist-camera look moves (fix for V3.26: the arm never left q0, cuRobo returned None for the single look pose).

L1 look-pose family geometry: every candidate's CAMERA looks exactly at the point from the requested distance,
   TCP above the floor limit, quaternions unit, top-down views first, no duplicates
L2 (numpy) URDF kinematics: for basket and holder centre most of the first 8 views have an IK solution with
   >= 8 deg joint margin and no cuRobo self-collision (the V3.26 single pose is reported for comparison)
L3 backend _v326_move_look with a fake skill core: BioIK screen -> ranked solutions -> MoveIt joint goal;
   joint-plan failure falls through to cuRobo pose plan, then to the next solution; no IK at all ->
   WRIST_LOOK_UNREACHABLE; standing at a look pose skips the q0 detour; marker lines
L4 fixed-camera module unchanged (hash as in V3.11..V3.26)
"""
import hashlib, math, os, re, sys, textwrap, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V326_Wrist_Camera as W  # noqa: E402

assert W.VERSION in ("V3.27", "V3.28", "V3.29", "V3.30", "V3.31", "V3.31.1", "V3.33", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
BASKET = [-0.425, -0.155, 0.043]
HOLDER = [-0.420, 0.155, 0.043]

# ------------------------------------------------------------------ L1
pc, fc = W.camera_in_tcp()
assert 0.03 < math.hypot(pc[0], pc[1]) < 0.12 and -0.12 < pc[2] < -0.03 and fc[2] > 0.97   # behind + beside the TCP (V3.31.1: on the gripper side face)
for P in (BASKET, HOLDER):
    C = W.look_pose_candidates(P)
    assert len(C) >= 60 and C[0]["tilt_deg"] == 0 and C[0]["dist_m"] == 0.36
    keys = set()
    for c in C:
        R = W.matrix_from_quat(c["quat_xyzw"])
        assert abs(sum(v * v for v in c["quat_xyzw"]) - 1) < 1e-6
        cam = [c["xyz"][i] + sum(R[i][k] * pc[k] for k in range(3)) for i in range(3)]
        f = [sum(R[i][k] * fc[k] for k in range(3)) for i in range(3)]
        v = [P[i] - cam[i] for i in range(3)]
        n = math.sqrt(sum(x * x for x in v))
        assert abs(n - c["dist_m"]) < 2e-3 and sum(v[i] / n * f[i] for i in range(3)) > 0.9999, c["tag"]
        assert f[2] < -0.70                                                         # looking down (<= 45 deg)
        keys.add((c["dist_m"], tuple(c["quat_xyzw"])))
    assert len(keys) == len(C)
    assert all(W.look_pose_ok(c) for c in C[:40])
    top = [c for c in C if c["tilt_deg"] == 0 and c["dist_m"] == 0.36]
    assert len(top) == 8 and all(abs(c["xyz"][2] - top[0]["xyz"][2]) < 1e-6 for c in top)
print(f"[L1] look family: {len(C)} views per point (distance x tilt x roll); camera on the point at the set distance; "
      f"top-down TCP z {top[0]['xyz'][2]:.3f} m")

# ------------------------------------------------------------------ L2
try:
    import numpy as np
    import xml.etree.ElementTree as ET
    have_np = True
except Exception:
    have_np = False
URDF = ROOT / "config" / "nero_7dof_with_tcp.urdf"
if not have_np or not URDF.is_file():
    print("[L2] numpy or URDF missing -> kinematic check skipped")
else:
    def rpy(a):
        r_, p, y = a
        Rx = np.array([[1, 0, 0], [0, math.cos(r_), -math.sin(r_)], [0, math.sin(r_), math.cos(r_)]])
        Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
        Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx
    jt = {}
    for j in ET.parse(URDF).getroot().iter("joint"):
        o = j.find("origin")
        T = np.eye(4)
        T[:3, :3] = rpy([float(v) for v in (o.get("rpy") or "0 0 0").split()])
        T[:3, 3] = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
        lim = j.find("limit")
        jt[j.find("child").get("link")] = (j.get("type"), j.find("parent").get("link"), T,
                                           (float(lim.get("lower")), float(lim.get("upper"))) if lim is not None else None)
    chain, l = [], "tcp_link"
    while l in jt:
        chain.append(jt[l]); l = jt[l][1]
    chain = chain[::-1]
    LIM = np.array([c[3] for c in chain if c[0] == "revolute"])

    def fk(q):
        T, i = np.eye(4), 0
        for typ, _p, To, _l in chain:
            T = T @ To
            if typ == "revolute":
                c, s = math.cos(q[i]), math.sin(q[i]); Rq = np.eye(4); Rq[:2, :2] = [[c, -s], [s, c]]; T = T @ Rq; i += 1
        return T

    def err(T, p, R):
        Re = R @ T[:3, :3].T
        return np.r_[p - T[:3, 3], 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])]

    def solvable(p, R, margin_deg=8.0, tries=6):
        lo, hi = LIM[:, 0] + math.radians(margin_deg), LIM[:, 1] - math.radians(margin_deg)
        rng = np.random.default_rng(1)
        for k in range(tries):
            q = rng.uniform(lo, hi) if k else np.radians([0, 30, 0, 80, 0, 30, 0])
            for _ in range(150):
                T = fk(q); e = err(T, p, R)
                if np.linalg.norm(e) < 1e-4:
                    break
                J = np.zeros((6, 7))
                for jj in range(7):
                    qq = q.copy(); qq[jj] += 1e-6; J[:, jj] = (e - err(fk(qq), p, R)) / 1e-6
                q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e), lo, hi)
            e = err(fk(q), p, R)
            if np.linalg.norm(e[:3]) < 1e-3 and np.linalg.norm(e[3:]) < 5e-3:
                return True
        return False
    T0 = fk(np.zeros(7))
    assert abs(T0[2, 3] - 0.8733) < 1e-3                                               # q0 TCP height as in the log
    for name, P in (("basket", BASKET), ("holder", HOLDER)):
        C = W.look_pose_candidates(P)[:8]
        ok = sum(solvable(np.array(c["xyz"]), np.array(W.matrix_from_quat(c["quat_xyzw"]))) for c in C)
        assert ok >= 6, (name, ok)
        print(f"[L2] {name}: {ok}/8 first views reachable with >= 8 deg joint margin (URDF IK)")

# ------------------------------------------------------------------ L3
B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
a, b = B.index("    # V3.27 WRIST LOOK FAMILY"), B.index("    # V3.28 VIEW-AWARE TRACKING HINTS")
code = textwrap.dedent(B[a:b])


class Val:
    def __init__(self, v=0): self.value = v


def make_env(ik_ok, joint_fail=(), pose_fail=True, start=(0.0, -0.002, 0.873), near_basket=False):
    logs, calls = [], {"q0": 0, "ik": 0, "joint": [], "pose": [], "exec": 0}
    state = {"tcp": list(start), "target": None}
    sc = types.SimpleNamespace()
    sc.emit = logs.append
    sc._current_joint_state_copy = lambda: "LIVE"
    sc._arm_positions_from_joint_state = lambda js: [0.0] * 7

    def ik(xyz, q, timeout_s, avoid_collisions, seed_joint_state):
        calls["ik"] += 1
        assert avoid_collisions is True and seed_joint_state == "LIVE"
        return ([0.2, 0.5, 0.0, 1.2, 0.0, 0.3, 0.1], "SUCCESS") if ik_ok(xyz, q) else (None, "NO_IK_SOLUTION")
    sc._compute_ik = ik

    def jplan(joints, **kw):
        calls["joint"].append(len(calls["joint"]))
        if len(calls["joint"]) - 1 in joint_fail:
            raise RuntimeError("JOINT-GOAL PLANNING FAILED: PLANNING_FAILED")
        return {"goal": state["cand"]}
    sc._plan_joint_solution = jplan

    def pplan(xyz, q, *a_):
        calls["pose"].append(xyz)
        if pose_fail:
            raise RuntimeError("GOAL_IK_UNREACHABLE: cuRobo plan_pose returned None 3x")
        return {"goal": state["cand"]}
    sc._plan_pose = pplan

    def execute(traj):
        calls["exec"] += 1
        state["tcp"] = list(traj["goal"]["xyz"])
    sc._execute_trajectory = execute
    sc.current_tcp_pose = lambda: (list(state["tcp"]), [1, 0, 0, 0])
    env = {"skill_core": sc, "os": os, "math": math, "safety_trip_count": Val(0),
           "_v218_slow_trajectory_in_place": lambda t, factor=1.0: t,
           "_v310_wait_tcp_settled": lambda label: None,
           "_V239_LOW_Z_M": 0.255,
           "_v24_xyz_distance": lambda p, q: math.dist([float(v) for v in p], [float(v) for v in q]),
           "_v307_posture_numbers": lambda: ({"near_basket": near_basket, "near_holder": False, "xyz": state["tcp"]}, {}),
           "_v307_go_q0": lambda label: calls.__setitem__("q0", calls["q0"] + 1),
           "v330_wrist_profile": lambda: "d435i"}
    exec(code, env)
    # remember which candidate the joints belong to: wrap _plan_joint_solution via the ranked solution order
    real = env["_v327_ik_solutions"]

    def wrapped(cands, live, qnow, max_tries, want):
        sols, tried = real(cands, live, qnow, max_tries, want)
        it = iter(sols)

        def jp(joints, **kw):
            state["cand"] = next(it)[2]
            return jplan(joints, **kw)
        sc._plan_joint_solution = jp
        return sols, tried
    env["_v327_ik_solutions"] = wrapped
    return env, logs, calls, state


# (a) first 5 views have no IK, first joint plan fails and its cuRobo fallback fails -> second solution executes
cnt = {"n": 0}


def ik_after5(xyz, q):
    cnt["n"] += 1
    return cnt["n"] > 5
env, logs, calls, state = make_env(ik_after5, joint_fail=(0,), pose_fail=True)
c = env["_v326_move_look"](BASKET, "observe_scene/basket")
assert calls["q0"] == 0 and calls["ik"] == 9 and len(calls["pose"]) == 1 and calls["exec"] == 1
assert state["tcp"] == c["xyz"] and env["_V327_LOOK"]["at"] == c["xyz"]
assert any(l.startswith("[V3.27 WRIST LOOK SCREEN] observe_scene/basket") and "solutions 4" in l for l in logs)
assert any(l.startswith("[V3.27 WRIST LOOK OK] observe_scene/basket") and "via MoveIt joint goal" in l for l in logs)
print(f"[L3a] screen 9 views -> 4 IK solutions; joint plan #1 + cuRobo fallback fail -> #2 executed ({c['tag']})")

# (b) no IK anywhere -> WRIST_LOOK_UNREACHABLE (the capture then runs from the current pose, as in V3.26)
env, logs, calls, state = make_env(lambda xyz, q: False)
try:
    env["_v326_move_look"](HOLDER, "observe_scene/holder")
    raise AssertionError("expected WRIST_LOOK_UNREACHABLE")
except RuntimeError as exc:
    assert str(exc).startswith("WRIST_LOOK_UNREACHABLE: no collision-free IK in 40 candidate views"), exc
assert calls["exec"] == 0 and env["_V327_LOOK"]["at"] is None
print("[L3b] no IK in 40 views -> WRIST_LOOK_UNREACHABLE, nothing executed")

# (c) standing at a look pose (z < 0.315 near the basket) -> next view without the q0 detour; elsewhere low -> q0
env, logs, calls, state = make_env(lambda xyz, q: True)
c1 = env["_v326_move_look"](BASKET, "observe_scene/basket")
env["_v307_posture_numbers"] = lambda: ({"near_basket": True, "near_holder": False, "xyz": state["tcp"]}, {})
c2 = env["_v326_move_look"](HOLDER, "observe_scene/holder")
assert calls["q0"] == 0 and calls["exec"] == 2 and state["tcp"] == c2["xyz"]
env["_V327_LOOK"]["at"] = [9, 9, 9]
env["_v326_move_look"](BASKET, "plan_grasp:egg_1/basket")
assert calls["q0"] == 1
print("[L3c] look -> look without q0; near the basket but not at a look pose -> q0 first (V3.26 rule kept)")

# (d) backend wiring
for mk in ("[V3.27 WRIST LOOK SCREEN]", "[V3.27 WRIST LOOK OK]", "WRIST_LOOK_UNREACHABLE", 'payload["look_failed"]',
           "_v326w.look_pose_candidates(pt", "skill_core._plan_joint_solution(joints"):
    assert mk in B, mk
assert "_plan_pose(xyz, _v27_top_down_quat(0.0), 0.20, 0.15)" not in B                  # the V3.26 single pose is gone
print("[L3d] backend markers present; V3.26 single-pose cuRobo look removed")

# ------------------------------------------------------------------ L4
h = hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest()
assert h.startswith("020cc6680f181872"), h
print("[L4] NERO_V308_Camera_Perception.py unchanged (fixed-camera algorithm)")
print("ALL V3.27 TESTS PASS")
