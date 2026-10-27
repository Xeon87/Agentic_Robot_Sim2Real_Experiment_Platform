#!/usr/bin/env python3
"""V3.29 wrist camera without a preset layout and without simulator truth in the checks.

H1 (numpy) holder perception on synthetic depth (table + MJCF-like holder + eggs): full view -> complete, slot
   centres within 2 mm, yaw within 0.5 deg (also for a holder moved/rotated by 15 deg); partial view -> not
   complete; holder_boxes() at the registered pose equal the registered MoveIt boxes
H2 scan points are a ring of directions around the base (no scene coordinates)
H3 (numpy) backend scan flow with a fake arm + synthetic camera: blind ring scan finds the holder (registered
   pose AND a moved/rotated one), all 4 eggs, the egg pile; slot dicts, holder centre, basket/pile centre, safe
   orientation XY and the MoveIt override file come from perception; slots unknown -> _v271_slot refuses
H4 no-truth object poses: held -> TCP x grasp transform, released over a slot -> that slot, otherwise perception;
   truth only in [V3.29 TRUTH LOG]; non-egg bodies and other pose sources unchanged
H5 wiring markers: wrist process forgets the registered layout, payload carries containers, pre-place holder
   look, simulator bridge publishes the perceived holder boxes, fixed-camera module unchanged
"""
import hashlib, json, math, os, sys, tempfile, textwrap, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import NERO_V326_Wrist_Camera as W  # noqa: E402

assert W.VERSION in ("V3.29", "V3.30", "V3.31", "V3.31.1", "V3.33", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
try:
    import numpy as np
    import NERO_V308_Camera_Perception as P8
    import v329_synth as S
    have_np = True
except Exception:
    have_np = False

EGGS = {"egg_1": ([-0.451, -0.181, 0.022], 0.3), "egg_2": ([-0.399, -0.181, 0.023], -0.5),
        "egg_3": ([-0.452, -0.130, 0.021], 1.1), "egg_4": ([-0.399, -0.129, 0.023], 2.0)}

# ------------------------------------------------------------------ H1
if have_np:
    K = P8.intrinsics(P8.camera_config())
    pc, fc = W.camera_in_tcp()

    def scene_points(hx=-0.420, hy=0.155, hyaw=0.0, eggs=EGGS, placed=()):
        parts = [S.table_points(), S.holder_points(hx, hy, hyaw)]
        for _n, (c, yw) in eggs.items():
            parts.append(S.egg_points(c, True, yw))
        for c in placed:
            parts.append(S.egg_points(c, False))
        return np.vstack(parts)

    def view(pt, P, dist=None):
        c = W.look_pose_candidates(pt, distances=dist or [0.36])[0]
        T = S.camera_T_for(c, pc, fc)
        return S.render_depth(P, K, T), T, c

    for hx, hy, hyaw in ((-0.420, 0.155, 0.0), (-0.380, 0.200, math.radians(15))):
        P = scene_points(hx, hy, hyaw, placed=[S.slot_truth(hx, hy, hyaw)["slot_1"][:2] + (0.054,)])
        D, T, _c = view([hx, hy, 0.043], P)
        sl1 = S.slot_truth(hx, hy, hyaw)["slot_1"]
        h = W.detect_holder(D, K, T, dets=[{"xyz_m": [sl1[0], sl1[1], 0.054], "axis_world": [0, 0, 1]}])
        tr = S.slot_truth(hx, hy, hyaw)
        errs = [math.dist(h["slots"][k]["xyz_m"][:2], tr[k][:2]) * 1000 for k in tr]
        assert h["complete"] and max(errs) < 2.0, (h, errs)
        assert abs(math.degrees(h["yaw_rad"] - hyaw)) < 0.5 and abs(h["slots"]["slot_3"]["xyz_m"][2] - 0.043) < 0.003
        print(f"[H1] holder at ({hx}, {hy}) yaw {math.degrees(hyaw):.0f} deg: complete, yaw err "
              f"{abs(math.degrees(h['yaw_rad'] - hyaw)):.2f} deg, slot err max {max(errs):.1f} mm (egg in slot_1)")
    P = scene_points()
    D, T, _c = view([-0.42, 0.0, 0.03], P, [0.40])
    hp = W.detect_holder(D, K, T)
    assert hp is None or not hp["complete"]
    reg = [("egg_holder_base", (-0.420, 0.155, 0.006), (0.250, 0.144, 0.012)),
           ("egg_holder_rim_xneg", (-0.548, 0.155, 0.025), (0.008, 0.152, 0.050)),
           ("egg_holder_rim_xpos", (-0.292, 0.155, 0.025), (0.008, 0.152, 0.050)),
           ("egg_holder_rim_yneg", (-0.420, 0.079, 0.025), (0.248, 0.008, 0.050)),
           ("egg_holder_rim_ypos", (-0.420, 0.231, 0.025), (0.248, 0.008, 0.050))]
    bx = W.holder_boxes({"center_xy_m": [-0.420, 0.155], "yaw_rad": 0.0, "table_z_m": 0.0})
    for (n1, c1, s1), (n2, c2, s2, y2) in zip(reg, bx):
        assert n1 == n2 and math.dist(c1, c2) < 1e-9 and list(s1) == list(s2) and y2 == 0.0
    print("[H1] partial view of the holder -> not complete; model boxes at the registered pose == registered MoveIt boxes")
else:
    print("[H1] numpy missing -> holder perception skipped")

# ------------------------------------------------------------------ H2
os.environ["NERO_V329_SCAN_AZ_DEG"] = "180,145,215,110,250"    # V3.31: the ring is opt-in (default = workspace grid)
sp = W.scan_points()
del os.environ["NERO_V329_SCAN_AZ_DEG"]
assert [v for v, _ in sp] == ["scan_az180", "scan_az145", "scan_az215", "scan_az110", "scan_az250"]
assert all(abs(math.hypot(p[0], p[1]) - 0.42) < 1e-3 for _, p in sp)
print("[H2] scan = ring r=0.42 m around the base, azimuths 180/145/215/110/250 deg (directions only)")

# ------------------------------------------------------------------ H3 / H4 backend block with fakes
B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
a = B.index("    _V329_TRUTH = {}")
b = B.index("    def _v326_wrist_capture(label, names=None):")
code = textwrap.dedent(B[a:b])


class Val:
    def __init__(self, v=0): self.value = v


def make_env(world, tmp):
    logs = []
    sc = types.SimpleNamespace(emit=logs.append)
    state = {"tcp": [0.0, -0.002, 0.873], "tq": [0, 0, 0, 1], "cand": None}
    sc.current_tcp_pose = lambda: (list(state["tcp"]), list(state["tq"]))
    sc.object_pose = lambda n: (list(world["truth"][n]), [0, 0, 0, 1])
    sc._current_joint_state_copy = lambda: "LIVE"
    sc._arm_positions_from_joint_state = lambda js: [0.0] * 7
    sc._plan_joint_solution = lambda *a_, **k: None
    slots_reg = {f"slot_{i + 1}": (x, y, 0.043) for i, (y, x) in enumerate(
        [(y, x) for y in (0.123, 0.187) for x in (-0.510, -0.450, -0.390, -0.330)])}
    env = {"os": os, "math": math, "json": json, "time": __import__("time"),
           "V271_SLOT_CENTERS_M": dict(slots_reg), "EGG_SLOT_POSES": dict(slots_reg),
           "EGG_BASKET": {"name": "egg_basket", "center_xy_m": [-0.425, -0.155], "radius_m": 0.088},
           "EGG_HOLDER": {"name": "egg_holder", "center_xyz_m": [-0.420, 0.155, 0.006]},
           "V27_BASKET_CENTER_XY_M": (-0.425, -0.155), "V271_HOLDER_RIM_TOP_Z_M": 0.044,
           "V271_SAFE_ORIENTATION_XY_M": (-0.4225, 0.0),
           "V329_LAYOUT_STATE": {"perceived_required": False, "holder_ok": False},
           "V24_OBJECT_GEOMETRY": {f"egg_{i}": {"approx_size_mm": [44.0 + i, 42.0, 62.0], "effective_width_mm": 44.0 + i}
                                   for i in range(1, 5)},
           "V329_SCENE_OVERRIDE_PATH": Path(tmp) / "logs" / "v329_moveit_holder.json",
           "v326_wrist_selected": lambda: True, "v311_mode": lambda: "wrist", "v331_fixed_no_prior": lambda: True,
           "skill_core": sc, "hold_active_shared": Val(False),
           "v24_skill_state": {"active_object": None}, "V311_STATE": {"last": None}, "V326_SHARED": {},
           "EGG_OBJECTS": ["egg_1", "egg_2", "egg_3", "egg_4"],
           "_v328_tag_with_hints": lambda t: t, "_v218_slow_trajectory_in_place": lambda t, factor=1.0: t,
           "_v310_wait_tcp_settled": lambda *a_: None}

    def slot_for(xyz):
        best = min(env["EGG_SLOT_POSES"].items(), key=lambda kv: math.hypot(xyz[0] - kv[1][0], xyz[1] - kv[1][1]))
        return best[0] if math.hypot(xyz[0] - best[1][0], xyz[1] - best[1][1]) <= 0.026 and xyz[2] <= 0.095 else None
    env["_egg_slot_for_xyz"] = slot_for
    env["_v311_obj_entry"] = lambda n: ((env["V311_STATE"].get("last") or {}).get("objects") or {}).get(n)

    def held_est(n, tcp, tq):
        return ([tcp[0], tcp[1], tcp[2] - 0.03], [0, 0, 0, 1])
    env["_v3121_held_pose_estimate"] = held_est

    def move_look(pt, label, distances=None, no_detour=False):
        state["cand"] = W.look_pose_candidates(pt, distances=distances)[0]
        state["tcp"] = list(state["cand"]["xyz"])
        world["moves"].append(label)
    env["_v326_move_look"] = move_look
    env.setdefault("V335_STATE", {"holder_hist": [], "holder_jump": [], "in_hand_bad": 0})   # V3.35
    env.setdefault("_v335_capture", lambda tag: W.capture_sync({}, tag)); env.setdefault("_v335_holder_filter", lambda h, tag=None, cam=None: h)
    exec(code, env)
    return env, logs, state


if have_np:
    seq = {"n": 0}

    def make_world(hx, hy, hyaw):
        truth = {n: c for n, (c, _y) in EGGS.items()}
        return {"truth": truth, "moves": [], "P": scene_points(hx, hy, hyaw), "holder": (hx, hy, hyaw)}

    def fake_capture_factory(world, state):
        def cap(shared, tag, timeout_s=None):
            seq["n"] += 1
            c = state["cand"]
            T = S.camera_T_for(c, pc, fc)
            D = S.render_depth(world["P"], K, T)
            objs, dets = {}, []
            for n, xyz in world["truth"].items():
                vis = W.in_view(xyz, K, T.tolist())
                objs[n] = {"xyz_m": list(xyz), "quat_xyzw": [0, 0, 0, 1], "seen": vis, "trusted": vis}
                if vis:
                    dets.append({"xyz_m": list(xyz), "axis_world": [1, 0, 0]})
            h = W.detect_holder(D, K, T, dets=dets)
            return {"capture": seq["n"], "tag": tag, "objects": objs, "containers": {"egg_holder": h},
                    "cam_pos": [round(float(v), 3) for v in T[:3, 3]], "wait_s": 0.0}
        return cap

    for hx, hy, hyaw in ((-0.420, 0.155, 0.0), (-0.380, 0.200, math.radians(15))):
        tmp = tempfile.mkdtemp(prefix="v329_")
        world = make_world(hx, hy, hyaw)
        env, logs, state = make_env(world, tmp)
        W.capture_sync = fake_capture_factory(world, state)
        env["_v329_reset_scene"]("test")
        assert env["V329_LAYOUT_STATE"] == {"perceived_required": True, "holder_ok": False, "pile_ok": False}
        assert json.loads(env["V329_SCENE_OVERRIDE_PATH"].read_text())["boxes"] == []          # MoveIt holder removed
        ready = env["_v329_scan"]("observe_scene", [], [], [])
        assert ready, logs
        tr = S.slot_truth(hx, hy, hyaw)
        errs = [math.dist(env["V271_SLOT_CENTERS_M"][k][:2], tr[k][:2]) * 1000 for k in tr]
        assert max(errs) < 2.0 and env["EGG_SLOT_POSES"] == env["V271_SLOT_CENTERS_M"], errs
        assert math.dist(env["EGG_HOLDER"]["center_xyz_m"][:2], [hx, hy]) < 0.002
        ec = [sum(c[0] for c, _ in EGGS.values()) / 4, sum(c[1] for c, _ in EGGS.values()) / 4]
        assert math.dist(env["EGG_BASKET"]["center_xy_m"], ec) < 0.003 and env["V27_BASKET_CENTER_XY_M"] == tuple(env["EGG_BASKET"]["center_xy_m"])
        assert env["V329_LAYOUT_STATE"]["holder_ok"]
        ov = json.loads(env["V329_SCENE_OVERRIDE_PATH"].read_text())
        assert len(ov["ids"]) == 5 and len(ov["boxes"]) == 5 and abs(ov["boxes"][0][3] - hyaw) < 0.01
        sx = env["V271_SAFE_ORIENTATION_XY_M"]
        assert math.dist(sx, [(ec[0] + hx) / 2, (ec[1] + hy) / 2]) < 0.003
        assert any(l.startswith("[V3.29 SCENE PERCEPTION]") for l in logs) and any(l.startswith("[V3.29 SCAN DONE]") for l in logs)
        print(f"[H3] blind scan, holder at ({hx}, {hy}) yaw {math.degrees(hyaw):.0f} deg: views {world['moves']} -> "
              f"holder complete, slot err max {max(errs):.1f} mm, 4 eggs, pile {env['EGG_BASKET']['center_xy_m']}, "
              "MoveIt boxes from perception")

    # H4 no-truth estimator (registered-pose run)
    env["_v329_install_no_truth"]()
    sc = env["skill_core"]
    env["V311_STATE"]["last"] = {"objects": {"egg_2": {"xyz_m": [-0.40, -0.18, 0.023], "quat_xyzw": [0, 0, 0, 1], "seen": True}}}
    world["truth"]["egg_2"] = [-0.30, -0.18, 0.023]                         # simulator disagrees by 100 mm
    xyz, _ = sc.object_pose("egg_2")
    assert xyz == [-0.40, -0.18, 0.023] and any("[V3.29 TRUTH LOG] egg_2" in l for l in logs)
    env["hold_active_shared"].value = True
    env["v24_skill_state"]["active_object"] = "egg_2"
    s1 = env["V271_SLOT_CENTERS_M"]["slot_1"]
    state["tcp"] = [s1[0] + 0.002, s1[1], 0.13]
    xyz, _ = sc.object_pose("egg_2")
    assert math.dist(xyz, [s1[0] + 0.002, s1[1], 0.10]) < 1e-9                  # TCP x grasp transform
    env["hold_active_shared"].value = False
    xyz, _ = sc.object_pose("egg_2")
    assert math.dist(xyz, [s1[0], s1[1], W.egg_center_in_slot_z(s1[2] - 0.043)]) < 1e-9                # released over slot_1 -> slot_1
    env["_v329_ingest"]({"objects": {"egg_2": {"seen": True, "xyz_m": [s1[0], s1[1], 0.054]}}})
    assert "egg_2" not in env["_V329_EST"]["released"]                         # camera sees it again -> perception
    env["v326_wrist_selected"] = lambda: False
    assert sc.object_pose("egg_2")[0] == world["truth"]["egg_2"]                # other pose sources unchanged
    print("[H4] no-truth poses: perception / TCP x grasp while held / released slot; truth only logged; other modes unchanged")
else:
    print("[H3/H4] numpy missing -> skipped")

# ------------------------------------------------------------------ H5
WM = (ROOT / "NERO_V326_Wrist_Camera.py").read_text()
for mk in ("P8.SLOTS = {}", '"containers": {"egg_holder": holder}', "holder = detect_holder(depth, K, T, dets=dets, robot_boxes=boxes)"):
    assert mk in WM, mk
for mk in ("_v329_preplace_look(obj, slot_name)", "ovr329 = _v329_read_scene_override()", "CollisionObject.REMOVE",
           "def _v329_read_scene_override():", 'if V329_LAYOUT_STATE.get("perceived_required") and not V329_LAYOUT_STATE.get("holder_ok"):',
           "_v329_reset_scene(str(reason))", "near_basket, near_holder = _v329_near_zone(x, y, z, 0.315)",
           "_tn330, txyz = _v330_nearest_truth(o[\"xyz_m\"])", "_tloc311 = llm_classify_object_location(_v330_nearest_truth(xyz)[1] if _v329_no_prior() else _v329_truth_pose(obj)[0])",
           "_v329_install_no_truth()", "V329_SCENE_OVERRIDE_PATH.unlink()"):
    assert mk in B, mk
assert "0.425" not in code and "0.155" not in code                             # the V3.29 block has no scene coordinates
h = hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest()
assert h.startswith("020cc6680f181872"), h
print("[H5] wiring: process forgets the registered layout, containers in payload, pre-place look, MoveIt holder from "
      "perception, slot guard, posture zones from the layout; fixed-camera module unchanged")
print("ALL V3.29 TESTS PASS")
