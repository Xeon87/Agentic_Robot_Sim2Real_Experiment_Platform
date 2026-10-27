#!/usr/bin/env python3
"""V3.36: V3.34 grasp physics back, frozen grasp / placement data, stricter carton height + moved check, egg
recognition that keeps the shadowed half of the egg.

Q1 physics = V3.34: the simulator function and the gripper builder are byte-identical to V3.34 (source hashes), the
   CONNECT auto-hold / clamp / GUI checkbox are the V3.34 ones, no V3.35 friction code is left
Q2 recognition (synthetic MuJoCo-like shading: darker textured shell, ambient + headlight + key, specular): the old
   V3.08 mask (brightness >= 0.22) loses the shadowed lower half -> triangulated centres too high; the V3.36 mask
   keeps it; touching eggs get NO size (not a wrong one), an isolated egg's size is measured; yaw fused over views;
   the height check re-intersects the rays at table + half width when the triangulated height disagrees
Q3 freeze: a frozen egg keeps the geometry it was planned with, an egg without a clean size gets the labelled
   placeholder + margin; wiring of the freeze / unfreeze points
Q4 carton height: 2 views or a too narrow view spread are NOT a measurement (V3.35 accepted 0.009 m from 2 views),
   3 spread views measure it and lock it; implausible heights are refused; pre-motion / insertion checks stop on a
   live-view suspicion (HOLDER_POSSIBLY_MOVED) and re-plan; camera process confirms a jump only from two positions
"""
import ast, hashlib, math, os, sys, textwrap, time, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np                       # noqa: E402
import NERO_V326_Wrist_Camera as W       # noqa: E402
import NERO_V308_Camera_Perception as P8  # noqa: E402
import v329_synth as S                   # noqa: E402

assert W.VERSION in ("V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
import re as _re343  # V3.43: the A2 scene-setup blocks are stripped; everything else must be unchanged
B = _re343.sub(r"\n[ \t]*# V3\.43 A2 BEGIN\n.*?\n[ \t]*# V3\.43 A2 END(?=\n)", "", B, flags=_re343.S)
A = (ROOT / "NERO_V2_7_1_Agent.py").read_text()
tree = ast.parse(B)


def src(name):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(B, n)
    raise KeyError(name)


# ------------------------------------------------------------------ Q1
H34 = {"run_sim": "f94de6b4ad67541b", "add_official_gripper_to_xml": "6e5f95bef01350f6"}


def _run_sim_core():
    """V3.37: run_sim minus hard_restore_arm (+ its V3.37 keep-hold helper and call argument) - the part that must
    still be byte-identical to V3.34 (contacts, friction, auto CONNECT hold, clamp, guards)."""
    s = src("run_sim")
    i, j = s.find("        _v337_keep = {"), s.find("        def hard_restore_arm(")
    s = s[:(i if i >= 0 else j)] + s[s.find("        def soft_freeze_arm("):]
    return hashlib.sha256(s.replace("                            keep_hold=True,\n", "").encode()).hexdigest()[:16]


for fn, h in H34.items():
    if fn == "run_sim" and "_v337_keep" in B:
        assert _run_sim_core() == "0564058f38cb9fab", "run_sim outside hard_restore_arm differs from V3.34"
        continue
    assert hashlib.sha256(src(fn).encode()).hexdigest()[:16] == h, fn
for mk in ('hold_enabled_shared = mp.Value("b", True)', 'text="Enable Auto HOLD"', "hold_var = tk.BooleanVar(value=True)",
           "if holding or (left >= 0.30 and right >= 0.30): break", "deadline = time.time() + 3.2"):
    assert mk in B, mk
for gone in ("_v335_physics_xml", "V335_SENSE_N", "[V3.35 GRASP LOST]", "noslip_iterations", "friction335",
             "NERO_V335_ASSIST_HOLD", "FRICTION GRASP SENSING"):
    assert gone not in B, gone
assert "friction only" not in A.lower() and "Eggs are held by friction only" not in A
print("[Q1] grasp physics = V3.34 (simulator + gripper source identical, auto CONNECT hold, V3.34 clamp); no V3.35 "
      "friction code left")

# ------------------------------------------------------------------ Q2
cfg = W.wrist_config("usb130"); K = P8.intrinsics(cfg); pc, fc = W.camera_in_tcp(cfg)
KEYD = S.KEY[1]


def egg_mj(c, yaw, a, b, cam, texv=0.5):
    """MuJoCo-like shading: textured shell (darker), headlight ambient 0.25 + diffuse 0.30 x (n.v), key light, weak
    fill, specular highlight (white, low saturation)."""
    cy_, sy_ = math.cos(yaw), math.sin(yaw)
    R = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1]]); axes = (a, b, b)
    P = S._ellipsoid(c, axes, R)
    loc = (P - np.asarray(c)) @ R
    N = (loc / np.asarray(axes) ** 2) @ R.T
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    v = cam[None, :] - P; v /= np.linalg.norm(v, axis=1, keepdims=True)
    head = 0.25 + 0.30 * np.clip((N * v).sum(1), 0, None)
    key = np.clip(N @ KEYD, 0, None)[:, None] * S.KEY[0]
    fill = np.clip(N @ S.FILL[1], 0, None)[:, None] * S.FILL[0] * 0.5
    alb = np.array([0.88, 0.62, 0.37]) * texv
    col = np.clip(alb * (head[:, None] + 0.9 * key + fill), 0, 1) * 255
    r = 2 * (N @ KEYD)[:, None] * N - KEYD
    col = np.clip(col + (np.clip((r * v).sum(1), 0, 1) ** 40)[:, None] * 255, 0, 255)
    return P, col


EGGS = [([-0.451, -0.181, 0.0219], 0.3, 0.0315, 0.0219), ([-0.40, -0.17, 0.0219], -0.5, 0.0315, 0.0219),
        ([-0.45, -0.12, 0.0185], 1.2, 0.027, 0.0185)]           # eggs 0 + 1 touch, egg 2 is alone
VIEW_PTS = ([-0.44, -0.105, 0.02], [-0.44, -0.315, 0.02], [-0.42, -0.16, 0.03], [-0.20, -0.315, 0.02])


def run(new_mask):
    os.environ["NERO_V336_EGG_MASK"] = "1" if new_mask else "0"
    M = W.EggMeasure()
    last = {}
    for k, pt in enumerate(VIEW_PTS):
        c = [x for x in W.look_pose_candidates(pt, cfg=cfg, distances=[0.45]) if abs(x["tilt_deg"]) == 30][0]
        T = S.camera_T_for(c, pc, fc)
        parts = [S.table_rgb_part()] + [egg_mj(c_, y, a, b, T[:3, 3]) for c_, y, a, b in EGGS]
        dets = W.perceive_eggs_rgb(S.render_rgb(parts, K, T, seed=k), K, T)
        trk = {}
        for d in dets:
            i = min(range(len(EGGS)), key=lambda i: math.dist(d["xyz_m"][:2], EGGS[i][0][:2]))
            if math.dist(d["xyz_m"][:2], EGGS[i][0][:2]) < 0.04:
                trk[f"e{i}"] = {"seen": True, "det_id": d["det_id"], "xyz_m": d["xyz_m"], "location": "table_or_pile",
                                "trusted": True}
        W.apply_measurements(trk, dets, M, K, T, k)
        last.update(trk)
    return M, last


M_old, _ = run(False)
M_new, last_new = run(True)
z_old = [abs(M_old.get(f"e{i}")["xyz_m"][2] - EGGS[i][0][2]) for i in range(3)]
z_new = [abs(M_new.get(f"e{i}")["xyz_m"][2] - EGGS[i][0][2]) for i in range(3)]
assert max(z_old) > 0.0055 and max(z_new) < 0.0045 and max(z_old) > max(z_new) + 0.0025, (z_old, z_new)
e2 = M_new.get("e2")
assert e2["size_state"] == "measured" and abs(e2["width_m"] - 0.037) / 0.037 < 0.15, e2
assert all(M_new.get(f"e{i}")["width_m"] is None for i in (0, 1))             # touching -> no size, not a wrong one
dyaw = abs(math.atan2(math.sin(2 * (e2["yaw_rad"] - 1.2)), math.cos(2 * (e2["yaw_rad"] - 1.2)))) / 2
assert dyaw < math.radians(15), e2["yaw_rad"]
assert last_new["e2"]["measurement"]["yaw_views_fused"] and abs(last_new["e2"]["axis_world"][2]) < 1e-9
os.environ.pop("NERO_V336_EGG_MASK", None)
# height check: rays of a static egg at (0, 0, 0.040) but the triangulated height says 0.055
M = W.EggMeasure()
O = [np.array([0.3, 0.0, 0.4]), np.array([0.0, 0.3, 0.4]), np.array([-0.3, 0.0, 0.4])]
tgt = np.array([0.0, 0.0, 0.040])
M.est["x"] = {"ok": True, "xyz_m": [0.0, 0.0, 0.055], "width_m": 0.040,
              "_rays": [(list(o), list((tgt - o) / np.linalg.norm(tgt - o))) for o in O]}
xyz, srcz = M.height_check("x", table_z=0.020)
assert abs(xyz[2] - 0.040) < 1e-6 and math.dist(xyz[:2], [0, 0]) < 1e-4 and "disagreed" in srcz
xyz2, src2 = M.height_check("x", table_z=0.034)                              # 0.054 -> within 8 mm: keep
assert xyz2 == [0.0, 0.0, 0.055] and "consistent" in src2
print(f"[Q2] shadowed shell: old mask z error {max(z_old) * 1000:.1f} mm -> V3.36 {max(z_new) * 1000:.1f} mm; touching "
      f"eggs: size NOT measured; lone egg {e2['width_m'] * 1000:.1f} mm (true 37.0), yaw fused "
      f"({math.degrees(dyaw):.0f} deg); height check re-intersects rays at table + half width")

# ------------------------------------------------------------------ Q3
V3 = {"frozen_grasp": {"egg_1": {"geom": {}}}}
G = {"egg_1": {"effective_width_mm": 44.0, "size_source": "planned"}, "egg_2": {"effective_width_mm": 1.0},
     "egg_3": {"effective_width_mm": 1.0}}
last3 = {"objects": {"egg_1": {"measurement": {"width_mm": 30.0, "length_mm": 50.0, "method": "multi_view_triangulation"}},
                     "egg_2": {"measurement": {"width_mm": None, "method": "multi_view_triangulation",
                                               "size_state": "views disagree"}},
                     "egg_3": {"measurement": {"width_mm": 41.0, "length_mm": 60.0, "size_views": 3,
                                               "method": "multi_view_triangulation"}}}}
env3 = {"os": os, "V335_STATE": V3, "V311_STATE": {"last": last3}, "V24_OBJECT_GEOMETRY": G}
exec(textwrap.dedent(src("_v335_measured_geometry")), env3)
env3["_v335_measured_geometry"]()
assert G["egg_1"] == {"effective_width_mm": 44.0, "size_source": "planned"}            # frozen: untouched
assert G["egg_2"]["effective_width_mm"] > 48.0 and "NOT measured" in G["egg_2"]["size_source"]
assert G["egg_3"]["effective_width_mm"] == 42.5 and "measured by the camera" in G["egg_3"]["size_source"]
for mk in ('fg336.clear()                               # a new grasp plan: freeze THIS egg\'s geometry now',
           '"[V3.36 FREEZE] {obj336}', 'elif name == "clamp_gripper" and ok336:', 'V335_STATE["frozen_place"] = {"obj"',
           '_v336_unfreeze_place(name)', 'if h and h.get("complete") and not V335_STATE.get("frozen_place"):',
           '"frozen_at_plan_grasp"', '(V335_STATE.get("frozen_grasp") or {}).pop(obj, None)'):
    assert mk in B, mk
print("[Q3] frozen egg keeps its planned geometry; unmeasured size -> labelled placeholder + margin; freeze at plan_grasp"
      " / plan_placement, unfreeze at clamp / new plan / release / moved carton")

# ------------------------------------------------------------------ Q4
hx, hy = -0.42, 0.155
parts4 = [S.table_rgb_part(), S.carton_rgb_part(hx, hy, 0.0)]


def view(pt, shift=0.0):
    c = [x for x in W.look_pose_candidates(pt, cfg=cfg, distances=[0.45]) if abs(x["tilt_deg"]) == 30][0]
    T = S.camera_T_for(c, pc, fc).copy(); T[0, 3] += shift
    h = W.detect_holder_rgb(S.render_rgb(parts4, K, T), K, T)
    return h.pop("_px_uv"), T


v_a, v_b, v_c = view([hx, hy, 0.043]), view([-0.44, 0.105, 0.02]), view([-0.44, 0.315, 0.02])
H = W.HolderHeight()
H.add(v_a[0], K, v_a[1]); H.add(v_b[0], K, v_b[1])
assert H.top_z() is None and not H.est["ok"] and "views 2" in H.est["reason"]                   # 2 views: no
H.add(v_c[0], K, v_c[1])
assert H.top_z() is not None and abs(H.top_z() - 0.048) < 0.004 and H.est["sharp_minimum"], H.est
locked = H.top_z()
H.add(*view([hx, hy, 0.043], 0.03)[:1], K, view([hx, hy, 0.043], 0.03)[1])
assert H.top_z() == locked                                                                    # locked
H2 = W.HolderHeight()
for vv in (view([hx, hy, 0.043]), view([hx, hy, 0.043], 0.05), view([hx, hy, 0.043], 0.11)):
    H2.add(vv[0], K, vv[1])
assert H2.top_z() is None and "view angle" in H2.est["reason"], H2.est                        # same direction: no
H3 = W.HolderHeight()
for vv in (v_a, v_b, v_c):
    H3.add(vv[0], K, vv[1], table_z=0.040)                                                    # 8 mm above "table"
assert H3.top_z() is None and H3.est["plausible_vs_table"] is False
env4 = {"os": os, "math": math, "time": time, "skill_core": types.SimpleNamespace(emit=lambda *a: None),
        "v326_wrist_selected": lambda: True, "_v329_no_prior": lambda: True, "v24_skill_state": {"active_object": "egg_1"},
        "V27_GRASP_RUNTIME": {"plans": {}}, "_v335_ingest_live": lambda p: None, "_v335_inhand_verdict": lambda: {},
        "V271_PLACEMENT_RUNTIME": {"plans": {"egg_1": {"slot_name": "slot_2", "slot_center_xyz_m": [-0.45, 0.123, 0.043]}}},
        "_v336_perceived_slot": lambda sl: (-0.45, 0.123, 0.043), "_v335_capture": lambda tag: {"objects": {}},
        "_v335_drain": lambda: None,
        "V335_STATE": {"frozen_place": {"t": 100.0}, "holder_suspect": {"t": 200.0, "d_mm": 27.0}}}
env4.setdefault("_v338_refine_slot", lambda *a, **k: None)   # V3.38 slot refine (tested in test_v3_38)
exec(textwrap.dedent(src("_v335_precheck")), env4)
exec(textwrap.dedent(src("_v335_slot_moved_now")), env4)
r4 = env4["_v335_precheck"]("move_to_slot_entry", {"object_name": "egg_1"})
assert r4["reason"] == "HOLDER_POSSIBLY_MOVED" and r4["recommended_next_actions"] == ["plan_placement"]
assert env4["_v335_slot_moved_now"]("egg_1", "slot_2", (-0.45, 0.123, 0.043)) == "HOLDER_POSSIBLY_MOVED_DURING_INSERTION"
env4["V335_STATE"]["holder_suspect"] = {"t": 50.0, "d_mm": 27.0}                              # older than the plan
assert env4["_v335_precheck"]("move_to_slot_entry", {"object_name": "egg_1"}) is None
WM = (ROOT / "NERO_V326_Wrist_Camera.py").read_text()
for mk in ('hjumps[pf] = jl + [{"c": list(holder["center_xy_m"]), "cam": cam336}]', "hh_meas[pf].reset()            # confirmed",
           'not holder.get("moved_suspect")', "hh_meas[pf].add(px335, K, T, table_z=zt_meas)"):
    assert mk in WM, mk
assert 'startswith(("SLOT_MOVED", "HOLDER_POSSIBLY"))' in B and "HOLDER_POSSIBLY_MOVED" in A
print(f"[Q4] carton height: 2 views / one viewing direction / implausible -> not measured; 3 spread views -> "
      f"{locked * 1000:.1f} mm (true 48) and locked; live suspicion -> HOLDER_POSSIBLY_MOVED / insertion stop -> "
      "re-plan with a dedicated look; camera process confirms a jump only from two positions")
print("ALL V3.36 TESTS PASS")
