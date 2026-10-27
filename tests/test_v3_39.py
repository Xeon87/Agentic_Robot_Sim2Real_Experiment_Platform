#!/usr/bin/env python3
"""V3.39: the USB (RGB-only) wrist camera finds the carton faster and grasps at the right height.

U1 consensus triangulation: rays of a pile egg mixed with its neighbour's (mis-association) -> V3.38 'not triangulated'
   or a wrong position; V3.39 keeps the largest consistent ray set
U2 table plane: without a measured egg size / measured table V3.38 skipped the height check (egg z 0.048 instead of
   ~0.022); V3.39 checks lying eggs against the calibrated table plane + class half width; badly conditioned eggs no
   longer 'measure' the table
U3 plane fallback: triangulation unusable (rays nearly parallel) but two views from different camera positions agree on
   the lying-egg plane -> measured + trusted
U4 close-view association: an untrusted (gripper-cut) detection near the MEASURED position gets the name (V3.38:
   'seen []' -> recapture + q0 round trips); ambiguous -> not assigned
U5 carton from two partial views (far edges on complementary axes)
U6 backend: scan refines a partial carton at once, tries points nearer the base, measures the height in the same scan,
   repeat scans skip empty views; plan_grasp takes extra close views in place; implausible grasp height refused
"""
import math, os, sys, textwrap, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np                       # noqa: E402
import NERO_V326_Wrist_Camera as W       # noqa: E402
import NERO_V308_Camera_Perception as P8  # noqa: E402
import v329_synth as S                   # noqa: E402
import hashlib                           # noqa: E402

assert W.VERSION in ("V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1") and (ROOT / "VERSION").read_text().strip() in ("V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
assert hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest().startswith("020cc6680f181872")
B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
A = (ROOT / "NERO_V2_7_1_Agent.py").read_text()
cfg = W.wrist_config("usb130"); K = P8.intrinsics(cfg); pc, fc = W.camera_in_tcp(cfg)


def env_set(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        os.environ[k] = v
    return old


def env_restore(old):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ------------------------------------------------------------------ U1
EGGS = [([-0.451, -0.181, 0.0219], 0.3), ([-0.407, -0.170, 0.0219], -0.4), ([-0.430, -0.222, 0.0219], 1.4)]   # a pile
VIEW_PTS = ([-0.44, -0.105, 0.02], [-0.44, -0.315, 0.02], [-0.42, -0.16, 0.03], [-0.20, -0.315, 0.02],
            [-0.52, -0.20, 0.02], [-0.36, -0.10, 0.02])
parts = [S.table_rgb_part()] + [S.egg_rgb_part(c, True, y) for c, y in EGGS]
views = []
for k, pt in enumerate(VIEW_PTS):
    c = [x for x in W.look_pose_candidates(pt, cfg=cfg, distances=[0.45]) if abs(x["tilt_deg"]) == 30][0]
    T = S.camera_T_for(c, pc, fc)
    dets = W.perceive_eggs_rgb(S.render_rgb(parts, K, T, seed=k), K, T)
    by = {}
    for d in dets:
        i = min(range(len(EGGS)), key=lambda i: math.dist(d["xyz_m"][:2], EGGS[i][0][:2]))
        if math.dist(d["xyz_m"][:2], EGGS[i][0][:2]) < 0.03:
            by[i] = d
    views.append((T, by))
WRONG = {1, 4}       # in these views egg 0 and egg 1 merged into ONE blob -> its centroid got egg 0's name (pile)


def merged(d0, d1):
    return dict(d0, px_uv=[(d0["px_uv"][i] + d1["px_uv"][i]) / 2.0 for i in range(2)],
                xyz_m=[(d0["xyz_m"][i] + d1["xyz_m"][i]) / 2.0 for i in range(3)], size_ok=False)


def measure_e0():
    M = W.EggMeasure()
    for k, (T, by) in enumerate(views):
        if 0 not in by:
            continue
        d = merged(by[0], by[1]) if k in WRONG else by[0]
        M.add("e0", d, K, T, k)
    return M


old = env_set(NERO_V339_TRI_CONSENSUS="0")
M38 = measure_e0()
env_restore(old)
M39 = measure_e0()
e38, e39 = M38.get("e0"), M39.get("e0")
bad38 = (not e38["ok"]) or math.dist(e38["xyz_m"], EGGS[0][0]) > 0.010
assert bad38, e38
assert e39["ok"] and math.dist(e39["xyz_m"], EGGS[0][0]) < 0.005 and e39["n_outlier_rays"] >= 1, e39
print(f"[U1] pile egg with {len(WRONG)} of {len(views)} rays from the neighbour: V3.38 "
      + (f"not triangulated ({e38.get('rms_m')})" if not e38["ok"] else f"{1000 * math.dist(e38['xyz_m'], EGGS[0][0]):.0f} mm off")
      + f" -> V3.39 consensus {1000 * math.dist(e39['xyz_m'], EGGS[0][0]):.1f} mm ({e39['n_outlier_rays']} rays dropped)")

# ------------------------------------------------------------------ U2
Mh = W.EggMeasure()
O = [np.array([0.3, 0.0, 0.4]), np.array([0.0, 0.3, 0.4]), np.array([-0.3, 0.0, 0.4])]
tgt = np.array([-0.40, 0.0, 0.0219])
Mh.est["x"] = {"ok": True, "xyz_m": [-0.40, 0.0, 0.048], "width_m": None, "n_views": 3, "max_angle_deg": 30.0,
               "rms_m": 0.002, "_rays": [(list(o), list((tgt - o) / np.linalg.norm(tgt - o))) for o in O]}


def run_apply(**kv):
    old_ = env_set(**kv)
    Mh_obs = Mh.obs
    trk = {"x": {"seen": False, "trusted": False, "location": "table_or_pile", "xyz_m": [-0.40, 0.0, 0.048]}}
    W.apply_measurements(trk, [], Mh, K, np.eye(4), 1)
    env_restore(old_)
    assert Mh.obs is Mh_obs
    return trk["x"]


x38 = run_apply(NERO_V339_TABLE_FALLBACK="0")
x39 = run_apply()
assert abs(x38["xyz_m"][2] - 0.048) < 1e-9 and x38["measurement"]["height_source"] == "triangulated"
assert abs(x39["xyz_m"][2] - 0.0219) < 1e-6 and math.dist(x39["xyz_m"][:2], tgt[:2]) < 1e-4, x39
assert "class half width" in x39["measurement"]["height_source"] and "calibrated" in x39["measurement"]["table_plane"]
assert W.table_plane(None)[0] == W.table_z_m() and W.table_plane(0.004)[0] == 0.004
assert W.table_plane(0.030)[0] == W.table_z_m() and "rejected" in W.table_plane(0.030)[1]
Mt = W.EggMeasure()                  # a badly conditioned egg (5 deg, z 26 mm too high) must not 'measure' the table
Mt.est["a"] = {"ok": True, "xyz_m": [0, 0, 0.048], "width_m": 0.044, "max_angle_deg": 5.0, "rms_m": 0.003}
assert Mt.table_z({"a": "table_or_pile"}) is None
Mt.est["b"] = {"ok": True, "xyz_m": [0, 0, 0.0225], "width_m": 0.044, "max_angle_deg": 25.0, "rms_m": 0.002}
assert abs(Mt.table_z({"a": "table_or_pile", "b": "table_or_pile"}) - 0.0005) < 1e-9
print("[U2] lying egg triangulated 26 mm too high: V3.38 kept z 0.048 (no measured size/table -> no check); V3.39 "
      "re-intersects at the calibrated table + class half width -> z 0.0219; ill-conditioned eggs no longer set the table")

# ------------------------------------------------------------------ U3
Mp = W.EggMeasure()
tgt3 = np.array([-0.45, 0.02, 0.0219])
for k, o in enumerate((np.array([-0.45, 0.02, 1.00]), np.array([-0.39, 0.02, 1.00]))):       # 6 cm apart, ~3.4 deg
    d = (tgt3 - o) / np.linalg.norm(tgt3 - o)
    Mp.obs.setdefault("y", []).append({"o": list(o), "d": list(d), "seq": k, "cov": None, "size_ok": False, "f": 600.0,
                                        "border": False, "yaw": None, "elong": None, "z_axis": [0, 0, -1],
                                        "plane_xyz": list(tgt3)})
Mp._solve("y")
assert not Mp.get("y")["ok"] and "parallel" in Mp.get("y")["fail_reason"]
trk3 = {"y": {"seen": False, "trusted": False, "location": "table_or_pile", "xyz_m": [0, 0, 0]}}
W.apply_measurements(trk3, [], Mp, K, np.eye(4), 3)
assert trk3["y"]["measurement"]["method"] == "rays_at_table_plane" and math.dist(trk3["y"]["xyz_m"], tgt3) < 1e-4
Mq = W.EggMeasure()                 # same rays but from the SAME camera position -> no confirmation
for k in range(2):
    o = np.array([-0.45, 0.02, 1.00]); d = (tgt3 - o) / np.linalg.norm(tgt3 - o)
    Mq.obs.setdefault("y", []).append({"o": list(o), "d": list(d), "seq": k, "border": False, "plane_xyz": list(tgt3)})
assert Mq.plane_estimate("y", 0.0219) is None
print("[U3] rays nearly parallel (not triangulable) but two views 6 cm apart agree on the lying-egg plane -> measured "
      "'rays_at_table_plane' (V3.38: single_view_plane_assumption forever); one camera position alone -> not trusted")

# ------------------------------------------------------------------ U4


def tracker_case(predict, det_xy, others=None):
    base = P8.PerceptionTracker()
    base.tracks = {"egg_1": {"xyz_m": [-0.40, -0.16, 0.0219], "quat_xyzw": [0, 0, 0, 1], "axis_world": [1, 0, 0],
                             "location": "table_or_pile", "last_seq": 1, "trusted": True}}
    for n, p in (others or {}).items():
        base.tracks[n] = {"xyz_m": p, "quat_xyzw": [0, 0, 0, 1], "axis_world": [1, 0, 0], "location": "table_or_pile",
                          "last_seq": 1, "trusted": True}
    vt = W.ViewTracker(base, names=["egg_1", "egg_2", "egg_3"])
    det = {"det_id": 0, "xyz_m": [det_xy[0], det_xy[1], 0.0219], "quat_xyzw": [0, 0, 0, 1], "axis_world": [1, 0, 0],
           "location": "table_or_pile", "trusted": False, "confidence": 0.45,
           "quality": {"coverage": 0.45, "border": False, "rgb_only": True}}
    c = W.look_pose_candidates([-0.40, -0.16, 0.02], cfg=cfg, distances=[0.36])[0]
    T = S.camera_T_for(c, pc, fc)
    return vt.update([det], 2, K, T.tolist(), hints={}, predict=predict)


meas_xy = [-0.437, -0.16]                         # measured position: 37 mm from the stale single-view track
old = env_set(NERO_V339_PREDICT_ASSOC="0")
out38, _d = tracker_case({"egg_1": [meas_xy[0], meas_xy[1], 0.0219]}, [-0.435, -0.161])
env_restore(old)
out39, d39 = tracker_case({"egg_1": [meas_xy[0], meas_xy[1], 0.0219]}, [-0.435, -0.161])
assert not out38["egg_1"]["seen"]
assert out39["egg_1"]["seen"] and out39["egg_1"]["trusted"] and "predicted" in out39["egg_1"]["association"]
assert d39["predicted_assoc"] == ["egg_1"] and 0 not in d39["unassigned_detections"]
out_amb, _ = tracker_case({"egg_1": [meas_xy[0], meas_xy[1], 0.0219], "egg_2": [-0.4335, -0.1625, 0.0219]},
                          [-0.435, -0.161], others={"egg_2": [-0.452, -0.178, 0.0219]})
assert not out_amb["egg_1"]["seen"] and not out_amb["egg_2"]["seen"]        # two candidates: do not guess
print("[U4] close view, gripper-cut (untrusted) blob 2 mm from the measured position, 35 mm from the stale track: "
      "V3.38 'seen []' -> V3.39 associated + trusted; two eggs equally near -> left unassigned")

# ------------------------------------------------------------------ U5
hx, hy = -0.42, 0.155
parts5 = [S.table_rgb_part(), S.carton_rgb_part(hx, hy, 0.0)]


def holder_view(dx, dy):
    c = [x for x in W.look_pose_candidates([hx + dx, hy + dy, 0.02], cfg=cfg, distances=[0.36]) if abs(x["tilt_deg"]) == 30][0]
    T = S.camera_T_for(c, pc, fc)
    h = W.detect_holder_rgb(S.render_rgb(parts5, K, T), K, T)
    h.pop("_px_uv", None)
    return h


ha, hb = holder_view(-0.12, -0.06), holder_view(0.18, 0.06)
assert not ha["complete"] and not hb["complete"] and ha["far_edges_in_view"] == [True, False] \
    and hb["far_edges_in_view"] == [False, True], (ha["far_edges_in_view"], hb["far_edges_in_view"])
fz = W.fuse_partial_holders([ha, hb])
assert fz and fz["complete"] and fz["method"] == "rgb_carton_far_edges_fused_partial_views"
err5 = math.dist(fz["center_xy_m"], [hx, hy]) * 1000
tr5 = S.slot_truth(hx, hy, 0.0)
serr = max(math.dist(fz["slots"][k]["xyz_m"][:2], tr5[k][:2]) for k in tr5) * 1000
assert err5 < 4.0 and serr < 5.0, (err5, serr)
assert W.fuse_partial_holders([ha]) is None and W.fuse_partial_holders([ha, ha]) is None   # one axis only: no
hb_rot = dict(hb, yaw_rad=hb["yaw_rad"] + math.radians(10))
assert W.fuse_partial_holders([ha, hb_rot]) is None                                         # yaw disagrees: no
hc = dict(hb, far_edges_in_view=[True, True], center_xy_m=[ha["center_xy_m"][0] + 0.02, hb["center_xy_m"][1]])
assert W.fuse_partial_holders([ha, hb, hc]) is None                                         # far edges disagree: no
WM = (ROOT / "NERO_V326_Wrist_Camera.py").read_text()
for mk in ("fz339 = fuse_partial_holders(hparts[pf])", 'diag["holder_partial_fusion"]', "predict=pred339)"):
    assert mk in WM, mk
print(f"[U5] carton from 2 partial views (far edge u from one, v from the other): centre {err5:.1f} mm, slots max "
      f"{serr:.1f} mm; one axis only / yaw or edges disagreeing -> not fused")

# ------------------------------------------------------------------ U6


def block(start, end):
    i = B.index(start); return textwrap.dedent(B[i:B.index(end, i)])


logs = []
moves = []
scene = {"holder": None, "pile": None, "seen": set(), "unreachable": set(), "views_without_new": 0, "scan_count": 0,
         "grid_hist": {}}
GRID = W.scan_points()
state = {"phase": 0}


def fake_move(pt, label, distances=None, no_detour=False):
    moves.append(label)
    if "holder_refine" in label and not label.endswith("holder_refine_in"):
        raise RuntimeError("WRIST_LOOK_UNREACHABLE: no collision-free IK in 40 candidate views")


def fake_capture(tag):
    v = tag.split("/")[-1]
    h = None
    if v == GRID[1][0]:
        h = {"complete": False, "center_xy_m": [hx, hy], "look_xy_m": [hx + 0.03, hy], "yaw_rad": 0.0, "table_z_m": 0.0,
             "slots_with_posts": 3}
    elif v == "holder_refine_in":
        h = {"complete": True, "center_xy_m": [hx, hy], "yaw_rad": 0.0, "table_z_m": 0.0, "slots_with_posts": 8,
             "top_z_source": "carton model on the table plane (not measured yet)"}
    elif v.startswith("holder_height_2"):
        h = {"complete": True, "center_xy_m": [hx, hy], "yaw_rad": 0.0, "table_z_m": 0.0, "slots_with_posts": 8,
             "top_z_source": "measured_multi_view_parallax"}
    eggs = {"egg_1": {"seen": v == GRID[3][0]}}
    return {"tag": tag, "objects": eggs, "containers": {"egg_holder": h}}


def fake_ingest(p):
    h = (p.get("containers") or {}).get("egg_holder")
    if h and (h.get("complete") or not scene["holder"]):
        scene["holder"] = h
    if any(o.get("seen") for o in p["objects"].values()):
        scene["pile"] = {"center_xy_m": [-0.43, -0.17]}


g = {"os": os, "math": math, "skill_core": types.SimpleNamespace(emit=logs.append), "_V329_SCENE": scene,
     "_v326_move_look": fake_move, "_v335_capture": fake_capture, "_v329_ingest": fake_ingest, "V311_STATE": {},
     "_v329_scene_ready": lambda: False}
exec(block("    def _v329_scan(label, payloads, views, look_failed):", "    def _v329_preplace_look(obj, slot_name):"), g)
pay, vw, lf = [], [], []
g["_v329_scan"]("observe_scene", pay, vw, lf)
i_ref = moves.index("observe_scene/holder_refine_in")
assert moves.index(f"observe_scene/{GRID[2][0]}") > i_ref, moves             # refined at once, before the next grid view
assert "observe_scene/holder_height_1" in moves and "observe_scene/holder_height_2" in moves
assert str(scene["holder"]["top_z_source"]).startswith("measured")
assert any("[V3.39 HOLDER HEIGHT]" in m for m in logs)
empty = [k for k, v in scene["grid_hist"].items() if v == "empty"]
assert empty and scene["scan_count"] == 1
moves.clear(); logs.clear()
g["_v329_scan"]("observe_scene", [], [], [])
skipped = sum("[V3.39 SCAN REUSE]" in m for m in logs)
assert skipped == len(empty) and scene["scan_count"] == 2, (skipped, empty)
print(f"[U6] scan: partial carton refined at once (3 unreachable refine views -> view nearer the base complete), "
      f"height views in the same scan; repeat scan skips {skipped} empty grid views")
# grasp issue checks
gi = {"os": os, "math": math, "v326_wrist_selected": lambda: True, "v330_wrist_profile": lambda: "usb130"}
exec(block("    def _v339_rgb_only():", "    def _v326_wrist_capture(label, names=None):"), gi)
issue = gi["_v339_grasp_issue"]
assert gi["_v339_rgb_only"]()
good = {"seen": True, "trusted": True, "location": "table_or_pile", "xyz_m": [-0.43, -0.17, 0.0222],
        "this_view_xyz_m": [-0.432, -0.171, 0.0219], "measurement": {"method": "multi_view_triangulation"}}
assert issue(good, {"measured": {"table_z_m": None}}) is None
assert "not seen" in issue(dict(good, seen=False))
assert "not measured" in issue(dict(good, measurement={"method": "single_view_plane_assumption (not measured yet)"}))
assert "differ by 25 mm" in issue(dict(good, this_view_xyz_m=[-0.455, -0.17, 0.0219]))
hi = issue(dict(good, xyz_m=[-0.43, -0.17, 0.048], this_view_xyz_m=[-0.43, -0.17, 0.0219]))
assert "from table + half width" in hi and "+26 mm" in hi, hi
for mk in ("[V3.39 GRASP VIEW] {label}", 'distances=([0.36, 0.40] if k339 % 2 == 0 else [0.40, 0.45]), no_detour=True',
           '"reason": "GRASP_HEIGHT_IMPLAUSIBLE"', 'if why339g and ("from table + half width" in why339g):',
           '"scan_count": 0, "grid_hist": {}'):
    assert mk in B, mk
i_x = B.index("[V3.39 GRASP VIEW] {label}"); i_q = B.index('_v307_go_q0("V3.26 WRIST LOOK DONE")')
assert i_x < i_q                                                                # extra views BEFORE the q0 return
assert "GRASP_HEIGHT_IMPLAUSIBLE" in A
print("[U6] plan_grasp: target unseen / not measured / measured vs this view > 10 mm / lying-egg z off -> extra close "
      "views in place before q0; z still > 10 mm off -> GRASP_HEIGHT_IMPLAUSIBLE (no grasp 27 mm too high)")
print("ALL V3.39 TESTS PASSED")
