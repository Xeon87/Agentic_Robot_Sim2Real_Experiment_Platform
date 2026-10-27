#!/usr/bin/env python3
"""V3.35: friction grasp, live perception, calibration error, measurement instead of priors, no truth fallback.

N1 calibration-error model: level 0 = identity; error grows with the level; believed K/T and residual lens distortion
N2 eggs TRIANGULATED from several RGB views (no plane / size prior in the result): position error < 3 mm at level 0,
   a small egg is measured small (size is a measurement, not the class prior); error grows with the calibration level
N3 carton top height MEASURED by multi-view parallax (table raised 30 mm -> measured, slots right afterwards)
N4 in-hand camera check: no shift -> 0 mm, a 30 mm slip in the fingers -> clearly reported
N5 the real wrist-camera process loop end-to-end with a stub MuJoCo renderer (synthetic images): calibration from the
   hint, measurement in the payload, live captures not written to disk, live_view.png/json labels = tracker names
N6 hints round trip (calib / live / held position)
N7 backend logic (AST-extracted, stubs): carton views fused, one outlier ignored, two consistent views = MOVED;
   pre-motion check refuses a grasp when the egg moved and a placement when the carton moved
N8 wiring: no truth fallback anywhere (wrist selected = camera; RUN TASK refused; tools stop), friction physics
   (elliptic cone, impratio, noslip, realistic mu, CONNECT assist off by default), grasp sensing, clamp waits for the
   force, GRASP_LOST, insertion stop, GUI slider + monitor, Agent plausibility rule
"""
import ast, json, math, os, sys, tempfile, textwrap, threading, time, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import NERO_V326_Wrist_Camera as W  # noqa: E402

assert W.VERSION in ("V3.35", "V3.35.1", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
A = (ROOT / "NERO_V2_7_1_Agent.py").read_text()
import numpy as np                       # noqa: E402
import NERO_V308_Camera_Perception as P8  # noqa: E402
import v329_synth as S                   # noqa: E402

cfg = W.wrist_config("usb130"); K = P8.intrinsics(cfg); pc, fc = W.camera_in_tcp(cfg)

# ------------------------------------------------------------------ N1
e0 = W.calib_error(0.0)
K0, T0 = W.calib_apply(K, np.eye(4), e0)
assert K0 == K and np.allclose(T0, np.eye(4))
e1, e3 = W.calib_error(1.0), W.calib_error(3.0)
assert abs(e3["focal_rel"]) > abs(e1["focal_rel"]) > 0.005 and abs(e3["t_m"][2]) > abs(e1["t_m"][2]) > 0.001
assert abs(W.calib_error(99.0)["level"] - W.V335_CALIB_MAX) < 1e-9
K1, T1 = W.calib_apply(K, np.eye(4), e1)
assert abs(K1["fx"] / K["fx"] - 1 - e1["focal_rel"]) < 1e-9 and np.linalg.norm(T1[:3, 3]) > 0.002
img = (np.indices((K["height"], K["width"])).sum(axis=0) % 251).astype(np.uint8)[..., None].repeat(3, 2)
assert np.array_equal(W.distort_residual(img, K, 0.0), img)
d1 = np.abs(W.distort_residual(img, K, 0.05).astype(int) - img.astype(int))
assert d1[0, 0].max() > 0 and d1[K["height"] // 2, K["width"] // 2].max() == 0        # corners move, centre does not
print(f"[N1] calibration error: level 1 = {W.calib_summary(e1)}; level 0 = exact; corners shift with k1")

# ------------------------------------------------------------------ N2
EGGS = [([-0.451, -0.181, 0.0219], 0.3, 0.0315, 0.0219), ([-0.380, -0.120, 0.0185], -0.5, 0.027, 0.0185)]
parts = [S.table_rgb_part()] + [S.egg_rgb_part(c, True, y, a, b) for c, y, a, b in EGGS]
VIEW_PTS = ([-0.44, -0.105, 0.02], [-0.44, -0.315, 0.02], [-0.42, -0.16, 0.03], [-0.20, -0.315, 0.02])


def measure(level):
    err = W.calib_error(level)
    M = W.EggMeasure()
    for k, pt in enumerate(VIEW_PTS):
        c = [x for x in W.look_pose_candidates(pt, cfg=cfg, distances=[0.45]) if abs(x["tilt_deg"]) == 30][0]
        T = S.camera_T_for(c, pc, fc)
        rgb = W.distort_residual(S.render_rgb(parts, K, T, seed=k), K, err["k1"])
        Kb, Tb = W.calib_apply(K, T, err)
        for d in W.perceive_eggs_rgb(rgb, Kb, Tb):
            i = min(range(len(EGGS)), key=lambda i: math.dist(d["xyz_m"][:2], EGGS[i][0][:2]))
            if math.dist(d["xyz_m"][:2], EGGS[i][0][:2]) < 0.05:
                M.add(f"e{i}", d, Kb, Tb, k)
    measure.M = M
    return [M.get(f"e{i}") for i in range(len(EGGS))]


r0 = measure(0.0)
M0 = measure.M
for (c, _y, a, b), e in zip(EGGS, r0):
    assert e and e["ok"], e
    assert math.dist(e["xyz_m"], c) < 0.003, (e, c)                      # incl. HEIGHT: no table/egg-size plane
    assert abs(e["width_m"] - 2 * b) / (2 * b) < 0.15 and abs(e["length_m"] - 2 * a) / (2 * a) < 0.15, e
assert r0[1]["width_m"] < r0[0]["width_m"] - 0.004                       # the small egg is measured small
# an egg moved by someone: 1st disagreeing view -> "possibly moved" (untrusted, this view's position), 2nd -> new
moved = [([EGGS[0][0][0] + 0.045, EGGS[0][0][1], EGGS[0][0][2]],) + EGGS[0][1:], EGGS[1]]
parts_m = [S.table_rgb_part()] + [S.egg_rgb_part(c, True, y, a, b) for c, y, a, b in moved]
states = []
for k, pt in enumerate((VIEW_PTS[0], VIEW_PTS[2])):
    c = [x for x in W.look_pose_candidates(pt, cfg=cfg, distances=[0.45]) if abs(x["tilt_deg"]) == 30][0]
    T = S.camera_T_for(c, pc, fc)
    dets = W.perceive_eggs_rgb(S.render_rgb(parts_m, K, T, seed=10 + k), K, T)
    d0 = min(dets, key=lambda d: math.dist(d["xyz_m"][:2], moved[0][0][:2]))
    trk = {"e0": {"seen": True, "trusted": True, "det_id": d0["det_id"], "xyz_m": list(d0["xyz_m"])}}
    W.apply_measurements(trk, dets, M0, K, T, 20 + k)
    states.append(trk["e0"])
assert not states[0]["trusted"] and "moved" in states[0]["measurement"]["method"]
assert math.dist(states[0]["xyz_m"][:2], moved[0][0][:2]) < 0.01            # reports where it is NOW
assert not states[1]["trusted"] and len(M0.obs["e0"]) == 1                   # new measurement started
errs = {lv: max(math.dist(e["xyz_m"], c[0]) for e, c in zip(measure(lv), EGGS)) for lv in (0.3, 1.0, 3.0)}
assert errs[0.3] < 0.008 and errs[0.3] < errs[1.0] < errs[3.0], errs
print("[N2] moved egg: 1st disagreeing view -> 'possibly moved' at its new place (untrusted), 2nd -> re-measured")
print(f"[N2] triangulated eggs: err {max(math.dist(e['xyz_m'], c[0]) for e, c in zip(r0, EGGS)) * 1000:.1f} mm (z incl.) | "
      f"measured widths {r0[0]['width_m'] * 1000:.1f} / {r0[1]['width_m'] * 1000:.1f} mm (true 43.8 / 37.0) | calib level "
      + ", ".join(f"{lv}: {v * 1000:.0f} mm" for lv, v in errs.items()))

# ------------------------------------------------------------------ N3
hx, hy = -0.42, 0.155
Pc, Cc = S.carton_rgb_part(hx, hy, 0.0)
Pt, Ct = S.table_rgb_part()
dz = 0.030                                                               # table 30 mm higher than the model assumes
parts3 = [(Pt + [0, 0, dz], Ct), (Pc + [0, 0, dz], Cc)]
tr = S.slot_truth(hx, hy, 0.0)
H = W.HolderHeight()
errs3 = []
for pt in ([hx, hy, 0.043], [-0.44, 0.105, 0.02], [-0.44, 0.315, 0.02], [hx, hy, 0.043]):
    c = [x for x in W.look_pose_candidates(pt, cfg=cfg, distances=[0.45]) if abs(x["tilt_deg"]) == 30][0]
    T = S.camera_T_for(c, pc, fc)
    h = W.detect_holder_rgb(S.render_rgb(parts3, K, T), K, T, z_top=H.top_z())
    assert h is not None
    H.add(h.pop("_px_uv"), K, T)
    if h["complete"]:
        errs3.append(max(math.dist(h["slots"][k]["xyz_m"][:2], tr[k][:2]) for k in tr))
assert H.top_z() is not None and abs(H.top_z() - (0.048 + dz)) < 0.004, H.est
assert errs3[0] > 0.015 and errs3[-1] < 0.004, errs3                     # model plane wrong -> measured plane right
print(f"[N3] carton top measured by parallax: {H.top_z() * 1000:.1f} mm (true {(0.048 + dz) * 1000:.0f}); slot error "
      f"{errs3[0] * 1000:.0f} mm with the model height -> {errs3[-1] * 1000:.1f} mm with the measured one")

# ------------------------------------------------------------------ N4
c4 = W.look_pose_candidates([-0.40, 0.0, 0.15], cfg=cfg, distances=[0.36])[0]
T4 = S.camera_T_for(c4, pc, fc)
R4 = np.array(W.matrix_from_quat(c4["quat_xyzw"])); tcp4 = np.array(c4["xyz"])
ih = W.InHandCheck()
res4 = []
for off in (0.0, 0.0, 0.030):
    egg = tcp4 + R4 @ np.array([off, 0.0, 0.0])
    res4.append(ih.check(S.render_rgb([S.table_rgb_part(), S.egg_rgb_part(list(egg), True, 0.0)], K, T4), K, T4,
                         "egg_1", list(tcp4), None))
assert all(r["visible"] for r in res4) and res4[1]["shift_mm"] < 1.0 and res4[2]["shift_mm"] > 12.0, res4
assert ih.check(None, K, T4, None, None, None) is None and not ih.ref                  # released -> reference cleared
print(f"[N4] in-hand camera check: still {res4[1]['shift_mm']} mm, 30 mm slip -> {res4[2]['shift_mm']} mm reported")

# ------------------------------------------------------------------ N5 camera process end-to-end (stub MuJoCo)
SCENE = [S.table_rgb_part(), S.carton_rgb_part(hx, hy, 0.0)] + [S.egg_rgb_part(c, True, y, a, b) for c, y, a, b in EGGS]
POSES = []
for pt in ([-0.44, -0.105, 0.02], [-0.44, -0.315, 0.02], [-0.42, -0.16, 0.03], [hx, hy, 0.043], [-0.44, 0.105, 0.02]):
    POSES.append(S.camera_T_for([x for x in W.look_pose_candidates(pt, cfg=cfg, distances=[0.45])
                                 if abs(x["tilt_deg"]) == 30][0], pc, fc))


class _Arr:
    def __init__(self, n, typ="d"):
        import multiprocessing as mp
        self.a = mp.Array(typ, n)

    def __getattr__(self, k):
        return getattr(self.a, k)

    def __getitem__(self, k):
        return self.a[k]

    def __setitem__(self, k, v):
        self.a[k] = v


fake = types.ModuleType("mujoco")
fake.mjtObj = types.SimpleNamespace(mjOBJ_CAMERA=1, mjOBJ_BODY=2, mjOBJ_JOINT=3)


class _Model:
    nq = 4
    vis = types.SimpleNamespace(global_=types.SimpleNamespace(offwidth=640, offheight=480))

    @staticmethod
    def from_xml_path(p):
        return _Model()


class _Data:
    def __init__(self, m):
        self.qpos = np.zeros(4)
        self.cam_xpos = np.zeros((1, 3)); self.cam_xmat = np.zeros((1, 9))


def _fwd(m, d):
    T = POSES[int(round(d.qpos[0]))]
    R = T[:3, :3]
    d.cam_xpos[0] = T[:3, 3]
    d.cam_xmat[0] = np.stack([R[:, 0], -R[:, 1], -R[:, 2]], axis=1).ravel()


class _Renderer:
    def __init__(self, model, height, width):
        self.d = None

    def update_scene(self, d, camera=None):
        self.d = d

    def render(self):
        return S.render_rgb(SCENE, K, W.optical_from_mujoco(self.d.cam_xpos[0], self.d.cam_xmat[0]))

    def enable_depth_rendering(self):
        raise AssertionError("USB camera must not render depth")

    def disable_depth_rendering(self):
        pass

    def close(self):
        pass


fake.MjModel, fake.MjData, fake.Renderer, fake.mj_forward = _Model, _Data, _Renderer, _fwd
fake.mj_name2id = lambda m, t, n: 0 if n == W.USB_CAMERA_NAME else -1
sys.modules["mujoco"] = fake
P8._truth_from_data = lambda mj, m, d: {f"egg_{i + 1}": {"xyz_m": list(c), "quat_xyzw": [0, 0, 0, 1], "location": "table"}
                                       for i, (c, _y, _a, _b) in enumerate(EGGS)}
P8._robot_geom_ids = lambda mj, m: []
P8._robot_boxes = lambda m, d, g: []
with tempfile.TemporaryDirectory() as td:
    import multiprocessing as mp
    qpos, meta, mpath = mp.Array("d", 1024), mp.Array("d", 4), mp.Array("c", 4096)
    b = b"/fake/model.xml"; mpath[:len(b)] = b; meta[0] = 4; meta[3] = 1.0
    trig_seq, trig_tag, status = mp.Value("i", 0), mp.Array("c", 256), mp.Value("i", 0)
    res_seq, res_buf = mp.Value("i", 0), mp.Array("c", 65536)
    stop = threading.Event()
    th = threading.Thread(target=W.wrist_camera_process_main, args=(qpos, meta, mpath, trig_seq, trig_tag, status, stop,
                                                                     td, "egl", res_seq, res_buf), daemon=True)
    th.start()
    shared = {"status": status, "trig_seq": trig_seq, "trig_tag": trig_tag, "res_seq": res_seq, "res_buf": res_buf}
    pays = []
    for k in range(len(POSES)):
        with qpos.get_lock():
            qpos[0] = float(k)
        live = k >= 3
        pays.append(W.capture_sync(shared, W.encode_hints(f"view{k}", cam="usb130", calib=0.3, live=live), timeout_s=60))
    stop.set(); th.join(10)
    for p in pays:
        assert not p.get("error"), p
    assert all(p["calib"]["level"] == 0.3 for p in pays)
    named = {n: o for n, o in pays[2]["objects"].items() if o.get("seen")}
    assert named and all((o.get("measurement") or {}).get("method") == "multi_view_triangulation" for o in named.values()), named
    for o in named.values():
        assert min(math.dist(o["xyz_m"], c[0]) for c in EGGS) < 0.010, o["xyz_m"]
    first = {n: o for n, o in pays[0]["objects"].items() if o.get("seen")}
    assert all(not o["trusted"] for o in first.values())                    # 1 view: not triangulated -> not trusted
    assert pays[0]["dir"] and pays[3]["dir"] is None and pays[4]["dir"] is None      # live captures not saved
    live_dir = Path(td) / "live"
    lv = json.loads((live_dir / "live_view.json").read_text())
    assert (live_dir / "live_view.png").stat().st_size > 1000 and lv["capture"] == pays[-1]["capture"]
    names_lbl = {l["name"] for l in lv["labels"] if l["kind"] == "egg" and l.get("name")}
    seen_last = {n for n, o in pays[-1]["objects"].items() if o.get("seen")}
    assert names_lbl <= set(pays[-1]["objects"]) and names_lbl == seen_last, (names_lbl, seen_last)
    assert any(l["kind"] == "holder" for l in lv["labels"]) and sum(l["kind"] == "slot" for l in lv["labels"]) == 8
    hh = [p["containers"]["egg_holder"] for p in pays if p["containers"]["egg_holder"]]
    assert hh and "slot_layout_source" in hh[-1] and "_px_uv" not in hh[-1]
print(f"[N5] wrist process loop (stub MuJoCo): {len(pays)} captures, eggs {sorted(named)} triangulated in the payload, "
      f"1-view eggs untrusted, live captures not written, monitor labels = tracker names {sorted(names_lbl)}")
del sys.modules["mujoco"]

# ------------------------------------------------------------------ N6
t, hn = W.decode_hints(W.encode_hints("x", held="egg_2", cam="usb130", calib=1.25, heldp=[-0.4, 0.1, 0.2], live=True,
                                      placed={"egg_1": [-0.51, 0.123, 0.054]}))
assert t == "x" and hn["held"] == "egg_2" and hn["calib"] == 1.25 and hn["live"] and hn["heldp"] == [-0.4, 0.1, 0.2]
assert hn["placed"]["egg_1"] == [-0.51, 0.123, 0.054] and hn["cam"] == "usb130"
print("[N6] hints: calib level / live / held position / placed round trip")

# ------------------------------------------------------------------ N7
tree = ast.parse(B)
fns = {}
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in ("_v335_holder_filter", "_v335_precheck"):
        fns[node.name] = ast.get_source_segment(B, node)
logs = []
env = {"os": os, "math": math, "time": time, "V335_STATE": {"holder_hist": [], "holder_jump": []},
       "skill_core": types.SimpleNamespace(emit=logs.append)}
exec(textwrap.dedent(fns["_v335_holder_filter"]), env)
F = env["_v335_holder_filter"]


def hd(x, y, yaw=0.0):
    return {"center_xy_m": [x, y], "yaw_rad": yaw, "complete": True, "top_z_source": "measured_multi_view_parallax",
            "slots": {"slot_1": {"xyz_m": [x - 0.09, y - 0.03, 0.043]}}}
F(hd(-0.420, 0.155), "observe_scene/holder", [-0.3, 0.1, 0.4]); out = F(hd(-0.424, 0.151), "observe_scene/scan_2", [-0.1, 0.2, 0.4])
assert out["fused_views"] == 2 and abs(out["center_xy_m"][0] + 0.422) < 1e-6
out = F(hd(-0.300, 0.155), "live", [-0.1, 0.2, 0.4])                    # a live view far away: only a suspicion
assert abs(out["center_xy_m"][0] + 0.422) < 1e-6 and "HOLDER SUSPECT" in logs[-1] and env["V335_STATE"]["holder_suspect"]
out = F(hd(-0.302, 0.156), "live", [-0.1, 0.201, 0.4])                  # same camera position again: still not believed
assert abs(out["center_xy_m"][0] + 0.422) < 1e-6
out = F(hd(-0.301, 0.155), "plan_placement:egg_1/holder", [-0.35, 0.05, 0.36])   # dedicated look elsewhere -> moved
assert abs(out["center_xy_m"][0] + 0.301) < 0.002 and "HOLDER MOVED" in logs[-1] and not env["V335_STATE"]["holder_suspect"]
env2 = dict(env, v326_wrist_selected=lambda: True, _v329_no_prior=lambda: True, v24_skill_state={"active_object": "egg_1"},
            V27_GRASP_RUNTIME={"plans": {"egg_1": {"object_xyz_at_plan_m": [-0.45, -0.18, 0.022]}}},
            V271_PLACEMENT_RUNTIME={"plans": {"egg_1": {"slot_name": "slot_2", "slot_center_xyz_m": [-0.45, 0.123, 0.043]}}},
            EGG_SLOT_POSES={"slot_2": (-0.45, 0.123, 0.043)}, _v335_ingest_live=lambda p: None,
            _v335_inhand_verdict=lambda: {})
env2["_v336_perceived_slot"] = lambda sl: env2["EGG_SLOT_POSES"].get(sl)
cap = {"p": {"objects": {"egg_1": {"seen": True, "xyz_m": [-0.45, -0.18, 0.022]}}}}
env2["_v335_capture"] = lambda tag: cap["p"]
env2.setdefault("_v338_refine_slot", lambda *a, **k: None)   # V3.38 slot refine (tested in test_v3_38)
exec(textwrap.dedent(fns["_v335_precheck"]), env2)
PC = env2["_v335_precheck"]
assert PC("move_to_pregrasp", {"object_name": "egg_1"}) is None
cap["p"] = {"objects": {"egg_1": {"seen": True, "xyz_m": [-0.40, -0.18, 0.022]}}}
r7 = PC("approach_planned_grasp", {"object_name": "egg_1"})
assert r7["reason"] == "TARGET_MOVED_SINCE_PLAN" and r7["recommended_next_actions"] == ["plan_grasp"]
cap["p"] = {"objects": {"egg_1": {"seen": False, "xyz_m": [-0.40, -0.18, 0.022]}}}
assert PC("approach_planned_grasp", {"object_name": "egg_1"}) is None                   # not in view: no verdict
assert PC("move_to_slot_entry", {"object_name": "egg_1"}) is None
env2["EGG_SLOT_POSES"]["slot_2"] = (-0.41, 0.123, 0.043)
r7b = PC("guarded_insert_and_release", {"object_name": "egg_1"})
assert r7b["reason"] == "SLOT_MOVED_SINCE_PLAN" and r7b["recommended_next_actions"] == ["plan_placement"]
assert PC("lift", {}) is None
print("[N7] carton views fused / single outlier ignored / 2 consistent views = MOVED; pre-motion check refuses a moved "
      "egg (re-plan grasp) and a moved carton (re-plan placement), no verdict when the target is out of view")

# ------------------------------------------------------------------ N8
for mk in ('return V311_STATE.get("mode") in ("wrist", "wrist_rgb")\n',
           'return "camera"                    # V3.35: never "truth" when a wrist camera is selected',
           "[V3.35 RUN TASK REFUSED]", "PERCEPTION_UNAVAILABLE: the wrist camera process is not running",
           '"reason": "PERCEPTION_UNAVAILABLE"', "llm_agent.request_stop()",
           "reason335 = (_v335_slot_moved_now(obj, slot_name, slot)", "[V3.35 INSERT STOP]",
           'v24_skill_state["stage"] = V24_STAGE_LIFTED', "Camera calib error:", "ttk.Scale(v311_frame",
           "Wrist camera monitor (V3.35)", "live_view.png", "def _v335_live_loop():",
           "_v335_holder_filter(h, payload.get(\"tag\"), payload.get(\"cam_pos\"))",
           "calib=V335_STATE.get(\"calib_level\"), heldp=heldp, live=live", "_v335_measured_geometry()",
           '"measurement"] = o["measurement"]', "camera_wrist_usb130_rgb_perception", "NERO_V335_REQUIRE_HOLDER_HEIGHT"):
    assert mk in B, mk
assert "\"camera_d435i_perception\"" not in B and "skills use JSON ground truth\")" not in B
assert B.count("_v335_capture(") >= 4 and "_v326w.capture_sync(V326_SHARED, _v328_tag_with_hints(f\"{label}" not in B
assert "V3.35 MEASURED, LIVE SCENE" in A and "38-50 mm wide" in A and "SLOT_MOVED_SINCE_PLAN" in A
print("[N8] no truth fallback (wrist = camera; RUN TASK refused; tools stop); insertion stop; GUI slider + monitor; "
      "Agent plausibility (V3.36: friction physics reverted to V3.34 - see test_v3_36)")

# ------------------------------------------------------------------ N9 GUI snippets executed with a recording Tk stub
from unittest import mock
a9 = B.index("    # V3.35 CAMERA CALIBRATION ERROR (continuous)")
b9 = B.index("    root.after(1500, _v311_tick)\n", a9)
g1 = textwrap.dedent(B[a9:b9])
a9 = B.index("    # ----- V3.35 WRIST CAMERA MONITOR")
b9 = B.index("    # ----- Manual V2 tool debugger")
g2 = textwrap.dedent(B[a9:b9])
with tempfile.TemporaryDirectory() as td9:
    lab9 = {"egg_1": {"seen": True, "trusted": True, "det_id": 0, "measurement": {"width_mm": 44.0, "length_mm": 61.0}}}
    W.live_view_write(Path(td9), np.zeros((720, 1280, 3), np.uint8), K, np.eye(4), lab9,
                      [{"det_id": 0, "px_bbox": [10, 10, 60, 50]}, {"det_id": 1, "px_bbox": [100, 10, 160, 50], "trusted": True}],
                      None, 7, "t", extra={"camera": "usb130", "calib": "x", "calib_level": 0.3, "in_hand": {"name": None}})
    drawn = []
    canvas = mock.MagicMock()
    canvas.create_text.side_effect = lambda *a, **k: drawn.append(k.get("text"))
    tkm = mock.MagicMock()
    tkm.Canvas.return_value = canvas
    tkm.StringVar.side_effect = lambda value="": mock.MagicMock(get=mock.MagicMock(return_value=value))
    tkm.DoubleVar.side_effect = lambda value=0.0: mock.MagicMock(get=mock.MagicMock(return_value=value))
    tkm.BooleanVar.side_effect = lambda value=False: mock.MagicMock(get=mock.MagicMock(return_value=value))
    V9 = {"calib_level": 0.3, "live_on": True, "live_hz": 1.5, "tool_running": False}
    env9 = {"tk": tkm, "ttk": mock.MagicMock(), "os": os, "json": json, "time": time, "V335_STATE": V9,
            "root": mock.MagicMock(), "llm_outer": mock.MagicMock(), "v311_frame": mock.MagicMock(),
            "v311_info_var": mock.MagicMock(), "_v335_drain": lambda: None, "v335_live_dir": lambda: Path(td9)}
    exec(g1, env9)
    env9["v335_calib_var"].get.return_value = 1.0
    env9["_v335_on_calib"]()
    assert V9["calib_level"] == 1.0 and "hand-eye" in str(env9["v335_calib_txt"].set.call_args)
    exec(g2, env9)
    env9["_v335_tick"]()
    assert "egg_1 44x61mm" in drawn and any("awaiting 2nd view" in str(t) for t in drawn), drawn
    assert "capture #7" in str(env9["v335_mon_info"].set.call_args)
print("[N9] GUI: calibration slider updates the level + believed-vs-real text; monitor draws the perception labels")

print("ALL V3.35 TESTS PASS")
