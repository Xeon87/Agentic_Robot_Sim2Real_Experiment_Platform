#!/usr/bin/env python3
"""V3.31 no priors for the wrist AND the fixed camera.

N1 wrist scan = uniform grid over the table workspace inside the arm's reach (both sides, no object coordinates)
N2 egg height in a slot from the holder MODEL + egg CLASS (no simulator value); narrow egg drops onto the well
N3 (numpy) fixed-camera process (fake MuJoCo): holder at a NON-registered pose found in depth, slots within 3 mm,
   eggs named from the pool in (x, y) order (simulator order is different -> no identity seeding), held/placed
   hints, unnamed eggs reported when the name pool is too small
N4 backend: no-prior mode also for the fixed camera; unknown layout at episode start (no zone, no slot, no pile
   in the posture checks); scan stops on stability only; fixed capture ingests holder/pile; supervisor starts the
   V3.31 process; placed height from the model; NERO_V331_FIXED_NO_PRIOR=0 -> V3.11 path
N5 NERO_V308_Camera_Perception.py unchanged; launcher compiles the new module
"""
import hashlib, json, math, os, sys, tempfile, textwrap, threading, types
import multiprocessing as _mp
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import NERO_V326_Wrist_Camera as W  # noqa: E402
import NERO_V331_Fixed_Camera as F  # noqa: E402

assert W.VERSION.startswith(("V3.31", "V3.32", "V3.33", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")) and F.VERSION == "V3.31"
B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()

# ------------------------------------------------------------------ N1
os.environ.pop("NERO_V329_SCAN_AZ_DEG", None)
sp = W.scan_points()
pts = [p for _n, p in sp]
assert len(pts) >= 6, sp
for x, y, _z in pts:
    r = math.hypot(x, y)
    assert 0.20 - 1e-9 <= r <= 0.62 + 1e-9 and -0.80 <= x <= -0.08 and abs(y) <= 0.42, (x, y)
assert any(y > 0.2 for _x, y, _z in pts) and any(y < -0.2 for _x, y, _z in pts)       # both sides of the table
assert len({round(x, 3) for x, _y, _z in pts}) >= 2                                   # more than one row
assert all(n.startswith("scan_") for n, _p in sp)
print(f"[N1] scan grid: {len(pts)} views over the workspace within reach "
      f"(rows x={sorted({round(p[0], 2) for p in pts})}), no object coordinates")

# ------------------------------------------------------------------ N2
z = W.egg_center_in_slot_z(0.0)
assert 0.050 < z < 0.058, z
assert abs(W.egg_center_in_slot_z(0.01) - z - 0.01) < 1e-12
zn = W.egg_center_in_slot_z(0.0, a=0.03, b=0.015)
assert abs(zn - (W.holder_model()["well_top"] + 0.03)) < 1e-12
print(f"[N2] egg centre in a slot from holder model + egg class: {z * 1000:.1f} mm above the table "
      f"(narrow egg -> on the well {zn * 1000:.1f} mm)")

# ------------------------------------------------------------------ N3
try:
    import numpy as np
    import NERO_V308_Camera_Perception as CP
    import v329_synth as S
    have_np = True
except Exception:
    have_np = False

if not have_np:
    print("[N3] numpy missing -> fixed process test skipped")
else:
    cfg = CP.camera_config(); K = CP.intrinsics(cfg); T = CP.extrinsics(cfg)
    HX, HY, HYAW = -0.36, 0.22, math.radians(20)                 # NOT the registered pose (-0.42, 0.155, 0)
    h, w = K["height"], K["width"]
    _vv, _uu = np.mgrid[0:h, 0:w]
    D = (np.stack([(_uu - K["ppx"]) / K["fx"], (_vv - K["ppy"]) / K["fy"], np.ones_like(_uu, float)], -1)
         .reshape(-1, 3)) @ T[:3, :3].T
    O = T[:3, 3]

    def qR(q):
        x, y, z_, w_ = q
        return np.array([[1-2*(y*y+z_*z_), 2*(x*y-z_*w_), 2*(x*z_+y*w_)], [2*(x*y+z_*w_), 1-2*(x*x+z_*z_), 2*(y*z_-x*w_)],
                         [2*(x*z_-y*w_), 2*(y*z_+x*w_), 1-2*(x*x+y*y)]])
    # simulator order deliberately differs from the (x, y) order perception uses for naming
    EGGS = {
        "egg_1": {"xyz_m": [-0.38201, -0.14007, 0.02304], "quat_xyzw": [-0.6184, 0.34229, -0.35353, 0.61273], "size": [0.0228, 0.0217, 0.0320]},
        "egg_2": {"xyz_m": [-0.45589, -0.22358, 0.02254], "quat_xyzw": [0.69017, -0.15385, -0.15393, 0.69014], "size": [0.0220, 0.0212, 0.0310]},
        "egg_3": {"xyz_m": [-0.39566, -0.08532, 0.02324], "quat_xyzw": [-0.70363, -0.06996, 0.06972, 0.70367], "size": [0.0231, 0.0219, 0.0324]},
        "egg_4": {"xyz_m": [-0.42858, -0.17333, 0.02224], "quat_xyzw": [0.55779, 0.43505, 0.42425, 0.56535], "size": [0.0217, 0.0209, 0.0305]},
    }
    for e in EGGS.values():
        e["quat_xyzw"] = list(np.asarray(e["quat_xyzw"]) / np.linalg.norm(e["quat_xyzw"]))

    def render(eggs):
        tbest = np.full(len(D), np.inf); col = np.tile(np.array([205, 203, 196], np.uint8), (len(D), 1))
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (0.0 - O[2]) / D[:, 2]
        hit = (tt > 0) & (np.abs(O[0] + tt * D[:, 0] + 0.385) < 0.335) & (np.abs(O[1] + tt * D[:, 1]) < 0.38)
        tbest[hit] = tt[hit]
        for e in eggs.values():
            R = qR(e["quat_xyzw"]); s = np.asarray(e["size"]); o = R.T @ (O - np.asarray(e["xyz_m"])) / s; dd = (D @ R) / s
            A = np.sum(dd * dd, 1); Bq = 2 * dd @ o; C = o @ o - 1; disc = Bq * Bq - 4 * A * C; ok = disc > 0
            t1 = np.where(ok, (-Bq - np.sqrt(np.maximum(disc, 0))) / (2 * A), np.inf)
            better = ok & (t1 > 0) & (t1 < tbest); tbest[better] = t1[better]; col[better] = np.array([214, 150, 88], np.uint8)
        dep = np.where(np.isfinite(tbest), tbest, 0.0).reshape(h, w)
        Dh = S.render_depth(S.holder_points(HX, HY, HYAW), K, T)          # holder in depth (grey like the table)
        m = (Dh > 0) & ((dep == 0) | (Dh < dep))
        dep[m] = Dh[m]
        return col.reshape(h, w, 3), dep.astype(np.float32)
    rgbF, depthF = render(EGGS)

    m = types.ModuleType("mujoco")

    class _mjtObj:
        mjOBJ_BODY, mjOBJ_CAMERA, mjOBJ_JOINT, mjOBJ_GEOM = 1, 7, 3, 5
    m.mjtObj = _mjtObj
    order = [EGGS[f"egg_{i}"] for i in range(1, 5)]

    class _Model:
        nq = 70
        ngeom = 0
        vis = types.SimpleNamespace(global_=types.SimpleNamespace(offwidth=640, offheight=480))

    class _Data:
        def __init__(self, model):
            self.qpos = np.zeros(70)
            self.xpos = np.array([e["xyz_m"] for e in order])
            self.xquat = np.array([[e["quat_xyzw"][3], *e["quat_xyzw"][:3]] for e in order])
    m.MjModel = types.SimpleNamespace(from_xml_path=lambda p: _Model())
    m.MjData = _Data
    m.mj_name2id = lambda model, t, n: (0 if n == CP.CAMERA_NAME else {f"obj_egg{i}": i - 1 for i in range(1, 5)}.get(n, -1))
    m.mj_forward = lambda model, data: None

    class _Renderer:
        def __init__(self, model, height, width): self.d = False
        def enable_depth_rendering(self): self.d = True
        def disable_depth_rendering(self): self.d = False
        def update_scene(self, data, camera=None): pass
        def render(self): return depthF.copy() if self.d else rgbF
        def close(self): pass
    m.Renderer = _Renderer
    sys.modules["mujoco"] = m

    def run_process(names, tags):
        os.environ["NERO_V330_OBJECT_NAMES"] = names
        sh = dict(qpos=_mp.Array("d", 1024), meta=_mp.Array("d", 4), model_path=_mp.Array("c", 4096),
                  trig_seq=_mp.Value("i", 0), trig_tag=_mp.Array("c", 256), status=_mp.Value("i", 0),
                  res_seq=_mp.Value("i", 0), res_buf=_mp.Array("c", 65536))
        bp = b"/fake/scene.xml"; sh["model_path"][:len(bp)] = bp; sh["model_path"][len(bp)] = b"\0"
        sh["meta"][0] = 70; sh["meta"][1] = 1.0; sh["meta"][3] = 1.0
        out = []
        with tempfile.TemporaryDirectory() as td:
            stop = threading.Event()
            th = threading.Thread(target=F.fixed_camera_process_main, args=(
                sh["qpos"], sh["meta"], sh["model_path"], sh["trig_seq"], sh["trig_tag"], sh["status"], stop, td,
                "fake", sh["res_seq"], sh["res_buf"]))
            th.start()
            import time
            t0 = time.time()
            while sh["status"].value != 1 and time.time() - t0 < 30:
                time.sleep(0.05)
            assert sh["status"].value == 1
            for i, tag in enumerate(tags, 1):
                b = tag.encode()[:255]
                sh["trig_tag"][:len(b)] = b; sh["trig_tag"][len(b)] = b"\0"
                sh["trig_seq"].value = i
                t0 = time.time()
                while sh["res_seq"].value != i and time.time() - t0 < 60:
                    time.sleep(0.05)
                raw = bytes(sh["res_buf"][:]).split(b"\0", 1)[0]
                out.append(json.loads(raw.decode()))
            stop.set(); th.join(10)
        return out

    r = run_process("egg_1,egg_2,egg_3,egg_4", ["observe_scene", "plan_grasp:egg_1|held=egg_1"])
    p1, p2 = r
    assert "error" not in p1, p1
    hol = p1["containers"]["egg_holder"]
    tr = S.slot_truth(HX, HY, HYAW)
    errs = [math.dist(hol["slots"][k]["xyz_m"][:2], tr[k][:2]) * 1000 for k in tr]
    assert hol["complete"] and max(errs) < 3.0 and abs(math.degrees(hol["yaw_rad"]) - 20) < 1.0, (hol, errs)
    by_xy = sorted(EGGS.values(), key=lambda e: (round(e["xyz_m"][0], 3), round(e["xyz_m"][1], 3)))
    for i, e in enumerate(by_xy, 1):
        o = p1["objects"][f"egg_{i}"]
        assert o["seen"] and o["trusted"] and math.dist(o["xyz_m"], e["xyz_m"]) < 0.004, (i, o, e["xyz_m"])
    assert sorted(p1["tracking"]["named"]) == ["egg_1", "egg_2", "egg_3", "egg_4"]
    mism = sum(1 for i in range(1, 5) if math.dist(p1["objects"][f"egg_{i}"]["xyz_m"], EGGS[f"egg_{i}"]["xyz_m"]) < 0.01)
    assert mism < 4                                    # names are perception's, not the simulator's
    assert p1["camera"] == "fixed_d435i" and p2["tracking"]["held_hint"] == "egg_1"
    r2 = run_process("egg_1,egg_2", ["observe_scene"])
    assert r2[0]["tracking"]["unnamed_trusted"] == 2 and sorted(r2[0]["tracking"]["named"]) == ["egg_1", "egg_2"]
    del sys.modules["mujoco"]
    os.environ.pop("NERO_V330_OBJECT_NAMES", None)
    print(f"[N3] fixed process: holder at ({HX}, {HY}) yaw 20 deg found in depth (slot err max {max(errs):.1f} mm), "
          f"eggs named in (x, y) order from the pool (no simulator seeding), held hint honoured, "
          f"2 eggs beyond a 2-name pool reported as unnamed")

# ------------------------------------------------------------------ N4
a = B.index("    _V329_TRUTH = {}")
b = B.index("    def _v326_wrist_capture(label, names=None):")
code = textwrap.dedent(B[a:b])
logs = []
sc = types.SimpleNamespace(emit=logs.append, object_pose=lambda n: ([0, 0, 0], [0, 0, 0, 1]))
slots_reg = {f"slot_{i + 1}": (x, y, 0.043) for i, (y, x) in enumerate(
    [(y, x) for y in (0.123, 0.187) for x in (-0.510, -0.450, -0.390, -0.330)])}
mode = {"wrist": False, "m": "camera", "fixed": True}
tmp = tempfile.mkdtemp(prefix="v331_")
env = {"os": os, "math": math, "json": json, "time": __import__("time"),
       "V271_SLOT_CENTERS_M": dict(slots_reg), "EGG_SLOT_POSES": dict(slots_reg),
       "EGG_BASKET": {"name": "egg_basket", "center_xy_m": [-0.425, -0.155], "radius_m": 0.088},
       "EGG_HOLDER": {"name": "egg_holder", "center_xyz_m": [-0.420, 0.155, 0.006]},
       "V27_BASKET_CENTER_XY_M": (-0.425, -0.155), "V271_HOLDER_RIM_TOP_Z_M": 0.044,
       "V271_SAFE_ORIENTATION_XY_M": (-0.4225, 0.0),
       "V329_LAYOUT_STATE": {"perceived_required": False, "holder_ok": False, "pile_ok": False},
       "V24_OBJECT_GEOMETRY": {f"egg_{i}": {"approx_size_mm": [44.0, 42.0, 62.0], "effective_width_mm": 44.0}
                               for i in range(1, 5)},
       "V329_SCENE_OVERRIDE_PATH": Path(tmp) / "logs" / "v329_moveit_holder.json",
       "v326_wrist_selected": lambda: mode["wrist"], "v311_mode": lambda: mode["m"],
       "v331_fixed_no_prior": lambda: mode["fixed"], "skill_core": sc,
       "hold_active_shared": types.SimpleNamespace(value=False), "v24_skill_state": {"active_object": None},
       "V311_STATE": {"last": None}, "V326_SHARED": {}, "EGG_OBJECTS": ["egg_1", "egg_2", "egg_3", "egg_4"],
       "_v328_tag_with_hints": lambda t: t, "_egg_slot_for_xyz": lambda xyz: None,
       "_v311_obj_entry": lambda n: None}
env.setdefault("V335_STATE", {"holder_hist": [], "holder_jump": [], "in_hand_bad": 0})   # V3.35
env.setdefault("_v335_capture", lambda tag: W.capture_sync({}, tag)); env.setdefault("_v335_holder_filter", lambda h, tag=None, cam=None: h)
exec(code, env)
assert env["_v329_no_prior"]()                                   # fixed camera, default
mode["fixed"] = False; assert not env["_v329_no_prior"](); mode["fixed"] = True
mode["m"] = "truth"; assert not env["_v329_no_prior"](); mode["m"] = "camera"
env["_v329_reset_scene"]("test")
assert env["V329_LAYOUT_STATE"] == {"perceived_required": True, "holder_ok": False, "pile_ok": False}
assert env["_v329_near_zone"](-0.425, -0.155, 0.05, 0.2) == (False, False)      # registered pile: not a zone now
assert env["_v329_near_zone"](-0.420, 0.155, 0.05, 0.2) == (False, False)       # registered holder: not a zone now
env["_V329_SCENE"].update(holder={"complete": True}, pile={"center_xy_m": [0, 0]}, views_without_new=0)
env["_V329_SCENE"]["seen"] = {"egg_1", "egg_2", "egg_3", "egg_4"}
assert not env["_v329_scene_ready"]()                            # all names used is NOT a stop reason any more
env["_V329_SCENE"]["views_without_new"] = 2
assert env["_v329_scene_ready"]()
env["_V329_SCENE"].update(holder=None, pile=None, views_without_new=1)
env["_v329_ingest"]({"tag": "t", "objects": {}, "tracking": {"unnamed_trusted": 1}})
assert env["_V329_SCENE"]["views_without_new"] == 0 and any("[V3.31 UNNAMED EGGS]" in l for l in logs)
zs = env["_v331_egg_in_slot_z"]((-0.5, 0.1, 0.043))
assert abs(zs - W.egg_center_in_slot_z(0.0)) < 1e-12
for mk in ('V329_LAYOUT_STATE.get("perceived_required") and not V329_LAYOUT_STATE.get("holder_ok"):\n            return None',
           'not V329_LAYOUT_STATE.get("pile_ok"):\n            return "table_or_transit"',
           "_unknown331 = bool(", "[V3.31 FIXED VIEW]", "payload = v311_capture_sync(_v328_tag_with_hints(label))",
           "target331 = _v331f.fixed_camera_process_main", "def v331_fixed_no_prior():",
           "_v311_capture(f\"plan_placement:{obj}/holder\", [obj])", "placed[str(obj)] = [float(c[0]), float(c[1]), _v331_egg_in_slot_z(c)]",
           "rel_xyz = [float(c[0]), float(c[1]), _v331_egg_in_slot_z(c)]",
           "_tloc311 = llm_classify_object_location(_v330_nearest_truth(xyz)[1] if _v329_no_prior() else"):
    assert mk in B, mk
assert 'len(_V329_SCENE["seen"]) >= len(EGG_OBJECTS)' not in B
assert "target=target331" in B and "payload = v311_capture_sync(label)\n        _v311_log_vs_truth" in B   # V3.11 path kept
print("[N4] backend: fixed camera in no-prior mode (switchable), layout unknown at episode start, scan stop on stability "
      "only, unnamed eggs reported, fixed capture ingests holder/pile with hints, V3.31 fixed process, placed height "
      "from the model")

# ------------------------------------------------------------------ N5
h8 = hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest()
assert h8.startswith("020cc6680f181872"), h8
L = (ROOT / "start_nero_sim.sh").read_text()
assert 'FIXED331="$HERE/NERO_V331_Fixed_Camera.py"' in L and ('"$WRIST" "$FIXED331" ||' in L or '"$WRIST" "$FIXED331" "$SCENES343" ||' in L)  # V3.43 adds the scene module
print("[N5] NERO_V308_Camera_Perception.py unchanged; launcher checks + compiles NERO_V331_Fixed_Camera.py")
print("ALL V3.31 TESTS PASS")
