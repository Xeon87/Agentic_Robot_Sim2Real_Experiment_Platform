#!/usr/bin/env python3
"""Offline tests for V3.08 camera + perception (no MuJoCo / ROS / GPU).

A numpy ray-caster renders the egg scene with EXACTLY the pinhole model and
extrinsics the camera module reports (aligned RGB + z-depth), using the egg
poses/sizes from the 2026-10-21 19:26 run.  This checks the whole chain:
look-at -> intrinsics -> back-projection -> segmentation -> spheroid fit ->
matching -> files.  (MuJoCo's own render convention is checked at runtime by
the camera process SELF-CHECK.)

C1  camera geometry: look-at axes orthonormal, MuJoCo z = -forward, HFOV ~ 69 deg.
C2  project(backproject(depth)) round trip < 1e-6 px.
P1  4 eggs lying in the basket (clean z16 depth): all detected, xy < 3 mm, axis < 6 deg.
P2  same scene with D435-like noise (sigma = 0.0025 z^2): all detected, xy < 5 mm, axis < 15 deg.
P3  2 eggs in holder slots (vertical) + 2 in basket: locations classified correctly.
F1  CaptureWriter writes rgb/depth/overlay PNG, json files, errors.csv, summary.json.
M1  MJCF injection adds exactly one camera before </worldbody>.
X1  camera process loop (fake mujoco): 2 triggers -> 2 capture folders, summary, self-check PASS.
X2  OpenGL backend failure -> exit code 3 so the supervisor tries the next backend.
B1  backend wiring present; camera module never imports mujoco at module level.
R1  V3.09: 48 egg-coloured holder-post specks (22:14 run cause: 12-seed cap filled, 0/4) -> 4/4.
R2  V3.09: orange finger pad removed by the robot self-filter -> no false positive.
R3  V3.09: 25 mm match gate -> a 40 mm-off blob is a miss + FP, not a detection.
"""
import csv, json, math, os, sys, tempfile
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NERO_V308_WIDTH", "848")      # smaller than default for test speed; same FOV
os.environ.setdefault("NERO_V308_HEIGHT", "480")
import NERO_V308_Camera_Perception as CP

cfg = CP.camera_config(); K = CP.intrinsics(cfg); T = CP.extrinsics(cfg)

# ---------- C1 ----------
x, y, f = CP.look_at_axes(cfg["pos"], cfg["target"])
assert abs(np.dot(x, y)) < 1e-9 and abs(np.dot(x, f)) < 1e-9 and abs(np.linalg.norm(np.cross(x, y) + f)) < 1e-9
assert 66.0 < K["hfov_deg"] < 72.0 and abs(K["vfov_deg"] - 42.0) < 1e-9, K
print(f"[OK] C1 camera geometry | fx={K['fx']:.1f}px hfov={K['hfov_deg']:.1f} vfov={K['vfov_deg']:.1f}")


def quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def render(eggs, noise_k=0.0, seed=0):
    h, w = K["height"], K["width"]
    vv, uu = np.mgrid[0:h, 0:w]
    d_opt = np.stack([(uu - K["ppx"]) / K["fx"], (vv - K["ppy"]) / K["fy"], np.ones_like(uu, float)], -1).reshape(-1, 3)
    D = d_opt @ T[:3, :3].T           # world ray dirs, parametrised so t == optical z-depth
    O = T[:3, 3]
    tbest = np.full(len(D), np.inf); col = np.tile(np.array([205, 203, 196], np.uint8), (len(D), 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = (0.0 - O[2]) / D[:, 2]   # table z = 0
    hit = (tt > 0) & (np.abs(O[0] + tt * D[:, 0] + 0.385) < 0.335) & (np.abs(O[1] + tt * D[:, 1]) < 0.38)
    tbest[hit] = tt[hit]
    for e in eggs.values():
        R = quat_xyzw_to_R(e["quat_xyzw"]); s = np.asarray(e["size"])
        o = R.T @ (O - np.asarray(e["xyz_m"])) / s
        dd = (D @ R) / s
        A = np.sum(dd * dd, 1); B = 2 * dd @ o; C = o @ o - 1
        disc = B * B - 4 * A * C
        ok = disc > 0
        t1 = np.where(ok, (-B - np.sqrt(np.maximum(disc, 0))) / (2 * A), np.inf)
        better = ok & (t1 > 0) & (t1 < tbest)
        tbest[better] = t1[better]
        col[better] = np.array([214, 150, 88], np.uint8)
    depth = np.where(np.isfinite(tbest), tbest, 0.0).reshape(h, w)
    rgb = col.reshape(h, w, 3)
    rng = np.random.default_rng(seed)
    c = dict(cfg, depth_noise_k=noise_k)
    return rgb, CP._depth_postprocess(depth, c, rng)


# eggs from the 19:26 run (pose) + scene XML (semi-axes)
BASKET_EGGS = {
    "egg_1": {"xyz_m": [-0.45589, -0.22358, 0.02254], "quat_xyzw": [0.69017, -0.15385, -0.15393, 0.69014], "size": [0.0220, 0.0212, 0.0310]},
    "egg_2": {"xyz_m": [-0.38201, -0.14007, 0.02304], "quat_xyzw": [-0.6184, 0.34229, -0.35353, 0.61273], "size": [0.0228, 0.0217, 0.0320]},
    "egg_3": {"xyz_m": [-0.42858, -0.17333, 0.02224], "quat_xyzw": [0.55779, 0.43505, 0.42425, 0.56535], "size": [0.0217, 0.0209, 0.0305]},
    "egg_4": {"xyz_m": [-0.39566, -0.08532, 0.02324], "quat_xyzw": [-0.70363, -0.06996, 0.06972, 0.70367], "size": [0.0231, 0.0219, 0.0324]},
}
for e in BASKET_EGGS.values():
    n = np.linalg.norm(e["quat_xyzw"]); e["quat_xyzw"] = list(np.asarray(e["quat_xyzw"]) / n)


def truth_of(eggs):
    return {k: {"xyz_m": v["xyz_m"], "quat_xyzw": v["quat_xyzw"], "location": CP.classify_location(v["xyz_m"])} for k, v in eggs.items()}


# ---------- C2 ----------
rgb, depth = render(BASKET_EGGS)
pw, vv, uu = CP.backproject(depth, K, T, stride=7)
uv, z = CP.project(pw, K, T)
assert np.max(np.abs(uv[:, 0] - uu)) < 1e-6 and np.max(np.abs(uv[:, 1] - vv)) < 1e-6
print("[OK] C2 project/backproject round trip")


def run(eggs, noise_k, label, xy_lim, ax_lim):
    rgb, depth = render(eggs, noise_k)
    dets, diag = CP.perceive_eggs(rgb, depth, K, T)
    rows, fps, _ = CP.evaluate(dets, truth_of(eggs))
    s = CP.summarize(rows, fps)
    assert s["detected"] == len(eggs) and s["false_positives"] == 0, (label, s, diag)
    assert s["err_xy_mm"]["max"] < xy_lim and s["axis_err_deg"]["max"] < ax_lim, (label, s, rows)
    print(f"[OK] {label} detected {s['detected']}/{s['truth']} | xy mean {s['err_xy_mm']['mean']} max {s['err_xy_mm']['max']} mm"
          f" | z mean {s['err_z_mm']['mean']} mm | axis mean {s['axis_err_deg']['mean']} max {s['axis_err_deg']['max']} deg")
    return rgb, depth, dets, rows, fps, s, diag


P1 = run(BASKET_EGGS, 0.0, "P1 basket, clean depth:", 3.0, 6.0)
run(BASKET_EGGS, 0.0025, "P2 basket, D435-like noise:", 5.0, 15.0)

# ---------- P3 ----------
mixed = {k: dict(v) for k, v in BASKET_EGGS.items()}
up = list(np.array([1.0, 0, 0, 0]))   # 180 deg about x: long axis vertical
mixed["egg_1"] = {"xyz_m": [-0.510, 0.123, 0.0549], "quat_xyzw": [0.70711, 0.70711, 0.0, 0.0], "size": [0.0220, 0.0212, 0.0310]}
mixed["egg_2"] = {"xyz_m": [-0.450, 0.1224, 0.0575], "quat_xyzw": up, "size": [0.0228, 0.0217, 0.0320]}
rgb3, depth3 = render(mixed)
dets3, _ = CP.perceive_eggs(rgb3, depth3, K, T)
rows3, fps3, _ = CP.evaluate(dets3, truth_of(mixed))
locs = {r["truth_name"]: (r.get("det_location"), r["truth_location"]) for r in rows3}
assert all(r["detected"] and r["location_match"] for r in rows3) and not fps3, (locs, rows3)
print("[OK] P3 vertical eggs in slots + basket eggs, locations:", {k: v[0] for k, v in locs.items()})

# ---------- F1 ----------
with tempfile.TemporaryDirectory() as td:
    rgb, depth, dets, rows, fps, s, diag = P1
    wr = CP.CaptureWriter(td, cfg, K, T)
    cdir = wr.write(1, "plan_grasp:egg_3", 12.5, rgb, depth, np.zeros(70), dets, truth_of(BASKET_EGGS), rows, fps, s, diag, K, T, 0.3)
    for fn in ("rgb.png", "depth_mm.png", "overlay.png", "qpos.npy", "perceived_scene.json", "truth_scene.json", "errors.json"):
        assert (cdir / fn).stat().st_size > 0, fn
    assert (cdir / "rgb.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    rows_csv = list(csv.DictReader(open(wr.run_dir / "errors.csv")))
    assert len(rows_csv) == 4 and rows_csv[0]["tag"] == "plan_grasp:egg_3"
    summ = json.loads((wr.run_dir / "summary.json").read_text())
    assert summ["overall"]["detection_rate"] == 1.0
    info = json.loads((wr.run_dir / "camera_info.json").read_text())
    assert abs(info["fx"] - K["fx"]) < 1e-9 and len(info["T_world_camera_optical"]) == 4
    print("[OK] F1 files:", sorted(p.name for p in cdir.iterdir()), "+ errors.csv, summary.json, camera_info.json")

# ---------- M1 ----------
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "scene.xml"
    p.write_text("<mujoco><worldbody><body name='a'/></worldbody></mujoco>")
    out = Path(CP.inject_camera_into_mjcf(str(p), cfg)).read_text()
    assert out.count("<camera") == 1 and out.index("<camera") < out.index("</worldbody>") and 'name="v308_d435i"' in out
    assert CP.inject_camera_into_mjcf(str(Path(td) / "_nero_motor_v308_camera_tmp.xml"), cfg)
    print("[OK] M1 MJCF camera injection")

# ---------- X1 / X2: camera process loop with a fake mujoco (trigger -> capture -> files) ----------
import threading, types, time as _t, multiprocessing as _mp


def fake_mujoco(eggs, gl_fail=False):
    m = types.ModuleType("mujoco")
    class mjtObj:
        mjOBJ_BODY, mjOBJ_CAMERA = 1, 7
    m.mjtObj = mjtObj
    names = {f"obj_egg{i}": i - 1 for i in range(1, 5)}
    order = [eggs[f"egg_{i}"] for i in range(1, 5)]
    class Model:
        nq = 70
        vis = types.SimpleNamespace(global_=types.SimpleNamespace(offwidth=640, offheight=480))
    class Data:
        def __init__(self, model):
            self.qpos = np.zeros(70)
            self.xpos = np.array([e["xyz_m"] for e in order])
            self.xquat = np.array([[e["quat_xyzw"][3], *e["quat_xyzw"][:3]] for e in order])
    m.MjModel = types.SimpleNamespace(from_xml_path=lambda p: Model())
    m.MjData = Data
    m.mj_name2id = lambda model, t, n: (0 if n == CP.CAMERA_NAME else names.get(n, -1))
    m.mj_forward = lambda model, data: None
    rgb, depth = render(eggs)
    class Renderer:
        def __init__(self, model, height, width):
            if gl_fail:
                raise RuntimeError("EGL init failed (test)")
            self.d = False
        def enable_depth_rendering(self): self.d = True
        def disable_depth_rendering(self): self.d = False
        def update_scene(self, data, camera=None): assert camera == 0
        def render(self): return depth.astype(np.float32) if self.d else rgb
    m.Renderer = Renderer
    return m


def shared():
    q = _mp.Array("d", 1024); meta = _mp.Array("d", 4); mp_path = _mp.Array("c", 4096)
    b = b"/fake/scene.xml"; mp_path[:len(b)] = b; mp_path[len(b)] = b"\0"
    with q.get_lock():
        meta[0] = 70; meta[1] = 3.25; meta[3] = 1.0
    return q, meta, mp_path, _mp.Value("i", 0), _mp.Array("c", 256), _mp.Value("i", 0)


with tempfile.TemporaryDirectory() as td:
    sys.modules["mujoco"] = fake_mujoco(BASKET_EGGS)
    q, meta, path, seq, tag, status = shared()
    stop = threading.Event()
    th = threading.Thread(target=CP.camera_process_main, args=(q, meta, path, seq, tag, status, stop, td, "fake"))
    th.start()
    for _ in range(100):
        if status.value == 1: break
        _t.sleep(0.05)
    assert status.value == 1
    for t_ in ("observe_scene", "plan_grasp:egg_1"):
        b = t_.encode(); tag[:len(b)] = b; tag[len(b)] = b"\0"
        with seq.get_lock(): seq.value += 1
        run_dirs = list(Path(td).iterdir())
        for _ in range(200):
            caps = list(run_dirs[0].glob("capture_*")) if run_dirs else []
            if len([c for c in caps if (c / "errors.json").exists()]) >= (1 if t_ == "observe_scene" else 2): break
            _t.sleep(0.05)
    stop.set(); th.join(5)
    run = run_dirs[0]
    caps = sorted(p.name for p in run.glob("capture_*"))
    assert caps == ["capture_0001_observe_scene", "capture_0002_plan_grasp_egg_1"], caps
    summ = json.loads((run / "summary.json").read_text())
    assert summ["overall"]["detection_rate"] == 1.0 and summ["overall"]["captures"] == 2, summ
    print("[OK] X1 camera loop: 2 triggers -> 2 captures, detection 100%,", summ["overall"])

sys.modules["mujoco"] = fake_mujoco(BASKET_EGGS, gl_fail=True)
q, meta, path, seq, tag, status = shared()
try:
    CP.camera_process_main(q, meta, path, seq, tag, status, threading.Event(), tempfile.mkdtemp(), "egl")
    raise AssertionError("expected SystemExit")
except SystemExit as e:
    assert e.code == CP.EXIT_GL_FAILED and status.value == 2
print("[OK] X2 OpenGL failure -> exit code 3 (supervisor tries next backend)")
del sys.modules["mujoco"]

# ---------- B1: backend wiring markers ----------
SRC = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
for m_ in ("v308_qpos=None", "inject_camera_into_mjcf", "_v308_camera_supervisor", "v308_trigger(", "kwargs=v308_kwargs",
           'os.environ["MUJOCO_GL"] = gl'):
    assert m_ in SRC, m_
import re as _re
assert not _re.search(r"^(import|from) mujoco", (ROOT / "NERO_V308_Camera_Perception.py").read_text(), _re.M), \
    "camera module must not import mujoco at module level"
print("[OK] B1 backend wiring present; camera module has no top-level mujoco import")

# ---------- V3.09 regressions from the 2026-10-21 22:14 run ----------
# R1: holder-post specular specks (egg-coloured, top ~48 mm) crowded the 12-seed cap -> 0/4 eggs.
def add_post_specks(rgb, depth, n_px=3):
    rgb = rgb.copy(); depth = depth.copy()
    tops = []
    for (sx, sy) in CP.SLOTS.values():
        for k in range(6):
            a_ = 2 * math.pi * k / 6
            tops.append([sx + 0.027 * math.cos(a_), sy + 0.024 * math.sin(a_), 0.048])
    uv, z = CP.project(np.array(tops), K, T)
    for (u, v), zz in zip(uv, z):
        u, v = int(round(u)), int(round(v))
        for du in range(n_px):
            if 0 <= v < depth.shape[0] and 0 <= u + du < depth.shape[1]:
                rgb[v, u + du] = (199, 150, 90)
                depth[v, u + du] = round(float(zz), 3)
    return rgb, depth, len(tops)

rgbS, depthS = render(BASKET_EGGS)
rgbS, depthS, nposts = add_post_specks(rgbS, depthS)
detsS, diagS = CP.perceive_eggs(rgbS, depthS, K, T)
rowsS, fpsS, _ = CP.evaluate(detsS, truth_of(BASKET_EGGS))
sS = CP.summarize(rowsS, fpsS)
assert sS["detected"] == 4 and sS["false_positives"] == 0 and sS["err_xy_mm"]["max"] < 3.0, (sS, diagS)
print(f"[OK] R1 {nposts} holder-post specks (egg colour, 48 mm high) -> still 4/4, xy max {sS['err_xy_mm']['max']} mm"
      f" | blob filter {diagS['mask_px_raw']}->{diagS['mask_px_after_blob_filter']} px, low-support seeds {diagS['seeds_rejected_low_support']}")

# R2: orange tactile-sensor pads on the fingers were fitted as eggs; robot self-filter removes them.
pad = {"xyz_m": [-0.40, -0.03, 0.075], "quat_xyzw": [0, 0, 0, 1.0], "size": [0.004, 0.018, 0.024]}
scene = dict(BASKET_EGGS); scene["pad"] = pad
rgbP, depthP = render(scene)
box = (np.array(pad["xyz_m"]), np.eye(3), np.array(pad["size"]))
d_no, _ = CP.perceive_eggs(rgbP, depthP, K, T)
d_yes, diagP = CP.perceive_eggs(rgbP, depthP, K, T, robot_boxes=[box])
r_yes, fp_yes, _ = CP.evaluate(d_yes, truth_of(BASKET_EGGS))
assert len(fp_yes) == 0 and all(r["detected"] for r in r_yes) and diagP["points_removed_robot_self_filter"] > 0, diagP
print(f"[OK] R2 finger pad: robot self-filter removed {diagP['points_removed_robot_self_filter']} pts -> 4/4, 0 FP "
      f"(without filter: {len(d_no)} detections)")

# R3: evaluation gate 25 mm: a detection 40 mm off is a miss + false positive, not a 'detected' egg.
far = [dict(d, det_id="det_x", xyz_m=list(np.add(BASKET_EGGS["egg_1"]["xyz_m"], [0.0, 0.0, 0.040]))) for d in P1[2][:1]]
rws, fps_, _ = CP.evaluate(far, {"egg_1": truth_of(BASKET_EGGS)["egg_1"]})
assert not rws[0]["detected"] and len(fps_) == 1
print("[OK] R3 match gate 25 mm: 40 mm-off blob -> miss + FP")
