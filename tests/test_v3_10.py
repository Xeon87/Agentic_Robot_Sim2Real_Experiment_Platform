#!/usr/bin/env python3
"""Offline tests for V3.10 perception robustness (no MuJoCo / ROS / GPU).

Reproduces the V3.09 noisy-depth failure (2026-10-22 run, k = 0.0025): touching lying eggs + depth noise
-> an extra seed on one END of an egg -> two overlapping fits -> V3.09 kept the LOWER-RMS one, which was
the half-egg fragment (750-1200 pts, 10-13 mm / 40-83 deg off).  Offline Monte-Carlo, same scene family,
60 scenes at 1280x720, k = 0.0025:
    V3.09: xy max 13.9 mm, axis max 87 deg, p95 axis 66 deg, 3 FP
    V3.10: trusted xy max 1.9 mm, axis max 8.7 deg, 0 FP (1 frame); 1.7 mm / 6.4 deg with 5 fused frames

T1  temporal_fuse: 5 frames -> depth noise x ~0.45, invalid pixels handled.
T2  view check: true pose consistent; a 12 mm-shifted pose is caught (see-through); occluded pose -> low coverage.
T3  Monte-Carlo (touching basket eggs, holder eggs, gripper occluder; noise k = 0.0025):
    trusted detections 0 FP, xy max < 3.5 mm, axis max < 15 deg.
T4  a mostly hidden egg is reported but NOT trusted.
T5  warm start from the previous capture adds no duplicates and keeps accuracy.
T6  resting-egg prior: lying eggs come out horizontal; eggs in holder slots are not forced horizontal.
T7  files: errors.csv has confidence/trusted/coverage, summary.json has overall_trusted, perceived_scene has quality.
T8  camera loop with noise: 5 depth frames fused, warm start used on the 2nd capture.
T9  backend: TCP settle wait before set_gripper_opening and after the V3.07 q0 move.
"""
import csv, json, math, os, sys, tempfile, threading, types, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NERO_V308_WIDTH", "848")      # test speed; same FOV as 1280x720
os.environ.setdefault("NERO_V308_HEIGHT", "480")
import NERO_V308_Camera_Perception as CP

cfg = CP.camera_config(); K = CP.intrinsics(cfg); T = CP.extrinsics(cfg)
assert CP.VERSION in ("V3.10", "V3.11")
h, w = K["height"], K["width"]
_vv, _uu = np.mgrid[0:h, 0:w]
D = (np.stack([(_uu - K["ppx"]) / K["fx"], (_vv - K["ppy"]) / K["fy"], np.ones_like(_uu, float)], -1).reshape(-1, 3)) @ T[:3, :3].T
O = T[:3, 3]


def quat_xyzw_to_R(q):
    x, y, z, w_ = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w_), 2*(x*z+y*w_)],
                     [2*(x*y+z*w_), 1-2*(x*x+z*z), 2*(y*z-x*w_)],
                     [2*(x*z-y*w_), 2*(y*z+x*w_), 1-2*(x*x+y*y)]])


def quat_lying(yaw, pitch=0.0):
    return CP._quat_from_z_to(np.array([math.cos(pitch)*math.cos(yaw), math.cos(pitch)*math.sin(yaw), math.sin(pitch)]))


def render_clean(eggs, boxes=()):
    tbest = np.full(len(D), np.inf); col = np.tile(np.array([205, 203, 196], np.uint8), (len(D), 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        tt = (0.0 - O[2]) / D[:, 2]
    hit = (tt > 0) & (np.abs(O[0] + tt * D[:, 0] + 0.385) < 0.335) & (np.abs(O[1] + tt * D[:, 1]) < 0.38)
    tbest[hit] = tt[hit]
    for e in eggs.values():
        R = quat_xyzw_to_R(e["quat_xyzw"]); s = np.asarray(e["size"])
        o = R.T @ (O - np.asarray(e["xyz_m"])) / s
        dd = (D @ R) / s
        A = np.sum(dd * dd, 1); B = 2 * dd @ o; C = o @ o - 1
        disc = B * B - 4 * A * C; ok = disc > 0
        t1 = np.where(ok, (-B - np.sqrt(np.maximum(disc, 0))) / (2 * A), np.inf)
        better = ok & (t1 > 0) & (t1 < tbest)
        tbest[better] = t1[better]; col[better] = np.array([214, 150, 88], np.uint8)
    for (c, half, color) in boxes:
        c = np.asarray(c); half = np.asarray(half)
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = (c - half - O) / D; t1 = (c + half - O) / D
        tmin = np.nanmax(np.minimum(t0, t1), axis=1); tmax = np.nanmin(np.maximum(t0, t1), axis=1)
        better = (tmax >= tmin) & (tmin > 0) & (tmin < tbest)
        tbest[better] = tmin[better]; col[better] = np.array(color, np.uint8)
    return col.reshape(h, w, 3), np.where(np.isfinite(tbest), tbest, 0.0).reshape(h, w).astype(np.float32)


SIZES = [[0.0220, 0.0212, 0.0310], [0.0228, 0.0217, 0.0320], [0.0217, 0.0209, 0.0305], [0.0231, 0.0219, 0.0324]]


def random_scene(rng, kind):
    eggs = {}
    if kind == "basket_cluster":                     # 4 touching eggs lying in the basket
        cx, cy = -0.425 + rng.uniform(-0.02, 0.02), -0.155 + rng.uniform(-0.02, 0.02)
        placed = []
        for i in range(4):
            for _ in range(500):
                yaw = rng.uniform(0, math.pi)
                p = np.array([cx + rng.uniform(-0.055, 0.055), cy + rng.uniform(-0.055, 0.055)])
                if all(np.linalg.norm(p - q) > 0.046 for q in placed) and np.linalg.norm(p - [cx, cy]) < 0.07:
                    placed.append(p); break
            s = SIZES[i]
            eggs[f"egg_{i+1}"] = {"xyz_m": [float(p[0]), float(p[1]), s[0] + 0.0005],
                                  "quat_xyzw": quat_lying(yaw, rng.uniform(-0.08, 0.08)), "size": s}
    else:                                            # 2 eggs in holder slots + 2 touching in the basket
        slots = list(CP.SLOTS.values())
        for i in range(4):
            s = SIZES[i]
            if i < 2:
                sx, sy = slots[i]
                eggs[f"egg_{i+1}"] = {"xyz_m": [sx + rng.uniform(-0.003, 0.003), sy + rng.uniform(-0.003, 0.003), 0.045 + rng.uniform(-0.004, 0.004)],
                                      "quat_xyzw": quat_lying(rng.uniform(-0.2, 0.2), rng.uniform(-0.15, 0.15)), "size": s}
            else:
                p = [-0.425 + 0.024 * (1 if i == 2 else -1) + rng.uniform(-0.004, 0.004), -0.155 + rng.uniform(-0.01, 0.01)]
                eggs[f"egg_{i+1}"] = {"xyz_m": [p[0], p[1], s[0] + 0.0005], "quat_xyzw": quat_lying(rng.uniform(1.2, 1.9)), "size": s}
    return eggs


def occluder_near(eggs, rng):                        # grey gripper-sized box 7 cm above one egg
    e = eggs[f"egg_{rng.integers(1, 5)}"]
    c = np.asarray(e["xyz_m"]) + [rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), 0.07]
    return [(c, [0.012, 0.03, 0.04], (150, 150, 155))]


def truth_of(eggs):
    return {k: {"xyz_m": v["xyz_m"], "quat_xyzw": v["quat_xyzw"], "location": CP.classify_location(v["xyz_m"])} for k, v in eggs.items()}


def noisy(d0, k, seed, frames=1):
    c = dict(cfg, depth_noise_k=k)
    return CP.temporal_fuse([CP._depth_postprocess(d0, c, np.random.default_rng(seed + j)) for j in range(frames)])


def eggp(e):
    ax = CP._axis_from_quat_xyzw(e["quat_xyzw"])
    return np.array([*e["xyz_m"], math.atan2(ax[1], ax[0]), math.asin(max(-1, min(1, ax[2])))])


A_, B_ = CP.EGG_PRIOR["half_long_m"], CP.EGG_PRIOR["half_short_m"]

# ---------- T1 temporal fusion ----------
flat = np.full((120, 160), 0.85, np.float32)
c1 = dict(cfg, depth_noise_k=0.0025)
one = CP._depth_postprocess(flat, c1, np.random.default_rng(1))
five = CP.temporal_fuse([CP._depth_postprocess(flat, c1, np.random.default_rng(10 + j)) for j in range(5)])
ratio = float(np.std(five - 0.85) / np.std(one - 0.85))
assert 0.38 < ratio < 0.55, ratio
holes = [one.copy() for _ in range(5)]
for j in range(3):
    holes[j][0, 0] = 0.0
fh = CP.temporal_fuse(holes)
assert fh[0, 0] == 0.0 and fh[0, 1] > 0, "pixel valid in only 2/5 frames must stay invalid"
print(f"[OK] T1 temporal_fuse 5 frames: noise x{ratio:.2f} (ideal 0.45), sparse pixels dropped")

# ---------- T2 view check ----------
rng = np.random.default_rng(3)
eggs = random_scene(rng, "basket_cluster")
rgb, d0 = render_clean(eggs)
dep = noisy(d0, 0.0025, 50)
e1 = eggs["egg_1"]
vt = CP.egg_view_check(eggp(e1), A_, B_, dep, K, T, stride=2, tol_m=0.006)
shift = eggp(e1).copy(); shift[0] += 0.012 * math.sin(shift[3]); shift[1] -= 0.012 * math.cos(shift[3])   # 12 mm sideways
vs = CP.egg_view_check(shift, A_, B_, dep, K, T, stride=2, tol_m=0.006)
assert vt["see_through_ratio"] < 0.03 and vt["match_ratio"] > 0.6, vt
assert vs["see_through_ratio"] > 0.10, vs
occ_box = [(np.asarray(e1["xyz_m"]) + [0, 0, 0.07], [0.04, 0.05, 0.03], (150, 150, 155))]
_, d_occ = render_clean(eggs, occ_box)
vo = CP.egg_view_check(eggp(e1), A_, B_, d_occ, K, T, stride=2, tol_m=0.006)
assert vo["coverage"] < 0.3, vo
print(f"[OK] T2 view check | true pose see-through {vt['see_through_ratio']} match {vt['match_ratio']} | "
      f"12 mm-off pose see-through {vs['see_through_ratio']} | occluded coverage {vo['coverage']}")

# ---------- T3 Monte-Carlo ----------
rng = np.random.default_rng(7)
t_xy, t_ax, all_ax, fp_trusted, n_untrusted, n_truth, n_det = [], [], [], 0, 0, 0, 0
t0 = time.time()
for n in range(10):
    kind = ["basket_cluster", "mixed"][n % 2]
    eggs = random_scene(rng, kind)
    boxes = occluder_near(eggs, rng) if n % 3 == 2 else ()
    rgb, d0 = render_clean(eggs, boxes)
    dep = noisy(d0, 0.0025, 1000 + 10 * n)
    rb = [(np.asarray(c), np.eye(3), np.asarray(hf)) for c, hf, _ in boxes]
    dets, diag = CP.perceive_eggs(rgb, dep, K, T, robot_boxes=rb or None)
    rows, fps, _ = CP.evaluate(dets, truth_of(eggs))
    s = CP.summarize(rows, fps)
    n_truth += s["truth"]; n_det += s["detected"]
    fp_trusted += s["trusted"]["false_positives"]; n_untrusted += s["trusted"]["untrusted_detections"]
    for r in rows:
        if r["detected"]:
            all_ax.append(r["axis_err_deg"])
            if r["trusted"]:
                t_xy.append(r["err_xy_mm"]); t_ax.append(r["axis_err_deg"])
assert fp_trusted == 0, fp_trusted
assert max(t_xy) < 3.5 and max(t_ax) < 15.0, (max(t_xy), max(t_ax))
assert np.mean(all_ax) < 6.0, np.mean(all_ax)
assert len(t_xy) >= 34, len(t_xy)
print(f"[OK] T3 Monte-Carlo 10 scenes, k=0.0025: detected {n_det}/{n_truth}, trusted {len(t_xy)}, untrusted {n_untrusted}, "
      f"trusted FP 0 | trusted xy max {max(t_xy):.2f} mm, axis max {max(t_ax):.2f} deg | all axis mean {np.mean(all_ax):.2f} deg "
      f"| {time.time()-t0:.0f}s")

# ---------- T4 hidden egg is not trusted ----------
rng = np.random.default_rng(11)
eggs = random_scene(rng, "basket_cluster")
e = eggs["egg_2"]
box = [(np.asarray(e["xyz_m"]) + [0.0, 0.012, 0.075], [0.03, 0.03, 0.035], (150, 150, 155))]
rgb, d0 = render_clean(eggs, box)
dets, _ = CP.perceive_eggs(rgb, noisy(d0, 0.0025, 77), K, T, robot_boxes=[(np.asarray(c), np.eye(3), np.asarray(hf)) for c, hf, _ in box])
rows, fps, _ = CP.evaluate(dets, truth_of(eggs))
r2 = [r for r in rows if r["truth_name"] == "egg_2"][0]
assert (not r2["detected"]) or (r2["trusted"] is False), r2
others = [r for r in rows if r["truth_name"] != "egg_2"]
assert all(r["detected"] for r in others), others
print(f"[OK] T4 mostly hidden egg_2: {'not detected' if not r2['detected'] else 'detected but untrusted, coverage ' + str(r2['coverage'])}; other eggs found")

# ---------- T5 warm start ----------
rng = np.random.default_rng(21)
eggs = random_scene(rng, "basket_cluster")
rgb, d0 = render_clean(eggs)
dA, _ = CP.perceive_eggs(rgb, noisy(d0, 0.0025, 5), K, T)
dB, diagB = CP.perceive_eggs(rgb, noisy(d0, 0.0025, 6), K, T, prior=[d for d in dA if d["trusted"]])
rB, fB, _ = CP.evaluate(dB, truth_of(eggs))
sB = CP.summarize(rB, fB)
assert len(dB) == 4 and sB["false_positives"] == 0 and sB["err_xy_mm"]["max"] < 3.0, (sB, diagB)
print(f"[OK] T5 warm start: {diagB['candidates']} candidates -> {len(dB)} eggs, no duplicates, xy max {sB['err_xy_mm']['max']} mm")

# ---------- T6 resting prior ----------
rng = np.random.default_rng(31)
eggs = random_scene(rng, "mixed")
for k_ in ("egg_1",):                                     # stand one holder egg up: must NOT be forced horizontal
    sx, sy = CP.SLOTS["slot_1"]
    eggs[k_] = {"xyz_m": [sx, sy, 0.0549], "quat_xyzw": [0.70711, 0.70711, 0.0, 0.0], "size": SIZES[0]}
rgb, d0 = render_clean(eggs)
dets, _ = CP.perceive_eggs(rgb, noisy(d0, 0.0025, 9), K, T)
rows, fps, _ = CP.evaluate(dets, truth_of(eggs))
byn = {r["truth_name"]: r for r in rows}
assert all(r["detected"] for r in rows) and not fps, rows
for nme in ("egg_3", "egg_4"):
    did = byn[nme]["det_id"]; d = [x for x in dets if x["det_id"] == did][0]
    assert abs(d["axis_world"][2]) < 0.10, (nme, d["axis_world"])
assert byn["egg_1"]["axis_err_deg"] < 15.0 and byn["egg_1"]["det_location"] == "egg_holder.slot_1", byn["egg_1"]
print(f"[OK] T6 resting prior: basket eggs horizontal; upright slot egg kept (axis err {byn['egg_1']['axis_err_deg']} deg)")

# ---------- T7 files ----------
with tempfile.TemporaryDirectory() as td:
    wr = CP.CaptureWriter(td, cfg, K, T)
    s = CP.summarize(rows, fps)
    cdir = wr.write(1, "plan_grasp:egg_2", 3.0, rgb, d0, np.zeros(10), dets, truth_of(eggs), rows, fps, s, {}, K, T, 0.5)
    hdr = next(csv.reader(open(wr.run_dir / "errors.csv")))
    assert {"confidence", "trusted", "coverage"} <= set(hdr), hdr
    summ = json.loads((wr.run_dir / "summary.json").read_text())
    assert "overall_trusted" in summ and summ["overall_trusted"]["false_positives"] == 0 and "axis_err_deg_max" in summ["overall"], summ
    per = json.loads((cdir / "perceived_scene.json").read_text())
    one_ = next(iter(per["objects"].values()))
    assert "confidence" in one_ and "trusted" in one_ and "coverage" in one_["quality"], one_
    info = json.loads((wr.run_dir / "camera_info.json").read_text())
    assert info["version"] == CP.VERSION and "trusted_rule" in info and info["depth_temporal_frames"] == cfg["depth_frames"]
print("[OK] T7 errors.csv + summary.json(overall_trusted) + perceived_scene.json(confidence/trusted/quality) + camera_info")

# ---------- T8 camera loop with noise: 5 fused frames + warm start ----------
os.environ["NERO_V308_DEPTH_NOISE_K"] = "0.0025"
os.environ["NERO_V310_DEPTH_FRAMES"] = "5"
rng = np.random.default_rng(41)
eggs = random_scene(rng, "basket_cluster")
rgbF, depthF = render_clean(eggs)
m = types.ModuleType("mujoco")
class _mjtObj:
    mjOBJ_BODY, mjOBJ_CAMERA, mjOBJ_JOINT = 1, 7, 3
m.mjtObj = _mjtObj
order = [eggs[f"egg_{i}"] for i in range(1, 5)]
class _Model:
    nq = 70
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
import multiprocessing as _mp
q = _mp.Array("d", 1024); meta = _mp.Array("d", 4); mpath = _mp.Array("c", 4096)
b_ = b"/fake/scene.xml"; mpath[:len(b_)] = b_; mpath[len(b_)] = b"\0"
meta[0] = 70; meta[1] = 1.0; meta[3] = 1.0
seq = _mp.Value("i", 0); tag = _mp.Array("c", 256); status = _mp.Value("i", 0)
with tempfile.TemporaryDirectory() as td:
    stop = threading.Event()
    th = threading.Thread(target=CP.camera_process_main, args=(q, meta, mpath, seq, tag, status, stop, td, "fake"))
    th.start()
    for _ in range(200):
        if status.value == 1: break
        time.sleep(0.05)
    assert status.value == 1
    run = next(Path(td).iterdir())
    for i_, t_ in enumerate(("observe_scene", "plan_grasp:egg_1"), 1):
        bt = t_.encode(); tag[:len(bt)] = bt; tag[len(bt)] = b"\0"
        with seq.get_lock(): seq.value += 1
        for _ in range(1200):
            if len([c for c in run.glob("capture_*") if (c / "errors.json").exists()]) >= i_: break
            time.sleep(0.05)
    stop.set(); th.join(10)
    caps = sorted(run.glob("capture_*"))
    assert len(caps) == 2, caps
    e1j = json.loads((caps[0] / "errors.json").read_text()); e2j = json.loads((caps[1] / "errors.json").read_text())
    assert e1j["diagnostics"]["depth_frames_fused"] == 5 and e2j["diagnostics"]["candidates"] > e1j["diagnostics"]["candidates"]
    per2 = json.loads((caps[1] / "perceived_scene.json").read_text())
    assert all((o["quality"] or {}).get("prev_capture") for o in per2["objects"].values()), per2
    summ = json.loads((run / "summary.json").read_text())
    assert summ["overall_trusted"]["false_positives"] == 0 and summ["overall"]["detection_rate"] == 1.0, summ
del sys.modules["mujoco"]
print(f"[OK] T8 camera loop: 5 depth frames fused, 2nd capture warm-started "
      f"({e1j['diagnostics']['candidates']} -> {e2j['diagnostics']['candidates']} candidates), prev_capture recorded, "
      f"trusted xy max {summ['overall_trusted']['err_xy_mm_max']} mm")

# ---------- T9 backend settle ----------
SRC = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
for mk in ("def _v310_wait_tcp_settled(", 'settle = _v310_wait_tcp_settled("set_gripper_opening")', '"v310_pre_settle": settle',
           '_v310_wait_tcp_settled(f"after q0 ({label})")', "NERO_V310_SETTLE"):
    assert mk in SRC, mk
i_set = SRC.index('settle = _v310_wait_tcp_settled("set_gripper_opening")'); i_before = SRC.index("tcp_before, _ = skill_core.current_tcp_pose()", i_set)
assert i_before - i_set < 300, "settle must run right before tcp_before is sampled"
print("[OK] T9 backend: TCP settle before set_gripper_opening baseline and after V3.07 q0")
