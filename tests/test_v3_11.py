#!/usr/bin/env python3
"""Offline tests for V3.11 (skills use camera perception; no MuJoCo / ROS / GPU).

K1  D435i body + floor stand: well-formed MJCF, visual only, reaches the floor, clear of the table,
    and NO part of the rig is visible in the camera image (everything behind / outside the frustum).
K2  identity tracker: names seeded once, then kept by perception alone - an egg moved basket -> slot
    keeps its name; an untrusted stray blob cannot steal a name; an unseen egg is reported stale.
K3  synchronous result channel: camera loop (fake MuJoCo) answers v311_capture_sync() with the tracked
    scene; a newer request is never answered with an older capture; camera off -> PERCEPTION_UNAVAILABLE.
K4  backend wiring: GUI pose-source switch, lamp in the log toolbar (old caption removed), camera-mode
    capture points, perception poses in observe/plan_grasp/verify, truth kept out of model-facing results.
"""
import json, math, os, re, sys, tempfile, textwrap, threading, time, types
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("NERO_V308_WIDTH", "848")
os.environ.setdefault("NERO_V308_HEIGHT", "480")
import NERO_V308_Camera_Perception as CP

assert CP.VERSION == "V3.11"
cfg = CP.camera_config(); K = CP.intrinsics(cfg); T = CP.extrinsics(cfg)

# ---------- K1 camera rig ----------
with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "scene.xml"
    p.write_text("<mujoco><worldbody><body name='a'/></worldbody></mujoco>")
    out = Path(CP.inject_camera_into_mjcf(str(p), cfg)).read_text()
    root = ET.fromstring(out)
    cams = root.findall(".//camera"); geoms = [g for g in root.iter("geom") if g.get("name", "").startswith("v311_")]
    assert len(cams) == 1 and len(geoms) >= 10, (len(cams), len(geoms))
    assert all(g.get("contype") == "0" and g.get("conaffinity") == "0" for g in geoms)
    os.environ["NERO_V311_DRAW_CAMERA"] = "0"
    p2 = Path(td) / "scene2.xml"; p2.write_text("<mujoco><worldbody></worldbody></mujoco>")
    out2 = Path(CP.inject_camera_into_mjcf(str(p2), cfg)).read_text()
    assert "v311_" not in out2 and out2.count("<camera") == 1
    os.environ.pop("NERO_V311_DRAW_CAMERA")

parts = CP.camera_rig_parts(cfg)
samples = []
for kind, name, a, b, R, rgba in parts:
    a = np.asarray(a, float)
    if kind == "capsule_ft":
        b = np.asarray(b, float)
        for t in np.linspace(0, 1, 60):
            c = a + t * (b - a)
            for d in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 0]):
                samples.append((name, c + 0.009 * np.asarray(d)))
    elif kind == "box":
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    samples.append((name, a + R @ (np.asarray(b) * [sx, sy, sz])))
    elif kind == "cylinder" and R is not None:
        for ang in np.linspace(0, 2 * math.pi, 16, endpoint=False):
            for sz in (-1, 1):
                samples.append((name, a + R @ np.array([b[0] * math.cos(ang), b[0] * math.sin(ang), sz * b[1]])))
    else:
        r = float(b[0]); hz = float(b[1]) if len(b) > 1 else r
        for d in ([r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, hz], [0, 0, -hz]):
            samples.append((name, a + np.asarray(d)))
names = [n for n, _ in samples]; P = np.array([q for _, q in samples])
uv, z = CP.project(P, K, T)
inside = (z > 0.005) & (uv[:, 0] >= 0) & (uv[:, 0] < K["width"]) & (uv[:, 1] >= 0) & (uv[:, 1] < K["height"])
assert not inside.any(), sorted({names[i] for i in np.where(inside)[0]})
zmin = P[:, 2].min()
assert abs(zmin - (-0.28)) < 0.02, zmin                       # stands on the floor (z = -0.28 m)
in_table = (P[:, 0] > -0.72) & (P[:, 0] < -0.05) & (np.abs(P[:, 1]) < 0.38) & (P[:, 2] > -0.06) & (P[:, 2] < 0.0)
assert not in_table.any(), "camera stand intersects the table"
body = [pt for (kind, n, a, b, R, rg) in parts if n == "v311_d435i_body" for pt in [a]][0]
assert np.linalg.norm(np.asarray(body) - np.asarray(cfg["pos"])) < 0.03
print(f"[OK] K1 D435i body + tripod stand: {len(geoms)} visual geoms, floor at {zmin:.3f} m, clear of table, "
      f"0/{len(P)} rig sample points visible in the camera image")

# ---------- K2 tracker ----------
def det(i, xyz, trusted=True, axis=(1, 0, 0)):
    return {"det_id": f"det_{i}", "xyz_m": list(xyz), "axis_world": list(axis), "quat_xyzw": CP._quat_from_z_to(np.asarray(axis, float)),
            "location": CP.classify_location(xyz), "confidence": 0.98 if trusted else 0.4, "trusted": trusted,
            "quality": {"coverage": 0.95 if trusted else 0.4}}

eggs0 = {"egg_1": [-0.456, -0.224, 0.0225], "egg_2": [-0.381, -0.141, 0.023], "egg_3": [-0.429, -0.174, 0.022], "egg_4": [-0.395, -0.087, 0.023]}
tr = CP.PerceptionTracker()
d0 = [det(k, v) for k, v in enumerate([eggs0["egg_3"], eggs0["egg_1"], eggs0["egg_4"], eggs0["egg_2"]], 1)]
match0 = {"egg_3": "det_1", "egg_1": "det_2", "egg_4": "det_3", "egg_2": "det_4"}
out0, dg0 = tr.update(d0, 1, seed_match=match0)
assert sorted(dg0["seeded"]) == ["egg_1", "egg_2", "egg_3", "egg_4"] and all(o["trusted"] for o in out0.values())
# capture 2: egg_1 is now in slot_1, other eggs jitter 1 mm, an UNTRUSTED blob near the basket; detection order shuffled;
# a WRONG seed_match is passed to prove that seeded names are never re-labelled from the matcher.
slot1 = [-0.510, 0.123, 0.055]
jit = lambda v: [v[0] + 0.001, v[1] - 0.001, v[2]]
d1 = [det(1, jit(eggs0["egg_4"])), det(2, slot1, axis=(0, 0, 1)), det(3, [-0.44, -0.20, 0.02], trusted=False),
      det(4, jit(eggs0["egg_2"])), det(5, jit(eggs0["egg_3"]))]
wrong = {"egg_1": "det_5", "egg_3": "det_2"}
out1, dg1 = tr.update(d1, 2, seed_match=wrong)
assert out1["egg_1"]["det_id"] == "det_2" and out1["egg_1"]["location"] == "egg_holder.slot_1", out1["egg_1"]
assert out1["egg_3"]["det_id"] == "det_5" and out1["egg_4"]["det_id"] == "det_1" and out1["egg_2"]["det_id"] == "det_4"
assert dg1["reassociated_moved"] == ["egg_1"] and dg1["unassigned_detections"] == ["det_3"] and not dg1["seeded"], dg1
# capture 3: egg_2 hidden by the gripper; only an untrusted fragment appears 40 mm away -> egg_2 stale, not stolen
d2 = [det(1, jit(eggs0["egg_4"])), det(2, slot1, axis=(0, 0, 1)), det(3, jit(eggs0["egg_3"])),
      det(4, [eggs0["egg_2"][0] + 0.04, eggs0["egg_2"][1], 0.03], trusted=False)]
out2, dg2 = tr.update(d2, 3)
assert out2["egg_2"]["seen"] is False and out2["egg_2"]["trusted"] is False and out2["egg_2"]["captures_since_seen"] == 1
assert abs(out2["egg_2"]["xyz_m"][0] - jit(eggs0["egg_2"])[0]) < 1e-9 and dg2["unassigned_detections"] == ["det_4"]
idc = CP._identity_check(out1, {"egg_1": "det_2", "egg_2": "det_4", "egg_3": "det_5", "egg_4": "det_1"})
assert idc == {"consistent": 4, "inconsistent": []}
print("[OK] K2 tracker: seeded once; egg_1 basket->slot keeps its name; untrusted blobs never steal a name; hidden egg reported stale")

# ---------- K3 synchronous result channel ----------
SRC = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
a = SRC.index("V311_STATE = {"); b = SRC.index("def _v308_camera_supervisor(", a)
import multiprocessing as _mp
g = {"os": os, "time": time, "json": json, "V308_SHARED": {}, "Path": Path, "__file__": str(ROOT / "NERO_V2_7_1_Unified_Backend.py")}
exec(SRC[a:b], g)
# camera off
try:
    g["v311_capture_sync"]("observe_scene", timeout_s=0.3)
    raise AssertionError("expected PERCEPTION_UNAVAILABLE")
except RuntimeError as e:
    assert "PERCEPTION_UNAVAILABLE" in str(e)
assert g["v311_mode"]() == "camera"     # V3.35: camera selected = camera, even when it is off (no truth fallback)

# camera on with a fake MuJoCo
sys.path.insert(0, str(ROOT / "tests"))
h, w = K["height"], K["width"]
_vv, _uu = np.mgrid[0:h, 0:w]
D = (np.stack([(_uu - K["ppx"]) / K["fx"], (_vv - K["ppy"]) / K["fy"], np.ones_like(_uu, float)], -1).reshape(-1, 3)) @ T[:3, :3].T
O = T[:3, 3]
def qR(q):
    x, y, z, w_ = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w_), 2*(x*z+y*w_)], [2*(x*y+z*w_), 1-2*(x*x+z*z), 2*(y*z-x*w_)], [2*(x*z-y*w_), 2*(y*z+x*w_), 1-2*(x*x+y*y)]])
EGGS = {
    "egg_1": {"xyz_m": [-0.45589, -0.22358, 0.02254], "quat_xyzw": [0.69017, -0.15385, -0.15393, 0.69014], "size": [0.0220, 0.0212, 0.0310]},
    "egg_2": {"xyz_m": [-0.38201, -0.14007, 0.02304], "quat_xyzw": [-0.6184, 0.34229, -0.35353, 0.61273], "size": [0.0228, 0.0217, 0.0320]},
    "egg_3": {"xyz_m": [-0.42858, -0.17333, 0.02224], "quat_xyzw": [0.55779, 0.43505, 0.42425, 0.56535], "size": [0.0217, 0.0209, 0.0305]},
    "egg_4": {"xyz_m": [-0.39566, -0.08532, 0.02324], "quat_xyzw": [-0.70363, -0.06996, 0.06972, 0.70367], "size": [0.0231, 0.0219, 0.0324]},
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
        A = np.sum(dd * dd, 1); B = 2 * dd @ o; C = o @ o - 1; disc = B * B - 4 * A * C; ok = disc > 0
        t1 = np.where(ok, (-B - np.sqrt(np.maximum(disc, 0))) / (2 * A), np.inf)
        better = ok & (t1 > 0) & (t1 < tbest); tbest[better] = t1[better]; col[better] = np.array([214, 150, 88], np.uint8)
    return col.reshape(h, w, 3), np.where(np.isfinite(tbest), tbest, 0.0).reshape(h, w).astype(np.float32)
rgbF, depthF = render(EGGS)
m = types.ModuleType("mujoco")
class _mjtObj:
    mjOBJ_BODY, mjOBJ_CAMERA, mjOBJ_JOINT = 1, 7, 3
m.mjtObj = _mjtObj
order = [EGGS[f"egg_{i}"] for i in range(1, 5)]
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
S = g["V308_SHARED"]
S.update(qpos=_mp.Array("d", 1024), meta=_mp.Array("d", 4), model_path=_mp.Array("c", 4096), trig_seq=_mp.Value("i", 0),
         trig_tag=_mp.Array("c", 256), status=_mp.Value("i", 0), res_seq=_mp.Value("i", 0), res_buf=_mp.Array("c", 65536))
bp = b"/fake/scene.xml"; S["model_path"][:len(bp)] = bp; S["model_path"][len(bp)] = b"\0"
S["meta"][0] = 70; S["meta"][1] = 1.0; S["meta"][3] = 1.0
with tempfile.TemporaryDirectory() as td:
    stop = threading.Event()
    th = threading.Thread(target=CP.camera_process_main, args=(S["qpos"], S["meta"], S["model_path"], S["trig_seq"], S["trig_tag"],
                                                               S["status"], stop, td, "fake", S["res_seq"], S["res_buf"]))
    th.start()
    assert g["v311_mode"]() == "camera"
    p1 = g["v311_capture_sync"]("observe_scene", timeout_s=60)
    assert sorted(p1["objects"]) == ["egg_1", "egg_2", "egg_3", "egg_4"], p1
    for n, o in p1["objects"].items():
        err = np.linalg.norm(np.subtract(o["xyz_m"], EGGS[n]["xyz_m"])) * 1000
        assert o["seen"] and o["trusted"] and err < 3.0, (n, err, o)
    # a stale result must never answer a newer request
    S["res_seq"].value = 0
    p2 = g["v311_capture_sync"]("plan_grasp:egg_2", timeout_s=60)
    assert p2["capture"] == p1["capture"] + 1 and p2["tag"] == "plan_grasp:egg_2" and p2["seq"] == 2, p2
    run = next(Path(td).iterdir())
    tj = json.loads((run / "capture_0002_plan_grasp_egg_2" / "tracked_scene.json").read_text())
    ej = json.loads((run / "capture_0002_plan_grasp_egg_2" / "errors.json").read_text())
    assert tj["objects"]["egg_2"]["seen"] and ej["diagnostics"]["tracking"]["identity_check"]["inconsistent"] == []
    stop.set(); th.join(10)
del sys.modules["mujoco"]
print(f"[OK] K3 synchronous capture: 4/4 eggs tracked, xy < 3 mm, wait {p1['wait_s']}s; newer request never gets an older "
      f"capture; camera off -> PERCEPTION_UNAVAILABLE; identity consistent")

# ---------- K4 backend wiring ----------
for mk in ('("camera", "Fixed stand camera - D435i RGB-D', '("truth", "JSON ground truth (MuJoCo)")', 'V311_STATE["mode"] = v311_var.get()',
           '_v236_lamp["parent"] = global_log_toolbar', 'c.pack(side="left",padx=(2,6))',
           'name == "plan_grasp" and args.get("_v307_inner")', '_v311_capture(_v311_tag)',
           'obj_xyz, obj_q = _v311_obj_pose(obj, require_trusted=True)', '_v307_go_q0("PERCEPTION RECAPTURE")',
           '_po, _qo = _v311_obj_pose(_other, require_trusted=False)', 'location = "not_confirmed_by_camera"',
           '"pose_source": (v335_pose_source_label() if cam311 else "mujoco_ground_truth")',
           'V311_MONITOR_BASELINE[obj] = skill_core.object_pose(obj) if v311_mode() == "camera" else None',
           '[V3.11 POSE SOURCE] this task:', 'V308_SHARED.get("res_seq"), V308_SHARED.get("res_buf")'):
    assert mk in SRC, mk
assert "Copies the entire Unified Runtime Log to the system clipboard" not in SRC
assert "c.place(x=8,y=8)" not in SRC, "lamp must not be placed over the log text"
# truth in camera-mode verify goes only to the log line, never into the returned verification dict
vblk = SRC[SRC.index('if name == "verify_object_in_slot":'):SRC.index('if name == "v306_safe_end":')]
assert "truth(log only)" in vblk and '"truth' not in vblk.replace('"truth(log only)', '')
AG = (ROOT / "NERO_V2_7_1_Agent.py").read_text()
assert "V3.11 CAMERA PERCEPTION" in AG and "PERCEPTION_TARGET_UNTRUSTED" in AG
print("[OK] K4 backend: pose-source switch, lamp in toolbar, camera capture points, perception poses, truth kept out of results")
