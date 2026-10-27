#!/usr/bin/env python3
"""V3.26 wrist camera (simulation), parallel to the fixed camera.

W1 MJCF: the wrist <camera> is inserted inside body link7 (idempotent), no geom unless asked; the fixed-camera
   module is byte-identical to V3.11
W2 mount geometry: optical axis crosses the approach axis ~0.35 m in front of the camera; the fingers are outside the
   image; MuJoCo camera pose -> optical frame (x right, y down, z forward)
W3 survey planning: look points per tool; merging views (latest sighting wins, never replaced by an unseen entry)
W4 capture protocol with shared memory (thread stands in for the camera process)
W5 (numpy) synthetic wrist RGB-D at the look pose above the basket (ray-cast eggs + table, D435 min range) ->
   the unchanged V3.10 perceive_eggs localises every egg in view within a few mm
W6 backend wiring: pose source camera | wrist | truth, wrist process started on demand, _v311_capture dispatch,
   look move + return to q0 before grasp planning, launcher checks the module
"""
import os as _os335; _os335.environ.setdefault("NERO_V335_CALIB_LEVEL", "0"); _os335.environ.setdefault("NERO_V335_REQUIRE_MEASURED", "0")   # V3.35: legacy tests assume an ideal camera, single-view trust
import hashlib, json, math, os, sys, tempfile, threading, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V326_Wrist_Camera as W  # noqa: E402

assert W.VERSION in ("V3.26", "V3.27", "V3.28", "V3.29", "V3.30", "V3.31", "V3.31.1", "V3.33", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
tmp = Path(tempfile.mkdtemp(prefix="v326_"))
src = tmp / "scene.xml"
src.write_text("<mujoco><worldbody><body name='link6'><body name='link7' pos='0 0 0.1'><geom type='sphere' size='0.02'/>"
               "</body></body></worldbody></mujoco>")
os.environ.pop("NERO_V326_DRAW_WRIST_CAMERA", None)
os.environ["NERO_V330_DRAW_WRIST_CAMERAS"] = "0"                               # V3.30 draws models by default
out = W.inject_wrist_camera_into_mjcf(str(src))
import xml.etree.ElementTree as ET
r = ET.parse(out).getroot()
cam = r.find(".//body[@name='link7']/camera")
assert cam is not None and cam.get("name") == W.WRIST_CAMERA_NAME and cam.get("mode") == "fixed"
assert len(r.findall(".//body[@name='link7']/geom")) == 1                     # nothing physical added
assert W.inject_wrist_camera_into_mjcf(out) == out                            # idempotent
os.environ["NERO_V326_DRAW_WRIST_CAMERA"] = "1"
out2 = W.inject_wrist_camera_into_mjcf(str(src))
g = [x for x in ET.parse(out2).getroot().findall(".//body[@name='link7']/geom") if x.get("name") == "v330_cam_d435i_housing"]
assert g and g[0].get("density") == "0" and g[0].get("contype") == "0" and g[0].get("conaffinity") == "0"
os.environ.pop("NERO_V326_DRAW_WRIST_CAMERA")
os.environ.pop("NERO_V330_DRAW_WRIST_CAMERAS")
assert hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest() == \
    "020cc6680f18187214c6c9d33fe24735bba9d4adb2cc8d9a9466e02e62b96efd"
print("[W1] wrist camera injected in link7 (camera only; optional massless non-colliding body); fixed-camera module unchanged")

cfg = {"mount_pos": [0.06, 0.0, 0.07], "mount_target": [0.40, 0.0, 0.0], "mount_up": [0, 0, 1.0], "fovy_deg": 42.0}
x, y, f = W.mount_axes(cfg)
p = cfg["mount_pos"]
# closest approach of the optical axis to the approach axis (link7 x axis, z = -0.021 for the TCP line)
s_cross = (0.0 - p[2]) / f[2]
assert 0.30 < s_cross < 0.40 and abs(f[1]) < 1e-9
for fp in ([0.13, 0.0, -0.02], [0.19, 0.03, -0.02], [0.19, -0.03, -0.02]):        # finger tips / pads
    d = [fp[i] - p[i] for i in range(3)]
    n = math.sqrt(sum(v * v for v in d))
    up_ang = math.degrees(math.asin(sum(d[i] * y[i] for i in range(3)) / n))
    assert up_ang < -21.0, (fp, up_ang)                                          # below the image (VFOV 42 deg)
try:
    import numpy as np
    have_np = True
except Exception:
    have_np = False
if have_np:
    Rm = np.stack([x, y, [-v for v in f]], axis=1)
    T = W.optical_from_mujoco([1, 2, 3], Rm.flatten())
    assert np.allclose(T[:3, 2], f) and np.allclose(T[:3, 0], x) and np.allclose(T[:3, 1], [-v for v in y])
print(f"[W2] optical axis crosses the approach axis {s_cross:.2f} m in front of the camera; fingers below the image; "
      "MuJoCo camera frame -> optical frame")

objs = {"egg_2": {"xyz_m": [-0.40, -0.13, 0.03]}, "egg_4": {"xyz_m": [-0.42, 0.16, 0.03]}}
assert [n for n, _ in W.look_points("observe_scene")] == ["basket", "holder"]
assert [n for n, _ in W.look_points("verify_object_in_slot:egg_1", objs)] == ["holder"]
lp = W.look_points("plan_grasp:egg_2", objs)
assert lp[0][0] == "above_egg_2" and lp[0][1][:2] == [-0.40, -0.13]
assert [n for n, _ in W.look_points("plan_grasp_recapture:egg_4", objs)] == ["basket"]   # egg_4 is in the holder
assert [n for n, _ in W.look_points("plan_grasp:egg_9", objs)] == ["basket"]
p1 = {"capture": 7, "tag": "a", "wait_s": 1.0, "objects": {"egg_1": {"xyz_m": [1, 0, 0], "seen": True, "trusted": True},
                                                           "egg_3": {"xyz_m": [3, 0, 0], "seen": False}}}
p2 = {"capture": 8, "tag": "b", "wait_s": 2.0, "objects": {"egg_1": {"xyz_m": [9, 9, 9], "seen": False},
                                                           "egg_3": {"xyz_m": [3.1, 0, 0], "seen": True, "trusted": True},
                                                           "egg_4": {"xyz_m": [4, 0, 0], "seen": False}}}
m = W.merge_payloads([p1, p2], ["basket", "holder"])
assert m["objects"]["egg_1"]["xyz_m"] == [1, 0, 0] and m["objects"]["egg_1"]["view"] == "basket"
assert m["objects"]["egg_3"]["xyz_m"] == [3.1, 0, 0] and m["objects"]["egg_4"]["seen"] is False
assert m["capture"] == 8 and m["captures"] == [7, 8] and m["wait_s"] == 3.0 and m["views"] == ["basket", "holder"]
print("[W3] look points: observe = basket + holder, plan_grasp = above the target (basket), verify = holder; views merged")

import multiprocessing as mp
sh = {"status": mp.Value("i", 1), "trig_seq": mp.Value("i", 0), "trig_tag": mp.Array("c", 256),
      "res_seq": mp.Value("i", 0), "res_buf": mp.Array("c", 4096)}


def fake_cam():
    last = 0
    t_end = time.time() + 5
    while time.time() < t_end:
        if sh["trig_seq"].value != last:
            last = sh["trig_seq"].value
            tag = bytes(sh["trig_tag"][:]).split(b"\0", 1)[0].decode()
            b = json.dumps({"seq": last, "capture": last, "tag": tag, "objects": {}}).encode()
            with sh["res_buf"].get_lock():
                sh["res_buf"][:len(b)] = b
                sh["res_buf"][len(b)] = b"\0"
            sh["res_seq"].value = last
        time.sleep(0.01)


threading.Thread(target=fake_cam, daemon=True).start()
pl = W.capture_sync(sh, "observe_scene/basket", timeout_s=3)
assert pl["tag"] == "observe_scene/basket" and pl["capture"] == 1 and "wait_s" in pl
print("[W4] capture protocol: trigger tag -> payload")

if not have_np:
    print("[W5] numpy not installed here -> synthetic wrist rendering skipped")
else:
    import NERO_V308_Camera_Perception as P8
    os.environ["NERO_V308_EGG_COLOR_FILTER"] = "0"
    cfgw = dict(P8.camera_config(), **cfg)
    K = P8.intrinsics(cfgw)
    # look pose above the basket: TCP = (bx, by, 0.03 + 0.29), gripper down -> link7 x = world -z
    bx, by = -0.425, -0.155
    tcp = np.array([bx, by, 0.03 + W.look_tcp_height_m()])
    R7 = np.stack([[0, 0, -1.0], [0, 1.0, 0], [1.0, 0, 0]], axis=1)             # x down, y = y, z = +x(world)
    p7 = tcp - R7 @ np.array([0.155, 0.0, -0.021])
    Rc_b = np.stack([x, y, [-v for v in f]], axis=1)                            # camera axes in link7
    Rc = R7 @ Rc_b
    pc = p7 + R7 @ np.array(cfg["mount_pos"])
    T = W.optical_from_mujoco(pc, Rc.flatten())
    a, b = P8.EGG_PRIOR["half_long_m"], P8.EGG_PRIOR["half_short_m"]
    eggs = [((bx + 0.035, by + 0.02, b), 0.3), ((bx - 0.03, by - 0.03, b), 1.4), ((bx + 0.0, by - 0.045, b), 2.4),
            ((bx - 0.045, by + 0.04, b), 0.9)]
    H, Wd = K["height"], K["width"]
    vv, uu = np.mgrid[0:H, 0:Wd]
    dirs_c = np.stack([(uu - K["ppx"]) / K["fx"], (vv - K["ppy"]) / K["fy"], np.ones_like(uu, dtype=float)], -1)
    Rw = T[:3, :3]
    o = T[:3, 3]
    dw = dirs_c @ Rw.T
    zbest = np.full((H, Wd), np.inf)
    s_tab = (0.0 - o[2]) / dw[..., 2]
    zbest = np.where(s_tab > 0, s_tab, np.inf)
    for (cx, cy, cz), yaw in eggs:
        Re = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]])
        inv = np.diag([1 / a, 1 / b, 1 / b]) @ Re.T
        oe = inv @ (o - np.array([cx, cy, cz]))
        de = dw @ inv.T
        A = (de * de).sum(-1)
        B = 2 * (de * oe).sum(-1)
        C = float(oe @ oe) - 1
        disc = B * B - 4 * A * C
        s = np.where(disc > 0, (-B - np.sqrt(np.maximum(disc, 0))) / (2 * A), np.inf)
        zbest = np.minimum(zbest, np.where(s > 0, s, np.inf))
    depth = np.where(np.isfinite(zbest), zbest, 0.0).astype(np.float32)        # s is z-depth (dirs have z = 1)
    rgb = np.full((H, Wd, 3), 90, np.uint8)
    res = {}
    for kname, knoise in (("clean", 0.0), ("noisy k=0.0025", 0.0025)):
        rng = np.random.default_rng(1)
        frames = [P8._depth_postprocess(depth, dict(cfgw, depth_noise_k=knoise), rng) for _ in range(5 if knoise else 1)]
        dz = P8.temporal_fuse(frames)
        dets, diag = P8.perceive_eggs(rgb, dz, K, T, stride=2)
        errs = []
        for (cx, cy, cz), _yaw in eggs:
            dmin = min((math.dist(d["xyz_m"], [cx, cy, cz]) for d in dets), default=9)
            errs.append(dmin * 1000)
        assert len(dets) >= 4 and max(errs) < 5.0, (kname, len(dets), errs)
        res[kname] = errs
    cam_h = float(o[2])
    print(f"[W5] synthetic wrist view above the basket (camera {cam_h:.2f} m high): all 4 eggs found; errors clean "
          + ", ".join(f"{e:.1f}" for e in res["clean"]) + " mm, with D435-like noise "
          + ", ".join(f"{e:.1f}" for e in res["noisy k=0.0025"]) + " mm (unchanged V3.10 perceive_eggs)")

if have_np:
    # W7: the wrist camera PROCESS loop with a fake mujoco (renderer returns the synthetic wrist images above)
    import types
    m = types.ModuleType("mujoco")
    m.mjtObj = types.SimpleNamespace(mjOBJ_BODY=1, mjOBJ_CAMERA=7, mjOBJ_JOINT=3)

    class _Model:
        nq = 70
        vis = types.SimpleNamespace(global_=types.SimpleNamespace(offwidth=640, offheight=480))

    egg_quats = []
    for (_c, yaw) in eggs:
        egg_quats.append([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])      # MuJoCo wxyz

    class _Data:
        def __init__(self, model):
            self.qpos = np.zeros(70)
            self.xpos = np.array([c for c, _ in eggs], dtype=float)
            self.xquat = np.array(egg_quats)
            self.cam_xpos = np.array([pc])
            self.cam_xmat = np.array([Rc.flatten()])
    m.MjModel = types.SimpleNamespace(from_xml_path=lambda p: _Model())
    m.MjData = _Data
    m.mj_name2id = lambda model, t, n: (0 if n == W.WRIST_CAMERA_NAME else {f"obj_egg{i}": i - 1 for i in range(1, 5)}.get(n, -1))
    m.mj_forward = lambda model, data: None
    dep_w = P8.temporal_fuse([P8._depth_postprocess(depth, dict(cfgw, depth_noise_k=0.0), np.random.default_rng(0))])

    class _Renderer:
        def __init__(self, model, height, width): self.d = False
        def enable_depth_rendering(self): self.d = True
        def disable_depth_rendering(self): self.d = False
        def update_scene(self, data, camera=None): assert camera == 0
        def render(self): return depth.copy() if self.d else rgb
        def close(self): pass
    m.Renderer = _Renderer
    sys.modules["mujoco"] = m
    q = mp.Array("d", 1024); meta = mp.Array("d", 4); mpath = mp.Array("c", 4096)
    b_ = b"/fake/scene.xml"; mpath[:len(b_)] = b_; mpath[len(b_)] = b"\0"
    meta[0] = 70; meta[1] = 2.0; meta[3] = 1.0
    shw = {"qpos": q, "meta": meta, "model_path": mpath, "trig_seq": mp.Value("i", 0), "trig_tag": mp.Array("c", 256),
           "status": mp.Value("i", 0), "res_seq": mp.Value("i", 0), "res_buf": mp.Array("c", 65536)}
    with tempfile.TemporaryDirectory() as td:
        stop = threading.Event()
        th = threading.Thread(target=W.wrist_camera_process_main, args=(
            shw["qpos"], shw["meta"], shw["model_path"], shw["trig_seq"], shw["trig_tag"], shw["status"], stop, td,
            "fake", shw["res_seq"], shw["res_buf"]))
        th.start()
        pay = W.capture_sync(shw, "observe_scene/basket", timeout_s=60)
        pay2 = W.capture_sync(shw, "plan_grasp:egg_2/above_egg_2", timeout_s=60)
        stop.set(); th.join(10)
        run = next(p_ for p_ in Path(td).iterdir() if p_.name != "live")   # V3.35: live/ = monitor image
        info = json.loads((run / "camera_info.json").read_text())
        caps = sorted(run.glob("capture_*"))
    del sys.modules["mujoco"]
    assert pay.get("camera") in ("wrist", "wrist_d435i") and pay["eggs_in_view"] == ["egg_1", "egg_2", "egg_3", "egg_4"], pay
    seen = {n: o for n, o in pay["objects"].items() if o["seen"]}
    assert len(seen) == 4 and all(o["trusted"] for o in seen.values()), pay["objects"]
    # V3.30: perception names its eggs itself (no simulator identity seeding) -> each egg matches ONE perceived name,
    # and the names stay the same in the second capture
    for (cx, cy, cz), _y in eggs:
        assert min(math.dist(o["xyz_m"], [cx, cy, cz]) for o in seen.values()) < 0.005
    for n_, o_ in pay2["objects"].items():
        if o_.get("seen") and n_ in seen:
            assert math.dist(o_["xyz_m"], seen[n_]["xyz_m"]) < 0.005
    assert pay2["capture"] == 2 and len(caps) == 2 and info["camera"] == W.WRIST_CAMERA_NAME
    assert abs(pay["cam_pos"][2] - float(o[2])) < 1e-3
    print(f"[W7] wrist camera process (fake MuJoCo): 2 captures, all 4 eggs seen + trusted, names stable, "
          f"camera pose per capture {pay['cam_pos']}, files in {run.name}/")

B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
for mk in ('("wrist", "Wrist camera - D435i', "def v326_start(stop_event)", "if v326_wrist_selected():", "def _v326_wrist_capture(",
           "def _v326_move_look(", '_v307_go_q0("V3.26 WRIST LOOK DONE")', "inject_wrist_camera_into_mjcf",
           '"wrist": "wrist"', "_v326_p.terminate()", "if V311_STATE.get(\"mode\") in (\"wrist\", \"wrist_rgb\"):"):
    assert mk in B, mk
assert '("camera", "Fixed stand camera - D435i RGB-D' in B and '("truth", "JSON ground truth (MuJoCo)")' in B  # V3.30 drop-down
assert "payload = v311_capture_sync(label)" in B                               # fixed-camera path still there
S = (ROOT / "start_nero_sim.sh").read_text()
assert 'WRIST="$HERE/NERO_V326_Wrist_Camera.py"' in S and '"$CAMERA" "$WRIST"' in S
print("[W6] backend: pose source camera (default) | wrist | truth; wrist process on demand; capture dispatch; "
      "look moves + q0 before planning; launcher checks the module")
print("ALL V3.26 TESTS PASS")
