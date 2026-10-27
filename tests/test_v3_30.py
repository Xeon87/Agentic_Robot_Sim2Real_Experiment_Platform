#!/usr/bin/env python3
"""V3.30: second wrist camera profile (InnoMaker U20CAM-1080P, 130 deg USB, RGB only), drop-down, camera models,
and the second anti-cheat pass.

C1 profiles: usb130 = rectified pinhole 1280x720, HFOV 100 deg (fovy from it), no depth, own mount; d435i as before;
   both cameras + visual models injected into link7, models massless / non-colliding / behind the optical centre
C2 hint protocol carries the camera profile; look poses use the selected camera's mount
C3 (numpy) RGB-only holder: dark wells -> table-plane rays -> 4x2 grid fit (registered pose, moved + 15 deg, eggs in
   2 slots); partial view -> not complete
C4 (numpy) RGB-only eggs: 4 eggs of a pile (some nearly touching) -> centres within 6 mm, long axis within 12 deg;
   an egg standing in a perceived slot -> located in that slot; robot self mask removes masked pixels
C5 anti-cheat: wrist process has no simulator identity seeding (generic naming in ViewTracker), one egg class
   model for every egg, scan stops without assuming the egg count, truth logs by nearest simulator egg
C6 GUI drop-down (4 sources), simulator draws only the selected camera model; fixed-camera module unchanged
"""
import os as _os335; _os335.environ.setdefault("NERO_V335_CALIB_LEVEL", "0"); _os335.environ.setdefault("NERO_V335_REQUIRE_MEASURED", "0")   # V3.35: legacy tests assume an ideal camera, single-view trust
import hashlib, math, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import NERO_V326_Wrist_Camera as W  # noqa: E402

assert W.VERSION in ("V3.30", "V3.31", "V3.31.1", "V3.33", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
# ------------------------------------------------------------------ C1
cu, cd = W.wrist_config("usb130"), W.wrist_config("d435i")
assert cu["name"] == W.USB_CAMERA_NAME and not cu["depth"] and cd["depth"] and cd["name"] == W.WRIST_CAMERA_NAME
hf = math.degrees(2 * math.atan(math.tan(math.radians(cu["fovy_deg"]) / 2) * cu["width"] / cu["height"]))
assert abs(hf - 100.0) < 1e-6 and cu["min_range_m"] == 0.0
import tempfile, xml.etree.ElementTree as ET
tmp = Path(tempfile.mkdtemp(prefix="v330_"))
src = tmp / "scene.xml"
src.write_text("<mujoco><worldbody><body name='link6'><body name='link7' pos='0 0 0.1'><geom type='sphere' size='0.02'/>"
               "</body></body></worldbody></mujoco>")
os.environ.pop("NERO_V330_DRAW_WRIST_CAMERAS", None); os.environ.pop("NERO_V326_DRAW_WRIST_CAMERA", None)
out = W.inject_wrist_camera_into_mjcf(str(src))
r = ET.parse(out).getroot()
cams = {c.get("name") for c in r.findall(".//body[@name='link7']/camera")}
assert cams == {W.WRIST_CAMERA_NAME, W.USB_CAMERA_NAME}
geoms = [g for g in r.findall(".//body[@name='link7']/geom") if (g.get("name") or "").startswith("v330_cam_")]
assert {g.get("name").split("_")[2] for g in geoms} == {"d435i", "usb130"}
for g in geoms:
    assert g.get("density") == "0" and g.get("contype") == "0" and g.get("conaffinity") == "0", g.attrib
for pf in W.PROFILES:
    c = W.wrist_config(pf)
    _x, _y, f = W.mount_axes(c)
    for g in W.camera_model_geoms(c):
        pos = [float(v) for v in g["pos"].split()]
        ahead = sum((pos[i] - c["mount_pos"][i]) * f[i] for i in range(3))
        half_depth = float(g["size"].split()[-1]) if g["type"] != "cylinder" else float(g["size"].split()[1])
        if "xyaxes" not in g:                       # V3.31.1 bracket: axis-aligned box in link7
            half_depth = sum(abs(float(sz) * f[i]) for i, sz in enumerate(g["size"].split()))
        assert ahead + half_depth <= 1e-6, (pf, g["name"], ahead, half_depth)        # never in front of the optics
assert W.inject_wrist_camera_into_mjcf(out) == out
print(f"[C1] usb130: {cu['width']}x{cu['height']} rectified, HFOV {hf:.0f} deg (fovy {cu['fovy_deg']:.1f}), RGB only, "
      f"mount {cu['mount_pos']}; both cameras + {len(geoms)} massless model geoms behind the optics in link7")

# ------------------------------------------------------------------ C2
s = W.encode_hints("observe_scene/scan_az180", cam="usb130", held="egg_1")
tag, h = W.decode_hints(s)
assert tag == "observe_scene/scan_az180" and h["cam"] == "usb130" and h["held"] == "egg_1"
assert W.decode_hints(W.encode_hints("x", cam="d435i"))[1]["cam"] == "d435i"
pc_u, fc_u = W.camera_in_tcp(cu)
pc_d, fc_d = W.camera_in_tcp(cd)
assert math.dist(pc_u, pc_d) < 0.02 and pc_u[0] * pc_d[0] > 0          # V3.33: both on the same side face
cand = W.look_pose_candidates([-0.42, 0.155, 0.043], cfg=cu)[0]
assert math.dist(cand["cam_xyz"], [-0.42 - 0.36 * cand["cam_fwd"][0], 0.155 - 0.36 * cand["cam_fwd"][1],
                                   0.043 - 0.36 * cand["cam_fwd"][2]]) < 2e-3
print("[C2] trigger tag carries cam=usb130; look poses aim the selected camera (its own mount)")

# ------------------------------------------------------------------ C3 / C4
try:
    import numpy as np
    import NERO_V308_Camera_Perception as P8
    import v329_synth as S
    have_np = True
except Exception:
    have_np = False
if have_np:
    os.environ.pop("NERO_V308_EGG_COLOR_FILTER", None)
    K = P8.intrinsics(cu)

    def T_for(pt, dist=0.36, k=0):
        c = W.look_pose_candidates(pt, cfg=cu, distances=[dist])[k]
        return S.camera_T_for(c, pc_u, fc_u)

    EGGS = [([-0.451, -0.181, 0.022], 0.3), ([-0.399, -0.181, 0.023], -0.5), ([-0.452, -0.130, 0.021], 1.1),
            ([-0.399, -0.129, 0.023], 2.0)]

    def parts(hx=-0.42, hy=0.155, hyaw=0.0, eggs=EGGS, in_slots=()):
        # V3.33 scene: white table (with a cool arm-shadow patch), moulded-pulp carton, lit brown eggs, grey floor
        ps = [S.table_rgb_part(n=0.0015, shadows=[(hx - 0.02, hx + 0.12, hy - 0.22, hy - 0.10)]),
              S.carton_rgb_part(hx, hy, hyaw)]
        for c, yw in eggs:
            ps.append(S.egg_rgb_part(c, True, yw))
        for k in in_slots:
            sl = S.slot_truth(hx, hy, hyaw)[k]
            ps.append(S.egg_rgb_part([sl[0], sl[1], 0.054], False))
        return ps

    for hx, hy, hyaw, occ in ((-0.420, 0.155, 0.0, ()), (-0.380, 0.200, math.radians(15), ("slot_1", "slot_6"))):
        T = T_for([hx, hy, 0.043])
        rgb = S.render_rgb(parts(hx, hy, hyaw, eggs=[], in_slots=occ), K, T)
        eggs_in = [d["xyz_m"][:2] for d in W.perceive_eggs_rgb(rgb, K, T, holder={"complete": True, "slots": {
            k: {"xyz_m": list(v)} for k, v in S.slot_truth(hx, hy, hyaw).items()}}) if d["location"] == "egg_holder_slot"]
        h = W.detect_holder_rgb(rgb, K, T, egg_xy_in_slots=eggs_in)
        tr = S.slot_truth(hx, hy, hyaw)
        errs = [math.dist(h["slots"][k]["xyz_m"][:2], tr[k][:2]) * 1000 for k in tr]
        assert h["complete"] and max(errs) < 3.0 and abs(math.degrees(h["yaw_rad"] - hyaw)) < 1.0, (h, errs)
        assert len(eggs_in) == len(occ)
        print(f"[C3] RGB carton at ({hx}, {hy}) yaw {math.degrees(hyaw):.0f} deg, eggs in {list(occ)}: "
              f"{h['method']}, complete, yaw err {abs(math.degrees(h['yaw_rad'] - hyaw)):.2f} deg, slot err max "
              f"{max(errs):.1f} mm")
    T = T_for([-0.42, -0.06, 0.03], 0.30)       # V3.31.1 mount: holder only partly in view from here
    hp = W.detect_holder_rgb(S.render_rgb(parts(), K, T), K, T)
    assert hp is None or not hp["complete"]
    print("[C3] holder cut by the image border -> not complete")

    T = T_for([-0.425, -0.155, 0.03])
    rgb = S.render_rgb(parts(), K, T)
    dets = W.perceive_eggs_rgb(rgb, K, T)
    assert len(dets) == 4, [d["xyz_m"] for d in dets]
    for c, yw in EGGS:
        d = min(dets, key=lambda d_: math.dist(d_["xyz_m"][:2], c[:2]))
        e = math.dist(d["xyz_m"][:2], c[:2]) * 1000
        ax = d["axis_world"]
        ang = math.degrees(math.acos(min(1.0, abs(ax[0] * math.cos(yw) + ax[1] * math.sin(yw)))))
        assert e < 6.0 and ang < 12.0 and d["trusted"], (c, d, e, ang)
    rmask = np.zeros((K["height"] // 2, K["width"] // 2), dtype=bool); rmask[:, :] = True
    assert W.perceive_eggs_rgb(rgb, K, T, robot_mask=rmask) == []
    boxes = [(np.array([-0.425, -0.155, 0.10]), np.eye(3), [0.02, 0.02, 0.02])]
    rm = W.robot_pixel_mask(K, T, boxes, rgb.shape[:2])
    assert 0 < rm.mean() < 0.5
    cam = np.asarray(T)[:3, 3]; Rc_ = np.asarray(T)[:3, :3]
    # a box beside the camera that straddles the camera plane: clipped at the near plane, not the whole image
    side = cam + Rc_ @ np.array([0.06, 0.0, 0.02])
    rm2 = W.robot_pixel_mask(K, T, [(side, Rc_, [0.03, 0.03, 0.03])], rgb.shape[:2])
    assert 0.01 < rm2.mean() < 0.6, rm2.mean()
    behind = cam - Rc_ @ np.array([0.0, 0.0, 0.02])
    assert W.robot_pixel_mask(K, T, [(behind, Rc_, [0.01, 0.01, 0.005])], rgb.shape[:2]).mean() == 0.0
    print(f"[C4] RGB eggs: 4/4 found in the pile, centre err max "
          f"{max(min(math.dist(d['xyz_m'][:2], c[:2]) for d in dets) for c, _ in EGGS) * 1000:.1f} mm, long axis "
          "< 12 deg; robot self mask (model boxes projected) removes masked pixels")
else:
    print("[C3/C4] numpy missing -> RGB perception skipped")

# ------------------------------------------------------------------ C7 (numpy) the wrist PROCESS with the usb130 camera
if have_np:
    import json, multiprocessing as mp, threading, types
    Tp = T_for([-0.42, 0.155, 0.043])                 # holder view; optical -> MuJoCo camera frame (x, -y, -z)
    rgb_h = S.render_rgb(parts(in_slots=("slot_2",)), K, Tp)
    Rm = np.asarray(Tp)[:3, :3]
    cam_xmat = np.stack([Rm[:, 0], -Rm[:, 1], -Rm[:, 2]], axis=1).flatten()
    m = types.ModuleType("mujoco")
    m.mjtObj = types.SimpleNamespace(mjOBJ_BODY=1, mjOBJ_CAMERA=7, mjOBJ_JOINT=3, mjOBJ_GEOM=5)

    class _Model:
        nq = 70
        vis = types.SimpleNamespace(global_=types.SimpleNamespace(offwidth=640, offheight=480))

    class _Data:
        def __init__(self, model):
            self.qpos = np.zeros(70)
            self.xpos = np.array([c for c, _ in EGGS], dtype=float)
            self.xquat = np.array([[1.0, 0, 0, 0]] * 4)
            self.cam_xpos = np.array([[0, 0, 1.0], np.asarray(Tp)[:3, 3]])
            self.cam_xmat = np.array([np.eye(3).flatten(), cam_xmat])
    m.MjModel = types.SimpleNamespace(from_xml_path=lambda p_: _Model())
    m.MjData = _Data
    ids = {W.WRIST_CAMERA_NAME: 0, W.USB_CAMERA_NAME: 1}
    m.mj_name2id = lambda model, t, n: ids.get(n, {f"obj_egg{i}": i - 1 for i in range(1, 5)}.get(n, -1))
    m.mj_forward = lambda model, data: None
    rendered = []

    class _Renderer:
        def __init__(self, model, height, width): self.d = False; self.cam = None
        def enable_depth_rendering(self): self.d = True
        def disable_depth_rendering(self): self.d = False
        def update_scene(self, data, camera=None): self.cam = camera
        def render(self):
            rendered.append((self.cam, self.d))
            assert not (self.cam == 1 and self.d), "the RGB-only camera must not render depth"
            return rgb_h.copy() if not self.d else np.zeros(rgb_h.shape[:2], np.float32)
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
        pay1 = W.capture_sync(shw, W.encode_hints("observe_scene/holder", cam="usb130"), timeout_s=60)
        pay = W.capture_sync(shw, W.encode_hints("observe_scene/holder2", cam="usb130"), timeout_s=60)
        stop.set(); th.join(10)
    # V3.33: an RGB-only egg gets its name only after a second view at the same place
    assert not [o for o in pay1["objects"].values() if o.get("seen")] and pay1["tracking"]["awaiting_second_view"] == 1, pay1
    del sys.modules["mujoco"]
    hh = pay["containers"]["egg_holder"]
    assert pay["camera"] == "wrist_usb130" and hh and hh["complete"] and hh["method"] == "rgb_carton_far_edges", pay
    tr = S.slot_truth()
    assert max(math.dist(hh["slots"][k]["xyz_m"][:2], tr[k][:2]) for k in tr) < 0.003
    assert all(c == 1 and not d for c, d in rendered)
    seen = [o for o in pay["objects"].values() if o.get("seen")]
    assert len(seen) == 1 and math.dist(seen[0]["xyz_m"][:2], tr["slot_2"][:2]) < 0.006, pay["objects"]
    print("[C7] wrist process, cam=usb130: RGB render only, carton complete from its far edges, the egg in slot_2 named "
          "after the second view at "
          f"{seen[0]['xyz_m']} and named by perception ({[n for n, o in pay['objects'].items() if o.get('seen')]})")

# ------------------------------------------------------------------ C5
WM = (ROOT / "NERO_V326_Wrist_Camera.py").read_text()
B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
assert "seed_match=None, hints=hints" in WM and "seed_match=match, hints=hints" not in WM
assert "V3.30 GENERIC NAMING" in WM
if have_np:
    base = P8.PerceptionTracker()
    tr = W.ViewTracker(base, names=["egg_1", "egg_2"])
    Kd = P8.intrinsics(cd)
    Tdown = [[1, 0, 0, -0.42], [0, -1, 0, -0.15], [0, 0, -1, 0.40], [0, 0, 0, 1]]
    dd = [{"det_id": i, "xyz_m": p, "quat_xyzw": [0, 0, 0, 1], "axis_world": [1, 0, 0], "location": "x", "trusted": True,
           "quality": {}} for i, p in enumerate([[-0.40, -0.15, 0.02], [-0.45, -0.15, 0.02]])]
    o1, g1 = tr.update(dd, 1, Kd, Tdown)
    assert o1["egg_1"]["xyz_m"] == [-0.45, -0.15, 0.02] and o1["egg_2"]["xyz_m"] == [-0.40, -0.15, 0.02] and g1["named"]
    o2, _ = tr.update(list(reversed(dd)), 2, Kd, Tdown)
    assert o2["egg_1"]["xyz_m"] == [-0.45, -0.15, 0.02]
    print("[C5] new eggs named in scan order from the name pool (no simulator matcher); names stay with the eggs")
for mk in ("def _v330_apply_class_geometry(on):", "egg class placeholder (V3.30) - NOT measured yet",
           "_v330_apply_class_geometry(_v329_no_prior())", 'NERO_V330_SCAN_STABLE_VIEWS', "def _v330_nearest_truth(xyz):",
           "_tn330, txyz = _v330_nearest_truth(o[\"xyz_m\"])"):
    assert mk in B, mk
print("[C5] one egg class model for all eggs, scan stop without the egg count, truth logs by nearest simulator egg")

# ------------------------------------------------------------------ C6
for mk in ('("wrist_rgb", "Wrist camera - USB 130deg RGB only, InnoMaker U20CAM-1080P (V3.30)")', "ttk.Combobox(v311_frame",
           'v330_combo.bind("<<ComboboxSelected>>", _v330_on_combo)', "def v330_wrist_profile():",
           "_v330_apply_wrist_model_visibility(model)", "cam=v330_wrist_profile()",
           "cfg=_v326w.wrist_config(v330_wrist_profile())", '"wrist_usb130": "wrist_rgb"'):
    assert mk in B, mk
assert "ttk.Radiobutton(v311_frame" not in B
h = hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest()
assert h.startswith("020cc6680f181872"), h
print("[C6] drop-down with 4 pose sources, simulator shows the selected wrist camera model; fixed-camera module unchanged")
print("ALL V3.30 TESTS PASS")
