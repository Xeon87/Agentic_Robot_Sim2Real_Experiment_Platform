#!/usr/bin/env python3
"""V3.33: scene + perception changes that make the RGB-only USB wrist camera work.

S1 carton CAD: watertight outward mesh; cells never cut the design egg; plateau = physical post top (egg rests where
   it rested); posts between the rows keep >= 5 mm from an egg in any cell
S2 scene builder (AST-extracted, no MuJoCo needed): grey floor + plain white table (no egg-coloured / dark-vein
   textures), visual pulp-carton mesh, the physical carton hidden (group 3, alpha 0, still colliding), old wells
   gone, <visual> shadow 8192 + headlight (with and without an existing <visual>), NERO_V333_PULP_CARTON=0 = old look
S3 (numpy) RGB carton detection on lit synthetic renders (warm key / cool fill / headlight, table shadow, grey floor):
   every COMPLETE detection within 3 mm / 1 deg over oblique views and three carton poses; cut views never complete;
   no egg false positives anywhere; carton mask rejects table shadows and the floor
S4 (numpy) depth (D435i) carton detection: complete, slots within 1 mm, egg in a cell keeps the cell observed
S5 RGB-only detections are named after a second view at the same place; D435i detections at once; a blob that
   moves between views is never named
S6 D435i mounted on the USB side face; no wrist camera drawn in non-wrist modes; scan grid without the never
   reachable r=0.23 m points
S7 backend / Agent wiring: unreachable views remembered, holder search + 3 refine looks at the seen part,
   HOLDER_NOT_PERCEIVED (observe once more, then stop), guard reads scene_snapshot (no re-survey) and does not nudge
   without a carton, newest complete holder wins, task label names the USB camera
"""
import os as _os335; _os335.environ.setdefault("NERO_V335_CALIB_LEVEL", "0"); _os335.environ.setdefault("NERO_V335_REQUIRE_MEASURED", "0")   # V3.35: legacy tests assume an ideal camera, single-view trust
import ast, collections, hashlib, math, os, re, struct, sys, tempfile, zlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import NERO_V326_Wrist_Camera as W  # noqa: E402

assert W.VERSION in ("V3.33", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
A = (ROOT / "NERO_V2_7_1_Agent.py").read_text()

try:
    import numpy as np
    import NERO_V308_Camera_Perception as P8
    import v329_synth as S
    have_np = True
except Exception:
    have_np = False

# ------------------------------------------------------------------ S1
m = W.holder_model()
if have_np:
    obj = W.carton_mesh_obj()
    F = [tuple(map(int, l.split()[1:])) for l in obj.splitlines() if l.startswith("f ")]
    Vv = np.array([[float(x) for x in l.split()[1:]] for l in obj.splitlines() if l.startswith("v ")])
    E = collections.Counter()
    for a, b, c in F:
        for x, y in ((a, b), (b, c), (c, a)):
            E[(x, y)] += 1
    assert all(n == 1 and E.get((y, x)) == 1 for (x, y), n in E.items()), "mesh not closed / not consistently oriented"
    vol = 0.0
    for a, b, c in F:
        vol += float(np.dot(Vv[a - 1], np.cross(Vv[b - 1], Vv[c - 1]))) / 6.0
    assert 0.6e-3 < vol < 2.1e-3, vol                                  # positive -> outward normals
    a_, b_, zc = 0.0315, 0.0219, W.egg_center_in_slot_z(0.0)
    assert abs(zc - 0.054) < 0.001 and abs(m["plateau_z"] - m["post_top"]) < 1e-9
    worst = 1.0
    for su in m["slot_u"]:
        for sv in m["slot_v"]:
            rr = np.linspace(0.0, b_ * 0.999, 60)
            for ang in np.linspace(0, 2 * math.pi, 24, endpoint=False):
                z_c = W.carton_height(su + rr * math.cos(ang), sv + rr * math.sin(ang))
                z_e = zc - a_ * np.sqrt(1 - (rr / b_) ** 2)
                worst = min(worst, float(np.min(z_e - z_c)))
    assert worst >= -0.0002, worst                                     # the cell surface stays below the egg
    for pu in m["carton_posts_u"]:
        for su in m["slot_u"]:
            for sv in m["slot_v"]:
                d = (abs(pu - su) + abs(sv)) / math.sqrt(2.0) - m["carton_post_half"][0]
                assert d - b_ >= 0.005 or abs(pu - su) > 0.05, (pu, su, sv, d)
    print(f"[S1] carton CAD: watertight mesh ({len(F)} faces, volume {vol * 1e6:.0f} cm^3), cells below the egg "
          f"everywhere (min gap {worst * 1000:.2f} mm), plateau = physical post top, posts clear of the eggs")
else:
    print("[S1] numpy missing -> skipped")


# ------------------------------------------------------------------ S2
def _read_png(path):
    data = Path(path).read_bytes()
    pos, idat = 8, b""
    w = h = 0
    while pos < len(data):
        n = struct.unpack(">I", data[pos:pos + 4])[0]; tag = data[pos + 4:pos + 8]; body = data[pos + 8:pos + 8 + n]
        if tag == b"IHDR":
            w, h = struct.unpack(">II", body[:8])
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    rows = [raw[i * (3 * w + 1) + 1:(i + 1) * (3 * w + 1)] for i in range(h)]           # filter 0 rows (our writer)
    return np.frombuffer(b"".join(rows), np.uint8).reshape(h, w, 3)


if have_np:
    tree = ast.parse(B)
    g = {"np": np, "re": re, "math": math, "Path": Path, "os": os}
    g["_v343_scene_name"] = lambda: "fill_egg_holder"     # V3.43: default (egg) scene
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("make_grid_world_xml", "_v333_visual_quality"):
            exec(compile(ast.Module([node], []), "backend", "exec"), g)
    import xml.etree.ElementTree as ET
    with tempfile.TemporaryDirectory() as td:
        res = {}
        for variant, xml0 in (("plain", "<mujoco model='t'><worldbody/></mujoco>".replace("<worldbody/>", "<worldbody></worldbody>")),
                              ("vis", "<mujoco model='t'><visual><quality shadowsize='2048' offsamples='4'/></visual>"
                                      "<asset></asset><worldbody></worldbody></mujoco>")):
            d = Path(td) / variant; d.mkdir()
            p = d / "nero_motor.xml"; p.write_text(xml0)
            r = ET.parse(g["make_grid_world_xml"](str(p))).getroot()
            res[variant] = r
            q = r.find("visual/quality"); hl = r.find("visual/headlight")
            assert q.get("shadowsize") == "8192" and hl.get("ambient") == "0.25 0.25 0.25", ET.tostring(r.find("visual"))
            if variant == "vis":
                assert q.get("offsamples") == "4"
        r = res["plain"]
        geoms = {x.get("name"): x for x in r.iter("geom")}
        hidden = [n for n, x in geoms.items() if n and n.startswith("egg_holder_") and x.get("group") == "3"]
        assert len(hidden) == 53 and all(geoms[n].get("contype") == "1" and geoms[n].get("rgba") == "0 0 0 0" for n in hidden)
        assert not [n for n in geoms if n and n.endswith("_well")]
        pulp = geoms["egg_holder_pulp_visual"]
        assert pulp.get("type") == "mesh" and pulp.get("contype") == "0" and pulp.get("group") == "0"
        mesh = [x for x in r.iter("mesh") if x.get("name") == "eggscene_carton_mesh"][0]
        assert Path(mesh.get("file")).is_file()
        tex = {t.get("name"): t.get("file") for t in r.iter("texture")}
        floor, table = _read_png(tex["eggscene_wood_tex"]), _read_png(tex["eggscene_marble_tex"])
        assert P8.egg_color_mask(floor).mean() == 0.0 and P8.egg_color_mask(table).mean() == 0.0
        assert floor[..., 2].mean() > floor[..., 0].mean() + 15                     # cool grey, not wood
        assert table.std() < 4.0 and table.min() > 225                            # plain white, no dark veins
        os.environ["NERO_V333_PULP_CARTON"] = "0"
        d = Path(td) / "legacy"; d.mkdir(); p = d / "nero_motor.xml"; p.write_text("<mujoco><worldbody></worldbody></mujoco>")
        r2 = ET.parse(g["make_grid_world_xml"](str(p))).getroot()
        del os.environ["NERO_V333_PULP_CARTON"]
        g2 = {x.get("name"): x for x in r2.iter("geom")}
        assert "egg_holder_pulp_visual" not in g2 and g2["egg_holder_base"].get("group") is None
    print("[S2] scene: grey floor + plain white table (0 egg-coloured pixels), pulp-carton mesh visible, physical carton "
          "hidden but colliding (53 geoms), wells gone, shadow 8192 + headlight in a new or an existing <visual>")
else:
    print("[S2] numpy missing -> skipped")

# ------------------------------------------------------------------ S3
if have_np:
    cfg = W.wrist_config("usb130"); K = P8.intrinsics(cfg); pc, fc = W.camera_in_tcp(cfg)
    EGGS = [([-0.451, -0.181, 0.022], 0.3), ([-0.399, -0.181, 0.023], -0.5), ([-0.452, -0.130, 0.021], 1.1),
            ([-0.399, -0.129, 0.023], 2.0)]
    n_complete = n_views = 0
    worst = 0.0
    for hx, hy, hyaw in ((-0.42, 0.155, 0.0), (-0.40, 0.17, math.radians(10)), (-0.45, 0.13, math.radians(-25))):
        tr = S.slot_truth(hx, hy, hyaw)
        base = ([S.table_rgb_part(shadows=[(hx - 0.13, hx + 0.05, hy - 0.06, hy - 0.01), (-0.50, -0.42, -0.25, -0.20)]),
                 S.carton_rgb_part(hx, hy, hyaw), S.egg_rgb_part([tr["slot_3"][0], tr["slot_3"][1], 0.054], False)]
                + [S.egg_rgb_part(c, True, y) for c, y in EGGS])
        for pt in ([hx, hy, 0.043], [-0.44, 0.105, 0.02], [-0.44, 0.315, 0.02]):
            cands = W.look_pose_candidates(pt, cfg=cfg, distances=[0.45])
            for c in [x for x in cands if abs(x["tilt_deg"]) == 30][:2] + cands[:1]:
                T = S.camera_T_for(c, pc, fc)
                rgb = S.render_rgb(base, K, T)
                h = W.detect_holder_rgb(rgb, K, T)
                n_views += 1
                for d in W.perceive_eggs_rgb(rgb, K, T, holder=h if h and h["complete"] else None):
                    e = min([math.dist(d["xyz_m"][:2], q[0][:2]) for q in EGGS] + [math.dist(d["xyz_m"][:2], tr["slot_3"][:2])])
                    assert e < 0.04, ("egg false positive", d["xyz_m"], c["tag"])
                if h and h["complete"]:
                    n_complete += 1
                    err = max(math.dist(h["slots"][k]["xyz_m"][:2], tr[k][:2]) for k in tr)
                    dy = abs(math.degrees(math.atan2(math.sin(h["yaw_rad"] - hyaw), math.cos(h["yaw_rad"] - hyaw))))
                    worst = max(worst, err)
                    assert err < 0.003 and dy < 1.0, (hx, hy, c["tag"], err, dy)
    assert n_complete >= n_views // 2, (n_complete, n_views)
    T = S.camera_T_for(W.look_pose_candidates([-0.42, -0.02, 0.03], cfg=cfg, distances=[0.30])[0], pc, fc)
    hp = W.detect_holder_rgb(S.render_rgb([S.table_rgb_part(), S.carton_rgb_part()], K, T), K, T)
    assert hp is None or not hp["complete"]
    lit_tab, shad_tab = S.shade(S.ALB_TABLE, np.array([[0, 0, 1.0]])), S.shade(S.ALB_TABLE, np.array([[0, 0, 1.0]]), [True])
    floor = np.array([S.floor_rgb()])
    pulp = S.shade(S.ALB_PULP, np.array([[0, 0, 1.0], [0.6, -0.3, 0.74], [-0.6, 0.3, 0.74]]))
    img = np.vstack([np.repeat(lit_tab, 50, 0), np.repeat(shad_tab, 50, 0), np.repeat(floor, 50, 0), np.repeat(pulp, 50, 0)])
    msk = W.carton_mask(np.clip(img, 0, 255).astype(np.uint8)[None])[0]
    assert not msk[:150].any() and msk[150:].mean() > 0.6
    assert not P8.egg_color_mask(np.clip(pulp, 0, 255).astype(np.uint8)[None]).any()
    print(f"[S3] RGB carton: {n_complete}/{n_views} oblique/scan views complete, all within {worst * 1000:.1f} mm and 1 deg "
          "(far-edge rule); cut view never complete; no egg false positives; mask rejects table shadow + floor")
else:
    print("[S3] numpy missing -> skipped")

# ------------------------------------------------------------------ S4
if have_np:
    cd = W.wrist_config("d435i"); Kd = P8.intrinsics(cd); pcd, fcd = W.camera_in_tcp(cd)
    hx, hy, hyaw = -0.38, 0.20, math.radians(15)
    tr = S.slot_truth(hx, hy, hyaw)
    Pall = np.vstack([S.table_points(), S.holder_points(hx, hy, hyaw), S.egg_points([tr["slot_1"][0], tr["slot_1"][1], 0.054], False)])
    for c in W.look_pose_candidates([hx, hy, 0.043], cfg=cd, distances=[0.45])[:3]:
        T = S.camera_T_for(c, pcd, fcd)
        h = W.detect_holder(S.render_depth(Pall, Kd, T), Kd, T,
                            dets=[{"xyz_m": [tr["slot_1"][0], tr["slot_1"][1], 0.054], "axis_world": [0, 0, 1]}])
        err = max(math.dist(h["slots"][k]["xyz_m"][:2], tr[k][:2]) for k in tr)
        assert h["complete"] and err < 0.0015 and h["slots"]["slot_1"]["post_sectors"] >= 3, (h, err)
        assert abs(h["slots"]["slot_4"]["xyz_m"][2] - 0.043) < 0.002
    print(f"[S4] depth carton (D435i): complete, slot err {err * 1000:.2f} mm, the cell with an egg still observed")

# ------------------------------------------------------------------ S5
if have_np:
    tr5 = W.ViewTracker(P8.PerceptionTracker(), names=["egg_1", "egg_2", "egg_3"])
    Kx = P8.intrinsics(W.wrist_config("d435i"))
    Td = [[1, 0, 0, -0.42], [0, -1, 0, -0.15], [0, 0, -1, 0.40], [0, 0, 0, 1]]

    def det(i, p, rgb=True):
        return {"det_id": i, "xyz_m": list(p), "quat_xyzw": [0, 0, 0, 1], "axis_world": [1, 0, 0], "location": "x",
                "trusted": True, "quality": {"rgb_only": True} if rgb else {}}
    o1, g1 = tr5.update([det(0, [-0.40, -0.15, 0.022]), det(1, [-0.50, -0.10, 0.022])], 1, Kx, Td)
    assert not g1["named"] and g1["awaiting_second_view"] == 2
    o2, g2 = tr5.update([det(0, [-0.401, -0.152, 0.022]), det(1, [-0.44, -0.20, 0.022])], 2, Kx, Td)
    assert g2["named"] == ["egg_1"] and math.dist(o2["egg_1"]["xyz_m"][:2], [-0.401, -0.152]) < 0.003, (g2, o2)
    o3, g3 = tr5.update([det(0, [-0.401, -0.152, 0.022]), det(1, [-0.47, -0.25, 0.022], rgb=False)], 3, Kx, Td)   # depth: at once
    assert g3["named"] == ["egg_2"]
    print("[S5] RGB-only eggs named after a 2nd view at the same place (a jumping blob never), depth eggs at once")

# ------------------------------------------------------------------ S6
cu, cd6 = W.wrist_config("usb130"), W.wrist_config("d435i")
assert cu["mount_pos"][1] < 0 and cd6["mount_pos"][1] < 0
assert '"NERO_V330_SHOW_WRIST_MODEL", "none"' in B and 'get("show", "none")' in B
assert all(math.hypot(p[0], p[1]) >= 0.24 for _n, p in W.scan_points())
print(f"[S6] D435i {cd6['mount_pos']} and USB {cu['mount_pos']} on the same (-y) side face; no wrist camera drawn in "
      f"non-wrist modes; scan grid {len(W.scan_points())} views, none closer than 0.24 m")

# ------------------------------------------------------------------ S7
for mk in ('"unreachable": set(), "holder_miss": 0', "[V3.33 SCAN SKIP]", "unreach.add(key333)", 'holder_search_{k + 1}',
           '(h2 or h_).get("look_xy_m") or (h2 or h_)["center_xy_m"]', "for vname_, d_in, d_ax, dists_ in plan:", "V3.33 HOLDER_NOT_PERCEIVED",
           '["observe_scene"] if miss < 2 else ["stop_and_report"]', 'if name == "scene_snapshot":',
           '(h.get("complete") or _w.holder_quality(h) >= _w.holder_quality(_V329_SCENE["holder"]))',
           "wrist camera USB 130deg RGB-only perception (V3.30)", "_v333_visual_quality(xml)"):
    assert mk in B, mk
for mk in ('self.tool_runner("scene_snapshot", {})', "[V3.33 GUARD] egg carton not perceived", "V3.33 NO CARTON"):
    assert mk in A, mk
WM = (ROOT / "NERO_V326_Wrist_Camera.py").read_text()
assert 'holder.get("complete") or holder_quality(holder) >= holder_quality(holder_best[pf]))' in WM
h8 = hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest()
assert h8.startswith("020cc6680f181872"), h8
print("[S7] wiring: unreachable views remembered, holder search + refine at the seen part, stop after two misses, guard "
      "without re-survey, newest complete holder wins, USB label; V3.08 module unchanged")
print("ALL V3.33 TESTS PASS")
