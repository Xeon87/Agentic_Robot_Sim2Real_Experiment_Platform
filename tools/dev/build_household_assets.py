#!/usr/bin/env python3
"""Dev tool (not used at runtime): build assets/household/ from downloaded GSO / YCB scans (V4.10).

    python3 tools/dev/build_household_assets.py <download dir> [<release dir>]

<download dir> holds gso/<model name>/model.obj + texture.png (Gazebo Fuel, owner GoogleResearch) and
ycb/x/013_apple/google_16k/textured.obj + texture_map.png (YCB google_16k).  For each item:
  - the mesh is scaled to the size listed in ITEMS, rotated (lying bottle) and moved so that the footprint centre is
    at x = y = 0 and the lowest point at z = 0 (the body frames in NERO_V410_Household_Scenes add the origin offset)
  - the texture is resized to 1024 px (the release stays small)
  - solid items (apple, bottle): CoACD convex decomposition -> hull_<k>.stl (collision); the scan is visual only
  - thin-walled items (bowl, mug, tray, bin): no hulls; the scene module builds their collision from primitives
    with the measured sizes written to meta.json (convex parts of thin walls are either very many or too thick)
Needs: trimesh, coacd, pillow, numpy.
"""
import json
import math
import os
import sys

import numpy as np

ITEMS = {
    # name: source, model path, scale, rotate, solid
    "bowl": {"src": "GSO", "model": "Room_Essentials_Bowl_Turquiose", "scale": 1.0, "rot": None, "solid": False},
    "mug": {"src": "GSO", "model": "Room_Essentials_Mug_White_Yellow", "scale": 1.0, "rot": None, "solid": False},
    "tray": {"src": "GSO", "model": "Threshold_Tray_Rectangle_Porcelain", "scale": 0.75, "rot": None, "solid": False},
    "storage_box": {"src": "GSO", "model": "Curver_Storage_Bin_Black_Small", "scale": 0.70, "rot": None, "solid": False},
    "bottle": {"src": "GSO", "model": "Weston_No_33_Signature_Sausage_Tonic_12_fl_oz", "scale": 0.82, "rot": "lying",
               "solid": True},
    "apple": {"src": "YCB", "model": "013_apple", "scale": 0.78, "rot": None, "solid": True},
}
LICENSE = {"GSO": "Google Scanned Objects, CC BY 4.0 (Google Research), via Gazebo Fuel (owner GoogleResearch)",
           "YCB": "YCB Object and Model Set (google_16k scan), CC BY 4.0, ycbbenchmarks.com"}
TEX_PX = 1024


def _paths(dl, it):
    if it["src"] == "GSO":
        d = os.path.join(dl, "gso", it["model"])
        return os.path.join(d, "model.obj"), os.path.join(d, "texture.png")
    d = os.path.join(dl, "ycb", "x", it["model"], "google_16k")
    return os.path.join(d, "textured.obj"), os.path.join(d, "texture_map.png")


def _write_obj(path, V, VT, F):
    with open(path, "w") as f:
        f.write("# V4.10 household asset (scaled / centred scan)\n")
        for v in V:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in VT:
            f.write(f"vt {t[0]:.6f} {t[1]:.6f}\n")
        for a, b, c in F + 1:
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")


def build(dl, out_root):
    import coacd
    import trimesh
    from PIL import Image
    os.makedirs(out_root, exist_ok=True)
    for name, it in ITEMS.items():
        obj, tex = _paths(dl, it)
        m = trimesh.load(obj, force="mesh", process=False)
        V = np.asarray(m.vertices, float) * float(it["scale"])
        uv = getattr(m.visual, "uv", None)
        if uv is None or len(uv) != len(V):
            raise RuntimeError(f"{name}: the scan has no per-vertex uv")
        if it["rot"] == "lying":                       # long axis z -> x, lying on its side
            V = V[:, [2, 1, 0]] * np.array([1.0, 1.0, -1.0])
        lo, hi = V.min(0), V.max(0)
        V = V - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        F = np.asarray(m.faces, int)
        if it["rot"] == "lying":
            F = F[:, [0, 2, 1]]                        # the axis swap mirrored the mesh: keep the normals outward
        d = os.path.join(out_root, name)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):                       # no stale convex parts from an earlier build
            if f.startswith("hull_"):
                os.remove(os.path.join(d, f))
        _write_obj(os.path.join(d, "visual.obj"), V, np.asarray(uv, float), F)
        Image.open(tex).convert("RGB").resize((TEX_PX, TEX_PX), Image.LANCZOS).save(os.path.join(d, "texture.png"),
                                                                                   optimize=True)
        size = (V.max(0) - V.min(0)).tolist()
        meta = {"name": name, "source": it["src"], "model": it["model"], "license": LICENSE[it["src"]],
                "scale": it["scale"], "size_m": [round(s, 4) for s in size], "hulls": 0}
        # horizontal profile (outer half widths at 8 heights), used by the scene module for thin-walled items
        prof = []
        for z in np.linspace(0.002, size[2] - 0.002, 8):
            b = V[np.abs(V[:, 2] - z) < 0.003]
            if len(b):
                prof.append([round(float(z), 4), round(float(b[:, 0].min()), 4), round(float(b[:, 0].max()), 4),
                             round(float(b[:, 1].min()), 4), round(float(b[:, 1].max()), 4)])
        meta["profile_z_xmin_xmax_ymin_ymax"] = prof
        if it["solid"]:
            mesh = coacd.Mesh(V, F)
            parts = coacd.run_coacd(mesh, threshold=0.08, max_convex_hull=8, preprocess_mode="auto")
            for k, (pv, pf) in enumerate(parts):
                trimesh.Trimesh(pv, pf).export(os.path.join(d, f"hull_{k}.stl"))
            meta["hulls"] = len(parts)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"{name:12s} {it['src']} {it['model'][:40]:40s} size_mm={np.round(np.array(size) * 1000).astype(int)} "
              f"faces={len(F)} hulls={meta['hulls']}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    build(sys.argv[1], os.path.join(sys.argv[2] if len(sys.argv) > 2 else here, "assets", "household"))
