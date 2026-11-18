#!/usr/bin/env python3
"""NERO V4.11.2 DOE SNAPSHOT - pictures of the final state of a DOE run, rendered OUTSIDE the physics loop.

The physics process only writes what a renderer needs when the DOE asks for the truth check (its own model saved
once as MJCF next to the generated scene file + qpos; NERO_V410_Household_Scenes.snapshot_state).  The console then
runs this module as a separate process: it loads that model, sets qpos, mj_forward (no stepping) and renders

  overview   oblique view of the table from the front (the side away from the robot)
  top        straight down on the table
  camera     the simulated fixed D435i (v308_d435i), when the model has it

into PNG files (stdlib zlib writer, no PIL).  OpenGL backends are tried in the order of MUJOCO_GL or egl, osmesa,
glfw - each in its own child process, so a backend that crashes cannot take the console with it.

    python3 NERO_V411_Snapshot.py --xml <model.xml> --state <state.json with "qpos"> --out <prefix> [--size 960x720]
prints one JSON line {"ok", "files", "gl", "why"}.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import zlib

VERSION = "V4.11.2"
MARKER = "V4.11.2 DOE SNAPSHOT"
HERE = os.path.dirname(os.path.realpath(__file__))
TABLE_CENTER = (-0.385, 0.0, 0.06)
CAMERA_NAMES = ("v308_d435i",)
VIEWS = {                                   # free cameras: lookat, distance, azimuth, elevation (MuJoCo degrees)
    "overview": (TABLE_CENTER, 1.35, 0.0, -38.0),        # from the far side of the table, the robot behind
    "top": ((TABLE_CENTER[0], TABLE_CENTER[1], 0.0), 1.15, 0.0, -89.0),
}


def write_png(path, rgb):
    """rgb: HxWx3 uint8 array -> PNG (zlib, filter 0)"""
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(png)
    os.replace(tmp, path)


def render(xml, qpos, out_prefix, width=960, height=720):
    """-> list of written files (this process; MUJOCO_GL must already be set)"""
    import numpy as np
    import mujoco
    m = mujoco.MjModel.from_xml_path(xml)
    if len(qpos) != m.nq:
        raise ValueError(f"qpos has {len(qpos)} values, the model nq = {m.nq}")
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), width)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), height)
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(qpos, float)
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, height=height, width=width)
    files = []
    for name, (look, dist, az, el) in VIEWS.items():
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = look
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        r.update_scene(d, camera=cam)
        p = f"{out_prefix}_{name}.png"
        write_png(p, r.render())
        files.append(p)
    for cn in CAMERA_NAMES:
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cn) >= 0:
            r.update_scene(d, camera=cn)
            p = f"{out_prefix}_camera.png"
            write_png(p, r.render())
            files.append(p)
            break
    return files


def _child(a):
    os.environ["MUJOCO_GL"] = a.gl
    if a.gl in ("egl", "osmesa"):
        os.environ["PYOPENGL_PLATFORM"] = a.gl
    with open(a.state) as f:
        st = json.load(f)
    w, h = (int(v) for v in a.size.lower().split("x"))
    files = render(a.xml, st["qpos"], a.out, w, h)
    print(json.dumps({"ok": True, "files": files, "gl": a.gl}), flush=True)


def snapshot(xml, qpos, out_prefix, size="960x720", timeout_s=60.0, python=None) -> dict:
    """run the renderer in child processes (GL backends in order) -> {"ok", "files", "gl", "why"}"""
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    st = out_prefix + "_state.json"
    with open(st, "w") as f:
        json.dump({"qpos": list(qpos)}, f)
    order = [os.environ["MUJOCO_GL"]] if os.environ.get("MUJOCO_GL") else ["egl", "osmesa", "glfw"]
    whys = []
    for gl in order:
        try:
            p = subprocess.run([python or sys.executable, os.path.realpath(__file__), "--child", "--gl", gl, "--xml",
                                xml, "--state", st, "--out", out_prefix, "--size", size],
                               capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            whys.append(f"{gl}: timeout")
            continue
        for line in reversed((p.stdout or "").splitlines()):
            if line.startswith("{"):
                try:
                    res = json.loads(line)
                except ValueError:
                    break
                if res.get("ok"):
                    return res
        tail = ((p.stderr or "") + (p.stdout or "")).strip().splitlines()[-1:] or ["no output"]
        whys.append(f"{gl}: {tail[0][:160]}")
    return {"ok": False, "files": [], "gl": None, "why": "; ".join(whys)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", default="960x720")
    ap.add_argument("--gl", default="egl")
    ap.add_argument("--child", action="store_true")
    a = ap.parse_args(argv)
    if a.child:
        return _child(a)
    with open(a.state) as f:
        qpos = json.load(f)["qpos"]
    print(json.dumps(snapshot(a.xml, qpos, a.out, a.size)), flush=True)


if __name__ == "__main__":
    main()
