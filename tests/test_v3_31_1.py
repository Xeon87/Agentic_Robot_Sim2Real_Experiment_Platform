#!/usr/bin/env python3
"""V3.31.1 wrist cameras mounted ON the gripper base (side faces), not out along the finger rail.

M1 nominal (no STL): D435i and USB both on the -y side face (V3.33), centred on the gripper axis height (z = -0.0235), i.e.
   perpendicular to the finger travel (link7 z); optical axis crosses the gripper axis 0.40 m ahead
M2 synthetic gripper_base.stl (binary, mesh frame of the backend geom): faces read from the mesh, every camera
   model geom outside the base, the bracket touches both the base face and the camera back
M3 env override (NERO_V326_MOUNT_POS ...) still wins; injection writes the bracket geoms
"""
import math, os, struct, sys, tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V326_Wrist_Camera as W  # noqa: E402

assert W.VERSION in ("V3.31.1", "V3.33", "V3.35", "V3.36", "V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
zc = W.GRIPPER_AXIS_Z


def corners(g):
    """World (link7) corners of a box / cylinder-bounding-box geom from the model dict."""
    p = [float(v) for v in g["pos"].split()]
    sz = [float(v) for v in g["size"].split()]
    if g["type"] == "cylinder":
        sz = [sz[0], sz[0], sz[1]]
    if "xyaxes" in g:
        a = [float(v) for v in g["xyaxes"].split()]
        ax, ay = a[:3], a[3:]
        az = [ax[1] * ay[2] - ax[2] * ay[1], ax[2] * ay[0] - ax[0] * ay[2], ax[0] * ay[1] - ax[1] * ay[0]]
    else:
        ax, ay, az = [1, 0, 0], [0, 1, 0], [0, 0, 1]
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz_ in (-1, 1):
                out.append([p[i] + sx * sz[0] * ax[i] + sy * sz[1] * ay[i] + sz_ * sz[2] * az[i] for i in range(3)])
    return out


def check(ext):
    for pf, side in (("d435i", -1), ("usb130", -1)):              # V3.33: D435i on the USB side face
        c = W.wrist_config(pf)
        p = c["mount_pos"]
        assert abs(p[2] - zc) < 1e-6 and side * p[1] > 0, (pf, p)            # side face, gripper-axis height
        x, _y, f = W.mount_axes(c)
        t = -(p[1]) / f[1]                                                   # where the axis crosses y = 0
        assert 0.25 < t < 0.40 and abs(p[2] + t * f[2] - zc) < 1e-6, (pf, t)
        assert abs(abs(x[2]) - 1.0) < 1e-6                                   # image rows along the finger rail
        face = ext["y_face_pos"] if side > 0 else -ext["y_face_neg"]
        for g in W.camera_model_geoms(c):
            ys = [side * q[1] for q in corners(g)]
            if g["name"].endswith("bracket"):
                assert min(ys) <= face + 1e-6, (pf, "bracket must reach the base", min(ys), face)
            else:
                assert min(ys) > face, (pf, g["name"], min(ys), face)       # outside the base
        br = [g for g in W.camera_model_geoms(c) if g["name"].endswith("bracket")][0]
        body = [g for g in W.camera_model_geoms(c) if g["name"].endswith(("housing", "pcb"))][0]
        assert max(side * q[1] for q in corners(br)) >= min(side * q[1] for q in corners(body)) - 1e-6
    return True


# ------------------------------------------------------------------ M1
os.environ.pop("NERO_V3311_GRIPPER_BASE_STL", None)
W._BASE_CACHE.clear()
os.environ["NERO_V3311_MOUNT_AUTO"] = "0"
ext, src = W.gripper_base_extent()
assert src.startswith("nominal") and check(ext)
print(f"[M1] nominal: D435i {W.wrist_config('d435i')['mount_pos']} (-y face), USB {W.wrist_config('usb130')['mount_pos']} "
      "(-y face), on the gripper axis height, perpendicular to the finger travel")

# ------------------------------------------------------------------ M2
os.environ["NERO_V3311_MOUNT_AUTO"] = "1"
R = W._quat_R(W.GRIPPER_BASE_QUAT); P = W.GRIPPER_BASE_POS
Rt = [[R[j][i] for j in range(3)] for i in range(3)]
box = [(x, y, z) for x in (0.0375, 0.095) for y in (-0.041, 0.044) for z in (zc - 0.11, zc + 0.11)]
mesh = [[sum(Rt[i][k] * (c[k] - P[k]) for k in range(3)) * 1000.0 for i in range(3)] for c in box]   # mm file
tris = [(0, 1, 2), (1, 2, 3), (4, 5, 6), (5, 6, 7), (0, 4, 1), (1, 5, 4)]
d = tempfile.mkdtemp(prefix="v3311_")
stl = Path(d) / "gripper_base.stl"
with open(stl, "wb") as fh:
    fh.write(b"\0" * 80 + struct.pack("<I", len(tris)))
    for t in tris:
        fh.write(struct.pack("<12fH", 0, 0, 0, *mesh[t[0]], *mesh[t[1]], *mesh[t[2]], 0))
os.environ["NERO_V3311_GRIPPER_BASE_STL"] = str(stl)
W._BASE_CACHE.clear()
ext, src = W.gripper_base_extent()
assert "gripper_base.stl" in src and abs(ext["y_face_pos"] - 0.044) < 1e-5 and abs(ext["y_face_neg"] + 0.041) < 1e-5
assert abs(ext["x_max"] - 0.095) < 1e-5 and check(ext)
assert abs(W.wrist_config("d435i")["mount_pos"][0] - 0.095) < 1e-4                  # front flush with the base
print(f"[M2] faces from gripper_base.stl (mm, mesh frame): +y {ext['y_face_pos']:.3f} / -y {ext['y_face_neg']:.3f} m, "
      f"front x {ext['x_max']:.3f} m -> cameras flush on the faces, bracket attached, nothing inside the base")

# ------------------------------------------------------------------ M3
xml = Path(d) / "m.xml"
xml.write_text("<mujoco><worldbody><body name='link7'/></worldbody></mujoco>")
r = ET.parse(W.inject_wrist_camera_into_mjcf(str(xml))).getroot()
names = {g.get("name") for g in r.findall(".//body[@name='link7']/geom")}
assert {"v330_cam_d435i_bracket", "v330_cam_usb130_bracket"} <= names
os.environ["NERO_V326_MOUNT_POS"] = "0.06,0.0,0.07"
c = W.wrist_config("d435i")
assert c["mount_pos"] == [0.06, 0.0, 0.07] and c["mount_source"] == "env override" and c["bracket"] is None
del os.environ["NERO_V326_MOUNT_POS"]
print("[M3] injection writes the brackets; NERO_V326_MOUNT_POS / NERO_V330_USB_MOUNT_POS still override the mount")
print("ALL V3.31.1 TESTS PASS")
