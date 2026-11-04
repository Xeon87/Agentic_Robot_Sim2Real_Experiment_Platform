#!/usr/bin/env python3
"""V3.78 offline tests (D435i bench, no camera here - a fake RealSense device + synthetic images):
M1 Versions V3.78 + marker; frozen modules unchanged; the backend and the Camera tab are not changed by V3.78.
M2 Profile: defaults = auto exposure / auto white balance OFF, 60 Hz anti-flicker, 1280x720 High Accuracy depth;
   save -> NERO_CALIB_DIR/d435i_profile.json -> load round trip; nothing is written into the release folder.
M3 apply_profile on a fake device: auto modes are switched off BEFORE the manual values, values are clamped to the
   sensor range and snapped to its step, the depth preset name is mapped, unsupported options are reported.
M4 Depth statistics: median_stack ignores holes (0); noise_report on a synthetic flat target (sigma 1.5 mm):
   temporal std ~ 1.5 mm raw, ~3x smaller with a 10-frame median; flatness of a tilted plane ~ 0; valid %.
M5 Board frame: a ChArUco 7x5 / 32 mm board rendered at a known pose -> board_pose() gives the camera pose in the
   bench frame (board centre, z up): board points within 1 mm, tilt within 0.5 deg; the inner corners project back
   onto the image corners.
M6 Accuracy maths: a located egg centre -> nearest corner, dx / dy / dz against half the egg width, summary
   (mean / std / max / p90), CSV + JSON written under NERO_CALIB_DIR/d435i_bench.
M7 Release folder unchanged.
"""
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v378_"))
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
os.environ["NERO_V378_DATA_DIR"] = str(TMP / "datasets")
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
H = lambda b: hashlib.sha256(b).hexdigest()


def snapshot():
    out = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rel = line.split(None, 1)[1].strip()
            out[rel] = H((ROOT / rel).read_bytes()) if (ROOT / rel).is_file() else None
    return out


SNAP0 = snapshot()
import numpy as np                          # noqa: E402
import NERO_V312_Console as CON             # noqa: E402
import NERO_V313_Real_Robot as RR           # noqa: E402
import NERO_V347_Yolo_Perception as YP      # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49   # noqa: E402
import NERO_V378_D435i_Bench as B           # noqa: E402

# ------------------------------------------------------------------ M1
for m in (CON, RR, YP, W49, B):
    assert m.VERSION == "V3.78", (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() == "V3.78"
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
for fn, h in (("NERO_V2_7_1_Unified_Backend.py", "5be2136f84ec4ccdfe6f8bc8d8998410ff9b3b827af4d571ea44373dc427527e"),
              ("NERO_V325_Camera.py", "a58e69b363a5a861b3ad0806ba276c49601fc214ffdfc07ffba9bdc0c2c65797"),
              ("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
assert B.MARKER in (ROOT / "NERO_V378_D435i_Bench.py").read_text(encoding="utf-8")
assert B.CALIB_DIR == str(TMP / "cameracalib") and B.DATA_DIR == str(TMP / "datasets")
print("[M1] V3.78; frozen modules, backend and Camera tab unchanged")

# ------------------------------------------------------------------ M2
p0 = B.load_profile()
assert p0["color"]["enable_auto_exposure"] == 0 and p0["color"]["enable_auto_white_balance"] == 0
assert p0["color"]["power_line_frequency"] == 2 and p0["depth"]["visual_preset"] == "high_accuracy"
assert B.COLOR_SIZE == (1280, 720) and B.DEPTH_SIZE == (1280, 720)
p0["color"]["exposure"] = 200.0
p0["locked"] = True
path = B.save_profile(p0)
assert path == str(TMP / "cameracalib" / "d435i_profile.json")
p1 = B.load_profile()
assert p1["color"]["exposure"] == 200.0 and p1["locked"] and p1["saved_at"]
print("[M2] profile defaults (AE / AWB off, 60 Hz, 1280x720 High Accuracy); saved in NERO_CALIB_DIR and reloaded")

# ------------------------------------------------------------------ M3


class _Range:
    def __init__(self, lo, hi, step):
        self.min, self.max, self.step = lo, hi, step


class _Sensor:
    def __init__(self, name, opts):
        self.name, self.opts, self.calls = name, opts, []

    def get_info(self, _k):
        return self.name

    def supports(self, opt):
        return opt in self.opts

    def get_option_range(self, opt):
        return self.opts[opt][1]

    def set_option(self, opt, v):
        self.calls.append((opt, v))
        self.opts[opt][0] = v

    def get_option(self, opt):
        return self.opts[opt][0]


class _Dev:
    def __init__(self, sensors):
        self.s = sensors

    def query_sensors(self):
        return self.s


class _Opt:
    pass


class _RS:
    class camera_info:
        name = "name"

    option = _Opt()
    rs400_visual_preset = type("P", (), {"high_accuracy": 3})


for k in ("enable_auto_exposure", "exposure", "gain", "enable_auto_white_balance", "white_balance",
          "power_line_frequency", "laser_power", "visual_preset"):
    setattr(_RS.option, k, k)
col = _Sensor("RGB Camera", {"enable_auto_exposure": [1, _Range(0, 1, 1)], "exposure": [100, _Range(1, 10000, 1)],
                             "gain": [64, _Range(0, 128, 1)], "enable_auto_white_balance": [1, _Range(0, 1, 1)],
                             "white_balance": [4000, _Range(2800, 6500, 10)],
                             "power_line_frequency": [3, _Range(0, 3, 1)]})
dep = _Sensor("Stereo Module", {"visual_preset": [0, _Range(0, 5, 1)], "enable_auto_exposure": [1, _Range(0, 1, 1)]})
prof = B.load_profile()
prof["color"].update(exposure=20000.0, white_balance=4604.0)
logs = []
rows = B.apply_profile(_Dev([col, dep]), prof, _RS, log=logs.append)
names = [c[0] for c in col.calls]
assert names.index("enable_auto_exposure") < names.index("exposure")
assert names.index("enable_auto_white_balance") < names.index("white_balance")
assert col.opts["exposure"][0] == 10000 and col.opts["white_balance"][0] == 4600 and col.opts["power_line_frequency"][0] == 2
assert dep.opts["visual_preset"][0] == 3.0
lp = [r for r in rows if r[1] == "laser_power"][0]
assert lp[3] is None and lp[4] == "not supported" and "not applied" in logs[0] and "laser_power" in logs[0]
assert any("clamped" in r[4] for r in rows if r[1] == "exposure")
print("[M3] apply_profile: auto off first, exposure clamped to 10000, WB snapped to 4600, preset mapped, "
      "laser power reported as not supported")

# ------------------------------------------------------------------ M4
rng = np.random.default_rng(378)
K = {"width": 1280, "height": 720, "fx": 910.0, "fy": 910.0, "ppx": 640.0, "ppy": 360.0}
vv, uu = np.mgrid[0:720, 0:1280]
plane = 0.70 + 0.0002 * (uu - 640) / 10.0                        # a tilted flat target, ~0.70 m
frames = []
for i in range(40):
    f = (plane + rng.normal(0, 0.0015, plane.shape)).astype(np.float32)
    f[rng.random(plane.shape) < 0.05] = 0.0                       # 5 % holes
    frames.append(f)
med = B.median_stack(frames[:10])
assert (med > 0).mean() > 0.999
roi = [560, 280, 720, 440]
raw = B.noise_report(frames, K, roi)
m10 = B.noise_report([B.median_stack(frames[i:i + 10]) for i in range(0, 40, 10)] * 1, K, roi)
assert 1.3 < raw["temporal_std_mm"] < 1.7 and 93 < raw["valid_pct"] < 97, raw
clean = B.noise_report([plane.astype(np.float32)] * 5, K, roi)
assert clean["flatness_rms_mm"] < 0.05 and clean["temporal_std_mm"] == 0.0, clean
assert m10["temporal_std_mm"] < raw["temporal_std_mm"] / 2.0, (raw, m10)
print(f"[M4] noise: raw std {raw['temporal_std_mm']} mm, 10-frame median {m10['temporal_std_mm']} mm; holes ignored; "
      f"flat target flatness {clean['flatness_rms_mm']} mm")

# ------------------------------------------------------------------ M5
cv2 = YP._cv2()
cfg = B.board_spec()
sq, mk = cfg["square_mm"] / 1000.0, cfg["marker_mm"] / 1000.0
ad = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["dict"]))
board = cv2.aruco.CharucoBoard((cfg["cols"], cfg["rows"]), sq, mk, ad)
ppm = 10000.0                                                    # board image: 10 px / mm
bimg = board.generateImage((int(cfg["cols"] * sq * ppm), int(cfg["rows"] * sq * ppm)), marginSize=0)


def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


R = rx(math.radians(25.0))
c_board = np.array([0.5 * cfg["cols"] * sq, 0.5 * cfg["rows"] * sq, 0.0])
t = -R @ c_board + np.array([0.01, -0.02, 0.62])
Km = np.array([[K["fx"], 0, K["ppx"]], [0, K["fy"], K["ppy"]], [0, 0, 1.0]])
Hm = Km @ np.column_stack([R[:, 0], R[:, 1], t]) @ np.diag([1 / ppm, 1 / ppm, 1.0])
img = cv2.warpPerspective(cv2.cvtColor(bimg, cv2.COLOR_GRAY2RGB), Hm, (1280, 720), borderValue=(200, 200, 200))
T, info = B.board_pose([img, img, img], K)
assert T is not None, info
rvec, _ = cv2.Rodrigues(R)
T_true = B.T_bench_cam_from_pose(rvec.ravel(), R @ c_board + t)
# what the measurement uses: board points expressed in the camera (a 0.25 deg tilt error moves the CAMERA position by
# ~2.6 mm at 0.6 m, but points on the board stay where they are)
Tc_true = np.linalg.inv(T_true)
dp = max(float(np.linalg.norm((np.linalg.inv(T) @ np.array([x, y, 0, 1.0]) - Tc_true @ np.array([x, y, 0, 1.0]))[:3]))
         for x, y in B.inner_corners().values()) * 1000.0
dR = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(T[:3, :3] @ T_true[:3, :3].T) - 1) / 2))))
assert dp < 1.0 and dR < 0.5, (dp, dR, info)
assert T[2, 3] > 0.5 and abs(info["cam_height_mm"] - T[2, 3] * 1000) < 0.1      # camera above the board (z up)
Tc = np.linalg.inv(T)
corners = B.inner_corners()
assert len(corners) == (cfg["cols"] - 1) * (cfg["rows"] - 1)
errs = []
for cid, (x, y) in corners.items():
    pc = Tc @ np.array([x, y, 0.0, 1.0])
    u, v = K["fx"] * pc[0] / pc[2] + K["ppx"], K["fy"] * pc[1] / pc[2] + K["ppy"]
    r_, c_ = divmod(cid, cfg["cols"] - 1)
    q = Hm @ np.array([(c_ + 1) * sq * ppm, (r_ + 1) * sq * ppm, 1.0])
    errs.append(math.hypot(u - q[0] / q[2], v - q[1] / q[2]))
assert max(errs) < 1.5, max(errs)
print(f"[M5] board pose from a rendered board: board points within {dp:.2f} mm, tilt {dR:.2f} deg; camera {info['cam_height_mm']} mm above; "
      f"{len(corners)} inner corners reproject within {max(errs):.2f} px")

# ------------------------------------------------------------------ M6
cx, cy = corners[7]
row = B.accuracy_row([cx + 0.0012, cy - 0.0005, B.EGG_HALF_WIDTH_M + 0.002], corners, B.EGG_HALF_WIDTH_M)
assert row["corner"] == 7 and abs(row["dx_mm"] - 1.2) < 1e-6 and abs(row["dy_mm"] + 0.5) < 1e-6 and abs(row["dz_mm"] - 2.0) < 1e-6
s_ = B.summarize([{"err_3d_mm": v} for v in (1.0, 2.0, 3.0, 4.0)])
assert s_["n"] == 4 and s_["mean"] == 2.5 and s_["max"] == 4.0
stem = B.write_results("accuracy", [row], {"x": s_})
assert stem.startswith(str(TMP / "cameracalib" / "d435i_bench")) and os.path.isfile(stem + ".csv")
assert json.loads(Path(stem + ".json").read_text(encoding="utf-8"))["summary"]["x"]["mean"] == 2.5
print(f"[M6] accuracy row {row}; summary {s_}; results in {Path(stem).parent}")

SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("[M7] release folder unchanged")
print("ALL V3.78 TESTS PASSED")
