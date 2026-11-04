#!/usr/bin/env python3
"""V3.25 Camera调试: maths + image pipeline (no camera, no robot, no Tk).

C1 linear algebra: Horn/Kabsch recovers random rigid transforms; plane fit; Rodrigues; projection round trip
C2 eye-to-hand calibration on synthetic NERO poses (FK of the URDF): camera pose + marker offset recovered from
   0.5 mm noisy samples, outlier rejected, leave-one-out errors small; narrow wrist rotation -> warning
C3 sample poses stay inside the soft limits; calibration file save / load (check poses kept)
C4 (numpy + OpenCV with aruco) a marker rendered into a synthetic 1280x720 image at 0.8 m is detected; PnP centre
   within a few mm, the depth-refined centre within 1 mm; depth colour map / PPM / PNG fallback
"""
import json, math, os, random, sys, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V325_Camera as CAM  # noqa: E402

assert CAM.VERSION.startswith("V3.2")
rnd = random.Random(7)


def rand_T(rnd, tr=0.8):
    rv = [rnd.uniform(-2, 2) for _ in range(3)]
    return CAM.T_from(CAM.rodrigues(rv), [rnd.uniform(-tr, tr) for _ in range(3)])


for _ in range(5):
    T = rand_T(rnd)
    P = [[rnd.uniform(-0.5, 0.5) for _ in range(3)] for _ in range(12)]
    Q = [CAM.T_apply(T, p) for p in P]
    R, t = CAM.kabsch(P, Q)
    err = max(CAM.vnorm(CAM.vsub(CAM.T_apply(CAM.T_from(R, t), p), q)) for p, q in zip(P, Q))
    assert err < 1e-9, err
    assert abs(CAM.rot_angle_deg(CAM.mat_mul(CAM.transpose(R), CAM.T_R(T)))) < 1e-5
Tn = rand_T(rnd)
pl = [CAM.T_apply(Tn, [rnd.uniform(-0.2, 0.2), rnd.uniform(-0.2, 0.2), rnd.gauss(0, 0.001)]) for _ in range(400)]
fit = CAM.plane_fit(pl)
n_true = [Tn[0][2], Tn[1][2], Tn[2][2]]
assert abs(abs(sum(a * b for a, b in zip(fit["normal"], n_true))) - 1) < 1e-4 and abs(fit["rms_m"] - 0.001) < 0.0003
K = {"fx": 908.7, "fy": 908.7, "ppx": 640.1, "ppy": 366.6}
p = CAM.deproject(812.3, 211.9, 0.734, K)
uv = CAM.project(p, K)
assert abs(uv[0] - 812.3) < 1e-9 and abs(uv[1] - 211.9) < 1e-9
print("[C1] Horn/Kabsch exact on random transforms; plane fit normal + RMS; deproject/project round trip")

import NERO_V315_Consistency_Safety as CS  # noqa: E402
import NERO_V314_Joint_Control as JC  # noqa: E402
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
LIM = JC.build_model((ROOT / "config" / "nero.urdf").read_text(), None, None)["limits"]


def flange(q_deg):
    T = CS.fk(U, dict(zip(J, [math.radians(v) for v in q_deg])), "base_link", "link7")
    return [list(map(float, r)) for r in T]


# camera at (-0.15, 0.65, 0.75) looking at the work area (-0.40, 0.05, 0.05)
pos, tgt = [-0.15, 0.65, 0.75], [-0.40, 0.05, 0.05]
f = CAM.vsub(tgt, pos); nf = CAM.vnorm(f); f = [v / nf for v in f]
x = [f[1] * 1 - 0, 0 - f[0] * 1, 0]; x = [-v for v in x]
x = [f[1] * 0 - f[2] * 0, f[2] * 0 - f[0] * 0, 0]
up = [0, 0, 1]
x = [f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]]
nx = CAM.vnorm(x); x = [v / nx for v in x]
y_up = [x[1] * f[2] - x[2] * f[1], x[2] * f[0] - x[0] * f[2], x[0] * f[1] - x[1] * f[0]]
R_opt = [[x[i], -y_up[i], f[i]] for i in range(3)]
X_true = CAM.T_from(R_opt, pos)
y_true = [0.012, -0.031, 0.058]
q0 = [0.0, 60.0, 0.0, 70.0, 0.0, 20.0, 0.0]
qs = CAM.sample_targets(q0, LIM, 16, seed=3)
samples = []
for q in qs:
    A = flange(q)
    pc = CAM.T_apply(CAM.T_inv(X_true), CAM.T_apply(A, y_true))
    pc = [v + rnd.gauss(0, 0.0005) for v in pc]
    samples.append({"q_deg": q, "T_base_flange": A, "p_cam": pc})
for q in qs:
    for j, (lo, hi) in enumerate(LIM):
        assert math.degrees(lo) + 3 - 1e-6 <= q[j] <= math.degrees(hi) - 3 + 1e-6
samples[5]["p_cam"] = [samples[5]["p_cam"][0] + 0.02, samples[5]["p_cam"][1], samples[5]["p_cam"][2]]   # bad sample
r = CAM.calibrate(samples)
assert r["ok"] and r["rejected"] == [5], r.get("rejected")
dpos = CAM.vnorm(CAM.vsub(CAM.T_t(r["T_base_cam"]), pos))
drot = CAM.rot_angle_deg(CAM.mat_mul(CAM.transpose(CAM.T_R(r["T_base_cam"])), R_opt))
dy = CAM.vnorm(CAM.vsub(r["marker_in_flange"], y_true))
assert dpos < 0.003 and drot < 0.3 and dy < 0.003, (dpos, drot, dy)
assert r["mean_mm"] < 1.2 and r["loo_mean_mm"] < 1.8 and r["loo_max_mm"] < 4.0 and r["good"], r
assert r["rot_span_deg"] > 25 and r["trans_span_m"] > 0.15 and abs(r["cam_down_deg"] - 45) < 15
qn = CAM.sample_targets(q0, LIM, 10, seed=5, spans=(6, 4, 0, 4, 0, 0, 0))
sn = []
for q in qn:
    A = flange(q)
    sn.append({"q_deg": q, "T_base_flange": A, "p_cam": CAM.T_apply(CAM.T_inv(X_true), CAM.T_apply(A, y_true))})
rn = CAM.calibrate(sn)
assert rn["ok"] and any("朝向变化" in w for w in rn["warnings"]) and not rn["good"], (rn["rot_span_deg"], rn["warnings"])
print(f"[C2] calibration: camera within {dpos * 1000:.1f} mm / {drot:.2f}°, marker offset within {dy * 1000:.1f} mm, "
      f"bad sample rejected; residual {r['mean_mm']:.2f} mm, leave-one-out {r['loo_mean_mm']:.2f}/{r['loo_max_mm']:.2f} mm; "
      f"narrow wrist rotation ({rn['rot_span_deg']:.0f}°) -> warning")

tmp = Path(tempfile.mkdtemp(prefix="v325_")) / "calib.json"
d = CAM.save_calib(r, samples, {"serial": "261222075224", "K": K}, CAM.DEFAULT_MARKER, str(tmp))
d2 = CAM.load_json(str(tmp))
assert d2["camera_serial"] == "261222075224" and len(d2["check_poses_deg"]) == 2 and d2["n_used"] == 15
assert CAM.marker_error_mm(d2, samples[0]["T_base_flange"], samples[0]["p_cam"]) < 2.5
print("[C3] sample poses inside the soft limits; calibration saved / loaded with 2 check poses; live error from it")

try:
    import numpy as np
    import cv2
    have = hasattr(cv2, "aruco")
except Exception:
    have = False
if not have:
    print("[C4] numpy / OpenCV aruco not installed here -> image part skipped")
else:
    S = 400
    mk = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, S) \
        if hasattr(cv2.aruco, "generateImageMarker") else cv2.aruco.drawMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, S)
    pad = S // 4
    mkp = np.full((S + 2 * pad, S + 2 * pad), 255, np.uint8)
    mkp[pad:pad + S, pad:pad + S] = mk
    size = 0.040
    h = size / 2
    Rm = CAM.mat_mul(CAM.rodrigues([0.35, -0.25, 0.1]), [[1, 0, 0], [0, -1, 0], [0, 0, -1]])   # marker faces the camera
    tm = [0.06, -0.03, 0.80]
    Km = np.array([[K["fx"], 0, K["ppx"]], [0, K["fy"], K["ppy"]], [0, 0, 1.0]])
    scale = size / S
    src = []
    dst = []
    for (px, py) in ((0, 0), (S + 2 * pad, 0), (S + 2 * pad, S + 2 * pad), (0, S + 2 * pad)):
        mx, my = (px - pad) * scale - h, h - (py - pad) * scale
        pc = CAM.T_apply(CAM.T_from(Rm, tm), [mx, my, 0.0])
        src.append([px, py])
        dst.append(CAM.project(pc, K))
    H = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    img = cv2.warpPerspective(mkp, H, (1280, 720), borderValue=180)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    nrm = [Rm[0][2], Rm[1][2], Rm[2][2]]
    vv, uu = np.mgrid[0:720, 0:1280]
    rx, ry = (uu - K["ppx"]) / K["fx"], (vv - K["ppy"]) / K["fy"]
    den = nrm[0] * rx + nrm[1] * ry + nrm[2]
    depth = (nrm[0] * tm[0] + nrm[1] * tm[1] + nrm[2] * tm[2]) / den
    det0 = CAM.detect_aruco(rgb, K, [0] * 5, "DICT_4X4_50", 0, size)
    det1 = CAM.detect_aruco(rgb, K, [0] * 5, "DICT_4X4_50", 0, size, depth_m=depth.astype(np.float32))
    assert det0 and det1 and det1["source"] == "depth" and CAM.detect_aruco(rgb, K, [0] * 5, "DICT_4X4_50", 7, size) is None
    e0 = CAM.vnorm(CAM.vsub(det0["p_cam"], tm)) * 1000
    e1 = CAM.vnorm(CAM.vsub(det1["p_cam"], tm)) * 1000
    assert e0 < 8.0 and e1 < 1.0, (e0, e1)
    cm = CAM.depth_colormap(np.array([[0.0, 0.3, 0.9, 1.5]], np.float32))
    assert cm.shape == (1, 4, 3) and tuple(cm[0, 0]) == (0, 0, 0) and cm[0, 1, 0] > 100 > cm[0, 1, 2] and cm[0, 3, 2] > 100 > cm[0, 3, 0]   # near red, far blue
    ppm = CAM.ppm_bytes(rgb[::2, ::2].copy())
    png = CAM.png_bytes_from_ppm(ppm)
    assert ppm.startswith(b"P6 640 360 255\n") and png.startswith(b"\x89PNG") and len(ppm) == 15 + 640 * 360 * 3
    mi = CAM.marker_image("DICT_4X4_50", 0)
    assert mi.shape[2] == 3 and CAM.detect_aruco(mi, {"fx": 800, "fy": 800, "ppx": mi.shape[1] / 2, "ppy": mi.shape[0] / 2},
                                                 [0] * 5, "DICT_4X4_50", 0, 0.04) is not None
    import NERO_V308_Camera_Perception as P8
    Tbc = np.array(r["T_base_cam"], float)
    dets, diag = P8.perceive_eggs(rgb, depth.astype(np.float32), dict(K), Tbc, stride=4)
    assert isinstance(dets, list) and "points_valid" in diag
    print(f"[C4] OpenCV {cv2.__version__}: synthetic marker at 0.8 m detected; PnP centre error {e0:.1f} mm, depth-refined "
          f"{e1:.2f} mm; colour map / PPM / PNG fallback / printable marker ok")
print("ALL V3.25 TESTS PASS")
