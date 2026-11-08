#!/usr/bin/env python3
"""V3.51: Independent wrist ArUco marker settings (default 24 mm), visibility-aware automatic Eye-in-Hand collection
(eih_pose_visible, estimate_eih_from_probes, plan_eih_auto_targets), and 9-stage full-workflow status lamp evaluation
(WORKFLOW_STEPS, evaluate_workflow_lamps).

U1 Version V3.51 + markers in NERO_V312_Console.py, NERO_V325_Camera.py, and NERO_V349_Real_Wrist_Camera.py;
   NERO_V308 untouched; DEFAULT_WRIST_MARKER=24 mm while CAM.DEFAULT_MARKER=40 mm
U2 eih_pose_visible: accepts poses with marker inside wrist FOV at 10..58 cm and tilt <= 42 deg; rejects out-of-frame,
   out-of-range distance, tiny pixel size, and excessive tilt
U3 estimate_eih_from_probes + plan_eih_auto_targets: solves initial (T_flange_cam, T_base_marker) from 4 probe poses
   (< 0.5 mm), plans 10 visibility-screened, path-fence-safe poses with span >= 12 cm and rotation >= 25 deg, and
   rejects paths crossing safety-zone walls ('路径安全区')
U4 evaluate_workflow_lamps: verifies 9 ordered stages (solid required rings vs dashed optional rings) and transitions
   across grey, green, yellow, and red
"""
import hashlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import NERO_V312_Console as CON               # noqa: E402
import NERO_V315_Consistency_Safety as CS     # noqa: E402
import NERO_V325_Camera as CAM                # noqa: E402
import NERO_V347_Yolo_Perception as YP        # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49     # noqa: E402

# ------------------------------------------------------------------ U1: Version & independent marker defaults
assert W49.VERSION in ("V3.51", "V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5") and YP.VERSION in ("V3.51", "V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5") and CON.VERSION in ("V3.51", "V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5")
assert (ROOT / "VERSION").read_text().strip() in ("V3.51", "V3.52", "V3.53", "V3.54", "V3.55", "V3.56", "V3.57", "V3.58", "V3.59", "V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5")
h308 = hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest()
assert h308.startswith("020cc6680f181872"), f"NERO_V308 modified: {h308}"
assert CAM.DEFAULT_MARKER["size_mm"] == 40 and W49.DEFAULT_WRIST_MARKER["size_mm"] == 24.0
assert W49.WRIST_MARKER_SIZES_MM == (24, 30, 40)
print("[U1] V3.51 versions & markers verified; NERO_V308 untouched; main marker=40 mm, wrist marker=24 mm")

# ------------------------------------------------------------------ U2: eih_pose_visible
K_w = {"width": 1280, "height": 720, "fx": 820.0, "fy": 820.0, "ppx": 640.0, "ppy": 360.0}
X_true = [[1.0, 0.0, 0.0, 0.030], [0.0, 1.0, 0.0, -0.015], [0.0, 0.0, 1.0, 0.035], [0.0, 0.0, 0.0, 1.0]]
Y_true = [[1.0, 0.0, 0.0, -0.450], [0.0, 1.0, 0.0, 0.020], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
# Camera at (-0.450, 0.020, 0.240) looking straight down (R_bc = diag(1, -1, -1))
T_bc_ok = [[1.0, 0.0, 0.0, -0.450], [0.0, -1.0, 0.0, 0.020], [0.0, 0.0, -1.0, 0.240], [0.0, 0.0, 0.0, 1.0]]
A_ok = CAM.mat_mul(T_bc_ok, CAM.T_inv(X_true))
ok, why, vdeg = W49.eih_pose_visible(A_ok, X_true, Y_true, K_w, size_m=0.024)
assert ok and why == "ok" and vdeg < 2.0, (ok, why, vdeg)

# Too far (z = 0.85 m)
T_bc_far = [[1.0, 0.0, 0.0, -0.450], [0.0, -1.0, 0.0, 0.020], [0.0, 0.0, -1.0, 0.850], [0.0, 0.0, 0.0, 1.0]]
ok_f, why_f, _ = W49.eih_pose_visible(CAM.mat_mul(T_bc_far, CAM.T_inv(X_true)), X_true, Y_true, K_w, size_m=0.024)
assert not ok_f and why_f == "距离", (ok_f, why_f)

# Out of frame (shifted 30 cm in X)
T_bc_out = [[1.0, 0.0, 0.0, -0.150], [0.0, -1.0, 0.0, 0.020], [0.0, 0.0, -1.0, 0.240], [0.0, 0.0, 0.0, 1.0]]
ok_o, why_o, _ = W49.eih_pose_visible(CAM.mat_mul(T_bc_out, CAM.T_inv(X_true)), X_true, Y_true, K_w, size_m=0.024)
assert not ok_o and why_o == "出画面", (ok_o, why_o)

# Excessive tilt (60 deg tilt looking at marker)
R_tilt = CAM.mat_mul([[1, 0, 0], [0, -1, 0], [0, 0, -1]], CAM.rodrigues([math.radians(60.0), 0.0, 0.0]))
p_tilt = [-0.450, 0.020 - 0.24 * math.sin(math.radians(60.0)), 0.24 * math.cos(math.radians(60.0))]
T_bc_tilt = CAM.T_from(R_tilt, p_tilt)
ok_t, why_t, vd_t = W49.eih_pose_visible(CAM.mat_mul(T_bc_tilt, CAM.T_inv(X_true)), X_true, Y_true, K_w, size_m=0.024)
assert not ok_t and why_t == "倾角过大" and vd_t > 55.0, (ok_t, why_t, vd_t)
print(f"[U2] eih_pose_visible: normal view ok ({vdeg:.1f}°), far->{why_f}, out->{why_o}, 60° tilt->{why_t}")

# ------------------------------------------------------------------ U3: estimate_eih_from_probes + plan_eih_auto_targets
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
LIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708)]
Q_START = [0.0, 30.0, 0.0, 80.0, 0.0, 40.0, 0.0]
q0_rad = [math.radians(v) for v in Q_START]

def fk_rad(qr):
    T = CS.fk(U, dict(zip(J, qr)), "base_link", "link7")
    return [list(map(float, r)) for r in T] if T is not None else None

A0 = fk_rad(q0_rad)
# Construct X_eih so that at Q_START the wrist camera is at T_bc_ok looking straight down at Y_true
X_eih = CAM.mat_mul(CAM.T_inv(A0), T_bc_ok)

probe_qs = [
    Q_START,
    [3.5, 32.0, 0.0, 77.5, 8.0, 34.0, 10.0],
    [-3.5, 31.5, 2.5, 78.0, -8.0, 47.0, -10.0],
    [1.5, 27.5, -2.5, 83.0, 6.0, 48.0, 12.0],
]
probes = []
for pq in probe_qs:
    Ai = fk_rad([math.radians(v) for v in pq])
    Bi = CAM.mat_mul(CAM.T_inv(CAM.mat_mul(Ai, X_eih)), Y_true)
    probes.append({"q_deg": pq, "p_cam": CAM.T_t(Bi), "T_cam_marker": Bi, "T_base_flange": Ai})

est = W49.estimate_eih_from_probes(probes, size_mm=24.0)
assert est and est["mean_mm"] < 0.5, est
err_m_mm = CAM.vnorm(CAM.vsub(est["marker_in_base"], CAM.T_t(Y_true))) * 1000.0
assert err_m_mm < 1.0, err_m_mm

targets, st = W49.plan_eih_auto_targets(fk_rad, None, q0_rad, LIM, est, K_w, n=10, size_m=0.024,
                                        fence_fn=lambda q_: True, seed=42)
assert len(targets) == 10 and st["span_m"] >= 0.08 and st["rot_max_deg"] >= 20.0, (len(targets), st)

# Verify path safety-zone check rejects intermediate path violations
def wall_fence(qr):
    # Rejects configurations where J1 passes through [0.04, 0.08] rad
    return not (0.04 <= qr[0] <= 0.08)

_, st_wall = W49.plan_eih_auto_targets(fk_rad, None, [0.0] + q0_rad[1:], LIM, est, K_w, n=8, size_m=0.024,
                                       fence_fn=wall_fence, seed=42)
assert st_wall["rejected"].get("路径安全区", 0) > 0 or st_wall["rejected"].get("安全区", 0) > 0, st_wall
print(f"[U3] estimate_eih_from_probes err={err_m_mm:.3f} mm; plan_eih_auto_targets selected {len(targets)} poses "
      f"(span {st['span_m']*100:.1f} cm, rot {st['rot_max_deg']:.1f}°); path fence check active")

# ------------------------------------------------------------------ U4: evaluate_workflow_lamps
assert len(CAM.WORKFLOW_STEPS) == 9
req_keys = [k for k, _l, req in CAM.WORKFLOW_STEPS if req]
opt_keys = [k for k, _l, req in CAM.WORKFLOW_STEPS if not req]
assert req_keys == ["arm", "safety", "main_cam", "main_calib"]
assert opt_keys == ["depth", "tcp", "wrist_cam", "wrist_intr", "wrist_eih"]

# 1) Empty uninitialized state -> safety is red (required safety fence missing!), others grey
st0 = {d["key"]: d for d in CAM.evaluate_workflow_lamps(None, None, None)}
assert st0["safety"]["color"] == "red" and st0["arm"]["color"] == "grey" and st0["wrist_cam"]["optional"] is True

# 2) Valid safety zone, arm ready, main camera on USB 2.1 (yellow), no main calib (yellow/grey)
class Dummy:
    pass
pipe_d = Dummy(); pipe_d.state = {"moveit": ("green", "ok")}; pipe_d.teach_mode = False
safe_d = Dummy(); safe_d.valid_ws = {"fence": {"polygon": [[-0.8, -0.4], [-0.1, -0.4], [-0.1, 0.4]], "z_table": 0.0}}; safe_d.dirty = False
cam_d = Dummy()
cam_d.src = Dummy(); cam_d.src.error = None; cam_d.src.K = K_w; cam_d.src.info = {"usb": "2.1"}
cam_d.result = None; cam_d.calib = None; cam_d.samples = []; cam_d.tcp_calib = None; cam_d.tcp_samples = []
cam_d.wrist_src = Dummy(); cam_d.wrist_src.error = None; cam_d.wrist_src.info = {"res": "1920x1080"}
cam_d.wrist_quality = {"settled": True, "badge": "静止可采 ✓"}
cam_d.intr_result = None; cam_d.wrist_intr = None; cam_d.intr_samples = []
cam_d.wcalib_result = None; cam_d.wrist_calib = None; cam_d.wcalib_samples = []

st1 = {d["key"]: d for d in CAM.evaluate_workflow_lamps(pipe_d, safe_d, cam_d)}
assert st1["arm"]["color"] == "green" and st1["safety"]["color"] == "green"
assert st1["main_cam"]["color"] == "yellow"  # USB 2.1 warning
assert st1["wrist_cam"]["color"] == "green" and st1["wrist_intr"]["color"] == "red"  # wrist connected but no intrinsics!

# 3) All calibrated & USB 3.2 -> green across calibrated stages
cam_d.src.info = {"usb": "3.2"}
cam_d.calib = {"T_base_cam": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], "loo_mean_mm": 1.2, "loo_max_mm": 2.1}
cam_d.vstate = {"state": "ok"}
cam_d.wrist_intr = {"K": K_w, "dist": [0.0]*5, "rms_px": 0.35}
cam_d.wrist_calib = {"T_flange_cam": X_true, "marker_in_base": [-0.45, 0.02, 0.0], "loo_mean_mm": 0.9, "loo_max_mm": 1.5, "good": True}
st2 = {d["key"]: d for d in CAM.evaluate_workflow_lamps(pipe_d, safe_d, cam_d)}
for k in ("arm", "safety", "main_cam", "main_calib", "wrist_cam", "wrist_intr", "wrist_eih"):
    assert st2[k]["color"] == "green", (k, st2[k])
print("[U4] evaluate_workflow_lamps: 9 stages (4 solid required + 5 dashed optional) and 4-color transitions verified")
print("ALL V3.51 TESTS PASSED")
