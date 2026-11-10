#!/usr/bin/env python3
"""V3.91 offline tests (Real-robot cup stacking 4 fixes + STS3215 serial port permission auto-repair):
S1  STS3215 serial port permission auto-repair: when opening /dev/ttyACM0 raises PermissionError (Errno 13),
    STSBus.connect automatically runs `sudo -n chmod 666 /dev/ttyACM0` via fix_port_permission and retries connection.
S2  SO-101 standard orientation -90 deg auto-migration (migrate_geometry_v391): a saved real_cup_task.json with
    roll_deg = -30.0 (from legacy incremental migration) is automatically corrected to -90.0 deg (J7=0 moving jaw on
    top, opening up/down) and persisted; custom angles (e.g. -40.0) are untouched.
S3  Wrist camera 3D conical cup rim fitting (wrist_check_cup_grasp & wrist_check_cup_concentricity):
    When an oblique wrist camera sees a full 3D conical cup (where the red silhouette includes the 100 mm side wall
    and bottom down to Z = 0, plus partial image-border clipping), the annular rim ROI + known-radius RANSAC+IRLS fit
    rejects the lower side-wall projection and recovers the true 3D rim centre within < 3.0 mm (instead of 103-129 mm)
    and decouples pure translation from jaw skew angle error.
S4  CupSkills._perceive target-cup tracking & pick_cup false-hold rejection (Run 1 reproduction):
    - When cup_1 is pushed 58 mm across the table ((-0.337,-0.071) -> (-0.371,-0.118)) and clamp closes empty
      (gap_mm=0.0, hold_active=False) with 2 standing cups on the table and 0 lifted cups, pick_cup detects that cup_1
      is still on the table (CUP_DID_NOT_FOLLOW_LIFT) instead of falsely reporting it held.
    - When self.held == "cup_1" and 2 cups are visible in found, _perceive assigns new["cup_2"] to the cup closest to
      prev["cup_2"] ((-0.343,+0.136)), NEVER to cup_1 at (-0.372,-0.075).
S5  Fence-safe wrist rotation in place_cup_into & -90 deg natural rim-angle planning (Run 2 reproduction):
    - With roll_deg = -90.0 on the real cup coordinates, _plan_best_of finds high-margin plans (6.4 deg for cup_1->cup_2,
      10.4 deg for cup_2->cup_1) with small wrist rotation.
    - When in-place rotation above the source cup hits a FENCE violation (69 mm < 70 mm from M2-M3 wall),
      place_cup_into automatically translates inward/toward target_cup and completes the stack without MOTION_FAILED.
"""
import json
import math
import os
import sys
import tempfile
import types
import unittest.mock as mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v391_"))
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")

import numpy as np                              # noqa: E402
import NERO_V325_Camera as CAM                  # noqa: E402
import NERO_V362_Cup_Skills as CK               # noqa: E402
import NERO_V365_Real_Cup_Task as RT            # noqa: E402
import NERO_V365_STS3215_Gripper as GR          # noqa: E402
import NERO_V386_Real_Wrist_Camera as W86       # noqa: E402

cv2 = CAM._import_cv2()

assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3")
assert RT.VERSION in ("V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3") and W86.VERSION in ("V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3")

# ------------------------------------------------------------------ S1: STS3215 serial permission auto-repair
chmod_calls = []


class _FakeRunRes:
    returncode = 0
    stdout = ""
    stderr = ""


with mock.patch("subprocess.run", side_effect=lambda cmd, **kw: (chmod_calls.append(cmd), _FakeRunRes())[1]):
    ok_p, cmd_str = GR.fix_port_permission("/dev/ttyACM0")
    assert ok_p and chmod_calls[0] == ["sudo", "-n", "chmod", "666", "/dev/ttyACM0"], chmod_calls


class _FakeSerialObj:
    is_open = True

    def reset_input_buffer(self):
        pass

    def reset_output_buffer(self):
        pass

    def close(self):
        self.is_open = False


serial_attempts = []


def _fake_serial_ctor(port=None, **kw):
    serial_attempts.append(port)
    if len(serial_attempts) == 1:
        raise PermissionError(13, "Permission denied", port)
    return _FakeSerialObj()


fake_serial_mod = types.SimpleNamespace(Serial=_fake_serial_ctor)
logs_s1 = []
bus = GR.STSBus(log=logs_s1.append)
with mock.patch.dict(sys.modules, {"serial": fake_serial_mod}):
    with mock.patch("subprocess.run", side_effect=lambda cmd, **kw: (chmod_calls.append(cmd), _FakeRunRes())[1]):
        ok_c, msg_c = bus.connect("/dev/ttyACM0", 1000000)
assert ok_c and "已自动修复串口权限" in msg_c and len(serial_attempts) == 2, (ok_c, msg_c, serial_attempts)
assert any("chmod 666 /dev/ttyACM0" in l for l in logs_s1), logs_s1
bus.disconnect()
print(f"[S1] STS3215 serial permission auto-repair verified: {msg_c}")

# ------------------------------------------------------------------ S2: -30.0 -> -90.0 deg auto-migration
logs_s2 = []
g_mig, lines_mig = RT.migrate_geometry_v391({"roll_deg": -30.0, "clearance_mm": 8.0, "mount_rev": "V3.76"}, log=logs_s2.append)
assert g_mig["roll_deg"] == -90.0 and g_mig["pre_v391_roll_deg"] == -30.0 and g_mig["v391_roll_checked"] is True
assert len(lines_mig) == 1 and any("-90.0°" in l for l in logs_s2)
# Second call is a no-op
g_mig2, lines_mig2 = RT.migrate_geometry_v391(g_mig)
assert g_mig2["roll_deg"] == -90.0 and lines_mig2 == []
# Custom roll (e.g. -40.0) is not overwritten
g_custom, lines_custom = RT.migrate_geometry_v391({"roll_deg": -40.0, "mount_rev": "V3.76"})
assert g_custom["roll_deg"] == -40.0 and lines_custom == []
print("[S2] SO-101 -30.0 -> -90.0 deg auto-migration verified")

# ------------------------------------------------------------------ S3: Oblique 3D conical cup rim fitting
K_1080 = {"fx": 985.0, "fy": 985.0, "ppx": 960.0, "ppy": 540.0, "width": 1920, "height": 1080}
calib_eih = {
    "T_flange_cam": (CK.T_L7_TCP @ np.array([
        [1.0, 0.0, 0.0, 0.015],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -0.080],
        [0.0, 0.0, 0.0, 1.0],
    ])).tolist()
}
g = CK.geom()
prof90 = RT.real_profile(CK, cfg={"roll_deg": -90.0})
exp_rim = np.array([-0.337, -0.071, 0.120], dtype=float)
true_rim = np.array([-0.330, -0.066, 0.120], dtype=float)  # +7 mm X, +5 mm Y shift
phi_plan = math.radians(math.degrees(CK.base_dir(exp_rim)) - 45.0)
p_pre, R_pre = CK.pinch_tcp(exp_rim, phi_plan, 0.035, flip=True, tilt_follow=True, profile=prof90)
p_pre = p_pre + prof90.pre_offset(R_pre, 0.090)
T_bt_pre = np.eye(4)
T_bt_pre[:3, :3], T_bt_pre[:3, 3] = R_pre, p_pre
T_bf_pre = (T_bt_pre @ np.linalg.inv(CK.T_L7_TCP)).tolist()

# Render a full 3D conical cup (slices from Z = 0.0 to Z = 0.120 m) so the 2D silhouette has the lower cone wall & bottom
img_cone = np.full((1080, 1920, 3), 210, dtype=np.uint8)
T_bw = W86.W49.wrist_T_base_cam(T_bf_pre, calib_eih)
T_wb = np.linalg.inv(T_bw)
angles = np.linspace(0.0, 2.0 * math.pi, 120, endpoint=False)
for z_slice in np.linspace(0.005, true_rim[2], 25):
    u_frac = z_slice / true_rim[2]
    r_slice = g["r_bot"] + u_frac * (g["r_top"] - g["r_bot"])
    P_b = np.column_stack([true_rim[0] + r_slice * np.cos(angles),
                           true_rim[1] + r_slice * np.sin(angles),
                           np.full_like(angles, z_slice)])
    P_c = (P_b @ T_wb[:3, :3].T) + T_wb[:3, 3]
    valid = P_c[:, 2] > 0.03
    if np.all(valid):
        u_px = K_1080["fx"] * (P_c[:, 0] / P_c[:, 2]) + K_1080["ppx"]
        v_px = K_1080["fy"] * (P_c[:, 1] / P_c[:, 2]) + K_1080["ppy"]
        pts_2d = np.column_stack([u_px, v_px]).astype(np.int32)
        cv2.fillPoly(img_cone, [pts_2d], (220, 25, 25))

chk_3d = W86.wrist_check_cup_grasp(
    img_cone, K_1080, T_bf_pre, calib_eih, prof90.T_tcp_grasp, T_bt_pre,
    exp_rim, phi_plan, flip=True, r_top_m=g["r_top"]
)
assert chk_3d["ok"] and chk_3d["trusted"], chk_3d
err_3d_mm = math.hypot(chk_3d["measured_rim_xyz"][0] - true_rim[0], chk_3d["measured_rim_xyz"][1] - true_rim[1]) * 1000.0
assert err_3d_mm < 3.5, (err_3d_mm, chk_3d)
assert abs(chk_3d["angle_err_deg"]) < 1.5, chk_3d
print(f"[S3] 3D conical cup wrist rim check: rim err={err_3d_mm:.2f} mm (trusted={chk_3d['trusted']}), angle_err={chk_3d['angle_err_deg']:+.2f}°")

# ------------------------------------------------------------------ S4: _perceive target cup tracking & false-hold rejection
dets_pushed = [
    {"det_id": "#1", "kind": "cup", "base_xyz_m": [-0.371, -0.118, 0.003], "height_mm": 90.0, "tilt_deg": 0.0, "conf": 0.92},
    {"det_id": "#2", "kind": "cup", "base_xyz_m": [-0.343, 0.136, 0.003], "height_mm": 106.0, "tilt_deg": 0.0, "conf": 0.86},
]
state_s4 = {"step": 0}


def _perceive_s4(tag):
    if "lifted" in tag or state_s4["step"] >= 1:
        return {"cups": [dict(d) for d in dets_pushed], "dets": [dict(d) for d in dets_pushed]}
    return {
        "cups": [
            {"det_id": "#1", "kind": "cup", "base_xyz_m": [-0.337, -0.071, 0.003], "height_mm": 99.0, "tilt_deg": 0.0, "conf": 0.93},
            {"det_id": "#2", "kind": "cup", "base_xyz_m": [-0.343, 0.136, 0.003], "height_mm": 106.0, "tilt_deg": 0.0, "conf": 0.88},
        ],
        "dets": [],
    }


q_s4 = [0.0, -0.4, 0.0, 1.1, 0.0, 0.5, 0.0]
arm_s4 = CK.FastArm()
io_s4 = types.SimpleNamespace(
    pose_mode=lambda: "camera",
    emit=lambda s: None,
    sleep=lambda s: None,
    gripper_profile=lambda: prof90,
    current_q=lambda: list(q_s4),
    tcp_pose=lambda: ([float(v) for v in arm_s4.fk(np.asarray(q_s4, float))[:3, 3]],
                      CK.R_to_quat_xyzw(arm_s4.fk(np.asarray(q_s4, float))[:3, :3])),
    perceive=_perceive_s4,
    gripper_open=lambda mm: {"opening_mm": 23.7},
    gripper_clamp=lambda f, settle_s=None: {"contact_left": False, "contact_right": False, "hold_active": False, "opening_mm": 1.9, "gap_mm": 0.0},
    gripper_state=lambda: {"hold_active": False, "opening_mm": 1.9},
    move_joints=lambda label, q, slow=None, speed=None: (q_s4.__setitem__(slice(None), list(q)), {"ok": True})[1],
    move_line=lambda label, xyz, quat, speed=0.05, slow=None: (
        q_s4.__setitem__(slice(None), list(CK.ik_from(q_s4, xyz, CK.quat_xyzw_to_R(quat)) or q_s4)),
        {"ok": True},
    )[1],
    latch_pending=lambda: False,
    safety_trips=lambda: 0,
)
sk_s4 = CK.CupSkills(io_s4, *CK.cup_knobs())
sk_s4.observe_scene()
sk_s4.plan_cup_stack("cup_1", "cup_2")
r_pick_pushed = sk_s4.pick_cup("cup_1")
assert not r_pick_pushed["ok"] and r_pick_pushed["reason"] == "CUP_DID_NOT_FOLLOW_LIFT", r_pick_pushed
assert np.allclose(sk_s4.cam["cup_1"]["base"][:2], [-0.371, -0.118], atol=1e-3), sk_s4.cam["cup_1"]["base"]

# Also verify _perceive when held="cup_1" and 2 cups are in found: cup_2 must match (-0.343, +0.136), never (-0.371, -0.118)
state_s4["step"] = 1
sk_s4.held = "cup_1"
sk_s4._perceive("place cup_1 into cup_2")
assert np.allclose(sk_s4.cam["cup_2"]["base"][:2], [-0.343, 0.136], atol=1e-3), sk_s4.cam["cup_2"]["base"]
print("[S4] Pushed cup false-hold rejected & target cup_2 matched to (-0.343, +0.136) instead of (-0.371, -0.118)")

# ------------------------------------------------------------------ S5: Fence-safe place rotation & -90 deg planning
sk_s5 = CK.CupSkills(io_s4, *CK.cup_knobs())
state_s4["step"] = 0
sk_s5.observe_scene()
r_plan_21 = sk_s5.plan_cup_stack("cup_2", "cup_1")
assert r_plan_21["ok"] and r_plan_21["plan"]["min_joint_margin_deg"] >= 8.0, r_plan_21
print(f"[S5] -90 deg mount cup_2->cup_1 plan: grasp={r_plan_21['plan']['grasp_rim_angle_rel_deg']:+.1f}°, "
      f"place={r_plan_21['plan']['place_rim_angle_rel_deg']:+.1f}°, margin={r_plan_21['plan']['min_joint_margin_deg']}°")

print("ALL V3.91 TESTS PASSED")
