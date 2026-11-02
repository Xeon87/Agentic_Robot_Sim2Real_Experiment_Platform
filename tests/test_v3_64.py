#!/usr/bin/env python3
"""V3.64 offline tests (stack_cups_so101: the SO-101 gripper on a 7 mm adapter ring, rolled 45 deg):
U1  Versions V3.64 + markers; NERO_V308 / NERO_V326 / Agent / Camera tab frozen; backend, fixed-camera process and cup
    perception byte-identical to V3.63 (the new scene is a variant of stack_cups: no backend change).
U2  SO-101 mount + jaw geometry: proper rotations; gripper origin = seat + 7 mm ring on the tool axis; the closing axis
    is the official one turned 45 deg about link7 +x (left-hand 45 deg about the outward axis); the tip pads close a
    2 mm wall inside the hinge range (~-6 deg) and the moving pad mirrors the fixed one at 0 deg; pregrasp tip gap
    32 mm inside the hinge range; the mount parameters can be overridden by environment variables.
U3  Scene registry: stack_cups_so101 is a variant (active_scene() = stack_cups, gripper_variant() = so101); the MJCF
    builder mounts the SO-101 gripper with every name the backend needs and it compiles in MuJoCo; the backend's hard
    restore (gripper_joint2 = -gripper_joint1) stays exact; stack_cups (official) and store_in_safe XML unchanged.
U4  Console drop-down: the 3 old entries unchanged + stack_cups_so101 last; start_nero_sim.sh accepts it (and checks
    the SO-101 meshes), still rejects unknown names.
U5  Gripper profiles: official profile = V3.63 behaviour (TCP +2 mm, both flips, vertical approach, no clearance, no
    settle, no release-gap cap); SO-101 profile: fixed jaw outside, 2 mm clearance, approach along the wall, 2.5 s
    settle after lift and carry, release gap <= 35 mm, release lean <= 9 deg;
    BackendIO picks the profile from the console scene (NERO_SCENE).
U6  Twin (URDF arm kinematics + SO-101 MJCF): both directions end fully nested; release sidestep executed.
U7  gui_run_task through BackendIO + fake backend core with the SO-101: cup_1 -> cup_2 nested, s=1, log names the
    gripper; with an OpenGL context also with the fixed D435i as pose source.
U8  Shipped SO-101 study (config/cup_stack_study_v364_so101.json): every run nested; nothing in the release folder is
    written by the tests.
"""
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v364_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
for k in ("NERO_SCENE", "NERO_V343_ACTIVE_SCENE", "NERO_V362_CUP_SOLREF", "NERO_V308_DEPTH_NOISE_K",
          "NERO_V364_SO101_SEAT_X_MM", "NERO_V364_SO101_RING_MM", "NERO_V364_SO101_ROLL_DEG",
          "NERO_V364_SO101_CLEARANCE_MM", "NERO_V364_SO101_RELEASE_GAP_MM"):
    os.environ.pop(k, None)
H = lambda b: hashlib.sha256(b).hexdigest()


def snapshot():
    out = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rel = line.split(None, 1)[1].strip()
            out[rel] = H((ROOT / rel).read_bytes()) if (ROOT / rel).is_file() else None
    out["__config__"] = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "config").rglob("*") if p.is_file())
    return out


SNAP0 = snapshot()
import numpy as np                          # noqa: E402
import NERO_V312_Console as CON             # noqa: E402
import NERO_V313_Real_Robot as RR           # noqa: E402
import NERO_V343_Scenes as SC               # noqa: E402
import NERO_V347_Yolo_Perception as YP      # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49   # noqa: E402
import NERO_V359_Strategy as ST             # noqa: E402
import NERO_V362_Cup_Skills as CK           # noqa: E402
import NERO_V364_SO101_Gripper as G         # noqa: E402

try:
    import mujoco
    HAVE_MJ = True
except Exception:
    HAVE_MJ = False

# ------------------------------------------------------------------ U1
for m in (CON, RR, YP, W49, ST, CK, G):
    assert m.VERSION == "V3.64", (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() == "V3.64" and SC.SCENES_REV == "V3.64"
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
assert H((ROOT / "NERO_V326_Wrist_Camera.py").read_bytes()) == "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a"
assert H((ROOT / "NERO_V2_7_1_Agent.py").read_bytes()) == "3e52d05ea84d94c8359d5875fa0fee8c2cc42d612b52e59109e1166912439696"
assert H((ROOT / "NERO_V325_Camera.py").read_bytes()) == "5f41c189ea742fc4c3b4d4b33b7190c5a456ba6e6eb6adf39bde92bf1ed82688"
for fn, h in (("NERO_V2_7_1_Unified_Backend.py", "bb9b7e5e927ed3fda9112b2edff0d99a482ade9ac3dc6a0b66901b70d257a04a"),
              ("NERO_V331_Fixed_Camera.py", "8314f82cce52016f3dd08bad22ce08dbd9bdb1b263fa6aef23453467a9b3a341"),
              ("NERO_V363_Cup_Perception.py", "b9633693657127fccc101bd902ad38e6bbc014fc907c41bd54227c1711a732c5")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed (V3.64 must not touch it)"
for fn, mk in (("NERO_V364_SO101_Gripper.py", G.MARKER), ("NERO_V362_Cup_Skills.py", CK.MARKER_V364),
               ("NERO_V343_Scenes.py", "V3.64 SO-101 GRIPPER ON NERO"), ("NERO_V312_Console.py", "V3.64 SO-101 GRIPPER ON NERO"),
               ("start_nero_sim.sh", "V3.64 SO-101 GRIPPER ON NERO"), ("NERO_V364_SO101_Gripper.py", "V3.64 LINE-CONTACT MOVING PAD"),
               ("NERO_V362_Cup_Skills.py", "V3.64 LIFT SETTLE"),
               ("NERO_V362_Cup_Skills.py", "V3.64 RELEASE SIDESTEP"), ("NERO_V362_Cup_Skills.py", "V3.64 LOW RELEASE"),
               ("NERO_V362_Cup_Skills.py", "V3.64 JAW SIDES"), ("NERO_V362_Cup_Skills.py", "V3.64 JOINT SETTLE"),
               ("NERO_V362_Cup_Skills.py", "V3.64 CARRY BRANCH")):
    assert mk in (ROOT / fn).read_text(encoding="utf-8"), (fn, mk)
for f in ("wrist_roll_follower_so101_v1.stl", "moving_jaw_so101_v1.stl", "sts3215_03a_v1.stl", "NOTICE.txt"):
    assert (ROOT / "assets" / "so101" / f).is_file(), f
assert "Apache" in (ROOT / "assets" / "so101" / "NOTICE.txt").read_text(encoding="utf-8")
print("[U1] V3.64; frozen modules unchanged; backend / fixed-camera process / cup perception = V3.63; SO-101 assets present")

# ------------------------------------------------------------------ U2
def proper(R):
    return abs(np.linalg.det(R) - 1.0) < 1e-9 and np.abs(R.T @ R - np.eye(3)).max() < 1e-9


T = G.T_l7_gripper()
assert proper(T[:3, :3]) and proper(G.T_gripper_grasp()[:3, :3]) and proper(G.T_tcp_grasp()[:3, :3])
assert np.allclose(T[:3, 3], [G.MOUNT["seat_x_m"] + 0.007, 0.0, -0.0235]) and G.MOUNT["ring_m"] == 0.007
assert np.allclose(T[:3, 2], [-1, 0, 0])                         # jaws point along link7 +x (away from the flange)
x_off = np.array([0.0, 0.0, -1.0])                               # official closing axis in link7
x_g = T[:3, 0]
ang = math.degrees(math.atan2(np.dot(np.cross(x_off, x_g), [1, 0, 0]), np.dot(x_off, x_g)))
assert abs(ang - (-45.0)) < 1e-6, ang                            # -45 deg right-hand about +x7 = 45 deg left-hand
th2 = G.jaw_angle_deg(G.opening_mm_for_tip_gap(2.0))
assert -10.0 < th2 < -3.0 and abs(G.tip_gap_mm(th2) - 2.0) < 0.05 and G.tip_gap_mm(-10.0) < 2.0
th32 = G.jaw_angle_deg(G.opening_mm_for_tip_gap(32.0))
assert th2 < th32 < 100.0 and abs(G.tip_gap_mm(th32) - 32.0) < 0.05
assert abs(G.jaw_angle_deg(G.opening_mm_for_angle(37.0)) - 37.0) < 1e-9
if HAVE_MJ:
    p = G.gripper_parts(angle_unit="radian")
    xml = (f"<mujoco><compiler angle='radian'/><asset>{p['asset']}</asset><worldbody><body name='link7'>{p['link7']}"
           f"</body></worldbody><equality>{p['equality']}</equality></mujoco>")
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)                                   # jaw at 0 deg
    gb = m.body("gripper_link1").id
    Rg, pg = d.xmat[gb].reshape(3, 3), d.xpos[gb]
    c1, c2 = (Rg.T @ (d.geom_xpos[m.geom(n).id] - pg) for n in ("finger1_rubber_pad", "finger2_rubber_pad"))
    assert np.allclose(c2, [-c1[0], c1[1], c1[2]], atol=1e-6) and np.allclose(c1, G.PAD_FIXED_CENTRE, atol=1e-9)
env_t = dict(os.environ, NERO_V364_SO101_RING_MM="9", NERO_V364_SO101_ROLL_DEG="45")
r = subprocess.run([sys.executable, "-c", "import NERO_V364_SO101_Gripper as G; print(G.MOUNT['ring_m'], G.MOUNT['roll_deg'])"],
                   cwd=str(ROOT), env=env_t, capture_output=True, text=True)
assert r.returncode == 0 and r.stdout.split() == ["0.009", "45.0"], (r.stdout, r.stderr)
print(f"[U2] mount: ring 7 mm, roll -45 deg about +x7 (left-hand 45); 2 mm wall clamped at {th2:.1f} deg, "
      f"32 mm pregrasp gap at {th32:.1f} deg; env overrides work")

# ------------------------------------------------------------------ U3
assert list(SC.SCENES) == ["fill_egg_holder", "store_in_safe", "stack_cups", "stack_cups_so101"]


def scene_env(name):
    os.environ["NERO_SCENE"] = name
    os.environ["NERO_V343_ACTIVE_SCENE"] = SC.selected_scene()          # what the backend exports in main()


scene_env("stack_cups_so101")
assert SC.selected_scene() == SC.active_scene() == "stack_cups" and SC.selected_variant() == "stack_cups_so101"
assert SC.gripper_variant() == "so101" and CK.BackendIO({"skill_core": None, "shared": {}}).gripper_profile().name == "so101"
scene_env("stack_cups")
assert SC.gripper_variant() == "official" and CK.BackendIO({"skill_core": None, "shared": {}}).gripper_profile() is CK.OFFICIAL_PROFILE
os.environ.pop("NERO_SCENE"), os.environ.pop("NERO_V343_ACTIVE_SCENE")
assert SC.active_scene() == "fill_egg_holder" and SC.gripper_variant() == "official"


def build(scene, body):
    scene_env(scene)
    try:
        td = Path(tempfile.mkdtemp(dir=TMP))
        pth = td / "nero_motor.xml"
        pth.write_text(f"<mujoco model='t'><compiler angle='radian'/><asset></asset><worldbody>{body}</worldbody></mujoco>")
        out = Path(SC.make_store_in_safe_xml(str(pth)))
        return out, out.read_text().replace(str(td.resolve()), "<TMP>")
    finally:
        os.environ.pop("NERO_SCENE")
        os.environ.pop("NERO_V343_ACTIVE_SCENE")


link7 = "<body name='link7' pos='0 0 0.8'><freejoint/><geom type='sphere' size='0.01' contype='0' conaffinity='0'/></body>"
out_o, xml_o = build("stack_cups", link7)
out_s, xml_s = build("stack_cups_so101", link7)
assert out_o.name == "_nero_motor_v343_stack_cups_tmp.xml" and out_s.name == "_nero_motor_v343_stack_cups_so101_tmp.xml"
assert "gripper_joint1" not in xml_o and "so101" not in xml_o
assert xml_s.count('name="gripper_joint1"') == 1 and xml_s.count('name="gripper_joint2"') == 1
_, xml_safe = build("store_in_safe", "<body name='link7'/>")
assert "so101" not in xml_safe
if HAVE_MJ:
    m = mujoco.MjModel.from_xml_path(str(out_s))
    need = [(mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint1"), (mujoco.mjtObj.mjOBJ_JOINT, "gripper_joint2"),
            (mujoco.mjtObj.mjOBJ_BODY, "gripper_link1"), (mujoco.mjtObj.mjOBJ_BODY, "gripper_link2"),
            (mujoco.mjtObj.mjOBJ_GEOM, "finger1_rubber_pad"), (mujoco.mjtObj.mjOBJ_GEOM, "finger2_rubber_pad"),
            (mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper"), (mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_force"),
            (mujoco.mjtObj.mjOBJ_EQUALITY, "gripper_coupling"), (mujoco.mjtObj.mjOBJ_EQUALITY, "hold_obj_cup_1")]
    for kind, n in need:
        assert mujoco.mj_name2id(m, kind, n) >= 0, n
    assert m.body(m.geom("finger1_rubber_pad").bodyid[0]).name == "gripper_link1"
    assert m.body(m.geom("finger2_rubber_pad").bodyid[0]).name == "gripper_link2"
    lo, hi = m.jnt_range[m.joint("so101_jaw_hinge").id]
    assert abs(math.degrees(lo) + 10) < 1e-3 and abs(math.degrees(hi) - 100) < 1e-3       # radian model: converted
    lo2, hi2 = m.jnt_range[m.joint("gripper_joint2").id]
    assert abs(lo2 + 0.05) < 1e-9 and abs(hi2) < 1e-9
    # backend hard restore: qpos(gripper_joint2) = -qpos(gripper_joint1) -> the coupling equalities are satisfied
    d = mujoco.MjData(m)
    q1 = 0.012
    d.qpos[m.joint("gripper_joint1").qposadr[0]] = q1
    d.qpos[m.joint("gripper_joint2").qposadr[0]] = -q1
    d.qpos[m.joint("so101_jaw_hinge").qposadr[0]] = math.radians(G.jaw_angle_deg(2000 * q1))
    mujoco.mj_forward(m, d)
    assert np.abs(d.efc_pos[d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY]).max() < 1e-5   # MJCF coefficients written with 6 digits
print("[U3] stack_cups_so101 = variant (backend sees stack_cups); SO-101 MJCF compiles with all backend names; "
      "hard restore exact; official / store_in_safe XML without SO-101")

# ------------------------------------------------------------------ U4
assert CON.SIM_SCENES[:3] == [("fill_egg_holder", "fill_egg_holder - 鸡蛋放入蛋托 (默认)"),
                              ("store_in_safe", "store_in_safe - 物体放入保险箱 (RoboDojo)"),
                              ("stack_cups", "stack_cups - 叠杯子 (RoboDojo)")]
assert CON.SIM_SCENES[3][0] == "stack_cups_so101" and "SO-101" in CON.SIM_SCENES[3][1]
assert [k for k, _ in CON.SIM_SCENES] == list(SC.SCENES) and CON.DEFAULT_SIM_SCENE == "fill_egg_holder"
assert CON.scene_env("stack_cups_so101") == {"NERO_SCENE": "stack_cups_so101"}
L = (ROOT / "start_nero_sim.sh").read_text()
case = L[L.index('SCENE344="${NERO_SCENE:-fill_egg_holder}"'):L.index('export NERO_SCENE="$SCENE344"')]
fake_here = TMP / "here"
(fake_here / "assets" / "so101").mkdir(parents=True)
(fake_here / "NERO_V364_SO101_Gripper.py").write_text("x = 1\n")
for val, ok, meshes in (("stack_cups_so101", True, True), ("stack_cups_so101", False, False), ("stack_cups", True, False),
                        ("nope", False, True)):
    for f in ("wrist_roll_follower_so101_v1.stl", "moving_jaw_so101_v1.stl", "sts3215_03a_v1.stl"):
        pth = fake_here / "assets" / "so101" / f
        pth.write_text("s") if meshes else (pth.unlink() if pth.exists() else None)
    sh = f'fatal(){{ echo "$*"; exit 3; }}\nHERE="{fake_here}"\n' + case
    r = subprocess.run(["bash", "-c", sh], env=dict(os.environ, NERO_SCENE=val), capture_output=True, text=True)
    assert (r.returncode == 0) == ok, (val, meshes, r.stdout, r.stderr)
print("[U4] console drop-down: 3 old entries unchanged + stack_cups_so101; launcher accepts it and checks the meshes")

# ------------------------------------------------------------------ U5
O = CK.OFFICIAL_PROFILE
assert np.allclose(O.T_tcp_grasp, CK._t(0.0, 0.002, 0.0)) and O.flips == (False, True) and O.approach_clearance_m == 0.0
assert O.lift_settle_s == 0.0 and O.release_tilt_max_deg == 0.0
assert O.release_gap_max_mm is None and np.allclose(O.pre_offset(np.eye(3), 0.09), [0, 0, 0.09])
R0 = CK.pinch_frame(0.3, True)
p_o, R_o = CK.pinch_tcp([-0.35, -0.13, 0.1226], 0.3, 0.035, True)
p_o2, _ = CK.pinch_tcp([-0.35, -0.13, 0.1226], 0.3, 0.035, True, profile=O)
assert np.allclose(p_o, p_o2) and np.allclose(R_o, R0)
S = CK.so101_profile()
assert S.flips == (True,) and abs(S.approach_clearance_m - 0.002) < 1e-12 and S.approach_along_wall
assert S.lift_settle_s == 2.5 and S.release_tilt_max_deg == 9.0
assert S.release_gap_max_mm == 35.0 and abs(S.open_mm - G.opening_mm_for_tip_gap(32.0)) < 1e-9
off = S.pre_offset(R0, 0.09)
assert abs(off[2] - 0.09) < 1e-12 and abs(float(np.dot(off / np.linalg.norm(off), -R0[:, 2])) - 1.0) < 1e-9
print("[U5] official profile = V3.63; SO-101: fixed jaw outside, 2 mm clearance, approach along the wall, settle 2.5 s after lift / carry, release gap 35 mm, release lean <= 9 deg")

# ------------------------------------------------------------------ U6 + U7
if HAVE_MJ:
    for src, dst in (("cup_1", "cup_2"), ("cup_2", "cup_1")):
        res, st, io = CK.twin_episode(log=lambda s_: None, gripper="so101", source=src, target=dst)
        assert st["nested_full"], (src, dst, [r_.get("reason") for r_ in res if not r_["ok"]])
        assert ("release sidestep", "line") in io.skills.motions and io.skills.prof.name == "so101"
        assert float(io.skills.k["cup_release_gap_mm"]) == 35.0
    print("[U6] twin SO-101: cup_1 -> cup_2 and cup_2 -> cup_1 fully nested (release sidestep executed)")
    import v362_backend_twin as HB  # noqa: E402
    r7 = HB.run_backend_episode(gripper="so101")
    assert r7["state"]["nested_full"] and r7["scores"][0]["s"] == 1.0, r7["log"][-8:]
    assert any("gripper: SO-101 gripper on a 7 mm ring, roll -45 deg" in l for l in r7["log"])
    assert os.environ.get("NERO_SCENE") is None
    gl_ok = False
    try:
        CK.TwinIO(gripper="so101").render_fixed_camera()
        gl_ok = True
    except Exception as exc:
        gl_reason = str(exc)[:80]
    if gl_ok:
        r7c = HB.run_backend_episode("Stack cup_2 into cup_1", "cup_2", "cup_1", gripper="so101", pose_mode="camera")
        assert r7c["state"]["nested_full"] and r7c["scores"][0]["s"] == 1.0, r7c["log"][-8:]
        print("[U7] gui_run_task + BackendIO with the SO-101: nested, s=1 (simulator truth and fixed D435i)")
    else:
        print(f"[U7] gui_run_task + BackendIO with the SO-101: nested, s=1; camera run skipped: no OpenGL ({gl_reason})")
else:
    print("[U6] [U7] skipped: MuJoCo not installed")

# ------------------------------------------------------------------ U8
study = json.loads((ROOT / "config" / "cup_stack_study_v364_so101.json").read_text(encoding="utf-8"))
assert study["version"] == "V3.64" and study["summary"]["runs"] >= 18
assert study["summary"]["nested_full"] == study["summary"]["runs"] and study["summary"]["camera_runs"] >= 6
assert all(r_["gripper"] == "so101" for r_ in study["rows"])
SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print(f"[U8] SO-101 study {study['summary']['nested_full']}/{study['summary']['runs']} nested "
      f"(camera {study['summary']['camera_nested_full']}/{study['summary']['camera_runs']}); release folder unchanged")
print("ALL V3.64 TESTS PASSED")
