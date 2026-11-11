#!/usr/bin/env python3
"""V3.67 offline tests (real cup task: jaw line along the cup radius, safer descent, robust perception, empty clamp):
Y1  Versions V3.67 + markers; NERO_V308 / Camera tab / V3.63 colour perception / backend / Agent / scenes / SO-101 model
    unchanged; the simulation's gripper profiles keep their V3.64 behaviour (no align-rotate, no descent stop, no extra
    wide search, SO-101 clearance 2 mm).
Y2  Real gripper geometry: real_mount() with no measurements = the NERO_V364 model exactly; a roll change rotates the
    gripper about link7 +x by that angle; the fingertip correction from a 4-pose touch calibration round-trips
    (model tip + offset -> the offset in the gripper frame) and > 25 mm is refused; the real profile has clearance 8 mm,
    align 60 mm, descent stop 10 mm, wide search below 8 deg.
Y3  Empty-clamp calibration (simulated STS3215 whose tips touch at +1.85 deg like the real one): calibrate_touch finds
    the touch angle; afterwards an empty clamp is NOT held (V3.66: held at 2.4 mm), a 0.6 mm wall is held, and
    open_gap(32) gives a 32 mm gap from the real touch point; nothing in the jaws -> calibration refused.
Y4  Perception: on the twin camera upright cups give a reliable tilt (< 3 deg); noisy wall depth -> tilt not trusted
    (0, warning) instead of 40 deg; the gripper at the rim -> kind "occluded" with the position from the lower wall
    (< 5 mm); a gripper above but not hiding the top does not change the result; failure snapshots go to
    NERO_CALIB_DIR/real_cup_logs (colour, depth, result) and only the newest N are kept.
Y5  Plan contract: a 4-8 deg plan is ok with executable=true and an "EXECUTABLE ... continue with pick_cup" warning;
    the agent rules say so (6c).
Y6  RealIO on the twin (real profile): mode 3 cup_1 -> cup_2 nests and verifies; the pick goes align above ->
    rotate in place -> down to the pregrasp -> stop 10 mm above the rim (camera check) -> grasp; at the stop the
    closing direction (fixed jaw -> moving jaw) lies along the cup radius within 5 deg; mode 2 uses the same
    align-rotate and never clamps.
Y7  Tab (mocked Tk): geometry fields with defaults, range checks, fingertip correction from the Camera tab's touch
    calibration, settings (geometry + gripper touch point) saved in NERO_CALIB_DIR; release folder unchanged.
"""
import hashlib
import json
import math
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v367_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
for k in ("NERO_SCENE", "NERO_V343_ACTIVE_SCENE", "NERO_V362_CUP_SOLREF", "NERO_V308_DEPTH_NOISE_K",
          "NERO_V364_SO101_CLEARANCE_MM", "NERO_V364_SO101_ROLL_DEG"):
    os.environ.pop(k, None)
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
import NERO_V362_Cup_Skills as CK           # noqa: E402
import NERO_V364_SO101_Gripper as G         # noqa: E402
import NERO_V365_Real_Cup_Task as RT        # noqa: E402
import NERO_V365_STS3215_Gripper as GR      # noqa: E402
import NERO_V366_Real_Cup_Yolo as YC        # noqa: E402

try:
    import mujoco  # noqa: F401
    HAVE_MJ = True
except Exception:
    HAVE_MJ = False

# ------------------------------------------------------------------ Y1
for m in (CON, RR, YP, W49, RT, GR, YC):
    assert m.VERSION in ("V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3", "V4.8.4"), (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3", "V4.8.4")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
for fn, h in (("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a"),
              ("NERO_V2_7_1_Agent.py", "3e52d05ea84d94c8359d5875fa0fee8c2cc42d612b52e59109e1166912439696"),
              ("NERO_V325_Camera.py", "82f2416d73671e7018d09d0cd72c17186154934a315b47c0d19fdb0427afd931"),
              ("NERO_V2_7_1_Unified_Backend.py", "5be2136f84ec4ccdfe6f8bc8d8998410ff9b3b827af4d571ea44373dc427527e"),
              ("NERO_V331_Fixed_Camera.py", "8314f82cce52016f3dd08bad22ce08dbd9bdb1b263fa6aef23453467a9b3a341"),
              ("NERO_V363_Cup_Perception.py", "b9633693657127fccc101bd902ad38e6bbc014fc907c41bd54227c1711a732c5")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
for fn in ("NERO_V343_Scenes.py", "NERO_V364_SO101_Gripper.py", "NERO_V359_Strategy.py"):
    assert "V3.67" not in (ROOT / fn).read_text(encoding="utf-8"), fn
src_ck = (ROOT / "NERO_V362_Cup_Skills.py").read_text(encoding="utf-8")
for fn, mk in (("NERO_V362_Cup_Skills.py", "V3.67 ALIGN-ROTATE-DESCEND"), ("NERO_V362_Cup_Skills.py", "V3.67 TWO-STAGE DESCENT"),
               ("NERO_V362_Cup_Skills.py", "V3.67 PLAN MARGIN"), ("NERO_V365_STS3215_Gripper.py", "V3.67 EMPTY-CLAMP CALIBRATION"),
               ("NERO_V365_Real_Cup_Task.py", "V3.67 REAL GRIPPER GEOMETRY"),
               ("NERO_V365_Real_Cup_Task.py", "V3.67 FAILURE SNAPSHOT"), ("NERO_V366_Real_Cup_Yolo.py", YC.MARKER_V367)):
    assert mk in (ROOT / fn).read_text(encoding="utf-8"), (fn, mk)
for prof in (CK.OFFICIAL_PROFILE, CK.so101_profile()):
    assert prof.align_rotate_m == 0.0 and prof.descent_stop_above_rim_m is None and prof.wide_search_below_deg is None
assert abs(CK.so101_profile().approach_clearance_m - 0.002) < 1e-12
print("[Y1] V3.67; frozen modules unchanged; simulation gripper profiles keep the V3.64 behaviour")

# ------------------------------------------------------------------ Y2
T_l7g, T_tg, mnt = RT.real_mount({})
assert np.abs(T_l7g - G.T_l7_gripper()).max() < 1e-12 and np.abs(T_tg - G.T_tcp_grasp()).max() < 1e-12
assert mnt == {"seat_mm": 37.5, "ring_mm": 7.0, "roll_deg": -90.0, "tip_corr_mm": [0.0, 0.0, 0.0]}   # V3.76 (V3.75: 0, before: -45)
T2 = RT.real_mount({"roll_deg": -30.0})[0]
dR = T2[:3, :3] @ T_l7g[:3, :3].T
assert np.abs(dR - G._rx(60.0)).max() < 1e-9                     # V3.76: relative to the default -90 (V3.75: -30 from 0)
T3 = RT.real_mount({"seat_mm": 40.0, "ring_mm": 9.0})[0]
assert abs(T3[0, 3] - 0.049) < 1e-12
off = np.array([0.002, -0.003, 0.004])
p_tip, _ = RT.model_fixed_tip_l7({})
corr, _raw = RT.tip_corr_from_touch(p_tip + off, {})
assert np.allclose(T_l7g[:3, :3] @ (np.asarray(corr) / 1000.0), off, atol=1e-5), corr
T4 = RT.real_mount({"tip_corr_mm": corr})[1]
p_tip4, _ = RT.model_fixed_tip_l7({"tip_corr_mm": corr})
assert np.allclose(p_tip4, p_tip + off, atol=1e-5)
assert RT.tip_corr_from_touch(p_tip + [0.03, 0, 0], {})[0] is None
rp = RT.real_profile(CK)
assert abs(rp.approach_clearance_m - 0.008) < 1e-12 and abs(rp.align_rotate_m - 0.06) < 1e-12
assert abs(rp.descent_stop_above_rim_m - 0.010) < 1e-12 and rp.wide_search_below_deg == 8.0 and rp.flips == (True,)
print(f"[Y2] real mount = model without measurements; roll / seat / ring / fingertip correction applied; correction "
      f"from touch {corr} mm; real profile 8 mm / 60 mm / 10 mm")

# ------------------------------------------------------------------ Y3
sim = GR.SimServo()
sim.obstacle_gap_mm = GR.gap_mm_for_open_deg(1.85)            # the real tips touch at +1.85 deg (V3.66 log)
grip = GR.SO101RealGripper(log=lambda s: None)
ok, msg = grip.connect(GR.SIM_PORT, sim_servo=sim)
assert ok, msg
r0 = grip.clamp(18.3)
assert r0["held"], r0                                          # V3.66 behaviour: empty clamp read as held
cal = grip.calibrate_touch()
assert cal["ok"] and abs(cal["touch_deg"] - 1.85) < 0.35, cal
assert grip.cfg["touch_calibrated"]
r1 = grip.clamp(18.3)
assert not r1["held"], r1
grip.open_deg(10.0)
sim.obstacle_gap_mm = GR.gap_mm_for_open_deg(1.85) + 0.6       # a 0.6 mm party-cup wall between the tips
r2 = grip.clamp(18.3)
assert r2["held"] and 0.3 <= r2["gap_mm"] <= 1.2, r2
sim.obstacle_gap_mm = None
r3 = grip.open_gap(32.0)
raw_a = grip.cfg["touch_deg"] + r3["open_deg"]
assert abs(r3["gap_mm"] - 32.0) < 0.8 and abs(raw_a - (grip.cfg["touch_deg"] + GR.open_deg_for_gap_mm(32.0))) < 0.6
sim.obstacle_gap_mm = GR.gap_mm_for_open_deg(9.0)              # something big in the jaws
g2 = GR.SO101RealGripper(log=lambda s: None)
g2.connect(GR.SIM_PORT, sim_servo=sim)
bad = g2.calibrate_touch()
assert not bad["ok"] and not g2.cfg["touch_calibrated"], bad
sim.obstacle_gap_mm = None
bad2 = g2.calibrate_touch()                                    # nothing stops the jaw
assert not bad2["ok"] and "没有停住" in bad2["error"], bad2
print(f"[Y3] empty-clamp calibration: touch {cal['touch_deg']:+.2f} deg; empty clamp not held, 0.6 mm wall held "
      f"(gap {r2['gap_mm']} mm); 32 mm from the real touch point; bad calibrations refused")

# ------------------------------------------------------------------ Y4 / Y6 need the twin camera
gl_ok = False
if HAVE_MJ:
    try:
        CK.TwinIO(gripper="so101").render_fixed_camera()
        gl_ok = True
    except Exception as exc:
        gl_reason = str(exc)[:80]
if gl_ok:
    import v365_real_twin as RTW  # noqa: E402
    tw = CK.TwinIO(gripper="so101", pose_mode="camera")
    camfn = RTW.twin_camera(tw)
    cam = camfn()
    res = YC.perceive_cups_yolo(cam)
    assert all(c["kind"] == "cup" and c["tilt_reliable"] and c["tilt_deg"] < 3.0 for c in res["cups"]), YC.summary(res)
    # noisy wall depth (the real D435i on a glossy red wall): no 40 deg tilt any more
    rng = np.random.default_rng(1)
    dep = cam["depths"][0].copy()
    for d in cam["yolo"]["detector"].next:
        m = d["mask"] & (dep > 0)
        dep[m] = dep[m] + rng.normal(0.0, 0.012, int(m.sum())).astype(dep.dtype)
    res_n = YC.perceive_cups_yolo(dict(cam, depths=[dep]))
    for c in res_n["cups"]:
        assert c["tilt_deg"] < 6.0, (c["tilt_deg"], c.get("tilt_measured_deg"), c.get("warnings"))
    assert any(not c["tilt_reliable"] and any("倾斜测不准" in w for w in c["warnings"]) for c in res_n["cups"]), \
        [(c["tilt_reliable"], c.get("tilt_measured_deg")) for c in res_n["cups"]]
    # the gripper at the rim of cup_1 (box like RealIO.gripper_boxes), then high above it
    r1_ = tw.cup_pose("cup_1")[0]
    box_at = [(np.array([r1_[0] + 0.045, r1_[1], r1_[2] + 0.05]), np.eye(3), np.array([0.02, 0.03, 0.06]))]
    res_o = YC.perceive_cups_yolo(cam, robot_boxes=box_at)
    c1 = min(res_o["cups"], key=lambda c_: math.hypot(c_["base_xyz_m"][0] - r1_[0], c_["base_xyz_m"][1] - r1_[1]))
    assert c1["kind"] == "occluded", YC.summary(res_o)
    assert math.hypot(c1["base_xyz_m"][0] - r1_[0], c1["base_xyz_m"][1] - r1_[1]) < 0.005, c1["base_xyz_m"]
    box_hi = [(np.array([r1_[0], r1_[1], r1_[2] + 0.35]), np.eye(3), np.array([0.02, 0.03, 0.03]))]
    res_h = YC.perceive_cups_yolo(cam, robot_boxes=box_hi)
    assert sorted(c_["kind"] for c_ in res_h["cups"]) == ["cup", "cup"], YC.summary(res_h)
    snapdir = TMP / "snaps"
    paths = [YC.save_snapshot(cam, res_o, f"t{i}", folder=str(snapdir), keep=2) for i in range(3)]
    left = sorted(os.listdir(snapdir))
    assert len(left) == 2 and os.path.basename(paths[-1]) in left
    for fn in ("color.png", "depth_mm.png", "result.json"):
        assert os.path.isfile(os.path.join(paths[-1], fn)), fn
    meta = json.loads(Path(paths[-1], "result.json").read_text(encoding="utf-8"))
    assert meta["dets"] and "contour" in meta["dets"][0]
    print("[Y4] tilt reliable on clean depth, not trusted on noisy depth (no false 40 deg); gripper at the rim -> "
          "occluded with the position (< 5 mm); gripper high above -> unchanged; snapshots kept newest N")

    # ---------------------------------------------------------------- Y6
    log6 = []
    io, tw6, arm6, grip6 = RTW.make(log=log6.append)
    axis_xy = np.asarray(tw6.cup_pose("cup_1")[0][:2], float)          # the source cup before the task
    out = RT.run_execute(io, "cup_1", "cup_2")
    a_, b_ = tw6.cup_pose("cup_1"), tw6.cup_pose("cup_2")
    st6 = CK.SC.cup_task_state(a_[0], a_[1], b_[0], b_[1], origin="rim")
    assert out["ok"] and st6["nested_full"], [(r_.get("tool"), r_.get("reason")) for r_ in out["results"] if not r_.get("ok")]
    labels = [c_["label"] for c_ in arm6.calls]
    i_al = next(i for i, l in enumerate(labels) if "align above cup_1" in l)
    i_dn = next(i for i, l in enumerate(labels) if "down to the pregrasp above cup_1" in l)
    i_st = next(i for i, l in enumerate(labels) if "stop above the rim of cup_1" in l)
    i_gr = next(i for i, l in enumerate(labels) if l.endswith("grasp cup_1"))
    assert i_al < i_dn < i_st < i_gr, labels[:8]
    assert any("check cup_1 before contact" in l for l in log6)
    pick = next(r_ for r_ in out["results"] if r_.get("tool") == "pick_cup")
    assert "align_rotate" in pick["physical_verification"] and "pre_contact_check" in pick["physical_verification"]
    # closing direction at the stop: along the cup radius (fixed jaw outside -> moving jaw towards the axis)
    q_stop = arm6.calls[i_st]["q_end"]
    T = CK.FastArm().fk(np.asarray(q_stop, float))
    Tg = T @ io.prof.T_tcp_grasp
    ycl = Tg[:3, 1]
    gpt = Tg[:3, 3]
    to_axis = axis_xy - gpt[:2]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(ycl[:2] @ to_axis) /
                                                (np.linalg.norm(ycl[:2]) * np.linalg.norm(to_axis))))))
    assert ang < 5.0, ang
    io, tw7, arm7, grip7 = RTW.make()
    outh = RT.run_hover(io, "cup_1", "cup_2")
    assert outh["ok"] and not grip7.clamps and any("rotate to the grasp angle" in c_["label"] or "align above" in c_["label"]
                                                   for c_ in arm7.calls)
    print(f"[Y6] twin, real profile: align -> rotate -> down -> stop 10 mm above the rim -> grasp; closing direction "
          f"{ang:.1f} deg from the cup radius; nested and verified; hover uses the same align-rotate, no clamp")
else:
    print(f"[Y4/Y6] skipped: no twin camera ({gl_reason if HAVE_MJ else 'MuJoCo not installed'})")

# ------------------------------------------------------------------ Y5
sk = CK.CupSkills(types.SimpleNamespace(pose_mode=lambda: "truth", emit=lambda s: None,
                                        gripper_profile=lambda: CK.OFFICIAL_PROFILE,
                                        current_q=lambda: [0.0] * 7,
                                        cup_pose=lambda n: ([-0.35, -0.13 if n == "cup_1" else 0.13, 0.1226],
                                                            [1.0, 0.0, 0.0, 0.0])), *CK.cup_knobs())
_orig = CK.plan_cup_stack
fake = {"ok": True, "min_joint_margin_deg": 7.0, "margin_ok": False, "grasp_margin_deg": 7.0, "place_margin_deg": 9.0,
        "grasp_binding": "J4", "place_binding": "J6", "_wps": {}, "_q": {}, "_qpath": {}}
CK.plan_cup_stack = lambda *a, **k: dict(fake)
try:
    r5 = sk.plan_cup_stack("cup_1", "cup_2")
finally:
    CK.plan_cup_stack = _orig
assert r5["ok"] and r5["plan"]["executable"] and any("EXECUTABLE" in w and "pick_cup" in w for w in r5["warnings"]), r5
assert "6c." in CK.CUP_SYSTEM_INSTRUCTIONS and "informational" in CK.CUP_SYSTEM_INSTRUCTIONS
print("[Y5] 4-8 deg plans: ok + executable=true + 'EXECUTABLE ... continue with pick_cup'; agent rule 6c")

# ------------------------------------------------------------------ Y7
import unittest.mock as mock  # noqa: E402


class FakeVar:
    def __init__(self, master=None, value=None, **_kw):
        self._v = value

    def get(self):
        return self._v

    def set(self, v):
        self._v = v

    def trace_add(self, *_a, **_k):
        return "t"


mtk, mttk = mock.MagicMock(), mock.MagicMock()
for nm in ("StringVar", "BooleanVar", "IntVar", "DoubleVar"):
    setattr(mtk, nm, FakeVar)
mtk.Text.return_value.yview.return_value = (0.0, 1.0)


class _Pipe:
    teach_mode = False

    def lamp(self, k):
        return "green"

    def supports_joint_path(self):
        return True

    def actual_arm_deg(self, m):
        return [0.0] * 7

    def move_busy(self):
        return False


class _Cam:
    src = wrist_src = worker = result = calib = None
    tcp_calib = {"tcp_in_flange": [float(v) for v in (RT.model_fixed_tip_l7({})[0] + [0.0, 0.0, 0.003])]}


logs7 = []
tab = RT.RealCupTaskTabUI(mtk, mttk, mock.MagicMock(), _Pipe(), types.SimpleNamespace(model={"joints": []}), _Cam(),
                          log=logs7.append, visible_fn=lambda: False)
gc = tab.geometry_cfg()
assert gc["clearance_mm"] == 8.0 and gc["align_rotate_mm"] == 60.0 and gc["descent_stop_mm"] == 10.0
assert gc["seat_mm"] == 37.5 and gc["ring_mm"] == 7.0 and gc["roll_deg"] == -90.0     # V3.76 (V3.75: 0, before: -45)
assert "原地转腕" in tab.v_geo.get() and "固定指离杯壁 8 mm" in tab.v_geo.get()
tab.v_clear.set("1")
try:
    tab.geometry_cfg()
    raise AssertionError("clearance 1 mm must be refused")
except ValueError:
    pass
tab.v_clear.set("9")
tab.on_tip_from_touch()
tab.poll()
assert any("指尖修正" in l for l in logs7), logs7[-3:]
# the touch point is 3 mm from the model tip (link7 z)
assert abs(float(np.linalg.norm(tab.tip_corr)) - 3.0) < 0.05, tab.tip_corr
tab.v_roll.set("-40")
tab.grip.cfg.update(touch_deg=1.9, touch_calibrated=True)
tab._save()
saved = json.loads((Path(os.environ["NERO_CALIB_DIR"]) / "real_cup_task.json").read_text(encoding="utf-8"))
assert saved["geometry"]["clearance_mm"] == 9.0 and saved["geometry"]["roll_deg"] == -40.0
assert saved["geometry"]["tip_corr_mm"] == tab.tip_corr and saved["gripper"]["touch_deg"] == 1.9
tab2 = RT.RealCupTaskTabUI(mtk, mttk, mock.MagicMock(), _Pipe(), types.SimpleNamespace(model={"joints": []}), _Cam(),
                           log=logs7.append, visible_fn=lambda: False)
assert tab2.geometry_cfg()["clearance_mm"] == 9.0 and tab2.grip.cfg["touch_calibrated"]
print("[Y7] tab: geometry fields + checks, fingertip correction from the touch calibration, geometry / touch point "
      "saved in NERO_CALIB_DIR and restored")

SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("ALL V3.67 TESTS PASSED")
