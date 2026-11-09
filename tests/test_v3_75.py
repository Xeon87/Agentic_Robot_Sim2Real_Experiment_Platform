#!/usr/bin/env python3
"""V3.75 offline tests (sort_items bins + MoveIt collision pre-check; new SO-101 adapter ring):
G1 Versions V3.75 + markers; frozen modules unchanged; the backend is not changed by V3.75.
G2 Bins: inner 23 x 17 cm, walls 5 cm, centre y = +-0.28; every item footprint (default poses and jittered seeds) is
   >= 7 cm (default poses; >= 4.5 cm with the optional +-15 mm / +-15 deg jitter) from the inner walls (V3.72: 1-2 cm);
   the outer walls stay on the table.
G3 Bin fit: the banana footprint is 20 cm end to end; it fits a bin only with its long axis along the bin depth (x);
   rectangle clearance / overlap; _slot picks a fitting centre.
G4 SO-101 mount: roll default 0 (new ring, V3.64-V3.74: -45); NERO_V364_SO101_ROLL_DEG still overrides; the real
   mount default follows; saved real settings of the old ring are migrated once (roll + 45, fingertip correction
   reset, old values kept in pre_v375, mount_rev V3.75); settings saved by V3.75 are not touched.
G5 Collision geometry: at the planned banana grasp the SO-101 boxes are clear of the table and bins, while the
   estimated official-gripper envelope (MoveIt's model) is in the table - the user's V3.74 failure; pairs are
   classified (gripper model -> model_only, arm link / world -> blocking).
G6 Backend path with a MoveIt emulation (official gripper envelope; Cartesian refusal "only xx% collision-free",
   joint goal "FAILURE (99999)"): V3.75 logs the colliding pairs ([V3.75 MOVEIT]), drives the skill's own checked
   straight path only where MoveIt's gripper model alone collides, 4/4 sorted.
G7 Same with NERO_V375_SO101_OWN_PATH=0: the planning pre-check rejects the grasp angles and pick_object reports
   GRASP_UNREACHABLE with collision_rejects naming agx_gripper_* <-> mujoco_play_table (no motion into the table).
G8 Twin (no MoveIt): scripted episode 4/4 with the new bins; the place reports the item yaw and a fitting slot.
G9 Release folder unchanged.
"""
import hashlib
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v375_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
os.environ["NERO_V368_TASK_DIR"] = str(TMP / "task")
for k in ("NERO_SCENE", "NERO_V343_ACTIVE_SCENE", "NERO_V372_ITEM_SEED", "NERO_V364_SO101_ROLL_DEG",
          "NERO_V364_SO101_RING_MM", "NERO_V364_SO101_SEAT_X_MM", "NERO_V375_SO101_OWN_PATH"):
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
import NERO_V343_Scenes as SC               # noqa: E402
import NERO_V347_Yolo_Perception as YP      # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49   # noqa: E402
import NERO_V364_SO101_Gripper as G         # noqa: E402
import NERO_V365_Real_Cup_Task as RT        # noqa: E402
import NERO_V372_Sort_Skills as S           # noqa: E402
import NERO_V375_Collision_Check as CC      # noqa: E402

try:
    import mujoco  # noqa: F401
    HAVE_MJ = True
except Exception:
    HAVE_MJ = False

# ------------------------------------------------------------------ G1
for m in (CON, RR, YP, W49, S, CC):
    assert m.VERSION in ("V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10"), (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10") and SC.SCENES_REV in ("V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
for fn, h in (("NERO_V2_7_1_Unified_Backend.py", "5be2136f84ec4ccdfe6f8bc8d8998410ff9b3b827af4d571ea44373dc427527e"),
              ("NERO_V2_7_1_Agent.py", "3e52d05ea84d94c8359d5875fa0fee8c2cc42d612b52e59109e1166912439696"),
              ("NERO_V325_Camera.py", "82f2416d73671e7018d09d0cd72c17186154934a315b47c0d19fdb0427afd931"),
              ("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
for fn, mk in (("NERO_V372_Sort_Skills.py", "V3.75 BIN FIT + COLLISION PRE-CHECK"),
               ("NERO_V372_Sort_Skills.py", "V3.75 OWN PATH WHEN ONLY MOVEIT'S GRIPPER MODEL COLLIDES"),
               ("NERO_V375_Collision_Check.py", "V3.75 COLLISION PRE-CHECK"),
               ("NERO_V362_Cup_Skills.py", "V3.75 COLLISION PRE-CHECK"),
               ("NERO_V343_Scenes.py", "V3.75 SMALLER BINS FARTHER OUT"), ("NERO_V343_Scenes.py", "V3.75 ITEM FOOTPRINT"),
               ("NERO_V364_SO101_Gripper.py", "V3.75 NEW ADAPTER RING"), ("NERO_V365_Real_Cup_Task.py", "V3.75 NEW ADAPTER RING")):
    assert mk in (ROOT / fn).read_text(encoding="utf-8"), (fn, mk)
for f in ("start_nero.sh", "start_nero_sim.sh"):
    assert "NERO_V375_Collision_Check.py" in (ROOT / f).read_text(encoding="utf-8"), f
print("[G1] V3.75; frozen modules and the backend unchanged; markers present")

# ------------------------------------------------------------------ G2
for col, b in SC.SORT_BINS.items():
    assert tuple(b["inner"]) == (0.23, 0.17) and b["wall_h"] == 0.05 and abs(abs(b["centre"][1]) - 0.28) < 1e-9, b
    assert abs(b["centre"][1]) + b["inner"][1] / 2 + b["wall_t"] < 0.38                     # table half width
def gap(seed):
    w = 1e9
    for n, (x, y, yaw) in SC.sort_item_poses(seed).items():
        rc, half, a = SC.sort_item_footprint(n, (x, y), yaw)
        for px, py in CC.rect_corners(rc, half, a):
            for b in SC.SORT_BINS.values():
                w = min(w, abs(b["centre"][1]) - b["inner"][1] / 2 - abs(py))
    return w


g0 = gap(None)
gj = min(gap(s_) for s_ in range(1, 9))
assert g0 >= 0.070 and gj >= 0.045, (g0, gj)            # jitter (NERO_V372_ITEM_SEED, +-15 mm / +-15 deg)
print(f"[G2] bins 23 x 17 x 5 cm at y = +-0.28; nearest item corner {g0 * 1000:.0f} mm from an inner wall "
      f"(with the optional jitter {gj * 1000:.0f} mm; V3.72: 10-20 mm)")

# ------------------------------------------------------------------ G3
rc, half, a = SC.sort_item_footprint("obj_banana", (0.0, 0.0), 0.0)
assert abs(2 * half[0] - 0.2012) < 0.002 and abs(2 * half[1] - 0.0644) < 0.002 and abs(rc[1] - 0.0152) < 0.001
b = SC.SORT_BINS["red"]


def fits(yaw):
    r = SC.sort_item_footprint("obj_banana", b["centre"], yaw)
    ix, iy = b["inner"][0] / 2 - S.FIT_MARGIN_M, b["inner"][1] / 2 - S.FIT_MARGIN_M
    return all(abs(px - b["centre"][0]) <= ix and abs(py - b["centre"][1]) <= iy for px, py in CC.rect_corners(*r))


assert fits(0.0) and fits(180.0) and not fits(90.0) and not fits(45.0)
r1 = ((0.0, 0.0), (0.05, 0.01), 0.0)
assert abs(CC.rect_dist(r1, ((0.0, 0.05), (0.05, 0.01), 0.0)) - 0.03) < 1e-9
assert CC.rect_dist(r1, ((0.02, 0.0), (0.05, 0.01), 0.5)) == 0.0
print("[G3] banana 201 x 64 mm fits a bin only lengthwise (0/180 deg yes, 45/90 deg no); rectangle clearance ok")

# ------------------------------------------------------------------ G4
# V3.76 (intended change): the SO-101 opens up / down, moving jaw on top -> roll default -90 (V3.75: 0); rev V3.76
assert G.MOUNT["roll_deg"] == -90.0 and G.describe()["roll_deg"] == -90.0
gp = G.describe()["grasp_point_in_tcp_mm"]
assert math.hypot(gp[0], gp[1]) < 5.0 and abs(gp[2] + 18.8) < 0.2, gp    # roll about link7 x moves the 2 mm axis offset
r = subprocess.run([sys.executable, "-c", "import NERO_V364_SO101_Gripper as G; print(G.MOUNT['roll_deg'])"],
                   cwd=str(ROOT), env=dict(os.environ, NERO_V364_SO101_ROLL_DEG="-45"), capture_output=True, text=True)
assert r.stdout.strip() == "-45.0", r.stdout + r.stderr
_T, _Tg, mnt = RT.real_mount({})
assert mnt["roll_deg"] == -90.0 and mnt["ring_mm"] == 7.0
old = {"clearance_mm": 8.0, "roll_deg": -45.0, "tip_corr_mm": [1.5, -2.0, 0.5], "ring_mm": 7.0, "seat_mm": 37.5}
logs = []
new, lines = RT.migrate_geometry_v375(old, log=logs.append)
assert new["roll_deg"] == -90.0 and new["tip_corr_mm"] == [0.0, 0.0, 0.0] and new["mount_rev"] == "V3.76"
assert new["pre_v375"]["roll_deg"] == -45.0 and new["pre_v375"]["tip_corr_mm"] == [1.5, -2.0, 0.5] and len(lines) == 2
assert logs and "转接环" in logs[0] and new["clearance_mm"] == 8.0
assert RT.migrate_geometry_v375(new)[1] == [] and RT.migrate_geometry_v375({"roll_deg": -43.0})[0]["roll_deg"] == -88.0
assert RT.migrate_geometry_v375({})[0] == {}
print("[G4] SO-101 roll -90 (V3.76; env override still works); real mount -90 deg; old settings: -45 -> -90, "
      "fingertip correction reset, kept in pre_v375; current settings untouched")

# ------------------------------------------------------------------ G5
if HAVE_MJ:
    tw = S.SortTwinIO()
    sk = S.SortSkills(tw)
    it = sk._item("item_1")
    plan = sk._plan_pick(it, tw.current_q())
    assert plan is not None, sk.v375_rejects
    qg = plan["q"]["grasp"]
    open_v = G.opening_mm_for_tip_gap(plan["gap_mm"])
    assert CC.check_parts(CC.so101_parts(qg, open_v), CC.world_boxes("sort_items")) == []
    hit = CC.check_parts(CC.official_gripper_parts(qg, open_v), CC.world_boxes("sort_items"), 0.0)
    assert ("agx_gripper_finger_1", "mujoco_play_table") in hit, hit
    T = CC.link_frames(qg)["link7"] @ G.T_L7_TCP
    print(f"[G5] banana grasp: official TCP z = {T[2, 3] * 1000:+.1f} mm (table top 0) - the official-gripper model is "
          f"in the table ({CC.pairs_text(hit)}), the SO-101 boxes are clear")
    bl, mo = CC.classify_pairs([("agx_gripper_finger_1", "mujoco_play_table"), ("link6", "sort_bin_red_wall_ypos"),
                                ("gripper_base", "link5")], CC.world_boxes("sort_items"))
    assert mo == [("agx_gripper_finger_1", "mujoco_play_table"), ("gripper_base", "link5")]
    assert bl == [("link6", "sort_bin_red_wall_ypos")]
    a = CC.assess(None, qg, open_v, CC.world_boxes("sort_items"))
    assert a["ok"] and not a["moveit_available"]

    # -------------------------------------------------------------- G6
    import v375_moveit_twin as MT  # noqa: E402
    r = MT.run_backend_episode()
    assert r["started"].get("label") == "RUN TASK (sort_items)" and not r["started"].get("error"), r["started"]
    assert r["state"] == {"item_1": "red", "item_2": "red", "item_3": "blue", "item_4": "blue"}, r["state"]
    info = [l for l in r["log"] if "[INFO] [V3.75 MOVEIT]" in l and "collides in MoveIt" in l]
    own = [l for l in r["log"] if "[WARN] [V3.75 MOVEIT]" in l and "own straight path" in l]
    assert info and "mujoco_play_table" in info[0] and "agx_gripper_finger" in info[0], info[:1]
    assert own and getattr(r["core"], "own_paths", 0) == len(own), (len(own), getattr(r["core"], "own_paths", 0))
    assert any("down to item_1" in l for l in own)
    assert not any("JOINT-GOAL PLANNING FAILED" in l and "pick_object" in l for l in r["log"])
    assert r["scores"] and r["scores"][0]["s"] == 1.0
    print(f"[G6] MoveIt emulation: pairs logged ({len(info)} refusals), own checked path on {len(own)} segments "
          f"(only MoveIt's gripper model collided), 4/4 sorted")

    # -------------------------------------------------------------- G7
    os.environ["NERO_V375_SO101_OWN_PATH"] = "0"
    try:
        r = MT.run_backend_episode()
    finally:
        os.environ.pop("NERO_V375_SO101_OWN_PATH")
    fails = [l for l in r["log"] if "[FAIL] [V3.72 SORT] pick_object: GRASP_UNREACHABLE" in l]
    assert fails and "collision_rejects" in fails[0] and "agx_gripper_finger" in fails[0] and \
        "mujoco_play_table" in fails[0], fails[:1]
    assert r["state"]["item_1"] is None and not any("own straight path" in l for l in r["log"])
    print("[G7] own path off: GRASP_UNREACHABLE with collision_rejects agx_gripper_finger_* <-> mujoco_play_table, "
          "no motion into the table")

    # -------------------------------------------------------------- G8
    tw = S.SortTwinIO()
    sk = S.SortSkills(tw)
    out = sk.scripted_episode()
    st = {iid: it["in_bin"] for iid, it in sk._items().items()}
    assert st == {"item_1": "red", "item_2": "red", "item_3": "blue", "item_4": "blue"}, st
    places = [o for o in out if o.get("tool") == "place_in_box"]
    pv = places[0]["physical_verification"]
    assert abs(math.sin(math.radians(pv["item_yaw_deg"]))) < 0.27, pv            # banana lengthwise in the bin
    for o in places:
        p_ = o["physical_verification"]
        b_ = SC.SORT_BINS[p_["box"]]
        assert abs(p_["slot_xy_m"][1] - b_["centre"][1]) <= b_["inner"][1] / 2 and \
            abs(p_["slot_xy_m"][0] - b_["centre"][0]) <= b_["inner"][0] / 2, p_
    print(f"[G8] twin: 4/4 sorted into the new bins; banana yaw {pv['item_yaw_deg']} deg (lengthwise)")
else:
    print("[G5-G8] skipped: MuJoCo not installed")

SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("[G9] release folder unchanged")
print("ALL V3.75 TESTS PASSED")
