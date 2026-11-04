#!/usr/bin/env python3
"""V3.76 offline tests (fixes after the user's V3.75 sort_items run):
I1  Versions V3.76 + markers; frozen modules unchanged; the backend is not changed by V3.76.
I2  Bin walls vs the backend's old basket-rim rule (V2-44 R2, frozen): the rule text is still in the backend; applied
    to the V3.75 full walls it cuts exactly the red bin's x- and y+ walls (the user's log); applied to every box of
    the V3.76 twin / scene model (MuJoCo geoms) it cuts nothing; the segments rebuild the same outline; MoveIt keeps
    the full walls.
I3  SO-101 standard orientation (user's choice A from the rendered A / B figure): roll default -90; with the tool
    horizontal, pointing forward and J7 = 0 the moving jaw points up; cup stack reachable in the twin; env override.
I4  Real settings migration: before V3.75 (-45) -> -90, V3.75 (0) -> -90, V3.76 untouched; fingertip correction reset.
I5  Soft pads only in the sorting scene / twin (solref 0.006); the cup gripper keeps solref 0.003.
I6  State after a safety stop: an item let go by the backend and lying in the requested bin -> place_in_box ok with
    reason ITEM_DROPPED_IN_BOX (warning), the skill no longer holds it, return_to_safe_pose works; let go elsewhere ->
    ITEM_DROPPED; a released item is noticed by place_in_box (NO_ITEM_HELD) and return_to_safe_pose.
I7  Score by the final position: a failed place whose item lies in the requested bin at the last check_bins counts.
I8  Start clear: TCP below 0.30 m -> straight up, then home; high -> no motion.
I9  Launcher timed stop: v376_stop ends processes that ignore INT / TERM (and their children / their process group)
    within 2.5 steps; cleanup uses it for MoveIt x2, backend, GPU server.  Console: stop countdown.
I10 Twin 4/4 and the backend path with the MoveIt emulation 4/4 with the -90 mount and soft pads; the held carry is
    stretched HELD_SLOW x.
I11 Release folder unchanged.
"""
import hashlib
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v376_"))
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
import NERO_V362_Cup_Skills as CK           # noqa: E402
import NERO_V364_SO101_Gripper as G         # noqa: E402
import NERO_V365_Real_Cup_Task as RT        # noqa: E402
import NERO_V372_Sort_Skills as S           # noqa: E402

try:
    import mujoco
    HAVE_MJ = True
except Exception:
    HAVE_MJ = False

# ------------------------------------------------------------------ I1
for m in (CON, RR, YP, W49, S, RT):
    assert m.VERSION == "V3.76", (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() == "V3.76" and SC.SCENES_REV == "V3.76"
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
for fn, h in (("NERO_V2_7_1_Unified_Backend.py", "5be2136f84ec4ccdfe6f8bc8d8998410ff9b3b827af4d571ea44373dc427527e"),
              ("NERO_V2_7_1_Agent.py", "3e52d05ea84d94c8359d5875fa0fee8c2cc42d612b52e59109e1166912439696"),
              ("NERO_V325_Camera.py", "a58e69b363a5a861b3ad0806ba276c49601fc214ffdfc07ffba9bdc0c2c65797"),
              ("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
for fn, mk in (("NERO_V343_Scenes.py", "V3.76 BIN WALLS OUT OF THE OLD BASKET-RIM RULE"),
               ("NERO_V364_SO101_Gripper.py", "V3.76 SO-101 STANDARD ORIENTATION"),
               ("NERO_V364_SO101_Gripper.py", "V3.76 SOFT PADS"),
               ("NERO_V372_Sort_Skills.py", "V3.76 STATE AFTER A SAFETY STOP"),
               ("NERO_V372_Sort_Skills.py", "V3.76 CARRY SLOWER WITH AN ITEM"),
               ("NERO_V372_Sort_Skills.py", "V3.76 START CLEAR"),
               ("NERO_V372_Sort_Skills.py", "V3.76 SCORE BY THE FINAL POSITION"),
               ("NERO_V365_Real_Cup_Task.py", "V3.76 SO-101 STANDARD ORIENTATION"),
               ("start_nero_sim.sh", "V3.76 TIMED STOP"), ("NERO_V312_Console.py", "V3.76 STOP COUNTDOWN")):
    assert mk in (ROOT / fn).read_text(encoding="utf-8"), (fn, mk)
print("[I1] V3.76; frozen modules and the backend unchanged; markers present")

# ------------------------------------------------------------------ I2
bsrc = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text(encoding="utf-8")
for frag in ("_thin=min(_sx,_sy) <= 0.020 and max(_sx,_sy) >= 0.045",
             "_near=(abs((_gx+_bx)-(-0.425)) <= 0.14 and abs((_gy+_by)-(-0.155)) <= 0.14)", "_target=0.014000000"):
    assert frag in bsrc, frag
assert SC.legacy_rim_hits(SC.sort_bin_boxes()) == ["sort_bin_red_wall_xneg", "sort_bin_red_wall_ypos"]
geoms = SC.sort_bin_geoms()
assert SC.legacy_rim_hits(geoms) == [] and all(max(h[0], h[1]) < 0.045 for n, c, h, _r in geoms if "_wall_" in n)
for n, c, h, _r in SC.sort_bin_boxes():                       # the segments rebuild the same wall
    segs = [g for g in geoms if g[0].startswith(n + "_") or g[0] == n]
    lo = np.min([np.subtract(g[1], g[2]) for g in segs], axis=0)
    hi = np.max([np.add(g[1], g[2]) for g in segs], axis=0)
    assert np.allclose(lo, np.subtract(c, h), atol=1e-9) and np.allclose(hi, np.add(c, h), atol=1e-9), n
if HAVE_MJ:
    m_ = mujoco.MjModel.from_xml_string(CK.twin_xml(gripper="so101", world_xml=SC.sort_world_xml(bare=True),
                                                    hold_names=SC.SORT_ITEM_NAMES, soft_pads=True))
    cut = []
    for gid in range(m_.ngeom):                                # the backend rule on the model, like its loader
        if int(m_.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        sx, sy, sz = (float(v) for v in m_.geom_size[gid])
        bid = int(m_.geom_bodyid[gid])
        gx, gy = float(m_.geom_pos[gid][0]) + float(m_.body_pos[bid][0]), float(m_.geom_pos[gid][1]) + float(m_.body_pos[bid][1])
        name = mujoco.mj_id2name(m_, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        thin = min(sx, sy) <= 0.020 and max(sx, sy) >= 0.045
        near = abs(gx + 0.425) <= 0.14 and abs(gy + 0.155) <= 0.14
        if thin and near and "floor" not in name and 2 * sz > 0.0145:
            cut.append(name)
    assert not any(n.startswith("sort_bin_") for n in cut), cut
assert len([b for b in SC.sort_bin_boxes()]) == 10                     # MoveIt planning boxes: full walls
print(f"[I2] old basket-rim rule: V3.75 walls -> {SC.legacy_rim_hits(SC.sort_bin_boxes())} (the user's log); V3.76 "
      f"{len(geoms) - 2} wall segments -> none cut; same outline; MoveIt keeps the 10 full boxes")

# ------------------------------------------------------------------ I3
assert G.MOUNT["roll_deg"] == -90.0
arm = CK.FastArm()
Th = arm.fk(np.array([0, 1.2, 0, 0.4, 0, 0, 0.0]))                     # tool horizontal, pointing forward, J7 = 0
assert Th[0, 2] < -0.99, Th[:3, 2]
T7 = Th @ np.linalg.inv(arm.tool)
up = (T7[:3, :3] @ G.R_l7_gripper())[:, 0]                             # closing axis: fixed jaw -> moving jaw
assert up[2] > 0.99, up                                                # moving jaw on top
if HAVE_MJ:
    tw3 = CK.TwinIO(gripper="so101")
    sk3 = CK.CupSkills(tw3)
    sk3.run_tool("observe_scene", {})
    r3 = sk3.run_tool("plan_cup_stack", {"source_cup": "cup_1", "target_cup": "cup_2"})
    assert r3["ok"], r3
r = subprocess.run([sys.executable, "-c", "import NERO_V364_SO101_Gripper as G; print(G.MOUNT['roll_deg'])"],
                   cwd=str(ROOT), env=dict(os.environ, NERO_V364_SO101_ROLL_DEG="0"), capture_output=True, text=True)
assert r.stdout.strip() == "0.0", r.stdout + r.stderr
print("[I3] roll -90: tool horizontal + J7 = 0 -> moving jaw on top (user's choice A); cup stack plan reachable; env override ok")

# ------------------------------------------------------------------ I4
logs = []
g0, l0 = RT.migrate_geometry_v375({"roll_deg": -45.0, "tip_corr_mm": [1.0, 0.0, 0.0], "ring_mm": 7.0}, log=logs.append)
assert g0["roll_deg"] == -90.0 and g0["tip_corr_mm"] == [0.0, 0.0, 0.0] and g0["mount_rev"] == "V3.76" and "pre_v375" in g0
g1, l1 = RT.migrate_geometry_v375({"roll_deg": 0.0, "mount_rev": "V3.75", "tip_corr_mm": [0.0, 2.0, 0.0]})
assert g1["roll_deg"] == -90.0 and g1["tip_corr_mm"] == [0.0, 0.0, 0.0] and "pre_v376" in g1 and len(l1) == 2
assert RT.migrate_geometry_v375(g1)[1] == [] and logs and "安装方向" in logs[0]
assert RT.real_mount({})[2]["roll_deg"] == -90.0
print("[I4] real settings: -45 (before V3.75) -> -90, 0 (V3.75) -> -90, V3.76 untouched; fingertip correction reset")

# ------------------------------------------------------------------ I5
assert 'solref="0.003 1"' in G.gripper_parts()["link7"] and 'solref="0.006 1"' in G.gripper_parts(soft_pads=True)["link7"]
if HAVE_MJ:
    tw_s = S.SortTwinIO()
    tw_c = CK.TwinIO(gripper="so101")
    sr = lambda tw: float(tw.m.geom_solref[tw.m.geom("finger1_rubber_pad").id][0])
    assert abs(sr(tw_s) - 0.006) < 1e-12 and abs(sr(tw_c) - 0.003) < 1e-12
print("[I5] soft pads (solref 0.006) in the sorting scene / twin only; cup gripper keeps 0.003")

if HAVE_MJ:
    # -------------------------------------------------------------- I6
    def put_item(tw, body, xyz):
        j = tw.m.body(body).jntadr[0]
        a = tw.m.jnt_qposadr[j]
        tw.d.qpos[a:a + 3] = xyz
        tw.d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        tw.d.qvel[tw.m.jnt_dofadr[j]:tw.m.jnt_dofadr[j] + 6] = 0
        mujoco.mj_forward(tw.m, tw.d)

    def dropping_run(target_xy):
        tw = S.SortTwinIO()
        sk = S.SortSkills(tw)
        assert sk.run_tool("pick_object", {"item_id": "item_1"})["ok"]
        orig = sk._go_line

        def go_line(label, *a, **kw):
            if label.startswith("carry"):                        # the backend let go during the carry
                tw._release_hold()
                tw.gripper_open(80.0)
                put_item(tw, "obj_banana", [target_xy[0], target_xy[1], 0.04])
                tw.step(400)
                raise sk._Abort("SAFETY_STOP", {"during": label, "error": "runaway (test)"})
            return orig(label, *a, **kw)
        sk._go_line = go_line
        return tw, sk, sk.run_tool("place_in_box", {"box": "blue"})

    tw, sk, r6 = dropping_run(SC.SORT_BINS["blue"]["centre"])
    assert r6["ok"] and r6["reason"] == "ITEM_DROPPED_IN_BOX" and r6["warnings"], r6
    assert sk.held is None and sk.placed.get("item_1") == "blue" and r6["physical_verification"]["dropped_by_safety_stop"]
    assert sk.run_tool("return_to_safe_pose", {})["ok"]
    tw, sk, r6b = dropping_run((-0.30, 0.02))
    assert not r6b["ok"] and r6b["reason"] == "ITEM_DROPPED" and sk.held is None, r6b
    tw = S.SortTwinIO()
    sk = S.SortSkills(tw)
    assert sk.run_tool("pick_object", {"item_id": "item_4"})["ok"]
    tw._release_hold()
    tw.gripper_open(80.0)
    tw.step(800)
    r6c = sk.run_tool("place_in_box", {"box": "blue"})
    assert not r6c["ok"] and r6c["reason"] == "NO_ITEM_HELD" and sk.run_tool("return_to_safe_pose", {})["ok"]
    print("[I6] safety stop let go in the blue bin -> ITEM_DROPPED_IN_BOX (ok + warning), elsewhere -> ITEM_DROPPED; "
          "released item noticed; return_to_safe_pose works")

# ------------------------------------------------------------------ I7
hist = [{"tool": "pick_object", "args": {"item_id": "item_1"}, "result": {"ok": True, "tool": "pick_object",
         "physical_verification": {"item": "item_1"}}},
        {"tool": "place_in_box", "args": {"box": "blue"}, "result": {"ok": False, "tool": "place_in_box",
         "reason": "SAFETY_STOP", "physical_verification": {"item": "item_1", "box": "blue"}}},
        {"tool": "check_bins", "args": {}, "result": {"ok": True, "tool": "check_bins", "physical_verification": {
            "items": {"item_1": {"in_bin": "blue"}}}}}]
sc = S._score(hist, 60.0, 5, {"code": "SI"}, "banana -> blue")
assert sc["s"] == 1.0 and sc["status"] == "SUCCESS", sc
hist[2]["result"]["physical_verification"]["items"]["item_1"]["in_bin"] = None
assert S._score(hist, 60.0, 5, {"code": "SI"}, "t")["s"] == 0.0
print("[I7] score: the banana that fell into the requested bin counts (s = 1); on the table it does not")

if HAVE_MJ:
    # -------------------------------------------------------------- I8
    tw = S.SortTwinIO()
    sk = S.SortSkills(tw)
    assert sk.v376_clear_start()["moved"] is False
    rl = CK.ik_chain([("low", np.array([-0.37, 0.24, 0.13]), S.grasp_R(0.0) @ G.T_tcp_grasp()[:3, :3].T)])
    assert rl["ok"], rl                                                 # like the V3.75 run: low next to the blue bin
    tw.move_joints("test: low next to the blue bin", rl["q"][-1])
    assert tw.tcp_pose()[0][2] < 0.2
    r8 = sk.v376_clear_start()
    assert r8["moved"] and r8["home"] and tw.tcp_pose()[0][2] >= 0.29, (r8, tw.tcp_pose())
    print(f"[I8] start clear: low TCP -> straight up + home (TCP z {tw.tcp_pose()[0][2]:.2f} m); high TCP -> no motion")

# ------------------------------------------------------------------ I9
src = (ROOT / "start_nero_sim.sh").read_text(encoding="utf-8")
fn = re.search(r"^v376_alive\(\).*?$", src, re.M).group(0) + "\n" + re.search(r"^v376_stop\(\)\{.*?^\}", src, re.M | re.S).group(0)
cl = re.search(r"^cleanup\(\)\{.*?^\}", src, re.M | re.S).group(0)
for lbl in ('"$TRAC_MOVEIT_PID" 1', '"$BIO_MOVEIT_PID" 1', '"$BACKEND_PID" 0', '"$GPU_PID" 0'):
    assert lbl in cl, lbl
(TMP / "stopfn.sh").write_text(fn + "\n", encoding="utf-8")
script = f"""source {TMP / 'stopfn.sh'}
bash -c 'trap "" INT TERM; sleep 311 & wait' & P1=$!
setsid bash -c 'trap "" INT TERM; sleep 312 & sleep 313 & wait' & P2=$!
sleep 0.5
v376_stop one "$P1" 0 & v376_stop grp "$P2" 1 & wait
ps -eo cmd | grep -E "^sleep 31[123]" || echo NONE_LEFT
"""
t0 = time.time()
r9 = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=dict(os.environ, NERO_V376_STOP_STEP_S="1.0"),
                    timeout=60)
dt = time.time() - t0
assert "NONE_LEFT" in r9.stdout and "stopped after SIGKILL" in r9.stdout and dt < 6.0, (r9.stdout, r9.stderr, dt)
csrc = (ROOT / "NERO_V312_Console.py").read_text(encoding="utf-8")
assert "stop_countdown()" in csrc and "最多再等" in csrc
print(f"[I9] timed stop: INT/TERM-ignoring process + group + children gone in {dt:.1f} s (step 1 s); cleanup uses it "
      "for all 4; console countdown")

if HAVE_MJ:
    # -------------------------------------------------------------- I10
    tw = S.SortTwinIO()
    sk = S.SortSkills(tw)
    out = sk.scripted_episode()
    st = {iid: it["in_bin"] for iid, it in sk._items().items()}
    assert st == {"item_1": "red", "item_2": "red", "item_3": "blue", "item_4": "blue"}, st
    import v375_moveit_twin as MT  # noqa: E402
    r10 = MT.run_backend_episode()
    assert r10["state"] == {"item_1": "red", "item_2": "red", "item_3": "blue", "item_4": "blue"}, r10["state"]
    assert any("roll -90 deg" in l for l in r10["log"]) and r10["scores"][0]["s"] == 1.0
    carry = [mv for mv in r10["core"].moves if mv[0] == "line"]          # the carries (MoveIt straight lines)
    assert carry, r10["core"].moves[:5]
    assert "slow=HELD_SLOW, free_fallback=True" in (ROOT / "NERO_V372_Sort_Skills.py").read_text(encoding="utf-8")
    print(f"[I10] -90 mount, soft pads: twin 4/4, backend path with MoveIt emulation 4/4 (carry moves {len(carry)}, "
          f"stretched {S.HELD_SLOW}x)")
else:
    print("[I5-I10 MuJoCo parts] skipped: MuJoCo not installed")

SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("[I11] release folder unchanged")
print("ALL V3.76 TESTS PASSED")
