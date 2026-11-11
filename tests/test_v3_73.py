#!/usr/bin/env python3
"""V3.73 offline tests (sort_items fixes after the first simulation run):
E1 Versions V3.73 + markers; frozen modules unchanged; the backend is not changed by V3.73.
E2 TF name: the backend's object_pose(name) looks up "sim_obj_" + name (NERO_V71 skill core) and the scene publishes
   "sim_" + body; tf_name("obj_banana") = "banana"; the test core now follows exactly that rule and raises the
   backend's TF error for an unknown frame.  The V3.72 call ("obj_banana" -> "sim_obj_obj_banana") reproduces the user's
   failure; the V3.73 call returns the banana pose.
E3 Full backend path with the strict TF rule (gui_run_task -> SortBackendIO -> agent loop, scripted model): observe_scene
   works, 4/4 sorted, no "[V3.64 CUP] ... release gap capped" line in the sorting log.
E4 Score: s = items the agent moved that are in the bin it chose / items it moved (only-fruit task -> 2/2 = 1.0);
   nothing moved -> 0; an item placed but gone from its bin at check_bins does not count.
E5 Release folder unchanged.
"""
import hashlib
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v373_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
os.environ["NERO_V368_TASK_DIR"] = str(TMP / "task")
for k in ("NERO_SCENE", "NERO_V343_ACTIVE_SCENE", "NERO_V372_ITEM_SEED"):
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
import NERO_V312_Console as CON             # noqa: E402
import NERO_V313_Real_Robot as RR           # noqa: E402
import NERO_V343_Scenes as SC               # noqa: E402
import NERO_V347_Yolo_Perception as YP      # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49   # noqa: E402
import NERO_V372_Sort_Skills as S           # noqa: E402

try:
    import mujoco  # noqa: F401
    HAVE_MJ = True
except Exception:
    HAVE_MJ = False

# ------------------------------------------------------------------ E1
for m in (CON, RR, YP, W49, S):
    assert m.VERSION in ("V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3"), (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
for fn, h in (("NERO_V2_7_1_Unified_Backend.py", "5be2136f84ec4ccdfe6f8bc8d8998410ff9b3b827af4d571ea44373dc427527e"),
              ("NERO_V2_7_1_Agent.py", "3e52d05ea84d94c8359d5875fa0fee8c2cc42d612b52e59109e1166912439696"),
              ("NERO_V325_Camera.py", "82f2416d73671e7018d09d0cd72c17186154934a315b47c0d19fdb0427afd931"),
              ("NERO_V326_Wrist_Camera.py", "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a")):
    assert H((ROOT / fn).read_bytes()) == h, f"{fn} changed"
src = (ROOT / "NERO_V372_Sort_Skills.py").read_text(encoding="utf-8")
for mk in ("V3.73 TF NAME", "V3.73 SCORE BY THE AGENT'S ITEMS"):
    assert mk in src, mk
print("[E1] V3.73; frozen modules and the backend unchanged")

# ------------------------------------------------------------------ E2
src71 = (ROOT / "NERO_V71_Hybrid_GPU_Planner_Core.py").read_text(encoding="utf-8")
assert 'self.wait_for_tf(f"sim_obj_{object_name}")' in src71          # the backend rule the fix follows
assert S.tf_name("obj_banana") == "banana" and S.tf_name("obj_mouse") == "mouse" and S.tf_name("banana") == "banana"
if HAVE_MJ:
    import v372_backend_twin as SB  # noqa: E402
    sh = SB.HB.make_shared()
    os.environ["NERO_SCENE"], os.environ["NERO_V343_ACTIVE_SCENE"] = "sort_items", "sort_items"
    try:
        tw = SB.SyncedSortTwin(sh)
        core = SB.SortCore(tw, [], sh)
        try:
            core.object_pose("obj_banana")                           # what V3.72 passed
            raise AssertionError("the V3.72 name must fail like the backend")
        except RuntimeError as exc:
            assert "sim_obj_obj_banana" in str(exc) and "does not exist" in str(exc)
        io = S.SortBackendIO({"skill_core": core, "shared": sh})
        p, q = io.object_pose("obj_banana")
        assert abs(p[0] - SC.SORT_ITEMS["obj_banana"]["xy"][0]) < 0.003 and abs(p[2] - 0.0345) < 0.003, p
    finally:
        os.environ.pop("NERO_SCENE"), os.environ.pop("NERO_V343_ACTIVE_SCENE")
    print(f"[E2] V3.72 name -> 'sim_obj_obj_banana' does not exist (reproduced); V3.73 name -> banana at "
          f"({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:.3f})")

    # -------------------------------------------------------------- E3
    r = SB.run_backend_episode()
    assert r["started"].get("label") == "RUN TASK (sort_items)" and not r["started"].get("error"), r["started"]
    assert r["state"] == {"item_1": "red", "item_2": "red", "item_3": "blue", "item_4": "blue"}, r["state"]
    assert not any("TF unavailable" in l for l in r["log"]), [l for l in r["log"] if "TF" in l][:3]
    assert not any("[V3.64 CUP]" in l and "release gap capped" in l for l in r["log"])
    assert r["scores"] and r["scores"][0]["s"] == 1.0 and r["scores"][0]["total"] == 4
    print("[E3] backend path with the strict TF rule: 4/4 sorted, s=1, no cup message in the log")
else:
    print("[E2-E3] skipped: MuJoCo not installed (tf_name checked)")

# ------------------------------------------------------------------ E4


def res(tool, ok=True, **pv):
    return {"ok": ok, "tool": tool, "physical_verification": pv}


skill = {"code": "SI", "short_name": "Sort Items"}
hist = [{"tool": "observe_scene", "args": {}, "result": res("observe_scene")},
        {"tool": "pick_object", "args": {"item_id": "item_1"}, "result": res("pick_object", item="item_1")},
        {"tool": "place_in_box", "args": {"box": "blue"}, "result": res("place_in_box", item="item_1", box="blue")},
        {"tool": "pick_object", "args": {"item_id": "item_2"}, "result": res("pick_object", item="item_2")},
        {"tool": "place_in_box", "args": {"box": "blue"}, "result": res("place_in_box", item="item_2", box="blue")},
        {"tool": "check_bins", "args": {}, "result": res("check_bins", items={
            "item_1": {"in_bin": "blue"}, "item_2": {"in_bin": "blue"}, "item_3": {"in_bin": None},
            "item_4": {"in_bin": None}})}]
sc = S._score(hist, 60.0, 6, skill, "Pick up fruit and put them in the blue container.")
assert sc["s"] == 1.0 and sc["total"] == 2 and sc["status"] == "SUCCESS" and "2/2" in sc["line"], sc
assert S._score(hist[:1], 10.0, 1, skill, "t")["s"] == 0.0 and S._score(hist[:1], 10.0, 1, skill, "t")["status"] == "FAILED"
hist2 = hist[:-1] + [{"tool": "check_bins", "args": {}, "result": res("check_bins", items={
    "item_1": {"in_bin": "blue"}, "item_2": {"in_bin": None}})}]
sc2 = S._score(hist2, 60.0, 6, skill, "t")
assert sc2["s"] == 0.5 and sc2["status"] == "PARTIAL", sc2
failed_pick = [{"tool": "pick_object", "args": {"item_id": "item_3"}, "result": {"ok": False, "tool": "pick_object",
                                                                                  "reason": "GRASP_UNREACHABLE"}}]
assert S._score(failed_pick, 10.0, 1, skill, "t")["total"] == 1                # a tried item counts even if it failed
print("[E4] score: only-fruit task 2/2 = 1.0; nothing moved = 0; an item gone from its bin = 0.5; a failed pick counts")

SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("[E5] release folder unchanged")
print("ALL V3.73 TESTS PASSED")
