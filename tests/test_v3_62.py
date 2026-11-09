#!/usr/bin/env python3
"""V3.62 offline tests (stack_cups Agent skills: RUN TASK in the simulation can stack the cups):
P1  Versions V3.62 + markers; NERO_V308 / NERO_V326 frozen; Agent + Camera tab unchanged since V3.60; the backend differs
    from V3.60 ONLY by the stack_cups hook inside the existing RUN TASK `# V3.43 A2` block (removing it gives the V3.60
    file byte-for-byte; test_v3_44 C1/C2/C3 still hold); the egg path of that block is unchanged.
P2  Scene: cup body origin = rim centre (CONNECT hold anchor above the centre of mass), stiff rim contact, mass-independent
    cup contact; cup_base_point / cup_task_state(origin="rim"); store_in_safe XML still byte-identical (ad6dfbcb...).
P3  Geometry: pinch frame orthonormal, approach leans inward by the wall angle (both flips), the pad clamp centre lies on
    the wall mid-surface `depth` below the rim.
P4  Planner (strict 6-DoF URDF IK): cup_1->cup_2 and cup_2->cup_1 reachable with every joint >= 8 deg from its limits;
    grasp and place use the SAME flip so the carried cup's axis lands on the target axis (< 1 mm) - the V3.61 planner
    mixed flips (2 x rim radius off).
P5  Skill knobs: group 6 = 11 knobs read by the sim skills ([仿真技能✓]), 56 knobs; CS defaults = twin study
    (config/cup_stack_study_v362.json: 16/16 default-like twin runs fully nested; the vertical approach is refused at
    planning, joint margin < 4 deg);
    shipped JSON == definitions; a non-cup active skill (EB) falls back to the built-in CS for cup tasks.
P6  Skill contract preconditions (twin): pick without plan, place without a cup, verify while not stacked, bad names,
    unknown tool -> refused with a reason and next actions, nothing moves.
P7  End to end through the backend entry gui_run_task (BackendIO unchanged, fake ROS core on the MuJoCo twin, backend
    shared values, simulation clock, scripted model on the real NeroLLMAgent transport): cup_1 ends fully nested in
    cup_2, 5 tools / 6 LLM rounds, [V3.62 SKILL SCORE] s=1, gentle carry / place speeds, posture check called.
P8  Failure paths: 0.2 mm centring tolerance -> place_cup_into refuses (NOT_CENTRED_OVER_TARGET, cup still held), the
    model stops, s=0, RED; a plan with a joint < 4 deg from its limit is refused (JOINT_MARGIN_TOO_SMALL).
P9  Mocked console (sim + real) builds; nothing inside the release folder is written.
"""
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
TMP = Path(tempfile.mkdtemp(prefix="nero_v362_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
for k in ("NERO_SCENE", "NERO_V343_ACTIVE_SCENE", "NERO_V362_CUP_SOLREF"):
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

try:
    import mujoco  # noqa: F401
    HAVE_MJ = True
except Exception:
    HAVE_MJ = False

# ------------------------------------------------------------------ P1
for m in (CON, RR, YP, W49, ST, CK):
    assert m.VERSION in ("V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9"), (m.__name__, m.VERSION)
V363 = CK.VERSION >= "V3.63"           # V3.63 intentionally changed the place motions / plan waypoints (CHANGELOG)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9") and SC.SCENES_REV in ("V3.62", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
assert H((ROOT / "NERO_V326_Wrist_Camera.py").read_bytes()) == "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a"
assert H((ROOT / "NERO_V2_7_1_Agent.py").read_bytes()) == "3e52d05ea84d94c8359d5875fa0fee8c2cc42d612b52e59109e1166912439696"
assert H((ROOT / "NERO_V325_Camera.py").read_bytes()) == "82f2416d73671e7018d09d0cd72c17186154934a315b47c0d19fdb0427afd931"
w49 = re.sub(r'VERSION = "V3\.\d+"', 'VERSION = "V3.58"', (ROOT / "NERO_V349_Real_Wrist_Camera.py").read_text(encoding="utf-8"), count=1)
assert H(w49.encode()) == "d5955ce5c9c8acf8d654faa163da49e1b54ad21775a5ec060ac353a13c76a622"
BK = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text(encoding="utf-8")
a = BK.index('        if _v343_scene_name() == "stack_cups":   # V3.62 STACK CUPS AGENT')
b = BK.index('        if _v343_scene_name() != "fill_egg_holder":\n            append_gui_log(f"[V3.43 SCENE] RUN TASK blocked')
hook = BK[a:b]
assert H((BK[:a] + BK[b:]).encode()) == "ffbd630d70edf57a49929363134ceaae3b95fe4675e0887509d3dc5d7b08f1ed", "backend: more than the hook changed"
blk = BK[BK.rindex("# V3.43 A2 BEGIN", 0, a):BK.index("# V3.43 A2 END", a)]
assert hook in blk and BK.count("# V3.43 A2 BEGIN") == BK.count("# V3.43 A2 END") == 11
assert "gui_run_task(" in hook and hook.rstrip().endswith("return") and "NERO_V362_Cup_Skills" in hook
for fn, mk in (("NERO_V362_Cup_Skills.py", CK.MARKER), ("NERO_V343_Scenes.py", "V3.62 RIM-CENTRE CUP ORIGIN"),
               ("NERO_V343_Scenes.py", "V3.62 MASS-INDEPENDENT CUP CONTACT"), ("NERO_V343_Scenes.py", "V3.62 STIFF RIM CONTACT"),
               ("NERO_V359_Strategy.py", "V3.62 CUP KNOBS READ BY THE SIM CUP SKILLS"),
               ("NERO_V2_7_1_Unified_Backend.py", "V3.62 STACK CUPS AGENT")):
    assert mk in (ROOT / fn).read_text(encoding="utf-8"), (fn, mk)
print("[P1] V3.62; V308/V326/Agent/Camera frozen; backend = V3.60 + the stack_cups hook in the RUN TASK A2 block only")

# ------------------------------------------------------------------ P2
g = SC.cup_geometry()
assert SC.CUP_ORIGIN == "rim"
x_rim = SC.cup_body_xml("obj_cup_1", (-0.35, -0.13), origin="rim")
x_base = SC.cup_body_xml("obj_cup_1", (-0.35, -0.13))
assert f'pos="-0.35000 -0.13000 {0.0005 + g["h"]:.5f}"' in x_rim and 'pos="-0.35000 -0.13000 0.00050"' in x_base
assert 'solref="-20000 -60"' in x_rim and 'solref="0.004 1"' in x_base and x_rim.count('margin="0.0005"') == 48
assert x_base.count('margin="0.0005"') == 24 and 'rim_00" type="capsule"' in x_rim
bp = SC.cup_base_point([-0.35, 0.13, 0.1215], [1, 0, 0, 0], "rim")
assert abs(bp[2] - (0.1215 - g["h"])) < 1e-9
st0 = SC.cup_task_state([-0.35, 0.13, 0.1215 + 0.0253], [1, 0, 0, 0], [-0.35, 0.13, 0.1215], [1, 0, 0, 0], origin="rim")
assert st0["stacked"] and st0["nested_full"] and abs(st0["base_dz_mm"] - 25.3) < 0.05


def build(scene):
    os.environ["NERO_SCENE"] = os.environ["NERO_V343_ACTIVE_SCENE"] = scene
    try:
        td = Path(tempfile.mkdtemp(dir=TMP))
        pth = td / "nero_motor.xml"
        pth.write_text("<mujoco model='t'><asset></asset><worldbody><body name='link7'/></worldbody></mujoco>")
        return Path(SC.make_store_in_safe_xml(str(pth))).read_text().replace(str(td.resolve()), "<TMP>")
    finally:
        os.environ.pop("NERO_SCENE")
        os.environ.pop("NERO_V343_ACTIVE_SCENE")


assert H(build("store_in_safe").encode()) == "ad6dfbcb83f83f6ba9bb3882763414efc7cd2e6525b4980e461aef55b90f2476"
print("[P2] cup origin = rim centre, stiff rim + mass-independent cup contact; store_in_safe XML unchanged")

# ------------------------------------------------------------------ P3
beta = math.atan2(g["r_top"] - g["r_bot"], g["h"])
rim = np.array([-0.35, -0.13, 0.0005 + g["h"]])
for flip in (False, True):
    for phi in (0.3, 1.7, -2.2):
        R = CK.pinch_frame(phi, flip)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and abs(np.linalg.det(R) - 1) < 1e-9
        u = np.array([math.cos(phi), math.sin(phi), 0.0])
        zt = R[:, 2]
        assert abs(math.degrees(math.acos(-zt[2])) - math.degrees(beta)) < 1e-6 and float(zt @ u) < 0   # leans inward
        assert abs(abs(float(R[:, 1] @ np.array([u[0], u[1], 0]))) - math.cos(beta)) < 1e-6           # closing ~radial
        p, R2 = CK.pinch_tcp(rim, phi, 0.035, flip)
        clamp = p + CK.CLAMP_CENTRE_Y_M * R2[:, 1]
        wall = rim + g["r_top"] * u + 0.035 * np.array([-math.sin(beta) * u[0], -math.sin(beta) * u[1], -math.cos(beta)])
        assert np.linalg.norm(clamp - wall) < 1e-9
print(f"[P3] pinch frame: approach tilted {math.degrees(beta):.1f} deg inward (both flips), clamp centre on the wall")

# ------------------------------------------------------------------ P4
K = dict(CK.CUP_KNOB_DEFAULTS)
q_up = [1.0, 0.0, 0.0, 0.0]
for src, dst in (("obj_cup_1", "obj_cup_2"), ("obj_cup_2", "obj_cup_1")):
    srim = [SC.CUP_XY[src][0], SC.CUP_XY[src][1], 0.0005 + g["h"]]
    drim = [SC.CUP_XY[dst][0], SC.CUP_XY[dst][1], 0.0005 + g["h"]]
    plan = CK.plan_cup_stack(K, srim, q_up, drim, q_up)
    assert plan["ok"] and plan["min_joint_margin_deg"] >= (4.0 if V363 else 8.0), plan   # V3.63: real-start planning -> test_v3_63 S4
    W = plan["_wps"]
    pg, Rg = W["grasp"]
    axis_in_tcp = Rg.T @ (np.array([srim[0], srim[1], srim[2]]) - pg)            # cup rim centre seen from the TCP
    pr, Rr = W["release"]
    pred = pr + Rr @ axis_in_tcp
    assert np.linalg.norm(pred[:2] - np.array(drim[:2])) < 0.001, (pred, drim)
    assert abs((pred[2] - g["h"]) - plan["base_release_z_m"]) < 1e-6
    assert [w["name"] for w in plan["waypoints"]] == (["pregrasp", "grasp", "lift"] + (["rotate"] if V363 else [])
                                                      + ["above_target", "release", "retreat"])
    print(f"[P4] {src}->{dst}: margins grasp {plan['grasp_margin_deg']} / place {plan['place_margin_deg']} deg, same flip, "
          f"carried axis lands {np.linalg.norm(pred[:2] - np.array(drim[:2])) * 1000:.2f} mm from the target axis")

# ------------------------------------------------------------------ P5
cup_specs = [s for s in ST.KNOB_SPECS if s["group"] == "cup_stacking"]
assert len(cup_specs) == 11 and all(s["wired"] == "sim_skill" for s in cup_specs) and len(ST.KNOB_SPECS) == 56
assert ST.WIRING_LABELS["sim_skill"] == "仿真技能✓"
assert {s["key"] for s in cup_specs} == set(CK.CUP_KNOB_DEFAULTS)
assert all(ST.BASELINE_KNOBS[k_] == v for k_, v in CK.CUP_KNOB_DEFAULTS.items())
for p in ST.builtin_strategy_presets():
    d = json.loads((ROOT / "config" / "strategies" / f"{p.skill_id}.json").read_text(encoding="utf-8"))
    e = p.to_dict()
    d.pop("saved"), e.pop("saved")
    assert d == e, p.skill_id
kn, sk = CK.cup_knobs()
assert sk["code"] == "CS" and "built-in CS" in sk["source"] and kn == CK.CUP_KNOB_DEFAULTS
study = json.loads((ROOT / "config" / "cup_stack_study_v362.json").read_text(encoding="utf-8"))
assert study["summary"] == {"default_like_runs": 16, "default_like_success": 16} and study["defaults"] == CK.CUP_KNOB_DEFAULTS
ctrl = [r for r in study["rows"] if r["knobs"].get("cup_wall_tilt_follow") is False]
assert len(ctrl) == 2 and all(not r["stacked"] and r["failed_at"] == "plan_cup_stack:JOINT_MARGIN_TOO_SMALL" for r in ctrl)
print("[P5] 11 cup knobs [仿真技能✓] (56 total), CS defaults = twin study (16/16), vertical approach refused at planning, EB -> CS")

# ------------------------------------------------------------------ P6
if HAVE_MJ:
    io = CK.TwinIO()
    sk6 = CK.CupSkills(io, dict(CK.CUP_KNOB_DEFAULTS), {"code": "CS"})
    tcp0, _ = io.tcp_pose()
    r = sk6.run_tool("pick_cup", {"source_cup": "cup_1"})
    assert not r["ok"] and r["reason"] == "PLAN_REQUIRED" and r["recommended_next_actions"] == ["plan_cup_stack"]
    assert sk6.run_tool("place_cup_into", {"target_cup": "cup_2"})["reason"] == "NO_CUP_HELD"
    r = sk6.run_tool("verify_stack", {"upper_cup": "cup_1", "lower_cup": "cup_2"})
    assert not r["ok"] and r["reason"] == "NOT_STACKED"
    assert sk6.run_tool("plan_cup_stack", {"source_cup": "cup_1", "target_cup": "cup_1"})["reason"] == "BAD_CUP_NAMES"
    assert sk6.run_tool("dance", {})["reason"] == "UNKNOWN_TOOL"
    obs = sk6.run_tool("observe_scene", {})
    assert obs["ok"] and obs["scene"]["stacked"] is None and set(obs["scene"]["objects"]) == {"cup_1", "cup_2"}
    assert all(o["upright"] and o["on_table"] for o in obs["scene"]["objects"].values())
    assert np.linalg.norm(np.array(io.tcp_pose()[0]) - np.array(tcp0)) < 1e-6
    assert [t_["name"] for t_ in CK.CUP_TOOLS] == ["observe_scene", "plan_cup_stack", "pick_cup", "place_cup_into",
                                                   "verify_stack", "return_to_safe_pose"]
    print("[P6] contract preconditions refused with reasons; observe_scene: 2 upright cups, not stacked; no motion")

    # -------------------------------------------------------------- P7
    import v362_backend_twin as HB
    res = HB.run_backend_episode()
    st = res["state"]
    assert st["stacked"] and st["nested_full"] and st["tilt_cup1_deg"] < 3 and st["tilt_cup2_deg"] < 3, st
    calls = [l for l in res["log"] if l.startswith("[LLM CALL #")]
    assert [c.split("] ")[1].split("(")[0] for c in calls] == ["observe_scene", "plan_cup_stack", "pick_cup",
                                                               "place_cup_into", "verify_stack"], calls
    sc = res["scores"][0]
    assert sc["s"] == 1.0 and sc["z"] == 6 and sc["skill_code"] == "CS" and sc["Y"] > 0
    assert any(l.startswith("[V3.62 SKILL SCORE] skill=CS") for l in res["log"])
    assert any(l.startswith("[TASK SUMMARY GREEN]") for l in res["log"]) and res["calls"]["start_check"] == 1
    if not V363:
        mv = {lab: (kind, spd) for kind, lab, spd in res["core"].moves}
        assert mv["V3.62 carry cup_1 over cup_2"] == ("line", 0.016) and mv["V3.62 lower cup_1 into cup_2"] == ("line", 0.004)
    else:                                # V3.63: joint-space rotate, checked carry / lowering (test_v3_63 covers them)
        labs = [lab for _k, lab, _d in res["core"].moves]
        assert any(l.startswith("V3.63 rotate") for l in labs) and any(l.startswith("V3.63 lower") for l in labs)
    assert res["llm"].requests[0]["instructions"] == CK.CUP_SYSTEM_INSTRUCTIONS
    assert res["llm"].requests[0]["tools"] == CK.CUP_TOOLS and res["started"]["label"] == "RUN TASK (stack_cups)"
    assert not res["shared"]["hold_active_shared"].value
    print(f"[P7] gui_run_task end to end: cup_1 fully nested in cup_2 (dz {st['base_dz_mm']} mm), 5 tools / 6 rounds, "
          f"Y={sc['Y']}, level carry 0.016 m/s, lowering 0.004 m/s")

    # -------------------------------------------------------------- P8
    res = HB.run_backend_episode(knobs={"cup_release_xy_tol_mm": 0.2})        # closed-loop centring reaches ~1-2 mm
    sc = res["scores"][0]
    fails = [l for l in res["log"] if l.startswith("[FAIL] [V3.62 CUP] place_cup_into")]
    assert fails and "NOT_CENTRED_OVER_TARGET" in fails[0] and sc["s"] == 0.0 and sc["status"] == "FAILED", res["log"][-12:]
    assert any(l.startswith("[TASK SUMMARY RED]") for l in res["log"]) and "Stopped" in res["started"]["result"]
    assert res["shared"]["hold_active_shared"].value                      # refused before opening: the cup is still held
    sk8 = CK.CupSkills(CK.TwinIO(), dict(CK.CUP_KNOB_DEFAULTS), {"code": "CS"})
    _orig_plan = CK.plan_cup_stack
    CK.plan_cup_stack = lambda *a_, **k_: dict(_orig_plan(*a_, **k_), min_joint_margin_deg=1.0, place_margin_deg=1.0)
    try:
        r8 = sk8.run_tool("plan_cup_stack", {"source_cup": "cup_1", "target_cup": "cup_2"})
    finally:
        CK.plan_cup_stack = _orig_plan
    assert not r8["ok"] and r8["reason"] == "JOINT_MARGIN_TOO_SMALL" and sk8.stage != "PLANNED", r8
    print("[P8] 0.2 mm centring tolerance: place refused (NOT_CENTRED_OVER_TARGET) with the cup still held, model stopped, "
          "s=0, RED; a plan with a joint 1 deg from its limit is refused (JOINT_MARGIN_TOO_SMALL)")
else:
    print("[P6-P8] skipped: MuJoCo not installed (pip install mujoco)")

# ------------------------------------------------------------------ P9
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
mtk.ttk = mttk
mtk.Text.return_value.yview.return_value = (0.0, 1.0)
mtk.Text.return_value.index.return_value = "1.0"
sys.modules["tkinter"] = mtk
sys.modules["tkinter.ttk"] = mttk
for sub in ("font", "messagebox", "simpledialog", "filedialog"):
    sys.modules[f"tkinter.{sub}"] = mock.MagicMock()
root = mock.MagicMock()
root.winfo_screenwidth.return_value = 1920
root.winfo_screenheight.return_value = 1080
mtk.Tk.return_value = root
root.mainloop.side_effect = lambda: None
CON.run_console("sim")
CON.run_console("real")
ui = ST.SkillStrategyTabUI(mtk, mttk, mock.MagicMock(), log=lambda *a: None, mode="sim")
assert "cup_stacking" in ui._sections and len([k_ for k_ in ui.knob_vars if k_.startswith("cup_")]) == 11
SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("[P9] console (sim + real) builds, sim skill tab has the 11 cup knobs; release folder unchanged")
print("ALL V3.62 TESTS PASSED")
