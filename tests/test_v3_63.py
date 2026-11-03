#!/usr/bin/env python3
"""V3.63 offline tests (stack_cups: joint-space safe motion after the V3.62 J7 limit trip + fixed D435i cup perception):
S1  Versions V3.63 + markers; NERO_V308 / NERO_V326 / Agent / Camera tab frozen; the backend still differs from V3.60
    only inside the stack_cups hook of the RUN TASK A2 block (now also passing capture / latch hooks).
S2  FastArm (numpy FK + analytic Jacobian) == NERO_V315 tool_pose; Jacobian == numeric; IK round trip.
S3  check_joint_path refuses a point < 4 deg from a limit and a > 5 deg jump between points (IK branch flip).
S4  Planner from the REAL start posture (home, all joints 0): both directions plan the whole chain (pregrasp ... retreat)
    with every held pose >= 8 deg from the limits and J7 >= 8 deg while placing (V3.62 single-branch planning from home
    left J7 ~1.6 deg from its limit); the plan stores the joint solution of every waypoint.
S5  Twin episode (URDF arm kinematics): a Cartesian path that breaks (IK continuity at a joint limit) is refused and the
    move is done in joint space to a margin-checked IK solution; cup_1 ends fully nested.
S6  gui_run_task end to end with an injected safety trip during the carry: SAFETY_STOP -> STOP + RECOVER (hold kept) ->
    observe_scene -> place_cup_into again -> stacked, s=1; a second trip of the same tool -> stop_and_report, s=0.
S7  Fixed D435i cup perception on rendered frames (tests/v363_cam_frames.npz): two upright cups, axis error <= 3 mm, rim
    height error <= 4 mm; after stacking one 'stack' cluster at the lower cup, nested height ~25 mm.
S8  The fixed-camera process (NERO_V331, fake MuJoCo returning those frames) in the stack_cups scene answers with
    "cups" and does not run the egg perception; the egg scene path is unchanged (test_v3_31 still passes).
S9  gui_run_task pose-source rules: wrist camera refused, fixed camera not running refused (no truth fallback); with an
    OpenGL context: the full task with the fixed camera as pose source (twin rendering) stacks the cups.
S10 Skill CS card mentions the V3.63 behaviour; shipped JSON == definitions; 56 knobs; mocked console builds; nothing in
    the release folder is written.
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
TMP = Path(tempfile.mkdtemp(prefix="nero_v363_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
for k in ("NERO_SCENE", "NERO_V343_ACTIVE_SCENE", "NERO_V362_CUP_SOLREF", "NERO_V308_DEPTH_NOISE_K"):
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
import NERO_V363_Cup_Perception as CP       # noqa: E402
import NERO_V315_Consistency_Safety as CS   # noqa: E402

try:
    import mujoco  # noqa: F401
    HAVE_MJ = True
except Exception:
    HAVE_MJ = False

# ------------------------------------------------------------------ S1
for m in (CON, RR, YP, W49, ST, CK, CP):
    assert m.VERSION in ("V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72"), (m.__name__, m.VERSION)
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
assert H((ROOT / "NERO_V326_Wrist_Camera.py").read_bytes()) == "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a"
assert H((ROOT / "NERO_V2_7_1_Agent.py").read_bytes()) == "3e52d05ea84d94c8359d5875fa0fee8c2cc42d612b52e59109e1166912439696"
assert H((ROOT / "NERO_V325_Camera.py").read_bytes()) == "a58e69b363a5a861b3ad0806ba276c49601fc214ffdfc07ffba9bdc0c2c65797"
BK = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text(encoding="utf-8")
a = BK.index('        if _v343_scene_name() == "stack_cups":   # V3.62 STACK CUPS AGENT')
b = BK.index('        if _v343_scene_name() != "fill_egg_holder":\n            append_gui_log(f"[V3.43 SCENE] RUN TASK blocked')
hook = BK[a:b]
assert H((BK[:a] + BK[b:]).encode()) == "ffbd630d70edf57a49929363134ceaae3b95fe4675e0887509d3dc5d7b08f1ed"
for key in ('"capture": v311_capture_sync', '"clear_latch": _v337_clear_latch', '"latch_pending": _v337_latch_pending',
            '"wrist_selected": v326_wrist_selected', '"camera_available": v311_camera_available'):
    assert key in hook, key
assert BK.count("# V3.43 A2 BEGIN") == BK.count("# V3.43 A2 END") == 11
for fn, mk in (("NERO_V362_Cup_Skills.py", CK.MARKER_V363), ("NERO_V363_Cup_Perception.py", CP.MARKER),
               ("NERO_V331_Fixed_Camera.py", "V3.63 FIXED D435i CUP PERCEPTION"),
               ("NERO_V2_7_1_Unified_Backend.py", "V3.63 FIXED D435i CUP PERCEPTION + SAFETY LATCH RECOVERY"),
               ("NERO_V362_Cup_Skills.py", "V3.63 JOINT-SPACE SAFE MOTION"), ("start_nero_sim.sh", "V3.63 FIXED D435i CUP PERCEPTION")):
    assert mk in (ROOT / fn).read_text(encoding="utf-8"), (fn, mk)
print("[S1] V3.63; frozen modules unchanged; backend = V3.60 + the stack_cups hook (capture / latch hooks added)")

# ------------------------------------------------------------------ S2
arm = CK.FastArm()
U = CS.load_sim_urdf()
T_tool = CS.tool_transform_from_sim(U)
J = [f"joint{i}" for i in range(1, 8)]
for q in ([0.3, 0.2, -0.4, 1.5, 0.4, 0.6, 1.1], [0.0] * 7, [-1.0, 0.8, 0.5, 0.2, -1.2, -0.5, -1.3]):
    assert np.abs(arm.fk(q) - np.array(CS.tool_pose(U, J, q, "base_link", T_tool))).max() < 1e-9
    Tq, Jm = arm.fk(q, True)
    for k in range(7):
        q2 = list(q)
        q2[k] += 1e-6
        assert np.abs((arm.fk(q2)[:3, 3] - Tq[:3, 3]) / 1e-6 - Jm[:3, k]).max() < 1e-5
qr, pe, re_ = arm.ik([[0.0] * 7], Tq[:3, 3], Tq[:3, :3])
assert qr is not None and pe < 5e-4 and re_ < 1.0
print("[S2] FastArm FK == NERO_V315 tool_pose, analytic Jacobian == numeric, IK round trip")

# ------------------------------------------------------------------ S3
ok_path = [[0.0] * 7, [0.02] * 7, [0.04] * 7]
assert CK.check_joint_path(ok_path)["ok"]
near = [[0.0] * 6 + [math.radians(88.0)]]
r = CK.check_joint_path(near)
assert not r["ok"] and r["binding"] == "J7" and "from its limit" in r["why"]
flip = [[0.0] * 7, [0.0] * 6 + [math.radians(20.0)]]
r = CK.check_joint_path(flip)
assert not r["ok"] and "jumps" in r["why"] and r["max_jump_deg"] == 20.0
print("[S3] path check: J7 2 deg from its limit refused, 20 deg jump between two points refused")

# ------------------------------------------------------------------ S4
g = SC.cup_geometry()
up = [1.0, 0.0, 0.0, 0.0]
K = dict(CK.CUP_KNOB_DEFAULTS)
for src, dst in (("obj_cup_1", "obj_cup_2"), ("obj_cup_2", "obj_cup_1")):
    srim = [SC.CUP_XY[src][0], SC.CUP_XY[src][1], 0.0005 + g["h"]]
    drim = [SC.CUP_XY[dst][0], SC.CUP_XY[dst][1], 0.0005 + g["h"]]
    plan = CK.plan_cup_stack(K, srim, up, drim, up, q_start=[0.0] * 7)
    assert plan["ok"], plan
    Q = plan["_q"]
    assert list(Q) == ["pregrasp", "grasp", "lift", "rotate", "above_target", "release", "retreat"]
    held = [Q[n] for n in ("grasp", "lift", "rotate", "above_target", "release")]
    m_held = min(min(CK.joint_margins_deg(q)) for q in held)
    j7 = min(CK.joint_margins_deg(q)[6] for q in held[2:])
    assert m_held >= 8.0 and j7 >= 8.0 and min(CK.joint_margins_deg(Q["retreat"])) >= 4.0, (src, m_held, j7)
    for n, q in Q.items():                           # every stored joint solution reaches its waypoint
        p, R = plan["_wps"][n]
        assert np.linalg.norm(arm.fk(q)[:3, 3] - p) < 1e-3, n
    print(f"[S4] {src}->{dst} from home: held poses >= {m_held:.1f} deg from the limits, J7 >= {j7:.1f} deg while "
          f"placing, joint solution stored for all 7 waypoints")

if HAVE_MJ:
    # -------------------------------------------------------------- S5
    res, st, io = CK.twin_episode(log=lambda s_: None)
    kinds = dict(io.skills.motions)
    assert st["nested_full"] and kinds.get("pregrasp cup_1") == "joints"
    refused = [lab for lab, k in io.skills.motions if "refused" in k]
    assert refused and all(lab.startswith("carry") for lab in refused), io.skills.motions
    print(f"[S5] twin: the Cartesian carry was refused (IK continuity broke at a joint limit) -> joint space; nested "
          f"(dz {st['base_dz_mm']} mm); motions {[k for _l, k in io.skills.motions]}")

    # -------------------------------------------------------------- S6
    import v362_backend_twin as HB
    r6 = HB.run_backend_episode(trip_on="carry")
    calls = [l.split("] ")[1].split("(")[0] for l in r6["log"] if l.startswith("[LLM CALL #")]
    assert calls == ["observe_scene", "plan_cup_stack", "pick_cup", "place_cup_into", "observe_scene",
                     "place_cup_into", "verify_stack"], calls
    stop = [l for l in r6["log"] if "[FAIL] [V3.62 CUP] place_cup_into: SAFETY_STOP" in l]
    assert stop and '"latch_recovered": true' in stop[0] and '"hold_active": true' in stop[0]
    assert r6["calls"]["clear_latch"] == 1 and r6["state"]["nested_full"] and r6["scores"][0]["s"] == 1.0
    r6b = HB.run_backend_episode(trip_on="carry", trips=2)
    last = [l for l in r6b["log"] if l.startswith("[LLM TOOL RESULT #")][-1]
    assert '"reason": "SAFETY_STOP"' in last and '"recommended_next_actions": ["stop_and_report"]' in last
    assert r6b["scores"][0]["s"] == 0.0 and r6b["shared"]["hold_active_shared"].value
    print("[S6] safety trip during the carry: SAFETY_STOP -> STOP+RECOVER (hold kept) -> retry -> nested (s=1); "
          "a second trip of the same tool -> stop_and_report (s=0)")
else:
    print("[S5-S6] skipped: MuJoCo not installed")

# ------------------------------------------------------------------ S7
import NERO_V308_Camera_Perception as P8  # noqa: E402
fr = np.load(ROOT / "tests" / "v363_cam_frames.npz")
truth = json.loads(str(fr["truth"]))
cfg = P8.camera_config()
Kc, Tc = P8.intrinsics(cfg), P8.extrinsics(cfg)
r7 = CP.perceive_cups(fr["rgb_two"], fr["depth_two_mm"].astype(np.float32) / 1000.0, Kc, Tc)
assert [c["kind"] for c in r7["cups"]] == ["cup", "cup"] and abs(r7["table_z_m"]) < 0.002
for c, name in zip(r7["cups"], ("cup_1", "cup_2")):              # sorted by y: cup_1 (y < 0) first
    t = truth["two"][name]
    exy = math.dist(c["rim_xyz_m"][:2], t[:2]) * 1000
    ez = abs(c["rim_xyz_m"][2] - t[2]) * 1000
    assert exy <= 3.0 and ez <= 4.0 and c["tilt_deg"] < 2.0 and abs(c["height_mm"] - 124.0) < 5.0, (name, c, t)
r7s = CP.perceive_cups(fr["rgb_stack"], fr["depth_stack_mm"].astype(np.float32) / 1000.0, Kc, Tc)
assert [c["kind"] for c in r7s["cups"]] == ["stack"], r7s
lower = truth["stack"]["cup_2"]
assert math.dist(r7s["cups"][0]["base_xyz_m"][:2], lower[:2]) * 1000 <= 4.0
assert abs((r7s["cups"][0]["height_mm"] - r7["cups"][0]["height_mm"]) - 25.3) < 3.0
print(f"[S7] fixed D435i frames: 2 cups (axis err <= 3 mm, rim z <= 4 mm), stacked -> one 'stack' cluster "
      f"(+{r7s['cups'][0]['height_mm'] - r7['cups'][0]['height_mm']:.1f} mm over a single cup)")

# ------------------------------------------------------------------ S8
import types  # noqa: E402
import threading  # noqa: E402
import multiprocessing as _mp  # noqa: E402
import NERO_V331_Fixed_Camera as F  # noqa: E402
fake = types.ModuleType("mujoco")


class _mjtObj:
    mjOBJ_BODY, mjOBJ_CAMERA, mjOBJ_JOINT, mjOBJ_GEOM = 1, 7, 3, 5


class _Model:
    nq = 30
    ngeom = 0
    vis = types.SimpleNamespace(global_=types.SimpleNamespace(offwidth=640, offheight=480))


class _Data:
    def __init__(self, model):
        self.qpos = np.zeros(30)
        self.xpos = np.zeros((0, 3))
        self.xquat = np.zeros((0, 4))


DEP = fr["depth_two_mm"].astype(np.float32) / 1000.0


class _Renderer:
    def __init__(self, model, height, width):
        self.dm = False

    def enable_depth_rendering(self):
        self.dm = True

    def disable_depth_rendering(self):
        self.dm = False

    def update_scene(self, data, camera=None):
        pass

    def render(self):
        return DEP.copy() if self.dm else fr["rgb_two"]

    def close(self):
        pass


fake.mjtObj = _mjtObj
fake.MjModel = types.SimpleNamespace(from_xml_path=lambda p: _Model())
fake.MjData = _Data
fake.mj_name2id = lambda model, t, n: 0 if n == P8.CAMERA_NAME else -1
fake.mj_forward = lambda model, data: None
fake.Renderer = _Renderer
real_mj = sys.modules.get("mujoco")
sys.modules["mujoco"] = fake
os.environ["NERO_V343_ACTIVE_SCENE"] = os.environ["NERO_SCENE"] = "stack_cups"
egg_calls = []
_orig_pe = P8.perceive_eggs
P8.perceive_eggs = lambda *a, **k: egg_calls.append(1) or ([], {})
try:
    sh = dict(qpos=_mp.Array("d", 1024), meta=_mp.Array("d", 4), model_path=_mp.Array("c", 4096),
              trig_seq=_mp.Value("i", 0), trig_tag=_mp.Array("c", 256), status=_mp.Value("i", 0),
              res_seq=_mp.Value("i", 0), res_buf=_mp.Array("c", 65536))
    bp = b"/fake/scene.xml"
    sh["model_path"][:len(bp)] = bp
    sh["model_path"][len(bp)] = b"\0"
    sh["meta"][0], sh["meta"][1], sh["meta"][3] = 30, 1.0, 1.0
    import time  # noqa: E402
    stop = threading.Event()
    with tempfile.TemporaryDirectory() as td:
        th = threading.Thread(target=F.fixed_camera_process_main, args=(
            sh["qpos"], sh["meta"], sh["model_path"], sh["trig_seq"], sh["trig_tag"], sh["status"], stop, td, "fake",
            sh["res_seq"], sh["res_buf"]))
        th.start()
        t0 = time.time()
        while sh["status"].value != 1 and time.time() - t0 < 30:
            time.sleep(0.05)
        assert sh["status"].value == 1
        tag = b"cups:observe_scene"
        sh["trig_tag"][:len(tag)] = tag
        sh["trig_tag"][len(tag)] = b"\0"
        sh["trig_seq"].value = 1
        t0 = time.time()
        while sh["res_seq"].value != 1 and time.time() - t0 < 60:
            time.sleep(0.05)
        payload = json.loads(bytes(sh["res_buf"][:]).split(b"\0", 1)[0].decode())
        stop.set()
        th.join(10)
finally:
    P8.perceive_eggs = _orig_pe
    if real_mj is not None:
        sys.modules["mujoco"] = real_mj
    else:
        sys.modules.pop("mujoco", None)
    os.environ.pop("NERO_V343_ACTIVE_SCENE")
    os.environ.pop("NERO_SCENE")
assert "error" not in payload, payload
assert [c["kind"] for c in payload["cups"]["cups"]] == ["cup", "cup"] and payload["camera"] == "fixed_d435i"
assert not egg_calls and payload["objects"] == {}
print("[S8] fixed-camera process in the stack_cups scene answers with 2 perceived cups, egg perception not run")

# ------------------------------------------------------------------ S9
if HAVE_MJ:
    r9 = HB.run_backend_episode(pose_mode="camera", wrist=True)
    assert any("wrist-camera cup perception is not available" in l for l in r9["log"]) and not r9["started"]
    r9 = HB.run_backend_episode(pose_mode="camera", camera_up=False)
    assert any("fixed camera process is not running" in l for l in r9["log"]) and not r9["started"]
    gl_ok = False
    try:
        _t = CK.TwinIO()
        _t.render_fixed_camera()
        gl_ok = True
    except Exception as exc:
        gl_reason = str(exc)[:80]
    if gl_ok:
        r9 = HB.run_backend_episode(pose_mode="camera")
        assert r9["state"]["nested_full"] and r9["scores"][0]["s"] == 1.0, r9["log"][-8:]
        assert r9["calls"].get("captures", 0) >= 5
        assert any("fixed stand D435i RGB-D cup perception" in l for l in r9["log"])
        print(f"[S9] pose-source rules OK; full task with the fixed D435i as pose source: nested, s=1, "
              f"{r9['calls']['captures']} captures")
    else:
        print(f"[S9] pose-source rules OK (wrist refused, camera down refused); camera task skipped: no OpenGL ({gl_reason})")
else:
    print("[S9] skipped: MuJoCo not installed")

# ------------------------------------------------------------------ S10
cs = next(p for p in ST.builtin_strategy_presets() if p.code == "CS")
assert "V3.63" in cs.skill_card["behavior_summary"] and "D435i" in cs.skill_card["behavior_summary"]
assert len(ST.KNOB_SPECS) == 56
for p in ST.builtin_strategy_presets():
    d = json.loads((ROOT / "config" / "strategies" / f"{p.skill_id}.json").read_text(encoding="utf-8"))
    e = p.to_dict()
    d.pop("saved"), e.pop("saved")
    assert d == e, p.skill_id
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
SNAP1 = snapshot()
assert SNAP1 == SNAP0, [k_ for k_ in SNAP0 if SNAP0[k_] != SNAP1.get(k_)]
print("[S10] CS card = V3.63 behaviour, JSON == definitions, 56 knobs, console builds, release folder unchanged")
print("ALL V3.63 TESTS PASSED")
