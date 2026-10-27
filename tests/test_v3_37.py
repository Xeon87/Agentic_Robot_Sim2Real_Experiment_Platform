#!/usr/bin/env python3
"""V3.37: no more egg drop from the safety restore, risky reorientation re-centred first, latch cleared before the
return, dropped egg = LOST + searched, GUI tab "NERO SimAgent" with foldable sections and a zoomable camera monitor.

R1 recenter: the whole-arm reorientation is flagged risky when < NERO_V337_MIN_FEASIBLE yaw goals are reachable
   (V3.36 egg_1: 1/60) or the best goal swings a joint > 150 deg; the egg is then carried (orientation unchanged) to
   the table centre and the yaw family is planned again from there
R2 keep-hold restore: the automatic RUNAWAY hard restore carries the held egg with the gripper (same pose relative
   to link7), keeps force mode + the CONNECT hold; a second restore within the window and the manual HARD RESET 0
   release as before; run_sim outside hard_restore_arm is still byte-identical to V3.34
R3 latch + lost egg: return_to_pick_location clears a held safety latch (STOP + RECOVER handshake) before moving;
   an egg that leaves the gripper outside a release skill is LOST (no pose in the scene, plan_grasp refused),
   observe_scene searches around the drop point, the trackers forget its stale track so the egg is named again
R4 GUI: tab "NERO SimAgent", every section folds / unfolds (Tk under Xvfb with python3.11 if available), monitor
   zoom 25-300 % (PIL or Tk zoom/subsample), pop-out fits its window
"""
import ast, hashlib, math, os, shutil, subprocess, sys, tempfile, textwrap, threading, time, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np                       # noqa: E402
import NERO_V326_Wrist_Camera as W       # noqa: E402
import NERO_V308_Camera_Perception as P8  # noqa: E402

assert W.VERSION in ("V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
assert (ROOT / "VERSION").read_text().strip() in ("V3.37", "V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
import re as _re343  # V3.43: the A2 scene-setup blocks are stripped; everything else must be unchanged
B = _re343.sub(r"\n[ \t]*# V3\.43 A2 BEGIN\n.*?\n[ \t]*# V3\.43 A2 END(?=\n)", "", B, flags=_re343.S)
A = (ROOT / "NERO_V2_7_1_Agent.py").read_text()


def block(start, end, dedent=True):
    i = B.index(start)
    j = B.index(end, i)
    s = B[i:j]
    return textwrap.dedent(s) if dedent else s


# ------------------------------------------------------------------ R1
ns = {"math": math, "os": os, "_V302_YAWS": list(range(12))}
cur = [0.0] * 7


class SK:
    def _current_joint_state_copy(self):
        return None

    def _arm_positions_from_joint_state(self, _js):
        return list(cur)


ns["skill_core"] = SK()
exec(block("            class _V337Recenter(Exception):", "            def _v302_vertical_goal("), ns)
risk = ns["_v337_risk"]
cand = lambda tag: (tag, [0, 0, 0], [0, 0, 0], [0, 0, 0, 1])
assert "only 1 feasible" in risk([((0,), cand("p2/yaw+180"), [0.1] * 7, "current")])
assert risk([]) is None or "no feasible" in risk([])          # V3.38: 0 feasible -> plan here first
ok13 = [((0,), cand(f"p3/yaw{k}"), [0.5] * 7, "current") for k in range(13)]
assert risk(ok13) is None
far = [((0,), cand("p0/yaw+180"), [0, 0, 0, 0, 0, 0, math.radians(170)], "current")] + ok13
assert "swings a joint 170deg" in risk(far)
for mk in ("raise _V337Recenter(why337)", "if risk337 and not use303:", "def _v337_recenter(why337):",
           "except _V337Recenter as r337:", '"NERO_V337_RECENTER_XY", "-0.42,-0.02"', "[V3.37 RECENTER]",
           "risk337=use337)", 'err302, tag302 = _v302_vertical_goal(_v302_anchors(), "whole_arm_yaw_family", max_plan302=4)\n',
           "_v224_execute_pose_plan(f\"v337_recenter({obj},{slot_name})\""):
    assert mk in B, mk
# the slot-above transport (corridor goals) never re-centres
assert B.count("risk337=") == 2 and "corridor303=True" in B
print("[R1] risky reorientation (1/60 feasible as V3.36 egg_1, or a 170 deg joint swing) -> recenter first; 13 "
      "feasible + small swing -> plan as before")

# ------------------------------------------------------------------ R2
src_tree = ast.parse(B)
run_sim_src = [ast.get_source_segment(B, n) for n in ast.walk(src_tree)
               if isinstance(n, ast.FunctionDef) and n.name == "run_sim"][0]
i0 = run_sim_src.find("        _v337_keep = {"); i1 = run_sim_src.find("        def soft_freeze_arm(")
core = run_sim_src[:run_sim_src.find("        _v337_keep = {")] + run_sim_src[i1:]
core = core.replace("                            keep_hold=True,\n", "")
assert hashlib.sha256(core.encode()).hexdigest()[:16] == "0564058f38cb9fab", "run_sim outside the restore changed"
assert "keep_hold=True" in run_sim_src and run_sim_src.count("keep_hold=True") == 1
assert "gripper_q=0.040,\n                        reason_code=3,\n                        count_trip=False,\n                    )" in run_sim_src
restore_src = textwrap.dedent(run_sim_src[i0:i1])


def rz(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])


def q2m(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def m2q(out, m):
    m = np.asarray(m, float).reshape(3, 3)
    w = math.sqrt(max(1e-12, 1 + m[0, 0] + m[1, 1] + m[2, 2])) / 2
    out[:] = [w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)]


class Val:
    def __init__(self, v):
        self.value = v


def make_sim():
    """Fake MuJoCo: body 1 = link7 (pos q0..q2, yaw q3), body 2 = egg on a free joint (qpos 9..15)."""
    data = types.SimpleNamespace(qpos=np.zeros(16), qvel=np.zeros(15), qacc=np.zeros(15), qfrc_applied=np.zeros(15),
                                 xfrc_applied=np.zeros((3, 6)), ctrl=np.zeros(2), eq_active=np.array([1]),
                                 xpos=np.zeros((3, 3)), xmat=np.zeros((3, 9)), xquat=np.zeros((3, 4)))
    model = types.SimpleNamespace(nu=2, body_jntadr=np.array([-1, -1, 0]), jnt_type=np.array([0]),
                                  jnt_qposadr=np.array([9]), jnt_dofadr=np.array([9]))

    def fwd(_m, d):
        d.xpos[1] = d.qpos[0:3]; d.xmat[1] = rz(d.qpos[3]).reshape(-1)
        d.xpos[2] = d.qpos[9:12]; d.xmat[2] = q2m(d.qpos[12:16]).reshape(-1)
    mj = types.SimpleNamespace(mj_forward=fwd, mju_mat2Quat=m2q,
                               mjtJoint=types.SimpleNamespace(mjJNT_FREE=0))
    g = {"np": np, "os": os, "time": time, "mujoco": mj, "data": data, "model": model, "link7_bid": 1,
         "hold_eq_by_body": {2: 0}, "hold_name_by_body": {2: "egg_1"}, "qpos_addr": list(range(7)),
         "dof_addr": list(range(7)), "JOINT_LIMITS_RAD": [(-3.0, 3.0)] * 7, "gripper_qpos_addr": 7,
         "gripper2_qpos_addr": 8, "gripper_dof_addr": 7, "gripper2_dof_addr": 8,
         "target_shared": [0.0] * 7, "actual_shared": [0.0] * 7, "print": lambda *a, **k: None}
    for n in ("gripper_target_mm", "gripper_actual_mm", "anti_slip_boost_shared", "anti_slip_slip_mm_s_shared",
              "anti_slip_mu_shared", "contact_left_n_shared", "contact_right_n_shared", "contact_tangent_n_shared",
              "contact_motor_cmd_shared", "hold_name_shared", "hold_ready_shared", "hold_status_code_shared",
              "safety_state", "safety_trip_count"):
        g[n] = Val(0)
    g["gripper_force_mode"] = Val(True); g["hold_active_shared"] = Val(True)
    g["hold_now_request"] = Val(False); g["hold_off_request"] = Val(False)

    def deact():
        data.eq_active[0] = 0
    g["deactivate_all_holds"] = deact
    code = ("def _outer():\n    q_ref = np.zeros(7); q_ref_prev = np.zeros(7); next_step_wall = 0.0; "
            "safety_cooldown_until = 0.0\n" + textwrap.indent(restore_src, "    ") + "\n    return hard_restore_arm\n")
    exec(code, g)
    # arm somewhere, egg held 0.12 m in front of link7 with a tilt
    data.qpos[0:4] = [-0.45, -0.22, 0.30, 0.7]
    data.qpos[7] = 0.0275
    data.qpos[9:12] = [-0.45 + 0.12 * math.cos(0.7), -0.22 + 0.12 * math.sin(0.7), 0.27]
    data.qpos[12:16] = [math.cos(0.3), math.sin(0.3), 0, 0]
    fwd(model, data)
    return g, data, model, fwd


def rel(data):
    R = data.xmat[1].reshape(3, 3)
    return R.T @ (data.xpos[2] - data.xpos[1]), R.T @ data.xmat[2].reshape(3, 3)


g, data, model, fwd = make_sim()
hr = g["_outer"]()
p_rel0, R_rel0 = rel(data)
safe_q = [-0.30, 0.05, 0.40, -1.2, 0.0, 0.0, 0.0]
hr(safe_q, gripper_q=0.040, reason_code=4, keep_hold=True)
p_rel1, R_rel1 = rel(data)
assert np.allclose(data.qpos[0:4], safe_q[:4]), "arm restored"
assert np.allclose(p_rel0, p_rel1, atol=1e-9) and np.allclose(R_rel0, R_rel1, atol=1e-6), "egg moved with the gripper"
assert abs(data.qpos[7] - 0.0275) < 1e-12, "fingers stay closed (not the stale last-safe opening)"
assert data.eq_active[0] == 1 and g["gripper_force_mode"].value is True and g["hold_active_shared"].value is True
assert g["safety_trip_count"].value == 1 and g["safety_state"].value == 4
assert np.allclose(data.qvel[9:15], 0)
# second restore inside the window -> old behaviour (release)
hr(safe_q, gripper_q=0.040, reason_code=4, keep_hold=True)
assert data.eq_active[0] == 0 and g["gripper_force_mode"].value is False and g["hold_active_shared"].value is False
# manual HARD RESET 0 (keep_hold=False) releases as in V3.34
g2, d2, _m2, _f2 = make_sim(); hr2 = g2["_outer"]()
hr2([0.0] * 7, gripper_q=0.040, reason_code=3, count_trip=False)
assert d2.eq_active[0] == 0 and g2["gripper_force_mode"].value is False and abs(d2.qpos[7] - 0.040) < 1e-12
# switch off
os.environ["NERO_V337_RESTORE_KEEP_HOLD"] = "0"
g3, d3, _m3, _f3 = make_sim(); g3["_outer"]()(safe_q, reason_code=4, keep_hold=True)
assert d3.eq_active[0] == 0
del os.environ["NERO_V337_RESTORE_KEEP_HOLD"]
print("[R2] RUNAWAY restore carries the held egg (relative pose to link7 kept to 1e-9 m), hold + force mode stay; "
      "2nd restore in 3 s / HARD RESET 0 / switch=0 release as in V3.34; run_sim outside the restore = V3.34")

# ------------------------------------------------------------------ R3
log = []
st = {"hold": True, "trip": 4}


class Skill:
    def emit(self, m):
        log.append(m)

    def cancel_active_motion(self):
        log.append("cancel")

    def _set_arm_command_gate(self, on):
        log.append(f"gate={on}")

    def _sync_arm_controller_to_feedback(self):
        log.append("sync")

    def current_tcp_pose(self):
        return [-0.40, -0.10, 0.25], [1, 0, 0, 0]


hold = Val(True); trips = Val(4); recover = Val(False); gate = Val(False)
EST = {"released": {}, "held_last": ("egg_1", ([-0.455, -0.224, 0.14], [0, 0, 0, 1])), "lost": {}}
V311 = {"last": {"capture": 80, "objects": {"egg_1": {"seen": True, "trusted": True, "xyz_m": [-0.455, -0.224, 0.022]},
                                            "egg_2": {"seen": True, "trusted": True, "xyz_m": [-0.43, -0.17, 0.02]}}}}
inner_calls = []
state = {"stage": "LIFTED", "active_object": "egg_1"}


def inner(name, args):
    inner_calls.append(name)
    if name == "orient_for_slot":
        hold.value = False                                  # the egg leaves the gripper during the tool
        return {"ok": False, "tool": name, "warnings": []}
    if name == "observe_scene":
        return {"ok": True, "tool": name, "warnings": []}
    return {"ok": True, "tool": name}


g = {"time": time, "math": math, "os": os, "skill_core": Skill(), "safety_trip_count": trips,
     "skill_poll_state": {"last_recovered_trip": 0, "safety_latch_held": False}, "recover_request": recover,
     "ros_arm_command_gate_shared": gate, "hold_active_shared": hold, "_V329_EST": EST, "V311_STATE": V311,
     "V271_PLACEMENT_RUNTIME": {"plans": {"egg_1": 1}, "oriented": {}, "slot_entry": {}, "released": {}},
     "V27_GRASP_RUNTIME": {"plans": {"egg_1": 1}, "selected": {}}, "V3121_HELD_REL": {"egg_1": 1},
     "V335_STATE": {"frozen_grasp": {}, "frozen_place": {"obj": "egg_1"}},
     "_v3121_held_name": lambda: "egg_1" if hold.value else None,
     "_v3121_held_pose_estimate": lambda n, t, q: ([-0.44, -0.20, 0.20], [0, 0, 0, 1]),
     "llm_scene_snapshot": lambda: {"objects": {"egg_1": {"xyz_m": [-0.455, -0.224, 0.022], "location": "egg_basket"},
                                                "egg_2": {"xyz_m": [-0.43, -0.17, 0.02]}}},
     "llm_tool_runner": inner, "_v329_no_prior": lambda: True, "_v24_state_snapshot": lambda: dict(state),
     "_v3121_pending_eggs": lambda: ["egg_1", "egg_2"]}
exec(block("    V337_INTENDED_RELEASE = ", "    _v335_cap_lock()"), g)
run = g["llm_tool_runner"]
# (a) latch held -> STOP + RECOVER handshake before return_to_pick_location


def mujoco_ack():
    t_end = time.time() + 2
    while time.time() < t_end:
        if recover.value:
            recover.value = False
            return
        time.sleep(0.01)


th = threading.Thread(target=mujoco_ack); th.start()
r = run("return_to_pick_location", {"object_name": "egg_1"}); th.join()
assert inner_calls[-1] == "return_to_pick_location" and r["ok"]
assert g["skill_poll_state"]["last_recovered_trip"] == 4 and "sync" in log and "cancel" in log
assert any("[V3.37 SAFETY LATCH CLEARED]" in m for m in log)
# no ACK -> refused, no motion
g["skill_poll_state"]["last_recovered_trip"] = 0; n0 = len(inner_calls)
r = run("return_to_pick_location", {"object_name": "egg_1"})
assert r["reason"] == "SAFETY_LATCH_NOT_CLEARED" and len(inner_calls) == n0
g["skill_poll_state"]["last_recovered_trip"] = 4
# (b) the egg leaves the gripper during orient_for_slot -> LOST
hold.value = True
r = run("orient_for_slot", {"object_name": "egg_1", "slot_name": "slot_1"})
lost = EST["lost"]["egg_1"]
assert r["v337_egg_lost"]["object_name"] == "egg_1" and r["recommended_next_actions"] == ["observe_scene"]
assert [-0.44, -0.20] in lost["search_xy"] and [-0.40, -0.10] in lost["search_xy"]
assert "egg_1" not in g["V3121_HELD_REL"] and g["V335_STATE"]["frozen_place"] is None
snap = g["llm_scene_snapshot"]()
assert snap["objects"]["egg_1"]["location"] == "lost_unknown" and "xyz_m" not in snap["objects"]["egg_1"]
assert snap["objects"]["egg_2"]["xyz_m"]
r = run("plan_grasp", {"object_name": "egg_1"})
assert r["reason"] == "EGG_LOST_NOT_RELOCATED" and inner_calls[-1] != "plan_grasp"
# observe: still the stale capture #80 (egg_1 "seen" there, but before the loss) -> NOT found
r = run("observe_scene", {})
assert "egg_1" in EST["lost"] and EST["lost"]["egg_1"]["searches"] == 1
# a new capture names egg_1 again -> found -> plan_grasp passes
V311["last"] = {"capture": 85, "objects": {"egg_1": {"seen": True, "trusted": True, "xyz_m": [-0.33, -0.27, 0.022]}}}
r = run("plan_grasp", {"object_name": "egg_1"})
assert "egg_1" not in EST["lost"] and inner_calls[-1] == "plan_grasp"
assert any("[V3.37 EGG FOUND] egg_1" in m for m in log)
# intended releases never mark lost
hold.value = True; n_lost = len(EST["lost"])
inner2 = g["_llm_tool_runner_v335"]


def rel_inner(name, args):
    hold.value = False
    return {"ok": True}


g["_llm_tool_runner_v335"] = rel_inner
run("guarded_insert_and_release", {"object_name": "egg_2", "slot_name": "slot_1"})
assert len(EST["lost"]) == n_lost
g["_llm_tool_runner_v335"] = inner2
# (c) lost hint -> trackers forget the stale track; the egg is named again where it now is
h = W.encode_hints("observe_scene/lost_egg_1_0", held=None, lost=["egg_1"])
tag, hh = W.decode_hints(h)
assert tag == "observe_scene/lost_egg_1_0" and hh["lost"] == ["egg_1"] and W.decode_hints("x")[1]["lost"] == []
Kd = P8.intrinsics(W.wrist_config("d435i"))
Tdown = [[1, 0, 0, -0.40], [0, -1, 0, -0.20], [0, 0, -1, 0.45], [0, 0, 0, 1]]


def det(i, p):
    return {"det_id": i, "xyz_m": p, "quat_xyzw": [0, 0, 0, 1], "axis_world": [1, 0, 0], "location": "x",
            "trusted": True, "quality": {}}


Tlow = [[1, 0, 0, -0.30], [0, -1, 0, 0.05], [0, 0, -1, 0.25], [0, 0, 0, 1]]     # close view, old poses out of it
for with_hint in (False, True):
    tr = W.ViewTracker(P8.PerceptionTracker(), names=["egg_1", "egg_2"])
    tr.update([det(0, [-0.40, -0.20, 0.02]), det(1, [-0.45, -0.20, 0.02])], 1, Kd, Tdown)   # egg_1 @-0.45, egg_2 @-0.40
    # V3.36 log: egg_1 dropped elsewhere; the camera sees it, but its stale track lies outside this view -> the
    # name pool was "full" ("1 trusted detection(s) without a free name") and the egg stayed unnamed
    assert W.in_view([-0.30, 0.04, 0.02], Kd, Tlow) and not W.in_view([-0.45, -0.20, 0.02], Kd, Tlow)
    o, dg = tr.update([det(0, [-0.30, 0.04, 0.02])], 2, Kd, Tlow, hints={"lost": ["egg_1"]} if with_hint else {})
    if with_hint:
        assert o["egg_1"]["seen"] and o["egg_1"]["xyz_m"] == [-0.30, 0.04, 0.02] and dg["lost_forgotten"] == ["egg_1"]
        assert not o["egg_2"]["seen"] and o["egg_2"]["xyz_m"] == [-0.40, -0.20, 0.02]
    else:
        assert not o["egg_1"]["seen"] and dg["unnamed_trusted"] == 1, (o, dg)
for mk in ('lost337 = dict(_V329_EST.get("lost") or {}) if (_v329_no_prior() and not holding) else {}',
           'extra337.append((f"lost_{lname}_{k}", [x, y, 0.02]))', "[V3.37 SEARCH HIT]",
           "live=live, lost=lost337)", '"NERO_V337_SEARCH_R_M", "0.12"'):
    assert mk in B, mk
assert "V3.37 SAFETY + LOST EGGS" in A and "EGG_LOST_NOT_RELOCATED" in A and "lost_unknown" in A
print("[R3] latch: STOP+RECOVER ACK + controller sync before return_to_pick_location (no ACK -> refused, no motion); "
      "egg out of the gripper during orient_for_slot -> LOST (no pose, plan_grasp refused, search area = held pose + "
      "TCP), stale capture does not count, a new capture finds it; the tracker re-names a dropped egg where it lies")

# ------------------------------------------------------------------ R4
assert 'notebook.add(llm_tab, text="NERO SimAgent")' in B and "NERO V2-9 | Agent" not in B
assert 'text="NERO SimAgent | Gripper-Aware Skill Contract Agent (simulation)"' in B
for mk in ("def _v337_make_collapsible(frame):", "frame.configure(labelwidget=lab)", '"NERO_V337_COLLAPSED"',
           'text="Collapse all"', 'text="Expand all"', 'V337_ZOOMS = ("25%", "50%", "75%", "100%", "125%", "150%", "200%", "300%")',
           '"<Control-MouseWheel>"', '_v335_draw(v335_mon_state["win_canvas"], meta, png, "fit")', '"NERO_V337_MONITOR_ZOOM"'):
    assert mk in B, mk
assert B.index("_v337_sections = [w for w in llm_outer.winfo_children()") > B.index('llm_util = ttk.LabelFrame(llm_outer')
gui = block("    # ----- V3.37 COLLAPSIBLE SECTIONS", "    def poll_llm_status")
zoom = block("    _V337_FRAC = ", "    def _v335_draw(")
script = r'''
import os, sys, tkinter as tk
from tkinter import ttk
root = tk.Tk(); llm_outer = ttk.Frame(root); llm_outer.pack(fill="both", expand=True)
secs = []
for t in ("Object pose source", "LLM agent controls", "Wrist camera monitor", "tool debugger"):
    f = ttk.LabelFrame(llm_outer, text=t); f.pack(fill="x", pady=4); secs.append(f)
    for k in range(3):
        ttk.Button(f, text=f"{t}{k}").pack(side="left")
    g = ttk.Frame(f); g.pack(fill="x"); ttk.Label(g, text="nested").grid(row=0, column=0)
os.environ["NERO_V337_COLLAPSED"] = "debugger"
if os.environ.get("NERO_TEST_NO_FONTS"):
    # some headless Tk builds abort on an explicit font / cursor (Xft, cursor font); both are cosmetic -> drop it in this test only
    _L = ttk.Label
    class _NoFontLabel(_L):
        def __init__(self, *a, **k):
            k.pop("font", None); k.pop("cursor", None)
            super().__init__(*a, **k)
    ttk.Label = _NoFontLabel
exec(compile(GUI, "gui", "exec"))
root.update()
st = V337_FOLD
assert len(st) == 4 and st[secs[3]]["collapsed"] and not st[secs[0]]["collapsed"]
assert st[secs[3]]["label"].cget("text").startswith("▶") and not secs[3].pack_slaves()[0].winfo_children()
h_open = secs[0].winfo_height()
_v337_fold(secs[0], True); root.update()
h_closed = secs[0].winfo_height()
assert h_closed < h_open - 10, (h_open, h_closed)
_v337_fold(secs[0], False); root.update()
names = [c.cget("text") for c in secs[0].pack_slaves() if isinstance(c, ttk.Button)]
assert names == ["Object pose source0", "Object pose source1", "Object pose source2"], names
assert abs(secs[0].winfo_height() - h_open) <= 2
st[secs[1]]["label"].event_generate("<Button-1>"); root.update()
assert st[secs[1]]["collapsed"]
for f in secs:
    _v337_fold(f, False)
root.update()
assert not any(st[f]["collapsed"] for f in secs)
# monitor zoom (Tk zoom/subsample path, no PIL here)
png = os.path.join(TMPD, "img.png")
tk.PhotoImage(width=64, height=36).write(png, format="png")
exec(compile(ZOOM, "zoom", "exec"))
for k, exp in ((0.5, 32), (1.0, 64), (1.5, 96), (2.0, 128), (0.25, 16), (3.0, 192), (1.3, 80)):
    img, kk = _v337_scaled(png, 64, 36, k)
    assert img.width() == exp, (k, img.width(), kk)
print("GUI_OK")
'''
tmpd = tempfile.mkdtemp(prefix="v337gui_")
xv = shutil.which("xvfb-run")
gui_ran = False
py = None
for cand_py in [os.environ.get("NERO_TEST_TK_PYTHON"), sys.executable, shutil.which("python3.11"), shutil.which("python3.12")]:
    if cand_py and subprocess.run([cand_py, "-c", "import tkinter; tkinter.Tcl()"], capture_output=True).returncode == 0:
        py = cand_py
        break
if py and xv:                      # headless only (never opens a window on the desktop)
    sp = Path(tmpd) / "gui.py"
    sp.write_text(f"GUI = {gui!r}\nZOOM = {zoom!r}\nTMPD = {tmpd!r}\n" + script)
    cmd = ([xv, "-a"] if xv else []) + [py, str(sp)]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert "GUI_OK" in out.stdout, out.stdout + out.stderr
    gui_ran = True
print("[R4] tab 'NERO SimAgent'; sections fold/unfold by title click, Collapse/Expand all, NERO_V337_COLLAPSED; "
      "order + nested widgets restored; monitor zoom 25-300 % "
      + ("(Tk run under Xvfb: OK)" if gui_ran else "(no tkinter/Xvfb here: source checks only)"))
print("ALL V3.37 TESTS PASSED")
