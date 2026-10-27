#!/usr/bin/env python3
"""V3.38: USB camera runs again, the task survives a network hiccup, a better-aimed recenter, more accurate slots.

S1 JSON: the V3.37 USB crash ('Object of type bool is not JSON serializable') came from a numpy bool in the carton
   height measurement (extent_ok); every published camera payload now goes through json_safe(); a full USB-style
   payload with a measured carton height serialises with the plain json module
S2 API retry: DNS failure / connection reset / timeout / HTTP 429 + 5xx are retried with back-off and the same request
   continues the task; non-transient errors (HTTP 401, bad JSON) fail at once; STOP ends the wait
S3 recenter: BioIK 0 feasible no longer triggers a move first (cuRobo plans here, as in the V3.37 log); 1-2 feasible
   or a big joint swing still do; a plan failure here -> recenter -> plan again; centre points IK-screened
S4 slots: carton centre fused over the views at the measured height; slot refine from two agreeing complete views
   (4..15 mm); pre-place look tries farther views one distance at a time, then the carton centre
"""
import json, math, os, sys, textwrap, threading, time, types
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np                       # noqa: E402
import NERO_V326_Wrist_Camera as W       # noqa: E402
import NERO_V308_Camera_Perception as P8  # noqa: E402
import v329_synth as S                   # noqa: E402

assert W.VERSION in ("V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1") and (ROOT / "VERSION").read_text().strip() in ("V3.38", "V3.39", "V3.40", "V3.41", "V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
WM = (ROOT / "NERO_V326_Wrist_Camera.py").read_text()
FM = (ROOT / "NERO_V331_Fixed_Camera.py").read_text()
import hashlib
assert hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest().startswith("020cc6680f181872")

# ------------------------------------------------------------------ S1
cfg = W.wrist_config("usb130"); K = P8.intrinsics(cfg); pc, fc = W.camera_in_tcp(cfg)
hx, hy = -0.42, 0.155
parts = [S.table_rgb_part(), S.carton_rgb_part(hx, hy, 0.0)]


def view(pt, shift=0.0):
    c = [x for x in W.look_pose_candidates(pt, cfg=cfg, distances=[0.45]) if abs(x["tilt_deg"]) == 30][0]
    T = S.camera_T_for(c, pc, fc).copy(); T[0, 3] += shift
    h = W.detect_holder_rgb(S.render_rgb(parts, K, T), K, T)
    return h, h.pop("_px_uv"), T


H = W.HolderHeight()
views = [view([hx, hy, 0.043]), view([-0.44, 0.105, 0.02]), view([-0.44, 0.315, 0.02])]
for h_, px_, T_ in views:
    H.add(px_, K, T_)
assert H.est["ok"] and H.top_z() is not None
assert all(type(v) in (bool, int, float, str, type(None)) for v in H.est.values()), \
    {k: type(v).__name__ for k, v in H.est.items()}
json.dumps(H.est)                                           # V3.37: TypeError here
payload = {"objects": {"egg_1": {"seen": np.bool_(True), "xyz_m": np.array([0.1, 0.2, 0.3]), "n": np.int64(3)}},
           "containers": {"egg_holder": dict(views[0][0], height_measurement=H.est)},
           "measured": {"holder_height": H.est, "nan": float("nan")}, "t": (1, 2)}
try:
    json.dumps(payload)
    raise AssertionError("expected the raw payload to fail")
except TypeError:
    pass
js = json.loads(json.dumps(W.json_safe(payload)))
assert js["objects"]["egg_1"] == {"seen": True, "xyz_m": [0.1, 0.2, 0.3], "n": 3} and js["measured"]["nan"] is None
assert "P8._post_result(res_seq, res_buf, cur, json_safe({" in WM and "json.dumps(json_safe(meta))" in WM
assert "P8._post_result(res_seq, res_buf, cur, W.json_safe({" in FM
print("[S1] carton height measurement = plain JSON types (extent_ok was numpy bool -> V3.37 USB crash); every "
      "camera payload passes json_safe()")

# ------------------------------------------------------------------ S2
import NERO_V2_7_1_Agent as AG                              # noqa: E402
os.environ["NERO_V338_API_RETRY_DELAYS_S"] = "0.01,0.01,0.01"
logs = []
ag = AG.NeroLLMAgent.__new__(AG.NeroLLMAgent)
ag.emit = logs.append
ag._stop_event = threading.Event()
calls = {"n": 0}


def flaky(payload, api_key, timeout_s=180.0):
    calls["n"] += 1
    if calls["n"] <= 2:
        raise AG.LLMAPIError("OpenAI API network error: <urlopen error [Errno -3] Temporary failure in name resolution>")
    return {"id": "ok"}


ag._post_responses_once = flaky
assert ag._post_responses({}, "k") == {"id": "ok"} and calls["n"] == 3
assert sum("[V3.38 API RETRY]" in m for m in logs) == 2
calls["n"] = -10                                            # always failing -> gives up after the delays
try:
    ag._post_responses({}, "k"); raise AssertionError
except AG.LLMAPIError:
    pass
for msg, transient in (("OpenAI API HTTP 401: bad key", False), ("OpenAI API HTTP 503: busy", True),
                       ("OpenAI API returned non-JSON response", False), ("OpenAI API HTTP 429: rate", True)):
    n = {"k": 0}

    def once(payload, api_key, timeout_s=180.0, m=msg, n=n):
        n["k"] += 1
        if n["k"] == 1:
            raise AG.LLMAPIError(m)
        return {"id": "x"}
    ag._post_responses_once = once
    if transient:
        assert ag._post_responses({}, "k") == {"id": "x"}
    else:
        try:
            ag._post_responses({}, "k"); raise AssertionError(msg)
        except AG.LLMAPIError:
            assert n["k"] == 1
ag._stop_event.set(); calls["n"] = 0; ag._post_responses_once = flaky
try:
    ag._post_responses({}, "k"); raise AssertionError
except AG.LLMAPIError:
    assert calls["n"] == 1                                   # STOP: no retry
print("[S2] DNS / reset / timeout / 429 / 5xx retried with back-off (same request -> task continues); 401 / bad JSON "
      "fail at once; STOP ends the wait")

# ------------------------------------------------------------------ S3


def block(start, end):
    i = B.index(start); return textwrap.dedent(B[i:B.index(end, i)])


ns = {"math": math, "os": os, "_V302_YAWS": list(range(12))}
ns["skill_core"] = types.SimpleNamespace(_current_joint_state_copy=lambda: None,
                                         _arm_positions_from_joint_state=lambda _j: [0.0] * 7)
exec(block("            class _V337Recenter(Exception):", "            def _v302_vertical_goal("), ns)
cand = lambda tag: (tag, [0, 0, 0], [0, 0, 0], [0, 0, 0, 1])
assert ns["_v337_risk"]([]) is None                            # 0 feasible: cuRobo first (V3.37 log)
assert "only 1 feasible" in ns["_v337_risk"]([((0,), cand("p2/yaw+180"), [0.1] * 7, "c")])
for mk in ('except RuntimeError as e338:', '"no yaw-family goal survived" in str(e338)',
           '_v337_recenter("no reorientation plan from this pose")', '"NERO_V338_RECENTER_POINTS"',
           '"NERO_V338_RECENTER_Z_M", "0.23,0.20,0.26"', "[V3.38 RECENTER SCREEN]",
           "j338, _e338 = skill_core._compute_ik(tcpt, list(tcpq)"):
    assert mk in B, mk
print("[S3] recenter only for 1-2 feasible goals / big joint swing, or after a failed plan here; centre points + "
      "heights IK-screened with the current grasp orientation")

# ------------------------------------------------------------------ S4
fcen = H.fused_center()
err_f = math.dist(fcen, [hx, hy]) * 1000
err_v = [math.dist(v[0]["center_xy_m"], [hx, hy]) * 1000 for v in views]
assert err_f < 3.0, (err_f, err_v)
assert W.HolderHeight().fused_center() is None
for mk in ("fc338 = hh_meas[pf].fused_center()", '"NERO_V338_FUSED_MAX_SHIFT_M", "0.015"', 'holder["fused_centre_shift_mm"]'):
    assert mk in WM, mk
# slot refine
logs = []
caps = []
g = {"math": math, "os": os, "skill_core": types.SimpleNamespace(emit=logs.append),
     "V271_SLOT_CENTERS_M": {"slot_2": (-0.450, 0.122, 0.043)}, "EGG_SLOT_POSES": {"slot_2": (-0.450, 0.122, 0.043)},
     "V335_STATE": {"frozen_place": {"slot": "slot_2", "xyz": [-0.450, 0.122, 0.043]}},
     "_v335_capture": lambda tag: caps.pop(0), "_v335_ingest_live": lambda p: None}
exec(block("    def _v338_view_slot(payload, sl):", "    def _v335_slot_moved_now("), g)


def pay(x, y, complete=True):
    return {"containers": {"egg_holder": {"complete": complete, "slots": {"slot_2": {"xyz_m": [x, y, 0.043]}}}}}


plan = {"slot_center_xyz_m": [-0.450, 0.122, 0.043]}
caps[:] = [pay(-0.4442, 0.1230)]
new = g["_v338_refine_slot"]("egg_2", "slot_2", plan, pay(-0.4440, 0.1228), "move_to_slot_entry")
assert new and abs(new[0] + 0.4441) < 1e-6 and plan["slot_center_xyz_m"][0] == new[0]
assert g["V271_SLOT_CENTERS_M"]["slot_2"][0] == new[0] and g["V335_STATE"]["frozen_place"]["xyz"][0] == new[0]
plan = {"slot_center_xyz_m": [-0.450, 0.122, 0.043]}
caps[:] = [pay(-0.4380, 0.1230)]                               # 2nd view disagrees by 6 mm -> keep
assert g["_v338_refine_slot"]("egg_2", "slot_2", plan, pay(-0.4440, 0.1228), "move_to_slot_entry") is None
assert g["_v338_refine_slot"]("egg_2", "slot_2", plan, pay(-0.4485, 0.122), "x") is None          # 1.5 mm: fine
assert g["_v338_refine_slot"]("egg_2", "slot_2", plan, pay(-0.430, 0.122), "x") is None           # 20 mm: moved
assert g["_v338_refine_slot"]("egg_2", "slot_2", plan, pay(-0.444, 0.122, complete=False), "x") is None
assert '_v338_refine_slot(obj, sl, plan, p, name)' in B and 'name in ("orient_for_slot", "move_to_slot_entry")' in B
for mk in ('"NERO_V338_PREPLACE_DIST_M", "0.45,0.40,0.52,0.36"', 'distances=[d338], no_detour=True',
           '("carton_centre", [float(h["center_xy_m"][0])', "[V3.38 PRE-PLACE LOOK]"):
    assert mk in B, mk
print(f"[S4] carton centre fused at the measured height: {err_f:.1f} mm (single views "
      f"{', '.join(f'{e:.1f}' for e in err_v)} mm); slot refine 4-15 mm from two agreeing complete views; pre-place "
      "look: farther views one distance at a time, then the carton centre")
print("ALL V3.38 TESTS PASSED")
