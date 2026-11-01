#!/usr/bin/env python3
"""V3.59 offline unit tests (written in V3.60: the V3.59 release referenced this file but did not ship it).
Covers the V3.59 features that V3.60 keeps; the V3.60 changes themselves are tested in tests/test_v3_60.py.
G1 NERO_V359_Strategy: 5 knob groups / 45 knob specs, every spec has key/group/label/type/default/unit/desc, numeric
   defaults inside [min, max], enum defaults inside choices; coerce/sanitize clamp and fall back to the baseline.
G2 StrategyProfile JSON round trip (user dir), diff_vs_baseline / describe_diff_vs_baseline, skill card sanitising.
G3 Dry-run helpers: parse_waypoint_spec (mm / m / label; V3.60 fix: metres with default z), build_dry_run_targets for the 4 route modes + selected rows.
G4 plan_hover_dry_run_sequence (transit lift above the higher hover, chained IK seeds, return home, failure
   message) + execute_hover_dry_run (move order, dwell, wrist check, stop request, move failure).
G5 export_llm_skill_catalog lists every built-in skill with its skill card.
G6 Main-camera calibration badge (V3.59 change, see CHANGELOG): "good" comes from the calibration result; LOO-only
   calibrations without warnings stay 合格 as in V3.58; LOO ok but warnings -> 未达标 with the explanation.
G7 Mocked console run_console('sim'/'real') builds the 「技能参数旋钮」 tab + Camera tab dry-run controls without errors.
"""
import math
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="nero_v359_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user")
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "active.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")

import NERO_V359_Strategy as ST        # noqa: E402
import NERO_V312_Console as CON        # noqa: E402
import NERO_V325_Camera as CAM         # noqa: E402

CAM.CALIB_PATH = str(TMP / "camera_calib.json")
CAM.LOCATE_PATH = str(TMP / "locate_settings.json")
CAM.CALIB_DIR = str(TMP / "cameracalib")

# ------------------------------------------------------------------ G1 knob specs
assert [g for g, _ in ST.KNOB_GROUPS] == ["grasp_scoring", "gripper_approach", "ik_posture", "placement_recovery", "hover_dry_run"]
assert len(ST.KNOB_SPECS) == 45 and len(ST.KNOB_BY_KEY) == 45
for s in ST.KNOB_SPECS:
    for f in ("key", "group", "label", "type", "default", "unit", "desc"):
        assert f in s, (s.get("key"), f)
    assert s["group"] in dict(ST.KNOB_GROUPS)
    if s["type"] == "float":
        assert s["min"] <= s["default"] <= s["max"], s["key"]
        assert ST.coerce_knob_value(s["key"], s["max"] + 100) == s["max"]
        assert ST.coerce_knob_value(s["key"], s["min"] - 100) == s["min"]
        assert ST.coerce_knob_value(s["key"], "nan") == s["default"] and ST.coerce_knob_value(s["key"], "x") == s["default"]
    elif s["type"] == "enum":
        assert s["default"] in s["choices"] and ST.coerce_knob_value(s["key"], "bogus") == s["default"]
    else:
        assert s["type"] == "bool" and ST.coerce_knob_value(s["key"], "true") is True and ST.coerce_knob_value(s["key"], "0") is False
k = ST.sanitize_knobs({"egg_open_margin_mm": 99, "unknown": 1})
assert k["egg_open_margin_mm"] == 26.0 and "unknown" not in k and len(k) == 45
print("[G1] 5 groups / 45 knobs: defaults inside bounds; clamping and fallbacks")

# ------------------------------------------------------------------ G2 profile JSON
p = ST.StrategyProfile(skill_id="t_profile", knobs={"egg_open_margin_mm": 13.0, "slot_fill_order": "far_to_near"},
                       skill_card={"name": "测试", "scene_tags": "a, b"}, short_name="Test Profile", code="TP")
fp = ST.save_strategy_file(p)
assert fp == str(TMP / "user" / "t_profile.json")
q = ST.load_strategy_file(fp)
assert q.knobs == p.knobs and q.skill_card["scene_tags"] == ["a", "b"] and (q.code, q.short_name) == ("TP", "Test Profile")
assert {d[0] for d in ST.diff_vs_baseline(q.knobs)} == {"egg_open_margin_mm", "slot_fill_order"}
txt = ST.describe_diff_vs_baseline(q.knobs)
assert "共修改 2 项" in txt and "16.0" in txt and "13.0" in txt
assert ST.describe_diff_vs_baseline(ST.BASELINE_KNOBS).startswith("100% 严格还原")
assert ST.StrategyProfile.from_dict("garbage").skill_id == ST.BASELINE_SKILL_ID
print("[G2] JSON round trip, diff and natural-language summary")

# ------------------------------------------------------------------ G3 waypoints / targets
w = ST.parse_waypoint_spec("-450, 123, 43")
assert w["xyz_m"] == [-0.45, 0.123, 0.043] and w["det_id"] == "target_B"
w = ST.parse_waypoint_spec("slot_1: -0.51, 0.123")
assert w["det_id"] == "slot_1" and w["xyz_m"] == [-0.51, 0.123, 0.035] and ST.parse_waypoint_spec("x") is None   # V3.60 fix
assert ST.parse_waypoint_spec("-450, 123")["xyz_m"] == [-0.45, 0.123, 0.035]
eggs = [{"det_id": "e1", "xyz_m": [0.3, 0.0, 0.03], "trusted": False},
        {"det_id": "e2", "xyz_m": [0.4, 0.1, 0.03], "trusted": True},
        {"det_id": "e3", "xyz_m": [0.5, 0.1, 0.03], "trusted": True},
        {"det_id": "slot_1", "xyz_m": [-0.5, 0.1, 0.04], "trusted": True},
        {"det_id": "bad", "xyz_m": [0.1, 0.1, 0.03], "trusted": True, "status": "rejected"}]
ids = lambda seq: [t["det_id"] for t in seq]
assert ids(ST.build_dry_run_targets(eggs)) == ["e2", "e3"]                       # trusted first, table order
assert ids(ST.build_dry_run_targets(eggs, dest_mode="all_trusted")) == ["e2", "e3", "slot_1"]
assert ids(ST.build_dry_run_targets(eggs, dest_mode="first_empty_slot")) == ["e2", "slot_1"]
assert ids(ST.build_dry_run_targets(eggs, dest_mode="manual_xyz", manual_dest_text="B: 100, 200, 50")) == ["e2", "B"]
assert ids(ST.build_dry_run_targets(eggs, selected_ids=["e3", "e1"])) == ["e3", "e1"]
print("[G3] waypoint parsing and the 4 dry-run route modes")

# ------------------------------------------------------------------ G4 dry-run plan + execute
seen = []


def fake_hover(urdf, joints, lim, q0, tool_T, xyz, base="base_link", fence_fn=None, soft_deg=3.0):
    seen.append((list(q0), list(xyz), soft_deg))
    return [len(seen) * 0.01] * 7, {"err_mm": 0.2}


tg = [{"det_id": "A", "xyz_m": [0.30, 0.00, 0.030]}, {"det_id": "B", "xyz_m": [0.30, 0.20, 0.050]}]
plan = ST.plan_hover_dry_run_sequence(None, [], [], [0.0] * 7, None, tg, z_table_m=0.0, plan_hover_fn=fake_hover,
                                      knobs={"cam_hover_verify_height_mm": 60.0, "dry_run_transit_z_lift_mm": 40.0})
assert plan["ok"] and [s["kind"] for s in plan["steps"]] == ["hover", "transit", "hover", "return_home"]
hA, tr, hB = plan["steps"][0]["tcp_xyz_m"], plan["steps"][1]["tcp_xyz_m"], plan["steps"][2]["tcp_xyz_m"]
assert abs(hA[2] - 0.09) < 1e-12 and abs(hB[2] - 0.11) < 1e-12 and abs(tr[2] - 0.15) < 1e-12 and abs(tr[1] - 0.10) < 1e-12
assert seen[1][0] == plan["steps"][0]["q_rad"] and seen[2][0] == plan["steps"][1]["q_rad"]      # chained seeds
assert plan["steps"][-1]["q_rad"] == [0.0] * 7 and all(x[2] == 3.0 for x in seen)
bad = ST.plan_hover_dry_run_sequence(None, [], [], [0.0] * 7, None, tg,
                                     plan_hover_fn=lambda *a, **kw: (None, "IK 无解"))
assert not bad["ok"] and "A" in bad["error"] and not ST.plan_hover_dry_run_sequence(None, [], [], [0] * 7, None, [])["ok"]
moves, logs = [], []
res = ST.execute_hover_dry_run(plan, move_fn=lambda q, label: (moves.append(label), {"ok": True})[1],
                               actual_q_fn=lambda: [0.0] * 7, fk_tcp_fn=lambda q: [0.30, 0.0, 0.091],
                               wrist_check_fn=lambda t: {"ok": True, "delta_tcp_mm": (0.5, -0.2), "delta_yaw_deg": 1.0},
                               sleep_fn=lambda s: None, log_fn=logs.append)
assert res["ok"] and res["completed_hovers"] == 2 and len(moves) == 4
assert abs(res["results"][0]["tcp_err_mm"] - 1.0) < 1e-6 and "wrist_refine" in res["results"][0]
res = ST.execute_hover_dry_run(plan, move_fn=lambda q, label: {"ok": True}, stop_fn=lambda: True, sleep_fn=lambda s: None)
assert res.get("stopped") and res["completed_hovers"] == 0
res = ST.execute_hover_dry_run(plan, move_fn=lambda q, label: {"ok": False, "error": "FENCE"}, sleep_fn=lambda s: None)
assert not res["ok"] and res["error"] == "FENCE"
print("[G4] dry-run plan (transit lift, chained seeds, return home) and execution (dwell, wrist check, stop, failure)")

# ------------------------------------------------------------------ G5 catalog
cat = ST.export_llm_skill_catalog(active_profile=ST.builtin_strategy_presets()[0])
assert [s["skill_id"] for s in cat["available_skills"][:4]] == [p_.skill_id for p_ in ST.builtin_strategy_presets()]
assert all(s["when_to_use"] and s["behavior_summary"] for s in cat["available_skills"])
print(f"[G5] LLM skill catalog: {len(cat['available_skills'])} skills")

# ------------------------------------------------------------------ G6 + G7 mocked Tk
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
mtk.ttk = mttk                       # `from tkinter import ttk` inside the console resolves to this
for nm in ("StringVar", "BooleanVar", "IntVar", "DoubleVar"):
    setattr(mtk, nm, FakeVar)
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
cam = CAM.CameraTabUI(mtk, mttk, mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), mock.MagicMock(),
                      lambda: {}, lambda *a: None, lambda *a: True)
base = {"T_base_cam": [[1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 1, 0.7], [0, 0, 0, 1]], "marker_in_flange": [0.0, 0.0, 0.03],
        "mean_mm": 1.0, "max_mm": 2.0, "loo_mean_mm": 1.5, "loo_max_mm": 3.0, "n_used": 12}
cam.result = dict(base)
assert "合格 ✓" in cam._calib_text().splitlines()[0]                         # V3.58 behaviour kept
cam.result = dict(base, warnings=["法兰朝向变化只有 20°（建议 ≥ 25°）"])
assert "留一精度已合格" in cam._calib_text().splitlines()[0]
cam.result = dict(base, good=True, warnings=["x"])
assert "合格 ✓" in cam._calib_text().splitlines()[0]
cam.result = dict(base, loo_mean_mm=9.0)
assert cam._calib_text().splitlines()[0].endswith("未达标 ⚠")
print("[G6] calibration badge: good flag / LOO-only / LOO ok + warnings")
CON.run_console("sim")
CON.run_console("real")
texts = [c.kwargs.get("text") for c in mttk.Notebook.return_value.add.call_args_list]
assert "技能参数旋钮" in texts and "Camera调试" in texts
print("[G7] mocked console builds the skill tab")
print("ALL V3.59 TESTS PASSED")
