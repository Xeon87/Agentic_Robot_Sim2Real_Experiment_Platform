#!/usr/bin/env python3
"""V3.60 offline unit tests (baseline-exact skill knobs, skill codes, read-only release, sim injection, sim tab, Y score):
F1  Versions V3.60 + markers; NERO_V308 untouched; sim wrist camera NERO_V326 = V3.58; USB wrist module NERO_V349 =
    V3.58 except VERSION; backend differs from V3.59 only inside one existing `# V3.43 A2` block (11 blocks kept).
F2  Zero-deviation equivalence with the backend's own code (extracted from NERO_V2_7_1_Unified_Backend.py by AST):
    grasp candidate score (V2.7 + V3.04), _v24_gripper_prior egg / sphere / box incl. retry bonus + opening + the
    NERO_V304_EGG_APERTURE_MARGIN_MM override, _v272_wrist_score / _v272_wrist_plan_ok, V3.04 numeric slot order,
    V3.41 limit-margin defaults; every baseline knob default equals the backend literal it claims.
F3  Sim env injection: the 6 wired variables + backend default strings; baseline EB injects no backend variable
    (shell exports kept); DP injects exactly its diffs; relaxed > required clamped; SimRunner passes them to the child.
F4  Real-robot V3.58 equivalence: default target order == V3.58 _selected_hover_target on 400 random tables;
    start-up never writes the hover field; explicit apply of QH writes 60 mm while wrist micro-align / crosshair /
    saved locate settings keep the user's value; user edits after that are the user's value again.
F5  Skills: built-ins EB / DP / HD / QH with unique 2-letter codes and short names; shipped JSON == built-in
    definitions; no unmeasured benchmark numbers; validate_new_skill rules; hover group counts for is_baseline.
F6  Release folder is read-only at run time: mocked console (sim + real) + every strategy-tab action (select, apply,
    reset, save valid / invalid, copy catalog, reload) leave every file of SHA256SUMS.txt and config/ unchanged;
    the active choice goes to logs/ (here: a temp file), user presets to the user dir.
F7  Sim GUI tab: add_sim_strategy_tab adds 「技能参数旋钮」 as its own notebook tab in sim mode; the backend A2 block calls it.
F8  Skill score Y = s/(t*z)*100: unit cases + an end-to-end NeroLLMAgent.run_task with a fake LLM (z = 3 rounds,
    s = 1/2 for "all eggs", s = 1/1 for a named egg), the [V3.60 SKILL SCORE] line and the jsonl record.
F9  Dry-run planner reports a skipped transit lift as a warning.
"""
import ast
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

TMP = Path(tempfile.mkdtemp(prefix="nero_v360_"))
os.environ["NERO_V360_USER_STRATEGY_DIR"] = str(TMP / "user_strategies")      # never ~/Downloads/cameracalib
os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"] = str(TMP / "logs" / "v360_active_strategy.json")
os.environ["NERO_CALIB_DIR"] = str(TMP / "cameracalib")
for k in ("NERO_V360_SKILL_CODE", "NERO_V360_SKILL_NAME", "NERO_V360_SKILL_ID", "NERO_V304_EGG_APERTURE_MARGIN_MM"):
    os.environ.pop(k, None)

H = lambda b: hashlib.sha256(b).hexdigest()


def release_snapshot():
    """sha256 of every file listed in SHA256SUMS.txt + the list of files under config/"""
    snap = {}
    for line in (ROOT / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rel = line.split(None, 1)[1].strip()
            fp = ROOT / rel
            snap[rel] = H(fp.read_bytes()) if fp.is_file() else None
    snap["__config_files__"] = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "config").rglob("*") if p.is_file())
    snap["__sums__"] = H((ROOT / "SHA256SUMS.txt").read_bytes())
    return snap


SNAP0 = release_snapshot()

import NERO_V359_Strategy as ST              # noqa: E402
import NERO_V312_Console as CON              # noqa: E402
import NERO_V313_Real_Robot as RR            # noqa: E402
import NERO_V325_Camera as CAM               # noqa: E402
import NERO_V347_Yolo_Perception as YP       # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49    # noqa: E402

CAM.CALIB_PATH = str(TMP / "camera_calib.json")
CAM.LOCATE_PATH = str(TMP / "locate_settings.json")
CAM.CALIB_DIR = str(TMP / "cameracalib")

BSRC = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text(encoding="utf-8")
BTREE = ast.parse(BSRC)

# ------------------------------------------------------------------ F1 versions / frozen files
for m in (CON, RR, YP, W49, ST):
    assert m.VERSION in ("V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3"), (m.__name__, m.VERSION)          # V3.61: version bump
assert (ROOT / "VERSION").read_text(encoding="utf-8").strip() in ("V3.60", "V3.61", "V3.62", "V3.63", "V3.64", "V3.65", "V3.66", "V3.67", "V3.68", "V3.69", "V3.70", "V3.71", "V3.72", "V3.73", "V3.74", "V3.75", "V3.76", "V3.77", "V3.78", "V3.79", "V3.80", "V3.81", "V3.82", "V3.83", "V3.84", "V3.85", "V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3")
assert H((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).startswith("020cc6680f181872")
assert H((ROOT / "NERO_V326_Wrist_Camera.py").read_bytes()) == \
    "6b5c5d683073e4252b82c1c703ad664173960a179947f9e72775bb9683697d0a", "sim wrist camera changed"
w49 = re.sub(r'VERSION = "V3\.\d+"', 'VERSION = "V3.58"', (ROOT / "NERO_V349_Real_Wrist_Camera.py").read_text(encoding="utf-8"), count=1)
assert H(w49.encode("utf-8")) == "d5955ce5c9c8acf8d654faa163da49e1b54ad21775a5ec060ac353a13c76a622", "USB wrist module changed"
A2 = re.compile(r"\n[ \t]*# V3\.43 A2 BEGIN\n.*?\n[ \t]*# V3\.43 A2 END(?=\n)", re.S)
assert BSRC.count("# V3.43 A2 BEGIN") == BSRC.count("# V3.43 A2 END") == 11
assert H(A2.sub("", BSRC).encode()) == "5e83581fb69bd7d8afd8278b26a2ebe72224f67e73b1b83abe12a533b2700405"
blocks360 = [b for b in A2.findall(BSRC) if "V3.60" in b]
assert len(blocks360) == 1 and "_ST360.add_sim_strategy_tab(tk, ttk, notebook, log=append_gui_log)" in blocks360[0]
assert BSRC.count("V3.60") == 1 + blocks360[0].count("V3.60") - 1 and BSRC.count("add_sim_strategy_tab") == 1
for fn, mk in (("NERO_V359_Strategy.py", "V3.60 BASELINE-EXACT DEFAULTS + SKILL CODES"),
               ("NERO_V325_Camera.py", "V3.60 V3.58-EXACT HOVER DEFAULTS + WRIST HOVER INDEPENDENT OF SKILLS"),
               ("NERO_V312_Console.py", "V3.60 SIM ENV INJECTION OF THE APPLIED SKILL"),
               ("NERO_V2_7_1_Agent.py", "V3.60 SKILL SCORE")):
    assert mk in (ROOT / fn).read_text(encoding="utf-8"), (fn, mk)
print("[F1] versions V3.60; V308 / V326 / V349 frozen; backend changed only inside one existing A2 block")


# ------------------------------------------------------------------ F2 zero-deviation equivalence with the backend
def bfunc(name, nested=False):
    for n in ast.walk(BTREE):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise KeyError(name)


def bassign(name):
    for n in ast.walk(BTREE):
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets):
            return n
    raise KeyError(name)


rng = random.Random(360)
# (a) grasp candidate score: the two statements inside _v27_make_grasp_candidates
mk = bfunc("_v27_make_grasp_candidates")
score_stmt = next(n for n in ast.walk(mk) if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "score")
nb_if = next(n for n in ast.walk(mk) if isinstance(n, ast.If) and "nb_clear" in ast.unparse(n.test))
assert ast.unparse(score_stmt) == "score = clearance * 1000.0 - yaw_rank * 4.0 - off * 1000.0 * 0.6"
assert ast.unparse(nb_if.test) == "nb_clear < 3.0"
code_score = compile(ast.Module([score_stmt, nb_if], []), "backend_score", "exec")
for _ in range(3000):
    c, yr, off = rng.uniform(-0.01, 0.03), rng.randint(0, 6), rng.choice([0.0, 0.004, 0.006, rng.uniform(0, 0.01)])
    nb = rng.choice([999.0, rng.uniform(-20, 12), 3.0, 2.9999])
    ns = {"clearance": c, "yaw_rank": yr, "off": off, "nb_clear": nb}
    exec(code_score, ns)
    assert ns["score"] == ST.score_grasp_candidate(c, yr, off, nb), (c, yr, off, nb)
# (b) wall-near / horizontal-strength constants the knobs claim
assert "wall_near = (V27_BASKET_SAFE_INNER_RADIUS_M - obj_radial_r) < 0.030" in BSRC
assert ST.BASELINE_KNOBS["wall_near_margin_mm"] == 30.0 and ST.BASELINE_KNOBS["horiz_proj_min_strength"] == 0.25
assert 'axis["horizontal_projection_strength"] >= 0.25' in BSRC
# (c) _v24_gripper_prior (base V2-7 + V3.04 egg override), run from the backend source
priors = [n for n in ast.walk(BTREE) if isinstance(n, ast.FunctionDef) and n.name == "_v24_gripper_prior"]
assert len(priors) == 2
alias = bassign("_v218_base_gripper_prior")
geom = ast.literal_eval(bassign("V24_OBJECT_GEOMETRY").value)
geom["ball_x"] = {"category": "ball", "shape": "sphere", "effective_width_mm": 52.0}
geom["cube_x"] = {"category": "cube", "shape": "box", "effective_width_mm": 45.0}
bonus = {k: 0.0 for k in geom}


def backend_prior(name, opening, retry, env_margin=None):
    os.environ.pop("NERO_V304_EGG_APERTURE_MARGIN_MM", None)
    if env_margin is not None:
        os.environ["NERO_V304_EGG_APERTURE_MARGIN_MM"] = str(env_margin)
    bonus[name] = retry
    g = {"os": os, "V24_OBJECT_GEOMETRY": geom, "V241_APERTURE_RETRY_BONUS_MM": bonus}
    exec(compile(ast.Module([priors[0]], []), "b", "exec"), g)
    exec(compile(ast.Module([alias], []), "b", "exec"), g)
    exec(compile(ast.Module([priors[1]], []), "b", "exec"), g)
    try:
        return g["_v24_gripper_prior"](name, opening)
    finally:
        os.environ.pop("NERO_V304_EGG_APERTURE_MARGIN_MM", None)
        bonus[name] = 0.0


META = {"category", "shape", "material_class", "fragile_model_enabled", "approx_size_mm"}
n_cmp = 0
for name, gm in geom.items():
    for opening in (None, 0.0, 58.3, 61.0, 88.0):
        for retry in (0.0, 8.0, 15.0, 23.0, 30.0):
            for env_m in (None, 12.5, 16, 20.0):
                if env_m is not None and not name.startswith("egg"):
                    continue
                b = backend_prior(name, opening, retry, env_m)
                knobs = dict(ST.BASELINE_KNOBS)
                if env_m is not None:
                    knobs["egg_open_margin_mm"] = float(env_m)
                sres = ST.compute_gripper_prior(name, gm["effective_width_mm"], gm["shape"], retry, opening, knobs=knobs)
                assert set(b) - set(sres) == META and set(sres) <= set(b), (set(b) ^ set(sres))
                for k in sres:
                    assert sres[k] == b[k], (name, opening, retry, env_m, k, sres[k], b[k])
                n_cmp += 1
# (d) wrist reorientation score / bound
g = {}
for nm in ("V272_JOINT_SCORE_WEIGHT", "V272_JOINT_STEP_LIMIT_DEG"):
    exec(compile(ast.Module([bassign(nm)], []), "b", "exec"), g)
for nm in ("_v272_wrist_score", "_v272_wrist_plan_ok"):
    exec(compile(ast.Module([bfunc(nm)], []), "b", "exec"), g)
for _ in range(3000):
    d = {f"joint{i}": rng.choice([0.0, rng.uniform(0, 6), rng.uniform(0, 30)]) for i in range(1, 8)}
    if rng.random() < 0.2:
        d.pop(f"joint{rng.randint(1, 7)}")
    assert g["_v272_wrist_score"](d) == ST.compute_wrist_reorient_score(d)
    assert g["_v272_wrist_plan_ok"](d) == ST.check_wrist_reorient_bounded(d, step_limits_deg=g["V272_JOINT_STEP_LIMIT_DEG"])
assert g["_v272_wrist_plan_ok"]({}) == ST.check_wrist_reorient_bounded({}) is False
# (e) V3.04 numeric slot order (observe_scene / v306_next): sorted(..., key=int(name.split('_')[-1]))
assert "key=lambda k: int(str(k).split(\"_\")[-1]))" in BSRC
for _ in range(300):
    slots = {f"slot_{i}": {"occupied": rng.random() < 0.4} for i in rng.sample(range(1, 9), 8)}
    exp = sorted([k for k, v in slots.items() if not v["occupied"]], key=lambda k: int(str(k).split("_")[-1]))
    assert ST.order_empty_slots(slots) == exp
# (f) constants the knobs claim (backend literal == knob default)
CLAIMS = {"pregrasp_hover_height_mm": ("V27_PREGRASP_HEIGHT_M", 1000.0), "grasp_z_offset_mm": ("V27_DEFAULT_GRASP_Z_OFFSET_M", 1000.0),
          "object_stale_pos_mm": ("V27_OBJECT_STALE_POS_M", 1000.0), "slot_entry_above_release_mm": ("V271_ENTRY_ABOVE_RELEASE_M", 1000.0),
          "slot_capture_depth_mm": ("V271_CAPTURE_DEPTH_M", 1000.0), "slot_xy_tol_mm": ("V271_SLOT_XY_TOL_M", 1000.0),
          "slot_axis_tol_deg": ("V271_AXIS_TOL_DEG", 1.0), "release_stage1_open_mm": ("V271_RELEASE_STAGE_OPEN_MM", 1.0),
          "bystander_warn_mm": ("V271_BYSTANDER_MOVE_TOL_M", 1000.0), "bystander_abort_mm": ("V216_BYSTANDER_ABORT_TOL_M", 1000.0)}
for knob, (const, scale) in CLAIMS.items():
    v = ast.literal_eval(bassign(const).value)
    assert abs(v * scale - ST.BASELINE_KNOBS[knob]) < 1e-9, (knob, const, v)
assert 'os.environ.get("NERO_V304_GRASP_Z_COMP_M", "0.0025")' in BSRC
assert 'os.environ.get("NERO_V341_LIMIT_MARGIN_DEG", "8")' in BSRC and 'os.environ.get("NERO_V341_LIMIT_MARGIN_MIN_DEG", "4")' in BSRC
assert "int(min(marg341[id(r)], 20.0) // 4)" in BSRC and ST.BASELINE_KNOBS["limit_margin_bucket_deg"] == 4.0
assert CAM.HOVER_DEFAULT_MM == 100.0 == ST.BASELINE_KNOBS["cam_hover_verify_height_mm"]
assert CAM.CALIB_MAX_SPEED == 0.10 and ST.BASELINE_KNOBS["cam_hover_max_speed_pct"] == 10.0
assert ST.BASELINE_KNOBS["target_order_mode"] == "table_trusted_first" and ST.BASELINE_KNOBS["slot_fill_order"] == "numeric_1_to_8"
print(f"[F2] zero deviation vs backend source: 3000 scores, {n_cmp} gripper priors, 3000 wrist scores/bounds, "
      f"300 slot orders, {len(CLAIMS)} constants")

# ------------------------------------------------------------------ F3 sim env injection
for knob, (var, dflt, scale) in ST.SIM_ENV_KNOBS.items():
    assert f'os.environ.get("{var}", "{dflt}")' in BSRC, var
    assert abs(float(dflt) - ST.BASELINE_KNOBS[knob] * scale) < 1e-12, knob
    assert ST.KNOB_BY_KEY[knob]["wired"] == "sim_env"
assert len(ST.SIM_ENV_KNOBS) == 6
presets = {p.code: p for _sid, _n, _fp, p in ST.list_strategy_files()}
eb, dp, hd, qh = presets["EB"], presets["DP"], presets["HD"], presets["QH"]
env, notes = ST.strategy_sim_env(eb, base_env={})
assert env == {"NERO_V360_SKILL_ID": "egg_grasp_v345_baseline", "NERO_V360_SKILL_CODE": "EB",
               "NERO_V360_SKILL_NAME": "Egg Baseline"} and notes == []
env, notes = ST.strategy_sim_env(eb, base_env={"NERO_V304_EGG_APERTURE_MARGIN_MM": "18"})
assert "NERO_V304_EGG_APERTURE_MARGIN_MM" not in env and any("来自终端环境变量" in n for n in notes)
env, notes = ST.strategy_sim_env(dp, base_env={"NERO_V304_FINAL_APPROACH_M": "0.03"})
assert {k: v for k, v in env.items() if not k.startswith("NERO_V360_")} == {
    "NERO_V304_EGG_APERTURE_MARGIN_MM": "12.5", "NERO_V304_FINAL_APPROACH_M": "0.028", "NERO_V304_FINAL_APPROACH_SLOW": "4.5"}
assert env["NERO_V360_SKILL_CODE"] == "DP" and any("被技能 DP" in n for n in notes)
for p_ in (hd, qh):                      # real-only skills never touch the sim backend
    assert not [k for k in ST.strategy_sim_env(p_, base_env={})[0] if not k.startswith("NERO_V360_")]
odd = ST.StrategyProfile(skill_id="odd", knobs={"limit_margin_required_deg": 6.0, "limit_margin_relaxed_deg": 7.0},
                         short_name="Odd", code="OD")
env, notes = ST.strategy_sim_env(odd, base_env={})
assert env["NERO_V341_LIMIT_MARGIN_DEG"] == "6" and env["NERO_V341_LIMIT_MARGIN_MIN_DEG"] == "6" and any("放宽" in n for n in notes)
script = TMP / "fake_sim.sh"
script.write_text('echo "M=$NERO_V304_EGG_APERTURE_MARGIN_MM S=$NERO_V304_FINAL_APPROACH_SLOW C=$NERO_V360_SKILL_CODE N=$NERO_SCENE"\n')


def run_fake(strategy_env):
    r = CON.SimRunner(str(script))
    r.strategy_env = strategy_env
    r.start(extra_env=CON.scene_env("fill_egg_holder"))
    out = []
    t_end = time.time() + 15
    while time.time() < t_end:
        ln = r.lines.get(timeout=15)
        if ln is None:
            break
        out.append(ln)
    return out


assert any("M=12.5 S=4.5 C=DP N=fill_egg_holder" in ln for ln in run_fake(ST.strategy_sim_env(dp, base_env={})[0]))
assert any("M= S= C=EB N=fill_egg_holder" in ln for ln in run_fake(ST.strategy_sim_env(eb, base_env={})[0]))
assert any("M= S= C= N=fill_egg_holder" in ln for ln in run_fake({}))
CSRC = (ROOT / "NERO_V312_Console.py").read_text(encoding="utf-8")
assert "pid = runner.start(extra_env=scene_env(scene))" in CSRC and "ST59.strategy_sim_env(prof360, base_env=clean_child_env())" in CSRC
print("[F3] 6 wired env vars = backend defaults; EB injects nothing; DP injects exactly its 3 diffs; SimRunner passes them")

# ------------------------------------------------------------------ F4 + F6 + F7: mocked Tk
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


mock_tk, mock_ttk = mock.MagicMock(), mock.MagicMock()
for nm in ("StringVar", "BooleanVar", "IntVar", "DoubleVar"):
    setattr(mock_tk, nm, FakeVar)
mock_tk.ttk = mock_ttk
mock_tk.Text.return_value.yview.return_value = (0.0, 1.0)
mock_tk.Text.return_value.index.return_value = "1.0"
sys.modules["tkinter"] = mock_tk
sys.modules["tkinter.ttk"] = mock_ttk
for sub in ("font", "messagebox", "simpledialog", "filedialog"):
    sys.modules[f"tkinter.{sub}"] = mock.MagicMock()
root_mock = mock.MagicMock()
root_mock.winfo_screenwidth.return_value = 1920
root_mock.winfo_screenheight.return_value = 1080
mock_tk.Tk.return_value = root_mock
root_mock.mainloop.side_effect = lambda: None
cams, tabs = [], []
_C, _T = CAM.CameraTabUI, ST.SkillStrategyTabUI


class RecCam(_C):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        cams.append(self)


class RecTab(_T):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        tabs.append(self)


CAM.CameraTabUI, ST.SkillStrategyTabUI = RecCam, RecTab
CON.run_console("sim")
CON.run_console("real")
CAM.CameraTabUI, ST.SkillStrategyTabUI = _C, _T
assert len(cams) == 2 and len(tabs) == 2
nb_texts = [c.kwargs.get("text") for c in mock_ttk.Notebook.return_value.add.call_args_list]
assert "技能参数旋钮" in nb_texts
cam, tab = cams[-1], tabs[-1]
assert tab.mode == "real" and tab.cam_tab is cam and tab.profile.code == "EB"
assert cam.v_hover_mm.get() == "100" and cam._hover_user_str() == "100"
assert not Path(os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"]).exists()        # start-up writes nothing

# F4a: V3.58 target selection == default
cam.tv_eggs = mock.MagicMock()
cam.tv_eggs.selection.return_value = ()


def v358_pick(eggs):
    valid = [e for e in eggs if e.get("xyz_m") and e.get("status") != "rejected"]
    if not valid:
        return None
    trusted = [e for e in valid if e.get("trusted")]
    return trusted[0] if trusted else valid[0]


for _ in range(400):
    eggs = [{"det_id": f"e{i}", "xyz_m": (None if rng.random() < 0.1 else [rng.uniform(-.6, .6), rng.uniform(-.4, .4), .03]),
             "trusted": rng.random() < 0.5, "confidence": rng.random(),
             "status": rng.choice(["ok", "ok", "rejected", "low"])} for i in range(rng.randint(0, 6))]
    cam.eggs = eggs
    assert cam._selected_hover_target() is v358_pick(eggs)
cam.eggs = [{"det_id": "near", "xyz_m": [0.05, 0, .03], "trusted": True, "confidence": 0.9, "status": "ok"},
            {"det_id": "first", "xyz_m": [0.5, 0, .03], "trusted": True, "confidence": 0.5, "status": "ok"}]
cam.eggs.reverse()
assert cam._selected_hover_target()["det_id"] == "first"
cam.apply_strategy_profile(qh, set_hover=False)
assert cam._selected_hover_target()["det_id"] == "near" and cam.v_hover_mm.get() == "100"    # QH order, no hover write

# F4b: hover field vs wrist independence
cam.apply_strategy_profile(eb, set_hover=False)
cam.v_hover_mm.set("85")                                     # the user's own value
tab.v_preset.set(next(lab for lab, (sid, _f, _p) in tab._preset_map.items() if sid == "quick_hover_verify"))
tab.on_select_preset()
assert tab.profile.code == "QH" and cam.strategy_profile.code == "QH"
assert cam.v_hover_mm.get() == "60" and cam._hover_user_str() == "85"
assert cam._locate_settings()["hover_mm"] == "85"
assert Path(os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"]).is_file()
assert ST.read_active_strategy_file().code == "QH"
cam.v_hover_mm.set("70")                                     # user edits after the skill wrote the field
assert cam._hover_user_str() == "70" and cam._hover_strategy_str is None
tab.on_reset_baseline()
assert tab.profile.code == "EB" and cam.v_hover_mm.get() == "100" and cam._hover_user_str() == "70"
cam._locate_apply({"hover_mm": "90"}, "test")                # loading saved settings = user's value
assert cam.v_hover_mm.get() == "90" and cam._hover_user_str() == "90"
CAMSRC = (ROOT / "NERO_V325_Camera.py").read_text(encoding="utf-8")
assert "z_work = max(0.06, float(self._hover_user_str()) / 1000.0)" in CAMSRC
i_align = CAMSRC.index("def on_wrist_align(self):")
assert "hover_mm = float(self._hover_user_str())" in CAMSRC[i_align:i_align + 1500]
i_hv = CAMSRC.index("def on_hover_target(self):")
assert "hover_mm = float(self.v_hover_mm.get())" in CAMSRC[i_hv:i_hv + 1500]
print("[F4] default target order == V3.58 on 400 random tables; start-up never writes hover; QH writes 60 mm for hover "
      "verify while wrist align / crosshair / saved settings keep the user's value")

# F6: every strategy action keeps the release folder read-only
tab.v_skill_id.set("egg_grasp_v345_baseline")
assert tab.on_save() is None                                  # built-in id refused
tab.v_skill_id.set("my_dense")
tab.v_short_name.set("")
tab.v_code.set("MD")
assert tab.on_save() is None                                  # short name missing
tab.v_short_name.set("My Dense")
tab.v_code.set("DP")
assert tab.on_save() is None                                  # duplicate code
tab.v_code.set("m1")
assert tab.on_save() is None                                  # invalid code
tab.v_code.set("MD")
tab.knob_vars["egg_open_margin_mm"].set("13")
saved = tab.on_save()
assert saved == str(TMP / "user_strategies" / "my_dense.json") and Path(saved).is_file()
assert ST.read_active_strategy_file().code == "MD"
assert [p.code for _s, _n, _f, p in ST.list_strategy_files()] == ["EB", "DP", "HD", "QH"] + (["CS"] if ST.VERSION >= "V3.61" else []) + ["MD"]
try:
    ST.save_strategy_file(tab.profile, path=str(ROOT / "config" / "strategies" / "x.json"))
    raise AssertionError("write into the release folder accepted")
except ValueError:
    pass
tab.on_apply()
cat = tab.on_copy_llm_catalog()
assert cat["active_skill_code"] == "MD" and [s_["code"] for s_ in cat["available_skills"]] == ["EB", "DP", "HD", "QH"] + (["CS"] if ST.VERSION >= "V3.61" else []) + ["MD"]
assert all(s_["sim_benchmark"].get("success_rate") is None for s_ in cat["available_skills"])
# the other window applies a skill -> reload when the tab is opened (no hover write)
cam.v_hover_mm.set("90")
ST.set_active_strategy(dp)
os.utime(os.environ["NERO_V360_ACTIVE_STRATEGY_FILE"], (time.time() + 5, time.time() + 5))
assert tab.reload_active_if_changed() and tab.profile.code == "DP" and cam.v_hover_mm.get() == "90"
assert not tab.reload_active_if_changed()
tab.on_reset_baseline()
SNAP1 = release_snapshot()
changed = [k for k in SNAP0 if SNAP0[k] != SNAP1.get(k)]
assert not changed, changed
print("[F6] select / apply / save (4 refusals + 1 user preset) / catalog / reload / reset: release folder unchanged")

# F7: sim GUI tab
nb_mock = mock.MagicMock()
logs7 = []
os.environ["NERO_V360_SKILL_CODE"], os.environ["NERO_V360_SKILL_NAME"] = "DP", "Dense Anti-Poke"
ui7 = ST.add_sim_strategy_tab(mock_tk, mock_ttk, nb_mock, log=logs7.append)
os.environ.pop("NERO_V360_SKILL_CODE")
os.environ.pop("NERO_V360_SKILL_NAME")
assert ui7.mode == "sim" and ui7.cam_tab is None
assert nb_mock.add.call_args.kwargs.get("text") == "技能参数旋钮"
assert any("DP Dense Anti-Poke" in ln for ln in logs7)
rg = bfunc("run_gui")
call_ln = next(n.lineno for n in ast.walk(rg) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add_sim_strategy_tab")
first_def = {}
for n in ast.walk(rg):
    if isinstance(n, ast.Import) and any(a.asname == "tk" for a in n.names):
        first_def.setdefault("tk", n.lineno)
    if isinstance(n, ast.ImportFrom) and n.module == "tkinter" and any(a.name == "ttk" for a in n.names):
        first_def.setdefault("ttk", n.lineno)
    if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "notebook":
        first_def.setdefault("notebook", n.lineno)
    if isinstance(n, ast.FunctionDef) and n.name == "append_gui_log":
        first_def.setdefault("append_gui_log", n.lineno)
assert set(first_def) == {"tk", "ttk", "notebook", "append_gui_log"} and max(first_def.values()) < call_ln, (first_def, call_ln)
llm_add = next(n.lineno for n in ast.walk(rg) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "add"
               and any(getattr(k_.value, "value", None) == "NERO SimAgent" for k_ in n.keywords))
assert llm_add < call_ln                                   # the skill tab comes right after NERO SimAgent
STSRC = (ROOT / "NERO_V359_Strategy.py").read_text(encoding="utf-8")
assert ('if self.mode != "sim":            # V3.60 SIM TAB: the dry-run drives the real arm -> console only\n            dry_lf.pack(' in STSRC
        or 'show=(self.mode != "sim"))' in STSRC)          # V3.61: the dry-run is a collapsible section, not shown in sim
assert release_snapshot() == SNAP1
print("[F7] sim GUI: own 「技能参数旋钮」 tab (sim mode, no dry-run panel), called from the backend A2 block")

# ------------------------------------------------------------------ F5 skills
codes = [p.code for p in ST.builtin_strategy_presets()]
assert codes[:4] == ["EB", "DP", "HD", "QH"] and len(set(codes)) == len(codes) and all(ST.SKILL_CODE_RE.match(c) for c in codes)
assert [p.short_name for p in ST.builtin_strategy_presets()][:4] == ["Egg Baseline", "Dense Anti-Poke", "Hover Dry-Run", "Quick Hover"]
for p in ST.builtin_strategy_presets():
    fp = ROOT / "config" / "strategies" / f"{p.skill_id}.json"
    d = json.loads(fp.read_text(encoding="utf-8"))
    exp = p.to_dict()
    d.pop("saved"), exp.pop("saved")
    assert d == exp, p.skill_id
    assert d["skill_card"]["sim_benchmark"]["success_rate"] is None and d["skill_card"]["sim_benchmark"]["avg_cycle_time_s"] is None
    assert "SO-101" not in json.dumps(d, ensure_ascii=False)
    assert (fp.stat().st_mode & 0o444) == 0o444, fp
dflt = json.loads((ROOT / "config" / "strategies" / "default.json").read_text(encoding="utf-8"))
assert dflt == json.loads((ROOT / "config" / "strategies" / "egg_grasp_v345_baseline.json").read_text(encoding="utf-8"))
assert ST.is_baseline_equivalent(eb.knobs) and not ST.is_baseline_equivalent(hd.knobs) and not ST.is_baseline_equivalent(qh.knobs)
assert qh.knobs["cam_hover_verify_height_mm"] == 60.0 and qh.knobs["target_order_mode"] == "confidence_then_nearest"
assert {k for k, *_ in ST.diff_vs_baseline(qh.knobs)} == {"cam_hover_verify_height_mm", "target_order_mode"}
assert ST.KNOB_BY_KEY["poke_retry_bonus_mm"]["wired"] == ST.KNOB_BY_KEY["safety_retry_bonus_mm"]["wired"] == "none"
assert sum(1 for s_ in ST.KNOB_SPECS if s_["wired"] == "sim_env") == 6
assert len([s_ for s_ in ST.KNOB_SPECS if s_["group"] != "cup_stacking"]) == 45      # V3.61 adds the cup group
assert ST.validate_new_skill(ST.StrategyProfile(skill_id="bad id", short_name="X", code="XX"), existing=[]) is not None
assert ST.validate_new_skill(ST.StrategyProfile(skill_id="ok_id", short_name="X", code="XX"), existing=[]) is None
assert ST.StrategyProfile(skill_id="x", code="abc").code == "" and ST.StrategyProfile(skill_id="x", code="ab").code == "AB"
print("[F5] built-in skills EB/DP/HD/QH (codes unique), shipped JSON == definitions, no unmeasured numbers, 0644")


# ------------------------------------------------------------------ F8 skill score Y
ASRC = (ROOT / "NERO_V2_7_1_Agent.py").read_text(encoding="utf-8")
atree = ast.parse(ASRC)
ga = {"os": os, "json": json, "time": time}
for n in atree.body:
    if isinstance(n, ast.FunctionDef) and n.name in ("_v360_skill_score", "_v360_append_score"):
        exec(compile(ast.Module([n], []), "agent", "exec"), ga)
score = ga["_v360_skill_score"]


def obs(locs):
    return {"tool": "observe_scene", "result": {"ok": True, "scene": {"objects": {
        n: {"category": "egg", "location": l} for n, l in locs.items()}}}}


def ver(n, ok=True):
    return {"tool": "verify_object_in_slot", "result": {"ok": ok, "object_name": n}}


hist = [obs({"egg_1": "egg_basket", "egg_2": "egg_basket", "egg_3": "egg_basket", "egg_4": "egg_basket"}),
        ver("egg_1"), ver("egg_2"), ver("egg_3"), ver("egg_4", ok=False)]
r = score(hist, 240.0, 30, requested=["egg_1", "egg_2", "egg_3", "egg_4"], status="FAILED")
assert (r["s"], r["done"], r["total"], r["t_min"], r["z"]) == (0.75, 3, 4, 4.0, 30)
assert abs(r["Y"] - 0.75 / (4.0 * 30) * 100) < 1e-4 and "Y = s/(t*z)*100 = 0.625" in r["line"]
assert r["skill_code"] == "EB" and "not injected" in r["line"]
r = score(hist[:1], 60.0, 5, status="FAILED")
assert r["s"] == 0.0 and r["Y"] == 0.0
r = score(hist, 0.0, 0, status="FAILED")
assert r["Y"] == 0.0 and "t*z = 0" in r["line"]
hist2 = [obs({"egg_1": "egg_holder.slot_1", "egg_2": "egg_basket", "egg_3": "egg_basket"}), ver("egg_2"), ver("egg_3")]
assert score(hist2, 120.0, 10)["s"] == 1.0 and score(hist2, 120.0, 10)["total"] == 2   # egg already in the holder excluded
assert score(hist2, 120.0, 10, named=["egg_2"])["total"] == 1
r = score([], 30.0, 2, requested=["<one_object>"], status="SUCCESS")
assert r["s"] == 1.0 and r["basis"].startswith("task status")
os.environ["NERO_V360_SKILL_CODE"], os.environ["NERO_V360_SKILL_NAME"] = "DP", "Dense Anti-Poke"
assert score(hist, 240.0, 30)["line"].startswith("[V3.60 SKILL SCORE] skill=DP Dense Anti-Poke | s=0.75")
os.environ.pop("NERO_V360_SKILL_CODE")
os.environ.pop("NERO_V360_SKILL_NAME")

# end-to-end: NeroLLMAgent.run_task with a fake LLM (3 round trips) and a fake backend
import NERO_V2_7_1_Agent as AG  # noqa: E402
AG._v360_append_score = lambda *a, **k: None          # do not write ROOT/logs from the test
os.environ["NERO_V342_FAILURE_ANALYSIS"] = "0"


def e2e(task, verify_names):
    lines, calls = [], []
    seq = [{"id": "r1", "output": [{"type": "function_call", "name": "observe_scene", "arguments": "{}", "call_id": "c1"}]}]
    for i, n in enumerate(verify_names):
        seq.append({"id": f"r{i + 2}", "output": [{"type": "function_call", "name": "verify_object_in_slot",
                                                    "arguments": json.dumps({"object_name": n, "slot_name": "slot_1"}),
                                                    "call_id": f"c{i + 2}"}]})
    seq.append({"id": "rf", "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}]})

    def runner(name, args):
        calls.append(name)
        if name in ("observe_scene", "scene_snapshot"):
            return {"ok": True, "scene": {"objects": {"egg_1": {"category": "egg", "location": "egg_basket"},
                                                      "egg_2": {"category": "egg", "location": "egg_basket"}}},
                    "objects": {"egg_1": {"location": "egg_holder.slot_1"}, "egg_2": {"location": "egg_holder.slot_2"}}}
        if name == "verify_object_in_slot":
            return {"ok": True, "object_name": args.get("object_name"), "physical_verification": {}}
        return {"ok": True}
    ag = AG.NeroLLMAgent(lines.append, runner)
    it = iter(seq)
    ag._post_responses = lambda payload, key, timeout_s=180.0: next(it, seq[-1])   # guard nudges -> final again
    ag.run_task(task, model="gpt-test", api_key="sk-test", max_tool_steps=20)
    return [ln for ln in lines if str(ln).startswith("[V3.60 SKILL SCORE]")], lines


sc, lines = e2e("put egg_1 into the egg holder", ["egg_1"])
assert len(sc) == 1 and "z=3 LLM rounds" in sc[0] and "s=1.00 (1/1 named eggs)" in sc[0], (sc, lines[-12:])
i_sum = next(i for i, ln in enumerate(lines) if str(ln).startswith("[TASK SUMMARY"))
i_st = next(i for i, ln in enumerate(lines) if str(ln).startswith("[V3.42 TASK STATS]"))
assert i_sum < i_st < lines.index(sc[0])
sc, lines = e2e("put all eggs into the egg holder one by one", ["egg_1"])
assert len(sc) == 1 and "s=0.50 (1/2 eggs)" in sc[0] and re.search(r"z=\d+ LLM rounds", sc[0]), sc
n_req = sum(1 for ln in lines if str(ln).startswith("[LLM RESPONSE #"))
assert f"z={n_req} LLM rounds" in sc[0], (n_req, sc)
os.environ.pop("NERO_V342_FAILURE_ANALYSIS")
print(f"[F8] Y = s/(t*z)*100 unit cases + end-to-end run_task (z counted from LLM responses): {sc[0][:120]}")

# ------------------------------------------------------------------ F9 dry-run planner warning
calls9 = []


def fake_hover(urdf, joints, lim, q0, tool_T, xyz, base="base_link", fence_fn=None, soft_deg=3.0):
    calls9.append(list(xyz))
    if len(calls9) in (2, 3):                       # both transit attempts fail
        return None, "IK 无解"
    return [0.0] * 7, {"err_mm": 0.1}


plan = ST.plan_hover_dry_run_sequence(None, [f"joint{i}" for i in range(1, 8)], [(-3, 3)] * 7, [0.0] * 7, None,
                                      [{"det_id": "A", "xyz_m": [0.3, 0.0, 0.03]}, {"det_id": "B", "xyz_m": [0.3, 0.2, 0.03]}],
                                      plan_hover_fn=fake_hover)
assert plan["ok"] and len(plan["warnings"]) == 1 and "跨点抬升" in plan["warnings"][0]
assert plan["hover_height_mm"] == 100.0 and [s_["kind"] for s_ in plan["steps"]] == ["hover", "hover", "return_home"]
assert release_snapshot() == SNAP1
print("[F9] dry-run: skipped transit lift reported as a warning; default hover 100 mm")

print("ALL V3.60 TESTS PASSED")
