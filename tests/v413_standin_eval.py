#!/usr/bin/env python3
"""NERO V4.13 real stand-in evaluation: the REAL general pick & place path (RealIO + perceive_general + the real
height map + AgxGripper + H1 V1) on MuJoCo twins (tests/v413_real_standin.py), driven by the test stand-in reasoner
(tests/v49_reasoner.py - NOT an LLM).  Success from the simulator truth.

    python3 tests/v413_standin_eval.py egg egg2 H1 H2 H3 H4 H5 seed:1-10 [--calib 3,0.3] [--miss egg_1] [out.json]

egg / egg2 = egg_1 -> cup_1 / egg_2 -> cup_2 on the real-finger layout; H* / seed:* = the household scenes.
--calib mm,deg = a hand-eye error on T_base_cam; --miss id = the detector never reports that object.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTS = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("NERO_CALIB_DIR", tempfile.mkdtemp(prefix="nero_v413_eval_"))
os.environ.setdefault("MUJOCO_GL", "osmesa")
if os.environ["MUJOCO_GL"] in ("osmesa", "egl"):
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])


def _prompts(labels):
    return sorted({str(l).replace("_", " ") for l in labels})


def episode(spec, calib=(0.0, 0.0), miss=(), verbose=False):
    import numpy as np
    import NERO_V49_Geometry as G49
    import v413_real_standin as SB
    from v49_reasoner import GeometricReasoner
    t0 = time.time()
    kw = dict(calib_err_mm=calib[0], calib_err_deg=calib[1], miss=miss)
    if spec in ("egg", "egg2"):
        import NERO_V49_Random_Scenes as RS
        src, tgt = ("egg_1", "cup_1") if spec == "egg" else ("egg_2", "cup_2")
        sb, sc = SB.egg_cup_scene(source=src, target=tgt, **kw)
        goals = [(src, "into", tgt)]
        labels = ["egg", "cup"]
    else:
        import NERO_V410_Household_Scenes as HS
        sb, sc = SB.household(spec, **kw)
        goals = list(sc.goals)
        labels = [G49.label_of(i.oid) for i in sc.items]
    sk = SB.make_skills(sb, "V1")
    rs = GeometricReasoner(sk, log=print if verbose else (lambda *_: None), prompts=_prompts(labels))
    outs = []
    for s, rel, t in goals:
        try:
            out = rs.run(s, t, "on" if rel in ("onto", "on") else "into")
        except Exception as e:
            out = {"ok": False, "why": f"exception {e!r}", "trace": traceback.format_exc()[-600:]}
        outs.append({k: v for k, v in out.items() if k != "relations"})
        sb.advance(0.8)
    sb.advance(1.5)
    if spec in ("egg", "egg2"):
        import NERO_V49_Random_Scenes as RS
        ok, det = RS.check_success(sb.tw, sc)
        st = [{"goal": f"{goals[0][0]} into {goals[0][2]}", "ok": bool(ok), "detail": det}]
    else:
        import NERO_V410_Household_Scenes as HS
        st = HS.goal_status(sb.tw.m, sb.tw.d, sc, held=HS.held_bodies(sb.tw.m, sb.tw.d))
    n_ok = sum(g["ok"] for g in st)
    status = "success" if n_ok == len(st) else ("partial" if n_ok else "fail")
    fails = [(n, r) for n, _a, ok, r in rs.calls if not ok][-6:]
    exc = [o["why"] for o in outs if str(o.get("why", "")).startswith("exception")]
    return {"scene": spec, "status": status, "goals": st, "reasoner": outs, "exceptions": exc, "tool_calls": len(rs.calls),
            "failed_calls": fails, "arm_paths": len(sb.arm.paths), "seconds": round(time.time() - t0, 1),
            "calib": list(calib), "miss": list(miss)}


def _specs(args):
    out = []
    for a in args:
        if a.startswith("seed:") and "-" in a:
            lo, hi = a[5:].split("-")
            out += [f"seed:{i}" for i in range(int(lo), int(hi) + 1)]
        else:
            out.append(a)
    return out


def main(argv):
    calib, miss, out_path, specs = (0.0, 0.0), [], None, []
    it = iter(argv)
    for a in it:
        if a == "--calib":
            v = next(it).split(",")
            calib = (float(v[0]), float(v[1]))
        elif a == "--miss":
            miss.append(next(it))
        elif a.endswith(".json"):
            out_path = a
        else:
            specs.append(a)
    res = []
    for sp in _specs(specs or ["egg", "H1", "H2"]):
        try:
            r = episode(sp, calib, miss)
        except Exception as e:
            r = {"scene": sp, "status": "error", "error": repr(e), "trace": traceback.format_exc()[-800:]}
        res.append(r)
        g = " | ".join(f"{x['goal']}: {'ok' if x['ok'] else 'NO'}" for x in r.get("goals") or []) or r.get("error", "")
        print(f"[{ {'success': 'OK ', 'partial': 'PRT', 'fail': 'BAD'}.get(r['status'], 'ERR') }] {sp:9s} calls "
              f"{r.get('tool_calls', 0):3d} {r.get('seconds', 0):6.1f}s | {g} | {r.get('failed_calls', '')}"
              + (f" | EXCEPTION {r['exceptions']}" if r.get("exceptions") else ""), flush=True)
    n = len(res)
    k = sum(r["status"] == "success" for r in res)
    print(f"SUCCESS {k}/{n}, PARTIAL {sum(r['status'] == 'partial' for r in res)}, "
          f"ERROR {sum(r['status'] == 'error' for r in res)}, EXCEPTIONS {sum(len(r.get('exceptions') or []) for r in res)}")
    if out_path:
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1, default=str)
    return res


if __name__ == "__main__":
    main(sys.argv[1:])
