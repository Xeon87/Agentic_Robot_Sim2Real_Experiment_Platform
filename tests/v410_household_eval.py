#!/usr/bin/env python3
"""NERO V4.10 household scenes in the MuJoCo twin, driven by the test stand-in reasoner (tests/v49_reasoner.py, NOT
the product and NOT an LLM): one goal after the other, success per goal from the simulator truth.

    python3 tests/v410_household_eval.py H1 H2 H3 H4 H5          fixed scenes
    python3 tests/v410_household_eval.py seed:1-20 [--noise] [out.json]
A scene counts as success when every goal holds at the end, partial when some do.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTS = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("NERO_CALIB_DIR", os.path.join(os.path.expanduser("~"), ".cache", "nero_v410_eval_calib"))


def episode(spec, noise=None, verbose=False):
    import NERO_V410_Household_Scenes as HS
    import NERO_V489_General_Pick_Place as GP
    from v49_reasoner import GeometricReasoner
    t0 = time.time()
    sc = HS.scene_from_spec(spec)
    io = HS.make_twin(sc, log=(print if verbose else (lambda *_: None)), noise=noise)
    sk = GP.GeneralPickPlaceSkills(io)
    rs = GeometricReasoner(sk, log=print if verbose else (lambda *_: None))
    outs = []
    for s, rel, t in sc.goals:
        try:
            out = rs.run(s, t, "on" if rel == "onto" else "into")
        except Exception as e:
            out = {"ok": False, "why": f"exception {e!r}", "trace": traceback.format_exc()[-600:]}
        outs.append({k: v for k, v in out.items() if k != "relations"})
        io.step(800)
    io.step(1500)
    held = [f"obj_{io.hold}"] if getattr(io, "hold", None) else []
    st = HS.goal_status(io.m, io.d, sc, held=held)
    n_ok = sum(g["ok"] for g in st)
    status = "success" if n_ok == len(st) else ("partial" if n_ok else "fail")
    return {"scene": spec, "label": sc.label, "task": sc.task, "objects": sc.summary(), "status": status, "goals": st,
            "reasoner": outs, "tool_calls": len(rs.calls), "failed_calls": [(n, r) for n, _a, ok, r in rs.calls if not ok][-6:],
            "seconds": round(time.time() - t0, 1)}


def _specs(args):
    out = []
    for a in args:
        if a.startswith("seed:") and "-" in a:
            lo, hi = a[5:].split("-")
            out += [f"seed:{i}" for i in range(int(lo), int(hi) + 1)]
        else:
            out.append(a)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_json = args.pop() if args and args[-1].endswith(".json") else None
    noise = True if "--noise" in sys.argv else None
    res = []
    for spec in _specs(args or ["H1", "H2", "H3", "H4", "H5"]):
        r = episode(spec, noise=noise, verbose="--verbose" in sys.argv)
        res.append(r)
        tag = {"success": "OK ", "partial": "PRT", "fail": "BAD"}[r["status"]]
        print(f"[{tag}] {spec:9s} {r['task'][:70]:70s} calls {r['tool_calls']:3d} {r['seconds']:5.1f}s | "
              + " | ".join(f"{g['goal']}: {'ok' if g['ok'] else 'NO'} ({g['detail']})" for g in r["goals"])
              + ("" if r["status"] == "success" else f" | {[o.get('why') for o in r['reasoner'] if not o.get('ok')]} "
                 f"{r['failed_calls'][-2:]}"), flush=True)
    k = sum(r["status"] == "success" for r in res)
    p = sum(r["status"] == "partial" for r in res)
    print(f"SUCCESS {k}/{len(res)}, PARTIAL {p}")
    if out_json:
        with open(out_json, "w") as f:
            json.dump(res, f, indent=1, default=str)


if __name__ == "__main__":
    main()
