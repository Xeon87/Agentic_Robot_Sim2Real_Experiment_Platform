#!/usr/bin/env python3
"""V4.9 random pick & place evaluation in the MuJoCo twin (needs the mujoco package).

    python3 tests/v49_random_eval.py [first_seed] [n_episodes] [out.json]

Each episode: NERO_V49_Random_Scenes.sample_scene(seed) -> twin -> GeneralPickPlaceSkills driven by the test
stand-in tests/v49_reasoner.GeometricReasoner -> success from the simulator truth (check_success).
Prints one line per episode and the success rate.
"""
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
os.environ.setdefault("NERO_CALIB_DIR", os.path.join(os.path.expanduser("~"), ".cache", "nero_v49_eval_calib"))


def episode(seed, verbose=False):
    import NERO_V49_Random_Scenes as RS
    import NERO_V489_General_Pick_Place as GP
    from v49_reasoner import GeometricReasoner
    t0 = time.time()
    sc = RS.sample_scene(seed)
    logs = []
    io = RS.make_twin(sc, log=logs.append)
    sk = GP.GeneralPickPlaceSkills(io)
    rs = GeometricReasoner(sk, log=print if verbose else (lambda *_: None))
    try:
        out = rs.run(sc.source, sc.target, sc.relation)
    except Exception as e:
        out = {"ok": False, "why": f"exception {e!r}", "trace": traceback.format_exc()[-800:]}
    io.step(1500)                                   # let the object settle
    ok, detail = RS.check_success(io, sc)
    return {"seed": seed, "task": sc.task, "kinds": [o.kind for o in sc.objects], "success": bool(ok), "detail": detail,
            "reasoner": {k: v for k, v in out.items() if k != "relations"}, "tool_calls": len(rs.calls),
            "failed_calls": [(n, r) for n, _a, okc, r in rs.calls if not okc][-6:],
            "safety_stops": int(io.safety_trips()) if hasattr(io, "safety_trips") else None,
            "seconds": round(time.time() - t0, 1)}


def main():
    s0 = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    out = sys.argv[3] if len(sys.argv) > 3 else None
    res = []
    for seed in range(s0, s0 + n):
        r = episode(seed)
        res.append(r)
        print(f"[{'OK ' if r['success'] else 'BAD'}] seed {seed:3d} {r['task']:<34} {r['detail']:<70} "
              f"calls {r['tool_calls']:3d} {r['seconds']:5.1f}s {'' if r['success'] else r['reasoner'].get('why', '')} "
              f"{r['failed_calls'][-2:] if not r['success'] else ''}", flush=True)
    k = sum(r["success"] for r in res)
    print(f"SUCCESS {k}/{len(res)} = {100.0 * k / max(1, len(res)):.0f} %")
    if out:
        with open(out, "w") as f:
            json.dump(res, f, indent=1, default=str)
    return k, len(res)


if __name__ == "__main__":
    main()
