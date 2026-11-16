#!/usr/bin/env python3
"""V4.9.2 real-LLM batch evaluation of the general pick & place harness in the MuJoCo twin (needs mujoco and an
API key; it calls the model for real and costs tokens).

    export OPENAI_API_KEY=...            (or DEEPSEEK_API_KEY / MOONSHOT_API_KEY, as in the NERO LLM panel)
    python3 tests/v49_llm_eval.py --model gpt-6.1-sol --effort high --seeds 201-210 --out llm_sol_high.json
    python3 tests/v49_llm_eval.py --model gpt-5.6-luna --effort low --scene user --repeat 3
    options: --noise (V4.9.2 real-like perception)  --steps 40 (tool step budget)  --dry (print the plan only)

--scene random (default): NERO_V49_Random_Scenes.sample_scene(seed) for each seed.
--scene user: the general_pick_place scene with "Put egg_1 into cup_1." and "Stack cup_1 into cup_2."
Per episode: success from the simulator truth, tool calls, failed calls, LLM rounds, tokens, seconds, the model's
final text.  Results are appended to <out>.jsonl as they come (a crash keeps the finished episodes) and summarised
in <out> (json) + the console.  The model only sees the tool results (the same tools / prompt as the GUI).
"""
import argparse
import json
import os
import re
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTS = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault("NERO_CALIB_DIR", tempfile.mkdtemp(prefix="nero_v49_llm_eval_"))

USER_TASKS = [("Put egg_1 into cup_1.", "egg_1", "cup_1", "into"), ("Stack cup_1 into cup_2.", "cup_1", "cup_2", "into")]


def user_success(io, src, tgt):
    """truth check for the general_pick_place scene: src centre within the target's rim and below it"""
    import numpy as np
    ps = np.asarray(io.d.body(f"obj_{src}").xpos, float)
    pt = np.asarray(io.d.body(f"obj_{tgt}").xpos, float)          # cup origin = rim centre
    if getattr(io, "hold", None):
        return False, "still in the gripper"
    d = float(np.hypot(*(ps[:2] - pt[:2])))
    below = float(ps[2]) < float(pt[2]) - (0.004 if src.startswith("egg") else -0.040)
    return bool(d < 0.040 and below), f"{1000 * d:.0f} mm from the target axis, z {1000 * (ps[2] - pt[2]):+.0f} mm vs its rim"


def run_one(io, task, agent, cfg, check):
    import NERO_V489_General_Pick_Place as GP
    logs = []
    io.log = logs.append
    t0 = time.time()
    u0 = dict(getattr(agent, "_usage_totals", {}) or {})
    try:
        res = GP.run_general_pick_place(io, task, mode="execute", agent=agent, llm_cfg=cfg)
        final, err = res.get("final", ""), None
    except Exception as e:
        final, err = "", f"{e!r}"[:300]
    io.step(1500)
    ok, detail = check()
    u1 = dict(getattr(agent, "_usage_totals", {}) or {})
    calls = [l for l in logs if "CHUNK #" in l]
    fails = [re.sub(r"\s*\|.*", "", l.split("GENERAL] ", 1)[-1]) for l in logs if "[FAIL] [V4.8.9 GENERAL]" in l]
    return {"task": task, "success": bool(ok), "detail": detail, "error": err, "final_text": (final or "")[:400],
            "tool_calls": len(calls), "failed_calls": fails[-12:],
            "llm_rounds": sum(1 for l in logs if "LLM #" in l),
            "input_tokens": int(u1.get("input_tokens", 0)) - int(u0.get("input_tokens", 0)),
            "output_tokens": int(u1.get("output_tokens", 0)) - int(u0.get("output_tokens", 0)),
            "safety_stops": int(io.safety_trips()) if hasattr(io, "safety_trips") else None,
            "seconds": round(time.time() - t0, 1)}


def make_agent():
    import NERO_V2_7_1_Agent as A
    return A.NeroLLMAgent(emit=lambda s: None, tool_runner=None)


def episodes(args):
    """yield (label, io_factory, task, check_factory)"""
    import NERO_V49_Random_Scenes as RS
    import NERO_V489_General_Pick_Place as GP
    noise = True if args.noise else None
    for rep in range(args.repeat):
        if args.scene == "user":
            for task, src, tgt, _rel in USER_TASKS:
                def make(noise=noise, rep=rep):
                    return GP.TwinGeneralIO(noise=(dict(GP.REAL_LIKE_NOISE, seed=rep) if noise else None))
                yield f"user r{rep}", make, task, (lambda io, s=src, t=tgt: (lambda: user_success(io, s, t)))
        else:
            a, b = [int(v) for v in args.seeds.split("-")] if "-" in args.seeds else (int(args.seeds), int(args.seeds))
            for seed in range(a, b + 1):
                sc = RS.sample_scene(seed)
                yield (f"seed {seed} r{rep}", (lambda sc=sc: RS.make_twin(sc, noise=noise)), sc.task,
                       (lambda io, sc=sc: (lambda: RS.check_success(io, sc))))


def main(argv=None, agent=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-6.1-sol")
    ap.add_argument("--effort", default="high")
    ap.add_argument("--scene", default="random", choices=["random", "user"])
    ap.add_argument("--seeds", default="201-210")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--out", default="v49_llm_eval.json")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args(argv)
    eps = list(episodes(args))
    print(f"[V4.9.2 LLM EVAL] model={args.model} effort={args.effort} scene={args.scene} noise={bool(args.noise)} "
          f"episodes={len(eps)} (each one calls the model ~10-40 times)", flush=True)
    if args.dry:
        for label, _f, task, _c in eps:
            print("  ", label, task)
        return []
    agent = agent or make_agent()
    cfg = {"model": args.model, "api_key": args.api_key, "reasoning_effort": args.effort, "max_tool_steps": args.steps}
    res = []
    with open(args.out + "l", "a") as jl:
        for label, make_io, task, make_check in eps:
            io = make_io()
            try:
                r = run_one(io, task, agent, cfg, make_check(io))
            except Exception:
                r = {"task": task, "success": False, "error": traceback.format_exc()[-400:]}
            r.update({"episode": label, "model": args.model, "effort": args.effort, "noise": bool(args.noise)})
            res.append(r)
            jl.write(json.dumps(r, ensure_ascii=False) + "\n")
            jl.flush()
            print(f"[{'OK ' if r['success'] else 'BAD'}] {label:<10} {task:<32} calls {r.get('tool_calls', '?'):>3} "
                  f"tokens in {r.get('input_tokens', '?')} out {r.get('output_tokens', '?')} {r.get('seconds', '?')} s "
                  f"| {r.get('detail', '')} {r.get('error') or ''}", flush=True)
    k = sum(r["success"] for r in res)
    summ = {"model": args.model, "effort": args.effort, "noise": bool(args.noise), "scene": args.scene,
            "episodes": len(res), "success": k,
            "mean_tool_calls": round(sum(r.get("tool_calls", 0) for r in res) / max(1, len(res)), 1),
            "mean_input_tokens": round(sum(r.get("input_tokens", 0) for r in res) / max(1, len(res))),
            "mean_seconds": round(sum(r.get("seconds", 0) for r in res) / max(1, len(res)), 1), "runs": res}
    with open(args.out, "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=1)
    print(f"SUCCESS {k}/{len(res)} | mean calls {summ['mean_tool_calls']} | mean input tokens "
          f"{summ['mean_input_tokens']} | mean {summ['mean_seconds']} s -> {args.out}")
    return res


if __name__ == "__main__":
    main()
