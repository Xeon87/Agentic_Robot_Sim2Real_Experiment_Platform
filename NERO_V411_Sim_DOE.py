#!/usr/bin/env python3
"""NERO V4.11 SIM DOE - batch evaluation of the general pick & place harness in the GUI simulation, from the console.

Design of experiments: task rows (scene + task sentence + structured goals) x LLM settings (provider / model /
reasoning effort) x replicates.  Every run goes through the real stack (console -> simulation GUI -> SimAgent RUN
TASK -> harness -> backend physics), exactly as a person would start it:

  per scene (fixed:H1 .. H5 or seed:<n>)  start the simulation with that scene (household_pick_place), wait for the
                                          sim bridge's ready file, then for each run of that scene:
    send     <task dir>/inbox/<run>.json = {task, llm, reset_objects}; the harness puts the objects back to their
             start poses at the task start (arm at home first), so every run starts from the same table
    wait     <task dir>/outbox/<run>.json (status + the harness's TASK_STATS); timeout -> STOP AGENT, then a restart
    check    simulator truth of the structured goals (NERO_V410_Household_Scenes.request_eval -> physics process)
    record   <release>/logs/doe/<time>/runs.jsonl (+ runs.csv, report.md, plan.json, doe.log)
  then stop the simulation and go to the next scene.

Success is the simulator truth of the goals only - never the LLM's own claim.  Lamps: grey = waiting, blinking green
= running, green = success, yellow = partial, red = fail, blue = the run could not be judged on the harness (sim crash,
time-out, task not started, blow-up of the physics, LLM / API error).

Pure logic (plan, expansion, classification, statistics, report) has no Tk; the runner is a state machine driven by
tick() (the tab calls it from root.after; tests call it directly with a fake simulation).
Tk rules of this project: no emoji in widget text, no tkinter.font.Font, worker threads never touch Tk variables.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import random
import sys
import time
from typing import Any, Callable, Dict, List, Optional

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

VERSION = "V4.11.2"
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")      # same list as the console / SimAgent panels
MARKER = "V4.11 SIM DOE"
MARKER_V4111 = "V4.11.1 DOE MODEL PICKER + FONT-SIZED ROWS"
MARKER_V4112 = "V4.11.2 DOE FINAL-STATE SNAPSHOTS + RUN TRACE"
BLOWUP_TRIPS = 5                        # same threshold as the harness's V4.10.3 auto reset

STATUS_COLOR = {"pending": "#9e9e9e", "running": "#2e7d32", "success": "#2e7d32", "partial": "#f2c037",
                "fail": "#c62828", "invalid": "#1e63c4", "skipped": "#cfd8dc"}
STATUS_TEXT = {"pending": "等待", "running": "运行中", "success": "成功", "partial": "部分成功", "fail": "失败",
               "invalid": "无法判定", "skipped": "未运行"}
CLASS_TEXT = {
    "success": "全部目标达成（仿真真值）",
    "partial": "部分目标达成",
    "ended_not_done": "LLM 结束了任务，但目标没有达成",
    "claimed_but_not_done": "harness 已到 VERIFIED（LLM 认为完成），真值没有达成",
    "step_budget": "工具步数用完",
    "safety_stops": "有安全停止，目标没有达成",
    "blowup": "仿真发散（一次工具调用 > 5 次安全停止，或物体掉下桌）",
    "llm_error": "LLM / API 出错",
    "stopped": "被停止",
    "timeout": "超时（已发 STOP）",
    "not_started": "任务没有开始",
    "sim_crash": "仿真进程退出",
    "no_eval": "没有拿到真值检查结果",
}


# ================================================================================================ plan
def default_rows(specs=("fixed:H1", "fixed:H2", "fixed:H3", "fixed:H4", "fixed:H5")) -> List[dict]:
    """one task row per scene: the scene's own task sentence and goals"""
    import NERO_V410_Household_Scenes as HS
    out = []
    for sp in specs:
        sc = HS.scene_from_spec(sp)
        out.append({"scene": sp, "task": sc.task, "goals": HS.goals_text(sc.goals)})
    return out


def check_row(row) -> List[str]:
    """problems of one task row (empty = ok): scene builds, goals parse and name objects of that scene"""
    import NERO_V410_Household_Scenes as HS
    probs = []
    try:
        sc = HS.scene_from_spec(str(row.get("scene") or ""))
    except Exception as exc:
        return [f"场景 {row.get('scene')!r} 无法生成：{exc}"]
    if not str(row.get("task") or "").strip():
        probs.append("任务句为空")
    try:
        goals = HS.parse_goals(row.get("goals"))
    except ValueError as exc:
        return probs + [str(exc)]
    if not goals:
        probs.append("没有结构化目标（成功与否要靠它判定）")
    ids = {it.oid for it in sc.items}
    for a, _r, b in goals:
        for o in (a, b):
            if o not in ids:
                probs.append(f"目标里的 {o} 不在场景 {row['scene']} 中（有：{', '.join(sorted(ids))}）")
    return probs


def parse_model(text) -> dict:
    """'deepseek:deepseek-v4-pro@high' / 'gpt-6.1-sol' -> {provider, model, effort}"""
    t = str(text or "").strip()
    effort = ""
    if "@" in t:
        t, effort = t.rsplit("@", 1)
    prov = ""
    if ":" in t:
        prov, t = t.split(":", 1)
    if not prov:
        try:
            import NERO_V2_7_1_Agent as A
            prov = A.provider_for_model(t)
        except Exception:
            prov = "openai"
    return {"provider": prov.strip(), "model": t.strip(), "effort": effort.strip() or "high"}


def model_label(m) -> str:
    return f"{m['provider']}:{m['model']}@{m.get('effort') or 'high'}"


def expand(plan) -> List[dict]:
    """plan -> ordered runs, grouped by scene (one simulation start per scene); inside a scene the order is
    row-major or shuffled with plan['order_seed'] (plan['shuffle'])"""
    rows, models = list(plan.get("rows") or []), list(plan.get("models") or [])
    reps = max(1, int(plan.get("replicates") or 1))
    scenes = []
    for r in rows:
        if r["scene"] not in scenes:
            scenes.append(r["scene"])
    out = []
    rng = random.Random(int(plan.get("order_seed") or 0))
    for sp in scenes:
        grp = []
        for ri, r in enumerate(rows):
            if r["scene"] != sp:
                continue
            for mi, m in enumerate(models):
                for k in range(reps):
                    grp.append({"row": ri, "model": mi, "rep": k + 1, "scene": sp, "task": r["task"],
                                "goals": r["goals"], "llm": dict(m, steps=int(plan.get("steps") or 60),
                                                                     dry_run=False)})
        if plan.get("shuffle"):
            rng.shuffle(grp)
        out += grp
    for i, r in enumerate(out):
        r["id"] = f"r{i + 1:03d}"
        r["status"] = "pending"
    return out


def estimate_s(plan, per_run_s=240.0, per_start_s=90.0) -> float:
    runs = expand(plan)
    return len(runs) * per_run_s + len({r["scene"] for r in runs}) * per_start_s


# ================================================================================================ classification
def classify(out: Optional[dict], ev: Optional[dict], timed_out=False, crashed=False) -> (str, str):
    """(lamp status, failure class) of one run from the outbox message, the truth check and the runner's flags"""
    st = (out or {}).get("stats") or {}
    err = str(st.get("error") or "")
    trips_tool = max([int(t.get("safety_stops") or 0) for t in st.get("tools") or []] or [0])
    off = any(o.get("off_table") for o in ((ev or {}).get("objects") or {}).values())
    if crashed:
        return "invalid", "sim_crash"
    if timed_out:
        return "invalid", "timeout"
    if out is None:
        return "invalid", "no_eval"
    if out.get("status") == "not_started":
        return "invalid", "not_started"
    truth = (ev or {}).get("status")
    if truth == "success":
        return "success", "success"
    if trips_tool > BLOWUP_TRIPS or off:
        return "invalid", "blowup"
    if err and "step budget" in err:
        return ("partial" if truth == "partial" else "fail"), "step_budget"
    if err and "stopped" in err.lower():
        return "invalid", "stopped"
    if err:
        return "invalid", "llm_error"
    if ev is None:
        return "invalid", "no_eval"
    if truth == "partial":
        return "partial", "partial"
    if int(st.get("safety_stops") or 0) > 0:
        return "fail", "safety_stops"
    if st.get("stage") == "VERIFIED" or st.get("ok"):
        return "fail", "claimed_but_not_done"
    return "fail", "ended_not_done"


def last_fail_reason(stats) -> Optional[str]:
    fails = [t for t in (stats or {}).get("tools") or [] if not t.get("ok") and t.get("reason")]
    return f"{fails[-1]['tool']}: {fails[-1]['reason']}" if fails else None


# ================================================================================================ statistics
def wilson(k, n, z=1.96):
    """95 % Wilson interval of a success rate"""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _group(runs, key):
    out = {}
    for r in runs:
        out.setdefault(key(r), []).append(r)
    return out


def summary_rows(runs, key) -> List[dict]:
    """per group: n judged, success, partial, fail, invalid, rate + CI (judged = success / partial / fail)"""
    res = []
    for k, rs in _group([r for r in runs if r.get("status") not in ("pending", "running", "skipped")], key).items():
        n_s = sum(r["status"] == "success" for r in rs)
        n_p = sum(r["status"] == "partial" for r in rs)
        n_f = sum(r["status"] == "fail" for r in rs)
        n_i = sum(r["status"] == "invalid" for r in rs)
        n = n_s + n_p + n_f
        lo, hi = wilson(n_s, n)
        calls = [len((r.get("stats") or {}).get("tools") or []) for r in rs if r.get("stats")]
        secs = [r["seconds"] for r in rs if r.get("seconds")]
        trips = sum(int((r.get("stats") or {}).get("safety_stops") or 0) for r in rs)
        toks = sum(int(((r.get("stats") or {}).get("usage") or {}).get("input_tokens") or 0)
                   + int(((r.get("stats") or {}).get("usage") or {}).get("output_tokens") or 0) for r in rs)
        res.append({"group": k, "runs": len(rs), "judged": n, "success": n_s, "partial": n_p, "fail": n_f,
                    "invalid": n_i, "rate": (n_s / n) if n else None, "ci95": (lo, hi) if n else None,
                    "mean_tool_calls": (sum(calls) / len(calls)) if calls else None,
                    "mean_s": (sum(secs) / len(secs)) if secs else None, "safety_stops": trips, "tokens": toks})
    return sorted(res, key=lambda x: str(x["group"]))


def _pct(x):
    return "-" if x is None else f"{100 * x:.0f}%"


def _md_table(rows, title):
    lines = [f"### {title}", "", "| 组 | 运行 | 可判定 | 成功 | 部分 | 失败 | 无法判定 | 成功率 | 95% 区间 | 平均调用 | 平均用时 s "
             "| 安全停止 | tokens |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = "-" if r["ci95"] is None else f"{_pct(r['ci95'][0])}–{_pct(r['ci95'][1])}"
        calls = "-" if r["mean_tool_calls"] is None else f"{r['mean_tool_calls']:.1f}"
        secs = "-" if r["mean_s"] is None else f"{r['mean_s']:.0f}"
        lines.append(f"| {r['group']} | {r['runs']} | {r['judged']} | {r['success']} | {r['partial']} | {r['fail']} | "
                     f"{r['invalid']} | {_pct(r['rate'])} | {ci} | {calls} | {secs} | {r['safety_stops']} | "
                     f"{r['tokens']} |")
    return "\n".join(lines) + "\n"


def report_md(plan, runs) -> str:
    done = [r for r in runs if r.get("status") not in ("pending", "running", "skipped")]
    rows = plan.get("rows") or []
    models = plan.get("models") or []
    head = [f"# NERO Sim DOE 报告（{plan.get('name') or ''}）", "",
            f"- 版本 {VERSION}；开始 {plan.get('started') or '-'}；结束 {plan.get('ended') or '-'}",
            f"- 计划：{len(rows)} 个任务 × {len(models)} 组 LLM 设置 × {plan.get('replicates')} 次 = {len(runs)} 次运行；"
            f"已完成 {len(done)}",
            "- 成功 = 仿真真值检查全部目标达成（不看 LLM 自己的说法）；成功率只算可判定的运行（成功 / 部分 / 失败），"
            "「无法判定」= 仿真或 API 的问题（崩溃、超时、发散、API 出错），单列。",
            "- 95% 区间为 Wilson 区间；次数少时区间很宽，结论要看区间。", ""]
    body = [_md_table(summary_rows(runs, lambda r: "全部"), "总体"),
            _md_table(summary_rows(runs, lambda r: model_label(r["llm"])), "按 LLM 设置"),
            _md_table(summary_rows(runs, lambda r: f"T{r['row'] + 1} {r['scene']}"), "按任务"),
            _md_table(summary_rows(runs, lambda r: (f"T{r['row'] + 1} × {model_label(r['llm'])}")), "任务 × LLM")]
    cls = _group(done, lambda r: r.get("klass") or "-")
    body.append("### 结果分类\n\n| 分类 | 次数 | 说明 |\n|---|---|---|\n" + "".join(
        f"| {k} | {len(v)} | {CLASS_TEXT.get(k, '')} |\n" for k, v in sorted(cls.items(), key=lambda x: -len(x[1]))))
    reasons = _group([r for r in done if r.get("last_fail")], lambda r: r["last_fail"])
    if reasons:
        body.append("\n### 未成功运行里最后一个失败的工具调用\n\n| 工具: 原因 | 次数 |\n|---|---|\n" + "".join(
            f"| {k} | {len(v)} |\n" for k, v in sorted(reasons.items(), key=lambda x: -len(x[1]))))
    body.append("\n### 每次运行（截图、工具调用、LLM 文字见 runs.html）\n\n"
                "| id | 任务 | LLM | 次 | 结果 | 分类 | 目标 | 调用 | 安全停止 | 用时 s | 截图 |\n"
                "|---|---|---|---|---|---|---|---|---|---|---|\n" + "".join(
                    f"| {r['id']} | T{r['row'] + 1} {r['scene']} | {model_label(r['llm'])} | {r['rep']} | "
                    f"{STATUS_TEXT.get(r.get('status'), r.get('status'))} | {r.get('klass') or '-'} | "
                    f"{(r.get('eval') or {}).get('n_ok', '-')}/{(r.get('eval') or {}).get('n_goals', '-')} | "
                    f"{len((r.get('stats') or {}).get('tools') or [])} | "
                    f"{(r.get('stats') or {}).get('safety_stops', '-')} | {r.get('seconds') or '-'} | "
                    f"{' '.join(f'[{os.path.basename(x).split(chr(95))[-1][:-4]}]({x})' for x in r.get('snapshots') or []) or '-'} |\n"
                    for r in runs))
    return "\n".join(head) + "\n".join(body)


def _esc(x):
    import html
    return html.escape(str(x if x is not None else ""))


def report_html(plan, runs) -> str:
    """V4.11.2 RUN TRACE: one page with every run - result, goal check (simulator truth), final-state pictures,
    every tool call (arguments, ok / reason, result excerpt), the LLM's text and its final answer"""
    css = ("body{font-family:sans-serif;margin:16px;color:#222}h2{margin:28px 0 6px}"
           ".run{border:1px solid #ccc;border-radius:6px;padding:10px 12px;margin:14px 0}"
           ".tag{display:inline-block;padding:1px 8px;border-radius:10px;color:#fff;font-size:13px}"
           "img{width:32%;min-width:260px;margin:4px 4px 0 0;border:1px solid #ddd}"
           "table{border-collapse:collapse;font-size:12px;margin-top:6px}td,th{border:1px solid #ddd;padding:2px 6px;"
           "vertical-align:top;text-align:left}code{font-size:12px;white-space:pre-wrap;word-break:break-all}"
           ".bad{color:#c62828}.muted{color:#777}summary{cursor:pointer;margin-top:6px}")
    out = [f"<!doctype html><html><head><meta charset='utf-8'><title>NERO Sim DOE {_esc(plan.get('name'))}</title>"
           f"<style>{css}</style></head><body><h1>NERO Sim DOE 运行记录</h1>",
           f"<p class='muted'>版本 {VERSION}；开始 {_esc(plan.get('started'))}；结束 {_esc(plan.get('ended'))}；"
           f"{len(runs)} 次运行。图片是任务结束、做真值检查那一刻的仿真状态（物理进程的模型 + 关节 / 物体位置，"
           f"在单独的进程里渲染）。统计见 report.md。</p>"]
    for r in runs:
        st = r.get("stats") or {}
        ev = r.get("eval") or {}
        col = STATUS_COLOR.get(r.get("status"), "#999")
        out.append(f"<div class='run' id='{_esc(r['id'])}'><b>{_esc(r['id'])}</b> "
                   f"<span class='tag' style='background:{col}'>{_esc(STATUS_TEXT.get(r.get('status'), r.get('status')))}"
                   f"</span> {_esc(CLASS_TEXT.get(r.get('klass'), r.get('klass') or ''))} &nbsp; "
                   f"T{r['row'] + 1} {_esc(r['scene'])} · {_esc(model_label(r['llm']))} · 第 {r['rep']} 次 · "
                   f"{_esc(r.get('seconds') or '-')} s")
        out.append(f"<div>任务句：{_esc(r['task'])}</div><div class='muted'>目标：{_esc(r['goals'])}</div>")
        if ev.get("goals"):
            out.append("<table><tr><th>目标</th><th>达成</th><th>真值</th></tr>" + "".join(
                f"<tr><td>{_esc(g.get('goal'))}</td><td>{'是' if g.get('ok') else '<span class=bad>否</span>'}</td>"
                f"<td>{_esc(g.get('detail'))}</td></tr>" for g in ev["goals"]) + "</table>")
            if ev.get("held"):
                out.append(f"<div class='bad'>结束时仍在夹爪里：{_esc(', '.join(ev['held']))}</div>")
            offs = [k for k, o in (ev.get("objects") or {}).items() if o.get("off_table")]
            if offs:
                out.append(f"<div class='bad'>掉下桌：{_esc(', '.join(offs))}</div>")
        if r.get("snapshots"):
            out.append("<div>" + "".join(f"<a href='{_esc(x)}'><img src='{_esc(x)}' alt='{_esc(x)}'></a>"
                                         for x in r["snapshots"]) + "</div>")
        elif r.get("snapshot_why"):
            out.append(f"<div class='muted'>没有截图：{_esc(r['snapshot_why'])}</div>")
        if st.get("error"):
            out.append(f"<div class='bad'>错误：{_esc(st['error'])}</div>")
        tools = st.get("tools") or []
        if tools:
            out.append(f"<details><summary>{len(tools)} 次工具调用（安全停止 {st.get('safety_stops', 0)}）</summary>"
                       "<table><tr><th>#</th><th>工具</th><th>结果</th><th>参数</th><th>结果摘录</th></tr>" + "".join(
                           f"<tr><td>{t.get('step')}</td><td>{_esc(t.get('tool'))}</td><td>"
                           + ("ok" if t.get("ok") else f"<span class=bad>{_esc(t.get('reason'))}</span>")
                           + (f" / 安全停止 {t['safety_stops']}" if t.get("safety_stops") else "")
                           + f"</td><td><code>{_esc(json.dumps(t.get('args') or {}, ensure_ascii=False))}</code></td>"
                           f"<td><code>{_esc(t.get('result') or '')}</code></td></tr>" for t in tools)
                       + "</table></details>")
        texts = st.get("texts") or []
        if texts:
            out.append("<details><summary>LLM 的文字</summary>" + "".join(
                f"<div><span class='muted'>#{x.get('round')}</span> {_esc(x.get('text'))}</div>" for x in texts)
                + "</details>")
        if st.get("final"):
            out.append(f"<div>LLM 最后的回答：{_esc(st['final'])}</div>")
        out.append("</div>")
    out.append("</body></html>")
    return "\n".join(out)


CSV_COLS = ("id", "scene", "row", "task", "goals", "provider", "model", "effort", "rep", "status", "klass", "n_ok",
            "n_goals", "tool_calls", "safety_stops", "seconds", "input_tokens", "output_tokens", "last_fail", "error")


def runs_csv(runs) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(CSV_COLS)
    for r in runs:
        st = r.get("stats") or {}
        u = st.get("usage") or {}
        ev = r.get("eval") or {}
        w.writerow([r["id"], r["scene"], r["row"] + 1, r["task"], r["goals"], r["llm"]["provider"], r["llm"]["model"],
                    r["llm"].get("effort"), r["rep"], r.get("status"), r.get("klass"), ev.get("n_ok"),
                    ev.get("n_goals"), len(st.get("tools") or []), st.get("safety_stops"), r.get("seconds"),
                    u.get("input_tokens"), u.get("output_tokens"), r.get("last_fail"), st.get("error")])
    return buf.getvalue()


# ================================================================================================ runner
class DOERunner:
    """state machine; tick() every ~0.5 s.  sim = {"start": f(scene_spec, env) -> (ok, msg), "stop": f(),
    "running": f() -> bool, optional "ready": f() -> bool (the launcher's SYSTEM READY)}; files in task_dir (inbox / outbox / ready.json) and the reset / eval files of
    NERO_V410_Household_Scenes"""

    def __init__(self, plan, sim, out_dir, task_dir, log=None, on_change=None, timeouts=None, now=time.time):
        self.plan, self.sim, self.out, self.tdir = plan, sim, out_dir, task_dir
        self.log_cb = log or (lambda s: None)
        self.on_change = on_change or (lambda r: None)
        self.now = now
        t = {"ready_s": 420.0, "settle_s": 8.0, "run_s": 60.0 * float(plan.get("run_timeout_min") or 20),
             "stop_grace_s": 90.0, "eval_s": 20.0, "sim_stop_s": 120.0, "after_stop_s": 5.0}
        t.update(timeouts or {})
        self.T = t
        self.runs = expand(plan)
        self.prefix = "doe" + os.path.basename(os.path.normpath(out_dir)).replace("-", "")
        self.i = 0
        self.state = "idle"
        self.t_state = now()
        self.scene_up = None
        self.pause_req = False
        self.abort_req = False
        self.finished = False
        os.makedirs(out_dir, exist_ok=True)
        for sub in ("inbox", "outbox", "done"):
            os.makedirs(os.path.join(task_dir, sub), exist_ok=True)

    # -- helpers
    def log(self, s):
        line = time.strftime("%H:%M:%S") + " " + s
        try:
            with open(os.path.join(self.out, "doe.log"), "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass
        self.log_cb(line)

    def _set(self, st):
        self.state, self.t_state = st, self.now()

    def _age(self):
        return self.now() - self.t_state

    def fid(self, r):
        """file id of a run in the task channel (unique per DOE: older outbox files never match)"""
        return f"{self.prefix}_{r['id']}"

    def cur(self):
        return self.runs[self.i] if self.i < len(self.runs) else None

    def _ready_since(self, t0):
        p = os.path.join(self.tdir, "ready.json")
        try:
            return os.path.getmtime(p) >= t0 - 0.5
        except OSError:
            return False

    def _write(self, name, obj):
        p = os.path.join(self.tdir, "inbox", name)
        with open(p + ".tmp", "w") as f:
            json.dump(obj, f)
        os.replace(p + ".tmp", p)

    def save(self):
        with open(os.path.join(self.out, "plan.json"), "w", encoding="utf-8") as f:
            json.dump(self.plan, f, ensure_ascii=False, indent=1)
        with open(os.path.join(self.out, "runs.jsonl"), "w", encoding="utf-8") as f:
            for r in self.runs:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        with open(os.path.join(self.out, "runs.csv"), "w", encoding="utf-8", newline="") as f:
            f.write(runs_csv(self.runs))
        with open(os.path.join(self.out, "report.md"), "w", encoding="utf-8") as f:
            f.write(report_md(self.plan, self.runs))
        with open(os.path.join(self.out, "runs.html"), "w", encoding="utf-8") as f:     # V4.11.2 run trace
            f.write(report_html(self.plan, self.runs))

    # -- control
    def start(self):
        self.plan["started"] = time.strftime("%Y-%m-%d %H:%M:%S")
        n_sc = len({r["scene"] for r in self.runs})
        self.log(f"[V4.11 DOE] start: {len(self.runs)} runs, {n_sc} scene(s), results -> {self.out}")
        self.save()
        self._set("next")

    def pause(self):
        self.pause_req = True
        self.log("[V4.11 DOE] pause requested: stops after the current run")

    def resume(self):
        self.pause_req = False
        if self.state == "paused":
            self.log("[V4.11 DOE] resumed")
            self._set("next")

    def abort(self):
        self.abort_req = True
        self.log("[V4.11 DOE] abort requested")

    # -- the state machine
    def tick(self):
        if getattr(self, "_in_tick", False):              # the console's stale-process check runs root.update()
            return
        self._in_tick = True
        try:
            self._tick()
        except Exception as exc:                          # never leave the DOE stuck silently
            self.log(f"[FAIL] [V4.11 DOE] runner error in state {self.state}: {type(exc).__name__}: {exc}")
            self._set("next")
        finally:
            self._in_tick = False

    def _finish(self, why):
        for r in self.runs:
            if r["status"] in ("pending", "running"):
                r["status"] = "skipped"
                self.on_change(r)
        self.plan["ended"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()
        if self.sim["running"]():
            self.sim["stop"]()
        self.finished = True
        self._set("done")
        self.log(f"[V4.11 DOE] {why}; report: {os.path.join(self.out, 'report.md')}")

    def _tick(self):
        s = self.state
        if s in ("idle", "done", "paused"):
            return
        r = self.cur()
        if s == "next":
            if self.abort_req:
                return self._finish("aborted")
            if r is None:
                return self._finish("finished")
            if self.pause_req:
                self._set("paused")
                self.log("[V4.11 DOE] paused")
                return
            if self.scene_up != r["scene"]:
                if self.sim["running"]():
                    self.log(f"[V4.11 DOE] stop the simulation (scene {self.scene_up} done)")
                    self.sim["stop"]()
                    self._set("stopping")
                    return
                self._set("starting")
                self.t_start_sim = self.now()
                ok, msg = self.sim["start"](r["scene"], {})
                self.log(f"[V4.11 DOE] start the simulation with {r['scene']}: {msg}")
                if not ok:
                    return self._finish(f"the simulation did not start ({msg})")
                return
            return self._send(r)
        if s == "stopping":
            if not self.sim["running"]():
                self.scene_up = None
                self._set("after_stop")
            elif self._age() > self.T["sim_stop_s"]:
                return self._finish("the simulation did not stop")
            return
        if s == "after_stop":
            if self._age() >= self.T["after_stop_s"]:
                self._set("next")
            return
        if s == "starting":
            if not self.sim["running"]():
                return self._finish("the simulation exited while starting")
            if self._ready_since(self.t_start_sim) and self.sim.get("ready", lambda: True)():
                self._set("settle")             # sim bridge watching + the launcher's SYSTEM READY (MoveIt up)
            elif self._age() > self.T["ready_s"]:
                self.log("[FAIL] [V4.11 DOE] the simulation did not become ready; stopping it")
                self.sim["stop"]()
                return self._finish("the simulation did not become ready")
            return
        if s == "settle":
            if self._age() >= self.T["settle_s"]:
                self.scene_up = r["scene"]
                self.log(f"[V4.11 DOE] simulation ready ({r['scene']})")
                self._set("next")
            return
        if s == "running":
            if self.abort_req:
                self._write(f"stop_{self.fid(r)}.json", {"id": self.fid(r), "cmd": "stop"})
                r["timed_out"] = True
                return self._finish("aborted")
            if not self.sim["running"]():
                self.log(f"[FAIL] [V4.11 DOE] {r['id']}: the simulation exited during the run")
                return self._end(r, None, None, crashed=True)
            p = os.path.join(self.tdir, "outbox", f"{self.fid(r)}.json")
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        out = json.load(f)
                except Exception:
                    return
                r["outbox"] = {k: out.get(k) for k in ("status", "status_text", "t_start", "t_end", "llm")}
                r["stats"] = out.get("stats") or {}
                if out.get("status") == "not_started":
                    return self._end(r, out, None)
                import NERO_V410_Household_Scenes as HS
                self.eval_file = HS.request_eval(HS.parse_goals(r["goals"]), rid=self.fid(r))
                self._out = out
                self._set("eval")
                return
            if self._age() > self.T["run_s"] and not r.get("timed_out"):
                r["timed_out"] = True
                self.log(f"[WARN] [V4.11 DOE] {r['id']}: time-out after {self.T['run_s'] / 60:.0f} min -> STOP AGENT")
                self._write(f"stop_{self.fid(r)}.json", {"id": self.fid(r), "cmd": "stop"})
                self._stop_t = self.now()
            if r.get("timed_out") and self.now() - getattr(self, "_stop_t", self.now()) > self.T["stop_grace_s"]:
                self.log(f"[FAIL] [V4.11 DOE] {r['id']}: no answer after STOP -> restart the simulation")
                return self._end(r, None, None, timed_out=True, restart=True)
            return
        if s == "snap":
            return self._snap_tick()
        if s == "eval":
            f = self.eval_file + ".done"
            if os.path.exists(f):
                try:
                    with open(f) as fh:
                        ev = json.load(fh)
                except Exception:
                    return
                return self._end(r, self._out, ev, timed_out=bool(r.get("timed_out")))
            if self._age() > self.T["eval_s"]:
                self.log(f"[WARN] [V4.11 DOE] {r['id']}: no truth check answer within {self.T['eval_s']:.0f} s")
                return self._end(r, self._out, None, timed_out=bool(r.get("timed_out")))
            return

    def _send(self, r):
        r["status"] = "running"
        r["t0"] = self.now()
        self._write(f"{self.fid(r)}.json", {"id": self.fid(r), "task": r["task"], "llm": r["llm"], "reset_objects":
                                        bool(self.plan.get("reset_each_run", True))})
        self.log(f"[V4.11 DOE] {r['id']} ({self.i + 1}/{len(self.runs)}) {r['scene']} T{r['row'] + 1} "
                 f"{model_label(r['llm'])} #{r['rep']}: sent")
        self.on_change(r)
        self._set("running")

    def _end(self, r, out, ev, timed_out=False, crashed=False, restart=False):
        r["eval"] = ev
        r["status"], r["klass"] = classify(out, ev, timed_out=timed_out, crashed=crashed)
        r["seconds"] = round(self.now() - r.get("t0", self.now()), 1)
        r["last_fail"] = last_fail_reason(r.get("stats")) if r["status"] != "success" else None
        self.log(f"[V4.11 DOE] {r['id']}: {STATUS_TEXT[r['status']]} ({r['klass']}) "
                 f"goals {(ev or {}).get('n_ok', '-')}/{(ev or {}).get('n_goals', '-')} | "
                 f"{len((r.get('stats') or {}).get('tools') or [])} tool calls | {r['seconds']:.0f} s")
        self.on_change(r)
        self.i += 1
        self.save()
        if crashed or restart or r["klass"] == "blowup" and self.plan.get("restart_after_blowup", True):
            self.scene_up = None                          # next run of this scene starts a fresh simulation
            if r["klass"] == "blowup":
                self.log("[V4.11 DOE] blow-up -> the next run starts a fresh simulation")
        if self._start_snapshot(r, ev):                   # V4.11.2: pictures of the final state first
            self._set("snap")
        else:
            self._set("next")

    # ---- V4.11.2 final-state snapshots
    def _start_snapshot(self, r, ev) -> bool:
        """render the final state in a separate process (model saved by the physics process + qpos of the truth
        check); the runner waits for it (it reads the scene file, which the next simulation start replaces)"""
        r["snapshots"] = []
        if not self.plan.get("snapshots", True):
            return False
        if not ev or not ev.get("qpos"):
            r["snapshot_why"] = "no final state (no truth check answer)"
            return False
        if not ev.get("model_xml"):
            r["snapshot_why"] = f"the simulation did not save its model ({ev.get('model_why') or 'unknown'})"
            return False
        import subprocess
        d = os.path.join(self.out, "snapshots")
        os.makedirs(d, exist_ok=True)
        prefix = os.path.join(d, f"{r['id']}_{r['status']}")
        st = prefix + "_state.json"
        with open(st, "w") as f:
            json.dump({"qpos": ev["qpos"], "sim_time": ev.get("sim_time"), "model_xml": ev["model_xml"]}, f)
        log = open(prefix + "_render.json", "w")
        try:
            p = subprocess.Popen([sys.executable, os.path.join(HERE, "NERO_V411_Snapshot.py"), "--xml",
                                  ev["model_xml"], "--state", st, "--out", prefix], stdout=log,
                                 stderr=subprocess.DEVNULL, cwd=HERE)
        except Exception as exc:
            log.close()
            r["snapshot_why"] = f"renderer not started: {exc}"
            return False
        self._snap = {"r": r, "proc": p, "log": log, "path": prefix + "_render.json", "t0": self.now()}
        return True

    def _snap_tick(self):
        sn = self._snap
        p, r = sn["proc"], sn["r"]
        if p.poll() is None and self.now() - sn["t0"] < self.T.get("snap_s", 120.0):
            return
        if p.poll() is None:
            p.kill()
        sn["log"].close()
        res = {}
        try:
            with open(sn["path"]) as f:
                lines = [x for x in f.read().splitlines() if x.startswith("{")]
            res = json.loads(lines[-1]) if lines else {}
        except Exception:
            pass
        if res.get("ok"):
            r["snapshots"] = [os.path.relpath(x, self.out) for x in res.get("files") or []]
            self.log(f"[V4.11.2 DOE] {r['id']}: {len(r['snapshots'])} picture(s) of the final state")
        else:
            r["snapshot_why"] = res.get("why") or "the renderer gave no answer"
            self.log(f"[WARN] [V4.11.2 DOE] {r['id']}: no pictures ({r['snapshot_why'][:200]})")
        self.on_change(r)
        self.save()
        self._snap = None
        self._set("next")


def new_out_dir(base=None) -> str:
    base = base or os.environ.get("NERO_V411_DOE_DIR") or os.path.join(HERE, "logs", "doe")
    return os.path.join(base, time.strftime("%Y%m%d_%H%M%S"))


def load_runs(out_dir) -> (dict, List[dict]):
    with open(os.path.join(out_dir, "plan.json"), encoding="utf-8") as f:
        plan = json.load(f)
    runs = []
    with open(os.path.join(out_dir, "runs.jsonl"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                runs.append(json.loads(line))
    return plan, runs


# ================================================================================================ Tk tab
class DOETab:
    """the console's 「仿真 DOE」 tab.  sim_api = {"start": f(spec, env) -> (ok, msg), "stop": f(),
    "running": f() -> bool, "busy": f() -> str|None (why the simulation cannot be used now), "llm_default": f() ->
    dict, "key_env": f(providers) -> dict}"""

    def __init__(self, tk, ttk, parent, root, sim_api, messagebox=None, filedialog=None):
        self.tk, self.ttk, self.root, self.api = tk, ttk, root, sim_api
        self.mb, self.fd = messagebox, filedialog
        self.rows: List[dict] = []
        self.models: List[dict] = []
        self.runner: Optional[DOERunner] = None
        self.blink = False
        self.lamp_items = {}
        # V4.11.1: row height and column widths from the real font size (large fonts / HiDPI screens)
        try:
            self.line_px = int(root.tk.call("font", "metrics", "TkDefaultFont", "-linespace"))
        except Exception:
            self.line_px = 16
        ttk.Style(root).configure("DOE.Treeview", rowheight=max(20, int(self.line_px * 1.4)))
        f = self.frame = ttk.Frame(parent, padding=10)
        ttk.Label(f, text="仿真 DOE：任务 × LLM 设置 × 重复次数，逐次在仿真里跑，成功与否按仿真真值判定（家居场景）",
                  font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        top = ttk.Frame(f)
        top.pack(fill="x", pady=(6, 4))
        # ---- task rows
        lt = ttk.LabelFrame(top, text="1. 任务（场景 + 任务句 + 结构化目标）", padding=6)
        lt.pack(side="left", fill="both", expand=True)
        rw = ttk.Frame(lt)
        rw.pack(fill="both", expand=True)
        self.tv_rows = ttk.Treeview(rw, columns=("n", "scene", "task", "goals"), show="headings", height=7,
                                    style="DOE.Treeview")
        for c, t, sample in (("n", "#", "T10"), ("scene", "场景", "seed:12345"), ("task", "任务句", "x" * 40),
                             ("goals", "目标", "x" * 30)):
            self.tv_rows.heading(c, text=t)
            self.tv_rows.column(c, width=self.px(sample), minwidth=self.px(t), stretch=(c in ("task", "goals")))
        ysb = ttk.Scrollbar(rw, orient="vertical", command=self.tv_rows.yview)
        self.tv_rows.configure(yscrollcommand=ysb.set)
        self.tv_rows.pack(side="left", fill="both", expand=True)
        ysb.pack(side="left", fill="y")
        self.tv_rows.bind("<Double-1>", lambda _e: self.edit_row())
        b = ttk.Frame(lt)
        b.pack(fill="x", pady=(4, 0))
        ttk.Button(b, text="加 H1-H5", command=self.add_fixed).pack(side="left")
        ttk.Label(b, text="  随机种子").pack(side="left")
        self.v_s0, self.v_s1 = tk.StringVar(value="1"), tk.StringVar(value="5")
        ttk.Spinbox(b, from_=0, to=99999, textvariable=self.v_s0, width=6).pack(side="left", padx=2)
        ttk.Label(b, text="到").pack(side="left")
        ttk.Spinbox(b, from_=0, to=99999, textvariable=self.v_s1, width=6).pack(side="left", padx=2)
        ttk.Button(b, text="加随机场景", command=self.add_seeds).pack(side="left", padx=(2, 8))
        ttk.Button(b, text="新建", command=lambda: self.edit_row(new=True)).pack(side="left")
        ttk.Button(b, text="编辑", command=self.edit_row).pack(side="left", padx=2)
        ttk.Button(b, text="删除", command=self.del_row).pack(side="left")
        # ---- models + design
        rt = ttk.Frame(top)
        rt.pack(side="left", fill="y", padx=(10, 0))
        lm = ttk.LabelFrame(rt, text="2. 要比较的 LLM（每行一组；每组都会跑全部任务）", padding=6)
        lm.pack(fill="x")
        self.lb_models = tk.Listbox(lm, height=4, width=40, exportselection=False)
        self.lb_models.pack(fill="x")
        # V4.11.1 MODEL PICKER: provider / model / effort chosen from lists (model also typed), then 「加入」
        pk = ttk.Frame(lm)
        pk.pack(fill="x", pady=(6, 0))
        self.prov_info = self._providers()
        self.v_pprov = tk.StringVar(value=self.prov_info.get("openai", {}).get("label", "OpenAI"))
        self.v_pmodel = tk.StringVar(value="")
        self.v_peff = tk.StringVar(value="high")
        ttk.Label(pk, text="提供方").grid(row=0, column=0, sticky="w")
        self.c_pprov = ttk.Combobox(pk, textvariable=self.v_pprov, state="readonly", width=14,
                                    values=[v["label"] for v in self.prov_info.values()])
        self.c_pprov.grid(row=0, column=1, sticky="w", padx=(4, 10))
        self.c_pprov.bind("<<ComboboxSelected>>", lambda _e: self._on_pick_provider())
        ttk.Label(pk, text="思考档位").grid(row=0, column=2, sticky="w")
        ttk.Combobox(pk, textvariable=self.v_peff, state="readonly", width=8, values=list(EFFORTS)).grid(
            row=0, column=3, sticky="w", padx=4)
        ttk.Label(pk, text="模型").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.c_pmodel = ttk.Combobox(pk, textvariable=self.v_pmodel, width=30)
        self.c_pmodel.grid(row=1, column=1, columnspan=3, sticky="we", padx=(4, 0), pady=(4, 0))
        mb = ttk.Frame(lm)
        mb.pack(fill="x", pady=(6, 0))
        ttk.Button(mb, text="加入", command=self.add_picked).pack(side="left")
        ttk.Button(mb, text="用仿真页当前设置", command=self.add_current_model).pack(side="left", padx=4)
        ttk.Button(mb, text="删除选中", command=self.del_model).pack(side="left")
        self.v_keys = tk.StringVar(value="")
        ttk.Label(lm, textvariable=self.v_keys, foreground="#555555", wraplength=self.px("x" * 52),
                  justify="left").pack(anchor="w", pady=(4, 0))
        self._on_pick_provider()
        ld = ttk.LabelFrame(rt, text="3. 设计", padding=6)
        ld.pack(fill="x", pady=(6, 0))
        self.v_rep, self.v_steps, self.v_tmo = tk.StringVar(value="3"), tk.StringVar(value="60"), tk.StringVar(value="20")
        self.v_shuffle, self.v_oseed = tk.BooleanVar(value=True), tk.StringVar(value="1")
        self.v_name = tk.StringVar(value="")
        for i, (lab, var, w) in enumerate((("每组重复次数", self.v_rep, 5), ("最大工具步数", self.v_steps, 5),
                                          ("每次超时（分钟）", self.v_tmo, 5))):
            ttk.Label(ld, text=lab).grid(row=i, column=0, sticky="w")
            ttk.Spinbox(ld, from_=1, to=(60 if var is self.v_steps else 500), textvariable=var, width=w, command=self.update_summary).grid(
                row=i, column=1, sticky="w", padx=4, pady=1)
        ttk.Checkbutton(ld, text="场景内随机顺序，顺序种子", variable=self.v_shuffle).grid(row=3, column=0, sticky="w")
        ttk.Entry(ld, textvariable=self.v_oseed, width=6).grid(row=3, column=1, sticky="w", padx=4)
        ttk.Label(ld, text="名称（可空）").grid(row=4, column=0, sticky="w")
        ttk.Entry(ld, textvariable=self.v_name, width=18).grid(row=4, column=1, sticky="w", padx=4)
        ttk.Label(ld, text="每次运行开始前桌面物品复位（手臂先回零位）；\n一个场景启动一次仿真；发散后重启仿真。",
                  foreground="#555555", justify="left").grid(row=5, column=0, columnspan=2, sticky="w", pady=(4, 0))
        # ---- controls
        cb = ttk.Frame(f)
        cb.pack(fill="x", pady=(4, 2))
        self.btn_start = ttk.Button(cb, text="开始 DOE", command=self.start)
        self.btn_pause = ttk.Button(cb, text="暂停（本次跑完后）", command=self.pause, state="disabled")
        self.btn_abort = ttk.Button(cb, text="停止", command=self.abort, state="disabled")
        for w_ in (self.btn_start, self.btn_pause, self.btn_abort):
            w_.pack(side="left", padx=(0, 6))
        ttk.Button(cb, text="保存计划", command=self.save_plan).pack(side="left", padx=(12, 2))
        ttk.Button(cb, text="载入计划", command=self.load_plan).pack(side="left")
        ttk.Button(cb, text="打开结果…", command=self.open_results).pack(side="left", padx=(12, 0))
        self.v_sum = tk.StringVar(value="")
        ttk.Label(cb, textvariable=self.v_sum, foreground="#333333").pack(side="left", padx=12)
        # ---- lamps + legend
        lf = ttk.LabelFrame(f, text="运行（每个点一次运行；鼠标停在点上看概要，点一下看截图和过程）", padding=6)
        lf.pack(fill="x")
        self.cv = tk.Canvas(lf, height=64, background="white", highlightthickness=0)
        self.cv.pack(fill="x")
        self.cv.bind("<Motion>", self._hover)
        self.cv.bind("<Button-1>", self._click)                     # V4.11.2: open the run (pictures + trace)
        leg = ttk.Frame(lf)
        leg.pack(fill="x")
        for k in ("pending", "running", "success", "partial", "fail", "invalid"):
            c = tk.Canvas(leg, width=14, height=14, highlightthickness=0)
            c.create_oval(2, 2, 12, 12, fill=STATUS_COLOR[k], outline="#555555")
            c.pack(side="left")
            ttk.Label(leg, text=STATUS_TEXT[k] + ("（闪烁）" if k == "running" else "")).pack(side="left", padx=(2, 10))
        self.v_hover = tk.StringVar(value="")
        ttk.Label(lf, textvariable=self.v_hover, foreground="#333333").pack(anchor="w")
        # ---- live summary + log
        bot = ttk.Panedwindow(f, orient="horizontal")
        bot.pack(fill="both", expand=True, pady=(6, 0))
        sf = ttk.LabelFrame(bot, text="实时统计（成功率只算可判定的运行）", padding=4)
        self.tv_sum = ttk.Treeview(sf, columns=("g", "n", "s", "p", "f", "i", "r", "ci"), show="headings", height=8,
                                   style="DOE.Treeview")
        for c, t, sample in (("g", "LLM 设置 / 任务", "openai:gpt-6.1-sol@high"), ("n", "可判定", "999"),
                             ("s", "成功", "999"), ("p", "部分", "999"), ("f", "失败", "999"), ("i", "无法判定", "999"),
                             ("r", "成功率", "100%"), ("ci", "95% 区间", "100%-100%")):
            self.tv_sum.heading(c, text=t)
            w = max(self.px(t), self.px(sample))
            self.tv_sum.column(c, width=w, minwidth=w if c != "g" else self.px(t), stretch=(c == "g"),
                               anchor="w" if c == "g" else "center")
        self.tv_sum.pack(fill="both", expand=True)
        bot.add(sf, weight=1)
        lgf = ttk.LabelFrame(bot, text="DOE 日志", padding=4)
        self.txt = tk.Text(lgf, height=10, wrap="none", font=("TkFixedFont", 9), state="disabled")
        sy = ttk.Scrollbar(lgf, orient="vertical", command=self.txt.yview)
        self.txt.configure(yscrollcommand=sy.set)
        self.txt.pack(side="left", fill="both", expand=True)
        sy.pack(side="left", fill="y")
        bot.add(lgf, weight=1)
        self._tick_id = None

    # ---- rows / models
    def _refresh_rows(self):
        self.tv_rows.delete(*self.tv_rows.get_children())
        for i, r in enumerate(self.rows):
            self.tv_rows.insert("", "end", iid=str(i), values=(f"T{i + 1}", r["scene"], r["task"], r["goals"]))
        self.update_summary()

    def _refresh_models(self):
        self.lb_models.delete(0, "end")
        for m in self.models:
            self.lb_models.insert("end", model_label(m))
        self.update_summary()
        try:
            self._update_keys()
        except Exception:
            pass

    def add_fixed(self):
        self.rows += default_rows()
        self._refresh_rows()

    def add_seeds(self):
        try:
            a, b = int(self.v_s0.get()), int(self.v_s1.get())
        except ValueError:
            return
        specs = []
        for k in range(min(a, b), max(a, b) + 1):
            try:
                specs.append(default_rows([f"seed:{k}"])[0])
            except Exception as exc:
                self.log(f"[WARN] [V4.11 DOE] seed {k}: no valid layout ({exc}) - skipped")
        self.rows += specs
        self._refresh_rows()

    def _sel_row(self):
        s = self.tv_rows.selection()
        return int(s[0]) if s else None

    def del_row(self):
        i = self._sel_row()
        if i is not None:
            del self.rows[i]
            self._refresh_rows()

    def edit_row(self, new=False):
        i = None if new else self._sel_row()
        if not new and i is None:
            return
        row = dict(self.rows[i]) if i is not None else {"scene": "fixed:H1", "task": "", "goals": ""}
        tk, ttk = self.tk, self.ttk
        dlg = tk.Toplevel(self.root)
        dlg.title("DOE 任务")
        dlg.transient(self.root)
        body = ttk.Frame(dlg, padding=10)
        body.pack(fill="both", expand=True)
        v_sc, v_goal = tk.StringVar(value=row["scene"]), tk.StringVar(value=row["goals"])
        ttk.Label(body, text="场景（fixed:H1 .. fixed:H5 或 seed:<数字>）").grid(row=0, column=0, sticky="w")
        ttk.Combobox(body, textvariable=v_sc, values=[f"fixed:H{k}" for k in range(1, 6)], width=16).grid(
            row=0, column=1, sticky="w", padx=4)
        ttk.Label(body, text="任务句（发给 LLM 的原话，可改写成不同说法）").grid(row=1, column=0, columnspan=2, sticky="w",
                                                                pady=(8, 0))
        txt = tk.Text(body, width=70, height=4, wrap="word")
        txt.insert("1.0", row["task"])
        txt.grid(row=2, column=0, columnspan=2, sticky="we")
        ttk.Label(body, text="结构化目标（判定用，不发给 LLM）：<物体> into|onto <目标>，多个用 ; 分开").grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Entry(body, textvariable=v_goal, width=70).grid(row=4, column=0, columnspan=2, sticky="we")
        msg = tk.StringVar(value="")
        ttk.Label(body, textvariable=msg, foreground="#c62828", wraplength=520, justify="left").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))

        def fill_default():
            try:
                d = default_rows([v_sc.get().strip()])[0]
            except Exception as exc:
                msg.set(f"场景无法生成：{exc}")
                return
            txt.delete("1.0", "end")
            txt.insert("1.0", d["task"])
            v_goal.set(d["goals"])
            msg.set("")

        def ok():
            r = {"scene": v_sc.get().strip(), "task": txt.get("1.0", "end").strip(), "goals": v_goal.get().strip()}
            probs = check_row(r)
            if probs:
                msg.set("；".join(probs))
                return
            if i is None:
                self.rows.append(r)
            else:
                self.rows[i] = r
            dlg.destroy()
            self._refresh_rows()

        bb = ttk.Frame(body)
        bb.grid(row=6, column=0, columnspan=2, sticky="e", pady=(8, 0))
        ttk.Button(bb, text="用场景默认任务", command=fill_default).pack(side="left", padx=(0, 16))
        ttk.Button(bb, text="确定", command=ok).pack(side="left")
        ttk.Button(bb, text="取消", command=dlg.destroy).pack(side="left", padx=(6, 0))
        dlg.grab_set()

    def add_model(self, text=None):
        t = (text if text is not None else self.v_model.get()).strip()
        if not t:
            return
        m = parse_model(t)
        if m["model"] and model_label(m) not in [model_label(x) for x in self.models]:
            self.models.append(m)
        self._refresh_models()

    def px(self, text):
        """V4.11.1: width of a text in the default font (+ padding), for column widths on any screen"""
        try:
            return int(self.root.tk.call("font", "measure", "TkDefaultFont", text)) + 20
        except Exception:
            return 8 * len(text) + 20

    def _providers(self):
        try:
            return dict(self.api["providers"]())
        except Exception:
            try:
                import NERO_V2_7_1_Agent as A
                return {k: {"label": v.get("label", k), "models": list(v.get("models") or [])}
                        for k, v in A.LLM_PROVIDERS.items()}
            except Exception:
                return {"openai": {"label": "OpenAI", "models": []}}

    def _pick_prov(self):
        lab = self.v_pprov.get()
        return next((k for k, v in self.prov_info.items() if v["label"] == lab), "openai")

    def _on_pick_provider(self):
        self.prov_info = self._providers()                 # the 仿真 page may have refreshed its model list
        ms = list(self.prov_info.get(self._pick_prov(), {}).get("models") or [])
        self.c_pmodel.configure(values=ms)
        if self.v_pmodel.get() not in ms:
            self.v_pmodel.set(ms[0] if ms else "")

    def add_picked(self):
        m = self.v_pmodel.get().strip()
        if m:
            self.add_model(f"{self._pick_prov()}:{m}@{self.v_peff.get() or 'high'}")

    def key_problems(self, models=None) -> List[str]:
        """providers of the plan without an API key (none typed / in the keyring / in the environment)"""
        provs = sorted({m["provider"] for m in (self.models if models is None else models)})
        try:
            have = self.api["key_env"](provs)
        except Exception:
            have = {}
        out = []
        for p in provs:
            try:
                import NERO_V2_7_1_Agent as A
                names = A.LLM_PROVIDERS.get(p, {}).get("key_env") or ()
            except Exception:
                names = ()
            if not any(n in have or os.environ.get(n) for n in names):
                out.append(p)
        return out

    def _update_keys(self):
        provs = sorted({m["provider"] for m in self.models})
        if not provs:
            self.v_keys.set("API key：用的是「仿真 (MuJoCo)」页存的 key（每个提供方各一个）")
            return
        miss = self.key_problems()
        self.v_keys.set("API key：" + "，".join(f"{self.prov_info.get(p, {}).get('label', p)} "
                                                + ("没有 key（到「仿真 (MuJoCo)」页选这个提供方，填写并保存）" if p in miss else "有")
                                                for p in provs))

    def add_current_model(self):
        try:
            d = self.api["llm_default"]()
        except Exception:
            return
        self.add_model(f"{d['provider']}:{d['model']}@{d['effort']}")

    def del_model(self):
        s = self.lb_models.curselection()
        if s:
            del self.models[int(s[0])]
            self._refresh_models()

    # ---- plan
    def plan(self) -> dict:
        def num(v, d):
            try:
                return int(float(v.get()))
            except ValueError:
                return d
        return {"name": self.v_name.get().strip(), "rows": [dict(r) for r in self.rows],
                "models": [dict(m) for m in self.models], "replicates": max(1, num(self.v_rep, 1)),
                "steps": max(10, min(60, num(self.v_steps, 60))), "run_timeout_min": max(2, num(self.v_tmo, 20)),
                "shuffle": bool(self.v_shuffle.get()), "order_seed": num(self.v_oseed, 1), "reset_each_run": True,
                "restart_after_blowup": True, "version": VERSION}

    def set_plan(self, p):
        self.rows = [dict(r) for r in p.get("rows") or []]
        self.models = [dict(m) for m in p.get("models") or []]
        self.v_rep.set(str(p.get("replicates", 3)))
        self.v_steps.set(str(p.get("steps", 60)))
        self.v_tmo.set(str(p.get("run_timeout_min", 20)))
        self.v_shuffle.set(bool(p.get("shuffle", True)))
        self.v_oseed.set(str(p.get("order_seed", 1)))
        self.v_name.set(str(p.get("name") or ""))
        self._refresh_rows()
        self._refresh_models()

    def update_summary(self):
        try:
            p = self.plan()
            n = len(expand(p))
            n_sc = len({r["scene"] for r in p["rows"]})
            h = estimate_s(p) / 3600.0
            self.v_sum.set(f"{len(p['rows'])} 个任务 × {len(p['models'])} 组 LLM × {p['replicates']} 次 = {n} 次运行，"
                           f"启动仿真 {n_sc} 次，粗估 {h:.1f} 小时（按每次 4 分钟）")
        except Exception:
            self.v_sum.set("")

    def save_plan(self):
        if not self.fd:
            return
        p = self.fd.asksaveasfilename(title="保存 DOE 计划", defaultextension=".json",
                                      initialdir=os.path.join(HERE, "logs", "doe"), filetypes=[("JSON", "*.json")])
        if p:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.plan(), f, ensure_ascii=False, indent=1)
            self.log(f"[V4.11 DOE] plan saved: {p}")

    def load_plan(self):
        if not self.fd:
            return
        p = self.fd.askopenfilename(title="载入 DOE 计划", initialdir=os.path.join(HERE, "logs", "doe"),
                                    filetypes=[("JSON", "*.json")])
        if p:
            with open(p, encoding="utf-8") as f:
                self.set_plan(json.load(f))
            self.log(f"[V4.11 DOE] plan loaded: {p}")

    def open_results(self):
        if not self.fd:
            return
        d = self.fd.askdirectory(title="选择 DOE 结果目录（logs/doe/<时间>）", initialdir=os.path.join(HERE, "logs", "doe"))
        if not d:
            return
        try:
            plan, runs = load_runs(d)
            self.view_dir = d
        except Exception as exc:
            self.log(f"[FAIL] [V4.11 DOE] {d}: {exc}")
            return
        with open(os.path.join(d, "report.md"), "w", encoding="utf-8") as f:
            f.write(report_md(plan, runs))
        with open(os.path.join(d, "runs.html"), "w", encoding="utf-8") as f:
            f.write(report_html(plan, runs))
        self.show_runs(runs)
        self.show_text("DOE 报告", report_md(plan, runs))

    # ---- run
    def log(self, s):
        self.txt.configure(state="normal")
        self.txt.insert("end", s + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def start(self):
        p = self.plan()
        probs = [f"T{i + 1}: {x}" for i, r in enumerate(p["rows"]) for x in check_row(r)]
        if not p["rows"]:
            probs.append("没有任务")
        if not p["models"]:
            probs.append("没有 LLM 设置")
        busy = self.api["busy"]()
        if busy:
            probs.append(busy)
        for p_ in self.key_problems(p["models"]):
            probs.append(f"提供方 {p_} 没有 API key：到「仿真 (MuJoCo)」页选这个提供方，填写 key 并保存（钥匙串）")
        if probs:
            if self.mb:
                self.mb.showerror("DOE 不能开始", "\n".join(probs[:12]))
            self.log("[FAIL] [V4.11 DOE] " + "; ".join(probs[:12]))
            return
        n = len(expand(p))
        if self.mb and not self.mb.askyesno("开始 DOE", f"{self.v_sum.get()}\n\n运行期间控制台会自动启动 / 停止仿真，"
                                                       "请不要手动操作仿真窗口。开始？"):
            return
        import NERO_V410_Sim_Bridge as BR
        env_keys = self.api["key_env"](sorted({m["provider"] for m in p["models"]}))
        sim = {"start": lambda spec, env: self.api["start"](spec, dict(env, **env_keys)), "stop": self.api["stop"],
               "running": self.api["running"], "ready": self.api.get("ready") or (lambda: True)}
        out = new_out_dir()
        self.runner = DOERunner(p, sim, out, BR.task_dir(), log=self.log, on_change=lambda r: self.draw())
        self.runner.start()
        self.log(f"[V4.11 DOE] {n} runs planned")
        self.draw()
        self.btn_start.configure(state="disabled")
        self.btn_pause.configure(state="normal", text="暂停（本次跑完后）")
        self.btn_abort.configure(state="normal")
        self._loop()

    def _loop(self):
        r = self.runner
        if r is None:
            return
        r.tick()
        self.blink = not self.blink
        self.draw(light=True)
        if r.finished:
            self.btn_start.configure(state="normal")
            self.btn_pause.configure(state="disabled")
            self.btn_abort.configure(state="disabled")
            self.draw()
            self.show_text("DOE 报告", report_md(r.plan, r.runs))
            return
        self.btn_pause.configure(text="继续" if r.state == "paused" else "暂停（本次跑完后）")
        self._tick_id = self.root.after(500, self._loop)

    def pause(self):
        r = self.runner
        if r is None:
            return
        if r.state == "paused" or r.pause_req:
            r.resume()
        else:
            r.pause()

    def abort(self):
        if self.runner and (not self.mb or self.mb.askyesno("停止 DOE", "停止 DOE？正在跑的任务会被 STOP，仿真会关闭；"
                                                                     "已完成的运行和报告保留。")):
            self.runner.abort()

    # ---- drawing
    def show_runs(self, runs):
        self.runner_view = runs
        self.draw()

    def _runs(self):
        return self.runner.runs if self.runner else getattr(self, "runner_view", [])

    def draw(self, light=False):
        runs = self._runs()
        cv = self.cv
        if not light:
            cv.delete("all")
            self.lamp_items = {}
            W = max(400, cv.winfo_width())
            per = max(1, (W - 10) // 16)
            rows_n = max(1, (len(runs) + per - 1) // per)
            cv.configure(height=min(220, 12 + 16 * rows_n))
            prev_scene = None
            for k, r in enumerate(runs):
                x, y = 8 + 16 * (k % per), 6 + 16 * (k // per)
                oid = cv.create_oval(x, y, x + 12, y + 12, fill=STATUS_COLOR[r["status"]],
                                     outline="#1b1b1b" if r["scene"] != prev_scene else "#666666",
                                     width=2 if r["scene"] != prev_scene else 1)
                prev_scene = r["scene"]
                self.lamp_items[oid] = k
            self._fill_summary(runs)
        for oid, k in self.lamp_items.items():
            r = runs[k]
            col = STATUS_COLOR[r["status"]]
            if r["status"] == "running":
                col = STATUS_COLOR["running"] if self.blink else "#a5d6a7"
            cv.itemconfigure(oid, fill=col)

    def _fill_summary(self, runs):
        self.tv_sum.delete(*self.tv_sum.get_children())
        groups = (summary_rows(runs, lambda r: "全部") + summary_rows(runs, lambda r: model_label(r["llm"]))
                  + summary_rows(runs, lambda r: f"T{r['row'] + 1} {r['scene']}"))
        for g in groups:
            ci = "-" if g["ci95"] is None else f"{_pct(g['ci95'][0])}–{_pct(g['ci95'][1])}"
            self.tv_sum.insert("", "end", values=(g["group"], g["judged"], g["success"], g["partial"], g["fail"],
                                                  g["invalid"], _pct(g["rate"]), ci))

    def _hover(self, e):
        runs = self._runs()
        for oid in self.cv.find_overlapping(e.x - 1, e.y - 1, e.x + 1, e.y + 1):
            k = self.lamp_items.get(oid)
            if k is not None and k < len(runs):
                r = runs[k]
                ev = r.get("eval") or {}
                self.v_hover.set(f"{r['id']}  T{r['row'] + 1} {r['scene']}  {model_label(r['llm'])}  第 {r['rep']} 次  "
                                 f"{STATUS_TEXT[r['status']]}"
                                 + (f"  {CLASS_TEXT.get(r.get('klass'), r.get('klass'))}" if r.get("klass") else "")
                                 + (f"  目标 {ev.get('n_ok')}/{ev.get('n_goals')}" if ev else "")
                                 + (f"  最后失败：{r['last_fail']}" if r.get("last_fail") else ""))
                return

    def _run_at(self, e):
        runs = self._runs()
        for oid in self.cv.find_overlapping(e.x - 1, e.y - 1, e.x + 1, e.y + 1):
            k = self.lamp_items.get(oid)
            if k is not None and k < len(runs):
                return runs[k]
        return None

    def _click(self, e):
        r = self._run_at(e)
        if r is not None:
            self.show_run(r)

    def show_run(self, r):
        """V4.11.2: one run - final-state pictures, goal check, tool calls, LLM text"""
        tk, ttk = self.tk, self.ttk
        base = self.runner.out if self.runner else getattr(self, "view_dir", "")
        w = tk.Toplevel(self.root)
        w.title(f"DOE {r['id']} - {STATUS_TEXT.get(r['status'], r['status'])}")
        top = ttk.Frame(w, padding=6)
        top.pack(fill="x")
        w._imgs = []
        for rel in r.get("snapshots") or []:
            try:
                img = tk.PhotoImage(file=os.path.join(base, rel))
                k = max(1, int(round(img.width() / 460.0)))
                img = img.subsample(k, k) if k > 1 else img
                w._imgs.append(img)
                cell = ttk.Frame(top)
                cell.pack(side="left", padx=4)
                ttk.Label(cell, image=img).pack()
                ttk.Label(cell, text=os.path.basename(rel)).pack()
            except Exception as exc:
                ttk.Label(top, text=f"{rel}: {exc}").pack(side="left")
        if not r.get("snapshots"):
            ttk.Label(top, text="没有截图：" + str(r.get("snapshot_why") or "这次运行还没结束")).pack(anchor="w")
        st, ev = r.get("stats") or {}, r.get("eval") or {}
        lines = [f"{r['id']}  T{r['row'] + 1} {r['scene']}  {model_label(r['llm'])}  第 {r['rep']} 次  "
                 f"{STATUS_TEXT.get(r['status'], r['status'])}  {CLASS_TEXT.get(r.get('klass'), r.get('klass') or '')}",
                 f"任务句：{r['task']}", f"目标：{r['goals']}", ""]
        for g in ev.get("goals") or []:
            lines.append(f"  [{'达成' if g.get('ok') else '未达成'}] {g.get('goal')}: {g.get('detail')}")
        if ev.get("held"):
            lines.append(f"  结束时仍在夹爪里：{', '.join(ev['held'])}")
        if st.get("error"):
            lines.append(f"错误：{st['error']}")
        lines.append("")
        lines.append(f"工具调用 {len(st.get('tools') or [])} 次，安全停止 {st.get('safety_stops', 0)} 次：")
        for t in st.get("tools") or []:
            lines.append(f"  #{t.get('step')} {t.get('tool')} -> {'ok' if t.get('ok') else t.get('reason')}"
                         + (f" (安全停止 {t['safety_stops']})" if t.get("safety_stops") else "")
                         + f"  {json.dumps(t.get('args') or {}, ensure_ascii=False)}")
        if st.get("texts"):
            lines.append("")
            lines.append("LLM 的文字：")
            lines += [f"  #{x.get('round')} {x.get('text')}" for x in st["texts"]]
        if st.get("final"):
            lines += ["", f"LLM 最后的回答：{st['final']}"]
        bot = ttk.Frame(w, padding=6)
        bot.pack(fill="both", expand=True)
        t = tk.Text(bot, width=140, height=22, wrap="word", font=("TkFixedFont", 9))
        sy = ttk.Scrollbar(bot, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=sy.set)
        t.insert("1.0", "\n".join(lines))
        t.configure(state="disabled")
        t.pack(side="left", fill="both", expand=True)
        sy.pack(side="left", fill="y")
        return w

    def show_text(self, title, text):
        tk, ttk = self.tk, self.ttk
        w = tk.Toplevel(self.root)
        w.title(title)
        t = tk.Text(w, width=120, height=40, wrap="none", font=("TkFixedFont", 9))
        sy = ttk.Scrollbar(w, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=sy.set)
        t.insert("1.0", text)
        t.configure(state="disabled")
        t.pack(side="left", fill="both", expand=True)
        sy.pack(side="left", fill="y")
