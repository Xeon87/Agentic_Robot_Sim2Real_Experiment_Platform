#!/usr/bin/env python3
"""V3.42: task statistics + failure analysis, DeepSeek / Kimi providers, 60-step default with operator extension.

A1 provider selection: explicit GUI choice or model-ID prefix; per-provider API base and key variables
A2 DeepSeek / Kimi through the Chat Completions adapter (fake HTTP): tools converted, tool results sent back as
   role=tool, reasoning_content kept in the history (required by thinking models with tools), usage mapped
A3 end of task: [V3.42 TASK STATS] (total, LLM / skills / other, by skill, by object, tokens of this task);
   a failed task triggers exactly one extra no-tool LLM call -> [V3.42 FAILURE ANALYSIS]
A4 step budget: default 60 (no auto-raise); at the limit the operator callback is asked; +N continues the task,
   0 ends it as before (safe end)
"""
import io, json, os, sys, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import NERO_V2_7_1_Agent as A          # noqa: E402
import NERO_V326_Wrist_Camera as W     # noqa: E402
import hashlib                         # noqa: E402

assert W.VERSION in ("V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1") and (ROOT / "VERSION").read_text().strip() in ("V3.42", "V3.43", "V3.43.1", "V3.44", "V3.45", "V3.46", "V3.47", "V3.47.1")
assert hashlib.sha256((ROOT / "NERO_V308_Camera_Perception.py").read_bytes()).hexdigest().startswith("020cc6680f181872")

# ------------------------------------------------------------------ A1
assert A.provider_for_model("deepseek-v4-pro") == "deepseek" and A.provider_for_model("kimi-k2.6") == "kimi"
assert A.provider_for_model("gpt-5.6-luna") == "openai" and A.provider_for_model("gpt-5.6-luna", "kimi") == "kimi"
assert A.LLM_PROVIDERS["deepseek"]["api_base"].startswith("https://api.deepseek.com")
assert A.LLM_PROVIDERS["kimi"]["api_base"].startswith("https://api.moonshot.")
logs = []
ag = A.NeroLLMAgent(emit=logs.append, tool_runner=lambda n, a: {"ok": True})
for k in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "MOONSHOT_API_KEY", "KIMI_API_KEY"):
    os.environ.pop(k, None)
ok, txt = ag.api_key_status("", "deepseek-v4-pro")
assert not ok and "DEEPSEEK_API_KEY" in txt
os.environ["KIMI_API_KEY"] = "k"
assert ag.api_key_status("", "kimi-k2.6")[0]
print("[A1] provider from GUI choice / model prefix; DeepSeek, Kimi bases and key variables")

# ------------------------------------------------------------------ fake HTTP
sent = []
script = []


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_urlopen(req, timeout=None):
    body = json.loads(req.data.decode())
    sent.append((req.full_url, body))
    return FakeResp(json.dumps(script.pop(0)).encode())


urllib.request.urlopen = fake_urlopen


def chat(content=None, calls=(), reasoning="thinking...", pt=1000, ct=50, hit=600):
    m = {"role": "assistant", "content": content, "reasoning_content": reasoning}
    if calls:
        m["tool_calls"] = [{"id": f"c{i}", "type": "function",
                            "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
    return {"choices": [{"message": m}], "usage": {"prompt_tokens": pt, "completion_tokens": ct,
                                                   "prompt_cache_hit_tokens": hit}}


# ------------------------------------------------------------------ A2 + A3 (success)
os.environ["DEEPSEEK_API_KEY"] = "d"
tools_run = []


def runner(name, args):
    tools_run.append(name)
    if name == "scene_snapshot":
        return {"scene": {"objects_policy": "perceived_only", "objects": {}}}
    return {"ok": True, "tool": name, "state_after": {"stage": "PLACEMENT_VERIFIED"},
            "physical_verification": {"object_name": args.get("object_name")}, "object_name": args.get("object_name")}


logs.clear()
ag = A.NeroLLMAgent(emit=logs.append, tool_runner=runner)
script[:] = [chat(calls=[("observe_scene", {})]),
             chat(calls=[("verify_object_in_slot", {"object_name": "egg_1", "slot_name": "slot_1"})]),
             chat(content="done", reasoning="r3")]
ag.run_task("Pick up egg_1 and place it into the first empty egg-holder slot.", model="deepseek-v4-pro",
            reasoning_effort="low", max_tool_steps=60)
url, b1 = sent[0]
assert url == "https://api.deepseek.com/chat/completions"
assert b1["messages"][0]["role"] == "system" and b1["messages"][1]["role"] == "user"
assert b1["tools"][0]["type"] == "function" and "name" in b1["tools"][0]["function"]
assert b1["thinking"] == {"type": "enabled"} and b1["reasoning_effort"] == "low" and b1["max_tokens"] >= 8192
_, b2 = sent[1]
asst = [m for m in b2["messages"] if m["role"] == "assistant"][0]
assert asst["reasoning_content"] == "thinking..." and asst["tool_calls"][0]["id"] == "c0"
assert b2["messages"][-1] == {"role": "tool", "tool_call_id": "c0", "content": b2["messages"][-1]["content"]}
_, b3 = sent[2]
assert len([m for m in b3["messages"] if m["role"] == "assistant"]) == 2      # whole history replayed
assert len(sent) == 3                                                          # success: no failure analysis
st = [l for l in logs if l.startswith("[V3.42 TASK STATS]")]
assert st and "TOTAL TIME" in st[0] and "BY SKILL" in st[0] and "verify_object_in_slot" in st[0]
assert "total 3,150 = input 3,000 (cached 1,800) + output 150 | 3 API requests" in st[0], st[0]
assert not any("[V3.42 FAILURE ANALYSIS]" in l for l in logs)
print("[A2] DeepSeek chat adapter: tools converted, role=tool results, reasoning_content kept, history replayed")
print("[A3] success -> stats only:\n" + st[0])

# ------------------------------------------------------------------ A3 (failure) + Kimi
sent.clear()
logs.clear()
ag = A.NeroLLMAgent(emit=logs.append, tool_runner=lambda n, a: {"ok": False, "tool": n,
                                                                  "physical_verification": {"reason": "V303_SLOT_DESCENT_UNREACHABLE"}})
script[:] = [chat(calls=[("observe_scene", {})]), chat(content="Unable to complete: egg_1 failed."),
             chat(content="首要原因: 抓取姿态决定了放置不可达\n证据: orient_for_slot V303_SLOT_DESCENT_UNREACHABLE\n建议: 抓取时预检放置")]
ag.run_task("Pick up egg_1 and place it into the first empty egg-holder slot.", model="kimi-k2.6", max_tool_steps=60)
assert sent[0][0] == "https://api.moonshot.ai/v1/chat/completions" and "thinking" not in sent[0][1]
fa = sent[-1][1]
assert "tools" not in fa and "V303_SLOT_DESCENT_UNREACHABLE" in fa["messages"][-1]["content"]
assert any("[V3.42 FAILURE ANALYSIS] (" in l and "首要原因" in l for l in logs)
assert any("TASK SUMMARY RED" in l for l in logs)
print("[A3] failure -> one extra Kimi call without tools:", [l for l in logs if "首要原因" in l][0].split("\n", 1)[1])

# ------------------------------------------------------------------ A4 step budget + extension
asked = []


def run_with(answer, n_calls):
    sent.clear()
    logs.clear()
    asked.clear()
    ag = A.NeroLLMAgent(emit=logs.append, tool_runner=runner)
    ag.step_extension_cb = lambda info: (asked.append(info), answer)[1]
    script[:] = [chat(calls=[("observe_scene", {})]) for _ in range(n_calls)] + [chat(content="done")] + \
        [chat(content="首要原因: 步数不够")]
    try:
        ag.run_task("Pick up egg_1 and place it into the first empty egg-holder slot.", model="deepseek-flash",
                    max_tool_steps=3)
        return True
    except RuntimeError as exc:
        return str(exc)


r = run_with(5, 5)
assert r is True and len(asked) == 1 and asked[0]["steps_limit"] == 3 and asked[0]["steps_used"] == 3
assert any("[V3.42 STEPS ADDED] +5 -> 8" in l for l in logs)
r = run_with(0, 5)
assert "exceeded max_tool_steps=3" in r and len(asked) == 1
assert any("[V3.42 STEPS NOT ADDED]" in l for l in logs)
assert any("[V3.42 FAILURE ANALYSIS] (" in l for l in logs)
B = (ROOT / "NERO_V2_7_1_Unified_Backend.py").read_text()
for mk in ('value=os.environ.get("NERO_V342_DEFAULT_STEPS", "60")', "def llm_step_extension_request(info):",
           "llm_agent.step_extension_cb = llm_step_extension_request", 'text="Provider"', "_v342_on_provider",
           '"允许增加 / Allow"'):
    assert mk in B, mk
print("[A4] budget 60 by default; at the limit the operator is asked: +5 -> task continues; 0 -> safe end + analysis")
print("ALL V3.42 TESTS PASSED")
