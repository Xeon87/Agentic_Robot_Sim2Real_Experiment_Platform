#!/usr/bin/env python3
"""NERO V4.8.3 TIDY DESK INTENT LAYER - "听懂弦外之音的桌面整理".

The user says something about their situation, not a list of moves ("我妈妈一会儿要来", "等下要开会",
"孩子要来玩了", "有点饿了，先把垃圾收一下").  This layer sits in front of the sort skills
(NERO_V372_Sort_Skills.SortSkills, scene tidy_desk = sort_items with the tidy layout, NERO original gripper):

  1. UNDERSTAND   the utterance -> an intent (guests / work / kids / snack / trash / leave / general) and a tidy
                  plan: for every item on the desk move (to which container) / keep (and why) / ask (a question with
                  options).  LLM (the tab's provider / model / key, JSON answer) or, without a key or when the LLM
                  fails, the offline rules below.  Explicit words win ("别动鼠标", "把饼干放进水果盘").
  2. VALIDATE     unknown items / containers, the gripper stroke (70 mm), container capacity, items in use
                  (the mouse) are only moved when the user says so, an item already in its container stays.
  3. PREFERENCES  remembered answers (NERO_CALIB_DIR = ~/Downloads/cameracalib/tidy_prefs.json, per intent and in general) answer the questions
                  of the next run ("按你上次的习惯").
  4. PREVIEW      a dialog: what the robot understood, every item with its planned place (editable), the questions,
                  "记住这次的选择"; nothing moves before 开始整理.
  5. EXECUTE      pick_object -> place_in_box per move (an item a container refuses is put back on the table,
                  return_item_to_table; an item whose grasp is blocked by its neighbours is retried at the end).
  6. REPORT       the whole desk is checked again (check_bins) and compared with the plan; a short Chinese report:
                  what went where, what stayed and why, what failed and why.

Entry points: gui_run_tidy(ctx, io, skills, cfg) from the simulation RUN TASK hook (NERO_V372 gui_run_task, tidy
layout), RealTidyFlow in NERO_V365 (real Agentic tab, scene tidy_desk), twin_tidy() / `python3 NERO_V483_Tidy.py
--twin "我妈妈一会儿要来"` offline in the MuJoCo twin.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time

VERSION = "V4.8.3"
MARKER = "V4.8.3 TIDY DESK INTENT LAYER"
KEEP = "keep"
STROKE_FREE_MM = 6.0

# ------------------------------------------------------------------------------------------------ vocabulary
ITEM_ALIASES = {
    "banana": ("香蕉",),
    "apple": ("苹果",),
    "orange": ("橙子", "橘子"),
    "crumpled paper": ("纸团", "废纸", "纸球", "纸"),
    "cookie pack": ("饼干", "零食", "曲奇"),
    "toy block": ("积木", "玩具"),
    "marker pen": ("马克笔", "记号笔", "笔"),
    "computer mouse": ("鼠标",),
}
BOX_ALIASES = {
    "fruit_bowl": ("水果盘", "果盘", "盘子", "碗"),
    "trash_bin": ("垃圾桶", "垃圾箱", "纸篓"),
    "pen_box": ("文具盒", "笔筒", "笔盒", "文具"),
    "toy_box": ("玩具箱", "玩具盒", "收纳箱"),
}
# intent id -> Chinese name, trigger words (checked in this order), the goal it implies
INTENTS = [
    ("kids", "孩子要来", ("孩子", "小孩", "宝宝", "儿子", "女儿", "侄子", "侄女", "外甥", "小朋友", "娃"),
     "孩子够得到的东西要安全：笔收好免得乱画，垃圾扔掉，水果收进盘子，玩具留给孩子玩"),
    ("guests", "有客人要来", ("客人", "妈妈", "爸爸", "爸妈", "父母", "朋友", "领导", "同事", "老师", "来访", "拜访",
                         "来家里", "要来", "过来", "体面", "整洁", "干净"),
     "桌面要整洁好看：东西各回各的容器，正在用的工具留着"),
    ("work", "要开会 / 工作", ("开会", "会议", "上课", "工作", "加班", "写东西", "写作业", "学习", "笔记本", "电脑",
                          "腾", "空出", "地方", "专心", "视频会"),
     "要腾出干净的工作区：吃的、垃圾、玩具都收走，文具和鼠标留着可能要用"),
    ("snack", "想吃东西", ("饿", "吃点", "零食", "下午茶", "嘴馋", "吃东西", "饭点"),
     "吃的留在手边，其余收好"),
    ("trash", "清垃圾", ("垃圾", "扔", "废纸"),
     "只清理垃圾，其他东西不动"),
    ("leave", "要离开 / 下班", ("下班", "走了", "出门", "回家", "睡觉", "明天见", "晚安", "离开", "放学"),
     "人走桌清：所有东西收好"),
    ("general", "整理桌面", ("收拾", "整理", "太乱", "乱", "清理"),
     "东西各回各的容器，拿不准的先问"),
]
INTENT_CN = {k: cn for k, cn, _w, _g in INTENTS}
INTENT_GOAL = {k: g for k, _cn, _w, g in INTENTS}
REASON_CN = {
    "GRASP_UNREACHABLE": "抓不到（够不着，或旁边的东西挡住了手指）",
    "PLACE_UNREACHABLE": "容器里找不到放得下、够得着的空位",
    "ITEM_NOT_LIFTED": "夹住后没有抬起来",
    "NO_TWO_FINGER_CONTACT": "两根手指没有都碰到它",
    "ITEM_NOT_IN_BOX": "松开后没落在容器里",
    "ITEM_TOO_WIDE": "比夹爪张开的宽度还宽",
    "MOTION_FAILED": "运动失败",
    "TARGET_UNREACHABLE": "目标位置够不着",
    "SAFETY_STOP": "安全停止",
    "SAFETY_LATCH_NOT_CLEARED": "安全锁没解除",
    "SKILL_EXCEPTION": "程序异常",
    "STOPPED": "被你停止",
}


# ------------------------------------------------------------------------------------------------ preferences
def prefs_path():
    # the calibration folder (as the real tab's CALIB_DIR): it outlives the version folders, so do the preferences
    d = os.environ.get("NERO_CALIB_DIR") or os.path.expanduser("~/Downloads/cameracalib")
    return os.path.join(d, "tidy_prefs.json")


def load_prefs(path=None):
    try:
        with open(path or prefs_path(), encoding="utf-8") as f:
            p = json.load(f)
        if isinstance(p, dict) and isinstance(p.get("rules"), dict):
            p.setdefault("log", [])
            return p
    except Exception:
        pass
    return {"version": 1, "rules": {}, "log": []}


def save_prefs(prefs, path=None):
    path = path or prefs_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(prefs, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
    return path


def pref_for(prefs, intent, label):
    """-> (decision, scope) or (None, None): the intent's own rule first, then the general one"""
    rules = (prefs or {}).get("rules") or {}
    for key, scope in ((f"{intent}|{label}", "intent"), (f"*|{label}", "general")):
        r = rules.get(key)
        if isinstance(r, dict) and r.get("decision"):
            return r["decision"], scope
    return None, None


def remember(prefs, plan, decisions, asked_only=False):
    """store the user's choices (answered questions + edited rows; asked_only: only the answers) per intent and in
    general -> list of (label, decision) stored"""
    out = []
    intent = plan.get("intent") or "general"
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    for st in plan["steps"]:
        d = decisions.get(st["item_id"])
        if d is None:
            continue
        planned = st.get("box") if st["action"] == "move" else (KEEP if st["action"] == "keep" else None)
        if st["action"] != "ask" and (asked_only or d == planned):
            continue
        for key in (f"{intent}|{st['label']}", f"*|{st['label']}"):
            r = prefs["rules"].get(key) or {}
            prefs["rules"][key] = {"decision": d, "n": int(r.get("n", 0)) + 1, "updated": now,
                                   "said": plan.get("utterance", "")[:80]}
        out.append((st["label"], d))
    prefs["log"] = (prefs.get("log") or [])[-19:] + [{"t": now, "said": plan.get("utterance", "")[:80],
                                                      "intent": intent, "stored": out}]
    return out


# ------------------------------------------------------------------------------------------------ desk snapshot
def desk_from_scene(scene, stroke_mm=70.0):
    """SortSkills.scene() -> the compact desk the intent layer works on"""
    items = []
    for it in scene.get("items") or []:
        items.append({"item_id": it["id"], "label": it["label"], "name_cn": it.get("name_cn") or it["label"],
                      "category": _category(it), "attributes": list(it.get("attributes") or []),
                      "width_mm": (it.get("size_mm") or {}).get("width"), "now": it.get("in_bin"),
                      "held": bool(it.get("held"))})
    boxes = []
    for b in scene.get("boxes") or []:
        boxes.append({"box": b["box"], "name_cn": b.get("name_cn") or b["box"], "accepts": list(b.get("accepts") or []),
                      "capacity": b.get("capacity"), "contains": list(b.get("contains") or []), "side": b.get("side")})
    return {"items": items, "boxes": boxes, "stroke_mm": float(stroke_mm)}


def _category(it):
    if it.get("category"):
        return it["category"]
    try:
        import NERO_V343_Scenes as SC
        for v in SC.SORT_ITEMS.values():
            if v["label"] == it.get("label"):
                return v.get("category")
    except Exception:
        pass
    return None


def _box_for_category(desk, cat):
    for b in desk["boxes"]:
        if cat in b["accepts"]:
            return b["box"]
    return None


def box_cn(desk, box):
    if box in (None, KEEP):
        return "留在桌上"
    return next((b["name_cn"] for b in desk["boxes"] if b["box"] == box), str(box))


def _names(it):
    return (it["name_cn"],) + tuple(ITEM_ALIASES.get(it["label"], ())) + (it["label"],)


# ------------------------------------------------------------------------------------------------ explicit words
_SPLIT = re.compile(r"[，,。.!！？?；;、\n]|然后|再|并且|而且|另外")


def explicit_orders(text, desk):
    """{item_id: decision} for what the user names directly: "别动鼠标" / "鼠标留着" -> keep, "把饼干放进水果盘" ->
    fruit_bowl, "纸团扔了" -> trash_bin"""
    out = {}
    t = str(text or "")
    for clause in [c for c in _SPLIT.split(t) if c.strip()]:
        hits = [it for it in desk["items"] if any(n and n in clause for n in _names(it))]
        if not hits:
            continue
        # the most specific name wins ("马克笔" over "笔")
        hits.sort(key=lambda it: -max(len(n) for n in _names(it) if n and n in clause))
        keep = re.search(r"(别|不要|不用|甭|先别|不许)(动|收|碰|拿|扔)|留着|留在|保留|放着别", clause)
        box = None
        for b in desk["boxes"]:
            if any(a in clause for a in (b["name_cn"],) + BOX_ALIASES.get(b["box"], ())):
                box = b["box"]
                break
        if box is None and re.search(r"扔(掉|了)?|丢(掉|了)?", clause) and not keep:
            box = next((b["box"] for b in desk["boxes"] if "trash" in b["accepts"]), None)
        decision = KEEP if keep else box
        if decision is None:
            continue
        top = hits[0]
        best = max(len(n) for n in _names(top) if n in clause)
        for it in hits:
            if max(len(n) for n in _names(it) if n in clause) == best:
                out[it["item_id"]] = decision
    return out


def detect_intent(text):
    t = str(text or "")
    for k, _cn, words, _g in INTENTS:
        if any(w in t for w in words):
            return k
    return "general"


# ------------------------------------------------------------------------------------------------ offline rules
def _rule(intent, it, desk):
    """-> (action, box, reason, question, options) for one item under an intent"""
    cat = it["category"]
    home = _box_for_category(desk, cat)
    cn = it["name_cn"]
    food_box = _box_for_category(desk, "fruit")
    if "in_use" in it["attributes"] or cat == "desk_tool":
        return KEEP, None, f"{cn}是你正在用的工具，留在原位", None, None
    if intent == "trash":
        if cat == "trash":
            return "move", home, "你说要清垃圾", None, None
        return KEEP, None, "这次只清垃圾，其他不动", None, None
    if intent == "snack":
        if cat in ("fruit", "snack"):
            return KEEP, None, f"你饿了，{cn}留在手边", None, None
    if intent == "work" and cat == "stationery":
        return KEEP, None, f"开会 / 工作可能要用{cn}，留在桌上", None, None
    if intent == "work" and cat == "snack":
        return KEEP, None, f"{cn}留着，忙的时候可以垫一下（水果盘也放不下了）", None, None
    if intent == "kids":
        if cat == "toy":
            return KEEP, None, f"{cn}留给孩子玩", None, None
        if cat == "stationery":
            return "move", home, f"{cn}有墨水，孩子容易拿去乱画，先收进{box_cn(desk, home)}", None, None
        if cat == "snack":
            return ("ask", None, "", f"孩子要来，{cn}要不要先收起来（等大人同意再给孩子吃）？",
                    [food_box, KEEP] if food_box else [KEEP])
    if cat == "trash" and "maybe_note" in it["attributes"] and intent not in ("leave",):
        return ("ask", None, "", f"{cn}是要扔的垃圾，还是可能记着东西的便条？",
                [home, KEEP] if home else [KEEP])
    if cat == "snack":
        if intent == "guests":
            return ("ask", None, "", f"客人要来，{cn}是收进{box_cn(desk, food_box)}，还是留在桌上招待？",
                    [food_box, KEEP] if food_box else [KEEP])
        if intent == "leave" and food_box:
            return "move", food_box, f"{cn}也是吃的，和水果一起收进{box_cn(desk, food_box)}，腾出桌面", None, None
        return ("ask", None, "", f"{cn}放哪儿？（没有专门放零食的容器）", [food_box, KEEP] if food_box else [KEEP])
    if home is None:
        return KEEP, None, f"没有放{cn}的容器，留在桌上", None, None
    why = {"guests": "客人要来，桌面要整洁", "work": "腾出工作区", "kids": "孩子要来，桌面收拾安全",
           "leave": "人走桌清", "snack": "吃的以外都收好"}.get(intent, "各回各的容器")
    return "move", home, f"{why}：{cn}放进{box_cn(desk, home)}", None, None


def rule_plan(text, desk, prefs=None):
    intent = detect_intent(text)
    steps = []
    for it in desk["items"]:
        act, box, why, q, opts = _rule(intent, it, desk)
        steps.append(_step(it, act, box, why, q, opts))
    return {"utterance": str(text), "intent": intent, "intent_cn": INTENT_CN.get(intent, intent),
            "understanding": INTENT_GOAL.get(intent, ""), "source": "rules", "steps": steps, "warnings": []}


def _step(it, action, box, reason, question=None, options=None):
    return {"item_id": it["item_id"], "label": it["label"], "name_cn": it["name_cn"], "category": it["category"],
            "now": it["now"], "action": action, "box": box if action == "move" else None, "reason": reason or "",
            "question": question, "options": [o for o in (options or []) if o], "pref": None, "explicit": False}


# ------------------------------------------------------------------------------------------------ LLM
LLM_SYSTEM = """You are the INTENT LAYER of a desk-tidying robot (NERO arm, two-finger gripper, 70 mm stroke).
The user does not give a list of moves - they say something about their situation (in Chinese). Read between the
lines: what does the situation imply for EACH item on the desk? Decide per item: move it into one container, keep it
on the desk, or ask the user (only when it really depends on something you cannot know; at most 2 questions).
Rules: an item with attribute "in_use" stays unless the user explicitly says to move it. Use a container only for
what it is meant for ("accepts") unless the situation clearly asks otherwise (say so in the reason). Respect
container capacity. Never invent items or containers. Reasons and questions in short, natural Chinese.
Answer with ONE JSON object only:
{"intent": one of kids|guests|work|snack|trash|leave|general,
 "understanding": "<one Chinese sentence: what the user means and what the desk should look like>",
 "steps": [{"item_id": "...", "action": "move"|"keep"|"ask", "box": "<container id or null>",
            "reason": "<Chinese>", "question": "<Chinese, only for ask>", "options": ["<container id>", "keep"]}]}"""


def _provider(provider):
    try:
        import NERO_V374_Vision_Review as VR
        return VR._provider_cfg(provider)
    except Exception:
        return provider or "openai", {"protocol": "responses", "api_base": "https://api.openai.com/v1",
                                      "key_env": ["OPENAI_API_KEY"], "label": "OpenAI"}


def post_text(provider, model, api_key, system, user, timeout_s=60.0):
    """one text request -> reply text (the tab's provider: OpenAI responses / chat-completions providers)"""
    import urllib.error
    import urllib.request
    prov, cfg = _provider(provider)
    key = (api_key or next((os.environ.get(e, "") for e in cfg["key_env"] if os.environ.get(e)), "")).strip()
    if not key:
        raise RuntimeError(f"{cfg.get('label', prov)} API key missing")
    if cfg["protocol"] == "responses":
        body = {"model": model, "instructions": system, "input": user, "max_output_tokens": 1500}
        url = cfg["api_base"] + "/responses"
    else:
        body = {"model": model, "max_tokens": 1500,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        url = cfg["api_base"] + "/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                          "User-Agent": "NERO-V4.8.3-tidy"})
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            obj = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:300]}")
    if cfg["protocol"] == "responses":
        texts = [c["text"] for it in obj.get("output") or [] for c in it.get("content") or []
                 if c.get("type") in ("output_text", "text") and c.get("text")]
        return "\n".join(texts) or obj.get("output_text") or ""
    return (((obj.get("choices") or [{}])[0].get("message")) or {}).get("content") or ""


def parse_json_reply(text):
    t = str(text or "").strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.M).strip()
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        raise ValueError("no JSON object in the reply")
    return json.loads(t[i:j + 1])


def llm_plan(text, desk, prefs, llm, post=None):
    user = json.dumps({"user_said": str(text), "desk_items": [
        {k: it[k] for k in ("item_id", "label", "name_cn", "category", "attributes", "width_mm", "now")}
        for it in desk["items"]],
        "containers": [{k: b[k] for k in ("box", "name_cn", "accepts", "capacity", "contains")} for b in desk["boxes"]],
        "remembered_preferences": {k: v.get("decision") for k, v in ((prefs or {}).get("rules") or {}).items()}},
        ensure_ascii=False)
    fn = post or post_text
    reply = fn(llm.get("provider") or "openai", llm.get("model") or "", llm.get("api_key") or "", LLM_SYSTEM, user)
    obj = parse_json_reply(reply)
    by_id = {it["item_id"]: it for it in desk["items"]}
    steps = []
    for s in obj.get("steps") or []:
        it = by_id.get(str(s.get("item_id")))
        if it is None or any(x["item_id"] == it["item_id"] for x in steps):
            continue
        act = str(s.get("action") or "").lower()
        act = act if act in ("move", KEEP, "ask") else KEEP
        steps.append(_step(it, act, s.get("box"), str(s.get("reason") or "")[:120],
                           str(s.get("question") or "")[:120] or None,
                           [KEEP if str(o).lower() in (KEEP, "none", "null", "desk") else str(o)
                            for o in (s.get("options") or [])]))
    intent = str(obj.get("intent") or "").lower()
    return {"utterance": str(text), "intent": intent if intent in INTENT_CN else detect_intent(text),
            "intent_cn": INTENT_CN.get(intent, INTENT_CN.get(detect_intent(text))),
            "understanding": str(obj.get("understanding") or "")[:200], "source": "llm", "steps": steps,
            "warnings": []}


# ------------------------------------------------------------------------------------------------ validate
def validate(plan, desk, prefs=None, explicit=None):
    """make the plan executable and honest: every desk item exactly once, explicit words first, then remembered
    preferences (they answer questions), container ids / stroke / capacity / in-use checks"""
    by_id = {it["item_id"]: it for it in desk["items"]}
    boxes = {b["box"]: b for b in desk["boxes"]}
    have = {s["item_id"]: s for s in plan["steps"] if s["item_id"] in by_id}
    if len(have) < len(by_id):                                # an item the LLM forgot -> the offline rule
        rp = rule_plan(plan.get("utterance", ""), desk)
        for s in rp["steps"]:
            if s["item_id"] not in have:
                have[s["item_id"]] = dict(s, reason=s["reason"] + "（按规则补上）")
    steps = [have[i] for i in by_id]
    warns = list(plan.get("warnings") or [])
    explicit = explicit or {}
    for st in steps:
        it = by_id[st["item_id"]]
        if st["item_id"] in explicit:
            d = explicit[st["item_id"]]
            st.update(action=KEEP if d == KEEP else "move", box=None if d == KEEP else d, explicit=True,
                      question=None, reason=f"你说了：{'留着' if d == KEEP else '放进' + box_cn(desk, d)}")
            continue
        dec, scope = pref_for(prefs, plan.get("intent"), st["label"])
        if dec is not None and (st["action"] == "ask" or scope == "intent"):
            if dec == KEEP or dec in boxes:
                st.update(action=KEEP if dec == KEEP else "move", box=None if dec == KEEP else dec, question=None,
                          pref=scope, reason=f"按你上次的习惯：{box_cn(desk, dec)}")
        if st["action"] == "move" and st.get("box") not in boxes:
            st.update(action="ask", question=st.get("question") or f"{st['name_cn']}放哪儿？",
                      options=[b for b in boxes] + [KEEP])
            warns.append(f"{st['name_cn']}：计划里的容器 {st.get('box')} 不存在，改成问你")
            st["box"] = None
        if st["action"] == "move" and ("in_use" in it["attributes"] or it["category"] == "desk_tool"):
            st.update(action=KEEP, box=None, reason=f"{st['name_cn']}正在用，没有你明确说就不动它")
        if st["action"] == "move" and it.get("width_mm") and it["width_mm"] + STROKE_FREE_MM > desk["stroke_mm"]:
            st.update(action=KEEP, box=None, reason=f"{st['name_cn']}宽 {it['width_mm']} mm，夹爪张不开这么大")
            warns.append(st["reason"])
        if st["action"] == "move" and st["box"] == it["now"]:
            st.update(action=KEEP, box=None, reason=f"{st['name_cn']}已经在{box_cn(desk, it['now'])}里了")
        if st["action"] == "ask":
            st["options"] = [o for o in (st.get("options") or []) if o == KEEP or o in boxes] or [KEEP]
            if not st.get("question"):
                st["question"] = f"{st['name_cn']}放哪儿？"
        elif st["action"] == "move":
            b = boxes[st["box"]]
            if it["category"] not in b["accepts"]:
                warns.append(f"{st['name_cn']}放进{b['name_cn']}（这个容器本来放 {'/'.join(b['accepts'])}）")
    # capacity (items already inside + planned moves)
    for k, b in boxes.items():
        cap = b.get("capacity")
        if not cap:
            continue
        n = len([i for i in b["contains"] if not any(s["item_id"] == i and s["action"] == "move" for s in steps)])
        for st in sorted(steps, key=lambda x: not x.get("explicit")):   # what you asked for gets the room first
            if st["action"] == "move" and st["box"] == k:
                n += 1
                if n > cap:
                    st.update(action="ask", box=None, question=f"{b['name_cn']}最多放 {cap} 件，已经满了：{st['name_cn']}放哪儿？",
                              options=[o for o in boxes if o != k] + [KEEP])
                    warns.append(f"{b['name_cn']}放不下{st['name_cn']}（容量 {cap}）")
    plan["steps"], plan["warnings"] = steps, warns
    return plan


def understand(text, desk, prefs=None, llm=None, post=None, log=print):
    """the whole intent step -> validated plan (LLM when a key / model is given and works, else the offline rules)"""
    prefs = prefs if prefs is not None else load_prefs()
    explicit = explicit_orders(text, desk)
    plan = None
    if llm and (llm.get("model") or post is not None):
        t0 = time.time()
        try:
            plan = llm_plan(text, desk, prefs, llm, post=post)
            log(f"[V4.8.3 TIDY] LLM 意图：{plan['intent']} · {plan['understanding']} ({time.time() - t0:.1f} s)")
        except Exception as exc:
            log(f"[WARN] [V4.8.3 TIDY] LLM 没给出可用的计划（{str(exc)[:160]}）-> 用离线规则")
            plan = None
    if plan is None:
        plan = rule_plan(text, desk, prefs)
        log(f"[V4.8.3 TIDY] 离线规则意图：{plan['intent_cn']} · {plan['understanding']}")
    return validate(plan, desk, prefs, explicit)


def initial_decisions(plan):
    """{item_id: box | keep | None (unanswered question)}"""
    out = {}
    for st in plan["steps"]:
        out[st["item_id"]] = st["box"] if st["action"] == "move" else (KEEP if st["action"] == KEEP else None)
    return out


def plan_text(plan, desk):
    lines = [f"你说：{plan['utterance']}", f"我理解：{plan['intent_cn']} - {plan['understanding']}"
             + ("（LLM）" if plan.get("source") == "llm" else "（离线规则）")]
    for st in plan["steps"]:
        if st["action"] == "ask":
            lines.append(f"  ? {st['name_cn']}：{st['question']}（可选：" + " / ".join(box_cn(desk, o) for o in st["options"]) + "）")
        else:
            lines.append(f"  - {st['name_cn']}：{box_cn(desk, st['box'] if st['action'] == 'move' else KEEP)}"
                         f" · {st['reason']}")
    for w in plan.get("warnings") or []:
        lines.append(f"  注意：{w}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ preview dialog
class TidyPreviewDialog:
    """modal preview: understanding, one row per item (editable place, the question for ask rows), remember box.
    .result = {"decisions": {item_id: box | keep}, "remember": bool} or None (cancelled).  No emoji (Ubuntu libXft)."""

    def __init__(self, parent, plan, desk, title="NERO 桌面整理 - 预览与确认"):
        import tkinter as tk
        from tkinter import ttk
        self.tk, self.plan, self.desk, self.result = tk, plan, desk, None
        self.top = top = tk.Toplevel(parent)
        top.title(title)
        top.transient(parent)
        f = ttk.Frame(top, padding=10)
        f.pack(fill="both", expand=True)
        ttk.Label(f, text=f"你说：{plan['utterance']}", font=("TkDefaultFont", 10, "bold"), wraplength=760).pack(anchor="w")
        ttk.Label(f, text=f"我理解：{plan['intent_cn']} - {plan['understanding']}"
                  + ("（LLM）" if plan.get("source") == "llm" else "（离线规则）"), wraplength=760).pack(anchor="w", pady=(2, 6))
        g = ttk.Frame(f)
        g.pack(fill="x")
        for c, h in enumerate(("物品", "现在在", "放到", "原因 / 问题")):
            ttk.Label(g, text=h, font=("TkDefaultFont", 9, "bold")).grid(row=0, column=c, sticky="w", padx=4)
        self.choices = {}
        self.vars = {}
        labels = [box_cn(desk, b["box"]) for b in desk["boxes"]] + [box_cn(desk, KEEP)]
        self.by_label = {box_cn(desk, b["box"]): b["box"] for b in desk["boxes"]}
        self.by_label[box_cn(desk, KEEP)] = KEEP
        dec0 = initial_decisions(plan)
        for r, st in enumerate(plan["steps"], start=1):
            ttk.Label(g, text=st["name_cn"]).grid(row=r, column=0, sticky="w", padx=4, pady=1)
            ttk.Label(g, text=box_cn(desk, st["now"]) if st["now"] else "桌上").grid(row=r, column=1, sticky="w", padx=4)
            v = tk.StringVar(value="" if dec0[st["item_id"]] is None else box_cn(desk, dec0[st["item_id"]]))
            vals = ([box_cn(desk, o) for o in st["options"]] if st["action"] == "ask" else labels)
            cb = ttk.Combobox(g, textvariable=v, values=vals, state="readonly", width=12)
            cb.grid(row=r, column=2, sticky="w", padx=4)
            txt = (f"问：{st['question']}" if st["action"] == "ask" else st["reason"]) + (
                "（记住的习惯）" if st.get("pref") else "")
            ttk.Label(g, text=txt, wraplength=440, foreground="#a04000" if st["action"] == "ask" else "").grid(
                row=r, column=3, sticky="w", padx=4)
            self.vars[st["item_id"]] = v
            self.choices[st["item_id"]] = cb
        for w in plan.get("warnings") or []:
            ttk.Label(f, text=f"注意：{w}", foreground="#a00000", wraplength=760).pack(anchor="w")
        self.v_remember = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="记住这次的回答和改动（下次同样的情况直接用，不再问）",
                        variable=self.v_remember).pack(anchor="w", pady=(8, 0))
        self.v_msg = tk.StringVar(value="")
        ttk.Label(f, textvariable=self.v_msg, foreground="#a00000").pack(anchor="w")
        b = ttk.Frame(f)
        b.pack(fill="x", pady=(6, 0))
        self.btn_ok = ttk.Button(b, text="开始整理", command=self.ok)
        self.btn_ok.pack(side="right")
        ttk.Button(b, text="取消", command=self.cancel).pack(side="right", padx=6)
        top.protocol("WM_DELETE_WINDOW", self.cancel)

    def decisions(self):
        return {iid: self.by_label.get(v.get()) for iid, v in self.vars.items()}

    def ok(self):
        d = self.decisions()
        miss = [st["name_cn"] for st in self.plan["steps"] if d.get(st["item_id"]) is None]
        if miss:
            self.v_msg.set("还有问题没回答：" + "、".join(miss))
            return
        self.result = {"decisions": d, "remember": bool(self.v_remember.get())}
        self.top.destroy()

    def cancel(self):
        self.result = None
        self.top.destroy()

    def show(self):
        try:
            self.top.grab_set()
        except Exception:
            pass
        self.top.wait_window()
        return self.result


# ------------------------------------------------------------------------------------------------ execute + report
class TidyRunner:
    """moves the items as decided (SortSkills tools), then checks the whole desk and writes the report"""

    def __init__(self, skills, log=print, stop_event=None):
        self.sk, self.log = skills, log
        self.stop = stop_event or threading.Event()

    def _tool(self, name, args=None):
        r = self.sk.run_tool(name, args or {})
        self.log(f"[V4.8.3 TIDY] {name}({', '.join(f'{k}={v}' for k, v in (args or {}).items())}) -> "
                 f"{'OK' if r.get('ok') else 'FAIL ' + str(r.get('reason'))}")
        return r

    def run(self, plan, decisions):
        t0 = time.time()
        res = {}
        moves = [(st, decisions.get(st["item_id"])) for st in plan["steps"]]
        moves = [(st, d) for st, d in moves if d not in (None, KEEP) and d != st["now"]]
        self._tool("observe_scene")
        todo, retry = list(moves), []
        for rnd in (0, 1):
            for st, box in todo:
                if self.stop.is_set():
                    res.setdefault(st["item_id"], {"ok": False, "box": box, "reason": "STOPPED"})
                    continue
                iid = st["item_id"]
                r = self._tool("pick_object", {"item_id": iid, "box": box})
                if not r.get("ok"):
                    reason = r.get("reason")
                    if rnd == 0 and reason in ("GRASP_UNREACHABLE", "ITEM_NOT_LIFTED", "NO_TWO_FINGER_CONTACT"):
                        retry.append((st, box))            # neighbours may be gone later / one more try
                    res[iid] = {"ok": False, "box": box, "reason": reason, "stage": "pick"}
                    continue
                r2 = self._tool("place_in_box", {"box": box})
                if r2.get("ok"):
                    res[iid] = {"ok": True, "box": box}
                    continue
                res[iid] = {"ok": False, "box": box, "reason": r2.get("reason"), "stage": "place"}
                if getattr(self.sk, "held", None):
                    r3 = self._tool("return_item_to_table")
                    res[iid]["returned_to_table"] = bool(r3.get("ok"))
            if not retry:
                break
            self.log(f"[V4.8.3 TIDY] 第二轮：回到安全位姿后再试 {len(retry)} 件（"
                     + "、".join(s["name_cn"] for s, _b in retry) + "）")
            if not getattr(self.sk, "held", None):
                self._tool("return_to_safe_pose")          # the grasp IK starts from the home posture again
            self._tool("observe_scene")
            todo, retry = retry, []
        chk = self._tool("check_bins")
        if not getattr(self.sk, "held", None):
            self._tool("return_to_safe_pose")
        rep = desk_report(plan, decisions, res, chk, self.sk, time.time() - t0)
        for line in rep["text"].splitlines():
            self.log(f"[V4.8.3 TIDY REPORT] {line}")
        return rep


def desk_report(plan, decisions, res, chk, skills, elapsed_s):
    """compare the whole desk (check_bins = ground truth / fresh perception) with the decisions -> report"""
    rows = ((chk or {}).get("physical_verification") or {}).get("items") or {}
    try:
        import NERO_V343_Scenes as SC
        bcn = SC.bin_label
    except Exception:
        def bcn(b):
            return str(b)
    by = {st["item_id"]: st for st in plan["steps"]}
    done, kept, failed, off = [], [], [], []
    for iid, st in by.items():
        want = decisions.get(iid)
        now = (rows.get(iid) or {}).get("in_bin", st["now"])
        if want in (None, KEEP):
            (kept if now == st["now"] else off).append((st, now))
        elif now == want:
            done.append((st, now))
        else:
            failed.append((st, now, (res.get(iid) or {}).get("reason")))
    lines = [f"整理结果（{plan['intent_cn']}）：{len(done)}/{len(done) + len(failed)} 件放好，用时 {elapsed_s / 60.0:.1f} 分钟"]
    by_box = {}
    for st, now in done:
        by_box.setdefault(now, []).append(st["name_cn"])
    for b, names in by_box.items():
        lines.append(f"  {bcn(b)}：" + "、".join(names))
    for st, now in kept:
        why = "你说留着" if st.get("explicit") else ("按你的回答" if st["action"] == "ask" else st["reason"])
        lines.append(f"  留在原处：{st['name_cn']}（{why}）")
    for st, now, reason in failed:
        lines.append(f"  没放好：{st['name_cn']} -> 想放进{bcn(decisions.get(st['item_id']))}，"
                     f"{REASON_CN.get(reason, reason or '原因不明')}；现在在" + (bcn(now) if now else "桌上"))
    for st, now in off:
        lines.append(f"  意外：{st['name_cn']}本该留着，现在在" + (bcn(now) if now else "桌上"))
    on_table = [st["name_cn"] for iid, st in by.items() if (rows.get(iid) or {}).get("in_bin", st["now"]) is None]
    lines.append("  桌上现在：" + ("、".join(on_table) if on_table else "没有东西了"))
    ok = not failed and not off
    if ok:
        lines.append("  桌面检查通过。")
    return {"ok": ok, "done": [s["item_id"] for s, _n in done], "kept": [s["item_id"] for s, _n in kept],
            "failed": [{"item": s["item_id"], "reason": r, "now": n} for s, n, r in failed],
            "unexpected": [s["item_id"] for s, _n in off], "elapsed_s": round(elapsed_s, 1), "text": "\n".join(lines)}


def save_report(rep, plan, decisions):
    d = os.path.join(os.path.dirname(prefs_path()), "tidy_reports")
    try:
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, time.strftime("tidy_%Y%m%d_%H%M%S.json"))
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"plan": plan, "decisions": decisions, "report": rep}, f, ensure_ascii=False, indent=1,
                      default=str)
        return path
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------ simulation entry
def _pump(root, fn):
    """run fn in a thread while the Tk main loop keeps drawing (called on the Tk thread)"""
    box = {}

    def work():
        try:
            box["v"] = fn()
        except Exception as exc:
            box["e"] = exc
    th = threading.Thread(target=work, daemon=True)
    th.start()
    while th.is_alive():
        try:
            root.update()
        except Exception:
            pass
        th.join(0.03)
    if "e" in box:
        raise box["e"]
    return box.get("v")


def gui_run_tidy(ctx, io, skills, cfg):
    """simulation RUN TASK (tidy_desk): intent + preview on the Tk thread, the robot run in the background"""
    import tkinter
    log = ctx["log"]
    root = getattr(tkinter, "_default_root", None)
    text = cfg.get("task") or ""
    desk = desk_from_scene(skills.scene(), stroke_mm=getattr(skills, "stroke_mm", 70.0))
    prefs = load_prefs()
    llm = {"provider": cfg.get("provider") or "openai", "model": cfg.get("model"), "api_key": cfg.get("api_key")}
    use_llm = bool(llm["model"])                     # no key / no network -> understand() falls back to the rules
    work = lambda: understand(text, desk, prefs, llm if use_llm else None, log=log)    # noqa: E731
    plan = _pump(root, work) if root is not None else work()
    for line in plan_text(plan, desk).splitlines():
        log(f"[V4.8.3 TIDY PLAN] {line}")
    if root is None:
        log("[FAIL] [V4.8.3 TIDY] no Tk window for the preview")
        return
    dlg = TidyPreviewDialog(root, plan, desk).show()
    if not dlg:
        log("[INFO] [V4.8.3 TIDY] 你取消了整理")
        return
    decisions = dlg["decisions"]
    if dlg["remember"]:
        st = remember(prefs, plan, decisions)
        save_prefs(prefs)
        if st:
            log("[V4.8.3 TIDY] 记住了：" + "、".join(f"{SC_label(l)} -> {d}" for l, d in st))
    if cfg.get("dry_run"):
        log("[INFO] [V4.8.3 TIDY] dry run：只预览，不动机械臂")
        return

    def _run():
        skills.v376_clear_start()
        ctx["start_check"](False)
        rep = TidyRunner(skills, log=io.emit).run(plan, decisions)
        p = save_report(rep, plan, decisions)
        if p:
            io.emit(f"[V4.8.3 TIDY] 报告已保存：{p}")
        return rep
    ctx["background"]("RUN TASK (tidy_desk)", _run, lock_motion=True)


def SC_label(label):
    try:
        import NERO_V343_Scenes as SC
        return SC.item_cn(label)
    except Exception:
        return label


# ------------------------------------------------------------------------------------------------ real robot
REAL_LAYOUT_NAME = "tidy_layout.json"
# YOLOE prompt words that name the same item as the scene label
KIND_ALIASES = (("computer mouse", "mouse"), ("marker pen", "marker"), ("crumpled paper", "paper ball"),
                ("crumpled paper", "paper"), ("cookie pack", "cookies"), ("cookie pack", "biscuits"),
                ("toy block", "block"), ("toy block", "building block"))
REAL_PROMPTS = "banana, apple, crumpled paper, cookie pack, toy block, marker pen, computer mouse"


def real_layout_path():
    return os.path.join(os.path.dirname(prefs_path()), REAL_LAYOUT_NAME)


def apply_real_layout(path=None, log=print):
    """the containers' real positions: NERO_CALIB_DIR/tidy_layout.json ({"containers": {id: {"centre": [x, y],
    "inner": [dx, dy], "wall_h": h}}}, robot base frame, metres).  Missing -> written with the simulation layout so
    it can be edited; the containers must stand where this file says (they are not detected)."""
    import NERO_V343_Scenes as SC
    SC.set_sort_layout("tidy")
    path = path or real_layout_path()
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        n = 0
        for k, c in (cfg.get("containers") or {}).items():
            if k in SC.SORT_BINS and isinstance(c, dict):
                b = SC.SORT_BINS[k] = dict(SC.SORT_BINS[k])
                for key in ("centre", "inner"):
                    if isinstance(c.get(key), (list, tuple)) and len(c[key]) == 2:
                        b[key] = (float(c[key][0]), float(c[key][1]))
                if c.get("wall_h") is not None:
                    b["wall_h"] = float(c["wall_h"])
                n += 1
        note = f"容器位置来自 {path}（{n} 个）"
    except FileNotFoundError:
        tmpl = {"note": "NERO V4.8.3 tidy desk: container positions in the robot base frame (m); robot faces -x, "
                        "y < 0 = robot's left. Edit to match your table.",
                "containers": {k: {"centre": list(b["centre"]), "inner": list(b["inner"]), "wall_h": b["wall_h"],
                                   "name_cn": b.get("cn")} for k, b in SC.SORT_BINS.items()}}
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(tmpl, f, ensure_ascii=False, indent=1)
        except Exception:
            pass
        note = f"没有 {path}：用默认容器位置（已写出模板，可改）"
    txt = "；".join(f"{b.get('cn', k)} 中心 ({b['centre'][0] * 100:.0f}, {b['centre'][1] * 100:.0f}) cm，"
                   f"内径 {b['inner'][0] * 100:.0f}x{b['inner'][1] * 100:.0f} cm" for k, b in SC.SORT_BINS.items())
    log(f"[V4.8.3 TIDY REAL] {note}：{txt}")
    return note


def yaw_from_contour(contour, K, T_base_cam, z_plane):
    """long-axis yaw (deg, base frame) of a detection mask outline projected onto the table plane; None for round /
    unclear outlines (elongation < 1.3)"""
    import numpy as np
    if not contour or len(contour) < 5 or not K:
        return None
    T = np.asarray(T_base_cam, float)
    pts = []
    for u, v in contour:
        d = np.array([(u - K["ppx"]) / K["fx"], (v - K["ppy"]) / K["fy"], 1.0])
        d = T[:3, :3] @ d
        if abs(d[2]) < 1e-6:
            continue
        s_ = (float(z_plane) - T[2, 3]) / d[2]
        if s_ > 0:
            pts.append(T[:3, 3] + s_ * d)
    if len(pts) < 5:
        return None
    P = np.asarray(pts)[:, :2]
    P = P - P.mean(axis=0)
    w, V = np.linalg.eigh(P.T @ P / len(P))
    if w[1] <= 0 or math_sqrt(w[1] / max(w[0], 1e-12)) < 1.3:
        return None
    ax = V[:, 1]
    import math
    return math.degrees(math.atan2(float(ax[1]), float(ax[0])))


def math_sqrt(x):
    import math
    return math.sqrt(max(0.0, float(x)))


def make_real_tidy_skills(io, log=print):
    """the sort skills on the real robot: items from the D435i + YOLOE detections (positions, mask yaw), the
    containers from tidy_layout.json, the original gripper's collision boxes (+ wrist camera bracket) vs table /
    containers, the gripper width deciding "held", a fresh camera view deciding "in the container"."""
    import numpy as np
    import NERO_V343_Scenes as SC
    import NERO_V372_Sort_Skills as SK
    import NERO_V375_Collision_Check as CC
    apply_real_layout(log=log)
    SK.sync_layout()
    try:
        import NERO_V4613_Tool_Collision as TC
        T_cam, _src = TC.find_handeye()
    except Exception:
        T_cam = None

    class RealTidySkills(SK.SortSkills):
        def __init__(self, io_):
            self.cam = {}
            self.table_z = 0.0
            super().__init__(io_, skill={"code": "TD", "short_name": "Tidy Desk (real)", "skill_id": "tidy_desk_real"})
            io_.state_check = self._state_check

        def _state_check(self, q, opening_mm=None, extra_parts=None):
            if opening_mm is None:
                opening_mm = (self.io.gripper_state() or {}).get("gap_mm") or 60.0
            return CC.assess(None, q, opening_mm, CC.world_boxes("sort_items"), extra_parts=extra_parts,
                             gripper="agx_gripper", T_flange_cam=T_cam)

        def ingest(self, res, merge=False):
            self.table_z = float(res.get("table_z_m") or 0.0)
            by_label = {v["label"]: k for k, v in SC.SORT_ITEMS.items()}
            by_label.update({a: by_label[l] for l, a in KIND_ALIASES if l in by_label})
            best = {}
            for d in res.get("items") or []:
                body = by_label.get(str(d.get("kind") or "").strip().lower())
                if body is None or (body in best and best[body]["conf"] >= float(d.get("conf") or 0.0)):
                    continue
                best[body] = d
            cam = res.get("_cam") or getattr(self.io, "last_cam", None) or {}
            new = {}
            for body, d in best.items():
                sp = SC.SORT_ITEMS[body]
                c = np.array([float(d["rim_xyz_m"][0]), float(d["rim_xyz_m"][1]), self.table_z + sp["cz"]])
                yaw = None
                if sp["shape"] != "round":
                    try:
                        yaw = yaw_from_contour(d.get("contour"), dict(cam.get("K") or {}), cam.get("T_base_cam"),
                                               self.table_z + sp["height"])
                    except Exception:
                        yaw = None
                new[body] = {"c": c, "yaw": 0.0 if yaw is None else yaw, "yaw_measured": yaw is not None,
                             "conf": float(d.get("conf") or 0.0)}
            if merge:
                self.cam.update(new)
            else:
                self.cam = new
            return new

        def observe_scene(self, **_):
            res = self.io.perceive_sort_items("tidy observe_scene")
            self.ingest(res)
            out = super().observe_scene()
            miss = [SC.item_cn(b) for b in SC.SORT_ITEM_NAMES if b not in self.cam]
            if miss:
                out.setdefault("warnings", []).append("没看到：" + "、".join(miss))
            return out

        def _item(self, iid):
            body = SK.ITEM_IDS[iid]
            e = self.cam[body]
            c = np.asarray(e["c"], float)
            # container test relative to the measured table height (the containers stand on it)
            in_bin = SC.item_in_bin([c[0], c[1], c[2] - self.table_z])
            return {"id": iid, "body": body, "spec": SC.SORT_ITEMS[body], "c": c, "yaw_deg": float(e["yaw"]),
                    "in_bin": in_bin, "up_z": 1.0}

        def _items(self):
            return {iid: self._item(iid) for iid, b in SK.ITEM_IDS.items() if b in self.cam}

        def _v376_sync_held(self):
            if not self.held:
                return None
            st = self.io.gripper_state() or {}
            gap = st.get("gap_mm")
            if gap is None or float(gap) > 3.0:
                return None
            iid, self.held, self.held_off = self.held, None, None
            self.io.emit(f"[WARN] [V4.8.3 TIDY REAL] {iid}: 夹爪已合到 {gap} mm - 东西掉了")
            return {"item": iid, "now_in_bin": None}

        def _measure_after_lift(self, item_id, z0, plan):
            st = self.io.gripper_state() or {}
            gap = st.get("gap_mm")
            w = float(self._item(item_id)["spec"]["width"]) * 1000.0
            held = gap is not None and max(3.0, w - 18.0) <= float(gap) <= w + 8.0
            c = self._item(item_id)["c"] + np.array([0.0, 0.0, self.lift_m])
            if held:
                self.cam[SK.ITEM_IDS[item_id]]["c"] = c
            return {"rise_m": self.lift_m if held else 0.0, "held": held, "c": c,
                    "reason": None if held else "NO_TWO_FINGER_CONTACT",
                    "extra": {"gripper_gap_mm": gap, "item_width_mm": round(w)}}

        def _where_after_place(self, iid, box, best):
            body = SK.ITEM_IDS[iid]
            sx, sy = best["slot"]
            self.cam[body]["c"] = np.array([sx, sy, self.table_z + SC.SORT_ITEMS[body]["cz"] + 0.004])
            try:
                self.io.go_home("V4.8.3 tidy: look at the container")
                res = self.io.perceive_sort_items(f"tidy check {iid}")
                seen = self.ingest(dict(res), merge=True)
                if body not in seen:                  # inside a container the camera may not see it -> the slot
                    self.cam[body] = {"c": np.array([sx, sy, self.table_z + SC.SORT_ITEMS[body]["cz"]]),
                                      "yaw": 0.0, "yaw_measured": False, "conf": 0.0, "unconfirmed": True}
                    self.io.emit(f"[INFO] [V4.8.3 TIDY REAL] {SC.item_cn(body)}放下后相机没看到（容器挡住了？）："
                                 "按放下的位置算")
            except Exception as exc:
                self.io.emit(f"[WARN] [V4.8.3 TIDY REAL] 放好后没能再看一眼：{str(exc)[:120]}")
            return self._item(iid)["in_bin"]

    return RealTidySkills(io)


def run_real_tidy(io, text, mode, llm, ask_preview, log=print, prefs_file=None):
    """real Agentic tab, scene tidy_desk.  mode plan: perceive + understand + preview + plan every grasp (no motion);
    hover: + the arm visits the pregrasp above every item to move (no descent, the gripper only opens);
    execute: the whole tidy run + report.  ask_preview(plan, desk) -> dialog result (on the Tk thread)."""
    sk = make_real_tidy_skills(io, log=log)
    io.skills = sk
    io.home_q = io.current_q()
    r = sk.run_tool("observe_scene", {})
    if not r.get("ok"):
        return {"ok": False, "reason": "OBSERVE_FAILED", "results": [r], "skills": sk}
    for w in r.get("warnings") or []:
        log(f"[WARN] [V4.8.3 TIDY REAL] {w}")
    desk = desk_from_scene(sk.scene(), stroke_mm=sk.stroke_mm)
    prefs = load_prefs(prefs_file)
    plan = understand(text, desk, prefs, llm if (llm or {}).get("model") else None, log=log)
    for st in plan["steps"]:
        e = sk.cam.get(SC_body(st["label"])) or {}
        if st["action"] == "move" and SC_long(st["label"]) and not e.get("yaw_measured", True):
            plan["warnings"].append(f"{st['name_cn']}的朝向没测出来（按 0 度抓）")
    for line in plan_text(plan, desk).splitlines():
        log(f"[V4.8.3 TIDY PLAN] {line}")
    dlg = ask_preview(plan, desk)
    if not dlg:
        log("[INFO] [V4.8.3 TIDY] 你取消了整理")
        return {"ok": False, "reason": "CANCELLED", "plan": plan, "skills": sk}
    decisions = dlg["decisions"]
    if dlg.get("remember"):
        stored = remember(prefs, plan, decisions)
        save_prefs(prefs, prefs_file)
        if stored:
            log("[V4.8.3 TIDY] 记住了：" + "、".join(f"{SC_label(l)} -> {d}" for l, d in stored))
    moves = [(st, decisions[st["item_id"]]) for st in plan["steps"]
             if decisions.get(st["item_id"]) not in (None, KEEP) and decisions[st["item_id"]] != st["now"]]
    if mode in ("plan", "hover"):
        rows = []
        q_now = io.current_q()
        for st, box in moves:
            it = sk._item(st["item_id"])
            pl = sk._plan_pick(it, q_now, box)
            rows.append((st, box, pl))
            log(f"[{'OK' if pl else 'FAIL'}] [V4.8.3 TIDY PLAN] {st['name_cn']} -> {box_cn(desk, box)}: "
                + (f"抓取方向 {pl['phi_deg']:.0f} 度，关节余量 {pl['margin']:.0f} 度" if pl else
                   "抓不到（" + json.dumps(getattr(sk, 'v375_rejects', [])[:2], ensure_ascii=False)[:160] + "）"))
        if mode == "hover":
            try:
                io.gripper_open(sk._cmd(60.0))
                for st, box, pl in rows:
                    if pl is None:
                        continue
                    io.check_stop()
                    rr = io.move_joints(f"V4.8.3 tidy hover above {st['name_cn']}", pl["q"]["pregrasp"])
                    log(f"[{'OK' if rr.get('ok') else 'FAIL'}] [V4.8.3 TIDY HOVER] {st['name_cn']} 上方 "
                        f"{SK_PRE_MM()} mm")
                    if not rr.get("ok"):
                        break
                    io.sleep(1.0)
            finally:
                io.go_home("V4.8.3 tidy hover back")
        ok = all(pl is not None for _s, _b, pl in rows)
        return {"ok": ok, "plan": plan, "decisions": decisions, "rows": rows, "skills": sk}
    rep = TidyRunner(sk, log=log, stop_event=getattr(io, "stop_ev", None)).run(plan, decisions)
    p = save_report(rep, plan, decisions)
    if p:
        log(f"[V4.8.3 TIDY] 报告已保存：{p}")
    return {"ok": rep["ok"], "plan": plan, "decisions": decisions, "report": rep, "skills": sk}


def SC_body(label):
    import NERO_V343_Scenes as SC
    return next((k for k, v in SC.SORT_ITEMS.items() if v["label"] == label), None)


def SC_long(label):
    import NERO_V343_Scenes as SC
    b = SC_body(label)
    return b is not None and SC.SORT_ITEMS[b]["shape"] != "round"


def SK_PRE_MM():
    import NERO_V372_Sort_Skills as SK
    return round(SK.PRE_M * 1000)


# ------------------------------------------------------------------------------------------------ offline twin
def twin_tidy(text, answers=None, llm=None, post=None, log=print, seed=None, prefs_file=None, remember_answers=False):
    """the whole flow in the MuJoCo twin; answers {label or item_id: box | keep} play the user in the preview"""
    import NERO_V343_Scenes as SC
    SC.set_sort_layout("tidy")
    import NERO_V372_Sort_Skills as SK
    io = SK.SortTwinIO(log=lambda s: None, seed=seed)
    sk = SK.SortSkills(io)
    desk = desk_from_scene(sk.scene(), stroke_mm=sk.stroke_mm)
    prefs = load_prefs(prefs_file)
    plan = understand(text, desk, prefs, llm, post=post, log=log)
    for line in plan_text(plan, desk).splitlines():
        log(f"[V4.8.3 TIDY PLAN] {line}")
    dec = initial_decisions(plan)
    for st in plan["steps"]:
        for k in (st["item_id"], st["label"], st["name_cn"]):
            if answers and k in answers:
                dec[st["item_id"]] = answers[k]
    for st in plan["steps"]:
        if dec[st["item_id"]] is None:
            dec[st["item_id"]] = KEEP                       # an unanswered question: leave it
            log(f"[V4.8.3 TIDY] 没回答「{st['question']}」-> {st['name_cn']}留着")
    if remember_answers:
        remember(prefs, plan, dec)
        save_prefs(prefs, prefs_file)
    rep = TidyRunner(sk, log=log).run(plan, dec)
    return plan, dec, rep, sk


if __name__ == "__main__":
    import sys
    if "--twin" in sys.argv:
        i = sys.argv.index("--twin")
        said = sys.argv[i + 1] if len(sys.argv) > i + 1 else "我妈妈一会儿要来"
        ans = {}
        for a in sys.argv:
            if "=" in a and not a.startswith("-"):
                k, v = a.split("=", 1)
                ans[k] = v
        _plan, _dec, _rep, _sk = twin_tidy(said, answers=ans)
        print(_rep["text"])
        sys.exit(0 if _rep["ok"] else 1)
    print(__doc__)
