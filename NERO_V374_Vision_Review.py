#!/usr/bin/env python3
"""NERO V3.74 LLM VISION REVIEW - a vision-capable LLM looks at the fixed camera image and says WHAT each detection is.

V3.74 LLM VISION REVIEW

Division of labour (user, V3.74): YOLOE finds the instances, the depth geometry only MEASURES them (position, size,
tilt, height); what an instance is - a single upright cup, two cups already nested, not a cup at all, a cup held by
the gripper - is answered by a vision LLM that sees the image with the numbered YOLOE outlines.

  review(rgb, dets, context)   annotated JPEG (outline + "#n" per detection) + the measured numbers as text ->
                               the provider's API (OpenAI Responses: input_image; DeepSeek / Kimi chat:
                               image_url) -> JSON {"objects": [{"id", "what", "upright", "in_gripper", "note"}]}
  apply_review(res, review)    sets each detection's kind from the answer (cup / stack / not_cup / lifted) and keeps
                               the geometry; every disagreement with the geometry is listed for the log
Calls are cached by the detection layout (count, kinds, positions to 1.5 cm) so a perception of an unchanged table
does not ask again.  No key / a model without image input / a timeout -> the review is skipped with the reason, and
the geometry result is used unchanged (nothing falls back silently).
The image is sent to the selected provider's API (the user asked for this, V3.74); nothing is stored by this module.
"""
import base64
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np                           # noqa: E402

VERSION = "V3.74"
MARKER = "V3.74 LLM VISION REVIEW"
WHAT_TO_KIND = {"cup": "cup", "nested_cups": "stack", "not_a_cup": "not_cup", "unsure": None}
REVIEW_TAG_PREFIXES = ("observe_scene", "识别杯子", "locate", "check ", "verify")   # key moments only
CACHE_S = 90.0
MAX_IMAGE_W = 960

PROMPT = """You review the detections of a robot's fixed overhead RGB camera (NERO arm, SO-101 two-finger gripper).
Each detection of the object detector is outlined and labelled "#n" in the image. The detector was asked for: {query}.
Measured by depth geometry (trust these numbers for position and size, not for identity):
{rows}
Task context: {context}
For EVERY "#n" answer what it is in the image:
  what = "cup" (one single upright cup), "nested_cups" (two or more cups already stacked one inside the other, seen as
  one outline), "not_a_cup" (something else), or "unsure";
  upright = true / false;  in_gripper = true only if the gripper is holding it (lifted, between the jaws).
Reply with JSON only: {{"objects": [{{"id": "#1", "what": "cup", "upright": true, "in_gripper": false, "note": "short"}}],
"scene": "one short sentence"}}"""


def _cv2():
    import NERO_V347_Yolo_Perception as YP
    return YP._cv2()


def annotate(rgb, dets, max_w=MAX_IMAGE_W):
    """RGB image with every detection's outline + "#n" label -> JPEG bytes (scaled to max_w)"""
    cv2 = _cv2()
    img = np.ascontiguousarray(np.asarray(rgb, np.uint8)[:, :, ::-1])      # BGR for cv2
    for e in dets or []:
        pts = np.asarray(e.get("contour") or [], np.int32)
        if len(pts) >= 3:
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, (0, 255, 255), 2)
            top = pts[np.argmin(pts[:, 1])]
            org = (int(top[0]) - 12, max(18, int(top[1]) - 8))
            cv2.putText(img, str(e.get("det_id", "?")), org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(img, str(e.get("det_id", "?")), org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    h, w = img.shape[:2]
    if w > max_w:
        img = cv2.resize(img, (max_w, int(round(h * max_w / float(w)))))
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def rows_text(dets):
    rows = []
    for e in dets or []:
        xy = e.get("base_xyz_m") or e.get("held_base_xyz_m")
        rows.append(f"  {e.get('det_id')}: detector conf {float(e.get('conf', 0)):.2f}"
                    + (f", centre ({xy[0]:+.3f}, {xy[1]:+.3f}) m" if xy else "")
                    + (f", height {e['height_mm']:.0f} mm" if e.get("height_mm") is not None else "")
                    + (f", lowest point off the table" if e.get("kind") == "lifted" else "")
                    + (f", geometry note: {e['reason']}" if e.get("reason") else ""))
    return "\n".join(rows) or "  (no detections)"


def layout_signature(dets):
    out = []
    for e in dets or []:
        xy = e.get("base_xyz_m") or e.get("held_base_xyz_m") or [0.0, 0.0]
        out.append((e.get("det_id"), round(float(xy[0]) / 0.015), round(float(xy[1]) / 0.015),
                    e.get("kind") == "lifted"))
    return tuple(sorted(out))


def parse_reply(text):
    """first JSON object in the model's text -> dict(objects=[...], scene=str) or None"""
    if not text:
        return None
    m = re.search(r"\{.*\}", str(text), re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("objects"), list):
        return None
    items = {}
    for o in obj["objects"]:
        if isinstance(o, dict) and o.get("id"):
            items[str(o["id"]).strip()] = {"what": str(o.get("what", "unsure")).strip().lower(),
                                           "upright": o.get("upright"), "in_gripper": bool(o.get("in_gripper", False)),
                                           "note": str(o.get("note", ""))[:160]}
    return {"items": items, "scene": str(obj.get("scene", ""))[:300]}


def _provider_cfg(provider):
    import NERO_V2_7_1_Agent as AG
    prov = provider if provider in AG.LLM_PROVIDERS else "openai"
    cfg = dict(AG.LLM_PROVIDERS[prov])
    if prov == "openai":
        cfg["api_base"] = os.environ.get("NERO_V374_OPENAI_API_BASE", AG.DEFAULT_API_BASE).rstrip("/")
    return prov, cfg


def post_vision(provider, model, api_key, prompt, jpeg, timeout_s=60.0):
    """one request with the image -> reply text (raises RuntimeError with the provider's message)"""
    prov, cfg = _provider_cfg(provider)
    key = (api_key or next((os.environ.get(e, "") for e in cfg["key_env"] if os.environ.get(e)), "")).strip()
    if not key:
        raise RuntimeError(f"{cfg['label']} API key missing")
    url_img = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")
    if cfg["protocol"] == "responses":
        body = {"model": model, "input": [{"role": "user", "content": [
            {"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": url_img}]}],
            "max_output_tokens": 800}
        url = cfg["api_base"] + "/responses"
    else:
        body = {"model": model, "max_tokens": 800, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": url_img}}]}]}
        url = cfg["api_base"] + "/chat/completions"
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                          "User-Agent": "NERO-V3.74-vision-review"})
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
            obj = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{cfg['label']} HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:400]}")
    except Exception as exc:
        raise RuntimeError(f"{cfg['label']} request failed: {exc}")
    if cfg["protocol"] == "responses":
        texts = []
        for it in obj.get("output") or []:
            for c in it.get("content") or []:
                if c.get("type") in ("output_text", "text") and c.get("text"):
                    texts.append(c["text"])
        return "\n".join(texts) or obj.get("output_text") or ""
    return (((obj.get("choices") or [{}])[0].get("message")) or {}).get("content") or ""


class VisionReviewer:
    """cfg_fn() -> {"provider", "model", "api_key"}; post = post_vision (tests inject a stand-in)"""

    def __init__(self, cfg_fn, log=print, post=None, query="cup"):
        self.cfg_fn, self.log, self.post, self.query = cfg_fn, log, post or post_vision, query
        self._cache = None
        self.calls, self.skipped = 0, 0
        self._warned = set()

    def wanted(self, tag):
        return str(tag).startswith(REVIEW_TAG_PREFIXES)

    def review(self, rgb, dets, context=""):
        sig = layout_signature(dets)
        if self._cache and self._cache[0] == sig and time.time() - self._cache[1] < CACHE_S:
            return dict(self._cache[2], cached=True)
        cfg = self.cfg_fn() or {}
        if not cfg.get("model"):
            return {"ok": False, "error": "no model selected"}
        t0 = time.perf_counter()
        try:
            jpeg = annotate(rgb, dets)
            prompt = PROMPT.format(query=self.query, rows=rows_text(dets), context=context or "-")
            text = self.post(cfg.get("provider", "openai"), cfg["model"], cfg.get("api_key", ""), prompt, jpeg)
        except Exception as exc:
            msg = str(exc)[:300]
            if msg not in self._warned:
                self._warned.add(msg)
                self.log(f"[WARN] [V3.74 VISION] LLM 看图复核没做：{msg}。原因可能是 API key / 模型不支持图片输入 / 网络；"
                         "本次用几何结果（只测位置尺寸，不判断叠没叠）。下一步：换支持图片的模型，或检查 key")
            return {"ok": False, "error": msg}
        self.calls += 1
        parsed = parse_reply(text)
        if parsed is None:
            self.log(f"[WARN] [V3.74 VISION] 看图复核的回答不是预期的 JSON：{str(text)[:200]}")
            return {"ok": False, "error": "unparsable reply", "raw": str(text)[:500]}
        out = dict(parsed, ok=True, elapsed_s=round(time.perf_counter() - t0, 2), model=cfg["model"])
        self._cache = (sig, time.time(), out)
        return out


def apply_review(res, review):
    """detection kinds from the review (geometry kept) -> list of disagreement texts"""
    notes = []
    if not review or not review.get("ok"):
        return notes
    for e in res.get("dets") or []:
        r = (review.get("items") or {}).get(str(e.get("det_id")))
        if not r:
            continue
        e["vision"] = r
        geo = e.get("kind")
        if geo in ("edge", "outside", "no_depth", "rejected"):
            continue                                  # position facts the image cannot overrule
        if r.get("in_gripper"):
            new = "lifted"
        else:
            new = WHAT_TO_KIND.get(r.get("what"))
            if geo == "lifted" and new is not None:
                notes.append(f"{e.get('det_id')}: 几何=离开桌面，看图=不在夹爪里 → 按看图（{r.get('what')}）")
        if new is None or new == geo:
            continue
        if not (geo == "lifted" and new in ("cup", "stack")):
            notes.append(f"{e.get('det_id')}: 几何={geo} → 看图={r.get('what')}"
                         + (f"（{r['note']}）" if r.get("note") else ""))
        e["kind"] = new
        if new == "lifted":
            e.setdefault("reason", "看图复核：在夹爪里")
        elif new == "not_cup":
            e["reason"] = "看图复核：不是杯子" + (f"（{r['note']}）" if r.get("note") else "")
    res["cups"] = [e for e in res.get("dets") or [] if e.get("base_xyz_m") and e.get("kind") != "not_cup"]
    res["vision"] = {"scene": review.get("scene"), "model": review.get("model"), "cached": bool(review.get("cached")),
                     "elapsed_s": review.get("elapsed_s")}
    return notes
