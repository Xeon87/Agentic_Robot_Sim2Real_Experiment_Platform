#!/usr/bin/env python3
"""V4.8.3 GUI test: the tidy preview dialog (needs tkinter + DISPLAY; skips otherwise).
P1 one row per desk item with its place; the question rows start empty and only offer their options.
P2 「开始整理」 with an unanswered question is refused with the item named; answered -> decisions + remember flag.
P3 an edited row (marker -> 留在桌上) is returned; 取消 -> None.
P4 the window fits 1400 x 900; no emoji in any widget text (Ubuntu 22.04 libXft).
P5 the real Agentic tab offers ④ 桌面整理 and its examples (scene switch fills the YOLOE prompt).
"""
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    print("[SKIP] test_v4_8_3_gui: no tkinter"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[SKIP] test_v4_8_3_gui: no DISPLAY"); sys.exit(0)
os.environ["NERO_CALIB_DIR"] = tempfile.mkdtemp(prefix="nero_v483_gui_")
import NERO_V343_Scenes as SC            # noqa: E402
import NERO_V483_Tidy as TD              # noqa: E402

SC.set_sort_layout("tidy")
ITEMS = [{"item_id": f"item_{i + 1}", "label": v["label"], "name_cn": v["cn"], "category": v["category"],
          "attributes": v.get("attrs", []), "width_mm": round(v["width"] * 1000), "now": None, "held": False}
         for i, v in enumerate(SC.SORT_ITEMS.values())]
DESK = {"items": ITEMS, "stroke_mm": 70.0,
        "boxes": [{"box": k, "name_cn": b["cn"], "accepts": b["accepts"], "capacity": b["capacity"], "contains": [],
                   "side": b["side"]} for k, b in SC.SORT_BINS.items()]}
EMOJI = [chr(c) for c in list(range(0x1F300, 0x1FAFF)) + list(range(0x2600, 0x27C0))]


def texts(w, out):
    for k in ("text",):
        try:
            out.append(str(w.cget(k)))
        except Exception:
            pass
    try:
        out.append(str(w.get()))
    except Exception:
        pass
    for c in w.winfo_children():
        texts(c, out)
    return out


root = tk.Tk()
root.geometry("1400x900")
plan = TD.understand("我妈妈一会儿要来", DESK, {"rules": {}, "log": []}, None, log=lambda s: None)
by = {st["label"]: st for st in plan["steps"]}
dlg = TD.TidyPreviewDialog(root, plan, DESK)
root.update()
# P1
assert set(dlg.vars) == {it["item_id"] for it in ITEMS}
ck = by["cookie pack"]["item_id"]
assert dlg.vars[ck].get() == "" and list(dlg.choices[ck].cget("values")) == ["水果盘", "留在桌上"], dlg.choices[ck].cget("values")
assert dlg.vars[by["banana"]["item_id"]].get() == "水果盘"
assert dlg.vars[by["computer mouse"]["item_id"]].get() == "留在桌上"
# P2
dlg.ok()
assert dlg.result is None and "饼干" in dlg.v_msg.get() and "纸团" in dlg.v_msg.get(), dlg.v_msg.get()
dlg.vars[ck].set("留在桌上")
dlg.vars[by["crumpled paper"]["item_id"]].set("垃圾桶")
# P3
dlg.vars[by["marker pen"]["item_id"]].set("留在桌上")
# P4
root.update()
dlg.top.update_idletasks()
assert dlg.top.winfo_reqwidth() <= 1400 and dlg.top.winfo_reqheight() <= 900, (dlg.top.winfo_reqwidth(), dlg.top.winfo_reqheight())
all_text = "".join(texts(dlg.top, []))
assert not [e for e in EMOJI if e in all_text], [e for e in EMOJI if e in all_text]
dlg.ok()
r = dlg.result
assert r and r["remember"] is True
assert r["decisions"][ck] == "keep" and r["decisions"][by["crumpled paper"]["item_id"]] == "trash_bin"
assert r["decisions"][by["marker pen"]["item_id"]] == "keep"
assert r["decisions"][by["banana"]["item_id"]] == "fruit_bowl"
dlg2 = TD.TidyPreviewDialog(root, plan, DESK)
root.update()
dlg2.cancel()
assert dlg2.result is None
# P5
import NERO_V365_Real_Cup_Task as RT     # noqa: E402
assert ("tidy_desk", "④ 桌面整理 tidy_desk（听懂弦外之音 · 原装夹爪）") in RT.TASK_SCENES
assert RT.SCENE_EXAMPLES["tidy_desk"][0] == ("妈妈要来", "我妈妈一会儿要来")
assert not [e for e in EMOJI if e in "".join(lab for _k, lab in RT.TASK_SCENES)]
root.destroy()
print("[OK] test_v4_8_3_gui: P1-P5")
