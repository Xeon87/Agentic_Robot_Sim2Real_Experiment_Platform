#!/usr/bin/env python3
"""NERO V4.8.3 MuJoCo twin test of the tidy desk (needs the mujoco package; skips without it).

W1 「我妈妈一会儿要来」 + answers (饼干 留着, 纸团 垃圾桶): every planned item ends in its container, the mouse and the
   cookies stay, the report says the desk check passed; the answers are remembered.
W2 「等下要开会，帮我腾个地方」 (second run, same preferences file): no question left; the fruit
   bowl is full with the fruit (capacity 2), so they stay; the marker stays for the meeting.
W3 「孩子一会儿来玩」, item poses jittered +-3 mm / +-5 deg (seed 2): the toy block stays out, the marker goes into the pen box.
Each run takes minutes (the twin steps the gripper physics); NERO_V483_TWIN=quick runs W1 only.
"""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
try:
    import mujoco  # noqa: F401
except Exception:
    print("[SKIP] test_v4_8_3_twin: no mujoco")
    sys.exit(0)
os.environ["NERO_CALIB_DIR"] = tempfile.mkdtemp(prefix="nero_v483_twin_")
os.environ.pop("NERO_GRIPPER", None)
import NERO_V483_Tidy as TD          # noqa: E402

PREFS = os.path.join(os.environ["NERO_CALIB_DIR"], "tidy_prefs.json")
LOG = []


def run(text, answers=None, seed=None, remember_answers=False):
    plan, dec, rep, sk = TD.twin_tidy(text, answers=answers, log=LOG.append, seed=seed, prefs_file=PREFS,
                                      remember_answers=remember_answers)
    print(rep["text"], flush=True)
    by = {st["label"]: st for st in plan["steps"]}
    now = {it["label"]: it["in_bin"] for it in sk.scene()["items"]}
    return plan, dec, rep, by, now


class TestV483Twin(unittest.TestCase):
    def test_w1_guests(self):
        plan, dec, rep, by, now = run("我妈妈一会儿要来，收拾一下桌子", {"饼干": "keep", "纸团": "trash_bin"},
                                      remember_answers=True)
        self.assertEqual(now["banana"], "fruit_bowl")
        self.assertEqual(now["apple"], "fruit_bowl")
        self.assertEqual(now["crumpled paper"], "trash_bin")
        self.assertEqual(now["toy block"], "toy_box")
        self.assertEqual(now["marker pen"], "pen_box")
        self.assertIsNone(now["computer mouse"])
        self.assertIsNone(now["cookie pack"])
        self.assertTrue(rep["ok"], rep["text"])
        self.assertIn("桌面检查通过", rep["text"])
        self.assertEqual(TD.pref_for(TD.load_prefs(PREFS), "guests", "cookie pack")[0], "keep")

    @unittest.skipIf(os.environ.get("NERO_V483_TWIN") == "quick", "quick")
    def test_w2_work(self):
        plan, dec, rep, by, now = run("等下要开会，帮我腾个地方")
        self.assertFalse([st for st in plan["steps"] if st["action"] == "ask"])
        self.assertIsNone(now["cookie pack"])                                # stays (the bowl holds 2)
        self.assertEqual(now["banana"], "fruit_bowl")
        self.assertEqual(now["apple"], "fruit_bowl")
        self.assertIsNone(now["marker pen"])
        self.assertTrue(rep["ok"], rep["text"])

    @unittest.skipIf(os.environ.get("NERO_V483_TWIN") == "quick", "quick")
    def test_w3_kids_jitter(self):
        plan, dec, rep, by, now = run("孩子一会儿来玩", {"饼干": "keep"}, seed=2)
        self.assertIsNone(now["toy block"])
        self.assertEqual(now["marker pen"], "pen_box")
        self.assertEqual(now["crumpled paper"], "trash_bin")
        self.assertTrue(rep["ok"], rep["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
