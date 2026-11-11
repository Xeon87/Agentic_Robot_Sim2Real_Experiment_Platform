#!/usr/bin/env python3
"""NERO V4.7.4 offline tests (the GUI parts are in tests/test_v4_7_4_gui.py).

N1 limits: live check (J7 82 deg yellow, 96.2 deg red, 9.2 deg over the commandable 87 deg); episode report with
   joint, peak, time span, frames.
N2 the user's Episode #1 (J7 96.2 deg): the replay is refused BEFORE anything moves (no move_and_settle call), the
   log names J7, 96.2 deg and the 90 deg limit; an episode inside the limits still replays.
N3 saving marks limit_ok / limit_report; the LeRobot export reader skips over-limit episodes (unless asked);
   episodes saved before V4.7.4 get their check once (only the joint array is read).
N4 YOLOE prompts from task texts (English / Chinese / plural / fallback).
N5 task edit: only the chosen episodes change, backup first, tasks.jsonl gets the new text; errors for empty text /
   unknown episodes; the copy inside the episode file is rewritten with every array bit-identical.
N6 dataset rename: name rules, existing folder refused, folder moved, stored file paths fixed.
N7 versions / markers.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
TMP = tempfile.mkdtemp(prefix="nero_v474_")
os.environ["NERO_CALIB_DIR"] = os.path.join(TMP, "calib")
os.environ["NERO_DRIVE_ROOT"] = os.path.join(TMP, "drive")
os.makedirs(os.environ["NERO_DRIVE_ROOT"], exist_ok=True)

import NERO_V4_Dataset_Recorder as DR          # noqa: E402
import NERO_V474_Episode_Tools as ET           # noqa: E402
import NERO_V44_SmolVLA_Worker as WK           # noqa: E402

TASK = "Put the red block into the left box, then put the blue block into the right box"
MODEL = {"joints": [f"joint{i}" for i in range(1, 8)], "limits": [list(x) for x in ET.LIMITS_RAD]}


def make(ds, j7_peak_deg=None, n=60, task=TASK):
    frames = []
    rng = np.random.default_rng(n)
    for k in range(n):
        st = np.zeros(8, np.float32)
        st[1], st[3] = math.radians(21.6), math.radians(61.2)
        j7 = 44.8 + (j7_peak_deg - 44.8) * math.sin(math.pi * k / (n - 1)) if j7_peak_deg else 44.8
        st[6] = math.radians(j7)
        st[7] = 1.0
        frames.append({"timestamp_ns": int(k * 5e7), "sync_skew_ms": 0.0, "state": st, "action": st.copy(),
                       "camera1": rng.integers(0, 255, (48, 64, 3), dtype=np.uint8),
                       "camera2": rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)})
    return ds.save_episode(frames, task_instruction=task)


class Pipe:
    teach_mode = False

    def __init__(self):
        self.moves = []

    def move_and_settle(self, model, q, speed, label=""):
        self.moves.append(label)
        return {"ok": True}

    def actual_arm_deg(self, model):
        return [0.0] * 7

    def move_joint_path(self, *a, **k):
        self.moves.append("path")
        return {"ok": True}

    def supports_joint_path(self):
        return True


class TestV474(unittest.TestCase):
    def ds(self, name):
        return DR.NEROLeRobotDataset(dataset_name=name, root_dir=os.path.join(TMP, "ds"), image_shape=(48, 64, 3))

    def test_n1_limits(self):
        q = [0.0] * 7
        q[6] = math.radians(82.0)
        h = ET.check_q(q)
        self.assertEqual([(x["joint"], x["level"]) for x in h], [(7, "warn")])
        self.assertIn("接近限位", ET.live_text(h))
        q[6] = math.radians(96.2)
        h = ET.check_q(q)
        self.assertEqual(h[0]["level"], "over")
        self.assertAlmostEqual(h[0]["over_deg"], 9.2, places=1)
        self.assertIn("J7 +96.2°", ET.live_text(h))
        q[6] = math.radians(45.0)
        self.assertEqual(ET.check_q(q), [])
        S = np.zeros((100, 8))
        S[30:41, 6] = math.radians(96.2)
        rep = ET.episode_limit_report(S, 20.0)
        v = rep["violations"][0]
        self.assertFalse(rep["ok"])
        self.assertEqual((v["joint"], v["peak_deg"], v["limit_deg"], v["frames"]), (7, 96.2, 87.0, 11))
        self.assertEqual((v["t_from_s"], v["t_to_s"]), (1.5, 2.0))
        self.assertTrue(ET.episode_limit_report(S, 20.0, margin=0.0)["violations"][0]["limit_deg"] == 90.0)

    def test_n2_replay_refused_before_moving(self):
        ds = self.ds("n2")
        bad = make(ds, j7_peak_deg=96.2)
        logs, pipe = [], Pipe()
        r = DR.replay_episode_real(ds, bad["episode_index"], pipe, MODEL, log=logs.append)
        self.assertFalse(r["ok"])
        self.assertEqual(pipe.moves, [], "nothing may move")
        msg = [s for s in logs if "[V4.7.4 REPLAY]" in s]
        self.assertTrue(msg and "J7 最大 +96.2°" in msg[0] and "限位 +90°" in msg[0], logs)
        good = make(ds, j7_peak_deg=80.0)
        logs2, pipe2 = [], Pipe()
        DR.replay_episode_real(ds, good["episode_index"], pipe2, MODEL, log=logs2.append)
        self.assertTrue(pipe2.moves and pipe2.moves[0].startswith("V4.6.9 replay"), logs2)

    def test_n3_save_export_backfill(self):
        ds = self.ds("n3")
        a = make(ds)
        b = make(ds, j7_peak_deg=96.2)
        self.assertTrue(a["limit_ok"])
        self.assertFalse(b["limit_ok"])
        self.assertEqual(b["limit_report"]["violations"][0]["joint"], 7)
        eps = WK.read_nero_episodes(ds.dataset_dir)
        self.assertEqual([e["episode_index"] for e in eps], [a["episode_index"]])
        eps = WK.read_nero_episodes(ds.dataset_dir, include_over_limit=True)
        self.assertEqual(len(eps), 2)
        rows = ds.list_episodes()                    # as if saved by V4.7.3: no limit fields
        for r in rows:
            r.pop("limit_ok"), r.pop("limit_report")
        ET._write_jsonl(ds._episodes_path(), rows)
        self.assertEqual(sorted(ET.backfill_limits(ds)), [a["episode_index"], b["episode_index"]])
        self.assertEqual([e["limit_ok"] for e in ds.list_episodes()], [True, False])
        self.assertEqual(ET.backfill_limits(ds), [])

    def test_n4_prompts(self):
        self.assertEqual(ET.task_prompts(TASK), ["red block", "box", "blue block"])
        self.assertEqual(ET.task_prompts("把红色方块放进左边的盒子，再把蓝色方块放进右边的盒子"),
                         ["red block", "box", "blue block"])
        self.assertEqual(ET.task_prompts("Put all eggs into the egg carton"), ["egg", "egg carton"])
        self.assertEqual(ET.task_prompts("Stack the red cup on the blue cup"), ["red cup", "blue cup"])
        self.assertEqual(ET.task_prompts("grasp", fallback=["cup"]), ["cup"])
        self.assertEqual(ET.parse_prompts(" cup，plate; box ,"), ["cup", "plate", "box"])

    def test_n5_edit_task(self):
        ds = self.ds("n5")
        m = [make(ds, n=20) for _ in range(3)]
        idx = [x["episode_index"] for x in m]
        self.assertFalse(ET.set_episode_task(ds, idx[:1], "  ")["ok"])
        self.assertFalse(ET.set_episode_task(ds, [99], "x")["ok"])
        r = ET.set_episode_task(ds, idx[1:], "Stack the cups")
        self.assertTrue(r["ok"])
        self.assertEqual([c[0] for c in r["changes"]], idx[1:])
        self.assertEqual([e["tasks"] for e in ds.list_episodes()], [[TASK], ["Stack the cups"], ["Stack the cups"]])
        self.assertIn("Stack the cups", [t["task"] for t in ds.list_tasks()])
        with open(os.path.join(r["backup"], "episodes.jsonl"), encoding="utf-8") as f:
            self.assertNotIn("Stack the cups", f.read())
        p = ds._episode_npz_path(idx[1])
        with np.load(p) as z:
            before = {k: z[k].copy() for k in z.files if k != "meta_json"}
        res = ET.update_npz_meta(p, {"tasks": ["Stack the cups"]})
        self.assertTrue(res["ok"])
        with np.load(p) as z:
            self.assertEqual(json.loads(str(z["meta_json"]))["tasks"], ["Stack the cups"])
            for k, v in before.items():
                np.testing.assert_array_equal(z[k], v)
        self.assertFalse(os.path.exists(p + ".v474tmp"))
        self.assertFalse(ET.update_npz_meta(p + ".missing", {})["ok"])

    def test_n6_rename(self):
        ds = self.ds("n6_old")
        make(ds, n=10)
        root = ds.root_dir
        self.assertIn("只能用", ET.check_dataset_name(root, "bad name"))
        self.assertIn("只能用", ET.check_dataset_name(root, "数据集"))
        os.makedirs(os.path.join(root, "n6_taken"), exist_ok=True)
        self.assertIn("同名", ET.check_dataset_name(root, "n6_taken"))
        r = ET.rename_dataset(root, "n6_old", "n6_new")
        self.assertTrue(r["ok"])
        self.assertFalse(os.path.exists(os.path.join(root, "n6_old")))
        ds2 = DR.NEROLeRobotDataset(dataset_name="n6_new", root_dir=root, image_shape=(48, 64, 3))
        e = ds2.list_episodes()[0]
        self.assertEqual(e["file"], ds2._episode_npz_path(e["episode_index"]))
        self.assertTrue(os.path.isfile(e["file"]))
        self.assertEqual(ds2.load_episode(e["episode_index"])["observation_state"].shape, (10, 8))

    def test_n7_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        import NERO_V4_Console_Tab as VT
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual(v, C.VERSION)
        self.assertEqual(v, RR.VERSION)
        self.assertIn(v, ("V4.7.4", "V4.8", "V4.8.1"))
        self.assertEqual(ET.MARKER, "V4.7.4 JOINT LIMITS + TASK PROMPTS + EDIT TASK + RENAME DATASET")
        for name in ("on_edit_task", "edit_task", "on_rename_dataset", "rename_dataset_to", "_check_live_limits",
                     "_ask_over_limit"):
            self.assertTrue(hasattr(VT.VLALeRobotTabUI, name), name)
        self.assertEqual([tuple(x) for x in ET.LIMITS_RAD],
                         [tuple(x) for x in __import__("NERO_V4_Control_Arbiter").JOINT_LIMITS_RAD])


if __name__ == "__main__":
    unittest.main(verbosity=1)
