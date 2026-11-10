#!/usr/bin/env python3
"""NERO V4.7 offline tests (the GUI parts are in tests/test_v4_7_gui.py).

W1 NpzVideoReader: frames read one by one from the compressed npz equal np.load(...) exactly, in any order (forward,
   backward, jumps); the whole array is never allocated; header / meta / fps.
W2 list_datasets: dataset folders under the root (newest first); episode_rows: newest first, duration / size / file,
   a missing file is flagged.
W3 draw_dets: boxes + labels drawn, scaled to the view; the input frame is not modified.
W4 versions / markers.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import tracemalloc
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
TMP = tempfile.mkdtemp(prefix="nero_v47_")
os.environ["NERO_CALIB_DIR"] = os.path.join(TMP, "calib")
os.environ["NERO_DRIVE_ROOT"] = os.path.join(TMP, "drive")
os.makedirs(os.environ["NERO_DRIVE_ROOT"], exist_ok=True)

import NERO_V4_Dataset_Recorder as DR          # noqa: E402
import NERO_V47_VLA_Tools as TL                # noqa: E402


def make_episode(ds, n=40, fps=20.0):
    frames = []
    rng = np.random.default_rng(n)
    for k in range(n):
        st = np.zeros(8, np.float32)
        st[0] = k * 0.01
        frames.append({"timestamp_ns": int(k * 1e9 / fps), "sync_skew_ms": 1.0, "state": st, "action": st.copy(),
                       "camera1": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
                       "camera2": np.full((480, 640, 3), k % 250, np.uint8)})
    return ds.save_episode(frames, task_instruction=f"demo {n}")


class TestV47(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ds = DR.NEROLeRobotDataset(dataset_name="v47_a", root_dir=os.path.join(TMP, "ds"))
        cls.m0 = make_episode(cls.ds, 40)
        cls.m1 = make_episode(cls.ds, 12)

    def test_w1_reader(self):
        p = self.ds._episode_npz_path(self.m0["episode_index"])
        ref = np.load(p)
        main, wrist = ref["cam_main"], ref["cam_wrist"]
        rd = TL.NpzVideoReader(p, cache_frames=4)
        self.assertEqual(rd.n, 40)
        self.assertEqual(rd.info["cam_main"]["shape"], (40, 480, 640, 3))
        self.assertEqual(rd.meta["episode_index"], self.m0["episode_index"])
        self.assertEqual(rd.fps, 20.0)
        for i in (0, 1, 2, 39, 5, 20, 19, 0, 38):                      # forward, jump, back, back to 0
            np.testing.assert_array_equal(rd.frame("cam_main", i), main[i])
            np.testing.assert_array_equal(rd.frame("cam_wrist", i), wrist[i])
        self.assertIsNone(rd.frame("cam_main", 40))
        self.assertIsNone(rd.frame("nope", 0))
        rd.close()
        rd = TL.NpzVideoReader(p)                                      # no full-array allocation
        tracemalloc.start()
        for i in range(0, 40, 3):
            rd.frame("cam_main", i)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        rd.close()
        self.assertLess(peak, 0.6 * main.nbytes, peak)

    def test_w2_lists(self):
        DR.NEROLeRobotDataset(dataset_name="v47_b", root_dir=os.path.join(TMP, "ds"))
        os.makedirs(os.path.join(TMP, "ds", "not_a_dataset"), exist_ok=True)
        time.sleep(0.02)
        names = TL.list_datasets(os.path.join(TMP, "ds"))
        self.assertIn("v47_a", names)
        self.assertIn("v47_b", names)
        self.assertNotIn("not_a_dataset", names)
        rows = TL.episode_rows(self.ds)
        self.assertEqual([r["idx"] for r in rows][:2], [self.m1["episode_index"], self.m0["episode_index"]])
        r0 = [r for r in rows if r["idx"] == self.m0["episode_index"]][0]
        self.assertAlmostEqual(r0["dur_s"], 2.0)
        self.assertTrue(r0["exists"] and r0["size_mb"] > 1.0)
        ds2 = DR.NEROLeRobotDataset(dataset_name="v47_c", root_dir=os.path.join(TMP, "ds"))
        m = make_episode(ds2, 5)
        os.remove(ds2._episode_npz_path(m["episode_index"]))
        r = TL.episode_rows(ds2)[0]
        self.assertFalse(r["exists"])

    def test_w3_draw(self):
        img = np.zeros((480, 640, 3), np.uint8)
        out = TL.draw_dets(img, [{"box": [100, 100, 200, 220], "name": "egg", "conf": 0.9}], (0.5, 0.5))
        self.assertEqual(int(img.sum()), 0)
        self.assertGreater(int(out[50:111, 50:101].sum()), 0)            # the scaled box
        self.assertEqual(int(out[300:, 300:].sum()), 0)

    def test_w4_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        import NERO_V4_Console_Tab as VT
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual((C.VERSION, RR.VERSION), (v, v))
        self.assertIn(v, ("V4.7", "V4.7.1"))
        self.assertEqual(TL.MARKER, "V4.7 EPISODE BROWSER + PLAYER")
        self.assertEqual(VT.MARKER_V47, "V4.7 ONE-CLICK PREPARE + EPISODE BROWSER")
        self.assertEqual([k for k, _t in VT.PREP_STEPS],
                         ["drive", "dirs", "env", "model", "dataset", "nero", "cam", "wrist", "grip", "vla"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
