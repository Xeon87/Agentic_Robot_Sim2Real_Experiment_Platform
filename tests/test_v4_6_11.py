#!/usr/bin/env python3
"""NERO V4.6.11 offline tests (Smol VLA teach recording; the GUI parts are in tests/test_v4_6_11_gui.py).

S1 the recorder keeps the images in a disk spool (full 640x480, lossless) instead of RAM; save_captured writes them
   to the episode npz from read-only memmaps; the npz images equal the recorded ones; the spool is deleted.
S2 discard_captured: nothing saved, the spool deleted, the log says so; next_episode_index unchanged.
S3 stop_capture is fast (no saving); the saved log line names the episode file.
S4 the observation bridge reads the SO-101 status at most 5 x per second; peek_observation reads no hardware.
S5 TeachBar.enter_now / exit_now: no dialog, done(result) callback, exit holds the pose (return_home=False).
S6 versions / markers.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import types
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
TMP = tempfile.mkdtemp(prefix="nero_v4611_")
os.environ["NERO_CALIB_DIR"] = os.path.join(TMP, "calib")
os.environ["NERO_DRIVE_ROOT"] = os.path.join(TMP, "drive")
os.makedirs(os.environ["NERO_DRIVE_ROOT"], exist_ok=True)

import NERO_V4_Dataset_Recorder as DR          # noqa: E402
import NERO_V4_Observation_Bridge as OB        # noqa: E402

DR.SPOOL_ROOT = os.path.join(TMP, "spool")


class Bridge:
    """deterministic observations: frame k has pixel value k % 250 in both cameras"""

    def __init__(self):
        self.k = 0

    def get_synchronized_observation(self):
        self.k += 1
        v = self.k % 250
        st = np.zeros(8, np.float32)
        st[0] = self.k * 0.001
        return {"timestamp_ns": time.monotonic_ns(), "sync_skew_ms": 1.0, "state": st,
                "camera1": np.full((480, 640, 3), v, np.uint8), "camera2": np.full((480, 640, 3), 255 - v, np.uint8)}

    def set_task_instruction(self, t):
        pass

    def update_gripper_state(self, **k):
        pass


def make(name):
    ds = DR.NEROLeRobotDataset(dataset_name=name, root_dir=os.path.join(TMP, "ds"))
    logs = []
    rec = DR.DemonstrationRecorder(Bridge(), ds, fps=50.0, log=logs.append)
    return rec, ds, logs


class TestV4611(unittest.TestCase):
    def test_s1_spool_save(self):
        rec, ds, logs = make("s1")
        rec.start_recording("demo")
        time.sleep(0.6)
        sp = rec._spool
        self.assertTrue(sp is not None and sp.dir.startswith(DR.SPOOL_ROOT) and os.path.isdir(sp.dir))
        with rec.lock:
            self.assertTrue(rec._frames and "camera1" not in rec._frames[0])          # no images in RAM
        t0 = time.perf_counter()
        cap = rec.stop_capture()
        self.assertLess(time.perf_counter() - t0, 1.0)
        self.assertFalse(rec.is_recording)
        n = len(cap["frames"])
        self.assertGreater(n, 5)
        r = rec.save_captured(cap)
        ep = ds.load_episode(r["episode"]["episode_index"])
        self.assertEqual(ep["cam_main"].shape, (n, 480, 640, 3))
        first = int(ep["cam_main"][0, 0, 0, 0])
        for i in (0, n // 2, n - 1):                                     # lossless, in order
            self.assertEqual(int(ep["cam_main"][i, 5, 7, 1]), (first + i) % 250)
            self.assertEqual(int(ep["cam_wrist"][i, 0, 0, 0]), 255 - (first + i) % 250)
        self.assertFalse(os.path.exists(sp.dir))
        self.assertEqual(r["episode"]["file"], ds._episode_npz_path(r["episode"]["episode_index"]))
        self.assertTrue(any("→ " + r["episode"]["file"] in s for s in logs), logs[-2:])
        self.assertEqual(ds.next_episode_index(), r["episode"]["episode_index"] + 1)

    def test_s2_discard(self):
        rec, ds, logs = make("s2")
        nxt = ds.next_episode_index()
        rec.start_recording("demo")
        time.sleep(0.3)
        d = rec._spool.dir
        r = rec.discard_captured(rec.stop_capture())
        self.assertTrue(r["discarded"])
        self.assertFalse(os.path.exists(d))
        self.assertEqual(ds.list_episodes(), [])
        self.assertEqual(ds.next_episode_index(), nxt)
        self.assertTrue(any("已放弃本次录制" in s and "没有保存" in s for s in logs))

    def test_s3_old_api_and_sim_steps(self):
        rec, ds, logs = make("s3")
        rec.use_spool = False                                             # in-RAM path still works
        rec.start_recording("demo")
        time.sleep(0.2)
        r = rec.stop_and_save()
        self.assertTrue(r["ok"] and r["episode"]["length"] > 2)
        rec.record_step_From_state_action([0] * 8, [0] * 8)               # simulation demo generation path
        rec.record_step_From_state_action([0] * 8, [0] * 8)
        self.assertTrue(rec.save_captured({"frames": list(rec._frames), "spool": None, "task": "t",
                                           "split": "train"})["ok"])

    def test_s4_bridge_reads(self):
        calls = []
        grip = types.SimpleNamespace(connected=True, last=None)

        def status():
            calls.append(time.monotonic())
            grip.last = {"gap_mm": 30.0}
            return grip.last
        grip.status = status
        b = OB.ObservationBridge(env_mode="mock", log=lambda s: None)
        b.bind_real_sources(gripper=grip)
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            b.get_synchronized_observation()
            time.sleep(0.01)
        self.assertLessEqual(len(calls), 6, len(calls))
        self.assertGreaterEqual(len(calls), 4)
        n = len(calls)
        for _ in range(20):
            o = b.peek_observation()
        self.assertEqual(len(calls), n)
        self.assertAlmostEqual(o["gripper_gap_mm"], 30.0)
        self.assertNotIn("camera1", o)

    def test_s5_teachbar_now(self):
        import NERO_V462_Teach_Gravity as TG
        seen, done = [], []
        pipe = types.SimpleNamespace(teach_mode=False, move_busy=lambda: False,
                                     set_teach_mode=lambda on, **k: seen.append((on, k.get("return_home", True)))
                                     or {"ok": True})
        bar = TG.TeachBar.__new__(TG.TeachBar)
        bar.pipe, bar.jpanel, bar.cam_tab, bar.worker = pipe, None, None, None
        bar.log = lambda s: None
        bar.busy_fn = lambda: ""
        bar.safe_tab_fn = lambda: None
        bar._ask = lambda *a: (_ for _ in ()).throw(AssertionError("no dialog expected"))
        self.assertTrue(bar.enter_now(done=done.append))
        bar.worker.join(3)
        self.assertTrue(bar.exit_now(done=done.append))
        bar.worker.join(3)
        self.assertEqual(seen, [(True, True), (False, False)])
        self.assertEqual(done, [{"ok": True}, {"ok": True}])
        bar.busy_fn = lambda: "任务正在执行"
        out = []
        self.assertFalse(bar.enter_now(done=out.append))
        self.assertFalse(out[0]["ok"])

    def test_s6_markers(self):
        import NERO_V312_Console as C
        import NERO_V313_Real_Robot as RR
        import NERO_V4_Console_Tab as VT
        with open(os.path.join(HERE, "VERSION")) as f:
            v = f.read().strip()
        self.assertEqual((C.VERSION, RR.VERSION), (v, v))
        self.assertIn(v, ("V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4", "V4.8"))
        self.assertEqual(DR.MARKER_V4611_REC, "V4.6.11 DISK SPOOL + BACKGROUND SAVE + DISCARD")
        self.assertEqual(VT.MARKER_V4611, "V4.6.11 AUTO TEACH RECORD + SPACE BINDTAG + RECENT EPISODES + LIGHT REFRESH")
        src = open(os.path.join(HERE, "NERO_V4_Console_Tab.py"), encoding="utf-8").read()
        self.assertIn('w.bindtags(("NeroVLASpace",) + tuple(tags))', src)
        self.assertIn("self._install_space_tag(parent)", src)


if __name__ == "__main__":
    unittest.main(verbosity=1)
