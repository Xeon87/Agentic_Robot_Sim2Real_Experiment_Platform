#!/usr/bin/env python3
"""NERO V4.4 offline tests — EMBODIED_AI drive storage for LeRobot + SmolVLA.

T1 layout on the drive (NERO_DRIVE_ROOT -> temp folder): sub-folders, marker, activate script, env vars, dataset /
   checkpoint defaults, local model resolution, newest lerobot-train checkpoint.
T2 drive missing: legacy fall-back paths, heavy jobs refused, model download refused (nothing written to the internal disk);
   a plain folder called EMBODIED_AI that is NOT a mount point is rejected.
T3 install script (bash -n, every cache / TMPDIR on the drive) and lerobot-train command.
T4 SmolVLA worker protocol end-to-end (fake worker, real subprocess, raw image bytes) -> backend "hf_smolvla_ext".
T5 detached BackgroundJob (log file on the drive, job file, read_new).
T6 NERO recorder dataset -> LeRobot export reader (no lerobot needed for the reader).
T7 migrating old internal-disk folders onto the drive.
No robot, no lerobot, no torch, no network needed.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import NERO_V44_Storage as ST  # noqa: E402
import NERO_V44_SmolVLA_Worker as WK  # noqa: E402


class _EnvPatch:
    def __init__(self, **kv):
        self.kv = kv
        self.old = {}

    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        return self

    def __exit__(self, *a):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestV44DriveStorage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nero_v44_")
        self.drive = os.path.join(self.tmp, "EMBODIED_AI")
        os.makedirs(self.drive)
        self.env = _EnvPatch(NERO_DRIVE_ROOT=self.drive, NERO_V4_DATASET_DIR=None, NERO_V4_CKPT_DIR=None,
                             NERO_V44_ALLOW_INTERNAL=None, NERO_LEROBOT_PYTHON=None)
        self.env.__enter__()

    def tearDown(self):
        self.env.__exit__()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _fake_weights(self, d):
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump({"type": "smolvla"}, f)
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(b"\0" * 16)

    def test_t1_layout_on_drive(self):
        L = ST.layout()
        self.assertTrue(L["on_drive"])
        root = os.path.join(self.drive, ST.LAYOUT_DIRNAME)
        self.assertEqual(L["root"], root)
        for k in ("venv", "hf_home", "smolvla_base_dir", "datasets_nero", "datasets_lerobot", "checkpoints",
                  "training", "pip_cache", "tmp", "logs"):
            self.assertTrue(L[k].startswith(root + os.sep), k)
        r = ST.ensure_layout(L)
        self.assertTrue(r["ok"], r)
        for d in ST.SUBDIRS:
            self.assertTrue(os.path.isdir(os.path.join(root, d)), d)
        self.assertTrue(os.path.isfile(L["marker"]))
        subprocess.run(["bash", "-n", L["activate_sh"]], check=True)
        env = ST.storage_env(L)
        for k in ("HF_HOME", "HF_HUB_CACHE", "HF_LEROBOT_HOME", "TORCH_HOME", "PIP_CACHE_DIR", "TMPDIR"):
            self.assertTrue(env[k].startswith(root), k)
        with _EnvPatch(PYTHONPATH="/opt/ros/humble/lib/python3.10/site-packages"):
            ce = ST.child_env(L)
        self.assertNotIn("PYTHONPATH", ce)
        self.assertTrue(ce["PATH"].startswith(os.path.join(L["venv"], "bin")))
        # defaults follow the drive
        self.assertEqual(ST.default_dataset_root(), L["datasets_nero"])
        self.assertEqual(ST.default_checkpoint_root(), L["checkpoints"])
        import NERO_V4_Dataset_Recorder as DRC
        ds = DRC.NEROLeRobotDataset(dataset_name="v44_ds")
        self.assertTrue(ds.dataset_dir.startswith(L["datasets_nero"]))
        # model resolution: base id -> local copy once downloaded
        self.assertEqual(ST.resolve_model_target("lerobot/smolvla_base", L) if not ST.has_weights(ST.LEGACY_MODEL_DIR)
                         else "lerobot/smolvla_base", "lerobot/smolvla_base")
        self._fake_weights(L["smolvla_base_dir"])
        self.assertEqual(ST.resolve_model_target("lerobot/smolvla_base", L), L["smolvla_base_dir"])
        # newest lerobot-train checkpoint
        a = os.path.join(L["training"], "run_a", "checkpoints", "002000", "pretrained_model")
        b = os.path.join(L["training"], "run_b", "checkpoints", "last", "pretrained_model")
        self._fake_weights(a)
        time.sleep(0.02)
        self._fake_weights(b)
        os.utime(a, (time.time() - 100, time.time() - 100))
        self.assertEqual(ST.latest_training_checkpoint(L), b)
        fs = ST.fs_checks(L["root"])
        self.assertTrue(fs["writable"])
        st = ST.status(probe_venv=True)
        self.assertFalse(st["venv"]["exists"])
        txt = ST.format_status(st)
        self.assertIn("EMBODIED_AI", txt)
        self.assertIn("未安装", txt)
        # apply_to_process_env sets HF_HOME, but does not override a user's own export
        with _EnvPatch(HF_HOME="/user/own/hf", _NERO_V44_SET_HF_HOME=None, NERO_V4_DATASET_DIR=None,
                       _NERO_V44_SET_NERO_V4_DATASET_DIR=None):
            ST.apply_to_process_env(L)
            self.assertEqual(os.environ["HF_HOME"], "/user/own/hf")
            self.assertEqual(os.environ["NERO_V4_DATASET_DIR"], L["datasets_nero"])
            self.assertEqual(ST.default_dataset_root(), L["datasets_nero"])

    def test_t2_drive_missing(self):
        with _EnvPatch(NERO_DRIVE_ROOT=os.path.join(self.tmp, "not_plugged")):
            L = ST.layout()
            self.assertFalse(L["on_drive"])
            self.assertEqual(L["datasets_nero"], ST.LEGACY_DATASET_ROOT)
            self.assertEqual(L["checkpoints"], ST.LEGACY_CKPT_ROOT)
            r = ST.ensure_layout(L)
            self.assertFalse(r["ok"])
            import NERO_V4_SmolVLA_Policy as VLA
            pol = VLA.SmolVLAPolicy(log=lambda _s: None)
            d = pol.download_model_weights(repo_id="lerobot/smolvla_base")
            self.assertFalse(d["ok"])
            self.assertIn("未挂载", d["error"])
            # no drive venv -> load() falls back to the built-in engine instead of failing
            res = pol.load("lerobot/smolvla_base")
            self.assertTrue(res["ok"])
            self.assertIn(pol.backend_type, ("nero_vla_pretrained", "hf_smolvla"))
        # stale folder named EMBODIED_AI that is not a mount point must NOT be used
        stale = os.path.join(self.tmp, "media", "EMBODIED_AI")
        os.makedirs(stale)
        orig = ST.read_mounts
        try:
            ST.read_mounts = lambda path="/proc/self/mounts": [
                {"device": "/dev/sdz1", "mountpoint": stale, "fstype": "ext4", "options": "rw"}]
            with _EnvPatch(NERO_DRIVE_ROOT=None):
                drv = ST.find_drive()
            self.assertFalse(drv["found"])
        finally:
            ST.read_mounts = orig

    def test_t3_install_script_and_train_command(self):
        L = ST.layout()
        ST.ensure_layout(L)
        sc = ST.install_script(L, base_py="/usr/bin/python3", pip_spec="lerobot[smolvla]", download_model=True,
                               worker_path=os.path.join(HERE, "NERO_V44_SmolVLA_Worker.py"))
        p = os.path.join(self.tmp, "inst.sh")
        with open(p, "w") as f:
            f.write(sc)
        subprocess.run(["bash", "-n", p], check=True)
        for k in ("PIP_CACHE_DIR", "TMPDIR", "HF_HOME"):
            self.assertIn(f"export {k}={L['root']}", sc.replace("'", ""))
        self.assertIn(L["venv"], sc)
        self.assertIn("unset PYTHONPATH", sc)
        self.assertIn("download --repo lerobot/smolvla_base", sc)
        if not ST.fs_checks(L["root"])["symlinks"]:
            self.assertIn("--copies", sc)
        else:  # default: private Python 3.12 from uv on the drive (LeRobot >= 0.5 needs 3.12), system Python fallback
            self.assertIn("/bin/uv\" python install 3.12", sc)
            self.assertIn("make_venv_system", sc)
            self.assertIn(f"export UV_PYTHON_INSTALL_DIR={L['uv_python_dir']}", sc.replace("'", ""))
        sc2 = ST.install_script(L, py_mode="system", download_model=False)
        with open(p, "w") as f:
            f.write(sc2)
        subprocess.run(["bash", "-n", p], check=True)
        self.assertIn("PY_MODE=system", sc2)
        cmd = ST.train_command(L, "nero/demo", os.path.join(L["datasets_lerobot"], "demo"), "run1", steps=100, batch_size=4)
        j = " ".join(cmd)
        self.assertIn("--dataset.root=" + os.path.join(L["datasets_lerobot"], "demo"), j)
        self.assertIn("--output_dir=" + os.path.join(L["training"], "run1"), j)
        self.assertIn("--policy.push_to_hub=false", j)
        self.assertIn("--steps=100", j)
        self.assertIn("lerobot.scripts.train", j)  # no lerobot-train console script in the (missing) venv
        self.assertNotIn("--wandb", j)  # LeRobot 0.7 rejects --wandb.* (real-install finding)
        # smolvla_base expects camera1/2/3 -> automatic --rename_map for the NERO camera keys (real-install finding)
        base = L["smolvla_base_dir"]
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, "config.json"), "w") as f:
            json.dump({"input_features": {"observation.state": {"type": "STATE", "shape": [6]},
                                          **{f"observation.images.camera{i}": {"type": "VISUAL", "shape": [3, 256, 256]} for i in (1, 2, 3)}}}, f)
        with open(os.path.join(base, "model.safetensors"), "wb") as f:
            f.write(b"0")
        self.assertEqual(ST.camera_rename_map(base), {"observation.images.cam_main": "observation.images.camera1",
                                                      "observation.images.cam_wrist": "observation.images.camera2"})
        j2 = " ".join(ST.train_command(L, "nero/demo", "/x", "run2"))
        self.assertIn("--policy.path=" + base, j2)
        self.assertIn('--rename_map={"observation.images.cam_main": "observation.images.camera1"', j2)
        self.assertNotIn("--rename_map", " ".join(ST.train_command(L, "nero/demo", "/x", "run3", extra_args="--rename_map={}")).replace("--rename_map={}", ""))
        # LeRobot >= 0.5 refuses to import while the deprecated LEROBOT_HOME is set (real-install finding)
        with _EnvPatch(LEROBOT_HOME="/old"):
            self.assertNotIn("LEROBOT_HOME", ST.child_env(L))
        self.assertIn("dataset", ST.DEFAULT_PIP_SPEC)
        self.assertIn("training", ST.DEFAULT_PIP_SPEC)

    def test_t4_worker_protocol_fake(self):
        import NERO_V4_Control_Arbiter as ARB
        import NERO_V4_LeRobot_Adapter as LRA
        import NERO_V4_Observation_Bridge as OBR
        import NERO_V4_SmolVLA_Policy as VLA
        import NERO_V4_VLA_Action_Executor as VEX
        ST.ensure_layout(ST.layout())
        logs = []
        pol = VLA.SmolVLAPolicy(chunk_size=12, execute_horizon=6, log=logs.append)
        r = pol.load_external("lerobot/smolvla_base", python_exe=sys.executable, fake=True, timeout_s=60)
        self.assertTrue(r["ok"], logs)
        self.assertEqual(pol.backend_type, "hf_smolvla_ext")
        self.assertEqual(r["config"]["action_dim"], 6)
        arb = ARB.ControlArbiter(log=lambda _s: None)
        bridge = OBR.ObservationBridge(env_mode="mock", log=lambda _s: None)
        bridge.start_mock_sources()
        ex = VEX.VLAActionExecutor(arb, bridge, log=lambda _s: None)
        ad = LRA.NEROLeRobotAdapter(obs_bridge=bridge, executor=ex, arbiter=arb, log=lambda _s: None)
        ad.connect()
        obs = ad.get_observation()
        obs = dict(obs)
        img = np.full_like(np.asarray(obs["observation.images.cam_main"], dtype=np.uint8), 128)
        obs["observation.images.cam_main"] = img
        pred = pol.predict_action_chunk(obs, task_instruction="stack the cups")
        self.assertEqual(pred["backend"], "hf_smolvla_ext")
        self.assertEqual(pred["action_chunk"].shape, (12, 8))
        # 6-D SO-100 style output -> gripper channel carries the image mean computed in the worker (bytes arrived)
        self.assertAlmostEqual(float(pred["action_chunk"][0, 7]), 128 / 255.0, places=3)
        self.assertTrue(pol._worker.alive())
        proc = pol._worker.proc
        pol.close()
        self.assertIsNotNone(proc.poll())
        self.assertEqual(pol.backend_type, "unloaded")
        with self.assertRaises(RuntimeError):
            pol.is_loaded = True
            pol.backend_type = "hf_smolvla_ext"
            pol.predict_action_chunk(obs, task_instruction="x")
        bridge.close()

    def test_t5_background_job(self):
        L = ST.layout()
        ST.ensure_layout(L)
        job = ST.BackgroundJob(L, log=lambda _s: None)
        r = job.start("unit", ["sh", "-c", "echo step1; printf 'a\\rb\\n'; echo NERO_JOB_DONE ok"])
        self.assertTrue(r["ok"])
        self.assertTrue(r["log"].startswith(L["logs"]))
        for _ in range(100):
            if not job.running():
                break
            time.sleep(0.05)
        self.assertEqual(job.returncode(), 0)
        txt = job.read_new()
        self.assertIn("step1", txt)
        self.assertIn("NERO_JOB_DONE ok", txt)
        self.assertIn("\nb", txt)
        with open(L["job_file"]) as f:
            self.assertEqual(json.load(f)["kind"], "unit")
        job2 = ST.BackgroundJob(L)
        self.assertFalse(job2.adopt_previous())  # finished -> nothing to adopt
        self.assertFalse(job.stop())

    def test_t6_export_reader(self):
        import NERO_V4_Control_Arbiter as ARB
        import NERO_V4_Dataset_Recorder as DRC
        import NERO_V4_Observation_Bridge as OBR
        import NERO_V4_VLA_Action_Executor as VEX
        ST.ensure_layout(ST.layout())
        arb = ARB.ControlArbiter(log=lambda _s: None)
        bridge = OBR.ObservationBridge(env_mode="mock", log=lambda _s: None)
        bridge.start_mock_sources()
        ex = VEX.VLAActionExecutor(arb, bridge, log=lambda _s: None)
        ds = DRC.NEROLeRobotDataset(dataset_name="exp_ds", fps=20.0)
        self.assertTrue(ds.dataset_dir.startswith(ST.layout()["datasets_nero"]))
        rec = DRC.DemonstrationRecorder(bridge, ds, executor=ex, fps=20.0, log=lambda _s: None)
        for split in ("train", "test"):
            rec._frames.clear()
            rec._current_task = "Put the cup on the cup"
            rec._current_split = split
            for i in range(10):
                s8 = [0.01 * i] * 7 + [1.0]
                rec.record_step_From_state_action(s8, s8, timestamp_ns=10 ** 12 + i * 50_000_000, sync_skew_ms=3.0)
            self.assertTrue(rec.stop_and_save(episode_success=True, lead_action_by_one_step=False)["ok"])
        eps = WK.read_nero_episodes(ds.dataset_dir, splits=("train",))
        self.assertEqual(len(eps), 1)
        frames = list(WK.iter_episode_frames(eps[0]))
        self.assertEqual(len(frames), 10)
        fr, task = frames[3]
        self.assertEqual(task, "Put the cup on the cup")
        self.assertEqual(fr["observation.state"].shape, (8,))
        self.assertEqual(fr["observation.images.cam_main"].dtype, np.uint8)
        self.assertEqual(fr["observation.images.cam_main"].ndim, 3)
        feats = WK.lerobot_features(fr["observation.images.cam_main"].shape, use_videos=True)
        self.assertEqual(feats["observation.images.cam_wrist"]["dtype"], "video")
        self.assertEqual(feats["action"]["shape"], (8,))
        bridge.close()

    def test_t7_migrate_legacy(self):
        L = ST.layout()
        old = {k: getattr(ST, k) for k in ("LEGACY_DATASET_ROOT", "LEGACY_CKPT_ROOT", "LEGACY_MODEL_DIR", "_ORIG_HF_HOME")}
        home = os.path.join(self.tmp, "home")
        try:
            ST.LEGACY_DATASET_ROOT = os.path.join(home, "ds")
            ST.LEGACY_CKPT_ROOT = os.path.join(home, "ck")
            ST.LEGACY_MODEL_DIR = os.path.join(home, "smolvla_base")
            ST._ORIG_HF_HOME = os.path.join(home, "hf")
            os.makedirs(os.path.join(ST.LEGACY_DATASET_ROOT, "demo", "meta"))
            with open(os.path.join(ST.LEGACY_DATASET_ROOT, "demo", "meta", "info.json"), "w") as f:
                f.write("{}")
            os.makedirs(os.path.join(ST.LEGACY_CKPT_ROOT, "c1"))
            with open(os.path.join(ST.LEGACY_CKPT_ROOT, "c1", "checkpoint.npz"), "wb") as f:
                f.write(b"x" * 10)
            self._fake_weights(ST.LEGACY_MODEL_DIR)
            os.makedirs(os.path.join(home, "hf", "hub", "models--lerobot--smolvla_base", "blobs"))
            with _EnvPatch(HOME=home):
                items = ST.legacy_locations()
                self.assertEqual({i["key"] for i in items if i["movable"]}, {"datasets", "checkpoints", "smolvla_base", "hf_cache"})
                r = ST.migrate_legacy(L, delete_source=False, log=lambda _s: None)
            self.assertTrue(r["ok"], r)
            self.assertTrue(os.path.isfile(os.path.join(L["datasets_nero"], "demo", "meta", "info.json")))
            self.assertTrue(os.path.isfile(os.path.join(L["checkpoints"], "c1", "checkpoint.npz")))
            self.assertTrue(ST.has_weights(L["smolvla_base_dir"]))
            self.assertTrue(os.path.isdir(os.path.join(L["hf_hub_cache"], "models--lerobot--smolvla_base")))
            self.assertTrue(os.path.isdir(ST.LEGACY_CKPT_ROOT))  # copy only
            n = ST.delete_sources(r["moved"], log=lambda _s: None)
            self.assertEqual(n, len(r["moved"]))
            self.assertFalse(os.path.isdir(os.path.join(ST.LEGACY_CKPT_ROOT, "c1")))
        finally:
            for k, v in old.items():
                setattr(ST, k, v)

    def test_t8_markers(self):
        with open(os.path.join(HERE, "VERSION")) as f:
            self.assertTrue(f.read().strip().startswith("V4."))
        self.assertEqual(ST.MARKER, "V4.4 EMBODIED_AI DRIVE STORAGE")
        self.assertEqual(WK.MARKER, "V4.4 SMOLVLA DRIVE WORKER")
        with open(os.path.join(HERE, "NERO_V4_Console_Tab.py"), encoding="utf-8") as f:
            tab = f.read()
        for m in ("V4.4 EMBODIED_AI DRIVE PANEL", "def on_install_env", "def on_train_lerobot", "def on_export_lerobot",
                  "def on_eject_drive", "def on_migrate_legacy", "self.vla_policy.close()"):
            self.assertIn(m, tab)
        with open(os.path.join(HERE, "start_nero.sh"), encoding="utf-8") as f:
            self.assertIn("V4.4 EMBODIED_AI DRIVE", f.read())


if __name__ == "__main__":
    unittest.main(verbosity=1)
