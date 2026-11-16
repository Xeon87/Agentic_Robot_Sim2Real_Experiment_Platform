#!/usr/bin/env python3
"""NERO V4.8.7 offline unit tests:
V1 cuRobo is the default IK planner in Agentic实机任务 (RealEggSkills & CupSkills) when available:
   - _fast_ik_chain and plan_egg_place use cuRobo first and report "ik_solver": "curobo".
   - CupSkills.ik_chain uses cuRobo first for key poses when available.
V2 Automatic fallback to 45-seed FastIK ("fast_ik_45seeds"):
   - When cuRobo is offline, disabled (NERO_NO_CUROBO=1), or yielded for SmolVLA VRAM, RealEggSkills and CupSkills
     automatically fall back to the 45-seed FastIK solver and report "ik_solver": "fast_ik_45seeds".
V3 Real-robot bring-up checks and starts cuRobo on demand:
   - NERO_GPU_Planner_Server.py is NOT in NERO_V313_Real_Robot.STALE_SCRIPTS (bring-up does not kill cuRobo).
   - RealPipeline.step_env and NERO_V312_Console.run_all_then_gripper invoke ensure_curobo_server().
V4 SmolVLA & training VRAM release:
   - stop_curobo_server() terminates a running NERO_GPU_Planner_Server.py process and marks vram_yielded=True.
   - Loading SmolVLA, running SmolVLA closed-loop, starting lerobot-train, and starting YOLOE fine-tuning all invoke
     stop_curobo_server() before allocating heavy GPU memory.
   - Returning to Agentic实机任务 re-enables cuRobo via ensure_curobo_server().
V5 Version & frozen-file integrity (NERO_V308_Camera_Perception.py, NERO_V325_Camera.py, NERO_GPU_Planner_Server.py,
   NERO_GPU_Planner_Client.py).
"""
from __future__ import annotations

import hashlib
import math
import os
import subprocess
import sys
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TESTS = os.path.dirname(os.path.realpath(__file__))
for p in (HERE, TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_v4_8_4 as T484                       # noqa: E402
import NERO_V312_Console as C                    # noqa: E402
import NERO_V313_Real_Robot as RR                # noqa: E402
import NERO_V362_Cup_Skills as CK                # noqa: E402
import NERO_V487_CuRobo_IK as C487               # noqa: E402


def _make_egg_skills():
    T484.use("agx_gripper")
    sk, io = T484.T13.user_scene()
    return sk, io


class TestV487(unittest.TestCase):
    def tearDown(self):
        C487.set_mock_solver(None)
        os.environ.pop("NERO_NO_CUROBO", None)
        T484.TC.find_handeye = T484._find

    def test_v1_curobo_default_ik_in_agentic_tasks(self):
        sk, _io = _make_egg_skills()
        fast = sk._fast
        curobo_calls = []

        def mock_curobo(q_start, pos, quat_xyzw, R_des):
            curobo_calls.append((list(pos), list(quat_xyzw)))
            a1 = math.atan2(float(pos[1]), float(pos[0]))
            a1 = a1 + math.pi if a1 < 0 else a1 - math.pi
            seeds = [list(q_start), [a1, 0.2, 0.0, 0.8, 0.0, 0.4, 0.0], [a1, 0.55, 0.0, 1.4, 0.0, 0.4, 0.9]] + sk._extra_ik_seeds(a1)
            q_sol, _pe, _re = fast.ik(seeds, pos, R_des, iters=45)
            return q_sol

        C487.set_mock_solver(mock_curobo)
        res = sk.plan_egg_place(source_egg="egg_1", target_slot="slot_1")
        self.assertTrue(res.get("ok"), res)
        pv = res.get("physical_verification") or {}
        self.assertEqual(pv.get("ik_solver"), "curobo")
        self.assertEqual(sk.plan.get("ik_solver"), "curobo")
        self.assertGreater(len(curobo_calls), 0)

        # Also verify CupSkills.ik_chain uses cuRobo for non-'~' poses when available
        curobo_calls.clear()
        q0 = [0.0, 0.35, 0.0, 1.15, 0.0, 0.25, 0.0]
        T0 = fast.fk(q0)
        p_target = (T0[:3, 3] + np.array([0.01, -0.01, 0.02])).tolist()
        R_target = T0[:3, :3].tolist()
        r_cup = CK.ik_chain([("pre", p_target, R_target)], q_start=q0)
        self.assertTrue(r_cup.get("ok"), r_cup)
        self.assertGreater(len(curobo_calls), 0)

    def test_v2_fallback_to_45seed_fast_ik_when_curobo_unavailable(self):
        C487.set_mock_solver(None)
        os.environ["NERO_NO_CUROBO"] = "1"
        sk, _io = _make_egg_skills()
        res = sk.plan_egg_place(source_egg="egg_1", target_slot="slot_1")
        self.assertTrue(res.get("ok"), res)
        pv = res.get("physical_verification") or {}
        self.assertEqual(pv.get("ik_solver"), "fast_ik_45seeds")
        self.assertEqual(sk.plan.get("ik_solver"), "fast_ik_45seeds")

    def test_v3_real_robot_bringup_keeps_and_starts_curobo(self):
        # 1) NERO_GPU_Planner_Server.py is NOT in RR.STALE_SCRIPTS
        self.assertNotIn("NERO_GPU_Planner_Server.py", RR.STALE_SCRIPTS)
        procs = [
            (101, ["/venv/bin/python", "/home/u/NERO/NERO_GPU_Planner_Server.py", "--prefer", "curobo"]),
            (102, ["python3", "/home/u/NERO/NERO_V2_7_1_Unified_Backend.py"]),
        ]
        stale_pids = {p for p, _c in RR.find_stale(procs)}
        self.assertNotIn(101, stale_pids)
        self.assertIn(102, stale_pids)

        # 2) RealPipeline has ensure_curobo_server and stop_curobo_server; console run_all_then_gripper calls ensure_curobo_server
        self.assertTrue(hasattr(RR.RealPipeline, "ensure_curobo_server"))
        self.assertTrue(hasattr(RR.RealPipeline, "stop_curobo_server"))
        with open(os.path.join(HERE, "NERO_V312_Console.py"), encoding="utf-8") as f:
            con = f.read()
        self.assertIn("pipe.ensure_curobo_server()", con)

    def test_v4_smolvla_and_training_stop_curobo_to_free_vram(self):
        # Spawn a dummy process masquerading as NERO_GPU_Planner_Server.py (via argv)
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)", "NERO_GPU_Planner_Server.py"],
            start_new_session=True,
        )
        try:
            time.sleep(0.15)
            pids = C487.find_server_pids()
            self.assertIn(proc.pid, pids)

            logs = []
            r = C487.stop_curobo_server(log=logs.append, reason="加载 SmolVLA 模型")
            self.assertTrue(r.get("ok"), r)
            self.assertIn(proc.pid, r.get("stopped", []))
            proc.wait(timeout=3.0)
            self.assertIsNotNone(proc.poll())

            # While yielded for VRAM, curobo_status reports vram_yielded=True and solver='fast_ik_45seeds'
            st = C487.curobo_status(force_ping=False)
            self.assertFalse(st.get("ok"))
            self.assertTrue(st.get("vram_yielded"))
            self.assertEqual(st.get("solver"), "fast_ik_45seeds")
            self.assertTrue(any("已关闭 cuRobo 规划服务并释放显存" in m for m in logs))
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=2.0)

        # Verify hooks in NERO_V4_Console_Tab.py, NERO_V4_SmolVLA_Policy.py, and NERO_V381_Yolo_Train.py
        with open(os.path.join(HERE, "NERO_V4_Console_Tab.py"), encoding="utf-8") as f:
            vla_tab = f.read()
        self.assertIn('self._release_curobo_vram("加载 SmolVLA 模型")', vla_tab)
        self.assertIn('self._release_curobo_vram("一键准备加载 SmolVLA 模型")', vla_tab)
        self.assertIn('self._release_curobo_vram("启动 LeRobot SmolVLA 微调训练")', vla_tab)
        self.assertIn('self._release_curobo_vram("执行 SmolVLA 闭环任务")', vla_tab)

        with open(os.path.join(HERE, "NERO_V4_SmolVLA_Policy.py"), encoding="utf-8") as f:
            vla_pol = f.read()
        self.assertIn('C487.stop_curobo_server(log=self.log, reason="加载 SmolVLA 模型")', vla_pol)
        self.assertIn('C487.stop_curobo_server(log=self.log, reason="执行 SmolVLA 闭环推理")', vla_pol)

        with open(os.path.join(HERE, "NERO_V381_Yolo_Train.py"), encoding="utf-8") as f:
            tr381 = f.read()
        self.assertIn('C487.stop_curobo_server(reason="启动 YOLOE 精调训练")', tr381)

    def test_v5_versions_and_frozen_hashes(self):
        with open(os.path.join(HERE, "VERSION"), encoding="utf-8") as f:
            v = f.read().strip()
        self.assertIn(v, ("V4.8.7", "V4.8.8", "V4.8.9", "V4.8.9", "V4.8.10", "V4.8.11", "V4.9", "V4.9.1")); self.assertEqual((v, C.VERSION, RR.VERSION), (v, v, v))
        self.assertEqual(C.MARKER_V487, "V4.8.7 CUROBO REAL IK + SMOLVLA VRAM RELEASE")
        self.assertEqual(C487.MARKER, "V4.8.7 CUROBO REAL IK + SMOLVLA VRAM RELEASE")

        with open(os.path.join(HERE, "NERO_V308_Camera_Perception.py"), "rb") as f:
            h308 = hashlib.sha256(f.read()).hexdigest()
        with open(os.path.join(HERE, "NERO_V325_Camera.py"), "rb") as f:
            h325 = hashlib.sha256(f.read()).hexdigest()
        with open(os.path.join(HERE, "NERO_GPU_Planner_Server.py"), "rb") as f:
            h_srv = hashlib.sha256(f.read()).hexdigest()
        with open(os.path.join(HERE, "NERO_GPU_Planner_Client.py"), "rb") as f:
            h_cli = hashlib.sha256(f.read()).hexdigest()
        self.assertTrue(h308.startswith("020cc6680f181872"), h308)
        self.assertTrue(h325.startswith("82f2416d"), h325)
        self.assertEqual(h_srv, "2728c2f191c8cd24c88276d8b39efffdfdb395e7fcdee5359a5727861e23a4bc")
        self.assertEqual(h_cli, "3956f7a596251149dfb0abc2b89f27334d8f9279a8fcf9e325eb5877d7605438")


if __name__ == "__main__":
    unittest.main(verbosity=1)
