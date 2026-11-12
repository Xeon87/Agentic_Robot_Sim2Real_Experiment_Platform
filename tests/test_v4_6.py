#!/usr/bin/env python3
"""NERO V4.6 offline tests.

SmolVLA (VLA / LeRobot tab, items 1-5):
  V1 model profile + REAL-robot gate: smolvla_base (6-D, SO-100) and the untrained built-in stand-in are refused on the
     real backend before any motion; a NERO-trained (8-D) model is allowed; mock backend runs with a warning.
  V2 every closed loop logs its start, every chunk and its end - also failures (e.g. control not allowed).
  V3 no success check -> "unverified" (not success, no retries); a fence violation is reported with its reason.
  V4 GUI: status lamps, one-run-at-a-time guard, start / end log of every ③ run (markers; the GUI itself is
     exercised by the xvfb smoke test).
  V5 drive label is case-insensitive ("Embodied_AI"), environment check waits up to 180 s.
Egg into carton:
  E1 camera agreement at the egg: <= 6 mm shift; 6-25 mm look again -> wrist position when both reads agree
     (+ the same correction for a slot the wrist did not measure); disagreeing reads or > 25 mm -> stop, no grasp.
  E2 off-centre grasp (tip gap < egg width - 8 mm, the V4.5 run had 24 mm) -> release, look again, regrasp once.
  E3 strict arrival before the egg is released (2 mm / 1 deg / 0.5 deg per joint, one correction) - an arm that stays
     short never releases the egg.
  E4 pre-pick wrist look at the slot with an empty gripper (two agreeing reads) + the cavity detector on a
     synthetic wrist image; release check (egg tip from measured joints).
  E5 log rotation (one file per run, newest 10 kept), Camera-tab calibration photos only after real arrival.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import threading
import unittest

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
TESTS = os.path.dirname(os.path.realpath(__file__))
if TESTS not in sys.path:
    sys.path.insert(0, TESTS)

import NERO_V365_Real_Cup_Task as RT  # noqa: E402
import NERO_V386_Real_Wrist_Camera as RWC  # noqa: E402
import NERO_V44_Storage as ST  # noqa: E402
import NERO_V4_Agent_Orchestrator as AOR  # noqa: E402
import NERO_V4_Control_Arbiter as ARB  # noqa: E402
import NERO_V4_LeRobot_Adapter as LRA  # noqa: E402
import NERO_V4_Observation_Bridge as OBR  # noqa: E402
import NERO_V4_Policy_Router as PRT  # noqa: E402
import NERO_V4_SmolVLA_Policy as VLA  # noqa: E402
import NERO_V4_Task_Evaluator as TEV  # noqa: E402
import NERO_V4_VLA_Action_Executor as VEX  # noqa: E402
import test_v4_5 as T45  # noqa: E402  (egg twin harness)

FRAME = {"rgb": np.zeros((8, 8, 3), np.uint8), "K": {"fx": 1}, "calib_eih": {"T_flange_cam": np.eye(4).tolist()}}


class FakePipe:
    """Stands in for RealPipeline: records every joint path the VLA executor would send to the real arm."""

    def __init__(self):
        self.paths = []

    def move_joint_path(self, model, path, times, label="", **_):
        self.paths.append(path)
        return {"ok": True}

    def move_busy(self):
        return False

    def cancel_move(self):
        pass


def vla_rig(logs):
    arb = ARB.ControlArbiter(log=lambda _s: None)
    br = OBR.ObservationBridge(env_mode="mock", log=lambda _s: None)
    br.start_mock_sources()
    ex = VEX.VLAActionExecutor(arb, br, log=lambda _s: None)
    ad = LRA.NEROLeRobotAdapter(obs_bridge=br, executor=ex, arbiter=arb, log=lambda _s: None)
    ad.connect()
    return arb, br, ex, ad


class TestV46VLA(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nero_v46_")
        os.environ["NERO_DRIVE_ROOT"] = self.tmp
        ST.ensure_layout(ST.layout())

    def tearDown(self):
        os.environ.pop("NERO_DRIVE_ROOT", None)
        os.environ.pop("NERO_V44_FAKE_DIMS", None)

    def _real(self, ex):
        pipe = FakePipe()
        ex.bind_real_robot(pipe, lambda: {"joints": ["j"] * 7})   # no jpanel / cam -> observations stay mock
        return pipe

    def test_v1_real_gate(self):
        logs = []
        arb, br, ex, ad = vla_rig(logs)
        arb.request_mode("smolvla", owner_tag="t")
        arb.set_allow_control(True)
        # (a) official smolvla_base (6-D SO-100) through the drive worker (fake, real subprocess)
        pol = VLA.SmolVLAPolicy(chunk_size=8, execute_horizon=4, log=logs.append)
        self.assertTrue(pol.load_external("lerobot/smolvla_base", python_exe=sys.executable, fake=True)["ok"])
        prof = pol.model_profile()
        self.assertFalse(prof["nero_trained"])
        self.assertFalse(prof["real_allowed"])
        pipe = self._real(ex)
        r = pol.run_vla_closed_loop(ad, "pick up the egg", max_chunks=2)
        self.assertFalse(r["ok"])
        self.assertEqual(r["status"], "refused")
        self.assertEqual(r["reason"], "MODEL_NOT_TRAINED_FOR_NERO")
        self.assertEqual(pipe.paths, [])                       # nothing was sent to the arm
        self.assertTrue(any("[FAIL] [V4.6 VLA LOOP]" in s and "MODEL_NOT_TRAINED_FOR_NERO" in s for s in logs))
        # (b) the same model on the mock backend: runs, with a warning, result "executed_unverified"
        ex.bind_mock()
        logs.clear()
        r = pol.run_vla_closed_loop(ad, "pick up the egg", max_chunks=2)
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["status"], "executed_unverified")
        self.assertTrue(any("不是为 NERO 训练的" in s for s in logs))
        pol.close()
        # (c) a NERO-trained (8-D) checkpoint is allowed on the real backend and does reach the arm
        os.environ["NERO_V44_FAKE_DIMS"] = "8,8"
        pol2 = VLA.SmolVLAPolicy(chunk_size=8, execute_horizon=4, log=logs.append)
        self.assertTrue(pol2.load_external("/x/pretrained_model", python_exe=sys.executable, fake=True)["ok"])
        self.assertTrue(pol2.model_profile()["real_allowed"])
        pipe = self._real(ex)
        r = pol2.run_vla_closed_loop(ad, "pick up the egg", max_chunks=2)
        self.assertTrue(r["ok"], r)
        self.assertEqual(len(pipe.paths), 2)
        pol2.close()
        # (d) the untrained built-in stand-in (no SmolVLA available) is refused on the real backend too
        os.environ["NERO_V44_DISABLE_EXT"] = "1"
        try:
            pol3 = VLA.SmolVLAPolicy(log=logs.append)
            pol3.load("lerobot/smolvla_base")
        finally:
            os.environ.pop("NERO_V44_DISABLE_EXT", None)
        self.assertIn(pol3.backend_type, ("nero_vla_pretrained",))
        r = pol3.run_vla_closed_loop(ad, "x", max_chunks=1)
        self.assertEqual(r["reason"], "MODEL_NOT_TRAINED_FOR_NERO")
        br.close()

    def test_v2_failures_are_logged(self):
        logs = []
        arb, br, ex, ad = vla_rig(logs)
        pol = VLA.SmolVLAPolicy(chunk_size=8, execute_horizon=4, log=logs.append)
        pol.load()
        # control NOT allowed (the user's silent clicks): the loop must say so
        r = pol.run_vla_closed_loop(ad, "pick", max_chunks=2)
        self.assertFalse(r["ok"])
        starts = [s for s in logs if "[V4.6 VLA LOOP] 开始" in s]
        ends = [s for s in logs if "[FAIL] [V4.6 VLA LOOP] 失败" in s]
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(ends), 1)
        self.assertTrue(r["reason"])
        self.assertIn(r["reason"], ends[0])                   # the reason is in the log line
        # allowed: per-chunk lines + an end line
        arb.request_mode("smolvla", owner_tag="t")
        arb.set_allow_control(True)
        logs.clear()
        r = pol.run_vla_closed_loop(ad, "pick", max_chunks=3)
        self.assertTrue(r["ok"])
        self.assertEqual(len([s for s in logs if "[V4.6 VLA LOOP] 块" in s]), 3)
        self.assertTrue(any("执行完毕（未验证任务是否完成）" in s for s in logs))
        br.close()

    def test_v3_unverified_is_not_success(self):
        arb, br, ex, ad = vla_rig([])
        pol = VLA.SmolVLAPolicy(chunk_size=8, execute_horizon=4, log=lambda _s: None)
        pol.load()
        logs = []
        ev = TEV.TaskEvaluator(obs_bridge=br, arbiter=arb, log=logs.append)
        router = PRT.PolicyRouter(arbiter=arb, adapter=ad, vla_policy=pol, evaluator=ev, log=logs.append)
        orch = AOR.AgentOrchestrator(router=router, log=logs.append)
        arb.request_mode("smolvla", owner_tag="t")
        arb.set_allow_control(True)
        r = orch.run_task("stack cup_1 into cup_2", policy="smolvla", max_vla_chunks=2)
        self.assertFalse(r["ok"])
        self.assertEqual(r["status"], "unverified")
        self.assertEqual(r["retry_count"], 0)
        self.assertTrue(any("无法确认是否完成" in s for s in logs))
        self.assertFalse(any("顶层任务全部完成" in s for s in logs))
        br.close()

    def test_v4_gui_markers(self):
        tab = open(os.path.join(HERE, "NERO_V4_Console_Tab.py"), encoding="utf-8").read()
        for m in ("class StatusLamp", "self.lamp_drive", "self.lamp_env", "self.lamp_model", "self.lamp_job",
                  "self.lamp_1", "self.lamp_2", "self.lamp_3", "self.lamp_4", "def _start_run", "还在执行，这次点击",
                  "③ 开始「", "③「{kind}」结束", "验证中…", "检测中…", "① 开始加载 SmolVLA", "def _refresh_lamps", "后端: "):
            self.assertIn(m, tab, m)
        for fn in ("on_run_vla_loop", "on_run_agent_task", "on_sim2real_ladder", "on_run_benchmark"):
            body = tab[tab.index(f"def {fn}"):]
            body = body[:body.index("\n    def ", 10)]
            self.assertIn("self._start_run(", body, fn)

    def test_v5_drive_label_and_timeout(self):
        d = tempfile.mkdtemp()
        dev = os.path.join(d, "sda1")
        open(dev, "w").close()
        os.symlink(dev, os.path.join(d, "Embodied_AI"))
        self.assertEqual(ST.label_device("EMBODIED_AI", by_label_dir=d), os.path.realpath(dev))
        self.assertIsNone(ST.label_device("OTHER", by_label_dir=d))
        self.assertGreaterEqual(ST.VENV_PROBE_TIMEOUT_S, 180.0)


class SeqGrip(T45.Grip):
    """clamp() returns the next tip gap of a sequence (off-centre grasp tests)."""

    def __init__(self, gaps):
        self.gaps = list(gaps)
        self.cur = self.gaps[0]

    def clamp(self, pct):
        self.cur = self.gaps.pop(0) if self.gaps else self.cur
        return {"ok": True, "held": True, "open_deg": 0.5, "gap_mm": self.cur, "load_pct": 22.0}

    def status(self):
        d = super().status()
        d["gap_mm"] = self.cur
        return d


class ShortArm(T45.DummyArm):
    """Stops 2 deg short on J7 during the insert; the corrective move either works or not."""

    def __init__(self, fix=True):
        super().__init__()
        self.fix = fix

    def exec_path(self, path, times, label):
        q = list(path[-1])
        if ("strict correction" in label and not self.fix) or ("insert" in label and "strict correction" not in label):
            q[6] += math.radians(2.0)
        self._q = q
        return {"ok": True}


def egg_io(arm=None, grip=None, wrist=False):
    io = RT.RealIO(arm or T45.DummyArm(), grip or T45.Grip(), lambda: {}, log=lambda _s: None, motion_allowed=True,
                   wrist_fn=(lambda: FRAME) if wrist else None)
    io.sleep = lambda _s: None
    io.cfg["still_window_s"] = 0.0
    io.cfg["settle_s"] = 0.0
    base = T45.make_io()
    io.perceive_eggs = base.perceive_eggs
    io.logs = []
    io.log = io.logs.append
    return io


def egg_cfg(**over):
    import NERO_V349_Real_Wrist_Camera as W49
    base = W49.egg_task_config()
    base.update(over)
    return lambda path=None: dict(base)


class Patch:
    def __init__(self, mod, name, fn):
        self.mod, self.name, self.fn = mod, name, fn

    def __enter__(self):
        self.old = getattr(self.mod, self.name)
        setattr(self.mod, self.name, self.fn)

    def __exit__(self, *a):
        setattr(self.mod, self.name, self.old)


def egg_reads(seq):
    """fake wrist egg detector: returns the next delta (m) of `seq` relative to the expected position."""
    it = iter(seq)

    def f(rgb, K, T_fl, calib, exp_xyz, kind="egg", undistort_intr=None, **_):
        d = next(it)
        if d is None:
            return {"ok": False, "reason": "test: no lock"}
        n = float(np.hypot(*d)) * 1000.0
        return {"ok": True, "trusted": n <= 25.0, "delta_xy_m": list(d), "shift_norm_mm": n, "kind": kind}
    return f


class TestV46Egg(unittest.TestCase):
    def _sk(self, io):
        sk = RT.RealEggSkills(io)
        sk.observe_scene()
        self.assertTrue(sk.plan_egg_place("egg_1", "slot_12")["ok"])
        sk._slot_looked = True              # slot look tested separately (E4)
        sk._slot_wrist_shift = None
        return sk

    def test_e1_camera_agreement(self):
        """V4.6.1 (user decision): the wrist egg reading is logged only - no shift, no second look, no stop."""
        for d in ((0.003, -0.002), (-0.0066, 0.0167), (0.030, 0.0), None):
            sk = self._sk(egg_io(wrist=True))
            g0 = np.asarray(sk.plan["_wps"]["grasp"][0], float).copy()
            egg0 = np.asarray(sk.cam["egg_1"]["rim"], float).copy()
            with Patch(RWC, "wrist_check_egg_or_slot", egg_reads([d, d])):
                r = sk.pick_egg("egg_1")
            self.assertTrue(r["ok"], (d, r))
            a = r["physical_verification"]["camera_agreement"]["egg_1"]
            self.assertEqual(a["action"], "log_only")
            np.testing.assert_allclose(np.asarray(sk.plan["_wps"]["grasp"][0]), g0)   # grasp NOT moved
            np.testing.assert_allclose(sk.cam["egg_1"]["rim"], egg0)
            if d is not None:
                self.assertTrue(any("只记录，按主相机位置夹" in s_ for s_ in sk.io.logs))

    def test_e2_off_centre_grasp(self):
        """V4.6.1: judged against the measured baseline held_gap_mm (26.5) - margin 6; default = log only."""
        sk = self._sk(egg_io(grip=SeqGrip([26.5])))                 # the user's three good grasps
        r = sk.pick_egg("egg_1")
        self.assertTrue(r["ok"], r)
        self.assertFalse(r["physical_verification"]["camera_agreement"].get("offcentre_suspected", False))
        sk = self._sk(egg_io(grip=SeqGrip([18.0])))                  # clearly off centre, default: log only
        r = sk.pick_egg("egg_1")
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["physical_verification"]["camera_agreement"]["offcentre_suspected"])
        self.assertEqual(sk.held, "egg_1")
        import NERO_V349_Real_Wrist_Camera as W49
        with Patch(W49, "egg_task_config", egg_cfg(offcentre_regrasp=True)):
            sk = self._sk(egg_io(grip=SeqGrip([18.0, 26.4])))
            r = sk.pick_egg("egg_1")
            self.assertTrue(r["ok"], r)
            self.assertTrue(any("松开、抬起、重新看、再夹一次" in s_ for s_ in sk.io.logs))
            sk = self._sk(egg_io(grip=SeqGrip([18.0, 17.0])))
            r = sk.pick_egg("egg_1")
            self.assertEqual(r["reason"], "EGG_GRASP_OFF_CENTRE")

    def test_e3_strict_arrival_before_release(self):
        for fix in (True, False):
            io = egg_io(arm=ShortArm(fix=fix))
            opened = []
            g_open = io.gripper_open
            io.gripper_open = lambda mm=None: (opened.append(mm), g_open(mm))[1]
            sk = self._sk(io)
            self.assertTrue(sk.pick_egg("egg_1")["ok"])
            self.assertTrue(sk.orient_for_slot("slot_12")["ok"])
            n_open = len(opened)
            r = sk.place_egg_into_slot("slot_12")
            if fix:
                self.assertTrue(r["ok"], r)
                self.assertTrue(any("小幅修正一次" in s for s in io.logs))
                rc = r["physical_verification"]["release_check"]
                self.assertLess(max(abs(v) for v in rc["tip_xy_err_mm"]), 1.0)
            else:
                self.assertFalse(r["ok"])
                self.assertEqual(r["reason"], "INSERT_NOT_PRECISE")
                self.assertEqual(len(opened), n_open)          # the egg was NOT released
                self.assertEqual(sk.held, "egg_1")

    def test_e4_prepick_slot_look_and_detector(self):
        # cavity detector on a synthetic wrist image: true slot (+8, -5) mm from the main-camera estimate
        import importlib.util
        spec = importlib.util.spec_from_file_location("synth", os.path.join(TESTS, "v46_synth_slot.py"))
        syn = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(syn)
        img, K, T, he, exp = syn.scene()
        r = RWC.wrist_find_slot_cavity(img, K, T, he, exp)
        self.assertTrue(r["ok"] and r["trusted"], r)
        self.assertAlmostEqual(r["delta_mm"][0], 8.0, delta=1.0)
        self.assertAlmostEqual(r["delta_mm"][1], -5.0, delta=1.0)
        self.assertGreaterEqual(r["n_neighbours"], 1)
        img2, K, T, he, exp = syn.scene(dark=175)                 # no cavities visible -> not ok
        self.assertFalse(RWC.wrist_find_slot_cavity(img2, K, T, he, exp)["ok"])
        # pre-pick look in the skill (V4.6.1: behind the switch wrist_slot_look_before_pick, default off)
        import NERO_V349_Real_Wrist_Camera as W49
        self._cfg_patch = Patch(W49, "egg_task_config", egg_cfg(wrist_slot_look_before_pick=True))
        self._cfg_patch.__enter__()
        self.addCleanup(self._cfg_patch.__exit__)
        io = egg_io(wrist=True)
        sk = RT.RealEggSkills(io)
        sk.observe_scene()
        self.assertTrue(sk.plan_egg_place("egg_1", "slot_12")["ok"])
        s0 = np.asarray(sk.cam["slot_12"]["rim"], float).copy()
        reads = iter([(0.006, -0.004), (0.0065, -0.0042)])

        def fake_cav(rgb, K, T_fl, calib, exp_xyz, **_):
            d = next(reads)
            return {"ok": True, "trusted": True, "measured_xyz": [exp_xyz[0] + d[0], exp_xyz[1] + d[1], exp_xyz[2]],
                    "delta_mm": [d[0] * 1000, d[1] * 1000], "radius_mm": 19.0, "n_neighbours": 2}
        with Patch(RWC, "wrist_find_slot_cavity", fake_cav), Patch(RWC, "wrist_check_egg_or_slot", egg_reads([None])):
            r = sk.pick_egg("egg_1")
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["physical_verification"]["camera_agreement"]["slot"]["accepted"])
        np.testing.assert_allclose(sk.cam["slot_12"]["rim"][:2] - s0[:2], [0.00625, -0.0041], atol=1e-6)
        self.assertEqual(sk.cam["slot_12"]["source"], "wrist")
        tip, _c, _R = T45.egg_tip_from_q(sk, sk.plan["_q"]["release"])
        self.assertLess(float(np.linalg.norm(tip[:2] - sk.cam["slot_12"]["rim"][:2])) * 1000.0, 1.0)
        # disagreeing reads -> not accepted, main-camera slot kept
        io = egg_io(wrist=True)
        sk = RT.RealEggSkills(io)
        sk.observe_scene()
        sk.plan_egg_place("egg_1", "slot_12")
        s0 = np.asarray(sk.cam["slot_12"]["rim"], float).copy()
        reads2 = iter([(0.006, 0.0), (-0.006, 0.0), (0.006, 0.0), (-0.006, 0.0), (0.006, 0.0), (-0.006, 0.0)])

        def fake_cav2(rgb, K, T_fl, calib, exp_xyz, **_):
            d = next(reads2)
            return {"ok": True, "trusted": True, "measured_xyz": [exp_xyz[0] + d[0], exp_xyz[1] + d[1], exp_xyz[2]],
                    "delta_mm": [d[0] * 1000, d[1] * 1000]}
        with Patch(RWC, "wrist_find_slot_cavity", fake_cav2), Patch(RWC, "wrist_check_egg_or_slot", egg_reads([None])):
            r = sk.pick_egg("egg_1")
        self.assertTrue(r["ok"], r)
        self.assertFalse(r["physical_verification"]["camera_agreement"]["slot"]["accepted"])
        np.testing.assert_allclose(sk.cam["slot_12"]["rim"], s0)

    def test_e5_log_rotation_and_calibration_arrival(self):
        import NERO_V312_Console as C
        d = tempfile.mkdtemp()
        for i in range(14):
            log = os.path.join(d, "console.log")
            with open(log, "w") as f:
                f.write(f"run {i}\n")
            os.utime(log, (1_700_000_000 + i * 60, 1_700_000_000 + i * 60))
            C.v46_rotate_console_log(d, log)
        olds = sorted(f for f in os.listdir(d) if f.startswith("console_"))
        self.assertEqual(len(olds), C.LOG_KEEP)
        self.assertEqual(open(os.path.join(d, "console.prev.log")).read(), "run 13\n")
        # V4.6.1: the arrival wait moved from the (frozen since V3.80) NERO_V325_Camera.py into
        # RealPipeline.move_and_settle, which the calibration photos use (tests/test_v4_6_1.py N8)
        rr = open(os.path.join(HERE, "NERO_V313_Real_Robot.py"), encoding="utf-8").read()
        self.assertIn("act = self._v46_wait_arrival(model, tgt, label)", rr)

    def test_e6_markers(self):
        with open(os.path.join(HERE, "VERSION")) as f:
            self.assertTrue(f.read().strip().startswith(("V4.6", "V4.7", "V4.8", "V4.8.1", "V4.8.2", "V4.8.3", "V4.8.4", "V4.8.5")))
        self.assertEqual(RT.MARKER_V46, "V4.6 EGG CAMERA AGREEMENT + STRICT ARRIVAL")
        self.assertEqual(RWC.MARKER_V46_SLOT, "V4.6 WRIST SLOT CAVITY DETECTOR")
        self.assertEqual((RT.STRICT_POS_MM, RT.STRICT_ROT_DEG, RT.STRICT_JOINT_DEG), (2.0, 1.0, 0.5))


if __name__ == "__main__":
    unittest.main(verbosity=1)
