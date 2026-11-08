#!/usr/bin/env python3
"""NERO V4.5 offline tests — egg place accuracy + fingertip z safety (+ V4.4 crash hardening).

T1 held-egg geometry: the egg centre in the grasp frame is MEASURED at the pick (~20 mm along the jaw axis, not the
   V4.3 constant 8.5 mm) and the planned release joints put the egg's LOWER TIP on the slot centre (< 1 mm, via FK);
   the V4.3/4.4 formula would have missed by > 10 mm.
T2 egg_place_grasp_origin: lower tip lands on the target for any pitch (tilt compensated).
T3 vertical pitch is tried first.
T4 fingertip z: lowest tip drop (fixed + moving jaw), pad at the equator, small eggs raised so the tip keeps >= 5 mm,
   fence check uses the fingertip (pad-centre check would pass where the tip fails), execution tool_T = fingertip.
T5 conservative table height (higher of camera / fence; refuse when they differ > 15 mm).
T6 two-stage descent (last 15 mm at 10 mm/s).
T7 pre-orient wrist look: trusted 1.5-15 mm shift re-plans the insertion; untrusted / too large -> unchanged.
T8 slot cavity refinement + measured pitch config.
T9 crash hardening markers (env only on the main thread, loads off the main thread, console.prev.log).
No robot, no ROS, no camera needed.
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

import NERO_V362_Cup_Skills as CK  # noqa: E402
import NERO_V365_Real_Cup_Task as RT  # noqa: E402

EGG_C = [-0.397, -0.031, 0.021]


class DummyArm:
    def __init__(self):
        self._q = [0.0, -0.35, 0.0, 0.85, 0.0, 0.50, 0.0]

    def q(self):
        return list(self._q)

    def exec_path(self, path, times, label):
        self._q = list(path[-1])
        return {"ok": True}


class Grip:
    connected = True
    held = True
    cfg = {"port": "sim", "baud": 1000000, "servo_id": 1, "open_sign": 1,
           "stall_torque_nm": 3.0, "touch_deg": 0.0, "touch_calibrated": True}

    def open_deg(self, deg):
        return {"ok": True, "open_deg": deg, "gap_mm": 52.0}

    def clamp(self, pct):
        return {"ok": True, "held": True, "open_deg": 0.5, "gap_mm": 38.0, "load_pct": 22.0}

    def status(self):
        return {"open_deg": 0.5, "servo_deg": 179.5, "gap_mm": 38.0, "load_pct": 22.0, "current_ma": 160.0,
                "voltage_v": 12.0, "temp_c": 30, "torque_limit_pct": 25.0, "errors": []}


def make_io(fence_z=None, egg=None, wrist_fn=None):
    io = RT.RealIO(DummyArm(), Grip(), lambda: {}, log=lambda _s: None, motion_allowed=True, wrist_fn=wrist_fn)
    io.sleep = lambda _s: None
    io.cfg["still_window_s"] = 0.0
    io.cfg["settle_s"] = 0.0
    e = dict({"center_base": list(EGG_C), "grasp_yaw_deg": -120.0, "width_m": 0.042, "length_m": 0.055}, **(egg or {}))

    def perceive(tag):
        occ = tag == "verify_egg_in_slot"
        slots = [{"slot_id": 1, "center_base": [-0.264, -0.015, 0.022], "occupied": False},
                 {"slot_id": 12, "center_base": [-0.323, +0.112, 0.022], "occupied": occ}]
        out = {"ok": True, "scene": "fill_egg_holder", "table_z_m": 0.0, "fence_z_table_m": fence_z, "n_2d": 3,
               "eggs": [] if occ else [e],
               "carton": {"center_base": [-0.283, 0.149, 0.025], "yaw_deg": -6.0, "slots": slots,
                          "free_slots": [s["slot_id"] for s in slots if not s["occupied"]],
                          "occupied_slots": [s["slot_id"] for s in slots if s["occupied"]]},
               "dets": [], "cups": []}
        io.last_perception = out
        return out
    io.perceive_eggs = perceive
    return io


def planned(io, slot="slot_12"):
    sk = RT.RealEggSkills(io)
    sk.observe_scene()
    r = sk.plan_egg_place("egg_1", slot)
    return sk, r


def egg_tip_from_q(sk, q, slot_world_unused=None):
    """FK of the planned release joints -> grasp frame -> held egg (measured offset) -> its lower tip."""
    T_tcp = sk._fast.fk(np.asarray(q, float))
    T_g = T_tcp @ np.linalg.inv(sk.prof.T_grasp_tcp)
    R, p_a = T_g[:3, :3], T_g[:3, 3]
    e_G = np.asarray(sk.plan["egg_in_G_m"], float)
    centre = p_a + R @ e_G
    ax = R[:, 0]
    down = ax if ax[2] <= 0 else -ax
    return centre + float(sk.plan["egg_half_len_m"]) * down, centre, R


class TestV45Egg(unittest.TestCase):
    def test_t1_held_egg_offset_and_lower_tip_on_slot(self):
        sk, r = planned(make_io())
        self.assertTrue(r["ok"], r)
        e = np.asarray(sk.plan["egg_in_G_m"]) * 1000.0
        self.assertAlmostEqual(e[0], 0.0, delta=0.5)
        self.assertAlmostEqual(e[1], 20.3, delta=0.5)          # half egg width - jaw wall, NOT 8.5 mm (V4.3)
        slot = np.asarray(sk.cam[sk.plan["target_slot"]]["rim"], float)
        tip, centre, R = egg_tip_from_q(sk, sk.plan["_q"]["release"])
        self.assertLess(float(np.linalg.norm(tip[:2] - slot[:2])) * 1000.0, 1.0)
        self.assertAlmostEqual(float(tip[2]), float(sk.plan["slot_tip_target"][2]), delta=0.001)
        # what the V4.3/4.4 formula did with the same release pose: centre over the slot assuming an 8.5 mm offset
        old_centre = (np.asarray(sk.plan["_wps"]["release"][0]) * 0 + (centre - R @ np.asarray(sk.plan["egg_in_G_m"]))
                      + R @ np.array([0.0, RT.EGG_HELD_CENTER_Y_M, 0.0]))
        err_old_mm = float(np.linalg.norm(old_centre[:2] - centre[:2])) * 1000.0
        self.assertGreater(err_old_mm, 10.0)                    # the systematic V4.4 error the user saw
        # the whole executed sequence still works end to end
        res = RT.run_execute(make_io(), "egg_1", "slot_1", scene="fill_egg_holder")
        self.assertTrue(res["ok"], [x.get("reason") for x in res["results"]])

    def test_t2_place_origin_compensates_tilt(self):
        e_G = np.array([0.0, 0.0203, -0.001])
        tgt = np.array([-0.3, 0.1, 0.01])
        for pitch in (-90.0, -75.0, -65.0, 70.0):
            sk = RT.RealEggSkills(make_io())
            R = sk._slot_vertical_grasp_R(150.0, pitch)
            p_a = RT.egg_place_grasp_origin(tgt, R, e_G, 0.0275)
            centre = p_a + R @ e_G
            ax = R[:, 0]
            down = ax if ax[2] <= 0 else -ax
            np.testing.assert_allclose(centre + 0.0275 * down, tgt, atol=1e-9)

    def test_t3_vertical_first(self):
        self.assertEqual(RT.EGG_PITCH_ORDER[:2], (-90.0, 90.0))
        self.assertEqual(sorted(RT.EGG_PITCH_ORDER, key=lambda p: abs(90 - abs(p))), list(RT.EGG_PITCH_ORDER))

    def test_t4_fingertip_z(self):
        drop = RT.so101_tip_drop_m()
        self.assertGreaterEqual(drop, CK.TIP_BEYOND_GRASP_M)
        self.assertLess(drop, CK.TIP_BEYOND_GRASP_M + 0.003)    # moving jaw ~1.5 mm deeper
        sk, r = planned(make_io())
        self.assertAlmostEqual(sk.plan["grasp_pad_z_mm"], 21.0, delta=0.1)      # equator of a 42 mm egg
        self.assertAlmostEqual(sk.plan["lowest_tip_z_mm"], 21.0 - drop * 1000.0, delta=0.1)
        self.assertGreaterEqual(sk.plan["lowest_tip_z_mm"], RT.TIP_PLAN_CLEAR_M * 1000.0)
        # a small 24 mm "egg": equator 12 mm would put the tips at ~0.5 mm -> pad raised so the tips keep 5 mm
        sk2, r2 = planned(make_io(egg={"width_m": 0.024, "center_base": [-0.397, -0.031, 0.012]}))
        self.assertTrue(r2["ok"], r2)
        self.assertAlmostEqual(sk2.plan["lowest_tip_z_mm"], 5.0, delta=0.1)
        # fence check uses the fingertip: lift the table so the pad still clears 5 mm but the tip does not
        q_g = sk.plan["_q"]["grasp"]
        self.assertTrue(sk._check_path_fence([q_g, q_g], 0.0))
        z_bad = (sk.plan["lowest_tip_z_mm"] - 2.0) / 1000.0     # tip now 2 mm above "table" (< 5), pad ~13 mm
        self.assertFalse(sk._check_path_fence([q_g, q_g], z_bad))
        # execution tool point = grasp origin moved by the tip drop along the approach axis
        T_l7g = RT.real_mount(None)[0]
        import NERO_V364_SO101_Gripper as G
        d = RT.so101_tip_tool_T(T_l7g)[:3, 3] - (np.asarray(T_l7g) @ G.T_gripper_grasp())[:3, 3]
        self.assertAlmostEqual(float(np.linalg.norm(d)), drop, places=6)
        self.assertGreaterEqual(RT.TIP_EXEC_GUARD_M, 0.003)

    def test_t5_table_height(self):
        z, src, err = RT.conservative_table_z({"table_z_m": 0.002, "fence_z_table_m": 0.006})
        self.assertAlmostEqual(z, 0.006)
        self.assertIsNone(err)
        z, src, err = RT.conservative_table_z({"table_z_m": 0.010, "fence_z_table_m": None})
        self.assertAlmostEqual(z, 0.010)
        z, src, err = RT.conservative_table_z({"table_z_m": 0.0, "fence_z_table_m": 0.020})
        self.assertIsNotNone(err)
        sk, r = planned(make_io(fence_z=0.020))
        self.assertFalse(r["ok"])
        self.assertEqual(r["reason"], "TABLE_HEIGHT_DISAGREE")
        sk, r = planned(make_io(fence_z=0.004))                  # fence table 4 mm higher -> grasp 4 mm higher
        self.assertTrue(r["ok"], r)
        self.assertAlmostEqual(sk.plan["table_z_m"], 0.004)

    def test_t6_two_stage_descent(self):
        io = make_io()
        calls = []
        orig = io.move_line

        def spy(label, xyz, quat, speed=0.05, slow=None):
            calls.append((label, np.asarray(xyz, float).copy(), speed))
            return orig(label, xyz, quat, speed, slow)
        io.move_line = spy
        sk, r = planned(io)
        self.assertTrue(sk.pick_egg("egg_1")["ok"])
        dn = [c for c in calls if "descend" in c[0]]
        self.assertEqual(len(dn), 2)
        self.assertAlmostEqual(float(dn[0][1][2] - dn[1][1][2]), RT.EGG_SLOW_LAST_M, places=6)
        self.assertAlmostEqual(dn[1][2], RT.EGG_SLOW_SPEED)

    def _wrist_case(self, delta_xy, trusted=True):
        import NERO_V386_Real_Wrist_Camera as RWC
        frame = {"rgb": np.zeros((8, 8, 3), np.uint8), "K": {"fx": 1}, "calib_eih": {"T_flange_cam": np.eye(4).tolist()}}
        io = make_io(wrist_fn=lambda: frame)
        sk, r = planned(io)
        self.assertTrue(r["ok"], r)
        self.assertTrue(sk.pick_egg("egg_1")["ok"])
        before = list(sk.plan["slot_tip_target"])
        seen = {}
        orig = RWC.wrist_check_egg_or_slot

        def fake(rgb, K, T_flange, calib, exp_xyz, kind="egg", undistort_intr=None):
            seen["kind"] = kind
            seen["T_flange_z"] = float(np.asarray(T_flange)[2, 3])
            return {"ok": True, "trusted": trusted, "delta_xy_m": list(delta_xy),
                    "shift_norm_mm": float(np.hypot(*delta_xy)) * 1000.0}
        RWC.wrist_check_egg_or_slot = fake
        try:
            ro = sk.orient_for_slot(sk.plan["target_slot"])
        finally:
            RWC.wrist_check_egg_or_slot = orig
        self.assertTrue(ro["ok"], ro)
        self.assertEqual(seen.get("kind"), "slot")
        return sk, before

    def test_t7_wrist_look_corrects_slot(self):
        sk, before = self._wrist_case((0.006, -0.004))
        after = sk.plan["slot_tip_target"]
        self.assertAlmostEqual(after[0] - before[0], 0.006, delta=0.0005)
        self.assertAlmostEqual(after[1] - before[1], -0.004, delta=0.0005)
        self.assertEqual(sk.plan.get("wrist_slot_shift_mm"), [6.0, -4.0])
        slot = np.asarray(sk.cam[sk.plan["target_slot"]]["rim"], float)
        tip, _c, _R = egg_tip_from_q(sk, sk.plan["_q"]["release"])
        self.assertLess(float(np.linalg.norm(tip[:2] - (slot[:2] + [0.006, -0.004]))) * 1000.0, 1.0)
        self.assertTrue(sk.place_egg_into_slot(sk.plan["target_slot"])["ok"])
        for d, tr in (((0.004, 0.0), False), ((0.030, 0.0), True)):
            sk2, before2 = self._wrist_case(d, trusted=tr)
            self.assertEqual(sk2.plan["slot_tip_target"], before2)
            self.assertIsNone(sk2.plan.get("wrist_slot_shift_mm"))

    def test_t8_slot_cavity_refinement(self):
        import NERO_V349_Real_Wrist_Camera as W
        R_, C_, pitch, flange_extra = 2, 3, 47.0, 30.0
        # outer box = 3 x 47 + 2 x 10 flange + 30 mm extra on one side (lid strip) -> nominal grid is off
        L = C_ * pitch + 20.0 + flange_extra
        Wd = R_ * pitch + 20.0
        true_c = [((c + 0.5 - 0.5 * C_) * pitch - 0.5 * flange_extra, (r + 0.5 - 0.5 * R_) * pitch)
                  for r in range(R_) for c in range(C_)]
        uu, vv = np.meshgrid(np.arange(-L / 2, L / 2, 1.5), np.arange(-Wd / 2, Wd / 2, 1.5))
        u, v = uu.ravel(), vv.ravel()
        h = np.full(u.shape, 36.0)
        for k, (cu, cv) in enumerate(true_c):
            rr = np.hypot(u - cu, v - cv)
            if k == 4:                                   # slot 5 holds an egg (higher than the top)
                h[rr < 20.0] = 60.0
            else:
                h[rr < 17.0] = 12.0 + 24.0 * (rr[rr < 17.0] / 17.0) ** 2
        cents, info = W.refine_slot_grid(u, v, h, R_, C_, L / C_, Wd / R_, h_top_mm=36.0)
        self.assertGreaterEqual(info["n_cavities"], 4)
        err_nominal = max(math.hypot(((k % C_) + 0.5 - 0.5 * C_) * L / C_ - true_c[k][0],
                                     ((k // C_) + 0.5 - 0.5 * R_) * Wd / R_ - true_c[k][1]) for k in range(6))
        self.assertGreater(err_nominal, 10.0)                   # outer-box grid (V4.4): > 10 mm off
        err_fit = max(math.hypot(cents[k][0] - true_c[k][0], cents[k][1] - true_c[k][1]) for k in range(6))
        self.assertLess(err_fit, 2.0)                            # V4.5: grid fitted to the cavities (pitch too)
        self.assertAlmostEqual(info["pitch_mm"][0], pitch, delta=1.0)
        # with the measured pitch the refined grid is within 2 mm everywhere (also the occupied slot 5)
        cents2, info2 = W.refine_slot_grid(u, v, h, R_, C_, pitch, pitch, h_top_mm=36.0)
        err2 = max(math.hypot(cents2[k][0] - true_c[k][0], cents2[k][1] - true_c[k][1]) for k in range(6))
        self.assertLess(err2, 2.0)
        # no depression (flat top / no depth) -> nominal grid unchanged
        cents3, info3 = W.refine_slot_grid(u, v, np.full(u.shape, 36.0), R_, C_, pitch, pitch)
        self.assertEqual(info3["n_cavities"], 0)
        # measured pitch config: env + file
        os.environ["NERO_EGG_SLOT_PITCH_MM"] = "47,46.5"
        try:
            self.assertEqual(W.egg_carton_config()["slot_pitch_mm"], [47.0, 46.5])
        finally:
            os.environ.pop("NERO_EGG_SLOT_PITCH_MM", None)
        self.assertIsNone(W.egg_carton_config()["slot_pitch_mm"])   # shipped file: null
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"slot_pitch_mm": [48, 47]}, f)
        try:
            self.assertEqual(W.egg_carton_config(f.name)["slot_pitch_mm"], [48.0, 47.0])
        finally:
            os.unlink(f.name)

    def test_t9_markers_and_crash_hardening(self):
        with open(os.path.join(HERE, "VERSION")) as f:
            self.assertEqual(f.read().strip(), "V4.5")
        self.assertEqual(RT.MARKER_V45, "V4.5 EGG PLACE ACCURACY + FINGERTIP Z SAFETY")
        tab = open(os.path.join(HERE, "NERO_V4_Console_Tab.py"), encoding="utf-8").read()
        self.assertIn('self.q.put(("apply_env", st["layout"]))', tab)
        self.assertNotIn("        ST.apply_to_process_env(st[\"layout\"])", tab)
        on_loop = tab[tab.index("def on_run_vla_loop"):tab.index("def on_run_agent_task")]
        self.assertNotIn("\n        if not self.vla_policy.is_loaded", on_loop)   # no load on the Tk thread
        con = open(os.path.join(HERE, "NERO_V312_Console.py"), encoding="utf-8").read()
        self.assertIn("console.prev.log", con)
        import NERO_V44_Storage as ST
        os.environ["NERO_DRIVE_ROOT"] = tempfile.mkdtemp()
        try:
            L = ST.layout()
            ST.apply_to_process_env(L)
            self.assertEqual(ST.apply_to_process_env(L), [])     # second call writes nothing
        finally:
            os.environ.pop("NERO_DRIVE_ROOT", None)


if __name__ == "__main__":
    unittest.main(verbosity=1)
