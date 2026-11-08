#!/usr/bin/env python3
"""V3.65 test double: the MuJoCo SO-101 twin as the REAL body of NERO_V365_Real_Cup_Task.RealIO.

TwinArm      - arm adapter (q(), model, exec_path(path, times)) that drives the twin joints along exactly the joint
               path RealIO would send to MoveIt's /execute_trajectory (no re-planning), and records every call
TwinGripper  - the SO101RealGripper interface (open_deg / clamp / status / held) acting on the twin's SO-101 jaw,
               with the real angle <-> virtual opening conversion of NERO_V365_STS3215_Gripper
twin_camera  - camera_fn: the twin's rendered fixed D435i (RGB + depth + K + T_base_cam), like camera_from_tab;
               V3.66: with a YOLOE stand-in (FakeYoloE) whose instance masks come from MuJoCo's segmentation render
               (one mask per visible cup, a nested pair = one mask, like YOLOE on the real camera)
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, ROOT)

import NERO_V362_Cup_Skills as CK            # noqa: E402
import NERO_V365_STS3215_Gripper as GR       # noqa: E402


class TwinArm:
    def __init__(self, tw):
        self.tw = tw
        self.model = {"joints": [f"joint{i}" for i in range(1, 8)], "limits": [tuple(v) for v in CK.JOINT_LIMITS],
                      "arm_group": "arm"}
        self.calls = []

    def q(self):
        return [float(v) for v in self.tw.q]

    def exec_path(self, path, times, label):
        assert len(path) == len(times) >= 2 and times[0] == 0.0
        assert all(b > a for a, b in zip(times, times[1:])), "times must increase"
        assert max(abs(a - b) for a, b in zip(path[0], self.tw.q)) < 1e-6, "the path must start at the current joints"
        for p in path:
            for v, (lo, hi) in zip(p, CK.JOINT_LIMITS):
                assert lo - 1e-9 <= v <= hi + 1e-9, (label, "joint limit")
        fa = getattr(self, "_fa", None) or CK.FastArm()
        self._fa = fa
        self.calls.append({"label": label, "n": len(path), "T": times[-1], "q_end": [float(v) for v in path[-1]],
                           "min_z": min(float(fa.fk(__import__("numpy").asarray(p, float))[2, 3]) for p in path),
                           "max_step_deg": max(max(abs(math.degrees(a - b)) for a, b in zip(p, q))
                                               for p, q in zip(path[1:], path[:-1]))})
        self.tw._run_q_path(path, max(0.2, float(times[-1])))
        return {"ok": True}


class TwinGripper:
    def __init__(self, tw):
        self.tw = tw
        self.cfg = dict(GR.DEFAULTS)
        self.held = False
        self.connected = True
        self.clamps = []

    def status(self):
        a = GR.open_deg_for_virtual_mm(self.tw.opening_mm())
        return {"open_deg": round(a, 2), "gap_mm": round(GR.gap_mm_for_open_deg(max(0.0, a)), 2), "errors": []}

    def open_deg(self, a):
        r = self.tw.gripper_open(GR.virtual_mm_for_open_deg(a))
        self.held = False
        st = self.status()
        return {"ok": True, "open_deg": st["open_deg"], "gap_mm": st["gap_mm"], "load_pct": 3.0, "errors": [],
                "virtual_mm": r["opening_mm"]}

    def clamp(self, pct, squeeze_deg=None):
        f = pct / 100.0 * self.cfg["stall_torque_nm"] / GR.lever_m()
        self.clamps.append((pct, f))
        r = self.tw.gripper_clamp(min(8.0, f))
        self.held = bool((r["contact_left"] and r["contact_right"]) or r["hold_active"])
        st = self.status()
        return {"ok": True, "held": self.held, "how": "stalled", "open_deg": st["open_deg"], "gap_mm": st["gap_mm"],
                "load_pct": pct if self.held else 3.0, "current_ma": 300.0, "errors": []}


class FakeYoloE:
    """V3.66 test double of NERO_V347 YoloEDetector.detect: returns the instance masks prepared by the camera
    function for the frame it just rendered (cls 0 = target, name "cup")"""
    def __init__(self):
        self.next = []
        self.calls = 0
        self.info = {"device": "CPU"}

    def detect(self, rgb, mode, model_path, ref=None, prompts=None, conf=0.25):
        self.calls += 1
        return [d for d in self.next if d["conf"] >= conf]


def _seg_dets(tw):
    import mujoco
    import numpy as np
    import NERO_V308_Camera_Perception as P8
    r = tw._renderer
    r.enable_segmentation_rendering()
    try:
        r.update_scene(tw.d, camera=P8.CAMERA_NAME)
        seg = r.render().copy()
    finally:
        r.disable_segmentation_rendering()
    m = tw.m
    masks = []
    for name in ("obj_cup_1", "obj_cup_2"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        gids = [g for g in range(m.ngeom) if m.geom_bodyid[g] == bid]
        mk = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 0], gids)
        if mk.sum() >= 300:
            masks.append(mk)

    def box(mk):
        ys, xs = np.nonzero(mk)
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    if len(masks) == 2:
        a, b = box(masks[0]), box(masks[1])
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        # V3.71: a cup held ABOVE the other one is its own instance (only a nested pair - rims within 6 cm - is one)
        dz = abs(float(tw.cup_pose("cup_1")[0][2]) - float(tw.cup_pose("cup_2")[0][2]))
        if ix > 0.3 * min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])) and dz < 0.06:
            masks = [masks[0] | masks[1]]                   # nested pair: one YOLOE instance
    return [{"cls": 0, "name": "cup", "conf": 0.9, "box": box(mk), "mask": mk} for mk in masks]


def twin_camera(tw, detector=None):
    det = detector or FakeYoloE()

    def fn():
        rgb, depth, K, T = tw.render_fixed_camera()
        det.next = _seg_dets(tw)
        return {"rgb": rgb, "depths": [depth], "K": K, "T_base_cam": T, "z_table": 0.0, "fence_poly": None,
                "calib_note": None, "yolo": {"detector": det, "mode": "text", "path": "fake-yoloe.pt",
                                             "prompts": ["cup"], "conf": 0.25}}
    fn.detector = det
    return fn


def make(log=None, cup_xy=None, cfg=None, motion_allowed=True):
    import NERO_V365_Real_Cup_Task as RT
    tw = CK.TwinIO(log=log, cup_xy=cup_xy, gripper="so101", pose_mode="camera")
    arm, grip = TwinArm(tw), TwinGripper(tw)
    io = RT.RealIO(arm, grip, twin_camera(tw), log=log or (lambda s: None), cfg=cfg, motion_allowed=motion_allowed)
    return io, tw, arm, grip
