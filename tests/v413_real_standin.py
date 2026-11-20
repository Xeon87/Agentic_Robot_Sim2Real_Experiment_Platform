#!/usr/bin/env python3
"""NERO V4.13 REAL STAND-IN (test support, not the product): the REAL general pick & place code path - RealIO
(NERO_V365_Real_Cup_Task), its perceive_general (YOLOE result handling, YP.locate_objects, the V4.13 real height map),
the AgxGripper class (NERO_V48_Gripper) and H1 V1 - driven by a MuJoCo twin instead of the hardware.

What is replaced (everything below RealIO / the gripper SDK / the camera driver / YOLOE):
  arm      TwinArm: q() / exec_path(path, times) move the twin's gripper along the joint path (FastArm kinematics, the
           same as the real planner's); a fingertip below the table is refused like the real fence guard
  gripper  TwinAgxBackend: the AgxGripper SDK commands (move width + force, status) on the twin fingers - real-like:
           the fingers close at ~60 mm/s with a capped force (they push an object until both pads touch, then the
           object is held - a weld, as in all twins: MuJoCo friction alone creeps); the reported width is the real
           pad gap (twin opening - 13.8 mm, measured); the twin gripper is shifted 10.3 mm toward the wrist so its
           fingertips are where the real AgxGripper model has them (19.7 mm past the grasp point)
  camera   the twin's fixed D435i (NERO_V308 pose / intrinsics): RGB + depth rendered (collision geometry), depth
           noise k * d^2, z16 quantisation, 5 frames; optional hand-eye error on T_base_cam
  YOLOE    MaskDetector: masks from the twin's segmentation render, named by the prompt that matches the object's
           label (synonym table, like an open-vocabulary detector); optional misses
What is NOT tested here: YOLOE on real images, the real depth on dark / shiny / transparent objects, real friction,
the real hand-eye accuracy, the real gripper's force and its CAN feed.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
for _p in (HERE, os.path.dirname(os.path.realpath(__file__))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GAP_OFFSET_MM = 13.8            # twin opening - pad (stud) gap, measured in the twin (V4.13)
TIP_SHIFT_M = 0.0103            # sim fingertip 30.0 mm past the grasp point, real model 19.7 mm
CLOSE_SPEED_M_S = 0.030         # per finger (60 mm/s gap), the AgxGripper at 40 % speed
SYN = {
    "egg": ("egg", "eggs"),
    "cup": ("cup", "mug", "paper cup", "cups"),
    "mug": ("mug", "cup", "coffee mug"),
    "bowl": ("bowl",),
    "apple": ("apple", "fruit"),
    "bottle": ("bottle", "water bottle"),
    "remote": ("remote", "remote control", "tv remote"),
    "tray": ("tray",),
    "storage_box": ("storage box", "box", "bin", "basket"),
    "egg_carton": ("egg carton", "carton", "egg holder", "egg tray"),
    "block": ("block", "cube"), "ball": ("ball",), "can": ("can",), "plate": ("plate",),
}


def _label_matches(label, prompt):
    p = str(prompt).strip().lower()
    lab = str(label).lower()
    names = SYN.get(lab, (lab.replace("_", " "),))
    return p in names or p == lab or p.replace(" ", "_") == lab


class TwinArm:
    def __init__(self, sb):
        import NERO_V362_Cup_Skills as CK
        self.sb, self.model = sb, {"joints": [f"joint{i}" for i in range(1, 8)], "limits": CK.JOINT_LIMITS}
        self.paths = []

    def q(self):
        return list(self.sb.tw.q)

    def exec_path(self, path, times, label):
        tw = self.sb.tw
        for q in path:                                   # the real fence guard: no fingertip below the table
            z = self.sb.fingertip_z(q)
            if z < self.sb.table_z - 0.001:
                return {"ok": False, "error": f"FENCE: fingertip {1000 * (z - self.sb.table_z):.1f} mm below the table "
                                              f"({label})", "fence_violation": True}
        self.paths.append((label, len(path), float(times[-1])))
        tw._run_q_path([list(map(float, p)) for p in path], max(0.2, float(times[-1]) * self.sb.time_scale))
        return {"ok": True, "kind": "path"}


class TwinAgxBackend:
    """the AgxGripper SDK requests (NERO_V48_Gripper backend interface) on the twin fingers"""

    def __init__(self, sb):
        self.sb, self.target_gap, self.force_set, self.enabled = sb, 60.0, 1.0, True
        self.log = []

    def request(self, cmd):
        c = cmd.get("cmd")
        self.log.append(dict(cmd))
        sb = self.sb
        if c == "status":
            sb.advance(0.03)
            g = sb.gap_mm()
            blocked = sb.mode == "close" and sb.tw.hold is not None
            return {"ok": True, "width_m": round(g / 1000.0, 6), "force": self.force_set if blocked else 0.05,
                    "is_ok": True, "fps": 200.0, "foc": {"driver_enable_status": self.enabled, "homing_status": True}}
        if c == "move":
            w = float(cmd.get("width_m", 0.0)) * 1000.0
            self.force_set = float(cmd.get("force", 1.0))
            sb.command_gap(w, self.force_set)
            sb.advance(0.04)
            return {"ok": True}
        if c == "teach_param":
            return {"ok": True, "max_range_m": 0.1}
        if c in ("disable", "reset", "ping"):
            return {"ok": True}
        return {"ok": False, "error": f"unknown cmd {c}"}

    def close(self):
        pass


class MaskDetector:
    """YOLOE stand-in: detect(rgb, mode, path, prompts, conf) -> [{cls, name, conf, box, mask}] (text mode: cls 0, name
    = the prompt, as NERO_V347_Yolo_Perception.parse_result)"""

    def __init__(self, sb, miss=()):
        self.sb, self.miss = sb, set(miss)
        self.info = {"device": "stand-in"}

    def detect(self, rgb, mode, path, prompts=None, conf=0.25):
        seg = self.sb.last_seg
        out = []
        for oid, mask in seg.items():
            if oid in self.miss or mask.sum() < 30:
                continue
            lab = self.sb.label_of(oid)
            name = next((p for p in (prompts or []) if _label_matches(lab, p)), None)
            if name is None:
                continue
            vv, uu = np.nonzero(mask)
            box = [float(uu.min()), float(vv.min()), float(uu.max()), float(vv.max())]
            out.append({"cls": 0, "name": str(name).strip().lower(), "conf": 0.9, "box": box, "mask": mask.copy()})
        return out


class StandIn:
    def __init__(self, twin, depth_k=0.0025, frames=5, calib_err_mm=0.0, calib_err_deg=0.0, seed=0, miss=(),
                 size=(848, 480), time_scale=1.0, log=None, cfg=None):
        import mujoco
        import NERO_V308_Camera_Perception as P8
        import NERO_V362_Cup_Skills as CK
        import NERO_V365_Real_Cup_Task as RT
        import NERO_V48_Gripper as GG
        self.mj, self.tw, self.CK, self.RT = mujoco, twin, CK, RT
        self.rng = np.random.default_rng(seed)
        self.table_z, self.time_scale = 0.0, float(time_scale)
        self.mode, self.goal_gap, self.force_n, self.q_hold, self.hold_gap = "open", 60.0, 10.0, 0.0, 0.0
        self.logs = []
        self._log = log or (lambda s: self.logs.append(str(s)))
        # the gripper: real-like fingers, mounted with the real fingertip position
        self._shift = np.eye(4)
        self._shift[2, 3] = -TIP_SHIFT_M
        tw = twin
        orig_set = tw._set_tcp
        shift = self._shift

        def _apply_q(q):
            T = tw.arm.fk(np.asarray(q, float)) @ shift
            orig_set(T[:3, 3], T[:3, :3])
        tw._apply_q = _apply_q
        tw.step = self._step
        tw.force_mode = False
        _apply_q(tw.q)
        self.command_gap(60.0, 1.0)
        self.advance(0.6)
        # the camera (the backend's fixed D435i), rendered at size
        os.environ.setdefault("NERO_V308_WIDTH", str(size[0]))
        os.environ.setdefault("NERO_V308_HEIGHT", str(size[1]))
        self.cam_cfg = dict(P8.camera_config(), width=int(size[0]), height=int(size[1]), depth_noise_k=float(depth_k))
        self.K = P8.intrinsics(self.cam_cfg)
        self.T_true = P8.extrinsics(self.cam_cfg)
        self.T_calib = self._calib(self.T_true, calib_err_mm, calib_err_deg)
        self.frames = int(frames)
        m = tw.m
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), int(size[0]))
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), int(size[1]))
        self.size = (int(size[0]), int(size[1]))
        self.r = mujoco.Renderer(m, height=int(size[1]), width=int(size[0]))
        self.opt = mujoco.MjvOption()
        for i in range(len(self.opt.geomgroup)):
            self.opt.geomgroup[i] = 1
        self.geom_oid = {}
        import NERO_V49_Geometry as G49
        for gi in range(m.ngeom):
            b = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gi])) or ""
            g = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
            oid = G49.object_id_for(b, g)
            if oid:
                self.geom_oid[gi] = oid
        self.last_seg = {}
        self.det = MaskDetector(self, miss)
        GG.set_active("agx")
        self.grip = GG.AgxGripper(backend=TwinAgxBackend(self))
        self.arm = TwinArm(self)
        c = dict(cfg or {})
        c.setdefault("settle_s", 0.02)
        c.setdefault("still_window_s", 0.03)
        c.setdefault("joint_vmax_deg_s", 40.0)
        c.setdefault("line_vmax_mm_s", 40.0)
        self.io = RT.RealIO(self.arm, self.grip, self.camera, log=self._log, cfg=c, motion_allowed=True)

    # -- calibration error
    def _calib(self, T, err_mm, err_deg):
        if not err_mm and not err_deg:
            return np.array(T, float)
        v = self.rng.normal(size=3)
        v /= np.linalg.norm(v)
        ax = self.rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        a = math.radians(err_deg)
        Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R = np.eye(3) + math.sin(a) * Kx + (1 - math.cos(a)) * Kx @ Kx
        T2 = np.array(T, float).copy()
        T2[:3, :3] = R @ T2[:3, :3]
        T2[:3, 3] = T2[:3, 3] + v * err_mm / 1000.0
        return T2

    # -- gripper physics
    def gap_mm(self):
        return max(0.0, self.tw.opening_mm() - GAP_OFFSET_MM)

    def command_gap(self, gap_mm, force_setting):
        g = float(gap_mm)
        self.force_n = float(min(20.0, max(2.0, 8.0 * float(force_setting))))
        ref = self.hold_gap if self.tw.hold is not None else self.gap_mm()   # a held object: its width decides
        if g < ref - 0.5:
            self.mode = "close"
        else:
            self.mode = "open"
            if self.tw.hold is not None:
                self.tw._release_hold()
        self.goal_gap = g

    def advance(self, s):
        self._step(int(max(1, round(float(s) / 0.001))))

    def _step(self, n=1):
        tw, d = self.tw, self.tw.d
        for _ in range(int(n)):
            q, qd = float(tw.j1.qpos[0]), float(tw.j1.qvel[0])
            if self.mode == "close":
                lt, rt, _ln, _rn, worst, bid = tw._pad_info()
                q_goal = (self.goal_gap + GAP_OFFSET_MM) / 2000.0
                if tw.hold is None and lt and rt and bid >= 0:
                    tw._activate_hold(bid)                       # both pads on the object: held (twin weld)
                    self.q_hold, self.hold_gap = q, self.gap_mm()
                if tw.hold is not None:                          # held: the fingers stay where they closed (a stiff
                    d.ctrl[tw.fa] = float(np.clip(3000.0 * (self.q_hold - q) - 30.0 * qd,   # PD (20 N cap); the weld
                                                  -20.0, 20.0))                             # carries the object
                    d.ctrl[tw.ga] = float(self.q_hold)
                else:
                    f = 0.0 if q <= q_goal + 1e-5 else float(np.clip(200.0 * (-CLOSE_SPEED_M_S - qd),
                                                                     -self.force_n, self.force_n))
                    d.ctrl[tw.fa] = f
                    d.ctrl[tw.ga] = min(max(q, 0.0), 0.05)
            else:
                d.ctrl[tw.fa] = 0.0
                d.ctrl[tw.ga] = float(np.clip((self.goal_gap + GAP_OFFSET_MM) / 2000.0, 0.0, 0.05))
            self.mj.mj_step(tw.m, d)
            tw.t += 0.001

    def fingertip_z(self, q):
        T = self.tw.arm.fk(np.asarray(q, float)) @ self._shift
        tip = T[:3, :3] @ np.array([0.0, 0.0, 0.0297]) + T[:3, 3]   # the twin studs reach 29.7 mm along TCP z
        return float(tip[2])

    # -- camera + detector
    def label_of(self, oid):
        import NERO_V49_Geometry as G49
        return G49.label_of(oid)

    def render(self):
        import NERO_V308_Camera_Perception as P8
        mj, r, tw = self.mj, self.r, self.tw
        mj.mj_forward(tw.m, tw.d)
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        r.update_scene(tw.d, camera=P8.CAMERA_NAME, scene_option=self.opt)
        rgb = r.render().copy()
        seg = None
        for _try in range(3):
            try:
                r.enable_segmentation_rendering()
                r.update_scene(tw.d, camera=P8.CAMERA_NAME, scene_option=self.opt)
                seg = r.render().copy()
                r.disable_segmentation_rendering()
                break
            except IndexError:                  # MuJoCo's segmentation decode fails now and then under OSMesa
                try:
                    r.close()
                except Exception:
                    pass
                self.r = r = mj.Renderer(tw.m, height=self.size[1], width=self.size[0])
        if seg is None:
            raise RuntimeError("stand-in: segmentation render failed 3 times")
        r.enable_depth_rendering()
        r.update_scene(tw.d, camera=P8.CAMERA_NAME, scene_option=self.opt)
        raw = r.render().copy()
        r.disable_depth_rendering()
        ids, types = seg[:, :, 0], seg[:, :, 1]
        masks = {}
        geom = types == int(mj.mjtObj.mjOBJ_GEOM)
        for gi, oid in self.geom_oid.items():
            m = geom & (ids == gi)
            if m.any():
                masks[oid] = masks.get(oid, np.zeros(ids.shape, bool)) | m
        self.last_seg = masks
        depths = [P8._depth_postprocess(raw, self.cam_cfg, self.rng) for _ in range(self.frames)]
        return rgb, depths

    def camera(self):
        rgb, depths = self.render()
        return {"rgb": rgb, "depths": depths, "K": dict(self.K), "T_base_cam": self.T_calib.tolist(),
                "z_table": self.table_z, "fence_poly": None, "calib_note": "stand-in",
                "yolo": {"detector": self.det, "path": "standin.pt", "prompts": ["object"], "conf": 0.25}}


def make_skills(sb, profile="V1"):
    import NERO_V489_General_Pick_Place as GP
    sb.io.reset_object_memory()
    sb.io.home_q = sb.io.current_q()
    return GP.GeneralPickPlaceSkills(sb.io, profile=profile)


def household(spec, **kw):
    import NERO_V410_Household_Scenes as HS
    sc = HS.scene_from_spec(spec)
    tw = HS.make_twin(sc)
    return StandIn(tw, **kw), sc


def egg_scene(**kw):
    import NERO_V489_General_Pick_Place as GP
    return StandIn(GP.TwinGeneralIO(), **kw)


EGG_CUP_LAYOUT = {"egg_1": (-0.42, -0.06), "egg_2": (-0.43, 0.09), "cup_1": (-0.33, -0.18), "cup_2": (-0.34, 0.20)}


def egg_cup_scene(layout=None, source="egg_1", target="cup_1", **kw):
    """the first real task (egg -> cup) laid out for the REAL finger size (stock AgxGripper pads are 26.5 mm thick:
    the V4.8.10 simulation layout has an egg 23 mm from the carton, too tight for them)"""
    import NERO_V49_Random_Scenes as RS
    rng = np.random.default_rng(0)
    objs = []
    for oid, xy in (layout or EGG_CUP_LAYOUT).items():
        kind, idx = oid.rsplit("_", 1)
        objs.append(RS._make_obj(kind, int(idx), rng, tuple(xy), list(RS.COLORS)[len(objs) % len(RS.COLORS)]))
    sc = RS.Scene(0, objs, source, target, "into", f"Put {source} into {target}.")
    sc.world_xml = "\n    ".join(o.xml for o in objs)
    sc.hold_names = [f"obj_{o.oid}" for o in objs if o.graspable]
    return StandIn(RS.make_twin(sc), **kw), sc
