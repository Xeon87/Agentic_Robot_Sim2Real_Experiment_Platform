#!/usr/bin/env python3
"""NERO V4.8.9 GENERAL PICK & PLACE: LLM-orchestrated chunk-based pick-and-place skill.

Instead of hardcoding object classes or fixed scripts, this module exposes 6 atomic,
safety-guarded chunk tools to the LLM Agent so it can dynamically plan and execute
each chunk (perception prompts -> grasp strategy -> grasp execution -> place strategy ->
place execution -> clear-view verification) for arbitrary natural-language instructions
such as "pick up the egg and put it into the cup".

All safety invariants remain strictly enforced in deterministic code:
  - Conservative table height (max of camera plane and safety fence z_table)
  - 2.0 s planning budget cap per chunk (never hangs on unreachable poses)
  - Pre-IK T_l7 tool-table collision screening (NERO_V4613_Tool_Collision)
  - cuRobo default IK + 45-seed FastArm IK fallback
  - Full workspace polygon fence & joint margin checks before execution
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

VERSION = "V4.9.3"
MARKER_V493 = "V4.9.3 TASK-START RECOVERY + SAFETY STOP + SINGULARITY + EGG ROLL"
MARKER_V491 = "V4.9.1 DENSE LINES + REACHABILITY"
MARKER_V492 = "V4.9.2 REAL-LIKE PERCEPTION + LLM EVAL + COMPACT OUTPUTS"
MARKER = "V4.8.9 GENERAL PICK&PLACE CHUNK LLM"
MARKER_V49 = "V4.9 GENERAL FACTS"
PLAN_BUDGET_S = 2.0
MIN_PLAN_MARGIN_DEG = 8.0      # V4.9: the executed path is checked at >= 4 deg; plan with room to spare
GOOD_PLAN_MARGIN_DEG = 14.0
SINGULAR_SIGMA_MIN = 0.08     # V4.9.3: smallest singular value of the Jacobian (position / 0.3 m); top-down poses >= 0.24
DENSE_LINE_MARGIN_DEG = 5.0    # V4.9.1: every straight line the IO will execute, checked point by point (exec >= 4)
MIN_PLACE_MARGIN_DEG = 5.0     # V4.9: release / above-target (the executed lowering line is checked at >= 4 deg)

import NERO_V49_Geometry as G49  # noqa: E402


def _clean_prompts(prompts: Any) -> List[str]:
    if isinstance(prompts, str):
        parts = [p.strip() for p in re.split(r"[,，;；\n]+", prompts) if p.strip()]
        return parts or ["object"]
    if isinstance(prompts, (list, tuple)):
        out = [str(p).strip() for p in prompts if str(p).strip()]
        return out or ["object"]
    return ["object"]


def _parse_target_xyz(target_id_or_xyz: str) -> Optional[np.ndarray]:
    s = str(target_id_or_xyz or "").strip()
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if s.startswith("[") or ("," in s and len(nums) == 3):
        try:
            return np.array([float(nums[0]), float(nums[1]), float(nums[2])], float)
        except Exception:
            return None
    return None


class GeneralPickPlaceSkills:
    """Atomic chunk skills for LLM-driven general pick-and-place on the real NERO arm."""

    STAGES = ("IDLE", "OBSERVED", "GRASP_PLANNED", "HOLDING", "PLACE_PLANNED", "RELEASED", "VERIFIED")

    def __init__(self, io, skill=None):
        import NERO_V362_Cup_Skills as CK
        import NERO_V365_Real_Cup_Task as RT
        import NERO_V372_Sort_Skills as SK72
        self.CK = CK
        self.RT = RT
        self.SK72 = SK72
        self.io = io
        self.prof = io.gripper_profile()
        self.is_sim = bool(getattr(io, "is_sim", False))
        self.agx = bool(getattr(io, "is_agx", RT.gripper_is_agx()))
        self.skill = skill or {"code": "GP", "short_name": "General Pick&Place", "skill_id": "general_pick_place"}
        self._fast = getattr(io, "fast", None) or CK.FastArm()
        self.stage = "IDLE"
        self.held: Optional[str] = None
        self.held_obj: Optional[Dict[str, Any]] = None
        self.picked_from: Optional[np.ndarray] = None
        self.grasp_plan: Optional[Dict[str, Any]] = None
        self.place_plan: Optional[Dict[str, Any]] = None
        self.plan: Optional[Dict[str, Any]] = None
        self.cam: Dict[str, Dict[str, Any]] = {}
        self.last_obs: Optional[Dict[str, Any]] = None
        self.try_harder = False

    def _tip_drop_m(self):
        if self.is_sim:
            return float(getattr(self.io, "tip_drop_m", self.CK.TIP_BEYOND_GRASP_M))
        return float(self.RT.so101_tip_drop_m())

    def _tool_model(self):
        if not hasattr(self, "_tool_cache"):
            try:
                import NERO_V4613_Tool_Collision as TC
                self._tool_cache = TC.make_tool_model()
            except Exception:
                self._tool_cache = None
        return self._tool_cache if (self._tool_cache is not None and self._tool_cache.ok) else None

    def _init_fence_checker(self):
        if not hasattr(self, "_fence_cache"):
            try:
                import NERO_V315_Consistency_Safety as CS
                U = CS.load_sim_urdf()
                ws = CS.load_workspace()
                f_cfg = dict((ws or {}).get("fence") or {})
                f_cfg["margin"] = max(float(f_cfg.get("margin", 0.04)), 0.04)
                f_cfg["link_radius"] = max(float(f_cfg.get("link_radius", 0.03)), 0.03)
                if self.is_sim:
                    T_tip = np.array(self.CK.T_L7_TCP, float).copy()
                    S = np.eye(4)
                    S[2, 3] = self._tip_drop_m()
                    so101_T = (T_tip @ S).tolist()
                else:
                    T_l7g = self.RT.real_mount(getattr(self.io, "cfg", None))[0]
                    so101_T = self.RT.so101_tip_tool_T(T_l7g).tolist()
                self._fence_cache = (CS, U, f_cfg, so101_T)
            except Exception:
                self._fence_cache = None
        return self._fence_cache

    def _check_path_fence(self, q_waypoints, zt: float) -> Tuple[bool, str]:
        tm = self._tool_model()
        if tm is not None:
            Ti = np.linalg.inv(self.CK.T_L7_TCP)
            for idx in range(len(q_waypoints) - 1):
                qa = np.asarray(q_waypoints[idx], float)
                qb = np.asarray(q_waypoints[idx + 1], float)
                for k in range(5):
                    q = qa + (qb - qa) * (k / 4.0)
                    ok_t, dz, part = tm.clear_of_table(self._fast.fk(q) @ Ti, zt)
                    if not ok_t:
                        return False, f"tool collision ({part}: {dz:.1f} mm above table)"
        cache = self._init_fence_checker()
        if cache is not None:
            CS, U, f_cfg0, so101_T = cache
            f_cfg = dict(f_cfg0, tool_z_min_override=float(zt) + self.RT.TIP_PLAN_CLEAR_M)
            pts = []
            for idx in range(len(q_waypoints) - 1):
                qa = np.asarray(q_waypoints[idx], float)
                qb = np.asarray(q_waypoints[idx + 1], float)
                n_seg = 12 if idx == 0 else 3
                for k in range(n_seg + (1 if idx == len(q_waypoints) - 2 else 0)):
                    pts.append(list(qa + (qb - qa) * (k / float(n_seg))))
            viol = CS.path_fence_check(U, self._fast.joints, self._fast.joints, pts, "base_link", f_cfg, so101_T)
            if viol:
                return False, "workspace polygon fence or fingertip z < z_table + 5 mm"
        return True, ""

    def _solve_chain(self, waypoints, q_seed, deadline: float):
        """FastArm quick seeds -> cuRobo GPU IK -> 45-seed FastArm fallback, bounded by deadline."""
        # 1) Fast 0.2ms local IK check with primary seeds
        p0 = waypoints[0][1]
        a1 = math.atan2(float(p0[1]), float(p0[0]))
        a1 = a1 + math.pi if a1 < 0 else a1 - math.pi
        quick_seeds = [
            list(q_seed),
            [a1, 0.2, 0.0, 0.8, 0.0, 0.4, 0.0],
            [a1, 0.55, 0.0, 1.4, 0.0, 0.4, 0.9],
        ]

        def _chain_from(sd):
            r0 = self._fast.ik([sd], waypoints[0][1], waypoints[0][2], iters=30)
            if not r0 or r0[0] is None:
                return None, None
            qs = [list(r0[0])]
            dense = [list(r0[0])]
            qp = qs[0]
            p_prev, R_prev = np.asarray(waypoints[0][1], float), waypoints[0][2]
            for _nm, p, R in waypoints[1:]:
                p_cur = np.asarray(p, float)
                for k in range(1, 5):
                    u = k / 4.0
                    pk = p_prev + u * (p_cur - p_prev)
                    rk = self._fast.ik([qp], pk, R, iters=25)
                    if not rk or rk[0] is None:
                        return None, None
                    qp = list(rk[0])
                    dense.append(qp)
                qs.append(qp)
                p_prev, R_prev = p_cur, R
            chk = self.CK.check_joint_path(dense, jump_deg=25.0)
            return (qs, chk) if chk.get("ok") else (None, None)

        for sd in quick_seeds:
            qs, chk = _chain_from(sd)
            if qs and float(chk.get("min_margin_deg", 0.0)) >= 6.0:
                return qs, dict(chk, ik_solver="fast_ik_quick")

        if time.time() > deadline:
            return None, None

        # 2) cuRobo GPU IK (not for the V4.9.1 reachability maps)
        if getattr(self, "_probing", False):
            return None, None
        try:
            import NERO_V487_CuRobo_IK as C487
            qs_cu, chk_cu = C487.solve_chain(self._fast, self.CK.check_joint_path, waypoints, q_seed, max_j1_deg=120.0)
            if qs_cu and chk_cu:
                return qs_cu, chk_cu
        except Exception:
            pass

        if time.time() > deadline:
            return None, None

        # 3) 45-seed FastArm fallback
        for sd in [
            [a1, 0.2, 0.0, 0.8, 0.0, -0.3, 0.0],
            [a1, 0.55, 0.0, 1.4, 0.0, -0.3, -0.9],
            [a1, 0.4, 1.2, 1.6, -1.2, -0.3, 0.0],
            [a1, 0.4, -1.2, 1.6, 1.2, -0.3, 0.0],
        ]:
            if time.time() > deadline:
                break
            qs, chk = _chain_from(sd)
            if qs:
                return qs, dict(chk, ik_solver="fast_ik_45seeds")
        return None, None

    def _sing_sigma(self, q) -> float:
        """V4.9.3 SINGULARITY: smallest singular value of the arm Jacobian (position rows / 0.3 m, rotation rows in
        rad) at q; NERO: 0 at J4 = 0 (elbow) and J6 = +-90 deg (wrist, J5 // J7); normal top-down poses >= 0.24"""
        q = np.asarray(q, float)
        T0 = self._fast.fk(q)
        J = np.zeros((6, 7))
        e = 1e-6
        for i in range(7):
            dq = q.copy()
            dq[i] += e
            T = self._fast.fk(dq)
            J[:3, i] = (T[:3, 3] - T0[:3, 3]) / e / 0.3
            dR = T[:3, :3] @ T0[:3, :3].T
            J[3:, i] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * e)
        return float(np.linalg.svd(J, compute_uv=False)[-1])

    def _sing_check(self, qs):
        """-> (ok, min sigma, why) over a joint path (every other point)"""
        worst, wq = 1e9, None
        for q in list(qs)[::2] + [qs[-1]]:
            v = self._sing_sigma(q)
            if v < worst:
                worst, wq = v, q
        if worst >= SINGULAR_SIGMA_MIN:
            return True, round(worst, 3), ""
        j4, j6 = math.degrees(float(wq[3])), math.degrees(float(wq[5]))
        near = "elbow J4 %.0f deg" % j4 if abs(j4) < 20.0 else ("wrist J6 %.0f deg" % j6 if abs(abs(j6) - 90.0) < 20.0 else "")
        return False, round(worst, 3), (f"passes near a singular arm posture (Jacobian sigma {worst:.3f} < "
                                        f"{SINGULAR_SIGMA_MIN}{'; ' + near if near else ''})")

    def _dense_line(self, q_start, p_end, R_end, step_m=0.004):
        """V4.9.1 DENSE LINE CHECK: the straight TCP line the IO will execute (IK continuity from q_start, one point
        every 4 mm / 2 deg, like SimGeneralIO.move_line) -> (qs or None, check dict)"""
        q0 = list(q_start)
        T0 = self._fast.fk(np.asarray(q0, float))
        p0, R0 = T0[:3, 3].copy(), T0[:3, :3].copy()
        p1 = np.asarray(p_end, float)
        R1 = np.asarray(R_end, float)
        dist = float(np.linalg.norm(p1 - p0))
        turn = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R1 @ R0.T)) - 1.0) / 2.0))))
        n = int(min(80, max(4, math.ceil(dist / step_m), math.ceil(turn / 2.0))))
        pts = self.SK72._turning_segment("s", p0, R0, p1, R1, n) + [("~end", p1, R1)]
        qs, qp = [q0], q0
        for _nm, pk, Rk in pts:
            r = self._fast.ik([qp], pk, Rk, iters=30)
            if not r or r[0] is None:
                return None, {"ok": False, "why": f"the straight line breaks after {len(qs) - 1} of {len(pts)} points "
                                                  "(no continuous arm posture)"}
            if max(abs(a - b) for a, b in zip(r[0], qp)) > math.radians(8.0):
                return None, {"ok": False, "why": "the arm posture flips on the straight line (IK branch change)"}
            qp = list(r[0])
            qs.append(qp)
        chk = self.CK.check_joint_path(qs, margin_deg=DENSE_LINE_MARGIN_DEG, jump_deg=8.0)
        if chk.get("ok"):
            ok_s, sig, why_s = self._sing_check(qs)
            chk = dict(chk, min_singularity_sigma=sig)
            if not ok_s:
                chk = dict(chk, ok=False, why=why_s)
        return (qs if chk.get("ok") else None), chk

    def _unlatch_if_needed(self, label: str) -> Tuple[bool, str]:
        if hasattr(self.io, "latch_pending") and self.io.latch_pending():
            if hasattr(self.io, "clear_latch"):
                ok_l, why_l = self.io.clear_latch(label)
                return bool(ok_l), str(why_l)
        return True, ""

    def _sync_held_state(self) -> Optional[Dict[str, Any]]:
        """Sync self.held with physical gripper state (if a safety stop or collision dropped the object, clear self.held)."""
        if not self.held or not getattr(self.io, "motion_allowed", True):
            return None
        try:
            st = self.io.gripper_state()
        except Exception:
            return None
        gap = st.get("gap_mm", st.get("opening_mm"))
        still_held = bool(st.get("hold_active")) or (
            not self.is_sim and gap is not None and 4.0 <= float(gap) <= 68.0
        )
        if still_held:
            return None
        dropped_id = self.held
        self.held = None
        self.stage = "OBSERVED"
        info = {
            "dropped_object": dropped_id,
            "gripper_state": st,
            "note": f"{dropped_id} is no longer held by the gripper (dropped during safety stop or contact). Call perceive_objects to inspect current object positions and uprightness.",
        }
        self.io.emit(f"[WARN] [V4.8.10 GENERAL] {dropped_id} is no longer in the gripper (hold_active=False) -> state synced")
        return info

    def _res(self, tool, ok, ver=None, reason=None, nxt=None, **extra):
        v = dict(ver or {})
        if reason:
            v["reason"] = reason
        out = {
            "ok": bool(ok),
            "tool": tool,
            "state_after": {"stage": self.stage, "held_object": self.held},
            "physical_verification": v,
            "recommended_next_actions": list(nxt or []),
        }
        if reason:
            out["reason"] = reason
        out.update(extra)
        if not getattr(self, "_probing", False):
            self.io.emit(f"[{'OK' if ok else 'FAIL'}] [V4.8.9 GENERAL] {tool}: " + (reason or "done")
                         + (f" | {json.dumps(v, ensure_ascii=False, default=str)[:240]}" if v else ""))
        return out

    # ------------------------------------------------------------------ V4.9 chunk tools (facts in, LLM decides)
    # V4.9 GENERAL FACTS: perception = measured geometry (height map, cross sections, side profile, cavity, relations);
    # preview_grasp / plan_* never move the arm; execute_* report what was measured (contacts, forces, object motion)

    def _meas_margin_m(self) -> float:
        """V4.9.2: collision checks count anything closer than this as touching (perception error); 0.5 mm with
        simulator-true geometry, 4 mm with the real-like (noisy) twin perception, 6 mm on the real robot"""
        if getattr(self.io, "perception_noise", None):
            return 0.004
        return 0.0005 if self.is_sim else 0.006

    def _tip_clear_m(self) -> float:
        return float(getattr(self.RT, "TIP_PLAN_CLEAR_M", 0.005)) + 0.0005

    def _fingers(self):
        if not hasattr(self, "_fm"):
            fm = None
            if hasattr(self.io, "finger_model"):
                try:
                    fm = self.io.finger_model()
                except Exception:
                    fm = None
            if fm is None:
                fm = G49.SIM_FINGERS if self.is_sim else G49.real_fingers()
            self._fm = fm
        return self._fm

    def _matches(self, label: str, oid: str, prompts: List[str]) -> bool:
        lab = str(label).lower().replace("_", " ")
        for p in prompts:
            q = str(p).strip().lower().replace("_", " ")
            if not q:
                continue
            if q in ("object", "objects", "all", "anything", "item", "items", "everything", "物体", "所有"):
                return True
            for cand in (q, q[:-1] if q.endswith("s") else q, q[:-2] if q.endswith("es") else q):
                if cand and (cand == lab or cand == lab.split()[-1] or lab in cand or cand == oid.lower()):
                    return True
        return False

    def _cam_entry(self, d: Dict[str, Any]) -> Dict[str, Any]:
        c = np.asarray(d["center_xyz_m"], float)
        top = float(d["top_z_m"])
        bot = float(d.get("bottom_z_m", top - float(d.get("height_mm") or 40.0) / 1000.0))
        return {"rim": np.array([c[0], c[1], top]), "center": c, "base": np.array([c[0], c[1], bot]),
                "top_z_m": top, "bottom_z_m": bot, "label": str(d.get("label") or "object"),
                "in_stack": str(d.get("label") or "object"), "yaw_deg": float(d.get("axis_a_yaw_deg", d.get("yaw_deg")) or 0.0),
                "size_mm": list(d.get("size_mm") or [50.0, 50.0]), "height_mm": float(d.get("height_mm") or 40.0),
                "conf": float(d.get("conf") or 0.99), "trusted": bool(d.get("trusted", True)), "raw": d, "desc": d}

    def perceive_objects(self, prompts=None, use_finetuned=True, **_):
        p_list = _clean_prompts(prompts)
        res = self.io.perceive_general(f"general observe ({', '.join(p_list)})", prompts=p_list,
                                       use_finetuned=bool(use_finetuned))
        self.last_obs = res
        zt = float(res.get("table_z_m") or 0.0)
        descs = []
        for o in res.get("objects") or []:
            d = dict(o)
            d.setdefault("object_id", str(d.get("det_id") or d.get("object_id")))
            d.setdefault("label", str(d.get("kind") or G49.label_of(d["object_id"])))
            if "bottom_z_m" not in d:
                d["bottom_z_m"] = round(float(d["top_z_m"]) - float(d.get("height_mm") or 40.0) / 1000.0, 4)
            descs.append(d)
        hm = res.get("heightmap")
        if hm is None:
            hm = G49.HeightMap.from_boxes(descs, zt)
        self.hm = hm
        prev_desc = dict(getattr(self, "desc", None) or {})
        self.desc = {d["object_id"]: d for d in descs}
        self.cam = {oid: self._cam_entry(d) for oid, d in self.desc.items()}
        rel = res.get("relations")
        if rel is None:
            rel = G49.relations([d for d in descs if "open_cavity" in d or "size_mm" in d])
        seen, other = [], []
        for d in descs:
            if self._matches(d["label"], d["object_id"], p_list):
                seen.append(self._compact_desc(d, prev_desc.get(d["object_id"])))
            else:
                other.append({"object_id": d["object_id"], "center_xyz_m": d["center_xyz_m"], "top_z_m": d["top_z_m"],
                              "size_mm": d.get("size_mm"), "note": "seen by the depth camera, not matched by the prompts"})
        if self.stage == "IDLE":
            self.stage = "OBSERVED"
        spec = self._fingers().llm_spec()
        spec["min_fingertip_above_table_mm"] = round(self._tip_clear_m() * 1000.0, 1)
        spec["collision_margin_mm"] = round(self._meas_margin_m() * 1000.0, 1)
        extra = {"objects": seen, "other_objects": other, "relations": rel}
        # V4.9.2 COMPACT OUTPUTS: the gripper geometry and the field legend once per task, then a pointer
        if not getattr(self, "_spec_sent", False):
            extra.update(gripper=spec, legend=DESC_LEGEND)
            self._spec_sent = True
        else:
            extra["gripper"] = "unchanged (see the first perceive_objects result)"
        return self._res("perceive_objects", True,
                         {"prompts": p_list, "count": len(seen), "table_z_m": round(zt, 4)},
                         nxt=["preview_grasp"] if seen else ["perceive_objects"], **extra)

    def _compact_desc(self, d, prev=None):
        """V4.9.2 COMPACT OUTPUTS: what the LLM gets for one object (internal keys dropped, the legend is sent once);
        an object that has not moved (< 3 mm, < 3 deg, same size) since the last look is a one-line entry"""
        if prev is not None:
            dc = float(np.linalg.norm(np.asarray(d["center_xyz_m"], float) - np.asarray(prev["center_xyz_m"], float)))
            dy = abs(((float(d.get("axis_a_yaw_deg") or 0) - float(prev.get("axis_a_yaw_deg") or 0) + 90.0) % 180.0) - 90.0)
            ds = max(abs(float(a) - float(b)) for a, b in zip(d.get("size_mm") or [0, 0], prev.get("size_mm") or [0, 0]))
            if dc < 0.003 and dy < 3.0 and ds < 4.0:
                return {"object_id": d["object_id"], "label": d.get("label"), "center_xyz_m": d["center_xyz_m"],
                        "top_z_m": d["top_z_m"], "unchanged": "same pose and geometry as the previous look"}
        out = {}
        for k, v in d.items():
            if k.startswith("_") or k in ("det_id", "kind", "size_axes", "yaw_deg", "conf", "trusted", "measurement"):
                continue
            if k == "top_surface_segments_mm":
                v = {kk: vv for kk, vv in v.items() if kk != "format"}
            elif k == "open_cavity" and v:
                v = {kk: vv for kk, vv in v.items() if kk != "note"}
            elif k == "side_profile":
                v = [{"z_above_table_mm": r["z_above_table_mm"],
                      "outer_half_width_mm": {kk: vv for kk, vv in r["outer_half_width_mm"].items() if vv is not None},
                      "inner_half_width_mm": {kk: vv for kk, vv in r["inner_half_width_mm"].items() if vv is not None}}
                     for r in v]
            out[k] = v
        if d.get("measurement"):
            out["measured_by"] = "depth camera (noisy, unseen sides missing)"
        return out

    # -- grasp geometry
    def _grasp_point(self, object_id, xy_offset_mm, grasp_z_offset_from_top_mm):
        obj = self.cam[object_id]
        c = np.asarray(obj["center"], float)
        dx, dy = 0.0, 0.0
        if isinstance(xy_offset_mm, (list, tuple)) and len(xy_offset_mm) >= 2:
            dx, dy = float(xy_offset_mm[0]) / 1000.0, float(xy_offset_mm[1]) / 1000.0
        z = float(obj["top_z_m"]) + float(grasp_z_offset_from_top_mm or 0.0) / 1000.0
        return np.array([c[0] + dx, c[1] + dy, z], float), (dx, dy)

    def preview_grasp(self, object_id="obj_1", xy_offset_mm=None, grasp_z_offset_from_top_mm=-20.0,
                      closing_yaw_deg=0.0, pitch_deg=0.0, open_mm=60.0, **_):
        """V4.9 PREVIEW GRASP: no motion - what the fingers would touch at this grasp"""
        if not self.last_obs or object_id not in self.cam:
            return self._res("preview_grasp", False, {"available": list(self.cam.keys())},
                             reason="OBSERVE_FIRST" if not self.last_obs else "OBJECT_NOT_FOUND", nxt=["perceive_objects"])
        p, _ = self._grasp_point(object_id, xy_offset_mm, grasp_z_offset_from_top_mm)
        om = min(float(open_mm), self._fingers().max_open_mm)
        pv = G49.preview_grasp(self.hm, self._fingers(), p, float(closing_yaw_deg), float(pitch_deg or 0.0), om, object_id,
                               table_clear_m=self._tip_clear_m(), margin_m=self._meas_margin_m())
        pv["grasp_point_above_table_mm"] = round((p[2] - self.hm.table_z) * 1000.0, 1)
        if float(open_mm) > om:
            pv["open_mm_clipped_to"] = om
        return self._res("preview_grasp", True, pv, nxt=["preview_grasp", "plan_grasp_chunk"])

    def plan_grasp_chunk(self, object_id="obj_1", xy_offset_mm=None, grasp_z_offset_from_top_mm=-20.0,
                         closing_yaw_deg=None, yaw_tolerance_deg=0.0, pitch_deg=0.0, open_mm=60.0,
                         pregrasp_height_mm=80.0, lift_height_mm=100.0, yaw_deg=None, **_):
        args = dict(object_id=object_id, xy_offset_mm=xy_offset_mm, grasp_z_offset_from_top_mm=grasp_z_offset_from_top_mm,
                    closing_yaw_deg=closing_yaw_deg, yaw_tolerance_deg=yaw_tolerance_deg, pitch_deg=pitch_deg,
                    open_mm=open_mm, pregrasp_height_mm=pregrasp_height_mm, lift_height_mm=lift_height_mm, yaw_deg=yaw_deg)
        r = self._plan_grasp_core(**args)
        if not r.get("ok") and r.get("reason") == "GRASP_PLAN_FAILED":
            try:
                r.setdefault("physical_verification", {})["reachability"] = self._grasp_reach_map(args)
            except Exception as e:
                r.setdefault("physical_verification", {})["reachability"] = {"error": repr(e)[:120]}
        return r

    def _grasp_reach_map(self, args):
        """V4.9.1 GRASP REACHABILITY: the same grasp turned about the object's vertical axis through center_xyz_m in
        30 deg steps (offset and closing direction turn together; with a zero offset only the closing direction
        turns), each checked like plan_grasp_chunk without cuRobo: plans / collision / no arm posture / joint limit"""
        if args["object_id"] not in self.cam:
            return {}
        t_end = time.time() + 6.0
        dx, dy = 0.0, 0.0
        xy = args.get("xy_offset_mm")
        if isinstance(xy, (list, tuple)) and len(xy) >= 2:
            dx, dy = float(xy[0]), float(xy[1])
        y0 = args.get("closing_yaw_deg")
        y0 = float(y0 if y0 is not None else (float(self.cam[args["object_id"]]["yaw_deg"]) + 90.0))
        rows = []
        self._probing = True
        saved = (self.grasp_plan, self.plan, self.stage)
        try:
            for d in range(0, 360, 30):
                if time.time() > t_end:
                    rows.append({"turn_deg": d, "status": "not checked (time)"})
                    continue
                a = math.radians(d)
                off = [round(dx * math.cos(a) - dy * math.sin(a), 1), round(dx * math.sin(a) + dy * math.cos(a), 1)]
                yaw = ((y0 + d + 180.0) % 360.0) - 180.0
                a_turn = dict(args, xy_offset_mm=off, closing_yaw_deg=yaw, yaw_tolerance_deg=0.0)
                r = self._plan_grasp_core(**a_turn)
                pv = r.get("physical_verification", {})
                if r.get("ok"):
                    st = f"plans (joint margin {pv.get('min_joint_margin_deg')} deg)"
                else:
                    st = re.sub(r"^yaw [+-]?\d+: ", "", "; ".join(pv.get("rejects", [])[:1]))[:100] or str(r.get("reason"))
                    st = st.replace("no arm posture (IK) for pregrasp -> grasp -> lift", "no arm posture (IK)")
                    pre0, lift0 = float(args.get("pregrasp_height_mm") or 80.0), float(args.get("lift_height_mm") or 100.0)
                    if "IK" in st or "line fails" in st or "joint" in st:   # lower pregrasp / lift?
                        for pre, lift in ((min(pre0, 40.0), round(lift0 * 0.6)), (30.0, 40.0)):
                            if time.time() > t_end or (pre >= pre0 and lift >= lift0):
                                continue
                            r2 = self._plan_grasp_core(**dict(a_turn, pregrasp_height_mm=pre, lift_height_mm=lift))
                            if r2.get("ok"):
                                st += f" | plans with pregrasp {pre:.0f} / lift {lift:.0f} mm"
                                break
                rows.append({"turn_deg": d, "xy_offset_mm": off, "closing_yaw_deg": round(yaw, 1), "status": st})
        finally:
            self._probing = False
            self.grasp_plan, self.plan, self.stage = saved
        return {"note": "the requested grasp turned about the object's vertical axis; facts only", "around_object": rows}

    def _plan_grasp_core(self, object_id="obj_1", xy_offset_mm=None, grasp_z_offset_from_top_mm=-20.0,
                         closing_yaw_deg=None, yaw_tolerance_deg=0.0, pitch_deg=0.0, open_mm=60.0,
                         pregrasp_height_mm=80.0, lift_height_mm=100.0, yaw_deg=None, **_):
        if not self.last_obs:
            return self._res("plan_grasp_chunk", False, reason="OBSERVE_FIRST", nxt=["perceive_objects"])
        if object_id not in self.cam:
            return self._res("plan_grasp_chunk", False, {"available": list(self.cam.keys())},
                             reason="OBJECT_NOT_FOUND", nxt=["perceive_objects"])
        if self.held:
            return self._res("plan_grasp_chunk", False, {"held_object": self.held}, reason="ALREADY_HOLDING",
                             nxt=["plan_place_chunk"])
        if closing_yaw_deg is None:
            closing_yaw_deg = yaw_deg if yaw_deg is not None else float(self.cam[object_id]["yaw_deg"]) + 90.0
        fm = self._fingers()
        om = min(float(open_mm), fm.max_open_mm)
        p_a, (dx_m, dy_m) = self._grasp_point(object_id, xy_offset_mm, grasp_z_offset_from_top_mm)
        zt = float(self.hm.table_z)
        tol = max(0.0, min(90.0, float(yaw_tolerance_deg or 0.0)))
        yaw0 = float(closing_yaw_deg)
        offs = [0.0]
        t = 10.0
        while t <= tol + 1e-6:
            offs += [t, -t]
            t += 10.0
        yaws = []
        for o in offs:
            for flip in (0.0, 180.0):                    # the two pads are alike: yaw and yaw + 180 are the same grasp
                yaws.append(((yaw0 + o + flip + 180.0) % 360.0) - 180.0)
        q_seed = self.io.current_q()
        deadline = time.time() + PLAN_BUDGET_S * (1.5 if tol > 0 else 1.0)
        tm = self._tool_model()
        Ti = np.linalg.inv(self.CK.T_L7_TCP)
        best, rej, geo_first = None, [], None
        for gyaw in yaws:
            if time.time() > deadline:
                rej.append("planning time budget used up")
                break
            pv = G49.preview_grasp(self.hm, fm, p_a, gyaw, float(pitch_deg or 0.0), om, object_id,
                                   margin_m=self._meas_margin_m())
            if geo_first is None:
                geo_first = pv
            if pv["collisions_at_grasp_pose"]:
                c0 = pv["collisions_at_grasp_pose"][0]
                rej.append(f"yaw {gyaw:+.0f}: {c0['part']} inside {c0['object']} by {c0['depth_mm']} mm")
                continue
            if pv["lowest_gripper_point_above_table_mm"] < self._tip_clear_m() * 1000.0:
                rej.append(f"yaw {gyaw:+.0f}: fingertip {pv['lowest_gripper_point_above_table_mm']} mm above the table "
                           f"(the safety limit is {self._tip_clear_m() * 1000.0:.0f} mm)")
                continue
            R_gr = G49.grasp_R(gyaw, float(pitch_deg or 0.0))
            p_g, R_g = self.prof.tcp_from_grasp(p_a, R_gr)
            if tm is not None:
                T_g = np.eye(4)
                T_g[:3, :3], T_g[:3, 3] = R_g, p_g
                ok_tm, dz_tm, part_tm = tm.clear_of_table(T_g @ Ti, zt)
                if not ok_tm:
                    rej.append(f"yaw {gyaw:+.0f}: tool {part_tm} {dz_tm:.0f} mm above the table")
                    continue
            p_pre = p_g + np.array([0.0, 0.0, float(pregrasp_height_mm) / 1000.0])
            p_lift = p_g + np.array([0.0, 0.0, float(lift_height_mm) / 1000.0])
            qs, chk = self._solve_chain([("pregrasp", p_pre, R_g), ("grasp", p_g, R_g), ("lift", p_lift, R_g)],
                                        q_seed, deadline)
            if not qs or not chk:
                rej.append(f"yaw {gyaw:+.0f}: no arm posture (IK) for pregrasp -> grasp -> lift")
                continue
            ok_f, why_f = self._check_path_fence([list(q_seed)] + qs, zt)
            if not ok_f:
                rej.append(f"yaw {gyaw:+.0f}: {why_f}")
                continue
            m = float(chk.get("min_margin_deg") or 0.0)
            if m < MIN_PLAN_MARGIN_DEG:
                rej.append(f"yaw {gyaw:+.0f}: joint {chk.get('binding')} only {m:.1f} deg from its limit")
                continue
            qd, chd = self._dense_line(qs[0], p_g, R_g)                  # V4.9.1: descent, as executed
            ql, chl = (self._dense_line(qd[-1], p_lift, R_g) if qd else (None, {}))   # and the lift
            if not qd or not ql:
                bad = chd if not qd else chl
                rej.append(f"yaw {gyaw:+.0f}: the {'descent' if not qd else 'lift'} line fails: {bad.get('why')}")
                continue
            qs = [qs[0], qd[-1], ql[-1]]
            m = min(m, float(chd.get("min_margin_deg") or 99.0), float(chl.get("min_margin_deg") or 99.0))
            chk = dict(chk, min_margin_deg=round(m, 1))
            cand = {"yaw": gyaw, "pv": pv, "qs": qs, "chk": chk, "p_g": p_g, "R_g": R_g, "p_pre": p_pre, "p_lift": p_lift}
            if best is None or m > float(best["chk"].get("min_margin_deg") or 0.0):
                best = cand
            if m >= GOOD_PLAN_MARGIN_DEG:
                break
        if best is None:
            return self._res("plan_grasp_chunk", False,
                             {"object_id": object_id, "grasp_point_xyz_m": [round(float(v), 4) for v in p_a],
                              "rejects": rej[:8], "geometry_at_requested_yaw": geo_first},
                             reason="GRASP_PLAN_FAILED", nxt=["preview_grasp", "plan_grasp_chunk", "perceive_objects"])
        qs, chk = best["qs"], best["chk"]
        chosen = {
            "object_id": object_id,
            "xy_offset_mm": [round(dx_m * 1000.0, 1), round(dy_m * 1000.0, 1)],
            "grasp_target_xyz_m": [round(float(v), 4) for v in p_a],
            "grasp_yaw_deg": round(best["yaw"], 1),
            "pitch_deg": round(float(pitch_deg or 0.0), 1),
            "open_mm": om,
            "grasp_z_mm": round((p_a[2] - zt) * 1000.0, 1),
            "table_z_m": round(zt, 4),
            "min_joint_margin_deg": chk.get("min_margin_deg"),
            "ik_solver": chk.get("ik_solver", "fast_ik"),
            "lift_height_mm": float(lift_height_mm),
            "_wps": {"pregrasp": (best["p_pre"], best["R_g"]), "grasp": (best["p_g"], best["R_g"]),
                     "lift": (best["p_lift"], best["R_g"])},
            "_q": {"pregrasp": qs[0], "grasp": qs[1], "lift": qs[2]},
            "_preview": best["pv"],
        }
        if getattr(self, "_probing", False):
            return self._res("plan_grasp_chunk", True, {"min_joint_margin_deg": chosen["min_joint_margin_deg"]})
        self.grasp_plan = chosen
        self.plan = chosen
        self.stage = "GRASP_PLANNED"
        out = {k: v for k, v in chosen.items() if not k.startswith("_")}
        out["finger_preview"] = best["pv"]
        if rej:
            out["other_yaws_rejected"] = rej[:6]
        return self._res("plan_grasp_chunk", True, out, nxt=["execute_grasp_chunk"])

    def _obj_xyz(self, oid):
        f = getattr(self.io, "object_position", None)
        if f is None:
            return None
        try:
            p = f(oid)
            return None if p is None else np.asarray(p, float)
        except Exception:
            return None

    def _held_now(self, st) -> bool:
        if self.is_sim:
            return bool(st.get("hold_active")) or (bool(st.get("contact_left")) and bool(st.get("contact_right")))
        if "held" in st:
            return bool(st.get("held"))
        gap = st.get("gap_mm", st.get("opening_mm"))
        return gap is not None and 4.0 <= float(gap) <= 68.0

    def _trips(self) -> Optional[int]:
        try:
            return int(self.io.safety_trips()) if hasattr(self.io, "safety_trips") else None
        except Exception:
            return None

    def _safety_hit(self, r, n0) -> bool:
        """V4.9.3: did this motion end in a safety stop / a large tracking mismatch?"""
        n1 = self._trips()
        txt = str(r.get("error") or r.get("why") or "")
        return bool(r.get("safety")) or (n0 is not None and n1 is not None and n1 > n0) or \
            any(k in txt for k in ("TRACKING MISMATCH", "RUNAWAY", "SAFETY"))

    def _safety_stop(self, tool, step, r, n0):
        """V4.9.3 SAFETY STOP = STOP: no fallback motion; report what happened and the state now"""
        facts = {"step": step, "motion_result": {k: r.get(k) for k in ("error", "why", "trip_reason",
                                                                       "closest_joint_limit", "tcp_error_mm") if r.get(k)}}
        n1 = self._trips()
        if n0 is not None and n1 is not None:
            facts["safety_stops"] = n1 - n0
        held = None
        try:
            st = dict(self.io.gripper_state())
            facts["gripper_now"] = st
            held = self._held_now(st)
        except Exception:
            pass
        if self.held or tool == "execute_place_chunk":
            facts["object_still_held"] = held
            if held is False:
                facts["note_hold"] = ("the object is no longer held (the safety recovery released it); "
                                      "look before doing anything else")
                self.held = None
                self.stage = "OBSERVED"
        facts["note"] = ("the safety guard stopped the arm; no further motion was attempted by this tool. The "
                         "simulator may need STOP+RECOVER (the next tool does it) or a restart if motions keep failing")
        self.sim_suspect = True
        self.io.emit(f"[WARN] [V4.9.3 GENERAL] {tool}: safety stop at {step} -> stopped (no fallback motion)")
        return self._res(tool, False, facts, reason="SAFETY_STOP", nxt=["perceive_objects", "verify_and_park_chunk"])

    def execute_grasp_chunk(self, clamp_force_n=8.0, open_mm=None, **_):
        if not self.grasp_plan:
            return self._res("execute_grasp_chunk", False, reason="GRASP_NOT_PLANNED", nxt=["plan_grasp_chunk"])
        oid = self.grasp_plan["object_id"]
        W, Q = self.grasp_plan["_wps"], self.grasp_plan["_q"]
        om = float(self.grasp_plan.get("open_mm") or (open_mm if open_mm is not None else 60.0))
        if not getattr(self.io, "motion_allowed", True):
            return self._res("execute_grasp_chunk", True, {"dry_run": True, "object_id": oid}, nxt=["plan_place_chunk"])
        self._unlatch_if_needed(f"V4.9 pregrasp {oid}")
        n0 = self._trips()
        p_before = self._obj_xyz(oid)
        self.io.gripper_open(om)
        r = self.io.move_joints(f"V4.9 pregrasp {oid}", Q["pregrasp"])
        if not r.get("ok"):
            if self._safety_hit(r, n0):
                return self._safety_stop("execute_grasp_chunk", "pregrasp", r, n0)
            return self._res("execute_grasp_chunk", False, r, reason="PREGRASP_FAILED", nxt=["plan_grasp_chunk"])
        p_g, R_g = W["grasp"]
        p_mid = np.asarray(p_g, float) + np.array([0.0, 0.0, 0.015])
        r = self.io.move_line(f"V4.9 descend {oid}", p_mid, self.CK.R_to_quat_xyzw(R_g), 0.03)
        if r.get("ok"):
            r = self.io.move_line(f"V4.9 grasp {oid}", p_g, self.CK.R_to_quat_xyzw(R_g), 0.012, strict=True)
        if not r.get("ok"):
            if self._safety_hit(r, n0):
                return self._safety_stop("execute_grasp_chunk", "descent", r, n0)
            self.io.move_line(f"V4.9 retreat {oid}", W["pregrasp"][0], self.CK.R_to_quat_xyzw(W["pregrasp"][1]), 0.035)
            return self._res("execute_grasp_chunk", False, r, reason="DESCENT_FAILED", nxt=["plan_grasp_chunk"])
        p_desc = self._obj_xyz(oid)
        clamp = dict(self.io.gripper_clamp(float(clamp_force_n)))
        p_clamp = self._obj_xyz(oid)
        facts = {"object_id": oid, "clamp": clamp}
        if p_before is not None and p_desc is not None:
            facts["object_moved_during_descent_mm"] = round(float(np.linalg.norm(p_desc - p_before)) * 1000.0, 1)
        if p_desc is not None and p_clamp is not None:
            dv = p_clamp - p_desc
            facts["object_moved_during_clamp_mm"] = {"xy": round(float(np.linalg.norm(dv[:2])) * 1000.0, 1),
                                                     "z": round(float(dv[2]) * 1000.0, 1)}
        self.io.emit(f"[INFO] [V4.9 GENERAL] clamp {oid}: {json.dumps(clamp, ensure_ascii=False, default=str)[:260]}")
        if not self._held_now(clamp):
            self.io.gripper_open(om)
            self.io.move_line(f"V4.9 abort retreat {oid}", W["pregrasp"][0], self.CK.R_to_quat_xyzw(W["pregrasp"][1]), 0.035)
            facts["planned_finger_preview"] = self.grasp_plan.get("_preview")
            facts["note"] = ("the clamp ended without a two-pad hold (contact flags / hold above); the object was not "
                             "lifted")
            self.stage = "OBSERVED"
            return self._res("execute_grasp_chunk", False, facts, reason="OBJECT_NOT_HELD",
                             nxt=["perceive_objects", "preview_grasp"])
        r = self.io.move_line(f"V4.9 lift {oid}", W["lift"][0], self.CK.R_to_quat_xyzw(W["lift"][1]), 0.04)
        if not r.get("ok") and self._safety_hit(r, n0):
            self.held = oid if not self._held_lost() else None
            return self._safety_stop("execute_grasp_chunk", "lift", r, n0)
        if not r.get("ok"):
            dropped = self._held_lost()
            facts.update({"lift": r})
            if not dropped:
                self.held = oid
                self.stage = "HOLDING"
                self._remember_held(oid)
                facts["note"] = "lift refused / stopped but the object is still in the gripper"
            return self._res("execute_grasp_chunk", False, facts, reason="LIFT_FAILED",
                             nxt=["verify_and_park_chunk", "plan_place_chunk"])
        st = dict(self.io.gripper_state())
        p_lift = self._obj_xyz(oid)
        facts["gripper_after_lift"] = st
        if p_clamp is not None and p_lift is not None:
            facts["object_rose_mm"] = round(float(p_lift[2] - p_clamp[2]) * 1000.0, 1)
        rose_ok = facts.get("object_rose_mm") is None or facts["object_rose_mm"] >= 0.5 * float(self.grasp_plan["lift_height_mm"])
        if not self._held_now(st) or not rose_ok:
            self.io.gripper_open(om)
            self.stage = "OBSERVED"
            return self._res("execute_grasp_chunk", False, facts, reason="OBJECT_NOT_LIFTED",
                             nxt=["perceive_objects", "preview_grasp"])
        self.held = oid
        self.held_obj = dict(self.cam.get(oid) or {})
        self.picked_from = np.asarray(self.held_obj.get("center"), float)[:2] if self.held_obj.get("center") is not None else None
        self.stage = "HOLDING"
        self._remember_held(oid)
        return self._res("execute_grasp_chunk", True, facts, nxt=["plan_place_chunk"])

    def _held_lost(self) -> bool:
        try:
            st = self.io.gripper_state()
        except Exception:
            return False
        return not self._held_now(st)

    def _grasp_T_now(self):
        if hasattr(self.io, "tcp_pose"):
            p, q = self.io.tcp_pose()
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = self.CK.quat_xyzw_to_R(q), np.asarray(p, float)
        else:
            T = self._fast.fk(np.asarray(self.io.current_q(), float))
        return T @ self.prof.T_tcp_grasp

    def _planned_lift_T(self):
        """grasp frame at the planned lift pose (what the arm was commanded to)"""
        try:
            p, R = self.grasp_plan["_wps"]["lift"]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = np.asarray(R, float), np.asarray(p, float)
            return T @ self.prof.T_tcp_grasp
        except Exception:
            return self._grasp_T_now()

    def _remember_held(self, oid, measure=True):
        """held object as a point cloud in the grasp frame (re-measured in the gripper when the IO can look)"""
        d = None
        f = getattr(self.io, "describe_object", None) if measure else None
        if f is not None:
            try:
                d = f(oid)
            except Exception:
                d = None
        if d is None:
            d0 = (self.desc or {}).get(oid)
            if d0 is None:
                self._held_cloud_G = None
                return
            d = json.loads(json.dumps(d0, default=float))
            lift = float(self.grasp_plan.get("lift_height_mm") or 0.0) / 1000.0
            for key in ("center_xyz_m",):
                d[key] = [d[key][0], d[key][1], d[key][2] + lift]
            d["top_z_m"] += lift
            d["bottom_z_m"] += lift
            for r in d.get("side_profile") or []:
                r["z_m"] = r["z_m"] + lift
        P = G49.held_cloud_from_desc(d)
        T = self._planned_lift_T()
        self._held_T = T
        self._held_cloud_G = (T[:3, :3].T @ (P - T[:3, 3]).T).T
        self._held_desc = d
        self._held_bottom_G = float(self._held_cloud_G[:, 2].max())      # z_G points down: the lowest point

    def _gripper_points_G(self, open_mm):
        out = []
        for name, lo, hi in self._fingers().boxes(open_mm):
            out.append(G49.box_points(lo, hi, 0.004 if name.startswith("pad") else 0.008))
        return np.vstack(out)

    def plan_place_chunk(self, target_id_or_xyz="obj_2", xy_offset_mm=None, held_bottom_above_target_top_mm=None,
                         yaw_change_deg=0.0, approach_height_mm=60.0, release_open_mm=None,
                         carry_lift_mm=0.0, release_above_target_mm=None, **_):
        args = dict(target_id_or_xyz=target_id_or_xyz, xy_offset_mm=xy_offset_mm,
                    held_bottom_above_target_top_mm=held_bottom_above_target_top_mm, yaw_change_deg=yaw_change_deg,
                    approach_height_mm=approach_height_mm, release_open_mm=release_open_mm, carry_lift_mm=carry_lift_mm,
                    release_above_target_mm=release_above_target_mm)
        r = self._plan_place_core(**args)
        if not r.get("ok") and r.get("reason") in ("PLACE_PLAN_FAILED", "CARRY_COLLISION", "PLACE_COLLISION"):
            try:
                r.setdefault("physical_verification", {})["reachability"] = self._place_reach_map(args)
            except Exception as e:                       # the map is extra information only
                r.setdefault("physical_verification", {})["reachability"] = {"error": repr(e)[:120]}
        return r

    def _place_reach_map(self, args):
        """V4.9.1 PLACE REACHABILITY: the same place with other yaw_change_deg / carry_lift_mm values (no motion, no
        cuRobo): which ones would plan, and why the others would not"""
        t_end = time.time() + 6.0
        out = {"note": "the same place request re-planned with one parameter changed; facts only"}
        self._probing = True
        saved = (self.place_plan, self.stage, self.plan)
        try:
            def status(a):
                if time.time() > t_end:
                    return "not checked (time)"
                r = self._plan_place_core(**a)
                if r.get("ok"):
                    return f"plans (joint margin {r['physical_verification'].get('min_joint_margin_deg')} deg)"
                pv = r.get("physical_verification", {})
                return f"{r.get('reason')}: {'; '.join(pv.get('problems', []))[:110]}"
            y0 = float(args.get("yaw_change_deg") or 0.0)
            out["yaw_change_deg"] = {f"{((y0 + d + 180.0) % 360.0) - 180.0:+.0f}":
                                     status(dict(args, yaw_change_deg=((y0 + d + 180.0) % 360.0) - 180.0))
                                     for d in range(0, 360, 30)}
            out["carry_lift_mm"] = {f"{cl:.0f}": status(dict(args, carry_lift_mm=cl))
                                    for cl in (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)}
            h0 = args.get("held_bottom_above_target_top_mm")
            h0 = float(h0 if h0 is not None else (args.get("release_above_target_mm") or 20.0))
            out["held_bottom_above_target_top_mm"] = {f"{h:+.0f}": status(dict(args, held_bottom_above_target_top_mm=h))
                                                      for h in sorted({round(h0 + d) for d in (-40, -20, 0, 20, 40, 60)} | {0, 10, 20})}
        finally:
            self._probing = False
            self.place_plan, self.stage, self.plan = saved
        return out

    def _plan_place_core(self, target_id_or_xyz="obj_2", xy_offset_mm=None, held_bottom_above_target_top_mm=None,
                         yaw_change_deg=0.0, approach_height_mm=60.0, release_open_mm=None,
                         carry_lift_mm=0.0, release_above_target_mm=None, **_):
        if not self.last_obs:
            return self._res("plan_place_chunk", False, reason="OBSERVE_FIRST", nxt=["perceive_objects"])
        virtual = False
        if not self.held:
            # plan / hover modes: plan the place for the object of the planned grasp, as if it had been lifted
            if self.grasp_plan and (not getattr(self.io, "motion_allowed", True) or self.stage == "GRASP_PLANNED"):
                virtual = True
                self._remember_held(self.grasp_plan["object_id"], measure=False)
            else:
                return self._res("plan_place_chunk", False, reason="NOT_HOLDING", nxt=["plan_grasp_chunk"])
        held_id = self.held or self.grasp_plan["object_id"]
        if held_bottom_above_target_top_mm is None:
            held_bottom_above_target_top_mm = release_above_target_mm if release_above_target_mm is not None else 20.0
        xyz_direct = _parse_target_xyz(target_id_or_xyz)
        tgt_name = None
        if xyz_direct is not None:
            p_tgt = xyz_direct
        elif target_id_or_xyz in self.cam:
            t_obj = self.cam[target_id_or_xyz]
            tgt_name = target_id_or_xyz
            p_tgt = np.array([float(t_obj["center"][0]), float(t_obj["center"][1]), float(t_obj["top_z_m"])], float)
        else:
            return self._res("plan_place_chunk", False, {"available": list(self.cam.keys())},
                             reason="TARGET_NOT_FOUND", nxt=["perceive_objects"])
        dx_m, dy_m = 0.0, 0.0
        if isinstance(xy_offset_mm, (list, tuple)) and len(xy_offset_mm) >= 2:
            dx_m, dy_m = float(xy_offset_mm[0]) / 1000.0, float(xy_offset_mm[1]) / 1000.0
        Tg0 = getattr(self, "_held_T", None)
        if Tg0 is None:
            Tg0 = self._grasp_T_now()
        C = getattr(self, "_held_cloud_G", None)
        if C is None:
            C = np.zeros((1, 3))
        a = math.radians(float(yaw_change_deg or 0.0))
        Rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])
        R_rel = Rz @ Tg0[:3, :3]
        Cw0 = (R_rel @ C.T).T                         # held cloud relative to the grasp point, release orientation
        cxy = Cw0[:, :2].mean(axis=0) if len(Cw0) > 1 else np.zeros(2)
        bottom_rel = float(Cw0[:, 2].min()) if len(Cw0) > 1 else -self._tip_drop_m()
        z_bottom = float(p_tgt[2]) + float(held_bottom_above_target_top_mm) / 1000.0
        p_rel_g = np.array([p_tgt[0] + dx_m - cxy[0], p_tgt[1] + dy_m - cxy[1], z_bottom - bottom_rel])
        zt = float(self.hm.table_z)
        hm_wo = self.hm.without([held_id])
        g_open = float(release_open_mm if release_open_mm is not None else self._fingers().max_open_mm)
        gap_now = None
        try:
            gap_now = self.io.gripper_state().get("gap_mm")
        except Exception:
            pass
        g_closed = float(gap_now if gap_now is not None else 20.0)
        held_w = Cw0 + p_rel_g
        fing_closed = (R_rel @ self._gripper_points_G(g_closed).T).T + p_rel_g
        fing_open = (R_rel @ self._gripper_points_G(g_open).T).T + p_rel_g
        facts = {
            "target": str(target_id_or_xyz),
            "held_object": held_id,
            "held_object_bottom_z_m": round(z_bottom, 4),
            "held_object_bottom_above_target_top_mm": float(held_bottom_above_target_top_mm),
            "grasp_point_at_release_xyz_m": [round(float(v), 4) for v in p_rel_g],
            "held_object_clearance": G49.cloud_clearance(hm_wo, held_w, margin_m=self._meas_margin_m()),
            "fingers_clearance_closed": G49.cloud_clearance(hm_wo, fing_closed, margin_m=self._meas_margin_m()),
            "fingers_clearance_when_opened_to_release_mm": G49.cloud_clearance(hm_wo, fing_open, margin_m=self._meas_margin_m()),
            "release_open_mm": g_open,
        }
        if tgt_name and (self.desc or {}).get(tgt_name):
            td = self.desc[tgt_name]
            facts["target_geometry"] = {"top_z_m": td["top_z_m"], "open_cavity": td.get("open_cavity"),
                                        "size_mm": td.get("size_mm")}
        bad = []
        for key in ("held_object_clearance", "fingers_clearance_closed", "fingers_clearance_when_opened_to_release_mm"):
            c = facts[key]
            if c.get("penetrates"):
                bad.append(f"{key}: inside {', '.join(c['penetrates'])} (min clearance "
                           f"{c['min_clearance_above_other_objects_mm']} mm)")
            if c["min_clearance_above_table_mm"] < 2.0:
                bad.append(f"{key}: {c['min_clearance_above_table_mm']} mm above the table")
        if bad:
            facts["problems"] = bad
            return self._res("plan_place_chunk", False, facts, reason="PLACE_COLLISION",
                             nxt=["plan_place_chunk", "perceive_objects"])
        p_rel, R_tcp = self.prof.tcp_from_grasp(p_rel_g, R_rel)
        p_above = p_rel + np.array([0.0, 0.0, float(approach_height_mm) / 1000.0])
        q_now = list(self.grasp_plan["_q"]["lift"]) if virtual else self.io.current_q()
        deadline = time.time() + PLAN_BUDGET_S
        via = None
        if float(carry_lift_mm or 0.0) > 0.5:          # V4.9 CARRY LIFT: straight up first, then carry
            T_now = self._fast.fk(np.asarray(q_now, float))
            p_via = T_now[:3, 3] + np.array([0.0, 0.0, float(carry_lift_mm) / 1000.0])
            qv, chv = self._dense_line(q_now, p_via, T_now[:3, :3])       # V4.9.1: as executed, point by point
            if not qv:
                facts["problems"] = [f"the {float(carry_lift_mm):.0f} mm straight-up lift fails: {chv.get('why')}"]
                return self._res("plan_place_chunk", False, facts, reason="PLACE_PLAN_FAILED", nxt=["plan_place_chunk"])
            via = (p_via, T_now[:3, :3], qv[-1])
            q_now = list(qv[-1])
        qs, chk = self._solve_chain([("above_target", p_above, R_tcp), ("release", p_rel, R_tcp)], q_now, deadline)
        if not qs or not chk:
            facts["problems"] = ["no arm posture (IK) for above-target -> release with this orientation"]
            return self._res("plan_place_chunk", False, facts, reason="PLACE_PLAN_FAILED",
                             nxt=["plan_place_chunk"])
        ok_f, why_f = self._check_path_fence([list(q_now)] + qs, zt)
        if not ok_f or float(chk.get("min_margin_deg") or 0.0) < MIN_PLACE_MARGIN_DEG:
            facts["problems"] = [why_f or f"joint {chk.get('binding')} only {chk.get('min_margin_deg')} deg from its limit"]
            return self._res("plan_place_chunk", False, facts, reason="PLACE_PLAN_FAILED", nxt=["plan_place_chunk"])
        ql, chl = self._dense_line(qs[0], p_rel, R_tcp)                   # V4.9.1: the vertical lowering line
        if not ql:
            facts["problems"] = [f"the lowering line above-target -> release fails: {chl.get('why')}"]
            return self._res("plan_place_chunk", False, facts, reason="PLACE_PLAN_FAILED", nxt=["plan_place_chunk"])
        qs = [qs[0], ql[-1]]
        # carry: V4.9 prefers a straight line at the lift orientation (no turn about the vertical, a held object
        # does not swing); else the joint-space move (slowed).  Held object + closed fingers vs the scene for both.
        Cg = getattr(self, "_held_cloud_G", None)
        Fg = self._gripper_points_G(g_closed)
        pts_G = Fg if Cg is None else np.vstack([Cg, Fg])

        def _clear_along(Ts):
            worst_, at_ = None, None
            for T in Ts:
                P = (T[:3, :3] @ pts_G.T).T + T[:3, 3]
                cc = G49.cloud_clearance(hm_wo, P, margin_m=self._meas_margin_m())
                v = cc.get("min_clearance_above_other_objects_mm")
                v = 1e9 if v is None else v
                v = min(v, cc["min_clearance_above_table_mm"])
                if worst_ is None or v < worst_:
                    worst_, at_ = v, cc.get("closest_object_below")
            return worst_, at_

        T_start = self._fast.fk(np.asarray(q_now, float))
        R_start = T_start[:3, :3]
        line_ok = False
        turn = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R_tcp @ R_start.T)) - 1.0) / 2.0))))
        if turn < 2.0:
            Ts_line = []
            for k in range(1, 21):
                T = np.eye(4)
                T[:3, :3] = R_start
                T[:3, 3] = T_start[:3, 3] + (p_above - T_start[:3, 3]) * (k / 20.0)
                Ts_line.append(T @ self.prof.T_tcp_grasp)
            w_line, at_line = _clear_along(Ts_line)
            line_ok = w_line is not None and w_line >= 5.0
        qa, qb = np.asarray(q_now, float), np.asarray(qs[0], float)
        w_joint, at_joint = _clear_along([self._fast.fk(qa + (qb - qa) * (k / 20.0)) @ self.prof.T_tcp_grasp
                                          for k in range(1, 21)])
        if line_ok:                                     # V4.9.3: the carry line too, as it will be executed
            qcl, chcl = self._dense_line(q_now, p_above, R_tcp)
            if not qcl:
                line_ok = False
                facts["carry_line_rejected"] = chcl.get("why")
        if not line_ok:
            qa_, qb_ = np.asarray(q_now, float), np.asarray(qs[0], float)
            ok_s, sig_s, why_s = self._sing_check([qa_ + (qb_ - qa_) * (k / 20.0) for k in range(21)])
            if not ok_s:
                facts["problems"] = [f"the joint-space carry {why_s}"]
                return self._res("plan_place_chunk", False, facts, reason="PLACE_PLAN_FAILED", nxt=["plan_place_chunk"])
        worst, worst_at = (w_line, (0, at_line)) if line_ok else (w_joint, (0, at_joint))
        facts["carry"] = "straight line at the lift orientation" if line_ok else "joint-space move (slow)"
        facts["carry_min_clearance_mm"] = None if worst is None or worst > 1e8 else round(worst, 1)
        facts["carry_closest_object"] = None if worst_at is None else worst_at[1]
        if worst is not None and worst < 5.0:
            facts["problems"] = [f"carry path passes {worst:.0f} mm above/into {worst_at[1]}; lift higher "
                                 f"(carry_lift_mm) or approach from higher"]
            return self._res("plan_place_chunk", False, facts, reason="CARRY_COLLISION",
                             nxt=["verify_and_park_chunk", "plan_place_chunk"])
        facts["carry_lift_mm"] = float(carry_lift_mm or 0.0)
        mins = [float(chk.get("min_margin_deg") or 0.0), float(chl.get("min_margin_deg") or 99.0)]
        if via is not None:
            mins.append(float(chv.get("min_margin_deg") or 99.0))
        chosen = {"target": str(target_id_or_xyz), "release_open_mm": g_open, "_via": via, "_carry_line": line_ok,
                  "min_joint_margin_deg": round(min(mins), 1), "ik_solver": chk.get("ik_solver", "fast_ik"),
                  "_wps": {"above_target": (p_above, R_tcp), "release": (p_rel, R_tcp)},
                  "_q": {"above_target": qs[0], "release": qs[1]}}
        if getattr(self, "_probing", False):
            return self._res("plan_place_chunk", True, {"min_joint_margin_deg": chosen["min_joint_margin_deg"]})
        self.place_plan = chosen
        if self.plan is None:
            self.plan = {}
        self.plan.setdefault("_wps", {}).update(chosen["_wps"])
        self.plan.setdefault("_q", {}).update(chosen["_q"])
        self.plan["_q"].setdefault("rotate", chosen["_q"]["above_target"])
        self.stage = "PLACE_PLANNED"
        facts.update({"min_joint_margin_deg": chosen["min_joint_margin_deg"], "ik_solver": chosen["ik_solver"]})
        if virtual:
            facts["note"] = "planned for the object of the planned grasp (not lifted yet)"
        return self._res("plan_place_chunk", True, facts, nxt=["execute_place_chunk"])

    def execute_place_chunk(self, retreat_up_mm=80.0, release_open_mm=None, **_):
        if not self.place_plan:
            return self._res("execute_place_chunk", False, reason="PLACE_NOT_PLANNED", nxt=["plan_place_chunk"])
        if not self.held:
            return self._res("execute_place_chunk", False, reason="NOT_HOLDING", nxt=["perceive_objects"])
        W, Q = self.place_plan["_wps"], self.place_plan["_q"]
        if not getattr(self.io, "motion_allowed", True):
            return self._res("execute_place_chunk", True, {"dry_run": True, "target": self.place_plan["target"]},
                             nxt=["verify_and_park_chunk"])
        oid = self.held
        self._unlatch_if_needed(f"V4.9 carry above {self.place_plan['target']}")
        n0 = self._trips()
        via = self.place_plan.get("_via")
        if via is not None:
            r = self.io.move_line("V4.9 carry lift", via[0], self.CK.R_to_quat_xyzw(via[1]), 0.04)
            if not r.get("ok") and self._safety_hit(r, n0):
                return self._safety_stop("execute_place_chunk", "carry lift", r, n0)
            if not r.get("ok"):
                return self._res("execute_place_chunk", False, dict(r, object_still_held=not self._held_lost()),
                                 reason="CARRY_LIFT_FAILED", nxt=["plan_place_chunk", "verify_and_park_chunk"])
        r = {"ok": False}
        if self.place_plan.get("_carry_line"):
            p_ab, R_ab = W["above_target"]
            r = self.io.move_line(f"V4.9 carry above {self.place_plan['target']}", p_ab, self.CK.R_to_quat_xyzw(R_ab),
                                  0.04)
            if not r.get("ok") and self._safety_hit(r, n0):         # V4.9.3: never fall back after a safety stop
                return self._safety_stop("execute_place_chunk", "carry", r, n0)
        if not r.get("ok"):
            r = self.io.move_joints(f"V4.9 carry above {self.place_plan['target']}", Q["above_target"], slow=3.0)
            if not r.get("ok") and self._safety_hit(r, n0):
                return self._safety_stop("execute_place_chunk", "carry", r, n0)
        if not r.get("ok"):
            self._unlatch_if_needed("V4.9 recover after carry fail")
            lost = self._held_lost()
            if lost:
                self.held, self.stage = None, "OBSERVED"
            return self._res("execute_place_chunk", False, dict(r, object_still_held=not lost),
                             reason="OBJECT_DROPPED_DURING_CARRY" if lost else "CARRY_FAILED",
                             nxt=["perceive_objects", "verify_and_park_chunk"])
        r = self.io.move_line(f"V4.9 lower to {self.place_plan['target']}", W["release"][0],
                              self.CK.R_to_quat_xyzw(W["release"][1]), 0.015, strict=True)
        if not r.get("ok") and self._safety_hit(r, n0):
            return self._safety_stop("execute_place_chunk", "lower", r, n0)
        if not r.get("ok"):
            self._unlatch_if_needed("V4.9 recover after lower fail")
            lost = self._held_lost()
            if lost:
                self.held, self.stage = None, "OBSERVED"
            return self._res("execute_place_chunk", False, dict(r, object_still_held=not lost),
                             reason="OBJECT_DROPPED_DURING_LOWER" if lost else "LOWER_FAILED",
                             nxt=["verify_and_park_chunk", "perceive_objects"])
        p_rel = self._obj_xyz(oid)
        g_open = float(release_open_mm if release_open_mm is not None else self.place_plan["release_open_mm"])
        self.io.gripper_open(g_open)
        self.io.sleep(0.35)
        self.held = None
        self.stage = "RELEASED"
        p_ret = np.asarray(W["release"][0], float) + np.array([0.0, 0.0, float(retreat_up_mm) / 1000.0])
        self.io.move_line("V4.9 retreat up", p_ret, self.CK.R_to_quat_xyzw(W["release"][1]), 0.04)
        facts = {"target": self.place_plan["target"], "released_open_mm": g_open}
        p_after = self._obj_xyz(oid)
        if p_rel is not None and p_after is not None:
            facts["object_moved_after_release_mm"] = {"xy": round(float(np.linalg.norm(p_after[:2] - p_rel[:2])) * 1000.0, 1),
                                                      "z": round(float(p_after[2] - p_rel[2]) * 1000.0, 1)}
        return self._res("execute_place_chunk", True, facts, nxt=["verify_and_park_chunk"])

    def verify_and_park_chunk(self, action="clear_view_and_perceive", prompts=None, **_):
        ok_l, _why = self._unlatch_if_needed("V4.9 verify_and_park_chunk")
        if self.held and not self._held_lost():
            return self._res("verify_and_park_chunk", False,
                             {"held_object": self.held, "latch_cleared": bool(ok_l),
                              "note": "the object is still in the gripper"},
                             reason="OBJECT_STILL_HELD", nxt=["plan_place_chunk", "execute_place_chunk"])
        self.held = None
        if not getattr(self.io, "motion_allowed", True):
            self.stage = "VERIFIED"
            return self._res("verify_and_park_chunk", True, {"dry_run": True, "action": action})
        r_home = self.io.go_home("V4.9 clear view / park")
        if action == "return_home":
            if self.stage == "RELEASED":
                self.stage = "VERIFIED"
            return self._res("verify_and_park_chunk", bool(r_home.get("ok", True)), {"action": action})
        before = dict(self.desc or {})
        obs = self.perceive_objects(prompts=_clean_prompts(prompts), use_finetuned=True)
        now_ids = {o["object_id"] for o in obs.get("objects", [])} | {o["object_id"] for o in obs.get("other_objects", [])}
        hidden = [d for oid, d in before.items() if oid not in now_ids and d.get("label") != "egg_slot"]
        for oid, d in before.items():                  # partly covered now (seen much smaller than before)
            dn = (self.desc or {}).get(oid)
            if dn is not None and d.get("label") != "egg_slot" and d.get("size_mm") and dn.get("size_mm"):
                a0 = float(d["size_mm"][0]) * float(d["size_mm"][1])
                a1 = float(dn["size_mm"][0]) * float(dn["size_mm"][1])
                if a1 < 0.7 * a0 and not dn.get("open_cavity") and d.get("open_cavity"):
                    hidden.append(d)
        rel = list(obs.get("relations", []))
        if hidden:                                     # V4.9: not seen now (e.g. covered from above): last pose
            hid_ids = {h["object_id"] for h in hidden}
            cur = [d for oid, d in (self.desc or {}).items() if oid not in hid_ids]
            for r in G49.relations(cur + [dict(h, object_id=h["object_id"]) for h in hidden]):
                if r["of"] in {h["object_id"] for h in hidden} or r["object"] in {h["object_id"] for h in hidden}:
                    rel.append(dict(r, note="uses the last measured pose of an object not visible now"))
        src_left = True
        if self.picked_from is not None:
            for o in obs.get("objects") or []:
                if float(np.linalg.norm(np.asarray(o["center_xyz_m"][:2], float) - self.picked_from)) < 0.030:
                    src_left = False
                    break
        if self.stage == "RELEASED":
            self.stage = "VERIFIED"
        return self._res("verify_and_park_chunk", bool(r_home.get("ok", True)), {
            "action": action,
            "source_left_original_pose": src_left,
            "relations": rel,
            "not_visible_now": [{"object_id": h["object_id"], "last_center_xyz_m": h["center_xyz_m"],
                                 "last_top_z_m": h["top_z_m"]} for h in hidden],
            "visible_objects": [{k: o.get(k) for k in ("object_id", "center_xyz_m", "top_z_m", "bottom_z_m", "size_mm",
                                                       "tilt_deg", "upright") if k in o}
                                for o in obs.get("objects", [])],
        })

    def run_tool(self, name, args=None):
        fn = {
            "perceive_objects": self.perceive_objects,
            "preview_grasp": self.preview_grasp,
            "plan_grasp_chunk": self.plan_grasp_chunk,
            "execute_grasp_chunk": self.execute_grasp_chunk,
            "plan_place_chunk": self.plan_place_chunk,
            "execute_place_chunk": self.execute_place_chunk,
            "verify_and_park_chunk": self.verify_and_park_chunk,
        }.get(name)
        if fn is None:
            return {"ok": False, "tool": name, "reason": "UNKNOWN_TOOL"}
        n0 = None
        if hasattr(self.io, "safety_trips"):
            try:
                n0 = int(self.io.safety_trips())
            except Exception:
                n0 = None
        res = fn(**(args or {}))
        if n0 is not None:
            try:
                n1 = int(self.io.safety_trips())
            except Exception:
                n1 = n0
            if n1 > n0 and isinstance(res, dict):
                pv = res.setdefault("physical_verification", {})
                pv["safety_events"] = {"count": n1 - n0, "note": "the safety guard stopped the arm during this tool "
                                       "(runaway joint speed / unstable contact in the simulator)"}
                self.io.emit(f"[WARN] [V4.9 GENERAL] {name}: {n1 - n0} safety stop(s) during this tool")
        return res


# ================================================================================================ LLM Tools & Loop
def _fn(name, desc, props, req):
    return {"type": "function", "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req, "additionalProperties": False},
            "strict": True}


_NUM = {"type": "number"}
_XY = {"type": "array", "items": {"type": "number"},
       "description": "[dx_mm, dy_mm] in world x / y from the object's center_xyz_m"}
_GRASP_PROPS = {
    "object_id": {"type": "string"},
    "xy_offset_mm": _XY,
    "grasp_z_offset_from_top_mm": {"type": "number", "description": "grasp point height relative to the object's top_z_m (mm)"},
    "closing_yaw_deg": {"type": "number", "description": "world direction of the pad-to-pad closing line (deg)"},
    "pitch_deg": {"type": "number", "description": "tilt of the gripper from vertical about its jaw-width axis (deg)"},
    "open_mm": {"type": "number", "description": "finger opening before the approach (mm)"},
}
GENERAL_TOOLS = [
    _fn("perceive_objects",
        "Look at the table (RGB-D). Returns measured geometry per object: center, top, bottom, footprint axes and size, "
        "top-surface cross sections, side profile (outer / inner half widths at several heights when measurable), "
        "open cavity, geometric relations between objects, and the gripper geometry. Objects not matched by the "
        "prompts are listed as other_objects (they are still obstacles).",
        {"prompts": {"type": "array", "items": {"type": "string"}}, "use_finetuned": {"type": "boolean"}},
        ["prompts", "use_finetuned"]),
    _fn("preview_grasp",
        "No motion. For a grasp point / closing direction / opening, report collisions of the gripper with any object "
        "or the table at the grasp pose, where each pad first touches material while closing, how far the first "
        "contact would push the object, and the pad gap when both touch.",
        dict(_GRASP_PROPS), list(_GRASP_PROPS)),
    _fn("plan_grasp_chunk",
        "No motion. Plan pregrasp -> vertical descent -> grasp -> vertical lift for these parameters (same meaning as "
        "preview_grasp). The planner may rotate the closing direction by up to yaw_tolerance_deg to find an arm "
        "posture. Returns the chosen pose and its finger preview, or the reasons each direction was rejected.",
        dict(_GRASP_PROPS, yaw_tolerance_deg=_NUM, pregrasp_height_mm=_NUM, lift_height_mm=_NUM),
        list(_GRASP_PROPS) + ["yaw_tolerance_deg", "pregrasp_height_mm", "lift_height_mm"]),
    _fn("execute_grasp_chunk",
        "Move: open, approach, descend, close with this force, lift. Reports the measured clamp (pad contacts, contact "
        "forces, opening, hold), how far the object moved during descent / clamp and whether it rose with the gripper.",
        {"clamp_force_n": _NUM}, ["clamp_force_n"]),
    _fn("plan_place_chunk",
        "No motion. With the held object, plan carry -> lower -> release over a target (object_id or '[x,y,z]' "
        "support point). The held object is positioned by its own measured geometry: its center goes to the target "
        "center + xy_offset_mm and its lowest point to target top + held_bottom_above_target_top_mm (may be negative "
        "to lower it into something). Reports clearances of the held object and the fingers (closed and opened to "
        "release_open_mm) above every other object, the carry path clearance, and problems.",
        {"target_id_or_xyz": {"type": "string"}, "xy_offset_mm": _XY, "held_bottom_above_target_top_mm": _NUM,
         "yaw_change_deg": _NUM, "approach_height_mm": _NUM, "release_open_mm": _NUM,
         "carry_lift_mm": {"type": "number", "description": "extra straight-up lift before the carry (mm, 0 = none)"}},
        ["target_id_or_xyz", "xy_offset_mm", "held_bottom_above_target_top_mm", "yaw_change_deg",
         "approach_height_mm", "release_open_mm", "carry_lift_mm"]),
    _fn("execute_place_chunk",
        "Move: carry over the target, lower vertically, open the fingers, retreat up. Reports how the object moved "
        "after release.",
        {"retreat_up_mm": _NUM}, ["retreat_up_mm"]),
    _fn("verify_and_park_chunk",
        "Move the arm to the camera-clear pose and (clear_view_and_perceive) look again: objects and their measured "
        "relations (inside / on).",
        {"action": {"type": "string", "enum": ["clear_view_and_perceive", "return_home"]},
         "prompts": {"type": "array", "items": {"type": "string"}}},
        ["action", "prompts"]),
]

GENERAL_SYSTEM_INSTRUCTIONS = """You are the planner of a 7-DOF NERO arm with a parallel two-finger gripper doing pick-and-place tasks given in natural language.
There are no object-specific skills or rules: every grasp and place parameter is your decision, made from the measured facts the tools return.
- perceive_objects gives measured geometry (center, top, bottom, footprint axes, cross sections, side profile, open cavity, relations) and the gripper's geometry. Labels come from the detector and may be wrong; the geometry is measured.
- preview_grasp, plan_grasp_chunk and plan_place_chunk never move the arm. Use them to check parameters before moving: they report collisions, where each pad would touch, how far an object would be pushed, clearances and rejections.
- execute_grasp_chunk and execute_place_chunk move the arm and report what was measured. If a result differs from what you expected, use the measurements to decide the next step.
- If a required object is not detected, do not invent coordinates; report what is missing.
- Finish with verify_and_park_chunk and report the outcome from its relations; do not claim more than it measured."""


def run_general_pick_place(io, task, mode="execute", agent=None, llm_cfg=None, recorder=None):
    """Run General Pick&Place in plan / hover / execute mode via LLM chunk orchestration."""
    sk = GeneralPickPlaceSkills(io)
    io.skills = sk
    if recorder is not None:
        recorder.wrap(sk, perception_fn=lambda: getattr(io, "last_perception", None))
    io.home_q = io.current_q()
    c = dict(llm_cfg or {})
    if agent is None:
        return {"ok": False, "reason": "LLM_AGENT_REQUIRED", "skills": sk,
                "final": "General pick&place 需要启用 LLM Agent（请检查模型与 API Key 配置）"}

    model = str(c.get("model") or "gpt-5.6-luna")
    api_key = str(c.get("api_key") or "")
    effort = str(c.get("reasoning_effort") or "low")
    max_steps = max(10, min(60, int(c.get("max_tool_steps", 30))))
    dry_run = (mode == "plan")

    with agent._lock:
        if agent._running:
            raise RuntimeError("LLM agent is already running")
        agent._running = True
    agent._stop_event.clear()
    agent._last_response_id = None
    emit = io.emit
    rounds, steps, final = 0, 0, ""
    try:
        ok, key_status = agent.api_key_status(api_key, model)
        if not ok:
            raise RuntimeError(key_status)
        emit(f"[V4.8.9 GENERAL START] mode={mode} | model={model} | task={task!r}")
        mode_note = ("\n[MODE=plan (dry_run): Call perceive_objects, plan_grasp_chunk, and plan_place_chunk ONLY; do NOT execute motion.]"
                     if mode == "plan" else
                     "\n[MODE=hover: Call perceive_objects, plan_grasp_chunk, and plan_place_chunk; physical hover motion will follow.]"
                     if mode == "hover" else "")
        payload = {
            "model": model,
            "instructions": GENERAL_SYSTEM_INSTRUCTIONS + mode_note,
            "input": str(task),
            "tools": GENERAL_TOOLS,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "reasoning": {"effort": effort},
            "max_output_tokens": 1200,
            "store": True,
        }
        while True:
            if agent._stop_event.is_set() or io.stop_ev.is_set():
                raise RuntimeError("Task stopped by user")
            rounds += 1
            t_req = time.perf_counter()
            resp = agent._post_responses(payload, api_key)
            agent._log_usage(resp, model)
            agent._last_response_id = resp.get("id")
            calls = agent._function_calls(resp)
            text = agent._extract_text(resp)
            emit(f"[V4.8.9 LLM #{rounds}] elapsed={time.perf_counter() - t_req:.2f}s | calls={len(calls)}")
            if text:
                emit(f"[LLM TEXT] {text}")
            if not calls:
                final = text
                break
            outs = []
            for call in calls:
                if agent._stop_event.is_set() or io.stop_ev.is_set():
                    raise RuntimeError("Task stopped by user")
                steps += 1
                if steps > max_steps:
                    raise RuntimeError(f"tool step budget {max_steps} used up")
                name = str(call.get("name", ""))
                try:
                    args = json.loads(call.get("arguments") or "{}")
                except Exception:
                    args = {}
                emit(f"[V4.8.9 CHUNK #{steps}] {name}({json.dumps(args, ensure_ascii=False)})")
                if mode in ("plan", "hover") and name in ("execute_grasp_chunk", "execute_place_chunk"):
                    res = {"ok": True, "dry_run": True, "tool": name, "note": f"skipped physical execution in mode={mode}"}
                else:
                    res = sk.run_tool(name, args)
                outs.append({
                    "type": "function_call_output",
                    "call_id": str(call.get("call_id", "")),
                    "output": json.dumps(res, ensure_ascii=False, default=str),
                })
            payload = {
                "model": model,
                "instructions": GENERAL_SYSTEM_INSTRUCTIONS + mode_note,
                "previous_response_id": agent._last_response_id,
                "input": outs,
                "tools": GENERAL_TOOLS,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "reasoning": {"effort": effort},
                "max_output_tokens": 1200,
                "store": True,
            }

        if mode == "hover" and sk.grasp_plan and sk.place_plan:
            try:
                io.gripper_open(60.0)
                io.move_joints("V4.8.9 hover pregrasp", sk.grasp_plan["_q"]["pregrasp"])
                io.move_line("V4.8.9 hover lift", sk.grasp_plan["_wps"]["lift"][0],
                             sk.CK.R_to_quat_xyzw(sk.grasp_plan["_wps"]["lift"][1]), 0.05)
                io.move_joints("V4.8.9 hover above target", sk.place_plan["_q"]["above_target"])
            finally:
                io.go_home("V4.8.9 hover back")

        ok_final = (sk.stage == "VERIFIED") if mode == "execute" else bool(sk.grasp_plan or sk.cam)
        return {"ok": ok_final, "final": final, "skills": sk}
    finally:
        with agent._lock:
            agent._running = False


# ================================================================================================ Sim IO & Entry (V4.8.10)
def _sim_prompt_matches(kind: str, det_id: str, prompts: List[str]) -> bool:
    ps = [str(p).strip().lower() for p in (prompts or []) if str(p).strip()]
    if not ps:
        return True
    for p in ps:
        if p in ("object", "objects", "all", "anything", "item", "items", "物体"):
            return True
        if p in kind.lower() or p in det_id.lower() or kind.lower() in p:
            return True
        if kind == "egg" and any(k in p for k in ("egg", "蛋", "鸡蛋")):
            return True
        if kind == "cup" and any(k in p for k in ("cup", "mug", "杯", "杯子")):
            return True
        if kind == "egg_slot" and any(k in p for k in ("slot", "carton", "tray", "holder", "蛋托", "槽")):
            return True
    return False


def _body_world_aabb(p_xyz, q_quat, local_min_m, local_max_m, quat_wxyz=False):
    """Transform a body's local 3D bounding box [local_min_m, local_max_m] by its world pose (p_xyz, q_quat)
    and return (center_xyz_m, top_z_m, height_mm, size_xy_mm, yaw_deg, tilt_deg, upright)."""
    import NERO_V362_Cup_Skills as CK
    p = np.asarray(p_xyz, float)
    if q_quat is not None and len(q_quat) == 4:
        qf = [float(v) for v in q_quat]
        q_xyzw = [qf[1], qf[2], qf[3], qf[0]] if quat_wxyz else qf
        nrm = float(np.linalg.norm(q_xyzw))
        q_xyzw = [v / nrm for v in q_xyzw] if nrm > 1e-9 else [0.0, 0.0, 0.0, 1.0]
        R = CK.quat_xyzw_to_R(q_xyzw)
    else:
        R = np.eye(3)
    lmin = np.asarray(local_min_m, float)
    lmax = np.asarray(local_max_m, float)
    corners = np.array([
        [x, y, z]
        for x in (lmin[0], lmax[0])
        for y in (lmin[1], lmax[1])
        for z in (lmin[2], lmax[2])
    ], float)
    world_corners = (R @ corners.T).T + p
    wmin = world_corners.min(axis=0)
    wmax = world_corners.max(axis=0)
    center = 0.5 * (wmin + wmax)
    top_z = float(wmax[2])
    span = (wmax - wmin) * 1000.0
    size_xy_mm = [round(float(span[0]), 1), round(float(span[1]), 1)]
    height_mm = round(float(span[2]), 1)
    zz = float(R[2, 2])
    tilt_deg = round(math.degrees(math.acos(max(-1.0, min(1.0, zz)))), 1)
    yaw_deg = round(math.degrees(math.atan2(float(R[1, 0]), float(R[0, 0]))), 1)
    upright = bool(tilt_deg <= 25.0)
    return [round(float(center[0]), 4), round(float(center[1]), 4), round(float(center[2]), 4)], round(top_z, 4), height_mm, size_xy_mm, yaw_deg, tilt_deg, upright


def _sim_general_perception(pose_fn, prompts=None):
    import NERO_V343_Scenes as SC
    p_list = _clean_prompts(prompts)
    objs = []
    # Local 3D bounding box (local_min_m, local_max_m) in each MuJoCo body's own frame
    local_bounds = {
        "obj_egg_1": ("egg_1", "egg", (-0.029, -0.022, -0.022), (0.029, 0.022, 0.022)),
        "obj_egg_2": ("egg_2", "egg", (-0.029, -0.022, -0.022), (0.029, 0.022, 0.022)),
        "obj_cup_1": ("cup_1", "cup", (-0.0475, -0.0475, -0.121), (0.0475, 0.0475, 0.0)),
        "obj_cup_2": ("cup_2", "cup", (-0.0475, -0.0475, -0.121), (0.0475, 0.0475, 0.0)),
    }
    quat_wxyz = bool(getattr(pose_fn, "_quat_wxyz", False))
    for body_name, (det_id, kind, lmin, lmax) in local_bounds.items():
        if not _sim_prompt_matches(kind, det_id, p_list):
            continue
        p, q = pose_fn(body_name)
        if p is None:
            continue
        center_xyz, top_z, h_mm, size_xy_mm, yaw_deg, tilt_deg, upright = _body_world_aabb(
            p, q, lmin, lmax, quat_wxyz=quat_wxyz
        )
        objs.append({
            "det_id": det_id,
            "kind": kind,
            "center_xyz_m": center_xyz,
            "top_z_m": top_z,
            "height_mm": h_mm,
            "size_mm": size_xy_mm,
            "yaw_deg": yaw_deg,
            "tilt_deg": tilt_deg,
            "upright": upright,
            "conf": 0.99,
            "trusted": True,
        })
    # Static egg carton & slots when the prompt mentions carton/tray/holder/slot/box
    want_carton = any(
        any(k in str(p).lower() for k in ("carton", "tray", "holder", "box", "蛋托", "盒"))
        for p in p_list
    )
    if want_carton:
        objs.append({
            "det_id": "egg_carton_1",
            "kind": "egg_carton",
            "center_xyz_m": [-0.530, 0.000, 0.016],
            "top_z_m": 0.032,
            "height_mm": 32.0,
            "size_mm": [110.0, 130.0],
            "yaw_deg": 0.0,
            "conf": 0.99,
            "trusted": True,
        })
    want_slots = any(
        any(k in str(p).lower() for k in ("slot", "carton", "tray", "holder", "蛋托", "槽"))
        for p in p_list
    )
    for slot_id, sxyz in (SC.GENERAL_SLOTS.items() if want_slots else ()):
        if True:
            objs.append({
                "det_id": slot_id,
                "kind": "egg_slot",
                "center_xyz_m": [round(float(sxyz[0]), 4), round(float(sxyz[1]), 4), round(float(sxyz[2]) * 0.5, 4)],
                "top_z_m": round(float(sxyz[2]), 4),
                "height_mm": round(float(sxyz[2]) * 1000.0, 1),
                "size_mm": [48.0, 48.0],
                "yaw_deg": 0.0,
                "conf": 0.99,
                "trusted": True,
            })
    return {
        "ok": True,
        "scene": "general_pick_place",
        "table_z_m": 0.0,
        "fence_z_table_m": 0.0,
        "objects": objs,
    }



# ================================================================================================ V4.9 probe perception
def _slot_objects(slots: Optional[Dict[str, Tuple[float, float, float]]]):
    out = []
    for slot_id, sxyz in (slots or {}).items():
        out.append({"object_id": slot_id, "label": "egg_slot", "kind": "egg_slot",
                    "center_xyz_m": [round(float(sxyz[0]), 4), round(float(sxyz[1]), 4), round(float(sxyz[2]) * 0.5, 4)],
                    "top_z_m": round(float(sxyz[2]), 4), "bottom_z_m": 0.0, "height_mm": round(float(sxyz[2]) * 1000.0, 1),
                    "size_mm": [48.0, 48.0], "axis_a_yaw_deg": 0.0, "yaw_deg": 0.0, "conf": 0.99, "trusted": True,
                    "note": "a detected carton cell (the cell itself is part of the carton's height map)"})
    return out


DESC_LEGEND = {                        # V4.9.2: sent once per task instead of inside every object
    "size_mm": "[along axis a, along axis b]; axis a yaw = axis_a_yaw_deg, axis b = axis a + 90 deg",
    "top_surface_segments_mm": "cross sections through center_xyz_m seen from above: [from_offset_mm, to_offset_mm, "
                               "surface height above table mm (null = not this object)]",
    "side_profile": "at each height above the table: outer / inner half width (mm) from center_xyz_m along +a, -a, "
                    "+b, -b; a missing direction was not measured",
    "open_cavity": "the surface inside the footprint is lower than its rim (measured from above)",
    "unchanged": "the object has not moved since the previous look; its full description is in that result",
    "enters_unobserved_space": "part of the gripper / held object would be in space the camera could not see (behind "
                               "an object), below the highest point anything hidden there could reach; collision "
                               "checks only use what was seen",
}

REAL_LIKE_NOISE = {                    # V4.9.2 REAL-LIKE PERCEPTION (twin): what a fixed D435i would roughly give
    "cam_pos": (-0.95, 0.0, 0.72),     # the backend's fixed camera (NERO_V308 camera_config)
    "pos_sigma_mm": 1.0,               # per look, per object (2 mm per frame, 5-frame average as V3.83 / V3.84)
    "yaw_sigma_deg": 1.5,
    "bias_sigma_mm": 3.0,              # per episode, all objects (hand-eye calibration error)
    "depth_k": 0.0025,                 # depth sigma = k * d^2 (2.3 mm at 0.95 m)
    "dropout": 0.02,
    "side_sigma_mm": 1.5,
    "seed": 0,
}


class ProbeVision:
    """V4.9 PROBE VISION: the sim scene objects in a robot-free MuJoCo copy, posed from the simulator before every
    look; ray-cast height map + descriptors + relations.  pose_fn(body) -> (xyz, quat_wxyz).
    V4.9.2: noise=REAL_LIKE_NOISE (or a dict) -> the height map comes from a simulated fixed depth camera (shadows
    = unobserved space, depth noise, dropouts), object poses get look / episode errors, the side profile keeps only
    the sides the camera sees (outer: facing it, inner: the far wall seen through an opening), the bottom is not
    measured (table, or the floor of the opening it stands in)."""

    def __init__(self, world_xml, assets, pose_fn, slots=None, noise=None):
        self.probe = G49.MjProbe(world_xml, assets)
        self.pose_fn = pose_fn
        self.slots = slots
        self.noise = dict(REAL_LIKE_NOISE, **noise) if isinstance(noise, dict) else (dict(REAL_LIKE_NOISE) if noise else None)
        if self.noise:
            self.rng = np.random.default_rng(int(self.noise.get("seed", 0)))
            self.bias = self.rng.normal(0.0, float(self.noise["bias_sigma_mm"]) / 1000.0, 3)

    def _noisy_pose(self, body):
        p, q = self.pose_fn(body)
        if p is None or not self.noise:
            return p, q
        n = self.noise
        p = np.asarray(p, float) + self.bias + np.r_[self.rng.normal(0.0, n["pos_sigma_mm"] / 1000.0, 2),
                                                     self.rng.normal(0.0, 0.5 * n["pos_sigma_mm"] / 1000.0)]
        if q is not None:
            a = math.radians(float(self.rng.normal(0.0, n["yaw_sigma_deg"]))) / 2.0
            w0, x0, y0, z0 = [float(v) for v in q]
            w1, z1 = math.cos(a), math.sin(a)                       # q_noise (about z) * q
            q = [w1 * w0 - z1 * z0, w1 * x0 - z1 * y0, w1 * y0 + z1 * x0, w1 * z0 + z1 * w0]
        return list(p), q

    def _side_fn(self, cam_xy):
        probe, n, rng = self.probe, self.noise, getattr(self, "rng", None)

        def fn(oid, ctr, yaw, levels, reach, floor_z):
            rows = probe.side_profile(oid, ctr, yaw, levels, reach, floor_z)
            if not n:
                return rows
            a = math.radians(yaw)
            dirs = {"+a": (math.cos(a), math.sin(a)), "-a": (-math.cos(a), -math.sin(a)),
                    "+b": (-math.sin(a), math.cos(a)), "-b": (math.sin(a), -math.cos(a))}
            to_cam = np.asarray(cam_xy, float) - np.asarray(ctr, float)[:2]
            to_cam /= max(float(np.linalg.norm(to_cam)), 1e-9)
            for r in rows:
                for k, (ux, uy) in dirs.items():
                    facing = ux * to_cam[0] + uy * to_cam[1]
                    o, i = r["outer_half_width_mm"].get(k), r["inner_half_width_mm"].get(k)
                    r["outer_half_width_mm"][k] = (round(o + float(rng.normal(0.0, n["side_sigma_mm"])), 1)
                                                   if (o is not None and facing > 0.2) else None)
                    r["inner_half_width_mm"][k] = (round(i + float(rng.normal(0.0, n["side_sigma_mm"])), 1)
                                                   if (i is not None and facing < -0.2) else None)
            return rows
        return fn

    def look(self, table_z=0.0):
        if not self.noise:
            self.probe.set_poses(self.pose_fn)
            hm = self.probe.heightmap(table_z=table_z)
            descs = []
            for oid in self.probe.names:
                d = G49.describe(hm, oid, self.probe.bottom_z(oid), self.probe.side_profile)
                if d is not None:
                    descs.append(d)
            return hm, descs, G49.relations(descs)
        self.probe.set_poses(self._noisy_pose)
        cam = self.noise["cam_pos"]
        hm = self.probe.camera_heightmap(cam, table_z=table_z, rng=self.rng, depth_k=self.noise["depth_k"],
                                         dropout=self.noise["dropout"])
        side = self._side_fn(cam[:2])
        descs = []
        for oid in self.probe.names:
            d = G49.describe(hm, oid, table_z, side)
            if d is not None:
                d["measurement"] = ("simulated fixed depth camera: position / depth noise, unseen sides not "
                                    "measured, bottom not measured (table or the floor of the opening it is in)")
                descs.append(d)
        for d in descs:                        # bottom: the floor of an opening it stands in, else the table
            for e in descs:
                cav = e.get("open_cavity")
                if e is d or not cav:
                    continue
                dxy = np.asarray(d["center_xyz_m"][:2]) - np.asarray(e["center_xyz_m"][:2])
                r_open = min(v for v in cav["opening_half_width_mm"].values()) / 1000.0
                if float(np.linalg.norm(dxy)) < r_open and float(d["top_z_m"]) < float(e["top_z_m"]) + 0.08:
                    fz = table_z + float(cav["floor_z_above_table_mm"]) / 1000.0
                    if fz < float(d["top_z_m"]) - 0.005:
                        d["bottom_z_m"] = round(fz, 4)
                        d["height_mm"] = round((float(d["top_z_m"]) - fz) * 1000.0, 1)
                        d["center_xyz_m"][2] = round((float(d["top_z_m"]) + fz) / 2.0, 4)
        return hm, descs, G49.relations(descs)

    def perceive(self, prompts=None):
        hm, descs, rel = self.look()
        want_slots = any(any(k in str(p).lower() for k in ("slot", "carton", "tray", "holder", "蛋托", "槽"))
                         for p in (prompts or []))
        objs = descs + (_slot_objects(self.slots) if want_slots else [])
        return {"ok": True, "scene": "general_pick_place", "table_z_m": 0.0, "fence_z_table_m": 0.0,
                "objects": objs, "heightmap": hm, "relations": rel}

    def describe_object(self, oid):
        hm, descs, _rel = self.look()
        for d in descs:
            if d["object_id"] == oid:
                return d
        return None

    def position(self, oid):
        for b in self.probe.free:
            if G49.object_id_for(b, "") == oid:
                p, _q = self._noisy_pose(b) if self.noise else self.pose_fn(b)
                return p
        return None


def _make_sim_io_class():
    import threading
    import NERO_V362_Cup_Skills as CK

    class SimGeneralIO(CK.BackendIO):
        """V4.8.10: BackendIO adapter for GeneralPickPlaceSkills in MuJoCo simulation."""
        name = "sim_general"
        is_sim = True
        is_agx = True
        tip_drop_m = 0.030

        def __init__(self, ctx, motion_allowed=True):
            super().__init__(ctx)
            self.CK = CK
            self.fast = CK.FastArm()
            self.cfg = {}
            self.stop_ev = threading.Event()
            self.motion_allowed = bool(motion_allowed)
            self.home_q = None
            self.last_perception = None

        def gripper_profile(self):
            return CK.OFFICIAL_PROFILE

        def _ensure_unlatched(self, label):
            if self.latch_pending():
                ok, why = self.clear_latch(f"V4.8.10 auto-unlatch before {label}")
                self.emit(f"[INFO] [V4.8.10 LATCH] auto-unlatch before {label}: ok={ok} ({why})")

        def move_joints(self, label, q, slow=None):
            self._ensure_unlatched(label)
            res = super().move_joints(label, q, slow=slow)
            if not res.get("ok") and self.latch_pending():
                self._ensure_unlatched(f"recover {label}")
            return res

        def go_home(self, label):
            self._ensure_unlatched(label)
            return super().go_home(label)

        def move_line(self, label, xyz, quat, speed=0.05, slow=None, strict=False):
            self._ensure_unlatched(label)
            self._sleep(0.20)
            eff_slow = float(slow) if slow else (2.5 if float(speed) <= 0.025 else 1.5)
            # Build a smooth, strictly continuous local-IK joint path from self.fast.fk(q0) (zero initial step discontinuity)
            # and retime via _make_incremental_joint_trajectory so J7 never spikes at segment 1.
            if hasattr(self.core, "_make_incremental_joint_trajectory"):
                try:
                    import NERO_V372_Sort_Skills as SK72
                    q0 = self.current_q()
                    T0 = self.fast.fk(np.asarray(q0, float))
                    p0, R0 = T0[:3, 3].copy(), T0[:3, :3].copy()
                    p1 = np.asarray(xyz, float)
                    R = CK.quat_xyzw_to_R(quat)
                    dist = float(np.linalg.norm(p1 - p0))
                    Rd = R @ R0.T
                    turn = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rd) - 1.0) / 2.0))))
                    n = int(min(60, max(4, math.ceil(dist / 0.004), math.ceil(turn / 2.0))))
                    ch = SK72._turning_segment("s", p0, R0, p1, R, n) + [("~end", p1, R)]
                    path = [list(q0)]
                    pts = [p0]
                    qp = list(q0)
                    ok_ik = True
                    for _nm, pk, Rk in ch:
                        rk = self.fast.ik([qp], pk, Rk, iters=30)
                        if not rk or rk[0] is None:
                            ok_ik = False
                            break
                        if max(abs(a - b) for a, b in zip(rk[0], qp)) > math.radians(8.0):
                            ok_ik = False
                            break
                        qp = list(rk[0])
                        path.append(qp)
                        pts.append(np.asarray(pk, float))
                    if ok_ik:
                        chk = CK.check_joint_path(path, jump_deg=8.0)
                        if chk.get("ok"):
                            traj = self.core._make_incremental_joint_trajectory(path, [list(map(float, p_)) for p_ in pts])
                            self.c["require_arm"]()
                            s0 = int(self.sh["safety_trip_count"].value)
                            f = max(1.2, 0.03 / max(float(speed), 1e-3))
                            err = self._exec(label, traj, f, s0)
                            if err:
                                if self.latch_pending():
                                    self._ensure_unlatched(f"recover {label}")
                                return err
                            r_post = self._post(label, xyz, s0)
                            r_post["path_check"] = chk
                            return r_post
                except Exception:
                    pass
            res = super().move_line(label, xyz, quat, speed=speed, slow=eff_slow)
            if not res.get("ok") and self.latch_pending():
                self._ensure_unlatched(f"recover {label}")
            return res

        def gripper_clamp(self, force_n, settle_s=2.2):
            res = super().gripper_clamp(force_n, settle_s=settle_s)
            res["gap_mm"] = res.get("opening_mm")
            return res

        def gripper_state(self):
            res = super().gripper_state()
            res["gap_mm"] = res.get("opening_mm")
            return res

        def perceive_general(self, tag, prompts=None, use_finetuned=True):
            def _pose(body_name):
                name = body_name[4:] if str(body_name).startswith("obj_") else str(body_name)
                # Bypass _v329_object_pose (which blocks egg_* when v311_mode=='camera' and no camera scan ran)
                # by reading simulator TF directly via wait_for_tf(f"sim_obj_{name}") first.
                if hasattr(self.core, "wait_for_tf"):
                    for tf_child in (f"sim_obj_{name}", f"sim_{body_name}"):
                        try:
                            tfm = self.core.wait_for_tf(tf_child, timeout_sec=0.35)
                            tr = tfm.transform.translation
                            rot = tfm.transform.rotation
                            return [float(tr.x), float(tr.y), float(tr.z)], [float(rot.x), float(rot.y), float(rot.z), float(rot.w)]
                        except Exception:
                            pass
                try:
                    p, q = self.core.object_pose(name)
                    return p, q
                except Exception:
                    try:
                        p, q = self.core.object_pose(body_name)
                        return p, q
                    except Exception:
                        return None, None

            def _pose_wxyz(body_name):
                p, q = _pose(body_name)
                if p is None:
                    return None, None
                q = list(q) if q is not None else [0.0, 0.0, 0.0, 1.0]
                return p, [q[3], q[0], q[1], q[2]]          # TF / object_pose give x, y, z, w
            if getattr(self, "_vision", None) is None:
                try:
                    import NERO_V343_Scenes as SC
                    self._vision = ProbeVision(SC.general_world_xml(bare=True), SC.CUP_ASSETS, _pose_wxyz,
                                               slots=SC.GENERAL_SLOTS)
                except Exception as e:
                    self._vision = False
                    self.emit(f"[WARN] [V4.9 GENERAL] probe vision unavailable ({e}); boxes from the old perception")
            if self._vision:
                obs = self._vision.perceive(prompts)
            else:
                obs = _sim_general_perception(_pose, prompts=prompts)
            self.last_perception = obs
            self.emit(f"[V4.9 SIM GENERAL PERCEIVE] {tag} | count={len(obs['objects'])}")
            return obs

        def finger_model(self):
            return G49.SIM_FINGERS

        def object_position(self, oid):
            v = getattr(self, "_vision", None)
            return v.position(oid) if v else None

        def describe_object(self, oid):
            v = getattr(self, "_vision", None)
            return v.describe_object(oid) if v else None

    return SimGeneralIO


def _make_twin_io_class():
    import threading
    import NERO_V343_Scenes as SC
    import NERO_V362_Cup_Skills as CK

    class TwinGeneralIO(CK.TwinIO):
        """V4.8.10: Headless MuJoCo twin adapter for GeneralPickPlaceSkills.  V4.9: any scene (world_xml + the names
        of the graspable bodies), perception = ProbeVision on the twin's own object poses."""
        name = "twin_general"
        is_sim = True
        is_agx = True
        tip_drop_m = 0.030

        def __init__(self, log=None, start_tcp=None, motion_allowed=True, world_xml=None, hold_names=None,
                     assets=None, slots="default", noise=None):
            wx = world_xml if world_xml is not None else SC.general_world_xml(bare=True)
            super().__init__(
                log=log,
                start_tcp=start_tcp,
                pose_mode="truth",
                gripper="official",
                world_xml=wx,
                hold_names=list(hold_names) if hold_names is not None else SC.GENERAL_OBJECT_NAMES,
            )
            self.CK = CK
            self.fast = self.arm
            self.cfg = {}
            self.stop_ev = threading.Event()
            self.motion_allowed = bool(motion_allowed)
            self.home_q = list(self.q)
            self.last_perception = None
            sl = SC.GENERAL_SLOTS if (slots == "default" and world_xml is None) else (None if slots == "default" else slots)
            self._vision = ProbeVision(wx, (assets if assets is not None else SC.CUP_ASSETS), self._pose_wxyz, slots=sl,
                                       noise=noise)
            self.perception_noise = self._vision.noise

        def _pose_wxyz(self, body_name):
            try:
                b = self.d.body(body_name)
            except Exception:
                return None, None
            return [float(v) for v in b.xpos], [float(v) for v in b.xquat]

        def gripper_profile(self):
            return CK.OFFICIAL_PROFILE

        def finger_model(self):
            return G49.SIM_FINGERS

        def current_q(self):
            return list(self.q)

        def move_line(self, label, xyz, quat, speed=0.05, slow=None, strict=False):
            return super().move_line(label, xyz, quat, speed=speed, slow=slow)

        def gripper_clamp(self, force_n, settle_s=2.2):
            res = super().gripper_clamp(force_n, settle_s=settle_s)
            res["gap_mm"] = res.get("opening_mm")
            return res

        def gripper_state(self):
            res = super().gripper_state()
            res["gap_mm"] = res.get("opening_mm")
            return res

        def body_pose(self, body_name):
            b = self.d.body(body_name)
            return [float(v) for v in b.xpos], [float(v) for v in b.xquat]

        def object_position(self, oid):
            return self._vision.position(oid)

        def describe_object(self, oid):
            return self._vision.describe_object(oid)

        def perceive_general(self, tag, prompts=None, use_finetuned=True):
            obs = self._vision.perceive(prompts)
            self.last_perception = obs
            self.emit(f"[V4.9 TWIN GENERAL PERCEIVE] {tag} | count={len(obs['objects'])}")
            return obs

    return TwinGeneralIO


SimGeneralIO = _make_sim_io_class()
TwinGeneralIO = _make_twin_io_class()


_V493_SEEN = {"trips": 0}


def recover_for_task(io, probe=True) -> Tuple[bool, str]:
    """V4.9.3 TASK-START RECOVERY: a failed task must not block the next one unless the simulator is really broken.
    1) STOP+RECOVER handshake when a latch is pending or new safety stops happened since the last task,
    2) release a hold left over, 3) home, 4) small joint probe (J1 +2 deg and back) - one retry after another
    STOP+RECOVER. Returns (ok, why); not ok = the simulator does not follow commands -> restart it."""
    emit = io.emit
    trips = lambda: (io.safety_trips() if hasattr(io, "safety_trips") else 0)

    def _clear(tag):
        if hasattr(io, "clear_latch"):
            try:
                ok, why = io.clear_latch(f"V4.9.3 task start {tag}")
                emit(f"[INFO] [V4.9.3 RECOVER] STOP+RECOVER ({tag}): ok={ok} ({why})")
            except Exception as exc:
                emit(f"[WARN] [V4.9.3 RECOVER] STOP+RECOVER ({tag}) failed: {str(exc)[:160]}")

    try:
        n = int(trips())
    except Exception:
        n = 0
    pending = False
    try:
        pending = bool(io.latch_pending()) if hasattr(io, "latch_pending") else False
    except Exception:
        pass
    if pending or n > _V493_SEEN["trips"]:
        _clear(f"latch_pending={pending}, safety stops {n - _V493_SEEN['trips']} since last task")
    try:
        st = dict(io.gripper_state())
        if st.get("hold_active") or st.get("force_mode"):
            io.gripper_open(60.0)
            emit("[INFO] [V4.9.3 RECOVER] released the hold left over from the last task")
    except Exception:
        pass
    why = "ok"
    for attempt in (1, 2):
        n0 = int(trips() or 0)
        r = io.go_home(f"V4.9.3 task start home (try {attempt})")
        ok = bool(r.get("ok", True))
        if ok and probe and getattr(io, "motion_allowed", True):
            try:
                q0 = list(io.current_q())
                q1 = list(q0)
                q1[0] += 0.035
                r1 = io.move_joints("V4.9.3 start probe", q1)
                r2 = io.move_joints("V4.9.3 start probe back", q0) if r1.get("ok") else r1
                ok = bool(r1.get("ok")) and bool(r2.get("ok"))
                if not ok:
                    why = f"joint probe failed: {(r2 or r1).get('error') or (r2 or r1).get('joint_error_deg')}"
            except Exception as exc:
                ok, why = False, f"joint probe error: {str(exc)[:160]}"
        elif not ok:
            why = f"home failed: {r.get('error')}"
        if ok and int(trips() or 0) == n0:
            _V493_SEEN["trips"] = int(trips() or 0)
            emit(f"[INFO] [V4.9.3 RECOVER] simulator follows commands (try {attempt}) - task can start")
            return True, "ok"
        if ok:
            why = "safety stop during the start probe"
        _clear(f"retry after: {why}")
    _V493_SEEN["trips"] = int(trips() or 0)
    return False, why


def gui_run_sim_task(ctx):
    """V4.8.10: called from NERO_V2_7_1_Unified_Backend.llm_run_task_cmd (# V3.43 A2) when NERO_SCENE=general_pick_place."""
    log = ctx["log"]
    cfg = ctx["cfg_fn"]()
    if not cfg.get("task"):
        log("[LLM UI] Task is empty")
        return
    if ctx["agent"] is None:
        log("[FAIL] [V4.8.10 GENERAL] the LLM agent module is not loaded")
        return
    if not ctx["safety_gate"]():
        log("[FAIL] [V4.8.10 GENERAL] RUN TASK blocked: stale physical safety latch was not recovered")
        return
    dry_run = bool(cfg.get("dry_run", False))
    io = SimGeneralIO(ctx, motion_allowed=not dry_run)
    mode = "plan" if dry_run else "execute"
    log(f"[V4.8.10 GENERAL SIM TASK] scene=general_pick_place | mode={mode} | model={cfg['model']} | task={cfg['task']!r}")

    def _run():
        if not dry_run:
            ok, why = recover_for_task(io)
            if ok:
                try:
                    ctx["start_check"](False)
                except Exception as exc:           # V4.9.3: a blocked start retreat is not fatal - home + probe
                    io.emit(f"[WARN] [V4.9.3 RECOVER] start posture check: {str(exc)[:200]} -> home + probe again")
                    ok, why = recover_for_task(io)
            if not ok:
                io.emit(f"[FAIL] [V4.9.3 GENERAL] simulator state not recoverable ({why}) - restart the simulation")
                return {"ok": False, "reason": "SIM_NOT_RECOVERABLE", "final": why}
        res = run_general_pick_place(io, cfg["task"], mode=mode, agent=ctx["agent"], llm_cfg=cfg)
        if res.get("final"):
            io.emit(f"[V4.8.10 GENERAL RESULT] {res['final']}")
        return res

    ctx["background"]("RUN TASK (general_pick_place)", _run, lock_motion=True)
