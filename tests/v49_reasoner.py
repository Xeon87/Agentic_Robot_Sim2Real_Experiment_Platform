#!/usr/bin/env python3
"""V4.9 test stand-in for the LLM: a generic geometric reasoner that drives the general pick & place tools.

It is NOT part of the product and it is NOT the LLM.  It exists so the tool layer can be tested offline (no API key
in the test machine): it may only use what the tools return (descriptors, previews, plan / execute facts) plus the
structured task (source id, target id, relation "into" / "on") - it never reads the simulator, the scene generator
or object labels.  Its rules are plain geometry (close across the narrower side, pinch a rim when the object has
an open cavity wider than the gripper, keep the fingertips above the table, lower an object into a cavity as deep
as the clearance allows) - the kind of reasoning the LLM is asked to do from the same facts.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class GeometricReasoner:
    def __init__(self, skills, log=print, max_grasp_attempts=3):
        self.sk = skills
        self.log = log
        self.max_grasp_attempts = max_grasp_attempts
        self.calls: List[Tuple[str, Dict[str, Any], bool, Optional[str]]] = []

    def call(self, name, **args):
        r = self.sk.run_tool(name, args)
        self.calls.append((name, args, bool(r.get("ok")), r.get("reason")))
        return r

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _obj(perc, oid):
        for o in perc.get("objects") or []:
            if o["object_id"] == oid:
                return o
        return None

    def _grasp_candidates(self, o, gripper, table_z):
        """generic candidates from the measured geometry, best first"""
        tip = float(gripper["fingertip_below_grasp_point_mm"]) / 1000.0
        min_tip = float(gripper.get("min_fingertip_above_table_mm", 5.0)) / 1000.0 + 0.002
        face_off = float(str(gripper["pad_face_gap_mm"]).split("-")[-1]) if "-" in str(gripper["pad_face_gap_mm"]) else 8.0
        max_open = float(gripper["max_opening_mm"])
        top, bot = float(o["top_z_m"]), float(o["bottom_z_m"])
        a = float(o.get("axis_a_yaw_deg") or 0.0)
        sa, sb = [float(v) for v in o["size_mm"]]
        z_floor = table_z + tip + min_tip                     # lowest grasp point the fingertips allow
        cands = []
        cav = o.get("open_cavity")
        widest_ok = max_open - face_off - 6.0                 # the pad faces must open past the object
        if cav and min(sa, sb) > widest_ok:
            # an open container wider than the gripper: pinch its rim (radially), grasp point between the rim faces
            ow = cav["opening_half_width_mm"]
            r_in = float(np.mean([ow[k] for k in ow]))
            r_out = (sa + sb) / 4.0
            z = max(z_floor, top - 0.020)
            rows = [r for r in (o.get("side_profile") or []) if r.get("z_m") is not None]
            if rows:                                          # the wall where the pads will be
                row = min(rows, key=lambda r: abs(float(r["z_m"]) - z))
                outs = [v for v in row["outer_half_width_mm"].values() if v is not None]
                ins = [v for v in row["inner_half_width_mm"].values() if v is not None]
                if outs:
                    r_out = float(np.mean(outs))
                if ins:
                    r_in = float(np.mean(ins))
            r_mid = (r_in + r_out) / 2.0
            for k in range(12):
                th = a + 30.0 * k
                d = (r_mid * math.cos(math.radians(th)), r_mid * math.sin(math.radians(th)))
                cands.append({"xy_offset_mm": [round(d[0], 1), round(d[1], 1)], "z": z, "yaw": th,
                              "open_mm": min(max_open, 2 * (r_out - r_in) + face_off + 24.0), "tol": 0.0})
            return cands
        # a solid object (or a container narrow enough to grip across): close across a side it fits through
        z_mid = bot + 0.5 * (top - bot)
        z = max(z_floor, min(top - 0.004, z_mid))           # the fingertip limit wins: pads may reach above the top
        sides = sorted([(sb, a + 90.0), (sa, a), (math.hypot(sa, sb) * 0.8, a + 45.0), (math.hypot(sa, sb) * 0.8, a - 45.0)])
        for w, yaw in sides:
            if w + 4.0 > max_open - face_off:               # the pad faces must clear the object by ~2 mm a side
                continue
            cands.append({"xy_offset_mm": [0.0, 0.0], "z": z, "yaw": yaw,
                          "open_mm": min(max_open, w + face_off + 14.0), "tol": 40.0})
        return cands

    def _good_preview(self, pv, target):
        if pv.get("collisions_at_grasp_pose") or pv.get("touches_table"):
            return False, "collision"
        a, b = pv.get("pad_pos_y"), pv.get("pad_neg_y")
        if not (a and b) or a["object"] != target or b["object"] != target:
            return False, pv.get("closes_on")
        if pv.get("first_contact_pushes_object_mm", 99) > 8.0:
            return False, f"push {pv.get('first_contact_pushes_object_mm')}"
        if min(a["contact_band_height_mm"], b["contact_band_height_mm"]) < 5.0:
            return False, "thin contact"
        return True, "ok"

    # ------------------------------------------------------------------ episode
    def run(self, source: str, target: str, relation: str) -> Dict[str, Any]:
        perc = self.call("perceive_objects", prompts=["object"], use_finetuned=True)
        table_z = float(perc["physical_verification"]["table_z_m"])
        gripper = perc["gripper"]
        tallest = max([float(o["top_z_m"]) for o in perc["objects"]] + [table_z])
        held = False
        for attempt in range(self.max_grasp_attempts):
            o = self._obj(perc, source)
            if o is None:
                return {"ok": False, "why": f"{source} not seen"}
            cands = self._grasp_candidates(o, gripper, table_z)
            planned = False
            for c in cands:
                off_z = round((c["z"] - float(o["top_z_m"])) * 1000.0, 1)
                pv = self.call("preview_grasp", object_id=source, xy_offset_mm=c["xy_offset_mm"],
                               grasp_z_offset_from_top_mm=off_z, closing_yaw_deg=c["yaw"], pitch_deg=0.0,
                               open_mm=c["open_mm"])["physical_verification"]
                good, why = self._good_preview(pv, source)
                if not good:
                    continue
                # lift so the held object's bottom clears the tallest other object
                need = (tallest - float(o["bottom_z_m"])) + 0.03
                for lift in sorted({max(0.06, min(0.22, need)), max(0.06, min(0.22, need)) + 0.03, 0.10, 0.08}):
                    r = self.call("plan_grasp_chunk", object_id=source, xy_offset_mm=c["xy_offset_mm"],
                                  grasp_z_offset_from_top_mm=off_z, closing_yaw_deg=c["yaw"], pitch_deg=0.0,
                                  open_mm=c["open_mm"], yaw_tolerance_deg=c["tol"], pregrasp_height_mm=60.0,
                                  lift_height_mm=round(lift * 1000.0))
                    if r.get("ok"):
                        planned = True
                        break
                if planned:
                    break
            if not planned:
                return {"ok": False, "why": "no grasp could be planned"}
            r = self.call("execute_grasp_chunk", clamp_force_n=8.0)
            if r.get("ok"):
                held = True
                break
            if r.get("reason") == "LIFT_FAILED" and self.sk.held:
                held = True
                break
            perc = self.call("perceive_objects", prompts=["object"], use_finetuned=True)
        if not held:
            return {"ok": False, "why": "grasp failed"}
        # ---- place
        perc_t = perc
        t = self._obj(perc_t, target)
        if t is None:
            return {"ok": False, "why": f"{target} not seen"}
        gap = None
        try:
            gap = float(self.sk.io.gripper_state().get("gap_mm"))
        except Exception:
            gap = 30.0
        if relation == "into":
            cav = t.get("open_cavity")
            depth = float(cav["depth_mm"]) if cav else 0.0
            heights = [-(depth - 8.0), -0.75 * depth, -0.5 * depth, -0.25 * depth, 0.0, 10.0, 25.0] if cav else [10.0, 25.0]
        else:
            heights = [3.0, 8.0, 15.0]
        placed = False
        ro = min(70.0, gap + 14.0)
        for hb in heights:
            deep_problem = False
            for cl in (0.0, 40.0, 80.0, 120.0):
                carry_problem = False
                for yc in (0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0):
                    r = self.call("plan_place_chunk", target_id_or_xyz=target, xy_offset_mm=[0.0, 0.0],
                                  held_bottom_above_target_top_mm=round(hb, 1), yaw_change_deg=yc,
                                  approach_height_mm=40.0, release_open_mm=round(ro, 1), carry_lift_mm=cl)
                    if r.get("ok"):
                        placed = True
                        break
                    probs = " ".join(r.get("physical_verification", {}).get("problems", []))
                    if r.get("reason") == "PLACE_COLLISION" and "held_object" in probs:
                        deep_problem = True              # this depth is too deep whatever the yaw / lift
                        break
                    if r.get("reason") == "PLACE_COLLISION" and "opened" in probs and ro > gap + 4.0:
                        ro = max(gap + 4.0, ro - 6.0)    # open less at release
                    if r.get("reason") == "CARRY_COLLISION":
                        carry_problem = True
                        break                            # lift more first
                    if "lift" in probs and "straight up" in probs:
                        break
                if placed or deep_problem:
                    break
                if not carry_problem and cl > 0.0:
                    break
            if placed:
                break
        if not placed:
            self.call("verify_and_park_chunk", action="return_home", prompts=["object"])
            return {"ok": False, "why": "no place could be planned"}
        r = self.call("execute_place_chunk", retreat_up_mm=60.0)
        if not r.get("ok"):
            return {"ok": False, "why": f"place failed: {r.get('reason')}"}
        v = self.call("verify_and_park_chunk", action="clear_view_and_perceive", prompts=["object"])
        rel = v.get("physical_verification", {}).get("relations", [])
        want = {"into": "inside", "on": "on"}[relation]
        seen = any(x["object"] == source and x["of"] == target and x["relation"] == want for x in rel)
        return {"ok": True, "verified_relation": seen, "relations": rel}
