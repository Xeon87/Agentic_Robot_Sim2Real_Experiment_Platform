#!/usr/bin/env python3
"""NERO V3.72 SORT ITEMS AGENT - simulation scene sort_items (SO-101 gripper on NERO).

V3.72 SORT ITEMS AGENT

Scene (NERO_V343_Scenes, V3.72 SORT ITEMS SCENE): a banana, an apple, a computer mouse and a thick marker pen lie on the
table; an open RED bin stands on the robot's left (y < 0), an open BLUE bin on its right (y > 0).  The task prompt says
how to sort (e.g. "put the fruit into the red bin and everything else into the blue bin"): the LLM agent reads the item
labels from observe_scene, decides each item's category from the prompt and calls the skills.

Skills (one physical tool at a time, like the cup skills; motion helpers, joint-path checks, safety latch handling
and the SO-101 gripper profile are the V3.62-V3.64 CupSkills ones - this class only adds the sorting tools):
  observe_scene            items (id, detector label, centre, size, long-axis yaw, bin it is in, held) + bins + gripper
  pick_object(item_id)     top-down grasp across the item's width (jaws closing perpendicular to its long axis; any
                           direction for round items), fixed jaw first with a small clearance, straight descent, clamp,
                           straight lift; verified by contact / hold and by the item rising with the gripper
  place_in_box(box)        rotate in place so the long axis lies along the bin's y side, carry above a free slot of the
                           bin, straight down until the item is 15 mm above the bin floor, open, straight retreat;
                           verified by the item being inside that bin
  check_bins               read-only: which item is in which bin
  return_to_safe_pose      open + home (refused while an item is held)
Pose source: simulator ground truth (TF of the item bodies) - item labels = the scene's object names, as a detector
would return them.  The fixed-camera / YOLOE path is not built for this scene yet (RUN TASK refuses the camera source).

Backend: only the RUN TASK `# V3.43 A2` hook (next to the stack_cups one) calls gui_run_task(ctx).
Offline: `python3 NERO_V372_Sort_Skills.py --twin` runs the scripted episode (fruit -> red, others -> blue) in the
headless twin (the V3.62 TwinIO with the SO-101 parts + the backend gripper logic, bins + items instead of cups).
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np                           # noqa: E402
import NERO_V343_Scenes as SC                # noqa: E402
import NERO_V362_Cup_Skills as CK            # noqa: E402

VERSION = "V3.73"
MARKER = "V3.72 SORT ITEMS AGENT"
SCENE = "sort_items"
ITEM_IDS = {f"item_{i + 1}": n for i, n in enumerate(SC.SORT_ITEM_NAMES)}
BODY_ID = {v: k for k, v in ITEM_IDS.items()}
BOXES = tuple(SC.SORT_BINS)                  # ("red", "blue")

PRE_M = 0.08                 # pregrasp: grasp point + 80 mm up
LIFT_M = 0.13                # lift: grasp TCP + 130 mm (item bottom well above the 60 mm bin walls)
# fixed pad face -> item side during the descent.  The backend gripper pushes with <= 0.6 N until BOTH pads touch: that
# does not slide a 90-180 g item (mouse / banana / apple) onto the fixed pad, so the fixed pad descends touching it
# (0.3 mm off the surface, inside the 0.5 mm contact band) and the moving jaw closes the pinch
FIXED_PAD_GAP_M = 0.0003
OPEN_MARGIN_MM = 14.0        # jaw tip gap = item width + clearance + this
TIP_Z_MIN_M = CK.TIP_BEYOND_GRASP_M + 0.004   # grasp height: SO-101 tips stay >= 4 mm above the table
CLAMP_N = 5.0
DESCENT_SPEED = 0.03
LIFT_SPEED = 0.05
LOWER_SPEED = 0.02
MIN_RISE_M = 0.03            # the item must rise at least this with the gripper
RELEASE_ABOVE_FLOOR_M = 0.015
ABOVE_WALL_M = 0.035         # item bottom this far above the bin walls while carried in
SLOT_DX_M = 0.045            # slots: bin centre -+ 45 mm along the closing direction (items lie side by side)
JAW_CLEAR_M = 0.008          # open jaws keep this from every other item (candidate filter)
PREFERRED_MARGIN_DEG = 8.0


def tf_name(body):
    """V3.73 TF NAME: MuJoCo body "obj_banana" -> the name the backend's object_pose() expects ("banana")"""
    return body[4:] if body.startswith("obj_") else body


def grasp_R(phi):
    """grasp frame for a top-down grasp: z = down (approach), y = closing direction (fixed -> moving jaw) at yaw phi"""
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([math.cos(phi), math.sin(phi), 0.0])
    return np.column_stack([np.cross(y, z), y, z])


def yaw_of_wxyz(q):
    w, x, y, z = [float(v) for v in q]
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


# =============================================================================================== IO adapters
class SortTwinIO(CK.TwinIO):
    """the V3.62 headless twin (SO-101 parts, backend gripper / CONNECT-hold logic, URDF arm kinematics) with the
    sort_items bins + items"""

    def __init__(self, log=None, seed=None, poses=None):
        world = SC.sort_world_xml(seed, bare=True, poses=poses)
        super().__init__(log=log, gripper="so101", world_xml=world, hold_names=SC.SORT_ITEM_NAMES)

    def object_pose(self, body):
        b = self.d.body(body)
        return [float(v) for v in b.xpos], [float(v) for v in b.xquat]


class SortBackendIO(CK.BackendIO):
    """the V3.62 backend adapter + item poses from TF (sim_obj_*) + the SO-101 profile"""

    def object_pose(self, body):
        # V3.73 TF NAME: the backend's object_pose(name) looks up TF "sim_obj_" + name and the scene publishes
        # "sim_" + body ("sim_obj_banana") -> pass the name WITHOUT "obj_" (V3.72 sent "obj_banana" ->
        # "sim_obj_obj_banana" does not exist: observe_scene failed in the first simulation run)
        p, q = self.core.object_pose(tf_name(body))
        return [float(v) for v in p], [float(q[3]), float(q[0]), float(q[1]), float(q[2])]

    def gripper_profile(self):
        return CK.so101_profile()

    def pose_mode(self):
        return "truth"


# =============================================================================================== skills
class SortSkills(CK.CupSkills):
    STAGES = ("IDLE", "OBSERVED", "HOLDING", "PLACED", "DONE")

    def __init__(self, io, skill=None):
        # V3.73: the cup knobs are not used here; a release gap at the SO-101 cap keeps the cup message
        # "[V3.64 CUP] so101: release gap capped ..." out of the sorting log
        cap = getattr(io, "gripper_profile", lambda: CK.OFFICIAL_PROFILE)().release_gap_max_mm
        knobs = dict(CK.CUP_KNOB_DEFAULTS, **({"cup_release_gap_mm": float(cap)} if cap is not None else {}))
        super().__init__(io, knobs, skill or {"code": "SI", "short_name": "Sort Items", "skill_id": "sort_items_top_grasp"})
        self.skill = skill or {"code": "SI", "short_name": "Sort Items", "skill_id": "sort_items_top_grasp"}
        self.held = None                 # item id
        self.held_off = None             # item centre in the TCP frame (measured after the lift)
        self.placed = {}                 # item id -> requested bin
        self.picked_from = {}

    # -- helpers
    def _res(self, tool, ok, verification=None, reason=None, nxt=None, warnings=None, **extra):
        v = dict(verification or {})
        if reason:
            v["reason"] = reason
        out = {"ok": bool(ok), "tool": tool, "state_after": {"stage": self.stage, "held_item": self.held},
               "physical_verification": v, "warnings": list(warnings or []), "recommended_next_actions": list(nxt or [])}
        if reason:
            out["reason"] = reason
        out.update(extra)
        self.io.emit(f"[{'OK' if ok else 'FAIL'}] [V3.72 SORT] {tool}: " + (reason or "done") +
                     (f" | {json.dumps(v, ensure_ascii=False)[:300]}" if v else ""))
        return out

    def _item(self, iid):
        body = ITEM_IDS[iid]
        p, q = self.io.object_pose(body)
        spec = SC.SORT_ITEMS[body]
        c = SC.sort_item_centre(body, p, q)          # V3.72: body origin = top centre
        return {"id": iid, "body": body, "spec": spec, "c": np.asarray(c, float), "yaw_deg": yaw_of_wxyz(q),
                "in_bin": SC.item_in_bin(c), "up_z": round(1.0 - 2.0 * (float(q[1]) ** 2 + float(q[2]) ** 2), 3)}

    def _items(self):
        return {iid: self._item(iid) for iid in ITEM_IDS}

    def scene(self):
        its = self._items()
        g = self.io.gripper_state()
        tcp, _q = self.io.tcp_pose()
        items = []
        for iid, it in its.items():
            sp = it["spec"]
            items.append({"id": iid, "label": sp["label"], "centre_xyz_m": [round(float(v), 4) for v in it["c"]],
                          "size_mm": {"length": round(sp["length"] * 1000), "width": round(sp["width"] * 1000),
                                      "height": round(sp["height"] * 1000)},
                          "long_axis_yaw_deg": None if sp["shape"] == "round" else round(_wrap(it["yaw_deg"]), 1),
                          "in_bin": it["in_bin"], "on_table": bool(it["in_bin"] is None and it["c"][2] < sp["cz"] + 0.02),
                          "held": self.held == iid})
        bins = []
        for col, b in SC.SORT_BINS.items():
            bins.append({"box": col, "colour": col, "side": b["side"], "centre_xy_m": list(b["centre"]),
                         "inner_size_mm": [round(v * 1000) for v in b["inner"]], "wall_height_mm": round(b["wall_h"] * 1000),
                         "open_top": True, "contains": [i["id"] for i in items if i["in_bin"] == col]})
        return {"scene": SCENE, "pose_source": "simulator ground truth (TF); labels = the scene's object names",
                "items": items, "boxes": bins, "held_item": self.held, "tcp_xyz_m": [round(v, 4) for v in tcp],
                "gripper": g}

    def observe_scene(self, **_):
        sc = self.scene()
        if self.stage == "IDLE":
            self.stage = "OBSERVED"
        nxt = ["place_in_box"] if self.held else ["pick_object"]
        return self._res("observe_scene", True, {"items": len(sc["items"])}, nxt=nxt, scene=sc)

    # -- grasp planning
    def _jaw_clear(self, it, p_a, yG, gap_m, others):
        """smallest distance of the two open jaw tips to any other item (centre segment minus half width)"""
        tips = [p_a[:2], (p_a + gap_m * yG)[:2]]
        best = 1e9
        for o in others:
            sp = o["spec"]
            a = math.radians(o["yaw_deg"])
            half = 0.0 if sp["shape"] == "round" else max(0.0, sp["length"] / 2.0 - sp["width"] / 2.0)
            u = np.array([math.cos(a), math.sin(a)])
            c = o["c"][:2]
            for t in tips:
                s = float(np.clip((t - c) @ u, -half, half))
                best = min(best, float(np.linalg.norm(t - (c + s * u))) - sp["width"] / 2.0)
        return best

    def _plan_pick(self, it, q_now):
        sp = it["spec"]
        width = float(sp["width"])
        gap_m = (width * 1000.0 + OPEN_MARGIN_MM) / 1000.0
        if sp["shape"] == "round":
            phis = list(range(0, 360, 30)) + list(range(15, 360, 30))
        else:
            base = it["yaw_deg"] + 90.0
            phis = [base + d for d in (0, 180, 15, -15, 195, 165, 30, -30, 210, 150, 45, -45, 225, 135)]
        others = [o for k, o in self._items().items() if k != it["id"] and o["in_bin"] is None]
        zg = float(it["c"][2])
        if sp.get("grasp_z") is not None:                # grasp height above the item bottom (flat sides)
            zg = float(it["c"][2]) - float(sp["cz"]) + float(sp["grasp_z"])
        zg = max(zg, TIP_Z_MIN_M)
        # half width at the grasp height (round cross-sections - banana / apple / marker - are narrower off-centre)
        dz = zg - float(it["c"][2])
        half = math.sqrt(max((width / 2.0) ** 2 - dz * dz, (width / 4.0) ** 2))
        import NERO_V364_SO101_Gripper as G
        cands = []
        for ph in phis:
            phi = math.radians(ph)
            R = grasp_R(phi)
            yG = R[:, 1]
            # grasp frame origin = fixed pad face + WALL_HALF; put the pad face FIXED_PAD_GAP off the item side
            p_a = np.array([it["c"][0], it["c"][1], zg]) - (half + FIXED_PAD_GAP_M - G.WALL_HALF) * yG
            clear = self._jaw_clear(it, p_a, yG, gap_m, others)
            if clear < JAW_CLEAR_M:
                continue
            pts = {"pregrasp": p_a + [0, 0, PRE_M], "grasp": p_a, "lift": p_a + [0, 0, LIFT_M]}
            tcp = {k: self.prof.tcp_from_grasp(v, R) for k, v in pts.items()}
            chain = ([("pregrasp", *tcp["pregrasp"])] + CK.segment_points("d", tcp["pregrasp"][0], tcp["grasp"][0],
                                                                          tcp["grasp"][1], 6)
                     + [("grasp", *tcp["grasp"])] + CK.segment_points("u", tcp["grasp"][0], tcp["lift"][0],
                                                                      tcp["lift"][1], 6) + [("lift", *tcp["lift"])])
            r = CK.ik_chain(chain, q_now)
            if not r["ok"] or r["min_margin_deg"] < CK.MIN_MARGIN_RELAXED_DEG:
                r = CK.ik_chain(chain, q_now, start_first=False)       # another IK branch of the 7-DoF arm
            if not r["ok"] or r["min_margin_deg"] < CK.MIN_MARGIN_RELAXED_DEG:
                continue
            names = [c[0] for c in chain]
            q = dict(zip(names, r["q"]))
            i_g, i_l = names.index("grasp"), names.index("lift")
            cands.append({"phi_deg": round(_wrap(ph), 1), "margin": r["min_margin_deg"], "binding": r["binding"],
                          "jaw_clear_mm": round(clear * 1000, 1), "tcp": tcp, "q": q,
                          "q_down": r["q"][:i_g + 1], "q_up": r["q"][i_g:i_l + 1], "gap_mm": gap_m * 1000.0,
                          "zg": zg})
            if r["min_margin_deg"] >= PREFERRED_MARGIN_DEG:
                break
        if not cands:
            return None
        return max(cands, key=lambda c: c["margin"])

    def pick_object(self, item_id="item_1", **_):
        tool = "pick_object"
        if item_id not in ITEM_IDS:
            return self._res(tool, False, reason="UNKNOWN_ITEM", nxt=["observe_scene"])
        if self.held:
            return self._res(tool, False, reason="ALREADY_HOLDING", nxt=["place_in_box"])
        it = self._item(item_id)
        v = {"item": item_id, "label": it["spec"]["label"]}
        if it["in_bin"] is not None:
            v["note"] = f"the item is in the {it['in_bin']} bin; picking it out again"
        try:
            q_now = self.io.current_q()
        except Exception:
            q_now = None
        plan = self._plan_pick(it, q_now)
        if plan is None:
            return self._res(tool, False, v, reason="GRASP_UNREACHABLE", nxt=["observe_scene", "stop_and_report"])
        v.update(closing_yaw_deg=plan["phi_deg"], min_joint_margin_deg=plan["margin"], binding=plan["binding"],
                 jaw_clear_mm=plan["jaw_clear_mm"])
        import NERO_V364_SO101_Gripper as G
        z0 = float(it["c"][2])
        try:
            self.io.gripper_open(G.opening_mm_for_tip_gap(plan["gap_mm"]))
            self._go_joints(f"above {item_id} ({it['spec']['label']})", plan["q"]["pregrasp"])
            p, R = plan["tcp"]["grasp"]
            self._go_line(f"down to {item_id}", p, R, DESCENT_SPEED, q_hint=plan["q"]["grasp"], q_path=plan["q_down"])
            g = self.io.gripper_clamp(CLAMP_N)
            v["clamp"] = {k: g.get(k) for k in ("contact_left", "contact_right", "hold_active", "opening_mm")}
            p, R = plan["tcp"]["lift"]
            self.held = item_id                     # dip check of the carry fallback while lifting
            self._go_line(f"lift {item_id}", p, R, LIFT_SPEED, q_hint=plan["q"]["lift"], q_path=plan["q_up"])
        except self._Abort as e:
            self.held = None
            return self._abort_result(tool, e, v)
        it2 = self._item(item_id)
        rise = float(it2["c"][2]) - z0
        st = self.io.gripper_state()
        v["item_rise_mm"] = round(rise * 1000.0, 1)
        held = rise >= MIN_RISE_M and (st.get("hold_active") or (st.get("contact_left") and st.get("contact_right")))
        if not held:
            self.held = None
            self.io.gripper_open(G.opening_mm_for_tip_gap(plan["gap_mm"]))
            self._retreat_up(0.05)
            reason = "ITEM_NOT_LIFTED" if rise < MIN_RISE_M else "NO_TWO_FINGER_CONTACT"
            return self._res(tool, False, v, reason=reason, nxt=["observe_scene", "pick_object"])
        tcp, qt = self.io.tcp_pose()
        self.held_off = CK.quat_xyzw_to_R(qt).T @ (it2["c"] - np.asarray(tcp, float))
        self.stage = "HOLDING"
        self.picked_from[item_id] = it["in_bin"]
        return self._res(tool, True, v, nxt=["place_in_box"])

    # -- place
    def _slot(self, box, iid, phi):
        """the freer of the two slots centre -+ SLOT_DX along the closing direction phi (the long axis is
        perpendicular to it, so two long items lie side by side); an empty bin -> the slot nearer the robot"""
        b = SC.SORT_BINS[box]
        cx, cy = b["centre"]
        inside = [o for k, o in self._items().items() if k != iid and o["in_bin"] == box]
        u = np.array([math.cos(phi), math.sin(phi)])
        slots = [np.array([cx, cy]) + SLOT_DX_M * u, np.array([cx, cy]) - SLOT_DX_M * u]
        if not inside:
            return min(slots, key=lambda s_: float(np.hypot(*s_)))
        return max(slots, key=lambda s_: min(float(np.hypot(*(s_ - o["c"][:2]))) for o in inside))

    def place_in_box(self, box="red", **_):
        tool = "place_in_box"
        if box not in BOXES:
            return self._res(tool, False, reason="UNKNOWN_BOX", nxt=["observe_scene"])
        if not self.held or self.held_off is None:
            return self._res(tool, False, reason="NO_ITEM_HELD", nxt=["observe_scene", "pick_object"])
        iid = self.held
        it = self._item(iid)
        sp = it["spec"]
        b = SC.SORT_BINS[box]
        v = {"item": iid, "label": sp["label"], "box": box}
        c_rel = sp["cz"] + b["floor_t"] + RELEASE_ABOVE_FLOOR_M
        c_above = b["wall_h"] + ABOVE_WALL_M + sp["cz"]
        q_now = self.io.current_q()
        tcp0, _q0 = self.io.tcp_pose()
        phis = list(range(0, 360, 30))
        best = None
        for rotate_first in (True, False):
            for ph in phis:
                # the long axis is perpendicular to the closing direction phi; the slot is offset along phi
                R = grasp_R(math.radians(ph))
                Rt = R @ self.prof.T_grasp_tcp[:3, :3]
                off = Rt @ self.held_off
                sx, sy = self._slot(box, iid, math.radians(ph))
                p_above = np.array([sx, sy, max(c_above, float(it["c"][2]))]) - off
                p_rel = np.array([sx, sy, c_rel]) - off
                p_rot = np.array([tcp0[0], tcp0[1], max(tcp0[2], p_above[2])])
                chain = (([("rotate", p_rot, Rt)] if rotate_first else []) + [("above", p_above, Rt)]
                         + CK.segment_points("l", p_above, p_rel, Rt, 6) + [("release", p_rel, Rt)])
                r = CK.ik_chain(chain, q_now)
                if (not r["ok"] or r["min_margin_deg"] < CK.MIN_MARGIN_RELAXED_DEG) and not rotate_first:
                    r = CK.ik_chain(chain, q_now, start_first=False)
                if not r["ok"] or r["min_margin_deg"] < CK.MIN_MARGIN_RELAXED_DEG:
                    continue
                names = [c[0] for c in chain]
                cand = {"phi": ph, "margin": r["min_margin_deg"], "binding": r["binding"], "q": dict(zip(names, r["q"])),
                        "q_down": r["q"][names.index("above"):], "p_rot": p_rot, "p_above": p_above, "p_rel": p_rel,
                        "Rt": Rt, "rotate_first": rotate_first, "slot": [round(float(sx), 3), round(float(sy), 3)]}
                if best is None or cand["margin"] > best["margin"]:
                    best = cand
                if cand["margin"] >= PREFERRED_MARGIN_DEG:
                    break
            if best is not None:
                break
        if best is None:
            return self._res(tool, False, v, reason="PLACE_UNREACHABLE", nxt=["place_in_box", "stop_and_report"])
        v.update(min_joint_margin_deg=best["margin"], binding=best["binding"], closing_yaw_deg=best["phi"],
                 slot_xy_m=best["slot"], rotate_in_place=best["rotate_first"])
        import NERO_V364_SO101_Gripper as G
        try:
            if np.linalg.norm(np.asarray(tcp0) - best["p_rot"]) > 0.003:
                self._go_line(f"up with {iid}", best["p_rot"], CK.quat_xyzw_to_R(_q0), LIFT_SPEED)
            if best["rotate_first"]:
                self._go_joints(f"rotate {iid} for the {box} bin", best["q"]["rotate"], slow=2.0)
            self._go_line(f"carry {iid} over the {box} bin", best["p_above"], best["Rt"], CK.CARRY_SPEED,
                          slow=1.5, free_fallback=True, q_hint=best["q"]["above"])
            self.io.sleep(0.5)
            self._go_line(f"lower {iid} into the {box} bin", best["p_rel"], best["Rt"], LOWER_SPEED,
                          q_hint=best["q"]["release"], q_path=best["q_down"])
        except self._Abort as e:
            return self._abort_result(tool, e, v)
        self.io.gripper_open(G.opening_mm_for_tip_gap(sp["width"] * 1000.0 + 18.0))
        self.io.sleep(0.8)
        self.held, self.held_off = None, None
        self.placed[iid] = box
        try:
            p, q = self.io.tcp_pose()
            self._go_line(f"retreat from the {box} bin", [p[0], p[1], p[2] + 0.10], CK.quat_xyzw_to_R(q), LIFT_SPEED)
        except self._Abort as e:
            v["retreat"] = f"{e.reason}: {e.info.get('error') or e.info.get('detail')}"
        self.io.sleep(0.5)
        where = self._item(iid)["in_bin"]
        v["now_in_bin"] = where
        self.stage = "PLACED"
        if where != box:
            return self._res(tool, False, v, reason="ITEM_NOT_IN_BOX", nxt=["observe_scene", "pick_object"])
        return self._res(tool, True, v, nxt=["observe_scene", "pick_object", "check_bins"])

    def check_bins(self, **_):
        its = self._items()
        rows = {iid: {"label": it["spec"]["label"], "in_bin": it["in_bin"]} for iid, it in its.items()}
        on_table = [k for k, r in rows.items() if r["in_bin"] is None]
        return self._res("check_bins", True, {"items": rows, "on_table": on_table,
                                              "red": [k for k, r in rows.items() if r["in_bin"] == "red"],
                                              "blue": [k for k, r in rows.items() if r["in_bin"] == "blue"]},
                         nxt=["finish_and_report"] if not on_table else ["pick_object"])

    def return_to_safe_pose(self, **_):
        if self.held:
            return self._res("return_to_safe_pose", False, reason="ITEM_HELD_PLACE_FIRST", nxt=["place_in_box"])
        if self.io.latch_pending():
            ok, why = self.io.clear_latch("V3.72 return_to_safe_pose")
            if not ok:
                return self._res("return_to_safe_pose", False, {"detail": why}, reason="SAFETY_LATCH_NOT_CLEARED",
                                 nxt=["stop_and_report"])
        self.io.gripper_open(self.prof.open_mm)
        r = self.io.go_home("V3.72 sort task")
        return self._res("return_to_safe_pose", bool(r.get("ok", True)), {"detail": r.get("error")})

    def _run_tool(self, name, args):
        fn = {"observe_scene": self.observe_scene, "pick_object": self.pick_object, "place_in_box": self.place_in_box,
              "check_bins": self.check_bins, "return_to_safe_pose": self.return_to_safe_pose}.get(name)
        if fn is None:
            return {"ok": False, "tool": name, "reason": "UNKNOWN_TOOL", "recommended_next_actions": ["observe_scene"]}
        try:
            return fn(**(args or {}))
        except Exception as exc:
            return {"ok": False, "tool": name, "reason": "SKILL_EXCEPTION", "error": str(exc)[:400],
                    "recommended_next_actions": ["observe_scene", "stop_and_report"]}

    def scripted_episode(self, rule=None):
        """without an LLM: fruit -> red, everything else -> blue (or rule {label: box}) -> results"""
        out = [self.run_tool("observe_scene", {})]
        for iid, body in ITEM_IDS.items():
            sp = SC.SORT_ITEMS[body]
            box = (rule or {}).get(sp["label"]) or ("red" if sp["category"] == "fruit" else "blue")
            r = self.run_tool("pick_object", {"item_id": iid})
            out.append(r)
            if not r.get("ok"):
                continue
            out.append(self.run_tool("place_in_box", {"box": box}))
        out.append(self.run_tool("check_bins", {}))
        return out


# =============================================================================================== LLM contract
_IDS = list(ITEM_IDS)
SORT_TOOLS = [
    {"type": "function", "name": "observe_scene",
     "description": "Read the sort_items scene: every item (id, detector label, centre, size, long-axis yaw, which bin it "
                    "is in, held), the two open bins (red / blue, side, contents), gripper, TCP. Observation only. "
                    "ALWAYS call first.",
     "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "pick_object",
     "description": "Grasp one item from above with the SO-101 gripper (jaws across its width), lift it. Plans the grasp "
                    "itself (reachable closing direction, joint margins, open jaws clear of the other items). Verifies "
                    "contact and that the item rose with the gripper.",
     "parameters": {"type": "object", "properties": {"item_id": {"type": "string", "enum": _IDS}},
                    "required": ["item_id"], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "place_in_box",
     "description": "Carry the held item into the open bin (red or blue), lower it to just above the bin floor, open, "
                    "retreat up. Verifies the item lies inside that bin.",
     "parameters": {"type": "object", "properties": {"box": {"type": "string", "enum": list(BOXES)}},
                    "required": ["box"], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "check_bins",
     "description": "Read-only final check: which item is in which bin, which items are still on the table.",
     "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "return_to_safe_pose",
     "description": "Open the empty gripper and return the arm to the safe neutral posture. Refused while an item is held.",
     "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}, "strict": True},
]

SORT_SYSTEM_INSTRUCTIONS = """You are NERO V3.72, the high-level task planner for a MuJoCo simulated 7-DOF NERO robot arm with an
SO-101 two-finger gripper (one fixed jaw, one moving jaw). Scene sort_items: several items lie on the table (each has an
id like item_1 and a detector label such as "banana"); an open RED bin stands on the robot's left and an open BLUE bin on
its right. The user's task says how the items are to be sorted into the bins (for example by category: fruit vs.
everything else). You control the robot ONLY through the function tools. Never invent coordinates or joint angles.

Procedure:
1. Call observe_scene first. From each item's label decide its category and its target bin according to the user's
   task. State your assignment (item id, label, category, bin) in one short line of text before the first pick.
2. For every item that is not yet in its target bin: pick_object(item_id) -> place_in_box(box). One physical tool at a
   time; after every tool read ok, reason, physical_verification and recommended_next_actions.
3. Items the task does not mention, or whose category is unclear: leave them on the table and say so (do not guess).
4. If pick_object fails (ITEM_NOT_LIFTED, NO_TWO_FINGER_CONTACT, MOTION_FAILED), the gripper is empty: call
   observe_scene and retry that item ONCE. If place_in_box fails with ITEM_NOT_IN_BOX the item fell outside: call
   observe_scene, pick it again and place it ONCE more. Never loop.
5. GRASP_UNREACHABLE / PLACE_UNREACHABLE / TARGET_UNREACHABLE / SAFETY_LATCH_NOT_CLEARED: skip that item and report it.
   SAFETY_STOP with latch_recovered=true: observe_scene, then the same tool ONCE more.
6. Finish with check_bins, then return_to_safe_pose. Success = every item the task assigns is in its bin per check_bins.
7. Keep the final message short: the assignment you used, what check_bins shows, any item left and why.
"""


def _score(history, elapsed_s, rounds, skill, task):
    """V3.73 SCORE BY THE AGENT'S ITEMS: s = items the agent put into a bin AND still there (last check_bins) /
    items the agent decided to move (every item it called pick_object for).  V3.72 divided by all 4 items, so a task
    that sorts only the fruit could never reach 1.0."""
    placed, tried = {}, []
    for h in history:
        r = h["result"] if isinstance(h["result"], dict) else {}
        if h["tool"] == "pick_object":
            iid = (r.get("physical_verification") or {}).get("item") or (h.get("args") or {}).get("item_id")
            if iid in ITEM_IDS and iid not in tried:
                tried.append(iid)
        if h["tool"] == "place_in_box" and r.get("ok"):
            pv = r.get("physical_verification") or {}
            placed[pv.get("item")] = pv.get("box")
    last = None
    for h in history:
        if h["tool"] == "check_bins" and isinstance(h["result"], dict) and h["result"].get("ok"):
            last = (h["result"].get("physical_verification") or {}).get("items")
    good = sum(1 for k, b in placed.items() if last is None or (last.get(k) or {}).get("in_bin") == b)
    total = len(tried)
    s = good / float(total) if total else 0.0
    t_min = max(0.0, float(elapsed_s)) / 60.0
    z = int(rounds)
    y = s / (t_min * z) * 100.0 if t_min > 0 and z > 0 else 0.0
    line = (f"[V3.72 SKILL SCORE] skill={skill.get('code')} {skill.get('short_name')} | scene=sort_items | "
            f"s={s:.2f} ({good}/{total} items the agent moved are in the bin it chose) | t={t_min:.2f} min | "
            f"z={z} LLM rounds | Y = s/(t*z)*100 = {y:.3f}")
    return {"skill_code": skill.get("code"), "skill_name": skill.get("short_name"), "skill_id": skill.get("skill_id"),
            "s": s, "done": good, "total": total, "basis": "items the agent moved, in the chosen bin",
            "t_min": round(t_min, 4), "z": z, "Y": round(y, 4),
            "status": "SUCCESS" if total and good == total else ("PARTIAL" if good else "FAILED"),
            "line": line}


def final_table(skills):
    """item -> label, simulator category (ground truth, for the log only) -> bin now"""
    rows = []
    for iid, it in skills._items().items():
        rows.append(f"{iid} {it['spec']['label']} ({it['spec']['category']}) -> {it['in_bin'] or 'table'}")
    return " · ".join(rows)


def run_sort_task(agent, skills, task, model, api_key="", reasoning_effort="low", max_tool_steps=40, emit=None,
                  dry_run=False):
    """bounded tool-calling episode for sort_items (NeroLLMAgent transport, like run_cup_task) -> final text"""
    emit = emit or agent.emit
    with agent._lock:
        if agent._running:
            raise RuntimeError("LLM agent is already running")
        agent._running = True
    agent._stop_event.clear()
    agent._last_response_id = None
    t0 = time.perf_counter()
    rounds, steps, history, final = 0, 0, [], ""
    try:
        ok, key_status = agent.api_key_status(api_key, model)
        if not ok:
            raise RuntimeError(key_status)
        emit(f"[V3.72 SORT TASK START] model={model} | effort={reasoning_effort} | max_tool_steps={max_tool_steps} | "
             f"skill={skills.skill.get('code')} | gripper {skills.prof.label}")
        emit(f"[LLM USER TASK] {task}")
        payload = {"model": str(model), "instructions": SORT_SYSTEM_INSTRUCTIONS, "input": str(task), "tools": SORT_TOOLS,
                   "tool_choice": "auto", "parallel_tool_calls": False, "reasoning": {"effort": str(reasoning_effort)},
                   "max_output_tokens": 1500, "store": True}
        while True:
            if agent._stop_event.is_set():
                raise RuntimeError("LLM task stopped by user")
            rounds += 1
            emit(f"[LLM REQUEST #{rounds}]")
            t_req = time.perf_counter()
            resp = agent._post_responses(payload, api_key)
            agent._log_usage(resp, model)
            agent._last_response_id = resp.get("id")
            calls = agent._function_calls(resp)
            text = agent._extract_text(resp)
            emit(f"[LLM RESPONSE #{rounds}] elapsed={time.perf_counter() - t_req:.2f}s | tool_calls={len(calls)}")
            if text:
                emit(f"[LLM TEXT] {text}")
            if not calls:
                final = text
                break
            outs = []
            for call in calls:
                if agent._stop_event.is_set():
                    raise RuntimeError("LLM task stopped by user")
                steps += 1
                name = str(call.get("name", ""))
                if steps > max_tool_steps:
                    raise RuntimeError(f"tool step budget {max_tool_steps} used up")
                try:
                    args = json.loads(call.get("arguments") or "{}")
                except Exception:
                    args = {}
                emit(f"[LLM CALL #{steps}] {name}(" + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")")
                t_tool = time.perf_counter()
                if dry_run and name not in ("observe_scene", "check_bins"):
                    res = {"ok": True, "dry_run": True, "tool": name, "arguments": args}
                    emit(f"[LLM DRY RUN] skipped physical tool {name}")
                else:
                    res = skills.run_tool(name, args)
                emit(f"[LLM TOOL RESULT #{steps}] {name} | elapsed={time.perf_counter() - t_tool:.2f}s | "
                     + json.dumps({k: v for k, v in res.items() if k != "scene"}, ensure_ascii=False)[:1500])
                history.append({"tool": name, "args": args, "result": res})
                outs.append({"type": "function_call_output", "call_id": str(call.get("call_id", "")),
                             "output": json.dumps(res, ensure_ascii=False, default=str)})
            payload = {"model": str(model), "instructions": SORT_SYSTEM_INSTRUCTIONS,
                       "previous_response_id": agent._last_response_id, "input": outs, "tools": SORT_TOOLS,
                       "tool_choice": "auto", "parallel_tool_calls": False,
                       "reasoning": {"effort": str(reasoning_effort)}, "max_output_tokens": 1500, "store": True}
        emit(f"[LLM FINAL] {final}")
        return final
    except Exception as exc:
        emit(f"[LLM TASK FAILED] {exc}")
        raise
    finally:
        sc = _score(history, time.perf_counter() - t0, rounds, skills.skill, task)
        try:
            emit(f"[V3.72 SORT RESULT] {final_table(skills)}")
        except Exception:
            pass
        emit(f"[TASK SUMMARY {'GREEN' if sc['status'] == 'SUCCESS' else ('YELLOW' if sc['status'] == 'PARTIAL' else 'RED')}] "
             f"STATUS: {sc['status']} | TASK: {task} | tools {steps} | LLM rounds {rounds} | {time.perf_counter() - t0:.1f} s")
        emit(sc["line"])
        try:
            import NERO_V2_7_1_Agent as AG
            AG._v360_append_score(sc, task, model)
        except Exception:
            pass
        with agent._lock:
            agent._running = False


# =============================================================================================== backend entry (A2 block)
def gui_run_task(ctx):
    """called from NERO_V2_7_1_Unified_Backend.llm_run_task_cmd (its RUN TASK `# V3.43 A2` hook) when NERO_SCENE=sort_items"""
    log = ctx["log"]
    mb = ctx.get("messagebox")
    cfg = ctx["cfg_fn"]()
    if not cfg.get("task"):
        log("[LLM UI] Task is empty")
        return
    if ctx.get("pose_source_mode", lambda: "truth")() == "camera":
        log("[FAIL] [V3.72 SORT] sort_items: camera item recognition is not built for this scene yet - select "
            "'JSON ground truth' as the pose source")
        if mb:
            mb.showerror("NERO V3.72", "分类场景目前只支持仿真真值（JSON ground truth）作为位姿来源；相机识别还没做。")
        return
    if ctx["agent"] is None:
        log("[FAIL] [V3.72 SORT] the LLM agent module is not loaded")
        return
    if not ctx["safety_gate"]():
        log("[FAIL] [V3.72 SORT] RUN TASK blocked: stale physical safety latch was not recovered")
        return
    io = SortBackendIO(ctx)
    skills = SortSkills(io)
    log(f"[V3.72 SORT TASK] scene=sort_items | skill SI Sort Items | gripper {skills.prof.label} | "
        f"model={cfg['model']} | task={cfg['task']!r}")

    def _run():
        if not bool(cfg.get("dry_run", False)):
            ctx["start_check"](False)
        return run_sort_task(ctx["agent"], skills, cfg["task"], cfg["model"], cfg["api_key"], cfg["reasoning_effort"],
                             max(10, min(60, int(cfg["max_tool_steps"]))), emit=io.emit,
                             dry_run=bool(cfg.get("dry_run", False)))

    ctx["background"]("RUN TASK (sort_items)", _run, lock_motion=True)


def twin_episode(log=print, seed=None, rule=None, snapshot=None):
    """offline: scripted sorting in the headless twin -> (results, {item: bin}, io, skills)"""
    io = SortTwinIO(log=log, seed=seed)
    sk = SortSkills(io)
    res = sk.scripted_episode(rule)
    state = {iid: it["in_bin"] for iid, it in sk._items().items()}
    if snapshot:
        try:
            io.snapshot(snapshot, lookat=[-0.37, 0.0, 0.05], azimuth=180, elevation=-55, distance=1.0)
        except Exception as exc:
            log(f"[INFO] [V3.72 SORT] no snapshot ({exc})")
    return res, state, io, sk


if __name__ == "__main__":
    if "--twin" in sys.argv:
        seed = None
        if "--seed" in sys.argv:
            seed = int(sys.argv[sys.argv.index("--seed") + 1])
        res, state, _io, sk = twin_episode(seed=seed)
        print(json.dumps(state, ensure_ascii=False))
        print(final_table(sk))
    else:
        print(__doc__)
