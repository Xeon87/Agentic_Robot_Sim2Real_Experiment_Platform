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
  pick_object(item_id, box)top-down grasp across the item's width (jaws closing perpendicular to its long axis; any
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
import NERO_V375_Collision_Check as CC       # noqa: E402

VERSION = "V3.77"
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
# V3.75 BIN FIT + COLLISION PRE-CHECK
FIT_MARGIN_M = 0.006         # an item placed in a bin keeps its footprint this far inside the inner walls (V3.75: 8)
ITEM_WALL_MARGIN_M = 0.004   # 3-D check: held item box vs bin walls at the release pose
MIN_ITEM_GAP_M = 0.003       # V3.76 NO OVERLAP: a placed item keeps this from the items already in the bin
MARKER_V375 = "V3.75 BIN FIT + COLLISION PRE-CHECK"
# V3.76 CARRY SLOWER WITH AN ITEM: the V3.75 run diverged (joint 5 / 7 runaway) while rotating / carrying a held item
HELD_SLOW = 3.0              # time stretch of the rotate / carry / lowering motions while an item is held (V3.75: 1.5-2)
# V3.77 HELD MARGIN: every joint >= this from its limit in every motion with an item in the gripper (lift, rotate,
# carry, lowering, release).  User run V3.76: J6 (range -42..+54 deg) was planned 5-10 deg from its limit while MuJoCo
# tracked with up to 5.6 deg error; the backend trips at 2.9 deg (0.05 rad) beyond a limit.
HELD_MARGIN_DEG = float(os.environ.get("NERO_V377_HELD_MARGIN_DEG", "10.0"))
START_CLEAR_Z_M = 0.30       # V3.76 START CLEAR: below this TCP height the task first rises straight up, then goes home
# V4.8.3 NERO ORIGINAL GRIPPER (two fingers closing symmetrically about the grasp line, official / agx_gripper profile)
SYM_OPEN_MARGIN_MM = 12.0    # open gap = item width + 2 x 6 mm
SYM_STROKE_MM = 70.0         # the user's AgxGripper stroke (the twin's official gripper opens 100 mm; capped the same)
SYM_MIN_FREE_MM = 6.0        # an item needs width + 6 mm <= stroke to be grasped at all
SYM_TIP_ABOVE_TABLE_M = 0.004
TWIN_OFFICIAL_TIP_M = 0.030  # twin official pads end 30 mm past the TCP (NERO_V362 _finger_xml)
SYM_CLAMP_N = 6.0
VIA_MIN_R_M = 0.40           # carry via point at least this far from the base axis
SYM_JAW_CLEAR_M = 0.006      # open fingers keep this from every other item (2-D, finger footprints)
SYM_ABOVE_WALL_M = 0.022     # tidy desk: low containers (30..60 mm) - a lower carry keeps the far side in reach
SYM_RELEASE_EXTRA_MM = 10.0  # release: open to item width + 10 mm (5 mm per side; the bins are small)


def sync_layout():
    """V4.8.3 TIDY DESK: item ids / boxes / tool enums from the current NERO_V343 layout (sort / tidy)"""
    global BOXES
    ITEM_IDS.clear()
    ITEM_IDS.update({f"item_{i + 1}": n for i, n in enumerate(SC.SORT_ITEM_NAMES)})
    BODY_ID.clear()
    BODY_ID.update({v: k for k, v in ITEM_IDS.items()})
    BOXES = tuple(SC.SORT_BINS)
    if "_IDS" in globals():
        _IDS[:] = list(ITEM_IDS)
        props = SORT_TOOLS[1]["parameters"]["properties"]
        props["item_id"]["enum"] = list(_IDS)
        props["box"]["enum"] = list(BOXES) + ["any"]
        SORT_TOOLS[2]["parameters"]["properties"]["box"]["enum"] = list(BOXES)
    return SC.sort_layout()


def tf_name(body):
    """V3.73 TF NAME: MuJoCo body "obj_banana" -> the name the backend's object_pose() expects ("banana")"""
    return body[4:] if body.startswith("obj_") else body


def grasp_R(phi):
    """grasp frame for a top-down grasp: z = down (approach), y = closing direction (fixed -> moving jaw) at yaw phi"""
    z = np.array([0.0, 0.0, -1.0])
    y = np.array([math.cos(phi), math.sin(phi), 0.0])
    return np.column_stack([np.cross(y, z), y, z])


def _turning_segment(tag, p0, R0, p1, R1, n=6):
    """V3.76: intermediate poses of a straight TCP line whose orientation turns from R0 to R1 (slerp), names "~..."
    (strict IK continuity, like CK.segment_points)"""
    q0, q1 = np.asarray(CK.R_to_quat_xyzw(R0), float), np.asarray(CK.R_to_quat_xyzw(R1), float)
    if float(q0 @ q1) < 0.0:
        q1 = -q1
    out = []
    for i in range(1, n):
        u = i / float(n)
        d = float(np.clip(q0 @ q1, -1.0, 1.0))
        th = math.acos(d)
        q = q0 if th < 1e-6 else (math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)
        out.append((f"~{tag}{i}", p0 + (p1 - p0) * u, CK.quat_xyzw_to_R(q / np.linalg.norm(q))))
    return out


def _v377_rejects(v, rej):
    """V3.75 collision rejects (MoveIt pairs / SO-101 boxes) and V3.77 joint-margin rejects into the tool result"""
    col = [r_ for r_ in rej if "collision" in r_]
    mar = sorted([r_ for r_ in rej if "joint_margin_deg" in r_], key=lambda r_: -float(r_["joint_margin_deg"]))
    if col:
        v["collision_rejects"] = col[:4]
        v["collision_rejected_angles"] = len(col)
    if mar:
        v["margin_rejects"] = mar[:3]
        v["note"] = (f"best plan only {mar[0]['joint_margin_deg']} deg from a joint limit while the item is held "
                     f"(needs >= {HELD_MARGIN_DEG:.0f} deg)")


def yaw_of_wxyz(q):
    w, x, y, z = [float(v) for v in q]
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


# =============================================================================================== IO adapters
class SortTwinIO(CK.TwinIO):
    """the V3.62 headless twin (SO-101 parts, backend gripper / CONNECT-hold logic, URDF arm kinematics) with the
    sort_items bins + items"""

    def __init__(self, log=None, seed=None, poses=None, gripper=None):
        sync_layout()
        if gripper is None:                                # V4.8.3: the tidy layout runs with the official gripper
            gripper = "official" if SC.sort_layout() == SC.LAYOUT_TIDY else "so101"
        world = SC.sort_world_xml(seed, bare=True, poses=poses)
        super().__init__(log=log, gripper=gripper, world_xml=world, hold_names=SC.SORT_ITEM_NAMES,
                         soft_pads=gripper == "so101")      # V3.76 SOFT PADS, like the sorting scene

    def object_pose(self, body):
        b = self.d.body(body)
        return [float(v) for v in b.xpos], [float(v) for v in b.xquat]

    def state_check(self, q, opening_mm=None, extra_parts=None):
        """V3.75 COLLISION PRE-CHECK (twin): no MoveIt here - SO-101 (V4.8.3: or official) + wrist coarse boxes vs
        table / bins"""
        if opening_mm is None:
            opening_mm = self.gripper_state().get("opening_mm")
        return CC.assess(None, q, opening_mm, CC.world_boxes(SCENE), extra_parts=extra_parts, gripper=self._gripper)


def _own_path_enabled():
    return os.environ.get("NERO_V375_SO101_OWN_PATH", "1").strip() != "0"


class SortBackendIO(CK.BackendIO):
    """the V3.62 backend adapter + item poses from TF (sim_obj_*) + the SO-101 profile"""

    def object_pose(self, body):
        # V3.73 TF NAME: the backend's object_pose(name) looks up TF "sim_obj_" + name and the scene publishes
        # "sim_" + body ("sim_obj_banana") -> pass the name WITHOUT "obj_" (V3.72 sent "obj_banana" ->
        # "sim_obj_obj_banana" does not exist: observe_scene failed in the first simulation run)
        p, q = self.core.object_pose(tf_name(body))
        return [float(v) for v in p], [float(q[3]), float(q[0]), float(q[1]), float(q[2])]

    def gripper_profile(self):
        # V4.8.3 TIDY DESK: the tidy_desk variant runs with the official gripper (the backend's own model)
        return CK.so101_profile() if SC.gripper_variant() == "so101" else CK.OFFICIAL_PROFILE

    def pose_mode(self):
        return "truth"

    # -- V3.75 COLLISION PRE-CHECK (backend): MoveIt (all contact pairs) + the SO-101 coarse geometry
    @property
    def v375_own_path(self):
        # V4.8.3: with the official gripper MoveIt's gripper model is the mounted one - no own path around it
        return _own_path_enabled() and SC.gripper_variant() == "so101"

    def state_check(self, q, opening_mm=None, extra_parts=None):
        if opening_mm is None:
            opening_mm = self.gripper_state().get("opening_mm")
        return CC.assess(self.core, q, opening_mm, CC.world_boxes(SCENE), extra_parts=extra_parts,
                         gripper="so101" if SC.gripper_variant() == "so101" else "official")

    def move_line(self, label, xyz, quat, speed=0.05, slow=None):
        """V3.75 OWN PATH WHEN ONLY MOVEIT'S GRIPPER MODEL COLLIDES: MoveIt's robot model carries the official AgileX
        gripper (its fingers ~19 mm below the SO-101 grasp point).  When MoveIt refuses a straight segment, the
        skill's own straight path (strict-continuity IK, every point checked by check_joint_path) is checked state by
        state: MoveIt contact pairs + the SO-101 coarse boxes vs table / bins.  Only if every MoveIt pair involves a
        link of MoveIt's gripper model (none an arm link 0-7) and the SO-101 is clear, that path is executed as one
        trajectory (NERO_V71 _make_incremental_joint_trajectory, 40 mm/s retime); otherwise the refusal stands (with
        the colliding pairs in the log).  NERO_V375_SO101_OWN_PATH=0 switches this off.
        V3.77 OWN PATH LIKE THE PLAN: the own path turns the wrist from the current to the target orientation along the
        line (slerp) exactly like the plan (_turning_segment) and MoveIt's Cartesian path - V3.76 held the target
        orientation from the first point, needed an IK branch change and fell back to a joint move with J6 at its
        limit (user run V3.76, apple).  It is also used when MoveIt's path was refused by the joint-path check (not
        only for gripper-model contacts), as long as MoveIt confirms that no arm link collides."""
        r = super().move_line(label, xyz, quat, speed=speed, slow=slow)
        if not r.get("refused") or not self.v375_own_path:
            return r
        R = CK.quat_xyzw_to_R(quat)
        try:
            q0 = self.current_q()
            p0_, q0_ = self.tcp_pose()
            p0, R0 = np.asarray(p0_, float), CK.quat_xyzw_to_R(q0_)
        except Exception:
            return r
        dist = float(np.linalg.norm(np.asarray(xyz, float) - p0))
        Rd = R @ R0.T
        turn = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Rd) - 1.0) / 2.0))))
        n = int(min(60, max(2, math.ceil(dist / 0.004), math.ceil(turn / 2.0))))
        ch = _turning_segment("s", p0, R0, np.asarray(xyz, float), R, n) + [("~end", np.asarray(xyz, float), R)]
        rr = CK.ik_chain(ch, q0)
        if not rr["ok"]:
            r["why"] = f"{r.get('why')}; own straight path: {rr['why']}"
            return r
        path = [q0] + rr["q"]
        chk = CK.check_joint_path(path)
        if not chk["ok"]:
            r["why"] = f"{r.get('why')}; own straight path: {chk['why']}"
            return r
        opening = self.gripper_state().get("opening_mm")
        idx = sorted(set(list(range(1, len(path), 3)) + [len(path) - 1]))
        model, avail = [], True
        for i in idx:
            a = self.state_check(path[i], opening)
            avail = avail and a["moveit_available"]
            if not a["ok"]:
                self.emit(f"[FAIL] [V3.75 MOVEIT] {label}: refused and the own straight path collides at point {i}/"
                          f"{len(path) - 1}: " + CC.pairs_text(a["blocking"] + a["so101"]))
                r["why"] = f"{r.get('why')}; collision: {CC.pairs_text(a['blocking'] + a['so101'])}"
                r["collision"] = a
                return r
            model += [p_ for p_ in a["model_only"] if p_ not in model]
        if not avail:
            # MoveIt not reachable: no confirmation that no arm link collides -> keep the refusal (V3.64 fallback)
            return r
        pts = [p0] + [np.asarray(c[1], float) for c in ch]
        try:
            traj = self.core._make_incremental_joint_trajectory(path, [list(map(float, p_)) for p_ in pts])
        except Exception as exc:
            r["why"] = f"{r.get('why')}; own path trajectory: {str(exc)[:120]}"
            return r
        self.emit(f"[WARN] [V3.75 MOVEIT] {label}: MoveIt refused ({r.get('why')}) - "
                  + (f"its only contacts are with its official-gripper model ({CC.pairs_text(model)})" if model else
                     "no arm link collides (V3.77: its own path failed the joint-path check)")
                  + f"; the mounted SO-101 is clear of table / bins -> the skill's own straight path ({len(path) - 1} "
                  f"points, wrist turn {turn:.0f} deg, min margin {chk['min_margin_deg']} deg)")
        s0 = int(self.sh["safety_trip_count"].value)
        f = max(1.0, 0.04 / max(float(speed), 1e-3))
        err = self._exec(label, traj, (f * float(slow)) if slow else (f if f > 1.0 else None), s0)
        if err:
            return err
        out = self._post(label, xyz, s0)
        out.update(path_check=chk, own_path=True, model_only_pairs=model)
        return out


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
        self.held_phi = None             # V3.76: closing direction (deg, world) of the current grasp
        self.placed = {}                 # item id -> requested bin
        self.picked_from = {}
        sync_layout()
        # V4.8.3 NERO ORIGINAL GRIPPER: two fingers closing symmetrically about the grasp line
        self.sym = self.prof.name in ("official", "agx_gripper")
        self.stroke_mm = SYM_STROKE_MM
        # V4.8.3: lift just high enough for the tallest container wall (+ 50 mm); V3.72: 130 mm for 60 mm walls
        self.lift_m = (max([float(b["wall_h"]) for b in SC.SORT_BINS.values()] + [0.05]) + 0.05) if self.sym else LIFT_M
        if self.prof.name == "agx_gripper":
            import NERO_V481_Agx_Geom as AG
            self.tip_drop_m = float(AG.tip_drop_m())
            try:
                import NERO_V48_Gripper as GG
                self.stroke_mm = float((GG.load_cfg().get("agx") or {}).get("stroke_mm") or SYM_STROKE_MM)
            except Exception:
                pass
        else:
            self.tip_drop_m = TWIN_OFFICIAL_TIP_M if self.sym else CK.TIP_BEYOND_GRASP_M

    def _cmd(self, gap_mm):
        """gripper opening command for a fingertip gap (SO-101: its jaw-angle mapping; original gripper: the gap)"""
        if self.sym:
            return float(min(max(float(gap_mm), 0.0), self.stroke_mm))
        import NERO_V364_SO101_Gripper as G
        return G.opening_mm_for_tip_gap(gap_mm)

    def _sym_fingers_fit(self, box, iid, slot, R, sp):
        """V4.8.3: at the release (fingers opened to the release gap around the item centre) both fingers stay inside
        the bin's inner walls and clear of the items already in it"""
        import NERO_V481_Agx_Geom as AG
        b = SC.SORT_BINS[box]
        cx, cy = b["centre"]
        ix, iy = b["inner"][0] / 2.0 - 0.002, b["inner"][1] / 2.0 - 0.002
        gap = self._cmd(sp["width"] * 1000.0 + SYM_RELEASE_EXTRA_MM) / 1000.0
        f_w, f_t = AG.finger_footprint_mm(None if self.prof.name == "agx_gripper" else "stock")   # V4.8.4 Softfin
        t = f_t / 1000.0
        xG, yG = R[:2, 0], R[:2, 1]
        ang = math.atan2(float(xG[1]), float(xG[0]))
        for s_ in (-1.0, 1.0):
            c = np.asarray(slot, float) + s_ * (gap / 2.0 + t / 2.0) * np.asarray(yG, float)
            for px, py in CC.rect_corners((float(c[0]), float(c[1])), (f_w / 2000.0, t / 2.0), ang):
                if abs(px - cx) > ix or abs(py - cy) > iy:
                    return False
        inside = [o for k, o in self._items().items() if k != iid and o["in_bin"] == box]
        return not inside or self._sym_jaw_clear(np.asarray(slot, float), R, gap, inside) >= 0.003

    def _sym_jaw_clear(self, p_a, R, gap_m, others):
        """V4.8.3: smallest clearance of the two open fingers (56 x 26.5 mm footprints) to the other items"""
        import NERO_V481_Agx_Geom as AG
        xG, yG = R[:2, 0], R[:2, 1]
        ang = math.atan2(float(xG[1]), float(xG[0]))
        f_w, f_t = AG.finger_footprint_mm(None if self.prof.name == "agx_gripper" else "stock")   # V4.8.4 Softfin
        t = f_t / 1000.0
        best = 1e9
        for s_ in (-1.0, 1.0):
            c = np.asarray(p_a[:2], float) + s_ * (gap_m / 2.0 + t / 2.0) * np.asarray(yG, float)
            fr = ((float(c[0]), float(c[1])), (f_w / 2000.0, t / 2.0), ang)
            for o in others:
                best = min(best, CC.rect_dist(fr, SC.sort_item_footprint(o["body"], o["c"][:2], o["yaw_deg"])))
        return best

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
                          "held": self.held == iid, "name_cn": SC.item_cn(it["body"]),
                          "attributes": list(sp.get("attrs") or [])})
        bins = []
        for col, b in SC.SORT_BINS.items():
            bins.append({"box": col, "colour": col, "side": b["side"], "centre_xy_m": list(b["centre"]),
                         "inner_size_mm": [round(v * 1000) for v in b["inner"]], "wall_height_mm": round(b["wall_h"] * 1000),
                         "open_top": True, "contains": [i["id"] for i in items if i["in_bin"] == col],
                         "name_cn": SC.bin_label(col), "accepts": list(b.get("accepts") or []),
                         "capacity": b.get("capacity")})
        return {"scene": SCENE, "pose_source": "simulator ground truth (TF); labels = the scene's object names",
                "items": items, "boxes": bins, "held_item": self.held, "tcp_xyz_m": [round(v, 4) for v in tcp],
                "gripper": g}

    def observe_scene(self, **_):
        sc = self.scene()
        if self.stage == "IDLE":
            self.stage = "OBSERVED"
        nxt = ["place_in_box"] if self.held else ["pick_object"]
        return self._res("observe_scene", True, {"items": len(sc["items"])}, nxt=nxt, scene=sc)

    # -- V3.76 STATE AFTER A SAFETY STOP
    def _v376_sync_held(self):
        """the backend may let go of the item after a safety stop (V3.75 run: RESTORE KEEP HOLD, a second runaway,
        then "HOLD CONTINUITY released"): the skill's "held" follows the gripper and the item, not its own record.
        -> None (still held / nothing held) or {"item", "now_in_bin"} when the item is no longer in the gripper"""
        if not self.held:
            return None
        st = self.io.gripper_state()
        it = self._item(self.held)
        tcp, _q = self.io.tcp_pose()
        near = float(np.linalg.norm(it["c"] - np.asarray(tcp, float))) < 0.15
        if near and (st.get("hold_active") or (st.get("contact_left") and st.get("contact_right"))):
            return None
        iid = self.held
        self.held, self.held_off = None, None
        self.stage = "PLACED" if it["in_bin"] else "OBSERVED"
        self.io.emit(f"[WARN] [V3.76 SORT] {iid} ({it['spec']['label']}) is no longer in the gripper (let go by the safety "
                     f"stop); it lies " + (f"in the {it['in_bin']} bin" if it["in_bin"] else "on the table / floor"))
        return {"item": iid, "now_in_bin": it["in_bin"]}

    def v376_clear_start(self):
        """V3.76 START CLEAR: the V3.75 run's second task started low next to a bin and the backend's start check
        refused its own straight retreat (MoveIt's official-gripper model).  Below START_CLEAR_Z_M: rise straight up
        (the IO's checked straight path, V3.75 own path where only MoveIt's gripper model collides), then go home."""
        tcp, q = self.io.tcp_pose()
        if float(tcp[2]) >= START_CLEAR_Z_M:
            return {"moved": False, "tcp_z_m": round(float(tcp[2]), 3)}
        # as high as a straight vertical line with the current wrist orientation reaches (>= +50 mm), then home
        z0 = float(tcp[2])
        goals = [z for z in (START_CLEAR_Z_M, 0.25, z0 + 0.10, z0 + 0.05) if z > z0 + 0.045]
        self.io.emit(f"[INFO] [V3.76 SORT] task start: TCP {z0 * 1000:.0f} mm high (< {START_CLEAR_Z_M * 1000:.0f} mm) -> "
                     "straight up, then home")
        last = None
        for z in sorted(set(goals), reverse=True):
            try:
                self._go_line(f"start clear (straight up to {z * 1000:.0f} mm)", [tcp[0], tcp[1], z],
                              CK.quat_xyzw_to_R(q), LIFT_SPEED, slow=HELD_SLOW if self.held else None)
                break
            except self._Abort as e:
                last = e
                if e.reason != "TARGET_UNREACHABLE":
                    break
        else:
            last = last or self._Abort("TARGET_UNREACHABLE", {})
        rose = float(self.io.tcp_pose()[0][2]) >= z0 + 0.04
        if not rose:
            # no straight rise from this posture (e.g. a joint near its limit): home directly (backend: MoveIt-planned,
            # collision-checked joint move)
            self.io.emit(f"[WARN] [V3.76 SORT] start clear: no straight rise ({last.reason if last else '?'}: "
                         f"{(last.info.get('detail') or last.info.get('error') or '') if last else ''}) -> home directly")
        r = self.io.go_home("V3.76 sort task start")
        return {"moved": True, "home": bool(r.get("ok", True)), "straight_rise": rose,
                "rose_to_mm": round(float(self.io.tcp_pose()[0][2]) * 1000)}

    def _v376_placeable(self, it, phi_deg, q_lift, p_grasp, boxes=None):
        """V3.76 PLACEABLE GRASP: in how many bins the item can be laid lengthwise from this grasp - the place planner
        itself (_plan_place, lengthwise angles only) on the state predicted after the lift (the item rigid in the
        gripper, LIFT_M higher)"""
        T_l = self._arm().fk(np.asarray(q_lift, float))
        p_l, R_l = T_l[:3, 3].copy(), T_l[:3, :3].copy()
        c_l = np.asarray(it["c"], float) + np.array([0.0, 0.0, self.lift_m])
        off = R_l.T @ (c_l - p_l)
        n = 0
        for box in (boxes or BOXES):
            # V3.77: +2 deg over the held margin - the real lift differs a little from the rigid prediction (test: the
            # predicted placement had >= 10 deg, after the lift the best one had 9.9 deg)
            best, _rej = self._plan_place(box, it["id"], it, q_lift, p_l, R_l, off, phi_deg, float(c_l[2]),
                                          lengthwise_only=it["spec"]["shape"] != "round",
                                          need_margin_deg=HELD_MARGIN_DEG + 2.0)
            n += int(best is not None)
        return n

    # -- V3.75 COLLISION PRE-CHECK
    def _v375_states_ok(self, states, opening_mm, extra=None):
        """every state: MoveIt pairs (backend) + SO-101 coarse boxes; MoveIt-gripper-model pairs are accepted only if
        the IO can drive the own checked path.  -> (ok, pairs text / None)"""
        fn = getattr(self.io, "state_check", None)
        if fn is None:
            return True, None
        own = bool(getattr(self.io, "v375_own_path", False))
        for i, q in enumerate(states):
            a = fn(q, opening_mm, extra_parts=(extra or {}).get(i))
            bad = a["blocking"] + a["so101"] + ([] if own else a["model_only"])
            if bad:
                return False, CC.pairs_text(bad, 3)
        return True, None

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

    def _plan_pick(self, it, q_now, box=None):
        sp = it["spec"]
        width = float(sp["width"])
        gap_m = (width * 1000.0 + OPEN_MARGIN_MM) / 1000.0
        self.v375_rejects = []
        if self.sym:
            if width * 1000.0 + SYM_MIN_FREE_MM > self.stroke_mm:      # V4.8.3: wider than the stroke allows
                self.v375_rejects = [{"too_wide_mm": round(width * 1000.0), "stroke_mm": self.stroke_mm}]
                return None
            gap_m = min(width * 1000.0 + SYM_OPEN_MARGIN_MM, self.stroke_mm) / 1000.0
        if sp["shape"] == "round":
            phis = list(range(0, 360, 30)) + list(range(15, 360, 30))
        else:
            base = it["yaw_deg"] + 90.0
            phis = [base + d for d in (0, 180, 15, -15, 195, 165, 30, -30, 210, 150, 45, -45, 225, 135)]
        others = [o for k, o in self._items().items() if k != it["id"] and o["in_bin"] is None]
        zg = float(it["c"][2])
        if sp.get("grasp_z") is not None:                # grasp height above the item bottom (flat sides)
            zg = float(it["c"][2]) - float(sp["cz"]) + float(sp["grasp_z"])
        zg = max(zg, (self.tip_drop_m + SYM_TIP_ABOVE_TABLE_M) if self.sym else TIP_Z_MIN_M)
        # half width at the grasp height (round cross-sections - banana / apple / marker - are narrower off-centre)
        dz = zg - float(it["c"][2])
        half = math.sqrt(max((width / 2.0) ** 2 - dz * dz, (width / 4.0) ** 2))
        import NERO_V364_SO101_Gripper as G
        cands = []
        open_v = self._cmd(gap_m * 1000.0)
        held_v = self._cmd(max(2.0, width * 1000.0))
        for ph in phis:
            phi = math.radians(ph)
            R = grasp_R(phi)
            yG = R[:, 1]
            if self.sym:
                # V4.8.3: the grasp line through the item centre, both fingers gap/2 off it
                p_a = np.array([it["c"][0], it["c"][1], zg])
                clear = self._sym_jaw_clear(p_a, R, gap_m, others)
            else:
                # grasp frame origin = fixed pad face + WALL_HALF; put the pad face FIXED_PAD_GAP off the item side
                p_a = np.array([it["c"][0], it["c"][1], zg]) - (half + FIXED_PAD_GAP_M - G.WALL_HALF) * yG
                clear = self._jaw_clear(it, p_a, yG, gap_m, others)
            if clear < (SYM_JAW_CLEAR_M if self.sym else JAW_CLEAR_M):
                continue
            pts = {"pregrasp": p_a + [0, 0, PRE_M], "grasp": p_a, "lift": p_a + [0, 0, self.lift_m]}
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
            # V3.77 HELD MARGIN: the lift (grasp -> lift) carries the item
            m_held = min(min(CK.joint_margins_deg(qq)) for qq in r["q"][i_g:i_l + 1])
            if m_held < HELD_MARGIN_DEG:
                self.v375_rejects.append({"phi_deg": round(_wrap(ph), 1),
                                          "joint_margin_deg": round(m_held, 1), "need_deg": HELD_MARGIN_DEG})
                continue
            # V3.75 COLLISION PRE-CHECK: pregrasp, the descent, the grasp (jaws open) and the lift (jaws closed)
            ok_o, why_o = self._v375_states_ok([r["q"][0], r["q"][2], r["q"][4], r["q"][i_g]], open_v)
            ok_c, why_c = (self._v375_states_ok([r["q"][i_l]], held_v) if ok_o else (False, None))
            if not (ok_o and ok_c):
                self.v375_rejects.append({"phi_deg": round(_wrap(ph), 1), "collision": why_o or why_c})
                continue
            # V3.76 PLACEABLE GRASP: a long item must later lie lengthwise in a bin; with J7 limited to +-90 deg the grasp
            # angle decides whether that wrist angle is reachable (V3.76 twin, seed 2: no lengthwise place for the banana)
            # V3.76 PICK FOR THE BIN: the agent says where the item goes (pick_object box) -> only that bin counts
            targets = (box,) if box in BOXES else BOXES
            n_place = self._v376_placeable(it, ph, r["q"][i_l], p_a, targets)     # long and round items
            # V3.76 ACROSS FIRST: how far the closing direction is from perpendicular to the long axis (a diagonal
            # grasp turns a banana by ~20 deg when the jaws open in the bin)
            off_perp = 0.0 if sp["shape"] == "round" else abs(_wrap(ph - float(it["yaw_deg"]) - 90.0))
            off_perp = min(off_perp, 180.0 - off_perp)
            cands.append({"phi_deg": round(_wrap(ph), 1), "margin": r["min_margin_deg"], "binding": r["binding"],
                          "jaw_clear_mm": round(clear * 1000, 1), "tcp": tcp, "q": q,
                          "q_down": r["q"][:i_g + 1], "q_up": r["q"][i_g:i_l + 1], "gap_mm": gap_m * 1000.0,
                          "zg": zg, "placeable_bins": n_place, "off_perp_deg": off_perp})
            if r["min_margin_deg"] >= PREFERRED_MARGIN_DEG and n_place == len(targets) and off_perp < 1.0:
                break
        if not cands:
            return None
        # placeable (in the target bin / in some bin) first, then a grasp across the item (a diagonal grasp lets the
        # item swing in the hold - V3.76 twin), then placeable in more bins, then the joint margin
        return max(cands, key=lambda c: (min(c["placeable_bins"], 1), -round(c["off_perp_deg"] / 15.0),
                                         c["placeable_bins"], min(c["margin"], PREFERRED_MARGIN_DEG), c["margin"]))

    def pick_object(self, item_id="item_1", box="any", **_):
        tool = "pick_object"
        if item_id not in ITEM_IDS:
            return self._res(tool, False, reason="UNKNOWN_ITEM", nxt=["observe_scene"])
        self._v376_sync_held()
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
        plan = self._plan_pick(it, q_now, box)
        if box in BOXES:
            v["planned_for_box"] = box
        if plan is not None and plan.get("placeable_bins", 1) == 0:
            v["warning_place"] = ("no grasp found from which the item can be laid into "
                                  + (f"the {box} bin" if box in BOXES else "a bin") + " (lengthwise) - placing may fail")
        if plan is None:
            rej = getattr(self, "v375_rejects", [])
            if rej and "too_wide_mm" in rej[0]:                  # V4.8.3: wider than the gripper stroke
                v.update(rej[0])
                return self._res(tool, False, v, reason="ITEM_TOO_WIDE", nxt=["observe_scene", "stop_and_report"])
            _v377_rejects(v, rej)
            return self._res(tool, False, v, reason="GRASP_UNREACHABLE", nxt=["observe_scene", "stop_and_report"])
        v.update(closing_yaw_deg=plan["phi_deg"], min_joint_margin_deg=plan["margin"], binding=plan["binding"],
                 jaw_clear_mm=plan["jaw_clear_mm"])
        import NERO_V364_SO101_Gripper as G
        z0 = float(it["c"][2])
        try:
            self.io.gripper_open(self._cmd(plan["gap_mm"]))
            self._go_joints(f"above {item_id} ({it['spec']['label']})", plan["q"]["pregrasp"])
            p, R = plan["tcp"]["grasp"]
            self._go_line(f"down to {item_id}", p, R, DESCENT_SPEED, q_hint=plan["q"]["grasp"], q_path=plan["q_down"])
            g = self.io.gripper_clamp(SYM_CLAMP_N if self.sym else CLAMP_N)
            v["clamp"] = {k: g.get(k) for k in ("contact_left", "contact_right", "hold_active", "opening_mm")}
            p, R = plan["tcp"]["lift"]
            self.held = item_id                     # dip check of the carry fallback while lifting
            self._go_line(f"lift {item_id}", p, R, LIFT_SPEED, q_hint=plan["q"]["lift"], q_path=plan["q_up"])
        except self._Abort as e:
            res = self._abort_result(tool, e, v)
            if self.held and self._v376_sync_held() is None:
                res["state_after"]["held_item"] = self.held       # V3.76: still in the gripper after the stop
                res["recommended_next_actions"] = ["place_in_box", "observe_scene"]
            else:
                self.held = None
                res["state_after"]["held_item"] = None
            return res
        m = self._measure_after_lift(item_id, z0, plan)        # V4.8.3: real robot -> the gripper width decides
        v["item_rise_mm"] = round(m["rise_m"] * 1000.0, 1)
        v.update(m.get("extra") or {})
        if not m["held"]:
            self.held = None
            self.io.gripper_open(self._cmd(plan["gap_mm"]))
            self._retreat_up(0.05)
            return self._res(tool, False, v, reason=m["reason"], nxt=["observe_scene", "pick_object"])
        tcp, qt = self.io.tcp_pose()
        self.held_off = CK.quat_xyzw_to_R(qt).T @ (np.asarray(m["c"], float) - np.asarray(tcp, float))
        self.stage = "HOLDING"
        self.held_phi = float(plan["phi_deg"])
        # V4.8.3: where it came from - return_item_to_table puts it back there when no container takes it
        self.held_pick = {"item": item_id, "grasp": plan["tcp"]["grasp"], "lift": plan["tcp"]["lift"],
                          "q_grasp": plan["q"]["grasp"], "q_lift": plan["q"]["lift"], "gap_mm": plan["gap_mm"],
                          "q_down": list(reversed(plan["q_up"]))}
        self.picked_from[item_id] = it["in_bin"]
        return self._res(tool, True, v, nxt=["place_in_box"])

    # -- place
    def _slot(self, box, iid, phi, yaw_deg=None, ranked=False):
        """V3.75 BIN FIT: item centre in the bin for the closing direction phi and the item's long-axis yaw at the
        release (yaw_deg).  Candidate centres on a grid; the item footprint (NERO_V343 sort_item_footprint) must stay
        FIT_MARGIN_M inside the inner walls.  Items already in the bin -> the candidate with the largest clearance to
        their footprints; an empty bin -> pushed to the side nearer the robot (room for the next item).  -> (x, y) or
        None when the item does not fit in this orientation.  (yaw_deg None: the V3.72 rule, slots +-45 mm along phi)"""
        b = SC.SORT_BINS[box]
        cx, cy = b["centre"]
        inside = [o for k, o in self._items().items() if k != iid and o["in_bin"] == box]
        if yaw_deg is None:
            u = np.array([math.cos(phi), math.sin(phi)])
            slots = [np.array([cx, cy]) + SLOT_DX_M * u, np.array([cx, cy]) - SLOT_DX_M * u]
            if not inside:
                return min(slots, key=lambda s_: float(np.hypot(*s_)))
            return max(slots, key=lambda s_: min(float(np.hypot(*(s_ - o["c"][:2]))) for o in inside))
        sp = SC.SORT_ITEMS[ITEM_IDS[iid]]
        if sp["shape"] == "round":
            yaw_deg = 0.0            # V3.76: a round item's footprint does not turn with the wrist
        a = math.radians(yaw_deg)
        ul, uc = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
        ix, iy = b["inner"][0] / 2.0 - FIT_MARGIN_M, b["inner"][1] / 2.0 - FIT_MARGIN_M
        fits = []
        for t in np.linspace(-0.08, 0.08, 33):           # V3.76: 5 mm grid (V3.75: 10 mm)
            for s_ in (0.0, -0.02, 0.02, -0.04, 0.04):
                c = np.array([cx, cy]) + t * uc + s_ * ul
                rc, half, ang = SC.sort_item_footprint(ITEM_IDS[iid], c, yaw_deg)
                if all(abs(px - cx) <= ix and abs(py - cy) <= iy for px, py in CC.rect_corners(rc, half, ang)):
                    fits.append((c, (rc, half, ang), t, s_))
        if not fits:
            return [] if ranked else None
        if inside:
            def clear(f):
                return min(CC.rect_dist(f[1], SC.sort_item_footprint(o["body"], o["c"][:2], o["yaw_deg"]))
                           for o in inside)
            # V3.76 NO OVERLAP: only spots MIN_ITEM_GAP_M clear of the items in the bin (V3.76 twin: apple on the banana)
            order = sorted([f for f in fits if clear(f) >= MIN_ITEM_GAP_M],
                           key=lambda f: (-round(clear(f), 3), abs(f[3]), float(np.hypot(*f[0]))))
            if ranked:
                return [(float(f[0][0]), float(f[0][1])) for f in order]
            if not order:
                return None
            best = order[0]
        else:
            # V3.76 FREE STRIP: against the side wall that leaves the widest free strip across the bin for the next
            # item (V3.75 took the side nearer the robot on a 10 mm grid: 70 mm left for the 74 mm apple)
            mid = [f for f in fits if f[3] == 0.0] or fits
            yw = (cy - b["inner"][1] / 2.0, cy + b["inner"][1] / 2.0)

            def free(f):
                ys = [py for _px, py in CC.rect_corners(*f[1])]
                return max(yw[1] - max(ys), min(ys) - yw[0])
            # V4.8.3: with the original gripper the first item goes to the FAR side (the strip left free is the one
            # nearer the robot, inside its reach for the next item: its TCP is 155 mm out)
            far = -1.0 if self.sym else 1.0
            order = sorted(mid, key=lambda f: (-round(free(f), 3), far * float(np.hypot(*f[0]))))
            if ranked:           # V3.76: the place loop tries them in turn (gripper clear of the walls)
                return [(float(f[0][0]), float(f[0][1])) for f in order]
            best = order[0]
        return float(best[0][0]), float(best[0][1])

    def _plan_place(self, box, iid, it, q_now, tcp0, R_now, held_off, held_phi, item_z, lengthwise_only=False,
                    need_margin_deg=None):
        """V3.76 ONE PLACE PLANNER: place_in_box and the pick-time check (_v376_placeable, on the predicted state after
        the lift) use this same search -> (best candidate or None, collision rejects)"""
        sp = it["spec"]
        b = SC.SORT_BINS[box]
        c_rel = sp["cz"] + b["floor_t"] + RELEASE_ABOVE_FLOOR_M
        c_above = b["wall_h"] + (SYM_ABOVE_WALL_M if self.sym else ABOVE_WALL_M) + sp["cz"]
        # V3.75 BIN FIT: the item's long-axis yaw at the release follows the wrist rotation; orientations whose
        # footprint does not fit the (smaller) bin are skipped; the long axis along the bin's long side first
        yaw_item = float(it["yaw_deg"])
        held_v = self._cmd(max(2.0, sp["width"] * 1000.0))
        long_x = b["inner"][0] >= b["inner"][1]

        def yaw_at(Rt_):
            Rr = Rt_ @ R_now.T
            return yaw_item + math.degrees(math.atan2(Rr[1, 0], Rr[0, 0]))

        def pref(ph_):
            ya = math.radians(yaw_at(grasp_R(math.radians(ph_)) @ self.prof.T_grasp_tcp[:3, :3]))
            return abs(math.sin(ya)) if long_x else abs(math.cos(ya))
        grid = [float(v) for v in range(0, 360, 15)]
        if sp["shape"] != "round":
            # V3.76 EXACT LENGTHWISE: the two angles that lay the long axis exactly along the bin (the 15 deg grid can
            # be 7.5 deg off - a 20 cm banana 7.5 deg off already takes 89 of the 170 mm bin width)
            d0 = yaw_at(grasp_R(0.0) @ self.prof.T_grasp_tcp[:3, :3]) - (0.0 if long_x else 90.0)
            grid += [round((-d0) % 360.0, 1), round((180.0 - d0) % 360.0, 1)]
        phis = sorted(set(grid), key=lambda ph_: (round(pref(ph_), 2), ph_ % 30 != 0))
        if lengthwise_only:          # V3.76 PLACEABLE GRASP: only the angles that lay the item along the bin
            phis = [ph_ for ph_ in phis if pref(ph_) <= 0.05]
        best = None
        rejects = []
        # V3.76 BOTH WAYS PER ANGLE: an angle is tried with the in-place turn, then with the wrist turning along the carry
        # (V3.75 tried the second way only when no angle worked the first way - the lengthwise banana angles were lost)
        got = set()
        for ph, rotate_first in [(p_, rf_) for p_ in phis for rf_ in (True, False)]:
            if ph not in got:
                # the long axis is perpendicular to the closing direction phi; the slot is offset along phi
                R = grasp_R(math.radians(ph))
                Rt = R @ self.prof.T_grasp_tcp[:3, :3]
                off = Rt @ held_off
                y_rel = yaw_at(Rt)
                # V3.76 GRIPPER CLEAR OF THE WALLS: the fixed finger sits outside the item; a slot pushed against a side
                # wall made it touch the wall (V3.76 twin: the lengthwise banana angles 75 / 255 were rejected) ->
                # the ranked spots of this angle are tried in turn (IK + collision check of the gripper / item)
                found = False
                slots = self._slot(box, iid, math.radians(ph), yaw_deg=y_rel, ranked=True)
                if self.sym:          # V4.8.3: the open fingers (56 x 26.5 mm each) must fit the bin beside the item
                    slots = [sl for sl in slots if self._sym_fingers_fit(box, iid, sl, R, sp)]
                for slot in slots[:8]:
                    sx, sy = slot
                    p_above = np.array([sx, sy, max(c_above, float(item_z))]) - off
                    p_rel = np.array([sx, sy, c_rel]) - off
                    p_rot = np.array([tcp0[0], tcp0[1], max(tcp0[2], p_above[2])])
                    # V3.76 CARRY LINE IN THE PLAN: the straight carry to "above" is part of the chain (fixed wrist after
                    # an in-place rotation, else the wrist turning along the line like the Cartesian path) - V3.76 twin:
                    # an unchecked carry ran J6 into its limit and the joint fallback would dip the held apple
                    if rotate_first:
                        carry = CK.segment_points("c", p_rot, p_above, Rt, 6)
                    else:
                        carry = _turning_segment("c", np.asarray(tcp0, float), R_now, p_above, Rt, 6)
                    chain = (([("rotate", p_rot, Rt)] if rotate_first else []) + carry + [("above", p_above, Rt)]
                             + CK.segment_points("l", p_above, p_rel, Rt, 6) + [("release", p_rel, Rt)])
                    r = CK.ik_chain(chain, q_now)
                    if (not r["ok"] or r["min_margin_deg"] < CK.MIN_MARGIN_RELAXED_DEG) and not rotate_first:
                        r = CK.ik_chain(chain, q_now, start_first=False)
                    via = None
                    if (not r["ok"] or r["min_margin_deg"] < CK.MIN_MARGIN_RELAXED_DEG) and self.sym and rotate_first:
                        # V4.8.3 CARRY VIA: a long carry past the base (e.g. to the trash bin next to the robot) breaks
                        # the IK branch on a straight line -> two straight lines through a point >= 0.40 m out
                        via = (np.asarray(p_rot, float) + np.asarray(p_above, float)) / 2.0
                        rr = float(np.hypot(via[0], via[1]))
                        if rr < VIA_MIN_R_M:
                            via[:2] *= VIA_MIN_R_M / max(rr, 1e-6)
                        via[2] = max(float(p_rot[2]), float(p_above[2]))
                        chain = ([("rotate", p_rot, Rt)] + CK.segment_points("c", p_rot, via, Rt, 4) + [("via", via, Rt)]
                                 + CK.segment_points("v", via, p_above, Rt, 4) + [("above", p_above, Rt)]
                                 + CK.segment_points("l", p_above, p_rel, Rt, 6) + [("release", p_rel, Rt)])
                        r = CK.ik_chain(chain, q_now)
                    if not r["ok"] or r["min_margin_deg"] < CK.MIN_MARGIN_RELAXED_DEG:
                        continue
                    need = HELD_MARGIN_DEG if need_margin_deg is None else float(need_margin_deg)
                    if r["min_margin_deg"] < need:                 # V3.77 HELD MARGIN (item in the gripper throughout)
                        rejects.append({"phi_deg": ph, "joint_margin_deg": r["min_margin_deg"], "binding": r["binding"],
                                        "need_deg": need})
                        continue
                    names = [c[0] for c in chain]
                    # V3.75 COLLISION PRE-CHECK: above / lowering / release with the held item's box at the release
                    i_a, i_r = names.index("above"), names.index("release")
                    sts = [r["q"][i_a], r["q"][(i_a + i_r) // 2], r["q"][i_r]]
                    ibox = CC.item_box((sx, sy), c_rel - sp["cz"], y_rel, sp)
                    ok_s, why_s = self._v375_states_ok(sts, held_v)
                    hit = CC.check_parts([ibox], CC.world_boxes(SCENE)[1:], ITEM_WALL_MARGIN_M) if ok_s else []
                    if not ok_s or hit:
                        rejects.append({"phi_deg": ph, "collision": why_s or CC.pairs_text(hit, 3),
                                                  "slot_xy_m": [round(sx, 3), round(sy, 3)]})
                        continue                   # V3.76: the next spot of this angle
                    found = True
                    break
                if not found:
                    continue
                cand = {"phi": ph, "margin": r["min_margin_deg"], "item_yaw_deg": round(_wrap(y_rel), 1), "binding": r["binding"], "q": dict(zip(names, r["q"])),
                        "q_down": r["q"][names.index("above"):], "p_rot": p_rot, "p_above": p_above, "p_rel": p_rel,
                        "Rt": Rt, "rotate_first": rotate_first, "slot": [round(float(sx), 3), round(float(sy), 3)],
                        "via": via}
                # V3.76 LENGTHWISE FIRST: alignment with the bin's long side ranks before the joint margin (V3.76 twin:
                # the banana lay 22 deg off and took the middle of the bin, the apple had no room)
                cand["align"] = round(pref(ph), 2)
                # V3.76 SMALL WRIST TURN: a 168 deg in-place turn swung the held banana ~30 deg on its CONNECT hold
                # (twin) -> after the alignment, the smaller turn from the grasp's closing direction wins
                cand["turn_deg"] = round(abs(_wrap(ph - held_phi)), 1) if held_phi is not None else 0.0
                key = lambda c_: (-c_["align"], -round(c_["turn_deg"] / 45.0), min(c_["margin"], PREFERRED_MARGIN_DEG),
                                  c_["margin"])
                got.add(ph)
                if best is None or key(cand) > key(best):
                    best = cand
                if cand["margin"] >= PREFERRED_MARGIN_DEG and cand["align"] <= 0.05 and cand["turn_deg"] <= 45.0:
                    break
        return best, rejects

    def place_in_box(self, box="red", **_):
        tool = "place_in_box"
        if box not in BOXES:
            return self._res(tool, False, reason="UNKNOWN_BOX", nxt=["observe_scene"])
        self._v376_sync_held()
        if not self.held or self.held_off is None:
            return self._res(tool, False, reason="NO_ITEM_HELD", nxt=["observe_scene", "pick_object"])
        iid = self.held
        it = self._item(iid)
        sp = it["spec"]
        b = SC.SORT_BINS[box]
        v = {"item": iid, "label": sp["label"], "box": box}
        q_now = self.io.current_q()
        tcp0, _q0 = self.io.tcp_pose()
        R_now = CK.quat_xyzw_to_R(_q0)
        best, self.v375_rejects = self._plan_place(box, iid, it, q_now, tcp0, R_now, self.held_off, self.held_phi,
                                                   float(it["c"][2]))
        if best is None:
            _v377_rejects(v, self.v375_rejects)
            return self._res(tool, False, v, reason="PLACE_UNREACHABLE", nxt=["place_in_box", "stop_and_report"])
        v.update(min_joint_margin_deg=best["margin"], binding=best["binding"], closing_yaw_deg=best["phi"],
                 slot_xy_m=best["slot"], rotate_in_place=best["rotate_first"], item_yaw_deg=best["item_yaw_deg"])
        try:
            # V3.76 CARRY SLOWER WITH AN ITEM: every motion with the item in the gripper is stretched HELD_SLOW x
            if np.linalg.norm(np.asarray(tcp0) - best["p_rot"]) > 0.003:
                self._go_line(f"up with {iid}", best["p_rot"], CK.quat_xyzw_to_R(_q0), LIFT_SPEED, slow=HELD_SLOW)
            if best["rotate_first"]:
                self._go_joints(f"rotate {iid} for the {box} bin", best["q"]["rotate"], slow=HELD_SLOW)
            if best.get("via") is not None:                # V4.8.3 CARRY VIA
                self._go_line(f"carry {iid} towards the {box} bin", best["via"], best["Rt"], CK.CARRY_SPEED,
                              slow=HELD_SLOW, free_fallback=True, q_hint=best["q"]["via"])
            self._go_line(f"carry {iid} over the {box} bin", best["p_above"], best["Rt"], CK.CARRY_SPEED,
                          slow=HELD_SLOW, free_fallback=True, q_hint=best["q"]["above"])
            self.io.sleep(0.5)
            self._go_line(f"lower {iid} into the {box} bin", best["p_rel"], best["Rt"], LOWER_SPEED,
                          slow=HELD_SLOW, q_hint=best["q"]["release"], q_path=best["q_down"])
        except self._Abort as e:
            res = self._abort_result(tool, e, v)
            d = self._v376_sync_held()                 # V3.76 STATE AFTER A SAFETY STOP
            if d is None:
                return res
            v2 = dict(res["physical_verification"], dropped_by_safety_stop=True, now_in_bin=d["now_in_bin"])
            v2.pop("reason", None)
            if d["now_in_bin"] == box:
                self.placed[iid] = box
                return self._res(tool, True, v2, reason="ITEM_DROPPED_IN_BOX", nxt=["observe_scene", "check_bins"],
                                 warnings=[f"{iid} fell into the {box} bin when the safety stop let go of it - it was not "
                                           "lowered and released normally"])
            return self._res(tool, False, v2, reason="ITEM_DROPPED", nxt=["observe_scene", "pick_object"])
        self.io.gripper_open(self._cmd(sp["width"] * 1000.0 + (SYM_RELEASE_EXTRA_MM if self.sym else 18.0)))
        self.io.sleep(0.8)
        self.held, self.held_off = None, None
        self.placed[iid] = box
        try:
            p, q = self.io.tcp_pose()
            self._go_line(f"retreat from the {box} bin", [p[0], p[1], p[2] + 0.10], CK.quat_xyzw_to_R(q), LIFT_SPEED)
        except self._Abort as e:
            v["retreat"] = f"{e.reason}: {e.info.get('error') or e.info.get('detail')}"
        self.io.sleep(0.5)
        where = self._where_after_place(iid, box, best)
        v["now_in_bin"] = where
        self.stage = "PLACED"
        if where != box:
            return self._res(tool, False, v, reason="ITEM_NOT_IN_BOX", nxt=["observe_scene", "pick_object"])
        return self._res(tool, True, v, nxt=["observe_scene", "pick_object", "check_bins"])

    # -- V4.8.3 verification hooks (the real tidy skills measure differently: gripper width / a fresh camera view)
    def _measure_after_lift(self, item_id, z0, plan):
        it2 = self._item(item_id)
        rise = float(it2["c"][2]) - z0
        st = self.io.gripper_state()
        held = rise >= MIN_RISE_M and bool(st.get("hold_active") or (st.get("contact_left") and st.get("contact_right")))
        reason = None if held else ("ITEM_NOT_LIFTED" if rise < MIN_RISE_M else "NO_TWO_FINGER_CONTACT")
        return {"rise_m": rise, "held": held, "reason": reason, "c": it2["c"]}

    def _where_after_place(self, iid, box, best):
        return self._item(iid)["in_bin"]

    def return_item_to_table(self, **_):
        """V4.8.3 TIDY DESK: put the held item back where it was picked (a container refused it: full, no free spot,
        unreachable) - straight back above the pick point, down to 4 mm above the grasp height, open, retreat up"""
        tool = "return_item_to_table"
        self._v376_sync_held()
        hp = getattr(self, "held_pick", None)
        if not self.held:
            return self._res(tool, False, reason="NO_ITEM_HELD", nxt=["observe_scene"])
        if not hp or hp.get("item") != self.held:
            return self._res(tool, False, {"item": self.held}, reason="NO_PICK_POSE", nxt=["place_in_box"])
        iid = self.held
        v = {"item": iid, "label": self._item(iid)["spec"]["label"]}
        try:
            p, R = hp["lift"]
            self._go_line(f"back above where {iid} was", p, R, CK.CARRY_SPEED, slow=HELD_SLOW, free_fallback=True,
                          q_hint=hp["q_lift"])
            p, R = hp["grasp"]
            self._go_line(f"put {iid} back on the table", np.asarray(p, float) + [0.0, 0.0, 0.004], R, LOWER_SPEED,
                          slow=HELD_SLOW, q_hint=hp["q_grasp"], q_path=hp["q_down"])
        except self._Abort as e:
            return self._abort_result(tool, e, v)
        self.io.gripper_open(self._cmd(hp["gap_mm"]))
        self.io.sleep(0.6)
        self.held, self.held_off, self.held_pick = None, None, None
        self._retreat_up(0.08)
        it = self._item(iid)
        v.update(on_table=it["in_bin"] is None, centre_xy_m=[round(float(c), 3) for c in it["c"][:2]])
        self.stage = "OBSERVED"
        return self._res(tool, it["in_bin"] is None, v, nxt=["observe_scene", "pick_object"])

    def check_bins(self, **_):
        its = self._items()
        rows = {iid: {"label": it["spec"]["label"], "in_bin": it["in_bin"]} for iid, it in its.items()}
        on_table = [k for k, r in rows.items() if r["in_bin"] is None]
        v = {"items": rows, "on_table": on_table}
        for b in BOXES:                                  # V4.8.3: every container of the layout
            v[b] = [k for k, r in rows.items() if r["in_bin"] == b]
        return self._res("check_bins", True, v, nxt=["finish_and_report"] if not on_table else ["pick_object"])

    def return_to_safe_pose(self, **_):
        self._v376_sync_held()                         # V3.76: the gripper decides whether an item is held
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
              "check_bins": self.check_bins, "return_to_safe_pose": self.return_to_safe_pose,
              "return_item_to_table": self.return_item_to_table}.get(name)
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
            box = (rule or {}).get(sp["label"]) or default_box(sp)
            if box is None:                               # V4.8.3: no container for it (stays on the desk)
                continue
            r = self.run_tool("pick_object", {"item_id": iid, "box": box})
            out.append(r)
            if not r.get("ok"):
                continue
            out.append(self.run_tool("place_in_box", {"box": box}))
        out.append(self.run_tool("check_bins", {}))
        return out


def default_box(spec):
    """V3.72 rule fruit -> red, rest -> blue; V4.8.3 tidy layout: the container whose "accepts" names the category"""
    if "red" in SC.SORT_BINS and "blue" in SC.SORT_BINS:
        return "red" if spec["category"] == "fruit" else "blue"
    for k, b in SC.SORT_BINS.items():
        if spec["category"] in (b.get("accepts") or []):
            return k
    return None


# =============================================================================================== LLM contract
_IDS = list(ITEM_IDS)
SORT_TOOLS = [
    {"type": "function", "name": "observe_scene",
     "description": "Read the sort_items scene: every item (id, detector label, centre, size, long-axis yaw, which bin it "
                    "is in, held), the two open bins (red / blue, side, contents), gripper, TCP. Observation only. "
                    "ALWAYS call first.",
     "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "pick_object",
     "description": "Grasp one item from above with the gripper (jaws across its width), lift it. Plans the grasp "
                    "itself (reachable closing direction, joint margins, open jaws clear of the other items) so that the "
                    "item can afterwards be laid into `box` (the bin you decided for it; 'any' if undecided). Verifies "
                    "contact and that the item rose with the gripper.",
     "parameters": {"type": "object", "properties": {"item_id": {"type": "string", "enum": _IDS},
                                                     "box": {"type": "string", "enum": list(BOXES) + ["any"]}},
                    "required": ["item_id", "box"], "additionalProperties": False}, "strict": True},
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
2. For every item that is not yet in its target bin: pick_object(item_id, box) -> place_in_box(box), with the same box
   (the grasp is chosen so that the item fits that bin). One physical tool at a
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
    placed, tried, requested = {}, [], {}
    for h in history:
        r = h["result"] if isinstance(h["result"], dict) else {}
        if h["tool"] == "pick_object":
            iid = (r.get("physical_verification") or {}).get("item") or (h.get("args") or {}).get("item_id")
            if iid in ITEM_IDS and iid not in tried:
                tried.append(iid)
        if h["tool"] == "place_in_box":
            pv = r.get("physical_verification") or {}
            if r.get("ok"):
                placed[pv.get("item")] = pv.get("box")
            elif pv.get("item") in ITEM_IDS and pv.get("box"):
                requested[pv["item"]] = pv["box"]       # V3.76: a failed place still says where the agent wanted it
    last = None
    for h in history:
        if h["tool"] == "check_bins" and isinstance(h["result"], dict) and h["result"].get("ok"):
            last = (h["result"].get("physical_verification") or {}).get("items")
    good = sum(1 for k, b in placed.items() if last is None or (last.get(k) or {}).get("in_bin") == b)
    # V3.76 SCORE BY THE FINAL POSITION: an item whose place failed but that lies in the bin the agent asked for at the
    # last check_bins counts too (V3.75 run: the banana fell into the blue bin on a safety stop, s was 0)
    if last is not None:
        good += sum(1 for k, b in requested.items() if k not in placed and (last.get(k) or {}).get("in_bin") == b)
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
    sync_layout()
    io = SortBackendIO(ctx)
    skills = SortSkills(io)
    if SC.sort_layout() == SC.LAYOUT_TIDY:            # V4.8.3 TIDY DESK: the intent layer drives this scene
        import NERO_V483_Tidy as TD
        log(f"[V4.8.3 TIDY TASK] scene=tidy_desk | gripper {skills.prof.label} | model={cfg['model']} | "
            f"said={cfg['task']!r}")
        TD.gui_run_tidy(ctx, io, skills, cfg)
        return
    log(f"[V3.72 SORT TASK] scene=sort_items | skill SI Sort Items | gripper {skills.prof.label} | "
        f"model={cfg['model']} | task={cfg['task']!r}")

    def _run():
        if not bool(cfg.get("dry_run", False)):
            skills.v376_clear_start()                  # V3.76 START CLEAR (low next to a bin -> straight up, home)
            ctx["start_check"](False)
        return run_sort_task(ctx["agent"], skills, cfg["task"], cfg["model"], cfg["api_key"], cfg["reasoning_effort"],
                             max(10, min(60, int(cfg["max_tool_steps"]))), emit=io.emit,
                             dry_run=bool(cfg.get("dry_run", False)))

    ctx["background"]("RUN TASK (sort_items)", _run, lock_motion=True)


def twin_episode(log=print, seed=None, rule=None, snapshot=None, layout=None, gripper=None):
    """offline: scripted sorting in the headless twin -> (results, {item: bin}, io, skills)
    V4.8.3: layout "tidy" = the tidy_desk items / containers (official gripper unless gripper= says otherwise)"""
    if layout is not None:
        SC.set_sort_layout(layout)
    io = SortTwinIO(log=log, seed=seed, gripper=gripper)
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
        res, state, _io, sk = twin_episode(seed=seed, layout="tidy" if "--tidy" in sys.argv else None,
                                           gripper=sys.argv[sys.argv.index("--gripper") + 1] if "--gripper" in sys.argv else None)
        print(json.dumps(state, ensure_ascii=False))
        print(final_table(sk))
    else:
        print(__doc__)
