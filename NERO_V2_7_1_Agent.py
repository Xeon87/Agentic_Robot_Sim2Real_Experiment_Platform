#!/usr/bin/env python3
"""NERO V2-9 public-release LLM task planner.

This module is deliberately *high level*.  The language model never sends joint
angles, CAN frames, or raw ROS commands.  Instead it chooses from a small set of
robot skills whose preconditions and physical postconditions are enforced by the
backend.  That separation is the main architectural idea of NERO:

    natural-language task
        -> LLM planner
        -> skill contract
        -> deterministic motion / gripper skill
        -> physical verification
        -> updated state / recovery decision

The current transport is the OpenAI Responses API with function calling.  The
model ID is configurable at runtime; the manipulation stack is not tied to one
specific model such as GPT-5.6 Luna.  Provider abstraction is intentionally left
as future work so this public release stays close to the version validated in
MuJoCo.

IMPORTANT
---------
- This file does not directly control real hardware.
- Safety decisions from the deterministic backend remain authoritative.
- Hand-crafted manipulation priors are engineering heuristics, not learned laws.
"""

import json
import os
import threading
import time
import urllib.error
import urllib.request


# ============================================================
# V3.04 COMPACT MODEL-FACING TOOL OUTPUT
# ------------------------------------------------------------
# The full result is still written to the log.  The model only needs the
# decision-relevant part: the static skill contract and state_before were
# repeated in every result, observe_scene repeated ~25 prior fields per egg,
# and plan_grasp returned all 14-21 candidates (~4-5k tokens).  V3.03 input
# grew to ~30k tokens per egg cycle.  Disable with NERO_V304_COMPACT_TOOL_OUTPUT=0.
# ============================================================
_V304_GG_KEEP = ("effective_width_mm", "recommended_opening_range_mm", "recommended_target_opening_mm", "current_opening_mm")
_V304_CAND_KEEP = ("candidate_id", "strategy", "gripper_yaw_deg", "grasp_center_offset_mm",
                   "moveit_pregrasp_ok", "moveit_pregrasp_error", "already_tried", "score", "neighbor_clearance_mm")


def _v304_compact_for_model(name, result):
    if os.environ.get("NERO_V304_COMPACT_TOOL_OUTPUT", "1") == "0" or not isinstance(result, dict):
        return result
    try:
        out = dict(result)
        out.pop("contract", None)
        out.pop("state_before", None)
        scene = out.get("scene")
        if isinstance(scene, dict):
            scene = dict(scene)
            objs = {}
            for on, ov in (scene.get("objects") or {}).items():
                ov = dict(ov)
                ov.pop("quat_xyzw", None)
                gg = ov.get("grasp_geometry")
                if isinstance(gg, dict):
                    ov["grasp_geometry"] = {k: gg[k] for k in _V304_GG_KEEP if k in gg}
                objs[on] = ov
            scene["objects"] = objs
            rec = scene.get("receptacles", {}).get("egg_holder", {}).get("slots")
            if isinstance(rec, dict):
                scene["receptacles"] = {"egg_holder": {"slots": {
                    k: {"occupied": v.get("occupied"), "occupied_by": v.get("occupied_by")} for k, v in rec.items()}}}
            out["scene"] = scene
        gp = out.get("grasp_plan")
        if isinstance(gp, dict) and isinstance(gp.get("candidates"), list):
            gp = dict(gp)
            cands = gp["candidates"]
            ready = [c for c in cands if c.get("moveit_pregrasp_ok") and not c.get("already_tried")]
            others = [c for c in cands if c not in ready and not c.get("already_tried")]
            show = (ready + others)[:3]
            gp["candidates"] = [{k: c.get(k) for k in _V304_CAND_KEEP if k in c} for c in show]
            gp["candidates_total"] = len(cands)
            gp["candidates_ready"] = len(ready)
            gp["candidates_untried_not_ready"] = len(others)
            for k in ("object_quat_at_plan_xyzw", "basket_center_xy_m", "basket_safe_inner_radius_mm"):
                gp.pop(k, None)
            out["grasp_plan"] = gp
        pp = out.get("placement_plan")
        if isinstance(pp, dict):
            out["placement_plan"] = {k: pp[k] for k in ("object_name", "slot_name", "axis_error_before_deg") if k in pp}
        return out
    except Exception:
        return result


def _v306_with_budget(out, used, limit):
    """V3.06: tell the model how many tool steps remain (only when it matters)."""
    try:
        left = int(limit) - int(used)
        if isinstance(out, dict) and left <= 14:
            out = dict(out)
            out["v306_budget"] = {"steps_used": int(used), "steps_left": left,
                                  "rule": "if steps_left < 10, finish the egg in hand and stop; do not start a new egg"}
    except Exception:
        pass
    return out


PROJECT_VERSION = "2.9.0"
PUBLIC_RELEASE_TAG = "v0.1.0"

# The public release uses model-neutral environment-variable names.  The older
# NERO_ASTRA_* names are still accepted so existing local launch setups keep
# working without edits.  The GUI model field is editable and can also be
# populated from GET /v1/models.
DEFAULT_MODEL = os.environ.get(
    "NERO_LLM_MODEL",
    os.environ.get("NERO_ASTRA_MODEL", "gpt-5.6-luna"),
)
DEFAULT_API_BASE = os.environ.get(
    "NERO_LLM_API_BASE",
    os.environ.get("NERO_ASTRA_API_BASE", "https://api.openai.com/v1"),
).rstrip("/")

# ============================================================
# V3.42 LLM PROVIDERS
# ------------------------------------------------------------
# openai  : OpenAI Responses API (unchanged path, server-side conversation state)
# deepseek: DeepSeek OpenAI-compatible Chat Completions (https://api.deepseek.com)
# kimi    : Moonshot Kimi OpenAI-compatible Chat Completions (https://api.moonshot.ai/v1;
#           China: NERO_KIMI_API_BASE=https://api.moonshot.cn/v1)
# The Chat Completions providers are driven through a local adapter that keeps the
# conversation (including reasoning_content, required by thinking models with tools)
# and returns Responses-shaped objects, so the task loop is the same for all providers.
# Model IDs are presets only: the field stays editable and "Refresh API Models" lists
# what the key can use (GET /models).
# ============================================================
LLM_PROVIDERS = {
    "openai": {
        "label": "OpenAI", "protocol": "responses",
        "api_base": DEFAULT_API_BASE, "key_env": ("OPENAI_API_KEY",),
        "models": [DEFAULT_MODEL],
    },
    "deepseek": {
        "label": "DeepSeek", "protocol": "chat",
        "api_base": os.environ.get("NERO_DEEPSEEK_API_BASE", "https://api.deepseek.com").rstrip("/"),
        "key_env": ("DEEPSEEK_API_KEY",),
        "models": [m.strip() for m in os.environ.get("NERO_DEEPSEEK_MODELS", "deepseek-v4-pro,deepseek-flash").split(",")
                   if m.strip()],
    },
    "kimi": {
        "label": "Kimi (Moonshot)", "protocol": "chat",
        "api_base": os.environ.get("NERO_KIMI_API_BASE", "https://api.moonshot.ai/v1").rstrip("/"),
        "key_env": ("MOONSHOT_API_KEY", "KIMI_API_KEY"),
        "models": [m.strip() for m in os.environ.get("NERO_KIMI_MODELS", "kimi-k2.6,kimi-k3").split(",") if m.strip()],
    },
}


def provider_for_model(model, explicit=None):
    """V3.42: explicit GUI choice wins; otherwise the model ID decides (deepseek-* / kimi-* / moonshot-*)."""
    if explicit and str(explicit) in LLM_PROVIDERS:
        return str(explicit)
    m = str(model or "").lower()
    if m.startswith("deepseek"):
        return "deepseek"
    if m.startswith(("kimi", "moonshot")):
        return "kimi"
    return "openai"


SKILL_CONTRACT_GUIDE = """\
NERO V2-9 EGG TASK SEMANTIC INTEGRATION / FAILURE-AWARE SKILL CONTRACTS

Semantic scene model:
- Manipulable objects are egg_1, egg_2, egg_3, egg_4. All four belong to category=egg.
- egg_basket is the source container.
- egg_holder is a slotted receptacle with slot_1 ... slot_8.
- Each slot reports center_xyz_m, entry_axis, occupied, and occupied_by.
- Eggs are rigid ellipsoids in this baseline. Fragility/tactile damage is NOT enabled yet; do not invent breakage events.
- The purpose of this layer is task grounding: the Agent reasons in terms of eggs, containers, receptacles and slots while V69 motion remains unchanged.

Core gripper principle:
- observe_scene reports continuous actual gripper opening and per-egg grasp geometry/manipulation prior.
- Before approaching an egg, compare actual_opening_mm with that egg's recommended_opening_range_mm and preferred target.
- V3.06: plan_grasp prepares the recommended aperture by itself when the gripper is outside the egg's range. Do NOT call set_gripper_opening before plan_grasp; use it only if plan_grasp explicitly fails on the aperture.
- A collision-check refusal or pre-grasp poke remains a bounded recoverable failure exactly as in V2-7.

State flow for one egg:
OBSERVED -> HOVER_REACHED -> GRASP_POSE_REACHED -> GRASP_ESTABLISHED -> LIFTED -> RECEPTACLE_HOVER_REACHED -> SLOT_PLACE_REACHED -> RELEASED -> PLACEMENT_VERIFIED
A pre-grasp safety trip temporarily enters SAFETY_STOPPED; one explicit reasoned recovery/retry is allowed per egg.

Skill meanings:
- observe_scene: read semantic scene/state only; nothing moves.
- set_gripper_opening(target): pre-grasp aperture only.
- move_above(egg): safe hover above selected egg; egg should not move.
- approach(egg): pre-grasp descent; egg should remain approximately stationary.
- recover_from_safety(...): explicit one-shot recovery for a latched pre-grasp safety stop.
- clamp_gripper(force): establish provisional two-finger grasp.
- lift(height): egg MUST follow upward; validates grasp.
- move_above_receptacle(egg_holder, slot): transport held egg above a selected empty slot.
- place_into_receptacle(egg_holder, slot): lower the still-held egg to the slot placement region; do not release yet.
- open_gripper(): release-only after SLOT_PLACE_REACHED.
- verify_object_in_slot(egg, slot): deterministic final placement check.

The semantic layer does not claim that the current grasp/placement motion is optimal for eggs. If V69 primitives prove physically inadequate, report the failure rather than inventing a new low-level motion.
"""

SYSTEM_INSTRUCTIONS = f"""You are NERO V2-9, the high-level task planner for a MuJoCo simulated 7-DOF NERO robot arm.
You control the robot ONLY through the provided function tools. Never invent joint angles, ROS commands, shell commands, or direct actuator commands.

{SKILL_CONTRACT_GUIDE}


V2-7 CONSTRAINT-AWARE EGG GRASP POLICY:
- For egg pickup from the basket, V2-7 adds a mandatory grasp-planning step.  Do not treat an egg as a point target.
- After observe_scene and aperture preparation, call plan_grasp(object_name).
- plan_grasp is observation/planning only.  It estimates the egg major-axis orientation from the current quaternion, models the two-finger projected footprint relative to the circular wicker rim, generates a bounded set of wrist-yaw / small inward-offset candidates, and MoveIt-preflights their pregrasp poses without executing motion.
- The preferred family aligns the finger-pad long axis with the egg's projected major axis so the closing direction pinches across the egg short axis.  Near-upright eggs use a basket-tangent fallback because horizontal egg yaw is underdetermined.
- plan_grasp returns candidate IDs and a recommended candidate.  Use move_to_pregrasp(object_name, candidate_id), then approach_planned_grasp(object_name, candidate_id).
- A candidate is never repeated after it is marked tried/blocked/contacting.  Re-observe and call plan_grasp so the backend chooses among the remaining bounded candidates.
- For V2-7 egg pickup, do NOT bypass the candidate layer by repeatedly calling legacy move_above/approach.  Those remain available for manual/debug compatibility only.
- V2-27 EDGE RECOVERY: if plan_grasp returns ALL_GRASP_CANDIDATES_EXHAUSTED_OR_INFEASIBLE AND physical_verification.edge_reposition_available=true, do NOT end that object. Call reposition_egg_inward(object_name) exactly once. After the nudge, MUST call observe_scene, re-check/set aperture if needed, then call plan_grasp again from the fresh pose. Never reuse the old candidate IDs. If edge_reposition_available=false, the one-shot budget is already used, or the reposition skill fails, then try another pending object or report the remaining object infeasible; never loop the same recovery.
- Egg breaking / damage is intentionally out of scope in V2-7.  Treat eggs as rigid ellipsoids for this version.


V2-7.1 PLACEMENT-AWARE EGG POLICY:
- V2-7.1 keeps the V2-7 short-axis grasp logic.  The grasp orientation is NOT automatically the correct placement orientation.
- After a successful egg lift, egg placement into the slotted holder MUST use:
    plan_placement -> orient_for_slot -> move_to_slot_entry -> guarded_insert_and_release -> verify_object_in_slot
- Do NOT use legacy move_above_receptacle/place_into_receptacle/open_gripper for egg placement in V2-7.1.  Those remain only for backward/manual debug compatibility.
- plan_placement explicitly reasons about the held egg LONG axis and the slot entry axis.  It also computes a pre-release capture depth so the gripper does not have to descend all the way into the holder walls.
- orient_for_slot performs the large held-object rotation at a HIGH, open workspace pose, not at the slot mouth.  It keeps TCP XYZ approximately fixed during the reorientation and uses the current physical pose as the BioIK seed.  This encourages wrist-dominant J6/J7 motion but does not hard-code joint angles.
- move_to_slot_entry preserves the already-correct long-axis orientation and requires egg-center XY error <= 7 mm before insertion.  It may make one deterministic micro-correction; the LLM should not oscillate around the slot.
- guarded_insert_and_release performs the final descent in bounded 10 mm / 5 mm Cartesian segments along the egg/slot axis.  After every segment the backend checks hold, safety, XY error, and long-axis error.  At capture depth it stops, opens in two stages, retreats vertically, and leaves final success to verify_object_in_slot.
- If observe_scene detects a placement state but hold_active=false, the backend resets the stale placement state so a fresh pickup can begin.  Do not continue a placement after a dropped egg.
- If clamp_gripper reports BYSTANDER_EGG_DISTURBED, a non-target egg moved during approach.  Do not clamp/lift; re-observe and plan another grasp candidate.
- Egg breaking/tactile damage remains intentionally out of scope in V2-7.1.

V2-7.2 placement safety additions:
- Egg identities are physical labels 1/2/3/4 attached to the actual egg bodies.
- Held-object reorientation is slow and segmented (~10 deg/step).
- Each orientation target is planned first; branches with excessive J1-J5 movement are rejected before execution.
- If WRIST_BOUNDED_PLAN_REJECTED occurs, do not loop the same motion. Observe and report feasibility failure.
- If hold_active becomes false during transport/placement, backend clears stale carrying state before the next observation.
- Safety stop or hold loss is never placement success.
- V3.02: if plan_placement returns reason=SLOT_ABOVE_UNREACHABLE_FROM_CURRENT_GRASP, the held egg cannot reach that slot from the current grasp. Call plan_placement again ONCE with the returned recommended_slot (it is empty and reachable), then continue normally. Mention the slot change in the final message.
- V3.03: if move_to_slot_entry returns reason=SLOT_ENTRY_UNREACHABLE, or orient_for_slot returns reason=V303_SLOT_DESCENT_UNREACHABLE, the egg is still held and vertical but that slot cannot be entered with the current grasp. The stage is LIFTED again. Do NOT end the task: call plan_placement ONCE with a different EMPTY slot (prefer one not yet tried, e.g. the next empty slot in numeric order), then orient_for_slot -> move_to_slot_entry -> guarded_insert_and_release -> verify_object_in_slot. Mention the slot change in the final message. Only report failure if the second slot also fails.
- V3.03: an error containing GOAL_IK_UNREACHABLE means the requested pose has no IK solution; repeating the same call will not help. Follow recommended_next_actions instead.
- V3.41 INSERT BLOCKED: guarded_insert_and_release now reports a blocked descent honestly. INSERT_BLOCKED_REPLAN_YAW (egg still held at the slot entry): call plan_placement for the SAME egg and slot, then orient_for_slot (another wrist yaw is chosen), move_to_slot_entry and guarded_insert_and_release again. INSERT_BLOCKED_JOINT_LIMIT: call return_to_pick_location, then continue with the other eggs (another slot may be tried later). Never report a blocked insertion as placed. An egg that was lost and is perceived again (location not lost_unknown) is a normal pending egg: plan_grasp it, do not search for it again.
- V3.39 USB GRASP HEIGHT: with the RGB-only wrist camera plan_grasp itself takes up to two extra close views when the target is not yet measured/consistent. If plan_grasp fails with GRASP_HEIGHT_IMPLAUSIBLE (perceived centre not at table + half width), call observe_scene once and plan_grasp the same egg again; if it repeats, try the next pending egg, then report.
- V3.37 SAFETY + LOST EGGS: after a safety stop the held egg stays in the gripper (the sim keeps the hold); return_to_pick_location clears the safety latch itself (STOP + RECOVER) before it moves - just call it. orient_for_slot may first carry the egg to the middle of the table when the reorientation is risky (V3.37 RECENTER); that is normal. If a result reports v337_egg_lost or an egg has location=lost_unknown, its position is UNKNOWN: call observe_scene (it searches around the drop point) and do not plan_grasp that egg until it is perceived again (plan_grasp refuses it with EGG_LOST_NOT_RELOCATED). After two searches without finding it, continue with the other eggs and report the lost egg. Never claim a lost egg is back in the basket.
- V3.35 MEASURED, LIVE SCENE: with a wrist camera every egg position is triangulated from several views and its size is MEASURED (objects.<egg>.perception.measurement: width_mm, length_mm); nothing about the eggs is assumed. YOU judge plausibility: a hen's egg is roughly 38-50 mm wide and 50-66 mm long; up to ~15% measurement error is normal. A null width_mm (size not measured, e.g. eggs touching each other) is NOT a reason to skip an egg - the backend then uses a conservative aperture. If an object is clearly outside that (e.g. < 30 mm or > 60 mm wide) do not grasp it and mention it in the report. An egg that is seen but not yet trusted (untrusted_reason "not triangulated yet") gets a second look inside plan_grasp automatically. The scene is re-perceived continuously: TARGET_MOVED_SINCE_PLAN -> call plan_grasp for that egg again; SLOT_MOVED_SINCE_PLAN / SLOT_MOVED_DURING_INSERTION / HOLDER_POSSIBLY_MOVED(_DURING_INSERTION) -> the egg is still held, call plan_placement again (it looks at the carton first; any free slot); a grasp / placement uses the values it was planned with (frozen) - a size that changes in later views does not matter until the next plan; IN_HAND_SHIFT_DETECTED -> stop and report; PERCEPTION_UNAVAILABLE -> stop and report at once (there is no fallback).
- V3.33 NO CARTON: if observe_scene reports holder_not_perceived (receptacles.egg_holder.slots empty), call observe_scene at most once more; if the carton is still not perceived, stop and report that the egg carton could not be found. Never pick up an egg without a perceived destination slot.
- V3.32 PERCEIVED OBJECTS ONLY: if scene.objects_policy=perceived_only, the eggs listed in scene.objects are ALL the eggs that exist in this episode (the number of eggs is not known in advance; one may have been removed or dropped). "All eggs" means all listed eggs. Never ask for or plan an egg name that is not listed; if plan_grasp returns OBJECT_NOT_PERCEIVED_IN_THIS_EPISODE, finish the listed eggs and report.
- V3.11 CAMERA PERCEPTION: object poses may come from a camera (scene.pose_source=camera_*_perception: fixed D435i RGB-D, wrist D435i or wrist USB RGB-only). Each egg then carries perception (seen, trusted, confidence); never invent or adjust coordinates yourself. If plan_grasp fails with PERCEPTION_TARGET_UNTRUSTED (it already re-looked once from a clear posture), try the next pending egg once, then report. If verify_object_in_slot fails with PERCEPTION_COULD_NOT_CONFIRM, call observe_scene once and judge from its slot occupancy; if a tool reports PERCEPTION_UNAVAILABLE / PERCEPTION_TIMEOUT / PERCEPTION_ERROR twice, stop and report.
- V3.12.1 HELD-EGG RECOVERY: use lift height_m 0.10-0.11 (the backend lowers the height by itself if a corridor is blocked; v3121_lift_heights_tried_mm shows that). orient_for_slot raises a low egg by itself before reorienting. If lift fails with V3121_LIFT_CORRIDOR_BLOCKED, or orient_for_slot fails with V224_ALL_ORIENTATION_STRATEGIES_FAILED (or recommends return_to_pick_location), do NOT try more lifts or more slots: call return_to_pick_location(object_name) for the held egg. On success the egg is back in the basket and the gripper is free: continue with the other pending eggs (v3121_next.try_next) and retry the returned egg once at the end. If return_to_pick_location fails, stop and report. In camera mode a held egg is reported with location=in_gripper (pose from the TCP, the camera cannot see it).
- V3.07 START-STATE RESET: plan_grasp may move the arm by itself between eggs: it can return the empty arm to SAFE_NEUTRAL q0 before planning (v307_post_place_posture) and, if every candidate fails from the current posture, it resets to q0 and re-plans once (v307_start_state_reset). If the result still fails with v307_start_state_reset.replanned_from_q0=true, that egg is infeasible on the goal side: try the next pending egg once, then report. If reason=START_STATE_PLANNING_FAILED, stop and report immediately. Never call observe_scene repeatedly to retry a failed plan.
- V3.05 EXECUTION_LAYER_FAULT: reason=EXECUTION_LAYER_FAULT means the controller/MuJoCo execution layer misbehaved (goal cancelled/rejected, huge tracking mismatch, duplicate controller). It is NOT a grasp, candidate, aperture or egg problem. The backend has already returned the arm to the safe neutral posture. Do NOT switch to another egg and do NOT change the gripper opening because of it. If recommended_next_actions contains retry_same_step_once: observe_scene, then redo the same step for the SAME egg once (call plan_grasp again before move_to_pregrasp). If it contains stop_and_report (or the same fault repeats), stop the task immediately and report the execution-layer fault; the task supervisor will halt anyway.

Operating policy:
-1. V2-7.1 EGG PLACEMENT OVERRIDE: after lift of an egg, use plan_placement -> orient_for_slot -> move_to_slot_entry -> guarded_insert_and_release -> verify_object_in_slot. Never use legacy move_above_receptacle/place_into_receptacle/open_gripper for V2-7.1 egg placement.
0. V2-27 EGG PICKUP OVERRIDE: keep the proven V2-24 pickup path unchanged: after mandatory observe_scene and aperture preparation, use plan_grasp -> move_to_pregrasp -> approach_planned_grasp. If plan_grasp explicitly exposes edge_reposition_available=true, call reposition_egg_inward once, then observe_scene and re-plan instead of ending the task.
1. The first action for every physical task MUST be observe_scene.
2. Exact egg names: egg_1, egg_2, egg_3, egg_4.
3. Exact receptacle name: egg_holder. Exact slots: slot_1 through slot_8.
4. For requests like "the eggs" or "all eggs", process every egg still outside egg_holder, one complete cycle at a time.
5. Choose an EMPTY slot from scene.receptacles.egg_holder.slots. V3.04 FILL ORDER: the holder is filled in numeric order. Use scene.egg_holder_fill_order.next_empty_slot (the lowest-numbered empty slot). Phrases like "first/second/third/fourth empty slot", "next slot" or "an empty slot" mean the NEXT slot in this fill order, NOT the N-th slot counted among the currently empty ones. Only an explicit slot name (e.g. slot_6) overrides this.
6. V3.06 STEP ECONOMY: one egg = plan_grasp -> move_to_pregrasp -> approach_planned_grasp -> clamp_gripper -> lift -> plan_placement -> orient_for_slot -> move_to_slot_entry -> guarded_insert_and_release -> verify_object_in_slot (10 tools). plan_grasp sets the aperture itself. After a SUCCESSFUL verify_object_in_slot, read v306_next (remaining eggs, next_empty_slot) and call plan_grasp(v306_next.next_egg) directly; do NOT call observe_scene between eggs. Observe only after a failure or when state is unclear. Each tool result may carry v306_budget; if steps_left is too small for another full egg (< 10), finish the egg in hand and stop.
7. Normal cycle: move_above -> approach -> clamp_gripper -> lift -> move_above_receptacle -> place_into_receptacle -> open_gripper -> verify_object_in_slot.
8. Do not use open_gripper for pre-grasp aperture preparation.
9. After every tool, read state_after, physical_verification, warnings, and recommended_next_actions before continuing.
10. Do NOT expect egg motion from move_above or approach. Egg motion becomes required at lift.
11. After release, verify the exact egg-holder slot before starting the next egg.
12. If normalized ok=false, do not blindly repeat the same physical action. Re-observe and reason from the reported failure.
13. PRE-GRASP SAFETY POLICY remains V2-7: if approach reports PREGRASP_SAFETY_STOP_REQUIRES_REFLECTION and retry_allowed=true, observe, diagnose, call recover_from_safety once, re-observe, materially adapt and retry once. Never loop.
14. PLANNING-BLOCK RECOVERY remains V2-7: if approach reports PREGRASP_PATH_BLOCKED_REQUIRES_ADAPTATION and planning_retry_allowed=true, follow the backend's proposed adaptation, re-observe and retry once.
15. V69 safety remains authoritative. Never continue physical motion while its latch is held.
16. Do not infer egg fragility behavior yet. The current eggs are rigid baseline objects; tactile/damage modeling comes later.
17. Never call multiple physical-motion tools in parallel.
18. A fresh confirmed RUN TASK with no held object begins from logical IDLE.
19. Keep final completion messages concise and factual.
"""

_EGGS = ["egg_1", "egg_2", "egg_3", "egg_4"]
_SLOTS = [f"slot_{i}" for i in range(1, 9)]

TOOLS = [

    {
        "type": "function",
        "name": "plan_placement",
        "description": "V2-7.1 PLACEMENT PLANNING — After lift, plan placement of the held egg into a selected empty egg-holder slot. Computes held egg LONG-axis alignment, safe high reorientation workspace, precise slot-entry height, and pre-release capture depth. Planning only; nothing moves.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "enum": ["egg_1", "egg_2", "egg_3", "egg_4"]},
                "receptacle_name": {"type": "string", "enum": ["egg_holder"]},
                "slot_name": {"type": "string", "enum": ["slot_1", "slot_2", "slot_3", "slot_4", "slot_5", "slot_6", "slot_7", "slot_8"]}
            },
            "required": ["object_name", "receptacle_name", "slot_name"],
            "additionalProperties": False
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "orient_for_slot",
        "description": "V2-7.1 HELD-OBJECT REORIENTATION — Move the held egg to a high safe workspace and rotate at approximately fixed TCP XYZ until the egg LONG axis aligns with the selected slot entry axis. BioIK is current-state seeded; wrist J6/J7 are preferred by minimum displacement but are not manually commanded.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "enum": ["egg_1", "egg_2", "egg_3", "egg_4"]},
                "slot_name": {"type": "string", "enum": ["slot_1", "slot_2", "slot_3", "slot_4", "slot_5", "slot_6", "slot_7", "slot_8"]}
            },
            "required": ["object_name", "slot_name"],
            "additionalProperties": False
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "move_to_slot_entry",
        "description": "V2-7.1 PRECISE SLOT ENTRY — With the egg already long-axis aligned, transport it above the selected slot while preserving orientation. Requires <=7 mm egg-center XY error before insertion and allows at most one deterministic micro-correction.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "enum": ["egg_1", "egg_2", "egg_3", "egg_4"]},
                "slot_name": {"type": "string", "enum": ["slot_1", "slot_2", "slot_3", "slot_4", "slot_5", "slot_6", "slot_7", "slot_8"]}
            },
            "required": ["object_name", "slot_name"],
            "additionalProperties": False
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "guarded_insert_and_release",
        "description": "V2-7.1 SLOW AXIAL INSERTION + RELEASE — From precise slot entry, descend the held egg along its LONG axis in bounded 10 mm then 5 mm Cartesian segments with hold/safety/XY/orientation checks. Stop at a pre-release capture depth above the final slot center, open the gripper in two stages, retreat vertically, then call verify_object_in_slot.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "enum": ["egg_1", "egg_2", "egg_3", "egg_4"]},
                "slot_name": {"type": "string", "enum": ["slot_1", "slot_2", "slot_3", "slot_4", "slot_5", "slot_6", "slot_7", "slot_8"]}
            },
            "required": ["object_name", "slot_name"],
            "additionalProperties": False
        },
        "strict": True,
    },

    {
        "type": "function",
        "name": "reposition_egg_inward",
        "description": "V2-27 EDGE RECOVERY — For an egg proven basket-edge-limited, use a wide-open finger to nudge it inward in small feedback-controlled segments. The skill re-reads egg position after each segment, stops once enough inward gain is measured, retreats high, and restores a high-neutral posture with slow q=0 fallback. One attempt per egg. After success, MUST call observe_scene before setting aperture/planning a new grasp.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "enum": ["egg_1", "egg_2", "egg_3", "egg_4"]}
            },
            "required": ["object_name"],
            "additionalProperties": False
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "plan_grasp",
        "description": "V2-7 CONSTRAINT-AWARE PLANNING — For an egg in the basket, generate bounded grasp candidates from egg orientation, gripper aperture and basket-rim clearance, then MoveIt-preflight candidate pregrasp poses WITHOUT moving the robot. Returns candidate IDs plus recommended_candidate_id. Mandatory before V2-7 egg pickup. V3.06: automatically sets the recommended gripper aperture first if needed (no separate set_gripper_opening call).",
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "enum": ["egg_1", "egg_2", "egg_3", "egg_4"]}
            },
            "required": ["object_name"],
            "additionalProperties": False
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "move_to_pregrasp",
        "description": "V2-7 CANDIDATE EXECUTION — Move the TCP to a specific candidate pregrasp pose returned by the latest plan_grasp. Applies candidate wrist yaw and small center offset while preserving the V69 safety/motion stack. Object should remain stationary.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "enum": ["egg_1", "egg_2", "egg_3", "egg_4"]},
                "candidate_id": {"type": "string", "minLength": 3}
            },
            "required": ["object_name", "candidate_id"],
            "additionalProperties": False
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "approach_planned_grasp",
        "description": "V2-7 CANDIDATE DESCENT — From the reached candidate pregrasp, execute a collision-checked straight descent to that candidate's grasp pose with fixed wrist yaw. If blocked/contacting, the candidate is exhausted and must not be repeated; re-observe and plan another candidate.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "enum": ["egg_1", "egg_2", "egg_3", "egg_4"]},
                "candidate_id": {"type": "string", "minLength": 3}
            },
            "required": ["object_name", "candidate_id"],
            "additionalProperties": False
        },
        "strict": True,
    },
    {
        "type": "function", "name": "observe_scene",
        "description": "Read the semantic MuJoCo scene: eggs, egg_basket, egg_holder/8 slot occupancy, TCP, continuous gripper opening, per-egg manipulation prior, hold and safety. Observation only; nothing moves. ALWAYS use first.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "return_to_pick_location",
        "description": "V3.12.1 recovery: put the HELD egg back where it was picked (straight down at the pick pose, open to the pre-grasp aperture, retreat). Use when lift reports V3121_LIFT_CORRIDOR_BLOCKED or orient_for_slot cannot reorient the egg. Afterwards the gripper is free; continue with other eggs.",
        "parameters": {"type": "object", "properties": {"object_name": {"type": "string", "enum": _EGGS}},
                       "required": ["object_name"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "recover_from_safety",
        "description": "Explicit one-shot recovery after a latched PRE-GRASP safety stop. Observe/diagnose first; recovery retreats and consumes the one retry budget for that egg.",
        "parameters": {"type": "object", "properties": {
            "object_name": {"type": "string", "enum": _EGGS},
            "strategy": {"type": "string", "enum": ["widen_aperture_and_retry_once", "retreat_and_retry_once"]},
            "diagnosis": {"type": "string", "minLength": 8}},
            "required": ["object_name", "strategy", "diagnosis"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "set_gripper_opening",
        "description": "Set continuous pre-grasp gripper opening in millimeters. Use only when no object is held; then re-observe actual opening.",
        "parameters": {"type": "object", "properties": {"target_opening_mm": {"type": "number", "minimum": 5.0, "maximum": 100.0}}, "required": ["target_opening_mm"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "move_above",
        "description": "Move TCP to a safe hover above the named egg. Requires a prepared robust aperture. Egg should remain stationary.",
        "parameters": {"type": "object", "properties": {"object_name": {"type": "string", "enum": _EGGS}}, "required": ["object_name"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "approach",
        "description": "Pre-grasp descent to the selected egg using unchanged V69 motion. Egg should remain approximately stationary; pre-grasp poke/path-block failures are bounded recoverable failures.",
        "parameters": {"type": "object", "properties": {"object_name": {"type": "string", "enum": _EGGS}}, "required": ["object_name"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "clamp_gripper",
        "description": "Close the two-finger gripper and establish provisional contact/hold. Current egg baseline is rigid; force is not yet a shell-damage model.",
        "parameters": {"type": "object", "properties": {"force_n": {"type": "number", "minimum": 0.5, "maximum": 15.0}}, "required": ["force_n"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "lift",
        "description": "Lift the grasped egg vertically; the egg MUST follow upward to validate the grasp. Use 0.10-0.11 m; the backend lowers the height automatically if the corridor is blocked.",
        "parameters": {"type": "object", "properties": {"height_m": {"type": "number", "minimum": 0.04, "maximum": 0.15}}, "required": ["height_m"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "move_above_receptacle",
        "description": "Transport the held egg above a selected EMPTY egg-holder slot. The grasp must remain active.",
        "parameters": {"type": "object", "properties": {
            "receptacle_name": {"type": "string", "enum": ["egg_holder"]},
            "slot_name": {"type": "string", "enum": _SLOTS}},
            "required": ["receptacle_name", "slot_name"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "place_into_receptacle",
        "description": "Lower the still-held egg from slot hover into the selected egg-holder slot placement region. This does not release the egg.",
        "parameters": {"type": "object", "properties": {
            "receptacle_name": {"type": "string", "enum": ["egg_holder"]},
            "slot_name": {"type": "string", "enum": _SLOTS}},
            "required": ["receptacle_name", "slot_name"], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "open_gripper",
        "description": "Release-only skill after SLOT_PLACE_REACHED. Opens the gripper and detaches the held egg; does not itself prove slot placement.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}, "strict": True,
    },
    {
        "type": "function", "name": "verify_object_in_slot",
        "description": "Read-only deterministic check that the named egg is actually in the requested egg-holder slot and is no longer held. V3.06: on success also returns v306_next = remaining eggs, next_egg, next_empty_slot (no observe_scene needed before the next egg).",
        "parameters": {"type": "object", "properties": {
            "object_name": {"type": "string", "enum": _EGGS},
            "slot_name": {"type": "string", "enum": _SLOTS}},
            "required": ["object_name", "slot_name"], "additionalProperties": False}, "strict": True,
    },
]


class LLMAPIError(RuntimeError):
    pass


class NeroLLMAgent:
    # V3.42 defaults (also for instances built without __init__ in tests)
    provider = None
    _model_hint = DEFAULT_MODEL
    _v342_llm_time = 0.0
    step_extension_cb = None
    _chat_seq = 0

    def __init__(self, emit, tool_runner, api_base=DEFAULT_API_BASE):
        self.emit = emit
        self.tool_runner = tool_runner
        self.api_base = str(api_base).rstrip("/")
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self._last_response_id = None
        self._last_final_text = ""
        self._usage_totals = {"input_tokens": 0, "output_tokens": 0, "estimated_usd": 0.0, "requests": 0,
                              "cached_input_tokens": 0}
        self.provider = None                 # V3.42: "openai" / "deepseek" / "kimi"; None = from the model ID
        self._model_hint = DEFAULT_MODEL
        self._chat_hist = {}                 # V3.42: response id -> chat messages (Chat Completions providers)
        self._chat_seq = 0
        self.step_extension_cb = None        # V3.42: GUI callback(info) -> extra steps (0 = no)
        self._v342_llm_time = 0.0

    @property
    def running(self):
        with self._lock:
            return bool(self._running)

    @property
    def last_response_id(self):
        return self._last_response_id

    def request_stop(self):
        self._stop_event.set()
        self.emit("[LLM STOP] Stop requested; no new LLM tool calls will be executed")

    def reset_context(self):
        if self.running:
            raise RuntimeError("Cannot reset LLM context while task is running")
        self._last_response_id = None
        self._last_final_text = ""
        self._stop_event.clear()
        self.emit("[LLM CONTEXT] Previous response/context cleared")

    def _prov(self, model=None):
        return provider_for_model(model or self._model_hint, self.provider)

    def _prov_cfg(self, model=None):
        return LLM_PROVIDERS[self._prov(model)]

    def _base(self, model=None):
        p = self._prov(model)
        return self.api_base if p == "openai" else LLM_PROVIDERS[p]["api_base"]

    def api_key_status(self, api_key="", model=None):
        cfg = self._prov_cfg(model)
        key = (api_key or next((os.environ.get(e, "") for e in cfg["key_env"] if os.environ.get(e)), "")).strip()
        if not key:
            return False, f"{cfg['label']}: {' / '.join(cfg['key_env'])} is not configured"
        return True, f"{cfg['label']} API key configured (value hidden)"

    def _resolve_key(self, api_key="", model=None):
        cfg = self._prov_cfg(model)
        key = (api_key or next((os.environ.get(e, "") for e in cfg["key_env"] if os.environ.get(e)), "")).strip()
        if not key:
            raise LLMAPIError(
                f"{cfg['label']} API key missing ({' / '.join(cfg['key_env'])}). Paste it in the NERO LLM panel "
                "or export it before launch."
            )
        return key

    def _post_responses(self, payload, api_key, timeout_s=180.0):
        """V3.38 TRANSIENT NETWORK RETRY: DNS failure / connection reset / timeout / HTTP 429 or 5xx are retried with
        back-off (NERO_V338_API_RETRY_DELAYS_S, default 2,4,8,16 s ~ 30 s) before the task fails. The same request
        (same previous_response_id) is sent again, so the Agent continues exactly where it was. A STOP ends the wait."""
        try:
            delays = [float(x) for x in os.environ.get("NERO_V338_API_RETRY_DELAYS_S", "2,4,8,16").split(",") if x.strip()]
        except ValueError:
            delays = [2.0, 4.0, 8.0, 16.0]
        attempt = 0
        while True:
            try:
                t342 = time.perf_counter()
                try:
                    self._model_hint = str(payload.get("model") or self._model_hint)
                    if LLM_PROVIDERS[self._prov(payload.get("model"))]["protocol"] == "chat":
                        return self._post_chat_once(payload, api_key, timeout_s)
                    return self._post_responses_once(payload, api_key, timeout_s)
                finally:
                    self._v342_llm_time += time.perf_counter() - t342
            except LLMAPIError as exc:
                msg = str(exc)
                transient = ("network error" in msg or "timed out" in msg
                             or any(f"HTTP {c}" in msg for c in (408, 409, 429, 500, 502, 503, 504)))
                if not transient or attempt >= len(delays) or self._stop_event.is_set():
                    raise
                d = delays[attempt]
                attempt += 1
                self.emit(f"[V3.38 API RETRY] {msg[:160]} -> retry {attempt}/{len(delays)} in {d:g}s (task continues)")
                if self._stop_event.wait(d):
                    raise

    # ------------------------------------------------------------ V3.42 Chat Completions adapter
    @staticmethod
    def _chat_tools(tools):
        out = []
        for t in tools or []:
            if t.get("type") != "function":
                continue
            out.append({"type": "function", "function": {
                "name": t.get("name"), "description": t.get("description", ""),
                "parameters": t.get("parameters") or {"type": "object", "properties": {}}}})
        return out

    def _chat_extra(self, prov, effort):
        """Provider-specific reasoning switches. DeepSeek: thinking on/off + reasoning_effort (low/high/max).
        Kimi: nothing by default (model defaults); NERO_KIMI_EXTRA_BODY / NERO_DEEPSEEK_EXTRA_BODY (JSON) override."""
        extra = {}
        e = str(effort or "").lower()
        if prov == "deepseek":
            if e == "none":
                extra["thinking"] = {"type": "disabled"}
            elif e:
                extra["thinking"] = {"type": "enabled"}
                extra["reasoning_effort"] = {"minimal": "low", "low": "low", "max": "max"}.get(e, "high")
        env = os.environ.get("NERO_DEEPSEEK_EXTRA_BODY" if prov == "deepseek" else "NERO_KIMI_EXTRA_BODY", "")
        if env.strip():
            try:
                extra.update(json.loads(env))
            except Exception:
                pass
        return extra

    def _post_chat_once(self, payload, api_key, timeout_s=180.0):
        model = str(payload.get("model"))
        prov = self._prov(model)
        cfg = LLM_PROVIDERS[prov]
        key = self._resolve_key(api_key, model)
        prev = payload.get("previous_response_id")
        if not isinstance(self.__dict__.get("_chat_hist"), dict):
            self._chat_hist = {}
        msgs = [dict(m) for m in self._chat_hist.get(prev, [])] if prev else []
        sysmsg = {"role": "system", "content": str(payload.get("instructions") or "")}
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = sysmsg
        else:
            msgs.insert(0, sysmsg)
        inp = payload.get("input")
        if isinstance(inp, str):
            msgs.append({"role": "user", "content": inp})
        else:
            for it in inp or []:
                if it.get("type") == "function_call_output":
                    msgs.append({"role": "tool", "tool_call_id": str(it.get("call_id")), "content": str(it.get("output"))})
                elif it.get("role"):
                    msgs.append({"role": it["role"], "content": str(it.get("content", ""))})
        body = {"model": model, "messages": msgs,
                "max_tokens": max(int(payload.get("max_output_tokens") or 0),
                                  int(os.environ.get("NERO_V342_CHAT_MAX_TOKENS", "8192")))}
        tools = self._chat_tools(payload.get("tools"))
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        body.update(self._chat_extra(prov, (payload.get("reasoning") or {}).get("effort")))
        req = urllib.request.Request(
            cfg["api_base"] + "/chat/completions",
            data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                     "User-Agent": "NERO-V3.42"},
            method="POST")
        name = cfg["label"]
        try:
            with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            raise LLMAPIError(f"{name} API HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:3000]}") from exc
        except urllib.error.URLError as exc:
            raise LLMAPIError(f"{name} API network error: {exc}") from exc
        except (TimeoutError, ConnectionError) as exc:
            raise LLMAPIError(f"{name} API request timed out / connection error: {exc}") from exc
        try:
            obj = json.loads(raw)
        except Exception as exc:
            raise LLMAPIError(f"{name} API returned non-JSON response") from exc
        if obj.get("error"):
            raise LLMAPIError(f"{name} API error: " + json.dumps(obj["error"], ensure_ascii=False))
        msg = ((obj.get("choices") or [{}])[0].get("message")) or {}
        keep = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            keep["tool_calls"] = msg["tool_calls"]
        if msg.get("reasoning_content"):
            keep["reasoning_content"] = msg["reasoning_content"]   # thinking + tools: must be sent back
        msgs.append(keep)
        self._chat_seq += 1
        rid = f"{prov}_{self._chat_seq}"
        self._chat_hist[rid] = msgs
        for old in [k for k in self._chat_hist if k != rid and k != prev]:
            self._chat_hist.pop(old, None)
        output = []
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            output.append({"type": "function_call", "name": fn.get("name"),
                           "arguments": fn.get("arguments") or "{}", "call_id": tc.get("id")})
        if msg.get("content"):
            output.append({"type": "message", "content": [{"type": "output_text", "text": msg.get("content")}]})
        u = obj.get("usage") or {}
        cached = int((u.get("prompt_tokens_details") or {}).get("cached_tokens")
                     or u.get("prompt_cache_hit_tokens") or u.get("cached_tokens") or 0)
        return {"id": rid, "output": output,
                "usage": {"input_tokens": int(u.get("prompt_tokens") or 0),
                          "output_tokens": int(u.get("completion_tokens") or 0),
                          "input_tokens_details": {"cached_tokens": cached}}}

    def _post_responses_once(self, payload, api_key, timeout_s=180.0):
        key = self._resolve_key(api_key)
        url = self.api_base + "/responses"
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
                "User-Agent": "NERO-V2-7-Public-Release/0.1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:3000]
            raise LLMAPIError(f"OpenAI API HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMAPIError(f"OpenAI API network error: {exc}") from exc
        except (TimeoutError, ConnectionError) as exc:
            raise LLMAPIError(f"OpenAI API request timed out / connection error: {exc}") from exc
        try:
            obj = json.loads(raw)
        except Exception as exc:
            raise LLMAPIError("OpenAI API returned non-JSON response") from exc
        if obj.get("error"):
            raise LLMAPIError("OpenAI API error: " + json.dumps(obj["error"], ensure_ascii=False))
        return obj

    # Dollar pricing changes over time, so the public repository intentionally
    # does NOT hard-code a price table.  If desired, set NERO_LLM_PRICING_FILE
    # to a JSON file with entries of the form:
    #   {"model-id": [input_per_M, cached_input_per_M, output_per_M]}
    # Token counts remain available even when no price file is configured.
    @staticmethod
    def _load_price_map():
        path = os.environ.get("NERO_LLM_PRICING_FILE", "").strip()
        if not path:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            result = {}
            for model_id, values in (raw or {}).items():
                if isinstance(values, (list, tuple)) and len(values) == 3:
                    result[str(model_id)] = tuple(float(v) for v in values)
            return result
        except Exception:
            # Pricing is advisory only; a malformed optional file must never
            # block robot operation.  The GUI will simply report cost unknown.
            return {}

    PRICE_PER_M = _load_price_map.__func__()

    def list_available_models(self, api_key=""):
        """Return model IDs visible to the user's API key via GET /v1/models."""
        key = self._resolve_key(api_key)
        req = urllib.request.Request(
            self._base() + "/models",
            headers={
                "Authorization": "Bearer " + key,
                "User-Agent": "NERO-V2-7-Public-Release/0.1.0",
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:3000]
            raise LLMAPIError(f"OpenAI API HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMAPIError(f"OpenAI API network error: {exc}") from exc
        obj = json.loads(raw)
        ids = sorted({str(x.get("id")) for x in (obj.get("data") or []) if x.get("id")})
        self.emit(f"[MODEL DISCOVERY] API key exposes {len(ids)} model IDs")
        return ids

    @staticmethod
    def _approx_tokens(text):
        # Local no-cost approximation; intentionally simple and conservative.
        return max(1, int((len(str(text)) + 3) / 4))

    @staticmethod
    def _task_object_names(task, scene_snapshot=None):
        """Resolve canonical egg objects still pending from household language.

        This is the deterministic grounding/completion guard for the current
        egg-task scene.  The LLM chooses skills/slots, but cannot declare a
        plural egg task complete while requested eggs remain outside egg_holder.
        """
        t = str(task).lower().replace("egg 1", "egg_1").replace("egg 2", "egg_2").replace("egg 3", "egg_3").replace("egg 4", "egg_4")
        canonical = ["egg_1", "egg_2", "egg_3", "egg_4"]
        names = [n for n in canonical if n in t]
        if names:
            requested = names
        elif any(p in t for p in ("all eggs", "four eggs", "the eggs", "eggs", "one by one")):
            requested = list(canonical)
        elif "egg" in t:
            requested = ["<one_object>"]
        else:
            return ["<one_object>"]

        if scene_snapshot and scene_snapshot.get("objects_policy") == "perceived_only":
            # V3.32: no-prior perception - only eggs perceived in this episode exist (generic plural request ->
            # all perceived eggs; a named egg that was never perceived is not pending, it is reported as missing)
            objs = scene_snapshot.get("objects") or {}
            seen = [n for n in objs if str(n).startswith("egg") and
                    str((objs.get(n) or {}).get("location", "")) not in ("not_perceived", "")]
            requested = [n for n in requested if n in seen] if names else seen
            return [n for n in requested
                    if not str((objs.get(n) or {}).get("location", "")).startswith("egg_holder.slot_")]
        if scene_snapshot and len(requested) > 1:
            objs = scene_snapshot.get("objects") or {}
            pending = []
            for n in requested:
                loc = str((objs.get(n) or {}).get("location", "unknown"))
                if not loc.startswith("egg_holder.slot_"):
                    pending.append(n)
            return pending
        return requested

    def estimate_task_usage(self, task, scene_snapshot, model=DEFAULT_MODEL, reasoning_effort="low", max_tool_steps=60):
        """No-API preflight estimate for token/cost confirmation.

        This is a heuristic, not a billing quote. It estimates an iterative
        function-calling task from current prompt/tool size and number of
        manipulated objects. Dollar range uses current standard short-context
        prices when known.
        """
        object_names = self._task_object_names(task, scene_snapshot)
        n_obj = max(1, len(object_names))
        current_opening = float((scene_snapshot.get("gripper") or {}).get("actual_opening_mm") or 0.0)

        # Normal object cycle: observe + 7 manipulation/verification skills = 8.
        # V2-5 may add set_gripper_opening + re-observe. First object uses actual
        # current aperture; later cycles conservatively assume aperture prep again.
        aperture_adjustments = 0
        for i, name in enumerate(object_names):
            needs_adjust = True
            if i == 0 and name != "<one_object>":
                geom = (((scene_snapshot.get("objects") or {}).get(name) or {}).get("grasp_geometry") or {})
                rr = geom.get("recommended_opening_range_mm") or []
                if len(rr) == 2:
                    needs_adjust = not (float(rr[0]) <= current_opening <= float(rr[1]))
            if needs_adjust:
                aperture_adjustments += 1

        expected_tool_steps = min(int(max_tool_steps), 10 * n_obj + 1)   # V3.06: 10 tools per egg + initial observe
        expected_requests = max(2, expected_tool_steps + 1)

        base_context = SYSTEM_INSTRUCTIONS + json.dumps(TOOLS, ensure_ascii=False) + str(task)
        base_tokens = self._approx_tokens(base_context)
        # Tool outputs / state observations make each continuation larger.
        incremental_tokens = 520
        input_total = sum(base_tokens + incremental_tokens * i for i in range(expected_requests))
        per_req_out = {
            "none": 20, "low": 40, "medium": 90,
            "high": 180, "xhigh": 320, "max": 500,
        }.get(str(reasoning_effort).lower(), 90)
        output_total = expected_requests * per_req_out

        prices = self.PRICE_PER_M.get(str(model))
        likely_cost = None
        conservative_cost = None
        if prices:
            pin, pcached, pout = prices
            # Conservative upper estimate: treat all input as uncached.
            conservative_cost = (input_total * pin + output_total * pout) / 1_000_000.0
            # Likely estimate: first request uncached; ~75% of later repeated input cached.
            first = base_tokens
            later = max(0, input_total - first)
            cached = int(later * 0.75)
            uncached = input_total - cached
            likely_cost = (uncached * pin + cached * pcached + output_total * pout) / 1_000_000.0

        return {
            "model": str(model),
            "reasoning_effort": str(reasoning_effort),
            "objects_estimated": n_obj,
            "aperture_adjustments_estimated": aperture_adjustments,
            "tool_steps_estimated": expected_tool_steps,
            "api_requests_estimated": expected_requests,
            "input_tokens_estimated": int(input_total),
            "output_tokens_estimated": int(output_total),
            "likely_cost_usd": None if likely_cost is None else round(likely_cost, 5),
            "conservative_cost_usd": None if conservative_cost is None else round(conservative_cost, 5),
            "pricing_known": bool(prices),
            "note": "Heuristic preflight estimate only; actual usage depends on model reasoning, retries, caching, and tool outputs.",
        }

    def _log_usage(self, response, model):
        usage = response.get("usage") or {}
        in_tok = int(usage.get("input_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or 0)
        cached = int((usage.get("input_tokens_details") or {}).get("cached_tokens") or 0)
        prices = self.PRICE_PER_M.get(str(model))
        if prices:
            pin, pcached, pout = prices
            uncached = max(0, in_tok - cached)
            estimated = (uncached * pin + cached * pcached + out_tok * pout) / 1_000_000.0
            self._usage_totals["estimated_usd"] += estimated
            price_text = f"estimated_request=${estimated:.5f} | estimated_session=${self._usage_totals['estimated_usd']:.5f}"
        else:
            price_text = "cost=unavailable (unknown model rate)"
        self._usage_totals["requests"] += 1
        self._usage_totals["input_tokens"] += in_tok
        self._usage_totals["output_tokens"] += out_tok
        self._usage_totals["cached_input_tokens"] = self._usage_totals.get("cached_input_tokens", 0) + cached
        self.emit(f"[MODEL USAGE] model={model} | input={in_tok} | cached_input={cached} | output={out_tok} | {price_text} | session_requests={self._usage_totals['requests']}")

    @staticmethod
    def _extract_text(response):
        parts = []
        for item in response.get("output", []) or []:
            if item.get("type") != "message":
                continue
            for content in item.get("content", []) or []:
                if content.get("type") == "output_text" and content.get("text"):
                    parts.append(str(content.get("text")))
                elif content.get("type") == "refusal" and content.get("refusal"):
                    parts.append(str(content.get("refusal")))
        return "\n".join(parts).strip()

    @staticmethod
    def _function_calls(response):
        return [x for x in (response.get("output", []) or []) if x.get("type") == "function_call"]

    def test_connection(self, model=DEFAULT_MODEL, api_key="", reasoning_effort="low"):
        self._model_hint = str(model)
        self.emit(f"[LLM TEST] Sending API connectivity test | provider={self._prov(model)} | model={model}")
        t0 = time.perf_counter()
        payload = {
            "model": str(model),
            "instructions": "Reply with exactly NERO_LLM_READY.",
            "input": "Connectivity test only. Do not call tools.",
            "reasoning": {"effort": str(reasoning_effort)},
            "max_output_tokens": 32,
            "store": True,
        }
        response = self._post_responses(payload, api_key)
        self._log_usage(response, model)
        text = self._extract_text(response)
        self._last_response_id = response.get("id")
        self.emit(
            f"[LLM TEST OK] id={self._last_response_id} | "
            f"elapsed={time.perf_counter()-t0:.2f}s | reply={text!r}"
        )
        return text

    def plan_only(self, task, scene_snapshot, model=DEFAULT_MODEL, api_key="", reasoning_effort="medium"):
        task = str(task).strip()
        if not task:
            raise ValueError("Task is empty")
        self.emit(f"[LLM PLAN ONLY] task={task!r} | model={model} | effort={reasoning_effort}")
        payload = {
            "model": str(model),
            "instructions": (
                SYSTEM_INSTRUCTIONS
                + "\nPLAN-ONLY MODE: Do not call any tools. Return a concise numbered sequence of the robot tools you would call, including verification steps."
            ),
            "input": "TASK:\n" + task + "\n\nCURRENT SCENE SNAPSHOT:\n" + json.dumps(scene_snapshot, ensure_ascii=False, indent=2),
            "reasoning": {"effort": str(reasoning_effort)},
            "max_output_tokens": 1000,
            "store": True,
        }
        t0 = time.perf_counter()
        response = self._post_responses(payload, api_key)
        self._log_usage(response, model)
        text = self._extract_text(response)
        self._last_response_id = response.get("id")
        self._last_final_text = text
        self.emit(f"[LLM PLAN] elapsed={time.perf_counter()-t0:.2f}s\n{text}")
        return text

    def run_task(
        self,
        task,
        model=DEFAULT_MODEL,
        api_key="",
        reasoning_effort="medium",
        max_tool_steps=60,
        dry_run=False,
    ):
        """Execute one natural-language task as a bounded tool-calling episode.

        ``run_task`` is an orchestrator, not a motion controller.  Each tool call
        is serialized, executed by the deterministic backend, and returned to the
        model together with measured postconditions.  The loop is bounded by
        ``max_tool_steps`` and by one-retry recovery budgets enforced downstream.
        """
        task = str(task).strip()
        if not task:
            raise ValueError("Task is empty")
        # ============================================================
        # V3.06 STEP BUDGET PER OBJECT
        # ------------------------------------------------------------
        # V3.05: fixed 48 (hard cap 60) ran out at egg_4 plan_placement (one
        # blocked candidate on egg_1 cost 4 extra steps).  Budget is now at
        # least 14 per requested egg + 8, cap NERO_V306_MAX_TOOL_STEPS_CAP (100).
        # ============================================================
        _cap306 = int(os.environ.get("NERO_V306_MAX_TOOL_STEPS_CAP", "100"))
        _req306 = self._task_object_names(task, None)
        _n306 = 1 if "<one_object>" in _req306 else max(1, len(_req306))
        _auto306 = int(os.environ.get("NERO_V306_STEPS_PER_OBJECT", "14")) * _n306 + int(os.environ.get("NERO_V306_STEPS_MARGIN", "8"))
        _user306 = int(max_tool_steps)
        if os.environ.get("NERO_V342_AUTO_BUDGET", "0") == "1":
            max_tool_steps = max(1, min(_cap306, max(_user306, _auto306)))
            v306_budget_note = f"user={_user306} auto={_auto306} (objects={_n306}) cap={_cap306} -> {max_tool_steps}"
        else:
            # V3.42: the budget is the GUI value (default 60); when it runs out the operator is asked for more
            max_tool_steps = max(1, min(_cap306, _user306))
            v306_budget_note = (f"user={_user306} cap={_cap306} -> {max_tool_steps} (V3.42: the operator is asked for "
                                f"more steps when they run out)")
        self._model_hint = str(model)
        v342_t = {"llm0": self._v342_llm_time, "tools": {}, "objects": {}, "cur_obj": None}
        v342_u0 = dict(self._usage_totals)
        v342_ext = []
        v306_grace_used = 0
        v306_last_stage = None
        v306_holding = False
        v307_infeasible = set()   # eggs proven infeasible after a q0 re-plan

        with self._lock:
            if self._running:
                raise RuntimeError("LLM agent is already running")
            self._running = True

        self._stop_event.clear()
        self._last_response_id = None
        self._last_final_text = ""
        t_task = time.perf_counter()
        tool_count = 0
        multi_object_guard_nudges = 0
        v305_exec_fault_streak = 0
        v305_exec_halt = None

        # V2-9 TASK SUMMARY — deterministic bookkeeping only.
        # Does not alter tool selection, retries, safety, or orchestration.
        v29_tool_history = []
        v29_final_text = ""
        v29_exception_text = None

        def _v29_emit_high_level_summary():
            history = list(v29_tool_history)
            failed = []
            warning_items = []
            verified_objects = set()

            for item in history:
                tool = str(item.get("tool", ""))
                result = item.get("result") or {}
                if not isinstance(result, dict):
                    continue

                ok = bool(result.get("ok", False))
                pv = result.get("physical_verification") or {}
                if not isinstance(pv, dict):
                    pv = {}

                if not ok:
                    reason = (
                        pv.get("reason")
                        or result.get("error")
                        or result.get("reason")
                        or "unspecified failure"
                    )
                    failed.append((tool, str(reason)))

                warnings = result.get("warnings") or []
                if isinstance(warnings, str):
                    warnings = [warnings]
                for warning in warnings:
                    warning_items.append(f"{tool}: {warning}")

                if tool in ("verify_object_in_slot", "verify_object_in_bin") and ok:
                    obj = (
                        result.get("object_name")
                        or pv.get("object_name")
                        or pv.get("object")
                    )
                    if obj:
                        verified_objects.add(str(obj))

            try:
                requested_raw = self._task_object_names(task, None) or []
            except Exception:
                requested_raw = []
            requested = [
                str(x) for x in requested_raw
                if x and not str(x).startswith("<")
            ]

            all_requested_verified = bool(
                requested and set(requested).issubset(verified_objects)
            )
            any_terminal_verified = bool(verified_objects)

            final_lower = str(v29_final_text or "").lower()
            failure_language = any(
                key in final_lower
                for key in (
                    "unable to", "could not complete", "cannot complete",
                    "task failed", "failed to", "not completed",
                    "infeasible", "aborted"
                )
            )

            if v29_exception_text:
                color = "RED"
                status = "FAILED"
            elif all_requested_verified or (not requested and any_terminal_verified):
                if failed or warning_items:
                    color = "YELLOW"
                    status = "SUCCESS WITH WARNINGS"
                else:
                    color = "GREEN"
                    status = "SUCCESS"
            elif failure_language or failed:
                color = "RED"
                status = "FAILED"
            else:
                color = "YELLOW"
                status = "ENDED WITHOUT FULL PHYSICAL VERIFICATION"

            if failed:
                last_failure = f"{failed[-1][0]}: {failed[-1][1]}"
            elif v29_exception_text:
                last_failure = str(v29_exception_text)
            else:
                last_failure = "none"

            outcome = str(v29_final_text or "").strip().replace("\n", " ")
            if not outcome:
                outcome = "No final model text was produced."
            if len(outcome) > 600:
                outcome = outcome[:597] + "..."

            summary = (
                f"[TASK SUMMARY {color}]\n"
                f"STATUS: {status}\n"
                f"TASK: {task}\n"
                f"OUTCOME: {outcome}\n"
                f"TOOLS: {len(history)} total | {len(failed)} failed | "
                f"{len(warning_items)} warnings\n"
                f"PHYSICAL VERIFY: {len(verified_objects)} object(s) verified | "
                f"requested={requested or 'unspecified'}\n"
                f"LAST FAILURE: {last_failure}"
            )
            self.emit(summary)
            return status

        def _v342_ask_more(reason, used, limit, pending=None):
            """V3.42: ask the operator (GUI dialog) how many more tool steps to allow. -> extra steps (0 = no)."""
            cb = self.step_extension_cb
            if cb is None or os.environ.get("NERO_V342_ASK_MORE_STEPS", "1") == "0" or self._stop_event.is_set():
                return 0
            info = {"reason": reason, "steps_used": int(used), "steps_limit": int(limit),
                    "pending_objects": list(pending or []), "holding": bool(v306_holding),
                    "stage": v306_last_stage, "extensions_so_far": list(v342_ext)}
            self.emit(f"[V3.42 STEP LIMIT] {reason} | used {used}/{limit} | pending={pending or []} | asking the operator")
            try:
                extra = int(cb(info) or 0)
            except Exception as exc342:
                self.emit(f"[V3.42 STEP LIMIT] dialog failed: {exc342}")
                extra = 0
            if extra > 0:
                v342_ext.append(extra)
                self.emit(f"[V3.42 STEPS ADDED] +{extra} -> {int(limit) + extra} tool steps (operator)")
            else:
                self.emit("[V3.42 STEPS NOT ADDED] operator declined / no answer -> ending as before")
            return max(0, extra)

        try:
            try:
                ok, key_status = self.api_key_status(api_key, model)
            except TypeError:
                ok, key_status = self.api_key_status(api_key)
            if not ok:
                raise LLMAPIError(key_status)
            self.emit(
                f"[LLM TASK START] provider={self._prov(model)} | model={model} | effort={reasoning_effort} | "
                f"max_tool_steps={max_tool_steps} | dry_run={bool(dry_run)}"
            )
            self.emit(f"[LLM USER TASK] {task}")
            self.emit(f"[V3.06 STEP BUDGET] {v306_budget_note}")

            payload = {
                "model": str(model),
                "instructions": SYSTEM_INSTRUCTIONS,
                "input": task,
                "tools": TOOLS,
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "reasoning": {"effort": str(reasoning_effort)},
                "max_output_tokens": 1200,
                "store": True,
            }

            request_index = 0
            while True:
                if self._stop_event.is_set():
                    raise RuntimeError("LLM task stopped by user")

                request_index += 1
                self.emit(
                    f"[LLM REQUEST #{request_index}] "
                    + ("initial task" if not self._last_response_id else f"continue from {self._last_response_id}")
                )
                t_req = time.perf_counter()
                response = self._post_responses(payload, api_key)
                self._log_usage(response, model)
                self._last_response_id = response.get("id")
                calls = self._function_calls(response)
                text = self._extract_text(response)
                self.emit(
                    f"[LLM RESPONSE #{request_index}] id={self._last_response_id} | "
                    f"elapsed={time.perf_counter()-t_req:.2f}s | tool_calls={len(calls)}"
                )
                if text:
                    self.emit(f"[LLM TEXT] {text}")

                if not calls:
                    # V2-7 deterministic completion guard for plural tasks.
                    # A small/fast model may otherwise summarize after finishing only
                    # one object. Read the scene directly and refuse premature task
                    # completion while requested eggs remain outside egg_holder.
                    requested_all = self._task_object_names(task, None)
                    is_multi_object = bool(len(requested_all) > 1 and "<one_object>" not in requested_all)
                    _left306 = max_tool_steps - tool_count
                    if (is_multi_object and multi_object_guard_nudges < 4 and _left306 < 10
                            and self.step_extension_cb is not None):
                        # V3.42: eggs still pending and too few steps -> ask the operator before giving up
                        try:
                            _snap342 = self.tool_runner("scene_snapshot", {})
                            _pend342 = self._task_object_names(task, (_snap342 or {}).get("scene") or {})
                            _pend342 = [o for o in _pend342 if o not in v307_infeasible]
                        except Exception:
                            _pend342 = []
                        if _pend342:
                            _x342 = _v342_ask_more(f"model wants to stop with {_left306} steps left", tool_count,
                                                   max_tool_steps, _pend342)
                            if _x342 > 0:
                                max_tool_steps += _x342
                                _left306 = max_tool_steps - tool_count
                    if is_multi_object and multi_object_guard_nudges < 4 and _left306 < 10:
                        self.emit(f"[V3.06 GUARD SKIP] only {_left306} tool steps left; not enough for another egg")
                    if is_multi_object and multi_object_guard_nudges < 4 and _left306 >= 10:
                        try:
                            # V3.33: without priors the perceived scene is already current -> read it, do not
                            # re-survey the table (each wrist-camera survey moves the arm for about a minute)
                            guard_scene = None
                            try:
                                snap = self.tool_runner("scene_snapshot", {})
                                if isinstance(snap, dict) and (snap.get("scene") or {}).get("objects_policy") == "perceived_only":
                                    guard_scene = snap.get("scene")
                            except Exception:
                                guard_scene = None
                            if guard_scene is None:
                                guard_result = self.tool_runner("observe_scene", {})
                                guard_scene = guard_result.get("scene") if isinstance(guard_result, dict) else None
                            pending = self._task_object_names(task, guard_scene or {})
                            hp333 = (((guard_scene or {}).get("receptacles") or {}).get("egg_holder") or {}).get("perception") or {}
                            if pending and hp333.get("perceived") is False:
                                self.emit(f"[V3.33 GUARD] egg carton not perceived -> no destination slot; not nudging for {pending}")
                                pending = []
                        except Exception as guard_exc:
                            pending = []
                            self.emit(f"[MULTI-OBJECT GUARD WARN] scene check failed: {guard_exc}")
                        _skip307 = [o for o in pending if o in v307_infeasible]
                        if _skip307:
                            pending = [o for o in pending if o not in v307_infeasible]
                            self.emit(f"[V3.07 GUARD] {_skip307} already re-planned from q0 and infeasible; not nudging for them")
                        if pending:
                            multi_object_guard_nudges += 1
                            self.emit(
                                f"[MULTI-OBJECT GUARD] model attempted to finish early; "
                                f"pending={pending} | nudge={multi_object_guard_nudges}/4"
                            )
                            payload = {
                                "model": str(model),
                                "instructions": SYSTEM_INSTRUCTIONS,
                                "previous_response_id": self._last_response_id,
                                "input": (
                                    "NERO V2-9 deterministic task-completion guard: the user's plural sorting "
                                    f"task is NOT complete. Eggs still outside egg_holder: {pending}. "
                                    "Continue with the next pending egg using the full egg-to-empty-slot cycle. "
                                    "Do not give a final summary while any requested object remains pending. "
                                    "If one object has exhausted its allowed recovery budget, skip that object and "
                                    "continue other pending objects, then report the failed object at the end."
                                ),
                                "tools": TOOLS,
                                "tool_choice": "auto",
                                "parallel_tool_calls": False,
                                "reasoning": {"effort": str(reasoning_effort)},
                                "max_output_tokens": 1200,
                                "store": True,
                            }
                            continue

                    self._last_final_text = text
                    v29_final_text = text
                    self.emit(
                        f"[LLM TASK COMPLETE] tool_steps={tool_count} | "
                        f"elapsed={time.perf_counter()-t_task:.2f}s"
                    )
                    if text:
                        self.emit(f"[LLM FINAL] {text}")
                    return text

                outputs = []
                for call in calls:
                    if self._stop_event.is_set():
                        raise RuntimeError("LLM task stopped by user")
                    tool_count += 1
                    name = str(call.get("name", ""))
                    if tool_count > max_tool_steps:
                        _x342 = _v342_ask_more(f"tool step budget {max_tool_steps} reached (next: {name})",
                                               tool_count - 1, max_tool_steps)
                        if _x342 > 0:
                            max_tool_steps += _x342
                    if tool_count > max_tool_steps:
                        # V3.06 FINISH GRACE: an egg in hand may be finished with
                        # placement-only tools; otherwise end safely.
                        _grace306 = int(os.environ.get("NERO_V306_FINISH_GRACE_STEPS", "6"))
                        _place306 = ("plan_placement", "orient_for_slot", "move_to_slot_entry",
                                     "guarded_insert_and_release", "verify_object_in_slot", "observe_scene",
                                     "return_to_pick_location")
                        _in_hand306 = v306_holding or v306_last_stage in (
                            "LIFTED", "PLACEMENT_ORIENTED", "SLOT_ENTRY_ALIGNED", "RELEASED")
                        if _in_hand306 and name in _place306 and v306_grace_used < _grace306:
                            v306_grace_used += 1
                            self.emit(f"[V3.06 FINISH GRACE] step budget {max_tool_steps} reached; "
                                      f"finishing the egg in hand: {name} ({v306_grace_used}/{_grace306})")
                        else:
                            try:
                                _se306 = self.tool_runner("v306_safe_end", {})
                                self.emit(f"[V3.06 BUDGET SAFE END] {json.dumps((_se306 or {}).get('physical_verification'), ensure_ascii=False)}")
                            except Exception as _e306:
                                self.emit(f"[V3.06 BUDGET SAFE END FAILED] {_e306}")
                            raise RuntimeError(
                                f"LLM exceeded max_tool_steps={max_tool_steps}; stopped before executing another tool "
                                f"(V3.06 safe end executed)"
                            )

                    call_id = str(call.get("call_id", ""))
                    raw_args = call.get("arguments", "{}") or "{}"
                    try:
                        args = json.loads(raw_args)
                        if not isinstance(args, dict):
                            raise ValueError("arguments must be a JSON object")
                    except Exception as exc:
                        result = {"ok": False, "error": f"Invalid tool arguments: {exc}", "raw": raw_args}
                    else:
                        self.emit(
                            f"[LLM CALL #{tool_count}] {name}("
                            + ", ".join(f"{k}={v!r}" for k, v in args.items())
                            + ")"
                        )
                        if dry_run and name not in ("observe_scene", "verify_object_in_slot"):
                            result = {"ok": True, "dry_run": True, "tool": name, "arguments": args}
                            self.emit(f"[LLM DRY RUN] skipped physical tool {name}")
                        else:
                            t_tool = time.perf_counter()
                            try:
                                result = self.tool_runner(name, args)
                                if not isinstance(result, dict):
                                    result = {"ok": bool(result), "result": result}
                            except Exception as exc:
                                # V3.03: never hand the model a bare exception; give it a next step.
                                _m305 = str(exc)
                                if ("EXECUTION_LAYER_FAULT" in _m305 or "rejected trajectory" in _m305
                                        or "rejected sync trajectory" in _m305):
                                    result = {"ok": False, "tool": name, "error": _m305,
                                              "reason": "EXECUTION_LAYER_FAULT",
                                              "recommended_next_actions": ["observe_scene", "retry_same_step_once"],
                                              "hint": "Controller/execution-layer fault, not a grasp problem. Observe, then redo the same step for the SAME egg once."}
                                else:
                                    result = {"ok": False, "tool": name, "error": _m305,
                                              "reason": "BACKEND_EXCEPTION",
                                              "recommended_next_actions": ["observe_scene"],
                                              "hint": "Observe the scene first. If an egg is still held and stage is LIFTED, plan_placement with a different empty slot."}
                            _dt342 = time.perf_counter() - t_tool
                            _o342 = str(args.get("object_name") or v342_t["cur_obj"] or "(scene)")
                            if args.get("object_name"):
                                v342_t["cur_obj"] = str(args.get("object_name"))
                            _tt342 = v342_t["tools"].setdefault(name, [0, 0.0])
                            _tt342[0] += 1
                            _tt342[1] += _dt342
                            v342_t["objects"][_o342] = v342_t["objects"].get(_o342, 0.0) + _dt342
                            self.emit(
                                f"[LLM TOOL RESULT #{tool_count}] {name} | "
                                f"elapsed={time.perf_counter()-t_tool:.2f}s | "
                                + json.dumps(result, ensure_ascii=False, separators=(",", ":"))
                            )

                    v29_tool_history.append({"tool": name, "result": result})
                    if isinstance(result, dict):
                        _sa306 = result.get("state_after") or {}
                        if _sa306.get("stage"):
                            v306_last_stage = str(_sa306.get("stage"))
                        _pv306 = result.get("physical_verification") or {}
                        if isinstance(_pv306, dict) and "hold_active" in _pv306:
                            v306_holding = bool(_pv306.get("hold_active"))
                        if name in ("guarded_insert_and_release", "verify_object_in_slot") and result.get("ok"):
                            v306_holding = False
                        # V3.07: plan_grasp that still fails after a q0 re-plan (or whose
                        # reset failed) is a final verdict for that egg; the completion
                        # guard must not keep nudging the model about it.
                        if name == "plan_grasp" and not result.get("ok"):
                            _r307 = result.get("v307_start_state_reset") or {}
                            if _r307.get("replanned_from_q0") or _r307.get("attempted") or \
                                    _pv306.get("reason") == "START_STATE_PLANNING_FAILED":
                                v307_infeasible.add(str(args.get("object_name", "")))
                    # V3.05 deterministic execution-layer supervisor.
                    if isinstance(result, dict) and result.get("reason") == "EXECUTION_LAYER_FAULT":
                        v305_exec_fault_streak += 1
                        _max305 = int(os.environ.get("NERO_V305_EXEC_FAULT_RETRIES", "1"))
                        if result.get("execution_layer_unhealthy") or v305_exec_fault_streak > _max305:
                            v305_exec_halt = (f"{name}: {result.get('error') or (result.get('physical_verification') or {}).get('execution_layer_fault')}")
                    elif (isinstance(result, dict) and result.get("ok")
                          and name not in ("observe_scene", "set_gripper_opening", "plan_grasp", "plan_placement")):
                        v305_exec_fault_streak = 0
                    outputs.append(
                        {
                            "type": "function_call_output",
                            "call_id": call_id,
                            "output": json.dumps(_v306_with_budget(_v304_compact_for_model(name, result), tool_count, max_tool_steps), ensure_ascii=False),
                        }
                    )

                if v305_exec_halt:
                    text = ("TASK HALTED: execution-layer fault repeated (controller/tracking), not a grasp problem. "
                            f"Last fault: {str(v305_exec_halt)[:300]}. No further eggs were attempted. "
                            "Check for duplicate/stale ros2_control or move_group processes, restart, and rerun.")
                    self._last_final_text = text
                    v29_final_text = text
                    self.emit(f"[V3.05 EXECUTION HALT] streak={v305_exec_fault_streak} | {str(v305_exec_halt)[:300]}")
                    self.emit(f"[LLM FINAL] {text}")
                    return text

                payload = {
                    "model": str(model),
                    "instructions": SYSTEM_INSTRUCTIONS,
                    "previous_response_id": self._last_response_id,
                    "input": outputs,
                    "tools": TOOLS,
                    "tool_choice": "auto",
                    "parallel_tool_calls": False,
                    "reasoning": {"effort": str(reasoning_effort)},
                    "max_output_tokens": 1200,
                    "store": True,
                }

        except Exception as exc:
            v29_exception_text = str(exc)
            self.emit(f"[LLM TASK FAILED] {exc}")
            raise
        finally:
            self.emit(f"[LLM TASK END] total_elapsed={time.perf_counter()-t_task:.2f}s")
            _status342 = "FAILED"
            try:
                _status342 = _v29_emit_high_level_summary() or "FAILED"
            except Exception as _summary_exc:
                self.emit(
                    "[TASK SUMMARY RED]\n"
                    "STATUS: FAILED\n"
                    f"TASK: {task}\n"
                    f"OUTCOME: Task-summary generation failed: {_summary_exc}"
                )
            try:
                self._v342_emit_stats(time.perf_counter() - t_task, v342_t, v342_u0, tool_count, max_tool_steps,
                                      v342_ext, model)
            except Exception as _st342:
                self.emit(f"[V3.42 TASK STATS] could not be computed: {_st342}")
            try:
                if (not str(_status342).startswith("SUCCESS")
                        and os.environ.get("NERO_V342_FAILURE_ANALYSIS", "1") != "0"):
                    if "stopped by user" in str(v29_exception_text or ""):
                        self.emit("[V3.42 FAILURE ANALYSIS] skipped: the task was stopped by the user")
                    elif isinstance(v29_exception_text, str) and "API key missing" in v29_exception_text:
                        self.emit("[V3.42 FAILURE ANALYSIS] skipped: no API key")
                    else:
                        self._v342_failure_analysis(task, model, api_key, reasoning_effort, v29_tool_history,
                                                    v29_final_text, v29_exception_text, _status342)
            except Exception as _fa342:
                self.emit(f"[V3.42 FAILURE ANALYSIS] failed: {str(_fa342)[:300]}")
            with self._lock:
                self._running = False

def _v342_fmt_s(s):
    s = float(s)
    return f"{int(s // 60)}m{s % 60:04.1f}s" if s >= 60 else f"{s:.1f}s"


def _v342_emit_stats(self, total_s, t, u0, tool_count, limit, ext, model):
    """V3.42 TASK STATS: total time; LLM (API) time; each tool's time (count, seconds, share); time per egg;
    tokens used by THIS task (input / cached / output / total, requests)."""
    total_s = max(1e-6, float(total_s))
    llm_s = max(0.0, self._v342_llm_time - float(t.get("llm0", 0.0)))
    tool_s = sum(v[1] for v in t["tools"].values())
    other_s = max(0.0, total_s - llm_s - tool_s)
    pct = (lambda x: f"{100.0 * x / total_s:5.1f}%")
    lines = ["[V3.42 TASK STATS]",
             f"TOTAL TIME : {_v342_fmt_s(total_s)} | tool steps {tool_count}/{limit}"
             + (f" (operator added {'+'.join(str(x) for x in ext)})" if ext else ""),
             f"  LLM thinking (API calls)  {_v342_fmt_s(llm_s):>9} {pct(llm_s)}",
             f"  robot skills (all tools)  {_v342_fmt_s(tool_s):>9} {pct(tool_s)}",
             f"  other (guards, overhead)  {_v342_fmt_s(other_s):>9} {pct(other_s)}",
             "BY SKILL (count | time | share of total):"]
    for name, (n, s) in sorted(t["tools"].items(), key=lambda kv: -kv[1][1]):
        lines.append(f"  {name:<28} x{n:<3} {_v342_fmt_s(s):>9} {pct(s)}")
    if t.get("objects"):
        lines.append("BY OBJECT (robot skill time):")
        for o, s in sorted(t["objects"].items(), key=lambda kv: -kv[1]):
            lines.append(f"  {o:<10} {_v342_fmt_s(s):>9} {pct(s)}")
    tin = self._usage_totals["input_tokens"] - int(u0.get("input_tokens", 0))
    tca = self._usage_totals.get("cached_input_tokens", 0) - int(u0.get("cached_input_tokens", 0))
    tout = self._usage_totals["output_tokens"] - int(u0.get("output_tokens", 0))
    nreq = self._usage_totals["requests"] - int(u0.get("requests", 0))
    usd = self._usage_totals.get("estimated_usd", 0.0) - float(u0.get("estimated_usd", 0.0))
    lines.append(f"TOKENS ({provider_for_model(model, self.provider)} / {model}): total {tin + tout:,} = input {tin:,} "
                 f"(cached {tca:,}) + output {tout:,} | {nreq} API requests"
                 + (f" | est. ${usd:.4f}" if usd > 0 else ""))
    self.emit("\n".join(lines))


def _v342_failure_analysis(self, task, model, api_key, effort, history, final_text, exc_text, status):
    """V3.42: one extra LLM call (no tools, fresh context, compact digest) -> the primary failure cause, short."""
    rows = []
    for i, h in enumerate(history[-70:], 1):
        r = h.get("result") or {}
        if not isinstance(r, dict):
            continue
        pv = r.get("physical_verification") if isinstance(r.get("physical_verification"), dict) else {}
        reason = pv.get("reason") or r.get("reason") or r.get("error") or ""
        warn = r.get("warnings") or []
        rows.append(f"{i}. {h.get('tool')} ok={bool(r.get('ok'))}"
                    + (f" reason={str(reason)[:160]}" if reason else "")
                    + (f" warn={str(warn[0])[:200]}" if warn and not r.get("ok") else ""))
    digest = ("TASK: " + str(task) + "\nSTATUS: " + str(status)
              + ("\nEXCEPTION: " + str(exc_text)[:400] if exc_text else "")
              + "\nFINAL MODEL TEXT: " + str(final_text or "")[:800]
              + "\nTOOL TRACE (oldest first):\n" + "\n".join(rows))
    lang = os.environ.get("NERO_V342_ANALYSIS_LANG", "zh")
    instr = ("You analyse a failed robot pick-and-place episode (NERO arm, eggs -> egg carton slots, LLM-driven skills). "
             "From the trace, name the PRIMARY cause of the failure (the first root cause, not the symptoms), "
             "with the decisive evidence (tool + reason) and one concrete fix direction. "
             + ("Answer in Simplified Chinese. " if lang == "zh" else "Answer in English. ")
             + "At most 3 short lines: 首要原因 / 证据 / 建议. No preamble.")
    payload = {"model": str(model), "instructions": instr, "input": digest,
               "reasoning": {"effort": "low" if str(effort) not in ("none",) else "none"},
               "max_output_tokens": 600, "store": False}
    self.emit("[V3.42 FAILURE ANALYSIS] asking the model for the primary cause ...")
    t0 = time.perf_counter()
    resp = self._post_responses(payload, api_key, timeout_s=90.0)
    self._log_usage(resp, model)
    text = self._extract_text(resp) or "(no answer)"
    self.emit(f"[V3.42 FAILURE ANALYSIS] ({time.perf_counter() - t0:.1f}s)\n{text.strip()}")
    return text


NeroLLMAgent._v342_emit_stats = _v342_emit_stats
NeroLLMAgent._v342_failure_analysis = _v342_failure_analysis

# Backward compatibility for private/local V2-5.x launchers that imported the
# historical class name.  New public code should import ``NeroLLMAgent``.
NeroAstraAgent = NeroLLMAgent

# V2-27 AGENT SEGMENTED EDGE REPOSITION EXTENSION
