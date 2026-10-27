#!/usr/bin/env bash
# Offline release check: integrity, compile, and V2-52.3 + V3.0x behavior markers + offline tests. No robot motion, no ROS needed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
sha256sum --quiet -c SHA256SUMS.txt && echo "[OK] SHA256SUMS match"
python3 -m py_compile NERO_*.py && echo "[OK] py_compile"
B=NERO_V2_7_1_Unified_Backend.py; V=NERO_V71_Hybrid_GPU_Planner_Core.py
for m in 'V2-43 PLAN-GRASP BRANCH + COMPACT PUSHER + FREE-PATH LIFT' 'V2-50 CLEAN V2-43 FORK / NO PHYSICAL BASKET' \
         'V2-52 MUJOCO EGG-HOLDER LID REMOVED' 'V2-52.1 RUN TASK SAFETY-LATCH GATE ONLY'; do
  grep -q "$m" "$B" || { echo "[FAIL] backend marker missing: $m"; exit 1; }
done
grep -q 'V2-51 TIMEOUT CANCEL BEFORE HANDOFF' "$V" || { echo "[FAIL] V71 V2-51 marker missing"; exit 1; }
! grep -q 'V2-53 HIGH-Z CONSTRAINT ORIENTATION STATE MACHINE' "$B" || { echo "[FAIL] V2-53 residue"; exit 1; }
! grep -q 'egg_holder_lid_panel\|egg_holder_lid_inner' "$B" || { echo "[FAIL] holder lid present"; exit 1; }
for m in 'V3.01 DURATION-AWARE EXECUTION TIMEOUT'; do grep -q "$m" "$V" || { echo "[FAIL] V71 missing: $m"; exit 1; }; done
for m in 'V3.01 MOTION DIAGNOSTICS' 'V3.01 TIMEOUT -> HANDOFF CHECK'; do grep -q "$m" "$B" || { echo "[FAIL] backend missing: $m"; exit 1; }; done
for m in 'V3.02 YAW-FREE VERTICAL TARGETS + NON-SINGULAR SAFE-HIGH HOLD' 'V3.02 SLOT-ABOVE REACHABILITY PRECHECK' 'NERO_V302_ALLOW_Q0_FALLBACK'; do grep -q "$m" "$B" || { echo "[FAIL] backend missing: $m"; exit 1; }; done
grep -q 'SLOT_ABOVE_UNREACHABLE_FROM_CURRENT_GRASP' NERO_V2_7_1_Agent.py || { echo "[FAIL] agent V3.02 rule missing"; exit 1; }
! grep -q 'STRATEGY 2: FREE WHOLE-ARM REORIENTATION' "$B" || { echo "[FAIL] old fixed-orientation Strategy 2 still active"; exit 1; }
! grep -q 'deadline = time.time() + 45.0' "$V" || { echo "[FAIL] fixed 45 s deadline still active"; exit 1; }
for m in 'V3.03 DESCENT-CORRIDOR GATE' 'V3.03 LAYERED SLOT-ENTRY DESCENT' 'SLOT_ENTRY_UNREACHABLE' 'V3.03 ALREADY VERTICAL'; do grep -q "$m" "$B" || { echo "[FAIL] backend missing: $m"; exit 1; }; done
grep -q 'GOAL_IK_UNREACHABLE' "$V" || { echo "[FAIL] V71 V3.03 early-stop missing"; exit 1; }
grep -q 'BACKEND_EXCEPTION' NERO_V2_7_1_Agent.py || { echo "[FAIL] agent V3.03 rule missing"; exit 1; }
for m in 'V3.04 EGG APERTURE' 'V3.04 FAST + DESCENT-AWARE GRASP PREFLIGHT' 'V3.04 PREGRASP CACHE' 'V3.04 SLOT REACH SKIPPED' 'V3.04 FILL ORDER' 'V3.04 APPROACH SPEED'; do grep -q "$m" "$B" || { echo "[FAIL] backend missing: $m"; exit 1; }; done
grep -q 'V3.04 COMPACT MODEL-FACING TOOL OUTPUT' NERO_V2_7_1_Agent.py && grep -q 'V3.04 FILL ORDER' NERO_V2_7_1_Agent.py || { echo "[FAIL] agent V3.04 missing"; exit 1; }
for m in 'V3.05 EXECUTION-LAYER HEALTH' '_v305_require_succeeded' 'DUPLICATE_ARM_CONTROLLER'; do grep -q "$m" "$V" || { echo "[FAIL] V71 missing: $m"; exit 1; }; done
for m in 'V3.05 EXECUTION-LAYER FAULT HANDLING' 'NERO_V305_PREGRASP_CACHE' 'V3.05 IK SCREEN' 'retry_same_step_once'; do grep -q "$m" "$B" || { echo "[FAIL] backend missing: $m"; exit 1; }; done
grep -q 'V3.05 EXECUTION HALT' NERO_V2_7_1_Agent.py && grep -q 'V3.05 EXECUTION_LAYER_FAULT' NERO_V2_7_1_Agent.py || { echo "[FAIL] agent V3.05 missing"; exit 1; }
for m in 'V3.06 AUTO APERTURE IN PLAN_GRASP' 'V3.06 VERIFY CARRIES NEXT-STEP STATE' 'V3.06 SAFE END' 'NERO_V306_MAX_TOOL_STEPS_CAP'; do grep -q "$m" "$B" || { echo "[FAIL] backend missing: $m"; exit 1; }; done
for m in 'V3.06 STEP BUDGET PER OBJECT' 'V3.06 FINISH GRACE' 'v306_safe_end' 'V3.06 STEP ECONOMY'; do grep -q "$m" NERO_V2_7_1_Agent.py || { echo "[FAIL] agent missing: $m"; exit 1; }; done
for m in 'V3.07 START-STATE RESET + POST-PLACE POSTURE' '_v307_inner' 'START_STATE_PLANNING_FAILED' 'NERO_V307_POST_PLACE_POSTURE'; do grep -q "$m" "$B" || { echo "[FAIL] backend missing: $m"; exit 1; }; done
for m in 'V3.07 START-STATE RESET' 'v307_infeasible' 'V3.07 GUARD'; do grep -q "$m" NERO_V2_7_1_Agent.py || { echo "[FAIL] agent missing: $m"; exit 1; }; done
for m in 'V3.07 PRE-LAUNCH STALE-PROCESS CLEANUP' 'V3.07 DUPLICATE ARM CONTROLLER' 'NERO_V307_PRELAUNCH_CLEANUP'; do grep -q "$m" start_nero_sim.sh || { echo "[FAIL] sim launcher missing: $m"; exit 1; }; done
bash -n start_nero.sh && bash -n start_nero_sim.sh && echo "[OK] start_nero.sh + start_nero_sim.sh syntax"
for m in 'V3.08 SIMULATED D435i CAMERA' '_v308_camera_supervisor' 'v308_trigger(' 'inject_camera_into_mjcf'; do grep -q "$m" "$B" || { echo "[FAIL] backend missing: $m"; exit 1; }; done
for m in 'V3.08 SIMULATED D435i CAMERA' 'def perceive_eggs' 'def camera_process_main' 'CaptureWriter' 'def remove_small_blobs' 'def points_inside_boxes' '_robot_boxes'; do grep -q "$m" NERO_V308_Camera_Perception.py || { echo "[FAIL] camera module missing: $m"; exit 1; }; done
for m in 'def temporal_fuse' 'def egg_view_check' 'def _confidence' 'def _is_resting' 'overall_trusted' '_temporal_annotate'; do grep -q "$m" NERO_V308_Camera_Perception.py || { echo "[FAIL] camera module missing V3.10: $m"; exit 1; }; done
for m in 'def _v310_wait_tcp_settled(' 'v310_pre_settle'; do grep -q "$m" "$B" || { echo "[FAIL] backend missing: $m"; exit 1; }; done
for m in 'class PerceptionTracker' 'def camera_rig_xml' 'def _post_result' 'VERSION = "V3.11"'; do grep -q "$m" NERO_V308_Camera_Perception.py || { echo "[FAIL] camera module missing V3.11: $m"; exit 1; }; done
for m in 'def v311_capture_sync(' 'V311_MONITOR_BASELINE' 'Object pose source for skills (V3.11)' '_v236_lamp["parent"] = global_log_toolbar'; do grep -qF "$m" "$B" || { echo "[FAIL] backend missing V3.11: $m"; exit 1; }; done
grep -q 'V3.11 CAMERA PERCEPTION' NERO_V2_7_1_Agent.py || { echo "[FAIL] agent V3.11 rule missing"; exit 1; }
grep -q 'NERO_V308_Camera_Perception' start_nero_sim.sh || { echo "[FAIL] sim launcher does not check the camera module"; exit 1; }
grep -q 'V3.12 CONSOLE ENTRY' start_nero.sh && grep -q 'V3.12 SIM LAUNCH CHAIN' start_nero_sim.sh && grep -q 'V3.12 CONSOLE' NERO_V312_Console.py \
  || { echo "[FAIL] V3.12 console / launcher markers missing"; exit 1; }
for m in 'V3.12.1 RECOVERY HELPERS' 'V3.12.1 LIFT FALLBACK' 'V3.12.1 ORIENT PRE-RAISE' 'V3.12.1 DESCENT PRECHECK' 'V3.12.1 RETREAT LADDER' 'if name == "return_to_pick_location":' 'RECAPTURE_CLEAR_VIEW_FAILED' '"location": "in_gripper"'; do grep -qF "$m" "$B" || { echo "[FAIL] backend missing V3.12.1: $m"; exit 1; }; done
grep -q 'V3.12.1 HELD-EGG RECOVERY' NERO_V2_7_1_Agent.py && grep -q '"return_to_pick_location"' NERO_V2_7_1_Agent.py || { echo "[FAIL] agent V3.12.1 missing"; exit 1; }
for m in 'V3.13 REAL ROBOT PIPELINE' 'class RealPipeline' 'def cmd_push(' 'def cmd_monitor(' 'def push_release_decision(' 'pkexec' 'V3.13.1: gs_usb adapters reject restart-ms'; do grep -qF "$m" NERO_V313_Real_Robot.py || { echo "[FAIL] real-robot module missing V3.13: $m"; exit 1; }; done
for m in 'V3.14 JOINT CONTROL' 'class JointPanelUI' 'def build_model(' 'class PoseLibrary' 'def step_test_targets('; do grep -qF "$m" NERO_V314_Joint_Control.py || { echo "[FAIL] joint control module missing V3.14: $m"; exit 1; }; done
for m in 'V3.15 CONSISTENCY + SAFETY' 'class ConsistencyTabUI' 'V3.16 SAFETY ZONE' 'def fk(' 'def path_fence_check(' 'def compare_models(' 'def ik_tool(' 'def point_in_polygon('; do grep -qF "$m" NERO_V315_Consistency_Safety.py || { echo "[FAIL] consistency/safety module missing V3.15: $m"; exit 1; }; done
grep -qF 'V3.16.1 FIXED LABEL' NERO_V314_Joint_Control.py && [ "$(grep -c 'fixed_label(' NERO_V312_Console.py NERO_V314_Joint_Control.py NERO_V315_Consistency_Safety.py NERO_V316_Safety_Editor.py | awk -F: '{s+=$2} END {print s}')" -ge 20 ] || { echo "[FAIL] V3.16.1 fixed live labels missing"; exit 1; }
grep -qF 'set_leader_mode' NERO_V313_Real_Robot.py && grep -qF 'def ctrl_mode_of(' NERO_V313_Real_Robot.py && grep -qF 'leader_joint_states' NERO_V313_Real_Robot.py || { echo "[FAIL] V3.20 teach-mode fix missing"; exit 1; }
grep -qF 'V3.21 LEADER READER' NERO_V313_Real_Robot.py && grep -qF 'def feedback_source(' NERO_V313_Real_Robot.py && grep -qF 'feedback_diag' NERO_V316_Safety_Editor.py || { echo "[FAIL] V3.21 leader reader missing"; exit 1; }
grep -qF 'V3.22 SLOW EXIT' NERO_V313_Real_Robot.py && grep -qF 'def leader_active(' NERO_V313_Real_Robot.py && grep -qF 'V3.22 FAKE-CONTROLLER SYNC' NERO_V313_Real_Robot.py && grep -qF 'def exit_home(' NERO_V316_Safety_Editor.py || { echo "[FAIL] V3.22 teach fixes missing"; exit 1; }
grep -qF 'V3.23 SAFETY RULES' NERO_V315_Consistency_Safety.py && grep -qF 'def wall_boxes(' NERO_V315_Consistency_Safety.py && grep -qF 'def merge_base(' NERO_V315_Consistency_Safety.py && grep -qF 'def on_merge_base(' NERO_V316_Safety_Editor.py || { echo "[FAIL] V3.23 safety rule A missing"; exit 1; }
grep -qF 'V3.24 PARK POSE D' NERO_V314_Joint_Control.py && grep -qF 'def go_park(' NERO_V314_Joint_Control.py && grep -qF '"待断电位置⚠", "park"' NERO_V312_Console.py && grep -qF 'TOP_X = (-0.95, 0.70)' NERO_V316_Safety_Editor.py || { echo "[FAIL] V3.24 park pose / wider view missing"; exit 1; }
grep -qF 'V3.25 HAND-EYE' NERO_V325_Camera.py && grep -qF 'class CameraTabUI' NERO_V325_Camera.py && grep -qF 'CAM.CameraTabUI(' NERO_V312_Console.py && grep -qF 'text="Camera调试"' NERO_V312_Console.py && grep -qF 'text="实机NERO调试"' NERO_V312_Console.py && grep -qF 'NERO_V325_Camera.py' start_nero.sh || { echo "[FAIL] V3.25 camera tab missing"; exit 1; }
grep -qF 'def wrist_camera_process_main(' NERO_V326_Wrist_Camera.py && grep -qF 'def _v326_wrist_capture(' NERO_V2_7_1_Unified_Backend.py && grep -qF '("wrist", "Wrist camera - D435i' NERO_V2_7_1_Unified_Backend.py && grep -qF 'NERO_V326_Wrist_Camera.py' start_nero_sim.sh || { echo "[FAIL] V3.26 wrist camera missing"; exit 1; }
grep -qF '[V3.27 WRIST LOOK SCREEN]' NERO_V2_7_1_Unified_Backend.py && grep -qF 'def look_pose_candidates(' NERO_V326_Wrist_Camera.py || { echo "[FAIL] V3.27 wrist look family missing"; exit 1; }
grep -qF 'class ViewTracker' NERO_V326_Wrist_Camera.py && grep -qF 'def _v328_tag_with_hints(tag' NERO_V2_7_1_Unified_Backend.py && grep -qF 'V3.28 Q0 FLOOR FROM START' NERO_V2_7_1_Unified_Backend.py || { echo "[FAIL] V3.28 wrist tracking / low-pose escape missing"; exit 1; }
grep -qF 'def detect_holder(' NERO_V326_Wrist_Camera.py && grep -qF 'def _v329_scan(' NERO_V2_7_1_Unified_Backend.py && grep -qF 'def _v329_object_pose(' NERO_V2_7_1_Unified_Backend.py && grep -qF 'ovr329 = _v329_read_scene_override()' NERO_V2_7_1_Unified_Backend.py || { echo "[FAIL] V3.29 no-prior wrist scene missing"; exit 1; }
grep -qF 'def detect_holder_rgb(' NERO_V326_Wrist_Camera.py && grep -qF 'def perceive_eggs_rgb(' NERO_V326_Wrist_Camera.py && grep -qF 'V3.30 GENERIC NAMING' NERO_V326_Wrist_Camera.py && grep -qF 'ttk.Combobox(v311_frame' NERO_V2_7_1_Unified_Backend.py && grep -qF 'def _v330_apply_wrist_model_visibility(model):' NERO_V2_7_1_Unified_Backend.py || { echo "[FAIL] V3.30 RGB wrist camera / drop-down / models missing"; exit 1; }
grep -qF 'V3.19 TEACH MODE' NERO_V313_Real_Robot.py && grep -qF 'V3.19 ASYNC SERVICE CALLS' NERO_V313_Real_Robot.py && grep -qF 'V3.19 HOME' NERO_V314_Joint_Control.py && grep -qF 'V3.19 ZONE FILES' NERO_V316_Safety_Editor.py || { echo "[FAIL] V3.19 changes missing"; exit 1; }
grep -qF 'V3.18 TARGET FOLLOW' NERO_V314_Joint_Control.py && grep -qF 'V3.18 SIDE LEGEND' NERO_V316_Safety_Editor.py || { echo "[FAIL] V3.18 changes missing"; exit 1; }
grep -qF 'V3.16 SAFETY EDITOR' NERO_V316_Safety_Editor.py && grep -qF 'class SafetyEditorUI' NERO_V316_Safety_Editor.py && grep -qF 'SE.SafetyEditorUI(' NERO_V312_Console.py && grep -qF 'NERO_V316_Safety_Editor.py' start_nero.sh || { echo "[FAIL] V3.16 safety editor wiring missing"; exit 1; }
grep -qF 'V3.15 REAL SUB-TABS' NERO_V312_Console.py && grep -qF 'def move_and_settle(' NERO_V313_Real_Robot.py && grep -qF 'ExecuteTrajectory' NERO_V313_Real_Robot.py && grep -qF 'NERO_V315_Consistency_Safety.py' start_nero.sh || { echo "[FAIL] V3.15 wiring missing"; exit 1; }
grep -qF 'V3.14 JOINT PANEL' NERO_V312_Console.py && grep -qF 'def move_joints(' NERO_V313_Real_Robot.py && grep -qF 'NERO_V314_Joint_Control.py' start_nero.sh || { echo "[FAIL] V3.14 wiring missing"; exit 1; }
grep -qF 'V3.13 REAL TAB' NERO_V312_Console.py && grep -qF 'NERO_V313_Real_Robot.py' start_nero.sh || { echo "[FAIL] V3.13 console / launcher wiring missing"; exit 1; }
grep -qF 'V3.47.1 CALIB FILES' NERO_V325_Camera.py && grep -qF 'class SilhouetteFit' NERO_V347_Yolo_Perception.py || { echo "[FAIL] V3.47.1 silhouette pose / calib files missing"; exit 1; }
grep -qF 'V3.47 YOLO LOCATE' NERO_V325_Camera.py && grep -qF 'class YoloEDetector' NERO_V347_Yolo_Perception.py && grep -qF 'NERO_V347_Yolo_Perception.py' start_nero.sh || { echo "[FAIL] V3.47 YOLO locate missing"; exit 1; }
echo "[OK] V2-52.3 behavior markers present; no V2-53 residue; V3.01-V3.30 changes present"
for t in tests/test_v3_02_math.py tests/test_v3_02_orient_harness.py tests/test_v3_03_corridor_entry.py tests/test_v3_04.py tests/test_v3_05.py tests/test_v3_06.py tests/test_v3_07.py tests/test_v3_08.py tests/test_v3_10.py tests/test_v3_11.py tests/test_v3_12.py tests/test_v3_12_1.py tests/test_v3_13.py tests/test_v3_14.py tests/test_v3_15.py tests/test_v3_16.py tests/test_v3_16_2.py tests/test_v3_17.py tests/test_v3_19.py tests/test_v3_20.py tests/test_v3_21.py tests/test_v3_22.py tests/test_v3_23.py tests/test_v3_24.py tests/test_v3_25.py tests/test_v3_26.py tests/test_v3_27.py tests/test_v3_28.py tests/test_v3_29.py tests/test_v3_30.py tests/test_v3_31.py tests/test_v3_31_1.py tests/test_v3_32.py tests/test_v3_33.py tests/test_v3_34.py tests/test_v3_35.py tests/test_v3_36.py tests/test_v3_37.py tests/test_v3_38.py tests/test_v3_39.py tests/test_v3_40.py tests/test_v3_41.py tests/test_v3_42.py tests/test_v3_44.py tests/test_v3_45.py tests/test_v3_46.py tests/test_v3_47.py tests/test_v3_47_1.py; do
  if python3 -c "import numpy" 2>/dev/null; then python3 "$t" >/dev/null && echo "[OK] $t" || { echo "[FAIL] $t"; exit 1; }; fi
done
python3 tests/test_v3_12_gui.py | tail -1   # skips without tkinter / DISPLAY
python3 tests/test_v3_13_gui.py | tail -1   # skips without tkinter / DISPLAY
python3 tests/test_v3_14_gui.py | tail -1   # skips without tkinter / DISPLAY
python3 tests/test_v3_15_gui.py | tail -1   # skips without tkinter / DISPLAY
python3 tests/test_v3_16_gui.py | tail -1   # skips without tkinter / DISPLAY
python3 tests/test_v3_16_1_gui.py | tail -1   # skips without tkinter / DISPLAY
python3 tests/test_v3_18_gui.py | tail -1   # skips without tkinter / DISPLAY
python3 tests/test_v3_19_gui.py | tail -1   # skips without tkinter / DISPLAY
python3 tests/test_v3_21_gui.py | tail -1   # skips without tkinter / DISPLAY
python3 tests/test_v3_22_gui.py | tail -1   # skips without tkinter / DISPLAY
python3 tests/test_v3_23_gui.py | tail -1   # skips without tkinter / DISPLAY
python3 tests/test_v3_24_gui.py | tail -1   # skips without tkinter / DISPLAY
python3 tests/test_v3_25_gui.py | tail -1   # skips without tkinter / DISPLAY
python3 tests/test_v3_45_gui.py | tail -1   # skips without tkinter / DISPLAY
python3 tests/test_v3_47_gui.py | tail -1   # skips without tkinter / DISPLAY
python3 tests/test_v3_47_1_gui.py | tail -1   # skips without tkinter / DISPLAY
