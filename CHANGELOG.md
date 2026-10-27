# Changelog

## V3.47.1
- 视觉定位: egg pose from the YOLO silhouette + local table plane (robust plane fit in a 3-6 cm band, other masks /
  invalid depth excluded) + egg size: SilhouetteFit (lying / standing, yaw, IoU pattern search, centroid alignment,
  shape IoU after area matching, mask bleed in px); depth on the egg only for the displayed height, the flat check
  (< 12 mm) and a standing hint.  Trusted: shape IoU >= 0.80, |bleed| <= 8 px, not at the border, YOLO >= 0.5, in the
  zone, plane found, calibration verified.
- Depth resolution 848x480 / 1280x720 and visual preset selectable (default unchanged), applied on connect.
- Calibration: 另存为 / 读取 (default ~/Downloads/cameracalib), timestamped copy on 保存标定, startup fallback to the
  newest copy; check after loading (serial + intrinsics, 5 live marker errors <= 5 mm, or 快速复核); lamp ④ shows it;
  unverified calibration -> locate results not trusted.
- 视觉定位 settings: 保存设置 / 另存为 / 读取 (config/locate_settings.json + copy in ~/Downloads/cameracalib, reference
  image embedded), restored on start.
- Tests: test_v3_47_1.py (P1-P6), test_v3_47_1_gui.py (H1-H4); V3.47 tests updated to the silhouette semantics; GUI
  tests never touch the user's folders (LOCATE_PATH / CALIB_DIR in temp).


## V3.47
- Camera调试 / 视觉定位: egg detection with YOLOE-26 (ultralytics in the console's python) -> new module
  NERO_V347_Yolo_Perception.py: visual prompt from a saved reference (config/yolo_ref.json + .jpg, egg / not-egg
  boxes) or text prompt (mobileclip2_b.ts required next to the model, no download); masks from the polygons.
- 3D: eroded mask, 5-frame depth fusion, spheroid of the real egg size (55 x 42 mm default, +-1.5 mm per semi-axis),
  resting-egg tilt prior; local table height in a ring around each object.
- Geometric check: height above the local table 34-63 mm, fit rms (reject > 4 mm, trust <= 2.5 mm), contact with the
  table, image border, YOLO confidence, safety zone; status egg / untrusted / rejected with reasons; z_table offset
  reported.
- Page: method drop-down (YOLOE 视觉提示 default, YOLOE 文字提示, 几何法 V3.08), model folder (~/Downloads/objectmodels)
  + model, threshold, prompts, egg size, 框选参考 / 保存参考 / 取消, sample saving (config/yolo_samples); table with
  height / YOLO / reason columns; overlay boxes.
- Unchanged: NERO_V308_Camera_Perception.py, simulation / wrist / fixed camera modes, motion.
- Tests: test_v3_47.py (Y1-Y6), test_v3_47_gui.py (G1-G4), helper tests/v347_fake.py (ray-cast scene + fake
  ultralytics); test_v3_25_gui uses the geometric method explicitly.


## V3.46
- Real robot: rviz2 aborted ('Could not load the Qt platform plugin "xcb" in .../cv2/qt/plugins') because pip
  opencv-python sets QT_QPA_PLATFORM_PLUGIN_PATH on import and the console's children inherited it (V3.45 imported cv2
  at console start through the calib page's ArUco check).
- clean_child_env() (NERO_V313_Real_Robot + NERO_V312_Console): cv2 entries removed from QT_QPA_PLATFORM_PLUGIN_PATH /
  QT_PLUGIN_PATH / QT_QPA_FONTDIR for ProcRunner, run() and SimRunner.
- NERO_V325_Camera: _import_cv2() restores the Qt variables; ArUco checked only after a camera connects.
- rviz2 'process has died' is not a driver failure; step_moveit -> AMBER with the RViz exit (run_all continues).
- Tests: test_v3_46.py (E1-E4).


## V3.45
- Camera调试 / 手眼标定: four numbered steps with lamps; buttons enabled by state; marker side presets 40 / 30 mm
  (default 40); sample table with font-sized row height, scrollbars, source + jitter columns, row tags; scrollable
  coloured result box.
- Capture: 15 frames, one source per pose (depth if >= 80 % frames, else PnP), outlier-trimmed jitter <= 3 mm,
  SETTLE_S 1.5 + joint-still check, one re-measure (CAPTURE_RETRIES).
- Auto collection: MarkerModel from the calibration or from 3 x 4 cm probes; plan_auto_targets (box +-12/12/8 cm,
  wrist <= 25 deg, in image, facing <= 45 deg, range, IK, soft limits, fence, farthest-point spread, NN order).
- detect_aruco returns p_pnp, p_depth and normal_cam; rotation_span_deg is pairwise.
- Fix: redraw race on self.det ('NoneType' object is not subscriptable); missing safety zone tolerated.
- Tests: test_v3_45.py (D1-D8), test_v3_45_gui.py (G1-G4).


## V3.44
- Scene chosen before start: drop-down in the NERO console's 仿真 tab (default fill_egg_holder on every console start)
  -> SimRunner.start(extra_env={"NERO_SCENE": ...}); start_nero_sim.sh validates and exports NERO_SCENE (unknown name
  -> fatal); the drop-down is locked while the simulation runs.
- Removed the V3.43 / V3.43.1 in-simulation switching: SimAgent scene bar, restart request file, backend exit 75 +
  os._exit, launcher wait loop / self-exec / pre-switch log copy, logs/v343_scene.txt selection.
- Kept: store_in_safe scene, door range unit fix, backend.log tail on fatal, RUN TASK gate in store_in_safe.
- Tests: test_v3_44.py (C1-C7) replaces test_v3_43.py.


## V3.43.1
- Scene switch did not restart: the launcher now restarts when the backend exits with 75 OR the GUI's
  logs/v343_restart.request exists (any exit code); a backend still alive 20 s after the request is stopped; the
  backend leaves with os._exit(75) after main(); logs of the run before the switch -> logs/v343_before_switch/;
  fatal() also prints the backend.log tail.
- store_in_safe door range follows <compiler angle> of nero_motor.xml (MJCF default degree).
- Tests: test_v3_43 C7.


## V3.43
- MuJoCo scene selector at the top of the 'NERO SimAgent' tab: fill_egg_holder (default) / store_in_safe. Choice saved
  in logs/v343_scene.txt (NERO_SCENE forces one); switching exits the backend with code 75 and start_nero_sim.sh
  re-executes the whole chain. main() exports NERO_V343_ACTIVE_SCENE so the GUI and the MuJoCo process agree.
- NERO_V343_Scenes.py: registry + store_in_safe MJCF (static safe body, hinged door 0..110 deg with damping /
  frictionloss, handle body obj_safe_handle, valuables box obj_valuables_box, hold equalities, door/world contact
  exclusion), MoveIt wall boxes + yawed door/handle boxes from the hinge angle, TF map, hold list, protected prefixes,
  task state (door open / object inside / door closed / success).
- Backend: 12 blocks marked '# V3.43 A2 BEGIN/END' (scene setup only: XML dispatch, egg labels skipped, TF map,
  MoveIt boxes, protected geoms, hold list, GUI selector + live safe state, RUN TASK gate, scene export, exit 75).
  Without them the backend is byte-identical to V3.42; the run_sim per-step loop is untouched.
- Tests: test_v3_43.py (C1-C6); test_v3_36/37 strip the A2 blocks before their V3.34 run_sim hash.


## V3.42
- No grasp / placement strategy change.
- End of every task: [V3.42 TASK STATS] (total time; LLM / skills / other with shares; per skill count/time/share;
  per object; tokens of this task incl. cached, requests). Failed tasks: one extra no-tool LLM call on a compact
  trace -> [V3.42 FAILURE ANALYSIS] (primary cause / evidence / fix). NERO_V342_FAILURE_ANALYSIS, _ANALYSIS_LANG.
- LLM providers: OpenAI (Responses, unchanged), DeepSeek and Kimi via an OpenAI-compatible Chat Completions adapter
  (local history incl. reasoning_content, tools converted, usage mapped). GUI provider selector; per-provider key
  memory; env DEEPSEEK_API_KEY / MOONSHOT_API_KEY (KIMI_API_KEY), NERO_DEEPSEEK_API_BASE / NERO_KIMI_API_BASE,
  NERO_DEEPSEEK_MODELS / NERO_KIMI_MODELS, NERO_*_EXTRA_BODY, NERO_V342_CHAT_MAX_TOKENS.
- Tool-step budget: GUI default 60 (auto per-egg raise only with NERO_V342_AUTO_BUDGET=1); when used up (or the model
  stops with eggs pending and < 10 steps left) the operator is asked how many steps to add (dialog).
- Tests: test_v3_42.py (A1-A4).

## V3.41
- From the V3.40 USB log (insertion 0 mm in 12 steps, J6 0.6 deg from its limit, reported as success; false
  HOLDER_POSSIBLY_MOVED from close views; stale release pose -> RETURN FAILED + LOST; re-found egg forgotten again):
- Joint-limit margin (all modes): IK screen (slot-above, whole-arm yaw family), every descent-corridor level (now to
  30 mm below the entry, NERO_V341_CORRIDOR_DEPTH_M) and the executed-branch post-check need every joint >= 8 deg
  from its limit (NERO_V341_LIMIT_MARGIN_DEG; fallback 4 deg NERO_V341_LIMIT_MARGIN_MIN_DEG); larger margin first;
  cuRobo endpoint near a limit -> MoveIt joint goal to the screened IK. NERO_V341_LIMIT_MARGIN=0 restores V3.40.
- Insertion (all modes): a blocked straight descent is a failure: BioIK 5 mm step + MoveIt joint goal (same branch,
  NERO_V341_INSERT_MAX_JOINT_STEP_DEG 12); still blocked / stalled -> retreat to the entry; INSERT_BLOCKED_REPLAN_YAW
  once per egg+slot (current yaw skipped, NERO_V341_INSERT_REPLANS), then INSERT_BLOCKED_JOINT_LIMIT with margins.
  NERO_V341_INSERT_FALLBACK=0 restores V3.40.
- Carton suspicion (wrist cameras): live / precheck views closer than 0.28 m, incomplete, border-cut or during a
  placement are ignored for the moved check (NERO_V341_HOLDER_VIEW_GATE, _MIN_CAM_DIST_M).
- Held pose (no-truth modes): refreshed after every executed trajectory while holding and before the
  return_to_pick release (NERO_V341_HELD_REFRESH).
- Wrist tracker: a lost name is forgotten once (tracks created after the first lost hint are kept,
  NERO_V341_LOST_ONCE); V3.39 predicted association only with the RGB-only camera (D435i back to V3.38).
- Agent rule V3.41 INSERT BLOCKED. Tests: test_v3_41.py (M1-M5); orientation harness scenario A2.

## V3.40
- From the V3.39 USB log (egg grasped, then false alarms + cuRobo refusing every move):
- Table plane: calibrated until >= 2 well-measured lying eggs agree within 5 mm (NERO_V340_TABLE_MAX_DEV_M), then
  locked for the episode; the same plane for egg height checks, carton detection and carton-height plausibility.
- Bystander check with perceived poses: horizontal motion of a re-seen neighbour minus 2 x its ray rms (max 6 mm);
  z ignored; neighbours not seen again are not judged. Truth-backed modes unchanged. NERO_V340_BYSTANDER_XY.
- Pre-place view (egg in hand) no longer replaces the fused carton: within 6 mm it goes through the V3.36 filter,
  otherwise it is ignored (NERO_V340_PREPLACE_AGREE_MM / _FILTER). A look trajectory the arm did not follow
  (PHYSICAL TRACKING MISMATCH) stops further pre-place looks.
- cuRobo failure: [V3.40 CUROBO DIAG] (joint-limit margins, MoveIt validity of the start state) + MoveIt joint goal
  to the IK-screened solution / BioIK pose fallback in the reorientation yaw family, the recenter move and the
  return_to_pick_location hover. NERO_V340_MOVEIT_FALLBACK.
- Tests: test_v3_40.py (V1-V4).

## V3.39
- USB (RGB-only) wrist camera, from the V3.38 log (slow, grasp 27 mm too high -> BYSTANDER_EGG_DISTURBED):
- Scan: a partly seen carton is refined at once (plus views nearer the base / along the carton when the usual ones
  are unreachable); two partial views with complementary far edges give the carton (fuse_partial_holders); carton
  height views (centre +- 0.10 m) in the same scan; a repeat scan skips grid views that were empty. NERO_V339_SCAN_FAST.
- Lying eggs always checked against the table plane: measured (well-conditioned eggs only) if it agrees with the
  calibrated plane, else the calibrated one (table_plane); class half width when the size is not measured.
- Consensus triangulation (largest consistent ray set) against pile mix-ups; plane fallback (two views from
  different camera positions agree on table + half width) -> measured + trusted.
- Close-view association: an unclaimed, not border-cut detection takes the name whose measured position is nearest
  (gate 45 mm, unambiguous) - no more 'seen []' -> recapture + q0 round trips.
- plan_grasp: up to 2 extra close views in place (target unseen / not measured / > 10 mm from this view / lying-egg z
  off) before the q0 return; GRASP_HEIGHT_IMPLAUSIBLE when the height is still > 10 mm off. Agent rule V3.39.
- Tests: test_v3_39.py (U1-U6); test_v3_33 markers adapted.

## V3.38
- Fix: HolderHeight.extent_ok was a numpy bool -> every USB capture failed after the 3rd carton view ("Object of type
  bool is not JSON serializable"); W.json_safe() on every wrist / fixed / monitor payload.
- Agent: _post_responses retries transient API errors (DNS, reset, timeout, HTTP 408/409/429/5xx) with back-off.
- orient_for_slot: 0 BioIK-feasible goals -> plan here first; recenter on 1-2 feasible / big swing or after a failed
  plan; IK-screened candidate centre points and heights.
- Slots: multi-view carton centre at the measured height (HolderHeight.fused_center); _v338_refine_slot from two
  agreeing complete views (4-15 mm); pre-place look per distance + carton-centre fallback.
- Tests: test_v3_38.py (S1-S4); test_v3_37 adapted.

## V3.37
- orient_for_slot: whole-arm yaw family flagged risky (< NERO_V337_MIN_FEASIBLE feasible goals or a joint swing
  > NERO_V337_MAX_JOINT_TRAVEL_DEG) -> the egg is carried to NERO_V337_RECENTER_XY first, then re-planned.
- run_sim hard_restore_arm(keep_hold=True) for the automatic RUNAWAY restore: held egg carried with link7, force mode
  + CONNECT hold kept (2nd restore within 3 s / HARD RESET 0 release as before); rest of run_sim = V3.34.
- return_to_pick_location clears a held safety latch first (STOP + RECOVER ACK, controller sync).
- Lost egg: marked lost_unknown, plan_grasp refused, observe_scene searches the drop area, trackers forget the stale
  track (hint lost=...), found again when re-perceived; Agent rule V3.37 SAFETY + LOST EGGS.
- GUI: tab "NERO SimAgent", foldable sections, monitor zoom 25-300 % + resizable pop-out.
- Tests: test_v3_37.py (R1-R4); test_v3_36 Q1 compares run_sim outside the restore.

## V3.36
- Grasp physics reverted to V3.34 (run_sim / add_official_gripper_to_xml identical to V3.34; auto CONNECT hold).
- RGB egg recognition: egg_mask_v336 (no brightness floor), hole filling, solidity-based split, ellipse centre;
  size only from clean blobs agreeing over >= 2 views (else not measured -> placeholder + margin); multi-view yaw;
  table + half-width height check.
- Freeze of the planned grasp (pose / geometry) and placement slot; live views only check for movement.
- Carton height: >= 3 views, baseline, view-angle spread, sharp minimum, plausibility, lock. Moved carton: graded
  views (live = suspicion only), confirmation from two camera positions incl. a skill view; HOLDER_POSSIBLY_MOVED.
- Tests: test_v3_36.py (Q1-Q4); test_v3_35 adapted.

## V3.35.1
- Fix: the simulator crashed at start with V3.35 (solver options changed at runtime after MjData existed).
  cone=elliptic / impratio / noslip_iterations and the pad / egg sliding friction are now written into the model file
  (_v335_physics_xml -> *_v335phys.xml) BEFORE MjModel / MjData are built; nothing is changed at runtime.
- Tests: test_v3_35 N10.

## V3.35
- Friction grasp (default): cone=elliptic, impratio 10, noslip_iterations 8, pad mu 0.9 (was 10), egg mu 0.5; the
  sim CONNECT hold is an opt-in assist (GUI / NERO_V335_ASSIST_HOLD=1); hold_active = gripper force sensing (both pads
  on one object); clamp waits for 70 % of the force; truth-based anti-slip boost off; GRASP_LOST while clamping.
- Live perception: background wrist captures (NERO_V335_LIVE / _LIVE_HZ, GUI), queued and folded in by the skill
  thread; fused carton pose with moved detection (2 consistent views); pre-motion checks (TARGET_MOVED_SINCE_PLAN,
  SLOT_MOVED_SINCE_PLAN, IN_HAND_SHIFT_DETECTED); insertion stops and lifts back when the carton moves.
- Camera calibration error (GUI slider 0-3, default 0.3): believed intrinsics / residual k1 / hand-eye vs the rendered
  camera, fixed per-session direction (NERO_V335_CALIB_SEED); live monitor with the perception labels + pop-out.
- Measurement instead of priors (USB RGB): triangulated egg centres (robust, moved-egg handling), measured
  width/length -> gripper aperture, measured table height, carton top height by multi-view parallax; egg class prior
  only a placeholder; plausibility left to the Agent (rule V3.35 MEASURED, LIVE SCENE).
- No truth fallback: wrist or fixed camera selected = camera; RUN TASK refused / tools stop with PERCEPTION_UNAVAILABLE
  when its process is not running; pose_source names the real camera.
- Tests: test_v3_35.py (N1-N9, incl. the wrist process loop with a stub MuJoCo renderer and the GUI snippets with a
  Tk stub); older tests adapted (version lists, test_v3_11 no-fallback, test_v3_28 capture helper).

## V3.34
- Launcher: automatic repair when the CUDA self-check fails (retry, stop leftover NERO GPU processes, list other
  /dev/nvidia-uvm holders without killing them, reload nvidia_uvm with one sudo/pkexec prompt, re-check) instead of
  requiring a reboot. NERO_V334_CUDA_AUTOFIX, NERO_V334_CUDA_PRIV.
- Tests: test_v3_34.py (runs the launcher block with stub sudo/modprobe/nvidia-smi).

## V3.33
- Scene: neutral grey floor (the oak floor was egg-coloured), plain white table (no veins), shadow map 8192 and a
  brighter headlight ambient; NERO_V333_FLOOR / NERO_V333_TABLE / NERO_V333_SHADOWSIZE / NERO_V333_HEADLIGHT_*.
- Supermarket moulded-pulp egg carton as a visual mesh generated from the holder CAD (W.carton_height /
  carton_mesh_obj); the physical carton (base, rims, posts) is unchanged but hidden (group 3). NERO_V333_PULP_CARTON=0.
- RGB carton detection (rgb_carton_far_edges): carton mask, projection on the CAD top plane, min-area rectangle,
  centre from the far edges; depth detection uses the carton footprint + plateau ring per cell.
- RGB-only detections named only after a second view at the same place (NERO_V333_CONFIRM_VIEWS / _DIST_M).
- D435i on the USB side face; no wrist camera drawn in non-wrist modes; scan R_MIN 0.24 m; unreachable views
  remembered; holder search + up to 3 refine looks; HOLDER_NOT_PERCEIVED stop; guard uses scene_snapshot and does not
  nudge without a carton; newest complete holder wins; USB task label.
- Tests: test_v3_33.py; v329_synth renders the lit carton scene; tests 26-32 adapted.

## V3.32
- Release zip keeps the executable bit of start_nero.sh / start_nero_sim.sh / tests/check_release.sh (lost in the
  V3.31 and V3.31.1 zips -> "Run as program" did not work). Test E1 checks it.
- No-prior modes: the episode's eggs = eggs perceived in this episode (+ held / released) (_v332_known_eggs); snapshot,
  verify next-step state, pending lists and the Agent completion guard use them (objects_policy=perceived_only);
  never-perceived names are pruned from results and plan_grasp of such a name is refused without motion.
- Tests: test_v3_32.py.

## V3.31.1
- Wrist cameras mounted ON the gripper base side faces (D435i +y, USB -y, perpendicular to the finger travel along
  link7 z, at the gripper-axis height, front flush with the base, bracket), faces from gripper_base.stl (nominal
  fallback); optical axis crosses the gripper axis 0.40 m ahead. [V3.31.1 WRIST MOUNT] log line.
- Tests: test_v3_31_1.py; test_v3_27 / test_v3_30 adapted to the new mount.

## V3.31
- Wrist no-prior mode: layout unknown at episode start (V329_LAYOUT_STATE holder_ok / pile_ok): no slot names, no
  near-zone posture checks, no holder/slot/pile coordinates in the snapshot until perceived.
- Scan = uniform grid over the table workspace within the reach annulus (NERO_V331_SCAN_*), ring opt-in.
- Egg-in-slot height / placed position from the holder model + egg class (egg_center_in_slot_z), adaptive RGB
  well threshold, scan stop on stability only, [V3.31 UNNAMED EGGS] when the name pool is too small.
- Fixed camera without priors (NERO_V331_FIXED_NO_PRIOR=1, default): NERO_V331_Fixed_Camera.py (generic naming,
  holder from depth, hints), backend estimator / perceived holder+pile / MoveIt holder from perception, pre-place
  capture. V3.08 module unchanged.
- Tests: test_v3_31.py; test_v3_29 adapted (ring opt-in, model-based placed height, fixed mode).

## V3.30
- Anti-cheat pass 2 (wrist modes): no simulator identity seeding (ViewTracker names new eggs from the name pool in
  scan order), one egg class geometry for all eggs (no per-egg MJCF sizes), scan stop without the egg count, truth
  logs against the nearest simulator egg.
- Second wrist camera profile "usb130" (InnoMaker U20CAM-1080P, 130 deg, RGB only): rectified pinhole 1280x720,
  HFOV 100 deg, own mount; RGB-only perception: eggs = colour blobs (k-means split) cast onto model planes, holder =
  dark slot wells cast onto the well plane + 4x2 grid fit, model-based robot self mask. One wrist process renders the
  camera named in the trigger tag (cam=usb130).
- Pose source drop-down (camera / wrist / wrist_rgb / truth); NERO_V311_POSE_SOURCE=wrist_rgb.
- Visual wrist camera models in link7 (massless, non-colliding, behind the optics); simulator shows the selected one
  (logs/v330_wrist_camera_model.json).
- Tests: test_v3_30.py (profiles + models, hints, RGB holder/eggs on synthetic colour renders, wrist process with the
  usb130 camera, generic naming, wiring); older tests adapted to the drop-down / generic naming.

## V3.29
- Wrist camera mode without a preset layout (NERO_V329_NO_PRIOR=1): per task the layout is unknown; blind ring
  scan (directions only) until the holder and all eggs are seen; holder pose from the depth image (plate + rims,
  min-area rectangle), slot centres refined from the six posts; source region = perceived egg pile (the scene has
  no physical basket). Slot dicts, holder centre, basket centre, safe orientation XY, near-zone posture checks and
  the MoveIt holder boxes (simulator bridge reads logs/v329_moveit_holder.json) use the perceived scene;
  _v271_slot refuses until the holder is perceived. The wrist process drops the registered SLOTS/BASKET.
- No simulator truth in the skill checks (wrist mode): skill_core.object_pose -> perception / TCP x grasp-time
  transform while held (hold/force decides) / slot where released; truth only in [V3.29 TRUTH LOG].
- Pre-place holder look in plan_placement, then back to the previous joint state.
- Tests: test_v3_29.py (synthetic RGB-D holder perception incl. moved/rotated holder, blind scan flow, no-truth
  estimator, wiring) + tests/v329_synth.py.

## V3.28
- Wrist camera view-aware tracking (NERO_V326_Wrist_Camera.ViewTracker around the unchanged V3.11 PerceptionTracker):
  association only for tracks whose last position projects into the current image; out-of-view tracks keep
  name + pose. Trigger-tag hints from the backend: held=<egg>, placed=<egg>@slot xyz (episode releases).
  Fixes the V3.27 egg_1 <-> egg_4 identity swap after the first placement.
- Low-pose escape: failed q0 detour before a look move -> direct MoveIt look plan; wrist recapture in plan_grasp
  without q0; V2-39 q0 path floor = min(180 mm, start z - 20 mm) (was 180 mm -> deadlock at z = 143 mm).
- Tests: test_v3_28.py (hint protocol, in_view, replay of the V3.27 swap, backend markers).

## V3.27
- Fix (sim, wrist camera): V3.26 planned ONE look pose per view with cuRobo (TCP 0.29 m above the point, yaw 0);
  cuRobo returned None 3x for basket and holder, the arm stayed at q0 and every capture looked at the ceiling.
- Look-pose family (NERO_V326_Wrist_Camera.look_pose_candidates): camera distance x tilt x roll, each camera aimed
  exactly at the point; BioIK screen (collision-aware, joint margin >= 3 deg), ranking by joint travel, MoveIt
  joint-goal plan, cuRobo pose plan as fallback, TCP check after execution. Look -> look without the q0 detour.
- WRIST_LOOK_UNREACHABLE + payload look_failed; log markers [V3.27 WRIST LOOK SCREEN] / [V3.27 WRIST LOOK OK].
- Tests: test_v3_27.py (geometry, URDF IK reachability, backend look logic with a fake skill core).

## V3.26
- Sim: wrist camera pose source (NERO_V326_Wrist_Camera.py), parallel to the fixed camera (NERO_V308 unchanged).
  MJCF camera inside link7, separate render/perception process started on demand, per-capture extrinsics,
  look-pose survey (basket / holder / above target), view merging, return to q0 before grasp planning.
- Sim GUI: pose source Camera (default) / Wrist camera / JSON ground truth; NERO_V311_POSE_SOURCE=wrist supported.
- start_nero_sim.sh checks + compiles the new module.
- Tests: test_v3_26.py (MJCF, mount geometry, survey/merge, capture protocol, synthetic wrist RGB-D perception,
  camera process loop with a fake MuJoCo, backend wiring).

## V3.25
- New top-level tab 「Camera调试」 (NERO_V325_Camera.py); 「实机 NERO」 renamed 「实机NERO调试」.
- RealSenseSource (in-process pyrealsense2 thread, aligned RGB-D, filters), live view with overlays, click measure,
  ROI plane fit, camera_info.json.
- Eye-to-hand calibration: ArUco on the flange (PnP + depth-refined centre), manual / automatic sampling (MoveIt,
  fence, ≤ 10 %), alternating least squares (Horn quaternion + closed-form marker offset), outlier rejection,
  leave-one-out errors, camera_calib.json, live error, quick check.
- Experimental egg detection on real RGB-D (V3.10/3.11 pipeline, display only).
- Tests: test_v3_25.py (maths; image part with numpy + OpenCV), test_v3_25_gui.py (fake D435i).

## V3.24
- Editor: top view x −0.95 .. +0.70 m (was +0.25), y ±0.85 m; side XZ −0.45 .. 1.00 m; +x warning only for zones that
  lie completely behind the base.
- Park pose D (「待断电位置⚠」 next to 「回 Home⚠」): JC.load_park / save_park / park_target (soft-limit clamp),
  JointPanelUI.go_park (preview, confirm, MoveIt + fence), 「当前设为待断电位置」, zone hint when refused.
- Tests: test_v3_24.py, test_v3_24_gui.py.

## V3.23
- Safety rule A (default): vertical walls on the zone polygon for the whole arm + gripper at any height; sampled
  chain spheres (link_radius / gripper_radius, sample_step) + margin (default 20 mm); base must be inside the zone.
- `merge_base` / 「并入底座」 (convex hull); auto-merge after teaching corners in rule A; red header when the base is outside.
- MoveIt walls: `wall_boxes` (one thin rotated box per edge), sent with the collision boxes on save / open / startup;
  monitor `do_scene` supports yaw; inspect retracts the walls and restores them.
- Teach / inspect relax: wrist + gripper + tool tip may touch the wall line; hand-guiding over a wall -> alarm.
- Rule B = the old layered rule (polygon ∪ base disc below xy_free_height), selectable in the editor.
- Tests: test_v3_23.py, test_v3_23_gui.py; older tests pin rule B via a legacy default shim.

## V3.22
- Fix: teach entry judged "failed" because ctrl_mode stays 0x01 on V111 -> gate reopened, no teach flag, corners not
  recorded. Entry evidence now also = leader angles streaming (pyAgxArm) or leader topic streaming + joint_states silent.
- `leader_active()`: hand guiding detected without the teach flag -> leader topic used for angles; UI label shows it.
- Slow exit: gate closed, ROS arm_controller synced to the pose (monitor `hold`), pyAgxArm speed 10 % + move_j(pose)
  before enable, gate reopened, tool lifted vertically, MoveIt home at 10 % (`exit_home`); out-of-limit joints clamped.
- Tests: test_v3_22.py, test_v3_22_gui.py; fake monitor FAKE_LJS_FILE / FAKE_HOLD_LOG / hold command.

## V3.21
- Fix: 「记录角点」在手动拖动时记录不到角点（joint_states 在 leader 模式停发）。
- New `leader` helper (pyAgxArm get_leader_joint_angles, 20 Hz), started on teach entry, stopped on exit.
- `feedback_source()` / `feedback_js()` / `feedback_diag()`; teach panel shows the angle source; failure log lists feed ages.
- Tests: test_v3_21.py, test_v3_21_gui.py; fake monitor FAKE_JS_PAUSE_FILE.

## V3.20
User: after 「进入示教模式」 the motors still held position. Causes: (1) V3.19 called `set_master_mode()`; the pyAgxArm
NERO API documents `set_leader_mode()` (leader = linkage teaching input, ctrl_mode 0x06) — master mode is not in
the NERO docs; (2) the ROS driver kept its control gate open and kept commanding the arm; (3) V3.19 never checked
the result. Now:
- helper: prints the available mode APIs + ctrl_mode before; `set_leader_mode()` (fallback `set_master_mode()`);
  success only if ctrl_mode becomes 0x06 / 0x02 within 3 s; exit = `set_normal_mode()` (fw ≤ 1.11) /
  `set_follower_mode()` (≥ 1.12) + enable, ctrl_mode must leave teach.
- pipeline: driver control gate (`control_enable`) closed before entering, reopened after leaving or a failed entry.
- teaching: joint angles from `/feedback/leader_joint_states` when fresh (panel, teach points, editor).
- editor: the mode line shows the arm's reported ctrl_mode (also when teach mode was entered with a button on the arm).
- Tests: `tests/test_v3_20.py`; V3.19 tests updated.

## V3.19
1. Teach mode buttons in 示教角点: 「进入示教模式（可用手拖动）」 (pyAgxArm `set_master_mode`, AgileX's zero-force
   teaching; helper `NERO_V313_Real_Robot.py mode --mode teach`) and 「退出示教模式」 (driver `exit_teach_mode` +
   `set_normal_mode` + `enable`). Mode shown next to the buttons; motions refused while teaching.
2. Safety-zone files: 「打开…」 / 「另存为…」 (any name / folder, default ~/nero_zones), 「保存」 = current file; last
   file remembered (~/.config/nero_console/state.json) and opened at the next start.
3. 「回 Home」 sometimes did nothing: it called the driver's move_home service and the console waited only 5 s,
   blocking the ROS monitor (homing takes longer -> "timeout", joint feedback paused, the panel flickered to
   disabled); the driver also ignores it in teach mode / while executing. Now Home = MoveIt joint goal to all zeros
   (preview, confirm, fence + path check, result logged); the service is a fallback with a 60 s timeout.
   Service calls in the monitor are asynchronous (polled in the spin loop).
- Tests: `tests/test_v3_19.py` (mode helper, set_teach_mode, async services), `tests/test_v3_19_gui.py`.

## V3.18
1. Joint panel — target follow (user: after 一键启动 the joints went to zero but the sliders stayed where they were
   when the panel became enabled). Targets now follow the actual angles continuously until the user edits a target
   (slider drag, entry, ± buttons, pose library); external motion (homing, RViz) no longer leaves stale targets.
   Editing stops following; a finished motion or 「同步为当前」 resumes it. Not while an entry box has the focus.
   Status line: 「目标：跟随当前角度」 / 「目标：已修改，未执行」.
2. Safety editor side view (user screenshot: texts overlapping). No text inside the plot any more; the four height
   lines are named with their values in a right-hand column, pushed apart with leader lines (spacing from the real
   font metrics); colour meanings in a static legend under the canvas; vertical range fitted to z 上限.
- Test `tests/test_v3_18_gui.py`: follow / edit / resume; side view label overlap with XZ, YZ and a 14 pt font.

## V3.17
User feedback on V3.16.2 (it runs now):
1. Work side: NERO picks on the −x side of base_link (sim: egg basket (−0.425, −0.155), holder (−0.420, +0.155);
   positive J2 leans forward). The V3.16 default zone was behind the robot (+x). Default zone is now
   x −0.15 … −0.70, y ±0.45 (contains basket and holder), table box at x = −0.30. A saved zone whose centre is at
   +x shows a red warning in the editor.
   Top view drawn from behind the robot: up = forward (−x), right = +y (drag / resize / IK unchanged in world terms;
   box resize handle = bottom-right on screen). Side view XZ: right = forward.
2. Tick labels moved into white margins (left: x cm, bottom: y cm) — no longer covered by the base disc / zone.
3. Side view explains itself: red band "防撞桌" between the table and z_min, green zone (low points must be inside),
   pale band above the XY limit height ("XY 不限"), every line labelled with its meaning; taller canvas.
- Tests: `tests/test_v3_17.py`; V3.15 / V3.16 tests mirrored to the −x work side.

## V3.16.2
- Fix: V3.16.1 segfaulted at startup on Ubuntu 22.04 (system python3.10 + Tk 8.6; faulthandler: segfault while
  creating a ttk widget in run_console). Cause: `fixed_label()` created a `tkinter.font.Font` object per label to
  measure the line height; each was garbage-collected (`font delete`), which on Tk 8.6 / Xft leaves a dangling font
  cache entry. Now the line height comes from `font metrics … -linespace` (no font object) and is cached.
- Colour-emoji-capable characters removed from the GUI (`🔒` → 「（固定）」, `✖` → `×`).
- Test `tests/test_v3_16_2.py`: no Font objects and no emoji characters in the GUI sources (part of check_release).

## V3.16.1
- Layout stability (user report: the live status area kept changing size, so the controls below jumped and were hard
  to click). New `fixed_label()` (NERO_V314_Joint_Control.py): live labels sit in holders of fixed pixel size; long
  text is wrapped and clipped instead of growing. Applied to: header mode, step messages, 实机 status line, launch
  preview, live status rows (CAN / ROS / joints / arm_status / gripper), joint panel state + per-column actual /
  deviation / warning + preview + TCP + speed, consistency banners + repeat / stop results, safety editor header /
  banner / test result / selection / teach info.
- The parameter box no longer collapses automatically at step 6 green (one click on 「参数 ▴」).
- Test: `tests/test_v3_16_1_gui.py` — 16 clickable widgets keep position and size over 25 rounds of empty / long /
  multi-line live texts while the arm moves (fails on V3.16).

## V3.16
Based on V3.15 (simulation unchanged).
- New `NERO_V316_Safety_Editor.py`: graphical safety-zone editor (top + side view, drag / add / delete corners,
  boxes, z lines, numeric fields, undo / redo, live arm + target ghost, 示教角点, 巡检, auto test) replacing the
  V3.15 number table.
- Fence model (in `NERO_V315_Consistency_Safety.py`): XY polygon ∪ base disc, XY free height above the table, z_min /
  z_max, margin warnings, 2 mm boundary tolerance, tool = sim TCP / pointer / none; teach (tool 5 mm below table) and
  inspect (tool 10 mm above table) relaxations, tool-only XY in those modes; speed cap 5 % in those modes.
- Path rule: a motion that starts outside may leave the violation (not getting further out) but must end inside.
- IK (Levenberg–Marquardt, active set at joint limits) for hover poses; V3.15 workspace files migrated.
- Joint panel: 执行 locked while the target is outside the zone. Default z 上限 1.0 m (the zero pose reaches 0.87 m).
- Tests: `tests/test_v3_16.py`, `tests/test_v3_16_gui.py`; V3.15 tests updated for the polygon model.

## V3.15
Based on V3.14 (simulation unchanged).
- New `NERO_V315_Consistency_Safety.py`: URDF forward kinematics, sim-vs-driver URDF comparison, live FK/TF
  consistency, direction wizard, repeatability, stop test, workspace config (collision boxes + software fence),
  goal and path fence checks, motion CSV log, result files; Tk tabs 一致性测试 / 工作区安全.
- ROS monitor helper: plan → path fence check → `/execute_trajectory` (two-phase) when the fence path check is on;
  plan-only goals never execute; ApplyPlanningScene / GetPlanningScene for `v315_*` boxes; TF of watched links;
  the driver URDF is passed to the GUI.
- RealPipeline: fence goal check before sending, `move_and_settle` (final error + CSV row), `apply_scene`,
  `set_tf_watch`, joint sample history for the stop test.
- Joint panel: 「只规划（不执行）」; all panel motions go through `move_and_settle`.
- Console: 实机 sub-tabs; the collapsed parameter box no longer leaves an empty band.
- URDF parser keeps joint origin / axis (kinematics).
- Tests: `tests/test_v3_15.py` (K1–K10), `tests/test_v3_15_gui.py`, fake monitor `tests/v315_fake_monitor.py`.

## V3.14
Based on V3.13.1 (simulation unchanged).
- New `NERO_V314_Joint_Control.py`: URDF/SRDF model (limits, 7-joint arm group, gripper group, named states, TCP
  frames), soft limits, entry parsing, preview / confirmation rule, step-test targets, enable rules, pose library,
  and the Tk joint panel (`JointPanelUI`).
- ROS monitor helper: 10 Hz joint events, robot model event (topics, then /move_group parameters), MoveGroup action
  joint goals with feedback / result / cancel and MoveIt error-code hints, TCP from TF.
- RealPipeline: `move_joints`, `move_gripper`, `cancel_move`, `actual_arm_deg`; stop_all releases a waiting motion.
- Console: scrollable 实机 tab, collapsible parameter box (auto-collapses at step 6 green), joint panel.
- Fix: worker-thread confirmations (step 3 enable, motion confirms) go through a queue served by the Tk main loop
  instead of calling Tk from the worker thread.
- Tests: `tests/test_v3_14.py` (J1–J7), `tests/test_v3_14_gui.py` (panel end-to-end with a fake ROS monitor).

## V3.13.1
- Step 2 (CAN 口): gs_usb rejected `restart-ms 100` ("Device doesn't support restart from Bus Off", rc=2), so can0
  was never configured. Now tries with restart-ms, falls back to `bitrate 1000000` only (still one pkexec prompt).
- Step 5: a driver that dies immediately now reports its real failure line (e.g. firmware version) instead of
  "launch process not running" (race between process exit and log pump).
- Test R1b: runs the generated shell script with a fake `ip` that rejects restart-ms -> can0 configured and UP.

## V3.13
Based on V3.12.1 (simulation unchanged: backend / agent / GPU server / camera byte-identical).
- New `NERO_V313_Real_Robot.py` (logic, no Tk) + 实机 tab in the console: six-step pipeline
  环境 / CAN 口 / CAN push·使能 / CAN 流量 / ROS2 驱动 / MoveIt·RViz with lamps, per-step run, 一键启动, 全部停止,
  清理残留进程, ■ 急停, parameter panel (arm_type nero locked, effector, fw_version, ROS_DOMAIN_ID, speed 1-30 % with
  unlock, auto_control_gate / follow / RViz), command preview, copy command / terminal environment, live status
  (CAN fps, joint_states Hz, joints, arm_status, gripper), service buttons.
- CAN setup via `pkexec` (password dialog every time). CAN push via pyAgxArm helper (`push`), ROS state via a
  persistent rclpy helper (`monitor`, JSON over stdin/stdout). Push-release test before the driver.
- Driver failure lines recognised (firmware version, CAN, missing package); first failure line is reported.
- Sim/real interlock; dangerous operations confirmed; window close stops everything (SIGINT -> SIGKILL per group).
- Tests: `tests/test_v3_13.py` (R1-R9, fake pyAgxArm / fake rclpy / fake CAN), `tests/test_v3_13_gui.py`.

## V3.12.1
Based on V3.12. The V3.12 camera run placed egg_2, then stopped holding egg_1: GPT chose lift(0.13) whose corridor was
97.9 % free (raised as BACKEND_EXCEPTION), then lift(0.04); orient_for_slot failed every strategy with the egg 62 mm
high; no tool could put the egg back, so egg_3/egg_4 were never attempted. Not caused by the V3.12 console (backend was
byte-identical); these are robustness gaps that run happened to hit.

- **Lift fallback**: blocked corridor (planning only, nothing moved) -> next lower height 108/90/70/50 mm; the result
  reports the height actually lifted (`v3121_lift_heights_tried_mm`). All blocked -> `V3121_LIFT_CORRIDOR_BLOCKED`,
  next action `return_to_pick_location`. Execution faults are never retried.
- **Orient pre-raise**: `orient_for_slot` raises a held egg whose TCP is below 160 mm before the strategies run;
  total orientation failure now recommends `return_to_pick_location` (no more slot hopping).
- **`return_to_pick_location`** (new tool): pick pose + aperture + in-hand transform are recorded at clamp. Lifted egg
  above its pick location -> straight down to 3 mm above the pick pose; moved/reoriented egg -> transit to the hover
  first; still at the pick pose -> release in place. Opens to the pre-grasp aperture, retreats, stage OBSERVED, returns
  `v3121_next` (pending eggs). Max 2 per egg per task. Agent rule `V3.12.1 HELD-EGG RECOVERY`.
- **Descent precheck** (the long-deferred egg_1 `major_aligned__inward6` gap): for every fully planned pregrasp, the
  straight descent (to 20 mm above, then grasp) is planned with MoveIt from that trajectory's END joint state; < 98 %
  -> candidate rejected (`V3121_DESCENT_CARTESIAN_REJECT`) and the next one is planned. Also in the screen fallback.
- **Blocked descent retreat**: approach_planned_grasp returns to the pregrasp height right after a blocked descent.
- **Retreat ladder** in the V3.07 q0 move: full straight retreat blocked -> +120/80/50 mm, else joint-space q0 directly.
  plan_grasp's camera re-look reports `RECAPTURE_CLEAR_VIEW_FAILED` when that move fails.
- **Held egg in camera mode**: scene location `in_gripper`, pose from TCP x grasp transform
  (`perception.source = tcp_x_grasp_transform`) instead of its stale basket pose.
- Agent: lift guidance 0.10-0.11 m; return tool allowed in the V3.06 finish grace.
- Tests: `tests/test_v3_12_1.py` (H1-H8); `test_v3_12.py` keeps the hash check for planner/GPU/camera only.

## V3.12
Based on V3.11 (camera-mode run: 4/4 eggs on the first try, perception err 0.4-2.0 mm, camera = truth on every verify).

### Console first
- `./start_nero.sh` no longer starts anything by itself: it checks SHA256SUMS + compiles the console and opens
  `NERO_V312_Console.py` (tkinter, standard library only).
- 仿真 (MuJoCo) tab: "▶ 启动仿真" runs `start_nero_sim.sh` in its own session (`NERO_NO_FAIL_PAUSE=1`, so a failed start
  never waits for Enter), streams its output, and drives 8 step lamps from its status lines (grey / yellow running /
  green / red failed / amber = optional TRAC-IK skipped). "■ 停止仿真" sends SIGINT to the launcher only; after 25 s
  (`NERO_V312_STOP_GRACE_S`) the whole session is SIGKILLed. Closing the console while the sim runs asks, then stops it.
- 实机 NERO tab: placeholder with the planned six steps (V3.13); shows the sim/real interlock state.
- `./start_nero.sh --sim`: old behaviour (sim chain directly). `--real`: console opened on the 实机 tab.

### Simulation chain
- `start_nero_sim.sh` = V3.11 `start_nero.sh` with one fix: `trap cleanup EXIT INT TERM` -> `trap cleanup EXIT` +
  `trap '...; exit 130' INT TERM`. With the old trap, a SIGINT delivered to the script alone (not the whole terminal
  group) ran cleanup and then **continued with the next startup step**; now it exits and cleanup runs exactly once.
- Backend, agent, planner core, GPU client/server and camera module are byte-identical to V3.11.

### Tests
- `tests/test_v3_12.py`: launcher split; V3.11 hashes of the unchanged files; step tracker on a recorded V3.11 startup,
  a failed start, TRAC-less start, silent exit; stop semantics on a fake launcher; the real `start_nero_sim.sh` fails
  fast without an Enter prompt and, stopped while waiting for the GPU server, exits 130 with one cleanup and no
  leftover GPU server.
- `tests/test_v3_12_gui.py`: drives the real console window (start -> 8 green lamps + SYSTEM READY, stop -> 已停止,
  start re-enabled). Skips without tkinter / DISPLAY.

## V3.11
Based on V3.10 (both V3.10 test rounds passed: noisy depth 48/48 detected, 0 FP, xy max 1.44 mm, axis max 4.43 deg).

### Skills use the camera
- Agent tab: **Object pose source for skills** switch - *Camera (D435i RGB-D perception)* (default,
  `NERO_V311_POSE_SOURCE`) or *JSON ground truth (MuJoCo)*. Takes effect at the next tool call; logged at task start.
- Camera mode, synchronous capture (skill waits, `NERO_V311_CAPTURE_TIMEOUT_S` = 20 s):
  - `observe_scene`: egg poses/locations/slot occupancy from the capture; each egg carries `perception {seen, trusted,
    confidence}`; `scene.pose_source = camera_d435i_perception`.
  - `plan_grasp`: capture taken in the inner call (after the V3.07 q0 move); target pose must be seen + trusted;
    otherwise q0 (clear view) + one recapture, then `PERCEPTION_TARGET_UNTRUSTED`. Neighbour eggs (finger-pad
    ranking) from their last perceived pose.
  - `verify_object_in_slot`: location from the capture (one recapture if untrusted), else `PERCEPTION_COULD_NOT_CONFIRM`.
    v306_next (remaining eggs / next slot) from perception.
  - No silent fallback: camera not running -> `PERCEPTION_UNAVAILABLE`.
- Still simulator-based (documented): in-hand / contact / motion monitors (lift follow, bystander nudge, held-egg axis
  while carrying, "egg moved since planning"). The last one now compares sim with sim (perceived quaternions have an
  arbitrary roll about the long axis), stored outside the plan so it never reaches the model.
- Truth is logged only: `[V3.11 PERCEPTION] <tag> | egg_k T err=..mm`, `[V3.11 VERIFY] ... truth(log only)=...`.
- Agent rule `V3.11 CAMERA PERCEPTION` (what to do on untrusted / unconfirmed / unavailable).

### Camera module
- `PerceptionTracker`: identities kept across captures from perception (static match <= 30 mm, 15 mm if untrusted;
  moved egg <- remaining trusted detection). Names are seeded once from the evaluation matcher (sim only, so they match
  the simulator bodies the in-hand checks read); per-capture identity check vs truth logged (`[V3.11 TRACKING WARN]`).
- Result channel (shared JSON buffer + sequence) for synchronous captures; `tracked_scene.json` per capture;
  errors.json `diagnostics.tracking`.
- D435i body (90 x 25 x 25 mm, lenses) + floor tripod drawn at the configured pose (visual only, behind the optical
  centre: no rig pixel can enter the image; `NERO_V311_DRAW_CAMERA=0` to hide).

### GUI
- Task-status lamp moved into the log toolbar right of "Copy Full Log" (no longer covers log text); the caption
  "Copies the entire Unified Runtime Log..." removed.

### Not in this release
- Descent pre-check gap (egg_1 `major_aligned__inward6` blocked at 25 % every run): needs a MoveIt Cartesian check
  from a custom start state inside the preflight; deferred so this release only changes the pose source.

Tests: new `tests/test_v3_11.py` (K1-K4); check_release runs it.

## V3.10
Based on V3.09. Perception robustness for noisy depth. **Still record only: skills use ground truth.**

V3.09 round 2 (2026-10-22, `NERO_V308_DEPTH_NOISE_K=0.0025`): robot 4/4, perception 46/48 detected, 4 FP,
outliers of 10-13 mm / 41-83 deg (egg_2 captures 2-3, egg_4 capture 5), egg_2 in slot_2 missed at captures 8-9.

Root cause (reproduced offline, 60 synthetic scenes at 1280x720 with touching eggs, holder eggs and a gripper occluder):
depth noise adds a second height-map seed on one END of a lying egg -> two overlapping fits -> V3.09 kept the one with
the LOWER RMS, which is the half-egg fragment (a fragment always fits better) -> 750-1200 points, centre 10-13 mm off,
axis 40-80 deg off; the leftover piece became a false positive. Clean depth never produced the extra seed.

Changes (camera module):
1. Temporal depth fusion: 5 frames per capture, per-pixel mean of valid samples (`NERO_V310_DEPTH_FRAMES`, RealSense
   temporal-filter analogue; noise x0.45). In sim the frames are 5 noise draws of one static snapshot.
2. Model selection by explained points: every candidate is refined on surface inliers of a 48 mm neighbourhood; the
   candidate explaining the most unclaimed points (penalised by see-through) wins, claims its points, the rest are
   re-scored. Extra resting-egg starts shifted half an egg along the PCA axis.
3. Free-space view check: each fit is ray-cast into the depth image; pixels where the camera sees BEHIND the predicted
   surface (away from the rim) prove the egg is not there; pixels with something in front count as occluded.
4. Resting-egg prior: eggs outside the holder with centre < short-semi-axis + 8 mm are fitted near-horizontal
   (soft, 6 deg). Holder eggs are not constrained.
5. Warm start from the previous capture's trusted eggs; consistency with the previous capture is recorded.
6. Per-detection `confidence` (coverage, support, see-through, residual vs. the frame's median residual) and
   `trusted` (conf >= 0.70, coverage >= 0.55, see-through <= 0.06). Outputs: errors.csv columns
   confidence/trusted/coverage, summary.json `overall_trusted` (+ max errors), perceived_scene.json `quality`,
   overlay yellow = untrusted. Log line shows trusted / untrusted / trusted-FP.
7. Camera loop takes its trigger baseline before announcing READY (a trigger right after READY was lost).

Offline Monte-Carlo (same 60 scenes, 1280x720, k = 0.0025):
| | xy max | axis max | axis p95 | FP |
|---|---|---|---|---|
| V3.09 | 13.9 mm | 87 deg | 66 deg | 3 |
| V3.10, 1 frame, trusted | 1.9 mm | 8.7 deg | - | 0 |
| V3.10, 5 frames, trusted | 1.7 mm | 6.4 deg | - | 0 |
13/240 detections untrusted (all heavily occluded by the gripper box, coverage < 0.5); 3 eggs fully hidden.

Backend:
- `set_gripper_opening` waits until the TCP is still (< 0.5 mm per 0.25 s, max 1.5 s) before sampling its baseline;
  `_v307_go_q0` waits the same way after q0. Fixes the two GRIPPER_APERTURE_TARGET_NOT_VERIFIED
  (tcp_motion 3.2 / 4.5 mm) right after the V3.07 q0 move. Report in `verification.v310_pre_settle`.

Tests: new `tests/test_v3_10.py` (T1-T9); `check_release.sh` runs it.

## V3.09
Based on V3.08. Fix for the first V3.08 run (2026-10-21 22:14): captures 1-2 found 0/4 eggs, captures 3-5 reported
"detected" eggs with 20-40 mm / 60-90 deg errors, capture 6 found only the egg in slot_1 (1.1 mm, 2.3 deg).
Camera, depth and projection were correct (SELF-CHECK PASS, slot egg 1.1 mm).

Root causes (reproduced offline: V3.08 code -> 0 detections, 12 seeds):
1. Seed starvation: specular highlights on the holder posts pass the egg-colour mask and sit at ~48 mm,
   higher than lying egg tops (~44 mm). V3.08 kept only the 12 highest seeds, all on posts; every post fit was
   rejected, eggs never got a seed. The vertical egg in slot_1 (top 86 mm) was always seeded first.
2. Orange tactile-sensor pads on the fingers pass the colour mask; near the basket they were fitted as eggs.
3. The 50 mm evaluation gate let such blobs be "matched" to nearby eggs, so captures 3-5 looked like
   detections with huge errors instead of misses + false positives.

Changes (perception only; robot behaviour unchanged):
- Mask cleanup: connected components < 150 px (at 1280x720) removed (`NERO_V308_MIN_BLOB_PX`).
- Seeds need egg-sized local support (>= 120 px in a 44 mm square, `NERO_V308_SEED_MIN_SUPPORT`); cap 12 -> 40.
- Robot self-filter: points inside the arm/gripper geoms (oriented boxes from the current JOINT STATE, +8 mm,
  `NERO_V308_ROBOT_FILTER_MARGIN_M`) are removed. Uses proprioception only, as the real robot will (URDF + encoders).
- Match gate 50 -> 25 mm (`NERO_V308_MATCH_GATE_M`).
- errors.json diagnostics: raw/filtered mask pixels, robot-filtered points, seed positions, low-support rejects.
- Renderer closed explicitly on exit (no "EGLError ... Exception ignored" at shutdown).
- Tests R1-R3 added to `tests/test_v3_08.py`.

## V3.08
Based on V3.07 (2026-10-21 run: all four eggs placed). First step of sim-to-real with an Intel RealSense D435i.
**Robot behaviour is unchanged: skills still read MuJoCo ground truth.** V3.08 only renders, perceives and records.

### 1. Simulated D435i camera
- New fixed camera `v308_d435i` injected into the scene MJCF (eye-to-hand, default pos (-0.95, 0, 0.72) m looking at
  (-0.42, 0, 0.03), i.e. from the far side of basket + holder so the arm rarely occludes the eggs).
- D435i color-stream pinhole model, depth aligned to color: 1280x720, vertical FOV 42 deg (f = 937.8 px, HFOV 68.6 deg).
  Depth: z-depth, valid 0.28-3.0 m, quantised to 1 mm (z16), optional noise sigma = k·z².
- Rendered in a separate process `nero_v308_camera` (offscreen OpenGL EGL -> OSMesa -> GLFW), fed by a 10 Hz qpos
  copy from the simulator; the physics/viewer loop only copies an array.
- Runtime projection SELF-CHECK on the first capture (`[V3.08 CAMERA] SELF-CHECK PASS/FAIL`).

### 2. RGB-D egg perception (images + calibration only)
- Back-project aligned depth to world points; egg-shell colour mask + height above the table.
- Top-down height map -> one seed per egg dome -> point clusters -> robust spheroid fit (egg size prior
  31.5 / 21.9 mm semi-axes) from several starts -> points re-assigned to the best-fitting egg (separates touching eggs) -> refit.
- Output per egg: centre, long axis, quaternion (local +Z = long axis, same convention as the skills), fit RMS,
  location (basket / holder slot) in the same format as the scene JSON. Receptacle geometry is registered, not perceived
  (real bench: fiducial tags + CAD).
- Offline (synthetic RGB-D, 19:26 egg poses): clean depth xy error mean 0.8 / max 1.1 mm, axis mean 1.9 deg;
  with sigma = 0.0025·z² at 848x480: xy mean 1.6 / max 3.4 mm, axis mean 6.4 / max 13.7 deg.

### 3. Error logging to files
- Captures at the start of observe_scene, plan_grasp and verify_object_in_slot.
- `logs/perception/<run>/`: camera_info.json, errors.csv, summary.json, and per capture rgb.png, depth_mm.png,
  overlay.png, perceived_scene.json, truth_scene.json, errors.json, qpos.npy.

### Launcher
- Checks/compiles/resolves the new module (4 NERO modules); reports whether PyOpenGL is present for EGL/OSMesa.

### Tests
- New `tests/test_v3_08.py` (C1-C2 geometry, P1-P3 perception, F1 files, M1 MJCF injection, X1-X2 camera loop with a
  fake MuJoCo, B1 wiring). Not covered offline: MuJoCo's real renderer (checked at runtime by SELF-CHECK).

## V3.07
Based on V3.06 (2026-10-21 19:26 run: egg_1/egg_2 placed; egg_3/egg_4 `plan_grasp` failed for every
candidate with cuRobo `success=false` from the posture left by egg_2's J7 +80 deg verticalization; nothing
returned the arm to a known posture, the guard then nudged 4x for nothing). Earlier 17:53 run halted on
`[V3.05 DUPLICATE ARM CONTROLLER] publishers=2` (stale processes from the previous run).

### 1. Start-state reset inside plan_grasp
- If `plan_grasp` fails with ALL_GRASP_CANDIDATES_EXHAUSTED_OR_INFEASIBLE while untried geometry-feasible
  candidates exist, nothing is held, no safety latch, and the arm is not already at q0: vertical retreat out of
  the basket/holder zone (wrist preserved) -> SAFE_NEUTRAL q0 (`_v239_plan_exec_q0_neutral`) -> re-plan once.
  Once per egg per episode. Log `[V3.07 START-STATE RESET]` with joints, joint margin, distal wrist and the
  count of start-class (`success=false`) vs goal-class (no IK / descent) failures.
  Disable: `NERO_V307_START_STATE_RESET=0`. Edge-reposition (V2-27) keeps priority.
### 2. Post-place posture check between eggs
- When `plan_grasp` is called after PLACEMENT_VERIFIED/RELEASED: joint margin < `NERO_V307_POST_PLACE_MIN_MARGIN_DEG`
  (20) or distal wrist > `NERO_V307_POST_PLACE_MAX_DISTAL_DEG` (120) -> retreat + q0 before planning.
  Log `[V3.07 POST-PLACE POSTURE]`, result field `v307_post_place_posture`. Disable: `NERO_V307_POST_PLACE_POSTURE=0`.
### 3. Honest failure reason
- Result field `v307_start_state_reset` (attempted, q0_ok, replanned_from_q0, failure classes, start posture).
- Still failing after a q0 re-plan -> warning "object/goal-side failure, not start-state".
- q0 reset itself failed -> reason `START_STATE_PLANNING_FAILED`, next=`finish_and_report`.
- Agent: rule "V3.07 START-STATE RESET"; the completion guard no longer nudges for eggs already proven
  infeasible from q0 (`[V3.07 GUARD]`).
### 4. Launcher: stale processes + single arm_controller
- Before anything starts, `start_nero.sh` lists leftover `ros2_control_node`, `move_group`, `spawner`,
  `rviz2`, `robot_state_publisher`, `ros2 launch agx_arm_moveit`, NERO backend / GPU server (matched by
  argv, not by text, so editors/tail are untouched), SIGTERM -> SIGKILL, then `ros2 daemon stop`.
  `NERO_V307_PRELAUNCH_CLEANUP=0` -> refuse to start instead of killing.
- Before SYSTEM READY: counts action servers on `/arm_controller/follow_joint_trajectory` and publishers on
  `/arm_controller/controller_state`; more than one -> startup fails (`V3.07 DUPLICATE ARM CONTROLLER`).

### Tests
- New `tests/test_v3_07.py` (R1-R5 reset logic, G1 guard); all earlier tests pass.

## V3.06
Based on V3.05 (2026-10-21 17:25 run: egg_1-3 placed, egg_4 lifted, then
"LLM exceeded max_tool_steps=48" at plan_placement with the egg held in the air).

### 1. Step budget per object
- `max_tool_steps = max(user value, 14 x requested eggs + 8)`, capped at `NERO_V306_MAX_TOOL_STEPS_CAP`
  (default 100; the old hard cap was 60 in both Agent and GUI). 4 eggs -> 64. Log `[V3.06 STEP BUDGET]`.
  Env: `NERO_V306_STEPS_PER_OBJECT` (14), `NERO_V306_STEPS_MARGIN` (8).

### 2. Fewer fixed steps per egg (12 -> 10)
- `plan_grasp` prepares the recommended aperture itself when the gripper is outside the egg's range
  (`[V3.06 AUTO APERTURE]`, result field `v306_auto_aperture`; disable with `NERO_V306_AUTO_APERTURE=0`).
- A successful `verify_object_in_slot` returns `v306_next` (remaining eggs, next_egg, next_empty_slot,
  occupied slots, aperture) and recommends `plan_grasp` directly; no `observe_scene` between eggs.
- Prompt updated accordingly (rule 6 STEP ECONOMY); observe only after a failure.

### 3. Budget endgame
- The model sees `v306_budget` (steps_used / steps_left) once <= 14 steps remain.
- The multi-object guard does not push for another egg when < 10 steps remain (`[V3.06 GUARD SKIP]`).
- When the budget is exhausted with an egg in hand, up to `NERO_V306_FINISH_GRACE_STEPS` (6) placement-only
  tools are still allowed to finish that egg (`[V3.06 FINISH GRACE]`).
- Otherwise the supervisor calls the internal `v306_safe_end` before stopping: holding -> keep the grasp and
  rise vertically to `NERO_V306_SAFE_END_Z_M` (0.25 m); not holding -> SAFE_NEUTRAL q0 (`[V3.06 SAFE END]`).

### Tests
- New `tests/test_v3_06.py` (B1, B2, A1, V1, S1); all earlier tests pass.

## V3.05
Based on V3.04. Fixes the 2026-10-21 16:52 run where not even egg_1 was picked: every
`move_to_pregrasp` "succeeded" within ~0.4 s while MuJoCo stayed 105-136 deg from the goal,
sync/trajectory goals were rejected, and the Agent kept switching eggs from half-way postures.

### 1. Pregrasp trajectory cache OFF by default
- `move_to_pregrasp` again plans fresh with `move_to_pose` (the V3.03 path). All 6 cached executions
  in V3.04 failed. `NERO_V305_PREGRASP_CACHE=1` re-enables reuse for A/B tests. Marker `[V3.05 PREGRASP CACHE OFF]`.

### 2. Execution results must be SUCCEEDED; single controller enforced
- `_execute_trajectory` / sync now require GoalStatus == SUCCEEDED. Cancelled / pre-empted goals still carry
  `error_code=0` and were previously counted as success. Marker `[V3.05 GOAL NOT SUCCEEDED]`.
- Before every execution: more than one publisher on `/arm_controller/follow_joint_trajectory/_action/status`
  or `/arm_controller/controller_state` -> refuse (`DUPLICATE_ARM_CONTROLLER`, marker
  `[V3.05 DUPLICATE ARM CONTROLLER]`). Usually stale ros2_control/move_group from an earlier run.
  Disable with `NERO_V305_SINGLE_CONTROLLER_CHECK=0`.

### 3. Execution-layer fault -> safe posture, bounded retry, halt
- A rejected goal, a non-SUCCEEDED goal, a duplicate controller or a final-joint mismatch >= 30 deg
  (`NERO_V305_EXEC_FAULT_MISMATCH_DEG`; contact-size mismatches stay as before) is reported as
  `reason=EXECUTION_LAYER_FAULT`. If nothing is held, the arm is returned to SAFE_NEUTRAL q0
  (`[V3.05 EXEC FAULT RECOVERY]`, disable with `NERO_V305_EXEC_FAULT_RECOVERY=0`).
- 1st fault: `recommended_next_actions=[observe_scene, retry_same_step_once]` (same egg, no aperture change).
  2nd consecutive fault (`NERO_V305_EXEC_FAULT_RETRIES`, default 1) or failed recovery:
  `execution_layer_unhealthy=true`, `stop_and_report`; the Agent supervisor halts the episode
  deterministically (`[V3.05 EXECUTION HALT]`), without multi-object-guard nudges.
- Agent maps bare "rejected trajectory" exceptions to EXECUTION_LAYER_FAULT as well.

### 4. IK screen and fallback
- Pregrasp IK screen: 3 seeds x 0.20 s (was 2 x 0.08 s; egg_2 had 14/14 false "no IK at pregrasp"),
  descent links 0.15 s. Sibling offsets of the same yaw reuse the first solution as seed.
  Env: `NERO_V305_IK_SCREEN_SEEDS`, `NERO_V305_IK_SCREEN_TIMEOUT_S`, `NERO_V305_IK_DESCENT_TIMEOUT_S`.
- Fallback (screen rejected everything) only re-admits candidates whose pregrasp IK failed; a blocked or
  jumping descent stays rejected. Fallback candidates are tagged `v305_screen_fallback=true`,
  `v304_ik_descent_ok=null`. Log counters fixed (`[V3.05 GRASP PREFLIGHT] ... full plans=N ok=M`).

### Tests
- New `tests/test_v3_05.py` (X1-X4, S1-S3); all earlier tests pass.

## V3.04
Based on V3.03 (all 4 eggs placed, 2026-10-21 15:43-16:33). Adopted items 1, 3, 4 of the review.

### 1. Grasp robustness
- **Egg aperture = width + 16 mm** (band +12..+22; `NERO_V304_EGG_APERTURE_MARGIN_MM`) instead of
  1.20 x width. Evidence: with 1.20x only ~0.8 mm total free clearance remained; every approach pushed
  the egg 10-14 mm and spread the fingers from ~53 to 76-87 mm; egg_3 ended 15.3/16.0 mm off-centre.
  44 mm egg: 52.8 -> 60 mm.
- **Grasp-Z compensation +2.5 mm** (`NERO_V304_GRASP_Z_COMP_M`): the arm settles 2.4-2.8 mm below the
  commanded pregrasp/grasp Z (TCP-object 24.9 vs 28.0 mm), now it lands on the intended 28 mm.
- **Slower final approach**: descent split at 20 mm above the grasp; last segment x3.5 slower
  (`NERO_V304_FINAL_APPROACH_M`, `NERO_V304_FINAL_APPROACH_SLOW`). Marker `[V3.04 APPROACH SPEED]`.
- **Descent-aware preflight**: every candidate gets a chained-BioIK check of the pregrasp and the vertical
  descent (80/40/20/10/0 mm above grasp, collision-aware, <=15 deg joint step). V3.03 egg_1: the top
  candidate hovered fine, then "CARTESIAN APPROACH BLOCKED 87.5%" twice (~15 s each).
- **Neighbour-aware ranking**: finger-pad clearance to neighbouring eggs (ellipse model) lowers a
  candidate's score (capped, cannot promote a long-axis clamp). Logged as `neighbor_clearance_mm`.
  The bystander-egg abort (8 mm) is unchanged; if it triggers more often with the wider opening, lower
  `NERO_V304_EGG_APERTURE_MARGIN_MM` to 12-14.

### 3. Speed
- **Grasp preflight**: full cuRobo+MoveIt plan only for the first 2 candidates that pass the cheap
  IK/descent screen (`NERO_V304_FULL_PREFLIGHT`), instead of up to 8 (~1 s each). Safety net: if the
  screen rejects everything, the old full planning runs for the best 4.
- **Trajectory reuse**: `move_to_pregrasp` executes the trajectory planned by `plan_grasp` if the arm
  moved < 0.6 deg since (`[V3.04 PREGRASP CACHE]`); otherwise it replans as before.
- **Slot precheck only when the egg is near vertical** (<= 30 deg, `NERO_V304_PRECHECK_MAX_AXIS_DEG`).
  After lift it is ~90 deg off and the screen is meaningless (V3.03: 0/12 on all 8 slots, 15.5 s, yet
  slot_1 succeeded). Marker `[V3.04 SLOT REACH SKIPPED]`.
- **Fewer LLM round trips / tokens**: `set_gripper_opening` now recommends `plan_grasp` (no extra
  observe_scene); the Agent sends a compact copy of each tool result to the model (no static contract,
  no state_before, compact grasp priors and slots, top-3 grasp candidates). Full results are still
  logged. Disable with `NERO_V304_COMPACT_TOOL_OUTPUT=0`. plan_grasp result ~7k -> ~0.8k chars.

### 4. Slot fill order
- observe_scene adds `egg_holder_fill_order.next_empty_slot` (lowest-numbered empty slot).
- Agent rule: "first/second/N-th empty slot", "next slot" etc. mean the next slot in numeric fill order;
  only an explicit slot_N overrides. (V3.03 run left slot_2/4/6 empty by counting among empty slots.)

Tests: `tests/test_v3_04.py` (P1-P5) + all earlier tests via `tests/check_release.sh`.
Not changed: placement chain (V3.03), control law, safety thresholds, hold model (clamp-force criterion
deferred to its own version as agreed).

## V3.03
Based on V3.02. Fixes the egg_2 -> slot_3 failure from the 2026-10-21 15:15 run
(`move_to_slot_entry`: cuRobo `plan_pose returned None` x10, raw exception, Agent gave up).
Root cause: the yaw chosen at slot-above (egg centre z=0.298 m) had **no IK solution at the
slot-entry depth** (z=0.123 m). V3.02 only checked reachability at hover height.

1. **Descent-corridor gate** (`orient_for_slot`, every slot-above goal). A yaw is accepted only if
   the SAME TCP orientation has a chained BioIK solution, seeded level by level every 40 mm,
   down to the insertion end depth (entry - 20 mm), each step <= 20 deg per joint
   (`NERO_V303_CORRIDOR_MAX_STEP_DEG`). After execution the corridor is re-checked from the
   joint branch cuRobo actually took; if that fails, one more yaw is tried at the hover.
   If no yaw has a corridor: `reason=V303_SLOT_DESCENT_UNREACHABLE`, egg still held, stage LIFTED,
   next = plan_placement with another slot. Disable: `NERO_V303_CORRIDOR=0`.
   Markers: `[V3.03 CORRIDOR]`, `[V3.03 CORRIDOR REJECT]`, `[V3.03 CORRIDOR POST-CHECK OK/FAIL]`.
2. **Lower slot hover.** Slot-above is now entry + 0.10 m (egg centre ~0.22 m) instead of the
   transport height (0.26-0.30 m); `NERO_V303_HOVER_ABOVE_ENTRY_M`. Orientation still happens high.
3. **Layered `move_to_slot_entry`**: L1 straight Cartesian descent at fixed orientation (rejects
   >10 deg joint jumps between waypoints); L2 cuRobo pose plan (old behaviour); L3 rotate the
   vertical egg about its own axis AT HOVER (yaw family, BioIK-screened hover + entry on one
   branch, max 3 tries), then L1. All failures are structured: `reason=SLOT_ENTRY_UNREACHABLE`,
   stage reset to LIFTED, next = plan_placement with another slot. Marker `[V3.03 SLOT ENTRY ...]`.
4. **Early stop on unreachable goals** (V71 `_plan_pose`): 3 consecutive `plan_pose returned None`
   -> `GOAL_IK_UNREACHABLE` (a new seed cannot fix a goal with no IK). `NERO_V303_NONE_ABORT`.
   Also shortens failing grasp preflights.
5. **Slot precheck** (`plan_placement`) uses the new hover height, up to 3 seeds, and counts a yaw
   only if the same branch also reaches the entry depth.
6. **Re-plan after slot change**: `orient_for_slot` skips the wrist search when the egg is already
   vertical (`[V3.03 ALREADY VERTICAL]`).
7. **Agent**: new rule - on SLOT_ENTRY_UNREACHABLE / V303_SLOT_DESCENT_UNREACHABLE, plan_placement
   ONCE with another empty slot and continue; GOAL_IK_UNREACHABLE is not retried. A tool that raises
   now returns `reason=BACKEND_EXCEPTION` + `recommended_next_actions` instead of a bare error.

Offline evidence (URDF numeric IK, kinematics only, 12 yaws x 8 slots, chained 5 mm descent from
0.298 m): only 4-6 of 12 yaws reach any slot-above at all, and some that do stop part-way
(slot_2 yaw 0 at 208 mm, slot_6 at 213/138 mm, slot_7 at 253/203/128 mm). slot_3 has continuous
corridors on some branches, so whether a descent works depends on the yaw AND the joint branch
actually reached - which V3.02 never checked.
New tests: `tests/test_v3_03_corridor_entry.py` (D0 shows the V3.02 choice failing; D/E1-E3 pass).

Not changed: grasp/approach/clamp/lift, Strategy 1-3 logic, guarded insertion/release, control law,
safety thresholds, V3.01 timeout + diagnostics.

## V3.02
Based on V3.01. Changes only in held-egg orientation / slot transport (`orient_for_slot`),
a planning-only precheck in `plan_placement`, and one Agent prompt rule.

1. **Yaw-free vertical goals.** Every "egg vertical at X" goal (Strategy 2, slot-above transport,
   hold-pose -> slot-above) is now a 12-member family rotated every 30 deg about world Z.
   An egg is a body of revolution, so all are task-equivalent. BioIK screens the family
   (current seed, then 2 wrist-centre seeds), cuRobo plans only the best 4, MoveIt + held-object
   checks unchanged. Old behaviour: ONE orientation per goal -> `plan_pose returned None` x10.
   Log markers: `[V3.02 IK SCREEN]`, `[V3.02 VERTICAL GOAL PLAN]`, `[V3.02 WHOLE-ARM SUCCESS]`.
2. **q=0 fallback replaced by a non-singular SAFE-HIGH hold.** q=0 has J1/J3/J5 collinear
   (Jacobian min singular value 0). V3.01 diag: RUNAWAY at 93% of the q0 path with J3 -6.1 rad/s
   vs J5 +3.6 rad/s, then divergence after freeze (J6 54.9 rad/s). New fallback: slow validated
   joint move into J2/J4 = 15/75 (or 10/80, 20/70) deg, J3=J5=0, J6 in {0,+-20}, J1 in {0,+-15},
   J7 swept; pose chosen by batch FK so the held egg is already vertical and high
   (offline: 54/837 family poses vertical <= 6 deg; min singular value ~0.14; no collinear pair).
   Then one yaw-free trajectory to slot-above. Markers: `[V3.02 SAFE-HIGH HOLD ...]`.
   The legacy q0 path still exists but only with `NERO_V302_ALLOW_Q0_FALLBACK=1`.
   The V2-51 'q0 timeout -> safe-high handoff' lived inside the old q0 block and is retired with it
   (V3.01 already scales the execution deadline with trajectory duration; a timeout of the new
   safe-high move is a structured failure with the egg still held).
3. **Slot reachability precheck** in `plan_placement` (no motion): BioIK checks 12 yaws at the
   slot-above goal. Only if the requested slot has 0 AND another empty slot has >= 2, the plan is
   refused with `SLOT_ABOVE_UNREACHABLE_FROM_CURRENT_GRASP` + `recommended_slot`.
   Disable with `NERO_V302_SLOT_PRECHECK=0`. Marker: `[V3.02 SLOT REACH]`.
4. **Agent prompt**: one rule - on that reason, re-plan once with `recommended_slot`.

Not changed: Strategy 1 (sparse wrist), grasp / approach / clamp / lift, slot entry, guarded
insertion, release, control law, safety thresholds, V3.01 timeout + diagnostics.
Deferred to V3.03 (needs the V3.01 diag CSVs): held-object control-model fix, grasp centring / aperture.

## V3.01
Based on V3.00-BASELINE (= V2-52.3 behavior). Only these changes:

1. **Duration-aware execution timeout** (`NERO_V71_Hybrid_GPU_Planner_Core.py`, `_execute_trajectory`).
   The fixed 45.0 s deadline was shorter than valid slow trajectories. Evidence: q0 fallback
   RETIME 27.78 s x slow_factor 1.65 = 45.8 s -> cancelled at 45.0 s with the arm ~1 s from q0
   (log 2026-10-21 14:43:16 -> 14:44:01). Deadline is now `max(45, 1.15 x planned + 15)`, capped 180 s,
   and logged as `[V3.01 EXEC TIMEOUT]` when it exceeds 45 s.
2. **V2-51 safe-high timeout handoff made reachable** (backend, `_v224_exec_joint_path`).
   `_v24_run_motion` re-raises any exception except three whitelisted ones, so the timeout skipped
   the V2-51 handoff code completely (it never ran). The orient joint-path executor now returns the
   timeout as a structured failure; the V2-51 checks (hold, safety, z >= 700 mm, state sync) run unchanged.
   Log marker: `[V3.01 TIMEOUT -> HANDOFF CHECK]`.
3. **Motion diagnostics, record only** (backend, MuJoCo process). While a trajectory executes,
   `logs/diag/diag_motion.csv` gets 20 Hz samples (q, qd, commanded qacc, actual qacc, torque, hold).
   Any automatic safety trip dumps the last ~1.5 s of every physics step to
   `logs/diag/diag_trip_<n>_<reason>.csv`. Control law, thresholds and safety behavior are unchanged.

Not changed: q=0 fallback target, Strategy 1/2 search, grasp, insertion, Agent prompt.

## V3.00-BASELINE
- Single-folder release. All NERO Python files run in place from this folder.
- One launcher `start_nero.sh` replaces the chain
  `start_nero_v2_52_3.sh -> v243/start_nero_v2_43_cuda_sanitized.sh -> runtime/start_nero_v2_7_2.sh`
  (same order: CUDA check -> cuRobo server + self-test -> backend -> BioIK MoveIt -> TRAC-IK MoveIt).
- No runtime patching / no file copying into `~/mujoco_ws/nero`.
- Startup integrity check (SHA256SUMS) + module-resolution check (path + hash of every loaded NERO module).
- Backend code change (path only, 2 lines): the MuJoCo model path now comes from
  `NERO_MODEL_FILE` (set by the launcher to `~/mujoco_ws/nero/nero_motor.xml`), and the gripper-mesh
  search starts from the model's folder. Previously both were derived from the backend's own location,
  which was the same folder. Generated temp XML/texture files are still written next to `nero_motor.xml`.
- Behavior baseline: V2-52 + RUN TASK Safety-Latch Gate (V2-52.3), V71 = V2-43 core + V2-51 timeout handoff.
