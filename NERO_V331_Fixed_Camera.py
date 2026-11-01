#!/usr/bin/env python3
"""
NERO V3.31 FIXED CAMERA WITHOUT PRIORS - process entry for the fixed stand D435i (replaces
NERO_V308_Camera_Perception.camera_process_main when NERO_V331_FIXED_NO_PRIOR=1, the default)
=================================================================================================

Same rendering, RGB-D egg perception (V3.10 perceive_eggs) and result protocol as V3.08-V3.11 - the V3.08 module is
NOT modified - but without anything the simulator or the scene file "knows":
  * no identity seeding from the simulator: eggs are named by perception (V3.30 ViewTracker generic naming);
  * no registered receptacle layout: the process-local copy of V3.08 BASKET / SLOTS is emptied; the egg holder is
    found in the depth image (V3.29 detect_holder) and its slots are used from then on;
  * backend hints (held / placed) as for the wrist camera;
  * the simulator is used only for the evaluation summary that is written to the log (never sent to the skills).
"""
from __future__ import annotations

import json
import os
import time
import traceback

VERSION = "V3.31"


def enabled():
    return os.environ.get("NERO_V331_FIXED_NO_PRIOR", "1") != "0"


def fixed_camera_process_main(qpos_arr, meta_arr, model_path_arr, trig_seq, trig_tag, status, stop_event, out_root,
                              gl_name, res_seq=None, res_buf=None):
    prefix = f"[{VERSION} FIXED CAMERA]"
    try:
        import mujoco
        import numpy as np
        import NERO_V308_Camera_Perception as P8
        import NERO_V326_Wrist_Camera as W
    except Exception as exc:
        print(f"{prefix} import failed: {exc}", flush=True)
        status.value = 2
        return
    cfg = P8.camera_config()
    t_wait = time.time()
    path = ""
    while not stop_event.is_set():
        path = bytes(model_path_arr[:]).split(b"\0", 1)[0].decode("utf-8", "ignore")
        if path and meta_arr[3] > 0.5:
            break
        if time.time() - t_wait > 180:
            print(f"{prefix} simulator never published its model; camera disabled", flush=True)
            status.value = 2
            return
        time.sleep(0.2)
    if stop_event.is_set():
        return
    try:
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, P8.CAMERA_NAME)
        if cam_id < 0:
            raise RuntimeError(f"camera {P8.CAMERA_NAME} not in model {path}")
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(cfg["width"]))
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(cfg["height"]))
    except Exception as exc:
        print(f"{prefix} model load failed: {exc}", flush=True)
        status.value = 2
        return
    try:
        renderer = mujoco.Renderer(model, height=int(cfg["height"]), width=int(cfg["width"]))
    except Exception as exc:
        print(f"{prefix} OpenGL backend '{gl_name}' unavailable: {str(exc)[:200]}", flush=True)
        status.value = 2
        raise SystemExit(P8.EXIT_GL_FAILED)
    K = P8.intrinsics(cfg)
    T = P8.extrinsics(cfg)                    # camera calibration (stand pose), not object knowledge
    writer = P8.CaptureWriter(out_root, cfg, K, T, save_images=os.environ.get("NERO_V308_SAVE_IMAGES", "1") != "0")
    rng = np.random.default_rng(331)
    nq = int(model.nq)
    robot_geoms = P8._robot_geom_ids(mujoco, model)
    names = [n for n in os.environ.get("NERO_V330_OBJECT_NAMES", "egg_1,egg_2,egg_3,egg_4").split(",") if n]
    tracker = W.ViewTracker(P8.PerceptionTracker(), names=names)
    P8.SLOTS = {}                             # process-local: forget the registered layout
    P8.BASKET = {"center_xy_m": [0.0, 0.0], "radius_m": 0.0}
    holder_best = None
    prev_dets = []
    capture_no = 0
    last_seq = int(trig_seq.value)
    status.value = 1
    print(f"{prefix} READY (no priors: generic naming, perceived holder) gl={gl_name} {K['width']}x{K['height']} "
          f"fx={K['fx']:.1f} pos={cfg['pos']} -> {writer.run_dir}", flush=True)
    while not stop_event.is_set():
        cur = int(trig_seq.value)
        if cur == last_seq:
            time.sleep(0.05)
            continue
        last_seq = cur
        raw_tag = bytes(trig_tag[:]).split(b"\0", 1)[0].decode("utf-8", "ignore") or "trigger"
        tag, hints = W.decode_hints(raw_tag)
        t0 = time.perf_counter()
        try:
            with qpos_arr.get_lock():
                n = int(meta_arr[0])
                q = np.frombuffer(qpos_arr.get_obj(), dtype=np.float64, count=n).copy()
                sim_time = float(meta_arr[1])
            if n != nq:
                raise RuntimeError(f"qpos size mismatch sim={n} camera_model={nq}")
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            renderer.disable_depth_rendering()
            renderer.update_scene(data, camera=cam_id)
            rgb = renderer.render().copy()
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=cam_id)
            raw_depth = renderer.render().copy()
            renderer.disable_depth_rendering()
            n_fr = int(cfg["depth_frames"]) if float(cfg["depth_noise_k"]) > 0 else 1
            depth = P8.temporal_fuse([P8._depth_postprocess(raw_depth, cfg, rng) for _ in range(n_fr)])
            boxes = P8._robot_boxes(model, data, robot_geoms)
            prior = [d for d in prev_dets if d.get("trusted")] if os.environ.get("NERO_V310_WARM_START", "1") != "0" else []
            dets, diag = P8.perceive_eggs(rgb, depth, K, T, robot_boxes=boxes, prior=prior)
            diag["depth_frames_fused"] = n_fr
            diag["camera"] = "fixed_d435i_no_prior"
            P8._temporal_annotate(dets, prev_dets)
            prev_dets = dets
            holder = None
            try:
                holder = W.detect_holder(depth, K, T, dets=dets, robot_boxes=boxes)
            except Exception as exc_h:
                diag["holder_error"] = str(exc_h)[:200]
            if holder is not None and W.holder_quality(holder) >= W.holder_quality(holder_best):
                holder_best = holder
                P8.SLOTS = {k: (v["xyz_m"][0], v["xyz_m"][1]) for k, v in holder["slots"].items()}
            diag["holder"] = {k: v for k, v in (holder or {}).items() if k != "slots"} if holder else None
            tracked, track_diag = tracker.update(dets, capture_no + 1, K, T.tolist(), seed_match=None, hints=hints,
                                                 z_max=float(cfg["max_range_m"]))
            diag["tracking"] = dict(track_diag)
            # LOG ONLY: simulator truth for the evaluation summary (never sent to the skills)
            truth = P8._truth_from_data(mujoco, model, data)
            rows, fps, _match = P8.evaluate(dets, truth)
            summ = P8.summarize(rows, fps)
            capture_no += 1
            elapsed = time.perf_counter() - t0
            cdir = writer.write(capture_no, tag, sim_time, rgb, depth, q, dets, truth, rows, fps, summ, diag, K, T, elapsed)
            P8._post_result(res_seq, res_buf, cur, W.json_safe({   # V3.38: numpy-safe
                "seq": cur, "capture": capture_no, "tag": tag, "sim_time_s": round(sim_time, 3),
                "elapsed_s": round(elapsed, 3), "objects": tracked, "n_detections": len(dets),
                "unassigned_detections": track_diag["unassigned_detections"], "dir": cdir.name,
                "camera": "fixed_d435i", "containers": {"egg_holder": holder},
                "tracking": {"out_of_view": track_diag.get("out_of_view", []), "held_hint": track_diag.get("held_hint"),
                             "placed_hint": track_diag.get("placed_hint", []), "named": track_diag.get("named", []), "unnamed_trusted": track_diag.get("unnamed_trusted", 0),
                             "reassociated_moved": track_diag.get("reassociated_moved", [])}}))
            xy = summ["err_xy_mm"]
            print(f"[{VERSION} FIXED PERCEPTION] #{capture_no} {tag} | detected {summ['detected']}/{summ['truth']} "
                  f"(+{summ['false_positives']} FP) | xy mean {xy['mean'] if xy else '-'} max {xy['max'] if xy else '-'} mm "
                  f"(log only) | holder {'complete' if (holder or {}).get('complete') else ('partial' if holder else '-')} "
                  f"| named {track_diag.get('named', [])} | {elapsed:.2f}s -> {cdir.name}", flush=True)
        except Exception as exc:
            print(f"[{VERSION} FIXED PERCEPTION ERROR] {tag}: {exc}", flush=True)
            traceback.print_exc()
            try:
                P8._post_result(res_seq, res_buf, cur, {"seq": cur, "tag": tag, "error": str(exc)[:300]})
            except Exception:
                pass
    try:
        renderer.close()
    except Exception:
        pass
    status.value = 0
