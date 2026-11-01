#!/usr/bin/env python3
"""V3.55 GUI test for Main Camera Hand-Eye Calibration (needs tkinter + DISPLAY; skips otherwise)."""
import math
import os
import random
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    print("[GUI55] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI55] no DISPLAY -> skipped"); sys.exit(0)

import NERO_V325_Camera as CAM               # noqa: E402
import NERO_V315_Consistency_Safety as CS    # noqa: E402

TMP = tempfile.mkdtemp(prefix="v355gui_")
CAM.CALIB_PATH = os.path.join(TMP, "camera_calib.json")
CAM.LOCATE_PATH = os.path.join(TMP, "locate_settings.json")
CAM.CALIB_DIR = os.path.join(TMP, "cameracalib")
CAM.CAMERA_INFO_PATH = os.path.join(TMP, "camera_info.json")
CAM.SETTLE_S = 0.0
CAM.FAST_SETTLE_S = 0.0
CAM.FAST_STILL_HOLD_S = 0.0

class SyncThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None, name=None):
        self.t, self.a, self.k, self.alive = target, args, kwargs or {}, False
    def start(self):
        self.alive = True
        try:
            self.t(*self.a, **self.k)
        finally:
            self.alive = False
    def is_alive(self):
        return self.alive

CAM.threading = type("T", (), {"Thread": SyncThread, "Lock": threading.Lock, "Event": threading.Event})
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
LIM = [(-2.70526, 2.70526), (-1.74, 1.74), (-2.75, 2.75), (-1.01, 2.14), (-2.75, 2.75), (-0.73, 0.95), (-1.5708, 1.5708)]

def fk(q_deg):
    return [list(map(float, r)) for r in CS.fk(U, dict(zip(J, [math.radians(v) for v in q_deg])), "base_link", "link7")]

def look_at(pos, tgt):
    f = CAM.vsub(tgt, pos); n = CAM.vnorm(f); f = [v / n for v in f]
    x = [f[1], -f[0], 0.0]; n = CAM.vnorm(x); x = [v / n for v in x]
    yu = [x[1] * f[2] - x[2] * f[1], x[2] * f[0] - x[0] * f[2], x[0] * f[1] - x[1] * f[0]]
    return CAM.T_from([[x[i], -yu[i], f[i]] for i in range(3)], pos)

K = {"width": 1280, "height": 720, "fx": 908.7, "fy": 908.7, "ppx": 640.1, "ppy": 366.6}
Q0 = [0.0, 30.0, 0.0, 80.0, 0.0, 40.0, 0.0]
T0 = fk(Q0)
Y = [0.015, -0.025, 0.06]
M0 = CAM.T_apply(T0, Y)
X = look_at([M0[0] + 0.35, M0[1] - 0.70, M0[2] + 0.45], M0)
XI = CAM.T_inv(X)
nb0 = CAM.vsub(CAM.T_t(X), M0)
N_F = CAM.mat_vec(CAM.transpose(CAM.T_R(T0)), [v / CAM.vnorm(nb0) for v in nb0])

class Pipe:
    def __init__(self):
        self.q, self.teach_mode, self.real_urdf, self.moves = list(Q0), False, None, 0
    def actual_arm_deg(self, model):
        return list(self.q)
    def move_and_settle(self, model, target_rad, speed, label="", **k):
        self.q = [math.degrees(v) for v in target_rad]
        self.moves += 1
        return {"ok": True}
    def fence_goal_check(self, model, q):
        return True, "", []
    def cancel_move(self):
        pass

class Safe:
    def __init__(self):
        self.sim, self.valid_ws = U, {"fence": {"polygon": [[-0.7, -0.4], [-0.1, -0.4], [-0.1, 0.4], [-0.7, 0.4]], "z_table": 0.0}}
    def arm_points(self, q):
        return [("link4", CAM.T_t(fk([math.degrees(v) for v in q])))]

class FakeMixedD435i:
    """Simulates D435i where some poses have depth and others lose IR depth (falling back to scaled PnP)."""
    def __init__(self):
        self.K = self.info = None
        self.dist, self.error, self.fps, self.fid, self.on = [0.0] * 5, None, 30.0, 0, False
        self.rnd = random.Random(5)
        self.call_k = 0
    def running(self): return self.on
    def start(self):
        self.on, self.K = True, dict(K)
        self.info = {"name": "D435I (fake)", "serial": "261222075224", "usb": "3.2", "K": dict(K)}
    def stop(self, timeout=0): self.on = False
    @property
    def frame_id(self):
        self.fid += 1
        return self.fid
    def aruco_status(self): return True, "fake"
    def display_ppm(self, view="color", size=None): return None
    def depth_at(self, u, v): return 0.8
    def roi_points(self, *a, **k): return [], 0
    def frames(self): return None, None, self.fid
    def detect_marker(self, marker, q_deg=None):
        if not q_deg: return None
        self.call_k += 1
        T = fk(q_deg)
        p = CAM.T_apply(XI, CAM.T_apply(T, Y))
        nc = CAM.mat_vec(CAM.T_R(XI), CAM.mat_vec(CAM.T_R(T), N_F))
        uv = CAM.project(p, K)
        cosv = -sum(a * b for a, b in zip(nc, p)) / CAM.vnorm(p)
        if not uv or not (0 < uv[0] < 1280 and 0 < uv[1] < 720) or cosv < math.cos(math.radians(60)):
            return None
        p_dep = [v + self.rnd.gauss(0, 0.0005) for v in p]
        p_pnp = [v * 1.020 + self.rnd.gauss(0, 0.0004) for v in p]
        has_dep = (int(round(q_deg[0] * 10)) % 3 != 0)
        side = 908.7 * marker["size_mm"] / 1000.0 / p[2]
        c = [[uv[0] - side / 2, uv[1] - side / 2], [uv[0] + side / 2, uv[1] - side / 2],
             [uv[0] + side / 2, uv[1] + side / 2], [uv[0] - side / 2, uv[1] + side / 2]]
        return {"id": marker["id"], "corners_px": c, "center_px": uv,
                "p_pnp": p_pnp, "p_depth": p_dep if has_dep else None,
                "p_cam": p_dep if has_dep else p_pnp,
                "source": "depth" if has_dep else "pnp", "normal_cam": nc, "side_px": side}

root = tk.Tk()
frame = ttk.Frame(root)
frame.pack(fill="both", expand=True)
logs = []
jp = type("JP", (), {"enabled": True, "model": {"joints": J, "limits": [list(l) for l in LIM], "base_link": "base_link"}})()
pipe = Pipe()
cam = CAM.CameraTabUI(tk, ttk, frame, pipe, jp, Safe(), lambda: {"speed_percent": 10}, logs.append, lambda *a: True,
                      visible_fn=lambda: True, source_factory=FakeMixedD435i)
cam.on_connect()
for _ in range(4):
    cam.poll(); root.update()

# Add a stale garbage sample first to verify on_auto clears stale samples
cam.samples.append({"q_deg": [0]*7, "p_cam": [9.0, 9.0, 9.0], "T_base_flange": T0, "spread_mm": 0.1, "n": 5, "n_frames": 5, "source": "pnp"})
cam.v_nauto.set("12")
t_start = time.time()
cam.on_auto()
for _ in range(4):
    cam.poll(); root.update()
dt_s = time.time() - t_start
assert cam.result and cam.result["ok"] and cam.result["good"], cam.result
assert cam.result["loo_max_mm"] <= 5.0 and cam.result["rot_span_deg"] >= 25.0, cam.result
assert any("已自动清空旧样本" in l for l in logs), logs
root.destroy()
print(f"ALL V3.55 GUI TESTS PASSED (dt={dt_s:.2f}s, LOO max={cam.result['loo_max_mm']:.2f} mm, rot_span={cam.result['rot_span_deg']:.1f} deg)")
