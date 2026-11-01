#!/usr/bin/env python3
"""V3.45 GUI test of the 手眼标定 page (needs tkinter + a display; skips otherwise).

The Camera tab is built alone (stub pipe / joint panel / safety tab, a fake D435i with a fixed camera, a marker on the
flange, 0.6 mm PnP noise and 10 % bad frames).  Worker jobs run synchronously in the Tk thread.
G1 four steps ①-④ with lamps; table with vertical + horizontal scrollbar and a row height from the font
G2 buttons follow the state (no camera -> only 生成标记图片; marker seen -> 记录 / 自动; < 6 samples -> 计算 grey ...)
G3 auto collection: probes + visibility-aware poses -> >= 12 samples, span >= 20 cm, wrist <= 25 deg -> 合格 -> save
   -> quick check green
G4 redraw with the detection flipping to None (old refresh bug) and without a safety zone: no exception
"""
import math
import os
import random
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
try:
    import tkinter as tk
    from tkinter import ttk
except Exception:
    print("[GUI45] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI45] no DISPLAY -> skipped"); sys.exit(0)
import NERO_V325_Camera as CAM               # noqa: E402
import NERO_V315_Consistency_Safety as CS    # noqa: E402

TMP = tempfile.mkdtemp(prefix="v345gui_")
CAM.CALIB_PATH = os.path.join(TMP, "camera_calib.json")
CAM.LOCATE_PATH = os.path.join(TMP, "locate_settings.json")      # V3.47.1: never the user's files
CAM.CALIB_DIR = os.path.join(TMP, "cameracalib")
CAM.CAMERA_INFO_PATH = os.path.join(TMP, "camera_info.json")
CAM.SETTLE_S = 0.0


class SyncThread:                                     # worker jobs + detection run inline (Tk thread only)
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


class Fake:
    def __init__(self):
        self.K = self.info = None
        self.dist, self.error, self.fps, self.fid, self.on = [0.0] * 5, None, 30.0, 0, False
        self.rnd = random.Random(2)

    def running(self):
        return self.on

    def start(self):
        self.on, self.K = True, dict(K)
        self.info = {"name": "D435I (fake)", "serial": "1", "usb": "3.2", "K": dict(K)}

    def stop(self, timeout=0):
        self.on = False

    @property
    def frame_id(self):
        self.fid += 1
        return self.fid

    def aruco_status(self):
        return True, "fake"

    def display_ppm(self, view="color"):
        return None                                   # no image: the page logic is under test, not the picture

    def depth_at(self, u, v):
        return 0.8

    def roi_points(self, *a, **k):
        return [], 0

    def frames(self):
        return None, None, self.fid

    def detect_marker(self, marker, q_deg=None):
        if not q_deg:
            return None
        T = fk(q_deg)
        p = CAM.T_apply(XI, CAM.T_apply(T, Y))
        nc = CAM.mat_vec(CAM.T_R(XI), CAM.mat_vec(CAM.T_R(T), N_F))
        uv = CAM.project(p, K)
        cosv = -sum(a * b for a, b in zip(nc, p)) / CAM.vnorm(p)
        if not uv or not (0 < uv[0] < 1280 and 0 < uv[1] < 720) or cosv < math.cos(math.radians(60)):
            return None
        bad = self.rnd.random() < 0.1
        pn = [v + self.rnd.gauss(0, 0.0006) + (0.03 if bad else 0.0) for v in p]
        side = 908.7 * marker["size_mm"] / 1000.0 / p[2]
        c = [[uv[0] - side / 2, uv[1] - side / 2], [uv[0] + side / 2, uv[1] - side / 2],
             [uv[0] + side / 2, uv[1] + side / 2], [uv[0] - side / 2, uv[1] + side / 2]]
        return {"id": marker["id"], "corners_px": c, "center_px": uv, "p_pnp": pn, "p_depth": None, "p_cam": pn,
                "source": "pnp", "normal_cam": nc, "side_px": side}


root = tk.Tk()
root.geometry("1400x900")
frame = ttk.Frame(root)
frame.pack(fill="both", expand=True)
logs = []
jp = type("JP", (), {"enabled": True, "model": {"joints": J, "limits": [list(l) for l in LIM], "base_link": "base_link"}})()
pipe = Pipe()
cam = CAM.CameraTabUI(tk, ttk, frame, pipe, jp, Safe(), lambda: {"speed_percent": 10}, logs.append, lambda *a: True,
                      visible_fn=lambda: True, source_factory=Fake)
for w in frame.winfo_children():
    pass
nbk = [w for w in frame.winfo_children()[1].winfo_children()[1].winfo_children() if w.winfo_class() == "TNotebook"][0]
nbk.select(2)


def tick(n=3):
    for _ in range(n):
        cam.poll()
        root.update()


def st(btn):
    return str(btn.cget("state"))


tick()
# ------------------------------------------------------------------ G1
lamps = [cam.l1, cam.l2, cam.l3, cam.l4]
assert all(l.winfo_class() == "Canvas" for l in lamps)
style_h = int(ttk.Style(root).lookup("V345Calib.Treeview", "rowheight"))
ls = int(root.tk.call("font", "metrics", "TkDefaultFont", "-linespace"))
assert style_h == ls + 8, (style_h, ls)
assert str(cam.tv.cget("yscrollcommand")) and str(cam.tv.cget("xscrollcommand"))
assert [cam.tv.heading(c, "text") for c in cam.tv["columns"]] == ["#", "来源", "关节角 J1…J7 (°)", "标记 xyz (mm)", "抖动", "残差", "留一"]
# ------------------------------------------------------------------ G2 (no camera)
assert getattr(cam.l1, "_v345_state") == "grey" and "连接相机" in cam.s1.get()
for b in (cam.b_rec, cam.b_auto, cam.b_stop, cam.b_del, cam.b_clr, cam.b_comp, cam.b_save, cam.b_check):
    assert st(b) == "disabled", b
cam.on_connect()
tick(4)
assert cam.det is not None and getattr(cam.l1, "_v345_state") == "green" and "看到 ID 0" in cam.s1.get(), cam.s1.get()
assert st(cam.b_rec) == "normal" and st(cam.b_auto) == "normal" and st(cam.b_comp) == "disabled"
cam.on_record()
tick()
assert len(cam.samples) == 1 and getattr(cam.l2, "_v345_state") == "yellow" and st(cam.b_comp) == "disabled", logs[-3:]
print(f"[G1] 4 steps with lamps; table rowheight {style_h} px = font {ls} + 8, vertical + horizontal scrollbar")
print("[G2] no camera -> every action grey, lamp ① grey; marker seen -> ① green, 记录 / 自动 enabled; 1 sample -> ② yellow")
# ------------------------------------------------------------------ G3
cam.v_nauto.set("14")
cam.on_auto()
tick()
L = "\n".join(logs)
n = len(cam.samples)
ts, rs = cam._spans()
assert n >= 12 and ts >= 0.20 and rs >= 25, (n, ts, rs, L[-2000:])
assert "相机方向已确定" in L and "选出" in L and "自动采集结束" in L, L[-1500:]
tick()
r = cam.result
assert r and r["ok"] and r["good"], (r and (r["loo_mean_mm"], r["loo_max_mm"], r["warnings"]))
assert getattr(cam.l2, "_v345_state") == "green" and getattr(cam.l3, "_v345_state") == "green"
assert st(cam.b_save) == "normal" and cam.b_save.cget("text") == "保存标定" and "合格 ✓" in cam.txt_res.get("1.0", "end")
cam.on_save_calib()
tick()
assert cam.calib and st(cam.b_check) == "normal" and getattr(cam.l4, "_v345_state") == "grey"
cam.on_quick_check()
tick()
assert getattr(cam.l4, "_v345_state") == "green" and "通过" in cam.s4.get(), (cam.s4.get(), logs[-3:])
rows = cam.tv.get_children()
assert len(rows) == n and cam.tv.set(rows[0], "src") == "PnP"
print(f"[G3] auto: {n} samples, span {ts * 100:.0f} cm, rotation {rs:.0f} deg -> loo {r['loo_mean_mm']:.2f}/"
      f"{r['loo_max_mm']:.2f} mm 合格 -> saved -> quick check ④ green ({cam.s4.get()})")
# ------------------------------------------------------------------ G4
orig = Fake.detect_marker
flip = {"k": 0}


def flaky(self, marker, q_deg=None):
    flip["k"] += 1
    return None if flip["k"] % 2 else orig(self, marker, q_deg)


Fake.detect_marker = flaky
cam.safe_tab.valid_ws = None
errs = [m for m in logs if "refresh" in m]
for _ in range(20):
    cam._force_redraw()
    cam._tick()
    root.update()
assert not [m for m in logs if "refresh" in m and m not in errs], [m for m in logs if "refresh" in m]
print("[G4] detection flipping to None + no safety zone: 20 redraws without a refresh error")
if os.environ.get("V345_SHOT"):
    Fake.detect_marker = orig
    cam.safe_tab.valid_ws = Safe().valid_ws
    tick(4)
    root.update()
    os.system(f"import -window root {os.environ['V345_SHOT']}")
cam.shutdown()
root.destroy()
print("ALL V3.45 GUI TESTS PASSED")
