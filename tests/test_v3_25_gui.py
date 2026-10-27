#!/usr/bin/env python3
"""V3.25 GUI test (needs tkinter + a display). Skips otherwise.

Tabs: 「实机NERO调试」 (renamed) + new 「Camera调试」. A pure-Python fake D435i (camera pose + marker offset known,
0.3 mm noise, table plane depth) stands in for the RealSense:
connect -> info; 自动采集 12 poses through MoveIt (fake monitor, fence) -> 计算标定 recovers the camera pose within
3 mm and the result is 合格; 保存标定; click on the image -> base_link coordinates of a table point within 3 mm;
深度检查 -> table z vs the safety zone; live marker error; 快速复核 moves to the stored poses; overlays draw.
"""
import json, math, os, random, sys, tempfile, time
HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
try:
    import tkinter as tk
    from tkinter import messagebox
except Exception:
    print("[GUI25] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI25] no DISPLAY -> skipped"); sys.exit(0)
import NERO_V312_Console as C
import NERO_V314_Joint_Control as JC
import NERO_V315_Consistency_Safety as CS
import NERO_V316_Safety_Editor as SE
import NERO_V313_Real_Robot as RR
import NERO_V325_Camera as CAM

TMP = tempfile.mkdtemp(prefix="v325gui_")
SE.STATE_PATH = os.path.join(TMP, "state.json")
CS.WORKSPACE_PATH = os.path.join(TMP, "ws.json")
CS.LOG_DIR = os.path.join(TMP, "logs")
CAM.CALIB_PATH = os.path.join(TMP, "camera_calib.json")
CAM.LOCATE_PATH = os.path.join(TMP, "locate_settings.json")      # V3.47.1: never the user's files
CAM.CALIB_DIR = os.path.join(TMP, "cameracalib")
CAM.CAMERA_INFO_PATH = os.path.join(TMP, "camera_info.json")
CS.save_workspace(CS.default_workspace(), CS.WORKSPACE_PATH)
messagebox.askyesno = lambda *a, **k: True
U = CS.load_sim_urdf()
J = [f"joint{i}" for i in range(1, 8)]
POS, TGT = [-0.15, 0.65, 0.75], [-0.40, 0.05, 0.05]


def look_at(pos, tgt):
    f = CAM.vsub(tgt, pos); n = CAM.vnorm(f); f = [v / n for v in f]
    up = [0, 0, 1]
    x = [f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2], f[0] * up[1] - f[1] * up[0]]
    n = CAM.vnorm(x); x = [v / n for v in x]
    yu = [x[1] * f[2] - x[2] * f[1], x[2] * f[0] - x[0] * f[2], x[0] * f[1] - x[1] * f[0]]
    return CAM.T_from([[x[i], -yu[i], f[i]] for i in range(3)], pos)


X_TRUE = look_at(POS, TGT)
Y_TRUE = [0.015, -0.025, 0.06]
K = {"width": 1280, "height": 720, "fx": 908.7, "fy": 908.7, "ppx": 640.1, "ppy": 366.6}
ZT = 0.0


class FakeD435:
    def __init__(self):
        self.info, self.K, self.dist, self.error, self.fps = {}, None, [0.0] * 5, None, 30.0
        self.on = False
        self.rnd = random.Random(1)

    def running(self):
        return self.on

    def start(self):
        self.on = True
        self.K = dict(K)
        self.info = {"name": "Intel RealSense D435I (fake)", "serial": "261222075224", "firmware": "5.17.3.10",
                     "usb": "3.2", "sdk": "2.58.4", "preset": "High Accuracy", "depth_scale_mm": 1.0, "K": dict(K),
                     "dist": [0.0] * 5, "model": "fake"}

    def stop(self, timeout=0):
        self.on = False

    @property
    def frame_id(self):
        return int(time.time() * 30)

    def aruco_status(self):
        return True, "fake"

    def display_ppm(self, view="color"):
        return b"P6 640 360 255\n" + (bytes([70, 90, 110]) if view == "color" else bytes([30, 30, 160])) * (640 * 360)

    def _ray_table(self, u, v):
        d = [(u - K["ppx"]) / K["fx"], (v - K["ppy"]) / K["fy"], 1.0]
        db = CAM.mat_vec(CAM.T_R(X_TRUE), d)
        o = CAM.T_t(X_TRUE)
        s = (ZT - o[2]) / db[2]
        return s if s > 0 else None          # z-depth along the optical axis = s (d has z = 1)

    def depth_at(self, u, v):
        s = self._ray_table(u, v)
        return None if s is None else s + self.rnd.gauss(0, 0.0005)

    def roi_points(self, u0, v0, u1, v1, stride=4):
        pts = []
        for v in range(int(min(v0, v1)), int(max(v0, v1)), stride * 2):
            for u in range(int(min(u0, u1)), int(max(u0, u1)), stride * 2):
                z = self.depth_at(u, v)
                if z:
                    pts.append(CAM.deproject(u, v, z, K))
        return pts, len(pts)

    def detect_marker(self, marker, q_deg=None):
        if not q_deg:
            return None
        T = CS.fk(U, dict(zip(J, [math.radians(v) for v in q_deg])), "base_link", "link7")
        A = [list(map(float, r)) for r in T]
        p = CAM.T_apply(CAM.T_inv(X_TRUE), CAM.T_apply(A, Y_TRUE))
        p = [v + self.rnd.gauss(0, 0.0003) for v in p]
        uv = CAM.project(p, K)
        c = [[uv[0] - 12, uv[1] - 12], [uv[0] + 12, uv[1] - 12], [uv[0] + 12, uv[1] + 12], [uv[0] - 12, uv[1] + 12]]
        return {"id": marker["id"], "corners_px": c, "center_px": uv, "p_cam": p, "p_pnp": p, "source": "depth", "side_px": 24}

    def frames(self):
        return None, None, self.frame_id


cap = {}
_c = CAM.CameraTabUI.__init__


def cinit(self, *a, **k):
    _c(self, *a, **k)
    cap["cam"] = self


CAM.CameraTabUI.__init__ = cinit
res = {}
_orig = tk.Tk.mainloop


def walk(w_):
    yield w_
    for c in w_.winfo_children():
        yield from walk(c)


def logs(root):
    return "\n".join(w_.get("1.0", "end") for w_ in walk(root) if w_.winfo_class() == "Text")


def driver(root):
    cam = cap["cam"]
    pipe = cam.pipe

    def wait(cond, t=30):
        t_end = time.time() + t
        while time.time() < t_end and not cond():
            root.update()
            time.sleep(0.03)
        return cond()

    def nb_select(text):
        for w_ in walk(root):
            if w_.winfo_class() == "TNotebook":
                for t in w_.tabs():
                    if w_.tab(t, "text") == text:
                        w_.select(t)

    def go():
        res["tabs"] = [w_.tab(t, "text") for w_ in walk(root) if w_.winfo_class() == "TNotebook" for t in w_.tabs()]
        pipe.monitor.start([sys.executable, "-u", os.path.join(HERE, "tests", "v315_fake_monitor.py")],
                           env=dict(os.environ, FAKE_Q0="0,0.5236,0,1.3963,0,0.6981,0", FAKE_RAD_S="30"), stdin_pipe=True)
        for k in RR.STEP_KEYS:
            pipe.state[k] = (RR.GREEN, "fake")
        wait(lambda: cam.jpanel.enabled)
        nb_select("Camera调试")
        cam.source_factory = FakeD435
        cam.on_connect()
        wait(lambda: "✓" in cam.v_cam.get() and cam.det is not None, 20)
        res["vcam"] = cam.v_cam.get()
        res["live0"] = cam.v_live.get()
        cam.v_nauto.set("12")
        cam.on_auto()
        wait(lambda: "自动采集结束" in logs(root), 240)
        wait(lambda: cam.result is not None, 20)
        r = cam.result
        res["n"] = len(cam.samples)
        res["dpos"] = CAM.vnorm(CAM.vsub(CAM.T_t(r["T_base_cam"]), POS)) if r else None
        res["good"] = r and r["good"]
        res["restext"] = cam.v_res.get()
        cam.on_save_calib()
        wait(lambda: cam.calib is not None and cam.result is None, 10)
        res["saved"] = json.load(open(CAM.CALIB_PATH))
        # click on a table point
        pb_true = [-0.42, 0.08, ZT]
        uv = CAM.project(CAM.T_apply(CAM.T_inv(X_TRUE), pb_true), K)
        pb = cam.measure(uv[0], uv[1])
        res["click_err"] = CAM.vnorm(CAM.vsub(pb, pb_true)) * 1000
        res["meas"] = cam.v_meas.get()
        cam.roi = (500, 400, 800, 560)
        cam.on_plane()
        res["plane"] = cam.v_plane.get()
        root.update()
        wait(lambda: "当前标定误差" in cam.v_live.get(), 10)
        res["live"] = cam.v_live.get()
        cam._force_redraw()
        root.update()
        res["items"] = len(cam.cv.find_all())
        cam.on_quick_check()
        wait(lambda: "快速复核" in logs(root) and ("误差" in logs(root).split("快速复核")[-1]), 120)
        if hasattr(cam, "v_method"):
            cam.v_method.set("几何法 V3.08")       # V3.47: the V3.10 pipeline stays selectable
        cam.on_eggs()
        wait(lambda: "[V3.25 locate]" in logs(root), 20)
        res["logs"] = logs(root)
        cam.on_disconnect()
        pipe.monitor.send({"quit": True})
        root.after(300, root.destroy)
    root.after(1200, go)


def patched(self, n=0):
    driver(self)
    _orig(self, n)


tk.Tk.mainloop = patched
C.run_console("real")
L = res["logs"]
assert "实机NERO调试" in res["tabs"] and "Camera调试" in res["tabs"] and "实机 NERO" not in res["tabs"], res["tabs"]
assert "D435I" in res["vcam"] and "USB 3.2" in res["vcam"] and "ArUco id 0" in res["live0"], (res["vcam"], res["live0"])
assert res["n"] >= 10 and res["dpos"] is not None and res["dpos"] < 0.003 and res["good"], (res["n"], res["dpos"], res["restext"])
assert "合格 ✓" in res["restext"] and res["saved"]["camera_serial"] == "261222075224" and len(res["saved"]["check_poses_deg"]) == 2
assert res["click_err"] < 3.0 and "base_link" in res["meas"], (res["click_err"], res["meas"])
assert "桌面 z =" in res["plane"] and "✓ 深度质量好" in res["plane"], res["plane"]
zline = [l for l in res["plane"].split("\n") if "桌面 z =" in l][0]
assert abs(float(zline.split("桌面 z = ")[1].split(" mm")[0])) < 3.0, zline
assert "当前标定误差" in res["live"] and "[FAIL] [V3.25 camera] refresh" not in L and res["items"] > 5, res["live"]
assert "[OK] [V3.25 calib] 快速复核" in L and "回到起点" in L
assert "[V3.25 locate]" in L
print(f"[GUI25] tabs 实机NERO调试 + Camera调试; fake D435i: auto calibration {res['n']} samples -> camera within "
      f"{res['dpos'] * 1000:.1f} mm (合格), saved; click -> base point within {res['click_err']:.1f} mm; table plane vs "
      "zone; live marker error; quick check; overlays drawn")
