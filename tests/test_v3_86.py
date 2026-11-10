#!/usr/bin/env python3
"""V3.86: Innomaker 4MP USB HDR Camera (U20-4MP OV4688) hardware integration, standby legacy 1080P preservation,
and dedicated Wrist Camera Lab GUI (「腕部相机高级测试」) under the 相机信息 tab.

Checks:
  W1 Hardware discovery & resolution support: list_wrist_cameras prioritizes U20-4MP OV4688 HDR over generic UVC and
     excludes RealSense; parse_res supports 2688×1520 (4MP), 1920×1080, and 1280×720; legacy W49.UsbWristSource remains
     intact as standby.
  W2 UVC & HDR profile management: apply_uvc_profile_to_cap, save_named_profile, activate_named_profile, and
     Innomaker4MPWristSource.apply_profile_dict.
  W3 Sharpness & Cup Rim / Gripper Angle preview: evaluate_focus_metrics (Laplacian + 4-quadrant uniformity) and
     evaluate_cup_rim_preview (red cup centre offset dx/dy mm & gripper closing axis perpendicularity error deg).
  W4 CameraTabUI integration & WristCameraLab GUI: 相机信息 tab shows Innomaker 4MP USB HDR Camera (U20-4MP OV4688),
     model selector (default 4MP HDR, standby 1080P 130°), resolution selector, and 「腕部相机高级测试」 button that opens
     the 3-page WristCameraLab window (① 腕部相机参数与HDR, ② 对焦与清晰度测试, ③ 标定与TCP准星复核).
"""
import os
import sys
import tempfile
import unittest.mock as mock

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import NERO_V325_Camera as CAM                  # noqa: E402
import NERO_V349_Real_Wrist_Camera as W49       # noqa: E402
import NERO_V386_Real_Wrist_Camera as W86       # noqa: E402

assert CAM.VERSION == "V3.25" and W49.VERSION in ("V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4") and W86.VERSION in ("V3.86", "V3.87", "V3.88", "V3.89", "V3.90", "V3.91", "V3.92", "V3.93", "V4.1", "V4.2", "V4.3", "V4.4", "V4.5", "V4.5.1", "V4.6", "V4.6.1", "V4.6.2", "V4.6.3", "V4.6.4", "V4.6.5", "V4.6.6", "V4.6.8", "V4.6.9", "V4.6.10", "V4.6.11", "V4.6.12", "V4.6.13", "V4.7", "V4.7.1", "V4.7.2", "V4.7.3", "V4.7.4")

# ------------------------------------------------------------------ W1: Discovery, 4MP resolution & Legacy Standby
with tempfile.TemporaryDirectory() as tmp:
    sysfs = os.path.join(tmp, "v4l")
    os.makedirs(sysfs)
    for node, name, idx in (
        ("video0", "Intel(R) RealSense(TM) Depth Camera 435i RGB", 0),
        ("video2", "Generic USB2.0 Camera", 0),
        ("video4", "Innomaker 1080P USB2.0 UVC Camera", 0),
        ("video6", "Innomaker 4MP USB HDR Camera (U20-4MP OV4688)", 0),
        ("video7", "Innomaker 4MP USB HDR Camera (U20-4MP OV4688)", 1),
    ):
        p = os.path.join(sysfs, node)
        os.makedirs(p)
        open(os.path.join(p, "name"), "w").write(name + "\n")
        open(os.path.join(p, "index"), "w").write(f"{idx}\n")

    devs = W86.list_wrist_cameras(sysfs)
    assert [d["device"] for d in devs] == ["/dev/video6", "/dev/video4", "/dev/video2"], devs
    assert devs[0]["is_4mp_hdr"] is True and devs[0]["model_tag"] == "U20-4MP OV4688"
    assert W86.parse_res("2688×1520") == (2688, 1520)
    assert W86.parse_res("1920×1080") == (1920, 1080)
    assert W86.parse_res("1280×720") == (1280, 720)
    assert "2688×1520" in W86.WRIST_RES_LIST_4MP 
    assert W86.LEGACY_WRIST_RES_LIST == ("1920×1080", "1280×720")
print("[W1] Innomaker U20-4MP OV4688 discovery priority, 2688x1520 4MP resolution, and legacy 1080P standby verified")

# ------------------------------------------------------------------ W2: UVC + HDR profile management
with tempfile.TemporaryDirectory() as tmp:
    os.environ["NERO_WRIST_PROFILE_PATH"] = os.path.join(tmp, "wrist_4mp_profile.json")
    os.environ["NERO_WRIST_PROFILES_DIR"] = os.path.join(tmp, "wrist_profiles")
    os.environ["NERO_WRIST_LAB_SETTINGS"] = os.path.join(tmp, "wrist_camera_lab.json")

    prof = W86.load_wrist_profile()
    assert prof["camera_model"] == W86.WRIST_CAMERA_MODELS[0]
    prof["color"]["exposure"] = 83.0
    prof["color"]["gain"] = 24.0
    prof["color"]["backlight_hdr"] = 1
    W86.save_named_profile("车库-双灯管-腕部", prof)
    assert "车库-双灯管-腕部" in W86.list_named_profiles()
    W86.activate_named_profile("车库-双灯管-腕部")
    loaded = W86.load_wrist_profile()
    assert loaded["locked"] is True and loaded["name"] == "车库-双灯管-腕部"
    assert loaded["color"]["exposure"] == 83.0 and loaded["color"]["backlight_hdr"] == 1

    class DummyCap:
        def __init__(self):
            self.props = {}

        def set(self, k, v):
            self.props[k] = v
            return True

        def get(self, k):
            return self.props.get(k, 0.0)

    cap = DummyCap()
    wsrc = W86.Innomaker4MPWristSource(device="/dev/video6", res="2688×1520")
    wsrc._cap = cap
    wsrc.info = {"name": "Innomaker 4MP", "device": "/dev/video6", "controls": {}}
    applied = wsrc.apply_profile_dict(loaded["color"])
    assert applied.get("exposure") == 83.0 and applied.get("gain") == 24.0 and applied.get("hdr") == "on"
print("[W2] UVC & OV4688 HDR profile save/load/activate and live control application verified")

# ------------------------------------------------------------------ W3: Focus metrics & Cup Rim / Gripper Angle preview
sharp_img = np.zeros((720, 1280, 3), dtype=np.uint8) + 210
for r in range(60, 660, 40):
    for c in range(100, 1180, 40):
        if ((r // 40) + (c // 40)) % 2 == 0:
            sharp_img[r:r + 40, c:c + 40] = 30
f_ok = W86.evaluate_focus_metrics(sharp_img)
f_blur = W86.evaluate_focus_metrics(cv2.GaussianBlur(sharp_img, (31, 31), 12.0))
assert f_ok["ok"] and f_ok["laplacian_var"] > 100.0 and f_ok["level"] == "ok", f_ok
assert f_blur["laplacian_var"] < f_ok["laplacian_var"] * 0.1

# Draw a red cup rim centered at (660, 380) with radius 95 px; TCP at (640, 360)
cup_img = np.zeros((720, 1280, 3), dtype=np.uint8) + 200
cv2.circle(cup_img, (660, 380), 95, (220, 25, 25), -1)
K_test = {"width": 1280, "height": 720, "fx": 900.0, "fy": 900.0, "ppx": 640.0, "ppy": 360.0}
T_fc = [[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 0.0], [0, 0, 0, 1]]
tool_T = [[1, 0, 0, 0.0], [0, 1, 0, 0.0], [0, 0, 1, 0.10], [0, 0, 0, 1]]
cup_prev = W86.evaluate_cup_rim_preview(cup_img, K=K_test, T_flange_cam=T_fc, tool_T=tool_T, z_work_m=0.10)
assert cup_prev["ok"] and abs(cup_prev["center_px"][0] - 660.0) < 3.0 and abs(cup_prev["center_px"][1] - 380.0) < 3.0
assert cup_prev["offset_norm_mm"] > 1.5
print(f"[W3] Focus evaluation (Lap={f_ok['laplacian_var']:.1f}) and cup rim preview (dx={cup_prev['dx_mm']} mm, dy={cup_prev['dy_mm']} mm, Δθ={cup_prev['angle_err_deg']}°) verified")

# ------------------------------------------------------------------ W4: CameraTabUI & WristCameraLab GUI verification
with tempfile.TemporaryDirectory() as tmp:
    os.environ["NERO_WRIST_PROFILE_PATH"] = os.path.join(tmp, "wrist_4mp_profile.json")
    os.environ["NERO_WRIST_PROFILES_DIR"] = os.path.join(tmp, "wrist_profiles")
    os.environ["NERO_WRIST_LAB_SETTINGS"] = os.path.join(tmp, "wrist_camera_lab.json")

    class FakeVar:
        def __init__(self, value=""):
            self._v = value

        def get(self):
            return self._v

        def set(self, v):
            self._v = v

    mock_tk = mock.MagicMock()
    mock_ttk = mock.MagicMock()
    mock_tk.StringVar.side_effect = lambda value="": FakeVar(str(value))
    mock_tk.BooleanVar.side_effect = lambda value=False: FakeVar(bool(value))
    mock_tk.DoubleVar.side_effect = lambda value=0.0: FakeVar(float(value))
    mock_tk.IntVar.side_effect = lambda value=0: FakeVar(int(value))

    logs = []
    cam_ui = CAM.CameraTabUI(
        mock_tk, mock_ttk, mock.MagicMock(), mock.MagicMock(), mock.MagicMock(),
        mock.MagicMock(), lambda: {}, logs.append, lambda *a: True
    )
    assert hasattr(cam_ui, "v_wcam_model") and cam_ui.v_wcam_model.get() == W86.WRIST_CAMERA_MODELS[0]
    assert hasattr(cam_ui, "v_wcam_res") and cam_ui.v_wcam_res.get() == "1920×1080"
    assert hasattr(cam_ui, "b_wrist_lab")
    assert "Innomaker 4MP HDR" in cam_ui._wrist_info_text()

    # Connect mock 4MP wrist source and open WristCameraLab
    class FakeWrist4MP(W86.Innomaker4MPWristSource):
        def __init__(self):
            super().__init__(device="/dev/video6", res="2688×1520")
            self.rgb = cup_img
            self.fps = 30.0
            self.info = {
                "name": "Innomaker 4MP USB HDR Camera (U20-4MP OV4688)",
                "camera_model": W86.WRIST_CAMERA_MODELS[0],
                "device": "/dev/video6",
                "format": "MJPG",
                "res": "2688×1520",
                "width": 2688,
                "height": 1520,
                "controls": {"hdr": "on"},
            }

        def start(self):
            pass

        def running(self):
            return True

    cam_ui.wrist_source_factory = FakeWrist4MP
    cam_ui.wrist_intr = {"K": dict(K_test), "dist": [0.0] * 5, "model_type": W49.DIST_MODELS[1], "rms_px": 0.22, "good": True}
    cam_ui.wrist_calib = {"T_flange_cam": T_fc, "marker_in_base": [-0.45, 0.0, 0.0], "loo_mean_mm": 0.45, "loo_max_mm": 0.85, "good": True}
    cam_ui.on_wrist_connect()
    assert "U20-4MP OV4688" in cam_ui._wrist_info_text() and "2688×1520" in cam_ui._wrist_info_text()

    lab = cam_ui.on_wrist_camera_lab()
    assert lab is not None
    lab.on_autotune()
    assert lab.last_tune is not None
    lab.v_pname.set("车库-测试4MP")
    lab.on_save_profile()
    assert "车库-测试4MP" in W86.list_named_profiles()
    focus_res = lab.on_measure_focus()
    assert focus_res["ok"]
    ver_res = lab.on_verify_once()
    assert ver_res["ok"] and "红杯杯口检测" in ver_res["text"] and "TCP 投影准星" in ver_res["text"]
    lab.close()
print("[W4] CameraTabUI 相机信息 4MP UI controls & WristCameraLab 3-tab workflow verified")
print("ALL V3.86 TESTS PASSED")
