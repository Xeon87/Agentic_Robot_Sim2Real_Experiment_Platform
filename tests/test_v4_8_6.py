#!/usr/bin/env python3
"""NERO V4.8.6 offline & GUI unit tests:
U1 Startup defaults to NERO original AgxGripper (agx_gripper) and automatically initiates gripper communication connection.
U2 实机NERO调试 "一键启动" (run_all_then_gripper / v486_start_cameras) also connects main camera (D435i) and wrist camera on the Tk main thread.
U3 Camera调试 -> 视觉定位 model selector (patched on CameraTabUI via NERO_V472_Wrist_Calib):
   - Defaults all available fine-tuned models to selected at startup (even after _locate_autoload loads locate_settings.json).
   - Canonicalizes absolute vs relative fine-tuned model paths so checkboxes and v_ymodel never desync.
   - Displays "原模型" (instead of "原模型 + 2 精调模型" or "yoloe-26s-seg.pt") when no fine-tuned models are checked.
U4 Crash fixes from crash.log & native_crash.log:
   - _wheel_skip_widget handles str widget paths (e.g. ttk.Combobox popdown listbox) without AttributeError.
   - v486_disable_color_emoji_fonts writes fontconfig_no_emoji.conf and rejects NotoColorEmoji.ttf in Fontconfig.
   - _v312_pick_cjk_font configures TkFixedFont so log/status labels do not fall back from Ubuntu Mono to NotoColorEmoji.ttf.
   - No high-plane colour-emoji characters (> 0xFFFF) remain in NERO_V359_Strategy.py or NERO_V379_Camera_Lab.py.
U5 Version & frozen-file integrity (NERO_V308_Camera_Perception.py, NERO_V325_Camera.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import NERO_V312_Console as C                    # noqa: E402
import NERO_V313_Real_Robot as RR                # noqa: E402
import NERO_V325_Camera as CAM                   # noqa: E402
import NERO_V347_Yolo_Perception as YP           # noqa: E402
import NERO_V472_Wrist_Calib as WC               # noqa: E402
import NERO_V48_Gripper as GG                    # noqa: E402


class TestV486(unittest.TestCase):
    def test_u1_startup_default_gripper_and_autoconnect(self):
        tmp = tempfile.mkdtemp(prefix="nero_v486_grip_")
        old_path = GG.CFG_PATH
        old_env = os.environ.pop("NERO_GRIPPER", None)
        old_keep = os.environ.pop("NERO_GRIPPER_KEEP_SAVED", None)
        old_noconn = os.environ.pop("NERO_NO_STARTUP_GRIPPER_CONNECT", None)
        try:
            GG.CFG_PATH = os.path.join(tmp, "gripper.json")
            # Simulate a saved config that had so101
            GG.save_cfg({"type": "so101"})
            logs = []
            chosen = C.v486_gripper_startup_default(log=logs.append)
            self.assertEqual(chosen, "agx_gripper")
            self.assertEqual(GG.active_type(), "agx_gripper")
            self.assertTrue(GG.is_agx())
            self.assertTrue(any("开机默认夹爪设为 NERO 原装夹爪" in m for m in logs))

            # Auto-connect calls grip_panel.on_connect() when not yet connected
            calls = []
            fake_dev = types.SimpleNamespace(connected=False)
            fake_panel = types.SimpleNamespace(
                agx=lambda: fake_dev,
                on_connect=lambda: calls.append("agx_connect"),
            )
            st = C.v486_gripper_startup_connect(fake_panel, log=logs.append)
            self.assertEqual(st, "agx_connecting")
            self.assertEqual(calls, ["agx_connect"])

            # Already connected -> does not reconnect
            fake_dev.connected = True
            st2 = C.v486_gripper_startup_connect(fake_panel, log=logs.append)
            self.assertEqual(st2, "already_connected")
            self.assertEqual(len(calls), 1)
        finally:
            GG.CFG_PATH = old_path
            if old_env is not None:
                os.environ["NERO_GRIPPER"] = old_env
            if old_keep is not None:
                os.environ["NERO_GRIPPER_KEEP_SAVED"] = old_keep
            if old_noconn is not None:
                os.environ["NERO_NO_STARTUP_GRIPPER_CONNECT"] = old_noconn
            shutil.rmtree(tmp, ignore_errors=True)

    def test_u2_one_click_start_starts_both_cameras(self):
        events = []

        class FakeCamTab:
            def __init__(self):
                self.src = None
                self.wrist_src = None

            def _ui(self, fn):
                events.append(("ui", fn.__name__))
                fn()

            def on_connect(self):
                events.append("main_connect")
                self.src = types.SimpleNamespace(running=True)

            def on_wrist_connect(self):
                events.append("wrist_connect")
                self.wrist_src = types.SimpleNamespace(running=True)

        cam = FakeCamTab()
        logs = []
        r1 = C.v486_start_cameras(cam, log=logs.append)
        self.assertEqual(r1, {"main": True, "wrist": True})
        self.assertIn("main_connect", events)
        self.assertIn("wrist_connect", events)
        self.assertIn(("ui", "on_connect"), events)
        self.assertIn(("ui", "on_wrist_connect"), events)

        # Second call when both cameras are already running does not restart them
        events.clear()
        r2 = C.v486_start_cameras(cam, log=logs.append)
        self.assertEqual(r2, {"main": False, "wrist": False})
        self.assertEqual(events, [])

        # Verify run_all_then_gripper in NERO_V312_Console.py invokes v486_start_cameras
        with open(os.path.join(HERE, "NERO_V312_Console.py"), encoding="utf-8") as _f: con_src = _f.read()
        self.assertIn("v486_start_cameras(cam_tab, log=pipe.log)", con_src)

    def test_u3_ymodel_menu_default_all_and_original_label(self):
        WC.patch_class(CAM.CameraTabUI)
        tmp = tempfile.mkdtemp(prefix="nero_v486_models_")
        old_yolo_dir = os.environ.get("NERO_YOLO_DIR")
        old_ft_dir = os.environ.get("NERO_V381_FINETUNE_DIR")
        old_def_dir = YP.DEFAULT_MODEL_DIR
        try:
            os.environ["NERO_YOLO_DIR"] = tmp
            os.environ["NERO_V381_FINETUNE_DIR"] = os.path.join(tmp, "finetuned")
            YP.DEFAULT_MODEL_DIR = tmp
            with open(os.path.join(tmp, "yoloe-26s-seg.pt"), "wb") as _f: _f.write(b"base")
            for sub, cls_name in (("egg_carton_2026", "egg carton"), ("egg_garage_2026", "egg")):
                d = os.path.join(tmp, "finetuned", sub)
                os.makedirs(d, exist_ok=True)
                pt = f"{sub}.pt"
                with open(os.path.join(d, pt), "wb") as _f: _f.write(b"ft")
                with open(os.path.join(d, "model_card.json"), "w", encoding="utf-8") as f:
                    json.dump({"name": sub, "file": pt, "classes": [cls_name], "base_model": "yoloe-26s-seg.pt"}, f)

            class FakeVar:
                def __init__(self, v=""):
                    self.v = v

                def get(self):
                    return self.v

                def set(self, v):
                    self.v = v if isinstance(v, bool) else str(v)

            class FakeMenu:
                def __init__(self):
                    self.items = []

                def delete(self, *_a):
                    self.items.clear()

                def add_checkbutton(self, label, **kw):
                    self.items.append((label, kw))

                def add_separator(self):
                    pass

            class FakeMB:
                def __init__(self):
                    self.text = ""

                def configure(self, **kw):
                    if "text" in kw:
                        self.text = kw["text"]

            class DummyCam:
                def __init__(self):
                    self.tk = types.SimpleNamespace(BooleanVar=lambda value=False: FakeVar(value))
                    self.YP = YP
                    self.v_ydir = FakeVar(tmp)
                    self.v_ymodel = FakeVar(YP.DEFAULT_MODEL)
                    self.mb_ymodel = FakeMB()
                    self.menu_ymodel = FakeMenu()
                    self._ymodel_vars = {}

                _update_ymodel_menu = CAM.CameraTabUI._update_ymodel_menu
                _on_ymodel_toggle = CAM.CameraTabUI._on_ymodel_toggle
                _refresh_ymodel_summary = CAM.CameraTabUI._refresh_ymodel_summary
                _locate_autoload = CAM.CameraTabUI._locate_autoload

            tab = DummyCam()
            # 1) Simulate locate_settings.json having saved ABSOLUTE paths from an earlier session
            abs_carton = f"/home/qiuren/Downloads/objectmodels/finetuned/egg_carton_2026/egg_carton_2026.pt"
            abs_egg = f"/home/qiuren/Downloads/objectmodels/finetuned/egg_garage_2026/egg_garage_2026.pt"
            tab.v_ymodel.set(f"yoloe-26s-seg.pt;{abs_carton};{abs_egg}")
            tab._ymodel_loaded_from_settings = True
            tab._update_ymodel_menu()

            # Both checkboxes must be True and paths canonicalized to relative finetuned/...
            self.assertEqual(len(tab._ymodel_vars), 2)
            for k, var in tab._ymodel_vars.items():
                self.assertTrue(var.get(), f"{k} should be checked")
            self.assertEqual(tab.mb_ymodel.text, "原模型 + 2 精调模型")

            # 2) Uncheck the first fine-tuned model -> 1 remaining
            ft_keys = sorted(tab._ymodel_vars.keys())
            tab._on_ymodel_toggle(ft_keys[0], False)
            self.assertIn("原模型 + egg_garage_2026", tab.mb_ymodel.text)

            # 3) Uncheck the second fine-tuned model -> must display "原模型" (NOT "原模型 + 2 精调模型" and NOT "yoloe-26s-seg.pt")
            tab._on_ymodel_toggle(ft_keys[1], False)
            self.assertEqual(tab.mb_ymodel.text, "原模型")
            self.assertEqual(tab.v_ymodel.get(), "yoloe-26s-seg.pt")

            # 4) Startup _locate_autoload selects all available fine-tuned models by default even after loading base-only settings
            WC._ORIG["_locate_autoload"] = lambda self_: self_.v_ymodel.set("yoloe-26s-seg.pt")
            tab._locate_autoload()
            self.assertEqual(tab.mb_ymodel.text, "原模型 + 2 精调模型")
            for k, var in tab._ymodel_vars.items():
                self.assertTrue(var.get(), f"{k} should be re-checked by _locate_autoload at startup")
        finally:
            YP.DEFAULT_MODEL_DIR = old_def_dir
            if old_yolo_dir is None:
                os.environ.pop("NERO_YOLO_DIR", None)
            else:
                os.environ["NERO_YOLO_DIR"] = old_yolo_dir
            if old_ft_dir is None:
                os.environ.pop("NERO_V381_FINETUNE_DIR", None)
            else:
                os.environ["NERO_V381_FINETUNE_DIR"] = old_ft_dir
            shutil.rmtree(tmp, ignore_errors=True)

    def test_u4_crash_fixes_wheel_and_fontconfig(self):
        # 1) _wheel_skip_widget handles str popdown paths and unmapped widget strings without raising AttributeError
        self.assertTrue(C._wheel_skip_widget(None, ".!notebook.!frame3.!combobox.popdown.f.l"))
        self.assertTrue(C._wheel_skip_widget(None, ".!nonexistent_widget"))
        self.assertTrue(C._wheel_skip_widget(None, None))
        fake_frame = types.SimpleNamespace(winfo_class=lambda: "TFrame")
        fake_combo = types.SimpleNamespace(winfo_class=lambda: "TCombobox")
        self.assertFalse(C._wheel_skip_widget(None, fake_frame))
        self.assertTrue(C._wheel_skip_widget(None, fake_combo))

        # 2) v486_disable_color_emoji_fonts writes fontconfig_no_emoji.conf and rejects NotoColorEmoji.ttf
        tmp = tempfile.mkdtemp(prefix="nero_v486_fc_")
        old_fc = os.environ.pop("FONTCONFIG_FILE", None)
        try:
            st = C.v486_disable_color_emoji_fonts(tmp)
            self.assertTrue(st.startswith("ok"), st)
            conf = os.path.join(tmp, "fontconfig_no_emoji.conf")
            self.assertTrue(os.path.isfile(conf))
            r = subprocess.run(["fc-match", "-s", "Noto Sans CJK SC"], capture_output=True, text=True, check=False)
            self.assertNotIn("NotoColorEmoji", r.stdout)
        finally:
            if old_fc is None:
                os.environ.pop("FONTCONFIG_FILE", None)
            else:
                os.environ["FONTCONFIG_FILE"] = old_fc
            shutil.rmtree(tmp, ignore_errors=True)

        # 3) No high-plane colour-emoji characters (> 0xFFFF) in Strategy or Camera Lab
        for fn in ("NERO_V359_Strategy.py", "NERO_V379_Camera_Lab.py", "NERO_V312_Console.py", "NERO_V4_Console_Tab.py"):
            with open(os.path.join(HERE, fn), encoding="utf-8") as _f: src = _f.read()
            hi = [ch for ch in src if ord(ch) > 0xFFFF]
            self.assertEqual(hi, [], f"{fn} still contains high-plane emoji: {hi}")

    def test_u5_versions_and_frozen_hashes(self):
        with open(os.path.join(HERE, "VERSION"), encoding="utf-8") as f:
            v = f.read().strip()
        self.assertIn(v, ("V4.8.6", "V4.8.7", "V4.8.8", "V4.8.9", "V4.8.8", "V4.8.9", "V4.8.9", "V4.8.10", "V4.8.11", "V4.9", "V4.9.1", "V4.9.2", "V4.9.3", "V4.10", "V4.10.1", "V4.10.2", "V4.10.3", "V4.11"))
        self.assertEqual((v, C.VERSION, RR.VERSION), (v, v, v))
        self.assertEqual(C.MARKER_V486, "V4.8.6 STARTUP GRIPPER + CAMERAS + YMODEL + XFT CRASH FIX")
        self.assertEqual(WC.MARKER_V486, "V4.8.6 CAMERA YMODEL MENU + STARTUP DEFAULTS")
        with open(os.path.join(HERE, "NERO_V308_Camera_Perception.py"), "rb") as _f: h308 = hashlib.sha256(_f.read()).hexdigest()
        with open(os.path.join(HERE, "NERO_V325_Camera.py"), "rb") as _f: h325 = hashlib.sha256(_f.read()).hexdigest()
        self.assertTrue(h308.startswith("020cc6680f181872"), h308)
        self.assertTrue(h325.startswith("82f2416d"), h325)


if __name__ == "__main__":
    unittest.main(verbosity=1)
