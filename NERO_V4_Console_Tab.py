#!/usr/bin/env python3
"""NERO V4.2 — Complete Console Tab for Dual-Path VLA / LeRobot (Phases 0–6).

NERO_V4_Console_Tab.py

Implements Section 11 of the NERO V4.0 Architecture Specification:
1. System Configuration (11.1):
   - LeRobot environment connection & IPC status
   - SmolVLA model selection, load status & fine-tuning trigger
   - CPU / GPU & VRAM status
   - MuJoCo / Real Robot / Mock backend selection
   - Traditional / SmolVLA policy & Control Arbiter mode selection
2. Live Status (11.2):
   - Fixed main camera & wrist camera synchronization status
   - 7-DOF joint angles + SO-101 gripper normalized opening & physical gap_mm
   - Current Agent subtask, active Policy, current Action Chunk, inference time & execution progress
   - Safety Supervisor status & trip counter
3. Demonstration Recording, Testing & Benchmark (11.3):
   - Start / Stop demonstration recording + Independent Gripper Teach buttons (Open / Close)
   - Episode management (list, quality check, replay, delete)
   - Dataset & Checkpoint version selection + SmolVLA Fine-tuning
   - Simulation & Real-Robot VLA Closed-Loop / Top-Level Agent Dual-Path execution
   - Phase 5 Sim-to-Real 6-Stage Safety Ladder & Phase 6 Three-Mode Benchmark (Modes A / B / C)
   * Strictly separates "① 加载模型/连接" (Load/Connect), "② 允许控制" (Allow Control),
     "③ 开始执行" (Start Execution), and "④ 停止/紧急急停" (Stop/E-Stop).
4. V4.4 EMBODIED_AI DRIVE PANEL (section 0):
   - Detects / mounts / safely ejects the removable drive EMBODIED_AI and shows file system, free space, venv,
     torch / CUDA / lerobot versions and the SmolVLA model status.
   - Installs LeRobot + SmolVLA into a venv ON THE DRIVE (pip cache, TMPDIR, HF cache all on the drive), downloads
     lerobot/smolvla_base (+ SmolVLM2 backbone) onto the drive, copies old internal-disk data to the drive.
   - Datasets, NERO checkpoints, exported LeRobotDatasets and `lerobot-train` runs are all written to the drive.
     Long jobs run detached with a log on the drive, so closing the console does not stop a training run.
"""
from __future__ import annotations

import json
import math
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np

VERSION = "V4.6"
MARKER = "V4.2 VLA LEROBOT CONSOLE TAB"
MARKER_V4611 = "V4.6.11 AUTO TEACH RECORD + SPACE BINDTAG + RECENT EPISODES + LIGHT REFRESH"
HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import NERO_V400_Phase0_Audit as AUD
import NERO_V44_Storage as ST
from NERO_V4_Agent_Orchestrator import AgentOrchestrator
from NERO_V4_Benchmark import DualPathBenchmark, SimToRealVerifier
from NERO_V4_Control_Arbiter import CONTROL_MODES, ControlArbiter
from NERO_V4_Dataset_Recorder import DemonstrationRecorder, NEROLeRobotDataset
from NERO_V4_Dataset_Recorder import replay_episode_real, REPLAY_OPEN_GAP_MM, REPLAY_CLAMP_PCT  # V4.6.9
from NERO_V4_LeRobot_Adapter import NEROIPCServer, NEROLeRobotAdapter
from NERO_V4_Observation_Bridge import ObservationBridge
from NERO_V4_Policy_Router import PolicyRouter
from NERO_V4_SmolVLA_Policy import WORKER_PATH, SmolVLAPolicy
from NERO_V4_Task_Evaluator import TaskEvaluator
from NERO_V4_VLA_Action_Executor import VLAActionExecutor


PREP_STEPS = (("drive", "硬盘"), ("dirs", "目录"), ("env", "环境"), ("model", "模型文件"), ("dataset", "数据集"),
              ("nero", "NERO"), ("cam", "主相机"), ("wrist", "腕部相机"), ("grip", "夹爪"), ("vla", "SmolVLA"))
MARKER_V47 = "V4.7 ONE-CLICK PREPARE + EPISODE BROWSER"
LAMP_COLORS = {"off": "#9e9e9e", "busy": "#ffb300", "ok": "#2e7d32", "warn": "#ef6c00", "bad": "#c62828"}


class StatusLamp:
    """V4.6 STATUS LAMPS: a coloured dot (Canvas, no font glyph) + a short status text."""

    def __init__(self, tk, ttk, parent, title: str, width_chars: int = 18, side: str = "left"):
        self.fr = ttk.Frame(parent)
        self.fr.pack(side=side, padx=(2, 6))
        self.cv = tk.Canvas(self.fr, width=14, height=14, highlightthickness=0, bd=0)
        self.cv.pack(side="left")
        self.dot = self.cv.create_oval(2, 2, 12, 12, fill=LAMP_COLORS["off"], outline="#616161")
        self.v = tk.StringVar(value=f"{title}: -")
        self.title = title
        ttk.Label(self.fr, textvariable=self.v, width=width_chars).pack(side="left", padx=(2, 0))
        self.state = "off"

    def set(self, state: str, text: Optional[str] = None) -> None:
        self.state = state if state in LAMP_COLORS else "off"
        try:
            self.cv.itemconfigure(self.dot, fill=LAMP_COLORS[self.state])
            if text is not None:
                self.v.set(f"{self.title}: {text}")
        except Exception:
            pass


class VLALeRobotTabUI:
    """「Smol VLA实机任务」 tab (V4.6.1; was 「VLA / LeRobot (V4)」) in NERO Console (Phases 0–6 integration)."""

    def __init__(
        self,
        tk,
        ttk,
        parent,
        pipe=None,
        jpanel=None,
        cam_tab=None,
        rct_tab=None,
        log: Optional[Callable[[str], None]] = None,
        ask: Optional[Callable[[str, str], bool]] = None,
        visible_fn: Callable[[], bool] = lambda: True,
        safe_tab=None,
    ):
        self.tk = tk
        self.safe_tab = safe_tab
        self.ttk = ttk
        self.parent = parent
        self.pipe = pipe
        self.jpanel = jpanel
        self.cam_tab = cam_tab
        self.rct_tab = rct_tab
        self.ext_log = log or (lambda s: None)
        self.ask = ask or (lambda t, q: True)
        self.visible_fn = visible_fn

        self.q: queue.Queue = queue.Queue()
        # V4.4: point HF / LeRobot / dataset / checkpoint paths at the EMBODIED_AI drive before anything else
        # V4.5.1: os.environ is changed only from _tick (Tk main thread, after the widgets exist)
        self.q.put(("apply_env", ST.layout()))
        self.job = ST.BackgroundJob(ST.layout(), log=self._log)
        self._job_seen_running = False
        self._venv_st: Dict[str, Any] = {}
        self._drive_found_last: Optional[bool] = None
        self._tick_n = 0
        self._run_busy = threading.Lock()          # V4.6: one ③ / benchmark / ladder run at a time
        self._run_kind = ""
        self._loading = False
        self.arbiter = ControlArbiter(log=self._log)
        self.obs_bridge = ObservationBridge(env_mode="mock", log=self._log)
        self.obs_bridge.start_mock_sources()
        self.executor = VLAActionExecutor(self.arbiter, self.obs_bridge, log=self._log)
        self.adapter = NEROLeRobotAdapter(
            obs_bridge=self.obs_bridge,
            executor=self.executor,
            arbiter=self.arbiter,
            log=self._log,
        )
        self.dataset = NEROLeRobotDataset(dataset_name="nero_v4_demo")
        self.recorder = DemonstrationRecorder(
            obs_bridge=self.obs_bridge,
            dataset=self.dataset,
            executor=self.executor,
            log=self._log,
        )
        self.vla_policy = SmolVLAPolicy(log=self._log)
        self.evaluator = TaskEvaluator(obs_bridge=self.obs_bridge, arbiter=self.arbiter, log=self._log)
        self.router = PolicyRouter(
            arbiter=self.arbiter,
            adapter=self.adapter,
            vla_policy=self.vla_policy,
            evaluator=self.evaluator,
            log=self._log,
        )
        self.orchestrator = AgentOrchestrator(router=self.router, log=self._log)
        self.sim2real = SimToRealVerifier(
            arbiter=self.arbiter, adapter=self.adapter, vla_policy=self.vla_policy, log=self._log
        )
        self.benchmark = DualPathBenchmark(
            orchestrator=self.orchestrator, router=self.router, vla_policy=self.vla_policy, log=self._log
        )
        self.ipc_server: Optional[NEROIPCServer] = None

        self._build(parent)
        # V4.6.11: the space bar must reach the gripper from every widget of this tab (see _on_space); the log box
        # is a Text widget but not for typing -> it gets the tag too (V4.6.10: a click into the log swallowed space)
        self._install_space_tag(parent)
        try:
            self.txt.bindtags(("NeroVLASpace",) + tuple(t for t in self.txt.bindtags() if t != "NeroVLASpace"))
        except Exception:
            pass
        self._schedule()
        threading.Thread(target=self._startup_drive_check, daemon=True).start()

    def _log(self, msg: str) -> None:
        self.q.put(("log", str(msg)))
        try:
            self.ext_log(str(msg))
        except Exception:
            pass

    def _build(self, parent) -> None:
        tk, ttk = self.tk, self.ttk
        import NERO_V314_Joint_Control as JC

        # Top status banner
        self.v_banner = tk.StringVar(
            value="V4.4 双路径 Agentic + SmolVLA 平台：控制权默认锁定在 Monitor Only（加载模型/连接不会自动授权运动）；LeRobot / 模型 / 训练文件都在移动硬盘 EMBODIED_AI"
        )
        JC.fixed_label(
            tk, parent, 2, width=1180, textvariable=self.v_banner,
            font=("TkDefaultFont", 9, "bold"), wraplength=1170, pack={"fill": "x", "pady": (0, 3)}
        )

        # V4.6.9 LAYOUT (user: "混乱、拥挤"): one fixed top bar (back end, policy, control authority, E-STOP always
        # visible) + 5 sub-pages in working order ① 准备 ② 示教采集 ③ 训练 ④ 执行 ⑤ 评估 / 高级; status, monitors
        # and the log below are shared by all pages
        top = ttk.Frame(parent)
        top.pack(fill="x", pady=(0, 2))
        es = ttk.Frame(top)                                     # packed FIRST on the right: never pushed off screen
        es.pack(side="right", padx=(6, 0))
        self.btn_estop = tk.Button(
            es, text="■ 紧急急停 (E-STOP)", bg="#c62828", fg="white",
            font=("TkDefaultFont", 10, "bold"), command=self.on_estop, height=1
        )
        self.btn_estop.pack(side="left")
        ttk.Button(es, text="解除急停", command=self.on_clear_estop).pack(side="left", padx=(4, 0))
        self.lamp_4 = StatusLamp(tk, ttk, es, "④急停", 10)
        tb = ttk.Frame(top)
        tb.pack(side="left", fill="x", expand=True)
        ttk.Label(tb, text="后端:", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        self.v_backend = tk.StringVar(value="mock")
        for val, lab in (("mock", "仿真替身"), ("mujoco", "MuJoCo"), ("real", "实机 NERO")):
            ttk.Radiobutton(tb, text=lab, value=val, variable=self.v_backend, command=self.on_backend_change).pack(side="left", padx=2)
        ttk.Label(tb, text="  策略:", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        self.v_policy = tk.StringVar(value="smolvla")
        for p_val, p_lab in (("smolvla", "SmolVLA"), ("traditional", "传统技能")):
            ttk.Radiobutton(tb, text=p_lab, value=p_val, variable=self.v_policy).pack(side="left", padx=2)
        self.v_arb_text = tk.StringVar(value="控制权: monitor · 只读")
        ttk.Label(tb, textvariable=self.v_arb_text, foreground="#37474f").pack(side="left", padx=(10, 0))

        self.sub_nb = ttk.Notebook(parent)
        self.sub_nb.pack(fill="x", pady=(0, 2))
        pg = {}
        for key, title in (("prep", "① 准备"), ("teach", "② 示教采集"), ("train", "③ 训练"), ("run", "④ 执行"),
                           ("adv", "⑤ 评估 / 高级")):
            f_ = ttk.Frame(self.sub_nb, padding=4)
            self.sub_nb.add(f_, text=title)
            pg[key] = f_
        self.pages = pg

        # ---------------------------------------------------------------- ① 准备
        # V4.7 ONE-CLICK PREPARE: one prominent button, the steps in their order as lamps, the result in one line;
        # the 「① 准备」 tab itself carries a lamp (grey = not checked, green = all ready, yellow = a step needs you)
        import NERO_V47_VLA_Tools as TL
        self._tab_imgs = TL.lamp_images(tk)
        self.sub_nb.tab(pg["prep"], image=self._tab_imgs["off"], compound="left")
        self.prep_state = "off"
        pf = ttk.LabelFrame(pg["prep"], text="一键准备：按顺序检查，能自动处理的自动处理（挂载硬盘、建目录、连相机和夹爪、加载模型）")
        pf.pack(fill="x", pady=(0, 4))
        pfi = ttk.Frame(pf)
        pfi.pack(fill="x", padx=6, pady=4)
        self.btn_prepare = tk.Button(pfi, text="一键准备", command=self.on_prepare_all, bg="#1565c0", fg="white",
                                     activebackground="#1e88e5", activeforeground="white",
                                     font=("TkDefaultFont", 13, "bold"), relief="raised", bd=1, padx=18, pady=8)
        self.btn_prepare.pack(side="left", padx=(0, 12), anchor="n")
        pr = ttk.Frame(pfi)
        pr.pack(side="left", fill="x", expand=True)
        grid = ttk.Frame(pr)
        grid.pack(fill="x")
        self.prep_lamps = {}
        for i, (k, t) in enumerate(PREP_STEPS):
            cell = ttk.Frame(grid)
            cell.grid(row=i // 5, column=i % 5, sticky="w", padx=(0, 4), pady=1)
            self.prep_lamps[k] = StatusLamp(tk, ttk, cell, f"{i + 1} {t}", 17)
        self.v_prep = tk.StringVar(value="还没检查。顺序：移动硬盘 → 目录 → LeRobot 环境 → 模型文件 → 数据集 → NERO → 主相机 → "
                                         "腕部相机 → 夹爪 → 加载 SmolVLA（NERO 不会被自动使能）")
        JC.fixed_label(tk, pr, 1, textvariable=self.v_prep, font=("TkDefaultFont", 9, "bold"), wraplength=1100,
                       pack={"fill": "x", "pady": (3, 0)})
        drv_f = ttk.LabelFrame(pg["prep"], text=f"移动硬盘 {ST.DRIVE_LABEL}：LeRobot + SmolVLA 环境 / 模型 / 数据集 / 训练文件")
        drv_f.pack(fill="x", pady=2)
        d0 = ttk.Frame(drv_f)                                   # V4.6 STATUS LAMPS
        d0.pack(fill="x", padx=6, pady=(2, 0))
        self.lamp_drive = StatusLamp(tk, ttk, d0, "硬盘", 22)
        self.lamp_env = StatusLamp(tk, ttk, d0, "环境", 22)
        self.lamp_model = StatusLamp(tk, ttk, d0, "模型", 34)
        self.lamp_job = StatusLamp(tk, ttk, d0, "后台任务", 26)
        self.lamp_drive.set("busy", "检测中")
        self.drv_detail = ttk.Frame(drv_f)
        self.btn_drv_fold = ttk.Button(d0, text="硬盘详情 ▾", command=self.toggle_drive_detail)
        self.btn_drv_fold.pack(side="right")
        self.v_drive = tk.StringVar(value=f"正在检测移动硬盘 {ST.DRIVE_LABEL} ...")
        JC.fixed_label(
            tk, self.drv_detail, 3, width=1160, textvariable=self.v_drive,
            font=("TkDefaultFont", 9), wraplength=1150, pack={"fill": "x", "padx": 6, "pady": 1}
        )
        d1 = ttk.Frame(self.drv_detail)
        d1.pack(fill="x", padx=6, pady=1)
        self.btn_drive = ttk.Button(d1, text="检测硬盘", command=self.on_drive_refresh)
        self.btn_drive.pack(side="left", padx=(0, 2))
        ttk.Button(d1, text=f"挂载 {ST.DRIVE_LABEL}", command=self.on_drive_mount).pack(side="left", padx=2)
        ttk.Button(d1, text="初始化目录", command=self.on_drive_init).pack(side="left", padx=2)
        self.btn_install = ttk.Button(d1, text="安装 LeRobot+SmolVLA 到移动硬盘", command=self.on_install_env)
        self.btn_install.pack(side="left", padx=2)
        self.btn_verify = ttk.Button(d1, text="验证环境", command=self.on_verify_env)
        self.btn_verify.pack(side="left", padx=2)
        ttk.Button(d1, text="停止后台任务", command=self.on_stop_job).pack(side="left", padx=2)
        ttk.Button(d1, text="打开存储目录", command=self.on_open_storage).pack(side="left", padx=2)
        ttk.Button(d1, text="本机旧文件→硬盘", command=self.on_migrate_legacy).pack(side="left", padx=2)
        ttk.Button(d1, text="安全移除硬盘", command=self.on_eject_drive).pack(side="right", padx=2)
        d2 = ttk.Frame(self.drv_detail)
        d2.pack(fill="x", padx=6, pady=(1, 3))
        ttk.Label(d2, text="Python:").pack(side="left")
        self._py_mode_labels = {"uv312": "3.12 (uv 装到硬盘 · 最新 LeRobot)", "system": "系统 Python (3.10 → LeRobot 0.4.4)"}
        self.v_py_mode = tk.StringVar(value=self._py_mode_labels.get(ST.DEFAULT_PY_MODE, self._py_mode_labels["uv312"]))
        ttk.Combobox(d2, textvariable=self.v_py_mode, values=list(self._py_mode_labels.values()), width=30,
                     state="readonly").pack(side="left", padx=2)
        ttk.Label(d2, text="基础 Python:").pack(side="left")
        self.v_base_py = tk.StringVar(value=os.environ.get("NERO_V44_BASE_PYTHON", "/usr/bin/python3"))
        ttk.Entry(d2, textvariable=self.v_base_py, width=16).pack(side="left", padx=2)
        ttk.Label(d2, text="pip 包:").pack(side="left")
        self.v_pip_spec = tk.StringVar(value=ST.DEFAULT_PIP_SPEC)
        ttk.Entry(d2, textvariable=self.v_pip_spec, width=22).pack(side="left", padx=2)
        self.v_dl_after = tk.BooleanVar(value=True)
        ttk.Checkbutton(d2, text="装完自动下载 smolvla_base", variable=self.v_dl_after).pack(side="left", padx=4)
        ttk.Label(d2, text="后台任务:").pack(side="left", padx=(6, 1))
        self.v_job = tk.StringVar(value="无")
        ttk.Label(d2, textvariable=self.v_job, width=60).pack(side="left", padx=2)
        mrow = ttk.Frame(pg["prep"])
        mrow.pack(fill="x", padx=2, pady=2)
        ttk.Label(mrow, text="SmolVLA 权重:").pack(side="left")
        self.v_ckpt = tk.StringVar(value="lerobot/smolvla_base")
        ttk.Entry(mrow, textvariable=self.v_ckpt, width=24).pack(side="left", padx=2)
        self.btn_dl_model = ttk.Button(mrow, text="下载 SmolVLA 模型", command=self.on_download_smolvla)
        self.btn_dl_model.pack(side="left", padx=2)
        self.btn_chk_status = ttk.Button(mrow, text="查询 LeRobot/模型状态", command=self.on_check_lerobot_status)
        self.btn_chk_status.pack(side="left", padx=2)
        self.btn_connect = ttk.Button(mrow, text="① 加载 SmolVLA & 连接 Adapter", command=self.on_connect_and_load)
        self.btn_connect.pack(side="left", padx=(10, 2))
        self.lamp_1 = StatusLamp(tk, ttk, mrow, "①", 10)
        trow = ttk.Frame(pg["prep"])
        trow.pack(fill="x", padx=2, pady=2)
        ttk.Label(trow, text="数据集名:").pack(side="left")
        self.v_ds_name = tk.StringVar(value="nero_v4_demo")
        ttk.Entry(trow, textvariable=self.v_ds_name, width=16).pack(side="left", padx=2)
        ttk.Label(trow, text="任务指令:").pack(side="left", padx=(8, 2))
        self.v_task = tk.StringVar(value="Put the red block into the left box, then put the blue block into the right box")
        ttk.Entry(trow, textvariable=self.v_task, width=42).pack(side="left", padx=2, fill="x", expand=True)

        # ---------------------------------------------------------------- ② 示教采集
        self.teach_bar = None
        if self.pipe is not None and hasattr(self.pipe, "set_teach_mode"):
            import NERO_V462_Teach_Gravity as TG
            self.teach_bar = TG.TeachBar(
                tk, ttk, pg["teach"], self.pipe, jpanel=self.jpanel, log=self._log, cam_tab=self.cam_tab,
                busy_fn=lambda: (f"「{self._run_kind}」正在执行" if self._run_busy.locked() else ""),
                safe_tab_fn=lambda: self.safe_tab, pack={"fill": "x", "padx": 2, "pady": (0, 2)})
        # V4.7: two columns - left = recording, right = the episode browser (browse / play / replay / delete)
        body = ttk.Frame(pg["teach"])
        body.pack(fill="x", padx=2, pady=(0, 2))
        left = ttk.Frame(body)
        left.pack(side="left", fill="y", anchor="n")
        rb = ttk.Frame(left)
        rb.pack(fill="x", pady=2)
        self.btn_rec = tk.Button(rb, text="● 开始示教录制", command=self.on_toggle_record, bg="#c62828", fg="white",
                                 activebackground="#e53935", activeforeground="white",
                                 font=("TkDefaultFont", 10, "bold"), relief="raised", bd=1, padx=8, pady=2)
        self.btn_rec.pack(side="left", padx=(0, 4))
        self.btn_discard = ttk.Button(rb, text="✗ 放弃本次（不保存）", command=self.on_discard_record, state="disabled")
        self.btn_discard.pack(side="left", padx=(0, 4))
        rbg = ttk.Frame(left)
        rbg.pack(fill="x", pady=2)
        ttk.Label(rbg, text="夹爪：").pack(side="left")
        ttk.Button(rbg, text="夹爪示教: 张开", command=lambda: self.on_teach_gripper(True)).pack(side="left", padx=2)
        ttk.Button(rbg, text="夹爪示教: 夹紧", command=lambda: self.on_teach_gripper(False)).pack(side="left", padx=2)
        self.v_grip_teach = tk.StringVar(value="空格键 = 张开 / 夹紧 切换")
        ttk.Label(rbg, textvariable=self.v_grip_teach, foreground="#1565c0").pack(side="left", padx=6)
        # V4.6.11: what was recorded so far + the next episode number, always in view on this page
        self.v_next_ep = tk.StringVar(value="下一个：Episode #0")
        ttk.Label(left, textvariable=self.v_next_ep, font=("TkDefaultFont", 12, "bold"),
                  foreground="#2e7d32").pack(anchor="w", pady=(4, 0))
        self.v_recent_eps = tk.StringVar(value="还没有录制的 Episode")
        JC.fixed_label(tk, left, 2, width=560, textvariable=self.v_recent_eps, font=("TkDefaultFont", 9),
                       wraplength=560, pack={"anchor": "w"})
        self.v_rec_status = tk.StringVar(value="录制 / 回放真机：先在上方选「实机 NERO」后端；「仿真替身」只录 / 回放虚拟机械臂")
        JC.fixed_label(tk, left, 2, width=560, textvariable=self.v_rec_status, font=("TkDefaultFont", 9),
                       wraplength=560, pack={"anchor": "w"})
        # ---- V4.7 EPISODE BROWSER
        epf = ttk.LabelFrame(body, text="Episode 浏览 · 播放 · 回放 · 删除")
        epf.pack(side="left", fill="both", expand=True, padx=(10, 0))
        eh = ttk.Frame(epf)
        eh.pack(fill="x", padx=4, pady=(2, 2))
        ttk.Label(eh, text="数据集").pack(side="left")
        self.v_browse_ds = tk.StringVar(value=self.dataset.dataset_name)
        self.cb_browse_ds = ttk.Combobox(eh, textvariable=self.v_browse_ds, width=22, state="readonly",
                                         postcommand=self._fill_ds_list)
        self.cb_browse_ds.pack(side="left", padx=4)
        self.cb_browse_ds.bind("<<ComboboxSelected>>", lambda _e: self.on_browse_dataset())
        ttk.Button(eh, text="刷新", command=self.refresh_episode_list).pack(side="left", padx=2)
        ttk.Button(eh, text="打开文件夹", command=self.on_open_dataset_folder).pack(side="left", padx=2)
        self.v_browse_info = tk.StringVar(value="")
        ttk.Label(eh, textvariable=self.v_browse_info, foreground="#607d8b").pack(side="left", padx=6)
        tvf = ttk.Frame(epf)
        tvf.pack(fill="x", padx=4)
        cols = ("ep", "dur", "frames", "state", "saved", "size", "task")
        self.ep_tree = ttk.Treeview(tvf, columns=cols, show="headings", height=5, selectmode="browse")
        for c, t, w, anc in (("ep", "Episode", 64, "center"), ("dur", "时长", 62, "e"), ("frames", "帧数", 58, "e"),
                             ("state", "状态", 64, "center"), ("saved", "保存时间", 120, "center"),
                             ("size", "大小", 70, "e"), ("task", "任务", 220, "w")):
            self.ep_tree.heading(c, text=t)
            self.ep_tree.column(c, width=w, anchor=anc, stretch=(c == "task"))
        self.ep_tree.tag_configure("bad", foreground="#c62828")
        self.ep_tree.tag_configure("invalid", foreground="#9e9e9e")
        ysb = ttk.Scrollbar(tvf, orient="vertical", command=self.ep_tree.yview)
        self.ep_tree.configure(yscrollcommand=ysb.set)
        self.ep_tree.pack(side="left", fill="x", expand=True)
        ysb.pack(side="left", fill="y")
        self.ep_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_ep_select())
        self.ep_tree.bind("<Double-1>", lambda _e: self.on_play_episode_video())
        ea = ttk.Frame(epf)
        ea.pack(fill="x", padx=4, pady=(3, 3))
        self.btn_play_video = ttk.Button(ea, text="▶ 播放录像（主相机 + 腕部相机）", command=self.on_play_episode_video)
        self.btn_play_video.pack(side="left", padx=(0, 6))
        ttk.Separator(ea, orient="vertical").pack(side="left", fill="y", padx=4)
        self.v_replay_speed = tk.StringVar(value="0.5×")
        ttk.Button(ea, text="回放 Episode", command=self.on_replay_episode).pack(side="left", padx=2)
        ttk.Combobox(ea, textvariable=self.v_replay_speed, values=("0.25×", "0.5×", "1×"), width=6,
                     state="readonly").pack(side="left", padx=2)
        ttk.Label(ea, text="（实机后端 = NERO 照着走）", foreground="#607d8b").pack(side="left")
        ttk.Button(ea, text="删除 Episode", command=self.on_delete_episode).pack(side="right", padx=2)
        self.v_ep_idx = tk.StringVar(value="0")
        self.v_ep_info = tk.StringVar(value="")
        self.players = []

        # ---------------------------------------------------------------- ③ 训练
        rb2 = ttk.Frame(pg["train"])
        rb2.pack(fill="x", padx=2, pady=2)
        ttk.Label(rb2, text="LeRobot 真实微调 (移动硬盘):").pack(side="left")
        ttk.Button(rb2, text="导出为 LeRobot 数据集", command=self.on_export_lerobot).pack(side="left", padx=2)
        self.v_export_overwrite = tk.BooleanVar(value=True)
        ttk.Checkbutton(rb2, text="覆盖", variable=self.v_export_overwrite).pack(side="left")
        ttk.Label(rb2, text="步数:").pack(side="left")
        self.v_train_steps = tk.StringVar(value="20000")
        ttk.Entry(rb2, textvariable=self.v_train_steps, width=7).pack(side="left")
        ttk.Label(rb2, text="batch:").pack(side="left")
        self.v_train_bs = tk.StringVar(value="8")
        ttk.Entry(rb2, textvariable=self.v_train_bs, width=4).pack(side="left")
        ttk.Label(rb2, text="额外参数:").pack(side="left")
        self.v_train_extra = tk.StringVar(value="")
        ttk.Entry(rb2, textvariable=self.v_train_extra, width=26).pack(side="left", padx=2)
        ttk.Button(rb2, text="开始 SmolVLA 微调 (GPU)", command=self.on_train_lerobot).pack(side="left", padx=2)
        ttk.Button(rb2, text="使用最新训练结果", command=self.on_use_latest_ckpt).pack(side="left", padx=2)
        rb4 = ttk.Frame(pg["train"])
        rb4.pack(fill="x", padx=2, pady=2)
        ttk.Label(rb4, text="高级（不是真 SmolVLA，只用于接口测试）:", foreground="#757575").pack(side="left")
        ttk.Button(rb4, text="快速微调 (内置 NERO VLA)", command=self.on_finetune).pack(side="left", padx=2)

        # ---------------------------------------------------------------- ④ 执行
        cb = ttk.Frame(pg["run"])
        cb.pack(fill="x", padx=2, pady=2)
        self.v_allow = tk.BooleanVar(value=False)
        self.chk_allow = ttk.Checkbutton(
            cb, text="② 允许机器人控制 (Allow Control)", variable=self.v_allow, command=self.on_toggle_allow
        )
        self.chk_allow.pack(side="left")
        self.lamp_2 = StatusLamp(tk, ttk, cb, "②", 8)
        cb2 = ttk.Frame(pg["run"])
        cb2.pack(fill="x", padx=2, pady=2)
        ttk.Button(cb2, text="③ 执行 SmolVLA 闭环", command=self.on_run_vla_loop).pack(side="left", padx=(0, 3))
        ttk.Button(cb2, text="③ 执行顶层 Agent 双路径任务", command=self.on_run_agent_task).pack(side="left", padx=3)
        self.lamp_3 = StatusLamp(tk, ttk, cb2, "③", 16)
        ttk.Button(cb2, text="④ 停止执行", command=self.on_stop_queue).pack(side="left", padx=3)
        ttk.Label(pg["run"], text="顺序：① 准备页加载模型 → ② 勾选允许控制 → ③ 执行 → ④ 停止；急停按钮在最上方，任何时候都能按。",
                  foreground="#607d8b").pack(anchor="w", padx=2)

        # ---------------------------------------------------------------- ⑤ 评估 / 高级
        ab = ttk.Frame(pg["adv"])
        ab.pack(fill="x", padx=2, pady=2)
        ttk.Button(ab, text="Sim-to-Real 6级验证", command=self.on_sim2real_ladder).pack(side="left", padx=2)
        ttk.Button(ab, text="运行 A/B/C 对比 Benchmark", command=self.on_run_benchmark).pack(side="left", padx=2)
        ttk.Button(ab, text="Phase 0 审查", command=self.on_phase0_audit).pack(side="left", padx=(10, 2))
        self.btn_ipc = ttk.Button(ab, text="IPC 服务", command=self.on_toggle_ipc)
        self.btn_ipc.pack(side="left", padx=2)
        ab2 = ttk.Frame(pg["adv"])
        ab2.pack(fill="x", padx=2, pady=2)
        ttk.Label(ab2, text="控制权模式（手动切换，一般不需要）:").pack(side="left")
        self.v_mode = tk.StringVar(value="monitor")
        self.cb_mode = ttk.Combobox(ab2, textvariable=self.v_mode, values=list(CONTROL_MODES), width=11, state="readonly")
        self.cb_mode.pack(side="left", padx=2)
        self.cb_mode.bind("<<ComboboxSelected>>", lambda _e: self.on_mode_select())

        # V4.6.9: space bar toggles the teach gripper (both hands are on the arm while recording)
        self._grip_teach_open = True
        self._rec_phase = "idle"                 # V4.6.11: idle / entering / recording / saving
        self._rec_auto_teach = False
        self._n_eps_cache = None
        try:
            parent.winfo_toplevel().bind_all("<KeyPress-space>", self._on_space, add="+")
        except Exception:
            pass
        self._update_ep_info()
        self._refresh_recent_eps()

        # V4.6.2: monitors (side by side) in one row, the log below them over the full width with 「Copy Log」
        bottom = ttk.Frame(parent)
        bottom.pack(fill="both", expand=True, pady=(3, 0))
        mon_row = ttk.Frame(bottom)
        mon_row.pack(side="top", fill="x")
        mons = ttk.Frame(mon_row)
        mons.pack(side="left", anchor="n")
        mon_side = ttk.Frame(mon_row)
        mon_side.pack(side="left", fill="both", expand=True, anchor="n")
        # Section 11.2: Live Status (V4.6.2: right of the monitors, so the log below gets the height)
        st_f = ttk.LabelFrame(mon_side, text="4. 实时状态（本体 · 双相机 · 数据集 · 安全）")
        st_f.pack(side="top", fill="x", pady=(0, 2))
        self.v_obs_stat = tk.StringVar(value="等待读取观测...")
        lab_st = JC.fixed_label(
            tk, st_f, 8, textvariable=self.v_obs_stat,
            font=("TkFixedFont", 9), wraplength=340, pack={"fill": "x", "padx": 6, "pady": 2}
        )
        # wrap at the width that is really left next to the monitors (narrow 1420 window ... full HD)
        lab_st.holder.bind("<Configure>", lambda e, l_=lab_st: l_.configure(wraplength=max(200, e.width - 8)), add="+")
        self.monitor = None
        if self.rct_tab is not None and hasattr(self.rct_tab, "_draw_main"):
            try:
                import NERO_V365_Real_Cup_Task as RCT
                self.monitor = RCT.CameraMonitorMirror(tk, ttk, mons, self.rct_tab, yolo_parent=mon_side)
            except Exception as exc:
                ttk.Label(mons, text=f"相机监控不可用：{exc}").pack(side="left", anchor="n")
        else:
            ttk.Label(mons, text="相机监控：「Agentic实机任务」页不可用", width=30).pack(side="left", anchor="n")
        # Log panel (full width; the Copy Log row is packed first at the bottom so it is never pushed off screen)
        logbar = ttk.Frame(bottom)
        logbar.pack(side="bottom", fill="x")
        ttk.Button(logbar, text="Copy Log", command=self.copy_log).pack(side="right")
        lf = ttk.Frame(bottom)
        lf.pack(side="top", fill="both", expand=True, pady=(3, 0))
        self.txt = tk.Text(lf, height=8, wrap="word", font=("TkFixedFont", 9))
        self.txt.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lf, command=self.txt.yview)
        sb.pack(side="right", fill="y")
        self.txt.configure(yscrollcommand=sb.set)

    def toggle_drive_detail(self, show: Optional[bool] = None) -> None:
        """V4.6.2: fold / unfold the drive details of section 0 (lamps stay)."""
        vis = bool(self.drv_detail.winfo_manager())
        show = (not vis) if show is None else bool(show)
        if show and not vis:
            self.drv_detail.pack(fill="x")
            self.btn_drv_fold.configure(text="硬盘详情 ▴")
        elif not show and vis:
            self.drv_detail.pack_forget()
            self.btn_drv_fold.configure(text="硬盘详情 ▾")

    def copy_log(self) -> None:
        """V4.6.2: whole log text -> clipboard."""
        try:
            t = self.txt.get("1.0", "end")
            self.txt.clipboard_clear()
            self.txt.clipboard_append(t)
            self._log("[OK] [V4.6.2 UI] 日志已复制到剪贴板")
        except Exception:
            pass

    # ------------------------------------------------------------------ Callbacks
    def on_backend_change(self) -> None:
        bk = self.v_backend.get()
        if bk == "real":
            grip = getattr(self.rct_tab, "grip", None) if self.rct_tab else None
            self.executor.bind_real_robot(
                self.pipe,
                lambda: getattr(self.jpanel, "model", None),
                gripper=grip,
                cam_tab=self.cam_tab,
                jpanel=self.jpanel,
            )
            self.benchmark.env_domain = "real_robot"
            self._log("[OK] [V4.2 UI] 已绑定实机 NERO + D435i/OV4688 + SO-101 夹爪数据源")
            prof = self.vla_policy.model_profile()
            if prof["loaded"] and not prof["real_allowed"]:
                self._log(f"[WARN] [V4.6 UI] 当前模型不允许在实机执行：{'；'.join(prof['reasons'])}")
        else:
            self.executor.bind_mock()
            self.benchmark.env_domain = "simulation"
            self._log(f"[OK] [V4.2 UI] 已切换至 {bk} 观测与执行后端")

    def on_mode_select(self) -> None:
        m = self.v_mode.get()
        r = self.arbiter.request_mode(m, owner_tag=f"ui_{m}", flush_fn=self.executor.stop_and_flush)
        if not r.get("ok"):
            self._log(f"[WARN] [V4.2 UI] 模式切换被拒: {r.get('error')}")
            self.v_mode.set(self.arbiter.mode)
        else:
            self._log(f"[OK] [V4.6 UI] 控制权模式 = {self.arbiter.mode}")
        self.v_allow.set(self.arbiter.status()["allow_control"])

    def on_toggle_allow(self) -> None:
        want = bool(self.v_allow.get())
        if want and self.arbiter.mode in ("monitor", "estop"):
            pol = self.v_policy.get() or "smolvla"
            self.arbiter.request_mode(pol, owner_tag=f"ui_{pol}", flush_fn=self.executor.stop_and_flush)
            self.v_mode.set(self.arbiter.mode)
        r = self.arbiter.set_allow_control(want, reason="UI Checkbutton")
        if not r.get("ok"):
            self._log(f"[WARN] [V4.2 UI] 无法开启控制权: {r.get('error')}")
            self.v_allow.set(False)
        elif want and self.v_backend.get() == "real":
            prof = self.vla_policy.model_profile()
            if prof["loaded"] and not prof["real_allowed"] and self.arbiter.mode == "smolvla":
                self._log("[WARN] [V4.6 UI] 已允许控制，但当前 SmolVLA 模型没有为 NERO 训练过，③ 在实机上会被拒绝执行")

    def on_connect_and_load(self) -> None:
        self.obs_bridge.set_task_instruction(self.v_task.get())
        ckpt_target = self.v_ckpt.get().strip()

        if self._loading:
            self._log("[INFO] [V4.6 UI] ① 正在加载中，请等待（不要重复点击）")
            return
        self._loading = True
        self.lamp_1.set("busy", "加载中")
        self.lamp_model.set("busy", "加载中…（硬盘上的 SmolVLA 要 10–90 s）")
        self.btn_connect.configure(state="disabled")
        self._log(f"[INFO] [V4.6 UI] ① 开始加载 SmolVLA：{ckpt_target or 'lerobot/smolvla_base'}")
        t0 = time.time()

        def _job():
            try:
                r = self.vla_policy.load(ckpt_target)
                if not self.adapter.is_connected:
                    self.adapter.connect()
                prof = self.vla_policy.model_profile()
                self._log(f"[OK] [V4.6 UI] ① 加载完成（{time.time() - t0:.1f} s）：后端={r.get('backend')} · "
                          f"{prof['state_dim'] or '?'}/{prof['action_dim'] or '?'} 维 · "
                          f"{'可以在实机执行' if prof['real_allowed'] else '不允许在实机执行'}"
                          + (f" · {'；'.join(prof['reasons'])}" if prof["reasons"] else ""))
                self.q.put(("btn_connect", "① 已加载 & 已连接"))
                self.q.put(("lamp", ("1", "ok", "已加载")))
            except Exception as exc:
                self._log(f"[FAIL] [V4.6 UI] ① 加载失败（{time.time() - t0:.1f} s）: {exc}")
                self.q.put(("lamp", ("1", "bad", "加载失败")))
            finally:
                self._loading = False
                self.q.put(("btn_connect_state", "normal"))

        threading.Thread(target=_job, daemon=True).start()

    # ------------------------------------------------------------------ V4.7 ONE-CLICK PREPARE
    def _prep_set(self, key, state, text) -> None:
        self.q.put(("prep_lamp", (key, state, text)))

    def on_prepare_all(self) -> None:
        """V4.7: check every link in its logical order (and fix what can be fixed safely: mount the drive, create the
        folders, connect the cameras / the gripper, load SmolVLA).  NERO itself is never started from here (enabling
        the arm stays a deliberate step on 「实机NERO调试」).  All ready -> the 「① 准备」 lamp turns green, any problem
        -> yellow, with the first problem in the result line."""
        if getattr(self, "_prep_busy", False):
            self._log("[INFO] [V4.7 PREPARE] 正在准备中，请稍等")
            return
        if self.recorder.is_recording or self._run_busy.locked():
            self._log("[WARN] [V4.7 PREPARE] 正在录制 / 执行任务，先停止再一键准备")
            return
        snap = {"backend": self.v_backend.get(), "ckpt": self.v_ckpt.get().strip() or "lerobot/smolvla_base",
                "task": self.v_task.get()}
        self._prep_busy = True
        self.btn_prepare.configure(state="disabled", text="准备中 …")
        for k, _t in PREP_STEPS:
            self.prep_lamps[k].set("off", "-")
        self._set_prep_tab("busy")
        self.v_prep.set("准备中 …（验证移动硬盘上的 LeRobot 环境第一次可能要 1–2 分钟）")
        self._log("[INFO] [V4.7 PREPARE] 一键准备开始")
        threading.Thread(target=lambda: self._prep_job(snap), daemon=True, name="v47_prepare").start()

    def _set_prep_tab(self, state) -> None:
        self.prep_state = state
        try:
            self.sub_nb.tab(self.pages["prep"], image=self._tab_imgs.get(state, self._tab_imgs["off"]))
        except Exception:
            pass

    def _wait_for(self, fn, timeout_s, step_s=0.25) -> bool:
        t_end = time.time() + timeout_s
        while time.time() < t_end:
            try:
                if fn():
                    return True
            except Exception:
                pass
            time.sleep(step_s)
        try:
            return bool(fn())
        except Exception:
            return False

    def _prep_job(self, snap) -> None:
        t0 = time.time()
        res = {}                                     # key -> (state, short, detail)

        def done(key, state, short, detail=""):
            res[key] = (state, short, detail)
            self._prep_set(key, state, short)
            self._log(f"[{'OK' if state == 'ok' else ('INFO' if state == 'off' else 'WARN')}] [V4.7 PREPARE] "
                      f"{dict(PREP_STEPS)[key]}：{short}" + (f" —— {detail}" if detail else ""))

        try:
            # 1. drive (mount it when it is plugged in but not mounted)
            self._prep_set("drive", "busy", "检测")
            st = ST.status(probe_venv=False)
            if not st["drive"].get("found"):
                r = ST.mount_drive()
                st = ST.status(probe_venv=False)
                if not st["drive"].get("found"):
                    done("drive", "warn", "未挂载", f"插上移动硬盘 {ST.DRIVE_LABEL}（挂载：{r.get('message') or r.get('error')}）")
            if st["drive"].get("found"):
                fs = st.get("fs") or {}
                done("drive", "ok" if fs.get("grade") == "good" else "warn", f"{fs.get('free_gb', '?')} GB",
                     "" if fs.get("grade") == "good" else "硬盘空间 / 文件系统需要注意（见「硬盘详情」）")
            st = self._drive_refresh_job(probe=False) or st
            on_drive = bool(st.get("drive", {}).get("found"))
            L = st.get("layout") or ST.layout()
            # 2. folders
            if not on_drive:
                done("dirs", "off", "跳过", "先挂载移动硬盘")
            elif st.get("initialized"):
                done("dirs", "ok", "就绪")
            else:
                r = ST.ensure_layout(L)
                done("dirs", "ok" if r.get("ok") else "warn", "已建立" if r.get("ok") else "失败", r.get("error") or "")
            # 3. LeRobot environment on the drive
            if not on_drive:
                done("env", "off", "跳过", "先挂载移动硬盘")
            else:
                self._prep_set("env", "busy", "验证中")
                st2 = self._drive_refresh_job(probe=True) or {}
                v = st2.get("venv") or {}
                if v.get("ready"):
                    done("env", "ok" if v.get("cuda") else "warn", "CUDA ✓" if v.get("cuda") else "无 CUDA",
                         "" if v.get("cuda") else "torch 看不到 GPU，训练会很慢")
                else:
                    done("env", "warn", "未就绪", "「硬盘详情」→ 安装 LeRobot+SmolVLA 到移动硬盘" if not v.get("exists")
                         else ("验证超时，再试一次" if v.get("timeout") else "环境不完整，重新安装"))
            # 4. model files
            ck = snap["ckpt"]
            local = os.path.isdir(os.path.expanduser(ck))
            have = (ST.has_weights(os.path.expanduser(ck)) if local
                    else (ck == "lerobot/smolvla_base" and ST.has_weights(L.get("smolvla_base_dir", ""))))
            if have:
                done("model", "ok", "在硬盘上" if not local else "本地")
            else:
                done("model", "warn", "没有", f"{ck} 不在本地：点「下载 SmolVLA 模型」，或在「③ 训练」用最新训练结果")
            # 5. dataset folder
            self.q.put(("repoint_dataset", None))
            time.sleep(0.4)
            root = os.path.abspath(self.dataset.root_dir)
            on_d = bool(on_drive and root.startswith(os.path.abspath(L.get("root", "/nonexistent"))))
            done("dataset", "ok" if on_d else "warn", "在硬盘上" if on_d else "在本机",
                 self.dataset.dataset_dir if on_d else f"示教数据会存到本机：{self.dataset.dataset_dir}")
            # 6-9. real robot side
            if snap["backend"] != "real":
                for k in ("nero", "cam", "wrist", "grip"):
                    done(k, "off", "跳过", "后端是「仿真替身 / MuJoCo」，不检查实机")
            else:
                if self.pipe is None:
                    done("nero", "warn", "无", "没有实机控制管线")
                else:
                    import NERO_V313_Real_Robot as RR
                    lam = {k: self.pipe.lamp(k) for k in RR.STEP_KEYS}
                    bad = [RR.STEP_LABELS[i] for i, k in enumerate(RR.STEP_KEYS) if lam[k] not in (RR.GREEN, RR.AMBER)]
                    done("nero", "ok" if not bad else "warn", "就绪" if not bad else "未启动",
                         "" if not bad else f"先到「实机NERO调试」页一键启动（停在 {bad[0]}）——一键准备不会自动使能机械臂")
                ct = self.cam_tab
                for key, attr, fn_name, label in (("cam", "src", "on_connect", "主相机"),
                                                  ("wrist", "wrist_src", "on_wrist_connect", "腕部相机")):
                    run = lambda a=attr: getattr(ct, a, None) is not None and getattr(ct, a).running()
                    if ct is None:
                        done(key, "warn", "无", "Camera调试 页不可用")
                        continue
                    if not run():
                        self._prep_set(key, "busy", "连接中")
                        fn = getattr(ct, fn_name, None)
                        if fn is not None:
                            self._post(fn)
                    ok = self._wait_for(run, 15.0)
                    done(key, "ok" if ok else "warn", "已连接" if ok else "没连上",
                         "" if ok else f"{label}没有画面：检查 USB，或在 Camera调试 页手动连接")
                rt = self.rct_tab
                g = getattr(rt, "grip", None) if rt is not None else None
                if g is None:
                    done("grip", "warn", "无", "Agentic实机任务 页不可用")
                else:
                    if not getattr(g, "connected", False) and hasattr(rt, "on_grip_connect"):
                        self._prep_set("grip", "busy", "连接中")
                        self._post(rt.on_grip_connect)
                    ok = self._wait_for(lambda: getattr(g, "connected", False), 10.0)
                    done("grip", "ok" if ok else "warn", "已连接" if ok else "没连上",
                         "" if ok else "SO-101 没连上：检查 USB（/dev/ttyACM*）和电源，或在 Agentic实机任务 页手动连接")
            # 10. load SmolVLA + connect the adapter
            if res["model"][0] != "ok" and not self.vla_policy.is_loaded:
                done("vla", "off", "跳过", "没有模型文件")
            else:
                self._prep_set("vla", "busy", "加载中")
                ok_l, why = self._load_vla_sync(ck)
                prof = self.vla_policy.model_profile()
                if not ok_l:
                    done("vla", "warn", "加载失败", why)
                elif snap["backend"] == "real" and not prof.get("real_allowed"):
                    done("vla", "warn", "不能上实机",
                         "已加载，但这个模型没为 NERO 训练，「④ 执行」在实机上会被拒绝（示教采集不受影响）："
                         + "；".join(prof.get("reasons") or []))
                else:
                    done("vla", "ok", "已加载")
        except Exception as exc:
            self._log(f"[FAIL] [V4.7 PREPARE] 出错：{exc}")
            res.setdefault("_error", ("warn", "出错", str(exc)))
        probs = [(k, v) for k, v in res.items() if v[0] not in ("ok", "off")]
        state = "ok" if not probs else "warn"
        if probs:
            k0, v0 = probs[0]
            text = (f"⚠ {len(probs)} 项需要处理 · 第一项：{dict(PREP_STEPS).get(k0, k0)} —— {v0[2] or v0[1]}"
                    f"（用时 {time.time() - t0:.0f} s，详情见下方日志）")
        else:
            skipped = [dict(PREP_STEPS)[k] for k, v in res.items() if v[0] == "off"]
            text = (f"✓ 准备就绪（用时 {time.time() - t0:.0f} s）" + (f" · 跳过：{'、'.join(skipped)}" if skipped else ""))
        self._log(f"[{'OK' if state == 'ok' else 'WARN'}] [V4.7 PREPARE] {text}")
        self.q.put(("prep_done", (state, text)))

    def _load_vla_sync(self, ckpt) -> tuple:
        """V4.7: the 「① 加载」 work in the calling (worker) thread -> (ok, error text)"""
        if self._loading:
            ok = self._wait_for(lambda: not self._loading, 180.0, 0.5)
            return (self.vla_policy.is_loaded, "" if ok else "上一次加载还没完成")
        if self.vla_policy.is_loaded and str(getattr(self.vla_policy, "model_id_or_path", "")) == ckpt and \
                self.adapter.is_connected:
            return True, ""
        self._loading = True
        self.q.put(("lamp", ("1", "busy", "加载中")))
        t0 = time.time()
        try:
            r = self.vla_policy.load(ckpt)
            if not self.adapter.is_connected:
                self.adapter.connect()
            self._log(f"[OK] [V4.7 PREPARE] SmolVLA 加载完成（{time.time() - t0:.1f} s）：后端={r.get('backend')}")
            self.q.put(("btn_connect", "① 已加载 & 已连接"))
            self.q.put(("lamp", ("1", "ok", "已加载")))
            return True, ""
        except Exception as exc:
            self.q.put(("lamp", ("1", "bad", "加载失败")))
            return False, str(exc)[:200]
        finally:
            self._loading = False

    def on_check_lerobot_status(self) -> None:
        target = self.v_ckpt.get().strip()
        threading.Thread(
            target=lambda: self.vla_policy.query_environment_and_model_status(target),
            daemon=True,
        ).start()

    def on_download_smolvla(self) -> None:
        target = self.v_ckpt.get().strip() or "lerobot/smolvla_base"
        L = ST.layout()
        if os.path.isfile(L["venv_python"]) and not os.path.isdir(os.path.expanduser(target)):
            # V4.4: download with the drive venv (huggingface_hub there, HF cache on the drive), detached + logged
            self._start_job("download", [L["venv_python"], "-I", WORKER_PATH, "download", "--repo", target,
                                         "--local-dir", L["smolvla_base_dir"] if target == "lerobot/smolvla_base"
                                         else os.path.join(L["models"], target.replace("/", "--"))])
            return
        self.btn_dl_model.configure(text="下载中...")

        def _dl_job():
            res = self.vla_policy.download_model_weights(repo_id=target)
            if res.get("ok") and res.get("local_dir"):
                self.q.put(("set_ckpt", str(res["local_dir"])))
                self.q.put(("btn_dl", "✓ 模型已下载"))
            else:
                self.q.put(("btn_dl", " 下载 SmolVLA 模型"))

        threading.Thread(target=_dl_job, daemon=True).start()

    # ------------------------------------------------------------------ V4.4 EMBODIED_AI drive
    def _startup_drive_check(self) -> None:
        self._drive_refresh_job(probe=False)
        try:
            if self.job.adopt_previous():
                self._log(f"[INFO] [V4.4 DRIVE] 上次启动的后台任务仍在运行: {self.job.kind} (pid {self.job.pid})，"
                          f"继续显示日志 {self.job.log_path}")
        except Exception:
            pass

    def _drive_refresh_job(self, probe: bool = False) -> Dict[str, Any]:
        try:
            st = ST.status(probe_venv=probe)
        except Exception as exc:
            self.q.put(("drive", f"检测移动硬盘失败: {exc}"))
            return {}
        if probe and st.get("venv") is not None:
            self._venv_st = st["venv"]
        elif self._venv_st and st["venv_exists"]:
            st["venv"] = self._venv_st
        self._drive_found_last = bool(st["drive"].get("found"))
        self.q.put(("apply_env", st["layout"]))   # V4.5 CRASH FIX: os.environ is changed on the Tk main thread only
        if not self.job.running():
            self.job.L = st["layout"]
        self.q.put(("drive", ST.format_status(st)))
        self.q.put(("repoint_dataset", None))
        fs = st.get("fs") or {}
        if not st["drive"].get("found"):
            self.q.put(("lamp", ("drive", "bad", "未挂载")))
        else:
            self.q.put(("lamp", ("drive", "ok" if fs.get("grade") == "good" else "warn",
                                 f"{fs.get('free_gb', '?')} GB 空闲" + ("" if fs.get("grade") == "good" else " ⚠"))))
        v = st.get("venv")
        if v is None:
            self.q.put(("lamp", ("env", "off" if not st["venv_exists"] else "warn",
                                 "未安装" if not st["venv_exists"] else "已安装（未验证）")))
        elif v.get("ready"):
            self.q.put(("lamp", ("env", "ok", f"就绪 · CUDA {'✓' if v.get('cuda') else '✗'}")))
        else:
            self.q.put(("lamp", ("env", "bad", "验证超时" if v.get("timeout") else ("未安装" if not v.get("exists") else "不完整"))))
        return st

    def _repoint_dataset(self) -> None:
        """Main thread: move the active dataset to the drive (or back) when the drive appears / disappears."""
        want = ST.default_dataset_root()
        if os.path.abspath(want) == os.path.abspath(self.dataset.root_dir) or self.recorder.is_recording:
            return
        try:
            self.dataset = NEROLeRobotDataset(dataset_name=self.dataset.dataset_name, root_dir=want)
            self.recorder.dataset = self.dataset
            self._log(f"[INFO] [V4.4 DRIVE] 示教数据集位置: {self.dataset.dataset_dir}")
        except Exception as exc:
            self._log(f"[WARN] [V4.4 DRIVE] 无法切换数据集位置到 {want}: {exc}")

    def on_drive_refresh(self) -> None:
        self._log(f"[INFO] [V4.6 UI] 检测移动硬盘 {ST.DRIVE_LABEL} ...")
        self.btn_drive.configure(state="disabled", text="检测中…")
        self.lamp_drive.set("busy", "检测中")

        def _job():
            st = self._drive_refresh_job(probe=False)
            if st:
                self._log(ST.format_status(st).split("\n")[0])
            self.q.put(("btn_drive", None))
        threading.Thread(target=_job, daemon=True).start()

    def on_verify_env(self) -> None:
        if str(self.btn_verify.cget("state")) == "disabled":
            self._log("[INFO] [V4.6 UI] 正在验证中，请等待")
            return
        self.btn_verify.configure(state="disabled", text="验证中…")
        self.lamp_env.set("busy", "验证中…")

        def _job():
            t0 = time.time()
            self._log(f"[INFO] [V4.4 DRIVE] 正在验证移动硬盘上的 LeRobot 环境（第一次从硬盘导入 torch 可能要 1–2 分钟，"
                      f"最多等 {ST.VENV_PROBE_TIMEOUT_S:.0f} s）...")
            st = self._drive_refresh_job(probe=True)
            if st:
                self._log(ST.format_status(st) + f"\n[INFO] [V4.6 UI] 验证用时 {time.time() - t0:.1f} s")
            self.q.put(("btn_verify", None))
        threading.Thread(target=_job, daemon=True).start()

    def on_drive_mount(self) -> None:
        def _job():
            r = ST.mount_drive()
            self._log(f"[{'OK' if r.get('ok') else 'FAIL'}] [V4.4 DRIVE] 挂载 {ST.DRIVE_LABEL}: "
                      f"{r.get('message') or r.get('error')}")
            self._drive_refresh_job(probe=False)
        threading.Thread(target=_job, daemon=True).start()

    def on_drive_init(self) -> None:
        def _job():
            r = ST.ensure_layout(ST.layout())
            if r.get("ok"):
                self._log(f"[OK] [V4.4 DRIVE] 目录已就绪: {r['root']}（终端里可 source activate_nero_lerobot.sh 使用硬盘环境）")
            else:
                self._log(f"[FAIL] [V4.4 DRIVE] {r.get('error')}")
            self._drive_refresh_job(probe=False)
        threading.Thread(target=_job, daemon=True).start()

    def _require_drive(self) -> Optional[Dict[str, Any]]:
        L = ST.layout()
        if not L["on_drive"] and not ST.allow_internal():
            self._log(f"[FAIL] [V4.4 DRIVE] {L['drive'].get('hint')}（大文件不会写到本机磁盘；"
                      f"确实要写本机请设置 NERO_V44_ALLOW_INTERNAL=1）")
            return None
        r = ST.ensure_layout(L)
        if not r.get("ok"):
            self._log(f"[FAIL] [V4.4 DRIVE] {r.get('error')}")
            return None
        return L

    def _start_job(self, kind: str, cmd, meta: Optional[Dict[str, Any]] = None) -> bool:
        L = self._require_drive()
        if L is None:
            return False
        if self.job.running():
            self._log(f"[WARN] [V4.4 JOB] 已有后台任务在运行: {self.job.kind}（先等它结束或点「 停止后台任务」）")
            return False
        self.job.L = L
        r = self.job.start(kind, list(cmd), env=ST.child_env(L, for_heavy_job=True), meta=meta)
        if r.get("ok"):
            self._job_seen_running = True
            self._log(f"[OK] [V4.4 JOB] 后台任务 {kind} 已启动 (pid {r['pid']})；日志写在移动硬盘: {r['log']}")
            return True
        self._log(f"[FAIL] [V4.4 JOB] {r.get('error')}")
        return False

    def on_install_env(self) -> None:
        base_py = self.v_base_py.get().strip() or "/usr/bin/python3"
        spec = self.v_pip_spec.get().strip() or ST.DEFAULT_PIP_SPEC
        dl_after = bool(self.v_dl_after.get())
        py_label = self.v_py_mode.get()   # Tk variables are read here, on the main thread
        py_mode = next((k for k, v in self._py_mode_labels.items() if v == py_label), ST.DEFAULT_PY_MODE)

        def _job():
            L = self._require_drive()
            if L is None:
                return
            fs = ST.fs_checks(L["root"])
            for n in fs["notes"]:
                self._log(f"[WARN] [V4.4 INSTALL] {n}")
            if fs["grade"] == "bad":
                self._log("[FAIL] [V4.4 INSTALL] 这个硬盘现在不能安装 Python 环境（见上面的提示）")
                return
            if fs["free_gb"] < ST.MIN_FREE_GB_INSTALL and not self.ask(
                    "移动硬盘空间偏小", f"{ST.DRIVE_LABEL} 只剩 {fs['free_gb']} GB（建议 ≥ {ST.MIN_FREE_GB_INSTALL:.0f} GB）。仍然安装？"):
                return
            chk = ST.base_python_check(base_py)
            if not chk.get("ok"):
                self._log(f"[FAIL] [V4.4 INSTALL] {chk.get('error')}")
                return
            if not self.ask("安装 LeRobot + SmolVLA",
                            f"将在移动硬盘创建虚拟环境（{py_label}）并安装 {spec}：\n{L['venv']}\n\n"
                            f"下载约 3–4 GB、占用约 10 GB，pip 缓存和临时文件也放在移动硬盘，不占本机磁盘。\n"
                            f"安装在后台进行（关闭控制台也不会中断），开始吗？"):
                return
            script = ST.install_script(L, base_py=base_py, pip_spec=spec, download_model=dl_after, worker_path=WORKER_PATH,
                                       py_mode=py_mode)
            sp = os.path.join(L["logs"], f"install_{time.strftime('%Y%m%d_%H%M%S')}.sh")
            with open(sp, "w", encoding="utf-8") as f:
                f.write(script)
            os.chmod(sp, 0o755)
            self._log(f"[INFO] [V4.4 INSTALL] 安装脚本: {sp}（也可以在终端里手动运行: bash {sp}）")
            self._start_job("install", ["bash", sp], meta={"script": sp})
        threading.Thread(target=_job, daemon=True).start()

    def on_stop_job(self) -> None:
        if self.job.stop():
            self._log(f"[INFO] [V4.4 JOB] 已发送停止信号给后台任务 {self.job.kind} (pid {self.job.pid})")
        else:
            self._log("[INFO] [V4.4 JOB] 没有正在运行的后台任务")

    def on_open_storage(self) -> None:
        L = ST.layout()
        path = L["root"] if os.path.isdir(L["root"]) else (L["drive"].get("mountpoint") or "")
        if not path:
            self._log(f"[WARN] [V4.4 DRIVE] {L['drive'].get('hint')}")
            return
        try:
            import subprocess
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            self._log(f"[WARN] [V4.4 DRIVE] 打不开文件夹 {path}: {exc}")

    def on_migrate_legacy(self) -> None:
        def _job():
            items = ST.legacy_locations()
            if not items:
                self._log("[INFO] [V4.4 MIGRATE] 本机没有找到旧的 LeRobot / SmolVLA 文件")
                return
            for it in items:
                self._log(f"[INFO] [V4.4 MIGRATE] 本机 {it['key']}: {it['path']} ({it['size_gb']} GB)"
                          + ("" if it["movable"] else "  —— 可在终端执行 `pip cache purge` 释放"))
            movable = [it for it in items if it["movable"]]
            L = ST.layout()
            if not movable:
                return
            if not L["on_drive"]:
                self._log(f"[FAIL] [V4.4 MIGRATE] {L['drive'].get('hint')}")
                return
            tot = sum(it["size_gb"] for it in movable)
            if not self.ask("复制到移动硬盘", f"把本机上 {len(movable)} 处旧文件（共 {tot:.2f} GB）复制到 {L['root']}？"):
                return
            r = ST.migrate_legacy(L, delete_source=False, log=self._log)
            if r.get("ok") and r.get("moved") and self.ask(
                    "释放本机空间", f"已全部复制到移动硬盘。删除本机上的原文件（{tot:.2f} GB）？"):
                ST.delete_sources(r["moved"], log=self._log)
            self._drive_refresh_job(probe=False)
        threading.Thread(target=_job, daemon=True).start()

    def on_eject_drive(self) -> None:
        def _job():
            if self.recorder.is_recording:
                self._log("[FAIL] [V4.4 DRIVE] 正在录制示教，请先停止并保存 Episode")
                return
            if self.job.running():
                if not self.ask("后台任务运行中", f"后台任务 {self.job.kind} 还在使用移动硬盘。停止它并移除硬盘？"):
                    return
                self.job.stop()
                time.sleep(2.0)
            self.vla_policy.close_worker()
            r = ST.unmount_drive()
            if r.get("ok"):
                self._log(f"[OK] [V4.4 DRIVE] {ST.DRIVE_LABEL} 已卸载{'并断电' if r.get('power_off') else ''}，可以拔下硬盘")
            else:
                self._log(f"[FAIL] [V4.4 DRIVE] {r.get('error')}")
            self._drive_refresh_job(probe=False)
        threading.Thread(target=_job, daemon=True).start()

    def _lerobot_export_root(self, L: Dict[str, Any]) -> str:
        return os.path.join(L["datasets_lerobot"], self.dataset.dataset_name)

    def on_export_lerobot(self) -> None:
        L = ST.layout()
        if not os.path.isfile(L["venv_python"]):
            self._log("[FAIL] [V4.4 EXPORT] 移动硬盘上还没有 LeRobot 环境，请先「 安装 LeRobot+SmolVLA」")
            return
        n_train = len(self.dataset.list_episodes(split="train", valid_only=True))
        if n_train == 0:
            self._log(f"[FAIL] [V4.4 EXPORT] 数据集 {self.dataset.dataset_name} 没有有效的 train Episode，先录制示教")
            return
        out = self._lerobot_export_root(L)
        cmd = [L["venv_python"], "-I", WORKER_PATH, "export", "--src", self.dataset.dataset_dir,
               "--repo-id", f"nero/{self.dataset.dataset_name}", "--root", out, "--fps", str(self.dataset.fps),
               "--splits", "train"] + (["--overwrite"] if self.v_export_overwrite.get() else [])
        if self._start_job("export", cmd, meta={"out": out}):
            self._log(f"[INFO] [V4.4 EXPORT] {n_train} 个 train Episode → {out}")

    def on_train_lerobot(self) -> None:
        steps_txt, bs_txt = self.v_train_steps.get(), self.v_train_bs.get()   # Tk variables: main thread only
        ck_txt, extra_txt = self.v_ckpt.get().strip(), self.v_train_extra.get()

        def _job():
            L = ST.layout()
            out = self._lerobot_export_root(L)
            if not os.path.isfile(os.path.join(out, "meta", "info.json")):
                self._log(f"[FAIL] [V4.4 TRAIN] 还没有导出的 LeRobot 数据集 {out}，请先点「 导出为 LeRobot 数据集」")
                return
            try:
                steps, bs = int(steps_txt), int(bs_txt)
            except ValueError:
                self._log("[FAIL] [V4.4 TRAIN] 步数 / batch 必须是整数")
                return
            vs = self._venv_st or ST.venv_status(L)
            self._venv_st = vs
            if not vs.get("ready"):
                self._log(f"[FAIL] [V4.4 TRAIN] LeRobot 环境不完整: {vs.get('error') or vs.get('lerobot_error') or vs.get('smolvla_error')}")
                return
            device = "cuda"
            if not vs.get("cuda"):
                if not self.ask("没有 CUDA", "移动硬盘环境里的 torch 检测不到 GPU，用 CPU 训练会非常慢。仍然用 CPU 训练？"):
                    return
                device = "cpu"
            ck = os.path.expanduser(ck_txt)
            policy_path = ck if ST.has_weights(ck) else ST.DEFAULT_BASE_MODEL_ID
            run = f"smolvla_{self.dataset.dataset_name}_{time.strftime('%Y%m%d_%H%M%S')}"
            cmd = ST.train_command(L, f"nero/{self.dataset.dataset_name}", out, run, policy_path=policy_path,
                                   steps=steps, batch_size=bs, save_freq=max(500, min(steps // 5, 5000)),
                                   device=device, extra_args=extra_txt)
            if self._start_job("train", cmd, meta={"run": os.path.join(L["training"], run)}):
                self._log(f"[INFO] [V4.4 TRAIN] 起点模型 {ST.resolve_model_target(policy_path, L)} · {steps} 步 · batch {bs} · "
                          f"{device} · 输出 {os.path.join(L['training'], run)}（关闭控制台训练也会继续）")
        threading.Thread(target=_job, daemon=True).start()

    def on_use_latest_ckpt(self) -> None:
        p = ST.latest_training_checkpoint()
        if not p:
            self._log("[INFO] [V4.4 TRAIN] 移动硬盘 training/ 下还没有训练好的 checkpoint")
            return
        self.v_ckpt.set(p)
        self._log(f"[OK] [V4.4 TRAIN] SmolVLA 权重已设为最新训练结果: {p}（点「① 加载」生效）")

    def _poll_job(self) -> None:
        txt = self.job.read_new()
        if txt.strip():
            lines = [ln for ln in txt.splitlines() if ln.strip()]
            for ln in lines[-40:]:
                self.txt.insert("end", f"[{self.job.kind}] {ln}\n")
            self.txt.see("end")
            if "NERO_JOB_DONE ok" in txt:
                self.job.meta["done_ok"] = True
        running = self.job.running()
        if running:
            el = int(time.time() - self.job.started_at)
            self.v_job.set(f"{self.job.kind} 运行中 {el // 60} 分 {el % 60} 秒 · 日志 {os.path.basename(self.job.log_path or '')}")
        elif self._job_seen_running or self.job.kind:
            if self._job_seen_running:
                rc = self.job.returncode()
                ok = self.job.meta.get("done_ok") or (rc == 0)
                self._log(f"[{'OK' if ok else 'FAIL'}] [V4.4 JOB] 后台任务 {self.job.kind} 结束"
                          f"{'' if rc is None else f' (退出码 {rc})'}；完整日志: {self.job.log_path}")
                if self.job.kind == "download" and ok:
                    L = ST.layout()
                    if ST.has_weights(L["smolvla_base_dir"]) and self.v_ckpt.get().strip() == "lerobot/smolvla_base":
                        self._log(f"[OK] [V4.4 DOWNLOAD] smolvla_base 在硬盘上: {L['smolvla_base_dir']}（加载时自动使用）")
                if self.job.kind == "train" and ok:
                    self.on_use_latest_ckpt()
                threading.Thread(target=lambda: self._drive_refresh_job(probe=self.job.kind == "install"), daemon=True).start()
            self._job_seen_running = False
            self.v_job.set(f"无（上次: {self.job.kind}）" if self.job.kind else "无")
        if running:
            self._job_seen_running = True

    def on_toggle_ipc(self) -> None:
        if self.ipc_server is None:
            self.ipc_server = NEROIPCServer(self.obs_bridge, self.executor, log=self._log)
            self.ipc_server.start()
            self.btn_ipc.configure(text="停止 IPC")
        else:
            self.ipc_server.stop()
            self.ipc_server = None
            self.btn_ipc.configure(text="IPC 服务")

    def on_phase0_audit(self) -> None:
        threading.Thread(
            target=lambda: self._log(AUD.format_audit_summary(AUD.run_phase0_audit(bridge=self.obs_bridge))),
            daemon=True,
        ).start()

    def _start_run(self, kind: str, fn) -> None:
        """V4.6: one run at a time; start / end always logged; lamp ③ shows running / result."""
        if not self._run_busy.acquire(blocking=False):
            self._log(f"[WARN] [V4.6 UI] 「{self._run_kind}」还在执行，这次点击「{kind}」被忽略（先等它结束或点 ④ 停止）")
            return
        self._run_kind = kind
        self.lamp_3.set("busy", "执行中")
        st = self.arbiter.status()
        self._log(f"[INFO] [V4.6 UI] ③ 开始「{kind}」· 后端={self.executor.status().get('backend_kind')} · "
                  f"控制权={st['mode']} · 允许控制={'是' if st['allow_control'] else '否'}")
        if not st["allow_control"] and kind != "Sim-to-Real 6级验证":
            self._log("[WARN] [V4.6 UI] 还没有勾选「② 允许机器人控制」：动作会被仲裁器拒绝")

        def _job():
            t0 = time.time()
            res: Dict[str, Any] = {}
            try:
                res = fn() or {}
            except Exception as exc:
                res = {"ok": False, "reason": f"EXCEPTION: {exc}"}
            finally:
                self._run_busy.release()
            status = res.get("status") or ("ok" if res.get("ok") else "failed")
            if status in ("executed_unverified", "unverified"):
                lamp, txt, tag = "warn", "已执行·未验证", "WARN"
            elif res.get("ok"):
                lamp, txt, tag = "ok", "完成", "OK"
            else:
                lamp, txt, tag = "bad", "被拒绝" if status == "refused" else "失败", "FAIL"
            self.q.put(("lamp", ("3", lamp, txt)))
            self._log(f"[{tag}] [V4.6 UI] ③「{kind}」结束：{status} · {res.get('reason') or ''} · {time.time() - t0:.1f} s")
        threading.Thread(target=_job, daemon=True).start()

    def on_run_vla_loop(self) -> None:
        # V4.5 CRASH / FREEZE FIX: loading SmolVLA from the drive takes 10-60 s -> never on the Tk main thread
        ckpt = self.v_ckpt.get().strip()
        task = self.v_task.get().strip()

        def _fn():
            if not self.vla_policy.is_loaded:
                self._log("[INFO] [V4.6 UI] 模型还没加载，先加载（相当于 ①）")
                self.vla_policy.load(ckpt)
            if not self.adapter.is_connected:
                self.adapter.connect()
            return self.vla_policy.run_vla_closed_loop(self.adapter, task, max_chunks=3)
        self._start_run("SmolVLA 闭环", _fn)

    def on_run_agent_task(self) -> None:
        ckpt = self.v_ckpt.get().strip()   # V4.5: Tk variables read here, slow loading in the worker thread
        task = self.v_task.get().strip()
        pol = self.v_policy.get() or "smolvla"

        def _fn():
            if pol == "smolvla" and not self.vla_policy.is_loaded:
                self._log("[INFO] [V4.6 UI] 模型还没加载，先加载（相当于 ①）")
                self.vla_policy.load(ckpt)
            if not self.adapter.is_connected:
                self.adapter.connect()
            r = self.orchestrator.run_task(task, policy=pol, max_vla_chunks=3)
            fr = ((r.get("failed_subtask") or {}).get("result") or {})
            return dict(r, reason=r.get("reason") or fr.get("reason"))
        self._start_run(f"顶层 Agent 任务（{pol}）", _fn)

    # ------------------------------------------------------------------ V4.6.9 teach gripper / replay
    def _real_gripper(self):
        g = getattr(self.rct_tab, "grip", None) if self.rct_tab is not None else None
        return g if (g is not None and getattr(g, "connected", False)) else None

    def on_teach_gripper(self, open_: bool) -> None:
        """V4.6.9: the teach gripper buttons / space bar DRIVE the real SO-101 on the real back end (before: only a
        number in the recording); the recorded gripper action follows; on the simulated back ends only the number."""
        v = 1.0 if open_ else 0.0
        self._grip_teach_open = bool(open_)
        word = "张开" if open_ else "夹紧"
        g = self._real_gripper() if self.v_backend.get() == "real" else None
        self.v_grip_teach.set(f"夹爪示教：{word}（空格键切换）")
        if g is None:
            self.recorder.set_teach_gripper_norm(v)
            why = ("后端不是「实机 NERO」" if self.v_backend.get() != "real" else
                   "SO-101 夹爪没有连接（Agentic实机任务 页 → 连接）")
            self._log(f"[WARN] [V4.6.9 TEACH GRIPPER] {word}：只记录数值，真夹爪不动（{why}）")
            return

        def _drive(_v):
            try:                                      # V4.6.11: a failure is reported (V4.6.9: silently swallowed)
                r = g.open_gap(REPLAY_OPEN_GAP_MM) if open_ else g.clamp(REPLAY_CLAMP_PCT)
            except Exception as exc:
                self._log(f"[FAIL] [V4.6.11 TEACH GRIPPER] SO-101 {word} 没执行：{exc}")
                return
            bad = isinstance(r, dict) and r.get("ok") is False and not r.get("held")
            self._log(f"[{'WARN' if bad else 'OK'}] [V4.6.9 TEACH GRIPPER] SO-101 {word}：{r}")
        threading.Thread(target=lambda: self.recorder.set_teach_gripper_norm(v, drive_gripper_fn=_drive),
                         daemon=True, name="v469_teach_gripper").start()

    TEXT_CLASSES = ("Entry", "TEntry", "Text", "TCombobox", "Spinbox", "TSpinbox")

    def _on_space(self, e=None):
        """V4.6.9 / V4.6.11: space bar = toggle the teach gripper while this tab is shown.
        V4.6.11 BUG FIX: a ttk button that was CLICKED keeps the keyboard focus (ttk::clickToFocus), and Tk's own
        <space> binding of TButton then pressed that button: after 「● 开始示教录制」 the space bar stopped the recording,
        after 「夹爪示教: 夹紧」 it pressed 「夹紧」 and this handler toggled to 「张开」 at the same time (two gripper commands
        racing -> nothing visible).  Now every widget of this tab carries the bindtag 'NeroVLASpace' FIRST, whose
        handler returns "break" so the class binding never runs; this bind_all handler covers widgets outside the
        tab.  Text fields keep their space."""
        try:
            if not self.visible_fn():
                return None
            w = getattr(e, "widget", None) or self.parent.winfo_toplevel().focus_get()
            if isinstance(w, str):
                w = self.parent.nametowidget(w)
            if (w is not None and w.winfo_class() in self.TEXT_CLASSES
                    and str(w) != str(getattr(self, "txt", ""))):      # the log box is not for typing
                return None
        except Exception:
            return None
        self.on_teach_gripper(not self._grip_teach_open)
        return "break"

    def _install_space_tag(self, root_widget=None) -> int:
        """V4.6.11: 'NeroVLASpace' in front of the bindtags of every non-text widget of this tab -> n widgets"""
        root_widget = root_widget or self.parent
        try:
            root_widget.bind_class("NeroVLASpace", "<KeyPress-space>", self._on_space)
        except Exception:
            return 0
        n, todo = 0, [root_widget]
        while todo:
            w = todo.pop()
            try:
                todo.extend(w.winfo_children())
                if w.winfo_class() in self.TEXT_CLASSES:
                    continue
                tags = list(w.bindtags())
                if "NeroVLASpace" not in tags:
                    w.bindtags(("NeroVLASpace",) + tuple(tags))
                    n += 1
            except Exception:
                pass
        return n

    def _update_ep_info(self) -> None:
        """V4.7: the selected episode in one line (the browser's header)"""
        try:
            idx = int(self.v_ep_idx.get())
        except ValueError:
            idx = None
        row = next((r for r in getattr(self, "_ep_rows", []) if r["idx"] == idx), None)
        if row is None:
            self.v_ep_info.set("（没有选中 Episode）")
        else:
            self.v_ep_info.set(f"#{row['idx']} · {row['dur_s']:.1f} s · {row['frames']} 帧 · "
                               f"{'有效' if row['valid'] else '无效'} · {row['saved_at']}"
                               + ("" if row["exists"] else " · ⚠ 文件不在了"))
        try:
            self.v_browse_info.set(self.v_ep_info.get())
        except Exception:
            pass

    # ------------------------------------------------------------------ V4.7 EPISODE BROWSER
    def _fill_ds_list(self) -> None:
        import NERO_V47_VLA_Tools as TL
        names = TL.list_datasets(self.dataset.root_dir)
        if self.dataset.dataset_name not in names:
            names.insert(0, self.dataset.dataset_name)
        self.cb_browse_ds.configure(values=names)

    def on_browse_dataset(self) -> None:
        """switch the active dataset (also the one new recordings go to)"""
        name = self.v_browse_ds.get().strip()
        if not name or name == self.dataset.dataset_name:
            return
        if self._rec_phase != "idle":
            self._log("[WARN] [V4.7 EPISODES] 录制 / 保存中，不能切换数据集")
            self.v_browse_ds.set(self.dataset.dataset_name)
            return
        self.dataset = NEROLeRobotDataset(dataset_name=name, root_dir=self.dataset.root_dir)
        self.recorder.dataset = self.dataset
        self.v_ds_name.set(name)
        self._log(f"[INFO] [V4.7 EPISODES] 数据集 → {self.dataset.dataset_dir}")
        self._refresh_recent_eps()

    def refresh_episode_list(self, select_idx=None) -> None:
        import NERO_V47_VLA_Tools as TL
        try:
            rows = TL.episode_rows(self.dataset)
        except Exception as exc:
            rows = []
            self._log(f"[WARN] [V4.7 EPISODES] 读不了 Episode 列表：{exc}")
        self._ep_rows = rows
        tv = self.ep_tree
        keep = select_idx
        if keep is None:
            try:
                keep = int(self.v_ep_idx.get())
            except ValueError:
                keep = None
        tv.delete(*tv.get_children())
        for r in rows:
            tags = () if r["exists"] else ("bad",)
            if r["exists"] and not r["valid"]:
                tags = ("invalid",)
            tv.insert("", "end", iid=str(r["idx"]), tags=tags, values=(
                f"#{r['idx']}", f"{r['dur_s']:.1f} s", r["frames"],
                ("有效" if r["valid"] else "无效") if r["exists"] else "文件缺失",
                r["saved_at"][5:16] if len(r["saved_at"]) >= 16 else r["saved_at"],
                (f"{r['size_mb']:.0f} MB" if r["size_mb"] >= 10 else f"{r['size_mb']:.1f} MB") if r["exists"] else "-",
                r["task"]))
        try:
            self.v_browse_ds.set(self.dataset.dataset_name)
        except Exception:
            pass
        if rows:
            want = keep if keep is not None and any(r["idx"] == keep for r in rows) else rows[0]["idx"]
            tv.selection_set(str(want))
            tv.see(str(want))
            self.v_ep_idx.set(str(want))
        self._update_ep_info()
        if not rows:
            self.v_browse_info.set(f"这个数据集还没有 Episode（{self.dataset.data_dir}）")

    def _on_ep_select(self) -> None:
        sel = self.ep_tree.selection()
        if sel:
            self.v_ep_idx.set(sel[0])
            self._update_ep_info()

    def _selected_row(self):
        try:
            idx = int(self.v_ep_idx.get())
        except ValueError:
            return None
        return next((r for r in getattr(self, "_ep_rows", []) if r["idx"] == idx), None)

    def on_open_dataset_folder(self) -> None:
        d = self.dataset.data_dir
        self._log(f"[INFO] [V4.7 EPISODES] 文件夹：{d}")
        try:
            import subprocess
            subprocess.Popen(["xdg-open", d], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            self._log(f"[WARN] [V4.7 EPISODES] 打不开文件夹（{exc}）：{d}")

    def _yolo_fn_for_player(self):
        """(fn(rgb) -> dets, note) with the Agentic tab's YOLOE settings, read here in the Tk thread"""
        rt = self.rct_tab
        if rt is None or not hasattr(rt, "yolo_cfg"):
            return None, "YOLOE 不可用：Agentic实机任务 页没有建立"
        try:
            cfg = rt.yolo_cfg()
        except Exception as exc:
            return None, f"YOLOE 不可用：{str(exc)[:60]}"
        if cfg.get("error"):
            return None, f"YOLOE 不可用：{cfg['error'][:60]}"
        det = cfg["detector"]
        prompts = list(cfg.get("prompts") or [])
        if not prompts or (len(prompts) == 1 and prompts[0].strip().lower() == "cup"):
            prompts = ["egg", "egg carton"]
        mode, path, conf = cfg.get("mode", "text"), cfg["path"], float(cfg.get("conf", 0.25))

        def fn(rgb):
            return det.detect(rgb, mode, path, prompts=prompts, conf=conf)
        return fn, f"YOLOE：{', '.join(prompts[:3])} · ≥ {conf:.2f}"

    def on_play_episode_video(self) -> None:
        """V4.7: play the recorded main + wrist camera video of the selected episode (YOLOE labels optional)"""
        row = self._selected_row()
        if row is None:
            self._log("[INFO] [V4.7 EPISODES] 先在列表里选一个 Episode")
            return
        if not row["exists"]:
            self._log(f"[WARN] [V4.7 EPISODES] Episode #{row['idx']} 的文件不在了：{row['file']}")
            return
        import NERO_V47_VLA_Tools as TL
        for pl in list(self.players):
            if pl.alive and pl.rd.path == row["file"]:
                pl.win.lift()
                return
        yfn, note = self._yolo_fn_for_player()
        try:
            pl = TL.EpisodePlayer(self.tk, self.ttk, self.parent.winfo_toplevel(), row["file"],
                                  title=f"Episode #{row['idx']} 录像 · {self.dataset.dataset_name}", yolo_fn=yfn,
                                  yolo_note=note, log=self._log,
                                  on_close=lambda p_: self.players.remove(p_) if p_ in self.players else None)
        except Exception as exc:
            self._log(f"[FAIL] [V4.7 EPISODES] 打不开录像：{exc}")
            return
        self.players.append(pl)
        self._log(f"[INFO] [V4.7 EPISODES] 播放 Episode #{row['idx']} 录像（{pl.rd.n} 帧）· {note}")

    def _post(self, fn) -> None:
        """run fn in the Tk thread (from any thread)"""
        self.q.put(("call", fn))

    def _real_teach_ok(self) -> bool:
        return (self.v_backend.get() == "real" and self.teach_bar is not None
                and hasattr(self.pipe, "set_teach_mode"))

    def _set_rec_buttons(self) -> None:
        ph = self._rec_phase
        txt = {"idle": "● 开始示教录制", "entering": "进入示教模式中 …", "recording": "■ 停止并保存 Episode",
               "saving": "保存中 …"}[ph]
        self.btn_rec.configure(text=txt, state="normal" if ph in ("idle", "recording") else "disabled",
                               bg={"idle": "#c62828", "recording": "#37474f"}.get(ph, "#9e9e9e"))
        self.btn_discard.configure(state="normal" if ph == "recording" else "disabled")

    def on_toggle_record(self) -> None:
        """V4.6.11: 「开始示教录制」 on the real back end enters teach mode FIRST (recording starts when the controller
        confirms), 「停止并保存」 stops the capture at once, exits teach mode (pose held) and saves in a worker thread
        (V4.6.10: compressing GBs of images froze the console)."""
        if self._rec_phase == "recording":
            self._finish_recording(discard=False)
            return
        if self._rec_phase != "idle":
            self._log(f"[INFO] [V4.6.11 RECORDER] 请稍等：{'正在进入示教模式' if self._rec_phase == 'entering' else '正在保存上一个 Episode'}")
            return
        if self.v_backend.get() != "real":
            self._log("[WARN] [V4.6.9 RECORDER] 后端是「" + {"mock": "仿真替身", "mujoco": "MuJoCo"}.get(self.v_backend.get(), "?")
                      + "」：录到的是虚拟机械臂，不是真机。要录真机示教：先在最上方选「实机 NERO」后端")
        ds_name = self.v_ds_name.get().strip() or "nero_v4_demo"
        if self.dataset.dataset_name != ds_name:
            self.dataset = NEROLeRobotDataset(dataset_name=ds_name)
            self.recorder.dataset = self.dataset
            self._refresh_recent_eps()
        if self._real_teach_ok() and not getattr(self.pipe, "teach_mode", False):
            if not self.ask("开始示教录制", "NERO 将先进入示教模式（可以用手带动），进入成功后自动开始录制。\n"
                                         "请先用手扶住机械臂（尤其是前臂和末端），确认周围无人。继续？"):
                return
            self._rec_phase = "entering"
            self._set_rec_buttons()
            self.teach_bar.enter_now(done=lambda r: self._post(lambda: self._teach_entered(r)))
            return
        self._start_recording(auto_teach=False)

    def _teach_entered(self, r) -> None:
        if not (r or {}).get("ok"):
            self._rec_phase = "idle"
            self._set_rec_buttons()
            self._log(f"[FAIL] [V4.6.11 RECORDER] 没进入示教模式（{(r or {}).get('error') or r}）→ 没有开始录制")
            return
        self._start_recording(auto_teach=True)

    def _start_recording(self, auto_teach) -> None:
        r = self.recorder.start_recording(self.v_task.get().strip(), split="train")
        if not r.get("ok"):
            self._rec_phase = "idle"
            self._set_rec_buttons()
            self._log(f"[FAIL] [V4.6.11 RECORDER] 没开始录制：{r.get('error')}")
            return
        self._rec_phase, self._rec_auto_teach = "recording", bool(auto_teach)
        self._set_rec_buttons()
        try:                                       # V4.6.11: keyboard to this tab, so the space bar reaches it
            self.parent.focus_set()
        except Exception:
            pass
        self._log(f"[INFO] [V4.6.11 RECORDER] 正在录制 Episode #{self.dataset.next_episode_index()}"
                  + ("（已进入示教模式）" if auto_teach else "") + " · 空格键 = 夹爪张开 / 夹紧")

    def on_discard_record(self) -> None:
        """V4.6.11: abandon the running recording - nothing is saved (user decision: discard)"""
        if self._rec_phase != "recording":
            return
        n = len(getattr(self.recorder, "_frames", []) or [])
        if not self.ask("放弃本次录制", f"放弃正在录制的 Episode（已录 {n} 帧，约 {n / max(1.0, self.recorder.fps):.0f} s）？\n"
                                     "放弃后不保存，临时文件直接删除。"):
            return
        self._finish_recording(discard=True)

    def _finish_recording(self, discard) -> None:
        cap = self.recorder.stop_capture()
        self._rec_phase = "saving"
        self._set_rec_buttons()
        if self._real_teach_ok() and getattr(self.pipe, "teach_mode", False):
            self.teach_bar.exit_now()                 # pose held, no move home
        self.v_rec_status.set("放弃中 …" if discard else
                              f"保存中 …（{len(cap['frames'])} 帧，在后台写入，界面可以继续用）")

        def job():
            try:
                res = self.recorder.discard_captured(cap) if discard else self.recorder.save_captured(cap, True)
            except Exception as exc:
                res = {"ok": False, "error": str(exc)}
                self._log(f"[FAIL] [V4.6.11 RECORDER] 保存失败：{exc}")
            self._post(lambda: self._after_save(res))
        threading.Thread(target=job, daemon=True, name="v4611_save_episode").start()

    def _after_save(self, res) -> None:
        self._rec_phase = "idle"
        self._set_rec_buttons()
        self._n_eps_cache = None
        if res.get("discarded"):
            self.v_rec_status.set(f"已放弃上一次录制（{res.get('frames')} 帧，没有保存）")
        elif res.get("ok"):
            ep = res["episode"]
            self.v_ep_idx.set(str(ep["episode_index"]))
            self.v_rec_status.set(f"已保存 Episode #{ep['episode_index']} → {ep.get('file')}")
        else:
            self.v_rec_status.set(f"保存失败：{res.get('error')}")
        self._update_ep_info()
        self._refresh_recent_eps()

    def _refresh_recent_eps(self) -> None:
        """V4.6.11: 「下一个：Episode #N」 + the last 5 episodes (duration, frames, valid, time) + the folder"""
        try:
            eps = self.dataset.list_episodes()
        except Exception:
            eps = []
        self._n_eps_cache = len(eps)
        try:
            self.refresh_episode_list()
        except Exception:
            pass
        nxt = (max(int(e["episode_index"]) for e in eps) + 1) if eps else 0
        self.v_next_ep.set(f"下一个：Episode #{nxt}")
        if not eps:
            self.v_recent_eps.set(f"还没有录制的 Episode · 保存位置：{self.dataset.data_dir}")
            return
        parts = []
        for e in sorted(eps, key=lambda x: -int(x["episode_index"]))[:5]:
            fps = float(e.get("fps") or 20.0)
            n = int(e.get("length") or 0)
            parts.append(f"#{e['episode_index']} {n / fps:.1f} s · {n} 帧 · {'有效' if e.get('valid') else '无效'} · "
                         f"{str(e.get('saved_at', ''))[11:16]}")
        self.v_recent_eps.set("最近：" + "  |  ".join(parts) + f"\n共 {len(eps)} 个 · 保存位置：{self.dataset.data_dir}")

    def on_replay_episode(self) -> None:
        """V4.6.9: real back end -> the real NERO replays the recording (one timed joint path per segment, gripper
        between segments, measured errors); simulated back ends -> the old executor replay, now with every failure
        in the log (before: a failed replay printed nothing)."""
        try:
            idx = int(self.v_ep_idx.get())
        except ValueError:
            return
        try:
            speed = float(self.v_replay_speed.get().rstrip("×x"))
        except ValueError:
            speed = 0.5
        if self.v_backend.get() == "real":
            model = getattr(self.jpanel, "model", None)
            if not self.ask("回放 Episode（实机）",
                            f"真实的 NERO 将回放 Episode #{idx}：先以 10% 速度移到录制起点，再按 {speed:g}× 速度走录下的轨迹，"
                            "夹爪按录制张开 / 夹紧。\n请确认周围无人、急停在手边。继续？"):
                return
            self._log(f"[INFO] [V4.6.9 REPLAY] 实机回放 Episode #{idx}（{speed:g}×）：控制权切到 teleop、允许控制")
            self.arbiter.request_mode("teleop", owner_tag="ui_replay")
            self.arbiter.set_allow_control(True, reason="Episode Replay (real)")
            self.v_mode.set(self.arbiter.mode)
            self.v_allow.set(True)
            g = self._real_gripper()
            threading.Thread(target=lambda: replay_episode_real(self.dataset, idx, self.pipe, model, gripper=g,
                                                                speed=speed, log=self._log,
                                                                stop_fn=lambda: self.arbiter.mode == "estop"),
                             daemon=True, name="v469_replay").start()
            return
        self._log(f"[INFO] [V4.6.9 REPLAY] 后端「{self.v_backend.get()}」：只在仿真里回放 Episode #{idx}，真机不会动"
                  "（要真机回放：最上方选「实机 NERO」）")
        self.arbiter.request_mode("teleop", owner_tag="ui_replay")
        self.arbiter.set_allow_control(True, reason="Episode Replay")
        self.v_mode.set(self.arbiter.mode)
        self.v_allow.set(True)

        def _job():
            try:
                r = self.recorder.replay_episode(idx, caller_mode="teleop")
            except Exception as exc:
                r = {"ok": False, "error": str(exc)}
            if not r.get("ok") and r.get("error"):
                self._log(f"[FAIL] [V4.6.9 REPLAY] 回放 Episode #{idx} 失败：{r.get('error')}")
        threading.Thread(target=_job, daemon=True).start()

    def on_delete_episode(self) -> None:
        """V4.7: confirm, close its player, delete the file + the index entry, refresh"""
        row = self._selected_row()
        if row is None:
            self._log("[INFO] [V4.7 EPISODES] 先在列表里选一个 Episode")
            return
        if self._rec_phase != "idle":
            self._log("[WARN] [V4.7 EPISODES] 录制 / 保存中，不能删除")
            return
        if not self.ask("删除 Episode", f"删除 Episode #{row['idx']}（{row['dur_s']:.1f} s，{row['frames']} 帧，"
                                       f"{row['saved_at']}）？\n文件会从硬盘上删掉，不能恢复：\n{row['file']}"):
            return
        for pl in list(self.players):
            if pl.rd.path == row["file"]:
                pl.close()
        ok = self.dataset.delete_episode(row["idx"])
        self._log(f"[{'OK' if ok else 'WARN'}] [V4.7 EPISODES] {'已删除' if ok else '索引已移除（文件本来就不在）'} "
                  f"Episode #{row['idx']}：{row['file']}")
        self._refresh_recent_eps()

    def on_finetune(self) -> None:
        n = len(self.dataset.list_episodes(split="train", valid_only=True))
        self._log(f"[INFO] [V4.6 UI] 快速微调（内置 NERO VLA，不是真 SmolVLA）：数据集 {self.dataset.dataset_name} · {n} 个 train Episode")

        def _job():
            r = self.vla_policy.finetune_on_dataset(self.dataset, epochs=20)
            if not r.get("ok"):
                self._log(f"[FAIL] [V4.6 UI] 快速微调失败：{r.get('error')}")
        threading.Thread(target=_job, daemon=True).start()

    def on_sim2real_ladder(self) -> None:
        self._start_run("Sim-to-Real 6级验证", lambda: self.sim2real.run_all_stages())

    def on_run_benchmark(self) -> None:
        task = self.v_task.get().strip()   # V4.5: read Tk variables on the main thread

        def _fn():
            for m_key in ("mode_a", "mode_b", "mode_c"):
                self.benchmark.run_single_trial(m_key, task, category="in_distribution", max_vla_chunks=2)
            self._log(self.benchmark.format_report_text())
            self._log("[INFO] [V4.6 UI] 注意：没有视觉 / 目标检查时，Benchmark 的结果都是「未验证」，成功率没有意义")
            return {"ok": True, "status": "done"}
        self._start_run("A/B/C Benchmark", _fn)

    def on_stop_queue(self) -> None:
        self._log("[INFO] [V4.6 UI] ④ 停止：取消当前子任务并清空动作队列")
        self.router.request_cancel()
        self.executor.stop_and_flush("manual_ui_stop")
        self._log("[INFO] [V4.2 UI] 已停止当前子任务与动作队列")

    def on_estop(self) -> None:
        self._log("[WARN] [V4.6 UI] 紧急急停！")
        self.router.request_cancel()
        self.executor.stop_and_flush("estop")
        self.arbiter.trigger_estop("UI 紧急急停按钮")
        self.v_mode.set("estop")
        self.v_allow.set(False)

    def on_clear_estop(self) -> None:
        self._log("[INFO] [V4.6 UI] 解除急停（控制权回到只读，需要重新勾选 ②）")
        self.arbiter.clear_estop()
        self.v_mode.set(self.arbiter.mode)
        self.v_allow.set(False)

    # ------------------------------------------------------------------ Periodic UI Refresh
    def _schedule(self) -> None:
        try:
            self.parent.after(200, self._tick)
        except Exception:
            pass

    def _tick(self) -> None:
        while True:
            try:
                kind, val = self.q.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self.txt.insert("end", val + "\n")
                self.txt.see("end")
            elif kind == "set_ckpt":
                self.v_ckpt.set(str(val))
            elif kind == "btn_dl":
                self.btn_dl_model.configure(text=str(val))
            elif kind == "btn_connect":
                self.btn_connect.configure(text=str(val))
            elif kind == "btn_connect_state":
                self.btn_connect.configure(state=str(val))
            elif kind == "btn_drive":
                self.btn_drive.configure(state="normal", text="检测硬盘")
            elif kind == "btn_verify":
                self.btn_verify.configure(state="normal", text="验证环境")
            elif kind == "lamp":
                name, state, text = val
                lamp = {"drive": self.lamp_drive, "env": self.lamp_env, "1": self.lamp_1, "3": self.lamp_3,
                        "job": self.lamp_job, "model": self.lamp_model}.get(name)
                if lamp is not None:
                    lamp.set(state, text)
                if name == "drive" and state == "bad":          # V4.6.2: drive missing -> show its buttons
                    self.toggle_drive_detail(True)
            elif kind == "drive":
                self.v_drive.set(str(val))
            elif kind == "prep_lamp":                 # V4.7 one-click prepare
                k, st_, tx = val
                if k in self.prep_lamps:
                    self.prep_lamps[k].set(st_, tx)
            elif kind == "prep_done":
                st_, tx = val
                self._prep_busy = False
                self.btn_prepare.configure(state="normal", text="一键准备")
                self.v_prep.set(tx)
                self._set_prep_tab(st_)
            elif kind == "call":                      # V4.6.11: a function posted by a worker thread
                try:
                    val()
                except Exception as exc:
                    self.txt.insert("end", f"[WARN] [V4.6.11 UI] {exc}\n")
            elif kind == "repoint_dataset":
                self._repoint_dataset()
            elif kind == "apply_env":
                try:
                    ST.apply_to_process_env(val)
                except Exception:
                    pass
        self._tick_n += 1
        try:
            self._poll_job()
        except Exception as exc:
            self.txt.insert("end", f"[WARN] [V4.4 JOB] {exc}\n")
        if self._tick_n % 25 == 0:  # ~5 s: notice the drive being plugged in / pulled out
            found = bool(ST.find_drive().get("found"))
            if self._drive_found_last is not None and found != self._drive_found_last:
                self._drive_found_last = found
                self._log(f"[INFO] [V4.4 DRIVE] 移动硬盘 {ST.DRIVE_LABEL} {'已插入' if found else '已拔出'}")
                if not found and getattr(self, "prep_state", "off") == "ok":     # V4.7: no longer ready
                    self._set_prep_tab("warn")
                    self.prep_lamps["drive"].set("warn", "已拔出")
                    self.v_prep.set("⚠ 移动硬盘已拔出：重新插上后再点「一键准备」")
                if not found:   # V4.5: closing the worker can wait up to 6 s -> not on the Tk main thread
                    threading.Thread(target=self.vla_policy.close_worker, daemon=True).start()
                threading.Thread(target=lambda: self._drive_refresh_job(probe=False), daemon=True).start()
        if self.visible_fn():
            self._refresh_status()
            if self.monitor is not None:
                self.monitor.refresh()
        self._schedule()

    def _refresh_status(self) -> None:
        # V4.6.11: the Tk thread does not read the hardware while recording (the record thread does, 20 Hz) and
        # otherwise only once a second; the episode count is cached (V4.6.10: episodes.jsonl read every 200 ms)
        if self.recorder.is_recording or self._tick_n % 5:
            obs = self.obs_bridge.peek_observation()
        else:
            obs = self.obs_bridge.get_synchronized_observation()
        q_deg = [round(math.degrees(float(v)), 1) for v in obs["robot_state"]]
        arb = self.arbiter.status()
        ex = self.executor.status()
        if self._n_eps_cache is None or self._tick_n % 25 == 0:
            self._n_eps_cache = len(self.dataset.list_episodes())
        n_eps = self._n_eps_cache
        avg_inf = round(float(np.mean(self.vla_policy.inference_latencies_ms)), 2) if self.vla_policy.inference_latencies_ms else 0.0
        bk = ex.get("backend_kind") or "mock"
        bk_txt = {"real": "实机 NERO", "mock": "仿真替身（机械臂不动）", "mujoco_twin": "MuJoCo 替身",
                  "mujoco_shared": "MuJoCo 仿真"}.get(bk, bk)
        self.v_banner.set(
            f"后端: {bk_txt} · 控制仲裁器: 模式={arb['mode'].upper()} · 允许控制={'✓ YES' if arb['allow_control'] else '✗ NO (只读保护)'} · "
            f"SmolVLA={self.vla_policy.backend_type} · 数据集={self.dataset.dataset_name} ({n_eps} eps) · "
            f"安全拦截={arb['trip_count']}"
        )
        self._refresh_lamps(arb)
        self.v_obs_stat.set(
            f"观测帧 #{obs['seq']} ({obs['env_mode']}) | 双相机同步差: {obs['sync_skew_ms']:.1f} ms ({'✓ 同步' if obs['sync_ok'] else '⚠ 超差'}) | "
            f"7关节 (°): {q_deg} | 夹爪: norm={obs['gripper_state']:.2f} ({obs['gripper_gap_mm']:.1f} mm)\n"
            f"执行器: 已执行 {ex['chunks_executed']} Chunks ({ex['steps_executed']} 步) · VLA平均推理 {avg_inf:.2f} ms · "
            f"当前子任务: {obs['task_instruction'] or '(空)'}"
        )

    def _refresh_lamps(self, arb: Dict[str, Any]) -> None:
        """V4.6: lamps that follow live state (② control authority, ④ e-stop, model, background job)."""
        self.lamp_2.set("ok" if arb["allow_control"] else "off", "允许" if arb["allow_control"] else "只读")
        self.lamp_4.set("bad" if arb["mode"] == "estop" else "off", "急停中" if arb["mode"] == "estop" else "正常")
        if hasattr(self, "v_arb_text"):                         # V4.6.9 top bar
            self.v_arb_text.set(f"控制权: {arb['mode']} · {'允许控制' if arb['allow_control'] else '只读'}")
            if hasattr(self, "recorder") and self.recorder.is_recording:
                n = len(getattr(self.recorder, "_frames", []) or [])
                self.v_rec_status.set(f"录制中：{n} 帧（{n / max(1.0, self.recorder.fps):.1f} s）· 后端 "
                                      f"{self.v_backend.get()}" + ("" if self.v_backend.get() == "real" else
                                                                   " —— 不是实机：录的是虚拟机械臂"))
        if not self._loading:
            prof = self.vla_policy.model_profile()
            if not prof["loaded"]:
                self.lamp_model.set("off", "未加载")
            elif prof["real_allowed"]:
                self.lamp_model.set("ok", f"{prof['backend']} · NERO 可用")
            else:
                self.lamp_model.set("warn", f"{prof['backend']} · 未为 NERO 训练（禁止实机）")
            if not prof["loaded"] and self.lamp_1.state == "ok":
                self.lamp_1.set("off", "未加载")
        if self.job.running():
            self.lamp_job.set("busy", f"{self.job.kind} 运行中")
        elif self.job.kind:
            ok = self.job.meta.get("done_ok") or self.job.returncode() == 0
            self.lamp_job.set("ok" if ok else "bad", f"{self.job.kind} {'完成' if ok else '失败/停止'}")
        else:
            self.lamp_job.set("off", "无")

    def shutdown(self) -> None:
        try:
            self.vla_policy.close()  # V4.4: stop the SmolVLA process running from the drive
        except Exception:
            pass
        try:
            if self.job.running():
                self.ext_log(f"[INFO] [V4.4 JOB] 后台任务 {self.job.kind} 继续运行 (pid {self.job.pid})，日志: {self.job.log_path}")
        except Exception:
            pass
        try:
            if self.recorder.is_recording:
                self.recorder.stop_and_save(episode_success=False)
        except Exception:
            pass
        try:
            if self.ipc_server is not None:
                self.ipc_server.stop()
        except Exception:
            pass
        try:
            self.adapter.disconnect()
        except Exception:
            pass
        try:
            self.obs_bridge.close()
        except Exception:
            pass
