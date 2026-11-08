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
HERE = os.path.dirname(os.path.realpath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import NERO_V400_Phase0_Audit as AUD
import NERO_V44_Storage as ST
from NERO_V4_Agent_Orchestrator import AgentOrchestrator
from NERO_V4_Benchmark import DualPathBenchmark, SimToRealVerifier
from NERO_V4_Control_Arbiter import CONTROL_MODES, ControlArbiter
from NERO_V4_Dataset_Recorder import DemonstrationRecorder, NEROLeRobotDataset
from NERO_V4_LeRobot_Adapter import NEROIPCServer, NEROLeRobotAdapter
from NERO_V4_Observation_Bridge import ObservationBridge
from NERO_V4_Policy_Router import PolicyRouter
from NERO_V4_SmolVLA_Policy import WORKER_PATH, SmolVLAPolicy
from NERO_V4_Task_Evaluator import TaskEvaluator
from NERO_V4_VLA_Action_Executor import VLAActionExecutor


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
    ):
        self.tk = tk
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

        # V4.4 Section 0: EMBODIED_AI removable drive (LeRobot venv, models, datasets, training runs)
        drv_f = ttk.LabelFrame(
            parent, text=f"0. 移动硬盘 {ST.DRIVE_LABEL}：LeRobot + SmolVLA 环境 / 模型 / 数据集 / 训练文件")
        drv_f.pack(fill="x", pady=2)
        d0 = ttk.Frame(drv_f)                                   # V4.6 STATUS LAMPS
        d0.pack(fill="x", padx=6, pady=(2, 0))
        self.lamp_drive = StatusLamp(tk, ttk, d0, "硬盘", 22)
        self.lamp_env = StatusLamp(tk, ttk, d0, "环境", 22)
        self.lamp_model = StatusLamp(tk, ttk, d0, "模型", 34)
        self.lamp_job = StatusLamp(tk, ttk, d0, "后台任务", 26)
        self.lamp_drive.set("busy", "检测中")
        self.v_drive = tk.StringVar(value=f"正在检测移动硬盘 {ST.DRIVE_LABEL} ...")
        JC.fixed_label(
            tk, drv_f, 3, width=1160, textvariable=self.v_drive,
            font=("TkDefaultFont", 9), wraplength=1150, pack={"fill": "x", "padx": 6, "pady": 1}
        )
        d1 = ttk.Frame(drv_f)
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
        d2 = ttk.Frame(drv_f)
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

        # Section 11.1: System Configuration
        cfg_f = ttk.LabelFrame(parent, text="1. 系统配置、策略路径与控制权仲裁 (System Config & Control Arbiter)")
        cfg_f.pack(fill="x", pady=2)

        r0 = ttk.Frame(cfg_f)
        r0.pack(fill="x", padx=6, pady=2)
        ttk.Label(r0, text="后端环境:").pack(side="left")
        self.v_backend = tk.StringVar(value="mock")
        for val, lab in (("mock", "仿真替身 (Twin/Mock)"), ("mujoco", "MuJoCo 双相机"), ("real", "实机 NERO + D435i/OV4688")):
            ttk.Radiobutton(r0, text=lab, value=val, variable=self.v_backend, command=self.on_backend_change).pack(side="left", padx=3)

        ttk.Label(r0, text="操作策略:").pack(side="left", padx=(8, 2))
        self.v_policy = tk.StringVar(value="smolvla")
        for p_val, p_lab in (("smolvla", "路径 B: SmolVLA"), ("traditional", "路径 A: Traditional Skills")):
            ttk.Radiobutton(r0, text=p_lab, value=p_val, variable=self.v_policy).pack(side="left", padx=3)

        ttk.Label(r0, text="控制权模式:").pack(side="left", padx=(8, 2))
        self.v_mode = tk.StringVar(value="monitor")
        self.cb_mode = ttk.Combobox(r0, textvariable=self.v_mode, values=list(CONTROL_MODES), width=11, state="readonly")
        self.cb_mode.pack(side="left", padx=2)
        self.cb_mode.bind("<<ComboboxSelected>>", lambda _e: self.on_mode_select())

        r1 = ttk.Frame(cfg_f)
        r1.pack(fill="x", padx=6, pady=2)
        ttk.Label(r1, text="数据集名:").pack(side="left")
        self.v_ds_name = tk.StringVar(value="nero_v4_demo")
        ttk.Entry(r1, textvariable=self.v_ds_name, width=16).pack(side="left", padx=2)
        ttk.Label(r1, text="SmolVLA 权重:").pack(side="left", padx=(6, 2))
        self.v_ckpt = tk.StringVar(value="lerobot/smolvla_base")
        ttk.Entry(r1, textvariable=self.v_ckpt, width=24).pack(side="left", padx=2)
        self.btn_dl_model = ttk.Button(r1, text="下载 SmolVLA 模型", command=self.on_download_smolvla)
        self.btn_dl_model.pack(side="left", padx=2)
        self.btn_chk_status = ttk.Button(r1, text="查询 LeRobot/模型状态", command=self.on_check_lerobot_status)
        self.btn_chk_status.pack(side="left", padx=2)
        ttk.Label(r1, text="高层任务指令:").pack(side="left", padx=(6, 2))
        self.v_task = tk.StringVar(value="Put the red block into the left box, then put the blue block into the right box")
        ttk.Entry(r1, textvariable=self.v_task, width=42).pack(side="left", padx=2, fill="x", expand=True)

        # Section 11.3A: Explicit 4-Stage Control Separation Bar
        ctrl_f = ttk.LabelFrame(parent, text="2. 四级安全分离控制栏（① 加载模型/连接 → ② 允许控制 → ③ 开始执行 → ④ 停止/急停）")
        ctrl_f.pack(fill="x", pady=2)
        # V4.6: E-STOP group packed FIRST on the right of the whole section, so a long row can never push it off screen
        es = ttk.Frame(ctrl_f)
        es.pack(side="right", fill="y", padx=(4, 6), pady=3)
        self.btn_estop = tk.Button(
            es, text="■ 紧急急停 (E-STOP)", bg="#c62828", fg="white",
            font=("TkDefaultFont", 10, "bold"), command=self.on_estop, height=2
        )
        self.btn_estop.pack(side="top", fill="x")
        es2 = ttk.Frame(es)
        es2.pack(side="top", fill="x", pady=(2, 0))
        ttk.Button(es2, text="解除急停", command=self.on_clear_estop).pack(side="left")
        self.lamp_4 = StatusLamp(tk, ttk, es2, "④急停", 10)
        cb = ttk.Frame(ctrl_f)                                  # row 1: ① load / connect, ② allow control
        cb.pack(fill="x", padx=6, pady=(3, 1))
        ttk.Button(cb, text="Phase 0 审查", command=self.on_phase0_audit).pack(side="left", padx=(0, 4))
        self.btn_connect = ttk.Button(cb, text="① 加载 SmolVLA & 连接 Adapter", command=self.on_connect_and_load)
        self.btn_connect.pack(side="left", padx=3)
        self.lamp_1 = StatusLamp(tk, ttk, cb, "①", 10)
        self.btn_ipc = ttk.Button(cb, text="IPC 服务", command=self.on_toggle_ipc)
        self.btn_ipc.pack(side="left", padx=3)
        self.v_allow = tk.BooleanVar(value=False)
        self.chk_allow = ttk.Checkbutton(
            cb, text="② 允许机器人控制 (Allow Control)", variable=self.v_allow, command=self.on_toggle_allow
        )
        self.chk_allow.pack(side="left", padx=(8, 0))
        self.lamp_2 = StatusLamp(tk, ttk, cb, "②", 8)
        cb2 = ttk.Frame(ctrl_f)                                 # row 2: ③ run, ④ stop
        cb2.pack(fill="x", padx=6, pady=(1, 3))
        ttk.Button(cb2, text="③ 执行 SmolVLA 闭环", command=self.on_run_vla_loop).pack(side="left", padx=(0, 3))
        ttk.Button(cb2, text="③ 执行顶层 Agent 双路径任务", command=self.on_run_agent_task).pack(side="left", padx=3)
        self.lamp_3 = StatusLamp(tk, ttk, cb2, "③", 16)
        ttk.Button(cb2, text="④ 停止执行", command=self.on_stop_queue).pack(side="left", padx=3)

        # Section 11.3B: Dataset Recording, Fine-Tuning, Sim-to-Real Ladder & Benchmark
        rec_f = ttk.LabelFrame(parent, text="3. 示教采集与回放 (Phase 2) · 微调训练 (Phase 3) · Sim-to-Real 阶梯与三模式 Benchmark (Phase 5/6)")
        rec_f.pack(fill="x", pady=2)
        rb = ttk.Frame(rec_f)
        rb.pack(fill="x", padx=6, pady=3)

        self.btn_rec = ttk.Button(rb, text="● 开始示教录制", command=self.on_toggle_record)
        self.btn_rec.pack(side="left", padx=2)
        ttk.Button(rb, text="夹爪示教: 张开", command=lambda: self.recorder.set_teach_gripper_norm(1.0)).pack(side="left", padx=2)
        ttk.Button(rb, text="夹爪示教: 夹紧", command=lambda: self.recorder.set_teach_gripper_norm(0.0)).pack(side="left", padx=2)
        ttk.Label(rb, text="Episode#:").pack(side="left", padx=(6, 1))
        self.v_ep_idx = tk.StringVar(value="0")
        ttk.Spinbox(rb, from_=0, to=999, textvariable=self.v_ep_idx, width=4).pack(side="left")
        ttk.Button(rb, text="回放 Episode", command=self.on_replay_episode).pack(side="left", padx=2)
        ttk.Button(rb, text="删除 Episode", command=self.on_delete_episode).pack(side="left", padx=2)
        ttk.Button(rb, text="快速微调 (内置 NERO VLA)", command=self.on_finetune).pack(side="left", padx=(8, 2))
        ttk.Button(rb, text="Sim-to-Real 6级验证", command=self.on_sim2real_ladder).pack(side="left", padx=4)
        ttk.Button(rb, text="运行 A/B/C 对比 Benchmark", command=self.on_run_benchmark).pack(side="left", padx=2)

        # V4.4: real LeRobot fine-tuning of SmolVLA on the EMBODIED_AI drive (export -> lerobot-train -> use result)
        rb2 = ttk.Frame(rec_f)
        rb2.pack(fill="x", padx=6, pady=(0, 3))
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

        # Section 11.2: Live Status
        st_f = ttk.LabelFrame(parent, text="4. 实时状态（8维本体状态 + 双相机同步 + 数据集/模型状态 + 安全监督器）")
        st_f.pack(fill="x", pady=2)
        self.v_obs_stat = tk.StringVar(value="等待读取观测...")
        JC.fixed_label(
            tk, st_f, 3, width=1160, textvariable=self.v_obs_stat,
            font=("TkFixedFont", 9), wraplength=1150, pack={"fill": "x", "padx": 6, "pady": 2}
        )

        # V4.6.1: the 「Agentic实机任务」 live monitors (main + wrist camera), mirrored here; log to their right
        bottom = ttk.Frame(parent)
        bottom.pack(fill="both", expand=True, pady=(3, 0))
        mons = ttk.Frame(bottom)
        mons.pack(side="left", anchor="n")
        logcol = ttk.Frame(bottom)
        logcol.pack(side="left", fill="both", expand=True)
        self.monitor = None
        if self.rct_tab is not None and hasattr(self.rct_tab, "_draw_main"):
            try:
                import NERO_V365_Real_Cup_Task as RCT
                self.monitor = RCT.CameraMonitorMirror(tk, ttk, mons, self.rct_tab, yolo_parent=logcol)
            except Exception as exc:
                ttk.Label(mons, text=f"相机监控不可用：{exc}").pack(side="left", anchor="n")
        else:
            ttk.Label(mons, text="相机监控：「Agentic实机任务」页不可用", width=30).pack(side="left", anchor="n")
        # Log panel
        lf = ttk.Frame(logcol)
        lf.pack(side="top", fill="both", expand=True)
        self.txt = tk.Text(lf, height=12, wrap="word", font=("TkFixedFont", 9))
        self.txt.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(lf, command=self.txt.yview)
        sb.pack(side="right", fill="y")
        self.txt.configure(yscrollcommand=sb.set)

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

    def on_toggle_record(self) -> None:
        ds_name = self.v_ds_name.get().strip() or "nero_v4_demo"
        if self.dataset.dataset_name != ds_name:
            self.dataset = NEROLeRobotDataset(dataset_name=ds_name)
            self.recorder.dataset = self.dataset
        if not self.recorder.is_recording:
            self.recorder.start_recording(self.v_task.get().strip(), split="train")
            self.btn_rec.configure(text="■ 停止并保存 Episode")
        else:
            res = self.recorder.stop_and_save(episode_success=True)
            self.btn_rec.configure(text="● 开始示教录制")
            if res.get("ok"):
                self.v_ep_idx.set(str(res["episode"]["episode_index"]))

    def on_replay_episode(self) -> None:
        try:
            idx = int(self.v_ep_idx.get())
        except ValueError:
            return
        self.arbiter.request_mode("teleop", owner_tag="ui_replay")
        self.arbiter.set_allow_control(True, reason="Episode Replay")
        self.v_mode.set(self.arbiter.mode)
        self.v_allow.set(True)
        threading.Thread(target=lambda: self.recorder.replay_episode(idx, caller_mode="teleop"), daemon=True).start()

    def on_delete_episode(self) -> None:
        try:
            idx = int(self.v_ep_idx.get())
        except ValueError:
            return
        if self.dataset.delete_episode(idx):
            self._log(f"[INFO] [V4.2 DATASET] 已删除 Episode #{idx}")

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
            elif kind == "drive":
                self.v_drive.set(str(val))
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
                if not found:   # V4.5: closing the worker can wait up to 6 s -> not on the Tk main thread
                    threading.Thread(target=self.vla_policy.close_worker, daemon=True).start()
                threading.Thread(target=lambda: self._drive_refresh_job(probe=False), daemon=True).start()
        if self.visible_fn():
            self._refresh_status()
            if self.monitor is not None:
                self.monitor.refresh()
        self._schedule()

    def _refresh_status(self) -> None:
        obs = self.obs_bridge.get_synchronized_observation()
        q_deg = [round(math.degrees(float(v)), 1) for v in obs["robot_state"]]
        arb = self.arbiter.status()
        ex = self.executor.status()
        n_eps = len(self.dataset.list_episodes())
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
