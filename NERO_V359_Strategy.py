#!/usr/bin/env python3
"""NERO V3.59 strategy profile, baseline-equivalent control knobs, LLM skill cards, and multi-point hover dry-run.

Key design guarantees:
1. 100% Baseline Equivalence:
   Every default parameter in BASELINE_KNOBS (and config/strategies/egg_grasp_v345_baseline.json) is extracted
   directly from the verified V2.7 / V2.7.2 / V2.9 / V2.16 / V3.02 / V3.04 / V3.40 / V3.41 / V3.48 codebase.
   Applying the default profile reproduces the original candidate scores, gripper openings, wrist reorientation
   penalties, joint-limit margin buckets, and slot ordering with zero numerical deviation.
2. Self-Describing Skill Cards (skill_card):
   Each strategy JSON file embeds structured natural-language metadata (name, when_to_use, when_not_to_use,
   behavior_summary, scene_tags, sim_benchmark) so LLM/VLM agents can inspect available skills in observe_scene
   and select the right strategy for the current scene.
3. Gripper-Free Multi-Point Hover Dry-Run:
   Plans and executes safe point-to-point hover sequences (Target A Hover -> Safe Transit Lift -> Target B Hover
   -> ... -> Return to Start) with 3D safety fence verification at every waypoint, allowing full pick-and-place
   motion verification before a physical gripper is attached.
"""

import copy
import glob
import json
import math
import os
import re
import time

VERSION = "V3.61"
# V3.61 COLLAPSIBLE SKILL TAB + CODES AFTER SKILL NAMES + STACK CUPS SKILL (CS)
# V3.60 BASELINE-EXACT DEFAULTS + SKILL CODES + READ-ONLY RELEASE FILES + SIM ENV INJECTION + SIM TAB
#   * defaults restore V3.58 real behaviour (hover 100 mm, table order + trusted first)
#   * every skill has short_name (English) + 2-letter code (EB / DP / HD / QH ...)
#   * nothing inside the release folder is written at run time: the active choice lives in logs/,
#     user presets in ~/Downloads/cameracalib/strategies (NERO_V360_USER_STRATEGY_DIR)
#   * simulation: the 6 knobs the backend already reads from the environment are injected (only when they differ
#     from the baseline); every knob carries a "wired" tag (sim_env / real / none) shown in the GUI
HERE = os.path.dirname(os.path.realpath(__file__))
STRATEGY_DIR = os.environ.get("NERO_V359_STRATEGY_DIR", os.path.join(HERE, "config", "strategies"))   # shipped, read-only
USER_STRATEGY_DIR = os.environ.get("NERO_V360_USER_STRATEGY_DIR") or os.path.join(
    os.environ.get("NERO_CALIB_DIR") or os.path.expanduser("~/Downloads/cameracalib"), "strategies")
BASELINE_SKILL_ID = "egg_grasp_v345_baseline"
ACTIVE_STRATEGY_FILE = os.environ.get("NERO_V360_ACTIVE_STRATEGY_FILE") or os.path.join(
    HERE, "logs", "v360_active_strategy.json")
SKILL_CODE_RE = re.compile(r"^[A-Z]{2}$")

# --------------------------------------------------------------------------- 5 knob groups & 45 exact baseline specs (V3.60: count fixed)
KNOB_GROUPS = [
    ("grasp_scoring", "① 候选抓取方向与防碰打分 (V2.7 + V3.04)"),
    ("gripper_approach", "② 夹爪预张开裕量与两段慢速下探 (V2.7 + V3.04)"),
    ("ik_posture", "③ 逆运动学姿态筛选与手腕翻转权重 (V2.7.2 + V3.41)"),
    ("placement_recovery", "④ 蛋托对齐入槽与邻蛋防扰动门限 (V2.9 + V3.40)"),
    ("hover_dry_run", "⑤ 实机悬停复核与无夹爪多点预演 (Dry-Run)"),
    # V3.61 STACK CUPS: rim-pinch cup stacking (values from NERO_V361_Cup_Stack offline study)
    ("cup_stacking", "⑥ 叠杯子：捏杯沿抓取 · 运送 · 对中松手 (V3.61 离线试验)"),
]

SLOT_FILL_ORDERS = ("numeric_1_to_8", "nearest_to_egg", "far_to_near")
# V3.60: table_trusted_first = V3.58 _selected_hover_target (first trusted row in table order, else first row)
TARGET_ORDER_MODES = ("table_trusted_first", "confidence_then_nearest", "nearest_first", "clearance_first",
                      "table_row_order")

KNOB_SPECS = [
    # --- Group 1: Grasp candidate generation & scoring (Unified_Backend.py:15269-15371) ---
    {
        "key": "horiz_proj_min_strength",
        "group": "grasp_scoring",
        "label": "长轴平躺判定阈值",
        "type": "float",
        "default": 0.25,
        "min": 0.10,
        "max": 0.60,
        "unit": "比例",
        "desc": "鸡蛋长轴水平投影 >= 该值时沿短轴夹取，否则判为近竖立蛋改用切向方向 (V2.7 原值 0.25)",
    },
    {
        "key": "wall_near_margin_mm",
        "group": "grasp_scoring",
        "label": "贴边向心微移触发距离",
        "type": "float",
        "default": 30.0,
        "min": 10.0,
        "max": 50.0,
        "unit": "mm",
        "desc": "目标距容器安全内圈 < 该值时尝试 [0, 4, 6] mm 向心微移，否则尝试 [0, 4] mm (V2.7 原值 30 mm)",
    },
    {
        "key": "w_basket_clearance",
        "group": "grasp_scoring",
        "label": "容器内壁净空加分权重",
        "type": "float",
        "default": 1.0,
        "min": 0.10,
        "max": 3.00,
        "unit": "分/mm",
        "desc": "指垫四角相对容器内壁每 1 mm 净空的加分系数 (V2.7 原式 clearance_m * 1000.0 = 1.0 分/mm)",
    },
    {
        "key": "w_yaw_rank_penalty",
        "group": "grasp_scoring",
        "label": "偏离短轴档位扣分权重",
        "type": "float",
        "default": 4.0,
        "min": 0.50,
        "max": 12.0,
        "unit": "分/档",
        "desc": "抓取角每偏离鸡蛋短轴 1 档(15°)的扣分 (V2.7 原式 -yaw_rank * 4.0)",
    },
    {
        "key": "w_center_offset_penalty",
        "group": "grasp_scoring",
        "label": "抓点向心微移扣分权重",
        "type": "float",
        "default": 0.60,
        "min": 0.10,
        "max": 2.50,
        "unit": "分/mm",
        "desc": "抓取中心向内微移每 1 mm 的扣分系数 (V2.7 原式 -off_m * 1000 * 0.60)",
    },
    {
        "key": "neighbor_safe_clearance_mm",
        "group": "grasp_scoring",
        "label": "邻蛋防碰安全净空阈值",
        "type": "float",
        "default": 3.0,
        "min": 1.0,
        "max": 10.0,
        "unit": "mm",
        "desc": "指垫与相邻鸡蛋椭圆净空 < 该值时触发防碰扣分 (V3.04 原值 3.0 mm)",
    },
    {
        "key": "w_neighbor_intrusion",
        "group": "grasp_scoring",
        "label": "邻蛋侵入扣分斜率",
        "type": "float",
        "default": 0.80,
        "min": 0.20,
        "max": 3.00,
        "unit": "分/mm",
        "desc": "指垫侵入邻蛋安全区每 1 mm 的扣分系数 (V3.04 原式 0.8 * (nb_clear - 3.0))",
    },
    {
        "key": "neighbor_penalty_cap",
        "group": "grasp_scoring",
        "label": "邻蛋防碰最大扣分下限",
        "type": "float",
        "default": -10.0,
        "min": -25.0,
        "max": -2.0,
        "unit": "分",
        "desc": "邻蛋扣分封顶值，保证只在 ±30° 短轴族内重排、绝不翻去夹长轴 (V3.04 原值 -10.0)",
    },
    {
        "key": "target_order_mode",
        "group": "grasp_scoring",
        "label": "多目标默认巡检/抓取顺序",
        "type": "enum",
        "default": "table_trusted_first",
        "choices": list(TARGET_ORDER_MODES),
        "unit": "模式",
        "desc": "未手动选中行时的目标顺序：table_trusted_first=表格顺序+可信优先（V3.58 原逻辑，默认）/ "
                "可信度优先+离基座最近 / 最近优先 / 空旷度优先 / 表格原序",
    },
    # --- Group 2: Gripper aperture & two-stage approach (Unified_Backend.py:4589-4665, 9374-9378, 15085-15093) ---
    {
        "key": "egg_open_margin_mm",
        "group": "gripper_approach",
        "label": "鸡蛋预张开总裕量",
        "type": "float",
        "default": 16.0,
        "min": 8.0,
        "max": 26.0,
        "unit": "mm",
        "desc": "鸡蛋抓取前夹爪目标张开量 = 实测短轴宽 + 该裕量 (V3.04 原值 16.0 mm，每侧约 8 mm)",
    },
    {
        "key": "egg_open_band_lo_mm",
        "group": "gripper_approach",
        "label": "鸡蛋张开合格带下偏",
        "type": "float",
        "default": -4.0,
        "min": -8.0,
        "max": -1.0,
        "unit": "mm",
        "desc": "相对目标张开量的合格下限偏移 (V3.04 原式 width + (16 - 4) = width + 12 mm)",
    },
    {
        "key": "egg_open_band_hi_mm",
        "group": "gripper_approach",
        "label": "鸡蛋张开合格带上偏",
        "type": "float",
        "default": 6.0,
        "min": 2.0,
        "max": 12.0,
        "unit": "mm",
        "desc": "相对目标张开量的合格上限偏移 (V3.04 原式 width + (16 + 6) = width + 22 mm)",
    },
    {
        "key": "sphere_open_margin_mm",
        "group": "gripper_approach",
        "label": "球体预张开目标裕量",
        "type": "float",
        "default": 20.0,
        "min": 12.0,
        "max": 28.0,
        "unit": "mm",
        "desc": "球形物体目标张开裕量 (V2.7 原值 +20.0 mm，合格带 +16~+28 mm)",
    },
    {
        "key": "box_open_margin_mm",
        "group": "gripper_approach",
        "label": "方块预张开目标裕量",
        "type": "float",
        "default": 15.0,
        "min": 8.0,
        "max": 24.0,
        "unit": "mm",
        "desc": "方形/积木目标张开裕量 (V2.7 原值 +15.0 mm，合格带 +13~+21 mm)",
    },
    {
        "key": "poke_retry_bonus_mm",
        "group": "gripper_approach",
        "label": "碰歪目标重试加宽量",
        "type": "float",
        "default": 8.0,
        "min": 2.0,
        "max": 16.0,
        "unit": "mm",
        "desc": "检测到下探推歪物体后下一次重试额外张开的宽度 (V2.4.1 原值 +8.0 mm)",
    },
    {
        "key": "safety_retry_bonus_mm",
        "group": "gripper_approach",
        "label": "安全停机重试加宽量",
        "type": "float",
        "default": 15.0,
        "min": 6.0,
        "max": 22.0,
        "unit": "mm",
        "desc": "触发预抓取安全急停恢复后额外张开的宽度 (V2.4.1 原值 +15.0 mm)",
    },
    {
        "key": "pregrasp_hover_height_mm",
        "group": "gripper_approach",
        "label": "仿真预抓取悬停高度",
        "type": "float",
        "default": 120.0,
        "min": 40.0,
        "max": 160.0,
        "unit": "mm",
        "desc": "仿真抓取点正上方预抓取悬停高度 (V2.7 原值 V27_PREGRASP_HEIGHT_M = 120.0 mm)",
    },
    {
        "key": "grasp_z_offset_mm",
        "group": "gripper_approach",
        "label": "抓取点相对物心 Z 偏移",
        "type": "float",
        "default": 28.0,
        "min": 18.0,
        "max": 40.0,
        "unit": "mm",
        "desc": "夹持中心相对物体基准高度偏移 (V2.7 原值 V27_DEFAULT_GRASP_Z_OFFSET_M = 28.0 mm)",
    },
    {
        "key": "grasp_z_settle_comp_mm",
        "group": "gripper_approach",
        "label": "末端重力沉降 Z 补偿",
        "type": "float",
        "default": 2.5,
        "min": 0.0,
        "max": 6.0,
        "unit": "mm",
        "desc": "补偿机械臂到位后 2.5~2.8 mm 的重力下沉 (V3.04 原值 V304_GRASP_Z_COMP_M = 2.5 mm)",
    },
    {
        "key": "final_approach_dist_mm",
        "group": "gripper_approach",
        "label": "末段慢速滑入距离",
        "type": "float",
        "default": 20.0,
        "min": 8.0,
        "max": 45.0,
        "unit": "mm",
        "desc": "下探最后贴近鸡蛋的慢速段长度 (V3.04 原值 NERO_V304_FINAL_APPROACH_M = 20.0 mm)",
    },
    {
        "key": "final_approach_slow_factor",
        "group": "gripper_approach",
        "label": "末段慢速时间拉长倍率",
        "type": "float",
        "default": 3.5,
        "min": 1.0,
        "max": 6.0,
        "unit": "倍",
        "desc": "最后 20 mm 轨迹耗时放大倍数，让指垫轻柔滑过蛋面 (V3.04 原值 3.5 倍)",
    },
    # --- Group 3: IK & Posture selection (Unified_Backend.py:131-136, 10325-10336, 15409-15458) ---
    {
        "key": "limit_margin_required_deg",
        "group": "ik_posture",
        "label": "关节限位首选安全裕量",
        "type": "float",
        "default": 8.0,
        "min": 4.0,
        "max": 18.0,
        "unit": "deg",
        "desc": "IK 筛选首选全部 7 轴离限位 >= 该角度的解，预留垂直入槽余量 (V3.41 原值 8.0°)",
    },
    {
        "key": "limit_margin_relaxed_deg",
        "group": "ik_posture",
        "label": "关节限位放宽保底裕量",
        "type": "float",
        "default": 4.0,
        "min": 1.5,
        "max": 8.0,
        "unit": "deg",
        "desc": "无满足首选裕量的解时放宽到的最小限位距离 (V3.41 原值 4.0°)",
    },
    {
        "key": "limit_margin_bucket_deg",
        "group": "ik_posture",
        "label": "关节限位裕量分桶步长",
        "type": "float",
        "default": 4.0,
        "min": 2.0,
        "max": 8.0,
        "unit": "deg",
        "desc": "按 int(min(margin, 20°) // bucket) 分桶优先排序 (V3.41 原值 4.0°)",
    },
    {
        "key": "wrist_weight_j1_j3",
        "group": "ik_posture",
        "label": "翻转时 J1~J3 大臂惩罚权重",
        "type": "float",
        "default": 8.0,
        "min": 2.0,
        "max": 16.0,
        "unit": "权重",
        "desc": "手腕立蛋时对 J1/J2/J3 肩部大关节移动的惩罚系数 (V2.7.2 原值 8.0)",
    },
    {
        "key": "wrist_weight_j4",
        "group": "ik_posture",
        "label": "翻转时 J4 手肘惩罚权重",
        "type": "float",
        "default": 4.0,
        "min": 1.0,
        "max": 10.0,
        "unit": "权重",
        "desc": "手腕立蛋时对 J4 肘关节移动的惩罚系数 (V2.7.2 原值 4.0)",
    },
    {
        "key": "wrist_weight_j5",
        "group": "ik_posture",
        "label": "翻转时 J5 前臂惩罚权重",
        "type": "float",
        "default": 3.0,
        "min": 1.0,
        "max": 8.0,
        "unit": "权重",
        "desc": "手腕立蛋时对 J5 前臂旋转的惩罚系数 (V2.7.2 原值 3.0)",
    },
    {
        "key": "wrist_weight_j6_j7",
        "group": "ik_posture",
        "label": "翻转时 J6~J7 腕部权重",
        "type": "float",
        "default": 1.0,
        "min": 0.5,
        "max": 4.0,
        "unit": "权重",
        "desc": "手腕立蛋时对 J6/J7 远端手腕关节的权重，越低越优先动腕部 (V2.7.2 原值 1.0)",
    },
    {
        "key": "wrist_proximal_sum_limit_deg",
        "group": "ik_posture",
        "label": "翻转时 J1~J5 总跨度上限",
        "type": "float",
        "default": 22.0,
        "min": 12.0,
        "max": 40.0,
        "unit": "deg",
        "desc": "手腕翻转单步规划允许的 J1~J5 角度变化总和硬上限，防甩肘 (V2.7.2 原值 22.0°)",
    },
    # --- Group 4: Slot placement & fault recovery (Unified_Backend.py:7495, 15087-15088, 15595-15609) ---
    {
        "key": "slot_fill_order",
        "group": "placement_recovery",
        "label": "蛋托空孔装填顺序",
        "type": "enum",
        "default": "numeric_1_to_8",
        "choices": list(SLOT_FILL_ORDERS),
        "unit": "顺序",
        "desc": "蛋托空槽默认装填顺序 (V3.04 原逻辑 numeric_1_to_8 即严格按 slot_1 -> slot_8)",
    },
    {
        "key": "slot_entry_above_release_mm",
        "group": "placement_recovery",
        "label": "孔位释放点上方对齐高度",
        "type": "float",
        "default": 55.0,
        "min": 35.0,
        "max": 90.0,
        "unit": "mm",
        "desc": "垂直入槽前在释放点正上方的对齐悬停余量 (V2.7.1 原值 55.0 mm)",
    },
    {
        "key": "slot_capture_depth_mm",
        "group": "placement_recovery",
        "label": "鸡蛋入孔捕获深度",
        "type": "float",
        "default": 8.0,
        "min": 3.0,
        "max": 15.0,
        "unit": "mm",
        "desc": "鸡蛋底部深入蛋托孔位沿口下方的深度，到位即松爪避免夹爪撞孔壁 (V2.7.1 原值 8.0 mm)",
    },
    {
        "key": "slot_xy_tol_mm",
        "group": "placement_recovery",
        "label": "孔位水平对齐合格阈值",
        "type": "float",
        "default": 7.0,
        "min": 3.0,
        "max": 12.0,
        "unit": "mm",
        "desc": "入槽前鸡蛋中心与孔心水平偏差合格门限 (V2.7.1 原值 7.0 mm)",
    },
    {
        "key": "slot_axis_tol_deg",
        "group": "placement_recovery",
        "label": "鸡蛋竖直轴向合格角",
        "type": "float",
        "default": 10.0,
        "min": 5.0,
        "max": 15.0,
        "unit": "deg",
        "desc": "入槽前鸡蛋长轴与竖直方向最大允许夹角 (V2.7.1 原值 10.0°)",
    },
    {
        "key": "release_stage1_open_mm",
        "group": "placement_recovery",
        "label": "入槽首段微张松开量",
        "type": "float",
        "default": 10.0,
        "min": 5.0,
        "max": 20.0,
        "unit": "mm",
        "desc": "在孔位内先微张该宽度让鸡蛋落座，退离孔口后再全张至 90 mm (V2.7.1 原值 10.0 mm)",
    },
    {
        "key": "bystander_warn_mm",
        "group": "placement_recovery",
        "label": "邻蛋被扰动警告门限",
        "type": "float",
        "default": 3.0,
        "min": 1.5,
        "max": 6.0,
        "unit": "mm",
        "desc": "抓取过程中旁观邻蛋位移超过该值记警告 (V2.16 原值 3.0 mm)",
    },
    {
        "key": "bystander_abort_mm",
        "group": "placement_recovery",
        "label": "邻蛋被扰动中止门限",
        "type": "float",
        "default": 8.0,
        "min": 5.0,
        "max": 15.0,
        "unit": "mm",
        "desc": "旁观邻蛋位移超过该值则在闭合前中止并重观察 (V2.16 原值 8.0 mm)",
    },
    {
        "key": "object_stale_pos_mm",
        "group": "placement_recovery",
        "label": "目标位移作废重算阈值",
        "type": "float",
        "default": 8.0,
        "min": 4.0,
        "max": 15.0,
        "unit": "mm",
        "desc": "目标鸡蛋相对规划时移动超过该距离则作废缓存重新规划 (V2.7 原值 8.0 mm)",
    },
    # --- Group 5: Real-robot hover verification & multi-point Dry-Run (NERO_V325_Camera.py + V3.59) ---
    {
        "key": "cam_hover_verify_height_mm",
        "group": "hover_dry_run",
        "label": "实机目标悬停复核高度",
        "type": "float",
        "default": 100.0,
        "min": 40.0,
        "max": 250.0,
        "unit": "mm",
        "desc": "实机「悬停复核」与多点 Dry-Run 时 TCP 在目标定位点（物心）正上方的高度；只在「应用/选择技能」时写入"
                "悬停输入框，腕部 USB 相机微调对齐与准星不跟随 (V3.48 原值 HOVER_DEFAULT_MM = 100 mm)",
    },
    {
        "key": "cam_hover_soft_limit_deg",
        "group": "hover_dry_run",
        "label": "实机悬停关节软限位裕量",
        "type": "float",
        "default": 3.0,
        "min": 1.0,
        "max": 10.0,
        "unit": "deg",
        "desc": "实机悬停 IK 求解时各关节距硬件极值的内缩保护角 (V3.48 原值 3.0°)",
    },
    {
        "key": "cam_hover_max_speed_pct",
        "group": "hover_dry_run",
        "label": "实机悬停/预演最高限速",
        "type": "float",
        "default": 10.0,
        "min": 2.0,
        "max": 30.0,
        "unit": "%",
        "desc": "实机悬停复核与多点悬停预演 (Dry-Run) 的最大允许速度百分比 (V3.48 原值 10%)",
    },
    {
        "key": "dry_run_transit_z_lift_mm",
        "group": "hover_dry_run",
        "label": "Dry-Run 跨点安全抬升量",
        "type": "float",
        "default": 40.0,
        "min": 0.0,
        "max": 120.0,
        "unit": "mm",
        "desc": "从一个悬停点移往下一个悬停点时，在两点较高者基础上额外抬升的安全过渡高度",
    },
    {
        "key": "dry_run_dwell_s",
        "group": "hover_dry_run",
        "label": "Dry-Run 悬停点驻留时间",
        "type": "float",
        "default": 1.2,
        "min": 0.1,
        "max": 5.0,
        "unit": "s",
        "desc": "多点悬停预演到达每个目标正上方后的停顿观察/腕部测量时间",
    },
    {
        "key": "dry_run_wrist_refine",
        "group": "hover_dry_run",
        "label": "Dry-Run 到位自动腕部测偏",
        "type": "bool",
        "default": True,
        "unit": "开关",
        "desc": "悬停到位后若腕部相机已连接并完成标定，自动拍一帧计算相对 TCP 偏差 (Δx, Δy, Δψ)",
    },
    {
        "key": "dry_run_return_home",
        "group": "hover_dry_run",
        "label": "Dry-Run 结束后回出发姿态",
        "type": "bool",
        "default": True,
        "unit": "开关",
        "desc": "多点悬停预演全部走完后，自动退回预演开始前的初始关节姿态",
    },
    # --- Group 6 (V3.61): stack_cups rim-pinch skill; defaults = NERO_V361_Cup_Stack study
    #     (config/cup_stack_study_v361.json: MuJoCo release sweep + strict 6-DoF URDF IK; not measured on hardware)
    {
        "key": "cup_rim_pinch_depth_mm", "group": "cup_stacking", "label": "捏杯沿深度", "type": "float",
        "default": 25.0, "min": 10.0, "max": 45.0, "unit": "mm",
        "desc": "指垫从杯口往下捏住杯壁的深度（一指在杯内、一指在杯外）；越深 TCP 越低、可达裕量越大（试验：25 mm 时最小限位裕量 12.6°，35 mm 时 16°），但受手指长度限制",
    },
    {
        "key": "cup_grasp_rim_angle_deg", "group": "cup_stacking", "label": "抓取杯沿方位", "type": "float",
        "default": 40.0, "min": -180.0, "max": 180.0, "unit": "deg",
        "desc": "在被抓杯子上捏哪一点：相对「杯心→机械臂基座」方向逆时针的角度（试验最优 +40°，24 个方位中 21 个可达）",
    },
    {
        "key": "cup_place_rim_angle_deg", "group": "cup_stacking", "label": "放置杯沿方位", "type": "float",
        "default": -55.0, "min": -180.0, "max": 180.0, "unit": "deg",
        "desc": "放到目标杯上方时被抓点所在方位（杯子轴对称，运送途中可绕杯轴转），相对目标杯「杯心→基座」方向（试验最优 −55°）",
    },
    {
        "key": "cup_pregrasp_height_mm", "group": "cup_stacking", "label": "抓杯前悬停高度", "type": "float",
        "default": 60.0, "min": 30.0, "max": 120.0, "unit": "mm",
        "desc": "TCP 在捏杯点正上方停稳的高度，再竖直下探捏住杯沿",
    },
    {
        "key": "cup_transport_clearance_mm", "group": "cup_stacking", "label": "运送越杯余量", "type": "float",
        "default": 20.0, "min": 10.0, "max": 60.0, "unit": "mm",
        "desc": "运送时被抓杯的杯底高出另一只杯口的距离；NERO 在 |x|≈0.35 m 处竖直向下最高约 0.245 m，余量越大越难够到",
    },
    {
        "key": "cup_release_gap_mm", "group": "cup_stacking", "label": "松手高度（相对套好位置）", "type": "float",
        "default": 85.0, "min": 0.0, "max": 100.0, "unit": "mm",
        "desc": "杯子下降到「完全套好」位置上方多少时松手（85 mm = 杯底刚进杯口约 11 mm）；试验：85 mm 时偏差 ≤15 mm、倾斜 ≤6° 全部套到底，低位松手易卡在半途",
    },
    {
        "key": "cup_release_xy_tol_mm", "group": "cup_stacking", "label": "松手前对中容差", "type": "float",
        "default": 7.5, "min": 2.0, "max": 15.0, "unit": "mm",
        "desc": "松手前两杯轴心水平偏差的允许上限（试验找到 15 mm，取 0.5 安全系数）；超过则先修正再松手",
    },
    {
        "key": "cup_release_tilt_tol_deg", "group": "cup_stacking", "label": "松手前倾斜容差", "type": "float",
        "default": 3.0, "min": 1.0, "max": 6.0, "unit": "deg",
        "desc": "被抓杯轴线相对竖直的允许倾斜（试验在 6° 内全部成功，取一半）",
    },
    {
        "key": "cup_nested_height_mm", "group": "cup_stacking", "label": "套好后杯底高差", "type": "float",
        "default": 25.3, "min": 10.0, "max": 40.0, "unit": "mm",
        "desc": "两杯完全套好时上杯杯底比下杯杯底高多少（仿真模型实测 25.3 mm；真杯子请实测后填写）",
    },
]

# V3.60 WIRING: where a knob really acts in this version.
#   sim_env : injected into the simulation backend through the environment variable it already reads
#   real    : used by the real-robot Camera tab (hover verify target / multi-point dry-run)
#   none    : formula parameterised and equivalence-tested, but nothing reads it yet (kept for later versions)
SIM_ENV_KNOBS = {   # knob -> (environment variable, backend default string, scale knob->env)
    "egg_open_margin_mm": ("NERO_V304_EGG_APERTURE_MARGIN_MM", "16", 1.0),
    "final_approach_dist_mm": ("NERO_V304_FINAL_APPROACH_M", "0.020", 0.001),
    "final_approach_slow_factor": ("NERO_V304_FINAL_APPROACH_SLOW", "3.5", 1.0),
    "grasp_z_settle_comp_mm": ("NERO_V304_GRASP_Z_COMP_M", "0.0025", 0.001),
    "limit_margin_required_deg": ("NERO_V341_LIMIT_MARGIN_DEG", "8", 1.0),
    "limit_margin_relaxed_deg": ("NERO_V341_LIMIT_MARGIN_MIN_DEG", "4", 1.0),
}
REAL_KNOBS = ("target_order_mode", "cam_hover_verify_height_mm", "cam_hover_soft_limit_deg", "cam_hover_max_speed_pct",
              "dry_run_transit_z_lift_mm", "dry_run_dwell_s", "dry_run_wrist_refine", "dry_run_return_home")
WIRING_LABELS = {"sim_env": "仿真✓", "real": "实机✓", "none": "未接入", "study": "离线试验✓"}
for _s360 in KNOB_SPECS:
    _s360["wired"] = "sim_env" if _s360["key"] in SIM_ENV_KNOBS else ("real" if _s360["key"] in REAL_KNOBS else "none")
    if _s360["group"] == "cup_stacking":          # V3.61: used by NERO_V361_Cup_Stack (offline study / planner)
        _s360["wired"] = "study"

KNOB_BY_KEY = {s["key"]: s for s in KNOB_SPECS}
BASELINE_KNOBS = {s["key"]: s["default"] for s in KNOB_SPECS}

BASELINE_SKILL_CARD = {
    "name": "抓鸡蛋入托标准基准技能 (V3.45 Baseline)",
    "when_to_use": (
        "默认通用抓鸡蛋与精细放置场景：桌面或篮筐内有 1~4 颗鸡蛋需抓起并垂直放入蛋托孔位，"
        "目标与邻蛋或容器壁存在常规间隙，追求稳健短轴夹取、手腕平稳立蛋与零碰撞入槽。"
    ),
    "when_not_to_use": (
        "当多颗鸡蛋紧密贴死且需要极窄张开角强行剥离、或仅在无夹爪状态下做高空快速巡检时，"
        "可切换对应的专用技能包。"
    ),
    "behavior_summary": (
        "100% 严格还原 V2.7~V3.45 仿真与实机验证通过的基准逻辑：夹爪按实测蛋宽 +16 mm 预张开"
        "（每侧约 8 mm 净空），下探最后 20 mm 以 3.5 倍慢速滑入；邻蛋净空 < 3 mm 时按 0.8 分/mm 扣分"
        "（封顶 -10 分）；IK 首选离限位 >= 8° 的解并按 8:4:3:1 权重压制大臂摆动；蛋托按 slot_1->slot_8 顺序装填。"
    ),
    "scene_tags": ["egg", "fragile", "egg_holder", "baseline", "short_axis_pinch", "sim2real_verified"],
    "sim_benchmark": {
        "trained_scene": "fill_egg_holder",
        "noise_level": None,
        "success_rate": None,          # V3.60: not measured yet (filled by the sim optimizer later)
        "avg_cycle_time_s": None,
        "notes": "V2.7 ~ V3.58 调好的基准参数；成功率 / 耗时尚未测量",
    },
}
BASELINE_SHORT_NAME, BASELINE_CODE = "Egg Baseline", "EB"


# --------------------------------------------------------------------------- validation & diff helpers
def coerce_knob_value(key, val):
    """Validate and clamp a single knob value according to KNOB_SPECS."""
    spec = KNOB_BY_KEY.get(key)
    if spec is None:
        return val
    t = spec["type"]
    if t == "bool":
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "on", "是")
        return bool(val)
    if t == "enum":
        s = str(val).strip()
        return s if s in spec["choices"] else spec["default"]
    try:
        x = float(val)
        if not math.isfinite(x):
            return spec["default"]
    except Exception:
        return spec["default"]
    lo, hi = float(spec["min"]), float(spec["max"])
    x = max(lo, min(hi, x))
    return int(round(x)) if t == "int" else round(x, 4)


def sanitize_knobs(raw_knobs=None):
    """Return a full dict of all knobs (len(KNOB_SPECS) = 45), filling missing keys from BASELINE_KNOBS."""
    out = dict(BASELINE_KNOBS)
    if isinstance(raw_knobs, dict):
        for k, v in raw_knobs.items():
            if k in KNOB_BY_KEY:
                out[k] = coerce_knob_value(k, v)
    return out


def sanitize_skill_card(raw_card=None, skill_id=BASELINE_SKILL_ID, knobs=None):
    """Ensure a complete skill_card structure with natural-language fields for LLM consumption."""
    base = copy.deepcopy(BASELINE_SKILL_CARD) if skill_id == BASELINE_SKILL_ID else {
        "name": f"自定义技能 ({skill_id})",
        "when_to_use": "适用于特定抓取或悬停预演场景（请填写自然语言触发条件，供 LLM 选策参考）。",
        "when_not_to_use": "不适用于超出该参数包假设的场景。",
        "behavior_summary": describe_diff_vs_baseline(knobs or BASELINE_KNOBS),
        "scene_tags": ["custom"],
        "sim_benchmark": {
            "trained_scene": "fill_egg_holder",
            "noise_level": "manual_or_sim",
            "success_rate": None,
            "avg_cycle_time_s": None,
            "notes": "",
        },
    }
    if isinstance(raw_card, dict):
        for k in ("name", "when_to_use", "when_not_to_use", "behavior_summary"):
            if raw_card.get(k) not in (None, ""):
                base[k] = str(raw_card[k]).strip()
        if isinstance(raw_card.get("scene_tags"), (list, tuple)):
            base["scene_tags"] = [str(x).strip() for x in raw_card["scene_tags"] if str(x).strip()]
        elif isinstance(raw_card.get("scene_tags"), str):
            base["scene_tags"] = [x.strip() for x in raw_card["scene_tags"].split(",") if x.strip()]
        if isinstance(raw_card.get("sim_benchmark"), dict):
            base["sim_benchmark"].update(raw_card["sim_benchmark"])
    return base


def diff_vs_baseline(knobs):
    """Return list of (key, spec, current_val, baseline_val) where current_val != baseline_val."""
    clean = sanitize_knobs(knobs)
    diffs = []
    for spec in KNOB_SPECS:
        k = spec["key"]
        v = clean[k]
        b = BASELINE_KNOBS[k]
        if isinstance(b, float) and not isinstance(b, bool):
            if abs(float(v) - float(b)) > 1e-6:
                diffs.append((k, spec, v, b))
        elif v != b:
            diffs.append((k, spec, v, b))
    return diffs


def is_baseline_equivalent(knobs):
    """True when every knob matches BASELINE_KNOBS (V3.60: the hover group counts too, because the real-robot
    hover height / target order now really act; V3.59 ignored group 5)."""
    return not diff_vs_baseline(knobs)


def describe_diff_vs_baseline(knobs):
    """Generate a concise Chinese natural-language summary comparing `knobs` against BASELINE_KNOBS."""
    diffs = diff_vs_baseline(knobs)
    if not diffs:
        return (
            f"100% 严格还原 V2.7~V3.45 调好抓鸡蛋基准逻辑（全部 {len(KNOB_SPECS)} 项参数与原版代码完全一致）。"
        )
    parts = []
    for _k, spec, v, b in diffs:
        u = spec.get("unit", "")
        u_str = f" {u}" if u and u not in ("模式", "顺序", "开关", "权重", "比例") else ""
        parts.append(f"{spec['label']}由 {b}{u_str} 调整为 {v}{u_str}")
    return f"相比抓鸡蛋基准 ({BASELINE_SKILL_ID}) 共修改 {len(diffs)} 项：" + "；".join(parts) + "。"


# --------------------------------------------------------------------------- 100% baseline-equivalent core formulas
def score_grasp_candidate(clearance_m, yaw_rank, offset_m, nb_clear_mm=999.0, knobs=None):
    """Exact parameterized version of _v27_make_grasp_candidates scoring (Unified_Backend.py:15336-15343).

    With default `knobs`, reduces identically to:
        score = (clearance_m * 1000.0) - yaw_rank * 4.0 - offset_m * 1000.0 * 0.60
        if nb_clear_mm < 3.0:
            score += max(-10.0, 0.8 * (nb_clear_mm - 3.0))
    """
    k = BASELINE_KNOBS if knobs is None else knobs
    w_b = float(k.get("w_basket_clearance", 1.0))
    w_y = float(k.get("w_yaw_rank_penalty", 4.0))
    w_o = float(k.get("w_center_offset_penalty", 0.60))
    tau_nb = float(k.get("neighbor_safe_clearance_mm", 3.0))
    w_nb = float(k.get("w_neighbor_intrusion", 0.80))
    cap_nb = float(k.get("neighbor_penalty_cap", -10.0))

    score = (float(clearance_m) * 1000.0 * w_b) - float(yaw_rank) * w_y - (float(offset_m) * 1000.0 * w_o)
    if float(nb_clear_mm) < tau_nb:
        score += max(cap_nb, w_nb * (float(nb_clear_mm) - tau_nb))
    return score


def compute_gripper_prior(object_name, effective_width_mm=50.0, shape="unknown", retry_bonus_mm=0.0,
                          actual_opening_mm=None, knobs=None, env_margin_override=None):
    """Exact parameterized version of _v24_gripper_prior (Unified_Backend.py:4574-4665).

    With default `knobs`, produces the exact same dictionary values as V2.7 + V3.04.
    """
    k = BASELINE_KNOBS if knobs is None else knobs
    width = float(effective_width_mm)
    bonus = float(retry_bonus_mm)
    name_s = str(object_name)

    if shape in ("sphere", "ellipsoid"):
        m_target = float(k.get("sphere_open_margin_mm", 20.0))
        base_min_margin = m_target - 4.0
        base_target_margin = m_target
        base_max_margin = m_target + 8.0
        envelope_allowance = 8.0
        policy = f"round/ellipsoid anti-poke prior: +{base_min_margin:.0f}/+{base_target_margin:.0f}/+{base_max_margin:.0f} mm total aperture margin"
    else:
        m_target = float(k.get("box_open_margin_mm", 15.0))
        base_min_margin = m_target - 2.0
        base_target_margin = m_target
        base_max_margin = m_target + 6.0
        envelope_allowance = 4.0
        policy = f"box anti-poke prior: +{base_min_margin:.0f}/+{base_target_margin:.0f}/+{base_max_margin:.0f} mm total aperture margin"

    poke_b = float(k.get("poke_retry_bonus_mm", 8.0))
    safe_b = float(k.get("safety_retry_bonus_mm", 15.0))
    rec_min = min(100.0, width + base_min_margin + bonus)
    rec_target = min(100.0, width + base_target_margin + bonus)
    rec_max = min(100.0, width + base_max_margin + bonus)

    result = {
        "effective_width_mm": round(width, 2),
        "recommended_opening_range_mm": [round(rec_min, 2), round(rec_max, 2)],
        "recommended_target_opening_mm": round(rec_target, 2),
        "recommended_total_clearance_mm": round(rec_target - width, 2),
        "estimated_pad_envelope_allowance_total_mm": round(envelope_allowance, 2),
        "estimated_free_clearance_at_target_mm": round(max(0.0, rec_target - width - envelope_allowance), 2),
        "adaptive_retry_bonus_mm": round(bonus, 2),
        "prior_basis": f"{policy}; +{poke_b:.0f} mm after detected poke; +{safe_b:.0f} mm after a pre-grasp safety-stop retry",
    }
    if actual_opening_mm is not None:
        opening = float(actual_opening_mm)
        result["current_opening_mm"] = round(opening, 2)
        result["geometric_clearance_total_mm"] = round(opening - width, 2)
        result["estimated_free_clearance_total_mm"] = round(opening - width - envelope_allowance, 2)
        result["estimated_free_clearance_per_side_mm"] = round((opening - width - envelope_allowance) / 2.0, 2)
        result["distance_to_recommended_target_mm"] = round(opening - rec_target, 2)

    if name_s.startswith("egg"):
        m_t = float(env_margin_override) if env_margin_override is not None else float(k.get("egg_open_margin_mm", 16.0))
        lo_off = float(k.get("egg_open_band_lo_mm", -4.0))
        hi_off = float(k.get("egg_open_band_hi_mm", 6.0))
        rec_min = min(100.0, width + max(8.0, m_t + lo_off) + bonus)
        rec_target = min(100.0, width + m_t + bonus)
        rec_max = min(100.0, width + m_t + hi_off + bonus)
        result.update({
            "recommended_opening_range_mm": [round(rec_min, 2), round(rec_max, 2)],
            "recommended_target_opening_mm": round(rec_target, 2),
            "recommended_total_clearance_mm": round(rec_target - width, 2),
            "estimated_free_clearance_at_target_mm": round(max(0.0, rec_target - width - envelope_allowance), 2),
            "aperture_ratio_target": round(rec_target / max(width, 1e-6), 3),
            "prior_basis": (
                f"V3.04 egg aperture prior: target = width + {m_t:.0f} mm "
                f"(about {m_t / 2:.0f} mm per side before pads); "
                f"band = width+{max(8.0, m_t + lo_off):.0f}..+{m_t + hi_off:.0f} mm"
            ),
        })
        if actual_opening_mm is not None:
            result["distance_to_recommended_target_mm"] = round(float(actual_opening_mm) - rec_target, 2)
    return result


def wrist_joint_score_weights(knobs=None):
    """Exact V272_JOINT_SCORE_WEIGHT mapping from `knobs` (defaults to {J1..3: 8, J4: 4, J5: 3, J6..7: 1})."""
    k = BASELINE_KNOBS if knobs is None else knobs
    w13 = float(k.get("wrist_weight_j1_j3", 8.0))
    w4 = float(k.get("wrist_weight_j4", 4.0))
    w5 = float(k.get("wrist_weight_j5", 3.0))
    w67 = float(k.get("wrist_weight_j6_j7", 1.0))
    return {
        "joint1": w13, "joint2": w13, "joint3": w13,
        "joint4": w4, "joint5": w5,
        "joint6": w67, "joint7": w67,
    }


def compute_wrist_reorient_score(joint_deltas_deg, knobs=None):
    """Exact parameterized version of _v272_wrist_score (Unified_Backend.py:15446)."""
    weights = wrist_joint_score_weights(knobs)
    return sum(float(weights.get(jn, 1.0)) * float(val) for jn, val in (joint_deltas_deg or {}).items())


def check_wrist_reorient_bounded(joint_deltas_deg, step_limits_deg=None, knobs=None):
    """Exact parameterized version of _v272_wrist_plan_ok (Unified_Backend.py:15450)."""
    if not joint_deltas_deg:
        return False
    lims = step_limits_deg or {
        "joint1": 4.0, "joint2": 4.0, "joint3": 4.0,
        "joint4": 8.0, "joint5": 10.0,
        "joint6": 25.0, "joint7": 25.0,
    }
    for name, lim in lims.items():
        if float(joint_deltas_deg.get(name, 0.0)) > float(lim):
            return False
    k = BASELINE_KNOBS if knobs is None else knobs
    prox_lim = float(k.get("wrist_proximal_sum_limit_deg", 22.0))
    proximal = sum(float(joint_deltas_deg.get(f"joint{i}", 0.0)) for i in range(1, 6))
    return proximal <= prox_lim


def order_empty_slots(slots_dict, ref_xy_m=None, knobs=None):
    """Order unoccupied holder slots according to `slot_fill_order` (default: numeric_1_to_8, matching V3.04)."""
    k = BASELINE_KNOBS if knobs is None else knobs
    mode = str(k.get("slot_fill_order", "numeric_1_to_8"))
    empty = [name for name, info in (slots_dict or {}).items() if not (info or {}).get("occupied")]

    def _num_key(s):
        try:
            return int(str(s).split("_")[-1])
        except Exception:
            return 999

    if mode == "nearest_to_egg" and ref_xy_m is not None:
        rx, ry = float(ref_xy_m[0]), float(ref_xy_m[1])
        return sorted(
            empty,
            key=lambda s: (
                math.hypot(
                    float(((slots_dict.get(s) or {}).get("center_xy_m") or (slots_dict.get(s) or {}).get("xyz_m") or [0, 0])[0]) - rx,
                    float(((slots_dict.get(s) or {}).get("center_xy_m") or (slots_dict.get(s) or {}).get("xyz_m") or [0, 0])[1]) - ry,
                ),
                _num_key(s),
            ),
        )
    if mode == "far_to_near":
        return sorted(
            empty,
            key=lambda s: (
                -math.hypot(
                    float(((slots_dict.get(s) or {}).get("center_xy_m") or (slots_dict.get(s) or {}).get("xyz_m") or [-0.4, 0])[0]),
                    float(((slots_dict.get(s) or {}).get("center_xy_m") or (slots_dict.get(s) or {}).get("xyz_m") or [-0.4, 0])[1]),
                ),
                _num_key(s),
            ),
        )
    return sorted(empty, key=_num_key)


def rank_detected_targets(targets, ref_xy_m=(0.0, 0.0), knobs=None):
    """Order detected visual targets (eggs, objects, slots) according to `target_order_mode`."""
    k = BASELINE_KNOBS if knobs is None else knobs
    mode = str(k.get("target_order_mode", "table_trusted_first"))
    valid = [t for t in (targets or []) if t.get("xyz_m") and t.get("status") != "rejected"]
    if mode == "table_row_order" or len(valid) <= 1:
        return list(valid)
    if mode == "table_trusted_first":          # V3.60: == V3.58 (trusted rows first, table order kept)
        return [t for t in valid if t.get("trusted")] + [t for t in valid if not t.get("trusted")]
    rx, ry = float(ref_xy_m[0]), float(ref_xy_m[1])

    def _dist(t):
        xyz = t.get("xyz_m") or (0.0, 0.0, 0.0)
        return math.hypot(float(xyz[0]) - rx, float(xyz[1]) - ry)

    def _min_nb(t):
        xyz = t.get("xyz_m") or (0.0, 0.0, 0.0)
        dmin = 999.0
        for o in valid:
            if o is t or o.get("det_id") == t.get("det_id"):
                continue
            oxyz = o.get("xyz_m") or (0.0, 0.0, 0.0)
            dmin = min(dmin, math.hypot(float(xyz[0]) - float(oxyz[0]), float(xyz[1]) - float(oxyz[1])))
        return dmin

    if mode == "nearest_first":
        return sorted(valid, key=lambda t: (not bool(t.get("trusted")), _dist(t)))
    if mode == "clearance_first":
        return sorted(valid, key=lambda t: (not bool(t.get("trusted")), -_min_nb(t), _dist(t)))
    # default: confidence_then_nearest
    return sorted(
        valid,
        key=lambda t: (
            not bool(t.get("trusted")),
            -round(float(t.get("confidence", 0.0)), 2),
            _dist(t),
        ),
    )


# --------------------------------------------------------------------------- StrategyProfile & file persistence
class StrategyProfile:
    """In-memory representation of a strategy JSON file (`skill_card` + `knobs`)."""

    def __init__(self, skill_id=BASELINE_SKILL_ID, skill_card=None, knobs=None, version=VERSION, saved="", path=None,
                 short_name=None, code=None):
        self.skill_id = str(skill_id or BASELINE_SKILL_ID).strip()
        # V3.60 SKILL CODES: short English name + 2-letter code (unique, used in logs / scores)
        if short_name is None and code is None and self.skill_id == BASELINE_SKILL_ID:
            short_name, code = BASELINE_SHORT_NAME, BASELINE_CODE
        self.short_name = str(short_name or "").strip()
        c360 = str(code or "").strip().upper()
        self.code = c360 if SKILL_CODE_RE.match(c360) else ""
        self.version = str(version or VERSION)
        self.knobs = sanitize_knobs(knobs)
        self.skill_card = sanitize_skill_card(skill_card, skill_id=self.skill_id, knobs=self.knobs)
        self.saved = str(saved or time.strftime("%Y-%m-%d %H:%M:%S"))
        self.path = path

    def get(self, key, default=None):
        if key in self.knobs:
            return self.knobs[key]
        return BASELINE_KNOBS.get(key, default)

    def set(self, key, val):
        if key not in KNOB_BY_KEY:
            raise KeyError(f"Unknown strategy knob: {key}")
        self.knobs[key] = coerce_knob_value(key, val)
        return self.knobs[key]

    def reset_to_baseline(self):
        self.skill_id = BASELINE_SKILL_ID
        self.short_name, self.code = BASELINE_SHORT_NAME, BASELINE_CODE
        self.knobs = dict(BASELINE_KNOBS)
        self.skill_card = copy.deepcopy(BASELINE_SKILL_CARD)
        return self

    def refresh_behavior_summary(self):
        self.skill_card["behavior_summary"] = describe_diff_vs_baseline(self.knobs)
        return self.skill_card["behavior_summary"]

    def to_dict(self):
        return {
            "skill_id": self.skill_id,
            "short_name": self.short_name,
            "code": self.code,
            "version": self.version,
            "saved": self.saved or time.strftime("%Y-%m-%d %H:%M:%S"),
            "is_baseline": is_baseline_equivalent(self.knobs),
            "skill_card": copy.deepcopy(self.skill_card),
            "knobs": dict(self.knobs),
        }

    @classmethod
    def from_dict(cls, d, path=None):
        if not isinstance(d, dict):
            return cls(path=path)
        sid = str(d.get("skill_id") or d.get("name") or BASELINE_SKILL_ID)
        knobs = sanitize_knobs(d.get("knobs") if isinstance(d.get("knobs"), dict) else d)
        card = sanitize_skill_card(d.get("skill_card"), skill_id=sid, knobs=knobs)
        return cls(
            skill_id=sid,
            skill_card=card,
            knobs=knobs,
            version=d.get("version", VERSION),
            saved=d.get("saved", ""),
            path=path,
            short_name=d.get("short_name"),
            code=d.get("code"),
        )

    def label(self):
        """V3.60: 'EB Egg Baseline' style tag for logs and scores."""
        return f"{self.code or '??'} {self.short_name or self.skill_id}"

    def llm_brief(self):
        """Compact summary dictionary for LLM `observe_scene` and system prompt reference."""
        c = self.skill_card
        return {
            "skill_id": self.skill_id,
            "code": self.code,
            "short_name": self.short_name,
            "name": c.get("name", self.skill_id),
            "is_baseline": is_baseline_equivalent(self.knobs),
            "when_to_use": c.get("when_to_use", ""),
            "when_not_to_use": c.get("when_not_to_use", ""),
            "behavior_summary": c.get("behavior_summary", ""),
            "scene_tags": list(c.get("scene_tags") or []),
            "sim_benchmark": dict(c.get("sim_benchmark") or {}),
        }


def builtin_strategy_presets():
    """Return built-in strategy profiles so `egg_grasp_v345_baseline.json` is always present out-of-the-box."""
    baseline = StrategyProfile(
        skill_id=BASELINE_SKILL_ID,
        skill_card=copy.deepcopy(BASELINE_SKILL_CARD),
        knobs=dict(BASELINE_KNOBS),
        saved="2026-11-01 00:00:00",
    )

    dense_knobs = dict(BASELINE_KNOBS)
    dense_knobs.update({
        "egg_open_margin_mm": 12.5,
        "neighbor_safe_clearance_mm": 5.0,
        "w_neighbor_intrusion": 1.80,
        "neighbor_penalty_cap": -16.0,
        "final_approach_dist_mm": 28.0,
        "final_approach_slow_factor": 4.5,
    })
    dense_card = {
        "name": "密集贴靠鸡蛋防碰剥离技能 (Dense Anti-Poke)",
        "when_to_use": (
            "当目标鸡蛋与相邻鸡蛋净空 < 12 mm，或位于角落/容器边缘且周围有多颗鸡蛋包围，"
            "标准 +16 mm 预张开容易在指垫下探时把相邻鸡蛋推歪时使用。"
        ),
        "when_not_to_use": (
            "当目标周围空旷（邻蛋距离 > 25 mm）时无需使用，因为该技能收窄了张开裕量并放慢了下探速度。"
        ),
        "behavior_summary": describe_diff_vs_baseline(dense_knobs),
        "scene_tags": ["egg", "fragile", "high_clutter", "neighbor_close", "anti_poke"],
        "sim_benchmark": {
            "trained_scene": "fill_egg_holder",
            "noise_level": None,
            "success_rate": None,
            "avg_cycle_time_s": None,
            "notes": "人工示例预设，未经仿真测量 / 寻优",
        },
    }
    dense = StrategyProfile(
        skill_id="dense_cluster_anti_poke",
        skill_card=dense_card,
        knobs=dense_knobs,
        saved="2026-11-01 00:00:00",
        short_name="Dense Anti-Poke",
        code="DP",
    )

    dry_knobs = dict(BASELINE_KNOBS)
    dry_knobs.update({
        "cam_hover_verify_height_mm": 60.0,
        "cam_hover_soft_limit_deg": 3.0,
        "cam_hover_max_speed_pct": 10.0,
        "dry_run_transit_z_lift_mm": 40.0,
        "dry_run_dwell_s": 1.5,
        "dry_run_wrist_refine": True,
        "dry_run_return_home": True,
    })
    dry_card = {
        "name": "无夹爪实机多点悬停预演技能 (Hover Dry-Run)",
        "when_to_use": (
            "在实机夹爪尚未接入或首次验证视觉-运动闭环时使用："
            "机械臂依次在目标物体 A 正上方 60 mm 悬停 -> 抬升 40 mm 跨点过渡 -> 下一目标点/蛋托孔 B 正上方 60 mm 悬停 -> 回起点。"
        ),
        "when_not_to_use": "需要真正下探闭合夹爪抓取实物入槽时，请切换回 egg_grasp_v345_baseline。",
        "behavior_summary": (
            "抓取相关旋钮与基准相同；实机悬停高度 60 mm（基准 100 mm）、40 mm 跨点抬升过渡、"
            "1.5 s 驻留观察与腕部准星自动复核，全程不下探、不驱动夹爪。"
        ),
        "scene_tags": ["dry_run", "hover_only", "no_gripper", "calibration_verify", "real_robot"],
        "sim_benchmark": {
            "trained_scene": "real_tabletop_dry_run",
            "noise_level": None,
            "success_rate": None,
            "avg_cycle_time_s": None,
            "notes": "专为无夹爪阶段实机多目标悬停巡检与抓放动线预演设计；未经测量",
        },
    }
    dry = StrategyProfile(
        skill_id="hover_dry_run_safe",
        skill_card=dry_card,
        knobs=dry_knobs,
        saved="2026-11-01 00:00:00",
        short_name="Hover Dry-Run",
        code="HD",
    )

    # V3.60: the V3.59 real-robot defaults (60 mm hover + confidence/nearest target order) as a NON-default preset
    qh_knobs = dict(BASELINE_KNOBS)
    qh_knobs.update({"cam_hover_verify_height_mm": 60.0, "target_order_mode": "confidence_then_nearest"})
    qh_card = {
        "name": "近距悬停复核技能 (Quick Hover)",
        "when_to_use": "实机「悬停复核」想用更低的 60 mm 悬停、并且未选中行时优先选置信度高、离基座近的目标时使用。",
        "when_not_to_use": "需要与 V3.58 完全一致的悬停复核行为（100 mm、表格顺序）时，请用基准 EB。",
        "behavior_summary": describe_diff_vs_baseline(qh_knobs),
        "scene_tags": ["real_robot", "hover_only", "verify"],
        "sim_benchmark": {"trained_scene": None, "noise_level": None, "success_rate": None,
                          "avg_cycle_time_s": None, "notes": "V3.59 的实机默认值，V3.60 起改为非默认预设；未经测量"},
    }
    qh = StrategyProfile(
        skill_id="quick_hover_verify",
        skill_card=qh_card,
        knobs=qh_knobs,
        saved="2026-11-01 00:00:00",
        short_name="Quick Hover",
        code="QH",
    )
    # V3.61 STACK CUPS: cup skill (rim pinch) - numbers from the offline study, egg knobs unchanged (not used)
    cs_card = {
        "name": "叠杯子·捏杯沿技能 (Cup Stack, rim pinch)",
        "when_to_use": "stack_cups 场景：两只直立的一次性派对杯（外红内白），把一只放进另一只里面叠起来。",
        "when_not_to_use": "杯子倒放、已被压扁或杯口朝下；杯子距机械臂太远（|x| > 0.36 m 时竖直向下够不到运送高度）。",
        "behavior_summary": ("竖直向下捏住杯沿（一指在内、一指在外，深 25 mm）→ 抬到杯底高出另一杯口 20 mm → 运到目标杯上方"
                             "（途中绕杯轴转到放置方位）→ 对中（≤7.5 mm、≤3°）→ 下降到套好位置上方 85 mm（杯底刚进杯口）松手，"
                             "让杯子自己滑到底。参数来自 MuJoCo 离线松手试验与 URDF 严格 IK 可达性试验。"),
        "scene_tags": ["stack_cups", "cup", "rim_pinch", "nesting"],
        "sim_benchmark": {"trained_scene": "stack_cups", "noise_level": None, "success_rate": None,
                          "avg_cycle_time_s": None,
                          "notes": "离线物理试验（只有杯子、刚体模型）+ IK 试验；机械臂完整闭环尚未在仿真里跑过，成功率未测"},
    }
    cs = StrategyProfile(
        skill_id="cup_stack_rim_pinch",
        skill_card=cs_card,
        knobs=dict(BASELINE_KNOBS),
        saved="2026-11-01 00:00:00",
        short_name="Cup Stack",
        code="CS",
    )
    return [baseline, dense, dry, qh, cs]


def _inside(path, folder):
    try:
        rp, rf = os.path.realpath(path), os.path.realpath(folder)
        return os.path.commonpath([rp, rf]) == rf
    except ValueError:
        return False


def save_strategy_file(profile, path=None):
    """Save a StrategyProfile to JSON (V3.60: user presets go to USER_STRATEGY_DIR; the shipped release folder
    config/strategies/ is read-only and refused)."""
    if isinstance(profile, dict):
        profile = StrategyProfile.from_dict(profile, path=path)
    target = path or os.path.join(USER_STRATEGY_DIR, f"{profile.skill_id}.json")
    if _inside(target, STRATEGY_DIR) or _inside(target, HERE) and not _inside(target, os.path.join(HERE, "logs")):
        raise ValueError(f"不能写入发布文件夹（{target}）：发布包内的文件只读；请另存到 {USER_STRATEGY_DIR}")
    os.makedirs(os.path.dirname(os.path.realpath(target)), exist_ok=True)
    profile.saved = time.strftime("%Y-%m-%d %H:%M:%S")
    profile.path = target
    data = profile.to_dict()
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, target)
    return target


def load_strategy_file(path):
    """Load a StrategyProfile from a JSON file path."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return StrategyProfile.from_dict(d, path=path)


def ensure_builtin_strategies(strategy_dir=None):
    """V3.60: never writes anything (the release folder is read-only).  Returns the shipped baseline file path, or
    None when it is missing (the built-in in-memory presets are used instead)."""
    base_fp = os.path.join(strategy_dir or STRATEGY_DIR, f"{BASELINE_SKILL_ID}.json")
    return base_fp if os.path.isfile(base_fp) else None


BUILTIN_SKILL_IDS = tuple(p.skill_id for p in builtin_strategy_presets())


def skill_display_name(profile):
    """V3.61: the name shown in the GUI = skill card name + its 2-letter code, e.g. '叠杯子·捏杯沿技能 (...) [CS]'."""
    name = (profile.skill_card or {}).get("name") or profile.short_name or profile.skill_id
    return f"{name} [{profile.code or '??'}]"


def list_strategy_files(strategy_dir=None, user_dir=None):
    """Return list of (skill_id, display_name, file_path_or_None, profile): shipped presets (built-in order, file
    if present else the in-memory copy), then user presets from USER_STRATEGY_DIR.  Duplicate skill_ids / codes
    are skipped (the first one wins)."""
    sdir = strategy_dir or STRATEGY_DIR
    udir = user_dir or USER_STRATEGY_DIR
    out, seen_ids, seen_codes = [], set(), set()

    def _add(prof, fp):
        if prof.skill_id in seen_ids or (prof.code and prof.code in seen_codes):
            return
        seen_ids.add(prof.skill_id)
        if prof.code:
            seen_codes.add(prof.code)
        out.append((prof.skill_id, prof.skill_card.get("name", prof.skill_id), fp, prof))

    for bp in builtin_strategy_presets():
        fp = os.path.join(sdir, f"{bp.skill_id}.json")
        try:
            _add(load_strategy_file(fp) if os.path.isfile(fp) else bp, fp if os.path.isfile(fp) else None)
        except Exception:
            _add(bp, None)
    extra = sorted(glob.glob(os.path.join(sdir, "*.json"))) + sorted(glob.glob(os.path.join(udir, "*.json")))
    for fp in extra:
        if os.path.basename(fp) == "default.json":
            continue
        try:
            _add(load_strategy_file(fp), fp)
        except Exception:
            continue
    return out


def validate_new_skill(profile, existing=None):
    """V3.60: a user preset needs its own skill_id, short English name and unique 2-letter code.
    -> error string or None"""
    existing = list_strategy_files() if existing is None else existing
    if profile.skill_id in BUILTIN_SKILL_IDS:
        return f"skill_id `{profile.skill_id}` 是内置技能（只读）：请改一个新的 skill_id 再保存"
    if not re.match(r"^[A-Za-z0-9_\-]+$", profile.skill_id or ""):
        return "skill_id 只能用英文字母、数字、下划线和减号"
    if not profile.short_name:
        return "请填写技能英文短名 (short_name)，例如 Dense Anti-Poke"
    if not profile.code:
        return "请填写两个大写英文字母的技能缩写 (code)，例如 DP"
    for sid, _n, _fp, prof in existing:
        if sid != profile.skill_id and prof.code == profile.code:
            return f"缩写 {profile.code} 已被 `{sid}` 使用：请换一个"
    return None


def export_llm_skill_catalog(strategy_dir=None, active_profile=None):
    """Build the natural-language skill catalog injected into `observe_scene` for LLM/VLM selection."""
    items = list_strategy_files(strategy_dir)
    act = active_profile or get_active_strategy()
    return {
        "active_skill_id": act.skill_id,
        "active_skill_code": act.code,
        "active_skill_name": act.skill_card.get("name", act.skill_id),
        "active_is_baseline": is_baseline_equivalent(act.knobs),
        "active_skill_card": act.llm_brief(),
        "available_skills": [prof.llm_brief() for _sid, _name, _fp, prof in items],
        "llm_usage_hint": (
            "Read each skill's `when_to_use`, `when_not_to_use`, and `behavior_summary` to choose the best "
            "strategy profile for the current tabletop arrangement. Default is `egg_grasp_v345_baseline`."
        ),
    }


# Global active strategy singleton shared inside process
_ACTIVE_PROFILE = None


def read_active_strategy_file(path=None):
    """V3.60: the last applied profile (logs/v360_active_strategy.json) or None."""
    fp = path or ACTIVE_STRATEGY_FILE
    try:
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        prof = StrategyProfile.from_dict(d.get("profile", d), path=d.get("source_path"))
        return prof
    except Exception:
        return None


def get_active_strategy(reload=False):
    """Return the active StrategyProfile: the last applied one (logs/) or the built-in baseline EB."""
    global _ACTIVE_PROFILE
    if _ACTIVE_PROFILE is not None and not reload:
        return _ACTIVE_PROFILE
    prof = read_active_strategy_file()
    if prof is None:
        prof = builtin_strategy_presets()[0]
        base_fp = ensure_builtin_strategies(STRATEGY_DIR)
        if base_fp:
            try:
                prof = load_strategy_file(base_fp)
            except Exception:
                pass
    _ACTIVE_PROFILE = prof
    return _ACTIVE_PROFILE


def set_active_strategy(profile, sync_default_json=True):
    """Set the active StrategyProfile in memory; with sync_default_json (name kept from V3.59) also persist it to
    logs/v360_active_strategy.json so the console and the simulation GUI share it.  Never writes config/."""
    global _ACTIVE_PROFILE
    if isinstance(profile, dict):
        profile = StrategyProfile.from_dict(profile)
    _ACTIVE_PROFILE = profile
    if sync_default_json:
        try:
            os.makedirs(os.path.dirname(os.path.realpath(ACTIVE_STRATEGY_FILE)), exist_ok=True)
            tmp = ACTIVE_STRATEGY_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"applied": time.strftime("%Y-%m-%d %H:%M:%S"), "source_path": profile.path,
                           "profile": profile.to_dict()}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, ACTIVE_STRATEGY_FILE)
        except Exception:
            pass
    return _ACTIVE_PROFILE


def strategy_sim_env(profile=None, base_env=None):
    """V3.60 SIM ENV INJECTION -> (env dict for start_nero_sim.sh, list of log notes).

    Identity variables (skill id / code / name) are always set.  Each of the 6 SIM_ENV_KNOBS is set only when the
    knob differs from the baseline, so the baseline EB never overrides anything (the backend keeps its own
    defaults, and a value the user exported in the shell stays in effect, as in V3.58)."""
    prof = profile or get_active_strategy()
    k = sanitize_knobs(prof.knobs)
    base_env = os.environ if base_env is None else base_env
    env = {"NERO_V360_SKILL_ID": str(prof.skill_id), "NERO_V360_SKILL_CODE": str(prof.code or "??"),
           "NERO_V360_SKILL_NAME": str(prof.short_name or prof.skill_id)}
    notes = []
    req, rel = float(k["limit_margin_required_deg"]), float(k["limit_margin_relaxed_deg"])
    if rel > req:
        notes.append(f"[WARN] [V3.60 策略] 放宽限位裕量 {rel:g}° > 首选 {req:g}°：注入时按 {req:g}° 处理")
        k["limit_margin_relaxed_deg"] = req
    for key, (var, _dflt, scale) in SIM_ENV_KNOBS.items():
        v, b = float(k[key]), float(BASELINE_KNOBS[key])
        if abs(v - b) <= 1e-9 and not (key == "limit_margin_relaxed_deg" and rel > req):
            if var in base_env:
                notes.append(f"[INFO] [V3.60 策略] {var}={base_env[var]} 来自终端环境变量（基准值不覆盖它）")
            continue
        val = f"{v * scale:.6g}"
        if var in base_env and base_env[var] != val:
            notes.append(f"[WARN] [V3.60 策略] 终端里的 {var}={base_env[var]} 被技能 {prof.code} 的 {val} 覆盖")
        env[var] = val
        notes.append(f"[OK] [V3.60 策略] 注入 {var}={val}（{KNOB_BY_KEY[key]['label']}，基准 {b:g}）")
    return env, notes

# --------------------------------------------------------------------------- Multi-Point Hover Dry-Run Planner
def parse_waypoint_spec(text, fallback_id="target_B"):
    """Parse a manual waypoint string like `-420, 120, 45` or `slot_1 (-510, 123, 43)` (mm) into a target dict.

    If numbers appear in meters (all |v| < 2.0), converts to meters; otherwise treats as millimeters.
    """
    if not text or not str(text).strip():
        return None
    s = str(text).strip()
    import re
    label = fallback_id
    if ":" in s:
        head, tail = s.split(":", 1)
        if head.strip():
            label = head.strip()
        s = tail
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", s)]
    if len(nums) < 2:
        return None
    # V3.60 FIX: decide mm vs m from the numbers the user typed only (V3.59 also looked at the default z = 35,
    # so "-0.51, 0.123" in metres was read as millimetres -> a point 0.5 mm from the base).
    given = nums[:3]
    in_mm = max(abs(v) for v in given) > 2.5
    x, y = nums[0], nums[1]
    if len(nums) >= 3:
        z = nums[2]
    else:
        z = 35.0 if in_mm else 0.035
    if in_mm:
        xyz_m = [x / 1000.0, y / 1000.0, z / 1000.0]
    else:
        xyz_m = [x, y, z]
    return {
        "det_id": label,
        "xyz_m": xyz_m,
        "trusted": True,
        "confidence": 1.0,
        "status": "manual",
    }


def build_dry_run_targets(detected_targets, selected_ids=(), dest_mode="next_target", manual_dest_text="",
                          holder_slots=None, knobs=None):
    """Select and order the list of targets for a Hover Dry-Run sequence.

    Supports:
    - Multiple explicitly selected rows in the table -> visits them in selection order
    - 1 selected row (or top ranked target) + `dest_mode`:
      * `"next_target"`: visits Target #1 -> Target #2 (and up to all trusted targets if `all_targets`)
      * `"all_trusted"`: visits all trusted targets in strategy order
      * `"first_empty_slot"`: visits Target #1 -> first empty egg-holder slot
      * `"manual_xyz"`: visits Target #1 -> manual `[x, y, z]` coordinate
    """
    ranked = rank_detected_targets(detected_targets, knobs=knobs)
    by_id = {str(t.get("det_id")): t for t in ranked if t.get("det_id")}

    chosen = []
    for sid in (selected_ids or ()):
        if str(sid) in by_id:
            chosen.append(by_id[str(sid)])

    if len(chosen) >= 2 and dest_mode not in ("manual_xyz", "first_empty_slot"):
        return chosen

    start_tgt = chosen[0] if chosen else (ranked[0] if ranked else None)
    seq = [start_tgt] if start_tgt is not None else []

    if dest_mode == "manual_xyz":
        m_tgt = parse_waypoint_spec(manual_dest_text, fallback_id="目标点B")
        if m_tgt is not None:
            seq.append(m_tgt)
        return seq

    if dest_mode == "first_empty_slot":
        slots = [t for t in (holder_slots or detected_targets or [])
                 if str(t.get("det_id", "")).startswith("slot_") and t.get("xyz_m")]
        if slots:
            seq.append(slots[0])
            return seq

    if dest_mode == "all_trusted":
        trusted = [t for t in ranked if t.get("trusted")]
        pool = trusted if trusted else ranked
        for t in pool:
            if all(t.get("det_id") != c.get("det_id") for c in seq):
                seq.append(t)
        return seq

    # default: "next_target" -> pick source target + next ranked target (at least 2 points when available)
    for t in ranked:
        if all(t.get("det_id") != c.get("det_id") for c in seq):
            seq.append(t)
            if len(seq) >= 2:
                break
    if len(seq) == 1 and manual_dest_text and str(manual_dest_text).strip():
        m_tgt = parse_waypoint_spec(manual_dest_text, fallback_id="目标点B")
        if m_tgt is not None:
            seq.append(m_tgt)
    return seq


def plan_hover_dry_run_sequence(urdf, joints, limits_rad, q0_rad, tool_T, targets, z_table_m=0.0,
                                base="base_link", fence_fn=None, knobs=None, plan_hover_fn=None):
    """Plan a complete gripper-free Hover Dry-Run sequence across 1..N targets.

    For each target `i`:
    - If moving from target `i-1` to `i` and horizontal travel > 25 mm with `dry_run_transit_z_lift_mm > 0`,
      inserts an intermediate `"transit"` waypoint at `z = max(hz_prev, hz_curr) + transit_lift_m` above the midpoint
      (or above the current target) to guarantee obstacle-free aerial transfer.
    - Plans the `"hover"` waypoint at `z = max(obj_z + hover_height_m, z_table_m + 0.05)` directly over `(obj_x, obj_y)`.
    - Chains IK seeds from the previous waypoint's joint solution `q_prev` so joint motion stays smooth and continuous.

    Returns:
      `{"ok": bool, "steps": [...], "error": str_or_None, "summary": str}`
    """
    if not targets:
        return {"ok": False, "steps": [], "error": "没有可悬停的目标点（请先识别目标或输入目标点坐标）"}

    if plan_hover_fn is None:
        import NERO_V325_Camera as CAM
        plan_hover_fn = CAM.plan_hover_target

    k = sanitize_knobs(knobs)
    hover_mm = float(k.get("cam_hover_verify_height_mm", 60.0))
    soft_deg = float(k.get("cam_hover_soft_limit_deg", 3.0))
    transit_lift_mm = float(k.get("dry_run_transit_z_lift_mm", 40.0))
    dwell_s = float(k.get("dry_run_dwell_s", 1.2))
    return_home = bool(k.get("dry_run_return_home", True))

    steps = []
    warnings = []
    q_prev = [float(v) for v in q0_rad]
    prev_hover_xyz = None

    for idx, tgt in enumerate(targets):
        det_id = str(tgt.get("det_id") or f"target_{idx + 1}")
        ox, oy, oz = [float(v) for v in tgt["xyz_m"]]
        hz = max(oz + hover_mm / 1000.0, float(z_table_m) + 0.05)
        hover_xyz = [ox, oy, hz]

        # Insert safe high-transit bridge between consecutive hover targets
        if prev_hover_xyz is not None and transit_lift_mm > 1.0:
            xy_dist = math.hypot(hover_xyz[0] - prev_hover_xyz[0], hover_xyz[1] - prev_hover_xyz[1])
            if xy_dist > 0.025:
                tz = max(prev_hover_xyz[2], hover_xyz[2]) + transit_lift_mm / 1000.0
                mx = 0.5 * (prev_hover_xyz[0] + hover_xyz[0])
                my = 0.5 * (prev_hover_xyz[1] + hover_xyz[1])
                transit_xyz = [mx, my, tz]
                q_tr, info_tr = plan_hover_fn(
                    urdf, joints, limits_rad, q_prev, tool_T, transit_xyz,
                    base=base, fence_fn=fence_fn, soft_deg=soft_deg,
                )
                if q_tr is None:
                    # Fallback: lift straight up above previous target to max safe reachable height
                    transit_xyz = [hover_xyz[0], hover_xyz[1], max(prev_hover_xyz[2], hover_xyz[2]) + 0.020]
                    q_tr, info_tr = plan_hover_fn(
                        urdf, joints, limits_rad, q_prev, tool_T, transit_xyz,
                        base=base, fence_fn=fence_fn, soft_deg=soft_deg,
                    )
                if q_tr is None:      # V3.60: say so instead of silently moving hover -> hover
                    warnings.append(f"{det_id} 之前的跨点抬升过渡点规划失败（{info_tr}），将直接从上一悬停点移到该悬停点")
                if q_tr is not None:
                    steps.append({
                        "kind": "transit",
                        "index": idx + 1,
                        "det_id": f"过渡->{det_id}",
                        "target_obj": tgt,
                        "obj_xyz_m": [ox, oy, oz],
                        "tcp_xyz_m": transit_xyz,
                        "q_rad": list(q_tr),
                        "q_deg": [round(math.degrees(v), 2) for v in q_tr],
                        "ik_err_mm": round(float(info_tr.get("err_mm", 0.0)), 2),
                        "dwell_s": 0.0,
                    })
                    q_prev = list(q_tr)

        q_hov, info_hov = plan_hover_fn(
            urdf, joints, limits_rad, q_prev, tool_T, hover_xyz,
            base=base, fence_fn=fence_fn, soft_deg=soft_deg,
        )
        if q_hov is None:
            return {
                "ok": False,
                "steps": steps,
                "warnings": warnings,
                "failed_target": det_id,
                "error": f"目标 {det_id} 悬停点 ({hover_xyz[0]*1000:+.0f}, {hover_xyz[1]*1000:+.0f}, {hover_xyz[2]*1000:+.0f}) mm 规划失败：{info_hov}",
            }

        steps.append({
            "kind": "hover",
            "index": idx + 1,
            "det_id": det_id,
            "target_obj": tgt,
            "obj_xyz_m": [ox, oy, oz],
            "tcp_xyz_m": hover_xyz,
            "hover_mm": hover_mm,
            "q_rad": list(q_hov),
            "q_deg": [round(math.degrees(v), 2) for v in q_hov],
            "ik_err_mm": round(float(info_hov.get("err_mm", 0.0)), 2),
            "dwell_s": dwell_s,
        })
        q_prev = list(q_hov)
        prev_hover_xyz = list(hover_xyz)

    if return_home:
        steps.append({
            "kind": "return_home",
            "index": len(targets) + 1,
            "det_id": "回起点姿态",
            "target_obj": None,
            "obj_xyz_m": None,
            "tcp_xyz_m": None,
            "q_rad": [float(v) for v in q0_rad],
            "q_deg": [round(math.degrees(v), 2) for v in q0_rad],
            "ik_err_mm": 0.0,
            "dwell_s": 0.0,
        })

    hover_ids = [s["det_id"] for s in steps if s["kind"] == "hover"]
    chain_txt = " → ".join(hover_ids) + (" → 回起点" if return_home else "")
    return {
        "ok": True,
        "steps": steps,
        "hover_count": len(hover_ids),
        "hover_ids": hover_ids,
        "hover_height_mm": hover_mm,
        "transit_lift_mm": transit_lift_mm,
        "dwell_s": dwell_s,
        "return_home": return_home,
        "summary": f"共 {len(hover_ids)} 个悬停站（{chain_txt}），悬停高 {hover_mm:.0f} mm，跨点抬升 +{transit_lift_mm:.0f} mm",
        "warnings": warnings,
        "error": None,
    }


def execute_hover_dry_run(plan_res, move_fn, actual_q_fn=None, fk_tcp_fn=None, wrist_check_fn=None,
                          stop_fn=None, sleep_fn=time.sleep, log_fn=None):
    """Execute a planned Hover Dry-Run sequence step by step.

    Pure execution helper so both real-robot `CameraTabUI` / `SkillStrategyTabUI` and offline tests can run it.
    """
    if not plan_res or not plan_res.get("ok"):
        err = (plan_res or {}).get("error") or "悬停预演规划无效"
        if log_fn:
            log_fn(f"[FAIL] [V3.59 dry-run] {err}")
        return {"ok": False, "completed_hovers": 0, "results": [], "error": err}

    results = []
    completed_hovers = 0
    steps = plan_res.get("steps") or []

    for s_i, step in enumerate(steps):
        if stop_fn and stop_fn():
            if log_fn:
                log_fn(f"[WARN] [V3.59 dry-run] 用户在第 {s_i + 1}/{len(steps)} 步请求停止悬停预演")
            return {"ok": False, "stopped": True, "completed_hovers": completed_hovers, "results": results, "error": "已停止"}

        kind = step["kind"]
        det_id = step["det_id"]
        q_rad = step["q_rad"]
        label = f"Dry-Run {det_id}"
        mres = move_fn(q_rad, label)
        if not (mres and mres.get("ok")):
            err = (mres or {}).get("error") or f"{det_id} 运动未完成"
            if log_fn:
                log_fn(f"[FAIL] [V3.59 dry-run] {det_id} 运动失败：{err}")
            return {"ok": False, "completed_hovers": completed_hovers, "results": results, "error": err}

        rec = {"kind": kind, "det_id": det_id, "ok": True}
        if kind == "hover":
            completed_hovers += 1
            txyz = step["tcp_xyz_m"]
            err_mm = 0.0
            act_xyz = txyz
            if actual_q_fn and fk_tcp_fn:
                qa_deg = actual_q_fn()
                if qa_deg:
                    act = fk_tcp_fn([math.radians(v) for v in qa_deg])
                    if act is not None:
                        act_xyz = list(act)
                        err_mm = math.sqrt(sum((act_xyz[i] - txyz[i]) ** 2 for i in range(3))) * 1000.0
            rec["target_tcp_mm"] = [round(v * 1000.0, 1) for v in txyz]
            rec["actual_tcp_mm"] = [round(v * 1000.0, 1) for v in act_xyz]
            rec["tcp_err_mm"] = round(err_mm, 2)

            if log_fn:
                log_fn(
                    f"[OK] [V3.59 dry-run] 悬停站 #{completed_hovers} ({det_id}) 到位："
                    f"目标 TCP ({txyz[0]*1000:+.0f}, {txyz[1]*1000:+.0f}, {txyz[2]*1000:+.0f}) mm · "
                    f"实测偏差 {err_mm:.1f} mm"
                )

            dwell = float(step.get("dwell_s") or 0.0)
            if dwell > 0.0:
                t_end = time.time() + dwell
                while time.time() < t_end:
                    if stop_fn and stop_fn():
                        break
                    sleep_fn(min(0.1, max(0.01, t_end - time.time())))

            if wrist_check_fn and step.get("target_obj") is not None:
                try:
                    w_res = wrist_check_fn(step["target_obj"])
                    if w_res and w_res.get("ok"):
                        rec["wrist_refine"] = w_res
                        dx, dy = w_res.get("delta_tcp_mm", (0.0, 0.0))
                        if log_fn:
                            log_fn(
                                f"[OK] [V3.59 dry-run] 悬停站 #{completed_hovers} ({det_id}) 腕部准星复核："
                                f"Δx={dx:+.1f} mm, Δy={dy:+.1f} mm, Δψ={w_res.get('delta_yaw_deg', 0.0):+.1f}°"
                            )
                except Exception as exc:
                    if log_fn:
                        log_fn(f"[WARN] [V3.59 dry-run] 腕部准星复核跳过：{exc}")
        elif kind == "return_home" and log_fn:
            log_fn("[OK] [V3.59 dry-run] 多点悬停预演完成，已回到出发前姿态 ✓")

        results.append(rec)

    return {"ok": True, "completed_hovers": completed_hovers, "results": results, "error": None}


# --------------------------------------------------------------------------- Tkinter GUI Tab: 技能参数旋钮
class SkillStrategyTabUI:
    """Dedicated GUI tab for managing Strategy Profiles, editing Skill Cards, tuning knobs, and running Hover Dry-Run."""

    def __init__(self, tk, ttk, parent, cam_tab=None, pipe=None, jpanel=None, log=None, ask=None, mode="real"):
        self.tk = tk
        self.ttk = ttk
        self.parent = parent
        self.mode = "sim" if mode == "sim" else "real"       # V3.60 SIM TAB: "sim" hides the real dry-run panel
        self.cam_tab = cam_tab
        self.pipe = pipe
        self.jpanel = jpanel
        self.log = log or (lambda msg: None)
        self.ask = ask or (lambda _t, _q: True)

        self._sections = {}                  # V3.61 COLLAPSIBLE: base state before any panel is built
        self.profile = get_active_strategy()
        self.knob_vars = {}
        self._last_dry_plan = None
        self._active_mtime = self._active_file_mtime()

        self._build_ui()
        # V3.60: startup only shows the active profile; it never rewrites files or the Camera-tab hover field
        self.load_profile_into_ui(self.profile, announce=False, persist=False, set_hover=False)

    def _build_ui(self):
        tk, ttk = self.tk, self.ttk
        import NERO_V314_Joint_Control as JC

        # Outer scrollable container so all 5 knob groups + Skill Card + Dry-Run panel fit comfortably
        outer = ttk.Frame(self.parent)
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0)
        sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        main = ttk.Frame(canvas, padding=8)
        win_id = canvas.create_window(0, 0, window=main, anchor="nw")
        main.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win_id, width=e.width))

        def _wheel(e):
            if e.widget.winfo_class() in ("Scale", "Text", "TCombobox", "Spinbox"):
                return
            canvas.yview_scroll(-1 if (getattr(e, "num", 0) == 4 or getattr(e, "delta", 0) > 0) else 1, "units")

        for seq in ("<Button-4>", "<Button-5>", "<MouseWheel>"):
            canvas.bind_all(seq, lambda e: _wheel(e) if str(e.widget).startswith(str(outer)) else None, add="+")

        # ---- 1. Top Strategy Bar ----
        top_lf = ttk.LabelFrame(main, text="技能参数包选择与存储（默认 EB = 还原 V2.7~V3.58 调好的抓鸡蛋逻辑）", padding=6)
        top_lf.pack(fill="x", pady=(0, 6))

        r_top = ttk.Frame(top_lf)
        r_top.pack(fill="x")
        ttk.Label(r_top, text="当前技能包:", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        self.v_preset = tk.StringVar(value="")
        self.cb_preset = ttk.Combobox(r_top, textvariable=self.v_preset, state="readonly", width=42)
        self.cb_preset.pack(side="left", padx=(4, 8))
        self.cb_preset.bind("<<ComboboxSelected>>", lambda _e: self.on_select_preset())

        self.b_apply = ttk.Button(r_top, text="✓ 应用当前参数", command=self.on_apply)
        self.b_apply.pack(side="left", padx=2)
        self.b_save = ttk.Button(r_top, text="💾 保存为用户技能", command=self.on_save)
        self.b_save.pack(side="left", padx=2)
        self.b_save_as = ttk.Button(r_top, text="另存为新技能…", command=self.on_save_as)
        self.b_save_as.pack(side="left", padx=2)
        self.b_load = ttk.Button(r_top, text="📂 打开外部 .json…", command=self.on_load_dialog)
        self.b_load.pack(side="left", padx=2)
        self.b_reset_base = ttk.Button(r_top, text="↺ 一键还原抓鸡蛋基准", command=self.on_reset_baseline)
        self.b_reset_base.pack(side="left", padx=(8, 2))
        self.b_copy_llm = ttk.Button(r_top, text="复制 LLM 技能目录 JSON", command=self.on_copy_llm_catalog)
        self.b_copy_llm.pack(side="right", padx=2)

        self.v_status = tk.StringVar(value="")
        JC.fixed_label(tk, top_lf, 2, textvariable=self.v_status, font=("TkFixedFont", 9), fg="#1b5e20",
                       wraplength=1200, justify="left", anchor="nw", pack={"fill": "x", "pady": (4, 0)})
        note360 = ("标签：[仿真✓] = 下次「启动仿真」时经环境变量注入后端（仅与基准不同时注入）；[实机✓] = Camera 页悬停复核 / "
                   "多点预演使用；[未接入] = 只记录，本版本还没有代码读取；[离线试验✓] = 数值来自离线仿真试验，运行代码尚未读取（V3.61）。内置技能只读，用户技能保存到 "
                   f"{USER_STRATEGY_DIR}；当前选择保存在 logs/。")
        if self.mode == "sim":
            note360 = ("仿真窗口：这里「应用」的参数在下次启动仿真时生效（本次运行已按启动时的技能注入）。 " + note360)
        ttk.Label(top_lf, text=note360, foreground="#607d8b", wraplength=1200, justify="left").pack(fill="x", pady=(2, 0))

        # V3.61 COLLAPSIBLE: one click folds / unfolds every section below (own row, never pushed off-screen)
        r_fold = ttk.Frame(main)
        r_fold.pack(fill="x", padx=4, pady=(4, 0))
        self.b_collapse_all = ttk.Button(r_fold, text="⊟ 全部折叠", command=self.collapse_all)
        self.b_collapse_all.pack(side="left", padx=2)
        self.b_expand_all = ttk.Button(r_fold, text="⊞ 全部展开", command=self.expand_all)
        self.b_expand_all.pack(side="left", padx=2)
        ttk.Label(r_fold, text="点击下面各区域标题前的 ▾ / ▸ 单独折叠或展开", foreground="#607d8b").pack(side="left", padx=8)

        # ---- 2. Skill Card (自然语言技能名片) ----
        card_lf = self._collapsible(main, "skill_card", "技能名片（Skill Card · 写入 .json 供 LLM/VLM 理解应用场景与调参依据）")

        rc1 = ttk.Frame(card_lf)
        rc1.pack(fill="x", pady=2)
        ttk.Label(rc1, text="技能 ID (skill_id):", width=16).pack(side="left")
        self.v_skill_id = tk.StringVar(value=BASELINE_SKILL_ID)
        ttk.Entry(rc1, textvariable=self.v_skill_id, width=28).pack(side="left", padx=(0, 12))
        ttk.Label(rc1, text="技能中文名 (name):").pack(side="left")
        self.v_card_name = tk.StringVar(value="")
        ttk.Entry(rc1, textvariable=self.v_card_name, width=42).pack(side="left", padx=(4, 12))
        ttk.Label(rc1, text="场景标签 (逗号分隔):").pack(side="left")
        self.v_card_tags = tk.StringVar(value="")
        ttk.Entry(rc1, textvariable=self.v_card_tags, width=28).pack(side="left", padx=4)

        rc1b = ttk.Frame(card_lf)          # V3.60 SKILL CODES
        rc1b.pack(fill="x", pady=2)
        ttk.Label(rc1b, text="英文短名 (short_name):", width=16).pack(side="left")
        self.v_short_name = tk.StringVar(value="")
        ttk.Entry(rc1b, textvariable=self.v_short_name, width=28).pack(side="left", padx=(0, 12))
        ttk.Label(rc1b, text="两字母缩写 (code):").pack(side="left")
        self.v_code = tk.StringVar(value="")
        ttk.Entry(rc1b, textvariable=self.v_code, width=4).pack(side="left", padx=4)
        ttk.Label(rc1b, text="（两个大写字母、唯一，用于日志与任务打分，如 EB / DP / HD / QH）",
                  foreground="#607d8b").pack(side="left", padx=4)

        rc2 = ttk.Frame(card_lf)
        rc2.pack(fill="x", pady=2)
        ttk.Label(rc2, text="适用场景 (when_to_use):", width=16).pack(side="left")
        self.v_when_use = tk.StringVar(value="")
        ttk.Entry(rc2, textvariable=self.v_when_use).pack(side="left", fill="x", expand=True, padx=2)

        rc3 = ttk.Frame(card_lf)
        rc3.pack(fill="x", pady=2)
        ttk.Label(rc3, text="禁忌/不适用 (when_not):", width=16).pack(side="left")
        self.v_when_not = tk.StringVar(value="")
        ttk.Entry(rc3, textvariable=self.v_when_not).pack(side="left", fill="x", expand=True, padx=2)

        rc4 = ttk.Frame(card_lf)
        rc4.pack(fill="x", pady=2)
        ttk.Label(rc4, text="行为特征摘要 (summary):", width=16).pack(side="left")
        self.v_behavior = tk.StringVar(value="")
        ttk.Entry(rc4, textvariable=self.v_behavior).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(rc4, text="⚡ 按当前旋钮自动生成摘要", command=self.on_auto_summary).pack(side="right", padx=(4, 0))

        # ---- 3. Gripper-Free Multi-Point Hover Dry-Run Panel ----
        # V3.60 SIM TAB: the dry-run drives the real arm -> console only (not shown in the simulation window)
        dry_lf = self._collapsible(
            main, "dry_run",
            "无夹爪多点悬停预演测试台（Hover Dry-Run · 从物体 A 上方悬停 → 安全抬升 → 下一目标点 B 上方悬停 → 回原位）",
            show=(self.mode != "sim"))

        rd1 = ttk.Frame(dry_lf)
        rd1.pack(fill="x", pady=2)
        ttk.Label(rd1, text="预演航线模式:", font=("TkDefaultFont", 9, "bold")).pack(side="left")
        self.dry_modes = [
            ("next_target", "① 源目标 → 下一个识别目标（表内前 2 个或所选多行）"),
            ("all_trusted", "② 依次巡检全部可信目标（#1 → #2 → #3 …）"),
            ("first_empty_slot", "③ 源目标 → 蛋托第 1 个空槽 (slot_1…)"),
            ("manual_xyz", "④ 源目标 → 右侧手动输入的目标点 B (x, y, z mm)"),
        ]
        self.v_dry_mode = tk.StringVar(value=self.dry_modes[0][1])
        self.cb_dry_mode = ttk.Combobox(rd1, textvariable=self.v_dry_mode,
                                        values=[m[1] for m in self.dry_modes], state="readonly", width=44)
        self.cb_dry_mode.pack(side="left", padx=(4, 10))
        ttk.Label(rd1, text="手动目标点 B (x, y, z mm):").pack(side="left")
        self.v_dry_manual = tk.StringVar(value="-450, 123, 43")
        ttk.Entry(rd1, textvariable=self.v_dry_manual, width=18).pack(side="left", padx=(4, 8))

        rd2 = ttk.Frame(dry_lf)
        rd2.pack(fill="x", pady=(4, 2))
        self.b_dry_check = ttk.Button(rd2, text="🔍 1. 预演航线规划校验（不动真机）", command=self.on_dry_run_plan_only)
        self.b_dry_check.pack(side="left", padx=(0, 6))
        self.b_dry_exec = ttk.Button(rd2, text="▶ 2. 执行实机多点悬停预演 (Dry-Run)", command=self.on_dry_run_execute)
        self.b_dry_exec.pack(side="left", padx=4)
        self.b_dry_stop = ttk.Button(rd2, text="■ 停止预演", command=self.on_dry_run_stop)
        self.b_dry_stop.pack(side="left", padx=4)
        ttk.Label(
            rd2,
            text="说明：全程只走上方悬停点与跨点抬升点，不下探、不驱动夹爪，每一步均经 3D 安全围栏校验。",
            foreground="#607d8b",
        ).pack(side="left", padx=8)

        self.v_dry_status = tk.StringVar(
            value="就绪：先在「Camera调试 → ④ 视觉定位」识别桌面物体（或在上方填目标点坐标），再点「预演航线规划校验」或「执行实机多点悬停预演」。"
        )
        JC.fixed_label(tk, dry_lf, 2, textvariable=self.v_dry_status, font=("TkFixedFont", 9), fg="#0d47a1",
                       wraplength=1200, justify="left", anchor="nw", pack={"fill": "x", "pady": (2, 0)})

        # ---- 4. Five Knob Group Panels (2-column grid per group for compact scanning) ----
        self.group_frames = {}
        for gid, glabel in KNOB_GROUPS:
            gf = self._collapsible(main, gid, glabel)
            self.group_frames[gid] = gf
            specs = [s for s in KNOB_SPECS if s["group"] == gid]
            for idx, spec in enumerate(specs):
                r_i = idx // 2
                c_i = (idx % 2) * 4
                k = spec["key"]
                ttk.Label(gf, text=f"{spec['label']}:", width=22, anchor="w").grid(
                    row=r_i, column=c_i, sticky="w", padx=(4, 2), pady=2
                )
                if spec["type"] == "bool":
                    var = tk.BooleanVar(value=bool(spec["default"]))
                    w = ttk.Checkbutton(gf, text="启用", variable=var)
                    w.grid(row=r_i, column=c_i + 1, sticky="w", padx=2, pady=2)
                elif spec["type"] == "enum":
                    var = tk.StringVar(value=str(spec["default"]))
                    w = ttk.Combobox(gf, textvariable=var, values=spec["choices"], state="readonly", width=22)
                    w.grid(row=r_i, column=c_i + 1, sticky="w", padx=2, pady=2)
                else:
                    var = tk.StringVar(value=f"{spec['default']:g}")
                    w = ttk.Entry(gf, textvariable=var, width=10)
                    w.grid(row=r_i, column=c_i + 1, sticky="w", padx=2, pady=2)
                self.knob_vars[k] = var
                def_txt = "启用" if spec["default"] is True else ("关闭" if spec["default"] is False else f"{spec['default']}")
                u_txt = f" {spec['unit']}" if spec.get("unit") else ""
                wired = spec.get("wired", "none")
                ttk.Label(gf, text=f"{u_txt} (默认: {def_txt}) [{WIRING_LABELS[wired]}]",
                          foreground=("#2e7d32" if wired != "none" else "#9e9e9e"), width=26, anchor="w").grid(
                    row=r_i, column=c_i + 2, sticky="w", padx=(2, 6), pady=2
                )
                ttk.Label(gf, text=spec["desc"], foreground="#555555", wraplength=340, justify="left").grid(
                    row=r_i, column=c_i + 3, sticky="w", padx=(0, 12), pady=2
                )
            gf.columnconfigure(3, weight=1)
            gf.columnconfigure(7, weight=1)

        self._refresh_preset_dropdown()

    # ------------------------------------------------------------------ V3.61 COLLAPSIBLE SECTIONS
    def _collapsible(self, parent, key, title, show=True):
        """a section with a ▾/▸ header button; returns the body frame (callers fill it like a LabelFrame)."""
        ttk = self.ttk
        if not hasattr(self, "_sections"):
            self._sections = {}
        box = ttk.Frame(parent)
        if show:
            box.pack(fill="x", pady=(0, 4))
        btn = ttk.Button(box, text=f"▾ {title}", command=lambda k=key: self.toggle_section(k))
        btn.pack(anchor="w", fill="x")
        body = ttk.LabelFrame(box, padding=6)
        body.pack(fill="x")
        self._sections[key] = {"box": box, "btn": btn, "body": body, "title": title, "open": True, "shown": bool(show)}
        return body

    def toggle_section(self, key, open_=None):
        sec = self._sections[key]
        want = (not sec["open"]) if open_ is None else bool(open_)
        if want == sec["open"]:
            return want
        if want:
            sec["body"].pack(fill="x")
        else:
            sec["body"].pack_forget()
        sec["open"] = want
        sec["btn"].configure(text=("▾ " if want else "▸ ") + sec["title"])
        return want

    def collapse_all(self):
        for k in self._sections:
            self.toggle_section(k, False)

    def expand_all(self):
        for k in self._sections:
            self.toggle_section(k, True)

    def section_states(self):
        return {k: v["open"] for k, v in self._sections.items() if v["shown"]}

    def _refresh_preset_dropdown(self):
        items = list_strategy_files(STRATEGY_DIR)
        self._preset_map = {}
        labels = []
        cur_label = None
        for sid, name, fp, prof in items:
            lab = f"{skill_display_name(prof)} · {sid}" + ("（内置）" if sid in BUILTIN_SKILL_IDS else "（用户）")
            self._preset_map[lab] = (sid, fp, prof)
            labels.append(lab)
            if sid == self.profile.skill_id:
                cur_label = lab
        self.cb_preset.configure(values=labels)
        if cur_label:
            self.v_preset.set(cur_label)
        elif labels:
            self.v_preset.set(labels[0])

    def load_profile_into_ui(self, profile, announce=True, persist=True, set_hover=True):
        self.profile = profile
        set_active_strategy(profile, sync_default_json=persist)
        if persist:
            self._active_mtime = self._active_file_mtime()
        self.v_skill_id.set(profile.skill_id)
        self.v_short_name.set(profile.short_name)
        self.v_code.set(profile.code)
        c = profile.skill_card
        self.v_card_name.set(c.get("name", ""))
        self.v_when_use.set(c.get("when_to_use", ""))
        self.v_when_not.set(c.get("when_not_to_use", ""))
        self.v_behavior.set(c.get("behavior_summary", ""))
        self.v_card_tags.set(", ".join(c.get("scene_tags") or []))
        for spec in KNOB_SPECS:
            k = spec["key"]
            val = profile.get(k)
            var = self.knob_vars[k]
            if spec["type"] == "bool":
                var.set(bool(val))
            elif spec["type"] == "enum":
                var.set(str(val))
            else:
                var.set(f"{float(val):g}" if isinstance(val, (int, float)) else str(val))
        self._sync_to_camera_tab(set_hover=set_hover)
        self._refresh_preset_dropdown()
        self._update_status_banner()
        if announce:
            self.log(
                f"[OK] [V3.60 策略] 已加载技能 {profile.label()} `{profile.skill_id}`（{c.get('name', '')}）· "
                f"{'100% 还原抓鸡蛋原版基准 ✓' if is_baseline_equivalent(profile.knobs) else describe_diff_vs_baseline(profile.knobs)}"
            )

    def collect_profile_from_ui(self):
        raw_knobs = {}
        for spec in KNOB_SPECS:
            k = spec["key"]
            raw_knobs[k] = self.knob_vars[k].get()
        knobs = sanitize_knobs(raw_knobs)
        sid = self.v_skill_id.get().strip() or BASELINE_SKILL_ID
        card = {
            "name": self.v_card_name.get().strip() or f"技能 ({sid})",
            "when_to_use": self.v_when_use.get().strip(),
            "when_not_to_use": self.v_when_not.get().strip(),
            "behavior_summary": self.v_behavior.get().strip() or describe_diff_vs_baseline(knobs),
            "scene_tags": [x.strip() for x in self.v_card_tags.get().split(",") if x.strip()],
            "sim_benchmark": copy.deepcopy(self.profile.skill_card.get("sim_benchmark") or {}),
        }
        same = sid == self.profile.skill_id
        return StrategyProfile(
            skill_id=sid,
            skill_card=card,
            knobs=knobs,
            path=self.profile.path if same else None,
            short_name=self.v_short_name.get().strip(),
            code=self.v_code.get().strip(),
        )

    def _sync_to_camera_tab(self, set_hover=True):
        if self.cam_tab is not None and hasattr(self.cam_tab, "apply_strategy_profile"):
            try:
                self.cam_tab.apply_strategy_profile(self.profile, set_hover=set_hover)
            except Exception as exc:
                self.log(f"[WARN] [V3.60 策略] 同步到 Camera 页失败：{exc}")

    @staticmethod
    def _active_file_mtime():
        try:
            return os.path.getmtime(ACTIVE_STRATEGY_FILE)
        except OSError:
            return None

    def reload_active_if_changed(self):
        """V3.60: the other window (console <-> simulation GUI) applied a skill -> show it here too (no hover write)."""
        mt = self._active_file_mtime()
        if mt is None or mt == self._active_mtime:
            return False
        self._active_mtime = mt
        prof = read_active_strategy_file()
        if prof is None:
            return False
        self.load_profile_into_ui(prof, announce=False, persist=False, set_hover=False)
        self.log(f"[INFO] [V3.60 策略] 另一个窗口已应用技能 {prof.label()}，本页已同步显示")
        return True

    def _update_status_banner(self):
        knobs = self.profile.knobs
        eq = is_baseline_equivalent(knobs)
        diffs = diff_vs_baseline(knobs)
        src_fp = self.profile.path or os.path.join(STRATEGY_DIR, f"{self.profile.skill_id}.json")
        src_fp = self.profile.path or "内置（内存）"
        if eq and not diffs:
            same = ("与基准 EB 完全相同" if self.profile.skill_id == BASELINE_SKILL_ID
                    else "旋钮数值与 EB 相同（技能按场景区分）")
            self.v_status.set(
                f"当前生效：{skill_display_name(self.profile)} · {self.profile.skill_id} | "
                f"{same}（{len(KNOB_SPECS)}/{len(KNOB_SPECS)} 项） | 来源：{src_fp}"
            )
        else:
            self.v_status.set(
                f"当前生效：{skill_display_name(self.profile)} · {self.profile.skill_id} | {describe_diff_vs_baseline(knobs)} | 来源：{src_fp}"
            )

    def on_select_preset(self):
        lab = self.v_preset.get()
        item = getattr(self, "_preset_map", {}).get(lab)
        if not item:
            return
        _sid, fp, _prof = item
        try:
            prof = load_strategy_file(fp)
            self.load_profile_into_ui(prof, announce=True)
        except Exception as exc:
            self.log(f"[FAIL] [V3.59 策略] 加载策略包失败：{exc}")

    def on_auto_summary(self):
        prof = self.collect_profile_from_ui()
        s = describe_diff_vs_baseline(prof.knobs)
        self.v_behavior.set(s)
        self.profile = prof
        self._update_status_banner()

    def on_apply(self):
        prof = self.collect_profile_from_ui()
        self.profile = prof
        set_active_strategy(prof, sync_default_json=True)
        self._active_mtime = self._active_file_mtime()
        self._sync_to_camera_tab(set_hover=True)
        self._update_status_banner()
        self.log(
            f"[OK] [V3.60 策略] 已应用技能 {prof.label()} `{prof.skill_id}`"
            + ("（仿真：下次启动仿真时生效）" if self.mode == "sim" else "") + "："
            f"悬停 {prof.get('cam_hover_verify_height_mm'):.0f} mm · "
            f"鸡蛋张开 +{prof.get('egg_open_margin_mm'):.1f} mm · "
            f"限位裕量 >={prof.get('limit_margin_required_deg'):.0f}° · "
            f"{'100% 还原原版抓鸡蛋基准 ✓' if is_baseline_equivalent(prof.knobs) else describe_diff_vs_baseline(prof.knobs)}"
        )

    def on_save(self, path=None):
        prof = self.collect_profile_from_ui()
        err = validate_new_skill(prof)
        if err:
            self.log(f"[FAIL] [V3.60 策略] 没有保存：{err}")
            return None
        target = path or os.path.join(USER_STRATEGY_DIR, f"{prof.skill_id}.json")
        try:
            saved_fp = save_strategy_file(prof, path=target)
        except Exception as exc:
            self.log(f"[FAIL] [V3.60 策略] 保存失败：{exc}（下一步：检查 {USER_STRATEGY_DIR} 的写权限）")
            return None
        self.profile = prof
        set_active_strategy(prof, sync_default_json=True)
        self._active_mtime = self._active_file_mtime()
        self._sync_to_camera_tab(set_hover=True)
        self._refresh_preset_dropdown()
        self._update_status_banner()
        self.log(f"[OK] [V3.60 策略] 用户技能 {prof.label()} 已保存 → {saved_fp}")
        return saved_fp

    def on_save_as(self, path=None):
        prof = self.collect_profile_from_ui()
        if path is None:
            from tkinter import filedialog
            init_name = f"{prof.skill_id}.json" if prof.skill_id != BASELINE_SKILL_ID else "custom_egg_skill.json"
            path = filedialog.asksaveasfilename(
                parent=self.parent,
                title="另存为新技能参数包 (.json)",
                initialdir=USER_STRATEGY_DIR,
                initialfile=init_name,
                defaultextension=".json",
                filetypes=[("JSON 技能参数包", "*.json"), ("所有文件", "*.*")],
            )
        if not path:
            return None
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem and stem != "default" and prof.skill_id == BASELINE_SKILL_ID:
            prof.skill_id = stem
            self.v_skill_id.set(stem)
        return self.on_save(path=path)

    def on_load_dialog(self, path=None):
        if path is None:
            from tkinter import filedialog
            path = filedialog.askopenfilename(
                parent=self.parent,
                title="读取技能参数包 (.json)",
                initialdir=USER_STRATEGY_DIR if os.path.isdir(USER_STRATEGY_DIR) else STRATEGY_DIR,
                filetypes=[("JSON 技能参数包", "*.json"), ("所有文件", "*.*")],
            )
        if not path:
            return
        try:
            prof = load_strategy_file(path)
            self.load_profile_into_ui(prof, announce=True)
        except Exception as exc:
            self.log(f"[FAIL] [V3.60 策略] 读取策略文件失败 ({path})：{exc}")

    def on_reset_baseline(self):
        prof = builtin_strategy_presets()[0]          # V3.60: in-memory baseline, no file is written
        prof.path = ensure_builtin_strategies(STRATEGY_DIR)
        self.load_profile_into_ui(prof, announce=False, persist=True, set_hover=True)
        self._active_mtime = self._active_file_mtime()
        self.log("[OK] [V3.60 策略] 已还原为基准技能 EB Egg Baseline (egg_grasp_v345_baseline) ✓")

    def on_copy_llm_catalog(self):
        cat = export_llm_skill_catalog(STRATEGY_DIR, active_profile=self.collect_profile_from_ui())
        txt = json.dumps(cat, ensure_ascii=False, indent=2)
        try:
            self.parent.clipboard_clear()
            self.parent.clipboard_append(txt)
        except Exception:
            pass
        self.log(f"[OK] [V3.59 策略] 已复制 LLM 技能名片目录 JSON（共 {len(cat['available_skills'])} 个技能包）到剪贴板")
        return cat

    def _selected_dry_mode_key(self):
        lab = self.v_dry_mode.get()
        for k, text in self.dry_modes:
            if text == lab:
                return k
        return "next_target"

    def on_dry_run_plan_only(self):
        self.on_apply()
        if self.cam_tab is None:
            self.v_dry_status.set("未绑定 CameraTab：请在主控制台运行。")
            return None
        mode_k = self._selected_dry_mode_key()
        plan = self.cam_tab.plan_dry_run_from_ui(
            dest_mode=mode_k,
            manual_dest_text=self.v_dry_manual.get(),
            knobs=self.profile.knobs,
        )
        self._last_dry_plan = plan
        if plan and plan.get("ok"):
            self.v_dry_status.set(f"✓ 预演规划通过：{plan['summary']}")
        else:
            self.v_dry_status.set(f"✗ 预演规划未通过：{(plan or {}).get('error', '未知原因')}")
        return plan

    def on_dry_run_execute(self):
        self.on_apply()
        if self.cam_tab is None:
            return
        mode_k = self._selected_dry_mode_key()
        self.cam_tab.on_hover_dry_run(
            dest_mode=mode_k,
            manual_dest_text=self.v_dry_manual.get(),
            knobs=self.profile.knobs,
            status_cb=lambda msg: self.v_dry_status.set(msg),
        )

    def on_dry_run_stop(self):
        if self.cam_tab is not None:
            self.cam_tab.on_stop()
            self.v_dry_status.set("已发送停止信号 ■")


# --------------------------------------------------------------------------- V3.60 SIM TAB
def add_sim_strategy_tab(tk, ttk, notebook, log=None):
    """V3.60 SIM TAB: add the same 「技能参数旋钮」 tab (without the real-robot dry-run panel) to the simulation GUI
    notebook, as its own top-level tab next to "NERO SimAgent".  Called from one line inside an existing
    `# V3.43 A2` block of the backend GUI (test_v3_44 invariants unchanged).  -> SkillStrategyTabUI"""
    outer = ttk.Frame(notebook, padding=6)
    notebook.add(outer, text="技能参数旋钮")
    ui = SkillStrategyTabUI(tk, ttk, outer, cam_tab=None, pipe=None, jpanel=None, log=log, mode="sim")
    run = os.environ.get("NERO_V360_SKILL_CODE")
    if log is not None:
        log(f"[V3.60 SKILL TAB] 本次仿真按启动时注入的技能运行：{run or '未注入（后端默认 = EB）'}"
            f"{' ' + os.environ.get('NERO_V360_SKILL_NAME', '') if run else ''}；在此页「应用」的技能下次启动仿真时生效")
    return ui
