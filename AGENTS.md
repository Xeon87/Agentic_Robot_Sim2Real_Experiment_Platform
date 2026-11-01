# NERO 项目开发与交接文档（V3.61 · Sim2Real 与自动寻优路线版）

> 放在仓库根目录，命名为 `AGENTS.md`（Codex / 后续开发 Agent 会自动读取）。
> 更新时间：2026-11-01。当前已交付最新版本：**V3.61**（仓库：`Xeon87/Agentic_Robot_Sim2Real_Experiment_Platform`）。
> 接手者请在做任何修改前通读本文件。本文件里“用户”指项目所有者 Jimmy。

## 0. 一句话现状

NERO 7 自由度机械臂（AgileX，两指夹爪）+ MuJoCo 仿真 + GPT Agent 做“鸡蛋放进蛋托”及通用操作任务，已全面进入 **Sim2Real 闭环阶段**。
- **已完成（V3.13–V3.58）**：CAN 驱动（固件 V1.2.1 `NeroFW.V121` 兼容）、ROS2 / MoveIt2、关节控制、示教平滑进出、待断电位置（Park Pose D）、3D 安全围栏（Fence + `z_table` + 路径碰撞检查）、固定主相机 D435i（Eye-to-Hand 6× 快速标定 + YOLOE-26 轮廓与局部桌面平面 3D 定位）、4 姿态碰点 TCP 标定、以及独立实机腕部广角相机模块 `NERO_V349_Real_Wrist_Camera.py`（双画面同屏、静止/清晰度门禁、TCP 准星投影、零先验 6 孔蛋托识别 `locate_egg_holder`、粗到精定位 `wrist_refine_target`、运动学锚定广角内参 + ChArUco 手眼一键联合标定）。
- **V3.59 / V3.60（阶段 1，已交付）**：`NERO_V359_Strategy.py` 45 项技能旋钮 + 技能名片 + 无夹爪多点悬停预演（V3.59）；V3.60 把默认技能 **EB** 改为严格还原 V3.58（悬停 100 mm、表格顺序 + 可信优先），加技能编号（EB/DP/HD/QH）、发布包运行时只读、仿真注入 6 个后端已有环境变量、仿真窗口技能页、每次仿真任务打分 `Y = s/(t*z)*100`。
- **V3.61（已交付）**：第三个仿真场景 `stack_cups`（叠杯子，RoboDojo；两只 16 oz 红色 party 杯）；新模块 `NERO_V361_Cup_Stack.py` 离线 MuJoCo 松手试验 + URDF 严格 6-DoF 可达性试验，得出技能 **CS**（Cup Stack）9 个旋钮（第 ⑥ 组，`wired="study"`，**尚无运行代码读取**）；技能页每个区域可折叠 + 一键全部折叠/展开；技能名称后带 `[XX]` 缩写（控制台与仿真窗口一致）。**后端未改，机械臂在仿真里还不会自动叠杯子**（待 V3.62 加后端工具，需先和用户确认接入方式）。
- **接下来的核心路线（V3.62 → V3.64，原 V3.61 → V3.63 再顺延一版）**：遵循**“一套大脑、两套身体（仿真与实机除眼睛和手脚外 100% 通用）”**原则，将算法拆分为**“铁打的底层骨架 + 可拨动的控制旋钮（Control Knobs / Strategy Profile）”**，通过**带噪仿真自动寻优试验（DOE / 贝叶斯优化）**自动搜索各场景鲁棒最优参数包，并按 `纯视觉干跑 → 5 cm 悬停预演 → 实机柔顺抓放` 稳步完成真机迁移。

## 1. 用户的沟通规则（必须遵守，原话优先）

1. **用中文、简洁地回答。** 结论先说，再给依据；不要长篇复述。
2. **“先分析，不要改” / “先别改” / “给我分析一下” → 只分析，绝不改代码。**
   用户的原话反馈：“我没让你改，你应该先分析一下，这次的 token 浪费了。”
   用户只是报告现象、没有明确说“改”时，也默认**先分析、给方案、等确认**。
3. **原话规则：“未来针对 USB-Camera 的策略改动，如果有影响到其他模式的话，要提前和我确认。”**
   （USB 相机 = 用户的 innomaker 1080P USB2.0 UVC 腕部相机，已封装在独立模块 `NERO_V349_Real_Wrist_Camera.py` 中，绝不污染仿真 `NERO_V326_Wrist_Camera.py`。）
4. **每次代码更新：所有文件放在一个平铺的文件夹里交付**（不引用外部/嵌套的其他版本文件），保证用户运行的就是他以为的那个版本，同时提供一个同名 `.zip`。
5. 用户是机械博士、做硬件/传感器出身，懂原理但不想被代码细节淹没；解释时讲“为什么”和“怎么验证”。
6. 方案有取舍时说清楚代价和风险；不确定的事情说不确定，并给出用户能做的验证步骤（命令、点哪里、看什么）。
7. 不要夸大测试结果：只在模拟 / 假数据上验证过的东西，要明确说“还没在真实硬件上验证”。

## 2. 开发纪律（必须严格执行）

### 2.1 改之前
- 先读相关代码再下结论，不凭记忆猜。改动前确认：谁调用它、哪些测试覆盖它、会不会影响其他模式（仿真 / 实机 / 腕部相机 / 固定主相机）。
- 最小改动原则：只改需求涉及的部分；**新功能优先放新模块或新函数**，不去重写稳定代码。
- 用户给的物理量（尺寸、型号、距离）直接用；不确定的硬件参数先查资料或请用户跑命令确认。

### 2.2 改的时候
- 每个改动块加**版本标记注释**，例如 `# V3.58 KINEMATIC-ANCHORED WIDE-ANGLE INTRINSICS ...`，`tests/check_release.sh` 会 grep 这些标记。
- 默认行为保持向后兼容；新行为做成可选项（下拉框 / 开关 / 策略配置项 / 环境变量）。
- 绝不修改第 4 节列出的“不变量”。
- 后台线程不直接碰 Tk：结果用 `self._ui(fn)` 放回 Tk 线程执行；Tk 变量只通过 `StringVar/BooleanVar.get/set` 读写（测试替身依赖这一点）。
- 在 `CameraTabUI.__init__` 中必须最先初始化 `self._btn_state = {}` 等基础状态字典，再调用各子面板构建函数（避免 `_build_intrinsics` 早于 `_build_calib` 调用 `_update_intr_ui` 时引发属性缺失闪退）。
- 所有给用户看的日志带版本标签：`[OK]/[INFO]/[WARN]/[FAIL] [V3.xx 模块] 中文说明`，失败信息要写“原因 + 下一步怎么做”。
- 不要创建 `tkinter.font.Font` 对象（V3.16.2 发现在 Ubuntu 22.04 / Tk 8.6 / Xft 上会导致段错误）；量文字用 `widget.tk.call("font", "metrics"/"measure", ...)`。
- `import cv2` 一律走 `NERO_V325_Camera._import_cv2()`（V3.46 Qt 变量保护，见 4.4）。
- 控制台启动的任何子进程都要用 `clean_child_env()` 的环境（见 4.4）。

### 2.3 测试
- **每个版本写一个新测试文件** `tests/test_v3_XX.py`（有界面改动再加 `tests/test_v3_XX_gui.py`），文件开头的 docstring 逐条列出测试项，每项通过时打印状态，最后打印 `ALL V3.XX TESTS PASSED`。
- 新测试加进 `tests/check_release.sh`：非 GUI 测试加到大 for 循环；GUI 测试加到末尾 `| tail -1` 列表；并加一行 grep 新版本的标记。
- 版本号变了，要把老测试里的版本白名单扩展（如 `"V3.57"` 扩展为 `"V3.57", "V3.58"`）。
- 行为有意改变时，**更新老测试并在 CHANGELOG 里说明**，不要删测试来让它通过。
- 测试**绝不能读写用户的真实目录**（`~/Downloads/cameracalib`、`~/Downloads/objectmodels`）：GUI 测试开头要改 `CAM.CALIB_PATH / CAM.LOCATE_PATH / CAM.CALIB_DIR` 到临时目录。
- `check_release.sh` 必须 `rc=0` 才能交付。

### 2.4 交付流程（逐条照做）
1. 在一个**新的平铺文件夹** `NERO_V3_XX` 里工作，从上一个版本复制过来。
2. 更新：`VERSION` 文件、`NERO_V313_Real_Robot.py` / `NERO_V312_Console.py` / `NERO_V347_Yolo_Perception.py` / `NERO_V349_Real_Wrist_Camera.py` 里的 `VERSION = "..."`；`README.md` 与 `CHANGELOG.md` 顶部加本版说明。
3. `chmod +x *.sh tests/*.sh`。
4. 删除 `__pycache__/` 和 `logs/`。
5. 重新生成校验和（V3.60 起扫描 3 层，包含 `config/strategies/*.json`）：`sha256sum $(find . -maxdepth 3 -type f ! -name "SHA256SUMS.txt" ! -path "./.git/*" ! -path "*/__pycache__/*" ! -path "./logs/*" | sed 's|^\./||' | sort) > SHA256SUMS.txt`。
6. 跑 `bash tests/check_release.sh`，必须 `rc=0`（跑完若产生 `__pycache__`，再次清理并刷新 `SHA256SUMS.txt`）。
7. 从工作目录打包 `NERO_V3_XX.zip`，保留 `.sh` 可执行权限。

## 3. 用户的硬件和环境（已确认的事实）

- **电脑**：Ubuntu 22.04，笔记本 RTX 4060 Laptop 8 GB。控制台用 `/usr/bin/python3`（3.10.12），包在 `~/.local/lib/python3.10/site-packages`：torch 2.14.1+cu130（CUDA 可用）、numpy 1.26.4、cv2 4.11.0（pip 版 opencv-python）、ultralytics 8.4.174、clip、pyrealsense2。
- **机械臂**：NERO 7DOF，**固件已升级至 V1.2.1（`NeroFW.V121`）**，USB-CAN（`can0`），AgileX `agx_arm_ros` / `pyAgxArm`；ROS2 Humble + MoveIt + cuRobo（GPU planner 独立进程）。
  - V1.2.1 固件退出示教模式时，必须在 `enable()` 前后调用 `set_follower_mode()` 并发送当前关节角保持指令（V3.54 已修复）。
- **固定主相机（Eye-to-Hand）**：Intel RealSense D435i，序列号 `261222075224`，距桌面约 0.71–0.80 m，光轴与桌面法线约 38°–43°。
  - 默认深度流自 V3.53 起设为 **`1280×720 · High Accuracy`**（手眼标定与视觉定位一致）。
  - 主相机法兰 ArUco 标记：`DICT_4X4_50 id 0`，打印实测 `38.5 × 37.5 mm`（主相机手眼标定默认边长 `40 mm` 或按实测填 `38 mm`）。
- **实机腕部相机（Eye-in-Hand）**：InnoMaker `U20CAM-1080P`（USB2.0 UVC，`1920×1080 @ 30fps` MJPG，32×32 mm 板，标准 `M12×0.5` 螺纹镜头座）。
  - 原装镜头为 `1.8 mm`（对角 130° / 水平 103° 超广角），在 `1920×1080` 下真实物理焦距 `fx ≈ 680–740 px`；因 130° 边缘桶形畸变大且画面下方易拍进黑色夹爪指尖，用户后续可换拧 `3.6 mm`（水平 ~82°）或 `2.8 mm` M12 无畸变镜头。
  - 腕部标定物：桌面平铺 A4 打印 `7×5 ChArUco 标定板`（方格 `32 mm`，内部 ArUco 码外沿实测约 `23.8–24.0 mm`，`DICT_4X4_50`）。注意需抽出透明塑料文件袋、四边用胶带拉平贴紧桌面以消除反光和纸面鼓包。
- **YOLO 模型**：`~/Downloads/objectmodels/` 里有 `yoloe-26s-seg.pt`、`yoloe-26s-seg-pf.pt`、`mobileclip2_b.ts`。
- **鸡蛋与桌面**：白色鸡蛋，实测长约 `55 mm`、宽约 `42 mm`。

## 4. 不变量（绝不能改；测试会强制检查）

1. **`NERO_V308_Camera_Perception.py` 一字不改**，sha256 前缀 = `020cc6680f181872`（所有版本测试均校验）。
2. **后端 `run_sim` 的逐步物理循环“B 段”**（从 `with mujoco.viewer.launch_passive(model, data) as viewer:` 到 `run_sim` 结束）与 V3.42 字节级一致，sha256 = `2b1f124e838087f65a4e93e65ed1ceba169ea3c725bbea99ff87998ad8cd7252`（`test_v3_44` C1）。
3. **后端去掉 11 个 `# V3.43 A2 BEGIN/END` 块后与 V3.42 字节级一致**（`test_v3_44` C2）；默认鸡蛋场景生成的 XML sha256 = `31c52c06965826a06612c970d1b522a05396d9da9b5c9d0919c8824c832a3ef2`（C3）。如果新阶段需要让仿真读取外部策略包（`StrategyProfile`），必须通过环境变量/配置文件钩子在 A2 块或独立桥接模块中非侵入式注入，或者同步维护 C2 块标记，绝不破坏 `test_v3_44` 校验。
4. **Qt 环境保护（V3.46）**：控制台进程里 `import cv2` 只能走 `NERO_V325_Camera._import_cv2()`；所有子进程用 `clean_child_env()`。
5. **实机腕部相机独立性**：实机腕部相机逻辑全部位于 `NERO_V349_Real_Wrist_Camera.py` 与 `NERO_V325_Camera.py`，绝不改动仿真腕部相机 `NERO_V326_Wrist_Camera.py`。
6. **发布包运行时只读（V3.60）**：程序运行时绝不写发布文件夹里的任何文件（`config/strategies/` 只读）；当前技能写 `logs/v360_active_strategy.json`，用户技能写 `~/Downloads/cameracalib/strategies/`（`NERO_V360_USER_STRATEGY_DIR`）。`test_v3_60` F6 / `test_v3_60_gui` H6 会校验。
7. **默认技能 EB = 原逻辑（V3.60）**：`BASELINE_KNOBS` 代入后必须与后端 / V3.58 原代码零偏差（`test_v3_60` F2 从后端源码 AST 抽取原函数比对）；改默认值等于改行为，要先和用户确认。

## 5. 代码地图（截至 V3.60）

| 文件 | 作用 |
| :--- | :--- |
| `start_nero.sh` | 入口：打开控制台（`--sim` 直接跑仿真，`--real` 打开实机页）；检查本文件夹里所有模块存在 |
| `start_nero_sim.sh` | 仿真链路启动（MuJoCo 后端 + cuRobo + ROS2/MoveIt + Agent），校验 `NERO_SCENE` |
| `NERO_V312_Console.py` | Tk 控制台：仿真 Tab、实机 NERO 调试 Tab、Camera 调试 Tab；底部 9 步全流程状态灯栏（`WorkflowLampBar`）与示教模式快捷按钮 |
| `NERO_V313_Real_Robot.py` | 实机六步启动（CAN → 使能 → 驱动 → ROS2 → MoveIt）、V1.2.1 固件兼容（`_resolve_nero_fw`）、`set_follower_mode` 示教退回修复、`move_and_settle` |
| `NERO_V314_Joint_Control.py` | 关节面板、示教模式、待断电位置 `go_park`（Park Pose D，自动退示教 + 临时移除后墙碰撞 + 忽略 XY 围栏） |
| `NERO_V315_Consistency_Safety.py` | URDF/FK/IK（`CS.fk`、`CS.ik_tool`）、一致性和 3D 安全围栏路径检查 |
| `NERO_V316_Safety_Editor.py` | 安全区（fence 多边形 + `z_table`）可视化编辑器 |
| `NERO_V325_Camera.py` | **Camera 调试 Tab**：主从相机双 `560×315` 16:9 实时监视画布、主相机手眼标定（V3.55 雅可比快速采样 + PnP/深度统一 + LOO 剔除）、4 姿态碰点 TCP 标定、腕部内参标定子页、腕部 Eye-in-Hand 标定子页、视觉定位与悬停预演子页 |
| `NERO_V347_Yolo_Perception.py` | **实机主相机感知模块**：YOLOE-26 视觉/文字提示 + 无提示 PF 多类别检测、ROI 快速局部桌面拟合、`SilhouetteFit`、3D 椭球轮廓定位 |
| `NERO_V349_Real_Wrist_Camera.py` | **实机腕部相机模块（V3.49–V3.58）**：UVC 相机枚举与曝光/白平衡/对焦锁定、ChArUco/棋盘格检测与运动学锚定内参标定（`calibrate_intrinsics`）、24 角点 ChArUco 6-DoF 检测（`detect_charuco_pose`）、自适应四象限大倾角内参采样规划（`plan_intr_auto_targets`）、带杆臂约束与 2D 尺度自标定的 Eye-in-Hand 求解器（`calibrate_eye_in_hand`）、粗到精闭环（`wrist_refine_target`）、零先验蛋托识别（`locate_egg_holder`） |
| `NERO_V359_Strategy.py` | **技能参数中心（V3.59–V3.60）**：`KNOB_SPECS` 45 项（每项 `wired` = `sim_env` 6 / `real` 8 / `none` 31）、`StrategyProfile`（skill_id + `short_name` + 两字母 `code` + `skill_card` + `knobs`）、内置技能 EB/DP/HD/QH、与后端等价的公式（`score_grasp_candidate`、`compute_gripper_prior`、手腕评分、`order_empty_slots`、`rank_detected_targets`）、`strategy_sim_env`（只注入与基准不同的 6 个环境变量）、多点悬停预演规划 / 执行、`SkillStrategyTabUI`（控制台 real 模式 / 仿真窗口 sim 模式）、`add_sim_strategy_tab` |
| `NERO_V361_Cup_Stack.py` | **叠杯子离线试验（V3.61）**：`release_trial` / `release_sweep`（只有杯子的 MuJoCo 松手试验）、`choose_release_params`、严格 6-DoF `pose_ik`（`CS.ik_tool` 会丢姿态，别用它算姿态）、`rim_pinch_pose`、`grasp_reachability`、`plan_cup_stack`（5 个航点）、`run_study`；结果存 `config/cup_stack_study_v361.json`（静态文件，运行时不写） |
| `config/strategies/` | 内置技能 JSON（只读，与 `builtin_strategy_presets()` 完全一致）：`egg_grasp_v345_baseline`（EB，`default.json` 是它的副本）、`dense_cluster_anti_poke`（DP）、`hover_dry_run_safe`（HD）、`quick_hover_verify`（QH） |
| `NERO_V308_Camera_Perception.py` | V3.08–V3.11 几何鸡蛋感知（仿真和“几何法”在用）——**字节级冻结** |
| `NERO_V326_Wrist_Camera.py` / `NERO_V331_Fixed_Camera.py` | 仿真里的腕部相机 / 固定相机模式 |
| `NERO_V343_Scenes.py` | 仿真场景数据（`fill_egg_holder` / `store_in_safe` / V3.61 `stack_cups`：`CUP`、`CUP_XY`、`cup_body_xml`、`make_stack_cups_xml`、`cup_task_state`）；`SCENES_REV="V3.61"`，`VERSION` 保持 V3.44 |
| `NERO_V2_7_1_Unified_Backend.py` | MuJoCo 仿真后端（含抓取/抬升/转腕/分层入槽/回撤状态机、规划调用、`run_sim`）；V3.60 只在一个已有 A2 块里加了仿真技能页的一行调用 |
| `NERO_V2_7_1_Agent.py` | GPT / VLM Agent（技能契约调用与异常恢复决策）；V3.60 任务结束打印 `[V3.60 SKILL SCORE]` 并写 `logs/v360_skill_scores.jsonl` |
| `NERO_V71_Hybrid_GPU_Planner_Core.py` | cuRobo + MoveIt2 混合运动规划核心 |
| `tests/` | `test_v3_02` 至 `test_v3_60` 全套离线/GUI 单测 + `check_release.sh` |

## 6. 实机双相机标定与感知关键设计（V3.47.2 – V3.58 演进总结）

### 6.1 主相机手眼标定提速与精度优化（V3.55）
- **雅可比快速逆解规划**：`plan_auto_targets` 用阻尼最小二乘雅可比伪逆替代慢速数值 IK，自动采集规划耗时从 ~3.5 s 降至 `< 0.5 s`，同时强制校验轨迹路径不越安全围栏。
- **混合深度/PnP 尺度统一**：D435i 在不同倾角下可能部分姿态用 IR 深度、部分姿态退回 RGB PnP（当打印码填 40 mm 而实测 38.5 mm 时会产生 ~30 mm 阶跃）。`calibrate()` 自动将全批次统一为同一观测源并用中位数深度比值对齐尺度，结合留一法（LOO）自动剔除坏点。

### 6.2 腕部相机内参 + Eye-in-Hand 联合标定（V3.56 – V3.58）
1. **130° 广角桌面标定的三大物理痛点与 V3.58 根治方案**：
   - **痛点 A（近似正视平面奇异性导致 `fx` 飙升至 `2147~5164 px`）**：标定板平放桌面且倾角较小时，张氏平面标定中焦距 $f$ 与距离 $t_z$ 共线（$f/t_z \approx \text{const}$）。**V3.58 方案**：在 `calibrate_intrinsics` 中引入**机械臂正运动学（`T_base_flange`）3D 位移锚定**，在广角物理焦距区间 `[0.31*w, 0.78*w]`（`600~1500 px`）内联合最小化 2D 重投影 RMS 与 3D 手眼位移残差，直接锁死真实广角焦距（`~705 px`，桌面距离 `~250 mm`）。
   - **痛点 B（画面下沿黑色夹爪阴影与塑封膜反光拉偏角点）**：**V3.58 方案**：开启 `CORNER_REFINE_SUBPIX` + `tryRefineMarkers`，并通过**干净子集去畸变 + 平面 RANSAC 单应性检验（`_clean_outlier_corners`）**自动剥离每张图里偏离 `> 1.15 px` 的 1–3 个孤立反光/夹爪遮挡坏角点（保留其余干净角点），而对 `>= 4` 个坏角点的运动模糊帧则整张剔除。
   - **痛点 C（起点偏左导致采样集中在左侧 + 纸面微翘）**：**V3.58 方案**：`plan_intr_auto_targets` 读取起点标定板实际位置 `s0_board["center_frac"]` 作为最远点采样种子，将四向及对角倾视基线提升至 `18°–28°`；`calibrate_eye_in_hand` 将 2D 深度比例因子 `alpha_z` 拓宽至 `[0.12, 1.85]`（以 `axy=1.0` 起算），并在自动采集达标后**一键同步保存 `wrist_intrinsics.json` 与 `wrist_handeye_calib.json`**。

## 7. 版本简史（V3.43 – V3.61）

| 版本 | 核心改动内容 |
| :--- | :--- |
| V3.43 – V3.46 | 场景启动前选择（`NERO_SCENE`）；主相机手眼标定四步状态灯；OpenCV Qt 插件导致 RViz 崩溃修复（`clean_child_env`、`_import_cv2`） |
| V3.47 / V3.47.1 | YOLOE-26 视觉/文字提示定位 + 椭球轮廓与局部桌面平面 3D 拟合（解决 D435i 白鸡蛋深度压扁问题）；标定与定位配置持久化及动态校验 |
| V3.47.2 | Treeview 表格自适应行高与列宽、画面标签防重叠避让、ROI 局部平面拟合 4× 提速（~0.25 s）、YOLO GPU/CPU 设备日志 |
| V3.48 | 4 姿态碰点法 TCP 标定（`calibrate_tcp`）、目标正上方悬停预演（`hover_above_target`）、PF 开放词汇多物体定位、清除标定按钮 |
| V3.49 | 新增独立实机腕部相机模块 `NERO_V349_Real_Wrist_Camera.py`：UVC 参数锁定、内参标定、Eye-in-Hand 手眼标定、零先验 6 孔蛋托识别 `locate_egg_holder`、自动采集路径安全区检查 |
| V3.50 | 主从相机双 `560×315` 实时画布、腕部静止/清晰度/过曝质量门禁、TCP 准星与 `r=21mm` 包络圆投影、粗到精定位 `wrist_refine_target` |
| V3.51 | 腕部独立 ArUco 设置（默认 24 mm）、可见性感知自动 Eye-in-Hand 采集、控制台底部 9 步全流程状态灯栏（①~⑨） |
| V3.52 | 8 参数 `CALIB_RATIONAL_MODEL` PnP 完整畸变透传、抗 IPPE 翻转的点对偶 Eye-in-Hand 求解器、非阻塞后台采集、示教按钮右对齐 |
| V3.53 | D435i 默认深度流升级为 `1280×720 High Accuracy`、V1.11 示教退出修复、待断电位置（Park Pose D）自动退示教与 XY 围栏豁免 |
| V3.54 | 适配 NERO **V1.2.1 固件（`NeroFW.V121`）**：退出示教前后强制 `set_follower_mode()` + 关节保持，修复退示教掉使能问题 |
| V3.55 | 主相机手眼标定 6× 提速（雅可比快速逆解 + 6 帧早停 + 混合 PnP/深度自动对齐 + LOO 坏点剔除） |
| V3.56 | 腕部 Eye-in-Hand 标定优化：纯平移初值探测 + 穹顶采样规划 + 全分辨率 ROI 亚像素 ArUco + 160 mm 杆臂约束与单轴尺度自标定 |
| V3.57 | 桌面 ChArUco 标定板全自动内参采集（`▶ 自动采集`）+ 24 角点 ChArUco 腕部手眼标定 + 8 参数防过拟合门禁 + 2D 各向异性尺度自标定 |
| **V3.58** | **机械臂运动学（FK）锚定广角内参求解器**（根治平面小倾角下 `fx=2147~5164 px` 退化）+ **RANSAC 单应性孤立坏角点剥离**（消除夹爪阴影/反光拉偏）+ **多帧亚像素稳像平均** + **内参/手眼一键双保存** |
| V3.59 | `NERO_V359_Strategy.py` 45 项技能旋钮 + 技能名片 + 「技能参数旋钮」页 + 无夹爪多点悬停预演；主相机标定标签改用结果自带 `good` |
| **V3.60** | 默认技能 EB 严格还原 V3.58（悬停 100 mm、表格顺序 + 可信优先，V3.59 默认值改为 QH）；技能编号 EB/DP/HD/QH；发布包运行时只读；仿真注入 6 个后端环境变量（只注入差异）；仿真窗口技能页；仿真任务打分 Y；腕部 USB 悬停值不跟随技能 |
| **V3.61** | 第三个仿真场景 `stack_cups` 叠杯子；离线松手 / 可达性试验 → 技能 CS + 第 ⑥ 组 9 个旋钮（离线试验✓，未接入运行代码）；技能页分区折叠 + 一键全部折叠/展开；技能名称后带缩写；后端 / Agent / Camera 页未改 |

## 8. 后续开发计划（V3.59–V3.61 已交付；V3.62 → V3.64：带噪仿真寻优与真机迁移，原编号顺延一版）

### 8.0 核心设计哲学（已与用户对齐确认）
1. **一套大脑，两套身体（仿真与实机 100% 逻辑通用）**：
   - 除底层的“眼睛（MuJoCo 渲染相机 vs 实机 D435i + 腕部相机）”和“手脚（MuJoCo 物理步进 vs 实机 MoveIt + 夹爪）”外，中间层的**目标选优、姿态规划、分阶段下探轨迹、异常恢复状态机**在仿真与实机之间 **100% 共用同一份代码**。
2. **铁打的骨架 + 可拨动的控制旋钮（Control Knobs / Strategy Profile）**：
   - **固定骨架（写死在底层）**：空间坐标变换、正逆运动学（FK/IK）、3D 安全围栏与桌面高度 `z_table` 防撞底线、基本动作工序（全局扫描 → 悬停 → 腕部精修 → 慢速下探 → 柔顺闭合 → 抬升 → 转腕 → 入槽）。
   - **控制旋钮（抽离为结构化 JSON 策略包）**：将所有与场景/物体相关的排序权重、姿态偏好权重、悬停高度、夹爪张合裕量、分段速度比抽离为 12 个标准旋钮，保存在 `config/strategies/*.json` 中，仿真和实机直接共读同一份策略文件。
3. **拒绝低效人工调参：用「带噪仿真最优化试验」自动寻找最优旋钮组合**：
   - 用户不希望靠人工拍脑袋去试参数，而是通过**在仿真中注入真实传感器噪声与尺寸扰动（Domain Randomization，防止仿真过拟合）**，利用 **拉丁超立方正交试验（LHS/DOE）+ 贝叶斯优化 / 进化策略** 在仿真里自动跑 30–50 轮试验，自动算出针对特定场景的**鲁棒最优参数包（Robust Optimal Profile）**，一键存盘供实机直接调用。
4. **与具身智能（Embodied AI）的融合**：
   - 这套控制旋钮既支持通过仿真试验自动寻优，也作为 VLM / Agent 大脑操控物理世界的参数化接口（Parameterized Skill API）——面对新物体或新场景时，Agent 可根据物体语义（易碎/坚硬、密集/稀疏）自动匹配策略包，并在抓空/滑落时通过自反思（Self-Reflection）微调旋钮参数。

### 8.1 阶段 1（V3.59 + V3.60，已实现）：

> **V3.60 实现说明（给接手者）**：
> - 默认技能 EB 与原代码零偏差（`test_v3_60` F2）；V3.59 写在本节下方的 `cam_hover_verify_height_mm=60.0` 已改为 **100.0**，新增 `target_order_mode=table_trusted_first`（默认）。
> - 仿真真正生效的只有 6 项（`SIM_ENV_KNOBS`）；其余旋钮的仿真逻辑仍写死在后端（受 `test_v3_44` 保护），要接入只能在 A2 块 / 独立桥接模块里非侵入地做，或经用户同意重做 `test_v3_44` 基准。
> - 技能目录目前**没有**注入 `observe_scene`（V3.59 文档写的“已注入”不准确），仅「复制 LLM 技能目录」按钮使用。
> - 打分 Y 是用户给的临时公式：s 用小数、t 分钟、z = LLM 往返次数（`[LLM RESPONSE #n]` 的个数）。

（以下为 V3.59 原文）100% 还原原版基准的「技能参数旋钮」+「技能名片 (Skill Card)」+ 无夹爪多点悬停预演 (Dry-Run)

- **核心原则（Baseline 零退化铁律）**：所有拆解出的控制旋钮默认值（保存在 `config/strategies/egg_grasp_v345_baseline.json`，GUI 启动默认加载）代入后，**必须 100% 严格还原 V2.7 ~ V3.48 已经调好的抓鸡蛋逻辑（零数值偏差）**。
- **核心模块（`NERO_V359_Strategy.py` + `NERO_V312_Console.py` + `NERO_V325_Camera.py` + `NERO_V2_7_1_Unified_Backend.py`）**：
  1. **5 大组共 35 项精确还原旋钮（`KNOB_SPECS` & `BASELINE_KNOBS`）**：
     - **组 ① 候选抓取方向与防碰打分 (`grasp_scoring`)**：`horiz_proj_min_strength=0.25`、`wall_near_margin_mm=30.0`、`w_basket_clearance=1.0`、`w_yaw_rank_penalty=4.0`、`w_center_offset_penalty=0.60`、`neighbor_safe_clearance_mm=3.0`、`w_neighbor_intrusion=0.80`、`neighbor_penalty_cap=-10.0`、`target_order_mode="confidence_then_nearest"`。
     - **组 ② 夹爪预张开裕量与两段慢速下探 (`gripper_approach`)**：`egg_open_margin_mm=16.0`（合格带 `-4.0 ~ +6.0 mm`）、`sphere_open_margin_mm=20.0`、`box_open_margin_mm=15.0`、`poke_retry_bonus_mm=8.0`、`safety_retry_bonus_mm=15.0`、`pregrasp_hover_height_mm=120.0`、`grasp_z_offset_mm=28.0`、`grasp_z_settle_comp_mm=2.5`、`final_approach_dist_mm=20.0`、`final_approach_slow_factor=3.5`。
     - **组 ③ 逆运动学姿态筛选与手腕翻转权重 (`ik_posture`)**：`limit_margin_required_deg=8.0`、`limit_margin_relaxed_deg=4.0`、`limit_margin_bucket_deg=4.0`、`wrist_weight_j1_j3=8.0`、`wrist_weight_j4=4.0`、`wrist_weight_j5=3.0`、`wrist_weight_j6_j7=1.0`、`wrist_proximal_sum_limit_deg=22.0`。
     - **组 ④ 蛋托对齐入槽与邻蛋防扰动门限 (`placement_recovery`)**：`slot_fill_order="numeric_1_to_8"`、`slot_entry_above_release_mm=55.0`、`slot_capture_depth_mm=8.0`、`slot_xy_tol_mm=7.0`、`slot_axis_tol_deg=10.0`、`release_stage1_open_mm=10.0`、`bystander_warn_mm=3.0`、`bystander_abort_mm=8.0`、`object_stale_pos_mm=8.0`。
     - **组 ⑤ 实机悬停复核与无夹爪多点预演 (`hover_dry_run`)**：`cam_hover_verify_height_mm=60.0`、`cam_hover_soft_limit_deg=3.0`、`cam_hover_max_speed_pct=10.0`、`dry_run_transit_z_lift_mm=40.0`、`dry_run_dwell_s=1.2`、`dry_run_wrist_refine=True`、`dry_run_return_home=True`。
  2. **结构化自然语言「技能名片（`skill_card`）」**：
     - 每个 `.json` 策略文件内含 `skill_card`（`name`、`when_to_use`、`when_not_to_use`、`behavior_summary`、`scene_tags`、`sim_benchmark`），支持一键对比 `BASELINE_KNOBS` 自动生成中文行为差异描述，并通过 `export_llm_skill_catalog()` 注入 `observe_scene["strategy_skills"]` 供 LLM 选策调用。
  3. **无夹爪多点悬停预演（Hover Dry-Run）**：
     - 支持在「技能参数旋钮」Tab 与「Camera调试 → ④ 视觉定位」直接执行 `源物体 A 悬停` $
ightarrow$ `跨点安全抬升 (+40 mm)` $
ightarrow$ `下一目标 / 蛋托孔位 / 手动坐标 B 悬停` $
ightarrow$ `回起点姿态`，方便在夹爪未就绪时完整验证视觉定位与跨点搬运动线。

### 8.2 阶段 2（V3.62 起，原 V3.60 → V3.61 → 因叠杯子场景再顺延；V3.62 先做叠杯子后端技能，需用户确认）：带噪仿真自动最优化试验台（Sim-in-the-Loop Auto-Optimizer）

- **核心目标**：在仿真中引入真实物理噪声（防止仿真过拟合），用科学试验设计（DOE/LHS）与贝叶斯/进化寻优算法自动跑批试验，为不同场景自动找出最优控制旋钮组合。
- **新增与修改模块**：
  1. **新建 `NERO_Sim_Optimizer.py`（带噪仿真寻优引擎）**：
     - **域随机化噪声注入器（Domain Randomization）**：在每次试验的感知输出与物理交互中注入可配置的真实毛刺——目标 3D 位置高斯噪声 $\mathcal{N}(0, 2.0\text{ mm})$、偏航角噪声 $\mathcal{N}(0, 1.5^\circ)$、鸡蛋长短轴尺寸随机浮动 $\pm 1.5\text{ mm}$、蛋托初始位置/角度微扰。
     - **综合目标评价函数 $J(\theta)$**：
       $$J(\theta) = w_{\text{succ}} \cdot \text{成功率(\%)} + w_{\text{safe}} \cdot \text{最小邻蛋安全净空(mm)} - w_{\text{time}} \cdot \text{单颗平均耗时(s)} - w_{\text{joint}} \cdot \text{关节总移动代价(rad)}$$
       内置两档寻优偏好：「稳健优先（Safety-First）」与「效率优先（Speed-First）」。
     - **两阶段自动寻优算法**：
       - **Phase A（拉丁超立方正交抽样 LHS，12–16 轮）**：均匀覆盖 12 维旋钮空间，自动计算各旋钮对成功率与耗时的敏感度排序；
       - **Phase B（高斯过程贝叶斯优化 / 自适应进化 CMA，20–35 轮）**：围绕 Phase A Top-3 种子自动向高分区域收敛，每个候选参数在多组随机噪声种子下评估均值与最差表现。
  2. **控制台寻优工作台 UI**：
     - 选择目标场景（如：`标准稀疏抓蛋`、`密集贴靠抓蛋`、`边缘极限位置抓蛋`），设置试验轮数，点击 **「▶ 启动自动寻优」**；
     - 实时显示每轮试验得分、当前最优旋钮组合，并在完成后自动保存为 `config/strategies/<scene>_optimal.json`。
- **阶段 2 验收标准**：
  - 给定带 `±2 mm` 视觉噪声的测试场景，寻优器在 30–40 轮内收敛，产出的 `<scene>_optimal.json` 在带噪测试集上的成功率与安全净空显著优于初始默认参数；`tests/test_v3_60.py` 全部通过。

### 8.3 阶段 3（V3.63，原 V3.61 → V3.62）：实机「主从视觉 + 5 cm 悬停预演（Hover Dry-Run）」安全闭环

- **核心目标**：建立虚实统一技能桥接器（`RealSkillBridge`），加载阶段 2 算出的最优策略包，在不触碰真实鸡蛋的前提下，完成“主相机粗定位 → 5 cm 悬停 → 腕部相机精修对齐 → 蛋托孔位 5 cm 悬停复核”的全自动预演闭环。
- **新增与修改模块**：
  1. **新建 `NERO_Real_Skill_Bridge.py`（实机统一技能执行桥）**：
     - 对上提供与仿真完全一致的技能接口（`scan_scene`、`plan_grasp`、`approach_and_refine`、`execute_pick`、`orient_and_approach_slot`、`place_in_slot`、`verify_state`）；
     - 对下串联 `NERO_V347`（D435i 主相机全局定位）、`NERO_V349`（腕部相机 `wrist_refine_target` 精修与 `locate_egg_holder` 6 孔蛋托识别）、`NERO_V315/V316`（3D 安全围栏 + `z_table` 硬限位检查）与 `NERO_V313`（`move_and_settle` + `_wait_still` 停稳确认）。
  2. **实机三级安全运行开关（Console 实机任务面板）**：
     - **档位 1（纯视觉规划预览）**：主从相机识别鸡蛋和蛋托 6 孔，按当前 `StrategyProfile` 计算抓取优先级、最佳抓持偏航角和目标槽位，在画面上画出完整规划结果，机械臂完全不动；
     - **档位 2（5 cm 悬停预演 Hover Dry-Run，默认安全档）**：机械臂自动运动到目标鸡蛋正上方 `hover_height_mm`（如 `50–60 mm`）处停稳 → 腕部相机拍照执行 `wrist_refine_target` 微调水平位置使 TCP 准星精确对准鸡蛋中心 → 随后平移到蛋托目标槽位正上方 `50–60 mm` 悬停对准 → 回安全位。**全程不进入最后 5 cm 下探区、不合夹爪**；
     - **档位 3（实机正式抓放）**：需通过档位 2 复核确认后方可解锁。
- **阶段 3 验收标准**：
  - 实机桌面随意摆放鸡蛋（或假蛋/乒乓球）与 6 孔蛋托，运行「档位 2（5 cm 悬停预演）」，机械臂平顺完成粗定位接近与腕部精修，腕部实时画面中 TCP 准星与真实鸡蛋中心、蛋槽中心偏差 `≤ 2.0 mm`，全程零越界、零碰撞；`tests/test_v3_61.py` 全部通过。

### 8.4 阶段 4（V3.64，原 V3.62 → V3.63）：实机柔顺抓放全流程 + Agent 智能策略调度

- **核心目标**：打通最后 5 cm 慢速垂直下探、真实夹爪柔顺闭合、双重抓持状态校验与失败自动补抓，实现真机连续抓鸡蛋入托，并支持 VLM Agent 根据桌面场景自动调度/微调策略包。
- **新增与修改模块**：
  1. **真实夹爪柔顺控制与双重抓持校验（`verify_state`）**：
     - 下探前按视觉实测鸡蛋短轴宽度 $W_{\text{egg}}$ 张开至 $W_{\text{egg}} + \text{grip\_open\_margin\_mm}$（避免碰翻邻蛋）；
     - 到达抓取高度停稳后，按 $W_{\text{egg}} - \text{grip\_close\_squeeze\_mm}$ 限速柔顺闭合；
     - 垂直抬升至悬停高度后，执行**双重抓持判定**：① 检查夹爪编码器实际停住宽度是否落在 `[min_held_width_mm, W_egg + 2mm]` 区间（若闭合到底 `< 28 mm` 则立即判为空抓）；② 相机复核原桌面位置是否已无鸡蛋。若抓空且未超 `max_retry_per_target`，自动触发回退重抓闭环。
  2. **分层垂直入槽与柔顺释放**：
     - 复用仿真验证成熟的转腕对齐与分层垂直下降逻辑（`V3.02/V3.03`），在蛋槽正上方完成腕部偏航角对齐后，以 `descent_speed_ratio` 慢速垂直入槽、张开夹爪释放并垂直回撤至安全高度。
  3. **VLM Agent 场景自适应策略匹配与自反思记忆**：
     - Agent 在任务开始时根据桌面视觉特征（目标类别、密集程度、距边界距离）自动从 `config/strategies/` 中选择最匹配的仿真寻优策略包（或动态微调旋钮），并在实机执行日志中记录每次抓放的耗时与成败，持续扩充策略经验库。
- **阶段 4 验收标准**：
  - 实机连续抓取多颗鸡蛋（先用假蛋验证 10 次，再用真鸡蛋验证）稳健放入 6 孔蛋托，不磕碰蛋托外壁、不伤蛋壳；改变鸡蛋密集度或更换其他抓取物时，切换/自动匹配对应策略包即可稳定复现；`tests/test_v3_62.py` 全部通过。

## 9. 测试环境的坑（给后续开发者的提醒）

- 在没有 `tkinter` / `DISPLAY` 的云端容器环境里，独立 `test_v3_XX_gui.py` 会打印 `skipped` 并返回 0；因此必须在非 GUI 单测（如 `test_v3_57.py` X5、`test_v3_58.py` Y5）中同时包含基于 `unittest.mock` 的 `run_console('sim')`、`run_console('real')` 和 `CameraTabUI` 交互回归测试，确保任何 UI 属性初始化顺序问题在无显示器环境下也能被 100% 捕获。
- 真机 GUI 显示效果（行高、列宽、双画布缩放）最终以用户在 Ubuntu 22.04 实机上运行 `bash tests/check_release.sh` 及实测截图为准。
- 不要在测试中联网下载模型；`ultralytics` 在单测中统一使用 `tests/v347_fake.py` 里的假模块（`F.install()` / `F.uninstall()`）。
