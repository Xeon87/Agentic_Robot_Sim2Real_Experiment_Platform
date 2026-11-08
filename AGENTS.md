# NERO 项目开发与交接文档（V3.82 · Sim2Real 与自动寻优路线版）

> 放在仓库根目录，命名为 `AGENTS.md`（Codex / 后续开发 Agent 会自动读取）。
> 更新时间：2026-11-05。当前已交付最新版本：**V4.6.1**（仓库：`Xeon87/Agentic_Robot_Sim2Real_Experiment_Platform`）。
> 接手者请在做任何修改前通读本文件。本文件里“用户”指项目所有者 Jimmy。

## 0. 一句话现状

NERO 7 自由度机械臂（AgileX，两指夹爪）+ MuJoCo 仿真 + GPT Agent 做“鸡蛋放进蛋托”及通用操作任务，已全面进入 **Sim2Real 闭环阶段**。
- **已完成（V3.13–V3.58）**：CAN 驱动（固件 V1.2.1 `NeroFW.V121` 兼容）、ROS2 / MoveIt2、关节控制、示教平滑进出、待断电位置（Park Pose D）、3D 安全围栏（Fence + `z_table` + 路径碰撞检查）、固定主相机 D435i（Eye-to-Hand 6× 快速标定 + YOLOE-26 轮廓与局部桌面平面 3D 定位）、4 姿态碰点 TCP 标定、以及独立实机腕部广角相机模块 `NERO_V349_Real_Wrist_Camera.py`（双画面同屏、静止/清晰度门禁、TCP 准星投影、零先验 6 孔蛋托识别 `locate_egg_holder`、粗到精定位 `wrist_refine_target`、运动学锚定广角内参 + ChArUco 手眼一键联合标定）。
- **V3.59 / V3.60（阶段 1，已交付）**：`NERO_V359_Strategy.py` 45 项技能旋钮 + 技能名片 + 无夹爪多点悬停预演（V3.59）；V3.60 把默认技能 **EB** 改为严格还原 V3.58（悬停 100 mm、表格顺序 + 可信优先），加技能编号（EB/DP/HD/QH）、发布包运行时只读、仿真注入 6 个后端已有环境变量、仿真窗口技能页、每次仿真任务打分 `Y = s/(t*z)*100`。
- **V3.61（已交付）**：第三个仿真场景 `stack_cups`（叠杯子，RoboDojo；两只 16 oz 红色 party 杯）；新模块 `NERO_V361_Cup_Stack.py` 离线 MuJoCo 松手试验 + URDF 严格 6-DoF 可达性试验，得出技能 **CS**（Cup Stack）9 个旋钮（第 ⑥ 组，`wired="study"`，**尚无运行代码读取**）；技能页每个区域可折叠 + 一键全部折叠/展开；技能名称后带 `[XX]` 缩写（控制台与仿真窗口一致）。**后端未改，机械臂在仿真里还不会自动叠杯子**（待 V3.62 加后端工具，需先和用户确认接入方式）。
- **V3.62（已交付）**：叠杯子 Agent 技能 `NERO_V362_Cup_Skills.py`（6 个工具的技能契约 + 复用 `NeroLLMAgent` 通信层的 Agent 循环 + 后端适配 `BackendIO` + 无界面 MuJoCo 替身 `TwinIO`）；后端只在 RUN TASK 的已有 A2 块里加了 `stack_cups` 分支；杯子原点改到杯口中心（CONNECT 抓持锁定点在重心上方）；第 ⑥ 组 11 个旋钮由仿真技能读取。**只在替身里验证过，完整仿真链路和真机都还没跑过**。
- **V3.63（已交付）**：修复 V3.62 实测放置时 J7 越限急停——联合规划整条关节链路（`FastArm` 解析雅可比，从真实关节出发，J7 优先 ≥ 15°）、执行前逐点检查轨迹（不合格改关节空间）、先转腕再平移、急停即停 + `_v337_clear_latch` 解锁重试一次；新增固定主相机 D435i 杯子感知 `NERO_V363_Cup_Perception.py`（相机进程 `NERO_V331` 在 stack_cups 场景改跑杯子感知）。替身带 URDF 运动学 + 固定相机渲染。**完整仿真链路待用户复测，真机未验证**。
- **V3.64（已交付）**：新仿真场景 `stack_cups_so101`（叠杯子 · SO-101 夹爪，7 mm 转接环，绕法向偏转 45°），是 `stack_cups` 的变体：`active_scene()` 仍为 `stack_cups`（后端 / 相机进程 / 感知 / Agent 字节级不变），夹爪由 `NERO_V343_Scenes.gripper_variant()` 决定；新模块 `NERO_V364_SO101_Gripper.py` + `assets/so101/`（官方网格 Apache-2.0）；杯子技能加夹爪档案 `GripperProfile`（原夹爪 = V3.63 行为）。替身试验见 `config/cup_stack_study_v364_so101.json`。**安装尺寸是估计值（座面 37.5 mm、环 7 mm、−45°），完整仿真链路待用户复测，真机未验证；实物需加指垫**。
- **V3.65（已交付）**：控制台新 TAB「NERO实机任务」（`NERO_V365_Real_Cup_Task.py`）：叠杯子 · SO-101 的仿真技能 / 参数 / 夹爪档案 / LLM 工具契约原样用于实机（`RealIO` 只换眼睛、手臂、手）；SO-101 动指舵机 STS3215 驱动 `NERO_V365_STS3215_Gripper.py`（180° = 夹紧，位置控制 + 扭矩上限夹紧，无力闭环）；`NERO_V313` 新增关节路径执行（MoveIt `/execute_trajectory`，路径围栏检查）。运行档位 ① 规划预览 ② 悬停预演 ③ 实机执行。**只在替身和模拟舵机上验证过，真机未验证**。
- **V3.66（已交付）**：实机任务的杯子识别改用 Camera调试页的 YOLOE 物体识别流水线（`NERO_V366_Real_Cup_Yolo.py`：同一检测器对象 / 模型 / 置信度 / 取帧 / `fuse_depth` / `locate_objects`），在 YOLOE 实例轮廓上做锥面杯轴 + **轮廓顶边量杯高**（D435i 杯口丢深度，V3.65 实测深度杯高 99–108 mm）+ 单杯 / 已叠好判断；YOLOE 不可用直接终止，不退回颜色法；主相机监控框画轮廓 + 标签。**只在替身（假 YOLOE 轮廓）上验证过，真机未验证**。
- **V3.67（已交付）**：实机抓杯「先对准 → 原地转腕（两指连线沿杯子半径）→ 竖直降到预抓取点 → 两段下探（杯口上方 10 mm 复查）」（共用技能里由夹爪档案 `align_rotate_m` / `descent_stop_above_rim_m` 打开，仿真档案为 0 / None 不变）；实机固定指离杯壁 8 mm；实测安装参数 + 碰点标定指尖修正（`real_mount`）；倾斜可信度 + 夹爪遮挡（`occluded`）+ 失败快照；规划余量 4–8° 写明可执行；STS3215 空夹标定（`touch_deg`）。**只在替身上验证过，真机未验证**。
- **V3.68（已交付）**：实机抓杯“抓了再判断”（`lenient_pick`：下探后一律夹紧抬起，抬起后夹爪 + 相机两条一致才算成功；修复 V3.67 用陈旧倾斜值放弃夹取的 bug）；规划记住上次角度 + 全角度 30° / 10 s 预算；识别各步计时、自有 YOLOE 实例、只在失败时存快照；主相机画模型指尖 F/M 校偏转角；偏转角扫描；任务记录 `task/`（每个工具调用前后截图 + steps.jsonl + summary.txt）+ `[V3.68 STEP n]` 日志 + `[REAL TASK SUMMARY]`。**只在替身上验证过，真机未验证**。
- **V3.69（已交付）**：真机第一次完整套好（V3.68）后的加固：验证前让开视线（`clear_view_before_verify` + `RealIO.clear_view`）、以相机判断夹住（`camera_decides_grip`）、规划三候选取最优 + 统一 15 s 预算（`_plan_best_of`）、YOLOE 强制 GPU（`yolo_device`）、“在夹爪里”需在夹爪旁、运行速度档 1X–10X（`apply_speed`，关节 ≤ 72 °/s、杯边直线 ≤ 60 mm/s）、任务录像（`TaskVideo` → `task/…/task_video.mp4`）。**只在替身上验证过，真机未验证**。
- **V3.70（已交付）**：自动采集闪退修复（Camera 页「退出示教模式」的安全区切换 + 重绘放回 Tk 主线程，原来在后台线程里 → libX11 段错误）；控制台崩溃记录 `logs/crash.log`（faulthandler + 线程 / Tk 回调异常）；实机页日志镜像进 `logs/console.log`；实机参数「示教负载」（`--payload`，SDK 有 `set_payload` 等接口才发送，NERO 驱动 d90e234 没有 → WARN）。**真机未验证；示教松手回位是否由负载引起待用户确认**。
- **V3.71（已交付）**：`RealIO` 运动真到位才结束（`wait_still` + 到位超时 `ARM_NOT_AT_GOAL`）、按实测速度降速（`joint_cap_deg_s`，硬件 `speed_percent` 限速）、失败收尾（`recover_after_failure`：停稳 + 竖直升到 0.22 m）、围栏绕行（先升高）；共用技能（实机档案打开、仿真档案关闭）：识别漏检重试、规划余量 ≥ 8° 即停、放置前测夹着的杯子并修正偏移（`NERO_V366.held_geometry`）、叠好判断按目标位置单簇 + 比单杯高 4–40 mm（`stack_by_cluster`，可填实测套叠增量）；监视器实时同步任务识别（`on_perception`）；`start_nero.sh` 清 `CUDA_VISIBLE_DEVICES`。**只在替身上验证过，真机未验证**。
- **V3.72（已交付）**：第 5 个仿真场景 `sort_items`（香蕉 / 苹果 / 鼠标 / 粗马克笔 → 左红框 / 右蓝框，SO-101 夹爪，场景本身带 `"gripper": "so101"`）；新模块 `NERO_V372_Sort_Skills.py`（`SortSkills` 继承 `CupSkills` 的运动 / 安全代码：observe_scene / pick_object / place_in_box / check_bins / return_to_safe_pose，LLM 按提示词从标签判断类别）；物体原点 = 顶面中心 + 隐形配重（后端 CONNECT 锁定点是球铰）；后端只在 RUN TASK A2 钩子段里加 sort_items 分支；只支持仿真真值位姿。**只在替身里验证过，完整仿真链路待用户复测**。
- **V3.73（已交付）**：sort_items 首跑失败修复——后端 `object_pose(name)` 查 `sim_obj_{name}`，所以 `SortBackendIO` 必须传去掉 `obj_` 的名字（`tf_name`）；测试替身 `tests/v372_backend_twin.SortCore` 严格按后端规则查 TF；分类得分按 Agent 处理过的物体。**只在替身里验证过**。
- **V3.74（已交付）**：实机杯子识别“几何只测量、不下结论”（`NERO_V366` 默认不按杯高分类，`classify_by_height=True` 才用 V3.66 分档）+ LLM 看图复核（新模块 `NERO_V374_Vision_Review.py`：带编号轮廓的主相机画面 → 所选模型，回答单杯 / 已叠好 / 不是杯子 / 在夹爪里；只在关键识别时刻；按布局缓存）+ “在夹爪里”需夹爪下到杯子高度 + 桌面偏差提示。**用户原则：不要加画蛇添足的判断准则；几何只量位置和尺寸，“是什么”交给看图的 LLM**。**看图复核未连真实 API；真机未验证**。
- **V3.75（已交付）**：sort_items 框缩小外移（内 23 × 17 cm、壁高 5 cm、中心 y = ±0.28，离物体 ≥ 7 cm）+ 放入前按物体占地检查放不放得下（`sort_item_footprint`）；新模块 `NERO_V375_Collision_Check.py`：候选抓放姿态先做 MoveIt `/check_state_validity`（全部碰撞对）+ SO-101 粗略盒子检查，MoveIt 拒绝时日志 `[V3.75 MOVEIT]` 写碰撞对；**MoveIt 机器人模型仍是原装 AgileX 夹爪**（`effector_type:=agx_gripper`），只有它的夹爪连杆碰、SO-101 与 link0–7 都不碰时，`SortBackendIO` 执行技能自己逐点检查过的直线路径（`NERO_V375_SO101_OWN_PATH=0` 关闭）；替身加同一套碰撞检查，`tests/v375_moveit_twin.py` 模拟 MoveIt。**SO-101 新转接环：偏转默认 0°（原 −45°）**，实机旧设置（无 `mount_rev`）打开时迁移一次（偏转 +45°、指尖修正清零）。**只在替身和 MoveIt 模拟里验证过；仿真链路待复测，真机未验证**。
- **V3.76（已交付）**：V3.75 仿真复测后的修复：夹着物体搬运时 MuJoCo 发散（关节 5 / 7 RUNAWAY）→ 分类场景 SO-101 指垫改软（`soft_pads`，solref 0.006；叠杯子不变）+ 持物运动慢 3 倍（`HELD_SLOW`）；后端冻结的 `V2-44 R2 BASKET RIM` 会把旧鸡蛋篮附近的薄盒子削到 14 mm → 框壁在 MuJoCo 里拆成半长 < 45 mm 的短段（`sort_bin_geoms`，MoveIt 仍用整墙，`legacy_rim_hits` 复刻规则供测试）；安全停后按夹爪 + 物体实际位置对齐（`_v376_sync_held`，`ITEM_DROPPED_IN_BOX` / `ITEM_DROPPED`），打分按最后位置；开始前低于 0.30 m 先竖直升高再回家（`v376_clear_start`）；**SO-101 偏转默认 −90°**（用户：上下开合、动指在上；“俯视顺时针 90°”与“动指在上”换算后相反，渲染 A/B 对比图后用户选 A = −90°；+90° 时 J7 ±90° 下叠杯子不可达），实机设置 `mount_rev` V3.76 迁移；`start_nero_sim.sh` 限时停止（`v376_stop` INT→TERM→KILL）+ 控制台倒计时。**发散只做了缓解、未拿到 diag CSV 确认；只在替身和 MoveIt 模拟里验证过，真机未验证**。
- **V3.77（已交付）**：新模块 `NERO_V377_Stale_Procs.py`（`classify` / `stale_procs` / `stop_procs`，控制台「启动仿真」前弹窗列出并关闭残余进程，`start_nero_sim.sh` V3.07 检查也改用它）；sort_items：`SortBackendIO.move_line` 自己的直线路径手腕边走边转（`_turning_segment`，与规划 / MoveIt 一致），MoveIt 因关节路径检查拒绝时也可用（MoveIt 确认机械臂连杆不碰）；夹物动作每个关节余量 ≥ `HELD_MARGIN_DEG` = 10°（`margin_rejects`）；安全停结果带 `trip_reason` / `closest_joint_limit`（`BackendIO.v377_trip_info` 读 `NERO_DIAG_DIR/diag_trip_*`，叠杯子同样）。**只在替身和 MoveIt 模拟里验证过；仿真链路待复测，真机未验证**。
- **V3.78（已交付）**：V4 准备工作第四组“感知精度”第 0–1 步（用户只有 D435i）：独立工具 `NERO_V378_D435i_Bench.py`（`profile` 锁定曝光 / 增益 / 白平衡 / 60 Hz → `NERO_CALIB_DIR/d435i_profile.json`；`noise` 5 种深度处理对比；`accuracy` ChArUco 板为坐标系 + Camera 页同一套 YOLOE + `YP.locate` 测鸡蛋中心误差；`capture` 训练图 → `~/Downloads/objectmodels/datasets/`）。Camera 页 / 后端未改（下一版让 Camera 页启动时套用这份参数）。**没有在真实 D435i 上运行过**。
- **V3.79（已交付）**：Camera 页「相机信息」→「保存相机信息」右边「相机高级测试」窗口（`NERO_V379_Camera_Lab.py`，用 Camera 页的相机线程 `run_on_device` / `collect_depth`）：① 相机参数（保存并锁定 → Camera 页每次连接自动套用，深度预设仍按 Camera 页）② 深度噪声 ③ 定位精度 ④ 采集 + YOLOE 自动标注 + 框级复核（`BoxEditor`，改框后 `recompute_outline`：YOLOE 轮廓裁到框内 → GrabCut → 框）；数据集 `<root>/classes.json` + `<类别>/images|labels/{train,val}` + `meta/` + `depth/`，每 5 张 1 张 val。V325 哈希因此更新（tests 61–79）。**没有在真实 D435i / YOLOE / Tk 上运行过**。下一版 V3.80：训练 / YOLOE 精调 + 量化评估（固定验证集 mAP、召回、3D 误差对比）+ 生成可在 Camera 页选用的模型。
- **V3.80（已交付）**：相机高级测试 4 项改进（仍在 `NERO_V379_Camera_Lab.py`，标记 `V3.80 CAMERA LAB UPGRADE`）：① `autotune`（工作区 ROI、曝光 = 闪烁周期整数倍、亮部 p99.5 < 245、最低增益到平均 110–150；反差大时以白色物体不过曝为先）+ 命名参数组 `cameracalib/d435i_profiles/` + 连接时自动套用 + `lighting_check`（亮度变化 > 20 % 提示重调；V325 只多传 `src=self`）② 表格用 `CAM.treeview_style` + `noise_verdict`（D435 理论 z²·0.08/(f·B)、平整度、最佳处理与代价、换算到 750 mm）③ 任意物体（`locate_fn`：一般物体 `locate_objects` + 高度误差 / 蛋形 `locate`）④ 类别清单（无默认类别、勾选的全部标注、当前类别 = 子文件夹、每类建议 150 张）。**没有在真实硬件上运行过**。计划见 8.5 / 8.6。
- **V3.81（已交付）**：相机参数文件另存 / 打开（`save_profile_file` / `load_profile_file`）；新模块 `NERO_V381_Yolo_Train.py` = 相机高级测试「⑤ 训练 / 评估」：`scan` / `class_table` / `plan_dataset`（默认只用已复核）/ `export_dataset`（`<root>/_runs/<run>/dataset`，硬链接 + 重映射标签 + data.yaml + manifest）→ 子进程 `job`（`YOLOEPESegTrainer`；快速 = `linear_probe_freeze` 只训 *cv3 分支最后一层，完整 = 不冻结；`progress.jsonl` 事件、`STOP` 文件停止、90 s 后强杀）→ `evaluate` / `compare`（自写匹配：同一份 val，原模型 set_classes(类别名) vs 新模型自带类别，识别率 / 精确率 / 误检每张 / AP50 / AP50-95 / 轮廓 IoU / ms，结论 better / same / worse + 验证集太小）→ `save_model` 到 `finetuned/<名>/`（模型 + `model_card.json`，**V3.82 才接入 Camera 页 / 实机**；精调模型绝不能再 `set_classes`）。V325 / 后端未改。测试用 `tests/v381_fake_yolo.py`（真实子进程 + 假 YOLOE）。**没有在真实 ultralytics / GPU 上训练过**。
- **V3.85（已交付）**：主相机手眼标定加固（`detect_aruco` 红外深度平整簇采纳门限放宽至 25%，抗长曝光白纸晕光；`calibrate()` 增加运动学探针步纯平移尺度反算 + 1D 最优尺度网格搜索自校准，消除 PnP 边长/光照晕光带来的尺度放大，自动恢复 1:1 公制尺度，留一残差恢复至合格线内；标定结果记录 `scale_factor`）。**没有在真实硬件上运行过**。
- **V3.84（已交付）**：主相机多帧目标跟踪与抓前重检（新模块 `NERO_V384_Object_Tracker.py`：固定 ID 关联、EMA 平滑、>15 mm 位移即时重置、抓前重检 ≤3 mm 通过 / 3–15 mm 微调 / >15 mm 中止）；`egg carton` + `egg` 精调模型联合定位 6 孔坐标（`slot_1..slot_6`）与空满状态（`BUILTIN_SYNONYMS` 含 `egg holder, egg tray, carton, tray`）；桌面平面基准锁定（`table_plane_lock.json`）与 >3 mm 漂移监控；视觉定位页新增「▶ 10帧对比评测」按钮量化输出单帧 vs 多帧滤波跟踪的抖动下降率、ID 跳变与 6 孔一致率。**没有在真实硬件上运行过**。
- **V3.83（已交付）**：GPU 自修复（`start_nero.sh` 启动前子进程预检，休眠唤醒后 `torch.cuda` 故障自动重载 `nvidia_uvm`，控制台「修复 GPU」按钮，训练子进程预检与报警）；组合模型（`v_ymodel` 用 `;` 分隔多模型；修复 `finetune_covers` 宽松匹配 bug，严格小写/单复数一致；`Camera调试` 页视觉定位模型单选下拉框改多选 Menubutton，原模型锁定选中，精调模型默认全选；`YoloEDetector.detect` 自动按物体分配提示词，精调模型固定类别不调用 `set_classes`，原模型文字提示补齐未覆盖物体，IoU ≥ 0.5 框去重，标记模型来源；⑤ 第 5 步「应用到 Camera调试 页」改为追加精调模型，「恢复原模型」清空精调模型；④ 采集标注增加“组合模型”选项）；深度滤波接入（Camera调试页可选“滤波 + 5 帧中值 / 10 帧中值”）与蛋表面到几何中心换算。**没有在真实硬件上运行过**。
- **V3.82（已交付）**：精调模型接入（**只改 V347**：`list_models` 列出 `finetuned/<名>/<文件>`、`finetune_card` / `finetune_covers` / `finetune_base_path`、`is_pf_model` 对精调模型为真 → 固定类别不 `set_classes`、`YoloEDetector._route_untrained` 把精调模型不认识的物体交给原模型并去重叠框；V325 / V366 字节未变）+ ⑤ 第 5 步「应用到 Camera 页 / 恢复原模型」；标注辅助（`DEFAULT_SYNONYMS` / 每类提示词 `class_prompts`、`AUTO_CONF` 0.10 找框 + `ACCEPT_CONF` 0.30 确认，低分 = `candidate`（不进标签 / 训练 / 计数，双击 / 回车确认，复核时丢弃）、来源：文字 / 视觉样例 `vp_ref` / 精调模型、`merge_relabel` 重新标注、实验窗口自己的检测器 `_ldet`）；曝光（`roi_stats` 镜面亮斑 `spots` / `p995_ex` / `sat_large_pct`、`mean_lo` 可调、`autotune` 伽马、B378 `apply_profile` 支持 `gamma`）；训练 1280（批 4 / 2），评估按 `YP.IMGSZ`。**没有在真实硬件上运行过**。
- **用户的 V4 准备清单（22 项，8 组；SmolVLA 接入前的准备）**：当前先做第四组（感知精度：YOLO 微调、D435i 深度处理与蛋中心换算、两路相机分工与跟踪）；顺序：第 0 步（照明 / 相机参数 / 精度测试）→ 深度 → 两路相机 → YOLO 微调（必须在相机参数固定后采图）。仿真 sort_items 不再细调抓取规则（V3.77 后结论：主要卡在仿真里 SO-101 的物理稳定性与 MoveIt 夹爪模型）。
- **已和用户确认的后续路线（V3.79 起，原 V3.78 → V3.80 顺延，可能被 V4 准备清单替代）**：仿真页用“任务登记表”按场景固定提示词 / 快捷按钮 / 工具调试栏（A2 钩子事后调整，不动后端冻结部分）；**实机页做成真正通用**：通用识别（任务提取物体名 → YOLOE + 深度 → 物体 / 容器 / 颜色 / 尺寸 / 关系；**用户要求开启“把相机画面发给支持看图的 LLM”**辅助识别）+ 通用动作库（observe / pick / place_into / place_on / place_at / verify，仿真与实机共用，仿真 sort_items 改用它）+ 专用技能库（叠杯子等带技能卡，由 LLM 选择）+ 档位 ①②③ 在运动层统一实现（② 统一截断到悬停高度）。V3.78：通用识别 + 实机新外壳（只开 ①）+ 仿真登记表；V3.79：通用动作库 + 实机 ②；V3.80：实机 ③ + 检查 + LLM 选专用技能。叠杯子必须保持不退化（V3.65–V3.71 测试原样通过）。
- **接下来的核心路线（V3.81 起，再顺延；先请用户在实机上按 ①→②→③ 复测叠杯子）**：遵循**“一套大脑、两套身体（仿真与实机除眼睛和手脚外 100% 通用）”**原则，将算法拆分为**“铁打的底层骨架 + 可拨动的控制旋钮（Control Knobs / Strategy Profile）”**，通过**带噪仿真自动寻优试验（DOE / 贝叶斯优化）**自动搜索各场景鲁棒最优参数包，并按 `纯视觉干跑 → 5 cm 悬停预演 → 实机柔顺抓放` 稳步完成真机迁移。

- **V4.0–V4.3**：双路径平台（Agent + 传统技能 / LeRobot + SmolVLA），控制台新页「VLA / LeRobot (V4)」；V3 的 Agent + 传统技能已能实机叠杯子、夹鸡蛋。
- **V4.4（2026-11-08）**：用户本机磁盘不够，**LeRobot + SmolVLA 的一切（venv、uv 的 Python 3.12、HF 缓存、smolvla_base、示教数据、导出的 LeRobotDataset、lerobot-train 训练输出、日志）都放在卷标 `EMBODIED_AI` 的移动硬盘 `<盘>/nero_lerobot/`**（`NERO_V44_Storage.py`）。控制台进程（/usr/bin/python3 3.10）**不导入** lerobot / 硬盘上的 torch；SmolVLA 推理在 `NERO_V44_SmolVLA_Worker.py` 子进程（硬盘 venv 的 Python，`-I`，去掉 ROS PYTHONPATH）。硬盘不在时大文件操作一律拒绝，不许写到本机磁盘。LeRobot ≥ 0.5 需要 Python ≥ 3.12（默认用 uv 把 3.12 装到硬盘；失败退回系统 3.10 → LeRobot 0.4.4）。

- **V4.5（2026-11-08）**：实机鸡蛋任务（`RealEggSkills`）修正：夹蛋时测量蛋心在夹爪系的位置并在放蛋时复用（旧常数 `EGG_HELD_CENTER_Y_M=8.5 mm` 是错的，实际 ~20 mm），鸡蛋**下端**对准槽心，先试 ±90° 立蛋，转腕前腕部相机俯视复测槽位；蛋托槽位由深度凹坑拟合（`refine_slot_grid`，可选 `config/egg_carton.json` 实测槽距）。**安全区 z 检查的工具点是 SO-101 最低指尖**（不是胶垫中心）：规划 ≥ 桌面 + 5 mm，执行 ≥ 3 mm；桌面取相机 / 安全区较高者，差 > 15 mm 拒绝。V4.4 闪退加固：SmolVLA 加载不在 Tk 主线程，`os.environ` 只在主线程写。

- **V4.5.1（2026-11-08）**：用户的 V4.5 控制台启动即段错误（Tk 内部，VLA 页第 0 区创建控件时）。VLA 页改为第一次打开才创建，`./start_nero.sh --no-vla` 可完全跳过；该页去掉全部 emoji（Ubuntu 22.04 libXft 2.3.4 + 彩色 emoji 是最可疑原因，未证实）。**GUI 文字不要再用 emoji。**

- **V4.6（2026-11-08）**：SmolVLA 页——未用 NERO 数据微调的模型（smolvla_base 6 维 / 内置替身）**禁止在实机执行**；没有视觉 / 目标检查时结果是 `unverified`，不许报成功；所有按钮要有状态灯和开始 / 结束日志；急停按钮必须始终可见。鸡蛋任务——两个相机对鸡蛋位置不一致时复看确认，否则停止；夹偏（指尖距离 < 蛋宽 − 8 mm）重夹一次；下插严格到位（2 mm / 1° / 0.5°）否则不松开；槽位在夹蛋前用空爪腕部相机测（圆形暗坑 + 相邻槽检查）。用户 V4.5.1 那次的数据：腕部比主相机偏 18 mm，主相机手眼标定需要在工作区重新做。

- **V4.6.1（2026-11-08）**：V4.6 鸡蛋三道保险全部依据了未验证的假设，导致三次失败（夹住又松开）。规则：**夹偏按实测基准 `held_gap_mm`=26.5 mm（SO-101 正常夹住鸡蛋的读数，不是蛋宽）判断，默认只记录**；腕部相机对鸡蛋的读数只记录（白蛋放在白色标定纸上无法识别）；夹蛋前看槽默认关闭；腕部识别排除夹爪区域和过曝画面。页面名：「Agentic实机任务」（原 NERO实机任务）、「Smol VLA实机任务」（原 VLA / LeRobot (V4)），后者复用前者的相机监控绘图（`CameraMonitorMirror`）。**新增版本号时必须同步扩展测试里的版本白名单**（V4.4–V4.6 漏了，很多 V3 测试一开头就停了）。**不要从单个读数推断硬件状态；新判断规则要先用实测数据验证。**

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
5. 重新生成校验和（V3.60 起扫描 3 层，包含 `config/strategies/*.json`；V3.68 起排除 `task/`）：`sha256sum $(find . -maxdepth 3 -type f ! -name "SHA256SUMS.txt" ! -path "./.git/*" ! -path "*/__pycache__/*" ! -path "./logs/*" ! -path "./task/*" | sed 's|^\./||' | sort) > SHA256SUMS.txt`。
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
6. **发布包运行时只读（V3.60）**：程序运行时绝不写发布文件夹里的任何文件（`config/strategies/` 只读）；例外只有运行时输出目录 `logs/` 和（V3.68 起，用户要求放在项目文件夹内的）`task/`，两者都在 `.gitignore` 里、不进校验和，测试用 `NERO_V368_TASK_DIR` 指到临时目录；当前技能写 `logs/v360_active_strategy.json`，用户技能写 `~/Downloads/cameracalib/strategies/`（`NERO_V360_USER_STRATEGY_DIR`）。`test_v3_60` F6 / `test_v3_60_gui` H6 会校验。
7. **默认技能 EB = 原逻辑（V3.60）**：`BASELINE_KNOBS` 代入后必须与后端 / V3.58 原代码零偏差（`test_v3_60` F2 从后端源码 AST 抽取原函数比对）；改默认值等于改行为，要先和用户确认。
8. **叠杯子（V3.62）/ 分类（V3.72）**：后端对叠杯子 / sort_items 只允许在 RUN TASK 的已有 A2 钩子段里分别调用 `NERO_V362_Cup_Skills.gui_run_task()` / `NERO_V372_Sort_Skills.gui_run_task()`（`test_v3_62` P1：去掉这段后与 V3.60 后端字节级一致；sort_items 分支必须留在这段内）；sort_items 物体原点 = 顶面中心（`SC.sort_item_centre` 换算中心）；杯子技能逻辑只写在 `NERO_V362_Cup_Skills.py`，`BackendIO` 与 `TwinIO` 共用同一套 `CupSkills`。场景杯子的原点是杯口中心（`SC.CUP_ORIGIN="rim"`），判断套叠用 `cup_task_state(..., origin="rim")`。
9. **叠杯子运动安全（V3.63）**：持杯运动只朝规划时检查过余量的关节解执行；任何直线插补轨迹先经 `check_joint_path`（≥ 4°、≤ 5°/点）才执行；技能遇急停必须停下并经后端 `_v337_clear_latch` 解锁，不能在锁住时继续发运动。固定相机杯子感知只在 `NERO_V331_Fixed_Camera.py` 的 stack_cups 分支里，鸡蛋场景路径不变（`test_v3_31`）。
10. **夹爪变体（V3.64 / V3.67–V3.69）**：`GripperProfile` 的实机选项（V3.67 `align_rotate_m`、`descent_stop_above_rim_m`、`wide_search_below_deg`；V3.68 `lenient_pick`、`plan_time_budget_s`、`remember_angles`；V3.69 `clear_view_before_verify`、`camera_decides_grip`；V3.71 `recheck_held_before_place`、`stack_by_cluster`、`perception_retries`、`recover_after_motion_failure`、`nested_dz_real_mm`）在 `OFFICIAL_PROFILE` / `so101_profile()` 中必须保持关闭（`test_v3_67` Y1）；新夹爪只能作为场景变体（`SCENES[...]['base']` + `'gripper'`）接入，`active_scene()` 返回基础场景；变体夹爪的 MJCF 必须保留后端用到的名字（`gripper_joint1` 开口 = 2000·q1、`gripper_joint2` = −q1、`gripper_link1/2`、`finger1/2_rubber_pad`、`gripper`/`gripper_force` 执行器、`gripper_coupling`）。原夹爪档案 `OFFICIAL_PROFILE` 的参数不能改（等于 V3.63 行为）。V3.76 起 SO-101 安装偏转默认 −90°（用户选定：工具水平、J7 = 0 时动指在上；V3.75 为 0°，V3.64–V3.74 为 −45°）。分类场景的框壁在 MuJoCo 里必须保持短段（半长 < 45 mm），否则后端冻结的 `V2-44 R2 BASKET RIM` 会把它削到 14 mm（`test_v3_76` I2）。
11. **实机叠杯子（V3.65 / V3.66）**：实机杯子识别只能走 Camera调试页的 YOLOE 流水线（`NERO_V366`，复用 `det_yolo` / `_grab_frames` / `YP.fuse_depth` / `YP.locate_objects`），不另写识别方法、不退回颜色法；实机只能复用仿真的 `CupSkills` / 旋钮 / 夹爪档案（不在实机模块里另写一套技能逻辑）；实机运动只走 `RealPipeline.move_joint_path`（或 `move_and_settle`），必须经过关节限位 + 目标围栏 + 路径围栏检查；运行档位 ② ③ 必须先弹窗确认；`NERO_V365` 运行时只写 `NERO_CALIB_DIR/real_cup_task.json`。

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
| `NERO_V347_Yolo_Perception.py` | **实机主相机感知模块**（V3.82：精调模型 = 固定类别 + 不认识的物体交给原模型）：YOLOE-26 视觉/文字提示 + 无提示 PF 多类别检测、ROI 快速局部桌面拟合、`SilhouetteFit`、3D 椭球轮廓定位 |
| `NERO_V349_Real_Wrist_Camera.py` | **实机腕部相机模块（V3.49–V3.58）**：UVC 相机枚举与曝光/白平衡/对焦锁定、ChArUco/棋盘格检测与运动学锚定内参标定（`calibrate_intrinsics`）、24 角点 ChArUco 6-DoF 检测（`detect_charuco_pose`）、自适应四象限大倾角内参采样规划（`plan_intr_auto_targets`）、带杆臂约束与 2D 尺度自标定的 Eye-in-Hand 求解器（`calibrate_eye_in_hand`）、粗到精闭环（`wrist_refine_target`）、零先验蛋托识别（`locate_egg_holder`） |
| `NERO_V359_Strategy.py` | **技能参数中心（V3.59–V3.60）**：`KNOB_SPECS` 45 项（每项 `wired` = `sim_env` 6 / `real` 8 / `none` 31）、`StrategyProfile`（skill_id + `short_name` + 两字母 `code` + `skill_card` + `knobs`）、内置技能 EB/DP/HD/QH、与后端等价的公式（`score_grasp_candidate`、`compute_gripper_prior`、手腕评分、`order_empty_slots`、`rank_detected_targets`）、`strategy_sim_env`（只注入与基准不同的 6 个环境变量）、多点悬停预演规划 / 执行、`SkillStrategyTabUI`（控制台 real 模式 / 仿真窗口 sim 模式）、`add_sim_strategy_tab` |
| `NERO_V362_Cup_Skills.py` | **叠杯子 Agent 技能（V3.62）**：`CupSkills`（observe_scene / plan_cup_stack / pick_cup / place_cup_into / verify_stack / return_to_safe_pose）、`plan_cup_stack`（捏杯沿方位 + 同一翻转 + 严格 IK，余量 < 4° 拒绝）、`pinch_frame/pinch_tcp`（沿杯壁倾斜 8.2°、夹紧中心 TCP y +2 mm）、`BackendIO`（后端 skill_core + 共享变量）、`TwinIO`（官方指垫 + 后端夹爪逻辑的 MuJoCo 替身，不含机械臂）、`run_cup_task`（Agent 循环 + `[V3.62 SKILL SCORE]`）、`gui_run_task`（后端 A2 块入口）、`twin_study`（`--study`） |
| `NERO_V363_Cup_Perception.py` | **固定主相机杯子感知（V3.63）**：`perceive_cups`（红色色调分割 → 深度反投影 → 机械臂自滤 → xy 聚类 → 锥面拟合杯轴 / 高度 / 倾斜，单杯 / 已套叠 / 未知）；由 `NERO_V331_Fixed_Camera.py` 在 stack_cups 场景调用 |
| `NERO_V364_SO101_Gripper.py` | **SO-101 夹爪（V3.64）**：安装（`MOUNT`，环境变量可改）、动指开口映射（`jaw_angle_deg` / `opening_mm_for_tip_gap`）、`T_tcp_grasp`（官方 TCP → SO-101 抓取系）、`gripper_parts` / `insert_into_model_xml`（场景与替身共用同一份 MJCF）；网格在 `assets/so101/` |
| `NERO_V365_Real_Cup_Task.py` | **NERO实机任务（V3.65）**：`RealIO`（CupSkills 的实机 IO：FastArm TCP、连续 IK 直线段 / 关节线性插值 → 关节路径执行、D435i 杯子感知（桌面 z 平移 + 围栏工作区）、STS3215 夹爪换算）、`run_plan_preview` / `run_hover` / `run_execute`、`PipelineArm`、`camera_from_tab`、`RealCupTaskTabUI` |
| `NERO_V366_Real_Cup_Yolo.py` | **实机杯子识别（V3.66）**：`yolo_cfg_from_tab`（Camera 页 YOLOE 设置，文字提示 / -pf 选择同 `_resolve_object_query`）、`perceive_cups_yolo`（detect → fuse_depth → locate_objects → 每个实例轮廓：锥面杯轴、轮廓顶边杯高 `silhouette_height`、`classify` 单杯 / 已叠好、lifted / outside / edge、重复框合并）、`summary` |
| `NERO_V372_Sort_Skills.py` | **分类场景 Agent 技能（V3.72）**：`SortSkills`（继承 `CupSkills` 的运动 / 安全 / 夹爪档案）、`SortTwinIO` / `SortBackendIO`、`SORT_TOOLS` / `SORT_SYSTEM_INSTRUCTIONS`、`run_sort_task`（Agent 循环 + `[V3.72 SKILL SCORE]` + `[V3.72 SORT RESULT]`）、`gui_run_task`（后端 A2 钩子入口）、`twin_episode`（`--twin`） |
| `NERO_V381_Yolo_Train.py` | **YOLOE 精调与评估（V3.81 / V3.82）**：V3.82 加 `TrainPage` 第 5 步（应用 / 恢复模型）、1280 训练；`scan` / `class_table` / `plan_dataset` / `export_dataset`、`linear_probe_freeze`、`evaluate` / `compare`、`run_job`（子进程：训练 → 原 / 新模型评估 → `eval.json`）、`TrainJob`（启动 / 进度 / 停止 / `pid_alive`）、`find_unfinished`、`save_model` / `list_saved` / `discard_run`、`TrainPage`（相机高级测试 ⑤） |
| `NERO_V379_Camera_Lab.py` | **相机高级测试（V3.79 / V3.80）**：V3.80 加 `autotune` / `roi_stats` / `judge_brightness` / `save_named` / `activate_named` / `lighting_check` / `noise_verdict` / `locate_fn` / `accuracy_verdict` / `Dataset.class_counts`；`apply_saved_profile`、`Dataset`（类别子文件夹 + YOLO 分割标签 + meta）、`dets_to_anns`、`BoxEditor`、`recompute_outline` / `grabcut_poly` / `clip_poly`、`measure_noise`、`accuracy_trials`、`CameraLab` 窗口（4 页）、`open_lab` |
| `NERO_V378_D435i_Bench.py` | **D435i 单机工具（V3.78）**：`load/save/apply_profile`（`NERO_CALIB_DIR/d435i_profile.json`）、`FilterChain`（空间 / 时间 / 补洞）、`median_stack`、`noise_report`、`board_pose` / `inner_corners` / `T_bench_cam_from_pose`（ChArUco 板 = 坐标系，z 朝上）、`accuracy_row` / `summarize` / `write_results`、交互模式 `profile` / `noise` / `accuracy` / `capture` |
| `NERO_V377_Stale_Procs.py` | **残余进程检查（V3.77）**：`classify`（按 argv 认 ROS / MoveIt / RViz / 实机驱动 / 任何文件夹的 `NERO_*.py`，不认控制台和编辑器）、`stale_procs`、`stop_procs`（INT→TERM→KILL，进程组）、`--list` / `--stop`（启动脚本用） |
| `NERO_V375_Collision_Check.py` | **碰撞预检（V3.75）**：`world_boxes`（桌面 + 分类框）、`so101_parts`（SO-101 + link5–7 粗略盒子，按候选关节 FK）、`obb_overlap`、`moveit_contacts`（`/check_state_validity` 全部碰撞对）、`classify_pairs`（blocking / 只有 MoveIt 夹爪模型）、`assess`、`official_gripper_parts`（原装夹爪估计外形，只给测试模拟 MoveIt 用）、`rect_corners` / `rect_dist`（占地矩形） |
| `NERO_V374_Vision_Review.py` | **LLM 看图复核（V3.74）**：`annotate`（轮廓 + #编号 JPEG）、`post_vision`（OpenAI Responses input_image / chat image_url）、`VisionReviewer`（关键时刻、布局缓存、失败只 WARN 一次）、`apply_review`（按回答设类别、列出与几何的分歧） |
| `NERO_V368_Real_Task_Record.py` | **实机任务记录（V3.68）**：`TaskRecorder`（每个工具调用前后截图、steps.jsonl、summary.txt，目录 `task/` 或 `NERO_V368_TASK_DIR`）、`step_line` / `task_summary`（日志）、`fingertip_points`（模型指尖叠加）、`roll_sweep`（偏转角扫描） |
| `NERO_V365_STS3215_Gripper.py` | **SO-101 实机夹爪（V3.65）**：Feetech STS 协议（取自用户工具）、`SO101RealGripper`（open_deg / open_gap / clamp / status）、角度↔指尖缝↔仿真虚拟开口换算、夹紧扭矩换算、`SimServo` 模拟舵机 |
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

## 7. 版本简史（V3.43 – V3.82）

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
| **V3.62** | 叠杯子 Agent 技能（填 API key + 输入任务即可在仿真里叠杯子）；后端 RUN TASK A2 块加 `stack_cups` 分支；杯子原点改杯口中心 + 卷边硬接触 + 与质量无关的杯子接触；第 ⑥ 组 11 旋钮 `[仿真技能✓]`；修正 V3.61 抓放翻转不一致；替身试验 16/16 |
| **V3.63** | 修复叠杯子放置 J7 越限急停（联合规划 + 逐点轨迹检查 + 先转腕再平移 + 急停即停解锁重试 + 不做小修正）；固定主相机 D435i 杯子感知（相机进程分支）；替身带 URDF 运动学与相机渲染；试验 20/20（相机 4/4） |
| **V3.64** | 新场景 `stack_cups_so101`（SO-101 夹爪 + 7 mm 转接环 + 偏转 45°，后端不变）；夹爪档案（抓取坐标系、固定指在外、沿杯壁下探、抬起 / 搬运后摆正等待、低位松手 + 松手后外移）；通用：全角度回退、搬运优先规划关节解、关节停稳等待；SO-101 替身试验 |
| **V3.85** | 主相机手眼标定加固（红外深度平整簇优先采纳 + PnP 运动学探针与 1D 最优尺度自校准，抗长曝光白纸晕光尺度偏差） |
| **V3.84** | 主相机多帧跟踪（固定 ID + EMA + 位移重置 + 抓前重检）+ `egg carton` 精调模型联合算 6 孔坐标与空满状态 + 桌面平面锁定与漂移监控 + 10 帧 A/B 对比评测 |
| **V3.83** | GPU 自修复（start_nero 预检 + nvidia_uvm 免密/交互重载 + 控制台 pkexec 修复按钮）；组合模型（严格单复数匹配、Menubutton 多选组合、按物体分配提示词、IoU 0.5 去重叠框、标注用组合模型）；深度中值滤波与蛋表面到几何中心换算 |
| **V3.82** | 精调模型应用到 Camera 页 / 实机（V347 固定类别 + 不认识的物体交给原模型）；标注辅助（同义提示词、低分候选一键确认、视觉样例、精调模型重新标注）；镜面亮斑 / 亮度下限 / 伽马；训练 1280 |
| **V3.81** | 相机参数文件另存 / 打开；「⑤ 训练 / 评估」：数据检查 → 一键 YOLOE 精调（快速 / 完整）→ 原 vs 新模型同一验证集评估与结论 → 保存模型 |
| **V3.80** | 相机高级测试：一键调参 + 命名参数组 + 连接后光照检查；噪声结论；任意物体定位精度；类别清单采集 |
| **V3.79** | Camera 页「相机高级测试」窗口：相机参数锁定并自动套用、深度噪声、定位精度、任意物体采集 + 自动标注 + 框级复核 |
| **V3.78** | D435i 单机工具：固定相机参数、深度滤波对比、鸡蛋定位精度（ChArUco 板坐标系）、训练图采集 |
| **V3.77** | 启动仿真前列出并关闭残余进程（控制台 + 启动脚本共用 `NERO_V377_Stale_Procs.py`）；分类：自己的直线路径与规划一致（手腕边走边转）、夹物关节余量 ≥ 10°、安全停原因写进结果 |
| **V3.76** | 搬运防发散（软指垫 + 持物慢 3 倍）+ 框壁短段避开后端削壁 + 安全停后状态对齐 / 按最后位置打分 + 开始前退开；SO-101 偏转 −90°（动指在上）；停止仿真限时 + 倒计时 |
| **V3.75** | sort_items 框缩小外移 + 放入前占地检查；MoveIt 碰撞预检 + 碰撞对日志 + 只有 MoveIt 原装夹爪模型碰撞时走自己检查过的直线路径；替身碰撞检查；SO-101 新转接环偏转 0°（实机旧设置迁移） |
| **V3.74** | 实机杯子识别：几何只测量 + LLM 看图复核（`NERO_V374_Vision_Review.py`）+ 夹爪须下到杯子高度才算夹起 + 桌面偏差提示 |
| **V3.73** | sort_items TF 帧名修复（`sim_obj_banana`）+ 测试替身严格按后端 TF 规则 + 分类得分按 Agent 处理的物体 |
| **V3.72** | 第 5 个仿真场景 sort_items（物品分类放入红 / 蓝框，SO-101，LLM 按提示词分类）；`NERO_V372_Sort_Skills.py`；后端 RUN TASK 钩子加 sort_items 分支 |
| **V3.71** | 运动真到位 + 实测降速 + 失败收尾 / 围栏绕行；漏检重试；规划 ≥ 8° 即停；放置前复查夹着的杯子；叠好按单簇 + 实测高度；监视器实时 |
| **V3.70** | 自动采集闪退修复（Tk 主线程）、`logs/crash.log`、实机日志进 console.log、示教负载选项 |
| **V3.69** | 真机首次套好后的加固：验证前让开视线、以相机判断夹住、规划三候选取最优 + 15 s 预算、YOLOE 强制 GPU、“在夹爪里”需靠近夹爪、速度档 1X–10X、任务录像 |
| **V3.68** | 实机抓杯“抓了再判断”+ 修陈旧倾斜 bug；规划记忆角度 + 限时全角度；识别计时 / 只在失败存快照；指尖叠加校偏转角；偏转角扫描；`task/` 每步前后截图记录 + 每步日志 + 结束总结 |
| **V3.67** | 实机抓杯先对准 → 原地转腕（两指沿杯子半径）→ 两段下探；实机离杯壁 8 mm；实测安装 / 碰点指尖修正；倾斜可信度、夹爪遮挡、失败快照；4–8° 规划可执行；空夹标定 |
| **V3.66** | 实机杯子识别 = Camera 调试页 YOLOE 流水线（同一检测器 / 模型 / 取帧 / 3D 定位）+ 锥面杯轴 + 轮廓顶边量杯高 + 单杯 / 已叠好；YOLOE 不可用即终止；监控框画 YOLOE 轮廓 |
| **V3.65** | 控制台「NERO实机任务」TAB：叠杯子 · SO-101 仿真技能和参数原样上实机（RealIO）；STS3215 夹爪驱动（位置 + 扭矩上限夹紧）；V313 关节路径执行（`/execute_trajectory` + 路径围栏）；运行档位 ①②③；替身当实机身体全流程测试 |

## 8. 后续开发计划（V3.59–V3.69 已交付；V3.70 → V3.72：带噪仿真寻优与真机迁移，原编号顺延五版）

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

### 8.2 阶段 2（V3.70 起，原 V3.60；因叠杯子场景顺延）：带噪仿真自动最优化试验台（Sim-in-the-Loop Auto-Optimizer）

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

### 8.3 阶段 3（V3.71，原 V3.61；叠杯子实机部分已在 V3.65–V3.69 先做）：实机「主从视觉 + 5 cm 悬停预演（Hover Dry-Run）」安全闭环

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

### 8.4 阶段 4（V3.72，原 V3.62）：实机柔顺抓放全流程 + Agent 智能策略调度

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

### 8.5 V3 逐步完善计划（2026-11-04 与用户确认；V4 准备清单第四组优先）

用户的 V4 准备清单（22 项、8 组，原文要点）与当前状态：

| 组 | 项目 | 状态（V3.82） |
| :--- | :--- | :--- |
| 一 安全和硬件 | 1 急停 / 关节限位 / 工作空间边界 / 速度加速度上限 / 跟踪误差停止；2 SO-101 夹爪（开合、电流上限、位置 + 负载判断抓到）；3 统一机器人模型（新夹爪 TCP 与外形同一份 URDF → cuRobo 碰撞球 / MoveIt / MuJoCo） | 1 部分（围栏、限位、`ARM_NOT_AT_GOAL`、速度档）；2 部分（V3.65 STS3215 位置 + 扭矩上限，无力闭环）；3 未做（MoveIt 仍是原装夹爪，V3.75） |
| 二 感知地基 | 4 照明和相机参数（漫射光、关自动曝光 / 白平衡、防闪烁，之后采图以此为准）；5 腕部 130° 内参（鱼眼 / 高阶畸变）；6 两路手眼标定 + TCP 到检测点实测偏差；7 桌面和蛋托位姿 | 4 V3.78–V3.82 工具已就绪（一键调参 + 参数组 + 参数文件 + 镜面亮斑 / 伽马）；用户车库实测：平均亮度 < 110 且鸡蛋过曝 ≥ 2 % → 结论是光太硬（点光源 + 暗环境），建议加柔光 / 深色哑光垫，待用户实测锁定；5、6 V3.58 已有，需按“实测偏差”验收；7 未做 |
| 三 评测基线 | 8 统一记录（图像、关节、规划、agent 决策）+ 固定条件 20–30 次，按检测 / 规划 / 抓取 / 放置分别统计 | 部分（V3.68 `task/` 记录）；统计未做 |
| 四 感知精度 | 9 微调 YOLO（第 4 步之后采图）；10 D435i 深度滤波 + 多帧平均 + 表面 → 蛋中心；11 两路相机分工 + 多帧跟踪 + 抓取前重检 | 9 采集标注 V3.79–V3.80 ✓、训练评估 V3.81 ✓、应用到实机 + 标注辅助 V3.82 ✓（均待真机验证）；10 测量工具 ✓，接入 Camera 页 V3.83；11 V3.84 |
| 五 规划成功率 | 12 IK 统一 cuRobo、MoveIt 只校验；13 放开绕竖直轴旋转 + 多候选抓取姿态；14 碰撞 padding、抓取段关夹爪-目标碰撞、当前构型作 IK 种子；15 标准化预抓取位（也是 V4 交接位） | 未做（叠杯子 / 分类技能里已有部分：多候选、当前构型起算） |
| 六 动作流畅度 | 16 按轨迹时间参数执行（MuJoCo + 真机）；17 合并分段减少停顿 | 未做 |
| 七 agent 鲁棒性 | 18 每步用感知验证；19 每个技能的失败恢复（重试 / 换姿态 / 放弃）；20 LLM 只能从技能表选、结构化输出、记录每次决策 | 部分（叠杯子 verify / recover、工具契约） |
| 八 V4 准备 | 21 MuJoCo 相机内参和安装位置与真机一致；22 结合 store_in_safe 把大技能拆成原语 | 未做 |

近期版本（每版先交付可在真机 / 实物上验证的东西；顺序可由用户调整）：
- **（已交付 V3.82）精调模型应用到实机（第 9 项收尾）+ 标注辅助 + 曝光改进（2026-11-05 用户要求并入）**：Camera 页模型下拉框列出 `finetuned/` 里保存的模型（读 `model_card.json`）；精调模型按“固定类别”使用（V347 `YoloEDetector` 对它不调用 `set_classes`，按模型自带类别 + 物体名筛选，类似 -pf）；按物体名自动选模型（不在精调模型类别里的物体自动用原模型）；实机任务（V366 `yolo_cfg_from_tab`）随 Camera 页生效；一键切回原模型；会改 V347 / V325，需更新哈希校验与测试。
- **（已交付 V3.81）训练与评估（第 9 项）**：相机高级测试新页「⑤ 训练 / 评估」——从 V3.79 数据集生成 data.yaml（每类子文件夹 → 全局类别号，val 固定不参与训练）；YOLOE-26 精调（ultralytics 8.4 的 YOLOE 训练器，先在用户电脑上用一条命令确认类名 / 参数；子进程 + `clean_child_env`，进度和日志回到界面，训练时提示停掉 Camera 页实时识别）；评估：同一份 val 上原模型 vs 新模型的 mAP50 / mAP50-95（框 + 轮廓）、精确率、召回、每帧 ms，再用 ③ 定位精度比 3D 误差；结果 `metrics.json` + 对比表；模型存 `~/Downloads/objectmodels/finetuned/<名字>/`，Camera 页模型下拉框可选，原模型保留（精调模型只认训练过的类别）。
- **（已交付 V3.83）GPU 自修复 + 组合模型 + 深度滤波与几何中心换算（第 10 项）**：start_nero 启动前预检与 nvidia_uvm 自动重载、控制台「修复 GPU」按钮；修复 finetune_covers 严格匹配，多选 Menubutton 组合模型，YoloEDetector 按物体分配提示词与框去重；Camera调试页可选 5/10 帧中值深度滤波，表面点云至几何质心推算。
- **V3.84 两路相机分工 + 跟踪（第 11 项）**：固定相机三维定位、腕部相机最后对准；多帧跟踪平滑（同一物体 ID、位置滤波）；抓取前在预抓取位重检一次，偏差超限就重新规划。
- **之后（用户在实验室有机械臂时）**：第二组 5–7 的实测验收（TCP 走到检测点量偏差）、第三组评测基线（固定条件 20–30 次分段统计），再进入第一 / 五 / 六 / 七组。

### 8.6 V4 开发目标：接入 SmolVLA（2026-11-04 记录；细节待与用户逐项确认）

- **目标**：在 V3 的“规划 + 技能 + LLM 选技能”的骨架上接入 SmolVLA（LeRobot 的小型视觉-语言-动作模型）：输入两路相机图像 + 关节状态 + 语言指令，输出一段动作（action chunk）。
- **分工设想**：V3 管线负责安全、全局定位和到达**标准预抓取位 / 交接位**（第 15 项）；SmolVLA 从交接位接手最后的接近、抓取、放置等精细动作；结束后交还 V3 做感知验证（第 18 项）和失败恢复（第 19 项）。安全层（第 1 项：限位、围栏、速度、跟踪误差停止）对 SmolVLA 的输出同样生效，不能绕过。
- **数据**：用第 8 项的统一记录（图像、关节、动作、结果）在固定条件下采集示教 / 成功轨迹，作为 SmolVLA 精调数据；相机参数必须是第 4 项锁定的那组（与 YOLO 采图同一条件）。
- **仿真**：第 21 项让 MuJoCo 相机内参和安装位置与真机一致，仿真里可以先做数据和闭环验证；第 22 项把大技能拆成原语，便于和 SmolVLA 的动作段衔接。
- **前置条件**：第一组（安全、夹爪、统一模型）、第二组（标定实测验收）、第三组（评测基线）、第四组（感知精度）至少达到“可量化、可复现”，否则无法判断 SmolVLA 的效果是否来自模型本身。
- **待用户确认**：SmolVLA 负责的动作范围（只做最后 5 cm，还是整个抓放）；动作空间（关节角 / TCP 位姿增量）；控制频率；示教方式（示教模式拖动 / 遥操作 / 仿真生成）；评测指标（沿用第 8 项的分段成功率）。

## 9. 测试环境的坑（给后续开发者的提醒）

- 在没有 `tkinter` / `DISPLAY` 的云端容器环境里，独立 `test_v3_XX_gui.py` 会打印 `skipped` 并返回 0；因此必须在非 GUI 单测（如 `test_v3_57.py` X5、`test_v3_58.py` Y5）中同时包含基于 `unittest.mock` 的 `run_console('sim')`、`run_console('real')` 和 `CameraTabUI` 交互回归测试，确保任何 UI 属性初始化顺序问题在无显示器环境下也能被 100% 捕获。
- 真机 GUI 显示效果（行高、列宽、双画布缩放）最终以用户在 Ubuntu 22.04 实机上运行 `bash tests/check_release.sh` 及实测截图为准。
- 不要在测试中联网下载模型；`ultralytics` 在单测中统一使用 `tests/v347_fake.py` 里的假模块（`F.install()` / `F.uninstall()`）。
