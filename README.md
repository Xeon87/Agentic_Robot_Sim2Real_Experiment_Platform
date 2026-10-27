# NERO V3.47.1 — single-folder release

**V3.47.1**（视觉定位：鸡蛋位置不再依赖鸡蛋上的深度；标定和视觉定位设置可以另存 / 读取）
1. 原因：0.8 m 处 D435 看白色鸡蛋，深度被"压扁"（实测 17–28 mm，真实 42 mm），甚至没有深度，按高度判断就把鸡蛋剔除了。
2. 新定位方法：在每个物体周围 3–6 cm 的一圈拟合桌面平面（自动避开线缆、其他物体、没有深度的地方）；鸡蛋按实测尺寸
   （默认 55×42 mm）放在这个平面上（平躺；竖立要轮廓明显更合适或实测高度够高），把投影轮廓对齐 YOLO 轮廓
   （先对齐轮廓重心，再比形状），得到中心和长轴方向。鸡蛋上的深度只用来显示「实测高」和剔除扁平的东西（< 12 mm）。
3. 可信条件：轮廓重合 ≥ 0.80（< 0.60 剔除）、YOLO 轮廓和鸡蛋尺寸差 ≤ 8 px、不贴边、YOLO 置信度 ≥ 0.5、在安全区内、
   测到了周围桌面、手眼标定已校验。AirPods 这类干扰物主要靠视觉提示里的「非鸡蛋」参考来排除。
4. 深度设置（视觉定位页）：848×480 / 1280×720，High Accuracy / Default / High Density / Medium Density，
   断开再连接相机后生效；默认仍是 848×480 · High Accuracy（手眼标定也用同一个深度流）。
5. 手眼标定 ③：「另存为…」「读取…」，默认文件夹 `~/Downloads/cameracalib/`；「保存标定」时自动在这个文件夹存一份带时间的副本。
   新版本文件夹里没有 config/camera_calib.json 时，启动自动读取这个文件夹里最新的一份。
6. 读取后的校验：连接相机时比对相机序列号和内参；然后让法兰上的标记出现在画面里，连续 5 次误差 ≤ 5 mm → 校验通过
   （> 10 mm → 不通过）；也可以点「快速复核」。没通过校验之前，视觉定位的结果都标「不可信」。④ 的灯：黄 = 待校验，
   绿 = 通过，红 = 不通过。
7. 视觉定位设置：「保存设置」「另存为…」「读取…」：方法、模型文件夹 / 模型、门槛、文字提示、鸡蛋尺寸、深度设置、
   样本开关和参考图（图片一起存进文件）；保存到 config/locate_settings.json，并在 `~/Downloads/cameracalib/` 存一份，
   新版本文件夹启动时自动读取。


**V3.47**（Camera调试 · 视觉定位：鸡蛋识别改用 YOLOE-26 + 深度 3D 复核）
1. 识别方法下拉框：「YOLOE 视觉提示」（默认）/「YOLOE 文字提示」/「几何法 V3.08」（原来的方法保留）。
   用控制台自己的 /usr/bin/python3 里的 ultralytics（≥ 8.4），不需要安装任何东西。
2. 模型文件夹默认 `~/Downloads/objectmodels/`（可改），模型默认 `yoloe-26s-seg.pt`；加载时日志写出模型路径和 sha256。
   文字提示需要同一文件夹里有 `mobileclip2_b.ts`，没有就提示，不会去联网下载；视觉提示不需要它。
3. 视觉提示的参考：点「框选参考」（画面定格）→ 在画面上拖框，选「鸡蛋」或「非鸡蛋（干扰物，例如 AirPods 盒）」→
   「保存参考」，存到 `config/yolo_ref.json` + `config/yolo_ref.jpg`。至少 1 个鸡蛋框。框选时拖框不会改深度检查区域。
4. 3D：YOLO 掩膜向内收几个像素 → 5 帧深度融合 → 掩膜里的点用手眼标定转到 base_link → 按实测鸡蛋尺寸
   （默认 长 55 / 宽 42 mm，可填，每个半轴 ±1.5 mm 自动微调）拟合椭球，得到中心和长轴方向。
5. 几何复核（不需要训练）：每个物体周围一圈测"局部桌面"高度（安全设置的 z_table 偏了也不影响，偏差 > 8 mm 会提示更新）；
   高出桌面 34–63 mm 才像鸡蛋（AirPods 盒约 22 mm → 剔除）；拟合残差 > 4 mm 剔除；贴画面边缘、残差 > 2.5 mm、
   底部不贴桌面、YOLO 置信度 < 0.5、在安全区外 → 列出但标"不可信"。
6. 表格：编号、xyz、高度、YOLO 置信度、综合置信度、可信、说明（剔除 / 不可信原因）；画面上绿框 = 鸡蛋、
   橙框 = 不可信、红虚框 = 剔除。
7. 「保存样本」（默认关）：每次识别把画面和结果存到 `config/yolo_samples/`，以后要微调时直接用。
8. 不改：NERO_V308_Camera_Perception.py（一字不改）、仿真 / 手腕相机 / 固定相机模式、运动控制。识别只显示，不运动。


**V3.46**（修复实机第 5 步 ROS2 驱动红灯：RViz 被 OpenCV 的 Qt 插件弄崩）
1. 原因：pip 版 opencv-python 第一次 `import cv2` 时把 `QT_QPA_PLATFORM_PLUGIN_PATH` 指到自带的 `cv2/qt/plugins`；
   控制台之后启动的 ros2 launch / rviz2 继承了这个变量，加载不兼容的 Qt 插件后退出（exit -6）。V3.45 的手眼标定页
   在控制台启动时就检查一次 ArUco（import cv2），所以每次都会触发；更早的版本在连过相机之后也会触发。
2. 控制台启动的所有子进程（驱动 / MoveIt / push / 监视器 / 仿真启动脚本 / 系统命令）都去掉 Qt 变量里指向 cv2 的部分。
3. Camera调试页通过一个保护函数 import cv2，导入后把 Qt 变量恢复原样；页面创建时不再 import cv2，连接相机后才检查 ArUco。
4. 驱动日志里只是 rviz2 "process has died" 不再判驱动失败：日志给出警告，第 6 步 MoveIt 显示黄灯（AMBER），一键启动继续；
   move_group / 驱动节点 / ros2_control 退出仍然是红灯。
5. 环境建议（可选）：`pip3 uninstall -y opencv-python opencv-contrib-python && pip3 install --user opencv-contrib-python-headless`
   （无界面版不改 Qt 变量，ArUco 照样可用）。


**V3.45**（Camera调试 · 手眼标定：界面按步骤重排 + 采集更稳 + 自动采集只选看得见的姿态）
1. 手眼标定页改成从上到下四步，每步有状态灯（灰 = 未开始 / 黄 = 进行中或不够 / 蓝 = 机械臂在动 / 绿 = 完成 / 红 = 不合格），
   按钮按状态自动变灰：① 标记设置（字典、ID、边长 40 / 30 mm 预设，默认 40×40）② 采集样本（记录当前姿态 / 自动采集 / 停止，
   样本表）③ 计算与保存（合格才显示「保存标定」，不合格显示「仍然保存（未达标）」）④ 复核。
2. 样本表：行高按字体自动设置（不再重叠）、竖向 + 横向滚动条、隔行底色、来源（深度 / PnP）和抖动列，红色 = 被剔除，
   橙色 = 留一误差 > 5 mm；新样本自动滚到可见。结果框可滚动，合格绿色 / 未达标红色 / ⚠ 橙色。
3. 测量：每个姿态 15 帧，只用一种来源（≥ 80% 帧有可靠深度才用深度，否则全部用 PnP，不再逐帧切换）；先去掉坏帧
   （偏离中位数 > 3 倍中位偏差），再看剩下帧的抖动（≤ 3 mm）；运动后多等 1.5 s 并确认关节不动；没过就再测一次。
4. 自动采集：先用现有标定（起点误差 ≤ 40 mm 时）或 3 个 4 cm 小平移确定相机方向，再在起点附近选姿态：法兰位置铺开
   ≥ 20 cm、手腕转动 ≤ 25°、预测标记在画面内（留 8% 边）、正对相机 ≤ 45°、距离 0.35–1.3 m、IK 有解、离限位 ≥ 3°、
   安全区通过；按最远点铺开、按关节距离排序。日志写出选了几个、淘汰原因、位置范围和最大手腕转动。
5. 修复 `[FAIL] [V3.25 camera] refresh: 'NoneType' object is not subscriptable`：检测线程随时把标记结果换成 None，
   画面刷新在"判断有标记"和"读取角点"之间被换掉；现在每次刷新只取一次快照，叠加层各自容错，没有安全区时不再报错。
6. 朝向变化改为任意两个样本之间的最大差（原来只和第 1 个比）。


**V3.44**（场景在控制台启动前选好；去掉 V3.43 / V3.43.1 的仿真内切换和自动重启）
1. NERO 控制台「仿真 (MuJoCo)」页的启动按钮左边新增"场景"下拉框：`fill_egg_holder`（默认，每次打开控制台都是它）/
   `store_in_safe`。点「▶ 启动仿真」时把选择作为 `NERO_SCENE` 传给 start_nero_sim.sh；仿真运行中下拉框锁定。
   要换场景：「■ 停止仿真」→ 选场景 →「▶ 启动仿真」。
2. 直接运行 `./start_nero.sh --sim` 时：`NERO_SCENE=store_in_safe ./start_nero.sh --sim`；不设 = 鸡蛋场景；
   未知的场景名会在启动前报错退出。
3. 删除：SimAgent 页顶部的场景下拉框和"切换并重启"、backend 退出码 75、launcher 自我重启 / 重启请求文件 /
   v343_before_switch 日志备份、logs/v343_scene.txt。仿真窗口里只在日志打印一行当前场景。
4. 保留：store_in_safe 场景本身（V3.43）、门的角度单位修正和启动失败时打印 backend.log 末尾（V3.43.1）、
   store_in_safe 下 RUN TASK 拦截（还没有 Agent 技能）。
5. 校验（tests/test_v3_44.py）：run_sim 每步物理循环（B）与 V3.42 逐字相同；backend 去掉 11 个 `# V3.43 A2` 块后
   与 V3.42 逐字相同；默认场景下三份列表和生成的 XML 与 V3.42 相同。

**V3.43**（MuJoCo 场景切换 + RoboDojo `store_in_safe` 场景；鸡蛋场景和物理循环不变）
 0. （V3.44 起场景改在控制台启动前选择，下面第 1 条的仿真内切换已删除）
1. NERO SimAgent 页面顶部新增"MuJoCo 场景 / Scene"：`fill_egg_holder`（默认，鸡蛋放入蛋托）/
   `store_in_safe`（打开保险箱门 → 放入首饰盒 → 关门）。选好后点"切换并重启仿真"：选择保存到
   `logs/v343_scene.txt`，整条仿真链（MuJoCo + MoveIt + 相机 + GPU 规划）自动重启（backend 退出码 75 →
   start_nero_sim.sh 重新执行自己），之后一直沿用，直到再次切换。环境变量 `NERO_SCENE=<场景名>` 可强制指定（此时下拉框只读）。
2. store_in_safe 场景（NERO_V343_Scenes.py，所有尺寸为参数）：
   - 保险箱 26×20×22 cm，15 mm 钢壁（safe_body_*，受穿透保护），红色绒垫；
   - 门：铰链在右侧（y=0.30），向机器人方向开 0–110°，阻尼 0.6 + 摩擦 0.08，开到哪停到哪；门与桌面/箱体不碰撞（铰链约束）；
   - 竖直把手 Ø18 mm × 90 mm，离门面 35 mm（独立 body `obj_safe_handle`，可用原 CONNECT 夹持）；
   - 首饰盒 70×50×45 mm、0.12 kg（`obj_valuables_box`，自由体），放在桌上 (-0.36, -0.17)；
   - MoveIt 里发布箱体 5 块墙 + 随门角度更新的门板/把手碰撞盒；TF：sim_obj_valuables_box / sim_obj_safe_handle / sim_obj_safe_door；
   - 页面实时显示：门角度、首饰盒位置、是否在箱内、门是否关好、任务是否完成。
3. 本版只提供场景：store_in_safe 下 RUN TASK 被拦截（Agent 技能在 V3.44）。可用 Basic Controls / Skill Test 手动检查物理。
4. 不变的部分（测试校验）：run_sim 每步物理循环（B）与 V3.42 逐字相同；backend 去掉 12 个 `# V3.43 A2` 场景块后与 V3.42
   逐字相同；默认场景下 TF 映射 / 蛋托碰撞盒 / 可夹持列表和生成的 XML 与 V3.42 相同。
5. 说明：RoboDojo 原任务是 ARX X5 双臂真机任务，论文附录里的具体物体未能获取，本场景按任务名复现、改为单臂版本。


**V3.42**（任务统计 + 失败分析；DeepSeek / Kimi；步数 60 + 用完可追加。抓取 / 放置策略不变）
1. 任务结束（成功或失败）打印 `[V3.42 TASK STATS]`：总时间；LLM 思考 / 机器人技能 / 其他 的时间和比例；
   每个技能（次数、时间、比例）；每个蛋的技能时间；本任务的 token（input / cached / output / 总数、请求数）。
   任务失败时再调用一次 LLM（不带工具、只给精简轨迹），打印 `[V3.42 FAILURE ANALYSIS]`：首要原因 / 证据 / 建议。
   用户 STOP 的任务不分析。NERO_V342_FAILURE_ANALYSIS=0 关闭；NERO_V342_ANALYSIS_LANG=en 改英文。
2. LLM Provider 下拉框：OpenAI（Responses API，原路径不变）/ DeepSeek / Kimi（OpenAI 兼容 Chat Completions）。
   - DeepSeek：https://api.deepseek.com，key = 输入框或 DEEPSEEK_API_KEY，预设 deepseek-v4-pro / deepseek-flash；
     Reasoning=none 关思考，其余映射到 low / high / max。
   - Kimi：https://api.moonshot.ai/v1（国内 NERO_KIMI_API_BASE=https://api.moonshot.cn/v1），key = 输入框或
     MOONSHOT_API_KEY / KIMI_API_KEY，预设 kimi-k2.6 / kimi-k3。
   - 模型名可编辑，"Refresh API Models" 列出该 key 可用的模型。每个 provider 的 key 在 GUI 运行期间分别记住。
   - 思考模型的 reasoning_content 会随历史回传（带工具时必须）。
3. 最大步数默认 60（不再按蛋数自动加到 64；NERO_V342_AUTO_BUDGET=1 恢复）。步数用完、或模型想结束但还有蛋没放
   且剩余步数不够时，弹窗询问是否增加、增加多少（+10/+20/+30/+50 或自定义）；不增加则按原逻辑安全结束。
   10 分钟无回应视为不增加（NERO_V342_EXTEND_TIMEOUT_S）。


**V3.41**（关节限位余量；插入被挡如实报失败；近距离视图不判蛋托移动；松开位置真实；找回的蛋不再丢）
影响范围：1、2 对所有模式生效（运动层）；3 对两种腕部相机生效；4 只在无真值模式（腕部 / 固定相机）生效；
5 只在腕部相机生效；V3.39 预测关联恢复为只在 USB（RGB-only）模式生效，D435i 回到 V3.38 行为。
1. 关节限位余量：槽上方姿态、下降走廊每一层（现在延伸到入口下方 30 mm，NERO_V341_CORRIDOR_DEPTH_M）、竖蛋整臂
   旋转的 IK 筛选，都要求 7 个关节离限位 ≥ 8°（NERO_V341_LIMIT_MARGIN_DEG）；一个都没有时退到 ≥ 4°
   （NERO_V341_LIMIT_MARGIN_MIN_DEG，日志写明 RELAXED）；余量大的 yaw 优先。cuRobo 终点离限位 < 8° 而筛选解
   余量更大 → 改用 MoveIt 关节目标到筛选解。日志 `[V3.41 LIMIT MARGIN]` `[V3.41 CORRIDOR MARGIN]` `[V3.41 ENDPOINT MARGIN]`。
   NERO_V341_LIMIT_MARGIN=0 关闭。
2. 插入被挡：CARTESIAN APPROACH BLOCKED 不再当成功。先用 BioIK 逐层（5 mm、保持姿态、同一分支 ≤ 12°）+ MoveIt
   关节目标；仍下不去（或一步下降 < 40%）→ 退回入口高度，返回 INSERT_BLOCKED_REPLAN_YAW（plan_placement →
   orient_for_slot 跳过当前 yaw 重规划，最多 1 次）；再挡 → INSERT_BLOCKED_JOINT_LIMIT（附 7 个关节余量），
   建议 return_to_pick_location。日志 `[V3.41 INSERT IK STEP]` `[V3.41 INSERT BLOCKED]`。
3. 近距离视图不判蛋托移动：live / precheck 视图在相机离蛋托 < 0.28 m（NERO_V341_HOLDER_MIN_CAM_DIST_M）、
   蛋托不完整或碰图像边缘、或放置进行中时，一律不触发"疑似移动"。插入阶段只认 plan_placement 的专门视角。
   日志 `[V3.41 HOLDER VIEW IGNORED]`。
4. 松开时记录真实位置：持蛋期间每次机械臂运动后都用当前 TCP 更新持蛋位姿，return_to_pick_location 松开前再更新
   一次，不再误报 RETURN FAILED / EGG LOST。NERO_V341_HELD_REFRESH=0 关闭。
5. 丢失的蛋只忘一次：重新看到 egg_1 后，后续仍带 lost 提示的图像不再把它忘掉；Agent 把它当作普通待处理蛋。


**V3.40**（USB：去掉感知噪声导致的误报；cuRobo 拒绝时有退路）
1. 桌面高度每个任务只定一次：先用标定桌面；≥ 2 个测得好的躺蛋与标定差 ≤ 5 mm 时改用测量值并锁定。蛋高度检查、
   蛋托检测、蛋托测高都用这同一个值（V3.39 里桌面漂了 17 mm，所有蛋 z 跟着跳）。
2. 邻蛋碰动检查（感知模式）：只看水平位移，减去 2 × 射线 rms（最多 6 mm）；新视角里没看到的邻蛋不判。
   真值模式不变。NERO_V340_BYSTANDER_XY=0 关闭。
3. 拿蛋看蛋托：这张图与融合蛋托差 ≤ 6 mm 才进滤波微调，否则忽略（日志 `[V3.40 PRE-PLACE VIEW IGNORED]`），
   不再出现假的 SLOT_MOVED_SINCE_PLAN。
4. cuRobo 失败：打印起始状态诊断 `[V3.40 CUROBO DIAG]`（关节限位余量、MoveIt 判定是否碰撞），并改用 MoveIt 关节
   目标（已筛过的 BioIK 解）/ BioIK 位姿规划兜底：竖蛋整臂旋转、居中挪动、return_to_pick_location。
   日志 `[V3.40 MOVEIT FALLBACK]`。NERO_V340_MOVEIT_FALLBACK=0 关闭。
5. 拿蛋看蛋托时机械臂没跟上轨迹（EXEC VERIFY FAILED）→ 立即停止换视角，用融合蛋托。日志 `[V3.40 PRE-PLACE LOOK STOP]`。

**V3.39**（腕部 USB 相机：更快找到蛋托，抓取高度正确）
1. 扫描提速：
   - 网格扫描中一看到部分蛋托就立刻补拍（不再等整圈扫完）；常规补拍视角够不到时，再试靠近机器人底座一侧
     和沿蛋托长轴 ±7 cm 的视角。日志 `[V3.39 HOLDER REFINE]`。
   - 两张局部图拼出完整蛋托：一张图里某个方向的远边完整，另一张图里另一个方向的远边完整，且朝向一致（≤ 4°）、
     重叠方向一致（≤ 8 mm）→ 得到完整蛋托（method rgb_carton_far_edges_fused_partial_views）。
   - 蛋托找到但高度未测出 → 同一次扫描里补两张高度视角（中心沿长轴 ±0.10 m）。日志 `[V3.39 HOLDER HEIGHT]`。
   - 第二次 observe 跳过上次什么都没看到的网格视角。日志 `[V3.39 SCAN REUSE]`。NERO_V339_SCAN_FAST=0 关闭以上。
2. 桌面高度兜底：躺着的蛋一律检查 z = 桌面 + 半个蛋宽。桌面用测量值（只用条件好的三角化：射线夹角 ≥ 12°、
   rms ≤ 4 mm），与标定桌面（NERO_V330_TABLE_Z_M）差 > 15 mm 或没测出时用标定值；蛋宽没测出时用类别先验半宽。
   V3.38 两者缺一就不检查，所以 egg_1 用了 z 0.048（真值约 0.022）。
3. 三角化更稳：
   - 一致性选射线：所有射线两两求交，取互相一致（≤ 6 mm）的最大一组，蛋堆里认错的射线被剔除（NERO_V339_TRI_INLIER_M）。
   - 三角化失败（射线近平行 / 混入邻蛋）时，用射线与"桌面 + 半宽"平面求交：至少两个相距 ≥ 5 cm 的相机位置结果一致
     （≤ 8 mm）即算测到并可信（method rays_at_table_plane）——修复 egg_2 有 10 个视角却一直 not triangulated。
4. 近距跟踪：夹爪遮挡导致"不可信"的检测，若离某个蛋的测量位置 ≤ 45 mm 且明显比离其他蛋近（1.4 倍），就关联给它，
   不再出现 `seen []`（payload tracking.predicted_assoc）。NERO_V339_PREDICT_ASSOC=0 关闭。
5. 抓取前检查（仅 USB）：plan_grasp 拍完先判断目标——没看到 / 未测到 / 测量值与本视角相差 > 10 mm / 躺蛋 z 偏离
   桌面 + 半宽 > 10 mm——就地补拍最多 2 张近景（不回 q0），之后再回 q0 规划。日志 `[V3.39 GRASP VIEW]`。
   高度仍偏 > 10 mm 时拒绝规划：GRASP_HEIGHT_IMPLAUSIBLE（Agent：observe_scene 一次后重试，再不行换蛋）。

**V3.38**
1. USB 相机修复：V3.37 里蛋托高度测量的 `extent_ok` 是 numpy 布尔值，相机结果无法转成 JSON，USB 模式第 3 个扫描视角起
   每次拍照都报 PERCEPTION_ERROR。已改成普通布尔值；另外所有相机结果（腕部 / 固定 / 监视窗）发布前都经过 json_safe()
   转成普通类型。
2. 网络中断不再终止任务：调用大模型 API 时遇到 DNS 失败、连接中断、超时、HTTP 429/5xx，按 2/4/8/16 s 退避重试
   （NERO_V338_API_RETRY_DELAYS_S），同一个请求重发，任务原地继续；401、返回格式错误等立即失败；按 STOP 结束等待。
   日志 `[V3.38 API RETRY]`。
3. 竖蛋前的居中挪动：BioIK 找到 0 个可行解时不再先挪（先让 cuRobo 原地规划，V3.37 日志里它能成功）；只有 1–2 个
   可行解或关节要大幅转动时先挪；原地规划失败后也会挪动再规划。挪动目标有 5 个候选点 × 3 个高度
   （NERO_V338_RECENTER_POINTS / NERO_V338_RECENTER_Z_M），先用当前抓取姿态做 IK 筛选再规划。
4. 槽位精度（主要针对 USB）：
   - 蛋托高度测出后，用所有视角在该高度上融合出的蛋托中心修正本视角的槽位（偏差 ≤ 15 mm 才用，挪动检测不受影响）。
   - orient_for_slot / move_to_slot_entry 前：本视角完整看到蛋托、且槽位偏离计划 4–15 mm 时补拍一次，两次一致
     （≤ 3 mm）就把目标槽位改成两次平均值，日志 `[V3.38 SLOT REFINE]`（NERO_V338_SLOT_REFINE=0 关闭）。
   - 拿着蛋看蛋托：按 0.45 / 0.40 / 0.52 / 0.36 m 逐个距离找视角，看不到槽位就看蛋托中心（NERO_V338_PREPLACE_DIST_M）。


**V3.37**
1. 难姿态先挪到中间再竖蛋：orient_for_slot 的整臂旋转在可行 IK 解 < 3 个（NERO_V337_MIN_FEASIBLE），或最优解某个关节
   要转 > 150°（NERO_V337_MAX_JOINT_TRAVEL_DEG）时，先保持抓取姿态把鸡蛋平移到桌子中间（NERO_V337_RECENTER_XY，
   默认 -0.42,-0.02，高度 ≥ 0.23 m），再重新求解竖蛋。日志 `[V3.37 RECENTER]`。NERO_V337_RECENTER=0 关闭。
2. 安全恢复不再丢蛋：仿真里关节失控触发的自动硬恢复，夹着的鸡蛋随夹爪一起恢复（相对 link7 的位姿不变），力控模式和
   HOLD 保持。3 s 内第二次硬恢复、手动 HARD RESET 0 仍按原来的方式松开。日志 `[V3.37 RESTORE KEEP HOLD]`。
   NERO_V337_RESTORE_KEEP_HOLD=0 用回旧行为。除硬恢复函数外，仿真进程与 V3.34 逐字节一致（测试校验）。
   return_to_pick_location 在安全锁未解时先自动 STOP + RECOVER（MuJoCo 确认、控制器同步）再动作。
3. 掉蛋处理：鸡蛋在非松开技能里离开夹爪 → 标记 LOST（scene 里 location=lost_unknown、没有坐标），plan_grasp 拒绝
   （EGG_LOST_NOT_RELOCATED）；下一次 observe_scene 先看掉落点和周围一圈（NERO_V337_SEARCH_R_M 0.12 m，最多 5 个视角），
   相机跟踪器丢掉它的旧轨迹，重新看到时恢复原名。两次没找到 → Agent 先做其他蛋并报告。
4. GUI：标签页改名 "NERO SimAgent"；每个功能区点标题可折叠/展开，另有 Collapse all / Expand all，
   NERO_V337_COLLAPSED=debugger,diagnostics 可设启动时折叠的区；腕部相机监视窗缩放 25–300%（下拉框、+/-、Ctrl+滚轮），
   弹出窗口随窗口大小缩放。


**V3.36**
1. 抓取物理回退到 V3.34：指垫摩擦 10、默认求解器、自动"粘住"（Enable Auto HOLD）、V3.34 的 clamp 判断。
   V3.35 的摩擦抓取代码全部去掉（仿真进程和夹爪构建函数与 V3.34 逐字节一致，测试 Q1 校验）。
2. 鸡蛋识别（USB 纯 RGB）：
   - 新的蛋壳颜色掩膜不再要求亮度 ≥ 0.22（旧掩膜会丢掉背光的下半个蛋，三角化中心偏高、宽度偏小）；只看色相/饱和度，
     桌面、阴影、蛋托、地板饱和度低，仍被排除。NERO_V336_EGG_MASK=0 用回旧掩膜。
   - 色块先补洞（高光、标签），用填充后色块的椭圆中心；凸度低的大色块拆成两个蛋。
   - 尺寸只用孤立、完整、未被手臂遮挡、凸度 ≥ 0.88 的色块，且至少 2 个视角一致（差 ≤ 25%）；否则"未测得"，
     开口用占位尺寸 + 4 mm 的保守余量（在 scene 里注明）。
   - 蛋的朝向由多个视角融合；平躺的蛋，三角化高度和"测得桌面 + 测得半宽"相差 > 8 mm 时，用该高度重新求交。
3. 冻结：plan_grasp 成功时冻结该蛋的位姿/尺寸/开口要求，到夹住为止；plan_placement 成功时冻结槽位，到松开为止。
   实时刷新只检查"有没有被挪动"，不再改正在执行的数据（不会再出现"规划完又说开口太窄"）。
4. 蛋托：
   - 高度测量至少 3 个视角、相机相距 ≥ 10 cm、视线夹角 ≥ 15°、最小值要明显、比桌面高 20–120 mm、外形不小于模型 85%；
     测准后锁定，蛋托被确认挪动后才重测。
   - "被挪动"：live/预检视角只能提出怀疑；需要两个相机位置相距 ≥ 5 cm 的视角一致、且至少一个是专门观察的视角才确认。
   - 放置中有怀疑 → HOLDER_POSSIBLY_MOVED（插入中则停下抬回），Agent 重新 plan_placement（先专门看一眼蛋托）。

**V3.35.1**（修复 V3.35 启动后 MuJoCo 闪退）
- 原因：V3.35 在 MjModel/MjData 已经建好之后才在运行时改 cone=elliptic / impratio / noslip_iterations，
  在你的机器上导致仿真进程直接崩溃（关掉这组设置就不崩，已由 NERO_V335_FRICTION_SOLVER=0 对照确认）。
- 现在这些设置和指垫/鸡蛋的摩擦系数在加载前写进模型文件（`*_v335phys.xml` 的 `<option>` 和 geom friction），
  运行时不再修改任何物理参数。启动日志：`[V3.35.1 FRICTION PHYSICS] from the model file: cone=elliptic ...`。
- NERO_V335_FRICTION_SOLVER=0 仍可关闭这组求解设置。

**V3.35**（减小 sim-to-real 差距：摩擦抓取、实时感知、相机误差、测量代替先验、相机缺失不回退真值）
1. 抓取：鸡蛋只靠摩擦夹住，默认不再用仿真"粘住"（CONNECT 约束）。
   - "MuJoCo 没有静摩擦"这个说法不准确：MuJoCo 有库仑摩擦，也能"粘住不滑"；但它的接触是软约束，默认金字塔摩擦锥 +
     impratio=1 时，持续的切向力（蛋的重量）会造成缓慢蠕滑——这就是以前怎么加力都慢慢滑落的原因（pad 摩擦系数 10 也没用）。
     MuJoCo 自己的解法：椭圆摩擦锥 + impratio>1 + noslip 求解器。V3.35 启用：cone=elliptic、impratio=10、
     noslip_iterations=8；pad 摩擦系数改为接近真实的 0.9，蛋 0.5（接触取两者较大值）。
   - "抓住了"= 夹爪在力控模式下两侧 pad 都压到同一物体（真实夹爪靠电流/力 + 位置也能得到）。clamp_gripper 等到两侧
     力达到目标的 70% 才算成功；夹着时蛋掉了 → [V3.35 GRASP LOST]，位置未知，直到相机重新看到。
   - 不再用仿真真值做"防滑加力"。GUI 夹爪页的"Sim assist HOLD"勾选可临时恢复旧的粘住（NERO_V335_ASSIST_HOLD=1）。
   - 参数：NERO_V335_IMPRATIO / _NOSLIP_ITER / _PAD_MU / _EGG_MU / _GRASP_SENSE_N / _CLAMP_FORCE_FRAC；
     NERO_V335_FRICTION_SOLVER=0 回到默认求解器。仿真变慢时可先把 NOSLIP_ITER 调小。
2. 实时感知：腕部相机在后台持续拍照刷新场景（默认 1.5 Hz，Agent 页可关/调频率）。
   - 每个依赖目标位置的动作前再拍一张核对：蛋被挪了 → TARGET_MOVED_SINCE_PLAN（不动，重新 plan_grasp）；
     蛋托被挪了 → SLOT_MOVED_SINCE_PLAN（蛋还在手里，重新 plan_placement）；插入过程中蛋托被挪 →
     SLOT_MOVED_DURING_INSERTION，停止下插、抬回入口高度。
   - 蛋托位姿 = 最近几次完整视图的融合；一次偏离不算，两次一致的偏离才认定"被挪动"。
   - 手里的蛋：位置仍由抓取时的手内关系推算，但相机持续核对（夹爪与相机刚性相连，蛋在图像里移动 = 在指间滑动/转动），
     偏移大时警告，>35 mm 连续两次则停止。另外夹爪力感知判断是否还夹着。
3. 相机参数误差：Agent 页 Pose source 旁边加了连续滑块"Camera calib error"（0–3，默认 0.3）。感知使用"标定值"，
   与真实（渲染）相机存在误差：焦距、主点、残余镜头畸变、手眼外参平移/旋转。每次会话误差方向固定，大小随滑块缩放；
   滑块下方显示当前误差。合成测试：0.3 级蛋位置误差约 5 mm，1 级约 12 mm，3 级约 35 mm。
   新增"Wrist camera monitor"框：实时显示相机画面，标签全部来自系统的识别结果（跟踪名、测得尺寸、"等待第二视角"的
   未命名检测、蛋托矩形和槽位），可弹出 640x360 窗口。
4. 测量代替先验：
   - 每个蛋的位置由多个视角的射线三角化得到（不再假设蛋在桌面 + 蛋半径的平面上），只有一个视角时不信任（plan_grasp 会自动再看一眼）；
   - 蛋的长宽由三角化深度下的图像尺寸测得，夹爪开口按测得宽度设定；不再使用"鸡蛋类别尺寸"；
   - 桌面高度由测得的蛋推出；蛋托顶面高度用多视角视差测得（不用 CAD 高度）；
   - 是否是鸡蛋由 Agent 根据测得尺寸判断（Agent 规则 V3.35）。
   - 仍保留的模型：蛋托 2×4 槽位相对外形的排布（槽中心没有被测量，scene 里注明 slot_layout_source）、
     一团颜色块拆分成几个蛋时用的面积参考。
5. （第 5 点按你的要求留到实机测试）
6. 相机缺失不再回退真值：选了腕部（或固定）相机但进程没起来 → RUN TASK 直接拒绝；任务中相机掉线 → 工具返回
   PERCEPTION_UNAVAILABLE 并停止 Agent。只有手动选择 "JSON ground truth" 才用真值。日志/scene 里的 pose_source 改为实际相机
   （camera_wrist_usb130_rgb_perception 等），不再写 D435i。

**V3.34**（启动时 CUDA 报错自动修复，不用重启电脑）
- 原因：`nvidia-smi` 正常、但 PyTorch 报 "CUDA is not available"，几乎都是 NVIDIA 的 `nvidia_uvm` 内核模块状态坏了
  （多发生在电脑睡眠唤醒之后，或上次 CUDA 程序异常退出）。重启能好，是因为重启重新加载了这个模块。
- 现在启动脚本在 CUDA 自检失败时自动处理，按顺序：
  1. 等 2 秒再查一次（偶发的直接过）；
  2. 结束残留的 NERO GPU 进程，再查；
  3. 检查还有谁占着 `/dev/nvidia-uvm`：如果是别的程序，只列出来、不动它，提示关掉后再启动；
  4. 重新加载 `nvidia_uvm`（需要管理员权限：能免密 sudo 就直接做；否则弹出系统密码框 pkexec；
     从终端启动则在终端里输密码），只问一次；
  5. 再查 CUDA，好了就继续正常启动。
- 日志里是 `[V3.34 CUDA AUTOFIX] ...`。仍然不行时会给出手动命令：
  `sudo modprobe -r nvidia_uvm && sudo modprobe nvidia_uvm`，以及防止睡眠后复发的建议。
- NERO_V334_CUDA_AUTOFIX=0 关闭；NERO_V334_CUDA_PRIV=auto|sudo|pkexec|none 选择获取权限的方式。


**V3.33**（让 USB 纯 RGB 腕部相机能用）
1. 场景：
   - 地板由橙色木纹改为中性偏冷的灰色（木纹的色相/饱和度和棕壳蛋几乎一样，是上次假蛋的主要来源）。NERO_V333_FLOOR=wood 可恢复。
   - 桌面改为纯白哑光，去掉大理石纹（深色纹路像洞或阴影）。NERO_V333_TABLE=marble 可恢复。
   - 阴影贴图 8192（原来手臂的阴影是一格一格的块状），头灯环境光调高，阴影变成浅灰而不是深灰。
2. 蛋托改成超市常见的灰色纸浆蛋托（参照模压纸浆蛋托的结构：蛋窝贴合鸡蛋、窝之间是平的鞍部、两排之间有三个
   方锥形支柱、四周直壁）。只改外观：外观是一个网格模型；原来的底板/四边/每格 6 根立柱仍然负责物理接触，但已隐藏，
   所以鸡蛋放进去的高度和整个放置流程完全不变。NERO_V333_PULP_CARTON=0 恢复旧外观。
3. USB 相机识别蛋托改为看"纸浆灰色区域 + 已知尺寸"：把灰色区域投到蛋托顶面，拟合最小外接矩形；靠近相机的外侧壁会让
   矩形向外偏，所以位置只用背离相机的两条边（远边精确）再减去半个尺寸。两条远边都在画面内、尺寸够、8 个格都在画面内
   才算"完整"。合成测试里所有"完整"的检测误差都在 2 mm / 1° 以内。D435i 深度识别也改成认这个蛋托外形。
4. 纯 RGB 检测到的新蛋，要在两个不同视角、同一位置（25 mm 内）都看到才给名字，单张图里的颜色误检不会再占用名字。
5. D435i 腕部相机改装到 USB 相机同一侧；非腕部模式不画腕部相机。
6. 流程：
   - 扫视网格去掉永远够不到的两个近点（0.23 m），本任务内够不到的视角不再重试。
   - 蛋托只看到一部分时，最多再看 3 次（常规距离 / 更远 / 换方向），目标是"实际看到的那部分的中心"；完全没看到时，
     再走 4 个错开半格的点找蛋托。
   - 还是找不到：observe_scene 报 HOLDER_NOT_PERCEIVED，只允许再找一次，之后让 LLM 停下报告；完成检查（GUARD）读取
     当前场景（不再整轮扫视），没有蛋托时不再催。上次 7、8 分钟就是这样反复整轮扫出来的。
   - 新的完整蛋托视图会替换旧的（不会被一次错误的"完整"永久锁死）。
   - 任务日志里 USB 模式显示为 "wrist camera USB 130deg RGB-only perception"。


**V3.32**
1. 启动脚本可执行权限修复：V3.31 / V3.31.1 的 zip 里 start_nero.sh、start_nero_sim.sh 丢了可执行位
   （打包方式的问题，脚本内容没变），所以右键「作为程序运行」失效。V3.32 的 zip 重新带上 rwx，测试 E1 会检查。
   旧包可临时用 `chmod +x start_nero.sh start_nero_sim.sh` 补救。
2. 无先验模式（腕部相机 / V3.31 固定相机）不再假设有几个蛋：
   - 本次任务的蛋 = 本次扫视/拍照中实际被感知并命名的蛋（加上正在夹着、已放进蛋格的）。名字池里没被看到过的名字
     （比如掉到地上的 egg_4）在场景快照、verify 的"剩余蛋"、完成检查里都不存在。
   - 完成检查（MULTI-OBJECT GUARD）只看感知到的蛋：桌上只有 3 个就做完 3 个结束，不会再反复 observe_scene
     去对准蛋堆和蛋托找第 4 个。
   - plan_grasp 一个从没看到过的名字时立即拒绝（OBJECT_NOT_PERCEIVED_IN_THIS_EPISODE），机械臂不动。
   - LLM 规则里说明：scene.objects_policy=perceived_only 时，列出的蛋就是全部。
   - 有先验的模式（V3.11 固定相机老路径 / JSON 真值）不变。


**V3.31.1**（修正腕部相机安装位置）
- 原来两个腕部相机沿 link7 z 方向偏出去（D435i +70 mm、USB −50 mm），正好是手指滑动的方向，看起来悬空在导轨外侧。
- 现在装在夹爪底座的两个侧面上（与手指滑动方向垂直）：D435i 在 +y 面，USB 板在 −y 面；高度与夹爪中心线对齐；
  前端与底座前端齐平；加了一块支架把相机连到底座上；光轴约 9° 内倾，在夹爪前方 0.40 m 处与夹爪中心线相交。
- 底座侧面和前端位置直接从 gripper_base.stl 读出（与后端用同样的路径搜索），找不到 STL 时用名义值。
  启动日志 `[V3.31.1 WRIST MOUNT]` 会打印两个相机的光心位置和数据来源。
- 相机外参随安装位置一起变（仿真渲染、视角规划用同一个配置）；NERO_V326_MOUNT_POS / NERO_V330_USB_MOUNT_POS 仍可手动覆盖。
- 其余功能与 V3.31 相同。


**V3.31**（仿真，腕部 + 固定相机都去掉"作弊"）
1. 腕部模式：
   - 每个任务开始时布局真正"未知"：蛋托/蛋格/蛋堆在被看到之前不存在——位置分类不给蛋格名、姿态检查不用
     near_basket / near_holder（只剩高度 + 关节裕度）、快照里蛋托和蛋格坐标为空。
   - 扫视改为整张桌面工作区内、臂可达环形范围（0.20–0.62 m）的均匀网格（8 个视角，近排先、蛇形），不再只扫前方扇区。
     NERO_V331_SCAN_SPACING_M / _R_MIN_M / _R_MAX_M 可调；设 NERO_V329_SCAN_AZ_DEG 恢复旧的环形扫视。
   - 蛋在格里的高度、放下后的位置 = 蛋托模型（立柱顶、内圈半径、窝顶）+ 蛋类别尺寸算出（54 mm），不再用经验常数 0.011。
   - USB RGB 模式蛋格"暗窝"阈值改为相对图像亮度自适应（NERO_V331_WELL_DARK_REL）。
   - 扫视停止不再看"名字用完"；名字池不够时报 `[V3.31 UNNAMED EGGS]` 并继续扫。
2. 固定支架相机模式（默认 NERO_V331_FIXED_NO_PRIOR=1）按腕部模式改：
   - 新模块 NERO_V331_Fixed_Camera.py（V3.08 模块一个字节没改）：不再用仿真器给蛋配名（按位置从名字池取名）、
     进程里删掉登记的蛋格/蛋篮，蛋托从深度图识别；收发 held/placed 提示。
   - 后端：检查不用真值（同腕部的估计器：感知 / 夹持时 TCP×抓取变换 / 放下的蛋格）；蛋托、蛋格、蛋堆、MoveIt 蛋托
     碰撞盒都来自感知；放蛋前固定相机补拍一张（不动臂）。真值只进日志。
   - NERO_V331_FIXED_NO_PRIOR=0 回到 V3.11 固定相机老路径。
- 日志：`[V3.31 FIXED CAMERA] READY (no priors ...)`、`[V3.31 FIXED VIEW]`、`[V3.31 FIXED PERCEPTION]`。
- 仍然保留的仿真便利（未改）：夹持焊接、按接触判安全、相机外参精确、深度无噪声。


**V3.30**（仿真，腕部相机）
1. 第二轮"作弊"排查（腕部模式）并改掉：
   - 蛋的名字不再由仿真器真值配对产生：第一次看到的蛋按扫视顺序（x、y 位置）从名字池 egg_1…egg_4 里取空名，
     之后由跟踪器保持（仿真里的夹持是按接触算的，与名字无关）。真值对比日志改为"离它最近的仿真蛋"。
   - 每个蛋的精确尺寸（来自 MJCF）不再交给夹爪开度 / 放置：所有蛋用同一个"蛋类别"模型（长 63、宽 43.8+1.5 mm）。
   - 扫视结束条件不再假设蛋的数量：蛋托完整 + 蛋堆已知 + 连续 2 个视角没有新蛋（或名字已用完）。
   - 保留并说明的先验：机器人模型 + 关节角（自身遮挡、相机外参）、桌面工作区范围、蛋托型号、蛋的类别外观/尺寸、
     名字池（任务里的物体登记表）。
2. 新增第二套腕部相机：InnoMaker U20CAM-1080P（USB 2.0 UVC，1080p30，M12 镜头 130°，32×32 mm 板，**只有 RGB、没有深度**）。
   - 仿真按"鱼眼去畸变后的针孔图"建模：1280×720，水平视场 100°（NERO_V330_USB_HFOV_DEG），装在夹爪另一侧
     （link7 坐标 0.06, 0, −0.05）。真机需要先用棋盘格做 OpenCV fisheye 标定再去畸变。
   - 单目算法（深度靠"射线 ∩ 已知平面"）：桌面高度来自机器人安装标定；蛋躺在桌上 → 中心在桌面 + 蛋短半轴；
     立在已识别的蛋格里 → 格内高度。蛋 = 棕色壳颜色分割，粘连的按面积拆分（k-means）；蛋托 = 8 个深色蛋格窝 →
     投到窝顶平面 → 4×2 格子模型拟合（被蛋盖住的格用格里的蛋补）。机器人本体用模型投影做遮挡掩膜。
   - 扫视、跟踪、放置前补拍、去真值检查等其余流程与 D435i 腕部模式相同。
   - Agent 页「Pose source」改为下拉菜单：固定支架 D435i（默认）/ 腕部 D435i / 腕部 USB 130° RGB / JSON 真值。
     启动时也可 NERO_V311_POSE_SOURCE=camera|wrist|wrist_rgb|truth。
3. MuJoCo 里夹爪腕部加了两种相机的示意模型（无质量、不碰撞、都在光心后面，不挡自己的视野）：D435i 机身 + 镜头，
   USB 相机 PCB + 镜头座 + M12 镜头。选了哪个腕部相机就只显示哪个；非腕部模式两个都显示。
   NERO_V330_DRAW_WRIST_CAMERAS=0 关闭。MoveIt 碰撞模型里没有加相机（真机安装后要补）。
- 日志：`[V3.30 WRIST PERCEPTION] #n usb130 ...`、`[V3.26 POSE SOURCE] ... USB 130deg RGB-only`。


**V3.29**（仿真，腕部相机模式）：不用任何预设场景坐标，检查也不用仿真真值。全局支架相机 / 真实值两种模式不变。
1. 通用扫视 + 容器识别：每个任务开始时场景布局清空；第一次拍照前机械臂绕底座扫一圈桌面
   （半径 0.42 m，方位 180/145/215/110/250°，只是方向，不含任何物体坐标），找到蛋托和全部蛋就停。
   - 蛋托：深度图里去掉桌面、机械臂和蛋，剩下的 12 mm 底板 + 围边拟合出中心和朝向（最小外接矩形），
     每个蛋格中心再由它周围 6 根立柱细化。事先只知道蛋托的“型号”（尺寸、4×2 格距、立柱圈），不知道它在哪。
     只看到一部分时，再对准已看到的部分补拍一张。
   - 篮子：这个 MuJoCo 场景里没有实体篮子（V2-50 起就删了），“篮子区域”= 识别到的、不在蛋托里的蛋的范围。
   - 之后所有步骤（蛋格坐标、蛋托中心、篮子中心、转向中转点、靠近篮子/蛋托的姿态判断、MoveIt 里的蛋托碰撞体）
     都换成识别结果。蛋托没识别出来时，放置类工具直接拒绝。
2. 去掉真值依赖：腕部相机模式下技能里所有“物体在哪”的查询（抓取前检查、夹取、抬起、插入、松开）改为：
   夹着 = TCP × 抓取时（由相机得到的）相对位姿，并以夹爪力 / hold 状态判断是否还夹着；松开在某蛋格上方 =
   落在该格；其余 = 相机识别结果。仿真真值只写进 `[V3.29 TRUTH LOG]`。
   放蛋前（plan_placement，蛋还横着、夹爪朝下）先到蛋托上方拍一张确认蛋格位置，再回到原姿态继续。
- 日志：`[V3.29 SCAN START/VIEW/DONE]`、`[V3.29 SCENE PERCEPTION]`（含与真值的误差，仅日志）、
  `[V3.29 PRE-PLACE LOOK]`、`[V3.29 NO-TRUTH]`、`[V3.29 TRUTH LOG]`。
- 环境变量：NERO_V329_NO_PRIOR（1）、NERO_V329_SCAN_R_M（0.42）、NERO_V329_SCAN_AZ_DEG、NERO_V329_SCAN_DIST_M
  （0.45,0.40,0.36）、NERO_V329_PREPLACE_LOOK（1）、NERO_V329_PILE_MARGIN_M（0.04）、NERO_V329_TRUTH_LOG_MM（10）。


**V3.28**（仿真，腕部相机）：修复 V3.27 第二个蛋失败的两个原因，其余不动。
1. 按视野跟踪：每次拍照只让「上次位置落在本次画面里」的蛋参与配对，画面外的蛋保持原名字和位置；
   后端每次拍照附带提示：夹爪里的蛋（held，不参与配对）、本轮已放进蛋格的蛋（placed，先把它的轨迹移到该格）。
   V3.27 的 egg_1 ↔ egg_4 身份互换不会再发生。日志 `[V3.28 WRIST TRACK]`。
2. 低位不再卡死：去观察位前回 q0 失败 → 直接从当前位置规划观察位（`[V3.28 WRIST LOOK DIRECT]`）；腕部相机模式下
   plan_grasp 重拍不再先回 q0；回 q0 路径的高度下限 = min(180 mm, 起点高度 − 20 mm)（`[V3.28 Q0 FLOOR FROM START]`）。
- 全局支架相机算法文件不变；V3.11 跟踪器本身不变（腕部相机外面包了一层）。
- 环境变量：NERO_V328_VIEW_MARGIN_PX（20）、NERO_V328_PLACED_DZ_M（0.011）、NERO_V328_Q0_FLOOR_DROP_M（0.020）。


**V3.27**（仿真，腕部相机修复）：V3.26 的观察位只有一个固定姿态（TCP 在点上方 0.29 m、yaw 0），cuRobo 对它返回 None，
机械臂一直停在 q0、相机朝天花板，所以一个蛋都看不到。现在每个视角是一组等效观察位：相机到观察点距离 0.36 / 0.40 / 0.33 m ×
倾角 0° / ±15° / ±30° × 绕竖直方向 8 个朝向（共 120 个），先用 BioIK（带碰撞检测）筛出可达的，按关节移动量排序，
再用 MoveIt 关节目标规划（碰撞检查，和 V2-39 回 q0 一样），失败才退回 cuRobo 位姿规划。
- 日志：`[V3.27 WRIST LOOK SCREEN]`（试了多少个、几个有解）、`[V3.27 WRIST LOOK OK]`（用了哪个视角、相机位置、距离）。
  全部不可达 → `WRIST_LOOK_UNREACHABLE`，照旧在当前位置拍，并在结果里标 look_failed。
- 从一个观察位到下一个观察位不再先回 q0（V3.26 的其他规则不变）。全局支架相机和真实值两种方式完全没动。
- 环境变量：NERO_V327_LOOK_DIST_M、NERO_V327_LOOK_ROLLS_DEG、NERO_V327_LOOK_MAX_IK（40）、NERO_V327_LOOK_KEEP（4）、
  NERO_V327_LOOK_IK_TIMEOUT_S（0.08）、NERO_V327_LOOK_MIN_TCP_Z_M（0.20）。


**V3.26**（仿真）：新增腕部相机方式，和原来的全局支架相机并行；原相机算法文件 NERO_V308_Camera_Perception.py 一字未改。
- 仿真界面 Agent 页「Object pose source for skills」三选一：Camera（全局支架相机，默认）/ **Wrist camera（腕部相机，V3.26）** /
  JSON ground truth（全局真实值）。也可用环境变量 NERO_V311_POSE_SOURCE=camera|wrist|truth 指定启动时的默认值。
- 腕部相机：D435i 装在 link7（夹爪）上，偏离夹爪轴线 70 mm，向轴线倾斜 11°（光轴在相机前约 0.35 m 处与夹爪轴线相交），
  分辨率 / 视场 / 深度噪声与全局相机相同；手指在画面外。选中后才启动它自己的渲染 + 识别进程（日志 logs/perception_wrist）。
- 拍照前机械臂先到「观察位」：夹爪朝下，TCP 在观察点上方 0.29 m（相机离蛋约 0.4 m，大于 D435 最近距离 0.28 m），停稳后拍：
  observe_scene = 篮子上方 + 蛋托上方两个视角；plan_grasp = 目标蛋上方（拍完回 q0 再规划抓取，保持原来的起始姿态逻辑）；
  verify_object_in_slot = 蛋托上方。多个视角在世界坐标下合并（每个蛋取最近一次看到它的视角）。夹着蛋时不移动，原地拍。
- 识别算法沿用 V3.10/V3.11（perceive_eggs + 跟踪），外参每次拍照时由关节状态算出。

**V3.25**: 新增顶层页「Camera调试」（固定安装的 D435i，eye-to-hand）；原「实机 NERO」页改名「实机NERO调试」。
- 相机：连接 / 断开（pyrealsense2，进程内），彩色 1280×720 + 深度 848×480 对齐到彩色，High Accuracy 预设 + 空域/时域滤波；
  型号、序列号、固件、USB、内参、帧率；「保存相机信息」→ config/camera_info.json。
- 画面：彩色 / 深度（绝对色标 0.3–1.5 m）/ 叠加；点击 = 距离 + 相机坐标（标定后 + base_link 坐标）；拖框 = 深度检查区域。
  叠加显示：机器人连杆、安全区（画在桌面上，应和桌面胶带重合）、ArUco、标定预测的标记位置、鸡蛋。
- 深度检查：框选空桌面 → 平面拟合（有效深度、残差、倾角；标定后给出桌面在 base_link 的高度并和安全区「桌面 z」比较）。
- 手眼标定：法兰贴 ArUco（「生成标记图片」→ config/aruco_*.png，打印后量实际边长）；
  手动「记录当前样本」（关节控制 / 示教拖动）或「自动采集」（当前姿态附近 N 个姿态，MoveIt + 安全区，≤ 10%，结束回起点）；
  「计算标定」：相机位姿 + 标记在 link7 的位置，自动剔除坏样本，留一误差作独立检验（平均 ≤ 3 mm、最大 ≤ 5 mm 合格）；
  「保存标定」→ config/camera_calib.json；看到标记时实时显示当前标定误差；「快速复核」走到存档姿态比对。
- 视觉定位（实验）：用 V3.10/V3.11 的鸡蛋识别处理真实图像，只显示，不运动。
- 依赖：pyrealsense2、numpy、OpenCV（带 aruco）。检查：`python3 -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"`。

**V3.24**:
- 「工作区安全」俯视图往底座后方扩到 70 cm（原来 25 cm），左右 ±85 cm；侧视图 XZ 后方到 45 cm。安全区角点可以拖到 NERO 后面。
- 新按钮「待断电位置⚠」（在「回 Home⚠」旁边）：经 MoveIt（带安全区 + 路径检查、预览、确认）运动到去使能 / 断电前的最佳位置 D。
  D 默认 = 2026-10-24 实机记录的 J1…J7 = 0.1, −100.6, 1.2, 126.0, −0.1, −0.2, 97.3°；其中 J2 / J4 / J7 超出 URDF 限位（手推到的），
  执行时收进可设范围（限位 − 3°）：−96.7 / 119.6 / 87.0°。关节控制页「当前设为待断电位置」可以重新记录（config/park_pose.json）。
  D 的肘部在底座后方约 31 cm：规则 A 下安全区后边界要到 x ≥ +0.38 m，否则被拒绝，日志会给出需要的位置。

**V3.23**: 安全规则 A（默认，"围墙"）：安全区多边形沿 z 方向拉成围墙（桌面 → z 上限），NERO 本体 + 夹爪在任何高度都不能越过。
- 检查方式：沿 link2 → link7 → 工具点每 5 cm 取一个检查球（连杆半径 / 夹爪半径，默认 50 mm），球面离每面墙 ≥ 离墙余量（默认 20 mm）。
- 多边形必须包住底座：没包住时标题栏标红、所有运动拒绝；「选择/编辑」→「并入底座」（凸包，新角点 B1…）；示教生成安全区时自动并入。
- 围墙同步到 MoveIt：每条边一面薄墙（带朝向），「保存」「另存为」「打开」以及 MoveIt 就绪时自动下发，MoveIt 直接绕墙规划。
- 示教 / 巡检：腕部、夹爪、工具尖可以贴到墙线（指针对准角点标记），上臂和前臂仍保持离墙距离；巡检期间 MoveIt 围墙暂时撤下、结束后恢复；
  手动拖动越墙时报警。
- 规则 B（"分层"，即 V3.16–V3.22 的规则）保留为可选：多边形 ∪ 底座圆，只在 XY 限高以下限制 XY；适合周围没有墙的开放桌面。

**V3.22**: 实机日志显示 NERO V111 进入主导臂（手动拖动）后 ctrl_mode 仍报 0x01，V3.21 因此判定"进入失败"、重开控制门、
没有切到示教状态，导致记录角点读不到角度。现在：
- 进入示教的判据增加"主导臂角度开始更新"（pyAgxArm）和"/feedback/leader_joint_states 在发且 /feedback/joint_states 停发"（ROS）；
  即使没有示教标志，只要检测到手动拖动，角度也取 leader 话题（状态显示「检测到手动拖动」）。
- 退出示教不再猛冲回零：控制门保持关闭 → ROS 侧 arm_controller 同步到当前位姿 → pyAgxArm 速度 10% + move_j(当前位姿) 后再使能
  → 开控制门 → 工具点先竖直抬起到桌面上方（xy_free_height + 30 mm）→ MoveIt 以 10% 速度回零位（带安全区检查）。

**V3.21**: 示教模式下「记录角点」修复 — 主导臂（手动拖动）模式下 V111 固件停发 /feedback/joint_states，角度改走 CAN 0x501–0x507。
进入示教模式后控制台自动启动 pyAgxArm `get_leader_joint_angles()` 读数进程（20 Hz）；角度来源优先级：ROS leader 话题 >
pyAgxArm 主导臂 > pyAgxArm 关节 > /feedback/joint_states。示教面板显示当前角度来源；记录失败时日志列出各来源的数据年龄。

**V3.20**: 「进入示教模式」 fixed — closes the driver's control gate, uses pyAgxArm `set_leader_mode()` (hand guiding)
and checks that the arm really reports teach mode (`ctrl_mode` 0x06 / 0x02); while teaching, joint angles come from
`/feedback/leader_joint_states`.

**V3.19**

**V3.19**: 示教角点 has 「进入示教模式（可用手拖动）」 / 「退出示教模式」 buttons; safety zones can be saved under any name
and folder (「另存为…」 / 「打开…」, the last file opens at the next start); 「回 Home」 now goes through MoveIt.

**V3.18**

**V3.18**: joint-panel targets follow the arm (e.g. homing after 一键启动) until you edit one; the safety editor's
side view keeps text out of the drawing (line names + values in a right-hand column, colour legend below).

**V3.17**

**V3.17**: NERO works on the −x side of base_link (the sim's egg basket is at x = −0.425 m). The safety editor now
draws forward (−x) up and +y right, as seen from behind the robot; the default zone and table are in front of the
robot; tick labels sit in the margins; the side view says what every height line means.

**V3.16.2**

**V3.16.2**: fixes the startup segfault of V3.16.1 on Ubuntu 22.04 (Tk 8.6).

**V3.16.1**

**V3.16.1**: the 实机 tab no longer jumps — every live text (step messages, status rows, banners, warnings) sits in a
fixed-size area, and the parameter box no longer collapses by itself.

**V3.16**

**V3.16: graphical safety-zone editor** — 实机 → 工作区安全: top view + side view, drag the zone, teach it from the
table marks, inspect it. See "Safety-zone editor (V3.16)" below.

**V3.15: sim/real consistency + workspace safety** — 实机 tab sub-tabs 关节控制 / 一致性测试 / 工作区安全.
See "Consistency + safety (V3.15)" below.

**V3.14: 实机 tab joint panel** — J1–J7 + gripper, vertical sliders with limits, angle entry, MoveIt execution,
small-step test, jog mode, pose library, TCP readout. See "Joint panel (V3.14)" below.

**V3.13: 实机 tab — six-step real-robot pipeline** (environment -> CAN port -> CAN push / enable -> CAN traffic ->
ROS2 driver -> MoveIt/RViz), lamps, one-click start, stop all, e-stop, command preview. See "Real robot (V3.13)" below.

**Behavior: V3.03 + wider egg aperture / grasp-Z compensation / slower final approach,
descent-aware fast grasp preflight with trajectory reuse, conditional slot precheck, compact
model-facing tool output, numeric slot fill order** (see `CHANGELOG.md`).
Legacy q0: `NERO_V302_ALLOW_Q0_FALLBACK=1 ./start_nero.sh`.

V3.07: start_nero.sh stops stale ROS/NERO processes from a previous run before starting
(`NERO_V307_PRELAUNCH_CLEANUP=0` refuses to start instead) and refuses SYSTEM READY if more
than one `/arm_controller` exists. plan_grasp returns the empty arm to SAFE_NEUTRAL q0 and
re-plans once when the current posture is the problem.

V3.08: a simulated Intel RealSense D435i (fixed, eye-to-hand) renders aligned RGB + depth in its own
process; an RGB-D egg perception module estimates every egg's centre and long axis from the images only and
compares them with MuJoCo ground truth. **Skills still use ground truth** — this release only records.
Output: `logs/perception/<run>/` (see "Camera output" below).

V3.10: perception robustness (still record only). Depth is fused over 5 frames, overlapping egg fits are chosen by
how many points they explain (not lowest RMS), every fit is checked against free space in the depth image, lying eggs
get a horizontal prior, and every detection carries `confidence` + `trusted`. Untrusted = a perception-driven skill
would look again instead of using it. `set_gripper_opening` waits for the arm to settle first.

**V3.11: skills can run on the camera.** Agent tab -> "Object pose source for skills": *Camera (D435i RGB-D
perception)* (default) or *JSON ground truth (MuJoCo)*. In camera mode `observe_scene`, `plan_grasp` (target and
neighbours) and `verify_object_in_slot` wait for a fresh capture and use the tracked perceived poses; an untrusted
target makes plan_grasp clear the view (q0) and look once more, then fail with `PERCEPTION_TARGET_UNTRUSTED`.
In-hand / contact / motion-monitoring checks still read the simulator (real robot: TCP x grasp transform + tactile).
Truth is only logged for comparison (`[V3.11 PERCEPTION] ... err=..mm`, `[V3.11 VERIFY] ... truth(log only)`).
The D435i body and a floor tripod are drawn at the configured camera pose. The task-status lamp now sits right of
"Copy Full Log".

**V3.12: console first, nothing starts automatically.** `./start_nero.sh` checks the release and opens the
**NERO console** (`NERO_V312_Console.py`) with two tabs:
- **仿真 (MuJoCo)**: "▶ 启动仿真" runs `start_nero_sim.sh` — the V3.11 launch chain, unchanged except the signal
  trap — with 8 step lamps (integrity, stale processes, GPU/cuRobo, modules, backend+MuJoCo, BioIK MoveIt+RViz,
  TRAC-IK optional, SYSTEM READY). The sim GUI (Agent tab etc.) opens as its own window exactly as before.
  "■ 停止仿真" or closing the sim GUI stops everything (cleanup + kinematics.yaml restore).
- **实机 NERO**: placeholder (six-step real-robot bring-up arrives in V3.13). Locked while the sim runs.
- `./start_nero.sh --sim` = old behaviour (full sim chain directly, no console); `./start_nero.sh --real` opens the
  console on the 实机 tab. The console needs `python3-tk` for `/usr/bin/python3`.
- Backend / agent / planner / GPU / camera files are byte-identical to V3.11 (checked by `tests/test_v3_12.py`).

**V3.12.1: held-egg recovery** (after a V3.12 camera run stopped with an egg stuck in the gripper):
- `lift` lowers the height by itself when the requested corridor is blocked (130 -> 108 -> 90 -> 70 -> 50 mm);
  all blocked -> `V3121_LIFT_CORRIDOR_BLOCKED` (structured, no BACKEND_EXCEPTION).
- `orient_for_slot` first raises a low held egg to TCP z >= 160 mm (`NERO_V3121_ORIENT_MIN_TCP_Z_M`).
- New tool **`return_to_pick_location`**: puts the held egg back where it was picked (straight down, open to the
  pre-grasp aperture, retreat) so the task continues with the other eggs; used when lift/orientation is impossible.
- Grasp preflight checks the straight descent from the *planned* pregrasp joint state (`NERO_V3121_DESCENT_PRECHECK=0`
  disables); a blocked descent also retreats back to the pregrasp height at once.
- q0 moves for a camera re-look use a shorter straight retreat when the full one is blocked; a failed clear-view move is
  reported as `RECAPTURE_CLEAR_VIEW_FAILED`, not as a perception failure.
- Camera mode: a held egg is reported as `location=in_gripper` with its pose from TCP x grasp transform.

Offline tests (no ROS/GPU): `bash tests/check_release.sh`

## Run

```bash
cd ~/Downloads
unzip NERO_V3_34.zip
cd NERO_V3_34
./start_nero.sh            # console; or ./start_nero.sh --sim
```

You can keep the folder anywhere (e.g. move it to `~/mujoco_ws/nero/`).

## What runs from this folder (and only from here)

| File | Role |
|---|---|
| `NERO_V2_7_1_Unified_Backend.py` | MuJoCo sim process + Tk GUI + all skills (V2-52.3 state) |
| `NERO_V2_7_1_Agent.py` | LLM task planner (OpenAI Responses API, 19 tools) |
| `NERO_V71_Hybrid_GPU_Planner_Core.py` | MoveIt/BioIK/cuRobo planning + execution (V2-43 + V2-51) |
| `NERO_GPU_Planner_Client.py` | client to the cuRobo server |
| `NERO_GPU_Planner_Server.py` | cuRobo GPU server (V2-24), started by `start_nero.sh` |
| `NERO_V308_Camera_Perception.py` | simulated D435i camera process + RGB-D egg perception (V3.10 robust fit + confidence) + error logging |
| `start_nero.sh` | entry point: release check, then the console (`--sim`: sim chain directly) |
| `NERO_V312_Console.py` | V3.12 console window: 仿真 tab (starts/stops `start_nero_sim.sh`, step lamps), 实机 tab (placeholder) |
| `start_nero_sim.sh` | the simulation launch chain (V3.11 `start_nero.sh`, trap fix only) |

Nothing is copied to `~/mujoco_ws/nero` and nothing is patched at startup.
`start_nero.sh` refuses to start if any file differs from `SHA256SUMS.txt`
(override with `NERO_ALLOW_MODIFIED=1`), and prints the path + hash of every
NERO module Python actually resolves. If any resolves outside this folder,
startup is refused.

V69 / V70 planner cores are **not** included: the runtime never imports them
(V71 is self-contained).

## External dependencies (referenced by path, not copied)

| Item | Default | Override |
|---|---|---|
| MuJoCo model + AgileX meshes | `~/mujoco_ws/nero/nero_motor.xml` | `NERO_ASSET_WS` |
| ROS 2 | `/opt/ros/humble` | `ROS_DISTRO` |
| AgileX MoveIt | `~/agx_arm_ws` | `AGX_ARM_WS` |
| BioIK / TRAC-IK | `~/bio_ik_ws`, `~/trac_ik_ws` | `BIO_IK_WS`, `TRAC_IK_WS` |
| MuJoCo Python venv | `~/mujoco_ws/venv` | `VENV` |
| cuRobo venv + config | `~/.nero_gpu_planner` | `NERO_GPU_ROOT` |

`config/` holds a reference copy of the cuRobo/URDF config for review; the
runtime uses `~/.nero_gpu_planner/config` exactly as before.

## Camera output (V3.08, extended in V3.10)

One folder per backend start: `logs/perception/<YYYYmmdd_HHMMSS>/`

| File | Content |
|---|---|
| `camera_info.json` | intrinsics (fx, fy, ppx, ppy, 1280x720), `T_world_camera_optical`, range, noise setting |
| `errors.csv` | one row per egg per capture: detected, 3D / XY / Z error (mm), long-axis error (deg), location match, **confidence, trusted, coverage** (V3.10) |
| `summary.json` | per-capture summary + `overall` (all detections) + **`overall_trusted`** (V3.10: what a skill could use; max errors, trusted FP) |
| `capture_NNNN_<tool>_<egg>/` | `tracked_scene.json` (V3.11: what the skills received), `rgb.png`, `depth_mm.png` (16-bit mm, RealSense z16 style), `overlay.png` (green = trusted, yellow = untrusted, red = truth), `perceived_scene.json`, `truth_scene.json`, `errors.json`, `qpos.npy` |

Captures happen at the start of `observe_scene`, `plan_grasp`, `verify_object_in_slot`.
Console/backend.log lines: `[V3.11 CAMERA] READY ...`, `[V3.11 CAMERA] SELF-CHECK PASS`, `[V3.11 PERCEPTION] #n ... trusted N untrusted M trusted-FP K ...`.

| Env | Default | Meaning |
|---|---|---|
| `NERO_V308_CAMERA` | 1 | 0 = no camera, no perception |
| `NERO_V308_CAM_POS` / `NERO_V308_CAM_TARGET` | `-0.95,0,0.72` / `-0.42,0,0.03` | camera position and look-at point (world, m); set to your real D435i mount |
| `NERO_V308_WIDTH` / `_HEIGHT` / `_FOVY_DEG` | 1280 / 720 / 42 | D435i color stream (depth aligned to color) |
| `NERO_V308_DEPTH_NOISE_K` | 0 | depth noise sigma = k·z² (try 0.0025); 0 = clean, mm-quantised |
| `NERO_V308_TRIGGER_TOOLS` | `observe_scene,plan_grasp,verify_object_in_slot` | which tools trigger a capture |
| `NERO_V308_PERIOD_S` | 0 | extra periodic capture every N s (0 = off) |
| `NERO_V308_SAVE_IMAGES` | 1 | 0 = keep only json/csv (~3 MB less per capture) |
| `NERO_V308_GL` | `egl,osmesa,glfw` | offscreen OpenGL backends tried in order |
| `NERO_V308_EGG_COLOR_FILTER` | 1 | 0 = height-only segmentation (white eggs) |
| `NERO_V310_DEPTH_FRAMES` | 5 | depth frames fused per capture (temporal filter; only used when noise is on) |
| `NERO_V310_EXPECTED_NOISE_K` | 0.0025 | noise level assumed for the inlier tolerance (quality terms use the measured fit residual) |
| `NERO_V310_RESTING_PRIOR` / `_RESTING_TILT_SIGMA_DEG` | 1 / 6 | eggs resting on the table/basket floor are fitted near-horizontal |
| `NERO_V310_WARM_START` | 1 | previous capture's trusted eggs are extra fit starts |
| `NERO_V310_TRUST_MIN_CONF` / `_MIN_COVERAGE` / `_MAX_SEE_THROUGH` | 0.70 / 0.55 / 0.06 | trusted rule |
| `NERO_V310_SETTLE` / `_SETTLE_TIMEOUT_S` / `_SETTLE_TOL_MM` | 1 / 1.5 / 0.5 | TCP settle wait before set_gripper_opening and after q0 |
| `NERO_V311_POSE_SOURCE` | camera | start-up value of the Agent-tab switch (`camera` or `truth`) |
| `NERO_V311_CAPTURE_TIMEOUT_S` | 20 | how long a skill waits for its capture |
| `NERO_V311_DRAW_CAMERA` / `NERO_V311_FLOOR_Z_M` | 1 / -0.28 | draw the D435i + tripod; floor height |

## Real robot (V3.13)

`./start_nero.sh` -> tab **实机**. Every step turns green before the next one can run; 一键启动 runs them in order and
stops at the first red lamp. The sim and real pipelines lock each other.

1. **环境**: ROS2 Humble, agx_arm_ws, agx_arm_ctrl, pyAgxArm, can-utils, USB-CAN, pkexec; no leftover ROS/NERO
   processes (清理残留进程 stops them); simulation not running.
2. **CAN 口**: can0 present, UP, 1 Mbps, ERROR-ACTIVE. Otherwise `pkexec` asks for the password (every time) and runs
   `ip link set can0 down; ip link set can0 type can bitrate 1000000 [restart-ms 100]; ip link set can0 up`
   (restart-ms only if the adapter supports it; gs_usb does not — V3.13.1).
3. **CAN push / 使能** (confirmation: motors are enabled): pyAgxArm NERO fw V111, `set_normal_mode()` until `enable()`
   succeeds, then reads joint angles / arm status (same as your nero_start.py). Runs as a helper process.
4. **CAN 流量**: raw SocketCAN reader; fps and groups 251-257 / 261-267 / 2A1-2A9; gripper IDs 517/527 (old) vs
   51A/52A. Old gripper IDs -> effector locked to `none`.
5. **ROS2 驱动**: first a push-release test (push helper stopped for 3 s, fps measured): `released` = arm keeps pushing;
   otherwise `keep_until_driver` (push restarted, stopped once /feedback/joint_states flows). Then
   `ros2 launch agx_arm_ctrl start_single_agx_arm_moveit.launch.py arm_type:=nero fw_version:=v111 ...` with
   ROS_DOMAIN_ID=42 (`NERO_V313_ROS_DOMAIN_ID`); green when joint_states flows and arm_status arrives.
6. **MoveIt/RViz**: move_group node + move_action + RViz alive.

Operations: control gate on/off, enable, disable (confirm), move home (confirm), exit teach mode, **■ 急停**
(`/emergency_stop`; needs the driver — otherwise use the Web e-stop or cut power). speed_percent is capped at 30 %
unless unlocked. 复制终端环境 copies `source ... ; export ROS_DOMAIN_ID=42` for your own terminals.
Log: real-tab log; push / driver / monitor output also goes to `logs/console.log`.

## Joint panel (V3.14)

Below the six steps of the 实机 tab (the tab scrolls; collapse the parameter box with 「参数 ▴」 for more room,
「参数 ▾」 opens it again). Enabled only when step 6 is green, /feedback/joint_states is fresh and no motion runs.

- **8 columns** J1–J7 + 夹爪: upper / lower limit (red) and the settable range ("可设到", 3° soft margin), vertical
  slider = target, green triangle = actual, entry box (Enter; out of range -> red + clamped), actual, deviation,
  −5/−1/+1/+5. Yellow column = within 5° of a limit.
- **Limits** are read from `/robot_description` (URDF), the planning group and named states from
  `/robot_description_semantic` (SRDF, topic or /move_group parameter). Not readable -> the sim table
  (same NERO URDF) and the label says so.
- **▶ 执行（MoveIt）**: MoveGroup action joint goal (collision-checked plan + execute), velocity/acceleration scaling =
  speed_percent. Confirmation on the first motion of the session and whenever any Δ > 30° or total Δ > 90°.
  **■ 停止运动** cancels the goal; ■ 急停 stays the hard stop. The log shows plan time, execution time and the final
  error per motion (> 1° -> WARN).
- **Targets follow the arm** until you edit one (slider, entry, ±, pose); the line under the panel says
  「目标：跟随当前角度」 or 「目标：已修改，未执行」. A finished motion or **⟲ 同步为当前** resumes following.
- **小步测试** Jn ±5° 往返: one joint out by 5° (or −5° near the upper limit) and back, with one confirmation.
- **点动模式**: ±1°/±5° buttons move immediately (one confirmation when switched on; max 5° per click).
- **姿态库**: 零位, SRDF named states (e.g. home), user poses saved from the actual angles in
  `config/real_poses.json`. Loading only sets the targets; 执行 still previews / confirms.
- **夹爪**: shown but locked while the CAN bus shows the old gripper ID (517/527), effector=none, no gripper planning
  group or no /feedback/gripper_status. Unlocks by itself once those are fixed.
- **TCP**: TF base -> tip (SRDF chain tip, else last arm link), read-only.

If MoveIt answers `INVALID_GROUP_NAME`, send `ros2 param get /move_group robot_description_semantic`.

## Consistency + safety (V3.15)

The 实机 tab now has three sub-tabs below the six steps: 关节控制 (V3.14 panel, plus 「只规划（不执行）」),
一致性测试 and 工作区安全. All motions go through MoveIt; every motion is a row in `logs/real_motion_log.csv`.

**一致性测试**
1. *模型对比 + 实时一致性*: the driver URDF (/robot_description) is compared joint by joint with the sim URDF
   (`config/nero.urdf`: origin, axis, limits). Live: the measured joint angles through both URDFs and the real TF
   (base_link → link7) side by side, differences in mm.
2. *关节方向 / 零位向导*: each joint +Δ (default 10°): predicted link7 displacement (sim URDF) vs measured (TF),
   automatic verdict (direction cosine ≥ 0.95, magnitude ±10 %) + your confirmation that the real arm turned like the
   RViz model, then back. The zero pose: load 「零位」 in the joint panel and compare with the AgileX zero pose by eye.
   → `logs/v315_direction_*.json`
3. *重复定位精度*: A = current pose ↔ B = A + offsets, N cycles at one or more speeds; joint error at A and TF link7
   spread (mm). → `logs/v315_repeat_*.json/.csv`
4. *停止测试*: one joint slowly, 「停止运动」 (cancel) after t s: stop distance and stop time. → `logs/v315_stop_*.json`.
   Test the ■ 急停 button by hand (and note how to recover: enable / control gate).

**工作区安全** (`config/real_workspace.json`; defaults assume the base sits on the table — red until you measure
and save)
- *碰撞体*: boxes in base_link (center + size, m) → MoveIt planning scene (applied automatically when step 6 turns
  green, or 「应用到 MoveIt」). Only `v315_*` objects are touched.
- *软件围栏*: a box + z_min for check points (link4–link7 origins + a tool point = the sim TCP 155 mm in front of
  link7). The target is checked before anything is sent; with 「检查规划路径」 the monitor plans first, checks every
  waypoint, and only then sends the trajectory to `/execute_trajectory`. Unknown link names -> refused, never passed.
- *安全测试* (no motion): fence check of the current joint-panel targets; automatic test that generates a target
  below the table and checks (a) the fence refuses it, (b) MoveIt (plan only) refuses it. MoveIt cannot see a tool
  that is not in the driver URDF — the fence is its only protection.

## Safety-zone editor (V3.16)

Replaces the V3.15 number table on 实机 → 工作区安全. `config/real_workspace.json` (V3.15 files are migrated: x/y
range → rectangle).

**The zone**: an XY polygon on the table (corners M1..Mn) ∪ the base disc (r = 200 mm). It applies to check points
(link4–link7 + tool) that are lower than *XY 限制高度* (150 mm above the table); higher points are only limited by
z 上限. Every point must stay above z_min. A point inside but closer than *内缩余量* (40 mm, dashed line) to an edge
is a yellow warning. The tool point is the sim TCP (155 mm in front of link7) or a pointer (length + link7 axis).

**Views**: top view (forward = −x up, +y right, as seen from behind the robot; 10 cm grid, reach ring, base disc, table, boxes, marks) and side view (XZ: forward right / YZ: +y right;
lines for table, z_min, z 上限 and XY 限制高度). Both show the live arm (blue) and the joint-panel target (purple
dashed); red = violating point. While the target is outside, 关节控制 → 执行 is locked.

**选择 / 编辑**: drag corners / edges / boxes (corner handle resizes), double-click an edge to add a corner,
right-click a corner to delete, drag the lines in the side view, or type values. Undo / redo. Edits take effect
immediately when valid (an invalid zone, e.g. self-intersecting, keeps the last valid one active); 保存 writes the
file, 应用到 MoveIt updates the collision boxes (also automatic at step 6 green).

**示教角点**: pointer on the flange, tool = pointer. Speed capped at 5 %; the tool may go 5 mm below the table and
the zone limits only the tool tip (z limits still apply to all points). Move the tip onto each mark (jog, or NERO
teach mode by hand) — lift between marks, sliding along the table is blocked — and press 记录角点. 生成安全区:
polygon from the marks (re-ordered if it would self-intersect), table height = mean z, flatness = max − min (warning
> 10 mm), table box and z_min follow. Optional cross-check with a measured reference point R.

**巡检**: the arm goes (≤ 5 %) via the zone centre to a hover pose above each corner (small IK, keeps the taught
orientation if possible) and asks whether the pointer is over the mark. → `logs/v315_inspect_*.json`

**自动越界测试** (no motion) as in V3.15.

## Teach mode, zone files, Home (V3.19)

- **示教模式**: 工作区安全 → 示教角点 → 「▶ 进入示教模式（可用手带动）」 closes the driver's control gate, then
  pyAgxArm `set_leader_mode()` (linkage-teaching input; older pyAgxArm: `set_master_mode()`) and waits until
  `ctrl_mode` is 0x06 / 0x02 — otherwise it reports the value and reopens the gate. Hold the arm first. 「■ 退出示教模式」
  calls the driver's `exit_teach_mode` service (if offered), then `set_normal_mode()` + `enable()`. The line next
  to the buttons shows the mode; joint-panel motions and Home are refused while NERO is in teach mode. The
  「退出示教模式」 button in the status area does the same.
- **安全区文件**: 「另存为…」 saves the zone as a .json under any name / folder (default `~/nero_zones/`), 「打开…」 loads
  one and applies it to MoveIt, 「保存」 writes the current file. The last opened / saved file is remembered in
  `~/.config/nero_console/state.json` and opened automatically at the next start (header shows the file).
- **回 Home**: MoveIt joint goal to all zeros with preview, confirmation and the safety-zone checks; the driver's
  `move_home` service is only used when MoveIt is not ready. Service calls no longer block the ROS monitor (they
  are polled; Home service timeout 60 s), so joint feedback keeps flowing during a call.

## Logs

All logs go to `./logs/`: `console.log` (V3.12 console), `launcher.log`, `startup_status.log`, `backend.log`,
`gpu_server.log`, `bio_moveit.log`, `trac_ref_moveit.log`, and `diag/` (motion diagnostics).

## Rollback

The old runtime folder and `NERO_V2_52_3_LLM_BACKGROUND_ANCHOR_FIX/start_nero_v2_52_3.sh`
are untouched and still work.

## Recommended: version control on your machine

```bash
cd NERO_V3_34
git init && git add -A && git commit -m "NERO V3.00 baseline (= V2-52.3 behavior)"
```
