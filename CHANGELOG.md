## V4.6.10 (2026-11-09) — 按松开点验证鸡蛋；Try harder 验证；未确认也继续；主相机默认 848×480 + Default
- `NERO_V365_Real_Cup_Task.py`（`V4.6.10 VERIFY BY RELEASE POSITION + TRY HARDER VERIFY + UNCONFIRMED CONTINUE`）：
  - `place_egg_into_slot` 记录 `last_place`（松开点 = 实测关节算出的鸡蛋下端 xy；取蛋位置）。`_release_tip_error` 返回 `tip_xy_m`。
  - `verify_egg_in_slot` 重写：所有证据都在松开点处取；新画面里离该点最近的槽（不管编号）；鸡蛋计数用放蛋前的槽位。第 k 次看抬高 k×100 mm，直线失败时用关节空间（`_raise_for_look`，抬不到就抬一半）。
  - 新工具 `verify_egg_try_harder`：腕部相机俯视松开点（`_wrist_look_at`）→ 判掉出（`EGG_FELL_OUT`）/ 没放成功（`NOT_PLACED`）/ 未确认（ok + `unconfirmed` + warnings）。
  - `_one_egg` 返回 True / "unconfirmed" / False，并维护 `placed_xy`。`observe_scene` 把 `placed_xy` 附近的槽算作已占，附近的鸡蛋算作"已在蛋托里"。
  - `fill_carton_episode`：未确认的鸡蛋记入 `unconfirmed`，继续下一轮；`run_execute` 返回 `unconfirmed`。
  - `is_fill_all_task`：识别复数 eggs / slots、one by one、逐个、依次、剩下的 等说法。
  - 总结标签：鸡蛋 / 分拣任务写"固定流程（+ Try harder）"，只有叠杯子任务写 "(LLM)"。
- `NERO_V368_Real_Task_Record.py`：步骤记录加 `unconfirmed`；`task_summary` 有未确认步骤时为 YELLOW `UNCONFIRMED`。
- `NERO_V312_Console.py`（`V4.6.10 DEPTH DEFAULT 848x480 + DEFAULT`）：`v4610_depth_defaults` 在启动时把主相机深度设为 848×480 · Default（`NERO_DEPTH_KEEP_SAVED=1` 保留设置文件里的值）。`NERO_V325_Camera.py` 未改动（冻结文件）。
- 测试：新 `tests/test_v4_6_10.py`（8 项，含按 V4.6.9 失败那次重建的旋转 37° 场景）；`test_v4_6_8` 的 f2 / f5 适配新的日志标签和返回字段；版本白名单补 V4.6.10。
- **没有在真实硬件上运行过。**

## V4.6.9 (2026-11-09) — Smol VLA 页 5 个子页 + 夹爪示教 / 空格键 + 实机回放；叠杯目标固定；LLM Try harder；深度覆盖率
- `NERO_V4_Console_Tab.py`：
  - 顶栏（后端 / 策略 / `v_arb_text` 控制权 / 急停）+ `self.sub_nb` 5 个子页（`self.pages`：prep / teach / train / run / adv）；原有控件名和按钮文字不变。
  - `on_teach_gripper(open_)`：实机后端 + SO-101 已连接时，在线程里 `open_gap(45)` / `clamp(25)`，再记录数值；否则只记录并写 WARN。
  - `_on_space`：`bind_all <KeyPress-space>`；页面可见且焦点不在输入框时切换。
  - `on_replay_episode`：实机后端确认后调 `replay_episode_real`；其他后端照旧仿真回放，失败写日志。
  - 回放速度下拉、Episode 信息、录制状态行；非实机录制写 WARN。
- `NERO_V4_Dataset_Recorder.py`（`V4.6.9 REAL REPLAY`）：
  - `plan_real_replay`：抽稀 + 计时 + 按夹爪切段。
  - `replay_episode_real`：示教模式拒绝；10% 速度到起点；每段一条 `move_joint_path`（第一点用实测姿态）；段间夹爪动作；每段实测误差。
  - 仿真回放失败写 `[FAIL] [V4.6.9 REPLAY]`。
- `NERO_V362_Cup_Skills.py`：
  - `TRY_HARDER_INSTRUCTIONS`、`try_harder_suggestions`。
  - `run_cup_task(try_harder)`：失败结果去掉 `stop_and_report` 并加 `try_harder` 字段；提前结束时最多追问 `TRY_HARDER_PUSHES = 3` 次。
  - `place_cup_into`：搬运前固定目标杯位置（`_pin_target`）；腕部同心修正不再覆盖目标位置，单次 ≤ 10 mm、总共 ≤ 12 mm。
- `NERO_V365_Real_Cup_Task.py`（`V4.6.9 VERIFY BY EGG COUNT + LONG-AXIS WARNING + DEPTH COVERAGE`）：
  - 验证新增"蛋托里鸡蛋数增加、且最近的槽是目标槽"。
  - 夹紧读数 > 36 mm 时写 WARN。
  - `run_execute` 把 `try_harder` 传给 `run_cup_task`（不再改任务文本）；日志写明 Try harder 的类型；蛋托日志行加深度覆盖率。
- `NERO_V349_Real_Wrist_Camera.py`：
  - 模板直接用全部深度像素（`TEMPLATE_MIN_PTS = 150`；碗形分数的覆盖门槛降到 4% / 10%）。
  - `tmpl_info` 加 `depth_cov` / `inner_pts`；退回外框时的说明带覆盖率，也保存诊断图（右半边显示哪里有深度）。
- `NERO_V366_Real_Cup_Yolo.py`：杯子 `depth_cov`，摘要里打印"深度覆盖 xx%"。
- 测试：
  - 新 `tests/test_v4_6_9.py`（9 项）。
  - 新 `tests/test_v4_6_9_gui.py`：5 个子页在 1420×1000 下都完整可见；夹爪按钮和空格键；回放接线。
  - `test_v4_6_8` f7 改为检查 `try_harder` 参数；版本白名单补 V4.6.9。
- **没有在真实硬件上运行过。**

## V4.6.8 (2026-11-08) — 悬停前重新识别；按"槽心凸起"判定放蛋；连续放满蛋托；标签精简；Try harder；Camera 页可滚动
- `NERO_V464_Vision_Check.py`（`V4.6.8 RECOGNIZE BEFORE EVERY HOVER`）：
  - `VisionCheck.run(..., refresh=True)`：先 `observe()`，记录目标相对上次识别移动了多少（超过 5 mm 写 WARN）。
  - CSV 新增 `recog_age_s`、`target_moved_mm`；面板新增「悬停前重新识别」（默认开），恢复第三排（dx / dy / 备注），状态行 2 行。
  - 不再把识别表缩到 3 行（Camera 页现在可以滚动）。
- `NERO_V349`：
  - 新增 `slot_rise`：在裁剪区域的**全部**深度像素里量每个槽，槽心第 90 百分位高度减去槽口环的中位高度；≥ 8 mm 判为有蛋（`SLOT_EGG_RISE_MM`）。
  - 每个槽的数据里新增 `rise`。
- `NERO_V365`（`V4.6.8 FILL CARTON + VERIFY BY RISE + TRY HARDER`）：
  - `verify_egg_in_slot(looks=2)`：三条证据——槽被占 / 0.5 个孔距内有鸡蛋 / 槽心比放蛋前高 ≥ 15 mm；第 1 次看在起点，第 2 次抬高 100 mm 再看；每次的结果写 `[V4.6.8 VERIFY]`。
  - `observe_scene`：离任一槽心小于 0.55 个孔距的鸡蛋算作"已在蛋托里"（`eggs_in_carton`），不作为要抓的蛋。
  - `_jaws_clear` / `_occupied_xy`：两个夹爪指尖与相邻已立鸡蛋的水平距离 ≥ 21 + 6 mm；有相邻鸡蛋时偏航角多试每 15° 一个。
  - `fill_carton_episode`：每轮先识别，空槽按离底座由远到近，`FILL_MAX_FAILS = 3`。
  - `scripted_episode(try_harder)` → `_one_egg`（各步重试策略）、`replan_place`、`refresh_slots`。
  - `run_execute(try_harder, fill_all)`：LLM 模式下在任务说明里追加 TRY HARDER 指示，工具步数翻倍（上限 90）。
  - `is_fill_all_task`：识别"全部 / 所有 / 放满 / all eggs"等说法。
  - 监控标签：槽只显示编号、鸡蛋显示 e1…、蛋托显示「蛋托 R×C」，字体缩小；槽和蛋不再画第二个标签。
  - 「Try harder」勾选框放在「速度」左边，状态保存在设置里。
- `NERO_V312_Console.py`（`V4.6.8 CAMERA TAB SCROLL`）：Camera调试 页包进 Canvas + 滚动条 + 鼠标滚轮，做法和「实机NERO调试」页一样；鼠标在 Text / Treeview / Combobox / Scale / Listbox 上时滚动它们自己。
- 测试：
  - 新 `tests/test_v4_6_8.py`（9 项）：
    - 合成画面里的槽占用判断；
    - 验证判定的三条证据和多次查看；
    - 蛋托里的鸡蛋不当作要抓的蛋；
    - 夹爪避让旁边的鸡蛋；
    - 连续放满（假场景，每次放蛋后场景随之变化）；
    - Try harder 下的夹取重试和重新规划放蛋；
    - run_execute；标签；界面接线。
  - `tests/v466_synth_carton.py` 支持在槽里立鸡蛋（`eggs_in`）。
  - `test_v4_6_4_gui` 先滚到页面底部再检查。
  - 版本白名单补 V4.6.8（跳过 V4.6.7）。
- **回归**（和 V4.6.6 用同一套测试对比）：非 GUI 106 个文件、GUI 41 个，**0 个回归**（`v3_56_gui` 照旧时好时坏）。
- **没有在真实硬件上运行过。**

## V4.6.6 (2026-11-08) — 蛋托固定模板（2×3，46×47 mm）+ 原生崩溃调用栈 + Tk 输入法默认关
- **蛋托**：
  - `NERO_V349`（`V4.6.6 FIXED CARTON TEMPLATE`）新增 `carton_template_config`（`config/egg_carton.json` 的 `carton_grid` + `slot_pitch_mm`；环境变量 `NERO_EGG_CARTON_GRID` 可覆盖，设为 `none` 时退回旧方法）。
  - `bowl_score_map`：1 mm 高度栅格；槽口环（0.36–0.52 × 槽距）平均高度减槽心（≤ 0.24 × 槽距）平均高度。
  - `fit_carton_template`：刚性网格，中心在外框中心 ±40 mm、朝向为长边方向或其 +90°（各 ±10°）范围内搜索；轮廓先内缩 9 px。
  - `save_carton_diag`：诊断图存到 `NERO_CALIB_DIR/carton_diag`，保留最近 30 张。
  - `locate_egg_holder` 有模板配置时不再猜行列、不再做 V4.5 槽穴修正；槽号固定（列沿 3 孔方向，离底座越远编号越大）；少于 2 个碗形槽穴时模板按外框中心 / 长边放置，并写明原因。
  - `NERO_V365` 任务日志新增 `[V4.6.6 CARTON]` 一行。
- **配置**：`config/egg_carton.json` 新增 `carton_grid: [2, 3]`，`slot_pitch_mm` 设为 `[46, 47]`（用户 2026-11-08 实测）。
- **原生崩溃调用栈**：新文件 `tools/nero_crashtrace.c`。`start_nero.sh` 用 gcc 编译到 `~/.cache/nero/nero_crashtrace_<sha>.so`，通过 `LD_PRELOAD` 加载，崩溃时把 `backtrace_symbols_fd` 的结果写入 `logs/native_crash.log`；Python 的 faulthandler 先写 crash.log 再把信号转交给它。控制台 `v466_native_trace_status()` 把状态写进 crash.log，并从环境中删掉 `LD_PRELOAD`，子进程（ROS、helper）不会加载它。
- **Tk 输入法**：`v466_tk_input_methods`，默认执行 `tk useinputmethods 0`；设置 `NERO_TK_IM=1` 时保留输入法。
- **方向提示**：`NERO_V464.axis_hint` 改为 8 个方向。
- **测试**：
  - 新 `tests/test_v4_6_6.py`（9 项），以及合成场景生成器 `tests/v466_synth_carton.py`（斜视 2×3 蛋托，z-buffer 渲染，轮廓外溢）。
  - `test_v3_49`、`test_v3_84` 的合成蛋托是用来测旧的零先验推断的，测试里设 `NERO_EGG_CARTON_GRID=none`。
  - `test_v4_5`、`test_v4_6_1` 原来断言槽距为 null，改为断言实测值 [46, 47]（有意的改动）。
  - 版本白名单补 V4.6.6。
- **回归**（和 V4.6.4 用同一套测试对比）：非 GUI 105 个文件、GUI 41 个，只有原有失败（`v3_56_gui` 时好时坏），**0 个回归**；`test_v3_49_gui`（蛋托 6 槽 + 悬停）改用模板后仍然通过。
- **没有在真实硬件上运行过。**

## V4.6.5 (2026-11-08) — 视觉精度验证面板修复：夹爪状态、Camera 页日志被压扁
- `NERO_V464_Vision_Check.py`：
  - 「合爪并悬停」点击时（确认框之前）从 Agentic 页重新取夹爪（`io.grip = rct.grip if connected`）。V4.6.4 在「识别目标」时就把夹爪定死了，之后再连接的夹爪一直被当作没连。
  - 面板改为两排 + 一行状态（去掉「备注」输入框）；状态同时写入 Camera 日志；`tv_eggs` 运行时显示 3 行。
  - 「视觉定位」页需要的高度 726 → 727 px（V4.6.4 是 807）。Camera 页高度由左侧相机监视（777 px）决定，日志框恢复到 V4.6.3 的 33 px。
- 测试：`tests/test_v4_6_4_gui.py` 增加两项检查：
  - 识别之后才连接的夹爪在点击时被采用，未连接时拒绝；
  - Camera 日志高度 ≥ 33 px（V4.6.4 只有 1 px）。
- 版本 V4.6.5，白名单补 V4.6.5。
- **回归**：受影响的测试（控制台、V4.6.x、`v3_57 / v3_58 / v3_70`、Camera 相关 GUI）全部重跑，只有 V4.3 原有的失败；完整回归结果见 V4.6.4。

## V4.6.4 (2026-11-08) — 视觉精度验证（合爪对准，三种测量方式）
- 新模块 `NERO_V464_Vision_Check.py`（`V4.6.4 VISION ACCURACY CHECK`），由控制台挂到 Camera调试 →「视觉定位」页底部（`VC.attach`，不改冻结的 `NERO_V325_Camera.py`）。
  - 识别：`RealEggSkills.observe_scene`（与鸡蛋任务同一套识别，Agentic 页 YOLOE 设置）。
  - 夹爪合拢：`open_deg(0)`，即空夹标定的碰合点；未做空夹标定时拒绝。
  - 合拢爪尖：在抓取坐标系 G 中位于 (0, −WALL_HALF, +TIP_BEYOND_GRASP_M)，即固定指胶垫内表面末端，加上 Agentic 页的指尖修正。
  - 运动：IK 在 15° 一档的偏航里找可达姿态（爪尖朝下），关节运动到顶端上方 60 mm，再直线 20 mm/s 严格到位降到顶端 + 悬停高度（默认 10 mm）。
  - 三种方式：① 主相机；② 腕部（`SLOT_LOOK_POSES` 观察位，两次可信读数相差 ≤ 4 mm）；③ 主相机 + 腕部（修正 ≤ 15 mm 才采用）。
  - 记录：CSV + 照片存到 `NERO_CALIB_DIR/vision_check/`；按方式汇总；主相机画面里 +X / +Y 方向提示。
- 「视觉定位」页识别表 `tv_eggs` 运行时显示 4 行（原 6 行），为面板腾出位置。
- 控制台 / 实机模块版本 V4.6.4。
- 测试：
  - 新 `tests/test_v4_6_4.py`（6 项）：
    - 合拢爪尖模型与 `real_mount` 一致，`tcp_for_tip` 与 `tip_from_tcp_T` 互逆；
    - 三种方式在假机械臂上爪尖最终位置正确；
    - 不可达目标和未做空夹标定都拒绝且不运动；
    - CSV、汇总和方向提示正确。
  - 新 `tests/test_v4_6_4_gui.py`：启动整个控制台，在 1420×1000 窗口内所有控件和文字行完整可见；无相机时按钮拒绝并给原因；记录偏差流程正常。
  - 版本白名单补 V4.6.4。
- **回归**（和 V4.6.3 用同一套测试对比）：非 GUI 104 个文件、GUI 41 个，只有原有失败。第一次跑出 `test_v3_57 / test_v3_58` 两个回归：这两个测试用模拟的 tkinter 启动控制台，面板挂载失败后写失败日志时碰了模拟的文本框。已改成通过实机管线的日志队列写，复测通过。`test_v4_6_4_gui` 把窗口固定为 1420×1000，在 1600×1000 和 1920×1080 的虚拟屏幕上都通过。
- 已知限制：屏幕高度小于 1080 时，控制台窗口会小于 1000 像素，面板底部可能被截掉，请最大化窗口。
- **没有在真实硬件上运行过。**

## V4.6.3 (2026-11-08) — 热修复：V4.6.2 鸡蛋任务中控制台 abort（Tk 主线程 C 层）
- 现象：`Fatal Python error: Aborted`，主线程在 `NERO_V316._status → lbl_now.configure`；任务线程在 `_send_move` 等待，其余线程（相机采集、子进程 reader / watcher、CAN、pump）不碰 Tk。C 层原因只打印在终端，未记录。
- 跨线程 Tk 探针（包装 tkapp，记录非主线程的 Tk 调用，跑全部 GUI 测试）：`real_cfg()` 在工作线程（`_fw_cfg / _run_mode_helper / set_teach_mode / 关节任务 / 一致性测试`）里读 Tk 变量。V4.6.3：`NERO_V312.real_cfg` 在非主线程返回主线程快照（`real_poll` 和 `start_job` 刷新）；重跑探针后只剩冻结的 `NERO_V325._marker_from_ui`（手眼标定）。
- `NERO_V312_Console.py`（`V4.6.3 NATIVE CRASH GUARD`）：`v463_xinitthreads()` 在 `tk.Tk()` 前调用 libX11 `XInitThreads`（`NERO_NO_XINITTHREADS=1` 关闭）；`v370_crash_note` 在 crash.log 写一行状态。版本 V4.6.3（`NERO_V313` 同步）。
- `start_nero.sh`：stderr → 终端 + `logs/stderr.log`（`tee`，进程 abort 后仍保留；>5 MB 转 `stderr.prev.log`），`LIBC_FATAL_STDERR_=1`。
- 测试：新 `tests/test_v4_6_3.py`（4 项，含模拟 abort 的 stderr 落盘）、`tests/test_v4_6_3_gui.py`（启动整个控制台，工作线程调用 `pipe.cfg_fn()`：0 次 Tk 调用、与主线程快照一致；同一测试在 V4.6.2 上失败——工作线程的读取 5 s 未返回）；版本白名单补 V4.6.3。
- `v463_x_error_handlers()`（tk.Tk() 前用 ctypes `XSetErrorHandler / XSetIOErrorHandler`）：X 错误写入 crash.log 并忽略（最多记 50 条），不再走 Xlib 默认处理的 `exit(1)`（V4.6.2 第二次闪退：点 Smol VLA 页，crash.log 无任何记录 = 不是信号）；IO 错误先记录再由 Xlib 结束；主循环结束写 `console closed normally` / 退出原因。`NERO_NO_X_ERROR_HANDLER=1` 关闭。测试 H5：故意制造 BadPixmap，进程存活并有记录（不装处理器时同样操作 rc=1、无痕退出）。
- **回归**：同一套测试，非 GUI 103 个文件、GUI 40 个，失败的都是 V4.3 原有的（非 GUI：`v3_16 / v3_16_2 / v3_17 / v3_19 / v3_60 / v3_61 / v3_62`；GUI：`v3_25_gui / v3_53_gui / v3_57_gui / v3_58_gui`，`v3_56_gui` 时好时坏），**0 个回归**；`test_v3_25` 现在通过（V4.6.2 的白名单脚本误改了它的一行，已恢复，脚本已修正）。X 错误处理器加入后，又把所有导入 / 启动控制台的 70 个测试重跑一遍，结果同上。
- **没有在真实硬件上运行过；不能确定这就是根因**，下次若再崩，`logs/stderr.log` 会有 C 层原因。

## V4.6.2 (2026-11-08) — 两个实机任务页的示教按钮 +「高级」示教重力补偿 + Smol VLA 页日志移到底部 + Copy Log
- 新模块 `NERO_V462_Teach_Gravity.py`（`V4.6.2 TEACH BAR + GRAVITY COMP`）：`TeachBar`（「进入示教模式」「退出示教模式」「高级」+ 状态；调用 `RealPipeline.set_teach_mode`，退出 `return_home=False`，与「实机NERO调试」页一致；任务运行 / 机械臂运动时拒绝）；`GravityDialog`（档位 / 自定义质量 + 质心、更轻 / 更重、保存、保存并立即生效、检测驱动接口）；设置 `NERO_CALIB_DIR/teach_gravity.json`（`load / save / normalize / payload_arg / describe`）。Tk 只在主线程操作。
- `NERO_V313_Real_Robot.py`：`apply_teach_payload` 接受 JSON `{"mass_kg","com_mm","fallback"}`——只发给参数**名字**明确是质量（mass / weight / kg）+ 质心（com / cog / center 或 x / y / z）的接口（`_custom_payload_call`，单位 kg / m），首参数注解为 `str` 的档位接口不会收到数字；没有这类接口时改发 fallback 档位，`teach_payload_line` 写明；新 helper 子命令 `probe` + `RealPipeline.probe_teach_apis()`（列出 load / payload / gravity 接口及签名，不切换模式）。版本号 V4.6.2。
- `NERO_V312_Console.py`：`real_cfg()["teach_payload"] = TG.payload_arg(TG.load())`；「参数 → 示教负载」下拉框与「高级」共用同一设置（双向同步）；`safe_tab` 传给两个任务页（退出示教时同步安全区编辑器的模式，与原按钮一致）。
- `NERO_V365_Real_Cup_Task.py`（`V4.6.2 TEACH BAR + GRAVITY COMP + FULL-WIDTH LOG`）：「运行档位」框加示教按钮条；Copy Log 按钮行先以 `side="bottom"` 打包（以前排在拉伸的日志框后面，被挤出窗口）。
- `NERO_V4_Console_Tab.py`：控制栏第二行加示教按钮条；底部改为「监控一行（右边是 4. 实时状态，按可用宽度换行）+ 全宽日志 + Copy Log」；「0. 移动硬盘」详情可折叠（默认收起，硬盘缺失时自动展开）；`copy_log()`。
- 测试：新 `tests/test_v4_6_2.py`（6 项：设置、自定义负载发送规则、helper 参数、接口检测、接线、版本）、`tests/test_v4_6_2_gui.py`（1920×1050 和 1420×1000 下两页按钮 / Copy Log 可见、日志在监控下方且全宽、进入 / 退出调用、「高级」保存 / 立即生效 / 检测、Copy Log 剪贴板、发布目录不被写入）；`test_v4_6_1` 的布局断言跟随修改；版本白名单补 V4.6.2。
- **完整回归对比（同一套测试，原始 V4.3 vs V4.6.2）**：非 GUI 102 个文件，94 个通过，8 个两边都失败（V4.3 原有，同 V4.6.1），**0 个回归**（第一次跑时 `test_v3_70` 因检查控制台源码里 V3.70 的「示教负载」写法而失败，已改为保留原写法后再叠加 V4.6.2 设置，复测通过）；GUI 39 个，34 个通过，5 个两边都失败（V4.3 原有），**0 个回归**；控制台在 Ubuntu 22.04 Python 3.10 / Tk 8.6.12 下启动、打开 Smol VLA 页、退出正常，`crash.log` 无记录。
- **没有在真实硬件上运行过**；当前 NERO 驱动是否有负载接口未知（这里无法访问 pyAgxArm）。

## V4.6.1 (2026-11-08) — 修正 V4.6 鸡蛋策略（夹住又松开）+ 页面改名 + Smol VLA 页相机监控 + 测试白名单修复
- **鸡蛋**：`config/egg_carton.json` 新增 `held_gap_mm`=26.5（实测）、`offcentre_margin_mm`=6、`offcentre_regrasp`=false（默认只记录）、`wrist_slot_look_before_pick`=false、`wrist_gripper_mask`（默认图像下方 40%）、`wrist_overexposed_frac`=0.35 / `wrist_overexposed_level`=245，读取函数 `NERO_V349.egg_task_config()`（只读）。`RealEggSkills._egg_wrist_align` 只写日志；夹偏按基准判断；夹蛋前看槽按开关。`NERO_V386`：`wrist_view_gate`（排除夹爪区域、过曝判"不可用"）+ `wrist_contour_clipped`（被截断的轮廓不计算），两个腕部识别都用。
- **到位等待从冻结的 `NERO_V325_Camera.py` 移出**：V4.6 改了 V325（V3.80 起由 18 个测试的快照哈希冻结）。V4.6.1 把 V325 恢复到 V4.3 原样，同样的功能改在 `NERO_V313.RealPipeline.move_and_settle` 里（`_v46_wait_arrival`：机械臂还在动就继续等，最多 8 s，到位 ≤ 0.3°；停在半路则不再等，并写 `[WARN] [V4.6.1 ARRIVAL] … J4 仍差 9.2°`）。`NERO_V313` 版本号改为 V4.6.1（`test_v3_46` 要求控制台 / 实机模块 / VERSION 三者一致）。
- **页面**：「NERO实机任务」→「Agentic实机任务」，「VLA / LeRobot (V4)」→「Smol VLA实机任务」（`NERO_V312.TAB_AGENTIC / TAB_SMOLVLA`）；`NERO_V365.CameraMonitorMirror` 在 Smol VLA 页复用 Agentic 页的绘图代码（`_draw_main / _draw_overlay / _draw_tips / _draw_wrist` 加了可选的 `view` 参数，默认行为不变）。
- **测试白名单（V4.4 起的遗留问题）**：77 个测试文件里的 167 处版本白名单补上 V4.1–V4.6.1（只改白名单）。以前从 V4.4 起，`test_v3_65` 等十几个测试一开头就因版本号停止，后面的检查没有运行；V4.4–V4.6 说的"和上一版结果一致"因此不充分。
- **完整回归对比（同一套修好白名单的测试，Python 3.13 + Ubuntu 22.04 Python 3.10/Tk 8.6.12）**：
  - 非 GUI：原始 V4.3 vs V4.6.1 —— 89 个文件两边都通过，8 个两边都失败且失败原因相同（V4.3 原有），**0 个回归**；V4.4 / V4.5 / V4.6 / V4.6.1 的新测试全部通过。
  - GUI（虚拟显示器）：32 个两边都通过，5 个两边都失败（V4.3 原有；其中 `test_v3_56_gui` 在 V4.3 上也时好时坏），**0 个回归**；新增 `test_v4_6_1_gui` 通过（Smol VLA 页监控和 Agentic 页逐项一致）。
- 新测试：`tests/test_v4_6_1.py`（8 项）、`tests/test_v4_6_1_gui.py`；`test_v4_6.py` 的 E1 / E2 / E4 / E5 按 V4.6.1 的行为修改；`test_v3_65` 的页面名断言改为「Agentic实机任务」。
- **没有在真实硬件上运行过**。

## V4.6 (2026-11-08) — SmolVLA 实机安全门 + 如实报告 + 状态灯 / 日志；鸡蛋放槽：相机一致性、夹偏重夹、严格到位、夹蛋前看槽
- `NERO_V4_SmolVLA_Policy.py`：`model_profile()`（状态/动作维度、训练数据集、是否为 NERO 训练、是否允许实机）；`run_vla_closed_loop` 在实机后端拒绝非 NERO 模型（`MODEL_NOT_TRAINED_FOR_NERO`），开始 / 每块 / 结束 / 每种失败都写日志，结果 `executed_unverified`。
- `NERO_V44_SmolVLA_Worker.py`：返回按归一化统计得到的动作维度、`train_config.json` 里的数据集名和训练步数；`NERO_V44_FAKE_DIMS` 供测试。
- `NERO_V4_Task_Evaluator.py`：没有仿真真值 / 视觉 / 目标状态时返回 `unverified`（`EXECUTED_NOT_VERIFIED`），不再是 `SUBTASK_COMPLETED`；unverified / 模型被拒 / 控制权被拒 → 不重试。`NERO_V4_Agent_Orchestrator.py`：顶层任务 `unverified` 时如实报告。
- `NERO_V4_Console_Tab.py`：`StatusLamp`（Canvas 圆点，不依赖字体）；硬盘 / 环境 / 模型 / 后台任务 / ① / ② / ③ / ④ 灯；`_start_run`（一次只跑一个、开始和结束都有日志）；按钮忙碌状态；横幅显示后端；急停按钮单独固定在控制栏右侧，控制栏分两行。
- `NERO_V44_Storage.py`：卷标不区分大小写；环境验证超时 180 s，超时提示可读。
- `NERO_V365_Real_Cup_Task.py`（`V4.6 EGG CAMERA AGREEMENT + STRICT ARRIVAL`）：`_wrist_look_slot_prepick`（空爪、最多 6 个观察位置、两次一致）、`_egg_wrist_align`（≤ 6 mm 修正 / 6–25 mm 复看确认 / 否则停止）、夹偏检测与重夹一次、`RealIO.move_line(strict=True)`（2 mm / 1° / 0.5°，修正一次）、下插不精确时不松开、`_release_tip_error`；删除 V4.5 的转腕前复测。
- `NERO_V386_Real_Wrist_Camera.py`：`wrist_find_slot_cavity`（圆形暗坑 + 相邻槽检查）。
- `NERO_V325_Camera.py`：`_wait_arrived`（标定拍照前每个关节 ≤ 0.3°，10 s 未到位则跳过该样本）。
- `NERO_V312_Console.py`：`v46_rotate_console_log`（`console_<时间>.log` 保留 10 份）。
- 测试：新 `tests/test_v4_6.py`（11 项）+ `tests/v46_synth_slot.py`；`test_v4_2` Benchmark 测试按设计修改；`test_v4_5` T6 / T7 跟随 V4.6 修改。
- **没有在真实硬件上运行过**。

## V4.5.1 (2026-11-08) — 热修复：控制台启动时在 VLA 页创建控件处段错误
- `NERO_V312_Console.py`（`V4.5.1 LAZY VLA TAB`）：「VLA / LeRobot (V4)」页在第一次打开时才 import 和创建；`NERO_NO_VLA_TAB=1` 完全不加载。
- `start_nero.sh`：新参数 `--no-vla`。
- `NERO_V4_Console_Tab.py`：去掉所有 emoji / 彩色符号（Ubuntu 22.04 libXft 2.3.4 + 彩色 emoji 字体是最可疑的原因，未证实）；`os.environ` 改到界面建好后由 `_tick` 设置。
- 验证：Ubuntu 22.04 的 Python 3.10 + Tk 8.6.12（虚拟显示器）下控制台 4 s 启动、打开 VLA 页正常创建（87 个控件）、正常退出、`crash.log` 无记录；`--no-vla` 正常；V4.1/V4.2/V4.4/V4.5 测试通过（`test_v4_5.py` 新增 T10）。**原始崩溃在这里没有复现**。

## V4.5 (2026-11-08) — 鸡蛋放进蛋托的精度 + 指尖 z 安全 + V4.4 闪退加固
- **`NERO_V365_Real_Cup_Task.py`（`V4.5 EGG PLACE ACCURACY + FINGERTIP Z SAFETY`）**：夹蛋时测量蛋心在夹爪坐标系的位置（`plan["egg_in_G_m"]`），放蛋用同一个量并把鸡蛋**下端**对准槽心（`egg_place_grasp_origin`）；俯仰角先试 ±90°（`EGG_PITCH_ORDER`）；转腕前腕部相机俯视复测槽位（`_wrist_look_slot`，可信偏差 1.5–15 mm 时重新规划）；夹持胶垫中心在鸡蛋赤道、最低指尖 ≥ 桌面 + 5 mm（`so101_tip_drop_m` 含动爪摆动）；安全区 z 检查改用最低指尖（规划 5 mm、执行 3 mm，`so101_tip_tool_T`）；桌面取相机 / 安全区的较高者，相差 > 15 mm 拒绝（`TABLE_HEIGHT_DISAGREE`）；下探最后 15 mm 以 10 mm/s；规划按执行同样的直线插值检查关节余量（多 1° 余量）；`_check_path_fence` 的 z 下限不再被第一次的桌面高度缓存住。
- **`NERO_V349_Real_Wrist_Camera.py`**：`refine_slot_grid`（空槽凹陷 → 拟合槽距 + 整体偏移 → 再检测，最多 5 轮）+ `egg_carton_config`（`config/egg_carton.json` 或 `NERO_EGG_SLOT_PITCH_MM`）；蛋托日志显示槽距来源和修正量；外框比 槽距×列数 大 40 mm 以上时提示“盖子或边沿被算进外框”。
- **闪退加固**：`NERO_V4_Console_Tab.py` 的 SmolVLA 加载全部移出 Tk 主线程；`os.environ` 只在主线程且值变化时写；拔盘时关闭 worker 改到后台线程；`NERO_V4_SmolVLA_Policy.py` 加载加锁；`NERO_V312_Console.py` 启动时保留 `logs/console.prev.log`。
- **测试**：新 `tests/test_v4_5.py`（9 项）；`test_v4_1.py` / `test_v4_2.py` 的夹持高度断言改为 21.0 / 11.0 mm（有意的改动）。
- **实机第一次怎么试**：① 量蛋托相邻两槽中心距离，填到 `config/egg_carton.json`（可选）；② 「NERO实机任务」选鸡蛋场景，先用计划预览 / 悬停档：日志 `plan_egg_place` 一行应有 `lowest_tip_z_mm` ≥ 5、`egg_in_G_mm` ≈ [0, 20, 0]、`place_pitch_deg` 尽量 ±90；③ 执行时注意 `[V4.5 WRIST SLOT]` 一行（腕部相机修正了多少，或为什么没用）；④ 如果出现 `TABLE_HEIGHT_DISAGREE`，先在 Camera调试页复核手眼标定和安全区桌面高度。
- **没有在真实硬件上运行过**。

## V4.4 (2026-11-08) — LeRobot + SmolVLA 环境、模型、数据集、训练文件全部放到移动硬盘 EMBODIED_AI
- **新模块 `NERO_V44_Storage.py`**（`V4.4 EMBODIED_AI DRIVE STORAGE`，只用标准库）：按卷标找硬盘（`/proc/self/mounts` + `/dev/disk/by-label`，必须是真实挂载点）；`<盘>/nero_lerobot/` 目录布局；文件系统检查（剩余空间、符号链接、`noexec`、FAT32/exFAT/NTFS 提示）；`udisksctl` 挂载 / 卸载 + 断电；把 `HF_HOME`、`HF_HUB_CACHE`、`HF_LEROBOT_HOME`、`TORCH_HOME`、`PIP_CACHE_DIR`、`UV_*`、`TMPDIR`、`XDG_CACHE_HOME` 都指向硬盘；生成安装脚本（可重复运行，中断后续装）；`lerobot-train` 命令；后台任务 `BackgroundJob`（与控制台分离的进程组，日志写硬盘，控制台重启后自动接上仍在运行的任务）；旧文件迁移。
- **新模块 `NERO_V44_SmolVLA_Worker.py`**（`V4.4 SMOLVLA DRIVE WORKER`，在硬盘 venv 里运行）：`serve`（标准输入输出协议，JSON + 原始图像字节；LeRobot ≥ 0.4 自动加载 pre/post processor，≤ 0.3 直接调用策略）、`download`（smolvla_base + 配置里的 VLM 主干）、`export`（NERO 示教 npz → 真实 LeRobotDataset，兼容 `add_frame(frame, task)` / `frame["task"]` 两种 API，视频编码失败自动改存 PNG）、`env`。
- **`NERO_V4_SmolVLA_Policy.py`**：新后端 `hf_smolvla_ext`（`SmolVLAWorkerClient`）；加载顺序：NERO checkpoint → 控制台内 lerobot → **硬盘 venv 里的 SmolVLA** → 内置替身；`lerobot/smolvla_base` 自动换成硬盘上的本地副本；模型下载默认到硬盘，硬盘不在时拒绝下载到本机；状态诊断加「5. 移动硬盘」行；Checkpoint 默认目录随硬盘。
- **`NERO_V4_Dataset_Recorder.py`**：数据集默认目录在创建时解析（硬盘 `datasets/nero`，硬盘不在时退回旧目录）。
- **`NERO_V400_Phase0_Audit.py`**：先找硬盘 venv；子进程去掉 ROS 的 `PYTHONPATH`；从 USB 硬盘第一次 `import lerobot` 的超时 5 s → 90 s。
- **`NERO_V4_Console_Tab.py`**（`V4.4 EMBODIED_AI DRIVE PANEL`）：新「0. 移动硬盘」区（状态 3 行 + 检测 / 挂载 / 初始化 / 安装 / 验证 / 停止后台任务 / 打开目录 / 旧文件迁移 / 安全移除；Python 模式、基础 Python、pip 包、装完自动下载模型）；第 3 区第二行「LeRobot 真实微调」（导出 → 步数 / batch / 额外参数 → 开始微调 → 使用最新训练结果）；每 5 s 检查硬盘插拔，自动把示教数据集切到硬盘 / 切回本机；关闭控制台时停止硬盘 SmolVLA 进程，训练任务继续运行。
- **`start_nero.sh`**：检查并编译新模块，启动时打印一行硬盘状态。**`config/nero.env.example`**：V4.4 环境变量说明。**`tests/test_v4_4.py`** + `check_release.sh` 标记检查。
- **第一次使用步骤**：
  1. 插上 EMBODIED_AI（最好 ext4，≥ 25 GB 空闲；实测装完占 16.5 GB）→ `./start_nero.sh`，终端第一行显示 `[OK] V4.4 drive EMBODIED_AI: ...`。
  2. 「VLA / LeRobot (V4)」→「📁 初始化目录」→「🛠 安装 LeRobot+SmolVLA 到移动硬盘」（约 10–30 分钟，看网速）；结束后日志出现 `NERO_JOB_DONE ok`，再点「🧪 验证环境」应显示 `CUDA ✓ NVIDIA GeForce RTX 4060 Laptop GPU`、`lerobot 0.x`、`SmolVLAPolicy ✓`。
  3. 后端选「仿真替身」，点「① 加载 SmolVLA & 连接 Adapter」，日志应出现 `[OK] [V4.4 SMOLVLA] 已在移动硬盘环境加载 SmolVLA`，横幅 `SmolVLA=hf_smolvla_ext`；再点「③ 执行 SmolVLA 闭环」看推理时间。**未授权「② 允许控制」时不会动机械臂**。
  4. 如果安装失败：日志在 `<盘>/nero_lerobot/logs/install_*.log`；脚本 `install_*.sh` 可以在终端直接 `bash` 重跑（会接着装）。
- **真实环境测试中发现并修复**：`LEROBOT_HOME` 已废弃（设了它 lerobot 0.7 直接拒绝导入）；pip 包默认改为 `lerobot[smolvla,dataset,training]`；去掉 `--wandb.enable`；从 smolvla_base 微调时自动 `--rename_map`（cam_main→camera1，cam_wrist→camera2）；微调后的模型状态维度按归一化统计（8 维）。
- **测试情况**：见 README V4.4 第 9 条。**没有在用户电脑、真实硬盘、RTX 4060、真实机械臂上运行过**。

## V3.89 (2026-11-06) — 腕部 4MP 相机光轴稳定化手眼求解器 + 甜区采样规划 + V4L2 防黑帧自动调参
- **光轴稳定化 Eye-in-Hand 求解器 (`solve_eih_stabilized` & `_calibrate_eih_at_scale_stabilized`)**：消除俯视采样中“深度缩放 `alpha_z`”与“相机沿光轴偏移 `t_X`”的共线奇异性（解决删减样本时 `link7->cam` 在 `+18..+146 mm` 间跳变及留一误差虚高至 `5.7 mm` 的问题），结合原始 2D 角点 `(fx, k1 ∈ [-0.28, +0.02])` 联合自标定，将留一误差稳定压入 `1.3–3.7 mm`。
- **距离与中心甜区约束自动采样规划 (`plan_eih_auto_targets_4mp`)**：将 24 mm 桌面标记的自动环绕采样距离控制在 `120–360 mm`、离轴比 `<= 0.40`，杜绝 `z=456 mm` 远距点与 `x=+146 mm` 极限边缘点混入；初值探测新增双向纯平移候选。
- **V4L2 防黑帧亚毫秒级一键自动调参 (`autotune_wrist_uvc` & `set_exposure_gain`)**：调参循环仅更新曝光与增益并等待 2 帧新画面，禁止下探 `< 0.8` 导致 V4L2 整数截断为 `0`（全黑帧 `平均 0 < 100`），遇到硬件曝光下限自动回滚，不再冲死 UVC 视频流。

## V3.88 (2026-11-06) — 腕部相机 Eye-in-Hand 手眼标定失败修复（隔离旧 130° 广角畸变内参 + 角点级焦距/畸变联合自校准 + OV4688 亚毫秒一键调参）
- **问题根因分析（车库实机日志）**：
  - 更换为新腕部相机 Innomaker 4MP (U20-4MP OV4688，标准约 90° 镜头) 后，`~/Downloads/cameracalib/` 中仍自动加载了旧 1080P 130° 广角镜头的内参文件（`fx ≈ 680 px` 及强桶形/8 参数有理畸变 `k1 ≈ -0.42`）；
  - 当采样点偏向画面边缘时，用旧 130° 畸变系数对新 4MP 相机角点做 `cv2.undistortPoints` 产生了严重的非线性反桶形拉伸，导致：
    1. 初值探测拟合残差高达 4.0–6.2 mm，自动规划出的采样姿态大量偏出新镜头较窄的视野（第 2 次采集 9 个姿态只检测到 0 帧）；
    2. 采集到的 13 个样本因边缘非线性畸变错配，线性尺度缩放无法消除非均匀误差，导致最终手眼残差高达 36.41/70.20 mm、留一 47.64/76.26 mm，`link7→cam` 被拉偏至 `(+52.3, +147.9, -31.5) mm`；
  - 此外，OV4688 传感器感光度极高，原一键自动调参最低曝光下限为 `10` (1.0 ms)，在车库双灯管下仍出现 34%–46% 过曝，需手动降到 `0.8` 才合适。
- **三重加固修复**（`NERO_V386_Real_Wrist_Camera.py` + `NERO_V325_Camera.py`）：
  1. **旧 130° 广角内参自动识别与隔离**（`sanitize_intrinsics_for_model`）：连接 Innomaker 4MP (OV4688) 相机时，自动检查当前加载的内参是否属于旧 130° 广角相机（`fx/w < 0.44` 或 `|k1| > 0.22` 或未标记 4MP 型号）；一旦发现不匹配，立即自动隔离旧畸变并切换为 OV4688 标准无畸变针孔初值（`fx = fy = 0.55*w, dist = 0`），彻底杜绝旧广角畸变污染新相机 PnP 与采样规划；
  2. **基于原始 2D 角点的 `(fx, k1)` + 手眼联合自校准**（`calibrate_eih_4mp`）：腕部采样记录时同步保存标记的 4 个原始像素角点 `corners_px`；计算手眼标定时，若发现留一误差 > 5 mm，自动利用 2D 角点在 `fx ∈ [0.34w, 0.78w]`、`k1 ∈ [-0.10, +0.02]` 空间内重算无污染 PnP 位姿并联合求解手眼变换，将 36–70 mm 的非线性畸变残差直接恢复至 < 1 mm，并同步更新相机内参；
  3. **OV4688 亚毫秒级一键自动调参**（`autotune_wrist_uvc`）：将腕部相机一键自动调参的曝光搜索范围向下扩展至 `15, 10, 5, 2.5, 1.5, 1.0, 0.8, 0.5, 0.2, 0.1`，适配 OV4688 高感光特性。
- **测试**：新增 `tests/test_v3_88.py`（A1–A3），1:1 复现旧 130° 广角内参污染下的 13 点手眼标定失败场景并验证自动恢复（LOO < 1.0 mm，`link7→cam` 恢复真实值）。
- **没有在真实 Innomaker U20-4MP 硬件与真实机械臂上运行过**。

## V3.87 (2026-11-05) — 腕部相机标定适配（第二阶段）+ 叠杯抓杯角度与同心度闭环视觉伺服（第三阶段）
- **第二阶段：OV4688 4MP / 1080P 镜头内参与 Eye-in-Hand 标定闭环**（`NERO_V386_Real_Wrist_Camera.py` + `NERO_V325_Camera.py`）：
  - `calibrate_intrinsics_4mp`：兼容 Innomaker U20-4MP OV4688 的标准/广角镜头（60°–130° FOV，支持 `2688×1520` 4MP 与 `1920×1080`），保留运动学焦距锚定与防过拟合门禁；
  - `scale_K_for_frame`：当标定内参分辨率（如 `2688×1520`）与实时伺服分辨率（如 `1920×1080`）不同时，自动按画面宽高比无损换算内参矩阵 $K$；
  - 原 `NERO_V349_Real_Wrist_Camera.py` 保持与 V3.58/V3.60 字节级一致（仅更新 `VERSION`），确保旧相机备用代码与历史校验完全兼容。
- **第三阶段：抓杯前夹爪开合角垂直度与杯沿位置闭环调整**（`wrist_check_cup_grasp` + `NERO_V362_Cup_Skills._wrist_recheck_grasp`）：
  - 在 `pick_cup` 到达杯口上方预抓取点（`pregrasp`）后、下插接触前，自动触发腕部相机俯视检测；
  - 将 2D 杯口边缘射线投影至 `base_link` 杯口水平面 $Z = z_{\text{rim}}$，使用代数 Kasa + 鲁棒中值剔除拟合 3D 杯口中心 $(x_c, y_c)$ 与半径；
  - 计算夹爪实际开合连线向量与杯沿夹取点径向法线的夹角误差 $\Delta\theta$ 以及杯口中心平移偏差 $(\Delta x, \Delta y)$：
    - 当平移偏差 $\ge 1.5\text{ mm}$ 或角度误差 $|\Delta\theta| \ge 2.0^\circ$（且在安全阈值 $\le 25\text{ mm} / 35^\circ$ 内）时，自动重算 `pregrasp / grasp / lift` 6-DoF IK 轨迹并原地微调手腕与水平位置，确保夹爪开合连线与杯壁严格垂直后再垂直下插。
- **第三阶段：叠杯套入前双杯同心度闭环调整**（`wrist_check_cup_concentricity` + `NERO_V362_Cup_Skills._wrist_recheck_place` + `NERO_V365_Real_Cup_Task.py`）：
  - `RealIO` 接入 `wrist_snapshot`（`wrist_from_tab`），在 `place_cup_into` 悬停于目标杯上方时触发腕部相机同心度复检；
  - 同帧提取目标杯外圈杯口与被夹杯内圈杯底（或结合标定夹持偏置），在各自水平高度平面拟合 3D 圆心 $(x_{\text{target}}, y_{\text{target}})$ 与 $(x_{\text{held}}, y_{\text{held}})$；
  - 当同心度误差 $\ge 1.5\text{ mm}$ 时自动执行水平微平移补偿，直至上下两杯同轴对齐后再垂直下降释放。
- **测试**：新增 `tests/test_v3_87.py`（Z1–Z4），全面验证 4MP (2688×1520) 内参与手眼标定精度、抓杯前位置+角度（-18° 偏角）3D 检出与修正、叠杯前双杯同心度偏差检出、以及 `CupSkills`（`pick_cup` + `place_cup_into`）全流程闭环执行；全量回归测试通过。
- **没有在真实 Innomaker U20-4MP 硬件与真实机械臂上运行过**。

## V3.86 (2026-11-05) — 腕部相机升级为 Innomaker 4MP USB HDR Camera (U20-4MP OV4688) + 腕部相机高级测试 GUI（第一阶段）
- **新腕部相机硬件驱动与旧版备用保留**（新模块 `NERO_V386_Real_Wrist_Camera.py` + `NERO_V349_Real_Wrist_Camera.py`）：
  - 新增 `Innomaker4MPWristSource` 驱动类，支持 Innomaker 4MP USB HDR Camera（U20-4MP OV4688）的原生 `2688×1520` (4MP)、`1920×1080` 与 `1280×720` 分辨率及硬件 HDR 宽动态背光补偿控制；
  - `list_wrist_cameras` 自动优先枚举 `U20-4MP / OV4688 / 4MP / HDR` 节点并过滤 RealSense 节点；
  - 原 `NERO_V349_Real_Wrist_Camera.py` 中的 `UsbWristSource`（Innomaker 1080P 130° 广角）完整保留作为备用，可在 GUI 型号下拉框中一键切回。
- **相机信息 TAB 界面更新与「腕部相机高级测试」入口**（`NERO_V325_Camera.py`）：
  - 「相机信息」页腕部相机标题更新为 `腕部相机（Innomaker 4MP USB HDR Camera · U20-4MP OV4688，eye-in-hand）`，新增型号选择框（默认 4MP HDR，支持切回 1080P 130° 备用）、`2688×1520` 分辨率选项与 HDR 锁定开关；
  - 在腕部相机操作行新增 **「腕部相机高级测试」** 按钮（`on_wrist_camera_lab`），点击弹出 3 页专用测试与校正复核窗口 `WristCameraLab`：
    - **① 腕部相机参数与HDR**：实时显示工作区亮度与高光统计，支持一键防闪烁自动调参、手动调整曝光/增益/白平衡/伽马/锐度/OV4688 HDR 宽动态，以及命名参数组（`NERO_CALIB_DIR/wrist_profiles/`）保存与连接时自动套用；
    - **② 对焦与清晰度测试**：实时计算中心与四角 Laplacian 清晰度方差、Tenengrad 边缘梯度、历史峰值保持与四角均衡度，辅助 50–150 mm 近景工作距调焦；
    - **③ 标定与TCP准星复核**：支持调节投影工作距离 Z（40–300 mm），实时叠加 TCP 中心十字准星、夹爪开合连线方向与左右爪尖落区，并提供红杯杯口中心偏差 $(\Delta x, \Delta y)$ mm 与夹爪开合线垂直度误差 $\Delta\theta$ (°) 实时检测预览。
- **测试**：新增 `tests/test_v3_86.py`（W1–W4），覆盖 OV4688 设备枚举优先级与 4MP 分辨率解析、UVC+HDR 参数组持久化与实时应用、清晰度与红杯杯沿/夹爪角度预览、以及 `CameraTabUI` 与 `WristCameraLab` 三页交互闭环。
- **没有在真实 Innomaker U20-4MP 硬件上运行过**。

## V3.85 (2026-11-05) — 主相机手眼标定加固（红外深度平整簇优先采纳 + PnP 运动学探针与 1D 最优尺度自校准）
- **红外深度平整簇采纳门限放宽**（`NERO_V325_Camera.detect_aruco`）：
  - 提取标记中心 20% 区域的深度并检测平面平整度（std < 8 mm）；
  - 将原版过严的 30 mm 门限升级为自适应容差 `max(0.080, 0.25 * p_pnp[2])`；
  - 解决暗光环境长曝光/高增益下白底晕光侵蚀黑边导致 PnP 虚高 50–70 mm 时深度被误剔除的问题，优先采用 D435i 真实物理双目红外深度。
- **主相机 PnP 尺度自校准**（`NERO_V325_Camera.calibrate`）：
  - 第一重：自动利用方向探测步（40 mm 纯平移）的真实机械臂位移反算尺度比例 $s_{	ext{probe}} = \Delta_{	ext{robot}} / \Delta_{	ext{pnp}}$；
  - 第二重：在 $lpha \in [0.70, 1.30]$ 范围内对手眼配准残差做一维最优尺度网格与黄金分割搜索；
  - 自动消除打印缩放、手工测量或晕光导致的 10%–20% 尺度误差，将残差由 10–20 mm 恢复至 < 2 mm；
  - 标定日志与 `camera_calib.json` 保存 `scale_factor` 与 `unified_source`。
- **测试**：新增 `tests/test_v3_85.py`（U1–U4），覆盖红外深度抗 64 mm PnP 偏差平整采纳、15% 尺度偏差自动校准至 $lpha pprox 0.8696$（LOO < 1 mm）、标定元数据持久化。
- **没有在真实硬件上运行过**。

## V3.84 (2026-11-05) — 主相机多帧跟踪 + egg carton 精调模型 6 孔槽位与空满联合定位 + 桌面平面基准锁定 + 10 帧 A/B 实测对比评测
- **主相机多帧目标跟踪与抓前重检**（新模块 `NERO_V384_Object_Tracker.py` + `Camera调试` 视觉定位子页第 6 行）：
  - `MultiFrameTracker`：在 `base_link` 坐标系下按类别与 3D 距离（默认 25 mm）跨帧关联目标，固定 `egg_1`、`egg_2` 编号不跳变；
  - 静止目标做指数滑动滤波（EMA，`alpha=0.35`），并在表格说明列实时对比 `抖动 X.Xmm(原 Y.Ymm)`；当目标位移 > 15 mm 时判定为真实移动，立即重置滤波历史，无拖尾延迟；
  - `pregrasp_recheck`（「抓前重检」按钮）：在悬停/下探前用主相机复核目标水平偏移，≤ 3 mm 放行，3–15 mm 提示微调对齐，> 15 mm 或丢失则中止下探。
- **`egg carton` 精调模型算 6 孔坐标与空满状态**（`locate_eggs_and_carton`）：
  - `NERO_V347_Yolo_Perception.BUILTIN_SYNONYMS` 为 `egg carton` 增加内置同义词 `egg holder, egg tray, carton, tray`，使蛋托提示词自动路由到用户的 `egg carton` 精调模型；
  - 点击「识别鸡蛋+蛋托6孔」（或「识别蛋托」），组合模型同帧检测鸡蛋与蛋托，分别拟合鸡蛋 3D 椭球中心与蛋托 2×3 槽位 `slot_1..slot_6` 的 3D 坐标，自动交叉比对鸡蛋落点与槽位高度判定“空槽 / 有蛋”。
- **桌面平面锁定与漂移监控**（`fit_workspace_table_plane` / `check_table_drift`）：
  - 点击「锁定当前桌面」将当前桌面法向量与中心高度保存至 `NERO_CALIB_DIR/table_plane_lock.json`；
  - 每次识别自动计算当前桌面相对锁定基准的 `Δz` 与倾角差，超过 3 mm 或 2° 时在界面状态栏给出 `⚠ 偏差超限` 提醒。
- **一键 10 帧 A/B 量化对比评测**（`run_ab_benchmark` / 「▶ 10帧对比评测」按钮）：
  - 方便用户在桌面上直接用数据验证改进收益：连续跑 10 次，输出单帧原始 vs 多帧中值+EMA跟踪的检出率、3D 坐标标准差（mm 及降低百分比）、ID 跳变次数与蛋托 6 孔状态一致率。
- **有意改动与老测试更新**：
  - tests 61–82 中 `NERO_V325_Camera.py` 的 SHA256 常量统一更新；老测试版本白名单扩展至 V3.84。
- **测试**：新增 `tests/test_v3_84.py`（T1–T6），覆盖多帧跟踪防跳变与位移重置、抓前重检三档门禁、鸡蛋+蛋托 6 孔空满联合定位、桌面锁定与漂移告警、10 帧 A/B 量化评测。
- **没有在真实 D435i / GPU / 机械臂上运行过**。

## V3.83 (2026-11-05) — GPU 自修复（start_nero 预检 + 控制台一键重载 nvidia_uvm）；组合模型（严格单复数匹配、Menubutton 多选组合、按物体分配提示词、去重叠框）；深度滤波接入与蛋表面到几何中心换算
- **GPU 自修复**（`NERO_V383_Gpu_Check.py` + `start_nero.sh`）：
  - 针对笔记本休眠唤醒后 `torch.cuda.is_available()=False`（此时 nvidia-smi 正常但 nvidia_uvm 损坏）导致 YOLOE 静默退回 CPU（0.5–6 s/帧）的痛点，在控制台启动前通过独立子进程（`clean_child_env`）自动预检 CUDA；
  - 若 `torch.cuda` 失败但 `nvidia-smi` 正常，优先尝试 `sudo -n /usr/sbin/rmmod nvidia_uvm && sudo -n /usr/sbin/modprobe nvidia_uvm` 免密重载；未配免密时在终端交互提示输入 sudo 密码并复核；
  - 仍失败则以 CPU 启动控制台并给出 `[WARN]` 及长期预防建议（`systemctl enable nvidia-suspend/resume/hibernate`）；
  - 控制台运行中若 YOLOE 处于 CPU，在 `Camera调试` 视觉定位页给出明确 WARN 并提供「修复 GPU」按钮（通过 `pkexec` 重载 `nvidia_uvm`，提示修好后重启控制台）；
  - 训练子进程（`NERO_V381_Yolo_Train.run_job`）启动同样进行 GPU 预检，降级 CPU 时通过 progress 事件在 ⑤ 页展示警告。
- **组合模型**（`NERO_V347_Yolo_Perception.py` + `NERO_V325_Camera.py` + `NERO_V381_Yolo_Train.py`）：
  - **Bug 修复**：`finetune_covers` 升级为严格匹配（小写完全一致，单复数视为一致），为 egg 补充内置同义词 `white egg, white round object`，彻底解决 “egg carton” 被误判给 egg 模型的问题；
  - **模型组合字符串**：`v_ymodel` 采用 `;` 连接多个模型（原模型 + 若干精调模型）。V347 新增 `parse_model_set` / `find_model_set`，`find_model` 返回兼容 `os.path` 的 `ModelSetPath`，保证 V366 实机任务等现有调用方零改动自动生效；
  - **YoloEDetector.detect 按物体分配**：
    - 每个提示词优先分配给覆盖它的精调模型（多模型冲突时按 recall 降序、创建时间降序选优）；
    - 精调模型以固定类别运行，绝不调用 `set_classes`；未被精调模型覆盖的提示词交由原模型文字提示检测；
    - 通配提示词（“全部”）自动执行各精调模型固定类别 + 原 -pf 模型；
    - 视觉提示（vp）模式固定走原模型；
    - 各模型各自缓存，消除反复重复加载；
    - 结果合并时对 IoU ≥ 0.5 的重叠框去重，并在结果标记 `model` 归属；
  - **Camera调试 页控件升级**：模型单选下拉框升级为 `ttk.Menubutton` + 勾选菜单，原模型锁定选中不可取消，精调模型默认全选，按钮显示实时摘要，配置自动持久化进 `locate_settings.json`；
  - **⑤ 第 5 步「应用到 Camera调试 页」**：改为将选中精调模型追加至组合，「恢复原模型」恢复为仅原模型；
  - **④ 采集标注**：「自动标注用」增加“组合模型”选项；
  - **命名规范**：统一修改为“Camera调试 页”。
- **深度滤波接入与几何中心换算（V4 准备清单第 10 项）**：
  - Camera调试页「深度」设置新增滤波模式（默认单帧 / 滤波 + 5 帧中值 / 10 帧中值），静止抓取识别前支持调用 `B378.median_stack` 消除边缘散斑抖动；
  - 完善蛋表面点云到椭球几何质心的换算，消除斜视投影下质心垂直与水平偏移。
- **有意改动与老测试更新**：
  - tests 61–82 中 `NERO_V325_Camera.py` 的 SHA256 常量统一更新；
  - 老测试版本白名单扩展至 V3.83；
  - `tests/test_v3_82.py` 适配组合模型的模型归属标签及 `on_apply_model` 追加语义。
- **测试**：新增 `tests/test_v3_83.py`（S1–S7），覆盖 GPU 检查自修复、严格同义词匹配、组合模型按物体分配、绝不调用 set_classes、框去重、单模型兼容、多选控件持久化、深度中值滤波。
- **没有在真实 GPU / D435i 上运行过**。

## V3.82 (2026-11-05) — 精调模型用到 Camera 页和实机；标注辅助（同义提示词、低分候选一键确认、视觉样例、用精调模型重新标注）；镜面亮斑 / 亮度下限 / 伽马；训练改 1280
- **精调模型用到实机**（第 9 项收尾）：
  - 在「⑤ 训练 / 评估」新增第 5 步「已保存的模型」：选一个模型，显示它的类别、日期、识别率和结论，点「应用到 Camera 页」或「恢复原模型」。
  - Camera 页的模型下拉框里也会列出 `finetuned/<名称>/<文件>`。实机任务用的就是 Camera 页的识别设置，所以选中后同时生效。
  - **只改了 V347**（识别模块），Camera 页（V325）和实机杯子识别（V366）都没改：
    - 精调模型按“固定类别”使用，和 -pf 模型一样，**不再设置文字提示**（设了会覆盖训练结果）。识别结果按物体名筛选，同义提示词也算。
    - **要找的物体不在精调模型里时，自动交给原模型**用文字提示去找；和精调模型结果重叠的框去掉。日志和结果里注明每个框来自哪个模型。
- **标注辅助**（「④ 采集标注」，解决鸡蛋几乎都要手画的问题）：
  - **每个类别可以设多个提示词**：选中类别后填「提示词」并保存。egg 默认是 `egg, white egg, white round object`，和 Camera 页一致。用任何一个提示词找到的框，都记到这个类别名下。
  - **两个门槛**：
    - 「找框门槛」默认 0.10；
    - 「直接确认门槛」默认 0.30：达到的直接算标注；低于它的显示成**灰色虚线候选框**，**双击或回车确认**，Del 删除，「全部确认（灰框）」一次确认整张。
    - 候选框不写进标签、不参与训练，也不计入张数；点「✓ 已复核」时，没确认的候选会被丢掉。
  - **「自动标注用」可以选三种来源**：
    - 文字提示（原模型）；
    - **视觉样例（原模型）**：在复核里框好一张图 →「用这张图做视觉样例」，之后按“长得像样例”去找，适合鸡蛋这种外观单一、文字提示认不准的物体；
    - **精调：<模型名>**：先人工标 20–30 张 → 快速训练 → 用新模型自动标注剩下的图。
  - 「重新自动标注这张」/「重新标注所有未复核的图」：人画过、改过、确认过的框，以及这次没覆盖到的类别，都保留；只替换这次覆盖类别的自动框；已复核的图不动。
  - 采集页用自己的识别器，不会让 Camera 页的识别反复重新加载模型。
- **曝光**（「① 相机参数」，针对车库：平均亮度不到 110、过曝 2 % 以上）：
  - **镜面亮斑单独统计**：很小的过曝斑点（光滑蛋壳在硬光下的高光）算“镜面亮斑”，不计入过曝。判断过曝只看大片过曝，以及去掉亮斑后最亮 0.5 % 的亮度。界面显示亮斑有几处，多了会提示加柔光。
  - **平均亮度下限可以改**（默认 110，车库可填 80）。
  - **自动提高伽马**：增益到顶还不够亮、又没过曝时，提高伽马来提亮中间调（D435i 彩色相机支持就用）。结果会写进参数文件，连接时一起套用。
- **训练**：
  - 默认图像尺寸改为 **1280**，和 Camera 页识别时一致；用 640 时鸡蛋只剩约 20 像素。
  - 批大小：快速 4，完整精调 2，按 8 GB 显存估计。
  - 评估固定按 Camera 页的 1280 进行。
  - 模型说明卡（`model_card.json`）里保存同义提示词。
  - 「已保存的模型」只列出有效的模型（模型文件确实存在）。
- **有意改动，老测试已更新**：
  - test_v3_79 / 80 的采集用实验窗口自己的识别器；
  - test_v3_81 完整精调的批大小改为 2；
  - 版本白名单加上 V3.82；
  - V347 的模型列表会多出精调模型（没保存过精调模型时和以前一样）。
- **测试**：新增 `tests/test_v3_82.py`（R1–R8），用假的 ultralytics、假相机传感器、模拟窗口。覆盖：精调模型不设文字提示、未训练的物体交给原模型、实机任务（V366）也走固定类别；伽马提亮（模拟传感器）、亮斑不缩短曝光；候选框 / 双击确认 / 视觉样例 / 重新标注 / 复核时丢弃；应用和恢复模型。
- **没有在真实 D435i、YOLOE、GPU 上运行过**。视觉样例模式用的是 ultralytics 的视觉提示功能（Camera 页早已在用），但用它批量标注的效果、伽马的实际提亮幅度，都要你实测。

## V3.81 (2026-11-05) — YOLOE 精调：数据 → 一键训练 → 原模型 vs 新模型评估 → 保存；相机参数文件另存 / 打开
- **相机参数文件（① 相机参数）**：
  - 「另存为文件…」：存到你选的文件夹，文件名自己起。
  - 「打开文件…」：读回以前存的参数文件，V3.80 的参数组、`d435i_profile.json`、V3.78 的文件都能打开。打开后立即套用到相机；勾选了「设为连接时自动套用」的话，也设为 Camera 页连接时用的参数。
  - 记住上次用的文件夹。不是相机参数的文件会被拒绝。
- **新页「⑤ 训练 / 评估」**（新模块 `NERO_V381_Yolo_Train.py`），从上到下 4 步。前一步没完成，后一步的按钮是灰的。
  1. **选择训练数据**：
     - 列表列出数据文件夹里所有类别，每类显示已复核几张 / 共几张、训练 / 验证张数、框数，可多选。
     - 「只用已复核的图（推荐）」默认勾选。
     - 「检查数据」检查以下几项，结论分绿 / 橙 / 红三色：
       - 训练集或验证集没有图 → 不能训练；
       - 训练图少于 20 张 → 提醒；
       - 验证框少于 5 个 → 评估结论不可靠；
       - 还没复核的图有几张；
       - 有几个框属于没勾选的类别（训练时去掉，图里的那些物体会被当成背景）。
  2. **训练**：
     - 两种模式：「快速（只训分类头，推荐）」40 轮，「完整精调」80 轮；轮数、批大小、图像尺寸可以改。
     - 训练在独立进程里跑，进程环境用 `clean_child_env` 清理过，控制台不卡。
     - 进度条显示第几轮、已用时间、剩余时间、验证 mAP50。
     - 「■ 停止」：这一轮结束后停下，用目前最好的结果继续评估。
     - 关掉窗口训练也会在后台继续，重新打开时接着显示进度。
     - 训练集是数据文件夹里 `_runs/<时间_名称>/dataset` 的一份副本：图片用硬链接，不额外占空间；标签换成这次的类别编号；验证集保持不变，训练时不会见到。
  3. **评估**：用我们自己写的同一套评估代码，在同一份验证图上分别跑：
     - 原模型（用类别名做文字提示，和 Camera 页现在的用法一样）；
     - 新模型（用它自己训练好的类别，不再设置文字提示）。
     
     表格逐类给出“原 → 新”：识别率（召回）、每张误检、mAP50、轮廓重合度，最后一行是平均。识别率和误检都按 Camera 页的置信度算。结论有三档，带颜色：
     - **有实质提升**：识别率提高 ≥ 5 个百分点，或 mAP50 提高 ≥ 0.05；同时没有哪类识别率下降 2 点以上，误检没有明显变多；
     - **没有实质提升**；
     - **变差了**：不要保存，并给出改进建议。
     
     验证集太小时，结论会注明“只能参考”。另有「打开结果文件夹」，可看 ultralytics 生成的曲线图。
  4. **保存**：
     - 「✓ 保存这个模型」：存到 `~/Downloads/objectmodels/finetuned/<名称>/`，包括模型文件、`model_card.json`（类别、模式、日期、指标、结论）、评估结果、训练集说明和训练曲线。
     - 「✗ 丢弃这次训练」：删掉这次训练的临时文件夹。
     - **V3.81 只保存，还不会被 Camera 页或实机任务使用**：Camera 页的模型列表只认模型文件夹顶层的 `yoloe*.pt`。应用到实机放在 V3.82。
- **训练方式**：两种模式都用 ultralytics 的 `YOLOEPESegTrainer`（你电脑上 8.4.174 已确认有这个类）。
  - **快速模式** = ultralytics 文档里的线性探测：冻结主干和检测头，只训练各分类分支（`cv3`，以及 YOLO26 这种不需要 NMS 的模型里的 `one2one_cv3`）的最后一层。
  - **完整精调**：全部可训。
  - 优化器 AdamW，连续 15 轮没有提升就提前停止。
- **Camera 页（V325）、后端、仿真都没有改。**
- **测试**：新增 `tests/test_v3_81.py`（Q1–Q9）和假的 ultralytics `tests/v381_fake_yolo.py`。
  - 训练在**真实的子进程**里跑：进度、停止、重新打开窗口后接着显示进度都走通了。
  - 在合成的红 / 蓝方块图上，评估得出“有实质提升”（原模型认不出蓝块，还有一个误检）。
  - 还测了数据检查、导出、冻结列表、评估公式、保存和丢弃、界面按钮随步骤开关。
  - 测试时发现并修好一个问题：训练结束、但还没被回收的子进程，会一直被判断为“还在跑”。
- **还没在真实的 ultralytics / GPU 上训练过**。第一次训练请注意：
  - ultralytics 可能会联网下载一个检查混合精度用的小模型，电脑没网时会报错，把日志发我；
  - 训练用到文字编码器 `mobileclip2_b.ts`，要和原模型放在同一个文件夹（现在就是）；
  - 「停止」如果 90 秒内没停下来，会强制结束训练进程，这时没有评估结果；
  - 显存不够时把批大小减半。
- 老测试：版本号白名单加上 V3.81；test_v3_79 允许出现第 5 页。

## V3.80 (2026-11-04) — 相机高级测试 4 项改进：一键调参 + 参数组、噪声结论、任意物体定位精度、类别清单采集
- **① 相机参数**：
  - **「一键自动调参」**，约 10 s：
    1. 先自动白平衡 3 s，记下白平衡值；
    2. 在**工作区**内（画面上拖框圈出桌面，默认中央 60 %，避开窗户 / 屏幕）找参数：曝光只取灯光闪烁周期的整数倍（50 Hz 电网 = 10 / 20 / 30 ms，60 Hz = 8.3 / 16.7 / 25 ms），取不让亮部过曝（最亮 0.5 % 的像素 < 245）的最长曝光；
    3. 平均亮度不到 110 时，再用二分法找到达 110–150 的最低增益。
  - 结果直接显示合格或不合格及原因：
    - 偏暗且增益到顶 →“需要加灯”；
    - 灯太亮，曝光短于一个闪烁周期 →“可能有条纹”；
    - 白色物体和桌面反差太大、平均亮度上不去 → 判合格并注明“以白色物体不过曝为先”。
  - **参数组**：起名保存到 `cameracalib/d435i_profiles/<名字>.json`，下拉选择后「套用这一组」；勾选「设为连接时自动套用」后，Camera 页每次连接都会套用这一组。
  - **连接后检查光照**：工作区亮度和保存时相差超过 20 % 时，日志提示重新调参（`[V3.80 camera lab] 光照和保存参数时不同…`）。Camera 页文件只改了一处：把自己传给这个检查（V325 校验值随之更新，tests 61–80）。
- **② 深度噪声**：
  - 表格改用 Camera 页的表格样式，行高按字体计算，不再重叠；
  - 画面上的标签改成英文，原来的 `????` 是 OpenCV 画不了中文；
  - 测完给出**结论**：
    - **判定**：与这个距离下 D435i 的理论抖动比较（z²·0.08 / (f·B)，1.13 m 时约 3.0 mm）；
    - **平整度检查**：平整度远大于抖动时，提示测量框里的表面不平、有物体或反光；
    - **建议**：哪种处理最好、比 Camera 页现在的好多少、代价多少；
    - **距离换算**：换算到 750 mm 工作距离下的预计值。
  - 用你 V3.79 的那次数据：判定正常；滤波后取 5 帧中值比现在好 17 %，每次定位约多等 0.17 s。
- **③ 定位精度**：不限于鸡蛋。填物体名，选形状：
  - 「一般物体」用 Camera 页的通用定位（`locate_objects`），同时比较它量出的物体高度；
  - 「蛋形 / 球形」用椭球拟合，长、宽自己填。
  - 真值 = 角点正上方、物体高度的一半。
  - 测完给出一句结论：水平误差 ≤ 3 mm、最大误差 ≤ 6 mm、检出率 ≥ 90 % 时，判定“可以用于抓取”。
- **④ 采集标注**：
  - **不再默认鸡蛋**，改用**类别清单**：从常用物品里选，或自己输入，然后「加入清单」；可勾选 / 取消、移出清单；「当前采集类别」= 存进哪个子文件夹。
  - 每次拍照，清单里勾选的类别**全部**自动标注，同一张图里的其他物体也带标签，避免训练时被当成背景。
  - 清单里每类显示已有多少张、多少个框，以及离建议的 150 张还差多少。
  - V3.79 的设置文件打开时，丢掉其中的“egg”默认值，保留你自己加的类别。
- **有意改动，老测试已更新**：test_v3_79 改用类别清单；保存参数必须先起名；tests 21–24 的版本前缀白名单加上 V3.8。
- **测试**：新增 `tests/test_v3_80.py`（P1–P7），在模拟传感器上验证：
  - 一键调参的各种情况：正常光、50 / 60 Hz、反差大、太亮、太暗；
  - 参数组的保存和套用，连接后的光照检查；
  - 用你的真实噪声数据检验结论文字；
  - 任意物体定位；
  - 窗口里的类别清单和采集。
- **没有在真实 D435i / YOLOE / Tk 上运行过**。一键调参的步数和等待时间（每步等 0.35 s、取 3 帧平均）要实测确认。
- **开发文档**：AGENTS.md 加了「8.5 V3 逐步完善计划」和「8.6 V4 开发目标（SmolVLA）」，并记下了你的 22 项准备清单和每项的状态。训练和评估顺延到 V3.81。

## V3.79 (2026-11-04) — Camera 页「相机高级测试」窗口：相机参数 / 深度噪声 / 定位精度 / 采集 + 自动标注 + 框级复核
- **入口**：Camera 调试页 →「相机信息」→「保存相机信息」右边的「相机高级测试」。
  - 打开一个独立窗口，直接用 Camera 页已经连上的 D435i 画面，不再单独打开相机。
  - V3.78 的命令行工具保留，但不需要再用它。
- **① 相机参数**：
  - 「自动调 3 秒后锁定」，或手动填曝光 / 增益 / 白平衡，再选防闪烁 50 / 60 Hz →「保存并锁定」。
  - 画面下方实时显示平均亮度（目标 110–150）和过曝比例（目标 < 0.5 %）。
  - **锁定后，Camera 页每次连接相机都会自动套用这组参数**（新行为，日志 `[V3.79 camera lab] 已套用锁定的相机参数`）。深度预设仍按 Camera 页的选择；没有锁定时和以前完全一样。
- **② 深度噪声**：一键测 V3.78 的 5 种深度处理，结果显示在表格里，同时存 CSV。
- **③ 定位精度**：
  1. 不放鸡蛋时「锁定标定板」，画面上会标出角点编号；
  2. 把鸡蛋放在某个角点上，点「测量」；
  3. 表格给出检出率和 3D / 水平 / 高度误差。
  - 模型和置信度用 Camera 页「视觉定位」的设置。
- **④ 采集标注**：
  - 自己选数据文件夹，填物体名（任意物体，逗号分隔，第 1 个是子文件夹名）。
  - 「拍一张」或「连拍 N 张 / 间隔 s 秒」：每张都用 YOLOE 文字提示自动标出轮廓，存成 YOLO 分割格式；可选同时存深度图。
  - 按类别算，每 5 张放 1 张进验证集。
  - 复核在右边的大图上做：
    - 点框 = 选中；拖框内 = 移动；拖白色角 = 改大小；在空白处拖 = 新框；Del = 删除；
    - 下拉框可以改选中框的类别；「✓ 已复核，下一张」标记这张已经看过。
  - 框改过之后轮廓自动重算：先用这张图原来的 YOLOE 轮廓裁到框内（框和原检测框重合 ≥ 50 %），否则在框内用 GrabCut，再不行就用框本身。
  - 文件结构：
    - `<数据文件夹>/classes.json`：全局类别编号；
    - `<类别>/images|labels/{train,val}/`；
    - `<类别>/meta/*.json`：框、轮廓、置信度、是否复核、YOLOE 原始结果；
    - `<类别>/depth/*.png`：深度图，单位 mm。
  - V3.80 的训练直接读这份结构。
- **V325 改动**（Camera 页，因此 tests 61–78 里 V325 的校验值更新为新文件的哈希，这是有意改动）：
  - 加了按钮；
  - 相机线程加两个入口：`run_on_device`（相机参数只能在相机线程里设）、`collect_depth`（按指定滤波取深度帧）；
  - 连接时套用锁定的参数。
  - Camera 页原有的取帧、滤波、标定、定位代码没有改。
- **只写这些位置**：`NERO_CALIB_DIR`（参数、结果、窗口设置 `camera_lab.json`）和你选的数据文件夹，不写发布文件夹。
- **测试**：新增 `tests/test_v3_79.py`（N1–N10），用假相机线程、假 YOLOE 检测和模拟 Tk 窗口：
  - 参数套用；
  - 相机线程两个入口；
  - 轮廓重算（GrabCut 在合成图上 IoU 0.99）；
  - 数据集结构和标注格式；
  - 框编辑的各种操作（测试时修掉了一个“拖角改大小不生效”的错误）；
  - 噪声和精度汇总；
  - 窗口里的拍照 → 复核 → 保存。
- **没有在真实 D435i 和真实 YOLOE 上运行过**；这个环境也没有 tkinter，窗口的实际布局和鼠标操作要你在 Ubuntu 上看。

## V3.78 (2026-11-04) — D435i 单机工具（不需要机械臂）：固定相机参数、深度滤波对比、鸡蛋定位精度、训练图采集
- **背景**：V4 准备工作第四组“感知精度”的第 0、1 步。你在办公室只有 D435i，所以这一版做一个不需要机械臂、也不需要打开控制台的独立工具 `NERO_V378_D435i_Bench.py`，有四个模式：
  - `python3 NERO_V378_D435i_Bench.py profile`：固定相机参数。
    - 按 `a` 先让自动曝光、自动白平衡稳定 3 s，读出数值后锁定。
    - 也可以手动调：`e/E` 曝光、`g/G` 增益、`w/W` 白平衡，`f` 在 50 / 60 Hz 防闪烁之间切换。
    - 画面上显示平均亮度（目标 110–150）和过曝比例（目标 < 0.5 %）。
    - 按 `s` 保存到 `~/Downloads/cameracalib/d435i_profile.json`。之后每个模式启动时都会先套用它。
  - `noise`：相机对着一块平整、不反光的表面（桌面或卡纸），按 `m`。
    - 依次测 5 种深度处理：原始、空间+时间滤波（Camera 页现在的做法）、再加补洞、滤波后取 5 帧中值、原始取 10 帧中值。
    - 每种给出有效像素比例、逐像素的时间抖动（mm）、平整度（拟合平面后的残差，mm）、距离。
    - 结果存为 CSV。
  - `accuracy`：把打印的 7×5 ChArUco 标定板（就是腕部标定那张）平铺在桌上。
    1. 先不放鸡蛋，按 `b` 锁定标定板位姿。标定板中心就是坐标原点，z 轴朝上，桌面为 z = 0。
    2. 把鸡蛋横放在画面上标出的某个内角点上，按 `m`。
    3. 用 Camera 页同一套方法（YOLOE 文字提示 “egg” + 轮廓与局部桌面拟合求蛋中心）各测 10 次，分两种设置：“滤波 + 5 帧中值”和“只用 1 帧”。
    4. 给出检出率，以及三维误差、水平误差、高度误差的均值 / 标准差 / 最大值 / 90 分位。真值是角点正上方、高度为蛋宽一半（21 mm）的点；`t` 切换到竖放（27.5 mm）。
    5. 注意：误差里包含手放鸡蛋的位置误差，大约 ±1–2 mm。
  - `capture`：在保存的参数下采集训练图。空格存彩色图 JPG，`d` 切换是否同时存对齐的深度图（16 位 PNG，单位 mm）。
    - 每次采集一个文件夹，放在 `~/Downloads/objectmodels/datasets/<时间>/`，同时记下相机内参和当时的实际参数。
    - 参数没锁定时会提醒。
- **只写这些位置**：`~/Downloads/cameracalib/`（参数和结果，环境变量 `NERO_CALIB_DIR`）、`~/Downloads/objectmodels/datasets/`（图片，环境变量 `NERO_V378_DATA_DIR`），不写发布文件夹。
- **没有改**：Camera 调试页和后端都没改。
  - Camera 页启动时还不会自动套用这份参数，计划下一版加。
  - 在那之前，请在同一次 USB 连接里先跑 `profile`（或其他任一模式）再开 Camera 页。相机通常会保持设好的参数，但拔插或断电后会恢复默认，这一点要你实测确认。
- **测试**：新增 `tests/test_v3_78.py`（M1–M7），用假的 RealSense 设备和程序生成的标定板图片：
  - 参数设置：先关自动、再设数值，数值按设备范围夹紧和取整；
  - 噪声统计：10 帧中值把 1.5 mm 的抖动降到约 0.45 mm；
  - 标定板位姿：板上各点误差 ≤ 0.6 mm，倾角误差 0.25°；
  - 误差计算和结果保存。
- **没有在真实 D435i 上运行过**（这个环境没有相机，也没有 pyrealsense2），按键流程和数值都要你实测。

## V3.77 (2026-11-04) — 启动仿真前检查残余进程；分类场景夹物动作关节余量 ≥ 10°、执行路径与规划一致、安全停原因写进结果
- **1. 启动仿真前检查残余进程**（新模块 `NERO_V377_Stale_Procs.py`）：
  - 控制台点「启动仿真」时先扫描上次运行留下的进程，在弹窗里列出，你确认后关闭（INT → TERM → KILL，每步最多 2.5 s，连 `ros2 launch` / RViz 的整个进程组一起）。选“否”就不启动。日志 `[V3.77 STALE]`。
  - 能识别：`rviz2`、`move_group`、`ros2_control_node`、`spawner`、`robot_state_publisher`、`ros2 launch agx_arm_ctrl`（实机驱动）、`ros2 launch agx_arm_moveit`（也包括用 python 启动的），以及任何文件夹里的 `NERO_*.py` 脚本（例如从 `NERO_V3_74` 文件夹启动、没关掉的旧后端）、GPU 规划服务。
  - 不会动：控制台本身（`NERO_V312_Console.py`）、只是提到 NERO 文件名的编辑器 / `tail -f`、无关的 python。
  - 启动脚本的 V3.07 检查改用同一个模块，作为第二道检查。原来它只认 7 种进程。
- **2. 执行路径和规划一致**（你 V3.76 苹果那次失败的直接原因）：
  - 规划时，搬运段假设手腕边移动边逐渐转到目标角度；而 V3.75 加的“自己检查过的直线路径”一开始就用目标姿态。结果需要换 IK 分支而被拒绝，退到关节空间运动时 J6 卡在限位上（余量 0），报 `MOTION_FAILED`。
  - 现在这条路径和规划、MoveIt 的直线路径一样，手腕边走边转。
  - MoveIt 因为“关节路径检查不过”拒绝时也用这条路径（V3.75 只在“只有 MoveIt 夹爪模型碰撞”时用），前提是 MoveIt 确认机械臂连杆都不碰。
- **3. 夹着物体时关节余量 ≥ 10°**（`HELD_MARGIN_DEG`，环境变量 `NERO_V377_HELD_MARGIN_DEG`）：
  - 抬起、转腕、搬运、下放、松手的每个点，所有关节离限位都要 ≥ 10°。原来只要求 ≥ 4°，V3.76 实际用过 5–10° 的方案。
  - 原因：J6 只有 −42°～+54°，换成 −90° 安装后手腕常在 J6 限位附近；MuJoCo 跟踪误差最大约 5.6°；后端在关节超出限位 2.9° 时就判失控。
  - 找不到这样的方案时，返回 `GRASP_UNREACHABLE` / `PLACE_UNREACHABLE`，结果里写 `margin_rejects`，并说明最好的方案离限位多少度。
  - **代价**（替身）：默认摆放和两个扰动摆放仍是 4/4，还比 V3.76 快（每次约 90–100 s，V3.76 为 150–260 s）。扰动摆放 seed 1（香蕉偏 15 mm / 转 15°）里，香蕉的最好方案只有 8.4°，现在报 `GRASP_UNREACHABLE`，其余三件照常放好。`test_v3_72` D5 按此修改。
- **4. 安全停原因写进结果**：
  - 后端每次安全停都会在 `logs/diag/` 写 `diag_trip_<n>_<原因>.csv`（`runaway_j6` = 6 号关节超速或超限，`penetration` = 机器人碰进受保护的表面）。
  - 停下后读出最新的原因和当时离限位最近的关节，写进技能结果（`trip_reason` / `trip_meaning` / `closest_joint_limit`），日志 `[V3.77 SAFETY]`。V3.76 的日志只显示 `STATE 2`，原因要去翻 `backend.log`。叠杯子也会这样报。
  - 抓取前检查“之后能不能放进框”时多留 2°（按 12° 检查）。原因：抬起后的实际状态和刚性预测略有出入，测试里预测 ≥ 10°，实际只剩 9.9°。
  - 物体不属于的框可能根本放不进去：例如这个摆放下，没有哪个抓法能把香蕉顺着放进蓝框。这时 `pick_object` 会给出 `warning_place`。
- **老测试有意修改**：版本白名单加 V3.77；`test_v3_72` D5 seed 1 改为预期香蕉被 10° 规则拒绝；`test_v3_76` I6 的“安全停后掉进框”场景改用香蕉自己的红框（10° 规则下香蕉放不进蓝框）。
- **测试**：新增 `tests/test_v3_77.py`（K1–K8）。
- **只在替身和 MoveIt 模拟里验证过；完整仿真链路待复测，真机未验证**。

## V3.76 (2026-11-04) — sort_items：搬运防发散、后端削框壁规避、安全停后状态对齐、开始前退开；SO-101 标准方向 −90°（动指在上）；停止仿真限时
- **你 V3.75 仿真的问题和原因**（`backend.log`）：
  - 两个任务的物体都抓起来了，在**夹着物体转腕 / 搬运**时，MuJoCo 物理计算发散：关节 5 / 7 的速度从 6.2 rad/s 涨到几百、上千 rad/s（安全停 RUNAWAY）。
  - 安全恢复时后端保留了抓持（`RESTORE KEEP HOLD`），随后再次发散，才松开物体（`HOLD CONTINUITY released`）。所以物体是从空中掉进框的，看起来像“扔进去”。
  - 后端开头有一段为老鸡蛋篮写的冻结代码（`V2-44 R2 BASKET RIM`），把旧鸡蛋篮位置 ±14 cm 内的薄盒子削到 14 mm。红框的 x− 墙和 y+ 墙落在这个范围里，被从 50 mm 削到 14 mm。
- **1. 搬运防发散**（只改分类场景，后端不改）：
  - SO-101 指垫接触改软：solref 从 0.003 改为 0.006，solimp 0.95 0.99 0.001，与物体本身的接触参数一样。物体被后端的点约束固定在夹爪上，同时又被指垫夹着；两套约束一起作用时，指垫太硬会和点约束硬顶，可能是发散的原因。叠杯子场景的指垫不变。
  - 夹着物体时，升高、转腕、搬运、下放都放慢 3 倍（`HELD_SLOW`，V3.75 是 1.5–2 倍）。
  - **没有你的 `diag_trip_*.csv`，原因没有完全确认；替身里没有机械臂动力学，复现不了发散。这次修改是否见效，要看你的仿真复测。**
- **2. 框壁避开后端的削壁规则**：MuJoCo 里的框壁拆成每段半长 < 45 mm 的短段，外形、高度、厚度不变；MoveIt 仍用整面墙。测试按后端同样的判断条件逐个检查：V3.75 的整面墙正好命中红框 x− / y+（与你的日志一致），V3.76 的分段一个都不命中。仿真日志 `[V3.76 SCENE]` 会写出这一点。
- **3. 安全停后对齐状态**：
  - 技能“手里有没有东西”改为看夹爪状态和物体位置，不再看自己的记录。
  - 安全停把物体放掉、物体落进了要求的框：结果 `ok`，原因 `ITEM_DROPPED_IN_BOX`，并附一条警告“是掉进去的，不是放进去的”。
  - 落在别处：结果为 `ITEM_DROPPED`。
  - 物体已经不在夹爪里时，`return_to_safe_pose` 不再拒绝。
  - 打分：`place_in_box` 失败，但最后 `check_bins` 显示物体在 Agent 要求的框里，也算成功。
- **4. 开始任务前先退开**：TCP 低于 0.30 m 时，先保持手腕姿态竖直升高，升到直线能到的最高处、至少 5 cm（能用 V3.75 的“自己检查过的直线路径”），然后回到初始姿态，再交给后端做开始检查；如果从当前姿态一点都升不上去（例如某个关节靠近限位），就直接回初始姿态（后端由 MoveIt 规划，做碰撞检查）。你 V3.75 的第二个任务就是因为从框旁低处开始，被拒绝了。
- **5. SO-101 标准方向**（你纠正的）：SO-101 是上下开合，动指在上。
  - 你的两句描述换算到模型里正好相反：“从上往下看顺时针 90°”对应 +90°，“动指在上”对应 −90°。我用仿真的夹爪模型画了 A（−90°）/ B（+90°）对比图，你选了 **A** → 偏转默认 **−90°**（V3.75 为 0°，更早为 −45°）。
  - 选 A 的含义：工具水平朝前、J7 = 0 时动指在上、往上张开；等于 V3.68 真机验证过的旧 −45° 再转 45°。
  - 选 +90° 的话，J7 只有 ±90° 行程，叠杯子会规划不出来。
  - 实机页旧设置迁移：V3.75 存的 0° → −90°；更早存的 −45° → −90°；指尖修正清零。
- **5b. −90° 安装下的分类放置**（J7 只有 ±90°：替身里默认摆放和扰动摆放先后出现“放不下 / 香蕉在框里斜着、苹果没地方”，逐一修正）：
  - 放置规划只有一个（`_plan_place`）。抓取时就用它检查：按抬起后的预测状态，这个抓取角度之后能不能把物体放进框。香蕉要求顺着框放，苹果要求有空位。
  - 除了 15° 一档的角度，再加上“正好顺着框”的两个精确角度。香蕉偏 7.5° 就会占掉 17 cm 框宽里的 8.9 cm，苹果就放不下了。
  - 框里的候选位置排好顺序后逐个试，直到夹爪和物体都不碰框壁。靠墙放时固定指会碰到框壁。
  - 搬运那段直线也纳入规划。以前它会让 J6 撞到限位，然后在动作途中失败。
  - 同样顺着框，优先选手腕转得少的放置角度。原地转 168° 会让挂着的香蕉晃偏约 30°。
  - 抓长条物体优先垂直于长轴抓。斜着抓的话，香蕉搬运时会在夹爪里转动。
  - 苹果是圆的，占地不随手腕转。框里已有物体时，至少留 3 mm 空隙，不再叠放。
  - 空框时贴着一侧放，给下一个物体留出最宽的空位。
  - **工具契约改动**：`pick_object(item_id, box)`，box 取 red / blue / any。Agent 把已经决定的框告诉抓取规划；提示词同步修改。
  - 替身结果：默认摆放和 3 个 ±15 mm / ±15° 扰动摆放都是 4/4。**代价**：每次抓取都要规划后面的放置，替身里一次分类从约 30 s 变成 2.5–3.5 min，真实仿真里每次抓取可能多等几十秒（未实测）。
- **6. 停止仿真限时**：
  - 启动脚本对每个进程（组）依次发 INT → TERM → KILL，每步最多 2.5 s（`NERO_V376_STOP_STEP_S`）。范围包括两个 MoveIt 进程组（含 RViz）、后端（含它的子进程）、GPU 服务。整个停止约 8 s 以内，原来要 25 s 以上。
  - 控制台显示“正在停止…（最多再等 n s）”，结束时写出用了多少秒。
  - 清理时只等这四个停止任务。如果等全部子进程，会连日志进程（tee）一起等，启动失败时脚本会卡住；`test_v3_12` 发现了这个问题，已修正。
- **老测试有意修改**：
  - 版本白名单加 V3.76。
  - `test_v3_64`、`test_v3_65_gui`、`test_v3_67`、`test_v3_67_gui`、`test_v3_75` 里的偏转默认值改为 −90°。
  - `test_v3_67` 的相对转角改为 +60°。
  - `test_v3_75` 的迁移改为 −45° → −90°、−43° → −88°。
  - 测试替身 `tests/v372_backend_twin.py` 换用软指垫。
- **测试**：新增 `tests/test_v3_76.py`（I1–I11）。
- **只在替身和 MoveIt 模拟里验证过**；搬运发散是否消失、停止是否真在 8 s 内完成，要以你的仿真复测为准。**真机未验证**。

## V3.75 (2026-11-04) — sort_items：框缩小外移 + MoveIt 碰撞预检 / 碰撞对日志 + 替身碰撞检查；SO-101 新转接环（无 45° 偏转）
- **问题（V3.74 仿真首跑）**：
  - 框内尺寸 24 × 24 cm、中心 y = ±0.25，内壁在 |y| = 0.125，离物体只有 1–2 cm（“都贴到物体上了”）。
  - 抓香蕉“down to item_1”被 MoveIt 拒绝：直线下探只有 12.5 % / 19 % 无碰撞，改用关节目标后报 FAILURE (99999)。
  - 原因：MoveIt 的机器人模型带的是原装 AgileX 夹爪（`effector_type:=agx_gripper`），不是仿真里真正装的 SO-101。原装夹爪的手指比 SO-101 抓取点低约 19 mm，位置也不同；再加上旧的 45° 偏转，MoveIt 认为它碰到了框壁 / 桌面，而 SO-101 实际上没有碰。
  - 替身里没有机械臂和 MoveIt 的碰撞模型，所以离线测试没发现。
- **1. 框缩小外移**（`NERO_V343_Scenes.SORT_BINS`）：
  - 内尺寸 23 cm（x，深度）× 17 cm（y），壁高 5 cm，中心 y = ±0.28。
  - 内壁到每个物体 ≥ 7 cm（默认位置；打开 `NERO_V372_ITEM_SEED` 的 ±15 mm / ±15° 扰动时最近约 5 cm），外壁在桌面以内。
  - 原方案的 18 × 18 cm 装不下香蕉：香蕉首尾长 20 cm（弦长 16.7 cm，两端各有 17 mm 半径），所以深度改成 23 cm，香蕉顺着放。
  - 新增 `sort_item_footprint`：按物体实际外形算占地矩形（香蕉的弧形向一侧弯）。
- **放进框时检查放不放得下**（`SortSkills._slot` / `place_in_box`）：
  - 按转腕后物体长轴的朝向算占地，离内壁至少 8 mm，放不下的朝向直接跳过；长轴顺着框的长边的朝向优先。
  - 框里已有物体时，选离它们最远的位置；空框时贴着靠近机械臂的一侧放，给下一个物体留位置。
  - 夹着的物体在松手姿态下再和框壁做一次 3D 检查。
- **2. MoveIt 碰撞预检**（新模块 `NERO_V375_Collision_Check.py`）：
  - 抓取 / 放置的每个候选角度，在执行前检查关键姿态：预抓取、下探、抓取、抬起 / 框上方、下放、松手。
  - 检查两部分：
    - MoveIt `/check_state_validity`，取全部碰撞对（`NERO_V71` 原函数只返回第一对）；
    - SO-101 夹爪、转接环和 link5–7 的粗略盒子对桌面 / 框壁。
  - 有任何机械臂连杆（base_link、link1–7）的碰撞，或者 SO-101 自己碰到，这个角度就跳过。
  - 只有 MoveIt 的原装夹爪模型碰到时（碰撞对里有不属于 link0–7 的夹爪连杆），这个角度保留。执行时如果 MoveIt 拒绝了直线段，就逐点检查技能自己的直线路径：严格连续的 IK + `check_joint_path` + 碰撞对 + SO-101 盒子。全部通过才用后端的 `_make_incremental_joint_trajectory`（40 mm/s 重新分配时间）作为一条轨迹执行，日志 `[WARN] [V3.75 MOVEIT] … own straight path`。
  - `NERO_V375_SO101_OWN_PATH=0` 关闭这个绕行；这时这些角度会被预检拒绝，`pick_object` 报 `GRASP_UNREACHABLE`，并列出 `collision_rejects`。
  - 后端没改：执行仍走后端的控制器，MuJoCo 的安全层照常监视。
- **3. 失败时把碰撞对写进日志**：
  - `BackendIO` 在 MoveIt 拒绝直线段或关节规划失败时，打印 `[INFO] [V3.75 MOVEIT] <动作>: target state collides in MoveIt: A<->B …`，并标出哪些是“只有 MoveIt 夹爪模型”的碰撞、SO-101 盒子有没有碰。
  - 这个日志叠杯子也会打印。它只记录，不改变动作。
- **4. 替身补上碰撞检查**：
  - `SortTwinIO.state_check` 用同一套 SO-101 / 手腕盒子检查桌面和框壁。
  - 新测试辅助 `tests/v375_moveit_twin.py` 在假后端里模拟 MoveIt：用估计的原装夹爪外形，直线路径有碰撞点就拒绝（“only xx% collision-free”），关节目标碰撞就报 FAILURE (99999)。这样能在离线测试里重现你这次的失败。
- **5. SO-101 新转接环**：
  - 你说新转接环没有 45° 偏转了，夹爪按 SO-101 标准方式安装（动指在上，往上打开）。`NERO_V364.MOUNT["roll_deg"]` 默认改为 0（原来是 −45），仿真两个 SO-101 场景和实机任务都跟着变。
  - 环厚度没有新数据，仍按 7 mm，请实测后在实机页填写。
  - 实机页的设置是 V3.75 以前存的（文件里没有 `mount_rev`）时，打开时自动迁移一次：
    - 偏转 +45°（−45 → 0）；
    - 指尖修正清零，因为是在旧环上测的，请重新做 4 姿态碰点；
    - 旧值保留在 `pre_v375`，日志 `[INFO] [V3.75 REAL]` 写明改了什么。
- **老测试有意修改**：
  - 版本白名单加 V3.75，`SCENES_REV` 改为 V3.75。
  - `test_v3_64` / `test_v3_65_gui` / `test_v3_67` / `test_v3_67_gui` 里写死的 −45° 改为新默认 0°（`test_v3_67` Y2 的相对转角随之改为 −30°）。
- **测试**：新增 `tests/test_v3_75.py`（G1–G9）。
- **只在替身和 MoveIt 模拟里验证过**；模拟用的原装夹爪外形是估计值，真正的碰撞对以下次仿真日志 `[V3.75 MOVEIT]` 为准。**完整仿真链路待复测，真机未验证**。

## V3.74 (2026-11-04) — 实机杯子识别：几何只测位置和尺寸，“是什么”由看图的 LLM 复核；桌面偏差提示
- **问题（V3.72 实机）**：两只同款杯子 YOLOE 都认出来了（置信度 0.55–0.9），但几何把远处那只量成 96–104 mm，低于按仿真模型定的单杯下限 104.6 mm，被判成“unknown”（监视器上的“cup_1 unknown / 不像杯子 104mm”），任务报 `CUP_NOT_PERCEIVED`。
- **几何只测量、不下结论**（用户要求）：`NERO_V366` 不再按杯高判断单杯 / 已叠好 / 不像杯子；每个站在桌上的 YOLOE 杯子实例都是“杯子”，杯高、位置、倾斜照常测量。叠没叠好、是不是杯子、在不在夹爪里，由看图复核回答；没有复核时，放下后用 V3.71 的“目标位置单簇 + 变高”判断叠好。`classify_by_height=True` 保留 V3.66 的高度分档（测试 / 对照用）。
- **LLM 看图复核**（新模块 `NERO_V374_Vision_Review.py`，实机任务页「LLM 看图复核」，默认开）：在关键识别时刻（observe / 识别杯子 / 重新识别 / 检查 / 验证；下探、搬运过程中不做）把主相机画面（每个检测画出轮廓和 #编号）连同几何量到的数字发给实机任务页选的模型（OpenAI：Responses `input_image`；DeepSeek / Kimi：chat `image_url`），让它对每个编号回答：单杯 / 已叠好的一对 / 不是杯子 / 不确定、是否直立、是否在夹爪里。按回答设定类别，几何数值保留；和几何不一致的地方逐条写进 `[V3.74 VISION]` 日志。桌面布局没变（数量、位置 1.5 cm 内）时 90 s 内不重复提问。没有 key / 模型不支持图片 / 网络出错：只 WARN 一次（原因 + 下一步），本次用几何结果。图片只发给你选的服务商，本模块不保存。
- **“在夹爪里”需要夹爪真的下到杯子高度**：V3.72 实测机械臂只是停在杯子旁边，桌上的杯子被判成“在夹爪里”。
- **“单独时杯高”只在手里没有杯子时记录**：搬着的杯子悬在目标杯上方会挡住目标杯杯口，量得偏矮（替身 109.5 mm，实际 125 mm），会让放下后的“变高”判断出错。
- **桌面偏差提示**：识别时拟合的桌面和安全区设定的桌面差 > 5 mm 时 WARN 一次（手眼标定 / 安全区桌面高度可能偏了，抓取高度会偏同样多）；> 10 mm 时 ② / ③ 的确认框里也写出来。
- **老测试有意修改**：版本白名单加 V3.74；`test_v3_66` 的“叠好的一对显示为 stack”改为用 `classify_by_height=True` 对照、最后一帧改为检查“只剩一个杯子实例且高于 140 mm”（默认已不按高度分类）；`test_v3_69` B6 的替身夹爪从杯口上方 6 cm 移到杯口高度（“在夹爪里”现在要求夹爪下到杯子高度）。监视器上站着的杯子标签由「单杯」改为「杯子」（`test_v3_66` X7 同步）。
- **测试**：新增 `tests/test_v3_74.py`（G1–G8，用你这次的数字 98 / 117 mm 重现并验证）。
- **看图复核只用假模型验证过请求格式和处理逻辑，没有连真实 API；只在替身上验证过，真机未验证**。

## V3.73 (2026-11-04) — sort_items 仿真首跑失败修复（TF 帧名）+ 测试替身按后端真实规则查 TF + 得分按 Agent 处理的物体
- **失败原因**：后端查物体位姿时会在名字前加 `sim_obj_`（`NERO_V71_Hybrid_GPU_Planner_Core.object_pose` → `sim_obj_{name}`），场景发布的帧是 `sim_obj_banana`；V3.72 的 `SortBackendIO` 传了带前缀的 `obj_banana`，查的是不存在的 `sim_obj_obj_banana`，`observe_scene` 两次失败，任务停止。你用 `tf2_echo base_link sim_obj_banana` 确认了正确帧名（香蕉在 (-0.310, -0.085, 0.035)）。
- **修复**：`SortBackendIO.object_pose` 先去掉 `obj_` 前缀（`tf_name("obj_banana") = "banana"`）。
- **测试漏洞修复**：`tests/v372_backend_twin.py` 的假后端原来直接按物体全名查，没有模拟后端加前缀的规则，所以替身里 4/4 而真实仿真失败。现在它严格按后端规则拼 `sim_obj_<名字>`、再按场景的 TF 表找物体，找不到就抛和后端一样的 TF 错误；`test_v3_73` E2 先用 V3.72 的名字重现你的失败，再验证修复。
- **日志**：分类任务不再打印叠杯子的 `[V3.64 CUP] so101: release gap capped ...`。
- **得分**：s = Agent 处理过（调用过 pick_object）的物体中、最后留在它所选框里的比例。V3.72 固定除以 4，只分水果的任务最高只能 0.5。
- **测试**：新增 `tests/test_v3_73.py`（E1–E5）；老测试版本白名单加 V3.73。
- **只在替身里验证过，完整仿真链路待你复测**（同样的任务再跑一次，`observe_scene` 应返回 4 个物体）。

## V3.72 (2026-11-03) — 第 5 个仿真场景 sort_items：物品分类放入红框 / 蓝框（SO-101 夹爪 + LLM Agent 按提示词分类）
- **场景**（`NERO_V343_Scenes.py`，控制台场景下拉框选 `sort_items`）：桌上一根香蕉、一个苹果、一个电脑鼠标、一支粗马克笔；机械臂左侧（y < 0）一个红色敞口框、右侧一个蓝色敞口框（内尺寸 24 × 24 cm、壁高 6 cm）。夹爪固定为 SO-101（与 stack_cups_so101 同一套模型和安装参数）。框壁 / 框底进 MoveIt 碰撞体，并和保险箱墙一样受保护（机械臂碰到即触发安全停）。`NERO_V372_ITEM_SEED` 可给物体位置加 ±15 mm / ±15° 扰动。
- **Agent 技能**（新模块 `NERO_V372_Sort_Skills.py`，复用叠杯子技能的运动 / 安全 / 夹爪档案代码）：
  - `observe_scene`：每个物体的编号（item_1…4）、识别标签（banana / apple / computer mouse / marker pen）、中心、尺寸、长轴方向、在哪个框里；两个框的颜色、左右、内容。**不给类别**——类别由 LLM 按用户的提示词判断（例如“水果放红框，其他放蓝框”）。
  - `pick_object(item_id)`：俯抓，两指连线垂直于物体长轴（圆的物体任意方向），自动选可达且张开的爪子不碰到别的物体的角度；固定指贴着物体侧面下探、夹紧、竖直抬起；以“两指接触 + 物体跟着升起 ≥ 3 cm”判定。
  - `place_in_box(red|blue)`：先原地转腕（不可达就带着姿态直接搬），搬到框里空着的位置上方，竖直放到离框底 15 mm，松开，竖直退回；以“物体中心在该框内”判定。
  - `check_bins`（只读）、`return_to_safe_pose`。
  - LLM 规则：先观察、写出分配表、逐个抓放、失败只重试一次、类别不清楚的物体留在桌上并说明、最后 check_bins。
  - 得分 `[V3.72 SKILL SCORE]`：s = 放进 Agent 所选框并留在框里的物体数 / 4；`[V3.72 SORT RESULT]` 列出每个物体的标签、仿真里的真实类别和最后所在位置，方便你判断分类对不对。
- **后端**：只在 RUN TASK 的 A2 钩子段里、叠杯子分支后面加了一个 `sort_items` 分支（调用 `NERO_V372_Sort_Skills.gui_run_task`）；`test_v3_44`（B 段 / A2 块）和 `test_v3_62` P1（去掉钩子段后与 V3.60 一致）仍通过。
- **仿真建模说明（为什么物体里有配重）**：后端抓住物体后用一个锁定点（球铰）把物体挂在夹爪上，而后端的“指垫压入过深就松开”保护让夹紧力只有约 0.5 N，物体会绕锁定点转到重心朝下。第一版测试里香蕉提起后绕长轴转了约 70°、鼠标被倒扣着放进框。和叠杯子（V3.62 杯口原点）同样处理：物体原点放在顶面中心，内部加一个不参与碰撞的配重，使重心在抓取点正下方。另外后端在两指都接触前最多推 0.6 N，推不动 90–180 g 的物体去贴固定指，所以固定指下探时直接贴着物体侧面（0.3 mm）。
- **只支持仿真真值位姿**：选了固定相机作为位姿来源时 RUN TASK 会拒绝并说明（这个场景的相机识别还没做）。
- **小改动**：`NERO_V362_Cup_Skills.TwinIO / twin_xml` 加可选参数 `world_xml` / `hold_names`（默认 = 原来的两只杯子，叠杯子替身不变）；控制台场景列表、`start_nero_sim.sh`、`start_nero.sh` 加入新场景 / 新模块。
- **老测试有意修改**：版本白名单加 V3.72；后端哈希更新（只多了 sort_items 钩子）；`test_v3_61` / `test_v3_64` 的场景列表断言允许末尾新增 `sort_items`；`test_v3_64` 的 `SCENES_REV` 允许 V3.72。
- **测试**：新增 `tests/test_v3_72.py`（D1–D7）和 `tests/v372_backend_twin.py`（假后端 + 按标签分配的脚本化“模型”，走真实的 Agent 循环和后端适配层）。替身里默认摆放 + 5 组扰动：24/24 件放进正确的框，21/24 件放下后保持原姿态（其余 3 件在框里斜靠 / 侧立）。
- **只在无界面替身里验证过（替身只有夹爪、没有机械臂本体的碰撞），完整仿真链路（MuJoCo 后端 + MoveIt + 真实 LLM）待你复测；不涉及真机。**

## V3.71 (2026-11-03) — 实机叠杯子：运动真到位才算结束、速度跟硬件走、失败后收尾、放置前再看夹着的杯子、叠好判断改用实测、监视器实时同步
- **运动“真到位”才算结束（V3.70 主要问题）**：V3.69 的速度档只按设定速度给轨迹定时间，轨迹时间一到、停 0.15 s 就读位置；4X 时真实机械臂被驱动的 `speed_percent` 限速、还差约 80°，程序已判失败并开始下一步（`TCP 306 / 457 mm from the goal`，随后 `CONTROL_FAILED`）。现在：每段运动开始前等机械臂静止；执行后一直等到 TCP 离目标 ≤ 6 mm 且静止（超时 = max(5 s, 3 × 计划时间 + 2 s)，≤ 60 s），超时报 `ARM_NOT_AT_GOAL`。
- **速度跟硬件走**：一段运动实际用时 > 计划 1.3 倍 + 0.5 s 时，按实测速度把之后的关节速度降下来，日志 `[WARN] [V3.71 REAL] 机械臂实际比设定慢 … 按实测关节速度 x °/s 定时间`；要更快请先调高实机参数 `speed_percent`。
- **失败后收尾**（实机档案 `recover_after_motion_failure`）：运动失败 / 不可达后先等停稳，再竖直升到 0.22 m 安全高度，再交给 LLM。
- **围栏绕行**：关节直线路径被围栏拒绝（例：离墙 69 mm < 70 mm）时，先竖直升到安全高度，再从那里走关节路径；还不行才报错。
- **识别漏检重试**（`perception_retries=2`）：一只杯子这一帧没认出来（置信度贴着门槛）就再识别，最多 2 次，不再直接 `CUP_NOT_PERCEIVED`。
- **规划提速**：三候选里只要有一个余量 ≥ 8° 就立即采用，不再把 15 s 预算用完。
- **放置前再看夹着的杯子**（`recheck_held_before_place`）：搬到目标杯上方停稳后，主相机测夹着的杯子（`NERO_V366.held_geometry`：杯底边定高度、下半段杯壁拟合锥面杯轴），与“夹取时的偏移”比较，差值 ≤ 25 mm 就修正偏移、水平微调对准，再测一次，然后才往下套；> 25 mm 认为测量不可信，不修正并警告。测到的倾斜只写日志。**说明**：原计划“降到松手高度后再看一次”没做——那时夹着的杯子下半截已进到目标杯里，两只杯子在画面里叠在一起测不准，所以改为在上方测两次。
- **叠好判断改用实测**（`stack_by_cluster`）：放下后，目标位置只剩一个杯子簇，并且比目标杯单独时（任务开始前实测）高 4–40 mm（填了实测套叠增量时为 ≤ 增量 + 8 mm），桌上没有别的单杯 → 已叠好。V3.70 两只真杯子量得 107 / 116 mm，比模型 122.6 mm 低，套好后会落进“单杯”一档被判失败。实机任务页「实机夹爪几何」新增「实测套叠增量 mm」（两只套好总高 − 单杯高；空 = 未测，此时不判断是否完全套到底）。
- **监视器实时同步**：任务中每次识别（observe / pick / place / verify / 复查）都立刻刷新主相机监视器的 YOLOE 轮廓、标签和杯子名；左下角显示「识别于 x s 前」，超过 4 s 的轮廓画成虚线并标「旧」。
- **GPU**：`start_nero.sh` 启动控制台前清掉继承的 `CUDA_VISIBLE_DEVICES`（和仿真启动脚本一样）；YOLOE 用 CPU 时日志写明环境变量、torch 的 CUDA 版本、`nvidia-smi` 输出和下一步。
- **测试替身**：`tests/v365_real_twin.py` 的 YOLOE 替身只在两只杯子杯口高度差 < 6 cm（真正套着）时才合成一个实例（夹在上方的杯子是单独一个实例，与真 YOLOE 一致）。
- **老测试有意修改**：版本白名单加 V3.71。
- **测试**：新增 `tests/test_v3_71.py`（C1–C10）。
- **只在替身 / 模拟机械臂上验证过，真机未验证**。

## V3.70 (2026-11-02) — 自动采集闪退修复、崩溃记录、实机日志进 console.log、示教负载（SO-101 较轻 → 重力补偿）
- **自动采集闪退（根因）**：Camera调试页「退出示教模式」在后台线程里把安全区编辑器从示教切回编辑并重绘画布（Tk / X11 操作），安全区页处于示教状态时会在 libX11 里段错误，整个控制台连同终端一起退出（dmesg：`python3 segfault … libX11.so`，与 console.log 最后一行同一秒）。现在这一步放回 Tk 主线程执行（`self._ui`），后台线程只负责和机械臂通信。控制台底部快捷「退出示教模式」本来就在主线程，不变。
- **崩溃记录**：控制台启动时打开 `faulthandler`，写到 `logs/crash.log`（追加，不清空）：再发生段错误时，所有线程的 Python 调用栈（文件 + 行号）会写进去；后台线程 / Tk 回调里没被捕获的异常也写进去。每次启动先写一行 `console start …`，下面没有内容 = 没有崩溃。
- **实机日志进 console.log**：实机NERO、Camera调试、技能参数旋钮、NERO实机任务页的 `[V3.xx …]` 日志行同时追加到 `logs/console.log`（行首带 `[时间 REAL]`），与子进程输出在同一个文件里。
- **示教负载（假设：SO-101 比原夹爪轻，控制器重力补偿按较重末端计算，松手后关节被拉回）**：实机参数里新增「示教负载」（empty / half / full / 不设置，默认 empty）。进入示教前，如果当前 pyAgxArm 提供负载接口（`set_payload` 等），先发送负载档，再 `set_leader_mode`；日志 `[OK] [V3.70 teach] 示教负载 = empty …`。**已查的 pyAgxArm（d90e234）NERO 驱动没有负载接口**（只有 Piper 有 `set_payload`），这时日志给 `[WARN] [V3.70 teach] …` 并写明下一步（看「模式接口」一行是否有 load / payload 字样的接口、用 AgileX 上位机设末端负载、或临时加配重验证假设）；示教照常进入。「模式接口」诊断行现在也列出 load / payload / gravity / tcp / param / config 相关接口，方便确认你电脑上的 SDK 有没有。不发送未公开的 CAN 帧。
- **老测试有意修改**：版本白名单加 V3.70；`test_v3_61`–`test_v3_69` 里固定的 `NERO_V325_Camera.py` 哈希更新为 V3.70（只改了上面的主线程修复）。
- **测试**：新增 `tests/test_v3_70.py`（A1–A6：主线程切换、crash.log 记录子进程段错误位置、console.log 镜像、负载接口有 / 无两种 SDK、参数一路传到 `--payload`）。
- **只在替身 / 模拟 SDK 上验证过，真机未验证**（示教回位是否由负载引起需要你在实机上确认）。

## V3.69 (2026-11-02) — 实机叠杯子第一次成功后的加固：验证前让开视线、以相机判断夹住、规划取最优、YOLOE 用 GPU、运行速度档、任务录像
- **背景**：V3.68 真机第一次完整套好（cup_1 → cup_2），但 `verify_stack` 因夹爪还挡在杯口上方被误判 NOT_STACKED；夹住时舵机停在 2.62 mm、空夹 2.39 mm，只差 0.23 mm；规划用了 4.9° 的全角度方案（窄范围已有 6.1°），耗时 28 s；YOLOE 跑在 CPU（每帧约 1 s）。
- **验证前让开视线**（共用技能，实机档案 `clear_view_before_verify`）：`verify_stack` 先让机械臂回到任务开始时的姿态再看；仍被挡住就再竖直升 100 mm 再看一次。日志 `[V3.69 REAL] 验证前让开视线`。
- **以相机为准判断夹住**（`camera_decides_grip`）：抬起后主相机看到杯子离开桌面就算夹住，舵机角度判为没夹住时只给警告（薄杯壁分不清）。
- **规划取最优**（`_plan_best_of`）：上次成功角度、窄范围、（余量 < 8° 时）全角度三种候选共用一个从开始计时的 15 s 预算，取关节余量最大的；日志列出每个候选的余量和选中的那个；到点即停。
- **YOLOE 设备**：实机任务自己的检测器在 CUDA 可用时强制用 `cuda:0`，日志写明 GPU 型号；不可用时写明原因（torch 版本 / `is_available()`）；每次识别日志末尾带设备。
- **“在夹爪里”的判断**：杯子最低点离开桌面之外，还要求杯子在夹爪旁 12 cm 以内；否则不算被夹起（避免把画面边缘 / 标定板旁杯底缺深度的杯子误判）。
- **运行速度档**（「▶ 运行」左边）：1X / 2X / 3X / 4X / 5X / 10X。关节运动速度按倍数放大，上限 72 °/s；杯子附近的直线段（下探 / 抬起 / 放下）上限 60 mm/s；停稳等待和悬停停留按倍数缩短（不少于 0.15 s / 1 s）。确认框显示实际速度，≥ 3X 提醒先在 ② 用同样速度试。
- **任务录像**（监控框下「任务录像」）：运行时以 5 fps 录主相机 + 腕部相机，叠加 YOLOE 轮廓，顶部标注时间戳、已用时间、档位、当前步骤 / 工具 / 状态 / 结果、阶段、手里的杯子，存为任务记录文件夹里的 `task_video.mp4`。
- **任务开始会自动识别杯子**（observe_scene 是第一步），不需要先手动点「识别杯子」。
- **老测试有意修改**：版本白名单加 V3.69；`test_v3_68` Z2 “夹爪没报夹住但杯子抬起来了”现在算成功（以相机为准），规划预算 15 s，Z3 接受 V3.69 的规划日志格式。
- **测试**：新增 `tests/test_v3_69.py`（B1–B8）、`tests/test_v3_69_gui.py`。
- **只在替身和模拟舵机上验证过，真机未验证**（GPU 是否生效以你电脑上的日志为准）。

## V3.68 (2026-11-02) — 实机叠杯子：抓了再判断、规划和识别提速、指尖对照校偏转角、任务截图记录 + 结束总结
- **“伸进杯子却不夹”的 bug**：V3.67 下探后用“下探前那次识别的倾斜绝对值 > 6°”判失败（夹爪挡住杯子时沿用旧值），真机把杯子测成 6–10° 时每次都“伸进去 → 不夹 → 提起”，还误报成“推动”。
- **放宽实机抓取判断（按用户要求：抓了再判断）**：实机下探后一律夹紧、抬起；夹之前只有相机**这次真的看到**杯子被推开 > 30 mm 或倾斜变化 > 25° 才放弃。抬起后两条都满足才算抓住：① 夹爪停在比空夹碰合点大的位置 ② 主相机看到杯子离开了桌面；否则松开、撤回，报 `NO_TWO_FINGER_CONTACT` / `CUP_DID_NOT_FOLLOW_LIFT` 并写明原因。杯口上方 10 mm 的复查阈值也放宽到 30 mm（只写日志）。（共用技能里由夹爪档案 `lenient_pick` 打开，仿真不变。）
- **规划提速**：记住上一次成功的抓 / 放杯沿角，下一次从那里开始（替身 2.8 s → 0.0 s）；全角度搜索改 30° 步长 + 10 s 时间预算（替身 3.7 s，原来真机约 50 s）。仅实机档案。
- **识别提速 + 计时**：实机任务用自己的 YOLOE 检测器实例（与 Camera 页同一个类、同一模型 / 置信度 / 提示词规则；避免 Camera 页换提示词后每次重新加载）；去掉开始时多取的一帧；锥面拟合最多 1500 点；每次识别日志末尾给出各步耗时（帧 / YOLOE / 融合 / 3D / 杯）；识别快照只在任务失败时保存。
- **看图校偏转角**：主相机监控框画出模型的固定指（F）、动指（M）指尖和两指连线（随机械臂当前位姿和「偏转 °」等参数实时更新），对照画面里真实的指头调「偏转 °」直到重合。
- **偏转角扫描**（只算不动）：识别当前两只杯子，对偏转 −90…+90°（15° 一档）各规划一次，日志给出每个角度能否规划、最小关节余量和受限关节。
- **任务记录**（新模块 `NERO_V368_Real_Task_Record.py`）：每次运行（① ② ③，含 LLM）在项目文件夹的 `task/<时间>_<档位>_<任务>/` 下保存每个工具调用（LLM 的每个决策 / 固定流程每一步）**执行前和执行后**的截图（主相机 + 腕部相机，YOLOE 轮廓，顶部标注步骤、工具、参数、结果、时间戳）、`steps.jsonl`、`task.json`、`summary.txt`；只保留最近 100 个任务。`task/` 是运行时输出（与 `logs/` 一样），已加入 `.gitignore`，不进发布包 / 校验和；可用环境变量 `NERO_V368_TASK_DIR` 改位置。
- **日志**：每个工具调用一行 `[V3.68 STEP n] 工具(参数) → 结果 | 耗时 | 关键数值`（转腕角、复查偏差、下探后位移 / 倾斜变化、夹紧停住的缝宽与是否夹住、抬起 TCP / 杯子上升量、规划杯沿角与余量…）；任务结束输出鸡蛋任务风格的 `[REAL TASK SUMMARY GREEN|YELLOW|RED]`（STATUS / TASK / SETUP / CUPS / OUTCOME / TOOLS 逐条 / LAST FAILURE / TIME（识别次数与平均耗时、运动段数与时间）/ RECORD）。③ 执行前若没做空夹标定会提示。
- **老测试有意修改**：版本白名单加 V3.68；GUI 测试把任务记录目录指到临时目录。
- **测试**：新增 `tests/test_v3_68.py`（Z1–Z7）、`tests/test_v3_68_gui.py`。
- **只在替身和模拟舵机上验证过，真机未验证**。

## V3.67 (2026-11-02) — 实机叠杯子：两指连线沿杯子半径（先对准、原地转腕、再下探）+ 更安全的下探 + 感知更稳 + 空夹标定
- **背景（V3.66 实机）**：① ② 成功；③ 下探时把 cup_1 推开 18.5 mm（相机 / 手眼 / 夹爪安装误差远大于仿真里固定指离杯壁的 2 mm），程序按设计不夹、抬起；LLM 一次因规划余量 7.3° 的警告自行停止，一次因 cup_2 被误测倾斜 42.5° 被拒；插进杯子后不夹是因为夹爪挡住杯子后倾斜 / 位置测量失真，被当成“推动了杯子”。这一版不用腕部相机。
- **(a) 两指连线沿杯子半径 · 先对准、原地转腕、再下探**（`NERO_V362_Cup_Skills.py` `_align_rotate`，只由实机夹爪档案打开）：先到预抓取点正上方 60 mm（保持机械臂原来的手腕朝向）→ 原地转到抓取角度（两指连线沿杯子半径，固定指在杯外）→ 竖直降到预抓取点 → 沿杯壁下探。② 悬停预演同样走这一步，便于先看角度对不对。
- **两段下探**：指尖到杯口上方 10 mm 停一下，主相机复查杯子；杯子估计位置偏离规划超过 6 mm 就不下探（`CUP_MOVED_BEFORE_CONTACT`，撤回后重新观察、规划）。
- **实机安全余量**：固定指离杯壁 2 mm → 8 mm（界面可改 2–15 mm）；只改实机档案，仿真仍 2 mm。
- **夹爪实测安装**：界面可填安装座面 / 转接环 / 偏转角；「读取碰点标定（固定指尖）」用 Camera 页 4 姿态碰点标定的结果（用 SO-101 固定指尖去碰）求出指尖修正。抓取坐标系、相机里扣除夹爪的范围都按这些实测值计算。
- **感知**（`NERO_V366_Real_Cup_Yolo.py`）：倾斜角只有在两半样本结果一致、残差小、点数够时才采用，否则按直立处理并在日志写明；夹爪把杯口挡住时只给位置（类型 `occluded`），不再据此判“被推动 / 倾斜”；识别不清或任务失败时自动保存当次彩色图、深度图和识别结果到 `NERO_CALIB_DIR/real_cup_logs/`（只保留最近 40 组）。
- **规划余量**：4–8° 的规划结果明确写“EXECUTABLE … continue with pick_cup”，并带 `executable=true`，Agent 规则 6c 同步；实机余量 < 8° 时自动再做一次全角度搜索。（这两项是共用技能代码：仿真 Agent 看到的文字也变了，仿真规划不变。）
- **空夹标定**（`NERO_V365_STS3215_Gripper.py`）：夹爪里不放东西点「空夹标定」，记录指尖真正碰合的角度（实测约 +1.85°）；之后所有张开角度从这个点算起，空夹不再被判成“夹住了”，判据改为停住时比碰合点多 ≥ 0.3 mm。
- **老测试有意修改**：版本白名单加 V3.67；`tests/v365_real_twin.py` 记录每段路径的终点关节；`test_v3_65` W7 悬停预演的第一段运动现在是“对准上方”（V3.67 先对准再转腕）。
- **测试**：新增 `tests/test_v3_67.py`（Y1–Y7）、`tests/test_v3_67_gui.py`。
- **只在替身和模拟舵机上验证过，真机未验证**。

## V3.66 (2026-11-02) — NERO实机任务的杯子识别改用 Camera 调试页的 YOLOE 方案（同一套代码）
- **为什么**：V3.65 实机第一次运行时，实机任务用的是 V3.63 为仿真写的“红色分割 + 深度最高点”方法：杯子其实找到了，但 D435i 在细薄反光的杯口丢深度，量出的杯高只有 99–108 mm（应为 121 mm），全部判成 unknown → `CUP_NOT_PERCEIVED`；而 Camera调试页输入 “cup” 用 YOLOE 可以识别。实机任务不应有第二套识别方法。
- **同一条流水线**（新模块 `NERO_V366_Real_Cup_Yolo.py`）：取帧 = Camera 页 `_grab_frames(FUSE_FRAMES)`；2D = Camera 页的同一个 YOLOE 检测器对象（同一模型文件、模型文件夹、置信度门槛、GPU/CPU，文字提示 / -pf 词库的选择规则与 Camera 页「物体识别」完全一致）；深度融合 = `YP.fuse_depth`；3D = `YP.locate_objects`（局部桌面平面、围栏、标定提示）。实机任务页只多一个「杯子识别 YOLOE 提示词」（默认 `cup`，存 `real_cup_task.json`）。
- **杯子几何（在 YOLOE 实例轮廓上做）**：杯轴 = 轮廓内深度点对已知锥面拟合（V3.63 `fit_axis`）；**杯高用轮廓顶边量**（杯轴上不同高度的杯口圆投影到图像，与 YOLOE 轮廓顶边对齐的高度即杯高），不再依赖杯口深度；单杯 / 已叠好（高约 25 mm）/ 不像杯子按这个高度判断；离开桌面（在夹爪里）、围栏外、被画面上边缘切掉的不当作桌上的杯子；同一只杯子的多个 YOLOE 框合并。
- **不退回颜色法**：YOLOE 不可用（没有 ultralytics、找不到模型、缺文字编码器、Camera 页没有检测器）→ 运行前检查直接拒绝并提示「YOLOE 不可用：原因」；运行中出现则日志 `[FAIL] [V3.66 REAL CUP YOLOE] YOLOE 不可用 … → 任务终止`，技能返回 `PERCEPTION_UNAVAILABLE`。
- **主相机监控框**：画出每个 YOLOE 轮廓和标签（编号、单杯 / 已叠好 / 不像杯子 / 在夹爪里 / 围栏外、轮廓量得的杯高、置信度），同时保留技能的杯名和抓取 / 松手点。日志每次识别打印轮廓杯高和深度杯高，便于对比。
- **未改**：Camera 调试页（`NERO_V325`）、YOLOE 模块的函数（`NERO_V347` 只改版本号）、V3.63 颜色感知（仿真仍在用）、杯子技能、场景、夹爪模块、后端、Agent。
- **老测试有意修改**：版本白名单加 V3.66；`test_v3_65` W7 接受 V3.66 的识别日志；`tests/v365_real_twin.py` 的替身相机改为 YOLOE 替身（MuJoCo 分割图给出每只杯子的实例轮廓，套好的一对是一个轮廓）；`test_v3_65_gui` 的假 Camera 页带 YOLOE 检测器。
- **测试**：新增 `tests/test_v3_66.py`（X1–X7：设置与 Camera 页一致、调用顺序与参数与 Camera 页一致、替身上杯轴 ≤ 3 mm / 杯高 ±3 mm、杯口深度丢失时轮廓杯高不变、叠好 / 在空中 / 围栏外 / 重复框、YOLOE 不可用即终止、替身全流程）和 `tests/test_v3_66_gui.py`（真实 Tk）。
- **只在替身（假 YOLOE 实例轮廓）上验证过；真实 YOLOE 在你的 D435i 画面上的轮廓精度、杯高判断都还没在真机上验证**。

## V3.65 (2026-11-02) — 叠杯子 · SO-101 夹爪：仿真技能和参数迁移到 NERO 实机（新 TAB「NERO实机任务」）
- **新 TAB「NERO实机任务」**（`NERO_V365_Real_Cup_Task.py`，在「技能参数旋钮」之后）：主相机 / 腕部相机两个实时监控框（共用 Camera调试 页的相机流，画出识别到的杯子和抓取 / 松手点），SO-101 夹爪面板，技能与参数，Agent（Provider / Model / API key / Reasoning / Max steps / 任务 + 示例），运行档位 ① 视觉规划预览 ② 悬停预演 ③ 实机执行，日志。设置存 `NERO_CALIB_DIR/real_cup_task.json`（不写发布文件夹）。
- **一套大脑，两套身体**：实机直接用仿真的 `CupSkills`、当前技能的第 ⑥ 组旋钮（默认 CS）、`stack_cups_so101` 的 SO-101 夹爪档案和 LLM 工具契约；新增 `RealIO` 只替换眼睛（固定 D435i + 手眼标定 → V3.63 杯子感知；桌面高度取安全区 `z_table`、工作区取围栏范围）、手臂（关节路径执行）和手（STS3215）。实机没有 CONNECT 抓持摆动，抬起后只等 0.5 s。仿真模块（后端、技能、场景、SO-101 模型、感知）一行未改。
- **STS3215 夹爪驱动**（`NERO_V365_STS3215_Gripper.py`）：协议、寄存器、遥测块读取取自用户的 `sts3215_gui_tool.py`。角度约定：180°（2048）= 夹紧（指尖缝 0），正角度 = 张开；开合与仿真同一套几何换算（预抓取 = 指尖缝 32 mm ≈ 23.8°）。STS3215 没有电流 / 力闭环 → 夹紧 = 位置控制 + 扭矩上限（目标比夹紧位再进 8°，杯壁挡住后限扭矩压住），扭矩上限 = 夹紧力 × 杠杆 69.9 mm ÷ 堵转扭矩 1.91 N·m（5 N ≈ 18 %，限 15–60 %）；停住时指尖缝 ≥ 0.8 mm 判为夹到。内置模拟舵机（串口选 SIMULATOR）可以不接硬件试界面。
- **关节路径执行**（`NERO_V313_Real_Robot.py`）：ROS monitor 新增 `do_path`（与 `do_move` 分开，原逻辑不变），把控制台算好的关节路径（时间戳 + 中点速度，首末速度 0）原样交给 MoveIt `/execute_trajectory`；发送前按 `do_move` 同样的方式做整条路径围栏检查；`RealPipeline.move_joint_path` 检查 MoveIt 状态、示教模式、每个点的关节限位和目标围栏。monitor 状态里带 `caps=["joint_path"]`；旧 monitor 时退回为一串短的 MoveIt 关节目标。
- **运行档位**：① 只识别 + 规划；② 预抓取悬停（指尖约在杯口上方 5 cm）→ 竖直升到抬起高度 → 原地转腕 → 目标杯上方 → 回起点，夹爪只张开；③ 固定流程（观察 → 规划 → 抓 → 放 → 验证 → 回起点）或 LLM Agent。② ③ 先检查相机 / 标定 / 关节反馈 / MoveIt / 夹爪，再弹窗确认；回到的“安全位”是任务开始时的姿态。
- **老测试有意修改**：版本白名单加 V3.65（`test_v3_64` 允许控制台等模块为 V3.65）。
- **测试**：新增 `tests/test_v3_65.py`（W1–W8：协议帧与用户工具一致、角度 / 缝宽映射、夹紧判定、路径时间、`move_joint_path` 门控与发送内容（假 ROS monitor 子进程）、`do_path` 在假 ROS 消息上运行、替身当实机身体跑 ①②③、控制台 TAB 位置）、`tests/test_v3_65_gui.py`（真实 Tk）、`tests/v365_real_twin.py`（替身身体）。
- **只在替身和模拟舵机上验证过，真机、真实舵机、真实杯子的识别都还没跑过**。实物需要：两只同款 16 oz 红色 party 杯（感知按红色 + 杯高 121 mm 判断）、SO-101 指尖加指垫、安全区围栏覆盖杯子所在区域。

## V3.64 (2026-11-02) — 新仿真场景：叠杯子 · SO-101 夹爪（7 mm 转接环，偏转 45°）
- **新场景 `stack_cups_so101`**：仿真页场景下拉框最后一项；原三个场景、界面和功能不变。它是 `stack_cups` 的变体：`active_scene()` 仍返回 `stack_cups`，所以后端、相机进程、杯子感知、Agent 全部原样运行（字节级不变），只有建模和杯子技能读取 `gripper_variant()`。
- **SO-101 夹爪建模（新模块 `NERO_V364_SO101_Gripper.py` + `assets/so101/`）**：TheRobotStudio SO-ARM100 官方 SO-101 网格（Apache-2.0，见 NOTICE）。安装：NERO 夹爪安装座面（link7 x = 37.5 mm，估计值）+ 7 mm 转接环 + 夹爪绕转接环法线（link7 +x）右手 −45°（= 用户说的“向外方向左手 45°”）。后端需要的名字全部保留：`gripper_joint1`（虚拟滑块，开口 = 2000·q1）、`gripper_joint2`（与官方一样 = −q1，后端急停硬复位写它仍然精确）、`gripper_link1/2`、`finger1/2_rubber_pad`；动指铰链 `so101_jaw_hinge` = −10° + k·q1。碰撞用简化方块，网格只显示。
- **指垫**：裸指尖闭合到 −10° 限位时仍隔约 2.4 mm，捏不住 2 mm 杯壁；两指尖各加 4 mm 指垫（约 −6° 夹紧）。**实物同样需要加橡胶 / TPU 指垫**。
- **算法（同一套 CS 技能 + 夹爪档案 `GripperProfile`）**：技能规划“抓取坐标系”，档案给出 TCP→抓取系的固定变换（含偏转和转接环）与开口。原夹爪档案 = V3.63 行为。SO-101 档案：固定指在杯外、动指在杯内（另一侧下落时杯子会卡在没张开的动指上 / 固定指在内时持杯倾斜 18°）；沿杯壁斜向下探，固定指离杯壁 2 mm（相机 ~1 mm 误差下不再触发后端指垫穿透保护）；抬起、搬运后各等 2.5 s 让杯子在 CONNECT 抓持上摆正（后端穿透保护在抬起时会松开夹爪，原夹爪也会摆 30°，只是张得更开、摆正更快）；松手高度上限 35 mm（原 85 mm）且允许 ≤ 9° 倾斜；松开后 TCP 水平外移 8 mm 让固定指离开杯壁（锥面杯下落时外侧杯壁会压向固定指）。
- **通用小修正（原夹爪也受益，不改变已能通过的规划）**：抓取窗口内找不到抓取或最佳链路余量 < 4° 时自动全角度搜索；搬运关节空间回退优先用规划好的关节解（V3.64 替身发现只从当前关节求解会落到 J1 离限位 0.5° 的分支）；关节目标停稳等待 1.2 s、误差门限 3° → 4°（V3.63 实测一次 3.22° 被判失败）；直线下探 / 上抬被拒时只走严格连续的直线关节路径，不再自由关节运动穿过杯子。
- **替身试验** `config/cup_stack_study_v364_so101.json`（`python3 NERO_V362_Cup_Skills.py --study-so101`）：两个方向、杯位 ±10 mm、捏深 30 / 40 mm、固定相机位姿源。后端适配层（`BackendIO` + 假后端核心）两个方向也套好。
- **老测试有意修改**：版本白名单加 V3.64；`test_v3_61` 场景列表允许第 4 项；`test_v3_62` SCENES_REV 允许 V3.64；`tests/v362_backend_twin.py` 支持 `gripper="so101"`。
- **测试**：新增 `tests/test_v3_64.py`（U1–U8）。
- 只在替身里验证过（不含机械臂动力学、ros2_control、真实 MoveIt / cuRobo；MoveIt 碰撞模型仍是原夹爪），**完整仿真链路需要你复测**，真机未验证。安装尺寸请实测后用环境变量 `NERO_V364_SO101_SEAT_X_MM / NERO_V364_SO101_RING_MM / NERO_V364_SO101_ROLL_DEG` 校正。

## V3.63 (2026-11-02) — 叠杯子：修复放置时 J7 越限急停（关节空间安全运动 5 项）+ 固定主相机 D435i 杯子感知
- **V3.62 实测失败原因**：放置时第一次几毫米的直线对中让 J7 越过 MuJoCo 限位（`diag_trip_1_runaway_j7`，“invalid arm state → emergency hard restore”）。根源：7 自由度臂冗余，MoveIt / cuRobo 执行时走的关节分支不是规划时检查过余量的那一组（规划器从真实 home 姿态出发也能复现：J7 只剩 1.6°），且 MoveIt 可能用比 MuJoCo 更宽的关节限位；V3.62 技能代码又没检查急停，锁住后继续空转并报成“没对中”。
- **修复 1 · 执行前检查轨迹**：每条直线插补轨迹逐点检查（7 个关节都离限位 ≥ 4°、相邻点关节跳变 ≤ 5°），不合格不执行，改为关节空间运动到自己算过余量的 IK 解；手里有杯子时还检查关节空间路径末端不下沉超过 10 mm（否则从中点分两段）。
- **修复 2 · 先原地转腕再等高平移**：放置分为「原地慢速转腕（关节空间）→ 等高平移 → 慢降」三步。
- **修复 3 · 联合规划 + J7 余量**：抓取候选（杯沿角 × 夹爪翻转 × 两种 IK 分支）与放置杯沿角联合搜索，整条链路（预抓取…上抬）每个持杯姿态离限位 ≥ 8°、放置时 J7 优先 ≥ 15°；规划从机械臂当前真实关节出发，每个航点存关节解，执行时朝这组关节解运动（预抓取直接用关节目标，不再用自由位姿目标）。窄窗口找不到再全角度搜索。逆解改为 numpy 解析雅可比（`FastArm`），一次规划约 4–9 s。
- **修复 4 · 急停即停**：每一步检查安全计数和执行结果；急停后返回 `SAFETY_STOP`，技能自动执行后端的 STOP + RECOVER（`_v337_clear_latch`，保持夹持），锁未解则不允许再动；Agent 规则：恢复成功可观察后重试同一工具一次，同一工具第二次急停即停止报告。
- **修复 5 · 不做小修正**：高处对中并入平移，残差 < 3 mm 不修正，降到松手高度后最多修正一次。
- **固定主相机 D435i 杯子感知（新模块 `NERO_V363_Cup_Perception.py`）**：沿用抓鸡蛋固定相机的路线（同一相机进程、同一 RGB-D 渲染与 D435i 模型、同一回传协议），只看图像：红色色调分割 → 深度反投影 → 去掉机械臂自身的点 → 按位置聚类 → 用已知杯型锥面拟合杯轴 / 高度 / 倾斜；簇高比单杯高约 25 mm 判为「已套叠」。`NERO_V331_Fixed_Camera.py` 在 `stack_cups` 场景改为杯子感知（鸡蛋场景不变）。位姿源选「fixed stand camera D435i」即可；腕部相机仍不支持（会提示）；相机进程没运行时拒绝执行，不退回真值。被夹住的杯子相机看不清，用“夹爪位姿 + 抓取时测得的相对偏移”推算（仿真里抓持锁定是刚性的）。
- **后端**：仍只在 RUN TASK 的同一个 A2 块里多传了 5 个后端函数（取图、相机状态、腕部判断、安全锁检查 / 解锁）；去掉该段与 V3.60 字节级一致，`test_v3_44` 不变。
- **替身升级**：替身现在带 URDF 机械臂运动学（关节空间运动、直线运动按连续 IK 路径检查），并带同一台固定 D435i 的渲染。`config/cup_stack_study_v363.json`：默认参数附近 20/20 完全套好（其中固定相机位姿源 4/4，含 D435i 深度噪声和杯位 ±10 mm）；竖直下探两组因可达性被拒绝。相机模式下被夹杯子的倾斜测不到（记为 0）。
- **老测试有意修改**：`test_v3_62`（P4 规划余量放宽到 ≥ 4°——真实起始姿态的检查在 test_v3_63 S4；P7 动作标签）、`test_v3_61`、版本白名单扩展到 V3.63。
- **测试**：新增 `tests/test_v3_63.py`（S1–S10：解析雅可比、路径检查、从 home 规划、替身拒绝直线改关节、急停恢复 + 二次急停停止、渲染帧感知精度、相机进程杯子分支、位姿源规则与固定相机完整任务）、测试素材 `tests/v363_cam_frames.npz`。
- 只在替身里验证过（不含机械臂动力学、ros2_control 和真实 MoveIt / cuRobo），**完整仿真链路需要你复测**，真机未验证。

