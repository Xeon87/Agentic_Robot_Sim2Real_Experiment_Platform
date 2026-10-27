#!/usr/bin/env python3
"""NERO V3.16 SAFETY EDITOR (V3.17: forward = −x is drawn up; side view explains each height): graphical safety-zone editor for the 实机 tab (replaces the V3.15 number table).

Top view (XY, +x up, +y left) and side view (XZ / YZ) of the workspace:
- safety zone polygon (drag corners / edges, double-click an edge to add a corner, right-click a corner to delete),
  inner margin line, base disc, collision boxes (drag to move, corner handle to resize), table marks M1..Mn
- side view: drag z_table / z_min / z_max lines
- live arm (link4..link7 + tool point) and the joint-panel target as a dashed ghost; violations in red
- 示教角点: record the tool tip on each table mark (FK of the measured joints + tool), generate the zone and the
  table height; while teaching only the tool point may go 5 mm below the table and speed is capped at 5 %
- 巡检: hover above each corner (small IK keeps the orientation), you confirm the pointer is over the mark
- undo / redo, save (config/real_workspace.json), apply to MoveIt, automatic fence + MoveIt test
Logic lives in NERO_V315_Consistency_Safety.py (polygon fence, IK); this file is Tk only.
"""
import json
import math
import os
import queue
import threading
import time

import NERO_V314_Joint_Control as JC
import NERO_V315_Consistency_Safety as CS

TEACH_TOOL_BELOW_TABLE = 0.005
INSPECT_TOOL_ABOVE_TABLE = 0.010
TEACH_MAX_SPEED = 0.05
HOVER_DEFAULT_MM = 50.0
# V3.19 ZONE FILES: zones can be saved under any name / folder; the last used file opens at the next start.
STATE_PATH = os.path.join(os.path.expanduser("~"), ".config", "nero_console", "state.json")
ZONE_DIR_DEFAULT = os.path.join(os.path.expanduser("~"), "nero_zones")


def load_state(path=None):
    try:
        with open(path or STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(d, path=None):
    path = path or STATE_PATH
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        cur = load_state(path)
        cur.update(d)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cur, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
        return True
    except Exception:
        return False
COL = {"zone": "#2e7d32", "zone_fill": "#cfe8cf", "box": "#e65100", "box_fill": "#ffcc80", "mark": "#d32f2f",
       "arm": "#1565c0", "arm_pt": "#64b5f6", "ghost": "#7b1fa2", "ghost_pt": "#ce93d8", "bad": "#d32f2f",
       "table": "#efe6d6", "table_line": "#b9a47e", "grid": "#e4e4e4", "sel": "#1565c0", "warn": "#f9a825"}
PT_LABEL = {"link4": "l4", "link5": "l5", "link6": "l6", "link7": "l7", "tool": "工具"}


class SafetyEditorUI(CS._WorkerMixin):
    TW, TH = 600, 540
    SW, SH = 560, 340

    def __init__(self, tk, ttk, parent, pipe, jpanel, cfg_fn, log, ask, path=None):
        self.tk, self.ttk, self.pipe, self.jpanel, self.cfg_fn, self.log, self.ask = tk, ttk, pipe, jpanel, cfg_fn, log, ask
        self.path = path or CS.WORKSPACE_PATH
        self.sim = CS.load_sim_urdf()
        last = load_state().get("last_zone_file") if path is None else None
        self.opened_last = None
        if last and os.path.isfile(last):
            w_last = CS.load_workspace(last)
            if not w_last.get("load_error"):
                self.path, self.opened_last = last, last
        self.ws = CS.load_workspace(self.path)
        self.valid_ws = json.loads(json.dumps(self.ws))
        self.undo_stack, self.redo_stack = [], []
        self.job = None
        self.ui_q = queue.Queue()
        self.stop_flag = False
        self.scene_applied_for = None
        self._auto_pending = False
        self.mode = "edit"
        self.sel = None
        self.drag = None
        self.teach_pts = []            # [{"name", "xyz", "q"}]
        self.dirty, self.invalid, self.sdrag = False, None, None
        self._build(parent)
        self._fill_fields()
        pipe.safety_fn = self.safety
        if self.opened_last:
            self.log(f"[OK] [V3.19 safety] 已打开上次使用的安全区文件：{self.opened_last}")
        if self.ws.get("migrated_from"):
            self.log(f"[INFO] [V3.16 safety] config/real_workspace.json 从 {self.ws['migrated_from']} 迁移：x/y 范围 → 矩形安全区，请检查后保存")

    def _build(self, parent):
        tk, ttk = self.tk, self.ttk
        f = ttk.Frame(parent, padding=4)
        f.pack(fill="both", expand=True)
        tb = ttk.Frame(f)
        tb.pack(fill="x")
        ttk.Label(tb, text="工具：", font=("TkDefaultFont", 10, "bold")).pack(side="left")
        self.v_mode = tk.StringVar(value="edit")
        for t_, m in (("选择 / 编辑", "edit"), ("示教角点", "teach"), ("巡检", "inspect")):
            tk.Radiobutton(tb, text=t_, value=m, variable=self.v_mode, indicatoron=0, padx=8, pady=2,
                           selectcolor="#9fc3f5", command=self.on_mode).pack(side="left", padx=1)
        ttk.Button(tb, text="＋ 碰撞体", command=self.on_add_box).pack(side="left", padx=(10, 2))
        ttk.Button(tb, text="撤销", command=self.on_undo).pack(side="left", padx=2)
        ttk.Button(tb, text="重做", command=self.on_redo).pack(side="left", padx=2)
        ttk.Button(tb, text="自动越界测试", command=self.on_test_auto).pack(side="left", padx=(10, 2))
        for t_, fn in (("从 MoveIt 清除", self.on_clear), ("应用到 MoveIt", self.on_apply), ("另存为…", self.on_save_as),
                       ("保存", self.on_save), ("打开…", self.on_open), ("重新载入", self.on_reload)):
            ttk.Button(tb, text=t_, command=fn).pack(side="right", padx=2)
        self.v_head = tk.StringVar(value="")
        self.lbl_head = JC.fixed_label(tk, f, 1, textvariable=self.v_head, font=("TkDefaultFont", 9, "bold"),
                                       pack={"pady": (3, 2)})
        main = ttk.Frame(f)
        main.pack(fill="both", expand=True)
        left = ttk.Frame(main)
        left.pack(side="left", fill="y")
        self.cv_top = tk.Canvas(left, width=self.TW, height=self.TH, bg="white", highlightthickness=1, highlightbackground="#999")
        self.cv_top.pack()
        ttk.Label(left, text="编辑：拖角点 / 拖边 / 拖碰撞体（右下角缩放）· 双击边加角点 · 右键角点删除 · Esc 取消选择",
                  foreground="#555555").pack(anchor="w")
        right = ttk.Frame(main)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        sr = ttk.Frame(right)
        sr.pack(fill="x")
        self.v_side = tk.StringVar(value="XZ")
        ttk.Label(sr, text="侧视图").pack(side="left")
        for m in ("XZ", "YZ"):
            ttk.Radiobutton(sr, text=m, value=m, variable=self.v_side, command=self.redraw).pack(side="left", padx=4)
        self.v_side_hint = tk.StringVar(value="")
        JC.fixed_label(tk, sr, 1, width=360, textvariable=self.v_side_hint, fg="#555555", pack={"side": "left", "padx": 6})
        self.cv_side = tk.Canvas(right, width=self.SW, height=self.SH, bg="white", highlightthickness=1, highlightbackground="#999")
        self.cv_side.pack(anchor="w")
        # V3.18 SIDE LEGEND: what the colours mean lives here (static), not inside the drawing
        lg = tk.Frame(right)
        lg.pack(anchor="w", pady=(2, 0))
        for sw_fill, sw_stip, sw_line, txt in (
                ("#f3b6b6", "gray50", "", "防撞桌区：桌面 → z_min，任何检查点不得进入"),
                (COL["zone_fill"], "", COL["zone"], "安全区（侧面投影）：规则 A = 围墙一直到 z 上限；规则 B = 只到 XY 限高"),
                ("#eef6ee", "", "", "XY 不限（只有规则 B）：XY 限高以上只限 z 上限"),
                ("", "", "", "拖动右边线头的小方块修改高度")):
            row = tk.Frame(lg)
            row.pack(anchor="w")
            cv = tk.Canvas(row, width=22, height=12, highlightthickness=0)
            if sw_fill:
                cv.create_rectangle(1, 1, 21, 11, fill=sw_fill, outline=sw_line or "#999999",
                                    **({"stipple": sw_stip} if sw_stip else {}))
            else:
                cv.create_rectangle(7, 2, 15, 10, fill="white", outline="#555555", width=2)
            cv.pack(side="left")
            tk.Label(row, text=txt, font=("TkDefaultFont", 8), fg="#444444").pack(side="left", padx=4)
        self.v_now = tk.StringVar(value="")
        self.lbl_now = JC.fixed_label(tk, right, 4, textvariable=self.v_now, font=("TkDefaultFont", 10, "bold"),
                                      wraplength=550, bd=1, relief="solid", padx=6, pady=3, pack={"pady": (6, 4)})
        self.v_test = tk.StringVar(value="")
        JC.fixed_label(tk, right, 3, textvariable=self.v_test, wraplength=560, fg="#333333")
        self.panel_holder = ttk.Frame(right)
        self.panel_holder.pack(fill="both", expand=True)
        self._build_edit_panel()
        self._build_teach_panel()
        self._build_inspect_panel()
        self._show_panel()
        # bindings
        c = self.cv_top
        c.bind("<Button-1>", self.on_top_press)
        c.bind("<B1-Motion>", self.on_top_drag)
        c.bind("<ButtonRelease-1>", self.on_top_release)
        c.bind("<Double-Button-1>", self.on_top_double)
        c.bind("<Button-3>", self.on_top_right)
        c.bind("<Escape>", lambda _e: self._select(None))
        s = self.cv_side
        s.bind("<Button-1>", self.on_side_press)
        s.bind("<B1-Motion>", self.on_side_drag)
        s.bind("<ButtonRelease-1>", self.on_side_release)


    def _build_edit_panel(self):
        tk, ttk = self.tk, self.ttk
        p = self.p_edit = ttk.Frame(self.panel_holder)
        sel = ttk.LabelFrame(p, text="选中对象")
        sel.pack(fill="x", pady=2)
        self.v_sel = tk.StringVar(value="（点击角点或碰撞体）")
        JC.fixed_label(tk, sel, 1, textvariable=self.v_sel, font=("TkDefaultFont", 9, "bold"), pack={"padx": 4})
        r = ttk.Frame(sel)
        r.pack(fill="x", padx=4, pady=2)
        self.sel_vars = {}
        for k, lab in (("name", "名称"), ("x", "x"), ("y", "y"), ("sx", "尺寸x"), ("sy", "y"), ("sz", "z"), ("cz", "中心z")):
            ttk.Label(r, text=lab).pack(side="left")
            self.sel_vars[k] = tk.StringVar()
            e = ttk.Entry(r, textvariable=self.sel_vars[k], width=9 if k == "name" else 6)
            e.pack(side="left", padx=(1, 5))
            e.bind("<Return>", lambda _e: self.on_sel_apply())
        ttk.Button(r, text="应用", command=self.on_sel_apply).pack(side="left")
        ttk.Button(r, text="删除", command=self.on_sel_delete).pack(side="left", padx=3)
        hz = ttk.LabelFrame(p, text="高度、余量、工具（m）")
        hz.pack(fill="x", pady=2)
        self.f_vars = {}
        rr = ttk.Frame(hz)                          # V3.23 safety rule
        rr.pack(fill="x", padx=4, pady=1)
        ttk.Label(rr, text="安全规则").pack(side="left")
        self.v_rule = tk.StringVar(value="A")
        ttk.Radiobutton(rr, text="A 围墙（默认：本体+夹爪任何高度都不越墙）", value="A", variable=self.v_rule,
                        command=self.on_fields_apply).pack(side="left", padx=2)
        ttk.Radiobutton(rr, text="B 分层（XY 限高以上不限 XY）", value="B", variable=self.v_rule,
                        command=self.on_fields_apply).pack(side="left", padx=2)
        rows = ((("z_table", "桌面 z"), ("z_min_points", "z_min"), ("z_max", "z 上限"), ("margin", "离墙余量")),
                (("link_radius", "A 连杆半径"), ("gripper_radius", "A 夹爪半径"), ("base_radius", "B 底座区半径"),
                 ("xy_free_height", "B XY 限高"), ("tool_length", "工具长度")))
        for row in rows:
            rr = ttk.Frame(hz)
            rr.pack(fill="x", padx=4, pady=1)
            for k, lab in row:
                ttk.Label(rr, text=lab).pack(side="left")
                self.f_vars[k] = tk.StringVar()
                e = ttk.Entry(rr, textvariable=self.f_vars[k], width=7)
                e.pack(side="left", padx=(1, 8))
                e.bind("<Return>", lambda _e: self.on_fields_apply())
        rr = ttk.Frame(hz)
        rr.pack(fill="x", padx=4, pady=1)
        ttk.Label(rr, text="工具点").pack(side="left")
        self.v_tool_mode = tk.StringVar()
        ttk.Combobox(rr, textvariable=self.v_tool_mode, values=["sim_tcp", "pointer", "none"], width=8,
                     state="readonly").pack(side="left", padx=2)
        ttk.Label(rr, text="指针轴(link7)").pack(side="left")
        self.v_tool_axis = tk.StringVar()
        ttk.Combobox(rr, textvariable=self.v_tool_axis, values=list(CS.TOOL_AXES), width=4, state="readonly").pack(side="left", padx=2)
        self.v_en, self.v_path = tk.BooleanVar(), tk.BooleanVar()
        ttk.Checkbutton(rr, text="围栏启用", variable=self.v_en, command=self.on_fields_apply).pack(side="left", padx=6)
        ttk.Checkbutton(rr, text="检查规划路径", variable=self.v_path, command=self.on_fields_apply).pack(side="left")
        ttk.Button(rr, text="应用数值", command=self.on_fields_apply).pack(side="right")
        rr = ttk.Frame(hz)
        rr.pack(fill="x", padx=4, pady=1)
        self.v_walls = tk.BooleanVar(value=True)
        ttk.Checkbutton(rr, text="A：围墙同步到 MoveIt（保存 / 打开后自动）", variable=self.v_walls,
                        command=self.on_fields_apply).pack(side="left")
        ttk.Button(rr, text="并入底座", command=self.on_merge_base).pack(side="left", padx=8)
        ttk.Label(p, text="A：沿 link2…link7→工具 每 5 cm 取一个检查球（连杆半径 / 夹爪半径），球面离每面墙 ≥ 离墙余量；"
                          "多边形必须包住底座。B：关节原点 + 工具点，只在 XY 限高以下限制 XY。"
                          "sim_tcp = 仿真 TCP（link7 前方 155 mm）；pointer = 示教用指针。", foreground="#555555",
                  wraplength=560, justify="left").pack(anchor="w", pady=2)


    def _build_teach_panel(self):
        tk, ttk = self.tk, self.ttk
        p = self.p_teach = ttk.Frame(self.panel_holder)
        st = ttk.LabelFrame(p, text="步骤")
        st.pack(fill="x", pady=2)
        for t_ in ("① 法兰装指针，「选择/编辑」里把工具点设为 pointer 并填长度和轴向",
                   f"② 速度自动限制 ≤ {TEACH_MAX_SPEED * 100:.0f}%；点动（关节控制页）或用 NERO 示教模式手带，把尖端放到标记十字中心",
                   "③ 点「记录角点」（按桌面标记的顺序：M1, M2, …）；示教模式下用手拖到标记上再记录",
                   "④ 记完 ≥ 3 个角点 → 「生成安全区」（同时得到桌面高度和平整度）",
                   f"示教期间只对工具点把 z_min 放宽到 桌面 − {TEACH_TOOL_BELOW_TABLE * 1000:.0f} mm、安全区 XY 只限制工具点；"
               "其他检查点的 z 限制照常；离开示教自动恢复"):
            ttk.Label(st, text=t_, wraplength=560, justify="left").pack(anchor="w", padx=4)
        r = ttk.Frame(st)
        r.pack(fill="x", padx=4, pady=3)
        tk.Button(r, text="● 记录角点", bg="#2e7d32", fg="white", font=("TkDefaultFont", 10, "bold"),
                  command=self.on_teach_record).pack(side="left")
        ttk.Button(r, text="撤销上一点", command=self.on_teach_undo).pack(side="left", padx=4)
        ttk.Button(r, text="清空", command=self.on_teach_clear).pack(side="left", padx=4)
        ttk.Button(r, text="生成安全区", command=self.on_teach_generate).pack(side="left", padx=4)
        tm = ttk.Frame(st)
        tm.pack(fill="x", padx=4, pady=(0, 3))
        tk.Button(tm, text="▶ 进入示教模式（可用手带动）", bg="#1565c0", fg="white", font=("TkDefaultFont", 9, "bold"),
                  command=self.on_teach_enter).pack(side="left")
        tk.Button(tm, text="■ 退出示教模式", font=("TkDefaultFont", 9, "bold"), command=self.on_teach_exit).pack(side="left", padx=6)
        self.v_teach_mode = tk.StringVar(value="NERO：正常控制")
        self.lbl_teach_mode = JC.fixed_label(tk, tm, 1, width=300, textvariable=self.v_teach_mode,
                                             font=("TkDefaultFont", 9, "bold"), pack={"side": "left", "padx": 6})
        self.v_tip = tk.StringVar(value="")
        JC.fixed_label(tk, p, 1, textvariable=self.v_tip, font=("TkFixedFont", 10, "bold"), fg="#1565c0", pack={"pady": 2})
        self.tv_teach = ttk.Treeview(p, columns=("n", "x", "y", "z"), show="headings", height=6)
        for c_, h in zip(self.tv_teach["columns"], ("角点", "x (m)", "y (m)", "z 桌面 (m)")):
            self.tv_teach.heading(c_, text=h)
            self.tv_teach.column(c_, width=110, anchor="center")
        self.tv_teach.pack(fill="x")
        rr = ttk.Frame(p)
        rr.pack(fill="x", pady=3)
        ttk.Label(rr, text="交叉核对：参考点 R 尺量到底座中心").pack(side="left")
        self.v_ref = tk.StringVar(value="500")
        ttk.Entry(rr, textvariable=self.v_ref, width=6).pack(side="left", padx=2)
        ttk.Label(rr, text="mm").pack(side="left")
        ttk.Button(rr, text="尖端放在 R 上 → 核对", command=self.on_teach_ref).pack(side="left", padx=6)
        self.v_teach_info = tk.StringVar(value="")
        JC.fixed_label(tk, p, 2, textvariable=self.v_teach_info, wraplength=560)


    def _build_inspect_panel(self):
        tk, ttk = self.tk, self.ttk
        p = self.p_inspect = ttk.Frame(self.panel_holder)
        r = ttk.Frame(p)
        r.pack(fill="x", pady=2)
        ttk.Label(r, text="悬停高度 mm").pack(side="left")
        self.v_hover = tk.StringVar(value=f"{HOVER_DEFAULT_MM:.0f}")
        ttk.Entry(r, textvariable=self.v_hover, width=5).pack(side="left", padx=3)
        ttk.Button(r, text="▶ 开始巡检", command=self.on_inspect).pack(side="left", padx=6)
        ttk.Button(r, text="■ 中止", command=self.on_stop).pack(side="left")
        self.tv_insp = ttk.Treeview(p, columns=("n", "h", "ik", "res"), show="headings", height=6)
        for c_, h, w in zip(self.tv_insp["columns"], ("角点", "悬停", "IK 误差", "你的判断"), (80, 80, 100, 160)):
            self.tv_insp.heading(c_, text=h)
            self.tv_insp.column(c_, width=w, anchor="center")
        self.tv_insp.pack(fill="x", pady=2)
        ttk.Label(p, text=f"机械臂依次停到每个角点上方（速度 ≤ {TEACH_MAX_SPEED * 100:.0f}%，示教过的角点沿用示教姿态），"
                          "每到一个点弹窗问你指针是否对准标记。偏差大 → 检查工具长度 / 零位（一致性测试页）。"
                          "路径照常经过围栏与 MoveIt 检查。", wraplength=560, justify="left", foreground="#555555").pack(anchor="w")



    # ================================================================ state helpers
    def fence(self):
        return self.ws["fence"]

    def _snapshot(self):
        return json.dumps({k: v for k, v in self.ws.items() if k not in ("load_error",)}, sort_keys=True)

    def _push_undo(self, snap=None):
        self.undo_stack.append(snap or self._snapshot())
        self.undo_stack = self.undo_stack[-100:]
        self.redo_stack.clear()

    def _changed(self, what=""):
        """After any edit: re-validate; a valid zone is enforced immediately (preview = enforcement)."""
        try:
            CS.validate_workspace(json.loads(json.dumps(self.ws)))
            self.valid_ws = json.loads(json.dumps(self.ws))
            self.dirty = True
            self.invalid = None
        except Exception as exc:
            self.invalid = str(exc)
        self._head()
        self.redraw()

    def on_undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append(self._snapshot())
        self.ws = json.loads(self.undo_stack.pop())
        self.sel = None
        self._fill_fields()
        self._changed("undo")

    def on_redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(self._snapshot())
        self.ws = json.loads(self.redo_stack.pop())
        self.sel = None
        self._fill_fields()
        self._changed("redo")

    def _head(self):
        if self.ws.get("load_error"):
            txt, col = f"config/real_workspace.json 读取失败 → 使用默认值：{self.ws['load_error']}", "#c62828"
        elif self.fence().get("rule") == "A" and not CS.base_inside(self.fence()):
            txt, col = ("⚠ 规则 A：安全区没有包住 NERO 底座 → 所有运动都会被拒绝。点「选择 / 编辑」里的「并入底座」"
                        "（或拖角点把底座包进去）后保存", "#c62828")
        elif all(p[0] > -0.05 for p in self.fence()["polygon"]):    # V3.24: zones may extend behind the base
            txt, col = ("⚠ 安全区在底座后方（+x）：NERO 的工作方向是 −x（V3.16 的默认安全区画反了），请重新示教或拖到前方后保存",
                        "#c62828")
        elif getattr(self, "invalid", None):
            txt, col = f"⚠ 当前编辑无效（仍执行上一个有效安全区）：{self.invalid}", "#c62828"
        elif not self.ws.get("measured"):
            txt, col = "⚠ 默认值（未测量）：请示教角点或按桌面标记拖动安全区，保存后再做大范围运动", "#c62828"
        elif getattr(self, "dirty", False):
            txt, col = "有未保存的修改（已立即生效；「保存」/「另存为…」写入文件，「应用到 MoveIt」更新碰撞体）", "#b26a00"
        else:
            txt, col = f"安全区已保存 {self.ws.get('saved', '')}", "#2e7d32"
        self.v_head.set(txt + f"   · 文件：{self.path}")
        self.lbl_head.configure(fg=col)

    def safety(self):
        """Provider for RealPipeline (any thread)."""
        w = self.valid_ws
        f = json.loads(json.dumps(w["fence"]))
        mode = self.mode
        if mode == "teach":
            f["tool_z_min_override"] = f["z_table"] - TEACH_TOOL_BELOW_TABLE
        elif mode == "inspect":
            f["tool_z_min_override"] = min(f["z_min_points"], f["z_table"] + INSPECT_TOOL_ABOVE_TABLE)
        if mode in ("teach", "inspect"):
            if f.get("rule") == "A":
                f["relax_tool"] = True              # V3.23: tip + gripper may touch the wall line (corner marks)
            else:
                f["xy_points_only"] = ["tool"]      # rule B: the zone limits the tool tip, z limits all points
        return {"fence": f if f.get("enabled", True) else None, "path_check": bool(f.get("path_check", True)),
                "tool_T": CS.tool_transform(f.get("tool"), self.sim), "sim_urdf": self.sim,
                "measured": bool(w.get("measured")), "corners": w.get("corners"),
                "max_speed": TEACH_MAX_SPEED if mode in ("teach", "inspect") else None, "mode": mode}

    def _urdf(self):
        return self.pipe.real_urdf or self.sim

    def _base(self):
        return self.jpanel.model.get("base_link") or "base_link"

    # ================================================================ fields
    def _fill_fields(self):
        f = self.fence()
        for k in ("z_table", "z_min_points", "z_max", "margin", "base_radius", "xy_free_height", "link_radius",
                  "gripper_radius"):
            self.f_vars[k].set(f"{f[k]:.3f}")
        self.v_rule.set(f.get("rule", "A"))
        self.v_walls.set(bool(f.get("walls_to_moveit", True)))
        self.f_vars["tool_length"].set(f"{f['tool'].get('length', 0.1):.3f}")
        self.v_tool_mode.set(f["tool"].get("mode", "sim_tcp"))
        self.v_tool_axis.set(f["tool"].get("axis", "+x"))
        self.v_en.set(bool(f.get("enabled", True)))
        self.v_path.set(bool(f.get("path_check", True)))
        self._fill_sel()
        self._head()

    def on_fields_apply(self):
        snap = self._snapshot()
        f = self.fence()
        try:
            vals = {k: float(self.f_vars[k].get()) for k in ("z_table", "z_min_points", "z_max", "margin", "base_radius",
                                                             "xy_free_height", "link_radius", "gripper_radius")}
            L = float(self.f_vars["tool_length"].get())
        except ValueError:
            self.log("[WARN] [V3.16 safety] 数值无效")
            return
        old_zt = f["z_table"]
        f.update(vals)
        f["tool"] = {"mode": self.v_tool_mode.get(), "length": L, "axis": self.v_tool_axis.get()}
        f["enabled"], f["path_check"] = bool(self.v_en.get()), bool(self.v_path.get())
        old_rule = f.get("rule", "A")
        f["rule"] = self.v_rule.get() or "A"
        f["walls_to_moveit"] = bool(self.v_walls.get())
        if f["rule"] != old_rule:
            self.log(f"[INFO] [V3.23 safety] 安全规则 → {'A 围墙' if f['rule'] == 'A' else 'B 分层'}"
                     f"{'' if f['rule'] == 'B' or CS.base_inside(f) else '（⚠ 底座不在安全区内：点「并入底座」）'}")
        if abs(f["z_table"] - old_zt) > 1e-9:
            self._table_to(f["z_table"])
        if self._snapshot() != snap:
            self._push_undo(snap)
            self._changed("fields")
        if not f["enabled"]:
            self.log("[WARN] [V3.16 safety] 围栏已关闭：目标和路径都不再检查（MoveIt 碰撞体仍有效）")

    def _table_to(self, z):
        for b in self.ws["boxes"]:
            if b["name"] == "table":
                b["pos"][2] = z - b["size"][2] / 2.0

    def _fill_sel(self):
        for v in self.sel_vars.values():
            v.set("")
        s = self.sel
        f = self.fence()
        if s and s[0] == "v" and s[1] < len(f["polygon"]):
            i = s[1]
            cm = self.ws["corners"][i] if i < len(self.ws["corners"]) else {"name": f"M{i + 1}", "source": "编辑"}
            self.v_sel.set(f"角点 {cm['name']}（来源：{cm.get('source', '编辑')}）")
            self.sel_vars["name"].set(cm["name"])
            self.sel_vars["x"].set(f"{f['polygon'][i][0]:.3f}")
            self.sel_vars["y"].set(f"{f['polygon'][i][1]:.3f}")
        elif s and s[0] == "box" and s[1] < len(self.ws["boxes"]):
            b = self.ws["boxes"][s[1]]
            self.v_sel.set(f"碰撞体 {b['name']}")
            self.sel_vars["name"].set(b["name"])
            for k, v in zip(("x", "y", "cz", "sx", "sy", "sz"), b["pos"] + b["size"]):
                self.sel_vars[k].set(f"{v:.3f}")
        else:
            self.v_sel.set("（点击角点或碰撞体）")

    def _select(self, s):
        self.sel = s
        self._fill_sel()
        self.redraw()

    def on_sel_apply(self):
        s = self.sel
        if not s:
            return
        snap = self._snapshot()
        try:
            if s[0] == "v":
                i = s[1]
                self.fence()["polygon"][i] = [float(self.sel_vars["x"].get()), float(self.sel_vars["y"].get())]
                self.ws["corners"][i]["name"] = self.sel_vars["name"].get().strip() or f"M{i + 1}"
                self.ws["corners"][i]["source"] = "编辑"
                self.ws["corners"][i]["q"] = None
            else:
                b = self.ws["boxes"][s[1]]
                name = self.sel_vars["name"].get().strip() or b["name"]
                if name != b["name"] and any(x["name"] == name for x in self.ws["boxes"]):
                    raise ValueError(f"名称 {name} 已存在")
                b["name"] = name
                b["pos"] = [float(self.sel_vars[k].get()) for k in ("x", "y", "cz")]
                b["size"] = [max(0.005, float(self.sel_vars[k].get())) for k in ("sx", "sy", "sz")]
        except Exception as exc:
            self.log(f"[WARN] [V3.16 safety] 数值无效：{exc}")
            return
        self._push_undo(snap)
        self._changed("sel")
        self._fill_sel()

    def on_sel_delete(self):
        s = self.sel
        if not s:
            return
        snap = self._snapshot()
        if s[0] == "v":
            if len(self.fence()["polygon"]) <= 3:
                self.log("[WARN] [V3.16 safety] 安全区至少保留 3 个角点")
                return
            del self.fence()["polygon"][s[1]]
            del self.ws["corners"][s[1]]
        else:
            if self.ws["boxes"][s[1]]["name"] == "table" and not self.ask("删除碰撞体", "删除桌面碰撞体 table？MoveIt 将看不到桌面。"):
                return
            del self.ws["boxes"][s[1]]
        self._push_undo(snap)
        self.sel = None
        self._fill_sel()
        self._changed("delete")

    def on_add_box(self):
        snap = self._snapshot()
        poly = self.fence()["polygon"]
        cx = sum(p[0] for p in poly) / len(poly)
        cy = sum(p[1] for p in poly) / len(poly)
        n = 1
        while any(b["name"] == f"box_{n}" for b in self.ws["boxes"]):
            n += 1
        zt = self.fence()["z_table"]
        self.ws["boxes"].append({"name": f"box_{n}", "size": [0.10, 0.10, 0.10], "pos": [cx, cy, zt + 0.05]})
        self._push_undo(snap)
        self.v_mode.set("edit")
        self.on_mode()
        self._select(("box", len(self.ws["boxes"]) - 1))
        self._changed("add box")

    # ================================================================ modes
    def _show_panel(self):
        for p in (self.p_edit, self.p_teach, self.p_inspect):
            p.pack_forget()
        {"edit": self.p_edit, "teach": self.p_teach, "inspect": self.p_inspect}[self.mode].pack(fill="both", expand=True)

    def on_mode(self):
        new = self.v_mode.get()
        if new == self.mode:
            return
        if self.mode == "teach":
            self.log("[INFO] [V3.16 safety] 离开示教：工具点 z_min 恢复，速度限制解除")
            if getattr(self.pipe, "teach_mode", False) and not getattr(self, "_teach_exiting", False):
                self.log("[WARN] [V3.19 teach] NERO 仍在示教模式（零力）：记得点「退出示教模式」")
        self.mode = new
        if new == "teach":
            self.log(f"[INFO] [V3.16 safety] 示教角点：速度限制 ≤ {TEACH_MAX_SPEED * 100:.0f}%，工具点 z_min 暂时 = 桌面 − "
                     f"{TEACH_TOOL_BELOW_TABLE * 1000:.0f} mm")
        self._show_panel()
        self.redraw()

    # ================================================================ geometry: views
    # V3.17 VIEW ORIENTATION: NERO works on the −x side of base_link (sim egg basket (−0.425, −0.155), holder
    # (−0.420, +0.155)). The top view is drawn as seen from behind the robot looking forward:
    # screen up = forward = −x, screen right = +y. The side view XZ puts forward (−x) to the right, YZ puts +y right.
    TOP_X = (-0.95, 0.70)          # x range (forward −0.95 m .. behind +0.70 m; V3.24: was +0.25)
    TOP_Y = (-0.85, 0.85)
    M_L, M_B, M_T = 34, 22, 26     # tick margins (left, bottom, top) in px

    def _top_tf(self):
        S = min((self.TW - self.M_L - 8) / (self.TOP_Y[1] - self.TOP_Y[0]),
                (self.TH - self.M_B - self.M_T) / (self.TOP_X[1] - self.TOP_X[0]))
        ox = self.M_L + (self.TW - self.M_L) / 2.0
        oy = self.M_T + (0 - self.TOP_X[0]) * S
        return S, ox, oy

    def w2t(self, x, y):
        S, ox, oy = self._top_tf()
        return ox + y * S, oy + x * S

    def t2w(self, px, py):
        S, ox, oy = self._top_tf()
        return (py - oy) / S, (px - ox) / S

    def _rect(self, b):
        x0, y0 = self.w2t(b["pos"][0] - b["size"][0] / 2, b["pos"][1] - b["size"][1] / 2)
        x1, y1 = self.w2t(b["pos"][0] + b["size"][0] / 2, b["pos"][1] + b["size"][1] / 2)
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    def _side_range(self):
        return (-0.45, 1.00) if self.v_side.get() == "XZ" else (-0.85, 0.85)      # V3.24: more room behind

    SIDE_GUTTER = 150              # right column for the height-line labels (V3.18)

    def _side_z(self):
        return (-0.06, max(0.6, self.fence()["z_max"]) + 0.06)

    def _side_tf(self):
        h0, h1 = self._side_range()
        z0, z1 = self._side_z()
        plot_w = self.SW - 44 - self.SIDE_GUTTER - 8
        S = min(plot_w / (h1 - h0), (self.SH - 16) / (z1 - z0))
        ox = 44 - h0 * S
        oz = self.SH - 8 + z0 * S
        return S, ox, oz

    def w2s(self, h, z):
        S, ox, oz = self._side_tf()
        return ox + h * S, oz - z * S

    def _h(self, p):
        """side-view horizontal coordinate: XZ -> forward distance (−x); YZ -> +y"""
        return -p[0] if self.v_side.get() == "XZ" else p[1]

    def arm_points(self, q_rad, fence=None):
        f = dict(fence or self.fence(), points=["link2", "link4", "link5", "link6", "link7", "tool"])
        sf = self.safety()
        pts = CS.fence_points(self._urdf(), self.jpanel.model["joints"], q_rad, self._base(), f, sf["tool_T"])
        out = [("base", (0.0, 0.0, 0.0))]
        if f.get("rule") == "A":                   # V3.23: the whole sampled chain, in order
            return out + list(pts.items())
        for n in ("link2", "link4", "link5", "link6", "link7", "tool"):
            if n in pts:
                out.append((n, pts[n]))
        return out

    # ================================================================ drawing
    def redraw(self):
        try:
            self._draw_top()
            self._draw_side()
        except Exception as exc:
            if getattr(self, "_draw_err", None) != str(exc):
                self._draw_err = str(exc)
                self.log(f"[WARN] [V3.17 safety] draw: {exc}")

    def _arm_state(self):
        model = self.jpanel.model
        act = self.pipe.actual_arm_deg(model)
        tgt = None
        if act is not None:
            t = self.jpanel.targets_deg()
            if max(abs(a - b) for a, b in zip(act, t)) > 0.3:
                tgt = t
        return act, tgt

    def _draw_top(self):
        c = self.cv_top
        c.delete("all")
        f = self.fence()
        S, ox, oy = self._top_tf()
        x_lo, x_hi = self.TOP_X
        y_lo, y_hi = self.TOP_Y
        # table + grid (lines only; tick labels go to the margins at the end)
        for b in self.ws["boxes"]:
            if b["name"] == "table":
                c.create_rectangle(*self._rect(b), fill=COL["table"], outline=COL["table_line"])
        for k in range(int(round(x_lo * 10)), int(round(x_hi * 10)) + 1):
            a, b_ = self.w2t(k / 10.0, y_lo)
            a2, b2 = self.w2t(k / 10.0, y_hi)
            c.create_line(a, b_, a2, b2, fill="#bdbdbd" if k == 0 else COL["grid"])
        for k in range(int(round(y_lo * 10)), int(round(y_hi * 10)) + 1):
            a, b_ = self.w2t(x_lo, k / 10.0)
            a2, b2 = self.w2t(x_hi, k / 10.0)
            c.create_line(a, b_, a2, b2, fill="#bdbdbd" if k == 0 else COL["grid"])
        c.create_oval(ox - 0.85 * S, oy - 0.85 * S, ox + 0.85 * S, oy + 0.85 * S, outline="#9e9e9e", dash=(2, 4))
        rx, ry = self.w2t(-0.62, -0.62)
        c.create_text(rx, ry, text="可达 ≈ 850 mm", fill="#888888", font=("TkDefaultFont", 8))
        if f.get("rule") != "A":                   # rule B: base disc is part of the zone
            r = f["base_radius"] * S
            c.create_oval(ox - r, oy - r, ox + r, oy + r, outline="#757575", dash=(4, 3), fill="#eeeeee")
        # zone + inner margin
        pts = [self.w2t(x, y) for x, y in f["polygon"]]
        c.create_polygon(*[v for p in pts for v in p], fill=COL["zone_fill"], outline=COL["zone"], width=2.5)
        # A: dashed line = where the link centre lines may go (link radius + margin from the walls)
        ins = CS.inset_polygon(f["polygon"], f["margin"] + (f.get("link_radius", 0.05) if f.get("rule") == "A" else 0.0))
        fi = [v for p in (self.w2t(x, y) for x, y in ins) for v in p]
        if len(fi) >= 6:
            c.create_polygon(*fi, fill="", outline=COL["zone"], dash=(6, 4))
        # boxes (non-table); resize handle = bottom-right corner on screen
        for i, b in enumerate(self.ws["boxes"]):
            if b["name"] == "table":
                continue
            x0, y0, x1, y1 = self._rect(b)
            selb = self.sel == ("box", i)
            c.create_rectangle(x0, y0, x1, y1, fill=COL["box_fill"], outline=COL["sel"] if selb else COL["box"],
                               width=3 if selb else 1.5)
            c.create_text((x0 + x1) / 2, y0 - 7, text=f"{b['name']}（高 {(b['pos'][2] + b['size'][2] / 2) * 1000:.0f} mm）",
                          fill="#b34700", font=("TkDefaultFont", 8))
            c.create_rectangle(x1 - 4, y1 - 4, x1 + 4, y1 + 4, fill=COL["box"], outline="")
        # marks + vertex handles; labels pushed outward from the zone centre
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        for i, (px, py) in enumerate(pts):
            cm = self.ws["corners"][i] if i < len(self.ws["corners"]) else {"name": f"M{i + 1}"}
            c.create_line(px - 9, py, px + 9, py, fill=COL["mark"], width=2)
            c.create_line(px, py - 9, px, py + 9, fill=COL["mark"], width=2)
            if self.mode == "edit":
                c.create_rectangle(px - 6, py - 6, px + 6, py + 6, fill=COL["sel"] if self.sel == ("v", i) else "white",
                                   outline=COL["zone"], width=2)
            dx, dy = px - cx, py - cy
            n = math.hypot(dx, dy) or 1.0
            c.create_text(px + dx / n * 20, py + dy / n * 16,
                          text=cm["name"] + (" ✎" if cm.get("source") == "编辑" else " ✓" if cm.get("source") == "示教" else ""),
                          fill=COL["zone"], font=("TkDefaultFont", 9, "bold"))
        for k, tp in enumerate(self.teach_pts):
            px, py = self.w2t(tp["xyz"][0], tp["xyz"][1])
            c.create_oval(px - 5, py - 5, px + 5, py + 5, outline="#1565c0", width=2)
            c.create_text(px + 8, py + 10, text=f"T{k + 1}", fill="#1565c0", font=("TkDefaultFont", 8, "bold"), anchor="w")
        # base + axes: forward (−x) up, +y right
        c.create_oval(ox - 13, oy - 13, ox + 13, oy + 13, fill="#555555", outline="")
        c.create_text(ox, oy, text="NERO", fill="white", font=("TkDefaultFont", 7))
        c.create_line(ox, oy - 14, ox, oy - 60, fill="#c62828", width=2, arrow="last")
        c.create_text(ox + 6, oy - 62, anchor="w", text="前方（−x）", fill="#c62828", font=("TkDefaultFont", 9, "bold"))
        c.create_line(ox + 14, oy, ox + 60, oy, fill="#2e7d32", width=2, arrow="last")
        c.create_text(ox + 62, oy + 10, anchor="w", text="+y", fill="#2e7d32", font=("TkDefaultFont", 9, "bold"))
        # arm + ghost
        act, tgt = self._arm_state()
        if act is not None:
            self._draw_arm(c, self.arm_points([math.radians(a) for a in act]), False, self.w2t)
        if tgt is not None:
            self._draw_arm(c, self.arm_points([math.radians(a) for a in tgt]), True, self.w2t)
        # margins with tick labels (drawn last, never covered by the zone / base disc)
        c.create_rectangle(0, 0, self.M_L, self.TH, fill="white", outline="")
        c.create_rectangle(0, self.TH - self.M_B, self.TW, self.TH, fill="white", outline="")
        c.create_rectangle(0, 0, self.TW, self.M_T, fill="white", outline="")
        c.create_line(self.M_L, self.M_T, self.M_L, self.TH - self.M_B, fill="#999999")
        c.create_line(self.M_L, self.TH - self.M_B, self.TW, self.TH - self.M_B, fill="#999999")
        for k in range(int(round(x_lo * 10)), int(round(x_hi * 10)) + 1):
            _a, yy = self.w2t(k / 10.0, 0)
            if self.M_T <= yy <= self.TH - self.M_B:
                c.create_text(self.M_L - 3, yy, text=f"{k * 10}", anchor="e", fill="#c62828" if k == 0 else "#777777",
                              font=("TkDefaultFont", 7))
        for k in range(int(round(y_lo * 10)), int(round(y_hi * 10)) + 1, 1):
            xx, _b = self.w2t(0, k / 10.0)
            if self.M_L <= xx <= self.TW and k % 2 == 0:
                c.create_text(xx, self.TH - self.M_B + 10, text=f"{k * 10}", fill="#2e7d32" if k == 0 else "#777777",
                              font=("TkDefaultFont", 7))
        c.create_text(4, self.M_T - 6, anchor="w", text="x cm", fill="#c62828", font=("TkDefaultFont", 7, "bold"))
        c.create_text(self.TW - 4, self.TH - 4, anchor="se", text="y cm", fill="#2e7d32", font=("TkDefaultFont", 7, "bold"))
        c.create_text(self.M_L + 4, 13, anchor="w",
                      text=f"俯视图（从机械臂后方往前看：上 = 前方 −x，右 = +y，网格 10 cm）· "
                           f"{ {'edit': '编辑', 'teach': '示教', 'inspect': '巡检'}[self.mode] }",
                      font=("TkDefaultFont", 9, "bold"))

    def _draw_arm(self, c, ap, ghost, proj, side=False):
        f = self.safety()["fence"] or dict(self.fence(), enabled=False)
        line = []
        for name, p in ap:
            line += list(proj(self._h(p), p[2]) if side else proj(p[0], p[1]))
        c.create_line(*line, fill=COL["ghost"] if ghost else COL["arm"], width=2 if ghost else 4,
                      dash=(6, 4) if ghost else None, capstyle="round")
        for name, p in ap:
            if name in ("base", "link2"):
                continue
            px, py = proj(self._h(p), p[2]) if side else proj(p[0], p[1])
            bad = bool(CS.classify_point(name, p, f)[0]) if f.get("enabled", True) else False
            warn = bool(CS.classify_point(name, p, f)[1]) if f.get("enabled", True) else False
            col = COL["bad"] if bad else (COL["warn"] if warn else (COL["ghost_pt"] if ghost else COL["arm_pt"]))
            if "#" in name or (name not in PT_LABEL and name != "link3"):
                if bad or warn:                     # V3.23 rule-A samples: only shown when near / over a wall
                    c.create_oval(px - 3, py - 3, px + 3, py + 3, fill=col, outline="")
                continue
            r = 6 if name == "tool" else 4.5
            c.create_oval(px - r, py - r, px + r, py + r, fill=col, outline="#7f0000" if bad else "#0d47a1")
            if not side and (not ghost or bad):
                c.create_text(px + 9, py, text=PT_LABEL.get(name, name), anchor="w",
                              fill=COL["bad"] if bad else "#0d47a1", font=("TkDefaultFont", 8, "bold" if bad else "normal"))

    def _draw_side(self):
        """V3.18: no text inside the plot; line names + values in the right gutter (de-overlapped, leader lines);
        colour meanings in the static legend below the canvas."""
        c = self.cv_side
        c.delete("all")
        f = self.fence()
        S, ox, oz = self._side_tf()
        xz = self.v_side.get() == "XZ"
        h0, h1 = self._side_range()
        z0, z1 = self._side_z()
        zt, zmin, zmax = f["z_table"], f["z_min_points"], f["z_max"]
        rule_a = f.get("rule") == "A"
        zf = zmax if rule_a else zt + f["xy_free_height"]          # V3.23 A: the wall goes up to z_max
        L, R = self.w2s(h0, 0)[0], self.w2s(h1, 0)[0]
        top, bot = self.w2s(0, z1)[1], self.w2s(0, z0)[1]
        self.v_side_hint.set("右 = 前方（−x），竖轴 = 高度 mm" if xz else "右 = +y，竖轴 = 高度 mm")
        for k in range(int(round(h0 * 10)), int(round(h1 * 10)) + 1):
            a, _b = self.w2s(k / 10.0, 0)
            c.create_line(a, top, a, bot, fill="#bdbdbd" if k == 0 else COL["grid"])
        for k in range(0, int(z1 * 10) + 1):
            _a, y = self.w2s(0, k / 10.0)
            c.create_line(L, y, R, y, fill=COL["grid"])
            if k % 2 == 0:
                c.create_text(L - 4, y, text=f"{k * 100}", anchor="e", fill="#999999", font=("TkDefaultFont", 7))
        y_max, y_free = self.w2s(0, zmax)[1], self.w2s(0, zf)[1]
        y_min, y_tab = self.w2s(0, zmin)[1], self.w2s(0, zt)[1]
        c.create_rectangle(L, y_max, R, y_free, fill="#eef6ee", outline="")              # XY free band
        hs = [(-p[0] if xz else p[1]) for p in f["polygon"]]
        a, _b = self.w2s(min(hs), 0)
        a2, _b = self.w2s(max(hs), 0)
        c.create_rectangle(a, y_free, a2, y_min, fill=COL["zone_fill"], outline=COL["zone"], width=2)
        c.create_rectangle(L, y_min, R, y_tab, fill="#f3b6b6", outline="", stipple="gray50")   # anti-collision band
        for b in self.ws["boxes"]:
            hc = -b["pos"][0] if xz else b["pos"][1]
            hw = b["size"][0] if xz else b["size"][1]
            a, b_ = self.w2s(hc - hw / 2, b["pos"][2] + b["size"][2] / 2)
            a2, b2 = self.w2s(hc + hw / 2, b["pos"][2] - b["size"][2] / 2)
            if b["name"] == "table":
                c.create_rectangle(L, b_, R, min(b2, bot), fill=COL["table"], outline=COL["table_line"])
            else:
                c.create_rectangle(a, b_, a2, b2, fill=COL["box_fill"], outline=COL["box"])
        bx, bz = self.w2s(0, zt)
        c.create_rectangle(bx - 10, bz - 0.138 * S, bx + 10, bz, fill="#555555", outline="")
        items = [("z_max", y_max, COL["zone"], (7, 4), f"z 上限 {zmax * 1000:.0f}" + (" · 围墙顶" if rule_a else "")),
                 ("xy_free_height", y_free, "#6d4c41", (2, 3), f"XY 限高 {f['xy_free_height'] * 1000:.0f}"),
                 ("z_min_points", y_min, COL["bad"], (7, 4), f"z_min {zmin * 1000:.0f}"),
                 ("z_table", y_tab, COL["table_line"], None, f"桌面 {zt * 1000:.0f}")]
        if rule_a:
            items = [it for it in items if it[0] != "xy_free_height"]
        for key, y, col, dash, _lab in items:
            c.create_line(L, y, R, y, fill=col, width=1 if key == "xy_free_height" else 2, dash=dash)
            c.create_rectangle(R - 5, y - 5, R + 5, y + 5, fill="white", outline=col, width=2)
        act, tgt = self._arm_state()
        if act is not None:
            self._draw_arm(c, self.arm_points([math.radians(a) for a in act]), False, self.w2s, side=True)
        if tgt is not None:
            self._draw_arm(c, self.arm_points([math.radians(a) for a in tgt]), True, self.w2s, side=True)
        # gutter labels, pushed apart so they never overlap
        font = ("TkDefaultFont", 8, "bold")
        try:
            gap = int(c.tk.call("font", "metrics", font, "-linespace")) + 1
        except Exception:
            gap = 14
        c.create_rectangle(R + 6, 0, self.SW, self.SH, fill="white", outline="")
        order = sorted(items, key=lambda it: it[1])                 # top to bottom
        ys = [it[1] for it in order]
        for k in range(1, len(ys)):
            ys[k] = max(ys[k], ys[k - 1] + gap)
        over = ys[-1] - (self.SH - gap / 2)
        if over > 0:
            ys = [y - over for y in ys]
            for k in range(len(ys) - 2, -1, -1):
                ys[k] = min(ys[k], ys[k + 1] - gap)
        for (key, y, col, _d, lab), yl in zip(order, ys):
            c.create_line(R + 6, y, R + 18, yl, fill=col)
            c.create_text(R + 21, yl, anchor="w", text=lab + " mm", fill=col, font=font)

    # ================================================================ mouse: top view
    def _hit_top(self, ex, ey):
        f = self.fence()
        for i, (x, y) in enumerate(f["polygon"]):
            px, py = self.w2t(x, y)
            if math.hypot(ex - px, ey - py) <= 9:
                return ("v", i)
        for i, b in enumerate(self.ws["boxes"]):
            if b["name"] == "table":
                continue
            x0, y0, x1, y1 = self._rect(b)
            if abs(ex - x1) <= 7 and abs(ey - y1) <= 7:
                return ("box_rs", i)
        for i, b in enumerate(self.ws["boxes"]):
            if b["name"] == "table":
                continue
            x0, y0, x1, y1 = self._rect(b)
            if x0 <= ex <= x1 and y0 <= ey <= y1:
                return ("box", i)
        n = len(f["polygon"])
        for i in range(n):
            a = self.w2t(*f["polygon"][i])
            b = self.w2t(*f["polygon"][(i + 1) % n])
            if CS.seg_dist(ex, ey, a, b) <= 6:
                return ("edge", i)
        for i, b in enumerate(self.ws["boxes"]):
            if b["name"] == "table":
                x0, y0, x1, y1 = self._rect(b)
                if x0 <= ex <= x1 and y0 <= ey <= y1:
                    return ("box", i)
        return None

    def on_top_press(self, e):
        self.cv_top.focus_set()
        if self.mode != "edit":
            return
        h = self._hit_top(e.x, e.y)
        self.drag = {"hit": h, "start": (e.x, e.y), "snap": self._snapshot(), "moved": False,
                     "orig": json.loads(json.dumps(self.ws))}
        if h is None:
            self._select(None)
        elif h[0] in ("v", "box"):
            self._select(h)
        elif h[0] == "box_rs":
            self._select(("box", h[1]))
        else:
            self._select(None)

    def on_top_drag(self, e):
        d = self.drag
        if not d or not d["hit"] or self.mode != "edit":
            return
        S = self._top_tf()[0]
        dx_w = (e.y - d["start"][1]) / S             # screen down = +x (backwards)
        dy_w = (e.x - d["start"][0]) / S             # screen right = +y
        o = d["orig"]
        kind, i = d["hit"]
        rnd = lambda v: round(v, 3)
        if kind == "v":
            x, y = o["fence"]["polygon"][i]
            self.fence()["polygon"][i] = [rnd(x + dx_w), rnd(y + dy_w)]
            self.ws["corners"][i]["source"] = "编辑"
            self.ws["corners"][i]["q"] = None
        elif kind == "edge":
            n = len(o["fence"]["polygon"])
            for j in (i, (i + 1) % n):
                x, y = o["fence"]["polygon"][j]
                self.fence()["polygon"][j] = [rnd(x + dx_w), rnd(y + dy_w)]
                self.ws["corners"][j]["source"] = "编辑"
                self.ws["corners"][j]["q"] = None
        elif kind == "box":
            b0 = o["boxes"][i]
            self.ws["boxes"][i]["pos"] = [rnd(b0["pos"][0] + dx_w), rnd(b0["pos"][1] + dy_w), b0["pos"][2]]
        elif kind == "box_rs":
            b0 = o["boxes"][i]
            # handle = screen bottom-right = the (+x, +y) corner; the (−x, −y) corner stays fixed
            fx, fy = b0["pos"][0] - b0["size"][0] / 2, b0["pos"][1] - b0["size"][1] / 2
            hx, hy = b0["pos"][0] + b0["size"][0] / 2 + dx_w, b0["pos"][1] + b0["size"][1] / 2 + dy_w
            sx, sy = max(0.01, hx - fx), max(0.01, hy - fy)
            self.ws["boxes"][i]["size"] = [rnd(sx), rnd(sy), b0["size"][2]]
            self.ws["boxes"][i]["pos"] = [rnd(fx + sx / 2), rnd(fy + sy / 2), b0["pos"][2]]
        d["moved"] = True
        self._fill_sel()
        self.redraw()

    def on_top_release(self, _e):
        d = self.drag
        self.drag = None
        if d and d["moved"]:
            self._push_undo(d["snap"])
            self._changed("drag")

    def on_top_double(self, e):
        if self.mode != "edit":
            return
        f = self.fence()
        n = len(f["polygon"])
        best = None
        for i in range(n):
            a = self.w2t(*f["polygon"][i])
            b = self.w2t(*f["polygon"][(i + 1) % n])
            d = CS.seg_dist(e.x, e.y, a, b)
            if d <= 10 and (best is None or d < best[0]):
                best = (d, i)
        if best is None:
            return
        snap = self._snapshot()
        x, y = self.t2w(e.x, e.y)
        i = best[1] + 1
        f["polygon"].insert(i, [round(x, 3), round(y, 3)])
        names = {c_["name"] for c_ in self.ws["corners"]}
        k = n + 1
        while f"M{k}" in names:
            k += 1
        self.ws["corners"].insert(i, {"name": f"M{k}", "source": "编辑", "q": None})
        self._push_undo(snap)
        self._select(("v", i))
        self._changed("insert")

    def on_top_right(self, e):
        if self.mode != "edit":
            return
        h = self._hit_top(e.x, e.y)
        if h and h[0] == "v":
            self.sel = h
            self.on_sel_delete()

    # ================================================================ mouse: side view
    def on_side_press(self, e):
        if self.mode != "edit":
            return
        f = self.fence()
        hit = None
        for key in ("z_min_points", "z_max", "z_table", "xy_free_height"):
            _x, y = self.w2s(0, f[key] + (f["z_table"] if key == "xy_free_height" else 0.0))
            if abs(e.y - y) <= 6:
                hit = key
                break
        self.sdrag = {"key": hit, "snap": self._snapshot(), "moved": False} if hit else None

    def on_side_drag(self, e):
        d = getattr(self, "sdrag", None)
        if not d:
            return
        S, ox, oz = self._side_tf()
        z = round((oz - e.y) / S, 3)
        f = self.fence()
        key = d["key"]
        if key == "xy_free_height":
            f[key] = round(max(0.03, z - f["z_table"]), 3)
        elif key == "z_table":
            delta = z - f["z_table"]
            f["z_table"] = z
            f["z_min_points"] = round(f["z_min_points"] + delta, 3)
            self._table_to(z)
        else:
            f[key] = z
        d["moved"] = True
        self._fill_fields()
        self.redraw()

    def on_side_release(self, _e):
        d = getattr(self, "sdrag", None)
        self.sdrag = None
        if d and d["moved"]:
            self._push_undo(d["snap"])
            self._changed("side")

    # ================================================================ periodic (main thread)
    def poll(self):
        self._drain_ui()
        tm = bool(getattr(self.pipe, "teach_mode", False))
        import NERO_V313_Real_Robot as _RR
        cm = _RR.ctrl_mode_of(((self.pipe.mon_status or {}).get("arm_status")))
        cms = "" if cm is None else f" · ctrl_mode=0x{cm:02X}"
        hw_teach = _RR.is_teach_ctrl_mode(cm)
        la = bool(getattr(self.pipe, "leader_active", lambda: False)())
        self.v_teach_mode.set(("NERO：示教模式（手动拖动）" if tm else
                               "NERO：机械臂报告示教模式" if hw_teach else
                               "NERO：检测到手动拖动" if la else "NERO：正常控制") + cms)
        self.lbl_teach_mode.configure(fg="#1565c0" if (tm or hw_teach or la) else "#2e7d32")
        mon = self.pipe.monitor
        if mon.running() and self.pipe.lamp("moveit") == "green" and mon.proc is not None and \
                self.scene_applied_for != mon.proc.pid and not self._auto_pending:
            self._auto_pending = True
            self.scene_applied_for = mon.proc.pid
            self.log("[INFO] [V3.16 safety] 第 6 步绿灯 → 自动把工作区碰撞体应用到 MoveIt")
            self.on_apply()
            self._auto_pending = False
        if self.drag is None and getattr(self, "sdrag", None) is None:
            self.redraw()
        self._status()

    def _status(self):
        model = self.jpanel.model
        act, tgt = self._arm_state()
        sf = self.safety()
        f = sf["fence"]
        if f is None:
            self.v_now.set("围栏关闭 ⚠（只剩 MoveIt 碰撞体）")
            self.lbl_now.configure(fg="#c62828", bg="#fdecea")
            return
        if act is None:
            self.v_now.set("等待关节反馈")
            self.lbl_now.configure(fg="#555555", bg="#f0f0f0")
            self.v_tip.set(("没有新鲜的关节角（记录角点会失败）：" + self.pipe.feedback_diag()) if self.mode == "teach" else "")
            return
        urdf, base = self._urdf(), self._base()
        pts = CS.fence_points(urdf, model["joints"], [math.radians(a) for a in act], base, f, sf["tool_T"])
        v, w = CS.fence_violations(pts, f), CS.fence_warnings(pts, f)
        corners = self.valid_ws.get("corners")
        lines = []
        if v:
            lines.append("× 当前姿态越界：" + CS.violation_text(v, f, corners))
            col, bg = "#b71c1c", "#fdecea"
        elif w:
            lines.append("● 当前姿态在安全区内，但在余量带：" + "; ".join(
                f"{x['point']} 离 {CS.edge_name(x['edge'], len(f['polygon']), corners)} 边 {x['dist'] * 1000:.0f} mm" for x in w[:3]))
            col, bg = "#7a5b00", "#fff6d6"
        else:
            lines.append("● 当前姿态在安全区内")
            col, bg = "#1b5e20", "#e3f4e5"
        if pts:
            low = min(pts.items(), key=lambda kv: kv[1][2])
            rule_a = f.get("rule") == "A"
            ins = []
            for n, p in pts.items():
                if CS.point_in_polygon(p[0], p[1], f["polygon"]):
                    d, e = CS.edge_distance(p[0], p[1], f["polygon"])
                    ins.append((n, (d - (CS.point_radius(n, f) if rule_a else 0.0), e)))
            near = min(ins, key=lambda t: t[1][0]) if ins else None
            lines[-1] += f" · 最低 {PT_LABEL.get(low[0], low[0])} z={low[1][2] * 1000:.0f} mm" + \
                ((f" · 离墙最近 {near[1][0] * 1000:.0f} mm（{PT_LABEL.get(near[0], near[0])}，"
                  f"{CS.edge_name(near[1][1], len(f['polygon']), corners)} 墙，已扣半径）") if near and rule_a else
                 (f" · 距边界最近 {near[1][0] * 1000:.0f} mm（{CS.edge_name(near[1][1], len(f['polygon']), corners)} 边）" if near else ""))
        if tgt is not None:
            ok, vt, _p = CS.fence_check(urdf, model["joints"], [math.radians(a) for a in tgt], base, f, sf["tool_T"])
            lines.append("目标（紫色虚线）：" + ("在安全区内 ✓" if ok else "越界 → 关节控制页「执行」已锁定：" + CS.violation_text(vt, f, corners)))
            if not ok:
                col, bg = "#b71c1c", "#fdecea"
        if self.mode in ("teach", "inspect"):
            lines.append(f"{'示教' if self.mode == 'teach' else '巡检'}模式：速度 ≤ {TEACH_MAX_SPEED * 100:.0f}%" +
                         ("，工具点可低于桌面 5 mm" if self.mode == "teach" else "，工具点可降到桌面上方 10 mm")
                         + ("，工具尖/夹爪可贴到墙线，连杆仍保持离墙距离" if f.get("rule") == "A"
                            else "，安全区 XY 只限制工具点（z 限制照常）"))
        hand = bool(getattr(self.pipe, "teach_mode", False) or getattr(self.pipe, "leader_active", lambda: False)())
        if hand and v and not getattr(self, "_wall_alarm", False):      # V3.23: hand guiding over a wall -> alarm
            try:
                self.cv_top.bell()
            except Exception:
                pass
            self.log("[WARN] [V3.23 safety] 手动拖动越界：" + CS.violation_text(v, f, corners))
        self._wall_alarm = bool(hand and v)
        if hand and v:
            lines.insert(0, "⚠ 手动拖动越界！请把机械臂拖回安全区")
        self.v_now.set("\n".join(lines))
        self.lbl_now.configure(fg=col, bg=bg)
        if self.mode == "teach" and "tool" in pts:
            p = pts["tool"]
            k = len(self.teach_pts)
            ref = ""
            if k < len(f["polygon"]):
                d = math.hypot(p[0] - f["polygon"][k][0], p[1] - f["polygon"][k][1])
                ref = f" · 离当前 M{k + 1} 位置 {d * 1000:.0f} mm"
            src = self.pipe.feedback_source()[0].replace("ROS ", "").replace("get_", "")
            self.v_tip.set(f"尖端 ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m · 桌面上方 {(p[2] - f['z_table']) * 1000:.1f} mm{ref}"
                           f" · 角度来源 {src}")
        elif self.mode == "teach":
            self.v_tip.set("工具点关闭（mode=none）→ 无法示教，请在编辑页设为 pointer")

    # ================================================================ teach
    def _tool_tip(self):
        model = self.jpanel.model
        act = self.pipe.actual_arm_deg(model)
        if act is None:
            return None, None
        q = [math.radians(a) for a in act]
        T = CS.tool_pose(self._urdf(), model["joints"], q, self._base(), self.safety()["tool_T"])
        if T is None or self.safety()["tool_T"] is None:
            return None, q
        return [T[0][3], T[1][3], T[2][3]], q

    def on_teach_record(self):
        p, q = self._tool_tip()
        if p is None:
            if self.safety()["tool_T"] is None and q is not None:
                self.log("[WARN] [V3.21 teach] 工具点关闭（mode=none）→ 在「选择/编辑」里把工具点设为 pointer 或 sim_tcp")
            else:
                self.log("[WARN] [V3.21 teach] 没有新鲜的关节角，角点未记录：" + self.pipe.feedback_diag())
            return
        name = f"M{len(self.teach_pts) + 1}"
        self.teach_pts.append({"name": name, "xyz": p, "q": q})
        self.tv_teach.insert("", "end", iid=name, values=(name, f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"))
        self.log(f"[OK] [V3.16 teach] 记录 {name}: ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m")
        self._teach_info()
        self.redraw()

    def on_teach_undo(self):
        if self.teach_pts:
            tp = self.teach_pts.pop()
            self.tv_teach.delete(tp["name"])
            self._teach_info()
            self.redraw()

    def on_teach_clear(self):
        for tp in self.teach_pts:
            self.tv_teach.delete(tp["name"])
        self.teach_pts = []
        self._teach_info()
        self.redraw()

    def _teach_info(self):
        if not self.teach_pts:
            self.v_teach_info.set("")
            return
        zs = [tp["xyz"][2] for tp in self.teach_pts]
        self.v_teach_info.set(f"已记录 {len(zs)} 个 · 桌面高度（平均）{sum(zs) / len(zs) * 1000:.1f} mm · 平整度（最大−最小）"
                              f"{(max(zs) - min(zs)) * 1000:.1f} mm" + ("" if len(zs) >= 3 else " · 至少 3 个才能生成"))

    def on_teach_generate(self):
        if len(self.teach_pts) < 3:
            self.log("[WARN] [V3.16 teach] 至少记录 3 个角点")
            return
        pts = [tp for tp in self.teach_pts]
        poly = [[round(tp["xyz"][0], 4), round(tp["xyz"][1], 4)] for tp in pts]
        if CS.polygon_self_intersects(poly):
            order = CS.order_polygon([[*p, k] for k, p in enumerate(poly)])
            pts = [pts[int(o[2])] for o in order]
            poly = [[round(tp["xyz"][0], 4), round(tp["xyz"][1], 4)] for tp in pts]
            self.log("[INFO] [V3.16 teach] 记录顺序会自相交 → 已按绕中心的顺序重排角点")
        zs = [tp["xyz"][2] for tp in self.teach_pts]
        zt = round(sum(zs) / len(zs), 4)
        flat = max(zs) - min(zs)
        snap = self._snapshot()
        f = self.fence()
        dz = f["z_min_points"] - f["z_table"]
        f["polygon"] = poly
        f["z_table"] = zt
        f["z_min_points"] = round(zt + dz, 4)
        self._table_to(zt)
        self.ws["corners"] = [{"name": tp["name"], "source": "示教", "q": [round(v, 6) for v in tp["q"]],
                               "xyz": [round(v, 5) for v in tp["xyz"]]} for tp in pts]
        if f.get("rule") == "A" and not CS.base_inside(f):          # V3.23 premise: the zone contains the base
            try:
                self.ws = CS.merge_base(self.ws)
                self.log(f"[INFO] [V3.23 teach] 规则 A：标记围成的区域没包住底座 → 已自动并入底座（凸包，新角点 B1…，"
                         f"共 {len(self.fence()['polygon'])} 个角点）；底座后方 / 侧面的角点可再拖到墙前")
            except Exception as exc:
                self.log(f"[WARN] [V3.23 teach] 自动并入底座失败：{exc}")
            f = self.fence()
        self._push_undo(snap)
        self._fill_fields()
        self._changed("teach")
        msg = (f"[OK] [V3.16 teach] 安全区生成：{len(poly)} 个角点，桌面 z = {zt * 1000:.1f} mm，平整度 {flat * 1000:.1f} mm，"
               f"z_min = {f['z_min_points'] * 1000:.0f} mm（未保存）")
        self.log(msg if flat <= 0.01 else msg.replace("[OK]", "[WARN]") + " ⚠ 平整度 > 10 mm：桌面不平或尖端没贴到桌面")
        self.on_teach_clear()

    def on_teach_ref(self):
        p, _q = self._tool_tip()
        try:
            ref = float(self.v_ref.get()) / 1000.0
        except ValueError:
            return
        if p is None:
            return
        d = math.hypot(p[0], p[1])
        txt = f"参考点 R：示教 {d * 1000:.1f} mm vs 尺量 {ref * 1000:.0f} mm → 差 {(d - ref) * 1000:+.1f} mm"
        ok = abs(d - ref) <= 0.005
        self.v_teach_info.set(txt + (" ✓" if ok else " ⚠ > 5 mm：检查工具长度 / 零位 / 尺量起点"))
        self.log(("[OK] " if ok else "[WARN] ") + "[V3.16 teach] " + txt)

    # ================================================================ V3.19 teach mode (NERO zero-force)
    def on_teach_enter(self):
        if self.pipe.move_busy():
            self.log("[WARN] [V3.19 teach] 正在运动，稍后再进入示教模式")
            return
        if not self.ask("进入示教模式", "NERO 将进入示教（主导臂 / 联动示教输入）模式，之后可以用手带动机械臂。\n"
                                     "驱动的控制门会先关闭。请先用手扶住机械臂（尤其是前臂和末端），确认周围无人。继续？"):
            return
        self.v_mode.set("teach")
        self.on_mode()
        self._run("teach_enter", lambda: self.pipe.set_teach_mode(True))

    def on_teach_exit(self):
        if not self.ask("退出示教模式", "NERO 将切回正常控制并重新使能（先保持当前位置），然后以 10% 速度经 MoveIt "
                                     "回零位（带安全区检查）。\n请扶稳机械臂、手在急停上。继续？"):
            return
        if self.mode == "teach":                   # leaving teach: the normal fence applies again (no 5 % cap)
            self._teach_exiting = True
            self.v_mode.set("edit")
            self.on_mode()
            self._teach_exiting = False
        jp = getattr(self, "jpanel", None)
        model = getattr(jp, "model", None) if jp is not None else None
        self._run("teach_exit", lambda: self.pipe.set_teach_mode(False, home_model=model, home_fn=self.exit_home))

    def exit_home(self, model, speed):
        """V3.22: after hand guiding the tool is often on the table at a zone corner; a direct joint move to zero
        would sweep it out of the zone close to the table. First lift the tool straight up (orientation kept) above
        xy_free_height, then go to zero. Both via MoveIt + fence check, at `speed` (10 %). -> result dict"""
        f = self.fence()
        toolT = self.safety()["tool_T"]
        a0 = self.pipe.actual_arm_deg(model)
        if a0 is None:
            return {"ok": False, "error": "没有当前关节角"}
        q0 = [math.radians(a) for a in a0]
        urdf, base = self._urdf(), self._base()
        if toolT is not None:
            T = CS.tool_pose(urdf, model["joints"], q0, base, toolT)
            z_up = f["z_table"] + f.get("xy_free_height", 0.15) + 0.03
            if T is not None and T[2][3] < z_up - 0.005:
                keep = [row[:3] for row in T[:3]]
                q1, err = CS.ik_tool(urdf, model["joints"], q0, base, toolT, [T[0][3], T[1][3], z_up], model["limits"],
                                     keep_R=keep)
                if q1 is not None and err < 0.005:
                    self.pipe.log(f"[INFO] [V3.22 teach] 先竖直抬起工具点到桌面上方 {(z_up - f['z_table']) * 1000:.0f} mm")
                    r1 = self.pipe.move_and_settle(model, q1, speed, label="退出示教 抬起")
                    if not r1.get("ok"):
                        return r1
                else:
                    self.pipe.log(f"[WARN] [V3.22 teach] 竖直抬起 IK 失败（{(err or 0) * 1000:.0f} mm）→ 直接回零位")
        return self.pipe.move_and_settle(model, [0.0] * 7, speed, label="退出示教回零")

    # ================================================================ V3.19 zone files
    def _zone_dir(self):
        d = os.path.dirname(self.path) if self.path and os.path.isdir(os.path.dirname(self.path)) else ZONE_DIR_DEFAULT
        try:
            os.makedirs(d, exist_ok=True)
        except Exception:
            pass
        return d

    def _remember(self, path):
        self.path = path
        save_state({"last_zone_file": os.path.abspath(path)})

    def on_open(self):
        from tkinter import filedialog
        fn = filedialog.askopenfilename(title="打开安全区文件", initialdir=self._zone_dir(),
                                        filetypes=[("NERO 安全区", "*.json"), ("所有文件", "*")])
        if not fn:
            return
        w = CS.load_workspace(fn)
        if w.get("load_error"):
            self.log(f"[FAIL] [V3.19 safety] 打不开 {fn}：{w['load_error']}")
            return
        snap = self._snapshot()
        self.ws = w
        self._push_undo(snap)
        self.valid_ws = json.loads(json.dumps(self.ws))
        self.dirty, self.invalid, self.sel = False, None, None
        self._remember(fn)
        self._fill_fields()
        self.redraw()
        self.log(f"[OK] [V3.19 safety] 已打开安全区 {fn}：{len(self.fence()['polygon'])} 个角点，{len(self.ws['boxes'])} 个碰撞体"
                 "（下次启动自动打开）")
        if self.pipe.monitor.running():
            self.on_apply()

    def on_save_as(self):
        from tkinter import filedialog
        fn = filedialog.asksaveasfilename(title="安全区另存为", initialdir=self._zone_dir(), defaultextension=".json",
                                          initialfile=f"安全区_{time.strftime('%Y%m%d_%H%M')}.json",
                                          filetypes=[("NERO 安全区", "*.json"), ("所有文件", "*")])
        if not fn:
            return
        self._save_to(fn)

    def _save_to(self, fn):
        try:
            w = json.loads(json.dumps(self.ws))
            w["name"] = os.path.splitext(os.path.basename(fn))[0]
            self.ws = CS.save_workspace(w, fn)
        except Exception as exc:
            self.log(f"[FAIL] [V3.19 safety] 保存失败（{fn}）：{exc}")
            return False
        self.valid_ws = json.loads(json.dumps(self.ws))
        self.dirty, self.invalid = False, None
        self._remember(fn)
        self._fill_fields()
        self.log(f"[OK] [V3.19 safety] 安全区已保存到 {fn}：{len(self.fence()['polygon'])} 个角点，{len(self.ws['boxes'])} 个碰撞体，"
                 f"围栏 {'启用' if self.fence()['enabled'] else '关闭'}（下次启动自动打开）")
        if self.pipe.monitor.running():
            self.on_apply()
        return True

    # ================================================================ inspect (worker)
    def on_stop(self):
        self.stop_flag = True
        self.pipe.cancel_move()

    def on_inspect(self):
        if not self._ready():
            return
        try:
            hov = float(self.v_hover.get()) / 1000.0
            assert 0.02 <= hov <= 0.20
        except Exception:
            self.log("[WARN] [V3.16 inspect] 悬停高度 20–200 mm")
            return
        for it in self.tv_insp.get_children():
            self.tv_insp.delete(it)
        self._run("inspect", lambda: self._inspect(hov))

    def _inspect(self, hov):
        model = self.jpanel.model
        w = json.loads(json.dumps(self.valid_ws))
        f = w["fence"]
        sf = self.safety()
        urdf, base, toolT = self._urdf(), self._base(), sf["tool_T"]
        if toolT is None:
            self.pipe.log("[WARN] [V3.16 inspect] 工具点关闭，无法巡检")
            return
        a0 = self.pipe.actual_arm_deg(model)
        if a0 is None:
            return
        start = [math.radians(a) for a in a0]
        if not self.ask("巡检", f"机械臂将依次停到 {len(f['polygon'])} 个角点上方 {hov * 1000:.0f} mm（速度 ≤ "
                               f"{TEACH_MAX_SPEED * 100:.0f}%），每个点请你确认指针是否对准标记。\n确认周围无障碍、手在急停上？"):
            return
        speed = min(TEACH_MAX_SPEED, max(0.01, float(self.cfg_fn().get("speed_percent", 5)) / 100.0))
        walls_off = f.get("rule") == "A" and f.get("walls_to_moveit", True) and bool(CS.wall_boxes(w))
        if walls_off:                               # V3.23: the pointer has to reach the wall line at the corners
            self.pipe.log("[INFO] [V3.23 inspect] 巡检期间 MoveIt 围墙暂时撤下（软件围栏照常：连杆离墙、工具尖可贴墙线）")
            self.on_apply(walls=False, wait=True)
        try:
            self._inspect_corners(model, w, f, urdf, base, toolT, start, hov, speed)
        finally:
            if walls_off:
                self.on_apply(wait=True)
                self.pipe.log("[INFO] [V3.23 inspect] MoveIt 围墙已恢复")

    def _inspect_corners(self, model, w, f, urdf, base, toolT, start, hov, speed):
        rows = []
        q_prev = start
        cx = sum(p[0] for p in f["polygon"]) / len(f["polygon"])
        cy = sum(p[1] for p in f["polygon"]) / len(f["polygon"])
        zc = f["z_table"] + max(hov, 0.10)
        for i, (x, y) in enumerate(f["polygon"]):
            if self.stop_flag:
                break
            cm = w["corners"][i] if i < len(w["corners"]) else {"name": f"M{i + 1}"}
            if cm.get("source") == "并入底座":       # V3.23: added around the base, no table mark there
                continue
            # corner-to-corner paths run along the boundary and bulge out: go through the zone centre first
            qc, ec = CS.ik_tool(urdf, model["joints"], q_prev, base, toolT, [cx, cy, zc], model["limits"])
            if qc is not None:
                rc = self.pipe.move_and_settle(model, qc, speed, label=f"巡检 中心（→ {cm['name']}）")
                if not rc.get("ok"):
                    self.pipe.log(f"[FAIL] [V3.16 inspect] 到中心失败: {rc.get('error')}")
                    break
                q_prev = qc
            seed = q_prev
            T = CS.tool_pose(urdf, model["joints"], seed, base, toolT)
            Tq = CS.tool_pose(urdf, model["joints"], cm["q"], base, toolT) if cm.get("q") else None
            keep = [row[:3] for row in Tq[:3]] if Tq is not None else None      # taught orientation, if any
            q, err = CS.ik_tool(urdf, model["joints"], seed, base, toolT, [x, y, f["z_table"] + hov], model["limits"],
                                keep_R=keep)
            if q is None:
                rows.append({"corner": cm["name"], "ik": "失败", "result": "跳过"})
                self._ui(lambda c_=cm["name"], e=err: self.tv_insp.insert("", "end", values=(c_, f"{hov * 1000:.0f} mm", f"失败 {e * 1000:.0f} mm", "跳过")))
                self.pipe.log(f"[WARN] [V3.16 inspect] {cm['name']}: IK 失败（{err * 1000:.0f} mm）→ 跳过")
                continue
            res = self.pipe.move_and_settle(model, q, speed, label=f"巡检 {cm['name']} 上方 {hov * 1000:.0f} mm")
            if not res.get("ok"):
                self.pipe.log(f"[FAIL] [V3.16 inspect] {cm['name']}: {res.get('error')}")
                rows.append({"corner": cm["name"], "ik_mm": err * 1000, "result": "运动失败", "error": res.get("error")})
                self._ui(lambda c_=cm["name"]: self.tv_insp.insert("", "end", values=(c_, f"{hov * 1000:.0f} mm", "-", "运动失败")))
                break
            q_prev = q
            ok = self.ask(f"巡检 {cm['name']}", f"机械臂停在 {cm['name']} 上方 {hov * 1000:.0f} mm。\n指针尖端是否正对桌面标记十字的中心？")
            rows.append({"corner": cm["name"], "ik_mm": round(err * 1000, 2), "result": "对准" if ok else "偏了"})
            self._ui(lambda c_=cm["name"], e=err, o=ok: self.tv_insp.insert("", "end", values=(
                c_, f"{hov * 1000:.0f} mm", f"{e * 1000:.2f} mm", "对准 ✓" if o else "偏了 ⚠")))
            self.pipe.log(f"{'[OK]' if ok else '[WARN]'} [V3.16 inspect] {cm['name']}: {'对准' if ok else '偏了'}")
        if self.ask("巡检", "回到巡检前的姿态？"):
            self.pipe.move_and_settle(model, start, speed, label="巡检 回起点")
        allok = rows and all(r["result"] == "对准" for r in rows)
        path = CS.write_result("inspect", {"hover_m": hov, "rows": rows, "all_ok": bool(allok)})
        self.pipe.log(f"{'[OK]' if allok else '[WARN]'} [V3.16 inspect] 完成 → {os.path.basename(path)}")

    # ================================================================ file / MoveIt
    def on_reload(self):
        snap = self._snapshot()
        self.ws = CS.load_workspace(self.path)
        self._push_undo(snap)
        self.valid_ws = json.loads(json.dumps(self.ws))
        self.dirty = False
        self.invalid = None
        self.sel = None
        self._fill_fields()
        self.redraw()
        self.log("[INFO] [V3.16 safety] 已重新载入 config/real_workspace.json")

    def on_save(self):
        self._save_to(self.path)

    def on_merge_base(self):
        """V3.23: rule A needs the base inside the zone -> convex hull of the zone and a square around the base."""
        snap = self._snapshot()
        try:
            w = CS.merge_base(self.ws)
        except Exception as exc:
            self.log(f"[FAIL] [V3.23 safety] 并入底座失败：{exc}")
            return
        self.ws = w
        self._push_undo(snap)
        self._changed("merge_base")
        self._fill_fields()
        self.log(f"[OK] [V3.23 safety] 已把底座并入安全区：{len(self.fence()['polygon'])} 个角点（新角点 B1…）；"
                 "检查后「保存」")

    def on_apply(self, walls=None, wait=False):
        objs = CS.scene_objects(self.valid_ws, walls=walls)
        nw = sum(1 for o in objs if "wall_" in o["id"])

        def job():
            r = self.pipe.apply_scene(objs, frame=self.valid_ws.get("frame", "base_link"))
            self.pipe.log(f"{'[OK]' if r.get('ok') else '[FAIL]'} [V3.16 safety] 规划场景: 添加 {r.get('added')} 移除 "
                          f"{r.get('removed')} {r.get('error') or ''}"
                          + (f" · [V3.23] 围墙 {nw} 面已同步到 MoveIt" if nw else ""))
            if r.get("ok") and walls is not False:
                self.scene_applied_for = self.pipe.monitor.proc.pid if self.pipe.monitor.proc else None
            return r
        if wait:
            return job()
        threading.Thread(target=job, daemon=True).start()

    def on_clear(self):
        def job():
            r = self.pipe.apply_scene([], frame=self.valid_ws.get("frame", "base_link"))
            self.pipe.log(f"{'[OK]' if r.get('ok') else '[FAIL]'} [V3.16 safety] 已从 MoveIt 移除 {r.get('removed')} {r.get('error') or ''}")
            self.scene_applied_for = None
        threading.Thread(target=job, daemon=True).start()

    # ================================================================ automatic test (no motion)
    def on_test_auto(self):
        if not self._ready():
            return
        self._run("auto_fence_test", self._auto_test)

    def _auto_test(self):
        model = self.jpanel.model
        sf = self.safety()
        if sf["fence"] is None:
            self.pipe.log("[WARN] [V3.16 safety] 围栏关闭，自动越界测试需要围栏启用")
            return
        urdf = self._urdf()
        act = self.pipe.actual_arm_deg(model)
        base = self._base()
        f = sf["fence"]
        q0 = [math.radians(a) for a in act]
        arm_only = dict(f, points=[p for p in f["points"] if p != "tool"])
        q, _v = CS.find_violating_target(urdf, model["joints"], q0, model["limits"], base, arm_only, None)
        tool_only = q is None
        if tool_only:
            q, _v = CS.find_violating_target(urdf, model["joints"], q0, model["limits"], base, f, sf["tool_T"])
        if q is None:
            self.pipe.log("[WARN] [V3.16 safety] 从当前姿态找不到会碰到桌面的 J2/J4 目标（换一个更低的起始姿态）")
            return
        v = CS.fence_check(urdf, model["joints"], q, base, f, sf["tool_T"])[1]
        tdeg = [math.degrees(x) for x in q]
        self.pipe.log("[INFO] [V3.16 safety] 自动越界目标: " + " ".join(f"J{i + 1} {x:+.1f}" for i, x in enumerate(tdeg))
                      + " | " + CS.violation_text(v, f))
        r1 = self.pipe.move_joints(model, q, 0.05, label="自动越界测试（围栏）")
        a_ok = (not r1.get("ok")) and "FENCE" in str(r1.get("error"))
        r2 = self.pipe._send_move("自动越界测试（MoveIt 只规划）", model["arm_group"], model["joints"], q, 0.05, 30.0,
                                  {"plan_only": True})
        b_ok = not r2.get("ok")
        txt = (f"(a) 围栏：{'拒绝 ✓' if a_ok else '未拒绝 ⚠ ' + str(r1.get('error'))}   "
               f"(b) MoveIt 场景（只规划）：" + ('拒绝 ✓ ' + str(r2.get('error')) if b_ok else
                    ('规划成功（未执行）→ 只有工具点越界：驱动 URDF 里没有工具/夹爪几何，MoveIt 看不到它，工具只受软件围栏保护'
                     if tool_only else '规划成功 ⚠ → 碰撞体未生效或尺寸/位置不对（未执行）')))
        if tool_only and not b_ok:
            b_ok = True
        self._ui(lambda: self.v_test.set(txt))
        self.pipe.log(f"{'[OK]' if a_ok and b_ok else '[WARN]'} [V3.16 safety] 自动越界测试: {txt}")
        CS.write_result("safety_test", {"target_deg": tdeg, "violations": v, "tool_only": tool_only, "fence_refused": a_ok,
                                        "fence_result": r1, "moveit_refused": b_ok, "moveit_result": r2})
