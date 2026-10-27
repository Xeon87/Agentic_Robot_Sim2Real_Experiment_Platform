#!/usr/bin/env python3
"""GUI test for the V3.16 graphical safety editor (needs tkinter + a display, e.g. xvfb-run). Skips otherwise.

Real mouse events on the top / side canvases: drag a corner, undo / redo, double-click an edge (new corner),
right-click delete, add + drag + resize a box, drag z_min in the side view, numeric edit, self-intersection
refused (last valid zone stays enforced), target outside -> joint panel 执行 locked. Teach: tool = pointer, arm
moved (speed capped at 5 %) to four table points, record, generate (zone + table height). Inspect: hover above each
corner (IK), all confirmed. Save -> measured. NERO_V316_SHOT=<prefix> saves <prefix>_edit.png / _teach.png.
"""
import json, math, os, subprocess, sys, tempfile, time
HERE = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, HERE)
try:
    import tkinter as tk
    from tkinter import messagebox
except Exception:
    print("[GUI16] tkinter missing -> skipped"); sys.exit(0)
if not os.environ.get("DISPLAY"):
    print("[GUI16] no DISPLAY -> skipped"); sys.exit(0)
import NERO_V312_Console as C
import NERO_V314_Joint_Control as JC
import NERO_V315_Consistency_Safety as CS
_v323_dw = CS.default_workspace


def _v323_legacy_default():          # V3.23: this test covers the V3.16-V3.22 layered rule (now rule B, 4 corners)
    w = _v323_dw()
    w["fence"].update(rule="B", margin=0.04, polygon=[[-0.15, 0.45], [-0.70, 0.45], [-0.70, -0.45], [-0.15, -0.45]])
    w["corners"] = w["corners"][:4]
    return w


CS.default_workspace = _v323_legacy_default
import NERO_V316_Safety_Editor as SE
import NERO_V313_Real_Robot as RR

TMP = tempfile.mkdtemp(prefix="v316gui_")
SE.STATE_PATH = os.path.join(TMP, "state.json")             # V3.19: never touch the real last-zone state
CS.WORKSPACE_PATH = os.path.join(TMP, "real_workspace.json")
CS.LOG_DIR = os.path.join(TMP, "logs")
asked = []
messagebox.askyesno = lambda t, x, **k: (asked.append((t, x)) or True)
cap = {}
_init = SE.SafetyEditorUI.__init__


def init(self, *a, **k):
    _init(self, *a, **k)
    cap["ed"] = self


SE.SafetyEditorUI.__init__ = init
U = JC.parse_urdf(open(os.path.join(HERE, "config", "nero.urdf")).read())
J = [f"joint{i}" for i in range(1, 8)]
LIM = JC.build_model(open(os.path.join(HERE, "config", "nero.urdf")).read(), None, None)["limits"]
PTR = {"mode": "pointer", "length": 0.10, "axis": "+x"}
TOOL = CS.tool_transform(PTR, U)
SEED = [0, 1.3, 0, 0.9, 0, 0.0, 0]                      # V3.17: work side is −x (mirror of +x pose)
MARKS = [(-0.40, 0.20), (-0.55, 0.20), (-0.55, -0.20), (-0.40, -0.20)]
TEACH_Q, HOVER_Q = [], []
for mx, my in MARKS:
    q, err = CS.ik_tool(U, J, SEED, "base_link", TOOL, [mx, my, 0.002], LIM)
    assert q is not None, (mx, my, err)
    qh, err = CS.ik_tool(U, J, q, "base_link", TOOL, [mx, my, 0.08], LIM)      # lift between marks, like a person would
    assert qh is not None, (mx, my, err)
    TEACH_Q.append(q)
    HOVER_Q.append(qh)
res = {}
_orig = tk.Tk.mainloop


def logs(root):
    def walk(w):
        yield w
        for c in w.winfo_children():
            yield from walk(c)
    return "\n".join(w.get("1.0", "end") for w in walk(root) if w.winfo_class() == "Text")


def driver(root):
    ed = cap["ed"]
    pipe, jp = ed.pipe, ed.jpanel
    speeds = []
    _sm = pipe._send_move

    def rec_send(label, group, joints, positions, speed, timeout_s, extra=None):
        speeds.append((label, speed))
        return _sm(label, group, joints, positions, speed, timeout_s, extra)
    pipe._send_move = rec_send

    def wait(cond, then, t_max=60):
        t0 = time.time()

        def chk():
            if cond():
                then()
            elif time.time() - t0 > t_max:
                res["fail"] = f"timeout waiting before {then.__name__}\n" + logs(root)[-2500:]
                root.destroy()
            else:
                root.after(100, chk)
        root.after(100, chk)

    def nb_select(text):
        def walk(w):
            yield w
            for c in w.winfo_children():
                yield from walk(c)
        for w in walk(root):
            if w.winfo_class() == "TNotebook":
                for t in w.tabs():
                    if w.tab(t, "text") == text:
                        w.select(t)

    def shot(name):
        pref = os.environ.get("NERO_V316_SHOT")
        if pref:
            nb_select("工作区安全")
            root.update()
            time.sleep(0.5)
            ed.redraw()
            root.update()
            subprocess.run(["import", "-window", "root", f"{pref}_{name}.png"], check=False)

    def mouse(cv, seq, x, y, **kw):
        cv.event_generate(seq, x=int(x), y=int(y), **kw)

    def drag(cv, a, b, steps=6):
        mouse(cv, "<ButtonPress-1>", *a)
        for k in range(1, steps + 1):
            mouse(cv, "<B1-Motion>", a[0] + (b[0] - a[0]) * k / steps, a[1] + (b[1] - a[1]) * k / steps)
        mouse(cv, "<ButtonRelease-1>", *b)

    def s0():
        res["head0"] = ed.v_head.get()
        pipe.monitor.start([sys.executable, "-u", os.path.join(HERE, "tests", "v315_fake_monitor.py")],
                           env=dict(os.environ, FAKE_RAD_S="20", FAKE_Q0="0,0.5,0,-0.6,0,-0.27,0"), stdin_pipe=True)
        for k in RR.STEP_KEYS:
            pipe.state[k] = (RR.GREEN, "fake")
        nb_select("工作区安全")
        wait(lambda: jp.enabled and pipe.real_urdf is not None and "规划场景: 添加 ['v315_table']" in logs(root), s1)

    def s1():
        root.update()
        c = ed.cv_top
        f = ed.fence()
        p0 = list(f["polygon"][1])                      # M2 (−0.70, 0.45)
        a = ed.w2t(*p0)
        b = (a[0] + 20, a[1] + 30)                      # screen right/down = +y / +x (forward is up)
        drag(c, a, b)
        S = ed._top_tf()[0]
        res["drag"] = (p0, list(ed.fence()["polygon"][1]), S)
        ed.on_undo()
        res["undo"] = list(ed.fence()["polygon"][1])
        ed.on_redo()
        res["redo"] = list(ed.fence()["polygon"][1])
        # double-click on the middle of edge M3-M4, then right-click the new corner
        f = ed.fence()
        m = [(f["polygon"][2][0] + f["polygon"][3][0]) / 2, (f["polygon"][2][1] + f["polygon"][3][1]) / 2]
        pm = ed.w2t(*m)
        ed.on_top_double(type("Ev", (), {"x": int(pm[0]), "y": int(pm[1])})())   # Tk cannot synthesize double clicks
        res["n_after_insert"] = len(ed.fence()["polygon"])
        res["corner_names"] = [x["name"] for x in ed.ws["corners"]]
        mouse(c, "<Button-3>", *pm)
        res["n_after_delete"] = len(ed.fence()["polygon"])
        # box: add, drag, resize
        ed.on_add_box()
        i = len(ed.ws["boxes"]) - 1
        b0 = json.loads(json.dumps(ed.ws["boxes"][i]))
        cpt = ed.w2t(b0["pos"][0], b0["pos"][1])
        drag(c, cpt, (cpt[0] - 0.10 * S, cpt[1]))          # screen left = −y by 0.10 m
        b1 = json.loads(json.dumps(ed.ws["boxes"][i]))
        hnd = ed.w2t(b1["pos"][0] + b1["size"][0] / 2, b1["pos"][1] + b1["size"][1] / 2)   # bottom-right handle
        drag(c, hnd, (hnd[0] + 0.05 * S, hnd[1] + 0.05 * S))
        b2 = json.loads(json.dumps(ed.ws["boxes"][i]))
        res["box"] = (b0, b1, b2)
        # side view: z_min line 30 mm -> 50 mm
        sv = ed.cv_side
        Ss = ed._side_tf()[0]
        y0 = ed.w2s(0.5, ed.fence()["z_min_points"])
        drag(sv, (ed.w2s(0.4, 0)[0], y0[1]), (ed.w2s(0.4, 0)[0], y0[1] - 0.02 * Ss))
        res["zmin"] = ed.fence()["z_min_points"]
        # numeric edit of M1
        ed._select(("v", 0))
        ed.sel_vars["x"].set("-0.200")
        ed.sel_vars["y"].set("0.420")
        ed.on_sel_apply()
        res["m1"] = list(ed.fence()["polygon"][0])
        # self-intersection: move M1 across -> invalid, last valid stays enforced
        enforced = json.dumps(ed.valid_ws["fence"]["polygon"])
        ed._select(("v", 0))
        ed.sel_vars["x"].set("-0.900")                  # M1 beyond the M2-M3 edge -> edges cross
        ed.sel_vars["y"].set("0.000")
        ed.on_sel_apply()
        res["invalid_head"] = ed.v_head.get()
        res["enforced_same"] = json.dumps(ed.valid_ws["fence"]["polygon"]) == enforced
        ed.on_undo()
        root.update()
        shot("edit")
        # target outside -> 执行 locked
        jp.sync_to_actual()
        a = jp.actual_deg()
        for j, d in ((0, 90), (1, 50), (3, 60)):         # swing to the side and down: low and outside the zone
            jp._set_target(j, a[j] + d)
        jp._mark_edited("test")                         # V3.18: programmatic targets count as an edit here
        root.after(900, s2)

    def s2():
        root.update()
        res["lock_state"] = jp.v_state.get()
        res["exec_btn"] = str(jp.btn_exec.cget("state"))
        res["now_target"] = ed.v_now.get()
        jp.sync_to_actual()
        # teach
        ed.v_tool_mode.set("pointer")
        ed.v_tool_axis.set("+x")
        ed.f_vars["tool_length"].set("0.100")
        ed.on_fields_apply()
        ed.v_mode.set("teach")
        ed.on_mode()
        res["teach_safety"] = ed.safety()
        res["teach_k"] = 0
        root.after(200, teach_next)

    def teach_next():
        k = res["teach_k"]
        if k >= len(TEACH_Q):
            ed.on_teach_generate()
            root.update()
            res["after_teach"] = json.loads(json.dumps(ed.ws))
            shot("teach")
            ed.v_mode.set("inspect")
            ed.on_mode()
            ed.on_inspect()
            wait(lambda: "[V3.16 inspect] 完成" in logs(root), s3, 180)
            return
        box = {}

        def mv():
            for qq, tag in ((HOVER_Q[k], "hover"), (TEACH_Q[k], "touch")):
                box["r"] = pipe.move_and_settle(jp.model, qq, 0.5, label=f"teach move {k + 1} {tag}", settle_s=0.3)
                if not box["r"].get("ok"):
                    return
        import threading
        th = threading.Thread(target=mv, daemon=True)
        th.start()

        def done():
            if th.is_alive():
                root.after(100, done)
                return
            if not box["r"].get("ok"):
                res["fail"] = f"teach move {k + 1} failed: {box['r']}"
                root.destroy()
                return
            root.after(300, rec)

        def rec():
            ed.on_teach_record()
            res["teach_k"] = k + 1
            root.after(100, teach_next)
        done()

    def s3():
        root.update()
        res["insp_rows"] = [ed.tv_insp.item(i, "values") for i in ed.tv_insp.get_children()]
        ed.v_mode.set("edit")
        ed.on_mode()
        res["edit_safety"] = ed.safety()
        ed.on_save()
        root.update()
        res["head1"] = ed.v_head.get()
        res["speeds"] = list(speeds)
        res["logs"] = logs(root)
        pipe.monitor.send({"quit": True})
        root.after(300, root.destroy)
    root.after(1200, s0)


def patched(self, n=0):
    driver(self)
    _orig(self, n)


tk.Tk.mainloop = patched
C.run_console("real")
assert "fail" not in res, res.get("fail")
L = res["logs"]
assert "未测量" in res["head0"]
p0, p1, S = res["drag"]
assert abs((p1[0] - p0[0]) - (30 / S)) < 0.002 and abs((p1[1] - p0[1]) - (20 / S)) < 0.002, res["drag"]
assert res["undo"] == p0 and res["redo"] == p1
assert res["n_after_insert"] == 5 and res["corner_names"][3] == "M5" and res["n_after_delete"] == 4, res
b0, b1, b2 = res["box"]
assert abs(b1["pos"][1] - b0["pos"][1] + 0.10) < 0.003 and abs(b1["pos"][0] - b0["pos"][0]) < 0.003
assert abs(b2["size"][0] - (b1["size"][0] + 0.05)) < 0.003 and abs(b2["size"][1] - (b1["size"][1] + 0.05)) < 0.003   # handle dragged outward
assert abs(res["zmin"] - 0.05) < 0.003, res["zmin"]
assert res["m1"] == [-0.2, 0.42]
assert "自相交" in res["invalid_head"] and res["enforced_same"], res["invalid_head"]
assert res["exec_btn"] == "disabled" and "目标越界" in res["lock_state"] and "执行」已锁定" in res["now_target"], res
ts = res["teach_safety"]
assert ts["max_speed"] == 0.05 and abs(ts["fence"]["tool_z_min_override"] - (-0.005)) < 1e-9 and ts["tool_T"][0][3] == 0.1
assert all(sp <= 0.05 + 1e-9 for lab, sp in res["speeds"] if lab.startswith("teach move") or lab.startswith("巡检")), res["speeds"]
at = res["after_teach"]
poly = at["fence"]["polygon"]
assert len(poly) == 4 and all(min(math.hypot(p[0] - mx, p[1] - my) for p in poly) < 0.002 for mx, my in MARKS), poly
assert abs(at["fence"]["z_table"] - 0.002) < 0.0015 and [c["source"] for c in at["corners"]] == ["示教"] * 4
assert abs(at["fence"]["z_min_points"] - (at["fence"]["z_table"] + 0.05)) < 0.003   # keeps the dragged 50 mm offset
tb = [b for b in at["boxes"] if b["name"] == "table"][0]
assert abs(tb["pos"][2] + tb["size"][2] / 2 - at["fence"]["z_table"]) < 1e-6
assert "[OK] [V3.16 teach] 安全区生成：4 个角点" in L
rows = res["insp_rows"]
assert len(rows) == 4 and all(r[3] == "对准 ✓" for r in rows), (rows, [l for l in L.splitlines() if "inspect" in l or "巡检" in l][-6:])
assert "[OK] [V3.16 inspect] 完成" in L
assert res["edit_safety"]["max_speed"] is None and "tool_z_min_override" not in res["edit_safety"]["fence"]
assert "已保存" in res["head1"] and json.load(open(CS.WORKSPACE_PATH))["measured"]
assert any(t.startswith("巡检 M") for t, _x in asked)
print("[GUI16] editor: drag corner + undo/redo, double-click add / right-click delete, box move + resize, z_min drag, "
      "numeric edit, self-intersection refused (old zone kept), target outside -> 执行 locked; teach 4 marks @≤5% -> zone + "
      "table height; inspect 4/4; save")
