#!/usr/bin/env python3
"""NERO V4.7 Smol VLA tab tools:

* NpzVideoReader   - the two camera streams of an episode npz (cam_main / cam_wrist, n x 480 x 640 x 3), read FRAME BY
                     FRAME from the compressed zip member (a 1-minute episode is ~2 GB raw - never loaded whole);
                     forward steps are sequential reads, a step back re-opens the member; a small LRU cache.
* list_datasets / episode_rows - the dataset folders on the drive and their episodes (duration, frames, size, file).
* draw_dets        - YOLOE boxes + labels on a frame.
* EpisodePlayer    - a Toplevel window: main + wrist camera side by side, play / pause, slider, speed, step,
                     optional YOLOE labels (detected in a worker thread, cached per frame), keyboard shortcuts.
* lamp_images      - small coloured dots for notebook tabs.
Worker threads never touch Tk (queue + after loop).
"""
from __future__ import annotations

import collections
import json
import os
import queue
import threading
import time
import zipfile
from typing import Any, Dict, List, Optional

import numpy as np

VERSION = "V4.7"
MARKER = "V4.7 EPISODE BROWSER + PLAYER"
CAM_KEYS = ("cam_main", "cam_wrist")
CAM_TITLES = {"cam_main": "主相机（D435i）", "cam_wrist": "腕部相机（OV4688）"}


# ============================================================================ npz video
class NpzVideoReader:
    """frame(cam, i) -> uint8 (H, W, 3) without loading the whole array"""

    def __init__(self, npz_path: str, cache_frames: int = 48):
        self.path = npz_path
        self.lock = threading.Lock()
        self.info: Dict[str, Dict[str, Any]] = {}
        self._streams: Dict[str, Any] = {}
        self._next: Dict[str, int] = {}
        self._cache: "collections.OrderedDict" = collections.OrderedDict()
        self._cache_n = int(cache_frames)
        self._zip = zipfile.ZipFile(npz_path)
        names = set(self._zip.namelist())
        for k in CAM_KEYS:
            if k + ".npy" in names:
                with self._zip.open(k + ".npy") as f:
                    shape, dtype, off = self._header(f)
                self.info[k] = {"shape": shape, "dtype": dtype, "frame_bytes": int(np.prod(shape[1:])) * dtype.itemsize}
        self.meta: Dict[str, Any] = {}
        if "meta_json.npy" in names:
            try:
                with self._zip.open("meta_json.npy") as f:
                    self.meta = json.loads(str(np.lib.format.read_array(f)))
            except Exception:
                self.meta = {}
        self.n = min((v["shape"][0] for v in self.info.values()), default=0)
        self.fps = float(self.meta.get("fps") or 20.0)

    @staticmethod
    def _header(f):
        ver = np.lib.format.read_magic(f)
        if ver == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(f)
        if fortran:
            raise ValueError("fortran-ordered image array")
        return tuple(int(v) for v in shape), np.dtype(dtype), None

    def _open(self, cam):
        s = self._streams.pop(cam, None)
        if s is not None:
            try:
                s.close()
            except Exception:
                pass
        f = self._zip.open(cam + ".npy")
        self._header(f)
        self._streams[cam] = f
        self._next[cam] = 0
        return f

    def frame(self, cam: str, i: int) -> Optional[np.ndarray]:
        if cam not in self.info or not (0 <= int(i) < self.info[cam]["shape"][0]):
            return None
        i = int(i)
        with self.lock:
            key = (cam, i)
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            inf = self.info[cam]
            fb = inf["frame_bytes"]
            f = self._streams.get(cam)
            if f is None or self._next.get(cam, 0) > i:
                f = self._open(cam)
            skip = i - self._next[cam]
            while skip > 0:                                   # decompress and drop (no full-array allocation)
                n = min(skip, 32)
                f.read(fb * n)
                skip -= n
            buf = f.read(fb)
            self._next[cam] = i + 1
            if len(buf) < fb:
                return None
            img = np.frombuffer(buf, dtype=inf["dtype"]).reshape(inf["shape"][1:])
            self._cache[key] = img
            while len(self._cache) > self._cache_n:
                self._cache.popitem(last=False)
            return img

    def close(self):
        with self.lock:
            for s in self._streams.values():
                try:
                    s.close()
                except Exception:
                    pass
            self._streams.clear()
            try:
                self._zip.close()
            except Exception:
                pass


# ============================================================================ datasets / episodes
def list_datasets(root: str) -> List[str]:
    """dataset folder names under root that have meta/episodes.jsonl or data/*.npz, newest first"""
    out = []
    try:
        for name in os.listdir(root):
            d = os.path.join(root, name)
            if not os.path.isdir(d):
                continue
            if os.path.isfile(os.path.join(d, "meta", "episodes.jsonl")) or os.path.isdir(os.path.join(d, "data")):
                out.append((os.path.getmtime(d), name))
    except OSError:
        return []
    return [n for _t, n in sorted(out, reverse=True)]


def episode_rows(dataset) -> List[Dict[str, Any]]:
    """dataset episodes newest first, with duration / file / size and whether the file is there"""
    rows = []
    for e in dataset.list_episodes():
        idx = int(e.get("episode_index", -1))
        p = e.get("file") or dataset._episode_npz_path(idx)
        if not os.path.isfile(p):
            p2 = dataset._episode_npz_path(idx)
            p = p2 if os.path.isfile(p2) else p
        fps = float(e.get("fps") or 20.0)
        n = int(e.get("length") or 0)
        exists = os.path.isfile(p)
        rows.append({"idx": idx, "frames": n, "dur_s": n / max(1.0, fps), "fps": fps, "valid": bool(e.get("valid")),
                     "task": (e.get("tasks") or [""])[0], "saved_at": str(e.get("saved_at", "")),
                     "file": p, "exists": exists, "size_mb": (os.path.getsize(p) / 1e6) if exists else 0.0,
                     "split": e.get("split", ""), "limit_ok": e.get("limit_ok"),
                     "limit_report": e.get("limit_report")})
    rows.sort(key=lambda r: -r["idx"])
    return rows


# ============================================================================ drawing
def draw_dets(rgb: np.ndarray, dets, scale_xy=(1.0, 1.0)) -> np.ndarray:
    """boxes + 'name conf' on a copy of rgb (dets: YOLOE dicts with box / name / conf)"""
    import cv2
    img = np.ascontiguousarray(rgb).copy()
    sx, sy = scale_xy
    palette = {"egg": (255, 196, 0), "egg carton": (0, 200, 120), "holder": (0, 200, 120), "cup": (80, 160, 255)}
    for d in dets or []:
        try:
            x0, y0, x1, y1 = [float(v) for v in d["box"]]
        except Exception:
            continue
        name = str(d.get("name", "?"))
        col = palette.get(name, (255, 64, 160))
        p0, p1 = (int(x0 * sx), int(y0 * sy)), (int(x1 * sx), int(y1 * sy))
        cv2.rectangle(img, p0, p1, col, 2)
        lab = f"{name} {float(d.get('conf', 0)):.2f}"
        (tw, th), _b = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = max(th + 4, p0[1])
        cv2.rectangle(img, (p0[0], ty - th - 4), (p0[0] + tw + 4, ty), col, -1)
        cv2.putText(img, lab, (p0[0] + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def lamp_images(tk, size: int = 12) -> Dict[str, Any]:
    """small filled circles (PhotoImage, no font glyph) for notebook tab lamps"""
    cols = {"off": "#9e9e9e", "busy": "#ffb300", "ok": "#2e7d32", "warn": "#f9a825", "bad": "#c62828"}
    out = {}
    r = (size - 1) / 2.0
    for k, c in cols.items():
        im = tk.PhotoImage(width=size, height=size)
        for y in range(size):
            row = []
            for x in range(size):
                d = ((x - r) ** 2 + (y - r) ** 2) ** 0.5
                row.append(c if d <= r - 0.3 else ("#616161" if d <= r + 0.5 else ""))
            for x, v in enumerate(row):
                if v:
                    im.put(v, (x, y))
        out[k] = im
    return out


# ============================================================================ player window
def _prompts_of_note(note):
    """'YOLOE：red block, box · ≥ 0.25' -> 'red block, box'"""
    t = str(note or "")
    if "YOLOE：" not in t:
        return ""
    return t.split("YOLOE：", 1)[1].split(" · ")[0].strip()


class EpisodePlayer:
    """Toplevel player for one episode: both cameras, play / pause / step / slider / speed, YOLOE labels.
    yolo_fn(rgb) -> list of dets (called in a worker thread) or None when YOLOE is off / unavailable."""

    VIEW_W, VIEW_H = 480, 360

    def __init__(self, tk, ttk, master, npz_path: str, title: str = "", yolo_fn=None, yolo_note: str = "",
                 log=None, on_close=None, yolo_factory=None):
        import NERO_V325_Camera as CAM
        self.tk, self.ttk, self.CAM = tk, ttk, CAM
        self.log = log or (lambda s: None)
        self.on_close_cb = on_close
        self.rd = NpzVideoReader(npz_path)
        self.yolo_fn = yolo_fn
        self.win = tk.Toplevel(master)
        self.win.title(title or f"Episode 回放 · {os.path.basename(npz_path)}")
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.q: "queue.Queue" = queue.Queue()
        self.i, self.playing, self.speed = 0, False, 1.0
        self.dets: Dict[Any, Any] = {}
        self._want = None
        self._stop = threading.Event()
        self._need = threading.Event()
        self._imgs = {}
        self.alive = True
        top = ttk.Frame(self.win, padding=6)
        top.pack(fill="both", expand=True)
        hdr = ttk.Frame(top)
        hdr.pack(fill="x")
        m = self.rd.meta
        ttk.Label(hdr, text=f"Episode #{m.get('episode_index', '?')} · {self.rd.n} 帧 · {self.rd.n / self.rd.fps:.1f} s · "
                            f"{'有效' if m.get('valid') else '无效'} · {(m.get('tasks') or [''])[0][:60]}",
                  font=("TkDefaultFont", 10, "bold")).pack(side="left")
        vids = ttk.Frame(top)
        vids.pack(fill="both", expand=True, pady=4)
        self.cv = {}
        for k in CAM_KEYS:
            col = ttk.Frame(vids)
            col.pack(side="left", padx=4)
            ttk.Label(col, text=CAM_TITLES[k]).pack(anchor="w")
            c = tk.Canvas(col, width=self.VIEW_W, height=self.VIEW_H, bg="#202020", highlightthickness=0)
            c.pack()
            self.cv[k] = c
        bar = ttk.Frame(top)
        bar.pack(fill="x", pady=(2, 0))
        self.btn_play = ttk.Button(bar, text="▶ 播放", width=8, command=self.toggle)
        self.btn_play.pack(side="left")
        ttk.Button(bar, text="◀ 1 帧", command=lambda: self.step(-1)).pack(side="left", padx=2)
        ttk.Button(bar, text="1 帧 ▶", command=lambda: self.step(1)).pack(side="left", padx=2)
        ttk.Label(bar, text="  速度").pack(side="left")
        self.v_speed = tk.StringVar(value="1×")
        cb = ttk.Combobox(bar, textvariable=self.v_speed, values=("0.25×", "0.5×", "1×", "2×", "4×"), width=6,
                          state="readonly")
        cb.pack(side="left", padx=2)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._set_speed())
        self.v_yolo = tk.BooleanVar(value=yolo_fn is not None)
        self._yolo_mirror = yolo_fn is not None
        self.chk_yolo = ttk.Checkbutton(bar, text="YOLOE 标签", variable=self.v_yolo, command=self._on_yolo)
        self.chk_yolo.pack(side="left", padx=(12, 2))
        if yolo_fn is None:
            self.chk_yolo.state(["disabled"])
        self._note0 = yolo_note
        self.v_note = tk.StringVar(value=yolo_note)
        ttk.Label(bar, textvariable=self.v_note, foreground="#607d8b").pack(side="left", padx=6)
        # V4.7.4: what YOLOE looks for, editable (comma separated); 「应用」 re-labels from the current frame
        self.yolo_factory, self._ygen = yolo_factory, 0
        self.v_prompts = tk.StringVar(value=_prompts_of_note(yolo_note))
        if yolo_factory is not None:
            pbar = ttk.Frame(top)
            pbar.pack(fill="x", pady=(2, 0))
            ttk.Label(pbar, text="YOLOE 找：").pack(side="left")
            self.ent_prompts = ttk.Entry(pbar, textvariable=self.v_prompts, width=44)
            self.ent_prompts.pack(side="left", padx=2)
            self.ent_prompts.bind("<Return>", lambda e: (self.apply_prompts(), "break")[1])
            ttk.Button(pbar, text="应用", command=self.apply_prompts).pack(side="left", padx=2)
            ttk.Label(pbar, text="（逗号分隔，英文名效果最好，例如 red block, blue block, box）",
                      foreground="#9e9e9e").pack(side="left", padx=4)
        self.v_pos = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.v_pos, width=22, anchor="e").pack(side="right")
        self.sl = tk.Scale(top, from_=0, to=max(0, self.rd.n - 1), orient="horizontal", showvalue=False,
                           command=self._on_slider, length=2 * self.VIEW_W)
        self.sl.pack(fill="x")
        ttk.Label(top, text="空格 = 播放 / 暂停 · ← → = 前后 1 帧 · Esc = 关闭", foreground="#9e9e9e").pack(anchor="w")
        for seq, fn in (("<space>", lambda e: self.toggle()), ("<Left>", lambda e: self.step(-1)),
                        ("<Right>", lambda e: self.step(1)), ("<Escape>", lambda e: self.close())):
            self.win.bind(seq, lambda e, f=fn: (f(e), "break")[1])
        # V4.7.3: raw video and YOLOE run in separate threads; a frame is drawn when it is not older than the one on
        # screen (V4.7 drew a frame only if it was STILL the current one when the worker finished - with YOLOE (2
        # inferences / frame, > 50 ms) every frame was late during playback -> black canvas, one still at the end)
        self._cur = None                                  # (i, {cam: view rgb}, {cam: (w, h)}) on screen
        self.n_drawn = 0                                  # frames drawn (tests)
        self._want_y = None
        self._need_y = threading.Event()
        self._worker = threading.Thread(target=self._work, daemon=True, name="v47_player")
        self._worker.start()
        self._yworker = threading.Thread(target=self._work_yolo, daemon=True, name="v47_player_yolo")
        self._yworker.start()
        self._show(0)
        self._t_last = time.time()
        self.win.after(30, self._tick)

    # -- Tk thread
    def apply_prompts(self):
        """V4.7.4: new YOLOE prompts -> new detector function, labels recomputed"""
        import NERO_V474_Episode_Tools as ET
        pr = ET.parse_prompts(self.v_prompts.get())
        if not pr or self.yolo_factory is None:
            return
        fn, note = self.yolo_factory(pr)
        self._ygen += 1
        self.dets = {}
        self.yolo_fn = fn
        self._note0 = note
        self.v_note.set(note)
        if fn is not None:
            self.chk_yolo.state(["!disabled"])
            self.v_yolo.set(True)
            self._yolo_mirror = True
        self._want_y = self.i
        self._need_y.set()
        self._redraw()
        self.log(f"[INFO] [V4.7.4 EPISODES] 回放 YOLOE 改找：{', '.join(pr)}")

    def _on_yolo(self):
        self._yolo_mirror = bool(self.v_yolo.get())
        self._want_y = self.i
        self._need_y.set()
        self._redraw()

    def _set_speed(self):
        try:
            self.speed = float(self.v_speed.get().rstrip("×x"))
        except ValueError:
            self.speed = 1.0

    def toggle(self):
        if self.rd.n <= 0:
            return
        if not self.playing and self.i >= self.rd.n - 1:
            self.i = 0
        self.playing = not self.playing
        self.btn_play.configure(text="‖ 暂停" if self.playing else "▶ 播放")
        self._t_last = time.time()

    def step(self, d):
        self.playing = False
        self.btn_play.configure(text="▶ 播放")
        self._show(min(max(0, self.i + int(d)), max(0, self.rd.n - 1)))

    def _on_slider(self, val):
        try:
            v = int(float(val))
        except ValueError:
            return
        if v != self.i:
            self._show(v, from_slider=True)

    def _show(self, i, from_slider=False):
        self.i = int(i)
        self._want = self.i
        self._need.set()
        self._want_y = self.i
        self._need_y.set()
        if not from_slider:
            try:
                self.sl.set(self.i)
            except Exception:
                pass
        self.v_pos.set(f"{self.i + 1} / {self.rd.n} 帧 · {self.i / self.rd.fps:.1f} s")

    def _tick(self):
        if not self.alive:
            return
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == "frame":
                    i, views, sizes = val
                    cur = -1 if self._cur is None else self._cur[0]
                    # playing: anything not older than the frame on screen; paused / stepping: the asked frame only
                    if i == self.i or (self.playing and i >= cur):
                        self._cur = (i, views, sizes)
                        self._redraw()
                elif kind == "dets":
                    if self._cur is not None and 0 <= self._cur[0] - val <= 2:
                        self._redraw()
                elif kind == "note":                       # V4.8: a note from older prompts is dropped
                    g_, txt = val if isinstance(val, tuple) else (self._ygen, val)
                    if g_ == self._ygen:
                        self.v_note.set(self._note0 if txt is None else txt)
        except queue.Empty:
            pass
        if self.playing:
            now = time.time()
            adv = int((now - self._t_last) * self.rd.fps * self.speed)
            if adv >= 1:
                self._t_last += adv / (self.rd.fps * self.speed)
                j = self.i + adv
                if j >= self.rd.n - 1:
                    j = self.rd.n - 1
                    self.playing = False
                    self.btn_play.configure(text="▶ 播放")
                self._show(j)
        self.win.after(15, self._tick)

    def _dets_near(self, k, i):
        """YOLOE boxes of frame i, else of the 2 frames before it (playback: YOLOE is slower than the video)"""
        for j in (i, i - 1, i - 2):
            d = self.dets.get((k, j))
            if d is not None:
                return d
        return None

    def _redraw(self):
        if self._cur is None:
            return
        i, views, sizes = self._cur
        yolo_on = self.yolo_fn is not None and bool(self._yolo_mirror)
        imgs = {}
        for k, view in views.items():
            if view is not None and yolo_on:
                d = self._dets_near(k, i)
                if d:
                    w, h = sizes[k]
                    view = draw_dets(view, d, (self.VIEW_W / float(w), self.VIEW_H / float(h)))
            imgs[k] = view
        self._draw(imgs)
        self.n_drawn += 1

    def _draw(self, imgs):
        for k, rgb in imgs.items():
            if rgb is None:
                continue
            ppm = self.CAM.ppm_bytes(np.ascontiguousarray(rgb))
            try:
                ph = self.tk.PhotoImage(data=ppm, format="PPM")
            except Exception:
                ph = self.tk.PhotoImage(data=self.CAM.png_bytes_from_ppm(ppm), format="PNG")
            self._imgs[k] = ph
            c = self.cv[k]
            c.delete("all")
            c.create_image(0, 0, image=ph, anchor="nw")

    # -- workers
    def _work(self):
        """the video: read + scale the wanted frame of both cameras (never waits for YOLOE)"""
        import cv2
        while not self._stop.is_set():
            self._need.wait(0.2)
            self._need.clear()
            i = self._want
            if i is None:
                continue
            views, sizes = {}, {}
            for k in CAM_KEYS:
                fr = self.rd.frame(k, i)
                if fr is None:
                    views[k] = None
                    continue
                h, w = fr.shape[:2]
                sizes[k] = (w, h)
                views[k] = cv2.resize(fr, (self.VIEW_W, self.VIEW_H), interpolation=cv2.INTER_AREA)
            self.q.put(("frame", (i, views, sizes)))

    def _work_yolo(self):
        """YOLOE on the newest wanted frame (skips frames it cannot keep up with); boxes cached per frame"""
        first = True
        while not self._stop.is_set():
            self._need_y.wait(0.2)
            self._need_y.clear()
            i = self._want_y
            fn, gen, dets = self.yolo_fn, self._ygen, self.dets
            if i is None or fn is None or not self._yolo_flag():
                continue
            done = False
            for k in CAM_KEYS:
                key = (k, i)
                if key in dets:
                    continue
                fr = self.rd.frame(k, i)
                if fr is None:
                    continue
                if first:
                    self.q.put(("note", (gen, "YOLOE 计算中（第一次要加载模型）…")))
                try:
                    dets[key] = fn(np.ascontiguousarray(fr)) or []      # into the dict of ITS prompts (V4.7.4)
                except Exception as exc:
                    dets[key] = []
                    self.q.put(("note", (gen, f"YOLOE 出错：{str(exc)[:60]}")))
                if first:
                    first = False
                    self.q.put(("note", (gen, None)))
                done = True
            if done and gen == self._ygen:
                self.q.put(("dets", i))

    def _yolo_flag(self):
        """the checkbox state mirrored into a plain attribute by the Tk thread (a worker must not read a Tk variable)"""
        return self._yolo_mirror

    def close(self):
        self.alive = False
        self._stop.set()
        self._need.set()
        self._need_y.set()
        try:
            self.win.destroy()
        except Exception:
            pass
        try:
            self.rd.close()
        except Exception:
            pass
        if self.on_close_cb:
            try:
                self.on_close_cb(self)
            except Exception:
                pass
