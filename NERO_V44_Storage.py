#!/usr/bin/env python3
"""NERO V4.4 — EMBODIED_AI removable-drive storage for LeRobot + SmolVLA.

NERO_V44_Storage.py   (V4.4 EMBODIED_AI DRIVE STORAGE)

Everything large that LeRobot / SmolVLA needs lives on the removable drive labelled ``EMBODIED_AI``
instead of the (full) internal disk:

    <EMBODIED_AI>/nero_lerobot/
        envs/lerobot/          isolated Python venv: lerobot[smolvla] + torch (+ CUDA libs)
        hf_home/               HF_HOME (hub cache: SmolVLA base + SmolVLM2 backbone), hf_home/lerobot = HF_LEROBOT_HOME
        models/smolvla_base/   local copy of lerobot/smolvla_base (loaded offline)
        datasets/nero/         NERO demonstration recorder datasets (npz, V4.2 format)
        datasets/lerobot/      the same demos exported to the real LeRobotDataset format (for lerobot-train)
        checkpoints/           NERO built-in VLA checkpoints (checkpoint.npz)
        training/              lerobot-train runs (one folder per run: checkpoints/, logs)
        cache/{pip,torch,xdg}  pip / torch / misc caches       tmp/  TMPDIR for pip builds and training
        logs/                  install / download / export / training / worker logs
        activate_nero_lerobot.sh   `source` this in a terminal to use the drive environment by hand
        nero_storage.json      layout marker

Rules:
- The drive is only used when it is REALLY mounted (``os.path.ismount``): an empty leftover
  ``/media/<user>/EMBODIED_AI`` folder after unplugging is on the internal disk and is rejected.
- Heavy jobs (install / model download / export / training) refuse to run when the drive is missing, unless
  ``NERO_V44_ALLOW_INTERNAL=1``.  Small things (demo datasets, NERO checkpoints) fall back to the old
  ``~/Downloads/cameracalib/...`` folders so the tab keeps working without the drive.
- ``NERO_DRIVE_ROOT=/some/path`` forces a root (any folder, mounted or not; used by the tests).
- Only the Python standard library is used (the console runs on /usr/bin/python3).
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

VERSION = "V4.6"
MARKER = "V4.4 EMBODIED_AI DRIVE STORAGE"

DRIVE_LABEL = os.environ.get("NERO_DRIVE_LABEL", "EMBODIED_AI")
LAYOUT_DIRNAME = "nero_lerobot"
LAYOUT_VERSION = 1

# Default pip requirement; editable in the GUI (e.g. "lerobot[smolvla]==0.4.4").
# LeRobot >= 0.5 needs Python >= 3.12; Ubuntu 22.04 ships 3.10 (-> pip picks lerobot 0.4.4). PY_MODE "uv312" therefore
# downloads a private Python 3.12 ONTO THE DRIVE with `uv` and builds the venv from it (falls back to the system
# Python automatically if that fails, e.g. no access to github.com).
PY_MODES = ("uv312", "system")
DEFAULT_PY_MODE = os.environ.get("NERO_V44_PY_MODE", "uv312")
UV_PYTHON_VERSION = "3.12"
# LeRobot >= 0.5 splits LeRobotDataset (`dataset`) and lerobot-train (`training`) into extras; 0.4.4 ignores unknown
# extras with a pip warning, so the same spec works on both.
DEFAULT_PIP_SPEC = os.environ.get("NERO_V44_PIP_SPEC", "lerobot[smolvla,dataset,training]")
DEFAULT_BASE_MODEL_ID = "lerobot/smolvla_base"
MIN_FREE_GB_INSTALL = 25.0          # measured: venv 5.9 GB + pip cache 3 GB + HF cache/models 7.6 GB = 16.5 GB, + data headroom
LEGACY_DATASET_ROOT = os.path.expanduser("~/Downloads/cameracalib/lerobot_datasets")
LEGACY_CKPT_ROOT = os.path.expanduser("~/Downloads/cameracalib/vla_checkpoints")
LEGACY_MODEL_DIR = os.path.expanduser("~/Downloads/objectmodels/smolvla_base")

GOOD_FS = ("ext4", "ext3", "xfs", "btrfs", "f2fs")
SOSO_FS = ("exfat", "ntfs", "ntfs3", "fuseblk")
BAD_FS = ("vfat", "msdos", "fat", "fat32")

_ORIG_HF_HOME = os.environ.get("HF_HOME")   # before apply_to_process_env() points it at the drive

SUBDIRS = (
    "envs", "python", "cache/uv", "hf_home", "hf_home/hub", "hf_home/lerobot", "models", "datasets/nero", "datasets/lerobot",
    "checkpoints", "training", "cache/pip", "cache/torch", "cache/xdg", "tmp", "logs",
)


# ----------------------------------------------------------------------------------------------- drive detection
def _unescape_mount(s: str) -> str:
    return s.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


def read_mounts(path: str = "/proc/self/mounts") -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4:
                    out.append({"device": _unescape_mount(parts[0]), "mountpoint": _unescape_mount(parts[1]),
                                "fstype": parts[2], "options": parts[3]})
    except OSError:
        pass
    return out


def _mount_of(path: str, mounts: Optional[List[Dict[str, str]]] = None) -> Optional[Dict[str, str]]:
    """The mounts entry that contains `path` (longest mountpoint prefix)."""
    mounts = mounts if mounts is not None else read_mounts()
    rp = os.path.realpath(path)
    best = None
    for m in mounts:
        mp = m["mountpoint"]
        if rp == mp or rp.startswith(mp.rstrip("/") + "/") or mp == "/":
            if best is None or len(mp) > len(best["mountpoint"]):
                best = m
    return best


def label_device(label: str = DRIVE_LABEL, by_label_dir: str = "/dev/disk/by-label") -> Optional[str]:
    """/dev/sdX1 of the partition with this filesystem label (present even when it is not mounted).
    V4.6: case-insensitive (the user's drive is labelled "Embodied_AI"; V4.4/4.5 looked for "EMBODIED_AI" only)."""
    d = by_label_dir
    p = os.path.join(d, label)
    if os.path.exists(p):
        return os.path.realpath(p)
    try:
        for name in os.listdir(d):
            if name.replace("\\x20", " ").lower() == label.lower():
                return os.path.realpath(os.path.join(d, name))
    except OSError:
        pass
    return None


def find_drive(label: str = DRIVE_LABEL) -> Dict[str, Any]:
    """Locate the EMBODIED_AI drive. Returns {found, mountpoint, device, fstype, options, source, hint}."""
    forced = os.environ.get("NERO_DRIVE_ROOT", "").strip()
    if forced:
        fp = os.path.abspath(os.path.expanduser(forced))
        m = _mount_of(fp) or {}
        return {"found": os.path.isdir(fp), "mountpoint": fp, "device": m.get("device"), "fstype": m.get("fstype", "?"),
                "options": m.get("options", ""), "source": "NERO_DRIVE_ROOT", "label": label,
                "hint": "" if os.path.isdir(fp) else f"NERO_DRIVE_ROOT={fp} 不存在"}
    mounts = read_mounts()
    dev = label_device(label)
    cands: List[Dict[str, str]] = []
    for m in mounts:
        if os.path.basename(m["mountpoint"].rstrip("/")).lower() == label.lower():
            cands.append(m)
        elif dev and os.path.realpath(m["device"]) == dev and m not in cands:
            cands.append(m)
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    for p in (f"/media/{user}/{label}", f"/run/media/{user}/{label}", f"/media/{label}", f"/mnt/{label}"):
        if os.path.ismount(p) and not any(c["mountpoint"] == p for c in cands):
            m = _mount_of(p, mounts) or {"device": "?", "fstype": "?", "options": ""}
            cands.append({"device": m["device"], "mountpoint": p, "fstype": m["fstype"], "options": m["options"]})
    for c in cands:
        mp = c["mountpoint"]
        if os.path.ismount(mp) and os.path.isdir(mp):
            return {"found": True, "mountpoint": mp, "device": c["device"], "fstype": c["fstype"],
                    "options": c["options"], "source": "mounts", "label": label, "hint": ""}
    hint = f"未检测到已挂载的移动硬盘 {label}。"
    if dev:
        hint += f" 系统里有这个分区 ({dev}) 但没有挂载：在文件管理器里点一下 {label}，或点「挂载 {label}」。"
    else:
        hint += f" 请插上硬盘；如果已插上，请确认分区卷标是 {label}（sudo blkid 可查看）。"
    return {"found": False, "mountpoint": None, "device": dev, "fstype": None, "options": "", "source": None,
            "label": label, "hint": hint}


def mount_drive(label: str = DRIVE_LABEL, timeout: float = 30.0) -> Dict[str, Any]:
    """Mount the labelled partition via udisksctl (no sudo needed on a desktop session)."""
    dev = label_device(label)
    if not dev:
        return {"ok": False, "error": f"找不到卷标为 {label} 的分区"}
    exe = shutil.which("udisksctl")
    if not exe:
        return {"ok": False, "error": "没有 udisksctl（sudo apt install udisks2），请在文件管理器里手动挂载"}
    try:
        r = subprocess.run([exe, "mount", "-b", dev], capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    msg = (r.stdout or r.stderr or "").strip()
    return {"ok": r.returncode == 0, "message": msg, "device": dev, **({} if r.returncode == 0 else {"error": msg})}


def unmount_drive(label: str = DRIVE_LABEL, power_off: bool = True, timeout: float = 60.0) -> Dict[str, Any]:
    drv = find_drive(label)
    dev = drv.get("device")
    exe = shutil.which("udisksctl")
    if not drv.get("found") or not dev or not exe or drv.get("source") == "NERO_DRIVE_ROOT":
        return {"ok": False, "error": "无法自动卸载（没有挂载 / 没有 udisksctl / 使用了 NERO_DRIVE_ROOT），请在文件管理器里点「弹出」"}
    try:
        os.sync()
    except Exception:
        pass
    r = subprocess.run([exe, "unmount", "-b", dev], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        return {"ok": False, "error": (r.stderr or r.stdout).strip() + "（还有程序在用硬盘上的文件？先关闭相关终端 / 文件管理器窗口）"}
    out = {"ok": True, "message": (r.stdout or "").strip()}
    if power_off:
        disk = dev.rstrip("0123456789")
        if disk.endswith("p") and ("nvme" in disk or "mmcblk" in disk):
            disk = disk[:-1]
        r2 = subprocess.run([exe, "power-off", "-b", disk], capture_output=True, text=True, timeout=timeout)
        out["power_off"] = r2.returncode == 0
    return out


# ----------------------------------------------------------------------------------------------- layout
def allow_internal() -> bool:
    return os.environ.get("NERO_V44_ALLOW_INTERNAL", "0") == "1"


def layout(drive: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """All paths. `on_drive` False -> the drive is missing (paths point to the legacy internal folders)."""
    drive = drive if drive is not None else find_drive()
    if drive.get("found"):
        root = os.path.join(drive["mountpoint"], LAYOUT_DIRNAME)
        on_drive = True
    else:
        root = os.path.expanduser(os.environ.get("NERO_V44_INTERNAL_ROOT", "~/nero_lerobot_internal"))
        on_drive = False
    j = lambda *p: os.path.join(root, *p)  # noqa: E731
    L = {
        "on_drive": on_drive,
        "drive": drive,
        "root": root,
        "venv": j("envs", "lerobot"),
        "uv_boot": j("envs", "_uv"),
        "uv_python_dir": j("python"),
        "uv_cache": j("cache", "uv"),
        "venv_python": j("envs", "lerobot", "bin", "python"),
        "hf_home": j("hf_home"),
        "hf_hub_cache": j("hf_home", "hub"),
        "lerobot_home": j("hf_home", "lerobot"),
        "models": j("models"),
        "smolvla_base_dir": j("models", "smolvla_base"),
        "datasets_nero": j("datasets", "nero") if on_drive else LEGACY_DATASET_ROOT,
        "datasets_lerobot": j("datasets", "lerobot"),
        "checkpoints": j("checkpoints") if on_drive else LEGACY_CKPT_ROOT,
        "training": j("training"),
        "pip_cache": j("cache", "pip"),
        "torch_home": j("cache", "torch"),
        "xdg_cache": j("cache", "xdg"),
        "tmp": j("tmp"),
        "logs": j("logs"),
        "activate_sh": j("activate_nero_lerobot.sh"),
        "marker": j("nero_storage.json"),
        "job_file": j("logs", "current_job.json"),
    }
    return L


def ensure_layout(L: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    L = L or layout()
    if not L["on_drive"] and not allow_internal():
        return {"ok": False, "error": L["drive"].get("hint") or f"移动硬盘 {DRIVE_LABEL} 未挂载"}
    for d in SUBDIRS:
        os.makedirs(os.path.join(L["root"], d), exist_ok=True)
    meta = {"layout_version": LAYOUT_VERSION, "nero_version": VERSION, "label": DRIVE_LABEL,
            "created_or_checked": time.strftime("%Y-%m-%d %H:%M:%S"), "subdirs": list(SUBDIRS)}
    with open(L["marker"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    write_activate_script(L)
    return {"ok": True, "root": L["root"]}


def storage_env(L: Optional[Dict[str, Any]] = None, for_heavy_job: bool = True) -> Dict[str, str]:
    """Environment variables that send every LeRobot / HF / torch / pip file onto the drive."""
    L = L or layout()
    env = {
        "HF_HOME": L["hf_home"],
        "HF_HUB_CACHE": L["hf_hub_cache"],
        "HF_LEROBOT_HOME": L["lerobot_home"],
        "TORCH_HOME": L["torch_home"],
        "PIP_CACHE_DIR": L["pip_cache"],
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "NERO_V4_DATASET_DIR": L["datasets_nero"],
        "NERO_V4_CKPT_DIR": L["checkpoints"],
        "NERO_LEROBOT_PYTHON": L["venv_python"],
        "WANDB_MODE": "disabled",
    }
    if for_heavy_job:
        env.update({"UV_CACHE_DIR": L["uv_cache"], "UV_PYTHON_INSTALL_DIR": L["uv_python_dir"]})
        env.update({"TMPDIR": L["tmp"], "XDG_CACHE_HOME": L["xdg_cache"], "TRITON_CACHE_DIR": os.path.join(L["xdg_cache"], "triton")})
    return env


def child_env(L: Optional[Dict[str, Any]] = None, for_heavy_job: bool = True) -> Dict[str, str]:
    """os.environ + storage_env, with ROS / system PYTHONPATH removed so the venv sees only its own packages."""
    env = dict(os.environ)
    # LEROBOT_HOME: deprecated - LeRobot >= 0.5 refuses to import while it is set (found in the V4.4 real install test)
    for k in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "CONDA_PREFIX", "LEROBOT_HOME"):
        env.pop(k, None)
    env.update(storage_env(L, for_heavy_job=for_heavy_job))
    L = L or layout()
    env["VIRTUAL_ENV"] = L["venv"]
    env["PATH"] = os.path.join(L["venv"], "bin") + os.pathsep + env.get("PATH", "")
    return env


def apply_to_process_env(L: Optional[Dict[str, Any]] = None) -> List[str]:
    """Point this (console) process at the drive too, without overriding what the user exported themselves."""
    L = L or layout()
    if not L["on_drive"]:
        return []
    changed = []
    # V4.5: setenv() is not thread-safe against getenv() in C libraries (Tk / X11 / CUDA / librealsense threads):
    # write only when a value really changes (normally once per drive mount), and only from the Tk main thread.
    for k, v in storage_env(L, for_heavy_job=False).items():
        if os.environ.get(k) == v:
            continue
        if k in ("NERO_LEROBOT_PYTHON",) or not os.environ.get(k) or os.environ.get(k) == os.environ.get("_NERO_V44_SET_" + k):
            os.environ[k] = v
            os.environ["_NERO_V44_SET_" + k] = v
            changed.append(k)
    return changed


def write_activate_script(L: Dict[str, Any]) -> str:
    lines = ["#!/usr/bin/env bash", f"# NERO {VERSION}: use the LeRobot + SmolVLA environment on the {DRIVE_LABEL} drive.",
             "#   source " + shlex.quote(L["activate_sh"]), ""]
    for k, v in storage_env(L, for_heavy_job=True).items():
        lines.append(f"export {k}={shlex.quote(v)}")
    lines += ["unset PYTHONPATH LEROBOT_HOME   # keep ROS python packages out of the venv; LEROBOT_HOME is deprecated",
              f"if [ -f {shlex.quote(os.path.join(L['venv'], 'bin', 'activate'))} ]; then",
              f"  source {shlex.quote(os.path.join(L['venv'], 'bin', 'activate'))}",
              "else", "  echo '[WARN] venv not installed yet (NERO console -> Smol VLA实机任务 -> 安装 LeRobot+SmolVLA)'", "fi",
              f"cd {shlex.quote(L['root'])}", ""]
    with open(L["activate_sh"], "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    try:
        os.chmod(L["activate_sh"], 0o755)
    except OSError:
        pass
    return L["activate_sh"]


# ----------------------------------------------------------------------------------------------- checks
def _fmt_gb(b: float) -> str:
    return f"{b / 1e9:.1f} GB"


def fs_checks(path: str) -> Dict[str, Any]:
    """Free space, writability, symlinks, exec permission and file-system type of `path`."""
    out: Dict[str, Any] = {"path": path, "writable": False, "symlinks": False, "exec": False, "free_gb": 0.0,
                           "total_gb": 0.0, "fstype": "?", "noexec": False, "grade": "bad", "notes": []}
    if not os.path.isdir(path):
        out["notes"].append("路径不存在")
        return out
    try:
        du = shutil.disk_usage(path)
        out["free_gb"], out["total_gb"] = round(du.free / 1e9, 1), round(du.total / 1e9, 1)
    except OSError:
        pass
    m = _mount_of(path) or {}
    fst = (m.get("fstype") or "?").lower()
    out["fstype"] = fst
    out["noexec"] = "noexec" in (m.get("options") or "").split(",")
    try:
        d = tempfile.mkdtemp(prefix=".nero_fscheck_", dir=path)
    except OSError as exc:
        out["notes"].append(f"不可写: {exc}")
        return out
    try:
        out["writable"] = True
        tgt = os.path.join(d, "t.txt")
        with open(tgt, "w") as f:
            f.write("x")
        try:
            os.symlink(tgt, os.path.join(d, "lnk"))
            out["symlinks"] = os.path.islink(os.path.join(d, "lnk"))
        except OSError:
            out["symlinks"] = False
        sh = os.path.join(d, "x.sh")
        with open(sh, "w") as f:
            f.write("#!/bin/sh\necho ok\n")
        try:
            os.chmod(sh, 0o755)
            r = subprocess.run([sh], capture_output=True, text=True, timeout=5)
            out["exec"] = r.returncode == 0 and r.stdout.strip() == "ok"
        except Exception:
            out["exec"] = False
    finally:
        shutil.rmtree(d, ignore_errors=True)
    notes = out["notes"]
    if fst in BAD_FS:
        notes.append(f"{fst} (FAT32) 单个文件最大 4 GB、不支持符号链接和权限，不能装 Python 环境 —— 请把硬盘格式化为 ext4")
    elif fst in SOSO_FS:
        notes.append(f"{fst} 可以用（venv 用 --copies 复制模式），但比 ext4 慢且不区分权限；长期训练建议格式化为 ext4")
    elif fst not in GOOD_FS and fst != "?":
        notes.append(f"文件系统 {fst} 未验证过")
    if out["noexec"] or not out["exec"]:
        notes.append("硬盘以 noexec 挂载，Python 程序不能在上面运行 —— 请卸载后重新挂载（去掉 noexec），或在 /etc/fstab 里加 exec")
    if out["free_gb"] < MIN_FREE_GB_INSTALL:
        notes.append(f"剩余空间 {out['free_gb']} GB < 建议的 {MIN_FREE_GB_INSTALL:.0f} GB（实测装完约 16.5 GB，另需数据 / 训练空间）")
    ok_exec = out["exec"] and not out["noexec"]
    if out["writable"] and ok_exec and fst not in BAD_FS:
        out["grade"] = "good" if (fst in GOOD_FS and out["symlinks"]) else "ok"
    return out


VENV_PROBE_TIMEOUT_S = 180.0   # V4.6: the first torch import from a cold USB drive took > 60 s on the user's laptop


def venv_status(L: Optional[Dict[str, Any]] = None, timeout: float = VENV_PROBE_TIMEOUT_S) -> Dict[str, Any]:
    """Probe the drive venv: python / torch / CUDA / lerobot / SmolVLAPolicy (runs inside the venv, 5-120 s)."""
    L = L or layout()
    py = L["venv_python"]
    st: Dict[str, Any] = {"exists": os.path.isfile(py) and os.access(py, os.X_OK), "python": py}
    if not st["exists"]:
        return st
    code = r"""
import json, sys
o = {"python_version": sys.version.split()[0]}
try:
    import torch
    o["torch"] = torch.__version__; o["cuda"] = bool(torch.cuda.is_available())
    if o["cuda"]:
        o["gpu"] = torch.cuda.get_device_name(0)
        o["vram_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**20)
except Exception as e:
    o["torch_error"] = repr(e)[:200]
try:
    import lerobot
    o["lerobot"] = getattr(lerobot, "__version__", None)
    if not o["lerobot"]:
        try:
            from importlib.metadata import version
            o["lerobot"] = version("lerobot")
        except Exception:
            o["lerobot"] = "installed"
except Exception as e:
    o["lerobot_error"] = repr(e)[:200]
for m in ("lerobot.policies.smolvla.modeling_smolvla", "lerobot.common.policies.smolvla.modeling_smolvla"):
    try:
        mod = __import__(m, fromlist=["SmolVLAPolicy"])
        if getattr(mod, "SmolVLAPolicy", None) is not None:
            o["smolvla_module"] = m; break
    except Exception as e:
        o.setdefault("smolvla_error", repr(e)[:300])
for m in ("lerobot.datasets.lerobot_dataset", "lerobot.common.datasets.lerobot_dataset"):
    try:
        __import__(m, fromlist=["LeRobotDataset"]); o["dataset_module"] = m; break
    except Exception as e:
        o.setdefault("dataset_error", repr(e)[:300])
try:
    import huggingface_hub
    o["hf_hub"] = huggingface_hub.__version__
except Exception:
    pass
print("NERO_JSON=" + json.dumps(o))
"""
    try:
        r = subprocess.run([py, "-I", "-c", code], capture_output=True, text=True, timeout=timeout,
                           env=child_env(L, for_heavy_job=False))
        for line in (r.stdout or "").splitlines():
            if line.startswith("NERO_JSON="):
                st.update(json.loads(line[len("NERO_JSON="):]))
                break
        else:
            st["error"] = (r.stderr or r.stdout or f"returncode={r.returncode}").strip()[-400:]
    except subprocess.TimeoutExpired:
        st["error"] = (f"验证超时（{timeout:.0f} s）：第一次从移动硬盘导入 torch 很慢，再点一次「验证环境」通常就好了；"
                       "若一直超时请检查 USB 接口 / 硬盘")
        st["timeout"] = True
    except Exception as exc:
        st["error"] = str(exc)
    st["ready"] = bool(st.get("lerobot") and st.get("smolvla_module") and st.get("torch"))
    return st


def has_weights(d: str) -> bool:
    try:
        return os.path.isdir(d) and any(fn.endswith((".safetensors", ".bin")) for fn in os.listdir(d)) \
            and os.path.isfile(os.path.join(d, "config.json"))
    except OSError:
        return False


def _user_env(key: str) -> Optional[str]:
    """A path the USER exported (not one apply_to_process_env() set for an earlier drive mount)."""
    v = os.environ.get(key, "").strip()
    if not v or v == os.environ.get("_NERO_V44_SET_" + key):
        return None
    return os.path.expanduser(v)


def default_dataset_root() -> str:
    return _user_env("NERO_V4_DATASET_DIR") or layout()["datasets_nero"]


def default_checkpoint_root() -> str:
    return _user_env("NERO_V4_CKPT_DIR") or layout()["checkpoints"]


def resolve_model_target(target: str, L: Optional[Dict[str, Any]] = None) -> str:
    """'lerobot/smolvla_base' -> the local copy on the drive when it has been downloaded."""
    L = L or layout()
    t = (target or "").strip() or DEFAULT_BASE_MODEL_ID
    if t == DEFAULT_BASE_MODEL_ID:
        for d in (L["smolvla_base_dir"], LEGACY_MODEL_DIR):
            if has_weights(d):
                return d
    return os.path.expanduser(t)


def dir_size_bytes(path: str, limit_s: float = 20.0) -> int:
    tot, t0 = 0, time.time()
    for dp, _dn, fns in os.walk(path):
        for fn in fns:
            try:
                fp = os.path.join(dp, fn)
                if not os.path.islink(fp):
                    tot += os.path.getsize(fp)
            except OSError:
                pass
        if time.time() - t0 > limit_s:
            break
    return tot


def latest_training_checkpoint(L: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Newest <training>/<run>/checkpoints/<step|last>/pretrained_model written by lerobot-train."""
    L = L or layout()
    best, best_t = None, -1.0
    if not os.path.isdir(L["training"]):
        return None
    for run in os.listdir(L["training"]):
        cdir = os.path.join(L["training"], run, "checkpoints")
        if not os.path.isdir(cdir):
            continue
        for step in os.listdir(cdir):
            pm = os.path.join(cdir, step, "pretrained_model")
            if has_weights(pm):
                t = os.path.getmtime(pm)
                if t > best_t:
                    best, best_t = pm, t
    return best


def status(probe_venv: bool = False) -> Dict[str, Any]:
    drv = find_drive()
    L = layout(drv)
    st: Dict[str, Any] = {"drive": drv, "layout": L, "fs": fs_checks(drv["mountpoint"]) if drv.get("found") else None,
                          "initialized": os.path.isfile(L["marker"]), "venv_exists": os.path.isfile(L["venv_python"]),
                          "model_local": has_weights(L["smolvla_base_dir"]), "latest_train_ckpt": latest_training_checkpoint(L)}
    if probe_venv:
        st["venv"] = venv_status(L)
    return st


def format_status(st: Dict[str, Any]) -> str:
    drv, L, fs = st["drive"], st["layout"], st.get("fs")
    if not drv.get("found"):
        line1 = f"✗ 移动硬盘 {DRIVE_LABEL}: {drv.get('hint')}  (现在数据集 / Checkpoint 临时存放在本机: {L['datasets_nero']})"
    else:
        grade = {"good": "✓", "ok": "△", "bad": "✗"}.get(fs["grade"] if fs else "bad", "?")
        line1 = (f"{grade} 移动硬盘 {DRIVE_LABEL}: {drv['mountpoint']} ({drv.get('device') or '?'}, {fs['fstype'] if fs else '?'}) · "
                 f"剩余 {fs['free_gb'] if fs else '?'} / {fs['total_gb'] if fs else '?'} GB · 存储根目录 {L['root']}"
                 f"{'' if st['initialized'] else '（未初始化）'}")
        if fs and fs["notes"]:
            line1 += "  ⚠ " + "；".join(fs["notes"])
    v = st.get("venv")
    if v is None:
        line2 = f"LeRobot 环境: {'已存在 ' + L['venv'] if st['venv_exists'] else '未安装'}（点「验证环境」查看 torch / CUDA / lerobot 版本）"
    elif not v.get("exists"):
        line2 = "LeRobot 环境: ✗ 未安装（点「安装 LeRobot+SmolVLA 到移动硬盘」）"
    else:
        cuda = f"CUDA ✓ {v.get('gpu')} {v.get('vram_mb')} MB" if v.get("cuda") else "CUDA ✗"
        line2 = (f"LeRobot 环境: {'✓ 就绪' if v.get('ready') else '✗ 不完整'} · Python {v.get('python_version')} · "
                 f"torch {v.get('torch') or '✗'} · {cuda} · lerobot {v.get('lerobot') or '✗'} · "
                 f"SmolVLAPolicy {'✓' if v.get('smolvla_module') else '✗'} · "
                 f"LeRobotDataset {'✓' if v.get('dataset_module') else '✗ (导出/训练需要，重新点「安装」补装)'}"
                 + (f" · 错误: {v.get('error') or v.get('lerobot_error') or v.get('smolvla_error')}" if not v.get("ready") else ""))
    line3 = (f"SmolVLA 模型: {'✓ ' + L['smolvla_base_dir'] if st['model_local'] else '✗ 未下载到硬盘'} · "
             f"数据集: {L['datasets_nero']} · 训练输出: {L['training']}"
             + (f" · 最新训练结果: {st['latest_train_ckpt']}" if st.get("latest_train_ckpt") else ""))
    return "\n".join([line1, line2, line3])


# ----------------------------------------------------------------------------------------------- legacy migration
def legacy_locations() -> List[Dict[str, Any]]:
    """Old internal-disk folders used by V4.0-V4.3 (and HF / pip caches that may hold partial SmolVLA downloads)."""
    hub = os.path.expanduser(os.path.join(_ORIG_HF_HOME or "~/.cache/huggingface", "hub"))
    items = [
        {"key": "datasets", "path": LEGACY_DATASET_ROOT, "dest_key": "datasets_nero", "movable": True},
        {"key": "checkpoints", "path": LEGACY_CKPT_ROOT, "dest_key": "checkpoints", "movable": True},
        {"key": "smolvla_base", "path": LEGACY_MODEL_DIR, "dest_key": "smolvla_base_dir", "movable": True},
    ]
    if os.path.isdir(hub):
        for name in sorted(os.listdir(hub)):
            if name.startswith(("models--lerobot--", "models--HuggingFaceTB--SmolVLM", "datasets--lerobot--")):
                items.append({"key": "hf_cache", "path": os.path.join(hub, name), "dest_key": "hf_hub_cache", "movable": True})
    items.append({"key": "pip_cache", "path": os.path.expanduser("~/.cache/pip"), "dest_key": None, "movable": False})
    out = []
    for it in items:
        if os.path.isdir(it["path"]):
            it["size_gb"] = round(dir_size_bytes(it["path"]) / 1e9, 2)
            out.append(it)
    return out


def migrate_legacy(L: Optional[Dict[str, Any]] = None, delete_source: bool = False,
                   log=print) -> Dict[str, Any]:
    """Copy old internal-disk datasets / checkpoints / models / HF cache entries onto the drive (then optionally delete)."""
    L = L or layout()
    if not L["on_drive"]:
        return {"ok": False, "error": "移动硬盘未挂载"}
    ensure_layout(L)
    moved, failed = [], []
    for it in legacy_locations():
        if not it["movable"]:
            continue
        src = it["path"]
        if it["key"] in ("datasets", "checkpoints"):
            dst_parent = L[it["dest_key"]]
            dsts = [(os.path.join(src, n), os.path.join(dst_parent, n)) for n in os.listdir(src)]
        elif it["key"] == "hf_cache":
            dsts = [(src, os.path.join(L["hf_hub_cache"], os.path.basename(src)))]
        else:
            dsts = [(src, L[it["dest_key"]])]
        for s, d in dsts:
            try:
                if os.path.realpath(s) == os.path.realpath(d):
                    continue
                if os.path.isdir(s):
                    shutil.copytree(s, d, symlinks=False, dirs_exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(d), exist_ok=True)
                    shutil.copy2(s, d)
                moved.append((s, d))
                log(f"[OK] [V4.4 MIGRATE] {s} -> {d}")
            except Exception as exc:
                failed.append((s, str(exc)))
                log(f"[FAIL] [V4.4 MIGRATE] {s}: {exc}")
    if delete_source and not failed:
        delete_sources(moved, log=log)
    return {"ok": not failed, "moved": moved, "failed": failed}


def delete_sources(moved: List[Any], log=print) -> int:
    """Delete the internal-disk originals of a successful migrate_legacy() (frees the internal disk)."""
    n = 0
    for s, _d in moved:
        try:
            shutil.rmtree(s) if os.path.isdir(s) else os.unlink(s)
            n += 1
            log(f"[OK] [V4.4 MIGRATE] 已删除本机旧文件 {s}")
        except Exception as exc:
            log(f"[WARN] [V4.4 MIGRATE] 删除 {s} 失败: {exc}")
    return n


# ----------------------------------------------------------------------------------------------- job scripts
def base_python_check(base_py: str) -> Dict[str, Any]:
    if not (os.path.isfile(base_py) and os.access(base_py, os.X_OK)):
        return {"ok": False, "error": f"找不到 Python: {base_py}"}
    r = subprocess.run([base_py, "-I", "-c", "import sys, venv, ensurepip; print(sys.version.split()[0])"],
                       capture_output=True, text=True, timeout=20)
    if r.returncode != 0:
        ver = subprocess.run([base_py, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                             capture_output=True, text=True, timeout=10).stdout.strip() or "3"
        return {"ok": False, "error": f"{base_py} 缺少 venv / ensurepip，请先执行: sudo apt install python{ver}-venv"}
    v = r.stdout.strip()
    major, minor = (int(x) for x in v.split(".")[:2])
    if (major, minor) < (3, 10):
        return {"ok": False, "error": f"LeRobot 需要 Python ≥ 3.10，{base_py} 是 {v}"}
    return {"ok": True, "version": v}


def install_script(L: Dict[str, Any], base_py: str = "/usr/bin/python3", pip_spec: str = DEFAULT_PIP_SPEC,
                   download_model: bool = True, worker_path: Optional[str] = None,
                   extra_pip_args: str = "", py_mode: str = DEFAULT_PY_MODE) -> str:
    """Bash script that creates the venv ON THE DRIVE and pip-installs lerobot[smolvla] (all caches / temp on the drive).
    py_mode "uv312": private Python 3.12 from `uv` on the drive (newest LeRobot); "system": venv from `base_py`."""
    fs = fs_checks(L["root"]) if os.path.isdir(L["root"]) else {"symlinks": False}
    copies = "" if fs.get("symlinks") else "--copies"
    if py_mode == "uv312" and not fs.get("symlinks"):
        py_mode = "system_nosymlink"   # uv venvs need symlinks (exFAT / NTFS): use the system Python
    q = shlex.quote
    env_lines = "\n".join(f"export {k}={q(v)}" for k, v in storage_env(L, for_heavy_job=True).items())
    dl = ""
    if download_model and worker_path:
        dl = (f'echo "=== [4/4] 下载 SmolVLA 模型到 {L["smolvla_base_dir"]} ==="\n'
              f'"$VENV/bin/python" -I {q(worker_path)} download --repo {q(DEFAULT_BASE_MODEL_ID)} --local-dir {q(L["smolvla_base_dir"])}\n')
    return f"""#!/usr/bin/env bash
# NERO {VERSION} — install LeRobot + SmolVLA onto the {DRIVE_LABEL} drive (generated; safe to re-run).
set -eo pipefail
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX LEROBOT_HOME
{env_lines}
BASE_PY={q(base_py)}
VENV={q(L["venv"])}
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME"
echo "=== NERO {VERSION} LeRobot 安装开始 $(date '+%F %T') ==="
echo "    venv      : $VENV"
echo "    pip cache : $PIP_CACHE_DIR"
echo "    TMPDIR    : $TMPDIR"
df -h {q(L["root"])} || true
PY_MODE={q(py_mode)}
WANT_VER=""
[ "$PY_MODE" = "uv312" ] && WANT_VER={q(UV_PYTHON_VERSION)}
if [ "$PY_MODE" = "system_nosymlink" ]; then
  echo "[WARN] 这个文件系统不支持符号链接，不能用 uv 的 Python {UV_PYTHON_VERSION}；改用系统 Python（LeRobot 版本受系统 Python 限制）"
  PY_MODE=system
fi
make_venv_system() {{
  echo "=== [1/4] 用系统 Python 创建虚拟环境: $BASE_PY ({copies or 'symlinks'}) ==="
  "$BASE_PY" -m venv --clear {copies} "$VENV"
}}
make_venv_uv() {{
  echo "=== [1/4] 用 uv 下载 Python {UV_PYTHON_VERSION} 到移动硬盘 ($UV_PYTHON_INSTALL_DIR) 并创建虚拟环境 ==="
  BOOT={q(L["uv_boot"])}
  if [ ! -x "$BOOT/bin/uv" ]; then
    "$BASE_PY" -m venv --clear "$BOOT" || return 1
    "$BOOT/bin/python" -I -m pip install --upgrade uv || return 1
  fi
  "$BOOT/bin/uv" python install {UV_PYTHON_VERSION} || return 1
  "$BOOT/bin/uv" venv --seed --allow-existing --python-preference only-managed --python {UV_PYTHON_VERSION} "$VENV" || return 1
}}
venv_ok() {{
  [ -x "$VENV/bin/python" ] || return 1
  "$VENV/bin/python" -I -m pip --version >/dev/null 2>&1 || return 1
  if [ -n "$WANT_VER" ]; then
    have=$("$VENV/bin/python" -I -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)
    [ "$have" = "$WANT_VER" ] || return 1
  fi
  return 0
}}
if venv_ok; then
  echo "=== [1/4] 虚拟环境已存在 ($("$VENV/bin/python" -I -V 2>&1))，跳过创建（继续安装 / 升级） ==="
else
  if [ -e "$VENV" ]; then echo "=== 旧的 / 不完整的虚拟环境，重新创建 ==="; rm -rf "$VENV"; fi
  if [ "$PY_MODE" = "uv312" ]; then
    if ! make_venv_uv; then
      echo "[WARN] uv / Python {UV_PYTHON_VERSION} 安装失败（github.com 不通？）；改用系统 Python $BASE_PY"
      rm -rf "$VENV"
      make_venv_system
    fi
  else
    make_venv_system
  fi
fi
echo "    Python    : $("$VENV/bin/python" -I -V 2>&1)"
echo "=== [2/4] 升级 pip ==="
"$VENV/bin/python" -I -m pip install --upgrade pip setuptools wheel
echo "=== [3/4] 安装 {pip_spec} (torch + CUDA 库约 7 GB，请耐心等待) ==="
"$VENV/bin/python" -I -m pip install {extra_pip_args} {q(pip_spec)}
"$VENV/bin/python" -I -c "import torch, lerobot; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('lerobot OK')"
{dl}df -h {q(L["root"])} || true
echo "NERO_JOB_DONE ok"
"""


NERO_CAMERA_KEYS = ("observation.images.cam_main", "observation.images.cam_wrist")


def camera_rename_map(policy_dir: str) -> Dict[str, str]:
    """NERO dataset camera keys -> the start model's camera keys (smolvla_base: camera1 / camera2 / camera3).
    LeRobot accepts a dataset that provides a SUBSET of the policy's cameras. Main camera -> first key, wrist ->
    second; the inference worker uses the same rule (keys containing 'camera2' / 'wrist' get the wrist image)."""
    try:
        with open(os.path.join(policy_dir, "config.json"), "r", encoding="utf-8") as f:
            feats = json.load(f).get("input_features") or {}
    except Exception:
        return {}
    imgs = sorted(k for k, v in feats.items() if isinstance(v, dict) and str(v.get("type", "")).upper() == "VISUAL")
    if not imgs or all(k in imgs for k in NERO_CAMERA_KEYS):
        return {}
    wrist = next((k for k in imgs if "wrist" in k.lower() or "camera2" in k.lower()), imgs[min(1, len(imgs) - 1)])
    main = next((k for k in imgs if k != wrist), imgs[0])
    return {NERO_CAMERA_KEYS[0]: main, NERO_CAMERA_KEYS[1]: wrist}


def train_command(L: Dict[str, Any], dataset_repo_id: str, dataset_root: str, run_name: str,
                  policy_path: str = DEFAULT_BASE_MODEL_ID, steps: int = 20000, batch_size: int = 8,
                  save_freq: int = 2000, device: str = "cuda", extra_args: str = "") -> List[str]:
    """`lerobot-train` command (LeRobot ≥ 0.3 CLI; older versions use `python -m lerobot.scripts.train`)."""
    out_dir = os.path.join(L["training"], run_name)
    pol = resolve_model_target(policy_path, L)
    exe = os.path.join(L["venv"], "bin", "lerobot-train")
    base = [exe] if os.path.isfile(exe) else [L["venv_python"], "-I", "-m", "lerobot.scripts.train"]
    cmd = base + [
        f"--policy.path={pol}",
        f"--dataset.repo_id={dataset_repo_id}",
        f"--dataset.root={dataset_root}",
        f"--output_dir={out_dir}",
        f"--job_name={run_name}",
        f"--batch_size={int(batch_size)}",
        f"--steps={int(steps)}",
        f"--save_freq={int(save_freq)}",
        f"--policy.device={device}",
        "--policy.push_to_hub=false",
    ]
    rmap = camera_rename_map(pol) if os.path.isdir(pol) else {}
    if rmap and "--rename_map" not in extra_args:
        cmd.append("--rename_map=" + json.dumps(rmap))
    if extra_args.strip():
        cmd += shlex.split(extra_args)
    return cmd


class BackgroundJob:
    """A long job (install / download / export / train) running DETACHED from the console: output goes to a log file
    on the drive, so closing the console does not kill a multi-hour training run. The GUI tails the log."""

    def __init__(self, L: Dict[str, Any], log=print):
        self.L = L
        self.log = log
        self.proc: Optional[subprocess.Popen] = None
        self.pid: Optional[int] = None
        self.kind: str = ""
        self.log_path: Optional[str] = None
        self._pos = 0
        self.started_at = 0.0
        self.meta: Dict[str, Any] = {}

    # --- lifecycle
    def running(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        if self.pid:
            try:
                os.kill(self.pid, 0)
                return True
            except OSError:
                return False
        return False

    def returncode(self) -> Optional[int]:
        return self.proc.poll() if self.proc is not None else None

    def start(self, kind: str, cmd: List[str], env: Optional[Dict[str, str]] = None, meta: Optional[Dict[str, Any]] = None,
              cwd: Optional[str] = None) -> Dict[str, Any]:
        if self.running():
            return {"ok": False, "error": f"已有后台任务在运行: {self.kind}"}
        os.makedirs(self.L["logs"], exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(self.L["logs"], f"{kind}_{ts}.log")
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(f"# NERO {VERSION} {kind} {time.strftime('%F %T')}\n# cmd: {' '.join(shlex.quote(c) for c in cmd)}\n")
        lf = open(self.log_path, "ab")
        try:
            self.proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         env=env, cwd=cwd or self.L["root"], start_new_session=True)
        finally:
            lf.close()
        self.pid = self.proc.pid
        self.kind = kind
        self._pos = 0
        self.started_at = time.time()
        self.meta = dict(meta or {})
        try:
            with open(self.L["job_file"], "w", encoding="utf-8") as f:
                json.dump({"pid": self.pid, "kind": kind, "log": self.log_path, "started": self.started_at, "meta": self.meta}, f)
        except OSError:
            pass
        return {"ok": True, "pid": self.pid, "log": self.log_path}

    def adopt_previous(self) -> bool:
        """Re-attach to a job still running from an earlier console session (reads logs/current_job.json)."""
        try:
            with open(self.L["job_file"], "r", encoding="utf-8") as f:
                j = json.load(f)
            os.kill(int(j["pid"]), 0)
        except Exception:
            return False
        self.pid, self.kind, self.log_path = int(j["pid"]), j.get("kind", "?"), j.get("log")
        self.started_at, self.meta = float(j.get("started", time.time())), j.get("meta") or {}
        try:  # show only the recent tail of a long log
            self._pos = max(0, os.path.getsize(self.log_path) - 4096) if self.log_path else 0
        except OSError:
            self._pos = 0
        return True

    def stop(self) -> bool:
        if not self.running():
            return False
        pid = self.proc.pid if self.proc is not None else self.pid
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                return False
        return True

    def read_new(self, max_bytes: int = 64 * 1024) -> str:
        if not self.log_path or not os.path.isfile(self.log_path):
            return ""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(self._pos)
                data = f.read(max_bytes)
                self._pos += len(data)
        except OSError:
            return ""
        # pip progress bars use \r; keep the last state of each line only
        txt = data.decode("utf-8", errors="replace")
        return "\n".join(seg.split("\r")[-1] for seg in txt.split("\n"))


# ----------------------------------------------------------------------------------------------- CLI
def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=f"NERO {VERSION} EMBODIED_AI drive storage")
    ap.add_argument("--brief", action="store_true", help="one-line status (start_nero.sh)")
    ap.add_argument("--init", action="store_true", help="create the folder layout on the drive")
    ap.add_argument("--probe", action="store_true", help="also probe the venv (torch / CUDA / lerobot)")
    a = ap.parse_args(argv)
    st = status(probe_venv=a.probe)
    if a.brief:
        d = st["drive"]
        if d.get("found"):
            fs = st["fs"] or {}
            print(f"[OK] V4.4 drive {DRIVE_LABEL}: {d['mountpoint']} ({fs.get('fstype')}, free {fs.get('free_gb')} GB)"
                  f" · LeRobot venv {'present' if st['venv_exists'] else 'not installed'}"
                  f" · SmolVLA model {'present' if st['model_local'] else 'not downloaded'}")
            for n in (fs.get("notes") or []):
                print(f"[WARN] V4.4 drive: {n}")
        else:
            print(f"[WARN] V4.4 drive {DRIVE_LABEL} not mounted — LeRobot / SmolVLA install, download and training are disabled")
        return 0
    if a.init:
        r = ensure_layout(st["layout"])
        print(json.dumps(r, ensure_ascii=False))
        if not r.get("ok"):
            return 1
        st = status(probe_venv=a.probe)
    print(format_status(st))
    return 0


if __name__ == "__main__":
    sys.exit(main())
