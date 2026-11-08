#!/usr/bin/env python3
"""NERO V3.83 GPU self-repair and preflight check.

Checks torch.cuda in a clean child process. When torch.cuda is unavailable but nvidia-smi is healthy
(typical after laptop suspend/resume due to broken nvidia_uvm state), attempts recovery:
  1. sudo -n rmmod nvidia_uvm && modprobe nvidia_uvm (passwordless sudo if configured)
  2. interactive sudo (if in a terminal)
  3. pkexec (from GUI button)
Provides clear warnings and instructions if recovery is not possible.
"""

import os
import shutil
import subprocess
import sys

VERSION = "V4.2"


def clean_child_env(env=None):
    """V3.46 QT ENV GUARD: clean Qt/OpenCV environment for child processes."""
    e = dict(env if env is not None else os.environ)
    for k in list(e.keys()):
        if k.startswith("QT_") or k in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
            if "cv2" in e[k] or "opencv" in e[k].lower():
                e.pop(k, None)
    return e


def check_torch_cuda(env=None, python_exe=None):
    """Run a child process to test torch.cuda.is_available() without polluting the parent process."""
    override = os.environ.get("NERO_TEST_TORCH_CUDA")
    if override is not None:
        if override.lower() in ("1", "true", "ok"):
            return True, "test_gpu"
        return False, override

    py = python_exe or sys.executable or "/usr/bin/python3"
    cmd = [py, "-c", "import torch; print('OK:' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'FAIL:not_available')"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10.0, env=clean_child_env(env))
        out = (r.stdout or "").strip()
        if out.startswith("OK:"):
            return True, out[3:].strip()
        err = (r.stderr or out or "torch.cuda.is_available() is False").strip()
        return False, err.splitlines()[-1] if err else "not_available"
    except Exception as exc:
        return False, str(exc)


def check_nvidia_smi():
    """Check if nvidia-smi runs and detects the GPU."""
    override = os.environ.get("NERO_TEST_NVIDIA_SMI")
    if override is not None:
        if override.lower() in ("1", "true", "ok"):
            return True, "test_nvidia_driver"
        return False, override

    smi = shutil.which("nvidia-smi")
    if not smi:
        return False, "找不到 nvidia-smi"
    try:
        r = subprocess.run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5.0)
        out = (r.stdout or "").strip()
        if r.returncode == 0 and out:
            return True, out.splitlines()[0]
        return False, r.stderr.strip() or f"rc={r.returncode}"
    except Exception as exc:
        return False, str(exc)


def _find_cmd(name):
    for p in (f"/usr/sbin/{name}", f"/sbin/{name}"):
        if os.path.exists(p):
            return p
    return name


def repair_uvm(interactive=True, run_cmd_fn=None):
    """Attempt to reload the nvidia_uvm kernel module."""
    runner = run_cmd_fn or subprocess.run
    rmmod = _find_cmd("rmmod")
    modprobe = _find_cmd("modprobe")

    # 1. Try passwordless sudo
    cmd_passless = ["sudo", "-n", rmmod, "nvidia_uvm"]
    try:
        r1 = runner(cmd_passless, capture_output=True, text=True, timeout=5.0)
        if r1.returncode == 0:
            r2 = runner(["sudo", "-n", modprobe, "nvidia_uvm"], capture_output=True, text=True, timeout=5.0)
            if r2.returncode == 0:
                return True, "免密 sudo 重载 nvidia_uvm 成功"
    except Exception:
        pass

    # 2. Try interactive sudo if in a terminal
    if interactive and sys.stdin.isatty():
        print("[INFO] [V3.83 GPU] 正在尝试重载 nvidia_uvm（请输入 sudo 密码）...")
        try:
            r1 = runner(["sudo", rmmod, "nvidia_uvm"])
            if r1.returncode == 0:
                r2 = runner(["sudo", modprobe, "nvidia_uvm"])
                if r2.returncode == 0:
                    return True, "交互式 sudo 重载 nvidia_uvm 成功"
        except Exception as exc:
            return False, f"sudo 执行失败：{exc}"

    return False, "无法免密执行 sudo 且无法进行终端交互"


def repair_uvm_pkexec(log=print, run_cmd_fn=None):
    """Attempt to reload nvidia_uvm using pkexec (suitable for GUI actions)."""
    runner = run_cmd_fn or subprocess.run
    rmmod = _find_cmd("rmmod")
    modprobe = _find_cmd("modprobe")
    shell_cmd = f"({rmmod} nvidia_uvm || true) && {modprobe} nvidia_uvm"

    override = os.environ.get("NERO_TEST_PKEXEC_SUCCESS")
    if override == "1":
        if log:
            log("[OK] [V3.83 GPU] pkexec 重载 nvidia_uvm 成功")
        return True, "pkexec 模拟成功"
    elif override == "0":
        if log:
            log("[FAIL] [V3.83 GPU] pkexec 执行失败")
        return False, "pkexec 模拟失败"

    pkexec = shutil.which("pkexec")
    if not pkexec:
        msg = "系统中未找到 pkexec"
        if log:
            log(f"[FAIL] [V3.83 GPU] {msg}")
        return False, msg

    try:
        r = runner([pkexec, "sh", "-c", shell_cmd], capture_output=True, text=True, timeout=30.0)
        if r.returncode == 0:
            msg = "pkexec 重载 nvidia_uvm 成功。由于当前 Python 进程内的 CUDA 初始化状态不可逆，请重启控制台使 GPU 生效。"
            if log:
                log(f"[OK] [V3.83 GPU] {msg}")
            return True, msg
        err = (r.stderr or r.stdout or f"rc={r.returncode}").strip()
        msg = f"pkexec 失败：{err}"
        if log:
            log(f"[FAIL] [V3.83 GPU] {msg}")
        return False, msg
    except Exception as exc:
        msg = f"pkexec 调用异常：{exc}"
        if log:
            log(f"[FAIL] [V3.83 GPU] {msg}")
        return False, msg


def preflight_check(log=print, interactive=True, run_cmd_fn=None):
    """Run before console launch. If CUDA is broken, attempt recovery."""
    ok, dev_or_reason = check_torch_cuda()
    if ok:
        log(f"[OK] [V3.83 GPU] CUDA 可用：cuda:0（{dev_or_reason}）")
        return True

    smi_ok, smi_info = check_nvidia_smi()
    if not smi_ok:
        log(f"[WARN] [V3.83 GPU] 未检测到正常 NVIDIA 驱动或 GPU 设备（{smi_info}），将使用 CPU。")
        return False

    log(f"[WARN] [V3.83 GPU] torch.cuda 不可用（{dev_or_reason}），但 nvidia-smi 正常（{smi_info}）。")
    log("[INFO] [V3.83 GPU] 正在尝试自动修复 nvidia_uvm 模块...")

    fixed, fix_msg = repair_uvm(interactive=interactive, run_cmd_fn=run_cmd_fn)
    if fixed:
        ok2, dev2 = check_torch_cuda()
        if ok2:
            log(f"[OK] [V3.83 GPU] 修复成功！CUDA 已恢复可用（{dev2}）。")
            return True
        log(f"[WARN] [V3.83 GPU] nvidia_uvm 已重载但 torch.cuda 仍不可用：{dev2}")

    log("[WARN] [V3.83 GPU] GPU 自修复未成功，控制台将继续以 CPU 模式启动（速度较慢）。")
    log("[WARN] [V3.83 GPU] 修复建议：")
    log("       1. 重启电脑；")
    log("       2. 或配置免密规则并手动执行：sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm；")
    log("       3. 长期预防：sudo systemctl enable nvidia-suspend nvidia-resume nvidia-hibernate")
    return False


def main():
    if "--preflight" in sys.argv:
        preflight_check()
    elif "--repair-pkexec" in sys.argv:
        repair_uvm_pkexec()
    else:
        ok, msg = check_torch_cuda()
        print(f"CUDA: {ok} ({msg})")


if __name__ == "__main__":
    main()
