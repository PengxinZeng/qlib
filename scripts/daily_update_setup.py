#!/usr/bin/env python3
"""
一次性安装脚本：注册 launchd 定时任务，每天 15:35 自动运行日频更新。

⚠️ macOS 专属（依赖 launchctl / LaunchAgents）。Windows 请改用任务计划程序
（schtasks）直接调度 `python scripts/daily_update.py`，路径由 path_config 解析。

用法:
  python scripts/daily_update_setup.py install    # 安装
  python scripts/daily_update_setup.py uninstall  # 卸载
  python scripts/daily_update_setup.py status     # 查看状态
"""
import subprocess
import sys
from pathlib import Path

LABEL = "com.qlib.daily-update"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"

# macOS 专属：固定本机路径（Windows 不使用本脚本）
QLIB_ROOT = Path("/Users/zengpengxin/workspace/CodeBase/qlib")
PYTHON = "/Users/zengpengxin/miniconda3/envs/rdagent/bin/python"
SCRIPT = str(QLIB_ROOT / "scripts" / "daily_update.py")
LOG_DIR = QLIB_ROOT / "logs" / "daily_update"

PLIST_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{label}</string>

    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>{script}</string>
    </array>

    <!-- 每天 15:35 触发；若系统当时处于睡眠，唤醒后立即补跑 -->
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key><integer>15</integer>
        <key>Minute</key><integer>35</integer>
    </dict>

    <!-- 标准输出/错误写入日志文件 -->
    <key>StandardOutPath</key>
    <string>{log_dir}/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/launchd_err.log</string>

    <!-- 工作目录 -->
    <key>WorkingDirectory</key>
    <string>{qlib_root}</string>

    <!-- 失败后不自动重启（更新失败不应无限循环） -->
    <key>KeepAlive</key>
    <false/>
</dict>
</plist>
"""


def install() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    plist_content = PLIST_TEMPLATE.format(
        label=LABEL,
        python=PYTHON,
        script=SCRIPT,
        log_dir=str(LOG_DIR),
        qlib_root=str(QLIB_ROOT),
    )
    PLIST_PATH.write_text(plist_content)
    print(f"已写入: {PLIST_PATH}")

    # 若已加载则先卸载
    result = subprocess.run(
        ["launchctl", "list", LABEL],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)], check=False)

    subprocess.run(["launchctl", "load", str(PLIST_PATH)], check=True)
    print(f"已加载 launchd 任务: {LABEL}")
    print("每天 15:35 自动触发（睡眠/关盖后唤醒立即补跑）")
    print(f"日志: {LOG_DIR}/launchd.log")


def uninstall() -> None:
    if PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(PLIST_PATH)], check=False)
        PLIST_PATH.unlink()
        print(f"已卸载并删除: {PLIST_PATH}")
    else:
        print("未找到 plist，无需卸载")


def status() -> None:
    result = subprocess.run(
        ["launchctl", "list", LABEL],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"✓ 已注册: {LABEL}")
        print(result.stdout)
    else:
        print(f"✗ 未注册: {LABEL}")

    if PLIST_PATH.exists():
        print(f"plist: {PLIST_PATH}")
    print(f"日志目录: {LOG_DIR}")


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "install":
        install()
    elif cmd == "uninstall":
        uninstall()
    elif cmd == "status":
        status()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
