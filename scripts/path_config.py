"""
path_config.py — 跨平台路径集中配置（Mac / Windows 兼容）

设计原则：
1. 仓库根 (QLIB_ROOT) 由环境变量 QLIB_ROOT 覆盖，否则取本文件上两级目录
   （即 scripts/ 的父目录 = 仓库根），与运行机器无关。
2. 数据根 (DATA_BASE) 由环境变量 QLIB_DATA_BASE 覆盖，否则按操作系统给默认值：
   - Windows: D:/Pengxin/CodeBase/Quant/QuantDataBank
   - macOS:   /Users/zengpengxin/workspace/DataBase/Quant/QlibBase
3. Python / qrun 解释器：
   - QLIB_PYTHON 环境变量优先；否则用 sys.executable（即当前运行解释器）
   - QLIB_QRUN 环境变量优先；否则在 sys.executable 同级目录查找
     Scripts/qrun.exe (Windows) 或 bin/qrun (POSIX)
4. 子进程（collector / pipeline / qrun）通过继承环境变量 QLIB_DATA_BASE / QLIB_ROOT
   获得一致的数据根，脚本内再次 import path_config 时解析结果相同。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 仓库根
# ---------------------------------------------------------------------------
_QLIB_ROOT_ENV = os.environ.get("QLIB_ROOT")
QLIB_ROOT = Path(_QLIB_ROOT_ENV) if _QLIB_ROOT_ENV else Path(__file__).resolve().parents[1]
SCRIPTS_DIR = QLIB_ROOT / "scripts"

# ---------------------------------------------------------------------------
# 数据根
# ---------------------------------------------------------------------------
_DATA_BASE_ENV = os.environ.get("QLIB_DATA_BASE")
if _DATA_BASE_ENV:
    DATA_BASE = Path(_DATA_BASE_ENV)
elif sys.platform == "win32":
    DATA_BASE = Path("D:/Pengxin/CodeBase/Quant/QuantDataBank")
else:  # macOS / Linux
    DATA_BASE = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase")

# ---------------------------------------------------------------------------
# 派生路径（镜像 Mac QlibBase 结构）
# ---------------------------------------------------------------------------
QLIB_BASE = DATA_BASE / "qlib_data_260415"          # 历史数据根（ETF 链）
ALL_WEATHER_BASE = DATA_BASE / "all_weather_data"    # 全天候数据链根

SOURCE_DIR = QLIB_BASE / "source"                    # 数据源根
ETF_INDEX_DIR = SOURCE_DIR / "etf_index"             # ETF 相关数据
FUNDS_LIST = SOURCE_DIR / "funds_list.csv"           # ETF→指数映射
BOND_FILE = SOURCE_DIR / "cn_bond_rate" / "cn_bond_yield.csv"   # 国债收益率
MERGED_DIR = ETF_INDEX_DIR / "merged"                # 合并清洗输出
QLIB_DATA_DIR = QLIB_BASE / "qlib_etf_index_Extend_wBond"       # dump 输出（HistRelaPB 数据链）
ALL_WEATHER_QLIB_DIR = ALL_WEATHER_BASE / "qlib_all_weather"    # pipeline dump 输出（EMVal 数据链）

# ---------------------------------------------------------------------------
# 解释器
# ---------------------------------------------------------------------------
def _default_python() -> Path:
    env = os.environ.get("QLIB_PYTHON")
    if env:
        return Path(env)
    return Path(sys.executable)


def _default_qrun() -> Path:
    env = os.environ.get("QLIB_QRUN")
    if env:
        return Path(env)
    py = Path(sys.executable)
    if sys.platform == "win32":
        candidate = py.with_name("Scripts") / "qrun.exe"
    else:
        candidate = py.with_name("qrun")
    if candidate.exists():
        return candidate
    return py.with_name("qrun")  # 兜底：PATH 中的 qrun


PYTHON = _default_python()
QRUN = _default_qrun()

# ---------------------------------------------------------------------------
# 环境变量注入：子进程继承后，内部 import path_config 得到一致路径
# ---------------------------------------------------------------------------
def export_env() -> None:
    """把本模块解析出的关键路径写入 os.environ，供子进程继承。"""
    os.environ.setdefault("QLIB_ROOT", str(QLIB_ROOT))
    os.environ.setdefault("QLIB_DATA_BASE", str(DATA_BASE))


# ---------------------------------------------------------------------------
# YAML 路径 token 注入
# ---------------------------------------------------------------------------
TOKEN_DATA_BASE = "__DATA_BASE__"
TOKEN_REPO_ROOT = "__REPO_ROOT__"

PATH_TOKEN_REPLACEMENTS = {
    TOKEN_DATA_BASE: str(DATA_BASE).replace("\\", "/"),
    TOKEN_REPO_ROOT: str(QLIB_ROOT).replace("\\", "/"),
}


def inject_yaml_tokens(text: str) -> str:
    """把 yaml 文本中的 __DATA_BASE__ / __REPO_ROOT__ token 替换为绝对路径（正斜杠）。"""
    for token, value in PATH_TOKEN_REPLACEMENTS.items():
        text = text.replace(token, value)
    return text


if __name__ == "__main__":
    print(f"QLIB_ROOT        = {QLIB_ROOT}")
    print(f"SCRIPTS_DIR      = {SCRIPTS_DIR}")
    print(f"DATA_BASE        = {DATA_BASE}")
    print(f"QLIB_BASE        = {QLIB_BASE}")
    print(f"ALL_WEATHER_BASE = {ALL_WEATHER_BASE}")
    print(f"SOURCE_DIR       = {SOURCE_DIR}")
    print(f"FUNDS_LIST       = {FUNDS_LIST}")
    print(f"BOND_FILE        = {BOND_FILE}")
    print(f"MERGED_DIR       = {MERGED_DIR}")
    print(f"QLIB_DATA_DIR    = {QLIB_DATA_DIR}")
    print(f"PYTHON           = {PYTHON}")
    print(f"QRUN             = {QRUN}")
