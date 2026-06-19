#!/usr/bin/env python3
"""日频更新入口：每个交易日收盘后运行
用法:
  /Users/zengpengxin/miniconda3/envs/rdagent/bin/python scripts/daily_update.py          # 全量
  /Users/zengpengxin/miniconda3/envs/rdagent/bin/python scripts/daily_update.py --symbols 510050  # 单只测试
  python scripts/daily_update.py --force            # 忽略交易日检查
"""
import argparse
import logging
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

# 使用 rdagent conda 环境的 Python（包含 qlib + akshare）
PYTHON = "/Users/zengpengxin/miniconda3/envs/rdagent/bin/python"

# ---------------------------------------------------------------------------
# 配置（集中管理路径 & 全局常量）
# ---------------------------------------------------------------------------

@dataclass
class Config:
    qlib_root: Path = Path("/Users/zengpengxin/workspace/CodeBase/qlib")
    qlib_base: Path = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415")
    symbols: str | None = None          # 逗号分隔的 ETF 代码；None 表示全量
    today: date = field(default_factory=date.today)
    max_index_retries: int = 0          # 0 = 无限重试

    @property
    def source_dir(self) -> Path:
        return self.qlib_base / "source"

    @property
    def etf_index_dir(self) -> Path:
        return self.source_dir / "etf_index"

    @property
    def holidays_file(self) -> Path:
        return self.qlib_root / "scripts" / "holidays_cn.txt"

    @property
    def workflow_config(self) -> Path:
        return self.qlib_root / "examples" / "benchmarks" / "HistRelaPB" / "workflow_config.yaml"

    @property
    def qlib_data_dir(self) -> Path:
        return self.qlib_base / "qlib_etf_index_Extend_wBond"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def is_trading_day(cfg: Config) -> bool:
    today = cfg.today
    if today.weekday() >= 5:
        return False
    if cfg.holidays_file.exists():
        if today.isoformat() in cfg.holidays_file.read_text().splitlines():
            return False
    return True


def run(cmd: list[str], cfg: Config) -> None:
    subprocess.run(cmd, check=True, cwd=cfg.qlib_root)


def fmt_elapsed(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds < 60 else f"{seconds / 60:.1f}min"


def last_non_null_date(path: Path, date_col: str = "date") -> pd.Timestamp | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    num_cols = [c for c in df.columns if c != date_col]
    valid = df.loc[df[num_cols].notna().any(axis=1), date_col].dropna()
    return valid.max() if not valid.empty else None


def index_symbols_for_etfs(etf_codes: list[str], cfg: Config) -> list[str]:
    """从 funds_list.csv 解析 ETF 对应的 LG 指数代码（如 '000016.SH'）"""
    df = pd.read_csv(cfg.source_dir / "funds_list.csv", comment="#", dtype=str)
    df = df.dropna(subset=["fund_code", "track_target_file"])
    df = df[df["track_target_file"].str.strip().ne("") & df["track_target_file"].str.strip().ne("N/A")]
    df["fund_code"] = df["fund_code"].str.strip()
    targets = df[df["fund_code"].isin(etf_codes)]["track_target_file"].unique()
    return [f"{f.replace('.csv', '')[2:]}.{f.replace('.csv', '')[:2]}" for f in targets]


# ---------------------------------------------------------------------------
# Command 模式：每个步骤封装为独立类
# ---------------------------------------------------------------------------

class UpdateStep(ABC):
    @property
    @abstractmethod
    def label(self) -> str: ...

    @abstractmethod
    def execute(self, cfg: Config) -> None: ...

    def run(self, cfg: Config, step_no: str) -> float:
        """执行并返回耗时（秒）"""
        logging.info(f"[{step_no}] {self.label}...")
        t0 = time.time()
        self.execute(cfg)
        cost = time.time() - t0
        logging.info(f"[{step_no}] 完成，耗时 {fmt_elapsed(cost)}")
        return cost


class ETFKlineStep(UpdateStep):
    label = "更新 ETF K线"

    def execute(self, cfg: Config) -> None:
        etf_args = (["--symbols", cfg.symbols] if cfg.symbols
                    else ["--funds_list", str(cfg.source_dir / "funds_list.csv")])
        costs: dict[str, float] = {}
        for fq, subdir, tag in [("hfq", "fund_kline_hfq", "hfq"), ("", "fund_kline_raw", "raw")]:
            t = time.time()
            run([PYTHON, "scripts/data_collector/tencent_etf/collector.py", "download_etf",
                 *etf_args,
                 "--source_dir", str(cfg.etf_index_dir / subdir),
                 "--fq_type", fq, "--delay", "0.5"], cfg)
            costs[tag] = time.time() - t
        logging.info(f"  细分: hfq={fmt_elapsed(costs['hfq'])}  raw={fmt_elapsed(costs['raw'])}")


class IndexValuationStep(UpdateStep):
    label = "更新指数估值"

    def execute(self, cfg: Config) -> None:
        index_dir = cfg.etf_index_dir / "index_data"
        dates = [d for f in index_dir.glob("*.csv") if (d := last_non_null_date(f)) is not None]
        ak_start = min(dates).strftime("%Y-%m-%d") if dates else None

        extra: list[str] = []
        if cfg.symbols:
            etf_list = [s.strip() for s in cfg.symbols.split(",")]
            idx_codes = index_symbols_for_etfs(etf_list, cfg)
            if not idx_codes:
                logging.warning(f"  {etf_list} 无对应指数，跳过")
                return
            extra = ["--symbols", ",".join(idx_codes)]
            logging.info(f"  indexes={idx_codes}, start={ak_start}")
        else:
            logging.info(f"  全部指数, start={ak_start}")

        cmd = [PYTHON, "scripts/data_collector/akshare/collector_index.py",
               "--save_dir", str(index_dir),
               "--delay", "3"]
        if ak_start:
            cmd += ["--start", ak_start]

        attempt = 0
        retry_interval = 60  # 秒
        while True:
            attempt += 1
            try:
                run(cmd + extra, cfg)
                return  # 成功退出
            except subprocess.CalledProcessError:
                if cfg.max_index_retries > 0 and attempt >= cfg.max_index_retries:
                    logging.warning(f"  指数估值已达最大重试次数 {cfg.max_index_retries}，跳过")
                    return
                next_time = time.strftime("%H:%M:%S", time.localtime(time.time() + retry_interval))
                logging.warning(
                    f"  指数估值下载失败（第 {attempt} 次），数据源可能宕机，"
                    f"{retry_interval}s 后重试（预计 {next_time}）..."
                )
                time.sleep(retry_interval)


class BondRateStep(UpdateStep):
    label = "更新国债收益率"

    def execute(self, cfg: Config) -> None:
        ld = last_non_null_date(cfg.source_dir / "cn_bond_rate" / "cn_bond_yield.csv")
        start = ld.strftime("%Y-%m-%d") if ld else "2000-01-01"
        logging.info(f"  start={start}")
        run([PYTHON, "scripts/data_collector/eastmoney_bond_rate/collector.py", "download_bond_rate",
             "--source_dir", str(cfg.source_dir / "cn_bond_rate"),
             "--start_date", start, "--delay", "0.5"], cfg)


class MergeConvertStep(UpdateStep):
    label = "合并清洗 + 转 qlib bin"

    def execute(self, cfg: Config) -> None:
        run([PYTHON, "scripts/data_processors/merge_etf_val/merge_clean_data.py"], cfg)
        run([PYTHON, "scripts/data_processors/merge_etf_val/dump_etf_index.py", "convert",
             "--data_path", str(cfg.etf_index_dir / "merged"),
             "--qlib_dir", str(cfg.qlib_data_dir)], cfg)


class BacktestStep(UpdateStep):
    label = "HistRelaPB 回测"

    def execute(self, cfg: Config) -> None:
        today_str = cfg.today.isoformat()
        logging.info(f"  更新 workflow_config.yaml 日期 → {today_str}")
        content = cfg.workflow_config.read_text()
        content = re.sub(r'(data_end:\s*&data_end\s*")[^"]*(")', rf'\g<1>{today_str}\2', content)
        content = re.sub(r'(backtest_end:\s*&backtest_end\s*")[^"]*(")', rf'\g<1>{today_str}\2', content)
        cfg.workflow_config.write_text(content)
        run(["/Users/zengpengxin/miniconda3/envs/rdagent/bin/qrun", str(cfg.workflow_config)], cfg)


def _latest_signals_csv(cfg: Config) -> Path | None:
    """返回最新一次回测产出的 _all_signals.csv 路径"""
    mlruns = cfg.qlib_root / "mlruns"
    candidates = []
    for exp_dir in mlruns.iterdir():
        if not exp_dir.is_dir():
            continue
        for run_dir in exp_dir.iterdir():
            sig = run_dir / "artifacts" / "signal_detail" / "_all_signals.csv"
            if sig.exists():
                candidates.append((run_dir.stat().st_mtime, sig))
    return max(candidates)[1] if candidates else None


def latest_experiment_result_dir(cfg: Config) -> Path | None:
    signals_csv = _latest_signals_csv(cfg)
    return signals_csv.parent.parent if signals_csv else None


def detect_signal(cfg: Config) -> dict:
    """对比最近两个交易日的持仓，识别调仓信号（以持仓数据变化为准）"""
    result_dir = latest_experiment_result_dir(cfg)
    if result_dir is None:
        return {"action": "unknown", "changes": [], "reason": "未找到实验结果目录"}

    pos_path = result_dir / "analysis_csvs" / "positions_daily.csv"
    if not pos_path.exists():
        return {"action": "unknown", "changes": [], "reason": "未找到持仓文件", "result_dir": str(result_dir)}

    pos_df = pd.read_csv(pos_path, index_col=0, parse_dates=True)
    # 移除统计行
    pos_df = pos_df[pos_df.index != "total_holding_ratio"]
    
    if len(pos_df) < 2:
        return {"action": "unknown", "changes": [], "reason": "持仓数据不足两日"}

    # 获取最近两个交易日（排除空行）
    valid_rows = pos_df.dropna(how="all")
    if len(valid_rows) < 2:
        return {"action": "hold", "changes": [], "reason": "有效持仓数据不足两日"}

    prev_date = valid_rows.index[-2]
    curr_date = valid_rows.index[-1]
    prev_pos = valid_rows.iloc[-2]
    curr_pos = valid_rows.iloc[-1]

    # ETF 列（排除 account_value 和 cash）
    etf_cols = [c for c in prev_pos.index if c not in ["account_value", "cash"]]
    
    changes = []
    for col in etf_cols:
        v1 = prev_pos.get(col, 0)
        v2 = curr_pos.get(col, 0)
        # 处理 NaN
        v1 = v1 if pd.notna(v1) else 0
        v2 = v2 if pd.notna(v2) else 0
        if v1 != v2:
            changes.append({"symbol": col, "prev_pos": v1, "curr_pos": v2})

    action = "rebalance" if changes else "hold"
    return {"action": action, "changes": changes,
            "prev_date": str(prev_date)[:10], "curr_date": str(curr_date)[:10],
            "result_dir": str(result_dir)}


def notify_macos(title: str, message: str) -> None:
    """通过 osascript 发送 macOS 系统通知"""
    script = f'display notification "{message}" with title "{title}" sound name "Glass"'
    subprocess.run(["osascript", "-e", script], check=False)


class SignalNotifyStep(UpdateStep):
    label = "信号检测 + 通知"

    def execute(self, cfg: Config) -> None:
        result = detect_signal(cfg)
        action = result.get("action", "unknown")
        changes = result.get("changes", [])
        result_dir = result.get("result_dir")

        if result_dir:
            logging.info(f"  [实验结果] {result_dir}")

        if action == "rebalance":
            detail_lines = []
            for c in changes:
                sym = c["symbol"].replace("_CLEAN", "")
                direction = "↑买入" if c["curr_pos"] > c["prev_pos"] else "↓卖出"
                detail_lines.append(f"{sym} {direction}")
            detail = "、".join(detail_lines[:5])
            if len(changes) > 5:
                detail += f" 等{len(changes)}只"
            msg = f"{result.get('curr_date', '')} 调仓: {detail}"
            logging.info(f"  [信号] 调仓  {msg}")
            notify_macos("HistRelaPB 调仓信号", msg)
        elif action == "hold":
            logging.info(f"  [信号] 无操作，持仓不变（{result.get('reason', '')}）")
        else:
            logging.info(f"  [信号] {result.get('reason', action)}")


# ---------------------------------------------------------------------------
# Pipeline：编排所有步骤
# ---------------------------------------------------------------------------

class DailyUpdatePipeline:
    STEPS: list[UpdateStep] = [
        ETFKlineStep(),
        IndexValuationStep(),
        BondRateStep(),
        MergeConvertStep(),
        BacktestStep(),
        SignalNotifyStep(),
    ]

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.total = len(self.STEPS)

    def run(self) -> None:
        t_total = time.time()
        costs: list[float] = []
        for i, step in enumerate(self.STEPS, start=1):
            cost = step.run(self.cfg, f"{i}/{self.total}")
            costs.append(cost)

        summary = "  |  ".join(
            f"{step.label}: {fmt_elapsed(c)}"
            for step, c in zip(self.STEPS, costs)
        )
        logging.info(f"--- 耗时统计 ---  {summary}")
        logging.info(f"=== 日频更新完成: {self.cfg.today}，总耗时 {fmt_elapsed(time.time() - t_total)} ===")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def _setup_logging(cfg: Config) -> None:
    log_dir = cfg.qlib_root / "logs" / "daily_update"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"{cfg.today}.log"),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="日频数据更新 + 回测")
    parser.add_argument("--symbols", type=str, default=None, help="逗号分隔的 ETF 代码，如 510050；不指定则全量")
    parser.add_argument("--force", action="store_true", help="忽略交易日判断，强制运行")
    parser.add_argument("--max_index_retries", type=int, default=0, help="指数估值最大重试次数，0=无限（默认）")
    args = parser.parse_args()

    cfg = Config(symbols=args.symbols, max_index_retries=args.max_index_retries)
    _setup_logging(cfg)

    if not args.force and not is_trading_day(cfg):
        logging.info(f"{cfg.today} 非交易日，跳过（--force 可强制运行）")
        return

    scope = f"[symbols={cfg.symbols}]" if cfg.symbols else "[全量]"
    logging.info(f"=== 日频更新开始: {cfg.today} {scope} ===")

    DailyUpdatePipeline(cfg).run()


if __name__ == "__main__":
    main()
