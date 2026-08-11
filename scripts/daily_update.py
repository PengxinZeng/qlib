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
QRUN = "/Users/zengpengxin/miniconda3/envs/rdagent/bin/qrun"

# ---------------------------------------------------------------------------
# 配置（集中管理路径 & 全局常量）
# ---------------------------------------------------------------------------

@dataclass
class Config:
    qlib_root: Path = Path("/Users/zengpengxin/workspace/CodeBase/qlib")
    qlib_base: Path = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415")
    all_weather_base: Path = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/all_weather_data")
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

    # ---- all_weather 数据链（SustainedBest 依赖） ----
    @property
    def pipeline_yaml(self) -> Path:
        return self.qlib_root / "scripts" / "data_pipline" / "pipeline.yaml"

    @property
    def run_pipeline(self) -> Path:
        return self.qlib_root / "scripts" / "data_pipline" / "run_pipeline.py"

    @property
    def emval_womom_config(self) -> Path:
        return self.qlib_root / "examples" / "benchmarks" / "EMVal" / "workflow_config_all_weather_WoMom.yaml"

    @property
    def emval_config(self) -> Path:
        return self.qlib_root / "examples" / "benchmarks" / "EMVal" / "workflow_config_all_weather.yaml"

    @property
    def sustained_best_config(self) -> Path:
        return self.qlib_root / "examples" / "benchmarks" / "EMEnsemble" / "workflow_config_all_weather_SustainedBest.yaml"


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


def update_yaml_dates(path: Path, today_str: str) -> None:
    """将 workflow_config 的 data_end / backtest_end 更新为 today"""
    content = path.read_text()
    content = re.sub(r'(data_end:\s*&data_end\s*")[^"]*(")', rf'\g<1>{today_str}\2', content)
    content = re.sub(r'(backtest_end:\s*&backtest_end\s*")[^"]*(")', rf'\g<1>{today_str}\2', content)
    path.write_text(content)


def latest_run_dir_for_experiment(cfg: Config, experiment_name: str) -> Path | None:
    """按 experiment_name 返回最新含 positions_daily.csv 的 run 目录（mtime 最新）"""
    mlruns = cfg.qlib_root / "mlruns"
    best: Path | None = None
    for exp_dir in mlruns.iterdir():
        if not exp_dir.is_dir():
            continue
        meta = exp_dir / "meta.yaml"
        if not meta.exists():
            continue
        name_line = next((ln for ln in meta.read_text().splitlines() if ln.startswith("name:")), None)
        if name_line is None:
            continue
        exp_name = name_line.split(":", 1)[1].strip().strip("'\"")
        if exp_name != experiment_name:
            continue
        for run_dir in exp_dir.iterdir():
            if not run_dir.is_dir():
                continue
            pos = run_dir / "artifacts" / "analysis_csvs" / "positions_daily.csv"
            if pos.exists() and (best is None or run_dir.stat().st_mtime > best.stat().st_mtime):
                best = run_dir
    return best


def detect_signal(result_dir: Path | None) -> dict:
    """
    识别调仓信号（基于持仓状态 × score 信号的状态机）。

    T+1 业务语义（第 t 日收盘后发信号，第 t+1 日执行）：
      - 持有中（position > 0）且 score < 0  → 明日卖出
      - 不持有（position == 0）且 score > 0 → 明日买入
      - 其余情况（持有+score>0 / 空仓+score<0 / score=0）→ 不动作

    用持仓状态与模型信号判断，不依赖持仓市值绝对值，故股价波动不会误判为调仓。
    """
    if result_dir is None:
        return {"action": "unknown", "changes": [], "reason": "未找到实验结果目录"}

    pos_path = result_dir / "artifacts" / "analysis_csvs" / "positions_daily.csv"
    pred_path = result_dir / "artifacts" / "analysis_csvs" / "pred_df.csv"
    if not pos_path.exists():
        return {"action": "unknown", "changes": [], "reason": "未找到持仓文件", "result_dir": str(result_dir)}
    if not pred_path.exists():
        return {"action": "unknown", "changes": [], "reason": "未找到信号文件 pred_df.csv", "result_dir": str(result_dir)}

    # 1) 持仓状态：positions_daily.csv（最近一个有效交易日 t）
    pos_df = pd.read_csv(pos_path, index_col=0, parse_dates=True)
    pos_df = pos_df[pos_df.index != "total_holding_ratio"]
    valid_rows = pos_df.dropna(how="all")
    if len(valid_rows) < 1:
        return {"action": "unknown", "changes": [], "reason": "无有效持仓数据"}
    curr_date = valid_rows.index[-1]
    curr_pos = valid_rows.iloc[-1]  # 第 t 日各标的持仓市值

    # 持仓列可能带 _CLEAN 后缀（HistRelaPB），统一用去后缀的标的代码对齐
    sym_codes = [c for c in curr_pos.index if c not in ["account_value", "cash"]]

    # 2) 信号：pred_df.csv 的 score（第 t 日各标的模型信号）
    pred_df = pd.read_csv(pred_path)
    if "score" not in pred_df.columns:
        return {"action": "unknown", "changes": [], "reason": "pred_df.csv 缺少 score 列", "result_dir": str(result_dir)}
    pred_df[pred_df.columns[0]] = pd.to_datetime(pred_df[pred_df.columns[0]], errors="coerce")
    # 标的代码统一去 _CLEAN 后缀，便于与持仓列对齐
    pred_df["instrument"] = pred_df["instrument"].astype(str).str.replace("_CLEAN", "", regex=False)
    # pivot：行=日期、列=标的、值=score
    sig_pivot = pred_df.pivot_table(index=pred_df.columns[0], columns="instrument", values="score", aggfunc="last")
    # 对齐到持仓最新交易日 t
    if curr_date not in sig_pivot.index:
        return {"action": "unknown", "changes": [], "reason": f"信号文件缺少交易日 {curr_date:%Y-%m-%d}"}

    # 3) 状态机判调仓
    changes = []
    holdings = []
    for col in sym_codes:
        code = col.replace("_CLEAN", "")  # 持仓列去后缀，统一标的代码
        pos = curr_pos.get(col, 0)
        pos = pos if pd.notna(pos) else 0
        # 第 t 日该标的 score（缺失视为 0 = 不动作）
        score = sig_pivot.at[curr_date, code] if code in sig_pivot.columns else 0.0
        score = score if pd.notna(score) else 0.0

        if pos > 0 and score < 0:
            changes.append({"symbol": code, "action": "卖出", "pos": pos, "score": score})
        elif pos <= 0 and score > 0:
            changes.append({"symbol": code, "action": "买入", "pos": pos, "score": score})

        # 当前持仓列表（持仓市值 > 0）
        if pos > 0:
            holdings.append({"symbol": code, "pos": pos})

    # 持仓按市值降序
    holdings.sort(key=lambda h: h["pos"], reverse=True)

    action = "rebalance" if changes else "hold"
    return {"action": action, "changes": changes, "holdings": holdings,
            "curr_date": str(curr_date)[:10],
            "result_dir": str(result_dir)}


def notify_macos(title: str, message: str) -> None:
    """通过 osascript 发送 macOS 系统通知"""
    script = f'display notification "{message}" with title "{title}" sound name "Glass"'
    subprocess.run(["osascript", "-e", script], check=False)


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
    label = "更新指数估值（全量）"

    def execute(self, cfg: Config) -> None:
        index_dir = cfg.etf_index_dir / "index_data"

        extra: list[str] = []
        if cfg.symbols:
            etf_list = [s.strip() for s in cfg.symbols.split(",")]
            idx_codes = index_symbols_for_etfs(etf_list, cfg)
            if not idx_codes:
                logging.warning(f"  {etf_list} 无对应指数，跳过")
                return
            extra = ["--symbols", ",".join(idx_codes)]
            logging.info(f"  indexes={idx_codes}")
        else:
            logging.info("  全部指数（全量）")

        cmd = [PYTHON, "scripts/data_collector/akshare/collector_index.py",
               "--save_dir", str(index_dir),
               "--delay", "3"]

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
    label = "更新国债收益率（全量）"

    def execute(self, cfg: Config) -> None:
        logging.info("  全量下载（start=2000-01-01）")
        run([PYTHON, "scripts/data_collector/eastmoney_bond_rate/collector.py", "download_bond_rate",
             "--source_dir", str(cfg.source_dir / "cn_bond_rate"),
             "--start_date", "2000-01-01", "--delay", "0.5"], cfg)


class MergeConvertStep(UpdateStep):
    label = "合并清洗 + 转 qlib bin（HistRelaPB 数据链）"

    def execute(self, cfg: Config) -> None:
        run([PYTHON, "scripts/data_processors/merge_etf_val/merge_clean_data.py"], cfg)
        run([PYTHON, "scripts/data_processors/merge_etf_val/dump_etf_index.py", "convert",
             "--data_path", str(cfg.etf_index_dir / "merged"),
             "--qlib_dir", str(cfg.qlib_data_dir)], cfg)


class AllWeatherUpdateStep(UpdateStep):
    """all_weather 数据链增量更新（SustainedBest 依赖的数据源）"""
    label = "更新 all_weather 数据（增量）"

    def execute(self, cfg: Config) -> None:
        run([PYTHON, str(cfg.run_pipeline),
             "--config", str(cfg.pipeline_yaml),
             "--incremental"], cfg)


class EMValUpdateStep(UpdateStep):
    """更新两个 EMVal 模型信号，并将最新 run 同步到 SustainedBest 配置的 exp_path"""
    label = "更新 EMVal 信号 + 同步 exp_path"

    def execute(self, cfg: Config) -> None:
        today_str = cfg.today.isoformat()
        for config in [cfg.emval_womom_config, cfg.emval_config]:
            logging.info(f"  更新 {config.name} 日期 → {today_str}")
            update_yaml_dates(config, today_str)
            run([QRUN, str(config)], cfg)

        womom_run = latest_run_dir_for_experiment(cfg, "em_val_all_weather_womom")
        emval_run = latest_run_dir_for_experiment(cfg, "em_val_all_weather")
        if womom_run is None or emval_run is None:
            raise RuntimeError(
                f"EMVal 最新 run 未找到: womom={womom_run}, emval={emval_run}"
            )

        # 将 SustainedBest 配置中两个 exp_path 依次替换为最新 run（绝对路径）。
        # 注意：SustainedBest 配置中 exp_path 第一次出现对应 emval_womom，第二次对应 emval，
        # 不能用两次 count=1 的 re.sub（第二次会再次命中第一个已替换的 exp_path），
        # 改为用计数器按出现顺序逐个替换（re.sub 回调从左到右调用，顺序与配置一致）。
        sb = cfg.sustained_best_config
        content = sb.read_text()
        new_paths = [str(womom_run.resolve()), str(emval_run.resolve())]
        counter = {"i": 0}

        def _swap_exp_path(m: re.Match) -> str:
            idx = counter["i"]
            counter["i"] += 1
            # 防御：超出范围则保留原值（正常配置只有 2 个 exp_path）
            replacement = new_paths[idx] if idx < len(new_paths) else m.group(2)
            return m.group(1) + replacement + m.group(2)

        content = re.sub(r'(exp_path:\s*")[^"]*(")', _swap_exp_path, content)
        sb.write_text(content)
        logging.info(f"  SustainedBest exp_path 已更新: womom={womom_run.name}, emval={emval_run.name}")


class EMEnsembleBacktestStep(UpdateStep):
    """EMEnsemble (SustainedBest) 回测：生成最新持仓"""
    label = "EMEnsemble (SustainedBest) 回测"

    def execute(self, cfg: Config) -> None:
        today_str = cfg.today.isoformat()
        logging.info(f"  更新 workflow_config_all_weather_SustainedBest.yaml 日期 → {today_str}")
        update_yaml_dates(cfg.sustained_best_config, today_str)
        run([QRUN, str(cfg.sustained_best_config)], cfg)


class BacktestStep(UpdateStep):
    label = "HistRelaPB 回测"

    def execute(self, cfg: Config) -> None:
        today_str = cfg.today.isoformat()
        logging.info(f"  更新 workflow_config.yaml 日期 → {today_str}")
        update_yaml_dates(cfg.workflow_config, today_str)
        run([QRUN, str(cfg.workflow_config)], cfg)


class SignalNotifyStep(UpdateStep):
    label = "信号检测 + 通知"

    def execute(self, cfg: Config) -> None:
        strategies = [
            ("HistRelaPB", "hist_rela_pb_etf"),
            ("SustainedBest", "em_ensemble_sustainedbest_all_weather"),
        ]
        for title, exp_name in strategies:
            result_dir = latest_run_dir_for_experiment(cfg, exp_name)
            result = detect_signal(result_dir)
            self._report(title, result)

    def _report(self, title: str, result: dict) -> None:
        action = result.get("action", "unknown")
        changes = result.get("changes", [])
        holdings = result.get("holdings", [])
        result_dir = result.get("result_dir")

        # 当前持仓列表（按市值降序）
        hold_str = "、".join(h["symbol"].replace("_CLEAN", "") for h in holdings)
        hold_txt = f"当前持仓: {hold_str}" if hold_str else "当前持仓: 空仓"

        if result_dir:
            logging.info(f"  [{title}] 实验结果: {result_dir}")

        if action == "rebalance":
            detail_lines = []
            for c in changes:
                sym = c["symbol"].replace("_CLEAN", "")
                direction = "↑买入" if c.get("action") == "买入" else "↓卖出"
                detail_lines.append(f"{sym} {direction}")
            detail = "、".join(detail_lines[:5])
            if len(changes) > 5:
                detail += f" 等{len(changes)}只"
            msg = f"{result.get('curr_date', '')} 需调仓: {detail}"
            logging.info(f"  [{title}] 需调仓  {msg}；{hold_txt}")
            notify_macos(f"{title} 明日需调仓", f"{msg}（明日执行）；{hold_txt}")
        elif action == "hold":
            msg = f"无操作，持仓不变；{hold_txt}"
            reason = result.get("reason", "")
            logging.info(f"  [{title}] {msg}" + (f"（{reason}）" if reason else ""))
            notify_macos(f"{title} 持仓不变", msg)
        else:
            reason = result.get("reason", action)
            msg = f"{reason}；{hold_txt}"
            logging.info(f"  [{title}] {msg}")
            notify_macos(f"{title} 检测异常", msg)


# ---------------------------------------------------------------------------
# Pipeline：编排所有步骤
# ---------------------------------------------------------------------------

class DailyUpdatePipeline:
    STEPS: list[UpdateStep] = [
        ETFKlineStep(),
        IndexValuationStep(),
        BondRateStep(),
        MergeConvertStep(),
        AllWeatherUpdateStep(),      # all_weather 数据增量更新
        EMValUpdateStep(),           # EMVal 信号 + exp_path 同步
        EMEnsembleBacktestStep(),    # SustainedBest 回测
        BacktestStep(),              # HistRelaPB 回测
        SignalNotifyStep(),          # 双策略检测通知
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