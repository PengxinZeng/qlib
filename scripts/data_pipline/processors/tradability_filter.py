"""
TradabilityFilter — 过滤不可交易日（无成交量代理）。

处理逻辑：
1. 遍历 input_dir 下所有 {symbol}_clean.csv（按 date 排序）
2. 以 volume 作为可交易性代理：volume 为 NaN 或 <= min_value 的日期视为不可交易，
   将该日的 open/high/low/close/volume 全部置 NaN
3. 保护场外基金：若某标的 volume 整列无有效值（全 NaN），视为该字段无意义，整体跳过不过滤
4. 运行前清理 output_dir，再写出每个 symbol 的 CSV

注：置 NaN（而非删行），保留日期以维持下游日历对齐。
"""

import shutil
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data_pipline.core.registry import register
from data_pipline.processors.base import BaseProcessor

logger = logging.getLogger(__name__)

_NAN_COLS = ["open", "high", "low", "close", "volume"]


@register("TradabilityFilter")
class TradabilityFilter(BaseProcessor):

    def __init__(self, cfg: dict, output_base: Path):
        super().__init__(cfg, output_base)
        self.input_dir: Path = (output_base / cfg["input_dir"]).resolve()
        self.output_dir: Path = (output_base / cfg["output_dir"]).resolve()
        self.field: str = cfg.get("field", "volume")
        self.min_value: float = float(cfg.get("min_value", 0))
        self.skip_if_all_missing: bool = bool(cfg.get("skip_if_all_missing", True))

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _load_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def _filter(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list, bool]:
        """将不可交易日的价格/成交量列置 NaN。

        返回 (df, untradable_dates, skipped)。skipped=True 表示该标的因 field
        整列无有效值而被跳过（不过滤）。
        """
        if self.field not in df.columns:
            return df, [], True

        col = df[self.field]
        if col.dropna().empty:
            # 整列无有效值（如场外基金 volume 恒为 NaN）→ 保护性跳过
            if self.skip_if_all_missing:
                return df, [], True

        df = df.copy()
        untradable_mask = col.isna() | (col <= self.min_value)

        untradable_dates = df.loc[untradable_mask, "date"].tolist()
        if untradable_dates:
            cols_present = [c for c in _NAN_COLS if c in df.columns]
            df.loc[untradable_mask, cols_present] = float("nan")

        return df, untradable_dates, False

    # ──────────────────────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────────────────────

    def __call__(self) -> None:
        logger.info("TradabilityFilter start")
        logger.info("  input_dir          : %s", self.input_dir)
        logger.info("  output_dir         : %s", self.output_dir)
        logger.info("  field              : %s", self.field)
        logger.info("  min_value          : %s", self.min_value)
        logger.info("  skip_if_all_missing: %s", self.skip_if_all_missing)

        if not self.input_dir.exists():
            logger.warning("input_dir does not exist, nothing to filter: %s", self.input_dir)
            return

        # 清理输出目录
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            logger.info("Cleared output_dir: %s", self.output_dir)
        self.output_dir.mkdir(parents=True)

        csv_paths = sorted(self.input_dir.glob("*.csv"))
        if not csv_paths:
            logger.warning("No CSV files found in input_dir — nothing to filter.")
            return

        logger.info("Found %d CSV files", len(csv_paths))

        success, skip, skipped_field, filter_report = 0, 0, [], {}

        for csv_path in tqdm(csv_paths, desc="Filtering tradability"):
            symbol = csv_path.stem
            try:
                df = self._load_csv(csv_path)
            except Exception as exc:
                logger.error("Failed to load %s: %s", csv_path, exc)
                skip += 1
                continue

            if df.empty:
                skip += 1
                continue

            df, untradable_dates, skipped = self._filter(df)
            if skipped:
                skipped_field.append(symbol)
            elif untradable_dates:
                filter_report[symbol] = untradable_dates

            out_path = self.output_dir / csv_path.name
            df.to_csv(out_path, index=False)
            success += 1

        # Report
        if skipped_field:
            logger.info(
                "Skipped %d symbols (no valid '%s' — e.g. OTC funds): %s",
                len(skipped_field), self.field, skipped_field,
            )
        if filter_report:
            logger.info("Tradability filter report (%d symbols affected):", len(filter_report))
            for sym, dates in filter_report.items():
                logger.info("  %s: %d untradable day(s) → NaN", sym, len(dates))
        else:
            logger.info("No untradable days detected.")

        logger.info(
            "TradabilityFilter done — success: %d, skip: %d, output: %s",
            success, skip, self.output_dir,
        )
