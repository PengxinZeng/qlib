"""
SpikeCleaner — 对 input_dir 中每个 symbol 的 CSV 做尖刺清洗，输出到 output_dir。

处理逻辑：
1. 遍历 input_dir 下所有 CSV（按 date 排序）
2. 尖刺清洗：close 单日涨跌幅绝对值 > spike_threshold，且次日反向回复 → open/high/low/close/volume 置 NaN
3. 运行前清理 output_dir，再写出每个 symbol 的 clean CSV
"""

import shutil
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data_pipline.core.registry import register
from data_pipline.processors.base import BaseProcessor

logger = logging.getLogger(__name__)

_PRICE_COLS = ["open", "high", "low", "close", "volume"]


@register("SpikeCleaner")
class SpikeCleaner(BaseProcessor):

    def __init__(self, cfg: dict, output_base: Path):
        super().__init__(cfg, output_base)
        self.input_dir: Path = (output_base / cfg["input_dir"]).resolve()
        self.output_dir: Path = (output_base / cfg["output_dir"]).resolve()
        self.spike_threshold: float = float(cfg.get("spike_threshold", 0.40))

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _load_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    def _clean_spikes(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """清除 close 列中的尖刺异常，返回 (cleaned_df, anomalous_dates)。"""
        if "close" not in df.columns or df["close"].isna().all():
            return df, []

        df = df.copy()
        close = df["close"]
        pct = close.pct_change(fill_method=None)
        pct_next = pct.shift(-1)

        spike_mask = (
            (pct.abs() > self.spike_threshold)
            & (pct_next.abs() > self.spike_threshold)
            & (pct * pct_next < 0)   # 方向相反（回复）
        )

        anomalous_dates = df.loc[spike_mask, "date"].tolist()
        if anomalous_dates:
            price_cols_present = [c for c in _PRICE_COLS if c in df.columns]
            df.loc[spike_mask, price_cols_present] = float("nan")

        return df, anomalous_dates

    # ──────────────────────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────────────────────

    def __call__(self) -> None:
        logger.info("SpikeCleaner start")
        logger.info("  input_dir      : %s", self.input_dir)
        logger.info("  output_dir     : %s", self.output_dir)
        logger.info("  spike_threshold: %.2f", self.spike_threshold)

        if not self.input_dir.exists():
            logger.warning("input_dir does not exist, nothing to clean: %s", self.input_dir)
            return

        # 清理输出目录
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            logger.info("Cleared output_dir: %s", self.output_dir)
        self.output_dir.mkdir(parents=True)

        csv_paths = sorted(self.input_dir.glob("*.csv"))
        if not csv_paths:
            logger.warning("No CSV files found in input_dir — nothing to clean.")
            return

        logger.info("Found %d CSV files", len(csv_paths))

        success, skip, spike_report = 0, 0, {}

        for csv_path in tqdm(csv_paths, desc="Cleaning spikes"):
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

            df, anomalous_dates = self._clean_spikes(df)
            if anomalous_dates:
                spike_report[symbol] = anomalous_dates

            out_path = self.output_dir / csv_path.name
            df.to_csv(out_path, index=False)
            success += 1

        # Spike report
        if spike_report:
            logger.info("Spike cleaning report (%d symbols affected):", len(spike_report))
            for sym, dates in spike_report.items():
                date_strs = ", ".join(
                    d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                    for d in dates
                )
                logger.info("  %s: %d spike(s) → %s", sym, len(dates), date_strs)
        else:
            logger.info("No price spikes detected.")

        logger.info(
            "SpikeCleaner done — success: %d, skip: %d, output: %s",
            success, skip, self.output_dir,
        )
