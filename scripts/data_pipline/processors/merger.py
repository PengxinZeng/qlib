"""
AllWeatherMerger — 合并多个 source_dirs 中的 symbol CSV，
输出 {symbol}_clean.csv 到 output_dir。

处理逻辑：
1. 遍历所有 source_dirs，读取每个 symbol 的 CSV（按 date 排序）
2. 同一 symbol 出现在多个 source_dirs 时，按 date 做外连接合并（union dates）
3. 运行前清理 output_dir，再写出每个 symbol 的 clean CSV

注：尖刺清洗已拆分至 SpikeCleaner（见 spike_cleaner.py），请在 pipeline.yaml 中
将 merge 步骤的输出作为 SpikeCleaner 的 input_dir 接续使用。
"""

import shutil
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data_pipline.core.registry import register
from data_pipline.processors.base import BaseProcessor

logger = logging.getLogger(__name__)


@register("AllWeatherMerger")
class AllWeatherMerger(BaseProcessor):

    def __init__(self, cfg: dict, output_base: Path):
        super().__init__(cfg, output_base)
        self.source_dirs: list[Path] = [
            (output_base / d).resolve() for d in cfg["source_dirs"]
        ]
        self.output_dir: Path = (output_base / cfg["output_dir"]).resolve()

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _collect_symbol_files(self) -> dict[str, list[Path]]:
        """返回 {symbol: [csv_path, ...]} — 同一 symbol 可来自多个 source_dir。"""
        symbol_files: dict[str, list[Path]] = {}
        for src_dir in self.source_dirs:
            if not src_dir.exists():
                logger.warning("source_dir does not exist, skipping: %s", src_dir)
                continue
            for csv_path in sorted(src_dir.glob("*.csv")):
                symbol = csv_path.stem
                symbol_files.setdefault(symbol, []).append(csv_path)
        return symbol_files

    def _load_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def _merge_symbol(self, paths: list[Path]) -> pd.DataFrame:
        """将同一 symbol 多个来源按 date 外连接合并。"""
        if len(paths) == 1:
            df = self._load_csv(paths[0])
            return df.sort_values("date").reset_index(drop=True)

        merged: pd.DataFrame = pd.DataFrame()
        for p in paths:
            df = self._load_csv(p)
            if merged.empty:
                merged = df
            else:
                merged = merged.merge(df, on="date", how="outer", suffixes=("", f"_{p.parent.name}"))
        return merged.sort_values("date").reset_index(drop=True)

    # ──────────────────────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────────────────────

    def __call__(self) -> None:
        logger.info("AllWeatherMerger start")
        logger.info("  source_dirs : %s", [str(d) for d in self.source_dirs])
        logger.info("  output_dir  : %s", self.output_dir)

        # 清理输出目录
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            logger.info("Cleared output_dir: %s", self.output_dir)
        self.output_dir.mkdir(parents=True)

        symbol_files = self._collect_symbol_files()
        if not symbol_files:
            logger.warning("No CSV files found in source_dirs — nothing to merge.")
            return

        logger.info("Found %d unique symbols", len(symbol_files))

        success, skip = 0, 0

        for symbol, paths in tqdm(symbol_files.items(), desc="Merging"):
            try:
                df = self._merge_symbol(paths)
            except Exception as exc:
                logger.error("Failed to load/merge symbol %s: %s", symbol, exc)
                skip += 1
                continue

            if df.empty:
                skip += 1
                continue

            out_path = self.output_dir / f"{symbol}_clean.csv"
            df.to_csv(out_path, index=False)
            success += 1

        logger.info(
            "AllWeatherMerger done — success: %d, skip: %d, output: %s",
            success, skip, self.output_dir,
        )
