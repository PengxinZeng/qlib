"""
EtfVisualizer — 读取 input_dir 下的 {symbol}_clean.csv，绘制 ETF 归一化走势图。

复用 data_visulizers/data_distribution/plot_lib.py 中的纯绘图函数：
- plot_aligned_etf   → all_etf_aligned.png   (所有 ETF 叠加归一化走势)
- plot_subplots_etf  → all_etf_subplots.png  (每只 ETF 一个子图)

处理逻辑：
1. 读 input_dir 下所有 *_clean.csv，symbol = stem 去掉 "_clean"
2. 组装成 plot_lib 约定的 df：MultiIndex(instrument, datetime) + 列 close
3. 可选读取 split_file(split_by_cumsum.csv) 得到数据集划分，作为图中背景/分割线
4. 调用两个绘图函数输出到 output_dir
"""

import logging
from pathlib import Path

import pandas as pd

from data_pipline.core.registry import register
from data_pipline.processors.base import BaseProcessor

logger = logging.getLogger(__name__)


@register("EtfVisualizer")
class EtfVisualizer(BaseProcessor):

    def __init__(self, cfg: dict, output_base: Path):
        super().__init__(cfg, output_base)
        self.input_dir: Path = (output_base / cfg["input_dir"]).resolve()
        self.output_dir: Path = (output_base / cfg["output_dir"]).resolve()
        # 可选：数据集划分文件（split_by_cumsum.csv），提供时按其划分绘制背景/分割线
        split_file = cfg.get("split_file")
        self.split_file: Path | None = (
            (output_base / split_file).resolve() if split_file else None
        )

    def _load_split_segments(self) -> list | None:
        """从 split_by_cumsum.csv 读取数据集划分段：[{name, start, end}]。"""
        if not self.split_file:
            return None
        if not self.split_file.exists():
            logger.warning("split_file not found, using default split: %s", self.split_file)
            return None
        sdf = pd.read_csv(self.split_file)
        required = {"dataset", "start_date", "end_date"}
        if not required.issubset(sdf.columns):
            logger.warning(
                "split_file missing columns %s, using default split", required - set(sdf.columns)
            )
            return None
        return [
            {"name": str(r["dataset"]), "start": str(r["start_date"]), "end": str(r["end_date"])}
            for _, r in sdf.iterrows()
        ]

    def _build_df(self) -> tuple[pd.DataFrame, dict, dict]:
        """组装 plot_lib 约定的 MultiIndex(instrument, datetime) + close 的 df，
        并从 CSV 的 name / track_index 列提取 {symbol: name}、{symbol: track_index} 映射。"""
        frames = []
        fund_names: dict[str, str] = {}
        track_index: dict[str, str] = {}
        for csv_path in sorted(self.input_dir.glob("*_clean.csv")):
            symbol = csv_path.stem.removesuffix("_clean")
            df = pd.read_csv(csv_path)
            if "close" not in df.columns or df.empty:
                continue
            if "name" in df.columns:
                names = df["name"].dropna()
                if not names.empty:
                    fund_names[symbol] = str(names.iloc[-1])
            if "track_index" in df.columns:
                idxs = df["track_index"].dropna()
                idxs = idxs[idxs.astype(str).str.strip() != ""]
                if not idxs.empty:
                    track_index[symbol] = str(idxs.iloc[-1])
            df = df.dropna(subset=["close"])
            if df.empty:
                continue
            df["datetime"] = pd.to_datetime(df["date"])
            df["instrument"] = symbol
            frames.append(df[["instrument", "datetime", "close"]])

        if not frames:
            return pd.DataFrame(), fund_names, track_index

        merged = pd.concat(frames, ignore_index=True)
        merged = merged.set_index(["instrument", "datetime"]).sort_index()
        return merged[["close"]], fund_names, track_index

    def __call__(self) -> None:
        logger.info("EtfVisualizer start")
        logger.info("  input_dir : %s", self.input_dir)
        logger.info("  output_dir: %s", self.output_dir)

        if not self.input_dir.exists():
            logger.warning("input_dir does not exist, nothing to visualize: %s", self.input_dir)
            return

        df, fund_names, track_index = self._build_df()
        if df.empty:
            logger.warning("No usable close data in %s — nothing to visualize.", self.input_dir)
            return

        n_symbols = df.index.get_level_values("instrument").nunique()
        logger.info(
            "Loaded %d symbols (%d with names, %d with track_index)",
            n_symbols, len(fund_names), len(track_index),
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        split_info = self._load_split_segments()
        if split_info:
            logger.info("Using dataset split from %s (%d segments)", self.split_file, len(split_info))

        # 延迟 import：绘图函数依赖 matplotlib，且仅在执行时需要
        import sys
        vis_dir = Path(__file__).resolve().parent.parent.parent / "data_visulizers" / "data_distribution"
        sys.path.insert(0, str(vis_dir))
        from plot_lib import plot_aligned_etf, plot_subplots_etf

        plot_aligned_etf(
            df, self.output_dir / "all_etf_aligned.png",
            fund_names=fund_names, track_index=track_index, split_info=split_info,
        )
        plot_subplots_etf(
            df, self.output_dir / "all_etf_subplots.png",
            fund_names=fund_names, track_index=track_index, split_info=split_info,
        )

        logger.info("EtfVisualizer done — output: %s", self.output_dir)
