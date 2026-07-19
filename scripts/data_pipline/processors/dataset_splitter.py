"""
DatasetSplitter — 基于每日可交易标的数量的累计和切分数据集。

处理逻辑：
1. 读 input_dir 下所有 {symbol}_clean.csv，构建全局日历（日期并集）
2. 统计每个交易日 close 非 NaN 的标的数量（close 非 NaN = 可交易）
3. 按 yaml 配置的 splits(集合名 + 比例) 对累计可交易量做切分
4. 输出到 output_dir：
   - split_info.json          : 各集合起止日期/天数/占比等（下游 workflow 直接取用）
   - split_by_cumsum.csv      : 各集合汇总统计
   - daily_split.csv          : 每日明细并标注所属集合
   - dataset_split_cumsum.png : 双子图可视化（每日数量按集合着色 + 累计曲线/分割线/阈值线）

切分方案（集合名、比例）完全由 pipeline.yaml 配置：
    splits:
      - {name: train, ratio: 0.50}
      - ...
"""

import json
import logging
from pathlib import Path

import pandas as pd

from data_pipline.core.registry import register
from data_pipline.processors.base import BaseProcessor

logger = logging.getLogger(__name__)


@register("DatasetSplitter")
class DatasetSplitter(BaseProcessor):

    def __init__(self, cfg: dict, output_base: Path):
        super().__init__(cfg, output_base)
        self.input_dir: Path = (output_base / cfg["input_dir"]).resolve()
        self.output_dir: Path = (output_base / cfg["output_dir"]).resolve()

        raw_splits = cfg.get("splits") or []
        self.splits: list[tuple[str, float]] = [
            (str(s["name"]), float(s["ratio"])) for s in raw_splits
        ]
        self._validate_splits()

    def _validate_splits(self) -> None:
        if not self.splits:
            raise ValueError("DatasetSplitter: 'splits' 配置为空")
        names = [n for n, _ in self.splits]
        if len(names) != len(set(names)):
            raise ValueError(f"DatasetSplitter: 集合名重复: {names}")
        if any(r <= 0 for _, r in self.splits):
            raise ValueError("DatasetSplitter: 存在非正的 ratio")
        total = sum(r for _, r in self.splits)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"DatasetSplitter: ratio 之和应为 1.0，当前为 {total:.6f}")

    def _build_daily_counts(self) -> pd.DataFrame:
        """从 cleaned CSV 统计每日可交易(close 非 NaN)标的数量。"""
        counts: dict[pd.Timestamp, int] = {}
        for csv_path in sorted(self.input_dir.glob("*_clean.csv")):
            df = pd.read_csv(csv_path, usecols=lambda c: c in ("date", "close"))
            if "close" not in df.columns or df.empty:
                continue
            df = df.dropna(subset=["close"])
            if df.empty:
                continue
            for d in pd.to_datetime(df["date"]):
                counts[d] = counts.get(d, 0) + 1

        if not counts:
            return pd.DataFrame(columns=["date", "tradeable_count"])

        out = pd.DataFrame(
            {"date": list(counts.keys()), "tradeable_count": list(counts.values())}
        )
        return out.sort_values("date").reset_index(drop=True)

    def __call__(self) -> None:
        logger.info("DatasetSplitter start")
        logger.info("  input_dir : %s", self.input_dir)
        logger.info("  output_dir: %s", self.output_dir)
        logger.info("  splits    : %s", self.splits)

        if not self.input_dir.exists():
            logger.warning("input_dir does not exist, nothing to split: %s", self.input_dir)
            return

        df = self._build_daily_counts()
        if df.empty:
            logger.warning("No tradeable data in %s — nothing to split.", self.input_dir)
            return

        logger.info(
            "Calendar: %d trading days, %s ~ %s",
            len(df), df["date"].min().date(), df["date"].max().date(),
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 延迟 import：切分/绘图函数（含 matplotlib），仅执行时需要
        import sys
        lib_dir = Path(__file__).resolve().parent.parent.parent / "data_processors" / "split_dataset"
        sys.path.insert(0, str(lib_dir))
        from split_lib import split_by_cumsum, build_results, plot_split

        splits_result, total_sum = split_by_cumsum(df, self.splits)
        df["cumsum"] = df["tradeable_count"].cumsum()
        results = build_results(df, splits_result, total_sum)

        # split_info.json
        split_info = {
            r["dataset"]: {
                "start_date": r["start_date"],
                "end_date": r["end_date"],
                "days": r["days"],
                "total_count": r["total_count"],
                "pct_of_total_count": r["pct_of_total_count"],
                "avg_tradeable": r["avg_tradeable"],
            }
            for r in results
        }
        with open(self.output_dir / "split_info.json", "w") as f:
            json.dump(split_info, f, indent=2, ensure_ascii=False)

        # 汇总 & 每日明细
        pd.DataFrame(results).to_csv(self.output_dir / "split_by_cumsum.csv", index=False)
        df["dataset"] = "unknown"
        for name, start_idx, end_idx in splits_result:
            df.loc[start_idx:end_idx, "dataset"] = name
        df.to_csv(self.output_dir / "daily_split.csv", index=False)

        # 可视化
        plot_split(df, results, total_sum, self.output_dir / "dataset_split_cumsum.png")

        for r in results:
            logger.info(
                "  %-8s %s ~ %s | %d days | %s of count",
                r["dataset"], r["start_date"], r["end_date"], r["days"], r["pct_of_total_count"],
            )
        logger.info("DatasetSplitter done — output: %s", self.output_dir)
