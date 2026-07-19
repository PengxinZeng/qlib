"""
BaseCollector — 抽象采集器基类

所有 Collector 继承此类，实现 __call__ 完成数据下载与写入。
内置增量更新逻辑：读取已有 CSV 末尾日期，仅拉取新数据并追加。
force=True 时全量覆盖。
"""

import abc
from pathlib import Path

import pandas as pd
from loguru import logger


class BaseCollector(abc.ABC):

    def __init__(self, cfg: dict, output_base: Path):
        self.output_dir = (output_base / cfg["output_dir"]).resolve()
        self.start: str = cfg["start"]          # 由 YAML anchor *start 注入
        self.force: bool = cfg.get("force", False)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 增量工具 ────────────────────────────────────────────────────────

    def _incremental_start(self, symbol: str) -> str:
        """
        增量模式：读取已有 CSV 的最后日期，返回下次拉取起点（末尾 +1 天）。
        文件不存在或 force=True 时，返回 self.start。
        """
        if self.force:
            return self.start
        csv_path = self.output_dir / f"{symbol}.csv"
        if not csv_path.exists():
            return self.start
        try:
            df = pd.read_csv(csv_path, usecols=["date"])
            if df.empty:
                return self.start
            last = pd.to_datetime(df["date"]).max()
            next_start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            logger.debug(f"  [{symbol}] incremental start: {next_start}")
            return next_start
        except Exception as e:
            logger.warning(f"  [{symbol}] failed to read existing CSV: {e}, fallback to {self.start}")
            return self.start

    def _append_csv(self, symbol: str, new_df: pd.DataFrame) -> None:
        """
        将新数据写入 CSV。
        - force=True 或文件不存在：直接覆盖写入。
        - 否则：读取旧数据，按 date 去重合并后写回。
        """
        if new_df is None or new_df.empty:
            logger.debug(f"  [{symbol}] no new data, skip writing")
            return

        csv_path = self.output_dir / f"{symbol}.csv"

        if self.force or not csv_path.exists():
            new_df.to_csv(csv_path, index=False)
            logger.debug(f"  [{symbol}] written {len(new_df)} rows → {csv_path}")
        else:
            old_df = pd.read_csv(csv_path)
            # 新数据优先：concat 时 new_df 在后，keep="last" 保留新行
            combined = (
                pd.concat([old_df, new_df], ignore_index=True)
                .drop_duplicates(subset=["date"], keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
            combined.to_csv(csv_path, index=False)
            logger.debug(
                f"  [{symbol}] appended {len(new_df)} rows "
                f"(total {len(combined)}) → {csv_path}"
            )

    # ── 子类实现 ────────────────────────────────────────────────────────

    @abc.abstractmethod
    def __call__(self) -> None:
        """下载数据并调用 _append_csv 写入 output_dir，每个 symbol 一个 CSV"""
