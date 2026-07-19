"""
QlibDumper — 将 merged/*_clean.csv 转换为 Qlib 二进制格式。

输出目录结构：
  {qlib_dir}/
    calendars/{freq}.txt      — 全局交易日历
    instruments/all.txt       — symbol\tstart\tend
    features/{symbol}/
      {field}.{freq}.bin      — float32 数组，首元素为 calendar offset

设计说明：
- 每个 .bin 文件以 float32 写入：[date_offset, v0, v1, ..., vN]
  date_offset 是该 symbol 首个交易日在全局日历中的索引（Qlib 约定）
- 多进程并发写出（ProcessPoolExecutor），max_workers 可配置
- 输入文件名约定：{symbol}_clean.csv，symbol 取 stem 去掉 "_clean" 后缀
"""

import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_pipline.core.registry import register
from data_pipline.processors.base import BaseProcessor

logger = logging.getLogger(__name__)

# 非特征列，不做 dump：
# - 字符串元信息(name/track_index)会导致 float 转换失败
# - 净值列(unit_nav/cumulative_nav)为中间口径，不作为 qlib 特征
_SKIP_COLS = {
    "date", "symbol", "data_source",
    "name", "track_index",
    "unit_nav", "cumulative_nav",
}


def _normalize_field(name: str) -> str:
    return name.lower()


def _save_symbol(args: tuple) -> None:
    """顶层函数（picklable），供 ProcessPoolExecutor 调用。"""
    (
        symbol,
        rows,          # list of (date, {field: value})
        date_to_idx,   # dict[pd.Timestamp, int]
        all_fields,    # list[str]
        features_dir,  # Path
        freq,          # str
    ) = args

    from qlib.utils import code_to_fname
    import numpy as np
    from pathlib import Path

    symbol_dir: Path = features_dir / code_to_fname(symbol.lower())
    symbol_dir.mkdir(parents=True, exist_ok=True)

    # Build per-field arrays
    # rows: [(date, dict), ...]  — pre-sorted by date
    dates = [r[0] for r in rows]
    date_index = date_to_idx[min(dates)]

    field_arrays: dict[str, np.ndarray] = {
        f: np.full(len(date_to_idx), np.nan, dtype=np.float32) for f in all_fields
    }

    for date, values in rows:
        idx = date_to_idx.get(date)
        if idx is None:
            continue
        for f, v in values.items():
            if f in field_arrays:
                if pd.notna(v):
                    field_arrays[f][idx] = float(v)

    for field, arr in field_arrays.items():
        # Slice from date_index onward
        field_data = arr[date_index:]
        bin_path = symbol_dir / f"{field}.{freq}.bin"
        np.hstack([date_index, field_data]).astype("<f").tofile(str(bin_path.resolve()))


@register("QlibDumper")
class QlibDumper(BaseProcessor):

    def __init__(self, cfg: dict, output_base: Path):
        super().__init__(cfg, output_base)
        self.data_path: Path = (output_base / cfg["data_path"]).resolve()
        self.qlib_dir: Path = (output_base / cfg["qlib_dir"]).resolve()
        self.freq: str = cfg.get("freq", "day")
        self.max_workers: int = int(cfg.get("max_workers", 16))

    # ──────────────────────────────────────────────────────────────

    def __call__(self) -> None:
        logger.info("QlibDumper start")
        logger.info("  data_path  : %s", self.data_path)
        logger.info("  qlib_dir   : %s", self.qlib_dir)
        logger.info("  freq       : %s", self.freq)
        logger.info("  max_workers: %d", self.max_workers)

        csv_files = sorted(self.data_path.glob("*_clean.csv"))
        if not csv_files:
            logger.warning("No *_clean.csv files found in %s", self.data_path)
            return
        logger.info("Found %d *_clean.csv files", len(csv_files))

        # ── Setup output dirs ──────────────────────────────────────
        calendars_dir = self.qlib_dir / "calendars"
        features_dir = self.qlib_dir / "features"
        instruments_dir = self.qlib_dir / "instruments"
        for d in (calendars_dir, features_dir, instruments_dir):
            d.mkdir(parents=True, exist_ok=True)

        # ── Read all CSVs ─────────────────────────────────────────
        logger.info("Reading CSV files...")
        all_dates: set[pd.Timestamp] = set()
        instruments_data: list[dict] = []
        # symbol -> list[(date, {field: value})]
        symbol_rows: dict[str, list] = {}
        all_fields_set: set[str] = set()

        for csv_path in tqdm(csv_files, desc="Reading"):
            raw_stem = csv_path.stem  # e.g. "510300_clean"
            symbol = raw_stem.removesuffix("_clean")

            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"])
            if df.empty:
                continue

            df = df.sort_values("date").reset_index(drop=True)

            # Normalize column names
            rename = {c: _normalize_field(c) for c in df.columns}
            df = df.rename(columns=rename)

            fields = [c for c in df.columns if c not in _SKIP_COLS]
            all_fields_set.update(fields)

            rows: list[tuple] = []
            for _, row in df.iterrows():
                date = row["date"]
                values = {f: row[f] for f in fields}
                rows.append((date, values))
                all_dates.add(date)

            symbol_rows[symbol] = rows

            start_date = df["date"].min()
            end_date = df["date"].max()
            instruments_data.append({
                "symbol": symbol.upper(),
                "start": start_date.strftime("%Y-%m-%d"),
                "end": end_date.strftime("%Y-%m-%d"),
            })

        if not symbol_rows:
            logger.warning("No data loaded — aborting.")
            return

        all_fields = sorted(all_fields_set)
        sorted_dates = sorted(all_dates)
        date_to_idx: dict[pd.Timestamp, int] = {d: i for i, d in enumerate(sorted_dates)}

        # ── Write calendar ─────────────────────────────────────────
        calendar_path = calendars_dir / f"{self.freq}.txt"
        with open(calendar_path, "w") as f:
            for d in sorted_dates:
                f.write(d.strftime("%Y-%m-%d") + "\n")
        logger.info("Calendar written: %d dates → %s", len(sorted_dates), calendar_path)

        # ── Write instruments ──────────────────────────────────────
        instruments_path = instruments_dir / "all.txt"
        with open(instruments_path, "w") as f:
            for inst in instruments_data:
                f.write(f"{inst['symbol']}\t{inst['start']}\t{inst['end']}\n")
        logger.info("Instruments written: %d symbols → %s", len(instruments_data), instruments_path)

        # ── Write features (parallel) ─────────────────────────────
        logger.info("Writing feature .bin files with %d workers...", self.max_workers)
        args_list = [
            (symbol, rows, date_to_idx, all_fields, features_dir, self.freq)
            for symbol, rows in symbol_rows.items()
        ]
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            list(tqdm(executor.map(_save_symbol, args_list), total=len(args_list), desc="Dumping"))

        logger.info(
            "QlibDumper done — %d symbols, %d dates, %d fields, qlib_dir: %s",
            len(symbol_rows), len(sorted_dates), len(all_fields), self.qlib_dir,
        )
