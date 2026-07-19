"""
AkshareCollector — AkShare 大宗商品 / 有色金属 K线采集器

支持通过 symbols 列表中每项的 code + source 字段，
路由到对应的 akshare 接口拉取数据，并增量更新到 CSV。
"""

from pathlib import Path

import akshare as ak
import pandas as pd
from loguru import logger
from tqdm import tqdm

from data_pipline.collectors.base import BaseCollector
from data_pipline.core.registry import register
from data_pipline.utils.http import retry

# ── akshare 接口路由 ──────────────────────────────────────────────────────────
# source 字段 → (akshare 函数名, 调用参数 key)
_SOURCE_MAP = {
    "futures_index_unsh_sina": ("futures_index_unsh_sina", "symbol"),
    "futures_zh_daily_sina":   ("futures_zh_daily_sina",   "symbol"),
}

# 各接口返回列名 → 标准列名映射
_COL_RENAME = {
    "date":   "date",
    "open":   "open",
    "high":   "high",
    "low":    "low",
    "close":  "close",
    "volume": "volume",
    "hold":   "volume",   # 部分期货接口用 hold 表示持仓/成交量
    "成交量":  "volume",
    "开盘价":  "open",
    "最高价":  "high",
    "最低价":  "low",
    "收盘价":  "close",
    "日期":    "date",
}


@retry(max_tries=4, delay=1.5, backoff=2.0, exceptions=(Exception,))
def _call_akshare(func_name: str, symbol: str) -> pd.DataFrame:
    func = getattr(ak, func_name)
    df = func(symbol=symbol)
    return df


def _normalize_df(df: pd.DataFrame, start: str) -> pd.DataFrame:
    """统一列名、类型，并过滤起始日期"""
    df = df.rename(columns={k: v for k, v in _COL_RENAME.items() if k in df.columns})
    if "date" not in df.columns:
        raise ValueError(f"No 'date' column found. Columns: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[df["date"] >= start]
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    else:
        df["volume"] = 0
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep].drop_duplicates("date").sort_values("date").reset_index(drop=True)


@register("AkshareCollector")
class AkshareCollector(BaseCollector):
    """
    AkShare 大宗商品 / 有色金属采集器。

    YAML 参数：
        symbols   : list[{code, source}]  各品种代码及数据源
        start     : str                   全量起始日期
        force     : bool                  True = 全量覆盖
        output_dir: str                   输出子目录
    """

    def __init__(self, cfg: dict, output_base: Path):
        super().__init__(cfg, output_base)
        self.symbols: list[dict] = cfg.get("symbols", [])

    def __call__(self) -> None:
        if not self.symbols:
            logger.warning("AkshareCollector: no symbols specified")
            return

        logger.info(f"AkshareCollector: {len(self.symbols)} symbols → {self.output_dir}")
        failed = []

        for item in tqdm(self.symbols, desc="Akshare"):
            code = item["code"]
            source = item["source"]
            if source not in _SOURCE_MAP:
                logger.warning(f"  [{code}] unknown source '{source}', skip")
                failed.append(code)
                continue
            func_name, _ = _SOURCE_MAP[source]
            try:
                inc_start = self._incremental_start(code)
                raw_df = _call_akshare(func_name, code)
                if raw_df is None or raw_df.empty:
                    logger.warning(f"  [{code}] empty response")
                    continue
                new_df = _normalize_df(raw_df, inc_start)
                self._append_csv(code, new_df)
            except Exception as e:
                logger.error(f"  [{code}] failed: {e}")
                failed.append(code)

        if failed:
            logger.warning(f"AkshareCollector: {len(failed)} failed: {failed}")
