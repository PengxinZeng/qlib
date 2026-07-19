"""
YahooCollector — Yahoo Finance ETF/股票 K线采集器

使用 yahooquery 拉取日线数据，支持代理配置与增量更新。
"""

import os
from pathlib import Path

import pandas as pd
from loguru import logger
from tqdm import tqdm

from data_pipline.collectors.base import BaseCollector
from data_pipline.core.registry import register
from data_pipline.utils.http import retry


@retry(max_tries=5, delay=2.0, backoff=2.0, exceptions=(Exception,))
def _fetch_yahoo(symbol: str, start: str, end: str) -> pd.DataFrame:
    from yahooquery import Ticker
    t = Ticker(symbol, timeout=30)
    df = t.history(start=start, end=end, interval="1d")
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level=0) if symbol in df.index.get_level_values(0) else df.reset_index(level=0, drop=True)
    df = df.reset_index()
    df = df.rename(columns={"date": "date", "open": "open", "high": "high",
                             "low": "low", "close": "close", "volume": "volume"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    keep = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in df.columns]
    return df[keep].drop_duplicates("date").sort_values("date").reset_index(drop=True)


@register("YahooCollector")
class YahooCollector(BaseCollector):
    """
    Yahoo Finance ETF K线采集器。

    YAML 参数：
        symbols   : list[str]  代码列表（如 QQQ, SPY）
        proxy     : str        HTTP 代理（可选，如 http://127.0.0.1:7890）
        start     : str        全量起始日期
        force     : bool       True = 全量覆盖
        output_dir: str        输出子目录
    """

    def __init__(self, cfg: dict, output_base: Path):
        super().__init__(cfg, output_base)
        self.symbols: list[str] = cfg.get("symbols", [])
        self.proxy: str | None = cfg.get("proxy")

    def __call__(self) -> None:
        if not self.symbols:
            logger.warning("YahooCollector: no symbols specified")
            return

        # 设置代理环境变量
        old_http = os.environ.get("HTTP_PROXY")
        old_https = os.environ.get("HTTPS_PROXY")
        if self.proxy:
            os.environ["HTTP_PROXY"] = self.proxy
            os.environ["HTTPS_PROXY"] = self.proxy
            logger.info(f"YahooCollector: proxy={self.proxy}")

        try:
            logger.info(f"YahooCollector: {len(self.symbols)} symbols → {self.output_dir}")
            end = (pd.Timestamp.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            failed = []

            for symbol in tqdm(self.symbols, desc="Yahoo"):
                try:
                    inc_start = self._incremental_start(symbol)
                    new_df = _fetch_yahoo(symbol, inc_start, end)
                    self._append_csv(symbol, new_df)
                except Exception as e:
                    logger.error(f"  [{symbol}] failed: {e}")
                    failed.append(symbol)

            if failed:
                logger.warning(f"YahooCollector: {len(failed)} failed: {failed}")
        finally:
            # 还原代理
            if self.proxy:
                if old_http is None:
                    os.environ.pop("HTTP_PROXY", None)
                else:
                    os.environ["HTTP_PROXY"] = old_http
                if old_https is None:
                    os.environ.pop("HTTPS_PROXY", None)
                else:
                    os.environ["HTTPS_PROXY"] = old_https
