"""
MacroCollector — 宏观经济数据采集器

支持的 source 类型：
    - eastmoney_bond_rate  : 东方财富 中美国债收益率（cn_2y/cn_10y/us_10y 等）
    - akshare_macro_china_cpi : AkShare 中国 CPI 月度数据
    - akshare_macro_china_ppi : AkShare 中国 PPI 月度数据

每个 indicator 输出为独立 CSV：{output_dir}/{name}.csv，列为 date + value（或多值列）。
"""

import sys
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

from data_pipline.collectors.base import BaseCollector
from data_pipline.core.registry import register
from data_pipline.utils.http import retry

# ── 东方财富国债收益率 ─────────────────────────────────────────────────────────

_BOND_API_URL = "https://datacenter.eastmoney.com/api/data/get"
_BOND_PAGE_SIZE = 500
_BOND_FIELD_MAP = {
    "SOLAR_DATE":    "date",
    "EMM00588704":   "cn_2y",
    "EMM00166462":   "cn_5y",
    "EMM00166466":   "cn_10y",
    "EMM00166469":   "cn_30y",
    "EMM01276014":   "cn_spread_10m2",
    "EMG00001306":   "us_2y",
    "EMG00001308":   "us_5y",
    "EMG00001310":   "us_10y",
    "EMG00001312":   "us_30y",
    "EMG01339436":   "us_spread_10m2",
}


@retry(max_tries=4, delay=1.0, backoff=2.0, exceptions=(Exception,))
def _fetch_bond_page(page: int) -> dict:
    resp = requests.get(
        _BOND_API_URL,
        params={
            "type": "RPTA_WEB_TREASURYYIELD",
            "sty": "ALL",
            "st": "SOLAR_DATE",
            "sr": "1",
            "token": "894050c76af8597a853f5b408b759f5d",
            "p": page,
            "ps": _BOND_PAGE_SIZE,
        },
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    return resp.json()


def _fetch_bond_all(start: str) -> pd.DataFrame:
    first = _fetch_bond_page(1)
    result = first.get("result", {})
    total_pages = result.get("pages", 1)
    all_records = list(result.get("data", []))
    for page in range(2, total_pages + 1):
        try:
            data = _fetch_bond_page(page).get("result", {}).get("data", [])
            all_records.extend(data)
        except Exception as e:
            logger.warning(f"  bond page {page} failed: {e}")

    df = pd.DataFrame(all_records).rename(columns={k: v for k, v in _BOND_FIELD_MAP.items()})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[df["date"] >= start].sort_values("date").reset_index(drop=True)
    return df


# ── AkShare 宏观接口 ──────────────────────────────────────────────────────────

@retry(max_tries=4, delay=1.5, backoff=2.0, exceptions=(Exception,))
def _fetch_akshare_macro(func_name: str) -> pd.DataFrame:
    import akshare as ak
    return getattr(ak, func_name)()


def _normalize_macro_df(df: pd.DataFrame, start: str) -> pd.DataFrame:
    """将 akshare 宏观 DataFrame 规范化：找 date 列、统一格式"""
    date_col = next((c for c in df.columns if "日期" in c or c.lower() == "date"), None)
    if date_col is None:
        raise ValueError(f"No date column found. Columns: {list(df.columns)}")
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[df["date"] >= start].sort_values("date").reset_index(drop=True)
    return df


# source → 实际拉取函数
_SOURCE_HANDLERS = {
    "eastmoney_bond_rate":    lambda start: _fetch_bond_all(start),
    "akshare_macro_china_cpi": lambda start: _normalize_macro_df(
        _fetch_akshare_macro("macro_china_cpi"), start),
    "akshare_macro_china_ppi": lambda start: _normalize_macro_df(
        _fetch_akshare_macro("macro_china_ppi"), start),
}


@register("MacroCollector")
class MacroCollector(BaseCollector):
    """
    宏观经济数据采集器。

    YAML 参数：
        indicators: list[{name, source}]  各指标名及数据源
        start     : str                   全量起始日期
        force     : bool                  True = 全量覆盖
        output_dir: str                   输出子目录
    """

    def __init__(self, cfg: dict, output_base: Path):
        super().__init__(cfg, output_base)
        self.indicators: list[dict] = cfg.get("indicators", [])

    def __call__(self) -> None:
        if not self.indicators:
            logger.warning("MacroCollector: no indicators specified")
            return

        logger.info(f"MacroCollector: {len(self.indicators)} indicators → {self.output_dir}")
        failed = []

        for item in tqdm(self.indicators, desc="Macro"):
            name = item["name"]
            source = item["source"]
            handler = _SOURCE_HANDLERS.get(source)
            if handler is None:
                logger.warning(f"  [{name}] unknown source '{source}', skip")
                failed.append(name)
                continue
            try:
                inc_start = self._incremental_start(name)
                df = handler(inc_start)
                if df is None or df.empty:
                    logger.warning(f"  [{name}] empty response")
                    continue
                self._append_csv(name, df)
            except Exception as e:
                logger.error(f"  [{name}] failed: {e}")
                failed.append(name)

        if failed:
            logger.warning(f"MacroCollector: {len(failed)} failed: {failed}")
