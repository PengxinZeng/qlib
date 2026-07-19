"""
EastmoneyFundCollector — 天天基金网场外基金净值采集器

用于 ETF/场内基金无法覆盖的场外主动管理基金（如债券基金、FOF），
通过东方财富"历史净值"接口按代码拉取单位净值(DWJZ)/累计净值(LJJZ)。
"""

import time
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

from data_pipline.collectors.base import BaseCollector
from data_pipline.core.registry import register
from data_pipline.utils.http import retry

_API_URL = "https://api.fund.eastmoney.com/f10/lsjz"
_PAGE_SIZE = 20
_HEADERS = {
    "Referer": "https://fundf10.eastmoney.com/",
    "User-Agent": "Mozilla/5.0",
}


@retry(max_tries=5, delay=1.0, backoff=2.0, exceptions=(Exception,))
def _fetch_page(fund_code: str, start_date: str, end_date: str, page_index: int) -> tuple[list, int]:
    """拉取单页净值记录，返回 (rows, total_count)。"""
    params = {
        "fundCode": fund_code,
        "pageIndex": page_index,
        "pageSize": _PAGE_SIZE,
        "startDate": start_date,
        "endDate": end_date,
    }
    resp = requests.get(_API_URL, params=params, headers=_HEADERS, timeout=30)
    data = resp.json()
    if data.get("ErrCode") != 0:
        raise ValueError(f"{fund_code}: 接口返回错误 ErrCode={data.get('ErrCode')} ErrMsg={data.get('ErrMsg')}")
    payload = data.get("Data") or {}
    rows = payload.get("LSJZList") or []
    total_count = data.get("TotalCount", 0)
    return rows, total_count


def _fetch_since(fund_code: str, start_date: str, delay: float) -> pd.DataFrame:
    """按日期区间分页拉取全部净值记录，返回 >= start_date 的记录。"""
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    all_rows: list = []
    page_index = 1

    while True:
        rows, total_count = _fetch_page(fund_code, start_date, end_date, page_index)
        if not rows:
            break
        all_rows.extend(rows)
        if page_index * _PAGE_SIZE >= total_count:
            break
        page_index += 1
        time.sleep(delay)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)[["FSRQ", "DWJZ", "LJJZ"]]
    df.columns = ["date", "unit_nav", "cumulative_nav"]
    df = df[df["cumulative_nav"].str.strip() != ""]  # 累计净值为空的行（如停牌）跳过
    df["unit_nav"] = pd.to_numeric(df["unit_nav"], errors="coerce")
    df["cumulative_nav"] = pd.to_numeric(df["cumulative_nav"], errors="coerce")
    df = df.dropna(subset=["cumulative_nav"])
    df = df[df["date"] >= start_date]
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    # 规范 schema：场外基金无日内 OHLC / volume，以累计净值(含分红再投，
    # 口径对齐 ETF 后复权)合成 OHLC，volume 置 NaN。原始净值保留备查。
    df["open"] = df["cumulative_nav"]
    df["high"] = df["cumulative_nav"]
    df["low"] = df["cumulative_nav"]
    df["close"] = df["cumulative_nav"]
    df["volume"] = float("nan")
    df = df[["date", "open", "high", "low", "close", "volume", "unit_nav", "cumulative_nav"]]
    return df


@register("EastmoneyFundCollector")
class EastmoneyFundCollector(BaseCollector):
    """
    天天基金网场外基金净值采集器。

    YAML 参数：
        symbols   : list[dict]  {code, name, index}
        start     : str         全量起始日期（YAML anchor）
        force     : bool        True = 全量覆盖
        output_dir: str         输出子目录（相对 output_base）
        delay     : float       请求间隔秒数（默认 0.3）

    输出列：date, open, high, low, close, volume, unit_nav, cumulative_nav, name, track_index
    注：场外基金无日内 OHLC / volume 概念。以累计净值(cumulative_nav，含分红再投，
        口径对齐 ETF 后复权 hfq)合成 open=high=low=close，volume 置 NaN；
        原始单位净值(unit_nav)与累计净值(cumulative_nav)保留备查。
    """

    def __init__(self, cfg: dict, output_base: Path):
        super().__init__(cfg, output_base)
        raw_symbols = cfg.get("symbols", [])
        self.symbols: list[dict] = [
            s if isinstance(s, dict) else {"code": s, "name": s, "index": ""}
            for s in raw_symbols
        ]
        self.delay: float = float(cfg.get("delay", 0.3))

    def __call__(self) -> None:
        if not self.symbols:
            logger.warning("EastmoneyFundCollector: no symbols specified")
            return

        logger.info(f"EastmoneyFundCollector: {len(self.symbols)} symbols → {self.output_dir}")

        failed = []
        for sym_info in tqdm(self.symbols, desc="EastmoneyFund"):
            code = str(sym_info["code"]).strip().zfill(6)
            name = sym_info.get("name", code)
            try:
                inc_start = self._incremental_start(code)
                new_df = _fetch_since(code, inc_start, self.delay)
                if not new_df.empty:
                    new_df["name"] = name
                    new_df["track_index"] = sym_info.get("index", "")
                self._append_csv(code, new_df)
            except Exception as e:
                logger.error(f"  [{code}] failed: {e}")
                failed.append(code)

        if failed:
            logger.warning(f"EastmoneyFundCollector: {len(failed)} failed: {failed}")
