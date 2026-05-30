# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
东方财富-中美国债收益率历史数据采集器

数据来源:
    东方财富数据中心（与 AkShare bond_zh_us_rate 同源）
    数据页面: https://data.eastmoney.com/cjsj/zmgzsyl.html

数据内容:
    - 日期
    - 中国国债收益率: 2年、5年、10年、30年
    - 美国国债收益率: 2年、5年、10年、30年
    - 历史覆盖: 约 1990 年至今

使用方法:
    # 下载全历史（默认保存到 ~/.qlib/stock_data/source/cn_bond_rate）
    python collector.py download_bond_rate \
        --source_dir ~/.qlib/stock_data/source/cn_bond_rate

    # 仅下载指定日期之后的数据
    python collector.py download_bond_rate \
        --source_dir ~/.qlib/stock_data/source/cn_bond_rate \
        --start_date 2010-01-01
"""

import sys
import fire
import requests
import pandas as pd
from pathlib import Path
from loguru import logger
import time

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseRun


class BondRateCollector:
    """10年期国债收益率历史数据采集器

    数据源: 东方财富数据中心 RPTA_WEB_TREASURYYIELD
    与 AkShare bond_zh_us_rate() 使用相同数据源，无外部依赖。

    输出字段:
        date            : 日期 (YYYY-MM-DD)
        cn_2y           : 中国国债收益率2年 (%)
        cn_5y           : 中国国债收益率5年 (%)
        cn_10y          : 中国国债收益率10年 (%)
        cn_30y          : 中国国债收益率30年 (%)
        cn_spread_10m2  : 中国 10年-2年 利差 (%)
        us_2y           : 美国国债收益率2年 (%)
        us_5y           : 美国国债收益率5年 (%)
        us_10y          : 美国国债收益率10年 (%)
        us_30y          : 美国国债收益率30年 (%)
        us_spread_10m2  : 美国 10年-2年 利差 (%)
    """

    API_URL = "https://datacenter.eastmoney.com/api/data/get"
    PAGE_SIZE = 500

    # 东方财富字段映射（与 AkShare bond_zh_us_rate 一致）
    FIELD_MAP = {
        "SOLAR_DATE": "date",
        "EMM00588704": "cn_2y",         # 中国国债收益率2年
        "EMM00166462": "cn_5y",         # 中国国债收益率5年
        "EMM00166466": "cn_10y",        # 中国国债收益率10年
        "EMM00166469": "cn_30y",        # 中国国债收益率30年
        "EMM01276014": "cn_spread_10m2",# 中国 10年-2年 利差
        "EMM00000024": "cn_gdp_yoy",    # 中国GDP年增率
        "EMG00001306": "us_2y",         # 美国国债收益率2年
        "EMG00001308": "us_5y",         # 美国国债收益率5年
        "EMG00001310": "us_10y",        # 美国国债收益率10年
        "EMG00001312": "us_30y",        # 美国国债收益率30年
        "EMG01339436": "us_spread_10m2",# 美国 10年-2年 利差
        "EMG00159635": "us_gdp_yoy",    # 美国GDP年增率
    }

    def __init__(
        self,
        save_dir: [str, Path],
        start_date: str = None,
        delay: float = 0.5,
    ):
        """
        Parameters
        ----------
        save_dir: str
            数据保存目录
        start_date: str
            起始日期 YYYY-MM-DD，默认获取所有历史数据（1990年起）
        delay: float
            翻页请求间隔秒数，默认 0.5
        """
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = start_date
        self.delay = delay

    def _fetch_page(self, page: int) -> dict:
        """获取单页数据"""
        params = {
            "type": "RPTA_WEB_TREASURYYIELD",
            "sty": "ALL",
            "st": "SOLAR_DATE",
            "sr": "1",   # 正序，从最早数据开始
            "token": "894050c76af8597a853f5b408b759f5d",
            "p": page,
            "ps": self.PAGE_SIZE,
            "pageNo": page,
            "pageNum": page,
        }
        resp = requests.get(
            self.API_URL,
            params=params,
            timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
        )
        return resp.json()

    def get_all_data(self) -> pd.DataFrame:
        """获取全量国债收益率历史数据

        Returns
        -------
        pd.DataFrame
            按日期升序排列的完整历史数据
        """
        # 先获取第一页，确定总页数
        logger.info("Fetching bond rate data from eastmoney...")
        try:
            first_page = self._fetch_page(1)
        except Exception as e:
            logger.error(f"Failed to fetch first page: {e}")
            return pd.DataFrame()

        result = first_page.get("result", {})
        if not result:
            logger.error(f"API returned no result: {first_page}")
            return pd.DataFrame()

        total_pages = result.get("pages", 1)
        total_count = result.get("count", 0)
        logger.info(f"Total records: {total_count}, total pages: {total_pages}")

        all_records = result.get("data", [])

        for page in range(2, total_pages + 1):
            try:
                page_data = self._fetch_page(page)
                records = page_data.get("result", {}).get("data", [])
                all_records.extend(records)
                logger.debug(f"Page {page}/{total_pages}: got {len(records)} records")
                time.sleep(self.delay)
            except Exception as e:
                logger.warning(f"Page {page} failed: {e}")

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)

        # 只保留已知字段
        available_cols = [c for c in self.FIELD_MAP.keys() if c in df.columns]
        df = df[available_cols].rename(columns=self.FIELD_MAP)

        # 数据类型转换
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        for col in df.columns:
            if col != "date":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

        # 按起始日期过滤
        if self.start_date:
            df = df[df["date"] >= self.start_date].reset_index(drop=True)

        return df

    def collect(self):
        """采集并保存国债收益率数据"""
        df = self.get_all_data()

        if df.empty:
            logger.error("No data collected, nothing saved")
            return

        filepath = self.save_dir / "cn_bond_yield.csv"

        # 追加模式：合并已有数据
        if filepath.exists():
            existing = pd.read_csv(filepath)
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

        df.to_csv(filepath, index=False)

        logger.info(
            f"Saved -> {filepath}\n"
            f"  rows={len(df)}\n"
            f"  date range: {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}\n"
            f"  columns: {df.columns.tolist()}"
        )


class Run(BaseRun):
    """命令行运行入口"""

    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d", region="CN"):
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return "BondRateCollector"

    @property
    def normalize_class_name(self):
        return None

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_bond_rate(
        self,
        start_date: str = None,
        delay: float = 0.5,
    ):
        """下载国债收益率历史数据（覆盖全历史，约1990年至今）

        数据包含中国和美国各期限国债收益率，输出文件: cn_bond_yield.csv

        Parameters
        ----------
        start_date: str
            起始日期 YYYY-MM-DD，默认获取所有历史（约 1990 年起）
        delay: float
            翻页请求间隔，默认 0.5 秒

        Examples
        ---------
            # 下载所有历史数据
            $ python collector.py download_bond_rate \\
                --source_dir ~/.qlib/stock_data/source/cn_bond_rate

            # 只下载 2010 年之后的数据
            $ python collector.py download_bond_rate \\
                --source_dir ~/.qlib/stock_data/source/cn_bond_rate \\
                --start_date 2010-01-01
        """
        collector = BondRateCollector(
            save_dir=self.source_dir,
            start_date=start_date,
            delay=delay,
        )
        collector.collect()


if __name__ == "__main__":
    fire.Fire(Run)
