# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
腾讯财经 ETF 数据采集器

支持获取 ETF 完整历史数据，支持后复权/前复权
通过分段请求解决接口单次最多返回800条的限制

使用方法:
    # 下载单只 ETF 后复权数据
    python collector.py download_etf \
        --symbols "510050" \
        --source_dir ~/.qlib/stock_data/source/cn_etf_tencent \
        --fq_type hfq

    # 下载多只 ETF
    python collector.py download_etf \
        --symbols "510050,159915,518880" \
        --source_dir ~/.qlib/stock_data/source/cn_etf_tencent

    # 从 funds_list.csv 批量下载
    python collector.py download_etf \
        --funds_list /path/to/funds_list.csv \
        --source_dir ~/.qlib/stock_data/source/cn_etf_tencent

数据源说明:
    - API: https://web.ifzq.gtimg.cn/appstock/app/fqkline/get
    - 单次请求最多返回 800 条数据，通过分段请求获取完整历史
    - 支持复权: hfq=后复权, qfq=前复权, 空=不复权
"""

import sys
import fire
import requests
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import List
import time

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseRun


class TencentETFCollector:
    """腾讯财经 ETF 数据采集器
    
    支持获取 ETF 完整历史数据，支持后复权/前复权
    通过分段请求解决接口单次最多返回800条的限制
    """

    API_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    MAX_RECORDS_PER_REQUEST = 800

    def __init__(
        self,
        save_dir: [str, Path],
        symbols: List[str] = None,
        funds_list_path: str = None,
        fq_type: str = "hfq",
        delay: float = 0.5,
    ):
        """
        Parameters
        ----------
        save_dir: str
            保存目录
        symbols: list
            ETF 代码列表，如 ['510050', '159915']
        funds_list_path: str
            funds_list.csv 文件路径（可选）
        fq_type: str
            复权类型: hfq=后复权(默认), qfq=前复权, 空=不复权
        delay: float
            请求间隔，默认 0.5 秒
        """
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.symbols = symbols or []
        self.funds_list_path = funds_list_path
        self.fq_type = str(fq_type) if fq_type else ""
        self.delay = delay

    @staticmethod
    def normalize_tencent_symbol(symbol: str) -> str:
        """ETF 代码转换为腾讯格式

        上交所 ETF: 51xxxx, 58xxxx -> sh510050
        深交所 ETF: 15xxxx, 56xxxx -> sz159915

        Parameters
        ----------
        symbol: str
            ETF 代码，如 510050, 159915

        Returns
        -------
        str
            腾讯格式的代码，如 sh510050；不支持的类型返回 None
        """
        symbol = str(symbol).strip().zfill(6)
        if symbol.startswith("51") or symbol.startswith("58"):
            return f"sh{symbol}"
        if symbol.startswith("15") or symbol.startswith("56"):
            return f"sz{symbol}"
        return None

    def get_etf_kline_segment(self, symbol: str, end_date: str = None) -> list:
        """获取一段 ETF K线数据

        Parameters
        ----------
        symbol: str
            腾讯格式的代码，如 sh510050
        end_date: str
            结束日期 YYYY-MM-DD，默认当天

        Returns
        -------
        list
            K线数据列表，每条为 [date, open, close, high, low, volume]
        """
        if end_date is None:
            end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

        params = f"{symbol},day,,{end_date},{self.MAX_RECORDS_PER_REQUEST},{self.fq_type}"
        url = f"{self.API_URL}?param={params}"

        try:
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            data = resp.json()

            if data.get("code") == 0 and isinstance(data.get("data"), dict):
                etf_data = data["data"].get(symbol, {})
                # 后复权数据在 hfqday/qfqday 键中
                key = f"{self.fq_type}day" if self.fq_type else "day"
                klines = etf_data.get(key, [])
                return klines
        except Exception as e:
            logger.warning(f"get {symbol} segment error (end={end_date}): {e}")

        return []

    def get_etf_kline_full(self, symbol: str) -> pd.DataFrame:
        """分段获取 ETF 完整历史数据

        Parameters
        ----------
        symbol: str
            ETF 代码，如 510050

        Returns
        -------
        pd.DataFrame
            完整的K线数据
        """
        tencent_symbol = self.normalize_tencent_symbol(symbol)
        if not tencent_symbol:
            logger.warning(f"Unsupported ETF symbol: {symbol}")
            return pd.DataFrame()

        all_data = []
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
        max_iterations = 30  # 防止无限循环

        for _ in range(max_iterations):
            klines = self.get_etf_kline_segment(tencent_symbol, end_date)

            if not klines:
                break

            all_data.extend(klines)
            logger.debug(f"{symbol}: got {len(klines)} records ending at {end_date}")

            # 如果返回的数据少于请求数量，说明已经到头了
            if len(klines) < self.MAX_RECORDS_PER_REQUEST:
                break

            # 获取最早日期作为下一次请求的结束日期
            earliest_date = klines[0][0]
            # 结束日期需要前移一天，避免重复
            end_date = (pd.Timestamp(earliest_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            time.sleep(self.delay)

        if not all_data:
            return pd.DataFrame()

        # 转换为 DataFrame 并去重
        df = pd.DataFrame(all_data, columns=["date", "open", "close", "high", "low", "volume"])
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

        # 转换数据类型
        for col in ["open", "close", "high", "low"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("int64")

        return df

    def load_etf_list(self) -> List[str]:
        """从 funds_list.csv 加载并过滤支持的 ETF

        Returns
        -------
        list
            ETF 代码列表
        """
        if self.symbols:
            return self.symbols

        if not self.funds_list_path:
            logger.warning("No funds_list_path or symbols provided")
            return []

        df = pd.read_csv(self.funds_list_path, dtype=str, comment="#")
        df = df.dropna(subset=["fund_code"])

        etf_list = []
        skipped = []
        for _, row in df.iterrows():
            code = str(row["fund_code"]).strip()
            if self.normalize_tencent_symbol(code):
                etf_list.append(code)
            else:
                skipped.append(f"{code}({row.get('fund_name', '')})")

        if skipped:
            logger.warning(f"Skipped {len(skipped)} non-ETF funds: {skipped}")
        logger.info(f"Loaded {len(etf_list)} ETF symbols")
        return etf_list

    def collector_etf_kline(self):
        """采集 ETF K线数据"""
        symbols = self.load_etf_list()
        if not symbols:
            logger.warning("No ETF symbols to download")
            return

        fq_label = {"hfq": "后复权", "qfq": "前复权", "": "不复权"}.get(self.fq_type, self.fq_type)
        logger.info(f"Start collecting ETF kline for {len(symbols)} symbols ({fq_label})...")

        success_count = 0
        failed_list = []

        for symbol in tqdm(symbols, desc=f"采集ETF K线({fq_label})"):
            try:
                df = self.get_etf_kline_full(symbol)
                if not df.empty:
                    filename = f"{symbol.upper()}.csv"
                    filepath = self.save_dir / filename
                    df.to_csv(filepath, index=False)
                    success_count += 1
                    logger.info(f"Saved {symbol} -> {filepath}, rows={len(df)}, range={df.iloc[0]['date']}~{df.iloc[-1]['date']}")
                else:
                    failed_list.append(symbol)
                time.sleep(self.delay)
            except Exception as e:
                logger.warning(f"get {symbol} ETF kline error: {e}")
                failed_list.append(symbol)

        logger.info(f"Total {len(symbols)}, success: {success_count}, failed: {len(failed_list)}")
        if failed_list:
            logger.warning(f"Failed symbols: {failed_list}")


class Run(BaseRun):
    """运行入口"""

    def __init__(self, source_dir=None, normalize_dir=None, max_workers=4, interval="1d", region="CN"):
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return "TencentETFCollector"

    @property
    def normalize_class_name(self):
        return None

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_etf(
        self,
        symbols: str = None,
        funds_list: str = None,
        fq_type: str = "hfq",
        delay: float = 0.5,
    ):
        """下载 ETF K线数据（支持完整历史+复权）

        腾讯财经接口支持 ETF 完整历史数据和复权，突破 BaoStock 仅3个月的限制

        Parameters
        ----------
        symbols: str
            ETF 代码，多个用逗号分隔，如 '510050' 或 '510050,159915'
        funds_list: str
            funds_list.csv 文件路径
        fq_type: str
            复权类型: hfq=后复权(默认), qfq=前复权, 空=不复权
        delay: float
            请求间隔，默认 0.5 秒

        Examples
        ---------
            # 下载单只 ETF 后复权数据
            $ python collector.py download_etf \
                --symbols "510050" \
                --source_dir ~/.qlib/stock_data/source/cn_etf_tencent \
                --fq_type hfq

            # 从 funds_list.csv 批量下载
            $ python collector.py download_etf \
                --funds_list /path/to/funds_list.csv \
                --source_dir ~/.qlib/stock_data/source/cn_etf_tencent
        """
        if not funds_list and not symbols:
            logger.warning("Please specify --funds_list or --symbols")
            return

        symbol_list = None
        if symbols:
            if isinstance(symbols, int):
                symbols = str(symbols)
            symbol_list = [s.strip() for s in str(symbols).split(",")]

        collector = TencentETFCollector(
            save_dir=self.source_dir,
            symbols=symbol_list,
            funds_list_path=funds_list,
            fq_type=str(fq_type) if fq_type else "",
            delay=delay,
        )
        collector.collector_etf_kline()


if __name__ == "__main__":
    fire.Fire(Run)
