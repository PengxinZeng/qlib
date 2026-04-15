# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
基金历史K线数据采集器 (基于 AkShare)

支持获取:
1. 场内ETF的历史K线数据（通过新浪财经接口）
2. 场外基金的历史净值数据（通过天天基金接口）

使用方法:
    # 下载单只基金数据
    python collector.py download_fund \
        --symbols "510050" \
        --source_dir ~/.qlib/fund_data/source/cn_fund

    # 下载多只基金
    python collector.py download_fund \
        --symbols "510050,159915,518880" \
        --source_dir ~/.qlib/fund_data/source/cn_fund

    # 从 funds_list.csv 批量下载
    python collector.py download_fund \
        --funds_list /path/to/funds_list.csv \
        --source_dir ~/.qlib/fund_data/source/cn_fund

数据源说明:
    - ETF数据: 新浪财经（fund_etf_hist_sina），返回不复权数据
    - 场外基金: 天天基金（fund_open_fund_info_em），返回累计净值数据
    - 数据从基金成立日期开始
"""

import sys
import fire
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import List, Tuple
import time


class AkshareFundCollector:
    """AkShare 基金数据采集器
    
    支持获取场内ETF和场外基金的历史数据
    """

    def __init__(
        self,
        save_dir: [str, Path],
        symbols: List[str] = None,
        funds_list_path: str = None,
        delay: float = 0.5,
    ):
        """
        Parameters
        ----------
        save_dir: str
            保存目录
        symbols: list
            基金代码列表，如 ['510050', '159915']
        funds_list_path: str
            funds_list.csv 文件路径（可选）
        delay: float
            请求间隔，默认 0.5 秒
        """
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.symbols = symbols or []
        self.funds_list_path = funds_list_path
        self.delay = delay
        
        # 延迟导入 akshare
        try:
            import akshare as ak
            self.ak = ak
        except ImportError:
            raise ImportError("请先安装 akshare: pip install akshare")

    @staticmethod
    def classify_fund(symbol: str) -> Tuple[str, str]:
        """判断基金类型
        
        Parameters
        ----------
        symbol: str
            基金代码
            
        Returns
        -------
        Tuple[str, str]
            (基金类型, 新浪格式代码)
            基金类型: 'etf_sh', 'etf_sz', 'fund'
            新浪格式代码: 如 'sh510050', 'sz159915', 或原始代码
        """
        symbol = str(symbol).strip().zfill(6)
        
        # 上交所ETF: 51xxxx, 58xxxx, 50xxxx
        if symbol.startswith(("51", "58", "50")):
            return "etf_sh", f"sh{symbol}"
        
        # 深交所ETF: 15xxxx, 16xxxx, 56xxxx
        if symbol.startswith(("15", "16", "56")):
            return "etf_sz", f"sz{symbol}"
        
        # 场外基金
        return "fund", symbol

    def get_etf_kline(self, symbol: str, sina_symbol: str) -> pd.DataFrame:
        """获取ETF K线数据（新浪财经）
        
        Parameters
        ----------
        symbol: str
            原始基金代码
        sina_symbol: str
            新浪格式代码，如 sh510050
            
        Returns
        -------
        pd.DataFrame
            K线数据
        """
        try:
            df = self.ak.fund_etf_hist_sina(symbol=sina_symbol)
            if df.empty:
                return pd.DataFrame()
            
            # 标准化列名
            df = df.rename(columns={
                "prevclose": "preclose",
            })
            
            # 确保有需要的列
            required_cols = ["date", "open", "high", "low", "close", "volume"]
            available_cols = [c for c in required_cols if c in df.columns]
            if "amount" in df.columns:
                available_cols.append("amount")
            if "preclose" in df.columns:
                available_cols.append("preclose")
            
            df = df[available_cols]
            
            # 转换日期格式
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            
            # 按日期升序排列
            df = df.sort_values("date").reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.warning(f"get ETF {symbol} kline error: {e}")
            return pd.DataFrame()

    def get_fund_nav(self, symbol: str) -> pd.DataFrame:
        """获取场外基金净值数据（天天基金）
        
        Parameters
        ----------
        symbol: str
            基金代码，如 000217
            
        Returns
        -------
        pd.DataFrame
            净值数据（包含累计净值走势）
        """
        try:
            df = self.ak.fund_open_fund_info_em(symbol=symbol, indicator="累计净值走势")
            if df.empty:
                return pd.DataFrame()
            
            # 重命名列
            df = df.rename(columns={
                "净值日期": "date",
                "累计净值": "nav",  # 累计净值
            })
            
            # 转换日期格式
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            
            # 按日期升序排列
            df = df.sort_values("date").reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logger.warning(f"get fund {symbol} nav error: {e}")
            return pd.DataFrame()

    def get_fund_data(self, symbol: str) -> Tuple[pd.DataFrame, str]:
        """获取基金数据
        
        Parameters
        ----------
        symbol: str
            基金代码
            
        Returns
        -------
        Tuple[pd.DataFrame, str]
            (数据DataFrame, 数据类型 'kline' 或 'nav')
        """
        fund_type, sina_symbol = self.classify_fund(symbol)
        
        if fund_type in ["etf_sh", "etf_sz"]:
            df = self.get_etf_kline(symbol, sina_symbol)
            return df, "kline"
        else:
            df = self.get_fund_nav(symbol)
            return df, "nav"

    def load_fund_list(self) -> List[str]:
        """从 funds_list.csv 加载基金列表
        
        Returns
        -------
        list
            基金代码列表
        """
        if self.symbols:
            return self.symbols

        if not self.funds_list_path:
            logger.warning("No funds_list_path or symbols provided")
            return []

        df = pd.read_csv(self.funds_list_path, dtype=str, comment="#")
        df = df.dropna(subset=["fund_code"])

        fund_list = []
        for _, row in df.iterrows():
            code = str(row["fund_code"]).strip()
            fund_list.append(code)

        logger.info(f"Loaded {len(fund_list)} fund symbols")
        return fund_list

    def collector_fund_data(self):
        """采集基金数据"""
        symbols = self.load_fund_list()
        if not symbols:
            logger.warning("No fund symbols to download")
            return

        logger.info(f"Start collecting fund data for {len(symbols)} symbols...")

        success_count = 0
        failed_list = []
        etf_count = 0
        fund_count = 0

        for symbol in tqdm(symbols, desc="采集基金数据(AkShare)"):
            try:
                df, data_type = self.get_fund_data(symbol)
                if not df.empty:
                    filename = f"{symbol.upper()}.csv"
                    filepath = self.save_dir / filename
                    df.to_csv(filepath, index=False)
                    success_count += 1
                    
                    if data_type == "kline":
                        etf_count += 1
                        logger.info(f"Saved ETF {symbol} -> {filepath}, rows={len(df)}, range={df.iloc[0]['date']}~{df.iloc[-1]['date']}")
                    else:
                        fund_count += 1
                        logger.info(f"Saved Fund {symbol} -> {filepath}, rows={len(df)}, range={df.iloc[0]['date']}~{df.iloc[-1]['date']}")
                else:
                    failed_list.append(symbol)
                time.sleep(self.delay)
            except Exception as e:
                logger.warning(f"get {symbol} data error: {e}")
                failed_list.append(symbol)

        logger.info(f"Total {len(symbols)}, success: {success_count} (ETF: {etf_count}, Fund: {fund_count}), failed: {len(failed_list)}")
        if failed_list:
            logger.warning(f"Failed symbols: {failed_list}")


def download_fund(
    symbols: str = None,
    funds_list: str = None,
    source_dir: str = None,
    delay: float = 0.5,
):
    """下载基金历史数据（AkShare数据源）

    获取基金完整历史数据（从基金成立日期开始）:
    - 场内ETF: 使用新浪财经接口获取K线数据（不复权）
    - 场外基金: 使用天天基金接口获取累计净值数据

    Parameters
    ----------
    symbols: str
        基金代码，多个用逗号分隔，如 '510050' 或 '510050,159915,000217'
    funds_list: str
        funds_list.csv 文件路径
    source_dir: str
        保存目录
    delay: float
        请求间隔，默认 0.5 秒

    Examples
    ---------
        # 下载单只基金数据
        $ python collector.py download_fund \
            --symbols "510050" \
            --source_dir ~/.qlib/fund_data/source/cn_fund

        # 下载多只基金
        $ python collector.py download_fund \
            --symbols "510050,159915,000217" \
            --source_dir ~/.qlib/fund_data/source/cn_fund

        # 从 funds_list.csv 批量下载
        $ python collector.py download_fund \
            --funds_list /path/to/funds_list.csv \
            --source_dir ~/.qlib/fund_data/source/cn_fund
    """
    if not source_dir:
        logger.warning("Please specify --source_dir")
        return
        
    if not funds_list and not symbols:
        logger.warning("Please specify --funds_list or --symbols")
        return

    symbol_list = None
    if symbols:
        if isinstance(symbols, int):
            symbols = str(symbols)
        symbol_list = [s.strip() for s in str(symbols).split(",")]

    collector = AkshareFundCollector(
        save_dir=source_dir,
        symbols=symbol_list,
        funds_list_path=funds_list,
        delay=delay,
    )
    collector.collector_fund_data()


if __name__ == "__main__":
    fire.Fire({
        "download_fund": download_fund,
    })
