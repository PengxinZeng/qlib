# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Baostock 完整K线数据采集器

支持获取A股日线K线数据 + 估值指标（PB、PE、PS、PCF）

使用方法:
    # 下载完整K线数据（含K线+估值）
    python collector.py download_full_kline \
        --symbols "600519" \
        --source_dir ~/.qlib/stock_data/source/cn_full_kline \
        --end 2026-04-11

    # 下载多只股票
    python collector.py download_full_kline \
        --symbols "600519,600000,000001" \
        --source_dir ~/.qlib/stock_data/source/cn_full_kline

    # 标准化数据
    python collector.py normalize_daily_data \
        --source_dir ~/.qlib/stock_data/source/cn_full_kline \
        --normalize_dir ~/.qlib/stock_data/source/cn_full_kline_nor

    # 导出到 Qlib 格式
    python dump_bin.py dump_all \
        --data_path ~/.qlib/stock_data/source/cn_full_kline_nor \
        --qlib_dir ~/.qlib/qlib_data/cn_data \
        --freq day \
        --exclude_fields date,symbol
"""

import sys
import fire
import numpy as np
import pandas as pd
import baostock as bs
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Iterable, List
import time

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseNormalize, BaseRun


def normalize_bs_symbol(symbol: str) -> str:
    """标准化股票代码为 Baostock 格式

    Parameters
    ----------
    symbol: str
        股票代码，如 sh.600519, 600519, SH600519 等

    Returns
    -------
    str
        Baostock 格式的股票代码，如 sh.600519
    """
    symbol = symbol.strip()
    if symbol.startswith("sh.") or symbol.startswith("sz."):
        return symbol
    if symbol.upper().startswith("SH"):
        return "sh." + symbol[2:]
    if symbol.upper().startswith("SZ"):
        return "sz." + symbol[2:]
    if symbol.startswith("6"):
        return f"sh.{symbol}"
    return f"sz.{symbol}"


class BaostockFullKLineCollector:
    """Baostock 完整K线数据采集器

    获取包含K线数据 + 估值指标的完整日线数据
    支持指定单个或多个股票代码

    数据字段:
        date, code, open, high, low, close, preclose, volume, amount,
        adjustflag, turn, peTTM, pbMRQ, psTTM, pcfNcfTTM, tradestatus, pctChg, isST
    """

    # 完整K线字段
    FULL_KLINE_FIELDS = (
        "date,code,open,high,low,close,preclose,volume,amount,"
        "adjustflag,turn,peTTM,pbMRQ,psTTM,pcfNcfTTM,tradestatus,pctChg,isST"
    )

    def __init__(
        self,
        save_dir: [str, Path],
        symbols: List[str] = None,
        start=None,
        end=None,
        delay=0.5,
    ):
        """
        Parameters
        ----------
        save_dir: str
            save directory
        symbols: list
            股票代码列表，如 ['sh.600519', '600000'] 或 ['600519']
        start: str
            开始日期，默认 None (从上市日期开始)
        end: str
            结束日期，默认 None (到今天)
        delay: float
            请求间隔，默认 0.5 秒
        """
        bs.login()
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.delay = delay
        self.symbols = symbols or []
        self.start_datetime = pd.Timestamp(start) if start else None
        self.end_datetime = pd.Timestamp(end) if end else pd.Timestamp.today()

    def _normalize_symbol(self, symbol: str):
        """标准化股票代码"""
        return symbol.replace(".", "").upper()

    def sleep(self):
        time.sleep(self.delay)

    @staticmethod
    def get_stock_ipo_date(symbol: str) -> str:
        """获取股票上市日期

        Parameters
        ----------
        symbol: str
            股票代码，如 sh.600519 或 600519

        Returns
        -------
        str
            上市日期字符串，如 '2001-08-27'，获取失败返回 None
        """
        bs_symbol = normalize_bs_symbol(symbol)
        rs = bs.query_stock_basic(code=bs_symbol)
        while rs.error_code == '0' and rs.next():
            row = rs.get_row_data()
            # fields: [code, code_name, ipoDate, outDate, stockType, status]
            if len(row) >= 3 and row[2]:
                return row[2]
        return None

    @staticmethod
    def get_full_kline_from_remote(
        symbol: str,
        start_datetime: pd.Timestamp = None,
        end_datetime: pd.Timestamp = None,
        adjustflag: str = "2"
    ) -> pd.DataFrame:
        """从 Baostock 获取单只股票的完整K线数据（含估值指标）

        Parameters
        ----------
        symbol: str
            股票代码，如 sh.600519 或 600519
        start_datetime: pd.Timestamp
            开始日期，默认从上市日期开始
        end_datetime: pd.Timestamp
            结束日期，默认到今天
        adjustflag: str
            复权类型: 1=后复权, 2=前复权(默认), 3=不复权

        Returns
        -------
        pd.DataFrame
            包含完整K线和估值指标的 DataFrame
        """
        bs_symbol = normalize_bs_symbol(symbol)

        # 如果没有指定开始日期，获取上市日期
        if start_datetime is None:
            ipo_date = BaostockFullKLineCollector.get_stock_ipo_date(symbol)
            if ipo_date:
                start_datetime = pd.Timestamp(ipo_date)
            else:
                start_datetime = pd.Timestamp("1990-01-01")

        if end_datetime is None:
            end_datetime = pd.Timestamp.today()

        rs = bs.query_history_k_data_plus(
            bs_symbol,
            BaostockFullKLineCollector.FULL_KLINE_FIELDS,
            start_date=str(start_datetime.strftime("%Y-%m-%d")),
            end_date=str(end_datetime.strftime("%Y-%m-%d")),
            frequency="d",
            adjustflag=adjustflag,
        )

        if rs.error_code == "0" and rs.data:
            df = pd.DataFrame(rs.data, columns=rs.fields)
            return df
        return pd.DataFrame()

    def collector_full_kline(self):
        """采集完整K线数据"""
        if not self.symbols:
            logger.warning("no symbols specified, please provide symbols list")
            return

        logger.info(f"start collector full kline data for {len(self.symbols)} symbols......")

        success_count = 0
        for symbol in tqdm(self.symbols, desc="采集完整K线"):
            try:
                df = self.get_full_kline_from_remote(
                    symbol=symbol,
                    start_datetime=self.start_datetime,
                    end_datetime=self.end_datetime,
                )
                if not df.empty:
                    filename = f"{self._normalize_symbol(symbol)}.csv"
                    filepath = self.save_dir / filename
                    df.to_csv(filepath, index=False)
                    success_count += 1
                self.sleep()
            except Exception as e:
                logger.warning(f"get {symbol} full kline error: {e}")

        logger.info(f"total {len(self.symbols)}, success: {success_count}")


class BaostockETFCollector(BaostockFullKLineCollector):
    """BaoStock ETF 数据采集器，继承自 BaostockFullKLineCollector
    
    支持从 funds_list.csv 批量下载 ETF K线数据
    ETF 不支持估值指标（peTTM, pbMRQ 等）
    """

    # ETF K线字段（不含估值指标）
    ETF_KLINE_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,pctChg"

    def __init__(
        self,
        save_dir: [str, Path],
        funds_list_path: str = None,
        symbols: List[str] = None,
        start=None,
        end=None,
        delay=0.5,
        adjustflag: str = "1",
    ):
        """
        Parameters
        ----------
        save_dir: str
            保存目录
        funds_list_path: str
            funds_list.csv 文件路径
        symbols: list
            ETF 代码列表（可选，与 funds_list_path 二选一）
        start: str
            开始日期，默认从上市日期开始
        end: str
            结束日期，默认今天
        delay: float
            请求间隔，默认 0.5 秒
        adjustflag: str
            复权类型: 1=后复权(默认), 2=前复权, 3=不复权
        """
        super().__init__(save_dir=save_dir, symbols=symbols, start=start, end=end, delay=delay)
        self.funds_list_path = funds_list_path
        self.adjustflag = str(adjustflag)  # 确保是字符串

    @staticmethod
    def normalize_etf_symbol(symbol: str) -> str:
        """ETF 代码转换为 BaoStock 格式

        上交所 ETF: 51xxxx, 58xxxx -> sh.
        深交所 ETF: 15xxxx, 56xxxx -> sz.

        Parameters
        ----------
        symbol: str
            ETF 代码，如 518880, 159915

        Returns
        -------
        str
            BaoStock 格式的代码，如 sh.518880；不支持的类型返回 None
        """
        symbol = str(symbol).strip().zfill(6)
        if symbol.startswith("51") or symbol.startswith("58"):
            return f"sh.{symbol}"
        if symbol.startswith("15") or symbol.startswith("56"):
            return f"sz.{symbol}"
        return None

    @staticmethod
    def get_etf_kline_from_remote(
        symbol: str,
        start_datetime: pd.Timestamp = None,
        end_datetime: pd.Timestamp = None,
        adjustflag: str = "1",
    ) -> pd.DataFrame:
        """从 BaoStock 获取 ETF K线数据

        Parameters
        ----------
        symbol: str
            ETF 代码，如 518880 或 sh.518880
        start_datetime: pd.Timestamp
            开始日期，默认从 2010-01-01 开始
        end_datetime: pd.Timestamp
            结束日期，默认今天
        adjustflag: str
            复权类型: 1=后复权(默认), 2=前复权, 3=不复权

        Returns
        -------
        pd.DataFrame
            ETF K线数据
        """
        bs_symbol = BaostockETFCollector.normalize_etf_symbol(symbol)
        if not bs_symbol:
            logger.warning(f"Unsupported ETF symbol: {symbol}")
            return pd.DataFrame()

        # ETF 不使用 query_stock_basic 获取上市日期，使用足够早的默认起始日期以覆盖所有 ETF 历史
        if start_datetime is None:
            start_datetime = pd.Timestamp("200-01-01")
        if end_datetime is None:
            end_datetime = pd.Timestamp.today()

        rs = bs.query_history_k_data_plus(
            bs_symbol,
            BaostockETFCollector.ETF_KLINE_FIELDS,
            start_date=start_datetime.strftime("%Y-%m-%d"),
            end_date=end_datetime.strftime("%Y-%m-%d"),
            frequency="d",
            adjustflag=adjustflag,
        )
        if rs.error_code == "0" and rs.data:
            return pd.DataFrame(rs.data, columns=rs.fields)
        return pd.DataFrame()

    def load_etf_list(self) -> List[str]:
        """从 funds_list.csv 加载并过滤 BaoStock 支持的 ETF

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

        df = pd.read_csv(self.funds_list_path, dtype=str, comment='#')
        df = df.dropna(subset=["fund_code"])

        etf_list = []
        skipped = []
        for _, row in df.iterrows():
            code = str(row["fund_code"]).strip()
            if self.normalize_etf_symbol(code):
                etf_list.append(code)
            else:
                skipped.append(f"{code}({row.get('fund_name', '')})")

        if skipped:
            logger.warning(f"Skipped {len(skipped)} non-ETF funds (not supported by BaoStock): {skipped}")
        logger.info(f"Loaded {len(etf_list)} ETF symbols from {self.funds_list_path}")
        return etf_list

    def collector_etf_kline(self):
        """采集 ETF K线数据"""
        symbols = self.load_etf_list()
        if not symbols:
            logger.warning("No ETF symbols to download")
            return

        logger.info(f"Start collecting ETF kline for {len(symbols)} symbols (adjustflag={self.adjustflag})...")
        success_count = 0
        failed_list = []

        for symbol in tqdm(symbols, desc="采集ETF K线"):
            try:
                df = self.get_etf_kline_from_remote(
                    symbol=symbol,
                    start_datetime=self.start_datetime,
                    end_datetime=self.end_datetime,
                    adjustflag=self.adjustflag,
                )
                if not df.empty:
                    filename = f"{self._normalize_symbol(symbol)}.csv"
                    filepath = self.save_dir / filename
                    df.to_csv(filepath, index=False)
                    success_count += 1
                    logger.debug(f"Saved {symbol} -> {filepath}, rows={len(df)}")
                else:
                    failed_list.append(symbol)
                self.sleep()
            except Exception as e:
                logger.warning(f"get {symbol} ETF kline error: {e}")
                failed_list.append(symbol)

        logger.info(f"Total {len(symbols)}, success: {success_count}, failed: {len(failed_list)}")
        if failed_list:
            logger.warning(f"Failed symbols: {failed_list}")


class BaostockNormalizeCN1d(BaseNormalize):
    """Baostock 日线数据标准化"""

    COLUMNS = ["open", "close", "high", "low", "volume"]
    DAILY_FORMAT = "%Y-%m-%d"

    @staticmethod
    def calc_change(df: pd.DataFrame, last_close: float) -> pd.Series:
        df = df.copy()
        _tmp_series = df["close"].ffill()
        _tmp_shift_series = _tmp_series.shift(1)
        if last_close is not None:
            _tmp_shift_series.iloc[0] = float(last_close)
        change_series = _tmp_series / _tmp_shift_series - 1
        return change_series

    def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
        from data_collector.utils import get_calendar_list

        return get_calendar_list("ALL")

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        symbol = df["code"].iloc[0] if "code" in df.columns else df["symbol"].iloc[0]
        symbol = symbol.replace(".", "").upper()

        # 选择需要的列
        cols = ["date", "open", "high", "low", "close", "volume", "amount", "pctChg"]
        available_cols = [c for c in cols if c in df.columns]
        df = df[available_cols].copy()

        # 重命名
        if "code" in df.columns:
            df = df.rename(columns={"code": "symbol"})
        if "symbol" not in df.columns:
            df["symbol"] = symbol

        # 转换日期格式
        df["date"] = pd.to_datetime(df["date"])

        # 过滤停牌日
        df = df[df["volume"].astype(str).str.strip() != ""]
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df[df["volume"] > 0]

        # 计算 change
        df["change"] = self.calc_change(df, None)

        # 移除停牌日
        df.loc[(df["volume"] <= 0) | df["volume"].isna(), self.COLUMNS] = np.nan
        df["change"] = self.calc_change(df, None)

        # 调整因子（使用前复权数据）
        df["factor"] = 1.0

        # 按日期排序
        df = df.sort_values("date").reset_index(drop=True)

        return df


class Run(BaseRun):
    """运行入口"""

    def __init__(self, source_dir=None, normalize_dir=None, max_workers=4, interval="1d", region="CN"):
        super().__init__(source_dir, normalize_dir, max_workers, interval)
        self.region = region

    @property
    def collector_class_name(self):
        return f"BaostockFullKLineCollector"

    @property
    def normalize_class_name(self):
        return f"BaostockNormalizeCN1d"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_full_kline(
        self,
        symbols: str = None,
        start=None,
        end=None,
        delay=0.5,
        adjustflag: str = "2",
    ):
        """下载完整K线数据（含K线 + 估值指标）

        支持指定单个或多个股票代码，自动从上市日期开始获取数据

        Parameters
        ----------
        symbols: str
            股票代码，多个用逗号分隔，如 '600519' 或 '600519,600000'
        start: str
            开始日期，默认从上市日期开始
        end: str
            结束日期，默认今天
        delay: float
            请求间隔，默认 0.5 秒
        adjustflag: str
            复权类型: 1=后复权, 2=前复权(默认), 3=不复权

        Examples
        ---------
            # 下载贵州茅台完整历史数据
            $ python collector.py download_full_kline \\
                --symbols "600519" \\
                --source_dir ~/.qlib/stock_data/source/cn_full_kline \\
                --end 2026-04-11

            # 下载多只股票
            $ python collector.py download_full_kline \\
                --symbols "600519,600000,000001" \\
                --source_dir ~/.qlib/stock_data/source/cn_full_kline

            # Python 调用
            $ python -c "
from scripts.data_collector.baostock_daily.collector import BaostockFullKLineCollector
df = BaostockFullKLineCollector.get_full_kline_from_remote('600519')
print(df.head())
"
        """
        if not symbols:
            logger.warning("please specify symbols, e.g., --symbols '600519' or --symbols '600519,600000'")
            return

        # fire 可能将参数解析为 int，需要转换为字符串
        if isinstance(symbols, int):
            symbols = str(symbols)

        # 解析股票代码列表
        symbol_list = [s.strip() for s in str(symbols).split(",")]

        collector = BaostockFullKLineCollector(
            save_dir=self.source_dir,
            symbols=symbol_list,
            start=start,
            end=end,
            delay=delay,
        )
        collector.collector_full_kline()

    def normalize_daily_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
    ):
        """标准化日线数据

        Examples
        ---------
            $ python collector.py normalize_daily_data \\
                --source_dir ~/.qlib/stock_data/source/cn_full_kline \\
                --normalize_dir ~/.qlib/stock_data/source/cn_full_kline_nor
        """
        from data_collector.base import Normalize

        _class = BaostockNormalizeCN1d
        yc = Normalize(
            source_dir=self.source_dir,
            target_dir=self.normalize_dir,
            normalize_class=_class,
            max_workers=self.max_workers,
            date_field_name=date_field_name,
            symbol_field_name=symbol_field_name,
            end_date=end_date,
        )
        yc.normalize()

    def download_etf_kline(
        self,
        funds_list: str = None,
        symbols: str = None,
        start: str = None,
        end: str = None,
        delay: float = 0.5,
        adjustflag: str = "1",
    ):
        """下载 ETF K线数据

        从 funds_list.csv 读取 ETF 列表，或指定单个/多个 ETF 代码
        支持后复权、前复权、不复权

        Parameters
        ----------
        funds_list: str
            funds_list.csv 文件路径
        symbols: str
            ETF 代码，多个用逗号分隔，如 '518880' 或 '518880,159915'
        start: str
            开始日期，默认从上市日期开始
        end: str
            结束日期，默认今天
        delay: float
            请求间隔，默认 0.5 秒
        adjustflag: str
            复权类型: 1=后复权(默认), 2=前复权, 3=不复权

        Examples
        ---------
            # 从 funds_list.csv 下载所有 ETF
            $ python collector.py download_etf_kline \\
                --funds_list /path/to/funds_list.csv \\
                --source_dir ~/.qlib/stock_data/source/cn_etf \\
                --adjustflag 1

            # 下载指定 ETF
            $ python collector.py download_etf_kline \\
                --symbols "518880,159915" \\
                --source_dir ~/.qlib/stock_data/source/cn_etf
        """
        if not funds_list and not symbols:
            logger.warning("Please specify --funds_list or --symbols")
            return

        symbol_list = None
        if symbols:
            if isinstance(symbols, int):
                symbols = str(symbols)
            symbol_list = [s.strip() for s in str(symbols).split(",")]

        collector = BaostockETFCollector(
            save_dir=self.source_dir,
            funds_list_path=funds_list,
            symbols=symbol_list,
            start=start,
            end=end,
            delay=delay,
            adjustflag=adjustflag,
        )
        collector.collector_etf_kline()


if __name__ == "__main__":
    fire.Fire(Run)

"""
conda activate rdagent
python scripts/data_collector/baostock_daily/collector.py
"""

"""
数据源	状态	原因
BaoStock	⚠️	ETF仅返回近3个月数据
东方财富/AkShare/efinance	❌	push2his.eastmoney.com 连接被断开
新浪财经	❌无复权	✅ 5138条	仅不复权
腾讯财经	✅ 后复权	✅ 5136条	推荐
网易财经	❌	502 错误 已确认
Tushare	❌	与新pandas不兼容
天天基金: 已知有分红数据可以下载
yfinance	❌	未安装
"""