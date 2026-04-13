# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Baostock 指数日线数据采集器

支持获取A股指数K线数据

使用方法:
    # 下载所有预设指数数据
    python collector.py download_index \
        --source_dir ~/.qlib/index_data/source \
        --end 2026-04-11

    # 下载指定指数
    python collector.py download_index \
        --symbols "sh.000300,sh.000016" \
        --source_dir ~/.qlib/index_data/source \
        --start 2015-01-01 \
        --end 2026-04-11

    # 下载规模指数
    python collector.py download_size_index \
        --source_dir ~/.qlib/index_data/source

    # 标准化数据
    python collector.py normalize_daily_data \
        --source_dir ~/.qlib/index_data/source \
        --normalize_dir ~/.qlib/index_data/normalize
"""

import sys
import time
import fire
import numpy as np
import pandas as pd
import baostock as bs
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import List

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

try:
    from data_collector.base import BaseNormalize, BaseRun
except ImportError:
    # 当qlib未安装时，提供基础实现
    import abc
    import time as time_module
    from pathlib import Path
    from typing import Iterable, List
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from loguru import logger

    class BaseNormalize(abc.ABC):
        def __init__(self, date_field_name: str = "date", symbol_field_name: str = "symbol", **kwargs):
            self._date_field_name = date_field_name
            self._symbol_field_name = symbol_field_name
            self.kwargs = kwargs
            self._calendar_list = self._get_calendar_list()

        @abc.abstractmethod
        def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
            raise NotImplementedError("")

        @abc.abstractmethod
        def _get_calendar_list(self) -> Iterable[pd.Timestamp]:
            raise NotImplementedError("")

    class BaseRun(abc.ABC):
        def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d"):
            from pathlib import Path
            if source_dir is None:
                source_dir = CUR_DIR / "source"
            self.source_dir = Path(source_dir).expanduser().resolve()
            self.source_dir.mkdir(parents=True, exist_ok=True)

            if normalize_dir is None:
                normalize_dir = CUR_DIR / "normalize"
            self.normalize_dir = Path(normalize_dir).expanduser().resolve()
            self.normalize_dir.mkdir(parents=True, exist_ok=True)

            self.max_workers = max_workers
            self.interval = interval

        @property
        @abc.abstractmethod
        def collector_class_name(self):
            raise NotImplementedError("rewrite collector_class_name")

        @property
        @abc.abstractmethod
        def normalize_class_name(self):
            raise NotImplementedError("rewrite normalize_class_name")

        @property
        @abc.abstractmethod
        def default_base_dir(self):
            raise NotImplementedError("rewrite default_base_dir")


# ============ 指数代码定义 ============

# 综合指数
INDEX_COMPREHENSIVE = [
    "sh.000001",  # 上证综指
    "sz.399106",  # 深证综指
]

# 规模指数
INDEX_SIZE = [
    "sh.000016",  # 上证50
    "sh.000300",  # 沪深300
    "sh.000905",  # 中证500
    "sz.399001",  # 深证成指
    "sh.000906",  # 中证800
    "sh.000907",  # 中证1000
]

# 一级行业指数
INDEX_INDUSTRY_L1 = [
    "sh.000037",  # 上证医药
    "sh.000038",  # 上证金融
    "sh.000039",  # 上证消费
    "sh.000040",  # 上证工业
    "sh.000041",  # 上证资源
    "sh.000042",  # 上证公用
    "sz.399433",  # 国证交运
    "sz.399434",  # 国证能源
    "sz.399436",  # 国证金融
]

# 二级行业指数
INDEX_INDUSTRY_L2 = [
    "sh.000952",  # 300地产
    "sz.399951",  # 300银行
    "sh.000913",  # 300医药
    "sh.000914",  # 300消费
    "sh.000915",  # 300可选
    "sh.000916",  # 300电信
    "sh.000917",  # 300信息
]

# 策略指数
INDEX_STRATEGY = [
    "sh.000050",  # 50等权
    "sh.000982",  # 500等权
    "sh.000926",  # 50低波
    "sh.000923",  # 500波动
    "sh.000921",  # 300红利
    "sh.000922",  # 500红利
]

# 成长指数
INDEX_GROWTH = [
    "sz.399376",  # 小盘成长
    "sz.399377",  # 大盘成长
    "sz.399378",  # 中盘成长
]

# 价值指数
INDEX_VALUE = [
    "sh.000029",  # 180价值
    "sz.399370",  # 300价值
    "sh.000030",  # 180成长
]

# 主题指数
INDEX_THEME = [
    "sh.000015",  # 红利指数
    "sh.000063",  # 上证周期
    "sh.000065",  # 上证非周
    "sh.000066",  # 上证新兴
    "sh.000068",  # 上证资源
]

# 基金指数
INDEX_FUND = [
    "sh.000011",  # 上证基金指数
    "sz.399305",  # 深证基金指数
]

# 债券指数
INDEX_BOND = [
    "sh.000012",  # 上证国债指数
    "sh.000013",  # 上证企债指数
    "sh.000014",  # 上证城投债指数
    "sh.000015",  # 上证沪深深信用债指数
]

# 所有预设指数
ALL_INDEX_CODES = (
    INDEX_COMPREHENSIVE
    + INDEX_SIZE
    + INDEX_INDUSTRY_L1
    + INDEX_INDUSTRY_L2
    + INDEX_STRATEGY
    + INDEX_GROWTH
    + INDEX_VALUE
    + INDEX_THEME
    + INDEX_FUND
    + INDEX_BOND
)


def normalize_bs_symbol(symbol: str) -> str:
    """标准化指数代码为 Baostock 格式

    Parameters
    ----------
    symbol: str
        指数代码，如 sh.000001, 000001, SH000001 等

    Returns
    -------
    str
        Baostock 格式的指数代码，如 sh.000001
    """
    symbol = symbol.strip()
    if symbol.startswith("sh.") or symbol.startswith("sz."):
        return symbol
    if symbol.upper().startswith("SH"):
        return "sh." + symbol[2:]
    if symbol.upper().startswith("SZ"):
        return "sz." + symbol[2:]
    # 6位数字代码
    if len(symbol) == 6 and symbol.isdigit():
        if symbol.startswith("0") or symbol.startswith("3"):
            return f"sz.{symbol}"
        return f"sh.{symbol}"
    return symbol


class BaostockIndexCollector:
    """Baostock 指数K线数据采集器

    获取A股指数K线数据
    支持指定单个或多个指数代码

    数据字段:
        date, code, open, high, low, close, preclose, volume, amount, pctChg
    """

    # 指数字段
    INDEX_FIELDS = "date,code,open,high,low,close,preclose,volume,amount,pctChg"

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
            保存目录
        symbols: list
            指数代码列表，如 ['sh.000001', '000300'] 或 ['000300']
        start: str
            开始日期，默认 None (从最早可取日期开始)
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

    def _normalize_symbol(self, symbol: str) -> str:
        """标准化指数代码"""
        bs_symbol = normalize_bs_symbol(symbol)
        # 转换为文件名格式: SH000001, SZ399001
        return bs_symbol.replace(".", "").upper()

    def sleep(self):
        time.sleep(self.delay)

    @staticmethod
    def get_index_data_from_remote(
        symbol: str,
        start_datetime: pd.Timestamp = None,
        end_datetime: pd.Timestamp = None,
    ) -> pd.DataFrame:
        """从 Baostock 获取单只指数的K线数据

        Parameters
        ----------
        symbol: str
            指数代码，如 sh.000001 或 000001
        start_datetime: pd.Timestamp
            开始日期，默认从最早可取日期开始
        end_datetime: pd.Timestamp
            结束日期，默认到今天

        Returns
        -------
        pd.DataFrame
            包含K线数据的 DataFrame
        """
        bs_symbol = normalize_bs_symbol(symbol)

        # 设置默认日期范围
        if start_datetime is None:
            start_datetime = pd.Timestamp("2005-01-01")  # 指数数据从2005年开始较多

        if end_datetime is None:
            end_datetime = pd.Timestamp.today()

        rs = bs.query_history_k_data_plus(
            bs_symbol,
            BaostockIndexCollector.INDEX_FIELDS,
            start_date=str(start_datetime.strftime("%Y-%m-%d")),
            end_date=str(end_datetime.strftime("%Y-%m-%d")),
            frequency="d",
        )

        if rs.error_code == "0" and rs.data:
            df = pd.DataFrame(rs.data, columns=rs.fields)
            return df
        return pd.DataFrame()

    def collector_index_data(self):
        """采集指数K线数据"""
        if not self.symbols:
            logger.warning("no symbols specified, please provide symbols list")
            return

        logger.info(f"start collector index data for {len(self.symbols)} symbols......")

        success_count = 0
        for symbol in tqdm(self.symbols, desc="采集指数数据"):
            try:
                df = self.get_index_data_from_remote(
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
                logger.warning(f"get {symbol} index data error: {e}")

        logger.info(f"total {len(self.symbols)}, success: {success_count}")

    def __del__(self):
        try:
            bs.logout()
        except Exception:
            pass


class BaostockIndexNormalize(BaseNormalize):
    """指数日线数据标准化"""

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

    def _get_calendar_list(self) -> List[pd.Timestamp]:
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

        # 过滤空数据日
        df = df[df["volume"].astype(str).str.strip() != ""]
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        df = df[df["volume"] > 0]

        # 计算 change
        df["change"] = self.calc_change(df, None)

        # 调整因子（指数不需要复权）
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
        return f"BaostockIndexCollector"

    @property
    def normalize_class_name(self):
        return f"BaostockIndexNormalize"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_index(
        self,
        symbols: str = None,
        start=None,
        end=None,
        delay=0.5,
    ):
        """下载指数K线数据

        支持指定单个或多个指数代码

        Parameters
        ----------
        symbols: str
            指数代码，多个用逗号分隔，如 'sh.000300' 或 'sh.000300,sh.000016'
            如果不指定，则下载所有预设指数
        start: str
            开始日期，默认从2005-01-01开始
        end: str
            结束日期，默认今天
        delay: float
            请求间隔，默认 0.5 秒

        Examples
        --------
            # 下载所有预设指数
            $ python collector.py download_index \
                --source_dir ~/.qlib/index_data/source \
                --end 2026-04-11

            # 下载指定指数
            $ python collector.py download_index \
                --symbols "sh.000300,sh.000016" \
                --source_dir ~/.qlib/index_data/source \
                --start 2015-01-01 \
                --end 2026-04-11
        """
        if symbols:
            # fire 可能将参数解析为 int，需要转换为字符串
            if isinstance(symbols, int):
                symbols = str(symbols)
            symbol_list = [s.strip() for s in str(symbols).split(",")]
        else:
            # 使用所有预设指数
            symbol_list = ALL_INDEX_CODES

        collector = BaostockIndexCollector(
            save_dir=self.source_dir,
            symbols=symbol_list,
            start=start,
            end=end,
            delay=delay,
        )
        collector.collector_index_data()

    def download_size_index(
        self,
        start=None,
        end=None,
        delay=0.5,
    ):
        """下载规模指数

        包括: 上证50、沪深300、中证500、深证成指等

        Parameters
        ----------
        start: str
            开始日期，默认从2005-01-01开始
        end: str
            结束日期，默认今天
        delay: float
            请求间隔，默认 0.5 秒

        Examples
        --------
            $ python collector.py download_size_index \
                --source_dir ~/.qlib/index_data/source
        """
        collector = BaostockIndexCollector(
            save_dir=self.source_dir,
            symbols=INDEX_SIZE,
            start=start,
            end=end,
            delay=delay,
        )
        collector.collector_index_data()

    def download_all_index(
        self,
        start=None,
        end=None,
        delay=0.5,
    ):
        """下载所有预设指数

        Examples
        --------
            $ python collector.py download_all_index \
                --source_dir ~/.qlib/index_data/source
        """
        collector = BaostockIndexCollector(
            save_dir=self.source_dir,
            symbols=ALL_INDEX_CODES,
            start=start,
            end=end,
            delay=delay,
        )
        collector.collector_index_data()

    def normalize_daily_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
    ):
        """标准化日线数据

        Examples
        --------
            $ python collector.py normalize_daily_data \
                --source_dir ~/.qlib/index_data/source \
                --normalize_dir ~/.qlib/index_data/normalize
        """
        from data_collector.base import Normalize

        _class = BaostockIndexNormalize
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


if __name__ == "__main__":
    fire.Fire(Run)
