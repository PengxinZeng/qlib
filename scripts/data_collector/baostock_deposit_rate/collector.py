# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Baostock 存款利率数据采集器

使用 baostock 的 query_deposit_rate_data 接口获取存款利率数据

使用方法:
    # 下载存款利率数据
    python collector.py download_deposit_rate \
        --save_dir /Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_test11

    # 指定日期范围
    python collector.py download_deposit_rate \
        --save_dir /Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_test11 \
        --start 2010-01-01 \
        --end 2026-04-15
"""

import fire
import pandas as pd
import baostock as bs
from pathlib import Path
from loguru import logger


class BaostockDepositRateCollector:
    """Baostock 存款利率数据采集器

    使用 baostock 的 query_deposit_rate_data 接口获取中国银行存款利率数据

    数据字段:
        pubDate: 公布日期
        depositType: 存款类型（活期、定期整存整取3个月、6个月、1年、2年、3年、5年等）
        depositRate: 存款利率（%）
    """

    def __init__(
        self,
        save_dir: [str, Path],
        start: str = None,
        end: str = None,
    ):
        """
        Parameters
        ----------
        save_dir: str
            保存目录
        start: str
            开始日期，格式 YYYY-MM-DD，默认 None（从最早数据开始）
        end: str
            结束日期，格式 YYYY-MM-DD，默认 None（到今天）
        """
        self.save_dir = Path(save_dir).expanduser().resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.start = start
        self.end = end

    @staticmethod
    def get_deposit_rate_from_remote(
        start: str = None,
        end: str = None,
    ) -> pd.DataFrame:
        """从 Baostock 获取存款利率数据

        Parameters
        ----------
        start: str
            开始日期，格式 YYYY-MM-DD
        end: str
            结束日期，格式 YYYY-MM-DD

        Returns
        -------
        pd.DataFrame
            存款利率数据
        """
        # 登录 baostock
        lg = bs.login()
        if lg.error_code != '0':
            logger.error(f"baostock login failed: {lg.error_msg}")
            return pd.DataFrame()

        try:
            # 设置默认日期范围
            if start is None:
                start = "1990-01-01"
            if end is None:
                end = pd.Timestamp.today().strftime("%Y-%m-%d")

            logger.info(f"Fetching deposit rate data from {start} to {end}...")

            # 调用 query_deposit_rate_data 获取存款利率数据
            rs = bs.query_deposit_rate_data(start_date=start, end_date=end)

            if rs.error_code != '0':
                logger.error(f"query_deposit_rate_data failed: {rs.error_msg}")
                return pd.DataFrame()

            # 收集数据
            data_list = []
            while rs.error_code == '0' and rs.next():
                data_list.append(rs.get_row_data())

            if not data_list:
                logger.warning("No deposit rate data returned")
                return pd.DataFrame()

            # 创建 DataFrame
            df = pd.DataFrame(data_list, columns=rs.fields)
            logger.info(f"Fetched {len(df)} deposit rate records")

            return df

        finally:
            # 登出
            bs.logout()

    def collect_deposit_rate(self):
        """采集存款利率数据"""
        logger.info("Start collecting deposit rate data...")

        df = self.get_deposit_rate_from_remote(
            start=self.start,
            end=self.end,
        )

        if df.empty:
            logger.warning("No deposit rate data collected")
            return

        # 保存数据
        filename = "deposit_rate.csv"
        filepath = self.save_dir / filename
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        logger.info(f"Saved deposit rate data to {filepath}, total {len(df)} records")

        # 打印数据概览
        logger.info(f"Data columns: {list(df.columns)}")
        if 'depositType' in df.columns:
            logger.info(f"Deposit types: {df['depositType'].unique().tolist()}")
        logger.info(f"Data preview:\n{df.head(10)}")

        return df


class Run:
    """运行入口"""

    def __init__(self, save_dir: str = None):
        """
        Parameters
        ----------
        save_dir: str
            数据保存目录
        """
        self.save_dir = save_dir or "/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_test11"

    def download_deposit_rate(
        self,
        start: str = None,
        end: str = None,
    ):
        """下载存款利率数据

        Parameters
        ----------
        start: str
            开始日期，格式 YYYY-MM-DD，默认从最早数据开始
        end: str
            结束日期，格式 YYYY-MM-DD，默认到今天

        Examples
        ---------
            # 下载所有存款利率数据
            $ python collector.py download_deposit_rate \\
                --save_dir /path/to/save

            # 指定日期范围
            $ python collector.py download_deposit_rate \\
                --save_dir /path/to/save \\
                --start 2015-01-01 \\
                --end 2026-04-15
        """
        collector = BaostockDepositRateCollector(
            save_dir=self.save_dir,
            start=start,
            end=end,
        )
        collector.collect_deposit_rate()


if __name__ == "__main__":
    fire.Fire(Run)
