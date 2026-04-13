# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Baostock 行业数据采集器

支持获取A股行业分类信息和各行业股票列表

使用方法:
    # 获取所有行业信息
    python collector.py get_industry_list

    # 获取行业详情
    python collector.py get_industry_detail

    # 获取各行业龙头股
    python collector.py get_industry_leaders
"""

import sys
import fire
import pandas as pd
import baostock as bs
from pathlib import Path
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))


# 各行业龙头股定义
INDUSTRY_LEADERS = {
    "I65软件和信息技术服务业": [
        {"code": "688111", "name": "金山办公"},
        {"code": "600588", "name": "用友网络"},
        {"code": "002230", "name": "科大讯飞"},
    ],
    "C39计算机、通信和其他电子设备制造业": [
        {"code": "601138", "name": "工业富联"},
        {"code": "002415", "name": "海康威视"},
        {"code": "000725", "name": "京东方A"},
    ],
    "C27医药制造业": [
        {"code": "600276", "name": "恒瑞医药"},
        {"code": "603259", "name": "药明康德"},
        {"code": "300760", "name": "迈瑞医疗"},
    ],
    "C35专用设备制造业": [
        {"code": "300750", "name": "宁德时代"},
        {"code": "600031", "name": "三一重工"},
        {"code": "688012", "name": "中微公司"},
    ],
    "C26化学原料和化学制品制造业": [
        {"code": "600309", "name": "万华化学"},
        {"code": "000792", "name": "盐湖股份"},
        {"code": "600486", "name": "扬农化工"},
    ],
    "C38电气机械和器材制造业": [
        {"code": "601012", "name": "隆基绿能"},
        {"code": "000333", "name": "美的集团"},
        {"code": "002129", "name": "TCL中环"},
    ],
    "C36汽车制造业": [
        {"code": "002594", "name": "比亚迪"},
        {"code": "601633", "name": "长城汽车"},
        {"code": "600104", "name": "上汽集团"},
    ],
    "I64互联网和相关服务": [
        {"code": "300059", "name": "东方财富"},
        {"code": "300124", "name": "汇川技术"},
        {"code": "002024", "name": "苏宁易购"},
    ],
    "K70房地产业": [
        {"code": "000002", "name": "万科A"},
        {"code": "600048", "name": "保利发展"},
        {"code": "001979", "name": "招商蛇口"},
    ],
    "J66货币金融服务": [
        {"code": "601398", "name": "工商银行"},
        {"code": "600036", "name": "招商银行"},
        {"code": "601288", "name": "农业银行"},
    ],
    "J67资本市场服务": [
        {"code": "600030", "name": "中信证券"},
        {"code": "300059", "name": "东方财富"},
        {"code": "601211", "name": "国泰君安"},
    ],
    "C15酒、饮料和精制茶制造业": [
        {"code": "600519", "name": "贵州茅台"},
        {"code": "000858", "name": "五粮液"},
        {"code": "000568", "name": "泸州老窖"},
    ],
    "D44电力、热力生产和供应业": [
        {"code": "600900", "name": "长江电力"},
        {"code": "601985", "name": "中国核电"},
        {"code": "600025", "name": "华能水电"},
    ],
    "F51批发业": [
        {"code": "601888", "name": "中国中免"},
        {"code": "601607", "name": "上海医药"},
        {"code": "600827", "name": "百联股份"},
    ],
    "F52零售业": [
        {"code": "601933", "name": "永辉超市"},
        {"code": "002024", "name": "苏宁易购"},
        {"code": "601116", "name": "三江购物"},
    ],
    "C13农副食品加工业": [
        {"code": "600887", "name": "伊利股份"},
        {"code": "603288", "name": "海天味业"},
        {"code": "002311", "name": "海大集团"},
    ],
    "N77生态保护和环境治理业": [
        {"code": "300070", "name": "碧水源"},
        {"code": "600323", "name": "瀚蓝环境"},
        {"code": "002672", "name": "东江环保"},
    ],
    "C30非金属矿物制品业": [
        {"code": "600585", "name": "海螺水泥"},
        {"code": "002271", "name": "东方雨虹"},
        {"code": "000877", "name": "天山股份"},
    ],
}


class BaostockIndustryCollector:
    """Baostock 行业数据采集器"""

    @staticmethod
    def get_industry_list() -> pd.DataFrame:
        """获取所有行业分类信息

        Returns
        -------
        pd.DataFrame
            包含 code, code_name, industry, list_date 字段
        """
        logger.info("get industry list from Baostock......")
        bs.login()

        rs = bs.query_stock_industry()
        if rs.error_code != "0":
            logger.error(f"query failed: {rs.error_msg}")
            bs.logout()
            return pd.DataFrame()

        data_list = []
        while rs.error_code == "0" and rs.next():
            data_list.append(rs.get_row_data())

        df = pd.DataFrame(data_list, columns=rs.fields)
        bs.logout()

        logger.info(f"total {len(df)} stocks, {df['industry'].nunique()} industries")
        return df

    @staticmethod
    def get_industry_summary(df: pd.DataFrame = None) -> pd.DataFrame:
        """获取行业统计摘要

        Parameters
        ----------
        df : pd.DataFrame, optional
            行业数据，如果为 None 则自动获取

        Returns
        -------
        pd.DataFrame
            各行业的股票数量统计
        """
        if df is None:
            df = BaostockIndustryCollector.get_industry_list()

        summary = df.groupby("industry").agg(
            stock_count=("code", "count"),
            stocks=("code", lambda x: ",".join(x[:5]))  # 只显示前5个
        ).reset_index()

        summary = summary.sort_values("stock_count", ascending=False)
        summary.columns = ["行业名称", "股票数量", "代表股票"]
        return summary

    @staticmethod
    def get_industry_leaders_df() -> pd.DataFrame:
        """获取各行业龙头股信息

        Returns
        -------
        pd.DataFrame
            包含 industry, code, name 字段
        """
        data_list = []
        for industry, stocks in INDUSTRY_LEADERS.items():
            for stock in stocks:
                data_list.append({
                    "industry": industry,
                    "code": stock["code"],
                    "name": stock["name"],
                })
        return pd.DataFrame(data_list)


class Run:
    """运行入口"""

    def __init__(self, save_dir: str = None):
        self.save_dir = Path(save_dir) if save_dir else CUR_DIR

    def get_industry_list(self, save: bool = True):
        """获取所有行业信息

        Parameters
        ----------
        save : bool
            是否保存到 CSV 文件，默认 True

        Examples
        --------
            $ python collector.py get_industry_list
        """
        df = BaostockIndustryCollector.get_industry_list()

        if df.empty:
            logger.warning("no data retrieved")
            return

        print(f"\n总股票数: {len(df)}")
        print(f"行业数: {df['industry'].nunique()}")

        # 显示前30个行业
        summary = BaostockIndustryCollector.get_industry_summary(df)
        print("\n=== 各行业股票数量 ===")
        print(summary.head(30).to_string(index=False))

        if save:
            save_path = self.save_dir / "industry_list.csv"
            df.to_csv(save_path, index=False, encoding="utf-8-sig")
            logger.info(f"saved to {save_path}")

    def get_industry_detail(self, industry: str = None, save: bool = True):
        """获取行业详情

        Parameters
        ----------
        industry : str, optional
            行业名称，如不指定则显示所有
        save : bool
            是否保存到 CSV 文件，默认 True

        Examples
        --------
            # 获取所有行业详情
            $ python collector.py get_industry_detail

            # 获取指定行业详情
            $ python collector.py get_industry_detail --industry "C27医药制造业"
        """
        df = BaostockIndustryCollector.get_industry_list()

        if industry:
            df = df[df["industry"] == industry]
            if df.empty:
                logger.warning(f"no stocks found for industry: {industry}")
                return

        print(f"\n共 {len(df)} 只股票")

        if industry:
            print(f"\n=== {industry} ===")
            print(df[["code", "code_name", "industry"]].to_string(index=False))

        if save:
            filename = f"industry_{industry or 'all'}.csv".replace("/", "_")
            save_path = self.save_dir / filename
            df.to_csv(save_path, index=False, encoding="utf-8-sig")
            logger.info(f"saved to {save_path}")

    def get_industry_leaders(self, save: bool = True):
        """获取各行业龙头股

        Examples
        --------
            $ python collector.py get_industry_leaders
        """
        df = BaostockIndustryCollector.get_industry_leaders_df()

        print("\n=== 各行业龙头股 ===\n")
        for industry in sorted(df["industry"].unique()):
            stocks = df[df["industry"] == industry]
            print(f"{industry}:")
            for _, row in stocks.iterrows():
                print(f"  {row['code']} {row['name']}")
            print()

        if save:
            save_path = self.save_dir / "industry_leaders.csv"
            df.to_csv(save_path, index=False, encoding="utf-8-sig")
            logger.info(f"saved to {save_path}")


if __name__ == "__main__":
    fire.Fire(Run)
