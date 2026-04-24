"""
PB/PE 价值投资策略实现
基于低估值ETF的轮动策略
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from qlib.contrib.strategy import Strategy


class PBPEValueStrategy(Strategy):
    """
    PB/PE价值投资策略

    选基逻辑:
    1. 估值过滤: PE和PB必须在合理范围内
    2. 相对估值: 相对于市场中位数
    3. 动量确认: 趋势向上
    4. 综合评分: 低估值 + 动量
    """

    def __init__(
        self,
        pe_max: float = 50.0,
        pe_min: float = 5.0,
        pb_max: float = 5.0,
        pb_min: float = 0.5,
        pe_ratio_threshold: float = 1.5,
        pb_ratio_threshold: float = 1.5,
        topk: int = 5,
        momentum_period: int = 20,
        hold_period: int = 5,
    ):
        """
        初始化策略参数

        Args:
            pe_max: PE最大值阈值
            pe_min: PE最小值阈值
            pb_max: PB最大值阈值
            pb_min: PB最小值阈值
            pe_ratio_threshold: PE相对中位数的最大比例
            pb_ratio_threshold: PB相对中位数的最大比例
            topk: 持仓基金数量
            momentum_period: 动量计算周期
            hold_period: 持仓周期(交易日)
        """
        self.pe_max = pe_max
        self.pe_min = pe_min
        self.pb_max = pb_max
        self.pb_min = pb_min
        self.pe_ratio_threshold = pe_ratio_threshold
        self.pb_ratio_threshold = pb_ratio_threshold
        self.topk = topk
        self.momentum_period = momentum_period
        self.hold_period = hold_period

        self.last_trade_date = None
        self.current_positions = {}

    def get_valuation_filter(
        self,
        pe_ttm: pd.Series,
        pb: pd.Series,
        pe_ttm_median: pd.Series,
        pb_median: pd.Series,
    ) -> pd.Series:
        """
        估值过滤

        Args:
            pe_ttm: 市盈率TTM序列
            pb: 市净率序列
            pe_ttm_median: 市盈率中位数序列
            pb_median: 市净率中位数序列

        Returns:
            满足条件的mask
        """
        # 绝对估值过滤
        pe_valid = (pe_ttm >= self.pe_min) & (pe_ttm <= self.pe_max)
        pb_valid = (pb >= self.pb_min) & (pb <= self.pb_max)

        # 相对估值过滤
        pe_ratio_valid = pe_ttm <= pe_ttm_median * self.pe_ratio_threshold
        pb_ratio_valid = pb <= pb_median * self.pb_ratio_threshold

        # 综合过滤
        valid = pe_valid & pb_valid & pe_ratio_valid & pb_ratio_valid
        return valid

    def get_momentum_signal(
        self,
        close: pd.DataFrame,
        period: int = 20,
    ) -> pd.DataFrame:
        """
        动量信号计算

        Args:
            close: 收盘价DataFrame, index为日期, columns为基金代码
            period: 计算周期

        Returns:
            动量信号DataFrame
        """
        # 计算收益率
        momentum = close.pct_change(period)
        return momentum

    def get_value_score(
        self,
        pe_ttm: pd.DataFrame,
        pb: pd.DataFrame,
        pe_ttm_median: pd.DataFrame,
        pb_median: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        价值评分

        低估值得高分:
        - PE越低越好
        - PB越低越好

        Args:
            pe_ttm: 市盈率TTM
            pb: 市净率
            pe_ttm_median: 市盈率中位数
            pb_median: 市净率中位数

        Returns:
            价值评分DataFrame
        """
        # PE评分: 相对于中位数的偏离
        # 值越小(低估)分数越高
        pe_score = -pe_ttm / pe_ttm_median

        # PB评分: 相对于中位数的偏离
        # 值越小(低估)分数越高
        pb_score = -pb / pb_median

        # 综合评分
        score = pe_score + pb_score
        return score

    def get_signal(
        self,
        dataset,
        current_date: str,
    ) -> pd.Series:
        """
        获取交易信号

        Args:
            dataset: Qlib数据集
            current_date: 当前日期

        Returns:
            信号Series, index为基金代码, 值为综合评分
        """
        # 获取数据
        close = dataset.get_df("close", current_date)
        pe_ttm = dataset.get_df("pe_ttm", current_date)
        pb = dataset.get_df("pb", current_date)
        pe_ttm_median = dataset.get_df("pe_ttm_median", current_date)
        pb_median = dataset.get_df("pb_median", current_date)

        if close is None or pe_ttm is None:
            return pd.Series(dtype=float)

        # 估值过滤
        valuation_valid = self.get_valuation_filter(
            pe_ttm, pb, pe_ttm_median, pb_median
        )

        # 动量信号
        momentum = self.get_momentum_signal(close, self.momentum_period)

        # 价值评分
        value_score = self.get_value_score(
            pe_ttm, pb, pe_ttm_median, pb_median
        )

        # 综合评分: 价值评分 + 动量信号 * 权重
        final_score = value_score.fillna(0)
        final_score += momentum.fillna(0) * 0.3

        # 应用估值过滤
        final_score[~valuation_valid] = np.nan

        return final_score

    def generate_trade(self, dataset, current_date: str) -> List[Dict]:
        """
        生成交易指令

        Args:
            dataset: Qlib数据集
            current_date: 当前日期

        Returns:
            交易指令列表
        """
        # 获取信号
        signal = self.get_signal(dataset, current_date)

        # 获取评分最高的基金
        top_funds = signal.dropna().nlargest(self.topk)

        trades = []
        for fund_code, score in top_funds.items():
            trades.append({
                "instrument": fund_code,
                "score": score,
                "direction": 1,  # 买入
            })

        return trades


class PBPEIndexStrategy(PBPEValueStrategy):
    """
    基于指数估值的PB/PE策略

    使用指数的PE/PB而非基金的PE/PB
    适用于指数型ETF
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_signal(self, dataset, current_date: str) -> pd.Series:
        """获取基于指数估值的信号"""
        # 获取指数数据
        index_close = dataset.get_df("index_close", current_date)
        pe_ttm = dataset.get_df("pe_ttm", current_date)
        pb = dataset.get_df("pb", current_date)

        if index_close is None:
            return pd.Series(dtype=float)

        # 动量信号
        momentum = self.get_momentum_signal(index_close, self.momentum_period)

        # 价值评分 (使用指数估值)
        value_score = -pe_ttm.fillna(0) - pb.fillna(0) * 2

        # 综合评分
        final_score = value_score.fillna(0)
        final_score += momentum.fillna(0) * 0.3

        return final_score
