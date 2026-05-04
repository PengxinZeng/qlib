# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
HistRelaPB Signal - 基于PB历史百分位的交易信号
"""

import numpy as np
import pandas as pd
from typing import Optional, Union

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class HistRelaPBSignal(Model):
    """
    PB历史百分位信号生成器

    每天计算每个标的的PB在其历史k天PB数据中的百分位：
    - percentile < oversold_threshold -> 信号 1 (买入)
    - percentile > overbought_threshold -> 信号 -1 (卖出)
    - 否则 -> 信号 0 (持有)
    """

    def __init__(
        self,
        lookback_days: int = 20,
        min_valid_days_ratio: float = 0.20,
        oversold_threshold: float = 0.10,
        overbought_threshold: float = 0.90,
        volume_threshold: float = 0.0,
        spread_threshold: Optional[float] = None,
        freq: str = "day",
    ):
        """
        Parameters
        ----------
        lookback_days : int
            历史回看窗口天数，用于计算PB百分位
        min_valid_days_ratio : float
            最小有效数据天数占 lookback_days 的比例，取值 (0, 1]
            实际 min_valid_days = max(1, int(lookback_days * min_valid_days_ratio))
        oversold_threshold : float
            超卖阈值，低于此百分位产生买入信号
        overbought_threshold : float
            超买阈值，高于此百分位产生卖出信号
        volume_threshold : float
            成交量阈值，低于此值的交易日数据将被忽略
        spread_threshold : float or None
            股债利差阈值（单位 %）：盈利收益率(1/PE_TTM*100) - CN 2Y 国债收益率
            仅当 spread >= spread_threshold 时才产生买入信号（score=1）
            默认 None，不启用利差过滤
        freq : str
            数据频率
        """
        self.lookback_days = int(lookback_days)
        self.min_valid_days_ratio = float(min_valid_days_ratio)
        self.min_valid_days = max(1, int(self.lookback_days * self.min_valid_days_ratio))
        self.oversold_threshold = float(oversold_threshold)
        self.overbought_threshold = float(overbought_threshold)
        self.volume_threshold = float(volume_threshold)
        self.spread_threshold = float(spread_threshold) if spread_threshold is not None else None
        self.freq = freq

    def fit(self, dataset: DatasetH, reweighter=None):
        """非深度学习模型，fit直接返回"""
        print("HistRelaPBSignal fit")
        pass

    def predict(self, dataset: DatasetH, segment: Union[str, slice] = "test") -> pd.DataFrame:
        """
        生成信号DataFrame，包含signal和debug信息

        Parameters
        ----------
        dataset : DatasetH
            数据集
        segment : Union[str, slice]
            数据区间，如"test"、"train"等

        Returns
        -------
        pd.DataFrame
            列：score, percentile, current_pb, close
            索引为(datetime, instrument)
        """
        # 获取原始segment的时间范围（不包含lookback）
        if isinstance(segment, str) and segment in dataset.segments:
            seg_start, seg_end = dataset.segments[segment]
        elif isinstance(segment, slice):
            seg_start, seg_end = segment.start, segment.stop
        elif isinstance(segment, (tuple, list)) and len(segment) == 2:
            seg_start, seg_end = segment[0], segment[1]
        else:
            seg_start, seg_end = None, None
        
        # 加载数据（包含lookback_days的历史数据）
        df = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I, lookback_days=self.lookback_days)
        if df.empty:
            return pd.DataFrame(columns=["score", "percentile", "current_pb", "close"])

        # 提取必要列
        base_cols = ["pb", "close", "volume"]
        if self.spread_threshold is not None:
            if "pe_ttm" not in df.columns or "cn_2y" not in df.columns:
                raise ValueError(
                    "spread_threshold 需要数据集包含 pe_ttm 和 cn_2y 字段，"
                    "请在 workflow_config 的 handler fields 中添加这两个字段。"
                )
            base_cols += ["pe_ttm", "cn_2y"]
        df = df[base_cols].copy()
        
        # 注意：经过Handler处理后，不需要额外过滤
        # 直接使用pb数据计算百分位

        # 按股票分组计算百分位 - 使用高效的向量化方法
        def calc_percentile_fast(group):
            pb_arr = group["pb"].values
            close_arr = group["close"].values
            volume_arr = group["volume"].values
            n_rows = len(pb_arr)
            percentiles = np.full(n_rows, np.nan)

            # 判断每个交易日 ETF 是否可交易：close > 0 且 volume > volume_threshold
            # 仅用于决定是否输出 score，不影响 percentile 计算
            tradable = (
                (~np.isnan(close_arr)) & (close_arr > 0) &
                (~np.isnan(volume_arr)) & (volume_arr > self.volume_threshold)
            )

            for i in range(n_rows):
                start_idx = max(0, i - self.lookback_days + 1)
                window_pb = pb_arr[start_idx:i+1]

                # percentile 只过滤 pb NaN，不限制是否可交易
                valid_window = window_pb[~np.isnan(window_pb)]

                if len(valid_window) >= self.min_valid_days:
                    current = valid_window[-1]
                    hist = valid_window[:-1]
                    if len(hist) > 0:
                        percentiles[i] = np.sum(hist < current) / len(hist)

            return pd.DataFrame({
                "current_pb": group["pb"].values,
                "close": group["close"].values,
                "percentile": percentiles,
                "tradable": tradable,
            }, index=group.index)

        result = df.groupby(level="instrument", group_keys=False).apply(calc_percentile_fast)

        # 生成信号：不可交易日强制 score=0（无论 percentile 如何）
        result["score"] = 0
        tradable_mask = result["tradable"]
        buy_mask = tradable_mask & (result["percentile"] < self.oversold_threshold)

        # 如果启用了利差过滤，计算 spread 并附加条件
        if self.spread_threshold is not None:
            pe_ttm = df["pe_ttm"].reindex(result.index)
            cn_2y  = df["cn_2y"].reindex(result.index)
            spread = (1.0 / pe_ttm * 100).where(pe_ttm > 0) - cn_2y
            result["spread"] = spread
            buy_mask = buy_mask & (spread >= self.spread_threshold)
        
        result.loc[buy_mask, "score"] = 1
        result.loc[tradable_mask & (result["percentile"] > self.overbought_threshold), "score"] = -1

        # 过滤结果，只返回原始segment范围内的数据（去除lookback的历史数据）
        if seg_start is not None and seg_end is not None:
            seg_start = pd.Timestamp(seg_start)
            seg_end = pd.Timestamp(seg_end)
            result_dates = result.index.get_level_values("datetime")
            mask = (result_dates >= seg_start) & (result_dates <= seg_end)
            result = result[mask]
        
        return result[["score", "percentile", "current_pb", "close"]]