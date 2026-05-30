# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
MACDSignal - 基于 MACD 金叉/死叉的趋势跟踪信号模型

信号语义（三值，配合 EvenWeightStrategy）：
  score =  1：DIF > DEA（多头趋势，买入/持续持仓）
  score =  0：DIF == DEA（极少，中性，已持有则保仓）
  score = -1：DIF < DEA（空头趋势，平仓）

每日独立计算，不维护跨日状态；持仓连续性由信号连续性自然实现。
"""

import numpy as np
import pandas as pd
from typing import Union

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class MACDSignalModel(Model):
    """
    MACD 趋势信号模型

    参数
    ----
    fast : int
        快线 EMA 周期，默认 12
    slow : int
        慢线 EMA 周期，默认 26
    signal : int
        信号线 EMA 周期，默认 9
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self.fast = int(fast)
        self.slow = int(slow)
        self.signal = int(signal)
        self._warmup = self.slow + self.signal  # 有效信号最小所需行数

    def fit(self, dataset: DatasetH, reweighter=None) -> None:
        """MACD 无参数，fit 为空操作"""
        pass

    def predict(self, dataset: DatasetH, segment: Union[str, slice] = "test") -> pd.DataFrame:
        """
        生成 MACD 信号 DataFrame

        Returns
        -------
        pd.DataFrame
            列：score, dif, dea, histogram
            索引为 (datetime, instrument)
        """
        # 解析 segment 的时间范围
        if isinstance(segment, str) and segment in dataset.segments:
            seg_start, seg_end = dataset.segments[segment]
        elif isinstance(segment, slice):
            seg_start, seg_end = segment.start, segment.stop
        elif isinstance(segment, (tuple, list)) and len(segment) == 2:
            seg_start, seg_end = segment[0], segment[1]
        else:
            seg_start, seg_end = None, None

        # 加载数据（包含预热期）
        df = dataset.prepare(
            segment, col_set="feature",
            data_key=DataHandlerLP.DK_I,
            lookback_days=self._warmup,
        )
        if df.empty:
            return pd.DataFrame(columns=["score", "dif", "dea", "histogram"])

        close = df["close"].copy()

        def calc_macd(group: pd.Series) -> pd.DataFrame:
            prices = group.values.astype(float)
            n = len(prices)

            dif = np.full(n, np.nan)
            dea = np.full(n, np.nan)

            # 向量化 EMA（adjust=False 对齐传统 MACD 计算方式）
            s = pd.Series(prices)
            if s.notna().sum() < self.slow:
                # 数据不足，全部返回 0
                score = np.zeros(n)
                return pd.DataFrame(
                    {"score": score, "dif": dif, "dea": dea,
                     "histogram": np.zeros(n)},
                    index=group.index,
                )

            ema_fast = s.ewm(span=self.fast, adjust=False).mean().values
            ema_slow = s.ewm(span=self.slow, adjust=False).mean().values
            dif_arr = ema_fast - ema_slow
            dea_arr = pd.Series(dif_arr).ewm(span=self.signal, adjust=False).mean().values
            hist_arr = dif_arr - dea_arr

            # 预热期内（前 warmup 行）信号置 0
            score = np.where(
                np.isnan(prices),
                0,                                      # close 为 NaN → 0
                np.where(dif_arr > dea_arr, 1,          # DIF > DEA → +1
                np.where(dif_arr < dea_arr, -1, 0)),    # DIF < DEA → -1，相等 → 0
            ).astype(float)
            score[:self._warmup] = 0  # 预热期强制置 0

            return pd.DataFrame(
                {"score": score, "dif": dif_arr, "dea": dea_arr,
                 "histogram": hist_arr},
                index=group.index,
            )

        result = close.groupby(level="instrument", group_keys=False).apply(calc_macd)

        # 裁剪到原始 segment 范围（去除预热数据）
        if seg_start is not None and seg_end is not None:
            seg_start = pd.Timestamp(seg_start)
            seg_end = pd.Timestamp(seg_end)
            dates = result.index.get_level_values("datetime")
            result = result[(dates >= seg_start) & (dates <= seg_end)]

        return result[["score", "dif", "dea", "histogram"]]
