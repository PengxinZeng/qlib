# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
HistRelaPB Signal - 基于PB历史百分位的交易信号
"""

import numpy as np
import pandas as pd
from typing import Union

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
        k: int = 20,
        n: int = 5,
        oversold_threshold: float = 0.10,
        overbought_threshold: float = 0.90,
        volume_threshold: float = 0.0,
        freq: str = "day",
    ):
        """
        Parameters
        ----------
        k : int
            历史窗口天数
        n : int
            有效数据观察期（至少需要n条有效数据才计算信号）
        oversold_threshold : float
            超卖阈值，低于此百分位产生买入信号
        overbought_threshold : float
            超买阈值，高于此百分位产生卖出信号
        volume_threshold : float
            成交量阈值，低于此值的交易日数据将被忽略
        freq : str
            数据频率
        """
        self.k = k
        self.n = n
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.volume_threshold = volume_threshold
        self.freq = freq

    def fit(self, dataset: DatasetH, reweighter=None):
        """非深度学习模型，fit直接返回"""
        pass

    def predict(self, dataset: DatasetH, segment: Union[str, slice] = "test") -> pd.Series:
        """
        生成信号Series

        Parameters
        ----------
        dataset : DatasetH
            数据集
        segment : Union[str, slice]
            数据区间，如"test"、"train"等

        Returns
        -------
        pd.Series
            索引为(instrument, datetime)的信号Series
        """
        # 获取数据的时间和范围
        df = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)

        dates = df.index.get_level_values("datetime").unique().tolist()
        all_signals = []

        # 提前获取所有instrument
        instruments = df.index.get_level_values("instrument").unique().tolist()

        for date_idx, date in enumerate(dates):
            day_data = df.xs(date, level="datetime")

            for inst in day_data.index:
                row = day_data.loc[inst]

                # 获取该instrument从开始到当前日期的数据
                try:
                    inst_data = df.loc[inst]
                except KeyError:
                    # 该instrument在当前日期没有数据
                    signal = 0
                    all_signals.append({"instrument": inst, "datetime": date, "signal": signal})
                    continue

                # 只取历史数据（到当前日期为止）
                inst_data = inst_data.iloc[: date_idx + 1]
                if len(inst_data) < self.k:
                    signal = 0
                else:
                    hist_data = inst_data.iloc[-self.k:]

                    valid_pb = []
                    for _, hist_row in hist_data.iterrows():
                        if pd.isna(hist_row.get("close")) or pd.isna(hist_row.get("pb")):
                            continue
                        if hist_row.get("volume", 0) <= self.volume_threshold:
                            continue
                        valid_pb.append(hist_row["pb"])

                    if len(valid_pb) < self.n:
                        signal = 0
                    else:
                        current_pb = row["pb"]
                        if pd.isna(current_pb):
                            signal = 0
                        else:
                            percentile = np.sum(np.array(valid_pb) < current_pb) / len(valid_pb)

                            if percentile < self.oversold_threshold:
                                signal = 1
                            elif percentile > self.overbought_threshold:
                                signal = -1
                            else:
                                signal = 0

                all_signals.append({"instrument": inst, "datetime": date, "signal": signal})

        signal_df = pd.DataFrame(all_signals)
        if not signal_df.empty:
            signal_df = signal_df.set_index(["instrument", "datetime"])["signal"]
        return signal_df