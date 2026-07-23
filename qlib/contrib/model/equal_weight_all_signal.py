# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
EqualWeightAllSignal — 等权全市场信号：对每个交易日出现的全部标的恒定输出买入信号。

配合 EvenWeightStrategy（三值信号 1/0/-1）实现"等权持有全市场"基准：
- fit  ：无需训练（空实现）
- predict(seg)：对该段内所有 (datetime, instrument) 输出 score=1

与 TopkAllInSignal 的区别：不做任何选股/排序，不依赖训练期存在性，
后上市标的会在其上市当日自然纳入等权组合（由 EvenWeightStrategy 每日再平衡）。
"""

import pandas as pd
from typing import Union

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP


class EqualWeightAllSignal(Model):
    """等权全市场信号：全部标的恒定 score=1。"""

    def __init__(self, **kwargs):
        pass

    def fit(self, dataset: DatasetH, reweighter=None):
        """等权基准无需训练。"""
        return

    def predict(self, dataset: DatasetH, segment: Union[str, slice] = "test") -> pd.DataFrame:
        """对该段全部标的输出 score=1（索引 (datetime, instrument)）。"""
        df = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        if df.empty:
            return pd.DataFrame(columns=["score", "close"])

        result = pd.DataFrame(index=df.index)
        result["score"] = 1
        result["close"] = df["close"].values if "close" in df.columns else float("nan")
        return result
