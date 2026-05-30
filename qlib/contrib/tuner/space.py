# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: skip-file
# flake8: noqa

import numpy as np  # noqa: F401 (reserved for future log-scale spaces)
from hyperopt import hp


TopkAmountStrategySpace = {
    "topk": hp.quniform("topk", 25, 45, 5),              # [25, 30, 35, 40, 45]
    "buffer_margin": hp.quniform("buffer_margin", 150, 350, 50),  # [150, 200, ..., 350]
}

QLibDataLabelSpace = {
    "labels": hp.choice(
        "labels",
        [["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["Ref($close, -5)/$close - 1"]],
    )
}

HistRelaPBModelSpace = {
    # 均匀整数分布 [500, 3000]，步长 1
    "lookback_days": hp.quniform("hist_rela_pb_lookback_days", 500, 3000, 1),
    # min_valid_days 占 lookback_days 的比例 [0.10, 0.60]，步长 0.05
    "min_valid_days_ratio": hp.quniform("hist_rela_pb_min_valid_days_ratio", 0.25, 1.0, 0.05),
    # 均匀离散分布，步长 0.01（hp.uniform 不支持 step，改用 quniform）
    "oversold_threshold": hp.quniform("hist_rela_pb_oversold_threshold", 0.01, 0.25, 0.01),
    # 均匀离散分布，步长 0.01
    "overbought_threshold": hp.quniform("hist_rela_pb_overbought_threshold", 0.40, 0.98, 0.01),
    "volume_threshold": 0.0,   # 固定值，不参与搜索
    "spread_threshold": 5.9,
    "freq": "day",
}

EvenWeightStrategySpace = {
    "risk_degree": 1.0,       # 固定值，不参与搜索
    "max_stock_weight": 1.0,
}

MACDSignalModelSpace = {
    # 快线 EMA 周期 [5, 20]，步长 1
    "fast": hp.quniform("macd_fast", 5, 20, 1),
    # 慢线 EMA 周期 [20, 60]，步长 1（搜索时须满足 fast < slow，由 model 内部处理）
    "slow": hp.quniform("macd_slow", 20, 60, 1),
    # 信号线 EMA 周期 [5, 15]，步长 1
    "signal": hp.quniform("macd_signal", 5, 15, 1),
}
