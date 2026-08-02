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

EMValModelSpace = {
    # 快线 EMA 周期 [5, 30]，步长 1
    "fast": 2, # hp.quniform("emval_fast", 2, 32, 1),
    # 中线 EMA 周期 [10, 60]，步长 1
    "mid": 4, # hp.quniform("emval_mid", 4, 64, 1),
    # 慢线 EMA 周期（估值线）[128, 8192]，对数均匀分布
    # qloguniform(low, high, q) = round(exp(uniform(low, high)) / q) * q
    # low = ln(128) ≈ 4.8520, high = ln(8192) ≈ 9.0109
    "slow": hp.qloguniform("emval_slow", 5.7, 9.010913347279293, 1),
    # 滚动窗口长度 [200, 2000]，步长 1
    "Nt": hp.quniform("emval_Nt", 200, 2400, 1),
    # 趋势死区宽度：None（不启用趋势确认）或 [0.0, 0.10]，步长 0.005
    "epsilon": None, # hp.choice("emval_epsilon", [None, hp.quniform("emval_epsilon_val", 0.0, 0.10, 0.005)]),
    # 最小有效数据比例 [0.10, 1.0]，步长 0.05
    "min_valid_days_ratio": 0.30, # hp.quniform("emval_min_valid_days_ratio", 0.10, 1.0, 0.05),
    # 买入分位阈值 [0.01, 0.20]，步长 0.01
    "buy_rank_thre": hp.quniform("emval_buy_rank_thre", 0.01, 0.20, 0.01),
    # 卖出分位阈值 [0.50, 0.98]，步长 0.01
    "sell_rank_thre": hp.quniform("emval_sell_rank_thre", 0.50, 0.99, 0.01),
}
