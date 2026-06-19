import pandas as pd

from qlib.contrib.strategy.signal_strategy import WeightStrategyBase


class PPOTradingStrategy(WeightStrategyBase):
    """Portfolio strategy that uses PPOModel output scores directly as target weights.

    The PPOModel predict() returns per-stock holding ratios (sum ≤ 1, non-negative).
    This strategy feeds them directly to the WeightStrategyBase rebalancing logic.
    """

    def __init__(self, signal, risk_degree=1.0, **kwargs):
        super().__init__(signal=signal, **kwargs)
        self.risk_degree = risk_degree

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        score = score.clip(lower=0)
        total = score.sum()
        if total <= 0:
            return {}
        weight = score / total * self.risk_degree
        return weight[weight > 0].to_dict()
