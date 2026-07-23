# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
TopK ALL-IN Signal — 训练期按某种收益指标排序选出 TopK 标的，测试期恒定输出买入信号。

配合 EvenWeightStrategy（三值信号 1/0/-1）实现 "均匀持有 TopK" 的买入持有效果：
- fit(train)  ：在训练区间按 score_type 指定的指标排序，取前 top_k 名
- predict(seg)：对选中的 TopK 标的在每个交易日输出 score=1（持续持有），
                其余标的输出 score=-1（不持有）

score_type 排序指标：
- "total_return"      ：区间总收益率 (last/first - 1)
- "annualized"        ：区间年化收益率，按各标的在区间内的实际交易日数年化
                        （成立晚的标的即用其成立到区间末的实际跨度年化）
- "information_ratio" ：日收益率的年化风险调整比率 mean/std*sqrt(252)
                        （无基准时等价于年化夏普）
"""

import numpy as np
import pandas as pd
from typing import Union

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

_TRADING_DAYS_PER_YEAR = 238


class TopkAllInSignal(Model):
    """
    TopK ALL-IN 信号生成器（选股 = 训练期指标排序，信号 = 对选中标的恒定买入）。

    Parameters
    ----------
    top_k : int or "fit_val"
        - int      ：直接选取训练期指标最高的前 top_k 个标的
        - "fit_val"：在验证集上按 K 从 1..max_k 评测"训练期TopK 篮子"的表现，
                     选择使验证集指标最优的 K
    score_type : str
        排序指标，取值 "total_return" / "annualized" / "information_ratio"
    max_k : int or None
        top_k="fit_val" 时搜索的最大 K，默认取候选标的总数
    """

    _VALID_SCORE_TYPES = {"total_return", "annualized", "information_ratio"}
    _FIT_VAL = "fit_val"

    def __init__(self, top_k=1, score_type: str = "total_return", max_k=None, **kwargs):
        if isinstance(top_k, str) and top_k == self._FIT_VAL:
            self.fit_val = True
            self.top_k = None
        else:
            self.fit_val = False
            self.top_k = int(top_k)
        if score_type not in self._VALID_SCORE_TYPES:
            raise ValueError(
                f"score_type 须为 {sorted(self._VALID_SCORE_TYPES)} 之一，收到 {score_type!r}"
            )
        self.score_type = score_type
        self.max_k = int(max_k) if max_k is not None else None
        self.selected: list[str] = []

    # ──────────────────────────────────────────────────────────────

    def _segment_close(self, dataset: DatasetH, segment) -> pd.DataFrame:
        df = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        if "close" not in df.columns:
            raise ValueError("TopkAllInSignal 需要数据集包含 close 字段（$close）")
        return df

    def _score(self, close: pd.Series):
        """按 score_type 计算单个标的的排序指标；无效返回 None。"""
        s = close.dropna()
        if len(s) < 2 or s.iloc[0] <= 0:
            return None

        if self.score_type == "total_return":
            return s.iloc[-1] / s.iloc[0] - 1.0

        if self.score_type == "annualized":
            # 按实际交易日跨度年化（成立晚的标的用其自身跨度）
            years = (len(s) - 1) / _TRADING_DAYS_PER_YEAR
            if years <= 0:
                return None
            return (s.iloc[-1] / s.iloc[0]) ** (1.0 / years) - 1.0

        # information_ratio
        rets = s.pct_change().dropna()
        if len(rets) < 2 or rets.std() == 0:
            return None
        return rets.mean() / rets.std() * np.sqrt(_TRADING_DAYS_PER_YEAR)

    def _scores_by_instrument(self, df: pd.DataFrame) -> dict:
        """计算某区间内每个标的的 score_type 指标（剔除无效）。"""
        out = {}
        for inst, g in df["close"].groupby(level="instrument"):
            metric = self._score(g)
            if metric is not None:
                out[inst] = metric
        return out

    def _select_k_by_valid(self, ranked: list, dataset: DatasetH) -> int:
        """在验证集上评测"训练期TopK 篮子"的等权平均指标，返回最优 K。"""
        valid_df = self._segment_close(dataset, "valid")
        if valid_df.empty:
            raise ValueError("TopkAllInSignal.fit: top_k='fit_val' 需要 valid 区间数据")
        valid_scores = self._scores_by_instrument(valid_df)

        upper = min(self.max_k or len(ranked), len(ranked))
        best_k, best_perf = 1, float("-inf")
        for k in range(1, upper + 1):
            basket = ranked[:k]
            vals = [valid_scores[s] for s in basket if s in valid_scores]
            if not vals:
                continue
            perf = float(np.mean(vals))
            print(f"  fit_val K={k:2d}: 验证集等权{self.score_type}={perf:.4f}  篮子={basket}")
            if perf > best_perf:
                best_perf, best_k = perf, k
        print(f">>> fit_val 选出最优 K={best_k}, 验证集指标={best_perf:.4f}")
        return best_k

    def fit(self, dataset: DatasetH, reweighter=None):
        """在训练区间按 score_type 指标排序，选出 TopK（或按验证集自动选 K）。"""
        df = self._segment_close(dataset, "train")
        if df.empty:
            raise ValueError("TopkAllInSignal.fit: 训练区间无数据")

        scores = self._scores_by_instrument(df)
        if not scores:
            raise ValueError("TopkAllInSignal.fit: 训练区间无有效标的")

        ranked = sorted(scores, key=scores.get, reverse=True)

        if self.fit_val:
            best_k = self._select_k_by_valid(ranked, dataset)
            self.selected = ranked[:best_k]
        else:
            self.selected = ranked[: self.top_k]

        print(
            f"TopkAllInSignal fit: score_type={self.score_type}, "
            f"K={len(self.selected)} → {self.selected}"
        )


    def predict(self, dataset: DatasetH, segment: Union[str, slice] = "test") -> pd.DataFrame:
        """对选中标的输出 score=1，其余 score=-1（索引 (datetime, instrument)）。"""
        df = self._segment_close(dataset, segment)
        if df.empty:
            return pd.DataFrame(columns=["score", "close"])

        inst_level = df.index.get_level_values("instrument")
        score = pd.Series(-1, index=df.index, dtype=int)
        score[inst_level.isin(self.selected)] = 1

        result = pd.DataFrame({"score": score, "close": df["close"].values}, index=df.index)
        return result

