# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
EMVal - 基于估值分位 + 趋势确认的均值回归信号模型

信号语义（三值，配合 EvenWeightStrategy）：
  score =  1：买入（估值偏低 + 趋势转强）
  score =  0：无信号，维持上一状态
  score = -1：卖出（估值偏高 + 趋势转弱）

每日独立计算，不维护跨日状态；持仓连续性由信号连续性自然实现。
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Union
from numpy.lib.stride_tricks import sliding_window_view as _sliding_window_view

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger


class EMValModel(Model):
    """
    EMVal 估值分位 + 趋势确认信号模型

    参数
    ----
    fast : int
        快线 EMA 周期，默认 11
    mid : int
        中线 EMA 周期，默认 26
    slow : int
        慢线 EMA 周期（估值线），默认 37
    Nt : int
        滚动窗口长度（分位计算窗口），默认 500
    epsilon : float, optional
        趋势死区宽度，Diff 需超过 ±epsilon 才确认方向，默认 None。
        设为 None 则只使用估值分位信号，不检查趋势方向。
    buy_rank_thre : float
        买入分位阈值，默认 0.05
    sell_rank_thre : float
        卖出分位阈值，默认 0.75
    """

    def __init__(
        self,
        fast: int = 11,
        mid: int = 26,
        slow: int = 37,
        Nt: int = 500,
        epsilon: Optional[float] = None,
        min_valid_days_ratio: float = 0.20,
        buy_rank_thre: float = 0.05,
        sell_rank_thre: float = 0.75,
    ):
        self.fast = int(fast)
        self.mid = int(mid)
        self.slow = int(slow)
        self.Nt = int(Nt)
        self.epsilon = float(epsilon) if epsilon is not None else None
        self.min_valid_days_ratio = float(min_valid_days_ratio)
        self.min_valid_days = max(1, int(self.Nt * self.min_valid_days_ratio))
        self.buy_rank_thre = float(buy_rank_thre)
        self.sell_rank_thre = float(sell_rank_thre)
        # 预热期：需要足够数据计算 EMA_slow 和滚动分位
        self._warmup = max(self.slow, self.Nt)
        self.logger = get_module_logger("EMValModel", level=logging.INFO)

    def fit(self, dataset: DatasetH, reweighter=None) -> None:
        """EMVal 无参数，fit 为空操作"""
        pass

    @staticmethod
    def _parse_segment(segment: Union[str, slice], dataset: DatasetH):
        """解析 segment 的时间范围，返回 (seg_start, seg_end)"""
        if isinstance(segment, str) and segment in dataset.segments:
            return dataset.segments[segment]
        elif isinstance(segment, slice):
            return segment.start, segment.stop
        elif isinstance(segment, (tuple, list)) and len(segment) == 2:
            return segment[0], segment[1]
        else:
            return None, None

    @staticmethod
    def _compute_rolling_rank(values: np.ndarray, window: int) -> np.ndarray:
        """
        计算滚动分位：rank[t] = fraction of values in [t-window, t] <= values[t]

        对窗口内的 NaN 做安全处理：
        - values[t] 为 NaN → rank[t] = NaN
        - 窗口内其他 NaN 不参与比较（不计入分母，避免偏误）
        """
        n = len(values)
        rank = np.full(n, np.nan)

        # Step 1: 前 window 个位置用 expanding window
        for t in range(min(window, n)):
            if np.isnan(values[t]):
                continue
            window_vals = values[:t + 1]
            valid_vals = window_vals[~np.isnan(window_vals)]
            if len(valid_vals) > 0:
                rank[t] = (values[t] >= valid_vals).mean()

        # Step 2: 后续位置用滑动窗口（完全向量化，NaN-safe）
        if n > window:
            windows = _sliding_window_view(values, window_shape=window)
            last_vals = windows[:, -1]  # shape: (n-window,)

            valid_mask = ~np.isnan(windows)
            is_last_valid = ~np.isnan(last_vals)

            # 每个窗口的非 NaN 数量（至少为 1 防止除零）
            window_counts = np.maximum(valid_mask.sum(axis=1), 1)

            # 只在双方都有效时才参与比较
            comparison_mask = valid_mask & is_last_valid[:, np.newaxis]
            rank_numer = np.where(
                comparison_mask,
                (windows <= last_vals[:, np.newaxis]),
                0,
            ).sum(axis=1)

            ranks = np.where(is_last_valid, rank_numer / window_counts, np.nan)
            rank[window - 1:] = ranks

        return rank

    @staticmethod
    def _compute_emval_single_stock(
        prices: np.ndarray,
        volumes: np.ndarray,
        dates: pd.Index,
        instrument: str,
        fast: int,
        mid: int,
        slow: int,
        Nt: int,
        min_valid_days: int,
        epsilon: Optional[float],
        buy_rank_thre: float,
        sell_rank_thre: float,
        logger: logging.Logger,
        group_index: pd.Index,
    ) -> pd.DataFrame:
        """计算单只股票的 EMVal 信号"""
        n = len(prices)
        price_series = pd.Series(prices)

        # 1) 计算三条 EMA
        ema_fast = price_series.ewm(span=fast, adjust=False).mean().values
        ema_mid = price_series.ewm(span=mid, adjust=False).mean().values
        ema_slow = price_series.ewm(span=slow, adjust=False).mean().values

        # 2) 估值线与溢价率：val == 0 时返回 NaN（避免预期外的 0）
        val = ema_slow
        over_val_pct = np.where(val != 0, (prices - val) / val, np.nan)

        # 3) 滚动分位（NaN-safe）
        over_val_rank = EMValModel._compute_rolling_rank(over_val_pct, Nt)

        # 4) 趋势线：val == 0 时返回 NaN（避免预期外的 0）
        diff_arr = np.where(val != 0, (ema_fast - ema_mid) / val, np.nan)

        # 5) 数据质量检查：预热期后若 NaN 比例过高则发警告
        post_warmup_rank = over_val_rank[Nt:]
        if len(post_warmup_rank) > 0:
            nan_ratio = np.isnan(post_warmup_rank).mean()
            if nan_ratio > 0.5:
                logger.warning(
                    "stock=%-10s over_val_rank NaN ratio=%.2f after warmup, "
                    "signals may be unreliable",
                    instrument, nan_ratio,
                )

        # 6) 生成信号
        score = np.zeros(n, dtype=float)

        # 买入/卖出条件
        if epsilon is not None:
            buy_cond = (over_val_rank < buy_rank_thre) & (diff_arr > epsilon)
            sell_cond = (over_val_rank >= sell_rank_thre) & (diff_arr < -epsilon)
        else:
            buy_cond = over_val_rank < buy_rank_thre
            sell_cond = over_val_rank >= sell_rank_thre

        # 有效数据掩码：排除 NaN，避免 NaN 误触信号
        valid = ~np.isnan(prices) & ~np.isnan(over_val_rank)
        if epsilon is not None:
            valid = valid & ~np.isnan(diff_arr)

        # 向量化赋值
        score[valid & buy_cond] = 1.0
        score[valid & sell_cond] = -1.0

        # 预热期内信号强制置 0（取 min_valid_days 作为最小预热天数）
        score[:min_valid_days] = 0.0

        # 日志输出：按时间顺序输出该股票的买卖信号
        signal_mask = valid & (buy_cond | sell_cond)
        signal_mask[:min_valid_days] = False  # 预热期不输出日志
        signal_indices = np.where(signal_mask)[0]
        for t in signal_indices:
            direction = "BUY" if buy_cond[t] else "SELL"
            logger.debug(
                "[%s] stock=%-10s date=%s close=%-10.4f volume=%-12.2f "
                "diff=%-10.6f over_val_pct=%-10.6f over_val_rank=%-7.4f "
                "ema_fast=%-10.4f ema_mid=%-10.4f ema_slow=%-10.4f",
                direction, instrument, dates[t], prices[t], volumes[t],
                diff_arr[t], over_val_pct[t], over_val_rank[t],
                ema_fast[t], ema_mid[t], ema_slow[t],
            )

        return pd.DataFrame(
            {"score": score, "diff": diff_arr,
             "over_val_pct": over_val_pct,
             "over_val_rank": over_val_rank,
             "ema_fast": ema_fast, "ema_mid": ema_mid,
             "ema_slow": ema_slow},
            index=group_index,
        )

    def predict(self, dataset: DatasetH, segment: Union[str, slice] = "test") -> pd.DataFrame:
        """
        生成 EMVal 信号 DataFrame

        Returns
        -------
        pd.DataFrame
            列：score, diff, over_val_pct, over_val_rank, ema_fast, ema_mid, ema_slow
            索引为 (datetime, instrument)
        """
        seg_start, seg_end = self._parse_segment(segment, dataset)

        # 加载数据（包含预热期）
        df = dataset.prepare(
            segment, col_set="feature",
            data_key=DataHandlerLP.DK_I,
            lookback_days=self._warmup,
        )
        if df.empty:
            return pd.DataFrame(
                columns=["score", "diff", "over_val_pct", "over_val_rank",
                         "ema_fast", "ema_mid", "ema_slow"]
            )

        close = df["close"].copy()
        volume = df["volume"].copy()

        def _apply_single_stock(group: pd.DataFrame) -> pd.DataFrame:
            prices = group["close"].values.astype(float)
            volumes = group["volume"].values.astype(float)

            if isinstance(group.index, pd.MultiIndex):
                instrument = str(group.index[0][1])
                dates = group.index.get_level_values("datetime")
            else:
                instrument = str(group.index[0])
                dates = group.index

            return self._compute_emval_single_stock(
                prices=prices,
                volumes=volumes,
                dates=dates,
                instrument=instrument,
                fast=self.fast,
                mid=self.mid,
                slow=self.slow,
                Nt=self.Nt,
                min_valid_days=self.min_valid_days,
                epsilon=self.epsilon,
                buy_rank_thre=self.buy_rank_thre,
                sell_rank_thre=self.sell_rank_thre,
                logger=self.logger,
                group_index=group.index,
            )

        result = pd.concat([close, volume], axis=1).groupby(level="instrument", group_keys=False).apply(_apply_single_stock)

        # 裁剪到原始 segment 范围（去除预热数据）
        if seg_start is not None and seg_end is not None:
            seg_start = pd.Timestamp(seg_start)
            seg_end = pd.Timestamp(seg_end)
            dates = result.index.get_level_values("datetime")
            result = result[(dates >= seg_start) & (dates <= seg_end)]

        return result[["score", "diff", "over_val_pct", "over_val_rank",
                        "ema_fast", "ema_mid", "ema_slow"]]