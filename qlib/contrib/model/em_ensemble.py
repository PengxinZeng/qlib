# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
EMEnsemble - 多模型 Ensemble 综合持仓信号模型

整体流程（需求 2b）：
  1. 遍历各模型，分别用「模型解释器」将模型原始交易信号解释为每日软交易信号
  2. 用「模型组装器」综合各模型软信号，得到最终交易信号（持仓比例）

模型解释器（需求 2c）：
  - ValuationInterpreter（基于估值的解释器）：
      输入模型实验路径，输出模型每日软交易信号。
      处理逻辑：
        1) 模型交易信号为 1 / -1（买入 / 卖出）则输出 1 / -1
        2) 模型交易信号为 0 则根据「是否持有」与「估值情况」输出得分

模型组装器（需求 2d）：
  - ReturnBasedAssembler（基于收益情况的组装器）：
      输入各模型原始交易信号 + 股价。
      处理逻辑：
        对每模型×每股票用「原始信号」× 股价模拟交易计算收益率，
        以收益率/回撤作为各模型对各股票拟合能力的指标，控制得分权重，
        按权重加权「软交易信号」得到持仓比例。
"""

import os
import logging
import warnings
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger


class ConstantInterpreter:
    """
    固定信号解释器：不需要 exp_path，直接生成恒定 soft_signal / score。

    用于基准对照：
      - ALL in（成立即买入并一直持有）：soft_signal=1.0, score=1.0
      - 绝对不买入（从不持有，触发清仓）：soft_signal=-1.0, score=-1.0

    与 ValuationInterpreter 不同的地方：
      - 无需加载 exp_path/artifacts/pred.pkl
      - interpret() 需要外部传入 index（(datetime, instrument) MultiIndex），
        由 EMEnsembleModel.predict() 使用 price.index 提供
    """

    def __init__(
        self,
        soft_signal: float = 1.0,
        score: float = 1.0,
        score_col: str = "score",
    ):
        self.soft_signal = float(soft_signal)
        self.score = float(score)
        self.score_col = score_col
        self.logger = get_module_logger("ConstantInterpreter", level=logging.INFO)

    def interpret(self, index=None) -> pd.DataFrame:
        """生成恒定信号 DataFrame"""
        if index is None:
            raise ValueError(
                "ConstantInterpreter.interpret requires `index` (MultiIndex of (datetime, instrument))"
            )
        n = len(index)
        return pd.DataFrame(
            {
                "soft_signal": np.full(n, self.soft_signal, dtype=float),
                self.score_col: np.full(n, self.score, dtype=float),
            },
            index=index,
        )


class MACDSignalInterpreter:
    """
    MACD 趋势信号解释器：将 MACD 金叉/死叉三值信号接入 EMEnsemble。

    与 ConstantInterpreter 类似无需 exp_path（不是 mlruns 实验产物）；
    与 ValuationInterpreter 不同，它由 EMEnsembleModel.predict() 传入价格数据
    （含 lookback 前序历史），内部实时计算 MACD 信号。

    soft_signal = score（pass-through，三值）：
      -  1：DIF > DEA（多头趋势，买入/持续持仓）
      -  0：DIF == DEA（中性）
      - -1：DIF < DEA（空头趋势，平仓）

    Parameters
    ----------
    fast : int
        快线 EMA 周期
    slow : int
        慢线 EMA 周期
    signal : int
        信号线 EMA 周期
    score_col : str
        score 列名（默认 "score"，与组装器契约一致）
    """

    def __init__(
        self,
        fast: int = 11,
        slow: int = 37,
        signal: int = 6,
        score_col: str = "score",
    ):
        self.fast = int(fast)
        self.slow = int(slow)
        self.signal = int(signal)
        self.score_col = score_col
        self._warmup = self.slow + self.signal  # 有效信号最小所需行数
        self.logger = get_module_logger("MACDSignalInterpreter", level=logging.INFO)

    @staticmethod
    def _calc_macd(group: pd.Series, fast: int, slow: int, signal: int, warmup: int) -> pd.DataFrame:
        """对单只股票的价格序列计算 MACD 三值信号（与 MACDSignalModel 对齐）"""
        prices = group.values.astype(float)
        n = len(prices)
        s = pd.Series(prices)
        if s.notna().sum() < slow:
            # 数据不足，全部返回 0
            score = np.zeros(n)
        else:
            ema_fast = s.ewm(span=fast, adjust=False).mean().values
            ema_slow = s.ewm(span=slow, adjust=False).mean().values
            dif = ema_fast - ema_slow
            dea = pd.Series(dif).ewm(span=signal, adjust=False).mean().values
            score = np.where(
                np.isnan(prices),
                0,                                      # close 为 NaN → 0
                np.where(dif > dea, 1.0,                # DIF > DEA → +1
                np.where(dif < dea, -1.0, 0.0)),        # DIF < DEA → -1，相等 → 0
            ).astype(float)
            score[:warmup] = 0  # 预热期强制置 0
        return pd.DataFrame({"soft_signal": score}, index=group.index)

    def interpret(self, price: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        生成 MACD 每日软交易信号

        Parameters
        ----------
        price : pd.DataFrame, optional
            index=(datetime, instrument)，列含 close。由 EMEnsembleModel.predict() 提供
            （含 lookback_days 前序历史，最终裁剪由 predict() 统一完成）。

        Returns
        -------
        pd.DataFrame
            index=(datetime, instrument)，列：
            - soft_signal：软交易信号（三值 1/0/-1）
            - score：原始交易信号（1/0/-1），供组装器计算收益使用
        """
        if price is None:
            raise ValueError(
                "MACDSignalInterpreter.interpret requires `price` "
                "(DataFrame with MultiIndex (datetime, instrument) and 'close' column)"
            )
        if "close" not in price.columns:
            raise ValueError(f"MACDSignalInterpreter requires 'close' column, got {list(price.columns)}")
        close = price["close"].astype(float)
        soft = close.groupby(level="instrument", group_keys=False).apply(
            lambda g: self._calc_macd(g, self.fast, self.slow, self.signal, self._warmup)
        )
        result = pd.DataFrame(
            {
                "soft_signal": soft["soft_signal"],
                self.score_col: soft["soft_signal"],
            },
            index=soft.index,
        )
        return result


class ValuationInterpreter:
    """
    基于估值的模型解释器

    输入：模型实验路径（mlruns 下某 run 的目录，含 artifacts/pred.pkl）
    输出：模型每日软交易信号 DataFrame（index=(datetime, instrument)，列含 soft_signal）

    软交易信号语义：soft_signal ∈ [-1, 1]
      - 1.0  = 满仓买入
      - -1.0 = 卖出/做空
      - 0.0  = 空仓
      - 中间值 = 部分仓位

    处理逻辑：
      - use_valuation_mapping=False（默认）：直接输出 soft_signal = score（NaN → 0.0）
      - use_valuation_mapping=True：基于估值分位映射
        1) 模型交易信号为 1 / -1（买入 / 卖出）则输出 1 / -1
        2) 模型交易信号为 0 则根据「是否持有」与「估值情况」输出得分：
           - 若当前持有该股票：根据估值分位 over_val_rank 在 [buy_rank_thre, sell_rank_thre]
             区间线性映射得分（估值越低得分越高，越接近 1；估值越高得分越低，越接近 0）
           - 若当前未持有：输出 0（不新买）
    """

    def __init__(
        self,
        exp_path: str,
        buy_rank_thre: float = 0.05,
        sell_rank_thre: float = 0.75,
        score_col: str = "score",
        rank_col: str = "over_val_rank",
        use_valuation_mapping: bool = False,
        min_buy_days: int = 5,
        min_signal_days: int = 2,
    ):
        self.exp_path = exp_path
        self.buy_rank_thre = float(buy_rank_thre)
        self.sell_rank_thre = float(sell_rank_thre)
        self.score_col = score_col
        self.rank_col = rank_col
        self.use_valuation_mapping = bool(use_valuation_mapping)
        # 每只股票全局第一段买入信号（score=1）的最短持续天数（防噪）
        # 段一旦出现且持续不足 min_buy_days，则从段首起强制 soft=1 补足至 min_buy_days 天
        # min_buy_days <= 1 时不启用该防噪逻辑
        self.min_buy_days = int(min_buy_days)
        # 所有 ±1 买卖信号段（不限首次）的最短持续天数（防噪）
        # 段一旦出现且持续不足 min_signal_days，从段首起将 soft 补足为该段符号至至少 min_signal_days 天
        # min_signal_days <= 1 时不启用该防噪逻辑
        self.min_signal_days = int(min_signal_days)
        self.logger = get_module_logger("ValuationInterpreter", level=logging.INFO)

        # 参数验证：sell_rank_thre 必须大于 buy_rank_thre
        if self.sell_rank_thre <= self.buy_rank_thre:
            raise ValueError(
                f"sell_rank_thre ({self.sell_rank_thre}) must be > "
                f"buy_rank_thre ({self.buy_rank_thre})"
            )

    def _load_pred(self) -> pd.DataFrame:
        """加载模型实验的 pred.pkl"""
        pred_path = os.path.join(self.exp_path, "artifacts", "pred.pkl")
        if not os.path.exists(pred_path):
            raise FileNotFoundError(f"pred.pkl not found: {pred_path}")
        pred = pd.read_pickle(pred_path)
        if self.score_col not in pred.columns:
            raise ValueError(f"pred.pkl missing column '{self.score_col}'")
        if self.rank_col not in pred.columns:
            raise ValueError(f"pred.pkl missing column '{self.rank_col}'")
        return pred

    def _extend_first_buy_segment(self, soft: pd.Series, score: pd.Series) -> pd.Series:
        """
        每只股票全局第一段买入信号（score=1）防噪：
        段一旦出现且持续天数 < min_buy_days，则从段首起强制 soft=1 补足至 min_buy_days 天；
        持续 >= min_buy_days 的段不受影响；后续买入段不受影响。

        Parameters
        ----------
        soft : pd.Series
            index=(datetime, instrument)，已生成的软信号
        score : pd.Series
            index=(datetime, instrument)，原始三值信号（1/0/-1）

        Returns
        -------
        pd.Series
            防噪后的软信号
        """
        if self.min_buy_days <= 1:
            return soft
        soft = soft.copy()
        for instrument, grp in score.groupby(level="instrument"):
            grp_score = grp.values
            n = len(grp_score)
            pos = np.where(grp_score == 1.0)[0]
            if len(pos) == 0:
                continue
            start = int(pos[0])
            # 第一段连续买入段的末尾
            end = start
            while end + 1 < n and grp_score[end + 1] == 1.0:
                end += 1
            seg_len = end - start + 1
            if seg_len >= self.min_buy_days:
                continue
            # 补足：段首起至少 min_buy_days 天 soft=1（不超出数据长度）
            ext_end = min(start + self.min_buy_days, n)
            soft.loc[grp.index[start:ext_end]] = 1.0
        return soft

    def _extend_min_signal_days(self, soft: pd.Series, score: pd.Series) -> pd.Series:
        """
        所有 ±1 买卖信号段（不限首次）最短持续 min_signal_days 天防噪：
        段一旦出现且持续不足 min_signal_days，则从段首起将 soft 补足为该段符号（+1/-1）
        至至少 min_signal_days 天（不超出数据长度）；持续充足的段不受影响。

        Parameters
        ----------
        soft : pd.Series
            index=(datetime, instrument)，已生成的软信号
        score : pd.Series
            index=(datetime, instrument)，原始三值信号（1/0/-1）

        Returns
        -------
        pd.Series
            防噪后的软信号
        """
        if self.min_signal_days <= 1:
            return soft
        soft = soft.copy()
        for instrument, grp in score.groupby(level="instrument"):
            grp_score = grp.values
            n = len(grp_score)
            i = 0
            while i < n:
                s = grp_score[i]
                if s not in (1.0, -1.0):
                    i += 1
                    continue
                j = i + 1
                # 连续同类信号段
                while j < n and grp_score[j] == s:
                    j += 1
                if j - i < self.min_signal_days:
                    # 补足：段首起至少 min_signal_days 天 soft = 段符号
                    ext_end = min(i + self.min_signal_days, n)
                    soft.loc[grp.index[i:ext_end]] = s
                i = j
        return soft

    def _interpret_pass_through(self, score: pd.Series) -> pd.Series:
        """
        直接使用原始信号作为软信号（不做估值映射）

        soft_signal = score，NaN → 0.0
        """
        return score.fillna(0.0)

    def _interpret_valuation_mapping(self, score: pd.Series, rank: pd.Series) -> pd.Series:
        """
        基于估值分位的软信号映射（向量化实现）

        逐股票状态机：
          - score=1  -> 持有，soft=1.0
          - score=-1 -> 不持有，soft=-1.0
          - score=0  -> 保持上一持有状态；若持有则按估值分位线性映射，否则 soft=0.0
          - score=NaN -> 保持上一持有状态，soft=0.0
        """
        soft = pd.Series(0.0, index=score.index, dtype=float)

        # 按股票分组处理（保持时间顺序）
        for instrument, grp in score.groupby(level="instrument"):
            grp_score = grp.values.astype(float)
            grp_rank = rank.loc[grp.index].values.astype(float)
            n = len(grp_score)

            # 1) 确定持有状态：用 ffill 传播最后非零信号
            #    score=1 -> 持有；score=-1 -> 不持有；score=0/NaN -> 保持上一状态
            last_signal = np.where(
                grp_score == 1.0,
                1.0,
                np.where(grp_score == -1.0, -1.0, np.nan),
            )
            last_signal_ffilled = pd.Series(last_signal).ffill().values
            holding = np.where(
                np.isnan(last_signal_ffilled),
                False,
                last_signal_ffilled == 1.0,
            )

            # 2) 计算 soft 信号
            grp_soft = np.zeros(n, dtype=float)

            # score=1 -> 1.0
            grp_soft[grp_score == 1.0] = 1.0
            # score=-1 -> -1.0
            grp_soft[grp_score == -1.0] = -1.0

            # score=0 且持有 -> 估值线性映射
            zero_holding_mask = (grp_score == 0.0) & holding
            if zero_holding_mask.any():
                r = grp_rank[zero_holding_mask]
                span = self.sell_rank_thre - self.buy_rank_thre  # > 0，已在 __init__ 验证
                mapped = np.clip((self.sell_rank_thre - r) / span, 0.0, 1.0)
                # NaN rank -> 0.0
                mapped = np.where(np.isnan(mapped), 0.0, mapped)
                grp_soft[zero_holding_mask] = mapped

            # score=0 且不持有 -> 0.0（默认）
            # score=NaN -> 0.0（默认）

            soft.loc[grp.index] = grp_soft

        return soft

    def interpret(self) -> pd.DataFrame:
        """
        生成模型每日软交易信号

        Returns
        -------
        pd.DataFrame
            index=(datetime, instrument)，列：
            - soft_signal：软交易信号，取值 [-1, 1]
            - score：原始交易信号（1/0/-1），供组装器计算收益使用
        """
        pred = self._load_pred()
        score = pred[self.score_col].astype(float)
        rank = pred[self.rank_col].astype(float)

        if self.use_valuation_mapping:
            soft = self._interpret_valuation_mapping(score, rank)
        else:
            soft = self._interpret_pass_through(score)

        # 统一后处理：首段买入信号防噪（不足 min_buy_days 则补足拉长）
        soft = self._extend_first_buy_segment(soft, score)
        # 统一后处理：所有 ±1 买卖信号段（不限首次）最短维持 min_signal_days 天
        soft = self._extend_min_signal_days(soft, score)

        result = pd.DataFrame(
            {"soft_signal": soft, self.score_col: score},
            index=pred.index,
        )
        return result


# =====================================================================
# 策略族 A：模型选择（ModelSelector）
# 决定每个时刻每只股票选哪个模型（one-hot 权重），策略可插拔
# =====================================================================


class BaseModelSelector:
    """
    模型选择策略抽象（Strategy）

    - build_weight_df(cum_ret_df) -> DataFrame(index=all_index, columns=model_names)，
      值为 one-hot 0/1：每行（每 datetime-instrument）指示被选中模型。
    - select_signal(cum_ret_df, raw_signals, close) -> Optional[pd.Series]：
      可选能力。返回直接作为最终 ensemble 交易信号（±1/0）的序列；None 表示不使用
      （此时走默认 one-hot × soft_signal 加权）。默认 None。
    """

    def build_weight_df(self, cum_ret_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def select_signal(
        self,
        cum_ret_df: pd.DataFrame,
        raw_signals: Optional[Dict[str, pd.DataFrame]] = None,
        close: Optional[pd.Series] = None,
    ) -> Optional[pd.Series]:
        """可选：直接生成最终 ensemble 交易信号（±1/0）。默认不使用。"""
        return None


class DynamicBestSelector(BaseModelSelector):
    """
    动态最优选择策略：每个交易日取累计收益最高模型，one-hot → 权重（模型可每日切换）

    与原 ReturnBasedAssembler 默认行为一致：
      weight_df[t, argmax(cum_ret[t])] = 1
    """

    def build_weight_df(self, cum_ret_df: pd.DataFrame) -> pd.DataFrame:
        model_names = cum_ret_df.columns
        all_index = cum_ret_df.index
        best_model_idx = cum_ret_df.values.argmax(axis=1)  # 每行最优模型
        n_rows, n_models = cum_ret_df.shape
        one_hot = np.zeros((n_rows, n_models))
        one_hot[np.arange(n_rows), best_model_idx] = 1.0
        return pd.DataFrame(one_hot, index=all_index, columns=model_names)


class LockedBestSelector(BaseModelSelector):
    """
    锁定最优选择策略：测试开始日（lock_start_time）每只股票选累计收益最高模型，整段 one-hot 冻结、不切换

    - lock_start_time：锁定基准日（通常 = backtest_start）。取该日或该日之前最近一个交易日该股票的
      cumret argmax 作为锁定模型；某股票该段无数据则回退其首个可用交易日。
    - lock_start_time 缺省：回退到数据首日选模型。
    """

    def __init__(self, lock_start_time: Optional[str] = None):
        self.lock_start_time = pd.Timestamp(lock_start_time) if lock_start_time is not None else None

    def build_weight_df(self, cum_ret_df: pd.DataFrame) -> pd.DataFrame:
        model_names = cum_ret_df.columns
        all_index = cum_ret_df.index
        vals = cum_ret_df.values
        n, n_models = vals.shape
        dates = all_index.get_level_values("datetime")
        instruments = all_index.get_level_values("instrument")

        # 确定锁定基准日（<= 该日的最近交易日为该股票锁定参考行）
        lock_ts = self.lock_start_time if self.lock_start_time is not None else dates.min()
        le_mask = dates <= lock_ts
        le_rows = np.where(le_mask)[0]

        # 每个 instrument 在基准区间内的最大行号（最近一个可用交易日）
        lock_row_by_inst = (
            pd.DataFrame({"row": le_rows, "inst": instruments[le_rows]})
            .groupby("inst")["row"]
            .max()
        )

        # lock_start_time 之前无数据（或早于首行）的股票 → 回退到全数据首日
        missing = pd.Index(sorted(set(instruments))).difference(lock_row_by_inst.index)
        if len(missing) > 0:
            first_row_by_inst = (
                pd.DataFrame({"row": np.arange(n), "inst": instruments})
                .groupby("inst")["row"]
                .min()
            )
            lock_row_by_inst = pd.concat([
                lock_row_by_inst,
                first_row_by_inst[missing],
            ]).sort_index()

        # 每只股票锁定模型 = 参考行 cumret argmax
        locked_model = np.zeros(n, dtype=int)
        for inst, row in lock_row_by_inst.items():
            inst_mask = instruments == inst
            locked_model[inst_mask] = int(vals[row].argmax())

        one_hot = np.zeros((n, n_models))
        one_hot[np.arange(n), locked_model] = 1.0
        return pd.DataFrame(one_hot, index=all_index, columns=model_names)


class FutureBestSelector(BaseModelSelector):
    """
    短期未来最优模型选择策略

    每个交易日 t、每只股票独立地，对每个模型 m：
      1a. 用 t-k（k=1）到 t 日的股价线性拟合，外推预测 t+1、t+2 日价格 P̂_{t+1}、P̂_{t+2}
      1b. 由模型原始 score（1/0/-1）状态机推持有状态（T+1 生效，0 保持）：
          H_{t+1} = f(score_t)；H_{t+2} = H_{t+1}
      1c. 复利外推 t+2 预期累积收益率：
          E_cum_{t+2} = (1+cum_t)·(1+H_{t+1}·(P̂_{t+1}/P_t-1))·(1+H_{t+2}·(P̂_{t+2}/P̂_{t+1}-1)) - 1
      1d. 选 E_cum_{t+2} 最高的模型为 BestM

    当日交易信号按「跟随 BestM」生成：
      1. BestM_score[t] = ±1     → EM_score[t] = BestM_score[t]
      2. BestM_score[t] = 0 且 BestM 持有 → EM_score[t] = +1（买入/维持持有）
      3. BestM_score[t] = 0 且 BestM 空仓 → EM_score[t] = -1（卖出/维持空仓）

    通过实现 select_signal() 直接输出最终 ensemble 交易信号（±1/0）。
    数据不足（无 k 个历史价格）时回退 DynamicBest（cum_ret argmax 选模型，再应用同规则）。

    downside_protect : bool, default False
        下跌保护：若预测 P̂(t+2) < P(t)（两日后价格低于今日），强制输出 -1（空仓），
        不看多。用于深熊区间避免"预测上涨→持有→下跌"的反复损耗。
    """

    def __init__(
        self,
        k: int = 1,
        horizon: int = 2,
        score_col: str = "score",
        downside_protect: bool = False,
    ):
        self.k = max(1, int(k))
        self.horizon = max(1, int(horizon))
        self.score_col = score_col
        self.downside_protect = bool(downside_protect)

    def build_weight_df(self, cum_ret_df: pd.DataFrame) -> pd.DataFrame:
        """回退到累计收益最高模型（DynamicBest），兼容只调用 build_weight_df 的场景。"""
        model_names = cum_ret_df.columns
        all_index = cum_ret_df.index
        best_model_idx = cum_ret_df.values.argmax(axis=1)
        n_rows, n_models = cum_ret_df.shape
        one_hot = np.zeros((n_rows, n_models))
        one_hot[np.arange(n_rows), best_model_idx] = 1.0
        return pd.DataFrame(one_hot, index=all_index, columns=model_names)

    @staticmethod
    def _scores_to_hold(scores: np.ndarray) -> np.ndarray:
        """score → 持有状态（T+1 生效，0 保持，-1→空仓）。与 _simulate_return 的 hold 语义一致。"""
        n = len(scores)
        event = np.where(scores == 1.0, 1.0, np.where(scores == -1.0, -1.0, np.nan))
        last = np.nan
        ev_ff = np.empty(n, dtype=float)
        for i in range(n):
            if not np.isnan(event[i]):
                last = event[i]
            ev_ff[i] = last
        hold = np.empty(n, dtype=float)
        hold[0] = 0.0
        hold[1:] = ev_ff[:-1]
        hold = np.where(hold == -1.0, 0.0, hold)
        return hold

    @staticmethod
    def _next_hold(sc_t: float, hold_t: float) -> float:
        """t+1 日持有状态：由 t 日 score 决定（±1 动作，0 保持当前状态）。"""
        if sc_t == 1.0:
            return 1.0
        if sc_t == -1.0:
            return 0.0
        return hold_t

    def select_signal(
        self,
        cum_ret_df: pd.DataFrame,
        raw_signals: Optional[Dict[str, pd.DataFrame]] = None,
        close: Optional[pd.Series] = None,
    ) -> Optional[pd.Series]:
        if raw_signals is None or close is None:
            return None
        model_names = list(cum_ret_df.columns)
        all_index = cum_ret_df.index
        result = pd.Series(0.0, index=all_index, dtype=float)

        for instrument, grp in cum_ret_df.groupby(level="instrument"):
            inst_dates = grp.index.get_level_values("datetime")
            inst_close = close.loc[(slice(None), instrument)].reindex(inst_dates).values.astype(float)
            scores = {}
            holds = {}
            cum_vals = grp.values  # shape=(n_days, n_models)
            for m in model_names:
                sc = raw_signals[m][self.score_col].reindex(grp.index).fillna(0.0).values.astype(float)
                scores[m] = sc
                holds[m] = self._scores_to_hold(sc)

            n = len(grp)
            for t in range(n):
                # 1a：线性拟合外推价格（需要 >=2 个有效价格点）
                price_ok = t >= self.k and not np.isnan(inst_close[t]) and inst_close[t] > 0
                if price_ok:
                    xs = np.arange(t - self.k, t + 1, dtype=float)
                    ys = inst_close[t - self.k:t + 1]
                    if not np.any(np.isnan(ys)):
                        slope, intercept = np.polyfit(xs, ys, 1)
                        p1 = float(slope * (t + 1) + intercept)
                        p2 = float(slope * (t + 2) + intercept)
                        predicted_down = self.downside_protect and (p2 < inst_close[t])
                        if predicted_down:
                            # 下跌保护：预测两日后价格低于今日 → 不看多，强制空仓
                            result.loc[grp.index[t]] = -1.0
                            continue
                        if p1 > 0:
                            # 1c：复利外推 E_cum_{t+2}
                            best_m, best_e = None, -np.inf
                            for m in model_names:
                                h_next = self._next_hold(scores[m][t], holds[m][t])
                                h_next2 = h_next  # t+2 无新信号 -> 保持
                                cum_t = float(cum_vals[t, model_names.index(m)])
                                ret1 = h_next * (p1 / inst_close[t] - 1.0)
                                ret2 = h_next2 * (p2 / p1 - 1.0)
                                e_cum = (1.0 + cum_t) * (1.0 + ret1) * (1.0 + ret2) - 1.0
                                if e_cum > best_e:
                                    best_e, best_m = e_cum, m
                            if best_m is None:
                                best_m = model_names[int(cum_vals[t].argmax())]
                        else:
                            best_m = model_names[int(cum_vals[t].argmax())]
                    else:
                        best_m = model_names[int(cum_vals[t].argmax())]
                else:
                    # 回退：cum_ret 最高模型
                    best_m = model_names[int(cum_vals[t].argmax())]

                # 信号规则（follow BestM）
                sc_b = scores[best_m][t]
                h_b = holds[best_m][t]
                if sc_b == 1.0 or sc_b == -1.0:
                    em_sig = sc_b
                elif h_b == 1.0:
                    em_sig = 1.0
                else:
                    em_sig = -1.0
                result.loc[grp.index[t]] = em_sig


class SustainedBestSelector(BaseModelSelector):
    """
    持续最优切换选择策略：测试开始日选累计收益最高模型 M，跟随其持仓状态/交易信号；
    当有其他模型 J 的累计收益连续（N 个交易日）优于 M 时才切换跟随 J。

    与 LockedBestSelector 的区别：Locked 整段冻结从不切换；
    与 DynamicBestSelector 的区别：Dynamic 每日按 argmax 频繁切换；
    本策略只在"其他模型持续领先 N 天后"才切换，兼顾长线优势与切换稳定性。

    选择语义（逐股票）：
      1. 数据首日（=测试开始日）current = argmax(cum_ret[0])，即收益最高模型 M
      2. 每日 t，对每个候选 J != current：
         - 若 cum_ret[J][t] > cum_ret[current][t]，则 J 的连续优势计数 +1
         - 否则该 J 的计数清零（中途被追回则重新累计）
      3. 一旦某 J 的连续优势计数达 N -> current = J，所有计数清零
      4. 当日交易信号跟随 current 的持仓状态/交易信号（与 FutureBestSelector 一致）：
         - current.score[t] = ±1     -> 直接采用该信号
         - current.score[t] = 0 且 current 持有 -> +1（买入/维持持有）
         - current.score[t] = 0 且 current 空仓 -> -1（卖出/维持空仓）

    通过实现 select_signal() 直接输出最终 ensemble 交易信号（±1/0），
    完全替代 one-hot×soft 加权（与 FutureBestSelector 相同机制）。

    N : int, default 256
        连续优势交易日阈值。
    """

    def __init__(self, N: int = 256, score_col: str = "score", follow_signal: bool = True):
        self.N = max(1, int(N))
        self.score_col = score_col
        self.follow_signal = bool(follow_signal)

    def _sustained_best(self, cum_ret_df: pd.DataFrame) -> pd.Series:
        """
        持续最优状态机（follow_signal=True / False 两分支共享的唯一实现）：

        逐股票独立运行，返回 Series(index=cum_ret_df.index, value=当日主导模型名)。
          1. 每只股票数据首日 current = argmax(cum_ret[0])，即收益最高模型 M
          2. 每日 t，对每个候选 J != current：
             - 若 cum_ret[J][t] > cum_ret[current][t]，则 J 的连续优势计数 +1
             - 否则该 J 的计数清零（中途被追回则重新累计）
          3. 一旦某 J 的连续优势计数达 N -> current = J，所有计数清零

        Parameters
        ----------
        cum_ret_df : pd.DataFrame
            index=MultiIndex(datetime, instrument)，columns=model_names，值为各模型累计收益

        Returns
        -------
        pd.Series
            index=cum_ret_df.index，value=每行主导模型名（object）
        """
        model_names = list(cum_ret_df.columns)
        best = pd.Series("", index=cum_ret_df.index, dtype=object)
        for instrument, grp in cum_ret_df.groupby(level="instrument"):
            cum_vals = grp.values  # shape=(n_days, n_models)，时间升序
            n = len(grp)
            # 数据首日锁定收益最高模型 M
            current = model_names[int(cum_vals[0].argmax())]
            cnt = {m: 0 for m in model_names}
            for t in range(n):
                cur_v = float(cum_vals[t, model_names.index(current)])
                # 更新各候选的连续优势计数
                for j in model_names:
                    if j == current:
                        continue
                    if float(cum_vals[t, model_names.index(j)]) > cur_v:
                        cnt[j] += 1
                    else:
                        cnt[j] = 0
                # 首个连续优于 M 达 N 天的候选 -> 切换
                hit = [m for m in model_names if m != current and cnt[m] >= self.N]
                if hit:
                    current = hit[0]
                    for m in model_names:
                        cnt[m] = 0
                best.loc[grp.index[t]] = current
        return best

    def build_weight_df(self, cum_ret_df: pd.DataFrame) -> pd.DataFrame:
        """按 _sustained_best 主导模型输出 one-hot 权重（逐股票独立，N 生效）。"""
        model_names = list(cum_ret_df.columns)
        best = self._sustained_best(cum_ret_df)
        # 记录逐日主导模型，供下游输出 best_model 列
        self._last_best_model = best
        n = len(best)
        col_idx = best.map({m: i for i, m in enumerate(model_names)}).to_numpy(dtype=int)
        one_hot = np.zeros((n, len(model_names)), dtype=float)
        one_hot[np.arange(n), col_idx] = 1.0
        return pd.DataFrame(one_hot, index=cum_ret_df.index, columns=model_names)

    def select_signal(
        self,
        cum_ret_df: pd.DataFrame,
        raw_signals: Optional[Dict[str, pd.DataFrame]] = None,
        close: Optional[pd.Series] = None,
    ) -> Optional[pd.Series]:
        if raw_signals is None or close is None:
            return None
        if not self.follow_signal:
            return None  # follow_signal=False -> one-hot x soft weighting
        model_names = list(cum_ret_df.columns)
        all_index = cum_ret_df.index
        result = pd.Series(0.0, index=all_index, dtype=float)
        # 记录逐日主导模型（SustainedBest 当前跟随的模型名），供下游输出 best_model
        best = self._sustained_best(cum_ret_df)
        self._last_best_model = best

        for instrument, grp in cum_ret_df.groupby(level="instrument"):
            inst_best = best.loc[grp.index]
            scores = {}
            holds = {}
            for m in model_names:
                sc = raw_signals[m][self.score_col].reindex(grp.index).fillna(0.0).values.astype(float)
                scores[m] = sc
                holds[m] = FutureBestSelector._scores_to_hold(sc)

            # 信号规则（follow current）
            for t, idx in enumerate(grp.index):
                cur_m = inst_best.iloc[t]
                sc_c = scores[cur_m][t]
                h_c = holds[cur_m][t]
                if sc_c == 1.0 or sc_c == -1.0:
                    em_sig = sc_c
                elif h_c == 1.0:
                    em_sig = 1.0
                else:
                    em_sig = -1.0
                result.loc[idx] = em_sig

        return result


        return result


# =====================================================================
# 策略族 B：信号增强（SignalEnhancer）
# 在基础 one-hot×soft 信号上追加强制买卖信号（覆盖 score=±1），默认 NoOp
# =====================================================================


class BaseSignalEnhancer:
    """
    信号增强策略抽象（Strategy）
    enhance(cum_ret_df) -> Optional[pd.Series](index=all_index, 值 0/±1)；None 表示不覆盖。
    """

    def enhance(self, cum_ret_df: pd.DataFrame) -> Optional[pd.Series]:
        raise NotImplementedError


class NoOpEnhancer(BaseSignalEnhancer):
    """空增强（Null Object）：不产生任何覆盖信号"""

    def enhance(self, cum_ret_df: pd.DataFrame) -> Optional[pd.Series]:
        return None


class OvertakeFollowEnhancer(BaseSignalEnhancer):
    """
    反超跟风 / 提前跟风增强器（原 ReturnBasedAssembler 步骤 2.5 逻辑）

    空仓/持仓两状态机（前日冠军基准）：
      基准 B[t]     = 前日冠军模型 今日的累计收益
      基准 B_prev[t] = 前日冠军模型 前日的累计收益
      gap_ref[t]     = B[t]     - cumret[t]     # 今日相对前日冠军今日值
      gap_ref_prev[t] = B_prev[t] - cumret[t-1]  # 前日相对前日冠军前日值

      买入（空仓才执行）：
        1a 前日 gap>=2*买阈的模型今日 gap 跨入 <买阈 → +1
        1b 前日 gap< 2*买阈的模型今日 gap<0（超过前日冠军今日值）→ +1
      卖出（持仓才执行）：
        2a 前日 gap>=2*卖阈的模型今日 gap<卖阈（大幅领先快速收窄、含跌穿 0）→ -1
        2b 前日 gap< 2*卖阈的模型今日 gap<0（被前日冠军超越/冠军崩落）→ -1
      状态机按当前空仓/持仓状态决定方向（空仓买、持仓卖），买卖严格交替。
    """

    def __init__(
        self,
        overtake_follow: bool = True,
        overtake_oracle: bool = False,
        overtake_lead_days: int = 1,
        pre_follow_buy_thre: float = 0.01,
        pre_follow_sell_thre: float = 0.01,
    ):
        self.overtake_follow = bool(overtake_follow)
        self.overtake_oracle = bool(overtake_oracle)
        self.overtake_lead_days = max(1, int(overtake_lead_days))
        self.pre_follow_buy_thre = float(pre_follow_buy_thre)
        self.pre_follow_sell_thre = float(pre_follow_sell_thre)
        self.logger = get_module_logger("OvertakeFollowEnhancer", level=logging.INFO)

    def enhance(self, cum_ret_df: pd.DataFrame) -> Optional[pd.Series]:
        all_index = cum_ret_df.index
        if not (self.overtake_follow or self.overtake_oracle):
            return None
        overtake_signal = pd.Series(0.0, index=all_index, dtype=float)
        for instrument, grp in cum_ret_df.groupby(level="instrument"):
            vals = grp.values  # shape=(n_days, n_models)，时间升序
            n_days, n_models = vals.shape
            if n_days < 2:
                continue
            grp_best = vals.argmax(axis=1)  # 各 t 日最优模型

            if self.overtake_oracle:
                # ---- 反超跟风奇迹（oracle 分支） ----
                changed = np.concatenate([[False], grp_best[1:] != grp_best[:-1]])
                if not changed.any():
                    continue
                diff = np.diff(vals, axis=0)
                chg_idx = np.where(changed)[0] - 1
                new_best = grp_best[changed]
                ret_chg = diff[chg_idx, new_best]
                dir_vals = np.where(ret_chg > 0, 1.0, -1.0)
                r_pos = np.where(changed)[0]
                for o, d in zip(r_pos, dir_vals):
                    start = max(0, o - self.overtake_lead_days)
                    overtake_signal.loc[grp.index[start:o + 1]] = d
                continue

            # ---- 空仓/持仓两状态机（前日冠军基准） ----
            buy_thre = self.pre_follow_buy_thre
            sell_thre = self.pre_follow_sell_thre

            # 前一日冠军索引（t=0 用当日冠军）
            prev_best_idx = np.concatenate([[grp_best[0]], grp_best[:-1]])
            # B[t]     = 前日冠军的今日累计收益
            # B_prev[t] = 前日冠军的前日累计收益
            B_col = vals[np.arange(n_days), prev_best_idx][:, None]
            B_prev_col = np.empty(n_days, dtype=float)
            B_prev_col[0] = vals[0, grp_best[0]]
            B_prev_col[1:] = vals[np.arange(n_days - 1), grp_best[:-1]]
            B_prev_col = B_prev_col[:, None]

            gap_ref = B_col - vals                              # 今日各模型相对前日冠军今日值
            gap_ref_prev = np.empty_like(gap_ref)
            gap_ref_prev[0] = gap_ref[0]
            gap_ref_prev[1:] = B_prev_col[1:] - vals[:-1]       # 前日各模型相对前日冠军前日值

            buy_outer = (gap_ref_prev >= 2.0 * buy_thre) & (gap_ref < buy_thre)
            buy_inner = (gap_ref_prev < 2.0 * buy_thre) & (gap_ref < 0.0)
            buy_sig = (buy_outer | buy_inner).any(axis=1)
            sell_outer = (
                (gap_ref_prev >= 2.0 * sell_thre)
                & (gap_ref < sell_thre)
            )
            sell_inner = (gap_ref_prev < 2.0 * sell_thre) & (gap_ref < 0.0)
            sell_sig = (sell_outer | sell_inner).any(axis=1)

            # 方案 A 状态机：空仓触 buy → +1；持仓触 sell → -1；动作严格交替
            dir_vals = np.zeros(n_days, dtype=float)
            state = 0  # 0=空仓, 1=持仓
            for t in range(n_days):
                if buy_sig[t] and state == 0:
                    dir_vals[t] = 1.0
                    state = 1
                elif sell_sig[t] and state == 1:
                    dir_vals[t] = -1.0
                    state = 0
            r_pos = np.where(dir_vals != 0)[0]
            if len(r_pos) > 0:
                overtake_signal.loc[grp.index[r_pos]] = dir_vals[r_pos]
        return overtake_signal


class ReturnBasedAssembler:
    """
    基于收益情况的模型组装器

    输入：各模型原始交易信号（pred.pkl 的 score）+ 股价
    输出：最终持仓比例 DataFrame（index=(datetime, instrument)，列：weight）

    软交易信号语义：soft_signal ∈ [-1, 1]
      - 1.0  = 满仓买入
      - -1.0 = 卖出/做空
      - 0.0  = 空仓
      - 中间值 = 部分仓位

    处理逻辑：
      1) 对每模型×每股票用「原始信号」× 股价模拟交易计算每日收益率：
             ret_t = signal_{t-1} * (price_t / price_{t-1} - 1)
      2) 对同一股票跨模型，计算从股票成立到 t 日的累计收益率（复利），
         第 t 日选择累计收益率最高的模型，其权重为 1，其余模型权重为 0（one-hot）
      3) 按权重加权「软交易信号」得到 Ensemble score（[-1, 1]，供 EvenWeightStrategy 使用）
    """

    def __init__(
        self,
        score_col: str = "score",
        selector: Optional[Union[BaseModelSelector, Dict]] = None,
        enhancer: Optional[Union[BaseSignalEnhancer, Dict]] = None,
        overtake_follow: bool = False,
        overtake_oracle: bool = False,
        overtake_lead_days: int = 1,
        pre_follow_buy_thre: float = 0.01,
        pre_follow_sell_thre: float = 0.01,
    ):
        """
        Parameters
        ----------
        selector : Optional[Union[BaseModelSelector, dict]]
            模型选择策略（Strategy A）。缺省 → DynamicBestSelector（每日最优、模型可切换）。
            传 dict 时用 init_instance_by_config 实例化（如配置 LockedBestSelector 锁定最优）。
        enhancer : Optional[Union[BaseSignalEnhancer, dict]]
            信号增强策略（Strategy B）。缺省且未显式传 overtake_* 平铺参数 → NoOpEnhancer（Null Object）。
            锁定（或自定义 selector）与 overtake 增强互斥；显式传 overtake_* 平铺参数时自动装配
            OvertakeFollowEnhancer，并建议迁移到 `enhancer` 嵌套配置。
        """
        from qlib.utils import init_instance_by_config

        self.score_col = score_col
        self.logger = get_module_logger("ReturnBasedAssembler", level=logging.INFO)

        # ---- Strategy A：模型选择 ----
        if selector is None:
            self.selector: BaseModelSelector = DynamicBestSelector()
        elif isinstance(selector, BaseModelSelector):
            self.selector = selector
        else:
            self.selector = init_instance_by_config(selector, accept_types=BaseModelSelector)

        # ---- Strategy B：信号增强（含旧平铺参数兼容迁移） ----
        migrate_old = (overtake_follow is not False) or (overtake_oracle is not False) \
            or (overtake_lead_days != 1) or (pre_follow_buy_thre != 0.01) or (pre_follow_sell_thre != 0.01)
        if enhancer is None:
            if migrate_old:
                # 兼容旧 YAML：平铺 overtake_* 参数 → 自动组装 OvertakeFollowEnhancer
                warnings.warn(
                    "平铺参数 overtake_follow/overtake_oracle/... 已弃用，请迁移到 `enhancer` 嵌套配置",
                    DeprecationWarning,
                    stacklevel=2,
                )
                self.enhancer: BaseSignalEnhancer = OvertakeFollowEnhancer(
                    overtake_follow=overtake_follow,
                    overtake_oracle=overtake_oracle,
                    overtake_lead_days=overtake_lead_days,
                    pre_follow_buy_thre=pre_follow_buy_thre,
                    pre_follow_sell_thre=pre_follow_sell_thre,
                )
            else:
                self.enhancer = NoOpEnhancer()
        elif isinstance(enhancer, BaseSignalEnhancer):
            self.enhancer = enhancer
        else:
            self.enhancer = init_instance_by_config(enhancer, accept_types=BaseSignalEnhancer)

        # ---- 互斥校验：锁定（非 Dynamic）selector 与 overtake 增强不兼容 ----
        if not isinstance(self.selector, DynamicBestSelector) and not isinstance(self.enhancer, NoOpEnhancer):
            raise ValueError(
                "锁定/自定义 selector 与信号增强（enhancer）互斥：锁定模式请使用 NoOpEnhancer（缺省）"
            )
        self.logger.info(
            "ReturnBasedAssembler initialized: selector=%s, enhancer=%s",
            type(self.selector).__name__,
            type(self.enhancer).__name__,
        )

    @staticmethod
    def _simulate_return(signal: pd.Series, price: pd.Series) -> pd.Series:
        """
        用原始信号模拟交易收益率：
            ret_t = hold_{t-1} * (price_t / price_{t-1} - 1)

        与 EvenWeightStrategy 的 T+1 调仓语义严格对齐（close 收盘价成交）：
          - t 日产生买入(1) / 卖出(-1) 事件，t+1 日收盘按信号调仓
          - 买入：t+1 收盘买入持有，t+2 起计益；买入当日(t+1)涨跌不计
          - 卖出：t+1 收盘卖出清仓，t+1 全天仍持有（成交日涨跌计入）；t+2 起空仓
          - 0 / NaN：保持上一状态（沿用最近的非零事件）

        实现（事件 ffill 后双重 shift）：
          1) event.shift(1)：事件在信号日次日(T+1)收盘生效 -> hold
             hold_t 表示 t 日收盘后的持仓状态，由 t-1 日（ffill 后）的事件决定
          2) ret 中 hold.shift(1)：t 日涨跌归属「t-1 收盘已持有」
        """
        # 0 / NaN 视为"保持"，置 NaN 以便 ffill 传播最近的非零事件
        event = signal.replace(0.0, np.nan)
        event = event.ffill()
        # T+1 收盘生效：t 日持仓状态由 t-1 日（保持后）的事件决定
        hold = event.shift(1).fillna(0.0)
        # -1 事件 -> 空仓状态 0（卖出清仓，不做空）
        hold = hold.replace(-1.0, 0.0)

        # t 日涨跌归属「t-1 收盘已持有」的状态
        price_shift = price.shift(1)
        ret = hold.shift(1) * (price / price_shift - 1.0)
        return ret

    def assemble(
        self,
        soft_signals: Dict[str, pd.DataFrame],
        raw_signals: Dict[str, pd.DataFrame],
        price: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        组装各模型软信号得到最终持仓比例

        Parameters
        ----------
        soft_signals : Dict[str, pd.DataFrame]
            {model_name: DataFrame(index=(datetime, instrument), 列含 soft_signal)}
        raw_signals : Dict[str, pd.DataFrame]
            {model_name: DataFrame(index=(datetime, instrument), 列含 score)}
        price : pd.DataFrame
            index=(datetime, instrument)，列含 close

        Returns
        -------
        pd.DataFrame
            index=(datetime, instrument)，列：
            - score：Ensemble 综合信号（one-hot × soft，[-1, 1]，供 EvenWeightStrategy 使用）
            - {model_name}_score：各模型原始信号（1/0/-1）
            - {model_name}_soft：各模型软信号（[-1, 1]）
            - {model_name}_cumret：各模型累计收益率（复利）
        """
        model_names = list(soft_signals.keys())
        if len(model_names) == 0:
            raise ValueError("No models provided to assembler")

        # 统一 index 顺序
        price = price.sort_index()
        close = price["close"].astype(float)
        all_index = price.index

        # 1) 对每模型×每股票用原始信号模拟交易算每日收益率，
        #    并计算从股票成立到 t 日的累计收益率（复利）
        cum_ret = {}  # {model_name: pd.Series(index=all_index, value=cumulative_return)}
        for name in model_names:
            raw = raw_signals[name].sort_index()
            signal = raw[self.score_col].astype(float)
            # 对齐到统一 index
            signal = signal.reindex(all_index).fillna(0.0)

            cum_ret_series = pd.Series(0.0, index=all_index, dtype=float)
            # 逐股票计算累计收益率（保持时间顺序）
            for instrument, grp in signal.groupby(level="instrument"):
                grp_price = close.loc[grp.index]
                ret = self._simulate_return(grp, grp_price)
                # 累计收益率（复利）：从股票成立到 t 日
                cum = (1.0 + ret.fillna(0.0)).cumprod() - 1.0
                cum_ret_series.loc[grp.index] = cum
            cum_ret[name] = cum_ret_series

        # 2) 模型选择策略（Strategy A）：生成 one-hot 权重
        cum_ret_df = pd.DataFrame(cum_ret)  # index=all_index, columns=model_names
        weight_df = self.selector.build_weight_df(cum_ret_df)

        # 2.5 信号增强策略（Strategy B）：可选强制覆盖 score=±1（如提前跟风/反超）
        #     实现逻辑见 BaseSignalEnhancer 子类（OvertakeFollowEnhancer / NoOpEnhancer）
        overtake_signal = self.enhancer.enhance(cum_ret_df)

        # 3) 按权重加权软信号得到持仓比例
        soft_aligned = {}
        for name in model_names:
            soft = soft_signals[name].sort_index()["soft_signal"].astype(float)
            soft_aligned[name] = soft.reindex(all_index).fillna(0.0)
        soft_matrix = pd.DataFrame(soft_aligned)  # index=all_index, columns=model_names
        final = (soft_matrix.values * weight_df.values).sum(axis=1)

        # 反超跟风信号覆盖：切换日强制 score=±1
        if overtake_signal is not None:
            mask = overtake_signal.values != 0
            final = np.where(mask, overtake_signal.values, final)

        # 3b) selector 直接信号覆盖：如 FutureBestSelector（跟随最优模型持仓状态生成 ±1/0）
        #     可完全替代 one-hot×soft 加权，实现"最优模型决定当日交易信号"
        selector_signal = self.selector.select_signal(
            cum_ret_df, raw_signals=raw_signals, close=close
        )
        if selector_signal is not None:
            final = selector_signal.reindex(all_index).fillna(0.0).values

        result = pd.DataFrame({"score": final}, index=all_index)

        # 3c) 当前主导模型（best_model）：由 selector 输出，下游无需重算
        best_model = getattr(self.selector, "_last_best_model", None)
        if best_model is not None:
            result["best_model"] = best_model.reindex(all_index).fillna("")

        # 4) 附加各模型原始信号、软信号、累计收益率（供可视化分析使用）
        for name in model_names:
            result[f"{name}_score"] = raw_signals[name].sort_index()[self.score_col].astype(float).reindex(all_index).fillna(0.0)
            result[f"{name}_soft"] = soft_aligned[name]
            result[f"{name}_cumret"] = cum_ret[name]

        return result


class EMEnsembleModel(Model):
    """
    多模型 Ensemble 综合持仓信号模型

    参数
    ----
    models : List[dict]
        模型列表，每个元素为 dict：
            {
                "name": str,                       # 模型名
                "interpreter": {                   # 模型解释器 cfg
                    "class": str,
                    "module_path": str,
                    "kwargs": {...},
                },
            }
    assembler : dict
        模型组装器 cfg：
            {
                "class": str,
                "module_path": str,
                "kwargs": {...},
            }
    lookback_days : int
        预测时向前加载的历史数据天数（默认 10000，足够触达 handler 的 start_time，
        使 ReturnBasedAssembler 的 cum_ret 能从股票数据起点复利累计）。
        test 段只需写实际回测区间，前序数据由本参数自动加载。
    """

    def __init__(
        self,
        models: List[dict],
        assembler: dict,
        price_field: str = "close",
        lookback_days: int = 10000,
        annualized_filter: Optional[dict] = None,
        min_pred_coverage: float = 0.8,
    ):
        from qlib.utils import init_instance_by_config

        self.models = models
        self.assembler_cfg = assembler
        self.price_field = price_field
        # 收益率过滤：{"enable": True, "min_5y_annualized": 0.08, "min_inception_annualized": 0.09}
        # 不达标（近5年年化<=阈值 且 成立年化<=阈值）的股票，score 强制置 -1
        self.annualized_filter = annualized_filter or {}
        # 预测时向前加载的历史数据天数：
        # - 组装器内部用含前序的完整价格计算 cum_ret（从股票数据起点复利累计）
        # - 最终输出按 segment 区间裁剪（仅保留回测区间）
        self.lookback_days = int(lookback_days)
        # 参考模型（ValuationInterpreter）pred 数据完整性阈值：
        # interpret() 输出 reindex 到 ensemble 运行区间（price.index）后非空行的最低比例。
        # exp_path 指向只含部分区间的旧 run 时会触发 ValueError，防止产生全 0 信号的脏结果。
        self.min_pred_coverage = float(min_pred_coverage)
        self.logger = get_module_logger("EMEnsembleModel", level=logging.INFO)

        # 初始化解释器
        self.interpreters = {}
        for m in models:
            name = m["name"]
            interp_cfg = m["interpreter"]
            self.interpreters[name] = init_instance_by_config(interp_cfg)

        # 初始化组装器
        self.assembler = init_instance_by_config(assembler)

    def fit(self, dataset: DatasetH, reweighter=None) -> None:
        """EMEnsemble 无参数，fit 为空操作"""
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

    def _load_price(self, dataset: DatasetH) -> pd.DataFrame:
        """从 dataset 加载价格数据（含前序 lookback_days 历史，供 cum_ret 从起点复利）"""
        df = dataset.prepare(
            "test",
            col_set="feature",
            data_key=DataHandlerLP.DK_I,
            lookback_days=self.lookback_days,
        )
        if self.price_field not in df.columns:
            raise ValueError(f"dataset missing price field '{self.price_field}'")
        return df[[self.price_field]]

    def _apply_annualized_filter(self, price, result):
        """keep if 5y-annualized > min_5y or inception-annualized > min_incep, else score = -1"""
        cfg = self.annualized_filter
        if not cfg.get("enable", False):
            return
        min_5y = float(cfg.get("min_5y_annualized", 0.08))
        min_incep = float(cfg.get("min_inception_annualized", 0.09))
        min_5y_dur = float(cfg.get("min_5y_duration_days", 183))
        min_incep_dur = float(cfg.get("min_incep_duration_days", 365))
        close = price["close"].astype(float)
        insts = result.index.get_level_values("instrument")
        keep = np.zeros(len(result), dtype=bool)
        for inst, grp in close.groupby(level="instrument"):
            if len(grp) < 2:
                continue
            dates = pd.DatetimeIndex(grp.index.get_level_values("datetime")) if isinstance(grp.index, pd.MultiIndex) else pd.DatetimeIndex(grp.index)
            vals = grp.values.astype(float)
            n = len(vals)
            d0 = (dates - dates[0]).days.astype(float).to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                incep = (vals / vals[0]) ** (365.0 / d0) - 1.0
                incep[d0 <= 0] = np.nan
            cut = dates - pd.DateOffset(years=5)
            idx5 = np.searchsorted(dates.values, cut.values, side="right") - 1
            valid5 = idx5 >= 0
            ann5 = np.full(n, np.nan)
            if valid5.any():
                i = valid5
                days5 = (dates[i] - dates[idx5[i]]).days.astype(float).to_numpy()
                ann5[i] = (vals[i] / vals[idx5[i]]) ** (365.0 / days5) - 1.0
                ann5[i] = np.where(days5 <= 0, np.nan, ann5[i])
            qual5 = ~np.isnan(ann5) & (ann5 > min_5y)
            quin = ~np.isnan(incep) & (incep > min_incep)
            dur5 = np.zeros(n)
            dur_in = np.zeros(n)
            for t in range(1, n):
                dd = float((dates[t] - dates[t - 1]).days)
                dur5[t] = dur5[t - 1] + dd if qual5[t] else 0.0
                dur_in[t] = dur_in[t - 1] + dd if quin[t] else 0.0
            ks = (dur5 >= min_5y_dur) | (dur_in >= min_incep_dur)
            keep[insts == inst] = ~np.isnan(ks) & ks
        km = keep
        result["score"] = np.where(km, result["score"], -1.0)
        self.logger.info(
            "AnnualizedFilter: %d rows -> -1 (keep %.1f%%)",
            int((~km).sum()),
            100.0 * km.mean() if len(km) else 0.0,
        )

    def predict(self, dataset: DatasetH, segment: Union[str, slice] = "test") -> pd.DataFrame:
        """
        生成最终持仓比例信号 DataFrame

        Returns
        -------
        pd.DataFrame
            索引为 (datetime, instrument)，列：
            - score：Ensemble 综合信号（one-hot × soft，[-1, 1]，供 EvenWeightStrategy 使用）
            - {model_name}_score：各模型原始信号（1/0/-1）
            - {model_name}_soft：各模型软信号（[-1, 1]）
            - {model_name}_cumret：各模型累计收益率（复利）
        """
        seg_start, seg_end = self._parse_segment(segment, dataset)

        # 加载价格（含前序 lookback_days 历史）
        price = self._load_price(dataset)

        # 1) 遍历模型，分别进行模型解释
        #    interpret() 一次加载 pred.pkl，同时产出软信号与原始信号，避免重复 IO
        soft_signals = {}
        raw_signals = {}
        for m in self.models:
            name = m["name"]
            interp = self.interpreters[name]
            self.logger.info("Interpreting model: %s", name)
            if isinstance(interp, ConstantInterpreter):
                # 恒定信号基准模型（ALL in / 绝对不买入）：无 exp_path，直接按价格 index 生成恒定信号
                soft = interp.interpret(index=price.index)
            elif isinstance(interp, MACDSignalInterpreter):
                # MACD 趋势信号：无 exp_path，由价格（含 lookback 前序历史）实时计算
                soft = interp.interpret(price=price)
            else:
                # ValuationInterpreter：从 exp_path 加载 pred.pkl
                soft = interp.interpret()
            # 参考模型数据完整性检查：exp_path 指向的 pred 必须覆盖 ensemble 运行区间。
            # 若只含部分区间（如 daily_update 误指向仅含后半段的 run），reindex 后大量行
            # 被 fillna(0.0) 填充，导致 {name}_score 几乎全 0 的脏结果，这里直接 raise。
            if not isinstance(interp, (ConstantInterpreter, MACDSignalInterpreter)):
                aligned = soft[interp.score_col].reindex(price.index)
                coverage = float(aligned.notna().mean())
                exp_path = getattr(interp, "exp_path", "N/A")
                if coverage < self.min_pred_coverage:
                    raise ValueError(
                        f"[EMEnsembleModel] 参考模型 '{name}' (exp_path={exp_path}) 数据不完整："
                        f"pred 覆盖 ensemble 运行区间的比例仅 {coverage:.2%} "
                        f"(阈值 {self.min_pred_coverage:.0%})。"
                        f"pred 区间无法覆盖回测区间，reindex 后会产生大量 0 信号，"
                        f"请检查 exp_path 是否指向包含完整历史区间的最新 run。"
                    )
                self.logger.info(
                    "Model '%s' pred coverage on ensemble range: %.2f%% (exp_path=%s)",
                    name,
                    coverage * 100.0,
                    exp_path,
                )
            soft_signals[name] = soft[["soft_signal"]]
            # 原始信号（score）用于组装器算收益
            raw_signals[name] = soft[[interp.score_col]]

        # 2) 组装各模型得到最终交易信号（持仓比例）
        self.logger.info("Assembling models: %s", list(soft_signals.keys()))
        result = self.assembler.assemble(
            soft_signals=soft_signals,
            raw_signals=raw_signals,
            price=price,
        )

        # 2.5) 收益率过滤（近5年年化>8% 或 成立年化>9% 保留，否则 score=-1）
        self._apply_annualized_filter(price, result)

        # 3) 将输出裁剪到 segment 区间（去除前序 lookback 数据）
        #    - 组装器内部已用含前序的全量价格计算 cum_ret（从股票数据起点复利累计）
        #    - 裁剪只影响最终预测输出（pred.pkl），使回测/报表与实际回测区间对齐
        self.logger.info(
            "Assembled prediction with lookback_days=%d, rows: %d",
            self.lookback_days,
            len(result),
        )
        if seg_start is not None and seg_end is not None:
            seg_start = pd.Timestamp(seg_start)
            seg_end = pd.Timestamp(seg_end)
            dates = result.index.get_level_values(0)
            mask = pd.Series(True, index=result.index)
            mask &= dates >= seg_start
            mask &= dates <= seg_end
            result = result[mask]
            self.logger.info(
                "Trimmed prediction output to [%s, %s], rows: %d -> %d",
                seg_start,
                seg_end,
                len(dates),
                len(result),
            )

        return result