# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
KDJ TopK Strategy Implementation
基于KDJ指标的TopK均匀持仓交易策略

策略逻辑:
1. 持有TopK只股票，每只分配1/K的资金
2. 每只股票独立运行KDJ策略（超卖区金叉买入，超买区死叉卖出）
"""

import numpy as np
import pandas as pd
import os
from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO


class KDJStrategy(BaseStrategy):
    """
    KDJ TopK 均匀持仓策略

    - 持有TopK只股票，每只分配1/K资金
    - 每只股票独立运行KDJ策略
    - 金叉买入，死叉卖出（带超买超卖过滤）
    """

    def __init__(
        self,
        topk_stocks: list = None,
        top_k: int = 5,
        n_period: int = 9,       # RSV计算周期
        k_smooth: int = 3,       # K线平滑次数
        d_smooth: int = 3,       # D线平滑次数
        overbuy: int = 80,       # 超买阈值
        oversell: int = 20,      # 超卖阈值
        save_dir: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.topk_stocks = topk_stocks or []
        self.top_k = top_k
        self.n_period = n_period
        self.k_smooth = k_smooth
        self.d_smooth = d_smooth
        self.overbuy = overbuy
        self.oversell = oversell
        self.save_dir = save_dir

        # 每只股票独立的KDJ状态
        self.kdj_states = {}
        for stock in self.topk_stocks:
            self.kdj_states[stock] = {
                "k": 50.0,      # 初始值50
                "d": 50.0,      # 初始值50
                "j": 50.0,
                "prev_k": None,
                "prev_d": None,
                "day_count": 0,
                "low_n": None,
                "high_n": None,
                "close_history": [],  # 存储最近N天的收盘价，用于计算高低点
            }

        # 交易记录
        self.trade_records = []

    def _calculate_rsv(self, stock, close_price, trade_start_time, trade_end_time):
        """计算RSV并更新KDJ指标"""
        state = self.kdj_states[stock]

        # 记录收盘价历史
        state["close_history"].append(close_price)
        if len(state["close_history"]) > self.n_period:
            state["close_history"].pop(0)

        # RSV计算需要N天数据
        if len(state["close_history"]) < self.n_period:
            return None, None, None

        # 获取最近N天的最低价和最高价（从close_history中获取）
        recent_low = min(state["close_history"])
        recent_high = max(state["close_history"])

        # 计算RSV
        if recent_high == recent_low:
            rsv = 50.0
        else:
            rsv = (close_price - recent_low) / (recent_high - recent_low) * 100

        # 更新K值: 2/3*昨日K + 1/3*今日RSV
        k = (2 * state["k"] + 1 * rsv) / 3

        # 更新D值: 2/3*昨日D + 1/3*今日K
        d = (2 * state["d"] + 1 * k) / 3

        # 计算J值: 3K - 2D
        j = 3 * k - 2 * d

        return k, d, j

    def generate_trade_decision(self, execute_result=None):
        """根据各股票的KDJ信号生成交易决策"""
        if not self.topk_stocks:
            return TradeDecisionWO([], self)

        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        current_date = trade_start_time

        # 计算总资产
        total_value = self.trade_position.get_cash()
        for stock in self.topk_stocks:
            try:
                price = self.trade_exchange.get_close(stock, trade_start_time, trade_end_time)
                if price and not np.isnan(price) and price > 0:
                    total_value += self.trade_position.get_stock_amount(stock) * price
            except:
                pass

        # 每只股票目标市值
        target_value_per_stock = total_value / self.top_k
        orders = []

        for stock in self.topk_stocks:
            try:
                price = self.trade_exchange.get_close(stock, trade_start_time, trade_end_time)
                if price is None or np.isnan(price) or price <= 0:
                    continue
            except Exception:
                continue

            state = self.kdj_states[stock]
            state["day_count"] += 1

            # 计算KDJ
            k, d, j = self._calculate_rsv(stock, price, trade_start_time, trade_end_time)

            # 获取当前持仓
            current_position = self.trade_position.get_stock_amount(stock)
            warmup_done = len(state["close_history"]) >= self.n_period

            signal = "HOLD"
            trade_amount = 0

            # 检测金叉/死叉
            if warmup_done and state["prev_k"] is not None and state["prev_d"] is not None:
                prev_k = state["prev_k"]
                prev_d = state["prev_d"]

                # 金叉：K从下向上穿越D
                golden_cross = prev_k < prev_d and k > d
                # 死叉：K从上向下穿越D
                death_cross = prev_k > prev_d and k < d

                # 金叉且当前空仓：在超卖区附近买入
                if golden_cross and current_position == 0:
                    # 当K值在超卖区附近（<30）时买入信号更强
                    if k < self.oversell or (state["k"] < 30 and k < 50):
                        target_shares = target_value_per_stock / price * 0.99
                        if target_shares > 0:
                            orders.append(Order(
                                stock_id=stock,
                                amount=target_shares,
                                start_time=trade_start_time,
                                end_time=trade_end_time,
                                direction=OrderDir.BUY,
                            ))
                            signal = "BUY"
                            trade_amount = target_shares

                # 死叉且当前持仓：在超买区附近卖出
                elif death_cross and current_position > 0:
                    # 当K值在超买区附近（>70）或J值过高时卖出
                    if k > self.overbuy or (state["k"] > 70 and k > 50) or j > 100:
                        orders.append(Order(
                            stock_id=stock,
                            amount=current_position,
                            start_time=trade_start_time,
                            end_time=trade_end_time,
                            direction=OrderDir.SELL,
                        ))
                        signal = "SELL"
                        trade_amount = current_position

            # 更新状态
            if k is not None and d is not None:
                state["prev_k"] = state["k"]
                state["prev_d"] = state["d"]
                state["k"] = k
                state["d"] = d
                state["j"] = j

            # 记录
            self.trade_records.append({
                "date": current_date,
                "stock": stock,
                "price": price,
                "k": k if k is not None else 0,
                "d": d if d is not None else 0,
                "j": j if j is not None else 0,
                "signal": signal,
                "position": current_position,
                "trade_amount": trade_amount,
            })

        return TradeDecisionWO(orders, self)

    def save_records(self):
        """保存交易记录"""
        if self.save_dir and self.trade_records:
            os.makedirs(self.save_dir, exist_ok=True)
            df = pd.DataFrame(self.trade_records)
            df.to_csv(os.path.join(self.save_dir, "kdj_records.csv"), index=False)
            print(f"交易记录已保存: {self.save_dir}/kdj_records.csv")
