# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
MACD TopK Strategy Implementation
基于MACD指标的TopK均匀持仓交易策略

策略逻辑:
1. 持有TopK只股票，每只分配1/K的资金
2. 每只股票独立运行MACD策略（金叉买入，死叉卖出）
"""

import numpy as np
import pandas as pd
import os
from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO


class MACDStrategy(BaseStrategy):
    """
    MACD TopK 均匀持仓策略

    - 持有TopK只股票，每只分配1/K资金
    - 每只股票独立运行MACD策略
    - 金叉买入，死叉卖出
    """

    def __init__(
        self,
        topk_stocks: list = None,
        top_k: int = 5,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        save_dir: str = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.topk_stocks = topk_stocks or []
        self.top_k = top_k
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.save_dir = save_dir

        # 每只股票独立的MACD状态
        self.macd_states = {}
        for stock in self.topk_stocks:
            self.macd_states[stock] = {
                "ema_fast": None,
                "ema_slow": None,
                "ema_signal": None,
                "prev_dif": None,
                "prev_dea": None,
                "day_count": 0,
            }

        # 交易记录
        self.trade_records = []

    def generate_trade_decision(self, execute_result=None):
        """根据各股票的MACD信号生成交易决策"""
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

            state = self.macd_states[stock]
            state["day_count"] += 1

            # 更新EMA
            if state["ema_fast"] is None:
                state["ema_fast"] = price
                state["ema_slow"] = price
            else:
                k_fast = 2 / (self.fast_period + 1)
                k_slow = 2 / (self.slow_period + 1)
                state["ema_fast"] = price * k_fast + state["ema_fast"] * (1 - k_fast)
                state["ema_slow"] = price * k_slow + state["ema_slow"] * (1 - k_slow)

            # 计算DIF
            dif = state["ema_fast"] - state["ema_slow"]

            # 更新DEA
            if state["ema_signal"] is None:
                state["ema_signal"] = dif
            else:
                k_signal = 2 / (self.signal_period + 1)
                state["ema_signal"] = dif * k_signal + state["ema_signal"] * (1 - k_signal)

            dea = state["ema_signal"]
            histogram = dif - dea

            # 获取当前持仓
            current_position = self.trade_position.get_stock_amount(stock)
            warmup_done = state["day_count"] >= self.slow_period + self.signal_period

            signal = "HOLD"
            trade_amount = 0

            # 检测金叉/死叉
            if warmup_done and state["prev_dif"] is not None and state["prev_dea"] is not None:
                golden_cross = state["prev_dif"] < state["prev_dea"] and dif > dea
                death_cross = state["prev_dif"] > state["prev_dea"] and dif < dea

                # 金叉且当前空仓：买入使该股达到目标市值
                if golden_cross and current_position == 0:
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

                # 死叉且当前持仓：全部卖出
                elif death_cross and current_position > 0:
                    orders.append(Order(
                        stock_id=stock,
                        amount=current_position,
                        start_time=trade_start_time,
                        end_time=trade_end_time,
                        direction=OrderDir.SELL,
                    ))
                    signal = "SELL"
                    trade_amount = current_position

            state["prev_dif"] = dif
            state["prev_dea"] = dea

            # 记录
            self.trade_records.append({
                "date": current_date,
                "stock": stock,
                "price": price,
                "dif": dif,
                "dea": dea,
                "histogram": histogram,
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
            df.to_csv(os.path.join(self.save_dir, "macd_records.csv"), index=False)
            print(f"交易记录已保存: {self.save_dir}/macd_records.csv")
