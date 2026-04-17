# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
TopK ALL-IN Strategy Implementation
TopK均匀持仓ALL-IN策略：在第一个交易日均匀买入TopK只标的，之后从不卖出
"""

from qlib.strategy.base import BaseStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO


class TopKAllInStrategy(BaseStrategy):
    """
    TopK ALL-IN买入持有策略

    在第一个交易日均匀买入TopK只标的（每只分配1/K的资金），之后从不卖出
    """

    def __init__(
        self,
        topk_stocks: list = None,
        top_k: int = 1,
        **kwargs
    ):
        """
        Args:
            topk_stocks: TopK股票代码列表，如 ["SH600519", "SZ000002"]
            top_k: TopK数量
        """
        super().__init__(**kwargs)
        self.topk_stocks = topk_stocks or []
        self.top_k = top_k
        self.has_bought = False

    def generate_trade_decision(
        self,
        execute_result=None,
    ):
        """生成交易决策"""
        # 如果已经买入，不再交易
        if self.has_bought:
            return TradeDecisionWO([], self)

        # 获取交易股票列表
        if not self.topk_stocks:
            return TradeDecisionWO([], self)

        stocks = self.topk_stocks

        # 获取当前时间和交易信息
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)

        # 获取当前现金
        current_cash = self.trade_position.get_cash()

        # 每只股票分配的金额（均匀分配）
        cash_per_stock = current_cash / len(stocks) * 0.99  # 留1%作为手续费缓冲

        orders = []
        for stock in stocks:
            # 获取目标股票当前价格
            try:
                price = self.trade_exchange.get_close(stock, trade_start_time, trade_end_time)
                if price is None or price <= 0:
                    continue
            except Exception:
                continue

            # 计算可买入数量（按手100股取整）
            amount = int(cash_per_stock / price / 100) * 100

            if amount <= 0:
                continue

            # 生成买入订单
            order = Order(
                stock_id=stock,
                amount=amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=OrderDir.BUY,
            )
            orders.append(order)

        self.has_bought = True
        return TradeDecisionWO(orders, self)
