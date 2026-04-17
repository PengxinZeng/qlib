#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MACD TopK Strategy Workflow
MACD TopK均匀持仓策略工作流

策略逻辑:
1. TRAIN Set: 在训练集上用MACD策略回测所有股票，选出TOP K只
2. VAL Set: 在验证集上选择最优K值
3. EVAL Set: 在测试集上使用最优K进行评测

Usage:
    python workflow_macd.py run --max_k=10
"""

import os
import sys
import fire
import yaml
import pandas as pd
import numpy as np
import qlib
from qlib.data import D
from qlib.backtest import backtest as qlib_backtest
from qlib.contrib.report import analysis_position

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config():
    """从config.yaml加载配置"""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    return {
        "provider_uri": yaml_config["qlib_init"]["provider_uri"],
        "region": yaml_config["qlib_init"]["region"],
        "fast_period": yaml_config["macd_params"]["fast_period"],
        "slow_period": yaml_config["macd_params"]["slow_period"],
        "signal_period": yaml_config["macd_params"]["signal_period"],
        "train_start": yaml_config["train_period"]["start_time"],
        "train_end": yaml_config["train_period"]["end_time"],
        "valid_start": yaml_config.get("valid_period", {}).get("start_time", "2020-10-26"),
        "valid_end": yaml_config.get("valid_period", {}).get("end_time", "2023-08-07"),
        "test_start": yaml_config["test_period"]["start_time"],
        "test_end": yaml_config["test_period"]["end_time"],
        "benchmark": yaml_config["backtest"]["benchmark"],
        "account": yaml_config["backtest"]["account"],
    }


INSTRUMENT_INFO = {
    "510050": "上证50ETF", "510300": "沪深300ETF", "510500": "中证500ETF",
    "518880": "黄金ETF", "512690": "酒ETF", "513100": "纳斯达克ETF",
    "SH600519": "贵州茅台", "SH600036": "招商银行", "SZ000333": "美的集团",
    "SZ300750": "宁德时代", "SZ002594": "比亚迪", "SH600900": "长江电力",
    "SH600030": "中信证券", "SZ000002": "万科A", "SH600031": "三一重工",
    "SH600309": "万华化学", "SH601012": "隆基绿能", "SH600276": "恒瑞医药",
    "SH600588": "用友网络", "SH601633": "长城汽车", "SH600887": "伊利股份",
    "SZ002415": "海康威视", "SH600104": "上汽集团", "SH601888": "中国中免",
    "159915": "创业板ETF", "159919": "深100ETF", "159922": "中证500ETF",
    "510310": "沪深300ETF", "510180": "上证180ETF", "510880": "红利ETF",
    "512010": "医药ETF", "512100": "中证1000ETF", "512480": "半导体ETF",
    "512690": "酒ETF", "512710": "军工ETF", "513030": "德国ETF",
    "513100": "纳斯达克ETF", "513500": "标普500ETF", "513600": "港股通ETF",
    "560010": "科创50ETF", "159601": "A50ETF", "159812": "新能源ETF",
    "159845": "地产ETF", "159850": "银行ETF", "159869": "游戏ETF",
}


def compute_macd_returns_batch(close_df, fast_period=12, slow_period=26, signal_period=9,
                              initial_cash=1000000, cost_rate=0.0006):
    """
    批量向量化计算所有股票的MACD策略收益（优化版）

    Args:
        close_df: DataFrame, index=datetime, columns=instruments, values=close prices
        initial_cash: 每只股票分配的初始资金

    Returns:
        pd.Series: {symbol: return}
    """
    import time
    t0 = time.time()

    warmup = slow_period + signal_period
    results = {}

    for col in close_df.columns:
        prices = close_df[col].dropna()
        if len(prices) < warmup + 10:
            results[col] = 0.0
            continue

        try:
            # 向量化计算EMA
            ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
            ema_slow = prices.ewm(span=slow_period, adjust=False).mean()
            dif = ema_fast - ema_slow
            dea = dif.ewm(span=signal_period, adjust=False).mean()

            # 向量化金叉死叉
            prev_dif = dif.shift(1)
            prev_dea = dea.shift(1)
            golden_cross = (prev_dif < prev_dea) & (dif > dea)
            death_cross = (prev_dif > prev_dea) & (dif < dea)

            # 向量化模拟交易
            cash = initial_cash
            position = 0
            price_arr = prices.values
            gc_arr = golden_cross.values
            dc_arr = death_cross.values

            for i in range(warmup, len(prices)):
                price = price_arr[i]
                if gc_arr[i] and position == 0:
                    amount = cash * 0.99 / price
                    if amount > 0:
                        cash -= amount * price * (1 + cost_rate)
                        position = amount
                elif dc_arr[i] and position > 0:
                    cash += position * price * (1 - cost_rate)
                    position = 0

            final_value = cash + position * price_arr[-1]
            results[col] = (final_value - initial_cash) / initial_cash
        except:
            results[col] = 0.0

    print(f"  计算耗时: {time.time()-t0:.1f}秒")
    return pd.Series(results)


def select_topk_stocks(config):
    """在训练集上用MACD策略选出TopK股票"""
    print(f"\n{'='*60}")
    print(f"TRAIN: [{config['train_start']} ~ {config['train_end']}]")
    print(f"MACD({config['fast_period']},{config['slow_period']},{config['signal_period']})")
    print(f"{'='*60}")

    inst_dict = D.list_instruments(D.instruments("all"),
                                    start_time=config["train_start"],
                                    end_time=config["train_end"])
    inst_list = list(inst_dict.keys())
    print(f"候选标的: {len(inst_list)} 个")

    # 批量加载数据
    print("加载数据...")
    close_df = D.features(inst_list, ["$close"],
                          start_time=config["train_start"],
                          end_time=config["train_end"],
                          freq="day")["$close"].unstack(level=0)
    print(f"数据形状: {close_df.shape}")

    # 批量计算MACD收益
    print("计算MACD收益...")
    returns_series = compute_macd_returns_batch(
        close_df,
        fast_period=config["fast_period"],
        slow_period=config["slow_period"],
        signal_period=config["signal_period"],
        initial_cash=config["account"],
    )

    # 构建结果
    df_results = returns_series.reset_index()
    df_results.columns = ["symbol", "total_return"]
    df_results["name"] = df_results["symbol"].map(lambda x: INSTRUMENT_INFO.get(x, x))
    df_results = df_results.sort_values("total_return", ascending=False)

    print(f"\nTRAIN MACD收益 TOP 20:")
    print("-" * 70)
    for _, row in df_results.head(20).iterrows():
        print(f"  {row['symbol']:12s} {row['name']:12s} 收益: {row['total_return']*100:8.2f}%")

    return df_results


def calculate_portfolio_return(df_stocks, top_k, val_start, val_end, config):
    """计算TopK组合在验证期间的MACD策略收益"""
    topk_symbols = df_stocks.head(top_k)["symbol"].tolist()

    try:
        close_df = D.features(topk_symbols, ["$close"],
                              start_time=val_start,
                              end_time=val_end,
                              freq="day")["$close"].unstack(level=0)
    except:
        return 0.0

    per_stock_cash = config["account"] / top_k

    # 使用批量函数
    returns_series = compute_macd_returns_batch(
        close_df,
        fast_period=config["fast_period"],
        slow_period=config["slow_period"],
        signal_period=config["signal_period"],
        initial_cash=per_stock_cash,
    )

    valid_returns = returns_series[returns_series != 0]
    return valid_returns.mean() if len(valid_returns) > 0 else 0.0


def select_best_k(df_stocks, config, max_k=10):
    """在验证集上选择最优K值"""
    print(f"\n{'='*60}")
    print(f"VAL: [{config['valid_start']} ~ {config['valid_end']}] 选择最优K")
    print(f"{'='*60}")

    k_results = []
    max_possible_k = min(max_k, len(df_stocks))

    for k in range(1, max_possible_k + 1):
        portfolio_return = calculate_portfolio_return(df_stocks, k,
                                                      config["valid_start"],
                                                      config["valid_end"], config)
        k_results.append({
            "k": k,
            "portfolio_return": portfolio_return,
            "portfolio_return_pct": portfolio_return * 100,
        })
        print(f"  K={k:2d}: 组合收益={portfolio_return*100:7.2f}%")

    df_k_results = pd.DataFrame(k_results)
    best_idx = df_k_results["portfolio_return"].idxmax()
    best_k = int(df_k_results.loc[best_idx, "k"])
    best_return = df_k_results.loc[best_idx, "portfolio_return"]

    print(f"\n>>> 最优 K={best_k}, VAL组合收益={best_return*100:.2f}%")
    return best_k, df_k_results


def run_backtest(df_stocks, best_k, config):
    """使用MACD TopK策略进行回测"""
    from strategy import MACDStrategy

    topk_stocks = df_stocks.head(best_k)["symbol"].tolist()

    print(f"\n{'='*60}")
    print(f"EVAL: [{config['test_start']} ~ {config['test_end']}]")
    print(f"TopK={best_k}, 标的: {topk_stocks}")
    print(f"{'='*60}")

    strategy = MACDStrategy(
        topk_stocks=topk_stocks,
        top_k=best_k,
        fast_period=config["fast_period"],
        slow_period=config["slow_period"],
        signal_period=config["signal_period"],
    )

    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        }
    }

    portfolio_metric_dict, indicator_dict = qlib_backtest(
        start_time=config["test_start"],
        end_time=config["test_end"],
        strategy=strategy,
        executor=executor_config,
        benchmark=config["benchmark"],
        account=config["account"],
        exchange_kwargs={
            "limit_threshold": 0.10,
            "deal_price": "close",
            "open_cost": 0.0003,
            "close_cost": 0.0003,
            "min_cost": 5,
            "trade_unit": 0.001,
        },
    )

    return portfolio_metric_dict, indicator_dict, topk_stocks, strategy


def plot_macd_analysis(topk_stocks, strategy, config, save_dir):
    """可视化各持仓股票的MACD和买卖点"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Warning: plotly not installed, skipping visualization")
        return

    trade_records = strategy.trade_records
    if not trade_records:
        print("Warning: No trade records found")
        return

    df_records = pd.DataFrame(trade_records)
    df_records['date'] = pd.to_datetime(df_records['date'])
    df_records.set_index('date', inplace=True)

    report_df_path = os.path.join(save_dir, "report_df.csv")
    if os.path.exists(report_df_path):
        report_df = pd.read_csv(report_df_path, index_col=0, parse_dates=True)
        report_df["cum_return"] = (1 + report_df["return"]).cumprod() - 1
    else:
        report_df = None

    n_stocks = len(topk_stocks)
    n_rows = n_stocks + 2  # +1 for holdings stack +1 for returns

    subplot_titles = ["持仓金额分布（堆叠面积图）"]
    for stock in topk_stocks:
        name = INSTRUMENT_INFO.get(stock, stock)
        subplot_titles.append(f"{name} ({stock})")
    subplot_titles.append("累计收益率")

    # 启用 secondary_y 用于 MACD 双轴显示
    specs = [[{}]]  # 第一行：持仓分布
    for _ in range(n_stocks):
        specs.append([{"secondary_y": True}])  # 股票详情：主Y轴价格，次Y轴MACD
    specs.append([{}])  # 最后一行：收益

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=subplot_titles,
        specs=specs
    )

    # Row 1: 持仓金额堆叠面积图（包含现金）
    position_pivot = df_records.pivot_table(
        index='date', columns='stock', values='position', aggfunc='last'
    ).fillna(0)
    price_pivot = df_records.pivot_table(
        index='date', columns='stock', values='price', aggfunc='last'
    )

    # 计算持仓市值
    position_value = position_pivot * price_pivot
    position_value = position_value.fillna(0)

    # 计算现金 = 总资产 - 总持仓市值
    total_position_value = position_value.sum(axis=1)

    # 获取每日总资产（从回测记录中）
    # 由于策略记录中没有总资产，我们用初始资金 + 累计收益来估算
    # 或者从report_df中获取
    if report_df is not None:
        # 计算总资产
        initial_account = config.get("account", 1000000)
        portfolio_value = initial_account * (1 + report_df["cum_return"])
        cash_value = portfolio_value - total_position_value.reindex(portfolio_value.index, method='ffill')
    else:
        # 如果没有report_df，假设现金为0
        cash_value = pd.Series(0, index=total_position_value.index)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    for idx, stock in enumerate(topk_stocks):
        if stock in position_value.columns:
            fig.add_trace(
                go.Scatter(
                    x=position_value.index,
                    y=position_value[stock],
                    name=f"{stock} 持仓",
                    stackgroup='one',
                    fillcolor=colors[idx % len(colors)],
                    line=dict(width=0),
                    showlegend=True,
                    legendgroup="holdings",
                ),
                row=1, col=1
            )

    # 添加现金
    fig.add_trace(
        go.Scatter(
            x=cash_value.index,
            y=cash_value.values,
            name="现金",
            stackgroup='one',
            fillcolor='#808080',
            line=dict(width=0),
            showlegend=True,
            legendgroup="holdings",
        ),
        row=1, col=1
    )

    # 总持仓市值（不含现金）
    fig.add_trace(
        go.Scatter(
            x=total_position_value.index,
            y=total_position_value.values,
            name="总持仓市值",
            line=dict(color="red", width=2, dash="dash"),
            showlegend=True,
            legendgroup="holdings",
        ),
        row=1, col=1
    )

    # Rows 2-(n+1): 各股票详细图
    for idx, stock in enumerate(topk_stocks, 2):
        stock_df = df_records[df_records['stock'] == stock].copy()
        if len(stock_df) == 0:
            continue

        buy_df = stock_df[stock_df['signal'] == 'BUY']
        sell_df = stock_df[stock_df['signal'] == 'SELL']

        row_legend_group = f"stock_{stock}"
        stock_name = INSTRUMENT_INFO.get(stock, stock)

        # 价格（主Y轴）
        fig.add_trace(
            go.Scatter(
                x=stock_df.index, y=stock_df['price'],
                name=f"{stock_name}股价",
                line=dict(color="blue", width=1.5),
                showlegend=True if idx == 2 else False,
                legendgroup=row_legend_group,
            ),
            row=idx, col=1, secondary_y=False
        )

        # 买入点（只在第一个子图显示legend）
        if len(buy_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_df.index, y=buy_df['price'],
                    mode="markers",
                    name="金叉买入",
                    marker=dict(color="red", size=10, symbol="triangle-up", line=dict(color='darkred', width=1)),
                    showlegend=True if idx == 2 else False,
                    legendgroup="signals",
                ),
                row=idx, col=1, secondary_y=False
            )

        # 卖出点
        if len(sell_df) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_df.index, y=sell_df['price'],
                    mode="markers",
                    name="死叉卖出",
                    marker=dict(color="green", size=10, symbol="triangle-down", line=dict(color='darkgreen', width=1)),
                    showlegend=True if idx == 2 else False,
                    legendgroup="signals",
                ),
                row=idx, col=1, secondary_y=False
            )

        # MACD - DIF（次Y轴）
        fig.add_trace(
            go.Scatter(
                x=stock_df.index, y=stock_df['dif'],
                name="DIF线",
                line=dict(color="purple", width=1.5),
                showlegend=True if idx == 2 else False,
                legendgroup="macd",
            ),
            row=idx, col=1, secondary_y=True
        )

        # MACD - DEA（次Y轴）
        fig.add_trace(
            go.Scatter(
                x=stock_df.index, y=stock_df['dea'],
                name="DEA线",
                line=dict(color="orange", width=1.5),
                showlegend=True if idx == 2 else False,
                legendgroup="macd",
            ),
            row=idx, col=1, secondary_y=True
        )

        # MACD柱（次Y轴）
        macd_colors = ["red" if v >= 0 else "green" for v in stock_df['histogram']]
        fig.add_trace(
            go.Bar(
                x=stock_df.index, y=stock_df['histogram'],
                name="MACD柱",
                marker_color=macd_colors,
                opacity=0.4,
                width=1000*3600*24,
                showlegend=True if idx == 2 else False,
                legendgroup="macd",
            ),
            row=idx, col=1, secondary_y=True
        )

        # 设置Y轴标题
        fig.update_yaxes(title_text="股价", row=idx, col=1, secondary_y=False)
        fig.update_yaxes(title_text="MACD", row=idx, col=1, secondary_y=True)

    # 最后一行: 累计收益对比
    row_return = n_rows
    if report_df is not None:
        fig.add_trace(
            go.Scatter(
                x=report_df.index, y=report_df["cum_return"] * 100,
                name="策略累计收益",
                line=dict(color="red", width=2),
                fill="tozeroy",
                fillcolor="rgba(255,0,0,0.1)",
                showlegend=True,
                legendgroup="returns",
            ),
            row=row_return, col=1
        )

        # 添加基准收益
        if "bench" in report_df.columns:
            report_df["bench_cum"] = (1 + report_df["bench"]).cumprod() - 1
            fig.add_trace(
                go.Scatter(
                    x=report_df.index, y=report_df["bench_cum"] * 100,
                    name="基准累计收益",
                    line=dict(color="gray", width=1.5, dash="dash"),
                    showlegend=True,
                    legendgroup="returns",
                ),
                row=row_return, col=1
            )

        fig.add_hline(y=0, line_dash="dash", line_color="black", row=row_return, col=1, line_width=0.5)

    fig.update_layout(
        height=300 + n_stocks * 280,
        title_text=f"MACD TopK={n_stocks} 策略分析<br><sub>{config['test_start']} ~ {config['test_end']}</sub>",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
        hovermode="x unified",
    )

    # 更新y轴标签
    fig.update_yaxes(title_text="持仓金额(元)", row=1, col=1)
    for i in range(2, n_rows):
        fig.update_yaxes(title_text="价格/MACD", row=i, col=1)
    fig.update_yaxes(title_text="累计收益(%)", row=row_return, col=1)

    fig.write_html(os.path.join(save_dir, "macd_analysis.html"))
    fig.write_image(os.path.join(save_dir, "macd_analysis.png"), width=1400, height=fig.layout.height, scale=2)
    print(f"  可视化已保存: {save_dir}/macd_analysis.html")


def analyze_results(portfolio_metric_dict, topk_stocks, best_k, config, save_dir, strategy=None):
    """分析并保存回测结果"""
    os.makedirs(save_dir, exist_ok=True)
    report_df = portfolio_metric_dict.get("1day")[0]

    total_return = report_df["return"].sum()
    annualized_return = report_df["return"].mean() * 252
    volatility = report_df["return"].std() * np.sqrt(252)
    sharpe = annualized_return / volatility if volatility > 0 else 0
    cumulative = report_df["return"].cumsum()
    max_drawdown = (cumulative.cummax() - cumulative).max()
    bench_annual = report_df["bench"].mean() * 252
    excess_return = annualized_return - bench_annual

    print(f"\n{'='*60}")
    print(f"EVAL 回测结果: TopK={best_k}")
    print(f"{'='*60}")
    print(f"  标的: {topk_stocks}")
    print(f"  总收益: {total_return*100:.2f}%")
    print(f"  年化收益: {annualized_return*100:.2f}%")
    print(f"  基准年化: {bench_annual*100:.2f}%")
    print(f"  超额收益: {excess_return*100:.2f}%")
    print(f"  夏普比率: {sharpe:.4f}")
    print(f"  最大回撤: {max_drawdown*100:.2f}%")

    report_df.to_csv(os.path.join(save_dir, "report_df.csv"))
    summary = {
        "top_k": best_k,
        "topk_stocks": ",".join(topk_stocks),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "bench_annual": bench_annual,
        "excess_return": excess_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }
    pd.Series(summary).to_csv(os.path.join(save_dir, "summary.csv"))

    try:
        fig_list = analysis_position.report_graph(report_df, show_notebook=False)
        for idx, fig in enumerate(fig_list):
            fig.write_image(os.path.join(save_dir, f"return_{idx}.png"), scale=2)
            fig.write_html(os.path.join(save_dir, f"return_{idx}.html"))
    except:
        pass

    # 生成MACD可视化
    if strategy is not None:
        plot_macd_analysis(topk_stocks, strategy, config, save_dir)

    return summary


def run(max_k=10):
    """运行完整流程: TRAIN选股 -> VAL选K -> EVAL回测"""
    config = load_config()
    qlib.init(provider_uri=config["provider_uri"], region=config["region"])

    # 1. TRAIN: 选股
    df_stocks = select_topk_stocks(config)

    # 2. VAL: 选K
    best_k, df_k_results = select_best_k(df_stocks, config, max_k=max_k)

    # 3. EVAL: 回测
    portfolio_metric_dict, indicator_dict, topk_stocks, strategy = run_backtest(df_stocks, best_k, config)

    # 4. 分析结果
    save_dir = os.path.join(os.path.dirname(__file__), "results", f"MACDTopK_K{best_k}")
    os.makedirs(save_dir, exist_ok=True)
    df_stocks.to_csv(os.path.join(save_dir, "stock_selection.csv"), index=False)
    df_k_results.to_csv(os.path.join(save_dir, "k_selection.csv"), index=False)

    summary = analyze_results(portfolio_metric_dict, topk_stocks, best_k, config, save_dir, strategy=strategy)

    print(f"\n{'='*60}")
    print(f"完成! K={best_k}, 年化={summary['annualized_return']*100:.2f}%")
    print(f"结果: {save_dir}")
    print(f"{'='*60}")

    return summary


if __name__ == "__main__":
    fire.Fire({"run": run})
