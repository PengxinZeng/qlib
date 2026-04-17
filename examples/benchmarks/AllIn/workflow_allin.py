#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TopK ALL-IN Strategy Workflow
TopK均匀持仓ALL-IN策略工作流

策略逻辑:
1. 在TRAIN Set上，根据收益率选出TopK股票
2. 在VAL Set上评测，均匀持有TopK股票，选择使综合收益最大化的K值
3. 在TEST Set上使用最优K进行评测

Usage:
    # 自动选择最优K
    python workflow_allin.py run

    # 指定最大K值
    python workflow_allin.py run --max_k=10

    # 固定K值
    python workflow_allin.py run --top_k=5
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


# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_config():
    """从config.yaml加载配置"""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        return {
            "provider_uri": yaml_config["qlib_init"]["provider_uri"],
            "region": yaml_config["qlib_init"]["region"],
            "train_start": yaml_config["train_period"]["start_time"],
            "train_end": yaml_config["train_period"]["end_time"],
            "valid_start": yaml_config.get("valid_period", {}).get("start_time", "2020-10-26"),
            "valid_end": yaml_config.get("valid_period", {}).get("end_time", "2023-08-07"),
            "test_start": yaml_config["test_period"]["start_time"],
            "test_end": yaml_config["test_period"]["end_time"],
            "benchmark": yaml_config["backtest"]["benchmark"],
            "account": yaml_config["backtest"]["account"],
        }
    return DEFAULT_CONFIG


DEFAULT_CONFIG = {
    "provider_uri": "/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415/qlib_data_fix_hfq",
    "region": "cn",
    "train_start": "1991-01-29",
    "train_end": "2020-10-23",
    "valid_start": "2020-10-26",
    "valid_end": "2023-08-07",
    "test_start": "2023-08-08",
    "test_end": "2026-04-15",
    "benchmark": "510300",
    "account": 1000000,
}

# 标的信息
INSTRUMENT_INFO = {
    "510050": "上证50ETF", "510300": "沪深300ETF", "510500": "中证500ETF",
    "518880": "黄金ETF", "512690": "酒ETF", "513100": "纳斯达克ETF",
    "SH600519": "贵州茅台", "SH600036": "招商银行", "SZ000333": "美的集团",
    "SZ300750": "宁德时代", "SZ002594": "比亚迪", "SH600900": "长江电力",
}


def select_topk_stocks(train_start, train_end, instruments="all"):
    """
    在训练集中选择收益最大的TopK标的

    Args:
        train_start: 训练开始时间
        train_end: 训练结束时间
        instruments: 股票池，支持:
            - "all": 所有标的
            - JSON字符串列表: 如 '["510050","510300"]'
            - Python列表: 如 ["510050","510300"]

    Returns:
        DataFrame: 包含所有股票收益信息的DataFrame
    """
    print(f"\n{'='*60}")
    print(f"选股阶段：在训练集 [{train_start} ~ {train_end}] 中按收益率排序")
    print(f"{'='*60}")

    # 解析 instruments 参数
    import json
    if isinstance(instruments, str) and instruments != "all":
        try:
            instruments = json.loads(instruments)
        except json.JSONDecodeError:
            pass
    if isinstance(instruments, list):
        inst_list = instruments
        print(f"标的池: {instruments}")
    else:
        # 获取所有标的
        inst_dict = D.list_instruments(D.instruments(instruments), start_time=train_start, end_time=train_end)
        inst_list = list(inst_dict.keys())

    results = []
    for inst in inst_list:
        try:
            df = D.features([inst], ["$close"], start_time=train_start, end_time=train_end)
            if df.empty or len(df) < 10:
                continue

            close_prices = df["$close"].droplevel(0)
            first_price = close_prices.iloc[0]
            last_price = close_prices.iloc[-1]

            if first_price > 0:
                total_return = (last_price - first_price) / first_price
                results.append({
                    "symbol": inst,
                    "first_price": first_price,
                    "last_price": last_price,
                    "total_return": total_return,
                    "days": len(close_prices),
                })
        except Exception:
            continue

    if not results:
        raise ValueError("No valid instruments found in training period")

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("total_return", ascending=False)
    df_results["name"] = df_results["symbol"].map(lambda x: INSTRUMENT_INFO.get(x, ""))

    print(f"\n训练期间收益 TOP 20:")
    print("-" * 80)
    for i, row in df_results.head(20).iterrows():
        name = row["name"] if row["name"] else row["symbol"]
        print(f"  {row['symbol']:12s} {name:12s} 收益率: {row['total_return']*100:8.2f}%  "
              f"({row['first_price']:.2f} -> {row['last_price']:.2f})")

    return df_results


def calculate_portfolio_return(stocks, valid_start, valid_end):
    """
    计算持有TopK股票组合在验证期间的收益率

    Args:
        stocks: 股票代码列表
        valid_start: 验证开始时间
        valid_end: 验证结束时间

    Returns:
        float: 组合收益率（等权）
    """
    returns = []
    for stock in stocks:
        try:
            df = D.features([stock], ["$close"], start_time=valid_start, end_time=valid_end)
            if df.empty or len(df) < 2:
                continue

            close_prices = df["$close"].droplevel(0)
            first_price = close_prices.iloc[0]
            last_price = close_prices.iloc[-1]

            if first_price > 0:
                ret = (last_price - first_price) / first_price
                returns.append(ret)
        except Exception:
            continue

    if not returns:
        return 0.0

    return np.mean(returns)


def select_best_k(df_stocks, valid_start, valid_end, max_k=10):
    """
    在验证集上选择最优的K值

    Args:
        df_stocks: 股票收益DataFrame（已按收益率降序排列）
        valid_start: 验证开始时间
        valid_end: 验证结束时间
        max_k: 最大K值

    Returns:
        tuple: (最优K值, 各K值对应的验证收益)
    """
    print(f"\n{'='*60}")
    print(f"选择最优K：在验证集 [{valid_start} ~ {valid_end}] 上评测")
    print(f"{'='*60}")

    k_results = []

    for k in range(1, min(max_k + 1, len(df_stocks) + 1)):
        topk_stocks = df_stocks.head(k)["symbol"].tolist()
        portfolio_return = calculate_portfolio_return(topk_stocks, valid_start, valid_end)
        k_results.append({
            "k": k,
            "stocks": ",".join(topk_stocks[:5]) + ("..." if len(topk_stocks) > 5 else ""),
            "portfolio_return": portfolio_return,
            "portfolio_return_pct": portfolio_return * 100,
        })
        print(f"  K={k:2d}: 组合收益={portfolio_return*100:7.2f}%  股票={topk_stocks[:3]}...")

    df_k_results = pd.DataFrame(k_results)
    best_idx = df_k_results["portfolio_return"].idxmax()
    best_k = df_k_results.loc[best_idx, "k"]
    best_return = df_k_results.loc[best_idx, "portfolio_return"]

    print(f"\n>>> 最优 K={best_k}, 验证集组合收益={best_return*100:.2f}%")

    return best_k, df_k_results


def run_backtest(topk_stocks, test_start, test_end, benchmark, account=1000000):
    """
    使用TopK ALL-IN策略进行回测

    Args:
        topk_stocks: TopK股票列表
        test_start: 测试开始时间
        test_end: 测试结束时间
        benchmark: 基准
        account: 初始资金

    Returns:
        dict: 回测结果
    """
    from strategy import TopKAllInStrategy

    print(f"\n{'='*60}")
    print(f"回测阶段：TopK ALL-IN策略 (K={len(topk_stocks)}) [{test_start} ~ {test_end}]")
    print(f"标的: {topk_stocks}")
    print(f"{'='*60}")

    # 创建策略
    strategy_config = {
        "class": "TopKAllInStrategy",
        "module_path": "strategy",
        "kwargs": {
            "topk_stocks": topk_stocks,
        }
    }

    # 执行器配置
    executor_config = {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        }
    }

    # 执行回测
    portfolio_metric_dict, indicator_dict = qlib_backtest(
        start_time=test_start,
        end_time=test_end,
        strategy=strategy_config,
        executor=executor_config,
        benchmark=benchmark,
        account=account,
        exchange_kwargs={
            "limit_threshold": 0.10,
            "deal_price": "close",
            "open_cost": 0.0003,
            "close_cost": 0.0003,
            "min_cost": 5,
        },
    )

    return portfolio_metric_dict, indicator_dict


def analyze_results(portfolio_metric_dict, topk_stocks, k, test_start, test_end, save_dir):
    """
    分析回测结果并生成报告（使用复利计算）
    """
    print(f"\n{'='*60}")
    print("分析阶段：生成回测报告")
    print(f"{'='*60}")

    os.makedirs(save_dir, exist_ok=True)

    report_df = portfolio_metric_dict.get("1day")[0]
    days = len(report_df)

    # 复利累计收益：(期末净值/期初净值) - 1
    strategy_cumprod = (1 + report_df["return"]).prod()
    bench_cumprod = (1 + report_df["bench"]).prod()

    total_return = strategy_cumprod - 1
    bench_return = bench_cumprod - 1

    # 单利年化收益：日均收益 * 252（与其他策略一致）
    annualized_return = report_df["return"].mean() * 252
    bench_annual = report_df["bench"].mean() * 252

    # 超额收益（单利年化）
    excess_return = annualized_return - bench_annual

    # 风险指标
    volatility = report_df["return"].std() * np.sqrt(252)
    sharpe = annualized_return / volatility if volatility > 0 else 0
    cumulative = (1 + report_df["return"]).cumprod() - 1
    max_drawdown = (cumulative.cummax() - cumulative).max()

    print(f"\n[回测结果摘要]")
    print(f"  TopK: {k}, 标的: {topk_stocks}")
    print(f"  回测期间: {test_start} ~ {test_end}")
    print(f"  交易天数: {days}")
    print(f"\n[收益指标]")
    print(f"  总收益率: {total_return*100:.2f}%")
    print(f"  年化收益: {annualized_return*100:.2f}%")
    print(f"  基准收益: {bench_return*100:.2f}%")
    print(f"  基准年化: {bench_annual*100:.2f}%")
    print(f"  超额年化: {excess_return*100:.2f}%")
    print(f"\n[风险指标]")
    print(f"  波动率:   {volatility*100:.2f}%")
    print(f"  夏普比率: {sharpe:.4f}")
    print(f"  最大回撤: {max_drawdown*100:.2f}%")

    report_df.to_csv(os.path.join(save_dir, "report_df.csv"))

    summary = {
        "top_k": k,
        "topk_stocks": ",".join(topk_stocks),
        "test_start": test_start,
        "test_end": test_end,
        "days": days,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "bench_return": bench_return,
        "bench_annual": bench_annual,
        "excess_return": excess_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }
    pd.Series(summary).to_csv(os.path.join(save_dir, "summary.csv"))

    try:
        fig_list = analysis_position.report_graph(report_df, show_notebook=False)
        for idx, fig in enumerate(fig_list):
            fig.write_image(os.path.join(save_dir, f"return_{idx}.png"), scale=2)
            fig.write_html(os.path.join(save_dir, f"return_{idx}.html"))
        print(f"\n[图表已保存到] {save_dir}")
    except Exception as e:
        print(f"\nWarning: Could not generate figures: {e}")

    return summary


def run(stocks=None, benchmark=None, max_k=10, top_k=None, instruments="all"):
    """
    运行ALL-IN策略

    Args:
        stocks: 直接指定股票列表，如 '["159919","510300"]'，不为空则直接回测
        benchmark: 基准代码，默认使用配置中的benchmark
        max_k: 最大K值（用于K值选择）
        top_k: 固定K值（跳过K值选择）
        instruments: 股票池，支持:
            - "all": 所有标的（默认）
            - JSON字符串列表: 如 '["510050","510300"]'
            - Python列表: 如 ["510050","510300"]
    """
    config = load_config()

    # 初始化Qlib
    qlib.init(provider_uri=config["provider_uri"], region=config["region"])

    # 处理stocks参数
    if stocks is not None:
        import json
        if isinstance(stocks, str):
            topk_stocks = json.loads(stocks)
        else:
            topk_stocks = stocks
        print(f"\n直接回测指定股票: {topk_stocks}")
    else:
        # 1. 选股阶段：在训练集上按收益率排序
        df_stocks = select_topk_stocks(
            config["train_start"],
            config["train_end"],
            instruments=instruments,
        )

        # 2. 选择最优K值
        if top_k is None:
            best_k, df_k_results = select_best_k(
                df_stocks,
                config["valid_start"],
                config["valid_end"],
                max_k=max_k,
            )
        else:
            best_k = top_k
            df_k_results = None
            print(f"\n使用固定 K={best_k}")

        # 获取TopK股票
        topk_stocks = df_stocks.head(best_k)["symbol"].tolist()

        # 保存选股和K值选择结果
        save_dir = os.path.join(os.path.dirname(__file__), "results", f"TopKAllIn_K{best_k}")
        os.makedirs(save_dir, exist_ok=True)
        df_stocks.to_csv(os.path.join(save_dir, "stock_selection.csv"), index=False)
        if df_k_results is not None:
            df_k_results.to_csv(os.path.join(save_dir, "k_selection.csv"), index=False)

    # 确定benchmark
    if benchmark is None:
        benchmark = config["benchmark"]
    benchmark = str(benchmark)

    # 3. 回测阶段
    portfolio_metric_dict, indicator_dict = run_backtest(
        topk_stocks=topk_stocks,
        test_start=config["test_start"],
        test_end=config["test_end"],
        benchmark=benchmark,
        account=config["account"],
    )

    # 4. 分析阶段
    save_dir = os.path.join(os.path.dirname(__file__), "results", f"{'_'.join(topk_stocks)}")
    summary = analyze_results(
        portfolio_metric_dict=portfolio_metric_dict,
        topk_stocks=topk_stocks,
        k=len(topk_stocks),
        test_start=config["test_start"],
        test_end=config["test_end"],
        save_dir=save_dir,
    )

    print(f"\n{'='*60}")
    print("TopK ALL-IN策略执行完成!")
    print(f"标的: {topk_stocks}")
    print(f"测试集年化收益={summary['annualized_return']*100:.2f}%")
    print(f"结果保存在: {save_dir}")
    print(f"{'='*60}")

    return summary


if __name__ == "__main__":
    fire.Fire({"run": run})
