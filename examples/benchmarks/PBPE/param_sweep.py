"""
PB/PE 策略参数敏感性分析 - 优化版
"""

import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import qlib
from qlib.data import D

DATA_DIR = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415/qlib_etf_index")
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 缩减参数搜索空间加速测试
PE_MIN_LIST = [5, 10]
PE_MAX_LIST = [30, 50]
PB_MIN_LIST = [0.5, 1.0]
PB_MAX_LIST = [2.0, 3.0]
PE_RATIO_LIST = [1.2, 1.5]
PB_RATIO_LIST = [1.2, 1.5]
TOPK_LIST = [3, 5]


def init_qlib():
    qlib.init(provider_uri=str(DATA_DIR), region="cn")


def get_all_instruments():
    with open(DATA_DIR / "instruments/all.txt", 'r') as f:
        return [line.strip().split('\t')[0] for line in f if line.strip()]


def preload_data_fast(start_time, end_time):
    """快速预加载数据"""
    print("[" + time.strftime("%H:%M:%S") + "] 获取基金列表...")
    instruments = get_all_instruments()
    print(f"[" + time.strftime("%H:%M:%S") + "] 基金数量: {len(instruments)}")

    fields = ['$close', '$pe_ttm', '$pb', '$pe_ttm_median', '$pb_median']

    print(f"[" + time.strftime("%H:%M:%S") + "] 加载所有数据...")
    start = time.time()

    # 一次性加载所有数据
    df = D.features(instruments, fields, freq='day', start_time=start_time, end_time=end_time)
    df.columns = [c.lstrip('$') for c in df.columns]

    print(f"[" + time.strftime("%H:%M:%S") + "] 数据加载完成, 耗时 {time.time()-start:.1f}s")
    print(f"[" + time.strftime("%H:%M:%S") + "] 数据形状: {df.shape}")

    # 转换为嵌套字典格式 {date_str: {instrument: {field: value}}}
    print(f"[" + time.strftime("%H:%M:%S") + "] 整理数据格式...")
    start = time.time()

    # 按日期分组
    all_dates = sorted(df.index.get_level_values('datetime').unique())
    all_instruments = df.index.get_level_values('instrument').unique()

    # 构建日期->数据映射
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    inst_to_idx = {inst: i for i, inst in enumerate(all_instruments)}

    # 构建高效查询结构
    # daily_data[date_idx] = {inst_idx: {field: value}}
    daily_data = {}
    for (inst, date), row in df.iterrows():
        date_idx = date_to_idx[date]
        inst_idx = inst_to_idx[inst]
        if date_idx not in daily_data:
            daily_data[date_idx] = {}
        daily_data[date_idx][inst_idx] = row.to_dict()

    # 同时保存idx->date和idx->inst映射
    idx_to_date = {i: d for d, i in date_to_idx.items()}
    idx_to_inst = {i: inst for inst, i in inst_to_idx.items()}

    print(f"[" + time.strftime("%H:%M:%S") + "] 整理完成, 耗时 {time.time()-start:.1f}s")
    print(f"[" + time.strftime("%H:%M:%S") + "] 总交易日: {len(all_dates)}")

    return daily_data, idx_to_date, idx_to_inst, all_dates


class SimplePBEStrategy:
    def __init__(self, pe_min, pe_max, pb_min, pb_max, pe_ratio, pb_ratio, topk):
        self.pe_min = pe_min
        self.pe_max = pe_max
        self.pb_min = pb_min
        self.pb_max = pb_max
        self.pe_ratio = pe_ratio
        self.pb_ratio = pb_ratio
        self.topk = topk

    def get_signal(self, day_data, inst_idx_list):
        """计算信号"""
        pe_ttm = np.array([day_data.get(i, {}).get('pe_ttm', np.nan) for i in inst_idx_list])
        pb = np.array([day_data.get(i, {}).get('pb', np.nan) for i in inst_idx_list])
        pe_median = np.array([day_data.get(i, {}).get('pe_ttm_median', np.nan) for i in inst_idx_list])
        pb_median = np.array([day_data.get(i, {}).get('pb_median', np.nan) for i in inst_idx_list])

        # 估值过滤
        pe_valid = (pe_ttm >= self.pe_min) & (pe_ttm <= self.pe_max)
        pb_valid = (pb >= self.pb_min) & (pb <= self.pb_max)
        pe_ratio_valid = pe_ttm <= pe_median * self.pe_ratio
        pb_ratio_valid = pb <= pb_median * self.pb_ratio
        valid = pe_valid & pb_valid & pe_ratio_valid & pb_ratio_valid

        # 评分
        score = np.zeros_like(pe_ttm)
        score[valid] = -pe_ttm[valid] / pe_median[valid] - pb[valid] / pb_median[valid]
        score[~valid] = np.nan

        return score


def run_single_backtest(daily_data, idx_to_date, idx_to_inst, dates, params, start_date, end_date):
    """单次回测"""
    strategy = SimplePBEStrategy(**params)

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    backtest_date_idx = [i for i, d in enumerate(dates) if start_dt <= d <= end_dt]
    if len(backtest_date_idx) == 0:
        print("No backtest dates found!")
        return None

    print(f"  Backtest dates: {len(backtest_date_idx)}, first: {dates[backtest_date_idx[0]]}, last: {dates[backtest_date_idx[-1]]}")

    inst_idx_list = list(idx_to_inst.keys())
    n_inst = len(inst_idx_list)
    n_dates = len(backtest_date_idx)

    account = 1000000.0
    cash = account
    positions = {}  # {inst_idx: shares}
    portfolio_values = []

    for k, date_idx in enumerate(backtest_date_idx):
        day_data = daily_data.get(date_idx, {})

        # 获取收盘价
        close = np.array([day_data.get(i, {}).get('close', 0) for i in inst_idx_list])

        # 检查是否有数据
        has_data = close.sum() > 0
        if not has_data:
            # 没有数据日，保持现金值
            portfolio_values.append(cash)
            continue

        # 获取信号
        signal = strategy.get_signal(day_data, inst_idx_list)

        # 获取有效信号的基金
        valid_mask = ~np.isnan(signal)
        n_valid = valid_mask.sum()

        if n_valid == 0:
            # 无有效信号，保持持仓
            total_value = cash + sum(close[i] * positions.get(i, 0) for i in range(n_inst))
            portfolio_values.append(total_value)
            continue

        # 选择高分基金
        sorted_idx = np.argsort(-signal)
        target_idx = sorted_idx[:strategy.topk]

        # 卖出
        for inst_idx in list(positions.keys()):
            if inst_idx not in target_idx:
                price = close[inst_idx]
                if price > 0:
                    shares = positions[inst_idx]
                    cash += shares * price * (1 - 0.0003)
                    del positions[inst_idx]

        # 买入
        buy_idx = [i for i in target_idx if i not in positions]
        if len(buy_idx) > 0:
            per_stock_value = cash * 0.95 / len(buy_idx)
            for inst_idx in buy_idx:
                price = close[inst_idx]
                if price > 0:
                    shares = int(per_stock_value / price / 100) * 100
                    if shares > 0:
                        cost = shares * price * (1 + 0.0003)
                        if cost <= cash:
                            positions[inst_idx] = shares
                            cash -= cost

        # 计算总市值
        total_value = cash + sum(close[i] * positions.get(i, 0) for i in range(n_inst))
        portfolio_values.append(total_value)

    if len(portfolio_values) < 2:
        return None

    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    annualized_return = (portfolio_values[-1] / portfolio_values[0]) ** (252 / len(portfolio_values)) - 1
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    max_drawdown = np.max(np.maximum.accumulate(portfolio_values) - portfolio_values) / np.maximum.accumulate(portfolio_values)[0]

    print(f"  Final value: {portfolio_values[-1]:.2f}, Returns: {returns.sum():.2%}, Sharpe: {sharpe:.2f}")
    print(f"  First 5 portfolio values: {portfolio_values[:5]}")

    return {
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
    }


def run_parameter_sweep():
    print("=" * 60)
    print("PB/PE 策略参数敏感性分析")
    print("=" * 60)

    init_qlib()

    print(f"[" + time.strftime("%H:%M:%S") + "] 开始预加载数据...")
    daily_data, idx_to_date, idx_to_inst, dates = preload_data_fast("2005-02-23", "2026-04-13")

    param_combinations = list(product(
        PE_MIN_LIST, PE_MAX_LIST, PB_MIN_LIST, PB_MAX_LIST,
        PE_RATIO_LIST, PB_RATIO_LIST, TOPK_LIST
    ))
    print(f"\n共 {len(param_combinations)} 组参数组合")

    test_start = "2021-03-29"
    test_end = "2023-10-09"

    results = []
    total = len(param_combinations)
    start_time = time.time()

    print(f"\n[" + time.strftime("%H:%M:%S") + "] 开始参数扫描...")
    print("-" * 60)

    for i, (pe_min, pe_max, pb_min, pb_max, pe_ratio, pb_ratio, topk) in enumerate(param_combinations):
        params = {
            "pe_min": pe_min, "pe_max": pe_max,
            "pb_min": pb_min, "pb_max": pb_max,
            "pe_ratio": pe_ratio, "pb_ratio": pb_ratio,
            "topk": topk,
        }

        metrics = run_single_backtest(daily_data, idx_to_date, idx_to_inst, dates, params, test_start, test_end)

        if metrics:
            results.append({
                "pe_min": pe_min, "pe_max": pe_max,
                "pb_min": pb_min, "pb_max": pb_max,
                "pe_ratio": pe_ratio, "pb_ratio": pb_ratio,
                "topk": topk,
                **metrics
            })

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            eta = elapsed * (total - i - 1) / (i + 1)
            print(f"进度: {i + 1}/{total} ({100*(i+1)/total:.1f}%) | "
                  f"已耗时: {elapsed:.1f}s | 预计剩余: {eta:.1f}s | 成功: {len(results)}")

    print(f"\n[" + time.strftime("%H:%M:%S") + "] 完成! 成功 {len(results)} 组参数")

    if len(results) == 0:
        print("没有成功的回测结果!")
        return

    df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("Top 10 参数组合 (按年化收益率)")
    print("=" * 60)
    top10 = df.nlargest(10, "annualized_return")
    print(top10.to_string(index=False))

    output_file = OUTPUT_DIR / "param_sweep_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n结果已保存: {output_file}")

    # 参数影响分析
    print("\n" + "=" * 60)
    print("参数影响分析")
    print("=" * 60)

    for param_name in ["pe_max", "pb_max", "topk", "pe_ratio"]:
        print(f"\n[{param_name} 影响]")
        analysis = df.groupby(param_name).agg({
            "annualized_return": ["mean", "std", "max"],
            "sharpe": "mean",
        }).round(4)
        print(analysis)

    # 可视化
    print("\n[" + time.strftime("%H:%M:%S") + "] 生成可视化...")
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, param_name, color in [
        (axes[0, 0], "pe_max", 'steelblue'),
        (axes[0, 1], "pb_max", 'darkorange'),
        (axes[1, 0], "topk", 'forestgreen'),
        (axes[1, 1], "pe_ratio", 'crimson'),
    ]:
        group = df.groupby(param_name)["annualized_return"].agg(["mean", "std"])
        ax.bar(group.index.astype(str), group["mean"], yerr=group["std"], capsize=5, color=color)
        ax.set_xlabel(param_name)
        ax.set_ylabel("年化收益率")
        ax.set_title(f"{param_name} 对年化收益率的影响")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "param_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"可视化已保存: {OUTPUT_DIR / 'param_sensitivity.png'}")

    return df


if __name__ == "__main__":
    run_parameter_sweep()
