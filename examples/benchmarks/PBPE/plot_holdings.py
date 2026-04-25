"""
PB/PE策略持仓可视化 - 使用最优参数在全部数据上运行
显示每只股票的K/E/B线以及持仓比例
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import qlib
from qlib.data import D

DATA_DIR = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415/qlib_etf_index")
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# 最优参数 (从param_sweep结果)
BEST_PARAMS = {
    'pe_min': 5, 'pe_max': 30, 'pb_min': 1.0, 'pb_max': 2.0,
    'pe_ratio': 1.5, 'pb_ratio': 1.2, 'topk': 5
}


def init_qlib():
    qlib.init(provider_uri=str(DATA_DIR), region="cn")


def get_all_instruments():
    with open(DATA_DIR / "instruments/all.txt", 'r') as f:
        return [line.strip().split('\t')[0] for line in f if line.strip()]


def load_fund_names():
    """从CSV加载基金名称映射"""
    fund_names = {}
    try:
        df = pd.read_csv(Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415/source/funds_list.csv"))
        for _, row in df.iterrows():
            code = str(row['fund_code'])
            name = row['fund_name']
            if pd.notna(name):
                fund_names[code] = name
    except Exception as e:
        print(f"加载基金名称失败: {e}")
    return fund_names


def get_fund_display_name(inst, fund_names):
    """获取基金的显示名称 (代码 + 名称)"""
    code = inst.replace('_NORMED', '')
    name = fund_names.get(code, code)
    return f"{code} {name}"


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

        pe_valid = (pe_ttm >= self.pe_min) & (pe_ttm <= self.pe_max)
        pb_valid = (pb >= self.pb_min) & (pb <= self.pb_max)
        pe_ratio_valid = pe_ttm <= pe_median * self.pe_ratio
        pb_ratio_valid = pb <= pb_median * self.pb_ratio
        valid = pe_valid & pb_valid & pe_ratio_valid & pb_ratio_valid

        score = np.zeros_like(pe_ttm)
        score[valid] = -pe_ttm[valid] / pe_median[valid] - pb[valid] / pb_median[valid]
        score[~valid] = np.nan

        return score


def preload_data(start_time, end_time):
    """预加载数据"""
    print("加载数据...")
    instruments = get_all_instruments()
    print(f"基金数量: {len(instruments)}")

    fields = ['$close', '$pe_ttm', '$pb', '$pe_ttm_median', '$pb_median']

    df = D.features(instruments, fields, freq='day', start_time=start_time, end_time=end_time)
    df.columns = [c.lstrip('$') for c in df.columns]

    all_dates = sorted(df.index.get_level_values('datetime').unique())
    all_instruments = df.index.get_level_values('instrument').unique()

    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    inst_to_idx = {inst: i for i, inst in enumerate(all_instruments)}

    daily_data = {}
    for (inst, date), row in df.iterrows():
        date_idx = date_to_idx[date]
        inst_idx = inst_to_idx[inst]
        if date_idx not in daily_data:
            daily_data[date_idx] = {}
        daily_data[date_idx][inst_idx] = row.to_dict()

    idx_to_date = {i: d for d, i in date_to_idx.items()}
    idx_to_inst = {i: inst for inst, i in inst_to_idx.items()}

    # 计算每个ETF的首日日期
    inst_first_date = {}
    for inst in all_instruments:
        inst_data = df.xs(inst, level='instrument')['close'].dropna()
        if len(inst_data) > 0:
            inst_first_date[inst_to_idx[inst]] = inst_data.index[0]
        else:
            inst_first_date[inst_to_idx[inst]] = None

    print(f"总交易日: {len(all_dates)}")
    return daily_data, idx_to_date, idx_to_inst, all_dates, inst_first_date


def run_backtest_with_positions(daily_data, idx_to_date, idx_to_inst, dates, inst_first_date, params, start_date, end_date):
    """运行回测并记录持仓信息"""
    strategy = SimplePBEStrategy(**params)

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    backtest_date_idx = [i for i, d in enumerate(dates) if start_dt <= d <= end_dt]
    print(f"回测期间: {dates[backtest_date_idx[0]]} ~ {dates[backtest_date_idx[-1]]}, 共{len(backtest_date_idx)}天")

    inst_idx_list = list(idx_to_inst.keys())
    n_inst = len(inst_idx_list)

    account = 1000000.0
    cash = account
    positions = {}  # {inst_idx: shares}

    # 记录每日持仓比例
    # position_ratio[date_idx] = {inst_idx: ratio}
    position_ratios = []

    for k, date_idx in enumerate(backtest_date_idx):
        day_data = daily_data.get(date_idx, {})
        current_date = dates[date_idx]

        eligible_inst_idx = [i for i in inst_idx_list
                           if inst_first_date.get(i) is not None
                           and inst_first_date.get(i) <= current_date]

        close = np.array([day_data.get(i, {}).get('close', np.nan) for i in inst_idx_list])
        close_clean = np.nan_to_num(close, nan=0.0)

        has_data = np.nansum(close) > 0

        if not has_data:
            # 没有数据日，计算持仓比例（全为0）
            total_value = cash + sum(close_clean[i] * positions.get(i, 0) for i in range(n_inst))
            ratio = {i: 0.0 for i in inst_idx_list}
            position_ratios.append(ratio)
            continue

        signal = strategy.get_signal(day_data, eligible_inst_idx)
        valid_mask = ~np.isnan(signal)
        n_valid = valid_mask.sum()

        if n_valid == 0 or n_valid < strategy.topk:
            # 无有效信号，保持持仓
            total_value = cash + sum(close_clean[i] * positions.get(i, 0) for i in range(n_inst))
            ratio = {}
            for i in inst_idx_list:
                if i in positions and total_value > 0:
                    ratio[i] = (positions[i] * close_clean[i]) / total_value
                else:
                    ratio[i] = 0.0
            position_ratios.append(ratio)
            continue

        sorted_idx = np.argsort(-signal)
        target_pos = sorted_idx[:strategy.topk]
        target_inst = [eligible_inst_idx[pos] for pos in target_pos]

        # 卖出
        for inst_idx in list(positions.keys()):
            if inst_idx not in target_inst:
                price = close[inst_idx]
                if price > 0:
                    shares = positions[inst_idx]
                    cash += shares * price * (1 - 0.0003)
                    del positions[inst_idx]

        # 买入
        buy_idx = [i for i in target_inst if i not in positions]
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

        # 计算持仓比例
        total_value = cash + sum(close_clean[i] * positions.get(i, 0) for i in range(n_inst))
        ratio = {}
        for i in inst_idx_list:
            if i in positions and total_value > 0:
                ratio[i] = (positions[i] * close_clean[i]) / total_value
            else:
                ratio[i] = 0.0
        position_ratios.append(ratio)

    return position_ratios, backtest_date_idx


def plot_holdings_with_keb(position_ratios, daily_data, idx_to_date, idx_to_inst, dates, inst_first_date, params, output_path):
    """绘制持仓比例与K/E/B线"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fund_names = load_fund_names()

    # 数据集划分信息
    split_info = {
        'train': {'start': '2005-02-23', 'end': '2017-01-03', 'color': '#2ecc71', 'label': 'Train(50%)'},
        'valid': {'start': '2017-01-04', 'end': '2019-10-08', 'color': '#3498db', 'label': 'Valid(10%)'},
        'test': {'start': '2019-10-09', 'end': '2026-04-09', 'color': '#e74c3c', 'label': 'Test(40%)'},
    }

    inst_idx_list = list(idx_to_inst.keys())
    n_inst = len(inst_idx_list)
    backtest_date_idx = list(range(len(position_ratios)))

    # 获取日期序列
    dates_for_ratios = [dates[i] for i in backtest_date_idx]

    # 计算分割线位置
    split_dates = [
        pd.Timestamp('2017-01-03'),
        pd.Timestamp('2019-10-08'),
    ]

    # 准备每只ETF的数据
    # 收集每只ETF的收盘价、PE、PB数据
    all_close = {}
    all_pe = {}
    all_pb = {}

    for inst_idx in inst_idx_list:
        inst_code = idx_to_inst.get(inst_idx)
        close_series = []
        pe_series = []
        pb_series = []

        for date_idx in backtest_date_idx:
            day_data = daily_data.get(date_idx, {})
            data = day_data.get(inst_idx, {})
            close_series.append(data.get('close', np.nan))
            pe_series.append(data.get('pe_ttm', np.nan))
            pb_series.append(data.get('pb', np.nan))

        all_close[inst_idx] = pd.Series(close_series, index=dates_for_ratios)
        all_pe[inst_idx] = pd.Series(pe_series, index=dates_for_ratios)
        all_pb[inst_idx] = pd.Series(pb_series, index=dates_for_ratios)

    # 按首日日期排序
    first_valid_dates = {}
    for inst_idx in inst_idx_list:
        series = all_close[inst_idx].dropna()
        if len(series) > 0:
            first_valid_dates[inst_idx] = series.index[0]
        else:
            first_valid_dates[inst_idx] = dates_for_ratios[-1]

    sorted_inst = sorted(inst_idx_list, key=lambda x: first_valid_dates[x])

    # 子图布局
    n_instruments = len(sorted_inst)
    n_cols = 3
    n_rows = (n_instruments + n_cols - 1) // n_cols + 1  # +1 for summary chart

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4.5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    colors = plt.cm.tab20(np.linspace(0, 1, min(n_instruments, 20)))

    # 绘制每只ETF
    for i, inst_idx in enumerate(sorted_inst):
        ax = axes[i]
        color = colors[i % len(colors)]
        inst_code = idx_to_inst.get(inst_idx)

        # 获取K线数据
        close_series = all_close[inst_idx].dropna()
        pe_series = all_pe[inst_idx].dropna()
        pb_series = all_pb[inst_idx].dropna()

        if len(close_series) == 0:
            ax.set_visible(False)
            continue

        # 归一化K线 (相对于首日收盘价)
        keb = close_series / close_series.iloc[0]

        # 计算E线: K / PE，归一化使均值与K相同
        if len(pe_series) > 0:
            common_dates = keb.index.intersection(pe_series.index)
            e_line = (keb.loc[common_dates] / pe_series.loc[common_dates]).dropna()
            if len(e_line) > 0 and e_line.mean() > 0:
                e_line = e_line * (keb.loc[e_line.index].mean() / e_line.mean())
        else:
            e_line = pd.Series()

        # 计算B线: K / PB，归一化使均值与K相同
        if len(pb_series) > 0:
            common_dates = keb.index.intersection(pb_series.index)
            b_line = (keb.loc[common_dates] / pb_series.loc[common_dates]).dropna()
            if len(b_line) > 0 and b_line.mean() > 0:
                b_line = b_line * (keb.loc[b_line.index].mean() / b_line.mean())
        else:
            b_line = pd.Series()

        # 获取持仓比例
        ratio_series = pd.Series(
            [position_ratios[j].get(inst_idx, 0.0) for j in backtest_date_idx],
            index=dates_for_ratios
        )

        # 绘制数据集背景
        for name, info in split_info.items():
            start = pd.Timestamp(info['start'])
            end = pd.Timestamp(info['end'])
            ax.axvspan(start, end, alpha=0.15, color=info['color'])

        # 绘制分割线
        for date in split_dates:
            ax.axvline(x=date, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

        # 绘制K线
        ax.plot(keb.index, keb.values, linewidth=1.5, alpha=0.9, color=color, linestyle='-', label='K线')

        # 绘制E线
        if len(e_line) > 0:
            ax.plot(e_line.index, e_line.values, linewidth=1.2, alpha=0.8, color='orange', linestyle='--', label='E线(K/PE)')

        # 绘制B线
        if len(b_line) > 0:
            ax.plot(b_line.index, b_line.values, linewidth=1.2, alpha=0.8, color='purple', linestyle=':', label='B线(K/PB)')

        # 绘制持仓比例 (右轴)
        ax2 = ax.twinx()
        ax2.fill_between(ratio_series.index, 0, ratio_series.values, alpha=0.3, color='green', label='持仓比例')
        ax2.set_ylabel('持仓比例', fontsize=8)
        ax2.set_ylim(0, 1.0)

        ax.set_title(get_fund_display_name(inst_code, fund_names), fontsize=10, loc='left')
        ax.grid(True, alpha=0.3)

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=6)

    # 最后一幅图: 当前持仓比例
    last_ax = axes[n_instruments]
    last_ratio = pd.Series(
        [position_ratios[j].get(inst_idx, 0.0) for j in backtest_date_idx],
        index=dates_for_ratios
    )

    # 绘制数据集背景
    for name, info in split_info.items():
        start = pd.Timestamp(info['start'])
        end = pd.Timestamp(info['end'])
        last_ax.axvspan(start, end, alpha=0.15, color=info['color'])

    for date in split_dates:
        last_ax.axvline(x=date, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

    # 绘制持仓比例
    for i, inst_idx in enumerate(sorted_inst):
        inst_code = idx_to_inst.get(inst_idx)
        ratio_series = pd.Series(
            [position_ratios[j].get(inst_idx, 0.0) for j in backtest_date_idx],
            index=dates_for_ratios
        )
        last_ax.plot(ratio_series.index, ratio_series.values, linewidth=1.5, alpha=0.8,
                     color=colors[i % len(colors)], label=get_fund_display_name(inst_code, fund_names))

    last_ax.axhline(y=1.0/BEST_PARAMS["topk"], color='red', linestyle='--', alpha=0.7, linewidth=1, label=f'等权基准({1.0/BEST_PARAMS["topk"]:.1%})')
    last_ax.set_title('当前持仓比例', fontsize=10, loc='left')
    last_ax.set_ylabel('持仓比例', fontsize=8)
    last_ax.set_ylim(0, 1.0)
    last_ax.grid(True, alpha=0.3)
    last_ax.legend(loc='upper left', fontsize=6, ncol=2)

    # 隐藏多余的子图
    for j in range(n_instruments + 1, len(axes)):
        axes[j].set_visible(False)

    # 添加总标题
    fig.suptitle(f'PB/PE策略持仓可视化 (最优参数: PE={params["pe_min"]}-{params["pe_max"]}, PB={params["pb_min"]}-{params["pb_max"]}, TOPK={params["topk"]})',
                 fontsize=14, y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"可视化已保存: {output_path}")


def main():
    print("=" * 60)
    print("PB/PE策略持仓可视化")
    print("=" * 60)

    init_qlib()

    daily_data, idx_to_date, idx_to_inst, dates, inst_first_date = preload_data("2005-02-23", "2026-04-13")

    print(f"\n最优参数: {BEST_PARAMS}")

    # 运行回测并记录持仓
    position_ratios, backtest_date_idx = run_backtest_with_positions(
        daily_data, idx_to_date, idx_to_inst, dates, inst_first_date,
        BEST_PARAMS, "2005-02-23", "2026-04-09"
    )

    # 绘制可视化
    output_path = OUTPUT_DIR / "pbpe_holdings_visualization.png"
    plot_holdings_with_keb(
        position_ratios, daily_data, idx_to_date, idx_to_inst, dates, inst_first_date,
        BEST_PARAMS, output_path
    )


if __name__ == "__main__":
    main()