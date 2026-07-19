"""
ETF 价格与成交量关系可视化
- volume: 成交量（股数/份数），来自 K 线数据
- amount: 成交额（总金额），来自估值数据

输出图片（保存至 .../data_distribution/AllETF/）：
1. all_etf_price_volume.png — 每只 ETF 一个子图，双 Y 轴：
     - 左轴：收盘价。
     - 右轴：成交量(股数，忽略 0 值)原始折线 + EMA20 平滑线。
     按基金首日日期排序排布子图。
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import qlib
from qlib.data import D

DATA_DIR = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415/qlib_etf_index_Extend_wBond")
OUTPUT_DIR = DATA_DIR / "data_distribution" / "AllETF"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FUNDS_LIST_CSV = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415/source/funds_list.csv")


def load_fund_names():
    fund_names = {}
    try:
        df = pd.read_csv(FUNDS_LIST_CSV)
        for _, row in df.iterrows():
            code = str(row['fund_code'])
            name = row['fund_name']
            if pd.notna(name):
                fund_names[code] = name
    except Exception as e:
        print(f"加载基金名称失败: {e}")
    return fund_names


def get_fund_display_name(inst, fund_names):
    code = inst.replace('_NORMED', '').replace('_clean', '')
    name = fund_names.get(code, code)
    return f"{code} {name}"


def plot_price_volume(output_path, fund_names=None):
    """
    双Y轴子图：左轴收盘价，右轴成交量(volume，股数)和成交额(amount，总金额)
    """
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    if fund_names is None:
        fund_names = load_fund_names()

    instruments_file = DATA_DIR / "instruments/all.txt"
    with open(instruments_file, 'r') as f:
        funds = [line.strip().split('\t')[0] for line in f if line.strip()]
    print(f"基金数量: {len(funds)}")

    df = D.features(funds, ['$close', '$volume'],
                    freq='day', start_time='2005-02-23', end_time='2026-04-13')
    df.columns = ['close', 'volume']

    df_wide = df.unstack(level='instrument')

    # 按首日排序
    first_valid_dates = {}
    for inst in funds:
        col = ('close', inst)
        if col in df_wide.columns:
            series = df_wide[col].dropna()
            if len(series) > 0:
                first_valid_dates[inst] = series.index[0]

    sorted_instruments = sorted(first_valid_dates.keys(), key=lambda x: first_valid_dates[x])

    n_instruments = len(sorted_instruments)
    n_cols = 3
    n_rows = (n_instruments + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else axes.flatten()

    colors = plt.cm.tab20(np.linspace(0, 1, min(len(sorted_instruments), 20)))

    for i, inst in enumerate(sorted_instruments):
        ax = axes[i]
        color = colors[i % len(colors)]

        close = df_wide[('close', inst)].dropna() if ('close', inst) in df_wide.columns else pd.Series()
        volume = df_wide[('volume', inst)].dropna() if ('volume', inst) in df_wide.columns else pd.Series()

        ax2 = ax.twinx()  # 右轴：成交量

        # 右轴：成交量(折线，忽略0值) + EMA平滑
        if len(volume) > 0:
            vol_nonzero = volume[volume > 0]
            if len(vol_nonzero) > 0:
                ax2.plot(vol_nonzero.index, vol_nonzero.values,
                         linewidth=0.6, alpha=0.35, color='steelblue', label='成交量(股数)')
                vol_ema = vol_nonzero.ewm(span=20, adjust=False).mean()
                ax2.plot(vol_ema.index, vol_ema.values,
                         linewidth=1.2, alpha=0.85, color='royalblue', label='成交量EMA20')
        ax2.set_ylabel('成交量(股数)', fontsize=6, color='steelblue')
        ax2.tick_params(axis='y', labelsize=5, colors='steelblue')
        ax2.yaxis.label.set_color('steelblue')

        # 左轴：收盘价
        if len(close) > 0:
            ax.plot(close.index, close.values, linewidth=1.2, alpha=0.9, color=color, label='收盘价', zorder=3)
        ax.set_ylabel('收盘价', fontsize=7)
        ax.tick_params(axis='y', labelsize=6)
        ax.tick_params(axis='x', labelsize=6)

        ax.set_title(get_fund_display_name(inst, fund_names), fontsize=9, loc='left')
        ax.grid(True, alpha=0.2)

        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=6)

    for j in range(n_instruments, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('ETF价格与成交量关系 (左轴:收盘价, 右轴:成交量[股数])', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {output_path}")


def main():
    print("=" * 60)
    print("ETF价格与成交量关系可视化")
    print("注: volume=成交量(股数/份数), amount=成交额(总金额)")
    print("=" * 60)

    qlib.init(provider_uri=str(DATA_DIR), region="cn")
    fund_names = load_fund_names()
    print(f"已加载 {len(fund_names)} 个基金名称")

    plot_price_volume(OUTPUT_DIR / "all_etf_price_volume.png", fund_names)


if __name__ == "__main__":
    main()
