"""
ETF 基金走势可视化（对齐归一化 + 估值 K/E/B 线）

按成立日期依次画出每条曲线：第 1 条曲线初始价格归一化为 1，
第 i 条曲线初始价格对齐到第 1 条（最早成立基金）在该曲线首日的价格。

输出图片（保存至 .../data_distribution/AllETF/）：
1. all_etf_aligned.png            — 所有 ETF 归一化收盘价叠加在同一张图，
                                    带 train/valid/test 数据集划分背景与分割线。
2. all_etf_subplots.png           — 每只 ETF 一个子图的归一化收盘价走势，
                                    带数据集划分标注。
3. all_etf_subplots_with_pepb.png — 每只 ETF 三行子图：归一化收盘价 / PE_TTM / PB，
                                    带数据集划分标注。
4. all_etf_subplots_keb.png       — 每只 ETF 一个子图的 K/E/B 三线：
                                    K 线=归一化收盘价，E 线=K/PE_TTM，B 线=K/PB
                                    （E、B 线归一化到与 K 线相同均值以便对比）。
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import qlib
from qlib.data import D

# 纯绘图函数复用自 plot_lib（无副作用），供本 CLI 与 pipeline 的 EtfVisualizer 共用
from plot_lib import get_fund_display_name, plot_aligned_etf, plot_subplots_etf

# 路径配置
DATA_DIR = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415/qlib_etf_index_Extend_wBond")
OUTPUT_DIR = DATA_DIR / "data_distribution" / "AllETF"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 基金名称映射
FUNDS_LIST_CSV = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415/source/funds_list.csv")


def load_fund_names():
    """从CSV加载基金名称映射"""
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


def load_all_etf_data(include_pe_pb=False):
    """加载所有ETF数据

    Args:
        include_pe_pb: 是否加载PE/PB估值数据
    """
    print("加载ETF数据...")

    instruments_file = DATA_DIR / "instruments/all.txt"
    with open(instruments_file, 'r') as f:
        funds = [line.strip().split('\t')[0] for line in f if line.strip()]

    print(f"基金数量: {len(funds)}")

    if include_pe_pb:
        fields = ['$close', '$pe_ttm', '$pb', '$pe_ttm_median', '$pb_median']
    else:
        fields = ['$close']

    df = D.features(funds, fields, freq='day', start_time='2005-02-23', end_time='2026-04-13')

    if include_pe_pb:
        df.columns = ['close', 'pe_ttm', 'pb', 'pe_ttm_median', 'pb_median']
    else:
        df.columns = ['close']

    return df


def plot_subplots_with_pepb(df, output_path, fund_names=None):
    """
    按子图方式画所有ETF数据，每个ETF一个子图组(close/PE/PB)
    添加数据集划分标注
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    if fund_names is None:
        fund_names = load_fund_names()

    # 数据集划分信息
    split_info = {
        'train': {'start': '2005-02-23', 'end': '2019-11-01', 'color': '#2ecc71', 'label': 'Train(50%)'},
        'valid': {'start': '2019-11-04', 'end': '2021-03-26', 'color': '#3498db', 'label': 'Valid(10%)'},
        'test_a1': {'start': '2021-03-29', 'end': '2023-10-09', 'color': '#e74c3c', 'label': 'Test A1(20%)'},
        'test_a2': {'start': '2023-10-10', 'end': '2025-01-03', 'color': '#f39c12', 'label': 'Test A2(10%)'},
        'test_b': {'start': '2025-01-06', 'end': '2026-04-09', 'color': '#9b59b6', 'label': 'Test B(10%)'},
    }

    # unstack: columns = (field, instrument)
    df_wide = df.unstack(level='instrument')

    # 按首日日期排序基金
    first_valid_dates = {}
    for col in df_wide.columns:
        if isinstance(col, tuple) and col[0] == 'close':
            inst = col[1]
            series = df_wide[col].dropna()
            if len(series) > 0:
                first_valid_dates[inst] = series.index[0]

    sorted_instruments = sorted(first_valid_dates.keys(), key=lambda x: first_valid_dates[x])

    # 以第1只基金(最早成立的)为基准
    base_inst = sorted_instruments[0]
    base_col = ('close', base_inst)
    base_prices = df_wide[base_col].dropna()

    # 计算子图布局: 每个ETF 3行(close, PE, PB), 每行1列
    n_instruments = len(sorted_instruments)
    n_rows_per_etf = 3  # close, PE, PB
    n_rows = n_instruments * n_rows_per_etf
    n_cols = 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3 * n_rows))
    # 处理axes维度
    if n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = plt.cm.tab20(np.linspace(0, 1, min(len(sorted_instruments), 20)))

    # 绘制分割线
    split_dates = [
        pd.Timestamp('2019-11-01'),  # train end
        pd.Timestamp('2021-03-26'),  # valid end
        pd.Timestamp('2023-10-09'),  # test_a1 end
        pd.Timestamp('2025-01-03'),   # test_a2 end
    ]

    row_labels = ['收盘价(归一化)', 'PE_TTM', 'PB']

    for i, inst in enumerate(sorted_instruments):
        color = colors[i % len(colors)]

        # 获取各字段数据
        close_series = df_wide[('close', inst)].dropna() if ('close', inst) in df_wide.columns else pd.Series()
        pe_series = df_wide[('pe_ttm', inst)].dropna() if ('pe_ttm', inst) in df_wide.columns else pd.Series()
        pb_series = df_wide[('pb', inst)].dropna() if ('pb', inst) in df_wide.columns else pd.Series()

        inst_first_date = close_series.index[0] if len(close_series) > 0 else None

        # 计算归一化价格
        if len(close_series) > 0 and i == 0:
            normalized_close = close_series / close_series.iloc[0]
        elif len(close_series) > 0:
            base_price_at_inst_first = base_prices.loc[inst_first_date]
            normalized_close = close_series / close_series.iloc[0] * base_price_at_inst_first
        else:
            normalized_close = pd.Series()

        # 3行数据
        plot_data = [
            (normalized_close, '归一化价格'),
            (pe_series, 'PE_TTM'),
            (pb_series, 'PB')
        ]

        for row_idx, (data, ylabel) in enumerate(plot_data):
            ax = axes[i * n_rows_per_etf + row_idx]

            if len(data) == 0:
                ax.set_visible(False)
                continue

            # 绘制数据集背景区域
            for name, info in split_info.items():
                start = pd.Timestamp(info['start'])
                end = pd.Timestamp(info['end'])
                ax.axvspan(start, end, alpha=0.15, color=info['color'], label=info['label'])

            # 绘制分割线
            for date in split_dates:
                ax.axvline(x=date, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

            ax.plot(data.index, data.values, linewidth=1.2, alpha=0.85, color=color)

            # 第一行显示标题
            if row_idx == 0:
                ax.set_title(get_fund_display_name(inst, fund_names), fontsize=10, loc='left')
            ax.set_ylabel(ylabel, fontsize=7)
            ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for j in range(n_instruments * n_rows_per_etf, len(axes)):
        axes[j].set_visible(False)

    # 创建数据集图例
    split_patches = [mpatches.Patch(color=info['color'], alpha=0.3, label=info['label'])
                     for info in split_info.values()]

    fig.legend(handles=split_patches, labels=[info['label'] for info in split_info.values()],
               loc='upper center', bbox_to_anchor=(0.5, 0.01), ncol=5, fontsize=9)

    fig.suptitle('所有ETF基金走势与估值数据 (Close/PE/PB)', fontsize=14, y=1.01)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"PE/PB子图已保存: {output_path}")


def plot_keb_subplots(df, output_path, fund_names=None):
    """
    按子图方式画所有ETF的K/E/B线
    K线: 归一化收盘价
    E线: K线 / PE_TTM (每点盈利)
    B线: K线 / PB (每点净值)
    每只ETF一个子图，包含3条线
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    if fund_names is None:
        fund_names = load_fund_names()

    # 数据集划分信息
    split_info = {
        'train': {'start': '2005-02-23', 'end': '2017-01-03', 'color': '#2ecc71', 'label': 'Train(50%)'},
        'valid': {'start': '2017-01-04', 'end': '2019-10-08', 'color': '#3498db', 'label': 'Valid(10%)'},
        'test_a1': {'start': '2019-10-09', 'end': '2026-04-09', 'color': '#e74c3c', 'label': 'Test A1(20%)'},
    }

    # unstack: columns = (field, instrument)
    df_wide = df.unstack(level='instrument')

    # 按首日日期排序基金
    first_valid_dates = {}
    for col in df_wide.columns:
        if isinstance(col, tuple) and col[0] == 'close':
            inst = col[1]
            series = df_wide[col].dropna()
            if len(series) > 0:
                first_valid_dates[inst] = series.index[0]

    sorted_instruments = sorted(first_valid_dates.keys(), key=lambda x: first_valid_dates[x])

    # 以第1只基金(最早成立的)为基准
    base_inst = sorted_instruments[0]
    base_col = ('close', base_inst)
    base_prices = df_wide[base_col].dropna()

    # 计算子图布局
    n_instruments = len(sorted_instruments)
    n_cols = 3
    n_rows = (n_instruments + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes.flatten()

    colors = plt.cm.tab20(np.linspace(0, 1, min(len(sorted_instruments), 20)))
    line_styles = ['-', '--', ':']  # K线实线, E线虚线, B线点线

    # 绘制分割线
    split_dates = [
        pd.Timestamp('2017-01-03'),  # train end
        pd.Timestamp('2019-10-08'),  # valid end
    ]

    for i, inst in enumerate(sorted_instruments):
        ax = axes[i]
        color = colors[i % len(colors)]

        # 获取各字段数据
        close_series = df_wide[('close', inst)].dropna() if ('close', inst) in df_wide.columns else pd.Series()
        pe_series = df_wide[('pe_ttm', inst)].dropna() if ('pe_ttm', inst) in df_wide.columns else pd.Series()
        pb_series = df_wide[('pb', inst)].dropna() if ('pb', inst) in df_wide.columns else pd.Series()

        inst_first_date = close_series.index[0] if len(close_series) > 0 else None

        # 计算归一化K线
        if len(close_series) > 0 and i == 0:
            normalized_close = close_series / close_series.iloc[0]
        elif len(close_series) > 0:
            base_price_at_inst_first = base_prices.loc[inst_first_date]
            normalized_close = close_series / close_series.iloc[0] * base_price_at_inst_first
        else:
            normalized_close = pd.Series()

        # 计算E线: K / PE，并归一化使其均值与K线相同
        if len(normalized_close) > 0 and len(pe_series) > 0:
            # 对齐日期
            common_dates = normalized_close.index.intersection(pe_series.index)
            e_line = (normalized_close.loc[common_dates] / pe_series.loc[common_dates]).dropna()
            # 归一化使均值与K线相同
            if len(e_line) > 0 and e_line.mean() > 0:
                e_line = e_line * (normalized_close.loc[e_line.index].mean() / e_line.mean())
        else:
            e_line = pd.Series()

        # 计算B线: K / PB，并归一化使其均值与K线相同
        if len(normalized_close) > 0 and len(pb_series) > 0:
            common_dates = normalized_close.index.intersection(pb_series.index)
            b_line = (normalized_close.loc[common_dates] / pb_series.loc[common_dates]).dropna()
            # 归一化使均值与K线相同
            if len(b_line) > 0 and b_line.mean() > 0:
                b_line = b_line * (normalized_close.loc[b_line.index].mean() / b_line.mean())
        else:
            b_line = pd.Series()

        # 绘制数据集背景区域
        for name, info in split_info.items():
            start = pd.Timestamp(info['start'])
            end = pd.Timestamp(info['end'])
            ax.axvspan(start, end, alpha=0.15, color=info['color'], label=info['label'])

        # 绘制分割线
        for date in split_dates:
            ax.axvline(x=date, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

        # 绘制K/E/B线
        if len(normalized_close) > 0:
            ax.plot(normalized_close.index, normalized_close.values,
                    linewidth=1.5, alpha=0.9, color=color, linestyle='-', label='K线')
        if len(e_line) > 0:
            ax.plot(e_line.index, e_line.values,
                    linewidth=1.2, alpha=0.8, color='orange', linestyle='--', label='E线(K/PE)')
        if len(b_line) > 0:
            ax.plot(b_line.index, b_line.values,
                    linewidth=1.2, alpha=0.8, color='purple', linestyle=':', label='B线(K/PB)')

        ax.set_title(get_fund_display_name(inst, fund_names), fontsize=10, loc='left')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=7)

    # 隐藏多余的子图
    for j in range(n_instruments, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('所有ETF基金K/E/B线走势 (K=收盘价, E=K/PE, B=K/PB)', fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"K/E/B线子图已保存: {output_path}")


def main():
    print("=" * 60)
    print("ETF基金K线走势可视化 (对齐归一化)")
    print("=" * 60)

    # 初始化qlib
    qlib.init(provider_uri=str(DATA_DIR), region="cn")

    # 加载基金名称映射
    fund_names = load_fund_names()
    print(f"已加载 {len(fund_names)} 个基金名称")

    # 加载数据(不含PE/PB)
    df = load_all_etf_data(include_pe_pb=False)
    print(f"数据形状: {df.shape}")
    print(f"日期范围: {df.index.get_level_values('datetime').min()} ~ {df.index.get_level_values('datetime').max()}")

    # 画图
    print("\n生成图表...")
    plot_aligned_etf(df, OUTPUT_DIR / "all_etf_aligned.png")
    plot_subplots_etf(df, OUTPUT_DIR / "all_etf_subplots.png", fund_names)

    # 加载数据(含PE/PB)并绘制
    print("\n加载PE/PB数据...")
    df_pepb = load_all_etf_data(include_pe_pb=True)
    print(f"PE/PB数据形状: {df_pepb.shape}")
    plot_subplots_with_pepb(df_pepb, OUTPUT_DIR / "all_etf_subplots_with_pepb.png", fund_names)
    plot_keb_subplots(df_pepb, OUTPUT_DIR / "all_etf_subplots_keb.png", fund_names)


if __name__ == "__main__":
    main()
