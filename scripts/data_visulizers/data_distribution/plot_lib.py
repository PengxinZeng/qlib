"""
plot_lib — 无副作用的纯绘图函数库

从 plot_etf_KEB_comparison.py 抽取，供 CLI 脚本与数据 pipeline(EtfVisualizer) 共同复用。
本模块不做任何 import 期副作用（不建目录、不依赖 qlib），仅接收组装好的 df 作图。

df 契约：MultiIndex 含 'instrument' 与 'datetime' 两级，含一列 'close'。
"""

import numpy as np
import pandas as pd


def get_fund_display_name(inst, fund_names):
    """获取基金的显示名称 (代码 + 名称)；无名称时仅返回代码。"""
    # 从 inst like "510050_NORMED" 提取代码
    code = inst.replace('_NORMED', '')
    name = fund_names.get(code)
    return f"{code} {name}" if name and name != code else code


def _make_color_fn(sorted_instruments, track_index):
    """返回 color_for(i, inst) 着色函数。

    - track_index 提供时：按跟踪指数分组着色（同指数同色）；无指数的标的以自身代码单独成组。
    - 否则：沿用按曲线序号的 tab20 着色。
    """
    import matplotlib.pyplot as plt
    if track_index:
        keys = list(dict.fromkeys(
            (track_index.get(inst) or inst) for inst in sorted_instruments
        ))
        palette = plt.cm.tab20(np.linspace(0, 1, min(max(len(keys), 1), 20)))
        idx_to_color = {k: palette[n % len(palette)] for n, k in enumerate(keys)}
        return lambda i, inst: idx_to_color[track_index.get(inst) or inst]

    palette = plt.cm.tab20(np.linspace(0, 1, min(max(len(sorted_instruments), 1), 20)))
    return lambda i, inst: palette[i % len(palette)]


# 默认数据集划分段（当未从外部传入 split_info 时的回退，保持原 CLI 行为）
_DEFAULT_SEGMENTS_ALIGNED = [
    {'name': 'train',   'start': '2005-02-23', 'end': '2019-11-01', 'color': '#2ecc71', 'label': 'Train(50%)'},
    {'name': 'valid',   'start': '2019-11-04', 'end': '2021-03-26', 'color': '#3498db', 'label': 'Valid(10%)'},
    {'name': 'test_a1', 'start': '2021-03-29', 'end': '2023-10-09', 'color': '#e74c3c', 'label': 'Test A1(20%)'},
    {'name': 'test_a2', 'start': '2023-10-10', 'end': '2025-01-03', 'color': '#f39c12', 'label': 'Test A2(10%)'},
    {'name': 'test_b',  'start': '2025-01-06', 'end': '2026-04-09', 'color': '#9b59b6', 'label': 'Test B(10%)'},
]
_DEFAULT_SEGMENTS_SUBPLOTS = [
    {'name': 'train',   'start': '2005-02-23', 'end': '2017-01-03', 'color': '#2ecc71', 'label': 'Train(50%)'},
    {'name': 'valid',   'start': '2017-01-04', 'end': '2019-11-03', 'color': '#3498db', 'label': 'Valid(10%)'},
    {'name': 'test_a1', 'start': '2019-11-04', 'end': '2026-04-09', 'color': '#e74c3c', 'label': 'Test A1(20%)'},
]


def _split_colors(names: list) -> dict:
    """为数据集段名分配颜色：已知名沿用固定色，其余按 tab10 顺序分配。"""
    import matplotlib.pyplot as plt
    preferred = {
        'train': '#2ecc71', 'valid': '#3498db',
        'test': '#e74c3c', 'testB': '#9b59b6',
        'test_a1': '#e74c3c', 'test_a2': '#f39c12', 'test_b': '#9b59b6',
    }
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    return {n: preferred.get(n, tuple(cmap[i % 10])) for i, n in enumerate(names)}


def _resolve_segments(split_info, default_segments) -> list:
    """将传入的 split_info（[{name,start,end,(color),(label)}]）规范化为绘图用段列表；
    未提供时回退到 default_segments。颜色缺省按段名自动分配，label 缺省用段名。"""
    segs = split_info if split_info else default_segments
    names = [s['name'] for s in segs]
    colors = _split_colors(names)
    out = []
    for s in segs:
        out.append({
            'name': s['name'],
            'start': pd.Timestamp(s['start']),
            'end': pd.Timestamp(s['end']),
            'color': s.get('color') or colors[s['name']],
            'label': s.get('label') or s['name'],
        })
    return out


def plot_aligned_etf(df, output_path, fund_names=None, track_index=None, split_info=None):
    """
    按对齐方式画所有ETF归一化走势
    第1条曲线初始价格为1
    第i条曲线初始价格为第1条曲线在第i条曲线首日的价格
    track_index 提供时按跟踪指数分组着色（同指数同色）。
    split_info 提供时按其数据集划分绘制背景与分割线（[{name,start,end}]），否则用内置默认。
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    if fund_names is None:
        fund_names = {}

    # 数据集划分段（外部传入优先，否则用默认）
    segments = _resolve_segments(split_info, _DEFAULT_SEGMENTS_ALIGNED)

    # unstack: columns = (field, instrument)
    df_wide = df.unstack(level='instrument')

    # 按首日日期排序基金
    first_valid_dates = {}
    for col in df_wide.columns:
        inst = col[1]
        series = df_wide[col].dropna()
        if len(series) > 0:
            first_valid_dates[inst] = series.index[0]

    sorted_instruments = sorted(first_valid_dates.keys(), key=lambda x: first_valid_dates[x])
    print(f"按成立日期排序的基金: {sorted_instruments}")

    # 以第1只基金(最早成立的)为基准
    base_inst = sorted_instruments[0]
    base_col = ('close', base_inst)
    base_prices = df_wide[base_col].dropna()

    fig, ax = plt.subplots(figsize=(20, 10))

    color_for = _make_color_fn(sorted_instruments, track_index)

    # 绘制数据集背景区域（不加 label，避免与下方 split_patches 图例重复）
    for seg in segments:
        ax.axvspan(seg['start'], seg['end'], alpha=0.15, color=seg['color'])

    # 绘制分割线（各段结束处，最后一段除外）
    for seg in segments[:-1]:
        ax.axvline(x=seg['end'], color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

    # 绘制基金曲线
    curve_entries = []  # (sort_key, label, handle) — 用于图例按颜色分组排序
    for i, inst in enumerate(sorted_instruments):
        color = color_for(i, inst)
        col = ('close', inst)
        series = df_wide[col].dropna()
        inst_first_date = series.index[0]

        if i == 0:
            normalized = series / series.iloc[0]
        else:
            base_price_at_inst_first = base_prices.loc[inst_first_date]
            normalized = series / series.iloc[0] * base_price_at_inst_first

        line, = ax.plot(normalized.index, normalized.values,
                        linewidth=1.2, alpha=0.85, color=color,
                        label=get_fund_display_name(inst, fund_names))
        # 按颜色（跟踪指数）分组排序图例；无 track_index 时保持原有序号顺序
        sort_key = (track_index.get(inst) or inst) if track_index else i
        curve_entries.append((sort_key, get_fund_display_name(inst, fund_names), line))

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('归一化价格', fontsize=12)
    ax.set_title('所有ETF基金K线走势 (按首日对齐归一化) - 数据集划分', fontsize=14)

    # 创建图例：基金按颜色（跟踪指数）分组，同组内保持成立日期顺序（稳定排序）
    curve_entries.sort(key=lambda e: str(e[0]))
    handles = [e[2] for e in curve_entries]
    labels = [e[1] for e in curve_entries]
    # 添加数据集图例（仅一次）
    split_patches = [mpatches.Patch(color=seg['color'], alpha=0.3, label=seg['label'])
                     for seg in segments]
    handles = handles + split_patches
    labels = labels + [seg['label'] for seg in segments]

    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图表已保存: {output_path}")


def plot_subplots_etf(df, output_path, fund_names=None, track_index=None, split_info=None):
    """
    按子图方式画所有ETF归一化走势，每个ETF一个子图
    添加数据集划分标注；track_index 提供时按跟踪指数分组着色（同指数同色）。
    split_info 提供时按其数据集划分绘制背景与分割线（[{name,start,end}]），否则用内置默认。
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    if fund_names is None:
        fund_names = {}

    # 数据集划分段（外部传入优先，否则用默认）
    segments = _resolve_segments(split_info, _DEFAULT_SEGMENTS_SUBPLOTS)

    # unstack: columns = (field, instrument)
    df_wide = df.unstack(level='instrument')

    # 按首日日期排序基金
    first_valid_dates = {}
    for col in df_wide.columns:
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

    color_for = _make_color_fn(sorted_instruments, track_index)

    for i, inst in enumerate(sorted_instruments):
        ax = axes[i]
        color = color_for(i, inst)
        col = ('close', inst)
        series = df_wide[col].dropna()
        inst_first_date = series.index[0]

        if i == 0:
            normalized = series / series.iloc[0]
        else:
            base_price_at_inst_first = base_prices.loc[inst_first_date]
            normalized = series / series.iloc[0] * base_price_at_inst_first

        # 绘制数据集背景区域
        for seg in segments:
            ax.axvspan(seg['start'], seg['end'], alpha=0.15, color=seg['color'])

        # 绘制分割线（各段结束处，最后一段除外）
        for seg in segments[:-1]:
            ax.axvline(x=seg['end'], color='gray', linestyle='--', alpha=0.7, linewidth=1.5)

        ax.plot(normalized.index, normalized.values, linewidth=1.2, alpha=0.85, color=color)
        ax.set_title(get_fund_display_name(inst, fund_names), fontsize=10)
        ax.grid(True, alpha=0.3)

        # 只在左侧显示y轴标签
        ax.set_ylabel('归一化价格', fontsize=8)

    # 隐藏多余的子图
    for j in range(n_instruments, len(axes)):
        axes[j].set_visible(False)

    # 创建数据集图例
    split_patches = [mpatches.Patch(color=seg['color'], alpha=0.3, label=seg['label'])
                     for seg in segments]

    fig.legend(handles=split_patches, labels=[seg['label'] for seg in segments],
               loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=len(segments), fontsize=9)

    fig.suptitle('所有ETF基金K线走势 (子图模式) - 数据集划分', fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"子图图表已保存: {output_path}")

