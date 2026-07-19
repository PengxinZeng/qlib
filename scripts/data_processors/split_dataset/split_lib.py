"""
split_lib — 无副作用的数据集切分与可视化函数库

从 split_dataset.py 抽取，供 CLI 脚本与数据 pipeline(DatasetSplitter) 共同复用。
本模块不做任何 import 期副作用（不建目录、不读固定路径）。

df 契约：含 'date'(datetime) 与 'tradeable_count' 两列，按 date 升序。
splits 契约：有序列表 [(name, ratio), ...]，ratio 之和应约等于 1.0。
"""

import numpy as np
import pandas as pd


def split_by_cumsum(df: pd.DataFrame, splits: list) -> tuple:
    """
    按每日可交易数量的累计和划分数据集。
    splits: [(name, ratio), ...]，顺序即集合先后。
    返回: ([(name, start_idx, end_idx), ...], total_sum)

    方案 B：用绝对累计比例边界(一次性从 total_sum 算出)对每行做 digitize 标签，
    避免"以上一段实际 cumsum 为锚点"带来的误差累计，保证全覆盖、无漏行。
    """
    df = df.copy().reset_index(drop=True)
    df['cumsum'] = df['tradeable_count'].cumsum()
    total_sum = df['cumsum'].iloc[-1]

    # 绝对累计边界：[r0, r0+r1, ..., 1.0] * total_sum；末边界强制 == total_sum 消除浮点误差
    cum_ratios = np.cumsum([r for _, r in splits])
    boundaries = (total_sum * cum_ratios).astype(float)
    boundaries[-1] = float(total_sum)

    # 逐行打标签：label = 严格小于该行 cumsum 的边界数量 → 落在第 label 段
    cumsum_values = df['cumsum'].to_numpy()
    labels = np.searchsorted(boundaries, cumsum_values, side='left')
    labels = np.clip(labels, 0, len(splits) - 1)

    result = []
    for i, (name, _) in enumerate(splits):
        idxs = np.where(labels == i)[0]
        if len(idxs) == 0:
            raise ValueError(
                f"split_by_cumsum: 集合 '{name}' 无数据（比例过小或数据不足），"
                f"请调整 splits 比例"
            )
        result.append((name, int(idxs[0]), int(idxs[-1])))

    return result, total_sum


def build_results(df: pd.DataFrame, splits_result: list, total_sum: float) -> list:
    """将 split_by_cumsum 的索引结果转为带统计信息的字典列表。"""
    df = df.reset_index(drop=True)
    results = []
    for name, start_idx, end_idx in splits_result:
        subset = df.iloc[start_idx:end_idx + 1]
        results.append({
            'dataset': name,
            'start_date': subset['date'].iloc[0].strftime('%Y-%m-%d'),
            'end_date': subset['date'].iloc[-1].strftime('%Y-%m-%d'),
            'start_idx': int(start_idx),
            'end_idx': int(end_idx),
            'days': int(len(subset)),
            'total_count': int(subset['tradeable_count'].sum()),
            'pct_of_total_count': f"{subset['tradeable_count'].sum() / total_sum * 100:.1f}%",
            'pct_of_days': f"{len(subset) / len(df) * 100:.1f}%",
            'avg_tradeable': f"{subset['tradeable_count'].mean():.1f}",
            'min_tradeable': int(subset['tradeable_count'].min()),
            'max_tradeable': int(subset['tradeable_count'].max()),
        })
    return results


def make_colors(names: list) -> dict:
    """为集合名分配颜色：已知名沿用固定色，其余按 tab10 顺序分配。"""
    import matplotlib.pyplot as plt
    preferred = {
        'train': '#2ecc71', 'valid': '#3498db',
        'test_a1': '#e74c3c', 'test_a2': '#f39c12', 'test_b': '#9b59b6',
    }
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    colors = {}
    for i, n in enumerate(names):
        colors[n] = preferred.get(n, tuple(cmap[i % 10]))
    return colors


def plot_split(df: pd.DataFrame, results: list, total_sum: float, output_path) -> None:
    """双子图：上=每日可交易数量按集合着色填充；下=累计曲线+分割竖线+累计阈值横线。"""
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    df = df.copy()
    if 'cumsum' not in df.columns:
        df['cumsum'] = df['tradeable_count'].cumsum()

    names = [r['dataset'] for r in results]
    colors = make_colors(names)

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # 图1：每日可交易数量，各集合区间填充
    ax1 = axes[0]
    for r in results:
        start, end = pd.Timestamp(r['start_date']), pd.Timestamp(r['end_date'])
        subset = df[(df['date'] >= start) & (df['date'] <= end)]
        ax1.fill_between(subset['date'], 0, subset['tradeable_count'],
                         color=colors[r['dataset']], alpha=0.4,
                         label=f"{r['dataset']} ({r['days']}天, {r['pct_of_total_count']})")
    ax1.plot(df['date'], df['tradeable_count'], color='gray', linewidth=0.5)
    ax1.set_ylabel('可交易股票数', fontsize=12)
    ax1.set_title('数据集划分 (按累计可交易数量)', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 图2：累计曲线 + 分割竖线 + 累计阈值横线
    ax2 = axes[1]
    ax2.plot(df['date'], df['cumsum'], color='blue', linewidth=1.5)
    for r in results:
        ax2.axvline(x=pd.Timestamp(r['end_date']), color=colors[r['dataset']],
                    linestyle='-', alpha=0.7)
    for r in results[:-1]:
        end_cum = df[df['date'] <= pd.Timestamp(r['end_date'])]['cumsum'].iloc[-1]
        ax2.axhline(y=end_cum, color=colors[r['dataset']], linestyle='--', alpha=0.7,
                    label=f"{end_cum / total_sum * 100:.0f}%")
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('累计可交易数量', fontsize=12)
    ax2.set_title('累计可交易数量', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
