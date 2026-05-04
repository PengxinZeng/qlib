"""
按每日可交易股票数量累计和划分数据集
5个集合: 训练(50%), 验证(10%), 测试A1(20%), 测试A2(10%), 测试B(10%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# 路径配置
QLIB_DIR = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415/qlib_etf_index")
OUTPUT_DIR = QLIB_DIR / "dataset_split"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_tradeable_stats():
    """加载每日可交易股票统计数据"""
    stats_path = QLIB_DIR / "data_distribution/tradeable_stats.csv"
    df = pd.read_csv(stats_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def split_by_cumsum(df: pd.DataFrame, ratios: tuple) -> list:
    """
    按每日可交易股票数量的累计和划分
    返回: [(dataset_name, start_idx, end_idx), ...]
    """
    df = df.copy().reset_index(drop=True)
    df['cumsum'] = df['tradeable_count'].cumsum()
    total_sum = df['cumsum'].iloc[-1]

    names = ['train', 'valid', 'test_a1', 'test_a2', 'test_b']
    splits = []

    prev_end_idx = -1
    prev_threshold = 0

    for name, ratio in zip(names, ratios):
        target_threshold = prev_threshold + total_sum * ratio

        # 找到最后一个 cumsum <= target_threshold 的索引
        mask = df['cumsum'] <= target_threshold
        if mask.any():
            end_idx = mask[mask].index[-1]
        else:
            end_idx = 0

        splits.append((name, prev_end_idx + 1, end_idx))
        prev_end_idx = end_idx
        prev_threshold = df.iloc[end_idx]['cumsum']

    return splits, total_sum


def main():
    print("=" * 60)
    print("数据集划分 (按累计可交易数量)")
    print("=" * 60)

    # 加载数据
    print("\n[1/4] 加载数据...")
    df = load_tradeable_stats()
    df = df.reset_index(drop=True)
    print(f"    总交易日: {len(df)}")
    print(f"    日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")

    # 计算累计和
    df['cumsum'] = df['tradeable_count'].cumsum()
    total_sum = df['cumsum'].iloc[-1]
    print(f"    总可交易数量(累计): {total_sum}")

    # 划分比例
    ratios = (0.50, 0.10, 0.20, 0.10, 0.10)

    # 划分
    print("\n[2/4] 数据划分...")
    splits, total_sum = split_by_cumsum(df, ratios)

    # 分析并保存结果
    print("\n[3/4] 保存结果...")

    results = []
    for name, start_idx, end_idx in splits:
        subset = df.iloc[start_idx:end_idx + 1]

        results.append({
            'dataset': name,
            'start_date': subset['date'].iloc[0].strftime('%Y-%m-%d'),
            'end_date': subset['date'].iloc[-1].strftime('%Y-%m-%d'),
            'start_idx': start_idx,
            'end_idx': end_idx,
            'days': len(subset),
            'total_count': int(subset['tradeable_count'].sum()),
            'pct_of_total_count': f"{subset['tradeable_count'].sum() / total_sum * 100:.1f}%",
            'pct_of_days': f"{len(subset) / len(df) * 100:.1f}%",
            'avg_tradeable': f"{subset['tradeable_count'].mean():.1f}",
            'min_tradeable': int(subset['tradeable_count'].min()),
            'max_tradeable': int(subset['tradeable_count'].max()),
        })

    results_df = pd.DataFrame(results)
    print("\n划分详情:")
    print("-" * 100)
    print(results_df.to_string(index=False))

    # 保存详细结果
    results_df.to_csv(OUTPUT_DIR / "split_by_cumsum.csv", index=False)

    # 保存JSON格式
    split_info = {}
    for r in results:
        split_info[r['dataset']] = {
            'start_date': r['start_date'],
            'end_date': r['end_date'],
            'days': r['days'],
            'total_count': r['total_count'],
            'pct_of_total_count': r['pct_of_total_count'],
            'avg_tradeable': r['avg_tradeable'],
        }

    with open(OUTPUT_DIR / "split_info_cumsum.json", 'w') as f:
        json.dump(split_info, f, indent=2)

    # 保存每日数据
    df['dataset'] = 'unknown'
    for name, start_idx, end_idx in splits:
        df.loc[start_idx:end_idx, 'dataset'] = name

    df.to_csv(OUTPUT_DIR / "daily_split_cumsum.csv", index=False)
    print(f"\n结果已保存至: {OUTPUT_DIR}")

    # 可视化
    print("\n[4/4] 可视化...")
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    colors = {'train': '#2ecc71', 'valid': '#3498db', 'test_a1': '#e74c3c', 'test_a2': '#f39c12', 'test_b': '#9b59b6'}

    # 图1: 每日可交易数量
    ax1 = axes[0]
    for _, row in results_df.iterrows():
        subset = df[(df['date'] >= row['start_date']) & (df['date'] <= row['end_date'])]
        ax1.fill_between(subset['date'], 0, subset['tradeable_count'],
                       color=colors[row['dataset']], alpha=0.4, label=f"{row['dataset']} ({row['days']}天, {row['pct_of_total_count']})")

    ax1.plot(df['date'], df['tradeable_count'], color='gray', linewidth=0.5)
    ax1.set_ylabel('可交易股票数', fontsize=12)
    ax1.set_title('数据集划分 (按累计可交易数量)', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 图2: 累计和
    ax2 = axes[1]
    ax2.plot(df['date'], df['cumsum'], color='blue', linewidth=1.5)

    # 标记分割点
    for _, row in results_df.iterrows():
        end_date = pd.Timestamp(row['end_date'])
        ax2.axvline(x=end_date, color=colors[row['dataset']], linestyle='-', alpha=0.7)

    # 添加阈值线
    ax2.axhline(y=total_sum * 0.5, color='green', linestyle='--', alpha=0.7, label='50%')
    ax2.axhline(y=total_sum * 0.6, color='blue', linestyle='--', alpha=0.7, label='60%')
    ax2.axhline(y=total_sum * 0.8, color='red', linestyle='--', alpha=0.7, label='80%')
    ax2.axhline(y=total_sum * 0.9, color='orange', linestyle='--', alpha=0.7, label='90%')

    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('累计可交易数量', fontsize=12)
    ax2.set_title('累计可交易数量', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dataset_split_cumsum.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化已保存: {OUTPUT_DIR / 'dataset_split_cumsum.png'}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()