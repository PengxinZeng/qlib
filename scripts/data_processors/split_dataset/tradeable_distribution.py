"""
统计 qlib 市场中每日可交易股票数量分布

数据来源：qlib 二进制目录（读 calendars/day.txt 与各 instrument 的 close.day.bin，
close 非 NaN 即视为当日可交易）。

输出内容（保存至 {QLIB_DIR}/data_distribution/）：
1. tradeable_distribution.png — 每日可交易股票数量的时间序列折线图，
                                左上角标注总交易日/平均/最大/最新可交易数量。
2. yearly_distribution.png    — 按年聚合：折线为年平均可交易数量，
                                阴影带为当年 min-max 范围。
3. tradeable_stats.csv        — 每日统计明细：date, tradeable_count, year, month。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
QLIB_DIR = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415/qlib_etf_index")
OUTPUT_DIR = QLIB_DIR / "data_distribution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def read_calendar(calendar_path: Path) -> list:
    """读取日历文件"""
    with open(calendar_path, 'r') as f:
        dates = [line.strip() for line in f.readlines()]
    return pd.to_datetime(dates)


def read_bin_file(bin_path: Path) -> tuple:
    """
    读取qlib bin文件
    返回: (date_index, values)
    """
    data = np.fromfile(bin_path, dtype='<f')
    if len(data) == 0:
        return 0, np.array([])
    date_index = int(data[0])
    values = data[1:]
    return date_index, values


def count_tradeable_instruments(qlib_dir: Path, calendar: list):
    """
    统计每个交易日可交易(有收盘价)的股票数量
    """
    features_dir = qlib_dir / "features"
    instrument_dirs = sorted(features_dir.glob("*_normed"))

    n_days = len(calendar)
    tradeable_counts = np.zeros(n_days, dtype=int)

    print(f"统计 {len(instrument_dirs)} 个instrument的数据分布...")

    for inst_dir in tqdm(instrument_dirs):
        close_bin = inst_dir / "close.day.bin"
        if not close_bin.exists():
            continue

        date_index, values = read_bin_file(close_bin)

        if len(values) == 0:
            continue

        # 标记有数据的日期
        for i in range(len(values)):
            if not np.isnan(values[i]) and (date_index + i) < n_days:
                tradeable_counts[date_index + i] += 1

    return tradeable_counts


def plot_distribution(calendar: list, tradeable_counts: list, output_path: Path):
    """绘制可交易数量分布图"""
    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(calendar, tradeable_counts, linewidth=0.8, alpha=0.8)

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('可交易股票数量', fontsize=12)
    ax.set_title('每日可交易股票数量分布', fontsize=14)

    ax.grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = f"总交易日: {len(calendar)}\n"
    stats_text += f"平均可交易: {np.mean(tradeable_counts):.1f}\n"
    stats_text += f"最大可交易: {np.max(tradeable_counts)}\n"
    stats_text += f"最新可交易: {tradeable_counts[-1]}"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图表已保存: {output_path}")


def plot_yearly_distribution(calendar: list, tradeable_counts: list, output_path: Path):
    """按年统计可交易数量"""
    df = pd.DataFrame({'date': calendar, 'count': tradeable_counts})
    df['year'] = df['date'].dt.year

    yearly_stats = df.groupby('year')['count'].agg(['mean', 'min', 'max', 'count'])

    fig, ax = plt.subplots(figsize=(14, 8))

    years = yearly_stats.index
    means = yearly_stats['mean'].values
    mins = yearly_stats['min'].values
    maxima = yearly_stats['max'].values

    ax.fill_between(years, mins, maxima, alpha=0.3, label='min-max范围')
    ax.plot(years, means, 'o-', linewidth=2, label='平均可交易数量')

    ax.set_xlabel('年份', fontsize=12)
    ax.set_ylabel('可交易股票数量', fontsize=12)
    ax.set_title('每年可交易股票数量统计', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"图表已保存: {output_path}")


def save_stats(calendar: list, tradeable_counts: list, output_path: Path):
    """保存统计数据"""
    df = pd.DataFrame({
        'date': calendar,
        'tradeable_count': tradeable_counts
    })

    # 添加年月信息
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    df.to_csv(output_path, index=False)
    print(f"统计数据已保存: {output_path}")


def main():
    print("=" * 60)
    print("qlib市场可交易股票数量统计")
    print("=" * 60)

    # 读取日历
    calendar_path = QLIB_DIR / "calendars" / "day.txt"
    calendar = read_calendar(calendar_path)
    print(f"日历文件: {calendar_path}")
    print(f"总交易日: {len(calendar)}")
    print(f"日期范围: {calendar[0]} ~ {calendar[-1]}")

    # 统计可交易数量
    tradeable_counts = count_tradeable_instruments(QLIB_DIR, calendar)

    # 绘制分布图
    print("\n生成可视化图表...")
    plot_distribution(calendar, tradeable_counts, OUTPUT_DIR / "tradeable_distribution.png")
    plot_yearly_distribution(calendar, tradeable_counts, OUTPUT_DIR / "yearly_distribution.png")

    # 保存统计
    save_stats(calendar, tradeable_counts, OUTPUT_DIR / "tradeable_stats.csv")

    # 打印部分统计信息
    print("\n" + "=" * 60)
    print("统计摘要")
    print("=" * 60)
    print(f"平均可交易: {np.mean(tradeable_counts):.1f}")
    print(f"最大可交易: {np.max(tradeable_counts)}")
    print(f"最小可交易: {np.min(tradeable_counts)}")
    print(f"最新可交易: {tradeable_counts[-1]}")


if __name__ == "__main__":
    main()