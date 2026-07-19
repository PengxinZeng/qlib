"""
按每日可交易股票数量的累计和划分数据集
5 个集合: 训练(50%), 验证(10%), 测试A1(20%), 测试A2(10%), 测试B(10%)

输入：{QLIB_DIR}/data_distribution/tradeable_stats.csv（由 tradeable_distribution.py 产出）。
划分方式：以每日可交易数量的累计和(cumsum)按比例切分，使各集合"信息量"（而非天数）近似目标占比。

输出内容（保存至 {QLIB_DIR}/dataset_split/）：
1. split_by_cumsum.csv      — 各集合汇总：起止日期、天数、累计可交易量及占比、天数占比、
                              平均/最小/最大可交易数量。
2. split_info_cumsum.json   — 各集合关键信息的 JSON（起止日期、天数、累计量、占比、平均值）。
3. daily_split_cumsum.csv   — 每日明细并标注所属集合(dataset 列)。
4. dataset_split_cumsum.png — 双子图可视化：
                              上图=每日可交易数量折线，各集合区间按颜色填充并在图例注明天数与占比；
                              下图=累计可交易数量曲线，叠加各集合分割竖线与 50/60/80/90% 阈值横线。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent))
from split_lib import split_by_cumsum, build_results, plot_split

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
    names = ['train', 'valid', 'test_a1', 'test_a2', 'test_b']
    ratios = (0.50, 0.10, 0.20, 0.10, 0.10)
    split_cfg = list(zip(names, ratios))

    # 划分
    print("\n[2/4] 数据划分...")
    splits, total_sum = split_by_cumsum(df, split_cfg)

    # 分析并保存结果
    print("\n[3/4] 保存结果...")

    results = build_results(df, splits, total_sum)

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
    plot_split(df, results, total_sum, OUTPUT_DIR / "dataset_split_cumsum.png")
    print(f"可视化已保存: {OUTPUT_DIR / 'dataset_split_cumsum.png'}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()