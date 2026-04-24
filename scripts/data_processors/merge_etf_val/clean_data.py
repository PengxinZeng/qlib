"""
清洗基金数据，按类型分类处理
1. 有后复权数据 -> 使用后复权k线
2. 只有除权数据 -> 使用除权k线
3. 拼接指数数据，保留所有列
4. 筛选k线和指数数据均完整的行
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# 配置
BASE_DIR = Path('/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/stock_data')
MERGED_DIR = BASE_DIR / 'merged2'
NORMED_DIR = BASE_DIR / 'normed'
REPORT_DIR = BASE_DIR / 'report'
STATUS_FILE = REPORT_DIR / 'merge_status.csv'

NORMED_DIR.mkdir(parents=True, exist_ok=True)


def classify_funds():
    """根据merge_status.csv将基金分成3类"""
    df = pd.read_csv(STATUS_FILE)

    has_hfq = df[df['hfq_rows'] > 0].to_dict('records')
    only_raw = df[(df['hfq_rows'] == 0) & (df['raw_rows'] > 0)].to_dict('records')
    no_data = df[(df['hfq_rows'] == 0) & (df['raw_rows'] == 0)].to_dict('records')

    return has_hfq, only_raw, no_data


def get_index_cols(df: pd.DataFrame) -> list:
    """获取指数列（不包括date）"""
    return [c for c in df.columns if c.startswith('index_')]


def get_fund_cols(df: pd.DataFrame, fund_type: str) -> list:
    """获取基金k线列"""
    prefix = 'hfq' if fund_type == 'hfq' else 'raw'
    return [c for c in df.columns if c.startswith(f'{prefix}_')]


def rename_fund_cols(df: pd.DataFrame, fund_type: str) -> pd.DataFrame:
    """统一基金列名：hfq_* 或 raw_* -> open/close/high/low/volume"""
    prefix = 'hfq' if fund_type == 'hfq' else 'raw'
    rename_map = {
        f'{prefix}_open': 'open',
        f'{prefix}_high': 'high',
        f'{prefix}_low': 'low',
        f'{prefix}_close': 'close',
        f'{prefix}_volume': 'volume',
    }
    return df.rename(columns=rename_map)


def get_index_valuation_cols() -> list:
    """获取指数估值相关列"""
    return ['amount', 'pctChg', 'pe_static_equal_weight', 'pe_static', 'pe_static_median',
            'pe_ttm_equal_weight', 'pe_ttm', 'pe_ttm_median', 'pb', 'pb_equal_weight', 'pb_median']


def clean_fund_data(fund_code: str, fund_type: str) -> pd.DataFrame:
    """
    清洗单个基金数据
    fund_type: 'hfq' or 'raw'
    """
    merged_path = MERGED_DIR / f"{fund_code}_merged.csv"
    if not merged_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(merged_path)
    df['date'] = pd.to_datetime(df['date'])

    if len(df) == 0:
        return pd.DataFrame()

    # 获取基金k线列
    fund_prefix = 'hfq' if fund_type == 'hfq' else 'raw'
    fund_kline_cols = [c for c in df.columns if c.startswith(f'{fund_prefix}_')]

    # 保留所有指数列：index_* + 估值列
    index_cols = [c for c in df.columns if c.startswith('index_')]
    valuation_cols = [c for c in get_index_valuation_cols() if c in df.columns]
    all_index_cols = index_cols + valuation_cols

    # 提取基金k线数据和指数数据
    available_fund_cols = [c for c in fund_kline_cols if c in df.columns]
    available_index_cols = [c for c in all_index_cols if c in df.columns]

    fund_df = df[['date'] + available_fund_cols].copy()
    index_df = df[['date'] + available_index_cols].copy()

    # 合并
    merged = fund_df.merge(index_df, on='date', how='outer')
    merged = merged.sort_values('date').reset_index(drop=True)

    # 筛选：k线和指数数据均完整的行
    fund_close_col = f'{fund_prefix}_close'
    index_close_col = 'index_close'

    if fund_close_col in merged.columns and index_close_col in merged.columns:
        before_count = len(merged)
        merged = merged.dropna(subset=[fund_close_col, index_close_col])
        after_count = len(merged)
    else:
        before_count = after_count = len(merged)

    # 统一列名：hfq_* 或 raw_* -> open/close/high/low/volume
    merged = rename_fund_cols(merged, fund_type)

    # 添加数据来源标记
    merged['data_source'] = fund_type

    return merged, before_count, after_count


def main():
    print("=" * 60)
    print("基金数据清洗")
    print("=" * 60)

    # 分类
    print("\n[1/5] 分类基金...")
    has_hfq, only_raw, no_data = classify_funds()

    print(f"    有后复权数据: {len(has_hfq)} 只")
    print(f"    只有除权数据: {len(only_raw)} 只")
    print(f"    后复权除权都没有: {len(no_data)} 只")

    # 清洗数据
    print("\n[2/5] 清洗有后复权数据...")
    hfq_results = []
    for record in tqdm(has_hfq, desc="    处理中"):
        fund_code = record['fund_code']
        merged, before, after = clean_fund_data(fund_code, 'hfq')
        if not merged.empty:
            output_path = NORMED_DIR / f"{fund_code}_normed.csv"
            merged.to_csv(output_path, index=False)
        hfq_results.append({
            'fund_code': fund_code,
            'fund_name': record['fund_name'],
            'index_file': record['index_file'],
            'before_rows': before,
            'after_rows': after,
            'kept_pct': round(after / before * 100, 2) if before > 0 else 0
        })

    print(f"    完成 {len(has_hfq)} 只")

    print("\n[3/5] 清洗只有除权数据...")
    raw_results = []
    for record in tqdm(only_raw, desc="    处理中"):
        fund_code = record['fund_code']
        merged, before, after = clean_fund_data(fund_code, 'raw')
        if not merged.empty:
            output_path = NORMED_DIR / f"{fund_code}_normed.csv"
            merged.to_csv(output_path, index=False)
        raw_results.append({
            'fund_code': fund_code,
            'fund_name': record['fund_name'],
            'index_file': record['index_file'],
            'before_rows': before,
            'after_rows': after,
            'kept_pct': round(after / before * 100, 2) if before > 0 else 0
        })

    print(f"    完成 {len(only_raw)} 只")

    # 保存报告
    print("\n[4/5] 保存分析报告...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 汇总结果
    all_results = hfq_results + raw_results
    results_df = pd.DataFrame(all_results)
    results_path = REPORT_DIR / f'normed_status_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)

    # 保存无数据基金
    no_data_df = pd.DataFrame(no_data)
    if not no_data_df.empty:
        no_data_path = REPORT_DIR / f'normed_no_data_{timestamp}.csv'
        no_data_df.to_csv(no_data_path, index=False)

    # 统计
    total_input = len(has_hfq) + len(only_raw)
    total_output = len([r for r in all_results if r['after_rows'] > 0])

    print("\n[5/5] 统计信息:")
    print(f"    输出目录: {NORMED_DIR}")
    print(f"    报告文件: {results_path}")
    print(f"    处理成功: {total_output}/{total_input}")
    print(f"    无数据基金: {len(no_data)} 只")

    # 打印详细结果
    print("\n" + "-" * 60)
    print("有后复权数据基金:")
    print("-" * 60)
    print(f"{'fund_code':<12} {'fund_name':<20} {'before':<10} {'after':<10} {'kept%':<8}")
    for r in hfq_results:
        print(f"{r['fund_code']:<12} {r['fund_name']:<20} {r['before_rows']:<10} {r['after_rows']:<10} {r['kept_pct']:<8}")

    print("\n" + "-" * 60)
    print("只有除权数据基金:")
    print("-" * 60)
    print(f"{'fund_code':<12} {'fund_name':<20} {'before':<10} {'after':<10} {'kept%':<8}")
    for r in raw_results:
        print(f"{r['fund_code']:<12} {r['fund_name']:<20} {r['before_rows']:<10} {r['after_rows']:<10} {r['kept_pct']:<8}")

    print("\n" + "=" * 60)
    print("清洗完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()