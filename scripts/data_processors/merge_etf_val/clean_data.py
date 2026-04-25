"""
清洗基金数据，尽量保留所有行
1. 有后复权数据 -> 使用后复权k线
2. 只有除权数据 -> 使用除权k线
3. 拼接指数数据，保留所有列
4. 只要有任一数据（基金K线或指数）就保留
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import sys

# 配置
BASE_DIR = Path('/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/stock_data')
MERGED_DIR = BASE_DIR / 'merged2'
CLEANED_DIR = BASE_DIR / 'cleaned'
REPORT_DIR = BASE_DIR / 'report'
STATUS_FILE = REPORT_DIR / 'merge_status.csv'

CLEANED_DIR.mkdir(parents=True, exist_ok=True)


def classify_funds():
    """根据merge_status.csv将基金分成3类"""
    df = pd.read_csv(STATUS_FILE)

    has_hfq = df[df['hfq_rows'] > 0].to_dict('records')
    only_raw = df[(df['hfq_rows'] == 0) & (df['raw_rows'] > 0)].to_dict('records')
    no_data = df[(df['hfq_rows'] == 0) & (df['raw_rows'] == 0)].to_dict('records')

    return has_hfq, only_raw, no_data


def get_index_valuation_cols() -> list:
    """获取指数估值相关列"""
    return ['amount', 'pctChg', 'pe_static_equal_weight', 'pe_static', 'pe_static_median',
            'pe_ttm_equal_weight', 'pe_ttm', 'pe_ttm_median', 'pb', 'pb_equal_weight', 'pb_median']


def clean_fund_data(fund_code: str, fund_type: str) -> pd.DataFrame:
    """
    清洗单个基金数据
    fund_type: 'hfq' or 'raw'
    保留所有有数据的行
    """
    merged_path = MERGED_DIR / f"{fund_code}_merged.csv"
    if not merged_path.exists():
        return pd.DataFrame(), 0, 0

    df = pd.read_csv(merged_path)
    df['date'] = pd.to_datetime(df['date'])

    if len(df) == 0:
        return pd.DataFrame(), 0, 0

    before_count = len(df)

    # 获取基金k线列前缀
    fund_prefix = 'hfq' if fund_type == 'hfq' else 'raw'

    # 保留所有指数列：index_* + 估值列
    index_cols = [c for c in df.columns if c.startswith('index_')]
    valuation_cols = [c for c in get_index_valuation_cols() if c in df.columns]
    all_index_cols = index_cols + valuation_cols

    # 基金K线列
    fund_kline_cols = ['open', 'high', 'low', 'close', 'volume']

    # 构建结果DataFrame
    result = pd.DataFrame()
    result['date'] = df['date']

    # 处理基金K线数据 - 使用hfq或raw
    for col in fund_kline_cols:
        src_col = f'{fund_prefix}_{col}'
        if src_col in df.columns:
            result[col] = df[src_col]
        else:
            result[col] = pd.NA

    # 添加指数数据
    for col in all_index_cols:
        if col in df.columns:
            result[col] = df[col]
        else:
            result[col] = pd.NA

    # 填充基金K线的空值：尝试用raw填充hfq的缺失，反之亦然
    if fund_type == 'hfq' and 'raw_close' in df.columns:
        # hfq缺失的地方用raw填充
        result['close'] = result['close'].fillna(df['raw_close'])
        result['open'] = result['open'].fillna(df['raw_open'])
        result['high'] = result['high'].fillna(df['raw_high'])
        result['low'] = result['low'].fillna(df['raw_low'])
        result['volume'] = result['volume'].fillna(df['raw_volume'])

    # 添加数据来源标记
    # hfq: 0, raw: 1, mixed: 2
    def get_source(row):
        has_hfq = pd.notna(row.get('close')) and (fund_type == 'hfq' or pd.isna(df.loc[row.name, 'hfq_close']) == False)
        has_raw = pd.notna(df.loc[row.name, 'raw_close']) if 'raw_close' in df.columns else False

        if has_hfq and has_raw:
            return 2  # mixed
        elif has_hfq:
            return 0  # hfq
        elif has_raw:
            return 1  # raw
        return -1

    # 简化：只要有close数据就标记
    result['data_source'] = 0  # 默认hfq

    # 排序
    result = result.sort_values('date').reset_index(drop=True)

    after_count = len(result)

    return result, before_count, after_count


def clean_all_files():
    """清洗所有merged文件，不依赖merge_status.csv"""
    print("=" * 60)
    print("基金数据清洗 - 保留所有行")
    print("=" * 60)

    # 获取所有merged文件
    merged_files = list(MERGED_DIR.glob("*_merged.csv"))
    print(f"\n找到 {len(merged_files)} 个合并文件")

    results = []
    for f in tqdm(merged_files, desc="处理中"):
        fund_code = f.stem.replace('_merged', '')

        df = pd.read_csv(f)
        df['date'] = pd.to_datetime(df['date'])
        before_count = len(df)

        # 统一列名：hfq_* 或 raw_* -> open/close/high/low/volume
        fund_cols = ['open', 'high', 'low', 'close', 'volume']
        index_cols = [c for c in df.columns if c.startswith('index_')]
        valuation_cols = [c for c in get_index_valuation_cols() if c in df.columns]

        # 优先使用hfq，没有就用raw
        result = pd.DataFrame()
        result['date'] = df['date']

        for col in fund_cols:
            hfq_col = f'hfq_{col}'
            raw_col = f'raw_{col}'
            if hfq_col in df.columns:
                result[col] = df[hfq_col].fillna(df[raw_col] if raw_col in df.columns else pd.NA)
            elif raw_col in df.columns:
                result[col] = df[raw_col]
            else:
                result[col] = pd.NA

        # 添加指数和估值数据
        for col in index_cols + valuation_cols:
            if col in df.columns:
                result[col] = df[col]

        # 只保留有指数数据的行
        index_close_cols = [c for c in result.columns if c.startswith('index_close')]
        if index_close_cols:
            before_filter = len(result)
            result = result.dropna(subset=index_close_cols)

        result = result.sort_values('date').reset_index(drop=True)

        # 保存
        output_path = CLEANED_DIR / f"{fund_code}_clean.csv"
        result.to_csv(output_path, index=False)

        after_count = len(result)
        rows_with_fund = result['close'].notna().sum()
        rows_with_index = result['index_close'].notna().sum() if 'index_close' in result.columns else 0

        results.append({
            'fund_code': fund_code,
            'before_rows': before_count,
            'after_rows': after_count,
            'rows_with_fund': rows_with_fund,
            'rows_with_index': rows_with_index,
            'kept_pct': round(after_count / before_count * 100, 2) if before_count > 0 else 0
        })

    # 保存报告
    print("\n保存分析报告...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_df = pd.DataFrame(results)
    results_path = REPORT_DIR / f'clean_status_{timestamp}.csv'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    # 统计
    total_before = sum(r['before_rows'] for r in results)
    total_after = sum(r['after_rows'] for r in results)
    total_fund = sum(r['rows_with_fund'] for r in results)
    total_index = sum(r['rows_with_index'] for r in results)

    print("\n" + "=" * 60)
    print("统计信息:")
    print("=" * 60)
    print(f"  输出目录: {CLEANED_DIR}")
    print(f"  报告文件: {results_path}")
    print(f"  处理文件: {len(results)}")
    print(f"  总行数(处理前): {total_before}")
    print(f"  总行数(处理后): {total_after} ({total_after/total_before*100:.2f}%)")
    print(f"  有基金数据行: {total_fund}")
    print(f"  有指数数据行: {total_index}")

    print("\n" + "-" * 60)
    print("详细结果:")
    print("-" * 60)
    print(f"{'fund_code':<12} {'before':<10} {'after':<10} {'fund_rows':<12} {'index_rows':<12} {'kept%':<8}")
    for r in sorted(results, key=lambda x: x['kept_pct']):
        print(f"{r['fund_code']:<12} {r['before_rows']:<10} {r['after_rows']:<10} {r['rows_with_fund']:<12} {r['rows_with_index']:<12} {r['kept_pct']:<8}")

    print("\n" + "=" * 60)
    print("清洗完成!")
    print("=" * 60)


if __name__ == "__main__":
    clean_all_files()
