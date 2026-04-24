"""
合并基金K线数据与指数数据
- 保留所有原始列和行
- 输出到 merged2
"""
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# 配置
BASE_DIR = Path('/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/stock_data')
HFQ_DIR = BASE_DIR / 'fund_kline_hfq'
RAW_DIR = BASE_DIR / 'fund_kline_raw'
INDEX_DIR = BASE_DIR / 'index_data'
MERGED_DIR = BASE_DIR / 'merged2'
REPORT_DIR = BASE_DIR / 'report'
FUNDS_LIST = '/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415/source/funds_list.csv'

MERGED_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_funds_list():
    """加载基金列表，过滤有效的track_target_file"""
    df = pd.read_csv(FUNDS_LIST, comment='#', dtype=str)
    df = df.dropna(subset=['fund_code', 'track_target_file'])
    df = df[df['track_target_file'] != 'N/A']
    df = df[df['track_target_file'].str.strip() != '']
    return df


def rename_fund_columns(df, prefix):
    """重命名基金k线列，添加前缀区分hfq和raw"""
    rename_map = {
        'open': f'{prefix}_open',
        'high': f'{prefix}_high',
        'low': f'{prefix}_low',
        'close': f'{prefix}_close',
        'volume': f'{prefix}_volume'
    }
    return df.rename(columns=rename_map)


def load_index_data(index_file: str) -> pd.DataFrame:
    """加载指数数据，保留所有列"""
    index_path = INDEX_DIR / index_file
    if not index_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(index_path)
    df['date'] = pd.to_datetime(df['date'])

    # 重命名指数的k线列，添加index_前缀
    kline_rename = {
        'open': 'index_open',
        'high': 'index_high',
        'low': 'index_low',
        'close': 'index_close',
        'volume': 'index_volume'
    }
    for old, new in kline_rename.items():
        if old in df.columns:
            df = df.rename(columns={old: new})

    # 删除code列（冗余信息）
    if 'code' in df.columns:
        df = df.drop(columns=['code'])

    return df


def merge_fund_data(fund_code: str, index_file: str) -> pd.DataFrame:
    """合并单个基金的所有数据"""
    hfq_path = HFQ_DIR / f"{fund_code}.csv"
    raw_path = RAW_DIR / f"{fund_code}.csv"

    hfq_df = pd.DataFrame()
    raw_df = pd.DataFrame()

    # 读取后复权数据
    if hfq_path.exists():
        hfq_df = pd.read_csv(hfq_path)
        hfq_df['date'] = pd.to_datetime(hfq_df['date'])
        hfq_df = rename_fund_columns(hfq_df, 'hfq')

    # 读取除权数据
    if raw_path.exists():
        raw_df = pd.read_csv(raw_path)
        raw_df['date'] = pd.to_datetime(raw_df['date'])
        raw_df = rename_fund_columns(raw_df, 'raw')

    # 读取指数数据
    index_df = load_index_data(index_file)

    # 合并所有数据（保留所有行和列）
    merged = pd.DataFrame()

    if not hfq_df.empty:
        merged = hfq_df

    if not raw_df.empty:
        if merged.empty:
            merged = raw_df
        else:
            merged = merged.merge(raw_df, on='date', how='outer')

    if not index_df.empty:
        if merged.empty:
            merged = index_df
        else:
            merged = merged.merge(index_df, on='date', how='outer')

    if not merged.empty:
        merged = merged.sort_values('date').reset_index(drop=True)

    return merged


def analyze_data(df: pd.DataFrame, fund_code: str, index_file: str) -> dict:
    """分析数据完整性"""
    analysis = {
        'fund_code': fund_code,
        'index_file': index_file,
        'total_rows': len(df),
        'date_range': {'start': None, 'end': None},
        'columns': list(df.columns),
        'completeness': {},
        'issues': []
    }

    if len(df) == 0:
        analysis['issues'].append('无数据')
        return analysis

    analysis['date_range']['start'] = str(df['date'].min())
    analysis['date_range']['end'] = str(df['date'].max())

    for col in df.columns:
        non_null = df[col].notna().sum()
        total = len(df)
        pct = (non_null / total * 100) if total > 0 else 0
        analysis['completeness'][col] = {'non_null': int(non_null), 'total': int(total), 'pct': round(pct, 2)}

    # 检查关键列
    key_cols = ['hfq_close', 'raw_close', 'index_close']
    for col in key_cols:
        if col not in df.columns or df[col].isna().all():
            analysis['issues'].append(f'{col} 无数据')

    return analysis


def main():
    print("=" * 60)
    print("基金数据合并")
    print("=" * 60)

    # 加载基金列表
    print("\n[1/4] 加载基金列表...")
    df = load_funds_list()
    print(f"    共 {len(df)} 只基金有有效跟踪目标")

    # 合并数据
    print("\n[2/4] 合并数据...")
    merge_results = []
    all_analyses = []
    success_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="    处理中"):
        fund_code = row['fund_code']
        fund_name = row['fund_name']
        index_file = row['track_target_file']

        # 合并数据
        merged = merge_fund_data(fund_code, index_file)

        if not merged.empty:
            # 保存
            merged_path = MERGED_DIR / f"{fund_code}_merged.csv"
            merged.to_csv(merged_path, index=False)
            success_count += 1

            # 分析
            analysis = analyze_data(merged, fund_code, index_file)
            all_analyses.append(analysis)

            date_start = merged['date'].min()
            date_end = merged['date'].max()
            hfq_rows = len(merged[merged['hfq_close'].notna()]) if 'hfq_close' in merged.columns else 0
            raw_rows = len(merged[merged['raw_close'].notna()]) if 'raw_close' in merged.columns else 0
            index_rows = len(merged[merged['index_close'].notna()]) if 'index_close' in merged.columns else 0
        else:
            date_start = date_end = None
            hfq_rows = raw_rows = index_rows = 0
            analysis = {'fund_code': fund_code, 'index_file': index_file, 'total_rows': 0,
                       'date_range': {'start': None, 'end': None}, 'columns': [],
                       'completeness': {}, 'issues': ['合并后无数据']}
            all_analyses.append(analysis)

        merge_results.append({
            'fund_code': fund_code,
            'fund_name': fund_name,
            'index_file': index_file,
            'hfq_rows': hfq_rows,
            'raw_rows': raw_rows,
            'index_rows': index_rows,
            'merged_rows': len(merged),
            'date_start': date_start,
            'date_end': date_end,
        })

    print(f"    成功合并 {success_count} 只基金数据")

    # 保存合并报告
    print("\n[3/4] 保存分析报告...")
    results_df = pd.DataFrame(merge_results)
    results_path = REPORT_DIR / 'merge_status.csv'
    results_df.to_csv(results_path, index=False)

    # 保存详细分析JSON
    report = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_funds': len(df),
        'success_count': success_count,
        'fund_analyses': all_analyses
    }
    json_path = REPORT_DIR / f'merge_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 打印汇总
    print("\n[4/4] 数据统计:")
    print(f"    - 输出目录: {MERGED_DIR}")
    print(f"    - 报告文件: {results_path}")
    print(f"    - 详细报告: {json_path}")
    print(f"    - 有后复权数据: {sum(1 for r in merge_results if r['hfq_rows'] > 0)}")
    print(f"    - 有除权数据: {sum(1 for r in merge_results if r['raw_rows'] > 0)}")
    print(f"    - 有指数数据: {sum(1 for r in merge_results if r['index_rows'] > 0)}")

    print("\n" + "=" * 60)
    print("合并完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()