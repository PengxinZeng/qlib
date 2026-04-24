"""
统计3个数据源中各基金的起止日期
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

# 路径配置
BASE_DIR = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase")
QLIB_DIR = BASE_DIR / "qlib_data_260415/qlib_etf_index"
NORMED_DIR = BASE_DIR / "stock_data/normed"
MERGED2_DIR = BASE_DIR / "stock_data/merged2"
OUTPUT_DIR = BASE_DIR / "stock_data"


def get_date_range_from_csv(csv_path: Path) -> tuple:
    """从csv文件获取日期范围"""
    try:
        df = pd.read_csv(csv_path)
        if 'date' not in df.columns or len(df) == 0:
            return None, None
        dates = pd.to_datetime(df['date'])
        return dates.min(), dates.max()
    except Exception as e:
        return None, None


def get_date_range_from_qlib(qlib_dir: Path) -> dict:
    """从qlib格式获取日期范围"""
    instruments_path = qlib_dir / "instruments/all.txt"
    if not instruments_path.exists():
        return {}

    result = {}
    with open(instruments_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                symbol = parts[0]
                # qlib symbol格式可能是 159919_NORMED，提取纯数字部分
                fund_code = ''.join(c for c in symbol if c.isdigit())
                if len(fund_code) == 6:
                    start = pd.to_datetime(parts[1])
                    end = pd.to_datetime(parts[2])
                    result[fund_code] = (start, end)
    return result


def main():
    print("=" * 80)
    print("数据源起止日期统计")
    print("=" * 80)

    # 1. 从qlib获取日期范围
    print("\n[1/3] 读取 qlib_etf_index ...")
    qlib_ranges = get_date_range_from_qlib(QLIB_DIR)
    print(f"    共 {len(qlib_ranges)} 个instrument")

    # 2. 从normed获取日期范围
    print("\n[2/3] 读取 normed ...")
    normed_files = sorted(NORMED_DIR.glob("*_normed.csv"))
    normed_ranges = {}
    for f in tqdm(normed_files, desc="    处理中"):
        fund_code = f.stem.replace('_normed', '')
        start, end = get_date_range_from_csv(f)
        if start is not None:
            normed_ranges[fund_code] = (start, end)
    print(f"    共 {len(normed_ranges)} 个文件")

    # 3. 从merged2获取日期范围
    print("\n[3/3] 读取 merged2 ...")
    merged2_files = sorted(MERGED2_DIR.glob("*_merged.csv"))
    merged2_ranges = {}
    for f in tqdm(merged2_files, desc="    处理中"):
        fund_code = f.stem.replace('_merged', '')
        start, end = get_date_range_from_csv(f)
        if start is not None:
            merged2_ranges[fund_code] = (start, end)
    print(f"    共 {len(merged2_ranges)} 个文件")

    # 4. 合并所有基金代码
    all_funds = set()
    all_funds.update(qlib_ranges.keys())
    all_funds.update(normed_ranges.keys())
    all_funds.update(merged2_ranges.keys())

    # 过滤掉非基金代码（如000903）
    all_funds = {f for f in all_funds if f.isdigit() and len(f) == 6}

    print(f"\n共发现 {len(all_funds)} 只基金")

    # 5. 构建汇总表
    results = []
    for fund_code in sorted(all_funds):
        row = {'fund_code': fund_code}

        # qlib
        if fund_code in qlib_ranges:
            start, end = qlib_ranges[fund_code]
            row['qlib_start'] = start.strftime('%Y-%m-%d') if start else ''
            row['qlib_end'] = end.strftime('%Y-%m-%d') if end else ''
        else:
            row['qlib_start'] = ''
            row['qlib_end'] = ''

        # normed
        if fund_code in normed_ranges:
            start, end = normed_ranges[fund_code]
            row['normed_start'] = start.strftime('%Y-%m-%d') if start else ''
            row['normed_end'] = end.strftime('%Y-%m-%d') if end else ''
        else:
            row['normed_start'] = ''
            row['normed_end'] = ''

        # merged2
        if fund_code in merged2_ranges:
            start, end = merged2_ranges[fund_code]
            row['merged2_start'] = start.strftime('%Y-%m-%d') if start else ''
            row['merged2_end'] = end.strftime('%Y-%m-%d') if end else ''
        else:
            row['merged2_start'] = ''
            row['merged2_end'] = ''

        results.append(row)

    # 6. 保存结果
    df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "fund_date_ranges_comparison.csv"
    df.to_csv(output_path, index=False)

    print(f"\n" + "=" * 80)
    print("结果已保存")
    print("=" * 80)
    print(f"输出文件: {output_path}")

    # 打印汇总
    print("\n" + "-" * 80)
    print("各数据源基金数量:")
    print("-" * 80)
    print(f"  qlib_etf_index: {len(qlib_ranges)} 个")
    print(f"  normed:         {len(normed_ranges)} 个")
    print(f"  merged2:        {len(merged2_ranges)} 个")

    print("\n" + "-" * 80)
    print("数据对比表:")
    print("-" * 80)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()