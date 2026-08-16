"""
将etf_index数据转换为qlib数据集
保留所有列：基金k线、指数k线、指数估值数据、国债收益率

环境要求: conda activate rdagent
"""

import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from qlib.utils import fname_to_code, code_to_fname
from qlib.constant import REG_CN
import fire

# 跨平台路径集中配置（Mac / Windows 兼容）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import path_config  # noqa: E402


def normalize_field_name(name: str) -> str:
    """标准化字段名，转为小写"""
    return name.lower()


def save_symbol_data(args):
    """保存单个symbol的特征数据"""
    symbol, combined_df, calendar_list, date_to_idx, all_fields, features_dir, freq = args

    symbol_df = combined_df[combined_df['symbol'] == symbol].copy()
    symbol_df = symbol_df.set_index('date').sort_index()

    features_symbol_dir = features_dir / code_to_fname(symbol.lower())
    features_symbol_dir.mkdir(parents=True, exist_ok=True)

    # date_index: symbol_df第一行在calendar_list中的索引
    date_index = date_to_idx[symbol_df.index.min()]

    for field in all_fields:
        if field not in symbol_df.columns:
            continue

        bin_path = features_symbol_dir / f"{field.lower()}.{freq}.bin"

        # field_data长度 = calendar_list长度
        # field_data[i]对应日历索引(date_index + i)的值
        field_data = np.full(len(calendar_list), np.nan, dtype=np.float32)

        for date, value in symbol_df[field].items():
            if date in date_to_idx:
                idx = date_to_idx[date]
                # 转换为field_data的相对索引
                rel_idx = idx - date_index
                if 0 <= rel_idx < len(field_data):
                    field_data[rel_idx] = value if pd.notna(value) else np.nan

        # 保存为bin文件 (格式: date_index + values)
        np.hstack([date_index, field_data]).astype('<f').tofile(str(bin_path.resolve()))


QLIB_BASE = path_config.QLIB_BASE
DEFAULT_DATA_PATH = f"{path_config.MERGED_DIR}"
DEFAULT_QLIB_DIR  = f"{path_config.QLIB_DATA_DIR}"


def convert_etf_index_to_qlib(
    data_path: str = DEFAULT_DATA_PATH,
    qlib_dir: str = DEFAULT_QLIB_DIR,
    freq: str = "day",
    max_workers: int = 16,
    date_field_name: str = "date",
    file_suffix: str = ".csv",
    symbol_field_name: str = "symbol",
):
    """
    将etf_index数据转换为qlib格式

    Parameters
    ----------
    data_path : str
        etf_index数据目录路径
    qlib_dir : str
        qlib数据输出目录
    freq : str
        数据频率
    max_workers : int
        并行工作线程数
    """
    from concurrent.futures import ProcessPoolExecutor

    data_path = Path(data_path)
    qlib_dir = Path(qlib_dir)

    # 读取所有csv文件
    csv_files = sorted(data_path.glob(f"*{file_suffix}"))
    print(f"找到 {len(csv_files)} 个数据文件")

    # 目录结构
    calendars_dir = qlib_dir / "calendars"
    features_dir = qlib_dir / "features"
    instruments_dir = qlib_dir / "instruments"

    calendars_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    instruments_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有日期和instruments
    all_dates = set()
    instruments_data = []

    # 收集所有数据
    all_data = []

    print("读取数据文件...")
    for file_path in tqdm(csv_files):
        fund_code = file_path.stem  # 如 159919

        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])

        if df.empty:
            continue

        # 添加symbol列
        df['symbol'] = fund_code

        # 标准化字段名（小写）
        rename_map = {col: normalize_field_name(col) for col in df.columns}
        df = df.rename(columns=rename_map)

        all_data.append(df)

        # 收集日期
        dates = df[date_field_name].tolist()
        all_dates.update(dates)

        # instruments信息
        start_date = df[date_field_name].min()
        end_date = df[date_field_name].max()
        instruments_data.append({
            'symbol': fund_code.upper(),
            'start_datetime': start_date.strftime('%Y-%m-%d'),
            'end_datetime': end_date.strftime('%Y-%m-%d')
        })

    if not all_data:
        print("没有数据可处理")
        return

    # 合并所有数据
    print("合并数据...")
    combined_df = pd.concat(all_data, ignore_index=True)

    # 获取所有字段（排除data_source等非数值列）
    all_fields = [col for col in combined_df.columns if col not in [date_field_name, symbol_field_name, 'data_source']]
    print(f"字段数量: {len(all_fields)}")
    print(f"字段列表: {all_fields}")

    # 保存日历
    print("保存日历...")
    sorted_dates = sorted(all_dates)
    calendars_path = calendars_dir / f"{freq}.txt"
    with open(calendars_path, 'w') as f:
        for d in sorted_dates:
            if isinstance(d, pd.Timestamp):
                f.write(d.strftime('%Y-%m-%d') + '\n')
            else:
                f.write(str(d) + '\n')

    # 保存instruments
    print("保存instruments...")
    instruments_path = instruments_dir / "all.txt"
    with open(instruments_path, 'w') as f:
        for inst in instruments_data:
            f.write(f"{inst['symbol']}\t{inst['start_datetime']}\t{inst['end_datetime']}\n")

    # 为每个symbol创建特征目录并保存bin文件
    print("保存特征数据...")

    symbols = combined_df[symbol_field_name].unique()
    calendar_list = sorted(all_dates)

    # 创建日期到索引的映射
    date_to_idx = {d: i for i, d in enumerate(calendar_list)}

    # 准备参数
    args_list = [
        (symbol, combined_df, calendar_list, date_to_idx, all_fields, features_dir, freq)
        for symbol in symbols
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(save_symbol_data, args_list), total=len(symbols), desc="保存特征"))

    print(f"\n转换完成!")
    print(f"qlib数据目录: {qlib_dir}")
    print(f"日历文件: {calendars_path}")
    print(f"instruments文件: {instruments_path}")
    print(f"特征目录: {features_dir}")
    print(f"symbol数量: {len(symbols)}")
    print(f"日期范围: {min(sorted_dates)} ~ {max(sorted_dates)}")


if __name__ == "__main__":
    fire.Fire({
        "convert": convert_etf_index_to_qlib,
    })