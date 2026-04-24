"""
时间序列归一化价格可视化 (merged2版本)
保留所有原始列
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
MERGED_DIR = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/stock_data/merged2")
REPORT_DIR = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/stock_data/report/normalized2")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def get_fund_name(fund_code: str) -> str:
    """从funds_list.csv获取基金名称"""
    try:
        funds_list = pd.read_csv("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415/source/funds_list.csv")
        fund_info = funds_list[funds_list['fund_code'] == fund_code]
        if not fund_info.empty:
            return fund_info.iloc[0]['fund_name']
    except Exception:
        pass
    return fund_code


def normalize_aligned(df: pd.DataFrame) -> pd.DataFrame:
    """
    对齐归一化：
    以基金后复权价格为基准，将跟踪指数对齐到后复权价格
    """
    df = df.copy()
    df = df.sort_values('date').reset_index(drop=True)

    price_cols = [col for col in ['raw_close', 'hfq_close', 'index_close'] if col in df.columns]

    if not price_cols:
        return df, None, None

    base_date = None
    base_price = None

    # 优先使用后复权价格作为基准
    if 'hfq_close' in df.columns:
        hfq_valid = df[df['hfq_close'].notna()]
        if len(hfq_valid) > 0:
            base_date = hfq_valid.iloc[0]['date']
            base_price = hfq_valid.iloc[0]['hfq_close']

    # 如果没有后复权，用除权价格
    if base_date is None and 'raw_close' in df.columns:
        raw_valid = df[df['raw_close'].notna()]
        if len(raw_valid) > 0:
            base_date = raw_valid.iloc[0]['date']
            base_price = raw_valid.iloc[0]['raw_close']

    if base_date is None or base_price is None or base_price == 0:
        return df, None, None

    # 后复权价格归一化（除以首日价格）
    if 'hfq_close' in df.columns:
        df['hfq_close_norm'] = df['hfq_close'] / base_price

    # 除权价格归一化（除以自己的首日价格）
    if 'raw_close' in df.columns:
        raw_first = df[df['raw_close'].notna()].iloc[0]['raw_close'] if len(df[df['raw_close'].notna()]) > 0 else None
        if raw_first and raw_first != 0:
            df['raw_close_norm'] = df['raw_close'] / raw_first

    # 指数对齐到基金后复权价格
    if 'index_close' in df.columns:
        idx_valid = df[df['index_close'].notna()]
        if len(idx_valid) == 0:
            pass
        else:
            index_first_date = idx_valid.iloc[0]['date']
            index_first_price = idx_valid.iloc[0]['index_close']

            if index_first_date > base_date:
                # 指数比基金晚出现，需要找到第一个重叠日期
                overlap = df[df['hfq_close'].notna() & df['index_close'].notna()]
                if len(overlap) > 0:
                    hfq_at_align = overlap.iloc[0]['hfq_close']
                    index_at_align = overlap.iloc[0]['index_close']

                    target_norm_value = hfq_at_align / base_price
                    df['index_close_norm'] = (df['index_close'] / index_at_align) * target_norm_value
                else:
                    df['index_close_norm'] = df['index_close'] / index_first_price
            else:
                # 指数在基金上市日或之前就有数据
                idx_row = df[df['date'] == base_date]
                if len(idx_row) > 0 and pd.notna(idx_row['index_close'].iloc[0]):
                    index_price_at_base = idx_row['index_close'].iloc[0]
                    df['index_close_norm'] = df['index_close'] / index_price_at_base
                else:
                    df['index_close_norm'] = df['index_close'] / index_first_price

    return df, base_date, base_price


def plot_normalized_time_series(fund_code: str, df: pd.DataFrame, fund_name: str):
    """绘制归一化价格时间序列图"""
    fig, ax = plt.subplots(figsize=(14, 8))

    df_norm, base_date, base_price = normalize_aligned(df)

    if base_date is None:
        ax.text(0.5, 0.5, '无有效价格数据', transform=ax.transAxes, ha='center', va='center')
        plt.close()
        return None

    colors = {
        'raw_close_norm': ('blue', '除权价格'),
        'hfq_close_norm': ('orange', '后复权价格'),
        'index_close_norm': ('gray', '跟踪指数')
    }

    has_data = False
    for col, (color, label) in colors.items():
        if col in df_norm.columns and not df_norm[col].isna().all():
            valid_data = df_norm.dropna(subset=[col])
            if len(valid_data) > 0:
                ax.plot(valid_data['date'], valid_data[col],
                       color=color, linewidth=1.2, alpha=0.8, label=label)
                has_data = True

    if not has_data:
        ax.text(0.5, 0.5, '无有效价格数据', transform=ax.transAxes, ha='center', va='center')
        plt.close()
        return None

    ax.axhline(y=1, color='black', linestyle=':', alpha=0.5, label='基准线(=1.0)')

    ax.axvline(x=base_date, color='red', linestyle='--', alpha=0.5)
    ax.text(base_date, 1.02, f'基金上市日\n{base_date.strftime("%Y-%m-%d")}',
           ha='center', fontsize=9, color='red',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_title(f'{fund_code} {fund_name}\n归一化价格时间序列 (基金上市日价格={base_price:.4f})', fontsize=14)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('归一化价格 (基金上市日=1.0)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    info_text = f"数据点数: {len(df)}\n基金上市日: {base_date.strftime('%Y-%m-%d')}\n后复权首日价: {base_price:.4f}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    output_file = REPORT_DIR / f'{fund_code}_{fund_name}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    return output_file


def main():
    merged_files = sorted(MERGED_DIR.glob("*.csv"))

    fund_plots = []

    print(f"生成 {len(merged_files)} 个归一化价格时间序列图...")

    for file_path in tqdm(merged_files):
        fund_code = file_path.stem.replace('_merged', '')
        fund_name = get_fund_name(fund_code)

        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])

        if len(df) == 0:
            continue

        output_file = plot_normalized_time_series(fund_code, df, fund_name)
        if output_file:
            fund_plots.append((fund_code, fund_name, output_file))
            print(f"  {fund_code}: {output_file.name}")

    print(f"\n图表已保存至 {REPORT_DIR}")


if __name__ == "__main__":
    main()