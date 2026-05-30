"""
合并 + 清洗基金数据（一步完成）

处理逻辑：
1. 合并基金k线(hfq/raw)、指数数据、国债收益率
2. 统一列名: hfq_*/raw_* → open/close/high/low/volume (hfq优先，raw填充缺失)
3. 不筛选行（保留全部行，含空值行）
4. 异常值清洗：检测单日价格尖刺，将异常行的基金价格列设为 NaN
   - 判定规则：close 单日涨跌幅 > SPIKE_THRESHOLD(40%)，且次日反向回复 > SPIKE_THRESHOLD
   - 仅清除 open/high/low/close，不影响 volume、指数、估值、国债列
5. 运行前清理输出目录

环境要求: conda activate rdagent
"""

import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ─── 路径配置 ────────────────────────────────────────────────────────────────
QLIB_BASE = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415")
SOURCE_DIR = QLIB_BASE / "source"

HFQ_DIR    = SOURCE_DIR / "etf_index" / "fund_kline_hfq"
RAW_DIR    = SOURCE_DIR / "etf_index" / "fund_kline_raw"
INDEX_DIR  = SOURCE_DIR / "etf_index" / "index_data"
MERGED_DIR = SOURCE_DIR / "etf_index" / "merged"
FUNDS_LIST = SOURCE_DIR / "funds_list.csv"
BOND_FILE  = SOURCE_DIR / "cn_bond_rate" / "cn_bond_yield.csv"

FUND_KLINE_COLS = ["open", "high", "low", "close", "volume"]
FUND_PRICE_COLS = ["open", "high", "low", "close"]   # 异常检测仅针对价格列（不含volume）
SPIKE_THRESHOLD = 0.40                                # 单日涨跌幅阈值：40%
INDEX_KLINE_RENAME = {c: f"index_{c}" for c in FUND_KLINE_COLS}
VALUATION_COLS = [
    "amount", "pctChg",
    "pe_static_equal_weight", "pe_static", "pe_static_median",
    "pe_ttm_equal_weight", "pe_ttm", "pe_ttm_median",
    "pb", "pb_equal_weight", "pb_median",
]


def load_funds_list() -> pd.DataFrame:
    df = pd.read_csv(FUNDS_LIST, comment="#", dtype=str)
    df = df.dropna(subset=["fund_code", "track_target_file"])
    df = df[df["track_target_file"].str.strip().ne("") & df["track_target_file"].str.strip().ne("N/A")]
    return df


def load_bond_data() -> pd.DataFrame:
    df = pd.read_csv(BOND_FILE)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_kline(path: Path, prefix: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    rename = {c: f"{prefix}_{c}" for c in FUND_KLINE_COLS if c in df.columns}
    return df.rename(columns=rename)


def load_index(index_file: str) -> pd.DataFrame:
    path = INDEX_DIR / index_file
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns=INDEX_KLINE_RENAME)
    if "code" in df.columns:
        df = df.drop(columns=["code"])
    return df


def merge_and_clean(fund_code: str, index_file: str, bond_df: pd.DataFrame) -> pd.DataFrame:
    hfq = load_kline(HFQ_DIR / f"{fund_code}.csv", "hfq")
    raw = load_kline(RAW_DIR / f"{fund_code}.csv", "raw")
    idx = load_index(index_file)

    # 外连接合并所有数据源
    merged = pd.DataFrame()
    for df in [hfq, raw, idx]:
        if df.empty:
            continue
        merged = df if merged.empty else merged.merge(df, on="date", how="outer")

    if merged.empty:
        return pd.DataFrame()

    merged = merged.sort_values("date").reset_index(drop=True)

    # 统一基金k线列名（hfq优先，raw填充缺失）
    result = pd.DataFrame({"date": merged["date"]})
    for col in FUND_KLINE_COLS:
        hfq_col = f"hfq_{col}"
        raw_col = f"raw_{col}"
        if hfq_col in merged.columns:
            series = merged[hfq_col]
            if raw_col in merged.columns:
                series = series.fillna(merged[raw_col])
            result[col] = series
        elif raw_col in merged.columns:
            result[col] = merged[raw_col]
        else:
            result[col] = pd.NA

    # 添加指数k线列
    for col in [f"index_{c}" for c in FUND_KLINE_COLS]:
        result[col] = merged[col] if col in merged.columns else pd.NA

    # 添加估值列
    for col in VALUATION_COLS:
        result[col] = merged[col] if col in merged.columns else pd.NA

    # 左连接国债收益率（精确匹配日期），缺失日用最近交易日数据前向填充
    bond_cols = [c for c in bond_df.columns if c != "date"]
    result = result.merge(bond_df[["date"] + bond_cols], on="date", how="left")
    result[bond_cols] = result[bond_cols].ffill()

    # 数据来源标记：0=hfq, 1=raw, -1=无基金数据
    if "hfq_close" in merged.columns:
        result["data_source"] = merged["hfq_close"].notna().map({True: 0, False: 1})
    else:
        result["data_source"] = -1

    return result


def clean_price_spikes(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    检测并清除基金价格列中的单日尖刺异常。

    判定规则：close 列的当日涨跌幅绝对值 > SPIKE_THRESHOLD(40%)，
              且次日涨跌幅绝对值 > SPIKE_THRESHOLD 且方向相反（回复）。

    处理：将命中行的 open/high/low/close 设为 NaN，其余列（volume、指数、债券等）保持不变。

    Returns
    -------
    (cleaned_df, anomalous_dates)
    """
    if "close" not in df.columns or df["close"].isna().all():
        return df, []

    df = df.copy()
    close = df["close"]
    pct = close.pct_change(fill_method=None)          # 当日涨跌幅
    pct_next = pct.shift(-1)          # 次日涨跌幅

    # 尖刺：当日大幅波动 + 次日方向相反回复
    spike_mask = (
        (pct.abs() > SPIKE_THRESHOLD) &
        (pct_next.abs() > SPIKE_THRESHOLD) &
        (pct * pct_next < 0)          # 方向相反
    )

    anomalous_dates = df.loc[spike_mask, "date"].tolist()
    if anomalous_dates:
        df.loc[spike_mask, FUND_PRICE_COLS] = float("nan")

    return df, anomalous_dates


def main():
    print("=" * 60)
    print("基金数据合并 + 清洗（含国债收益率）")
    print("=" * 60)

    # 清理输出目录
    print(f"\n[1/5] 清理输出目录: {MERGED_DIR}")
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    MERGED_DIR.mkdir(parents=True)

    # 加载公共数据
    print("[2/5] 加载基金列表和国债数据...")
    funds = load_funds_list()
    bond_df = load_bond_data()
    print(f"    基金数量: {len(funds)}")
    print(f"    国债数据: {len(bond_df)} 行，{bond_df['date'].min().date()} ~ {bond_df['date'].max().date()}")

    # 处理每只基金
    print(f"\n[3/5] 合并处理 {len(funds)} 只基金...")
    success, skip = 0, 0
    all_spikes = {}   # fund_code -> [anomalous_dates]
    for _, row in tqdm(funds.iterrows(), total=len(funds), desc="    处理中"):
        fund_code  = row["fund_code"]
        index_file = row["track_target_file"]

        result = merge_and_clean(fund_code, index_file, bond_df)
        if result.empty:
            skip += 1
            continue

        result, spikes = clean_price_spikes(result)
        if spikes:
            all_spikes[fund_code] = spikes

        out_path = MERGED_DIR / f"{fund_code}_clean.csv"
        result.to_csv(out_path, index=False)
        success += 1

    # 尖刺清洗报告
    print(f"\n[4/5] 异常价格清洗报告:")
    if all_spikes:
        for code, dates in all_spikes.items():
            date_strs = ", ".join(d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in dates)
            print(f"    {code}: {len(dates)} 处尖刺 → {date_strs}（open/high/low/close 已设为 NaN）")
    else:
        print("    未发现异常尖刺")

    # 汇总
    print(f"\n[5/5] 完成")
    print(f"    输出目录: {MERGED_DIR}")
    print(f"    成功: {success} 只  跳过(无数据): {skip} 只")
    print("=" * 60)


if __name__ == "__main__":
    main()
