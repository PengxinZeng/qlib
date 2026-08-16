# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
批量下载基金K线数据（后复权和除权）

使用腾讯财经接口获取ETF/LOF完整历史数据
"""

import sys
import os
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from loguru import logger

# 添加路径（path_config 集中管理，跨平台）
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))  # scripts/
import path_config  # noqa: E402
from data_collector.tencent_etf.collector import TencentETFCollector

# 配置
BASE_DIR = path_config.DATA_BASE / 'stock_data'
FUNDS_LIST = str(path_config.FUNDS_LIST)
HFQ_DIR = BASE_DIR / 'fund_kline_hfq'
RAW_DIR = BASE_DIR / 'fund_kline_raw'

API_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
MAX_RECORDS = 800


def normalize_symbol(code: str) -> tuple:
    """将基金代码转换为腾讯格式 (返回前缀和代码)"""
    code = str(code).strip().zfill(6)
    if code.startswith("51") or code.startswith("58"):
        return f"sh{code}", "sh", code
    if code.startswith("15") or code.startswith("56") or code.startswith("13"):
        return f"sz{code}", "sz", code
    return None, None, None


def get_kline(symbol: str, fq_type: str = "", end_date: str = None, limit: int = MAX_RECORDS) -> list:
    """获取一段K线数据"""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    params = f"{symbol},day,,{end_date},{limit},{fq_type}"
    url = f"{API_URL}?param={params}"

    try:
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        data = resp.json()

        if data.get("code") == 0 and isinstance(data.get("data"), dict):
            key = f"{fq_type}day" if fq_type else "day"
            klines = data["data"].get(symbol, {}).get(key, [])
            return klines
    except Exception as e:
        logger.warning(f"get {symbol} segment error: {e}")

    return []


def get_full_kline(symbol: str, fq_type: str = "") -> pd.DataFrame:
    """分段获取完整历史K线"""
    all_data = []
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")
    max_iterations = 30

    for _ in range(max_iterations):
        klines = get_kline(symbol, fq_type, end_date)

        if not klines:
            break

        all_data.extend(klines)

        if len(klines) < MAX_RECORDS:
            break

        earliest_date = klines[0][0]
        end_date = (pd.Timestamp(earliest_date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        time.sleep(0.3)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["date", "open", "close", "high", "low", "volume"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    for col in ["open", "close", "high", "low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype("int64")

    return df


def download_all_funds():
    """下载所有基金数据"""
    # 读取基金列表
    df = pd.read_csv(FUNDS_LIST, comment='#', dtype=str)
    df = df.dropna(subset=['fund_code', 'track_target_file'])
    df = df[df['track_target_file'] != 'N/A']

    # 确保目录存在
    HFQ_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="下载基金数据"):
        code = row['fund_code']
        name = row['fund_name']
        index_file = row['track_target_file']

        tencent_symbol, market, raw_code = normalize_symbol(code)
        if not tencent_symbol:
            logger.warning(f"不支持的基金代码: {code}")
            continue

        logger.info(f"正在下载 {code} ({name})...")

        # 下载后复权数据
        hfq_df = get_full_kline(tencent_symbol, "hfq")
        if not hfq_df.empty:
            hfq_path = HFQ_DIR / f"{code}.csv"
            hfq_df.to_csv(hfq_path, index=False)
            hfq_start = hfq_df['date'].iloc[0]
            hfq_end = hfq_df['date'].iloc[-1]
            hfq_rows = len(hfq_df)
        else:
            hfq_start, hfq_end, hfq_rows = None, None, 0

        time.sleep(0.3)

        # 下载除权数据
        raw_df = get_full_kline(tencent_symbol, "")
        if not raw_df.empty:
            raw_path = RAW_DIR / f"{code}.csv"
            raw_df.to_csv(raw_path, index=False)
            raw_start = raw_df['date'].iloc[0]
            raw_end = raw_df['date'].iloc[-1]
            raw_rows = len(raw_df)
        else:
            raw_start, raw_end, raw_rows = None, None, 0

        results.append({
            'fund_code': code,
            'fund_name': name,
            'market': market,
            'index_file': index_file,
            'hfq_start': hfq_start,
            'hfq_end': hfq_end,
            'hfq_rows': hfq_rows,
            'raw_start': raw_start,
            'raw_end': raw_end,
            'raw_rows': raw_rows,
        })

        time.sleep(0.5)

    # 保存下载报告
    results_df = pd.DataFrame(results)
    results_path = BASE_DIR / 'report' / 'download_status.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"下载完成，结果保存至 {results_path}")

    return results_df


if __name__ == "__main__":
    results = download_all_funds()
    print(results.to_string())
