"""
新浪/腾讯财经基金历史K线数据收集器

通过新浪财经API获取ETF/基金的不复权K线数据
通过天天基金获取分红数据，自行计算复权价格

支持前复权、后复权、不复权
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

# 跨平台路径集中配置（Mac / Windows 兼容）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import path_config  # noqa: E402


def _call_with_retry(func: Callable[..., Any], *args: Any, retry: int = 3, **kwargs: Any) -> Any:
    """带重试机制的函数调用"""
    last_error = None
    for i in range(retry):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if i < retry - 1:
                time.sleep(1 + i)
    raise RuntimeError(f"Request failed after {retry} retries: {last_error}") from last_error


def load_funds_list(csv_path: str) -> pd.DataFrame:
    """读取基金列表配置文件"""
    df = pd.read_csv(csv_path, dtype=str, comment='#')
    required_cols = ["fund_code", "fund_name", "fund_type", "track_target"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df.dropna(subset=["fund_code"])
    return df


def get_market_prefix(code: str) -> str:
    """
    根据基金代码判断市场前缀
    - 5开头、1开头 -> sh (上海)
    - 0开头 -> sz (深圳)
    """
    code = str(code).zfill(6)
    if code.startswith('5') or code.startswith('1'):
        return 'sh'
    elif code.startswith('0'):
        return 'sz'
    return 'sh'


def fetch_dividend_eastmoney(code: str) -> pd.DataFrame:
    """
    从天天基金获取ETF分红数据
    
    Args:
        code: 基金代码（6位）
    
    Returns:
        DataFrame包含: ex_date(除息日), dividend(每份分红金额)
    """
    code = str(code).zfill(6)
    url = f'http://fundf10.eastmoney.com/fhsp_{code}.html'
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    }
    
    def _fetch():
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        resp.encoding = 'utf-8'
        
        # 提取分红表格数据
        # 格式: 权益登记日, 除息日, 每份分红
        pattern = r'<td[^>]*>(\d{4}-\d{2}-\d{2})</td>\s*<td[^>]*>(\d{4}-\d{2}-\d{2})</td>\s*<td[^>]*>每份派现金(\d+\.\d+)元'
        matches = re.findall(pattern, resp.text)
        
        if not matches:
            return pd.DataFrame()
        
        records = []
        for reg_date, ex_date, dividend in matches:
            records.append({
                "ex_date": ex_date,
                "dividend": float(dividend),
            })
        
        return pd.DataFrame(records)
    
    try:
        return _call_with_retry(_fetch)
    except Exception as e:
        print(f"  [warn] Failed to fetch dividend data: {e}")
        return pd.DataFrame()


def calculate_adjust_factor(df: pd.DataFrame, dividend_df: pd.DataFrame, adjust: str = "hfq") -> pd.DataFrame:
    """
    根据分红数据计算复权因子并应用到价格
    
    后复权计算方法（以最早价格为基准）：
    - 早期价格保持不变
    - 除息日及之后的价格向上调整（反映分红再投资的收益）
    - 因子 = 除息前收盘价 / (除息前收盘价 - 分红)
    
    前复权计算方法（以最新价格为基准）：
    - 最新价格保持不变
    - 早期价格向下调整
    - 因子 = (除息前收盘价 - 分红) / 除息前收盘价
    
    Args:
        df: 原始K线数据，需包含 date, open, high, low, close
        dividend_df: 分红数据，包含 ex_date, dividend
        adjust: 复权方式 - "qfq"(前复权), "hfq"(后复权)
    
    Returns:
        复权后的DataFrame，包含 adjust_factor 列
    """
    if adjust not in ("qfq", "hfq") or dividend_df.empty:
        df = df.copy()
        df["adjust_factor"] = 1.0
        return df
    
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    
    # 处理分红数据
    dividend_df = dividend_df.copy()
    dividend_df["ex_date"] = pd.to_datetime(dividend_df["ex_date"])
    dividend_df = dividend_df.sort_values("ex_date")
    
    # 初始化复权因子
    df["adjust_factor"] = 1.0
    
    # 计算每次分红的复权因子
    for _, div_row in dividend_df.iterrows():
        ex_date = div_row["ex_date"]
        dividend = div_row["dividend"]
        
        # 找到除息日前一个交易日的收盘价
        pre_ex_df = df[df["date"] < ex_date]
        if pre_ex_df.empty:
            continue
        
        pre_close = pre_ex_df.iloc[-1]["close"]
        if pre_close <= 0 or dividend <= 0:
            continue
        
        if adjust == "hfq":
            # 后复权：以最早价格为基准，除息日及之后的数据向上调整
            # 因子 = pre_close / (pre_close - dividend) > 1
            factor = pre_close / (pre_close - dividend)
            if factor <= 1:
                continue
            df.loc[df["date"] >= ex_date, "adjust_factor"] *= factor
        else:  # qfq
            # 前复权：以最新价格为基准，除息日之前的数据向下调整
            # 因子 = (pre_close - dividend) / pre_close < 1
            factor = (pre_close - dividend) / pre_close
            if factor <= 0 or factor >= 1:
                continue
            df.loc[df["date"] < ex_date, "adjust_factor"] *= factor
    
    # 应用复权因子到价格
    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col] * df["adjust_factor"]
            df[col] = df[col].round(4)
    
    # 格式化日期
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    
    return df


def fetch_etf_kline_sina(code: str, datalen: int = 9999) -> pd.DataFrame:
    """
    通过新浪财经获取ETF历史K线数据（不复权）
    
    Args:
        code: 基金代码（6位）
        datalen: 获取数据条数
    
    Returns:
        DataFrame包含: date, open, high, low, close, volume
    """
    code = str(code).zfill(6)
    market = get_market_prefix(code)
    symbol = f"{market}{code}"
    
    url = (
        f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
        f"CN_MarketData.getKLineData?symbol={symbol}&scale=240&ma=no&datalen={datalen}"
    )
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Referer": "https://finance.sina.com.cn/",
    }
    
    def _fetch():
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        
        text = resp.text.strip()
        if not text or text == "null":
            return pd.DataFrame()
        
        data = json.loads(text)
        if not data:
            return pd.DataFrame()
        
        return pd.DataFrame(data)
    
    df = _call_with_retry(_fetch)
    
    if df.empty:
        return df
    
    # 标准化列名
    column_map = {
        "day": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    df = df.rename(columns=column_map)
    
    # 确保数值类型
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 标准化日期格式
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)
    
    output_cols = ["date", "open", "high", "low", "close", "volume"]
    final_cols = [c for c in output_cols if c in df.columns]
    
    return df[final_cols]


def download_one(
    fund_code: str,
    fund_name: str,
    save_dir: str,
    datalen: int = 9999,
    adjust: str = "hfq",
) -> None:
    """
    下载单个基金数据并保存
    
    Args:
        fund_code: 基金代码
        fund_name: 基金名称
        save_dir: 保存目录
        datalen: 获取数据条数
        adjust: 复权方式 - "qfq"(前复权), "hfq"(后复权), "raw"(不复权)
    """
    # 获取不复权K线数据
    df = fetch_etf_kline_sina(fund_code, datalen=datalen)
    
    if df.empty:
        raise RuntimeError(f"No data for fund {fund_code} ({fund_name})")
    
    # 获取分红数据并计算复权
    if adjust in ("qfq", "hfq"):
        dividend_df = fetch_dividend_eastmoney(fund_code)
        if not dividend_df.empty:
            print(f"  [info] {fund_code}: found {len(dividend_df)} dividend records")
            df = calculate_adjust_factor(df, dividend_df, adjust=adjust)
        else:
            print(f"  [info] {fund_code}: no dividend records found")
            df["adjust_factor"] = 1.0
    else:
        df["adjust_factor"] = 1.0
    
    # 创建输出目录（按复权方式分子目录）
    output_dir = Path(save_dir).expanduser().resolve()
    adjust_subdir = adjust if adjust else "raw"
    output_dir = output_dir / adjust_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 选择输出列
    output_cols = ["date", "open", "high", "low", "close", "volume", "adjust_factor"]
    final_cols = [c for c in output_cols if c in df.columns]
    df = df[final_cols]
    
    out_file = output_dir / f"{fund_code}.csv"
    df.to_csv(out_file, index=False)
    
    adjust_label = {"qfq": "前复权", "hfq": "后复权", "raw": "不复权"}.get(adjust, adjust)
    print(f"[ok] {fund_code} {fund_name} ({adjust_label}) -> {out_file}, rows={len(df)}")


def parse_funds(symbols_text: str | None, funds_df: pd.DataFrame) -> pd.DataFrame:
    """解析基金代码参数，返回筛选后的基金列表"""
    if not symbols_text:
        return funds_df
    
    codes = [x.strip() for x in symbols_text.split(",") if x.strip()]
    filtered = funds_df[funds_df["fund_code"].isin(codes)]
    if filtered.empty:
        raise ValueError(f"No matching funds for: {symbols_text}")
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Download fund historical K-line data with dividend adjustment")
    parser.add_argument("--symbols", type=str, default=None, help="Fund codes, comma separated (e.g., 510300,159915)")
    parser.add_argument("--save_dir", type=str, default=str(Path(__file__).resolve().parent / "source"))
    parser.add_argument("--funds_list", type=str, default=str(path_config.DATA_BASE / "qlib_data_test11" / "funds_list.csv"))
    parser.add_argument("--datalen", type=int, default=9999, help="Number of data points to fetch (default: 9999)")
    parser.add_argument("--interval", type=float, default=1.0, help="Interval in seconds between downloads (default: 1.0)")
    parser.add_argument(
        "--adjust",
        type=str,
        default="hfq",
        choices=["qfq", "hfq", "raw"],
        help="Adjustment type: qfq(前复权), hfq(后复权), raw(不复权). Default: hfq",
    )
    args = parser.parse_args()
    
    funds_df = load_funds_list(args.funds_list)
    funds_df = parse_funds(args.symbols, funds_df)
    
    total_funds = len(funds_df)
    adjust_label = {"qfq": "前复权", "hfq": "后复权", "raw": "不复权"}.get(args.adjust, args.adjust)
    print(f"Start downloading {total_funds} funds ({adjust_label})...")
    print(f"Data source: Sina Finance (K-line) + EastMoney (dividend)")
    
    failed: list[str] = []
    for idx, row in funds_df.iterrows():
        fund_code = str(row["fund_code"]).zfill(6)
        fund_name = row["fund_name"]
        
        try:
            download_one(fund_code, fund_name, args.save_dir, datalen=args.datalen, adjust=args.adjust)
        except Exception as exc:
            failed.append(fund_code)
            print(f"[error] {fund_code} {fund_name}: {exc}")
        
        if args.interval > 0:
            time.sleep(args.interval)
    
    print(f"\nDone. success={total_funds - len(failed)}, failed={len(failed)}")
    if failed:
        print("Failed funds:", ", ".join(failed))


if __name__ == "__main__":
    main()
