import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Callable

import akshare as ak
import pandas as pd


def _call_with_retry(func: Callable[..., pd.DataFrame], *args: Any, retry: int = 3, **kwargs: Any) -> pd.DataFrame:
    """带重试机制的函数调用"""
    last_error = None
    for i in range(retry):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if i < retry - 1:
                time.sleep(1 + i)
    raise RuntimeError(f"AkShare request failed after {retry} retries: {last_error}") from last_error


def _normalize_ymd(date_text: str | None, default_value: str) -> str:
    if not date_text:
        return default_value
    return str(date_text).replace("-", "")


def _standardize_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    df = df.copy()
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def load_funds_list(csv_path: str) -> pd.DataFrame:
    """读取基金列表配置文件"""
    df = pd.read_csv(csv_path, dtype=str, comment='#')
    required_cols = ["fund_code", "fund_name", "fund_type", "track_target"]
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    # 过滤空行
    df = df.dropna(subset=["fund_code"])
    return df


def fetch_fund_hist(fund_code: str, start: str | None, end: str | None, adjust: str = "qfq") -> pd.DataFrame:
    """使用 akshare 获取 ETF 基金历史行情
    
    Args:
        adjust: 复权方式 - "qfq"(前复权), "hfq"(后复权), ""(不复权)
    """
    return _call_with_retry(
        ak.fund_etf_hist_em,
        symbol=fund_code,
        period="daily",
        start_date=_normalize_ymd(start, "19900101"),
        end_date=_normalize_ymd(end, "20501231"),
        adjust=adjust,
    )


def fetch_fund_nav(fund_code: str, start: str | None, end: str | None) -> pd.DataFrame:
    """获取 ETF 历史净值数据（估值/净值维度）"""
    return _call_with_retry(
        ak.fund_etf_fund_info_em,
        fund=fund_code,
        start_date=_normalize_ymd(start, "19900101"),
        end_date=_normalize_ymd(end, "20501231"),
    )


def load_estimation_snapshot_map(fund_codes: list[str]) -> dict[str, dict[str, Any]]:
    """加载实时估值快照，失败时降级为空"""
    if not hasattr(ak, "fund_value_estimation_em"):
        return {}

    try:
        estimation_df = _call_with_retry(ak.fund_value_estimation_em)
    except Exception:  # noqa: BLE001
        return {}

    if estimation_df.empty or "基金代码" not in estimation_df.columns:
        return {}

    estimation_df = estimation_df.copy()
    estimation_df["基金代码"] = estimation_df["基金代码"].astype(str).str.zfill(6)
    code_set = {str(code).zfill(6) for code in fund_codes}

    snapshot_map: dict[str, dict[str, Any]] = {}
    for _, row in estimation_df.iterrows():
        code = row.get("基金代码")
        if code not in code_set:
            continue
        snapshot_map[code] = {
            "estimation_value": row.get("估算值"),
            "estimation_growth_rate": row.get("估算增长率"),
            "estimation_unit_nav": row.get("单位净值"),
            "estimation_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    return snapshot_map


def rename_hist_columns_to_english(df: pd.DataFrame) -> pd.DataFrame:
    """历史行情列名标准化"""
    column_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
    }
    return df.rename(columns=column_map)


def rename_nav_columns_to_english(df: pd.DataFrame) -> pd.DataFrame:
    """历史净值列名标准化"""
    column_map = {
        "净值日期": "date",
        "单位净值": "unit_nav",
        "累计净值": "accum_nav",
        "日增长率": "nav_daily_growth",
        "申购状态": "subscribe_status",
        "赎回状态": "redeem_status",
    }
    return df.rename(columns=column_map)


def filter_by_date(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """按日期范围筛选数据"""
    df = df.copy()
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce")
        if start:
            df = df[dt >= pd.Timestamp(start)]
            dt = pd.to_datetime(df["date"], errors="coerce")
        if end:
            df = df[dt <= pd.Timestamp(end)]
    return df


def merge_price_and_nav(price_df: pd.DataFrame, nav_df: pd.DataFrame) -> pd.DataFrame:
    """按日期合并历史行情与净值"""
    if nav_df.empty:
        merged = price_df.copy()
    else:
        merged = price_df.merge(nav_df, on="date", how="outer")

    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged = merged.sort_values("date")
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
    return merged


def append_estimation_snapshot(df: pd.DataFrame, snapshot: dict[str, Any] | None) -> pd.DataFrame:
    """附加实时估值快照（同一次下载快照值会写入全部行）"""
    if not snapshot:
        return df

    df = df.copy()
    for col in ["estimation_value", "estimation_growth_rate", "estimation_unit_nav", "estimation_time"]:
        df[col] = snapshot.get(col)
    return df


def download_one(
    fund_code: str,
    fund_name: str,
    save_dir: str,
    start: str | None,
    end: str | None,
    estimation_snapshot: dict[str, Any] | None,
    adjust: str = "qfq",
) -> None:
    """下载单个基金数据并保存（行情 + 估值/净值）
    
    Args:
        adjust: 复权方式 - "qfq"(前复权), "hfq"(后复权), ""(不复权)
    """
    price_df = fetch_fund_hist(fund_code, start, end, adjust=adjust)
    if price_df.empty:
        raise RuntimeError(f"No price data for fund {fund_code} ({fund_name})")

    price_df = rename_hist_columns_to_english(price_df)
    price_df = _standardize_date(price_df, "date")
    price_df = filter_by_date(price_df, start, end)

    nav_df = pd.DataFrame()
    try:
        nav_df = fetch_fund_nav(fund_code, start, end)
        nav_df = rename_nav_columns_to_english(nav_df)
        nav_df = _standardize_date(nav_df, "date")
        nav_df = filter_by_date(nav_df, start, end)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] {fund_code}: nav data unavailable, reason={exc}")

    merged = merge_price_and_nav(price_df, nav_df)
    merged = append_estimation_snapshot(merged, estimation_snapshot)

    if merged.empty:
        raise RuntimeError(f"No merged data for fund {fund_code} ({fund_name})")

    ordered_cols = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
        "unit_nav",
        "accum_nav",
        "nav_daily_growth",
        "subscribe_status",
        "redeem_status",
        "estimation_value",
        "estimation_growth_rate",
        "estimation_unit_nav",
        "estimation_time",
    ]
    final_cols = [c for c in ordered_cols if c in merged.columns]
    merged = merged[final_cols]

    output_dir = Path(save_dir).expanduser().resolve()
    # 根据复权方式创建子目录: qfq(前复权), hfq(后复权), raw(不复权)
    adjust_subdir = adjust if adjust else "raw"
    output_dir = output_dir / adjust_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{fund_code}.csv"
    merged.to_csv(out_file, index=False)

    adjust_label = {"qfq": "前复权", "hfq": "后复权", "": "不复权"}.get(adjust, adjust)
    print(f"[ok] {fund_code} {fund_name} ({adjust_label}) -> {out_file}, rows={len(merged)}")


def _download_one_worker(
    fund_code: str,
    fund_name: str,
    save_dir: str,
    start: str | None,
    end: str | None,
    estimation_snapshot: dict[str, Any] | None,
    adjust: str,
    queue: mp.Queue,
) -> None:
    """多进程 worker"""
    try:
        download_one(fund_code, fund_name, save_dir, start, end, estimation_snapshot, adjust=adjust)
        queue.put((True, ""))
    except Exception as exc:  # noqa: BLE001
        queue.put((False, str(exc)))


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
    parser = argparse.ArgumentParser(description="Download fund historical data with valuation using AkShare")
    parser.add_argument("--symbols", type=str, default=None, help="Fund codes, comma separated (e.g., 510300,563020)")
    parser.add_argument("--save_dir", type=str, default=str(Path(__file__).resolve().parent / "source"))
    parser.add_argument("--funds_list", type=str, default=str(Path(__file__).resolve().parent / "funds_list.csv"))
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYYMMDD or YYYY-MM-DD)")
    parser.add_argument("--task_timeout", type=int, default=180, help="Timeout in seconds for each fund download")
    parser.add_argument(
        "--adjust",
        type=str,
        default="hfq",
        help="Adjustment type: qfq(前复权), hfq(后复权), raw(不复权), all(全部下载)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Interval in seconds between downloads (default: 3.0)",
    )
    args = parser.parse_args()

    funds_df = load_funds_list(args.funds_list)
    funds_df = parse_funds(args.symbols, funds_df)

    fund_codes = funds_df["fund_code"].astype(str).str.zfill(6).tolist()
    estimation_map = load_estimation_snapshot_map(fund_codes)

    # 确定要下载的复权方式
    if args.adjust == "all":
        adjust_modes = [("qfq", "前复权"), ("hfq", "后复权"), ("", "不复权")]
    elif args.adjust == "raw":
        adjust_modes = [("", "不复权")]
    else:
        label_map = {"qfq": "前复权", "hfq": "后复权"}
        adjust_modes = [(args.adjust, label_map.get(args.adjust, args.adjust))]

    total_funds = len(funds_df)
    total_modes = len(adjust_modes)
    print(f"Start downloading {total_funds} funds x {total_modes} adjust modes, timeout={args.task_timeout}s each...")

    failed: list[str] = []
    for adjust, adjust_label in adjust_modes:
        print(f"\n=== 下载 {adjust_label} 数据 ===")
        for _, row in funds_df.iterrows():
            fund_code = str(row["fund_code"]).zfill(6)
            fund_name = row["fund_name"]
            snapshot = estimation_map.get(fund_code)

            queue: mp.Queue = mp.Queue()
            proc = mp.Process(
                target=_download_one_worker,
                args=(fund_code, fund_name, args.save_dir, args.start, args.end, snapshot, adjust, queue),
            )
            proc.start()
            proc.join(args.task_timeout)

            if proc.is_alive():
                proc.terminate()
                proc.join()
                failed.append(f"{fund_code}({adjust_label})")
                print(f"[error] {fund_code}: timeout after {args.task_timeout}s")
                continue

            if queue.empty():
                if proc.exitcode != 0:
                    failed.append(f"{fund_code}({adjust_label})")
                    print(f"[error] {fund_code}: process exited with code {proc.exitcode}")
                continue

            ok, msg = queue.get()
            if not ok:
                failed.append(f"{fund_code}({adjust_label})")
                print(f"[error] {fund_code}: {msg}")

            # 间隔一段时间再下载下一个
            if args.interval > 0:
                time.sleep(args.interval)

    total_tasks = total_funds * total_modes
    print(f"\nDone. success={total_tasks - len(failed)}, failed={len(failed)}")
    if failed:
        print("Failed funds:", ", ".join(failed))


if __name__ == "__main__":
    main()
