import argparse
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any, Callable

import akshare as ak
import baostock as bs
import pandas as pd

try:
    from .index_codes import LG_INDEX_SYMBOL_MAP
except ImportError:
    from index_codes import LG_INDEX_SYMBOL_MAP


def _call_with_retry(func: Callable[..., pd.DataFrame], *args: Any, retry: int = 3, **kwargs: Any) -> pd.DataFrame:
    last_error = None
    for i in range(retry):
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if i < retry - 1:
                time.sleep(1 + i)
    raise RuntimeError(f"AkShare request failed after {retry} retries: {last_error}") from last_error


def _to_ak_symbol(lg_code: str) -> str:
    # 000016.SH -> sh000016
    code, market = lg_code.split(".")
    return f"{market.lower()}{code}"


def _to_filename(lg_code: str) -> str:
    # 000016.SH -> SH000016
    code, market = lg_code.split(".")
    return f"{market.upper()}{code}"


def _filter_by_date(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    df = df.copy()
    date_col = "date" if "date" in df.columns else "日期"
    dt = pd.to_datetime(df[date_col], errors="coerce")
    if start:
        df = df[dt >= pd.Timestamp(start)]
        dt = pd.to_datetime(df[date_col], errors="coerce")
    if end:
        df = df[dt <= pd.Timestamp(end)]
        dt = pd.to_datetime(df[date_col], errors="coerce")
    df[date_col] = dt.dt.strftime("%Y-%m-%d")
    return df


def _to_baostock_symbol(symbol: str) -> str:
    # sh000016 -> sh.000016
    return f"{symbol[:2]}.{symbol[2:]}"


def fetch_price(symbol: str, start: str | None, end: str | None) -> pd.DataFrame:
    bs_code = _to_baostock_symbol(symbol)
    fields = "date,code,open,high,low,close,volume,amount,pctChg"

    login_res = bs.login()
    if login_res.error_code != "0":
        raise RuntimeError(f"baostock login failed: {login_res.error_msg}")

    try:
        rs = bs.query_history_k_data_plus(
            code=bs_code,
            fields=fields,
            start_date=start or "1990-01-01",
            end_date=end or "2050-01-01",
            frequency="d",
        )
        if rs.error_code != "0":
            raise RuntimeError(f"baostock query failed for {bs_code}: {rs.error_msg}")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            raise RuntimeError(f"empty baostock price data for {bs_code}")

        df = pd.DataFrame(rows, columns=fields.split(","))
        for col in ["open", "high", "low", "close", "volume", "amount", "pctChg"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return _filter_by_date(df, start, end)
    finally:
        bs.logout()


def fetch_valuation(index_name: str, start: str | None, end: str | None) -> pd.DataFrame:
    pb_df = _call_with_retry(ak.stock_index_pb_lg, symbol=index_name)
    pe_df = _call_with_retry(ak.stock_index_pe_lg, symbol=index_name)

    pb_df = _filter_by_date(pb_df, start, end)
    pe_df = _filter_by_date(pe_df, start, end)

    return pe_df.merge(pb_df, on="日期", how="inner", suffixes=("_pe", "_pb"))


def rename_columns_to_english(df: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "等权静态市盈率": "pe_static_equal_weight",
        "静态市盈率": "pe_static",
        "静态市盈率中位数": "pe_static_median",
        "等权滚动市盈率": "pe_ttm_equal_weight",
        "滚动市盈率": "pe_ttm",
        "滚动市盈率中位数": "pe_ttm_median",
        "市净率": "pb",
        "等权市净率": "pb_equal_weight",
        "市净率中位数": "pb_median",
    }
    return df.rename(columns=column_map)


def parse_indexes(symbols_text: str | None) -> list[str]:
    if not symbols_text:
        return list(LG_INDEX_SYMBOL_MAP.keys())

    reverse_map = {v: k for k, v in LG_INDEX_SYMBOL_MAP.items()}
    indexes: list[str] = []
    for item in [x.strip() for x in symbols_text.split(",") if x.strip()]:
        if item in LG_INDEX_SYMBOL_MAP:
            indexes.append(item)
        elif item.upper() in reverse_map:
            indexes.append(reverse_map[item.upper()])
        else:
            raise ValueError(f"unsupported index: {item}")
    return indexes


def download_one(index_name: str, save_dir: str, start: str | None, end: str | None) -> None:
    lg_code = LG_INDEX_SYMBOL_MAP[index_name]
    symbol = _to_ak_symbol(lg_code)

    price_df = fetch_price(symbol, start, end)
    valuation_df = fetch_valuation(index_name, start, end)

    merged = price_df.merge(valuation_df, left_on="date", right_on="日期", how="inner")
    if merged.empty:
        raise RuntimeError(f"no merged rows for {index_name}")

    merged = merged.drop(columns=["日期", "指数_pe", "指数_pb"], errors="ignore")
    merged = rename_columns_to_english(merged)

    output_dir = Path(save_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{_to_filename(lg_code)}.csv"
    merged.to_csv(out_file, index=False)

    print(f"[ok] {index_name} -> {out_file.name}, rows={len(merged)}")


def _download_one_worker(index_name: str, save_dir: str, start: str | None, end: str | None, queue: mp.Queue) -> None:
    try:
        download_one(index_name, save_dir, start, end)
        queue.put((True, ""))
    except Exception as exc:  # noqa: BLE001
        queue.put((False, str(exc)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LG-supported index price + valuation data")
    parser.add_argument("--symbols", type=str, default=None, help="index names or LG codes, comma separated")
    parser.add_argument("--save_dir", type=str, default=str(Path(__file__).resolve().parent / "source"))
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--task_timeout", type=int, default=180, help="timeout in seconds for each index download")
    args = parser.parse_args()

    indexes = parse_indexes(args.symbols)
    print(f"start downloading {len(indexes)} LG-supported indexes, timeout={args.task_timeout}s each...")

    failed: list[str] = []
    for idx in indexes:
        queue: mp.Queue = mp.Queue()
        proc = mp.Process(target=_download_one_worker, args=(idx, args.save_dir, args.start, args.end, queue))
        proc.start()
        proc.join(args.task_timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            failed.append(idx)
            print(f"[error] {idx}: timeout after {args.task_timeout}s")
            continue

        if queue.empty():
            if proc.exitcode != 0:
                failed.append(idx)
                print(f"[error] {idx}: process exited with code {proc.exitcode}")
            continue

        ok, msg = queue.get()
        if not ok:
            failed.append(idx)
            print(f"[error] {idx}: {msg}")

    print(f"done. success={len(indexes) - len(failed)}, failed={len(failed)}")
    if failed:
        print("failed indexes:", ",".join(failed))


if __name__ == "__main__":
    main()
