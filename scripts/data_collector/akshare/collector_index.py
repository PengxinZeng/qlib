import argparse
import time
from pathlib import Path
from typing import Any, Callable

import akshare as ak
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


def download_one(index_name: str, save_dir: str, start: str | None, end: str | None) -> tuple[str, int]:
    """下载单只指数，返回 (index_name, rows)"""
    import time as _time
    lg_code = LG_INDEX_SYMBOL_MAP[index_name]
    t0 = _time.time()

    try:
        valuation_df = fetch_valuation(index_name, start, end)
    except Exception as e:
        raise RuntimeError(f"valuation failed: {e}") from e

    if valuation_df.empty:
        raise RuntimeError("valuation empty")

    merged = valuation_df.rename(columns={"日期": "date"})
    merged = merged.drop(columns=["指数_pe", "指数_pb"], errors="ignore")
    merged = rename_columns_to_english(merged)

    output_dir = Path(save_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{_to_filename(lg_code)}.csv"

    if out_file.exists():
        existing = pd.read_csv(out_file)
        merged = pd.concat([existing, merged], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    merged.to_csv(out_file, index=False)
    elapsed = _time.time() - t0
    print(f"[ok] {index_name} -> {out_file.name}  rows={len(merged)}  {elapsed:.1f}s", flush=True)
    return index_name, len(merged)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LG-supported index valuation data (AkShare)")
    parser.add_argument("--symbols", type=str, default=None, help="index names or LG codes, comma separated")
    parser.add_argument("--save_dir", type=str, default=str(Path(__file__).resolve().parent / "source"))
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--delay", type=float, default=3.0, help="每只指数下载后等待秒数，避免触发频率限制（默认 3s）")
    args = parser.parse_args()

    indexes = parse_indexes(args.symbols)
    total = len(indexes)
    print(f"start downloading {total} indexes (serial, delay={args.delay}s)...", flush=True)

    done = 0
    failed: list[str] = []
    for idx in indexes:
        done += 1
        try:
            _, rows = download_one(idx, args.save_dir, args.start, args.end)
            print(f"  进度 [{done}/{total}]  {idx} ✓  rows={rows}", flush=True)
        except Exception as exc:
            failed.append(idx)
            print(f"  进度 [{done}/{total}]  {idx} ✗  {exc}", flush=True)
        if done < total and args.delay > 0:
            time.sleep(args.delay)

    status = "success" if not failed else f"success={total-len(failed)}, failed={len(failed)}"
    print(f"done. {status}", flush=True)
    if failed:
        print("failed indexes:", ",".join(failed))
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
