#!/usr/bin/env python3
"""export_holdings.py — 导出 SustainedBest 当前持仓明细 CSV（仅解析，不重算 Best Model）

数据来源：pred_df.csv（含 best_model 列，由 EMEnsemble 模型输出）/ positions_daily.csv / pipeline.yaml
输出列（按 index 排序）：
  index, code, name, ensemble_score, best_model, {model}_score..., best_model_score,
  buy_start, buy_end, annualized_return, annualized_1y/3y/5y/10y, inception_date, pos_value

buy_start/buy_end 买入时机窗口：最近一次"空仓->买入"转变日 t 起，到最后一个 score>0 日 end。
annualized_* ：成立/近 N 年年化收益率，不满年限置空（对齐 recalculate_annualized.py 规则）。

用法:
  python scripts/export_holdings.py                 # 默认最新 run
  python scripts/export_holdings.py --run_dir <p>   # 指定 run
  python scripts/export_holdings.py --out <p>       # 指定输出
"""
import argparse
import sys
import yaml
from pathlib import Path

import pandas as pd

# 跨平台路径集中配置（Mac / Windows 兼容）
sys.path.insert(0, str(Path(__file__).resolve().parent))
import path_config  # noqa: E402

QLIB_ROOT = path_config.QLIB_ROOT
MLRUNS = QLIB_ROOT / "mlruns"
PIPELINE_YAML = QLIB_ROOT / "scripts" / "data_pipline" / "pipeline.yaml"
EXPERIMENT_NAME = "em_ensemble_sustainedbest_all_weather"
ALL_WEATHER_BASE = path_config.ALL_WEATHER_BASE
CLEANED_DIR = ALL_WEATHER_BASE / "cleaned"
YEARS = (1, 3, 5, 10)


def load_latest_run_dir(run_dir=None):
    if run_dir:
        return Path(run_dir)
    best = None
    for exp in MLRUNS.iterdir():
        meta = exp / "meta.yaml"
        if not (exp.is_dir() and meta.exists()):
            continue
        nl = next((ln for ln in meta.read_text().splitlines() if ln.startswith("name:")), None)
        if not nl or nl.split(":", 1)[1].strip().strip("'\"") != EXPERIMENT_NAME:
            continue
        for run in exp.iterdir():
            if run.is_dir() and (run / "artifacts/analysis_csvs/positions_daily.csv").exists():
                if best is None or run.stat().st_mtime > best.stat().st_mtime:
                    best = run
    return best


def load_pred_df(run_dir):
    df = pd.read_csv(run_dir / "artifacts/analysis_csvs/pred_df.csv")
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors="coerce")
    df["instrument"] = df["instrument"].astype(str).str.zfill(6).str.replace("_CLEAN", "", regex=False)
    return df


def load_cleaned_close(code):
    """读 cleaned/{code}_clean.csv 的 close（按日期升序，去空值）"""
    path = CLEANED_DIR / f"{code}_clean.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date")
    if df.empty:
        return None
    return df["date"], df["close"].astype(float)


def calc_annualized(first_date, first_price, last_date, last_price):
    """年化收益率 = (最新价/首期价)^(365/days) - 1（对齐 recalculate_annualized.py）"""
    try:
        days = (pd.Timestamp(last_date) - pd.Timestamp(first_date)).days
        first_price, last_price = float(first_price), float(last_price)
        if days <= 0 or first_price <= 0 or last_price <= 0:
            return ""
        return (last_price / first_price) ** (365.0 / days) - 1
    except (ValueError, TypeError):
        return ""


def load_holdings(run_dir):
    pos = pd.read_csv(run_dir / "artifacts/analysis_csvs/positions_daily.csv", index_col=0, parse_dates=True)
    pos = pos[pos.index != "total_holding_ratio"].dropna(how="all")
    last_date = str(pos.index[-1])[:10]
    last = pos.iloc[-1]
    cols = [c for c in last.index if c not in ("account_value", "cash")]
    hold = {c.replace("_CLEAN", ""): float(last[c]) for c in cols if pd.notna(last[c]) and last[c] > 0}
    return hold, last_date


def load_symbol_meta(pipeline_yaml=PIPELINE_YAML):
    with open(pipeline_yaml) as f:
        cfg = yaml.safe_load(f)
    meta = {}
    for step in cfg.get("pipelines", []):
        for it in step.get("symbols") or []:
            if isinstance(it, dict):
                meta[it["code"]] = (it.get("name", ""), it.get("index", ""))
            else:
                meta[str(it)] = (str(it), "")
    return meta


def find_buy_window(score_series, latest_date):
    """最近一轮买入窗口: 最近一次 <=0->>0 转变日 t 起, 到最后一个 score>0 日 end"""
    if score_series.empty:
        return "", ""
    s = score_series.fillna(0.0)
    s = s[s.index <= latest_date]
    if s.empty:
        return "", ""
    vals = s.values.astype(float)
    dates = s.index
    n = len(vals)
    t = -1
    for i in range(n - 1, -1, -1):
        if vals[i] > 0:
            j = i
            while j > 0 and vals[j - 1] > 0:
                j -= 1
            t = j
            break
    if t < 0:
        return "", ""
    end_idx = t
    while end_idx + 1 < n and vals[end_idx + 1] > 0:
        end_idx += 1
    return str(dates[t].date()), str(dates[end_idx].date())


def build_report(run_dir, pipeline_yaml=PIPELINE_YAML):
    pred = load_pred_df(run_dir)
    hold, last_date = load_holdings(run_dir)
    meta = load_symbol_meta(pipeline_yaml)
    if "best_model" not in pred.columns:
        raise ValueError("pred_df.csv 缺少 best_model 列，请用含 best_model 输出的 EMEnsemble 重跑")

    latest_ts = pd.Timestamp(last_date)
    td = pred[pred[pred.columns[0]] == latest_ts].set_index("instrument")
    model_cols = [c for c in pred.columns if c.endswith("_score") and c != "best_model"]
    model_names = [c[:-6] for c in model_cols]

    rows = []
    for code in sorted(hold):
        name, index = meta.get(code, ("", ""))
        best = td.at[code, "best_model"] if code in td.index else ""
        best_col = f"{best}_score" if best else ""
        row = {
            "index": index, "code": code, "name": name,
            "ensemble_score": td.at[code, "score"] if code in td.index else "",
            "best_model": best,
        }
        for m in model_names:
            fc = f"{m}_score"
            row[fc] = td.at[code, fc] if (code in td.index and fc in td.columns) else ""
        row["best_model_score"] = td.at[code, best_col] if (best and code in td.index and best_col in td.columns) else ""
        code_series = pred[pred["instrument"] == code].set_index(pred.columns[0])[best_col] if best_col else pd.Series(dtype=float)
        bs, be = find_buy_window(code_series, latest_ts)
        row["buy_start"], row["buy_end"], row["pos_value"] = bs, be, round(hold[code], 2)

        # 年化：cleaned 首个有效 close -> 最新有效日期 close
        cleaned = load_cleaned_close(code)
        if cleaned is not None:
            dates, closes = cleaned
            last_dt, last_px = dates.iloc[-1], closes.iloc[-1]
            ann = calc_annualized(dates.iloc[0], closes.iloc[0], last_dt, last_px)
            row["annualized_return"] = f"{ann:.2%}" if isinstance(ann, float) else ""
            row["inception_date"] = str(dates.iloc[0].date())
            # 近 N 年年化：last_date 往前推 N 年，数据起点早于窗口起点才计算，否则置空
            for n in YEARS:
                cut = last_dt - pd.DateOffset(years=n)
                eligible = dates <= cut
                if eligible.any():
                    i0 = eligible.values.nonzero()[0][-1]
                    ann_n = calc_annualized(dates.iloc[i0], closes.iloc[i0], last_dt, last_px)
                    row[f"annualized_{n}y"] = f"{ann_n:.2%}" if isinstance(ann_n, float) else ""
                else:
                    row[f"annualized_{n}y"] = ""
        else:
            row["annualized_return"] = ""
            row["inception_date"] = ""
            for n in YEARS:
                row[f"annualized_{n}y"] = ""
        rows.append(row)

    cols = (["index", "code", "name", "ensemble_score", "best_model"]
            + model_cols + ["best_model_score", "buy_start", "buy_end",
                            "annualized_return", "annualized_1y", "annualized_3y",
                            "annualized_5y", "annualized_10y", "inception_date", "pos_value"])
    df = pd.DataFrame(rows)[cols]
    return df.sort_values(["index", "code"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="导出 SustainedBest 持仓明细 CSV")
    ap.add_argument("--run_dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_dir = load_latest_run_dir(Path(args.run_dir) if args.run_dir else None)
    if run_dir is None:
        print("未找到最新 SustainedBest run"); return
    df = build_report(run_dir)
    if df.empty:
        print("当前无持仓"); return
    _, last_date = load_holdings(run_dir)
    out = Path(args.out) if args.out else QLIB_ROOT / "output" / f"sustainedbest_holdings_{last_date}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"run_dir: {run_dir}")
    print(f"已输出 {len(df)} 只持仓 -> {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()