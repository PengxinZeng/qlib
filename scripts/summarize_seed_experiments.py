"""汇总 seed 实验指标到 output/XXX.csv"""
import os
import sys
from pathlib import Path

import pandas as pd

from mlflow.tracking import MlflowClient

MLRUNS = "file:D:/Pengxin/CodeBase/Quant/qlib/mlruns"
OUTPUT_DIR = Path("D:/Pengxin/CodeBase/Quant/qlib/output")

# 实验名 -> seed
EXPERIMENTS = {
    "mlp_all_weather_alpha158_zscore_seed0": 0,
    "mlp_all_weather_alpha158_zscore_seed1": 1,
    "mlp_all_weather_alpha158_zscore_seed3": 3,
}

# 需要提取的指标
METRICS = [
    "IC",
    "ICIR",
    "Rank IC",
    "Rank ICIR",
    "1day.return_with_cost.annualized_return",
    "1day.return_with_cost.max_drawdown",
    "1day.return_with_cost.information_ratio",
    "1day.return_without_cost.annualized_return",
    "1day.excess_return_with_cost.annualized_return",
    "1day.excess_return_with_cost.information_ratio",
    "1day.excess_return_without_cost.annualized_return",
    "1day.excess_return_without_cost.information_ratio",
]


def main():
    client = MlflowClient(MLRUNS)
    rows = []
    for exp_name, seed in EXPERIMENTS.items():
        exp = client.get_experiment_by_name(exp_name)
        if exp is None:
            print(f"[skip] experiment not found: {exp_name}")
            continue
        runs = client.search_runs([exp.experiment_id])
        for r in sorted(runs, key=lambda x: x.info.start_time):
            m = r.data.metrics
            # 只保留完成且回测指标完整的 run（排除卡死残留的 RUNNING/无指标 run）
            if r.info.status != "FINISHED":
                print(f"[skip] 非 FINISHED: {exp_name}/{r.info.run_id[:8]} ({r.info.status})")
                continue
            ar_cost = m.get("1day.return_with_cost.annualized_return")
            if ar_cost is None or pd.isna(ar_cost):
                print(f"[skip] 回测指标不完整: {exp_name}/{r.info.run_id[:8]}")
                continue
            row = {
                "experiment": exp_name,
                "run_id": r.info.run_id,
                "seed": r.data.params.get("model.kwargs.seed", seed),
                "status": r.info.status,
                "start_time": pd.Timestamp(r.info.start_time, unit="ms", tz="Asia/Shanghai").strftime("%m-%d %H:%M:%S"),
            }
            for k in METRICS:
                row[k] = m.get(k)
            rows.append(row)

    df = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "seed_experiments_summary.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"已写入: {out_path} ({len(df)} 行)")
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
