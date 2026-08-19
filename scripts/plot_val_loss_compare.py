"""对比 CSZScoreNorm vs ZScoreNorm 的 val loss 曲线"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# 中文字体：Windows 用微软雅黑，mac 用 PingFang；找不到则回退 DejaVu Sans
from matplotlib import font_manager

for _f in ("Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS"):
    if any(_f.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False

MLRUNS = "D:/Pengxin/CodeBase/Quant/qlib/mlruns"
OUT = Path("D:/Pengxin/CodeBase/Quant/qlib/output")
OUT.mkdir(parents=True, exist_ok=True)

# 实验名 -> (run_id, 标签, 颜色)
EXPS = {
    "mlp_all_weather_alpha158_seed0": {"label": "CSZScoreNorm (按日zscore)", "color": "#d62728"},
    "mlp_all_weather_alpha158_globalnorm": {"label": "ZScoreNorm (train全局)", "color": "#1f77b4"},
}


def load_series(rid: str, name: str) -> pd.DataFrame:
    # 先找 run 目录
    import glob

    pats = glob.glob(f"{MLRUNS}/**/{rid}/metrics/{name}", recursive=True)
    if not pats:
        raise FileNotFoundError(f"metrics/{name} not found for {rid}")
    rows = []
    with open(pats[0], encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                rows.append({"step": int(parts[2]), "value": float(parts[1])})
    return pd.DataFrame(rows)


def get_run_id(exp_name: str) -> str:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(f"file:{MLRUNS}")
    exp = client.get_experiment_by_name(exp_name)
    runs = client.search_runs([exp.experiment_id])
    # 取最新 FINISHED 的 run
    runs = [r for r in runs if r.info.status == "FINISHED"]
    if not runs:
        raise ValueError(f"no finished run in {exp_name}")
    return sorted(runs, key=lambda r: r.info.start_time)[-1].info.run_id


def main():
    plt.figure(figsize=(12, 6))
    for exp_name, meta in EXPS.items():
        rid = get_run_id(exp_name)
        df = load_series(rid, "val_loss")
        plt.plot(df.step, df.value, label=meta["label"], color=meta["color"], linewidth=1.8)
        best = df.loc[df.value.idxmin()]
        plt.scatter([best.step], [best.value], color=meta["color"], zorder=5, s=40)
        plt.annotate(
            f"{meta['label']}\nbest={best.value:.4f} @ step {int(best.step)}",
            xy=(best.step, best.value),
            xytext=(best.step + 60, best.value + 0.05),
            color=meta["color"],
            fontsize=9,
            arrowprops=dict(arrowstyle="->", color=meta["color"], lw=0.8),
        )
        print(f"{meta['label']}: n_points={len(df)}, min={df.value.min():.4f} @ {int(best.step)}")

    plt.xlabel("Step (每 20 步评估一次)")
    plt.ylabel("valid loss (MSE)")
    plt.title("CSZScoreNorm vs ZScoreNorm 的 valid loss 训练曲线（seed=0, 对齐base 157维）")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT / "val_loss_compare_csz_vs_global.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
