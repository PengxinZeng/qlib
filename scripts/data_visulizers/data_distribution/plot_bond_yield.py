"""
中美国债收益率历史曲线可视化

数据源: cn_bond_yield.csv（东方财富数据中心）

输出图片（保存至 .../data_distribution/BondYield/）：
1. bond_yield_all.png       — 中美各期限(2/5/10/30Y)国债收益率曲线叠加在同一张图，
                              x 轴为时间，y 轴为收益率(%)，中国/美国分组图例。
2. bond_yield_cn_us_10y.png — 上下双子图：上图为中美 10 年期收益率对比，
                              下图为中美 10 年期利差(CN − US)填色面积图(带零轴)。
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from pathlib import Path
import sys

# 跨平台路径集中配置（Mac / Windows 兼容）
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import path_config  # noqa: E402

# 中文字体配置
mpl.rcParams["font.family"] = ["PingFang HK", "STHeiti", "Arial Unicode MS", "sans-serif"]
mpl.rcParams["axes.unicode_minus"] = False

# ── 路径配置 ──────────────────────────────────────────────────────────────────
DATA_PATH = path_config.BOND_FILE
OUTPUT_DIR = path_config.QLIB_DATA_DIR / "data_distribution" / "BondYield"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 曲线配置：(列名, 显示名称, 颜色, 线型) ────────────────────────────────────
CURVES = [
    # 中国国债
    ("cn_2y",  "中国 2年",  "#1a6faf", "-"),
    ("cn_5y",  "中国 5年",  "#3a9ad9", "--"),
    ("cn_10y", "中国 10年", "#0d3f6e", "-"),
    ("cn_30y", "中国 30年", "#5bc4f5", ":"),
    # 美国国债
    ("us_2y",  "美国 2年",  "#c0392b", "-"),
    ("us_5y",  "美国 5年",  "#e67e22", "--"),
    ("us_10y", "美国 10年", "#7b241c", "-"),
    ("us_30y", "美国 30年", "#f1948a", ":"),
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def plot_all_curves(df: pd.DataFrame):
    """将所有收益率曲线绘制在一张图上"""
    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    for col, label, color, ls in CURVES:
        if col not in df.columns:
            continue
        series = df[["date", col]].dropna()
        if series.empty:
            continue
        ax.plot(
            series["date"],
            series[col],
            label=label,
            color=color,
            linestyle=ls,
            linewidth=1.4,
            alpha=0.85,
        )

    # ── 格式 ──
    ax.set_title("中美国债收益率历史走势", fontsize=16, fontweight="bold", pad=14)
    ax.set_xlabel("日期", fontsize=12)
    ax.set_ylabel("收益率 (%)", fontsize=12)

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    plt.xticks(rotation=45, ha="right")

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="gray")
    ax.grid(axis="x", linestyle=":", alpha=0.25, color="gray")

    # 分组图例：中国 / 美国
    handles, labels = ax.get_legend_handles_labels()
    cn_h = [(h, l) for h, l in zip(handles, labels) if l.startswith("中国")]
    us_h = [(h, l) for h, l in zip(handles, labels) if l.startswith("美国")]

    leg_cn = ax.legend(
        *zip(*cn_h), title="中国国债", loc="upper left",
        fontsize=9, title_fontsize=9, framealpha=0.85,
    )
    ax.add_artist(leg_cn)
    ax.legend(
        *zip(*us_h), title="美国国债", loc="upper right",
        fontsize=9, title_fontsize=9, framealpha=0.85,
    )

    ax.set_xlim(df["date"].min(), df["date"].max())

    plt.tight_layout()

    out_path = OUTPUT_DIR / "bond_yield_all.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {out_path}")
    plt.close()


def plot_cn_vs_us_10y(df: pd.DataFrame):
    """单独绘制中美10年期对比（附利差）"""
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    fig.patch.set_facecolor("#f8f9fa")
    for ax in (ax1, ax2):
        ax.set_facecolor("#f8f9fa")

    # 主图：两条收益率曲线
    for col, label, color in [
        ("cn_10y", "中国 10年期", "#0d3f6e"),
        ("us_10y", "美国 10年期", "#7b241c"),
    ]:
        s = df[["date", col]].dropna()
        ax1.plot(s["date"], s[col], label=label, color=color, linewidth=1.5)

    ax1.set_title("中美10年期国债收益率对比", fontsize=14, fontweight="bold", pad=10)
    ax1.set_ylabel("收益率 (%)", fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax1.legend(fontsize=10, framealpha=0.85)
    ax1.grid(linestyle="--", alpha=0.4, color="gray")

    # 副图：中美10年期利差
    spread = df[["date", "cn_10y", "us_10y"]].dropna()
    spread["diff"] = spread["cn_10y"] - spread["us_10y"]
    ax2.fill_between(
        spread["date"], spread["diff"],
        where=(spread["diff"] >= 0), color="#1a6faf", alpha=0.5, label="CN > US"
    )
    ax2.fill_between(
        spread["date"], spread["diff"],
        where=(spread["diff"] < 0), color="#c0392b", alpha=0.5, label="CN < US"
    )
    ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("利差 (%)", fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax2.legend(fontsize=9, framealpha=0.85)
    ax2.grid(linestyle="--", alpha=0.3, color="gray")

    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "bond_yield_cn_us_10y.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {out_path}")
    plt.close()


if __name__ == "__main__":
    df = load_data()
    print(f"数据加载完成: {len(df)} 行，{df['date'].min().date()} ~ {df['date'].max().date()}")

    # 图1：全部曲线
    plot_all_curves(df)

    # 图2：中美10年期对比 + 利差
    plot_cn_vs_us_10y(df)
