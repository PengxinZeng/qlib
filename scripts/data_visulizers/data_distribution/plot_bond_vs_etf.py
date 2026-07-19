"""
国债收益率 vs 510050 股债性价比可视化

输出图片（保存至 .../data_distribution/BondVsETF/）：
1. bond_yield_vs_510050.png — 单张双轴图：
     - 左轴(收益率 %)：中国 2/5Y、美国 2/5Y 国债收益率，
                       510050 盈利收益率(1/PE_TTM)，
                       以及利差(1/PE − CN 2Y，带零轴参考线)。
     - 右轴：510050 收盘价。
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
from pathlib import Path

# ── 中文字体 ──────────────────────────────────────────────────────────────────
mpl.rcParams["font.family"] = ["PingFang HK", "STHeiti", "Arial Unicode MS", "sans-serif"]
mpl.rcParams["axes.unicode_minus"] = False

# ── 路径配置 ──────────────────────────────────────────────────────────────────
BASE = Path("/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415")
BOND_CSV   = BASE / "source/cn_bond_rate/cn_bond_yield.csv"
QLIB_DIR   = BASE / "qlib_etf_index_Extend_wBond"
OUTPUT_DIR = BASE / "qlib_etf_index_Extend_wBond/data_distribution/BondVsETF"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# qlib
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
import qlib
from qlib.data import D

# ── 左轴曲线配置（收益率同轴，单位 %）────────────────────────────────────────
BOND_CURVES = [
    # (列名,      图例,          颜色,      线宽, 线型)
    ("cn_2y",  "CN 2Y",      "#1565C0", 0.8, "-"),
    ("cn_5y",  "CN 5Y",      "#1E88E5", 0.8, "-"),
    ("us_2y",  "US 2Y",      "#B71C1C", 0.8, "-"),
    ("us_5y",  "US 5Y",      "#E53935", 0.8, "-"),
]


def load_bond(start="2004-01-01") -> pd.DataFrame:
    df = pd.read_csv(BOND_CSV, parse_dates=["date"])
    df = df[df["date"] >= start].sort_values("date").reset_index(drop=True)
    return df


def load_etf510050(start="2004-01-01") -> pd.DataFrame:
    qlib.init(provider_uri=str(QLIB_DIR), region="cn")
    df = D.features(
        ["510050_CLEAN"],
        ["$close", "$pe_ttm"],
        freq="day",
        start_time=start,
        end_time="2026-05-03",
    )
    df.columns = ["close", "pe_ttm"]
    df = df.reset_index()
    if "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"])
    else:
        df["date"] = pd.to_datetime(df.iloc[:, 1])
    df = df[["date", "close", "pe_ttm"]].sort_values("date").reset_index(drop=True)
    # 计算 PE 倒数（盈利收益率），转换为百分比，过滤异常值
    df["earnings_yield"] = (1.0 / df["pe_ttm"] * 100).where(df["pe_ttm"] > 0)
    return df


def calc_spread(bond: pd.DataFrame, etf: pd.DataFrame) -> pd.DataFrame:
    """计算 510050 盈利收益率 - 中国2年期国债收益率 的利差"""
    merged = pd.merge_asof(
        etf[["date", "earnings_yield"]].dropna().sort_values("date"),
        bond[["date", "cn_2y"]].dropna().sort_values("date"),
        on="date", direction="nearest",
    )
    merged["spread_ey_cn2y"] = merged["earnings_yield"] - merged["cn_2y"]
    return merged[["date", "spread_ey_cn2y"]]


def plot(bond: pd.DataFrame, etf: pd.DataFrame):
    spread = calc_spread(bond, etf)
    fig, ax_left = plt.subplots(figsize=(18, 7))
    fig.patch.set_facecolor("#fafafa")
    ax_left.set_facecolor("#fafafa")

    ax_close = ax_left.twinx()   # 右轴：收盘价

    # ── 左轴：国债收益率 ──────────────────────────────────────────────────────
    left_handles = []
    for col, label, color, lw, ls in BOND_CURVES:
        s = bond[["date", col]].dropna()
        if s.empty:
            continue
        line, = ax_left.plot(
            s["date"], s[col],
            color=color, linewidth=lw, linestyle=ls, alpha=0.75, label=label,
        )
        left_handles.append(line)

    # ── 左轴：510050 PE倒数（盈利收益率）─────────────────────────────────────
    ey = etf[["date", "earnings_yield"]].dropna()
    ey_line, = ax_left.plot(
        ey["date"], ey["earnings_yield"],
        color="#FF6F00", linewidth=1.3, linestyle="-", alpha=0.9,
        label="510050 盈利收益率 (1/PE)",
    )
    left_handles.append(ey_line)

    # ── 左轴：利差（1/PE - CN 2Y）────────────────────────────────────────────
    sp_line, = ax_left.plot(
        spread["date"], spread["spread_ey_cn2y"],
        color="#7B1FA2", linewidth=1.1, linestyle="--", alpha=0.85,
        label="利差 (1/PE − CN 2Y)",
    )
    # 零轴参考线
    ax_left.axhline(0, color="#7B1FA2", linewidth=0.6, linestyle=":", alpha=0.5)
    left_handles.append(sp_line)

    ax_left.set_ylabel("收益率 (%)", fontsize=10, color="#333333")
    ax_left.tick_params(axis="y", labelcolor="#333333", labelsize=8)
    ax_left.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax_left.set_zorder(2)
    ax_left.patch.set_visible(False)

    # ── 右轴：510050 收盘价 ───────────────────────────────────────────────────
    close_line, = ax_close.plot(
        etf["date"], etf["close"],
        color="#2E7D32", linewidth=1.2, linestyle="-", alpha=0.55,
        label="510050 收盘价",
    )
    ax_close.set_ylabel("510050 收盘价", fontsize=10, color="#2E7D32")
    ax_close.tick_params(axis="y", labelcolor="#2E7D32", labelsize=8)
    ax_close.spines["right"].set_color("#2E7D32")
    ax_close.set_zorder(1)

    # ── x 轴格式 ──────────────────────────────────────────────────────────────
    ax_left.xaxis.set_major_locator(mdates.YearLocator(2))
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_left.xaxis.set_minor_locator(mdates.YearLocator(1))
    plt.setp(ax_left.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)

    ax_left.grid(axis="x", linestyle=":", alpha=0.3, color="gray", zorder=0)
    ax_left.grid(axis="y", linestyle="--", alpha=0.2, color="gray", zorder=0)

    # ── 标题 & 图例 ──────────────────────────────────────────────────────────
    ax_left.set_title(
        "国债收益率 vs 510050 盈利收益率(1/PE) vs 收盘价  |  利差=1/PE−CN 2Y",
        fontsize=14, fontweight="bold", pad=12,
    )

    # 左轴图例
    leg_left = ax_left.legend(
        handles=left_handles,
        title="收益率（左轴）", title_fontsize=8,
        loc="upper left", fontsize=8,
        framealpha=0.88, ncol=2,
    )
    ax_left.add_artist(leg_left)

    # 右轴图例
    ax_close.legend(
        handles=[close_line],
        loc="upper right", fontsize=8, framealpha=0.88,
    )

    ax_left.set_xlim(
        max(bond["date"].min(), etf["date"].min()),
        min(bond["date"].max(), etf["date"].max()),
    )

    plt.tight_layout()
    out = OUTPUT_DIR / "bond_yield_vs_510050.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"图表已保存: {out}")
    plt.close()


if __name__ == "__main__":
    START = "2004-01-01"
    bond = load_bond(start=START)
    etf  = load_etf510050(start=START)
    print(f"国债数据: {len(bond)} 行, {bond['date'].iloc[0].date()} ~ {bond['date'].iloc[-1].date()}")
    print(f"510050:  {len(etf)} 行, {etf['date'].iloc[0].date()} ~ {etf['date'].iloc[-1].date()}")
    plot(bond, etf)
