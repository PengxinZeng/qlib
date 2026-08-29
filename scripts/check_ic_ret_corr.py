# -*- coding: utf-8 -*-
"""
验证: 测试 IC 与测试收益的相关强度（日度层面 + 3 点层面）
- 日度: 日 IC 序列(ic.pkl) vs 实际回测日收益(report return) 逐日相关
- 3点:  测试 IC / 验证 IC@best_step / 年化收益 的 Spearman
"""
import os
import pickle

import numpy as np
import pandas as pd

MLRUNS = r"D:/Pengxin/CodeBase/Quant/qlib/mlruns"
RUNS = {
    "seed0": ("267449130026730172", "79326c098519402e82d371352bc4f6f3"),
    "seed1": ("883734610144807679", "a8677fc819ee49869372fd687d6b41a7"),
    "seed2": ("128358302298500901", "a1558bb97d884891be703e91bec48b1e"),
}
SEEDS = list(RUNS.keys())


def load_pkl(seed, name):
    exp, run = RUNS[seed]
    with open(f"{MLRUNS}/{exp}/{run}/artifacts/{name}", "rb") as f:
        return pickle.load(f)


def metric_curve(seed, name):
    exp, run = RUNS[seed]
    path = f"{MLRUNS}/{exp}/{run}/metrics/{name}"
    df = pd.read_csv(path, sep=" ", header=None, names=["ts", "value", "step"])
    return df.set_index("step")["value"]


def spearman(x, y):
    rx = pd.Series(x).rank()
    ry = pd.Series(y).rank()
    return np.corrcoef(rx, ry)[0, 1]


def main():
    print("=" * 80)
    print("1) 3 点层面：测试 IC / 验证 IC@best / 年化收益(含成本) 排序")
    print("=" * 80)
    test_ic = {}
    val_ic = {}
    ann = {}
    for s in SEEDS:
        ic = load_pkl(s, "sig_analysis/ic.pkl")
        test_ic[s] = ic.mean()
        best = int(metric_curve(s, "best_step").iloc[-1])
        val_ic[s] = metric_curve(s, "val_metric").loc[best]
        rep = load_pkl(s, "portfolio_analysis/report_normal_1day.pkl")
        r = rep["return"]
        ann[s] = (1.0 + r.mean()) ** 252 - 1.0
    df = pd.DataFrame({"测试IC": test_ic, "验证IC@best": val_ic, "年化收益": ann}).T
    print(df.round(4).to_string())

    print("\nSpearman（n=3，仅示意，无统计意义）:")
    v_ic = list(val_ic.values())
    t_ic = list(test_ic.values())
    a = list(ann.values())
    print(f"  测试IC vs 年化收益   : {spearman(t_ic, a):+.2f}")
    print(f"  验证IC vs 年化收益   : {spearman(v_ic, a):+.2f}")
    print(f"  验证IC vs 测试IC     : {spearman(v_ic, t_ic):+.2f}")

    print()
    print("=" * 80)
    print("2) 日度层面：日 IC 序列 vs 实际回测日收益（含成本）的逐日相关")
    print("=" * 80)
    print("IC[t] 是 t 日信号的预测力，组合日收益 return[t] 是 t-1 决策 t 日实现")
    print("→ 严格对齐用 IC[t] vs return[t+1]（shift 1）；同时给同索引版本参考")
    for s in SEEDS:
        ic = load_pkl(s, "sig_analysis/ic.pkl")
        rep = load_pkl(s, "portfolio_analysis/report_normal_1day.pkl")
        ret = rep["return"]
        # 对齐共同日期
        idx = ic.index.intersection(ret.index)
        ic_a = ic.loc[idx]
        ret_a = ret.loc[idx]
        # 日度 Pearson
        r0 = np.corrcoef(ic_a.values, ret_a.values)[0, 1]
        # shift: IC[t] 预测 t+1 收益
        ret_shift = ret_a.shift(-1)
        m = ic_a.notna() & ret_shift.notna()
        r1 = np.corrcoef(ic_a[m].values, ret_shift[m].values)[0, 1]
        # 日度 Spearman
        rs0 = spearman(ic_a, ret_a)
        # 累计: IC 高日子 vs 低日子的收益差（按日 IC 中位数分组）
        med = ic_a.median()
        hi = ret_a[ic_a >= med].mean()
        lo = ret_a[ic_a < med].mean()
        print(f"[{s}] n={len(ic_a)}  日IC vs 同日收益 pearson={r0:+.3f} | "
              f"日IC vs T+1收益 pearson={r1:+.3f} | 日rank相关={rs0:+.3f}")
        print(f"        IC≥中位日子均收益={hi*100:.3f}%   IC<中位日子均收益={lo*100:.3f}%  "
              f"(差={(hi-lo)*100:.3f}%)")

    print()
    print("=" * 80)
    print("3) 验证 IC 的'预选'能力：验证 IC 与测试 IC 是否一致")
    print("=" * 80)
    print("（3 点不足以定论，仅列数字；真正检验需多 seed + 多训练配置）")
    for s in SEEDS:
        print(f"  [{s}] 验证IC@best={val_ic[s]:.4f}   测试IC={test_ic[s]:.4f}   "
              f"验证段(2018-2021) vs 测试段(2021-2026)")


if __name__ == "__main__":
    main()
