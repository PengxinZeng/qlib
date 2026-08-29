# -*- coding: utf-8 -*-
"""
分析 MLP all_weather alpha158 globalnorm 三个 seed 收益/IC 方差来源。

三个 run（seed0/1/2，0820-083814 批次）：
  - 年化收益(无成本): 23.96% / 14.78% / 9.61%   -> 差距很大
  - 测试 IC: 0.0335 / 0.0133 / 0.0094           -> 差距也很大

问题：
  Q1: 方差是 topk 策略导致还是模型本身导致？
  Q2: 模型本身方差大，是不是"选到的 iter 对应 val loss / train loss 差"？

方法：
  A. 信号层面（模型输出差异）：pred 全样本 Pearson 相关、每日截面 Rank 相关、
     top5 集合 Jaccard 重合度、日 IC 序列相关/均值
  B. 训练层面：best_step、best_step 处 val_loss/val_metric/train_loss、
     同一 step(1340) 处三 seed 的 val_metric（区分"选点差异"与"轨迹差异"）
  C. 策略放大：用同一批信号模拟 ①top5 等权(无dropout) ②全样本平滑加权 组合，
     对比跨 seed 年化收益离散度 vs 实际 topk-dropout 的离散度
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

MLRUNS = r"D:/Pengxin/CodeBase/Quant/qlib/mlruns"
DATA_URI = r"D:/Pengxin/CodeBase/Quant/QuantDataBank/all_weather_data/qlib_all_weather"

RUNS = {
    "seed0": ("267449130026730172", "79326c098519402e82d371352bc4f6f3"),
    "seed1": ("883734610144807679", "a8677fc819ee49869372fd687d6b41a7"),
    "seed2": ("128358302298500901", "a1558bb97d884891be703e91bec48b1e"),
}
SEEDS = list(RUNS.keys())


def load_pkl(seed: str, name: str):
    exp, run = RUNS[seed]
    with open(f"{MLRUNS}/{exp}/{run}/artifacts/{name}", "rb") as f:
        return pickle.load(f)


def load_metric_curve(seed: str, name: str):
    """mlflow metric 文件 -> Series(step -> value)"""
    exp, run = RUNS[seed]
    path = f"{MLRUNS}/{exp}/{run}/metrics/{name}"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep=" ", header=None, names=["ts", "value", "step"])
    return df.set_index("step")["value"]


def annualized(ret: pd.Series, ann: int = 252) -> float:
    return (1.0 + ret.mean()) ** ann - 1.0 if len(ret) else np.nan


def main():
    print("=" * 90)
    print("A. 信号层面：模型预测差异（pred.pkl）")
    print("=" * 90)
    preds = {s: load_pkl(s, "pred.pkl") for s in SEEDS}
    # pred 结构：MultiIndex (datetime, instrument)，列 score
    for s, p in preds.items():
        print(f"[{s}] pred shape={p.shape}, 范围 {p.index.get_level_values(0).min()} ~ "
              f"{p.index.get_level_values(0).max()}, 标的数={p.index.get_level_values(1).nunique()}")

    # 对齐：pivot 成 datetime x instrument
    wide = {s: p["score"].unstack() for s, p in preds.items()}
    # 只用三份都非空的日期
    common_days = set(wide["seed0"].index) & set(wide["seed1"].index) & set(wide["seed2"].index)
    common_days = sorted(common_days)
    common_inst = sorted(
        set(wide["seed0"].columns) & set(wide["seed1"].columns) & set(wide["seed2"].columns)
    )
    print(f"共同交易日 {len(common_days)} 天，共同标的 {len(common_inst)} 只")
    W = {s: wide[s].loc[common_days, common_inst].astype(float) for s in SEEDS}

    # 1) 全样本 pooled Pearson
    print("\n-- 全样本 pooled 相关（所有天*所有标的摊平）--")
    pooled = {}
    for s in SEEDS:
        pooled[s] = W[s].stack()
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = SEEDS[i], SEEDS[j]
            r = np.corrcoef(pooled[a].values, pooled[b].values)[0, 1]
            print(f"  corr({a},{b}) = {r:.4f}")

    # 2) 每日截面 rank 相关（Spearman，按天算再平均）
    print("\n-- 每日截面 Spearman rank 相关（按天计算后平均）--")
    daily_rank_corr = {f"{a}-{b}": [] for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]}
    for d in common_days:
        for i, a in enumerate(SEEDS):
            for b in SEEDS[i + 1:]:
                da = W[a].loc[d].dropna()
                db = W[b].loc[d].dropna()
                idx = da.index.intersection(db.index)
                if len(idx) < 5:
                    continue
                xa = da.loc[idx].rank().values
                xb = db.loc[idx].rank().values
                daily_rank_corr[f"{a}-{b}"].append(np.corrcoef(xa, xb)[0, 1])
    for k, v in daily_rank_corr.items():
        print(f"  mean rank-corr({k}) = {np.mean(v):.4f}  (std={np.std(v):.4f}, n={len(v)})")

    # 3) top5 集合 Jaccard 重合度
    print("\n-- 每日 top5 集合重合度（Jaccard，平均）--")
    top5 = {s: W[s].apply(lambda row: set(row.sort_values(ascending=False).head(5).index), axis=1)
            for s in SEEDS}
    for i, a in enumerate(SEEDS):
        for b in SEEDS[i + 1:]:
            js = [len(x & y) / len(x | y) for x, y in zip(top5[a], top5[b])]
            ov = [len(x & y) / 5.0 for x, y in zip(top5[a], top5[b])]
            print(f"  Jaccard({a},{b}) = {np.mean(js):.4f}   重合率(交集/5) = {np.mean(ov):.4f}")

    # 4) 日 IC 序列
    print("\n-- 日 IC / RankIC 序列（sig_analysis/ic.pkl）--")
    ics = {s: load_pkl(s, "sig_analysis/ic.pkl") for s in SEEDS}
    rics = {s: load_pkl(s, "sig_analysis/ric.pkl") for s in SEEDS}
    ic_df = pd.DataFrame({s: ics[s] for s in SEEDS}).dropna()
    ric_df = pd.DataFrame({s: rics[s] for s in SEEDS}).dropna()
    print("日 IC 均值 / 标准差 / 序列相关:")
    for s in SEEDS:
        m = ic_df[s].mean()
        sd = ic_df[s].std()
        se = sd / np.sqrt(len(ic_df))
        print(f"  [{s}] IC mean={m:.4f} std={sd:.4f} SE={se:.4f} n={len(ic_df)}")
    for i, a in enumerate(SEEDS):
        for b in SEEDS[i + 1:]:
            r = ic_df[a].corr(ic_df[b])
            rr = ric_df[a].corr(ric_df[b])
            print(f"  日IC序列相关({a},{b})={r:.4f}   日RankIC序列相关={rr:.4f}")

    print()
    print("=" * 90)
    print("B. 训练层面：best_step 与 val/train loss")
    print("=" * 90)
    for s in SEEDS:
        best = load_metric_curve(s, "best_step")
        best_step = int(best.iloc[-1]) if best is not None else None
        vl = load_metric_curve(s, "val_loss")
        vm = load_metric_curve(s, "val_metric")
        tl = load_metric_curve(s, "train_loss")
        vl_best = vl.loc[best_step] if best_step in vl.index else np.nan
        vm_best = vm.loc[best_step] if best_step in vm.index else np.nan
        tl_best = tl.loc[best_step] if best_step in tl.index else np.nan
        min_vl = vl.min()
        min_vl_step = int(vl.idxmin())
        # 同一 step 对比（1340 = 三 seed 中最早的 best_step）
        vm_1340 = vm.loc[1340] if 1340 in vm.index else np.nan
        vm_1780 = vm.loc[1780] if 1780 in vm.index else np.nan
        print(f"[{s}] best_step={best_step}  val_loss@{best_step}={vl_best:.4f}  "
              f"val_metric@{best_step}={vm_best:.4f}  train_loss@{best_step}={tl_best:.4f}  "
              f"| min val_loss={min_vl:.4f}@{min_vl_step}  | val_metric@1340={vm_1340:.4f}  "
              f"val_metric@1780={vm_1780:.4f}  | val_metric 终值={vm.iloc[-1]:.4f}")
        print(f"    val_metric 曲线 (每40步抽样): "
              f"{[round(float(x), 3) for x in vm[::2].values]}")

    # val_metric 在"各自 best_step" vs IC 的一致性
    print("\n-- val_metric@best_step 与测试 IC 对照 --")
    ics_mean = {s: ic_df[s].mean() for s in SEEDS}
    for s in SEEDS:
        best = int(load_metric_curve(s, "best_step").iloc[-1])
        vm = load_metric_curve(s, "val_metric")
        print(f"  [{s}] val_metric@best_step={vm.loc[best]:.4f}   IC={ics_mean[s]:.4f}")

    print()
    print("=" * 90)
    print("C. 策略放大效应：同一信号 → 不同组合构建方式 → 跨 seed 离散度")
    print("=" * 90)
    # 加载真实 close 计算日收益（无成本）
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D

    qlib.init(provider_uri=DATA_URI, region=REG_CN)
    start, end = common_days[0], common_days[-1]
    close = D.features(common_inst, ["$close"], start_time=start, end_time=end, freq="day")
    close = close["$close"].unstack().T  # datetime x instrument
    ret = close.pct_change()
    # 用信号日 t 的 top5 / 权重，赚 t+1 的收益 -> 对齐：ret 与 W 同索引
    ret_a = ret.loc[W["seed0"].index].shift(-1)  # t 日决策 → t+1 收益
    ret_a = ret_a.iloc[:-1]
    days = ret_a.index

    def top5_ret(w):
        """每日 top5 等权（无 dropout、无成本）"""
        r = []
        for d in days:
            s = w.loc[d].sort_values(ascending=False)
            top5_inst = s.head(5).index
            r.append(ret_a.loc[d, top5_inst].mean())
        return pd.Series(r, index=days)

    def soft_ret(w):
        """全样本平滑加权：权重 ∝ max(score,0) 归一化（无成本）"""
        r = []
        for d in days:
            s = w.loc[d]
            wts = s.clip(lower=0)
            if wts.sum() <= 0:
                wts = pd.Series(1.0, index=s.index)
            wts = wts / wts.sum()
            r.append((ret_a.loc[d] * wts).sum())
        return pd.Series(r, index=days)

    def eq_ret(w):
        """全样本等权（基准，跨 seed 应完全一致）"""
        r = []
        for d in days:
            r.append(ret_a.loc[d].mean())
        return pd.Series(r, index=days)

    def eq_ret(w):
        """全样本等权（基准，跨 seed 应完全一致）"""
        r = []
        for d in days:
            r.append(ret_a.loc[d].mean())
        return pd.Series(r, index=days)

    def topk_dropout_ret(w, topk=5, n_drop=1):
        """复刻 TopkDropoutStrategy(method_buy=top, method_sell=bottom) 无成本版"""
        held = set()
        r = []
        for d in days:
            if held:
                scored = w.loc[d].reindex(held).sort_values()
                sell = set(scored.head(n_drop).index)
                held -= sell
            candidates = w.loc[d].drop(index=list(held)).sort_values(ascending=False)
            need = topk - len(held)
            if need > 0:
                held |= set(candidates.head(need).index)
            r.append(float(np.nanmean(ret_a.loc[d, list(held)].values)))
        return pd.Series(r, index=days)

    schemes = {
        "① top5等权(无dropout)": top5_ret,
        "② top5-dropout模拟(无成本)": topk_dropout_ret,
        "③ 全样本平滑加权(score>0)": soft_ret,
        "④ 全样本等权(基准)": eq_ret,
    }
    actual = {}
    for s in SEEDS:
        rep = load_pkl(s, "portfolio_analysis/report_normal_1day.pkl")
        # report 结构: MultiIndex (datetime, account?), 列含 return
        actual[s] = rep["return"] if "return" in rep.columns else rep.iloc[:, 0]

    print(f"{'方案':<28}{'seed0 年化':>12}{'seed1 年化':>12}{'seed2 年化':>12}{'跨seed std':>12}")
    all_res = {}
    for name, fn in schemes.items():
        res = {s: annualized(fn(W[s])) for s in SEEDS}
        all_res[name] = res
        std = np.std(list(res.values()))
        print(f"{name:<28}{res['seed0']:>12.4f}{res['seed1']:>12.4f}{res['seed2']:>12.4f}{std:>12.4f}")
    # 实际 topk-dropout（含成本，取自 report）
    res_act = {s: annualized(actual[s]) for s in SEEDS}
    std_act = np.std(list(res_act.values()))
    print(f"{'⑤ 实际topk-dropout(含成本)':<28}{res_act['seed0']:>12.4f}{res_act['seed1']:>12.4f}"
          f"{res_act['seed2']:>12.4f}{std_act:>12.4f}")

    # 实际持仓重合度
    print("\n-- 实际回测持仓集合重合率（positions）--")
    pos = {s: load_pkl(s, "portfolio_analysis/positions_normal_1day.pkl") for s in SEEDS}
    for i, a in enumerate(SEEDS):
        for b in SEEDS[i + 1:]:
            pa, pb = pos[a], pos[b]
            common_d = sorted(set(pa.keys()) & set(pb.keys()))
            overlap = []
            for d in common_d:
                ha = set(pa[d].get_stock_list())
                hb = set(pb[d].get_stock_list())
                if ha and hb:
                    overlap.append(len(ha & hb) / min(len(ha), len(hb)))
            print(f"  持仓重合率(交集/较小持仓数, 平均) [{a}-{b}] = {np.mean(overlap):.4f}  "
                  f"n={len(common_d)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback

        traceback.print_exc()
        sys.exit(1)
