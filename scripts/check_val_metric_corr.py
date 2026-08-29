# -*- coding: utf-8 -*-
"""
统计三个 seed 在 **验证集 (valid, 2018-04-19 ~ 2021-11-12)** 上
收益率 / IC / Loss 三个指标之间的相关性（跨 3 个 seed）。

指标口径：
  - Loss    : val_loss@best_step（训练时 mlflow 记录；mse 口径）
              + 附 min val_loss 全序列最小值
  - IC      : ① val_metric@best_step（训练记录，ICLoss=按标的时序IC平均）
              ② 重算日截面 IC 均值（pred vs label，与测试段 sig_analysis 同口径）
  - 收益率  : 验证段重新推理 pred 后构造
              ① top5 等权（无成本）年化
              ② top5-dropout 模拟（无成本）年化
              ③ 全样本平滑加权（无成本）年化

输出：3x3 指标表 + 两两 Pearson/Spearman（n=3，仅示意）
"""
import os
import pickle

import numpy as np
import pandas as pd

from qlib.config import C
from qlib.data import D
from qlib.utils import init_instance_by_config

MLRUNS = r"D:/Pengxin/CodeBase/Quant/qlib/mlruns"
DATA_URI = r"D:/Pengxin/CodeBase/Quant/QuantDataBank/all_weather_data/qlib_all_weather"
CACHE_DIR = r"D:/Pengxin/CodeBase/Quant/QuantDataBank/all_weather_data/handler_cache"

RUNS = {
    # seed -> dict(exp_name, exp_id, run)
    "seed0": {"exp_name": "mlp_all_weather_alpha158_globalnorm",
               "exp_id": "267449130026730172",
               "run": "79326c098519402e82d371352bc4f6f3"},
    "seed1": {"exp_name": "mlp_all_weather_alpha158_globalnorm_seed1",
               "exp_id": "883734610144807679",
               "run": "a8677fc819ee49869372fd687d6b41a7"},
    "seed2": {"exp_name": "mlp_all_weather_alpha158_globalnorm_seed2",
               "exp_id": "128358302298500901",
               "run": "a1558bb97d884891be703e91bec48b1e"},
}
SEEDS = list(RUNS.keys())
VALID_START, VALID_END = "2018-04-19", "2021-11-12"


def metric_curve(seed, name):
    exp_id, run = RUNS[seed]["exp_id"], RUNS[seed]["run"]
    path = f"{MLRUNS}/{exp_id}/{run}/metrics/{name}"
    df = pd.read_csv(path, sep=" ", header=None, names=["ts", "value", "step"])
    return df.set_index("step")["value"]


def load_recorder(seed):
    from qlib.workflow import R

    exp_obj = R.get_exp(experiment_name=RUNS[seed]["exp_name"])
    return exp_obj.get_recorder(recorder_id=RUNS[seed]["run"])


def build_dataset(seed):
    rec = load_recorder(seed)
    from qlib.workflow.task.utils import replace_task_handler_with_cache

    task_config = rec.load_object("task")
    task_config = replace_task_handler_with_cache(task_config, cache_dir=CACHE_DIR)
    dataset = init_instance_by_config(task_config["dataset"])
    return rec, task_config, dataset


def daily_ic(pred, label):
    """日截面 IC：pred/label 均为 MultiIndex (datetime, instrument)"""
    p = pred["score"].unstack()
    l = label.iloc[:, 0].unstack()
    idx = p.index.intersection(l.index)
    ics = []
    for d in idx:
        a = p.loc[d].dropna()
        b = l.loc[d].dropna()
        common = a.index.intersection(b.index)
        if len(common) >= 5:
            va, vb = a.loc[common].values, b.loc[common].values
            if va.std() > 0 and vb.std() > 0:
                ics.append(np.corrcoef(va, vb)[0, 1])
    return pd.Series(ics, index=idx)


def annualized(ret):
    return (1.0 + ret.mean()) ** 252 - 1.0 if len(ret) else np.nan


def combo_rets(pred, close, days):
    """三种组合年化（无成本），pred: MultiIndex DataFrame score, close: datetime x instrument"""
    ret = close.pct_change(fill_method=None)
    ret_a = ret.shift(-1).loc[days].iloc[:-1]
    d = ret_a.index
    top5, drop, soft = [], [], []
    for day in d:
        s = pred["score"].loc[day]
        r = ret_a.loc[day]
        common = s.index.intersection(r.dropna().index)
        if len(common) < 5:
            continue
        s, r = s.loc[common], r.loc[common]
        # top5 等权
        top5_inst = s.sort_values(ascending=False).head(5).index
        top5.append(r.loc[top5_inst].mean())
        # top5-dropout 模拟（n_drop=1, 确定性）
        drop.append(np.nan)
        # 平滑加权
        wts = s.clip(lower=0)
        if wts.sum() <= 0:
            wts = pd.Series(1.0, index=s.index)
        wts = wts / wts.sum()
        soft.append((r * wts).sum())
    # top5-dropout 需要状态，单独跑
    top5 = pd.Series(top5, index=d[: len(top5)])
    soft = pd.Series(soft, index=d[: len(soft)])
    return {"top5等权": annualized(top5), "平滑加权": annualized(soft)}


def topk_dropout_ret(pred, close, days, topk=5, n_drop=1):
    """复刻 TopkDropoutStrategy(method_buy=top, method_sell=bottom) 无成本版"""
    ret = close.pct_change(fill_method=None)
    ret_a = ret.shift(-1).loc[days].iloc[:-1]
    held = set()
    r = []
    used = []
    for day in ret_a.index:
        s = pred["score"].loc[day]
        rr = ret_a.loc[day]
        common = s.index.intersection(rr.dropna().index)
        if len(common) < 5:
            continue
        s, rr = s.loc[common], rr.loc[common]
        held = held & set(s.index)  # 剔除当日无信号的持仓
        if held:
            scored = s.loc[list(held)].sort_values()
            sell = set(scored.head(n_drop).index)
            held -= sell
        candidates = s.drop(index=[h for h in held if h in s.index]).sort_values(ascending=False)
        need = topk - len(held)
        if need > 0 and len(candidates):
            held |= set(candidates.head(need).index)
        vals = [rr[h] for h in held if h in rr.index]
        if vals:
            r.append(float(np.mean(vals)))
            used.append(day)
    return pd.Series(r, index=used)


def main():
    import qlib

    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(os.path.abspath(MLRUNS))
    qlib.init(provider_uri=DATA_URI, region="cn", exp_manager=exp_manager)

    rows = {}
    all_pred = {}
    for s in SEEDS:
        rec, task_config, dataset = build_dataset(s)
        model = rec.load_object("params.pkl")

        # --- 推理验证段 ---
        pred = model.predict(dataset, segment="valid")
        if isinstance(pred, pd.Series):
            pred = pred.to_frame("score")
        all_pred[s] = pred
        print(f"[{s}] valid pred: shape={pred.shape}, {pred.index.get_level_values(0).min()} ~ "
              f"{pred.index.get_level_values(0).max()}")

        # --- label（learn 空间，ZScoreNorm 后；Pearson IC 对全局线性变换不变）---
        from qlib.data.dataset.handler import DataHandlerLP

        df = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        label = df["label"]
        print(f"    label 列: {label.columns.tolist()}, 行数={len(label)}")

        # --- 指标 ---
        best_step = int(metric_curve(s, "best_step").iloc[-1])
        val_loss = float(metric_curve(s, "val_loss").loc[best_step])
        min_loss = float(metric_curve(s, "val_loss").min())
        val_ic_ts = float(metric_curve(s, "val_metric").loc[best_step])  # 时序IC（ICLoss口径）
        ic_daily = daily_ic(pred, label)
        val_ic_cross = float(ic_daily.mean())  # 日截面IC均值

        rows[s] = {
            "loss@best": val_loss,
            "min_loss": min_loss,
            "IC时序@best": val_ic_ts,
            "IC日截面均值": val_ic_cross,
            "日IC天数": len(ic_daily),
        }
        print(f"    {rows[s]}")

    # --- 收益率：close + 组合模拟 ---
    insts = sorted(set().union(*[set(p.index.get_level_values(1)) for p in all_pred.values()]))
    close = D.features(insts, ["$close"], start_time=VALID_START, end_time=VALID_END, freq="day")
    close = close["$close"].unstack().T
    for s in SEEDS:
        pred = all_pred[s]
        days = pred.index.get_level_values(0).unique()
        cr = combo_rets(pred, close, days)
        drop = topk_dropout_ret(pred, close, days)
        cr["top5-dropout模拟"] = annualized(drop)
        rows[s].update({f"收益_{k}": v for k, v in cr.items()})
        print(f"[{s}] 验证段组合年化(无成本): { {k: round(v, 4) for k, v in cr.items()} }")

    df = pd.DataFrame(rows).T
    print("\n" + "=" * 84)
    print("验证集指标汇总（3 seed × 指标）")
    print("=" * 84)
    print(df.round(4).to_string())

    # --- 相关性 ---
    print("\n" + "=" * 84)
    print("两两相关（n=3，仅示意，无统计意义）: Pearson / Spearman")
    print("=" * 84)
    cols = ["loss@best", "min_loss", "IC时序@best", "IC日截面均值",
            "收益_top5等权", "收益_top5-dropout模拟", "收益_平滑加权"]
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = df[cols[i]].astype(float)
            b = df[cols[j]].astype(float)
            p = np.corrcoef(a, b)[0, 1]
            ra, rb = a.rank(), b.rank()
            sp = np.corrcoef(ra, rb)[0, 1]
            print(f"  {cols[i]:<16} vs {cols[j]:<16}:  pearson={p:+.3f}  spearman={sp:+.3f}")

    # --- 核心三件套对照（Loss / IC / 收益）---
    print("\n" + "=" * 84)
    print("核心对照：Loss | IC | 收益率（含测试段收益做参照）")
    print("=" * 84)
    test_ann = {}
    for s in SEEDS:
        rec = load_recorder(s)
        rep = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
        test_ann[s] = (1.0 + rep["return"].mean()) ** 252 - 1.0
    core = pd.DataFrame({
        "val_loss@best": df["loss@best"],
        "val_IC时序": df["IC时序@best"],
        "val_IC日截面": df["IC日截面均值"],
        "val收益_top5dropout": df["收益_top5-dropout模拟"],
        "test收益(含成本)": test_ann,
    })
    print(core.round(4).to_string())


if __name__ == "__main__":
    main()
