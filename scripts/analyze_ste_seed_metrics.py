# -*- coding: utf-8 -*-
"""
统计 3 个 seed 的 STE v2（softmax 训练 + hard 评测）指标关系：
  train_loss   : mlflow train_loss（softmax 组合收益负值，每步记录）
  val_loss     : mlflow val_loss@best_step（hard top5 组合收益负值）
  val_收益     : mlflow val_metric@best_step（hard top5 组合日收益均值，正值）
  test_loss    : 用 best 模型在 test 段推理，hard top5 组合收益负值（label 用 learn 空间）
  test_收益(hard) : test 段 hard top5 等权日收益均值（无成本）
  test_收益(回测) : report 含成本年化收益
输出：指标表 + 两两 Pearson/Spearman（n=3）
"""
import os
import pickle

import numpy as np
import pandas as pd

import qlib
from qlib.config import C
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP

MLRUNS = r"D:/Pengxin/CodeBase/Quant/qlib/mlruns"
CACHE_DIR = r"D:/Pengxin/CodeBase/Quant/QuantDataBank/all_weather_data/handler_cache"

RUNS = {
    "seed0": {"exp_name": "mlp_all_weather_alpha158_globalnorm_ste", "run": "64bfd3e7161744768e9cbae5c676152f"},
    "seed1": {"exp_name": "mlp_all_weather_alpha158_globalnorm_ste_seed1", "run": "473cb39d8996481d88c6ea1ab913adbd"},
    "seed2": {"exp_name": "mlp_all_weather_alpha158_globalnorm_ste_seed2", "run": "0ce672fa861a4d1d89d55a7ef73dba34"},
}
SEEDS = list(RUNS.keys())


def metric_curve(seed, name):
    from qlib.workflow import R

    exp = R.get_exp(experiment_name=RUNS[seed]["exp_name"])
    path = f"{MLRUNS}/{exp.id}/{RUNS[seed]['run']}/metrics/{name}"
    df = pd.read_csv(path, sep=" ", header=None, names=["ts", "value", "step"])
    return df.set_index("step")["value"]


def load_recorder(seed):
    from qlib.workflow import R

    exp = R.get_exp(experiment_name=RUNS[seed]["exp_name"])
    return exp.get_recorder(recorder_id=RUNS[seed]["run"])


def build_dataset(seed):
    rec = load_recorder(seed)
    from qlib.workflow.task.utils import replace_task_handler_with_cache

    task_config = rec.load_object("task")
    task_config = replace_task_handler_with_cache(task_config, cache_dir=CACHE_DIR)
    dataset = init_instance_by_config(task_config["dataset"])
    return rec, dataset


def hard_topk_daily_ret(pred, label, topk=5):
    """hard top5 等权日收益均值：pred/label 均 MultiIndex (datetime, instrument)"""
    p = pred["score"].unstack()
    l = label.iloc[:, 0].unstack()
    idx = p.index.intersection(l.index)
    rets = []
    for d in idx:
        a = p.loc[d].dropna()
        b = l.loc[d].dropna()
        common = a.index.intersection(b.index)
        if len(common) < topk:
            continue
        top = a.loc[common].sort_values(ascending=False).head(topk).index
        rets.append(b.loc[top].mean())
    return pd.Series(rets, index=idx)


def annualized(ret):
    return (1.0 + ret.mean()) ** 252 - 1.0 if len(ret) else np.nan


def main():
    import qlib
    from qlib.workflow import R

    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(os.path.abspath(MLRUNS))
    qlib.init(provider_uri=r"D:/Pengxin/CodeBase/Quant/QuantDataBank/all_weather_data/qlib_all_weather",
              region="cn", exp_manager=exp_manager)

    rows = {}
    for s in SEEDS:
        rec, dataset = build_dataset(s)
        model = rec.load_object("params.pkl")
        best = int(metric_curve(s, "best_step").iloc[-1])

        tl = metric_curve(s, "train_loss")
        vl = metric_curve(s, "val_loss").loc[best]
        vm = metric_curve(s, "val_metric").loc[best]
        train_loss_best = tl.loc[best] if best in tl.index else np.nan
        train_loss_avg = tl.mean()
        train_loss_last = tl.iloc[-1]

        # test 段推理（learn 空间 label 也在，做 hard top5 组合收益）
        pred_test = model.predict(dataset, segment="test")
        if isinstance(pred_test, pd.Series):
            pred_test = pred_test.to_frame("score")
        df_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        label_test = df_test["label"]
        hard_test = hard_topk_daily_ret(pred_test, label_test)
        test_loss = -hard_test.mean()          # hard top5 组合收益负值 = test loss
        test_ret_hard = hard_test.mean()       # 无成本日收益均值
        test_ann_hard = annualized(hard_test)

        # 回测含成本年化（report）
        rep = rec.load_object("portfolio_analysis/report_normal_1day.pkl")
        test_ann_bt = (1.0 + rep["return"].mean()) ** 252 - 1.0

        # 验证段 hard top5 无成本年化（对比用）
        df_v = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        pred_v = model.predict(dataset, segment="valid")
        if isinstance(pred_v, pd.Series):
            pred_v = pred_v.to_frame("score")
        val_hard = hard_topk_daily_ret(pred_v, df_v["label"])
        val_ann_hard = annualized(val_hard)

        rows[s] = {
            "best_step": best,
            "train_loss@best": train_loss_best,
            "train_loss_avg": train_loss_avg,
            "val_loss@best": vl,
            "val_收益日均值": vm,               # val_metric = hard top5 日收益均值
            "val_收益年化": val_ann_hard,
            "test_loss": test_loss,
            "test_收益日均值": test_ret_hard,
            "test_收益年化(hard无成本)": test_ann_hard,
            "test_收益年化(回测含成本)": test_ann_bt,
        }
        print(f"[{s}] best_step={best}  train_loss_avg={train_loss_avg:.4f}  "
              f"val_收益(日)={vm:.4f}  test_loss={test_loss:.4f}  "
              f"test_收益(回测年化)={test_ann_bt:.4f}")

    df = pd.DataFrame(rows).T
    print("\n" + "=" * 96)
    print("STE v2 三 seed 指标总表")
    print("=" * 96)
    print(df.round(4).to_string())

    print("\n" + "=" * 96)
    print("两两相关（n=3，仅示意）Pearson / Spearman")
    print("=" * 96)
    cols = ["train_loss@best", "train_loss_avg", "val_loss@best", "val_收益日均值",
            "test_loss", "test_收益年化(hard无成本)", "test_收益年化(回测含成本)"]
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = df[cols[i]].astype(float)
            b = df[cols[j]].astype(float)
            p = np.corrcoef(a, b)[0, 1]
            sp = np.corrcoef(a.rank(), b.rank())[0, 1]
            print(f"  {cols[i]:<22} vs {cols[j]:<24}:  pearson={p:+.3f}  spearman={sp:+.3f}")

    # 核心问题：val 收益能否预测 test 收益（训练目标是否有效迁移）
    print("\n" + "=" * 96)
    print("核心对照（val 训练指标 vs test 收益）")
    print("=" * 96)
    core = pd.DataFrame({
        "val_loss@best": df["val_loss@best"],
        "val_收益年化": df["val_收益年化"],
        "test_收益年化(hard)": df["test_收益年化(hard无成本)"],
        "test_收益年化(回测)": df["test_收益年化(回测含成本)"],
        "test_IC": None,  # 下面填
    })
    # 加 test IC（SigAnaRecord 产物）
    for s in SEEDS:
        ics = load_recorder(s).load_object("sig_analysis/ic.pkl")
        core.loc[s, "test_IC"] = ics.mean()
    print(core.round(4).to_string())
    for a, b in [("val_收益年化", "test_收益年化(回测)"), ("val_loss@best", "test_收益年化(回测)"),
                 ("val_收益年化", "test_IC"), ("test_收益年化(hard)", "test_收益年化(回测)")]:
        x = core[a].astype(float)
        y = core[b].astype(float)
        print(f"  {a} vs {b}: pearson={np.corrcoef(x, y)[0, 1]:+.3f}  "
              f"spearman={np.corrcoef(x.rank(), y.rank())[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
