# -*- coding: utf-8 -*-
"""
三个验证段选择实验 × 5 seed，统计早停段 portfolio loss 与测试段收益的相关性。

实验设计（段划分取自 examples/experiments/experiments.yaml）：
  train = [2012-05-28, 2018-04-18]   （三个实验相同）
  Val   = [2018-04-19, 2021-11-12]
  A     = [2021-11-15, 2024-05-23]
  B     = [2024-05-24, 2026-07-17]
  AB    = [2021-11-15, 2026-07-17]
  TrainVal = [2012-05-28, 2021-11-12]（train+valid 合并）

实验：
  E1: train 上训练，Val 早停    → val_loss(Val)    vs AB 区间收益
  E2: train 上训练，TrainVal 早停 → val_loss(TrainVal) vs AB 区间收益
  E3: train 上训练，Val+A 早停  → val_loss(Val+A)  vs B  区间收益

每个实验 seed = 0..4（5 个），训练用 loss=portfolio（softmax 训练 + hard top5 评测）。
val_loss = 早停段 hard top5 组合收益负值（mlflow val_loss@best_step）。
测试收益 = PortAnaRecord 真实回测（含成本）年化。
"""
import argparse
import copy
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import qlib
from qlib.config import C
from qlib.model.trainer import task_train
from qlib.utils import init_instance_by_config

BASE_YAML = Path(r"D:/Pengxin/CodeBase/Quant/qlib/examples/benchmarks/MLP/workflow_config_mlp_all_weather_alpha158_globalnorm_ste.yaml")
MLRUNS = r"D:/Pengxin/CodeBase/Quant/qlib/mlruns"
OUTPUT_DIR = Path(r"D:/Pengxin/CodeBase/Quant/qlib/output")

TRAIN = ["2012-05-28", "2018-04-18"]
VAL = ["2018-04-19", "2021-11-12"]
A = ["2021-11-15", "2024-05-23"]
B = ["2024-05-24", "2026-07-17"]
AB = ["2021-11-15", "2026-07-17"]
TRAINVAL = ["2012-05-28", "2021-11-12"]
VAL_A = ["2018-04-19", "2024-05-23"]

EXPERIMENTS = {
    "E1_Val_AB": {"valid": VAL, "test": AB},       # Val 早停 → AB
    "E2_TrainVal_AB": {"valid": TRAINVAL, "test": AB},  # TrainVal 早停 → AB
    "E3_ValA_B": {"valid": VAL_A, "test": B},      # Val+A 早停 → B
}
SEEDS = [0, 1, 2, 3, 4]
KEEP_RECORDS = ["SignalRecord", "SigAnaRecord", "PortAnaRecord"]


def load_base_task():
    from qlib.cli.run import load_config

    cfg = load_config(str(BASE_YAML))
    return cfg


def build_task(base_cfg, valid_range, test_range, seed):
    task = copy.deepcopy(base_cfg["task"])
    task["dataset"]["kwargs"]["segments"]["train"] = list(TRAIN)
    task["dataset"]["kwargs"]["segments"]["valid"] = list(valid_range)
    task["dataset"]["kwargs"]["segments"]["test"] = list(test_range)
    task["model"]["kwargs"]["seed"] = seed
    # 裁剪 record 提速（去掉图表导出）
    if KEEP_RECORDS:
        task["record"] = [r for r in task.get("record", []) if r.get("class") in KEEP_RECORDS]
    # 回测窗口 = test 段
    for rec in task.get("record", []):
        if rec.get("class") == "PortAnaRecord":
            bt = rec["kwargs"]["config"]["backtest"]
            bt["start_time"] = test_range[0]
            bt["end_time"] = test_range[1]
    task["_qlib_init"] = base_cfg.get("qlib_init", {})
    return task


def collect(recorder):
    """占位（实际在 main 中读取）"""
    return {}


def read_metrics(recorder):
    from qlib.workflow import R

    eid = recorder.info["experiment_id"]
    rid = recorder.info["id"]
    path = Path(MLRUNS) / str(eid) / rid / "metrics"
    out = {}
    for name in ["best_step", "val_loss", "val_metric"]:
        f = path / name
        if f.exists():
            df = pd.read_csv(f, sep=" ", header=None, names=["ts", "value", "step"])
            out[name] = df
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="覆盖 seed 列表")
    parser.add_argument("--experiments", nargs="*", default=None, help="只跑指定实验 E1/E2/E3")
    args = parser.parse_args()
    seeds = args.seeds or SEEDS
    exps = args.experiments or list(EXPERIMENTS.keys())

    base_cfg = load_base_task()
    qlib_init = base_cfg.get("qlib_init", {})
    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(Path(MLRUNS).resolve())
    init_kwargs = {k: v for k, v in qlib_init.items() if k != "exp_manager"}
    init_kwargs["exp_manager"] = exp_manager
    qlib.init(**init_kwargs)

    rows = []
    for ename in exps:
        valid_range, test_range = EXPERIMENTS[ename]["valid"], EXPERIMENTS[ename]["test"]
        for seed in seeds:
            exp_name = f"mlp_ste_valrel_{ename}_seed{seed}"
            print(f"\n{'='*70}\nRUN {exp_name}: valid={valid_range}, test={test_range}, seed={seed}\n{'='*70}", flush=True)
            try:
                task = build_task(base_cfg, valid_range, test_range, seed)
                recorder = task_train(task, experiment_name=exp_name)
                m = read_metrics(recorder)
                best_step = int(m["best_step"].iloc[-1]["value"]) if "best_step" in m and len(m["best_step"]) else None
                vl = float(m["val_loss"].set_index("step")["value"].loc[best_step]) if best_step is not None else float("nan")
                vm = float(m["val_metric"].set_index("step")["value"].loc[best_step]) if best_step is not None else float("nan")
                # test 回测指标
                adf = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
                ann_bt = float(adf.loc[("return_with_cost", "annualized_return"), "risk"])
                ann_nc = float(adf.loc[("return_without_cost", "annualized_return"), "risk"])
                ic = recorder.load_object("sig_analysis/ic.pkl").mean()
                rows.append({
                    "experiment": ename, "valid": str(valid_range), "test": str(test_range), "seed": seed,
                    "best_step": best_step, "val_loss": vl, "val_metric": vm,
                    "test_ann_bt": ann_bt, "test_ann_nocost": ann_nc, "test_ic": ic,
                    "status": "ok",
                })
                print(f"  best_step={best_step} val_loss={vl:.4f} val_metric={vm:.4f} "
                      f"test_ann_bt={ann_bt:.4f} test_ann_nocost={ann_nc:.4f} test_ic={ic:.4f}", flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                rows.append({"experiment": ename, "valid": str(valid_range), "test": str(test_range),
                             "seed": seed, "status": "failed", "error": str(e)})

    df = pd.DataFrame(rows)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = OUTPUT_DIR / f"valrel_{ts}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{'='*70}\n汇总:\n{'='*70}")
    print(df.to_string(index=False))
    print(f"\n已保存: {out_csv}")

    # 相关性统计
    print("\n" + "=" * 70)
    print("相关性统计（每个实验 n=5 seed）")
    print("=" * 70)
    for ename in exps:
        sub = df[(df["experiment"] == ename) & (df["status"] == "ok")]
        if len(sub) < 3:
            print(f"[{ename}] 有效 seed 数 <3，跳过")
            continue
        for xname, yname in [("val_loss", "test_ann_bt"), ("val_metric", "test_ann_bt"),
                             ("val_loss", "test_ic"), ("val_loss", "test_ann_nocost")]:
            x = sub[xname].astype(float)
            y = sub[yname].astype(float)
            if x.std() == 0 or y.std() == 0:
                print(f"[{ename}] {xname} vs {yname}: 常数序列，跳过")
                continue
            p = np.corrcoef(x, y)[0, 1]
            sp = np.corrcoef(x.rank(), y.rank())[0, 1]
            print(f"[{ename}] {xname:<10} vs {yname:<16}: pearson={p:+.3f}  spearman={sp:+.3f}")


if __name__ == "__main__":
    main()
