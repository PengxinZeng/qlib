# -*- coding: utf-8 -*-
"""
E2_TrainVal_AB（排除 Train 阶段未上市标的）两组实验 × 5 seed：
  - survivors      : t+1 close（signal_shift=1 默认）
  - survivors_tc   : t   close（signal_shift=0）
统计 val_loss@best（TrainVal 早停段 hard top5 组合收益负值）与 AB 测试段收益相关性。
"""
import argparse
import copy
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

import qlib
from qlib.config import C
from qlib.model.trainer import task_train

BASE_DIR = Path(r"D:/Pengxin/CodeBase/Quant/qlib")
OUTPUT_DIR = BASE_DIR / "output"
MLRUNS = BASE_DIR / "mlruns"

YAMLS = {
    "survivors": "examples/benchmarks/MLP/workflow_config_mlp_all_weather_alpha158_globalnorm_ste_survivors.yaml",
    "survivors_tclose": "examples/benchmarks/MLP/workflow_config_mlp_all_weather_alpha158_globalnorm_ste_survivors_tclose.yaml",
}
SEEDS = [0, 1, 2, 3, 4]
KEEP_RECORDS = ["SignalRecord", "SigAnaRecord", "PortAnaRecord"]


def load_yaml(path):
    from qlib.cli.run import load_config

    return load_config(str(path))


def build_task(base_cfg, seed):
    task = copy.deepcopy(base_cfg["task"])
    task["model"]["kwargs"]["seed"] = seed
    if KEEP_RECORDS:
        task["record"] = [r for r in task.get("record", []) if r.get("class") in KEEP_RECORDS]
    task["_qlib_init"] = base_cfg.get("qlib_init", {})
    return task


def read_metrics(recorder):
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
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--experiments", nargs="*", default=None)
    args = parser.parse_args()
    seeds = args.seeds or SEEDS
    exps = args.experiments or list(YAMLS.keys())

    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(MLRUNS.resolve())
    qlib_init = load_yaml(BASE_DIR / YAMLS[exps[0]]).get("qlib_init", {})
    init_kwargs = {k: v for k, v in qlib_init.items() if k != "exp_manager"}
    init_kwargs["exp_manager"] = exp_manager
    qlib.init(**init_kwargs)

    rows = []
    for ename in exps:
        base_cfg = load_yaml(BASE_DIR / YAMLS[ename])
        for seed in seeds:
            exp_name = f"mlp_ste_valrel_{ename}_seed{seed}"
            print(f"\n{'='*70}\nRUN {exp_name}\n{'='*70}", flush=True)
            try:
                task = build_task(base_cfg, seed)
                recorder = task_train(task, experiment_name=exp_name)
                m = read_metrics(recorder)
                best_step = int(m["best_step"].iloc[-1]["value"]) if "best_step" in m and len(m["best_step"]) else None
                vl = float(m["val_loss"].set_index("step")["value"].loc[best_step]) if best_step is not None else float("nan")
                vm = float(m["val_metric"].set_index("step")["value"].loc[best_step]) if best_step is not None else float("nan")
                adf = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
                ann_bt = float(adf.loc[("return_with_cost", "annualized_return"), "risk"])
                ann_nc = float(adf.loc[("return_without_cost", "annualized_return"), "risk"])
                ic = recorder.load_object("sig_analysis/ic.pkl").mean()
                rows.append({"experiment": ename, "seed": seed, "best_step": best_step,
                             "val_loss": vl, "val_metric": vm, "test_ann_bt": ann_bt,
                             "test_ann_nocost": ann_nc, "test_ic": ic, "status": "ok"})
                print(f"  best_step={best_step} val_loss={vl:.4f} val_metric={vm:.4f} "
                      f"test_ann_bt={ann_bt:.4f} test_ann_nocost={ann_nc:.4f} test_ic={ic:.4f}", flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback

                traceback.print_exc()
                rows.append({"experiment": ename, "seed": seed, "status": "failed", "error": str(e)})

    df = pd.DataFrame(rows)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = OUTPUT_DIR / f"valrel_survivors_{ts}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{'='*70}\n汇总:\n{'='*70}")
    print(df.to_string(index=False))
    print(f"\n已保存: {out_csv}")

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
