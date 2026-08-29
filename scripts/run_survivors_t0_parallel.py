# -*- coding: utf-8 -*-
"""
t+0 close 成交口径修复后的两组实验 × 5 seed，多进程并行运行：

  - ste_t0      : loss=portfolio（soft 训练 + hard top5 评测），label = Ref($close,-1)/$close-1
  - ste_mse_t0  : loss=portfolio_mse（portfolio 项 + MSE 项加权混合，w=portfolio_weight=0.5），label 同上

两组回测均为 signal_shift=0 + deal_price=close（t 日收盘成交），与 label 口径一致：
signal[t]（特征至 close(t)）→ t 日收盘买入 → 赚 close(t+1)/close(t)-1。

统计 val_loss@best（TrainVal 早停段 hard top5 组合收益负值）与 AB 测试段收益/IC 相关性。
Windows 用 spawn 上下文；每个 worker 独立 qlib.init + load_config + task_train。
"""
import argparse
import copy
import datetime as dt
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(r"D:/Pengxin/CodeBase/Quant/qlib")
OUTPUT_DIR = BASE_DIR / "output"
MLRUNS = BASE_DIR / "mlruns"
DATA_BASE = "D:/Pengxin/CodeBase/Quant/QuantDataBank"

# key -> (yaml 相对路径, 实验名前缀)
YAMLS = {
    "ste_t0": "examples/benchmarks/MLP/workflow_config_mlp_all_weather_alpha158_globalnorm_ste_survivors_tclose.yaml",
    "ste_mse_t0": "examples/benchmarks/MLP/workflow_config_mlp_all_weather_alpha158_globalnorm_ste_mse_survivors_tclose.yaml",
    "mse_t0": "examples/benchmarks/MLP/workflow_config_mlp_all_weather_alpha158_globalnorm_mse_survivors_tclose.yaml",  # w=0 纯 MSE 消融
}
SEEDS = [0, 1, 2, 3, 4]
KEEP_RECORDS = ["SignalRecord", "SigAnaRecord", "PortAnaRecord"]


def ensure_data_base_env():
    """模板 {{ QLIB_DATA_BASE }} 依赖环境变量；未设置时注入默认值（正斜杠）。"""
    if "QLIB_DATA_BASE" not in os.environ:
        os.environ["QLIB_DATA_BASE"] = DATA_BASE.replace("\\", "/")


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


def run_one_inited(yaml_key, seed):
    """在已 qlib.init 的进程内跑一个 (yaml_key, seed)。返回结果 dict。"""
    exp_name = f"mlp_{yaml_key}_valrel_seed{seed}"
    print(f"\n{'='*70}\nRUN {exp_name} (yaml={YAMLS[yaml_key]})\n{'='*70}", flush=True)
    try:
        from qlib.model.trainer import task_train

        base_cfg = load_yaml(BASE_DIR / YAMLS[yaml_key])
        task = build_task(base_cfg, seed)
        recorder = task_train(task, experiment_name=exp_name)

        m = read_metrics(recorder)
        best_step = int(m["best_step"].iloc[-1]["value"]) if "best_step" in m and len(m["best_step"]) else None
        vl = float(m["val_loss"].set_index("step")["value"].loc[best_step]) if best_step is not None else float("nan")
        vm = float(m["val_metric"].set_index("step")["value"].loc[best_step]) if best_step is not None else float("nan")
        adf = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
        ann_bt = float(adf.loc[("return_with_cost", "annualized_return"), "risk"])
        ann_nc = float(adf.loc[("return_without_cost", "annualized_return"), "risk"])
        ic = float(recorder.load_object("sig_analysis/ic.pkl").mean())
        row = {
            "experiment": yaml_key, "seed": seed, "best_step": best_step,
            "val_loss": vl, "val_metric": vm, "test_ann_bt": ann_bt,
            "test_ann_nocost": ann_nc, "test_ic": ic, "status": "ok",
        }
        print(f"  {yaml_key} seed={seed}: best_step={best_step} val_loss={vl:.4f} val_metric={vm:.4f} "
              f"test_ann_bt={ann_bt:.4f} test_ann_nocost={ann_nc:.4f} test_ic={ic:.4f}", flush=True)
        return row
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return {"experiment": yaml_key, "seed": seed, "status": "failed", "error": str(e)}


def run_one(task_tuple):
    """worker 入口：init 后跑一个 (yaml_key, seed)。"""
    yaml_key, seed = task_tuple
    try:
        ensure_data_base_env()
        import qlib
        from qlib.config import C

        base_cfg = load_yaml(BASE_DIR / YAMLS[yaml_key])
        qlib_init = base_cfg.get("qlib_init", {})
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(MLRUNS.resolve())
        init_kwargs = {k: v for k, v in qlib_init.items() if k != "exp_manager"}
        init_kwargs["exp_manager"] = exp_manager
        qlib.init(**init_kwargs)
        return run_one_inited(yaml_key, seed)
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        return {"experiment": yaml_key, "seed": seed, "status": "failed", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="覆盖 seed 列表")
    parser.add_argument("--experiments", nargs="*", default=None, help="只跑指定实验 ste_t0/ste_mse_t0")
    parser.add_argument("--jobs", type=int, default=5, help="并行 worker 数（默认 5 = 5 seed 并行）")
    parser.add_argument("--serial", action="store_true", help="串行模式：单进程顺序跑，避免并发数据加载内存耗尽")
    parser.add_argument("--merge", type=str, default=None, help="合并已有结果 CSV（如 output/valrel_t0_*.csv），一起输出汇总")
    args = parser.parse_args()
    seeds = args.seeds or SEEDS
    exps = args.experiments or list(YAMLS.keys())

    ensure_data_base_env()
    tasks = [(ename, s) for ename in exps for s in seeds]
    print(f"{'串行' if args.serial else '并行'}运行 {len(tasks)} 个 run"
          f"{'' if args.serial else ', jobs=' + str(args.jobs)}\n", flush=True)
    for t in tasks:
        print(f"  mlp_{t[0]}_valrel_seed{t[1]}", flush=True)

    if args.serial:
        # 串行：单进程 init 一次，循环调用（复用 exp_manager，避免重复初始化）
        import qlib
        from qlib.config import C

        base_cfg = load_yaml(BASE_DIR / YAMLS[exps[0]])
        qlib_init = base_cfg.get("qlib_init", {})
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(MLRUNS.resolve())
        init_kwargs = {k: v for k, v in qlib_init.items() if k != "exp_manager"}
        init_kwargs["exp_manager"] = exp_manager
        qlib.init(**init_kwargs)
        results = [run_one_inited(ename, s) for ename, s in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.jobs) as pool:
            results = pool.map(run_one, tasks)

    df = pd.DataFrame(results)
    # 合并历史结果（同 experiment+seed 去重，保留最新）
    if args.merge:
        merge_path = Path(args.merge)
        if merge_path.exists():
            old = pd.read_csv(merge_path)
            df = pd.concat([old, df], ignore_index=True)
            df = df.drop_duplicates(subset=["experiment", "seed"], keep="last")
            print(f"已合并历史 CSV: {merge_path.resolve()}（旧 {len(old)} 行 → 总 {len(df)} 行）")
        else:
            print(f"警告：--merge 指定的文件不存在，跳过合并: {merge_path}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = OUTPUT_DIR / f"valrel_t0_{ts}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n{'='*70}\n汇总:\n{'='*70}")
    print(df.to_string(index=False))
    print(f"\n已保存: {out_csv}")

    print("\n" + "=" * 70)
    print("相关性统计（每个实验 n=5 seed；合并 CSV 后对全部实验）")
    print("=" * 70)
    for ename in sorted(df["experiment"].dropna().unique()):
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
