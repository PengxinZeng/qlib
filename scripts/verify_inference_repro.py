"""
验证：用训练好的模型参数重新推理评测，与第一次训练直接推理评测是否一致。

流程
----
1. 从 mlflow 加载指定 run 的已训练模型（params.pkl 内含 DNNModelPytorch 权重）
2. 用该模型对 test 段重新推理 → pred_re
3. 加载第一次训练时保存的 pred.pkl（SignalRecord 产物）→ pred_orig
4. 对比 pred_re vs pred_orig（最大绝对差、相关系数、逐样本是否一致）
5. 用 pred_re 重新计算 IC / Rank IC（与 SigAnaRecord 相同逻辑），与 mlflow 记录的 IC 对比
6. 用 pred_re 重新跑回测（PortAnaRecord 相同逻辑），与 mlflow 记录的回测指标对比

用法
----
    python scripts/verify_inference_repro.py \
        --experiment mlp_all_weather_alpha158_zscore_seed1 \
        --run-id 0442c539bdc54a4da8793374a34e7b20 \
        [--data-base D:/Pengxin/CodeBase/Quant/QuantDataBank/all_weather_data/qlib_all_weather]
"""
import argparse
import json
import os
import sys
from pathlib import Path

# 线程数限制（与 run_qrun_cached 一致）
for _var in ("NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "4")

import numpy as np
import pandas as pd

import qlib
from qlib.config import C
from qlib.workflow import R
from qlib.workflow.record_temp import SigAnaRecord, PortAnaRecord, SignalRecord
from qlib.utils import init_instance_by_config
from qlib.data.dataset.handler import DataHandlerLP


def load_recorder(args):
    """从 mlflow 加载 recorder"""
    exp = R.get_exp(experiment_name=args.experiment)
    rec = exp.get_recorder(recorder_id=args.run_id)
    return rec


def main():
    parser = argparse.ArgumentParser(description="验证模型重新推理评测与第一次是否一致")
    parser.add_argument("--experiment", required=True, help="mlflow 实验名")
    parser.add_argument("--run-id", required=True, help="mlflow run id（recorder id）")
    parser.add_argument("--data-base", default=None, help="provider_uri；默认用 QLIB_DATA_BASE 环境变量")
    args = parser.parse_args()

    provider_uri = args.data_base or os.environ.get(
        "QLIB_DATA_BASE", "D:/Pengxin/CodeBase/Quant/QuantDataBank"
    ) + "/all_weather_data/qlib_all_weather"

    # qlib.init（file mlruns）
    exp_manager = C["exp_manager"]
    exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / "mlruns")
    qlib.init(provider_uri=provider_uri, region="cn", exp_manager=exp_manager)

    rec = load_recorder(args)
    print(f"== 加载 recorder: {args.experiment}/{args.run_id} ==")

    # 1) 加载已训练模型（params.pkl 由 task_train 保存，含完整权重）
    model = rec.load_object("params.pkl")
    print(f"== 模型: {type(model).__name__}, fitted={model.fitted} ==")

    # 2) 重建 dataset：从 task 配置构建，handler 复用缓存 pickle（dump_all=True，含 _infer/_learn）
    #    不直接加载 recorder 里的 dataset —— 它默认 dump_all=False，下划线属性（_infer/_learn）未保存
    from qlib.workflow.task.utils import replace_task_handler_with_cache

    task_config = rec.load_object("task")
    cache_dir = Path(os.environ.get("QLIB_DATA_BASE", "D:/Pengxin/CodeBase/Quant/QuantDataBank")) / "all_weather_data" / "handler_cache"
    task_config = replace_task_handler_with_cache(task_config, cache_dir=cache_dir)
    dataset = init_instance_by_config(task_config["dataset"])
    print(f"== dataset 重建完成: {type(dataset).__name__}, handler={type(dataset.handler).__name__} ==")

    # 3) 用训练好的模型重新推理 test 段
    pred_re = model.predict(dataset, segment="test")
    if isinstance(pred_re, pd.Series):
        pred_re = pred_re.to_frame("score")
    print(f"== 重新推理 pred_re: shape={pred_re.shape} ==")
    print(pred_re.head(5))

    # 4) 加载第一次训练时的 pred.pkl
    pred_orig = rec.load_object("pred.pkl")
    print(f"== 原始 pred.pkl: shape={pred_orig.shape} ==")
    print(pred_orig.head(5))

    # 5) 对比 pred_re vs pred_orig
    idx_common = pred_re.index.intersection(pred_orig.index)
    re_v = pred_re.loc[idx_common, "score"].values
    or_v = pred_orig.loc[idx_common, "score"].values
    max_abs_diff = float(np.max(np.abs(re_v - or_v)))
    corr = float(np.corrcoef(re_v, or_v)[0, 1])
    n_same = int(np.sum(np.isclose(re_v, or_v, rtol=1e-10, atol=1e-10)))
    print(f"\n==== 推理一致性对比（公共样本 {len(idx_common)} 条）====")
    print(f"最大绝对差     : {max_abs_diff:.3e}")
    print(f"相关系数       : {corr:.8f}")
    print(f"逐样本完全一致 : {n_same}/{len(idx_common)} ({100.0 * n_same / len(idx_common):.4f}%)")
    print(f"pred_re 均值/标准差: {re_v.mean():.6f} / {re_v.std():.6f}")
    print(f"pred_orig 均值/标准差: {or_v.mean():.6f} / {or_v.std():.6f}")

    # 6) 重新计算 IC（与 SigAnaRecord 相同：基于 pred vs label）
    from qlib.contrib.eva.alpha import calc_ic

    label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_R)
    ic_re, ric_re = calc_ic(pred_re["score"], label["LABEL0"])
    print(f"\n==== 重新计算 IC（pred_re vs label）====")
    print(ic_re.describe() if hasattr(ic_re, "describe") else ic_re)

    # 7) 与 mlflow 记录的 IC 对比
    m = rec.list_metrics()
    print(f"\n==== mlflow 记录的指标（第一次）====")
    for k in ("IC", "ICIR", "Rank IC", "Rank ICIR"):
        print(f"  {k}: {m.get(k)}")
    print(f"  重新计算 IC    (mean): {ic_re.mean():.6f}")
    print(f"  重新计算 RankIC(mean): {ric_re.mean():.6f}")

    # 8) 重新跑回测（PortAnaRecord 相同配置与计算逻辑）
    record_cfg = task_config.get("record", [])
    port_cfg = None
    for r in record_cfg:
        if isinstance(r, dict) and r.get("class") == "PortAnaRecord":
            port_cfg = r.get("kwargs", {}).get("config")
    if port_cfg is not None:
        from qlib.backtest import backtest as normal_backtest
        from qlib.contrib.evaluate import risk_analysis
        from qlib.contrib.strategy import TopkDropoutStrategy

        print(f"\n==== 重新回测（pred_re）====")
        port_cfg = json.loads(json.dumps(port_cfg, default=str))  # deep copy
        # 替换 signal 为重新推理的 pred（与 PortAnaRecord.fill_placeholder("<PRED>") 等价）
        port_cfg["strategy"]["kwargs"]["signal"] = pred_re
        strat = TopkDropoutStrategy(**port_cfg["strategy"]["kwargs"])
        executor_cfg = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        }
        bt_config = port_cfg["backtest"]
        portfolio_metric_dict, indicator_dict = normal_backtest(
            executor=executor_cfg,
            strategy=strat,
            start_time=bt_config["start_time"],
            end_time=bt_config["end_time"],
            account=bt_config.get("account", 1000000),
            benchmark=bt_config.get("benchmark"),
            exchange_kwargs=bt_config.get("exchange_kwargs", {}),
        )
        report_normal, positions = portfolio_metric_dict["1day"]

        # 与 PortAnaRecord._generate 完全相同的指标计算
        analysis = dict()
        analysis["excess_return_without_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"], freq="1day"
        )
        analysis["excess_return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq="1day"
        )
        analysis["return_with_cost"] = risk_analysis(
            report_normal["return"] - report_normal["cost"], freq="1day", mode="product"
        )
        analysis["return_without_cost"] = risk_analysis(
            report_normal["return"], freq="1day", mode="product"
        )

        print("重新回测（pred_re）:")
        for key in ["return_with_cost", "return_without_cost", "excess_return_with_cost", "excess_return_without_cost"]:
            a = analysis[key]
            print(f"  {key}: annualized_return={a.loc['annualized_return','risk']:.6f}, "
                  f"max_drawdown={a.loc['max_drawdown','risk']:.6f}, IR={a.loc['information_ratio','risk']:.4f}")

        print("\n==== 第一次回测（mlflow 记录）====")
        for key in ["1day.return_with_cost.annualized_return", "1day.return_with_cost.max_drawdown",
                    "1day.return_with_cost.information_ratio",
                    "1day.excess_return_with_cost.annualized_return", "1day.excess_return_with_cost.information_ratio"]:
            print(f"  {key}: {m.get(key)}")
    else:
        print("\n(task 中未找到 PortAnaRecord 配置，跳过回测对比)")

    print("\n==== 验证完成 ====")


if __name__ == "__main__":
    main()
