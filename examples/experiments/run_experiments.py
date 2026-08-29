#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用实验评测 runner（与具体方法解耦）。

职责：读取一份 runner 配置(experiments.yaml)，对其中每个 "完整 qrun 配置 × regime"
组合运行 qlib workflow，收集 PortAnaRecord 指标，汇总成对比表。

runner 不 import 任何模型类，只依赖两个 qlib 标准契约：
  1) 输入是 qrun 格式的 task 配置；
  2) 输出可从 PortAnaRecord 读取。
因此对 AllIn 及任何走 DatasetH + PortAnaRecord 的方法均通用。

用法：
    python examples/experiments/run_experiments.py  # 用同目录 experiments.yaml
    python examples/experiments/run_experiments.py --config path/to.yaml
    python examples/experiments/run_experiments.py --regimes A             # 只跑部分 regime
    python examples/experiments/run_experiments.py --configs a.yaml b.yaml # 临时指定配置路径(覆盖清单)
    
"""

import argparse
import copy
import datetime as dt
from pathlib import Path

import pandas as pd
import yaml

import qlib
from qlib.model.trainer import task_train


# ──────────────────────────────────────────────────────────────
# 配置加载与解析
# ──────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    # 复用 qlib 的 load_config：渲染 {{ ENV }} 模板 + 处理 BASE_CONFIG_PATH，
    # 与 qrun 行为完全一致（直接用 yaml.safe_load 会得到未渲染的模板字符串）
    try:
        from qlib.cli.run import load_config
        return load_config(str(path))
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def _resolve(base_dir: Path, p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (base_dir / q).resolve()


def load_runner_config(config_path: Path):
    cfg = _load_yaml(config_path)
    base_dir = config_path.parent
    configs = []
    for item in cfg["configs"]:
        configs.append({
            "alias": item["alias"],
            "path": _resolve(base_dir, item["path"]),
            # 可选：不重新训练，从 mlflow 加载已训练模型评测
            #   pretrained: {experiment: <exp_name>, run_id: <rid>}
            "pretrained": item.get("pretrained"),
        })
    regimes = cfg["regimes"]
    output_dir = _resolve(base_dir, cfg.get("output_dir", "results"))
    keep_records = cfg.get("keep_records")
    return configs, regimes, output_dir, keep_records


# ──────────────────────────────────────────────────────────────
# 单组合的 task 配置构建（套用 regime + 裁剪 records）
# ──────────────────────────────────────────────────────────────

def build_task(base_config: dict, regime: dict, keep_records):
    """基于完整 workflow_config，套用 regime(仅覆盖 test 段与回测窗口)，返回可训练的 task 子树。"""
    cfg = copy.deepcopy(base_config)
    task = cfg["task"]

    test_range = regime["test"]

    # 1) 覆盖 dataset 的 test 段（train/valid 沿用配置自带，用于 fit/选 K）
    task["dataset"]["kwargs"]["segments"]["test"] = list(test_range)

    # 2) 覆盖 PortAnaRecord 内嵌回测窗口（qrun 实际使用的是 record 里的 config）
    for rec in task.get("record", []):
        if rec.get("class") == "PortAnaRecord":
            bt = rec["kwargs"]["config"]["backtest"]
            bt["start_time"] = test_range[0]
            bt["end_time"] = test_range[1]

    # 3) 可选裁剪 records 提速
    if keep_records:
        task["record"] = [r for r in task.get("record", []) if r.get("class") in keep_records]

    # 4) 附带 qlib_init（pretrained 评测需 provider_uri/region 初始化数据）
    task["_qlib_init"] = cfg.get("qlib_init", {})

    return task, cfg.get("qlib_init", {})


# ──────────────────────────────────────────────────────────────
# 指标收集（模型无关：只读 PortAnaRecord / SignalRecord 产物）
# ──────────────────────────────────────────────────────────────

def _risk(adf: pd.DataFrame, category: str, metric: str):
    try:
        return float(adf.loc[(category, metric), "risk"])
    except Exception:
        return float("nan")


def collect_metrics(recorder) -> dict:
    out = {}
    try:
        adf = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
        out["ann_return_with_cost"] = _risk(adf, "return_with_cost", "annualized_return")
        out["excess_ann_with_cost"] = _risk(adf, "excess_return_with_cost", "annualized_return")
        out["information_ratio"] = _risk(adf, "return_with_cost", "information_ratio")
        out["max_drawdown"] = _risk(adf, "return_with_cost", "max_drawdown")
    except Exception as e:  # noqa: BLE001
        out["metrics_error"] = str(e)

    # 选中标的 / K（从 pred.pkl 的 score==1 推断）
    try:
        pred = recorder.load_object("pred.pkl")
        selected = (
            pred[pred["score"] == 1].index.get_level_values("instrument").unique().tolist()
        )
        out["top_k"] = len(selected)
        out["selected"] = ",".join(map(str, selected))
    except Exception:
        out["top_k"] = float("nan")
        out["selected"] = ""
    return out


# ──────────────────────────────────────────────────────────────
# 不重新训练：加载已训练模型，仅对新 test 段评测
# ──────────────────────────────────────────────────────────────

def evaluate_pretrained(pretrained: dict, task: dict, exp_name: str, uri_folder: str = "mlruns"):
    """
    从 mlflow 加载已训练模型（params.pkl），只对新 test 段做预测 + 回测，不训练。

    流程：
      1. qlib.init（同 qrun，file mlruns + 配置里的 provider_uri/region）
      2. 从 pretrained.experiment 加载 run 的模型 params.pkl
      3. 用 task 构建 dataset（handler 缓存命中则复用，新 test 段）
      4. 新建 recorder（exp_name），SignalRecord 预测 → PortAnaRecord 回测
      5. 返回 recorder（供 collect_metrics）

    pretrained: {"experiment": str, "run_id": str|None}
      - experiment: mlflow 实验名
      - run_id: 指定 run；None 时取该实验最新 FINISHED run
    """
    from pathlib import Path
    import os
    from qlib.workflow import R
    from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
    from qlib.utils import init_instance_by_config
    from qlib.config import C

    # 1) qlib.init：provider_uri/region 来自配置的 qlib_init（与 qrun 一致），
    #    否则会用默认 ~/.qlib/qlib_data/cn_data 导致数据不存在
    qlib_init_cfg = task.get("_qlib_init", {})
    init_kwargs = {k: v for k, v in qlib_init_cfg.items() if k != "exp_manager"}
    if "exp_manager" in qlib_init_cfg:
        init_kwargs["exp_manager"] = qlib_init_cfg["exp_manager"]
    else:
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)
        init_kwargs["exp_manager"] = exp_manager
    qlib.init(**init_kwargs)

    # 2) 加载已训练模型
    src_exp = R.get_exp(experiment_name=pretrained["experiment"])
    rid = pretrained.get("run_id")
    if rid:
        src_rec = src_exp.get_recorder(recorder_id=rid)
    else:
        recs = src_exp.list_recorders()
        finished = [r for r in recs.values() if r.info.get("status") == "FINISHED"]
        src_rec = finished[-1] if finished else list(recs.values())[-1]
    model = src_rec.load_object("params.pkl")
    print(f"  加载预训练模型: {pretrained['experiment']}/{src_rec.info['id']}")

    # 3) 构建 dataset（新 test 段）
    dataset = init_instance_by_config(task["dataset"])
    print(f"  dataset 构建完成: handler={type(dataset.handler).__name__}")

    # 4) 新建 recorder 并评测
    with R.start(experiment_name=exp_name):
        recorder = R.get_recorder()
        recorder.save_objects(**{"params.pkl": model, "dataset": dataset, "task": task})

        # SignalRecord：model.predict(dataset, "test") → pred.pkl
        SignalRecord(model=model, dataset=dataset, recorder=recorder).generate()

        # PortAnaRecord：回测（config 取自 task 的 record，含新 test 窗口）
        port_cfg = None
        for rec_cfg in task.get("record", []):
            if rec_cfg.get("class") == "PortAnaRecord":
                port_cfg = rec_cfg["kwargs"]["config"]
                break
        if port_cfg is None:
            raise ValueError("task 中无 PortAnaRecord 配置，无法回测")
        PortAnaRecord(recorder=recorder, config=port_cfg).generate()

    return recorder


# ──────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="通用实验评测 runner")
    here = Path(__file__).resolve().parent
    parser.add_argument("--config", default=str(here / "experiments.yaml"), help="runner 配置路径")
    parser.add_argument("--configs", nargs="*", default=None, help="临时指定 workflow_config 路径列表(覆盖清单)")
    parser.add_argument("--regimes", nargs="*", default=None, help="只跑指定 regime(如 A B)")
    args = parser.parse_args()

    runner_cfg_path = Path(args.config).resolve()
    configs, regimes, output_dir, keep_records = load_runner_config(runner_cfg_path)

    if args.configs:
        configs = [
            {"alias": Path(p).stem, "path": Path(p).resolve()} for p in args.configs
        ]
    if args.regimes:
        regimes = {k: v for k, v in regimes.items() if k in set(args.regimes)}

    run_dir = output_dir / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_dir = run_dir / "resolved"
    resolved_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for c in configs:
        base_config = _load_yaml(c["path"])
        for regime_name, regime in regimes.items():
            alias = c["alias"]
            exp_name = f"{alias}_{regime_name}"
            mode = "PRETRAINED" if c.get("pretrained") else "TRAIN"
            print(f"\n{'='*70}\nRUN: {exp_name}  [{mode}]  (config={c['path'].name}, test={regime['test']})\n{'='*70}")

            row = {"alias": alias, "regime": regime_name, "test_start": regime["test"][0],
                   "test_end": regime["test"][1], "mode": mode}
            try:
                task, qlib_init = build_task(base_config, regime, keep_records)
                # 落盘 resolved 配置便于追溯
                with open(resolved_dir / f"{exp_name}.yaml", "w", encoding="utf-8") as f:
                    yaml.safe_dump({"qlib_init": qlib_init, "task": task}, f, allow_unicode=True, sort_keys=False)

                if c.get("pretrained"):
                    # 不重新训练：加载已训练模型，仅对新 test 段评测
                    recorder = evaluate_pretrained(c["pretrained"], task, exp_name)
                else:
                    qlib.init(**qlib_init)
                    recorder = task_train(task, experiment_name=exp_name)
                row.update(collect_metrics(recorder))
                row["status"] = "ok"
            except Exception as e:  # noqa: BLE001
                row["status"] = "failed"
                row["error"] = str(e)
                # 用 ASCII 标记，避免 GBK 控制台无法打印 ✗ 导致 UnicodeEncodeError
                print(f"  [FAIL] {exp_name} failed: {e}")
            rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = run_dir / "summary.csv"
    md_path = run_dir / "summary.md"
    df.to_csv(csv_path, index=False)
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_string(index=False)
    with open(md_path, "w") as f:
        f.write(md)

    print(f"\n{'='*70}\n汇总结果:\n{'='*70}")
    print(df.to_string(index=False))
    print(f"\n已保存: {csv_path}\n         {md_path}")


if __name__ == "__main__":
    main()
