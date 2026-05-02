# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import fire
import pandas as pd
import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.contrib.report import analysis_model, analysis_position
import qlib.cli.run
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord


def draw_analysis_figures(recorder_id, experiment_id, dataset):
    # analyze graphs
    recorder = R.get_recorder(recorder_id=recorder_id, experiment_id=experiment_id)
    print(f"Recorder: {recorder}")
    print(f"Experiment ID: {recorder.experiment_id}")
    print(f"Recorder ID: {recorder.id}")

    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    # 使用recorder的artifact_uri获取正确路径
    artifact_uri = str(recorder.artifact_uri).replace('file://', '')
    save_root = os.path.join(artifact_uri, "analysis_csvs")
    os.makedirs(save_root, exist_ok=True)

    csv_dict = {
        "pred_df": pred_df,
        "report_normal_df": report_normal_df,
        # "positions": positions,
        "analysis_df": analysis_df,
    }
    for csv_type, csv_data in csv_dict.items():
        save_path = os.path.join(save_root, f"{csv_type}.csv")
        print(f"Save {csv_type} to local file: {save_path}")
        csv_data.to_csv(save_path, index=True)

    # analysis position
    # report
    fig_list_return = analysis_position.report_graph(report_normal_df, show_notebook=False)

    # risk analysis
    fig_list_risk = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)

    # analysis model
    label_df = dataset.prepare("test", col_set="label")
    label_df.columns = ["label"]

    # score IC
    pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
    fig_list_score_ic = analysis_position.score_ic_graph(pred_label, show_notebook=False)

    # model performance
    try:
        fig_list_model_performance = analysis_model.model_performance_graph(pred_label, show_notebook=False)
    except Exception as e:
        print(f"Model performance figure generation failed: {e}")
        fig_list_model_performance = []
    fig_dict = {
        "return": fig_list_return,
        "risk": fig_list_risk,
        "score_ic": fig_list_score_ic,
        "model_performance": fig_list_model_performance,
    }
    save_root = os.path.join(artifact_uri, "analysis_figures")
    os.makedirs(save_root, exist_ok=True)
    for fig_type, _fig_list in fig_dict.items():
        for idx, _fig in enumerate(_fig_list):
            # NOTE: displays figures: https://plotly.com/python/renderers/
            # default: plotly_mimetype+notebook
            # support renderers: import plotly.io as pio; print(pio.renderers)
            renderer = None
            try:
                # in notebook
                _ipykernel = str(type(get_ipython()))
                if "google.colab" in _ipykernel:
                    renderer = "colab"
                _fig.show(renderer=renderer)
            except NameError:
                save_path = os.path.join(save_root, f"{fig_type}_{idx}.png")
                print(f"Not in notebook, save the figure to local file: {save_path}")
                _fig.write_image(save_path)
                # py.plot(_fig, auto_open=True)


def backtest(recorder_id, experiment_id, dataset, port_analysis_config):
    # backtest and analysis
    with R.start(experiment_name="backtest_analysis"):
        recorder = R.get_recorder(recorder_id=recorder_id, experiment_id=experiment_id)
        model = recorder.load_object("params.pkl")

        # prediction
        recorder = R.get_recorder()
        new_recorder_id = recorder.id
        new_experiment_id = recorder.experiment_id
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # backtest & analysis
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
    return new_recorder_id, new_experiment_id


def backtest_analysis(config_path, recorder_id, experiment_id):
    config = qlib.cli.run.load_config(config_path)
    task = config.get("task", {})

    # model initialization
    qlib.init(**config.get("qlib_init"))
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    # start exp to train model
    # with R.start(experiment_name="train_model"):
    #     R.log_params(**flatten_dict(task))
    #     model.fit(dataset)
    #     R.save_objects(trained_model=model)
    #     rid = R.get_recorder().id

    recorder_id, experiment_id = backtest(
        recorder_id=recorder_id, 
        experiment_id=str(experiment_id), 
        dataset=dataset, 
        port_analysis_config=config.get("port_analysis_config", {}),
    )
    
    draw_analysis_figures(recorder_id=recorder_id, experiment_id=experiment_id, dataset=dataset)


def analyze_existing_results(config_path, recorder_id, experiment_id):
    """分析已有的实验结果（不重新运行backtest）"""
    config = qlib.cli.run.load_config(config_path)
    task = config.get("task", {})

    # 初始化qlib，指定正确的mlruns路径
    if "exp_manager" in config.get("qlib_init", {}):
        qlib.init(**config.get("qlib_init"))
    else:
        from qlib.config import C
        exp_manager = C["exp_manager"]
        # 使用实际的mlruns路径
        mlruns_uri = "file:///Users/zengpengxin/workspace/CodeBase/qlib/mlruns"
        exp_manager["kwargs"]["uri"] = mlruns_uri
        qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)
        print(f"MLflow tracking URI: {mlruns_uri}")

    # 初始化数据集（用于label对比）
    dataset = init_instance_by_config(task["dataset"])

    # 直接分析已有结果
    draw_analysis_figures(recorder_id=recorder_id, experiment_id=str(experiment_id), dataset=dataset)


def main():
    # 分析 HistRelaPB 实验结果
    analyze_existing_results(
        config_path="benchmarks/HistRelaPB/workflow_config.yaml",
        recorder_id="70a489f6ed1c49a79fb54371c07dc34e",
        experiment_id="615697135128701704",
    )


if __name__ == "__main__":
    # fire.Fire()
    main()
