# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pandas as pd
import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.contrib.report import analysis_model, analysis_position
import qlib.cli.run


def backtest_analysis(config_path, recorder_id, experiment_id):
    config = qlib.cli.run.load_config(config_path)
    qlib.init(**config.get("qlib_init"))


    data_handler_config = config.get("data_handler_config", {})

    task = config.get("task", {})

    # model initialization
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    # start exp to train model
    # with R.start(experiment_name="train_model"):
    #     R.log_params(**flatten_dict(task))
    #     model.fit(dataset)
    #     R.save_objects(trained_model=model)
    #     rid = R.get_recorder().id

    ###################################
    # prediction, backtest & analysis
    ###################################
    port_analysis_config = config.get("port_analysis_config", {})

    # backtest and analysis
    # with R.start(experiment_name="backtest_analysis"):
    #     recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
    #     model = recorder.load_object("trained_model")

    #     # prediction
    #     recorder = R.get_recorder()
    #     ba_rid = recorder.id
    #     sr = SignalRecord(model, dataset, recorder)
    #     sr.generate()

    #     # backtest & analysis
    #     par = PortAnaRecord(recorder, port_analysis_config, "day")
    #     par.generate()

    # analyze graphs
    recorder = R.get_recorder(recorder_id=recorder_id, experiment_id=experiment_id)
    print(recorder)
    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

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
    fig_list_model_performance = analysis_model.model_performance_graph(pred_label, show_notebook=False)
    fig_dict = {
        "return": fig_list_return,
        "risk": fig_list_risk,
        "score_ic": fig_list_score_ic,
        "model_performance": fig_list_model_performance,
    }
    save_root = os.path.join(recorder._uri.replace('file:', ''), recorder.experiment_id, recorder.id, "analysis_figures")
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


def main():
    backtest_analysis(
        config_path="examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml", 
        recorder_id="c1b25a3c949c455f8f56ed5febddb1b8", 
        experiment_id="724594780217528450",
    )


if __name__ == "__main__":
    main()
