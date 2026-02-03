# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys, site
from pathlib import Path

################################# NOTE #################################
#  Please be aware that if colab installs the latest numpy and pyqlib  #
#  in this cell, users should RESTART the runtime in order to run the  #
#  following cells successfully.                                       #
########################################################################

try:
    import qlib
except ImportError:
    # install qlib
    # ! pip install --upgrade numpy
    # ! pip install pyqlib
    # if "google.colab" in sys.modules:
    #     # The Google colab environment is a little outdated. We have to downgrade the pyyaml to make it compatible with other packages
    #     ! pip install pyyaml==5.4.1
    # # reload
    # site.main()
    raise ImportError("Please restart the runtime after installing qlib")

scripts_dir = Path.cwd().parent.joinpath("scripts")
if not scripts_dir.joinpath("get_data.py").exists():
    # download get_data.py script
    scripts_dir = Path("~/tmp/qlib_code/scripts").expanduser().resolve()
    scripts_dir.mkdir(parents=True, exist_ok=True)
    import requests

    with requests.get("https://raw.githubusercontent.com/microsoft/qlib/main/scripts/get_data.py", timeout=10) as resp:
        with open(scripts_dir.joinpath("get_data.py"), "wb") as fp:
            fp.write(resp.content)

import qlib
import pandas as pd
from qlib.constant import REG_CN
from qlib.utils import exists_qlib_data, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict

# use default data
# NOTE: need to download data from remote: python scripts/get_data.py qlib_data_cn --target_dir ~/.qlib/qlib_data/cn_data
provider_uri = "D:/Pengxin/CodeBase/Quant/QuantDataBank/qlib_bin"  # target_dir
if not exists_qlib_data(provider_uri):
    print(f"Qlib data is not found in {provider_uri}")
    raise FileNotFoundError(f"Please run 'python scripts/get_data.py qlib_data_cn --target_dir {provider_uri}' to download the data first.")
    # sys.path.append(str(scripts_dir))
    # from get_data import GetData

    # GetData().qlib_data(target_dir=provider_uri, region=REG_CN)
qlib.init(provider_uri=provider_uri, region=REG_CN)

market = "csi300"
benchmark = "SH000300"

###################################
# train model
###################################
data_handler_config = {
    "start_time": "2008-01-01",
    "end_time": "2020-08-01",
    "fit_start_time": "2008-01-01",
    "fit_end_time": "2014-12-31",
    "instruments": market,
}

task = {
    "model": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.8879,
            "learning_rate": 0.0421,
            "subsample": 0.8789,
            "lambda_l1": 205.6999,
            "lambda_l2": 580.9768,
            "max_depth": 8,
            "num_leaves": 210,
            "num_threads": 20,
        },
    },
    "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha158",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": data_handler_config,
            },
            "segments": {
                "train": ("2008-01-09", "2024-01-01"),
                "valid": ("2024-01-01", "2025-01-02"),
                "test": ("2025-01-01", "2026-02-02"),
            },
        },
    },
}

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
port_analysis_config = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
        "kwargs": {
            "model": model,
            "dataset": dataset,
            "topk": 50,
            "n_drop": 5,
        },
    },
    "backtest": {
        "start_time": "2017-01-01",
        "end_time": "2020-08-01",
        "account": 100000000,
        "benchmark": benchmark,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}

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
from qlib.contrib.report import analysis_model, analysis_position
from qlib.data import D

recorder = R.get_recorder(recorder_id="5fd8019d507c4b83876cb296f93b7fe4", experiment_id="937709646515528717")
print(recorder)
pred_df = recorder.load_object("pred.pkl")
report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

# analysis position
# report
analysis_position.report_graph(report_normal_df)

# risk analysis
analysis_position.risk_analysis_graph(analysis_df, report_normal_df)

# analysis model
label_df = dataset.prepare("test", col_set="label")
label_df.columns = ["label"]

# score IC
pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
analysis_position.score_ic_graph(pred_label)

# model performance
analysis_model.model_performance_graph(pred_label)
