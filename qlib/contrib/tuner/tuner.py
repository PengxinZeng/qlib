# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: skip-file
# flake8: noqa

import os
import yaml
import json
import copy
import pickle
import logging
import datetime
import importlib
import subprocess
import pandas as pd
import numpy as np

from abc import abstractmethod

from ...log import get_module_logger, TimeInspector
from ...utils.pickle_utils import restricted_pickle_load
from hyperopt import fmin, tpe, Trials
from hyperopt import STATUS_OK, STATUS_FAIL


class Tuner:
    TRIALS_NAME = "trials.pkl"

    def __init__(self, tuner_config, optim_config):
        self.logger = get_module_logger("Tuner", level=logging.INFO)

        self.tuner_config = tuner_config
        self.optim_config = optim_config

        self.max_evals = self.tuner_config.get("max_evals", 10)
        self.ex_dir = os.path.join(
            self.tuner_config["experiment"]["dir"],
            self.tuner_config["experiment"]["name"],
        )

        self.best_params = None
        self.best_res = None

        self.space = self.setup_space()

    def _load_trials(self):
        """加载已有的 Trials，同步加载 search_history.csv；若不存在则初始化为空。
        一致性校验：两者行数必须相同，否则 raise RuntimeError。
        """
        trials_path = os.path.join(self.ex_dir, self.TRIALS_NAME)
        hist_path = os.path.join(self.ex_dir, self.LOCAL_HIST_NAME) if hasattr(self, "LOCAL_HIST_NAME") else None

        trials_exists = os.path.exists(trials_path)
        hist_exists = hist_path is not None and os.path.exists(hist_path)

        if trials_exists:
            with open(trials_path, "rb") as fp:
                trials = pickle.load(fp)
            n_trials = len(trials.trials)
            self.logger.info("Resuming from existing Trials: {} evals done, path: {}".format(n_trials, trials_path))
        else:
            trials = Trials()
            n_trials = 0
            self.logger.info("Starting fresh Trials.")

        if hist_exists:
            hist_df = pd.read_csv(hist_path)
            n_hist = len(hist_df)
            self.logger.info("Loaded search_history.csv: {} rows, path: {}".format(n_hist, hist_path))
        else:
            hist_df = pd.DataFrame()
            n_hist = 0

        # 一致性校验
        if n_trials != n_hist:
            raise RuntimeError(
                "Trials ({} evals) and search_history.csv ({} rows) are inconsistent. "
                "Please delete one or both to restart, or fix the mismatch manually.\n"
                "  trials: {}\n  history: {}".format(n_trials, n_hist, trials_path, hist_path)
            )

        if hasattr(self, "LOCAL_HIST_NAME"):
            self._hist_df = hist_df

        return trials

    def _save_trials(self, trials):
        """持久化 Trials 到磁盘，供下次接续使用。"""
        os.makedirs(self.ex_dir, exist_ok=True)
        trials_path = os.path.join(self.ex_dir, self.TRIALS_NAME)
        with open(trials_path, "wb") as fp:
            pickle.dump(trials, fp)
        self.logger.info("Trials saved to: {}".format(trials_path))

    def _inject_to_trials(self, trials, params, result):
        """将手动运行的参数和结果注入 Trials，供 TPE 作为先验观测（默认空实现）。"""
        pass

    def tune(self):
        TimeInspector.set_time_mark()
        trials = self._load_trials()
        n_done = len(trials.trials)

        # 运行人工指定的初始参数（仅在全新开始时执行，续跑时跳过）
        initial_params_list = self.tuner_config.get("initial_params", [])
        if initial_params_list and n_done == 0:
            self.logger.info(
                "Running {} manually specified initial params before TPE...".format(len(initial_params_list))
            )
            for idx, init_params in enumerate(initial_params_list):
                self.logger.info(
                    "[{}/{}] Initial param: {}".format(idx + 1, len(initial_params_list), init_params)
                )
                result = self.objective(init_params)
                self._inject_to_trials(trials, init_params, result)
            # 立即持久化，保证 Trials 与 search_history.csv 一致
            self._save_trials(trials)
            n_done = len(trials.trials)

        total_evals = n_done + self.max_evals
        fmin(
            fn=self.objective,
            space=self.space,
            algo=tpe.suggest,
            max_evals=total_evals,
            trials=trials,
            show_progressbar=False,
        )
        self._save_trials(trials)
        self.logger.info("Local best params: {} ".format(self.best_params))
        TimeInspector.log_cost_time(
            "Finished searching best parameters in Tuner {}.".format(self.tuner_config["experiment"]["id"])
        )

        self.save_local_best_params()

    @abstractmethod
    def objective(self, params):
        """
        Implement this method to give an optimization factor using parameters in space.
        :return: {'loss': a factor for optimization, float type,
                  'status': the status of this evaluation step, STATUS_OK or STATUS_FAIL}.
        """
        pass

    @abstractmethod
    def setup_space(self):
        """
        Implement this method to setup the searching space of tuner.
        :return: searching space, dict type.
        """
        pass

    @abstractmethod
    def save_local_best_params(self):
        """
        Implement this method to save the best parameters of this tuner.
        """
        pass


class WorkflowConfigTuner(Tuner):
    WORKFLOW_CONFIG_NAME = "workflow_config_{}.yaml"
    LOCAL_BEST_PARAMS_NAME = "local_best_params.json"
    LOCAL_HIST_NAME = "search_history.csv"

    def _inject_to_trials(self, trials, params, result):
        """
        将手动运行的参数和结果注入 Trials，供 TPE 作为先验观测。
        通过递归遍历 pyll 表达式树，提取 hp 内部参数名与用户参数 key 的映射。
        """
        from hyperopt.base import JOB_STATE_DONE
        from hyperopt.pyll.base import Apply

        # 递归提取 hp 参数名 → (group_name, user_key)
        hp_mapping = {}

        def find_hp_nodes(node, group_name, user_key):
            if not isinstance(node, Apply):
                return
            if node.name == 'hyperopt_param':
                hp_name = node.pos_args[0]._obj
                hp_mapping[hp_name] = (group_name, user_key)
                return
            for arg in node.pos_args:
                find_hp_nodes(arg, group_name, user_key)
            for _, v in node.named_args:
                find_hp_nodes(v, group_name, user_key)

        for group_name, group_space in self.space.items():
            if isinstance(group_space, dict):
                for user_key, val in group_space.items():
                    if isinstance(val, Apply):
                        find_hp_nodes(val, group_name, user_key)

        if not hp_mapping:
            return  # 无 hp 参数（全固定值），跳过注入

        # 构建 idxs / vals
        tid = len(trials.trials)
        idxs, vals = {}, {}
        for hp_name, (group_name, user_key) in hp_mapping.items():
            group_params = params.get(group_name, {})
            if isinstance(group_params, dict) and user_key in group_params:
                idxs[hp_name] = [tid]
                vals[hp_name] = [float(group_params[user_key])]

        trial_doc = {
            'tid': tid,
            'state': JOB_STATE_DONE,
            'result': result,
            'misc': {
                'tid': tid,
                'cmd': ('domain_attachment', 'FMinIter_Domain'),
                'workdir': None,
                'idxs': idxs,
                'vals': vals,
            },
            'spec': None,
            'owner': None,
            'book_time': datetime.datetime.now(),
            'refresh_time': datetime.datetime.now(),
            'exp_key': None,
        }
        trials.insert_trial_docs([trial_doc])
        trials.refresh()
        self.logger.info("Injected initial trial (tid={}) into Trials for TPE prior.".format(tid))

    def setup_space(self):
        space = {}
        model_space_name = self.tuner_config.get("model", {}).get("space")
        strategy_space_name = self.tuner_config.get("strategy", {}).get("space")

        if model_space_name is not None:
            space["model_space"] = getattr(
                importlib.import_module(".space", package="qlib.contrib.tuner"),
                model_space_name,
            )
        if strategy_space_name is not None:
            space["strategy_space"] = getattr(
                importlib.import_module(".space", package="qlib.contrib.tuner"),
                strategy_space_name,
            )
        if not space:
            raise ValueError("Please give at least one search space for workflow tuning.")
        return space

    def objective(self, params):
        workflow_path = self.setup_workflow_config(params)
        self.logger.info("Searching params: {} ".format(params))
        try:
            res, res_info = self.run_workflow(workflow_path)
        except Exception:
            self.logger.exception("Workflow experiment failed when using this searching parameters")
            self._append_hist(params, np.nan, None, STATUS_FAIL)
            return {"loss": np.nan, "status": STATUS_FAIL}

        status = STATUS_FAIL if np.isnan(res) else STATUS_OK
        if status == STATUS_OK and (self.best_res is None or self.best_res > res):
            self.best_res = res
            self.best_params = params
        self._append_hist(params, res, res_info, status)
        return {"loss": res, "status": status}

    def _append_hist(self, params, res, res_info, status):
        """记录每次调优的参数和完整结果，追加到 search_history.csv"""
        row = {"eval": getattr(self, "eval_count", "?"), "loss": res, "status": status}

        # 展开 res_info（analysis_df.loc[report_type]，index=report_factor，columns=risk）
        if res_info is not None:
            for factor, factor_row in res_info.iterrows():
                row[factor] = factor_row["risk"] if "risk" in factor_row.index else factor_row.iloc[0]

        for group, group_params in params.items():
            if isinstance(group_params, dict):
                for k, v in group_params.items():
                    row[k] = v
            else:
                row[group] = group_params

        self._hist_df = pd.concat([self._hist_df, pd.DataFrame([row])], ignore_index=True)

        os.makedirs(self.ex_dir, exist_ok=True)
        hist_path = os.path.join(self.ex_dir, self.LOCAL_HIST_NAME)
        self._hist_df.to_csv(hist_path, index=False)
        self.logger.info("Search history saved to: {}".format(hist_path))

    def setup_workflow_config(self, params):
        workflow_config_path = self.tuner_config.get("workflow_config_path")
        if workflow_config_path is None:
            raise ValueError("Please give workflow_config_path for WorkflowConfigTuner.")

        with open(workflow_config_path) as fp:
            workflow_config = yaml.safe_load(fp)

        self.eval_count = getattr(self, "eval_count", 0) + 1
        base_experiment_name = self.tuner_config.get(
            "workflow_experiment_name",
            workflow_config.get("experiment_name", "workflow"),
        )
        workflow_config["experiment_name"] = "{}_tuner_{}_{}".format(
            base_experiment_name,
            self.tuner_config["experiment"]["id"],
            self.eval_count,
        )

        model_params = params.get("model_space", {})
        strategy_params = params.get("strategy_space", {})
        workflow_config["task"]["model"].setdefault("kwargs", {}).update(model_params)
        self._update_port_analysis_strategy(workflow_config, strategy_params)

        os.makedirs(self.ex_dir, exist_ok=True)
        workflow_path = os.path.join(self.ex_dir, self.WORKFLOW_CONFIG_NAME.format(self.eval_count))
        with open(workflow_path, "w") as fp:
            yaml.safe_dump(workflow_config, fp, sort_keys=False)
        return workflow_path

    def _update_port_analysis_strategy(self, workflow_config, strategy_params):
        if not strategy_params:
            return
        if "port_analysis_config" in workflow_config:
            workflow_config["port_analysis_config"]["strategy"].setdefault("kwargs", {}).update(strategy_params)

        for record_config in workflow_config["task"].get("record", []):
            if record_config.get("class") != "PortAnaRecord":
                continue
            config = record_config.setdefault("kwargs", {}).get("config")
            if config is not None:
                config["strategy"].setdefault("kwargs", {}).update(strategy_params)

    def run_workflow(self, workflow_path):
        import qlib
        from pathlib import Path
        from qlib.cli.run import load_config
        from qlib.config import C
        from qlib.model.trainer import task_train

        config = load_config(workflow_path)
        qlib_init_config = config.get("qlib_init")
        if "exp_manager" in qlib_init_config:
            qlib.init(**qlib_init_config)
        else:
            exp_manager = copy.deepcopy(C["exp_manager"])
            mlflow_uri = self.tuner_config.get("mlflow_tracking_uri", os.path.join(self.ex_dir, "mlruns"))
            exp_manager["kwargs"]["uri"] = "file:" + str(Path(mlflow_uri).resolve())
            qlib.init(**qlib_init_config, exp_manager=exp_manager)

        recorder = task_train(config.get("task"), experiment_name=config["experiment_name"])
        recorder.save_objects(config=config)
        res, res_info = self.fetch_result(recorder)
        return res, res_info

    def fetch_result(self, recorder):
        if self.optim_config.report_type == "model":
            raise ValueError("WorkflowConfigTuner only supports portfolio analysis metrics now.")

        analysis_freq = self.tuner_config.get("analysis_freq", "1day")
        analysis_df = recorder.load_object("portfolio_analysis/port_analysis_{}.pkl".format(analysis_freq))
        res_info = analysis_df.loc[self.optim_config.report_type]
        res = res_info.loc[self.optim_config.report_factor]
        value = res["risk"] if isinstance(res, pd.Series) and "risk" in res else res.values[0]

        if self.optim_config.optim_type == "min":
            return value, res_info
        if self.optim_config.optim_type == "max":
            return -value, res_info
        return np.abs(value - 1), res_info

    def save_local_best_params(self):
        TimeInspector.set_time_mark()
        os.makedirs(self.ex_dir, exist_ok=True)
        local_best_params_path = os.path.join(self.ex_dir, self.LOCAL_BEST_PARAMS_NAME)
        with open(local_best_params_path, "w") as fp:
            json.dump(self.best_params, fp)
        TimeInspector.log_cost_time(
            "Finished saving local best tuner parameters to: {} .".format(local_best_params_path)
        )


class QLibTuner(Tuner):
    ESTIMATOR_CONFIG_NAME = "estimator_config.yaml"
    EXP_INFO_NAME = "exp_info.json"
    EXP_RESULT_DIR = "sacred/{}"
    EXP_RESULT_NAME = "analysis.pkl"
    LOCAL_BEST_PARAMS_NAME = "local_best_params.json"

    def objective(self, params):
        # 1. Setup an config for a specific estimator process
        estimator_path = self.setup_estimator_config(params)
        self.logger.info("Searching params: {} ".format(params))

        # 2. Use subprocess to do the estimator program, this process will wait until subprocess finish
        sub_fails = subprocess.call("estimator -c {}".format(estimator_path), shell=True)
        if sub_fails:
            # If this subprocess failed, ignore this evaluation step
            self.logger.info("Estimator experiment failed when using this searching parameters")
            return {"loss": np.nan, "status": STATUS_FAIL}

        # 3. Fetch the result of subprocess, and check whether the result is Nan
        res = self.fetch_result()
        if np.isnan(res):
            status = STATUS_FAIL
        else:
            status = STATUS_OK

        # 4. Save the best score and params
        if self.best_res is None or self.best_res > res:
            self.best_res = res
            self.best_params = params

        # 5. Return the result as optim objective
        return {"loss": res, "status": status}

    def fetch_result(self):
        # 1. Get experiment information
        exp_info_path = os.path.join(self.ex_dir, QLibTuner.EXP_INFO_NAME)
        with open(exp_info_path) as fp:
            exp_info = json.load(fp)
        estimator_ex_id = exp_info["id"]

        # 2. Return model result if needed
        if self.optim_config.report_type == "model":
            if self.optim_config.report_factor == "model_score":
                # if estimator experiment is multi-label training, user need to process the scores by himself
                # Default method is return the average score
                return np.mean(exp_info["performance"]["model_score"])
            elif self.optim_config.report_factor == "model_pearsonr":
                # pearsonr is a correlation coefficient, 1 is the best
                return np.abs(exp_info["performance"]["model_pearsonr"] - 1)

        # 3. Get backtest results
        exp_result_dir = os.path.join(self.ex_dir, QLibTuner.EXP_RESULT_DIR.format(estimator_ex_id))
        exp_result_path = os.path.join(exp_result_dir, QLibTuner.EXP_RESULT_NAME)
        with open(exp_result_path, "rb") as fp:
            analysis_df = restricted_pickle_load(fp)

        # 4. Get the backtest factor which user want to optimize, if user want to maximize the factor, then reverse the result
        res = analysis_df.loc[self.optim_config.report_type].loc[self.optim_config.report_factor]
        # res = res.values[0] if self.optim_config.optim_type == 'min' else -res.values[0]
        if self.optim_config == "min":
            return res.values[0]
        elif self.optim_config == "max":
            return -res.values[0]
        else:
            # self.optim_config == 'correlation'
            return np.abs(res.values[0] - 1)

    def setup_estimator_config(self, params):
        estimator_config = copy.deepcopy(self.tuner_config)
        estimator_config["model"].update({"args": params["model_space"]})
        estimator_config["strategy"].update({"args": params["strategy_space"]})
        if params.get("data_label_space", None) is not None:
            estimator_config["data"]["args"].update(params["data_label_space"])

        estimator_path = os.path.join(
            self.tuner_config["experiment"].get("dir", "../"),
            QLibTuner.ESTIMATOR_CONFIG_NAME,
        )

        with open(estimator_path, "w") as fp:
            yaml.dump(estimator_config, fp)

        return estimator_path

    def setup_space(self):
        # 1. Setup model space
        model_space_name = self.tuner_config["model"].get("space", None)
        if model_space_name is None:
            raise ValueError("Please give the search space of model.")
        model_space = getattr(
            importlib.import_module(".space", package="qlib.contrib.tuner"),
            model_space_name,
        )

        # 2. Setup strategy space
        strategy_space_name = self.tuner_config["strategy"].get("space", None)
        if strategy_space_name is None:
            raise ValueError("Please give the search space of strategy.")
        strategy_space = getattr(
            importlib.import_module(".space", package="qlib.contrib.tuner"),
            strategy_space_name,
        )

        # 3. Setup data label space if given
        if self.tuner_config.get("data_label", None) is not None:
            data_label_space_name = self.tuner_config["data_label"].get("space", None)
            if data_label_space_name is not None:
                data_label_space = getattr(
                    importlib.import_module(".space", package="qlib.contrib.tuner"),
                    data_label_space_name,
                )
        else:
            data_label_space_name = None

        # 4. Combine the searching space
        space = dict()
        space.update({"model_space": model_space})
        space.update({"strategy_space": strategy_space})
        if data_label_space_name is not None:
            space.update({"data_label_space": data_label_space})

        return space

    def save_local_best_params(self):
        TimeInspector.set_time_mark()
        local_best_params_path = os.path.join(self.ex_dir, QLibTuner.LOCAL_BEST_PARAMS_NAME)
        with open(local_best_params_path, "w") as fp:
            json.dump(self.best_params, fp)
        TimeInspector.log_cost_time(
            "Finished saving local best tuner parameters to: {} .".format(local_best_params_path)
        )
