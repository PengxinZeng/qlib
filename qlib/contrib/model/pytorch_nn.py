# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


from __future__ import division
from __future__ import print_function
from collections import defaultdict

import os
import gc
import numpy as np
import pandas as pd
from packaging import version
from typing import Callable, Optional, Text, Union
from sklearn.metrics import roc_auc_score, mean_squared_error

import torch
import torch.nn as nn
import torch.optim as optim

from .pytorch_utils import count_parameters
from ...model.base import Model
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.weight import Reweighter
from ...utils import (
    auto_filter_kwargs,
    init_instance_by_config,
    unpack_archive_with_buffer,
    save_multiple_parts_file,
    get_or_create_path,
)
from ...log import get_module_logger
from ...workflow import R
from qlib.contrib.meta.data_selection.utils import ICLoss
from torch.nn import DataParallel


class TopKPortfolioLoss(nn.Module):
    """Portfolio-return loss: differentiable surrogate for TopkDropoutStrategy(topk) return.

    ``mode="soft"`` (training objective): weights = temperature softmax over the daily
    cross-section, so **forward and gradient are the same function** — the model learns
    exactly what it optimizes. It only needs >=2 instruments per day (non-degenerate
    softmax), so early years with a few ETFs remain usable for training.

    ``mode="hard"`` (evaluation metric): exact top-k equal-weight portfolio return,
    forward-only (no gradient). Used as the validation metric so early stopping /
    checkpoint selection matches the true strategy return.

    ``idx`` must be a sorted index of (datetime, instrument) aligned with pred/y
    (same convention as :class:`ICLoss`).
    """

    def __init__(self, topk=5, tau=0.2, skip_size=2, tau_min=0.1, anneal_steps=2000, mode="soft"):
        super().__init__()
        self.topk = topk
        self.tau = tau
        self.skip_size = skip_size
        self.tau_min = tau_min
        self.anneal_steps = anneal_steps
        self.mode = mode
        self._step = 0

    def _current_tau(self):
        """linear anneal from ``tau`` down to ``tau_min`` over ``anneal_steps``"""
        if self.anneal_steps <= 0:
            return self.tau
        self._step += 1
        ratio = min(1.0, self._step / self.anneal_steps)
        return self.tau + (self.tau_min - self.tau) * ratio

    def forward(self, pred, y, idx):
        # split by day (assume idx sorted by <datetime, instrument>, like ICLoss)
        prev = None
        diff_point = []
        for i, (date, inst) in enumerate(idx):
            if date != prev:
                diff_point.append(i)
            prev = date
        diff_point.append(None)

        tau = self._current_tau()
        loss_sum = 0.0
        n_days = 0
        for start_i, end_i in zip(diff_point, diff_point[1:]):
            p = pred[start_i:end_i]
            r = y[start_i:end_i]
            # soft 模式只要 ≥skip_size 只；hard 模式要凑满 topk 才能取 top-k
            min_n = max(self.skip_size, self.topk) if self.mode == "hard" else self.skip_size
            if p.shape[0] < min_n:
                continue
            if p.std() < 1e-8 or r.std() < 1e-8:
                continue
            # per-day z-score of predictions: scale-invariant, removes day drift
            p_n = (p - p.mean()) / (p.std() + 1e-8)
            if self.mode == "soft":
                # softmax 加权组合收益：forward == gradient，模型学的就是它
                w = torch.softmax(p_n / tau, dim=0)
                day_ret = (w * r).sum()
            else:
                # hard：精确 top-k 等权收益（仅前向，评测/选点用）
                _, topk_idx = torch.topk(p_n, self.topk)
                day_ret = r[topk_idx].mean()
            loss_sum = loss_sum - day_ret  # negative mean portfolio return
            n_days += 1
        if n_days <= 0:
            raise ValueError("No enough data for TopKPortfolioLoss: n_days=0")
        return loss_sum / n_days


class DNNModelPytorch(Model):
    """DNN Model
    Parameters
    ----------
    input_dim : int
        input dimension
    output_dim : int
        output dimension
    layers : tuple
        layer sizes
    lr : float
        learning rate
    optimizer : str
        optimizer name
    GPU : int
        the GPU ID used for training
    """

    def __init__(
        self,
        lr=0.001,
        max_steps=300,
        batch_size=2000,
        early_stop_rounds=50,
        eval_steps=20,
        optimizer="gd",
        loss="mse",
        GPU=0,
        seed=None,
        weight_decay=0.0,
        data_parall=False,
        scheduler: Optional[Union[Callable]] = "default",  # when it is Callable, it accept one argument named optimizer
        init_model=None,
        eval_train_metric=False,
        pt_model_uri="qlib.contrib.model.pytorch_nn.Net",
        pt_model_kwargs={
            "input_dim": 360,
            "layers": (256,),
        },
        valid_key=DataHandlerLP.DK_L,
        ic_skip_size=50,  # ICLoss 按日计算 IC 时的最小样本数（仅 mse/binary 模式的指标用）
        # ---- portfolio loss (loss=portfolio) 专属参数 ----
        min_days_size=2,  # portfolio 训练：每日最少标的数。softmax 梯度不要求凑满 topk，≥2 只即可
        topk=5,  # 组合持仓数（hard 评测口径 + 与 TopkDropoutStrategy topk 对齐）
        tau=0.2,  # softmax 温度；退火起点
        tau_min=0.1,  # 退火终点
        anneal_steps=2000,  # tau 从 tau 线性退火到 tau_min 的步数
        turnover_penalty=0.0,  # v1 暂未启用（跨日标的对齐），保留参数位
        dates_per_step=16,  # portfolio 模式下每步采样的交易日数（每日全横截面）
        portfolio_weight=0.5,  # portfolio_mse 混合 loss：组合收益项的权重，(1-w) 给 MSE 项
        # TODO: Infer Key is a more reasonable key. But it requires more detailed processing on label processing
    ):
        # Set logger.
        self.logger = get_module_logger("DNNModelPytorch")
        self.logger.info("DNN pytorch version...")

        # set hyper-parameters.
        self.lr = lr
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.early_stop_rounds = early_stop_rounds
        self.eval_steps = eval_steps
        self.ic_skip_size = ic_skip_size
        self.min_days_size = min_days_size
        self.topk = topk
        self.tau = tau
        self.tau_min = tau_min
        self.anneal_steps = anneal_steps
        self.turnover_penalty = turnover_penalty
        self.dates_per_step = dates_per_step
        self.portfolio_weight = portfolio_weight
        self.optimizer = optimizer.lower()
        self.loss_type = loss
        if isinstance(GPU, str):
            self.device = torch.device(GPU)
        else:
            self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self.weight_decay = weight_decay
        self.data_parall = data_parall
        self.eval_train_metric = eval_train_metric
        self.valid_key = valid_key

        self.best_step = None

        self.logger.info(
            "DNN parameters setting:"
            f"\nlr : {lr}"
            f"\nmax_steps : {max_steps}"
            f"\nbatch_size : {batch_size}"
            f"\nearly_stop_rounds : {early_stop_rounds}"
            f"\neval_steps : {eval_steps}"
            f"\noptimizer : {optimizer}"
            f"\nloss_type : {loss}"
            f"\nseed : {seed}"
            f"\ndevice : {self.device}"
            f"\nuse_GPU : {self.use_gpu}"
            f"\nweight_decay : {weight_decay}"
            f"\nenable data parall : {self.data_parall}"
            f"\npt_model_uri: {pt_model_uri}"
            f"\npt_model_kwargs: {pt_model_kwargs}"
            f"\ntopk : {topk} (hard eval)  min_days_size : {min_days_size} (soft train)"
            f"\ntau : {tau} -> tau_min={tau_min} over {anneal_steps} steps"
            f"\nturnover_penalty : {turnover_penalty}"
            f"\ndates_per_step : {dates_per_step}"
            f"\nportfolio_weight : {portfolio_weight} (portfolio_mse 混合 loss)"
        )

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        if loss not in {"mse", "binary", "portfolio", "portfolio_mse"}:
            raise NotImplementedError("loss {} is not supported!".format(loss))
        self._scorer = mean_squared_error if loss in ("mse", "portfolio_mse") else roc_auc_score

        # 训练用 portfolio loss（soft 模式：softmax 加权收益，梯度=目标；每日 ≥ min_days_size 只即可）
        self._portfolio_loss = TopKPortfolioLoss(
            topk=self.topk,
            tau=self.tau,
            skip_size=self.min_days_size,
            tau_min=self.tau_min,
            anneal_steps=self.anneal_steps,
            mode="soft",
        )
        # 验证/评测用实例（hard 模式：精确 top-k 等权收益，不退火，早停选点与策略同口径）
        self._portfolio_loss_eval = TopKPortfolioLoss(
            topk=self.topk,
            tau=self.tau_min,
            skip_size=self.min_days_size,
            tau_min=self.tau_min,
            anneal_steps=0,
            mode="hard",
        )

        if init_model is None:
            self.dnn_model = init_instance_by_config({"class": pt_model_uri, "kwargs": pt_model_kwargs})

            if self.data_parall:
                self.dnn_model = DataParallel(self.dnn_model).to(self.device)
        else:
            self.dnn_model = init_model

        self.logger.info("model:\n{:}".format(self.dnn_model))
        self.logger.info("model size: {:.4f} MB".format(count_parameters(self.dnn_model)))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.dnn_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.dnn_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError("optimizer {} is not supported!".format(optimizer))

        if scheduler == "default":
            # In torch version 2.7.0, the verbose parameter has been removed. Reference Link:
            # https://github.com/pytorch/pytorch/pull/147301/files#diff-036a7470d5307f13c9a6a51c3a65dd014f00ca02f476c545488cd856bea9bcf2L1313
            if version.parse(str(torch.__version__).split("+", maxsplit=1)[0]) <= version.parse("2.6.0"):
                # Reduce learning rate when loss has stopped decrease
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(  # pylint: disable=E1123
                    self.train_optimizer,
                    mode="min",
                    factor=0.5,
                    patience=10,
                    verbose=True,
                    threshold=0.0001,
                    threshold_mode="rel",
                    cooldown=0,
                    min_lr=0.00001,
                    eps=1e-08,
                )
            else:
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.train_optimizer,
                    mode="min",
                    factor=0.5,
                    patience=10,
                    threshold=0.0001,
                    threshold_mode="rel",
                    cooldown=0,
                    min_lr=0.00001,
                    eps=1e-08,
                )
        elif scheduler is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler(optimizer=self.train_optimizer)

        self.fitted = False
        self.dnn_model.to(self.device)

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def fit(
        self,
        dataset: DatasetH,
        evals_result=dict(),
        verbose=True,
        save_path=None,
        reweighter=None,
    ):
        has_valid = "valid" in dataset.segments
        segments = ["train", "valid"]
        vars = ["x", "y", "w"]
        all_df = defaultdict(dict)  # x_train, x_valid y_train, y_valid w_train, w_valid
        all_t = defaultdict(dict)  # tensors
        for seg in segments:
            if seg in dataset.segments:
                # df_train df_valid
                df = dataset.prepare(
                    seg, col_set=["feature", "label"], data_key=self.valid_key if seg == "valid" else DataHandlerLP.DK_L
                )
                all_df["x"][seg] = df["feature"]
                all_df["y"][seg] = df["label"].copy()  # We have to use copy to remove the reference to release mem
                if reweighter is None:
                    all_df["w"][seg] = pd.DataFrame(np.ones_like(all_df["y"][seg].values), index=df.index)
                elif isinstance(reweighter, Reweighter):
                    all_df["w"][seg] = pd.DataFrame(reweighter.reweight(df))
                else:
                    raise ValueError("Unsupported reweighter type.")

                # get tensors
                for v in vars:
                    all_t[v][seg] = torch.from_numpy(all_df[v][seg].values).float()
                    # if seg == "valid": # accelerate the eval of validation
                    all_t[v][seg] = all_t[v][seg].to(self.device)  # This will consume a lot of memory !!!!

                evals_result[seg] = []
                # free memory
                del df
                del all_df["x"]
                gc.collect()

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        train_loss = 0
        best_loss = np.inf
        # train
        self.logger.info("training...")
        self.fitted = True
        # return
        # prepare training data
        train_num = all_t["y"]["train"].shape[0]
        is_portfolio = self.loss_type in ("portfolio", "portfolio_mse")
        if is_portfolio:
            # softmax 训练目标只需 ≥ min_days_size 只标的（不必凑满 topk），
            # 因此早期上市 ETF 少的日子也能参与训练。
            train_index = all_df["y"]["train"].index  # MultiIndex (datetime, instrument)
            train_dates_arr = train_index.get_level_values(0).to_numpy()
            min_samples = self.min_days_size
            dates, counts = np.unique(train_dates_arr, return_counts=True)
            train_dates_unique = dates[counts >= min_samples]
            self.logger.info(
                "portfolio loss: %d/%d unique train dates have >=%d instruments, "
                "sampling %d dates per step",
                len(train_dates_unique),
                len(dates),
                min_samples,
                self.dates_per_step,
            )

        for step in range(1, self.max_steps + 1):
            if stop_steps >= self.early_stop_rounds:
                if verbose:
                    self.logger.info("\tearly stop")
                break
            loss = AverageMeter()
            self.dnn_model.train()
            self.train_optimizer.zero_grad()
            if is_portfolio:
                if self.dates_per_step > len(train_dates_unique):
                    dates_chosen = np.random.choice(train_dates_unique, self.dates_per_step, replace=True)
                else:
                    dates_chosen = np.random.choice(train_dates_unique, self.dates_per_step, replace=False)
                mask = np.isin(train_dates_arr, dates_chosen)
                choice = np.flatnonzero(mask)
                step_idx = train_index[choice]
            else:
                choice = np.random.choice(train_num, self.batch_size)
                step_idx = None
            x_batch_auto = all_t["x"]["train"][choice].to(self.device)
            y_batch_auto = all_t["y"]["train"][choice].to(self.device)
            w_batch_auto = all_t["w"]["train"][choice].to(self.device)

            # forward
            preds = self.dnn_model(x_batch_auto)
            cur_loss = self.get_loss(preds, w_batch_auto, y_batch_auto, self.loss_type, idx=step_idx)
            cur_loss.backward()
            self.train_optimizer.step()
            loss.update(cur_loss.item())
            R.log_metrics(train_loss=loss.avg, step=step)

            # validation
            train_loss += loss.val
            # for evert `eval_steps` steps or at the last steps, we will evaluate the model.
            if step % self.eval_steps == 0 or step == self.max_steps:
                if has_valid:
                    stop_steps += 1
                    train_loss /= self.eval_steps

                    with torch.no_grad():
                        self.dnn_model.eval()

                        # forward
                        preds = self._nn_predict(all_t["x"]["valid"], return_cpu=False)
                        if is_portfolio:
                            # 验证 loss 用 hard 模式 eval 实例（精确 top-k 收益，不退火，不推进训练退火计数）
                            cur_loss_val = self._portfolio_loss_eval(
                                preds.reshape(-1), all_t["y"]["valid"].reshape(-1), all_df["y"]["valid"].index
                            )
                        else:
                            cur_loss_val = self.get_loss(
                                preds, all_t["w"]["valid"], all_t["y"]["valid"], self.loss_type
                            )
                        loss_val = cur_loss_val.item()
                        metric_val = (
                            self.get_metric(
                                preds.reshape(-1), all_t["y"]["valid"].reshape(-1), all_df["y"]["valid"].index
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .item()
                        )
                        R.log_metrics(val_loss=loss_val, step=step)
                        R.log_metrics(val_metric=metric_val, step=step)

                        if self.eval_train_metric:
                            metric_train = (
                                self.get_metric(
                                    self._nn_predict(all_t["x"]["train"], return_cpu=False),
                                    all_t["y"]["train"].reshape(-1),
                                    all_df["y"]["train"].index,
                                )
                                .detach()
                                .cpu()
                                .numpy()
                                .item()
                            )
                            R.log_metrics(train_metric=metric_train, step=step)
                        else:
                            metric_train = np.nan
                    if verbose:
                        self.logger.info(
                            f"[Step {step}]: train_loss {train_loss:.6f}, valid_loss {loss_val:.6f}, train_metric {metric_train:.6f}, valid_metric {metric_val:.6f}"
                        )
                    evals_result["train"].append(train_loss)
                    evals_result["valid"].append(loss_val)
                    if loss_val < best_loss:
                        if verbose:
                            self.logger.info(
                                "\tvalid loss update from {:.6f} to {:.6f}, save checkpoint.".format(
                                    best_loss, loss_val
                                )
                            )
                        best_loss = loss_val
                        self.best_step = step
                        R.log_metrics(best_step=self.best_step, step=step)
                        stop_steps = 0
                        torch.save(self.dnn_model.state_dict(), save_path)
                    train_loss = 0
                    # update learning rate
                    if self.scheduler is not None:
                        auto_filter_kwargs(self.scheduler.step, warning=False)(metrics=cur_loss_val, epoch=step)
                    R.log_metrics(lr=self.get_lr(), step=step)
                else:
                    # retraining mode
                    if self.scheduler is not None:
                        self.scheduler.step(epoch=step)

        if has_valid:
            # restore the optimal parameters after training
            self.dnn_model.load_state_dict(torch.load(save_path, map_location=self.device))
        if self.use_gpu:
            torch.cuda.empty_cache()

    def get_lr(self):
        assert len(self.train_optimizer.param_groups) == 1
        return self.train_optimizer.param_groups[0]["lr"]

    def get_loss(self, pred, w, target, loss_type, idx=None):
        pred, target = pred.reshape(-1), target.reshape(-1)
        if loss_type in ("portfolio", "portfolio_mse"):
            # 组合收益 loss：需要 (datetime, instrument) 索引按日分组；忽略 reweighter w
            pl = self._portfolio_loss(pred, target, idx)
            if loss_type == "portfolio":
                return pl
            # portfolio_mse：组合收益项 + MSE 项加权混合（w=portfolio_weight）
            mse = torch.mul(pred - target, pred - target).mean()
            return self.portfolio_weight * pl + (1.0 - self.portfolio_weight) * mse
        w = w.reshape(-1)
        if loss_type == "mse":
            sqr_loss = torch.mul(pred - target, pred - target)
            loss = torch.mul(sqr_loss, w).mean()
            return loss
        elif loss_type == "binary":
            loss = nn.BCEWithLogitsLoss(weight=w)
            return loss(pred, target)
        else:
            raise NotImplementedError("loss {} is not supported!".format(loss_type))

    def get_metric(self, pred, target, index):
        if self.loss_type in ("portfolio", "portfolio_mse"):
            # 验证/测试期的 top-k 等权组合日收益均值（正值越大越好；hard 口径与策略一致）
            return -self._portfolio_loss_eval(pred, target, index)  # pylint: disable=E1130
        # NOTE: the order of the index must follow <datetime, instrument> sorted order
        return -ICLoss(skip_size=self.ic_skip_size)(pred, target, index)  # pylint: disable=E1130

    def _nn_predict(self, data, return_cpu=True):
        """Reusing predicting NN.
        Scenarios
        1) test inference (data may come from CPU and expect the output data is on CPU)
        2) evaluation on training (data may come from GPU)
        """
        if not isinstance(data, torch.Tensor):
            if isinstance(data, pd.DataFrame):
                data = data.values
            data = torch.Tensor(data)
        data = data.to(self.device)
        preds = []
        self.dnn_model.eval()
        with torch.no_grad():
            batch_size = 8096
            for i in range(0, len(data), batch_size):
                x = data[i : i + batch_size]
                preds.append(self.dnn_model(x.to(self.device)).detach().reshape(-1))
        if return_cpu:
            preds = np.concatenate([pr.cpu().numpy() for pr in preds])
        else:
            preds = torch.cat(preds, axis=0)
        return preds

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        x_test_pd = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        preds = self._nn_predict(x_test_pd)
        return pd.Series(preds.reshape(-1), index=x_test_pd.index)

    def save(self, filename, **kwargs):
        with save_multiple_parts_file(filename) as model_dir:
            model_path = os.path.join(model_dir, os.path.split(model_dir)[-1])
            # Save model
            torch.save(self.dnn_model.state_dict(), model_path)

    def load(self, buffer, **kwargs):
        with unpack_archive_with_buffer(buffer) as model_dir:
            # Get model name
            _model_name = os.path.splitext(list(filter(lambda x: x.startswith("model.bin"), os.listdir(model_dir)))[0])[
                0
            ]
            _model_path = os.path.join(model_dir, _model_name)
            # Load model
            self.dnn_model.load_state_dict(torch.load(_model_path, map_location=self.device))
        self.fitted = True


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Net(nn.Module):
    def __init__(self, input_dim, output_dim=1, layers=(256,), act="LeakyReLU"):
        super(Net, self).__init__()

        layers = [input_dim] + list(layers)
        dnn_layers = []
        drop_input = nn.Dropout(0.05)
        dnn_layers.append(drop_input)
        hidden_units = input_dim
        for i, (_input_dim, hidden_units) in enumerate(zip(layers[:-1], layers[1:])):
            fc = nn.Linear(_input_dim, hidden_units)
            if act == "LeakyReLU":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)
            elif act == "SiLU":
                activation = nn.SiLU()
            else:
                raise NotImplementedError(f"This type of input is not supported")
            bn = nn.BatchNorm1d(hidden_units)
            seq = nn.Sequential(fc, bn, activation)
            dnn_layers.append(seq)
        drop_input = nn.Dropout(0.05)
        dnn_layers.append(drop_input)
        fc = nn.Linear(hidden_units, output_dim)
        dnn_layers.append(fc)
        # optimizer  # pylint: disable=W0631
        self.dnn_layers = nn.ModuleList(dnn_layers)
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        cur_output = x
        for i, now_layer in enumerate(self.dnn_layers):
            cur_output = now_layer(cur_output)
        return cur_output
