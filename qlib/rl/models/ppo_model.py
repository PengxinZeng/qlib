"""PPO Model: qlib Model interface wrapper for MultiStockActorCritic + PPOTrainer"""

import logging
from typing import Union, Text

import numpy as np
import pandas as pd
import torch

from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from .actor_critic import MultiStockActorCritic
from .ppo_config import PPOConfig
from ..multi_stock.env import MultiStockDailyTradingEnv


KLINE_COLS = ["open", "high", "low", "close", "volume"]
VAL_COLS   = ["pb", "pb_median", "pe_ttm", "pe_ttm_median"]
MACRO_COLS = ["cn_2y", "cn_5y", "cn_10y", "us_2y", "us_5y", "us_10y"]


class FeatureScaler:
    """Per-feature Z-score normalizer for (n_dates, M, n_features) arrays."""

    def fit(self, *arrays: np.ndarray) -> "FeatureScaler":
        data = np.concatenate([a.reshape(-1, a.shape[-1]) for a in arrays], axis=0)
        self.mean_ = np.nanmean(data, axis=0)[np.newaxis, np.newaxis]  # (1, 1, n_features)
        self.std_  = np.nanstd(data, axis=0)[np.newaxis, np.newaxis] + 1e-8
        return self

    def transform(self, arr: np.ndarray) -> np.ndarray:
        return (arr - self.mean_) / self.std_


class PPOModel(Model):
    """qlib Model wrapper for PPO multi-stock trading strategy."""

    def __init__(self, config):
        if isinstance(config, dict):
            config = PPOConfig(**config)
        self.ppo_config = config
        self.device = "cpu"
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.fitted = False
        self.kline_scaler = FeatureScaler()
        self.val_scaler   = FeatureScaler()
        self.macro_scaler = FeatureScaler()

    def _dataset_to_arrays(self, df: pd.DataFrame):
        """Convert DatasetH DataFrame → (n_dates, M, n_features) arrays."""
        feature_df = df["feature"]
        dates   = feature_df.index.get_level_values("datetime").unique().sort_values()
        tickers = feature_df.index.get_level_values("instrument").unique().tolist()
        n_dates, M = len(dates), len(tickers)

        def _pivot(cols):
            frames = [
                feature_df[c].unstack("instrument").reindex(index=dates, columns=tickers)
                .values if c in feature_df.columns
                else np.full((n_dates, M), np.nan)
                for c in cols
            ]
            return np.stack(frames, axis=-1).astype(np.float32)

        return (
            _pivot(KLINE_COLS), _pivot(VAL_COLS), _pivot(MACRO_COLS),
            dates.to_numpy(), tickers,
        )

    def _build_env(self, kline_raw, kline_norm, valuation, macro, dates):
        return MultiStockDailyTradingEnv(
            kline_data=kline_raw, valuation_data=valuation, macro_data=macro,
            dates=dates, stock_tickers=list(range(kline_raw.shape[1])),
            lookback_window=self.ppo_config.lookback_window,
            transaction_cost=self.ppo_config.transaction_cost,
            kline_norm=kline_norm,
        )

    def fit(self, dataset: DatasetH, **kwargs):
        from ..trainer.ppo_trainer import PPOTrainer

        self.logger.info("Preparing data …")
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L,
        )

        kline_tr_raw, val_tr, mac_tr, dates_tr, _ = self._dataset_to_arrays(df_train)
        kline_vl_raw, val_vl, mac_vl, dates_vl, _ = self._dataset_to_arrays(df_valid)

        kline_tr_norm = self.kline_scaler.fit(kline_tr_raw).transform(kline_tr_raw)
        kline_vl_norm = self.kline_scaler.transform(kline_vl_raw)
        val_tr   = self.val_scaler.fit(val_tr).transform(val_tr)
        val_vl   = self.val_scaler.transform(val_vl)
        mac_tr   = self.macro_scaler.fit(mac_tr).transform(mac_tr)
        mac_vl   = self.macro_scaler.transform(mac_vl)

        train_env = self._build_env(kline_tr_raw, kline_tr_norm, val_tr, mac_tr, dates_tr)
        val_env   = self._build_env(kline_vl_raw, kline_vl_norm, val_vl, mac_vl, dates_vl)

        self.model = MultiStockActorCritic(self.ppo_config).to(self.device)
        trainer = PPOTrainer(train_env, self.model, self.ppo_config, device=self.device)

        self.logger.info("Training PPO for %d updates …", self.ppo_config.total_updates)
        trainer.train(num_updates=self.ppo_config.total_updates, val_env=val_env)
        self.fitted = True

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test") -> pd.Series:
        if not self.fitted:
            raise ValueError("Model is not fitted yet.")

        df_test = dataset.prepare(segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        kline_raw, valuation, macro, dates, tickers = self._dataset_to_arrays(df_test)

        kline_norm = self.kline_scaler.transform(kline_raw)
        valuation  = self.val_scaler.transform(valuation)
        macro      = self.macro_scaler.transform(macro)

        env = self._build_env(kline_raw, kline_norm, valuation, macro, dates)
        self.model.eval()
        state = env.reset()
        predictions = {}

        while True:
            action, _, _ = self.model.act(state, deterministic=True)
            date = env.current_date
            for i, ticker in enumerate(tickers):
                predictions[(date, ticker)] = float(action[i])
            state, _, done = env.step(action)
            if done:
                break

        idx = pd.MultiIndex.from_tuples(list(predictions.keys()), names=["datetime", "instrument"])
        return pd.Series(list(predictions.values()), index=idx)
