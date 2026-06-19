"""Multi-Stock Daily Trading Environment

Simulator for multi-stock daily frequency trading with K-line, valuation, and macro data.
"""

from typing import Dict, Tuple, Optional
import numpy as np
import torch
from abc import ABC, abstractmethod


class MultiStockDailyTradingEnv:
    """Multi-stock daily trading environment
    
    Simulates daily trading with historical market data including K-line,
    valuation indicators, and macro economic indicators.
    
    State: {
        'kline': (1, M, lookback_window, n_kline_features),
        'valuation': (1, M, lookback_window, n_valuation_features),
        'macro': (1, M, lookback_window, n_macro_features),
        'holdings': (1, M+1) - current holdings [stocks..., cash],
        'dates': (lookback_window,) - trading dates in window
    }
    
    Action: (M+1,) - target holding ratio (normalized to sum=1)
    
    Reward: float - daily return
    Done: bool - whether backtest ends
    """

    def __init__(
        self,
        kline_data: np.ndarray,  # (n_dates, M, n_kline_features) — raw, used for price/return calc
        valuation_data: np.ndarray,  # (n_dates, M, n_valuation_features)
        macro_data: np.ndarray,  # (n_dates, M, n_macro_features)
        dates: np.ndarray,  # (n_dates,) trading dates
        stock_tickers: list,  # M stock identifiers
        lookback_window: int = 1000,
        initial_capital: float = 1000000.0,
        transaction_cost: float = 0.0003,
        device: str = 'cpu',
        kline_norm: np.ndarray = None,  # (n_dates, M, n_kline_features) — normalized, used for obs
    ):
        """Initialize environment
        
        Args:
            kline_data: Shape (n_dates, M, n_kline_features)
            valuation_data: Shape (n_dates, M, n_valuation_features)
            macro_data: Shape (n_dates, M, n_macro_features)
            dates: Trading dates array
            stock_tickers: List of M stock identifiers
            lookback_window: Historical window length
            initial_capital: Initial portfolio value
            transaction_cost: Trading cost rate
            device: 'cpu' or 'cuda'
        """
        # Convert to torch tensors (float32)
        self.kline_data = torch.from_numpy(kline_data).float().to(device)
        self.kline_obs  = torch.from_numpy(kline_norm if kline_norm is not None else kline_data).float().to(device)
        self.valuation_data = torch.from_numpy(valuation_data).float().to(device)
        self.macro_data = torch.from_numpy(macro_data).float().to(device)
        self.dates = dates  # Keep as numpy (for dates, no tensor conversion needed)
        self.stock_tickers = stock_tickers
        self.n_stocks = len(stock_tickers)
        self.lookback_window = lookback_window
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.device = device

        self.n_dates = len(dates)
        self.reset()

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset environment and return initial state
        
        Returns:
            state: Dict with keys [kline, valuation, macro, holdings, dates]
        """
        self.current_date_idx = 0
        self.current_holdings = torch.zeros(self.n_stocks + 1, device=self.device)
        self.current_holdings[-1] = 1.0
        self.portfolio_value = self.initial_capital
        self.last_buy_days = torch.full((self.n_stocks,), float('inf'), device=self.device)

        return self._get_state()

    def step(self, action: torch.Tensor) -> Tuple[Dict, float, bool]:
        """Execute one trading day
        
        Args:
            action: (M+1,) target holding ratio [stocks..., cash]
        
        Returns:
            next_state: Dict with market data for next day
            reward: Daily return
            done: Whether backtest ends
        """
        # Normalize action to ensure sum=1
        action = torch.clamp(action, 0, 1)
        action = action / (action.sum() + 1e-8)

        # Get current close prices (from today's kline)
        today_close = self.kline_data[self.current_date_idx, :, 3]  # Close is 4th feature

        # Calculate transaction costs
        cost = torch.abs(action[:-1] - self.current_holdings[:-1]).sum().item() * self.transaction_cost

        # Update last buy days
        prev_holdings = self.current_holdings
        for i in range(self.n_stocks):
            if action[i] > prev_holdings[i] and action[i] > 1e-8:
                # Increased or new position
                self.last_buy_days[i] = 0
            elif action[i] > 1e-8:
                # Maintained position
                self.last_buy_days[i] += 1
            else:
                # Sold out
                self.last_buy_days[i] = torch.tensor(float('inf'), device=self.device)
        self.current_holdings = action
        # TODO: 这里假设T+0交易完成

        # Move to next day
        self.current_date_idx += 1
        done = self.current_date_idx >= self.n_dates

        if done:
            reward = 0.0  # No reward on final day
        else:
            # Calculate portfolio return
            tomorrow_close = self.kline_data[self.current_date_idx, :, 3]
            daily_returns = (tomorrow_close - today_close) / (today_close + 1e-8)
            daily_returns = torch.nan_to_num(daily_returns, nan=0.0)
            portfolio_return = torch.dot(self.current_holdings[:-1], daily_returns).item() - cost
            reward = portfolio_return
            self.portfolio_value *= (1 + reward)

            # Update holdings ratio to reflect price changes (NaN stocks keep their value, only ratio changes)
            new_values = self.current_holdings[:-1] * (1 + daily_returns)  # NaN stocks: +0 → unchanged value
            cash = self.current_holdings[-1:]
            total = new_values.sum() + cash.sum()
            self.current_holdings = torch.cat([new_values, cash]) / (total + 1e-8)

        return self._get_state(), reward, done

    def _pad_to_lookback(self, arr: torch.Tensor) -> torch.Tensor:
        """Pad (T, M, F) to (lookback_window, M, F) with NaN at the front."""
        T = arr.shape[0]
        if T < self.lookback_window:
            pad = torch.full((self.lookback_window - T, *arr.shape[1:]), float('nan'),
                             dtype=arr.dtype, device=arr.device)
            arr = torch.cat([pad, arr], dim=0)
        return arr

    def _get_state(self) -> Dict[str, torch.Tensor]:
        """Get current state observation
        
        Returns:
            state: Dict with market data and holdings info (all as torch tensors)
        """
        window_start = max(0, self.current_date_idx - self.lookback_window + 1)
        end = min(self.current_date_idx + 1, self.n_dates)
        kline = self._pad_to_lookback(self.kline_obs[window_start:end])
        val   = self._pad_to_lookback(self.valuation_data[window_start:end])
        macro = self._pad_to_lookback(self.macro_data[window_start:end])

        return {
            'kline': kline.transpose(0, 1).unsqueeze(0),           # (1, M, lookback_window, feat)
            'valuation': val.transpose(0, 1).unsqueeze(0),
            'macro': macro.transpose(0, 1).unsqueeze(0),
            'holdings': self.current_holdings.clone().unsqueeze(0),
            'last_buy_days': self.last_buy_days.clone().unsqueeze(0),
            'dates': self.dates[window_start:min(self.current_date_idx + 1, self.n_dates)],
        }

    @property
    def current_date(self):
        """Get current trading date"""
        if 0 <= self.current_date_idx < self.n_dates:
            return self.dates[self.current_date_idx]
        return None

    @property
    def progress(self):
        """Get training progress (0-1)"""
        return max(0, min(1.0, self.current_date_idx / self.n_dates))
