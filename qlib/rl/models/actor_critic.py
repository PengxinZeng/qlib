"""Multi-Stock Actor-Critic Model for PPO Trading Strategy"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from .feature_extractor import FeatureExtractor
from .ppo_config import PPOConfig


class MultiStockActorCritic(nn.Module):
    """Actor-Critic network for multi-stock daily trading
    
    Architecture:
        FeatureExtractor: (kline, valuation, macro) → (M, feature_dim=32)
                         ↓
        Portfolio Aggregation: (M, 32) → (32,)
                         ↓
        ├─ Actor: (32,) → (64,) → M (logits) + 0 (cash)
        └─ Critic: (32,) → (64,) → 1 (value)
    """

    def __init__(
        self,
        config,
    ):
        """Initialize Actor-Critic network
        
        Args:
            config: PPOConfig instance or dict with all hyperparameters
        """
        super().__init__()

        if isinstance(config, dict):
            config = PPOConfig(**config)
        self.config = config
        self.feature_dim = config.feature_dim

        # Feature extraction layer
        self.feature_extractor = FeatureExtractor(
            n_kline_features=config.n_kline_features,
            n_valuation_features=config.n_valuation_features,
            n_macro_features=config.n_macro_features,
            lookback_window=config.lookback_window,
            projection_dim=config.projection_dim,
            num_heads=config.num_heads,
            feature_dim=config.feature_dim,
        )

        # Actor network (policy): shared hidden, per-stock logit
        self.actor_hidden = nn.Linear(config.feature_dim, config.actor_hidden_dim)
        self.actor_logits = nn.Linear(config.actor_hidden_dim, 1)

        # Critic network (value function)
        self.critic_hidden = nn.Linear(config.feature_dim, config.critic_hidden_dim)
        self.critic_value = nn.Linear(config.critic_hidden_dim, 1)   

    def forward(
        self,
        state_dict: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass
        
        Args:
            state_dict: Dict with 'kline', 'valuation', 'macro', 'holdings', 'last_buy_days' tensors
        
        Returns:
            logits: (batch, M+1) action logits
            value: (batch,) state value estimate
        """
        kline = state_dict.get('kline')
        if kline is None:
            raise ValueError("'kline' must be in state_dict")

        batch_size = kline.shape[0]
        n_stocks = kline.shape[1]

        # Extract features: (batch, M, feature_dim)
        stock_features, stock_mask = self.feature_extractor(state_dict)  # (batch, M, fd), (batch, M)
        portfolio_feat = stock_features.masked_fill(stock_mask.unsqueeze(-1), 0.0)
        valid_counts = (~stock_mask).float().sum(dim=1, keepdim=True).clamp(min=1)
        portfolio_feat = portfolio_feat.sum(dim=1) / valid_counts  # (batch, feature_dim)

        # Actor: per-stock hidden → per-stock logit
        actor_hidden = F.relu(self.actor_hidden(stock_features))       # (batch, M, actor_hidden_dim)
        stock_logits = self.actor_logits(actor_hidden).squeeze(-1)     # (batch, M)
        stock_logits = stock_logits.masked_fill(stock_mask, float('-inf'))  # invalid stocks → -inf
        cash_logit = torch.zeros(batch_size, 1, device=stock_logits.device, dtype=stock_logits.dtype)
        logits = torch.cat([stock_logits, cash_logit], dim=1)          # (batch, M+1)

        # Critic
        critic_hidden = F.relu(self.critic_hidden(portfolio_feat))
        value = self.critic_value(critic_hidden).squeeze(dim=-1)  # (batch,)

        return logits, value

    def act(
        self,
        state_dict: dict,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            logits, value = self.forward(state_dict)                      # (1, M+1), (1,)
            concentration = F.softplus(logits[0]) + 1e-6                  # (M+1,) > 0
            dist = torch.distributions.Dirichlet(concentration)
            action = concentration / concentration.sum() if deterministic else dist.sample()
            return action.detach(), dist.log_prob(action), value[0]

    def get_value(self, state_dict: dict) -> torch.Tensor:
        with torch.no_grad():
            _, value = self.forward(state_dict)  # (1,)
            return value[0]
