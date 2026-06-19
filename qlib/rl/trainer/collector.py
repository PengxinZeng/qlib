"""Trajectory Collector for PPO Training

Collects rollouts (trajectories) from environment interactions.
"""

from typing import List, Dict, Tuple
import numpy as np
import torch
from dataclasses import dataclass


@dataclass
class Transition:
    """Single trajectory step
    
    Attributes:
        state: Dict with market state (torch tensors, no dates)
        action: (M+1,) action taken (torch tensor)
        log_prob: Log probability of action (scalar tensor)
        reward: Immediate reward (float)
        done: Whether episode terminates (bool)
        value: Value estimate at this state (scalar tensor)
    """
    state: Dict  # torch tensors
    action: torch.Tensor  # (M+1,)
    log_prob: torch.Tensor  # scalar
    reward: float
    done: bool
    value: torch.Tensor  # scalar


class Collector:
    """Collect trajectories from environment
    
    Rolls out episodes using the current policy model.
    """

    def __init__(self, env, model, config):
        """Initialize collector
        
        Args:
            env: MultiStockDailyTradingEnv instance
            model: MultiStockActorCritic model
            config: PPOConfig instance
        """
        self.env = env
        self.model = model
        self.config = config

    def collect_rollout(self) -> List[Transition]:
        """Collect one rollout episode"""
        transitions = []
        state = self.env.reset()

        for step in range(self.config.rollout_days):
            # TODO: self.config.rollout_days == 120, sould be 2k
            action, log_prob, value = self.model.act(state, deterministic=False)
            next_state, reward, done = self.env.step(action)

            transitions.append(Transition(
                state=state, action=action, log_prob=log_prob,
                reward=reward, done=done, value=value,
            ))

            state = self.env.reset() if done else next_state

        return transitions

