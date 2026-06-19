"""Advantage Computation using Generalized Advantage Estimation (GAE)"""

from typing import Tuple, List
import torch
from .collector import Transition


def compute_returns_and_advantages(
    transitions: List[Transition],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute returns and advantages using GAE"""
    values = torch.stack([t.value for t in transitions])    # (T,)
    rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
    masks = torch.tensor([0.0 if t.done else 1.0 for t in transitions])

    T = len(transitions)
    advantages = torch.zeros(T)
    adv = torch.tensor(0.0)
    next_val = torch.tensor(0.0)

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_val * masks[t] - values[t]
        adv = delta + gamma * gae_lambda * masks[t] * adv
        advantages[t] = adv
        next_val = values[t]

    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages
