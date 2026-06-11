"""A standalone, simplified PPO example.

This file intentionally does not depend on the surrounding project. It trains a
small actor-critic agent on a toy 1D control task using only PyTorch and NumPy.
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.optim import Adam


class LineWorldEnv:
    """A tiny 1D environment for demonstrating PPO.

    State: current position normalized to [-1, 1].
    Action: 0 moves left, 1 stays, 2 moves right.
    Goal: reach the right edge as quickly as possible.
    """

    def __init__(self, size: int = 11, max_steps: int = 30) -> None:
        self.size = size
        self.max_steps = max_steps
        self.position = 0
        self.steps = 0

    def reset(self) -> np.ndarray:
        self.position = self.size // 2
        self.steps = 0
        return self._state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        self.steps += 1
        self.position += action - 1
        self.position = int(np.clip(self.position, 0, self.size - 1))

        reached_goal = self.position == self.size - 1
        timed_out = self.steps >= self.max_steps
        reward = 1.0 if reached_goal else -0.01
        return self._state(), reward, reached_goal or timed_out

    def _state(self) -> np.ndarray:
        midpoint = (self.size - 1) / 2
        return np.array([(self.position - midpoint) / midpoint], dtype=np.float32)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(states)
        return self.actor(features), self.critic(features).squeeze(-1)

    def act(self, state: np.ndarray) -> Tuple[int, float, float]:
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        logits, value = self.forward(state_tensor)
        distribution = Categorical(logits=logits)
        action = distribution.sample()
        return int(action.item()), float(distribution.log_prob(action).item()), float(value.item())


@dataclass
class Transition:
    state: np.ndarray
    action: int
    log_prob: float
    reward: float
    done: bool
    value: float


@dataclass
class PPOConfig:
    rollout_steps: int = 256
    update_epochs: int = 6
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    total_updates: int = 80
    seed: int = 7


def collect_rollout(env: LineWorldEnv, model: ActorCritic, config: PPOConfig) -> List[Transition]:
    transitions = []
    state = env.reset()

    for _ in range(config.rollout_steps):
        action, log_prob, value = model.act(state)
        next_state, reward, done = env.step(action)
        transitions.append(Transition(state, action, log_prob, reward, done, value))
        state = env.reset() if done else next_state

    return transitions


def compute_returns_and_advantages(
    transitions: List[Transition], config: PPOConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    returns = []
    advantages = []
    next_value = 0.0
    advantage = 0.0

    for transition in reversed(transitions):
        mask = 0.0 if transition.done else 1.0
        delta = transition.reward + config.gamma * next_value * mask - transition.value
        advantage = delta + config.gamma * config.gae_lambda * mask * advantage
        next_value = transition.value
        advantages.append(advantage)
        returns.append(advantage + transition.value)

    advantages = torch.tensor(list(reversed(advantages)), dtype=torch.float32)
    returns = torch.tensor(list(reversed(returns)), dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


def update_model(
    model: ActorCritic,
    optimizer: Adam,
    transitions: List[Transition],
    returns: torch.Tensor,
    advantages: torch.Tensor,
    config: PPOConfig,
) -> None:
    states = torch.tensor(np.array([transition.state for transition in transitions]), dtype=torch.float32)
    actions = torch.tensor([transition.action for transition in transitions], dtype=torch.long)
    old_log_probs = torch.tensor([transition.log_prob for transition in transitions], dtype=torch.float32)

    indices = np.arange(len(transitions))
    for _ in range(config.update_epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), config.batch_size):
            batch_indices = indices[start : start + config.batch_size]
            logits, values = model(states[batch_indices])
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions[batch_indices])
            entropy = distribution.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs[batch_indices])
            unclipped = ratio * advantages[batch_indices]
            clipped = torch.clamp(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * advantages[batch_indices]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = (returns[batch_indices] - values).pow(2).mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()


def evaluate(env: LineWorldEnv, model: ActorCritic, episodes: int = 10) -> float:
    rewards = []
    for _ in range(episodes):
        state = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(state_tensor)
            action = int(torch.argmax(logits, dim=-1).item())
            state, reward, done = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    return float(np.mean(rewards))


def train() -> None:
    config = PPOConfig()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    env = LineWorldEnv()
    model = ActorCritic(state_dim=1, action_dim=3)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    for update in range(1, config.total_updates + 1):
        transitions = collect_rollout(env, model, config)
        returns, advantages = compute_returns_and_advantages(transitions, config)
        update_model(model, optimizer, transitions, returns, advantages, config)

        if update == 1 or update % 10 == 0:
            score = evaluate(env, model)
            print(f"update={update:03d} average_eval_reward={score:.3f}")


if __name__ == "__main__":
    train()
