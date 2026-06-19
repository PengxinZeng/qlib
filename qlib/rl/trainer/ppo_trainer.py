"""PPO Trainer for Multi-Stock Trading

Implements Proximal Policy Optimization training loop.
"""

from typing import List, Dict
import torch
import torch.optim as optim

from ..models import MultiStockActorCritic, PPOConfig
from .collector import Collector, Transition
from .advantage import compute_returns_and_advantages


def _masked_dirichlet_entropy(concentration: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Compute Dirichlet entropy over valid dimensions only.

    concentration: (B, K), valid_mask: (B, K) bool
    Returns: (B,)
    """
    mask = valid_mask.float()
    c = concentration * mask + (1 - mask)          # set invalid dims to 1 to avoid log(0)
    alpha0 = (concentration * mask).sum(-1)        # sum of valid concentrations
    lgB = (torch.lgamma(c) * mask).sum(-1) - torch.lgamma(alpha0)
    digamma_adj = ((c - 1) * torch.digamma(c) * mask).sum(-1)
    return lgB + (alpha0 - mask.sum(-1)) * torch.digamma(alpha0) - digamma_adj


class PPOTrainer:
    """PPO trainer for multi-stock trading strategy"""

    def __init__(
        self,
        env,
        model: MultiStockActorCritic,
        config: PPOConfig,
        device: str = 'cpu',
    ):
        """Initialize PPO trainer
        
        Args:
            env: MultiStockDailyTradingEnv instance
            model: MultiStockActorCritic model
            config: PPOConfig instance
            device: 'cpu' or 'cuda'
        """
        self.env = env
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.collector = Collector(env, model, config)
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        self.train_step = 0

    def collect_rollouts(self) -> List[Transition]:
        """Collect rollouts for one training epoch
        
        Returns:
            transitions: List of Transition objects
        """
        # Collect enough transitions for batch updates
        num_rollouts = max(1, self.config.rollout_days // self.config.lookback_window)
        transitions = self.collector.collect_rollout()

        return transitions

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        transitions = self.collect_rollouts()

        if len(transitions) == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

        returns, advantages = compute_returns_and_advantages(
            transitions,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        # Stack all state tensors (skip 'dates')
        keys = [k for k in transitions[0].state.keys() if k != 'dates']
        state_tensors = {k: torch.cat([t.state[k] for t in transitions], dim=0) for k in keys}

        actions = torch.stack([t.action for t in transitions])          # (T, M+1)
        old_log_probs = torch.stack([t.log_prob for t in transitions])  # (T,)

        dataset = torch.utils.data.TensorDataset(
            *[state_tensors[k] for k in keys],
            actions, old_log_probs, returns, advantages,
        )

        metrics = {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
        num_batches = 0

        for epoch in range(self.config.update_epochs):
            loader = torch.utils.data.DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)

            for batch in loader:
                *state_vals, actions_b, old_lp_b, returns_b, adv_b = [x.to(self.device) for x in batch]
                state_dict = {k: v for k, v in zip(keys, state_vals)}

                logits, values = self.model(state_dict)

                # Dirichlet distribution
                concentration = torch.nn.functional.softplus(logits) + 1e-6
                dist = torch.distributions.Dirichlet(concentration)
                # clamp then renormalize to keep on simplex
                actions_safe = actions_b.clamp(min=1e-6)
                actions_safe = actions_safe / actions_safe.sum(dim=-1, keepdim=True)
                new_log_probs = dist.log_prob(actions_safe)
                valid_mask = concentration > 1e-6 + 1e-9  # (B, K)
                entropy = _masked_dirichlet_entropy(concentration, valid_mask).mean()

                ratio = torch.exp(new_log_probs - old_lp_b)
                ratio = torch.clamp(ratio, min=1e-8, max=1e3)  # Prevent Inf
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values - returns_b) ** 2).mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropy.item()
                num_batches += 1

        for key in metrics:
            metrics[key] /= (num_batches + 1e-8)

        self.train_step += 1
        return metrics

    def _evaluate(self, val_env) -> float:
        """Evaluate model on validation environment"""
        self.model.eval()
        state = val_env.reset()
        total_reward = 0.0

        while True:
            action, _, _ = self.model.act(state, deterministic=True)
            state, reward, done = val_env.step(action)
            total_reward += reward
            if done:
                break

        self.model.train()
        return total_reward

    def train(self, num_updates: int, val_env=None) -> Dict:
        """Train policy for multiple updates"""
        import numpy as np
        history = {'train_policy_loss': [], 'train_value_loss': [], 'train_entropy': [], 'val_reward': []}
        best_val_reward = -np.inf
        patience_counter = 0
        val_freq = self.config.val_freq
        patience   = self.config.patience

        for update in range(1, num_updates + 1):
            metrics = self.train_epoch()
            history['train_policy_loss'].append(metrics['policy_loss'])
            history['train_value_loss'].append(metrics['value_loss'])
            history['train_entropy'].append(metrics['entropy'])

            if val_env is not None and update % val_freq == 0:
                val_reward = self._evaluate(val_env)
                history['val_reward'].append(val_reward)
                print(f"Update {update}/{num_updates} val_reward={val_reward:.4f}")
                if val_reward > best_val_reward:
                    best_val_reward = val_reward
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at update {update}")
                    break

            if update % val_freq == 0:
                print(f"Update {update}/{num_updates}: policy_loss={metrics['policy_loss']:.4f}, "
                      f"value_loss={metrics['value_loss']:.4f}, entropy={metrics['entropy']:.4f}")

        return history

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_step': self.train_step,
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_step = checkpoint['train_step']
