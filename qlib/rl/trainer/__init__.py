# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Train, test, inference utilities."""

try:
    from .api import backtest, train
except ImportError:
    # tianshou not installed, skip api
    backtest = None
    train = None

from .callbacks import Checkpoint, EarlyStopping, MetricsWriter
from .trainer import Trainer
from .vessel import TrainingVessel, TrainingVesselBase

# PPO Multi-Stock Trading
from .collector import Collector, Transition
from .advantage import compute_returns_and_advantages
from .ppo_trainer import PPOTrainer

__all__ = [
    "Trainer",
    "TrainingVessel",
    "TrainingVesselBase",
    "Checkpoint",
    "EarlyStopping",
    "MetricsWriter",
    "train",
    "backtest",
    # PPO Multi-Stock
    "Collector",
    "Transition",
    "compute_returns_and_advantages",
    "PPOTrainer",
]
