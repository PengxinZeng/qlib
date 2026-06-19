"""PPO Algorithm Configuration Management"""

from dataclasses import dataclass


@dataclass
class PPOConfig:
    """PPO Training Configuration
    
    Data Pipeline Parameters
    ========================
    n_stocks: int = 50
        Number of stocks in portfolio (M)
    n_kline_features: int = 5
        K-line feature dimension (OHLCV)
    n_valuation_features: int = 9
        Valuation feature dimension (PE, PB, ROE, DY, etc.)
    n_macro_features: int = 12
        Macro feature dimension (GDP_growth, Inflation, Interest_Rate, etc.)
    lookback_window: int = 1000
        Historical data window (days)
    
    Feature Extraction Parameters
    =============================
    projection_dim: int = 16
        Projection dimension for each modality (K-line/valuation/macro)
    num_heads: int = 8
        Number of attention heads
    head_dim: int = 4
        Dimension per head (32 / 8)
    feature_dim: int = 32
        Final feature dimension (2 × projection_dim)
    
    Model Parameters
    ================
    actor_hidden_dim: int = 64
        Actor network hidden layer dimension
    critic_hidden_dim: int = 64
        Critic network hidden layer dimension
    
    Training Parameters
    ===================
    rollout_days: int = 120
        Days collected per training update
    update_epochs: int = 4
        Training epochs per update round
    batch_size: int = 32
        Mini-batch size for gradient descent
    learning_rate: float = 1e-4
        Optimizer learning rate
    total_updates: int = 500
        Total number of training updates
    
    PPO Algorithm Parameters
    ========================
    gamma: float = 0.99
        Discount factor
    gae_lambda: float = 0.95
        GAE decay factor
    clip_ratio: float = 0.2
        PPO gradient clipping range
    
    Loss Weights
    ============
    value_coef: float = 0.5
        Weight for value loss
    entropy_coef: float = 0.01
        Weight for entropy regularization
    
    Trading Parameters
    ==================
    transaction_cost: float = 0.0005
        Trading cost rate (commission + slippage)
    
    Other Parameters
    ================
    seed: int = 42
        Random seed for reproducibility
    """

    # Data Pipeline
    n_stocks: int = 50
    n_kline_features: int = 5
    n_valuation_features: int = 9
    n_macro_features: int = 12
    lookback_window: int = 1000

    # Feature Extraction
    projection_dim: int = 16
    num_heads: int = 8
    head_dim: int = 4
    feature_dim: int = 32

    # Model
    actor_hidden_dim: int = 64
    critic_hidden_dim: int = 64

    # Training
    rollout_days: int = 120
    update_epochs: int = 4
    batch_size: int = 32
    learning_rate: float = 1e-4
    total_updates: int = 500

    # PPO
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2

    # Losses
    value_coef: float = 0.5
    entropy_coef: float = 0.01

    # Trading
    transaction_cost: float = 0.0005

    # Other
    seed: int = 42
    val_freq: int = 10      # validate every N updates
    patience: int = 10      # early stopping patience

    def __post_init__(self):
        """Validate configuration"""
        assert self.n_stocks > 0, "n_stocks must be positive"
        assert self.lookback_window > 0, "lookback_window must be positive"
        assert self.projection_dim > 0, "projection_dim must be positive"
        assert self.feature_dim == 2 * self.projection_dim, \
            f"feature_dim ({self.feature_dim}) must equal 2 × projection_dim ({2 * self.projection_dim})"
        assert self.head_dim == self.feature_dim // self.num_heads, \
            f"head_dim ({self.head_dim}) must equal feature_dim // num_heads ({self.feature_dim // self.num_heads})"
        assert 0 < self.gamma < 1, "gamma must be in (0, 1)"
        assert 0 < self.gae_lambda < 1, "gae_lambda must be in (0, 1)"
        assert self.clip_ratio > 0, "clip_ratio must be positive"
        assert self.value_coef >= 0, "value_coef must be non-negative"
        assert self.entropy_coef >= 0, "entropy_coef must be non-negative"
        assert 0 <= self.transaction_cost < 0.1, "transaction_cost must be in [0, 0.1)"
