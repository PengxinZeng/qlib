"""Feature Extraction for Multi-Stock Trading

Four-layer progressive feature extraction:
1. Projectors - Independent K-line/valuation/macro projection
2. Position Encoding - Relative date RoPE encoding
3. Stock Aggregation - Per-stock multi-head self-attention
4. Cross-Stock Interaction - Multi-head attention + holdings embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


class MaskedMultiHeadAttention(nn.Module):
    """Multi-head attention with NaN/Inf key masking, residual + FFN.

    forward(q, k, v, key_mask):
        q: (N, Sq, d) or (N, d)
        k, v: (N, Sk, d)
        key_mask: (N, Sk) bool, True = invalid (will be masked)
    Returns: same shape as q, with residual applied
    """

    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.feature_dim = feature_dim
        self.scale = np.sqrt(self.head_dim)

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.ffn = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                key_mask: torch.Tensor = None,
                q_pos: torch.Tensor = None, k_pos: torch.Tensor = None,
                apply_q_pos=None, apply_k_pos=None) -> torch.Tensor:
        """
        q_pos/k_pos: additive position embedding (N, S, d)
        apply_q_pos/apply_k_pos: callable(x: (N,S,d)) -> (N,S,d) for non-additive pos (e.g. RoPE)
        """
        N, h, hd = q.shape[0], self.num_heads, self.head_dim
        Sq, Sk = q.shape[1], k.shape[1]

        q_in = q if q_pos is None else q + q_pos
        k_in = k if k_pos is None else k + k_pos
        if apply_q_pos is not None:
            q_in = apply_q_pos(q_in)
        if apply_k_pos is not None:
            k_in = apply_k_pos(k_in)
        Q = self.q_proj(self.norm(q_in)).reshape(N, Sq, h, hd)  # pre-norm on q+pos
        K = self.k_proj(k_in).reshape(N, Sk, h, hd)
        V = self.v_proj(v).reshape(N, Sk, h, hd)

        logits = torch.einsum('nshd,nthd->nhst', Q, K) / self.scale
        if key_mask is not None:
            logits = logits.masked_fill(key_mask.unsqueeze(1).unsqueeze(2), float('-1e6'))  # clamp to finite value instead of -inf
        attn = F.softmax(logits, dim=-1)

        out = torch.einsum('nhst,nthd->nshd', attn, V).reshape(N, Sq, self.feature_dim)
        assert torch.all(torch.isfinite(out)), out
        return q + F.relu(self.ffn(out))  # residual + FFN


class FeatureExtractor(nn.Module):
    """Multi-layer feature extraction for multi-stock trading
    
    Input: {
        'kline':     (batch, M, lookback_window, n_kline_features),
        'valuation': (batch, M, lookback_window, n_valuation_features),
        'macro':     (batch, M, lookback_window, n_macro_features),
        'holdings':  (batch, M+1) optional
        'last_buy_days': (batch, M) optional
    }
    
    Output: (batch, M, feature_dim=32)
    """

    def __init__(
        self,
        n_kline_features: int = 5,
        n_valuation_features: int = 9,
        n_macro_features: int = 12,
        lookback_window: int = 1000,
        projection_dim: int = 16,
        num_heads: int = 8,
        feature_dim: int = 32,
    ):
        super().__init__()

        self.lookback_window = lookback_window
        self.projection_dim = projection_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.feature_dim = feature_dim

        # Layer 1: Data Projectors
        self.proj_kline = nn.Linear(n_kline_features, projection_dim)
        self.proj_valuation = nn.Linear(n_valuation_features, projection_dim)
        self.proj_macro = nn.Linear(n_macro_features, projection_dim)
        self.proj_combine = nn.Linear(3 * projection_dim, feature_dim)

        # Layer 3: Per-stock attention (learnable query aggregates history)
        self.stock_query_emb = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.per_stock_attn = MaskedMultiHeadAttention(feature_dim, num_heads)

        # Layer 4: Cross-stock attention
        self.cross_stock_attn = MaskedMultiHeadAttention(feature_dim, num_heads)
        self.holdings_pos_proj = nn.Linear(2, feature_dim)

    def forward(
        self,
        state: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Extract features from multi-modal data
        
        Args:
            state: Dict with 'kline', 'valuation', 'macro' tensors
                   Shape: (batch, M, lookback_window, n_feat)
        
        Returns:
            features: (batch, M, feature_dim)
        """
        kline = state['kline']      # (batch, M, lookback, n_kline)
        valuation = state['valuation']
        macro = state['macro']
        holdings = state.get('holdings', None)
        last_buy_days = state.get('last_buy_days', None)

        batch_size, n_stocks = kline.shape[0], kline.shape[1]
        bm = batch_size * n_stocks

        # Flatten to (batch*M, lookback, feat) for per-stock ops
        def flatten(x): return x.reshape(bm, x.shape[2], x.shape[3])

        kline_flat = flatten(kline)         # (bm, lookback, n_kline)
        val_flat   = flatten(valuation)
        mac_flat   = flatten(macro)

        # Record key_mask BEFORE nan_to_num — NaN means "no data this timestep"
        # A timestep is invalid if ANY modality has NaN
        key_mask = (
            ~torch.isfinite(kline_flat).all(dim=-1) |
            ~torch.isfinite(val_flat).all(dim=-1) |
            ~torch.isfinite(mac_flat).all(dim=-1)
        )  # (bm, T)

        # Clean NaN BEFORE projection to prevent gradient corruption
        kline_flat = torch.nan_to_num(kline_flat, nan=0.0)
        val_flat   = torch.nan_to_num(val_flat, nan=0.0)
        mac_flat   = torch.nan_to_num(mac_flat, nan=0.0)

        # Layer 1: Project each modality
        combined = torch.cat([
            self.proj_kline(kline_flat),
            self.proj_valuation(val_flat),
            self.proj_macro(mac_flat),
        ], dim=-1)  # (batch*M, lookback, 3*proj_dim)
        combined = self.proj_combine(combined)  # (batch*M, lookback, feature_dim)

        # Layer 2+3: Per-stock attention with RoPE on K only (Q is learnable, no pos needed)
        q = self.stock_query_emb.expand(bm, 1, -1)  # (bm, 1, fd)
        stock_features = self.per_stock_attn(
            q, combined, combined, key_mask,
            apply_k_pos=self._apply_rope,
        ).squeeze(1)  # (bm, fd)

        # Unflatten back to (batch, M, feature_dim)
        stock_features = stock_features.reshape(batch_size, n_stocks, self.feature_dim)

        # Layer 4: Cross-stock attention with holdings position embedding on Q/K only
        holdings_norm = holdings[:, :-1] / (holdings[:, :-1].sum(dim=1, keepdim=True) + 1e-8)
        days_norm = 1.0 / (last_buy_days + 1)
        pos_embedding = self.holdings_pos_proj(
            torch.stack([holdings_norm, days_norm], dim=-1)
        )  # (batch, M, fd)
        if torch.isnan(stock_features).any():
            print(f"[DEBUG] stock_features before cross_stock has NaN: {torch.isnan(stock_features).sum()}")
        
        key_mask_cross = key_mask.reshape(batch_size, n_stocks, -1).all(dim=-1)  # (batch, M)

        out = self.cross_stock_attn(stock_features, stock_features, stock_features, key_mask_cross,
                                     q_pos=pos_embedding, k_pos=pos_embedding)
        
        if torch.isnan(out).any():
            print(f"[DEBUG] cross_stock_attn output has NaN: {torch.isnan(out).sum()}")
        
        return (out, key_mask_cross)

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """RoPE on (batch*M, seq_len, feature_dim)"""
        seq_len = x.shape[1]
        relative_pos = torch.arange(seq_len, dtype=x.dtype, device=x.device) - (seq_len - 1)
        inv_freq = 1.0 / (10000.0 ** (
            torch.arange(0, self.feature_dim, 2, dtype=x.dtype, device=x.device) / self.feature_dim
        ))
        freqs = torch.einsum('i,j->ij', relative_pos, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return x * emb.cos().unsqueeze(0) + x * emb.sin().unsqueeze(0)
