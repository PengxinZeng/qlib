# N股训练 → M股推理：跨股票池泛化方案

这是一个关键的实用问题：**用 N 支股票的历史数据训练模型，能否在 M 支不同的股票上做推理？**

答案是：**可以，但需要特殊设计**。

---

## 问题分析

### 为什么不能直接迁移？

假设训练时用 5 只股票（沪深 300 成分股），推理时要用 10 只股票（创业板）：

```python
# 训练时
train_stocks = ["SH600000", "SH600001", "SH600010", "SH601398", "SH601988"]
n_train = 5

# 推理时（新股票池）
test_stocks = ["SZ000858", "SZ000651", "SZ300750", ...]
m_test = 10

# ❌ 直接用会失败
action_space = spaces.Box(low=-∞, high=∞, shape=(5,))  # 硬编码了 n=5
# 但推理时需要 shape=(10,)，维度不匹配！
```

### 核心障碍

| 组件 | 问题 | 原因 |
|---|---|---|
| **动作空间** | 维度硬编码 | 权重向量长度固定为 N |
| **网络输入** | 特征矩阵维度 | [batch, N, feature_dim] 无法接收 [batch, M, feature_dim] |
| **特征分布** | 不同股票的统计特性差异 | A股与创业板的波动率、流动性不同 |

---

## 解决方案

### 方案 1：股票无关的网络架构（推荐 ⭐）

**核心思想**：设计**不依赖股票数量 N 的网络**，对任意 M 都适用

#### 1.1 用注意力机制（Attention）处理可变长输入

```python
class StockAgnosticNetwork(nn.Module):
    """
    对任意数量的股票输入都能处理
    核心：Attention 自动对齐和聚合
    """
    
    def __init__(self, feature_dim: int = 12, hidden_dim: int = 64):
        super().__init__()
        
        # 特征提取（单股级别）
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 跨股票聚合（不依赖股票数）
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )
        
        # 全局汇聚
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 对任意长度求平均
        
        # 输出（独立于股票数）
        self.output_head = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, stock_features: torch.Tensor) -> torch.Tensor:
        """
        Input:  [batch, num_stocks, feature_dim]  ← num_stocks 可变！
        Output: [batch, hidden_dim]                ← 固定维度
        """
        
        # Step 1: 单股特征提取
        # [batch, num_stocks, feature_dim] → [batch, num_stocks, hidden_dim]
        embedded = self.feature_extractor(stock_features)
        
        # Step 2: 跨股票注意力（对任意长度都适用）
        # [batch, num_stocks, hidden_dim] → [batch, num_stocks, hidden_dim]
        attn_out, _ = self.self_attention(embedded, embedded, embedded)
        
        # Step 3: 全局汇聚（对任意长度都适用）
        # [batch, num_stocks, hidden_dim] → [batch, hidden_dim]
        pooled = self.global_pool(attn_out.transpose(1, 2)).squeeze(-1)
        
        # Step 4: 输出
        out = self.output_head(pooled)  # [batch, hidden_dim]
        
        return out
```

**优点** ✓：
- ✓ 无论 N 还是 M，都能处理
- ✓ 自动学习股票间重要性
- ✓ 无需修改代码

#### 1.2 动作空间设计：多资产权重向量（股票 + 现金）

**我们选择的建模方式**：动作直接表示组合权重，包含当前股票池里的 M 支股票，以及 1 个现金仓位。

```text
action = [w_stock_1, w_stock_2, ..., w_stock_M, w_cash]
```

参考 PPO 时，Actor 先输出 M+1 个动作分布参数，在训练阶段从 logits 空间采样动作，再通过 `softmax` 转成合法权重：

```text
raw_action ~ πθ(raw_action | state)
weights = softmax(raw_action)
sum(weights) = 1
weights_i >= 0
```

推理阶段可以直接使用 Actor 输出的均值 logits 做 `softmax`。这样现金也是一个可学习资产：当模型认为股票机会不足或风险过高时，可以把更多权重分配给现金。

```python
class StockCashWeightInterpreter:
    """
    将网络输出的 M+1 维 logits 解释为：
    - 前 M 维：当前股票池中每只股票的目标权重
    - 最后 1 维：现金权重
    """
    
    def interpret(self, obs, logits: np.ndarray) -> Dict[str, float]:
        """
        obs: 当前观察，包含 stock_ids
        logits: 网络输出的 M+1 维未归一化分数
        
        Output: 股票目标权重 + 现金权重
        """
        
        stock_ids = obs["stock_ids"]
        assert len(logits) == len(stock_ids) + 1
        
        exp_logits = np.exp(logits - np.max(logits))
        weights = exp_logits / exp_logits.sum()
        
        target_weights = {
            stock_id: weights[i]
            for i, stock_id in enumerate(stock_ids)
        }
        target_weights["CASH"] = weights[-1]
        
        return target_weights
```

**动作空间**：
```python
# 对 M 支股票推理时，动作维度为 M+1：M 个股票权重 + 1 个现金权重
action_space = spaces.Box(low=0.0, high=1.0, shape=(m + 1,))

# 模型输出
policy_out = model(obs)  # [batch, m, hidden_dim] 或经过 per-stock head 的表示
actor_head = nn.Linear(hidden_dim, 1)
stock_logits = actor_head(policy_out).squeeze(-1)  # [batch, m]

cash_head = nn.Linear(hidden_dim, 1)
pooled = policy_out.mean(dim=1)                    # [batch, hidden_dim]
cash_logit = cash_head(pooled)                     # [batch, 1]

logits = torch.cat([stock_logits, cash_logit], dim=-1)  # [batch, m+1]
action = torch.softmax(logits, dim=-1)                  # [batch, m+1]
```

**优点** ✓：
- ✓ 动作天然满足非负、权重和为 1 的组合约束
- ✓ 现金仓位可学习，能表达空仓/防守状态
- ✓ 对任意 M 支股票，只需输出 M 个股票 logits + 1 个现金 logit

---

## PPO 策略建模

这里不再生成监督学习的 GT 权重标签，而是参考 PPO，把模型作为随机策略 `πθ(a|s)` 训练：状态 `s_t` 输入网络，策略采样组合权重 `a_t`，环境根据下一期真实行情返回 reward。

### Actor-Critic 结构

Actor 输出 `M+1` 维动作分布参数，Critic 输出当前状态价值 `V(s)`：

```python
class PortfolioActorCritic(nn.Module):
    def __init__(self, feature_dim: int = 12, hidden_dim: int = 64):
        super().__init__()
        self.extractor = StockAgnosticNetwork(feature_dim, hidden_dim)
        self.stock_mu_head = nn.Linear(hidden_dim, 1)
        self.cash_mu_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
    
    def forward(self, features: torch.Tensor):
        stock_emb = self.extractor(features)                 # [batch, m, hidden_dim]
        stock_mu = self.stock_mu_head(stock_emb).squeeze(-1) # [batch, m]
        pooled = stock_emb.mean(dim=1)                       # [batch, hidden_dim]
        cash_mu = self.cash_mu_head(pooled)                  # [batch, 1]
        mu_logits = torch.cat([stock_mu, cash_mu], dim=-1)   # [batch, m+1]
        value = self.value_head(pooled).squeeze(-1)          # [batch]
        return mu_logits, self.log_std.exp(), value
```

### 动作采样

训练时使用 logistic-normal 策略：先在 logits 空间采样，再 `softmax` 成合法组合权重；推理时直接对均值 logits 做 `softmax`。

```python
def sample_action(mu_logits: torch.Tensor, std: torch.Tensor):
    dist = torch.distributions.Normal(mu_logits, std)
    raw_action = dist.rsample()                       # [batch, m+1]
    weights = torch.softmax(raw_action, dim=-1)       # [batch, m+1]
    log_prob = dist.log_prob(raw_action).sum(dim=-1)  # PPO ratio 使用
    entropy = dist.entropy().sum(dim=-1)
    return weights, raw_action, log_prob, entropy

# inference
action = torch.softmax(mu_logits, dim=-1)
```

---

## Reward 设计

Reward 不来自人工标签，而来自执行动作后的真实组合表现。对第 `t` 日动作 `w_t`，用下一期或未来窗口收益计算：

```text
w_t = [w_stock_1, ..., w_stock_M, w_cash]
r_{t+1} = [r_stock_1, ..., r_stock_M, r_cash]
```

### 单步组合收益

```python
portfolio_return = (weights * next_returns).sum(dim=-1)
```

### 加入交易成本

权重变化越大，换手越高；交易成本直接从 reward 中扣除：

```python
turnover = torch.abs(weights - prev_weights).sum(dim=-1)
cost = transaction_cost_rate * turnover
```

### 加入风险惩罚

可以用组合波动、回撤或下行波动作为惩罚项。简单版本用横截面风险暴露近似：

```python
risk = (weights[:, :-1] * stock_volatility).sum(dim=-1)
```

### 最终 Reward

```python
def compute_reward(
    weights: torch.Tensor,
    next_returns: torch.Tensor,
    prev_weights: torch.Tensor,
    stock_volatility: torch.Tensor,
    transaction_cost_rate: float = 0.001,
    risk_penalty_weight: float = 0.01,
) -> torch.Tensor:
    portfolio_return = (weights * next_returns).sum(dim=-1)
    turnover = torch.abs(weights - prev_weights).sum(dim=-1)
    cost = transaction_cost_rate * turnover
    risk = (weights[:, :-1] * stock_volatility).sum(dim=-1)
    return portfolio_return - cost - risk_penalty_weight * risk
```

现金收益作为 `next_returns` 的最后一维进入 reward，因此模型可以在股票预期风险收益差时主动提高现金权重。

---

## PPO Loss 设计

PPO 优化的是采样策略相对旧策略的 clipped surrogate objective，不需要 `GT Label`。

### Advantage 计算

使用 GAE 计算优势函数：

```python
def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    advantages = []
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages.insert(0, gae)
        next_value = values[t]
    advantages = torch.stack(advantages)
    returns = advantages + values
    return advantages, returns
```

### PPO clipped loss

```python
def ppo_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    entropy: torch.Tensor,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
) -> torch.Tensor:
    ratio = torch.exp(new_log_prob - old_log_prob)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    policy_loss_1 = ratio * advantages
    policy_loss_2 = torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
    policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
    value_loss = torch.nn.functional.mse_loss(values, returns)
    entropy_bonus = entropy.mean()
    return policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
```

---

## 完整实现流程

### 训练阶段（N 支股票）

```python
import numpy as np
import torch

# 1. 定义 PPO actor-critic
policy = PortfolioActorCritic(feature_dim=12, hidden_dim=64)
rollout_buffer = []
prev_weights = init_equal_cash_weights(n_stocks=len(train_stocks))  # [n+1]

# 2. 与环境交互，收集 rollout
for date in train_dates:
    features = get_features(train_stocks, date)                       # [n, feature_dim]
    next_returns = get_next_returns_with_cash(train_stocks, date)     # [n+1]
    stock_volatility = get_stock_volatility(train_stocks, date)       # [n]
    
    mu_logits, std, value = policy(torch.tensor(features).unsqueeze(0))
    weights, raw_action, log_prob, entropy = sample_action(mu_logits, std)
    reward = compute_reward(
        weights=weights,
        next_returns=torch.tensor(next_returns).unsqueeze(0),
        prev_weights=prev_weights,
        stock_volatility=torch.tensor(stock_volatility).unsqueeze(0),
    )
    
    rollout_buffer.append({
        "features": features,
        "raw_action": raw_action.detach(),
        "old_log_prob": log_prob.detach(),
        "value": value.detach(),
        "reward": reward.detach(),
        "done": is_episode_end(date),
    })
    prev_weights = weights.detach()

# 3. 用 GAE 计算 advantage / return
advantages, returns = compute_gae(
    rewards=torch.stack([x["reward"] for x in rollout_buffer]),
    values=torch.stack([x["value"] for x in rollout_buffer]),
    dones=torch.tensor([x["done"] for x in rollout_buffer]),
)

# 4. PPO 多 epoch 更新
for batch in iterate_minibatches(rollout_buffer, advantages, returns):
    mu_logits, std, values = policy(batch["features"])
    dist = torch.distributions.Normal(mu_logits, std)
    new_log_prob = dist.log_prob(batch["raw_action"]).sum(dim=-1)
    entropy = dist.entropy().sum(dim=-1)
    loss = ppo_loss(
        new_log_prob=new_log_prob,
        old_log_prob=batch["old_log_prob"],
        advantages=batch["advantages"],
        returns=batch["returns"],
        values=values,
        entropy=entropy,
    )
    loss.backward()
    optimizer.step()
```

### 推理阶段（M 支股票）

```python
trained_policy = torch.load("trained_policy.pth")
test_stocks = ["SZ000858", "SZ000651", "SZ300750", ...]  # M 支新股票

for date in test_dates:
    features_today = get_features(test_stocks, date)
    features_tensor = torch.tensor(features_today).unsqueeze(0)  # [1, m, feature_dim]
    
    with torch.no_grad():
        mu_logits, _, _ = trained_policy(features_tensor)  # [1, m+1]
        logits_np = mu_logits.squeeze(0).cpu().numpy()      # [m+1]
    
    action_interp = StockCashWeightInterpreter()
    target_weights = action_interp.interpret(
        obs={"stock_ids": test_stocks},
        logits=logits_np,
    )
    
    print(f"{date}: {target_weights}")
```

---

## 预期效果与注意事项

### 能否直接推理新股票？

| 情况 | 可行性 | 原因 | 处理方式 |
|---|---|---|---|
| **同样的市场、同样风格** | ✓ 高 | 特征分布相近，网络按单股特征和横截面关系打分 | 直接推理 |
| **股票数量变化 N → M** | ✓ 高 | 股票 logits 是逐股输出，现金 logit 是全局输出 | 输出 M+1 维权重 |
| **完全不同的资产类（股票 vs 债券）** | ❌ 低 | 特征含义和收益结构不同 | 不建议直接使用 |

### 关键注意事项

1. **训练没有 GT 权重标签**：PPO 通过 reward 和 advantage 更新策略，不做未来收益 soft label 监督。
2. **reward 可以使用下一期行情**：下一期收益只在环境结算时用于计算 reward，不能进入当前状态特征。
3. **现金收益要进入 reward**：现金是第 `M+1` 个资产，否则模型无法学习空仓/防守。
4. **交易成本要直接扣 reward**：否则策略会倾向于频繁切换到短期最优股票。
5. **股票顺序必须对齐**：`features`、`weights`、`next_returns` 的前 `M` 维必须使用同一 `stock_ids` 顺序。

---

## 推荐方案（综合）

```python
# 1. 网络设计：股票无关 actor-critic
policy = PortfolioActorCritic(feature_dim=12, hidden_dim=64)

# 2. 动作建模：logistic-normal 采样 + softmax 到 N/M 股票 + 1 现金权重
mu_logits, std, value = policy(features)
weights, raw_action, log_prob, entropy = sample_action(mu_logits, std)

# 3. Reward：组合收益 - 交易成本 - 风险惩罚
reward = compute_reward(weights, next_returns, prev_weights, stock_volatility)

# 4. Loss：PPO clipped policy loss + value loss - entropy bonus
advantages, returns = compute_gae(rewards, values, dones)
loss = ppo_loss(new_log_prob, old_log_prob, advantages, returns, values, entropy)
```

---

## 扩展阅读

- Domain Adaptation / Transfer Learning
- Meta-Learning（MAML）在多资产中的应用
- 时间序列的分布适配
- 在线学习与在线调整
