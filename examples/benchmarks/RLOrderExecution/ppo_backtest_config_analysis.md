# `backtest_ppo.yml` 配置分析

本文分析 `examples/rl_order_execution/exp_configs/backtest_ppo.yml` 在 RL Order Execution 场景下的输入、模型、loss、reward 与推理/回测流程，并补充与其配套训练配置 `train_ppo.yml` 的关系。

## 1. 配置定位

`backtest_ppo.yml` 是 PPO 订单执行策略的回测配置：

- 回测入口：`python -m qlib.rl.contrib.backtest --config_path exp_configs/backtest_ppo.yml`
- 回测订单：`./data/orders/test_orders.pkl`
- 回测输出：`outputs/ppo/backtest_result.csv`
- 策略层级：`1day` 层使用 RL 策略 `SAOEIntStrategy`，`30min` 层使用规则策略 `TWAPStrategy`
- 数据粒度：底层行情为 `5min`，RL 决策步长由外层 `30min` executor 驱动

需要注意：`backtest_ppo.yml` 本身只负责加载策略和执行回测，不配置训练 loss，也不配置 reward。PPO 的 reward 和 loss 来自训练配置 `train_ppo.yml` 与 `qlib.rl.order_execution.policy.PPO` / Tianshou `PPOPolicy`。

## 2. 输入数据

### 2.1 订单输入

配置项：

```yaml
order_file: ./data/orders/test_orders.pkl
start_time: "9:30"
end_time: "14:54"
```

`order_file` 被 `qlib.rl.contrib.backtest.backtest()` 读取为订单 DataFrame。每条订单通常包含：

- `datetime`：订单日期
- `instrument`：股票代码
- `amount`：需要执行的总股数/数量
- `direction`：买卖方向，映射为 `OrderDir`

回测代码会对每条订单构造 `qlib.backtest.decision.Order`：

```text
Order(stock_id, amount, direction, start_time, end_time)
```

其中 `start_time` / `end_time` 会替换为订单当天日期内的 `9:30` 到 `14:54`。

### 2.2 行情输入

配置项：

```yaml
qlib:
  provider_uri_5min: ./data/bin/
data_granularity: "5min"
exchange:
  deal_price: ["$close", "$close"]
```

含义：

- `./data/bin/`：Qlib 5 分钟行情数据目录
- `data_granularity: "5min"`：底层撮合/成交指标使用 5 分钟频率
- `deal_price: ["$close", "$close"]`：买卖两个方向都使用 `$close` 作为成交价格
- `limit_threshold: null`、`volume_threshold: null`：不启用涨跌停/成交量阈值限制

### 2.3 模型观察输入

`SAOEIntStrategy` 使用 `FullHistoryStateInterpreter` 将 `SAOEState` 转成 PPO 网络观察：

```yaml
state_interpreter:
  class: FullHistoryStateInterpreter
  kwargs:
    data_dim: 5
    data_ticks: 48
    max_step: 8
    processed_data_provider:
      class: HandlerProcessedDataProvider
      kwargs:
        data_dir: ./data/pickle/
        feature_columns_today: ["$high", "$low", "$open", "$close", "$volume"]
        feature_columns_yesterday: ["$high_1", "$low_1", "$open_1", "$close_1", "$volume_1"]
```

`FullHistoryStateInterpreter` 输出的 observation 主要包括：

- `data_processed`：当天特征，shape 约为 `[48, 5]`，包含 `$high/$low/$open/$close/$volume`；未来时刻会被 mask 为 0，避免信息泄露
- `data_processed_prev`：前一交易日特征，shape 约为 `[48, 5]`
- `acquiring`：是否为买入方向，买入为 1，否则为 0
- `cur_tick`：当前 5 分钟 tick 位置
- `cur_step`：当前 30 分钟决策步位置
- `num_step`：总决策步数，上限为 8
- `target`：原始订单总量
- `position`：当前剩余未执行数量
- `position_history`：历史各决策步执行后的剩余仓位/订单量轨迹

在该配置下，一天可用 5 分钟 tick 为 `48`，RL 决策被分成 `8` 个 30 分钟 step。

## 3. 动作空间与动作解释

配置项：

```yaml
action_interpreter:
  class: CategoricalActionInterpreter
  kwargs:
    max_step: 8
    values: 4
```

`values: 4` 会生成离散动作映射：

```text
action_values = [0, 0.25, 0.5, 0.75, 1.0]
```

PPO 策略输出一个离散动作 `a ∈ {0,1,2,3,4}`，动作解释器将其转换为本步计划执行数量：

```text
exec_amount = min(current_position, order.amount * action_values[a])
```

如果已经到最后一个 step，即 `cur_step >= max_step - 1`，动作解释器强制返回全部剩余 `position`，确保订单在最后一步被尽量执行完。

## 4. 模型结构

### 4.1 策略模型

配置项：

```yaml
network:
  class: Recurrent
  kwargs: {}
policy:
  class: PPO
  kwargs:
    lr: 0.0001
```

回测时 `SAOEIntStrategy` 会按如下顺序构建模型：

1. 用 `FullHistoryStateInterpreter.observation_space` 初始化 `Recurrent`
2. 用 `CategoricalActionInterpreter.action_space` 初始化 PPO 策略
3. PPO 策略内部创建 actor 和 critic
4. 如果 `policy.kwargs.weight_file` 配置了 checkpoint，则加载训练权重

当前 `backtest_ppo.yml` 中 `weight_file` 是注释状态：

```yaml
# weight_file: outputs/ppo/checkpoints/latest.pth
```

因此如果不取消注释，回测会使用随机初始化模型，结果没有实际意义。正式回测应在训练完成后改为：

```yaml
policy:
  class: PPO
  kwargs:
    lr: 0.0001
    weight_file: outputs/ppo/checkpoints/latest.pth
```

### 4.2 `Recurrent` 特征抽取网络

`qlib.rl.order_execution.network.Recurrent` 是一个 GRU/RNN/LSTM 风格的时序特征抽取器，默认参数为：

- `hidden_dim = 64`
- `output_dim = 32`
- `rnn_type = "gru"`
- `rnn_num_layers = 1`

网络将输入拆成三类 source：

1. **当天市场序列**：`data_processed` 经过 `raw_fc` 后进入 `raw_rnn`，取当前 `cur_tick` 对应的 hidden state
2. **私有订单状态序列**：`position_history / target` 与归一化 step 组成 `[position_ratio, step_ratio]`，经过 `pri_fc` 与 `pri_rnn`，取当前 `cur_step` hidden state
3. **买卖方向特征**：`acquiring` 与 `1 - acquiring` 组成方向 one-hot，经 `dire_fc` 编码

三路特征拼接后进入 `fc`：

```text
[market_feature, private_position_feature, direction_feature]
  -> Linear/ReLU/Linear/ReLU
  -> output_dim=32
```

### 4.3 Actor-Critic

`qlib.rl.order_execution.policy.PPO` 是 Tianshou `PPOPolicy` 的包装：

- `PPOActor`：`Recurrent` 输出 32 维特征后，接 `Linear(32, action_dim)` 和 `Softmax`，输出离散动作概率
- `PPOCritic`：`Recurrent` 输出 32 维特征后，接 `Linear(32, 1)`，输出状态价值 `V(s)`
- actor 与 critic 共享同一个 `Recurrent` extractor，优化器会对共享参数去重
- 动作分布为 `torch.distributions.Categorical`
- 回测/评估默认 `deterministic_eval=True`

## 5. Loss 设计

`backtest_ppo.yml` 不训练模型，所以没有运行 loss。训练时使用 `train_ppo.yml` 中的 PPO 配置：

```yaml
policy:
  class: PPO
  kwargs:
    lr: 0.0001
trainer:
  max_epoch: 500
  repeat_per_collect: 25
  episode_per_collect: 10000
  batch_size: 1024
```

实际 loss 由 Tianshou `PPOPolicy` 实现，`qlib.rl.order_execution.policy.PPO` 传入的关键参数为：

- `discount_factor = 1.0`
- `gae_lambda = 1.0`
- `eps_clip = 0.3`
- `value_clip = True`
- `vf_coef = 1.0`
- `reward_normalization = True`
- `max_grad_norm = 100.0`
- `max_batch_size = 256`

概念上 PPO 优化目标为：

```text
policy_loss = - E[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]
value_loss  = MSE(Vθ(s_t), return_t) 或 clipped value loss
entropy     = 策略熵正则
loss        = policy_loss + vf_coef * value_loss - entropy_coef * entropy
```

其中：

- `r_t(θ) = πθ(a_t|s_t) / π_old(a_t|s_t)`
- `A_t` 由 GAE 根据 reward 与 value 估计得到
- 该实现中 `eps_clip=0.3`，即策略更新比例被限制在 `[0.7, 1.3]` 附近

## 6. Reward 设计

### 6.1 回测配置中的 reward

`backtest_ppo.yml` 不配置 reward。回测只加载训练好的策略并执行，最终使用 Qlib backtest 指标（如 `pa`、`ffr` 等）评价执行效果。

### 6.2 PPO 训练 reward

与 `backtest_ppo.yml` 配套的 `train_ppo.yml` 配置了：

```yaml
reward:
  class: PPOReward
  kwargs:
    max_step: 8
    start_time_index: 0
    end_time_index: 46
```

`PPOReward` 的特点是：

- 非终止 step reward 为 `0.0`
- 到最后一步或订单剩余 `position < 1e-6` 时，根据最终执行均价与 TWAP 价格比较给出离散 reward
- 如果全程没有成交，使用普通平均市场价格作为 `vwap_price`
- 如果有成交，使用成交量加权市场价格作为 `vwap_price`

计算逻辑：

```text
SELL: ratio = vwap_price / twap_price
BUY : ratio = twap_price / vwap_price

ratio < 1.0  -> reward = -1.0
ratio < 1.1  -> reward =  0.0
otherwise    -> reward =  1.0
```

含义：

- 对卖单，执行均价越高越好，因此 `vwap_price / twap_price` 越大越好
- 对买单，执行均价越低越好，因此 `twap_price / vwap_price` 越大越好
- reward 是 episode 末端稀疏奖励，训练信号集中在订单完成/最后一步

## 7. 推理与回测流程

### 7.1 总体流程

运行命令：

```bash
python -m qlib.rl.contrib.backtest --config_path exp_configs/backtest_ppo.yml
```

主流程：

1. `get_backtest_config_fromfile()` 读取 YAML，并补充默认交易成本、现金限制、输出目录等配置
2. `read_order_file()` 读取 `test_orders.pkl`
3. 按股票拆分订单，并用 `joblib.Parallel` 并行回测，配置中 `concurrency: 16`
4. 每个股票池子内调用 `single_with_collect_data_loop()`
5. 构造多层 executor：
   - 最内层：`SimulatorExecutor`，频率 `5min`
   - 中间层：`NestedExecutor` + `TWAPStrategy`，频率 `30min`
   - 外层：`NestedExecutor` + `SAOEIntStrategy`，频率 `1day`
6. 使用 Qlib `collect_data_loop()` 推动回测
7. 汇总 `indicator_dict["1day"]` 中的订单执行指标
8. 将 `pa` 放大 `10000` 倍，对齐训练指标尺度
9. 写出 `outputs/ppo/backtest_result.csv`

### 7.2 每个 30 分钟决策步的推理细节

`SAOEIntStrategy._generate_trade_decision()` 是 PPO 推理的核心：

1. 从外层 trade decision 中取出当前订单
2. 通过 `SAOEStateAdapter` 获取当前 `SAOEState`
3. `FullHistoryStateInterpreter` 将 state 转为 observation
4. 把 observation 包成 Tianshou `Batch`
5. `self._policy(Batch(obs_batch))` 前向推理，得到离散动作 `act`
6. `CategoricalActionInterpreter` 将离散动作转为计划执行量 `exec_vol`
7. 用 Qlib `OrderHelper` 创建本 step 的子订单
8. 返回 `TradeDecisionWithDetails`，其中 `details` 记录：
   - `instrument`
   - `datetime`
   - `freq`
   - `rl_exec_vol`
   - `rl_action`

### 7.3 执行状态更新

每个 step 执行后，`SAOEStateAdapter.update()` 会根据执行结果更新状态：

- 从 executor/exchange 获取真实成交量、市场价格、市场成交量
- 计算本 step 的成交金额、成交均价、剩余 position、FFR、PA 等指标
- 追加到 `history_exec` 与 `history_steps`
- 减少 `position`
- 推进 `cur_time`

下一次 PPO 推理时，新的 `position`、`position_history`、`cur_step` 和历史执行信息会进入 observation。

## 8. 配置关系总结

| 模块 | `backtest_ppo.yml` 配置 | 代码实现 | 作用 |
|---|---|---|---|
| 订单输入 | `order_file` | `qlib.rl.contrib.utils.read_order_file` | 读取测试订单 |
| 行情输入 | `qlib.provider_uri_5min`, `data_granularity`, `exchange` | Qlib backtest/exchange | 提供 5min 价格、成交量与成交价格 |
| 状态解释 | `FullHistoryStateInterpreter` | `qlib.rl.order_execution.interpreter` | 将 SAOEState 转为模型输入 |
| 动作解释 | `CategoricalActionInterpreter(values=4,max_step=8)` | `qlib.rl.order_execution.interpreter` | 将离散动作映射到执行数量 |
| 网络 | `Recurrent` | `qlib.rl.order_execution.network` | 抽取市场、订单私有状态、方向特征 |
| 策略 | `PPO(lr=0.0001)` | `qlib.rl.order_execution.policy` | Actor-Critic PPO 策略 |
| Reward | 回测不配置；训练用 `PPOReward` | `qlib.rl.order_execution.reward` | 训练阶段根据最终 VWAP/TWAP 比较给稀疏奖励 |
| Loss | 回测不配置；训练由 Tianshou PPO 实现 | `tianshou.policy.PPOPolicy` | clipped policy loss + value loss + entropy |
| 推理输出 | `outputs/ppo/backtest_result.csv` | `qlib.rl.contrib.backtest` | 输出订单执行指标 |

## 9. 使用注意事项

1. 回测前必须训练并配置 `weight_file`，否则 PPO 策略随机初始化，回测结果不可解释。
2. `backtest_ppo.yml` 的策略结构要与训练 checkpoint 的结构一致，包括 `state_interpreter`、`action_interpreter`、`network` 和 `policy` 参数。
3. `data_ticks=48`、`max_step=8`、`data_granularity=5` 与交易时间窗口强相关，修改交易时间或行情频率时需要同步调整。
4. 回测使用更真实的 Qlib backtest/exchange 流程，训练阶段测试使用的 simulator 可能不同，因此 README 中也提示训练测试结果与独立 backtest 结果可能不完全一致。
5. `pa` 在回测输出时会乘以 `10000`，用于与训练指标尺度对齐。