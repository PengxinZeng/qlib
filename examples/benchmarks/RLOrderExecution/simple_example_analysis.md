# `examples/rl/simple_example.ipynb` 配置解析

本文总结 `examples/rl/simple_example.ipynb` 中 Qlib RL 最小示例的输入、模型、loss、Reward 与训练/推理流程。该 notebook 不是订单执行真实业务配置，而是用一个极简环境演示 Qlib RL 的组件拼装方式：`Simulator`、`StateInterpreter`、`ActionInterpreter`、`Reward`、`PPO policy`、`Dataset`、`train` 与手工 backtest。

---

## 1. 任务定义

### 环境目标

示例环境维护一个固定的 `value`，智能体每一步输出一个动作，动作越接近 `value`，即时 reward 越高。

该例子的最优策略非常简单：每一步都选择最大动作，即 `action = value`，对应 reward 为 `1.0`。因此它主要用于解释 Qlib RL 框架接口，而不是验证复杂策略能力。

### Episode 长度

```python
NSTEPS = 10
```

每个 simulator 固定运行 10 步：

- `SimpleSimulator.remain_steps` 初始化为 `nsteps`
- 每次 `step()` 后减 1
- `done()` 在 `remain_steps == 0` 时结束 episode

---

## 2. 输入配置

### 初始状态输入

训练时的初始状态由 `SimpleDataset` 提供：

```python
SimpleDataset([10.0, 50.0, 100.0])
```

每个数字都会作为 `SimpleSimulator(initial=position, nsteps=NSTEPS)` 的初始 `value`，因此训练环境包含 3 类初始规模：

- `10.0`
- `50.0`
- `100.0`

### Simulator 状态

状态定义为：

```python
State = namedtuple("State", ["value", "last_action"])
```

字段含义：

- `value`：环境中的固定数值，动作上界也是该值
- `last_action`：上一步实际传入 simulator 的 float 动作，用于计算 reward

### Policy 观察输入

`SimpleStateInterpreter` 将 simulator 状态转换为模型可接收的 observation：

```python
return np.array([state.value], dtype=np.float32)
```

因此模型输入是 1 维连续特征：

```text
obs = [value]
observation_space = Box(0, +inf, shape=(1,), dtype=float32)
```

注意：`last_action` 虽然在 `State` 里，但不会进入 policy 输入，只用于 reward 计算。

### Policy 动作空间

`SimpleActionInterpreter(n_value=10)` 定义离散动作空间：

```python
action_space = Discrete(11)
```

policy 输出的动作是整数 `0 ~ 10`，再被解释为 simulator 可执行的 float 动作：

```python
sim_action = simulator_state.value * (policy_action / 10)
```

对应关系如下：

| policy action | simulator action |
|---:|---:|
| `0` | `0.0 * value` |
| `1` | `0.1 * value` |
| `5` | `0.5 * value` |
| `10` | `1.0 * value` |

---

## 3. 模型配置

### 网络结构

notebook 使用自定义的全连接特征提取网络 `SimpleFullyConnect(dims=[16, 8])`：

```text
input_dim = 1
Linear(1, 16) + ReLU
Linear(16, 8) + ReLU
output_dim = 8
```

该网络本身只输出 8 维 hidden feature，不直接输出动作。动作头和价值头由 `qlib.rl.order_execution.PPO` 包装器自动创建。

### PPO Actor

`PPO` 内部用 `PPOActor` 包装网络：

```text
SimpleFullyConnect: obs -> feature[8]
Linear(8, action_dim=11)
Softmax(dim=-1)
Categorical distribution
```

含义：

- Actor 输出 11 个离散动作的概率
- 分布类型是 `torch.distributions.Categorical`
- 训练时从 categorical policy 中采样动作
- 评估/推理默认 `deterministic_eval=True`，倾向选择确定性动作

### PPO Critic

`PPO` 内部同时创建 `PPOCritic`：

```text
SimpleFullyConnect: obs -> feature[8]
Linear(8, 1)
```

Critic 输出当前 observation 的状态价值：

```text
V(s)
```

### 优化器与超参数

notebook 显式配置：

```python
lr = 0.01
```

`PPO` 包装器默认参数包括：

```text
discount_factor = 1.0
reward_normalization = True
eps_clip = 0.3
value_clip = True
vf_coef = 1.0
gae_lambda = 1.0
max_grad_norm = 100.0
max_batch_size = 256
deterministic_eval = True
```

---

## 4. Reward 设计

Reward 类为 `SimpleReward`：

```python
rew = simulator_state.last_action / simulator_state.value
```

因为 `last_action` 来自动作解释器：

```python
last_action = value * (policy_action / 10)
```

所以 reward 等价于：

```text
reward = policy_action / 10
```

奖励范围：

```text
[0.0, 1.0]
```

关键特点：

- reward 只依赖上一动作与 `value` 的比例
- 不依赖市场行情、成交成本、风险或长期收益
- 每一步即时 reward 独立，环境的 `value` 不会被 action 改变
- 最优离散动作是 `10`，即执行 `100% * value`

---

## 5. Loss 设计

notebook 没有手写 loss，而是使用 `qlib.rl.order_execution.PPO`，其继承自 Tianshou 的 `PPOPolicy`。

该 loss 属于标准 PPO actor-critic 目标，核心由三部分组成：

### Policy clipped surrogate loss

```text
ratio = exp(log_prob_new - log_prob_old)
policy_loss = -mean(min(ratio * advantage,
                        clip(ratio, 1 - eps_clip, 1 + eps_clip) * advantage))
```

本示例中：

```text
eps_clip = 0.3
```

### Value loss

Critic 学习状态价值 `V(s)`，目标是 rollout return：

```text
value_loss = MSE(V(s), return)
```

本示例中：

```text
value_clip = True
vf_coef = 1.0
```

### Advantage / Return

PPO 使用 GAE 计算 advantage：

```text
discount_factor = 1.0
gae_lambda = 1.0
```

在这个极简环境里，由于每一步 reward 与动作直接相关，且 `value` 不随动作变化，学习信号会非常直接：更大的离散动作带来更高 reward。

---

## 6. 训练流程

训练入口：

```python
train(
    simulator_fn=lambda position: SimpleSimulator(position, NSTEPS),
    state_interpreter=state_interpreter,
    action_interpreter=action_interpreter,
    policy=policy,
    reward=reward,
    initial_states=SimpleDataset([10.0, 50.0, 100.0]),
    trainer_kwargs=trainer_kwargs,
    vessel_kwargs=vessel_kwargs,
)
```

### Trainer 配置

```python
trainer_kwargs = {
    "max_iters": 10,
    "finite_env_type": "dummy",
    "callbacks": [
        Checkpoint(
            dirpath=Path("./checkpoints"),
            every_n_iters=1,
            save_latest="copy",
        )
    ],
}
```

含义：

- 最多训练 10 个 iteration
- 使用 `dummy` 类型 finite env
- 每个 iteration 保存 checkpoint
- 最新 checkpoint 复制为 latest 文件

### Vessel 配置

```python
vessel_kwargs = {
    "update_kwargs": {"batch_size": 16, "repeat": 5},
    "episode_per_iter": 50,
}
```

含义：

- 每个 iteration 收集 50 个 episode
- policy update 时 batch size 为 16
- 每批数据重复优化 5 次

### 单步交互顺序

Qlib RL 通过 `EnvWrapper` 把 simulator、interpreter、reward 组装成 gym-style environment。每一步顺序如下：

```text
1. simulator.get_state()
2. state_interpreter(state) -> obs
3. policy(obs) -> policy_action，离散整数 0~10
4. action_interpreter(state, policy_action) -> simulator_action，float
5. simulator.step(simulator_action)
6. simulator.get_state()
7. reward(new_state) -> reward
8. 将 obs/action/reward/done 交给 PPO 更新
```

---

## 7. 推理 / Backtest 流程

notebook 的推理示例是手工执行 1 步 backtest，而不是调用 `qlib.rl.trainer.backtest` API。

代码流程：

```python
simulator = SimpleSimulator(100.0, NSTEPS)
state = simulator.get_state()
obs = [{"obs": state_interpreter.interpret(state)}]
policy_out = policy(Batch(obs))
act = float(action_interpreter.interpret(state, policy_out.act))

simulator.step(act)
rew = float(reward(simulator.get_state()))
```

### 推理输入

```text
initial value = 100.0
obs = [100.0]
```

### 模型输出

```text
policy_out.act = 离散动作，范围 0~10
```

### 动作解释

```text
act = 100.0 * (policy_out.act / 10)
```

如果训练充分，期望输出：

```text
policy_out.act ≈ 10
act ≈ 100.0
reward ≈ 1.0
```

### 推理结算

```text
simulator.step(act)
reward = last_action / value
```

最终打印：

```python
print(f"Action = {act:.6f}, Reward = {rew:.6f}.")
```

---

## 8. 与真实 RLOrderExecution 的差异

该 notebook 使用的是 `qlib.rl.order_execution.PPO` 包装器，但环境不是实际订单执行环境。

主要差异：

| 维度 | simple_example | 真实订单执行 RL |
|---|---|---|
| 状态 | 单个 `value` | 订单、时间、成交量、价格、剩余量、市场特征等 |
| 动作 | 离散比例 `0~10` | 每个时间片的下单比例或执行量 |
| Reward | `last_action / value` | 成交收益、价格偏差、成本、风险等 |
| 环境动态 | `value` 不变 | 剩余订单、成交状态和市场状态持续变化 |
| 目标 | 学习选最大动作 | 在市场约束下优化执行成本/收益 |

因此，该 notebook 更适合作为 Qlib RL 接口模板：

```text
Simulator + StateInterpreter + ActionInterpreter + Reward + PPO + train/backtest
```

而不是可直接迁移到真实交易的策略逻辑。

---

## 9. 总结

`simple_example.ipynb` 的核心配置可以概括为：

```text
输入：obs = [value]，shape=(1,)
动作：Discrete(11)，表示 0%~100% 的 value 比例
模型：MLP(1 -> 16 -> 8) + PPO 自动 Actor/Critic head
Reward：last_action / value，即 action 桶编号 / 10
Loss：Tianshou PPO clipped policy loss + value loss + GAE advantage
训练：3 个初始 value，10 个 iteration，每轮收集 50 个 episode
推理：obs -> PPO policy -> 离散动作 -> float 动作 -> simulator.step -> reward
```

最终学习目标是让 PPO policy 在任意给定 `value` 下选择最大离散动作 `10`，从而获得最高 reward `1.0`。
