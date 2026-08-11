# EMEnsemble - 多模型 Ensemble 综合持仓信号

## 简介

EMEnsemble 是一个多模型 Ensemble 交易算法：综合多个模型的交易信号，通过「模型解释器」将各模型原始交易信号解释为软交易信号，再通过「模型组装器」按各模型对各股票的拟合能力加权，得到最终每日持仓比例。

## 整体流程

```
各模型原始交易信号 (pred.pkl)
        │
        ▼
┌─────────────────────────────┐
│  模型解释器 (Interpreter)    │  逐个模型解释
│  - ValuationInterpreter     │  输出每日软交易信号
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  模型组装器 (Assembler)      │  综合各模型
│  - ReturnBasedAssembler     │  输出最终持仓比例
└─────────────────────────────┘
        │
        ▼
   持仓比例信号 (weight)
        │
        ▼
┌─────────────────────────────┐
│  TargetWeightStrategy       │  自动调仓达到持仓比例
└─────────────────────────────┘
```

## 组件说明

### 1. 模型解释器（Interpreter）

输入模型实验路径，输出模型每日软交易信号。

#### ValuationInterpreter（基于估值的解释器）

- **输入**：模型实验路径（mlruns 下某 run 目录，含 `artifacts/pred.pkl`）
- **输出**：每日软交易信号（`soft_signal`，取值 [0, 1]）
- **处理逻辑**：
  1. 模型交易信号为 `1`（买入）→ 输出 `1`
  2. 模型交易信号为 `-1`（卖出）→ 输出 `-1`
  3. 模型交易信号为 `0` → 根据「是否持有」与「估值情况」输出得分：
     - 若当前持有：根据估值分位 `over_val_rank` 在 `[buy_rank_thre, sell_rank_thre]` 区间线性映射得分（估值越低得分越高）
     - 若当前未持有：输出 `0`（不新买）

### 2. 模型组装器（Assembler）

输入各模型软信号 + 原始信号 + 股价，输出最终持仓比例。

#### ReturnBasedAssembler（基于收益情况的组装器）

- **输入**：各模型原始交易信号（`score`）+ 股价
- **输出**：最终持仓比例（`weight`，取值 [0, max_total_weight]）
- **处理逻辑**：
  1. 对每模型×每股票用**原始信号**×股价模拟交易计算收益率：`ret_t = signal_{t-1} × (price_t / price_{t-1} - 1)`
  2. 以收益率/回撤作为各模型对各股票拟合能力的指标（可配 `metric`：calmar/sharpe/annualized_return/total_return）
  3. 对同一股票跨模型做 softmax 归一化得到各模型权重
  4. 按权重加权**软交易信号**得到持仓比例，clip 到 `[0, max_total_weight]` 并每日缩放

### 3. 交易策略（Strategy）

#### TargetWeightStrategy

直接以信号值（持仓比例）作为目标权重，通过 `OrderGenWOInteract` 自动生成买卖订单，使实际持仓逐步调整到目标持仓比例。

## 配置说明

`workflow_config_all_weather.yaml` 中 `task.model` 配置：

```yaml
model:
    class: EMEnsembleModel
    module_path: qlib.contrib.model.em_ensemble
    kwargs:
        models:
            - name: emval_womom
              interpreter:
                  class: ValuationInterpreter
                  module_path: qlib.contrib.model.em_ensemble
                  kwargs:
                      exp_path: "/path/to/run1"      # 模型实验路径
                      buy_rank_thre: 0.07
                      sell_rank_thre: 0.94
            - name: emval
              interpreter:
                  class: ValuationInterpreter
                  module_path: qlib.contrib.model.em_ensemble
                  kwargs:
                      exp_path: "/path/to/run2"
                      buy_rank_thre: 0.32
                      sell_rank_thre: 0.88
        assembler:
            class: ReturnBasedAssembler
            module_path: qlib.contrib.model.em_ensemble
            kwargs:
                metric: calmar              # 拟合能力指标
                softmax_temperature: 1.0    # softmax 温度
                max_total_weight: 1.0       # 每日总仓位上限
```

## 运行

```bash
python -u qlib/cli/run.py examples/benchmarks/EMEnsemble/workflow_config_all_weather.yaml
```

## 当前 Ensemble 模型

| 模型名 | 实验路径 | buy_rank_thre | sell_rank_thre |
|--------|----------|---------------|----------------|
| emval_womom | `mlruns/264173616341034074/4fc3a7677d58403f870ebdb1166ed2a9` | 0.07 | 0.94 |
| emval | `mlruns/264173616341034074/7e4640b19d714b9c893c553e90c99428` | 0.32 | 0.88 |