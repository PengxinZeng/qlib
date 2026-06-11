# MACD 参数说明

MACD 由三个参数控制：`fast`、`slow`、`signal`。

---

## 参数含义

### `fast`（快线周期）
- 计算短期 EMA（指数移动平均）的窗口长度
- 对价格变化**反应快**，能较早捕捉趋势启动
- 数值越小，快线越灵敏，噪音越多

### `slow`（慢线周期）
- 计算长期 EMA 的窗口长度
- 代表**中长期趋势基准**
- DIF = EMA(fast) − EMA(slow)，即快线减慢线，衡量短期偏离长期的程度

### `signal`（信号线周期）
- 对 DIF 再做一次 EMA 平滑，得到 DEA（信号线）
- DEA 是 DIF 的**滞后平均**，起到平滑过滤的作用
- Histogram = DIF − DEA，代表两线的距离
- 数值越大，DEA 越平滑、越滞后，与 DIF 的距离越大，**金叉/死叉触发频率越低**

---

## 三者关系

### EMA 递推公式

EMA 是**指数加权移动平均**，对近期价格赋予更高权重：

```
α = 2 / (N + 1)                     # N 为周期（fast 或 slow）

EMA[0] = close[0]                    # 初始值取第一个收盘价
EMA[t] = α × close[t] + (1−α) × EMA[t−1]
```

α 越大（N 越小），EMA 对最新价格反应越快。

### DIF、DEA、Histogram

```
EMA_fast[t]  = α_fast  × close[t] + (1−α_fast)  × EMA_fast[t−1]
EMA_slow[t]  = α_slow  × close[t] + (1−α_slow)  × EMA_slow[t−1]

DIF[t]       = EMA_fast[t] − EMA_slow[t]          # 快线

β            = 2 / (signal + 1)
DEA[0]       = DIF[0]
DEA[t]       = β × DIF[t] + (1−β) × DEA[t−1]     # 对 DIF 再做一次 EMA

Histogram[t] = DIF[t] − DEA[t]                    # 柱状图（两线之差）
```

### 信号

```
score = +1  当 DIF > DEA（多头，买入）
score = -1  当 DIF < DEA（空头，平仓）
score =  0  当 DIF = DEA（极少出现，保持）
```

---

## 参数对交易频率的影响

| 调整方向 | 效果 |
|----------|------|
| 减小 `fast` | 快线更灵敏，信号更早但噪音更多 |
| 增大 `slow` | 趋势判断更保守，减少假信号 |
| **增大 `signal`** | **DEA 更平滑，DIF-DEA 间距更大，交易频率降低** |
| 减小 `signal` | DEA 紧跟 DIF，两线几乎贴合，反复触发买卖（见下图问题） |

---

## 当前配置（workflow_config.yaml）

```yaml
fast:   11
slow:   37
signal: 6
```

### 已知问题

`signal=6` 偏小，导致 DEA 快速追踪 DIF，两线几乎重合，在震荡行情中频繁触发金叉/死叉。

**建议调整**：将 `signal` 增大至 `9`（标准值）或更高，以减少无效交易。

---

## 标准参数参考

| 参数 | 标准值 | 本项目当前值 |
|------|--------|-------------|
| fast | 12 | 11 |
| slow | 26 | 37 |
| signal | 9 | 6 |
