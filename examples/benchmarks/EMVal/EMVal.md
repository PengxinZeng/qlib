### EMA 递推公式

EMA 是**指数加权移动平均**，对近期价格赋予更高权重：

```
α = 2 / (N + 1)                     # N 为周期（fast 或 mid 或 slow）

EMA[0] = close[0]                    # 初始值取第一个收盘价
EMA[t] = α × close[t] + (1−α) × EMA[t−1]
```

α 越大（N 越小），EMA 对最新价格反应越快。

### 估值线、溢价率、溢价分位

```
Val[t] = EMA_slow[t]
OverValPct[t] = (close[t] - Val[t]) / Val[t]
OverValRank = (OverValPct[t] >= OverValPct[t-Nt:t]).mean() # Nt为滚动窗口，避免分位钝化

```

### 趋势线

```
Diff[t] = EMA_fast[t] - EMA_mid[t]
```


### 买卖信号

```
SellRankThre = 0.75
SellSignal = （OverValRank >= SellRankThre）and (Diff[t] < −ε)

BuyRankThre = 0.05
BuySignal = （OverValRank < BuyRankThre）and (Diff[t] > +ε)

无信号→维持上一状态

```
