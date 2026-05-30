# Qlib ETF Index Benchmark Summary

## 1. 数据集信息

### 数据路径
```
/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415/qlib_etf_index_Extend_wBond
```

### 数据构成
- **市场范围**: All (全市场)
- **基准指数**: 510050_CLEAN (上证50ETF)
- **数据周期**: 2005-02-23 至 2026-04-13
- **数据字段**:
  - 价格数据: open, high, low, close, volume
  - 估值指标: pb (市净率), pb_median (市净率中位数)
  - 盈利指标: pe_ttm (市盈率TTM), pe_ttm_median (市盈率TTM中位数)
  - 宏观数据: cn_2y (中国2年期国债收益率)

### 数据划分

| 数据集 | 起始时间 | 结束时间 | 天数 | 说明 |
|--------|----------|----------|------|------|
| **Train** | 2005-02-23 | 2017-01-03 | ~12年 | 训练集 |
| **Valid** | 2017-01-04 | 2019-10-08 | ~2.8年 | 验证集 |
| **Test** | 2019-10-09 | 2026-04-13 | ~6.5年 | 测试集 |

---

## 2. 方法与指标

| 方法 | Train+Valid<br>Annualized Return | Train+Valid<br>Information Ratio | Train+Valid<br>Max Drawdown | Test<br>Annualized Return | Test<br>Information Ratio | Test<br>Max Drawdown | Avg Turnover Rate | Rebalance Day Ratio |
|------|----------------------------------|----------------------------------|----------------------------|--------------------------|--------------------------|---------------------|-------------------|---------------------|
| Benchmark (510050) | 0.093 | 0.351 | -0.703 | 0.015 | 0.101 | -0.365 | - | - |
| MACD | 0.150 | 0.669 | -0.519 | 0.014 | 0.090 | -0.393 | 0.359 | 0.486 |
| HistRelaPB | 0.140 | 0.857 | -0.337 | 0.086 | 0.554 | -0.232 | 0.010 | 0.014 |

---

## 3. 实验结论


## 4. 补充说明

### 指标定义
- **Annualized Return (年化收益率)**: 策略的年化收益率
- **Information Ratio (信息比率)**: 收益与波动的比值，衡量风险调整后的收益能力
- **Max Drawdown (最大回撤)**: 策略从峰值到谷底的最大跌幅
- **Avg Turnover Rate (平均换手率)**: 每次调仓时持仓变动的平均比例，反映交易频繁程度；值越小表示每次调仓改动越少
- **Rebalance Day Ratio (调仓日比率)**: 实际发生调仓的交易日占总交易日的比例；值越小表示调仓越不频繁

