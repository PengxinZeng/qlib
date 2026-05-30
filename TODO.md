# Data
## 数据收集
- [x] 股票：白酒等行业龙头个股K线数据+估值
- [x] 指数ETF基金：K线数据+估值
- [ ] 其他基金（持有中）：短债、黄金、混合型、灵活理财
- [ ] AI：AI，AI算力，材料；龙头，新股，ETF，
- [ ] 恒生科技
- [ ] 日频数据更新

**数据获取建议:**
- **个股**: K线+估值：BaoStock
- **ETF基金**: K线：腾讯财经；估值：用AkShare/东方财富获取对应跟踪标的指数的估值
- **商品/主动基金**: 腾讯财经
- **海外ETF (GLD等)**: yfinance (需要代理+设置 `CURL_CFFI_SSL_BACKEND=openssl`)

## 数据整理
- [x] K线数据+估值
- [x] 首日设置为有效市场首日
- <span style="color:gray">[ ] 对齐首日收盘价为1.0</span>
- [x] 训练测试集合划分
- [x] 对齐最终日
- [x] 训练（AI参数调优），验证（人工参数调优），测试（模型选择）集合划分
  - train: 2005-02-23 ~ 2017-01-03 
  - valid: 2017-01-04 ~ 2019-10-08 
  - test: 2019-10-09 ~ 2026-04-09 
- [x] 设置不可交易日：比如茅台分红前；根据交易量
- [x] 慢牛行情与利率/国债关系
- [x] 慢牛行情与交易量关系：关联不强

## 评测
- [x] 换手率，操作频率
- <span style="color:gray">[ ] 基金动态卖出费率/买入费率</span>

# Model
## Hand Craft
- [x] ALL-IN + Val set Top-K选股
- [x] MACD + Val set Top-K选股
- [x] MACD-周频月频
- [ ] 钟摆/波浪理论
- [x] KDJ
- [x] PB/PE基本面策略：已完成策略设计与参数分析，熊市期年化-8.6%显著优于基准-20.91%
- [ ] 强化学习

```
实现MACD策略：
1. 在/Users/zengpengxin/workspace/CodeBase/qlib/examples/benchmarks新增文件目录实现策略；
2. 策略逻辑是：根据MACD选择买卖点；根据最终总收益选股
3. 数据集配置参考examples/benchmarks/LightGBM/workflow_config_lightgbm_etf.yaml
4. 在训练集中训练模型并保存训练日志、模型参数；在VAL set中调优参数
5. 在测试集中回测收益并生成分析报告

这是一个多步骤的策略实现任务，按照 Spec-Driven 流程进行。
```

## RL

# 其他
