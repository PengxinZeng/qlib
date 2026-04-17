# Data
## 数据收集
- [x] 股票：白酒等行业龙头个股K线数据+估值
- [x] 指数ETF基金：K线数据+估值
- [ ] 其他基金（持有中）：短债、黄金、混合型、灵活理财

## 数据源对比
| 数据源 | 状态 | 说明 |
|--------|------|------|
| BaoStock | ⚠️ 有限制 | ETF仅返回近3个月数据 |
| 东方财富/AkShare/efinance | ❌ 不可用 | push2his.eastmoney.com 连接被断开 |
| 新浪财经 | ⚠️ 无复权 | 5138条数据，仅支持不复权 |
| 腾讯财经 | ✅ 推荐 | 5136条数据，支持后复权 |
| 网易财经 | ❌ 不可用 | 502 错误 |
| Tushare | ❌ 不可用 | 与新版pandas不兼容 |
| 天天基金 | ✅ 可用 | 有分红数据可以下载 |
| yfinance | ❌ 不可用 | 未安装 |
| 奇牛财经 | ❌ 不可用 | 格式错误 |

## 数据整理
- [ ] K线数据+估值
- [ ] 对齐首日收盘价为1.0
- [x] 训练测试集合划分

## 评测
- [ ] 设置不可交易日：比如茅台分红前；根据交易量

# Model
## Handy
- [x] ALL-IN + Val set Top-K选股
- [x] MACD + Val set Top-K选股
- [ ] KJD
- [ ] PB/PE
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

# Code Push
git add .
git commit -m "add index data collector"
git push origin main
