# 目标
全天候模型

# Data
## 资产分布
现金
债券
股票
大宗商品
有色金属

## 市场分布
大A
港
美
欧
日
韩

## 数据分布
现金：K线
- 余额宝
- 朝朝宝

债券：K线
- [x] 短债券基金
- 中长债券基金
- 各市场国债基金

股票：K线，估值
- 各市场宽基指数基金
- - 中国ETF
- - - [x] 沪深300、中证500、中证1000
- - - 上证指数、上证50
- - - 深证成指、深证100
- - - 北证50
- - 港股ETF
- - - [x] 恒生、恒生科技
- - 美股ETF
- - - [x] 纳斯达克100
- - - 标普500、道琼斯
- - 欧股ETF
- - 日股ETF
- - 韩股ETF
- 各市场策略指数基金：
- - 中国
- - - [x] 中国红利
- - - [x] 中国现金流

大宗商品：K线
- 石油、天然气
- 多晶硅

有色金属：K线
- [x] 黄金
- 白银

宏观经济：K线
- 国债利率
- 通货膨胀数据（CPI、PPI）
- 就业数据

## 数据脚本

-------

**数据获取建议:**
- **个股**: K线+估值：BaoStock
- **ETF基金**: K线：腾讯财经；估值：用AkShare/东方财富获取对应跟踪标的指数的估值
- **商品/主动基金**: 腾讯财经
- **海外ETF (GLD等)**: yfinance (需要代理+设置 `CURL_CFFI_SSL_BACKEND=openssl`)


# Model
## Hand Craft
- [x] ALL-IN + Val set Top-K选股
- [x] MACD + Val set Top-K选股
- [x] MACD-周频月频
- <span style="color:gray">[ ] 钟摆/波浪理论</span>
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
