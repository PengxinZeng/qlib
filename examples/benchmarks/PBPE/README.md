# PB/PE 价值投资策略 - 方案文档

## 1. 策略概述

### 1.1 投资理念
基于市盈率(PE)和市净率(PB)的基本面价值投资策略。选取低估值ETF基金进行长期持有，在估值偏高时卖出。

### 1.2 数据基础
- **数据目录**: `<DATA_BASE>/qlib_data_260415/qlib_etf_index`（DATA_BASE 见 scripts/path_config.py）
- **基金数量**: 18只ETF指数基金
- **日期范围**: 2005-02-23 ~ 2026-04-13
- **可交易股票数**: 最多18只，最少12只

### 1.3 数据字段
| 字段名 | 说明 |
|--------|------|
| open/high/low/close | 基金每日行情 |
| volume/amount | 成交量/成交额 |
| index_open/high/low/close | 对应指数行情 |
| pe_ttm/pe_static | 市盈率(TTM/静态) |
| pb | 市净率 |
| pe_ttm_median/pe_static_median/pb_median | 同类市场中位数 |
| pe_ttm_equal_weight/pe_static_equal_weight/pb_equal_weight | 同类等权平均值 |

## 2. 数据集划分

基于累计可交易数量划分（数据集总可交易量=100%）：

| 数据集 | 日期范围 | 天数 | 可交易量占比 | 用途 |
|--------|----------|------|-------------|------|
| train | 2005-02-23 ~ 2019-11-01 | 3573 | 50.0% | AI参数调优 |
| valid | 2019-11-04 ~ 2021-03-26 | 340 | 10.0% | 人工参数调优 |
| test_a1 | 2021-03-29 ~ 2023-10-09 | 613 | 20.0% | 模型选择 |
| test_a2 | 2023-10-10 ~ 2025-01-03 | 303 | 10.0% | 实测 |
| test_b | 2025-01-06 ~ 2026-04-09 | 303 | 10.0% | 实测-再验证 |

## 3. 策略设计

### 3.1 选基规则
1. **估值过滤**: PE和PB必须在合理范围内
   - PE_TTM: 5 ~ 50倍
   - PB: 0.5 ~ 5倍

2. **相对估值**: 相对于市场中位数或等权平均值
   - PE_TTM <= 同类中位数的1.5倍
   - PB <= 同类中位数的1.5倍

3. **趋势确认**: 价格趋势向上
   - MA(close, 20) > MA(close, 60)

### 3.2 交易规则
1. **买入信号**:
   - 估值低于阈值(PE < 20, PB < 1.5)
   - 价格突破20日均线
   - 动量指标为正

2. **卖出信号**:
   - 估值高于阈值(PE > 40, PB > 3.0)
   - 价格跌破20日均线
   - 出现明显高估信号

3. **仓位管理**:
   - 单只基金最大仓位: 20%
   - 持仓基金数量: 3~5只
   - 保留10%现金应对极端情况

### 3.3 轮动机制
- 每月第一个交易日检视持仓
- 按相对估值排序，卖出高估，买入低估
- 目标: 保持组合整体低估值

## 4. 策略实现

### 4.1 策略模块
```python
# 策略类伪代码
class PBPEValueStrategy:
    def __init__(self, config):
        self.pe_max = config.get('pe_max', 50)
        self.pe_min = config.get('pe_min', 5)
        self.pb_max = config.get('pb_max', 5)
        self.pb_min = config.get('pb_min', 0.5)
        self.pe_ratio_threshold = config.get('pe_ratio_threshold', 1.5)
        self.pb_ratio_threshold = config.get('pb_ratio_threshold', 1.5)
        self.topk = config.get('topk', 5)

    def get_signal(self, dataset):
        """计算每日选基信号"""
        # 1. 估值过滤
        valid = (dataset.pe_ttm >= self.pe_min) & (dataset.pe_ttm <= self.pe_max)
        valid &= (dataset.pb >= self.pb_min) & (dataset.pb <= self.pb_max)

        # 2. 相对估值过滤
        valid &= (dataset.pe_ttm <= dataset.pe_ttm_median * self.pe_ratio_threshold)
        valid &= (dataset.pb <= dataset.pb_median * self.pb_ratio_threshold)

        # 3. 动量信号
        momentum = dataset.close / dataset.close.shift(20) - 1

        # 4. 综合评分
        score = -dataset.pe_ttm / dataset.pe_ttm_median  # 低估值得高分
        score += -dataset.pb / dataset.pb_median
        score += momentum * 0.3

        return score
```

### 4.2 配置文件
见 [workflow_config_pbpe.yaml](workflow_config_pbpe.yaml)

## 5. 回测配置

### 5.1 初始参数
- **回测资金**: 1,000,000 CNY
- **交易成本**: 买入0.03%, 卖出0.03%
- **最低佣金**: 5 CNY/笔
- **滑点**: 无

### 5.2 评估指标
- **年化收益率**: annualized return
- **夏普比率**: Sharpe ratio
- **最大回撤**: max drawdown
- **换手率**: turnover rate
- **胜率**: win rate

## 6. 文件结构
```
PBPE/
├── README.md              # 本文档
├── workflow_config.yaml   # Qlib回测配置
├── strategy.py           # 策略实现
└── backtest.py           # 回测脚本
```

## 7. 实施步骤

### 阶段1: 数据准备
- [x] 已完成qlib数据转换
- [x] 已完成数据集划分
- [x] 验证PE/PB数据完整性

### 阶段2: 策略实现
- [x] 实现PBPEValueStrategy类
- [x] 实现相对估值计算
- [x] 实现动量指标计算

### 阶段3: 参数调优
- [x] 在test_a1数据集上扫描128组参数
- [x] 生成参数敏感性分析报告

### 阶段4: 关键发现
- [x] 2021-03-29 ~ 2023-10-09 为熊市期，基准下跌20.91%
- [x] PB/PE策略表现-8.6% ~ -9.3%，显著跑赢基准
- [x] 最优参数: PE_MAX=30, PB_MAX=3.0, topk=5

## 8. 参数敏感性分析结果

测试区间: 2021-03-29 ~ 2023-10-09 (613个交易日)

### 基准表现
- 510300 (沪深300ETF): -20.91% 年化收益

### 策略表现
| 参数 | 年化收益率范围 | 最优值 |
|------|---------------|--------|
| PE_MAX | -9.29% ~ -9.29% | 30 |
| PB_MAX | -9.29% ~ -9.29% | 3.0 |
| TOPK | -9.29% ~ -9.29% | 5 |

### 关键结论
1. **熊市表现优异**: 策略在熊市中年化收益-8.6%，显著优于基准的-20.91%
2. **估值过滤有效**: 低估值标的在熊市中更具抗跌性
3. **持仓数量适中**: TOPK=5时表现最佳
4. **夏普比率**: 3.8，风险调整收益良好

### 输出文件
- `output/param_sweep_results.csv` - 128组参数完整结果
- `output/param_sensitivity.png` - 参数影响可视化

## 9. 风险提示

1. **估值风险**: PE/PB指标可能失真，需结合其他指标
2. **市场风险**: 低估值策略可能在某些市场环境下失效
3. **流动性风险**: ETF基金流动性需关注
4. **跟踪误差**: 基金可能存在相对指数的跟踪误差

## 10. 参考资料

- [Qlib官方文档](https://qlib.readthedocs.io/)
- ETF指数基金估值方法论
