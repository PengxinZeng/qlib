# Baostock 日线数据采集器

基于 Baostock 获取中国 A 股日线数据和基本面数据。

## 特点

- 完全免费，无需 API Key
- 支持获取所有 A 股日线数据
- 支持获取基本面数据（盈利能力、营运能力、成长能力、偿债能力）
- 大陆访问稳定

## 安装

```bash
pip install baostock
```

## 使用方法

### 1. 下载日线数据

```bash
python collector.py download_daily_data \
    --source_dir ~/.qlib/stock_data/source/cn_daily \
    --start 2020-01-01 \
    --end 2024-12-31 \
    --delay 0.5 \
    --max_workers 4
```

### 2. 下载基本面数据

```bash
python collector.py download_fundamental_data \
    --source_dir ~/.qlib/stock_data/source/cn_fundamental \
    --start 2020-01-01 \
    --end 2024-12-31 \
    --delay 0.5 \
    --include_types profit,operation,growth,balance
```

### 3. 标准化数据

```bash
python collector.py normalize_daily_data \
    --source_dir ~/.qlib/stock_data/source/cn_daily \
    --normalize_dir ~/.qlib/stock_data/source/cn_daily_nor
```

### 4. 导出到 Qlib 格式

```bash
python dump_bin.py dump_all \
    --data_path ~/.qlib/stock_data/source/cn_daily_nor \
    --qlib_dir ~/.qlib/qlib_data/cn_data \
    --freq day \
    --exclude_fields date,symbol
```

## Baostock 基本数据类型

| 类型 | API 方法 | 包含字段 | 说明 |
|------|----------|----------|------|
| profit | query_profit_data | roeAvg, npMargin, gpMargin, netProfit, epsTTM, totalShare, liqaShare | 盈利能力 |
| operation | query_operation_data | NRTurnRatio, NRTurnDays, INVTurnRatio, INVTurnDays, CATurnRatio, AssetTurnRatio | 营运能力 |
| growth | query_growth_data | YOYEquity, YOYAsset, YOYNI, YOYEPSBasic, YOYPNI | 成长能力 |
| balance | query_balance_data | currentRatio, quickRatio, cashRatio, YOYLiability, liabilityToAsset, assetToEquity | 偿债能力 |
| dupont | query_dupont_data | dupontROE, dupontAssetStoEquity, dupontAssetTurn, dupontPnitoni, dupontNitogr | 杜邦分析 |

## 关于 PB 数据

**Baostock 不提供直接的 PB（市净率）数据。**

如需 PB、PE（市盈率）、PS（市销率）等日线级别数据，建议使用以下方案：

| 数据源 | PB | PE | PS | 费用 |
|--------|----|----|-----|------|
| **Tushare Pro** | ✓ | ✓ | ✓ | 积分制 (2000积分≈200元/年) |
| **Baostock** | ✗ | ✗ | ✗ | 免费（无日线级别估值数据） |

### Tushare Pro 获取 PB 数据示例

```python
import tushare as ts

# 初始化
pro = ts.pro_api('your_token')

# 获取日线基本数据（包含 PB、PE、PS）
df = pro.daily_basic(
    ts_code='000001.SZ',
    start_date='20200101',
    end_date='20241231',
    fields='ts_code,trade_date,close,pb,pe_ttm,ps_ttm,total_mv,circ_mv'
)
print(df.head())
```

## 数据字段说明

### profit (盈利能力)

| 字段 | 说明 |
|------|------|
| code | 股票代码 |
| pubDate | 发布日期 |
| statDate | 统计日期 |
| roeAvg | 净资产收益率(平均) |
| npMargin | 净利率 |
| gpMargin | 毛利率 |
| netProfit | 净利润 |
| epsTTM | 每股收益(TTM) |
| totalShare | 总股本 |
| liqaShare | 流通股本 |

### operation (营运能力)

| 字段 | 说明 |
|------|------|
| NRTurnRatio | 应收账款周转率 |
| NRTurnDays | 应收账款周转天数 |
| INVTurnRatio | 存货周转率 |
| INVTurnDays | 存货周转天数 |
| CATurnRatio | 流动资产周转率 |
| AssetTurnRatio | 资产周转率 |

### growth (成长能力)

| 字段 | 说明 |
|------|------|
| YOYEquity | 净资产增长率 |
| YOYAsset | 总资产增长率 |
| YOYNI | 净利润增长率 |
| YOYEPSBasic | 每股收益同比增长率 |
| YOYPNI | 归属净利润同比增长率 |

### balance (偿债能力)

| 字段 | 说明 |
|------|------|
| currentRatio | 流动比率 |
| quickRatio | 速动比率 |
| cashRatio | 现金比率 |
| YOYLiability | 负债增长率 |
| liabilityToAsset | 资产负债率 |
| assetToEquity | 权益乘数 |
