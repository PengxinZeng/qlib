# Baostock 指数日线数据采集器

基于 Baostock API 的 A股指数K线数据采集工具，支持下载多种类型的指数数据。

## 支持的指数类型

| 类型 | 说明 | 示例代码 |
|------|------|---------|
| 综合指数 | 上证综指、深证综指 | sh.000001, sz.399106 |
| 规模指数 | 上证50、沪深300、中证500、深证成指 | sh.000016, sh.000300, sh.000905, sz.399001 |
| 一级行业指数 | 上证医药、国证交运等 | sh.000037, sz.399433 |
| 二级行业指数 | 300地产、300银行等 | sh.000952, sz.399951 |
| 策略指数 | 50等权、500等权等 | sh.000050, sh.000982 |
| 成长指数 | 小盘成长、大盘成长 | sz.399376, sz.399377 |
| 价值指数 | 180价值、300价值 | sh.000029, sz.399370 |
| 主题指数 | 红利指数、上证周期等 | sh.000015, sh.000063 |
| 基金指数 | 上证基金、深证基金 | sh.000011, sz.399305 |
| 债券指数 | 上证国债、上证企债 | sh.000012, sh.000013 |

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 下载所有预设指数数据

```bash
python collector.py download_index \
    --source_dir ~/.qlib/index_data/source \
    --end 2026-04-11
```

### 下载指定指数

```bash
python collector.py download_index \
    --symbols "sh.000300,sh.000016" \
    --source_dir ~/.qlib/index_data/source \
    --start 2015-01-01 \
    --end 2026-04-11
```

### 下载特定类型指数

```bash
# 下载规模指数
python collector.py download_size_index \
    --source_dir ~/.qlib/index_data/source

# 下载所有指数
python collector.py download_all_index \
    --source_dir ~/.qlib/index_data/source
```

### 标准化数据

```bash
python collector.py normalize_daily_data \
    --source_dir ~/.qlib/index_data/source \
    --normalize_dir ~/.qlib/index_data/normalize
```

### Python API 使用

```python
from baostock_index_daily.collector import BaostockIndexCollector

# 创建采集器
collector = BaostockIndexCollector(
    save_dir="~/.qlib/index_data/source",
    symbols=["sh.000300", "sh.000016"],
    start="2015-01-01",
    end="2026-04-11",
)

# 采集数据
collector.collector_index_data()

# 获取单只指数数据
df = BaostockIndexCollector.get_index_data_from_remote(
    symbol="sh.000300",
    start_datetime="2015-01-01",
    end_datetime="2026-04-11"
)
print(df.head())
```

## 数据字段

| 字段 | 说明 |
|------|------|
| date | 交易日期 |
| code | 证券代码 |
| open | 今开盘价 |
| high | 最高价 |
| low | 最低价 |
| close | 今收盘价 |
| preclose | 昨收盘价 |
| volume | 成交数量 |
| amount | 成交金额 |
| pctChg | 涨跌幅 |

## 预设指数代码

预设包含以下指数代码（可通过 `--symbols` 参数覆盖）：

- **综合指数**: sh.000001, sz.399106
- **规模指数**: sh.000016, sh.000300, sh.000905, sz.399001
- **一级行业**: sh.000037-sh.000042, sz.399433
- **二级行业**: sh.000952, sz.399951, sh.000913, sh.000914
- **策略指数**: sh.000050, sh.000982, sh.000926, sh.000923
- **成长指数**: sz.399376, sz.399377
- **价值指数**: sh.000029, sz.399370
- **主题指数**: sh.000015, sh.000063, sh.000065
- **基金指数**: sh.000011, sz.399305
- **债券指数**: sh.000012, sh.000013
