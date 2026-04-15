# 基金历史K线数据采集器

基于 AkShare 的基金历史数据采集工具，支持获取基金从成立日期开始的完整历史数据。

> **注意**: 原本计划使用网易财经接口，但目前网易财经接口返回 502 错误不可用，因此改用 AkShare 作为数据源。

## 数据源

| 基金类型 | 数据源 | 数据类型 | 字段 |
|---------|-------|---------|------|
| 场内ETF (51xxxx, 58xxxx, 50xxxx, 15xxxx, 16xxxx, 56xxxx) | 新浪财经 | K线数据 | date, open, high, low, close, volume, amount |
| 场外基金 (00xxxx, 01xxxx, 02xxxx) | 天天基金 | 净值数据 | date, nav (累计净值) |

## 使用方法

### 安装依赖

```bash
pip install akshare pandas fire loguru tqdm
```

### 下载单只基金

```bash
python collector.py download_fund \
    --symbols "510050" \
    --source_dir ~/.qlib/fund_data/source/cn_fund
```

### 下载多只基金

```bash
python collector.py download_fund \
    --symbols "510050,159915,000217" \
    --source_dir ~/.qlib/fund_data/source/cn_fund
```

### 从 funds_list.csv 批量下载

```bash
python collector.py download_fund \
    --funds_list /path/to/funds_list.csv \
    --source_dir ~/.qlib/fund_data/source/cn_fund
```

## 参数说明

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| `--symbols` | 基金代码，多个用逗号分隔 | - |
| `--funds_list` | funds_list.csv 文件路径 | - |
| `--source_dir` | 数据保存目录 | (必需) |
| `--delay` | 请求间隔（秒） | 0.5 |

## funds_list.csv 格式

```csv
fund_code,fund_name,fund_type,track_target
510050,上证50ETF,ETF,上证50指数
159915,创业板ETF,ETF,创业板指数
000217,华安黄金ETF联接C,ETF联接,华安黄金ETF
```

- 以 `#` 开头的行为注释，会被忽略
- `fund_code` 为必需字段

## 输出文件格式

### ETF (K线数据)

```csv
date,open,high,low,close,volume,amount,preclose
2005-02-23,0.881,0.882,0.866,0.876,1269742542,1111793167,0.887
```

### 场外基金 (净值数据)

```csv
date,nav
2013-08-22,1.0
```

## 注意事项

1. 场内ETF数据为**不复权**数据
2. 场外基金返回的是**累计净值**，不是单位净值
3. 建议设置适当的请求间隔（delay），避免被封IP
4. 北证50等新市场的ETF可能暂不支持
