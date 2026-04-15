# 新浪财经基金数据收集器

通过新浪财经API获取ETF/基金的历史每日K线数据，通过天天基金获取分红数据并自动计算复权价格。

**支持复权方式：**
- 前复权 (qfq) - 以最新价格为基准
- 后复权 (hfq) - 以上市价格为基准（默认）
- 不复权 (raw)

## 数据来源

| 数据类型 | 来源 |
|---------|------|
| K线数据 | 新浪财经 |
| 分红数据 | 天天基金 (eastmoney) |

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 下载全部基金数据（后复权）

```bash
python collector.py
```

### 下载指定基金

```bash
python collector.py --symbols 510300,159915
```

### 指定复权方式

```bash
# 前复权
python collector.py --symbols 510050 --adjust qfq

# 后复权（默认）
python collector.py --symbols 510050 --adjust hfq

# 不复权
python collector.py --symbols 510050 --adjust raw
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbols` | 无 | 指定基金代码，逗号分隔 |
| `--save_dir` | `./source` | 数据保存目录 |
| `--funds_list` | 默认路径 | 基金列表CSV文件路径 |
| `--datalen` | 9999 | 获取数据条数 |
| `--interval` | 1.0 | 下载间隔（秒） |
| `--adjust` | hfq | 复权方式: qfq/hfq/raw |

## 输出格式

数据按复权方式保存在不同子目录：
- `source/hfq/` - 后复权数据
- `source/qfq/` - 前复权数据
- `source/raw/` - 不复权数据

每个基金一个CSV文件，包含以下字段：
- `date`: 日期 (YYYY-MM-DD)
- `open`: 开盘价
- `high`: 最高价
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量
- `adjust_factor`: 复权因子

## 复权计算说明

后复权计算公式：
```
复权价格 = 不复权价格 × 累计复权因子
复权因子 = (除息前收盘价 - 每份分红) / 除息前收盘价
```

示例（510050 上证50ETF）：
- 分红记录：18次
- 早期复权因子：0.6907（累计分红约占原价31%）
- 2005-02-23 收盘价：不复权 0.876 → 后复权 0.6051
