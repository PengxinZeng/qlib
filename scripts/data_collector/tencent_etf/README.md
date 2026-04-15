# 腾讯财经 ETF 数据采集器

基于腾讯财经接口采集 ETF 历史 K 线数据，支持完整历史数据和复权。

## 优势

- **完整历史数据**: 可获取 ETF 成立以来的全部历史数据
- **支持复权**: 后复权(hfq)、前复权(qfq)、不复权
- **突破限制**: 通过分段请求解决接口单次最多返回 800 条的限制

## 数据源对比

| 数据源 | 状态 | 说明 |
|--------|------|------|
| BaoStock | ⚠️ | ETF 仅返回近 3 个月数据 |
| 腾讯财经 | ✅ | 支持完整历史 + 复权，推荐 |
| 东方财富 | ❌ | 连接被拒绝 |
| 新浪财经 | ⚠️ | 仅不复权数据 |

## 使用方法

### 下载单只 ETF

```bash
python collector.py download_etf \
    --symbols "510050" \
    --source_dir ~/.qlib/stock_data/source/cn_etf_tencent \
    --fq_type hfq
```

### 下载多只 ETF

```bash
python collector.py download_etf \
    --symbols "510050,159915,518880" \
    --source_dir ~/.qlib/stock_data/source/cn_etf_tencent
```

### 从 funds_list.csv 批量下载

```bash
python collector.py download_etf \
    --funds_list /path/to/funds_list.csv \
    --source_dir ~/.qlib/stock_data/source/cn_etf_tencent
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--symbols` | ETF 代码，多个用逗号分隔 | - |
| `--funds_list` | funds_list.csv 文件路径 | - |
| `--source_dir` | 数据保存目录 | - |
| `--fq_type` | 复权类型: hfq=后复权, qfq=前复权, 空=不复权 | hfq |
| `--delay` | 请求间隔(秒) | 0.5 |

## 支持的 ETF 类型

- 上交所 ETF: 51xxxx, 58xxxx
- 深交所 ETF: 15xxxx, 56xxxx
