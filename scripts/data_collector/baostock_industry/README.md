# Baostock 行业数据采集器

基于 Baostock 获取 A 股行业分类信息和各行业股票列表。

## 特点

- 完全免费，无需 API Key
- 支持获取所有 A 股行业分类
- 包含各行业龙头股定义

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 获取所有行业信息

```bash
python collector.py get_industry_list
```

输出示例：
```
总股票数: 5512
行业数: 84

=== 各行业股票数量 ===
                        行业名称  股票数量
0  C39计算机、通信和其他电子设备制造业    651
1                  C35专用设备制造业    357
2        C26化学原料和化学制品制造业    352
...
```

### 2. 获取行业详情

```bash
# 获取所有行业详情
python collector.py get_industry_detail

# 获取指定行业详情
python collector.py get_industry_detail --industry "C27医药制造业"
```

### 3. 获取各行业龙头股

```bash
python collector.py get_industry_leaders
```

输出示例：
```
=== 各行业龙头股 ===

C15酒、饮料和精制茶制造业:
  600519 贵州茅台
  000858 五粮液
  000568 泸州老窖

C26化学原料和化学制品制造业:
  600309 万华化学
  000792 盐湖股份
  600486 扬农化工
...
```

## 主要行业龙头股一览

| 行业 | 龙头股 |
|------|--------|
| 白酒 | 贵州茅台(600519)、五粮液(000858) |
| 医药 | 恒瑞医药(600276)、药明康德(603259) |
| 新能源电池 | 宁德时代(300750) |
| 白电 | 美的集团(000333)、格力电器(000651) |
| 银行 | 工商银行(601398)、招商银行(600036) |
| 券商 | 中信证券(600030)、东方财富(300059) |
| 光伏 | 隆基绿能(601012) |
| 软件 | 金山办公(688111)、用友网络(600588) |

## Python 调用

```python
from scripts.data_collector.baostock_industry.collector import BaostockIndustryCollector

# 获取行业列表
df = BaostockIndustryCollector.get_industry_list()
print(df.head())

# 获取行业统计
summary = BaostockIndustryCollector.get_industry_summary(df)
print(summary)

# 获取龙头股
leaders = BaostockIndustryCollector.get_industry_leaders_df()
print(leaders)
```

## 数据字段说明

### query_stock_industry 返回字段

| 字段 | 说明 |
|------|------|
| code | 股票代码 |
| code_name | 股票名称 |
| industry | 所属行业 |
| list_date | 上市日期 |
