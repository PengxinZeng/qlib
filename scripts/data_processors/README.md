# 数据处理与可视化流程

## 环境要求

```bash
conda activate rdagent
```

## 目录结构

```
scripts/
├── data_processors/
│   ├── merge_etf_val/
│   │   ├── merge_clean_data.py     # 步骤1: 合并+清洗数据（含国债收益率）
│   │   └── dump_etf_index.py       # 步骤2: 转换为qlib格式
│   ├── split_dataset/
│   │   ├── tradeable_distribution.py  # 统计每日可交易股票数量分布
│   │   └── split_dataset.py           # 按累计量划分训练/验证/测试集
│   └── analyze/
│       └── analyze_data.py         # 数据质量分析报告
│
└── data_visulizers/
    └── data_distribution/
        ├── plot_tracking_error.py     # ETF归一化价格对比（跟踪误差）
        ├── plot_bond_vs_etf.py        # 国债收益率 vs ETF对比分析
        ├── plot_bond_yield.py         # 国债收益率曲线
        └── plot_etf_KEB_comparison.py # 全部ETF的K/E/B线汇总对比图
```

---

## 一、数据处理流程（data_processors）

### 数据流图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           数据源                                     │
├─────────────────────────────────────────────────────────────────────┤
│  qlib_data_260415/source/etf_index/fund_kline_hfq/  基金后复权k线   │
│  qlib_data_260415/source/etf_index/fund_kline_raw/   基金除权k线    │
│  qlib_data_260415/source/etf_index/index_data/       指数数据       │
│  qlib_data_260415/source/funds_list.csv              基金列表        │
│  qlib_data_260415/source/cn_bond_rate/cn_bond_yield.csv  国债收益率 │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  merge_etf_val/merge_clean_data.py                                   │
│  - 合并基金k线(hfq优先, raw备用)与指数数据                           │
│  - 统一列名: hfq_*/raw_* → open/close/high/low/volume               │
│  - 左连接国债收益率数据(按日期精确匹配)                               │
│  - 异常价格清洗：单日尖刺(>40%涨跌且次日反向) → open/high/low/close=NaN │
│  - 不筛选行（保留全部行，含空值行）                                   │
│  - 运行前清理输出目录                                                 │
│  - 输出: qlib_data_260415/source/etf_index/merged/                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  merge_etf_val/dump_etf_index.py                                     │
│  - 转换为qlib二进制格式                                               │
│  - 输出: qlib_data_260415/qlib_etf_index_Extend_wBond/              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  split_dataset/tradeable_distribution.py                             │
│  - 统计每日可交易ETF数量，输出统计图和CSV                              │
│  - 输出: qlib_etf_index/data_distribution/                           │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  split_dataset/split_dataset.py                                      │
│  - 按可交易数量累计和划分5个数据集                                     │
│  - 训练(50%) / 验证(10%) / 测试A1(20%) / 测试A2(10%) / 测试B(10%)  │
│  - 输出: qlib_etf_index/dataset_split/                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 1. merge_etf_val/merge_clean_data.py

合并基金k线、指数数据和国债收益率，统一列名，不筛选行，并清除价格尖刺异常。

**处理逻辑**:
1. 运行前清理输出目录
2. 外连接合并 hfq + raw + index 数据，统一列名（hfq优先，raw填充缺失）
3. 按日期左连接国债收益率
4. **异常价格清洗**：检测单日价格尖刺并清零
   - 判定：`close` 单日涨跌幅绝对值 > 40% 且次日反向回复 > 40%
   - 处理：将命中行的 `open/high/low/close` 设为 NaN，`volume`、指数、估值、债券列保持不变
5. 保留所有行（不删除任何行）

**输入**:
- `qlib_data_260415/source/etf_index/fund_kline_hfq/` - 基金后复权k线
- `qlib_data_260415/source/etf_index/fund_kline_raw/` - 基金除权k线
- `qlib_data_260415/source/etf_index/index_data/` - 指数数据（含PE/PB估值）
- `qlib_data_260415/source/funds_list.csv` - 基金列表
- `qlib_data_260415/source/cn_bond_rate/cn_bond_yield.csv` - 国债收益率

**输出**: `qlib_data_260415/source/etf_index/merged/{fund_code}_clean.csv`

**输出列结构**:
```
date, open, close, high, low, volume,
index_open, index_high, index_low, index_close, index_volume,
amount, pctChg,
pe_static_equal_weight, pe_static, pe_static_median,
pe_ttm_equal_weight, pe_ttm, pe_ttm_median,
pb, pb_equal_weight, pb_median,
cn_2y, cn_5y, cn_10y, cn_30y, cn_spread_10m2, cn_gdp_yoy,
us_2y, us_5y, us_10y, us_30y, us_spread_10m2, us_gdp_yoy,
data_source
```

**启动命令**:
```bash
conda activate rdagent
python scripts/data_processors/merge_etf_val/merge_clean_data.py
```

---

### 2. merge_etf_val/dump_etf_index.py

将清洗后的CSV转换为qlib二进制格式。

**输入**: `qlib_data_260415/source/etf_index/merged/`

**输出**:
```
qlib_data_260415/qlib_etf_index_Extend_wBond/
├── calendars/day.txt
├── features/{fund_code}/{field}.day.bin
└── instruments/all.txt
```

**启动命令**:
```bash
conda activate rdagent   # macOS；Windows 用 conda activate qlib
# 使用默认路径（推荐；默认路径由 scripts/path_config.py 跨平台解析）
python scripts/data_processors/merge_etf_val/dump_etf_index.py convert

# 或指定路径（<DATA_BASE> 见 scripts/path_config.py）
python scripts/data_processors/merge_etf_val/dump_etf_index.py convert \
    --data_path <DATA_BASE>/qlib_data_260415/source/etf_index/merged \
    --qlib_dir <DATA_BASE>/qlib_data_260415/qlib_etf_index_Extend_wBond \
    --freq day \
    --max_workers 16
```

---

### 3. split_dataset/tradeable_distribution.py

统计 qlib 数据集中每日可交易ETF数量，生成分布图和CSV。

**输入**: `qlib_data_260415/qlib_etf_index/`

**输出**: `qlib_etf_index/data_distribution/tradeable_stats.csv` + 分布图

**启动命令**:
```bash
conda activate rdagent
python scripts/data_processors/split_dataset/tradeable_distribution.py
```

---

### 4. split_dataset/split_dataset.py

按每日可交易数量的累计和将数据集划分为5个子集。

**输入**: `qlib_etf_index/data_distribution/tradeable_stats.csv`

**输出**: `qlib_etf_index/dataset_split/`（训练/验证/测试集日期范围JSON）

**启动命令**:
```bash
conda activate rdagent
python scripts/data_processors/split_dataset/split_dataset.py
```

---

### 5. analyze/analyze_data.py

数据质量分析报告，检查各基金k线和指数数据完整性。

**启动命令**:
```bash
conda activate rdagent
python scripts/data_processors/analyze/analyze_data.py
```

---

## 二、可视化工具（data_visulizers）

### 1. data_distribution/plot_tracking_error.py

绘制ETF归一化价格对比图（基金后复权 vs 跟踪指数），展示跟踪误差。

**输入**: `qlib_data_260415/source/etf_index/merged/`

**输出**: `qlib_etf_index_Extend_wBond/data_distribution/TrackingError/*.png`

**启动命令**:
```bash
conda activate rdagent
python scripts/data_visulizers/data_distribution/plot_tracking_error.py
```

---

### 2. data_distribution/plot_bond_vs_etf.py

多轴对比图：国债收益率 + 1/PE(盈利收益率) + 利差 vs 510050收盘价。

**启动命令**:
```bash
conda activate rdagent
python scripts/data_visulizers/data_distribution/plot_bond_vs_etf.py
```

---

### 3. data_distribution/plot_bond_yield.py

国债收益率历史曲线图（中美各期限对比）。

**启动命令**:
```bash
conda activate rdagent
python scripts/data_visulizers/data_distribution/plot_bond_yield.py
```

---

### 4. data_distribution/plot_etf_KEB_comparison.py

全部ETF的K/E/B线汇总对比图（K=收盘价归一化，E=K/PE，B=K/PB），含数据集划分标注。

**启动命令**:
```bash
conda activate rdagent
python scripts/data_visulizers/data_distribution/plot_etf_KEB_comparison.py
```

---

## 完整流程示例

```bash
conda activate rdagent

# === 数据处理 ===
# 步骤1: 合并+清洗数据（含国债收益率）
python scripts/data_processors/merge_etf_val/merge_clean_data.py

# 步骤2: 转换为qlib格式
python scripts/data_processors/merge_etf_val/dump_etf_index.py convert

# 步骤3: 统计可交易分布（用于数据集划分）
python scripts/data_processors/split_dataset/tradeable_distribution.py

# 步骤4: 划分训练/验证/测试集
python scripts/data_processors/split_dataset/split_dataset.py

# === 可视化 ===
# 跟踪误差分析
python scripts/data_visulizers/data_distribution/plot_tracking_error.py

# 债券收益率 vs ETF分析
python scripts/data_visulizers/data_distribution/plot_bond_vs_etf.py
```

## 数据目录说明

| 目录 | 说明 |
|------|------|
| `qlib_data_260415/source/etf_index/fund_kline_hfq/` | 基金后复权日线数据 |
| `qlib_data_260415/source/etf_index/fund_kline_raw/` | 基金除权日线数据 |
| `qlib_data_260415/source/etf_index/index_data/` | 指数日线+估值数据 |
| `qlib_data_260415/source/cn_bond_rate/cn_bond_yield.csv` | 国债收益率 |
| `qlib_data_260415/source/etf_index/merged/` | 合并+清洗后的数据 |
| `qlib_data_260415/qlib_etf_index_Extend_wBond/` | qlib二进制格式数据（含债券字段） |
| `qlib_data_260415/qlib_etf_index_Extend_wBond/data_distribution/TrackingError/` | 跟踪误差可视化图表 |
| `qlib_data_260415/qlib_etf_index/data_distribution/` | 可交易分布统计 |
| `qlib_data_260415/qlib_etf_index/dataset_split/` | 数据集划分结果 |
