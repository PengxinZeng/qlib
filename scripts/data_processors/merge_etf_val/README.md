# ETF指数数据处理流程

## 目录结构

```
merge_etf_val/
├── merge_data.py           # 步骤1: 合并基金k线和指数数据
├── clean_data.py           # 步骤2: 清洗数据，标准化列名
├── dump_etf_index.py      # 步骤3: 转换为qlib格式
├── visualize_normalized.py  # 步骤4: 可视化归一化价格
└── analyze_data.py         # 数据分析工具
```

## 数据处理流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           数据源                                     │
├─────────────────────────────────────────────────────────────────────┤
│  fund_kline_hfq/     基金后复权k线                                  │
│  fund_kline_raw/      基金除权k线                                    │
│  index_data/          指数数据(含PE/PB估值)                          │
│  qlib_data_260415/source/funds_list.csv  基金列表(基金代码+跟踪目标) │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  merge_data.py                                                        │
│  - 合并基金k线(hfq+raw)与指数数据                                     │
│  - 保留所有原始列                                                      │
│  - 输出: stock_data/merged2/                                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  clean_data.py                                                        │
│  - 分类: 有后复权 / 只有除权 / 无数据                                 │
│  - 统一列名: hfq_* → open/close/high/low/volume                     │
│  - 统一列名: raw_* → open/close/high/low/volume                     │
│  - 拼接指数数据(保留所有列)                                            │
│  - 筛选: k线和指数数据均完整的行                                        │
│  - 输出: stock_data/normed/                                           │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  dump_etf_index.py / visualize_normalized.py                        │
│  - dump_etf_index.py: 转换为qlib格式 → qlib_data/qlib_etf_index/   │
│  - visualize_normalized.py: 可视化 → report/normalized2/             │
└─────────────────────────────────────────────────────────────────────┘
```

## 脚本详细说明

### 1. merge_data.py

合并基金k线数据与指数数据。

**输入**:
- `stock_data/fund_kline_hfq/` - 基金后复权k线
- `stock_data/fund_kline_raw/` - 基金除权k线
- `stock_data/index_data/` - 指数数据
- `qlib_data_260415/source/funds_list.csv` - 基金列表

**处理逻辑**:
- 读取基金列表，筛选有有效 `track_target_file` 的基金
- 读取后复权数据和除权数据
- 读取对应指数数据（k线+估值）
- 外连接合并所有数据

**输出**:
- `stock_data/merged2/{fund_code}_merged.csv`

**使用**:
```bash
conda activate rdagent
python scripts/data_collector/merge_etf_val/merge_data.py
```

---

### 2. clean_data.py

清洗数据，标准化列名，筛选有效行。

**输入**:
- `stock_data/merged2/` - 合并后的数据

**处理逻辑**:
1. 根据 `report/merge_status.csv` 分类基金:
   - **有后复权数据**: 使用后复权作为基金k线
   - **只有除权数据**: 使用除权作为基金k线
   - **无数据**: 跳过

2. 统一列名:
   - `hfq_open/hfq_close/...` → `open/close/...`
   - `raw_open/raw_close/...` → `open/close/...`

3. 拼接指数数据，保留所有列:
   - 指数k线: `index_open/high/low/close/volume`
   - 指数估值: `pe_*/pb_*/amount/pctChg`

4. 筛选: 仅保留基金k线和指数数据均完整的行

**输出**:
- `stock_data/normed/{fund_code}_normed.csv`

**使用**:
```bash
conda activate rdagent
python scripts/data_collector/merge_etf_val/clean_data.py
```

**输出列结构** (所有CSV统一):
```
date, open, close, high, low, volume,
index_open, index_high, index_low, index_close, index_volume,
amount, pctChg,
pe_static_equal_weight, pe_static, pe_static_median,
pe_ttm_equal_weight, pe_ttm, pe_ttm_median,
pb, pb_equal_weight, pb_median,
data_source
```

---

### 3. dump_etf_index.py

将清洗后的数据转换为qlib二进制格式。

**输入**:
- `qlib_data_260415/source/etf_index/` - 清洗后的CSV数据

**处理逻辑**:
- 读取所有CSV文件
- 生成日历文件 (所有交易日的集合)
- 生成instruments文件 (每个基金的代码和日期范围)
- 为每个基金创建feature目录
- 将每个字段保存为 `.day.bin` 文件

**输出** (qlib格式):
```
qlib_data_260415/source/qlib_etf_index/
├── calendars/
│   └── day.txt           # 日历文件
├── features/
│   ├── {fund_code}_normed/
│   │   ├── open.day.bin
│   │   ├── close.day.bin
│   │   ├── ...
│   │   └── pe_ttm.day.bin
│   └── ...
└── instruments/
    └── all.txt           # instrument列表
```

**使用**:
```bash
conda activate rdagent
python scripts/data_collector/merge_etf_val/dump_etf_index.py convert \
    --data_path /path/to/etf_index \
    --qlib_dir /path/to/qlib_etf_index \
    --freq day \
    --max_workers 16
```

---

### 4. visualize_normalized.py

绘制归一化价格时间序列图。

**输入**:
- `stock_data/normed/` - 清洗后的数据

**输出**:
- `stock_data/report/normalized2/*.png` - 每个基金的归一化图
- `stock_data/report/normalized_plots_merged2.html` - 汇总HTML页面

**使用**:
```bash
conda activate rdagent
python scripts/data_collector/merge_etf_val/visualize_normalized2.py
```

---

### 5. analyze_data.py

数据分析工具，用于生成数据完整性报告。

**输入**:
- `stock_data/normed/` - 清洗后的数据

**输出**:
- 数据完整性统计

**使用**:
```bash
conda activate rdagent
python scripts/data_collector/merge_etf_val/analyze_data.py
```

---

## 完整使用示例

```bash
# 1. 激活环境
conda activate rdagent

# 2. 合并数据
python scripts/data_collector/merge_etf_val/merge_data.py

# 3. 清洗数据
python scripts/data_collector/merge_etf_val/clean_data.py

# 4. 转换为qlib格式
python scripts/data_collector/merge_etf_val/dump_etf_index.py convert \
    --data_path /path/to/etf_index \
    --qlib_dir /path/to/qlib_etf_index

# 5. 可视化
python scripts/data_collector/merge_etf_val/visualize_normalized2.py
```

## 数据目录说明

| 目录 | 说明 |
|------|------|
| `stock_data/fund_kline_hfq/` | 基金后复权日线数据 |
| `stock_data/fund_kline_raw/` | 基金除权日线数据 |
| `stock_data/index_data/` | 指数日线+估值数据 |
| `stock_data/merged2/` | 合并后的原始数据(未清洗) |
| `stock_data/normed/` | 清洗后的标准化数据 |
| `stock_data/report/` | 分析报告 |
| `stock_data/report/normalized2/` | 归一化可视化图表 |
| `qlib_data/qlib_etf_index/` | qlib二进制格式数据 |