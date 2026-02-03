import akshare as ak
import pandas as pd
import os

source_path = "D:/Pengxin/CodeBase/Quant/qlib/gold_source"
os.makedirs(source_path, exist_ok=True)
# 尝试获取伦敦金的历史行情（新浪源）
try:
    # XAU 是伦敦金的国际通用代码
    gold_df = ak.futures_index_unsh_sina(symbol="XAU") 
    print("获取成功！")
except:
    # 如果还是不行，获取国内沪金连续合约，数据走势与国际金高度同步
    print("伦敦金接口失效，正在切换至国内沪金主力合约...")
    gold_df = ak.futures_zh_daily_sina(symbol="AU0") 

# 后续清洗逻辑
gold_df['date'] = pd.to_datetime(gold_df['date'])
# gold_df = gold_df[gold_df['date'] >= '2026-01-01']
# ... 后面接之前的保存和 Dump 逻辑
# 3. 字段映射以符合 Qlib 规范
# Qlib 必须包含: open, high, low, close, volume, factor
gold_data = pd.DataFrame({
    'date': gold_df['date'],
    'open': gold_df['open'].astype(float),
    'high': gold_df['high'].astype(float),
    'low': gold_df['low'].astype(float),
    'close': gold_df['close'].astype(float),
    'volume': gold_df['hold'].astype(float),
    'factor': 1.0  # 黄金无需除权，统一设为 1
})

# 4. 保存为 CSV (Qlib 识别文件名作为代码)
gold_data.to_csv(os.path.join(source_path, "XAU.csv"), index=False)
print(f"数据处理完成，共 {len(gold_data)} 行。已保存至: {source_path}/XAU.csv")