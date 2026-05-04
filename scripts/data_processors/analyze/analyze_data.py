"""
数据质量分析报告
"""
import pandas as pd
from pathlib import Path
from datetime import datetime

# 配置
BASE_DIR = Path('/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/stock_data')
MERGED_DIR = BASE_DIR / 'merged'
HFQ_DIR = BASE_DIR / 'fund_kline_hfq'
RAW_DIR = BASE_DIR / 'fund_kline_raw'
REPORT_DIR = BASE_DIR / 'report'
FUNDS_LIST = '/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415/source/funds_list.csv'

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 读取基金列表
df = pd.read_csv(FUNDS_LIST, comment='#', dtype=str)
df = df.dropna(subset=['fund_code', 'track_target_file'])
df = df[df['track_target_file'] != 'N/A']

report_data = []

for _, row in df.iterrows():
    code = row['fund_code']
    name = row['fund_name']
    index_file = row['track_target_file']

    merged_path = MERGED_DIR / f"{code}.csv"

    if not merged_path.exists():
        report_data.append({
            'fund_code': code,
            'fund_name': name,
            'status': '无合并数据',
            'hfq_exists': False,
            'raw_exists': False,
            'merged_exists': False,
        })
        continue

    merged_df = pd.read_csv(merged_path)
    merged_df['date'] = pd.to_datetime(merged_df['date'])

    # 检查各列数据
    hfq_exists = (HFQ_DIR / f"{code}.csv").exists()
    raw_exists = (RAW_DIR / f"{code}.csv").exists()

    # 数据完整性
    total_rows = len(merged_df)

    # 检查关键列
    hfq_close_missing = merged_df['hfq_close'].isna().sum() if 'hfq_close' in merged_df.columns else total_rows
    raw_close_missing = merged_df['raw_close'].isna().sum() if 'raw_close' in merged_df.columns else total_rows
    index_pb_missing = merged_df['index_pb'].isna().sum() if 'index_pb' in merged_df.columns else total_rows
    index_pe_missing = merged_df['index_pe_ttm'].isna().sum() if 'index_pe_ttm' in merged_df.columns else total_rows

    # 日期范围
    date_start = merged_df['date'].min()
    date_end = merged_df['date'].max()

    # 判断状态
    status = '正常'
    if hfq_close_missing > total_rows * 0.1 or raw_close_missing > total_rows * 0.1:
        status = '数据缺失>10%'
    if not hfq_exists and not raw_exists:
        status = '无K线数据'

    report_data.append({
        'fund_code': code,
        'fund_name': name,
        'status': status,
        'hfq_exists': hfq_exists,
        'raw_exists': raw_exists,
        'merged_exists': True,
        'total_rows': total_rows,
        'date_start': date_start,
        'date_end': date_end,
        'hfq_close_missing': hfq_close_missing,
        'raw_close_missing': raw_close_missing,
        'index_pb_missing': index_pb_missing,
        'index_pe_missing': index_pe_missing,
        'hfq_missing_pct': f"{hfq_close_missing/total_rows*100:.1f}%",
        'raw_missing_pct': f"{raw_close_missing/total_rows*100:.1f}%",
    })

# 保存报告
report_df = pd.DataFrame(report_data)
report_path = REPORT_DIR / 'data_quality_report.csv'
report_df.to_csv(report_path, index=False)

# 生成HTML报告
html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>基金数据质量分析报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .warning {{ background-color: #fff3cd; }}
        .error {{ background-color: #f8d7da; }}
        .success {{ background-color: #d4edda; }}
        .summary {{ margin: 20px 0; padding: 15px; background-color: #e9ecef; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>基金数据质量分析报告</h1>
    <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="summary">
        <h3>数据统计</h3>
        <p>总基金数: {len(report_df)}</p>
        <p>有K线数据: {report_df['hfq_exists'].sum() + report_df['raw_exists'].sum()}</p>
        <p>数据正常: {report_df[report_df['status'] == '正常'].shape[0]}</p>
    </div>

    <h2>详细数据</h2>
    <table>
        <tr>
            <th>基金代码</th>
            <th>基金名称</th>
            <th>状态</th>
            <th>后复权</th>
            <th>除权</th>
            <th>总行数</th>
            <th>开始日期</th>
            <th>结束日期</th>
            <th>后复权缺失%</th>
            <th>除权缺失%</th>
        </tr>
"""

for _, row in report_df.iterrows():
    status_class = 'success' if row['status'] == '正常' else 'warning'
    html_content += f"""        <tr class="{status_class}">
            <td>{row['fund_code']}</td>
            <td>{row['fund_name']}</td>
            <td>{row['status']}</td>
            <td>{'是' if row['hfq_exists'] else '否'}</td>
            <td>{'是' if row['raw_exists'] else '否'}</td>
            <td>{row.get('total_rows', 'N/A')}</td>
            <td>{row.get('date_start', 'N/A')}</td>
            <td>{row.get('date_end', 'N/A')}</td>
            <td>{row.get('hfq_missing_pct', 'N/A')}</td>
            <td>{row.get('raw_missing_pct', 'N/A')}</td>
        </tr>
"""

html_content += """    </table>
</body>
</html>"""

html_path = REPORT_DIR / 'data_quality_report.html'
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"数据质量报告已保存:")
print(f"  CSV: {report_path}")
print(f"  HTML: {html_path}")

# 打印汇总
print("\n=== 数据汇总 ===")
print(report_df.to_string())
