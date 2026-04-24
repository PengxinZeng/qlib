#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
股票价格走势图绘制工具

绘制股票收盘价走势图，支持标注 Train/Valid/Test 时间段

使用方法:
    # 绘制茅台股价走势图
    python plot_stock_price.py plot \
        --data_path /path/to/SH600519.csv \
        --save_path /path/to/output.png

    # 指定时间段配置
    python plot_stock_price.py plot \
        --data_path /path/to/SH600519.csv \
        --save_path /path/to/output.png \
        --train_start 1991-01-29 --train_end 2020-10-23 \
        --valid_start 2020-10-26 --valid_end 2023-08-07 \
        --test_start 2023-08-08 --test_end 2026-04-15
"""

import fire
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def plot_stock_price(
    data_path: str,
    save_path: str,
    title: str = None,
    train_start: str = "1991-01-29",
    train_end: str = "2020-10-23",
    valid_start: str = "2020-10-26",
    valid_end: str = "2023-08-07",
    test_start: str = "2023-08-08",
    test_end: str = "2026-04-15",
    price_type: str = "后复权",
    figsize: tuple = (14, 7),
    dpi: int = 150,
):
    """绘制股票价格走势图

    Parameters
    ----------
    data_path: str
        股票数据CSV文件路径，需包含 date 和 close 列
    save_path: str
        图片保存路径
    title: str
        图表标题，默认从文件名提取股票代码
    train_start, train_end: str
        训练集时间范围
    valid_start, valid_end: str
        验证集时间范围
    test_start, test_end: str
        测试集时间范围
    price_type: str
        价格类型描述，如 "后复权" 或 "前复权"
    figsize: tuple
        图表尺寸
    dpi: int
        图片分辨率
    """
    # 读取数据
    data_path = Path(data_path)
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # 从文件名提取股票代码
    symbol = data_path.stem  # e.g., SH600519
    if title is None:
        title = f'{symbol} 股价走势图'

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Heiti TC', 'PingFang SC']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图表
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制收盘价
    ax.plot(df['date'], df['close'], linewidth=0.8, color='#1f77b4', label='收盘价')

    # 定义时间段
    segments = {
        'Train': (train_start, train_end, '#2ecc71', 0.15),
        'Valid': (valid_start, valid_end, '#f39c12', 0.15),
        'Test': (test_start, test_end, '#e74c3c', 0.15),
    }

    # 标注时间段背景
    for name, (start, end, color, alpha) in segments.items():
        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)
        ax.axvspan(start_dt, end_dt, alpha=alpha, color=color, label=f'{name}: {start} ~ {end}')

    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('日期', fontsize=11)
    ax.set_ylabel(f'收盘价 ({price_type})', fontsize=11)

    # 设置x轴日期格式
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=45)

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.3)

    # 添加图例
    ax.legend(loc='upper left', fontsize=9)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    # 打印信息
    print(f'图表已保存到: {save_path}')
    print(f'数据范围: {df["date"].min().strftime("%Y-%m-%d")} ~ {df["date"].max().strftime("%Y-%m-%d")}')
    print(f'数据条数: {len(df)}')
    print(f'收盘价范围: {df["close"].min():.2f} ~ {df["close"].max():.2f}')


class Run:
    """运行入口"""

    def plot(
        self,
        data_path: str,
        save_path: str,
        title: str = None,
        train_start: str = "1991-01-29",
        train_end: str = "2020-10-23",
        valid_start: str = "2020-10-26",
        valid_end: str = "2023-08-07",
        test_start: str = "2023-08-08",
        test_end: str = "2026-04-15",
        price_type: str = "后复权",
    ):
        """绘制股票价格走势图

        Examples
        ---------
            # 绘制茅台股价
            python plot_stock_price.py plot \\
                --data_path ~/.qlib/stock_data/source/SH600519.csv \\
                --save_path ./maotai_price.png

            # 自定义时间段
            python plot_stock_price.py plot \\
                --data_path ~/.qlib/stock_data/source/SH600519.csv \\
                --save_path ./maotai_price.png \\
                --train_end 2019-12-31 \\
                --valid_start 2020-01-01 --valid_end 2022-12-31 \\
                --test_start 2023-01-01 --test_end 2026-04-15
        """
        plot_stock_price(
            data_path=data_path,
            save_path=save_path,
            title=title,
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            test_start=test_start,
            test_end=test_end,
            price_type=price_type,
        )


if __name__ == "__main__":
    fire.Fire(Run)
