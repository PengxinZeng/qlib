#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETF/Stock LightGBM Workflow
使用Qlib对ETF和股票数据进行LightGBM模型训练和回测

Usage:
    # 训练并回测
    python workflow_etf.py train_and_backtest

    # 仅训练
    python workflow_etf.py train

    # 分析已有结果
    python workflow_etf.py analyze --recorder_id=xxx --experiment_id=xxx
"""

import os
import sys
import fire
import pandas as pd
import qlib
import yaml
import plotly.graph_objects as go
from pathlib import Path
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord


# 默认配置路径
DEFAULT_CONFIG = "examples/benchmarks/LightGBM/workflow_config_lightgbm_etf.yaml"

# 跨平台路径集中配置（Mac / Windows 兼容）
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import path_config  # noqa: E402

# 标的信息路径
INSTRUMENT_INFO_PATH = str(path_config.QLIB_BASE / "qlib_data" / "instruments" / "instrument_info.csv")

# 颜色配置
COLORS = [
    '#808080',  # cash
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
    '#c49c94', '#f7b6d2', '#c7c7c7', '#FFD700',
]


def load_config(config_path=DEFAULT_CONFIG):
    """加载YAML配置文件"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_instrument_info(info_path=None):
    """加载标的信息（代码、中文名、类型）"""
    info_path = info_path or INSTRUMENT_INFO_PATH
    info_dict = {}
    try:
        df = pd.read_csv(info_path)
        for _, row in df.iterrows():
            info_dict[row['symbol']] = {'name': row['name'], 'type': row['type']}
    except Exception as e:
        print(f"Warning: Could not load instrument info: {e}")
    return info_dict


def positions_to_dataframe(positions):
    """将持仓数据转换为DataFrame"""
    data = []
    for date, pos in positions.items():
        row = {'date': date}
        row['cash'] = float(pos.position.get('cash', 0))
        for symbol, info in pos.position.items():
            if symbol in ['cash', 'now_account_value']:
                continue
            if isinstance(info, dict):
                row[symbol] = float(info.get('amount', 0) * info.get('price', 0))
        data.append(row)
    return pd.DataFrame(data).set_index('date').sort_index().fillna(0)


def get_display_name(symbol, instrument_info):
    """获取显示名称（代码+中文名）"""
    if symbol == '其他':
        return '其他'
    info = instrument_info.get(symbol, {})
    return f"{symbol} {info.get('name', '')}" if info.get('name') else symbol


def generate_position_chart(positions, top_n=18, instrument_info=None):
    """生成持仓金额堆叠面积图"""
    instrument_info = instrument_info or load_instrument_info()
    df = positions_to_dataframe(positions)

    # 选出TOP N标的
    col_sums = df.drop(columns=['cash'], errors='ignore').sum()
    top_symbols = col_sums.nlargest(top_n).index.tolist()
    other_symbols = [c for c in df.columns if c not in top_symbols and c != 'cash']
    df['其他'] = df[other_symbols].sum(axis=1)

    # 排序
    top_symbols_sorted = sorted(top_symbols, key=lambda x: col_sums.get(x, 0), reverse=True)
    ordered_cols = ['cash'] + top_symbols_sorted + ['其他']

    fig = go.Figure()
    for i, col in enumerate(ordered_cols):
        if col in df.columns and df[col].sum() > 0:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col], mode='lines', name=get_display_name(col, instrument_info),
                stackgroup='one', fillcolor=COLORS[i % len(COLORS)],
                line=dict(width=0.5, color=COLORS[i % len(COLORS)]),
                hovertemplate='%{x}<br>%{y:,.0f}<extra>%{fullData.name}</extra>'
            ))

    fig.update_layout(
        title=f'持仓金额变化 - 堆叠面积图 (TOP {top_n})',
        xaxis_title='日期', yaxis_title='持仓金额',
        legend_title='标的', hovermode='x unified', template='plotly_white',
        width=1400, height=800,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    return fig


def generate_position_percent_chart(positions, top_n=18, instrument_info=None):
    """生成持仓比例堆叠面积图（百分比）"""
    instrument_info = instrument_info or load_instrument_info()
    df = positions_to_dataframe(positions)

    # 选出TOP N标的
    col_sums = df.drop(columns=['cash'], errors='ignore').sum()
    top_symbols = col_sums.nlargest(top_n).index.tolist()
    other_symbols = [c for c in df.columns if c not in top_symbols and c != 'cash']
    df['其他'] = df[other_symbols].sum(axis=1)

    # 保留需要的列并计算百分比
    keep_cols = ['cash'] + top_symbols + ['其他']
    df = df[[c for c in keep_cols if c in df.columns]]
    df_percent = df.div(df.sum(axis=1), axis=0).fillna(0) * 100

    # 排序
    top_symbols_sorted = sorted(top_symbols, key=lambda x: col_sums.get(x, 0), reverse=True)
    ordered_cols = ['cash'] + top_symbols_sorted + ['其他']

    fig = go.Figure()
    for i, col in enumerate(ordered_cols):
        if col in df_percent.columns:
            fig.add_trace(go.Scatter(
                x=df_percent.index, y=df_percent[col], mode='lines', name=get_display_name(col, instrument_info),
                stackgroup='one', groupnorm='percent', fillcolor=COLORS[i % len(COLORS)],
                line=dict(width=0.5, color=COLORS[i % len(COLORS)]),
                hovertemplate='%{x}<br>%{y:.1f}%<extra>%{fullData.name}</extra>'
            ))

    fig.update_layout(
        title=f'持仓比例变化 - 堆叠面积图 (TOP {top_n})',
        xaxis_title='日期', yaxis_title='持仓比例', yaxis=dict(ticksuffix='%', range=[0, 100]),
        legend_title='标的', hovermode='x unified', template='plotly_white',
        width=1400, height=800,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    return fig


def save_figures_to_path(fig, fig_percent, save_path):
    """保存图表到指定目录"""
    os.makedirs(save_path, exist_ok=True)
    fig.write_image(os.path.join(save_path, "position_values.png"), scale=2)
    fig.write_html(os.path.join(save_path, "position_values.html"))
    fig_percent.write_image(os.path.join(save_path, "position_percent.png"), scale=2)
    fig_percent.write_html(os.path.join(save_path, "position_percent.html"))
    print(f"[持仓图表已保存] {save_path}")


def print_performance(report_df):
    """打印绩效指标"""
    total_return = report_df['return'].sum()
    annualized_return = report_df['return'].mean() * 252
    volatility = report_df['return'].std() * (252 ** 0.5)
    sharpe = annualized_return / volatility if volatility > 0 else 0
    max_drawdown = (report_df['return'].cumsum().cummax() - report_df['return'].cumsum()).max()

    print(f"\n[Portfolio Performance]")
    print(f"  Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
    print(f"  Annualized Return: {annualized_return:.4f} ({annualized_return*100:.2f}%)")
    print(f"  Volatility: {volatility:.4f}")
    print(f"  Sharpe Ratio: {sharpe:.4f}")
    print(f"  Max Drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")


def train(config_path=DEFAULT_CONFIG, experiment_name="etf_lgb_train"):
    """训练LightGBM模型"""
    config = load_config(config_path)
    qlib.init(**config.get("qlib_init"))
    task = config.get("task", {})

    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    with R.start(experiment_name=experiment_name):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(params=model)

        recorder = R.get_recorder()
        print(f"Training completed! Recorder ID: {recorder.id}, Experiment ID: {recorder.experiment_id}")

    return recorder.id, recorder.experiment_id


def backtest(recorder_id, experiment_id, config_path=DEFAULT_CONFIG):
    """基于已有模型进行回测"""
    config = load_config(config_path)
    qlib.init(**config.get("qlib_init"))
    task = config.get("task", {})

    dataset = init_instance_by_config(task["dataset"])
    port_analysis_config = config.get("port_analysis_config", {})

    with R.start(experiment_name="etf_backtest"):
        model = R.get_recorder(recorder_id=recorder_id, experiment_id=str(experiment_id)).load_object("params")

        recorder = R.get_recorder()
        SignalRecord(model, dataset, recorder).generate()
        SigAnaRecord(recorder).generate()
        PortAnaRecord(recorder, port_analysis_config, "day").generate()

        print(f"Backtest completed! Recorder ID: {recorder.id}, Experiment ID: {recorder.experiment_id}")

    return recorder.id, recorder.experiment_id


def train_and_backtest(config_path=DEFAULT_CONFIG, experiment_name="etf_lgb"):
    """训练模型并进行回测（完整流程）"""
    config = load_config(config_path)
    qlib.init(**config.get("qlib_init"))
    task = config.get("task", {})

    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])
    port_analysis_config = config.get("port_analysis_config", {})

    with R.start(experiment_name=experiment_name):
        R.log_params(**flatten_dict(task))
        print("Training model...")
        model.fit(dataset)
        R.save_objects(params=model)

        recorder = R.get_recorder()

        print("Generating signals...")
        SignalRecord(model, dataset, recorder).generate()
        SigAnaRecord(recorder, ana_long_short=False, ann_scaler=252).generate()

        print("Running backtest...")
        PortAnaRecord(recorder, port_analysis_config, "day").generate()

        print(f"\n{'='*50}")
        print(f"Training and backtest completed!")
        print(f"Recorder ID: {recorder.id}")
        print(f"Experiment ID: {recorder.experiment_id}")
        print(f"{'='*50}")

        # 打印结果摘要
        report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
        print(f"\nBacktest period: {report_df.index[0]} to {report_df.index[-1]}")
        print_performance(report_df)

    return recorder.id, recorder.experiment_id


def analyze(recorder_id, experiment_id, config_path=DEFAULT_CONFIG, save_figures=True, save_path=None):
    """分析回测结果

    Args:
        recorder_id: recorder ID
        experiment_id: 实验ID
        config_path: 配置文件路径
        save_figures: 是否保存图表
        save_path: 图表保存路径（可选）
    """
    config = load_config(config_path)
    qlib.init(**config.get("qlib_init"))
    task = config.get("task", {})

    dataset = init_instance_by_config(task["dataset"])
    recorder = R.get_recorder(recorder_id=recorder_id, experiment_id=str(experiment_id))

    # 加载结果
    pred_df = recorder.load_object("pred.pkl")
    report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    print("\n" + "="*60)
    print("BACKTEST ANALYSIS REPORT")
    print("="*60)
    print(f"\n[Prediction Summary]")
    print(f"  Prediction shape: {pred_df.shape}")
    print(f"  Date range: {pred_df.index.get_level_values('datetime').min()} to {pred_df.index.get_level_values('datetime').max()}")

    print_performance(report_df)

    # 保存CSV
    if save_figures and save_path:
        os.makedirs(save_path, exist_ok=True)
        pred_df.to_csv(os.path.join(save_path, "pred_df.csv"))
        report_df.to_csv(os.path.join(save_path, "report_df.csv"))
        analysis_df.to_csv(os.path.join(save_path, "analysis_df.csv"))
        print(f"\n[Results saved to] {save_path}")

    # 生成图表
    if save_figures and save_path:
        try:
            from qlib.contrib.report import analysis_position

            # 收益曲线
            fig_list = analysis_position.report_graph(report_df, show_notebook=False)
            for idx, fig in enumerate(fig_list):
                fig.write_image(os.path.join(save_path, f"return_{idx}.png"))

            # 风险分析
            fig_list = analysis_position.risk_analysis_graph(analysis_df, report_df, show_notebook=False)
            for idx, fig in enumerate(fig_list):
                fig.write_image(os.path.join(save_path, f"risk_{idx}.png"))

            # IC分析
            label_df = dataset.prepare("test", col_set="label")
            label_df.columns = ["label"]
            pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
            fig_list = analysis_position.score_ic_graph(pred_label, show_notebook=False)
            for idx, fig in enumerate(fig_list):
                fig.write_image(os.path.join(save_path, f"score_ic_{idx}.png"))

            # 持仓图表
            fig = generate_position_chart(positions)
            fig_percent = generate_position_percent_chart(positions)
            save_figures_to_path(fig, fig_percent, save_path)

            print(f"[所有图表已保存] {save_path}")

        except Exception as e:
            import traceback
            print(f"\nWarning: Could not generate figures: {e}")
            traceback.print_exc()

    return pred_df, report_df, analysis_df


if __name__ == "__main__":
    fire.Fire({
        "train": train,
        "backtest": backtest,
        "train_and_backtest": train_and_backtest,
        "analyze": analyze,
    })
