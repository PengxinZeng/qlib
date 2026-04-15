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
    
    # 基于已有模型进行回测
    python workflow_etf.py backtest --recorder_id=xxx --experiment_id=xxx
"""

import os
import fire
import pandas as pd
import qlib
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord


# 默认配置路径
DEFAULT_CONFIG = "examples/benchmarks/LightGBM/workflow_config_lightgbm_etf.yaml"


def load_config(config_path=DEFAULT_CONFIG):
    """加载YAML配置文件"""
    import yaml
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train(config_path=DEFAULT_CONFIG, experiment_name="etf_lgb_train"):
    """
    训练LightGBM模型
    
    Args:
        config_path: 配置文件路径
        experiment_name: 实验名称
    
    Returns:
        tuple: (recorder_id, experiment_id)
    """
    config = load_config(config_path)
    task = config.get("task", {})
    
    # 初始化Qlib
    qlib.init(**config.get("qlib_init"))
    
    # 初始化模型和数据集
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])
    
    # 开始训练
    with R.start(experiment_name=experiment_name):
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(params=model)  # 保存模型
        
        recorder = R.get_recorder()
        recorder_id = recorder.id
        experiment_id = recorder.experiment_id
        
        print(f"Training completed!")
        print(f"Recorder ID: {recorder_id}")
        print(f"Experiment ID: {experiment_id}")
    
    return recorder_id, experiment_id


def backtest(recorder_id, experiment_id, config_path=DEFAULT_CONFIG):
    """
    基于训练好的模型进行回测
    
    Args:
        recorder_id: 已保存模型的recorder_id
        experiment_id: 实验ID
        config_path: 配置文件路径
    """
    config = load_config(config_path)
    task = config.get("task", {})
    
    # 初始化Qlib
    qlib.init(**config.get("qlib_init"))
    
    # 加载数据集（用于回测）
    dataset = init_instance_by_config(task["dataset"])
    port_analysis_config = config.get("port_analysis_config", {})
    
    # 加载已训练的模型
    with R.start(experiment_name="etf_backtest"):
        original_recorder = R.get_recorder(
            recorder_id=recorder_id, 
            experiment_id=str(experiment_id)
        )
        model = original_recorder.load_object("params.pkl")
        
        # 获取当前recorder
        recorder = R.get_recorder()
        new_recorder_id = recorder.id
        new_experiment_id = recorder.experiment_id
        
        # 生成预测信号
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        
        # 信号分析
        sig_ana = SigAnaRecord(recorder)
        sig_ana.generate()
        
        # 回测分析
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
        
        print(f"Backtest completed!")
        print(f"New Recorder ID: {new_recorder_id}")
        print(f"New Experiment ID: {new_experiment_id}")
    
    return new_recorder_id, new_experiment_id


def train_and_backtest(config_path=DEFAULT_CONFIG, experiment_name="etf_lgb"):
    """
    训练模型并进行回测（完整流程）
    
    Args:
        config_path: 配置文件路径
        experiment_name: 实验名称
    """
    config = load_config(config_path)
    task = config.get("task", {})
    
    # 初始化Qlib
    qlib.init(**config.get("qlib_init"))
    
    # 初始化模型和数据集
    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])
    port_analysis_config = config.get("port_analysis_config", {})
    
    # 训练并记录
    with R.start(experiment_name=experiment_name):
        R.log_params(**flatten_dict(task))
        
        # 训练模型
        print("Training model...")
        model.fit(dataset)
        R.save_objects(params=model)
        
        recorder = R.get_recorder()
        recorder_id = recorder.id
        experiment_id = recorder.experiment_id
        
        # 生成预测信号
        print("Generating signals...")
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()
        
        # 信号分析
        print("Analyzing signals...")
        sig_ana = SigAnaRecord(recorder, ana_long_short=False, ann_scaler=252)
        sig_ana.generate()
        
        # 回测分析
        print("Running backtest...")
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()
        
        print(f"\n{'='*50}")
        print(f"Training and backtest completed!")
        print(f"Recorder ID: {recorder_id}")
        print(f"Experiment ID: {experiment_id}")
        print(f"{'='*50}")
        
        # 加载并显示结果摘要
        try:
            pred_df = recorder.load_object("pred.pkl")
            report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
            
            print(f"\nPrediction shape: {pred_df.shape}")
            print(f"\nBacktest period: {report_df.index[0]} to {report_df.index[-1]}")
            print(f"Total return: {report_df['return'].sum():.4f}")
            print(f"Annualized return: {report_df['return'].mean() * 252:.4f}")
            print(f"Max drawdown: {report_df['return'].cumsum().cummax() - report_df['return'].cumsum()}")
        except Exception as e:
            print(f"Could not load results summary: {e}")
    
    return recorder_id, experiment_id


def analyze(recorder_id, experiment_id, config_path=DEFAULT_CONFIG, save_figures=True):
    """
    分析已完成的回测结果
    
    Args:
        recorder_id: recorder ID
        experiment_id: 实验ID
        config_path: 配置文件路径
        save_figures: 是否保存图表
    """
    config = load_config(config_path)
    task = config.get("task", {})
    
    # 初始化Qlib
    qlib.init(**config.get("qlib_init"))
    
    # 加载数据集
    dataset = init_instance_by_config(task["dataset"])
    
    # 获取recorder
    recorder = R.get_recorder(recorder_id=recorder_id, experiment_id=str(experiment_id))
    
    # 加载结果
    pred_df = recorder.load_object("pred.pkl")
    report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    
    # 打印分析结果
    print("\n" + "="*60)
    print("BACKTEST ANALYSIS REPORT")
    print("="*60)
    
    print(f"\n[Prediction Summary]")
    print(f"  Prediction shape: {pred_df.shape}")
    print(f"  Date range: {pred_df.index.get_level_values('datetime').min()} to {pred_df.index.get_level_values('datetime').max()}")
    
    print(f"\n[Portfolio Performance]")
    total_return = report_df['return'].sum()
    annualized_return = report_df['return'].mean() * 252
    volatility = report_df['return'].std() * (252 ** 0.5)
    sharpe = annualized_return / volatility if volatility > 0 else 0
    
    cumulative_return = report_df['return'].cumsum()
    max_drawdown = (cumulative_return.cummax() - cumulative_return).max()
    
    print(f"  Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
    print(f"  Annualized Return: {annualized_return:.4f} ({annualized_return*100:.2f}%)")
    print(f"  Volatility: {volatility:.4f}")
    print(f"  Sharpe Ratio: {sharpe:.4f}")
    print(f"  Max Drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
    
    # 保存CSV
    if save_figures:
        save_root = os.path.join(
            recorder._uri.replace('file:', ''), 
            recorder.experiment_id, 
            recorder.id, 
            "analysis_csvs"
        )
        os.makedirs(save_root, exist_ok=True)
        
        pred_df.to_csv(os.path.join(save_root, "pred_df.csv"))
        report_df.to_csv(os.path.join(save_root, "report_df.csv"))
        analysis_df.to_csv(os.path.join(save_root, "analysis_df.csv"))
        
        print(f"\n[Results saved to]")
        print(f"  {save_root}")
    
    # 生成图表
    if save_figures:
        try:
            from qlib.contrib.report import analysis_position
            
            # 收益曲线
            fig_list_return = analysis_position.report_graph(report_df, show_notebook=False)
            
            # 风险分析
            fig_list_risk = analysis_position.risk_analysis_graph(analysis_df, report_df, show_notebook=False)
            
            # IC分析
            label_df = dataset.prepare("test", col_set="label")
            label_df.columns = ["label"]
            pred_label = pd.concat([label_df, pred_df], axis=1, sort=True).reindex(label_df.index)
            fig_list_score_ic = analysis_position.score_ic_graph(pred_label, show_notebook=False)
            
            # 保存图表
            fig_save_root = os.path.join(
                recorder._uri.replace('file:', ''), 
                recorder.experiment_id, 
                recorder.id, 
                "analysis_figures"
            )
            os.makedirs(fig_save_root, exist_ok=True)
            
            for idx, fig in enumerate(fig_list_return):
                fig.write_image(os.path.join(fig_save_root, f"return_{idx}.png"))
            for idx, fig in enumerate(fig_list_risk):
                fig.write_image(os.path.join(fig_save_root, f"risk_{idx}.png"))
            for idx, fig in enumerate(fig_list_score_ic):
                fig.write_image(os.path.join(fig_save_root, f"score_ic_{idx}.png"))
            
            print(f"\n[Figures saved to]")
            print(f"  {fig_save_root}")
        except Exception as e:
            print(f"\nWarning: Could not generate figures: {e}")
    
    return pred_df, report_df, analysis_df


if __name__ == "__main__":
    fire.Fire({
        "train": train,
        "backtest": backtest,
        "train_and_backtest": train_and_backtest,
        "analyze": analyze,
    })
