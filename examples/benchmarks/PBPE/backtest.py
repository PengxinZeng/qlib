"""
PB/PE 价值投资策略回测脚本
使用方法:
    cd /Users/zengpengxin/workspace/CodeBase/qlib
    source activate rdagent
    python examples/benchmarks/PBPE/backtest.py
"""

import sys
import os
from pathlib import Path

# 添加qlib路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.utils.file import remove_date_suffix_in_dir

# 配置文件路径
STRATEGY_DIR = Path(__file__).parent
CONFIG_FILE = STRATEGY_DIR / "workflow_config_pbpe.yaml"


def init_qlib():
    """初始化qlib"""
    qlib.init(
        provider_uri="/Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415/qlib_etf_index",
        region="cn"
    )


def run_backtest():
    """运行回测"""
    # 初始化
    init_qlib()

    # 读取配置
    import yaml
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)

    # 创建任务
    task = config["task"]

    # 初始化模型
    model = init_instance_by_config(task["model"])

    # 初始化数据集
    dataset = init_instance_by_config(task["dataset"])

    # 训练模型
    print("=" * 60)
    print("开始训练模型...")
    print("=" * 60)

    model.fit(dataset)

    print("\n模型训练完成!")
    print(f"模型参数: {model.get_params()}")

    # 保存模型
    print("\n保存模型...")
    rid = R.get_recorder_id()
    R.get_recorder().save_objects(model=model)

    # 生成信号
    print("\n生成预测信号...")
    pred_train = model.predict(dataset, segment="train")
    pred_valid = model.predict(dataset, segment="valid")
    pred_test = model.predict(dataset, segment="test")

    print(f"训练集预测数: {len(pred_train)}")
    print(f"验证集预测数: {len(pred_valid)}")
    print(f"测试集预测数: {len(pred_test)}")

    # 信号分析
    print("\n" + "=" * 60)
    print("信号分析...")
    print("=" * 60)

    # 计算IC值
    train_ic = pred_train.corrwith(dataset.get_label("train"))
    valid_ic = pred_valid.corrwith(dataset.get_label("valid"))
    test_ic = pred_test.corrwith(dataset.get_label("test"))

    print(f"训练集IC: {train_ic:.4f}")
    print(f"验证集IC: {valid_ic:.4f}")
    print(f"测试集IC: {test_ic:.4f}")

    # 回测分析
    print("\n" + "=" * 60)
    print("开始回测...")
    print("=" * 60)

    # 创建回测记录器
    port_analysis_config = config["port_analysis_config"]

    # 初始化策略
    from strategy import PBPEValueStrategy
    strategy = PBPEValueStrategy()

    # 执行回测
    from qlib.contrib.evaluate import backtest_daily
    from qlib.data.dataset import DatasetH

    # 回测区间
    backtest_config = port_analysis_config["backtest"]

    print(f"回测区间: {backtest_config['start_time']} ~ {backtest_config['end_time']}")
    print(f"初始资金: {backtest_config['account']}")

    # 执行回测
    report,_metrics = backtest_daily(
        dataset=dataset,
        strategy=strategy,
        **backtest_config
    )

    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)

    for key, value in metrics.items():
        print(f"{key}: {value}")

    # 保存结果
    output_dir = STRATEGY_DIR / "output"
    output_dir.mkdir(exist_ok=True)

    report.to_csv(output_dir / "backtest_report.csv")
    print(f"\n回测报告已保存: {output_dir / 'backtest_report.csv'}")

    return metrics


def main():
    """主函数"""
    print("=" * 60)
    print("PB/PE 价值投资策略回测")
    print("=" * 60)
    print(f"配置文件: {CONFIG_FILE}")
    print(f"数据目录: /Users/zengpengxin/workspace/DataBase/Quant/QlibBase/qlib_data_260415/qlib_etf_index")
    print()

    try:
        metrics = run_backtest()
        print("\n" + "=" * 60)
        print("回测完成!")
        print("=" * 60)
    except Exception as e:
        print(f"\n回测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
