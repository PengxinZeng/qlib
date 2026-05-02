# 环境
```
conda activate rdagent
```

# 数据
流程是1. 下载数据; 2. 转化为qlib bin; 3. 数据质量检查
## 下载数据
### qlib官方csi300数据下载
```
python -m qlib.cli.data qlib_data --target_dir ~/.qlib/qlib_data/cn_data --region cn
```

### yahoo数据下载
```
python scripts/data_collector/yahoo/collector.py download_data `
    --source_dir ~/.qlib/qlib_data/gold_source `
    --start_date 2025-01-01 `
    --end_date 2026-12-31 `
    --delay 10 `
    --code_list "GC=F"
```

## 转化为qlib bin
```
python scripts/dump_bin.py dump_all \
    --data_path ~/.qlib/qlib_data/gold_source/ \
    --qlib_dir ~/.qlib/qlib_data/gold_source/ \
    --include_fields open,high,low,close,volume,factor \
    --date_field_name date
```

## 数据质量检查
```
python scripts/check_data_health.py check_data --qlib_dir ~/.qlib/qlib_data/gold_source
```

# 模型
## 训练
```
python qlib/cli/run.py examples/benchmarks/HistRelaPB/workflow_config.yaml
# 训练结果保存在mlruns/<experiment_id>/<recorder_id>/; 
# 其中experiment_id，recorder_id在训练日志中

cd /Users/zengpengxin/workspace/CodeBase/qlib && conda activate rdagent && python qlib/cli/run.py examples/benchmarks/HistRelaPB/workflow_config.yaml
```

## Tuner
```
python qlib/contrib/tuner/launcher.py -c /Users/zengpengxin/workspace/CodeBase/qlib/examples/benchmarks/HistRelaPB/tuner_config.yaml
```

## 查看实验结果
ls mlruns/<experiment_id>/<recorder_id>/artifacts/
```

# 代码
```
git add .
git commit -m "Hist Rela Pb"
git push origin main
```

# Prompts
请帮我使用qlib/contrib/tuner/tuner.py，调优/Users/zengpengxin/workspace/CodeBase/qlib/examples/benchmarks/HistRelaPB/workflow_config.yaml里面的参数
先使用较小迭代次数验证代码跑通

注意：
1. 环境：conda activate rdagent
2. 尽量在复用原有代码，不要自己新写整个工作流程

