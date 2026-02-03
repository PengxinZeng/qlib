
python -m qlib.cli.data qlib_data --target_dir D:/Pengxin/CodeBase/Quant/QuantDataBank/qlib_bin --region cn
python scripts/check_data_health.py check_data --qlib_dir D:/Pengxin/CodeBase/Quant/QuantDataBank/gold_source

qrun benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml
qrun examples/benchmarks/MLP/workflow_config_mlp_Alpha360.yaml
qrun benchmarks/LightGBM/workflow_config_lightgbm_gold.yaml


python scripts/dump_bin.py dump_all `
    --data_path D:/Pengxin/CodeBase/Quant/QuantDataBank/CustomData `
    --qlib_dir D:/Pengxin/CodeBase/Quant/QuantDataBank/CustomData `
    --include_fields open,high,low,close,volume,factor `
    --date_field_name date

python scripts/data_collector/yahoo/collector.py download_data `
    --source_dir D:/Pengxin/CodeBase/Quant/QuantDataBank/gold_source `
    --start_date 2025-01-01 `
    --end_date 2026-12-31 `
    --delay 10 `
    --code_list "GC=F"
    
python D:/Pengxin/CodeBase/Quant/qlib/scripts/dump_bin.py dump_all `
    --data_path D:/Pengxin/CodeBase/Quant/QuantDataBank/gold_source/ `
    --qlib_dir D:/Pengxin/CodeBase/Quant/QuantDataBank/gold_source/ `
    --include_fields open,high,low,close,volume,factor `
    --date_field_name date
