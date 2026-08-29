# Project Guidelines — qlib 量化工作区（Windows）

本仓库是 **qlib 库源码 + 量化实验代码的混合仓库**：`qlib/` 是库本体，`examples/benchmarks/` 是 workflow 配置（yaml），`scripts/` 是自定义实验脚本，`output/` 是运行日志。**数据不在本仓库**，位于 `QuantDataBank/`（路径解析见 `scripts/path_config.py`）。

## Architecture

- `qlib/` — qlib 库本体（本仓库同时维护库源码与实验，改动库代码时注意别破坏其他 benchmark）
- `examples/benchmarks/MLP/` — 全天候 ETF 实验的 workflow yaml（每个实验一个 yaml，`experiment_name` 即 mlflow 实验名）
- `scripts/` — 自定义实验脚本（`run_*_5seed.py` 多 seed 实验、`verify_inference_repro.py` 推理复现、`daily_update.py` 数据更新、`data_pipline/` 数据管道）
- `output/` — 运行日志统一输出目录
- `mlruns/` — mlflow 实验结果记录

**数据根**：默认 `D:/Pengxin/CodeBase/Quant/QuantDataBank`，由 `scripts/path_config.py` 集中解析；环境变量 `QLIB_ROOT` / `QLIB_DATA_BASE` / `QLIB_PYTHON` / `QLIB_QRUN` 可覆盖。workflow yaml 用 `{{ QLIB_DATA_BASE }}` 模板占位、运行时注入，不要在 yaml 里写死绝对路径。

## Build and Test — 运行实验（Windows 强制约定）

- **Python 解释器**：一律使用 qlib conda 环境 `D:\Pengxin\software\Anaconda\envs\qlib\python.exe`，绝不使用系统/其他环境 python。
- **前台运行**：不加 `&`（不后台化），跑完立即分析结论。
- **标准命令形式**（PowerShell 中执行，用 `cmd /c` 重定向，日志合并到**单个**文件 `output/<脚本名>.log`）：

```powershell
cmd /c '"D:\Pengxin\software\Anaconda\envs\qlib\python.exe" -u -X utf8 <script.py> <args...> > output/<script>.log 2>&1'
```

- **运行 workflow yaml**：不要直接调用 `qrun.exe`（Windows 下易因环境/编码失败），用模块形式：

```powershell
cmd /c '"D:\Pengxin\software\Anaconda\envs\qlib\python.exe" -u -X utf8 -m qlib.cli.run examples/benchmarks/MLP/<workflow>.yaml > output/<workflow>.log 2>&1'
```

## Conventions — Windows 已知坑

- **必须用 `cmd /c` 重定向，勿用 PowerShell `2>&1`**：qlib 日志（`logging.StreamHandler`）写到 **stderr**，PowerShell 会把 stderr 逐行包装成 ErrorRecord、**按 80 列强制换行**并加 `python.exe :` 前缀（长路径/日志被截断）；`cmd /c` 是字节流透传，不换行、不加前缀。
- **必须加 `-X utf8`**：Windows 控制台默认 GBK(cp936)，不加则中文日志、tqdm `█` 块字符在 VS Code 中乱码；`-X utf8`（Python 3.7+ UTF-8 模式）强制 UTF-8 输出。脚本自身已 reconfigure 的（如 `verify_inference_repro.py`）可省略，但统一保留更保险。
- **加 `-u`（unbuffered）**：保证日志实时写入文件。
- **日志只写 `output/`**：stdout+stderr 合并到单个文件，不要拆成 `.log` + `.err.log`；目录不存在时先创建。
- **实验配置模式**：新实验遵循"新建 yaml，不动现有版本"；多 seed 实验优先复用 `scripts/run_*_5seed.py` 类脚本而非手写循环。
