"""
run_pipeline.py — 全天候数据 Pipeline 执行入口

用法：
    # 执行全部步骤
    python run_pipeline.py

    # 只执行指定 type 的步骤（可多个）
    python run_pipeline.py --only TencentETFCollector YahooCollector

    # 使用指定配置文件
    python scripts/data_pipline/run_pipeline.py --config scripts/data_pipline/pipeline.yaml --only equity_cn
"""

import sys
import logging
from pathlib import Path
import argparse

from loguru import logger

# 将当前目录及其父目录加入 sys.path
# - data_pipline/ 自身：使 core/collectors/processors 可作为顶层包 import
# - scripts/：使 data_pipline 可作为包 import（collectors 内部使用 data_pipline.* 路径）
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from data_pipline.core.pipeline import PipelineRunner
import data_pipline.collectors  # noqa: F401 — 触发 @register 注册
import data_pipline.processors  # noqa: F401


class _InterceptHandler(logging.Handler):
    """把标准 logging 记录转发到 loguru，使 processors 里 logging.getLogger 的
    INFO 日志与 PipelineRunner 的 loguru 输出统一可见。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # 回溯到真正的调用栈帧，保证日志显示正确的来源
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def setup_logging(level: int = logging.INFO) -> None:
    """将标准 logging 全量重定向到 loguru（覆盖已有 handler）。"""
    logging.basicConfig(handlers=[_InterceptHandler()], level=level, force=True)


def main():
    parser = argparse.ArgumentParser(description="全天候数据 Pipeline 执行器")
    parser.add_argument("--config", default="pipeline.yaml", help="YAML 配置文件路径")
    parser.add_argument(
        "--only",
        nargs="*",
        metavar="TYPE",
        help="只运行指定 type 的步骤，默认全部",
    )
    args = parser.parse_args()

    setup_logging()

    runner = PipelineRunner(args.config)
    runner.run(only=args.only)


if __name__ == "__main__":
    main()
