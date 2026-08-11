"""
PipelineRunner：读取 pipeline.yaml，顺序执行各步骤
"""

import yaml
from pathlib import Path
from loguru import logger

from data_pipline.core.registry import get


class PipelineRunner:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.output_base = Path(cfg["output_base"]).expanduser().resolve()
        self.steps = cfg["pipelines"]
        self.output_base.mkdir(parents=True, exist_ok=True)

    def set_incremental(self, incremental: bool = True) -> None:
        """
        设置是否走增量更新模式。

        增量模式：将所有 collector 步骤的 force 覆盖为 False，
        使其通过 _incremental_start 从已有 CSV 末尾追加新数据；
        processors（merge/filter/clean/dump）无 force 概念，保持全量重建（幂等）。
        """
        for step_cfg in self.steps:
            if "force" in step_cfg:
                step_cfg["force"] = not incremental

    def run(self, only: list[str] | None = None):
        errors = []
        for i, step_cfg in enumerate(self.steps):
            cls_name = step_cfg["type"]
            step_name = step_cfg.get("name", cls_name)  # name 优先，无则回退 type
            if only and step_name not in only:
                continue
            logger.info(f"[{i+1}/{len(self.steps)}] Running {step_name} ({cls_name}) ...")
            try:
                get(cls_name)(step_cfg, self.output_base)()
                logger.info(f"  ✓ {step_name} done")
            except Exception as e:
                logger.error(f"  ✗ {step_name} failed: {e}")
                errors.append((step_name, e))

        if errors:
            logger.warning(f"\n{'='*40}\n{len(errors)} step(s) failed:")
            for name, err in errors:
                logger.warning(f"  - {name}: {err}")
