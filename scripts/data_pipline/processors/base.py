"""
BaseProcessor — 抽象处理器基类（Merger / Dumper 继承此类）
"""

import abc
from pathlib import Path


class BaseProcessor(abc.ABC):

    def __init__(self, cfg: dict, output_base: Path):
        self.cfg = cfg
        self.output_base = output_base.resolve()

    @abc.abstractmethod
    def __call__(self) -> None:
        """读取源数据，处理后写出"""
