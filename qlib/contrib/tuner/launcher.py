# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: skip-file
# flake8: noqa

# coding=utf-8

import argparse
import importlib
import os
import yaml

from qlib.contrib.tuner.config import TunerConfigManager


args_parser = argparse.ArgumentParser(prog="tuner")
args_parser.add_argument(
    "-c",
    "--config_path",
    required=False,
    type=str,
    default='/Users/zengpengxin/workspace/CodeBase/qlib/examples/benchmarks/HistRelaPB/tuner_config.yaml',
    help="config path indicates where to load yaml config.",
)

args = args_parser.parse_args()

TUNER_CONFIG_MANAGER = TunerConfigManager(args.config_path)


def run():
    # 1. Get pipeline class.
    tuner_pipeline_class = getattr(importlib.import_module(".pipeline", package="qlib.contrib.tuner"), "Pipeline")
    # 2. Init tuner pipeline.
    tuner_pipeline = tuner_pipeline_class(TUNER_CONFIG_MANAGER)
    # 3. Begin to tune
    tuner_pipeline.run()

if __name__ == "__main__":
    run()
