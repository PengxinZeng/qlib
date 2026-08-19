#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT License.
import logging
import os
from pathlib import Path
import sys

# 限制 CPU 线程数（必须早于 import qlib / numpy / numexpr）。
# 默认 32（接近物理核数），避免过度限制影响特征工程速度；
# 若担心系统卡死可设 QLIB_RUN_THREADS 调低（如 4）；已有环境变量（用户/系统级）时优先保留。
for _var in ("NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("QLIB_RUN_THREADS", "32"))

import fire
from jinja2 import Template, meta
from ruamel.yaml import YAML

import qlib
from qlib.config import C
from qlib.log import get_module_logger
from qlib.model.trainer import task_train
from qlib.utils import set_log_with_config
from qlib.utils.data import update_config
from qlib.workflow.task.utils import replace_task_handler_with_cache

set_log_with_config(C.logging_config)
logger = get_module_logger("qrun", logging.INFO)


def set_below_normal_priority(enable: bool = True) -> bool:
    """
    将当前进程优先级降为 BelowNormal（Windows）/ nice +5（Unix），
    避免与用户交互式应用抢 CPU，保证系统响应。失败时静默降级（不影响运行）。

    Returns
    -------
    bool
        是否设置成功。
    """
    if not enable:
        return False
    try:
        if sys.platform == "win32":
            import ctypes

            # 64 位 Windows：HANDLE 是指针，必须显式声明 argtypes/restype，
            # 否则 ctypes 默认按 c_int(32 位) 传递导致句柄截断、SetPriorityClass 失败
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.SetPriorityClass.restype = ctypes.c_int
            kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            handle = kernel32.GetCurrentProcess()
            ok = bool(kernel32.SetPriorityClass(handle, 0x00004000))  # PROCESS_MODE_BELOW_NORMAL
            if ok:
                logger.info("Process priority set to BelowNormal.")
            return ok
        else:
            os.nice(5)  # Unix: nice +5
            logger.info("Process priority lowered (nice +5).")
            return True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to lower process priority (ignored): {e}")
        return False


def get_handler_cache_dir(cache_dir=None) -> "Path | None":
    """
    解析 handler 缓存目录（None 表示不启用缓存），优先级：
      1. 显式传入 cache_dir
      2. 环境变量 QLIB_HANDLER_CACHE_DIR
      3. QLIB_DATA_BASE 存在时回退 <QLIB_DATA_BASE>/all_weather_data/handler_cache
    """
    _dir = cache_dir or os.environ.get("QLIB_HANDLER_CACHE_DIR")
    if not _dir and os.environ.get("QLIB_DATA_BASE"):
        _dir = os.path.join(os.environ["QLIB_DATA_BASE"], "all_weather_data", "handler_cache")
    if not _dir:
        return None
    p = Path(_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def inject_handler_cache(config: dict, cache_dir=None) -> bool:
    """
    将 config["task"] 中的 dict 类型 handler 替换为 handler 缓存（file://xxx.pkl）。
    - 首次运行：构建缓存 pickle（含特征工程结果，dump_all=True）
    - 后续运行：直接反序列化加载，跳过 Loading data + ProcessInf + fit/process
    返回是否注入了缓存。
    """
    task = config.get("task")
    if not (task and isinstance(task.get("dataset", {}).get("kwargs", {}).get("handler"), dict)):
        return False
    cache_path = get_handler_cache_dir(cache_dir)
    if cache_path is None:
        return False
    task = replace_task_handler_with_cache(task, cache_dir=cache_path)
    config["task"] = task
    logger.info(f"Use handler cache dir: {cache_path}")
    return True


def get_path_list(path):
    if isinstance(path, str):
        return [path]
    else:
        return list(path)


def sys_config(config, config_path):
    """
    Configure the `sys` section

    Parameters
    ----------
    config : dict
        configuration of the workflow.
    config_path : str
        path of the configuration
    """
    sys_config = config.get("sys", {})

    # abspath
    for p in get_path_list(sys_config.get("path", [])):
        sys.path.append(p)

    # relative path to config path
    for p in get_path_list(sys_config.get("rel_path", [])):
        sys.path.append(str(Path(config_path).parent.resolve().absolute() / p))


def render_template(config_path: str) -> str:
    """
    render the template based on the environment

    Parameters
    ----------
    config_path : str
        configuration path

    Returns
    -------
    str
        the rendered content
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = f.read()
    # Set up the Jinja2 environment
    template = Template(config)

    # Parse the template to find undeclared variables
    env = template.environment
    parsed_content = env.parse(config)
    variables = meta.find_undeclared_variables(parsed_content)

    # Get context from os.environ according to the variables
    # Windows 环境变量通常含反斜杠（如 D:\data），渲染进 yaml 双引号字符串会触发
    # 非法转义（\P 等），统一规范化为正斜杠。
    context = {var: os.getenv(var, "").replace("\\", "/") for var in variables if var in os.environ}
    logger.info(f"Render the template with the context: {context}")

    # Render the template with the context
    rendered_content = template.render(context)
    return rendered_content


def load_config(config_path):
    """
    Load and process the configuration from the given config file path.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    dict
        The processed configuration dictionary.
    """
    # Render the template
    rendered_yaml = render_template(config_path)
    yaml = YAML(typ="safe", pure=True)
    config = yaml.load(rendered_yaml)

    base_config_path = config.get("BASE_CONFIG_PATH", None)
    if base_config_path:
        logger.info(f"Use BASE_CONFIG_PATH: {base_config_path}")
        base_config_path = Path(base_config_path)

        # it will find config file in absolute path and relative path
        if base_config_path.exists():
            path = base_config_path
        else:
            logger.info(
                f"Can't find BASE_CONFIG_PATH base on: {Path.cwd()}, "
                f"try using relative path to config path: {Path(config_path).absolute()}"
            )
            relative_path = Path(config_path).absolute().parent.joinpath(base_config_path)
            if relative_path.exists():
                path = relative_path
            else:
                raise FileNotFoundError(f"Can't find the BASE_CONFIG file: {base_config_path}")

        with open(path, encoding="utf-8") as fp:
            yaml = YAML(typ="safe", pure=True)
            base_config = yaml.load(fp)
        logger.info(f"Load BASE_CONFIG_PATH succeed: {path.resolve()}")
        config = update_config(base_config, config)

    # config the `sys` section
    sys_config(config, config_path)

    return config


# workflow handler function
def workflow(config_path, experiment_name="workflow", uri_folder="mlruns", cache_dir=None):
    """
    This is a Qlib CLI entrance.
    User can run the whole Quant research workflow defined by a configure file
    - the code is located here ``qlib/cli/run.py``

    User can specify a base_config file in your workflow.yml file by adding "BASE_CONFIG_PATH".
    Qlib will load the configuration in BASE_CONFIG_PATH first, and the user only needs to update the custom fields
    in their own workflow.yml file.

    For examples:

        qlib_init:
            provider_uri: "~/.qlib/qlib_data/cn_data"
            region: cn
        BASE_CONFIG_PATH: "workflow_config_lightgbm_Alpha158_csi500.yaml"
        market: csi300

    cache_dir : str, optional
        handler 缓存目录（如 examples/benchmarks/MLP/_handler_cache）。
        - 传 None 时按环境变量 QLIB_HANDLER_CACHE_DIR -> QLIB_DATA_BASE/all_weather_data/handler_cache 自动解析
        - 若均未配置则跳过缓存（行为与官方 qrun 完全一致）
        首次运行构建缓存 pickle；后续运行直接加载，跳过 Loading data + ProcessInf + fit/process（约省 60-80s/次）。

    """
    # 默认降低进程优先级（可用环境变量 QLIB_LOW_PRIORITY=0 关闭），保证系统响应
    set_below_normal_priority(os.environ.get("QLIB_LOW_PRIORITY", "1") != "0")

    config = load_config(config_path)

    if "exp_manager" in config.get("qlib_init"):
        qlib.init(**config.get("qlib_init"))
    else:
        exp_manager = C["exp_manager"]
        exp_manager["kwargs"]["uri"] = "file:" + str(Path(os.getcwd()).resolve() / uri_folder)
        qlib.init(**config.get("qlib_init"), exp_manager=exp_manager)

    # handler 缓存注入（须在 qlib.init 之后：缓存构建需要 qlib 环境）
    inject_handler_cache(config, cache_dir)

    if "experiment_name" in config:
        experiment_name = config["experiment_name"]
    else:
        raise ValueError("`experiment_name` not found in config, please specify it in the config file or command line")
    recorder = task_train(config.get("task"), experiment_name=experiment_name)
    recorder.save_objects(config=config)


# function to run workflow by config
def run():
    fire.Fire(workflow)


if __name__ == "__main__":
    run()
