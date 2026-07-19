"""
通用 HTTP 工具：指数退避重试装饰器
"""

import time
import functools
from loguru import logger


def retry(max_tries: int = 5, delay: float = 1.0, backoff: float = 2.0, exceptions=(Exception,)):
    """
    指数退避重试装饰器。

    Parameters
    ----------
    max_tries : int
        最大尝试次数（含第一次）
    delay : float
        初始等待秒数
    backoff : float
        退避倍数，每次失败后等待时间 *= backoff
    exceptions : tuple
        需要捕获的异常类型
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            wait = delay
            for attempt in range(1, max_tries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_tries:
                        logger.error(f"{func.__name__} failed after {max_tries} attempts: {e}")
                        raise
                    logger.warning(
                        f"{func.__name__} attempt {attempt}/{max_tries} failed: {e}. "
                        f"Retrying in {wait:.1f}s ..."
                    )
                    time.sleep(wait)
                    wait *= backoff
        return wrapper
    return decorator
