"""
组件注册表（Registry Pattern）

用法：
    @register("MyCollector")
    class MyCollector(BaseCollector):
        ...

    cls = get("MyCollector")   # 返回 MyCollector 类
"""

from typing import Type

_REGISTRY: dict[str, Type] = {}


def register(name: str):
    """类装饰器：将类注册到全局注册表"""
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator


def get(name: str) -> Type:
    if name not in _REGISTRY:
        raise KeyError(
            f"Component '{name}' not registered. Available: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]
