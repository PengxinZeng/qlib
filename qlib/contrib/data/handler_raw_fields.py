# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Raw Fields Data Handler
直接读取指定的原始字段，支持 YAML 配置字段列表
"""

from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.processor import Processor
from ...utils import get_callable_kwargs
from ...data.dataset import processor as processor_module
from inspect import getfullargspec


def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    new_l = []
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                assert (
                    fit_start_time is not None and fit_end_time is not None
                ), "Make sure `fit_start_time` and `fit_end_time` are not None."
                pkwargs.update(
                    {
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    }
                )
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l


_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]

_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class RawFieldsHandler(DataHandlerLP):
    """
    直接读取指定的原始字段，不生成计算特征
    支持通过 fields 参数配置需要读取的字段
    
    Example:
        handler = RawFieldsHandler(
            instruments="csi300",
            start_time="2020-01-01",
            end_time="2020-12-31",
            fields=["$open", "$high", "$low", "$close", "$volume", "$pb", "$pe_ttm"],
            field_names=["open", "high", "low", "close", "volume", "pb", "pe_ttm"]
        )
    """
    
    def __init__(
        self,
        instruments="all",
        start_time=None,
        end_time=None,
        freq="day",
        fields=None,
        field_names=None,
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        """
        初始化 RawFieldsHandler
        
        Parameters:
        -----------
        fields : list
            要读取的字段列表，使用 $ 前缀表示原始字段
            例如: ["$open", "$high", "$low", "$close", "$volume", "$pb"]
            
        field_names : list
            对应的输出列名列表
            例如: ["open", "high", "low", "close", "volume", "pb"]
            
        其他参数同 DataHandlerLP
        """
        
        # 默认字段配置：所有可用的原始字段
        if fields is None:
            fields = [
                "$close", "$high", "$low", "$open", "$volume", "$amount", "$pctchg",
                "$index_close", "$index_high", "$index_low", "$index_open", "$index_volume",
                "$pb", "$pb_equal_weight", "$pb_median",
                "$pe_ttm", "$pe_ttm_equal_weight", "$pe_ttm_median",
                "$pe_static", "$pe_static_equal_weight", "$pe_static_median",
            ]
        
        if field_names is None:
            field_names = [
                "close", "high", "low", "open", "volume", "amount", "pctchg",
                "index_close", "index_high", "index_low", "index_open", "index_volume",
                "pb", "pb_equal_weight", "pb_median",
                "pe_ttm", "pe_ttm_equal_weight", "pe_ttm_median",
                "pe_static", "pe_static_equal_weight", "pe_static_median",
            ]
        
        # 保存字段配置供 get_feature_config() 使用
        self._fields = fields
        self._field_names = field_names
        
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (self._fields, self._field_names),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_label_config(self):
        """获取标签配置"""
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
