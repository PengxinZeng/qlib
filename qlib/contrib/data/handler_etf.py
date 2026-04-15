# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
ETF/Stock Data Handler without VWAP
适用于只有OHLCV+factor的数据
"""

from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import Processor
from qlib.utils import get_callable_kwargs
from qlib.data.dataset import processor as processor_module
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


class AlphaETF(DataHandlerLP):
    """
    ETF/Stock Data Handler without VWAP
    基于OHLCV生成158个特征（不含VWAP相关特征）
    """
    def __init__(
        self,
        instruments="all",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
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

    @staticmethod
    def get_feature_config():
        """
        生成基于OHLCV的特征配置（不含VWAP）
        """
        fields = []
        names = []
        
        # KBAR特征 (9个)
        fields += [
            "($close-$open)/$open",
            "($high-$low)/$open",
            "($close-$open)/($high-$low+1e-12)",
            "($high-Greater($open, $close))/$open",
            "($high-Greater($open, $close))/($high-$low+1e-12)",
            "(Less($open, $close)-$low)/$open",
            "(Less($open, $close)-$low)/($high-$low+1e-12)",
            "(2*$close-$high-$low)/$open",
            "(2*$close-$high-$low)/($high-$low+1e-12)",
        ]
        names += [
            "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",
        ]
        
        # 价格特征 (OPEN, HIGH, LOW, CLOSE在不同时间窗口)
        price_windows = [0, 1, 2, 3, 4]
        for field in ["OPEN", "HIGH", "LOW", "CLOSE"]:
            field_lower = field.lower()
            fields += ["Ref($%s, %d)/$close" % (field_lower, d) if d != 0 else "$%s/$close" % field_lower for d in price_windows]
            names += [field + str(d) for d in price_windows]
        
        # 成交量特征
        volume_windows = [0, 1, 2, 3, 4]
        fields += ["Ref($volume, %d)/($volume+1e-12)" % d if d != 0 else "$volume/($volume+1e-12)" for d in volume_windows]
        names += ["VOLUME" + str(d) for d in volume_windows]
        
        # Rolling特征
        windows = [5, 10, 20, 30, 60]
        
        # ROC: Rate of Change
        fields += ["Ref($close, %d)/$close" % d for d in windows]
        names += ["ROC%d" % d for d in windows]
        
        # MA: Moving Average
        fields += ["Mean($close, %d)/$close" % d for d in windows]
        names += ["MA%d" % d for d in windows]
        
        # STD: Standard Deviation
        fields += ["Std($close, %d)/$close" % d for d in windows]
        names += ["STD%d" % d for d in windows]
        
        # BETA: Slope
        fields += ["Slope($close, %d)/$close" % d for d in windows]
        names += ["BETA%d" % d for d in windows]
        
        # RSQR: R-squared
        fields += ["Rsquare($close, %d)" % d for d in windows]
        names += ["RSQR%d" % d for d in windows]
        
        # RESI: Residual
        fields += ["Resi($close, %d)/$close" % d for d in windows]
        names += ["RESI%d" % d for d in windows]
        
        # MAX: Maximum High
        fields += ["Max($high, %d)/$close" % d for d in windows]
        names += ["MAX%d" % d for d in windows]
        
        # MIN: Minimum Low
        fields += ["Min($low, %d)/$close" % d for d in windows]
        names += ["MIN%d" % d for d in windows]
        
        # QTLU: 80% Quantile
        fields += ["Quantile($close, %d, 0.8)/$close" % d for d in windows]
        names += ["QTLU%d" % d for d in windows]
        
        # QTLD: 20% Quantile
        fields += ["Quantile($close, %d, 0.2)/$close" % d for d in windows]
        names += ["QTLD%d" % d for d in windows]
        
        # RSV: Relative Strength Value
        fields += ["($close-Min($low, %d))/(Max($high, %d)-Min($low, %d)+1e-12)" % (d, d, d) for d in windows]
        names += ["RSV%d" % d for d in windows]
        
        # CORR: Correlation
        fields += ["Corr($close, Log($volume+1), %d)" % d for d in windows]
        names += ["CORR%d" % d for d in windows]
        
        # CORD: Delta Correlation
        fields += ["Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), %d)" % d for d in windows]
        names += ["CORD%d" % d for d in windows]
        
        # CNTP: Count Positive
        fields += ["Mean($close>Ref($close, 1), %d)" % d for d in windows]
        names += ["CNTP%d" % d for d in windows]
        
        # CNTN: Count Negative
        fields += ["Mean($close<Ref($close, 1), %d)" % d for d in windows]
        names += ["CNTN%d" % d for d in windows]
        
        # CNTD: Count Delta
        fields += ["Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)" % (d, d) for d in windows]
        names += ["CNTD%d" % d for d in windows]
        
        # SUMP: Sum of Positive Returns
        fields += ["Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d) for d in windows]
        names += ["SUMP%d" % d for d in windows]
        
        # SUMN: Sum of Negative Returns
        fields += ["Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d) for d in windows]
        names += ["SUMN%d" % d for d in windows]
        
        # SUMD: Sum Delta
        fields += ["(Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d))/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d, d) for d in windows]
        names += ["SUMD%d" % d for d in windows]
        
        # VMA: Volume Moving Average
        fields += ["Mean($volume, %d)/($volume+1e-12)" % d for d in windows]
        names += ["VMA%d" % d for d in windows]
        
        # VSTD: Volume Standard Deviation
        fields += ["Std($volume, %d)/($volume+1e-12)" % d for d in windows]
        names += ["VSTD%d" % d for d in windows]
        
        # WVMA: Weighted Volume MA
        fields += ["Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)" % (d, d) for d in windows]
        names += ["WVMA%d" % d for d in windows]
        
        # VSUMP: Volume Sum Positive
        fields += ["Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d) for d in windows]
        names += ["VSUMP%d" % d for d in windows]
        
        # VSUMN: Volume Sum Negative
        fields += ["Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d) for d in windows]
        names += ["VSUMN%d" % d for d in windows]
        
        # VSUMD: Volume Sum Delta
        fields += ["(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d) for d in windows]
        names += ["VSUMD%d" % d for d in windows]
        
        return fields, names

    @staticmethod
    def get_label_config():
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]
