# Import all collectors to trigger @register decorators
from .tencent_etf import TencentETFCollector
from .yahoo import YahooCollector
from .akshare import AkshareCollector
from .macro import MacroCollector
from .eastmoney_fund import EastmoneyFundCollector

__all__ = [
    "TencentETFCollector",
    "YahooCollector",
    "AkshareCollector",
    "MacroCollector",
    "EastmoneyFundCollector",
]
