"""
TencentETFCollector — 腾讯财经 A股/港股 ETF K线采集器（CN 市场适配层）

共享拉取机制（分页回溯/DataFrame 构建/增量写盘）见 tencent_common.py，
本文件只描述 CN 市场差异：
    - 符号 6 位代码 → sh/sz 前缀
    - K 线数组键名随 fq_type 变化：不复权 "day"、前复权 "qfqday"、后复权 "hfqday"
    - 行固定 6 列（date/open/close/high/low/volume），无需截取
    - 复权可空（不复权标的走 fq_type=""）
"""

from data_pipline.collectors.tencent_common import TencentKlineCollector
from data_pipline.core.registry import register

_CN_API_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"


def _normalize_cn(symbol: str) -> str | None:
    """ETF 代码 → 腾讯格式（sh/sz 前缀），不支持时返回 None"""
    s = str(symbol).strip().zfill(6)
    if s.startswith(("51", "58")):
        return f"sh{s}"
    if s.startswith(("15", "56")):
        return f"sz{s}"
    return None


@register("TencentETFCollector")
class TencentETFCollector(TencentKlineCollector):
    """
    腾讯财经 CN ETF K线采集器。

    YAML 参数：
        symbols   : list[str|dict]  ETF 代码列表
        fq_type   : str             复权类型 hfq/qfq/""（默认 hfq）
        start     : str             全量起始日期（YAML anchor）
        force     : bool            True = 全量覆盖
        output_dir: str             输出子目录（相对 output_base）
        delay     : float           请求间隔秒数（默认 0.3）
    """

    _MARKET_SPEC = dict(
        base_url=_CN_API_URL,
        kline_key=lambda fq: f"{fq}day" if fq else "day",  # 不复权 "day" / 前复权 "qfqday" / 后复权 "hfqday"
        row_slice=slice(None),   # cn 行固定 6 列，全取
        fq_required=False,       # 允许空复权（不复权标的走 fq_type=""）
        default_fq="hfq",
    )
    _NORMALIZE = staticmethod(_normalize_cn)
