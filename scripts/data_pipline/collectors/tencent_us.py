"""
TencentUSCollector — 腾讯财经 美股/指数 K线采集器（US 市场适配层）

使用 usfqkline/get 端点（普通 fqkline/get 对 us 标的仅返回最近 1 天）。
与 CN 的差异（其余逻辑复用 tencent_common.TencentKlineCollector）：
    - 符号加小写 "us" 前缀透传（"NDX" → "usNDX"）
    - K 线数组键名恒为 "day"（CN 随 fq_type 变化为 qfqday/hfqday）
    - 每行 11 列，取前 6 列（date/open/close/high/low/volume）
    - 复权参数必须为 qfq/hfq（空复权返回 0 行）；指数 qfq == hfq
"""

from data_pipline.collectors.tencent_common import TencentKlineCollector
from data_pipline.core.registry import register

_US_API_URL = "https://web.ifzq.gtimg.cn/appstock/app/usfqkline/get"


def _normalize_us(symbol: str) -> str | None:
    """美股/指数代码 → 腾讯格式（小写 us 前缀），空值返回 None"""
    s = str(symbol).strip()
    if not s:
        return None
    return f"us{s}" if not s.lower().startswith("us") else s.lower()


@register("TencentUSCollector")
class TencentUSCollector(TencentKlineCollector):
    """
    腾讯财经 US 指数/ETF K线采集器。

    YAML 参数（同 TencentETFCollector）：
        symbols   : list[str|dict]  代码列表（如 "NDX"、"DJI"、"QQQ"）
        fq_type   : str             复权类型，必须 qfq/hfq（默认 qfq）
        start     : str             全量起始日期（YAML anchor）
        force     : bool            True = 全量覆盖
        output_dir: str             输出子目录（相对 output_base）
        delay     : float           请求间隔秒数（默认 0.3）

    注意：code 不带 "us" 前缀（如 "NDX"），CSV 文件名/下游 symbol 即为 "NDX"。
    """

    _MARKET_SPEC = dict(
        base_url=_US_API_URL,
        kline_key=lambda fq: "day",  # usfqkline 的 K 线键名恒为 "day"
        row_slice=slice(6),          # us 行 11 列，取前 6
        fq_required=True,            # 必须 qfq/hfq（空复权返回 0 行）
        default_fq="qfq",
    )
    _NORMALIZE = staticmethod(_normalize_us)
