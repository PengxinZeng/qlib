"""
TencentETFCollector — 腾讯财经 ETF K线采集器

复用 data_collector/tencent_etf/collector.py 的 API 逻辑，
封装增量更新：读取已有 CSV 末尾日期，仅拉取缺失部分并追加。
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

# 使 data_collector 可导入
_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SCRIPTS_DIR))

from data_pipline.collectors.base import BaseCollector
from data_pipline.core.registry import register
from data_pipline.utils.http import retry

_API_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
_MAX_PER_REQ = 800


def _normalize_symbol(symbol: str) -> str | None:
    """ETF 代码 → 腾讯格式（sh/sz 前缀），不支持时返回 None"""
    s = str(symbol).strip().zfill(6)
    if s.startswith(("51", "58")):
        return f"sh{s}"
    if s.startswith(("15", "56")):
        return f"sz{s}"
    return None


@retry(max_tries=3, delay=1.0, backoff=2.0, exceptions=(Exception,))
def _fetch_segment(tencent_sym: str, end_date: str, fq_type: str) -> list:
    """拉取单段 K线（最多 _MAX_PER_REQ 条）"""
    params = f"{tencent_sym},day,,{end_date},{_MAX_PER_REQ},{fq_type}"
    resp = requests.get(
        f"{_API_URL}?param={params}",
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    data = resp.json()
    if data.get("code") == 0 and isinstance(data.get("data"), dict):
        etf_data = data["data"].get(tencent_sym, {})
        # 腾讯接口的K线数据键名随 fq_type 变化：
        # 不复权 -> "day"，前复权 -> "qfqday"，后复权 -> "hfqday"
        key = f"{fq_type}day" if fq_type else "day"
        if key not in etf_data:
            if fq_type:
                # 部分标的接口不提供复权数据（如 563020），此时不应静默回退到
                # 未复权 "day"，否则会产出错误的除权价格，破坏下游一致性假设。
                raise ValueError(
                    f"{tencent_sym}: 接口未返回 '{key}'（fq_type={fq_type}），"
                    f"可用键: {list(etf_data.keys())}"
                )
            return []
        return etf_data[key]
    return []


def _fetch_since(tencent_sym: str, start_date: str, fq_type: str, delay: float) -> pd.DataFrame:
    """
    分段向前拉取，直到 start_date 之前的数据，返回 >= start_date 的记录。
    """
    all_rows = []
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    for _ in range(50):  # 最多 50 段，约 40000 条，足够全量
        rows = _fetch_segment(tencent_sym, end_date, fq_type)
        if not rows:
            break
        all_rows.extend(rows)
        # 若最早日期已在 start_date 之前，停止
        if rows[0][0] <= start_date:
            break
        if len(rows) < _MAX_PER_REQ:
            break
        end_date = (pd.Timestamp(rows[0][0]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        time.sleep(delay)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["date", "open", "close", "high", "low", "volume"])
    df = df[df["date"] >= start_date]
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    for col in ["open", "close", "high", "low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    return df


@register("TencentETFCollector")
class TencentETFCollector(BaseCollector):
    """
    腾讯财经 ETF K线采集器。

    YAML 参数：
        symbols   : list[str]  ETF 代码列表
        fq_type   : str        复权类型 hfq/qfq/""（默认 hfq）
        start     : str        全量起始日期（YAML anchor）
        force     : bool       True = 全量覆盖
        output_dir: str        输出子目录（相对 output_base）
        delay     : float      请求间隔秒数（默认 0.3）
    """

    def __init__(self, cfg: dict, output_base: Path):
        super().__init__(cfg, output_base)
        # symbols 支持两种格式：
        #   - str: "510300"
        #   - dict: {code: "510300", name: "沪深300ETF", index: "000300"}
        raw_symbols = cfg.get("symbols", [])
        self.symbols: list[dict] = [
            s if isinstance(s, dict) else {"code": s, "name": s, "index": ""}
            for s in raw_symbols
        ]
        self.fq_type: str = cfg.get("fq_type", "hfq")
        self.delay: float = float(cfg.get("delay", 0.3))

    def __call__(self) -> None:
        if not self.symbols:
            logger.warning("TencentETFCollector: no symbols specified")
            return

        logger.info(f"TencentETFCollector: {len(self.symbols)} symbols → {self.output_dir}")

        failed = []
        for sym_info in tqdm(self.symbols, desc="TencentETF"):
            code = sym_info["code"]
            name = sym_info.get("name", code)
            tencent_sym = _normalize_symbol(code)
            if not tencent_sym:
                logger.warning(f"  [{code}] unsupported symbol format, skip")
                failed.append(code)
                continue
            try:
                inc_start = self._incremental_start(code)
                new_df = _fetch_since(tencent_sym, inc_start, self.fq_type, self.delay)
                if not new_df.empty:
                    # 附加 name/index 元信息列，方便下游使用
                    new_df["name"] = name
                    new_df["track_index"] = sym_info.get("index", "")
                self._append_csv(code, new_df)
            except Exception as e:
                logger.error(f"  [{code}] failed: {e}")
                failed.append(code)

        if failed:
            logger.warning(f"TencentETFCollector: {len(failed)} failed: {failed}")
