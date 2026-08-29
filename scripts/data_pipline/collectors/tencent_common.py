"""
tencent_common — 腾讯财经 K线采集共享机制（CN/US 市场共用）

抽取腾讯不同市场 K 线接口的公共逻辑：
- fetch_segment: 单段拉取（含重试）
- fetch_since  : 分段向前回溯 + DataFrame 构建
- TencentKlineCollector: Collector 公共骨架（__init__ / __call__）

市场差异不体现在逻辑分支中，而是由子类的 _MARKET_SPEC 声明式描述：
    base_url    : str                  端点 URL（cn: fqkline/get, us: usfqkline/get）
    kline_key   : Callable[[str], str] K 线数组键名策略（fq_type -> 键名）
    row_slice   : slice                对每行的列截取（cn 6 列全取, us 11 列取前 6）
    fq_required : bool                 是否强制要求复权参数（us 必须 qfq/hfq）
    default_fq  : str                  默认复权类型（cn: hfq, us: qfq）
"""

import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd
import requests
from loguru import logger
from tqdm import tqdm

# 使 scripts/ 可导入（复用 BaseCollector / utils）
_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_SCRIPTS_DIR))

from data_pipline.collectors.base import BaseCollector
from data_pipline.utils.http import retry

# 腾讯 K 线接口单次最大返回条数
MAX_PER_REQ = 800
# 分段上限：800 * 50 = 40000 条，足够全量历史
_MAX_SEGMENTS = 50


# ──────────────────────────────────────────────────────────────
# 共享拉取机制
# ──────────────────────────────────────────────────────────────

@retry(max_tries=3, delay=1.0, backoff=2.0, exceptions=(Exception,))
def fetch_segment(
    base_url: str,
    tencent_sym: str,
    end_date: str,
    fq_type: str,
    kline_key: Callable[[str], str],
) -> list:
    """拉取单段 K线（最多 MAX_PER_REQ 条）。"""
    params = f"{tencent_sym},day,,{end_date},{MAX_PER_REQ},{fq_type}"
    resp = requests.get(
        f"{base_url}?param={params}",
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    data = resp.json()
    if data.get("code") == 0 and isinstance(data.get("data"), dict):
        market_data = data["data"].get(tencent_sym, {})
        key = kline_key(fq_type)
        if key not in market_data:
            if fq_type:
                # 部分标的接口不提供复权数据（如 563020），不应静默回退到未复权 "day"，
                # 否则会产出错误的除权价格，破坏下游一致性假设。
                raise ValueError(
                    f"{tencent_sym}: 接口未返回 '{key}'（fq_type={fq_type}），"
                    f"可用键: {list(market_data.keys())}"
                )
            return []
        return market_data[key]
    return []


def fetch_since(
    base_url: str,
    tencent_sym: str,
    start_date: str,
    fq_type: str,
    delay: float,
    kline_key: Callable[[str], str],
    row_slice: slice = slice(None),
) -> pd.DataFrame:
    """
    分段向前拉取，直到 start_date 之前的数据，返回 >= start_date 的记录。

    row_slice: 对每行的列截取。腾讯不同市场返回列数不同：
               cn 固定 6 列（slice(None) 全取）；us 11 列取前 6（slice(6)）。
    """
    all_rows = []
    end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    for _ in range(_MAX_SEGMENTS):  # 最多 50 段，约 40000 条，足够全量
        rows = fetch_segment(base_url, tencent_sym, end_date, fq_type, kline_key)
        if not rows:
            break
        rows = [row[row_slice] for row in rows]
        all_rows.extend(rows)
        # 若最早日期已在 start_date 之前，停止
        if rows[0][0] <= start_date:
            break
        if len(rows) < MAX_PER_REQ:
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


# ──────────────────────────────────────────────────────────────
# Collector 公共骨架
# ──────────────────────────────────────────────────────────────

class TencentKlineCollector(BaseCollector):
    """
    腾讯 K线采集器公共骨架：市场差异全部由子类声明，本类不含市场分支。

    子类需覆盖：
        _MARKET_SPEC : dict(base_url, kline_key, row_slice, fq_required, default_fq)
        _NORMALIZE   : Callable[[str], str | None]  符号 → 腾讯格式（如 "NDX" → "usNDX"）
    """

    _MARKET_SPEC: dict = {}
    _NORMALIZE = staticmethod(lambda symbol: symbol)

    def __init__(self, cfg: dict, output_base: Path):
        super().__init__(cfg, output_base)
        self.spec = self._MARKET_SPEC
        # symbols 支持两种格式：
        #   - str: "510300"
        #   - dict: {code: "510300", name: "沪深300ETF", index: "000300"}
        raw_symbols = cfg.get("symbols", [])
        self.symbols: list[dict] = [
            s if isinstance(s, dict) else {"code": s, "name": s, "index": ""}
            for s in raw_symbols
        ]
        self.fq_type: str = cfg.get("fq_type", self.spec.get("default_fq", "hfq"))
        self.delay: float = float(cfg.get("delay", 0.3))
        if self.spec.get("fq_required") and not self.fq_type:
            raise ValueError(
                f"{type(self).__name__}: fq_type 必须为 qfq/hfq（该接口不支持空复权）"
            )

    def __call__(self) -> None:
        cls_name = type(self).__name__
        if not self.symbols:
            logger.warning(f"{cls_name}: no symbols specified")
            return

        logger.info(f"{cls_name}: {len(self.symbols)} symbols → {self.output_dir}")

        failed = []
        for sym_info in tqdm(self.symbols, desc=cls_name):
            code = sym_info["code"]
            name = sym_info.get("name", code)
            tencent_sym = self._NORMALIZE(code)
            if not tencent_sym:
                logger.warning(f"  [{code}] unsupported symbol format, skip")
                failed.append(code)
                continue
            try:
                inc_start = self._incremental_start(code)
                new_df = fetch_since(
                    self.spec["base_url"],
                    tencent_sym,
                    inc_start,
                    self.fq_type,
                    self.delay,
                    kline_key=self.spec["kline_key"],
                    row_slice=self.spec["row_slice"],
                )
                if not new_df.empty:
                    # 附加 name/index 元信息列，方便下游使用
                    new_df["name"] = name
                    new_df["track_index"] = sym_info.get("index", "")
                self._append_csv(code, new_df)
            except Exception as e:
                logger.error(f"  [{code}] failed: {e}")
                failed.append(code)

        if failed:
            logger.warning(f"{cls_name}: {len(failed)} failed: {failed}")
