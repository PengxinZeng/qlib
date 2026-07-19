"""
SanityChecker — 用人工统计的 anno file 校验 source_dirs 中下载数据的准确性。

处理逻辑：
1. 读取 anno file（如 EtfCompare/eft_data.csv），按"代码"分组，每行是一条
   (start_date, start_price, end_date, end_price, annualized) 记录。
2. 对每条 anno 记录，在 source_dirs 下按代码找到对应 symbol CSV，
   取 start_date/end_date 当天的 close 价格。
   - 若该日期不是交易日（在 CSV 中不存在），标记为检测失败（无法对比）。
3. 用 source 数据重新计算年化收益率，与 anno file 中的"年化"列对比，
   误差 <= tolerance（默认 1e-5）则通过。
4. 无代码（无法匹配 symbol）的 anno 行，单独列为"未匹配"。
5. 输出检测结果，按误差从大到小排序，并打印通过率汇总。

YAML 参数：
    source_dirs: list[str]   数据源目录（相对 output_base）
    anno_file  : str         人工标注 CSV 绝对路径或相对 output_base 路径
    tolerance  : float       年化收益率误差容忍度，默认 1e-5
    report_path: str         检测结果输出 CSV（相对 output_base），可选
"""

import csv
from pathlib import Path

import pandas as pd
from loguru import logger

from data_pipline.core.registry import register
from data_pipline.processors.base import BaseProcessor

DATE_FORMAT = "%Y-%m-%d"


def _parse_price(value: str) -> float:
    value = (value or "").strip().replace(",", "")
    if not value:
        raise ValueError("价格为空")
    return float(value)


def _parse_annualized(value: str) -> float:
    value = (value or "").strip().rstrip("%")
    if not value:
        raise ValueError("年化为空")
    return float(value) / 100.0


def _calc_annualized(start_date: pd.Timestamp, start_price: float, end_date: pd.Timestamp, end_price: float) -> float:
    days = (end_date - start_date).days
    if days <= 0:
        raise ValueError("当前日必须晚于首期日")
    if start_price <= 0 or end_price <= 0:
        raise ValueError("价格必须大于0")
    return (end_price / start_price) ** (365 / days) - 1


@register("SanityChecker")
class SanityChecker(BaseProcessor):

    def __init__(self, cfg: dict, output_base: Path):
        super().__init__(cfg, output_base)
        self.source_dirs: list[Path] = [
            (output_base / d).resolve() for d in cfg["source_dirs"]
        ]
        anno_file = Path(cfg["anno_file"]).expanduser()
        self.anno_file: Path = anno_file if anno_file.is_absolute() else (output_base / anno_file).resolve()
        self.tolerance: float = float(cfg.get("tolerance", 1e-5))
        report_path = cfg.get("report_path")
        self.report_path: Path | None = (
            (output_base / report_path).resolve() if report_path else None
        )

    # ──────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────

    def _build_symbol_index(self) -> dict[str, Path]:
        """返回 {symbol: csv_path}。多个 source_dirs 中重复的 symbol，后者覆盖前者。"""
        index: dict[str, Path] = {}
        for src_dir in self.source_dirs:
            if not src_dir.exists():
                logger.warning(f"source_dir does not exist, skipping: {src_dir}")
                continue
            for csv_path in sorted(src_dir.glob("*.csv")):
                index[csv_path.stem] = csv_path
        return index

    def _load_anno_rows(self) -> list[dict]:
        with self.anno_file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if any((v or "").strip() for v in row.values())]
        return rows

    def _load_close_series(self, csv_path: Path) -> pd.Series:
        df = pd.read_csv(csv_path, usecols=["date", "close"])
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["close"]

    def _check_row(self, row: dict, close_series: pd.Series) -> dict:
        fund_name = row.get("基金名") or row.get("代码") or "未知"
        code = (row.get("代码") or "").strip()

        try:
            start_date = pd.to_datetime(row["首期日"].strip(), format=DATE_FORMAT)
            end_date = pd.to_datetime(row["当前日"].strip(), format=DATE_FORMAT)
            anno_annualized = _parse_annualized(row.get("年化"))
        except Exception as e:
            return {
                "基金名": fund_name, "代码": code, "首期日": row.get("首期日"), "当前日": row.get("当前日"),
                "status": "FAIL", "reason": f"anno行解析失败: {e}", "abs_error": float("inf"),
            }

        if start_date not in close_series.index:
            return {
                "基金名": fund_name, "代码": code,
                "首期日": row["首期日"], "当前日": row["当前日"],
                "status": "FAIL", "reason": f"source无{row['首期日']}交易日数据", "abs_error": float("inf"),
            }
        if end_date not in close_series.index:
            return {
                "基金名": fund_name, "代码": code,
                "首期日": row["首期日"], "当前日": row["当前日"],
                "status": "FAIL", "reason": f"source无{row['当前日']}交易日数据", "abs_error": float("inf"),
            }

        src_start_price = float(close_series.loc[start_date])
        src_end_price = float(close_series.loc[end_date])

        try:
            src_annualized = _calc_annualized(start_date, src_start_price, end_date, src_end_price)
        except Exception as e:
            return {
                "基金名": fund_name, "代码": code,
                "首期日": row["首期日"], "当前日": row["当前日"],
                "status": "FAIL", "reason": f"年化计算失败: {e}", "abs_error": float("inf"),
            }

        abs_error = abs(src_annualized - anno_annualized)
        return {
            "基金名": fund_name, "代码": code,
            "首期日": row["首期日"], "当前日": row["当前日"],
            "anno_annualized": anno_annualized, "src_annualized": src_annualized,
            "abs_error": abs_error,
            "status": "PASS" if abs_error <= self.tolerance else "FAIL",
            "reason": "" if abs_error <= self.tolerance else "误差超出容忍范围",
        }

    # ──────────────────────────────────────────────────────────────
    # Entry point
    # ──────────────────────────────────────────────────────────────

    def __call__(self) -> None:
        logger.info("SanityChecker start")
        logger.info(f"  source_dirs: {[str(d) for d in self.source_dirs]}")
        logger.info(f"  anno_file  : {self.anno_file}")
        logger.info(f"  tolerance  : {self.tolerance:g}")

        symbol_index = self._build_symbol_index()
        if not symbol_index:
            logger.warning("No CSV files found in source_dirs — nothing to check.")
            return

        anno_rows = self._load_anno_rows()
        logger.info(f"Loaded {len(anno_rows)} anno rows")

        results: list[dict] = []
        unmatched: list[dict] = []
        close_cache: dict[str, pd.Series] = {}

        for row in anno_rows:
            code = (row.get("代码") or "").strip()
            fund_name = row.get("基金名") or code or "未知"

            if not code:
                unmatched.append({"基金名": fund_name, "代码": "", "reason": "anno行无代码"})
                continue
            if code not in symbol_index:
                unmatched.append({"基金名": fund_name, "代码": code, "reason": "source_dirs中未找到该代码"})
                continue

            if code not in close_cache:
                try:
                    close_cache[code] = self._load_close_series(symbol_index[code])
                except Exception as e:
                    unmatched.append({"基金名": fund_name, "代码": code, "reason": f"读取source CSV失败: {e}"})
                    continue

            results.append(self._check_row(row, close_cache[code]))

        # 按误差从大到小排序（FAIL 且无法对比的 inf 排最前）
        results.sort(key=lambda r: r["abs_error"], reverse=True)

        passed = sum(1 for r in results if r["status"] == "PASS")
        failed = len(results) - passed

        logger.info(
            f"SanityChecker done — checked: {len(results)} (PASS {passed} / FAIL {failed}), "
            f"unmatched: {len(unmatched)}"
        )
        logger.info(f"匹配上的检测结果 ({len(results)} 条，按误差降序):")
        for r in results:
            if r["abs_error"] == float("inf"):
                logger.info(f"  [FAIL] {r['基金名']}({r['代码']}) {r['首期日']}~{r['当前日']}: {r['reason']}")
            else:
                logger.info(
                    f"  [{r['status']}] {r['基金名']}({r['代码']}) {r['首期日']}~{r['当前日']}: "
                    f"anno={r['anno_annualized']*100:.4f}% src={r['src_annualized']*100:.4f}% err={r['abs_error']:.6f}"
                )

        if unmatched:
            logger.warning(f"未匹配记录 ({len(unmatched)} 条):")
            for u in unmatched:
                logger.warning(f"  {u['基金名']}({u['代码']}): {u['reason']}")

        if self.report_path:
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            report_df = pd.DataFrame(results)
            report_df.to_csv(self.report_path, index=False)
            logger.info(f"检测结果CSV已输出: {self.report_path}")

            if unmatched:
                unmatched_path = self.report_path.with_name(self.report_path.stem + "_unmatched.csv")
                pd.DataFrame(unmatched).to_csv(unmatched_path, index=False)
                logger.info(f"未匹配记录CSV已输出: {unmatched_path}")
        else:
            logger.info("未配置 report_path，检测结果未写出CSV文件")
