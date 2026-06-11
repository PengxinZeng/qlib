#!/usr/bin/env python3
"""Recalculate annualized returns in eft_data.csv."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


DEFAULT_CSV = Path(__file__).with_name("eft_data.csv")
DATE_FORMAT = "%Y-%m-%d"
ANNUALIZED_COLUMN = "年化"


def parse_price(value: str) -> float:
    value = value.strip().replace(",", "")
    if not value:
        raise ValueError("价格为空")
    return float(value)


def parse_date(value: str) -> datetime:
    return datetime.strptime(value.strip(), DATE_FORMAT)


def calculate_annualized(start_date: str, start_price: str, current_date: str, current_price: str) -> float:
    start = parse_date(start_date)
    current = parse_date(current_date)
    days = (current - start).days
    if days <= 0:
        raise ValueError("当前日必须晚于首期日")

    first_price = parse_price(start_price)
    latest_price = parse_price(current_price)
    if first_price <= 0 or latest_price <= 0:
        raise ValueError("价格必须大于0")

    return (latest_price / first_price) ** (365 / days) - 1


def format_percent(value: float) -> str:
    return f"{value:.2%}"


def is_empty_row(row: dict[str, str | None]) -> bool:
    return not any((value or "").strip() for value in row.values())


def recalculate_csv(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("CSV缺少表头")

        fieldnames = reader.fieldnames
        required_columns = ["首期日", "首期价格", "当前日", "当前价格", ANNUALIZED_COLUMN]
        missing_columns = [column for column in required_columns if column not in fieldnames]
        if missing_columns:
            raise ValueError(f"CSV缺少必要列: {', '.join(missing_columns)}")

        rows = []
        for line_number, row in enumerate(reader, start=2):
            if is_empty_row(row):
                continue
            if not (row.get("基金名") or row.get("代码")):
                continue

            try:
                annualized = calculate_annualized(
                    row["首期日"] or "",
                    row["首期价格"] or "",
                    row["当前日"] or "",
                    row["当前价格"] or "",
                )
            except Exception as error:
                fund_name = row.get("基金名") or row.get("代码") or f"第{line_number}行"
                raise ValueError(f"{fund_name} 年化计算失败: {error}") from error

            row[ANNUALIZED_COLUMN] = format_percent(annualized)
            rows.append({field: row.get(field, "") or "" for field in fieldnames})

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="根据CSV中的日期和价格重新计算ETF年化收益率")
    parser.add_argument("csv_path", nargs="?", type=Path, default=DEFAULT_CSV, help="CSV文件路径，默认使用脚本同目录下的eft_data.csv")
    args = parser.parse_args()

    updated_count = recalculate_csv(args.csv_path)
    print(f"已更新 {updated_count} 行年化收益率: {args.csv_path}")


if __name__ == "__main__":
    main()
