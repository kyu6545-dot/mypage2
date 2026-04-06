#!/usr/bin/env python
"""Build GitHub-hosted price data artifacts from an editable universe CSV.

This script uses OpenBB's yfinance provider fetchers directly so it can keep
working even when the higher-level ``obb`` router package build is flaky.
The output is designed to be consumed by the Apps Script data-sync layer:

- data/adj_close_matrix.csv
- data/universe.csv
- data/manifest.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openbb_yfinance.models.crypto_historical import YFinanceCryptoHistoricalFetcher
from openbb_yfinance.models.equity_historical import YFinanceEquityHistoricalFetcher
from openbb_yfinance.models.index_historical import YFinanceIndexHistoricalFetcher


TRUTHY = {"1", "true", "t", "yes", "y"}
FALSY = {"0", "false", "f", "no", "n", ""}


@dataclass
class UniverseRow:
    active: bool
    ticker: str
    provider_symbol: str
    display_name: str
    asset_type: str
    benchmark: bool
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OpenBB price matrix for Google Sheets sync.")
    parser.add_argument(
        "--universe",
        default="config/universe.csv",
        help="Input universe CSV path.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory for output CSV/JSON artifacts.",
    )
    parser.add_argument(
        "--start-date",
        default="2020-01-02",
        help="Historical start date in YYYY-MM-DD.",
    )
    parser.add_argument(
        "--end-date",
        default=date.today().isoformat(),
        help="Historical end date in YYYY-MM-DD.",
    )
    return parser.parse_args()


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in TRUTHY:
        return True
    if text in FALSY:
        return False
    return default


def read_universe(path: Path) -> list[UniverseRow]:
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")
    rows: list[UniverseRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            ticker = str(raw.get("ticker", "")).strip().upper()
            if not ticker:
                continue
            rows.append(
                UniverseRow(
                    active=parse_bool(raw.get("active"), True),
                    ticker=ticker,
                    provider_symbol=str(raw.get("provider_symbol") or ticker).strip(),
                    display_name=str(raw.get("display_name") or "").strip(),
                    asset_type=str(raw.get("asset_type") or "").strip().lower(),
                    benchmark=parse_bool(raw.get("benchmark"), ticker == "QQQ"),
                    notes=str(raw.get("notes") or "").strip(),
                )
            )
    active_rows = [row for row in rows if row.active]
    if not active_rows:
        raise ValueError("Universe file has no active tickers.")
    return active_rows


def choose_fetcher(asset_type: str):
    normalized = (asset_type or "").strip().lower()
    if normalized in {"crypto", "coin"}:
        return YFinanceCryptoHistoricalFetcher
    if normalized in {"index"}:
        return YFinanceIndexHistoricalFetcher
    return YFinanceEquityHistoricalFetcher


async def fetch_symbol_history(row: UniverseRow, start_date: str, end_date: str) -> dict[str, Any]:
    fetcher = choose_fetcher(row.asset_type)
    result = await fetcher.fetch_data(
        {
            "symbol": row.provider_symbol,
            "start_date": start_date,
            "end_date": end_date,
            "interval": "1d",
            "adjustment": "splits_and_dividends",
        },
        credentials=None,
    )
    if not result:
        raise ValueError(f"No history returned for {row.ticker}")

    frame = pd.DataFrame([item.model_dump() for item in result])
    if frame.empty or "date" not in frame.columns or "close" not in frame.columns:
        raise ValueError(f"Unexpected history payload for {row.ticker}")

    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame = frame[["date", "close"]].dropna(subset=["close"]).sort_values("date")
    if frame.empty:
        raise ValueError(f"No close prices returned for {row.ticker}")

    series = frame.set_index("date")["close"]
    series.name = row.ticker
    return {
        "row": row,
        "series": series,
        "rows_fetched": int(series.shape[0]),
        "first_date": series.index.min().isoformat(),
        "last_price_date": series.index.max().isoformat(),
        "status": "ok",
        "error": "",
    }


async def build_dataset(rows: list[UniverseRow], start_date: str, end_date: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    tasks = [fetch_symbol_history(row, start_date, end_date) for row in rows]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    series_list: list[pd.Series] = []
    audit_rows: list[dict[str, Any]] = []

    for row, result in zip(rows, results):
        if isinstance(result, Exception):
            audit_rows.append(
                {
                    "active": row.active,
                    "ticker": row.ticker,
                    "provider_symbol": row.provider_symbol,
                    "display_name": row.display_name,
                    "asset_type": row.asset_type,
                    "benchmark": row.benchmark,
                    "import_status": "error",
                    "rows_fetched": 0,
                    "first_date": "",
                    "last_price_date": "",
                    "notes": row.notes,
                    "error": str(result),
                }
            )
            continue

        series_list.append(result["series"])
        audit_rows.append(
            {
                "active": row.active,
                "ticker": row.ticker,
                "provider_symbol": row.provider_symbol,
                "display_name": row.display_name,
                "asset_type": row.asset_type,
                "benchmark": row.benchmark,
                "import_status": "ok",
                "rows_fetched": result["rows_fetched"],
                "first_date": result["first_date"],
                "last_price_date": result["last_price_date"],
                "notes": row.notes,
                "error": "",
            }
        )

    if not series_list:
        raise RuntimeError("No symbols were fetched successfully.")

    matrix = pd.concat(series_list, axis=1, join="outer").sort_index()
    matrix.index = pd.to_datetime(matrix.index)
    matrix.index.name = "date"
    matrix = matrix.sort_index()
    return matrix, audit_rows


def write_outputs(output_dir: Path, matrix: pd.DataFrame, audit_rows: list[dict[str, Any]], start_date: str, end_date: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_to_write = matrix.copy()
    matrix_to_write.index = matrix_to_write.index.strftime("%Y-%m-%d")
    matrix_path = output_dir / "adj_close_matrix.csv"
    matrix_to_write.to_csv(matrix_path, float_format="%.8f")

    audit_frame = pd.DataFrame(audit_rows)
    audit_frame = audit_frame.sort_values(["benchmark", "ticker"], ascending=[False, True])
    universe_path = output_dir / "universe.csv"
    audit_frame.to_csv(universe_path, index=False, encoding="utf-8-sig")

    ok_rows = [row for row in audit_rows if row["import_status"] == "ok"]
    error_rows = [row for row in audit_rows if row["import_status"] != "ok"]
    manifest = {
        "provider": "openbb_yfinance_fetcher",
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "start_date": start_date,
        "end_date": end_date,
        "field": "adjusted_close",
        "matrix_rows": int(matrix.shape[0]),
        "ticker_count": int(len(audit_rows)),
        "success_count": int(len(ok_rows)),
        "error_count": int(len(error_rows)),
        "success_tickers": [row["ticker"] for row in ok_rows],
        "failed_tickers": [row["ticker"] for row in error_rows],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    universe_path = Path(args.universe)
    output_dir = Path(args.output_dir)
    rows = read_universe(universe_path)
    matrix, audit_rows = asyncio.run(build_dataset(rows, args.start_date, args.end_date))
    write_outputs(output_dir, matrix, audit_rows, args.start_date, args.end_date)
    success_count = sum(1 for row in audit_rows if row["import_status"] == "ok")
    print(
        f"Done: {success_count}/{len(audit_rows)} tickers fetched, "
        f"{matrix.shape[0]} rows written to {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
