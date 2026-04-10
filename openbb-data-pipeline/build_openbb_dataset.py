#!/usr/bin/env python
"""Build GitHub-hosted market data artifacts for Google Sheets sync.

Outputs:
- data/adj_close_matrix.csv
- data/universe.csv
- data/forward_signals.csv
- data/market_regime.csv
- data/manifest.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from openbb_yfinance.models.crypto_historical import YFinanceCryptoHistoricalFetcher
    from openbb_yfinance.models.equity_historical import YFinanceEquityHistoricalFetcher
    from openbb_yfinance.models.index_historical import YFinanceIndexHistoricalFetcher
    from openbb_yfinance.models.key_metrics import YFinanceKeyMetricsFetcher
    from openbb_yfinance.models.price_target_consensus import (
        YFinancePriceTargetConsensusFetcher,
    )

    OPENBB_YFINANCE_AVAILABLE = True
except ImportError:
    YFinanceCryptoHistoricalFetcher = None
    YFinanceEquityHistoricalFetcher = None
    YFinanceIndexHistoricalFetcher = None
    YFinanceKeyMetricsFetcher = None
    YFinancePriceTargetConsensusFetcher = None
    OPENBB_YFINANCE_AVAILABLE = False


TRUTHY = {"1", "true", "t", "yes", "y"}
FALSY = {"0", "false", "f", "no", "n", ""}
TRADING_DAYS = 252


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
    parser = argparse.ArgumentParser(description="Build market data artifacts for Google Sheets sync.")
    parser.add_argument("--universe", default="config/universe.csv", help="Input universe CSV path.")
    parser.add_argument("--output-dir", default="data", help="Directory for output CSV/JSON artifacts.")
    parser.add_argument("--start-date", default="2020-01-02", help="Historical start date in YYYY-MM-DD.")
    parser.add_argument("--end-date", default=date.today().isoformat(), help="Historical end date in YYYY-MM-DD.")
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


def build_macro_rows() -> list[UniverseRow]:
    return [
        UniverseRow(True, "VIX", "^VIX", "CBOE Volatility Index", "index", False, "macro"),
        UniverseRow(True, "TNX", "^TNX", "US 10Y Treasury Yield", "index", False, "macro"),
        UniverseRow(True, "IRX", "^IRX", "US 13 Week T-Bill Yield", "index", False, "macro"),
        UniverseRow(True, "HYG", "HYG", "iShares iBoxx High Yield Bond ETF", "etf", False, "macro"),
        UniverseRow(True, "LQD", "LQD", "iShares iBoxx Investment Grade Corporate Bond ETF", "etf", False, "macro"),
        UniverseRow(True, "QQQ", "QQQ", "Invesco QQQ Trust", "etf", True, "macro"),
    ]


def choose_fetcher(asset_type: str):
    normalized = (asset_type or "").strip().lower()
    if not OPENBB_YFINANCE_AVAILABLE:
        return None
    if normalized in {"crypto", "coin"}:
        return YFinanceCryptoHistoricalFetcher
    if normalized in {"index"}:
        return YFinanceIndexHistoricalFetcher
    return YFinanceEquityHistoricalFetcher


def _yfinance_history(provider_symbol: str, start_date: str, end_date: str) -> pd.Series:
    import yfinance as yf

    history = yf.Ticker(provider_symbol).history(
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True,
    )
    if history.empty or "Close" not in history.columns:
        raise ValueError(f"No history returned for {provider_symbol}")
    series = history["Close"].dropna()
    if series.empty:
        raise ValueError(f"No close prices returned for {provider_symbol}")
    series.index = pd.to_datetime(series.index).date
    return series


async def fetch_symbol_history(row: UniverseRow, start_date: str, end_date: str) -> dict[str, Any]:
    fetcher = choose_fetcher(row.asset_type)
    if fetcher is not None:
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
    else:
        series = await asyncio.to_thread(_yfinance_history, row.provider_symbol, start_date, end_date)

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


def coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def compute_momentum_12_1(series: pd.Series, lookback_days: int = 252, skip_days: int = 21) -> float | None:
    values = series.dropna()
    if len(values) <= lookback_days or len(values) <= skip_days + 1:
        return None
    end_pos = len(values) - 1 - skip_days
    start_pos = len(values) - lookback_days
    if end_pos <= start_pos or start_pos < 0:
        return None
    start_val = coerce_float(values.iloc[start_pos])
    end_val = coerce_float(values.iloc[end_pos])
    if start_val is None or end_val is None or start_val <= 0:
        return None
    return end_val / start_val - 1


def compute_drawdown(series: pd.Series, lookback_days: int = 252) -> float | None:
    values = series.dropna().tail(lookback_days)
    if values.empty:
        return None
    peak = coerce_float(values.max())
    last_val = coerce_float(values.iloc[-1])
    if peak is None or peak <= 0 or last_val is None:
        return None
    return last_val / peak - 1


def compute_annualized_vol(series: pd.Series, window_days: int) -> float | None:
    returns = series.dropna().pct_change().dropna().tail(window_days)
    if len(returns) < 2:
        return None
    return coerce_float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))


def compute_beta(series: pd.Series, benchmark_series: pd.Series, window_days: int = 252) -> float | None:
    aligned = pd.concat([series, benchmark_series], axis=1, join="inner").dropna()
    if aligned.empty:
        return None
    returns = aligned.pct_change().dropna().tail(window_days)
    if len(returns) < 20:
        return None
    asset_returns = returns.iloc[:, 0]
    bench_returns = returns.iloc[:, 1]
    bench_var = coerce_float(bench_returns.var(ddof=1))
    if bench_var is None or bench_var == 0:
        return None
    covariance = coerce_float(asset_returns.cov(bench_returns))
    if covariance is None:
        return None
    return covariance / bench_var


def latest_rolling_z(series: pd.Series, window: int = 252, absolute: bool = False) -> float | None:
    values = series.dropna()
    if absolute:
        values = values.abs()
    values = values.tail(window)
    if len(values) < 20:
        return None
    mean = coerce_float(values.mean())
    std = coerce_float(values.std(ddof=1))
    last_val = coerce_float(values.iloc[-1])
    if mean is None or std is None or std == 0 or last_val is None:
        return None
    return (last_val - mean) / std


def softmax_probabilities(raw_scores: dict[str, float]) -> dict[str, float]:
    max_score = max(raw_scores.values())
    exps = {key: math.exp(value - max_score) for key, value in raw_scores.items()}
    total = sum(exps.values()) or 1.0
    return {key: exps[key] / total for key in raw_scores}


async def fetch_key_metrics_snapshot(row: UniverseRow) -> dict[str, Any]:
    result = {"forward_pe": None, "beta": None, "error": ""}
    try:
        if YFinanceKeyMetricsFetcher is not None:
            try:
                data = await YFinanceKeyMetricsFetcher.fetch_data({"symbol": row.provider_symbol}, credentials=None)
                if data:
                    payload = data[0].model_dump()
                    result["forward_pe"] = coerce_float(payload.get("forward_pe"))
                    result["beta"] = coerce_float(payload.get("beta"))
                    return result
            except Exception:
                pass
        import yfinance as yf

        info = await asyncio.to_thread(lambda: yf.Ticker(row.provider_symbol).get_info())
        result["forward_pe"] = coerce_float(info.get("forwardPE"))
        result["beta"] = coerce_float(info.get("beta"))
        return result
    except Exception as err:  # noqa: BLE001
        result["error"] = f"key_metrics:{err}"
        return result


async def fetch_price_target_snapshot(row: UniverseRow) -> dict[str, Any]:
    result = {"target_consensus": None, "current_price": None, "price_target_gap": None, "error": ""}
    try:
        if YFinancePriceTargetConsensusFetcher is not None:
            try:
                data = await YFinancePriceTargetConsensusFetcher.fetch_data({"symbol": row.provider_symbol}, credentials=None)
                if data:
                    payload = data[0].model_dump()
                    target_consensus = coerce_float(payload.get("target_consensus"))
                    current_price = coerce_float(payload.get("current_price"))
                    result["target_consensus"] = target_consensus
                    result["current_price"] = current_price
                    result["price_target_gap"] = (
                        target_consensus / current_price - 1
                        if target_consensus is not None and current_price not in (None, 0)
                        else None
                    )
                    return result
            except Exception:
                pass
        import yfinance as yf

        info = await asyncio.to_thread(lambda: yf.Ticker(row.provider_symbol).get_info())
        target_consensus = coerce_float(info.get("targetMeanPrice"))
        current_price = coerce_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        result["target_consensus"] = target_consensus
        result["current_price"] = current_price
        result["price_target_gap"] = (
            target_consensus / current_price - 1
            if target_consensus is not None and current_price not in (None, 0)
            else None
        )
        return result
    except Exception as err:  # noqa: BLE001
        result["error"] = f"price_target:{err}"
        return result


async def build_snapshot_maps(rows: list[UniverseRow]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    metric_tasks = [fetch_key_metrics_snapshot(row) for row in rows]
    target_tasks = [fetch_price_target_snapshot(row) for row in rows]
    metric_results = await asyncio.gather(*metric_tasks)
    target_results = await asyncio.gather(*target_tasks)
    metric_map = {row.ticker: payload for row, payload in zip(rows, metric_results)}
    target_map = {row.ticker: payload for row, payload in zip(rows, target_results)}
    return metric_map, target_map


def build_forward_signal_frame(
    rows: list[UniverseRow],
    matrix: pd.DataFrame,
    audit_rows: list[dict[str, Any]],
    metric_map: dict[str, dict[str, Any]],
    target_map: dict[str, dict[str, Any]],
    snapshot_date: str,
    benchmark_series: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    success_tickers = {row["ticker"] for row in audit_rows if row["import_status"] == "ok"}
    records: list[dict[str, Any]] = []
    failed_tickers: list[str] = []
    error_count = 0

    for row in rows:
        if row.ticker not in success_tickers or row.ticker not in matrix.columns:
            failed_tickers.append(row.ticker)
            error_count += 1
            continue

        series = matrix[row.ticker].dropna()
        metric_payload = metric_map.get(row.ticker, {})
        target_payload = target_map.get(row.ticker, {})
        price_target_gap = coerce_float(target_payload.get("price_target_gap"))
        beta_252 = compute_beta(series, benchmark_series, 252)
        forward_pe = coerce_float(metric_payload.get("forward_pe"))

        if metric_payload.get("error"):
            error_count += 1
            failed_tickers.append(row.ticker)
        elif target_payload.get("error"):
            error_count += 1
            failed_tickers.append(row.ticker)

        records.append(
            {
                "date": snapshot_date,
                "ticker": row.ticker,
                "forward_pe": forward_pe,
                "price_target_gap": price_target_gap,
                "beta_252": beta_252 if beta_252 is not None else coerce_float(metric_payload.get("beta")),
                "momentum_12_1": compute_momentum_12_1(series),
                "drawdown_252": compute_drawdown(series, 252),
                "vol_20": compute_annualized_vol(series, 20),
                "vol_60": compute_annualized_vol(series, 60),
            }
        )
        records[-1]["vol_ratio_20_60"] = safe_div(records[-1]["vol_20"], records[-1]["vol_60"])

    if not records:
        raise RuntimeError("forward_signals.csv could not be generated because no valid ticker snapshots were available.")

    frame = pd.DataFrame(records)
    return frame, {
        "forward_signal_error_count": int(error_count),
        "forward_signal_failed_tickers": sorted(set(failed_tickers)),
    }


def build_market_regime_frame(
    macro_matrix: pd.DataFrame, qqq_series: pd.Series, snapshot_date: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {"VIX", "TNX", "IRX", "HYG", "LQD"}
    available = {column for column in macro_matrix.columns if column in required}
    missing = sorted(required - available)
    if missing:
        raise RuntimeError(f"market_regime.csv requires macro series: missing {', '.join(missing)}")

    vix = macro_matrix["VIX"].dropna()
    tnx = macro_matrix["TNX"].dropna()
    irx = macro_matrix["IRX"].dropna()
    hyg = macro_matrix["HYG"].dropna()
    lqd = macro_matrix["LQD"].dropna()
    qqq = qqq_series.dropna()

    hyg_lqd_ratio_series = (hyg / lqd).dropna()
    hyg_lqd_momentum = hyg_lqd_ratio_series / hyg_lqd_ratio_series.shift(20) - 1
    term_spread_series = (tnx - irx).dropna()
    qqq_momentum = compute_momentum_12_1(qqq)
    qqq_drawdown = compute_drawdown(qqq, 252)
    qqq_vol_20 = compute_annualized_vol(qqq, 20)
    qqq_vol_60 = compute_annualized_vol(qqq, 60)
    qqq_vol_ratio = safe_div(qqq_vol_20, qqq_vol_60)

    components = {
        "vix_z": latest_rolling_z(vix, 252),
        "vol_ratio_z": latest_rolling_z((qqq.pct_change().rolling(20).std(ddof=1) / qqq.pct_change().rolling(60).std(ddof=1)).dropna(), 252),
        "drawdown_abs_z": latest_rolling_z((qqq / qqq.rolling(252).max() - 1).dropna(), 252, absolute=True),
        "term_spread_z": latest_rolling_z(term_spread_series, 252),
        "hyg_lqd_mom_z": latest_rolling_z(hyg_lqd_momentum, 252),
        "qqq_momentum_z": latest_rolling_z((qqq / qqq.shift(252) - 1).dropna(), 252),
    }
    missing_inputs = sorted([name for name, value in components.items() if value is None])
    available_inputs = len(components) - len(missing_inputs)
    if available_inputs < 4:
        raise RuntimeError("market_regime.csv could not be generated because too many macro regime inputs were missing.")

    def v(name: str) -> float:
        return components[name] if components[name] is not None else 0.0

    risk_off_raw = (
        v("vix_z")
        + v("vol_ratio_z")
        + v("drawdown_abs_z")
        - v("term_spread_z")
        - v("hyg_lqd_mom_z")
        - v("qqq_momentum_z")
    )
    risk_on_raw = (
        -v("vix_z")
        - v("vol_ratio_z")
        - v("drawdown_abs_z")
        + v("term_spread_z")
        + v("hyg_lqd_mom_z")
        + v("qqq_momentum_z")
    )
    neutral_raw = -abs(risk_on_raw - risk_off_raw)
    probabilities = softmax_probabilities(
        {"risk_on": risk_on_raw, "neutral": neutral_raw, "risk_off": risk_off_raw}
    )
    completeness = available_inputs / len(components)
    confidence = max(probabilities.values()) * completeness
    regime = max(probabilities, key=probabilities.get)

    row = {
        "date": snapshot_date,
        "vix_close": coerce_float(vix.iloc[-1]),
        "tnx_close": coerce_float(tnx.iloc[-1]),
        "irx_close": coerce_float(irx.iloc[-1]),
        "term_spread_10y_3m": coerce_float(term_spread_series.iloc[-1]) if not term_spread_series.empty else None,
        "hyg_lqd_ratio": coerce_float(hyg_lqd_ratio_series.iloc[-1]) if not hyg_lqd_ratio_series.empty else None,
        "qqq_momentum_12_1": qqq_momentum,
        "qqq_drawdown_252": qqq_drawdown,
        "qqq_vol_ratio_20_60": qqq_vol_ratio,
        "risk_on_prob": probabilities["risk_on"],
        "neutral_prob": probabilities["neutral"],
        "risk_off_prob": probabilities["risk_off"],
        "forecast_confidence": confidence,
        "market_regime": regime,
    }
    frame = pd.DataFrame([row])
    return frame, {
        "market_regime_error_count": int(len(missing_inputs)),
        "market_regime_failed_inputs": missing_inputs,
    }


def upsert_snapshot_frame(existing_path: Path, new_frame: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    if existing_path.exists():
        existing_frame = pd.read_csv(existing_path, encoding="utf-8-sig")
        combined = pd.concat([existing_frame, new_frame], ignore_index=True)
    else:
        combined = new_frame.copy()
    combined = combined.drop_duplicates(subset=key_columns, keep="last")
    combined = combined.sort_values(key_columns).reset_index(drop=True)
    return combined


def write_outputs(
    output_dir: Path,
    matrix: pd.DataFrame,
    audit_rows: list[dict[str, Any]],
    forward_signals_frame: pd.DataFrame,
    market_regime_frame: pd.DataFrame,
    forward_signal_meta: dict[str, Any],
    market_regime_meta: dict[str, Any],
    start_date: str,
    end_date: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_to_write = matrix.copy()
    matrix_to_write.index = matrix_to_write.index.strftime("%Y-%m-%d")
    matrix_path = output_dir / "adj_close_matrix.csv"
    matrix_to_write.to_csv(matrix_path, float_format="%.8f", encoding="utf-8")

    audit_frame = pd.DataFrame(audit_rows).sort_values(["benchmark", "ticker"], ascending=[False, True])
    universe_path = output_dir / "universe.csv"
    audit_frame.to_csv(universe_path, index=False, encoding="utf-8-sig")

    forward_signals_path = output_dir / "forward_signals.csv"
    forward_signals_full = upsert_snapshot_frame(forward_signals_path, forward_signals_frame, ["date", "ticker"])
    forward_signals_full.to_csv(forward_signals_path, index=False, encoding="utf-8", float_format="%.8f")

    market_regime_path = output_dir / "market_regime.csv"
    market_regime_full = upsert_snapshot_frame(market_regime_path, market_regime_frame, ["date"])
    market_regime_full.to_csv(market_regime_path, index=False, encoding="utf-8", float_format="%.8f")

    ok_rows = [row for row in audit_rows if row["import_status"] == "ok"]
    error_rows = [row for row in audit_rows if row["import_status"] != "ok"]
    manifest = {
        "provider": "openbb_yfinance_fetcher_with_snapshot_signals",
        "generated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "start_date": start_date,
        "end_date": end_date,
        "field": "adjusted_close",
        "matrix_rows": int(matrix.shape[0]),
        "first_price_date": matrix.index.min().date().isoformat() if not matrix.empty else "",
        "last_price_date": matrix.index.max().date().isoformat() if not matrix.empty else "",
        "ticker_count": int(len(audit_rows)),
        "success_count": int(len(ok_rows)),
        "error_count": int(len(error_rows)),
        "success_tickers": [row["ticker"] for row in ok_rows],
        "failed_tickers": [row["ticker"] for row in error_rows],
        "forward_signal_rows": int(forward_signals_full.shape[0]),
        "market_regime_rows": int(market_regime_full.shape[0]),
        "forward_signal_error_count": int(forward_signal_meta.get("forward_signal_error_count", 0)),
        "market_regime_error_count": int(market_regime_meta.get("market_regime_error_count", 0)),
        "forward_signal_failed_tickers": forward_signal_meta.get("forward_signal_failed_tickers", []),
        "market_regime_failed_inputs": market_regime_meta.get("market_regime_failed_inputs", []),
        "forward_signal_snapshot_date": str(forward_signals_frame.iloc[-1]["date"]),
        "market_regime_snapshot_date": str(market_regime_frame.iloc[-1]["date"]),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


async def build_all_artifacts(
    rows: list[UniverseRow], start_date: str, end_date: str
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    matrix, audit_rows = await build_dataset(rows, start_date, end_date)
    macro_matrix, _macro_audit = await build_dataset(build_macro_rows(), start_date, end_date)
    snapshot_date = matrix.index.max().date().isoformat()
    regime_snapshot_date = macro_matrix.index.max().date().isoformat()
    metric_map, target_map = await build_snapshot_maps(rows)
    if "QQQ" in matrix.columns:
        benchmark_series = matrix["QQQ"]
    elif "QQQ" in macro_matrix.columns:
        benchmark_series = macro_matrix["QQQ"]
    else:
        raise RuntimeError("QQQ benchmark series is required for forward signals and market regime calculations.")
    forward_signals_frame, forward_signal_meta = build_forward_signal_frame(
        rows, matrix, audit_rows, metric_map, target_map, snapshot_date, benchmark_series
    )
    market_regime_frame, market_regime_meta = build_market_regime_frame(
        macro_matrix, benchmark_series, regime_snapshot_date
    )
    return (
        matrix,
        audit_rows,
        forward_signals_frame,
        market_regime_frame,
        forward_signal_meta,
        market_regime_meta,
    )


def main() -> None:
    args = parse_args()
    universe_path = Path(args.universe)
    output_dir = Path(args.output_dir)
    rows = read_universe(universe_path)
    (
        matrix,
        audit_rows,
        forward_signals_frame,
        market_regime_frame,
        forward_signal_meta,
        market_regime_meta,
    ) = asyncio.run(build_all_artifacts(rows, args.start_date, args.end_date))
    write_outputs(
        output_dir,
        matrix,
        audit_rows,
        forward_signals_frame,
        market_regime_frame,
        forward_signal_meta,
        market_regime_meta,
        args.start_date,
        args.end_date,
    )
    success_count = sum(1 for row in audit_rows if row["import_status"] == "ok")
    print(
        f"Done: {success_count}/{len(audit_rows)} tickers fetched, "
        f"{matrix.shape[0]} price rows, "
        f"{forward_signals_frame.shape[0]} forward signal rows, "
        f"{market_regime_frame.shape[0]} regime rows written to {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
