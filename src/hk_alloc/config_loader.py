from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class ValuationConfig:
    history_years: int = 3
    roll_window: int = 252
    sell_quantile: float = 0.95
    extreme_quantile: float = 0.99


@dataclass(frozen=True)
class PortfolioConfig:
    name: str
    currency: str
    total_capital: float
    scenario_capitals: tuple[float, ...] = ()
    scenario_top_ns: tuple[int, ...] = ()
    positions_csv: str | None = None
    positions_side: str | None = "long"
    market: str = "hk"
    require_stock_connect: bool = True
    allocation_method: str = "equal"
    secondary_fill_enabled: bool = True
    secondary_fill_avoid_high_valuation: bool = True
    secondary_fill_avoid_high_valuation_strict: bool = False
    secondary_fill_max_steps: int = 5000
    secondary_fill_allow_over_alloc: bool = False
    secondary_fill_max_over_alloc_ratio: float = 0.0
    secondary_fill_max_over_alloc_amount: float = 0.0
    secondary_fill_max_over_alloc_lots_per_ticker: int = 1
    secondary_fill_cash_buffer_ratio: float = 0.0
    secondary_fill_cash_buffer_amount: float = 0.0
    secondary_fill_estimated_fee_per_order: float = 0.0
    valuation: ValuationConfig = ValuationConfig()


@dataclass(frozen=True)
class TickerConfig:
    ticker: str
    name: str | None = None
    weight: float | None = None
    rank: int | None = None
    signal: float | None = None
    source: str = "yaml"
    enabled: bool = True


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        raise ValueError(f"Config file is empty: {path}")
    if not isinstance(payload, dict):
        raise ValueError("Top-level YAML must be a mapping")
    return payload


def _parse_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "是"}:
        return True
    if text in {"false", "0", "no", "n", "否"}:
        return False
    return default


def _resolve_path(cfg_path: Path, raw_path: str) -> Path:
    resolved = Path(raw_path).expanduser()
    if not resolved.is_absolute():
        resolved = (cfg_path.parent / resolved).resolve()
    return resolved


def _load_tickers_from_positions_csv(
    csv_path: Path,
    side_filter: str | None,
) -> list[TickerConfig]:
    if not csv_path.exists():
        raise FileNotFoundError(f"positions csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"positions csv is empty: {csv_path}")
    if "ts_code" not in df.columns:
        raise ValueError(f"positions csv missing required column: ts_code ({csv_path})")

    working = df.copy()
    working["ts_code"] = working["ts_code"].astype(str).str.strip()
    working = working[working["ts_code"] != ""]

    normalized_side = str(side_filter).strip().lower() if side_filter is not None else ""
    if normalized_side:
        if "side" not in working.columns:
            raise ValueError(f"positions csv missing side column for side filter '{normalized_side}': {csv_path}")
        side_values = working["side"].astype(str).str.strip().str.lower()
        working = working[side_values == normalized_side]

    if working.empty:
        raise ValueError(f"No ticker rows found in positions csv after filtering: {csv_path}")

    if "rank" in working.columns:
        working["_rank_sort"] = pd.to_numeric(working["rank"], errors="coerce")
        working = working.sort_values(["_rank_sort", "ts_code"], ascending=[True, True], na_position="last")
    elif "signal" in working.columns:
        working["_signal_sort"] = pd.to_numeric(working["signal"], errors="coerce")
        working = working.sort_values(["_signal_sort", "ts_code"], ascending=[False, True], na_position="last")

    working["_ticker_upper"] = working["ts_code"].str.upper()
    working = working.drop_duplicates(subset=["_ticker_upper"], keep="first")

    out: list[TickerConfig] = []
    for _, row in working.iterrows():
        weight_raw = pd.to_numeric(pd.Series([row.get("weight")]), errors="coerce").iloc[0]
        rank_raw = pd.to_numeric(pd.Series([row.get("rank")]), errors="coerce").iloc[0]
        signal_raw = pd.to_numeric(pd.Series([row.get("signal")]), errors="coerce").iloc[0]

        rank = int(rank_raw) if pd.notna(rank_raw) else None
        signal = float(signal_raw) if pd.notna(signal_raw) else None
        weight = float(weight_raw) if pd.notna(weight_raw) else None

        out.append(
            TickerConfig(
                ticker=str(row["ts_code"]).strip(),
                name=str(row["name"]).strip() if "name" in working.columns and pd.notna(row.get("name")) else None,
                weight=weight,
                rank=rank,
                signal=signal,
                source="positions_csv",
                enabled=_parse_bool(row.get("enabled"), default=True),
            )
        )

    return out


def _parse_positive_float_list(raw: Any, key: str) -> tuple[float, ...]:
    if raw is None:
        return tuple()
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"{key} must be a non-empty list")
    values: list[float] = []
    for idx, item in enumerate(raw):
        value = float(item)
        if value <= 0:
            raise ValueError(f"{key}[{idx}] must be > 0")
        values.append(value)
    deduped = list(dict.fromkeys(values))
    return tuple(deduped)


def _parse_positive_int_list(raw: Any, key: str) -> tuple[int, ...]:
    if raw is None:
        return tuple()
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"{key} must be a non-empty list")
    values: list[int] = []
    for idx, item in enumerate(raw):
        value = int(item)
        if value <= 0:
            raise ValueError(f"{key}[{idx}] must be > 0")
        values.append(value)
    deduped = list(dict.fromkeys(values))
    return tuple(deduped)


def load_portfolio_yaml(path: str | Path) -> tuple[PortfolioConfig, list[TickerConfig]]:
    cfg_path = Path(path)
    data = _read_yaml(cfg_path)

    portfolio_data = data.get("portfolio")
    tickers_data = data.get("tickers")
    universe_data = data.get("universe", {})
    scenarios_data = data.get("scenarios", {})
    if not isinstance(portfolio_data, dict):
        raise ValueError("YAML must contain top-level key: portfolio (mapping)")
    if universe_data is None:
        universe_data = {}
    if scenarios_data is None:
        scenarios_data = {}
    if not isinstance(universe_data, dict):
        raise ValueError("YAML key universe must be a mapping")
    if not isinstance(scenarios_data, dict):
        raise ValueError("YAML key scenarios must be a mapping")

    valuation_data = portfolio_data.get("valuation", {})
    trading_data = portfolio_data.get("trading", {})
    allocation_data = portfolio_data.get("allocation", {})
    secondary_fill_data = allocation_data.get("secondary_fill", {})
    positions_csv_raw = universe_data.get("positions_csv")
    positions_side = universe_data.get("positions_side", universe_data.get("side", "long"))
    positions_csv_path: Path | None = None
    if positions_csv_raw is not None:
        if not isinstance(positions_csv_raw, str) or not positions_csv_raw.strip():
            raise ValueError("universe.positions_csv must be a non-empty string path")
        positions_csv_path = _resolve_path(cfg_path, positions_csv_raw.strip())

    valuation = ValuationConfig(
        history_years=int(valuation_data.get("history_years", 3)),
        roll_window=int(valuation_data.get("roll_window", 252)),
        sell_quantile=float(valuation_data.get("sell_quantile", 0.95)),
        extreme_quantile=float(valuation_data.get("extreme_quantile", 0.99)),
    )

    config = PortfolioConfig(
        name=str(portfolio_data.get("name", "hk_portfolio")),
        currency=str(portfolio_data.get("currency", "HKD")),
        total_capital=float(portfolio_data["total_capital"]),
        positions_csv=str(positions_csv_path) if positions_csv_path is not None else None,
        positions_side=str(positions_side).strip() if positions_side is not None else None,
        market=str(trading_data.get("market", "hk")),
        require_stock_connect=bool(trading_data.get("require_stock_connect", True)),
        allocation_method=str(allocation_data.get("method", "equal")).lower(),
        secondary_fill_enabled=bool(secondary_fill_data.get("enabled", True)),
        secondary_fill_avoid_high_valuation=bool(secondary_fill_data.get("avoid_high_valuation", True)),
        secondary_fill_avoid_high_valuation_strict=bool(
            secondary_fill_data.get("avoid_high_valuation_strict", False)
        ),
        secondary_fill_max_steps=int(secondary_fill_data.get("max_steps", 5000)),
        secondary_fill_allow_over_alloc=bool(secondary_fill_data.get("allow_over_alloc", False)),
        secondary_fill_max_over_alloc_ratio=float(secondary_fill_data.get("max_over_alloc_ratio", 0.0)),
        secondary_fill_max_over_alloc_amount=float(secondary_fill_data.get("max_over_alloc_amount", 0.0)),
        secondary_fill_max_over_alloc_lots_per_ticker=int(
            secondary_fill_data.get("max_over_alloc_lots_per_ticker", 1)
        ),
        secondary_fill_cash_buffer_ratio=float(secondary_fill_data.get("cash_buffer_ratio", 0.0)),
        secondary_fill_cash_buffer_amount=float(secondary_fill_data.get("cash_buffer_amount", 0.0)),
        secondary_fill_estimated_fee_per_order=float(
            secondary_fill_data.get("estimated_fee_per_order", 0.0)
        ),
        valuation=valuation,
    )

    if config.total_capital <= 0:
        raise ValueError("portfolio.total_capital must be > 0")
    if config.allocation_method not in {"equal", "custom"}:
        raise ValueError("portfolio.allocation.method must be one of: equal, custom")
    if config.secondary_fill_max_steps <= 0:
        raise ValueError("portfolio.allocation.secondary_fill.max_steps must be > 0")
    if config.secondary_fill_max_over_alloc_ratio < 0:
        raise ValueError("portfolio.allocation.secondary_fill.max_over_alloc_ratio must be >= 0")
    if config.secondary_fill_max_over_alloc_amount < 0:
        raise ValueError("portfolio.allocation.secondary_fill.max_over_alloc_amount must be >= 0")
    if config.secondary_fill_max_over_alloc_lots_per_ticker < 0:
        raise ValueError("portfolio.allocation.secondary_fill.max_over_alloc_lots_per_ticker must be >= 0")
    if config.secondary_fill_cash_buffer_ratio < 0:
        raise ValueError("portfolio.allocation.secondary_fill.cash_buffer_ratio must be >= 0")
    if config.secondary_fill_cash_buffer_amount < 0:
        raise ValueError("portfolio.allocation.secondary_fill.cash_buffer_amount must be >= 0")
    if config.secondary_fill_estimated_fee_per_order < 0:
        raise ValueError("portfolio.allocation.secondary_fill.estimated_fee_per_order must be >= 0")
    if config.secondary_fill_allow_over_alloc and config.secondary_fill_max_over_alloc_lots_per_ticker == 0:
        raise ValueError(
            "portfolio.allocation.secondary_fill.max_over_alloc_lots_per_ticker must be > 0 when allow_over_alloc=true"
        )
    if not (0.0 < valuation.sell_quantile < 1.0):
        raise ValueError("portfolio.valuation.sell_quantile must be in (0, 1)")
    if not (0.0 < valuation.extreme_quantile < 1.0):
        raise ValueError("portfolio.valuation.extreme_quantile must be in (0, 1)")
    if valuation.sell_quantile >= valuation.extreme_quantile:
        raise ValueError("sell_quantile must be less than extreme_quantile")
    if valuation.roll_window <= 1:
        raise ValueError("portfolio.valuation.roll_window must be > 1")
    if valuation.history_years <= 0:
        raise ValueError("portfolio.valuation.history_years must be > 0")

    tickers: list[TickerConfig]
    if positions_csv_path is not None:
        tickers = _load_tickers_from_positions_csv(positions_csv_path, side_filter=config.positions_side)
    else:
        if not isinstance(tickers_data, list):
            raise ValueError("YAML must contain either universe.positions_csv or top-level tickers (list)")
        tickers = []
        for idx, item in enumerate(tickers_data):
            if isinstance(item, str):
                ticker_cfg = TickerConfig(ticker=item.strip())
            elif isinstance(item, dict):
                ticker_value = item.get("ticker")
                if not isinstance(ticker_value, str) or not ticker_value.strip():
                    raise ValueError(f"tickers[{idx}] missing non-empty ticker")
                rank_raw = item.get("rank")
                signal_raw = item.get("signal")
                ticker_cfg = TickerConfig(
                    ticker=ticker_value.strip(),
                    name=item.get("name"),
                    weight=float(item["weight"]) if "weight" in item and item["weight"] is not None else None,
                    rank=int(rank_raw) if rank_raw is not None else None,
                    signal=float(signal_raw) if signal_raw is not None else None,
                    source="yaml",
                    enabled=_parse_bool(item.get("enabled"), default=True),
                )
            else:
                raise ValueError(f"tickers[{idx}] must be a string or mapping")
            tickers.append(ticker_cfg)

    active_tickers = [item for item in tickers if item.enabled]
    if not active_tickers:
        raise ValueError("No enabled tickers found")

    dedup_check: set[str] = set()
    for item in active_tickers:
        key = item.ticker.upper()
        if key in dedup_check:
            raise ValueError(f"Duplicate ticker detected: {item.ticker}")
        dedup_check.add(key)

    if config.allocation_method == "custom":
        weights = [item.weight for item in active_tickers]
        if any(weight is None for weight in weights):
            raise ValueError("custom allocation requires weight for each enabled ticker")
        if sum(weight for weight in weights if weight is not None) <= 0:
            raise ValueError("custom allocation weight sum must be > 0")

    raw_capitals = scenarios_data.get("capitals", [config.total_capital])
    raw_top_ns = scenarios_data.get("top_ns", [len(active_tickers)])
    scenario_capitals = _parse_positive_float_list(raw_capitals, "scenarios.capitals")
    scenario_top_ns = _parse_positive_int_list(raw_top_ns, "scenarios.top_ns")
    max_top_n = max(scenario_top_ns)
    if max_top_n > len(active_tickers):
        raise ValueError(
            f"scenarios.top_ns contains {max_top_n}, but only {len(active_tickers)} enabled tickers are available"
        )

    config = replace(
        config,
        scenario_capitals=scenario_capitals,
        scenario_top_ns=scenario_top_ns,
    )

    return config, tickers


def get_active_tickers(tickers: list[TickerConfig]) -> list[TickerConfig]:
    return [item for item in tickers if item.enabled]
