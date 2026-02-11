from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    market: str = "hk"
    require_stock_connect: bool = True
    allocation_method: str = "equal"
    secondary_fill_enabled: bool = True
    secondary_fill_avoid_high_valuation: bool = True
    secondary_fill_max_steps: int = 100000
    valuation: ValuationConfig = ValuationConfig()


@dataclass(frozen=True)
class TickerConfig:
    ticker: str
    name: str | None = None
    weight: float | None = None
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


def load_portfolio_yaml(path: str | Path) -> tuple[PortfolioConfig, list[TickerConfig]]:
    cfg_path = Path(path)
    data = _read_yaml(cfg_path)

    portfolio_data = data.get("portfolio")
    tickers_data = data.get("tickers")
    if not isinstance(portfolio_data, dict) or not isinstance(tickers_data, list):
        raise ValueError("YAML must contain top-level keys: portfolio (mapping), tickers (list)")

    valuation_data = portfolio_data.get("valuation", {})
    trading_data = portfolio_data.get("trading", {})
    allocation_data = portfolio_data.get("allocation", {})
    secondary_fill_data = allocation_data.get("secondary_fill", {})

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
        market=str(trading_data.get("market", "hk")),
        require_stock_connect=bool(trading_data.get("require_stock_connect", True)),
        allocation_method=str(allocation_data.get("method", "equal")).lower(),
        secondary_fill_enabled=bool(secondary_fill_data.get("enabled", True)),
        secondary_fill_avoid_high_valuation=bool(secondary_fill_data.get("avoid_high_valuation", True)),
        secondary_fill_max_steps=int(secondary_fill_data.get("max_steps", 100000)),
        valuation=valuation,
    )

    if config.total_capital <= 0:
        raise ValueError("portfolio.total_capital must be > 0")
    if config.allocation_method not in {"equal", "custom"}:
        raise ValueError("portfolio.allocation.method must be one of: equal, custom")
    if config.secondary_fill_max_steps <= 0:
        raise ValueError("portfolio.allocation.secondary_fill.max_steps must be > 0")
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

    tickers: list[TickerConfig] = []
    for idx, item in enumerate(tickers_data):
        if isinstance(item, str):
            ticker_cfg = TickerConfig(ticker=item.strip())
        elif isinstance(item, dict):
            ticker_value = item.get("ticker")
            if not isinstance(ticker_value, str) or not ticker_value.strip():
                raise ValueError(f"tickers[{idx}] missing non-empty ticker")
            ticker_cfg = TickerConfig(
                ticker=ticker_value.strip(),
                name=item.get("name"),
                weight=float(item["weight"]) if "weight" in item and item["weight"] is not None else None,
                enabled=bool(item.get("enabled", True)),
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

    return config, tickers


def get_active_tickers(tickers: list[TickerConfig]) -> list[TickerConfig]:
    return [item for item in tickers if item.enabled]
