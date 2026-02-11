from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .config_loader import PortfolioConfig, TickerConfig, get_active_tickers


def ticker_to_rq_order_book_id(ticker: str) -> str:
    normalized = ticker.strip().upper()
    if normalized.endswith(".XHKG"):
        return normalized

    parts = normalized.split(".")
    if len(parts) != 2:
        raise ValueError(f"Invalid ticker format: {ticker}")

    code, market = parts
    if market != "HK":
        raise ValueError(f"Unsupported ticker market: {ticker}")
    return f"{code.zfill(5)}.XHKG"


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (float, np.floating)) and math.isnan(float(value)):
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "nan"}
    return False


def _pick_last_non_missing(values: Sequence[Any]) -> Any:
    for value in reversed(list(values)):
        if not _is_missing_value(value):
            return value
    return None


def _pick_round_lot(values: Sequence[Any]) -> float:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.max())


def _coerce_scalar(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        if value.empty:
            return None
        value = value.iloc[-1]
    if isinstance(value, pd.Series):
        return _pick_last_non_missing(value.tolist())
    return value


def _resolve_display_name(config_name: str | None, instrument_symbol: Any) -> str | None:
    if isinstance(config_name, str) and config_name.strip():
        return config_name.strip()
    symbol = _coerce_scalar(instrument_symbol)
    if _is_missing_value(symbol):
        return None
    text = str(symbol).strip()
    return text or None


def _get_instrument_field(instruments_df: pd.DataFrame, oid: str, field: str, default: Any) -> Any:
    if oid not in instruments_df.index or field not in instruments_df.columns:
        return default
    value = _coerce_scalar(instruments_df.loc[oid, field])
    if _is_missing_value(value):
        return default
    return value


def classify_valuation(
    percentile: float,
    zscore: float,
    sell_quantile: float = 0.95,
    extreme_quantile: float = 0.99,
) -> str:
    if math.isnan(percentile) and math.isnan(zscore):
        return "NA"
    if (
        not math.isnan(percentile)
        and percentile >= extreme_quantile
    ) or (
        not math.isnan(zscore)
        and zscore >= 2.5
    ):
        return "EXTREME"
    if (
        not math.isnan(percentile)
        and percentile >= sell_quantile
    ) or (
        not math.isnan(zscore)
        and zscore >= 2.0
    ):
        return "HIGH"
    if (
        not math.isnan(percentile)
        and percentile <= (1 - sell_quantile)
    ) or (
        not math.isnan(zscore)
        and zscore <= -2.0
    ):
        return "LOW"
    return "NEUTRAL"


def calc_lots(
    target_value: float,
    price: float,
    round_lot: float,
    tradable: bool,
) -> int:
    if not tradable:
        return 0
    if any(math.isnan(x) for x in [target_value, price, round_lot]):
        return 0
    if target_value <= 0 or price <= 0 or round_lot <= 0:
        return 0
    return int(math.floor(target_value / (price * round_lot)))


def _import_rqdatac() -> Any:
    try:
        import rqdatac  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "rqdatac is not installed. Run 'uv sync' first."
        ) from exc
    return rqdatac


def init_rqdata() -> Any:
    rqdatac = _import_rqdatac()
    rqdatac.init()
    return rqdatac


def _normalize_close_output(px: pd.DataFrame | None, order_book_ids: Sequence[str]) -> pd.DataFrame:
    if px is None:
        return pd.DataFrame(columns=list(order_book_ids))

    if isinstance(px.index, pd.MultiIndex):
        close = px["close"].unstack(0)
    elif isinstance(px.columns, pd.MultiIndex):
        close = px["close"] if "close" in px.columns.get_level_values(0) else px.xs("close", axis=1, level=1)
    else:
        if list(px.columns) == ["close"]:
            close = px.rename(columns={"close": order_book_ids[0]})
        else:
            close = px.copy()

    if isinstance(close, pd.Series):
        close = close.to_frame(name=order_book_ids[0])

    close.index = pd.to_datetime(close.index)
    close = close.sort_index()

    for oid in order_book_ids:
        if oid not in close.columns:
            close[oid] = np.nan

    return close[list(order_book_ids)]


def fetch_instruments(rqdatac_module: Any, order_book_ids: Sequence[str], market: str) -> pd.DataFrame:
    instruments = rqdatac_module.instruments(list(order_book_ids), market=market)
    if not isinstance(instruments, list):
        instruments = [instruments]

    rows: list[dict[str, Any]] = []
    for ins in instruments:
        if ins is None:
            continue
        rows.append(
            {
                "order_book_id": ins.order_book_id,
                "symbol": getattr(ins, "symbol", None),
                "round_lot": safe_float(getattr(ins, "round_lot", np.nan)),
                "stock_connect": getattr(ins, "stock_connect", None),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["symbol", "round_lot", "stock_connect"])

    raw = pd.DataFrame(rows)
    grouped = (
        raw.groupby("order_book_id", sort=False, as_index=True)
        .agg(
            symbol=("symbol", lambda s: _pick_last_non_missing(s.tolist())),
            round_lot=("round_lot", _pick_round_lot),
            stock_connect=("stock_connect", lambda s: _pick_last_non_missing(s.tolist())),
        )
    )
    return grouped


def fetch_close_prices(
    rqdatac_module: Any,
    order_book_ids: Sequence[str],
    start_date: date,
    end_date: date,
    market: str,
    adjust_type: str,
) -> pd.DataFrame:
    px = rqdatac_module.get_price(
        list(order_book_ids),
        start_date=start_date,
        end_date=end_date,
        frequency="1d",
        fields=["close"],
        adjust_type=adjust_type,
        market=market,
        expect_df=True,
    )
    return _normalize_close_output(px, order_book_ids)


def _last_value_percentile(values: np.ndarray) -> float:
    if values.size == 0 or np.isnan(values[-1]):
        return np.nan
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return np.nan
    return float(np.mean(valid <= values[-1]))


def compute_valuation_metrics(
    close_pre: pd.DataFrame,
    window: int,
    sell_quantile: float,
    extreme_quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    q_high = close_pre.rolling(window=window, min_periods=window).quantile(sell_quantile)
    q_extreme = close_pre.rolling(window=window, min_periods=window).quantile(extreme_quantile)
    percentile = close_pre.rolling(window=window, min_periods=window).apply(_last_value_percentile, raw=True)

    log_price = np.log(close_pre.where(close_pre > 0))
    mean = log_price.rolling(window=window, min_periods=window).mean()
    std = log_price.rolling(window=window, min_periods=window).std()
    zscore = (log_price - mean) / std

    return percentile, zscore, q_high, q_extreme


def build_target_values(
    total_capital: float,
    tickers: Sequence[TickerConfig],
    allocation_method: str,
) -> dict[str, float]:
    active = get_active_tickers(list(tickers))
    if not active:
        raise ValueError("No enabled tickers")

    if allocation_method == "equal":
        value = total_capital / len(active)
        return {item.ticker: value for item in active}

    if allocation_method != "custom":
        raise ValueError(f"Unsupported allocation method: {allocation_method}")

    weight_map = {item.ticker: float(item.weight or 0) for item in active}
    total_weight = sum(weight_map.values())
    if total_weight <= 0:
        raise ValueError("Total custom weight must be > 0")

    return {
        ticker: (weight / total_weight) * total_capital
        for ticker, weight in weight_map.items()
    }


def _is_stock_connect_tradable(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in {"", "none", "nan", "false", "0", "no"}


def _to_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _get_previous_trading_date(
    rqdatac_module: Any,
    ref_date: date,
    n: int,
    market: str,
) -> date:
    try:
        result = rqdatac_module.get_previous_trading_date(ref_date, n=n, market=market)
    except TypeError:
        result = rqdatac_module.get_previous_trading_date(ref_date, n, market=market)
    return _to_date(result)


def apply_secondary_fill(
    allocation_df: pd.DataFrame,
    total_capital: float,
    enabled: bool,
    avoid_high_valuation: bool,
    max_steps: int,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    updated = allocation_df.copy()
    if "lots_extra" not in updated.columns:
        updated["lots_extra"] = 0

    if not enabled or updated.empty:
        return (
            updated,
            {
                "secondary_fill_enabled": bool(enabled),
                "secondary_fill_steps": 0,
                "secondary_fill_spent": 0.0,
                "cash_remaining_after_fill": max(total_capital - float(updated["est_value"].sum()), 0.0),
            },
        )

    eps = 1e-9
    valuation_rank = {"LOW": 0, "NEUTRAL": 1, "HIGH": 2, "EXTREME": 3, "NA": 4}
    disallowed_when_avoid = {"HIGH", "EXTREME"}

    def candidate_rows(cash_left: float) -> pd.DataFrame:
        candidates = updated[
            (updated["tradable"] == True)
            & (updated["lot_cost"] > 0)
            & (updated["lot_cost"] <= cash_left + eps)
        ].copy()
        if candidates.empty:
            return candidates
        if avoid_high_valuation:
            preferred = candidates[~candidates["valuation"].isin(disallowed_when_avoid)]
            if not preferred.empty:
                return preferred
        return candidates

    def ranking_key(row: pd.Series) -> tuple[float, float, float, str]:
        valuation = str(row.get("valuation", "NA"))
        rank = valuation_rank.get(valuation, 5)
        deviation_after_lot = abs(float(row["gap_to_target"]) - float(row["lot_cost"]))
        lot_cost = float(row["lot_cost"])
        ticker = str(row.get("ticker", ""))
        return (float(rank), deviation_after_lot, -lot_cost, ticker)

    cash_left = max(total_capital - float(updated["est_value"].sum()), 0.0)
    steps = 0
    spent = 0.0

    while cash_left > eps and steps < max_steps:
        candidates = candidate_rows(cash_left)
        if candidates.empty:
            break

        selected_idx = min(candidates.index, key=lambda idx: ranking_key(candidates.loc[idx]))
        row = updated.loc[selected_idx]
        lot_cost = float(row["lot_cost"])
        if lot_cost <= 0 or lot_cost > cash_left + eps:
            break

        updated.at[selected_idx, "lots"] = int(row["lots"]) + 1
        updated.at[selected_idx, "lots_extra"] = int(row.get("lots_extra", 0)) + 1
        updated.at[selected_idx, "shares"] = int(row["shares"]) + int(row["round_lot"])
        updated.at[selected_idx, "est_value"] = float(row["est_value"]) + lot_cost
        updated.at[selected_idx, "gap_to_target"] = float(row["gap_to_target"]) - lot_cost

        cash_left -= lot_cost
        spent += lot_cost
        steps += 1

    # Keep consistent with actual totals after updates.
    cash_left = max(total_capital - float(updated["est_value"].sum()), 0.0)
    return (
        updated,
        {
            "secondary_fill_enabled": bool(enabled),
            "secondary_fill_steps": int(steps),
            "secondary_fill_spent": float(spent),
            "cash_remaining_after_fill": float(cash_left),
        },
    )


def build_allocation_table(
    rqdatac_module: Any,
    portfolio: PortfolioConfig,
    ticker_configs: Sequence[TickerConfig],
    as_of: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = get_active_tickers(list(ticker_configs))
    tickers = [item.ticker for item in active]
    order_book_ids = [ticker_to_rq_order_book_id(ticker) for ticker in tickers]
    ticker_to_oid = dict(zip(tickers, order_book_ids))

    instruments_df = fetch_instruments(rqdatac_module, order_book_ids, market=portfolio.market)

    # Use the latest available close within a short lookback window.
    # On the current trading date, daily close may be unavailable before market close.
    price_start = _get_previous_trading_date(rqdatac_module, as_of, n=10, market=portfolio.market)
    close_today = fetch_close_prices(
        rqdatac_module,
        order_book_ids,
        start_date=price_start,
        end_date=as_of,
        market=portfolio.market,
        adjust_type="none",
    )

    hist_days = max(portfolio.valuation.history_years * 252, portfolio.valuation.roll_window + 5)
    hist_start = _get_previous_trading_date(rqdatac_module, as_of, n=hist_days, market=portfolio.market)

    close_pre = fetch_close_prices(
        rqdatac_module,
        order_book_ids,
        start_date=hist_start,
        end_date=as_of,
        market=portfolio.market,
        adjust_type="pre",
    )

    percentile, zscore, q_high, q_extreme = compute_valuation_metrics(
        close_pre,
        window=portfolio.valuation.roll_window,
        sell_quantile=portfolio.valuation.sell_quantile,
        extreme_quantile=portfolio.valuation.extreme_quantile,
    )

    if close_pre.empty:
        raise RuntimeError("No historical HK price data returned; check RQData permissions or ticker list.")
    latest_row = close_pre.index.max()
    target_values = build_target_values(portfolio.total_capital, active, portfolio.allocation_method)
    if close_today.empty:
        raise RuntimeError("No recent HK close price returned for allocation.")
    close_today_clean = close_today.dropna(how="all")
    if close_today_clean.empty:
        raise RuntimeError("Recent HK close prices are all missing for the selected tickers.")
    price_row = close_today_clean.iloc[-1]

    rows: list[dict[str, Any]] = []
    for item in active:
        ticker = item.ticker
        oid = ticker_to_oid[ticker]

        symbol = _get_instrument_field(instruments_df, oid, "symbol", None)
        round_lot = safe_float(_get_instrument_field(instruments_df, oid, "round_lot", np.nan))
        stock_connect = _get_instrument_field(instruments_df, oid, "stock_connect", None)
        price = safe_float(price_row.get(oid, np.nan))

        is_connect_tradable = _is_stock_connect_tradable(stock_connect)
        tradable = (not portfolio.require_stock_connect or is_connect_tradable) and price > 0 and round_lot > 0

        target_value = float(target_values[ticker])
        lots = calc_lots(target_value, price, round_lot, tradable)
        lot_cost = float(price * round_lot) if tradable else float("nan")
        shares = int(lots * round_lot) if round_lot > 0 and not math.isnan(round_lot) else 0
        est_value = float(shares * price) if price > 0 and not math.isnan(price) else 0.0
        gap_to_target = float(target_value - est_value)

        pct = safe_float(percentile.loc[latest_row, oid]) if oid in percentile.columns else np.nan
        z_1y = safe_float(zscore.loc[latest_row, oid]) if oid in zscore.columns else np.nan
        high_line = safe_float(q_high.loc[latest_row, oid]) if oid in q_high.columns else np.nan
        extreme_line = safe_float(q_extreme.loc[latest_row, oid]) if oid in q_extreme.columns else np.nan

        if math.isnan(high_line) or math.isnan(extreme_line):
            overpriced_range = None
        else:
            overpriced_range = f"[{high_line:.4f}, {extreme_line:.4f}]"

        rows.append(
            {
                "ticker": ticker,
                "order_book_id": oid,
                "name": _resolve_display_name(item.name, symbol),
                "price": price,
                "round_lot": round_lot,
                "stock_connect": stock_connect,
                "target_value": target_value,
                "lot_cost": lot_cost,
                "lots": lots,
                "lots_extra": 0,
                "shares": shares,
                "est_value": est_value,
                "gap_to_target": gap_to_target,
                "pct_1y": pct,
                "z_1y": z_1y,
                "valuation": classify_valuation(
                    pct,
                    z_1y,
                    sell_quantile=portfolio.valuation.sell_quantile,
                    extreme_quantile=portfolio.valuation.extreme_quantile,
                ),
                "overpriced_low": high_line,
                "overpriced_high": extreme_line,
                "overpriced_range": overpriced_range,
                "tradable": tradable,
            }
        )

    allocation_df = pd.DataFrame(rows)
    allocation_df, fill_stats = apply_secondary_fill(
        allocation_df,
        total_capital=portfolio.total_capital,
        enabled=portfolio.secondary_fill_enabled,
        avoid_high_valuation=portfolio.secondary_fill_avoid_high_valuation,
        max_steps=portfolio.secondary_fill_max_steps,
    )
    summary_df = pd.DataFrame(
        [
            {
                "as_of": as_of,
                "pricing_date": _to_date(close_today_clean.index[-1]),
                "portfolio_name": portfolio.name,
                "num_tickers": len(active),
                "total_capital": portfolio.total_capital,
                "total_est_value": float(allocation_df["est_value"].sum()),
                "total_gap": float(allocation_df["gap_to_target"].sum()),
                "cash_used_ratio": (
                    float(allocation_df["est_value"].sum()) / portfolio.total_capital
                    if portfolio.total_capital > 0
                    else np.nan
                ),
                "secondary_fill_enabled": fill_stats["secondary_fill_enabled"],
                "secondary_fill_steps": fill_stats["secondary_fill_steps"],
                "secondary_fill_spent": fill_stats["secondary_fill_spent"],
                "cash_remaining_after_fill": fill_stats["cash_remaining_after_fill"],
            }
        ]
    )

    return allocation_df, summary_df


def build_sell_signals(
    rqdatac_module: Any,
    portfolio: PortfolioConfig,
    ticker_configs: Sequence[TickerConfig],
    as_of: date,
) -> pd.DataFrame:
    active = get_active_tickers(list(ticker_configs))
    tickers = [item.ticker for item in active]
    order_book_ids = [ticker_to_rq_order_book_id(ticker) for ticker in tickers]
    ticker_to_oid = dict(zip(tickers, order_book_ids))

    hist_days = max(portfolio.valuation.history_years * 252, portfolio.valuation.roll_window + 5)
    hist_start = _get_previous_trading_date(rqdatac_module, as_of, n=hist_days, market=portfolio.market)

    close_pre = fetch_close_prices(
        rqdatac_module,
        order_book_ids,
        start_date=hist_start,
        end_date=as_of,
        market=portfolio.market,
        adjust_type="pre",
    )
    if close_pre.empty:
        raise RuntimeError("No historical HK price data returned; cannot build sell signals.")

    percentile, zscore, q_high, q_extreme = compute_valuation_metrics(
        close_pre,
        window=portfolio.valuation.roll_window,
        sell_quantile=portfolio.valuation.sell_quantile,
        extreme_quantile=portfolio.valuation.extreme_quantile,
    )

    signal = (close_pre >= q_high) & (close_pre.shift(1) < q_high.shift(1))
    latest_row = close_pre.index.max()

    rows: list[dict[str, Any]] = []
    for item in active:
        ticker = item.ticker
        oid = ticker_to_oid[ticker]
        col_signal = signal[oid].fillna(False)

        signal_dates = col_signal.index[col_signal]
        last_signal_date = signal_dates.max() if len(signal_dates) > 0 else pd.NaT

        current_price = safe_float(close_pre.loc[latest_row, oid]) if oid in close_pre.columns else np.nan
        pct = safe_float(percentile.loc[latest_row, oid]) if oid in percentile.columns else np.nan
        z_1y = safe_float(zscore.loc[latest_row, oid]) if oid in zscore.columns else np.nan
        high_line = safe_float(q_high.loc[latest_row, oid]) if oid in q_high.columns else np.nan
        extreme_line = safe_float(q_extreme.loc[latest_row, oid]) if oid in q_extreme.columns else np.nan

        rows.append(
            {
                "ticker": ticker,
                "order_book_id": oid,
                "as_of": _to_date(latest_row),
                "close_pre": current_price,
                "pct_1y": pct,
                "z_1y": z_1y,
                "sell_trigger": high_line,
                "extreme_trigger": extreme_line,
                "last_sell_signal_date": _to_date(last_signal_date) if pd.notna(last_signal_date) else None,
                "valuation": classify_valuation(
                    pct,
                    z_1y,
                    sell_quantile=portfolio.valuation.sell_quantile,
                    extreme_quantile=portfolio.valuation.extreme_quantile,
                ),
            }
        )

    return pd.DataFrame(rows)


def write_report(
    output_path: Path,
    allocation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    sell_signals_df: pd.DataFrame,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        allocation_df.to_excel(writer, sheet_name="allocation", index=False)
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        sell_signals_df.to_excel(writer, sheet_name="sell_signals", index=False)
    return output_path
