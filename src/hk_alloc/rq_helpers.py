from __future__ import annotations

import math
import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config_loader import PortfolioConfig, TickerConfig, get_active_tickers

LOGGER = logging.getLogger(__name__)

STOCK_CONNECT_TRUE_VALUES: set[str] = {
    "1",
    "true",
    "yes",
    "y",
    "是",
    "沪港通",
    "深港通",
    "southbound",
    "eligible",
    "sh",
    "sz",
    "沪",
    "深",
}

STOCK_CONNECT_FALSE_VALUES: set[str] = {
    "",
    "0",
    "false",
    "no",
    "n",
    "none",
    "nan",
    "null",
    "否",
    "不是",
    "不可",
    "不支持",
    "not eligible",
}


VALUATION_CN_MAP: dict[str, str] = {
    "LOW": "偏低",
    "NEUTRAL": "中性",
    "HIGH": "偏高",
    "EXTREME": "极高",
    "NA": "NA",
}

PRICE_SOURCE_CN_MAP: dict[str, str] = {
    "snapshot": "快照最新价",
    "1m_close": "1分钟收盘",
    "1d_close": "日线收盘",
    "mixed": "混合",
}

ALLOCATION_EXPORT_ORDER: list[str] = [
    "ticker",
    "name",
    "lots",
    "price",
    "valuation",
    "overpriced_high",
    "order_book_id",
    "tradable",
    "stock_connect",
    "price_source",
    "pricing_date",
    "round_lot",
    "lot_cost",
    "target_value",
    "lots_base",
    "lots_extra",
    "shares",
    "est_value",
    "gap_to_target",
    "gap_ratio",
    "pct_1y",
    "z_1y",
    "overpriced_low",
    "overpriced_range",
]

ALLOCATION_EXPORT_RENAME: dict[str, str] = {
    "name": "名称",
    "ticker": "股票代码",
    "order_book_id": "查询代码",
    "tradable": "可交易",
    "stock_connect": "港股通",
    "price_source": "价格来源",
    "pricing_date": "定价日期",
    "price": "当前价格",
    "round_lot": "每手股数",
    "lot_cost": "每手成本",
    "target_value": "目标金额",
    "lots_base": "初始手数",
    "lots_extra": "补仓手数",
    "lots": "合计手数",
    "shares": "股数",
    "est_value": "预计金额",
    "gap_to_target": "与目标差额",
    "gap_ratio": "偏离比例",
    "valuation": "估值分层",
    "pct_1y": "1年分位",
    "z_1y": "1年Z分",
    "overpriced_low": "统计高位下沿(未复权)",
    "overpriced_high": "统计高位上沿(未复权)",
    "overpriced_range": "统计高位区间(未复权)",
}

SUMMARY_EXPORT_RENAME: dict[str, str] = {
    "as_of": "统计日期",
    "pricing_date": "定价日期",
    "pricing_source": "价格来源",
    "pricing_source_detail": "价格来源明细",
    "portfolio_name": "组合名称",
    "num_tickers": "标的数量",
    "total_capital": "总资金",
    "total_est_value": "预计总金额",
    "total_gap": "总差额",
    "cash_used_ratio": "资金使用率",
    "secondary_fill_enabled": "启用二次补仓",
    "secondary_fill_steps": "补仓步数",
    "secondary_fill_spent": "补仓金额",
    "secondary_fill_fee_spent": "补仓估算费用",
    "secondary_fill_cash_buffer": "补仓现金缓冲",
    "secondary_fill_budget_after_buffer": "补仓可用资金",
    "cash_remaining_after_fill": "补仓后剩余现金",
}

SUMMARY_EXPORT_ORDER: list[str] = [
    "portfolio_name",
    "as_of",
    "pricing_date",
    "pricing_source",
    "pricing_source_detail",
    "num_tickers",
    "total_capital",
    "total_est_value",
    "total_gap",
    "cash_used_ratio",
    "secondary_fill_enabled",
    "secondary_fill_steps",
    "secondary_fill_spent",
    "secondary_fill_fee_spent",
    "secondary_fill_cash_buffer",
    "secondary_fill_budget_after_buffer",
    "cash_remaining_after_fill",
]

SELL_SIGNALS_EXPORT_RENAME: dict[str, str] = {
    "name": "名称",
    "ticker": "股票代码",
    "order_book_id": "查询代码",
    "as_of": "统计日期",
    "close_pre": "前复权收盘价",
    "pct_1y": "1年分位",
    "z_1y": "1年Z分",
    "sell_trigger": "偏高阈值",
    "extreme_trigger": "极高阈值",
    "last_sell_signal_date": "最近卖出信号日期",
    "valuation": "估值分层",
}

SELL_SIGNALS_EXPORT_ORDER: list[str] = [
    "ticker",
    "name",
    "close_pre",
    "valuation",
    "sell_trigger",
    "extreme_trigger",
    "last_sell_signal_date",
    "pct_1y",
    "z_1y",
    "order_book_id",
    "as_of",
]


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
    raw = pd.to_numeric(pd.Series(list(values)), errors="coerce")
    numeric = raw.dropna()
    if numeric.empty:
        return float("nan")
    unique_values = sorted({float(v) for v in numeric.tolist()})
    if len(unique_values) > 1:
        LOGGER.warning("Multiple round_lot values found %s; using mode then last non-missing.", unique_values)

    counts = numeric.value_counts()
    if counts.empty:
        return float("nan")

    top_count = int(counts.max())
    mode_values = [float(val) for val, count in counts.items() if int(count) == top_count]
    if len(mode_values) == 1:
        return mode_values[0]

    for value in reversed(raw.tolist()):
        if pd.isna(value):
            continue
        value_float = float(value)
        if value_float in mode_values:
            return value_float
    return mode_values[0]


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


def _get_attr_or_key(record: Any, key: str) -> Any:
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, key, None)


def _to_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.to_datetime(value, errors="coerce")
    if isinstance(ts, pd.Timestamp):
        return ts
    return pd.NaT


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


def _empty_live_price_frame(order_book_ids: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        index=list(order_book_ids),
        data={"price": np.nan, "pricing_ts": pd.NaT},
    )


def fetch_snapshot_prices(
    rqdatac_module: Any,
    order_book_ids: Sequence[str],
    market: str,
) -> pd.DataFrame:
    empty = _empty_live_price_frame(order_book_ids)

    try:
        snapshots = rqdatac_module.current_snapshot(list(order_book_ids), market=market)
    except Exception:
        return empty

    if snapshots is None:
        return empty
    if not isinstance(snapshots, list):
        snapshots = [snapshots]

    rows: list[dict[str, Any]] = []
    for snap in snapshots:
        if snap is None:
            continue

        oid = _get_attr_or_key(snap, "order_book_id")
        if _is_missing_value(oid):
            continue

        last = safe_float(_get_attr_or_key(snap, "last"))
        close = safe_float(_get_attr_or_key(snap, "close"))
        price = last if last > 0 else close if close > 0 else np.nan
        pricing_ts = _to_timestamp(_get_attr_or_key(snap, "datetime"))
        rows.append(
            {
                "order_book_id": str(oid),
                "price": price,
                "pricing_ts": pricing_ts,
            }
        )

    if not rows:
        return empty

    snap_df = pd.DataFrame(rows).drop_duplicates(subset=["order_book_id"], keep="last")
    snap_df = snap_df.set_index("order_book_id")[["price", "pricing_ts"]]

    empty.update(snap_df)
    return empty


def fetch_current_minute_prices(
    rqdatac_module: Any,
    order_book_ids: Sequence[str],
    market: str,
) -> pd.DataFrame:
    empty = _empty_live_price_frame(order_book_ids)

    minute_df: pd.DataFrame | None
    try:
        minute_df = rqdatac_module.current_minute(list(order_book_ids), fields=["close"], market=market)
    except TypeError:
        try:
            minute_df = rqdatac_module.current_minute(list(order_book_ids), market=market)
        except Exception:
            return empty
    except Exception:
        return empty

    if minute_df is None or minute_df.empty:
        return empty

    frame = minute_df.reset_index()
    if "order_book_id" not in frame.columns:
        return empty

    price_field = "close" if "close" in frame.columns else "last" if "last" in frame.columns else None
    if price_field is None:
        return empty
    if "datetime" not in frame.columns:
        frame["datetime"] = pd.NaT

    frame["price"] = pd.to_numeric(frame[price_field], errors="coerce")
    frame["pricing_ts"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame = frame.sort_values(["order_book_id", "pricing_ts"], na_position="last")
    frame = frame.groupby("order_book_id", as_index=False).tail(1)

    latest = frame.set_index("order_book_id")[["price", "pricing_ts"]]
    empty.update(latest)
    return empty


def _should_try_live_prices(as_of: date, market: str) -> bool:
    market_key = str(market).strip().lower()
    if market_key == "hk":
        today = datetime.now(ZoneInfo("Asia/Hong_Kong")).date()
        return as_of == today
    return as_of == datetime.now().date()


def build_latest_price_frame(
    rqdatac_module: Any,
    order_book_ids: Sequence[str],
    as_of: date,
    market: str,
) -> pd.DataFrame:
    price_start = _get_previous_trading_date(rqdatac_module, as_of, n=10, market=market)
    close_today = fetch_close_prices(
        rqdatac_module,
        order_book_ids,
        start_date=price_start,
        end_date=as_of,
        market=market,
        adjust_type="none",
    )
    if close_today.empty:
        raise RuntimeError("No recent HK close price returned for allocation.")

    close_today_clean = close_today.dropna(how="all")
    if close_today_clean.empty:
        raise RuntimeError("Recent HK close prices are all missing for the selected tickers.")

    latest_close_ts = pd.Timestamp(close_today_clean.index[-1])
    latest_close_row = close_today_clean.iloc[-1].reindex(list(order_book_ids))

    price_frame = pd.DataFrame(
        index=list(order_book_ids),
        data={
            "price": pd.to_numeric(latest_close_row, errors="coerce"),
            "price_source": "1d_close",
            "pricing_ts": latest_close_ts,
        },
    )

    if _should_try_live_prices(as_of=as_of, market=market):
        snapshot_frame = fetch_snapshot_prices(rqdatac_module, order_book_ids, market=market)
        minute_frame = fetch_current_minute_prices(rqdatac_module, order_book_ids, market=market)

        for oid in order_book_ids:
            snapshot_price = safe_float(snapshot_frame.at[oid, "price"])
            if snapshot_price > 0:
                price_frame.at[oid, "price"] = snapshot_price
                price_frame.at[oid, "price_source"] = "snapshot"
                snapshot_ts = _to_timestamp(snapshot_frame.at[oid, "pricing_ts"])
                if pd.notna(snapshot_ts):
                    price_frame.at[oid, "pricing_ts"] = snapshot_ts
                continue

            minute_price = safe_float(minute_frame.at[oid, "price"])
            if minute_price > 0:
                price_frame.at[oid, "price"] = minute_price
                price_frame.at[oid, "price_source"] = "1m_close"
                minute_ts = _to_timestamp(minute_frame.at[oid, "pricing_ts"])
                if pd.notna(minute_ts):
                    price_frame.at[oid, "pricing_ts"] = minute_ts

    price_frame["pricing_date"] = pd.to_datetime(price_frame["pricing_ts"], errors="coerce").dt.date
    fallback_date = _to_date(latest_close_ts)
    price_frame["pricing_date"] = price_frame["pricing_date"].fillna(fallback_date)
    return price_frame


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
    if _is_missing_value(value):
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_is_stock_connect_tradable(item) for item in value)

    text = re.sub(r"\s+", " ", str(value).strip().lower())
    if text in STOCK_CONNECT_TRUE_VALUES:
        return True
    if text in STOCK_CONNECT_FALSE_VALUES:
        return False

    tokens = [token for token in re.split(r"[,\s/|]+", text) if token]
    if tokens:
        if any(token in STOCK_CONNECT_TRUE_VALUES for token in tokens):
            return True
        if all(token in STOCK_CONNECT_FALSE_VALUES for token in tokens):
            return False
    return False


def _to_date(value: Any) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, datetime):
        return value.date()
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
    avoid_high_valuation_strict: bool,
    max_steps: int,
    allow_over_alloc: bool,
    max_over_alloc_ratio: float,
    max_over_alloc_amount: float,
    max_over_alloc_lots_per_ticker: int,
    cash_buffer_ratio: float,
    cash_buffer_amount: float,
    estimated_fee_per_order: float,
) -> tuple[pd.DataFrame, dict[str, float | int | bool]]:
    updated = allocation_df.copy()
    if "lots_extra" not in updated.columns:
        updated["lots_extra"] = 0
    updated["lots_extra"] = pd.to_numeric(updated["lots_extra"], errors="coerce").fillna(0).astype(int)

    def recompute_position_columns(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for idx, row in out.iterrows():
            lots_raw = safe_float(row.get("lots", 0))
            lots = max(int(lots_raw) if not math.isnan(lots_raw) else 0, 0)
            round_lot = safe_float(row.get("round_lot", np.nan))
            price = safe_float(row.get("price", np.nan))
            lot_cost_existing = safe_float(row.get("lot_cost", np.nan))
            if (math.isnan(price) or price <= 0) and round_lot > 0 and not math.isnan(round_lot):
                if lot_cost_existing > 0:
                    price = lot_cost_existing / round_lot
            target_value = safe_float(row.get("target_value", np.nan))
            tradable = bool(row.get("tradable", False))

            shares = int(round(lots * round_lot)) if round_lot > 0 and not math.isnan(round_lot) else 0
            est_value = float(shares * price) if price > 0 and not math.isnan(price) else 0.0
            lot_cost = float(price * round_lot) if tradable and price > 0 and round_lot > 0 else float("nan")
            if math.isnan(target_value):
                target_value = 0.0

            out.at[idx, "lots"] = lots
            out.at[idx, "shares"] = shares
            out.at[idx, "est_value"] = est_value
            out.at[idx, "lot_cost"] = lot_cost
            out.at[idx, "gap_to_target"] = float(target_value - est_value)
        return out

    buffer_amount = float(total_capital * max(cash_buffer_ratio, 0.0) + max(cash_buffer_amount, 0.0))
    available_budget = max(float(total_capital - buffer_amount), 0.0)

    if not enabled or updated.empty:
        updated = recompute_position_columns(updated)
        return (
            updated,
            {
                "secondary_fill_enabled": bool(enabled),
                "secondary_fill_steps": 0,
                "secondary_fill_spent": 0.0,
                "secondary_fill_fee_spent": 0.0,
                "secondary_fill_cash_buffer": float(buffer_amount),
                "secondary_fill_budget_after_buffer": float(available_budget),
                "cash_remaining_after_fill": max(total_capital - float(updated["est_value"].sum()), 0.0),
            },
        )

    eps = 1e-9
    over_alloc_caps: list[float] = []
    if allow_over_alloc and max_over_alloc_ratio > 0:
        over_alloc_caps.append(float(total_capital * max_over_alloc_ratio))
    if allow_over_alloc and max_over_alloc_amount > 0:
        over_alloc_caps.append(float(max_over_alloc_amount))
    max_over_alloc_value = min(over_alloc_caps) if over_alloc_caps else (float("inf") if allow_over_alloc else 0.0)

    valuation_rank = {"LOW": 0, "NEUTRAL": 1, "HIGH": 2, "EXTREME": 3, "NA": 4}
    disallowed_when_avoid = {"HIGH", "EXTREME"}
    over_alloc_count_by_idx: dict[Any, int] = {idx: 0 for idx in updated.index}

    def candidate_rows(cash_left: float) -> pd.DataFrame:
        candidates = updated[
            (updated["tradable"] == True)
            & (updated["lot_cost"] > 0)
            & (updated["gap_to_target"] > eps)
        ].copy()
        if candidates.empty:
            return candidates

        candidates["required_cash"] = pd.to_numeric(candidates["lot_cost"], errors="coerce").fillna(0.0) + max(
            estimated_fee_per_order, 0.0
        )
        candidates = candidates[candidates["required_cash"] <= cash_left + eps]
        if candidates.empty:
            return candidates

        candidates["new_gap"] = candidates["gap_to_target"] - candidates["lot_cost"]
        candidates["improves_gap"] = (candidates["new_gap"].abs() + eps) < candidates["gap_to_target"].abs()
        candidates = candidates[candidates["improves_gap"] == True]
        if candidates.empty:
            return candidates

        if not allow_over_alloc:
            candidates = candidates[candidates["new_gap"] >= -eps]
        else:
            candidates = candidates[candidates["new_gap"] >= (-max_over_alloc_value - eps)]
            if max_over_alloc_lots_per_ticker <= 0:
                candidates = candidates[candidates["new_gap"] >= -eps]
            else:
                over_limit_mask = []
                for idx, row in candidates.iterrows():
                    over_after = float(row["new_gap"]) < -eps
                    over_count = int(over_alloc_count_by_idx.get(idx, 0))
                    over_limit_mask.append(not (over_after and over_count >= max_over_alloc_lots_per_ticker))
                candidates = candidates.loc[over_limit_mask]
        if candidates.empty:
            return candidates

        if avoid_high_valuation:
            preferred = candidates[~candidates["valuation"].isin(disallowed_when_avoid)]
            if not preferred.empty:
                return preferred
            if avoid_high_valuation_strict:
                return preferred
        return candidates

    def ranking_key(row: pd.Series) -> tuple[float, float, float, str]:
        valuation = str(row.get("valuation", "NA"))
        rank = valuation_rank.get(valuation, 5)
        deviation_after_lot = abs(float(row["gap_to_target"]) - float(row["lot_cost"]))
        lot_cost = float(row["lot_cost"])
        ticker = str(row.get("ticker", ""))
        # Prefer smaller lot_cost for finer-grained cash usage under equal deviation.
        return (float(rank), deviation_after_lot, lot_cost, ticker)

    cash_left = max(available_budget - float(updated["est_value"].sum()), 0.0)
    tradable_costs = pd.to_numeric(updated.loc[updated["tradable"] == True, "lot_cost"], errors="coerce")
    tradable_costs = tradable_costs[tradable_costs > 0]
    if tradable_costs.empty:
        step_limit = 0
    else:
        min_required_cash = float(tradable_costs.min() + max(estimated_fee_per_order, 0.0))
        if min_required_cash <= 0:
            step_limit = max_steps
        else:
            theoretical_limit = int(math.floor(cash_left / min_required_cash)) + 1
            step_limit = min(max_steps, max(theoretical_limit, 0))

    steps = 0
    spent = 0.0
    fee_spent = 0.0

    while cash_left > eps and steps < step_limit:
        candidates = candidate_rows(cash_left)
        if candidates.empty:
            break

        selected_idx = min(candidates.index, key=lambda idx: ranking_key(candidates.loc[idx]))
        row = updated.loc[selected_idx]
        lot_cost = float(row["lot_cost"])
        required_cash = lot_cost + max(estimated_fee_per_order, 0.0)
        if lot_cost <= 0 or required_cash > cash_left + eps:
            break

        updated.at[selected_idx, "lots"] = int(row["lots"]) + 1
        updated.at[selected_idx, "lots_extra"] = int(row.get("lots_extra", 0)) + 1

        new_gap = float(row["gap_to_target"]) - lot_cost
        if new_gap < -eps:
            over_alloc_count_by_idx[selected_idx] = int(over_alloc_count_by_idx.get(selected_idx, 0)) + 1

        cash_left -= required_cash
        spent += lot_cost
        fee_spent += max(estimated_fee_per_order, 0.0)
        steps += 1

    updated = recompute_position_columns(updated)
    # Keep consistent with actual totals after updates (including estimated fees spent by fill).
    cash_left = max(total_capital - float(updated["est_value"].sum()) - float(fee_spent), 0.0)
    return (
        updated,
        {
            "secondary_fill_enabled": bool(enabled),
            "secondary_fill_steps": int(steps),
            "secondary_fill_spent": float(spent),
            "secondary_fill_fee_spent": float(fee_spent),
            "secondary_fill_cash_buffer": float(buffer_amount),
            "secondary_fill_budget_after_buffer": float(available_budget),
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
    latest_prices = build_latest_price_frame(
        rqdatac_module,
        order_book_ids=order_book_ids,
        as_of=as_of,
        market=portfolio.market,
    )

    hist_days = max(portfolio.valuation.history_years * 252, portfolio.valuation.roll_window + 5)
    hist_start = _get_previous_trading_date(rqdatac_module, as_of, n=hist_days, market=portfolio.market)

    close_hist = fetch_close_prices(
        rqdatac_module,
        order_book_ids,
        start_date=hist_start,
        end_date=as_of,
        market=portfolio.market,
        adjust_type="none",
    )

    percentile, zscore, q_high, q_extreme = compute_valuation_metrics(
        close_hist,
        window=portfolio.valuation.roll_window,
        sell_quantile=portfolio.valuation.sell_quantile,
        extreme_quantile=portfolio.valuation.extreme_quantile,
    )

    if close_hist.empty:
        raise RuntimeError("No historical HK price data returned; check RQData permissions or ticker list.")
    latest_row = close_hist.index.max()
    target_values = build_target_values(portfolio.total_capital, active, portfolio.allocation_method)

    rows: list[dict[str, Any]] = []
    for item in active:
        ticker = item.ticker
        oid = ticker_to_oid[ticker]

        symbol = _get_instrument_field(instruments_df, oid, "symbol", None)
        round_lot = safe_float(_get_instrument_field(instruments_df, oid, "round_lot", np.nan))
        stock_connect = _get_instrument_field(instruments_df, oid, "stock_connect", None)
        price = safe_float(latest_prices.at[oid, "price"])
        price_source = str(latest_prices.at[oid, "price_source"])
        pricing_date = _to_date(latest_prices.at[oid, "pricing_date"])

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
                "price_source": price_source,
                "pricing_date": pricing_date,
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
        avoid_high_valuation_strict=portfolio.secondary_fill_avoid_high_valuation_strict,
        max_steps=portfolio.secondary_fill_max_steps,
        allow_over_alloc=portfolio.secondary_fill_allow_over_alloc,
        max_over_alloc_ratio=portfolio.secondary_fill_max_over_alloc_ratio,
        max_over_alloc_amount=portfolio.secondary_fill_max_over_alloc_amount,
        max_over_alloc_lots_per_ticker=portfolio.secondary_fill_max_over_alloc_lots_per_ticker,
        cash_buffer_ratio=portfolio.secondary_fill_cash_buffer_ratio,
        cash_buffer_amount=portfolio.secondary_fill_cash_buffer_amount,
        estimated_fee_per_order=portfolio.secondary_fill_estimated_fee_per_order,
    )

    allocation_df["lots_base"] = allocation_df["lots"] - allocation_df["lots_extra"]
    allocation_df["gap_ratio"] = np.where(
        allocation_df["target_value"] > 0,
        allocation_df["gap_to_target"] / allocation_df["target_value"],
        np.nan,
    )

    pricing_dates = pd.to_datetime(latest_prices["pricing_date"], errors="coerce").dropna()
    summary_pricing_date = _to_date(pricing_dates.max()) if not pricing_dates.empty else as_of
    source_counts = latest_prices["price_source"].value_counts(dropna=False).to_dict()
    source_parts = [f"{str(source)}:{int(count)}" for source, count in source_counts.items()]
    summary_pricing_source = next(iter(source_counts.keys())) if len(source_counts) == 1 else "mixed"

    summary_df = pd.DataFrame(
        [
            {
                "as_of": as_of,
                "pricing_date": summary_pricing_date,
                "pricing_source": summary_pricing_source,
                "pricing_source_detail": ", ".join(source_parts),
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
                "secondary_fill_fee_spent": fill_stats["secondary_fill_fee_spent"],
                "secondary_fill_cash_buffer": fill_stats["secondary_fill_cash_buffer"],
                "secondary_fill_budget_after_buffer": fill_stats["secondary_fill_budget_after_buffer"],
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
    instruments_df = fetch_instruments(rqdatac_module, order_book_ids, market=portfolio.market)

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

    sell_trigger = q_high.shift(1)
    extreme_trigger = q_extreme.shift(1)
    signal = (close_pre >= sell_trigger) & (close_pre.shift(1) < sell_trigger)
    latest_row = close_pre.index.max()

    rows: list[dict[str, Any]] = []
    for item in active:
        ticker = item.ticker
        oid = ticker_to_oid[ticker]
        symbol = _get_instrument_field(instruments_df, oid, "symbol", None)
        col_signal = signal[oid].fillna(False)

        signal_dates = col_signal.index[col_signal]
        last_signal_date = signal_dates.max() if len(signal_dates) > 0 else pd.NaT

        current_price = safe_float(close_pre.loc[latest_row, oid]) if oid in close_pre.columns else np.nan
        pct = safe_float(percentile.loc[latest_row, oid]) if oid in percentile.columns else np.nan
        z_1y = safe_float(zscore.loc[latest_row, oid]) if oid in zscore.columns else np.nan
        high_line = safe_float(sell_trigger.loc[latest_row, oid]) if oid in sell_trigger.columns else np.nan
        extreme_line = (
            safe_float(extreme_trigger.loc[latest_row, oid]) if oid in extreme_trigger.columns else np.nan
        )

        rows.append(
            {
                "name": _resolve_display_name(item.name, symbol),
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


def _to_yes_no(value: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"
    if _is_missing_value(value):
        return "否"
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "是"}:
        return "是"
    if text in {"false", "0", "no", "n", "否"}:
        return "否"
    return str(value)


def _format_stock_connect(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        tokens = {str(item).strip().lower() for item in value if not _is_missing_value(item)}
        if "sh" in tokens and "sz" in tokens:
            return "沪/深"
        if "sh" in tokens:
            return "沪"
        if "sz" in tokens:
            return "深"
        return "是" if len(tokens) > 0 else "否"
    return "是" if _is_stock_connect_tradable(value) else "否"


def _localize_price_source(value: Any) -> str:
    text = str(value).strip() if not _is_missing_value(value) else ""
    return PRICE_SOURCE_CN_MAP.get(text, text or "未知")


def _prepare_allocation_export_df(allocation_df: pd.DataFrame) -> pd.DataFrame:
    out = allocation_df.copy()
    if "tradable" in out.columns:
        out["tradable"] = out["tradable"].map(_to_yes_no)
    if "stock_connect" in out.columns:
        out["stock_connect"] = out["stock_connect"].map(_format_stock_connect)
    if "valuation" in out.columns:
        out["valuation"] = out["valuation"].map(lambda x: VALUATION_CN_MAP.get(str(x), str(x)))
    if "price_source" in out.columns:
        out["price_source"] = out["price_source"].map(_localize_price_source)

    ordered_cols = [col for col in ALLOCATION_EXPORT_ORDER if col in out.columns]
    extra_cols = [col for col in out.columns if col not in ordered_cols]
    out = out[ordered_cols + extra_cols]
    return out.rename(columns=ALLOCATION_EXPORT_RENAME)


def _prepare_summary_export_df(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df.copy()
    if "secondary_fill_enabled" in out.columns:
        out["secondary_fill_enabled"] = out["secondary_fill_enabled"].map(_to_yes_no)
    if "pricing_source" in out.columns:
        out["pricing_source"] = out["pricing_source"].map(_localize_price_source)
    ordered_cols = [col for col in SUMMARY_EXPORT_ORDER if col in out.columns]
    extra_cols = [col for col in out.columns if col not in ordered_cols]
    out = out[ordered_cols + extra_cols]
    return out.rename(columns=SUMMARY_EXPORT_RENAME)


def _prepare_sell_signals_export_df(sell_signals_df: pd.DataFrame) -> pd.DataFrame:
    out = sell_signals_df.copy()
    if "valuation" in out.columns:
        out["valuation"] = out["valuation"].map(lambda x: VALUATION_CN_MAP.get(str(x), str(x)))
    ordered_cols = [col for col in SELL_SIGNALS_EXPORT_ORDER if col in out.columns]
    extra_cols = [col for col in out.columns if col not in ordered_cols]
    out = out[ordered_cols + extra_cols]
    return out.rename(columns=SELL_SIGNALS_EXPORT_RENAME)


def write_report(
    output_path: Path,
    allocation_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    sell_signals_df: pd.DataFrame,
) -> Path:
    allocation_export = _prepare_allocation_export_df(allocation_df)
    summary_export = _prepare_summary_export_df(summary_df)
    sell_signals_export = _prepare_sell_signals_export_df(sell_signals_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        allocation_export.to_excel(writer, sheet_name="分配", index=False)
        summary_export.to_excel(writer, sheet_name="汇总", index=False)
        sell_signals_export.to_excel(writer, sheet_name="卖出信号", index=False)
    return output_path
