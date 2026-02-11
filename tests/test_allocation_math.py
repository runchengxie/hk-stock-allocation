import pandas as pd

from hk_alloc.config_loader import TickerConfig
from hk_alloc.rq_helpers import apply_secondary_fill, build_target_values, calc_lots


def _fill_kwargs(**overrides):
    base = {
        "total_capital": 1_000.0,
        "enabled": True,
        "avoid_high_valuation": True,
        "avoid_high_valuation_strict": False,
        "max_steps": 10,
        "allow_over_alloc": False,
        "max_over_alloc_ratio": 0.0,
        "max_over_alloc_amount": 0.0,
        "max_over_alloc_lots_per_ticker": 1,
        "cash_buffer_ratio": 0.0,
        "cash_buffer_amount": 0.0,
        "estimated_fee_per_order": 0.0,
    }
    base.update(overrides)
    return base


def test_calc_lots_whole_lot_only() -> None:
    assert calc_lots(50_000, 80.4, 500, tradable=True) == 1


def test_calc_lots_zero_when_untradable() -> None:
    assert calc_lots(50_000, 80.4, 500, tradable=False) == 0


def test_build_target_values_equal() -> None:
    tickers = [TickerConfig(ticker="00941.HK"), TickerConfig(ticker="09987.HK")]
    targets = build_target_values(1_000_000, tickers, allocation_method="equal")
    assert targets["00941.HK"] == 500_000
    assert targets["09987.HK"] == 500_000


def test_build_target_values_custom() -> None:
    tickers = [
        TickerConfig(ticker="00941.HK", weight=1.0),
        TickerConfig(ticker="09987.HK", weight=3.0),
    ]
    targets = build_target_values(1_000_000, tickers, allocation_method="custom")
    assert targets["00941.HK"] == 250_000
    assert targets["09987.HK"] == 750_000


def test_apply_secondary_fill_reduces_cash_gap() -> None:
    allocation = pd.DataFrame(
        [
            {
                "ticker": "A.HK",
                "valuation": "LOW",
                "tradable": True,
                "price": 1.2,
                "lot_cost": 120.0,
                "target_value": 500.0,
                "lots": 3,
                "lots_extra": 0,
                "round_lot": 100,
                "shares": 300,
                "est_value": 360.0,
                "gap_to_target": 140.0,
            },
            {
                "ticker": "B.HK",
                "valuation": "NEUTRAL",
                "tradable": True,
                "price": 0.7,
                "lot_cost": 70.0,
                "target_value": 500.0,
                "lots": 6,
                "lots_extra": 0,
                "round_lot": 100,
                "shares": 600,
                "est_value": 420.0,
                "gap_to_target": 80.0,
            },
        ]
    )

    updated, stats = apply_secondary_fill(allocation, **_fill_kwargs())

    assert float(updated["est_value"].sum()) > 780.0
    assert float(updated["gap_to_target"].sum()) < 220.0
    assert stats["secondary_fill_steps"] >= 1


def test_apply_secondary_fill_avoids_high_first() -> None:
    allocation = pd.DataFrame(
        [
            {
                "ticker": "LOW.HK",
                "valuation": "LOW",
                "tradable": True,
                "price": 0.9,
                "lot_cost": 90.0,
                "target_value": 500.0,
                "lots": 3,
                "lots_extra": 0,
                "round_lot": 100,
                "shares": 300,
                "est_value": 270.0,
                "gap_to_target": 230.0,
            },
            {
                "ticker": "HIGH.HK",
                "valuation": "HIGH",
                "tradable": True,
                "price": 1.0,
                "lot_cost": 100.0,
                "target_value": 500.0,
                "lots": 3,
                "lots_extra": 0,
                "round_lot": 100,
                "shares": 300,
                "est_value": 300.0,
                "gap_to_target": 200.0,
            },
        ]
    )

    updated, stats = apply_secondary_fill(allocation, **_fill_kwargs(max_steps=1))

    assert stats["secondary_fill_steps"] == 1
    low_extra = int(updated.loc[updated["ticker"] == "LOW.HK", "lots_extra"].iloc[0])
    high_extra = int(updated.loc[updated["ticker"] == "HIGH.HK", "lots_extra"].iloc[0])
    assert low_extra == 1
    assert high_extra == 0


def test_apply_secondary_fill_strictly_avoids_high() -> None:
    allocation = pd.DataFrame(
        [
            {
                "ticker": "HIGH_ONLY.HK",
                "valuation": "HIGH",
                "tradable": True,
                "price": 1.0,
                "lot_cost": 100.0,
                "target_value": 500.0,
                "lots": 3,
                "lots_extra": 0,
                "round_lot": 100,
                "shares": 300,
                "est_value": 300.0,
                "gap_to_target": 200.0,
            }
        ]
    )

    updated, stats = apply_secondary_fill(
        allocation,
        **_fill_kwargs(avoid_high_valuation=True, avoid_high_valuation_strict=True),
    )

    assert stats["secondary_fill_steps"] == 0
    assert int(updated.loc[0, "lots_extra"]) == 0


def test_apply_secondary_fill_does_not_over_alloc_by_default() -> None:
    allocation = pd.DataFrame(
        [
            {
                "ticker": "A.HK",
                "valuation": "LOW",
                "tradable": True,
                "price": 1.0,
                "lot_cost": 100.0,
                "target_value": 560.0,
                "lots": 5,
                "lots_extra": 0,
                "round_lot": 100,
                "shares": 500,
                "est_value": 500.0,
                "gap_to_target": 60.0,
            }
        ]
    )

    updated, stats = apply_secondary_fill(allocation, **_fill_kwargs())

    assert stats["secondary_fill_steps"] == 0
    assert float(updated.loc[0, "gap_to_target"]) == 60.0


def test_apply_secondary_fill_allows_bounded_over_alloc_when_enabled() -> None:
    allocation = pd.DataFrame(
        [
            {
                "ticker": "A.HK",
                "valuation": "LOW",
                "tradable": True,
                "price": 1.0,
                "lot_cost": 100.0,
                "target_value": 560.0,
                "lots": 5,
                "lots_extra": 0,
                "round_lot": 100,
                "shares": 500,
                "est_value": 500.0,
                "gap_to_target": 60.0,
            }
        ]
    )

    updated, stats = apply_secondary_fill(
        allocation,
        **_fill_kwargs(
            allow_over_alloc=True,
            max_over_alloc_amount=60.0,
            max_over_alloc_lots_per_ticker=1,
        ),
    )

    assert stats["secondary_fill_steps"] == 1
    assert float(updated.loc[0, "gap_to_target"]) == -40.0


def test_apply_secondary_fill_respects_cash_buffer_and_fees() -> None:
    allocation = pd.DataFrame(
        [
            {
                "ticker": "A.HK",
                "valuation": "LOW",
                "tradable": True,
                "price": 1.0,
                "lot_cost": 100.0,
                "target_value": 1_000.0,
                "lots": 7,
                "lots_extra": 0,
                "round_lot": 100,
                "shares": 700,
                "est_value": 700.0,
                "gap_to_target": 300.0,
            }
        ]
    )

    updated, stats = apply_secondary_fill(
        allocation,
        **_fill_kwargs(
            total_capital=1_000.0,
            cash_buffer_amount=100.0,
            estimated_fee_per_order=20.0,
        ),
    )

    # Budget after buffer is 900. Initial est_value is 700, so available cash is 200.
    # One step needs 120 cash (100 lot + 20 fee), so only one fill is possible.
    assert stats["secondary_fill_steps"] == 1
    assert float(updated.loc[0, "est_value"]) == 800.0
    assert float(stats["secondary_fill_fee_spent"]) == 20.0
    assert float(stats["secondary_fill_cash_buffer"]) == 100.0
    assert float(stats["cash_remaining_after_fill"]) == 180.0
