from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from hk_alloc.config_loader import PortfolioConfig, TickerConfig, ValuationConfig
from hk_alloc.rq_helpers import (
    MarketDataBundle,
    ScenarioReport,
    _is_stock_connect_tradable,
    _normalize_close_output,
    _prepare_allocation_export_df,
    _prepare_sell_signals_export_df,
    _prepare_summary_export_df,
    _resolve_display_name,
    build_allocation_table,
    build_latest_price_frame,
    build_sell_signals,
    fetch_instruments,
    write_report,
    write_scenario_grid_report,
)


def test_normalize_close_output_handles_none() -> None:
    out = _normalize_close_output(None, ["00941.XHKG", "01211.XHKG"])
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["00941.XHKG", "01211.XHKG"]
    assert out.empty


def test_fetch_instruments_deduplicates_and_keeps_valid_values() -> None:
    class DummyRQ:
        def instruments(self, order_book_ids: list[str], market: str):  # noqa: ANN201
            assert order_book_ids == ["00300.XHKG"]
            assert market == "hk"
            return [
                SimpleNamespace(
                    order_book_id="00300.XHKG",
                    symbol=None,
                    round_lot=None,
                    stock_connect=None,
                ),
                SimpleNamespace(
                    order_book_id="00300.XHKG",
                    symbol="Midea Group",
                    round_lot=100,
                    stock_connect=["sh", "sz"],
                ),
            ]

    out = fetch_instruments(DummyRQ(), ["00300.XHKG"], market="hk")
    assert list(out.index) == ["00300.XHKG"]
    assert out.index.is_unique
    assert out.loc["00300.XHKG", "symbol"] == "Midea Group"
    assert out.loc["00300.XHKG", "round_lot"] == 100.0
    assert out.loc["00300.XHKG", "stock_connect"] == ["sh", "sz"]


def test_resolve_display_name_falls_back_to_symbol_when_config_name_missing() -> None:
    assert _resolve_display_name("Configured Name", "RQ Symbol") == "Configured Name"
    assert _resolve_display_name("   ", "RQ Symbol") == "RQ Symbol"
    assert _resolve_display_name(None, None) is None


def test_build_latest_price_frame_prefers_snapshot_then_minute_then_close() -> None:
    class DummyRQ:
        def get_previous_trading_date(self, ref_date, n, market):  # noqa: ANN201, ANN001
            return ref_date

        def get_price(  # noqa: ANN201, ANN001
            self,
            order_book_ids,
            start_date,
            end_date,
            frequency,
            fields,
            adjust_type,
            market,
            expect_df,
        ):
            assert frequency == "1d"
            assert fields == ["close"]
            assert adjust_type == "none"
            assert market == "hk"
            return pd.DataFrame(
                {"00941.XHKG": [80.0], "01211.XHKG": [300.0]},
                index=[pd.Timestamp("2026-02-10")],
            )

        def current_snapshot(self, order_book_ids, market):  # noqa: ANN201, ANN001
            assert market == "hk"
            return [
                SimpleNamespace(
                    order_book_id="00941.XHKG",
                    last=81.0,
                    close=80.0,
                    datetime=pd.Timestamp("2026-02-11 10:15:00"),
                ),
                SimpleNamespace(
                    order_book_id="01211.XHKG",
                    last=float("nan"),
                    close=float("nan"),
                    datetime=pd.Timestamp("2026-02-11 10:15:00"),
                ),
            ]

        def current_minute(self, order_book_ids, fields, market):  # noqa: ANN201, ANN001
            assert fields == ["close"]
            assert market == "hk"
            idx = pd.MultiIndex.from_tuples(
                [
                    ("01211.XHKG", pd.Timestamp("2026-02-11 10:16:00")),
                ],
                names=["order_book_id", "datetime"],
            )
            return pd.DataFrame({"close": [301.0]}, index=idx)

    as_of = pd.Timestamp.now(tz="Asia/Hong_Kong").date()
    out = build_latest_price_frame(
        DummyRQ(),
        order_book_ids=["00941.XHKG", "01211.XHKG"],
        as_of=as_of,
        market="hk",
    )

    assert float(out.at["00941.XHKG", "price"]) == 81.0
    assert out.at["00941.XHKG", "price_source"] == "snapshot"
    assert float(out.at["01211.XHKG", "price"]) == 301.0
    assert out.at["01211.XHKG", "price_source"] == "1m_close"


def test_prepare_allocation_export_df_localizes_headers_and_values() -> None:
    allocation = pd.DataFrame(
        [
            {
                "name": "中国移动",
                "ticker": "00941.HK",
                "order_book_id": "00941.XHKG",
                "tradable": True,
                "stock_connect": ["sh", "sz"],
                "price_source": "snapshot",
                "pricing_date": pd.Timestamp("2026-02-11").date(),
                "price": 80.5,
                "round_lot": 500,
                "lot_cost": 40250.0,
                "target_value": 50000.0,
                "lots_base": 1,
                "lots_extra": 1,
                "lots": 2,
                "shares": 1000,
                "est_value": 80500.0,
                "gap_to_target": -30500.0,
                "gap_ratio": -0.61,
                "valuation": "HIGH",
                "pct_1y": 0.98,
                "z_1y": 2.1,
                "overpriced_low": 78.0,
                "overpriced_high": 82.0,
                "overpriced_range": "[78.0000, 82.0000]",
            }
        ]
    )

    out = _prepare_allocation_export_df(allocation)
    assert list(out.columns[:6]) == ["股票代码", "名称", "合计手数", "当前价格", "估值分层", "统计高位上沿(未复权)"]
    assert out.loc[0, "可交易"] == "是"
    assert out.loc[0, "港股通"] == "沪/深"
    assert out.loc[0, "价格来源"] == "快照最新价"
    assert out.loc[0, "估值分层"] == "偏高"


def test_write_report_uses_chinese_sheet_names(tmp_path: Path) -> None:
    allocation = pd.DataFrame([{"ticker": "00941.HK", "name": "中国移动", "lots": 2, "price": 80.5}])
    summary = pd.DataFrame([{"portfolio_name": "hk_core_20"}])
    sell_signals = pd.DataFrame([{"ticker": "00941.HK", "valuation": "HIGH"}])

    out_path = tmp_path / "report.xlsx"
    write_report(out_path, allocation, summary, sell_signals)

    with pd.ExcelFile(out_path) as xls:
        assert xls.sheet_names == ["分配", "汇总", "卖出信号"]


def test_write_scenario_grid_report_writes_overview_and_scenario_sheets(tmp_path: Path) -> None:
    report_a = ScenarioReport(
        scenario_id="C100w_N20",
        allocation_df=pd.DataFrame([{"ticker": "00941.HK", "lots": 2}]),
        summary_df=pd.DataFrame([{"portfolio_name": "hk_core_20", "total_capital": 1_000_000.0}]),
        sell_signals_df=pd.DataFrame([{"ticker": "00941.HK", "valuation": "HIGH"}]),
    )
    report_b = ScenarioReport(
        scenario_id="C50w_N10",
        allocation_df=pd.DataFrame([{"ticker": "01211.HK", "lots": 1}]),
        summary_df=pd.DataFrame([{"portfolio_name": "hk_core_20", "total_capital": 500_000.0}]),
        sell_signals_df=pd.DataFrame([{"ticker": "01211.HK", "valuation": "LOW"}]),
    )

    out_path = tmp_path / "scenario_grid.xlsx"
    write_scenario_grid_report(out_path, [report_a, report_b])

    with pd.ExcelFile(out_path) as xls:
        assert xls.sheet_names == [
            "场景总览",
            "C100w_N20_分配",
            "C100w_N20_汇总",
            "C100w_N20_卖出",
            "C50w_N10_分配",
            "C50w_N10_汇总",
            "C50w_N10_卖出",
        ]


def test_prepare_summary_export_df_orders_and_localizes() -> None:
    summary = pd.DataFrame(
        [
            {
                "as_of": pd.Timestamp("2026-02-11").date(),
                "pricing_date": pd.Timestamp("2026-02-11").date(),
                "pricing_source": "snapshot",
                "pricing_source_detail": "snapshot:20",
                "portfolio_name": "hk_core_20",
                "num_tickers": 20,
                "total_capital": 1_000_000.0,
                "total_est_value": 980_000.0,
                "total_gap": 20_000.0,
                "cash_used_ratio": 0.98,
                "secondary_fill_enabled": True,
                "secondary_fill_steps": 3,
                "secondary_fill_spent": 60_000.0,
                "secondary_fill_fee_spent": 100.0,
                "secondary_fill_cash_buffer": 2_000.0,
                "secondary_fill_budget_after_buffer": 998_000.0,
                "cash_remaining_after_fill": 20_000.0,
            }
        ]
    )

    out = _prepare_summary_export_df(summary)
    assert list(out.columns[:6]) == ["组合名称", "统计日期", "定价日期", "价格来源", "价格来源明细", "标的数量"]
    assert out.loc[0, "价格来源"] == "快照最新价"
    assert out.loc[0, "启用二次补仓"] == "是"
    assert out.loc[0, "补仓估算费用"] == 100.0


def test_fetch_instruments_round_lot_prefers_mode_over_max(caplog: pytest.LogCaptureFixture) -> None:
    class DummyRQ:
        def instruments(self, order_book_ids: list[str], market: str):  # noqa: ANN201
            assert order_book_ids == ["00300.XHKG"]
            assert market == "hk"
            return [
                SimpleNamespace(
                    order_book_id="00300.XHKG",
                    symbol="Midea Group",
                    round_lot=100,
                    stock_connect=["sh", "sz"],
                ),
                SimpleNamespace(
                    order_book_id="00300.XHKG",
                    symbol="Midea Group",
                    round_lot=200,
                    stock_connect=["sh", "sz"],
                ),
                SimpleNamespace(
                    order_book_id="00300.XHKG",
                    symbol="Midea Group",
                    round_lot=100,
                    stock_connect=["sh", "sz"],
                ),
            ]

    with caplog.at_level("WARNING"):
        out = fetch_instruments(DummyRQ(), ["00300.XHKG"], market="hk")
    assert out.loc["00300.XHKG", "round_lot"] == 100.0
    assert "Multiple round_lot values found" in caplog.text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("yes", True),
        ("是", True),
        ("沪港通", True),
        ("sh,sz", True),
        ("否", False),
        ("not eligible", False),
        ("unknown", False),
        (["sh", "sz"], True),
        (["none", "否"], False),
    ],
)
def test_is_stock_connect_tradable_handles_multilingual_values(raw, expected) -> None:  # noqa: ANN001
    assert _is_stock_connect_tradable(raw) is expected


def test_build_allocation_table_uses_unadjusted_history_for_thresholds() -> None:
    calls: list[str] = []

    class DummyRQ:
        def get_previous_trading_date(self, ref_date, n, market):  # noqa: ANN201, ANN001
            return ref_date

        def get_price(  # noqa: ANN201, ANN001
            self,
            order_book_ids,
            start_date,
            end_date,
            frequency,
            fields,
            adjust_type,
            market,
            expect_df,
        ):
            calls.append(str(adjust_type))
            assert adjust_type == "none"
            assert order_book_ids == ["00941.XHKG"]
            return pd.DataFrame(
                {"00941.XHKG": [80.0, 81.0, 82.0]},
                index=pd.to_datetime(["2026-02-08", "2026-02-09", "2026-02-10"]),
            )

        def instruments(self, order_book_ids: list[str], market: str):  # noqa: ANN201
            return [
                SimpleNamespace(
                    order_book_id="00941.XHKG",
                    symbol="China Mobile",
                    round_lot=500,
                    stock_connect=["sh", "sz"],
                )
            ]

    portfolio = PortfolioConfig(
        name="demo",
        currency="HKD",
        total_capital=100_000,
        secondary_fill_enabled=False,
        valuation=ValuationConfig(
            history_years=1,
            roll_window=2,
            sell_quantile=0.8,
            extreme_quantile=0.9,
        ),
    )
    tickers = [TickerConfig(ticker="00941.HK")]
    allocation_df, _ = build_allocation_table(
        DummyRQ(),
        portfolio=portfolio,
        ticker_configs=tickers,
        as_of=pd.Timestamp("2026-02-10").date(),
    )

    assert calls
    assert set(calls) == {"none"}
    assert "overpriced_high" in allocation_df.columns


def test_build_allocation_and_sell_signals_use_prefetched_bundle() -> None:
    class DummyRQ:
        pass

    portfolio = PortfolioConfig(
        name="demo",
        currency="HKD",
        total_capital=100_000,
        secondary_fill_enabled=False,
        valuation=ValuationConfig(
            history_years=1,
            roll_window=2,
            sell_quantile=0.8,
            extreme_quantile=0.9,
        ),
    )
    tickers = [TickerConfig(ticker="00941.HK", rank=1, signal=0.123)]
    oid = "00941.XHKG"

    instruments_df = pd.DataFrame(
        [{"order_book_id": oid, "symbol": "China Mobile", "round_lot": 500, "stock_connect": ["sh", "sz"]}]
    ).set_index("order_book_id")
    latest_prices = pd.DataFrame(
        index=[oid],
        data={
            "price": [80.0],
            "price_source": ["1d_close"],
            "pricing_ts": [pd.Timestamp("2026-02-10")],
            "pricing_date": [pd.Timestamp("2026-02-10").date()],
        },
    )
    close_none = pd.DataFrame(
        {oid: [78.0, 79.0, 80.0]},
        index=pd.to_datetime(["2026-02-08", "2026-02-09", "2026-02-10"]),
    )
    close_pre = pd.DataFrame(
        {oid: [1.0, 0.5, 2.0]},
        index=pd.to_datetime(["2026-02-08", "2026-02-09", "2026-02-10"]),
    )
    bundle = MarketDataBundle(
        tickers=("00941.HK",),
        order_book_ids=(oid,),
        ticker_to_oid={"00941.HK": oid},
        instruments_df=instruments_df,
        latest_prices=latest_prices,
        close_none=close_none,
        close_pre=close_pre,
    )

    allocation_df, _ = build_allocation_table(
        DummyRQ(),
        portfolio=portfolio,
        ticker_configs=tickers,
        as_of=pd.Timestamp("2026-02-10").date(),
        market_data=bundle,
    )
    sell_df = build_sell_signals(
        DummyRQ(),
        portfolio=portfolio,
        ticker_configs=tickers,
        as_of=pd.Timestamp("2026-02-10").date(),
        market_data=bundle,
    )

    assert allocation_df.loc[0, "rank"] == 1
    assert allocation_df.loc[0, "signal"] == 0.123
    assert sell_df.loc[0, "rank"] == 1
    assert sell_df.loc[0, "signal"] == 0.123


def test_build_sell_signals_uses_prior_day_trigger_for_cross() -> None:
    class DummyRQ:
        def get_previous_trading_date(self, ref_date, n, market):  # noqa: ANN201, ANN001
            return ref_date

        def get_price(  # noqa: ANN201, ANN001
            self,
            order_book_ids,
            start_date,
            end_date,
            frequency,
            fields,
            adjust_type,
            market,
            expect_df,
        ):
            assert adjust_type == "pre"
            return pd.DataFrame(
                {"00941.XHKG": [1.0, 0.5, 2.0]},
                index=pd.to_datetime(["2026-02-08", "2026-02-09", "2026-02-10"]),
            )

        def instruments(self, order_book_ids: list[str], market: str):  # noqa: ANN201
            return [
                SimpleNamespace(
                    order_book_id="00941.XHKG",
                    symbol="China Mobile",
                    round_lot=500,
                    stock_connect=["sh", "sz"],
                )
            ]

    portfolio = PortfolioConfig(
        name="demo",
        currency="HKD",
        total_capital=100_000,
        valuation=ValuationConfig(
            history_years=1,
            roll_window=2,
            sell_quantile=0.8,
            extreme_quantile=0.9,
        ),
    )
    tickers = [TickerConfig(ticker="00941.HK")]
    out = build_sell_signals(
        DummyRQ(),
        portfolio=portfolio,
        ticker_configs=tickers,
        as_of=pd.Timestamp("2026-02-10").date(),
    )

    assert out.loc[0, "last_sell_signal_date"] == pd.Timestamp("2026-02-10").date()


def test_prepare_sell_signals_export_df_orders_and_localizes() -> None:
    sell_signals = pd.DataFrame(
        [
            {
                "name": "中国移动",
                "ticker": "00941.HK",
                "order_book_id": "00941.XHKG",
                "as_of": pd.Timestamp("2026-02-11").date(),
                "close_pre": 80.5,
                "pct_1y": 0.98,
                "z_1y": 2.1,
                "sell_trigger": 0.95,
                "extreme_trigger": 0.99,
                "last_sell_signal_date": pd.Timestamp("2026-01-20").date(),
                "valuation": "HIGH",
            }
        ]
    )

    out = _prepare_sell_signals_export_df(sell_signals)
    assert list(out.columns[:6]) == ["股票代码", "名称", "前复权收盘价", "估值分层", "偏高阈值", "极高阈值"]
    assert out.loc[0, "估值分层"] == "偏高"
