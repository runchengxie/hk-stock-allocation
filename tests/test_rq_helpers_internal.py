from types import SimpleNamespace

import pandas as pd

from hk_alloc.rq_helpers import (
    _normalize_close_output,
    _prepare_allocation_export_df,
    _resolve_display_name,
    build_latest_price_frame,
    fetch_instruments,
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
    assert list(out.columns[:5]) == ["股票代码", "合计手数", "估值分层", "高估上沿", "名称"]
    assert out.loc[0, "可交易"] == "是"
    assert out.loc[0, "港股通"] == "沪/深"
    assert out.loc[0, "价格来源"] == "快照最新价"
    assert out.loc[0, "估值分层"] == "偏高"
