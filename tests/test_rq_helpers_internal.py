from types import SimpleNamespace

import pandas as pd

from hk_alloc.rq_helpers import _normalize_close_output, _resolve_display_name, fetch_instruments


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
