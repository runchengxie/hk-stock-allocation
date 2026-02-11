import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("RQDATAC_ENABLED") != "1",
    reason="Set RQDATAC_ENABLED=1 to run integration tests with real RQData",
)


@pytest.mark.integration
def test_rqdata_fetch_hk_price() -> None:
    rqdatac = pytest.importorskip("rqdatac")

    rqdatac.init()
    as_of = rqdatac.get_latest_trading_date(market="hk")
    start = rqdatac.get_previous_trading_date(as_of, n=5, market="hk")
    px = rqdatac.get_price(
        "00941.XHKG",
        start_date=start,
        end_date=as_of,
        frequency="1d",
        fields=["close"],
        market="hk",
        expect_df=True,
    )

    assert px is not None
    assert len(px) > 0
