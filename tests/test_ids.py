from hk_alloc.rq_helpers import ticker_to_rq_order_book_id


def test_ticker_to_rq_order_book_id_zero_pad() -> None:
    assert ticker_to_rq_order_book_id("316.HK") == "00316.XHKG"
    assert ticker_to_rq_order_book_id("00941.HK") == "00941.XHKG"


def test_ticker_to_rq_order_book_id_passthrough() -> None:
    assert ticker_to_rq_order_book_id("00941.XHKG") == "00941.XHKG"
