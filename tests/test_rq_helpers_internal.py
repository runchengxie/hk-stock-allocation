import pandas as pd

from hk_alloc.rq_helpers import _normalize_close_output


def test_normalize_close_output_handles_none() -> None:
    out = _normalize_close_output(None, ["00941.XHKG", "01211.XHKG"])
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["00941.XHKG", "01211.XHKG"]
    assert out.empty
