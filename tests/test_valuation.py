import numpy as np
import pandas as pd

from hk_alloc.rq_helpers import classify_valuation, compute_valuation_metrics


def test_classify_valuation_extreme_by_percentile() -> None:
    assert classify_valuation(0.995, 0.0) == "EXTREME"


def test_classify_valuation_high_by_zscore() -> None:
    assert classify_valuation(0.6, 2.1) == "HIGH"


def test_classify_valuation_neutral() -> None:
    assert classify_valuation(0.5, 0.3) == "NEUTRAL"


def test_compute_valuation_metrics_shapes() -> None:
    idx = pd.date_range("2025-01-01", periods=6, freq="D")
    close = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6]}, index=idx)
    pct, z, qh, qe = compute_valuation_metrics(close, window=3, sell_quantile=0.8, extreme_quantile=0.9)

    assert pct.shape == close.shape
    assert z.shape == close.shape
    assert qh.shape == close.shape
    assert qe.shape == close.shape
    assert np.isfinite(pct.iloc[-1, 0])
