from pathlib import Path

import pytest

from hk_alloc.config_loader import load_portfolio_yaml


def test_load_portfolio_yaml_minimal(tmp_path: Path) -> None:
    cfg = tmp_path / "portfolio.yml"
    cfg.write_text(
        """
portfolio:
  total_capital: 1000000
  allocation:
    method: equal
tickers:
  - ticker: "00941.HK"
  - ticker: "09987.HK"
""".strip(),
        encoding="utf-8",
    )

    portfolio, tickers = load_portfolio_yaml(cfg)
    assert portfolio.total_capital == 1_000_000
    assert portfolio.allocation_method == "equal"
    assert portfolio.secondary_fill_enabled is True
    assert portfolio.secondary_fill_avoid_high_valuation is True
    assert portfolio.secondary_fill_avoid_high_valuation_strict is False
    assert portfolio.secondary_fill_max_steps == 5000
    assert portfolio.secondary_fill_allow_over_alloc is False
    assert portfolio.secondary_fill_cash_buffer_ratio == 0.0
    assert len(tickers) == 2
    assert tickers[0].ticker == "00941.HK"


def test_load_portfolio_yaml_secondary_fill_config(tmp_path: Path) -> None:
    cfg = tmp_path / "portfolio.yml"
    cfg.write_text(
        """
portfolio:
  total_capital: 1000000
  allocation:
    method: equal
    secondary_fill:
      enabled: false
      avoid_high_valuation: false
      avoid_high_valuation_strict: true
      max_steps: 123
      allow_over_alloc: true
      max_over_alloc_ratio: 0.002
      max_over_alloc_amount: 1500
      max_over_alloc_lots_per_ticker: 1
      cash_buffer_ratio: 0.003
      cash_buffer_amount: 2000
      estimated_fee_per_order: 30
tickers:
  - ticker: "00941.HK"
""".strip(),
        encoding="utf-8",
    )

    portfolio, _ = load_portfolio_yaml(cfg)
    assert portfolio.secondary_fill_enabled is False
    assert portfolio.secondary_fill_avoid_high_valuation is False
    assert portfolio.secondary_fill_avoid_high_valuation_strict is True
    assert portfolio.secondary_fill_max_steps == 123
    assert portfolio.secondary_fill_allow_over_alloc is True
    assert portfolio.secondary_fill_max_over_alloc_ratio == 0.002
    assert portfolio.secondary_fill_max_over_alloc_amount == 1500
    assert portfolio.secondary_fill_max_over_alloc_lots_per_ticker == 1
    assert portfolio.secondary_fill_cash_buffer_ratio == 0.003
    assert portfolio.secondary_fill_cash_buffer_amount == 2000
    assert portfolio.secondary_fill_estimated_fee_per_order == 30


def test_load_portfolio_yaml_secondary_fill_max_steps_invalid(tmp_path: Path) -> None:
    cfg = tmp_path / "portfolio.yml"
    cfg.write_text(
        """
portfolio:
  total_capital: 1000000
  allocation:
    method: equal
    secondary_fill:
      max_steps: 0
tickers:
  - ticker: "00941.HK"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="secondary_fill.max_steps"):
        load_portfolio_yaml(cfg)


def test_load_portfolio_yaml_secondary_fill_over_alloc_lots_invalid(tmp_path: Path) -> None:
    cfg = tmp_path / "portfolio.yml"
    cfg.write_text(
        """
portfolio:
  total_capital: 1000000
  allocation:
    method: equal
    secondary_fill:
      allow_over_alloc: true
      max_over_alloc_lots_per_ticker: 0
tickers:
  - ticker: "00941.HK"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_over_alloc_lots_per_ticker"):
        load_portfolio_yaml(cfg)
