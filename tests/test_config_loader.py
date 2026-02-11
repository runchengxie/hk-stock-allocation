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
      max_steps: 123
tickers:
  - ticker: "00941.HK"
""".strip(),
        encoding="utf-8",
    )

    portfolio, _ = load_portfolio_yaml(cfg)
    assert portfolio.secondary_fill_enabled is False
    assert portfolio.secondary_fill_avoid_high_valuation is False
    assert portfolio.secondary_fill_max_steps == 123


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
