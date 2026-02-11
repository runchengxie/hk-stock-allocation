from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path

from .config_loader import load_portfolio_yaml
from .rq_helpers import (
    build_allocation_table,
    build_sell_signals,
    init_rqdata,
    write_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HK allocation and sell signal report")
    parser.add_argument(
        "--config",
        default="configs/portfolio.yml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="As-of date (YYYY-MM-DD). Default is latest HK trading day.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output xlsx path. Default: output/{portfolio}_{as_of}.xlsx",
    )
    return parser.parse_args()


def _parse_as_of(as_of_raw: str | None) -> date | None:
    if not as_of_raw:
        return None
    return datetime.strptime(as_of_raw, "%Y-%m-%d").date()


def cli() -> None:
    args = _parse_args()
    portfolio, tickers = load_portfolio_yaml(args.config)

    rqdatac = init_rqdata()
    as_of = _parse_as_of(args.as_of)
    if as_of is None:
        as_of = rqdatac.get_latest_trading_date(market=portfolio.market)

    allocation_df, summary_df = build_allocation_table(
        rqdatac,
        portfolio=portfolio,
        ticker_configs=tickers,
        as_of=as_of,
    )
    sell_signals_df = build_sell_signals(
        rqdatac,
        portfolio=portfolio,
        ticker_configs=tickers,
        as_of=as_of,
    )

    output_path = Path(args.output) if args.output else Path("output") / f"{portfolio.name}_{as_of}.xlsx"
    output_path = write_report(output_path, allocation_df, summary_df, sell_signals_df)

    print(f"Saved report: {output_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    cli()
