from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .config_loader import load_portfolio_yaml
from .rq_helpers import (
    ScenarioReport,
    build_allocation_table,
    build_sell_signals,
    init_rqdata,
    prefetch_market_data,
    write_report,
    write_scenario_grid_report,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HK allocation and sell signal report")
    parser.add_argument(
        "--config",
        default="configs/universe/portfolio.yml",
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
        help="Output xlsx path. Default: output/{portfolio}_{as_of}[...].xlsx",
    )
    return parser.parse_args()


def _parse_as_of(as_of_raw: str | None) -> date | None:
    if not as_of_raw:
        return None
    return datetime.strptime(as_of_raw, "%Y-%m-%d").date()


def _format_capital_tag(capital: float) -> str:
    if float(capital).is_integer() and int(capital) % 10_000 == 0:
        return f"{int(capital) // 10_000}w"
    if float(capital).is_integer():
        return str(int(capital))
    return str(capital).replace(".", "p")


def _build_scenario_id(capital: float, top_n: int) -> str:
    return f"C{_format_capital_tag(capital)}_N{int(top_n)}"


def _default_output_path(
    portfolio_name: str,
    as_of: date,
    capitals: tuple[float, ...],
    top_ns: tuple[int, ...],
) -> Path:
    if len(capitals) == 1 and len(top_ns) == 1:
        return Path("output") / f"{portfolio_name}_{as_of}.xlsx"
    capital_part = "-".join(_format_capital_tag(c) for c in capitals)
    top_n_part = "-".join(str(n) for n in top_ns)
    return Path("output") / f"{portfolio_name}_{as_of}_grid_C{capital_part}_N{top_n_part}.xlsx"


def cli() -> None:
    args = _parse_args()
    portfolio, tickers = load_portfolio_yaml(args.config)
    active_tickers = [item for item in tickers if item.enabled]
    if len(active_tickers) == 0:
        raise ValueError("No enabled tickers found")

    scenario_capitals = portfolio.scenario_capitals or (portfolio.total_capital,)
    scenario_top_ns = portfolio.scenario_top_ns or (len(active_tickers),)
    max_top_n = max(scenario_top_ns)
    scenario_universe = active_tickers[:max_top_n]

    rqdatac = init_rqdata()
    as_of = _parse_as_of(args.as_of)
    if as_of is None:
        as_of = rqdatac.get_latest_trading_date(market=portfolio.market)

    market_data = prefetch_market_data(
        rqdatac,
        portfolio=portfolio,
        ticker_configs=scenario_universe,
        as_of=as_of,
    )

    scenario_reports: list[ScenarioReport] = []
    for capital in scenario_capitals:
        for top_n in scenario_top_ns:
            scenario_id = _build_scenario_id(capital, top_n)
            scenario_portfolio = replace(portfolio, total_capital=float(capital))
            scenario_tickers = scenario_universe[:top_n]

            allocation_df, summary_df = build_allocation_table(
                rqdatac,
                portfolio=scenario_portfolio,
                ticker_configs=scenario_tickers,
                as_of=as_of,
                market_data=market_data,
            )
            summary_df["scenario_id"] = scenario_id
            summary_df["scenario_capital"] = float(capital)
            summary_df["scenario_top_n"] = int(top_n)

            sell_signals_df = build_sell_signals(
                rqdatac,
                portfolio=scenario_portfolio,
                ticker_configs=scenario_tickers,
                as_of=as_of,
                market_data=market_data,
            )
            scenario_reports.append(
                ScenarioReport(
                    scenario_id=scenario_id,
                    allocation_df=allocation_df,
                    summary_df=summary_df,
                    sell_signals_df=sell_signals_df,
                )
            )

    output_path = (
        Path(args.output)
        if args.output
        else _default_output_path(portfolio.name, as_of, scenario_capitals, scenario_top_ns)
    )
    if len(scenario_reports) == 1:
        only = scenario_reports[0]
        output_path = write_report(output_path, only.allocation_df, only.summary_df, only.sell_signals_df)
    else:
        output_path = write_scenario_grid_report(output_path, scenario_reports)

    summary_all = pd.concat([item.summary_df for item in scenario_reports], ignore_index=True)

    print(f"Saved report: {output_path}")
    print(summary_all.to_string(index=False))


if __name__ == "__main__":
    cli()
