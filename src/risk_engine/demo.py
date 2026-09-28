"""End-to-end run: from the trade journal and closes to every figure of the report."""

from __future__ import annotations

import pandas as pd

from risk_engine import (
    backtest,
    benchmark,
    dividends,
    exposures,
    journal,
    liquidity,
    performance,
    portfolio,
    quality,
    register,
    risk,
    stress,
)
from risk_engine.data import fx
from risk_engine.data.synthetic import MarketData
from witness import risk_witness

WITNESS_TOLERANCE = 1e-9


def engine_figures(result: dict) -> dict[str, float]:
    """The engine's risk figures, keyed like the witness output."""
    out = {"n_days": float(result["n_days"]),
           "volatility_daily_pct": result["volatility_daily_pct"],
           "volatility_annualized_pct": result["volatility_annualized_pct"],
           "max_drawdown_pct": result["max_drawdown_pct"]}
    for (method, level), m in result["measures"].items():
        out[f"{method}_var_{level * 100:g}"] = m["var_pct"]
        out[f"{method}_es_{level * 100:g}"] = m["es_pct"]
    return out


def witness_check(market: MarketData, quantities: pd.Series, as_of: pd.Timestamp,
                  result: dict) -> dict:
    """Compare the engine with the independent witness; the gap must stay below 1e-9."""
    w = risk_witness.independent_calculation(market.prices, market.fx_to_eur,
                                             market.securities, quantities, as_of)
    gap, where = risk_witness.max_relative_difference(engine_figures(result), w)
    return {"max_rel_diff": gap, "where": where, "ok": gap <= WITNESS_TOLERANCE}


def seed_study(seeds: range) -> pd.DataFrame:
    """VaR backtest of both methods on many synthetic histories, to separate a pattern
    from the luck of one draw."""
    from risk_engine.data.synthetic import generate_market

    rows = []
    for seed in seeds:
        market = generate_market(seed)
        as_of = market.prices.index[market.prices.index.weekday < 5][-1]
        eur = fx.to_eur(market.prices, market.securities, market.fx_to_eur)
        qty = portfolio.current_quantities(journal.build_lots(market.journal).lots, as_of)
        values = portfolio.value_positions(qty, eur, as_of)["lines"]["value_eur"]
        returns = risk.compute_advanced_risk(values, eur, as_of)["returns_pct"]
        for method in ("historical", "normal"):
            b = backtest.backtest_var(returns, method)
            rows.append({"seed": seed, "method": method, "exceptions": b["exceptions"],
                         "expected": b["expected"], "kupiec_p": b["kupiec_p"]})
    return pd.DataFrame(rows)


def run(market: MarketData, live: bool = False) -> dict:
    """Every block of the report, computed in dependency order."""
    secs = market.securities
    as_of = market.prices.index[market.prices.index.weekday < 5][-1]
    eur = fx.to_eur(market.prices, secs, market.fx_to_eur)
    eur_quote = fx.to_eur(market.prices, secs, market.fx_to_eur, apply_unit_factor=False)
    journal.validate_journal(market.journal, set(secs["currency"]), as_of)
    book = journal.build_lots(market.journal)
    journal.reconcile_with_holdings(book, market.declared_holdings)
    qty = portfolio.current_quantities(book.lots, as_of)
    cash = fx.cash_to_eur(market.cash, market.fx_to_eur, as_of)
    valuation = portfolio.value_positions(qty, eur, as_of, cash)
    values = valuation["lines"]["value_eur"]
    divs = dividends.dividends_eur(market.dividends, market.prices, eur_quote)
    history = portfolio.portfolio_value_history(book.lots, eur, divs)
    periods = [performance.period_block(book.lots, eur, history, p, as_of)
               for p in performance.PERIODS]
    received = dividends.dividends_received(book.lots, divs)
    reg = register.build_register(market.journal, book, market.prices, eur, received, secs)
    exp = exposures.compute_exposures(values, secs, cash, market.betas)
    adv = liquidity.average_volumes(market.volumes)
    result = risk.compute_advanced_risk(values, eur, as_of)
    equity_venue = secs.loc[secs["asset_class"] == "equity", "venue"]
    out = {
        "live": live, "as_of": as_of, "book": book, "quantities": qty, "securities": secs,
        "local_close": market.prices.loc[:as_of].ffill().iloc[-1],
        "valuation": valuation, "history": history, "periods": periods,
        "register": reg, "register_summary": register.register_summary(reg),
        "yield_on_cost": dividends.yield_on_cost(received, book.open_cost_eur,
                                                 as_of - pd.DateOffset(years=1)),
        "exposures": exp, "sentences": exposures.interpretation_sentences(exp),
        "risk": result, "witness": witness_check(market, qty, as_of, result),
        "backtest": [backtest.backtest_var(result["returns_pct"], m)
                     for m in ("historical", "normal")],
        "liquidity": liquidity.compute_liquidity_metrics(qty, values, adv),
        "scenarios": stress.run_scenarios(values, secs, cash),
        "price_jumps": quality.scan_price_jumps(market.prices),
        "freshness": quality.freshness(market.prices, equity_venue, as_of),
        "gap_shortfall": quality.gap_shortfall(market.prices, equity_venue),
        "raw_unit_trades": quality.raw_unit_trades(market.journal, market.prices, secs),
    }
    if live:
        out["historical"] = [dict(stress.worst_window(eur, values, h["start"], h["end"]) or {},
                                  name=h["name"]) for h in stress.load_scenarios()["historical"]]
    else:
        worst = stress.worst_window(eur, values)
        out["historical"] = [dict(worst, name="Worst 5 sessions of the sample")]
        weights, unmapped = benchmark.composite_weights(exp["currency_ex_cash"])
        held_from = book.lots[book.lots["ticker"].isin(qty.index)].groupby("ticker")["date"].min()
        out["benchmark"] = benchmark.compute_alpha_beta(
            history["total_return_index"], history["price_index"], market.bricks,
            market.brick_distributions, weights, held_from.max())
        out["benchmark"]["unmapped"] = unmapped
    return out
