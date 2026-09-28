"""Console report and the two charts of the README."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
BLUE, ORANGE, VIOLET, RED = "#2a78d6", "#eb6834", "#4a3aa7", "#e34948"


def _eur(x: float) -> str:
    return f"{x:>14,.0f}"


def _rule(title: str) -> str:
    return f"\n{title}\n{'-' * len(title)}"


def format_report(r: dict) -> str:
    """The whole demo output as plain text."""
    v, rk = r["valuation"], r["risk"]
    lines = [f"Portfolio risk engine - {'live Yahoo closes' if r['live'] else 'synthetic data'}"
             f", valuation date {r['as_of'].date()}, reference currency EUR",
             _rule("Valuation"),
             f"Securities {_eur(v['securities_eur'])}   cash {_eur(v['cash_eur'])}   "
             f"total {_eur(v['total_eur'])}" + ("" if v["complete"] else "  (incomplete)")]
    lines.append(_rule("Performance (time-weighted, total return, not annualised)"))
    lines.append(f"{'window':<6} {'from':>10} {'return':>8} {'market EUR':>14} {'flow EUR':>14}")
    for p in r["periods"]:
        lines.append(f"{p['period']:<6} {p['start'].date()!s:>10} {p['return_pct']:>7.2f}% "
                     f"{_eur(p['market_eur'])} {_eur(p['flow_eur'])}")
    lines.append(_rule(f"Risk, 1 day, {rk['n_days']} sessions "
                       f"({rk['first_day'].date()} to {rk['last_day'].date()}), loss < 0"))
    lines.append(f"{'method':<11} {'level':>6} {'VaR %':>8} {'ES %':>8} {'VaR EUR':>12} "
                 f"{'ES EUR':>12}")
    for (method, level), m in rk["measures"].items():
        if method == "rank":
            continue
        lines.append(f"{method:<11} {level:>6.1%} {m['var_pct']:>8.3f} {m['es_pct']:>8.3f} "
                     f"{m['var_eur']:>12,.0f} {m['es_eur']:>12,.0f}")
    lines.append(f"Volatility {rk['volatility_annualized_pct']:.2f} % a year; max drawdown "
                 f"{rk['max_drawdown_pct']:.2f} % ({rk['peak'].date()} to {rk['trough'].date()}, "
                 f"{rk['sessions_to_trough']} sessions; recovered "
                 f"{rk['recovery'].date() if rk['recovery'] is not None else 'not yet'})")
    w = r["witness"]
    lines.append(f"Witness check: {'OK' if w['ok'] else 'FAILED'} "
                 f"(max rel. diff {w['max_rel_diff']:.1e} on {w['where']})")
    lines.append(_rule("Backtest of the 99 % VaR, 250-session rolling window"))
    lines.append(f"{'method':<11} {'days':>5} {'exc.':>5} {'expected':>8} {'Kupiec p':>9} "
                 f"{'Christ. p':>9} {'last 250':>8} {'zone':>7} {'worst 250':>9}")
    for b in r["backtest"]:
        lines.append(f"{b['method']:<11} {b['days']:>5} {b['exceptions']:>5} "
                     f"{b['expected']:>8.1f} {b['kupiec_p']:>9.4f} {b['christoffersen_p']:>9.4f} "
                     f"{b['exceptions_last_250']:>8} {b['zone_last_250']:>7} "
                     f"{b['worst_250_exceptions']:>9}")
    lines.append(_rule("Stress tests (instant shocks on today's wealth)"))
    for s in r["scenarios"]:
        driver = s["contributions"].abs().idxmax()
        lines.append(f"{s['name']:<32} {_eur(s['pnl_eur'])} {s['pnl_pct']:>7.2f}%   "
                     f"main driver: {driver} {s['contributions'][driver]:,.0f}")
    for h in r["historical"]:
        if "pnl_eur" in h:
            lines.append(f"{h['name']:<32} {_eur(h['pnl_eur'])} {h['pnl_pct']:>7.2f}%   "
                         f"{h['start'].date()} to {h['end'].date()}")
    liq = r["liquidity"]
    lines.append(_rule("Liquidity stress: days to sell at 20 % / 10 % / 5 % of 10-day volume"))
    lines.append(f"{'line':<22} {'weight':>7} {'normal':>8} {'moderate':>9} {'severe':>8}")
    book = liq["lines"]
    worst = book.dropna(subset=["days_normal"]).sort_values("days_normal", ascending=False)
    for ticker, row in worst.head(5).iterrows():
        lines.append(f"{ticker:<22} {row['value_eur'] / book['value_eur'].sum():>7.1%} "
                     f"{row['days_normal']:>8.2f} {row['days_moderate']:>9.2f} "
                     f"{row['days_severe']:>8.2f}")
    s = liq["score_days"]
    lines.append(f"{'book, value-weighted':<22} {liq['coverage_pct'] / 100:>7.1%} "
                 f"{s['normal']:>8.2f} {s['moderate']:>9.2f} {s['severe']:>8.2f}")
    if "benchmark" in r and r["benchmark"].get("ok"):
        b = r["benchmark"]
        lines.append(_rule("Against the composite benchmark"))
        lines.append(f"beta {b['beta']:.2f}   alpha {b['alpha_annual_pct']:.2f} % a year   "
                     f"R2 {b['r_squared']:.2f}   n {b['n']}")
    lines.append(_rule("Exposures"))
    lines += [f"  {s}" for s in r["sentences"]]
    return "\n".join(lines)


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_distribution(r: dict, path: Path | BinaryIO) -> None:
    """Histogram of daily P&L in EUR with the VaR and ES lines."""
    rk = r["risk"]
    pnl = rk["pnl_eur"] / 1000
    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.hist(pnl, bins=90, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    marks = [(("historical", 0.99), "var_eur", ORANGE, "-", "VaR 99 % historical"),
             (("normal", 0.99), "var_eur", VIOLET, "-", "VaR 99 % normal"),
             (("historical", 0.975), "es_eur", ORANGE, "--", "ES 97.5 % historical")]
    for key, field, color, style, label in marks:
        x = rk["measures"][key][field] / 1000
        ax.axvline(x, color=color, linestyle=style, linewidth=2, label=f"{label}: {x:,.0f}k")
    ax.set_xlabel("Daily P&L of today's book, thousand EUR", color=INK_2)
    ax.set_ylabel("Sessions", color=INK_2)
    ax.set_title("Daily P&L of the current book: the normal VaR sits inside the historical one",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_backtest(r: dict, path: Path | BinaryIO) -> None:
    """Daily return against the previous day's 99 % VaR, exceptions in red, one panel per method."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, b, color in zip(axes, r["backtest"], (ORANGE, VIOLET), strict=True):
        _style(ax)
        realised, forecast = b["realised"], b["forecast"]
        hits = realised < forecast
        ax.plot(realised.index, realised, color="#b7d3f6", linewidth=0.7, label="Daily return")
        ax.plot(forecast.index, forecast, color=color, linewidth=1.6,
                label=f"VaR 99 % {b['method']}")
        ax.scatter(realised.index[hits], realised[hits], s=18, color=RED, zorder=3,
                   edgecolor=SURFACE, linewidth=0.8, label=f"Exceptions ({b['exceptions']})")
        ax.set_title(f"{b['method'].capitalize()} VaR: {b['exceptions']} exceptions for "
                     f"{b['expected']:.1f} expected, Kupiec p = {b['kupiec_p']:.3f}",
                     color=INK, fontsize=10, loc="left")
        ax.set_ylabel("Return, %", color=INK_2)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left", ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_charts(r: dict, folder: Path) -> list[Path]:
    """Write both charts and return their paths."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / "pnl_distribution.png", folder / "var_backtest.png"]
    plot_distribution(r, paths[0])
    plot_backtest(r, paths[1])
    return paths
