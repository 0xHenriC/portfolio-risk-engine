"""Self-contained HTML dashboard of one demo run: monitoring first, then risk and controls.

One file, no external resource: charts are inline SVG or embedded PNG, styles and the
small hover script are inline, so the page reads offline and can be attached as is.
"""

from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from risk_engine import report

SECTIONS = [("overview", "Overview"), ("positions", "Positions and register"),
            ("benchmark", "Portfolio against its benchmark"), ("exposures", "Exposures"),
            ("risk", "Market risk"), ("stress", "Backtest and stress tests"),
            ("quality", "Data quality controls")]

CSS = """
:root { color-scheme: light; --bg:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7; --s1:#2a78d6; --s2:#eb6834;
  --band:#f3f3f0; --good:#006300; --warn:#8a5a00; --bad:#b42323; }
@media (prefers-color-scheme: dark) { :root { color-scheme: dark; --bg:#0d0d0d; --card:#1a1a19;
  --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --s1:#3987e5; --s2:#d95926; --band:#222220; --good:#0ca30c; --warn:#fab219; --bad:#e66767; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:1120px; margin:0 auto; padding:24px 16px 48px; }
h1 { font-size:26px; margin:0 0 2px; } h2 { font-size:19px; margin:0 0 12px; }
h3 { font-size:15px; margin:18px 0 8px; color:var(--ink2); }
.sub { color:var(--ink2); margin:0 0 16px; }
nav { display:flex; flex-wrap:wrap; gap:6px 14px; margin:0 0 20px; font-size:14px; }
nav a { color:var(--s1); text-decoration:none; }
section { background:var(--card); border:1px solid var(--grid); border-radius:10px;
  padding:20px; margin:0 0 18px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px;
  margin:0 0 16px; }
.tile { border:1px solid var(--grid); border-radius:8px; padding:10px 12px; }
.tile .k { color:var(--ink2); font-size:13px; } .tile .v { font-size:22px; font-weight:600; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th, td { padding:6px 10px; border-bottom:1px solid var(--grid); text-align:left;
  white-space:nowrap; }
th { color:var(--ink2); font-weight:600; } tbody tr:nth-child(2n) { background:var(--band); }
td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; }
.pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12.5px;
  font-weight:600; border:1px solid currentColor; }
.ok { color:var(--good); } .info { color:var(--warn); } .alert { color:var(--bad); }
.grid2 { display:grid; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); gap:18px; }
figure { margin:8px 0 0; } figcaption { color:var(--ink2); font-size:13px; margin-top:4px; }
img.chart { width:100%; height:auto; background:#fcfcfb; border-radius:6px; }
svg text { fill:var(--ink2); font-size:12px; }
.bars { display:grid; grid-template-columns:max-content 1fr; gap:6px 10px; align-items:center;
  font-size:13px; }
.bl { color:var(--ink2); text-align:right; }
.bt { display:flex; align-items:center; gap:6px; min-width:0; }
.bt i { display:block; height:14px; min-width:2px; background:var(--s1);
  border-radius:0 4px 4px 0; }
.bt span { color:var(--ink2); font-variant-numeric:tabular-nums; white-space:nowrap; }
.legend { display:flex; gap:16px; font-size:13px; color:var(--ink2); margin:0 0 4px; }
.sw { display:inline-block; width:14px; height:3px; vertical-align:middle; margin-right:6px; }
#tip { position:fixed; pointer-events:none; background:var(--card); border:1px solid var(--grid);
  border-radius:6px; padding:6px 8px; font-size:12.5px; display:none;
  box-shadow:0 2px 8px rgba(0,0,0,.12); }
ul.notes { margin:8px 0 0; padding-left:20px; color:var(--ink2); }
footer { color:var(--muted); font-size:12.5px; text-align:center; }
"""

HOVER_JS = """
document.querySelectorAll('svg[data-series]').forEach(function (svg) {
  var d = JSON.parse(svg.dataset.series), tip = document.getElementById('tip');
  var line = svg.querySelector('.cross'), box = svg.getBoundingClientRect.bind(svg);
  var x0 = +svg.dataset.x0, x1 = +svg.dataset.x1, w = +svg.dataset.w;
  svg.addEventListener('mousemove', function (e) {
    var r = box(), x = (e.clientX - r.left) * w / r.width;
    var i = Math.round((x - x0) / (x1 - x0) * (d.dates.length - 1));
    if (i < 0 || i >= d.dates.length) { tip.style.display = 'none'; return; }
    var px = x0 + i / (d.dates.length - 1) * (x1 - x0);
    line.setAttribute('x1', px); line.setAttribute('x2', px); line.style.display = '';
    tip.innerHTML = '<b>' + d.dates[i] + '</b>' + d.names.map(function (n, k) {
      return '<br>' + n + ': ' + d.values[k][i].toFixed(1); }).join('');
    tip.style.display = 'block';
    tip.style.left = (e.clientX + 14) + 'px'; tip.style.top = (e.clientY + 14) + 'px';
  });
  svg.addEventListener('mouseleave', function () {
    tip.style.display = 'none'; line.style.display = 'none'; });
});
"""


def _e(x: object) -> str:
    return html.escape(str(x))


def _eur(x: float | None) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x:,.0f}"


def _pct(x: float | None, signed: bool = True, digits: int = 2) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:+.{digits}f} %" if signed else f"{x:.{digits}f} %"


def _table(headers: list[str], rows: list[list[object]], numeric: set[int]) -> str:
    """Plain table; numeric columns right-aligned with tabular figures."""
    head = "".join(f'<th class="{"n" if i in numeric else ""}">{_e(h)}</th>'
                   for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(
        f'<td class="{"n" if i in numeric else ""}">{c}</td>' for i, c in enumerate(row))
        + "</tr>" for row in rows)
    return (f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


def _tiles(items: list[tuple[str, str]]) -> str:
    return '<div class="tiles">' + "".join(
        f'<div class="tile"><div class="k">{_e(k)}</div><div class="v">{v}</div></div>'
        for k, v in items) + "</div>"


def _pill(level: str) -> str:
    label = {"ok": "OK", "info": "Info", "alert": "Alert"}[level]
    return f'<span class="pill {level}">{label}</span>'


def _png(plot, r: dict) -> str:
    buffer = io.BytesIO()
    plot(r, buffer)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def line_chart(series: dict[str, pd.Series], label: str) -> str:
    """Rebased lines on one axis, direct end labels, a legend and a hover crosshair."""
    frame = pd.DataFrame(series).dropna()
    w, h, left, right, top, bottom = 880, 300, 44, 120, 12, 26
    lo, hi = float(frame.min().min()), float(frame.max().max())
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad
    x_of = np.linspace(left, w - right, len(frame))

    def y_of(v: np.ndarray) -> np.ndarray:
        return top + (hi - np.asarray(v)) / (hi - lo) * (h - top - bottom)

    parts = []
    raw_step = (hi - lo) / 4
    magnitude = 10 ** np.floor(np.log10(raw_step))
    step = magnitude * min((1, 2, 5, 10), key=lambda m: abs(m * magnitude - raw_step))
    for tick in np.arange(np.ceil(lo / step) * step, hi, step):
        y = y_of(tick)
        parts.append(f'<line x1="{left}" x2="{w - right}" y1="{y:.1f}" y2="{y:.1f}" '
                     f'stroke="var(--grid)"/><text x="{left - 6}" y="{y + 4:.1f}" '
                     f'text-anchor="end">{tick:.0f}</text>')
    years = frame.index.year
    for k in np.flatnonzero(np.r_[True, years[1:] != years[:-1]])[1:]:
        parts.append(f'<text x="{x_of[k]:.1f}" y="{h - 6}" text-anchor="middle">{years[k]}</text>')
    parts.append(f'<line x1="{left}" x2="{w - right}" y1="{h - bottom}" y2="{h - bottom}" '
                 'stroke="var(--axis)"/>')
    legend = []
    ends = y_of(frame.iloc[-1].to_numpy())
    if len(ends) == 2 and abs(ends[0] - ends[1]) < 16:
        mid, sign = ends.mean(), 1 if ends[0] >= ends[1] else -1
        ends = np.array([mid + 8 * sign, mid - 8 * sign])
    for n, (name, color) in enumerate(zip(frame.columns, ("--s1", "--s2"), strict=False)):
        ys = y_of(frame[name].to_numpy())
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(x_of, ys, strict=True))
        parts.append(f'<polyline fill="none" stroke="var({color})" stroke-width="2" '
                     f'points="{points}"/>')
        parts.append(f'<text x="{w - right + 6}" y="{ends[n] + 4:.1f}">{_e(name)} '
                     f'{frame[name].iloc[-1]:.1f}</text>')
        legend.append(f'<span><i class="sw" style="background:var({color})"></i>{_e(name)}</span>')
    parts.append(f'<line class="cross" y1="{top}" y2="{h - bottom}" stroke="var(--muted)" '
                 'style="display:none"/>')
    data = {"dates": [d.strftime("%Y-%m-%d") for d in frame.index], "names": list(frame.columns),
            "values": [np.round(frame[c].to_numpy(), 2).tolist() for c in frame.columns]}
    return (f'<div class="legend">{"".join(legend)}</div>'
            f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{_e(label)}" '
            f'data-x0="{left}" data-x1="{w - right}" data-w="{w}" '
            f"data-series='{_e(json.dumps(data))}'>{''.join(parts)}</svg>")


def bar_chart(weights: pd.Series, label: str) -> str:
    """Horizontal bars for shares of a whole, value written at the end of each bar."""
    weights = weights[weights > 0]
    top = float(weights.max())
    rows = "".join(
        f'<div class="bl">{_e(name)}</div><div class="bt" title="{_e(name)}: {share:.1%}">'
        f'<i style="width:{share / top * 100:.1f}%"></i><span>{share:.1%}</span></div>'
        for name, share in weights.items())
    return f'<div class="bars" role="img" aria-label="{_e(label)}">{rows}</div>'


def _overview(r: dict) -> str:
    v, rk = r["valuation"], r["risk"]
    itd = next(p for p in r["periods"] if p["period"] == "ITD")
    tiles = _tiles([("Total wealth, EUR", _eur(v["total_eur"])),
                    ("Securities, EUR", _eur(v["securities_eur"])),
                    ("Cash, EUR", _eur(v["cash_eur"])),
                    ("Return since inception", _pct(itd["return_pct"])),
                    ("Positions", str(len(v["lines"]))),
                    ("Volatility, annualised", _pct(rk["volatility_annualized_pct"], False))])
    names = {"1W": "1 week", "1M": "1 month", "YTD": "Year to date", "1Y": "1 year",
             "ITD": "Since inception"}
    rows = [[names[p["period"]], p["start"].date(), p["end"].date(), _pct(p["return_pct"]),
             _eur(p["market_eur"]), _eur(p["flow_eur"]),
             _eur(p["value_end_eur"] - p["value_start_eur"]),
             _eur(p["realized"]["gains_eur"] + p["realized"]["losses_eur"])] for p in r["periods"]]
    table = _table(["Window", "From", "To", "Return", "Market effect", "Flow effect",
                    "Change in value", "Realised P&L"], rows, {3, 4, 5, 6, 7})
    note = ("<ul class='notes'><li>Return: time-weighted, chained daily, gross dividends "
            "included, not annualised.</li><li>Market effect + flow effect = change in value, "
            "checked to the cent; amounts in EUR.</li></ul>")
    status = "complete" if v["complete"] else "incomplete: " + ", ".join(v["missing_price"])
    return (f"{tiles}{table}{note}<p class='sub'>Valuation total {status}; "
            f"{int(v['lines']['stale'].sum())} line(s) valued at an earlier close.</p>")


def _positions(r: dict) -> str:
    v, reg, secs = r["valuation"]["lines"], r["register"], r["securities"]
    cost = r["book"].open_cost_eur
    total = v["value_eur"].sum()
    rows = []
    for t, line in v.sort_values("value_eur", ascending=False).iterrows():
        unreal = line["value_eur"] - cost[t]
        rows.append([f"{_e(t)}<br><small>{_e(secs.at[t, 'name'])}</small>",
                     f"{line['quantity']:,.0f}",
                     f"{r['local_close'][t]:,.2f} {secs.at[t, 'currency']}",
                     _eur(line["value_eur"]), _pct(line["value_eur"] / total * 100, False, 1),
                     f"{cost[t] / line['quantity']:,.2f}", _eur(unreal),
                     _pct(unreal / cost[t] * 100)])
    held = _table(["Line", "Quantity", "Last close, per quote unit", "Value EUR", "Weight",
                   "Avg cost EUR, per held unit",
                   "Unrealised EUR", "Unrealised"], rows, {1, 2, 3, 4, 5, 6, 7})
    reg_rows = [[_e(t), row["state"], row["days_held"], _eur(row["invested_eur"]),
                 _eur(row["proceeds_eur"]), _eur(row["remaining_eur"]), _eur(row["dividends_eur"]),
                 _pct(row["local_return_pct"]), _pct(row["total_return_pct"]),
                 _pct(row["fx_effect_pct"]), _eur(row["realized_eur"])]
                for t, row in reg.iterrows()]
    register = _table(["Line", "State", "Days held", "Invested", "Proceeds", "Remaining",
                       "Dividends", "Local return", "EUR return", "Currency effect",
                       "Realised"], reg_rows, set(range(2, 11)))
    return ("<h3>Positions held (weights on securities, cash excluded)</h3>" + held
            + "<h3>Register: every line ever held, over its whole life (EUR unless stated)</h3>"
            + register + "<ul class='notes'><li>Cost basis: weighted average cost per holding "
            "cycle.</li><li>Currency effect = (1 + EUR return) / (1 + local return) - 1."
            "</li></ul>")


def _benchmark(r: dict) -> str:
    b = r.get("benchmark")
    if not b or not b.get("ok"):
        tr = r["history"]["total_return_index"]
        tr = tr[tr.index.weekday < 5]
        return (line_chart({"Portfolio": tr / tr.iloc[0] * 100}, "Portfolio index")
                + "<p class='sub'>No benchmark in this mode.</p>")
    chart = line_chart({"Portfolio": b["portfolio_index"], "Benchmark": b["benchmark_index"]},
                       "Portfolio and benchmark, rebased to 100")
    weights = ", ".join(f"{k} {w:.0%}" for k, w in b["weights"].items())
    tiles = _tiles([("Beta", f"{b['beta']:.2f}"), ("Alpha, a year", _pct(b["alpha_annual_pct"])),
                    ("R squared", f"{b['r_squared']:.2f}"),
                    ("Tracking error", _pct(b["tracking_error_pct"], False)),
                    ("Observations", str(b["n"]))])
    return (f"{chart}{tiles}<p class='sub'>Composite benchmark weighted like the currency "
            f"exposure of the securities: {_e(weights)}. Total return on both sides; the "
            f"gross-against-net dividend bias is {b['withholding_bias_band_pct'][0]:.2f} to "
            f"{b['withholding_bias_band_pct'][1]:.2f} points a year in favour of the book.</p>")


def _exposures(r: dict) -> str:
    exp = r["exposures"]
    blocks = [("Currency, share of wealth", exp["currency"]),
              ("Sector, share of equities", exp["sector"]),
              ("Geographic zone, share of wealth", exp["zone"])]
    charts = "".join(f"<figure><h3>{_e(t)}</h3>{bar_chart(s, t)}</figure>" for t, s in blocks)
    notes = "".join(f"<li>{_e(s)}</li>" for s in r["sentences"])
    return f"<div class='grid2'>{charts}</div><ul class='notes'>{notes}</ul>"


def _risk(r: dict) -> str:
    rk = r["risk"]
    m = rk["measures"]
    tiles = _tiles([("VaR 95 % historical", _pct(m[("historical", 0.95)]["var_pct"])),
                    ("ES 97.5 % historical", _pct(m[("historical", 0.975)]["es_pct"])),
                    ("Volatility, annualised", _pct(rk["volatility_annualized_pct"], False)),
                    ("Maximum drawdown", _pct(rk["max_drawdown_pct"])),
                    ("Witness check", _pill("ok" if r["witness"]["ok"] else "alert"))])
    rows = [[method, f"{level:.1%}", _pct(v["var_pct"]), _pct(v["es_pct"]), _eur(v["var_eur"]),
             _eur(v["es_eur"])] for (method, level), v in m.items()]
    table = _table(["Method", "Level", "VaR", "ES", "VaR EUR", "ES EUR"], rows, {2, 3, 4, 5})
    recovery = rk["recovery"].date() if rk["recovery"] is not None else "not yet"
    return (f"{tiles}{table}<ul class='notes'><li>One day, {rk['n_days']} sessions from "
            f"{rk['first_day'].date()} to {rk['last_day'].date()}; today's positions revalued "
            f"on each past day; a loss is negative.</li><li>Drawdown: peak {rk['peak'].date()}, "
            f"trough {rk['trough'].date()}, recovered {recovery}.</li><li>Witness: max relative "
            f"difference {r['witness']['max_rel_diff']:.1e} with an independent implementation."
            f"</li></ul><figure><img class='chart' alt='Distribution of daily P&amp;L' "
            f"src='{_png(report.plot_distribution, r)}'></figure>")


def _stress(r: dict) -> str:
    bt = [[b["method"], b["days"], b["exceptions"], f"{b['expected']:.1f}",
           f"{b['kupiec_p']:.3f}", f"{b['christoffersen_p']:.3f}",
           f"{b['exceptions_last_250']} ({b['zone_last_250']})", b["worst_250_exceptions"]]
          for b in r["backtest"]]
    backtest = _table(["99 % VaR", "Days", "Exceptions", "Expected", "Kupiec p",
                       "Christoffersen p", "Last 250", "Worst 250"], bt, set(range(1, 8)))
    shocks = [[_e(s["name"]), _eur(s["pnl_eur"]), _pct(s["pnl_pct"]),
               _e(s["contributions"].abs().idxmax())] for s in r["scenarios"]]
    shocks += [[_e(h["name"]), _eur(h["pnl_eur"]), _pct(h["pnl_pct"]),
                f"{h['start'].date()} to {h['end'].date()}"]
               for h in r["historical"] if "pnl_eur" in h]
    market = _table(["Scenario", "P&L EUR", "Share of wealth", "Main driver or window"], shocks,
                    {1, 2})
    liq = r["liquidity"]
    lines = liq["lines"].dropna(subset=["days_normal"]).sort_values("days_normal",
                                                                    ascending=False)
    total = liq["lines"]["value_eur"].sum()
    rows = [[_e(t), _pct(x["value_eur"] / total * 100, False, 1), f"{x['days_normal']:.2f}",
             f"{x['days_moderate']:.2f}", f"{x['days_severe']:.2f}"]
            for t, x in lines.head(6).iterrows()]
    s = liq["score_days"]
    rows.append(["<b>Book, value-weighted</b>",
                 f"{liq['coverage_pct']:.1f} % covered",
                 f"{s['normal']:.2f}", f"{s['moderate']:.2f}", f"{s['severe']:.2f}"])
    liquidity = _table(["Line", "Weight", "20 % of volume", "10 %", "5 %"], rows, {1, 2, 3, 4})
    return (f"<h3>VaR backtest, 250-session rolling window</h3>{backtest}"
            f"<figure><img class='chart' alt='VaR backtest' "
            f"src='{_png(report.plot_backtest, r)}'></figure>"
            f"<h3>Market stress, instant shocks on today's wealth</h3>{market}"
            f"<h3>Liquidity stress: days to sell at a share of 10-day volume</h3>{liquidity}")


def _quality(r: dict) -> str:
    v, fresh, gaps = r["valuation"], r["freshness"], r["gap_shortfall"]
    worst_fresh = "alert" if (fresh["level"] == "alert").any() else (
        "info" if (fresh["level"] == "info").any() else "ok")
    checks = [
        ("Trade journal", "ok", "Sides, signs, dates and currencies valid; lots rebuilt."),
        ("Journal against declared holdings", "ok", "Running quantity equals every holding."),
        ("Period decomposition", "ok", "Market + flow = change in value, to the cent, "
                                       "on every window."),
        ("Valuation total", "ok" if v["complete"] else "alert",
         f"{len(v['missing_price'])} line(s) without a recent close; "
         f"{int(v['lines']['stale'].sum())} valued at an earlier close."),
        ("Split-like price jumps", "ok" if r["price_jumps"].empty else "alert",
         f"{len(r['price_jumps'])} close(s) at a round multiple of the previous one."),
        ("Journal prices in pre-split units", "ok" if r["raw_unit_trades"].empty else "info",
         f"{len(r['raw_unit_trades'])} trade(s) flagged."),
        ("Freshness of closes", worst_fresh,
         ", ".join(f"{n} {lvl}" for lvl, n in fresh["level"].value_counts().items())),
        ("Session gaps against exchange peers",
         "alert" if (gaps["status"] == "alert").any() else (
             "info" if (gaps["status"] == "not comparable").any() else "ok"),
         ", ".join(f"{n} {st}" for st, n in gaps["status"].value_counts().items())
         + " (an exchange needs at least five tickers to be compared)"),
        ("Independent witness", "ok" if r["witness"]["ok"] else "alert",
         f"Max relative difference {r['witness']['max_rel_diff']:.1e} (tolerance 1e-9)."),
    ]
    return _table(["Control", "Status", "Detail"],
                  [[_e(n), _pill(lvl), _e(d)] for n, lvl, d in checks], set())


def render_dashboard(r: dict) -> str:
    """The whole page as one HTML string."""
    mode = "live closes" if r["live"] else "synthetic data"
    origin = ("Yahoo closes, not redistributed" if r["live"] else
              "Synthetic prices; listed names are labels only and FIC lines are fictitious")
    blocks = {"overview": _overview, "positions": _positions, "benchmark": _benchmark,
              "exposures": _exposures, "risk": _risk, "stress": _stress, "quality": _quality}
    nav = "".join(f'<a href="#{k}">{_e(t)}</a>' for k, t in SECTIONS)
    body = "".join(f'<section id="{k}"><h2>{_e(t)}</h2>{blocks[k](r)}</section>'
                   for k, t in SECTIONS)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>Portfolio dashboard</title><style>{CSS}</style></head><body><main>"
            f"<h1>Portfolio dashboard</h1><p class='sub'>{mode} · valuation date "
            f"{r['as_of'].date()} · reference currency EUR</p><nav>{nav}</nav>{body}"
            f"<footer>Generated by <code>python -m risk_engine demo --html</code>. "
            f"{origin}.</footer>"
            f"</main><div id='tip'></div><script>{HOVER_JS}</script></body></html>")


def write_dashboard(r: dict, path: Path) -> Path:
    """Write the dashboard file and return its path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard(r), encoding="utf-8")
    return path
