"""Command line.

    python -m risk_engine demo [--live] [--seed N] [--no-charts] [--html [--html-out PATH]]
    python -m risk_engine seeds [--n 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from risk_engine import demo, report
from risk_engine.data import synthetic


def _seeds(n: int) -> int:
    study = demo.seed_study(range(1, n + 1))
    summary = study.groupby("method").agg(
        mean_exceptions=("exceptions", "mean"), expected=("expected", "first"),
        kupiec_rejections_5pct=("kupiec_p", lambda p: int((p < 0.05).sum())))
    print(f"99 % VaR backtest over {n} synthetic histories")
    print(summary.round(2).to_string())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="risk_engine")
    parser.add_argument("command", choices=["demo", "seeds"])
    parser.add_argument("--live", action="store_true", help="Yahoo closes instead of synthetic")
    parser.add_argument("--refresh", action="store_true", help="re-download the live cache")
    parser.add_argument("--seed", type=int, default=synthetic.DEFAULT_SEED)
    parser.add_argument("--n", type=int, default=30, help="number of seeds for 'seeds'")
    parser.add_argument("--no-charts", action="store_true")
    parser.add_argument("--charts-dir", type=Path, default=Path("docs/img"))
    parser.add_argument("--html", action="store_true", help="also write the HTML dashboard")
    parser.add_argument("--html-out", type=Path, default=Path("docs/dashboard.html"))
    args = parser.parse_args(argv)

    if args.command == "seeds":
        return _seeds(args.n)
    if args.live:
        from risk_engine.data import live

        market = live.load_market(refresh=args.refresh)
    else:
        market = synthetic.generate_market(args.seed)
    result = demo.run(market, live=args.live)
    print(report.format_report(result))
    if not args.no_charts:
        for path in report.save_charts(result, args.charts_dir):
            print(f"chart written: {path.as_posix()}")
    if args.html:
        from risk_engine import dashboard

        path = dashboard.write_dashboard(result, args.html_out)
        print(f"dashboard written: {path.as_posix()}")
    return 0 if result["witness"]["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
