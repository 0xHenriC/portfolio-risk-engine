import re

import pandas as pd

from risk_engine import dashboard
from risk_engine.__main__ import main


def test_sections_come_in_the_expected_order(result):
    page = dashboard.render_dashboard(result)
    ids = re.findall(r'<section id="([a-z]+)"', page)
    assert ids == ["overview", "positions", "benchmark", "exposures", "risk", "stress", "quality"]


def test_page_is_self_contained(result):
    page = dashboard.render_dashboard(result)
    assert not re.search(r'(src|href)="https?://', page)
    assert "<link" not in page and "<script src" not in page
    assert page.count("data:image/png;base64,") == 2
    assert page.count("<svg") >= 1


def test_page_carries_the_figures_of_the_run(result):
    page = dashboard.render_dashboard(result)
    v = result["valuation"]
    assert f"{v['total_eur']:,.0f}" in page
    for ticker in result["quantities"].index:
        assert ticker in page
    assert "KNEBV.HE" in page  # the closed line stays in the register
    for p in result["periods"]:
        assert f"{p['market_eur']:,.0f}" in page and f"{p['flow_eur']:,.0f}" in page
    assert f"{result['witness']['max_rel_diff']:.1e}" in page


def test_line_chart_has_both_series_and_a_hover_layer():
    idx = pd.bdate_range("2024-01-01", periods=300)
    svg = dashboard.line_chart({"Portfolio": pd.Series(range(300), index=idx, dtype=float) + 100,
                                "Benchmark": pd.Series(100.0, index=idx)}, "test")
    assert svg.count("<polyline") == 2
    assert "data-series=" in svg and 'class="cross"' in svg


def test_cli_writes_the_dashboard(tmp_path):
    out = tmp_path / "dash.html"
    assert main(["demo", "--no-charts", "--html", "--html-out", str(out)]) == 0
    assert out.stat().st_size > 50_000
