import ast
from pathlib import Path

from risk_engine import demo
from witness import risk_witness

WITNESS_DIR = Path(__file__).resolve().parents[1] / "witness"


def test_engine_and_witness_agree_within_1e9(result):
    w = result["witness"]
    assert w["ok"], w
    assert w["max_rel_diff"] <= 1e-9


def test_every_witness_figure_is_compared(market, result):
    as_of = result["as_of"]
    w = risk_witness.independent_calculation(market.prices, market.fx_to_eur,
                                             market.securities, result["quantities"], as_of)
    assert set(w) <= set(demo.engine_figures(result["risk"]))
    assert len(w) == 4 + 3 * 6


def test_witness_imports_nothing_from_the_engine():
    for path in WITNESS_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(n.startswith("risk_engine") for n in names), path.name


def test_witness_detects_a_wrong_figure(result):
    figures = demo.engine_figures(result["risk"])
    tampered = dict(figures, historical_var_99=figures["historical_var_99"] * 1.001)
    gap, where = risk_witness.max_relative_difference(tampered, figures)
    assert where == "historical_var_99" and gap > 1e-9
