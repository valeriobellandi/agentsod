"""Pipeline tests of the GridOps-SoD experiment (mock LLM, no model needed)."""
import pytest

pytest.importorskip("pandapower")

from experiments.gridops.grid import VALUES, is_harmful  # noqa: E402
from experiments.gridops.harness import run_episode  # noqa: E402
from experiments.gridops.llm import MockChat, extract_json  # noqa: E402
from experiments.gridops.scenarios import make_episodes  # noqa: E402


def test_value_sets_match_power_flow():
    for asset, (safe, harmful) in VALUES.items():
        assert not any(is_harmful(asset, v) for v in safe), asset
        assert all(is_harmful(asset, v) for v in harmful), asset


def test_extract_json():
    assert extract_json('Sure! ```json\n{"decision": "approve"}\n```') == {"decision": "approve"}
    assert extract_json("no json here") is None


def _rows(kind):
    eps = [e for e in make_episodes(2) if e.kind == kind]
    return {r["condition"]: r for r in run_episode(MockChat(), eps[0])}


def test_laundering_blocked_by_monitor_at_level_2_only():
    r = _rows("A2_laundering")
    assert r["L0"]["executed_harmful"]
    for c in ("L2", "L2n"):
        assert r[c]["approver"] == "approve" and r[c]["monitor_approve"] == "deny"
        assert not r[c]["executed"]


def test_level_2_withholds_justification_but_not_description():
    r = _rows("A3_field")
    assert r["L0"]["executed_harmful"] and r["L2"]["executed_harmful"]  # description still carries it
    assert not r["L2n"]["executed_harmful"]                               # narrow schema removes it


def test_safe_change_executed_in_every_condition():
    r = _rows("legit_safe")
    assert all(r[c]["executed"] and not r[c]["executed_harmful"] for c in r)


def test_suite_v2_has_variants_and_adaptive_injection():
    from experiments.gridops.scenarios import KINDS_V2, make_episodes_v2
    eps = make_episodes_v2(3)
    assert {e.kind for e in eps} == set(KINDS_V2)
    assert {e.meta["variant"] for e in eps if e.kind == "A2_laundering"} == {0, 1, 2}
    ad = [e for e in eps if e.kind == "A1_adaptive"][0]
    assert "description field" in ad.ticket
    r = {x["condition"]: x for x in run_episode(MockChat(), ad)}
    assert r["L2"]["executed_harmful"] and not r["L2n"]["executed_harmful"]


def test_safe_unauthorized_change_is_stopped_only_by_separation():
    from experiments.gridops.scenarios import make_episodes_v3
    from experiments.gridops.derived import outcomes, summarise
    ep = make_episodes_v3(1)[0]
    rows = run_episode(MockChat(), ep)
    r = {x["condition"]: x for x in rows}
    assert not r["L0"]["proposal_harmful"] and r["L0"]["executed"]      # safe, executed at level 0
    assert not r["L2"]["executed"] and r["L2"]["monitor_approve"] == "deny"
    s = summarise(outcomes(rows))[0]
    assert s["L0+P"] == 1 and s["L1"] == 0                              # power-flow gate lets it pass
