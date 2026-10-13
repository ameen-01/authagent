from authagent.common import decide, load_policy
from authagent.rules import evaluate


def crit(policy_id, cid):
    return next(c for c in load_policy(policy_id)["criteria"] if c["id"] == cid)


def test_lab_threshold_is_inclusive_and_uses_most_recent():
    c = crit("POL-GLP1", "C3")
    assert evaluate(c, {"value": 7.0, "date": "2026-09-20"})["status"] == "met"
    assert evaluate(c, {"value": 6.9, "date": "2026-09-20"})["status"] == "unmet"
    older_high_newer_low = {"measurements": [{"value": 9.1, "date": "2026-03-01"}, {"value": 6.4, "date": "2026-09-01"}]}
    assert evaluate(c, older_high_newer_low)["status"] == "unmet"


def test_stale_lab_is_missing_not_met():
    c = crit("POL-GLP1", "C3")
    res = evaluate(c, {"value": 9.5, "date": "2026-05-01"})
    assert res["status"] == "missing" and res["computed"]["age_days"] > 90


def test_categorical_lab():
    c = crit("POL-RA", "C5")
    assert evaluate(c, {"name": "QuantiFERON-TB Gold", "value": "negative", "date": "2026-06-01"})["status"] == "met"
    assert evaluate(c, {"name": "QuantiFERON-TB Gold", "value": "Positive", "date": "2026-06-01"})["status"] == "unmet"
    assert evaluate(c, {"name": "LDL-C", "value": "negative", "date": "2026-06-01"})["status"] == "no_match"


def test_med_duration_boundaries_and_wrong_drug():
    c = crit("POL-RA", "C3")
    assert evaluate(c, {"drug": "methotrexate", "start": "2026-01-01", "end": "2026-04-01"})["status"] == "met"  # 90 days
    assert evaluate(c, {"drug": "methotrexate", "start": "2026-01-02", "end": "2026-04-01"})["status"] == "unmet"  # 89
    assert evaluate(c, {"drug": "methotrexate", "start": "2025-01-01", "end": None})["status"] == "met"  # ongoing
    assert evaluate(c, {"drug": "hydroxychloroquine", "start": "2025-01-01", "end": None})["status"] == "no_match"


def test_age_and_diagnosis():
    assert evaluate(crit("POL-RA", "C2"), {"age": 18})["status"] == "met"
    assert evaluate(crit("POL-RA", "C2"), {"age": 17})["status"] == "unmet"
    assert evaluate(crit("POL-PCSK9", "C1"), {"code": "E78.01"})["status"] == "met"
    assert evaluate(crit("POL-PCSK9", "C1"), {"code": "E78.5"})["status"] == "no_match"


def test_bad_input_is_an_error_not_a_guess():
    assert evaluate(crit("POL-RA", "C2"), {"years": 40})["status"] == "error"
    assert evaluate(crit("POL-RA", "C4"), {})["status"] == "error"  # documentation -> verifier


def test_decision_logic():
    assert decide(["met", "met"]) == "approve"
    assert decide(["met", "missing"]) == "pend"
    assert decide(["missing", "unmet"]) == "deny"
