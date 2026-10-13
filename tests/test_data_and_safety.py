import json

import pytest

from authagent.common import decide, load_policy
from authagent.index import Retriever, chart_to_docs, tokenize
from authagent.llm import PHILeakError, parse_json
from authagent.mock_llm import MockLLM
from authagent.redact import find_leaks, redact_chart


def test_gold_labels_are_consistent(patients):
    assert len(patients) == 45
    for p in patients:
        gold = p["gold"]
        assert gold["decision"] == decide(g["status"] for g in gold["criteria"].values())
        docs = {d["id"] for d in chart_to_docs(redact_chart(p["chart"]))}
        for cid, g in gold["criteria"].items():
            assert set(g["evidence_ids"]) <= docs, (p["chart"]["patient_id"], cid)
            if g["status"] in ("met", "unmet"):
                assert g["evidence_ids"], (p["chart"]["patient_id"], cid)


def test_all_decisions_and_hard_cases_present(patients):
    decisions = {p["gold"]["decision"] for p in patients}
    assert decisions == {"approve", "pend", "deny"}
    hard = sum(g["hard"] for p in patients for g in p["gold"]["criteria"].values())
    assert hard >= 40


def test_no_identifiers_survive_redaction(patients):
    for p in patients:
        red = redact_chart(p["chart"])
        assert "phi" not in red
        assert find_leaks(json.dumps(red), p["chart"]["phi"]) == []


def test_llm_guard_refuses_identifiers(patients):
    phi = patients[0]["chart"]["phi"]
    with pytest.raises(PHILeakError):
        MockLLM().chat([{"role": "user", "content": f"Patient {phi['name']} needs review"}], purpose="judge", phi=phi)


def test_tokenizer_splits_hyphenated_terms():
    assert {"quantiferon-tb", "quantiferon", "tb"} <= set(tokenize("QuantiFERON-TB Gold"))


def test_retrieval_finds_structured_rows(patients):
    p = patients[0]
    ret = Retriever(chart_to_docs(redact_chart(p["chart"])), mode="bm25")
    assert ret.search("demographics age", k=1)[0]["id"] == "DEMO"


def test_parse_json_tolerates_fences_and_escaped_quotes():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('Sure! {"tool": "finish", "args": {}} done') == {"tool": "finish", "args": {}}
    assert parse_json('{"thought": "say \\"hi\\" {x}", "tool": "finish"}')["tool"] == "finish"
