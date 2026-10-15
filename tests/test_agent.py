import re
from authagent.agent import run_agent
from authagent.common import load_policy
from authagent.evaluate import case_metrics, mcnemar, report
from authagent.mock_llm import MockLLM
from authagent.runner import run_config
from authagent.tools import CaseEnv
from authagent.verifier import LexicalVerifier, LLMJudge


def env_for(p):
    return CaseEnv(p["chart"], load_policy(p["chart"]["policy_id"]), LexicalVerifier(), retrieval_mode="bm25")


def test_decisions_must_be_grounded(patients):
    env = env_for(patients[0])
    res = env.record_decision("C1", "met", ["DX-1"], "trust me")
    assert "error" in res and "not grounded" in res["error"]
    res = env.record_decision("C4", "missing", [], "")
    assert "error" in res and "search at least twice" in res["error"]
    assert "error" in env.record_decision("C1", "met", ["N9.s99"])  # invented evidence id


def test_grounded_decision_is_accepted(patients):
    p = patients[0]
    env = env_for(p)
    env.search_chart("demographics age", criterion_id="C2")
    rule = env.evaluate_rule("C2", {"age": env.chart["demographics"]["age"]}, ["DEMO"])
    assert rule["status"] == "met"
    assert env.record_decision("C2", "met", ["DEMO"], "age")["recorded"] == "C2"


def test_missing_allowed_after_two_searches(patients):
    env = env_for(patients[0])
    env.search_chart("tuberculosis screening", criterion_id="C5")
    env.search_chart("TB test result", criterion_id="C5")
    assert env.record_decision("C5", "missing", [], "")["status"] == "missing"


def test_finish_requires_every_criterion(patients):
    env = env_for(patients[0])
    assert "error" in env.finish()


def test_agent_loop_finishes_on_every_case(patients):
    for p in patients[:9]:
        llm = MockLLM()
        res = run_agent(p["chart"], load_policy(p["chart"]["policy_id"]), llm, LLMJudge(llm), retrieval_mode="bm25")
        assert res["stop_reason"] == "finished"
        assert set(res["statuses"]) == {"C1", "C2", "C3", "C4", "C5"}
        assert res["draft"].splitlines()[0] in ("PRIOR AUTHORIZATION REQUEST (draft)",
                                                "REQUEST FOR MISSING INFORMATION (do not submit yet)",
                                                "LIKELY-DENIAL REPORT (criteria not met; do not submit)")


def test_unparseable_reply_is_retried(patients):
    class Flaky(MockLLM):
        calls = 0

        def chat(self, messages, purpose="chat", phi=None):
            Flaky.calls += 1
            r = super().chat(messages, purpose, phi)
            if Flaky.calls == 2:
                r.text = "I think I should search now."
            return r

    p = patients[0]
    llm = Flaky()
    res = run_agent(p["chart"], load_policy(p["chart"]["policy_id"]), llm, LLMJudge(llm), retrieval_mode="bm25")
    assert res["parse_failures"] == 1 and res["stop_reason"] == "finished"


def test_runner_report_and_stats(patients, results_dir):
    recs = patients[:6]
    for cfg in ("single_prompt", "agent"):
        rows = run_config(cfg, MockLLM(), recs, retrieval_mode="bm25", resume=False, verbose=False)
        assert len(rows) == 6 and not any(r.get("error") for r in rows)
    m = case_metrics(rows)
    assert 0 <= m["decision_accuracy"] <= 1 and m["llm_calls_per_case"] > 1
    out = report(["single_prompt", "agent"])
    assert "Decision acc." in out["markdown"] and (results_dir / "report.md").exists()


def test_mcnemar_exact():
    a = [{"patient_id": str(i), "decision": "approve", "gold_decision": "approve"} for i in range(10)]
    b = [{"patient_id": str(i), "decision": "deny" if i < 8 else "approve", "gold_decision": "approve"} for i in range(10)]
    t = mcnemar(a, b)
    assert t["a_only_correct"] == 8 and t["b_only_correct"] == 0
    assert abs(t["p_value"] - 2 / 2 ** 8) < 1e-12


def test_context_compaction_drops_finished_criteria(patients):
    sent = []

    class Spy(MockLLM):
        def chat(self, messages, purpose="chat", phi=None):
            if purpose == "agent":
                sent.append([m["content"] for m in messages])
            return super().chat(messages, purpose, phi)

    p = patients[0]
    llm = Spy()
    res = run_agent(p["chart"], load_policy(p["chart"]["policy_id"]), llm, LLMJudge(llm), retrieval_mode="bm25")
    assert res["stop_reason"] == "finished"
    first_search = next(s for s in res["trace"] if s.get("tool") == "search_chart")["args"]["query"]
    late = sent[-1]
    assert not any(f'"query": "{first_search}"' in m for m in late[4:])  # old work is gone
    assert "decided_so_far" in late[5]  # ...but summarised
    uncompacted = 2 + 2 * res["steps"]  # system + task + one exchange per step
    assert max(len(m) for m in sent) < uncompacted / 2


def test_nested_rule_values_are_unwrapped(patients):
    p = next(x for x in patients if x["chart"]["policy_id"] == "POL-RA")
    env = env_for(p)
    ev = next(d for d in env.docs if "QuantiFERON" in d["text"] and re.search(r"\d{4}-\d{2}-\d{2}", d["text"]))
    date = re.search(r"\d{4}-\d{2}-\d{2}", ev["text"]).group(0)
    value = "negative" if "negative" in ev["text"] else "positive"
    res = env.evaluate_rule("C5", {"lab": {"name": "QuantiFERON-TB Gold",
                                           "measurements": [{"value": value, "date": date}]}}, [ev["id"]])
    assert res["status"] in ("met", "unmet", "missing") and "error" not in res


def test_rule_values_must_appear_in_cited_evidence(patients):
    env = env_for(patients[0])
    real_age = env.chart["demographics"]["age"]
    assert "error" in env.evaluate_rule("C2", {"age": real_age})  # no evidence cited
    res = env.evaluate_rule("C2", {"age": 54 if real_age != 54 else 55}, ["DEMO"])  # value not in DEMO
    assert "do not appear" in res["error"]
    assert env.evaluate_rule("C2", {"age": real_age}, ["DEMO"])["status"] == "met"


def test_runner_guards_models_and_stops_on_daily_limit(patients, results_dir):
    import pytest

    recs = patients[:4]
    run_config("single_prompt", MockLLM(), recs, retrieval_mode="bm25", resume=False, verbose=False)

    class OtherModel(MockLLM):
        name = "other-model"

    with pytest.raises(SystemExit, match="--tag"):
        run_config("single_prompt", OtherModel(), recs, retrieval_mode="bm25", verbose=False)

    class DailyLimit(MockLLM):
        calls = 0

        def chat(self, messages, purpose="chat", phi=None):
            DailyLimit.calls += 1
            if DailyLimit.calls > 1:
                raise RuntimeError("Error code: 429 - Rate limit reached on tokens per day (TPD)")
            return super().chat(messages, purpose, phi)

    rows = run_config("single_prompt", DailyLimit(), patients[4:8], retrieval_mode="bm25", verbose=False)
    assert len(rows) == 1 and not rows[0].get("error")  # stopped after the first case, nothing recorded as error
    from authagent.common import read_jsonl

    kept = read_jsonl(results_dir / "runs" / "single_prompt.jsonl")
    assert len(kept) == 5  # 4 earlier rows + 1 new; earlier results are never dropped


def test_med_rule_rejects_another_drugs_dates(patients):
    p = next(x for x in patients if x["chart"]["policy_id"] == "POL-RA"
             and any("hydroxychloroquine" in m["name"].lower() for m in x["chart"]["medications"]))
    env = env_for(p)
    doc = next(d for d in env.docs if "hydroxychloroquine" in d["text"].lower())
    import re
    dates = re.findall(r"\d{4}-\d{2}-\d{2}", doc["text"])
    res = env.evaluate_rule("C3", {"drug": "methotrexate", "start": dates[0], "end": None}, [doc["id"]])
    assert "error" in res and "drug methotrexate" in res["error"]


def test_tool_args_tolerate_null_or_string_evidence_ids(patients):
    env = env_for(patients[0])
    age = env.chart["demographics"]["age"]
    assert env.call("evaluate_rule", {"criterion_id": "C2", "values": {"age": age}, "evidence_ids": "DEMO"})["status"] == "met"
    assert "recorded" in env.call("record_decision", {"criterion_id": "C2", "status": "met", "evidence_ids": None})


def test_stuck_agent_is_sent_to_human_review(patients):
    class Stuck(MockLLM):
        def chat(self, messages, purpose="chat", phi=None):
            r = super().chat(messages, purpose, phi)
            if purpose == "agent" and len(messages) > 4:
                r.text = '{"tool": "evaluate_rule", "args": {"criterion_id": "C2", "values": {"age": 999}, "evidence_ids": ["DEMO"]}}'
            return r

    p = patients[0]
    llm = Stuck()
    res = run_agent(p["chart"], load_policy(p["chart"]["policy_id"]), llm, LLMJudge(llm), retrieval_mode="bm25")
    assert res["stop_reason"] == "stuck" and res["decision"] == "review"
    assert res["steps"] < 10
    obs = res["trace"][-1]["observation"]
    assert "cited_text" in obs and "warning" in obs


def test_disconnect_mid_run_never_loses_earlier_results(patients, results_dir):
    import pytest
    from authagent.common import read_jsonl

    recs = patients[:4]
    run_config("single_prompt", MockLLM(), recs, retrieval_mode="bm25", resume=False, verbose=False)

    class Disconnect(MockLLM):
        calls = 0

        def chat(self, messages, purpose="chat", phi=None):
            Disconnect.calls += 1
            if Disconnect.calls == 2:
                raise KeyboardInterrupt  # like a killed Colab runtime: not caught by the runner
            return super().chat(messages, purpose, phi)

    with pytest.raises(KeyboardInterrupt):
        run_config("single_prompt", Disconnect(), recs, retrieval_mode="bm25", resume=False, verbose=False)
    assert len(read_jsonl(results_dir / "runs" / "single_prompt.jsonl")) == 4


def test_missing_cannot_cite_unchecked_or_refuting_evidence(patients):
    env = env_for(patients[0])
    env.search_chart("methotrexate response", criterion_id="C4")
    env.search_chart("intolerance of methotrexate", criterion_id="C4")
    note = next(d["id"] for d in env.docs if d["id"].startswith("N"))
    res = env.record_decision("C4", "missing", [note])
    assert "have not checked" in res["error"]
    env.grounding["C4"].append({"status": "unmet", "evidence_ids": [note], "via": "test"})
    env.checked.add(("C4", note))
    assert "Record that status" in env.record_decision("C4", "missing", [note])["error"]
    assert env.record_decision("C4", "missing", [])["status"] == "missing"  # no evidence cited: allowed


def test_prompt_has_no_copyable_example_values():
    from authagent.agent import system_prompt

    sp = system_prompt(True)
    assert '"age": 54' not in sp and "7.8" not in sp


def test_rule_criteria_cannot_be_judged_by_the_llm(patients):
    env = env_for(patients[0])
    hits = env.search_chart("methotrexate", criterion_id="C3")["hits"]
    note = next((h["id"] for h in hits if h["id"].startswith("N")), hits[0]["id"])
    res = env.verify_evidence("C3", note)  # C3 is med_duration: dates go to evaluate_rule
    assert "evaluate_rule" in res["error"] and "cited_text" in res


def test_verify_needs_evidence_found_for_that_criterion(patients):
    env = env_for(patients[0])
    note = next(h["id"] for h in env.search_chart("methotrexate", criterion_id="C3")["hits"] if h["id"].startswith("N"))
    assert "not returned by a search for C4" in env.verify_evidence("C4", note)["error"]
    env.search_chart("methotrexate", criterion_id="C4")
    if note in env.retrieved["C4"]:
        assert "error" not in env.verify_evidence("C4", note)


def test_search_budget_per_criterion(patients):
    env = env_for(patients[0])
    for i in range(6):
        out = env.search_chart(f"query {i}", criterion_id="C4")
        assert ("warning" in out) == (i >= 3)
    assert "Search limit" in env.search_chart("one more", criterion_id="C4")["error"]
    assert env.record_decision("C4", "missing", [])["status"] == "missing"
