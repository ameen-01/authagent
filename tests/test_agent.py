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
