"""The agent loop: the LLM picks one tool per turn, code runs it, the result goes back.

Plain Python on purpose (no framework), so every step can be explained and traced.
"""

from __future__ import annotations

import json
import time

from .llm import parse_json
from .tools import CaseEnv

SYSTEM_PROMPT = """You are AuthAgent, a prior-authorization assistant. You decide, for each criterion of an
insurer policy, whether the patient's chart shows it is met, unmet, or missing.

Reply with exactly ONE JSON object per turn and nothing else:
{"thought": "<one short sentence>", "tool": "<tool name>", "args": {...}}

Tools:
- get_checklist {} -> the policy criteria (id, text, kind, params).
- search_chart {"query": str, "criterion_id": str} -> top chart chunks with ids such as DEMO, DX-1, MED-2,
  LAB-1, N2.s4. Search is keyword + semantic. If results are not relevant, rephrase: use drug or test
  names, abbreviations, synonyms, or how a clinician would write it.
{rule_tool}- verify_evidence {"criterion_id": str, "evidence_id": str} -> whether that chunk SUPPORTS, REFUTES or gives
  NOT_ENOUGH_INFO for the criterion. Use it for documentation criteria and for free-text statements.
- record_decision {"criterion_id": str, "status": "met"|"unmet"|"missing", "evidence_ids": [ids], "reason": str}
  -> must be backed by an earlier {backing} result with the same status. "missing" needs two different
  searches for that criterion first (or a rule result of "missing").
- finish {} -> when every criterion has a decision.

Rules:
1. First call get_checklist. Then handle the criteria one at a time, in order.
2. Cite only chunk ids returned by search_chart. Never invent evidence.
{rule_rule}4. Look-alike text does not satisfy a criterion: another drug, a family member, an order or plan, a
   different condition. When unsure, call verify_evidence.
5. If nothing relevant is found after rephrasing, record "missing". Do not guess.
6. As soon as a check settles a criterion, record it with record_decision before moving on.
   Observations may include "next": follow it unless you have a reason not to.
7. Keep each thought to one short sentence."""

RULE_TOOL = """- evaluate_rule {"criterion_id": str, "values": {...}, "evidence_ids": [ids]} -> deterministic check for
  criteria of kind age, diagnosis, lab, med_duration. values formats:
    age:          {"age": <age from DEMO>}
    diagnosis:    {"code": "<ICD-10 code from the problem list>"}
    lab:          {"name": "<test>", "measurements": [{"value": <value>, "date": "<YYYY-MM-DD>"}, ...]}
    med_duration: {"drug": "<drug>", "start": "<YYYY-MM-DD>", "end": "<YYYY-MM-DD>" or null if ongoing}
  Copy every value exactly from the chunks you cite; never use a value you have not read in the chart.
"""
RULE_RULE = """3. Never compare numbers or dates yourself: read the values from the chart (structured rows or note
   text) and pass them to evaluate_rule. For labs, pass every result of that test you found; the rule
   uses the most recent one and checks how old it is.
"""
NO_RULE_RULE = """3. Judge every criterion with verify_evidence on the most relevant chunk.
"""


def system_prompt(use_rules: bool) -> str:
    return (SYSTEM_PROMPT
            .replace("{rule_tool}", RULE_TOOL if use_rules else "")
            .replace("{rule_rule}", RULE_RULE if use_rules else NO_RULE_RULE)
            .replace("{backing}", "evaluate_rule or verify_evidence" if use_rules else "verify_evidence"))


def task_message(chart: dict, policy: dict) -> str:
    return (f"Policy {policy['policy_id']}: {policy['title']}.\n"
            f"Requested drug: {chart['requested_drug']}. Prescriber specialty: {chart['prescriber_specialty']}.\n"
            f"Today's date: {chart['as_of']}.\n"
            "Decide every criterion, then finish. Start by calling get_checklist.")


def compact(messages: list[dict], base_end: int, work_start: int | None) -> list[dict]:
    """Working-memory compaction: what the model sees on each step.

    Keeps the system prompt, the task, the checklist exchange and everything since the last
    recorded decision (whose observation lists every decision so far). Finished criteria
    are therefore summarised in one line instead of replaying their whole search history,
    which keeps each call small and the total tokens per case roughly flat.
    """
    if work_start is None or work_start <= base_end:
        return messages
    return messages[:base_end] + messages[work_start:]


def run_agent(chart: dict, policy: dict, llm, verifier, retrieval_mode: str = "hybrid",
              use_rules: bool = True, max_steps: int = 40, max_parse_failures: int = 4,
              compaction: bool = True, max_repeats: int = 5) -> dict:
    env = CaseEnv(chart, policy, verifier, retrieval_mode=retrieval_mode, use_rules=use_rules)
    messages = [{"role": "system", "content": system_prompt(use_rules)},
                {"role": "user", "content": task_message(env.chart, policy)}]
    trace: list[dict] = []
    parse_failures = 0
    stop_reason = "max_steps"
    base_end, work_start = None, None
    last_signature, repeats = None, 0
    t0 = time.time()
    for step in range(1, max_steps + 1):
        context = compact(messages, base_end, work_start) if (compaction and base_end) else messages
        resp = llm.chat(context, purpose="agent", phi=env.phi)
        messages.append({"role": "assistant", "content": resp.text})
        try:
            action = parse_json(resp.text)
            tool, args = action["tool"], action.get("args", {}) or {}
        except (ValueError, KeyError, TypeError) as exc:
            parse_failures += 1
            trace.append({"step": step, "error": f"unparseable reply: {exc}", "raw": resp.text[:300]})
            if parse_failures > max_parse_failures:
                stop_reason = "parse_failures"
                break
            messages.append({"role": "user", "content": "Your reply was not a valid JSON action. Reply with exactly "
                                                        'one JSON object: {"thought": ..., "tool": ..., "args": {...}}'})
            continue
        obs = env.call(tool, args)
        signature = json.dumps([tool, args], sort_keys=True)
        repeats = repeats + 1 if signature == last_signature and "error" in obs else 0
        last_signature = signature
        if repeats >= 2:  # third identical failing call in a row: break the loop
            obs = {**obs, "warning": "You have repeated this exact failing call. Do something different: "
                                     "read the error and the cited text, or call search_chart."}
        trace.append({"step": step, "thought": action.get("thought", ""), "tool": tool, "args": args,
                      "observation": obs})
        if repeats >= max_repeats:
            stop_reason = "stuck"
            break
        messages.append({"role": "user", "content": "Observation: " + json.dumps(obs, ensure_ascii=False)})
        if tool == "get_checklist" and "criteria" in obs and base_end is None:
            base_end = len(messages)
        if "recorded" in obs:
            work_start = len(messages) - 2  # the record_decision call and its observation
        if tool == "finish" and env.finished:
            stop_reason = "finished"
            break
    outcome = env.outcome()
    decision = outcome["decision"]
    draft = env.draft()
    if stop_reason != "finished":
        # Ran out of budget or got stuck: never guess; hand the case to a human reviewer.
        decision = "review"
        draft = "NEEDS HUMAN REVIEW (agent did not finish: " + stop_reason + ")\n\n" + draft
    return {
        "statuses": outcome["statuses"],
        "decision": decision,
        "decisions": env.decisions,
        "draft": draft,
        "steps": len(trace),
        "parse_failures": parse_failures,
        "stop_reason": stop_reason,
        "verifier_calls": env.verifier_calls,
        "escalations": env.escalations,
        "wall_s": round(time.time() - t0, 3),
        "trace": trace,
    }
