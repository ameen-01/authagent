"""Baselines that isolate what the agent adds.

single_prompt   : the whole (redacted) chart and policy in one LLM call. No tools, no agency.
fixed_pipeline  : the same tools in a fixed order: one search per criterion using the criterion
                  text, one extraction or verification, no rephrasing, no second look.
"""

from __future__ import annotations

import json
import time

from .common import STATUSES, decide
from .llm import parse_json
from .rules import EXPECTED
from .tools import RULE_KINDS, CaseEnv

SINGLE_SYSTEM = """You are a prior-authorization reviewer. Today's date: {today}.
Given an insurer policy's criteria and a patient's chart, decide each criterion:
- met: the chart shows it is satisfied
- unmet: the chart shows it is not satisfied
- missing: no evidence, or the evidence is outdated or incomplete
Reply with JSON only:
{{"criteria": {{"C1": {{"status": "met"|"unmet"|"missing", "evidence_ids": ["..."]}}, ...}}}}"""


def render_chart(docs: list[dict]) -> str:
    lines = []
    for d in docs:
        where = f" ({d['note']})" if d.get("note") else ""
        lines.append(f"[{d['id']}]{where} {d['text']}")
    return "\n".join(lines)


def run_single_prompt(chart: dict, policy: dict, llm, **_) -> dict:
    env = CaseEnv(chart, policy, verifier=None, retrieval_mode="bm25")
    crit = [{"id": c["id"], "text": c["text"]} for c in policy["criteria"]]
    messages = [
        {"role": "system", "content": SINGLE_SYSTEM.format(today=env.chart["as_of"])},
        {"role": "user", "content": f"Policy: {policy['title']}\nCriteria: {json.dumps(crit)}\n\n"
                                    f"Chart:\n{render_chart(env.docs)}"},
    ]
    t0 = time.time()
    r = llm.chat(messages, purpose="single_prompt", phi=env.phi)
    try:
        parsed = parse_json(r.text).get("criteria", {})
    except ValueError:
        parsed = {}
    for c in policy["criteria"]:
        item = parsed.get(c["id"], {}) if isinstance(parsed, dict) else {}
        status = str(item.get("status", "missing")).lower()
        status = status if status in STATUSES else "missing"
        ids = [e for e in item.get("evidence_ids", []) if e in env.doc_by_id] if isinstance(item, dict) else []
        env.decisions[c["id"]] = {"status": status, "evidence_ids": ids, "reason": "single prompt"}
    statuses = {cid: d["status"] for cid, d in env.decisions.items()}
    return {"statuses": statuses, "decision": decide(statuses.values()), "decisions": env.decisions,
            "draft": env.draft(), "steps": 1, "parse_failures": 0, "stop_reason": "single_call",
            "verifier_calls": 0, "escalations": 0, "wall_s": round(time.time() - t0, 3), "trace": []}


EXTRACT_SYSTEM = """You extract values from chart text for a deterministic rule. Today's date: {today}.
Criterion: {criterion}
Return JSON only: {{"values": <object or null>, "evidence_ids": [ids you used]}}
values format for kind "{kind}": {fmt}
Use null if the chunks do not contain the needed values. Do not judge whether the criterion is met."""


def run_fixed_pipeline(chart: dict, policy: dict, llm, verifier, retrieval_mode: str = "hybrid", **_) -> dict:
    env = CaseEnv(chart, policy, verifier, retrieval_mode=retrieval_mode)
    t0 = time.time()
    trace = []
    for c in policy["criteria"]:
        cid, kind = c["id"], c["kind"]
        hits = env.search_chart(c["text"], criterion_id=cid)["hits"]
        status, evidence = "missing", []
        if kind in RULE_KINDS:
            messages = [
                {"role": "system", "content": EXTRACT_SYSTEM.format(today=env.chart["as_of"], criterion=json.dumps(c),
                                                                    kind=kind, fmt=EXPECTED[kind])},
                {"role": "user", "content": "Chunks: " + json.dumps(hits)},
            ]
            r = llm.chat(messages, purpose="extract", phi=env.phi)
            try:
                out = parse_json(r.text)
                values = out.get("values")
                ids = [e for e in out.get("evidence_ids", []) if e in env.doc_by_id]
            except ValueError:
                values, ids = None, []
            if values:
                res = env.evaluate_rule(cid, values, ids)
                trace.append({"criterion_id": cid, "tool": "evaluate_rule", "values": values, "observation": res})
                if res.get("status") in STATUSES:
                    status, evidence = res["status"], ids
        if status == "missing" and kind in ("diagnosis", "documentation"):
            note_hits = [h for h in hits if "." in h["id"] and h["id"].startswith("N")]
            if note_hits:
                res = env.verify_evidence(cid, note_hits[0]["id"])
                trace.append({"criterion_id": cid, "tool": "verify_evidence", "observation": res})
                if res.get("label") == "SUPPORTS":
                    status, evidence = "met", [note_hits[0]["id"]]
                elif res.get("label") == "REFUTES":
                    status, evidence = "unmet", [note_hits[0]["id"]]
        env.decisions[cid] = {"status": status, "evidence_ids": evidence, "reason": "fixed pipeline"}
    statuses = {cid: d["status"] for cid, d in env.decisions.items()}
    return {"statuses": statuses, "decision": decide(statuses.values()), "decisions": env.decisions,
            "draft": env.draft(), "steps": len(trace), "parse_failures": 0, "stop_reason": "pipeline",
            "verifier_calls": env.verifier_calls, "escalations": env.escalations,
            "wall_s": round(time.time() - t0, 3), "trace": trace}
