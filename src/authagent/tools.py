"""The agent's tools for one case, with grounding guardrails.

Guardrails enforced in code (not just asked for in the prompt):
* Evidence can only be cited by chunk id; ids that do not exist are rejected.
* verify_evidence fetches the chunk text itself, so the LLM cannot paraphrase or invent evidence.
* record_decision("met"/"unmet") must match an earlier evaluate_rule or verify_evidence result
  for the same criterion and evidence.
* record_decision("missing") needs either a rule result of "missing" (e.g. stale lab) or at
  least two different searches for that criterion, so the agent cannot give up after one try.
"""

from __future__ import annotations

from .common import decide
from .index import Retriever, chart_to_docs
from .redact import redact_chart
from .rules import evaluate

RULE_KINDS = {"age", "diagnosis", "lab", "med_duration"}
LABEL_TO_STATUS = {"SUPPORTS": "met", "REFUTES": "unmet"}


class CaseEnv:
    def __init__(self, chart: dict, policy: dict, verifier, retrieval_mode: str = "hybrid",
                 use_rules: bool = True, k: int = 5):
        self.phi = chart["phi"]
        self.chart = redact_chart(chart)
        self.policy = policy
        self.criteria = {c["id"]: c for c in policy["criteria"]}
        self.docs = chart_to_docs(self.chart)
        self.doc_by_id = {d["id"]: d for d in self.docs}
        self.retriever = Retriever(self.docs, mode=retrieval_mode)
        self.verifier = verifier
        self.use_rules = use_rules
        self.k = k
        self.decisions: dict[str, dict] = {}
        self.grounding: dict[str, list[dict]] = {cid: [] for cid in self.criteria}
        self.searches: dict[str, set[str]] = {cid: set() for cid in self.criteria}
        self.verifier_calls = 0
        self.escalations = 0
        self.finished = False

    # ------------------------------------------------------------------ dispatch
    def tool_names(self) -> list[str]:
        names = ["get_checklist", "search_chart", "verify_evidence", "record_decision", "finish"]
        if self.use_rules:
            names.insert(2, "evaluate_rule")
        return names

    def call(self, tool: str, args: dict) -> dict:
        fn = {
            "get_checklist": self.get_checklist,
            "search_chart": self.search_chart,
            "evaluate_rule": self.evaluate_rule if self.use_rules else None,
            "verify_evidence": self.verify_evidence,
            "record_decision": self.record_decision,
            "finish": self.finish,
        }.get(tool)
        if fn is None:
            return {"error": f"Unknown tool '{tool}'. Available: {self.tool_names()}"}
        if not isinstance(args, dict):
            return {"error": "args must be a JSON object."}
        try:
            return fn(**args)
        except TypeError as exc:
            return {"error": f"Bad arguments for {tool}: {exc}"}

    # ------------------------------------------------------------------ tools
    def get_checklist(self) -> dict:
        crit = []
        for c in self.policy["criteria"]:
            item = {"id": c["id"], "text": c["text"], "kind": c["kind"]}
            if c["params"]:
                item["params"] = c["params"]
            crit.append(item)
        return {"policy_id": self.policy["policy_id"], "title": self.policy["title"], "criteria": crit}

    def search_chart(self, query: str, criterion_id: str | None = None, k: int | None = None) -> dict:
        if criterion_id in self.searches:
            self.searches[criterion_id].add(" ".join(query.lower().split()))
        hits = self.retriever.search(query, k=min(int(k or self.k), 8))
        return {"query": query, "hits": hits or [], "note": None if hits else "No matching chart text."}

    def evaluate_rule(self, criterion_id: str, values: dict, evidence_ids: list[str] | None = None) -> dict:
        crit = self.criteria.get(criterion_id)
        if crit is None:
            return {"error": f"Unknown criterion '{criterion_id}'."}
        evidence_ids = list(evidence_ids or [])
        bad = [e for e in evidence_ids if e not in self.doc_by_id]
        if bad:
            return {"error": f"Unknown evidence ids {bad}. Cite ids returned by search_chart."}
        values = values or {}
        # Models sometimes nest the values one level deep ({"lab": {...}}, {"values": {...}}): unwrap it.
        if len(values) == 1:
            (only,) = values.values()
            if isinstance(only, dict):
                values = only
        if crit["kind"] in RULE_KINDS:
            if not evidence_ids:
                return {"error": "Pass evidence_ids: the chunk ids (from search_chart) you read these values from."}
            unsupported = ungrounded_values(crit["kind"], values, [self.doc_by_id[e]["text"] for e in evidence_ids])
            if unsupported:
                return {"error": f"These values do not appear in the cited evidence {evidence_ids}: {unsupported}. "
                                 "Copy values exactly from the chart chunks, or search for the right chunk first."}
        res = evaluate(crit, values)
        if res["status"] in ("met", "unmet", "missing"):
            self.grounding[criterion_id].append({"status": res["status"], "evidence_ids": evidence_ids, "via": "rule"})
        out = {"criterion_id": criterion_id, **res}
        if res["status"] in ("met", "unmet", "missing") and criterion_id not in self.decisions:
            out["next"] = (f"record_decision for {criterion_id}: status {res['status']}, "
                           f"evidence_ids {evidence_ids}")
        return out

    def verify_evidence(self, criterion_id: str, evidence_id: str) -> dict:
        crit = self.criteria.get(criterion_id)
        if crit is None:
            return {"error": f"Unknown criterion '{criterion_id}'."}
        doc = self.doc_by_id.get(evidence_id)
        if doc is None:
            return {"error": f"Unknown evidence id '{evidence_id}'. Cite ids returned by search_chart."}
        if self.use_rules and crit["kind"] in ("age", "lab", "med_duration") and doc["source"] == "structured":
            return {"error": "Structured values are checked with evaluate_rule, not verify_evidence."}
        res = self.verifier.predict(crit["text"], doc["text"], phi=self.phi) if _takes_phi(self.verifier) \
            else self.verifier.predict(crit["text"], doc["text"])
        self.verifier_calls += 1
        self.escalations += int(bool(res.get("escalated")) or res.get("source") == "llm")
        status = LABEL_TO_STATUS.get(res["label"])
        if status:
            self.grounding[criterion_id].append({"status": status, "evidence_ids": [evidence_id], "via": res["source"]})
        out = {"criterion_id": criterion_id, "evidence_id": evidence_id, "label": res["label"],
               "confidence": res.get("confidence"), "source": res["source"]}
        if status and criterion_id not in self.decisions:
            out["next"] = f"record_decision for {criterion_id}: status {status}, evidence_ids ['{evidence_id}']"
        return out

    def record_decision(self, criterion_id: str, status: str, evidence_ids: list[str] | None = None,
                        reason: str = "") -> dict:
        if criterion_id not in self.criteria:
            return {"error": f"Unknown criterion '{criterion_id}'."}
        if criterion_id in self.decisions:
            return {"error": f"{criterion_id} is already recorded as {self.decisions[criterion_id]['status']}. "
                             f"Continue with: {self._remaining()}."}
        status = str(status).lower().strip()
        if status not in ("met", "unmet", "missing"):
            return {"error": "status must be met, unmet or missing."}
        evidence_ids = list(evidence_ids or [])
        bad = [e for e in evidence_ids if e not in self.doc_by_id]
        if bad:
            return {"error": f"Unknown evidence ids {bad}."}
        g = self.grounding[criterion_id]
        if status in ("met", "unmet"):
            ok = [x for x in g if x["status"] == status and (not x["evidence_ids"] or not evidence_ids
                                                             or set(x["evidence_ids"]) & set(evidence_ids))]
            if not ok:
                return {"error": f"Decision '{status}' for {criterion_id} is not grounded. First get a matching "
                                 "result from evaluate_rule or verify_evidence, then cite that evidence."}
            if not evidence_ids:
                evidence_ids = ok[-1]["evidence_ids"]
        else:
            rule_missing = any(x["status"] == "missing" for x in g)
            if not rule_missing and len(self.searches[criterion_id]) < 2:
                return {"error": f"Before concluding 'missing' for {criterion_id}, search at least twice with "
                                 "different wording (pass criterion_id to search_chart)."}
        self.decisions[criterion_id] = {"status": status, "evidence_ids": evidence_ids, "reason": reason}
        remaining = [c for c in self.criteria if c not in self.decisions]
        decided = {cid: f"{d['status']} ({', '.join(d['evidence_ids']) or 'no evidence'})"
                   for cid, d in self.decisions.items()}
        out = {"recorded": criterion_id, "status": status, "decided_so_far": decided, "remaining": remaining}
        unrecorded = self._grounded_unrecorded()
        if unrecorded:  # survives context compaction: results checked but not yet recorded
            out["checked_not_recorded"] = unrecorded
        out["next"] = (f"record_decision for {list(unrecorded)[0]}" if unrecorded
                       else f"work on {remaining[0]}" if remaining else "finish")
        return out

    def _remaining(self) -> list[str]:
        return [c for c in self.criteria if c not in self.decisions]

    def _grounded_unrecorded(self) -> dict:
        out = {}
        for cid, g in self.grounding.items():
            if cid not in self.decisions and g:
                last = g[-1]
                out[cid] = f"{last['status']} ({', '.join(last['evidence_ids']) or 'no evidence id'})"
        return out

    def finish(self) -> dict:
        remaining = [c for c in self.criteria if c not in self.decisions]
        if remaining:
            return {"error": f"Criteria without a decision: {remaining}. Decide them before finishing.",
                    "checked_not_recorded": self._grounded_unrecorded()}
        self.finished = True
        return {"done": True, **self.outcome()}

    # ------------------------------------------------------------------ outputs
    def outcome(self) -> dict:
        statuses = {cid: self.decisions.get(cid, {"status": "missing"})["status"] for cid in self.criteria}
        return {"statuses": statuses, "decision": decide(statuses.values())}

    def draft(self) -> str:
        return draft_output(self.policy, self.chart, self.decisions, self.doc_by_id)


def ungrounded_values(kind: str, values: dict, texts: list[str]) -> list[str]:
    """Values the model passed to a rule that cannot be found in the evidence it cited.

    Stops the model from evaluating numbers it invented or copied from the prompt's examples.
    """
    import re

    blob = " ".join(texts).lower()
    numbers = {float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", blob)}

    def has_number(v) -> bool:
        try:
            return float(v) in numbers
        except (TypeError, ValueError):
            return False

    def has_text(v) -> bool:
        return v is not None and str(v).strip().lower() in blob

    missing = []
    try:
        if kind == "age" and "age" in values and not has_number(values["age"]):
            missing.append(f"age {values['age']}")
        elif kind == "diagnosis" and "code" in values and not has_text(values["code"]):
            missing.append(f"code {values['code']}")
        elif kind == "lab":
            ms = values.get("measurements") or ([{"value": values.get("value"), "date": values.get("date")}]
                                                if "value" in values else [])
            for m in ms:
                v, d = m.get("value"), m.get("date")
                if not (has_number(v) or has_text(v)):
                    missing.append(f"value {v}")
                if d and not has_text(str(d)[:10]):
                    missing.append(f"date {d}")
        elif kind == "med_duration":
            for key in ("start", "end"):
                d = values.get(key)
                if d and not has_text(str(d)[:10]):
                    missing.append(f"{key} {d}")
    except AttributeError:
        return []  # malformed values: the rule engine reports the error
    return missing


def _takes_phi(verifier) -> bool:
    import inspect

    try:
        return "phi" in inspect.signature(verifier.predict).parameters
    except (TypeError, ValueError):
        return False


def draft_output(policy: dict, chart: dict, decisions: dict, doc_by_id: dict) -> str:
    """Deterministic output document: request draft, missing-info request, or likely-denial report."""
    statuses = {c["id"]: decisions.get(c["id"], {"status": "missing"})["status"] for c in policy["criteria"]}
    decision = decide(statuses.values())
    head = {"approve": "PRIOR AUTHORIZATION REQUEST (draft)",
            "pend": "REQUEST FOR MISSING INFORMATION (do not submit yet)",
            "deny": "LIKELY-DENIAL REPORT (criteria not met; do not submit)"}[decision]
    lines = [head, f"Policy: {policy['title']} [{policy['policy_id']}]",
             f"Requested: {chart['requested_drug']} | Prescriber: {chart['prescriber_specialty']} | Patient: [PATIENT]",
             ""]
    for c in policy["criteria"]:
        d = decisions.get(c["id"], {"status": "missing", "evidence_ids": [], "reason": "not assessed"})
        mark = {"met": "MET", "unmet": "NOT MET", "missing": "MISSING"}[d["status"]]
        lines.append(f"- [{mark}] {c['id']}: {c['text']}")
        for eid in d.get("evidence_ids", []):
            if eid in doc_by_id:
                lines.append(f"    evidence {eid}: {doc_by_id[eid]['text']}")
        if d.get("reason"):
            lines.append(f"    reason: {d['reason']}")
    if decision == "pend":
        lines += ["", "Please provide: " + "; ".join(c["text"] for c in policy["criteria"] if statuses[c["id"]] == "missing")]
    return "\n".join(lines)
