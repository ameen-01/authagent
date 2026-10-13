"""A scripted, offline stand-in for the LLM. FOR PLUMBING TESTS ONLY.

It reads the same prompts and observations a real model would and follows a fixed search /
extract / verify recipe, so the whole system (tools, guardrails, baselines, metrics) can be
exercised without a network or an API key. Its numbers say nothing about real model quality
and must never be reported as results.
"""

from __future__ import annotations

import json
import re

from .llm import LLMResponse, Meter
from .verifier import LexicalVerifier

DATE = r"\d{4}-\d{2}-\d{2}"


def _act(tool: str, args: dict | None = None, thought: str = "") -> str:
    return json.dumps({"thought": thought or f"Call {tool}.", "tool": tool, "args": args or {}})


def _accepted(name: str, accepted: list[str]) -> bool:
    name = name.lower()
    return any(a.lower() in name or name in a.lower() for a in accepted)


def extract_values(c: dict, hits: list[dict]) -> tuple[dict | None, list[str]]:
    """Pull rule inputs out of search hits with regexes (what a model would read off the text)."""
    kind, p = c["kind"], c.get("params", {})
    if kind == "age":
        for h in hits:
            m = re.search(r"age (\d+)", h["text"])
            if h["id"] == "DEMO" and m:
                return {"age": int(m.group(1))}, ["DEMO"]
    if kind == "diagnosis":
        for h in hits:
            m = re.search(r"ICD-10 ([A-Z]\d[\w.]*)\)", h["text"])
            if m and any(m.group(1).startswith(px) for px in p["code_prefixes"]):
                return {"code": m.group(1)}, [h["id"]]
    if kind == "lab":
        meas, ids = [], []
        for h in hits:
            m = re.search(r"Lab result: (.+?) = ([\w.]+)(?: \S+)? on (" + DATE + ")", h["text"])
            if m and _accepted(m.group(1), p["labs"]):
                meas.append({"value": m.group(2), "date": m.group(3)})
                ids.append(h["id"])
                continue
            if any(lab.lower() in h["text"].lower() for lab in p["labs"]):
                m = re.search(r"(" + DATE + r")\)?(?::| was)\s*([0-9.]+|negative|positive)", h["text"])
                if m:
                    meas.append({"value": m.group(2), "date": m.group(1)})
                    ids.append(h["id"])
        if meas:
            return {"name": p["labs"][0], "measurements": meas}, ids
    if kind == "med_duration":
        for h in hits:
            m = re.search(r"Medication: (\S+) .*?started (" + DATE + r"), (?:stopped (" + DATE + r")|ongoing)", h["text"])
            if m and _accepted(m.group(1), p["drugs"]):
                return {"drug": m.group(1), "start": m.group(2), "end": m.group(3)}, [h["id"]]
            if any(d in h["text"].lower() for d in p["drugs"]):
                dates = re.findall(DATE, h["text"])
                drug = next(d for d in p["drugs"] if d in h["text"].lower())
                if dates:
                    return {"drug": drug, "start": dates[0], "end": dates[1] if len(dates) > 1 else None}, [h["id"]]
    return None, []


def queries_for(c: dict) -> list[str]:
    kind, p = c["kind"], c.get("params", {})
    if kind == "diagnosis":
        return [p["condition"], "assessment diagnosis " + p["condition"], "ICD-10 " + " ".join(p["code_prefixes"])]
    if kind == "age":
        return ["demographics age", "age years"]
    if kind == "lab":
        return [" ".join(p["labs"]), p["labs"][0] + " result"]
    if kind == "med_duration":
        return [" ".join(p["drugs"]), p["drugs"][0] + " started stopped"]
    words = [w for w in re.findall(r"[A-Za-z]{5,}", c["text"]) if w.lower() not in {"documented", "patient"}]
    return [c["text"], " ".join(words[:4])]


class MockLLM:
    name = "mock"

    def __init__(self):
        self.meter = Meter()
        self.lex = LexicalVerifier()

    def chat(self, messages: list[dict], purpose: str = "chat", phi: dict | None = None) -> LLMResponse:
        if phi:  # same guard as the real client
            from .redact import find_leaks
            from .llm import PHILeakError

            leaks = find_leaks(json.dumps(messages), phi)
            if leaks:
                raise PHILeakError(f"Refusing to send identifiers to the LLM: {leaks}")
        fn = {"agent": self._agent, "judge": self._judge, "extract": self._extract,
              "single_prompt": self._single}.get(purpose)
        text = fn(messages) if fn else "{}"
        r = LLMResponse(text, False, prompt_tokens=sum(len(m["content"]) for m in messages) // 4,
                        completion_tokens=len(text) // 4, latency_s=0.0)
        self.meter.add(purpose, r)
        return r

    # ---------------------------------------------------------------- purposes
    def _judge(self, messages):
        u = messages[-1]["content"]
        crit = re.search(r"Criterion: (.*)", u).group(1)
        ev = re.search(r"Chart text: (.*)", u, re.DOTALL).group(1)
        return json.dumps({"label": self.lex.predict(crit, ev)["label"], "rationale": "lexical mock"})

    def _extract(self, messages):
        crit = json.loads(re.search(r"Criterion: (\{.*\})", messages[0]["content"]).group(1))
        hits = json.loads(messages[1]["content"][len("Chunks: "):])
        values, ids = extract_values(crit, hits)
        return json.dumps({"values": values, "evidence_ids": ids})

    def _single(self, messages):
        # Deliberately naive: anything with a keyword hit counts as met.
        u = messages[-1]["content"]
        crit = json.loads(re.search(r"Criteria: (\[.*?\])\n", u).group(1))
        chart = u.split("Chart:\n", 1)[1].lower()
        out = {}
        for c in crit:
            words = [w for w in re.findall(r"[a-z]{6,}", c["text"].lower())]
            out[c["id"]] = {"status": "met" if any(w in chart for w in words) else "missing", "evidence_ids": []}
        return json.dumps({"criteria": out})

    def _agent(self, messages):
        pairs = []
        for i in range(2, len(messages) - 1, 2):
            try:
                act = json.loads(messages[i]["content"])
            except ValueError:
                act = None
            txt = messages[i + 1]["content"]
            obs = json.loads(txt[len("Observation: "):]) if txt.startswith("Observation: ") else None
            pairs.append((act, obs))
        if not pairs:
            return _act("get_checklist")
        checklist = next((o for a, o in pairs if a and a["tool"] == "get_checklist"), None)
        if checklist is None:
            return _act("get_checklist")
        recorded = set()
        for a, o in pairs:  # with compaction, the latest decision lists all earlier ones
            if o and "recorded" in o:
                recorded |= set(o.get("decided_so_far", {})) | {o["recorded"]}
        pending = [c for c in checklist["criteria"] if c["id"] not in recorded]
        if not pending:
            return _act("finish")
        c = pending[0]
        cid = c["id"]
        hist = [(a, o) for a, o in pairs if a and a.get("args", {}).get("criterion_id") == cid]
        use_rules = "evaluate_rule" in messages[0]["content"]

        if hist:
            a, o = hist[-1]
            if a["tool"] == "evaluate_rule" and o.get("status") in ("met", "unmet", "missing"):
                return _act("record_decision", {"criterion_id": cid, "status": o["status"],
                                                "evidence_ids": a["args"].get("evidence_ids", []), "reason": o["reason"]})
            if a["tool"] == "verify_evidence" and o.get("label") in ("SUPPORTS", "REFUTES"):
                return _act("record_decision", {"criterion_id": cid,
                                                "status": "met" if o["label"] == "SUPPORTS" else "unmet",
                                                "evidence_ids": [a["args"]["evidence_id"]], "reason": "verified"})

        hits = [h for a, o in hist if a["tool"] == "search_chart" for h in o.get("hits", [])]
        tried_rules = [json.dumps(a["args"].get("values"), sort_keys=True) for a, _ in hist if a["tool"] == "evaluate_rule"]
        verified = {a["args"]["evidence_id"] for a, _ in hist if a["tool"] == "verify_evidence"}

        if use_rules and c["kind"] in ("age", "diagnosis", "lab", "med_duration"):
            values, ids = extract_values(c, hits)
            if values and json.dumps(values, sort_keys=True) not in tried_rules:
                return _act("evaluate_rule", {"criterion_id": cid, "values": values, "evidence_ids": ids})
        if (c["kind"] in ("diagnosis", "documentation") or not use_rules) and len(verified) < 3:
            for h in hits:
                if h["id"] not in verified and (h["id"].startswith("N") or not use_rules):
                    return _act("verify_evidence", {"criterion_id": cid, "evidence_id": h["id"]})

        queries = queries_for(c)
        done = [a["args"]["query"] for a, _ in hist if a["tool"] == "search_chart"]
        nxt = [q for q in queries if q not in done]
        if nxt:
            return _act("search_chart", {"query": nxt[0], "criterion_id": cid})
        return _act("record_decision", {"criterion_id": cid, "status": "missing", "evidence_ids": [],
                                        "reason": "not found after rephrasing"})
